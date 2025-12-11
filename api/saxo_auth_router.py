"""
SaxoBank OAuth2 Authentication Router

Endpoints:
- GET /api/saxo/auth/login - Generate OAuth URL
- GET /api/saxo/callback - Handle OAuth redirect
- GET /api/saxo/auth/status - Connection status
- POST /api/saxo/auth/refresh - Manual token refresh
- POST /api/saxo/auth/disconnect - Logout

Multi-tenant aware: Uses X-User header or get_active_user dependency
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import RedirectResponse

from api.deps import get_active_user
from api.utils import success_response, error_response
from api.services.user_fs import UserScopedFS
from connectors.saxo_api import SaxoOAuthClient, generate_state
from services.saxo_auth_service import SaxoAuthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/saxo", tags=["SaxoBank OAuth"])

# In-memory PKCE state storage (fallback if Redis unavailable)
# Key: state → {"code_verifier": "...", "user_id": "demo", "expires_at": ...}
_pkce_store: Dict[str, Dict[str, Any]] = {}


def store_pkce_verifier(state: str, code_verifier: str, user_id: str) -> None:
    """
    Store PKCE code_verifier temporarily (TTL: 10 minutes).

    Production: Should use Redis with TTL
    Development: In-memory dict is acceptable

    Args:
        state: Random state string
        code_verifier: PKCE verifier to retrieve later
        user_id: User who initiated OAuth flow
    """
    from datetime import datetime, timedelta

    _pkce_store[state] = {
        "code_verifier": code_verifier,
        "user_id": user_id,
        "expires_at": datetime.now() + timedelta(minutes=10)
    }

    logger.debug(f"Stored PKCE verifier for state '{state[:8]}...'")

    # Cleanup expired entries
    now = datetime.now()
    expired_keys = [k for k, v in _pkce_store.items() if v["expires_at"] < now]
    for key in expired_keys:
        del _pkce_store[key]


def retrieve_pkce_verifier(state: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve and delete PKCE code_verifier.

    Args:
        state: State string from callback

    Returns:
        {"code_verifier": "...", "user_id": "demo"} or None
    """
    from datetime import datetime

    data = _pkce_store.pop(state, None)

    if not data:
        logger.warning(f"PKCE verifier not found for state '{state[:8]}...'")
        return None

    # Check expiration
    if data["expires_at"] < datetime.now():
        logger.warning(f"PKCE verifier expired for state '{state[:8]}...'")
        return None

    logger.debug(f"Retrieved PKCE verifier for state '{state[:8]}...'")
    return data


@router.get("/auth/login")
async def saxo_login(
    user: str = Depends(get_active_user)
):
    """
    Generate OAuth2 authorization URL for user to login at Saxo.

    Flow:
        1. Generate PKCE pair (verifier + challenge)
        2. Store verifier temporarily (10min TTL)
        3. Generate authorization URL
        4. Return URL to frontend (user navigates to it)

    Returns:
        {
            "ok": true,
            "data": {
                "authorization_url": "https://sim.logonvalidation.net/authorize?...",
                "state": "random_string"
            }
        }

    Frontend Action:
        Open authorization_url in popup window
    """
    try:
        oauth_client = SaxoOAuthClient(user_id=user)

        # Generate PKCE
        pkce_pair = oauth_client.generate_pkce_pair()
        code_verifier = pkce_pair["code_verifier"]
        code_challenge = pkce_pair["code_challenge"]

        # Generate state (CSRF protection)
        state = generate_state()

        # Store verifier for callback
        store_pkce_verifier(state, code_verifier, user)

        # Generate authorization URL
        authorization_url = oauth_client.get_authorization_url(state, code_challenge)

        logger.info(f"🔐 OAuth login initiated for user '{user}'")

        return success_response({
            "authorization_url": authorization_url,
            "state": state,
            "environment": oauth_client.environment
        })

    except Exception as e:
        logger.error(f"Error generating login URL: {e}")
        return error_response(f"Failed to generate login URL: {str(e)}", code=500)


@router.get("/callback")
async def saxo_callback(
    request: Request,
    code: str = Query(..., description="Authorization code from Saxo"),
    state: str = Query(..., description="State for CSRF protection")
):
    """
    Handle OAuth2 callback from Saxo after user authentication.

    Query Params:
        code: Authorization code
        state: State string (matches /login)

    Flow:
        1. Retrieve PKCE verifier from storage
        2. Exchange code for tokens
        3. Save tokens to user storage
        4. Redirect to dashboard with success message

    Redirects:
        Success: /settings.html?status=connected
        Error: /settings.html?status=error&message=...
    """
    # Build absolute URL for redirect (fixes localhost issue in production)
    # Use the Host header that the client used (e.g., 192.168.1.200:8080)
    host = request.headers.get("host", "localhost:8080")
    scheme = request.url.scheme  # http or https
    base_url = f"{scheme}://{host}"

    try:
        # Retrieve PKCE verifier
        pkce_data = retrieve_pkce_verifier(state)

        if not pkce_data:
            logger.error(f"Invalid or expired state: {state[:8]}...")
            return RedirectResponse(
                url=f"{base_url}/static/settings.html?status=error&message=Invalid+or+expired+state"
            )

        code_verifier = pkce_data["code_verifier"]
        user_id = pkce_data["user_id"]

        oauth_client = SaxoOAuthClient(user_id=user_id)
        tokens = await oauth_client.exchange_code_for_tokens(code, code_verifier)

        # Save tokens
        auth_service = SaxoAuthService(user_id)
        await auth_service.save_tokens(tokens)

        logger.info(f"✅ OAuth callback successful for user '{user_id}'")

        # Redirect to settings page with success (absolute URL)
        return RedirectResponse(url=f"{base_url}/static/settings.html?status=connected")

    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        error_msg = str(e).replace(" ", "+")
        return RedirectResponse(
            url=f"{base_url}/static/settings.html?status=error&message={error_msg}"
        )


@router.get("/auth/status")
async def saxo_status(
    user: str = Depends(get_active_user)
):
    """
    Get current connection status for user.

    Returns:
        {
            "ok": true,
            "data": {
                "connected": true,
                "environment": "sim",
                "expires_at": "2025-11-28T15:35:00",
                "last_update": "2025-11-28T15:15:00",
                "account_key": "Cf4x..."
            }
        }

    Use Cases:
        - Display connection status in UI
        - Check if user needs to reconnect
        - Show token expiration time
    """
    try:
        auth_service = SaxoAuthService(user)
        status = auth_service.get_connection_status()

        return success_response(status)

    except Exception as e:
        logger.error(f"Error getting Saxo status: {e}")
        return error_response(f"Failed to get status: {str(e)}", code=500)


@router.post("/auth/refresh")
async def saxo_refresh(
    user: str = Depends(get_active_user)
):
    """
    Manually refresh access token.

    Use Cases:
        - User clicks "Refresh" button
        - Proactive refresh before expiration

    Returns:
        {
            "ok": true,
            "data": {
                "refreshed": true,
                "expires_at": "2025-11-28T15:55:00"
            }
        }

    Errors:
        - 401: Refresh token expired (user must reconnect)
        - 500: Other error
    """
    try:
        auth_service = SaxoAuthService(user)

        # Get valid token (triggers refresh if needed)
        access_token = await auth_service.get_valid_access_token()

        if not access_token:
            return error_response(
                "Refresh token expired - please reconnect",
                code=401
            )

        # Get new expiration
        status = auth_service.get_connection_status()

        logger.info(f"✅ Token refreshed for user '{user}'")

        return success_response({
            "refreshed": True,
            "expires_at": status.get("expires_at")
        })

    except Exception as e:
        logger.error(f"Error refreshing token: {e}")

        if "401" in str(e) or "expired" in str(e).lower():
            return error_response(
                "Refresh token expired - please reconnect",
                code=401
            )

        return error_response(f"Failed to refresh token: {str(e)}", code=500)


@router.post("/auth/disconnect")
async def saxo_disconnect(
    user: str = Depends(get_active_user)
):
    """
    Disconnect user from Saxo (logout).

    Use Cases:
        - User clicks "Disconnect" button
        - Security incident
        - Manual token revocation

    Returns:
        {
            "ok": true,
            "data": {
                "disconnected": true
            }
        }
    """
    try:
        auth_service = SaxoAuthService(user)
        await auth_service.disconnect()

        logger.info(f"✅ User '{user}' disconnected from Saxo")

        return success_response({"disconnected": True})

    except Exception as e:
        logger.error(f"Error disconnecting: {e}")
        return error_response(f"Failed to disconnect: {str(e)}", code=500)


@router.get("/resolve-instruments")
async def resolve_instruments(
    uics: str = Query(..., description="Comma-separated UICs (e.g., '34909,12345')"),
    asset_types: Optional[str] = Query(None, description="Comma-separated asset types (e.g., 'Etf,Stock')"),
    user: str = Depends(get_active_user)
):
    """
    Resolve UICs to instrument metadata (Symbol, Name, ISIN).

    Query Parameters:
        uics: Comma-separated UICs (e.g., "34909,12345,67890")
        asset_types: Optional comma-separated asset types (e.g., "Etf,Stock,Stock")
                     If not provided, defaults to "Stock" for all

    Returns:
        {
            "ok": true,
            "data": {
                "34909": {"symbol": "AAPL", "name": "Apple Inc.", "isin": "US0378331005", "currency": "USD"},
                "12345": {"symbol": "MSFT", "name": "Microsoft Corp.", "isin": "...", "currency": "USD"},
                ...
            },
            "meta": {
                "requested": 3,
                "resolved": 2,
                "failed": 1
            }
        }

    Use Cases:
        - Debug: Test UIC resolution manually
        - Frontend: Resolve UICs on-demand
        - Bulk: Resolve multiple UICs at once
    """
    try:
        auth_service = SaxoAuthService(user)

        # Check connection
        if not auth_service.is_connected():
            return error_response(
                "Not connected to Saxo - please connect first",
                code=401,
                details={"action": "reconnect"}
            )

        # Get access token
        access_token = await auth_service.get_valid_access_token()
        if not access_token:
            return error_response(
                "Access token expired - please reconnect",
                code=401,
                details={"action": "reconnect"}
            )

        # Parse UICs
        uic_list_raw = [int(u.strip()) for u in uics.split(",") if u.strip()]

        # Parse asset types (default to "Stock")
        if asset_types:
            asset_types_list = [t.strip() for t in asset_types.split(",")]
            # Extend with "Stock" if fewer types than UICs
            while len(asset_types_list) < len(uic_list_raw):
                asset_types_list.append("Stock")
        else:
            asset_types_list = ["Stock"] * len(uic_list_raw)

        # Build UIC list with types
        uic_list = list(zip(uic_list_raw, asset_types_list))

        # Resolve UICs
        from services.saxo_uic_resolver import SaxoUICResolver
        resolver = SaxoUICResolver(user_id=user)

        resolved_data = {}
        failed_uics = []

        for uic, asset_type in uic_list:
            try:
                metadata = await resolver.resolve_uic(
                    access_token=access_token,
                    uic=uic,
                    asset_type=asset_type
                )

                if metadata:
                    resolved_data[str(uic)] = metadata
                else:
                    failed_uics.append(uic)

            except Exception as e:
                logger.warning(f"⚠️ Failed to resolve UIC {uic}: {e}")
                failed_uics.append(uic)

        logger.info(f"✅ Resolved {len(resolved_data)}/{len(uic_list)} UICs for user '{user}'")

        return success_response(resolved_data, meta={
            "requested": len(uic_list),
            "resolved": len(resolved_data),
            "failed": len(failed_uics),
            "failed_uics": failed_uics
        })

    except ValueError as e:
        return error_response(f"Invalid UICs format: {str(e)}", code=400)
    except Exception as e:
        logger.error(f"Error resolving UICs: {e}")
        return error_response(f"Failed to resolve UICs: {str(e)}", code=500)


@router.get("/api-positions")
async def get_saxo_api_positions(
    user: str = Depends(get_active_user),
    use_cache: bool = Query(False, description="Force use cached data"),
    max_cache_age_hours: int = Query(24, description="Maximum cache age in hours")
):
    """
    Fetch current positions from Saxo OpenAPI (live/real-time data).

    Flow:
        1. Check if user connected (has valid tokens)
        2. Get valid access_token (auto-refresh if needed)
        3. Get account_key from user config or API
        4. Fetch positions from Saxo OpenAPI
        5. Normalize format (compatible with CSV structure)
        6. Cache results for offline fallback
        7. On API failure → return cached data with warning

    Query Parameters:
        use_cache: If true, skip API call and use cached data
        max_cache_age_hours: Maximum age of cache to accept (default: 24h)

    Returns:
        {
            "ok": true,
            "data": {
                "positions": [...],
                "total_value": 250000.0,
                "currency": "EUR",
                "source": "api" | "cache",
                "timestamp": "2025-11-28T15:30:00"
            },
            "meta": {
                "count": 15,
                "cache_age_hours": 2.5,
                "environment": "sim"
            }
        }

    Errors:
        - 401: Not connected or tokens expired
        - 404: No data available (no API access and no cache)
        - 500: API error and no cache fallback
    """
    try:
        auth_service = SaxoAuthService(user)

        # Check connection
        if not auth_service.is_connected():
            return error_response(
                "Not connected to Saxo - please connect first",
                code=401,
                details={"action": "reconnect"}
            )

        # Try cached data first if requested
        if use_cache:
            cached_data = await auth_service.get_cached_positions(max_age_hours=max_cache_age_hours)
            if cached_data:
                positions = cached_data.get("positions", [])
                logger.info(f"📦 Returning cached positions for user '{user}'")
                return success_response({
                    "positions": positions,
                    "cash_balance": cached_data.get("cash_balance", 0.0),
                    "total_value": cached_data.get("total_value", 0.0),
                    "currency": "USD",
                    "source": "cache",
                    "timestamp": datetime.now().isoformat()
                }, meta={
                    "count": len(positions),
                    "source": "cache"
                })

        # Get valid access token (auto-refresh)
        access_token = await auth_service.get_valid_access_token()

        if not access_token:
            return error_response(
                "Access token expired - please reconnect",
                code=401,
                details={"action": "reconnect"}
            )

        # Get account_key (from config or fetch from API)
        account_key = await _get_account_key(auth_service, access_token, user)

        if not account_key:
            return error_response(
                "Account key not found - please configure in settings",
                code=400,
                details={"action": "configure_account_key"}
            )

        # Fetch positions from API
        oauth_client = SaxoOAuthClient(user_id=user)

        try:
            # Fetch positions and balances in parallel
            import asyncio
            positions_task = oauth_client.get_positions(access_token, account_key)
            balances_task = oauth_client.get_balances(access_token, account_key)

            positions_raw, balances_data = await asyncio.gather(positions_task, balances_task)

            # DEBUG: Log first position structure
            if positions_raw:
                logger.info(f"📋 First position structure (sample): {positions_raw[0]}")

            # Resolve UICs to symbols (Live mode only)
            uic_metadata = await _resolve_uics_for_positions(positions_raw, access_token, user)

            # Normalize format (compatible with CSV structure) with enriched metadata
            positions_normalized = _normalize_positions(positions_raw, uic_metadata)

            # Extract cash balance
            # DEBUG: Log all balance fields to identify correct cash field
            logger.info(f"🔍 Saxo API balances_data keys: {list(balances_data.keys())}")
            logger.info(f"🔍 Saxo API full balances: {balances_data}")

            cash_balance = balances_data.get("CashBalance", 0.0)
            total_value_api = balances_data.get("TotalValue", 0.0)
            currency = balances_data.get("Currency", "EUR")

            # ✅ CRITICAL: Convert EUR → USD for frontend consistency
            # Frontend expects USD everywhere, Saxo returns EUR
            EUR_TO_USD_RATE = 1.16  # TODO: Use dynamic rate from FX service

            # Convert positions market_value to USD
            for pos in positions_normalized:
                if pos.get("market_value"):
                    pos["market_value"] = pos["market_value"] * EUR_TO_USD_RATE
                if pos.get("current_price"):
                    pos["current_price"] = pos["current_price"] * EUR_TO_USD_RATE
                if pos.get("avg_price"):
                    pos["avg_price"] = pos["avg_price"] * EUR_TO_USD_RATE
                if pos.get("pnl"):
                    pos["pnl"] = pos["pnl"] * EUR_TO_USD_RATE

            # ✅ CRITICAL: ALWAYS use Saxo API TotalValue (already includes positions + cash)
            # The API knows best - don't recalculate!
            total_value_eur = total_value_api
            cash_balance_eur = cash_balance
            total_value_usd = total_value_eur * EUR_TO_USD_RATE
            cash_balance_usd = cash_balance_eur * EUR_TO_USD_RATE

            # Log manual calculation for debug only (now in USD)
            positions_total_usd = sum(p.get("market_value", 0.0) for p in positions_normalized)
            total_value_calculated_usd = positions_total_usd + cash_balance_usd

            if abs(total_value_usd - total_value_calculated_usd) > 1.0:
                logger.warning(f"⚠️ Manual calculation mismatch: API=${total_value_usd:.2f} vs Calculated=${total_value_calculated_usd:.2f} USD")

            logger.info(f"✅ Saxo API: {len(positions_normalized)} positions, cash={cash_balance_eur:.2f} EUR (${cash_balance_usd:.2f} USD), total={total_value_eur:.2f} EUR (${total_value_usd:.2f} USD)")

            # Cache for offline fallback (including cash_balance and total_value)
            await auth_service.cache_positions(positions_normalized, cash_balance_usd, total_value_usd)

            return success_response({
                "positions": positions_normalized,
                "cash_balance": cash_balance_usd,  # USD for frontend
                "total_value": total_value_usd,    # USD for frontend
                "currency": "USD",  # Converted to USD
                "source": "api",
                "timestamp": datetime.now().isoformat()
            }, meta={
                "count": len(positions_normalized),
                "environment": oauth_client.environment,
                "original_currency": currency,  # Keep EUR for reference
                "eur_to_usd_rate": EUR_TO_USD_RATE
            })

        except Exception as api_error:
            # API call failed → try cache fallback
            logger.warning(f"⚠️ Saxo API call failed: {api_error}")

            cached_data = await auth_service.get_cached_positions(max_age_hours=max_cache_age_hours)

            if cached_data:
                positions = cached_data.get("positions", [])
                logger.info(f"📦 Returning cached positions (API failed) for user '{user}'")
                return success_response({
                    "positions": positions,
                    "cash_balance": cached_data.get("cash_balance", 0.0),
                    "total_value": cached_data.get("total_value", 0.0),
                    "currency": "USD",
                    "source": "cache_fallback",
                    "timestamp": datetime.now().isoformat(),
                    "warning": f"API unavailable - using cached data: {str(api_error)}"
                }, meta={
                    "count": len(positions),
                    "source": "cache_fallback"
                })

            # No cache available
            return error_response(
                f"API call failed and no cache available: {str(api_error)}",
                code=500,
                details={"api_error": str(api_error)}
            )

    except Exception as e:
        logger.error(f"Error fetching Saxo positions: {e}")
        return error_response(f"Failed to fetch positions: {str(e)}", code=500)


async def _resolve_uics_for_positions(
    positions_raw: List[Dict[str, Any]],
    access_token: str,
    user_id: str
) -> Dict[int, Dict[str, str]]:
    """
    Resolve UICs to instrument metadata for Live positions.

    Args:
        positions_raw: Raw positions from Saxo API
        access_token: Valid Saxo Bearer token
        user_id: User identifier

    Returns:
        {
            34909: {"symbol": "AAPL", "name": "Apple Inc.", "isin": "US0378331005", "currency": "USD"},
            12345: {"symbol": "MSFT", "name": "Microsoft Corp.", ...},
            ...
        }

    Note: Only processes Live positions (PositionBase format).
          Sim positions already have symbols.
    """
    from services.saxo_uic_resolver import SaxoUICResolver

    uic_metadata = {}

    # Extract UICs from Live positions
    uic_list = []
    for pos in positions_raw:
        if "PositionBase" in pos:
            position_base = pos.get("PositionBase", {})
            uic = position_base.get("Uic")
            asset_type = position_base.get("AssetType", "Stock")

            if uic:
                uic_list.append((uic, asset_type))

    if not uic_list:
        logger.debug("No UICs to resolve (Sim mode or empty positions)")
        return uic_metadata

    # Resolve UICs using cache
    resolver = SaxoUICResolver(user_id=user_id)

    logger.info(f"🔍 Resolving {len(uic_list)} UICs for user '{user_id}'...")

    for uic, asset_type in uic_list:
        try:
            metadata = await resolver.resolve_uic(
                access_token=access_token,
                uic=uic,
                asset_type=asset_type
            )

            if metadata:
                uic_metadata[uic] = metadata

        except Exception as e:
            logger.warning(f"⚠️ Failed to resolve UIC {uic}: {e}")
            continue

    resolved_count = len(uic_metadata)
    cache_hit_rate = (resolved_count / len(uic_list) * 100) if uic_list else 0

    logger.info(f"✅ Resolved {resolved_count}/{len(uic_list)} UICs ({cache_hit_rate:.1f}% success)")

    return uic_metadata


async def _get_account_key(
    auth_service: SaxoAuthService,
    access_token: str,
    user_id: str
) -> Optional[str]:
    """
    Get account_key from user config or fetch from Saxo API.

    Priority:
        1. User config (data/users/{user_id}/config.json)
        2. Cached in tokens (from previous fetch)
        3. Fetch from Saxo API /port/v1/accounts/me

    Args:
        auth_service: Auth service instance
        access_token: Valid access token
        user_id: User identifier

    Returns:
        Account key string or None
    """
    # Check tokens cache first
    account_key = auth_service.get_account_key()
    if account_key:
        logger.debug(f"✅ Account key found in tokens for user '{user_id}'")
        return account_key

    # Check user config (data/users/{user_id}/config.json)
    try:
        user_fs = UserScopedFS(".", user_id)
        user_config = user_fs.read_json("config.json")
        account_key = user_config.get("saxo_api", {}).get("account_key")

        if account_key:
            logger.debug(f"✅ Account key found in config for user '{user_id}'")
            return account_key
    except FileNotFoundError:
        logger.debug(f"No config.json found for user '{user_id}'")
    except Exception as e:
        logger.warning(f"Error reading user config: {e}")

    # Fetch from API
    try:
        oauth_client = SaxoOAuthClient(user_id=user_id)
        url = f"{oauth_client.api_base}/port/v1/accounts/me"
        headers = {"Authorization": f"Bearer {access_token}"}

        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"🔍 Fetching account key from Saxo API for user '{user_id}'")
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            accounts = data.get("Data", [])

            if not accounts:
                logger.warning(f"⚠️ No accounts found for user '{user_id}'")
                return None

            # Use first account
            first_account = accounts[0]
            logger.info(f"📋 Account data from Saxo: {first_account}")  # DEBUG: voir structure complète

            # Try ClientKey first (Live), then AccountKey (Sim)
            account_key = first_account.get("ClientKey") or first_account.get("AccountKey")

            # Cache account_key in tokens for future use
            if account_key:
                tokens = auth_service._load_tokens()
                if tokens:
                    tokens["account_key"] = account_key
                    await auth_service.save_tokens(tokens)
                    logger.info(f"✅ Account key cached for user '{user_id}'")

            return account_key

    except Exception as e:
        logger.error(f"Error fetching account key from API: {e}")
        return None


def _normalize_positions(
    positions_raw: List[Dict[str, Any]],
    uic_metadata: Optional[Dict[int, Dict[str, str]]] = None
) -> List[Dict[str, Any]]:
    """
    Normalize Saxo API positions to match CSV structure.

    Handles two different API formats:
    - Sim: DisplayAndFormat structure
    - Live: PositionBase/PositionView structure

    Args:
        positions_raw: Raw positions from Saxo API
        uic_metadata: Optional UIC resolution cache
                      {34909: {"symbol": "AAPL", "name": "Apple Inc.", "isin": "...", "currency": "USD"}}

    Returns:
        List of normalized positions with enriched metadata
    """
    normalized = []

    for pos in positions_raw:
        try:
            # Initialize variables
            uic = None

            # Detect format: Live (PositionBase/PositionView) or Sim (DisplayAndFormat)
            if "PositionBase" in pos and "PositionView" in pos:
                # Live format
                position_base = pos.get("PositionBase", {})
                position_view = pos.get("PositionView", {})

                uic = position_base.get("Uic", "")
                amount = position_base.get("Amount", 1)
                asset_type = position_base.get("AssetType", "Unknown")
                avg_price = position_base.get("OpenPrice", 0.0)

                market_value = position_view.get("MarketValue", 0.0)
                current_price = position_view.get("CurrentPrice", 0.0)
                pnl = position_view.get("ProfitLossOnTradeInBaseCurrency", 0.0)
                currency = position_view.get("ExposureCurrency", "EUR")

                # Try to resolve UIC to Symbol/Name using cache
                if uic_metadata and uic in uic_metadata:
                    metadata = uic_metadata[uic]
                    symbol = metadata.get("symbol", f"UIC-{uic}")
                    name = metadata.get("name", f"Instrument {uic}")
                    isin = metadata.get("isin", "")
                    # Use API currency if metadata currency is empty
                    if not currency and metadata.get("currency"):
                        currency = metadata.get("currency")
                else:
                    # Fallback if not resolved
                    symbol = f"UIC-{uic}"
                    name = f"Instrument {uic}"
                    isin = ""

            else:
                # Sim format (original)
                display = pos.get("DisplayAndFormat", {})
                symbol_raw = display.get("Symbol", "")
                symbol = symbol_raw.split(":")[0] if ":" in symbol_raw else symbol_raw
                name = display.get("Description", symbol)

                market_value = pos.get("MarketValue", 0.0)
                amount = pos.get("Amount", 1)
                current_price = market_value / amount if amount != 0 else 0.0

                single_pos = pos.get("SinglePositionBase", {})
                avg_price = single_pos.get("OpenPrice", 0.0)
                pnl = single_pos.get("ProfitLossOnTrade", 0.0)

                asset_type = pos.get("AssetType", "Unknown")
                isin = pos.get("Isin", "")
                currency = pos.get("Currency", "EUR")

            # ✅ Build tags for frontend compatibility (dashboard chart grouping)
            tags = []
            if asset_type:
                tags.append(f"asset_class:{asset_type}")

            # Try to get sector from UIC metadata
            sector = None
            if uic_metadata and uic in uic_metadata:
                sector = uic_metadata[uic].get("gics_sector") or uic_metadata[uic].get("sector")

            if sector:
                tags.append(f"sector:{sector}")

            normalized.append({
                "symbol": symbol,
                "name": name,
                "quantity": amount,
                "market_value": market_value,
                "avg_price": avg_price,
                "current_price": current_price,
                "pnl": pnl,
                "asset_type": asset_type,
                "asset_class": asset_type,
                "isin": isin,
                "currency": currency,
                "uic": uic,  # Preserve UIC for debugging (None for Sim positions)
                "tags": tags,  # ✅ CRITICAL: Add tags for frontend chart grouping
                "sector": sector  # ✅ Add sector for filtering/grouping
            })

        except Exception as e:
            logger.warning(f"⚠️ Failed to normalize position: {e}")
            continue

    return normalized
