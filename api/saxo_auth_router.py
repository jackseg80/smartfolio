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
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import RedirectResponse

from api.deps import get_active_user
from api.utils import success_response, error_response
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
        oauth_client = SaxoOAuthClient()

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
    try:
        # Retrieve PKCE verifier
        pkce_data = retrieve_pkce_verifier(state)

        if not pkce_data:
            logger.error(f"Invalid or expired state: {state[:8]}...")
            return RedirectResponse(
                url="/settings.html?status=error&message=Invalid+or+expired+state"
            )

        code_verifier = pkce_data["code_verifier"]
        user_id = pkce_data["user_id"]

        # Exchange code for tokens
        oauth_client = SaxoOAuthClient()
        tokens = await oauth_client.exchange_code_for_tokens(code, code_verifier)

        # Save tokens
        auth_service = SaxoAuthService(user_id)
        await auth_service.save_tokens(tokens)

        logger.info(f"✅ OAuth callback successful for user '{user_id}'")

        # Redirect to settings page with success
        return RedirectResponse(url="/static/settings.html?status=connected")

    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        error_msg = str(e).replace(" ", "+")
        return RedirectResponse(
            url=f"/static/settings.html?status=error&message={error_msg}"
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
