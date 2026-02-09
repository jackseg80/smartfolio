"""Wealth namespace endpoints unifying holdings across providers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Depends

from adapters import banks_adapter, crypto_adapter, saxo_adapter
from api.deps import get_required_user
from models.wealth import (
    AccountModel,
    InstrumentModel,
    PositionModel,
    TransactionModel,
    PricePoint,
    ProposedTrade,
    BankAccountInput,
    BankAccountOutput,
    WealthItemInput,
    WealthItemOutput,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wealth", tags=["Wealth"])
_SUPPORTED_MODULES = {"crypto", "saxo", "banks"}


async def _module_available(module: str, user_id: Optional[str] = None) -> bool:
    """Check if a wealth module has data available for the user."""
    try:
        if module == "crypto":
            return await crypto_adapter.has_data(user_id)
        if module == "saxo":
            return await saxo_adapter.has_data(user_id)
        if module == "banks":
            return await banks_adapter.has_data(user_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[wealth] availability probe failed for %s: %s", module, exc)
        return False
    return False


def _ensure_module(module: str) -> None:
    if module not in _SUPPORTED_MODULES:
        raise HTTPException(status_code=404, detail="unknown_module")


# ===== Wealth Item CRUD Endpoints =====


@router.get("/items")
async def list_wealth_items(
    user: str = Depends(get_required_user),
    category: Optional[str] = Query(None, regex="^(liquidity|tangible|liability|insurance)$"),
    type: Optional[str] = Query(None, description="Item type filter"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip")
) -> dict:
    """
    List wealth items for user with optional filters and pagination.

    PERFORMANCE FIX: Added pagination (limit/offset) to prevent loading all items.

    Args:
        user: Active user ID (injected via Depends)
        category: Optional category filter
        type: Optional type filter
        limit: Max items per page (default 50)
        offset: Pagination offset (default 0)

    Returns:
        Paginated response with items and metadata
    """
    from services.wealth.wealth_service import list_items

    items = list_items(user, category=category, type=type)

    # Apply pagination
    total_count = len(items)
    paginated_items = items[offset:offset + limit]

    logger.info(f"[wealth] listed {len(paginated_items)}/{total_count} items for user={user}")

    # Convert Pydantic models to dicts for proper JSON serialization
    items_as_dicts = [item.model_dump() for item in paginated_items]

    return {
        "success": True,
        "count": len(items_as_dicts),
        "total_count": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total_count,
        "items": items_as_dicts,
        "filters_applied": {
            "category": category,
            "type": type
        }
    }


@router.get("/items/{item_id}")
async def get_wealth_item(
    item_id: str,
    user: str = Depends(get_required_user)
) -> dict:
    """Get a specific wealth item by ID."""
    from services.wealth.wealth_service import get_item

    item = get_item(user, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="item_not_found")

    logger.info(f"[wealth] retrieved item id={item_id} user={user}")
    return item


@router.post("/items", status_code=201)
async def create_wealth_item(
    item: "WealthItemInput",
    user: str = Depends(get_required_user)
) -> dict:
    """
    Create a new wealth item for the user.

    Args:
        item: Wealth item data
        user: Active user ID (injected via Depends)

    Returns:
        WealthItemOutput with generated ID and USD conversion
    """
    from services.wealth.wealth_service import create_item

    new_item = create_item(user, item)
    logger.info(
        f"[wealth] item created id={new_item.id} user={user} category={item.category}"
    )
    return new_item


@router.put("/items/{item_id}")
async def update_wealth_item(
    item_id: str,
    item: "WealthItemInput",
    user: str = Depends(get_required_user)
) -> dict:
    """
    Update an existing wealth item.

    Args:
        item_id: Item ID to update
        item: Updated item data
        user: Active user ID (injected via Depends)

    Returns:
        Updated WealthItemOutput

    Raises:
        HTTPException 404 if item not found
    """
    from services.wealth.wealth_service import update_item

    updated_item = update_item(user, item_id, item)
    if not updated_item:
        logger.warning(f"[wealth] item not found id={item_id} user={user}")
        raise HTTPException(status_code=404, detail="item_not_found")

    logger.info(f"[wealth] item updated id={item_id} user={user}")
    return updated_item


@router.delete("/items/{item_id}", status_code=204)
async def delete_wealth_item(
    item_id: str,
    user: str = Depends(get_required_user)
):
    """
    Delete a wealth item.

    Args:
        item_id: Item ID to delete
        user: Active user ID (injected via Depends)

    Returns:
        204 No Content on success

    Raises:
        HTTPException 404 if item not found
    """
    from services.wealth.wealth_service import delete_item

    deleted = delete_item(user, item_id)
    if not deleted:
        logger.warning(f"[wealth] item not found for deletion id={item_id} user={user}")
        raise HTTPException(status_code=404, detail="item_not_found")

    logger.info(f"[wealth] item deleted id={item_id} user={user}")


@router.get("/summary")
async def get_wealth_summary(
    user: str = Depends(get_required_user)
) -> dict:
    """
    Get wealth summary for user.

    Returns breakdown by category with total net worth in USD.

    Args:
        user: Active user ID (injected via Depends)

    Returns:
        Dict with net_worth, breakdown by category, and counts
    """
    from services.wealth.wealth_service import get_summary

    summary = get_summary(user)
    logger.info(f"[wealth] summary generated for user={user} net_worth={summary['net_worth']:.2f}")
    return summary


# ===== Banks CRUD Endpoints (RETROCOMPAT - redirects to wealth service) =====


def _wealth_item_to_bank_account(item: WealthItemOutput) -> BankAccountOutput:
    """Convert WealthItemOutput to BankAccountOutput for retrocompat."""
    # Extract bank_name and account_type from metadata
    metadata = item.metadata or {}
    bank_name = metadata.get("bank_name", "Unknown Bank")
    account_type = metadata.get("account_type", "other")

    return BankAccountOutput(
        id=item.id,
        bank_name=bank_name,
        account_type=account_type,
        balance=item.value,
        currency=item.currency,
        balance_usd=item.value_usd,
    )


@router.get("/banks/accounts", response_model=list[BankAccountOutput])
async def list_bank_accounts(
    user: str = Depends(get_required_user)
) -> list[BankAccountOutput]:
    """
    List all bank accounts for user with balance_usd calculated.

    RETROCOMPAT: This endpoint now redirects to wealth service internally.

    Args:
        user: Active user ID (injected via Depends)

    Returns:
        List of BankAccountOutput with USD conversions
    """
    from services.wealth.wealth_service import list_items

    # Get wealth items filtered by category=liquidity and type=bank_account
    items = list_items(user, category="liquidity", type="bank_account")

    # Convert to BankAccountOutput format
    result = [_wealth_item_to_bank_account(item) for item in items]

    logger.info("[wealth][banks][retrocompat] listed %s accounts for user=%s", len(result), user)
    return result


@router.post("/banks/accounts", response_model=BankAccountOutput, status_code=201)
async def create_bank_account(
    account: BankAccountInput,
    user: str = Depends(get_required_user)
) -> BankAccountOutput:
    """
    Create a new bank account for the user.

    RETROCOMPAT: This endpoint now redirects to wealth service internally.

    Args:
        account: Bank account data (bank_name, account_type, balance, currency)
        user: Active user ID (injected via Depends)

    Returns:
        BankAccountOutput with generated ID and USD conversion

    Example:
        POST /api/wealth/banks/accounts
        {
            "bank_name": "UBS",
            "account_type": "current",
            "balance": 5000.0,
            "currency": "CHF"
        }
    """
    from services.wealth.wealth_service import create_item

    # Convert BankAccountInput to WealthItemInput
    wealth_item = WealthItemInput(
        name=f"{account.bank_name} ({account.account_type})",
        category="liquidity",
        type="bank_account",
        value=account.balance,
        currency=account.currency,
        acquisition_date=None,
        notes=None,
        metadata={
            "bank_name": account.bank_name,
            "account_type": account.account_type,
        },
    )

    # Create using wealth service
    new_item = create_item(user, wealth_item)

    logger.info(
        "[wealth][banks][retrocompat] account created id=%s user=%s bank=%s",
        new_item.id,
        user,
        account.bank_name
    )

    # Convert back to BankAccountOutput
    return _wealth_item_to_bank_account(new_item)


@router.put("/banks/accounts/{account_id}", response_model=BankAccountOutput)
async def update_bank_account(
    account_id: str,
    account: BankAccountInput,
    user: str = Depends(get_required_user)
) -> BankAccountOutput:
    """
    Update an existing bank account.

    RETROCOMPAT: This endpoint now redirects to wealth service internally.

    Args:
        account_id: Account ID to update
        account: Updated account data
        user: Active user ID (injected via Depends)

    Returns:
        Updated BankAccountOutput

    Raises:
        HTTPException 404 if account not found
    """
    from services.wealth.wealth_service import update_item

    # Convert BankAccountInput to WealthItemInput
    wealth_item = WealthItemInput(
        name=f"{account.bank_name} ({account.account_type})",
        category="liquidity",
        type="bank_account",
        value=account.balance,
        currency=account.currency,
        acquisition_date=None,
        notes=None,
        metadata={
            "bank_name": account.bank_name,
            "account_type": account.account_type,
        },
    )

    # Update using wealth service
    updated_item = update_item(user, account_id, wealth_item)

    if not updated_item:
        logger.warning("[wealth][banks][retrocompat] account not found id=%s user=%s", account_id, user)
        raise HTTPException(status_code=404, detail="account_not_found")

    logger.info(
        "[wealth][banks][retrocompat] account updated id=%s user=%s bank=%s",
        account_id,
        user,
        account.bank_name
    )

    # Convert back to BankAccountOutput
    return _wealth_item_to_bank_account(updated_item)


@router.delete("/banks/accounts/{account_id}", status_code=204)
async def delete_bank_account(
    account_id: str,
    user: str = Depends(get_required_user)
):
    """
    Delete a bank account.

    RETROCOMPAT: This endpoint now redirects to wealth service internally.

    Args:
        account_id: Account ID to delete
        user: Active user ID (injected via Depends)

    Returns:
        204 No Content on success

    Raises:
        HTTPException 404 if account not found
    """
    from services.wealth.wealth_service import delete_item

    # Delete using wealth service
    deleted = delete_item(user, account_id)

    if not deleted:
        logger.warning("[wealth][banks][retrocompat] account not found for deletion id=%s user=%s", account_id, user)
        raise HTTPException(status_code=404, detail="account_not_found")

    logger.info("[wealth][banks][retrocompat] account deleted id=%s user=%s", account_id, user)


# ===== Generic Wealth Endpoints =====


@router.get("/modules", response_model=List[str])
async def list_modules(user: str = Depends(get_required_user)) -> List[str]:
    """List available wealth modules for the user."""
    discovered = []
    for module in _SUPPORTED_MODULES:
        if await _module_available(module, user):
            discovered.append(module)
    if not discovered:
        discovered = sorted(_SUPPORTED_MODULES)
    logger.info("[wealth] modules_available=%s for user=%s", discovered, user)
    return discovered


@router.get("/{module}/accounts", response_model=List[AccountModel])
async def get_accounts(
    module: str,
    user: str = Depends(get_required_user),
    source: str = Query("auto", description="Crypto source resolver"),
    file_key: Optional[str] = Query(None, description="Specific file to load (for Saxo)"),
) -> List[AccountModel]:
    """List accounts for a module (read-only from sources)."""
    _ensure_module(module)

    # Check user has data for this module
    if not await _module_available(module, user):
        logger.warning("[wealth] no data available for module=%s user=%s", module, user)
        return []

    if module == "crypto":
        accounts = await crypto_adapter.list_accounts(user_id=user, source=source)
    elif module == "saxo":
        accounts = await saxo_adapter.list_accounts(user_id=user, file_key=file_key)
    else:
        accounts = await banks_adapter.list_accounts(user_id=user)

    logger.info("[wealth] served %s accounts for module=%s user=%s", len(accounts), module, user)
    return accounts


@router.get("/{module}/instruments", response_model=List[InstrumentModel])
async def get_instruments(
    module: str,
    user: str = Depends(get_required_user),
    source: str = Query("auto"),
    file_key: Optional[str] = Query(None, description="Specific file to load (for Saxo)"),
) -> List[InstrumentModel]:
    _ensure_module(module)
    if module == "crypto":
        instruments = await crypto_adapter.list_instruments(user_id=user, source=source)
    elif module == "saxo":
        instruments = await saxo_adapter.list_instruments(user_id=user, file_key=file_key)
    else:
        instruments = await banks_adapter.list_instruments(user_id=user)
    logger.info("[wealth] served %s instruments for module=%s", len(instruments), module)
    return instruments


@router.get("/{module}/positions", response_model=List[PositionModel])
async def get_positions(
    module: str,
    user: str = Depends(get_required_user),
    source: str = Query("auto"),
    file_key: Optional[str] = Query(None, description="Specific file to load (for Saxo)"),
    min_usd_threshold: float = Query(1.0, description="Minimum USD value to filter dust assets"),
    asof: Optional[str] = Query(None, description="Date override (yyyy-mm-dd)")
) -> List[PositionModel]:
    _ensure_module(module)
    if module == "crypto":
        positions = await crypto_adapter.list_positions(user_id=user, source=source, min_usd_threshold=min_usd_threshold)
    elif module == "saxo":
        positions = await saxo_adapter.list_positions(user_id=user, file_key=file_key)
    else:
        positions = await banks_adapter.list_positions(user_id=user)
    logger.info("[wealth] served %s positions for module=%s asof=%s", len(positions), module, asof or "latest")
    return positions


@router.get("/{module}/transactions", response_model=List[TransactionModel])
async def get_transactions(
    module: str,
    user: str = Depends(get_required_user),
    source: str = Query("auto"),
    file_key: Optional[str] = Query(None, description="Specific file to load (for Saxo)"),
    start: Optional[str] = Query(None, alias="from"),
    end: Optional[str] = Query(None, alias="to"),
) -> List[TransactionModel]:
    _ensure_module(module)
    if module == "crypto":
        transactions = await crypto_adapter.list_transactions(user_id=user, source=source, start=start, end=end)
    elif module == "saxo":
        transactions = await saxo_adapter.list_transactions(user_id=user, file_key=file_key, start=start, end=end)
    else:
        transactions = await banks_adapter.list_transactions(start=start, end=end, user_id=user)
    logger.info(
        "[wealth] served %s transactions for module=%s window=%s/%s",
        len(transactions),
        module,
        start or "*",
        end or "*",
    )
    return transactions


@router.get("/{module}/prices", response_model=List[PricePoint])
async def get_prices(
    module: str,
    ids: List[str] = Query(..., description="Instrument identifiers"),
    granularity: str = Query("daily", regex="^(daily|intraday)$"),
    user: str = Depends(get_required_user),
    source: str = Query("auto"),
    file_key: Optional[str] = Query(None, description="Specific file to load (for Saxo)"),
) -> List[PricePoint]:
    _ensure_module(module)
    if not ids:
        raise HTTPException(status_code=400, detail="missing_ids")
    if module == "crypto":
        prices = await crypto_adapter.get_prices(ids, granularity=granularity)
    elif module == "saxo":
        prices = await saxo_adapter.get_prices(ids, granularity=granularity, user_id=user, file_key=file_key)
    else:
        prices = await banks_adapter.get_prices(ids, granularity=granularity, user_id=user)
    logger.info("[wealth] served %s price points for module=%s", len(prices), module)
    return prices


@router.post("/{module}/rebalance/preview", response_model=List[ProposedTrade])
async def preview_rebalance(
    module: str,
    payload: Optional[dict] = Body(default=None, description="Module-specific preview payload"),
    user: str = Depends(get_required_user),
    source: str = Query("auto"),
    file_key: Optional[str] = Query(None, description="Specific file to load (for Saxo)"),
) -> List[ProposedTrade]:
    _ensure_module(module)
    if module == "crypto":
        trades = await crypto_adapter.preview_rebalance(user_id=user, source=source)
    elif module == "saxo":
        trades = await saxo_adapter.preview_rebalance(user_id=user, file_key=file_key)
    else:
        trades = await banks_adapter.preview_rebalance(user_id=user)
    logger.info(
        "[wealth] rebalance preview module=%s payload_keys=%s returned=%s",
        module,
        sorted(list(payload.keys())) if payload else [],
        len(trades),
    )
    return trades


@router.get("/global/summary")
async def global_summary(
    user: str = Depends(get_required_user),
    source: str = Query("auto", description="Crypto source resolver"),
    min_usd_threshold: float = Query(1.0, description="Minimum USD value to filter dust assets"),
    bourse_file_key: Optional[str] = Query(None, description="Bourse file key for specific CSV selection"),
    bourse_source: Optional[str] = Query(None, description="Bourse source (api:saxobank_api or saxo:file_key)")
) -> dict:
    """
    Global aggregation of all wealth modules (crypto + saxo + wealth items).

    Returns unified summary with total_value_usd and breakdown by module.

    Args:
        user: User ID (from authenticated context)
        source: Source resolver for crypto
        bourse_file_key: Optional file key for specific Bourse CSV selection

    Returns:
        Dict with total_value_usd, breakdown, and metadata

    Example:
        GET /api/wealth/global/summary?source=auto&bourse_file_key=saxo_25-09-2025.csv
        {
            "total_value_usd": 556100.0,
            "breakdown": {
                "crypto": 133100.0,
                "saxo": 423000.0,
                "wealth": 0.0
            },
            "user_id": "jack",
            "timestamp": "2025-10-12T..."
        }
    """
    from datetime import datetime

    breakdown = {
        "crypto": 0.0,
        "saxo": 0.0,
        "wealth": 0.0
    }

    # 1) Crypto
    try:
        logger.info(f"[wealth][global] Checking Crypto availability for user={user}")
        crypto_available = await _module_available("crypto", user)
        logger.info(f"[wealth][global] Crypto available: {crypto_available}")

        if crypto_available:
            logger.info(f"[wealth][global] Loading Crypto positions with source={source}")
            crypto_positions = await crypto_adapter.list_positions(user_id=user, source=source, min_usd_threshold=min_usd_threshold)
            logger.info(f"[wealth][global] Got {len(crypto_positions)} Crypto positions")
            breakdown["crypto"] = sum((p.market_value or 0.0) for p in crypto_positions)
            logger.info(f"[wealth][global] crypto={breakdown['crypto']:.2f} USD for user={user} (threshold={min_usd_threshold})")
        else:
            logger.warning(f"[wealth][global] Crypto module not available for user={user}")
    except Exception as e:
        logger.error(f"[wealth][global] crypto failed for user={user}: {e}", exc_info=True)

    # 2) Saxo (with API and file_key support + cash)
    try:
        logger.info(f"[wealth][global] Checking Saxo availability for user={user}")
        saxo_available = await _module_available("saxo", user)
        logger.info(f"[wealth][global] Saxo available: {saxo_available}")

        if saxo_available:
            logger.info(f"[wealth][global] Bourse source param: {bourse_source}, file_key param: {bourse_file_key}")
            if bourse_source == "manual_bourse":
                logger.info(f"[wealth][global] Loading Saxo via Manual mode: {bourse_source}")
                try:
                    from services.sources import source_registry

                    project_root = str(Path(__file__).parent.parent)
                    manual_source = source_registry.get_source("manual_bourse", user, project_root)

                    if manual_source:
                        items = await manual_source.get_balances()
                        total_value = sum(float(item.value_usd or 0) for item in items)
                        breakdown["saxo"] = total_value
                        logger.info(f"[wealth][global] Manual bourse: {len(items)} positions, total=${total_value:.2f} USD")
                    else:
                        logger.warning(f"[wealth][global] Manual bourse source not available for user {user}")
                        breakdown["saxo"] = 0.0
                except Exception as manual_error:
                    logger.error(f"[wealth][global] Manual bourse load failed: {manual_error}", exc_info=True)
                    breakdown["saxo"] = 0.0
            elif bourse_source and bourse_source.startswith('api:'):
                logger.info(f"[wealth][global] Loading Saxo via API mode: {bourse_source}")
                try:
                    from services.saxo_auth_service import SaxoAuthService

                    auth_service = SaxoAuthService(user)

                    if not auth_service.is_connected():
                        logger.warning(f"[wealth][global] User {user} not connected to Saxo API")
                        breakdown["saxo"] = 0.0
                    else:
                        cached_data = await auth_service.get_cached_positions(max_age_hours=24)

                        if cached_data and cached_data.get("total_value", 0) > 0:
                            total_value = cached_data.get("total_value", 0.0)
                            cash_balance = cached_data.get("cash_balance", 0.0)
                            positions_count = len(cached_data.get("positions", []))

                            breakdown["saxo"] = total_value
                            logger.info(f"[wealth][global] Saxo (cached): {positions_count} positions, total=${total_value:.2f} USD (cash=${cash_balance:.2f})")
                        else:
                            logger.warning(f"[wealth][global] No cached Saxo data available for user {user}")
                            breakdown["saxo"] = 0.0

                except Exception as api_error:
                    logger.error(f"[wealth][global] Saxo cache read failed: {api_error}", exc_info=True)
                    breakdown["saxo"] = 0.0
            else:
                # CSV mode: use file_key
                logger.info(f"[wealth][global] Loading Saxo positions with file_key={bourse_file_key}")
                saxo_positions = await saxo_adapter.list_positions(user_id=user, file_key=bourse_file_key)
                logger.info(f"[wealth][global] Got {len(saxo_positions)} Saxo positions")
                breakdown["saxo"] = sum((p.market_value or 0.0) for p in saxo_positions)

                # Add cash/liquidities if available
                try:
                    import json
                    cash_key = bourse_file_key or "default"
                    cash_dir = Path(f"data/users/{user}/saxobank/cash")
                    cash_file = cash_dir / f"{cash_key}_cash.json"

                    if cash_file.exists():
                        with open(cash_file, 'r', encoding='utf-8') as f:
                            cash_data = json.load(f)
                            cash_amount = float(cash_data.get("cash_amount", 0.0))
                            breakdown["saxo"] += cash_amount
                            logger.info(f"[wealth][global] Added cash ${cash_amount:.2f} to Saxo total")
                except Exception as cash_error:
                    logger.debug(f"[wealth][global] Cash file not found or error (non-blocking): {cash_error}")

                logger.info(f"[wealth][global] saxo={breakdown['saxo']:.2f} USD for user={user} file_key={bourse_file_key}")
        else:
            logger.warning(f"[wealth][global] Saxo module not available for user={user}")
    except Exception as e:
        logger.error(f"[wealth][global] saxo failed for user={user}: {e}", exc_info=True)

    # 3) Wealth items (net worth: assets - liabilities)
    try:
        from services.wealth import wealth_service

        logger.info(f"[wealth][global] Loading Wealth summary for user={user}")
        wealth_summary = wealth_service.get_summary(user_id=user)

        # Use net_worth (assets - liabilities) for accurate wealth representation
        breakdown["wealth"] = wealth_summary.get("net_worth", 0.0)
        logger.info(f"[wealth][global] wealth={breakdown['wealth']:.2f} USD (net worth) for user={user}")
    except Exception as e:
        logger.warning(f"[wealth][global] wealth items failed for user={user}: {e}")

    total_value_usd = sum(breakdown.values())

    # Calculate P&L Today (aggregate across all modules)
    pnl_today = 0.0
    pnl_today_pct = 0.0

    try:
        from services.portfolio import portfolio_analytics

        # Calculate P&L for crypto module
        if breakdown["crypto"] > 0:
            try:
                crypto_metrics = {"total_value_usd": breakdown["crypto"]}
                crypto_perf = portfolio_analytics.calculate_performance_metrics(
                    crypto_metrics,
                    user_id=user,
                    source=source,
                    anchor="prev_snapshot",
                    window="24h"
                )
                if crypto_perf.get("performance_available"):
                    pnl_today += crypto_perf.get("absolute_change_usd", 0.0)
                    logger.debug(f"[wealth][global] Crypto P&L: {crypto_perf.get('absolute_change_usd', 0.0):.2f} USD")
            except Exception as e:
                logger.debug(f"[wealth][global] Crypto P&L calculation skipped: {e}")

        # Calculate P&L for saxo module (use 'saxobank' as source)
        if breakdown["saxo"] > 0:
            try:
                saxo_metrics = {"total_value_usd": breakdown["saxo"]}
                saxo_perf = portfolio_analytics.calculate_performance_metrics(
                    saxo_metrics,
                    user_id=user,
                    source="saxobank",
                    anchor="prev_snapshot",
                    window="24h"
                )
                if saxo_perf.get("performance_available"):
                    pnl_today += saxo_perf.get("absolute_change_usd", 0.0)
                    logger.debug(f"[wealth][global] Saxo P&L: {saxo_perf.get('absolute_change_usd', 0.0):.2f} USD")
            except Exception as e:
                logger.debug(f"[wealth][global] Saxo P&L calculation skipped: {e}")

        # Calculate P&L for wealth module
        if breakdown["wealth"] > 0:
            try:
                wealth_metrics = {"total_value_usd": breakdown["wealth"]}
                wealth_perf = portfolio_analytics.calculate_performance_metrics(
                    wealth_metrics,
                    user_id=user,
                    source="wealth",
                    anchor="prev_snapshot",
                    window="24h"
                )
                if wealth_perf.get("performance_available"):
                    pnl_today += wealth_perf.get("absolute_change_usd", 0.0)
                    logger.debug(f"[wealth][global] Wealth P&L: {wealth_perf.get('absolute_change_usd', 0.0):.2f} USD")
            except Exception as e:
                logger.debug(f"[wealth][global] Wealth P&L calculation skipped: {e}")

        # Calculate percentage if we have historical data
        if total_value_usd > 0:
            historical_value = total_value_usd - pnl_today
            if historical_value > 0:
                pnl_today_pct = (pnl_today / historical_value) * 100

        logger.info(f"[wealth][global] Total P&L Today: {pnl_today:.2f} USD ({pnl_today_pct:.2f}%)")
    except Exception as e:
        logger.warning(f"[wealth][global] P&L Today calculation failed: {e}")
        # Set to 0 on error (non-blocking)
        pnl_today = 0.0
        pnl_today_pct = 0.0

    return {
        "total_value_usd": total_value_usd,
        "breakdown": breakdown,
        "pnl_today": pnl_today,
        "pnl_today_pct": pnl_today_pct,
        "user_id": user,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==================== EXPORT LISTS ====================

@router.get("/banks/export-lists")
async def export_bank_lists(
    user: str = Depends(get_required_user),
    format: str = Query("json", regex="^(json|csv|markdown)$")
) -> dict:
    """
    Export bank accounts list in multiple formats.

    Args:
        user: User ID (from authenticated context)
        format: Output format (json, csv, markdown)

    Returns:
        Exported data in requested format with Content-Type header
    """
    try:
        from services.export_formatter import ExportFormatter
        from services.fx_service import convert as fx_convert
        from fastapi.responses import PlainTextResponse

        snapshot = banks_adapter.load_snapshot(user)
        accounts = snapshot.get("accounts", [])

        accounts_list = []
        total_value_usd = 0

        for acc in accounts:
            balance = acc.get("balance", 0)
            currency = acc.get("currency", "USD").upper()
            balance_usd = fx_convert(balance, currency, "USD")

            accounts_list.append({
                "bank_name": acc.get("bank_name", ""),
                "account_type": acc.get("account_type", ""),
                "balance": balance,
                "currency": currency,
                "balance_usd": balance_usd
            })

            total_value_usd += balance_usd

        export_data = {
            "accounts": accounts_list,
            "summary": {
                "total_value_usd": total_value_usd,
                "accounts_count": len(accounts_list)
            }
        }

        formatter = ExportFormatter('banks')

        if format == 'json':
            content = formatter.to_json(export_data)
            return PlainTextResponse(content, media_type="application/json")
        elif format == 'csv':
            content = formatter.to_csv(export_data)
            return PlainTextResponse(content, media_type="text/csv")
        elif format == 'markdown':
            content = formatter.to_markdown(export_data)
            return PlainTextResponse(content, media_type="text/markdown")

    except Exception as e:
        logger.exception("Error exporting bank lists")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/export-lists")
async def export_wealth_lists(
    user: str = Depends(get_required_user),
    format: str = Query("json", regex="^(json|csv|markdown)$")
) -> dict:
    """
    Export wealth items list in multiple formats.
    Includes all categories: liquidity, tangible, liability, insurance.

    Args:
        user: User ID (from authenticated context)
        format: Output format (json, csv, markdown)

    Returns:
        Exported data in requested format with Content-Type header
    """
    try:
        from services.export_formatter import ExportFormatter
        from services.wealth.wealth_service import list_items, get_summary
        from fastapi.responses import PlainTextResponse

        all_items = list_items(user)
        summary = get_summary(user)

        items_by_category = {
            "liquidity": [],
            "tangible": [],
            "liability": [],
            "insurance": []
        }

        for item in all_items:
            category = item.category
            items_by_category[category].append({
                "id": item.id,
                "name": item.name,
                "type": item.type,
                "value": item.value,
                "currency": item.currency,
                "value_usd": item.value_usd,
                "acquisition_date": item.acquisition_date,
                "notes": item.notes
            })

        export_data = {
            "items_by_category": items_by_category,
            "summary": {
                "net_worth": summary["net_worth"],
                "total_assets": summary["total_assets"],
                "total_liabilities": summary["total_liabilities"],
                "breakdown": summary["breakdown"],
                "counts": summary["counts"]
            }
        }

        formatter = ExportFormatter('wealth')

        if format == 'json':
            content = formatter.to_json(export_data)
            return PlainTextResponse(content, media_type="application/json")
        elif format == 'csv':
            content = formatter.to_csv(export_data)
            return PlainTextResponse(content, media_type="text/csv")
        elif format == 'markdown':
            content = formatter.to_markdown(export_data)
            return PlainTextResponse(content, media_type="text/markdown")

    except Exception as e:
        logger.exception("Error exporting wealth lists")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
