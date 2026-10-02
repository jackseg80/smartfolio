"""Authenticated adapters to the shared, read-only capability service."""
from fastapi import APIRouter, Depends, Query
from typing import Optional
from datetime import datetime, timezone
from api.deps import get_required_user
from api.schemas.ml_contract import UnifiedMLRequest, UnifiedMLResponse, ModelType, Horizon
from services.ml.reliability import capability_service

router = APIRouter(tags=["ML Unified Contract"])

@router.get("/overview")
async def overview(user: str = Depends(get_required_user), source: str = Query("cointracking", min_length=1, max_length=100), market: str = Query("crypto", pattern="^(crypto|stocks)$"), assets: str = Query("BTC,ETH,SOL", max_length=1000), mode: str = Query("benchmarks", pattern="^(portfolio|benchmarks)$"), limit: int = Query(25, ge=1, le=250), file_key: str | None = Query(None)):
    context = None
    if mode == "portfolio":
        from services.ml.portfolio_context import read_context, select_assets
        try:
            raw = await read_context(user, source, market, file_key)
            symbols, context = select_assets(raw, market, limit)
        except (ValueError, FileNotFoundError, OSError) as exc:
            symbols = []
            context = dict(availability="Unavailable", reason=str(exc), held_positions=None, selected_assets=0)
    else:
        symbols = list(dict.fromkeys(a.strip().upper() for a in assets.split(",") if a.strip()))[:50]
    if capability_service.refresh_observations:
        import asyncio
        semaphore=asyncio.Semaphore(4)
        async def prepare(asset):
            async with semaphore:
                try:
                    await asyncio.to_thread(capability_service.history,market,asset)
                except (ValueError,FileNotFoundError,OSError):
                    pass  # The result adapter exposes the exact unavailable reason.
        await asyncio.gather(*(prepare(asset) for asset in symbols))
    result = await capability_service.overview(user, source, symbols, market)
    result["scope"] = "selected_authenticated_portfolio" if mode == "portfolio" else "explicit_market_universe"
    result["portfolio_context"] = context
    return result

@router.post("/unified/predict", response_model=UnifiedMLResponse)
async def unified_predict(request: UnifiedMLRequest, user: str = Depends(get_required_user)):
    start = datetime.now(timezone.utc)
    results = [await capability_service.result(asset, request.market, request.model_type, request.horizon) for asset in request.assets]
    capability_service.journal(user, request.source, results)
    if not request.include_uncertainty:
        for result in results:
            result.uncertainty = None
    return UnifiedMLResponse(success=True, model_type=request.model_type, horizon=request.horizon,
        predictions=results, user_id=user, source=request.source,
        failed_assets=[r.asset for r in results if r.value is None],
        warnings=[f"{r.asset}: {r.reason}" for r in results if r.value is None],
        processing_time_ms=(datetime.now(timezone.utc)-start).total_seconds()*1000)

@router.get("/unified/volatility/{symbol}", response_model=UnifiedMLResponse)
async def unified_volatility_predict(symbol: str, horizon: Horizon = Query(Horizon.D30),
    include_uncertainty: bool = Query(True), include_metadata: bool = Query(False),
    source: str = Query("cointracking"), market: str = Query("crypto", pattern="^(crypto|stocks)$"),
    user: str = Depends(get_required_user)):
    return await unified_predict(UnifiedMLRequest(assets=[symbol.upper()], model_type=ModelType.VOLATILITY,
        horizon=horizon, market=market, source=source, include_uncertainty=include_uncertainty,
        include_metadata=include_metadata), user)

async def _get_raw_prediction(asset: str, model_type: ModelType, horizon: Optional[Horizon]) -> Optional[float]:
    """Compatibility helper; diagnostics never masquerade as forecasts."""
    result = await capability_service.result(asset, "crypto", model_type, horizon)
    return result.value if result.nature == "forecast" and isinstance(result.value, (int, float)) else None


@router.get("/cycle-history")
async def cycle_history(user: str = Depends(get_required_user)):
    """Verified observed closes for descriptive cycle comparisons; no forecast."""
    import asyncio
    from services.ml.cycle_diagnostics import historical_cycle_comparison
    try:
        close, receipt = await asyncio.to_thread(capability_service.history, "crypto", "BTC")
        return historical_cycle_comparison(capability_service.root, close, receipt)
    except (ValueError, FileNotFoundError, OSError) as exc:
        return {"availability": "Unavailable", "reason": str(exc), "cycles": []}
