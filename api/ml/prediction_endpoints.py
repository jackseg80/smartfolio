"""
ML Prediction Endpoints - Prédictions ML unifiées

Ce module gère:
- Prédictions de volatilité
- Prédictions de régime
- Sentiment analysis
- Prédictions live
- Corrélations

Extrait de unified_ml_endpoints.py pour modularité (Fév 2026).
"""

from fastapi import APIRouter, HTTPException, Query, Body, Depends
from typing import Dict, List, Optional, Any
import logging
import numpy as np
from datetime import datetime
from pydantic import BaseModel

from services.ml.orchestrator import get_orchestrator, get_ml_predictions
from services.ml_pipeline_manager_optimized import optimized_pipeline_manager as pipeline_manager
from api.deps import get_required_user
from api.utils.formatters import success_response, error_response
from shared.error_handlers import handle_api_errors, handle_service_errors
from .cache_utils import get_ml_cache, cache_get, cache_set

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ML Predictions"])


# ===== PYDANTIC MODELS =====

class PredictionRequest(BaseModel):
    """Request model for ML predictions"""
    assets: List[str]
    horizon_days: int = 30
    horizons: Optional[List[int]] = None  # [1, 7, 30] pour multi-horizon
    include_regime: bool = True
    include_volatility: bool = True
    include_confidence: bool = False


class PredictionResponse(BaseModel):
    """Response model for ML predictions"""
    success: bool
    predictions: Optional[Dict]
    regime_prediction: Optional[Dict]
    volatility_forecast: Optional[Dict]
    model_status: Dict
    timestamp: str


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""
    success: bool = True
    symbol: str
    aggregated_sentiment: Dict[str, Any]
    sources_used: List[str] = []
    metadata: Dict[str, Any] = {}


# ===== UNIFIED PREDICTIONS =====

@router.post("/predict", response_model=PredictionResponse)
@handle_api_errors(
    fallback={"predictions": {}, "regime_prediction": None, "volatility_forecast": None, "model_status": {}},
    reraise_http_errors=True
)
async def unified_predictions(request: PredictionRequest):
    """
    Prédictions ML unifiées - volatilité, régime, corrélations
    Support multi-horizon: horizons=[1, 7, 30] pour 1j, 7j, 30j
    """
    ml_cache = get_ml_cache()
    cache_key = f"predictions_{hash(str(request.dict()))}"
    cached_result = cache_get(ml_cache, cache_key, 300)  # 5 min cache
    if cached_result:
        return cached_result

    orchestrator = get_orchestrator()
    horizons = request.horizons if request.horizons else [request.horizon_days]

    predictions = await get_ml_predictions(symbols=request.assets)

    enhanced_predictions = {}
    if request.include_volatility and len(horizons) > 1:
        enhanced_predictions = await _get_multi_horizon_predictions(
            request.assets, horizons, request.include_confidence
        )

    final_predictions = predictions.get("predictions", {})
    if enhanced_predictions:
        for symbol in request.assets:
            if symbol in enhanced_predictions:
                if symbol not in final_predictions:
                    final_predictions[symbol] = {}
                final_predictions[symbol].update(enhanced_predictions[symbol])

    if request.include_confidence:
        final_predictions = await _add_confidence_metrics(final_predictions, request.assets)

    result = PredictionResponse(
        success=True,
        predictions=final_predictions,
        regime_prediction=predictions.get("regime") if request.include_regime else None,
        volatility_forecast=predictions.get("volatility") if request.include_volatility else None,
        model_status=predictions.get("model_status", {}),
        timestamp=datetime.now().isoformat()
    )

    cache_set(ml_cache, cache_key, result)
    logger.info(f"Unified predictions generated for {len(request.assets)} assets, horizons: {horizons}")
    return result


# ===== VOLATILITY PREDICTIONS =====

@router.get("/volatility/predict/{symbol}")
@handle_api_errors(fallback={"volatility_forecast": None})
async def predict_volatility(symbol: str, horizon_days: int = Query(30, ge=1, le=365)) -> dict:
    """
    Prédiction de volatilité pour un asset spécifique
    """
    ml_cache = get_ml_cache()
    cache_key = f"volatility_{symbol}_{horizon_days}"
    cached_result = cache_get(ml_cache, cache_key, 600)  # 10 min cache
    if cached_result:
        return cached_result

    orchestrator = get_orchestrator()
    prediction = await orchestrator.predict_volatility(symbol, horizon_days)

    result = success_response({
        "symbol": symbol,
        "horizon_days": horizon_days,
        "volatility_forecast": prediction,
        "timestamp": datetime.now().isoformat()
    })

    cache_set(ml_cache, cache_key, result)
    return result


@router.post("/volatility/train-portfolio")
@handle_api_errors(fallback={"trainable_assets": 0, "loaded": 0, "results": {}})
async def alias_train_portfolio(symbols: Optional[List[str]] = Query(None)) -> dict:
    """Alias that preloads requested volatility models instead of training."""
    req_symbols = symbols or ["BTC", "ETH"]
    results = {}
    for s in req_symbols:
        results[s] = pipeline_manager.load_volatility_model(s)
    loaded = sum(1 for v in results.values() if v)
    return success_response({
        "trainable_assets": len(req_symbols),
        "estimated_duration_minutes": 1,
        "results": results,
        "loaded": loaded
    })


@router.post("/volatility/batch-predict")
@handle_api_errors(fallback={"predictions": {}})
async def alias_batch_predict(payload: Dict[str, Any] = Body(default={})) -> dict:
    """Alias that forwards to unified /predict."""
    assets = payload.get("symbols") or payload.get("assets") or ["BTC", "ETH"]
    horizons = [1, 7, 30]
    req = PredictionRequest(assets=assets, horizons=horizons, include_regime=False, include_volatility=True)
    return await unified_predictions(req)


# ===== REGIME PREDICTIONS =====

@router.get("/regime/current")
@handle_api_errors(fallback={"regime_prediction": {"available": False, "reason": "Regime inference failed"}})
async def alias_regime_current() -> dict:
    """Alias that returns current/live regime signal."""
    live = await get_live_predictions()
    regime_val = live.get("regime_prediction") or live.get("market_regime")

    if isinstance(regime_val, dict) and regime_val.get("available") is True:
        regime_obj = {
            "available": True,
            "regime_name": regime_val.get("regime_name") or regime_val.get("name") or "Unknown",
            "confidence": regime_val.get("confidence"),
            "duration_days": regime_val.get("duration_days", 0)
        }
    else:
        regime_obj = {
            "available": False,
            "regime_name": None,
            "confidence": None,
            "duration_days": None,
            "reason": live.get("reason", "No verified live regime inference is connected")
        }

    return success_response({
        "regime_prediction": regime_obj,
        "timestamp": live.get("timestamp")
    })


# ===== LIVE PREDICTIONS =====

@router.get("/predictions/live")
@handle_api_errors(fallback={"available": False, "reason": "Live prediction lookup failed", "models_used": {}})
async def get_live_predictions() -> dict:
    """
    Obtenir les prédictions en temps réel basées sur les modèles entraînés
    """
    orchestrator = get_orchestrator()
    pipeline_status = await orchestrator.get_model_status()

    regime_loaded = pipeline_status.get('pipeline_status', {}).get('regime_models', {}).get('model_loaded', False)

    return {
        "available": False,
        "btc_volatility": None,
        "eth_volatility": None,
        "market_regime": None,
        "fear_greed_index": None,
        "reason": "No verified live inference is connected to this compatibility endpoint",
        "models_used": {
            "regime_model_loaded": regime_loaded,
            "based_on_training": False
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/portfolio-metrics")
@handle_api_errors(fallback={"available": False, "metrics": None})
async def get_portfolio_metrics() -> dict:
    """
    Obtenir les métriques de portefeuille ML (stub endpoint)
    """
    return success_response({
        "available": False,
        "metrics": None,
        "reason": "No identity-bound portfolio metric calculation is connected to this compatibility endpoint",
        "timestamp": datetime.now().isoformat()
    })


# ===== SENTIMENT ENDPOINTS =====

@router.get("/sentiment/{symbol}")
@handle_api_errors(fallback={"available": False, "aggregated_sentiment": None})
async def get_sentiment(symbol: str, days: int = Query(default=1, ge=1, le=30)) -> dict:
    """
    Obtenir le sentiment pour un asset (stub endpoint)
    """
    return {
        "available": False,
        "symbol": symbol.upper(),
        "aggregated_sentiment": None,
        "sources_used": [],
        "reason": "No verified sentiment inference is connected to this compatibility endpoint",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/sentiment/fear-greed")
@handle_api_errors(fallback={"available": False, "fear_greed_data": None})
async def get_fear_greed_sentiment(days: int = Query(default=1, ge=1, le=30)) -> dict:
    """
    Obtenir Fear & Greed index (stub endpoint)
    """
    return success_response({
        "available": False,
        "fear_greed_data": None,
        "reason": "No verified Fear & Greed observation is connected to this endpoint",
        "timestamp": datetime.now().isoformat()
    })


@router.get("/sentiment/analyze")
@handle_api_errors(fallback={"results": {}})
async def alias_sentiment_analyze(symbols: str = Query("BTC,ETH"), days: int = Query(7)) -> dict:
    """Alias that aggregates sentiment for multiple symbols."""
    syms = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    results = {}
    for s in syms:
        single = await get_sentiment(s, days)
        results[s] = single.get("aggregated_sentiment") if isinstance(single, dict) else None
    return success_response({"results": results, "days": days})


@router.get("/sentiment/symbol/{symbol}", response_model=SentimentResponse)
@handle_api_errors(
    fallback={
        "success": False,
        "symbol": "BTC",
        "aggregated_sentiment": {"available": False},
        "sources_used": [],
        "metadata": {"error": "Sentiment analysis failed"}
    },
    reraise_http_errors=False
)
async def get_symbol_sentiment(
    symbol: str,
    days: int = Query(1, ge=1, le=30, description="Number of days for sentiment analysis"),
    include_breakdown: bool = Query(True, description="Include detailed source breakdown")
):
    """
    Get sentiment analysis for a cryptocurrency symbol
    """
    logger.debug(f"Getting sentiment analysis for {symbol} over {days} days")

    ml_cache = get_ml_cache()
    cache_key = f"sentiment:{symbol}:{days}:{include_breakdown}"
    cached_result = cache_get(ml_cache, cache_key, 900)
    if cached_result:
        logger.debug(f"Returning cached sentiment for {symbol}")
        return cached_result

    try:
        from services.execution.governance import governance_engine
        current_state = await governance_engine.get_current_state()
        signals = current_state.signals if current_state else None
    except Exception as e:
        logger.warning(f"Could not get verified governance sentiment: {e}")
        signals = None

    is_available = bool(
        signals
        and getattr(signals, "available", False)
        and isinstance(signals.sentiment, dict)
        and signals.sentiment
    )
    if not is_available:
        result = SentimentResponse(
            success=False,
            symbol=symbol.upper(),
            aggregated_sentiment={
                "available": False,
                "reason": "No verified sentiment observation is available",
                "analysis_period_days": days
            },
            sources_used=[],
            metadata={"timestamp": datetime.now().isoformat(), "data_quality": "unavailable"}
        )
        cache_set(ml_cache, cache_key, result)
        return result

    result = SentimentResponse(
        success=True,
        symbol=symbol.upper(),
        aggregated_sentiment={
            "available": True,
            **signals.sentiment,
            "analysis_period_days": days
        },
        sources_used=list(signals.sources_used),
        metadata={
            "timestamp": datetime.now().isoformat(),
            "data_quality": "reported_by_governance",
            "last_updated": datetime.now().isoformat()
        }
    )

    cache_set(ml_cache, cache_key, result)
    return result


# ===== CORRELATION ENDPOINT =====

@router.get("/correlation/matrix/current")
@handle_api_errors(fallback={"assets": [], "correlations": {}, "market_metrics": {}})
async def alias_correlation_matrix(
    user: str = Depends(get_required_user),
    source: str = Query(..., description="Explicit portfolio data source"),
    window_days: int = Query(30)
) -> dict:
    """Alias routed to risk correlation endpoint logic."""
    from api.unified_data import get_unified_filtered_balances
    from services.risk_management import risk_manager

    balances_response = await get_unified_filtered_balances(source=source, min_usd=1.0, user_id=user)
    balances = balances_response.get('items', [])
    corr_matrix = await risk_manager.calculate_correlation_matrix(holdings=balances, lookback_days=window_days)

    avg_corr = None
    try:
        corrs = corr_matrix.correlations or {}
        vals = []
        for a, row in corrs.items():
            if not isinstance(row, dict):
                continue
            for b, v in row.items():
                if a == b:
                    continue
                try:
                    vals.append(abs(float(v)))
                except Exception as e:
                    logger.debug(f"Failed to parse correlation value for {a}-{b}: {e}")
                    pass
        if vals:
            avg_corr = sum(vals) / len(vals)
    except Exception as e:
        logger.warning(f"Failed to calculate average correlation: {e}")
        avg_corr = None

    return success_response({
        "assets": list({row.get('symbol') for row in balances if row.get('symbol')}),
        "correlations": corr_matrix.correlations,
        "market_metrics": {
            "diversification_ratio": corr_matrix.diversification_ratio,
            "effective_assets": corr_matrix.effective_assets,
            "eigen_values": corr_matrix.eigen_values[:5],
            "average_correlation": avg_corr
        },
        "calculation_time": None
    })


# ===== HELPER FUNCTIONS =====

@handle_service_errors(silent=False, default_return={})
async def _get_multi_horizon_predictions(assets: List[str], horizons: List[int], include_confidence: bool = False) -> Dict[str, Any]:
    """
    Obtenir des prédictions multi-horizon pour les assets spécifiés
    """
    multi_horizon_results = {}

    for symbol in assets:
        symbol_predictions = {}

        for horizon in horizons:
            symbol_predictions[f"horizon_{horizon}d"] = {
                "available": False,
                "volatility": None,
                "expected_return": None,
                "confidence": None if include_confidence else None,
                "prediction_interval": None if include_confidence else None,
                "horizon_days": horizon,
                "reason": "No verified multi-horizon model inference is connected"
            }

        multi_horizon_results[symbol] = symbol_predictions

    return multi_horizon_results


@handle_service_errors(silent=False, default_return=None)
async def _add_confidence_metrics(predictions: Dict[str, Any], assets: List[str]) -> Dict[str, Any]:
    """
    Ajouter des métriques de confiance aux prédictions existantes
    """
    enhanced_predictions = predictions.copy()

    for symbol in assets:
        if symbol in enhanced_predictions:
            confidence_metrics = {
                "available": False,
                "model_confidence": None,
                "data_quality_score": None,
                "prediction_stability": None,
                "market_condition_factor": None,
                "overall_confidence": None,
                "reason": "Prediction confidence has not been calibrated out of sample"
            }

            if isinstance(enhanced_predictions[symbol], dict):
                enhanced_predictions[symbol]["confidence_metrics"] = confidence_metrics
            else:
                enhanced_predictions[symbol] = {
                    "base_prediction": enhanced_predictions[symbol],
                    "confidence_metrics": confidence_metrics
                }

    return enhanced_predictions
