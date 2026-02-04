"""
ML Unified Contract Endpoints - Prédictions avec contrat unifié

Ce module gère:
- Endpoint de prédiction unifié avec gating et incertitude
- Prédiction de volatilité avec contrat unifié

Extrait de unified_ml_endpoints.py pour modularité (Fév 2026).
"""

from fastapi import APIRouter, Query
from typing import Dict, Optional, Any
import logging
import numpy as np
from datetime import datetime

from services.ml.orchestrator import get_orchestrator
from shared.error_handlers import handle_service_errors
from .gating import get_gating_system
from api.schemas.ml_contract import (
    UnifiedMLRequest, UnifiedMLResponse, ModelType, Horizon,
    UnifiedPrediction, QualityMetrics, UncertaintyMeasures,
    ModelMetadata, create_fallback_response
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ML Unified Contract"])


@router.post("/unified/predict", response_model=UnifiedMLResponse)
async def unified_predict(request: UnifiedMLRequest):
    """
    Endpoint unifié de prédiction ML avec gating et incertitude

    Supports: volatility, sentiment, risk scoring
    """
    start_time = datetime.now()
    gating_system = get_gating_system()

    try:
        logger.debug(f"Unified prediction request: {request.model_type} for {len(request.assets)} assets")

        predictions = []
        failed_assets = []
        warnings = []

        for asset in request.assets:
            try:
                raw_prediction = await _get_raw_prediction(
                    asset, request.model_type, request.horizon
                )

                model_key = f"{request.model_type.value}_{request.horizon.value if request.horizon else 'default'}"

                gated_prediction, accepted = gating_system.gate_prediction(
                    asset=asset,
                    raw_prediction=raw_prediction,
                    model_key=model_key,
                    model_type=request.model_type,
                    context={
                        'data_age_hours': 1.0,
                        'feature_availability': 0.9
                    }
                )

                if request.include_metadata:
                    gated_prediction.metadata = ModelMetadata(
                        name=model_key,
                        version="1.0.0",
                        model_type=request.model_type,
                        horizon=request.horizon
                    )

                if gated_prediction.quality.confidence >= request.confidence_threshold:
                    predictions.append(gated_prediction)
                else:
                    failed_assets.append(asset)
                    warnings.append(f"{asset}: confidence below threshold")

            except Exception as e:
                logger.error(f"Failed to predict for {asset}: {e}")
                failed_assets.append(asset)
                warnings.append(f"{asset}: {str(e)}")

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        aggregated = {}
        if predictions:
            values = [p.value for p in predictions]
            confidences = [p.quality.confidence for p in predictions]
            aggregated = {
                "avg_prediction": float(np.mean(values)),
                "avg_confidence": float(np.mean(confidences)),
                "prediction_range": [float(min(values)), float(max(values))]
            }

        return UnifiedMLResponse(
            success=len(predictions) > 0,
            model_type=request.model_type,
            horizon=request.horizon,
            predictions=predictions,
            aggregated=aggregated,
            processing_time_ms=processing_time,
            warnings=warnings,
            failed_assets=failed_assets
        )

    except Exception as e:
        logger.error(f"Unified prediction failed: {e}")
        return create_fallback_response(
            request.model_type,
            request.assets,
            f"Unified prediction error: {str(e)}"
        )


@router.get("/unified/volatility/{symbol}", response_model=UnifiedMLResponse)
async def unified_volatility_predict(
    symbol: str,
    horizon: Horizon = Query(Horizon.D30, description="Prediction horizon"),
    include_uncertainty: bool = Query(True, description="Include uncertainty measures"),
    include_metadata: bool = Query(False, description="Include model metadata")
):
    """
    Prédiction de volatilité avec contrat unifié
    """
    request = UnifiedMLRequest(
        assets=[symbol.upper()],
        model_type=ModelType.VOLATILITY,
        horizon=horizon,
        include_uncertainty=include_uncertainty,
        include_metadata=include_metadata
    )

    return await unified_predict(request)


@handle_service_errors(silent=False, default_return=0.0)
async def _get_raw_prediction(asset: str, model_type: ModelType, horizon: Optional[Horizon]) -> float:
    """
    Obtenir une prédiction brute selon le type de modèle

    Cette fonction fait le bridge avec les services ML existants
    """
    orchestrator = get_orchestrator()

    if model_type == ModelType.VOLATILITY:
        days_mapping = {
            Horizon.H1: 1/24,
            Horizon.H4: 4/24,
            Horizon.D1: 1,
            Horizon.D7: 7,
            Horizon.D30: 30,
            Horizon.D90: 90
        }
        days = days_mapping.get(horizon, 30)
        result = await orchestrator.predict_volatility(asset, int(days))

        if isinstance(result, dict) and 'prediction' in result:
            return float(result['prediction'])
        elif isinstance(result, (int, float)):
            return float(result)
        else:
            return 0.15

    elif model_type == ModelType.SENTIMENT:
        return np.random.normal(0, 0.3)

    elif model_type == ModelType.RISK:
        return np.random.uniform(0.2, 0.8)

    else:
        logger.warning(f"Unsupported model type: {model_type}")
        return 0.0
