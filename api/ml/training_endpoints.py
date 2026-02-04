"""
ML Training Endpoints - Entraînement des modèles ML

Ce module gère:
- Entraînement des modèles (background tasks)
- Alias pour compatibilité

Extrait de unified_ml_endpoints.py pour modularité (Fév 2026).
"""

from fastapi import APIRouter, BackgroundTasks
from typing import List
import logging
from datetime import datetime
from pydantic import BaseModel

from services.ml.orchestrator import get_orchestrator
from shared.error_handlers import handle_api_errors, handle_service_errors
from .cache_utils import get_ml_cache
from .model_endpoints import load_regime_model

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ML Training"])


class TrainingRequest(BaseModel):
    """Request model for ML training"""
    assets: List[str]
    lookback_days: int = 730
    include_market_indicators: bool = True
    save_models: bool = True


@router.post("/train")
@handle_api_errors(fallback={"message": "Training failed to start", "assets": [], "background_task": False})
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Entraîner les modèles ML de manière unifiée
    """
    ml_cache = get_ml_cache()

    # Lancer l'entraînement en arrière-plan
    background_tasks.add_task(
        _train_models_background,
        request.assets,
        request.lookback_days,
        request.include_market_indicators,
        request.save_models
    )

    # Invalider les caches de prédiction
    keys_to_remove = [k for k in ml_cache.keys() if "predictions_" in k]
    for key in keys_to_remove:
        del ml_cache[key]

    return {
        "success": True,
        "message": f"Training started for {len(request.assets)} assets",
        "assets": request.assets,
        "estimated_duration_minutes": len(request.assets) * 2,
        "background_task": True
    }


@handle_service_errors(silent=True, default_return=None)
async def _train_models_background(
    assets: List[str],
    lookback_days: int,
    include_market_indicators: bool,
    save_models: bool
):
    """
    Tâche d'entraînement en arrière-plan
    """
    orchestrator = get_orchestrator()

    await orchestrator.train_models(
        assets=assets,
        lookback_days=lookback_days,
        include_market_indicators=include_market_indicators,
        save_models=save_models
    )

    logger.info(f"Background training completed for assets: {assets}")


@router.post("/regime/train")
async def alias_regime_train():
    """Alias that loads the regime model."""
    return await load_regime_model()
