"""
ML Training Endpoints - Entraînement des modèles ML

Ce module gère:
- Entraînement des modèles (background tasks)
- Alias pour compatibilité

Extrait de unified_ml_endpoints.py pour modularité (Fév 2026).
"""

from fastapi import APIRouter, BackgroundTasks, Depends
from api.deps import require_admin_role
from typing import List, Literal
import logging
from datetime import datetime
from pydantic import BaseModel

from services.ml.orchestrator import get_orchestrator
from shared.error_handlers import handle_api_errors, handle_service_errors
from .cache_utils import get_ml_cache
from .model_endpoints import load_regime_model

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ML Training"], dependencies=[Depends(require_admin_role)])


class TrainingRequest(BaseModel):
    """Request model for ML training"""
    assets: List[str]
    lookback_days: int = 730
    include_market_indicators: bool = True
    save_models: bool = True
    market: Literal["crypto", "stocks"] = "crypto"


@router.post("/train")
@handle_api_errors(fallback={"message": "Training failed to start", "assets": [], "background_task": False})
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> dict:
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
        request.save_models,
        request.market
    )

    # Invalider les caches de prédiction
    keys_to_remove = [k for k in ml_cache.keys() if "predictions_" in k]
    for key in keys_to_remove:
        del ml_cache[key]

    return {
        "success": True,
        "message": f"Training started for {len(request.assets)} assets",
        "assets": request.assets,
        "action": "evaluate_frozen_volatility_protocol",
        "background_task": True
    }


@handle_service_errors(silent=True, default_return=None)
async def _train_models_background(
    assets: List[str],
    lookback_days: int,
    include_market_indicators: bool,
    save_models: bool,
    market: Literal["crypto", "stocks"] = "crypto"
):
    """
    Tâche d'entraînement en arrière-plan
    """
    import asyncio
    import json
    from services.ml.risk_evaluation import evaluate
    from services.ml.reliability import capability_service
    reports = []
    for asset in assets:
        for horizon in (7, 30):
            try:
                reports.append(await asyncio.to_thread(evaluate, capability_service, market, asset.upper(), horizon, publish=save_models))
            except Exception as exc:
                reports.append({"asset": asset, "horizon": horizon, "state": "not_evaluable", "reason": str(exc), "published": False})
    destination = capability_service.root / "outputs/ml-reliability/admin-evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(reports, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("Explicit ML evaluation completed; conclusions recorded at %s", destination)


@router.post("/regime/train")
async def alias_regime_train() -> dict:
    """Alias that loads the regime model."""
    return {"success": False, "availability": "Unavailable", "reason": "Legacy training alias retired. Use the explicit administrator evaluation workflow."}
