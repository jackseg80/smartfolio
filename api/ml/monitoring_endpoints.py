"""
ML Monitoring Endpoints - Health et métriques des modèles

Ce module gère:
- Santé globale du système ML
- Métriques par modèle
- Versions des modèles

Extrait de unified_ml_endpoints.py pour modularité (Fév 2026).
"""

from fastapi import APIRouter, HTTPException, Body, Depends
from api.deps import require_admin_role
from typing import Dict, Optional, Any
import logging
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path

from fastapi.responses import JSONResponse
from services.ml_pipeline_manager_optimized import optimized_pipeline_manager as pipeline_manager
from api.utils.formatters import success_response, error_response
from shared.error_handlers import handle_api_errors
from .gating import get_gating_system
from api.schemas.ml_contract import MLSystemHealth, ModelHealth

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ML Monitoring"])


@router.get("/monitoring/health", response_model=MLSystemHealth)
async def get_ml_system_health():
    from services.ml.reliability import capability_service
    catalog = capability_service.catalog()
    return MLSystemHealth(overall_health=None, models_status=[
        ModelHealth(model_name=entry["path"], version=entry["sha256"][:16],
            is_healthy=None, last_prediction=None, avg_confidence=None, error_rate_24h=None)
        for entry in catalog],
        system_metrics={"files_present": len(catalog), "loaded_models": sum(e["loaded"] for e in catalog),
            "successful_models": sum(e["inference_succeeded"] for e in catalog),
            "active_models": sum(e["loaded"] for e in catalog),
            "healthy_models": None, "total_predictions_24h": None,
            "reason": "No performance health estimate is derived from artifact presence or confidence"})


@router.get("/metrics/{model_name}")
@handle_api_errors(fallback={"error": "Metrics unavailable"}, reraise_http_errors=True)
async def get_model_metrics(model_name: str, version: Optional[str] = None) -> dict:
    from services.ml.reliability import capability_service
    try:
        market, asset, days = model_name.rsplit("_", 2)
        artifact, path = capability_service.artifact(market, asset, int(days.rstrip("d")))
        return success_response({"model": model_name, "version": artifact["code_version"], "availability": "Available",
            "metrics": artifact["validation"]["metrics"], "validation_state": "retrospectively_validated"})
    except Exception:
        return success_response({"model": model_name, "version": None, "metrics": None, "availability": "Unavailable",
            "reason": "No verified confirmation evaluation is registered for this model"})


@router.get("/versions/{model_name}")
@handle_api_errors(fallback={"available_versions": [], "total_versions": 0})
async def get_model_versions(model_name: str) -> dict:
    """
    Lister les versions disponibles d'un modèle
    """
    metrics_file = Path("data/ml_metrics.json")

    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)

        model_data = all_metrics.get(model_name, {})
        versions = list(model_data.get("versions", {}).keys())

        return success_response({
            "model": model_name,
            "available_versions": versions,
            "total_versions": len(versions)
        })

    return success_response({
        "model": model_name,
        "available_versions": [],
        "total_versions": 0,
        "reason": "No version observation is registered"
    })


@router.post("/metrics/{model_name}/update")
@handle_api_errors(fallback={"message": "Failed to update metrics"}, reraise_http_errors=True)
async def update_model_metrics(
    model_name: str,
    version: str,
    metrics: Dict[str, Any] = Body(...),
    user: str = Depends(require_admin_role)
) -> dict:
    """
    Mettre à jour les métriques d'un modèle (version spécifique)
    """
    os.makedirs("data", exist_ok=True)
    metrics_file = Path("data/ml_metrics.json")

    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    if model_name not in all_metrics:
        all_metrics[model_name] = {"versions": {}}

    all_metrics[model_name]["versions"][version] = {
        **metrics,
        "last_updated": datetime.now().isoformat()
    }

    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    return success_response({
        "model": model_name,
        "version": version,
        "message": "Metrics updated successfully"
    })
