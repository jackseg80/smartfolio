"""
ML Registry Endpoints - Gestion du registre des modèles

Ce module gère:
- Liste des modèles enregistrés
- Information détaillée des modèles
- Gestion des versions et statuts

Extrait de unified_ml_endpoints.py pour modularité (Fév 2026).
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Optional, Any
import logging

from services.ml.model_registry import get_model_registry, ModelStatus
from shared.error_handlers import handle_api_errors

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ML Registry"])


@router.get("/registry/models")
@handle_api_errors(fallback={"models": [], "total": 0})
async def list_registered_models(model_type: Optional[str] = None):
    """
    Lister les modèles enregistrés dans le registry
    """
    registry = get_model_registry()
    models = registry.list_models(model_type=model_type)

    return {
        "success": True,
        "models": models,
        "total": len(models)
    }


@router.get("/registry/models/{model_name}")
@handle_api_errors(fallback={"manifest": {}}, reraise_http_errors=True)
async def get_model_info(model_name: str, version: Optional[str] = None):
    """
    Obtenir les informations détaillées d'un modèle
    """
    registry = get_model_registry()
    manifest = registry.get_manifest(model_name, version)

    return {
        "success": True,
        "manifest": manifest.to_dict()
    }


@router.get("/registry/models/{model_name}/versions")
@handle_api_errors(fallback={"versions": [], "total_versions": 0}, reraise_http_errors=True)
async def get_model_versions_registry(model_name: str):
    """
    Obtenir toutes les versions d'un modèle depuis le registry
    """
    registry = get_model_registry()

    if model_name not in registry.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    versions_info = []
    for version, manifest in registry.models[model_name].items():
        versions_info.append({
            "version": version,
            "status": manifest.status,
            "created_at": manifest.created_at,
            "model_type": manifest.model_type,
            "file_size": manifest.file_size,
            "validation_metrics": manifest.validation_metrics,
            "tags": manifest.tags
        })

    versions_info.sort(key=lambda x: x['created_at'], reverse=True)

    return {
        "success": True,
        "model_name": model_name,
        "versions": versions_info,
        "total_versions": len(versions_info),
        "latest_version": registry.get_latest_version(model_name)
    }


@router.post("/registry/models/{model_name}/versions/{version}/status")
@handle_api_errors(fallback={"message": "Failed to update status"}, reraise_http_errors=True)
async def update_model_status(
    model_name: str,
    version: str,
    status: ModelStatus,
    reason: Optional[str] = Body(None)
):
    """
    Mettre à jour le statut d'un modèle
    """
    registry = get_model_registry()

    if status == ModelStatus.DEPRECATED:
        registry.deprecate_model(model_name, version, reason)
    else:
        registry.update_status(model_name, version, status)

    return {
        "success": True,
        "model": model_name,
        "version": version,
        "new_status": status,
        "message": f"Status updated to {status}"
    }


@router.post("/registry/models/{model_name}/versions/{version}/metrics")
@handle_api_errors(fallback={"message": "Failed to update metrics"}, reraise_http_errors=True)
async def update_model_performance_metrics(
    model_name: str,
    version: str,
    validation_metrics: Optional[Dict[str, float]] = Body(None),
    test_metrics: Optional[Dict[str, float]] = Body(None)
):
    """
    Mettre à jour les métriques de performance d'un modèle
    """
    registry = get_model_registry()
    registry.update_metrics(model_name, version, validation_metrics, test_metrics)

    return {
        "success": True,
        "model": model_name,
        "version": version,
        "message": "Performance metrics updated"
    }
