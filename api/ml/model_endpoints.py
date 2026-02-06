"""
ML Model Endpoints - Gestion statut et chargement des modèles

Ce module gère:
- Statut du pipeline ML
- Chargement/déchargement des modèles
- Information sur les modèles chargés
- Préchargement des modèles prioritaires

Extrait de unified_ml_endpoints.py pour modularité (Fév 2026).
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Header
from typing import List, Optional
import logging
import os
from datetime import datetime

from services.ml_pipeline_manager_optimized import optimized_pipeline_manager as pipeline_manager
from shared.error_handlers import handle_api_errors, handle_service_errors
from .cache_utils import get_ml_cache, cache_get, cache_set, cache_clear_expired

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ML Models"])


# ===== STATUS ENDPOINTS =====

@router.get("/status")
@handle_api_errors(
    fallback={
        "pipeline_status": {
            "pipeline_initialized": False,
            "models_base_path": "models",
            "volatility_models": {"models_count": 0, "models_loaded": 0, "last_updated": None},
            "regime_models": {"model_exists": False, "model_loaded": False, "last_updated": None},
            "loaded_models_count": 0,
            "total_models_count": 0,
            "loading_mode": "fallback"
        }
    },
    include_traceback=True
)
async def get_unified_pipeline_status():
    """
    Obtenir le statut complet du pipeline ML unifié
    """
    status = pipeline_manager.get_pipeline_status()
    return {
        "success": True,
        "pipeline_status": status,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/models/status")
async def alias_models_status():
    """Alias of /api/ml/status for legacy front-ends."""
    return await get_unified_pipeline_status()


@router.get("/volatility/models/status")
async def alias_volatility_models_status():
    """Expose a volatility-focused status for legacy widgets."""
    base = await get_unified_pipeline_status()
    ps = base.get("pipeline_status", {}) if isinstance(base, dict) else {}
    return {
        "success": True,
        "pipeline_trained": bool(ps.get("loaded_models_count", 0) > 0) or bool((ps.get("regime_models") or {}).get("model_loaded")),
        "volatility_models": ps.get("volatility_models", {}),
        "regime_models": ps.get("regime_models", {}),
        "timestamp": base.get("timestamp") if isinstance(base, dict) else None,
    }


# ===== LOADING ENDPOINTS =====

@router.post("/models/load-volatility")
@handle_api_errors(
    fallback={"loaded_models": 0, "total_attempted": 0, "results": {}}
)
async def load_volatility_models(
    symbols: Optional[List[str]] = Query(None, description="Specific symbols to load (None = all)"),
    background_tasks: BackgroundTasks = None
):
    """
    Charger les modèles de volatilité
    """
    ml_cache = get_ml_cache()

    if symbols:
        results = {}
        for symbol in symbols:
            results[symbol] = pipeline_manager.load_volatility_model(symbol)
    else:
        if background_tasks:
            background_tasks.add_task(_load_all_volatility_background)
            return {
                "success": True,
                "message": "Loading all volatility models in background",
                "estimated_duration_minutes": 2
            }
        else:
            results = pipeline_manager.load_all_volatility_models()

    loaded_count = sum(1 for success in results.values() if success)

    # Invalider le cache de statut
    if "pipeline_status" in ml_cache:
        del ml_cache["pipeline_status"]

    return {
        "success": True,
        "loaded_models": loaded_count,
        "total_attempted": len(results),
        "results": results
    }


@handle_service_errors(silent=True, default_return=None)
async def _load_all_volatility_background():
    """Tâche en arrière-plan pour charger tous les modèles de volatilité"""
    results = pipeline_manager.load_all_volatility_models()
    logger.info(f"Background loading completed: {results}")


@router.post("/models/load-regime")
@handle_api_errors(fallback={"message": "Failed to load regime model"})
async def load_regime_model():
    """
    Charger le modèle de détection de régime
    """
    ml_cache = get_ml_cache()
    success = pipeline_manager.load_regime_model()

    if "pipeline_status" in ml_cache:
        del ml_cache["pipeline_status"]

    if success:
        return {
            "success": True,
            "message": "Regime detection model loaded successfully"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail="Regime model files not found or incomplete"
        )


@router.post("/models/preload")
@handle_api_errors(fallback={"preload_results": {}, "loaded_models": 0, "total_requested": 0})
async def preload_priority_models(
    symbols: List[str] = Query(default=["BTC", "ETH"], description="Priority symbols to preload")
):
    """
    Précharger des modèles prioritaires (BTC, ETH par défaut)
    """
    results = {}

    # Charger le modèle de régime (toujours utile)
    results["regime"] = pipeline_manager.load_regime_model()

    # Charger les modèles de volatilité prioritaires
    for symbol in symbols:
        results[f"volatility_{symbol}"] = pipeline_manager.load_volatility_model(symbol)

    loaded_count = sum(1 for success in results.values() if success)
    total_requested = len(results)

    return {
        "success": True,
        "preload_results": results,
        "loaded_models": loaded_count,
        "total_requested": total_requested,
        "message": f"Preloaded {loaded_count}/{total_requested} priority models",
        "timestamp": datetime.now().isoformat()
    }


# ===== MODEL INFO ENDPOINTS =====

@router.get("/models/loaded")
@handle_api_errors(fallback={"loaded_models": {}})
async def get_loaded_models():
    """
    Obtenir la liste des modèles actuellement chargés
    """
    ml_cache = get_ml_cache()
    cache_key = "loaded_models"
    cached_result = cache_get(ml_cache, cache_key, 30)
    if cached_result:
        logger.info("Returning cached loaded models")
        return cached_result

    summary = pipeline_manager.get_loaded_models_summary()

    result = {
        "success": True,
        "loaded_models": summary
    }

    cache_set(ml_cache, cache_key, result)
    cache_clear_expired(ml_cache, 30)

    return result


@router.get("/models/{model_key}/info")
@handle_api_errors(fallback={"model_info": {}}, reraise_http_errors=True)
async def get_model_info(model_key: str):
    """
    Obtenir les informations détaillées d'un modèle chargé
    """
    model_info = pipeline_manager.get_model(model_key)

    if not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_key} not loaded"
        )

    safe_info = {
        "type": model_info.get("type"),
        "loaded_at": model_info.get("loaded_at"),
        "metadata": model_info.get("metadata", {}),
        "has_scaler": "scaler" in model_info,
        "has_features": "features" in model_info
    }

    return {
        "success": True,
        "model_key": model_key,
        "model_info": safe_info
    }


@router.get("/models/loading-status")
@handle_api_errors(fallback={"loading_status": {}, "models_in_cache": 0, "cache_memory_usage": 0})
async def get_models_loading_status():
    """
    Obtenir le statut de chargement en temps réel des modèles
    """
    loading_status = getattr(pipeline_manager, 'loading_status', {})
    cache_stats = pipeline_manager.get_cache_stats()

    return {
        "success": True,
        "loading_status": loading_status,
        "models_in_cache": cache_stats.get("cached_models", 0),
        "cache_memory_usage": cache_stats.get("total_size_mb", 0),
        "timestamp": datetime.now().isoformat()
    }


# ===== UNLOAD ENDPOINTS =====

@router.delete("/models/{model_key}")
@handle_api_errors(fallback={"message": "Failed to unload model"})
async def unload_model(model_key: str):
    """
    Décharger un modèle spécifique de la mémoire
    """
    ml_cache = get_ml_cache()
    success = pipeline_manager.unload_model(model_key)

    cache_keys_to_clear = ["pipeline_status", "loaded_models"]
    for key in cache_keys_to_clear:
        if key in ml_cache:
            del ml_cache[key]

    if success:
        return {
            "success": True,
            "message": f"Model {model_key} unloaded successfully"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_key} not found in loaded models"
        )


@router.delete("/models/clear-all")
@handle_api_errors(fallback={"cleared_count": 0})
async def clear_all_models():
    """
    Décharger tous les modèles de la mémoire
    """
    ml_cache = get_ml_cache()
    count = pipeline_manager.clear_all_models()
    ml_cache.clear()

    return {
        "success": True,
        "message": f"Cleared {count} models from memory",
        "cleared_count": count
    }


# ===== PERFORMANCE ENDPOINTS =====

@router.get("/performance/summary")
@handle_api_errors(fallback={"summary": {}})
async def get_performance_summary():
    """
    Obtenir un résumé des performances des modèles
    """
    ml_cache = get_ml_cache()
    cache_key = "performance_summary"
    cached_result = cache_get(ml_cache, cache_key, 300)
    if cached_result:
        logger.info("Returning cached performance summary")
        return cached_result

    loaded_models = pipeline_manager.get_loaded_models_summary()

    performance_data = {}
    for model_key in pipeline_manager.loaded_models.keys():
        perf = pipeline_manager.get_model_performance(model_key)
        if perf:
            performance_data[model_key] = perf

    result = {
        "success": True,
        "summary": {
            "total_loaded_models": loaded_models["total_loaded"],
            "models_by_type": loaded_models["by_type"],
            "performance_data_available": len(performance_data),
            "performance_metrics": performance_data
        },
        "timestamp": datetime.now().isoformat()
    }

    cache_set(ml_cache, cache_key, result)
    cache_clear_expired(ml_cache, 300)

    return result


# ===== DEBUG ENDPOINTS =====

@router.get("/debug/pipeline-info")
async def debug_pipeline_info(x_admin_key: str = Header(None)):
    """
    Endpoint de debug pour analyser les instances du pipeline manager

    Returns basic info for unauthenticated requests, detailed info for admin.
    """
    try:
        is_admin = False
        if x_admin_key:
            expected_key = os.getenv("ADMIN_KEY")
            if expected_key and x_admin_key == expected_key:
                is_admin = True

        full_status = pipeline_manager.get_pipeline_status()

        basic_info = {
            "loaded_models_count": full_status.get('loaded_models_count', 0),
            "loading_mode": full_status.get('loading_mode', 'unknown'),
            "pipeline_initialized": full_status.get('pipeline_initialized', False),
            "admin_access": is_admin
        }

        if is_admin:
            pm_id = id(pipeline_manager)
            pm_type = type(pipeline_manager).__name__
            pm_module = pipeline_manager.__class__.__module__
            cache_size = len(pipeline_manager.model_cache.cache)
            cache_keys = list(pipeline_manager.model_cache.cache.keys())

            basic_info.update({
                "pipeline_manager_id": pm_id,
                "pipeline_manager_type": pm_type,
                "pipeline_manager_module": pm_module,
                "cache_size": cache_size,
                "cache_keys": cache_keys,
                "full_status_keys": list(full_status.keys())
            })

        return {
            "debug_info": basic_info,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in pipeline-info endpoint: {e}")
        return {
            "debug_info": {
                "error": str(e),
                "admin_access": False,
                "pipeline_initialized": False
            },
            "timestamp": datetime.now().isoformat()
        }
