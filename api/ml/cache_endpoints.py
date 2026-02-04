"""
ML Cache Endpoints - Gestion du cache et optimisation mémoire

Ce module gère:
- Statistiques du cache
- Nettoyage du cache
- Optimisation mémoire

Extrait de unified_ml_endpoints.py pour modularité (Fév 2026).
"""

from fastapi import APIRouter
from datetime import datetime
import logging

from services.ml_pipeline_manager_optimized import optimized_pipeline_manager as pipeline_manager
from shared.error_handlers import handle_api_errors
from .cache_utils import get_ml_cache, cache_clear_all

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ML Cache"])


@router.post("/cache/clear")
@handle_api_errors(fallback={"cleared_entries": 0})
async def clear_ml_cache():
    """
    Vider le cache des endpoints ML
    """
    cache_size = cache_clear_all()

    return {
        "success": True,
        "message": f"Cleared {cache_size} cache entries",
        "cleared_entries": cache_size
    }


@router.get("/cache/stats")
@handle_api_errors(fallback={"cache_stats": {}})
async def get_cache_statistics():
    """
    Obtenir les statistiques détaillées du cache optimisé
    """
    cache_stats = pipeline_manager.get_cache_stats()

    return {
        "success": True,
        "cache_stats": cache_stats,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/memory/optimize")
@handle_api_errors(fallback={"optimization_result": {}, "message": "Memory optimization failed"})
async def optimize_memory_usage():
    """
    Optimiser l'utilisation mémoire des modèles ML
    """
    optimization_result = pipeline_manager.optimize_memory()

    return {
        "success": True,
        "optimization_result": optimization_result,
        "message": f"Memory optimization completed. Freed {optimization_result.get('evicted_models', 0)} models.",
        "timestamp": datetime.now().isoformat()
    }
