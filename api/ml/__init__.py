"""
ML module for unified prediction processing

Ce module expose:
- Système de gating et calibration
- Sous-routers modulaires pour les endpoints ML
- Utilitaires de cache partagés

Refactoré en modules (Fév 2026).
"""

# Gating system
from .gating import get_gating_system, initialize_gating_system, GatingConfig

# Cache utilities
from .cache_utils import get_ml_cache, cache_get, cache_set, cache_clear_expired, cache_clear_all

# Routers (pour import direct si nécessaire)
from .model_endpoints import router as model_router
from .prediction_endpoints import router as prediction_router
from .training_endpoints import router as training_router
from .cache_endpoints import router as cache_router
from .monitoring_endpoints import router as monitoring_router
from .registry_endpoints import router as registry_router
from .unified_contract_endpoints import router as unified_contract_router

__all__ = [
    # Gating
    "get_gating_system",
    "initialize_gating_system",
    "GatingConfig",
    # Cache
    "get_ml_cache",
    "cache_get",
    "cache_set",
    "cache_clear_expired",
    "cache_clear_all",
    # Routers
    "model_router",
    "prediction_router",
    "training_router",
    "cache_router",
    "monitoring_router",
    "registry_router",
    "unified_contract_router",
]
