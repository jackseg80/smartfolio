"""
Unified ML Pipeline API Endpoints - Orchestrateur

Ce module est l'orchestrateur principal des endpoints ML.
Il importe et expose les sous-routers modulaires:
- model_endpoints: Statut et chargement des modèles
- prediction_endpoints: Prédictions ML
- training_endpoints: Entraînement des modèles
- cache_endpoints: Gestion du cache
- monitoring_endpoints: Health et métriques
- registry_endpoints: Registre des modèles
- unified_contract_endpoints: Prédictions avec contrat unifié

Refactoré en modules (Fév 2026): 1728L -> ~50L orchestrateur.
"""

from fastapi import APIRouter
import logging

# Import des sous-routers modulaires
from api.ml.model_endpoints import router as model_router
from api.ml.prediction_endpoints import router as prediction_router
from api.ml.training_endpoints import router as training_router
from api.ml.cache_endpoints import router as cache_router
from api.ml.monitoring_endpoints import router as monitoring_router
from api.ml.registry_endpoints import router as registry_router
from api.ml.unified_contract_endpoints import router as unified_contract_router

# Re-export des modèles Pydantic pour compatibilité
from api.ml.prediction_endpoints import PredictionRequest, PredictionResponse, SentimentResponse
from api.ml.training_endpoints import TrainingRequest

logger = logging.getLogger(__name__)

# Router principal qui agrège tous les sous-routers
router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])

# Inclure tous les sous-routers
router.include_router(model_router)
router.include_router(prediction_router)
router.include_router(training_router)
router.include_router(cache_router)
router.include_router(monitoring_router)
router.include_router(registry_router)
router.include_router(unified_contract_router)

# Log de l'initialisation
logger.info("Unified ML endpoints initialized with modular routers")

# Exports pour compatibilité avec les imports existants
__all__ = [
    "router",
    "PredictionRequest",
    "PredictionResponse",
    "SentimentResponse",
    "TrainingRequest",
]
