"""
Services d'alertes prédictives pour le Decision Engine

Ce module implémente un système d'alertes discipliné qui:
- Consomme les signaux ML existants sans créer de 4ème logique
- Respecte l'architecture single-writer (propose, n'exécute pas)
- Intègre anti-bruit robuste (hystérésis, rate-limit, dedup)
- Supporte escalade automatique et snooze intelligent

Refactoré en modules (Fév 2026):
- phase_context.py: PhaseSnapshot, PhaseAwareContext
- metrics.py: AlertMetrics
- evaluators/risk_evaluator.py: AdvancedRiskEvaluator
"""

from .alert_engine import AlertEngine
from .alert_types import AlertType, AlertSeverity, Alert, AlertEvaluator, AlertRule
from .alert_storage import AlertStorage
from .phase_context import PhaseSnapshot, PhaseAwareContext
from .metrics import AlertMetrics
from .evaluators.risk_evaluator import AdvancedRiskEvaluator

__all__ = [
    # Main engine
    "AlertEngine",
    "AlertStorage",
    # Types
    "AlertType",
    "AlertSeverity",
    "Alert",
    "AlertEvaluator",
    "AlertRule",
    # Phase context (refactoré)
    "PhaseSnapshot",
    "PhaseAwareContext",
    # Metrics (refactoré)
    "AlertMetrics",
    # Evaluators (refactoré)
    "AdvancedRiskEvaluator",
]