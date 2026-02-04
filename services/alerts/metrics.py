"""
Alert Metrics Module - Collecteur de métriques pour observabilité

Ce module gère:
- AlertMetrics: Collecteur de métriques interne (counters, gauges, labels)
- Métriques pour monitoring et debug

Extrait de alert_engine.py pour modularité.
"""

from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class AlertMetrics:
    """
    Collecteur de métriques pour observabilité AlertEngine

    Responsabilités:
    - Gère les compteurs (alerts émises, supprimées, etc.)
    - Gère les gauges (dernière évaluation, alertes actives, etc.)
    - Gère les labels (origine policy, etc.)
    """

    def __init__(self):
        """Initialise le collecteur de métriques."""
        self.counters = {
            "alerts_emitted_total": {},      # {type:severity: count}
            "alerts_suppressed_total": {},   # {reason: count}
            "policy_changes_total": {},      # {mode: count}
            "freeze_seconds_total": {},      # {reason: count}
            "alerts_ack_total": {},          # {user: count}
            "alerts_snoozed_total": {},      # {alert_id: count}
            "streaming_broadcasts_success": 0,
            "streaming_broadcasts_failed": 0,
            "streaming_risk_events_success": 0,
            "streaming_risk_events_failed": 0
        }

        self.gauges = {
            "last_alert_eval_ts": 0,
            "last_policy_change_ts": 0,
            "active_alerts_count": 0
        }

        self.labels = {
            "policy_origin": "manual"  # manual|alert|api
        }

    def increment(self, metric: str, labels: Dict[str, str] = None, value: int = 1):
        """
        Incrémente un compteur.

        Args:
            metric: Nom de la métrique
            labels: Labels optionnels pour la métrique
            value: Valeur à incrémenter (défaut: 1)
        """
        try:
            # Métriques simples (sans labels)
            if metric in self.counters and isinstance(self.counters[metric], int):
                self.counters[metric] += value
                return

            if labels:
                key = f"{metric}:{':'.join(f'{k}={v}' for k, v in labels.items())}"
            else:
                key = metric

            if metric not in self.counters:
                self.counters[metric] = {}

            if isinstance(self.counters[metric], dict):
                self.counters[metric][key] = self.counters[metric].get(key, 0) + value
            else:
                self.counters[metric] = value

        except Exception as e:
            logger.warning(f"Error incrementing metric {metric}: {e}")

    def set_gauge(self, metric: str, value: float):
        """
        Met à jour une gauge.

        Args:
            metric: Nom de la gauge
            value: Nouvelle valeur
        """
        try:
            self.gauges[metric] = value
        except Exception as e:
            logger.warning(f"Error setting gauge {metric}: {e}")

    def set_label(self, label: str, value: str):
        """
        Met à jour un label.

        Args:
            label: Nom du label
            value: Nouvelle valeur
        """
        try:
            self.labels[label] = value
        except Exception as e:
            logger.warning(f"Error setting label {label}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne toutes les métriques au format JSON.

        Returns:
            Dict contenant counters, gauges, labels et timestamp
        """
        return {
            "counters": self.counters,
            "gauges": self.gauges,
            "labels": self.labels,
            "timestamp": datetime.now().isoformat()
        }

    def get_counter(self, metric: str) -> int:
        """
        Retourne la valeur d'un compteur.

        Args:
            metric: Nom du compteur

        Returns:
            Valeur du compteur ou 0 si non trouvé
        """
        value = self.counters.get(metric, 0)
        if isinstance(value, dict):
            return sum(value.values())
        return value

    def get_gauge(self, metric: str) -> float:
        """
        Retourne la valeur d'une gauge.

        Args:
            metric: Nom de la gauge

        Returns:
            Valeur de la gauge ou 0 si non trouvée
        """
        return self.gauges.get(metric, 0)

    def reset_counters(self):
        """Réinitialise tous les compteurs."""
        for key in self.counters:
            if isinstance(self.counters[key], dict):
                self.counters[key] = {}
            else:
                self.counters[key] = 0
        logger.info("Alert metrics counters reset")

    def reset_all(self):
        """Réinitialise toutes les métriques."""
        self.__init__()
        logger.info("Alert metrics fully reset")
