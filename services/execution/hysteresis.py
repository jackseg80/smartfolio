"""
Hysteresis Module - Gestion des états d'hystérésis pour Governance Engine

Ce module gère:
- Hystérésis anti-yo-yo pour VaR et stale detection
- Réduction cap par AlertEngine
- Clear progressif des alertes

Phase 4: Hystérésis avancée pour éviter oscillations
"""

from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# Configuration par défaut de l'hystérésis
DEFAULT_HYSTERESIS_CONFIG = {
    "var_activate_threshold": 75,      # Active hystérésis si VaR > 75
    "var_deactivate_threshold": 65,    # Désactive si VaR < 65 (gap anti-yo-yo)
    "stale_activate_seconds": 3600,    # Active si stale > 1h
    "stale_deactivate_seconds": 1800,  # Désactive si fresh < 30min (gap anti-yo-yo)
    "history_window": 5,               # Fenêtre historique pour trend detection
    "trend_stability_required": 3      # Nb points stables requis avant changement
}


class HysteresisManager:
    """
    Gestionnaire d'hystérésis pour éviter les oscillations yo-yo

    Responsabilités:
    - Gestion des états VaR et staleness avec hystérésis
    - Réduction cap par AlertEngine avec cooldown
    - Clear progressif des alertes
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialise le gestionnaire d'hystérésis

        Args:
            config: Configuration optionnelle (utilise defaults si non fourni)
        """
        self.config = config or DEFAULT_HYSTERESIS_CONFIG.copy()

        # États d'hystérésis
        self._var_hysteresis_state = "normal"   # "normal" | "prudent"
        self._stale_hysteresis_state = "normal"  # "normal" | "stale"

        # Historiques pour trend detection
        self._var_hysteresis_history = []
        self._stale_hysteresis_history = []

        # Alert cap reduction
        self._alert_cap_reduction = 0.0
        self._alert_cooldown_until = datetime.min
        self._last_progressive_clear = datetime.now()

    def update_hysteresis_state(self, signals: Any, signals_age: float) -> Tuple[str, str]:
        """
        Phase 4: Hystérésis anti-yo-yo avec seuils d'activation/désactivation distincts

        Args:
            signals: Signaux ML actuels
            signals_age: Âge des signaux en secondes

        Returns:
            Tuple[var_state, stale_state] : États d'hystérésis ("normal"/"prudent", "normal"/"stale")
        """
        try:
            # 1. VaR Hysteresis - utilise blended_score comme proxy VaR
            var_proxy_raw = getattr(signals, 'blended_score', None)
            if var_proxy_raw is None:
                var_proxy = 70.0
            else:
                try:
                    var_proxy = float(var_proxy_raw)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid blended_score {var_proxy_raw!r}; falling back to 70.0")
                    var_proxy = 70.0

            try:
                signals_age = float(signals_age)
            except (TypeError, ValueError):
                logger.warning(f"Invalid signals_age {signals_age!r}; defaulting to 0.0")
                signals_age = 0.0

            self._var_hysteresis_history.append(var_proxy)

            # Maintenir fenêtre historique
            if len(self._var_hysteresis_history) > self.config["history_window"]:
                self._var_hysteresis_history.pop(0)

            # Logique d'hystérésis VaR avec gap anti-yo-yo
            if self._var_hysteresis_state == "normal":
                # Condition d'activation prudent : VaR élevé + tendance stable
                if (var_proxy > self.config["var_activate_threshold"] and
                    len(self._var_hysteresis_history) >= self.config["trend_stability_required"]):
                    # Vérifier tendance stable vers le haut
                    recent_values = self._var_hysteresis_history[-self.config["trend_stability_required"]:]
                    if all(v > self.config["var_activate_threshold"] for v in recent_values):
                        self._var_hysteresis_state = "prudent"
                        logger.warning(f"VaR hysteresis activated: prudent mode (score={var_proxy})")
            else:  # "prudent"
                # Condition de désactivation : VaR bas + tendance stable (gap anti-yo-yo)
                if (var_proxy < self.config["var_deactivate_threshold"] and
                    len(self._var_hysteresis_history) >= self.config["trend_stability_required"]):
                    # Vérifier tendance stable vers le bas
                    recent_values = self._var_hysteresis_history[-self.config["trend_stability_required"]:]
                    if all(v < self.config["var_deactivate_threshold"] for v in recent_values):
                        self._var_hysteresis_state = "normal"
                        logger.info(f"VaR hysteresis deactivated: normal mode (score={var_proxy})")

            # 2. Stale Hysteresis - utilise signals_age
            self._stale_hysteresis_history.append(signals_age)

            # Maintenir fenêtre historique
            if len(self._stale_hysteresis_history) > self.config["history_window"]:
                self._stale_hysteresis_history.pop(0)

            # Logique d'hystérésis staleness avec gap anti-yo-yo
            if self._stale_hysteresis_state == "normal":
                # Condition d'activation stale : signaux anciens + tendance stable
                if (signals_age > self.config["stale_activate_seconds"] and
                    len(self._stale_hysteresis_history) >= self.config["trend_stability_required"]):
                    # Vérifier tendance stable vers staleness
                    recent_ages = self._stale_hysteresis_history[-self.config["trend_stability_required"]:]
                    if all(age > self.config["stale_activate_seconds"] for age in recent_ages):
                        self._stale_hysteresis_state = "stale"
                        logger.warning(f"Stale hysteresis activated: stale mode (age={signals_age:.0f}s)")
            else:  # "stale"
                # Condition de désactivation : signaux frais + tendance stable (gap anti-yo-yo)
                if (signals_age < self.config["stale_deactivate_seconds"] and
                    len(self._stale_hysteresis_history) >= self.config["trend_stability_required"]):
                    # Vérifier tendance stable vers freshness
                    recent_ages = self._stale_hysteresis_history[-self.config["trend_stability_required"]:]
                    if all(age < self.config["stale_deactivate_seconds"] for age in recent_ages):
                        self._stale_hysteresis_state = "normal"
                        logger.info(f"Stale hysteresis deactivated: normal mode (age={signals_age:.0f}s)")

            return self._var_hysteresis_state, self._stale_hysteresis_state

        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"Data error in hysteresis state update: {e}")
            return "normal", "normal"  # Safe fallback
        except Exception as e:
            logger.exception(f"Unexpected error in hysteresis state update: {e}")
            return "normal", "normal"

    def apply_alert_cap_reduction(self, reduction_percentage: float, alert_id: str, reason: str) -> bool:
        """
        Phase 1B: AlertEngine peut déclencher réduction cap
        Max rule: pas d'empilement, cooldown 60min, remontée progressive

        Args:
            reduction_percentage: Réduction en pourcentage (ex: 0.03 pour -3%)
            alert_id: ID de l'alerte qui déclenche
            reason: Raison (VaR>4%, contradiction>55%, etc.)

        Returns:
            True si réduction appliquée, False sinon
        """
        try:
            # Cooldown check: ne pas réduire si déjà en cooldown
            if datetime.now() < self._alert_cooldown_until:
                logger.info(f"Alert cap reduction ignored (cooldown until {self._alert_cooldown_until})")
                return False

            # Max rule: prendre la plus forte réduction, pas additive
            new_reduction = max(self._alert_cap_reduction, reduction_percentage)

            if new_reduction > self._alert_cap_reduction:
                old_reduction = self._alert_cap_reduction
                self._alert_cap_reduction = new_reduction
                # Cooldown 60min après nouvelle réduction
                self._alert_cooldown_until = datetime.now() + timedelta(minutes=60)

                logger.warning(f"Alert cap reduction applied: {old_reduction:.1%} → {new_reduction:.1%} "
                             f"(Alert: {alert_id}, Reason: {reason})")
                return True
            else:
                logger.info(f"Alert cap reduction {reduction_percentage:.1%} ignored "
                          f"(current: {self._alert_cap_reduction:.1%} is higher)")
                return False

        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid alert cap reduction parameters: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error applying alert cap reduction: {e}")
            return False

    def clear_alert_cap_reduction(self, progressive: bool = True) -> bool:
        """
        Phase 1B: Nettoyer réduction cap AlertEngine
        Si progressive=True, remontée +1pt/30min

        Args:
            progressive: Si True, remontée progressive +1pt

        Returns:
            True si clear effectué
        """
        try:
            if self._alert_cap_reduction <= 0:
                return True  # Déjà à 0

            if progressive:
                # Remontée progressive +1pt/30min
                step = 0.01  # 1 point de pourcentage
                self._alert_cap_reduction = max(0, self._alert_cap_reduction - step)
                logger.info(f"Alert cap reduction progressive clear: {self._alert_cap_reduction:.1%} remaining")
            else:
                # Clear immédiat
                old_reduction = self._alert_cap_reduction
                self._alert_cap_reduction = 0.0
                self._alert_cooldown_until = datetime.min
                logger.info(f"Alert cap reduction cleared: {old_reduction:.1%} → 0%")

            return True

        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid parameters clearing alert cap reduction: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error clearing alert cap reduction: {e}")
            return False

    def check_progressive_clear(self, interval_seconds: int = 1800) -> bool:
        """
        Vérifie si un clear progressif est nécessaire

        Args:
            interval_seconds: Intervalle entre clears (default 30min)

        Returns:
            True si clear effectué
        """
        if self._alert_cap_reduction <= 0:
            return False

        time_since_last_clear = (datetime.now() - self._last_progressive_clear).total_seconds()
        if time_since_last_clear > interval_seconds:
            old_reduction = self._alert_cap_reduction
            self.clear_alert_cap_reduction(progressive=True)
            self._last_progressive_clear = datetime.now()
            logger.warning(
                f"Auto progressive clear: {old_reduction:.1%} → {self._alert_cap_reduction:.1%} "
                f"(next clear in {interval_seconds//60}min)"
            )
            return True

        return False

    # Getters
    def get_var_state(self) -> str:
        """Retourne l'état d'hystérésis VaR"""
        return self._var_hysteresis_state

    def get_stale_state(self) -> str:
        """Retourne l'état d'hystérésis staleness"""
        return self._stale_hysteresis_state

    def get_alert_cap_reduction(self) -> float:
        """Retourne la réduction cap actuelle"""
        return self._alert_cap_reduction

    def is_in_cooldown(self) -> bool:
        """Vérifie si en cooldown d'alerte"""
        return datetime.now() < self._alert_cooldown_until

    def get_cooldown_remaining(self) -> float:
        """Retourne le temps restant de cooldown en secondes"""
        if datetime.now() >= self._alert_cooldown_until:
            return 0.0
        return (self._alert_cooldown_until - datetime.now()).total_seconds()

    def reset(self) -> None:
        """Réinitialise tous les états d'hystérésis"""
        self._var_hysteresis_state = "normal"
        self._stale_hysteresis_state = "normal"
        self._var_hysteresis_history = []
        self._stale_hysteresis_history = []
        self._alert_cap_reduction = 0.0
        self._alert_cooldown_until = datetime.min
        self._last_progressive_clear = datetime.now()
        logger.info("Hysteresis manager reset to default state")
