"""
Policy Engine Module - Dérivation des politiques d'exécution

Ce module gère:
- Modèle Policy (structure de la politique d'exécution)
- Dérivation de policy depuis les signaux ML
- Logique cap/mode depuis UnifiedInsights
- Smoothing et garde-fous

Phase 1A: Avec hystérésis + smoothing pour éviter oscillations
"""

from typing import Dict, Any, Optional, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from .signals import MLSignals

logger = logging.getLogger(__name__)

# Type pour le mode d'exécution
ExecMode = Literal["Freeze", "Slow", "Normal", "Aggressive"]


class Policy(BaseModel):
    """Politique d'exécution dérivée des signaux ML + gouvernance"""
    mode: ExecMode = Field(default="Normal", description="Mode d'exécution")
    cap_daily: float = Field(default=0.08, ge=0.01, le=0.50, description="Cap quotidien [1-50%]")
    ramp_hours: int = Field(default=12, ge=1, le=72, description="Ramping sur N heures")
    min_trade: float = Field(default=100.0, ge=10.0, description="Trade minimum en USD")
    slippage_limit_bps: int = Field(default=50, ge=1, le=500, description="Limite slippage [1-500 bps]")

    # TTL vs Cooldown separation (critique essentielle)
    signals_ttl_seconds: int = Field(default=3600, ge=60, le=7200, description="TTL des signaux ML [1min-2h] - optimized to 1h")
    plan_cooldown_hours: int = Field(default=24, ge=1, le=168, description="Cooldown publication plans [1-168h]")

    # No-trade zone et coûts
    no_trade_threshold_pct: float = Field(default=0.02, ge=0.0, le=0.10, description="Zone no-trade [0-10%]")
    execution_cost_bps: int = Field(default=15, ge=0, le=100, description="Cout d'execution estime [0-100 bps]")

    notes: Optional[str] = Field(default=None, description="Notes explicatives")


class PolicyEngine:
    """
    Moteur de dérivation des politiques d'exécution

    Responsabilités:
    - Dérive la policy depuis les signaux ML
    - Applique smoothing et garde-fous
    - Gère les caps et modes d'exécution
    """

    def __init__(self, signals_ttl_seconds: int = 3600, plan_cooldown_hours: int = 24):
        self._signals_ttl_seconds = signals_ttl_seconds
        self._plan_cooldown_hours = plan_cooldown_hours

        # Cap stability variables (hystérésis + smoothing)
        self._last_cap = 0.08  # Dernière cap calculée pour smoothing
        self._prudent_mode = False  # État hystérésis prudent/normal

    def enforce_policy_bounds(self, policy: Policy) -> Policy:
        """Clamp defensif des champs critiques de policy."""
        data = policy.dict()

        def _as_float(value, default):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _as_int(value, default):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        cap_value = _as_float(data.get("cap_daily", 0.08), 0.08)
        data["cap_daily"] = max(0.01, min(0.20, cap_value))

        no_trade_value = _as_float(data.get("no_trade_threshold_pct", 0.02), 0.02)
        data["no_trade_threshold_pct"] = max(0.0, min(0.10, no_trade_value))

        cost_value = _as_int(data.get("execution_cost_bps", 15), 15)
        data["execution_cost_bps"] = max(0, min(100, cost_value))

        return Policy(**data)

    def derive_execution_policy(
        self,
        signals: MLSignals,
        governance_mode: str,
        manual_policy: Optional[Policy] = None,
        alert_cap_reduction: float = 0.0,
        var_state: str = "normal",
        stale_state: str = "normal",
        signals_age: float = 0.0
    ) -> Policy:
        """
        Dérive la politique d'exécution depuis les signaux ML
        Extrait la logique cap/mode depuis UnifiedInsights
        Phase 1A: Avec hystérésis + smoothing pour éviter oscillations

        Args:
            signals: MLSignals actuels
            governance_mode: Mode de gouvernance actuel
            manual_policy: Policy manuelle si applicable
            alert_cap_reduction: Réduction cap par AlertEngine
            var_state: État hystérésis VaR ("normal"/"prudent")
            stale_state: État hystérésis staleness ("normal"/"stale")
            signals_age: Âge des signaux en secondes

        Returns:
            Policy dérivée
        """
        try:
            contradiction = signals.contradiction_index
            confidence = signals.confidence

            # Override manuel si applicable
            if governance_mode == "manual" and manual_policy is not None:
                enforced_policy = self.enforce_policy_bounds(manual_policy)
                self._last_cap = enforced_policy.cap_daily
                logger.info("[governance] manual policy override in effect (mode=%s, cap=%.2f%%)",
                          enforced_policy.mode, enforced_policy.cap_daily * 100)
                return enforced_policy

            # Phase 1A: Hystérésis pour éviter flip-flop mode prudent/normal
            # Prudent si contradiction ≥ 0.45, Normal si contradiction ≤ 0.40
            if contradiction >= 0.45:
                self._prudent_mode = True
            elif contradiction <= 0.40:
                self._prudent_mode = False
            # Entre 0.40-0.45 : conserver l'état précédent (hystérésis)

            # Logique extraite d'UnifiedInsights avec hystérésis appliquée
            if contradiction > 0.7 or confidence < 0.3:
                # Mode défensif
                mode = "Freeze" if contradiction > 0.8 else "Slow"
                cap_raw = max(0.03, 0.12 - contradiction * 0.09)  # 3-12% inversé
                ramp_hours = 48

            elif self._prudent_mode or confidence < 0.6:  # Utilise hystérésis
                # Mode prudent (avec hystérésis)
                mode = "Slow"
                cap_raw = 0.07  # 7% comme dans UnifiedInsights "Rotate"
                ramp_hours = 24

            elif confidence > 0.8 and contradiction < 0.2:
                # Mode agressif
                mode = "Aggressive"
                cap_raw = 0.12  # 12% comme dans UnifiedInsights "Deploy"
                ramp_hours = 6

            else:
                # Mode normal
                mode = "Normal"
                cap_raw = 0.08  # 8% baseline
                ramp_hours = 12

            # Phase 1A: Dead zone anti-oscillation (ignore micro-variations <0.5%)
            DEAD_ZONE_BPS = 0.005  # 50 bps = 0.5%
            if abs(cap_raw - self._last_cap) < DEAD_ZONE_BPS:
                cap_raw = self._last_cap  # Keep stable, ignore noise

            # Phase 1A: Smoothing cap with increased stability (80/20 instead of 70/30)
            cap_smoothed = 0.80 * self._last_cap + 0.20 * cap_raw

            # Garde-fou : pas de variation > 2 pts entre runs (sauf stale/error)
            max_variation = 0.02  # 2 points de pourcentage
            if abs(cap_smoothed - self._last_cap) > max_variation:
                if cap_smoothed > self._last_cap:
                    cap_smoothed = self._last_cap + max_variation
                else:
                    cap_smoothed = self._last_cap - max_variation

            cap = cap_smoothed

            # Garde-fou: ne jamais 'Aggressive' si blended < 70 (si disponible)
            try:
                if mode == "Aggressive":
                    bscore = getattr(signals, 'blended_score', None)
                    if isinstance(bscore, (int, float)) and bscore < 70:
                        mode = "Normal"
                        cap = min(cap, 0.08)
                        ramp_hours = max(ramp_hours, 12)
            except (AttributeError, TypeError) as e:
                logger.debug(f"Blended score not available for aggressive mode check: {e}")

            # Ajustements selon governance mode
            if governance_mode == "freeze":
                mode = "Freeze"
                cap = 0.01

            # Phase 4: Ordre de priorité caps avec hystérésis: error(5%) > stale(8%) > alert > engine
            cap_engine = cap  # Cap calculé par engine
            # FIX Oct 2025: Alert reduction with floor 3% to prevent spiral down to 1%
            cap_alert = max(0.03, cap_engine - alert_cap_reduction)  # Floor 3% même si alert active
            cap_stale = None
            cap_error = None

            # Appliquer caps basées sur états d'hystérésis (plus stables)
            if stale_state == "stale":
                cap_stale = 0.08  # 8% stale clamp avec hystérésis

            # Error condition : toujours prioritaire, sans hystérésis (urgence)
            if signals_age > 7200:  # 2h = error critique immédiat
                cap_error = 0.05  # 5% error clamp

            # Modifier mode selon hystérésis VaR
            if var_state == "prudent" and mode == "Aggressive":
                mode = "Normal"  # Downgrade si VaR élevé persistant
                cap = min(cap, 0.08)  # Plafonner cap

            # Appliquer la priorité stricte
            if cap_error is not None:
                cap = cap_error
                mode = "Freeze"  # Error force freeze
            elif cap_stale is not None:
                cap = min(cap, cap_stale)
            elif alert_cap_reduction > 0:
                cap = cap_alert

            # Ajuster no-trade zone et coûts selon la volatilité
            vol_signals = signals.volatility
            avg_volatility = sum(vol_signals.values()) / len(vol_signals) if vol_signals else 0.15

            # No-trade zone plus large si volatilité élevée (évite le churning)
            no_trade_threshold = min(0.10, 0.02 + avg_volatility * 0.5)  # 2-10% selon volatilité

            # Coûts d'exécution estimés (spread + slippage + frais)
            execution_cost = 15 + (avg_volatility * 100)  # 15-30 bps selon volatilité

            # Enrichir les notes avec les informations de caps et hystérésis
            cap_notes = []
            if cap_error is not None:
                cap_notes.append("ERROR_CLAMP(5%)")
            elif cap_stale is not None:
                cap_notes.append("STALE_HYSTERESIS(8%)")
            elif alert_cap_reduction > 0:
                cap_notes.append(f"ALERT_REDUCTION(-{alert_cap_reduction:.1%})")

            # Phase 4: Notes d'hystérésis avancées
            hysteresis_notes = []
            if var_state == "prudent":
                hysteresis_notes.append("VAR_HYSTERESIS")
            if stale_state == "stale":
                hysteresis_notes.append("STALE_HYSTERESIS")

            if hysteresis_notes:
                cap_notes.extend(hysteresis_notes)

            # Legacy support (pour compatibilité UI)
            if self._prudent_mode:
                cap_notes.append("HYSTERESIS_PRUDENT_LEGACY")

            cap_info = f" [{', '.join(cap_notes)}]" if cap_notes else ""

            cap = max(0.01, min(0.20, cap))
            no_trade_threshold = max(0.0, min(0.10, no_trade_threshold))
            execution_cost_bps = int(max(0, min(100, round(execution_cost))))

            # Mettre a jour _last_cap pour le prochain smoothing (mais seulement si pas stale/error/alert)
            # FIX Oct 2025: Ne pas updater si alert active → évite spiral down
            if cap_error is None and cap_stale is None and alert_cap_reduction == 0:
                self._last_cap = cap

            policy = Policy(
                mode=mode,
                cap_daily=cap,
                ramp_hours=ramp_hours,
                min_trade=100.0,
                slippage_limit_bps=50,
                signals_ttl_seconds=self._signals_ttl_seconds,
                plan_cooldown_hours=self._plan_cooldown_hours,
                no_trade_threshold_pct=no_trade_threshold,
                execution_cost_bps=execution_cost_bps,
                notes=f"ML: contradiction={contradiction:.2f}, confidence={confidence:.2f}, vol={avg_volatility:.3f}{cap_info}"
            )

            # Logging enrichi Phase 1A avec détails complets pour debug oscillation
            logger.info(
                f"[CAP_FLOW] contradiction={contradiction:.3f}, confidence={confidence:.3f}, "
                f"cap_raw={cap_raw:.4f} ({cap_raw*100:.2f}%), cap_smoothed={cap_smoothed:.4f} ({cap_smoothed*100:.2f}%), "
                f"cap_engine={cap_engine:.4f} ({cap_engine*100:.2f}%), "
                f"cap_stale={'%.4f (%.2f%%)' % (cap_stale, cap_stale*100) if cap_stale else 'N/A'}, "
                f"cap_error={'%.4f (%.2f%%)' % (cap_error, cap_error*100) if cap_error else 'N/A'}, "
                f"cap_alert={cap_alert:.4f} ({cap_alert*100:.2f}%), cap_final={cap:.4f} ({cap*100:.2f}%), "
                f"var_state={var_state}, stale_state={stale_state}, "
                f"signals_age={signals_age:.0f}s, prudent_mode={self._prudent_mode}, "
                f"mode={mode}{cap_info}"
            )

            return policy

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            # Known data structure/validation errors - apply graceful fallback
            logger.warning(f"Data error deriving execution policy: {e}")
            return self.enforce_policy_bounds(Policy(mode="Freeze", cap_daily=0.08, notes=f"Error fallback: {e}"))

        except Exception as e:
            # Unexpected critical error in policy derivation - freeze with full logging
            logger.exception(f"Unexpected critical error deriving execution policy: {e}")
            return self.enforce_policy_bounds(Policy(mode="Freeze", cap_daily=0.08, notes=f"Critical error fallback: {e}"))

    def get_last_cap(self) -> float:
        """Retourne la dernière cap calculée"""
        return self._last_cap

    def set_last_cap(self, cap: float) -> None:
        """Définit la dernière cap (pour initialisation)"""
        self._last_cap = cap

    def is_prudent_mode(self) -> bool:
        """Retourne si le mode prudent est actif"""
        return self._prudent_mode
