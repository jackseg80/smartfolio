"""
Advanced Risk Evaluator Module - Évaluation des alertes de risque avancé

Ce module gère les alertes de Phase 3A:
- VAR_BREACH: VaR limite dépassée
- STRESS_TEST_FAILED: Échec stress test critique
- MONTE_CARLO_EXTREME: Scénario extrême détecté MC
- RISK_CONCENTRATION: Concentration risque excessive

Extrait de alert_engine.py pour modularité.
"""

from typing import Dict, Any, Optional, Tuple
import logging

from ..alert_types import AlertType

logger = logging.getLogger(__name__)


class AdvancedRiskEvaluator:
    """
    Évaluateur d'alertes Advanced Risk (Phase 3A)

    Responsabilités:
    - Calcul VaR et vérification limites
    - Exécution stress tests
    - Simulation Monte Carlo
    - Détection concentration risque
    """

    # Types d'alertes gérés par cet évaluateur
    SUPPORTED_ALERT_TYPES = {
        AlertType.VAR_BREACH,
        AlertType.STRESS_TEST_FAILED,
        AlertType.MONTE_CARLO_EXTREME,
        AlertType.RISK_CONCENTRATION
    }

    def __init__(self, risk_engine, config: Dict[str, Any] = None):
        """
        Initialise l'évaluateur de risque avancé.

        Args:
            risk_engine: Instance de AdvancedRiskEngine
            config: Configuration des limites et seuils
        """
        self.risk_engine = risk_engine
        self.config = config or {}

    def supports_alert_type(self, alert_type: AlertType) -> bool:
        """Vérifie si ce type d'alerte est supporté."""
        return alert_type in self.SUPPORTED_ALERT_TYPES

    async def evaluate(
        self,
        alert_type: AlertType,
        signals: Dict[str, Any],
        portfolio_weights: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Évalue une alerte de risque avancé.

        Args:
            alert_type: Type d'alerte à évaluer
            signals: Signaux ML actuels
            portfolio_weights: Poids du portfolio {symbol: weight}
            portfolio_value: Valeur totale du portfolio

        Returns:
            Tuple[should_trigger, enriched_signals, metadata]
            - should_trigger: True si l'alerte doit être déclenchée
            - enriched_signals: Signaux enrichis avec données de risque
            - metadata: Métadonnées additionnelles pour l'alerte
        """
        if not self.risk_engine:
            logger.warning("Risk engine not available for evaluation")
            return False, None, None

        if not portfolio_weights:
            logger.debug("No portfolio weights available for risk evaluation")
            return False, None, None

        try:
            if alert_type == AlertType.VAR_BREACH:
                return await self._evaluate_var_breach(signals, portfolio_weights, portfolio_value)
            elif alert_type == AlertType.STRESS_TEST_FAILED:
                return await self._evaluate_stress_test(signals, portfolio_weights, portfolio_value)
            elif alert_type == AlertType.MONTE_CARLO_EXTREME:
                return await self._evaluate_monte_carlo(signals, portfolio_weights, portfolio_value)
            elif alert_type == AlertType.RISK_CONCENTRATION:
                return await self._evaluate_concentration(signals, portfolio_weights, portfolio_value)
            else:
                logger.debug(f"Unsupported alert type for risk evaluator: {alert_type}")
                return False, None, None

        except Exception as e:
            logger.error(f"Error evaluating {alert_type.value}: {e}")
            return False, None, None

    async def _evaluate_var_breach(
        self,
        signals: Dict[str, Any],
        portfolio_weights: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Évalue si la VaR dépasse les limites."""
        try:
            from services.risk.advanced_risk_engine import VaRMethod

            var_result = self.risk_engine.calculate_var(
                portfolio_weights, portfolio_value,
                method=VaRMethod.PARAMETRIC, confidence_level=0.95
            )

            # Limites VaR (configurables)
            var_limits = self.config.get("var_limits", {
                "daily_95": 0.05,  # 5% du portfolio
                "daily_99": 0.08  # 8% du portfolio
            })

            var_limit_95 = portfolio_value * var_limits["daily_95"]
            var_breach = var_result.var_absolute > var_limit_95

            if var_breach:
                enriched_signals = signals.copy()
                enriched_signals["var_breach"] = {
                    "var_current": var_result.var_absolute,
                    "var_limit": var_limit_95,
                    "var_method": var_result.method.value,
                    "confidence_level": var_result.confidence_level,
                    "var_ratio": var_result.var_absolute / var_limit_95,
                    "horizon": var_result.horizon.value
                }

                metadata = {
                    "var_breach_severity": "critical" if var_result.var_absolute > var_limit_95 * 2 else "major",
                    "var_excess": var_result.var_absolute - var_limit_95
                }

                return True, enriched_signals, metadata

            logger.debug("VaR within limits, no VAR_BREACH alert")
            return False, None, None

        except ImportError:
            logger.warning("VaRMethod not available - advanced risk engine may not be initialized")
            return False, None, None

    async def _evaluate_stress_test(
        self,
        signals: Dict[str, Any],
        portfolio_weights: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Évalue si un stress test échoue."""
        try:
            stress_results = self.risk_engine.run_stress_test(
                portfolio_weights, portfolio_value
            )

            # Trouver le pire scénario
            worst_scenario = min(stress_results, key=lambda x: x.portfolio_pnl_pct)
            stress_threshold = self.config.get("stress_threshold", -0.15)  # -15% max acceptable

            if worst_scenario.portfolio_pnl_pct < stress_threshold:
                enriched_signals = signals.copy()
                enriched_signals["stress_test_failed"] = {
                    "stress_scenario": worst_scenario.scenario,
                    "stress_loss": abs(worst_scenario.portfolio_pnl),
                    "stress_loss_pct": abs(worst_scenario.portfolio_pnl_pct),
                    "worst_asset": worst_scenario.worst_asset,
                    "recovery_days": worst_scenario.recovery_time_days
                }

                metadata = {
                    "failed_scenarios": len([r for r in stress_results if r.portfolio_pnl_pct < stress_threshold]),
                    "worst_loss_pct": abs(worst_scenario.portfolio_pnl_pct)
                }

                return True, enriched_signals, metadata

            logger.debug("All stress tests passed, no STRESS_TEST_FAILED alert")
            return False, None, None

        except Exception as e:
            logger.error(f"Error in stress test evaluation: {e}")
            return False, None, None

    async def _evaluate_monte_carlo(
        self,
        signals: Dict[str, Any],
        portfolio_weights: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Évalue si Monte Carlo détecte des scénarios extrêmes."""
        try:
            mc_result = self.risk_engine.run_monte_carlo_simulation(
                portfolio_weights, portfolio_value, horizon_days=30
            )

            # Seuil extrême (P5 outcome)
            extreme_threshold = self.config.get("monte_carlo_extreme_threshold", -0.25)  # -25% loss
            extreme_prob = mc_result.confidence_intervals["P5"] < extreme_threshold

            if extreme_prob or mc_result.confidence_intervals["P1"] < -0.40:
                enriched_signals = signals.copy()
                enriched_signals["monte_carlo_extreme"] = {
                    "mc_extreme_prob": abs(mc_result.confidence_intervals["P5"]),
                    "mc_threshold": portfolio_value * 0.25,  # 25% threshold
                    "max_dd_p99": mc_result.max_drawdown_p99,
                    "horizon": mc_result.horizon_days
                }

                metadata = {
                    "simulation_count": mc_result.simulations_count,
                    "worst_p1": mc_result.confidence_intervals["P1"]
                }

                return True, enriched_signals, metadata

            logger.debug("Monte Carlo within acceptable range, no alert")
            return False, None, None

        except Exception as e:
            logger.error(f"Error in Monte Carlo evaluation: {e}")
            return False, None, None

    async def _evaluate_concentration(
        self,
        signals: Dict[str, Any],
        portfolio_weights: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Évalue si la concentration de risque est excessive."""
        try:
            concentration = signals.get("concentration", 0.0)
            top_assets = signals.get("top_contributors", [])
            concentration_threshold = self.config.get("concentration_threshold", 0.25)

            if concentration > concentration_threshold:
                enriched_signals = signals.copy()
                enriched_signals["risk_concentration"] = {
                    "concentration_ratio": concentration,
                    "top_assets": top_assets[:5]
                }

                metadata = {
                    "concentration_ratio": concentration,
                    "top_assets": top_assets[:3]
                }

                return True, enriched_signals, metadata

            logger.debug("Risk concentration within tolerance, no alert")
            return False, None, None

        except Exception as e:
            logger.error(f"Error in concentration evaluation: {e}")
            return False, None, None

    async def get_portfolio_context(self, governance_engine) -> Tuple[Dict[str, float], float]:
        """
        Extrait le contexte portfolio depuis governance engine.

        Args:
            governance_engine: Instance de GovernanceEngine

        Returns:
            Tuple[portfolio_weights, portfolio_value]
        """
        portfolio_weights: Dict[str, float] = {}
        portfolio_value = 100000  # Fallback

        try:
            current_state = await governance_engine.get_current_state()
            if not current_state:
                return portfolio_weights, portfolio_value

            # Extraire les poids depuis le plan
            plan = getattr(current_state, "current_plan", None) or getattr(current_state, "proposed_plan", None)
            if plan and getattr(plan, "targets", None):
                cleaned_weights: Dict[str, float] = {}
                for target in plan.targets:
                    symbol = getattr(target, "symbol", None)
                    raw_weight = getattr(target, "weight", None)
                    if not symbol or raw_weight is None:
                        continue
                    try:
                        cleaned_weights[symbol] = float(raw_weight)
                    except (TypeError, ValueError):
                        logger.debug(f"Skipping target {symbol} with invalid weight {raw_weight!r}")
                if cleaned_weights:
                    portfolio_weights = cleaned_weights

            # Fallback: extraire depuis execution_policy.target_allocation
            if not portfolio_weights and hasattr(current_state.execution_policy, "target_allocation"):
                maybe_targets = getattr(current_state.execution_policy, "target_allocation", {})
                if isinstance(maybe_targets, dict):
                    try:
                        portfolio_weights = {k: float(v) for k, v in maybe_targets.items() if v is not None}
                    except (TypeError, ValueError):
                        portfolio_weights = {}

            # Récupérer la valeur réelle du portfolio (seulement si user_id explicite)
            user_id = getattr(current_state, 'user_id', None)
            if user_id and user_id != 'demo':
                try:
                    from services.balance_service import balance_service

                    source = getattr(current_state, 'source', 'cointracking_api')

                    balance_result = await balance_service.resolve_current_balances(source=source, user_id=user_id)
                    balances = balance_result.get('items', []) if isinstance(balance_result, dict) else []

                    if balances:
                        portfolio_value = sum(float(b.get('value_usd', 0)) for b in balances)
                        logger.debug(f"Real portfolio value: ${portfolio_value:,.2f}")

                except Exception as e:
                    logger.warning(f"Failed to retrieve portfolio value: {e}, using fallback $100,000")
            else:
                logger.debug("No explicit user_id in current_state, using fallback portfolio value $100,000")

        except Exception as e:
            logger.error(f"Error extracting portfolio context: {e}")

        return portfolio_weights, portfolio_value
