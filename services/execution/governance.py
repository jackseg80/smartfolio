"""
Governance Engine pour Decision Engine Unifié

Ce module centralise la gouvernance des décisions d'allocation :
- Single-writer strict pour les targets
- State machine governance (manual/ai_assisted/full_ai)
- Policy d'exécution unifiée (mode/cap/ramp)
- Centralisation contradiction index depuis composite-score-v2.js

Refactoré en modules:
- freeze_policy.py: FreezeType, FreezeSemantics
- signals.py: MLSignals, SignalExtractor, RealSignalExtractor
- policy_engine.py: Policy, PolicyEngine
- hysteresis.py: HysteresisManager
"""

from typing import Dict, List, Any, Optional, Literal, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import asyncio
import httpx

# Custom exceptions for better error handling (from shared, not api)
from shared.exceptions import (
    ConfigurationException,
    APIException,
    GovernanceException,
    DataException
)

# Import refactored modules
from .freeze_policy import FreezeType, FreezeSemantics
from .signals import MLSignals, SignalExtractor, RealSignalExtractor, create_default_signals
from .policy_engine import Policy, PolicyEngine, ExecMode
from .hysteresis import HysteresisManager

# Logger setup
logger = logging.getLogger(__name__)

# Import real ML orchestrator
try:
    from ..ml.orchestrator import get_orchestrator
    ML_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML Orchestrator not available: {e}")
    ML_ORCHESTRATOR_AVAILABLE = False

# Phase 3C: Import Hybrid Intelligence components
try:
    from ..intelligence.explainable_ai import ExplainableAIEngine
    from ..intelligence.human_loop import HumanInTheLoopEngine
    from ..intelligence.feedback_learning import FeedbackLearningEngine
    from ..orchestration.hybrid_orchestrator import HybridOrchestrator
    HYBRID_INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Phase 3C Hybrid Intelligence not available: {e}")
    HYBRID_INTELLIGENCE_AVAILABLE = False

# Types pour la governance
GovernanceMode = Literal["manual", "ai_assisted", "full_ai", "freeze"]
PlanStatus = Literal["DRAFT", "REVIEWED", "APPROVED", "ACTIVE", "EXECUTED", "CANCELLED"]


class Target(BaseModel):
    """Cible d'allocation pour un groupe/asset"""
    symbol: str = Field(..., description="Symbole ou groupe (BTC, ETH, Stablecoins, etc.)")
    weight: float = Field(..., ge=0.0, le=1.0, description="Poids d'allocation [0-1]")


class DecisionPlan(BaseModel):
    """Plan de décision avec targets et métadonnées"""
    plan_id: str = Field(..., description="ID unique du plan")
    created_at: datetime = Field(default_factory=datetime.now, description="Date de création")
    status: PlanStatus = Field(default="DRAFT", description="Statut du plan")
    version: int = Field(default=1, ge=1, description="Version pour concurrency")
    etag: str = Field(..., description="ETag pour optimistic concurrency")

    # State transition timestamps
    reviewed_at: Optional[datetime] = Field(default=None, description="Date de review")
    approved_at: Optional[datetime] = Field(default=None, description="Date d'approbation")
    activated_at: Optional[datetime] = Field(default=None, description="Date d'activation")
    executed_at: Optional[datetime] = Field(default=None, description="Date d'exécution")
    cancelled_at: Optional[datetime] = Field(default=None, description="Date d'annulation")

    # Contenu du plan
    targets: List[Target] = Field(..., description="Cibles d'allocation")
    governance_mode: GovernanceMode = Field(..., description="Mode de gouvernance")

    # Constraints et validation
    total_weight: float = Field(default=1.0, description="Somme des poids (doit = 1.0)")
    risk_budget: Optional[float] = Field(default=None, description="Budget de risque")
    non_removable: List[str] = Field(default_factory=list, description="Assets non supprimables")

    # State transition metadata
    created_by: str = Field(default="system", description="Créateur du plan")
    reviewed_by: Optional[str] = Field(default=None, description="Reviewer du plan")
    approved_by: Optional[str] = Field(default=None, description="Approbateur")
    notes: Optional[str] = Field(default=None, description="Notes du plan")
    review_notes: Optional[str] = Field(default=None, description="Notes de review")
    approval_notes: Optional[str] = Field(default=None, description="Notes d'approbation")


class DecisionState(BaseModel):
    """État global du Decision Engine"""
    # Plan actuel
    current_plan: Optional[DecisionPlan] = Field(default=None, description="Plan actuellement actif")
    proposed_plan: Optional[DecisionPlan] = Field(default=None, description="Plan proposé en attente")

    # Governance
    governance_mode: GovernanceMode = Field(default="manual", description="Mode de gouvernance global")
    execution_policy: Policy = Field(default_factory=Policy, description="Politique d'exécution")
    last_applied_policy: Optional[Policy] = Field(default=None, description="Derniere policy appliquee manuellement")
    last_manual_policy_update: Optional[datetime] = Field(default=None, description="Timestamp derniere activation manuelle")

    # Signaux ML
    signals: MLSignals = Field(default_factory=MLSignals, description="Signaux ML actuels")
    raw_signals: Dict[str, Any] = Field(default_factory=dict, description="Signaux ML bruts pour XAI (structure orchestrator)")

    # Métadonnées
    last_update: datetime = Field(default_factory=datetime.now, description="Dernière MAJ")
    system_status: str = Field(default="operational", description="Statut système")
    auto_unfreeze_at: Optional[datetime] = Field(default=None, description="Auto-unfreeze programmé")

    # Phase 1C: Type de freeze avec sémantique claire
    freeze_type: Optional[str] = Field(default=None, description="Type de freeze actuel (s3_freeze, error_freeze, full_freeze)")


class GovernanceEngine:
    """
    Moteur de gouvernance centralisé pour les décisions d'allocation

    Responsabilités :
    - Centralise la logique de contradiction depuis composite-score-v2.js
    - Extrait la policy logic depuis UnifiedInsights
    - Gère les transitions d'état DRAFT→ACTIVE
    - Interface unique avec les signaux ML

    Utilise les modules refactorisés:
    - PolicyEngine pour dérivation des policies
    - HysteresisManager pour hystérésis anti-yo-yo
    - FreezeSemantics pour sémantique freeze
    - SignalExtractor/RealSignalExtractor pour extraction signaux
    """

    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url.rstrip("/")
        self.current_state = DecisionState()

        # TTL vs Cooldown separation (critique essentielle)
        self._last_signals_fetch = datetime.min
        self._last_plan_publication = datetime.min
        self._signals_ttl_seconds = 60  # TEMPORARY: 60 seconds to force refresh
        self._plan_cooldown_hours = 24

        # Initialize refactored components
        self.policy_engine = PolicyEngine(
            signals_ttl_seconds=self._signals_ttl_seconds,
            plan_cooldown_hours=self._plan_cooldown_hours
        )
        self.hysteresis_manager = HysteresisManager()

        # Phase 3C: Initialize Hybrid Intelligence components
        self.hybrid_intelligence_enabled = HYBRID_INTELLIGENCE_AVAILABLE
        if self.hybrid_intelligence_enabled:
            try:
                self.explainable_ai = ExplainableAIEngine()
                self.human_loop = HumanInTheLoopEngine()
                self.feedback_learning = FeedbackLearningEngine()
                self.hybrid_orchestrator = HybridOrchestrator()
                logger.info("Phase 3C Hybrid Intelligence components initialized")
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Failed to initialize Phase 3C components: {e}")
                self.hybrid_intelligence_enabled = False
            except Exception as e:
                logger.exception(f"Unexpected error initializing Phase 3C components: {e}")
                self.hybrid_intelligence_enabled = False
        else:
            self.explainable_ai = None
            self.human_loop = None
            self.feedback_learning = None
            self.hybrid_orchestrator = None

        logger.info(f"GovernanceEngine initialized with TTL/cooldown separation, Hybrid Intelligence: {self.hybrid_intelligence_enabled}")

    async def get_current_state(self) -> DecisionState:
        """
        Retourne l'état actuel du Decision Engine
        Agrège : store local + signaux ML + policy dérivée
        Sépare TTL (signaux) et cooldown (plans)
        """
        try:
            # Check auto-unfreeze TTL
            await self.check_auto_unfreeze()

            # TTL check : Refresh signals si TTL expiré
            signals_expired = (datetime.now() - self._last_signals_fetch).total_seconds() > self._signals_ttl_seconds
            if signals_expired:
                await self._refresh_ml_signals()
                logger.debug(f"ML signals refreshed (TTL {self._signals_ttl_seconds}s expired)")

            # Calculate signals age
            signals_age = 0.0
            if hasattr(self.current_state.signals, 'as_of') and self.current_state.signals.as_of:
                signals_age = (datetime.now() - self.current_state.signals.as_of).total_seconds()

            # Update hysteresis state
            var_state, stale_state = self.hysteresis_manager.update_hysteresis_state(
                self.current_state.signals, signals_age
            )

            # Check progressive clear
            self.hysteresis_manager.check_progressive_clear()

            # Dérive la policy avec les composants refactorisés
            self.current_state.execution_policy = self.policy_engine.derive_execution_policy(
                signals=self.current_state.signals,
                governance_mode=self.current_state.governance_mode,
                manual_policy=getattr(self.current_state, "last_applied_policy", None),
                alert_cap_reduction=self.hysteresis_manager.get_alert_cap_reduction(),
                var_state=var_state,
                stale_state=stale_state,
                signals_age=signals_age
            )

            # Cooldown check : Vérifier si on peut publier de nouveaux plans
            plan_cooldown_active = (datetime.now() - self._last_plan_publication).total_seconds() < (self._plan_cooldown_hours * 3600)

            # Actualiser l'état global
            self.current_state.last_update = datetime.now()

            logger.debug(f"Current governance state: mode={self.current_state.governance_mode}, "
                        f"contradiction={self.current_state.signals.contradiction_index:.3f}, "
                        f"signals_fresh={(not signals_expired)}, cooldown_active={plan_cooldown_active}")

            return self.current_state

        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            # FIX Oct 2025: Fallback safe 8% (not 1% ultra-restrictive)
            return DecisionState(
                governance_mode="manual",
                execution_policy=Policy(
                    mode="Freeze",
                    cap_daily=0.08,
                    signals_ttl_seconds=300,
                    plan_cooldown_hours=48
                ),
                system_status="error"
            )

    async def _refresh_ml_signals(self) -> None:
        """Refresh les signaux ML depuis le vrai MLOrchestrator ou fallback vers endpoints"""
        try:
            logger.info(f"[DEBUG] _refresh_ml_signals called, ML_ORCHESTRATOR_AVAILABLE={ML_ORCHESTRATOR_AVAILABLE}")
            if ML_ORCHESTRATOR_AVAILABLE:
                # Use real ML orchestrator
                orchestrator = get_orchestrator()

                # Get unified predictions from all models
                ml_predictions = await orchestrator.get_unified_predictions(
                    symbols=['BTC', 'ETH', 'SOL'],
                    horizons=[1, 7, 30]
                )
                logger.info(f"[DEBUG] Got ML predictions, keys: {list(ml_predictions.keys())}")

                if 'error' not in ml_predictions:
                    # Extract signals from real ML models using RealSignalExtractor
                    self.current_state.signals = MLSignals(
                        as_of=datetime.now(),
                        volatility=RealSignalExtractor.extract_volatility_signals(ml_predictions),
                        regime=RealSignalExtractor.extract_regime_signals(ml_predictions),
                        correlation=RealSignalExtractor.extract_correlation_signals(ml_predictions),
                        sentiment=RealSignalExtractor.extract_sentiment_signals(ml_predictions),
                        decision_score=ml_predictions.get('ensemble', {}).get('confidence_level', 0.6),
                        confidence=RealSignalExtractor.calculate_confidence(ml_predictions),
                        contradiction_index=RealSignalExtractor.compute_contradiction_index(ml_predictions),
                        sources_used=list(ml_predictions.get('models', {}).keys())
                    )

                    # Store raw signals for XAI (Phase 3C)
                    self.current_state.raw_signals = ml_predictions

                    self._last_signals_fetch = datetime.now()
                    logger.debug("Real ML signals refreshed successfully")
                    return
                else:
                    logger.warning(f"ML orchestrator error: {ml_predictions.get('error')}")

            # Fallback to API endpoint
            async with httpx.AsyncClient() as client:
                signals_response = await client.get(f"{self.api_base_url}/api/ml/status", timeout=5.0)
                if signals_response.status_code == 200:
                    ml_status = signals_response.json()

                    # Fallback to simulated signals using SignalExtractor
                    self.current_state.signals = MLSignals(
                        as_of=datetime.now(),
                        volatility=SignalExtractor.extract_volatility_signals(ml_status),
                        regime=SignalExtractor.extract_regime_signals(ml_status),
                        correlation=SignalExtractor.extract_correlation_signals(ml_status),
                        sentiment=SignalExtractor.extract_sentiment_signals(ml_status),
                        decision_score=0.6,
                        confidence=0.75,
                        contradiction_index=SignalExtractor.compute_contradiction_index(ml_status),
                        sources_used=["volatility_fallback", "regime_fallback", "correlation_fallback", "sentiment_fallback"]
                    )

                    # Store raw signals for XAI (Phase 3C) - fallback format
                    self.current_state.raw_signals = ml_status

                    self._last_signals_fetch = datetime.now()
                    logger.debug("ML signals refreshed via fallback API")

        except httpx.HTTPError as e:
            logger.warning(f"HTTP error refreshing ML signals: {e}")
            if not self.current_state.signals:
                self.current_state.signals = create_default_signals()
        except (ValueError, KeyError) as e:
            logger.warning(f"Data parsing error in ML signals: {e}")
            if not self.current_state.signals:
                self.current_state.signals = create_default_signals()
        except Exception as e:
            logger.exception(f"Unexpected error refreshing ML signals: {e}")
            if not self.current_state.signals:
                self.current_state.signals = create_default_signals()

    async def get_current_ml_signals(self) -> Optional[MLSignals]:
        """Retourne les signaux ML actuels (wrapper pour endpoints)"""
        try:
            state = await self.get_current_state()
            return state.signals if state else None
        except (AttributeError, TypeError) as e:
            logger.debug(f"State error getting current ML signals: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error getting current ML signals: {e}")
            return None

    def _derive_execution_policy(self) -> Policy:
        """
        Wrapper pour compatibilité avec le code existant.
        Délègue à policy_engine.derive_execution_policy().
        """
        # Calculate signals age
        signals_age = 0.0
        if hasattr(self.current_state.signals, 'as_of') and self.current_state.signals.as_of:
            signals_age = (datetime.now() - self.current_state.signals.as_of).total_seconds()

        return self.policy_engine.derive_execution_policy(
            signals=self.current_state.signals,
            governance_mode=self.current_state.governance_mode,
            manual_policy=getattr(self.current_state, "last_applied_policy", None),
            alert_cap_reduction=self.hysteresis_manager.get_alert_cap_reduction(),
            var_state=self.hysteresis_manager.get_var_state(),
            stale_state=self.hysteresis_manager.get_stale_state(),
            signals_age=signals_age
        )

    @property
    def _last_cap(self) -> float:
        """Wrapper property pour accéder à _last_cap du policy_engine"""
        return self.policy_engine.get_last_cap()

    @_last_cap.setter
    def _last_cap(self, value: float) -> None:
        """Wrapper setter pour définir _last_cap du policy_engine"""
        self.policy_engine.set_last_cap(value)

    # ========================================================================
    # Freeze Management - Délègue à FreezeSemantics
    # ========================================================================

    async def freeze_system(self, reason: str, duration_minutes: Optional[int] = None, freeze_type: str = None) -> bool:
        """Phase 1C: Freeze système avec sémantique claire"""
        try:
            # Déterminer le type de freeze automatiquement si non spécifié
            if freeze_type is None:
                freeze_type = FreezeSemantics.infer_freeze_type(reason)

            logger.info(f"Freezing system: {reason} (Type: {freeze_type}, TTL: {duration_minutes}min)"
                       if duration_minutes else f"Freezing system: {reason} (Type: {freeze_type})")

            # Set governance mode to freeze avec type sémantique
            self.current_state.governance_mode = "freeze"
            self.current_state.freeze_type = freeze_type
            self.current_state.system_status = "frozen"
            self.current_state.last_update = datetime.now()

            # Store auto-unfreeze time if TTL specified
            if duration_minutes:
                self.current_state.auto_unfreeze_at = datetime.now() + timedelta(minutes=duration_minutes)
                logger.info(f"Auto-unfreeze scheduled for: {self.current_state.auto_unfreeze_at}")
            else:
                self.current_state.auto_unfreeze_at = None

            # Caps et policy selon le type de freeze
            cap_daily, ramp_hours = FreezeSemantics.get_freeze_caps(freeze_type)
            allowed_ops = FreezeSemantics.get_allowed_operations(freeze_type)
            ops_summary = [k for k, v in allowed_ops.items() if v]

            self.current_state.execution_policy = Policy(
                mode="Freeze",
                cap_daily=cap_daily,
                ramp_hours=ramp_hours,
                plan_cooldown_hours=168,
                notes=f"Freeze {freeze_type}: {reason}. Allowed: {', '.join(ops_summary)}" +
                      (f" (auto-unfreeze: {duration_minutes}min)" if duration_minutes else "")
            )

            logger.info(f"System frozen with {freeze_type}: allowed operations: {ops_summary}")
            return True

        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Invalid parameters freezing system: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error freezing system: {e}")
            return False

    async def unfreeze_system(self) -> bool:
        """Phase 1C: Unfreeze système avec nettoyage sémantique"""
        try:
            previous_freeze_type = self.current_state.freeze_type
            logger.info(f"Unfreezing system (was: {previous_freeze_type})")

            self.current_state.governance_mode = "manual"
            self.current_state.freeze_type = None
            self.current_state.system_status = "operational"
            self.current_state.last_update = datetime.now()
            self.current_state.auto_unfreeze_at = None

            # Recalculate signals age for policy derivation
            signals_age = 0.0
            if hasattr(self.current_state.signals, 'as_of') and self.current_state.signals.as_of:
                signals_age = (datetime.now() - self.current_state.signals.as_of).total_seconds()

            # Derive normal execution policy
            self.current_state.execution_policy = self.policy_engine.derive_execution_policy(
                signals=self.current_state.signals,
                governance_mode="manual",
                alert_cap_reduction=self.hysteresis_manager.get_alert_cap_reduction(),
                var_state=self.hysteresis_manager.get_var_state(),
                stale_state=self.hysteresis_manager.get_stale_state(),
                signals_age=signals_age
            )

            logger.info(f"System successfully unfrozen from {previous_freeze_type}")
            return True

        except (AttributeError, TypeError) as e:
            logger.warning(f"State error unfreezing system: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error unfreezing system: {e}")
            return False

    def validate_operation(self, operation_type: str) -> Tuple[bool, str]:
        """Phase 1C: Valide si une opération est autorisée selon le freeze actuel"""
        try:
            current_freeze = self.current_state.freeze_type
            if current_freeze is None:
                return True, "No freeze active - operation allowed"

            return FreezeSemantics.validate_operation(current_freeze, operation_type)

        except Exception as e:
            logger.error(f"Error validating operation {operation_type}: {e}")
            return False, f"Error validating operation: {str(e)}"

    def get_freeze_status(self) -> Dict[str, Any]:
        """Phase 1C: Retourne le statut freeze détaillé pour UI"""
        try:
            freeze_type = self.current_state.freeze_type
            auto_unfreeze = self.current_state.auto_unfreeze_at

            if freeze_type is None:
                return {
                    "is_frozen": False,
                    "freeze_type": None,
                    "allowed_operations": FreezeSemantics.get_allowed_operations("normal"),
                    "status_message": "System operational - all operations allowed"
                }

            allowed_ops = FreezeSemantics.get_allowed_operations(freeze_type)
            remaining_minutes = None

            if auto_unfreeze:
                remaining_seconds = (auto_unfreeze - datetime.now()).total_seconds()
                remaining_minutes = max(0, int(remaining_seconds / 60))

            status_msg = FreezeSemantics.get_status_message(freeze_type)

            return {
                "is_frozen": True,
                "freeze_type": freeze_type,
                "allowed_operations": allowed_ops,
                "remaining_minutes": remaining_minutes,
                "auto_unfreeze_at": auto_unfreeze.isoformat() if auto_unfreeze else None,
                "status_message": status_msg
            }

        except Exception as e:
            logger.error(f"Error getting freeze status: {e}")
            return {
                "is_frozen": False,
                "freeze_type": None,
                "error": str(e)
            }

    async def check_auto_unfreeze(self) -> bool:
        """Vérifie et exécute auto-unfreeze si TTL écoulé"""
        try:
            if (self.current_state.governance_mode == "freeze" and
                self.current_state.auto_unfreeze_at and
                datetime.now() >= self.current_state.auto_unfreeze_at):

                logger.info("Auto-unfreeze TTL expired, unfreezing system")
                await self.unfreeze_system()
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking auto-unfreeze: {e}")
            return False

    # ========================================================================
    # Alert Cap Management - Délègue à HysteresisManager
    # ========================================================================

    def apply_alert_cap_reduction(self, reduction_percentage: float, alert_id: str, reason: str) -> bool:
        """Phase 1B: AlertEngine peut déclencher réduction cap"""
        return self.hysteresis_manager.apply_alert_cap_reduction(reduction_percentage, alert_id, reason)

    def clear_alert_cap_reduction(self, progressive: bool = True) -> bool:
        """Phase 1B: Nettoyer réduction cap AlertEngine"""
        return self.hysteresis_manager.clear_alert_cap_reduction(progressive)

    # ========================================================================
    # Plan Management
    # ========================================================================

    async def review_plan(self, plan_id: str, reviewed_by: str, notes: Optional[str] = None, expected_etag: Optional[str] = None) -> bool:
        """Transition DRAFT → REVIEWED with optional ETag validation"""
        try:
            plan = self._find_plan_by_id(plan_id)
            if not plan:
                logger.error(f"Plan {plan_id} not found")
                return False

            if expected_etag and not self.validate_etag(plan_id, expected_etag):
                logger.error(f"ETag validation failed for plan {plan_id}")
                return False

            if plan.status != "DRAFT":
                logger.error(f"Plan {plan_id} is not in DRAFT state (current: {plan.status})")
                return False

            plan.status = "REVIEWED"
            plan.reviewed_at = datetime.now()
            plan.reviewed_by = reviewed_by
            plan.review_notes = notes
            plan.version += 1
            plan.etag = f"etag_{datetime.now().timestamp()}"

            self.current_state.last_update = datetime.now()
            logger.info(f"Plan {plan_id} reviewed by {reviewed_by}")
            return True

        except Exception as e:
            logger.error(f"Error reviewing plan {plan_id}: {e}")
            return False

    async def approve_plan(self, plan_id: str, approved_by: str, notes: Optional[str] = None, expected_etag: Optional[str] = None) -> bool:
        """Transition REVIEWED → APPROVED with optional ETag validation"""
        try:
            plan = self._find_plan_by_id(plan_id)
            if not plan:
                logger.error(f"Plan {plan_id} not found")
                return False

            if expected_etag and not self.validate_etag(plan_id, expected_etag):
                logger.error(f"ETag validation failed for plan {plan_id}")
                return False

            if plan.status != "REVIEWED":
                logger.error(f"Plan {plan_id} is not in REVIEWED state (current: {plan.status})")
                return False

            # Phase 3C: Check if human-in-the-loop intervention is needed
            if self.hybrid_intelligence_enabled and self.human_loop:
                try:
                    risk_assessment = await self._assess_decision_risk(plan)

                    if risk_assessment.get("requires_human_review", False):
                        human_request = {
                            "plan_id": plan_id,
                            "decision_type": "plan_approval",
                            "risk_level": risk_assessment.get("risk_level", "medium"),
                            "context": {
                                "targets": [{"symbol": t.symbol, "weight": t.weight} for t in plan.targets],
                                "risk_factors": risk_assessment.get("risk_factors", []),
                                "ai_confidence": risk_assessment.get("ai_confidence", 0.5)
                            },
                            "deadline": datetime.now() + timedelta(hours=2),
                            "fallback_action": "reject"
                        }

                        human_decision = await self.human_loop.request_human_decision(
                            decision_request=human_request,
                            urgency="high" if risk_assessment.get("risk_level") == "high" else "medium"
                        )

                        if human_decision and not human_decision.get("completed", False):
                            plan.status = "PENDING_HUMAN_REVIEW"
                            plan.human_review_requested = datetime.now()
                            plan.human_review_context = risk_assessment
                            logger.info(f"Plan {plan_id} requires human review due to {risk_assessment.get('primary_concern', 'high risk')}")
                            return True

                except Exception as e:
                    logger.error(f"Phase 3C human loop assessment failed for plan {plan_id}: {e}")

            plan.status = "APPROVED"
            plan.approved_at = datetime.now()
            plan.approved_by = approved_by
            plan.approval_notes = notes
            plan.version += 1
            plan.etag = f"etag_{datetime.now().timestamp()}"

            self.current_state.last_update = datetime.now()
            logger.info(f"Plan {plan_id} approved by {approved_by}")
            return True

        except Exception as e:
            logger.error(f"Error approving plan {plan_id}: {e}")
            return False

    async def reject_plan(self, plan_id: str, rejected_by: str, notes: Optional[str] = None, expected_etag: Optional[str] = None) -> bool:
        """Reject a plan in DRAFT or REVIEWED state"""
        try:
            plan = self._find_plan_by_id(plan_id)
            if not plan:
                logger.error(f"Plan {plan_id} not found")
                return False

            if expected_etag and not self.validate_etag(plan_id, expected_etag):
                logger.error(f"ETag validation failed for plan {plan_id}")
                return False

            if plan.status not in ["DRAFT", "REVIEWED"]:
                logger.error(f"Plan {plan_id} cannot be rejected from {plan.status} state")
                return False

            plan.status = "REJECTED"
            plan.rejected_at = datetime.now()
            plan.rejected_by = rejected_by
            plan.rejection_notes = notes or "Plan rejected"
            plan.version += 1
            plan.etag = f"etag_{datetime.now().timestamp()}"

            self.current_state.last_update = datetime.now()
            logger.info(f"Plan {plan_id} rejected by {rejected_by}: {notes}")
            return True

        except Exception as e:
            logger.error(f"Error rejecting plan {plan_id}: {e}")
            return False

    async def activate_plan(self, plan_id: str, expected_etag: Optional[str] = None) -> bool:
        """Transition APPROVED → ACTIVE with optional ETag validation"""
        try:
            plan = self._find_plan_by_id(plan_id)
            if not plan:
                logger.error(f"Plan {plan_id} not found")
                return False

            if expected_etag and not self.validate_etag(plan_id, expected_etag):
                logger.error(f"ETag validation failed for plan {plan_id}")
                return False

            if plan.status != "APPROVED":
                logger.error(f"Plan {plan_id} is not in APPROVED state (current: {plan.status})")
                return False

            # Deactivate current active plan if any
            if self.current_state.current_plan and self.current_state.current_plan.status == "ACTIVE":
                self.current_state.current_plan.status = "EXECUTED"
                self.current_state.current_plan.executed_at = datetime.now()
                logger.info(f"Previous plan {self.current_state.current_plan.plan_id} marked as executed")

            plan.status = "ACTIVE"
            plan.activated_at = datetime.now()
            plan.version += 1
            plan.etag = f"etag_{datetime.now().timestamp()}"

            self.current_state.current_plan = plan
            if self.current_state.proposed_plan and self.current_state.proposed_plan.plan_id == plan_id:
                self.current_state.proposed_plan = None

            self.current_state.last_update = datetime.now()
            logger.info(f"Plan {plan_id} activated")
            return True

        except Exception as e:
            logger.error(f"Error activating plan {plan_id}: {e}")
            return False

    async def execute_plan(self, plan_id: str, expected_etag: Optional[str] = None) -> bool:
        """Transition ACTIVE → EXECUTED with optional ETag validation"""
        try:
            plan = self._find_plan_by_id(plan_id)
            if not plan:
                logger.error(f"Plan {plan_id} not found")
                return False

            if expected_etag and not self.validate_etag(plan_id, expected_etag):
                logger.error(f"ETag validation failed for plan {plan_id}")
                return False

            if plan.status != "ACTIVE":
                logger.error(f"Plan {plan_id} is not in ACTIVE state (current: {plan.status})")
                return False

            plan.status = "EXECUTED"
            plan.executed_at = datetime.now()
            plan.version += 1
            plan.etag = f"etag_{datetime.now().timestamp()}"

            self.current_state.last_update = datetime.now()
            logger.info(f"Plan {plan_id} marked as executed")
            return True

        except Exception as e:
            logger.error(f"Error executing plan {plan_id}: {e}")
            return False

    async def cancel_plan(self, plan_id: str, cancelled_by: str, reason: Optional[str] = None, expected_etag: Optional[str] = None) -> bool:
        """Cancel plan from any state → CANCELLED with optional ETag validation"""
        try:
            plan = self._find_plan_by_id(plan_id)
            if not plan:
                logger.error(f"Plan {plan_id} not found")
                return False

            if expected_etag and not self.validate_etag(plan_id, expected_etag):
                logger.error(f"ETag validation failed for plan {plan_id}")
                return False

            if plan.status in ["EXECUTED", "CANCELLED"]:
                logger.error(f"Plan {plan_id} cannot be cancelled (current: {plan.status})")
                return False

            plan.status = "CANCELLED"
            plan.cancelled_at = datetime.now()
            plan.notes = f"{plan.notes or ''} | Cancelled by {cancelled_by}: {reason or 'No reason provided'}"
            plan.version += 1
            plan.etag = f"etag_{datetime.now().timestamp()}"

            if self.current_state.current_plan and self.current_state.current_plan.plan_id == plan_id:
                self.current_state.current_plan = None
            if self.current_state.proposed_plan and self.current_state.proposed_plan.plan_id == plan_id:
                self.current_state.proposed_plan = None

            self.current_state.last_update = datetime.now()
            logger.info(f"Plan {plan_id} cancelled by {cancelled_by}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling plan {plan_id}: {e}")
            return False

    def _find_plan_by_id(self, plan_id: str) -> Optional[DecisionPlan]:
        """Helper to find plan by ID in current state"""
        if self.current_state.current_plan and self.current_state.current_plan.plan_id == plan_id:
            return self.current_state.current_plan
        if self.current_state.proposed_plan and self.current_state.proposed_plan.plan_id == plan_id:
            return self.current_state.proposed_plan
        return None

    def validate_etag(self, plan_id: str, provided_etag: str) -> bool:
        """Valide l'ETag pour le contrôle de concurrence optimiste"""
        plan = self._find_plan_by_id(plan_id)
        if not plan:
            logger.warning(f"Plan {plan_id} not found for ETag validation")
            return False

        current_etag = plan.etag
        if current_etag != provided_etag:
            logger.warning(f"ETag mismatch for plan {plan_id}: expected {current_etag}, got {provided_etag}")
            return False

        logger.debug(f"ETag validation successful for plan {plan_id}")
        return True

    # ========================================================================
    # Utilities
    # ========================================================================

    def is_change_within_no_trade_zone(self, current_weights: Dict[str, float], target_weights: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """Vérifie si les changements d'allocation respectent la no-trade zone"""
        try:
            policy = self.current_state.execution_policy
            no_trade_threshold = policy.no_trade_threshold_pct if policy else 0.02

            changes = {}
            all_within_zone = True

            all_symbols = set(current_weights.keys()) | set(target_weights.keys())

            for symbol in all_symbols:
                current_weight = current_weights.get(symbol, 0.0)
                target_weight = target_weights.get(symbol, 0.0)
                change = abs(target_weight - current_weight)

                changes[symbol] = {
                    'current': current_weight,
                    'target': target_weight,
                    'change': change,
                    'within_zone': change <= no_trade_threshold
                }

                if change > no_trade_threshold:
                    all_within_zone = False

            logger.debug(f"No-trade zone check: threshold={no_trade_threshold:.1%}, all_within={all_within_zone}")
            return all_within_zone, changes

        except Exception as e:
            logger.error(f"Error checking no-trade zone: {e}")
            return False, {}

    def estimate_execution_cost(self, target_weights: Dict[str, float], current_portfolio_usd: float = 100000) -> Dict[str, Any]:
        """Estime les coûts d'exécution pour un changement d'allocation"""
        try:
            policy = self.current_state.execution_policy
            execution_cost_bps = policy.execution_cost_bps if policy else 15

            total_volume = 0
            trade_details = {}

            for symbol, target_weight in target_weights.items():
                current_weight = 0
                weight_change = abs(target_weight - current_weight)
                trade_volume_usd = weight_change * current_portfolio_usd

                trade_details[symbol] = {
                    'volume_usd': trade_volume_usd,
                    'cost_bps': execution_cost_bps,
                    'estimated_cost_usd': trade_volume_usd * execution_cost_bps / 10000
                }

                total_volume += trade_volume_usd

            total_cost_usd = total_volume * execution_cost_bps / 10000
            cost_percentage = (total_cost_usd / current_portfolio_usd) * 100

            execution_summary = {
                'total_volume_usd': total_volume,
                'total_cost_usd': total_cost_usd,
                'cost_percentage': cost_percentage,
                'cost_bps': execution_cost_bps,
                'trade_details': trade_details,
                'cost_efficient': cost_percentage < 0.5
            }

            logger.debug(f"Execution cost estimate: ${total_cost_usd:.2f} ({cost_percentage:.2f}%)")
            return execution_summary

        except Exception as e:
            logger.error(f"Error estimating execution cost: {e}")
            return {'error': str(e)}

    async def approve_decision(self, decision_id: str, approved: bool, reason: Optional[str] = None) -> bool:
        """Legacy method - redirect to approve_plan for compatibility"""
        try:
            logger.info(f"Decision {decision_id}: approved={approved}, reason={reason}")

            if approved:
                plan = self._find_plan_by_id(decision_id)
                if plan:
                    if plan.status == "DRAFT":
                        await self.review_plan(decision_id, "system", "Auto-reviewed for approval")
                    if plan.status == "REVIEWED":
                        await self.approve_plan(decision_id, "system", reason)
                    if plan.status == "APPROVED":
                        await self.activate_plan(decision_id)
                    return True
                else:
                    logger.warning("No plan found to approve")
                    return False
            else:
                return await self.cancel_plan(decision_id, "system", reason)

        except Exception as e:
            logger.error(f"Error approving decision: {e}")
            return False

    async def set_governance_mode(self, mode: str, reason: str = "Mode change") -> bool:
        """Change le mode de gouvernance"""
        try:
            logger.info(f"Changing governance mode to '{mode}': {reason}")

            if mode not in ["manual", "ai_assisted", "full_ai", "freeze"]:
                logger.error(f"Invalid governance mode: {mode}")
                return False

            self.current_state.governance_mode = mode
            self.current_state.last_update = datetime.now()

            if mode == "freeze":
                await self.freeze_system(reason)
            elif self.current_state.governance_mode == "freeze" and mode != "freeze":
                self.current_state.system_status = "operational"

            logger.info(f"Governance mode changed to '{mode}'")
            return True

        except Exception as e:
            logger.error(f"Error setting governance mode: {e}")
            return False

    def can_publish_new_plan(self) -> Tuple[bool, str]:
        """Vérifie si on peut publier un nouveau plan (respecte le cooldown)"""
        try:
            policy = self.current_state.execution_policy
            if not policy:
                return False, "No execution policy available"

            time_since_last = (datetime.now() - self._last_plan_publication).total_seconds()
            cooldown_seconds = policy.plan_cooldown_hours * 3600

            if time_since_last < cooldown_seconds:
                remaining_hours = (cooldown_seconds - time_since_last) / 3600
                return False, f"Cooldown active: {remaining_hours:.1f}h remaining"

            return True, "Cooldown expired, can publish"

        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return False, f"Error checking cooldown: {e}"

    async def create_proposed_plan(self, targets: List[Dict], reason: str = "New proposal", force_override_cooldown: bool = False) -> Tuple[bool, str]:
        """Crée un plan proposé en respectant le cooldown"""
        try:
            logger.info(f"Creating proposed plan: {reason}")

            if not force_override_cooldown:
                can_publish, cooldown_reason = self.can_publish_new_plan()
                if not can_publish:
                    logger.warning(f"Plan creation blocked by cooldown: {cooldown_reason}")
                    return False, f"Cannot publish plan: {cooldown_reason}"

            target_objects = []
            total_weight = 0.0

            for target in targets:
                symbol = target.get("symbol", "")
                weight = target.get("weight", 0.0)

                if not symbol or weight <= 0:
                    error_msg = f"Invalid target: {target}"
                    logger.error(error_msg)
                    return False, error_msg

                target_objects.append(Target(symbol=symbol, weight=weight))
                total_weight += weight

            if abs(total_weight - 1.0) > 0.001:
                logger.info(f"Normalizing weights from {total_weight:.3f} to 1.0")
                for target_obj in target_objects:
                    target_obj.weight = target_obj.weight / total_weight

            proposed_plan = DecisionPlan(
                plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                status="DRAFT",
                targets=target_objects,
                governance_mode=self.current_state.governance_mode,
                etag=f"etag_{datetime.now().timestamp()}",
                created_by="UI_System",
                notes=reason
            )

            # Phase 3C: Add Hybrid Intelligence analysis
            if self.hybrid_intelligence_enabled and self.explainable_ai:
                try:
                    decision_context = {
                        "plan_id": proposed_plan.plan_id,
                        "targets": [{"symbol": t.symbol, "weight": t.weight} for t in target_objects],
                        "governance_mode": self.current_state.governance_mode,
                        "reason": reason,
                        "ml_signals": self.current_state.raw_signals
                    }

                    xai_features = {}
                    if self.current_state.raw_signals:
                        signals = self.current_state.raw_signals
                        if isinstance(signals, dict):
                            xai_features = {k: float(v) if isinstance(v, (int, float)) else 0.0
                                          for k, v in signals.items()
                                          if isinstance(v, (int, float, bool))}

                    mode_to_prediction = {
                        "s3_alert_freeze": 0.9,
                        "error_freeze": 0.8,
                        "full_freeze": 1.0,
                        "prudent_mode": 0.6,
                        "normal": 0.3
                    }
                    prediction_value = mode_to_prediction.get(self.current_state.governance_mode, 0.5)

                    explanation = await self.explainable_ai.explain_decision(
                        model_name="allocation_plan",
                        prediction=prediction_value,
                        features=xai_features,
                        model_data={"context": decision_context, "signals": self.current_state.raw_signals}
                    )

                    if explanation:
                        proposed_plan.notes += f"\n\nAI Explanation: {explanation.summary}"
                        logger.debug(
                            f"Phase 3C XAI: confidence={explanation.confidence:.2f} "
                            f"({explanation.confidence_level.value if hasattr(explanation.confidence_level, 'value') else str(explanation.confidence_level)}), "
                            f"key_factors={[f.feature_name for f in explanation.feature_contributions[:3]]}"
                        )

                    logger.info(f"Phase 3C XAI explanation added to plan {proposed_plan.plan_id}")

                except Exception as e:
                    logger.error(f"Phase 3C XAI analysis failed for plan {proposed_plan.plan_id}: {e}")

            self.current_state.proposed_plan = proposed_plan
            self.current_state.last_update = datetime.now()
            self._last_plan_publication = datetime.now()

            success_msg = f"Proposed plan created with Phase 3C analysis: {proposed_plan.plan_id}"
            logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"Error creating proposed plan: {e}"
            logger.error(error_msg)
            return False, error_msg

    async def _assess_decision_risk(self, plan: DecisionPlan) -> Dict[str, Any]:
        """Phase 3C: Assess risk level of a decision plan to determine human oversight needs"""
        try:
            risk_factors = []
            risk_score = 0.0

            current_allocation = {}
            for target in plan.targets:
                current_weight = current_allocation.get(target.symbol, 0.0)
                weight_change = abs(target.weight - current_weight)

                if weight_change > 0.2:
                    risk_factors.append(f"Large allocation change for {target.symbol}: {weight_change:.1%}")
                    risk_score += 0.3
                elif weight_change > 0.1:
                    risk_factors.append(f"Significant allocation change for {target.symbol}: {weight_change:.1%}")
                    risk_score += 0.15

            ai_context = getattr(plan, 'context', {}).get("ai_explanation", {})
            ai_confidence = ai_context.get("confidence", 0.5)

            if ai_confidence < 0.6:
                risk_factors.append(f"Low AI confidence: {ai_confidence:.2f}")
                risk_score += 0.2
            elif ai_confidence < 0.7:
                risk_factors.append(f"Medium AI confidence: {ai_confidence:.2f}")
                risk_score += 0.1

            if self.current_state.raw_signals:
                volatility_data = self.current_state.raw_signals.get('models', {}).get('volatility', {})
                high_vol_assets = []

                for symbol, vol_predictions in volatility_data.items():
                    if isinstance(vol_predictions, dict):
                        for horizon, data in vol_predictions.items():
                            if isinstance(data, dict) and data.get('volatility_forecast', 0) > 0.4:
                                high_vol_assets.append(symbol)
                                break

                if high_vol_assets:
                    risk_factors.append(f"High volatility assets: {', '.join(high_vol_assets)}")
                    risk_score += 0.2

            if plan.governance_mode in ["full_ai"]:
                risk_factors.append("Full AI mode requires additional oversight")
                risk_score += 0.1

            if risk_score >= 0.5:
                risk_level = "high"
                requires_human_review = True
                primary_concern = "Multiple high-risk factors detected"
            elif risk_score >= 0.3:
                risk_level = "medium"
                requires_human_review = True
                primary_concern = "Moderate risk factors present"
            else:
                risk_level = "low"
                requires_human_review = False
                primary_concern = None

            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "requires_human_review": requires_human_review,
                "risk_factors": risk_factors,
                "primary_concern": primary_concern,
                "ai_confidence": ai_confidence,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error assessing decision risk: {e}")
            return {
                "risk_score": 0.5,
                "risk_level": "medium",
                "requires_human_review": True,
                "risk_factors": ["Risk assessment failed"],
                "primary_concern": "Unable to assess risk",
                "ai_confidence": 0.5,
                "error": str(e)
            }

    async def record_plan_outcome(self, plan_id: str, outcome: str, performance_data: Dict[str, Any] = None):
        """Phase 3C: Record plan execution outcome for feedback learning"""
        if not self.hybrid_intelligence_enabled or not self.feedback_learning:
            return

        try:
            plan = self._find_plan_by_id(plan_id)
            if not plan:
                logger.warning(f"Cannot record outcome for unknown plan: {plan_id}")
                return

            feedback_data = {
                "plan_id": plan_id,
                "outcome": outcome,
                "targets": [{"symbol": t.symbol, "weight": t.weight} for t in plan.targets],
                "governance_mode": plan.governance_mode,
                "ai_explanation": getattr(plan, 'context', {}).get("ai_explanation", {}),
                "risk_assessment": getattr(plan, 'human_review_context', {}),
                "performance_data": performance_data or {},
                "timestamp": datetime.now().isoformat()
            }

            await self.feedback_learning.record_decision_outcome(
                decision_id=plan_id,
                outcome=outcome,
                feedback_data=feedback_data
            )

            logger.info(f"Phase 3C: Recorded outcome '{outcome}' for plan {plan_id}")

        except Exception as e:
            logger.error(f"Phase 3C: Error recording plan outcome for {plan_id}: {e}")


# Instance globale pour réutilisation
governance_engine = GovernanceEngine()
