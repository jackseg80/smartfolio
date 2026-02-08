"""
Modèles Pydantic pour les endpoints d'exécution

Ces modèles définissent les schémas de requêtes et réponses
pour l'API d'exécution et de gouvernance.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional


# ===========================
# Modèles d'Exécution
# ===========================

class ExecutionRequest(BaseModel):
    """Requête d'exécution d'un plan"""
    rebalance_actions: List[Dict[str, Any]] = Field(..., description="Actions de rebalancement")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata (CCS, etc.)")
    dry_run: bool = Field(default=True, description="Mode simulation")
    max_parallel: int = Field(default=3, description="Ordres en parallèle max")


class ValidationResponse(BaseModel):
    """Réponse de validation"""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    plan_id: str
    total_orders: int
    total_volume: float
    large_orders_count: int


class ExecutionResponse(BaseModel):
    """Réponse d'exécution"""
    success: bool
    plan_id: str
    execution_id: str
    message: str
    estimated_duration_seconds: Optional[float] = None


class ExecutionStatus(BaseModel):
    """Statut d'exécution"""
    plan_id: str
    status: str
    is_active: bool
    total_orders: int
    completed_orders: int
    failed_orders: int
    success_rate: float
    volume_planned: float
    volume_executed: float
    total_fees: float
    execution_time: float
    completion_percentage: float
    start_time: Optional[str] = None
    end_time: Optional[str] = None


# ===========================
# Modèles de Score et Phase
# ===========================

class ScoreComponents(BaseModel):
    """Sous-scores explicatifs du score de décision"""
    trend_regime: float = Field(..., ge=0.0, le=100.0, description="Tendance et régime")
    risk: float = Field(..., ge=0.0, le=100.0, description="Métriques de risque")
    breadth_rotation: float = Field(..., ge=0.0, le=100.0, description="Largeur de marché et rotation")
    sentiment: float = Field(..., ge=0.0, le=100.0, description="Sentiment de marché")


class CanonicalScores(BaseModel):
    """Scores canoniques unifiés"""
    decision: float = Field(..., ge=0.0, le=100.0, description="Score décisionnel principal 0-100")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance dans la décision")
    contradiction: float = Field(..., ge=0.0, le=1.0, description="Index de contradiction")
    components: ScoreComponents = Field(..., description="Sous-scores explicatifs")
    as_of: str = Field(..., description="Timestamp de calcul")


class PhaseInfo(BaseModel):
    """Information sur la phase de rotation"""
    phase_now: str = Field(..., description="Phase actuelle (btc/eth/large/alt)")
    phase_probs: Dict[str, float] = Field(..., description="Probabilities per phase")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance dans la détection")
    explain: List[str] = Field(..., description="2-3 explications principales")
    next_likely: Optional[str] = Field(None, description="Phase suivante probable")


class ExecutionPressure(BaseModel):
    """Pression d'exécution court-terme"""
    pressure: float = Field(..., ge=0.0, le=100.0, description="Pression d'exécution 0-100")
    cost_estimate_bps: float = Field(..., description="Estimated execution cost in bps")
    market_impact: str = Field(..., description="Impact marché estimé (low/medium/high)")
    optimal_window_hours: int = Field(..., description="Optimal execution window")


# ===========================
# Modèles de Signaux
# ===========================

class MarketSignals(BaseModel):
    """Signaux de marché agrégés"""
    volatility: Dict[str, float] = Field(default_factory=dict, description="Volatility per asset")
    regime: Dict[str, float] = Field(default_factory=dict, description="Regime probabilities")
    correlation: Dict[str, Any] = Field(default_factory=dict, description="Key correlations (avg_correlation: float, systemic_risk: str)")
    sentiment: Dict[str, float] = Field(default_factory=dict, description="Sentiment indicators")


class CycleSignals(BaseModel):
    """Signaux de cycle et rotation"""
    btc_cycle: Dict[str, float] = Field(default_factory=dict, description="Position cycle BTC")
    rotation: Dict[str, float] = Field(default_factory=dict, description="Signaux de rotation")


class UnifiedSignals(BaseModel):
    """Bus de signaux unifié"""
    market: MarketSignals = Field(default_factory=MarketSignals, description="Signaux de marché")
    cycle: CycleSignals = Field(default_factory=CycleSignals, description="Signaux de cycle")
    as_of: str = Field(..., description="Timestamp des signaux")


class UpdateSignalsRequest(BaseModel):
    """Payload pour mise à jour partielle des signaux ML (ex: blended score issu du front)"""
    blended_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Blended Decision Score 0-100")


class RecomputeSignalsRequest(BaseModel):
    """Optionally provide components for blended recomputation.
    If omitted, backend falls back to neutral values.
    """
    ccs_mixte: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    onchain_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)


# ===========================
# Modèles de Gouvernance
# ===========================

class PortfolioMetrics(BaseModel):
    """Métriques de portefeuille actuelles"""
    var_95_pct: Optional[float] = Field(None, description="VaR 95% en %")
    sharpe_ratio: Optional[float] = Field(None, description="Ratio de Sharpe")
    hhi_concentration: Optional[float] = Field(None, description="Index HHI de concentration")
    avg_correlation: Optional[float] = Field(None, description="Weighted average correlation")
    beta_btc: Optional[float] = Field(None, description="Bêta vs BTC")
    exposures: Dict[str, float] = Field(default_factory=dict, description="Expositions par groupe")


class SuggestionIA(BaseModel):
    """Proposition IA canonique (lecture seule)"""
    targets: List[Dict[str, Any]] = Field(..., description="Cibles suggérées")
    rationale: str = Field(..., description="Logique de la suggestion")
    policy_hint: str = Field(..., description="Suggestion de policy (Slow/Normal/Aggressive)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance dans la suggestion")
    generated_at: str = Field(..., description="Timestamp de génération")


class GovernanceStateResponse(BaseModel):
    """État du système de gouvernance étendu"""
    # Champs existants (compatibilité)
    current_state: str
    mode: str
    last_decision_id: Optional[str] = None
    contradiction_index: float
    ml_signals_timestamp: Optional[str] = None
    active_policy: Optional[Dict[str, Any]] = None
    pending_approvals: int
    next_update_time: Optional[str] = None
    etag: Optional[str] = None  # ETag pour concurrency control
    auto_unfreeze_at: Optional[str] = None  # TTL auto-unfreeze timestamp

    # NOUVEAUX CHAMPS - Unification
    scores: Optional[CanonicalScores] = Field(None, description="Scores canoniques unifiés")
    phase: Optional[PhaseInfo] = Field(None, description="Phase de rotation actuelle")
    exec: Optional[ExecutionPressure] = Field(None, description="Pression d'exécution")
    signals: Optional[UnifiedSignals] = Field(None, description="Bus de signaux unifié")
    portfolio: Optional[Dict[str, Any]] = Field(None, description="État du portefeuille")
    suggestion: Optional[SuggestionIA] = Field(None, description="Suggestion IA canonique")


class SetModeRequest(BaseModel):
    """Request to change governance mode"""
    mode: str = Field(..., description="Governance mode: manual, ai_assisted, full_ai, freeze")
    reason: str = Field(default="Mode change via UI", max_length=500, description="Reason for mode change")

    @validator('mode')
    def validate_mode(cls, v):
        v = v.lower()
        if v not in ("manual", "ai_assisted", "full_ai", "freeze"):
            raise ValueError("Mode must be one of: manual, ai_assisted, full_ai, freeze")
        return v


class ProposeDecisionRequest(BaseModel):
    """Request to propose a new rebalancing decision"""
    targets: List[Dict[str, Any]] = Field(
        default=[{"symbol": "BTC", "weight": 0.6}, {"symbol": "ETH", "weight": 0.4}],
        description="Target allocations"
    )
    reason: str = Field(default="Test proposal from UI", max_length=500, description="Proposal reason")
    force_override_cooldown: bool = Field(default=False, description="Override cooldown period")


class ReviewPlanRequest(BaseModel):
    """Request to review a governance plan"""
    reviewed_by: str = Field(default="system", max_length=100, description="Reviewer identifier")
    notes: str = Field(default="Reviewed via API", max_length=500, description="Review notes")


class CancelPlanRequest(BaseModel):
    """Request to cancel a governance plan"""
    cancelled_by: str = Field(default="system", max_length=100, description="Canceller identifier")
    reason: str = Field(default="Cancelled via API", max_length=500, description="Cancellation reason")


class ValidateAllocationRequest(BaseModel):
    """Request to validate an allocation change"""
    current_weights: Dict[str, float] = Field(default_factory=dict, description="Current portfolio weights")
    target_weights: Dict[str, float] = Field(default_factory=dict, description="Target portfolio weights")
    portfolio_usd: float = Field(default=100000, ge=0, description="Total portfolio value in USD")


class ApprovalRequest(BaseModel):
    """Requête d'approbation d'une décision"""
    decision_id: str
    approved: bool
    reason: Optional[str] = None


class UnifiedApprovalRequest(BaseModel):
    """Requête d'approbation unifiée pour décisions et plans"""
    resource_type: str = Field(..., pattern="^(decision|plan)$", description="Type: decision ou plan")
    approved: bool = Field(..., description="Approuver (true) ou rejeter (false)")
    approved_by: str = Field(default="system", description="Identifiant de l'approbateur")
    reason: Optional[str] = Field(None, max_length=500, description="Raison de l'approbation/rejet")
    notes: Optional[str] = Field(None, max_length=500, description="Notes additionnelles")


class FreezeRequest(BaseModel):
    """Requête de gel du système avec TTL"""
    reason: str = Field(..., max_length=140, description="Raison du freeze")
    ttl_minutes: int = Field(default=360, ge=15, le=1440, description="TTL auto-unfreeze [15min-24h]")
    source_alert_id: Optional[str] = Field(None, description="ID alerte source si applicable")


class ApplyPolicyRequest(BaseModel):
    """Requete d'application de policy depuis alerte - NOUVEAU"""
    mode: str = Field(..., description="Mode de policy")
    cap_daily: float = Field(..., ge=-1.0, le=1.0, description="Cap quotidien brut (sera clampe +/-20%)")
    ramp_hours: int = Field(..., ge=1, le=72, description="Ramping [1-72h]")
    reason: str = Field(..., max_length=140, description="Raison du changement")
    source_alert_id: Optional[str] = Field(None, description="ID de l'alerte source")
    min_trade: float = Field(default=100.0, ge=10.0, description="Trade minimum en USD")
    slippage_limit_bps: int = Field(default=50, ge=1, le=500, description="Limite slippage [1-500 bps]")
    signals_ttl_seconds: int = Field(default=1800, ge=60, le=7200, description="TTL signaux [60-7200s]")
    plan_cooldown_hours: int = Field(default=24, ge=1, le=168, description="Cooldown plans [1-168h]")
    no_trade_threshold_pct: float = Field(default=0.02, description="No-trade zone brute (sera clampee)")
    execution_cost_bps: int = Field(default=15, ge=-1000, le=1000, description="Cout brut en bps (sera clampe [0-100])")
    notes: Optional[str] = Field(default=None, max_length=280, description="Notes additionnelles")

    @validator('mode')
    def validate_mode(cls, v):
        allowed_modes = ["Slow", "Normal", "Aggressive"]
        if v not in allowed_modes:
            raise ValueError(f"Mode must be one of {allowed_modes}. Use freeze endpoint for Freeze mode.")
        return v
