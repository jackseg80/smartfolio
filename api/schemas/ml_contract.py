"""
Contrat prédictif unifié pour les modèles ML
Schémas standardisés pour entrées/sorties ML avec gestion d'incertitude et métadonnées
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
from enum import Enum


# === TYPES DE BASE ===

class ModelType(str, Enum):
    """Types de modèles supportés"""
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    REGIME = "regime"
    CORRELATION = "correlation"
    RISK = "risk"
    BLENDED = "blended"


class Horizon(str, Enum):
    """Horizons temporels standardisés"""
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    D7 = "7d"
    D30 = "30d"
    D90 = "90d"


class Availability(str, Enum):
    AVAILABLE = "Available"
    PARTIAL = "Partial"
    UNAVAILABLE = "Unavailable"
    EXPERIMENTAL = "Experimental"
    REJECTED = "Rejected"


class Provenance(BaseModel):
    provider: Optional[str] = None
    dataset_id: Optional[str] = None
    evaluation_dataset_id: Optional[str] = None
    observation_sha256: Optional[str] = None
    code_version: Optional[str] = None
    artifact_sha256: Optional[str] = None
    training_start: Optional[datetime] = None
    training_end: Optional[datetime] = None
    adjustment_policy: Optional[str] = None
    method: Optional[str] = None


class ValidationStatus(BaseModel):
    state: Literal["descriptive", "retrospectively_validated", "unvalidated", "rejected", "not_evaluable"] = "not_evaluable"
    reason: str
    protocol_id: Optional[str] = None
    evaluated_at: Optional[datetime] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ConfidenceLevel(str, Enum):
    """Confidence levels for intervals."""
    LOW = "80"
    MEDIUM = "90"
    HIGH = "95"
    VERY_HIGH = "99"


# === METADATA ET QUALITE ===

class ModelMetadata(BaseModel):
    """Metadata for the model used."""
    model_config = {"protected_namespaces": ()}

    name: str = Field(description="Model name")
    version: str = Field(description="Model version (semver)")
    trained_at: Optional[datetime] = Field(None, description="Training date")
    features_used: Optional[List[str]] = Field(None, description="Features used")
    model_type: ModelType = Field(description="Model type")
    horizon: Optional[Horizon] = Field(None, description="Predicted horizon")


class UncertaintyMeasures(BaseModel):
    """Uncertainty measures standardisées"""
    std: Optional[float] = Field(None, description="Prediction standard deviation")
    lower_bound: Optional[float] = Field(None, description="Lower bound (PI)")
    upper_bound: Optional[float] = Field(None, description="Upper bound (PI)")
    confidence_level: Optional[ConfidenceLevel] = Field(None, description="Confidence level")
    nominal_coverage: Optional[float] = Field(None, ge=0, le=1, description="Nominal prediction interval coverage")
    confirmation_coverage: Optional[float] = Field(None, ge=0, le=1, description="Observed coverage on reserved confirmation data")
    calibration_score: Optional[float] = Field(None, ge=0, le=1, description="Calibration score [0,1]")


class QualityMetrics(BaseModel):
    """Prediction quality metrics."""
    model_config = {"protected_namespaces": ()}

    confidence: Optional[float] = Field(None, ge=0, le=1, description="Evaluated confidence [0,1]; absent when not evaluated")
    data_freshness: Optional[float] = Field(None, description="Data freshness (hours)")
    feature_coverage: Optional[float] = Field(None, ge=0, le=1, description="Feature coverage [0,1]")
    model_health: Optional[float] = Field(None, ge=0, le=1, description="Model health [0,1]")


# === REQUETES UNIFIEES ===

class UnifiedMLRequest(BaseModel):
    """Unified ML request."""
    model_config = {"protected_namespaces": ()}

    assets: List[str] = Field(max_length=50, description="Assets to analyze")
    model_type: ModelType = Field(description="Requested prediction type")
    horizon: Optional[Horizon] = Field(None, description="Time horizon")
    market: Literal["crypto", "stocks"] = "crypto"
    source: str = Field("cointracking", min_length=1, max_length=100, description="Selected portfolio source")

    # Options de qualité
    include_uncertainty: bool = Field(False, description="Include uncertainty measures")
    include_metadata: bool = Field(False, description="Include model metadata")
    confidence_threshold: float = Field(0.5, ge=0, le=1, description="Minimum confidence threshold")

    # Paramètres contextuels
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    cache_ttl: Optional[int] = Field(300, description="Cache TTL in seconds")


class BatchMLRequest(BaseModel):
    """Requête ML batch pour multiple modèles/horizons"""
    assets: List[str] = Field(max_length=20, description="Assets to analyze")
    requests: List[Dict[str, Any]] = Field(max_length=10, description="Multiple requests")
    global_options: Optional[Dict[str, Any]] = Field(None, description="Global options")


# === REPONSES UNIFIEES ===

class UnifiedPrediction(BaseModel):
    """Single prediction with uncertainty."""
    asset: str = Field(description="Target asset")
    value: Optional[Union[float, str, Dict[str, Any]]] = Field(None, description="Observed or predicted value; null when unavailable")
    market: Literal["crypto", "stocks"] = "crypto"
    nature: Literal["diagnostic", "forecast"] = "forecast"
    target: str = "unspecified"
    horizon: Optional[Horizon] = None
    unit: Optional[str] = None
    availability: Availability = Availability.UNAVAILABLE
    reason: str = "No verified result is available"
    data_as_of: Optional[datetime] = None
    target_date: Optional[datetime] = None
    validation: ValidationStatus = Field(default_factory=lambda: ValidationStatus(reason="No evaluation record"))
    provenance: Provenance = Field(default_factory=Provenance)

    # Incertitude (optionnel)
    uncertainty: Optional[UncertaintyMeasures] = Field(None, description="Uncertainty measures")

    # Qualité
    quality: QualityMetrics = Field(default_factory=QualityMetrics, description="Measured quality metrics")

    # Métadonnées (optionnel)
    metadata: Optional[ModelMetadata] = Field(None, description="Model metadata")


class UnifiedMLResponse(BaseModel):
    """Réponse ML unifiée"""
    model_config = {"protected_namespaces": ()}

    success: bool = Field(True, description="Success status")
    model_type: ModelType = Field(description="Model type used")
    horizon: Optional[Horizon] = Field(None, description="Predicted horizon")

    # Données principales
    predictions: List[UnifiedPrediction] = Field(description="Predictions per asset")

    # Agrégations (optionnel)
    aggregated: Optional[Dict[str, Any]] = Field(None, description="Aggregated metrics")
    user_id: Optional[str] = None
    source: Optional[str] = None

    # Contexte global
    processed_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    cache_hit: bool = Field(False, description="Result served from cache")
    processing_time_ms: Optional[float] = Field(None, description="Processing time")

    # Gestion d'erreurs
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    failed_assets: List[str] = Field(default_factory=list, description="Failed assets")


class BatchMLResponse(BaseModel):
    """Réponse ML batch"""
    success: bool = Field(True, description="Global success")
    responses: Dict[str, UnifiedMLResponse] = Field(description="Responses by request")
    global_metadata: Dict[str, Any] = Field(default_factory=dict, description="Global metadata")
    total_processing_time_ms: Optional[float] = Field(None, description="Total processing time")


# === SCHEMAS SPECIFIQUES ===

class VolatilityPrediction(UnifiedPrediction):
    """Prédiction de volatilité avec spécificités"""
    annualized_vol: Optional[float] = Field(None, description="Annualized volatility")
    regime_context: Optional[str] = Field(None, description="Regime context")


class SentimentPrediction(UnifiedPrediction):
    """Prédiction de sentiment avec détails"""
    sentiment_breakdown: Optional[Dict[str, float]] = Field(None, description="Breakdown by source")
    fear_greed_index: Optional[float] = Field(None, ge=0, le=100, description="External Fear & Greed index")


class RiskScorePrediction(UnifiedPrediction):
    """Score de risque avec composantes"""
    components: Optional[Dict[str, float]] = Field(None, description="Score components")
    risk_category: Optional[str] = Field(None, description="Risk category")


# === SCHEMAS DE MONITORING ===

class ModelHealth(BaseModel):
    """Model health."""
    model_config = {"protected_namespaces": ()}

    model_name: str = Field(description="Model name")
    version: str = Field(description="Version")
    is_healthy: Optional[bool] = Field(None, description="Measured health status; unknown without inference evidence")
    last_prediction: Optional[datetime] = Field(None, description="Last prediction")
    error_rate_24h: Optional[float] = Field(None, description="24-hour error rate")
    avg_confidence: Optional[float] = Field(None, description="Average confidence")
    drift_score: Optional[float] = Field(None, description="Measured drift score")


class MLSystemHealth(BaseModel):
    """Overall ML system health."""
    overall_health: Optional[float] = Field(None, ge=0, le=1, description="Measured overall health [0,1]")
    models_status: List[ModelHealth] = Field(description="Status by model")
    system_metrics: Dict[str, Any] = Field(default_factory=dict, description="System metrics")
    last_check: datetime = Field(default_factory=datetime.now, description="Last check")


# === UTILITY FUNCTIONS ===

def create_fallback_response(
    model_type: ModelType,
    assets: List[str],
    error_msg: str = "Model unavailable"
) -> UnifiedMLResponse:
    """Create an explicit unavailable response without synthetic predictions."""
    return UnifiedMLResponse(
        success=False,
        model_type=model_type,
        predictions=[],
        warnings=[error_msg],
        failed_assets=assets
    )


def validate_prediction_quality(prediction: UnifiedPrediction, min_confidence: float = 0.3) -> bool:
    """Valider la qualité d'une prédiction"""
    return prediction.quality.confidence is not None and prediction.quality.confidence >= min_confidence
