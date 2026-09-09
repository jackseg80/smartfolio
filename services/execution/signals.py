"""
ML Signals Module - Gestion des signaux ML pour Governance Engine

Ce module gère:
- Modèle MLSignals (structure des signaux)
- Extraction des signaux depuis le ML Orchestrator (réels ou fallback)
- Calcul de l'index de contradiction
- Calcul de la confiance globale

Phase 1: Signaux ML centralisés pour décisions governance
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class MLSignals(BaseModel):
    """Signaux ML agrégés pour la prise de décision"""
    as_of: datetime = Field(default_factory=datetime.now, description="Signals timestamp")
    available: bool = Field(default=False, description="Whether the signals are backed by verified observations")
    unavailable_reason: Optional[str] = Field(default="Signals have not been loaded", description="Why verified signals are unavailable")

    # Missing data stays missing. UI placeholders must never become decisions.
    volatility: Dict[str, float] = Field(
        default_factory=dict,
        description="Volatility forecast per asset"
    )
    regime: Dict[str, float] = Field(
        default_factory=dict,
        description="Regime probabilities"
    )
    correlation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Correlation metrics"
    )
    sentiment: Dict[str, float] = Field(
        default_factory=dict,
        description="Sentiment indicators"
    )

    # Signaux dérivés
    decision_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Global decision score")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Decision confidence")
    contradiction_index: float = Field(default=1.0, ge=0.0, le=1.0, description="Contradiction index")
    blended_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Blended Decision Score (0-100) if available")

    # Metadata
    ttl_seconds: int = Field(default=1800, ge=60, description="Signals TTL")
    sources_used: List[str] = Field(default_factory=list, description="ML sources used")


class SignalExtractor:
    """
    Classe utilitaire pour extraire les signaux ML depuis différentes sources
    """

    @staticmethod
    def extract_volatility_signals(ml_status: Dict[str, Any]) -> Dict[str, float]:
        """A model-status payload does not contain a volatility observation."""
        return {}

    @staticmethod
    def extract_regime_signals(ml_status: Dict[str, Any]) -> Dict[str, float]:
        """A loaded-model flag is not a regime inference."""
        return {}

    @staticmethod
    def extract_correlation_signals(ml_status: Dict[str, Any]) -> Dict[str, Any]:
        """Cache metadata is not a correlation observation or forecast."""
        return {}

    @staticmethod
    def extract_sentiment_signals(ml_status: Dict[str, Any]) -> Dict[str, float]:
        """Model status does not contain a sentiment observation."""
        return {}

    @staticmethod
    def compute_contradiction_index(ml_status: Dict[str, Any]) -> float:
        """
        Centralise le calcul de contradiction depuis composite-score-v2.js

        Logique basée sur :
        - Conflits vol/regime (high vol + bull regime = contradiction)
        - Sentiment vs regime (extreme fear + bull = contradiction)
        - Corrélations vs diversification
        """
        try:
            contradictions = 0.0
            evaluated_weight = 0.0

            # Check 1: Volatilité vs Régime
            vol_signals = SignalExtractor.extract_volatility_signals(ml_status)
            regime_signals = SignalExtractor.extract_regime_signals(ml_status)

            regime_bull = regime_signals.get("bull", 0.0) > 0.6
            if vol_signals and regime_signals:
                vol_high = any(v > 0.15 for v in vol_signals.values())
                if vol_high and regime_bull:
                    contradictions += 0.3
                evaluated_weight += 0.3
            else:
                vol_high = False

            # Check 2: Sentiment vs Régime
            sentiment_data = SignalExtractor.extract_sentiment_signals(ml_status)
            fear_greed = sentiment_data.get("fear_greed")
            if isinstance(fear_greed, (int, float)) and regime_signals:
                sentiment_extreme_fear = fear_greed < 25
                sentiment_extreme_greed = fear_greed > 75
                if (sentiment_extreme_greed and not regime_bull) or (sentiment_extreme_fear and regime_bull):
                    contradictions += 0.25
                evaluated_weight += 0.25

            # Check 3: Corrélations élevées (risque systémique)
            corr_data = SignalExtractor.extract_correlation_signals(ml_status)
            avg_correlation = corr_data.get("avg_correlation")
            high_correlation = isinstance(avg_correlation, (int, float)) and avg_correlation > 0.7
            if isinstance(avg_correlation, (int, float)):
                if high_correlation:
                    contradictions += 0.2
                evaluated_weight += 0.2

            # Normaliser [0-1]
            contradiction_index = min(1.0, contradictions / evaluated_weight) if evaluated_weight > 0 else 1.0

            logger.debug(f"Contradiction index computed: {contradiction_index:.3f} "
                        f"(vol_high={vol_high}, regime_bull={regime_bull}, high_corr={high_correlation})")

            return contradiction_index

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Data error computing contradiction index: {e}")
            return 1.0
        except Exception as e:
            logger.exception(f"Unexpected error computing contradiction index: {e}")
            return 1.0


class RealSignalExtractor:
    """
    Classe pour extraire les signaux depuis les vraies prédictions ML (orchestrator)
    """

    @staticmethod
    def extract_volatility_signals(ml_predictions: Dict[str, Any]) -> Dict[str, float]:
        """Extrait les signaux de volatilité depuis les vraies prédictions ML"""
        try:
            volatility_data = ml_predictions.get('models', {}).get('volatility', {})
            if not volatility_data:
                return {}

            volatility_signals = {}
            for symbol, predictions in volatility_data.items():
                if isinstance(predictions, dict):
                    # Take average volatility across horizons
                    vol_values = []
                    for horizon_key, horizon_data in predictions.items():
                        if isinstance(horizon_data, dict) and 'volatility_forecast' in horizon_data:
                            vol_values.append(horizon_data['volatility_forecast'])

                    if vol_values:
                        volatility_signals[symbol] = sum(vol_values) / len(vol_values)

            logger.debug(f"Extracted real volatility signals: {volatility_signals}")
            return volatility_signals

        except Exception as e:
            logger.warning(f"Error extracting real volatility signals: {e}")
            return {}

    @staticmethod
    def extract_regime_signals(ml_predictions: Dict[str, Any]) -> Dict[str, float]:
        """Extrait les signaux de régime depuis les vraies prédictions ML"""
        try:
            regime_data = ml_predictions.get('models', {}).get('regime', {})
            if not regime_data or regime_data.get('available') is False:
                return {}

            # Map regime names to probabilities
            current_regime = regime_data.get('current_regime')
            regime_prob = regime_data.get('regime_probability')
            if not isinstance(regime_prob, (int, float)) or not 0 <= regime_prob <= 1:
                return {}

            # Convert regime to our expected format (canonical names from regime_constants)
            from services.regime_constants import normalize_regime_name
            current_regime = normalize_regime_name(current_regime)

            regime_type = {
                'Bear Market': 'bear',
                'Correction': 'neutral',
                'Bull Market': 'bull',
                'Expansion': 'bull',
            }.get(current_regime)

            if regime_type is None:
                return {}
            regime_signals = {regime_type: float(regime_prob)}

            logger.debug(f"Extracted real regime signals: {regime_signals}")
            return regime_signals

        except Exception as e:
            logger.warning(f"Error extracting real regime signals: {e}")
            return {}

    @staticmethod
    def extract_correlation_signals(ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les signaux de corrélation depuis les vraies prédictions ML"""
        try:
            correlation_data = ml_predictions.get('models', {}).get('correlation', {})
            logger.debug(f"[DEBUG CORRELATION] Raw correlation_data: {correlation_data}")
            if not correlation_data or correlation_data.get('available') is False:
                return {}

            # PRIORITY 1: Use pre-calculated aggregate fields if they exist (from orchestrator.py)
            if 'avg_correlation' in correlation_data and 'systemic_risk' in correlation_data:
                avg_correlation = correlation_data['avg_correlation']
                systemic_risk_level = correlation_data['systemic_risk']

                if not isinstance(avg_correlation, (int, float)):
                    return {}

                logger.debug(f"Extracted correlation signals from aggregates: avg_corr={avg_correlation}, risk={systemic_risk_level}")
                return {
                    "avg_correlation": avg_correlation,
                    "systemic_risk": systemic_risk_level
                }

            # FALLBACK: Calculate from pair-wise correlations (legacy path)
            correlations = []
            for pair, corr_info in correlation_data.items():
                # Skip aggregate fields if mixed in
                if pair in ['avg_correlation', 'systemic_risk']:
                    continue
                if isinstance(corr_info, dict):
                    current_corr = corr_info.get('current_correlation')
                    forecast_corr = corr_info.get('forecast_correlation')
                    values = [value for value in (current_corr, forecast_corr) if isinstance(value, (int, float))]
                    if values:
                        correlations.append(max(values))

            if correlations:
                avg_correlation = sum(correlations) / len(correlations)
                systemic_risk_level = "high" if avg_correlation > 0.7 else "medium" if avg_correlation > 0.5 else "low"
            else:
                return {}

            logger.debug(f"Extracted real correlation signals from pairs: avg_corr={avg_correlation}, risk={systemic_risk_level}")
            return {
                "avg_correlation": avg_correlation,
                "systemic_risk": systemic_risk_level
            }

        except Exception as e:
            logger.warning(f"Error extracting real correlation signals: {e}")
            return {}

    @staticmethod
    def extract_sentiment_signals(ml_predictions: Dict[str, Any]) -> Dict[str, float]:
        """Extrait les signaux de sentiment depuis les vraies prédictions ML"""
        try:
            sentiment_data = ml_predictions.get('models', {}).get('sentiment', {})
            if not sentiment_data or sentiment_data.get('available') is False:
                return {}

            # Extract sentiment scores
            sentiment_scores = []
            fear_greed_values = []

            for symbol, sentiment_info in sentiment_data.items():
                if isinstance(sentiment_info, dict):
                    if 'sentiment_score' in sentiment_info:
                        sentiment_scores.append(sentiment_info['sentiment_score'])
                    if 'fear_greed_index' in sentiment_info:
                        fear_greed_values.append(sentiment_info['fear_greed_index'])

            # Calculate averages
            if not sentiment_scores and not fear_greed_values:
                return {}

            sentiment_signals = {}
            if fear_greed_values:
                sentiment_signals["fear_greed"] = sum(fear_greed_values) / len(fear_greed_values)
            if sentiment_scores:
                sentiment_signals["sentiment_score"] = sum(sentiment_scores) / len(sentiment_scores)

            logger.debug(f"Extracted real sentiment signals: {sentiment_signals}")
            return sentiment_signals

        except Exception as e:
            logger.warning(f"Error extracting real sentiment signals: {e}")
            return {}

    @staticmethod
    def calculate_confidence(ml_predictions: Dict[str, Any]) -> float:
        """Calcule la confiance globale depuis les vraies prédictions ML"""
        try:
            confidence_scores = ml_predictions.get('confidence_scores', {})
            if not confidence_scores:
                return 0.0

            # Weight different model confidences
            model_weights = {
                'volatility': 0.25,
                'sentiment': 0.20,
                'regime': 0.30,
                'correlation': 0.25
            }

            weighted_confidence = 0.0
            total_weight = 0.0

            for model, confidence in confidence_scores.items():
                if model in model_weights and isinstance(confidence, (int, float)):
                    weighted_confidence += confidence * model_weights[model]
                    total_weight += model_weights[model]

            if total_weight > 0:
                final_confidence = weighted_confidence / total_weight
            else:
                overall = confidence_scores.get('overall')
                final_confidence = overall if isinstance(overall, (int, float)) else 0.0

            logger.debug(f"Calculated real confidence: {final_confidence:.3f}")
            return min(1.0, max(0.0, final_confidence))

        except Exception as e:
            logger.warning(f"Error calculating real confidence: {e}")
            return 0.0

    @staticmethod
    def compute_contradiction_index(ml_predictions: Dict[str, Any]) -> float:
        """Calcule l'index de contradiction depuis les vraies prédictions ML"""
        try:
            ensemble = ml_predictions.get('ensemble', {})
            if not ensemble:
                return 1.0

            # Use ensemble conflicting signals
            conflicting_signals = ensemble.get('conflicting_signals', [])
            consensus_strength = ensemble.get('consensus_strength')
            if not isinstance(consensus_strength, (int, float)):
                return 1.0

            # Base contradiction from conflicting signals
            base_contradiction = len(conflicting_signals) / 4.0  # Normalize by max models

            # Adjust by consensus weakness
            consensus_contradiction = 1.0 - consensus_strength

            # Combine both measures
            contradiction_index = (base_contradiction * 0.6) + (consensus_contradiction * 0.4)

            # Cap at reasonable levels
            contradiction_index = min(1.0, max(0.0, contradiction_index))

            logger.debug(f"Computed real contradiction index: {contradiction_index:.3f} "
                        f"(conflicts: {len(conflicting_signals)}, consensus: {consensus_strength:.2f})")

            return contradiction_index

        except Exception as e:
            logger.warning(f"Error computing real contradiction index: {e}")
            return 1.0


def create_default_signals(reason: str = "Verified ML signals are unavailable") -> MLSignals:
    """Create an explicit unavailable state; never synthesize market signals."""
    return MLSignals(
        as_of=datetime.now(),
        available=False,
        unavailable_reason=reason,
        volatility={},
        regime={},
        correlation={},
        sentiment={},
        decision_score=0.0,
        confidence=0.0,
        contradiction_index=1.0,
        sources_used=[]
    )
