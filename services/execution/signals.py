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
import time

logger = logging.getLogger(__name__)


class MLSignals(BaseModel):
    """Signaux ML agrégés pour la prise de décision"""
    as_of: datetime = Field(default_factory=datetime.now, description="Signals timestamp")

    # Signaux individuels - with sensible defaults to avoid 0% display issues
    volatility: Dict[str, float] = Field(
        default_factory=lambda: {"BTC": 0.35, "ETH": 0.45},
        description="Volatility forecast per asset"
    )
    regime: Dict[str, float] = Field(
        default_factory=lambda: {"bull": 0.5, "bear": 0.25, "neutral": 0.25},
        description="Regime probabilities"
    )
    correlation: Dict[str, Any] = Field(
        default_factory=lambda: {"avg_correlation": 0.5, "systemic_risk": "medium"},
        description="Correlation metrics - avg_correlation should be 0.4-0.7 for crypto"
    )
    sentiment: Dict[str, float] = Field(
        default_factory=lambda: {"fear_greed": 50, "sentiment_score": 0.0},
        description="Sentiment indicators"
    )

    # Signaux dérivés
    decision_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Global decision score")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Decision confidence")
    contradiction_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Contradiction index")
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
        """Extrait les signaux de volatilité depuis le ML status (fallback)"""
        try:
            pipeline = ml_status.get("pipeline_status", {})
            vol_models = pipeline.get("volatility_models", {})

            # Simulation basée sur le nombre de modèles chargés
            loaded_count = vol_models.get("models_loaded", 0)
            if loaded_count > 0:
                return {
                    "BTC": 0.08 + (loaded_count * 0.005),
                    "ETH": 0.12 + (loaded_count * 0.007),
                    "SOL": 0.15 + (loaded_count * 0.010)
                }
            return {}

        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Data parsing error extracting volatility signals: {e}")
            return {}

    @staticmethod
    def extract_regime_signals(ml_status: Dict[str, Any]) -> Dict[str, float]:
        """Extrait les signaux de régime depuis le ML status (fallback)"""
        try:
            pipeline = ml_status.get("pipeline_status", {})
            regime_models = pipeline.get("regime_models", {})

            if regime_models.get("model_loaded", False):
                return {
                    "bull": 0.4,
                    "neutral": 0.35,
                    "bear": 0.25
                }
            return {"neutral": 1.0}

        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Data parsing error extracting regime signals: {e}")
            return {"neutral": 1.0}

    @staticmethod
    def extract_correlation_signals(ml_status: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les signaux de corrélation depuis le ML status (fallback)"""
        try:
            pipeline = ml_status.get("pipeline_status", {})
            cache_stats = pipeline.get("cache_stats", {})

            models_loaded = cache_stats.get("cached_models", 0)
            avg_correlation = min(0.8, 0.4 + (models_loaded * 0.05))

            # Ensure avg_correlation is never 0 or None (fallback to 0.4 minimum)
            avg_correlation = max(0.4, avg_correlation) if avg_correlation else 0.4

            return {
                "avg_correlation": avg_correlation,
                "systemic_risk": "medium" if avg_correlation > 0.6 else "low"
            }

        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Data parsing error extracting correlation signals: {e}")
            return {"avg_correlation": 0.5, "systemic_risk": "unknown"}

    @staticmethod
    def extract_sentiment_signals(ml_status: Dict[str, Any]) -> Dict[str, float]:
        """Extrait les signaux de sentiment (Fear & Greed, etc.) (fallback)"""
        try:
            # Simulation stable basée sur l'heure pour éviter le bruit
            hour_seed = int(time.time() / 3600) % 100
            fear_greed = 45 + (hour_seed % 30)  # 45-75, stable par heure

            return {
                "fear_greed": fear_greed,
                "sentiment_score": (fear_greed - 50) / 50  # [-1, 1]
            }

        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error extracting sentiment signals: {e}")
            return {"fear_greed": 50, "sentiment_score": 0.0}

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
            total_checks = 0.0

            # Check 1: Volatilité vs Régime
            vol_signals = SignalExtractor.extract_volatility_signals(ml_status)
            regime_signals = SignalExtractor.extract_regime_signals(ml_status)

            vol_high = any(v > 0.15 for v in vol_signals.values())
            regime_bull = regime_signals.get("bull", 0.0) > 0.6

            if vol_high and regime_bull:
                contradictions += 0.3  # High vol + bull regime = contradiction
            total_checks += 1.0

            # Check 2: Sentiment vs Régime
            sentiment_data = SignalExtractor.extract_sentiment_signals(ml_status)
            sentiment_extreme_fear = sentiment_data.get("fear_greed", 50) < 25
            sentiment_extreme_greed = sentiment_data.get("fear_greed", 50) > 75

            if (sentiment_extreme_greed and not regime_bull) or (sentiment_extreme_fear and regime_bull):
                contradictions += 0.25
            total_checks += 1.0

            # Check 3: Corrélations élevées (risque systémique)
            corr_data = SignalExtractor.extract_correlation_signals(ml_status)
            high_correlation = corr_data.get("avg_correlation", 0.0) > 0.7

            if high_correlation:
                contradictions += 0.2  # Faible diversification
            total_checks += 1.0

            # Normaliser [0-1]
            contradiction_index = min(1.0, contradictions / max(1.0, total_checks)) if total_checks > 0 else 0.0

            logger.debug(f"Contradiction index computed: {contradiction_index:.3f} "
                        f"(vol_high={vol_high}, regime_bull={regime_bull}, high_corr={high_correlation})")

            return contradiction_index

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Data error computing contradiction index: {e}")
            return 0.5  # Valeur neutre par défaut
        except Exception as e:
            logger.exception(f"Unexpected error computing contradiction index: {e}")
            return 0.5


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
            if not regime_data:
                return {"neutral": 1.0}

            # Map regime names to probabilities
            current_regime = regime_data.get('current_regime', 'unknown')
            regime_prob = regime_data.get('regime_probability', 0.5)

            # Convert regime to our expected format (canonical names from regime_constants)
            from services.regime_constants import normalize_regime_name
            current_regime = normalize_regime_name(current_regime)

            regime_mapping = {
                'Bear Market': {'bull': 0.05, 'neutral': 0.15, 'bear': 0.8},
                'Correction': {'bull': 0.2, 'neutral': 0.4, 'bear': 0.4},
                'Bull Market': {'bull': 0.7, 'neutral': 0.2, 'bear': 0.1},
                'Expansion': {'bull': 0.85, 'neutral': 0.1, 'bear': 0.05},
            }

            if current_regime in regime_mapping:
                base_probs = regime_mapping[current_regime]
                # Adjust by actual confidence
                regime_signals = {}
                for regime_type, base_prob in base_probs.items():
                    regime_signals[regime_type] = base_prob * regime_prob + (1 - regime_prob) * 0.33
            else:
                regime_signals = {"bull": 0.33, "neutral": 0.34, "bear": 0.33}

            logger.debug(f"Extracted real regime signals: {regime_signals}")
            return regime_signals

        except Exception as e:
            logger.warning(f"Error extracting real regime signals: {e}")
            return {"neutral": 1.0}

    @staticmethod
    def extract_correlation_signals(ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les signaux de corrélation depuis les vraies prédictions ML"""
        try:
            correlation_data = ml_predictions.get('models', {}).get('correlation', {})
            logger.debug(f"[DEBUG CORRELATION] Raw correlation_data: {correlation_data}")
            if not correlation_data:
                logger.warning("[DEBUG CORRELATION] correlation_data is empty, using fallback 0.5")
                return {"avg_correlation": 0.5, "systemic_risk": "medium"}

            # PRIORITY 1: Use pre-calculated aggregate fields if they exist (from orchestrator.py)
            if 'avg_correlation' in correlation_data and 'systemic_risk' in correlation_data:
                avg_correlation = correlation_data['avg_correlation']
                systemic_risk_level = correlation_data['systemic_risk']

                # Ensure avg_correlation is never 0 or None (fallback to 0.5 minimum)
                avg_correlation = max(0.4, avg_correlation) if avg_correlation else 0.5

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
                    current_corr = corr_info.get('current_correlation', 0.5)
                    forecast_corr = corr_info.get('forecast_correlation', current_corr)
                    correlations.append(max(current_corr, forecast_corr))

            if correlations:
                avg_correlation = sum(correlations) / len(correlations)
                systemic_risk_level = "high" if avg_correlation > 0.7 else "medium" if avg_correlation > 0.5 else "low"
            else:
                avg_correlation = 0.5
                systemic_risk_level = "medium"

            # Ensure avg_correlation is never 0 or None (fallback to 0.5 minimum)
            avg_correlation = max(0.4, avg_correlation) if avg_correlation else 0.5

            logger.debug(f"Extracted real correlation signals from pairs: avg_corr={avg_correlation}, risk={systemic_risk_level}")
            return {
                "avg_correlation": avg_correlation,
                "systemic_risk": systemic_risk_level
            }

        except Exception as e:
            logger.warning(f"Error extracting real correlation signals: {e}")
            return {"avg_correlation": 0.5, "systemic_risk": "medium"}

    @staticmethod
    def extract_sentiment_signals(ml_predictions: Dict[str, Any]) -> Dict[str, float]:
        """Extrait les signaux de sentiment depuis les vraies prédictions ML"""
        try:
            sentiment_data = ml_predictions.get('models', {}).get('sentiment', {})
            if not sentiment_data:
                return {"fear_greed": 50, "sentiment_score": 0.0}

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
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            avg_fear_greed = sum(fear_greed_values) / len(fear_greed_values) if fear_greed_values else 50.0

            sentiment_signals = {
                "fear_greed": avg_fear_greed,
                "sentiment_score": avg_sentiment
            }

            logger.debug(f"Extracted real sentiment signals: {sentiment_signals}")
            return sentiment_signals

        except Exception as e:
            logger.warning(f"Error extracting real sentiment signals: {e}")
            return {"fear_greed": 50, "sentiment_score": 0.0}

    @staticmethod
    def calculate_confidence(ml_predictions: Dict[str, Any]) -> float:
        """Calcule la confiance globale depuis les vraies prédictions ML"""
        try:
            confidence_scores = ml_predictions.get('confidence_scores', {})
            if not confidence_scores:
                return 0.5

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
                final_confidence = confidence_scores.get('overall', 0.5)

            logger.debug(f"Calculated real confidence: {final_confidence:.3f}")
            return min(1.0, max(0.0, final_confidence))

        except Exception as e:
            logger.warning(f"Error calculating real confidence: {e}")
            return 0.5

    @staticmethod
    def compute_contradiction_index(ml_predictions: Dict[str, Any]) -> float:
        """Calcule l'index de contradiction depuis les vraies prédictions ML"""
        try:
            ensemble = ml_predictions.get('ensemble', {})
            if not ensemble:
                return 0.3  # Default moderate contradiction

            # Use ensemble conflicting signals
            conflicting_signals = ensemble.get('conflicting_signals', [])
            consensus_strength = ensemble.get('consensus_strength', 0.5)

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
            return 0.3


def create_default_signals() -> MLSignals:
    """Crée des signaux ML par défaut pour les cas d'erreur"""
    return MLSignals(
        as_of=datetime.now(),
        volatility={"BTC": 0.35, "ETH": 0.45},
        regime={"bull": 0.5, "bear": 0.25, "neutral": 0.25},
        correlation={"avg_correlation": 0.5, "systemic_risk": "medium"},
        sentiment={"fear_greed": 50, "sentiment_score": 0.0},
        decision_score=0.5,
        confidence=0.5,
        contradiction_index=0.3,
        sources_used=["fallback_default"]
    )
