"""
Tests for services/execution/signals.py
Covers MLSignals pydantic model, SignalExtractor, RealSignalExtractor, create_default_signals.
"""

import pytest
from datetime import datetime
from services.execution.signals import (
    MLSignals,
    SignalExtractor,
    RealSignalExtractor,
    create_default_signals,
)


# ── MLSignals ──────────────────────────────────────────────────────

class TestMLSignals:

    def test_default_creation(self):
        signals = MLSignals()
        assert signals.decision_score == 0.5
        assert signals.confidence == 0.5
        assert signals.contradiction_index == 0.0
        assert signals.ttl_seconds == 1800
        assert "BTC" in signals.volatility
        assert "bull" in signals.regime

    def test_custom_values(self):
        signals = MLSignals(
            decision_score=0.8,
            confidence=0.9,
            contradiction_index=0.3,
            sources_used=["volatility", "regime"],
        )
        assert signals.decision_score == 0.8
        assert signals.confidence == 0.9
        assert signals.contradiction_index == 0.3
        assert len(signals.sources_used) == 2

    def test_validation_decision_score_bounds(self):
        """decision_score must be 0-1."""
        with pytest.raises(Exception):
            MLSignals(decision_score=1.5)

    def test_validation_confidence_bounds(self):
        with pytest.raises(Exception):
            MLSignals(confidence=-0.1)

    def test_validation_contradiction_bounds(self):
        with pytest.raises(Exception):
            MLSignals(contradiction_index=2.0)

    def test_validation_ttl_minimum(self):
        with pytest.raises(Exception):
            MLSignals(ttl_seconds=10)  # < 60

    def test_blended_score_optional(self):
        signals = MLSignals()
        assert signals.blended_score is None

    def test_blended_score_bounded(self):
        signals = MLSignals(blended_score=75.0)
        assert signals.blended_score == 75.0

    def test_as_of_auto_set(self):
        signals = MLSignals()
        assert isinstance(signals.as_of, datetime)

    def test_default_volatility(self):
        signals = MLSignals()
        assert signals.volatility["BTC"] == 0.35
        assert signals.volatility["ETH"] == 0.45

    def test_default_regime(self):
        signals = MLSignals()
        assert signals.regime["bull"] == 0.5
        assert signals.regime["bear"] == 0.25

    def test_default_correlation(self):
        signals = MLSignals()
        assert signals.correlation["avg_correlation"] == 0.5

    def test_default_sentiment(self):
        signals = MLSignals()
        assert signals.sentiment["fear_greed"] == 50


# ── SignalExtractor ────────────────────────────────────────────────

class TestSignalExtractor:

    def test_extract_volatility_with_models(self):
        ml_status = {
            "pipeline_status": {
                "volatility_models": {"models_loaded": 3}
            }
        }
        result = SignalExtractor.extract_volatility_signals(ml_status)
        assert "BTC" in result
        assert "ETH" in result
        assert result["BTC"] > 0

    def test_extract_volatility_no_models(self):
        ml_status = {
            "pipeline_status": {
                "volatility_models": {"models_loaded": 0}
            }
        }
        result = SignalExtractor.extract_volatility_signals(ml_status)
        assert result == {}

    def test_extract_volatility_empty(self):
        result = SignalExtractor.extract_volatility_signals({})
        assert result == {}

    def test_extract_regime_loaded(self):
        ml_status = {
            "pipeline_status": {
                "regime_models": {"model_loaded": True}
            }
        }
        result = SignalExtractor.extract_regime_signals(ml_status)
        assert "bull" in result
        assert "neutral" in result
        assert "bear" in result

    def test_extract_regime_not_loaded(self):
        ml_status = {
            "pipeline_status": {
                "regime_models": {"model_loaded": False}
            }
        }
        result = SignalExtractor.extract_regime_signals(ml_status)
        assert result == {"neutral": 1.0}

    def test_extract_regime_empty(self):
        result = SignalExtractor.extract_regime_signals({})
        assert result == {"neutral": 1.0}

    def test_extract_correlation_with_cache(self):
        ml_status = {
            "pipeline_status": {
                "cache_stats": {"cached_models": 5}
            }
        }
        result = SignalExtractor.extract_correlation_signals(ml_status)
        assert "avg_correlation" in result
        assert "systemic_risk" in result
        assert result["avg_correlation"] >= 0.4

    def test_extract_correlation_high(self):
        ml_status = {
            "pipeline_status": {
                "cache_stats": {"cached_models": 10}  # High → corr > 0.6
            }
        }
        result = SignalExtractor.extract_correlation_signals(ml_status)
        assert result["avg_correlation"] > 0.6
        assert result["systemic_risk"] == "medium"

    def test_extract_correlation_empty(self):
        result = SignalExtractor.extract_correlation_signals({})
        assert result["avg_correlation"] >= 0.4  # Floor at 0.4

    def test_extract_sentiment(self):
        result = SignalExtractor.extract_sentiment_signals({})
        assert "fear_greed" in result
        assert "sentiment_score" in result
        assert 45 <= result["fear_greed"] <= 75

    def test_compute_contradiction_index_low(self):
        """Low volatility + neutral regime → low contradiction."""
        ml_status = {
            "pipeline_status": {
                "volatility_models": {"models_loaded": 1},
                "regime_models": {"model_loaded": False},
                "cache_stats": {"cached_models": 1},
            }
        }
        idx = SignalExtractor.compute_contradiction_index(ml_status)
        assert 0.0 <= idx <= 1.0

    def test_compute_contradiction_empty(self):
        idx = SignalExtractor.compute_contradiction_index({})
        assert 0.0 <= idx <= 1.0


# ── RealSignalExtractor ───────────────────────────────────────────

class TestRealSignalExtractor:

    def test_extract_volatility_real(self):
        ml_predictions = {
            "models": {
                "volatility": {
                    "BTC": {
                        "7d": {"volatility_forecast": 0.12},
                        "30d": {"volatility_forecast": 0.15},
                    }
                }
            }
        }
        result = RealSignalExtractor.extract_volatility_signals(ml_predictions)
        assert "BTC" in result
        assert result["BTC"] == pytest.approx(0.135)  # avg of 0.12, 0.15

    def test_extract_volatility_empty(self):
        result = RealSignalExtractor.extract_volatility_signals({})
        assert result == {}

    def test_extract_volatility_no_models(self):
        result = RealSignalExtractor.extract_volatility_signals({"models": {}})
        assert result == {}

    def test_extract_regime_real(self):
        ml_predictions = {
            "models": {
                "regime": {
                    "current_regime": "Bull Market",
                    "regime_probability": 0.8,
                }
            }
        }
        result = RealSignalExtractor.extract_regime_signals(ml_predictions)
        assert "bull" in result
        assert result["bull"] > result["bear"]

    def test_extract_regime_empty(self):
        result = RealSignalExtractor.extract_regime_signals({})
        assert result == {"neutral": 1.0}

    def test_extract_regime_unknown(self):
        ml_predictions = {
            "models": {
                "regime": {
                    "current_regime": "some_unknown_regime",
                    "regime_probability": 0.5,
                }
            }
        }
        result = RealSignalExtractor.extract_regime_signals(ml_predictions)
        # Unknown regime → equal distribution
        assert "bull" in result

    def test_extract_correlation_aggregates(self):
        ml_predictions = {
            "models": {
                "correlation": {
                    "avg_correlation": 0.65,
                    "systemic_risk": "medium",
                }
            }
        }
        result = RealSignalExtractor.extract_correlation_signals(ml_predictions)
        assert result["avg_correlation"] == 0.65
        assert result["systemic_risk"] == "medium"

    def test_extract_correlation_pairwise_fallback(self):
        ml_predictions = {
            "models": {
                "correlation": {
                    "BTC-ETH": {
                        "current_correlation": 0.70,
                        "forecast_correlation": 0.75,
                    },
                    "BTC-SOL": {
                        "current_correlation": 0.50,
                        "forecast_correlation": 0.55,
                    },
                }
            }
        }
        result = RealSignalExtractor.extract_correlation_signals(ml_predictions)
        assert result["avg_correlation"] > 0.4
        assert result["systemic_risk"] in ("low", "medium", "high")

    def test_extract_correlation_empty(self):
        result = RealSignalExtractor.extract_correlation_signals({})
        assert result["avg_correlation"] == 0.5

    def test_extract_correlation_min_floor(self):
        """avg_correlation should never drop below 0.4."""
        ml_predictions = {
            "models": {
                "correlation": {
                    "avg_correlation": 0.1,  # Very low
                    "systemic_risk": "low",
                }
            }
        }
        result = RealSignalExtractor.extract_correlation_signals(ml_predictions)
        assert result["avg_correlation"] >= 0.4

    def test_extract_sentiment_real(self):
        ml_predictions = {
            "models": {
                "sentiment": {
                    "BTC": {"sentiment_score": 0.3, "fear_greed_index": 65},
                    "ETH": {"sentiment_score": 0.1, "fear_greed_index": 55},
                }
            }
        }
        result = RealSignalExtractor.extract_sentiment_signals(ml_predictions)
        assert result["fear_greed"] == pytest.approx(60.0)
        assert result["sentiment_score"] == pytest.approx(0.2)

    def test_extract_sentiment_empty(self):
        result = RealSignalExtractor.extract_sentiment_signals({})
        assert result["fear_greed"] == 50
        assert result["sentiment_score"] == 0.0

    def test_calculate_confidence_weighted(self):
        ml_predictions = {
            "confidence_scores": {
                "volatility": 0.80,
                "sentiment": 0.70,
                "regime": 0.90,
                "correlation": 0.85,
            }
        }
        conf = RealSignalExtractor.calculate_confidence(ml_predictions)
        assert 0.0 <= conf <= 1.0
        # Weighted: 0.80*0.25 + 0.70*0.20 + 0.90*0.30 + 0.85*0.25 = 0.8225
        assert conf == pytest.approx(0.8225, abs=0.01)

    def test_calculate_confidence_empty(self):
        conf = RealSignalExtractor.calculate_confidence({})
        assert conf == 0.5

    def test_calculate_confidence_partial(self):
        ml_predictions = {
            "confidence_scores": {
                "volatility": 0.90,
            }
        }
        conf = RealSignalExtractor.calculate_confidence(ml_predictions)
        assert 0.0 <= conf <= 1.0

    def test_compute_contradiction_real(self):
        ml_predictions = {
            "ensemble": {
                "conflicting_signals": ["vol_vs_regime"],
                "consensus_strength": 0.6,
            }
        }
        idx = RealSignalExtractor.compute_contradiction_index(ml_predictions)
        assert 0.0 <= idx <= 1.0

    def test_compute_contradiction_no_ensemble(self):
        idx = RealSignalExtractor.compute_contradiction_index({})
        assert idx == 0.3  # Default

    def test_compute_contradiction_high_conflict(self):
        ml_predictions = {
            "ensemble": {
                "conflicting_signals": ["a", "b", "c", "d"],
                "consensus_strength": 0.2,
            }
        }
        idx = RealSignalExtractor.compute_contradiction_index(ml_predictions)
        assert idx > 0.5  # High contradiction


# ── create_default_signals ─────────────────────────────────────────

class TestCreateDefaultSignals:

    def test_returns_ml_signals(self):
        signals = create_default_signals()
        assert isinstance(signals, MLSignals)

    def test_default_values(self):
        signals = create_default_signals()
        assert signals.decision_score == 0.5
        assert signals.confidence == 0.5
        assert signals.contradiction_index == 0.3
        assert "fallback_default" in signals.sources_used

    def test_volatility_defaults(self):
        signals = create_default_signals()
        assert signals.volatility["BTC"] == 0.35
        assert signals.volatility["ETH"] == 0.45

    def test_regime_defaults(self):
        signals = create_default_signals()
        assert signals.regime["bull"] == 0.5
        assert signals.regime["bear"] == 0.25
