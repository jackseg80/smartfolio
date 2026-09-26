"""Unavailable stock predictions must remain valid API responses."""

from api.ml_bourse_endpoints import RegimeDetectionResponse, VolatilityForecastResponse


def test_volatility_fallback_has_no_fabricated_confidence():
    response = VolatilityForecastResponse(
        symbol="AAPL",
        timestamp="2026-09-26T00:00:00",
        predictions={"30d": {"predicted_volatility": 0.2, "confidence_interval": None}},
        model_type="historical_fallback",
        lookback_days=365,
        confidence_level=None,
    )
    assert response.confidence_level is None


def test_rule_based_regime_has_no_probability():
    response = RegimeDetectionResponse(
        current_regime="Correction",
        confidence=None,
        regime_probabilities={},
        benchmark="SPY",
        timestamp="2026-09-26T00:00:00",
        characteristics={"trend": "sideways"},
        model_type="moving_average_fallback",
    )
    assert response.confidence is None
