"""Unavailable ML paths must never emit numeric pseudo-predictions."""

import pytest

from api.ml.unified_contract_endpoints import _get_raw_prediction
from api.ml.gating import MLGatingSystem
from api.schemas.ml_contract import Horizon, ModelType, create_fallback_response


def test_fallback_response_contains_no_predictions():
    response = create_fallback_response(
        ModelType.SENTIMENT,
        ["BTC", "ETH"],
        "Model unavailable",
    )

    assert response.success is False
    assert response.predictions == []
    assert response.failed_assets == ["BTC", "ETH"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", [ModelType.SENTIMENT, ModelType.RISK])
async def test_unconnected_model_types_return_no_raw_value(model_type):
    assert await _get_raw_prediction("BTC", model_type, Horizon.D7) is None


def test_uncalibrated_prediction_is_rejected_without_numeric_fallback():
    prediction, accepted = MLGatingSystem().gate_prediction(
        asset="BTC",
        raw_prediction=0.42,
        model_key="risk_D7",
        model_type=ModelType.RISK,
        context={},
    )

    assert accepted is False
    assert prediction is None
