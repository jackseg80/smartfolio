from unittest.mock import AsyncMock, Mock
import pytest
from services.ml.bourse.stocks_adapter import StocksMLAdapter
from api.schemas.ml_contract import UnifiedPrediction


@pytest.mark.asyncio
async def test_volatility_model_uses_resolved_listing_and_preserves_requested_symbol(monkeypatch):
    from services.ml.reliability import capability_service
    adapter = object.__new__(StocksMLAdapter)
    adapter.volatility_predictor = Mock()
    adapter.volatility_predictor.train_model.side_effect=AssertionError("Reads never train")
    async def result(asset,market,kind,horizon):
        return UnifiedPrediction(asset=asset,market=market,horizon=horizon,value=None,reason="Verified artifact unavailable")
    mock=AsyncMock(side_effect=result)
    monkeypatch.setattr(capability_service,"result",mock)
    response=await adapter.predict_volatility("IWDA:xams")
    assert all(call.args[0]=="IWDA.AS" for call in mock.call_args_list)
    assert response["symbol"]=="IWDA:xams"
    assert response["model_type"]=="verified_daily_risk"
    assert response["predictions"]["1d"]["predicted_volatility"] is None
    adapter.volatility_predictor.train_model.assert_not_called()
    adapter.volatility_predictor.load_model.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_regime_quality_gate_cannot_certify_a_probability(monkeypatch):
    from services.ml.reliability import capability_service
    adapter=object.__new__(StocksMLAdapter)
    adapter.regime_detector=Mock()
    monkeypatch.setattr(capability_service,"result",AsyncMock(return_value=UnifiedPrediction(asset="SPY",market="stocks",nature="diagnostic",value="Bull Market",reason="Descriptive trailing-price rules")))
    response=await adapter.detect_market_regime(force_retrain=True)
    assert response["model_type"]=="descriptive_rules"
    assert response["confidence"] is None and response["regime_probabilities"]=={}
    adapter.regime_detector.train_model.assert_not_called()
    adapter.regime_detector.load_model.assert_not_called()


def test_stock_reload_preserves_validation_temperature(monkeypatch, tmp_path):
    import services.ml.models.regime_detector as module
    detector = module.RegimeDetector(model_dir=str(tmp_path), trading_days=252)
    for name in ('regime_neural_best.pth', 'regime_scaler.pkl', 'regime_features.pkl', 'regime_metadata.pkl'):
        (tmp_path / name).touch()
    def load(path):
        if path.name == 'regime_metadata.pkl':
            return {'optimal_temperature': .8}
        return ['close']
    network = Mock()
    network.to.return_value = network
    monkeypatch.setattr(module, 'safe_pickle_load', load)
    monkeypatch.setattr(module, 'safe_torch_load', lambda *args, **kwargs: {})
    monkeypatch.setattr(module, 'RegimeClassificationNetwork', lambda **kwargs: network)
    assert detector.load_model()
    assert detector.temperature == .8
