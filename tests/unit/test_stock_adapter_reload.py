from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from services.ml.bourse.stocks_adapter import StocksMLAdapter


@pytest.mark.asyncio
async def test_volatility_model_uses_resolved_listing_and_preserves_requested_symbol():
    data = pd.DataFrame({'close': range(100, 190)}, index=pd.bdate_range('2026-01-01', periods=90))
    data.attrs['history_symbol'] = 'IWDA.AS'
    adapter = object.__new__(StocksMLAdapter)
    adapter.data_source = SimpleNamespace(get_ohlcv_data=AsyncMock(return_value=data))
    predictor = SimpleNamespace(models={}, metadata={})
    def load(symbol):
        predictor.models[symbol] = object()
        predictor.metadata[symbol] = {'validated_against_baseline': True}
    predictor.load_model = Mock(side_effect=load)
    predictor.train_model = Mock(side_effect=AssertionError('GET must never train'))
    predictor.predict_volatility = Mock(return_value={'predictions': {}})
    adapter.volatility_predictor = predictor

    result = await adapter.predict_volatility('IWDA:xams')

    predictor.load_model.assert_called_once_with('IWDA.AS')
    assert predictor.predict_volatility.call_args.kwargs['symbol'] == 'IWDA.AS'
    predictor.train_model.assert_not_called()
    assert result['symbol'] == 'IWDA:xams'
    assert result['model_type'] == 'LSTM'


@pytest.mark.asyncio
async def test_saved_regime_model_is_loaded_before_quality_gate():
    data = pd.DataFrame({'close': range(100, 190)}, index=pd.bdate_range('2026-01-01', periods=90))
    adapter = object.__new__(StocksMLAdapter)
    adapter.data_source = SimpleNamespace(get_benchmark_data_cached=AsyncMock(return_value=data))
    detector = SimpleNamespace(neural_model=None, training_metadata={})
    def load():
        detector.neural_model = object()
        detector.training_metadata = {'temporal_test_accuracy': 0.8, 'baseline_test_accuracy': 0.5}
        return True
    detector.load_model = Mock(side_effect=load)
    detector.train_model = Mock(side_effect=AssertionError('GET must never train'))
    detector.predict_regime = Mock(return_value={
        'predicted_regime': 2, 'confidence': 0.7,
        'regime_probabilities': {'Bull Market': 0.7},
    })
    adapter.regime_detector = detector

    result = await adapter.detect_market_regime()

    assert result['model_type'] == 'causal_neural'
    detector.load_model.assert_called_once()
    detector.train_model.assert_not_called()


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
