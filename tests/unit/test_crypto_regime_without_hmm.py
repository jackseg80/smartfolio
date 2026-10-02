"""A missing HMM must neither suppress economic rules nor invent a regime."""
import json
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from services.ml.models.btc_regime_detector import BTCRegimeDetector


@pytest.mark.asyncio
@pytest.mark.parametrize('trend,expected', [(0.08, 'Bull Market'), (0.01, 'Unknown')])
async def test_missing_hmm_retains_real_rules_without_training(monkeypatch, trend, expected):
    detector = BTCRegimeDetector()
    features = pd.DataFrame({'drawdown_from_peak': [-0.1], 'days_since_peak': [30],
                             'trend_30d': [trend], 'market_volatility': [0.3]},
                            index=pd.to_datetime(['2026-10-01']))
    monkeypatch.setattr(detector, 'prepare_regime_features', AsyncMock(return_value=features))
    monkeypatch.setattr(detector, 'load_model', lambda name: False)
    training = AsyncMock(side_effect=AssertionError('Inference must not train'))
    monkeypatch.setattr(detector, 'train_hmm', training)
    result = await detector.predict_regime('ETH', 365)
    assert result['regime_name'] == expected
    assert result['confidence'] is None and result['regime_probabilities'] == {}
    assert result['hmm_state'] is None and result['hmm_state_features'] == {}
    assert result['hmm_availability'] == 'Unavailable'
    assert result['data_as_of'] == '2026-10-01T00:00:00'
    assert 'No compatible HMM artifact for ETH' in result['rule_reason']
    assert result['availability'] == ('Partial' if expected != 'Unknown' else 'Unavailable')
    training.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_eth_history_preserves_unknown_intervals_and_no_btc_events(monkeypatch):
    import api.ml_crypto_endpoints as module
    index = pd.date_range('2024-03-11', periods=5)
    features = pd.DataFrame({'drawdown_from_peak': [-0.5, 0, 0, 0, -0.5],
                             'days_since_peak': 30, 'trend_30d': -0.2, 'market_volatility': 0.3}, index=index)
    history = [(int(d.timestamp()), 100 + i) for i, d in enumerate(index)]
    class Detector:
        async def prepare_regime_features(self, **kwargs):
            return features.copy()
        def load_model(self, name):
            assert name == 'eth_regime_hmm.pkl'
            return False
    monkeypatch.setattr(module, 'BTCRegimeDetector', Detector)
    monkeypatch.setattr(module.price_history, 'get_cached_history', lambda *args, **kwargs: history)
    monkeypatch.setattr(module, '_regime_history_cache', {})
    response = await module.get_crypto_regime_history(symbol='ETH', lookback_days=365)
    data = json.loads(response.body)['data']
    assert response.status_code == 200
    assert data['regimes'] == ['Bear Market', 'Unknown', 'Unknown', 'Unknown', 'Bear Market']
    assert data['regime_ids'] == [0, 8, 8, 8, 0]
    assert data['unknown_days'] == 3 and data['rule_classified_days'] == 2
    assert data['regime_id_mapping']['8'] == 'Unknown'
    assert data['hmm_available'] is False and data['events'] == []
    assert 'No compatible HMM artifact' in data['note']
    assert data['retrospective'] is True
    # A cached response must retain the limitation and unknown intervals.
    repeated = await module.get_crypto_regime_history(symbol='ETH', lookback_days=365)
    assert json.loads(repeated.body)['data'] == data


@pytest.mark.asyncio
async def test_current_endpoint_exposes_missing_model_metadata(monkeypatch):
    import api.ml_crypto_endpoints as module
    result = {'regime_name': 'Bull Market', 'confidence': None, 'detection_method': 'rule_based',
              'regime_info': {}, 'prediction_date': '2026-10-02', 'model_metadata': {},
              'hmm_availability': 'Unavailable', 'hmm_unavailability_reason': 'No compatible ETH artifact'}
    class Detector:
        async def predict_regime(self, **kwargs):
            return result
    monkeypatch.setattr(module, 'BTCRegimeDetector', Detector)
    response = await module.get_crypto_regime(symbol='ETH', lookback_days=365)
    data = json.loads(response.body)['data']
    assert data['current_regime'] == 'Bull Market'
    assert data['confidence'] is None
    assert data['hmm_availability'] == 'Unavailable'
    assert data['hmm_unavailability_reason'] == 'No compatible ETH artifact'
