"""Authenticated API transport without touching files, market providers or a server."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock


@pytest.fixture
def client(monkeypatch):
    from api.ml_bourse_endpoints import router
    from api.deps import get_required_user
    from services.ml.bourse import market_snapshot, market_analysis
    app = FastAPI(); app.include_router(router)
    app.dependency_overrides[get_required_user] = lambda: 'test'
    loader = AsyncMock(return_value={'snapshot_id': 'test'})
    analyst = AsyncMock(return_value={'version': 2, 'context': {'user_id': 'test'}, 'snapshot_id': 'test'})
    monkeypatch.setattr(market_snapshot, 'load_snapshot', loader)
    monkeypatch.setattr(market_analysis, 'analyze', analyst)
    return TestClient(app), loader, analyst


def test_scan_forwards_authenticated_user_and_selected_source_and_never_caches(client):
    http, loader, analyst = client
    response = http.get('/api/bourse/opportunities?source=saxobank_csv&file_key=selected.csv&candidate_sector=Technology')
    assert response.status_code == 200
    assert response.json()['data']['version'] == 2
    assert response.headers['cache-control'] == 'no-store'
    assert loader.call_args.args[:3] == ('test', 'saxobank_csv', 'selected.csv')
    assert analyst.call_args.args[2] == 'Technology'


def test_source_mismatch_is_a_visible_validation_error(client):
    http, loader, _ = client
    loader.side_effect = ValueError('Selection mismatch')
    response = http.get('/api/bourse/opportunities')
    assert response.status_code == 422 and response.json()['detail'] == 'Selection mismatch'


def test_unknown_scenario_fields_and_non_boolean_history_are_rejected(client):
    http, loader, _ = client
    response = http.post('/api/bourse/opportunities/scenario', json={'order': {'side': 'buy'}})
    assert response.status_code == 422
    response = http.post('/api/bourse/opportunities/scenario', json={'scenario': {'include_history': 'yes'}})
    assert response.status_code == 422
    loader.assert_not_called()


def test_invalid_target_policy_fails_before_private_source_load(client):
    http, loader, _ = client
    response = http.get('/api/bourse/opportunities', params={'sector_targets': '{"Europe": 100}'})
    assert response.status_code == 422
    loader.assert_not_called()


def test_scenario_transport_returns_only_read_only_cash_conserving_result(client):
    http, loader, analyst = client
    from services.ml.bourse.fund_exposure import SECTORS
    targets = {s: 0 for s in SECTORS}; targets['Technology'] = 100
    position = dict(id='held', symbol='TEST', value_usd=100, classification={'weights': {'Technology': 1}, 'known_fraction': 1})
    from services.ml.bourse.market_analysis import exposure_summary
    analyst.return_value = dict(snapshot_id='test', context={'user_id': 'test', 'source': 'saxobank_csv'},
        positions=[position], candidates=[], cash={'value_usd': 50}, targets=targets, min_gap_pct=5,
        exposure=exposure_summary([position], targets))
    response = http.post('/api/bourse/opportunities/scenario', json={'source': 'saxobank_csv', 'file_key': 'selected.csv',
        'scenario': {'snapshot_id': 'test', 'trades': [{'side': 'sell', 'id': 'held', 'amount_usd': 25}],
                     'costs_usd': 1, 'slippage_pct': 1, 'acknowledge_dated_values': True}})
    assert response.status_code == 200
    data = response.json()['data']['scenario']
    assert data['cash_after_usd'] == 73.75 and data['after_total_usd'] == 148.75
    assert 'positions_after' not in data and 'orders' not in data
    assert response.headers['cache-control'] == 'no-store'
