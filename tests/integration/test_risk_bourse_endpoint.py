"""
Tests d'integration pour endpoint /api/risk/bourse/dashboard

Verifie:
- Endpoint retourne metriques valides
- Score canonique 0-100 (plus haut = plus robuste)
- Multi-tenant respecte (X-User obligatoire)
- Fallback gracieux si 0 positions

Updated: 2026-02 - Use X-User header instead of user_id query param
"""

import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


@pytest.fixture
def selected_csv(monkeypatch):
    """Use a selected source without depending on a local demo account."""
    monkeypatch.setattr(
        "services.portfolio_export_service.resolve_saxo_file_key",
        lambda user_id, file_key: "selected.csv",
    )
    monkeypatch.setattr(
        "services.portfolio_export_service.read_saxo_cash",
        lambda user_id, file_key: {"value_usd": 0.0},
    )
    monkeypatch.setattr(
        "adapters.saxo_adapter.list_portfolios_overview",
        lambda **kwargs: [{"portfolio_id": "selected"}],
    )
    monkeypatch.setattr(
        "adapters.saxo_adapter.get_portfolio_detail",
        lambda **kwargs: {"positions": [{"symbol": "AAPL", "market_value_usd": 100.0}]},
    )


class TestRiskBourseEndpoint:
    """Tests de l'endpoint risk bourse."""

    def test_high_min_usd_returns_filtered_state(self, selected_csv):
        """Very high min_usd should filter all positions gracefully."""
        # Use X-User header (required by get_required_user)
        # Use min_usd very high to filter all positions below threshold
        response = client.get(
            "/api/risk/bourse/dashboard",
            params={"min_usd": 999999999},
            headers={"X-User": "demo"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is False
        assert data["risk"]["score"] is None
        assert data["risk"]["level"] == "N/A"

    def test_user_id_via_header(self, selected_csv):
        """X-User header must be present (multi-tenant)."""
        response = client.get(
            "/api/risk/bourse/dashboard",
            params={"min_usd": 999999999},
            headers={"X-User": "demo"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "demo"

    def test_missing_user_header_returns_422(self):
        """Missing X-User header should return 422."""
        response = client.get("/api/risk/bourse/dashboard")
        assert response.status_code == 422

    def test_api_cache_unpacks_positions_and_cash(self, monkeypatch):
        from services.risk.bourse.calculator import IncompleteBourseMarketData
        positions = [{'symbol': 'AAPL', 'market_value_usd': 100.0}]
        monkeypatch.setattr('services.saxo_auth_service.SaxoAuthService.get_cached_positions',
                            AsyncMock(return_value={'positions': positions, 'cash_balance': 25.0, 'total_value': 125.0}))
        calculate = AsyncMock(side_effect=IncompleteBourseMarketData(['AAPL'], 0.0))
        monkeypatch.setattr('api.risk_bourse_endpoints.BourseRiskCalculator.calculate_portfolio_risk', calculate)
        response = client.get('/api/risk/bourse/dashboard', params={'source': 'saxobank_api'}, headers={'X-User': 'demo'})
        assert response.status_code == 200
        assert response.json()['total_value_usd'] == 125.0
        assert calculate.call_args.kwargs['cash_amount'] == 25.0
        assert calculate.call_args.kwargs['positions'] == positions

    def test_short_position_is_not_silently_filtered(self, monkeypatch, selected_csv):
        monkeypatch.setattr('adapters.saxo_adapter.get_portfolio_detail', lambda **kwargs: {
            'positions': [{'symbol': 'AAA', 'market_value_usd': 100.0},
                          {'symbol': 'BBB', 'market_value_usd': -20.0, 'quantity': -1}]})
        response = client.get('/api/risk/bourse/dashboard', headers={'X-User': 'demo'})
        assert response.status_code == 422
        assert 'unleveraged long positions' in response.json()['detail']


# Commande pour lancer ces tests:
# pytest tests/integration/test_risk_bourse_endpoint.py -v
