"""
Tests unitaires pour valider les corrections des incohérences de données de wallet
- Test des métadonnées normalisées dans /api/risk/dashboard
- Test de cohérence user/source

Note: X-User header is REQUIRED (get_required_user). Without it, endpoint returns 422.
User IDs must be <= 50 characters.
Test users with no data get {"success": False} because no balances are found.
"""

from unittest.mock import AsyncMock, call, patch

import pytest
from fastapi.testclient import TestClient

import api.risk_endpoints as risk_endpoints
from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_risk_dashboard_cache():
    """Prevent cached responses from leaking between endpoint tests."""
    risk_endpoints._risk_cache.clear()


def _empty_portfolio(source: str) -> dict:
    return {"items": [], "source_used": source}


def test_risk_dashboard_returns_200_with_valid_user():
    """An empty portfolio is a valid, deterministic dashboard response."""
    with patch(
        "api.unified_data.get_unified_filtered_balances",
        new=AsyncMock(return_value=_empty_portfolio("stub")),
    ):
        response = client.get(
            "/api/risk/dashboard",
            headers={"X-User": "demo"},
            params={"source": "stub", "min_usd": 1.0},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "message" in data


def test_risk_dashboard_requires_x_user_header():
    """Test that /api/risk/dashboard returns 422 without X-User header"""
    response = client.get("/api/risk/dashboard")

    # X-User is required (get_required_user dependency)
    assert response.status_code == 422


def test_risk_dashboard_rejects_long_user_id():
    """Test that user IDs longer than 50 characters are rejected"""
    long_user_id = "a" * 51

    response = client.get(
        "/api/risk/dashboard",
        headers={"X-User": long_user_id},
        params={"source": "cointracking", "min_usd": 1.0}
    )

    assert response.status_code == 400


def test_risk_dashboard_empty_portfolio():
    """A source without holdings should return a clear empty response."""
    with patch(
        "api.unified_data.get_unified_filtered_balances",
        new=AsyncMock(return_value=_empty_portfolio("manual_bourse")),
    ):
        response = client.get(
            "/api/risk/dashboard",
            headers={"X-User": "demo"},
            params={"source": "manual_bourse", "min_usd": 1.0},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "message" in data


def test_risk_dashboard_passes_each_user_to_the_balance_resolver():
    """Balance resolution remains isolated for each authenticated user."""
    mocked_balances = AsyncMock(return_value=_empty_portfolio("cointracking"))
    with patch(
        "api.unified_data.get_unified_filtered_balances",
        new=mocked_balances,
    ):
        response_user1 = client.get(
            "/api/risk/dashboard",
            headers={"X-User": "demo"},
            params={"source": "cointracking", "min_usd": 1.0},
        )
        response_user2 = client.get(
            "/api/risk/dashboard",
            headers={"X-User": "jack"},
            params={"source": "cointracking", "min_usd": 1.0},
        )

    assert response_user1.status_code == 200
    assert response_user2.status_code == 200
    assert mocked_balances.await_args_list == [
        call(source="cointracking", min_usd=1.0, user_id="demo"),
        call(source="cointracking", min_usd=1.0, user_id="jack"),
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
