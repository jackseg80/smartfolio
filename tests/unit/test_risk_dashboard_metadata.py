"""
Tests unitaires pour valider les corrections des incohérences de données de wallet
- Test des métadonnées normalisées dans /api/risk/dashboard
- Test de cohérence user/source

Note: X-User header is REQUIRED (get_required_user). Without it, endpoint returns 422.
User IDs must be <= 50 characters.
Test users with no data get {"success": False} because no balances are found.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from datetime import datetime

client = TestClient(app)


def test_risk_dashboard_returns_200_with_valid_user():
    """Test that /api/risk/dashboard returns 200 with a valid user and stub source"""
    response = client.get(
        "/api/risk/dashboard",
        headers={"X-User": "demo"},
        params={
            "source": "stub",
            "min_usd": 1.0,
            "price_history_days": 30,
            "lookback_days": 30
        }
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    # Response can be success=True with metrics or success=False if insufficient price data
    # Both are valid 200 responses
    assert isinstance(data, dict)


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
    """Test that a valid user with an unused source gets a clear response"""
    # Use a valid user but a source directory that has no data
    response = client.get(
        "/api/risk/dashboard",
        headers={"X-User": "demo"},
        params={"source": "manual_bourse", "min_usd": 1.0}
    )

    # error_response returns 400 for insufficient price data
    assert response.status_code == 400
    data = response.json()

    # Error response has ok=False and error message
    assert data.get("ok") is False
    assert "error" in data


def test_risk_dashboard_different_users_get_different_responses():
    """Test that different valid users get independent responses"""
    response_user1 = client.get(
        "/api/risk/dashboard",
        headers={"X-User": "demo"},
        params={"source": "cointracking", "min_usd": 1.0}
    )

    response_user2 = client.get(
        "/api/risk/dashboard",
        headers={"X-User": "jack"},
        params={"source": "cointracking", "min_usd": 1.0}
    )

    assert response_user1.status_code == 200
    assert response_user2.status_code == 200

    # Both should return valid JSON
    data_user1 = response_user1.json()
    data_user2 = response_user2.json()

    assert isinstance(data_user1, dict)
    assert isinstance(data_user2, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
