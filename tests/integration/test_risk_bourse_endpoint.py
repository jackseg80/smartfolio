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
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestRiskBourseEndpoint:
    """Tests de l'endpoint risk bourse."""

    def test_high_min_usd_returns_filtered_state(self):
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
        assert data["ok"] is True
        # When all positions are below threshold, risk score should be 0
        assert data["risk"]["score"] == 0
        assert data["risk"]["level"] == "N/A"

    def test_user_id_via_header(self):
        """X-User header must be present (multi-tenant)."""
        response = client.get(
            "/api/risk/bourse/dashboard",
            headers={"X-User": "demo"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "demo"

    def test_missing_user_header_returns_422(self):
        """Missing X-User header should return 422."""
        response = client.get("/api/risk/bourse/dashboard")
        assert response.status_code == 422


# Commande pour lancer ces tests:
# pytest tests/integration/test_risk_bourse_endpoint.py -v
