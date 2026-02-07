"""
Integration test for Sources V2 system.

Tests the complete flow:
1. Source registry discovery (available sources)
2. Active source management
3. Backward compatibility with legacy endpoints
4. Manual source CRUD endpoints respond correctly

Note: Tests use the real app with known users (demo/jack) since
monkeypatching PROJECT_ROOT does not affect the already-loaded
balance_service singleton.

Updated: 2026-02 - Fixed to work with real app state and V2 response formats
"""
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def app_client():
    """Create test client."""
    return TestClient(app)


class TestSourcesV2Integration:
    """Integration tests for Sources V2 system."""

    def test_source_registry_discovery(self, app_client):
        """Test that all sources are registered and discoverable."""
        response = app_client.get(
            "/api/sources/v2/available",
            headers={"X-User": "demo"}
        )

        assert response.status_code == 200
        data = response.json()
        # success_response wraps in {ok, data, ...}
        sources = data.get("data", data)

        # sources may be a list or wrapped structure
        if isinstance(sources, list):
            source_ids = [s["id"] for s in sources]
        elif isinstance(sources, dict) and "sources" in sources:
            source_ids = [s["id"] for s in sources["sources"]]
        else:
            source_ids = [s["id"] for s in sources] if isinstance(sources, list) else []

        # Verify core expected sources are registered
        expected = [
            "manual_crypto",
            "manual_bourse",
        ]

        for expected_id in expected:
            assert expected_id in source_ids, f"Source {expected_id} not registered"

    def test_backward_compatibility_with_legacy_endpoints(self, app_client):
        """Test that legacy source= parameters still work."""
        # Old endpoint should still work with known user
        response = app_client.get(
            "/balances/current",
            params={"source": "cointracking"},
            headers={"X-User": "demo"}
        )

        assert response.status_code == 200
        # Should work (even if empty due to no CSV files)
        assert "items" in response.json()

    def test_balances_current_returns_valid_structure(self, app_client):
        """Test that /balances/current returns valid structure for known users."""
        response = app_client.get(
            "/balances/current",
            params={"source": "auto"},
            headers={"X-User": "demo"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "source_used" in data
        assert isinstance(data["items"], list)

    def test_manual_crypto_assets_endpoint_exists(self, app_client):
        """Test that manual crypto assets endpoint responds."""
        response = app_client.get(
            "/api/sources/v2/crypto/manual/assets",
            headers={"X-User": "demo"}
        )

        # Should return 200 with ok response or empty list
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True

    def test_manual_bourse_positions_endpoint_exists(self, app_client):
        """Test that manual bourse positions endpoint responds."""
        response = app_client.get(
            "/api/sources/v2/bourse/manual/positions",
            headers={"X-User": "demo"}
        )

        # Should return 200
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True

    def test_get_active_source_crypto(self, app_client):
        """Test getting active source for crypto category."""
        response = app_client.get(
            "/api/sources/v2/crypto/active",
            headers={"X-User": "demo"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True

    def test_get_active_source_bourse(self, app_client):
        """Test getting active source for bourse category."""
        response = app_client.get(
            "/api/sources/v2/bourse/active",
            headers={"X-User": "demo"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True

    def test_category_balances_crypto(self, app_client):
        """Test getting balances for crypto category."""
        response = app_client.get(
            "/api/sources/v2/crypto/balances",
            headers={"X-User": "demo"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True

    def test_sources_summary(self, app_client):
        """Test getting sources summary."""
        response = app_client.get(
            "/api/sources/v2/summary",
            headers={"X-User": "demo"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True

    def test_categories_list(self, app_client):
        """Test listing source categories."""
        response = app_client.get(
            "/api/sources/v2/categories",
            headers={"X-User": "demo"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
