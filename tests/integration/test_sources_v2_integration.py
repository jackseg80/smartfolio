"""
Integration test for Sources V2 system.

Tests the complete flow:
1. Dashboard → loadBalanceData() → /balances/current
2. balance_service.resolve_current_balances() → SourceRegistry
3. Manual sources return data correctly
4. Migration works for existing users
"""
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


@pytest.fixture
def test_user_dir(tmp_path):
    """Create a test user directory structure."""
    user_dir = tmp_path / "data" / "users" / "test_user"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Create test client with isolated data directory."""
    # Patch project root
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    # Import after patching
    from api.main import app
    return TestClient(app)


class TestSourcesV2Integration:
    """Integration tests for Sources V2 system."""

    def test_new_user_defaults_to_v2_manual_sources(self, app_client, test_user_dir):
        """New users should default to V2 with empty manual sources."""
        # Simulate new user with no config
        response = app_client.get(
            "/balances/current",
            params={"source": "auto"},
            headers={"X-User": "test_user"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should use V2 mode with manual sources
        assert data.get("mode") == "category_based"
        assert "manual_crypto" in data.get("source_used", "")
        assert "manual_bourse" in data.get("source_used", "")

        # Items should be empty for new user
        assert data.get("items") == []

    def test_manual_crypto_source_crud(self, app_client, test_user_dir):
        """Test adding/reading manual crypto assets."""
        # Add a crypto asset
        asset_data = {
            "symbol": "BTC",
            "amount": 0.5,
            "value_usd": 25000,
            "location": "Cold Wallet"
        }

        response = app_client.post(
            "/api/sources/v2/crypto/manual/assets",
            json=asset_data,
            headers={"X-User": "test_user"}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        asset_id = result["data"]["asset"]["id"]

        # Verify asset appears in balances
        balances_response = app_client.get(
            "/balances/current",
            params={"source": "auto"},
            headers={"X-User": "test_user"}
        )

        balances_data = balances_response.json()
        items = balances_data.get("items", [])

        assert len(items) == 1
        assert items[0]["symbol"] == "BTC"
        assert items[0]["amount"] == 0.5
        assert items[0]["value_usd"] == 25000

    def test_manual_bourse_source_crud(self, app_client, test_user_dir):
        """Test adding/reading manual bourse positions."""
        position_data = {
            "symbol": "AAPL",
            "quantity": 10,
            "value": 1500,
            "currency": "USD",
            "name": "Apple Inc.",
            "asset_class": "EQUITY"
        }

        response = app_client.post(
            "/api/sources/v2/bourse/manual/positions",
            json=position_data,
            headers={"X-User": "test_user"}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True

        # Verify position appears in balances
        balances_response = app_client.get(
            "/balances/current",
            params={"source": "auto"},
            headers={"X-User": "test_user"}
        )

        balances_data = balances_response.json()
        items = balances_data.get("items", [])

        assert len(items) == 1
        assert items[0]["symbol"] == "AAPL"
        assert items[0]["amount"] == 10

    def test_migration_from_cointracking_csv(self, app_client, test_user_dir):
        """Test migration from legacy CoinTracking CSV to V2."""
        # Create old-style config
        config_path = test_user_dir / "config.json"
        old_config = {
            "data_source": "cointracking",
            "csv_selected_file": "test_export.csv"
        }
        config_path.write_text(json.dumps(old_config))

        # Create dummy CSV file
        csv_dir = test_user_dir / "cointracking" / "data"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_file = csv_dir / "test_export.csv"
        csv_file.write_text("Symbol,Amount,Value USD\nBTC,1.0,50000\n")

        # Trigger migration by accessing balances
        response = app_client.get(
            "/balances/current",
            params={"source": "auto"},
            headers={"X-User": "test_user"}
        )

        assert response.status_code == 200

        # Check config was migrated
        new_config = json.loads(config_path.read_text())
        assert new_config["data_source"] == "category_based"
        assert "sources" in new_config
        assert new_config["sources"]["crypto"]["active_source"] == "cointracking_csv"

        # Original CSV file should be preserved
        assert csv_file.exists()

    def test_switching_between_manual_and_csv_sources(self, app_client, test_user_dir):
        """Test switching active source for a category."""
        # Start with manual source
        config_path = test_user_dir / "config.json"
        config = {
            "data_source": "category_based",
            "sources": {
                "crypto": {
                    "active_source": "manual_crypto"
                }
            }
        }
        config_path.write_text(json.dumps(config))

        # Add manual asset
        app_client.post(
            "/api/sources/v2/crypto/manual/assets",
            json={"symbol": "ETH", "amount": 5, "value_usd": 10000},
            headers={"X-User": "test_user"}
        )

        # Verify manual data loads
        response1 = app_client.get(
            "/api/sources/v2/crypto/balances",
            headers={"X-User": "test_user"}
        )
        assert len(response1.json()["data"]["items"]) == 1

        # Switch to CSV source (with no CSV = empty results)
        switch_response = app_client.put(
            "/api/sources/v2/crypto/active",
            json={"source_id": "cointracking_csv"},
            headers={"X-User": "test_user"}
        )
        assert switch_response.status_code == 200

        # Verify source switched
        response2 = app_client.get(
            "/api/sources/v2/crypto/active",
            headers={"X-User": "test_user"}
        )
        assert response2.json()["data"]["active_source"] == "cointracking_csv"

    def test_category_isolation_crypto_vs_bourse(self, app_client, test_user_dir):
        """Test that crypto and bourse categories are independent."""
        # Add crypto asset
        app_client.post(
            "/api/sources/v2/crypto/manual/assets",
            json={"symbol": "BTC", "amount": 1, "value_usd": 50000},
            headers={"X-User": "test_user"}
        )

        # Add bourse position
        app_client.post(
            "/api/sources/v2/bourse/manual/positions",
            json={"symbol": "AAPL", "quantity": 10, "value": 1500, "currency": "USD"},
            headers={"X-User": "test_user"}
        )

        # Get combined balances
        response = app_client.get(
            "/balances/current",
            params={"source": "auto"},
            headers={"X-User": "test_user"}
        )

        items = response.json()["items"]
        assert len(items) == 2

        symbols = {item["symbol"] for item in items}
        assert symbols == {"BTC", "AAPL"}

        # Verify sources are independent
        sources = response.json()["sources"]
        assert sources["crypto"] == "manual_crypto"
        assert sources["bourse"] == "manual_bourse"

    def test_source_registry_discovery(self, app_client):
        """Test that all sources are registered and discoverable."""
        response = app_client.get(
            "/api/sources/v2/available",
            headers={"X-User": "test_user"}
        )

        assert response.status_code == 200
        sources = response.json()["data"]

        source_ids = [s["id"] for s in sources]

        # Verify all expected sources are registered
        expected = [
            "manual_crypto",
            "manual_bourse",
            "cointracking_csv",
            "cointracking_api",
            "saxobank_csv"
        ]

        for expected_id in expected:
            assert expected_id in source_ids, f"Source {expected_id} not registered"

    def test_backward_compatibility_with_legacy_endpoints(self, app_client, test_user_dir):
        """Test that legacy source= parameters still work."""
        # Create old-style user with cointracking source
        config_path = test_user_dir / "config.json"
        config_path.write_text(json.dumps({"data_source": "cointracking"}))

        # Old endpoint should still work
        response = app_client.get(
            "/balances/current",
            params={"source": "cointracking"},
            headers={"X-User": "test_user"}
        )

        assert response.status_code == 200
        # Should work (even if empty due to no CSV files)
        assert "items" in response.json()


@pytest.mark.asyncio
class TestSourcesV2Dashboard:
    """Test dashboard integration with Sources V2."""

    async def test_dashboard_loads_v2_manual_sources(self, app_client):
        """Test that dashboard can load V2 manual source data."""
        # This simulates what dashboard.html does via loadBalanceData()

        # Add test data
        app_client.post(
            "/api/sources/v2/crypto/manual/assets",
            json={"symbol": "BTC", "amount": 1, "value_usd": 50000},
            headers={"X-User": "test_user"}
        )

        # Dashboard calls /balances/current
        response = app_client.get(
            "/balances/current",
            params={"source": "auto", "min_usd": 1.0},
            headers={"X-User": "test_user"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should return data in expected format
        assert "items" in data
        assert len(data["items"]) > 0
        assert data["items"][0]["symbol"] == "BTC"

        # Should indicate V2 mode
        assert data.get("mode") == "category_based"

    async def test_analytics_endpoints_work_with_v2_sources(self, app_client):
        """Test that analytics endpoints work with V2 source data."""
        # Add diverse portfolio
        app_client.post(
            "/api/sources/v2/crypto/manual/assets",
            json={"symbol": "BTC", "amount": 0.5, "value_usd": 25000},
            headers={"X-User": "test_user"}
        )
        app_client.post(
            "/api/sources/v2/crypto/manual/assets",
            json={"symbol": "ETH", "amount": 10, "value_usd": 20000},
            headers={"X-User": "test_user"}
        )

        # Get portfolio metrics
        response = app_client.get(
            "/portfolio/metrics",
            params={"source": "auto"},
            headers={"X-User": "test_user"}
        )

        assert response.status_code == 200
        metrics = response.json()

        # Should calculate metrics correctly
        assert "total_value_usd" in metrics
        assert metrics["total_value_usd"] == 45000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
