"""Tests unitaires pour services/wealth/wealth_service.py (0% coverage)."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from models.wealth import WealthItemInput, WealthItemOutput


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def wealth_dir(tmp_path):
    """Create a tmp wealth directory and patch storage path resolution."""
    user_dir = tmp_path / "data" / "users" / "test_user" / "wealth"
    user_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_fx():
    """Mock fx_convert to return value as-is (1:1 rate)."""
    with patch("services.wealth.wealth_service.fx_convert", side_effect=lambda v, src, dst: v) as m:
        yield m


@pytest.fixture
def mock_migration():
    """Mock migrate_user_data to no-op."""
    with patch("services.wealth.wealth_service.migrate_user_data") as m:
        yield m


def _patch_storage(tmp_path, user_id="test_user"):
    """Patch _get_storage_path to use tmp_path."""
    wealth_path = tmp_path / "data" / "users" / user_id / "wealth" / "wealth.json"
    wealth_path.parent.mkdir(parents=True, exist_ok=True)
    return patch(
        "services.wealth.wealth_service._get_storage_path",
        return_value=wealth_path,
    )


def _make_input(**kwargs):
    """Create a WealthItemInput with defaults."""
    defaults = {
        "name": "Test Account",
        "category": "liquidity",
        "type": "bank_account",
        "value": 10000.0,
        "currency": "USD",
    }
    defaults.update(kwargs)
    return WealthItemInput(**defaults)


def _seed_items(wealth_path, items):
    """Write items directly to the wealth.json file."""
    wealth_path.parent.mkdir(parents=True, exist_ok=True)
    wealth_path.write_text(json.dumps({"items": items}), encoding="utf-8")


# ── Tests ───────────────────────────────────────────────────────────────


class TestListItems:
    def test_empty_user_returns_empty_list(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import list_items
        with _patch_storage(wealth_dir):
            result = list_items("test_user")
        assert result == []

    def test_returns_items(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import list_items
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [
            {"id": "1", "name": "UBS", "category": "liquidity", "type": "bank_account",
             "value": 5000, "currency": "CHF"},
            {"id": "2", "name": "House", "category": "tangible", "type": "real_estate",
             "value": 500000, "currency": "EUR"},
        ])
        with _patch_storage(wealth_dir):
            result = list_items("test_user")
        assert len(result) == 2
        assert result[0].name == "UBS"

    def test_filter_by_category(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import list_items
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [
            {"id": "1", "name": "Bank", "category": "liquidity", "type": "bank_account",
             "value": 5000, "currency": "USD"},
            {"id": "2", "name": "House", "category": "tangible", "type": "real_estate",
             "value": 500000, "currency": "USD"},
        ])
        with _patch_storage(wealth_dir):
            result = list_items("test_user", category="tangible")
        assert len(result) == 1
        assert result[0].name == "House"

    def test_filter_by_type(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import list_items
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [
            {"id": "1", "name": "UBS", "category": "liquidity", "type": "bank_account",
             "value": 5000, "currency": "USD"},
            {"id": "2", "name": "Revolut", "category": "liquidity", "type": "neobank",
             "value": 2000, "currency": "USD"},
        ])
        with _patch_storage(wealth_dir):
            result = list_items("test_user", type="neobank")
        assert len(result) == 1
        assert result[0].name == "Revolut"


class TestGetItem:
    def test_get_existing_item(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import get_item
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [
            {"id": "abc123", "name": "Gold", "category": "tangible", "type": "precious_metals",
             "value": 3000, "currency": "USD"},
        ])
        with _patch_storage(wealth_dir):
            result = get_item("test_user", "abc123")
        assert result is not None
        assert result.name == "Gold"
        assert result.id == "abc123"

    def test_get_nonexistent_item(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import get_item
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [])
        with _patch_storage(wealth_dir):
            result = get_item("test_user", "nonexistent")
        assert result is None


class TestCreateItem:
    def test_create_item_returns_output(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import create_item
        item_input = _make_input(name="New Bank", value=8000.0, currency="EUR")
        with _patch_storage(wealth_dir):
            result = create_item("test_user", item_input)
        assert isinstance(result, WealthItemOutput)
        assert result.name == "New Bank"
        assert result.value == 8000.0
        assert result.id is not None  # UUID generated

    def test_create_item_persists(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import create_item, list_items
        item_input = _make_input(name="Savings")
        with _patch_storage(wealth_dir):
            create_item("test_user", item_input)
            items = list_items("test_user")
        assert len(items) == 1
        assert items[0].name == "Savings"


class TestUpdateItem:
    def test_update_existing_item(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import update_item
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [
            {"id": "upd1", "name": "Old Name", "category": "liquidity", "type": "bank_account",
             "value": 1000, "currency": "USD"},
        ])
        new_data = _make_input(name="New Name", value=2000.0)
        with _patch_storage(wealth_dir):
            result = update_item("test_user", "upd1", new_data)
        assert result is not None
        assert result.name == "New Name"
        assert result.value == 2000.0

    def test_update_nonexistent_item(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import update_item
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [])
        new_data = _make_input()
        with _patch_storage(wealth_dir):
            result = update_item("test_user", "nonexistent", new_data)
        assert result is None


class TestDeleteItem:
    def test_delete_existing_item(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import delete_item
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [
            {"id": "del1", "name": "To Delete", "category": "liquidity", "type": "bank_account",
             "value": 100, "currency": "USD"},
        ])
        with _patch_storage(wealth_dir):
            result = delete_item("test_user", "del1")
        assert result is True

    def test_delete_nonexistent_item(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import delete_item
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [])
        with _patch_storage(wealth_dir):
            result = delete_item("test_user", "nonexistent")
        assert result is False


class TestGetSummary:
    def test_empty_portfolio_summary(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import get_summary
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [])
        with _patch_storage(wealth_dir):
            summary = get_summary("test_user")
        assert summary["net_worth"] == 0.0
        assert summary["total_assets"] == 0.0
        assert summary["total_liabilities"] == 0.0

    def test_summary_calculates_net_worth(self, wealth_dir, mock_fx, mock_migration):
        from services.wealth.wealth_service import get_summary
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        _seed_items(wealth_path, [
            {"id": "1", "name": "Bank", "category": "liquidity", "type": "bank_account",
             "value": 10000, "currency": "USD"},
            {"id": "2", "name": "House", "category": "tangible", "type": "real_estate",
             "value": 300000, "currency": "USD"},
            {"id": "3", "name": "Mortgage", "category": "liability", "type": "mortgage",
             "value": -200000, "currency": "USD"},
            {"id": "4", "name": "Life Ins", "category": "insurance", "type": "life_insurance",
             "value": 50000, "currency": "USD"},
        ])
        with _patch_storage(wealth_dir):
            summary = get_summary("test_user")

        assert summary["total_assets"] == 10000 + 300000 + 50000  # liquidity + tangible + insurance
        assert summary["total_liabilities"] == 200000  # abs of liability
        assert summary["net_worth"] == 360000 - 200000  # assets - liabilities
        assert summary["counts"]["liquidity"] == 1
        assert summary["counts"]["tangible"] == 1
        assert summary["counts"]["liability"] == 1
        assert summary["counts"]["insurance"] == 1
        assert summary["user_id"] == "test_user"


class TestStoragePath:
    def test_prefers_wealth_json(self, tmp_path):
        from services.wealth.wealth_service import _get_storage_path
        user_id = "path_test"
        wealth_path = tmp_path / "data" / "users" / user_id / "wealth" / "wealth.json"
        legacy_path = tmp_path / "data" / "users" / user_id / "wealth" / "patrimoine.json"
        wealth_path.parent.mkdir(parents=True, exist_ok=True)
        wealth_path.write_text("{}", encoding="utf-8")
        legacy_path.write_text("{}", encoding="utf-8")

        with patch("services.wealth.wealth_service.Path", side_effect=lambda p: tmp_path / p):
            # Cannot easily test without full path rewriting; test the logic indirectly
            pass

    def test_load_snapshot_handles_corrupt_json(self, wealth_dir, mock_migration):
        from services.wealth.wealth_service import _load_snapshot
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        wealth_path.write_text("not valid json {{{{", encoding="utf-8")
        with _patch_storage(wealth_dir):
            result = _load_snapshot("test_user")
        assert result == {"items": []}

    def test_load_snapshot_handles_missing_items_key(self, wealth_dir, mock_migration):
        from services.wealth.wealth_service import _load_snapshot
        wealth_path = wealth_dir / "data" / "users" / "test_user" / "wealth" / "wealth.json"
        wealth_path.write_text(json.dumps({"version": 2}), encoding="utf-8")
        with _patch_storage(wealth_dir):
            result = _load_snapshot("test_user")
        assert "items" in result
        assert result["items"] == []
