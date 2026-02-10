"""Tests for services/instruments_registry.py — ISIN validation, resolve, catalog management."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from services.instruments_registry import (
    resolve, clear_cache, add_to_catalog, _is_valid_isin,
    _ensure_loaded_global, _load_user_catalog,
    ISIN_PATTERN, GLOBAL_CATALOG_PATH, GLOBAL_ISIN_MAP_PATH,
)
import services.instruments_registry as registry_module


@pytest.fixture(autouse=True)
def clear_registry_cache():
    """Clear all caches before each test."""
    clear_cache()
    yield
    clear_cache()


# ===========================================================================
# _is_valid_isin
# ===========================================================================

class TestIsValidIsin:
    def test_valid_ireland(self):
        assert _is_valid_isin("IE00B4L5Y983") is True

    def test_valid_usa(self):
        assert _is_valid_isin("US0378331005") is True

    def test_valid_france(self):
        assert _is_valid_isin("FR0000120271") is True

    def test_valid_germany(self):
        assert _is_valid_isin("DE0005140008") is True

    def test_valid_luxembourg(self):
        assert _is_valid_isin("LU0290358497") is True

    def test_valid_lowercase_normalized(self):
        assert _is_valid_isin("ie00b4l5y983") is True

    def test_invalid_ticker(self):
        assert _is_valid_isin("AAPL") is False

    def test_invalid_too_short(self):
        assert _is_valid_isin("IE00") is False

    def test_invalid_too_long(self):
        assert _is_valid_isin("IE00B4L5Y983X") is False

    def test_invalid_numeric_prefix(self):
        assert _is_valid_isin("12B4L5Y98300") is False

    def test_invalid_empty(self):
        assert _is_valid_isin("") is False

    def test_invalid_special_chars(self):
        assert _is_valid_isin("IE00B4L5-983") is False


# ===========================================================================
# resolve — with mocked file I/O
# ===========================================================================

def _mock_loaded_state(catalog=None, isin_map=None):
    """Set module-level caches directly for testing."""
    registry_module._global_catalog_cache = catalog or {}
    registry_module._isin_to_ticker_cache = isin_map or {}
    registry_module._ticker_to_isin_cache = {v.upper(): k for k, v in (isin_map or {}).items()}
    registry_module._resolved_cache = {}


class TestResolve:
    def test_direct_catalog_hit(self):
        _mock_loaded_state(catalog={
            "AAPL": {"id": "AAPL", "symbol": "AAPL", "name": "Apple Inc.", "isin": None}
        })
        result = resolve("AAPL")
        assert result["name"] == "Apple Inc."
        assert result["symbol"] == "AAPL"

    def test_case_insensitive(self):
        _mock_loaded_state(catalog={
            "MSFT": {"id": "MSFT", "symbol": "MSFT", "name": "Microsoft"}
        })
        result = resolve("msft")
        assert result["name"] == "Microsoft"

    def test_whitespace_trimmed(self):
        _mock_loaded_state(catalog={
            "GOOG": {"id": "GOOG", "symbol": "GOOG", "name": "Google"}
        })
        result = resolve("  GOOG  ")
        assert result["name"] == "Google"

    def test_isin_to_ticker_lookup(self):
        _mock_loaded_state(
            catalog={"IWDA.AMS": {"id": "IWDA.AMS", "symbol": "IWDA.AMS", "name": "iShares World"}},
            isin_map={"IE00B4L5Y983": "IWDA.AMS"},
        )
        result = resolve("IE00B4L5Y983")
        assert result["name"] == "iShares World"

    def test_ticker_to_isin_reverse_lookup(self):
        _mock_loaded_state(
            catalog={"IE00B4L5Y983": {"id": "IE00B4L5Y983", "symbol": "IWDA", "name": "iShares"}},
            isin_map={"IE00B4L5Y983": "IWDA.AMS"},
        )
        # Looking up "IWDA.AMS" → reverse maps to ISIN → catalog has ISIN entry
        result = resolve("IWDA.AMS")
        assert result["name"] == "iShares"

    def test_fallback_minimal_record_ticker(self):
        _mock_loaded_state()
        result = resolve("UNKNOWN_TICKER", fallback_symbol="UNK")
        assert result["id"] == "UNKNOWN_TICKER"
        assert result["symbol"] == "UNK"
        assert result["name"] == "UNK"
        assert result["isin"] is None
        assert result["exchange"] is None

    def test_fallback_minimal_record_isin_format(self):
        _mock_loaded_state()
        result = resolve("IE00B4L5Y983")
        assert result["isin"] == "IE00B4L5Y983"
        assert result["symbol"] == "IE00B4L5Y983"

    def test_fallback_without_symbol(self):
        _mock_loaded_state()
        result = resolve("XYZ123")
        assert result["symbol"] == "XYZ123"
        assert result["name"] == "XYZ123"

    def test_cache_hit(self):
        _mock_loaded_state(catalog={
            "AAPL": {"id": "AAPL", "symbol": "AAPL", "name": "Apple"}
        })
        r1 = resolve("AAPL")
        r2 = resolve("AAPL")
        assert r1 is r2  # Same object from cache

    def test_user_catalog_priority(self):
        _mock_loaded_state(catalog={
            "AAPL": {"id": "AAPL", "symbol": "AAPL", "name": "Apple Global"}
        })
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value=json.dumps({
                 "AAPL": {"id": "AAPL", "symbol": "AAPL", "name": "Apple User"}
             })):
            result = resolve("AAPL", user_id="jack")
        assert result["name"] == "Apple User"

    def test_user_cache_key_isolation(self):
        _mock_loaded_state(catalog={
            "AAPL": {"id": "AAPL", "symbol": "AAPL", "name": "Apple Global"}
        })
        # Resolve without user
        r_global = resolve("AAPL")
        assert r_global["name"] == "Apple Global"

        # Resolve with user (no user catalog → falls through to global)
        with patch.object(Path, 'exists', return_value=False):
            r_user = resolve("AAPL", user_id="demo")
        assert r_user["name"] == "Apple Global"


# ===========================================================================
# _load_user_catalog
# ===========================================================================

class TestLoadUserCatalog:
    def test_no_file_returns_empty(self):
        with patch.object(Path, 'exists', return_value=False):
            result = _load_user_catalog("test_user")
        assert result == {}

    def test_valid_file(self):
        data = {"AAPL": {"name": "Apple"}}
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value=json.dumps(data)):
            result = _load_user_catalog("test_user")
        assert result == data

    def test_corrupt_file_returns_empty(self):
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value="not json"):
            result = _load_user_catalog("test_user")
        assert result == {}


# ===========================================================================
# _ensure_loaded_global
# ===========================================================================

class TestEnsureLoadedGlobal:
    def test_creates_empty_catalog_if_missing(self):
        with patch.object(Path, 'exists', return_value=False), \
             patch.object(Path, 'mkdir'), \
             patch.object(Path, 'write_text') as mock_write, \
             patch('services.instruments_registry.FileLock', return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())):
            _ensure_loaded_global()
        assert registry_module._global_catalog_cache == {}
        assert registry_module._isin_to_ticker_cache == {}

    def test_loads_catalog_from_disk(self):
        catalog = {"AAPL": {"id": "AAPL"}}
        isin_map = {"US0378331005": "AAPL"}

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps(catalog)
            return json.dumps(isin_map)

        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', side_effect=side_effect):
            _ensure_loaded_global()

        assert "AAPL" in registry_module._global_catalog_cache
        assert "US0378331005" in registry_module._isin_to_ticker_cache
        assert "AAPL" in registry_module._ticker_to_isin_cache

    def test_no_reload_if_already_loaded(self):
        registry_module._global_catalog_cache = {"X": {}}
        with patch.object(Path, 'read_text') as mock_read:
            _ensure_loaded_global()
        mock_read.assert_not_called()


# ===========================================================================
# add_to_catalog
# ===========================================================================

class TestAddToCatalog:
    def test_add_to_global(self):
        _mock_loaded_state()
        meta = {"symbol": "TEST", "name": "Test Corp"}
        with patch.object(Path, 'write_text'):
            add_to_catalog("TEST", meta, persist=True)
        assert "TEST" in registry_module._global_catalog_cache

    def test_add_invalidates_cache(self):
        _mock_loaded_state(catalog={"OLD": {"name": "Old"}})
        # Resolve to fill cache
        resolve("OLD")
        assert "OLD" in registry_module._resolved_cache

        # Add to catalog invalidates that key
        with patch.object(Path, 'write_text'):
            add_to_catalog("OLD", {"name": "New"}, persist=True)
        assert "OLD" not in registry_module._resolved_cache

    def test_add_to_user_catalog(self):
        _mock_loaded_state()
        meta = {"symbol": "PRIV", "name": "Private"}
        with patch.object(Path, 'exists', return_value=False), \
             patch.object(Path, 'mkdir'), \
             patch.object(Path, 'read_text', return_value="{}"), \
             patch.object(Path, 'write_text') as mock_write:
            add_to_catalog("PRIV", meta, user_id="jack", persist=True)
        mock_write.assert_called_once()

    def test_add_no_persist(self):
        _mock_loaded_state()
        with patch.object(Path, 'write_text') as mock_write:
            add_to_catalog("NOWRITE", {"name": "X"}, persist=False)
        mock_write.assert_not_called()
        assert "NOWRITE" in registry_module._global_catalog_cache

    def test_uppercase_key(self):
        _mock_loaded_state()
        with patch.object(Path, 'write_text'):
            add_to_catalog("lowercase", {"name": "X"}, persist=True)
        assert "LOWERCASE" in registry_module._global_catalog_cache


# ===========================================================================
# clear_cache
# ===========================================================================

class TestClearCache:
    def test_clears_all(self):
        _mock_loaded_state(catalog={"A": {}}, isin_map={"B": "C"})
        registry_module._resolved_cache = {"D": {}}
        clear_cache()
        assert registry_module._global_catalog_cache is None
        assert registry_module._isin_to_ticker_cache is None
        assert registry_module._ticker_to_isin_cache is None
        assert registry_module._resolved_cache == {}


# ===========================================================================
# ISIN_PATTERN regex
# ===========================================================================

class TestIsinPattern:
    def test_exact_12_chars(self):
        assert ISIN_PATTERN.match("AB1234567890")

    def test_too_short(self):
        assert ISIN_PATTERN.match("AB12345678") is None

    def test_too_long(self):
        assert ISIN_PATTERN.match("AB12345678901") is None

    def test_lowercase_not_matched_directly(self):
        assert ISIN_PATTERN.match("ab1234567890") is None
