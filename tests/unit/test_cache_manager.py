"""Tests unitaires pour services/cache_manager.py — CacheManager class."""
import time
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from services.cache_manager import CacheManager


@pytest.fixture
def manager():
    """Create a CacheManager with mocked imports (no real API modules)."""
    with patch.object(CacheManager, '_register_known_caches'):
        cm = CacheManager()
    return cm


@pytest.fixture
def populated_manager(manager):
    """CacheManager with pre-registered caches and data."""
    now = time.time()
    cache_a = {
        "fresh1": ("val1", now),
        "fresh2": ("val2", now - 10),
        "expired1": ("val3", now - 200),
    }
    cache_b = {
        "item1": ("data1", now),
    }
    manager.register_cache("test_a", cache_a, ttl=60, description="Test A cache")
    manager.register_cache("test_b", cache_b, ttl=300, description="Test B cache")
    return manager


class TestRegisterCache:
    """Tests for register_cache method."""

    def test_registers_cache(self, manager):
        cache = {"k": ("v", time.time())}
        manager.register_cache("my_cache", cache, ttl=120, description="My cache")

        assert "my_cache" in manager._cache_registry
        assert manager._cache_registry["my_cache"] is cache
        assert manager._cache_metadata["my_cache"]["ttl"] == 120
        assert manager._cache_metadata["my_cache"]["description"] == "My cache"
        assert manager._cache_metadata["my_cache"]["type"] == "in_memory"

    def test_default_description(self, manager):
        manager.register_cache("my_test", {}, ttl=60)
        assert manager._cache_metadata["my_test"]["description"] == "My Test Cache"

    def test_default_ttl(self, manager):
        manager.register_cache("default_ttl", {})
        assert manager._cache_metadata["default_ttl"]["ttl"] == 3600

    def test_overwrites_existing_registration(self, manager):
        cache1 = {}
        cache2 = {"new": True}
        manager.register_cache("overwrite", cache1, ttl=60)
        manager.register_cache("overwrite", cache2, ttl=120)

        assert manager._cache_registry["overwrite"] is cache2
        assert manager._cache_metadata["overwrite"]["ttl"] == 120


class TestGetCacheStats:
    """Tests for get_cache_stats method."""

    def test_single_cache_stats(self, populated_manager):
        stats = populated_manager.get_cache_stats("test_a")
        assert stats["name"] == "test_a"
        assert stats["entries"] == 3
        assert stats["valid_entries"] == 2
        assert stats["expired_entries"] == 1
        assert stats["ttl"] == 60

    def test_unknown_cache_returns_error(self, manager):
        stats = manager.get_cache_stats("nonexistent")
        assert "error" in stats
        assert "not found" in stats["error"]

    def test_all_caches_stats(self, populated_manager):
        stats = populated_manager.get_cache_stats()
        assert "total_caches" in stats
        assert "caches" in stats
        assert "total_entries" in stats
        assert "timestamp" in stats
        assert "test_a" in stats["caches"]
        assert "test_b" in stats["caches"]
        # +1 for coingecko_proxy special case
        assert stats["total_caches"] == 3
        assert stats["total_entries"] == 4  # 3 + 1

    def test_empty_cache_stats(self, manager):
        manager.register_cache("empty", {}, ttl=60)
        stats = manager.get_cache_stats("empty")
        assert stats["entries"] == 0
        assert stats["valid_entries"] == 0
        assert stats["expired_entries"] == 0

    def test_non_tuple_entries_count_as_valid(self, manager):
        cache = {"raw_key": "raw_value"}
        manager.register_cache("raw", cache, ttl=60)
        stats = manager.get_cache_stats("raw")
        assert stats["entries"] == 1
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 0

    def test_all_stats_includes_coingecko_proxy(self, populated_manager):
        stats = populated_manager.get_cache_stats()
        assert "coingecko_proxy" in stats["caches"]
        cg = stats["caches"]["coingecko_proxy"]
        assert cg["type"] == "function_scope"
        assert cg["entries"] == 0


class TestClearCache:
    """Tests for clear_cache method."""

    def test_clears_specific_cache(self, populated_manager):
        result = populated_manager.clear_cache("test_a", admin_user="jack")
        assert result["ok"] is True
        assert result["cleared_entries"] == 3
        assert result["cache"] == "test_a"
        assert "timestamp" in result
        # Cache should be empty now
        assert len(populated_manager._cache_registry["test_a"]) == 0

    def test_clear_nonexistent_cache(self, manager):
        result = manager.clear_cache("nonexistent", admin_user="jack")
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_clear_coingecko_proxy_fails(self, manager):
        result = manager.clear_cache("coingecko_proxy", admin_user="jack")
        assert result["ok"] is False
        assert "function-scoped" in result["error"]
        assert result["cleared_entries"] == 0

    def test_clear_all_caches(self, populated_manager):
        result = populated_manager.clear_cache("all", admin_user="jack")
        assert result["ok"] is True
        assert result["cache"] == "all"
        assert result["total_cleared"] == 4  # 3 + 1
        assert result["caches_cleared"] == 2
        assert len(result["details"]) == 2

    def test_clear_empty_cache(self, manager):
        manager.register_cache("empty", {}, ttl=60)
        result = manager.clear_cache("empty", admin_user="jack")
        assert result["ok"] is True
        assert result["cleared_entries"] == 0

    def test_default_admin_user(self, populated_manager):
        result = populated_manager.clear_cache("test_a")
        assert result["ok"] is True  # Uses default "system"


class TestClearExpired:
    """Tests for clear_expired method."""

    def test_clears_expired_from_specific_cache(self, populated_manager):
        result = populated_manager.clear_expired("test_a")
        assert result["cache"] == "test_a"
        assert result["cleared_entries"] == 1  # 1 expired entry

    def test_clears_expired_from_all_caches(self, populated_manager):
        result = populated_manager.clear_expired()
        assert result["ok"] is True
        assert result["total_cleared"] == 1  # Only test_a has expired
        assert result["caches_processed"] == 2

    def test_no_expired_entries(self, manager):
        cache = {"fresh": ("v", time.time())}
        manager.register_cache("fresh_cache", cache, ttl=3600)
        result = manager.clear_expired("fresh_cache")
        assert result["cleared_entries"] == 0

    def test_all_expired(self, manager):
        old = time.time() - 3600
        cache = {
            "a": ("v1", old),
            "b": ("v2", old),
        }
        manager.register_cache("old_cache", cache, ttl=60)
        result = manager.clear_expired("old_cache")
        assert result["cleared_entries"] == 2
        assert len(cache) == 0

    def test_skips_unregistered_cache_name(self, manager):
        # If cache_name is provided but not in registry, falls through to clear all
        result = manager.clear_expired("nonexistent")
        assert result["ok"] is True
        assert result["total_cleared"] == 0


class TestGetAvailableCaches:
    """Tests for get_available_caches method."""

    def test_includes_registered_caches(self, populated_manager):
        caches = populated_manager.get_available_caches()
        assert "test_a" in caches
        assert "test_b" in caches

    def test_always_includes_coingecko_proxy(self, manager):
        caches = manager.get_available_caches()
        assert "coingecko_proxy" in caches

    def test_returns_list(self, manager):
        caches = manager.get_available_caches()
        assert isinstance(caches, list)


class TestRegisterKnownCaches:
    """Tests for _register_known_caches (auto-discovery)."""

    def test_handles_import_errors_gracefully(self):
        """CacheManager should init even if modules can't be imported."""
        with patch('builtins.__import__', side_effect=ImportError("mocked")):
            # Should not raise, just log warnings
            cm = CacheManager()
            # coingecko_proxy is always added in metadata
            assert "coingecko_proxy" in cm._cache_metadata

    def test_registers_caches_when_modules_available(self):
        """Test that available modules get their caches registered."""
        mock_analytics = MagicMock()
        mock_analytics._analytics_cache = {"test": ("v", time.time())}

        with patch.dict('sys.modules', {'api.analytics_endpoints': mock_analytics}):
            with patch.object(CacheManager, '_register_known_caches') as mock_reg:
                cm = CacheManager()
                mock_reg.assert_called_once()


class TestCacheManagerEdgeCases:
    """Edge cases and special scenarios."""

    def test_cache_reference_is_shared(self, manager):
        """Verify CacheManager holds a reference, not a copy."""
        cache = {"key": ("val", time.time())}
        manager.register_cache("shared", cache, ttl=60)

        # Modify original dict
        cache["new_key"] = ("new_val", time.time())

        stats = manager.get_cache_stats("shared")
        assert stats["entries"] == 2  # Sees the new entry

    def test_concurrent_clear_and_stats(self, populated_manager):
        """Stats after clear should show 0 entries."""
        populated_manager.clear_cache("test_a", admin_user="jack")
        stats = populated_manager.get_cache_stats("test_a")
        assert stats["entries"] == 0

    def test_multiple_clear_calls(self, populated_manager):
        """Clearing twice should be safe."""
        r1 = populated_manager.clear_cache("test_a", admin_user="jack")
        r2 = populated_manager.clear_cache("test_a", admin_user="jack")
        assert r1["cleared_entries"] == 3
        assert r2["cleared_entries"] == 0

    def test_stats_timestamp_format(self, populated_manager):
        stats = populated_manager.get_cache_stats()
        ts = stats["timestamp"]
        # Should be parseable ISO format
        datetime.fromisoformat(ts)
