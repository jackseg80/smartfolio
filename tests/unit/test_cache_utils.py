"""Tests unitaires pour api/utils/cache.py — Simple TTL cache utilities."""
import time
import pytest
from unittest.mock import patch

from api.utils.cache import cache_get, cache_set, cache_clear_expired


class TestCacheGet:
    """Tests for cache_get function."""

    def test_returns_value_when_fresh(self):
        cache = {"key1": ("value1", time.time())}
        result = cache_get(cache, "key1", ttl=60)
        assert result == "value1"

    def test_returns_none_when_expired(self):
        cache = {"key1": ("value1", time.time() - 120)}
        result = cache_get(cache, "key1", ttl=60)
        assert result is None

    def test_returns_none_for_missing_key(self):
        cache = {}
        result = cache_get(cache, "missing", ttl=60)
        assert result is None

    def test_boundary_not_expired_at_exact_ttl(self):
        # At exactly TTL boundary, time.time() - ts < ttl is False (equal)
        now = time.time()
        cache = {"key1": ("value1", now - 60)}
        result = cache_get(cache, "key1", ttl=60)
        assert result is None  # Equal to TTL = expired

    def test_returns_value_just_before_expiry(self):
        cache = {"key1": ("value1", time.time() - 59)}
        result = cache_get(cache, "key1", ttl=60)
        assert result == "value1"

    def test_works_with_dict_values(self):
        data = {"nested": "dict", "count": 42}
        cache = {"complex": (data, time.time())}
        result = cache_get(cache, "complex", ttl=3600)
        assert result == data

    def test_works_with_list_values(self):
        data = [1, 2, 3]
        cache = {"list_key": (data, time.time())}
        result = cache_get(cache, "list_key", ttl=3600)
        assert result == data

    def test_works_with_none_value(self):
        cache = {"none_key": (None, time.time())}
        result = cache_get(cache, "none_key", ttl=3600)
        assert result is None  # Cannot distinguish None value from miss

    def test_zero_ttl_always_expired(self):
        cache = {"key1": ("value1", time.time())}
        result = cache_get(cache, "key1", ttl=0)
        assert result is None

    def test_large_ttl_value(self):
        cache = {"key1": ("value1", time.time() - 86400)}  # 1 day old
        result = cache_get(cache, "key1", ttl=86400 * 7)  # 7 day TTL
        assert result == "value1"


class TestCacheSet:
    """Tests for cache_set function."""

    def test_stores_value_with_timestamp(self):
        cache = {}
        before = time.time()
        cache_set(cache, "key1", "value1")
        after = time.time()

        assert "key1" in cache
        val, ts = cache["key1"]
        assert val == "value1"
        assert before <= ts <= after

    def test_overwrites_existing_value(self):
        cache = {"key1": ("old", time.time() - 100)}
        cache_set(cache, "key1", "new")

        val, ts = cache["key1"]
        assert val == "new"
        assert time.time() - ts < 1  # Fresh timestamp

    def test_stores_complex_values(self):
        cache = {}
        data = {"nested": {"deep": True}, "list": [1, 2, 3]}
        cache_set(cache, "complex", data)

        val, _ = cache["complex"]
        assert val == data

    def test_multiple_keys(self):
        cache = {}
        cache_set(cache, "a", 1)
        cache_set(cache, "b", 2)
        cache_set(cache, "c", 3)

        assert len(cache) == 3
        assert cache["a"][0] == 1
        assert cache["b"][0] == 2
        assert cache["c"][0] == 3


class TestCacheClearExpired:
    """Tests for cache_clear_expired function."""

    def test_removes_expired_entries(self):
        now = time.time()
        cache = {
            "fresh": ("v1", now),
            "expired": ("v2", now - 120),
        }
        cache_clear_expired(cache, ttl=60)
        assert "fresh" in cache
        assert "expired" not in cache

    def test_keeps_fresh_entries(self):
        now = time.time()
        cache = {
            "a": ("v1", now),
            "b": ("v2", now - 10),
            "c": ("v3", now - 30),
        }
        cache_clear_expired(cache, ttl=60)
        assert len(cache) == 3

    def test_clears_all_when_all_expired(self):
        old = time.time() - 3600
        cache = {
            "a": ("v1", old),
            "b": ("v2", old),
            "c": ("v3", old),
        }
        cache_clear_expired(cache, ttl=60)
        assert len(cache) == 0

    def test_empty_cache_no_error(self):
        cache = {}
        cache_clear_expired(cache, ttl=60)
        assert len(cache) == 0

    def test_boundary_at_exact_ttl(self):
        now = time.time()
        cache = {"at_boundary": ("v1", now - 60)}
        cache_clear_expired(cache, ttl=60)
        assert "at_boundary" not in cache  # >= TTL = expired

    def test_mixed_expired_and_fresh(self):
        now = time.time()
        cache = {
            "fresh1": ("v1", now - 5),
            "expired1": ("v2", now - 200),
            "fresh2": ("v3", now - 30),
            "expired2": ("v4", now - 500),
            "fresh3": ("v5", now),
        }
        cache_clear_expired(cache, ttl=60)
        assert set(cache.keys()) == {"fresh1", "fresh2", "fresh3"}


class TestCacheIntegration:
    """Integration tests: set → get → clear cycle."""

    def test_set_then_get(self):
        cache = {}
        cache_set(cache, "test", "hello")
        result = cache_get(cache, "test", ttl=60)
        assert result == "hello"

    def test_set_then_expire_then_get(self):
        cache = {}
        cache["test"] = ("hello", time.time() - 120)
        result = cache_get(cache, "test", ttl=60)
        assert result is None

    def test_full_cycle(self):
        cache = {}
        # Set multiple entries
        cache_set(cache, "a", 1)
        cache_set(cache, "b", 2)
        # Manually expire one
        cache["c"] = ("expired", time.time() - 200)

        assert cache_get(cache, "a", ttl=60) == 1
        assert cache_get(cache, "b", ttl=60) == 2
        assert cache_get(cache, "c", ttl=60) is None

        # Clear expired
        cache_clear_expired(cache, ttl=60)
        assert len(cache) == 2
        assert "c" not in cache
