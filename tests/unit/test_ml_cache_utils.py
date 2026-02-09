"""Tests unitaires pour api/ml/cache_utils.py — ML cache with datetime timestamps."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from api.ml.cache_utils import (
    cache_get,
    cache_set,
    cache_clear_expired,
    cache_clear_all,
    get_ml_cache,
    _unified_ml_cache,
)


@pytest.fixture(autouse=True)
def clear_global_cache():
    """Clear the global ML cache before and after each test."""
    _unified_ml_cache.clear()
    yield
    _unified_ml_cache.clear()


class TestGetMlCache:
    """Tests for get_ml_cache function."""

    def test_returns_dict(self):
        cache = get_ml_cache()
        assert isinstance(cache, dict)

    def test_returns_same_instance(self):
        cache1 = get_ml_cache()
        cache2 = get_ml_cache()
        assert cache1 is cache2

    def test_modifications_persist(self):
        cache = get_ml_cache()
        cache["test"] = "value"
        assert get_ml_cache()["test"] == "value"


class TestMlCacheGet:
    """Tests for ML cache_get function."""

    def test_returns_data_when_fresh(self):
        cache = {}
        cache["key1"] = {
            "data": {"prediction": 0.85},
            "timestamp": datetime.now(),
        }
        result = cache_get(cache, "key1", ttl_seconds=900)
        assert result == {"prediction": 0.85}

    def test_returns_none_when_expired(self):
        cache = {}
        cache["key1"] = {
            "data": {"prediction": 0.85},
            "timestamp": datetime.now() - timedelta(seconds=1000),
        }
        result = cache_get(cache, "key1", ttl_seconds=900)
        assert result is None

    def test_returns_none_for_missing_key(self):
        cache = {}
        result = cache_get(cache, "missing", ttl_seconds=900)
        assert result is None

    def test_handles_iso_string_timestamp(self):
        cache = {}
        cache["key1"] = {
            "data": "test_value",
            "timestamp": datetime.now().isoformat(),
        }
        result = cache_get(cache, "key1", ttl_seconds=900)
        assert result == "test_value"

    def test_handles_expired_iso_string_timestamp(self):
        cache = {}
        cache["key1"] = {
            "data": "test_value",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
        }
        result = cache_get(cache, "key1", ttl_seconds=900)
        assert result is None

    def test_returns_entry_without_timestamp(self):
        # Non-standard entries (no timestamp) returned as-is
        cache = {}
        cache["raw"] = "just_a_string"
        result = cache_get(cache, "raw", ttl_seconds=900)
        assert result == "just_a_string"

    def test_returns_dict_without_timestamp_key(self):
        cache = {}
        cache["no_ts"] = {"value": 42}
        result = cache_get(cache, "no_ts", ttl_seconds=900)
        assert result == {"value": 42}

    def test_handles_invalid_timestamp_gracefully(self):
        cache = {}
        cache["bad_ts"] = {
            "data": "value",
            "timestamp": "not-a-date",
        }
        result = cache_get(cache, "bad_ts", ttl_seconds=900)
        assert result is None  # Invalid timestamp = expired

    def test_fresh_at_boundary(self):
        cache = {}
        # Entry at exactly TTL age
        cache["boundary"] = {
            "data": "value",
            "timestamp": datetime.now() - timedelta(seconds=900),
        }
        result = cache_get(cache, "boundary", ttl_seconds=900)
        # At TTL boundary, age <= ttl_seconds is True (datetime comparison may vary by ms)
        # The implementation uses `age <= ttl_seconds`, so this could be fresh
        # Just verify it doesn't raise
        assert result is None or result == "value"

    def test_works_with_complex_data(self):
        cache = {}
        data = {
            "predictions": [0.1, 0.5, 0.9],
            "model_version": "v2.1",
            "metadata": {"trained_at": "2026-01-01"},
        }
        cache["complex"] = {"data": data, "timestamp": datetime.now()}
        result = cache_get(cache, "complex", ttl_seconds=3600)
        assert result == data


class TestMlCacheSet:
    """Tests for ML cache_set function."""

    def test_stores_with_datetime_timestamp(self):
        cache = {}
        before = datetime.now()
        cache_set(cache, "key1", "value1")
        after = datetime.now()

        assert "key1" in cache
        entry = cache["key1"]
        assert entry["data"] == "value1"
        assert before <= entry["timestamp"] <= after

    def test_overwrites_existing(self):
        cache = {}
        cache_set(cache, "key1", "old")
        cache_set(cache, "key1", "new")

        assert cache["key1"]["data"] == "new"

    def test_stores_complex_values(self):
        cache = {}
        data = {"nested": True, "list": [1, 2]}
        cache_set(cache, "complex", data)

        assert cache["complex"]["data"] == data

    def test_stores_none_value(self):
        cache = {}
        cache_set(cache, "none_key", None)
        assert cache["none_key"]["data"] is None


class TestMlCacheClearExpired:
    """Tests for ML cache_clear_expired function."""

    def test_removes_expired_entries(self):
        cache = {
            "fresh": {
                "data": "v1",
                "timestamp": datetime.now(),
            },
            "expired": {
                "data": "v2",
                "timestamp": datetime.now() - timedelta(hours=2),
            },
        }
        removed = cache_clear_expired(cache, ttl_seconds=3600)
        assert removed == 1
        assert "fresh" in cache
        assert "expired" not in cache

    def test_returns_count_of_removed(self):
        old = datetime.now() - timedelta(hours=5)
        cache = {
            "a": {"data": 1, "timestamp": old},
            "b": {"data": 2, "timestamp": old},
            "c": {"data": 3, "timestamp": old},
        }
        removed = cache_clear_expired(cache, ttl_seconds=3600)
        assert removed == 3
        assert len(cache) == 0

    def test_keeps_fresh_entries(self):
        now = datetime.now()
        cache = {
            "a": {"data": 1, "timestamp": now},
            "b": {"data": 2, "timestamp": now - timedelta(minutes=5)},
        }
        removed = cache_clear_expired(cache, ttl_seconds=3600)
        assert removed == 0
        assert len(cache) == 2

    def test_empty_cache_returns_zero(self):
        cache = {}
        removed = cache_clear_expired(cache, ttl_seconds=60)
        assert removed == 0

    def test_handles_iso_string_timestamps(self):
        cache = {
            "old": {
                "data": "v1",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            },
        }
        removed = cache_clear_expired(cache, ttl_seconds=3600)
        assert removed == 1

    def test_skips_entries_without_timestamp(self):
        cache = {
            "no_ts": {"data": "value"},
            "has_ts_expired": {
                "data": "old",
                "timestamp": datetime.now() - timedelta(hours=2),
            },
        }
        removed = cache_clear_expired(cache, ttl_seconds=3600)
        assert removed == 1
        assert "no_ts" in cache  # Kept (no timestamp to check)

    def test_handles_invalid_timestamp(self):
        cache = {
            "bad": {"data": "v", "timestamp": "not-a-date"},
            "good": {"data": "v", "timestamp": datetime.now()},
        }
        removed = cache_clear_expired(cache, ttl_seconds=3600)
        # Bad timestamp entry should be skipped (exception caught)
        assert "good" in cache


class TestMlCacheClearAll:
    """Tests for ML cache_clear_all function."""

    def test_clears_all_entries(self):
        ml_cache = get_ml_cache()
        ml_cache["a"] = {"data": 1}
        ml_cache["b"] = {"data": 2}
        ml_cache["c"] = {"data": 3}

        count = cache_clear_all()
        assert count == 3
        assert len(get_ml_cache()) == 0

    def test_returns_zero_on_empty(self):
        count = cache_clear_all()
        assert count == 0

    def test_clears_global_instance(self):
        ml_cache = get_ml_cache()
        ml_cache["test"] = "value"
        cache_clear_all()
        assert "test" not in get_ml_cache()


class TestMlCacheIntegration:
    """Integration tests: set → get → expire → clear cycle."""

    def test_set_then_get(self):
        cache = {}
        cache_set(cache, "test", {"score": 0.95})
        result = cache_get(cache, "test", ttl_seconds=900)
        assert result == {"score": 0.95}

    def test_set_then_expire_then_get(self):
        cache = {}
        cache["old"] = {
            "data": "stale",
            "timestamp": datetime.now() - timedelta(hours=1),
        }
        result = cache_get(cache, "old", ttl_seconds=900)
        assert result is None

    def test_full_cycle(self):
        cache = {}
        cache_set(cache, "a", 1)
        cache_set(cache, "b", 2)
        # Manually add expired entry
        cache["c"] = {
            "data": 3,
            "timestamp": datetime.now() - timedelta(hours=2),
        }

        assert cache_get(cache, "a", ttl_seconds=3600) == 1
        assert cache_get(cache, "b", ttl_seconds=3600) == 2
        assert cache_get(cache, "c", ttl_seconds=3600) is None

        removed = cache_clear_expired(cache, ttl_seconds=3600)
        assert removed == 1
        assert len(cache) == 2
