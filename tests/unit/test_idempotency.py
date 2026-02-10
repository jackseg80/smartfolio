"""Tests for services/alerts/idempotency.py — IdempotencyManager local cache mode"""

import time
import pytest
from unittest.mock import MagicMock, patch
from services.alerts.idempotency import IdempotencyManager, get_idempotency_manager


# ---------------------------------------------------------------------------
# TestIdempotencyManagerInit
# ---------------------------------------------------------------------------
class TestIdempotencyManagerInit:
    def test_default_init_no_redis(self):
        mgr = IdempotencyManager()
        assert mgr.redis_available is False
        assert mgr.default_ttl == 300
        assert mgr._local_cache == {}

    def test_custom_ttl(self):
        mgr = IdempotencyManager(default_ttl_seconds=60)
        assert mgr.default_ttl == 60

    def test_with_redis_client(self):
        fake_redis = MagicMock()
        mgr = IdempotencyManager(redis_client=fake_redis)
        assert mgr.redis_available is True


# ---------------------------------------------------------------------------
# TestCheckAndStoreLocal
# ---------------------------------------------------------------------------
class TestCheckAndStoreLocal:
    def setup_method(self):
        self.mgr = IdempotencyManager(default_ttl_seconds=300)

    def test_new_key_returns_none(self):
        result = self.mgr.check_and_store("key1", {"action": "buy"})
        assert result is None

    def test_duplicate_key_returns_previous_data(self):
        self.mgr.check_and_store("key1", {"action": "buy", "amount": 100})
        result = self.mgr.check_and_store("key1", {"action": "buy", "amount": 100})
        assert result == {"action": "buy", "amount": 100}

    def test_different_keys_independent(self):
        self.mgr.check_and_store("key1", {"a": 1})
        result = self.mgr.check_and_store("key2", {"b": 2})
        assert result is None  # key2 is new

    def test_expired_key_treated_as_new(self):
        self.mgr.check_and_store("key1", {"old": True}, ttl_seconds=1)
        time.sleep(1.1)
        result = self.mgr.check_and_store("key1", {"new": True}, ttl_seconds=300)
        assert result is None  # expired, treated as new

    def test_custom_ttl_overrides_default(self):
        self.mgr.check_and_store("key1", {"data": 1}, ttl_seconds=1)
        entry = self.mgr._local_cache["key1"]
        # TTL should be ~1 second from now
        assert entry["expires_at"] - entry["created_at"] == pytest.approx(1.0, abs=0.1)

    def test_stores_data_correctly(self):
        data = {"action": "sell", "amount": 500, "price": 42.5}
        self.mgr.check_and_store("order-123", data)
        assert "order-123" in self.mgr._local_cache
        assert self.mgr._local_cache["order-123"]["data"] == data


# ---------------------------------------------------------------------------
# TestCleanupExpired
# ---------------------------------------------------------------------------
class TestCleanupExpired:
    def test_cleanup_removes_expired(self):
        mgr = IdempotencyManager(default_ttl_seconds=1)
        mgr._cleanup_interval = 0  # force cleanup every call
        mgr.check_and_store("key1", {"a": 1}, ttl_seconds=1)
        time.sleep(1.1)
        # Trigger cleanup via check_and_store
        mgr._last_cleanup = 0  # force cleanup
        mgr.check_and_store("key2", {"b": 2}, ttl_seconds=300)
        assert "key1" not in mgr._local_cache
        assert "key2" in mgr._local_cache

    def test_cleanup_skipped_within_interval(self):
        mgr = IdempotencyManager(default_ttl_seconds=1)
        mgr._cleanup_interval = 9999  # never auto-cleanup
        mgr.check_and_store("key1", {"a": 1}, ttl_seconds=1)
        time.sleep(1.1)
        # key1 is expired but cleanup interval not reached → stays in cache
        # (but check_and_store for key1 will still see it as expired and remove it)
        result = mgr.check_and_store("key1", {"new": True})
        assert result is None  # expired key treated as new


# ---------------------------------------------------------------------------
# TestInvalidate
# ---------------------------------------------------------------------------
class TestInvalidate:
    def setup_method(self):
        self.mgr = IdempotencyManager(default_ttl_seconds=300)

    def test_invalidate_existing_key(self):
        self.mgr.check_and_store("key1", {"a": 1})
        assert self.mgr.invalidate("key1") is True
        assert "key1" not in self.mgr._local_cache

    def test_invalidate_nonexistent_key(self):
        assert self.mgr.invalidate("nonexistent") is False

    def test_invalidate_then_store_fresh(self):
        self.mgr.check_and_store("key1", {"old": True})
        self.mgr.invalidate("key1")
        result = self.mgr.check_and_store("key1", {"new": True})
        assert result is None  # treated as new after invalidation


# ---------------------------------------------------------------------------
# TestGetStats
# ---------------------------------------------------------------------------
class TestGetStats:
    def test_stats_no_redis(self):
        mgr = IdempotencyManager(default_ttl_seconds=60)
        stats = mgr.get_stats()
        assert stats["redis_available"] is False
        assert stats["default_ttl_seconds"] == 60
        assert stats["local_cache_size"] == 0

    def test_stats_with_entries(self):
        mgr = IdempotencyManager()
        mgr.check_and_store("k1", {"a": 1})
        mgr.check_and_store("k2", {"b": 2})
        stats = mgr.get_stats()
        assert stats["local_cache_size"] == 2

    def test_stats_has_last_cleanup(self):
        mgr = IdempotencyManager()
        stats = mgr.get_stats()
        assert "last_cleanup" in stats


# ---------------------------------------------------------------------------
# TestRedisMode
# ---------------------------------------------------------------------------
class TestRedisMode:
    def test_redis_check_and_store_new_key(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mgr = IdempotencyManager(redis_client=mock_redis)
        result = mgr.check_and_store("key1", {"action": "buy"})
        assert result is None
        mock_redis.setex.assert_called_once()

    def test_redis_check_existing_key(self):
        import json
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({"action": "buy"})
        mgr = IdempotencyManager(redis_client=mock_redis)
        result = mgr.check_and_store("key1", {"action": "buy"})
        assert result == {"action": "buy"}
        mock_redis.setex.assert_not_called()

    def test_redis_error_falls_back_to_local(self):
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("Redis down")
        mgr = IdempotencyManager(redis_client=mock_redis)
        result = mgr.check_and_store("key1", {"data": 1})
        assert result is None  # Fallback to local cache
        assert "key1" in mgr._local_cache

    def test_redis_invalidate(self):
        mock_redis = MagicMock()
        mock_redis.delete.return_value = 1
        mgr = IdempotencyManager(redis_client=mock_redis)
        assert mgr.invalidate("key1") is True
        mock_redis.delete.assert_called_once_with("idempotency:key1")


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_check_and_store_general_exception(self):
        """If both Redis and local fail, check_and_store returns None"""
        mgr = IdempotencyManager()
        with patch.object(mgr, '_local_check_and_store', side_effect=Exception("boom")):
            result = mgr.check_and_store("key1", {"data": 1})
            assert result is None

    def test_invalidate_exception_returns_false(self):
        mgr = IdempotencyManager()
        with patch.object(mgr, '_cache_lock', side_effect=Exception("lock error")):
            # This won't actually raise because invalidate catches Exception
            # Let's mock more precisely
            pass
        # Direct test: invalidate with broken lock
        mgr_redis = IdempotencyManager(redis_client=MagicMock())
        mgr_redis.redis_client.delete.side_effect = Exception("Redis error")
        assert mgr_redis.invalidate("key1") is False


# ---------------------------------------------------------------------------
# TestGetIdempotencyManager
# ---------------------------------------------------------------------------
class TestGetIdempotencyManager:
    def test_returns_singleton(self):
        import services.alerts.idempotency as mod
        old_mgr = mod._idempotency_manager
        mod._idempotency_manager = None
        try:
            mgr1 = get_idempotency_manager()
            mgr2 = get_idempotency_manager()
            assert mgr1 is mgr2
        finally:
            mod._idempotency_manager = old_mgr

    def test_factory_creates_manager(self):
        import services.alerts.idempotency as mod
        old_mgr = mod._idempotency_manager
        mod._idempotency_manager = None
        try:
            mgr = get_idempotency_manager()
            assert isinstance(mgr, IdempotencyManager)
        finally:
            mod._idempotency_manager = old_mgr
