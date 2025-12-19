"""
Unified Cache Management Service

Provides centralized management for all application caches:
- In-memory caches (analytics, ML, risk, etc.)
- Redis cache (optional)
- Cache statistics and clearing

Used by Admin Dashboard for cache monitoring and maintenance.
"""
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CacheManager:
    """Unified cache manager for all application caches"""

    def __init__(self):
        """Initialize cache manager with registry of known caches"""
        self._cache_registry: Dict[str, Dict] = {}
        self._cache_metadata: Dict[str, Dict[str, Any]] = {}
        self._register_known_caches()

    def _register_known_caches(self):
        """Register all known application caches"""
        # Import caches from various modules
        try:
            from api.advanced_analytics_endpoints import _advanced_cache
            self.register_cache("advanced_analytics", _advanced_cache, ttl=1800)
        except ImportError:
            logger.warning("Could not import advanced_analytics cache")

        try:
            from api.analytics_endpoints import _analytics_cache
            self.register_cache("analytics", _analytics_cache, ttl=900)
        except ImportError:
            logger.warning("Could not import analytics cache")

        try:
            from api.risk_endpoints import _risk_cache
            self.register_cache("risk", _risk_cache, ttl=1800)
        except ImportError:
            logger.warning("Could not import risk cache")

        try:
            from api.unified_ml_endpoints import _unified_ml_cache
            self.register_cache("unified_ml", _unified_ml_cache, ttl=900)
        except ImportError:
            logger.warning("Could not import unified_ml cache")

        try:
            from api.ml_crypto_endpoints import _regime_history_cache
            self.register_cache("ml_regime_history", _regime_history_cache, ttl=3600)
        except ImportError:
            logger.warning("Could not import ml_regime_history cache")

        try:
            from api.execution.signals_endpoints import _RECOMPUTE_CACHE
            self.register_cache("execution_signals", _RECOMPUTE_CACHE, ttl=1800)
        except ImportError:
            logger.warning("Could not import execution_signals cache")

        # CoinGecko proxy cache (special handling - nested in function scope)
        self._cache_metadata["coingecko_proxy"] = {
            "ttl": 900,
            "description": "CoinGecko API proxy cache",
            "type": "function_scope"
        }

        logger.info(f"✅ Cache manager initialized with {len(self._cache_registry)} registered caches")

    def register_cache(self, name: str, cache_dict: Dict, ttl: int = 3600, description: str = ""):
        """Register a cache for management"""
        self._cache_registry[name] = cache_dict
        self._cache_metadata[name] = {
            "ttl": ttl,
            "description": description or f"{name.replace('_', ' ').title()} Cache",
            "type": "in_memory"
        }
        logger.debug(f"Registered cache: {name} (TTL: {ttl}s)")

    def get_cache_stats(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for one or all caches"""
        if cache_name:
            return self._get_single_cache_stats(cache_name)

        # Get stats for all caches
        stats = {
            "total_caches": len(self._cache_registry) + 1,  # +1 for CoinGecko
            "caches": {},
            "total_entries": 0,
            "total_size_kb": 0,
            "timestamp": datetime.utcnow().isoformat()
        }

        for name in self._cache_registry.keys():
            cache_stats = self._get_single_cache_stats(name)
            stats["caches"][name] = cache_stats
            stats["total_entries"] += cache_stats.get("entries", 0)

        # Add CoinGecko special case
        stats["caches"]["coingecko_proxy"] = {
            "name": "coingecko_proxy",
            "entries": 0,  # Cannot introspect function-scoped cache
            "type": "function_scope",
            "ttl": 900,
            "description": "CoinGecko API proxy cache"
        }

        return stats

    def _get_single_cache_stats(self, cache_name: str) -> Dict[str, Any]:
        """Get stats for a single cache"""
        if cache_name not in self._cache_registry:
            return {
                "name": cache_name,
                "error": "Cache not found or not registered",
                "available_caches": list(self._cache_registry.keys())
            }

        cache = self._cache_registry[cache_name]
        metadata = self._cache_metadata.get(cache_name, {})

        # Calculate stats
        now = time.time()
        ttl = metadata.get("ttl", 3600)

        total_entries = len(cache)
        expired_entries = 0
        valid_entries = 0

        for key, value in cache.items():
            if isinstance(value, tuple) and len(value) == 2:
                _, ts = value
                if now - ts >= ttl:
                    expired_entries += 1
                else:
                    valid_entries += 1
            else:
                # Non-standard format (no timestamp)
                valid_entries += 1

        return {
            "name": cache_name,
            "entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "ttl": ttl,
            "type": metadata.get("type", "in_memory"),
            "description": metadata.get("description", "")
        }

    def clear_cache(self, cache_name: str, admin_user: str = "system") -> Dict[str, Any]:
        """Clear a specific cache"""
        if cache_name == "all":
            return self._clear_all_caches(admin_user)

        if cache_name == "coingecko_proxy":
            # Special case: cannot clear function-scoped cache
            logger.warning(f"⚠️ Cannot clear coingecko_proxy cache (function-scoped) - requested by {admin_user}")
            return {
                "ok": False,
                "cache": cache_name,
                "error": "Cannot clear function-scoped cache. Restart server to clear.",
                "cleared_entries": 0
            }

        if cache_name not in self._cache_registry:
            logger.warning(f"❌ Cache '{cache_name}' not found - requested by {admin_user}")
            return {
                "ok": False,
                "cache": cache_name,
                "error": "Cache not found",
                "available_caches": list(self._cache_registry.keys())
            }

        cache = self._cache_registry[cache_name]
        entries_before = len(cache)
        cache.clear()

        logger.info(f"✅ Cache '{cache_name}' cleared ({entries_before} entries) by {admin_user}")

        return {
            "ok": True,
            "cache": cache_name,
            "cleared_entries": entries_before,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _clear_all_caches(self, admin_user: str) -> Dict[str, Any]:
        """Clear all registered caches"""
        results = []
        total_cleared = 0

        for cache_name in self._cache_registry.keys():
            cache = self._cache_registry[cache_name]
            entries_before = len(cache)
            cache.clear()
            total_cleared += entries_before
            results.append({
                "cache": cache_name,
                "cleared_entries": entries_before
            })

        logger.info(f"✅ All caches cleared ({total_cleared} total entries) by {admin_user}")

        return {
            "ok": True,
            "cache": "all",
            "total_cleared": total_cleared,
            "caches_cleared": len(results),
            "details": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    def clear_expired(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """Clear expired entries from one or all caches"""
        if cache_name and cache_name in self._cache_registry:
            return self._clear_expired_single(cache_name)

        # Clear expired from all caches
        total_cleared = 0
        results = []

        for name in self._cache_registry.keys():
            result = self._clear_expired_single(name)
            total_cleared += result.get("cleared_entries", 0)
            results.append(result)

        return {
            "ok": True,
            "total_cleared": total_cleared,
            "caches_processed": len(results),
            "details": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _clear_expired_single(self, cache_name: str) -> Dict[str, Any]:
        """Clear expired entries from a single cache"""
        cache = self._cache_registry[cache_name]
        metadata = self._cache_metadata.get(cache_name, {})
        ttl = metadata.get("ttl", 3600)

        now = time.time()
        expired_keys = []

        for key, value in list(cache.items()):
            if isinstance(value, tuple) and len(value) == 2:
                _, ts = value
                if now - ts >= ttl:
                    expired_keys.append(key)

        for key in expired_keys:
            del cache[key]

        return {
            "cache": cache_name,
            "cleared_entries": len(expired_keys)
        }

    def get_available_caches(self) -> List[str]:
        """Get list of all available cache names"""
        return list(self._cache_registry.keys()) + ["coingecko_proxy"]


# Global cache manager instance
cache_manager = CacheManager()
