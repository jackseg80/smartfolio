"""
SaxoBank UIC Resolver with Redis Cache

Resolves UICs (Unique Instrument Codes) to instrument metadata:
- Symbol (e.g., "AAPL")
- Name/Description (e.g., "Apple Inc.")
- ISIN (e.g., "US0378331005")

Architecture:
- Redis cache (TTL: 7 days) - Instrument metadata rarely changes
- Fallback: In-memory cache if Redis unavailable
- Batch resolution support (future)

Usage:
    resolver = SaxoUICResolver(user_id="jack")
    instrument = await resolver.resolve_uic(access_token, uic=34909, asset_type="Etf")
    # → {"symbol": "AAPL", "name": "Apple Inc.", "isin": "US0378331005"}
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, Optional
from datetime import timedelta

from connectors.saxo_api import SaxoOAuthClient

logger = logging.getLogger(__name__)

# In-memory fallback cache (if Redis unavailable)
_uic_cache: Dict[str, Dict[str, Any]] = {}


class SaxoUICResolver:
    """
    Resolves SaxoBank UICs to instrument metadata with caching.

    Cache Strategy:
        - Primary: Redis (TTL: 7 days)
        - Fallback: In-memory dict
        - Cache key: saxo:uic:{uic}:{asset_type}

    Why 7 days TTL?
        - Instrument metadata (Symbol, Name, ISIN) rarely changes
        - Even if ticker changes, 7 days delay is acceptable
        - Reduces API calls significantly (120 positions → 0-5 calls)
    """

    def __init__(self, user_id: str):
        """
        Initialize resolver for a specific user.

        Args:
            user_id: User identifier (for logging/debugging)
        """
        self.user_id = user_id
        self.redis_client = self._get_redis_client()
        self.ttl_days = 7

    def _get_redis_client(self):
        """
        Get Redis client if available.

        Returns:
            Redis client or None
        """
        try:
            from api.deps import get_redis_client
            client = get_redis_client()
            if client:
                logger.debug(f"✅ Redis connected for UIC resolver (user: {self.user_id})")
                return client
            else:
                logger.warning(f"⚠️ Redis unavailable for UIC resolver - Using in-memory cache")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Redis error for UIC resolver: {e} - Using in-memory cache")
            return None

    async def resolve_uic(
        self,
        access_token: str,
        uic: int,
        asset_type: str = "Stock"
    ) -> Optional[Dict[str, str]]:
        """
        Resolve UIC to instrument metadata.

        Args:
            access_token: Valid Saxo Bearer token
            uic: Unique Instrument Code
            asset_type: Asset type (Stock, Etf, Bond, etc.)

        Returns:
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "isin": "US0378331005",
                "currency": "USD"
            }

        Returns None if resolution fails (network error, 404, etc.)

        Caching:
            - Cache hit → return immediately
            - Cache miss → fetch from API + cache result
        """
        cache_key = f"saxo:uic:{uic}:{asset_type}"

        # Try cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.debug(f"💾 Cache HIT for UIC {uic} ({asset_type})")
            return cached

        # Cache miss → fetch from API
        logger.debug(f"🔄 Cache MISS for UIC {uic} ({asset_type}) - Fetching from API")

        oauth_client = SaxoOAuthClient()
        instrument_data = await oauth_client.get_instrument_details(
            access_token=access_token,
            uic=uic,
            asset_type=asset_type
        )

        if not instrument_data:
            logger.warning(f"⚠️ Failed to resolve UIC {uic} ({asset_type})")
            return None

        # Extract relevant fields
        symbol_raw = instrument_data.get("Symbol", "")
        symbol = symbol_raw.split(":")[0] if ":" in symbol_raw else symbol_raw

        resolved = {
            "symbol": symbol or f"UIC-{uic}",
            "name": instrument_data.get("Description", f"Instrument {uic}"),
            "isin": instrument_data.get("Isin", ""),
            "currency": instrument_data.get("Currency", "")
        }

        # Cache result
        self._set_in_cache(cache_key, resolved)

        logger.info(f"✅ Resolved UIC {uic} → {resolved['symbol']} ({resolved['name']})")

        return resolved

    async def resolve_batch(
        self,
        access_token: str,
        uic_list: list[tuple[int, str]]
    ) -> Dict[int, Dict[str, str]]:
        """
        Resolve multiple UICs in batch (future optimization).

        Args:
            access_token: Valid Saxo Bearer token
            uic_list: List of (uic, asset_type) tuples

        Returns:
            {
                34909: {"symbol": "AAPL", "name": "Apple Inc.", ...},
                12345: {"symbol": "MSFT", "name": "Microsoft Corp.", ...},
                ...
            }

        Note: Currently resolves sequentially. Future: Use asyncio.gather
              for parallel resolution (rate limit aware).
        """
        results = {}

        for uic, asset_type in uic_list:
            resolved = await self.resolve_uic(access_token, uic, asset_type)
            if resolved:
                results[uic] = resolved

        logger.info(f"✅ Batch resolved {len(results)}/{len(uic_list)} UICs")

        return results

    def _get_from_cache(self, key: str) -> Optional[Dict[str, str]]:
        """
        Get instrument metadata from cache.

        Args:
            key: Cache key (saxo:uic:{uic}:{asset_type})

        Returns:
            Instrument metadata dict or None
        """
        # Try Redis first
        if self.redis_client:
            try:
                cached_json = self.redis_client.get(key)
                if cached_json:
                    return json.loads(cached_json)
            except Exception as e:
                logger.warning(f"⚠️ Redis cache read error: {e}")

        # Fallback to in-memory
        return _uic_cache.get(key)

    def _set_in_cache(self, key: str, value: Dict[str, str]) -> None:
        """
        Store instrument metadata in cache.

        Args:
            key: Cache key (saxo:uic:{uic}:{asset_type})
            value: Instrument metadata dict
        """
        # Store in Redis
        if self.redis_client:
            try:
                ttl_seconds = int(timedelta(days=self.ttl_days).total_seconds())
                self.redis_client.setex(
                    key,
                    ttl_seconds,
                    json.dumps(value)
                )
                logger.debug(f"💾 Cached in Redis: {key} (TTL: {self.ttl_days} days)")
            except Exception as e:
                logger.warning(f"⚠️ Redis cache write error: {e}")

        # Store in-memory (fallback)
        _uic_cache[key] = value

    def clear_cache(self, uic: Optional[int] = None) -> None:
        """
        Clear cache for specific UIC or all UICs.

        Args:
            uic: UIC to clear (None = clear all)

        Use Cases:
            - Instrument metadata changed (rare)
            - Debug/testing
        """
        if uic:
            # Clear specific UIC (all asset types)
            pattern = f"saxo:uic:{uic}:*"
            if self.redis_client:
                try:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                        logger.info(f"🗑️ Cleared Redis cache for UIC {uic}")
                except Exception as e:
                    logger.warning(f"⚠️ Redis cache clear error: {e}")

            # Clear in-memory
            keys_to_delete = [k for k in _uic_cache if f":uic:{uic}:" in k]
            for key in keys_to_delete:
                del _uic_cache[key]

        else:
            # Clear all
            if self.redis_client:
                try:
                    pattern = "saxo:uic:*"
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                        logger.info(f"🗑️ Cleared all Redis UIC cache")
                except Exception as e:
                    logger.warning(f"⚠️ Redis cache clear error: {e}")

            _uic_cache.clear()
            logger.info(f"🗑️ Cleared all in-memory UIC cache")
