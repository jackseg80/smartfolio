"""
CoinGecko Proxy Router
Provides a backend proxy for CoinGecko API to avoid CORS and rate limiting issues.

Features:
- Caching (15 minutes default - optimized Oct 2025) to reduce API calls
- CORS-free (backend-to-backend)
- Automatic fallback to cached data on API failures
- Rate limit handling
- Multi-tenant: uses user's CoinGecko API key from secrets.json
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, Optional
import asyncio
import httpx
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from services.user_secrets import get_coingecko_api_key
from api.deps import get_required_user
from shared.circuit_breaker import coingecko_circuit

logger = logging.getLogger("crypto-rebalancer")

def get_user_coingecko_key(user_id: str) -> Optional[str]:
    """
    Load CoinGecko API key from user's secrets.json.
    Falls back to .env COINGECKO_API_KEY if user secrets don't have one.

    Args:
        user_id: User ID

    Returns:
        CoinGecko API key or None
    """
    # Use UserSecretsManager via helper function
    key = get_coingecko_api_key(user_id)

    if key:
        logger.debug(f"Using CoinGecko key from user {user_id} secrets")
        return key

    # Fallback to .env (user secrets may not have a CoinGecko key)
    env_key = os.getenv("COINGECKO_API_KEY", "")
    if env_key:
        logger.debug(f"Using CoinGecko key from .env for user {user_id}")
        return env_key

    logger.debug(f"No CoinGecko API key found for user {user_id}")
    return None

router = APIRouter(prefix="/api/coingecko-proxy", tags=["coingecko-proxy"])

# Simple in-memory cache
# Format: {cache_key: {"data": {...}, "timestamp": datetime, "ttl": int}}
_cache: Dict[str, Dict[str, Any]] = {}
_cache_writes = 0  # Counter for periodic cleanup

def get_cached_data(cache_key: str, ttl_seconds: int = 300) -> Optional[Dict[str, Any]]:
    """
    Get cached data if available and not expired.

    Args:
        cache_key: Unique key for the cached data
        ttl_seconds: Time-to-live in seconds (default 5 minutes)

    Returns:
        Cached data or None if expired/not found
    """
    if cache_key not in _cache:
        return None

    cached = _cache[cache_key]
    age = (datetime.now() - cached["timestamp"]).total_seconds()

    if age > ttl_seconds:
        logger.debug(f"Cache expired for {cache_key} (age: {age:.1f}s, ttl: {ttl_seconds}s)")
        del _cache[cache_key]
        return None

    logger.debug(f"Cache hit for {cache_key} (age: {age:.1f}s)")
    return cached["data"]

def cleanup_expired_cache() -> int:
    """
    Proactively remove expired entries from cache.

    PERFORMANCE FIX (Dec 2025): Prevents memory leak from never-accessed expired entries.

    Returns:
        Number of entries removed
    """
    now = datetime.now()
    expired_keys = []

    for cache_key, cached in _cache.items():
        age = (now - cached["timestamp"]).total_seconds()
        if age > cached["ttl"]:
            expired_keys.append(cache_key)

    for key in expired_keys:
        del _cache[key]

    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    return len(expired_keys)

def set_cached_data(cache_key: str, data: Dict[str, Any], ttl_seconds: int = 300):
    """
    Store data in cache with timestamp.

    Args:
        cache_key: Unique key for the cached data
        data: Data to cache
        ttl_seconds: Time-to-live in seconds
    """
    global _cache_writes

    _cache[cache_key] = {
        "data": data,
        "timestamp": datetime.now(),
        "ttl": ttl_seconds
    }
    logger.debug(f"Cached data for {cache_key} (ttl: {ttl_seconds}s)")

    # PERFORMANCE FIX: Periodic cleanup every 10 writes to prevent memory leak
    _cache_writes += 1
    if _cache_writes % 10 == 0:
        cleanup_expired_cache()

async def _fetch_with_cache_and_fallback(
    url: str,
    cache_key: str,
    cache_ttl: int,
    params: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Helper function to fetch from CoinGecko with caching and stale fallback.

    Args:
        url: CoinGecko API URL
        cache_key: Unique cache key
        cache_ttl: Cache TTL in seconds
        params: Optional query parameters
        api_key: CoinGecko API key (from user config)

    Returns:
        Response dict with data, cached status, etc.
    """
    # Try cache first
    cached = get_cached_data(cache_key, cache_ttl)
    if cached:
        # Return data directly (frontend expects raw CoinGecko response)
        return cached

    # Circuit breaker: fail-fast with stale cache fallback
    if not coingecko_circuit.is_available():
        logger.warning(f"CoinGecko circuit OPEN — using stale cache for {cache_key}")
        if cache_key in _cache:
            return _cache[cache_key]["data"]
        raise HTTPException(status_code=503, detail="CoinGecko API unavailable (circuit open) and no cache")

    max_retries = 3
    headers = {}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(max_retries + 1):
                logger.debug(f"Fetching from CoinGecko: {url} (attempt {attempt + 1})")
                response = await client.get(url, params=params, headers=headers)

                if response.status_code == 429:
                    if attempt < max_retries:
                        backoff = 2 ** attempt  # 1s, 2s, 4s
                        logger.warning(f"CoinGecko 429 — retry {attempt + 1}/{max_retries} in {backoff}s")
                        await asyncio.sleep(backoff)
                        continue
                    logger.warning(f"CoinGecko 429 — exhausted {max_retries} retries")
                    # Fall through to stale cache below
                    if cache_key in _cache:
                        stale_data = _cache[cache_key]["data"]
                        age = (datetime.now() - _cache[cache_key]["timestamp"]).total_seconds()
                        logger.info(f"Using stale cache (age: {age:.1f}s) after rate limit retries exhausted")
                        return stale_data
                    raise HTTPException(status_code=429, detail="CoinGecko API rate limit exceeded and no cache available")

                response.raise_for_status()
                data = response.json()

                # Cache the response
                set_cached_data(cache_key, data, cache_ttl)
                coingecko_circuit.record_success()

                logger.info(f"Successfully fetched data from CoinGecko: {url}")
                return data

    except httpx.TimeoutException:
        logger.error("CoinGecko API timeout")
        coingecko_circuit.record_failure()
        if cache_key in _cache:
            stale_data = _cache[cache_key]["data"]
            age = (datetime.now() - _cache[cache_key]["timestamp"]).total_seconds()
            logger.info(f"Using stale cache (age: {age:.1f}s) due to timeout")
            return stale_data
        raise HTTPException(status_code=504, detail="CoinGecko API timeout and no cache available")

    except httpx.HTTPStatusError as e:
        logger.error(f"CoinGecko API error: {e.response.status_code}")
        coingecko_circuit.record_failure()
        if cache_key in _cache:
            stale_data = _cache[cache_key]["data"]
            age = (datetime.now() - _cache[cache_key]["timestamp"]).total_seconds()
            logger.info(f"Using stale cache (age: {age:.1f}s) due to API error {e.response.status_code}")
            return stale_data
        raise HTTPException(status_code=e.response.status_code, detail=f"CoinGecko API error and no cache available")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error fetching CoinGecko data: {e}")
        coingecko_circuit.record_failure()
        if cache_key in _cache:
            stale_data = _cache[cache_key]["data"]
            age = (datetime.now() - _cache[cache_key]["timestamp"]).total_seconds()
            logger.info(f"Using stale cache (age: {age:.1f}s) due to unexpected error")
            return stale_data
        logger.warning(f"No cache available for {cache_key}, returning empty fallback")
        return {}

@router.get("/bitcoin")
async def get_bitcoin_data(
    localization: bool = Query(False, description="Include localization data"),
    tickers: bool = Query(False, description="Include tickers data"),
    market_data: bool = Query(True, description="Include market data"),
    community_data: bool = Query(False, description="Include community data"),
    developer_data: bool = Query(False, description="Include developer data"),
    sparkline: bool = Query(False, description="Include sparkline data"),
    cache_ttl: int = Query(900, description="Cache TTL in seconds (default 15min)", ge=60, le=7200),
    user: str = Depends(get_required_user)
) -> dict:
    """
    Proxy endpoint for CoinGecko Bitcoin data (multi-tenant).

    This endpoint caches responses to avoid rate limiting and provides
    graceful degradation when CoinGecko API is unavailable.

    Uses the CoinGecko API key from the user's config.json.

    Example:
        GET /api/coingecko-proxy/bitcoin?market_data=true

    Headers:
        X-User: Required user ID

    Returns:
        Bitcoin market data from CoinGecko API (or cached data)
    """

    # Build cache key from parameters (include user for multi-tenant cache)
    cache_key = f"bitcoin_{user}_{localization}_{tickers}_{market_data}_{community_data}_{developer_data}_{sparkline}"

    # Build CoinGecko URL and params
    url = "https://api.coingecko.com/api/v3/coins/bitcoin"
    params = {
        "localization": str(localization).lower(),
        "tickers": str(tickers).lower(),
        "market_data": str(market_data).lower(),
        "community_data": str(community_data).lower(),
        "developer_data": str(developer_data).lower(),
        "sparkline": str(sparkline).lower()
    }

    # Get user's CoinGecko API key
    api_key = get_user_coingecko_key(user)

    return await _fetch_with_cache_and_fallback(url, cache_key, cache_ttl, params, api_key)

@router.get("/global")
async def get_global_data(
    cache_ttl: int = Query(900, description="Cache TTL in seconds (default 15min)", ge=60, le=7200),
    user: str = Depends(get_required_user)
) -> dict:
    """
    Proxy endpoint for CoinGecko global cryptocurrency market data (multi-tenant).
    Used for BTC dominance and market cap data.

    Example:
        GET /api/coingecko-proxy/global

    Headers:
        X-User: Required user ID

    Returns:
        Global market data from CoinGecko API (or cached data)
    """
    cache_key = f"global_{user}"
    url = "https://api.coingecko.com/api/v3/global"

    # Get user's CoinGecko API key
    api_key = get_user_coingecko_key(user)

    return await _fetch_with_cache_and_fallback(url, cache_key, cache_ttl, None, api_key)

@router.get("/simple/price")
async def get_simple_price(
    ids: str = Query(..., description="Comma-separated coin IDs (e.g., bitcoin,ethereum)"),
    vs_currencies: str = Query("usd", description="Comma-separated fiat currencies"),
    cache_ttl: int = Query(180, description="Cache TTL in seconds (default 3min for prices)", ge=60, le=7200),
    user: str = Depends(get_required_user)
) -> dict:
    """
    Proxy endpoint for CoinGecko simple price endpoint (multi-tenant).
    Used for getting current prices of multiple coins.

    Example:
        GET /api/coingecko-proxy/simple/price?ids=bitcoin,ethereum&vs_currencies=usd

    Headers:
        X-User: Required user ID

    Returns:
        Simple price data from CoinGecko API (or cached data)
    """
    cache_key = f"simple_price_{user}_{ids}_{vs_currencies}"
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ids,
        "vs_currencies": vs_currencies
    }

    # Get user's CoinGecko API key
    api_key = get_user_coingecko_key(user)

    return await _fetch_with_cache_and_fallback(url, cache_key, cache_ttl, params, api_key)

@router.get("/market_chart")
async def get_market_chart(
    coin_id: str = Query("bitcoin", description="Coin ID (e.g., bitcoin)"),
    vs_currency: str = Query("usd", description="Fiat currency"),
    days: int = Query(7, description="Number of days", ge=1, le=730),
    interval: str = Query("daily", description="Data interval"),
    cache_ttl: int = Query(900, description="Cache TTL in seconds (default 15min)", ge=60, le=7200),
    user: str = Depends(get_required_user)
) -> dict:
    """
    Proxy endpoint for CoinGecko market chart endpoint (multi-tenant).
    Used for historical price data and volatility calculations.

    Example:
        GET /api/coingecko-proxy/market_chart?coin_id=bitcoin&vs_currency=usd&days=7&interval=daily

    Headers:
        X-User: Required user ID

    Returns:
        Market chart data from CoinGecko API (or cached data)
    """
    cache_key = f"market_chart_{user}_{coin_id}_{vs_currency}_{days}_{interval}"
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": str(days),
        "interval": interval
    }

    # Get user's CoinGecko API key
    api_key = get_user_coingecko_key(user)

    return await _fetch_with_cache_and_fallback(url, cache_key, cache_ttl, params, api_key)

@router.get("/cache/stats")
async def get_cache_stats() -> dict:
    """
    Get cache statistics (useful for debugging).

    Returns:
        Cache size, keys, ages, and expired count
    """
    now = datetime.now()
    expired_count = 0
    entries = []

    for key, value in _cache.items():
        age = (now - value["timestamp"]).total_seconds()
        is_expired = age > value["ttl"]
        if is_expired:
            expired_count += 1

        entries.append({
            "key": key,
            "age_seconds": age,
            "ttl_seconds": value["ttl"],
            "expired": is_expired
        })

    return {
        "cache_size": len(_cache),
        "expired_count": expired_count,
        "cache_writes": _cache_writes,
        "entries": entries
    }

@router.post("/cache/cleanup")
async def trigger_cache_cleanup() -> dict:
    """
    Manually trigger cleanup of expired cache entries.

    Returns:
        Number of entries removed
    """
    removed = cleanup_expired_cache()
    return {
        "removed": removed,
        "cache_size": len(_cache)
    }

@router.delete("/cache/clear")
async def clear_cache() -> dict:
    """
    Clear the entire cache (useful for debugging/testing).

    Returns:
        Number of entries cleared
    """
    global _cache
    count = len(_cache)
    _cache = {}
    logger.info(f"Cleared {count} cache entries")
    return {"cleared": count, "message": f"Cleared {count} cache entries"}
