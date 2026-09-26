"""Shared, user-independent market breadth calculations."""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)
_CACHE_TTL_SECONDS = 300
_cached_result: dict[str, Any] | None = None
_cached_at = 0.0
_cache_lock = asyncio.Lock()


def _empty_result(status: str, *, error: str | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "status": status,
        "source": "coingecko",
        "timestamp": datetime.now().isoformat(),
        "assets_analyzed": 0,
    }
    if error:
        meta["error"] = error
    return {
        "advance_decline_ratio": 0.5,
        "new_highs_count": 0,
        "volume_concentration": 0.5,
        "momentum_dispersion": 0.5,
        "meta": meta,
    }


async def _fetch_global_market_data(limit: int) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "24h",
            },
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []


def calculate_market_breadth(market_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate global advance/decline, near-ATH, volume, and momentum metrics."""
    if not market_data:
        return _empty_result("no_data")

    returns = [
        float(coin["price_change_percentage_24h"])
        for coin in market_data
        if coin.get("price_change_percentage_24h") is not None
    ]
    advancing = sum(value > 0 for value in returns)
    total = len(returns)
    volumes = [max(0.0, float(coin.get("total_volume") or 0)) for coin in market_data]
    total_volume = sum(volumes)
    volume_concentration = sum(sorted(volumes, reverse=True)[:10]) / total_volume if total_volume else 0.5
    dispersion = min(1.0, statistics.stdev(returns) / 10.0) if len(returns) > 1 else 0.5
    near_ath = sum(
        coin.get("ath_change_percentage") is not None
        and float(coin["ath_change_percentage"]) >= -5
        for coin in market_data
    )
    return {
        "advance_decline_ratio": round(advancing / total, 3) if total else 0.5,
        "new_highs_count": near_ath,
        "volume_concentration": round(volume_concentration, 3),
        "momentum_dispersion": round(dispersion, 3),
        "meta": {
            "assets_analyzed": len(market_data),
            "advancing_assets": advancing,
            "declining_assets": total - advancing,
            "total_assets": total,
            "timestamp": datetime.now().isoformat(),
            "source": "coingecko_global_top100",
        },
    }


async def get_market_breadth_metrics(limit: int = 100) -> dict[str, Any]:
    """Get cached global market breadth; never reads user portfolio data."""
    global _cached_result, _cached_at
    async with _cache_lock:
        if _cached_result is not None and time.monotonic() - _cached_at < _CACHE_TTL_SECONDS:
            return dict(_cached_result)
        try:
            result = calculate_market_breadth(await _fetch_global_market_data(limit))
            if result["meta"].get("status") != "no_data":
                _cached_result = result
                _cached_at = time.monotonic()
            return result
        except httpx.TimeoutException as exc:
            logger.warning("CoinGecko market breadth request timed out: %s", exc)
            return _empty_result("error", error="timeout")
        except Exception as exc:
            logger.warning("Failed to fetch market breadth: %s", exc)
            return _empty_result("error", error=type(exc).__name__)