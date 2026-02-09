"""Tests for services/pricing_service.py"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.wealth import PricePoint


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_price_point(instrument_id="BTC", price=50000.0, currency="USD",
                      source="pricing_service"):
    return PricePoint(
        instrument_id=instrument_id,
        ts=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
        price=price,
        currency=currency,
        source=source,
    )

# ---------------------------------------------------------------------------
# TestCachePath
# ---------------------------------------------------------------------------

class TestCachePath:

    def test_replaces_colon_in_id(self):
        from services.pricing_service import _cache_path
        p = _cache_path("CASH:EUR", "daily")
        assert "CASH_EUR" in p.name and ":" not in p.name

    def test_includes_granularity(self):
        from services.pricing_service import _cache_path
        p = _cache_path("BTC", "hourly")
        assert "hourly" in p.name

    def test_daily_granularity(self):
        from services.pricing_service import _cache_path
        p = _cache_path("ETH", "daily")
        assert p.name == "ETH_daily.json"


# ---------------------------------------------------------------------------
# TestTtlFor
# ---------------------------------------------------------------------------

class TestTtlFor:

    @patch("services.pricing_service._CRYPTO_SYMBOLS", {"BTC", "ETH"})
    def test_crypto_short_ttl(self):
        from services.pricing_service import _ttl_for
        assert _ttl_for("BTC") == 180

    @patch("services.pricing_service._CRYPTO_SYMBOLS", {"BTC", "ETH"})
    def test_non_crypto_long_ttl(self):
        from services.pricing_service import _ttl_for
        assert _ttl_for("AAPL") == 1800

    @patch("services.pricing_service._CRYPTO_SYMBOLS", {"BTC", "ETH"})
    def test_case_insensitive(self):
        from services.pricing_service import _ttl_for
        assert _ttl_for("btc") == 180

# ---------------------------------------------------------------------------
# TestLoadCached
# ---------------------------------------------------------------------------

class TestLoadCached:

    def test_returns_none_when_no_file(self, tmp_path):
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            from services.pricing_service import _load_cached
            assert _load_cached("MISSING", "daily", 300) is None

    def test_returns_none_when_expired(self, tmp_path):
        cache_file = tmp_path / "EXPIRED_daily.json"
        cache_file.write_text(json.dumps({
            "instrument_id": "EXPIRED",
            "ts": "2026-02-09T12:00:00+00:00",
            "price": 100.0,
            "currency": "USD",
            "source": "test",
            "fetched_at": time.time() - 9999,
        }))
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            from services.pricing_service import _load_cached
            assert _load_cached("EXPIRED", "daily", 300) is None

    def test_returns_price_point_when_fresh(self, tmp_path):
        cache_file = tmp_path / "FRESH_daily.json"
        cache_file.write_text(json.dumps({
            "instrument_id": "FRESH",
            "ts": "2026-02-09T12:00:00+00:00",
            "price": 42000.0,
            "currency": "USD",
            "source": "test",
            "fetched_at": time.time(),
        }))
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            from services.pricing_service import _load_cached
            result = _load_cached("FRESH", "daily", 300)
            assert result is not None
            assert result.price == 42000.0
            assert result.instrument_id == "FRESH"

    def test_returns_none_when_fetched_at_zero(self, tmp_path):
        cache_file = tmp_path / "ZERO_daily.json"
        cache_file.write_text(json.dumps({
            "instrument_id": "ZERO",
            "ts": "2026-02-09T12:00:00+00:00",
            "price": 100.0,
            "currency": "USD",
            "source": "test",
            "fetched_at": 0,
        }))
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            from services.pricing_service import _load_cached
            assert _load_cached("ZERO", "daily", 300) is None

# ---------------------------------------------------------------------------
# TestStoreCache
# ---------------------------------------------------------------------------

class TestStoreCache:

    def test_stores_valid_json(self, tmp_path):
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            from services.pricing_service import _store_cache
            pp = _make_price_point(instrument_id="ETH", price=3000.0)
            _store_cache(pp, "daily")
            path = tmp_path / "ETH_daily.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["price"] == 3000.0
            assert data["instrument_id"] == "ETH"

    def test_roundtrip_store_then_load(self, tmp_path):
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            from services.pricing_service import _store_cache, _load_cached
            pp = _make_price_point(instrument_id="SOL", price=120.0)
            _store_cache(pp, "hourly")
            loaded = _load_cached("SOL", "hourly", 300)
            assert loaded is not None
            assert loaded.price == 120.0
            assert loaded.instrument_id == "SOL"

# ---------------------------------------------------------------------------
# TestPriceFromStaticFile
# ---------------------------------------------------------------------------

class TestPriceFromStaticFile:

    def test_returns_none_when_file_missing(self, tmp_path):
        with patch("services.pricing_service._PRICES_FILE", tmp_path / "nope.json"):
            from services.pricing_service import _price_from_static_file
            assert _price_from_static_file("XRP") is None

    def test_numeric_value(self, tmp_path):
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({"BTC": 51000.0}))
        with patch("services.pricing_service._PRICES_FILE", prices_file):
            from services.pricing_service import _price_from_static_file
            result = _price_from_static_file("BTC")
            assert result is not None
            assert result.price == 51000.0
            assert result.source == "pricing_file"

    def test_dict_value_with_price_key(self, tmp_path):
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({"ETH": {"price": 3200.0, "name": "Ethereum"}}))
        with patch("services.pricing_service._PRICES_FILE", prices_file):
            from services.pricing_service import _price_from_static_file
            result = _price_from_static_file("ETH")
            assert result is not None
            assert result.price == 3200.0

    def test_unknown_symbol_returns_none(self, tmp_path):
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({"BTC": 50000}))
        with patch("services.pricing_service._PRICES_FILE", prices_file):
            from services.pricing_service import _price_from_static_file
            assert _price_from_static_file("UNKNOWN") is None

    def test_string_value_returns_none(self, tmp_path):
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({"BAD": "not_a_number"}))
        with patch("services.pricing_service._PRICES_FILE", prices_file):
            from services.pricing_service import _price_from_static_file
            assert _price_from_static_file("BAD") is None

    def test_case_insensitive_lookup(self, tmp_path):
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({"BTC": 48000}))
        with patch("services.pricing_service._PRICES_FILE", prices_file):
            from services.pricing_service import _price_from_static_file
            result = _price_from_static_file("btc")
            assert result is not None
            assert result.price == 48000.0

# ---------------------------------------------------------------------------
# TestFetchPrice
# ---------------------------------------------------------------------------

class TestFetchPrice:

    @pytest.mark.asyncio
    @patch("services.pricing_service.fx_convert", create=True)
    async def test_cash_instrument_uses_fx(self, mock_fx):
        mock_fx.return_value = 1.08
        with patch.dict("sys.modules", {"services.fx_service": MagicMock(convert=mock_fx)}):
            from services.pricing_service import _fetch_price
            result = await _fetch_price("CASH:EUR")
            assert result is not None
            assert result.price == 1.08
            assert result.source == "fx_service"

    @pytest.mark.asyncio
    async def test_normal_instrument_calls_aget_price(self):
        mock_aget = AsyncMock(return_value=52000.0)
        mock_module = MagicMock()
        mock_module.aget_price_usd = mock_aget
        with patch.dict("sys.modules", {"services.pricing": mock_module}):
            from services.pricing_service import _fetch_price
            result = await _fetch_price("BTC")
            assert result is not None
            assert result.price == 52000.0
            assert result.source == "pricing_service"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_price(self):
        mock_aget = AsyncMock(return_value=None)
        mock_module = MagicMock()
        mock_module.aget_price_usd = mock_aget
        with patch.dict("sys.modules", {"services.pricing": mock_module}):
            with patch("services.pricing_service._price_from_static_file", return_value=None):
                from services.pricing_service import _fetch_price
                result = await _fetch_price("UNKNOWN_TICKER")
                assert result is None

    @pytest.mark.asyncio
    async def test_falls_back_to_static_file(self):
        mock_aget = AsyncMock(return_value=None)
        mock_module = MagicMock()
        mock_module.aget_price_usd = mock_aget
        fallback_pp = _make_price_point(instrument_id="RARE", price=99.0, source="pricing_file")
        with patch.dict("sys.modules", {"services.pricing": mock_module}):
            with patch("services.pricing_service._price_from_static_file", return_value=fallback_pp):
                from services.pricing_service import _fetch_price
                result = await _fetch_price("RARE")
                assert result is not None
                assert result.price == 99.0
                assert result.source == "pricing_file"

# ---------------------------------------------------------------------------
# TestGetPrices
# ---------------------------------------------------------------------------

class TestGetPrices:

    @pytest.mark.asyncio
    async def test_returns_cached_prices(self, tmp_path):
        cache_file = tmp_path / "BTC_daily.json"
        cache_file.write_text(json.dumps({
            "instrument_id": "BTC",
            "ts": "2026-02-09T12:00:00+00:00",
            "price": 50000.0,
            "currency": "USD",
            "source": "cached",
            "fetched_at": time.time(),
        }))
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            with patch("services.pricing_service._CRYPTO_SYMBOLS", {"BTC"}):
                from services.pricing_service import get_prices
                results = await get_prices(["BTC"])
                assert len(results) == 1
                assert results[0].price == 50000.0

    @pytest.mark.asyncio
    async def test_fetches_when_not_cached(self, tmp_path):
        fetched_pp = _make_price_point(instrument_id="ETH", price=3500.0)
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            with patch("services.pricing_service._CRYPTO_SYMBOLS", set()):
                with patch("services.pricing_service._fetch_price", new_callable=AsyncMock, return_value=fetched_pp):
                    with patch("services.pricing_service._store_cache"):
                        from services.pricing_service import get_prices
                        results = await get_prices(["ETH"])
                        assert len(results) == 1
                        assert results[0].price == 3500.0

    @pytest.mark.asyncio
    async def test_deduplicates_instruments(self, tmp_path):
        fetched_pp = _make_price_point(instrument_id="BTC", price=51000.0)
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            with patch("services.pricing_service._CRYPTO_SYMBOLS", set()):
                with patch("services.pricing_service._fetch_price", new_callable=AsyncMock, return_value=fetched_pp) as mock_fetch:
                    with patch("services.pricing_service._store_cache"):
                        from services.pricing_service import get_prices
                        results = await get_prices(["btc", "BTC", "Btc"])
                        assert len(results) == 1
                        mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_empty_strings(self, tmp_path):
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            with patch("services.pricing_service._CRYPTO_SYMBOLS", set()):
                with patch("services.pricing_service._fetch_price", new_callable=AsyncMock, return_value=None):
                    from services.pricing_service import get_prices
                    results = await get_prices(["", ""])
                    assert len(results) == 0

    @pytest.mark.asyncio
    async def test_stores_fetched_in_cache(self, tmp_path):
        fetched_pp = _make_price_point(instrument_id="SOL", price=120.0)
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            with patch("services.pricing_service._CRYPTO_SYMBOLS", set()):
                with patch("services.pricing_service._fetch_price", new_callable=AsyncMock, return_value=fetched_pp):
                    with patch("services.pricing_service._store_cache") as mock_store:
                        from services.pricing_service import get_prices
                        await get_prices(["SOL"])
                        mock_store.assert_called_once_with(fetched_pp, "daily")

    @pytest.mark.asyncio
    async def test_results_sorted_by_key(self, tmp_path):
        pp_a = _make_price_point(instrument_id="ADA", price=1.0)
        pp_z = _make_price_point(instrument_id="ZEC", price=50.0)

        async def mock_fetch(inst):
            return {"ADA": pp_a, "ZEC": pp_z}.get(inst)

        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            with patch("services.pricing_service._CRYPTO_SYMBOLS", set()):
                with patch("services.pricing_service._fetch_price", side_effect=mock_fetch):
                    with patch("services.pricing_service._store_cache"):
                        from services.pricing_service import get_prices
                        results = await get_prices(["ZEC", "ADA"])
                        assert results[0].instrument_id == "ADA"
                        assert results[1].instrument_id == "ZEC"

    @pytest.mark.asyncio
    async def test_handles_fetch_exception(self, tmp_path):
        with patch("services.pricing_service._CACHE_DIR", tmp_path):
            with patch("services.pricing_service._CRYPTO_SYMBOLS", set()):
                with patch("services.pricing_service._fetch_price", new_callable=AsyncMock, side_effect=Exception("network error")):
                    from services.pricing_service import get_prices
                    results = await get_prices(["FAIL"])
                    assert len(results) == 0
