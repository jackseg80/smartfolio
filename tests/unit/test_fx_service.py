"""Tests unitaires pour services/fx_service.py."""
import pytest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime

import services.fx_service as fx


class TestConvert:
    def test_same_currency_returns_amount(self):
        result = fx.convert(100.0, "USD", "USD")
        assert result == 100.0

    def test_same_currency_case_insensitive(self):
        result = fx.convert(50.0, "usd", "USD")
        assert result == 50.0

    def test_none_currency_raises(self):
        with pytest.raises(ValueError):
            fx.convert(100.0, None, "USD")
        with pytest.raises(ValueError):
            fx.convert(100.0, "USD", None)

    def test_eur_to_usd_uses_rate(self):
        # Force fresh cache to skip API fetch
        with patch.object(fx, '_ensure_rates_fresh'):
            # EUR->USD rate is ~1.087 in fallback
            result = fx.convert(100.0, "EUR", "USD")
            assert result > 100.0  # EUR is worth more than USD

    def test_usd_to_eur(self):
        with patch.object(fx, '_ensure_rates_fresh'):
            result = fx.convert(100.0, "USD", "EUR")
            assert result < 100.0  # Takes fewer EUR

    def test_chf_to_gbp_cross_rate(self):
        with patch.object(fx, '_ensure_rates_fresh'):
            result = fx.convert(100.0, "CHF", "GBP")
            assert result > 0

    def test_unknown_currency_defaults_to_parity(self):
        with patch.object(fx, '_ensure_rates_fresh'):
            result = fx.convert(100.0, "XYZ", "USD")
            assert result == 100.0


class TestResolveRate:
    def test_usd_rate_is_one(self):
        with patch.object(fx, '_ensure_rates_fresh'):
            rate = fx._resolve_rate("USD")
            assert rate == 1.0

    def test_known_currency_has_rate(self):
        with patch.object(fx, '_ensure_rates_fresh'):
            rate = fx._resolve_rate("EUR")
            assert rate > 0 and rate != 1.0

    def test_unknown_currency_returns_parity(self):
        with patch.object(fx, '_ensure_rates_fresh'):
            rate = fx._resolve_rate("UNKNOWN_CURRENCY")
            assert rate == 1.0


class TestGetRates:
    def test_usd_base_returns_all_currencies(self):
        with patch.object(fx, '_ensure_rates_fresh'):
            rates = fx.get_rates("USD")
            assert "USD" in rates
            assert rates["USD"] == 1.0
            assert "EUR" in rates
            assert "CHF" in rates

    def test_eur_base(self):
        with patch.object(fx, '_ensure_rates_fresh'):
            rates = fx.get_rates("EUR")
            assert rates["EUR"] == 1.0
            assert "USD" in rates


class TestGetSupportedCurrencies:
    def test_returns_list(self):
        with patch.object(fx, '_ensure_rates_fresh'):
            currencies = fx.get_supported_currencies()
            assert isinstance(currencies, list)
            assert "USD" in currencies
            assert "EUR" in currencies


class TestGetCacheInfo:
    def test_returns_dict(self):
        info = fx.get_cache_info()
        assert "cached_currencies" in info
        assert "cache_ttl_seconds" in info
        assert info["cache_ttl_seconds"] == 14400

    def test_cache_age_none_when_never_fetched(self):
        original = fx._RATES_CACHE_TIMESTAMP
        fx._RATES_CACHE_TIMESTAMP = 0
        try:
            info = fx.get_cache_info()
            assert info["cache_age_seconds"] is None
            assert info["cache_fresh"] is False
        finally:
            fx._RATES_CACHE_TIMESTAMP = original


class TestEnsureRatesFresh:
    def test_does_not_fetch_when_fresh(self):
        fx._RATES_CACHE_TIMESTAMP = time.time()  # Just now
        with patch.object(fx, '_fetch_live_rates') as mock_fetch:
            fx._ensure_rates_fresh()
            mock_fetch.assert_not_called()

    def test_fetches_when_expired(self):
        fx._RATES_CACHE_TIMESTAMP = 0  # Long ago
        with patch.object(fx, '_fetch_live_rates') as mock_fetch:
            fx._ensure_rates_fresh()
            mock_fetch.assert_called_once()


class TestFetchLiveRates:
    def test_handles_api_failure(self):
        # httpx is imported inside the function, mock via builtins import
        import httpx as real_httpx
        with patch.object(real_httpx, "Client") as mock_client_cls:
            mock_client_cls.side_effect = Exception("Network error")
            result = fx._fetch_live_rates()
            assert result is False

    def test_handles_non_200(self):
        import httpx as real_httpx
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        with patch.object(real_httpx, "Client", return_value=mock_client):
            result = fx._fetch_live_rates()
            assert result is False


class TestFallbackRates:
    def test_fallback_rates_present(self):
        assert "USD" in fx._FALLBACK_RATES_TO_USD
        assert "EUR" in fx._FALLBACK_RATES_TO_USD
        assert "CHF" in fx._FALLBACK_RATES_TO_USD
        assert "GBP" in fx._FALLBACK_RATES_TO_USD
        assert fx._FALLBACK_RATES_TO_USD["USD"] == 1.0
