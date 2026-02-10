"""Tests for api/utils/data_processing.py — Pure utility functions"""

import pytest
from api.utils.data_processing import (
    normalize_location,
    classify_location,
    pick_primary_location_for_symbol,
    to_rows,
    calculate_price_deviation,
    parse_min_usd,
    get_data_age_minutes,
)


# ---------------------------------------------------------------------------
# normalize_location
# ---------------------------------------------------------------------------
class TestNormalizeLocation:
    def test_known_exchange_binance(self):
        assert normalize_location("Binance") == "binance"

    def test_known_exchange_variant(self):
        assert normalize_location("binance.com") == "binance"
        assert normalize_location("Binance Spot") == "binance"

    def test_known_exchange_coinbase_pro(self):
        assert normalize_location("coinbase pro") == "coinbase"

    def test_known_exchange_kraken(self):
        assert normalize_location("Kraken.com") == "kraken"

    def test_hardware_wallets(self):
        assert normalize_location("Ledger") == "ledger"
        assert normalize_location("Trezor") == "trezor"

    def test_metamask(self):
        assert normalize_location("MetaMask") == "metamask"

    def test_unknown_passthrough(self):
        assert normalize_location("MyCustomWallet") == "mycustomwallet"

    def test_empty_string(self):
        assert normalize_location("") == ""

    def test_none_returns_empty(self):
        assert normalize_location(None) == ""

    def test_extra_spaces(self):
        assert normalize_location("  binance  spot  ") == "binance"

    def test_case_insensitive(self):
        assert normalize_location("KRAKEN") == "kraken"


# ---------------------------------------------------------------------------
# classify_location
# ---------------------------------------------------------------------------
class TestClassifyLocation:
    def test_major_exchange_priority_1(self):
        assert classify_location("binance") == 1
        assert classify_location("Kraken") == 1
        assert classify_location("coinbase") == 1

    def test_wallet_priority_2(self):
        assert classify_location("metamask") == 2
        assert classify_location("My Wallet") == 2

    def test_hardware_priority_3(self):
        assert classify_location("ledger") == 3
        assert classify_location("trezor") == 3
        # Note: "hardware wallet" matches "wallet" first → priority 2
        assert classify_location("hardware wallet") == 2
        assert classify_location("hardware") == 3

    def test_unknown_priority_4(self):
        assert classify_location("unknown") == 4
        assert classify_location("SomeExchange") == 4

    def test_empty_string(self):
        assert classify_location("") == 4

    def test_none_returns_4(self):
        assert classify_location(None) == 4


# ---------------------------------------------------------------------------
# pick_primary_location_for_symbol
# ---------------------------------------------------------------------------
class TestPickPrimaryLocation:
    def test_single_location(self):
        holdings = {"BTC": {"binance": 1.5}}
        assert pick_primary_location_for_symbol("BTC", holdings) == "binance"

    def test_exchange_over_wallet(self):
        holdings = {"ETH": {"metamask": 5.0, "kraken": 2.0}}
        # kraken=priority 1, metamask=priority 2 → kraken wins
        assert pick_primary_location_for_symbol("ETH", holdings) == "kraken"

    def test_higher_balance_tiebreaker(self):
        holdings = {"SOL": {"binance": 10.0, "coinbase": 50.0}}
        # Both priority 1, coinbase has higher balance → coinbase wins
        assert pick_primary_location_for_symbol("SOL", holdings) == "coinbase"

    def test_symbol_not_found(self):
        assert pick_primary_location_for_symbol("XRP", {}) == "unknown"

    def test_empty_locations(self):
        holdings = {"BTC": {}}
        assert pick_primary_location_for_symbol("BTC", holdings) == "unknown"


# ---------------------------------------------------------------------------
# to_rows
# ---------------------------------------------------------------------------
class TestToRows:
    def test_basic_conversion(self):
        raw = [{"symbol": "btc", "balance": 1.0, "value_usd": 50000, "location": "Binance", "price_usd": 50000}]
        rows = to_rows(raw)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "BTC"  # uppercased
        assert rows[0]["location"] == "binance"  # normalized
        assert rows[0]["balance"] == 1.0
        assert rows[0]["value_usd"] == 50000.0

    def test_filters_zero_balance(self):
        raw = [{"symbol": "ETH", "balance": 0, "value_usd": 100, "price_usd": 100}]
        assert to_rows(raw) == []

    def test_filters_zero_value(self):
        raw = [{"symbol": "ETH", "balance": 1.0, "value_usd": 0, "price_usd": 0}]
        assert to_rows(raw) == []

    def test_skips_non_dict(self):
        raw = ["not a dict", 42, None]
        assert to_rows(raw) == []

    def test_defaults_for_missing_fields(self):
        raw = [{"symbol": "SOL", "balance": 10, "value_usd": 500}]
        rows = to_rows(raw)
        assert len(rows) == 1
        assert rows[0]["price_usd"] == 0.0
        assert rows[0]["location"] == "unknown"

    def test_empty_input(self):
        assert to_rows([]) == []

    def test_multiple_items(self):
        raw = [
            {"symbol": "BTC", "balance": 1.0, "value_usd": 50000, "price_usd": 50000},
            {"symbol": "ETH", "balance": 10, "value_usd": 30000, "price_usd": 3000},
        ]
        rows = to_rows(raw)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# calculate_price_deviation
# ---------------------------------------------------------------------------
class TestCalculatePriceDeviation:
    def test_no_deviation(self):
        assert calculate_price_deviation(100.0, 100.0) == 0.0

    def test_positive_deviation(self):
        result = calculate_price_deviation(110.0, 100.0)
        assert abs(result - 10.0) < 0.01

    def test_negative_deviation(self):
        result = calculate_price_deviation(90.0, 100.0)
        assert abs(result - 10.0) < 0.01  # abs deviation

    def test_zero_market_price(self):
        assert calculate_price_deviation(100.0, 0.0) == 0.0

    def test_large_deviation(self):
        result = calculate_price_deviation(200.0, 100.0)
        assert abs(result - 100.0) < 0.01


# ---------------------------------------------------------------------------
# parse_min_usd
# ---------------------------------------------------------------------------
class TestParseMinUsd:
    def test_valid_string(self):
        assert parse_min_usd("5.0") == 5.0

    def test_none_returns_default(self):
        assert parse_min_usd(None) == 1.0

    def test_empty_returns_default(self):
        assert parse_min_usd("") == 1.0

    def test_negative_clamped_to_zero(self):
        assert parse_min_usd("-5") == 0.0

    def test_invalid_string_returns_default(self):
        assert parse_min_usd("not_a_number") == 1.0

    def test_custom_default(self):
        assert parse_min_usd(None, default=10.0) == 10.0

    def test_zero_is_valid(self):
        assert parse_min_usd("0") == 0.0


# ---------------------------------------------------------------------------
# get_data_age_minutes
# ---------------------------------------------------------------------------
class TestGetDataAgeMinutes:
    def test_cointracking(self):
        assert get_data_age_minutes("cointracking") == 5.0

    def test_binance(self):
        assert get_data_age_minutes("binance") == 1.0

    def test_kraken(self):
        assert get_data_age_minutes("kraken") == 2.0

    def test_local(self):
        assert get_data_age_minutes("local") == 10.0

    def test_fallback(self):
        assert get_data_age_minutes("fallback") == 15.0

    def test_unknown_source(self):
        assert get_data_age_minutes("unknown_source") == 10.0

    def test_case_insensitive(self):
        assert get_data_age_minutes("BINANCE") == 1.0
        assert get_data_age_minutes("CoinTracking") == 5.0
