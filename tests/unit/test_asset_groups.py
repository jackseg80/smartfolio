"""Tests unitaires pour shared/asset_groups.py."""
from shared.asset_groups import (
    ASSET_GROUPS,
    get_asset_group,
    get_group_symbols,
    get_all_groups,
    PORTFOLIO_COLORS,
    EXCHANGE_PRIORITY,
)


class TestGetAssetGroup:
    def test_btc_in_btc_group(self):
        assert get_asset_group("BTC") == "BTC"

    def test_eth_in_eth_group(self):
        assert get_asset_group("ETH") == "ETH"

    def test_usdt_in_stablecoins(self):
        assert get_asset_group("USDT") == "Stablecoins"

    def test_case_insensitive(self):
        assert get_asset_group("btc") == "BTC"
        assert get_asset_group("eth") == "ETH"

    def test_unknown_returns_others(self):
        assert get_asset_group("UNKNOWN_COIN_XYZ") == "Others"

    def test_sol_in_sol_group(self):
        assert get_asset_group("SOL") == "SOL"

    def test_defi_tokens(self):
        assert get_asset_group("UNI") == "DeFi"
        assert get_asset_group("AAVE") == "DeFi"


class TestGetGroupSymbols:
    def test_btc_group_contains_btc(self):
        symbols = get_group_symbols("BTC")
        assert "BTC" in symbols
        assert "WBTC" in symbols

    def test_unknown_group_returns_empty(self):
        assert get_group_symbols("NonexistentGroup") == []


class TestGetAllGroups:
    def test_returns_list(self):
        groups = get_all_groups()
        assert isinstance(groups, list)
        assert "BTC" in groups
        assert "ETH" in groups
        assert "Stablecoins" in groups


class TestConstants:
    def test_portfolio_colors_is_list(self):
        assert isinstance(PORTFOLIO_COLORS, list)
        assert len(PORTFOLIO_COLORS) > 0

    def test_exchange_priority_has_binance(self):
        assert "binance" in EXCHANGE_PRIORITY
        assert EXCHANGE_PRIORITY["binance"] == 1
