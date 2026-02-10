"""Tests for services/taxonomy.py — auto-classification, Taxonomy dataclass, helpers."""

import pytest
from unittest.mock import patch, MagicMock

from services.taxonomy import (
    DEFAULT_GROUPS_ORDER,
    DEFAULT_ALIASES,
    AUTO_CLASSIFICATION_RULES,
    auto_classify_symbol,
    get_classification_suggestions,
    _keynorm,
    _canonical_group,
    _canonicalize_alias_mapping,
    Taxonomy,
)


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants:
    def test_default_groups_order_has_11_groups(self):
        assert len(DEFAULT_GROUPS_ORDER) == 11

    def test_groups_order_starts_with_btc(self):
        assert DEFAULT_GROUPS_ORDER[0] == "BTC"

    def test_groups_order_ends_with_others(self):
        assert DEFAULT_GROUPS_ORDER[-1] == "Others"

    def test_default_aliases_contains_btc(self):
        assert DEFAULT_ALIASES["BTC"] == "BTC"

    def test_default_aliases_contains_stablecoins(self):
        assert DEFAULT_ALIASES["USDT"] == "Stablecoins"
        assert DEFAULT_ALIASES["USDC"] == "Stablecoins"

    def test_auto_classification_rules_keys(self):
        assert "stablecoins_patterns" in AUTO_CLASSIFICATION_RULES
        assert "meme_patterns" in AUTO_CLASSIFICATION_RULES
        assert "ai_patterns" in AUTO_CLASSIFICATION_RULES


# ===========================================================================
# auto_classify_symbol
# ===========================================================================

class TestAutoClassifySymbol:
    def test_empty_symbol(self):
        assert auto_classify_symbol("") == "Others"

    def test_stablecoin_usdc(self):
        assert auto_classify_symbol("USDC") == "Stablecoins"

    def test_stablecoin_usdt(self):
        assert auto_classify_symbol("USDT") == "Stablecoins"

    def test_stablecoin_dai(self):
        assert auto_classify_symbol("DAI") == "Stablecoins"

    def test_memecoin_doge(self):
        assert auto_classify_symbol("DOGE") == "Memecoins"

    def test_memecoin_shib(self):
        assert auto_classify_symbol("SHIB") == "Memecoins"

    def test_ai_render(self):
        assert auto_classify_symbol("RENDER") == "AI/Data"

    def test_ai_fetch(self):
        assert auto_classify_symbol("FET") == "AI/Data"

    def test_gaming_axs(self):
        assert auto_classify_symbol("AXS") == "Gaming/NFT"

    def test_l2_arb(self):
        assert auto_classify_symbol("ARB") == "L2/Scaling"

    def test_unknown_symbol(self):
        assert auto_classify_symbol("XYZUNKNOWN") == "Others"

    def test_case_insensitive(self):
        assert auto_classify_symbol("usdc") == "Stablecoins"
        assert auto_classify_symbol("Doge") == "Memecoins"


# ===========================================================================
# get_classification_suggestions
# ===========================================================================

class TestGetClassificationSuggestions:
    def test_known_symbols(self):
        result = get_classification_suggestions(["USDC", "DOGE", "RENDER"])
        assert result["USDC"] == "Stablecoins"
        assert result["DOGE"] == "Memecoins"
        assert result["RENDER"] == "AI/Data"

    def test_unknown_symbols_excluded(self):
        result = get_classification_suggestions(["XYZUNKNOWN", "RANDOM"])
        assert len(result) == 0

    def test_mixed(self):
        result = get_classification_suggestions(["USDC", "XYZUNKNOWN"])
        assert "USDC" in result
        assert "XYZUNKNOWN" not in result

    def test_empty_list(self):
        result = get_classification_suggestions([])
        assert result == {}


# ===========================================================================
# Helper functions
# ===========================================================================

class TestKeynorm:
    def test_basic(self):
        assert _keynorm("BTC") == "BTC"

    def test_lowercase(self):
        assert _keynorm("btc") == "BTC"

    def test_spaces_removed(self):
        assert _keynorm("L1/L0 majors") == "L1/L0MAJORS"

    def test_empty(self):
        assert _keynorm("") == ""


class TestCanonicalGroup:
    def test_exact_match(self):
        assert _canonical_group("BTC", DEFAULT_GROUPS_ORDER) == "BTC"

    def test_case_insensitive(self):
        assert _canonical_group("btc", DEFAULT_GROUPS_ORDER) == "BTC"

    def test_spaces_insensitive(self):
        assert _canonical_group("L1/L0majors", DEFAULT_GROUPS_ORDER) == "L1/L0 majors"

    def test_unknown_returns_as_is(self):
        assert _canonical_group("CustomGroup", DEFAULT_GROUPS_ORDER) == "CustomGroup"

    def test_empty(self):
        assert _canonical_group("", DEFAULT_GROUPS_ORDER) == ""


class TestCanonicalizeAliasMapping:
    def test_uppercase_keys(self):
        result = _canonicalize_alias_mapping({"btc": "BTC"}, DEFAULT_GROUPS_ORDER)
        assert "BTC" in result

    def test_canonical_group_values(self):
        result = _canonicalize_alias_mapping({"XYZ": "btc"}, DEFAULT_GROUPS_ORDER)
        assert result["XYZ"] == "BTC"

    def test_none_aliases(self):
        result = _canonicalize_alias_mapping(None, DEFAULT_GROUPS_ORDER)
        assert result == {}


# ===========================================================================
# Taxonomy dataclass
# ===========================================================================

class TestTaxonomy:
    def test_default_init(self):
        t = Taxonomy()
        assert t.groups_order == DEFAULT_GROUPS_ORDER
        assert t.aliases == DEFAULT_ALIASES

    def test_group_for_alias_known(self):
        t = Taxonomy()
        assert t.group_for_alias("BTC") == "BTC"
        assert t.group_for_alias("USDT") == "Stablecoins"
        assert t.group_for_alias("UNI") == "DeFi"

    def test_group_for_alias_unknown(self):
        t = Taxonomy()
        assert t.group_for_alias("XYZUNKNOWN") == "Others"

    def test_group_for_alias_empty(self):
        t = Taxonomy()
        assert t.group_for_alias("") == "Others"

    def test_group_for_alias_group_name(self):
        """Si l'alias est un nom de groupe, retourne le groupe canonique."""
        t = Taxonomy()
        assert t.group_for_alias("Stablecoins") == "Stablecoins"
        assert t.group_for_alias("DeFi") == "DeFi"

    def test_group_for_alias_case_insensitive(self):
        t = Taxonomy()
        assert t.group_for_alias("btc") == "BTC"
        assert t.group_for_alias("eth") == "ETH"

    def test_to_dict(self):
        t = Taxonomy()
        d = t.to_dict()
        assert "groups_order" in d
        assert "aliases" in d
        assert isinstance(d["groups_order"], list)
        assert isinstance(d["aliases"], dict)

    def test_to_dict_is_copy(self):
        t = Taxonomy()
        d = t.to_dict()
        d["groups_order"].append("NewGroup")
        assert "NewGroup" not in t.groups_order

    def test_load_no_file(self):
        """Sans fichier, load retourne les defauts."""
        Taxonomy._instance = None  # reset singleton
        with patch("services.taxonomy.os.path.exists", return_value=False):
            t = Taxonomy.load(reload=True)
        assert t.groups_order == DEFAULT_GROUPS_ORDER
        Taxonomy._instance = None

    def test_load_cached(self):
        """Second appel sans reload retourne le cache."""
        Taxonomy._instance = None
        with patch("services.taxonomy.os.path.exists", return_value=False):
            t1 = Taxonomy.load(reload=True)
            t2 = Taxonomy.load(reload=False)
        assert t1 is t2
        Taxonomy._instance = None
