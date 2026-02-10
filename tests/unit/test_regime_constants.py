"""Tests for services/regime_constants.py — Unified Market Regime Constants"""

import pytest
from services.regime_constants import (
    MarketRegime,
    REGIME_NAMES,
    REGIME_IDS,
    REGIME_SCORE_RANGES,
    REGIME_COLORS,
    REGIME_DESCRIPTIONS,
    LEGACY_TO_CANONICAL,
    score_to_regime,
    regime_name,
    normalize_regime_name,
    smooth_regime_sequence,
    regime_to_key,
)


# ---------------------------------------------------------------------------
# TestMarketRegimeEnum
# ---------------------------------------------------------------------------
class TestMarketRegimeEnum:
    def test_values(self):
        assert MarketRegime.BEAR_MARKET == 0
        assert MarketRegime.CORRECTION == 1
        assert MarketRegime.BULL_MARKET == 2
        assert MarketRegime.EXPANSION == 3

    def test_is_int_enum(self):
        assert isinstance(MarketRegime.BEAR_MARKET, int)


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------
class TestConstants:
    def test_regime_names_length(self):
        assert len(REGIME_NAMES) == 4

    def test_regime_names_order(self):
        assert REGIME_NAMES[0] == "Bear Market"
        assert REGIME_NAMES[3] == "Expansion"

    def test_regime_ids_mapping(self):
        assert REGIME_IDS["Bear Market"] == 0
        assert REGIME_IDS["Correction"] == 1
        assert REGIME_IDS["Bull Market"] == 2
        assert REGIME_IDS["Expansion"] == 3

    def test_regime_colors_all_present(self):
        for i in range(4):
            assert i in REGIME_COLORS
            assert REGIME_COLORS[i].startswith("#")

    def test_regime_descriptions_all_present(self):
        for i in range(4):
            assert i in REGIME_DESCRIPTIONS
            desc = REGIME_DESCRIPTIONS[i]
            assert "name" in desc
            assert "description" in desc
            assert "strategy" in desc
            assert "risk_level" in desc

    def test_score_ranges_cover_0_to_100(self):
        # Verify ranges: (0,25,0), (26,50,1), (51,75,2), (76,100,3)
        assert REGIME_SCORE_RANGES[0] == (0, 25, 0)
        assert REGIME_SCORE_RANGES[1] == (26, 50, 1)
        assert REGIME_SCORE_RANGES[2] == (51, 75, 2)
        assert REGIME_SCORE_RANGES[3] == (76, 100, 3)


# ---------------------------------------------------------------------------
# TestScoreToRegime
# ---------------------------------------------------------------------------
class TestScoreToRegime:
    def test_bear_market_range(self):
        assert score_to_regime(0) == MarketRegime.BEAR_MARKET
        assert score_to_regime(10) == MarketRegime.BEAR_MARKET
        assert score_to_regime(25) == MarketRegime.BEAR_MARKET

    def test_correction_range(self):
        assert score_to_regime(26) == MarketRegime.CORRECTION
        assert score_to_regime(40) == MarketRegime.CORRECTION
        assert score_to_regime(50) == MarketRegime.CORRECTION

    def test_bull_market_range(self):
        assert score_to_regime(51) == MarketRegime.BULL_MARKET
        assert score_to_regime(60) == MarketRegime.BULL_MARKET
        assert score_to_regime(75) == MarketRegime.BULL_MARKET

    def test_expansion_range(self):
        assert score_to_regime(76) == MarketRegime.EXPANSION
        assert score_to_regime(90) == MarketRegime.EXPANSION
        assert score_to_regime(100) == MarketRegime.EXPANSION

    def test_boundary_values(self):
        assert score_to_regime(25) == MarketRegime.BEAR_MARKET
        assert score_to_regime(25.1) == MarketRegime.CORRECTION
        assert score_to_regime(50) == MarketRegime.CORRECTION
        assert score_to_regime(50.1) == MarketRegime.BULL_MARKET
        assert score_to_regime(75) == MarketRegime.BULL_MARKET
        assert score_to_regime(75.1) == MarketRegime.EXPANSION

    def test_negative_score(self):
        assert score_to_regime(-10) == MarketRegime.BEAR_MARKET

    def test_above_100(self):
        assert score_to_regime(150) == MarketRegime.EXPANSION


# ---------------------------------------------------------------------------
# TestRegimeName
# ---------------------------------------------------------------------------
class TestRegimeName:
    def test_valid_ids(self):
        assert regime_name(0) == "Bear Market"
        assert regime_name(1) == "Correction"
        assert regime_name(2) == "Bull Market"
        assert regime_name(3) == "Expansion"

    def test_clamps_negative(self):
        assert regime_name(-1) == "Bear Market"
        assert regime_name(-100) == "Bear Market"

    def test_clamps_above_3(self):
        assert regime_name(4) == "Expansion"
        assert regime_name(100) == "Expansion"


# ---------------------------------------------------------------------------
# TestNormalizeRegimeName
# ---------------------------------------------------------------------------
class TestNormalizeRegimeName:
    def test_canonical_names_unchanged(self):
        assert normalize_regime_name("Bear Market") == "Bear Market"
        assert normalize_regime_name("Bull Market") == "Bull Market"
        assert normalize_regime_name("Correction") == "Correction"
        assert normalize_regime_name("Expansion") == "Expansion"

    def test_legacy_crypto_cycle_names(self):
        assert normalize_regime_name("Accumulation") == "Bear Market"
        assert normalize_regime_name("Euphoria") == "Bull Market"
        assert normalize_regime_name("Distribution") == "Expansion"
        assert normalize_regime_name("Contraction") == "Bear Market"

    def test_short_names(self):
        assert normalize_regime_name("Bull") == "Bull Market"
        assert normalize_regime_name("Bear") == "Bear Market"
        assert normalize_regime_name("Sideways") == "Correction"

    def test_snake_case_variants(self):
        assert normalize_regime_name("bull_market") == "Bull Market"
        assert normalize_regime_name("bear_market") == "Bear Market"

    def test_lowercase_variants(self):
        assert normalize_regime_name("accumulation") == "Bear Market"
        assert normalize_regime_name("euphoria") == "Bull Market"

    def test_french_variants(self):
        assert normalize_regime_name("Euphorie") == "Bull Market"
        assert normalize_regime_name("euphorie") == "Bull Market"

    def test_unknown_fallbacks(self):
        assert normalize_regime_name("neutral") == "Correction"
        assert normalize_regime_name("unknown") == "Correction"
        assert normalize_regime_name("Unknown") == "Correction"
        assert normalize_regime_name("Consolidation") == "Correction"

    def test_unrecognized_returns_as_is(self):
        assert normalize_regime_name("SomeNewRegime") == "SomeNewRegime"

    def test_early_expansion_legacy(self):
        assert normalize_regime_name("early_expansion") == "Correction"
        assert normalize_regime_name("Early Expansion") == "Correction"


# ---------------------------------------------------------------------------
# TestSmoothRegimeSequence
# ---------------------------------------------------------------------------
class TestSmoothRegimeSequence:
    def test_short_sequence_unchanged(self):
        seq = [0, 0, 1, 1, 2]
        assert smooth_regime_sequence(seq, min_duration=7) == seq

    def test_single_regime_unchanged(self):
        seq = [2] * 30
        assert smooth_regime_sequence(seq) == seq

    def test_removes_short_blip(self):
        # 15 days bull, 3 days correction, 15 days bull → blip removed
        seq = [2] * 15 + [1] * 3 + [2] * 15
        smoothed = smooth_regime_sequence(seq, min_duration=7)
        assert all(r == 2 for r in smoothed)

    def test_preserves_long_segment(self):
        # 15 days bull, 10 days correction, 15 days bull → correction preserved
        seq = [2] * 15 + [1] * 10 + [2] * 15
        smoothed = smooth_regime_sequence(seq, min_duration=7)
        assert 1 in smoothed  # correction still present

    def test_first_segment_short(self):
        # 3 days bear, then 20 days bull → bear replaced
        seq = [0] * 3 + [2] * 20
        smoothed = smooth_regime_sequence(seq, min_duration=7)
        assert smoothed[0] == 2  # replaced by next segment

    def test_last_segment_short(self):
        # 20 days bull, then 3 days bear → bear replaced
        seq = [2] * 20 + [0] * 3
        smoothed = smooth_regime_sequence(seq, min_duration=7)
        assert smoothed[-1] == 2  # replaced by previous segment

    def test_multiple_short_blips(self):
        # bull → correction blip → bull → bear blip → bull
        seq = [2] * 15 + [1] * 2 + [2] * 15 + [0] * 2 + [2] * 15
        smoothed = smooth_regime_sequence(seq, min_duration=7)
        assert all(r == 2 for r in smoothed)

    def test_empty_sequence(self):
        assert smooth_regime_sequence([]) == []

    def test_min_duration_1_no_smoothing(self):
        seq = [0, 1, 2, 3, 0, 1, 2]
        smoothed = smooth_regime_sequence(seq, min_duration=1)
        assert smoothed == seq

    def test_picks_longer_neighbor(self):
        # 20 days bull, 3 days correction, 10 days bear → correction → bull (longer)
        seq = [2] * 20 + [1] * 3 + [0] * 10
        smoothed = smooth_regime_sequence(seq, min_duration=7)
        # Middle segment should be replaced by bull (longer neighbor)
        for i in range(20, 23):
            assert smoothed[i] == 2


# ---------------------------------------------------------------------------
# TestRegimeToKey
# ---------------------------------------------------------------------------
class TestRegimeToKey:
    def test_canonical_names(self):
        assert regime_to_key("Bear Market") == "bear_market"
        assert regime_to_key("Correction") == "correction"
        assert regime_to_key("Bull Market") == "bull_market"
        assert regime_to_key("Expansion") == "expansion"

    def test_legacy_names_normalized_first(self):
        assert regime_to_key("Accumulation") == "bear_market"
        assert regime_to_key("Euphoria") == "bull_market"

    def test_snake_case_input(self):
        assert regime_to_key("bull_market") == "bull_market"

    def test_unknown_passthrough(self):
        assert regime_to_key("SomeNewRegime") == "somenewregime"


# ---------------------------------------------------------------------------
# TestLegacyToCanonical
# ---------------------------------------------------------------------------
class TestLegacyToCanonical:
    def test_all_values_are_canonical(self):
        canonical = {"Bear Market", "Correction", "Bull Market", "Expansion"}
        for key, value in LEGACY_TO_CANONICAL.items():
            assert value in canonical, f"'{key}' maps to non-canonical '{value}'"

    def test_covers_common_variants(self):
        # Verify key variants exist
        for key in ["Bull", "Bear", "Sideways", "Accumulation", "Euphoria",
                     "bull_market", "bear_market", "neutral", "unknown"]:
            assert key in LEGACY_TO_CANONICAL, f"Missing legacy key: {key}"
