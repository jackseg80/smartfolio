"""
Test regime naming consistency across the codebase.

Verifies that all regime-related code uses the canonical names:
  Bear Market (0), Correction (1), Bull Market (2), Expansion (3)
"""

import pytest
from services.regime_constants import (
    MarketRegime,
    REGIME_NAMES,
    REGIME_IDS,
    REGIME_SCORE_RANGES,
    REGIME_COLORS,
    LEGACY_TO_CANONICAL,
    score_to_regime,
    regime_name,
    normalize_regime_name,
    regime_to_key,
)


class TestRegimeConstants:
    """Verify the regime constants module is self-consistent."""

    def test_four_canonical_regimes(self):
        assert len(MarketRegime) == 4

    def test_regime_ids_match_enum(self):
        for regime in MarketRegime:
            assert REGIME_IDS[regime.name.replace('_', ' ').title()] == regime.value

    def test_regime_names_ordered(self):
        assert REGIME_NAMES == ['Bear Market', 'Correction', 'Bull Market', 'Expansion']

    def test_score_ranges_cover_0_to_100(self):
        ranges = REGIME_SCORE_RANGES
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 100
        for i in range(len(ranges) - 1):
            assert ranges[i][1] + 1 == ranges[i + 1][0]

    def test_score_to_regime_boundaries(self):
        assert score_to_regime(0) == MarketRegime.BEAR_MARKET
        assert score_to_regime(25) == MarketRegime.BEAR_MARKET
        assert score_to_regime(26) == MarketRegime.CORRECTION
        assert score_to_regime(50) == MarketRegime.CORRECTION
        assert score_to_regime(51) == MarketRegime.BULL_MARKET
        assert score_to_regime(75) == MarketRegime.BULL_MARKET
        assert score_to_regime(76) == MarketRegime.EXPANSION
        assert score_to_regime(100) == MarketRegime.EXPANSION

    def test_score_to_regime_edge_cases(self):
        assert score_to_regime(-10) == MarketRegime.BEAR_MARKET
        assert score_to_regime(200) == MarketRegime.EXPANSION

    def test_regime_name_from_id(self):
        assert regime_name(0) == 'Bear Market'
        assert regime_name(1) == 'Correction'
        assert regime_name(2) == 'Bull Market'
        assert regime_name(3) == 'Expansion'


class TestNormalizeRegimeName:
    """Verify normalize_regime_name handles all legacy conventions."""

    @pytest.mark.parametrize("legacy,expected", [
        # Convention A (canonical)
        ('Bear Market', 'Bear Market'),
        ('Correction', 'Correction'),
        ('Bull Market', 'Bull Market'),
        ('Expansion', 'Expansion'),
        # Convention B (old HMM)
        ('Bull', 'Bull Market'),
        ('Bear', 'Bear Market'),
        ('Sideways', 'Correction'),
        ('Distribution', 'Expansion'),
        # Convention C (old frontend)
        ('Accumulation', 'Bear Market'),
        ('Euphoria', 'Bull Market'),
        # Snake case
        ('bear_market', 'Bear Market'),
        ('bull_market', 'Bull Market'),
        # Lowercase variants in LEGACY_TO_CANONICAL
        ('expansion', 'Expansion'),
        ('correction', 'Correction'),
        # Consolidation legacy
        ('Consolidation', 'Correction'),
        ('consolidation', 'Correction'),
    ])
    def test_legacy_normalization(self, legacy, expected):
        assert normalize_regime_name(legacy) == expected

    def test_unknown_returns_input(self):
        """Unknown names return the input unchanged."""
        assert normalize_regime_name('UnknownRegime') == 'UnknownRegime'

    def test_unknown_lowercase_returns_correction(self):
        """'unknown' lowercase is in LEGACY_TO_CANONICAL."""
        assert normalize_regime_name('unknown') == 'Correction'


class TestRegimeToKey:
    """Verify regime_to_key produces valid JS/CSS keys."""

    def test_canonical_keys(self):
        assert regime_to_key('Bear Market') == 'bear_market'
        assert regime_to_key('Correction') == 'correction'
        assert regime_to_key('Bull Market') == 'bull_market'
        assert regime_to_key('Expansion') == 'expansion'


class TestRegimeColors:
    """Verify all regimes have assigned colors."""

    def test_all_regimes_have_colors(self):
        for regime in MarketRegime:
            assert regime.value in REGIME_COLORS, f"Missing color for {regime.name}"

    def test_colors_are_hex(self):
        for regime_id, color in REGIME_COLORS.items():
            assert color.startswith('#'), f"Color for regime {regime_id} should be hex: {color}"
