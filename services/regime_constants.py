"""
Unified Market Regime Constants - Single Source of Truth.

All regime-related code MUST import from here.
Scores: 0-100 where 100 = best market conditions.
"""

from enum import IntEnum
from typing import Dict, List, Tuple


class MarketRegime(IntEnum):
    BEAR_MARKET = 0
    CORRECTION = 1
    BULL_MARKET = 2
    EXPANSION = 3


REGIME_NAMES: List[str] = ['Bear Market', 'Correction', 'Bull Market', 'Expansion']

REGIME_IDS: Dict[str, int] = {
    'Bear Market': 0,
    'Correction': 1,
    'Bull Market': 2,
    'Expansion': 3,
}

# Score-to-regime mapping (0-100, 100 = best)
REGIME_SCORE_RANGES: List[Tuple[int, int, int]] = [
    (0, 25, 0),     # Bear Market
    (26, 50, 1),    # Correction
    (51, 75, 2),    # Bull Market
    (76, 100, 3),   # Expansion
]

REGIME_COLORS: Dict[int, str] = {
    0: '#dc2626',  # Bear Market - red
    1: '#ea580c',  # Correction - orange
    2: '#22c55e',  # Bull Market - green
    3: '#3b82f6',  # Expansion - blue
}

REGIME_DESCRIPTIONS: Dict[int, Dict[str, str]] = {
    0: {
        'name': 'Bear Market',
        'description': 'Severe market downturn with high risk',
        'strategy': 'Defensive positioning, increase cash/bonds, hedge risk',
        'risk_level': 'High',
    },
    1: {
        'name': 'Correction',
        'description': 'Market pullback or consolidation',
        'strategy': 'Wait for confirmation, selective accumulation on dips',
        'risk_level': 'Moderate',
    },
    2: {
        'name': 'Bull Market',
        'description': 'Stable uptrend with sustainable growth',
        'strategy': 'DCA consistently, follow trend, hold long-term',
        'risk_level': 'Low',
    },
    3: {
        'name': 'Expansion',
        'description': 'Strong growth, post-crash recovery or late-cycle momentum',
        'strategy': 'Ride momentum but prepare for consolidation',
        'risk_level': 'Moderate',
    },
}

# Legacy name -> canonical name mapping for backward compatibility
LEGACY_TO_CANONICAL: Dict[str, str] = {
    # Convention C (Legacy crypto cycle names)
    'Accumulation': 'Bear Market',
    'accumulation': 'Bear Market',
    'Euphoria': 'Bull Market',
    'euphoria': 'Bull Market',
    'Distribution': 'Expansion',
    'distribution': 'Expansion',
    'Contraction': 'Bear Market',
    'contraction': 'Bear Market',
    # Convention B (Training script short names)
    'Bull': 'Bull Market',
    'bull': 'Bull Market',
    'Bear': 'Bear Market',
    'bear': 'Bear Market',
    'Sideways': 'Correction',
    'sideways': 'Correction',
    # Snake-case variants (orchestrator)
    'bull_market': 'Bull Market',
    'bear_market': 'Bear Market',
    'expansion': 'Expansion',
    'Expansion': 'Expansion',
    'correction': 'Correction',
    'Correction': 'Correction',
    'Bull Market': 'Bull Market',
    'Bear Market': 'Bear Market',
    # French variants
    'Euphorie': 'Bull Market',
    'euphorie': 'Bull Market',
    # Neutral / Unknown fallbacks
    'neutral': 'Correction',
    'unknown': 'Correction',
    'Unknown': 'Correction',
    'Consolidation': 'Correction',
    'consolidation': 'Correction',
    # Old legacy names with capitalization
    'early_expansion': 'Correction',
    'Early Expansion': 'Correction',
}


def score_to_regime(score: float) -> int:
    """Convert score (0-100, 100=best) to regime ID."""
    if score <= 25:
        return MarketRegime.BEAR_MARKET
    elif score <= 50:
        return MarketRegime.CORRECTION
    elif score <= 75:
        return MarketRegime.BULL_MARKET
    else:
        return MarketRegime.EXPANSION


def regime_name(regime_id: int) -> str:
    """Get regime name from ID. Clamps to valid range."""
    return REGIME_NAMES[max(0, min(3, regime_id))]


def normalize_regime_name(name: str) -> str:
    """Convert any legacy/variant regime name to canonical form.

    Returns the canonical name ('Bear Market', 'Correction', 'Bull Market', 'Expansion')
    or the input unchanged if not recognized.
    """
    if name in LEGACY_TO_CANONICAL:
        return LEGACY_TO_CANONICAL[name]
    lower = name.lower() if isinstance(name, str) else ''
    if lower in LEGACY_TO_CANONICAL:
        return LEGACY_TO_CANONICAL[lower]
    return name


def regime_to_key(name: str) -> str:
    """Convert regime name to snake_case key for JS/dict usage.

    'Bear Market' -> 'bear_market', 'Bull Market' -> 'bull_market', etc.
    """
    canonical = normalize_regime_name(name)
    return canonical.lower().replace(' ', '_')
