"""Tests for services/di_backtest/trading_strategies.py — DI Trading Strategies

Focuses on static methods (pure computation) and strategy get_weights logic.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from services.di_backtest.trading_strategies import (
    DIStrategyConfig,
    ReplicaParams,
    TrendGateParams,
    RotationParams,
    ContinuousParams,
    DIThresholdStrategy,
    DIMomentumStrategy,
    DIContrarianStrategy,
    DIRiskParityStrategy,
    DISignalStrategy,
    DISmartfolioReplicaStrategy,
    DITrendGateStrategy,
    DICycleRotationStrategy,
    DIAdaptiveContinuousStrategy,
    DI_STRATEGIES,
    get_di_strategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_price_data(assets=("BTC", "USDT"), n_days=30, base=50000):
    """Create a simple price DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    data = {}
    for i, asset in enumerate(assets):
        price = base / (10 ** i) if asset != "USDT" else 1.0
        data[asset] = [price * (1 + 0.01 * np.sin(j / 5)) for j in range(n_days)]
    return pd.DataFrame(data, index=dates)


def _make_weights(assets=("BTC", "USDT"), values=(0.5, 0.5)):
    return pd.Series(values, index=list(assets))


# ---------------------------------------------------------------------------
# TestDataclassDefaults
# ---------------------------------------------------------------------------
class TestDataclassDefaults:
    def test_di_strategy_config_defaults(self):
        c = DIStrategyConfig()
        assert c.di_extreme_fear == 20.0
        assert c.alloc_extreme_greed == 0.85
        assert c.momentum_lookback == 7
        assert c.signal_min_holding_days == 14

    def test_replica_params_defaults(self):
        p = ReplicaParams()
        assert p.risk_budget_min == 0.20
        assert p.risk_budget_max == 0.85
        assert p.enable_risk_budget is True
        assert p.max_governance_penalty == 0.25

    def test_trend_gate_params_defaults(self):
        p = TrendGateParams()
        assert p.sma_period == 200
        assert p.risk_on_alloc == 0.80
        assert p.risk_off_alloc == 0.20
        assert p.whipsaw_days == 5

    def test_rotation_params_defaults(self):
        p = RotationParams()
        assert p.phase_bearish_max == 70.0
        assert p.phase_bullish_min == 90.0
        assert p.alloc_bear == (0.15, 0.05, 0.80)
        assert p.smoothing_alpha == 0.15

    def test_continuous_params_defaults(self):
        p = ContinuousParams()
        assert p.alloc_floor == 0.10
        assert p.alloc_ceiling == 0.85
        assert p.sma_fast == 50
        assert p.sma_slow == 200


# ---------------------------------------------------------------------------
# TestDIThresholdStrategy
# ---------------------------------------------------------------------------
class TestDIThresholdStrategy:
    def setup_method(self):
        self.strategy = DIThresholdStrategy()
        self.price_data = _make_price_data()
        self.current_weights = _make_weights()

    def test_extreme_fear_allocation(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights, di_value=10.0)
        # DI < 20 → extreme fear → 30% risky
        assert abs(w["BTC"] - 0.30) < 0.01

    def test_fear_allocation(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights, di_value=35.0)
        # DI 20-40 → fear → 50% risky
        assert abs(w["BTC"] - 0.50) < 0.01

    def test_neutral_allocation(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights, di_value=55.0)
        assert abs(w["BTC"] - 0.60) < 0.01

    def test_greed_allocation(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights, di_value=75.0)
        assert abs(w["BTC"] - 0.75) < 0.01

    def test_extreme_greed_allocation(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights, di_value=90.0)
        assert abs(w["BTC"] - 0.85) < 0.01

    def test_no_di_defaults_to_neutral(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights)
        # No DI → default 50 → neutral → 60%
        assert abs(w["BTC"] - 0.60) < 0.01

    def test_weights_sum_to_one(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights, di_value=42.0)
        assert abs(w.sum() - 1.0) < 0.001

    def test_di_series_injection(self):
        dates = self.price_data.index
        di_series = pd.Series([80.0] * len(dates), index=dates)
        self.strategy.set_di_series(di_series)
        w = self.strategy.get_weights(dates[15], self.price_data, self.current_weights)
        assert w["BTC"] > 0.80  # extreme greed


# ---------------------------------------------------------------------------
# TestDIContrarianStrategy
# ---------------------------------------------------------------------------
class TestDIContrarianStrategy:
    def setup_method(self):
        self.strategy = DIContrarianStrategy()
        self.price_data = _make_price_data()
        self.current_weights = _make_weights()

    def test_extreme_fear_goes_aggressive(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights, di_value=10.0)
        assert abs(w["BTC"] - 0.85) < 0.01  # contrarian: buy fear

    def test_extreme_greed_goes_defensive(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights, di_value=90.0)
        assert abs(w["BTC"] - 0.30) < 0.01  # contrarian: sell greed

    def test_neutral_stays_balanced(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights, di_value=55.0)
        assert abs(w["BTC"] - 0.60) < 0.01


# ---------------------------------------------------------------------------
# TestSmartfolioReplicaStaticMethods
# ---------------------------------------------------------------------------
class TestComputeAdaptiveWeights:
    def test_no_contradiction(self):
        """Scores aligned → standard weights"""
        cw, ow, rw = DISmartfolioReplicaStrategy._compute_adaptive_weights(70, 70, 50)
        assert abs(cw + ow + rw - 1.0) < 0.001
        # No divergence → close to base weights
        assert cw > 0.45

    def test_high_contradiction(self):
        """Large divergence → cycle weight reduced, risk increased"""
        cw, ow, rw = DISmartfolioReplicaStrategy._compute_adaptive_weights(90, 30, 50)
        # Divergence = 60 → full contradiction
        assert rw > 0.25  # risk weight increased from base 0.20
        assert cw < 0.45  # cycle weight reduced from base 0.50

    def test_weights_always_sum_to_one(self):
        for c, o, r in [(0, 100, 50), (50, 50, 50), (100, 0, 100), (30, 80, 20)]:
            cw, ow, rw = DISmartfolioReplicaStrategy._compute_adaptive_weights(c, o, r)
            assert abs(cw + ow + rw - 1.0) < 0.001


class TestComputeRiskBudget:
    def test_neutral_scores(self):
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(50, 50, 50)
        assert 0.20 <= risky <= 0.85

    def test_all_high_scores(self):
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(90, 90, 90)
        assert risky >= 0.70  # High allocation

    def test_all_low_scores(self):
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(10, 10, 10)
        assert risky == pytest.approx(0.20, abs=0.01)  # Clamped to min

    def test_risk_budget_clamp_min(self):
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(0, 0, 0)
        assert risky >= 0.20

    def test_risk_budget_clamp_max(self):
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(100, 100, 100)
        assert risky <= 0.85

    def test_low_risk_score_override(self):
        """Risk score <= 30 → force stables >= 50%"""
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(80, 80, 25)
        assert risky <= 0.50

    def test_on_chain_divergence_override(self):
        """|cycle - onchain| >= 30 → +10% stables"""
        risky_no_div = DISmartfolioReplicaStrategy._compute_risk_budget(70, 70, 70)
        risky_with_div = DISmartfolioReplicaStrategy._compute_risk_budget(90, 50, 70)
        assert risky_with_div < risky_no_div

    def test_overrides_disabled(self):
        params = ReplicaParams(enable_market_overrides=False)
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(90, 50, 25, params)
        # Without overrides, low risk doesn't force stables >= 50%
        assert risky > 0.20

    def test_direction_penalty(self):
        """High cycle + descending direction → penalty"""
        risky_no_dir = DISmartfolioReplicaStrategy._compute_risk_budget(
            85, 70, 70, cycle_direction=None
        )
        risky_with_dir = DISmartfolioReplicaStrategy._compute_risk_budget(
            85, 70, 70, cycle_direction=-0.8, cycle_confidence=0.7
        )
        assert risky_with_dir < risky_no_dir

    def test_direction_penalty_only_high_cycle(self):
        """Direction penalty should NOT apply when cycle <= 80"""
        risky_no_dir = DISmartfolioReplicaStrategy._compute_risk_budget(
            60, 60, 60, cycle_direction=None
        )
        risky_with_dir = DISmartfolioReplicaStrategy._compute_risk_budget(
            60, 60, 60, cycle_direction=-0.9, cycle_confidence=0.9
        )
        assert risky_with_dir == pytest.approx(risky_no_dir, abs=0.01)


class TestComputeContradictionIndex:
    def test_no_contradiction(self):
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.20, cycle_score=50, onchain_score=50, di_value=50
        )
        assert ci == 0.0

    def test_high_vol_bull_contradiction(self):
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.80, cycle_score=80, onchain_score=80, di_value=50
        )
        assert ci > 0.0  # High vol + bull = contradiction check 1

    def test_di_extreme_fear_bull_contradiction(self):
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.20, cycle_score=80, onchain_score=80, di_value=20
        )
        assert ci > 0.0  # DI fear + cycle bull = check 2

    def test_score_divergence_contradiction(self):
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.20, cycle_score=90, onchain_score=40, di_value=50
        )
        assert ci > 0.0  # |90-40| = 50 >= 40 → check 3

    def test_max_contradiction(self):
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.80, cycle_score=85, onchain_score=30, di_value=20
        )
        # All 3 checks triggered
        assert ci > 0.2

    def test_bounded_0_to_1(self):
        for _ in range(10):
            ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
                np.random.uniform(0, 1), np.random.uniform(0, 100),
                np.random.uniform(0, 100), np.random.uniform(0, 100)
            )
            assert 0.0 <= ci <= 1.0


class TestComputeGovernancePenalty:
    def test_low_contradiction_no_penalty(self):
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(0.10)
        assert penalty == 0.0

    def test_moderate_contradiction(self):
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(0.40)
        assert 0.0 < penalty < 0.25

    def test_high_contradiction(self):
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(0.75)
        assert penalty > 0.10

    def test_penalty_capped(self):
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(1.0, btc_volatility=1.0)
        assert penalty <= 0.25  # max_governance_penalty default

    def test_high_vol_amplifies(self):
        penalty_low_vol = DISmartfolioReplicaStrategy._compute_governance_penalty(0.50, btc_volatility=0.20)
        penalty_high_vol = DISmartfolioReplicaStrategy._compute_governance_penalty(0.50, btc_volatility=0.80)
        assert penalty_high_vol >= penalty_low_vol


class TestComputeExposureCap:
    def test_high_blended_high_risk(self):
        cap = DISmartfolioReplicaStrategy._compute_exposure_cap(80, 90, di_value=70)
        assert cap >= 0.75  # Expansion regime

    def test_low_blended_low_risk(self):
        cap = DISmartfolioReplicaStrategy._compute_exposure_cap(20, 20, di_value=30)
        assert cap <= 0.40  # Bear regime

    def test_bounded_20_to_95(self):
        for bs in [0, 25, 50, 75, 100]:
            for rs in [0, 50, 100]:
                cap = DISmartfolioReplicaStrategy._compute_exposure_cap(bs, rs)
                assert 0.20 <= cap <= 0.95

    def test_volatility_penalty(self):
        cap_low_vol = DISmartfolioReplicaStrategy._compute_exposure_cap(70, 70, btc_volatility=0.10)
        cap_high_vol = DISmartfolioReplicaStrategy._compute_exposure_cap(70, 70, btc_volatility=0.80)
        assert cap_high_vol <= cap_low_vol


# ---------------------------------------------------------------------------
# TestDICycleRotation
# ---------------------------------------------------------------------------
class TestDICycleRotation:
    def test_detect_phase_peak(self):
        strategy = DICycleRotationStrategy()
        assert strategy._detect_phase(95, 0.5) == "peak"

    def test_detect_phase_bull_building(self):
        strategy = DICycleRotationStrategy()
        assert strategy._detect_phase(80, 0.5) == "bull_building"

    def test_detect_phase_distribution(self):
        strategy = DICycleRotationStrategy()
        assert strategy._detect_phase(80, -0.5) == "distribution"

    def test_detect_phase_accumulation(self):
        strategy = DICycleRotationStrategy()
        assert strategy._detect_phase(50, 0.5) == "accumulation"

    def test_detect_phase_bear(self):
        strategy = DICycleRotationStrategy()
        assert strategy._detect_phase(50, -0.5) == "bear"

    def test_phase_targets_all_sum_to_one(self):
        strategy = DICycleRotationStrategy()
        for phase in ["accumulation", "bull_building", "peak", "distribution", "bear"]:
            targets = strategy._phase_targets(phase)
            assert abs(sum(targets) - 1.0) < 0.01

    def test_reset_state(self):
        strategy = DICycleRotationStrategy()
        strategy._smoothed_weights = {"BTC": 0.5, "ETH": 0.3, "STABLES": 0.2}
        strategy.reset_state()
        assert strategy._smoothed_weights is None


# ---------------------------------------------------------------------------
# TestDIAdaptiveContinuous static methods
# ---------------------------------------------------------------------------
class TestDiToAllocation:
    def test_bear_floor(self):
        alloc = DIAdaptiveContinuousStrategy._di_to_allocation(10)
        assert alloc == 0.10  # floor

    def test_di_25_boundary(self):
        alloc = DIAdaptiveContinuousStrategy._di_to_allocation(25)
        assert alloc == pytest.approx(0.10, abs=0.01)

    def test_di_50_midpoint(self):
        alloc = DIAdaptiveContinuousStrategy._di_to_allocation(50)
        assert alloc == pytest.approx(0.40, abs=0.01)

    def test_di_75_high(self):
        alloc = DIAdaptiveContinuousStrategy._di_to_allocation(75)
        assert alloc == pytest.approx(0.70, abs=0.01)

    def test_di_100_ceiling(self):
        alloc = DIAdaptiveContinuousStrategy._di_to_allocation(100)
        assert alloc == pytest.approx(0.85, abs=0.01)

    def test_monotonically_increasing(self):
        prev = -1
        for di in range(0, 101, 5):
            alloc = DIAdaptiveContinuousStrategy._di_to_allocation(di)
            assert alloc >= prev
            prev = alloc

    def test_custom_floor_ceiling(self):
        alloc = DIAdaptiveContinuousStrategy._di_to_allocation(10, floor=0.05, ceiling=0.95)
        assert alloc == 0.05


class TestComputeTrendBoost:
    def test_insufficient_data(self):
        prices = pd.Series([100.0] * 50)
        boost = DIAdaptiveContinuousStrategy._compute_trend_boost(prices, sma_fast=50, sma_slow=200)
        assert boost == 0.0  # Not enough data

    def test_golden_cross(self):
        # Create uptrend: price and SMA50 above SMA200
        n = 250
        prices = pd.Series([100 + i * 0.5 for i in range(n)])
        boost = DIAdaptiveContinuousStrategy._compute_trend_boost(prices)
        assert boost == pytest.approx(0.10, abs=0.01)  # Golden cross = +boost

    def test_death_cross(self):
        # Create downtrend: price and SMA50 below SMA200
        n = 250
        prices = pd.Series([200 - i * 0.5 for i in range(n)])
        boost = DIAdaptiveContinuousStrategy._compute_trend_boost(prices)
        assert boost == pytest.approx(-0.10, abs=0.01)  # Death cross = -boost


# ---------------------------------------------------------------------------
# TestDISignalStrategy
# ---------------------------------------------------------------------------
class TestDISignalStrategy:
    def setup_method(self):
        self.strategy = DISignalStrategy()
        self.price_data = _make_price_data()
        self.current_weights = _make_weights()

    def test_neutral_without_di_series(self):
        date = self.price_data.index[15]
        w = self.strategy.get_weights(date, self.price_data, self.current_weights)
        # Equal weight without DI series
        assert abs(w.sum() - 1.0) < 0.001

    def test_reset_state(self):
        self.strategy._current_position = "long"
        self.strategy.reset_state()
        assert self.strategy._current_position == "neutral"
        assert self.strategy._last_trade_date is None


# ---------------------------------------------------------------------------
# TestDIMomentumStrategy
# ---------------------------------------------------------------------------
class TestDIMomentumStrategy:
    def test_weights_sum_to_one(self):
        strategy = DIMomentumStrategy()
        price_data = _make_price_data()
        current_weights = _make_weights()
        dates = price_data.index
        # Create DI series with upward momentum
        di_series = pd.Series([30 + i for i in range(len(dates))], index=dates)
        strategy.set_di_series(di_series)
        w = strategy.get_weights(dates[15], price_data, current_weights)
        assert abs(w.sum() - 1.0) < 0.01

    def test_fallback_without_di_series(self):
        strategy = DIMomentumStrategy()
        price_data = _make_price_data()
        current_weights = _make_weights()
        date = price_data.index[15]
        w = strategy.get_weights(date, price_data, current_weights, di_value=50.0)
        assert abs(w.sum() - 1.0) < 0.01


# ---------------------------------------------------------------------------
# TestDIRiskParityStrategy
# ---------------------------------------------------------------------------
class TestDIRiskParityStrategy:
    def test_inverse_vol_weighting(self):
        strategy = DIRiskParityStrategy(vol_lookback=10)
        price_data = _make_price_data(n_days=30)
        current_weights = _make_weights()
        date = price_data.index[20]
        w = strategy.get_weights(date, price_data, current_weights)
        assert abs(w.sum() - 1.0) < 0.01
        assert all(w >= 0)

    def test_with_di_scaling(self):
        strategy = DIRiskParityStrategy(vol_lookback=10)
        price_data = _make_price_data(n_days=30)
        current_weights = _make_weights()
        date = price_data.index[20]
        di_series = pd.Series([80.0] * 30, index=price_data.index)
        strategy.set_di_series(di_series)
        w = strategy.get_weights(date, price_data, current_weights)
        assert abs(w.sum() - 1.0) < 0.01


# ---------------------------------------------------------------------------
# TestGetDiStrategy factory
# ---------------------------------------------------------------------------
class TestGetDiStrategy:
    def test_all_strategies_available(self):
        expected = {
            "di_threshold", "di_momentum", "di_contrarian", "di_risk_parity",
            "di_signal", "di_smartfolio_replica", "di_trend_gate",
            "di_cycle_rotation", "di_adaptive_continuous",
        }
        assert set(DI_STRATEGIES.keys()) == expected

    def test_factory_creates_each_strategy(self):
        for name in DI_STRATEGIES:
            strategy = get_di_strategy(name)
            assert strategy is not None

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="inconnue"):
            get_di_strategy("nonexistent")

    def test_risk_parity_has_vol_lookback(self):
        strategy = get_di_strategy("di_risk_parity")
        assert isinstance(strategy, DIRiskParityStrategy)
        assert strategy.vol_lookback == 30
