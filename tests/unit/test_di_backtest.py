"""
Comprehensive unit tests for the DI Backtest module.

Module 1: services/di_backtest/data_sources.py — HistoricalDataSources
  - Cycle score deterministic computation (halvings, double-sigmoid, derivative, confidence)
  - compute_historical_cycle_scores date range generation

Module 2: services/di_backtest/trading_strategies.py — Trading Strategies
  - DIStrategyConfig / ReplicaParams dataclass defaults
  - DIThresholdStrategy (S1): allocation by DI thresholds
  - DIContrarianStrategy (S3): contrarian allocation
  - DISignalStrategy (S5): signal-based with state and confirmation
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from services.di_backtest.data_sources import (
    BITCOIN_HALVINGS,
    CYCLE_PARAMS,
    HistoricalDataSources,
)
from services.di_backtest.trading_strategies import (
    DIContrarianStrategy,
    DISignalStrategy,
    DIStrategyConfig,
    DIThresholdStrategy,
    ReplicaParams,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def ds():
    """Fresh HistoricalDataSources instance."""
    return HistoricalDataSources()


@pytest.fixture
def price_data_3col():
    """100-day price DataFrame with BTC, ETH, USDT columns."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    btc = 40000 + np.random.lognormal(0, 0.02, 100).cumsum()
    eth = 2000 + np.random.lognormal(0, 0.03, 100).cumsum()
    usdt = np.ones(100)
    return pd.DataFrame({"BTC": btc, "ETH": eth, "USDT": usdt}, index=dates)


@pytest.fixture
def price_data_risky_only():
    """100-day price DataFrame with BTC and ETH only (no stablecoin)."""
    np.random.seed(7)
    dates = pd.date_range("2024-06-01", periods=100, freq="D")
    btc = 50000 + np.random.lognormal(0, 0.02, 100).cumsum()
    eth = 3000 + np.random.lognormal(0, 0.03, 100).cumsum()
    return pd.DataFrame({"BTC": btc, "ETH": eth}, index=dates)


# ============================================================
# Module 1: HistoricalDataSources — Constants
# ============================================================

class TestBitcoinHalvingConstants:
    """Verify halving dates and cycle params are correct."""

    def test_halvings_count(self):
        assert len(BITCOIN_HALVINGS) == 4

    def test_halvings_chronological(self):
        for i in range(len(BITCOIN_HALVINGS) - 1):
            assert BITCOIN_HALVINGS[i] < BITCOIN_HALVINGS[i + 1]

    def test_halving_dates_exact(self):
        assert BITCOIN_HALVINGS[0] == datetime(2012, 11, 28)
        assert BITCOIN_HALVINGS[1] == datetime(2016, 7, 9)
        assert BITCOIN_HALVINGS[2] == datetime(2020, 5, 11)
        assert BITCOIN_HALVINGS[3] == datetime(2024, 4, 20)

    def test_cycle_params_keys(self):
        expected_keys = {"m_rise_center", "m_fall_center", "k_rise", "k_fall",
                         "p_shape", "floor", "ceil"}
        assert set(CYCLE_PARAMS.keys()) == expected_keys

    def test_cycle_params_values(self):
        assert CYCLE_PARAMS["m_rise_center"] == 5.0
        assert CYCLE_PARAMS["m_fall_center"] == 24.0
        assert CYCLE_PARAMS["floor"] == 0
        assert CYCLE_PARAMS["ceil"] == 100


# ============================================================
# Module 1: get_months_after_halving
# ============================================================

class TestGetMonthsAfterHalving:
    """Test months-since-halving calculation."""

    def test_right_after_halving(self, ds):
        """Day after 2024 halving should be ~0 months."""
        date = datetime(2024, 4, 21)
        months = ds.get_months_after_halving(date)
        assert months == pytest.approx(1 / 30.44, abs=0.05)

    def test_one_year_after_halving(self, ds):
        """One year after 2024 halving should be ~12 months."""
        date = datetime(2025, 4, 20)
        months = ds.get_months_after_halving(date)
        assert months == pytest.approx(12.0, abs=0.5)

    def test_before_first_halving_returns_fallback(self, ds):
        """Date before first known halving returns 48.0 fallback."""
        date = datetime(2012, 1, 1)
        months = ds.get_months_after_halving(date)
        assert months == 48.0

    def test_between_halvings(self, ds):
        """Mid-cycle date between 2nd and 3rd halving."""
        date = datetime(2018, 7, 9)  # 2 years after 2016 halving
        months = ds.get_months_after_halving(date)
        assert months == pytest.approx(24.0, abs=0.5)

    def test_on_halving_date_returns_zero(self, ds):
        """Exactly on a halving date should return 0 months."""
        date = datetime(2020, 5, 11)
        months = ds.get_months_after_halving(date)
        assert months == 0.0

    def test_months_never_negative(self, ds):
        """Result should never be negative for any post-halving date."""
        for year in range(2013, 2026):
            date = datetime(year, 6, 15)
            months = ds.get_months_after_halving(date)
            assert months >= 0, f"Negative months for {date}: {months}"


# ============================================================
# Module 1: cycle_score_from_months
# ============================================================

class TestCycleScoreFromMonths:
    """Test double-sigmoid cycle score model."""

    def test_negative_months_returns_50(self, ds):
        assert ds.cycle_score_from_months(-1.0) == 50.0
        assert ds.cycle_score_from_months(-10.0) == 50.0

    def test_month_zero_low_score(self, ds):
        """Very early cycle should have a low score."""
        score = ds.cycle_score_from_months(0.0)
        assert score < 20, f"Month 0 score should be low, got {score:.1f}"

    def test_rising_phase_around_center(self, ds):
        """Around m_rise_center (~5 months), score should be rising fast."""
        score_3 = ds.cycle_score_from_months(3.0)
        score_5 = ds.cycle_score_from_months(5.0)
        score_8 = ds.cycle_score_from_months(8.0)
        assert score_3 < score_5 < score_8, (
            f"Expected monotonic rise: {score_3:.1f} < {score_5:.1f} < {score_8:.1f}"
        )

    def test_peak_area(self, ds):
        """Around months 12-18, score should be near peak (high)."""
        score = ds.cycle_score_from_months(15.0)
        assert score > 70, f"Peak area should have high score, got {score:.1f}"

    def test_falling_phase_around_center(self, ds):
        """Around m_fall_center (~24 months), score should be declining."""
        score_20 = ds.cycle_score_from_months(20.0)
        score_24 = ds.cycle_score_from_months(24.0)
        score_30 = ds.cycle_score_from_months(30.0)
        assert score_20 > score_24 > score_30, (
            f"Expected decline: {score_20:.1f} > {score_24:.1f} > {score_30:.1f}"
        )

    def test_late_cycle_low(self, ds):
        """Month ~40 (late bear) should be low."""
        score = ds.cycle_score_from_months(40.0)
        assert score < 15, f"Late cycle should be low, got {score:.1f}"

    def test_result_always_clamped_0_100(self, ds):
        """Score should always be in [0, 100] for any input."""
        for m in np.arange(0, 100, 0.5):
            score = ds.cycle_score_from_months(m)
            assert 0 <= score <= 100, f"Score out of range at month {m}: {score}"

    def test_cyclic_48_months(self, ds):
        """Score at month M and M+48 should be identical (modulo 48)."""
        for m in [5, 15, 25, 35]:
            score_m = ds.cycle_score_from_months(m)
            score_m48 = ds.cycle_score_from_months(m + 48)
            assert score_m == pytest.approx(score_m48, abs=0.01), (
                f"Month {m} vs {m + 48}: {score_m:.2f} vs {score_m48:.2f}"
            )


# ============================================================
# Module 1: cycle_score_derivative
# ============================================================

class TestCycleScoreDerivative:
    """Test finite-difference derivative of cycle score."""

    def test_positive_derivative_early_cycle(self, ds):
        """Early cycle (months 2-8) should have positive derivative (rising)."""
        for m in [2, 5, 8]:
            deriv = ds.cycle_score_derivative(float(m))
            assert deriv > 0, f"Month {m}: expected positive derivative, got {deriv:.2f}"

    def test_negative_derivative_late_cycle(self, ds):
        """Late cycle (months 20-30) should have negative derivative (falling)."""
        for m in [22, 25, 28]:
            deriv = ds.cycle_score_derivative(float(m))
            assert deriv < 0, f"Month {m}: expected negative derivative, got {deriv:.2f}"

    def test_near_zero_at_peak(self, ds):
        """Around month 13-15 (peak), derivative should be near zero."""
        deriv = ds.cycle_score_derivative(14.0)
        assert abs(deriv) < 5, f"Peak derivative should be near zero, got {deriv:.2f}"

    def test_near_zero_at_trough(self, ds):
        """Around month 40-44 (trough), derivative should be near zero."""
        deriv = ds.cycle_score_derivative(42.0)
        assert abs(deriv) < 3, f"Trough derivative should be near zero, got {deriv:.2f}"

    def test_derivative_is_finite_difference(self, ds):
        """Verify derivative matches manual finite difference calculation."""
        m = 10.0
        delta = 0.5
        expected = (ds.cycle_score_from_months(m + delta) -
                    ds.cycle_score_from_months(m - delta)) / (2 * delta)
        actual = ds.cycle_score_derivative(m)
        assert actual == pytest.approx(expected, abs=1e-10)


# ============================================================
# Module 1: cycle_confidence
# ============================================================

class TestCycleConfidence:
    """Test cycle confidence based on phase center distance."""

    def test_confidence_at_accumulation_center(self, ds):
        """At accumulation center (month 3), confidence should be 0.9."""
        conf = ds.cycle_confidence(3.0)
        assert conf == pytest.approx(0.9, abs=0.01)

    def test_confidence_at_bull_build_center(self, ds):
        """At bull_build center (month 12), confidence should be 0.9."""
        conf = ds.cycle_confidence(12.0)
        assert conf == pytest.approx(0.9, abs=0.01)

    def test_confidence_at_peak_center(self, ds):
        """At peak center (month 21), confidence should be 0.9."""
        conf = ds.cycle_confidence(21.0)
        assert conf == pytest.approx(0.9, abs=0.01)

    def test_confidence_at_bear_center(self, ds):
        """At bear center (month 30), confidence should be 0.9."""
        conf = ds.cycle_confidence(30.0)
        assert conf == pytest.approx(0.9, abs=0.01)

    def test_confidence_at_pre_accumulation_center(self, ds):
        """At pre_accumulation center (month 42), confidence should be 0.9."""
        conf = ds.cycle_confidence(42.0)
        assert conf == pytest.approx(0.9, abs=0.01)

    def test_confidence_at_boundary_is_lower(self, ds):
        """At phase boundaries, confidence drops toward 0.4."""
        # Boundary between accumulation (0-7) and bull_build (7-19)
        conf_boundary = ds.cycle_confidence(6.9)
        conf_center = ds.cycle_confidence(3.0)
        assert conf_boundary < conf_center, (
            f"Boundary confidence {conf_boundary:.2f} should be < center {conf_center:.2f}"
        )

    def test_confidence_always_in_range(self, ds):
        """Confidence always in [0.4, 0.9] for any month in [0, 48)."""
        for m in np.arange(0, 48, 0.5):
            conf = ds.cycle_confidence(float(m))
            assert 0.4 <= conf <= 0.9, f"Month {m}: confidence {conf} out of [0.4, 0.9]"

    def test_confidence_cyclic(self, ds):
        """Confidence at month M and M+48 should be identical."""
        for m in [5, 15, 25, 35, 45]:
            conf_m = ds.cycle_confidence(float(m))
            conf_m48 = ds.cycle_confidence(float(m) + 48.0)
            assert conf_m == pytest.approx(conf_m48, abs=0.01)


# ============================================================
# Module 1: compute_historical_cycle_scores
# ============================================================

class TestComputeHistoricalCycleScores:
    """Test date-range cycle score series generation."""

    def test_series_length_matches_date_range(self, ds):
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 31)
        series = ds.compute_historical_cycle_scores(start, end)
        expected_days = (end - start).days + 1  # inclusive
        assert len(series) == expected_days

    def test_all_values_in_0_100(self, ds):
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)
        series = ds.compute_historical_cycle_scores(start, end)
        assert series.min() >= 0
        assert series.max() <= 100

    def test_index_is_datetimeindex(self, ds):
        start = datetime(2023, 6, 1)
        end = datetime(2023, 6, 30)
        series = ds.compute_historical_cycle_scores(start, end)
        assert isinstance(series.index, pd.DatetimeIndex)

    def test_series_name(self, ds):
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        series = ds.compute_historical_cycle_scores(start, end)
        assert series.name == "cycle_score"

    def test_single_day_range(self, ds):
        date = datetime(2024, 4, 20)
        series = ds.compute_historical_cycle_scores(date, date)
        assert len(series) == 1


# ============================================================
# Module 2: Dataclass Defaults
# ============================================================

class TestDIStrategyConfigDefaults:
    """Verify DIStrategyConfig has sensible defaults."""

    def test_default_thresholds(self):
        c = DIStrategyConfig()
        assert c.di_extreme_fear == 20.0
        assert c.di_fear == 40.0
        assert c.di_neutral_low == 50.0
        assert c.di_neutral_high == 60.0
        assert c.di_greed == 70.0
        assert c.di_extreme_greed == 80.0

    def test_default_allocations(self):
        c = DIStrategyConfig()
        assert c.alloc_extreme_fear == 0.30
        assert c.alloc_fear == 0.50
        assert c.alloc_neutral == 0.60
        assert c.alloc_greed == 0.75
        assert c.alloc_extreme_greed == 0.85

    def test_default_signal_params(self):
        c = DIStrategyConfig()
        assert c.signal_entry_threshold == 40.0
        assert c.signal_exit_threshold == 60.0
        assert c.signal_confirmation_days == 3
        assert c.signal_min_holding_days == 14


class TestReplicaParamsDefaults:
    """Verify ReplicaParams defaults match production."""

    def test_layer_toggles_default_true(self):
        p = ReplicaParams()
        assert p.enable_risk_budget is True
        assert p.enable_market_overrides is True
        assert p.enable_exposure_cap is True
        assert p.enable_governance_penalty is True
        assert p.enable_direction_penalty is True

    def test_risk_budget_bounds(self):
        p = ReplicaParams()
        assert p.risk_budget_min == 0.20
        assert p.risk_budget_max == 0.85

    def test_exposure_confidence(self):
        p = ReplicaParams()
        assert p.exposure_confidence == 0.65

    def test_max_governance_penalty(self):
        p = ReplicaParams()
        assert p.max_governance_penalty == 0.25


# ============================================================
# Module 2: DIThresholdStrategy (S1)
# ============================================================

class TestDIThresholdStrategy:
    """Test threshold-based allocation strategy."""

    def test_extreme_fear_allocation(self, price_data_3col):
        """DI < 20 -> 30% risky."""
        strategy = DIThresholdStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=10.0)

        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.30, abs=0.01)
        assert weights["USDT"] == pytest.approx(0.70, abs=0.01)

    def test_fear_allocation(self, price_data_3col):
        """DI between 20 and 40 -> 50% risky."""
        strategy = DIThresholdStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=30.0)

        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.50, abs=0.01)

    def test_neutral_allocation(self, price_data_3col):
        """DI between 40 and 60 -> 60% risky."""
        strategy = DIThresholdStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=50.0)

        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.60, abs=0.01)

    def test_greed_allocation(self, price_data_3col):
        """DI between 60 and 80 -> 75% risky."""
        strategy = DIThresholdStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=70.0)

        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.75, abs=0.01)

    def test_extreme_greed_allocation(self, price_data_3col):
        """DI > 80 -> 85% risky."""
        strategy = DIThresholdStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=90.0)

        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.85, abs=0.01)

    def test_weights_sum_to_one(self, price_data_3col):
        """Weights must always sum to 1.0."""
        strategy = DIThresholdStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        for di in [0, 15, 30, 50, 70, 85, 100]:
            weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=float(di))
            assert weights.sum() == pytest.approx(1.0, abs=1e-10), (
                f"DI={di}: weights sum = {weights.sum()}"
            )

    def test_stablecoins_get_stable_allocation(self, price_data_3col):
        """USDT (stablecoin) gets the stable portion of allocation."""
        strategy = DIThresholdStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=10.0)
        # DI=10 -> extreme fear -> 30% risky, 70% stable
        assert weights["USDT"] == pytest.approx(0.70, abs=0.01)

    def test_risky_equally_split(self, price_data_3col):
        """Risky allocation split equally among risky assets."""
        strategy = DIThresholdStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=50.0)
        # 60% risky / 2 risky assets = 30% each
        assert weights["BTC"] == pytest.approx(0.30, abs=0.01)
        assert weights["ETH"] == pytest.approx(0.30, abs=0.01)

    def test_no_stablecoins_all_risky(self, price_data_risky_only):
        """When no stablecoins, all assets treated as risky."""
        strategy = DIThresholdStrategy()
        date = price_data_risky_only.index[50]
        current_weights = pd.Series(0.0, index=price_data_risky_only.columns)

        weights = strategy.get_weights(date, price_data_risky_only, current_weights, di_value=50.0)
        # When no stable assets, risky_assets = all assets, stable_assets = []
        # risky_pct goes to all assets, stable_pct is not distributed
        # Weights still sum to risky_pct (0.60) split between BTC and ETH
        # But stable_pct has nowhere to go -> only risky portion is allocated
        # Actually: risky_pct / 2 per asset, and stable_assets is empty so no stable alloc
        assert weights.sum() == pytest.approx(0.60, abs=0.01)

    def test_default_di_when_none(self, price_data_3col):
        """When di_value is None and no di_series, defaults to 50 (neutral)."""
        strategy = DIThresholdStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights)
        risky_total = weights["BTC"] + weights["ETH"]
        # DI=50 -> neutral -> 60% risky
        assert risky_total == pytest.approx(0.60, abs=0.01)

    def test_di_series_lookup(self, price_data_3col):
        """Strategy uses injected DI series when di_value not in kwargs."""
        strategy = DIThresholdStrategy()
        di_series = pd.Series(
            15.0, index=price_data_3col.index, name="di"
        )
        strategy.set_di_series(di_series)

        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights)
        risky_total = weights["BTC"] + weights["ETH"]
        # DI=15 -> extreme fear -> 30% risky
        assert risky_total == pytest.approx(0.30, abs=0.01)

    def test_custom_config(self, price_data_3col):
        """Custom DIStrategyConfig overrides default thresholds."""
        config = DIStrategyConfig(
            di_extreme_fear=10.0,
            alloc_extreme_fear=0.20,
        )
        strategy = DIThresholdStrategy(config=config)
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        # DI=5 < custom extreme_fear=10 -> 20% risky
        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=5.0)
        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.20, abs=0.01)


# ============================================================
# Module 2: DIContrarianStrategy (S3)
# ============================================================

class TestDIContrarianStrategy:
    """Test contrarian allocation strategy."""

    def test_extreme_fear_goes_aggressive(self, price_data_3col):
        """DI < 20 (extreme fear) -> 85% risky (contrarian: buy fear)."""
        strategy = DIContrarianStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=10.0)
        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.85, abs=0.01)

    def test_extreme_greed_goes_defensive(self, price_data_3col):
        """DI > 80 (extreme greed) -> 30% risky (contrarian: sell greed)."""
        strategy = DIContrarianStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=90.0)
        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.30, abs=0.01)

    def test_neutral_zone(self, price_data_3col):
        """DI in neutral zone (40-70) -> 60% risky."""
        strategy = DIContrarianStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=55.0)
        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.60, abs=0.01)

    def test_fear_zone(self, price_data_3col):
        """DI between 20 and 40 -> 70% risky (contrarian: more risk in fear)."""
        strategy = DIContrarianStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=30.0)
        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.70, abs=0.01)

    def test_greed_zone(self, price_data_3col):
        """DI between 70 and 80 -> 45% risky (contrarian: less risk in greed)."""
        strategy = DIContrarianStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=75.0)
        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(0.45, abs=0.01)

    def test_weights_sum_to_one(self, price_data_3col):
        """Contrarian weights always sum to 1.0."""
        strategy = DIContrarianStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        for di in [5, 15, 30, 55, 75, 90]:
            weights = strategy.get_weights(date, price_data_3col, current_weights, di_value=float(di))
            assert weights.sum() == pytest.approx(1.0, abs=1e-10), (
                f"DI={di}: weights sum = {weights.sum()}"
            )

    def test_contrarian_is_opposite_of_threshold(self, price_data_3col):
        """In extreme zones, contrarian gives opposite allocation to threshold."""
        threshold = DIThresholdStrategy()
        contrarian = DIContrarianStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        # Extreme fear: threshold is defensive, contrarian is aggressive
        t_weights = threshold.get_weights(date, price_data_3col, current_weights, di_value=10.0)
        c_weights = contrarian.get_weights(date, price_data_3col, current_weights, di_value=10.0)
        t_risky = t_weights["BTC"] + t_weights["ETH"]
        c_risky = c_weights["BTC"] + c_weights["ETH"]
        assert c_risky > t_risky, (
            f"Contrarian should be more aggressive in fear: {c_risky:.2f} vs {t_risky:.2f}"
        )

        # Extreme greed: threshold is aggressive, contrarian is defensive
        t_weights = threshold.get_weights(date, price_data_3col, current_weights, di_value=90.0)
        c_weights = contrarian.get_weights(date, price_data_3col, current_weights, di_value=90.0)
        t_risky = t_weights["BTC"] + t_weights["ETH"]
        c_risky = c_weights["BTC"] + c_weights["ETH"]
        assert c_risky < t_risky, (
            f"Contrarian should be more defensive in greed: {c_risky:.2f} vs {t_risky:.2f}"
        )


# ============================================================
# Module 2: DISignalStrategy (S5)
# ============================================================

class TestDISignalStrategy:
    """Test signal-based strategy with state management."""

    def test_reset_state(self):
        """reset_state clears all internal state."""
        strategy = DISignalStrategy()
        strategy._current_position = "long"
        strategy._confirmation_count = 5
        strategy._last_trade_date = pd.Timestamp("2024-01-15")

        strategy.reset_state()

        assert strategy._current_position == "neutral"
        assert strategy._confirmation_count == 0
        assert strategy._last_trade_date is None

    def test_equal_weight_without_di_series(self, price_data_3col):
        """Without DI series, returns equal weight across all assets."""
        strategy = DISignalStrategy()
        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights)

        expected_weight = 1.0 / len(price_data_3col.columns)
        for asset in price_data_3col.columns:
            assert weights[asset] == pytest.approx(expected_weight, abs=1e-10)

    def test_neutral_position_50_50(self, price_data_3col):
        """In neutral position with DI series, 50/50 risky/stable."""
        strategy = DISignalStrategy()
        di_series = pd.Series(50.0, index=price_data_3col.index)
        strategy.set_di_series(di_series)

        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights)

        risky_total = weights["BTC"] + weights["ETH"]
        stable_total = weights["USDT"]
        assert risky_total == pytest.approx(0.50, abs=0.01)
        assert stable_total == pytest.approx(0.50, abs=0.01)

    def test_long_position_full_risky(self, price_data_3col):
        """In long position, 100% risky (no stables)."""
        strategy = DISignalStrategy()
        strategy._current_position = "long"

        # Create DI series that stays above entry threshold
        di_series = pd.Series(50.0, index=price_data_3col.index)
        strategy.set_di_series(di_series)

        date = price_data_3col.index[50]
        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(date, price_data_3col, current_weights)

        risky_total = weights["BTC"] + weights["ETH"]
        assert risky_total == pytest.approx(1.0, abs=0.01)
        assert weights["USDT"] == pytest.approx(0.0, abs=0.01)

    def test_confirmation_required_for_entry(self, price_data_3col):
        """Entry requires signal_confirmation_days (default 3) consecutive crossings."""
        config = DIStrategyConfig(signal_confirmation_days=3)
        strategy = DISignalStrategy(config=config)

        # Build DI series that crosses above entry threshold (40) at index 50
        di_values = np.full(100, 35.0)
        di_values[50] = 42.0  # First crossing
        di_values[51] = 42.0  # Second crossing (not a crossing, stays above)
        di_values[52] = 42.0  # Third... but prev is already above, no new crossing
        di_series = pd.Series(di_values, index=price_data_3col.index)
        strategy.set_di_series(di_series)

        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        # After first crossing, confirmation count should be 1, not enough
        weights = strategy.get_weights(
            price_data_3col.index[50], price_data_3col, current_weights
        )
        # Should still be neutral since only 1 confirmation
        assert strategy._current_position == "neutral"

    def test_strategy_name(self):
        """Strategy has correct name."""
        strategy = DISignalStrategy()
        assert strategy.name == "DI Signal"

    def test_holding_minimum_prevents_early_exit(self, price_data_3col):
        """Cannot exit position before signal_min_holding_days."""
        config = DIStrategyConfig(signal_min_holding_days=14)
        strategy = DISignalStrategy(config=config)

        # Manually set long position with recent trade date
        strategy._current_position = "long"
        strategy._last_trade_date = price_data_3col.index[50]

        # Create DI that drops below exit threshold at index 55 (only 5 days later)
        di_values = np.full(100, 70.0)
        di_values[54] = 65.0  # prev_di
        di_values[55] = 55.0  # current_di crosses below exit=60
        di_series = pd.Series(di_values, index=price_data_3col.index)
        strategy.set_di_series(di_series)

        current_weights = pd.Series(0.0, index=price_data_3col.columns)

        weights = strategy.get_weights(
            price_data_3col.index[55], price_data_3col, current_weights
        )

        # Still long because minimum holding period not met (5 < 14 days)
        assert strategy._current_position == "long"


# ============================================================
# Module 2: Strategy name attribute
# ============================================================

class TestStrategyNames:
    """Verify strategy name attributes are set correctly."""

    def test_threshold_strategy_name(self):
        assert DIThresholdStrategy().name == "DI Threshold"

    def test_contrarian_strategy_name(self):
        assert DIContrarianStrategy().name == "DI Contrarian"

    def test_signal_strategy_name(self):
        assert DISignalStrategy().name == "DI Signal"
