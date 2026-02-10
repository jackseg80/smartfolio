"""Tests for services/portfolio_metrics.py — cached functions, dataclasses, helper methods."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from services.portfolio_metrics import (
    _cached_var_cvar,
    _cached_downside_deviation,
    _cached_correlation_matrix,
    PortfolioMetrics,
    CorrelationMetrics,
    PortfolioMetricsService,
)


# ===========================================================================
# _cached_var_cvar
# ===========================================================================

class TestCachedVarCvar:
    def test_basic_returns(self):
        returns = tuple(np.random.normal(0.001, 0.02, 500).tolist())
        result = _cached_var_cvar(returns, 0.95, 0.99)
        assert "var_95" in result and "var_99" in result
        assert "cvar_95" in result and "cvar_99" in result

    def test_var_99_more_extreme_than_95(self):
        returns = tuple(np.random.normal(0.0, 0.03, 1000).tolist())
        result = _cached_var_cvar(returns, 0.95, 0.99)
        assert result["var_99"] <= result["var_95"]

    def test_cvar_more_extreme_than_var(self):
        returns = tuple(np.random.normal(-0.01, 0.05, 1000).tolist())
        result = _cached_var_cvar(returns, 0.95, 0.99)
        assert result["cvar_95"] <= result["var_95"]

    def test_all_positive_returns(self):
        returns = tuple([0.01] * 100)
        result = _cached_var_cvar(returns, 0.95, 0.99)
        assert result["var_95"] == pytest.approx(0.01)

    def test_all_negative_returns(self):
        returns = tuple([-0.05] * 100)
        result = _cached_var_cvar(returns, 0.95, 0.99)
        assert result["var_95"] == pytest.approx(-0.05)

    def test_returns_float_values(self):
        returns = tuple(np.random.normal(0, 0.02, 200).tolist())
        result = _cached_var_cvar(returns, 0.95, 0.99)
        for key in ("var_95", "var_99", "cvar_95", "cvar_99"):
            assert isinstance(result[key], float)

    def test_cache_returns_same_result(self):
        returns = tuple([0.01, -0.02, 0.03, -0.01, 0.005] * 20)
        r1 = _cached_var_cvar(returns, 0.95, 0.99)
        r2 = _cached_var_cvar(returns, 0.95, 0.99)
        assert r1 == r2


# ===========================================================================
# _cached_downside_deviation
# ===========================================================================

class TestCachedDownsideDeviation:
    def test_all_positive_returns_zero(self):
        returns = tuple([0.01, 0.02, 0.03, 0.005, 0.015] * 20)
        assert _cached_downside_deviation(returns) == 0.0

    def test_mixed_returns_positive_result(self):
        returns = tuple(np.random.normal(-0.01, 0.03, 200).tolist())
        dd = _cached_downside_deviation(returns)
        assert dd > 0

    def test_all_negative_returns(self):
        returns = tuple([-0.01, -0.02, -0.03, -0.005] * 50)
        dd = _cached_downside_deviation(returns)
        assert dd > 0

    def test_annualized(self):
        """Result should be annualized (multiplied by sqrt(252))."""
        returns = tuple([-0.01, -0.02, -0.03] * 50)
        dd = _cached_downside_deviation(returns)
        # Raw std of [-0.01, -0.02, -0.03] * sqrt(252) ~ large positive
        assert dd > 0.05

    def test_custom_threshold(self):
        returns = tuple([0.01, 0.02, 0.005, 0.015] * 20)
        # All returns < 0.03 threshold
        dd = _cached_downside_deviation(returns, threshold=0.03)
        assert dd > 0


# ===========================================================================
# _cached_correlation_matrix
# ===========================================================================

class TestCachedCorrelationMatrix:
    def test_perfect_correlation(self):
        data = tuple(tuple([float(i), float(i)]) for i in range(100))
        cols = ("A", "B")
        result = _cached_correlation_matrix(data, cols)
        # Perfect correlation between identical columns
        assert abs(result[0][1] - 1.0) < 0.01

    def test_returns_nested_tuples(self):
        data = tuple(tuple(np.random.randn(3).tolist()) for _ in range(50))
        cols = ("X", "Y", "Z")
        result = _cached_correlation_matrix(data, cols)
        assert isinstance(result, tuple)
        assert isinstance(result[0], tuple)
        assert len(result) == 3
        assert len(result[0]) == 3

    def test_diagonal_is_one(self):
        data = tuple(tuple(np.random.randn(2).tolist()) for _ in range(50))
        cols = ("A", "B")
        result = _cached_correlation_matrix(data, cols)
        assert abs(result[0][0] - 1.0) < 0.01
        assert abs(result[1][1] - 1.0) < 0.01

    def test_symmetric(self):
        data = tuple(tuple(np.random.randn(3).tolist()) for _ in range(100))
        cols = ("A", "B", "C")
        result = _cached_correlation_matrix(data, cols)
        assert abs(result[0][1] - result[1][0]) < 0.001


# ===========================================================================
# PortfolioMetrics dataclass
# ===========================================================================

class TestPortfolioMetricsDataclass:
    def test_create_with_all_fields(self):
        pm = PortfolioMetrics(
            total_return_pct=10.5, annualized_return_pct=12.0,
            volatility_annualized=0.25, sharpe_ratio=1.5,
            sortino_ratio=2.0, calmar_ratio=3.0,
            max_drawdown=-0.15, current_drawdown=-0.05,
            max_drawdown_duration_days=30,
            skewness=-0.3, kurtosis=4.0,
            var_95_1d=-0.02, var_99_1d=-0.03,
            cvar_95_1d=-0.025, cvar_99_1d=-0.04,
            ulcer_index=0.05, positive_months_pct=60.0,
            win_loss_ratio=1.5,
        )
        assert pm.total_return_pct == 10.5
        assert pm.sharpe_ratio == 1.5

    def test_default_values(self):
        pm = PortfolioMetrics(
            total_return_pct=0, annualized_return_pct=0,
            volatility_annualized=0, sharpe_ratio=0,
            sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, current_drawdown=0,
            max_drawdown_duration_days=0,
            skewness=0, kurtosis=0,
            var_95_1d=0, var_99_1d=0,
            cvar_95_1d=0, cvar_99_1d=0,
            ulcer_index=0, positive_months_pct=0,
            win_loss_ratio=0,
        )
        assert pm.overall_risk_level == "medium"
        assert pm.risk_score == 50.0
        assert pm.data_points == 0
        assert pm.calculation_date is None
        assert pm.confidence_level == 0.95


# ===========================================================================
# CorrelationMetrics dataclass
# ===========================================================================

class TestCorrelationMetricsDataclass:
    def test_create(self):
        cm = CorrelationMetrics(
            diversification_ratio=1.5,
            effective_assets=5.0,
            top_correlations=[{"asset1": "A", "asset2": "B", "correlation": 0.9}],
        )
        assert cm.diversification_ratio == 1.5
        assert cm.effective_assets == 5.0
        assert len(cm.top_correlations) == 1

    def test_default_matrix_none(self):
        cm = CorrelationMetrics(
            diversification_ratio=1.0, effective_assets=3.0, top_correlations=[],
        )
        assert cm.correlation_matrix is None


# ===========================================================================
# PortfolioMetricsService — helper methods
# ===========================================================================

class TestServiceHelpers:
    @pytest.fixture
    def svc(self):
        return PortfolioMetricsService()

    @pytest.fixture
    def returns_up(self):
        """Mostly positive returns."""
        return pd.Series([0.01, 0.02, -0.005, 0.015, 0.008, -0.002, 0.012] * 10)

    @pytest.fixture
    def returns_down(self):
        """Mostly negative returns."""
        return pd.Series([-0.02, -0.01, 0.005, -0.015, -0.008, 0.002, -0.012] * 10)

    # --- _calculate_total_return ---
    def test_total_return_positive(self, svc, returns_up):
        r = svc._calculate_total_return(returns_up)
        assert r > 0

    def test_total_return_negative(self, svc, returns_down):
        r = svc._calculate_total_return(returns_down)
        assert r < 0

    def test_total_return_zero(self, svc):
        r = svc._calculate_total_return(pd.Series([0.0] * 50))
        assert r == pytest.approx(0.0)

    # --- _calculate_annualized_return ---
    def test_annualized_return_positive(self, svc, returns_up):
        r = svc._calculate_annualized_return(returns_up)
        assert r > 0

    def test_annualized_return_scales_with_time(self, svc):
        short = pd.Series([0.01] * 30)
        long = pd.Series([0.01] * 252)
        r_short = svc._calculate_annualized_return(short)
        r_long = svc._calculate_annualized_return(long)
        # Short period annualized should be larger (extrapolated)
        assert r_short > r_long

    # --- _calculate_volatility ---
    def test_volatility_positive(self, svc, returns_up):
        v = svc._calculate_volatility(returns_up)
        assert v > 0

    def test_volatility_zero_for_constant(self, svc):
        v = svc._calculate_volatility(pd.Series([0.01] * 100))
        assert v == pytest.approx(0.0)

    def test_volatility_annualized(self, svc):
        returns = pd.Series(np.random.normal(0, 0.01, 252))
        v = svc._calculate_volatility(returns)
        # Annualized vol ~ daily_std * sqrt(252) ~ 0.01 * 15.87 ~ 0.16
        assert 0.05 < v < 0.40

    # --- _calculate_sharpe_ratio ---
    def test_sharpe_ratio_positive(self, svc):
        r = svc._calculate_sharpe_ratio(annualized_return=0.10, volatility=0.20)
        # (0.10 - 0.02) / 0.20 = 0.40
        assert r == pytest.approx(0.40)

    def test_sharpe_ratio_zero_vol(self, svc):
        assert svc._calculate_sharpe_ratio(0.10, 0.0) == 0

    def test_sharpe_ratio_negative(self, svc):
        r = svc._calculate_sharpe_ratio(-0.05, 0.20)
        assert r < 0

    # --- _calculate_sortino_ratio ---
    def test_sortino_all_positive_inf(self, svc):
        returns = pd.Series([0.01, 0.02, 0.03] * 30)
        r = svc._calculate_sortino_ratio(returns, annualized_return=0.10)
        assert r == float('inf')

    def test_sortino_mixed(self, svc, returns_up):
        ann = svc._calculate_annualized_return(returns_up)
        r = svc._calculate_sortino_ratio(returns_up, ann)
        assert isinstance(r, float)

    # --- _calculate_drawdown_metrics ---
    def test_drawdown_metrics_keys(self, svc, returns_up):
        dd = svc._calculate_drawdown_metrics(returns_up)
        assert "max_drawdown" in dd
        assert "current_drawdown" in dd
        assert "max_duration_days" in dd

    def test_drawdown_max_negative(self, svc, returns_down):
        dd = svc._calculate_drawdown_metrics(returns_down)
        assert dd["max_drawdown"] < 0

    def test_drawdown_zero_for_monotonic_up(self, svc):
        returns = pd.Series([0.01] * 50)
        dd = svc._calculate_drawdown_metrics(returns)
        assert dd["max_drawdown"] == pytest.approx(0.0)

    # --- _calculate_skewness ---
    def test_skewness_symmetric_near_zero(self, svc):
        returns = pd.Series(np.random.normal(0, 0.01, 10000))
        s = svc._calculate_skewness(returns)
        assert abs(s) < 0.1

    # --- _calculate_kurtosis ---
    def test_kurtosis_normal_near_zero(self, svc):
        """Excess kurtosis of normal distribution ~ 0."""
        returns = pd.Series(np.random.normal(0, 0.01, 10000))
        k = svc._calculate_kurtosis(returns)
        assert abs(k) < 0.5

    # --- _calculate_ulcer_index ---
    def test_ulcer_index_positive(self, svc, returns_down):
        ui = svc._calculate_ulcer_index(returns_down)
        assert ui > 0

    def test_ulcer_index_zero_monotonic_up(self, svc):
        returns = pd.Series([0.01] * 50)
        ui = svc._calculate_ulcer_index(returns)
        assert ui == pytest.approx(0.0)

    # --- _calculate_positive_months_pct ---
    def test_positive_pct_all_positive(self, svc):
        returns = pd.Series([0.01, 0.02, 0.03] * 20)
        assert svc._calculate_positive_months_pct(returns) == 100.0

    def test_positive_pct_all_negative(self, svc):
        returns = pd.Series([-0.01, -0.02] * 20)
        assert svc._calculate_positive_months_pct(returns) == 0.0

    def test_positive_pct_half(self, svc):
        returns = pd.Series([0.01, -0.01] * 20)
        assert svc._calculate_positive_months_pct(returns) == 50.0

    # --- _calculate_win_loss_ratio ---
    def test_win_loss_all_wins(self, svc):
        returns = pd.Series([0.01, 0.02, 0.03] * 20)
        assert svc._calculate_win_loss_ratio(returns) == float('inf')

    def test_win_loss_balanced(self, svc):
        returns = pd.Series([0.02, -0.01] * 20)
        r = svc._calculate_win_loss_ratio(returns)
        assert r == pytest.approx(2.0)

    def test_win_loss_all_losses(self, svc):
        returns = pd.Series([-0.01, -0.02] * 20)
        assert svc._calculate_win_loss_ratio(returns) == 0

    # --- _calculate_effective_assets ---
    def test_effective_assets_equal_weight(self, svc):
        n = 10
        weights = np.ones(n) / n
        corr = pd.DataFrame(np.eye(n))
        result = svc._calculate_effective_assets(corr, weights)
        assert result == pytest.approx(n)

    def test_effective_assets_single(self, svc):
        weights = np.array([1.0])
        corr = pd.DataFrame([[1.0]])
        assert svc._calculate_effective_assets(corr, weights) == pytest.approx(1.0)

    # --- _get_top_correlations ---
    def test_top_correlations_filters_by_threshold(self, svc):
        corr = pd.DataFrame(
            [[1.0, 0.9, 0.3], [0.9, 1.0, 0.5], [0.3, 0.5, 1.0]],
            columns=["A", "B", "C"], index=["A", "B", "C"],
        )
        result = svc._get_top_correlations(corr, threshold=0.7)
        assert len(result) == 1
        assert result[0]["asset1"] == "A"
        assert result[0]["asset2"] == "B"

    def test_top_correlations_sorted_descending(self, svc):
        corr = pd.DataFrame(
            [[1.0, 0.8, 0.9], [0.8, 1.0, 0.7], [0.9, 0.7, 1.0]],
            columns=["A", "B", "C"], index=["A", "B", "C"],
        )
        result = svc._get_top_correlations(corr, threshold=0.7)
        assert result[0]["correlation"] >= result[-1]["correlation"]

    def test_top_correlations_max_10(self, svc):
        n = 20
        data = np.ones((n, n)) * 0.9
        np.fill_diagonal(data, 1.0)
        corr = pd.DataFrame(data, columns=[f"A{i}" for i in range(n)],
                            index=[f"A{i}" for i in range(n)])
        result = svc._get_top_correlations(corr, threshold=0.5)
        assert len(result) <= 10

    def test_top_correlations_empty_below_threshold(self, svc):
        corr = pd.DataFrame(
            [[1.0, 0.1], [0.1, 1.0]],
            columns=["A", "B"], index=["A", "B"],
        )
        result = svc._get_top_correlations(corr, threshold=0.7)
        assert len(result) == 0


# ===========================================================================
# PortfolioMetricsService — risk_free_rate
# ===========================================================================

class TestServiceConfig:
    def test_default_risk_free_rate(self):
        svc = PortfolioMetricsService()
        assert svc.risk_free_rate == 0.02
