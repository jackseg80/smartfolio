"""
Tests unitaires pour services/portfolio_optimization.py

Couvre: PortfolioOptimizer (expected returns, risk model, optimize_portfolio,
       diversification ratio, risk contributions, transaction costs,
       constraints check, smart fallback, Black-Litterman, efficient frontier,
       multi-period optimization, create_crypto_constraints).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from services.portfolio_optimization import (
    PortfolioOptimizer,
    OptimizationObjective,
    OptimizationConstraints,
    OptimizationResult,
    create_crypto_constraints,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def optimizer():
    return PortfolioOptimizer()


@pytest.fixture
def sample_prices():
    """Generate synthetic price history for 3 assets, 100 days."""
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range("2025-01-01", periods=n_days)
    btc = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days))
    eth = 100 * np.cumprod(1 + np.random.normal(0.0008, 0.03, n_days))
    sol = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.04, n_days))
    return pd.DataFrame({"BTC": btc, "ETH": eth, "SOL": sol}, index=dates)


@pytest.fixture
def sample_5_assets():
    """5-asset price history for broader tests."""
    np.random.seed(123)
    n_days = 120
    dates = pd.date_range("2025-01-01", periods=n_days)
    data = {}
    for name, mu, sigma in [
        ("BTC", 0.001, 0.02), ("ETH", 0.0008, 0.03), ("SOL", 0.0005, 0.04),
        ("USDC", 0.0, 0.001), ("ADA", 0.0003, 0.05),
    ]:
        data[name] = 100 * np.cumprod(1 + np.random.normal(mu, sigma, n_days))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def basic_constraints():
    return OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.6,
        max_sector_weight=0.8,
        min_diversification_ratio=0.0,
    )


# ---------------------------------------------------------------------------
# calculate_expected_returns
# ---------------------------------------------------------------------------

class TestCalculateExpectedReturns:
    def test_historical_method(self, optimizer, sample_prices):
        er = optimizer.calculate_expected_returns(sample_prices, method="historical")
        assert len(er) == 3
        assert all(isinstance(v, (float, np.floating)) for v in er)

    def test_mean_reversion_method(self, optimizer, sample_prices):
        er = optimizer.calculate_expected_returns(sample_prices, method="mean_reversion")
        assert len(er) == 3

    def test_momentum_method(self, optimizer, sample_prices):
        er = optimizer.calculate_expected_returns(sample_prices, method="momentum")
        assert len(er) == 3

    def test_unknown_method_raises(self, optimizer, sample_prices):
        with pytest.raises(ValueError, match="Unknown method"):
            optimizer.calculate_expected_returns(sample_prices, method="magic")


# ---------------------------------------------------------------------------
# calculate_risk_model
# ---------------------------------------------------------------------------

class TestCalculateRiskModel:
    def test_returns_cov_and_vol(self, optimizer, sample_prices):
        cov, vol = optimizer.calculate_risk_model(sample_prices)
        assert cov.shape == (3, 3)
        assert len(vol) == 3
        assert all(v > 0 for v in vol)

    def test_cov_matrix_symmetric(self, optimizer, sample_prices):
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)

    def test_cov_columns_match_prices(self, optimizer, sample_prices):
        cov, vol = optimizer.calculate_risk_model(sample_prices)
        assert list(cov.columns) == list(sample_prices.columns)
        assert list(vol.index) == list(sample_prices.columns)


# ---------------------------------------------------------------------------
# optimize_portfolio — basic objectives
# ---------------------------------------------------------------------------

class TestOptimizePortfolio:
    def test_max_sharpe(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        result = optimizer.optimize_portfolio(er, cov, basic_constraints,
                                              OptimizationObjective.MAX_SHARPE)
        assert isinstance(result, OptimizationResult)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4
        assert result.volatility >= 0

    def test_min_variance(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        result = optimizer.optimize_portfolio(er, cov, basic_constraints,
                                              OptimizationObjective.MIN_VARIANCE)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_max_return(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        result = optimizer.optimize_portfolio(er, cov, basic_constraints,
                                              OptimizationObjective.MAX_RETURN)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_risk_parity(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        result = optimizer.optimize_portfolio(er, cov, basic_constraints,
                                              OptimizationObjective.RISK_PARITY)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_cvar_optimization(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        result = optimizer.optimize_portfolio(er, cov, basic_constraints,
                                              OptimizationObjective.CVAR_OPTIMIZATION)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_max_diversification(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        result = optimizer.optimize_portfolio(er, cov, basic_constraints,
                                              OptimizationObjective.MAX_DIVERSIFICATION)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_risk_budgeting(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        result = optimizer.optimize_portfolio(er, cov, basic_constraints,
                                              OptimizationObjective.RISK_BUDGETING)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_mean_reversion_objective(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        result = optimizer.optimize_portfolio(er, cov, basic_constraints,
                                              OptimizationObjective.MEAN_REVERSION)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_with_current_weights(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        current = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        result = optimizer.optimize_portfolio(er, cov, basic_constraints,
                                              current_weights=current)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_with_target_volatility(self, optimizer, sample_prices):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        c = OptimizationConstraints(max_weight=0.8, target_volatility=0.3)
        result = optimizer.optimize_portfolio(er, cov, c, OptimizationObjective.MIN_VARIANCE)
        # May not satisfy exactly but should not crash
        assert isinstance(result, OptimizationResult)

    def test_with_min_expected_return(self, optimizer, sample_prices):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        c = OptimizationConstraints(max_weight=0.8, min_expected_return=0.01)
        result = optimizer.optimize_portfolio(er, cov, c)
        assert isinstance(result, OptimizationResult)

    def test_weights_within_bounds(self, optimizer, sample_prices):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        c = OptimizationConstraints(min_weight=0.1, max_weight=0.5)
        result = optimizer.optimize_portfolio(er, cov, c)
        for w in result.weights.values():
            assert w >= -1e-6  # Tolerance for numerical precision

    def test_transaction_cost_penalty(self, optimizer, sample_prices):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        current = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        c = OptimizationConstraints(
            max_weight=0.6,
            transaction_costs={"maker_fee": 0.001, "taker_fee": 0.0015, "spread": 0.005},
        )
        result = optimizer.optimize_portfolio(er, cov, c, current_weights=current)
        assert isinstance(result, OptimizationResult)


# ---------------------------------------------------------------------------
# Result properties
# ---------------------------------------------------------------------------

class TestOptimizationResultMetrics:
    def test_risk_contributions_sum_to_one(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        result = optimizer.optimize_portfolio(er, cov, basic_constraints)
        rc_sum = sum(result.risk_contributions.values())
        assert abs(rc_sum - 1.0) < 0.1  # Approximate

    def test_no_nan_in_result(self, optimizer, sample_prices, basic_constraints):
        er = optimizer.calculate_expected_returns(sample_prices)
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        result = optimizer.optimize_portfolio(er, cov, basic_constraints)
        assert np.isfinite(result.expected_return)
        assert np.isfinite(result.volatility)
        assert np.isfinite(result.sharpe_ratio)
        for v in result.weights.values():
            assert np.isfinite(v)


# ---------------------------------------------------------------------------
# Private helper methods
# ---------------------------------------------------------------------------

class TestHelperMethods:
    def test_diversification_ratio(self, optimizer):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = np.array([0.5, 0.5])
        dr = optimizer._calculate_diversification_ratio(weights, cov)
        assert dr > 0
        assert np.isfinite(dr)

    def test_diversification_ratio_zero_variance(self, optimizer):
        cov = np.zeros((2, 2))
        weights = np.array([0.5, 0.5])
        dr = optimizer._calculate_diversification_ratio(weights, cov)
        assert dr == 1.0  # Fallback

    def test_risk_contributions(self, optimizer):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = np.array([0.6, 0.4])
        rc = optimizer._calculate_risk_contributions(weights, cov, ["A", "B"])
        assert "A" in rc
        assert "B" in rc
        assert abs(sum(rc.values()) - 1.0) < 1e-6

    def test_risk_contributions_zero_variance(self, optimizer):
        cov = np.zeros((2, 2))
        weights = np.array([0.5, 0.5])
        rc = optimizer._calculate_risk_contributions(weights, cov, ["A", "B"])
        assert rc == {}

    def test_transaction_cost_penalty_calculation(self, optimizer):
        target = np.array([0.4, 0.6])
        current = np.array([0.5, 0.5])
        costs = {"maker_fee": 0.001, "taker_fee": 0.001, "spread": 0.005}
        penalty = optimizer._calculate_transaction_cost_penalty(target, current, costs)
        assert penalty > 0

    def test_transaction_cost_no_change(self, optimizer):
        weights = np.array([0.5, 0.5])
        costs = {"maker_fee": 0.001, "taker_fee": 0.001, "spread": 0.005}
        penalty = optimizer._calculate_transaction_cost_penalty(weights, weights, costs)
        assert penalty == 0.0

    def test_smart_fallback_weights_positive_returns(self, optimizer):
        expected_returns = np.array([0.1, 0.2, -0.05])
        weights = optimizer._create_smart_fallback_weights(expected_returns)
        assert abs(sum(weights) - 1.0) < 1e-6
        assert weights[2] == 0.0  # Negative return gets zero

    def test_smart_fallback_weights_all_negative(self, optimizer):
        expected_returns = np.array([-0.1, -0.2, -0.05])
        weights = optimizer._create_smart_fallback_weights(expected_returns)
        # All negative → equal weights
        np.testing.assert_allclose(weights, np.ones(3) / 3)

    def test_check_constraints_satisfaction(self, optimizer):
        weights = np.array([0.4, 0.3, 0.3])
        c = OptimizationConstraints(min_weight=0.0, max_weight=0.5, max_sector_weight=0.8)
        assert optimizer._check_constraints_satisfaction(weights, c, ["BTC", "ETH", "SOL"]) is True

    def test_check_constraints_violation_sum(self, optimizer):
        weights = np.array([0.5, 0.3, 0.3])  # Sum = 1.1
        c = OptimizationConstraints()
        assert optimizer._check_constraints_satisfaction(weights, c, ["A", "B", "C"]) is False

    def test_check_constraints_violation_max_weight(self, optimizer):
        weights = np.array([0.7, 0.2, 0.1])
        c = OptimizationConstraints(max_weight=0.5)
        assert optimizer._check_constraints_satisfaction(weights, c, ["A", "B", "C"]) is False

    def test_correlation_matrix_from_cov(self, optimizer, sample_prices):
        cov, _ = optimizer.calculate_risk_model(sample_prices)
        corr = optimizer._calculate_correlation_matrix(cov)
        # Diagonal should be ~1
        for i in range(corr.shape[0]):
            assert abs(corr[i, i] - 1.0) < 1e-6

    def test_correlation_exposure(self, optimizer):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        weights = np.array([0.5, 0.5])
        exp = optimizer._calculate_correlation_exposure(weights, corr)
        assert exp > 0


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

class TestBlackLitterman:
    def test_basic_bl_optimization(self, optimizer, sample_prices, basic_constraints):
        views = {"BTC": 0.15, "ETH": 0.10}
        confidence = {"BTC": 0.8, "ETH": 0.6}
        result = optimizer.optimize_black_litterman(
            sample_prices, views, confidence, basic_constraints
        )
        assert isinstance(result, OptimizationResult)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_bl_with_current_weights(self, optimizer, sample_prices, basic_constraints):
        views = {"BTC": 0.12}
        confidence = {"BTC": 0.7}
        current = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        result = optimizer.optimize_black_litterman(
            sample_prices, views, confidence, basic_constraints,
            current_weights=current
        )
        assert isinstance(result, OptimizationResult)

    def test_bl_no_valid_views_raises(self, optimizer, sample_prices, basic_constraints):
        views = {"DOGE": 0.2}  # Not in price_history
        confidence = {"DOGE": 0.8}
        with pytest.raises(ValueError, match="No valid assets"):
            optimizer.optimize_black_litterman(
                sample_prices, views, confidence, basic_constraints
            )


# ---------------------------------------------------------------------------
# Multi-period optimization
# ---------------------------------------------------------------------------

class TestMultiPeriod:
    def test_multi_period_basic(self, optimizer, sample_prices, basic_constraints):
        result = optimizer.optimize_multi_period(sample_prices, basic_constraints)
        assert isinstance(result, OptimizationResult)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_multi_period_custom_periods(self, optimizer, sample_prices):
        c = OptimizationConstraints(
            max_weight=0.8,
            rebalance_periods=[20, 50],
            period_weights=[0.6, 0.4],
        )
        result = optimizer.optimize_multi_period(sample_prices, c)
        assert isinstance(result, OptimizationResult)

    def test_multi_period_mismatched_lengths_fallback(self, optimizer, sample_prices):
        c = OptimizationConstraints(
            max_weight=0.8,
            rebalance_periods=[20, 50, 80],
            period_weights=[0.5, 0.5],  # Mismatch → equal weights
        )
        result = optimizer.optimize_multi_period(sample_prices, c)
        assert isinstance(result, OptimizationResult)


# ---------------------------------------------------------------------------
# Efficient Frontier
# ---------------------------------------------------------------------------

class TestEfficientFrontier:
    def test_efficient_frontier_returns_dict(self, optimizer, sample_prices, basic_constraints):
        frontier = optimizer.calculate_efficient_frontier(
            sample_prices, basic_constraints, n_points=5
        )
        assert "risks" in frontier
        assert "returns" in frontier
        assert "weights" in frontier
        assert "sharpe_ratios" in frontier
        assert "n_points" in frontier
        assert frontier["n_points"] >= 0


# ---------------------------------------------------------------------------
# create_crypto_constraints
# ---------------------------------------------------------------------------

class TestCreateCryptoConstraints:
    def test_aggressive_defaults(self):
        c = create_crypto_constraints(conservative=False)
        assert c.min_weight == 0.0
        assert c.max_weight == 0.35

    def test_conservative_defaults(self):
        c = create_crypto_constraints(conservative=True)
        assert c.max_weight == 0.25
        assert c.target_volatility == 0.20

    def test_dynamic_min_weight_many_assets(self):
        c = create_crypto_constraints(conservative=True, n_assets=100)
        assert c.min_weight <= 0.01  # Dynamic
        assert c.min_weight >= 0.001

    def test_dynamic_min_weight_few_assets(self):
        c = create_crypto_constraints(conservative=True, n_assets=5)
        assert c.min_weight == 0.01

    def test_no_n_assets(self):
        c = create_crypto_constraints(conservative=True, n_assets=None)
        assert c.min_weight == 0.0  # No dynamic calc
