"""
Unit tests for services/performance_optimizer.py

Tests cache key generation, portfolio metrics, expected returns,
correlation matrix, covariance matrix, and batch preprocessing.
"""
import pytest
import numpy as np
import pandas as pd
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from services.performance_optimizer import PortfolioPerformanceOptimizer


@pytest.fixture
def optimizer(tmp_path):
    return PortfolioPerformanceOptimizer(cache_dir=str(tmp_path / "cache"))


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    n_days = 252
    n_assets = 5
    data = np.random.randn(n_days, n_assets) * 0.02
    cols = ["BTC", "ETH", "SOL", "ADA", "DOT"]
    idx = pd.date_range("2025-01-01", periods=n_days, freq="B")
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def sample_prices():
    np.random.seed(42)
    n_days = 300
    n_assets = 5
    cols = ["BTC", "ETH", "SOL", "ADA", "DOT"]
    idx = pd.date_range("2025-01-01", periods=n_days, freq="B")
    prices = pd.DataFrame(
        100 * np.cumprod(1 + np.random.randn(n_days, n_assets) * 0.02, axis=0),
        index=idx, columns=cols
    )
    return prices


class TestGetCacheKey:

    def test_dict_input(self, optimizer):
        key = optimizer.get_cache_key({"a": 1, "b": 2}, prefix="test")
        assert key.startswith("test_")
        assert len(key) > 5

    def test_string_input(self, optimizer):
        key = optimizer.get_cache_key("hello", prefix="pfx")
        assert key.startswith("pfx_")

    def test_dataframe_input(self, optimizer, sample_returns):
        key = optimizer.get_cache_key(sample_returns, prefix="df")
        assert key.startswith("df_")

    def test_empty_dataframe(self, optimizer):
        df = pd.DataFrame()
        key = optimizer.get_cache_key(df, prefix="empty")
        assert key.startswith("empty_")

    def test_deterministic(self, optimizer):
        d = {"x": 10, "y": 20}
        k1 = optimizer.get_cache_key(d, prefix="t")
        k2 = optimizer.get_cache_key(d, prefix="t")
        assert k1 == k2

    def test_different_data_different_keys(self, optimizer):
        k1 = optimizer.get_cache_key({"a": 1})
        k2 = optimizer.get_cache_key({"a": 2})
        assert k1 != k2

    def test_no_prefix(self, optimizer):
        key = optimizer.get_cache_key("data")
        assert key.startswith("_")


class TestEfficientPortfolioMetrics:

    def test_equal_weights(self, optimizer):
        n = 4
        weights = np.ones(n) / n
        expected_returns = np.array([0.10, 0.15, 0.08, 0.12])
        cov = np.eye(n) * 0.04
        result = optimizer.efficient_portfolio_metrics(weights, expected_returns, cov)
        assert "expected_return" in result
        assert "volatility" in result
        assert "variance" in result
        assert "diversification_ratio" in result
        assert "risk_contributions" in result
        assert "concentration" in result

    def test_expected_return_calculation(self, optimizer):
        weights = np.array([0.5, 0.5])
        exp_ret = np.array([0.10, 0.20])
        cov = np.eye(2) * 0.01
        result = optimizer.efficient_portfolio_metrics(weights, exp_ret, cov)
        assert result["expected_return"] == pytest.approx(0.15)

    def test_single_asset_portfolio(self, optimizer):
        weights = np.array([1.0])
        exp_ret = np.array([0.12])
        cov = np.array([[0.04]])
        result = optimizer.efficient_portfolio_metrics(weights, exp_ret, cov)
        assert result["expected_return"] == pytest.approx(0.12)
        assert result["volatility"] == pytest.approx(0.2)
        assert result["concentration"] == pytest.approx(1.0)

    def test_zero_volatility_handling(self, optimizer):
        weights = np.array([0.5, 0.5])
        exp_ret = np.array([0.10, 0.10])
        cov = np.zeros((2, 2))
        result = optimizer.efficient_portfolio_metrics(weights, exp_ret, cov)
        assert result["volatility"] >= 0

    def test_concentration_index(self, optimizer):
        weights = np.array([1.0, 0.0, 0.0])
        exp_ret = np.array([0.1, 0.1, 0.1])
        cov = np.eye(3) * 0.01
        result = optimizer.efficient_portfolio_metrics(weights, exp_ret, cov)
        assert result["concentration"] == pytest.approx(1.0)

    def test_diversified_concentration(self, optimizer):
        n = 10
        weights = np.ones(n) / n
        exp_ret = np.ones(n) * 0.1
        cov = np.eye(n) * 0.01
        result = optimizer.efficient_portfolio_metrics(weights, exp_ret, cov)
        assert result["concentration"] == pytest.approx(0.1)

    def test_risk_contributions_sum(self, optimizer):
        n = 3
        weights = np.array([0.4, 0.35, 0.25])
        exp_ret = np.array([0.1, 0.12, 0.08])
        cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.03, 0.008], [0.005, 0.008, 0.02]])
        result = optimizer.efficient_portfolio_metrics(weights, exp_ret, cov)
        assert sum(result["risk_contributions"]) == pytest.approx(1.0, abs=0.01)

    def test_diversification_ratio_greater_one(self, optimizer):
        weights = np.array([0.5, 0.5])
        exp_ret = np.array([0.1, 0.15])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = optimizer.efficient_portfolio_metrics(weights, exp_ret, cov)
        assert result["diversification_ratio"] >= 1.0


class TestCalculateExpectedReturns:

    def test_robust_mean_method(self, optimizer, sample_returns):
        result = optimizer.calculate_expected_returns(sample_returns, method="robust_mean")
        assert isinstance(result, np.ndarray)
        assert len(result) == sample_returns.shape[1]

    def test_shrinkage_method(self, optimizer, sample_returns):
        result = optimizer.calculate_expected_returns(sample_returns, method="shrinkage")
        assert isinstance(result, np.ndarray)
        assert len(result) == sample_returns.shape[1]

    def test_simple_mean_method(self, optimizer, sample_returns):
        result = optimizer.calculate_expected_returns(sample_returns, method="simple")
        assert isinstance(result, np.ndarray)
        assert len(result) == sample_returns.shape[1]

    def test_shrinkage_pulls_toward_mean(self, optimizer):
        np.random.seed(99)
        n = 252
        data = pd.DataFrame({
            "A": np.random.randn(n) * 0.01 + 0.001,
            "B": np.random.randn(n) * 0.01 - 0.001,
        })
        simple = optimizer.calculate_expected_returns(data, method="simple")
        shrunk = optimizer.calculate_expected_returns(data, method="shrinkage")
        spread_simple = abs(simple[0] - simple[1])
        spread_shrunk = abs(shrunk[0] - shrunk[1])
        assert spread_shrunk <= spread_simple

    def test_robust_mean_handles_outliers(self, optimizer):
        np.random.seed(10)
        n = 252
        data = pd.DataFrame({"X": np.random.randn(n) * 0.01})
        data.iloc[0] = 10.0
        robust = optimizer.calculate_expected_returns(data, method="robust_mean")
        simple = optimizer.calculate_expected_returns(data, method="simple")
        assert abs(robust[0]) < abs(simple[0])

    def test_returns_annualized(self, optimizer):
        n = 252
        daily_ret = 0.001
        data = pd.DataFrame({"A": [daily_ret] * n})
        result = optimizer.calculate_expected_returns(data, method="simple")
        assert result[0] == pytest.approx(daily_ret * 252)


class TestFastCorrelationMatrix:

    def test_diagonal_is_one(self, optimizer):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        corr = optimizer.fast_correlation_matrix(cov)
        assert corr[0, 0] == pytest.approx(1.0)
        assert corr[1, 1] == pytest.approx(1.0)

    def test_off_diagonal_range(self, optimizer):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        corr = optimizer.fast_correlation_matrix(cov)
        assert -1.0 <= corr[0, 1] <= 1.0

    def test_symmetric(self, optimizer):
        cov = np.array([[0.04, 0.02, 0.01], [0.02, 0.09, 0.03], [0.01, 0.03, 0.16]])
        corr = optimizer.fast_correlation_matrix(cov)
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_identity_cov_gives_identity_corr(self, optimizer):
        cov = np.eye(3) * 0.05
        corr = optimizer.fast_correlation_matrix(cov)
        np.testing.assert_array_almost_equal(corr, np.eye(3))

    def test_perfect_correlation(self, optimizer):
        cov = np.array([[0.04, 0.06], [0.06, 0.09]])
        corr = optimizer.fast_correlation_matrix(cov)
        assert corr[0, 1] == pytest.approx(1.0)

    def test_zero_variance_handled(self, optimizer):
        cov = np.array([[0.0, 0.0], [0.0, 0.04]])
        corr = optimizer.fast_correlation_matrix(cov)
        assert np.isfinite(corr).all()


class TestOptimizedCovarianceMatrix:

    def test_returns_ndarray(self, optimizer, sample_returns):
        cov = optimizer.optimized_covariance_matrix(sample_returns)
        assert isinstance(cov, np.ndarray)

    def test_shape_matches_assets(self, optimizer, sample_returns):
        n = sample_returns.shape[1]
        cov = optimizer.optimized_covariance_matrix(sample_returns)
        assert cov.shape == (n, n)

    def test_symmetric(self, optimizer, sample_returns):
        cov = optimizer.optimized_covariance_matrix(sample_returns)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_positive_diagonal(self, optimizer, sample_returns):
        cov = optimizer.optimized_covariance_matrix(sample_returns)
        assert (np.diag(cov) > 0).all()

    def test_no_exponential_weight(self, optimizer, sample_returns):
        cov = optimizer.optimized_covariance_matrix(sample_returns, exponential_weight=False)
        assert isinstance(cov, np.ndarray)
        assert cov.shape[0] == sample_returns.shape[1]

    def test_zero_shrinkage(self, optimizer, sample_returns):
        cov = optimizer.optimized_covariance_matrix(sample_returns, shrinkage=0.0)
        assert isinstance(cov, np.ndarray)

    def test_high_shrinkage_toward_identity(self, optimizer, sample_returns):
        cov_high = optimizer.optimized_covariance_matrix(sample_returns, shrinkage=0.99)
        off_diag = cov_high[0, 1]
        diag = cov_high[0, 0]
        assert abs(off_diag / diag) < 0.1


class TestBatchOptimizationPreprocessing:

    def test_basic_output_keys(self, optimizer, sample_prices):
        result = optimizer.batch_optimization_preprocessing(sample_prices)
        assert "price_data" in result
        assert "returns" in result
        assert "expected_returns" in result
        assert "cov_matrix" in result
        assert "corr_matrix" in result
        assert "n_assets" in result
        assert "assets" in result

    def test_n_assets_correct(self, optimizer, sample_prices):
        result = optimizer.batch_optimization_preprocessing(sample_prices)
        assert result["n_assets"] == sample_prices.shape[1]

    def test_returns_shape(self, optimizer, sample_prices):
        result = optimizer.batch_optimization_preprocessing(sample_prices)
        assert result["returns"].shape[1] == sample_prices.shape[1]

    def test_expected_returns_length(self, optimizer, sample_prices):
        result = optimizer.batch_optimization_preprocessing(sample_prices)
        assert len(result["expected_returns"]) == sample_prices.shape[1]

    def test_cov_matrix_shape(self, optimizer, sample_prices):
        result = optimizer.batch_optimization_preprocessing(sample_prices)
        n = sample_prices.shape[1]
        assert result["cov_matrix"].shape == (n, n)

    def test_assets_list(self, optimizer, sample_prices):
        result = optimizer.batch_optimization_preprocessing(sample_prices)
        assert result["assets"] == list(sample_prices.columns)


class TestCacheMatrixOperation:

    def test_caches_in_memory(self, optimizer):
        def compute():
            return np.array([1, 2, 3])
        result1 = optimizer.cache_matrix_operation("test_key", compute)
        result2 = optimizer.cache_matrix_operation("test_key", compute)
        np.testing.assert_array_equal(result1, result2)
        assert "test_key" in optimizer.memory_cache

    def test_caches_to_disk(self, optimizer):
        def compute():
            return np.array([[1, 2], [3, 4]])
        optimizer.cache_matrix_operation("disk_test", compute)
        cache_file = optimizer.cache_dir / "disk_test.json"
        assert cache_file.exists()

    def test_eviction_when_full(self, optimizer):
        optimizer.max_cache_size = 3
        for i in range(5):
            optimizer.cache_matrix_operation(f"key_{i}", lambda: np.array([i]))
        assert len(optimizer.memory_cache) <= 3


class TestClearCache:

    def test_clears_memory(self, optimizer):
        optimizer.memory_cache["test"] = np.array([1])
        optimizer.clear_cache(older_than_days=0)
        assert len(optimizer.memory_cache) == 0

    def test_init_creates_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "new_cache"
        opt = PortfolioPerformanceOptimizer(cache_dir=str(cache_dir))
        assert cache_dir.exists()
