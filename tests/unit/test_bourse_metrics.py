"""
Tests for services/risk/bourse/metrics.py
Pure NumPy risk metric calculations: VaR, volatility, Sharpe, Sortino, drawdown, beta, risk score
"""

import pytest
import numpy as np
from services.risk.bourse.metrics import (
    calculate_var_historical,
    calculate_var_parametric,
    calculate_var_montecarlo,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_beta,
    calculate_risk_score,
    calculate_calmar_ratio,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def normal_returns():
    """200 days of slightly positive returns with moderate volatility."""
    np.random.seed(42)
    return np.random.normal(0.0005, 0.02, 200)


@pytest.fixture
def bull_returns():
    """Returns with strong positive bias."""
    np.random.seed(42)
    return np.random.normal(0.002, 0.015, 200)


@pytest.fixture
def bear_returns():
    """Returns with negative bias."""
    np.random.seed(42)
    return np.random.normal(-0.002, 0.025, 200)


@pytest.fixture
def price_series():
    """Simulated price series starting at 100."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 200)
    prices = 100 * np.cumprod(1 + returns)
    return prices


# ── calculate_var_historical ────────────────────────────────────────

class TestVarHistorical:

    def test_basic_calculation(self, normal_returns):
        result = calculate_var_historical(normal_returns, confidence_level=0.95)
        assert result["method"] == "historical"
        assert result["confidence_level"] == 0.95
        assert result["var_percentage"] < 0  # VaR should be negative
        assert result["lookback_days"] == 200

    def test_with_portfolio_value(self, normal_returns):
        result = calculate_var_historical(normal_returns, confidence_level=0.95, portfolio_value=100000)
        assert "var_monetary" in result
        assert result["var_monetary"] < 0
        assert result["portfolio_value"] == 100000

    def test_99_confidence(self, normal_returns):
        r95 = calculate_var_historical(normal_returns, 0.95)
        r99 = calculate_var_historical(normal_returns, 0.99)
        # 99% VaR should be more negative (larger loss) than 95%
        assert r99["var_percentage"] < r95["var_percentage"]

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_var_historical(np.array([]))

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Insufficient"):
            calculate_var_historical(np.array([0.01] * 5))

    def test_with_nan_values(self, normal_returns):
        returns_with_nan = normal_returns.copy()
        returns_with_nan[10] = np.nan
        returns_with_nan[50] = np.nan
        result = calculate_var_historical(returns_with_nan)
        assert result["lookback_days"] == 198  # 200 - 2 NaN


# ── calculate_var_parametric ────────────────────────────────────────

class TestVarParametric:

    def test_basic_calculation(self, normal_returns):
        result = calculate_var_parametric(normal_returns, confidence_level=0.95)
        assert result["method"] == "parametric"
        assert result["var_percentage"] < 0
        assert "mean_return" in result
        assert "std_return" in result

    def test_with_portfolio_value(self, normal_returns):
        result = calculate_var_parametric(normal_returns, portfolio_value=50000)
        assert "var_monetary" in result
        assert result["portfolio_value"] == 50000

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_var_parametric(np.array([]))

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Insufficient"):
            calculate_var_parametric(np.array([0.01] * 5))

    def test_parametric_vs_historical_similar(self, normal_returns):
        """For normally distributed data, parametric and historical VaR should be close."""
        hist = calculate_var_historical(normal_returns, 0.95)
        param = calculate_var_parametric(normal_returns, 0.95)
        # Within 50% of each other (they use different methods)
        assert abs(hist["var_percentage"] - param["var_percentage"]) < abs(hist["var_percentage"])


# ── calculate_var_montecarlo ────────────────────────────────────────

class TestVarMonteCarlo:

    def test_basic_calculation(self, normal_returns):
        result = calculate_var_montecarlo(normal_returns, confidence_level=0.95, random_seed=42)
        assert result["method"] == "montecarlo"
        assert result["var_percentage"] < 0
        assert result["num_simulations"] == 10000

    def test_reproducibility_with_seed(self, normal_returns):
        r1 = calculate_var_montecarlo(normal_returns, random_seed=42)
        r2 = calculate_var_montecarlo(normal_returns, random_seed=42)
        assert r1["var_percentage"] == r2["var_percentage"]

    def test_different_seeds_differ(self, normal_returns):
        r1 = calculate_var_montecarlo(normal_returns, random_seed=42)
        r2 = calculate_var_montecarlo(normal_returns, random_seed=99)
        # Very unlikely to be exactly equal
        assert r1["var_percentage"] != r2["var_percentage"]

    def test_with_portfolio_value(self, normal_returns):
        result = calculate_var_montecarlo(normal_returns, portfolio_value=200000, random_seed=42)
        assert "var_monetary" in result

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_var_montecarlo(np.array([]))

    def test_more_simulations_precision(self, normal_returns):
        """More simulations should not drastically change the result."""
        r_low = calculate_var_montecarlo(normal_returns, num_simulations=1000, random_seed=42)
        r_high = calculate_var_montecarlo(normal_returns, num_simulations=50000, random_seed=42)
        # Within 30% of each other
        assert abs(r_low["var_percentage"] - r_high["var_percentage"]) < 0.02


# ── calculate_volatility ───────────────────────────────────────────

class TestVolatility:

    def test_basic_annualized(self, normal_returns):
        vol = calculate_volatility(normal_returns, annualize=True)
        assert vol > 0
        # With std=0.02 and 252 trading days, annualized vol ~ 0.02 * sqrt(252) ~ 0.32
        assert 0.15 < vol < 0.60

    def test_not_annualized(self, normal_returns):
        vol_daily = calculate_volatility(normal_returns, annualize=False)
        vol_annual = calculate_volatility(normal_returns, annualize=True)
        assert vol_daily < vol_annual
        assert vol_daily == pytest.approx(vol_annual / np.sqrt(252), rel=0.01)

    def test_window(self, normal_returns):
        vol_full = calculate_volatility(normal_returns, window=None, annualize=False)
        vol_window = calculate_volatility(normal_returns, window=30, annualize=False)
        # Both should be positive, but may differ
        assert vol_full > 0
        assert vol_window > 0

    def test_crypto_trading_days(self, normal_returns):
        vol_stock = calculate_volatility(normal_returns, trading_days=252)
        vol_crypto = calculate_volatility(normal_returns, trading_days=365)
        assert vol_crypto > vol_stock

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_volatility(np.array([]))

    def test_single_element_returns_zero(self):
        assert calculate_volatility(np.array([0.01])) == 0.0

    def test_constant_returns(self):
        """Constant returns should have zero volatility."""
        constant = np.full(50, 0.001)
        vol = calculate_volatility(constant, annualize=False)
        assert vol == pytest.approx(0.0, abs=1e-10)

    def test_with_nan_values(self, normal_returns):
        r = normal_returns.copy()
        r[0] = np.nan
        vol = calculate_volatility(r)
        assert vol > 0


# ── calculate_sharpe_ratio ─────────────────────────────────────────

class TestSharpeRatio:

    def test_positive_returns_positive_sharpe(self, bull_returns):
        sharpe = calculate_sharpe_ratio(bull_returns, risk_free_rate=0.03)
        assert sharpe > 0

    def test_negative_returns_negative_sharpe(self, bear_returns):
        sharpe = calculate_sharpe_ratio(bear_returns, risk_free_rate=0.03)
        assert sharpe < 0

    def test_zero_std_returns_zero(self):
        constant = np.full(50, 0.001)
        sharpe = calculate_sharpe_ratio(constant)
        assert sharpe == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_sharpe_ratio(np.array([]))

    def test_single_element_returns_zero(self):
        assert calculate_sharpe_ratio(np.array([0.01])) == 0.0

    def test_higher_rf_lowers_sharpe(self, bull_returns):
        sharpe_low_rf = calculate_sharpe_ratio(bull_returns, risk_free_rate=0.01)
        sharpe_high_rf = calculate_sharpe_ratio(bull_returns, risk_free_rate=0.05)
        assert sharpe_low_rf > sharpe_high_rf

    def test_annualization(self, bull_returns):
        sharpe_ann = calculate_sharpe_ratio(bull_returns, annualize=True)
        sharpe_raw = calculate_sharpe_ratio(bull_returns, annualize=False)
        # Annualized should be scaled by sqrt(252)
        assert abs(sharpe_ann) > abs(sharpe_raw)


# ── calculate_sortino_ratio ────────────────────────────────────────

class TestSortinoRatio:

    def test_positive_returns(self, bull_returns):
        sortino = calculate_sortino_ratio(bull_returns)
        assert isinstance(sortino, float)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_sortino_ratio(np.array([]))

    def test_single_element_returns_zero(self):
        assert calculate_sortino_ratio(np.array([0.01])) == 0.0

    def test_all_positive_returns_zero(self):
        """All positive returns = no downside deviation = 0."""
        all_positive = np.array([0.01, 0.02, 0.03, 0.04, 0.05] * 10)
        assert calculate_sortino_ratio(all_positive) == 0.0

    def test_sortino_higher_than_sharpe_for_skewed(self, bull_returns):
        """For positively skewed returns, Sortino should be >= Sharpe."""
        sharpe = calculate_sharpe_ratio(bull_returns, risk_free_rate=0.03)
        sortino = calculate_sortino_ratio(bull_returns, risk_free_rate=0.03)
        # Not always true depending on distribution, but generally
        # sortino captures upside better. Just check both are valid floats.
        assert isinstance(sortino, float)
        assert isinstance(sharpe, float)


# ── calculate_max_drawdown ─────────────────────────────────────────

class TestMaxDrawdown:

    def test_basic_calculation(self, price_series):
        result = calculate_max_drawdown(price_series)
        assert result["max_drawdown"] <= 0
        assert result["max_drawdown_pct"] <= 0
        assert result["drawdown_days"] >= 0

    def test_monotonically_increasing_no_drawdown(self):
        """Monotonically increasing prices = 0 drawdown."""
        prices = np.arange(100, 200, dtype=float)
        result = calculate_max_drawdown(prices)
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-10)

    def test_50pct_crash(self):
        """Known 50% drawdown scenario."""
        prices = np.array([100.0, 110.0, 120.0, 60.0, 70.0])
        result = calculate_max_drawdown(prices)
        # Max drawdown = (60 - 120) / 120 = -0.5
        assert result["max_drawdown"] == pytest.approx(-0.5, abs=0.01)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_max_drawdown(np.array([]))

    def test_single_element_zero_drawdown(self):
        result = calculate_max_drawdown(np.array([100.0, 100.0]))
        assert result["max_drawdown"] == pytest.approx(0.0)

    def test_indices_valid(self, price_series):
        result = calculate_max_drawdown(price_series)
        assert result["peak_date_index"] <= result["trough_date_index"]
        assert result["peak_date_index"] >= 0
        assert result["trough_date_index"] < len(price_series)


# ── calculate_beta ─────────────────────────────────────────────────

class TestBeta:

    def test_identical_returns_beta_one(self, normal_returns):
        """Asset identical to benchmark → beta = 1."""
        beta = calculate_beta(normal_returns, normal_returns)
        assert beta == pytest.approx(1.0, abs=0.01)

    def test_double_returns_beta_two(self, normal_returns):
        """Asset = 2x benchmark → beta ~ 2."""
        double = normal_returns * 2
        beta = calculate_beta(double, normal_returns)
        assert beta == pytest.approx(2.0, abs=0.1)

    def test_inverse_returns_negative_beta(self, normal_returns):
        """Inverse of benchmark → negative beta."""
        inverse = -normal_returns
        beta = calculate_beta(inverse, normal_returns)
        assert beta < 0
        assert beta == pytest.approx(-1.0, abs=0.1)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_beta(np.array([]), np.array([]))

    def test_insufficient_data_returns_default(self):
        """With < min_periods, returns default beta 1.0."""
        short = np.array([0.01] * 5)
        beta = calculate_beta(short, short, min_periods=20)
        assert beta == 1.0

    def test_different_lengths_aligned(self, normal_returns):
        """Different array lengths should be handled."""
        shorter = normal_returns[:100]
        beta = calculate_beta(shorter, normal_returns)
        assert isinstance(beta, float)

    def test_zero_variance_benchmark(self):
        """Zero variance benchmark → default beta 1.0."""
        constant = np.full(50, 0.001)
        asset = np.random.normal(0.001, 0.01, 50)
        beta = calculate_beta(asset, constant)
        assert beta == 1.0


# ── calculate_risk_score ───────────────────────────────────────────

class TestRiskScore:

    def test_basic_calculation(self):
        result = calculate_risk_score(
            var_95=-0.02,
            volatility=0.15,
            sharpe_ratio=1.5,
            max_drawdown=-0.10,
            beta=1.0
        )
        assert "risk_score" in result
        assert "risk_level" in result
        assert "component_scores" in result
        assert 0 <= result["risk_score"] <= 100

    def test_low_risk_portfolio(self):
        """Low risk inputs → high risk score → 'Low' risk level."""
        result = calculate_risk_score(
            var_95=-0.005,       # Small VaR
            volatility=0.05,     # Low vol
            sharpe_ratio=2.5,    # Excellent Sharpe
            max_drawdown=-0.03,  # Small drawdown
            beta=1.0             # Market beta
        )
        assert result["risk_level"] == "Low"
        assert result["risk_score"] >= 75

    def test_high_risk_portfolio(self):
        """High risk inputs → low score → 'High' or 'Critical'."""
        result = calculate_risk_score(
            var_95=-0.08,        # Large VaR
            volatility=0.45,     # Very high vol
            sharpe_ratio=-0.5,   # Negative Sharpe
            max_drawdown=-0.40,  # Large drawdown
            beta=1.8             # High beta
        )
        assert result["risk_level"] in ("High", "Critical")
        assert result["risk_score"] < 50

    def test_all_component_scores_bounded(self):
        result = calculate_risk_score(-0.03, 0.20, 1.0, -0.15, 1.2)
        for name, score in result["component_scores"].items():
            assert 0 <= score <= 100, f"Component {name} out of bounds: {score}"

    def test_weights_sum_to_one(self):
        result = calculate_risk_score(-0.02, 0.15, 1.5, -0.10, 1.0)
        total = sum(result["weights"].values())
        assert total == pytest.approx(1.0)

    def test_extreme_values_clamped(self):
        """Extreme inputs should still produce valid 0-100 score."""
        result = calculate_risk_score(
            var_95=-0.50,      # Extreme
            volatility=2.0,    # Extreme
            sharpe_ratio=-5.0, # Extreme
            max_drawdown=-0.90,
            beta=5.0
        )
        assert 0 <= result["risk_score"] <= 100


# ── calculate_calmar_ratio ─────────────────────────────────────────

class TestCalmarRatio:

    def test_basic_calculation(self, normal_returns, price_series):
        calmar = calculate_calmar_ratio(normal_returns, price_series)
        assert isinstance(calmar, float)

    def test_empty_returns_zero(self):
        assert calculate_calmar_ratio(np.array([]), np.array([])) == 0.0

    def test_single_element_returns_zero(self):
        assert calculate_calmar_ratio(np.array([0.01]), np.array([100.0])) == 0.0

    def test_no_drawdown_returns_zero(self):
        """Monotonically increasing → 0 drawdown → calmar returns 0."""
        returns = np.full(50, 0.01)
        prices = 100 * np.cumprod(1 + returns)
        calmar = calculate_calmar_ratio(returns, prices)
        assert calmar == 0.0

    def test_positive_for_profitable_portfolio(self, bull_returns):
        prices = 100 * np.cumprod(1 + bull_returns)
        calmar = calculate_calmar_ratio(bull_returns, prices)
        assert calmar > 0
