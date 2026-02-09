"""
Comprehensive tests for services.risk.bourse.advanced_analytics

Tests all public methods of AdvancedRiskAnalytics:
- calculate_position_var (position-level VaR decomposition)
- calculate_correlation_matrix (correlation + clustering)
- run_stress_test (predefined + custom scenarios)
- analyze_fx_exposure (currency exposure analysis)
- _find_max_correlation_pair / _find_min_correlation_pair (helpers)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from services.risk.bourse.advanced_analytics import AdvancedRiskAnalytics


# ==================== Fixtures ====================

@pytest.fixture
def analytics():
    return AdvancedRiskAnalytics()


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    return {
        "AAPL": pd.Series(np.random.normal(0.001, 0.02, 100), index=dates),
        "MSFT": pd.Series(np.random.normal(0.0008, 0.018, 100), index=dates),
        "GOOGL": pd.Series(np.random.normal(0.0005, 0.022, 100), index=dates),
    }


@pytest.fixture
def sample_weights():
    return {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}


@pytest.fixture
def sample_positions_data():
    return {
        "AAPL": {"current_price": 180.0, "quantity": 50},
        "MSFT": {"current_price": 380.0, "quantity": 30},
        "GOOGL": {"current_price": 140.0, "quantity": 40},
    }


@pytest.fixture
def sample_fx_positions():
    return [
        {"ticker": "AAPL", "currency": "USD", "market_value_usd": 9000.0},
        {"ticker": "MSFT", "currency": "USD", "market_value_usd": 11400.0},
        {"ticker": "NESN", "currency": "CHF", "market_value_usd": 5000.0},
        {"ticker": "SAP", "currency": "EUR", "market_value_usd": 3000.0},
        {"ticker": "NOVN", "currency": "CHF", "market_value_usd": 4000.0},
    ]


@pytest.fixture
def correlated_returns():
    np.random.seed(99)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    base = np.random.normal(0.001, 0.02, n)
    return {
        "A": pd.Series(base + np.random.normal(0, 0.002, n), index=dates),
        "B": pd.Series(base + np.random.normal(0, 0.003, n), index=dates),
        "C": pd.Series(-base + np.random.normal(0, 0.005, n), index=dates),
    }

# ==================== Position VaR Tests ====================

class TestPositionVaR:

    def test_basic_var_structure(self, analytics, sample_returns, sample_weights):
        result = analytics.calculate_position_var(sample_returns, sample_weights)
        expected_keys = {
            "position_var", "marginal_var", "component_var",
            "total_portfolio_var", "diversification_benefit",
            "sum_components", "method", "confidence_level",
            "num_positions", "observation_days",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_var_is_negative(self, analytics, sample_returns, sample_weights):
        result = analytics.calculate_position_var(sample_returns, sample_weights)
        assert result["total_portfolio_var"] < 0

    def test_var_tickers_present(self, analytics, sample_returns, sample_weights):
        result = analytics.calculate_position_var(sample_returns, sample_weights)
        for key in ("position_var", "marginal_var", "component_var"):
            assert set(result[key].keys()) == {"AAPL", "MSFT", "GOOGL"}

    def test_var_num_positions(self, analytics, sample_returns, sample_weights):
        result = analytics.calculate_position_var(sample_returns, sample_weights)
        assert result["num_positions"] == 3

    def test_var_observation_days(self, analytics, sample_returns, sample_weights):
        result = analytics.calculate_position_var(sample_returns, sample_weights)
        assert result["observation_days"] == 100

    def test_diversification_benefit_non_negative(self, analytics, sample_returns, sample_weights):
        result = analytics.calculate_position_var(sample_returns, sample_weights)
        assert result["diversification_benefit"] >= -1e-10

    def test_higher_confidence_gives_larger_var(self, analytics, sample_returns, sample_weights):
        var_95 = analytics.calculate_position_var(sample_returns, sample_weights, confidence_level=0.95)
        var_99 = analytics.calculate_position_var(sample_returns, sample_weights, confidence_level=0.99)
        assert var_99["total_portfolio_var"] <= var_95["total_portfolio_var"]

    def test_method_stored(self, analytics, sample_returns, sample_weights):
        result = analytics.calculate_position_var(
            sample_returns, sample_weights, method="parametric"
        )
        assert result["method"] == "parametric"

    def test_confidence_level_stored(self, analytics, sample_returns, sample_weights):
        result = analytics.calculate_position_var(
            sample_returns, sample_weights, confidence_level=0.99
        )
        assert result["confidence_level"] == 0.99

    def test_insufficient_data_raises(self, analytics, sample_weights):
        dates = pd.date_range("2025-01-01", periods=10, freq="B")
        short_returns = {
            "AAPL": pd.Series(np.random.normal(0, 0.02, 10), index=dates),
            "MSFT": pd.Series(np.random.normal(0, 0.02, 10), index=dates),
            "GOOGL": pd.Series(np.random.normal(0, 0.02, 10), index=dates),
        }
        with pytest.raises(ValueError, match="Insufficient data"):
            analytics.calculate_position_var(short_returns, sample_weights)

    def test_single_position_no_diversification(self, analytics):
        np.random.seed(7)
        dates = pd.date_range("2025-01-01", periods=60, freq="B")
        returns = {"AAPL": pd.Series(np.random.normal(0, 0.02, 60), index=dates)}
        weights = {"AAPL": 1.0}
        result = analytics.calculate_position_var(returns, weights)
        assert abs(result["diversification_benefit"]) < 1e-10

    def test_values_are_float(self, analytics, sample_returns, sample_weights):
        result = analytics.calculate_position_var(sample_returns, sample_weights)
        assert isinstance(result["total_portfolio_var"], float)
        assert isinstance(result["diversification_benefit"], float)
        for v in result["position_var"].values():
            assert isinstance(v, float)

# ==================== Correlation Matrix Tests ====================

class TestCorrelationMatrix:

    def test_basic_structure(self, analytics, sample_returns):
        result = analytics.calculate_correlation_matrix(sample_returns)
        expected_keys = {
            "correlation_matrix", "avg_correlation", "max_correlation",
            "min_correlation", "max_pair", "min_pair", "clustering",
            "method", "num_positions", "timestamp",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_diagonal_is_one(self, analytics, sample_returns):
        result = analytics.calculate_correlation_matrix(sample_returns)
        corr = result["correlation_matrix"]
        for ticker in ("AAPL", "MSFT", "GOOGL"):
            assert abs(corr[ticker][ticker] - 1.0) < 1e-10

    def test_symmetry(self, analytics, sample_returns):
        result = analytics.calculate_correlation_matrix(sample_returns)
        corr = result["correlation_matrix"]
        tickers = list(corr.keys())
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i + 1:]:
                assert abs(corr[t1][t2] - corr[t2][t1]) < 1e-10

    def test_correlation_range(self, analytics, sample_returns):
        result = analytics.calculate_correlation_matrix(sample_returns)
        assert -1.0 <= result["avg_correlation"] <= 1.0
        assert -1.0 <= result["max_correlation"] <= 1.0
        assert -1.0 <= result["min_correlation"] <= 1.0

    def test_high_correlation_detected(self, analytics, correlated_returns):
        result = analytics.calculate_correlation_matrix(correlated_returns)
        assert result["max_correlation"] > 0.8
        max_pair_set = set(result["max_pair"]["pair"])
        assert max_pair_set == {"A", "B"}

    def test_anti_correlation_detected(self, analytics, correlated_returns):
        result = analytics.calculate_correlation_matrix(correlated_returns)
        assert result["min_correlation"] < -0.5
        min_pair_set = set(result["min_pair"]["pair"])
        assert "C" in min_pair_set

    def test_spearman_method(self, analytics, sample_returns):
        result = analytics.calculate_correlation_matrix(sample_returns, method="spearman")
        assert result["method"] == "spearman"
        assert -1.0 <= result["avg_correlation"] <= 1.0

    def test_clustering_output(self, analytics, sample_returns):
        result = analytics.calculate_correlation_matrix(sample_returns)
        clustering = result["clustering"]
        assert "linkage_matrix" in clustering
        assert "labels" in clustering
        assert set(clustering["labels"]) == {"AAPL", "MSFT", "GOOGL"}
        assert len(clustering["linkage_matrix"]) == 2

    def test_insufficient_data_raises(self, analytics):
        dates = pd.date_range("2025-01-01", periods=10, freq="B")
        short_returns = {
            "X": pd.Series(np.random.normal(0, 0.02, 10), index=dates),
            "Y": pd.Series(np.random.normal(0, 0.02, 10), index=dates),
        }
        with pytest.raises(ValueError, match="Insufficient data"):
            analytics.calculate_correlation_matrix(short_returns)

    def test_num_positions(self, analytics, sample_returns):
        result = analytics.calculate_correlation_matrix(sample_returns)
        assert result["num_positions"] == 3

    def test_timestamp_is_isoformat(self, analytics, sample_returns):
        result = analytics.calculate_correlation_matrix(sample_returns)
        dt = datetime.fromisoformat(result["timestamp"])
        assert isinstance(dt, datetime)

# ==================== Stress Testing ====================

class TestStressTest:

    def test_market_crash_scenario(self, analytics, sample_positions_data):
        result = analytics.run_stress_test(sample_positions_data, scenario="market_crash")
        assert result["total_pnl"] < 0
        assert result["pnl_pct"] == pytest.approx(-10.0, abs=0.01)
        assert result["scenario"] == "market_crash"

    def test_market_rally_scenario(self, analytics, sample_positions_data):
        result = analytics.run_stress_test(sample_positions_data, scenario="market_rally")
        assert result["total_pnl"] > 0
        assert result["pnl_pct"] == pytest.approx(10.0, abs=0.01)

    def test_covid_crash_scenario(self, analytics, sample_positions_data):
        result = analytics.run_stress_test(sample_positions_data, scenario="covid_crash")
        assert result["pnl_pct"] == pytest.approx(-34.0, abs=0.01)

    def test_all_predefined_scenarios(self, analytics, sample_positions_data):
        scenarios = [
            "market_crash", "market_rally", "moderate_selloff", "rate_hike",
            "covid_crash", "financial_crisis_2008", "dotcom_bubble",
            "black_monday_1987", "flash_crash_2010", "brexit_2016",
        ]
        for scenario in scenarios:
            result = analytics.run_stress_test(sample_positions_data, scenario=scenario)
            assert result["scenario"] == scenario
            assert result["num_positions"] == 3

    def test_custom_shocks(self, analytics, sample_positions_data):
        shocks = {"AAPL": -0.20, "MSFT": 0.05, "GOOGL": -0.10}
        result = analytics.run_stress_test(
            sample_positions_data, scenario="custom", custom_shocks=shocks
        )
        assert result["scenario"] == "custom"
        assert result["position_impacts"]["AAPL"]["pnl_pct"] == pytest.approx(-20.0, abs=0.01)
        assert result["position_impacts"]["MSFT"]["pnl_pct"] == pytest.approx(5.0, abs=0.01)

    def test_worst_best_position(self, analytics, sample_positions_data):
        shocks = {"AAPL": -0.30, "MSFT": -0.05, "GOOGL": -0.10}
        result = analytics.run_stress_test(
            sample_positions_data, scenario="custom", custom_shocks=shocks
        )
        assert result["worst_position"] == "AAPL"
        assert result["best_position"] == "GOOGL"

    def test_pnl_math(self, analytics):
        positions = {"X": {"current_price": 100.0, "quantity": 10}}
        result = analytics.run_stress_test(positions, scenario="market_crash")
        assert result["total_pnl"] == pytest.approx(-100.0, abs=0.01)
        assert result["total_value_before"] == pytest.approx(1000.0, abs=0.01)
        assert result["total_value_after"] == pytest.approx(900.0, abs=0.01)

    def test_position_impacts_structure(self, analytics, sample_positions_data):
        result = analytics.run_stress_test(sample_positions_data, scenario="market_crash")
        for ticker, impact in result["position_impacts"].items():
            assert "pnl" in impact
            assert "pnl_pct" in impact
            assert "value_before" in impact
            assert "value_after" in impact
            assert "shock_applied" in impact

    def test_unknown_scenario_raises(self, analytics, sample_positions_data):
        with pytest.raises(ValueError, match="Unknown scenario"):
            analytics.run_stress_test(sample_positions_data, scenario="alien_invasion")

    def test_empty_positions(self, analytics):
        result = analytics.run_stress_test({}, scenario="market_crash")
        assert result["total_pnl"] == 0.0
        assert result["num_positions"] == 0

    def test_timestamp_present(self, analytics, sample_positions_data):
        result = analytics.run_stress_test(sample_positions_data, scenario="market_crash")
        dt = datetime.fromisoformat(result["timestamp"])
        assert isinstance(dt, datetime)

# ==================== FX Exposure Tests ====================

class TestFXExposure:

    def test_basic_structure(self, analytics, sample_fx_positions):
        result = analytics.analyze_fx_exposure(sample_fx_positions)
        expected_keys = {
            "exposures_by_currency", "total_exposure", "dominant_currency",
            "dominant_pct", "diversification_score", "num_currencies",
            "hedging_suggestions", "base_currency", "timestamp",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_total_exposure(self, analytics, sample_fx_positions):
        result = analytics.analyze_fx_exposure(sample_fx_positions)
        assert result["total_exposure"] == pytest.approx(32400.0, abs=0.01)

    def test_currency_count(self, analytics, sample_fx_positions):
        result = analytics.analyze_fx_exposure(sample_fx_positions)
        assert result["num_currencies"] == 3
        assert set(result["exposures_by_currency"].keys()) == {"USD", "CHF", "EUR"}

    def test_dominant_currency(self, analytics, sample_fx_positions):
        result = analytics.analyze_fx_exposure(sample_fx_positions)
        assert result["dominant_currency"] == "USD"
        assert result["dominant_pct"] == pytest.approx(62.96, abs=0.1)

    def test_currency_percentages_sum_to_100(self, analytics, sample_fx_positions):
        result = analytics.analyze_fx_exposure(sample_fx_positions)
        total_pct = sum(exp["pct"] for exp in result["exposures_by_currency"].values())
        assert total_pct == pytest.approx(100.0, abs=0.01)

    def test_num_positions_per_currency(self, analytics, sample_fx_positions):
        result = analytics.analyze_fx_exposure(sample_fx_positions)
        exposures = result["exposures_by_currency"]
        assert exposures["USD"]["num_positions"] == 2
        assert exposures["CHF"]["num_positions"] == 2
        assert exposures["EUR"]["num_positions"] == 1

    def test_diversification_score_range(self, analytics, sample_fx_positions):
        result = analytics.analyze_fx_exposure(sample_fx_positions)
        assert 0 <= result["diversification_score"] <= 100

    def test_single_currency_diversification_zero(self, analytics):
        positions = [
            {"ticker": "AAPL", "currency": "USD", "market_value_usd": 5000},
            {"ticker": "MSFT", "currency": "USD", "market_value_usd": 5000},
        ]
        result = analytics.analyze_fx_exposure(positions)
        assert result["diversification_score"] == pytest.approx(0.0, abs=0.01)
        assert result["num_currencies"] == 1

    def test_equal_two_currency_diversification(self, analytics):
        positions = [
            {"ticker": "AAPL", "currency": "USD", "market_value_usd": 5000},
            {"ticker": "NESN", "currency": "CHF", "market_value_usd": 5000},
        ]
        result = analytics.analyze_fx_exposure(positions)
        assert result["diversification_score"] == pytest.approx(50.0, abs=0.01)

    def test_hedging_suggestion_for_large_exposure(self, analytics):
        positions = [
            {"ticker": "AAPL", "currency": "USD", "market_value_usd": 5000},
            {"ticker": "NESN", "currency": "CHF", "market_value_usd": 5000},
        ]
        result = analytics.analyze_fx_exposure(positions, base_currency="USD")
        suggestions = result["hedging_suggestions"]
        assert any("CHF" in s for s in suggestions)

    def test_high_concentration_suggestion(self, analytics):
        positions = [
            {"ticker": "AAPL", "currency": "USD", "market_value_usd": 8000},
            {"ticker": "NESN", "currency": "CHF", "market_value_usd": 2000},
        ]
        result = analytics.analyze_fx_exposure(positions, base_currency="USD")
        suggestions = result["hedging_suggestions"]
        assert any("concentration" in s.lower() or "diversify" in s.lower() for s in suggestions)

    def test_base_currency_stored(self, analytics, sample_fx_positions):
        result = analytics.analyze_fx_exposure(sample_fx_positions, base_currency="EUR")
        assert result["base_currency"] == "EUR"

    def test_default_currency_fallback(self, analytics):
        positions = [
            {"ticker": "X", "market_value_usd": 5000},
        ]
        result = analytics.analyze_fx_exposure(positions, base_currency="USD")
        assert "USD" in result["exposures_by_currency"]

# ==================== Helper Method Tests ====================

class TestHelperMethods:

    def test_find_max_pair(self, analytics):
        data = pd.DataFrame({
            "A": [1.0, 0.9, 0.1],
            "B": [0.9, 1.0, -0.5],
            "C": [0.1, -0.5, 1.0],
        }, index=["A", "B", "C"])
        result = analytics._find_max_correlation_pair(data)
        assert set(result["pair"]) == {"A", "B"}
        assert result["correlation"] == pytest.approx(0.9, abs=0.01)

    def test_find_min_pair(self, analytics):
        data = pd.DataFrame({
            "A": [1.0, 0.9, 0.1],
            "B": [0.9, 1.0, -0.5],
            "C": [0.1, -0.5, 1.0],
        }, index=["A", "B", "C"])
        result = analytics._find_min_correlation_pair(data)
        assert set(result["pair"]) == {"B", "C"}
        assert result["correlation"] == pytest.approx(-0.5, abs=0.01)

    def test_cache_init(self, analytics):
        assert analytics.cache == {}
