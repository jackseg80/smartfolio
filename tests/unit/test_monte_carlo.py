"""Tests for services/risk/monte_carlo.py — MonteCarloResult dataclass + simulation logic"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from services.risk.monte_carlo import MonteCarloResult, run_monte_carlo_simulation


# ---------------------------------------------------------------------------
# TestMonteCarloResult
# ---------------------------------------------------------------------------
class TestMonteCarloResult:
    def test_dataclass_creation(self):
        result = MonteCarloResult(
            num_simulations=1000,
            horizon_days=30,
            confidence_level=0.95,
            mean_return_pct=2.5,
            median_return_pct=2.0,
            std_return_pct=8.0,
            worst_case_pct=-25.0,
            best_case_pct=30.0,
            percentile_5_pct=-12.0,
            percentile_95_pct=18.0,
            prob_loss_any=0.45,
            prob_loss_5=0.20,
            prob_loss_10=0.10,
            prob_loss_20=0.03,
            prob_loss_30=0.01,
            var_95_pct=12.0,
            cvar_95_pct=16.0,
            var_99_pct=20.0,
            cvar_99_pct=24.0,
            distribution_percentiles={1: -25, 5: -12, 50: 2, 95: 18, 99: 30},
            portfolio_value=100000.0,
            num_assets=5,
            timestamp=datetime.now(),
        )
        assert result.num_simulations == 1000
        assert result.horizon_days == 30
        assert result.var_95_pct == 12.0
        assert result.num_assets == 5

    def test_all_probabilities_between_0_and_1(self):
        result = MonteCarloResult(
            num_simulations=100,
            horizon_days=7,
            confidence_level=0.95,
            mean_return_pct=0, median_return_pct=0, std_return_pct=5,
            worst_case_pct=-20, best_case_pct=20,
            percentile_5_pct=-10, percentile_95_pct=10,
            prob_loss_any=0.5, prob_loss_5=0.3, prob_loss_10=0.15,
            prob_loss_20=0.05, prob_loss_30=0.01,
            var_95_pct=10, cvar_95_pct=12, var_99_pct=18, cvar_99_pct=20,
            distribution_percentiles={}, portfolio_value=10000, num_assets=3,
            timestamp=datetime.now(),
        )
        for prob in [result.prob_loss_any, result.prob_loss_5, result.prob_loss_10,
                     result.prob_loss_20, result.prob_loss_30]:
            assert 0 <= prob <= 1

    def test_worst_case_less_than_best_case(self):
        result = MonteCarloResult(
            num_simulations=100, horizon_days=30, confidence_level=0.95,
            mean_return_pct=1, median_return_pct=0.5, std_return_pct=5,
            worst_case_pct=-30, best_case_pct=25,
            percentile_5_pct=-15, percentile_95_pct=15,
            prob_loss_any=0.4, prob_loss_5=0.2, prob_loss_10=0.1,
            prob_loss_20=0.05, prob_loss_30=0.02,
            var_95_pct=10, cvar_95_pct=14, var_99_pct=20, cvar_99_pct=25,
            distribution_percentiles={}, portfolio_value=50000, num_assets=4,
            timestamp=datetime.now(),
        )
        assert result.worst_case_pct < result.best_case_pct


# ---------------------------------------------------------------------------
# TestRunMonteCarloSimulation
# ---------------------------------------------------------------------------
class TestRunMonteCarloSimulation:
    """Test the async simulation function with mocked price data"""

    def _make_price_data(self, n_days=100, base_price=100.0, volatility=0.02):
        """Generate fake price history: list of (timestamp, price) tuples"""
        import time
        prices = []
        price = base_price
        t0 = time.time() - n_days * 86400
        for i in range(n_days):
            price *= (1 + np.random.normal(0, volatility))
            prices.append((t0 + i * 86400, max(price, 1.0)))
        return prices

    @pytest.mark.asyncio
    async def test_basic_simulation(self):
        np.random.seed(42)
        btc_prices = self._make_price_data(100, 50000, 0.03)
        eth_prices = self._make_price_data(100, 3000, 0.04)

        def mock_get_history(symbol, days=365):
            if symbol == "BTC":
                return btc_prices
            elif symbol == "ETH":
                return eth_prices
            return []

        holdings = [
            {"symbol": "BTC", "value_usd": 50000},
            {"symbol": "ETH", "value_usd": 30000},
        ]

        with patch("services.price_history.get_cached_history", side_effect=mock_get_history):
            result = await run_monte_carlo_simulation(
                holdings, num_simulations=500, horizon_days=30
            )

        assert isinstance(result, MonteCarloResult)
        assert result.num_simulations == 500
        assert result.horizon_days == 30
        assert result.num_assets == 2
        assert result.portfolio_value == 80000.0
        # Basic sanity: worst < mean < best
        assert result.worst_case_pct < result.best_case_pct

    @pytest.mark.asyncio
    async def test_zero_portfolio_raises(self):
        holdings = [{"symbol": "BTC", "value_usd": 0}]
        with pytest.raises(ValueError, match="zero"):
            await run_monte_carlo_simulation(holdings)

    @pytest.mark.asyncio
    async def test_insufficient_assets_raises(self):
        """Need at least 2 assets with sufficient price data"""
        def mock_get_history(symbol, days=365):
            return []  # No data for any asset

        holdings = [
            {"symbol": "BTC", "value_usd": 50000},
            {"symbol": "ETH", "value_usd": 30000},
        ]

        with patch("services.price_history.get_cached_history", side_effect=mock_get_history):
            with pytest.raises(ValueError, match="Insufficient"):
                await run_monte_carlo_simulation(holdings, num_simulations=100)

    @pytest.mark.asyncio
    async def test_dust_assets_filtered(self):
        """Assets with <0.1% weight should be filtered out"""
        np.random.seed(42)
        btc_prices = self._make_price_data(100, 50000, 0.03)
        eth_prices = self._make_price_data(100, 3000, 0.04)
        dust_prices = self._make_price_data(100, 1, 0.05)

        def mock_get_history(symbol, days=365):
            mapping = {"BTC": btc_prices, "ETH": eth_prices, "DUST": dust_prices}
            return mapping.get(symbol, [])

        holdings = [
            {"symbol": "BTC", "value_usd": 50000},
            {"symbol": "ETH", "value_usd": 30000},
            {"symbol": "DUST", "value_usd": 0.05},  # <0.1% of 80000
        ]

        with patch("services.price_history.get_cached_history", side_effect=mock_get_history):
            result = await run_monte_carlo_simulation(
                holdings, num_simulations=100, horizon_days=7
            )

        assert result.num_assets == 2  # DUST filtered

    @pytest.mark.asyncio
    async def test_distribution_percentiles_structure(self):
        np.random.seed(42)
        btc_prices = self._make_price_data(100, 50000, 0.03)
        eth_prices = self._make_price_data(100, 3000, 0.04)

        def mock_get_history(symbol, days=365):
            return {"BTC": btc_prices, "ETH": eth_prices}.get(symbol, [])

        holdings = [
            {"symbol": "BTC", "value_usd": 50000},
            {"symbol": "ETH", "value_usd": 30000},
        ]

        with patch("services.price_history.get_cached_history", side_effect=mock_get_history):
            result = await run_monte_carlo_simulation(
                holdings, num_simulations=500, horizon_days=30
            )

        # distribution_percentiles should have keys 1, 5, 10, 25, 50, 75, 90, 95, 99
        expected_keys = {1, 5, 10, 25, 50, 75, 90, 95, 99}
        assert set(result.distribution_percentiles.keys()) == expected_keys
        # Percentiles should be monotonically increasing
        vals = [result.distribution_percentiles[k] for k in sorted(result.distribution_percentiles)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]

    @pytest.mark.asyncio
    async def test_var_cvar_ordering(self):
        """CVaR should be >= VaR (more extreme)"""
        np.random.seed(42)
        btc_prices = self._make_price_data(100, 50000, 0.03)
        eth_prices = self._make_price_data(100, 3000, 0.04)

        def mock_get_history(symbol, days=365):
            return {"BTC": btc_prices, "ETH": eth_prices}.get(symbol, [])

        holdings = [
            {"symbol": "BTC", "value_usd": 50000},
            {"symbol": "ETH", "value_usd": 30000},
        ]

        with patch("services.price_history.get_cached_history", side_effect=mock_get_history):
            result = await run_monte_carlo_simulation(
                holdings, num_simulations=1000, horizon_days=30
            )

        assert result.cvar_95_pct >= result.var_95_pct
        assert result.cvar_99_pct >= result.var_99_pct
        assert result.var_99_pct >= result.var_95_pct
