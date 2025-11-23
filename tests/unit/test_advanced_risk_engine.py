"""
Unit tests for Phase 3A Advanced Risk Engine
Tests VaR calculations, stress testing, Monte Carlo simulation, and risk attribution

FIXED: Refactored from async to sync to match current implementation (Nov 2025)
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict

from services.risk.advanced_risk_engine import (
    AdvancedRiskEngine, create_advanced_risk_engine,
    VaRMethod, RiskHorizon, StressScenario,
    VaRResult, StressTestResult, MonteCarloResult
)


class TestAdvancedRiskEngineFixed:
    """Test cases for the Advanced Risk Engine (sync implementation)"""

    @pytest.fixture
    def config(self):
        """Default configuration for testing"""
        return {
            "enabled": True,
            "var": {
                "confidence_levels": [0.95, 0.99],
                "methods": ["parametric", "historical", "monte_carlo"],
                "lookback_days": 252,
                "min_observations": 100
            },
            "var_limits": {
                "daily_95": 0.05,
                "daily_99": 0.08,
                "weekly_95": 0.12,
                "monthly_95": 0.20
            },
            "stress_testing": {
                "enabled_scenarios": ["crisis_2008", "covid_2020", "china_ban"],
                "max_acceptable_loss": 0.15,
                "recovery_model": "exponential"
            },
            "monte_carlo": {
                "simulations": 1000,  # Reduced for faster tests
                "max_horizon_days": 30,
                "distribution": "student_t",
                "correlation_model": "dynamic",
                "extreme_percentiles": [1, 5, 95, 99]
            }
        }

    @pytest.fixture
    def sample_portfolio(self):
        """Sample portfolio for testing"""
        return {
            "weights": {"BTC": 0.4, "ETH": 0.3, "SOL": 0.2, "AVAX": 0.1},
            "value": 100000.0
        }

    @pytest.fixture
    def mock_price_data(self):
        """Mock price data for testing"""
        # Create sample price data (252 days, 4 assets)
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

        # Simulate crypto-like returns with higher volatility
        btc_returns = np.random.normal(0.001, 0.04, 252)  # 4% daily vol
        eth_returns = np.random.normal(0.0008, 0.05, 252)  # 5% daily vol
        sol_returns = np.random.normal(0.002, 0.07, 252)  # 7% daily vol
        avax_returns = np.random.normal(0.001, 0.06, 252)  # 6% daily vol

        # Add correlation between BTC and ETH
        eth_returns = 0.7 * btc_returns + 0.3 * eth_returns

        price_data = pd.DataFrame({
            'BTC': 50000 * np.exp(np.cumsum(btc_returns)),
            'ETH': 3000 * np.exp(np.cumsum(eth_returns)),
            'SOL': 100 * np.exp(np.cumsum(sol_returns)),
            'AVAX': 50 * np.exp(np.cumsum(avax_returns))
        }, index=dates)

        return price_data

    @pytest.fixture
    def risk_engine(self, config):
        """Create risk engine with config"""
        engine = create_advanced_risk_engine(config)
        return engine

    def test_create_advanced_risk_engine(self, config):
        """Test risk engine creation"""
        engine = create_advanced_risk_engine(config)
        assert isinstance(engine, AdvancedRiskEngine)
        assert engine.config == config

    def test_create_risk_engine_disabled(self):
        """Test risk engine creation when disabled"""
        config = {"enabled": False}
        engine = create_advanced_risk_engine(config)
        assert engine is None

    def test_parametric_var_calculation(self, risk_engine, sample_portfolio):
        """Test parametric VaR calculation"""
        # No mocking needed - _get_returns_matrix generates simulated data
        result = risk_engine.calculate_var(
            portfolio_weights=sample_portfolio["weights"],
            portfolio_value=sample_portfolio["value"],
            method=VaRMethod.PARAMETRIC,
            confidence_level=0.95,
            horizon=RiskHorizon.DAILY
        )

        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.PARAMETRIC
        assert result.confidence_level == 0.95
        assert result.horizon == RiskHorizon.DAILY
        assert result.portfolio_value == sample_portfolio["value"]
        assert result.var_absolute > 0  # Should be positive (loss amount)
        assert result.cvar_absolute >= result.var_absolute  # CVaR >= VaR
        assert isinstance(result.calculated_at, datetime)

    def test_historical_var_calculation(self, risk_engine, sample_portfolio):
        """Test historical VaR calculation"""
        result = risk_engine.calculate_var(
            portfolio_weights=sample_portfolio["weights"],
            portfolio_value=sample_portfolio["value"],
            method=VaRMethod.HISTORICAL,
            confidence_level=0.99,
            horizon=RiskHorizon.WEEKLY
        )

        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.HISTORICAL
        assert result.confidence_level == 0.99
        assert result.horizon == RiskHorizon.WEEKLY
        assert result.var_absolute > 0
        assert result.cvar_absolute >= result.var_absolute

    def test_monte_carlo_var_calculation(self, risk_engine, sample_portfolio):
        """Test Monte Carlo VaR calculation"""
        result = risk_engine.calculate_var(
            portfolio_weights=sample_portfolio["weights"],
            portfolio_value=sample_portfolio["value"],
            method=VaRMethod.MONTE_CARLO,
            confidence_level=0.95,
            horizon=RiskHorizon.DAILY
        )

        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.MONTE_CARLO
        assert result.var_absolute > 0
        assert result.cvar_absolute >= result.var_absolute

    def test_stress_test_2008_crisis(self, risk_engine, sample_portfolio):
        """Test stress testing with 2008 crisis scenario"""
        results = risk_engine.run_stress_test(
            portfolio_weights=sample_portfolio["weights"],
            portfolio_value=sample_portfolio["value"],
            scenarios=["crisis_2008"]  # Note: scenarios plural, string format
        )

        assert isinstance(results, list)
        assert len(results) > 0
        result = results[0]
        assert isinstance(result, StressTestResult)
        assert result.scenario == "crisis_2008"
        assert result.portfolio_pnl < 0  # Should show losses (negative P&L)
        assert result.portfolio_pnl_pct < 0  # Negative percentage (loss)
        assert len(result.asset_pnl) == 4  # All assets should be impacted
        assert isinstance(result.calculated_at, datetime)

    def test_stress_test_covid_crash(self, risk_engine, sample_portfolio):
        """Test COVID crash scenario"""
        results = risk_engine.run_stress_test(
            portfolio_weights=sample_portfolio["weights"],
            portfolio_value=sample_portfolio["value"],
            scenarios=["covid_2020"]  # Note: scenarios plural, string format
        )

        assert isinstance(results, list)
        assert len(results) > 0
        result = results[0]
        assert isinstance(result, StressTestResult)
        assert result.scenario == "covid_2020"
        assert result.portfolio_pnl < 0  # Losses
        assert result.recovery_time_days is None or result.recovery_time_days > 0

    def test_monte_carlo_simulation(self, risk_engine, sample_portfolio):
        """Test Monte Carlo simulation"""
        result = risk_engine.run_monte_carlo_simulation(
            portfolio_weights=sample_portfolio["weights"],
            portfolio_value=sample_portfolio["value"],
            horizon_days=30,
            simulations=500  # Reduced for speed
        )

        assert isinstance(result, MonteCarloResult)
        assert result.simulations_count == 500
        assert result.horizon_days == 30
        assert result.var_95 > 0
        assert result.var_99 > result.var_95  # 99% VaR should be higher
        assert result.cvar_95 >= result.var_95
        assert result.cvar_99 >= result.var_99
        assert isinstance(result.calculated_at, datetime)

    def test_var_method_enum_validation(self, risk_engine, sample_portfolio):
        """Test VaR method enum validation"""
        # Should accept enum
        result = risk_engine.calculate_var(
            portfolio_weights=sample_portfolio["weights"],
            portfolio_value=sample_portfolio["value"],
            method=VaRMethod.PARAMETRIC,
            confidence_level=0.95
        )
        assert result.method == VaRMethod.PARAMETRIC

        # Test with historical method
        result = risk_engine.calculate_var(
            portfolio_weights=sample_portfolio["weights"],
            portfolio_value=sample_portfolio["value"],
            method=VaRMethod.HISTORICAL,
            confidence_level=0.95
        )
        assert result.method == VaRMethod.HISTORICAL

    def test_confidence_level_validation(self, risk_engine, sample_portfolio):
        """Test confidence level validation"""
        # Valid confidence levels
        for cl in [0.90, 0.95, 0.99]:
            result = risk_engine.calculate_var(
                portfolio_weights=sample_portfolio["weights"],
                portfolio_value=sample_portfolio["value"],
                method=VaRMethod.PARAMETRIC,
                confidence_level=cl
            )
            assert result.confidence_level == cl

    def test_horizon_scaling(self, risk_engine, sample_portfolio):
        """Test that VaR scales appropriately with horizon"""
        # Daily VaR
        daily = risk_engine.calculate_var(
            portfolio_weights=sample_portfolio["weights"],
            portfolio_value=sample_portfolio["value"],
            method=VaRMethod.PARAMETRIC,
            confidence_level=0.95,
            horizon=RiskHorizon.DAILY
        )

        # Weekly VaR should be higher (more risk over longer period)
        weekly = risk_engine.calculate_var(
            portfolio_weights=sample_portfolio["weights"],
            portfolio_value=sample_portfolio["value"],
            method=VaRMethod.PARAMETRIC,
            confidence_level=0.95,
            horizon=RiskHorizon.WEEKLY
        )

        # Weekly VaR should be roughly sqrt(5-7) times daily VaR
        assert weekly.var_absolute > daily.var_absolute

    def test_crisis_2008_scenario_exists(self, risk_engine):
        """Test that 2008 crisis scenario is defined"""
        assert StressScenario.FINANCIAL_CRISIS_2008 in StressScenario
        assert hasattr(risk_engine, 'stress_scenarios')

    def test_covid_2020_scenario_exists(self, risk_engine):
        """Test that COVID 2020 scenario is defined"""
        assert StressScenario.COVID_CRASH_2020 in StressScenario

    def test_china_ban_scenario_exists(self, risk_engine):
        """Test that China ban scenario is defined"""
        assert StressScenario.CHINA_BAN_CRYPTO in StressScenario
