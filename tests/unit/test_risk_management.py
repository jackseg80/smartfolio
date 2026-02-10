"""
Tests unitaires pour services/risk_management.py

Couvre les méthodes synchrones de AdvancedRiskManager:
- _build_stress_scenarios
- _check_risk_threshold_alerts
- _check_performance_alerts
- _check_correlation_alerts
- _check_concentration_alerts
- _check_data_quality_alerts
- _generate_asset_universe
- _calculate_asset_contributions
- _calculate_group_contributions
- _calculate_attribution_effects
- get_system_status
"""

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from services.risk_management import AdvancedRiskManager
from services.risk.models import (
    RiskLevel,
    StressScenario,
    RiskMetrics,
    CorrelationMatrix,
    AlertSeverity,
    AlertCategory,
    RiskAlert,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    return AdvancedRiskManager()


@pytest.fixture
def high_risk_metrics():
    """RiskMetrics with values that should trigger high/critical alerts."""
    return RiskMetrics(
        var_95_1d=0.30,
        var_99_1d=0.45,
        volatility_annualized=1.5,
        sharpe_ratio=-1.0,
        max_drawdown=0.55,
        current_drawdown=0.45,
        confidence_level=0.3,
        data_points=5,
    )


@pytest.fixture
def medium_risk_metrics():
    """RiskMetrics with moderate values."""
    return RiskMetrics(
        var_95_1d=0.12,
        var_99_1d=0.18,
        volatility_annualized=0.65,
        sharpe_ratio=-0.3,
        max_drawdown=0.25,
        current_drawdown=0.18,
        confidence_level=0.7,
        data_points=30,
    )


@pytest.fixture
def low_risk_metrics():
    """RiskMetrics with low values — should trigger few alerts."""
    return RiskMetrics(
        var_95_1d=0.03,
        var_99_1d=0.05,
        volatility_annualized=0.15,
        sharpe_ratio=1.5,
        max_drawdown=0.08,
        current_drawdown=0.02,
        confidence_level=0.9,
        data_points=60,
    )


@pytest.fixture
def high_corr_matrix():
    """CorrelationMatrix with low diversification."""
    return CorrelationMatrix(
        correlations={
            "BTC": {"BTC": 1.0, "ETH": 0.95, "SOL": 0.92},
            "ETH": {"BTC": 0.95, "ETH": 1.0, "SOL": 0.93},
            "SOL": {"BTC": 0.92, "ETH": 0.93, "SOL": 1.0},
        },
        diversification_ratio=0.3,
        effective_assets=1.2,
    )


@pytest.fixture
def low_corr_matrix():
    """CorrelationMatrix with good diversification."""
    return CorrelationMatrix(
        correlations={
            "BTC": {"BTC": 1.0, "ETH": 0.4, "USDC": 0.05},
            "ETH": {"BTC": 0.4, "ETH": 1.0, "USDC": 0.02},
            "USDC": {"BTC": 0.05, "ETH": 0.02, "USDC": 1.0},
        },
        diversification_ratio=1.2,
        effective_assets=2.8,
    )


@pytest.fixture
def sample_holdings():
    return [
        {"symbol": "BTC", "alias": "bitcoin", "value_usd": 50000},
        {"symbol": "ETH", "alias": "ethereum", "value_usd": 30000},
        {"symbol": "SOL", "alias": "solana", "value_usd": 20000},
    ]


@pytest.fixture
def concentrated_holdings():
    return [
        {"symbol": "BTC", "alias": "bitcoin", "value_usd": 95000},
        {"symbol": "ETH", "alias": "ethereum", "value_usd": 5000},
    ]


# ---------------------------------------------------------------------------
# _build_stress_scenarios
# ---------------------------------------------------------------------------

class TestBuildStressScenarios:
    def test_all_scenarios_present(self, manager):
        scenarios = manager.stress_scenarios
        assert StressScenario.BEAR_MARKET_2018 in scenarios
        assert StressScenario.COVID_CRASH_2020 in scenarios
        assert StressScenario.LUNA_COLLAPSE_2022 in scenarios
        assert StressScenario.FTX_COLLAPSE_2022 in scenarios

    def test_scenario_has_required_fields(self, manager):
        for scenario, config in manager.stress_scenarios.items():
            assert "name" in config
            assert "description" in config
            assert "asset_shocks" in config
            assert "correlation_increase" in config
            assert "duration_days" in config
            assert "volatility_multiplier" in config

    def test_scenario_shocks_are_negative(self, manager):
        for scenario, config in manager.stress_scenarios.items():
            shocks = config["asset_shocks"]
            # Most shocks should be negative (except stablecoins)
            negative_count = sum(1 for v in shocks.values() if v < 0)
            assert negative_count >= len(shocks) - 2  # Allow 1-2 positive (stablecoins)


# ---------------------------------------------------------------------------
# _check_risk_threshold_alerts
# ---------------------------------------------------------------------------

class TestRiskThresholdAlerts:
    def test_critical_var_triggers_alert(self, manager, high_risk_metrics):
        alerts = manager._check_risk_threshold_alerts(high_risk_metrics)
        var_alerts = [a for a in alerts if "var" in a.id.lower()]
        assert len(var_alerts) >= 1
        assert any(a.severity == AlertSeverity.CRITICAL for a in var_alerts)

    def test_critical_volatility_triggers_alert(self, manager, high_risk_metrics):
        alerts = manager._check_risk_threshold_alerts(high_risk_metrics)
        vol_alerts = [a for a in alerts if "volatility" in a.id.lower()]
        assert len(vol_alerts) >= 1
        assert any(a.severity == AlertSeverity.CRITICAL for a in vol_alerts)

    def test_critical_drawdown_triggers_alert(self, manager, high_risk_metrics):
        alerts = manager._check_risk_threshold_alerts(high_risk_metrics)
        dd_alerts = [a for a in alerts if "drawdown" in a.id.lower()]
        assert len(dd_alerts) >= 1

    def test_medium_risk_triggers_medium_alerts(self, manager, medium_risk_metrics):
        alerts = manager._check_risk_threshold_alerts(medium_risk_metrics)
        # Should have some alerts but not critical
        severities = [a.severity for a in alerts]
        assert AlertSeverity.CRITICAL not in severities or len(alerts) == 0

    def test_low_risk_no_alerts(self, manager, low_risk_metrics):
        alerts = manager._check_risk_threshold_alerts(low_risk_metrics)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# _check_performance_alerts
# ---------------------------------------------------------------------------

class TestPerformanceAlerts:
    def test_very_negative_sharpe_triggers_high(self, manager, high_risk_metrics):
        alerts = manager._check_performance_alerts(high_risk_metrics)
        sharpe_alerts = [a for a in alerts if "sharpe" in a.id.lower()]
        assert len(sharpe_alerts) >= 1
        assert any(a.severity == AlertSeverity.HIGH for a in sharpe_alerts)

    def test_critical_max_drawdown_triggers_alert(self, manager, high_risk_metrics):
        alerts = manager._check_performance_alerts(high_risk_metrics)
        dd_alerts = [a for a in alerts if "drawdown" in a.id.lower()]
        assert len(dd_alerts) >= 1

    def test_good_sharpe_no_alert(self, manager, low_risk_metrics):
        alerts = manager._check_performance_alerts(low_risk_metrics)
        sharpe_alerts = [a for a in alerts if "sharpe" in a.id.lower()]
        assert len(sharpe_alerts) == 0

    def test_low_drawdown_no_alert(self, manager, low_risk_metrics):
        alerts = manager._check_performance_alerts(low_risk_metrics)
        dd_alerts = [a for a in alerts if "drawdown" in a.id.lower()]
        assert len(dd_alerts) == 0


# ---------------------------------------------------------------------------
# _check_correlation_alerts
# ---------------------------------------------------------------------------

class TestCorrelationAlerts:
    def test_low_diversification_triggers_alert(self, manager, high_corr_matrix):
        alerts = manager._check_correlation_alerts(high_corr_matrix)
        div_alerts = [a for a in alerts if "diversification" in a.id.lower()]
        assert len(div_alerts) >= 1

    def test_extreme_correlations_trigger_alert(self, manager, high_corr_matrix):
        alerts = manager._check_correlation_alerts(high_corr_matrix)
        corr_alerts = [a for a in alerts if "correlation" in a.id.lower()]
        assert len(corr_alerts) >= 1

    def test_good_diversification_no_alert(self, manager, low_corr_matrix):
        alerts = manager._check_correlation_alerts(low_corr_matrix)
        div_alerts = [a for a in alerts if "diversification" in a.id.lower()]
        assert len(div_alerts) == 0

    def test_empty_correlation_matrix(self, manager):
        empty = CorrelationMatrix()
        alerts = manager._check_correlation_alerts(empty)
        # Should not crash with empty correlations
        assert isinstance(alerts, list)


# ---------------------------------------------------------------------------
# _check_concentration_alerts
# ---------------------------------------------------------------------------

class TestConcentrationAlerts:
    def test_high_concentration_triggers_alert(self, manager, concentrated_holdings):
        alerts = manager._check_concentration_alerts(concentrated_holdings)
        assert len(alerts) >= 1
        # 95% in BTC should trigger critical
        btc_alerts = [a for a in alerts if "BTC" in a.id]
        assert len(btc_alerts) >= 1
        assert any(a.severity == AlertSeverity.CRITICAL for a in btc_alerts)

    def test_balanced_portfolio_fewer_alerts(self, manager, sample_holdings):
        alerts = manager._check_concentration_alerts(sample_holdings)
        critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical) == 0

    def test_empty_portfolio_no_alerts(self, manager):
        alerts = manager._check_concentration_alerts([])
        assert len(alerts) == 0

    def test_zero_value_portfolio_no_alerts(self, manager):
        holdings = [{"symbol": "BTC", "value_usd": 0}]
        alerts = manager._check_concentration_alerts(holdings)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# _check_data_quality_alerts
# ---------------------------------------------------------------------------

class TestDataQualityAlerts:
    def test_low_confidence_triggers_alert(self, manager, high_risk_metrics):
        alerts = manager._check_data_quality_alerts(high_risk_metrics, CorrelationMatrix())
        conf_alerts = [a for a in alerts if "confidence" in a.id.lower()]
        assert len(conf_alerts) >= 1

    def test_insufficient_data_triggers_alert(self, manager, high_risk_metrics):
        alerts = manager._check_data_quality_alerts(high_risk_metrics, CorrelationMatrix())
        data_alerts = [a for a in alerts if "data" in a.id.lower() or "insufficient" in a.id.lower()]
        assert len(data_alerts) >= 1

    def test_good_data_no_alerts(self, manager, low_risk_metrics):
        alerts = manager._check_data_quality_alerts(low_risk_metrics, CorrelationMatrix())
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# _generate_asset_universe
# ---------------------------------------------------------------------------

class TestGenerateAssetUniverse:
    def test_btc_eth_allocation(self, manager):
        allocations = {"BTC": 0.5, "ETH": 0.3, "Stablecoins": 0.2}
        universe = manager._generate_asset_universe(allocations)
        symbols = [h["symbol"] for h in universe]
        assert "BTC" in symbols
        assert "ETH" in symbols
        assert "USDT" in symbols or "USDC" in symbols

    def test_unknown_group_creates_representative(self, manager):
        allocations = {"BTC": 0.5, "CustomGroup": 0.5}
        universe = manager._generate_asset_universe(allocations)
        symbols = [h["symbol"] for h in universe]
        assert any("CustomGroup" in s for s in symbols)

    def test_all_groups_produce_assets(self, manager):
        allocations = {
            "BTC": 0.2, "ETH": 0.2, "Stablecoins": 0.1,
            "L1/L0 majors": 0.1, "L2/Scaling": 0.1,
            "DeFi": 0.1, "AI/Data": 0.1, "Gaming/NFT": 0.05,
            "Memecoins": 0.05,
        }
        universe = manager._generate_asset_universe(allocations)
        assert len(universe) >= 9  # At least 1 per group


# ---------------------------------------------------------------------------
# _calculate_asset_contributions
# ---------------------------------------------------------------------------

class TestCalculateAssetContributions:
    def test_basic_contributions(self, manager):
        holdings = [
            {"symbol": "BTC", "alias": "bitcoin", "value_usd": 60000},
            {"symbol": "ETH", "alias": "ethereum", "value_usd": 40000},
        ]
        returns_data = [
            {"BTC": 0.02, "ETH": 0.01},
            {"BTC": -0.01, "ETH": 0.03},
            {"BTC": 0.005, "ETH": -0.005},
        ]
        result = manager._calculate_asset_contributions(holdings, returns_data, 100000)
        assert len(result) == 2
        assert all("contribution_pct" in c for c in result)
        assert all("weight" in c for c in result)

    def test_sorted_by_contribution(self, manager):
        holdings = [
            {"symbol": "BTC", "alias": "bitcoin", "value_usd": 50000},
            {"symbol": "ETH", "alias": "ethereum", "value_usd": 50000},
        ]
        returns_data = [
            {"BTC": 0.05, "ETH": -0.02},
        ]
        result = manager._calculate_asset_contributions(holdings, returns_data, 100000)
        assert result[0]["contribution_pct"] >= result[-1]["contribution_pct"]

    def test_zero_portfolio_value(self, manager):
        holdings = [{"symbol": "BTC", "alias": "bitcoin", "value_usd": 0}]
        returns_data = [{"BTC": 0.01}]
        result = manager._calculate_asset_contributions(holdings, returns_data, 0)
        assert len(result) == 1
        assert result[0]["weight"] == 0


# ---------------------------------------------------------------------------
# _calculate_group_contributions
# ---------------------------------------------------------------------------

class TestCalculateGroupContributions:
    def test_group_aggregation(self, manager):
        holdings = [
            {"symbol": "BTC", "alias": "bitcoin", "value_usd": 60000},
            {"symbol": "ETH", "alias": "ethereum", "value_usd": 40000},
        ]
        asset_contributions = [
            {
                "symbol": "BTC", "alias": "bitcoin", "group": "BTC",
                "weight": 0.6, "value_usd": 60000,
                "asset_return": 0.05, "contribution_pct": 0.03,
                "contribution_usd": 3000, "volatility": 0.4,
                "sharpe_ratio": 1.2, "daily_returns": [0.01, 0.02, 0.02],
            },
            {
                "symbol": "ETH", "alias": "ethereum", "group": "ETH",
                "weight": 0.4, "value_usd": 40000,
                "asset_return": -0.02, "contribution_pct": -0.008,
                "contribution_usd": -800, "volatility": 0.6,
                "sharpe_ratio": -0.3, "daily_returns": [0.03, -0.02, -0.03],
            },
        ]
        result = manager._calculate_group_contributions(holdings, asset_contributions)
        assert "BTC" in result
        assert "ETH" in result
        assert result["BTC"]["num_assets"] == 1
        assert result["BTC"]["total_weight"] == 0.6


# ---------------------------------------------------------------------------
# _calculate_attribution_effects
# ---------------------------------------------------------------------------

class TestAttributionEffects:
    def test_brinson_effects_calculated(self, manager):
        asset_contributions = [
            {"group": "BTC", "asset_return": 0.05, "weight": 0.6, "contribution_pct": 0.03},
            {"group": "ETH", "asset_return": -0.02, "weight": 0.4, "contribution_pct": -0.008},
        ]
        group_contributions = {
            "BTC": {"total_weight": 0.6, "group_return": 0.05},
            "ETH": {"total_weight": 0.4, "group_return": -0.02},
        }
        effects = manager._calculate_attribution_effects(
            asset_contributions, group_contributions, None, []
        )
        assert "allocation" in effects
        assert "selection" in effects
        assert "interaction" in effects

    def test_all_effects_are_numeric(self, manager):
        asset_contributions = [
            {"group": "BTC", "asset_return": 0.0, "weight": 1.0, "contribution_pct": 0.0},
        ]
        group_contributions = {
            "BTC": {"total_weight": 1.0, "group_return": 0.0},
        }
        effects = manager._calculate_attribution_effects(
            asset_contributions, group_contributions, None, []
        )
        for v in effects.values():
            assert isinstance(v, (int, float))


# ---------------------------------------------------------------------------
# get_system_status
# ---------------------------------------------------------------------------

class TestGetSystemStatus:
    def test_returns_operational(self, manager):
        status = manager.get_system_status()
        assert status["status"] == "operational"
        assert status["risk_manager_initialized"] is True
        assert status["alert_system_active"] is True
        assert "supported_scenarios" in status
        assert "timestamp" in status

    def test_supported_scenarios_list(self, manager):
        status = manager.get_system_status()
        scenarios = status["supported_scenarios"]
        assert len(scenarios) == 4  # 4 predefined scenarios

    def test_cache_size_initially_zero(self, manager):
        status = manager.get_system_status()
        assert status["cache_size"] == 0


# ---------------------------------------------------------------------------
# Init & Configuration
# ---------------------------------------------------------------------------

class TestManagerInit:
    def test_risk_free_rate(self, manager):
        assert manager.risk_free_rate == 0.02

    def test_risk_thresholds_all_levels(self, manager):
        for level in RiskLevel:
            assert level in manager.risk_thresholds

    def test_max_history_days(self, manager):
        assert manager.max_history_days == 365

    def test_var_calculator_initialized(self, manager):
        assert manager.var_calculator is not None
        assert manager.var_calculator.risk_free_rate == 0.02

    def test_alert_system_initialized(self, manager):
        assert manager.alert_system is not None
        assert len(manager.alert_system.thresholds) > 0
