"""Tests for services/risk/stress_testing.py — Scenarios, dataclasses, and simulation"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from services.risk.stress_testing import (
    StressScenario,
    StressTestResult,
    PREDEFINED_SCENARIOS,
    get_available_scenarios,
    calculate_stress_test,
)


# ---------------------------------------------------------------------------
# TestPredefinedScenarios
# ---------------------------------------------------------------------------
class TestPredefinedScenarios:
    def test_six_scenarios_defined(self):
        assert len(PREDEFINED_SCENARIOS) == 6

    def test_all_scenario_ids(self):
        expected = {"crisis_2008", "covid_2020", "china_ban", "tether_collapse", "fed_emergency", "exchange_hack"}
        assert set(PREDEFINED_SCENARIOS.keys()) == expected

    def test_all_scenarios_have_required_fields(self):
        for sid, scenario in PREDEFINED_SCENARIOS.items():
            assert isinstance(scenario, StressScenario)
            assert scenario.id == sid
            assert len(scenario.name) > 0
            assert len(scenario.description) > 0
            assert scenario.impact_min_pct < 0
            assert scenario.impact_max_pct < 0
            assert scenario.impact_min_pct >= scenario.impact_max_pct  # min is less negative
            assert 0 < scenario.probability_10y < 1
            assert scenario.duration_months_min > 0
            assert scenario.duration_months_min <= scenario.duration_months_max

    def test_all_scenarios_have_group_shocks(self):
        for sid, scenario in PREDEFINED_SCENARIOS.items():
            assert len(scenario.group_shocks) > 0
            for group, shock in scenario.group_shocks.items():
                assert -1.0 <= shock <= 0.0, f"{sid}/{group}: shock {shock} out of range"

    def test_btc_shock_present_in_all(self):
        for sid, scenario in PREDEFINED_SCENARIOS.items():
            assert "BTC" in scenario.group_shocks, f"{sid} missing BTC shock"

    def test_stablecoins_least_affected(self):
        """Stablecoins should have the smallest absolute shock in each scenario"""
        for sid, scenario in PREDEFINED_SCENARIOS.items():
            if "Stablecoins" in scenario.group_shocks:
                stable_shock = abs(scenario.group_shocks["Stablecoins"])
                btc_shock = abs(scenario.group_shocks["BTC"])
                assert stable_shock <= btc_shock, f"{sid}: stablecoins more affected than BTC"


# ---------------------------------------------------------------------------
# TestStressScenarioDataclass
# ---------------------------------------------------------------------------
class TestStressScenarioDataclass:
    def test_create_custom_scenario(self):
        scenario = StressScenario(
            id="custom",
            name="Custom Crisis",
            description="A test scenario",
            impact_min_pct=-20,
            impact_max_pct=-40,
            probability_10y=0.05,
            duration_months_min=3,
            duration_months_max=6,
            context="Test context",
            group_shocks={"BTC": -0.30, "ETH": -0.35},
        )
        assert scenario.id == "custom"
        assert scenario.group_shocks["BTC"] == -0.30


# ---------------------------------------------------------------------------
# TestStressTestResultDataclass
# ---------------------------------------------------------------------------
class TestStressTestResultDataclass:
    def test_create_result(self):
        result = StressTestResult(
            scenario_id="crisis_2008",
            scenario_name="2008 Crisis",
            scenario_description="Test",
            portfolio_loss_pct=-42.5,
            portfolio_loss_usd=-12450.0,
            portfolio_value_before=29300.0,
            portfolio_value_after=16850.0,
            group_impacts={"BTC": {"value_before": 20000, "value_after": 10000, "loss_pct": -50, "loss_usd": -10000}},
            worst_groups=[{"group": "BTC", "loss_usd": -10000}],
            best_groups=[{"group": "Stablecoins", "loss_usd": -150}],
            probability_10y=0.02,
            duration_estimate="6-12 mois",
            timestamp=datetime.now(),
        )
        assert result.portfolio_loss_pct == -42.5
        assert result.portfolio_value_after < result.portfolio_value_before


# ---------------------------------------------------------------------------
# TestGetAvailableScenarios
# ---------------------------------------------------------------------------
class TestGetAvailableScenarios:
    def test_returns_list(self):
        scenarios = get_available_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) == 6

    def test_scenario_dict_structure(self):
        scenarios = get_available_scenarios()
        for s in scenarios:
            assert "id" in s
            assert "name" in s
            assert "description" in s
            assert "impact_range" in s
            assert "min" in s["impact_range"]
            assert "max" in s["impact_range"]
            assert "probability_10y" in s
            assert "duration" in s
            assert "context" in s

    def test_ids_match_predefined(self):
        scenarios = get_available_scenarios()
        ids = {s["id"] for s in scenarios}
        assert ids == set(PREDEFINED_SCENARIOS.keys())

    def test_impact_ranges_negative(self):
        scenarios = get_available_scenarios()
        for s in scenarios:
            assert s["impact_range"]["min"] < 0
            assert s["impact_range"]["max"] < 0


# ---------------------------------------------------------------------------
# TestCalculateStressTest
# ---------------------------------------------------------------------------
class TestCalculateStressTest:
    """Test the async calculate_stress_test with mocked taxonomy"""

    def _mock_taxonomy(self):
        mock_tax = MagicMock()
        mock_tax.group_for_alias.side_effect = lambda symbol: {
            "BTC": "BTC",
            "ETH": "ETH",
            "USDT": "Stablecoins",
            "DOGE": "Memecoins",
            "SOL": "SOL",
        }.get(symbol, "Others")
        return mock_tax

    @pytest.mark.asyncio
    async def test_crisis_2008_basic(self):
        mock_tax = self._mock_taxonomy()
        holdings = [
            {"symbol": "BTC", "value_usd": 50000},
            {"symbol": "ETH", "value_usd": 30000},
            {"symbol": "USDT", "value_usd": 10000},
        ]

        with patch("services.taxonomy.Taxonomy") as MockTaxonomy:
            MockTaxonomy.load.return_value = mock_tax
            result = await calculate_stress_test(holdings, "crisis_2008")

        assert isinstance(result, StressTestResult)
        assert result.scenario_id == "crisis_2008"
        assert result.portfolio_value_before == 90000.0
        assert result.portfolio_loss_pct < 0
        assert result.portfolio_value_after < result.portfolio_value_before

    @pytest.mark.asyncio
    async def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            await calculate_stress_test([{"symbol": "BTC", "value_usd": 1000}], "nonexistent")

    @pytest.mark.asyncio
    async def test_zero_portfolio_raises(self):
        mock_tax = self._mock_taxonomy()
        with patch("services.taxonomy.Taxonomy") as MockTaxonomy:
            MockTaxonomy.load.return_value = mock_tax
            with pytest.raises(ValueError, match="zero"):
                await calculate_stress_test([{"symbol": "BTC", "value_usd": 0}], "crisis_2008")

    @pytest.mark.asyncio
    async def test_group_impacts_correct_math(self):
        """Verify that loss = value * shock for each group"""
        mock_tax = self._mock_taxonomy()
        holdings = [
            {"symbol": "BTC", "value_usd": 100000},
        ]

        with patch("services.taxonomy.Taxonomy") as MockTaxonomy:
            MockTaxonomy.load.return_value = mock_tax
            result = await calculate_stress_test(holdings, "crisis_2008")

        # BTC shock in crisis_2008 = -0.50
        btc_impact = result.group_impacts["BTC"]
        assert btc_impact["value_before"] == 100000.0
        assert btc_impact["loss_usd"] == pytest.approx(-50000.0)
        assert btc_impact["value_after"] == pytest.approx(50000.0)
        assert btc_impact["loss_pct"] == pytest.approx(-50.0)

    @pytest.mark.asyncio
    async def test_unknown_group_gets_default_shock(self):
        """Unknown asset groups get -40% default shock"""
        mock_tax = MagicMock()
        mock_tax.group_for_alias.return_value = "UnknownGroup"
        holdings = [
            {"symbol": "BTC", "value_usd": 50000},
            {"symbol": "MYSTERY", "value_usd": 50000},
        ]

        with patch("services.taxonomy.Taxonomy") as MockTaxonomy:
            MockTaxonomy.load.return_value = mock_tax
            result = await calculate_stress_test(holdings, "covid_2020")

        # Both mapped to "UnknownGroup" → default -0.40 shock
        for group_name, impact in result.group_impacts.items():
            assert impact["shock_applied"] == -0.40

    @pytest.mark.asyncio
    async def test_worst_best_groups_populated(self):
        mock_tax = self._mock_taxonomy()
        holdings = [
            {"symbol": "BTC", "value_usd": 40000},
            {"symbol": "ETH", "value_usd": 30000},
            {"symbol": "USDT", "value_usd": 20000},
            {"symbol": "DOGE", "value_usd": 10000},
        ]

        with patch("services.taxonomy.Taxonomy") as MockTaxonomy:
            MockTaxonomy.load.return_value = mock_tax
            result = await calculate_stress_test(holdings, "tether_collapse")

        assert len(result.worst_groups) <= 3
        assert len(result.best_groups) <= 3
        # Worst group should have largest negative loss_usd
        if result.worst_groups:
            assert result.worst_groups[0]["loss_usd"] < 0

    @pytest.mark.asyncio
    async def test_duration_estimate_format(self):
        mock_tax = self._mock_taxonomy()
        holdings = [{"symbol": "BTC", "value_usd": 10000}]

        with patch("services.taxonomy.Taxonomy") as MockTaxonomy:
            MockTaxonomy.load.return_value = mock_tax
            result = await calculate_stress_test(holdings, "fed_emergency")

        assert "mois" in result.duration_estimate
        assert "-" in result.duration_estimate

    @pytest.mark.asyncio
    async def test_all_scenarios_produce_valid_results(self):
        """Run all 6 scenarios and verify basic consistency"""
        mock_tax = self._mock_taxonomy()
        holdings = [
            {"symbol": "BTC", "value_usd": 50000},
            {"symbol": "ETH", "value_usd": 30000},
        ]

        for scenario_id in PREDEFINED_SCENARIOS:
            with patch("services.taxonomy.Taxonomy") as MockTaxonomy:
                MockTaxonomy.load.return_value = mock_tax
                result = await calculate_stress_test(holdings, scenario_id)

            assert result.portfolio_loss_pct < 0, f"{scenario_id} should have negative loss"
            assert result.portfolio_value_after > 0, f"{scenario_id} after-value should be positive"
            assert result.portfolio_value_before == 80000.0
