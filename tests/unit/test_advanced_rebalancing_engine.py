"""
Tests for services/advanced_rebalancing.py
Covers enums, dataclasses, sync helper methods of AdvancedRebalancingEngine.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from services.advanced_rebalancing import (
    RebalancingStrategy,
    MarketMetrics,
    OptimizationConstraints,
    RebalancingResult,
    AdvancedRebalancingEngine,
)


# ── RebalancingStrategy Enum ───────────────────────────────────────

class TestRebalancingStrategy:

    def test_all_strategies(self):
        assert RebalancingStrategy.PROPORTIONAL.value == "proportional"
        assert RebalancingStrategy.RISK_PARITY.value == "risk_parity"
        assert RebalancingStrategy.MOMENTUM.value == "momentum"
        assert RebalancingStrategy.MEAN_REVERSION.value == "mean_reversion"
        assert RebalancingStrategy.MULTI_OBJECTIVE.value == "multi_objective"
        assert RebalancingStrategy.SMART_CONSOLIDATION.value == "smart_consolidation"

    def test_count(self):
        assert len(RebalancingStrategy) == 6


# ── MarketMetrics Dataclass ────────────────────────────────────────

class TestMarketMetrics:

    def test_creation(self):
        m = MarketMetrics(symbol="BTC", current_price=50000.0)
        assert m.symbol == "BTC"
        assert m.current_price == 50000.0
        assert m.volatility_30d is None
        assert m.momentum_7d is None

    def test_full_creation(self):
        m = MarketMetrics(
            symbol="ETH",
            current_price=3000.0,
            volatility_30d=0.70,
            momentum_7d=0.05,
            momentum_30d=-0.02,
            liquidity_score=85.0,
            fee_tier="maker",
        )
        assert m.volatility_30d == 0.70
        assert m.liquidity_score == 85.0

    def test_default_last_updated(self):
        m = MarketMetrics(symbol="SOL", current_price=100.0)
        assert isinstance(m.last_updated, datetime)


# ── OptimizationConstraints Dataclass ──────────────────────────────

class TestOptimizationConstraints:

    def test_defaults(self):
        c = OptimizationConstraints()
        assert c.max_trade_size_usd == 10000.0
        assert c.min_trade_size_usd == 25.0
        assert c.max_allocation_change == 0.15
        assert c.max_simultaneous_trades == 20
        assert c.preserve_staking is True
        assert c.consolidate_duplicates is True

    def test_custom(self):
        c = OptimizationConstraints(
            max_trade_size_usd=5000.0,
            min_trade_size_usd=50.0,
            preferred_exchanges=["Kraken"],
        )
        assert c.max_trade_size_usd == 5000.0
        assert c.preferred_exchanges == ["Kraken"]

    def test_default_preferred_exchanges(self):
        c = OptimizationConstraints()
        assert "Binance" in c.preferred_exchanges
        assert "Kraken" in c.preferred_exchanges


# ── RebalancingResult Dataclass ────────────────────────────────────

class TestRebalancingResult:

    def test_creation(self):
        r = RebalancingResult(
            strategy_used="proportional",
            actions=[],
            optimization_score=70.0,
            estimated_total_fees=5.0,
            risk_metrics={"diversification_score": 0.8},
            duplicate_consolidations=[],
            market_timing_score=50.0,
            execution_complexity="Low",
            warnings=[],
        )
        assert r.strategy_used == "proportional"
        assert r.optimization_score == 70.0
        assert r.execution_complexity == "Low"

    def test_default_metadata(self):
        r = RebalancingResult(
            strategy_used="test",
            actions=[],
            optimization_score=0,
            estimated_total_fees=0,
            risk_metrics={},
            duplicate_consolidations=[],
            market_timing_score=0,
            execution_complexity="Low",
            warnings=[],
        )
        assert r.metadata == {}


# ── AdvancedRebalancingEngine sync helpers ─────────────────────────

class TestAdvancedRebalancingEngine:

    @pytest.fixture
    def engine(self):
        return AdvancedRebalancingEngine()

    # -- _calculate_liquidity_score --

    def test_liquidity_btc_binance(self, engine):
        holding = {"group": "BTC", "location": "Binance", "value_usd": 10000}
        score = engine._calculate_liquidity_score(holding)
        # BTC base (95) + Binance bonus (10) = 105 → capped to 100
        assert score == 100

    def test_liquidity_others_unknown(self, engine):
        holding = {"group": "Others", "location": "Unknown", "value_usd": 5000}
        score = engine._calculate_liquidity_score(holding)
        assert score == 25  # Others base, no exchange bonus

    def test_liquidity_small_position_penalty(self, engine):
        holding = {"group": "ETH", "location": "Binance", "value_usd": 500}
        score = engine._calculate_liquidity_score(holding)
        # ETH base (90) + Binance (10) - small position penalty (-10) = 90
        assert score == 90

    def test_liquidity_very_small_position(self, engine):
        holding = {"group": "ETH", "location": "Unknown", "value_usd": 50}
        score = engine._calculate_liquidity_score(holding)
        # ETH base (90) + 0 - small penalty (-10) = 80
        assert score == 80

    def test_liquidity_score_bounded(self, engine):
        # Very low group, no exchange, tiny amount
        holding = {"group": "Memecoins", "location": "Unknown DeFi", "value_usd": 5}
        score = engine._calculate_liquidity_score(holding)
        assert 0 <= score <= 100

    # -- _calculate_execution_priority --

    def test_priority_sell_higher(self, engine):
        sell = {"action": "sell", "usd": -2000}
        buy = {"action": "buy", "usd": 2000}
        p_sell = engine._calculate_execution_priority(sell, RebalancingStrategy.PROPORTIONAL)
        p_buy = engine._calculate_execution_priority(buy, RebalancingStrategy.PROPORTIONAL)
        assert p_sell < p_buy  # Lower number = higher priority

    def test_priority_large_amount(self, engine):
        large = {"action": "buy", "usd": 5000}
        small = {"action": "buy", "usd": 50}
        p_large = engine._calculate_execution_priority(large, RebalancingStrategy.PROPORTIONAL)
        p_small = engine._calculate_execution_priority(small, RebalancingStrategy.PROPORTIONAL)
        assert p_large < p_small

    def test_priority_momentum_volatile(self, engine):
        """Volatile assets prioritized in momentum strategy."""
        volatile = {"action": "buy", "usd": 1000, "market_metrics": {"volatility": 2.0}}
        calm = {"action": "buy", "usd": 1000, "market_metrics": {"volatility": 0.5}}
        p_vol = engine._calculate_execution_priority(volatile, RebalancingStrategy.MOMENTUM)
        p_calm = engine._calculate_execution_priority(calm, RebalancingStrategy.MOMENTUM)
        assert p_vol <= p_calm

    def test_priority_illiquid_lower(self, engine):
        illiquid = {"action": "buy", "usd": 1000, "market_metrics": {"liquidity_score": 20}}
        liquid = {"action": "buy", "usd": 1000, "market_metrics": {"liquidity_score": 80}}
        p_ill = engine._calculate_execution_priority(illiquid, RebalancingStrategy.PROPORTIONAL)
        p_liq = engine._calculate_execution_priority(liquid, RebalancingStrategy.PROPORTIONAL)
        assert p_ill > p_liq

    def test_priority_bounded(self, engine):
        action = {"action": "sell", "usd": -50000, "market_metrics": {"volatility": 5.0, "liquidity_score": 10}}
        p = engine._calculate_execution_priority(action, RebalancingStrategy.MOMENTUM)
        assert 1 <= p <= 10

    # -- _split_large_order --

    def test_split_basic(self, engine):
        engine.constraints = OptimizationConstraints(max_trade_size_usd=5000)
        action = {"usd": 12000, "est_quantity": 0.24, "symbol": "BTC"}
        splits = engine._split_large_order(action)
        assert len(splits) == 3  # 12000 / 5000 = 2.4 → ceil = 3
        total = sum(s["usd"] for s in splits)
        assert total == pytest.approx(12000, abs=1)

    def test_split_sell_negative_usd(self, engine):
        engine.constraints = OptimizationConstraints(max_trade_size_usd=5000)
        action = {"usd": -15000, "symbol": "ETH"}
        splits = engine._split_large_order(action)
        assert all(s["usd"] < 0 for s in splits)

    def test_split_info_metadata(self, engine):
        engine.constraints = OptimizationConstraints(max_trade_size_usd=5000)
        action = {"usd": 10500, "symbol": "BTC"}
        splits = engine._split_large_order(action)
        for i, split in enumerate(splits):
            assert split["split_info"]["is_split"] is True
            assert split["split_info"]["split_index"] == i + 1
            assert split["split_info"]["total_splits"] == len(splits)
            assert split["split_info"]["original_amount"] == 10500

    def test_split_execution_delays(self, engine):
        engine.constraints = OptimizationConstraints(max_trade_size_usd=5000)
        action = {"usd": 20000, "symbol": "SOL"}
        splits = engine._split_large_order(action)
        for i, split in enumerate(splits):
            assert split["execution_delay_minutes"] == i * 5

    def test_no_split_under_limit(self, engine):
        """Order under max should produce 1 split."""
        engine.constraints = OptimizationConstraints(max_trade_size_usd=10000)
        action = {"usd": 5000, "symbol": "BTC"}
        splits = engine._split_large_order(action)
        assert len(splits) == 1

    # -- _calculate_optimization_score --

    def test_optimization_score_empty_actions(self, engine):
        assert engine._calculate_optimization_score([], RebalancingStrategy.PROPORTIONAL) == 0.0

    def test_optimization_score_base(self, engine):
        actions = [{"usd": 1000, "exchange_hint": "binance"}]
        score = engine._calculate_optimization_score(actions, RebalancingStrategy.PROPORTIONAL)
        assert score >= 70  # Base is 70

    def test_optimization_score_multi_objective_bonus(self, engine):
        actions = [{"usd": 1000, "exchange_hint": "binance"}]
        score_prop = engine._calculate_optimization_score(actions, RebalancingStrategy.PROPORTIONAL)
        score_mo = engine._calculate_optimization_score(actions, RebalancingStrategy.MULTI_OBJECTIVE)
        assert score_mo > score_prop

    def test_optimization_score_small_actions_penalty(self, engine):
        # Many small actions → penalty
        small_actions = [{"usd": 20, "exchange_hint": "unknown"} for _ in range(10)]
        score = engine._calculate_optimization_score(small_actions, RebalancingStrategy.PROPORTIONAL)
        # Should be lower due to small actions penalty
        normal_actions = [{"usd": 1000, "exchange_hint": "binance"} for _ in range(3)]
        score_normal = engine._calculate_optimization_score(normal_actions, RebalancingStrategy.PROPORTIONAL)
        assert score < score_normal

    def test_optimization_score_bounded(self, engine):
        actions = [{"usd": 100, "exchange_hint": "binance", "split_info": {"is_split": True}} for _ in range(5)]
        score = engine._calculate_optimization_score(actions, RebalancingStrategy.RISK_PARITY)
        assert 0 <= score <= 100

    # -- _calculate_portfolio_risk_metrics --

    def test_risk_metrics_empty(self, engine):
        assert engine._calculate_portfolio_risk_metrics([]) == {}

    def test_risk_metrics_single_holding(self, engine):
        holdings = [{"symbol": "BTC", "value_usd": 10000}]
        result = engine._calculate_portfolio_risk_metrics(holdings)
        assert "diversification_score" in result
        assert "concentration_risk" in result
        # Single holding → HHI = 1.0, entropy normalized = 0 (but log(1)=0 → NaN risk)

    def test_risk_metrics_diversified(self, engine):
        holdings = [
            {"symbol": "BTC", "value_usd": 5000},
            {"symbol": "ETH", "value_usd": 3000},
            {"symbol": "SOL", "value_usd": 2000},
        ]
        result = engine._calculate_portfolio_risk_metrics(holdings)
        assert result["diversification_score"] > 0
        assert result["concentration_risk"] < 1.0

    # -- _calculate_market_timing_score --

    def test_timing_empty(self, engine):
        assert engine._calculate_market_timing_score([], RebalancingStrategy.PROPORTIONAL) == 50.0

    def test_timing_momentum_good(self, engine):
        """Buying assets with positive momentum → good timing."""
        actions = [
            {"usd": 1000, "market_metrics": {"momentum_7d": 0.05}},
            {"usd": 500, "market_metrics": {"momentum_7d": 0.03}},
        ]
        score = engine._calculate_market_timing_score(actions, RebalancingStrategy.MOMENTUM)
        assert score > 50

    def test_timing_momentum_bad(self, engine):
        """Buying assets with negative momentum → neutral timing."""
        actions = [
            {"usd": 1000, "market_metrics": {"momentum_7d": -0.05}},
        ]
        score = engine._calculate_market_timing_score(actions, RebalancingStrategy.MOMENTUM)
        assert score == 50  # No bonus

    def test_timing_mean_reversion_buy_dip(self, engine):
        """Mean reversion: buying negative momentum = good timing."""
        actions = [
            {"usd": 1000, "market_metrics": {"momentum_7d": -0.05}},
            {"usd": 500, "market_metrics": {"momentum_7d": -0.03}},
        ]
        score = engine._calculate_market_timing_score(actions, RebalancingStrategy.MEAN_REVERSION)
        assert score > 50

    def test_timing_proportional_neutral(self, engine):
        actions = [{"usd": 1000, "market_metrics": {"momentum_7d": 0.05}}]
        score = engine._calculate_market_timing_score(actions, RebalancingStrategy.PROPORTIONAL)
        assert score == 50.0  # No timing adjustment for proportional

    # -- _assess_execution_complexity --

    def test_complexity_empty(self, engine):
        assert engine._assess_execution_complexity([]) == "Low"

    def test_complexity_few_actions(self, engine):
        actions = [{"usd": 500, "exchange_hint": "binance"}] * 5
        assert engine._assess_execution_complexity(actions) == "Low"

    def test_complexity_many_actions(self, engine):
        actions = [{"usd": 500, "exchange_hint": "binance"}] * 25
        result = engine._assess_execution_complexity(actions)
        assert result in ("Medium", "High")

    def test_complexity_many_splits(self, engine):
        actions = [{"usd": 500, "split_info": {"is_split": True}}] * 8
        result = engine._assess_execution_complexity(actions)
        assert result in ("Medium", "High")

    def test_complexity_many_exchanges(self, engine):
        actions = [
            {"usd": 500, "exchange_hint": f"exchange_{i}"}
            for i in range(5)
        ]
        result = engine._assess_execution_complexity(actions)
        assert result in ("Medium", "High")

    def test_complexity_large_orders(self, engine):
        actions = [{"usd": 8000, "exchange_hint": "binance"}] * 5
        result = engine._assess_execution_complexity(actions)
        assert result in ("Medium", "High")

    # -- _generate_warnings --

    def test_warnings_empty_actions(self, engine):
        warnings = engine._generate_warnings([], 0, "Low")
        assert "Aucune action" in warnings[0]

    def test_warnings_high_fees(self, engine):
        actions = [{"usd": 1000}] * 3
        total_volume = 3000
        high_fees = total_volume * 0.01  # 1% fees
        warnings = engine._generate_warnings(actions, high_fees, "Low")
        assert any("Frais" in w for w in warnings)

    def test_warnings_high_complexity(self, engine):
        actions = [{"usd": 1000}]
        warnings = engine._generate_warnings(actions, 1.0, "High")
        assert any("complexe" in w.lower() or "Exécution" in w for w in warnings)

    def test_warnings_small_actions(self, engine):
        actions = [{"usd": 20}] * 10  # All small
        warnings = engine._generate_warnings(actions, 0.1, "Low")
        assert any("petites" in w for w in warnings)

    def test_warnings_illiquid_assets(self, engine):
        actions = [
            {"usd": 1000, "market_metrics": {"liquidity_score": 15}, "symbol": "SHIB"},
        ]
        warnings = engine._generate_warnings(actions, 1.0, "Low")
        assert any("illiquide" in w.lower() for w in warnings)

    def test_warnings_large_orders(self, engine):
        actions = [
            {"usd": 8000, "market_metrics": {"liquidity_score": 80}},
        ]
        warnings = engine._generate_warnings(actions, 8.0, "Low")
        assert any("gros" in w.lower() for w in warnings)

    # -- Strategy params --

    def test_default_strategy_params(self, engine):
        assert "risk_parity" in engine.strategy_params
        assert "momentum" in engine.strategy_params
        assert "mean_reversion" in engine.strategy_params

    def test_risk_parity_params(self, engine):
        params = engine.strategy_params["risk_parity"]
        assert params["target_vol"] == 0.20
        assert params["min_vol"] == 0.05

    def test_momentum_params(self, engine):
        params = engine.strategy_params["momentum"]
        assert params["short_period"] == 7
        assert params["long_period"] == 30

    # -- _estimate_volatility / _estimate_momentum --

    @patch("services.advanced_rebalancing.Taxonomy")
    def test_estimate_volatility_btc(self, mock_taxonomy, engine):
        mock_tax = MagicMock()
        mock_tax.group_for_alias.return_value = "BTC"
        mock_taxonomy.load.return_value = mock_tax

        import random
        random.seed(42)
        vol = engine._estimate_volatility("BTC")
        assert 0.4 < vol < 1.0  # BTC base 0.60 * noise (0.8-1.2)

    @patch("services.advanced_rebalancing.Taxonomy")
    def test_estimate_volatility_stablecoin(self, mock_taxonomy, engine):
        mock_tax = MagicMock()
        mock_tax.group_for_alias.return_value = "Stablecoins"
        mock_taxonomy.load.return_value = mock_tax

        import random
        random.seed(42)
        vol = engine._estimate_volatility("USDC")
        assert vol < 0.10  # Stablecoin base 0.05 * noise

    @patch("services.advanced_rebalancing.Taxonomy")
    def test_estimate_momentum(self, mock_taxonomy, engine):
        mock_tax = MagicMock()
        mock_tax.group_for_alias.return_value = "BTC"
        mock_taxonomy.load.return_value = mock_tax

        import random
        random.seed(42)
        mom = engine._estimate_momentum("BTC", 30)
        assert isinstance(mom, float)
        assert -0.20 < mom < 0.20  # Reasonable range
