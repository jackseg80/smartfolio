"""Tests for services/backtesting_engine.py — strategies, engine helpers, metrics."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from services.backtesting_engine import (
    RebalanceFrequency,
    BacktestMetric,
    TransactionCosts,
    BacktestConfig,
    EqualWeightStrategy,
    MarketCapStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    RiskParityStrategy,
    BacktestingEngine,
    backtesting_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(days=100, assets=None):
    """Cree un DataFrame de prix simule."""
    if assets is None:
        assets = ["BTC", "ETH", "SOL"]
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    data = {}
    for asset in assets:
        # Random walk positif
        returns = np.random.normal(0.001, 0.02, days)
        prices = 100 * np.cumprod(1 + returns)
        data[asset] = prices
    return pd.DataFrame(data, index=dates)


def _default_config(days=100):
    start = "2024-01-01"
    end = (pd.Timestamp(start) + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")
    return BacktestConfig(start_date=start, end_date=end)


# ===========================================================================
# Enums
# ===========================================================================

class TestEnums:
    def test_rebalance_frequency_values(self):
        assert RebalanceFrequency.DAILY.value == "daily"
        assert RebalanceFrequency.WEEKLY.value == "weekly"
        assert RebalanceFrequency.MONTHLY.value == "monthly"
        assert RebalanceFrequency.QUARTERLY.value == "quarterly"
        assert RebalanceFrequency.BIWEEKLY.value == "biweekly"

    def test_backtest_metric_values(self):
        assert BacktestMetric.TOTAL_RETURN.value == "total_return"
        assert BacktestMetric.SHARPE_RATIO.value == "sharpe_ratio"
        assert BacktestMetric.MAX_DRAWDOWN.value == "max_drawdown"
        assert BacktestMetric.TRANSACTION_COSTS.value == "transaction_costs"


# ===========================================================================
# Dataclasses
# ===========================================================================

class TestTransactionCosts:
    def test_defaults(self):
        tc = TransactionCosts()
        assert tc.maker_fee == 0.001
        assert tc.taker_fee == 0.0015
        assert tc.slippage_bps == 5.0
        assert tc.min_trade_size == 10.0

    def test_custom(self):
        tc = TransactionCosts(maker_fee=0.002, taker_fee=0.003, slippage_bps=10.0, min_trade_size=50.0)
        assert tc.taker_fee == 0.003
        assert tc.min_trade_size == 50.0


class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig(start_date="2024-01-01", end_date="2024-12-31")
        assert cfg.initial_capital == 10000.0
        assert cfg.rebalance_frequency == RebalanceFrequency.MONTHLY
        assert cfg.benchmark == "BTC"
        assert cfg.risk_free_rate == 0.02
        assert cfg.max_position_size == 0.5


# ===========================================================================
# Strategies — get_weights
# ===========================================================================

class TestEqualWeightStrategy:
    def test_name(self):
        s = EqualWeightStrategy()
        assert s.name == "Equal Weight"

    def test_weights_sum_to_one(self):
        s = EqualWeightStrategy()
        df = _make_price_df(30)
        w = s.get_weights(df.index[10], df, pd.Series(0.0, index=df.columns))
        assert abs(w.sum() - 1.0) < 1e-10

    def test_weights_equal(self):
        s = EqualWeightStrategy()
        df = _make_price_df(30, assets=["A", "B", "C", "D"])
        w = s.get_weights(df.index[5], df, pd.Series(0.0, index=df.columns))
        for v in w.values:
            assert abs(v - 0.25) < 1e-10


class TestMarketCapStrategy:
    def test_name(self):
        s = MarketCapStrategy()
        assert s.name == "Market Cap Weighted"

    def test_weights_sum_to_one(self):
        s = MarketCapStrategy()
        df = _make_price_df(30)
        w = s.get_weights(df.index[10], df, pd.Series(0.0, index=df.columns))
        assert abs(w.sum() - 1.0) < 1e-10

    def test_higher_price_higher_weight(self):
        """L'asset au prix le plus eleve doit avoir le poids le plus eleve."""
        s = MarketCapStrategy()
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({"A": [100]*10, "B": [200]*10, "C": [50]*10}, index=dates)
        w = s.get_weights(dates[5], df, pd.Series(0.0, index=df.columns))
        assert w["B"] > w["A"] > w["C"]


class TestMomentumStrategy:
    def test_name_includes_lookback(self):
        s = MomentumStrategy(lookback_days=60)
        assert "60" in s.name

    def test_weights_sum_to_one(self):
        s = MomentumStrategy(lookback_days=30)
        df = _make_price_df(60)
        w = s.get_weights(df.index[40], df, pd.Series(0.0, index=df.columns))
        assert abs(w.sum() - 1.0) < 1e-10

    def test_fallback_when_insufficient_data(self):
        """Si pas assez de donnees, fallback equal weight."""
        s = MomentumStrategy(lookback_days=90)
        df = _make_price_df(10)
        w = s.get_weights(df.index[0], df, pd.Series(0.0, index=df.columns))
        # Should fall back to equal weights
        assert abs(w.sum() - 1.0) < 1e-10


class TestMeanReversionStrategy:
    def test_name_includes_lookback(self):
        s = MeanReversionStrategy(lookback_days=20)
        assert "20" in s.name

    def test_weights_sum_to_one(self):
        s = MeanReversionStrategy(lookback_days=15)
        df = _make_price_df(60)
        w = s.get_weights(df.index[30], df, pd.Series(0.0, index=df.columns))
        assert abs(w.sum() - 1.0) < 1e-10


class TestRiskParityStrategy:
    def test_name_includes_lookback(self):
        s = RiskParityStrategy(vol_lookback=45)
        assert "45" in s.name

    def test_weights_sum_to_one(self):
        s = RiskParityStrategy(vol_lookback=20)
        df = _make_price_df(60)
        w = s.get_weights(df.index[30], df, pd.Series(0.0, index=df.columns))
        assert abs(w.sum() - 1.0) < 1e-10

    def test_lower_vol_gets_higher_weight(self):
        """L'asset le moins volatile doit avoir le poids le plus eleve (inverse vol)."""
        s = RiskParityStrategy(vol_lookback=20)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        np.random.seed(0)
        # A = low vol, B = high vol
        a_prices = 100 + np.cumsum(np.random.normal(0, 0.1, 30))
        b_prices = 100 + np.cumsum(np.random.normal(0, 2.0, 30))
        df = pd.DataFrame({"A": a_prices, "B": b_prices}, index=dates)
        w = s.get_weights(dates[25], df, pd.Series(0.0, index=df.columns))
        assert w["A"] > w["B"]


# ===========================================================================
# Engine — Helper Methods
# ===========================================================================

class TestApplyPositionLimits:
    def setup_method(self):
        self.engine = BacktestingEngine()

    def test_no_cap_needed(self):
        weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
        result = self.engine._apply_position_limits(weights, max_position=0.5)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_cap_applied(self):
        weights = pd.Series({"A": 0.8, "B": 0.1, "C": 0.1})
        result = self.engine._apply_position_limits(weights, max_position=0.5)
        assert result["A"] <= 0.5 + 1e-10

    def test_renormalized(self):
        weights = pd.Series({"A": 0.6, "B": 0.6})  # sum > 1
        result = self.engine._apply_position_limits(weights, max_position=0.5)
        assert abs(result.sum() - 1.0) < 1e-10


class TestCalculateTrades:
    def setup_method(self):
        self.engine = BacktestingEngine()
        self.tc = TransactionCosts()

    def test_no_trade_needed(self):
        current = pd.Series({"A": 0.5, "B": 0.5})
        target = pd.Series({"A": 0.5, "B": 0.5})
        prices = pd.Series({"A": 100.0, "B": 200.0})
        trades = self.engine._calculate_trades(current, target, 10000, prices, self.tc)
        assert len(trades) == 0

    def test_trade_generated(self):
        current = pd.Series({"A": 0.5, "B": 0.5})
        target = pd.Series({"A": 0.8, "B": 0.2})
        prices = pd.Series({"A": 100.0, "B": 200.0})
        trades = self.engine._calculate_trades(current, target, 10000, prices, self.tc)
        assert len(trades) == 2
        actions = {t["asset"]: t["action"] for t in trades}
        assert actions["A"] == "buy"
        assert actions["B"] == "sell"

    def test_small_trade_filtered(self):
        """Trades < min_trade_size sont ignores."""
        current = pd.Series({"A": 0.5000, "B": 0.5000})
        target = pd.Series({"A": 0.5005, "B": 0.4995})  # ~$5 diff on 10k < min $10
        prices = pd.Series({"A": 100.0, "B": 200.0})
        trades = self.engine._calculate_trades(current, target, 10000, prices, self.tc)
        assert len(trades) == 0

    def test_trade_cost_positive(self):
        current = pd.Series({"A": 0.0})
        target = pd.Series({"A": 1.0})
        prices = pd.Series({"A": 100.0})
        trades = self.engine._calculate_trades(current, target, 10000, prices, self.tc)
        assert len(trades) == 1
        assert trades[0]["cost"] > 0


class TestExecuteTrades:
    def setup_method(self):
        self.engine = BacktestingEngine()
        self.tc = TransactionCosts()

    def test_summary(self):
        trades = [
            {"asset": "A", "action": "buy", "quantity": 10, "price": 100, "value": 1000, "cost": 2.0},
            {"asset": "B", "action": "sell", "quantity": 5, "price": 200, "value": 1000, "cost": 2.0},
        ]
        result = self.engine._execute_trades(trades, 10000, self.tc)
        assert result["total_cost"] == 4.0
        assert result["total_volume"] == 2000
        assert abs(result["cost_ratio"] - 0.0004) < 1e-10


class TestGetRebalancingDates:
    def setup_method(self):
        self.engine = BacktestingEngine()

    def test_daily(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        result = self.engine._get_rebalancing_dates(dates, RebalanceFrequency.DAILY)
        assert len(result) == 30

    def test_weekly(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        result = self.engine._get_rebalancing_dates(dates, RebalanceFrequency.WEEKLY)
        assert len(result) > 0
        assert len(result) < 60

    def test_monthly(self):
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        result = self.engine._get_rebalancing_dates(dates, RebalanceFrequency.MONTHLY)
        assert len(result) > 0
        assert len(result) < 30


# ===========================================================================
# Engine — Metrics
# ===========================================================================

class TestCalculateMetrics:
    def setup_method(self):
        self.engine = BacktestingEngine()

    def test_basic_metrics(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        values = 10000 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
        portfolio = pd.Series(values, index=dates)
        returns = portfolio.pct_change().dropna()
        cfg = _default_config(100)
        metrics = self.engine._calculate_metrics(portfolio, returns, cfg)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics

    def test_empty_returns(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        portfolio = pd.Series([10000, 10100], index=dates)
        returns = pd.Series([], dtype=float)
        cfg = _default_config(2)
        metrics = self.engine._calculate_metrics(portfolio, returns, cfg)
        assert metrics == {}


class TestCalculateRiskMetrics:
    def setup_method(self):
        self.engine = BacktestingEngine()

    def test_basic_risk(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        benchmark = pd.Series(np.random.normal(0.0005, 0.025, 100))
        cfg = _default_config(100)
        risk = self.engine._calculate_risk_metrics(returns, benchmark, cfg)
        assert "var_95" in risk
        assert "var_99" in risk
        assert "beta" in risk
        assert "tracking_error" in risk

    def test_empty_returns(self):
        returns = pd.Series([], dtype=float)
        benchmark = pd.Series([], dtype=float)
        cfg = _default_config(2)
        risk = self.engine._calculate_risk_metrics(returns, benchmark, cfg)
        assert risk == {}


# ===========================================================================
# Engine — Full backtest (integration)
# ===========================================================================

class TestRunBacktest:
    def setup_method(self):
        self.engine = BacktestingEngine()

    def test_equal_weight_backtest(self):
        df = _make_price_df(100)
        cfg = _default_config(100)
        result = self.engine.run_backtest(df, "equal_weight", cfg)
        assert result.summary["strategy_name"] == "Equal Weight"
        assert result.summary["final_value"] > 0
        assert len(result.portfolio_value) == 100

    def test_unknown_strategy_raises(self):
        df = _make_price_df(100)
        cfg = _default_config(100)
        with pytest.raises(ValueError, match="Unknown strategy"):
            self.engine.run_backtest(df, "nonexistent", cfg)

    def test_insufficient_data_raises(self):
        df = _make_price_df(10)
        cfg = BacktestConfig(start_date="2024-01-01", end_date="2024-01-10")
        with pytest.raises(ValueError, match="Insufficient data"):
            self.engine.run_backtest(df, "equal_weight", cfg)

    def test_add_custom_strategy(self):
        self.engine.add_strategy("custom_eq", EqualWeightStrategy())
        assert "custom_eq" in self.engine.strategies

    def test_compare_strategies(self):
        df = _make_price_df(100)
        cfg = _default_config(100)
        result = self.engine.compare_strategies(df, ["equal_weight", "risk_parity"], cfg)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_momentum_backtest(self):
        df = _make_price_df(100)
        cfg = _default_config(100)
        result = self.engine.run_backtest(df, "momentum_90d", cfg)
        assert result.summary["final_value"] > 0

    def test_mean_reversion_backtest(self):
        df = _make_price_df(100)
        cfg = _default_config(100)
        result = self.engine.run_backtest(df, "mean_reversion", cfg)
        assert result.summary["final_value"] > 0


class TestGlobalInstance:
    def test_backtesting_engine_instance(self):
        assert isinstance(backtesting_engine, BacktestingEngine)

    def test_default_strategies_registered(self):
        assert "equal_weight" in backtesting_engine.strategies
        assert "market_cap" in backtesting_engine.strategies
        assert "momentum_90d" in backtesting_engine.strategies
        assert "momentum_30d" in backtesting_engine.strategies
        assert "mean_reversion" in backtesting_engine.strategies
        assert "risk_parity" in backtesting_engine.strategies
