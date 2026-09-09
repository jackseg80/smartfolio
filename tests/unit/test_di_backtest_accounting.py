from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from services.backtesting_engine import PortfolioStrategy
from services.di_backtest.di_backtest_engine import DIBacktestEngine
from services.di_backtest.historical_di_calculator import DIHistoryPoint


def _history(prices):
    start = datetime(2025, 1, 1)
    return [
        DIHistoryPoint(
            date=start + timedelta(days=index),
            decision_index=50,
            cycle_score=50,
            onchain_score=50,
            risk_score=50,
            sentiment_score=50,
            phase="moderate",
            phase_factor=1.0,
            macro_penalty=0,
            raw_score=50,
            btc_price=price,
        )
        for index, price in enumerate(prices)
    ]


class HoldCurrentWeights(PortfolioStrategy):
    def __init__(self):
        super().__init__("Hold current weights")

    def get_weights(self, date, price_data, current_weights, **kwargs):
        return current_weights.copy()


class SwitchToBitcoinAfterFirstDay(PortfolioStrategy):
    def __init__(self):
        super().__init__("Switch to bitcoin")

    def get_weights(self, date, price_data, current_weights, **kwargs):
        if len(price_data) == 1:
            return current_weights.copy()
        return pd.Series({"BTC": 1.0, "STABLES": 0.0})


def test_weights_drift_without_implicit_free_rebalancing():
    engine = DIBacktestEngine(transaction_cost=0.0, rebalance_threshold=1.0)

    result = engine.run_backtest(
        _history([100.0, 200.0, 100.0]),
        HoldCurrentWeights(),
        initial_capital=100.0,
        rebalance_frequency="daily",
    )

    assert result.rebalance_count == 0
    assert result.final_value == pytest.approx(100.0)
    assert result.total_return == pytest.approx(0.0)


def test_performance_metrics_use_net_returns_after_transaction_costs():
    engine = DIBacktestEngine(transaction_cost=0.10, rebalance_threshold=0.0)

    result = engine.run_backtest(
        _history([100.0, 100.0, 100.0]),
        SwitchToBitcoinAfterFirstDay(),
        initial_capital=100.0,
        rebalance_frequency="daily",
    )

    expected_daily_returns = np.array([-0.05, 0.0])
    expected_volatility = np.std(expected_daily_returns) * np.sqrt(365)

    assert result.final_value == pytest.approx(95.0)
    assert result.total_return == pytest.approx(-0.05)
    assert result.volatility == pytest.approx(expected_volatility)


def test_each_altcoin_uses_its_own_price_series_instead_of_bitcoin_returns():
    history = _history([100.0, 200.0, 300.0])
    dates = [point.date for point in history]
    prices = pd.DataFrame({
        "BTC": [100.0, 200.0, 300.0],
        "SOL": [50.0, 50.0, 50.0],
    }, index=dates)
    engine = DIBacktestEngine(transaction_cost=0.0, rebalance_threshold=1.0)

    result = engine.run_backtest(
        history,
        HoldCurrentWeights(),
        initial_capital=100.0,
        asset_prices=prices,
        initial_weights={"SOL": 0.5, "STABLES": 0.5},
    )

    assert result.final_value == pytest.approx(100.0)
    assert result.benchmark_curve.iloc[-1] == pytest.approx(300.0)


def test_rejects_a_held_asset_without_its_own_price_history():
    engine = DIBacktestEngine(transaction_cost=0.0)
    history = _history([100.0, 101.0])
    prices = pd.DataFrame({"BTC": [100.0, 101.0]}, index=[point.date for point in history])

    with pytest.raises(ValueError, match="Missing price histories for initial assets"):
        engine.run_backtest(
            history,
            HoldCurrentWeights(),
            asset_prices=prices,
            initial_weights={"SOL": 0.5, "STABLES": 0.5},
        )
