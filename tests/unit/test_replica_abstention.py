"""Causal abstention contracts for the historical SmartFolio replica."""

import pandas as pd

from services.di_backtest.trading_strategies import (
    DISmartfolioReplicaStrategy,
    ReplicaParams,
)


def _prices() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=40, freq="D")
    return pd.DataFrame(
        {
            "BTC": [100.0 + value for value in range(40)],
            "USDC": [1.0] * 40,
        },
        index=index,
    )


def test_missing_score_keeps_current_allocation():
    strategy = DISmartfolioReplicaStrategy()
    current = pd.Series({"BTC": 0.35, "USDC": 0.65})

    weights = strategy.get_weights(
        pd.Timestamp("2026-02-09"),
        _prices(),
        current,
        cycle_score=70.0,
        onchain_score=None,
        risk_score=60.0,
        di_value=55.0,
    )

    pd.testing.assert_series_equal(weights, current, check_names=False)


def test_cycle_series_never_reads_a_future_observation():
    params = ReplicaParams(enable_exposure_cap=False, enable_governance_penalty=False)
    strategy = DISmartfolioReplicaStrategy(replica_params=params)
    strategy.set_cycle_series(
        pd.Series(
            [10.0, 95.0],
            index=pd.to_datetime(["2026-01-01", "2026-02-20"]),
        )
    )
    current = pd.Series({"BTC": 0.50, "USDC": 0.50})

    weights = strategy.get_weights(
        pd.Timestamp("2026-02-09"),
        _prices(),
        current,
        onchain_score=10.0,
        risk_score=10.0,
    )

    assert weights["BTC"] == 0.20
    assert weights["USDC"] == 0.80


def test_missing_di_abstains_when_legacy_layers_require_it():
    strategy = DISmartfolioReplicaStrategy()
    current = pd.Series({"BTC": 0.40, "USDC": 0.60})

    weights = strategy.get_weights(
        pd.Timestamp("2026-02-09"),
        _prices(),
        current,
        cycle_score=80.0,
        onchain_score=70.0,
        risk_score=65.0,
    )

    pd.testing.assert_series_equal(weights, current, check_names=False)
