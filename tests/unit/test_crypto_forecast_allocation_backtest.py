import pandas as pd
import pytest

from services.forecasting.allocation_backtest import (
    AllocationPolicy,
    ConfirmationState,
    _drift_weights,
    build_target_weights,
    capped_inverse_volatility_weights,
    simulate_policy,
    simulate_static_benchmark,
)


def test_confirmation_state_requires_entry_and_exit_counts():
    state = ConfirmationState(5, 2)
    assert [state.update(True) for _ in range(4)] == [None] * 4
    assert state.update(True) is True
    assert state.update(False) is True
    assert state.update(False) is False


def test_capped_inverse_volatility_weights_preserve_cap_and_budget():
    weights = capped_inverse_volatility_weights({"A": 0.1, "B": 0.2, "C": 0.4}, 0.25, 0.10)

    assert sum(weights.values()) == pytest.approx(0.25)
    assert max(weights.values()) <= 0.10 + 1e-12
    assert weights["A"] >= weights["B"] >= weights["C"]


def test_target_weights_preserve_budget_without_negative_positions():
    policy = AllocationPolicy("test", 0.2, 0.8, "risky_60")
    weights = build_target_weights(
        policy,
        True,
        ["SOL", "ADA", "LINK"],
        {"SOL": 0.5, "ADA": 0.6, "LINK": 0.7},
        3,
        btc_eth_split={"BTC": 0.6, "ETH": 0.4},
        altcoin_weight_cap=0.1,
    )

    assert weights is not None
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(weight >= 0.0 for weight in weights.values())
    assert (
        sum(weight for symbol, weight in weights.items() if symbol not in {"BTC", "ETH", "CASH"})
        <= 0.3 + 1e-12
    )


def test_price_round_trip_without_trading_returns_to_initial_capital():
    weights = {"BTC": 0.5, "CASH": 0.5}
    first_weights, first_growth = _drift_weights(weights, {"BTC": 1.0})
    _, second_growth = _drift_weights(first_weights, {"BTC": -0.5})

    assert first_growth * second_growth == pytest.approx(1.0)


def test_signal_executes_after_next_close_return_not_before_it():
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"BTC": [100.0, 200.0, 100.0], "ETH": [100.0] * 3}, index=dates)
    features = pd.DataFrame({"BTC|distance_sma_200d": [1.0] * 3}, index=dates)
    policy = AllocationPolicy("test", 1.0, 1.0, "none")
    config = {
        "market_entry_confirmations": 1,
        "market_exit_confirmations": 1,
        "rotation_entry_confirmations": 1,
        "rotation_exit_confirmations": 1,
        "btc_eth_split": {"BTC": 1.0, "ETH": 0.0},
        "altcoin_weight_cap": 0.1,
        "cost_per_traded_amount": 0.0,
        "annualization_days": 365,
    }

    metrics, _ = simulate_policy(
        dates[:2],
        prices,
        features,
        policy=policy,
        variant="reference",
        selected_predictions={},
        config=config,
        cost_multiplier=1.0,
    )

    assert metrics["total_return"] == pytest.approx(-0.5)


def test_static_benchmark_deploys_at_next_close_then_holds_quantities():
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"BTC": [100.0, 200.0, 100.0]}, index=dates)
    config = {"cost_per_traded_amount": 0.0, "annualization_days": 365}

    metrics, daily = simulate_static_benchmark(
        dates[:2],
        prices,
        target_weights={"BTC": 1.0, "CASH": 0.0},
        config=config,
        cost_multiplier=1.0,
    )

    assert metrics["total_return"] == pytest.approx(-0.5)
    assert metrics["turnover"] == pytest.approx(1.0)
    assert daily.iloc[0]["net_return"] == pytest.approx(0.0)


def test_static_cash_benchmark_has_no_cost_or_risk():
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"BTC": [100.0, 200.0, 100.0]}, index=dates)
    config = {"cost_per_traded_amount": 0.003, "annualization_days": 365}

    metrics, _ = simulate_static_benchmark(
        dates[:2],
        prices,
        target_weights={"CASH": 1.0},
        config=config,
        cost_multiplier=2.0,
    )

    assert metrics["total_return"] == pytest.approx(0.0)
    assert metrics["cost_paid"] == pytest.approx(0.0)
    assert metrics["average_risk_exposure"] == pytest.approx(0.0)


def test_policy_only_rebalances_on_configured_review_days():
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"BTC": [100.0] * 5, "ETH": [100.0] * 5}, index=dates)
    features = pd.DataFrame({"BTC|distance_sma_200d": [1.0] * 5}, index=dates)
    config = {
        "market_entry_confirmations": 1,
        "market_exit_confirmations": 1,
        "rotation_entry_confirmations": 1,
        "rotation_exit_confirmations": 1,
        "btc_eth_split": {"BTC": 1.0, "ETH": 0.0},
        "altcoin_weight_cap": 0.1,
        "cost_per_traded_amount": 0.0,
        "annualization_days": 365,
        "rebalance_interval_days": 3,
    }

    metrics, daily = simulate_policy(
        dates[:4],
        prices,
        features,
        policy=AllocationPolicy("test", 0.2, 0.8, "none"),
        variant="reference",
        selected_predictions={},
        config=config,
        cost_multiplier=1.0,
    )

    assert metrics["rebalance_days"] == 1
    assert daily["review_due"].tolist() == [True, False, False, True]


def test_hybrid_uses_reference_market_when_no_altcoins_exist():
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"BTC": [100.0, 100.0, 110.0], "ETH": [100.0] * 3}, index=dates)
    features = pd.DataFrame({"BTC|distance_sma_200d": [1.0] * 3}, index=dates)
    config = {
        "market_entry_confirmations": 1,
        "market_exit_confirmations": 1,
        "rotation_entry_confirmations": 1,
        "rotation_exit_confirmations": 1,
        "btc_eth_split": {"BTC": 1.0, "ETH": 0.0},
        "altcoin_weight_cap": 0.1,
        "cost_per_traded_amount": 0.0,
        "annualization_days": 365,
        "rebalance_interval_days": 1,
    }

    reference, _ = simulate_policy(
        dates[:2],
        prices,
        features,
        policy=AllocationPolicy("test", 1.0, 1.0, "none"),
        variant="reference",
        selected_predictions={},
        config=config,
        cost_multiplier=1.0,
    )
    hybrid, _ = simulate_policy(
        dates[:2],
        prices,
        features,
        policy=AllocationPolicy("test", 1.0, 1.0, "none"),
        variant="hybrid",
        selected_predictions={},
        config=config,
        cost_multiplier=1.0,
    )

    assert hybrid["total_return"] == pytest.approx(reference["total_return"])
