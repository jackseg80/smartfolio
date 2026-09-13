"""Causal allocation comparison using next-close execution and explicit costs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ALLOCATION_SCHEMA_VERSION = "crypto-forecast-allocation-backtest-v1"
CASH = "CASH"


@dataclass(frozen=True)
class AllocationPolicy:
    name: str
    risk_low: float
    risk_high: float
    alt_mode: str


class ConfirmationState:
    """Two-sided confirmation counter with an explicit uninitialized state."""

    def __init__(self, entry_confirmations: int, exit_confirmations: int) -> None:
        self.entry_confirmations = entry_confirmations
        self.exit_confirmations = exit_confirmations
        self.state: bool | None = None
        self._positive = 0
        self._negative = 0

    def update(self, favorable: bool | None) -> bool | None:
        if favorable is None:
            self._positive = 0
            self._negative = 0
            return self.state
        if favorable:
            self._positive += 1
            self._negative = 0
            if self._positive >= self.entry_confirmations:
                self.state = True
        else:
            self._negative += 1
            self._positive = 0
            if self._negative >= self.exit_confirmations:
                self.state = False
        return self.state


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_forecast_artifact(directory: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    results_path = root / str(manifest["results_file"])
    predictions_path = root / str(manifest["predictions_file"])
    if file_sha256(results_path) != manifest["results_sha256"]:
        raise ValueError("Forecast results SHA-256 mismatch")
    if file_sha256(predictions_path) != manifest["predictions_sha256"]:
        raise ValueError("Forecast predictions SHA-256 mismatch")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    predictions["decision_date"] = pd.to_datetime(predictions["decision_date"]).dt.normalize()
    return results, predictions


def _selected_predictions(
    results: Mapping[str, Any],
    predictions: pd.DataFrame,
    experiment_name: str,
    horizon: int,
    task: str,
) -> pd.DataFrame:
    experiment = next(
        item
        for item in results["experiments"]
        if item["name"] == experiment_name and int(item["horizon_days"]) == horizon
    )
    if experiment["status"] != "complete":
        raise ValueError(f"Required forecast is unavailable: {experiment_name} {horizon}d")
    selection_key = "regression_selection" if task == "regression" else "classification_selection"
    selected_model = experiment["development"][selection_key]["selected_model"]
    selected = predictions[
        (predictions["experiment"] == experiment_name)
        & (predictions["horizon_days"] == horizon)
        & (predictions["task"] == task)
        & (predictions["model"] == selected_model)
    ].copy()
    if selected.empty:
        raise ValueError(f"No selected predictions for {experiment_name} {horizon}d {task}")
    return selected


def common_prediction_blocks(
    results: Mapping[str, Any], predictions: pd.DataFrame, minimum_days: int
) -> tuple[list[pd.DatetimeIndex], dict[str, pd.DataFrame]]:
    selected = {
        "market_return_30": _selected_predictions(
            results,
            predictions,
            "btc_absolute_return",
            30,
            "classification_expected_return",
        ),
        "market_probability_30": _selected_predictions(
            results, predictions, "btc_absolute_return", 30, "classification"
        ),
        "market_return_7": _selected_predictions(
            results,
            predictions,
            "btc_absolute_return",
            7,
            "classification_expected_return",
        ),
        "market_probability_7": _selected_predictions(
            results, predictions, "btc_absolute_return", 7, "classification"
        ),
        "asset_return_30": _selected_predictions(
            results,
            predictions,
            "asset_rotation_relative_to_btc",
            30,
            "classification_expected_return",
        ),
        "asset_probability_30": _selected_predictions(
            results, predictions, "asset_rotation_relative_to_btc", 30, "classification"
        ),
    }
    date_sets = [set(frame["decision_date"]) for frame in selected.values()]
    common = sorted(set.intersection(*date_sets))
    blocks: list[list[pd.Timestamp]] = []
    for date in common:
        if not blocks or date - blocks[-1][-1] != pd.Timedelta(days=1):
            blocks.append([date])
        else:
            blocks[-1].append(date)
    return [pd.DatetimeIndex(block) for block in blocks if len(block) >= minimum_days], selected


def capped_inverse_volatility_weights(
    volatilities: Mapping[str, float], total_weight: float, per_asset_cap: float
) -> dict[str, float]:
    if total_weight < -1e-12 or per_asset_cap <= 0:
        raise ValueError("Weights and caps must be non-negative")
    eligible = {
        symbol: 1.0 / float(volatility)
        for symbol, volatility in volatilities.items()
        if math.isfinite(float(volatility)) and float(volatility) > 0.0
    }
    capacity = len(eligible) * per_asset_cap
    remaining = min(max(total_weight, 0.0), capacity)
    weights = {symbol: 0.0 for symbol in eligible}
    active = set(eligible)
    while active and remaining > 1e-12:
        denominator = sum(eligible[symbol] for symbol in active)
        saturated: list[str] = []
        for symbol in sorted(active):
            proposed = remaining * eligible[symbol] / denominator
            available = per_asset_cap - weights[symbol]
            if proposed >= available - 1e-12:
                weights[symbol] += available
                saturated.append(symbol)
        if saturated:
            remaining = min(max(total_weight, 0.0), capacity) - sum(weights.values())
            active.difference_update(saturated)
            continue
        for symbol in active:
            weights[symbol] += remaining * eligible[symbol] / denominator
        remaining = 0.0
    return weights


def build_target_weights(
    policy: AllocationPolicy,
    market_favorable: bool | None,
    favorable_alts: Sequence[str],
    alt_volatilities: Mapping[str, float],
    total_alt_universe: int,
    *,
    btc_eth_split: Mapping[str, float],
    altcoin_weight_cap: float,
) -> dict[str, float] | None:
    if market_favorable is None:
        return None
    risk_budget = policy.risk_high if market_favorable else policy.risk_low
    desired_alt = 0.0
    if market_favorable and favorable_alts and total_alt_universe > 0:
        fraction = len(favorable_alts) / total_alt_universe
        if policy.alt_mode == "portfolio_30":
            desired_alt = min(0.30, risk_budget) * fraction
        elif policy.alt_mode == "risky_60":
            desired_alt = 0.60 * risk_budget * fraction
        elif policy.alt_mode != "none":
            raise ValueError(f"Unknown alt mode: {policy.alt_mode}")
    alt_inputs = {
        symbol: alt_volatilities[symbol] for symbol in favorable_alts if symbol in alt_volatilities
    }
    alt_weights = capped_inverse_volatility_weights(alt_inputs, desired_alt, altcoin_weight_cap)
    actual_alt = sum(alt_weights.values())
    core_budget = risk_budget - actual_alt
    weights = {
        "BTC": core_budget * float(btc_eth_split["BTC"]),
        "ETH": core_budget * float(btc_eth_split["ETH"]),
        **alt_weights,
        CASH: 1.0 - risk_budget,
    }
    if any(value < -1e-12 for value in weights.values()):
        raise AssertionError("Target allocation contains a negative weight")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise AssertionError("Target allocation does not preserve the budget")
    return weights


def _drift_weights(
    weights: Mapping[str, float], returns: Mapping[str, float]
) -> tuple[dict[str, float], float]:
    values = {
        symbol: weight * (1.0 if symbol == CASH else 1.0 + float(returns[symbol]))
        for symbol, weight in weights.items()
    }
    growth = sum(values.values())
    if growth <= 0:
        raise ValueError("Portfolio value became non-positive")
    return {symbol: value / growth for symbol, value in values.items()}, growth


def _performance_metrics(records: pd.DataFrame, annualization_days: int) -> dict[str, Any]:
    returns = records["net_return"].to_numpy(dtype=float)
    equity = records["equity"].to_numpy(dtype=float)
    volatility = (
        float(np.std(returns, ddof=1) * math.sqrt(annualization_days)) if len(returns) > 1 else 0.0
    )
    mean_return = float(np.mean(returns)) if len(returns) else 0.0
    downside = returns[returns < 0]
    downside_deviation = (
        float(np.std(downside, ddof=1) * math.sqrt(annualization_days))
        if len(downside) > 1
        else 0.0
    )
    sharpe = mean_return * annualization_days / volatility if volatility > 0 else None
    sortino = (
        mean_return * annualization_days / downside_deviation if downside_deviation > 0 else None
    )
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    drawdowns = np.concatenate(([1.0], equity)) / running_peak - 1.0
    recovery_start: int | None = None
    recovery_durations: list[int] = []
    values_with_start = np.concatenate(([1.0], equity))
    peak = values_with_start[0]
    for index, value in enumerate(values_with_start[1:], start=1):
        if value >= peak:
            if recovery_start is not None:
                recovery_durations.append(index - recovery_start)
                recovery_start = None
            peak = value
        elif recovery_start is None:
            recovery_start = index - 1
    if recovery_start is not None:
        recovery_durations.append(len(values_with_start) - 1 - recovery_start)
    monthly = records.set_index("date")["equity"].resample("ME").last().pct_change().dropna()
    return {
        "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": float(sharpe) if sharpe is not None else None,
        "sortino_zero_cash_rate": float(sortino) if sortino is not None else None,
        "maximum_drawdown": float(drawdowns.min()),
        "maximum_recovery_days": max(recovery_durations, default=0),
        "final_drawdown_recovered": recovery_start is None,
        "worst_month": float(monthly.min()) if len(monthly) else None,
        "turnover": float(records["turnover"].sum()),
        "cost_paid": float(records["cost_paid"].sum()),
        "average_risk_exposure": float(records["risk_exposure"].mean()),
        "average_alt_exposure": float(records["alt_exposure"].mean()),
        "rebalance_days": int((records["turnover"] > 1e-12).sum()),
        "observation_days": int(len(records)),
        "abstention_days": int(records["abstained"].sum()),
    }


def simulate_policy(
    dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    *,
    policy: AllocationPolicy,
    variant: str,
    selected_predictions: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    cost_multiplier: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    market_state = ConfirmationState(
        int(config["market_entry_confirmations"]), int(config["market_exit_confirmations"])
    )
    alt_symbols = sorted(symbol for symbol in prices.columns if symbol not in {"BTC", "ETH"})
    alt_states = {
        symbol: ConfirmationState(
            int(config["rotation_entry_confirmations"]),
            int(config["rotation_exit_confirmations"]),
        )
        for symbol in alt_symbols
    }
    prediction_maps: dict[str, pd.Series] = {}
    for name, frame in selected_predictions.items():
        if name.startswith("asset_"):
            prediction_maps[name] = frame.set_index(["decision_date", "entity"])["prediction"]
        else:
            prediction_maps[name] = frame.set_index("decision_date")["prediction"]
    weights: dict[str, float] = {CASH: 1.0}
    equity = 1.0
    records: list[dict[str, Any]] = []
    rebalance_interval_days = int(config.get("rebalance_interval_days", 1))
    if rebalance_interval_days <= 0:
        raise ValueError("Rebalance interval must be positive")
    for decision_index, decision_date in enumerate(dates):
        next_date = decision_date + pd.Timedelta(days=1)
        if next_date not in prices.index:
            continue
        row = features.loc[decision_date]
        if variant in {"reference", "hybrid"}:
            market_condition = bool(float(row["BTC|distance_sma_200d"]) > 0.0)
        elif variant == "forecast":
            probability = float(prediction_maps["market_probability_30"].loc[decision_date])
            expected_return = float(prediction_maps["market_return_30"].loc[decision_date])
            if probability >= float(
                config["forecast_market_probability_high"]
            ) and expected_return > float(config["round_trip_cost"]):
                market_condition = True
            elif probability <= float(config["forecast_market_probability_low"]):
                market_condition = False
            else:
                market_condition = None
        else:
            raise ValueError(f"Unknown policy variant: {variant}")
        favorable_market = market_state.update(market_condition)
        favorable_alts: list[str] = []
        volatilities: dict[str, float] = {}
        for symbol in alt_symbols:
            volatility = row.get(f"{symbol}|past_volatility_60d")
            if pd.notna(volatility):
                volatilities[symbol] = float(volatility)
            if variant == "reference":
                trend = row.get(f"{symbol}|distance_sma_90d")
                relative = row.get(f"{symbol}|relative_btc_return_90d")
                condition = (
                    bool(float(trend) > 0.0 and float(relative) > 0.0)
                    if pd.notna(trend) and pd.notna(relative)
                    else None
                )
            elif variant in {"forecast", "hybrid"}:
                key = (decision_date, symbol)
                probability_series = prediction_maps["asset_probability_30"]
                return_series = prediction_maps["asset_return_30"]
                if key not in probability_series.index or key not in return_series.index:
                    condition = None
                else:
                    condition = bool(
                        float(probability_series.loc[key])
                        >= float(config["forecast_rotation_probability"])
                        and float(return_series.loc[key]) > float(config["round_trip_cost"])
                    )
            else:
                raise ValueError(f"Unknown policy variant: {variant}")
            if alt_states[symbol].update(condition) is True:
                favorable_alts.append(symbol)
        target = build_target_weights(
            policy,
            favorable_market,
            favorable_alts if favorable_market else [],
            volatilities,
            len(alt_symbols),
            btc_eth_split=config["btc_eth_split"],
            altcoin_weight_cap=0.0 + float(config["altcoin_weight_cap"]),
        )
        held_symbols = [
            symbol for symbol, weight in weights.items() if symbol != CASH and weight > 0
        ]
        asset_returns = {
            symbol: float(prices.at[next_date, symbol] / prices.at[decision_date, symbol] - 1.0)
            for symbol in held_symbols
        }
        drifted, gross_growth = _drift_weights(weights, asset_returns)
        gross_equity = equity * gross_growth
        abstained = target is None
        review_due = decision_index % rebalance_interval_days == 0
        if target is None or not review_due:
            target = drifted
        all_symbols = (set(drifted) | set(target)) - {CASH}
        turnover = sum(
            abs(float(target.get(symbol, 0.0)) - float(drifted.get(symbol, 0.0)))
            for symbol in all_symbols
        )
        cost_rate = float(config["cost_per_traded_amount"]) * cost_multiplier
        cost_paid = gross_equity * turnover * cost_rate
        new_equity = gross_equity - cost_paid
        net_return = new_equity / equity - 1.0
        weights = {symbol: float(weight) for symbol, weight in target.items() if weight > 1e-15}
        equity = new_equity
        records.append(
            {
                "date": next_date,
                "equity": equity,
                "net_return": net_return,
                "turnover": turnover,
                "cost_paid": cost_paid,
                "risk_exposure": 1.0 - weights.get(CASH, 0.0),
                "alt_exposure": sum(
                    weight
                    for symbol, weight in weights.items()
                    if symbol not in {CASH, "BTC", "ETH"}
                ),
                "abstained": abstained,
                "review_due": review_due,
            }
        )
    record_frame = pd.DataFrame(records)
    if record_frame.empty:
        raise ValueError("Simulation produced no observations")
    return _performance_metrics(record_frame, int(config["annualization_days"])), record_frame


def simulate_static_benchmark(
    dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    *,
    target_weights: Mapping[str, float],
    config: Mapping[str, Any],
    cost_multiplier: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Deploy once at the first next close, then hold quantities without rebalancing."""
    if any(float(weight) < -1e-12 for weight in target_weights.values()):
        raise ValueError("Benchmark target contains a negative weight")
    if abs(sum(float(weight) for weight in target_weights.values()) - 1.0) > 1e-9:
        raise ValueError("Benchmark target must sum to one")
    unknown = set(target_weights) - set(prices.columns) - {CASH}
    if unknown:
        raise ValueError(f"Benchmark target contains unknown assets: {sorted(unknown)}")

    weights: dict[str, float] = {CASH: 1.0}
    equity = 1.0
    deployed = False
    records: list[dict[str, Any]] = []
    for decision_date in dates:
        next_date = decision_date + pd.Timedelta(days=1)
        if next_date not in prices.index:
            continue
        held_symbols = [
            symbol for symbol, weight in weights.items() if symbol != CASH and weight > 0
        ]
        asset_returns = {
            symbol: float(prices.at[next_date, symbol] / prices.at[decision_date, symbol] - 1.0)
            for symbol in held_symbols
        }
        drifted, gross_growth = _drift_weights(weights, asset_returns)
        gross_equity = equity * gross_growth
        turnover = 0.0
        deployment_due = not deployed
        if deployment_due:
            turnover = sum(
                abs(float(target_weights.get(symbol, 0.0)) - float(drifted.get(symbol, 0.0)))
                for symbol in (set(drifted) | set(target_weights)) - {CASH}
            )
            weights = {
                symbol: float(weight) for symbol, weight in target_weights.items() if weight > 1e-15
            }
            deployed = True
        else:
            weights = drifted
        cost_rate = float(config["cost_per_traded_amount"]) * cost_multiplier
        cost_paid = gross_equity * turnover * cost_rate
        new_equity = gross_equity - cost_paid
        net_return = new_equity / equity - 1.0
        equity = new_equity
        records.append(
            {
                "date": next_date,
                "equity": equity,
                "net_return": net_return,
                "turnover": turnover,
                "cost_paid": cost_paid,
                "risk_exposure": 1.0 - weights.get(CASH, 0.0),
                "alt_exposure": sum(
                    weight
                    for symbol, weight in weights.items()
                    if symbol not in {CASH, "BTC", "ETH"}
                ),
                "abstained": False,
                "review_due": deployment_due,
            }
        )
    record_frame = pd.DataFrame(records)
    if record_frame.empty:
        raise ValueError("Benchmark simulation produced no observations")
    return _performance_metrics(record_frame, int(config["annualization_days"])), record_frame


def prepare_price_feature_panels(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = dataset[dataset["scope"] == "asset"].copy()
    prices = assets.pivot(
        index="decision_date", columns="entity", values="decision_price"
    ).sort_index()
    feature_names = [
        "distance_sma_200d",
        "distance_sma_90d",
        "relative_btc_return_90d",
        "past_volatility_60d",
    ]
    feature_rows = []
    for _, row in assets.iterrows():
        values = {"decision_date": row["decision_date"]}
        for feature in feature_names:
            values[f"{row['entity']}|{feature}"] = row.get(feature)
        feature_rows.append(values)
    features = pd.DataFrame(feature_rows).groupby("decision_date").first().sort_index()
    return prices, features
