"""Run lot-4 offline allocation comparisons from frozen forecast predictions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy  # noqa: E402
import pandas  # noqa: E402

from services.forecasting.allocation_backtest import (  # noqa: E402
    ALLOCATION_SCHEMA_VERSION,
    AllocationPolicy,
    common_prediction_blocks,
    file_sha256,
    prepare_price_feature_panels,
    simulate_policy,
    simulate_static_benchmark,
    verify_forecast_artifact,
)
from services.forecasting.evaluation import (
    canonical_json_sha256,
    load_verified_dataset,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_allocation.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-lot4",
    )
    return parser.parse_args()


def _block_name(dates: pandas.DatetimeIndex, market_predictions: pandas.DataFrame) -> str:
    windows = market_predictions[market_predictions["decision_date"].isin(dates)]["window"].unique()
    if len(windows) == 1:
        return str(windows[0])
    return f"intersection_{dates.min().date().isoformat()}_{dates.max().date().isoformat()}"


def _defensive_signal_count(
    dates: pandas.DatetimeIndex,
    selected: dict[str, pandas.DataFrame],
    config: dict[str, object],
) -> int:
    probabilities = selected["market_probability_7"].set_index("decision_date")["prediction"]
    returns = selected["market_return_7"].set_index("decision_date")["prediction"]
    required = int(config["forecast_defensive_confirmations"])
    streak = 0
    signals = 0
    for date in dates:
        condition = 1.0 - float(probabilities.loc[date]) >= float(
            config["forecast_defensive_probability"]
        ) and float(returns.loc[date]) < -float(config["round_trip_cost"])
        streak = streak + 1 if condition else 0
        if streak == required:
            signals += 1
    return signals


def _forecast_gate_diagnostics(
    dates: pandas.DatetimeIndex,
    selected: dict[str, pandas.DataFrame],
    config: dict[str, object],
) -> dict[str, object]:
    market_probability = (
        selected["market_probability_30"].set_index("decision_date")["prediction"].loc[dates]
    )
    market_return = selected["market_return_30"].set_index("decision_date")["prediction"].loc[dates]
    favorable_market = (market_probability >= float(config["forecast_market_probability_high"])) & (
        market_return > float(config["round_trip_cost"])
    )
    unfavorable_market = market_probability <= float(config["forecast_market_probability_low"])

    asset_probability = selected["asset_probability_30"]
    asset_return = selected["asset_return_30"]
    asset_panel = asset_probability.merge(
        asset_return,
        on=["decision_date", "entity"],
        suffixes=("_probability", "_return"),
    )
    asset_panel = asset_panel[asset_panel["decision_date"].isin(dates)].copy()
    favorable_assets = (
        asset_panel["prediction_probability"] >= float(config["forecast_rotation_probability"])
    ) & (asset_panel["prediction_return"] > float(config["round_trip_cost"]))
    return {
        "market_decision_days": int(len(dates)),
        "market_favorable_gate_days": int(favorable_market.sum()),
        "market_unfavorable_gate_days": int(unfavorable_market.sum()),
        "market_neutral_gate_days": int((~(favorable_market | unfavorable_market)).sum()),
        "market_probability_min": float(market_probability.min()),
        "market_probability_max": float(market_probability.max()),
        "market_expected_return_min": float(market_return.min()),
        "market_expected_return_max": float(market_return.max()),
        "asset_candidate_rows": int(len(asset_panel)),
        "asset_favorable_gate_rows": int(favorable_assets.sum()),
        "asset_days_with_at_least_one_favorable_gate": int(
            asset_panel.loc[favorable_assets, "decision_date"].nunique()
        ),
        "asset_probability_max": float(asset_panel["prediction_probability"].max()),
        "asset_expected_relative_return_max": float(asset_panel["prediction_return"].max()),
        "defensive_7d_proposal_signals_not_executed": _defensive_signal_count(
            dates, selected, config
        ),
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != ALLOCATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported allocation schema: {config.get('schema_version')}")
    dataset, dataset_manifest = load_verified_dataset(args.dataset_dir)
    forecast_results, predictions = verify_forecast_artifact(args.forecast_dir)
    if forecast_results["dataset"]["dataset_sha256"] != dataset_manifest["dataset_sha256"]:
        raise ValueError("Forecast artifact is not linked to the supplied dataset")
    blocks, selected = common_prediction_blocks(
        forecast_results, predictions, int(config["minimum_block_days"])
    )
    if not blocks:
        raise ValueError("No common out-of-sample prediction block is available")
    prices, features = prepare_price_feature_panels(dataset)
    policies = [AllocationPolicy(**item) for item in config["policies"]]
    results: list[dict[str, object]] = []
    daily_frames: list[pandas.DataFrame] = []
    gate_diagnostics: list[dict[str, object]] = []
    for block_index, dates in enumerate(blocks, start=1):
        block_name = _block_name(dates, selected["market_return_30"])
        gate_diagnostics.append(
            {
                "block": block_name,
                "block_index": block_index,
                **_forecast_gate_diagnostics(dates, selected, config),
            }
        )
        for policy in policies:
            for variant in ("reference", "forecast", "hybrid"):
                for multiplier in config["cost_multipliers"]:
                    metrics, daily = simulate_policy(
                        dates,
                        prices,
                        features,
                        policy=policy,
                        variant=variant,
                        selected_predictions=selected,
                        config=config,
                        cost_multiplier=float(multiplier),
                    )
                    identity = {
                        "block": block_name,
                        "block_index": block_index,
                        "decision_start": dates.min().date().isoformat(),
                        "decision_end": dates.max().date().isoformat(),
                        "decision_days": len(dates),
                        "policy": policy.name,
                        "variant": variant,
                        "cost_multiplier": float(multiplier),
                    }
                    results.append({**identity, "metrics": metrics})
                    daily = daily.assign(**identity)
                    daily_frames.append(daily)
        for benchmark in config["benchmarks"]:
            for multiplier in config["cost_multipliers"]:
                metrics, daily = simulate_static_benchmark(
                    dates,
                    prices,
                    target_weights=benchmark["target_weights"],
                    config=config,
                    cost_multiplier=float(multiplier),
                )
                identity = {
                    "block": block_name,
                    "block_index": block_index,
                    "decision_start": dates.min().date().isoformat(),
                    "decision_end": dates.max().date().isoformat(),
                    "decision_days": len(dates),
                    "policy": benchmark["name"],
                    "variant": "benchmark",
                    "cost_multiplier": float(multiplier),
                }
                results.append({**identity, "metrics": metrics})
                daily = daily.assign(**identity)
                daily_frames.append(daily)
        defensive_signals = _defensive_signal_count(dates, selected, config)
        for result in results:
            if result["block_index"] == block_index and result["variant"] in {
                "forecast",
                "hybrid",
            }:
                result["defensive_7d_proposal_signals_not_executed"] = defensive_signals
    daily_results = pandas.concat(daily_frames, ignore_index=True)
    runtime = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }
    service_path = PROJECT_ROOT / "services" / "forecasting" / "allocation_backtest.py"
    identity = {
        "schema_version": ALLOCATION_SCHEMA_VERSION,
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "forecast_results_sha256": file_sha256(args.forecast_dir / "results.json"),
        "forecast_predictions_sha256": file_sha256(args.forecast_dir / "predictions.csv"),
        "config_sha256": file_sha256(args.config),
        "allocation_code_sha256": file_sha256(service_path),
        "runner_code_sha256": file_sha256(Path(__file__).resolve()),
        "runtime_versions": runtime,
    }
    artifact_id = f"{ALLOCATION_SCHEMA_VERSION}-{canonical_json_sha256(identity)[:16]}"
    output_directory = args.output_root / artifact_id
    output_directory.mkdir(parents=True, exist_ok=False)
    payload = {
        **identity,
        "artifact_id": artifact_id,
        "portfolio_provenance": (
            "hypothetical_cash_start_per_block; no dated real-portfolio composition was supplied"
        ),
        "execution_convention": "decision at close t; returns accrue to close t+1; rebalance at close t+1",
        "current_portfolio_transition_cost": "unavailable_without_dated_real_positions",
        "benchmark_convention": "deploy once at the first next close, then hold quantities",
        "review_convention": (
            f"signals observed daily; allocation reviewed every "
            f"{int(config['rebalance_interval_days'])} calendar days from each block start"
        ),
        "forecast_gate_diagnostics": gate_diagnostics,
        "real_orders_created": False,
        "production_configuration_changed": False,
        "results": results,
    }
    results_path = output_directory / "results.json"
    daily_path = output_directory / "daily_results.csv"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    daily_results.to_csv(daily_path, index=False)
    manifest = {
        **identity,
        "artifact_id": artifact_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_file": results_path.name,
        "results_sha256": file_sha256(results_path),
        "daily_results_file": daily_path.name,
        "daily_results_sha256": file_sha256(daily_path),
        "real_orders_created": False,
        "production_configuration_changed": False,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output_directory": str(output_directory.resolve()), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
