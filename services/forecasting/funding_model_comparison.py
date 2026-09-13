"""Frozen offline comparison of baseline and funding-enriched crypto classifiers."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from services.forecasting.binance_funding_features import FEATURE_FIELDS
from services.forecasting.evaluation import (
    EvaluationWindow,
    apply_platt_calibrator,
    build_development_windows,
    build_final_window,
    classification_metrics,
    file_sha256,
    fit_platt_calibrator,
    load_verified_dataset,
    panel_classification_ranking_metrics,
    resolve_feature_columns,
)
from services.forecasting.temporal_validation import TrainOnlyPreprocessor

COMPARISON_SCHEMA_VERSION = "crypto-forecast-funding-model-comparison-v1"


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + ("\n" if pretty else "")).encode("utf-8")


def validate_config(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    if config.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        raise ValueError(f"Unsupported comparison schema: {config.get('schema_version')}")
    if list(config.get("forecast_horizons_days", [])) != [7, 30]:
        raise ValueError("Forecast horizons must match the frozen 7/30-day contract")
    if list(config.get("variants", [])) != ["baseline", "baseline_plus_funding"]:
        raise ValueError("Variants must match the frozen baseline comparison")
    if list(config.get("funding_feature_fields", [])) != FEATURE_FIELDS:
        raise ValueError("Funding features do not match the frozen 5Q contract")
    baseline = resolve_feature_columns(config["baseline_feature_sets"])
    if len(baseline) != 17:
        raise ValueError("Baseline must contain the frozen 17 price and volume features")
    model = config.get("model")
    expected_model = {
        "name": "hist_gradient_boosting_classifier",
        "max_depth": 3,
        "max_iter": 100,
        "learning_rate": 0.05,
        "random_state": 42,
    }
    if model != expected_model:
        raise ValueError("Model must match the frozen histogram gradient boosting contract")
    expected_policies = {
        "probability_calibration": "platt_on_distinct_chronological_calibration_period",
        "preprocessing_policy": (
            "finite_nonconstant_selection_and_standardization_on_training_only"
        ),
        "selection_policy": "fixed_model_and_thresholds_before_final_confirmation",
        "final_holdout_selection_policy": (
            "never_use_final_holdout_for_model_or_threshold_selection"
        ),
        "network_policy": "offline_only",
        "order_policy": "no_orders",
        "production_policy": "no_production_changes",
    }
    for field, expected in expected_policies.items():
        if config.get(field) != expected:
            raise ValueError(f"Policy differs from the frozen contract: {field}")
    entities = [str(value) for value in config["entities"]]
    if entities != sorted(set(entities)) or len(entities) != int(config["minimum_panel_entities"]):
        raise ValueError("Entities must be unique, sorted, and match the frozen panel size")
    start = pd.Timestamp(config["common_start_date"])
    end = pd.Timestamp(config["common_end_date"])
    expected_rows = ((end - start).days + 1) * len(entities)
    if expected_rows != int(config["expected_common_rows"]):
        raise ValueError("Expected common rows do not match the frozen date/entity grid")
    durations = [
        int(config[field])
        for field in (
            "purge_days",
            "minimum_training_days",
            "calibration_days",
            "development_test_days",
            "development_step_days",
            "final_holdout_days",
            "minimum_partition_rows",
        )
    ]
    if any(value <= 0 for value in durations):
        raise ValueError("All frozen split durations and row minima must be positive")
    thresholds = config["go_thresholds"]
    expected_thresholds = {
        "minimum_mean_development_brier_improvement": 0.002,
        "minimum_final_brier_improvement": 0.002,
        "minimum_mean_development_daily_auc_improvement": 0.01,
        "minimum_final_daily_auc_improvement": 0.01,
        "maximum_single_fold_brier_degradation": 0.001,
        "maximum_single_fold_daily_auc_degradation": 0.01,
        "required_horizons_pass": 2,
    }
    if thresholds != expected_thresholds:
        raise ValueError("Go thresholds differ from the frozen contract")
    return baseline, list(FEATURE_FIELDS)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def load_comparison_panel(
    dataset_root: Path, funding_root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    baseline_features, funding_features = validate_config(config)
    dataset_manifest_path = dataset_root / "manifest.json"
    if file_sha256(dataset_manifest_path) != str(config["dataset_manifest_sha256"]):
        raise ValueError("Dataset manifest SHA-256 does not match the frozen input")
    dataset, dataset_manifest = load_verified_dataset(dataset_root)
    if (
        dataset_manifest.get("dataset_version") != config["dataset_artifact_id"]
        or dataset_manifest.get("dataset_sha256") != config["dataset_sha256"]
    ):
        raise ValueError("Dataset artifact does not match the frozen input")

    funding_manifest_path = funding_root / "manifest.json"
    funding_manifest = _load_json(funding_manifest_path, "funding feature manifest")
    if (
        funding_manifest.get("artifact_id") != config["funding_artifact_id"]
        or funding_manifest.get("result_file") != "funding_feature_result.json"
        or funding_manifest.get("feature_table_file") != config["funding_table_file"]
        or funding_manifest.get("result_sha256") != config["funding_result_sha256"]
        or funding_manifest.get("feature_table_sha256") != config["funding_table_sha256"]
        or funding_manifest.get("model_trained") is not False
        or funding_manifest.get("targets_created") is not False
    ):
        raise ValueError("Funding feature artifact does not match the frozen input")
    result_path = funding_root / str(funding_manifest.get("result_file", ""))
    table_path = funding_root / str(config["funding_table_file"])
    if file_sha256(result_path) != str(config["funding_result_sha256"]):
        raise ValueError("Funding result SHA-256 does not match the frozen input")
    if file_sha256(table_path) != str(config["funding_table_sha256"]):
        raise ValueError("Funding table SHA-256 does not match the frozen input")
    funding_result = _load_json(result_path, "funding feature result")
    if (
        funding_result.get("decision") != "GO_CAUSAL_FUNDING_FEATURES"
        or funding_result.get("future_values_used") is not False
        or funding_result.get("targets_created") is not False
    ):
        raise ValueError("Funding features are not an approved causal input")

    funding = pd.read_csv(table_path, low_memory=False)
    required_funding = ["date_utc", "symbol", "market_symbol", "model_eligible", *funding_features]
    missing_funding = [column for column in required_funding if column not in funding]
    if missing_funding:
        raise ValueError(f"Funding table is missing columns: {missing_funding}")
    funding["decision_date"] = pd.to_datetime(funding["date_utc"], errors="raise").dt.normalize()
    funding["entity"] = funding["symbol"].astype(str)
    funding["model_eligible"] = (
        funding["model_eligible"].astype(str).str.lower().map({"true": True, "false": False})
    )
    if funding["model_eligible"].isna().any():
        raise ValueError("Funding eligibility must be explicitly boolean")
    if funding.duplicated(["decision_date", "entity"]).any():
        raise ValueError("Funding table contains duplicate date/entity keys")

    entities = [str(value) for value in config["entities"]]
    start = pd.Timestamp(config["common_start_date"])
    end = pd.Timestamp(config["common_end_date"])
    asset_rows = dataset[
        (dataset["scope"] == "asset")
        & dataset["entity"].isin(entities)
        & dataset["decision_date"].between(start, end)
    ].copy()
    funding_rows = funding[
        funding["entity"].isin(entities)
        & funding["decision_date"].between(start, end)
        & funding["model_eligible"]
    ][["decision_date", "entity", *funding_features]].copy()
    if asset_rows.duplicated(["decision_date", "entity"]).any():
        raise ValueError("Dataset contains duplicate date/entity keys")
    panel = asset_rows.merge(
        funding_rows,
        on=["decision_date", "entity"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["decision_date", "entity"])
    if len(panel) != int(config["expected_common_rows"]):
        raise ValueError("Joined panel row count differs from the frozen contract")
    counts = panel.groupby("decision_date")["entity"].nunique()
    if (
        len(counts) != (end - start).days + 1
        or not (counts == int(config["minimum_panel_entities"])).all()
    ):
        raise ValueError("Joined panel does not contain every frozen entity on every date")
    numeric_columns = [
        *baseline_features,
        *funding_features,
        *(
            str(config["regression_target_template"]).format(horizon=horizon)
            for horizon in config["forecast_horizons_days"]
        ),
        *(
            str(config["classification_target_template"]).format(horizon=horizon)
            for horizon in config["forecast_horizons_days"]
        ),
    ]
    numeric = panel[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Joined comparison panel contains missing or non-finite values")
    panel[numeric_columns] = numeric
    return panel, baseline_features, funding_features


def build_windows(config: Mapping[str, Any]) -> tuple[list[EvaluationWindow], EvaluationWindow]:
    start = pd.Timestamp(config["common_start_date"])
    end = pd.Timestamp(config["common_end_date"])
    final = build_final_window(
        start,
        end,
        minimum_training_days=int(config["minimum_training_days"]),
        calibration_days=int(config["calibration_days"]),
        final_holdout_days=int(config["final_holdout_days"]),
        purge_days=int(config["purge_days"]),
    )
    if final is None:
        raise ValueError("Frozen common period cannot support the final window")
    development_end = final.calibration_start - pd.Timedelta(days=int(config["purge_days"]) + 1)
    development = build_development_windows(
        start,
        development_end,
        minimum_training_days=int(config["minimum_training_days"]),
        calibration_days=int(config["calibration_days"]),
        test_days=int(config["development_test_days"]),
        step_days=int(config["development_step_days"]),
        purge_days=int(config["purge_days"]),
    )
    if len(development) != int(config["expected_development_folds"]):
        raise ValueError("Development fold count differs from the frozen contract")
    return development, final


def _partition(panel: pd.DataFrame, window: EvaluationWindow, config: Mapping[str, Any]):
    partitions = tuple(
        panel[panel["decision_date"].between(start, end)].copy()
        for start, end in (
            (window.train_start, window.train_end),
            (window.calibration_start, window.calibration_end),
            (window.test_start, window.test_end),
        )
    )
    if any(len(frame) < int(config["minimum_partition_rows"]) for frame in partitions):
        raise ValueError(f"Insufficient partition rows for {window.name}")
    return partitions


def _evaluate_variant(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str,
    feature_columns: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    preprocessor = TrainOnlyPreprocessor()
    train_features = preprocessor.fit_transform(train, feature_columns)
    calibration_features = preprocessor.transform(calibration)
    test_features = preprocessor.transform(test)
    model_params = {key: value for key, value in config["model"].items() if key != "name"}
    model = HistGradientBoostingClassifier(**model_params)
    train_target = train[target].to_numpy(dtype=int)
    calibration_target = calibration[target].to_numpy(dtype=int)
    if np.unique(train_target).size != 2 or np.unique(calibration_target).size != 2:
        raise ValueError("Training and calibration partitions must contain both classes")
    model.fit(train_features, train_target)
    calibration_raw = model.predict_proba(calibration_features)[:, 1]
    test_raw = model.predict_proba(test_features)[:, 1]
    calibrator = fit_platt_calibrator(calibration_target, calibration_raw)
    probabilities = apply_platt_calibrator(calibrator, test_raw)
    metrics = {
        **classification_metrics(test[target], probabilities),
        **panel_classification_ranking_metrics(
            test[target], probabilities, test["decision_date"], test["entity"]
        ),
        "preprocessing": preprocessor.metadata(),
        "calibration": "platt_distinct_calibration_period",
    }
    return metrics, probabilities


def assess_horizon(
    development: Sequence[Mapping[str, Any]],
    final: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    development_brier = [float(item["brier_improvement"]) for item in development]
    development_auc = [float(item["daily_auc_improvement"]) for item in development]
    mean_brier = float(np.mean(development_brier))
    mean_auc = float(np.mean(development_auc))
    gates = {
        "mean_development_brier": mean_brier
        >= float(thresholds["minimum_mean_development_brier_improvement"]),
        "final_brier": float(final["brier_improvement"])
        >= float(thresholds["minimum_final_brier_improvement"]),
        "mean_development_daily_auc": mean_auc
        >= float(thresholds["minimum_mean_development_daily_auc_improvement"]),
        "final_daily_auc": float(final["daily_auc_improvement"])
        >= float(thresholds["minimum_final_daily_auc_improvement"]),
        "no_material_single_fold_brier_degradation": min(development_brier)
        >= -float(thresholds["maximum_single_fold_brier_degradation"]),
        "no_material_single_fold_daily_auc_degradation": min(development_auc)
        >= -float(thresholds["maximum_single_fold_daily_auc_degradation"]),
    }
    return {
        "mean_development_brier_improvement": mean_brier,
        "final_brier_improvement": float(final["brier_improvement"]),
        "mean_development_daily_auc_improvement": mean_auc,
        "final_daily_auc_improvement": float(final["daily_auc_improvement"]),
        "gates": gates,
        "pass": all(gates.values()),
    }


def run_funding_model_comparison(
    panel: pd.DataFrame,
    baseline_features: Sequence[str],
    funding_features: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    validate_config(config)
    development_windows, final_window = build_windows(config)
    results = []
    prediction_rows = []
    for horizon in config["forecast_horizons_days"]:
        target = str(config["classification_target_template"]).format(horizon=horizon)
        window_results = []
        for window in [*development_windows, final_window]:
            train, calibration, test = _partition(panel, window, config)
            variants = {}
            for variant, columns in (
                ("baseline", list(baseline_features)),
                ("baseline_plus_funding", [*baseline_features, *funding_features]),
            ):
                metrics, probabilities = _evaluate_variant(
                    train,
                    calibration,
                    test,
                    target=target,
                    feature_columns=columns,
                    config=config,
                )
                variants[variant] = {"feature_count": len(columns), **metrics}
                for decision_date, entity, actual, probability in zip(
                    test["decision_date"],
                    test["entity"],
                    test[target],
                    probabilities,
                    strict=True,
                ):
                    prediction_rows.append(
                        {
                            "horizon_days": int(horizon),
                            "window": window.name,
                            "variant": variant,
                            "decision_date": decision_date.date().isoformat(),
                            "entity": str(entity),
                            "target": int(actual),
                            "probability": float(probability),
                        }
                    )
            baseline = variants["baseline"]
            enriched = variants["baseline_plus_funding"]
            deltas = {
                "brier_improvement": float(baseline["brier_score"] - enriched["brier_score"]),
                "daily_auc_improvement": float(
                    enriched["daily_roc_auc"] - baseline["daily_roc_auc"]
                ),
            }
            window_results.append(
                {
                    "window": window.to_dict(),
                    "partition_rows": {
                        "train": len(train),
                        "calibration": len(calibration),
                        "test": len(test),
                    },
                    "variants": variants,
                    "deltas": deltas,
                }
            )
        development = [
            {**item["deltas"], "window": item["window"]["name"]}
            for item in window_results
            if item["window"]["name"].startswith("development_")
        ]
        final_item = next(
            item for item in window_results if item["window"]["name"] == "final_confirmation"
        )
        assessment = assess_horizon(development, final_item["deltas"], config["go_thresholds"])
        results.append(
            {
                "horizon_days": int(horizon),
                "target": target,
                "windows": window_results,
                "assessment": assessment,
            }
        )
    passed = sum(bool(item["assessment"]["pass"]) for item in results)
    overall_go = passed >= int(config["go_thresholds"]["required_horizons_pass"])
    result = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "dataset_artifact_id": config["dataset_artifact_id"],
        "dataset_sha256": config["dataset_sha256"],
        "funding_artifact_id": config["funding_artifact_id"],
        "funding_table_sha256": config["funding_table_sha256"],
        "panel": {
            "rows": len(panel),
            "entities": list(config["entities"]),
            "first_date": config["common_start_date"],
            "last_date": config["common_end_date"],
            "baseline_feature_count": len(baseline_features),
            "funding_feature_count": len(funding_features),
        },
        "protocol": {
            key: config[key]
            for key in (
                "model",
                "probability_calibration",
                "purge_days",
                "minimum_training_days",
                "calibration_days",
                "development_test_days",
                "development_step_days",
                "final_holdout_days",
                "go_thresholds",
                "preprocessing_policy",
                "selection_policy",
                "final_holdout_selection_policy",
            )
        },
        "horizons": results,
        "horizons_passed": passed,
        "decision": "GO_PREDICTIVE_FUNDING" if overall_go else "NO_GO_PREDICTIVE_FUNDING",
        "final_holdout_used_for_selection": False,
        "thresholds_tuned_after_results": False,
        "network_used": False,
        "orders_placed": False,
        "production_touched": False,
        "economic_evaluation_performed": False,
    }
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["horizon_days", "window", "variant", "decision_date", "entity"]
    )
    return result, predictions


def write_comparison_artifact(
    result: Mapping[str, Any],
    predictions: pd.DataFrame,
    *,
    config_sha256: str,
    comparison_code_sha256: str,
    output_root: Path,
) -> Path:
    predictions_payload = predictions.to_csv(index=False, lineterminator="\n").encode("utf-8")
    identity = {
        **result,
        "config_sha256": config_sha256,
        "comparison_code_sha256": comparison_code_sha256,
        "predictions_sha256": hashlib.sha256(predictions_payload).hexdigest(),
    }
    artifact_id = (
        f"{COMPARISON_SCHEMA_VERSION}-" f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Comparison artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        result_path = temporary_path / "funding_model_comparison_result.json"
        predictions_path = temporary_path / "predictions.csv"
        result_path.write_bytes(_json_bytes(identity, pretty=True))
        predictions_path.write_bytes(predictions_payload)
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_file": result_path.name,
            "result_sha256": file_sha256(result_path),
            "predictions_file": predictions_path.name,
            "predictions_sha256": file_sha256(predictions_path),
            "final_holdout_used_for_selection": False,
            "thresholds_tuned_after_results": False,
            "network_used": False,
            "credentials_used": False,
            "orders_placed": False,
            "production_touched": False,
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
