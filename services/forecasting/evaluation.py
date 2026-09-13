"""Leakage-resistant offline evaluation for crypto forecast candidates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from services.forecasting.dataset import BASE_FEATURE_COLUMNS, VOLUME_FEATURE_COLUMNS
from services.forecasting.temporal_validation import TrainOnlyPreprocessor

EXPERIMENT_SCHEMA_VERSION = "crypto-forecast-experiment-v1"
RELATIVE_BTC_FEATURE_COLUMNS = (
    "relative_btc_return_7d",
    "relative_btc_return_30d",
    "relative_btc_return_90d",
)
RELATIVE_GROUP_FEATURE_COLUMNS = (
    "relative_group_return_7d",
    "relative_group_return_30d",
    "relative_group_return_90d",
)


@dataclass(frozen=True)
class EvaluationWindow:
    """One train/calibration/test window with explicit purges."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_days: int

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, pd.Timestamp):
                values[key] = value.date().isoformat()
        return values


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_verified_dataset(dataset_directory: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a lot-2 artifact only after verifying its declared content hash."""
    directory = Path(dataset_directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = directory / str(manifest.get("dataset_file", "forecast_dataset.csv"))
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {dataset_path}")
    expected_hash = str(manifest.get("dataset_sha256", ""))
    actual_hash = file_sha256(dataset_path)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            f"Dataset SHA-256 mismatch: expected {expected_hash or 'missing'}, got {actual_hash}"
        )
    frame = pd.read_csv(dataset_path, low_memory=False)
    if "decision_date" not in frame:
        raise ValueError("Dataset must contain decision_date")
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    return frame, manifest


def resolve_feature_columns(feature_sets: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for feature_set in feature_sets:
        if feature_set == "base":
            selected = BASE_FEATURE_COLUMNS
        elif feature_set == "relative_btc":
            selected = RELATIVE_BTC_FEATURE_COLUMNS
        elif feature_set == "relative_group":
            selected = RELATIVE_GROUP_FEATURE_COLUMNS
        elif feature_set == "volume":
            selected = VOLUME_FEATURE_COLUMNS
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
        for column in selected:
            if column not in columns:
                columns.append(column)
    return columns


def _normalise_date(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def build_development_windows(
    minimum_date: object,
    development_end: object,
    *,
    minimum_training_days: int,
    calibration_days: int,
    test_days: int,
    step_days: int,
    purge_days: int,
) -> list[EvaluationWindow]:
    """Build expanding windows; insufficient history returns no window."""
    if min(minimum_training_days, calibration_days, test_days, step_days) <= 0:
        raise ValueError("Window durations must be positive")
    if purge_days < 0:
        raise ValueError("purge_days must be non-negative")
    first_date = _normalise_date(minimum_date)
    last_date = _normalise_date(development_end)
    train_end = first_date + pd.Timedelta(days=minimum_training_days - 1)
    calibration_start = train_end + pd.Timedelta(days=purge_days + 1)
    calibration_end = calibration_start + pd.Timedelta(days=calibration_days - 1)
    test_start = calibration_end + pd.Timedelta(days=purge_days + 1)
    windows: list[EvaluationWindow] = []
    index = 1
    while test_start <= last_date:
        test_end = test_start + pd.Timedelta(days=test_days - 1)
        if test_end > last_date:
            break
        windows.append(
            EvaluationWindow(
                name=f"development_{index:02d}",
                train_start=first_date,
                train_end=calibration_start - pd.Timedelta(days=purge_days + 1),
                calibration_start=calibration_start,
                calibration_end=calibration_end,
                test_start=test_start,
                test_end=test_end,
                purge_days=purge_days,
            )
        )
        index += 1
        calibration_start += pd.Timedelta(days=step_days)
        calibration_end += pd.Timedelta(days=step_days)
        test_start += pd.Timedelta(days=step_days)
    return windows


def build_final_window(
    minimum_date: object,
    maximum_date: object,
    *,
    minimum_training_days: int,
    calibration_days: int,
    final_holdout_days: int,
    purge_days: int,
) -> EvaluationWindow | None:
    """Reserve the trailing holdout, preceded by calibration and two purges."""
    first_date = _normalise_date(minimum_date)
    last_date = _normalise_date(maximum_date)
    test_start = last_date - pd.Timedelta(days=final_holdout_days - 1)
    calibration_end = test_start - pd.Timedelta(days=purge_days + 1)
    calibration_start = calibration_end - pd.Timedelta(days=calibration_days - 1)
    train_end = calibration_start - pd.Timedelta(days=purge_days + 1)
    if train_end - first_date < pd.Timedelta(days=minimum_training_days - 1):
        return None
    return EvaluationWindow(
        name="final_confirmation",
        train_start=first_date,
        train_end=train_end,
        calibration_start=calibration_start,
        calibration_end=calibration_end,
        test_start=test_start,
        test_end=last_date,
        purge_days=purge_days,
    )


def _window_frame(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame["decision_date"] >= start) & (frame["decision_date"] <= end)].copy()


def _validate_window_rows(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    minimum_partition_rows: int,
) -> None:
    counts = {
        "train": len(train),
        "calibration": len(calibration),
        "test": len(test),
    }
    insufficient = {name: count for name, count in counts.items() if count < minimum_partition_rows}
    if insufficient:
        raise ValueError(f"Insufficient partition rows: {insufficient}")


def expected_calibration_error(
    targets: Iterable[float], probabilities: Iterable[float], *, bins: int = 10
) -> float:
    y_true = np.asarray(list(targets), dtype=float)
    y_prob = np.asarray(list(probabilities), dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError("Targets and probabilities must have the same non-zero length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(y_prob, edges[1:-1], right=False), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(y_true[mask].mean()) - float(y_prob[mask].mean())
            )
    return error


def fit_platt_calibrator(
    calibration_targets: Iterable[int], calibration_probabilities: Iterable[float]
) -> LogisticRegression:
    """Fit Platt scaling on calibration rows only."""
    targets = np.asarray(list(calibration_targets), dtype=int)
    probabilities = np.asarray(list(calibration_probabilities), dtype=float)
    if len(targets) != len(probabilities) or len(targets) == 0:
        raise ValueError("Calibration targets and probabilities must align")
    if np.unique(targets).size != 2:
        raise ValueError("Probability calibration requires both target classes")
    logits = np.log(
        np.clip(probabilities, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probabilities, 1e-6, 1.0)
    )
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(logits.reshape(-1, 1), targets)
    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression, probabilities: Iterable[float]
) -> np.ndarray:
    raw = np.asarray(list(probabilities), dtype=float)
    logits = np.log(np.clip(raw, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - raw, 1e-6, 1.0))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def regression_metrics(targets: Iterable[float], predictions: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(targets), dtype=float)
    y_pred = np.asarray(list(predictions), dtype=float)
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        correlation = 0.0
    else:
        correlation = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "directional_accuracy": float(np.mean((y_true >= 0.0) == (y_pred >= 0.0))),
        "spearman_rank_correlation": float(correlation) if pd.notna(correlation) else 0.0,
    }


def panel_ranking_metrics(
    targets: Iterable[float],
    predictions: Iterable[float],
    dates: Iterable[object],
    entities: Iterable[object],
) -> dict[str, float | int]:
    """Average cross-sectional Spearman correlation independently for each date."""
    frame = pd.DataFrame(
        {
            "target": list(targets),
            "prediction": list(predictions),
            "decision_date": pd.to_datetime(list(dates)),
            "entity": list(entities),
        }
    )
    correlations: list[float] = []
    for _, daily in frame.groupby("decision_date"):
        if daily["entity"].nunique() < 2:
            continue
        if daily["target"].nunique() < 2 or daily["prediction"].nunique() < 2:
            continue
        correlation = daily["target"].corr(daily["prediction"], method="spearman")
        if pd.notna(correlation):
            correlations.append(float(correlation))
    return {
        "daily_rank_correlation": float(np.mean(correlations)) if correlations else 0.0,
        "ranked_dates": len(correlations),
    }


def panel_classification_ranking_metrics(
    targets: Iterable[int],
    probabilities: Iterable[float],
    dates: Iterable[object],
    entities: Iterable[object],
) -> dict[str, float | int]:
    """Average daily cross-sectional AUC when both outcome classes exist."""
    frame = pd.DataFrame(
        {
            "target": list(targets),
            "probability": list(probabilities),
            "decision_date": pd.to_datetime(list(dates)),
            "entity": list(entities),
        }
    )
    scores: list[float] = []
    for _, daily in frame.groupby("decision_date"):
        if daily["entity"].nunique() < 2 or daily["target"].nunique() != 2:
            continue
        scores.append(float(roc_auc_score(daily["target"], daily["probability"])))
    return {
        "daily_roc_auc": float(np.mean(scores)) if scores else 0.5,
        "ranked_dates": len(scores),
    }


def classification_metrics(
    targets: Iterable[int], probabilities: Iterable[float]
) -> dict[str, float]:
    y_true = np.asarray(list(targets), dtype=int)
    y_prob = np.clip(np.asarray(list(probabilities), dtype=float), 1e-9, 1.0 - 1e-9)
    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "accuracy_at_0_5": float(accuracy_score(y_true, y_prob >= 0.5)),
        "expected_calibration_error_10_bins": expected_calibration_error(y_true, y_prob),
    }


def _regression_candidates(config: Mapping[str, Any]) -> list[str]:
    return list(config["regression_models"].keys())


def _classification_candidates(config: Mapping[str, Any]) -> list[str]:
    return list(config["classification_models"].keys())


def _fit_regression_predict(
    name: str,
    config: Mapping[str, Any],
    train: pd.DataFrame,
    predict: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str],
    horizon: int,
) -> np.ndarray:
    if name == "zero":
        return np.zeros(len(predict), dtype=float)
    if name == "momentum":
        return predict[f"past_return_{horizon}d"].to_numpy(dtype=float)
    preprocessor = TrainOnlyPreprocessor()
    train_features = preprocessor.fit_transform(train, feature_columns)
    predict_features = preprocessor.transform(predict)
    parameters = dict(config["regression_models"][name])
    if name == "ridge":
        model = Ridge(**parameters)
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(**parameters)
    else:
        raise ValueError(f"Unknown regression model: {name}")
    model.fit(train_features, train[target_column].to_numpy(dtype=float))
    return np.asarray(model.predict(predict_features), dtype=float)


def _fit_classifier_probabilities(
    name: str,
    config: Mapping[str, Any],
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, str]:
    targets = train[target_column].to_numpy(dtype=int)
    if name == "train_prevalence":
        return np.full(len(test), float(targets.mean()), dtype=float), "not_applicable_train_only"
    if np.unique(targets).size != 2:
        raise ValueError("Classifier training requires both target classes")
    preprocessor = TrainOnlyPreprocessor()
    train_features = preprocessor.fit_transform(train, feature_columns)
    calibration_features = preprocessor.transform(calibration)
    test_features = preprocessor.transform(test)
    parameters = dict(config["classification_models"][name])
    if name == "logistic":
        model = LogisticRegression(solver="lbfgs", **parameters)
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(**parameters)
    else:
        raise ValueError(f"Unknown classification model: {name}")
    model.fit(train_features, targets)
    raw_calibration = model.predict_proba(calibration_features)[:, 1]
    raw_test = model.predict_proba(test_features)[:, 1]
    calibrator = fit_platt_calibrator(calibration[target_column], raw_calibration)
    return apply_platt_calibrator(calibrator, raw_test), "platt_distinct_calibration_period"


def classification_hurdle_expected_returns(
    training_returns: Iterable[float],
    training_labels: Iterable[int],
    probabilities: Iterable[float],
) -> tuple[np.ndarray, dict[str, float]]:
    """Map calibrated probabilities to returns using train-only class means."""
    returns = np.asarray(list(training_returns), dtype=float)
    labels = np.asarray(list(training_labels), dtype=int)
    predicted_probabilities = np.asarray(list(probabilities), dtype=float)
    if len(returns) == 0 or len(returns) != len(labels):
        raise ValueError("Training returns and labels must align")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Hurdle return mapping requires both training classes")
    negative_mean = float(returns[labels == 0].mean())
    positive_mean = float(returns[labels == 1].mean())
    expected = (
        predicted_probabilities * positive_mean + (1.0 - predicted_probabilities) * negative_mean
    )
    return expected, {
        "training_negative_mean_return": negative_mean,
        "training_positive_mean_return": positive_mean,
    }


def _evaluate_window(
    frame: pd.DataFrame,
    window: EvaluationWindow,
    *,
    config: Mapping[str, Any],
    regression_target: str,
    classification_target: str,
    feature_columns: Sequence[str],
    horizon: int,
    panel_mode: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = list(feature_columns) + [regression_target, classification_target]
    usable = frame.dropna(subset=required).copy()
    train = _window_frame(usable, window.train_start, window.train_end)
    calibration = _window_frame(usable, window.calibration_start, window.calibration_end)
    test = _window_frame(usable, window.test_start, window.test_end)
    _validate_window_rows(train, calibration, test, int(config["minimum_partition_rows"]))
    result: dict[str, Any] = {
        "window": window.to_dict(),
        "partition_rows": {
            "train": int(len(train)),
            "calibration": int(len(calibration)),
            "test": int(len(test)),
        },
        "regression": {},
        "classification": {},
        "classification_hurdle_return": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    for model_name in _regression_candidates(config):
        predictions = _fit_regression_predict(
            model_name,
            config,
            train,
            test,
            regression_target,
            feature_columns,
            horizon,
        )
        metrics = regression_metrics(test[regression_target], predictions)
        if panel_mode:
            metrics.update(
                panel_ranking_metrics(
                    test[regression_target], predictions, test["decision_date"], test["entity"]
                )
            )
        result["regression"][model_name] = metrics
        groups = test["group"] if "group" in test else [None] * len(test)
        for date, entity, group, target, prediction in zip(
            test["decision_date"],
            test["entity"],
            groups,
            test[regression_target],
            predictions,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "window": window.name,
                    "task": "regression",
                    "model": model_name,
                    "decision_date": date.date().isoformat(),
                    "entity": str(entity),
                    "group": str(group) if group is not None else None,
                    "target": float(target),
                    "prediction": float(prediction),
                }
            )
    for model_name in _classification_candidates(config):
        probabilities, calibration_method = _fit_classifier_probabilities(
            model_name,
            config,
            train,
            calibration,
            test,
            classification_target,
            feature_columns,
        )
        metrics = {
            **classification_metrics(test[classification_target], probabilities),
            "calibration": calibration_method,
        }
        if panel_mode:
            metrics.update(
                panel_classification_ranking_metrics(
                    test[classification_target],
                    probabilities,
                    test["decision_date"],
                    test["entity"],
                )
            )
        result["classification"][model_name] = metrics
        hurdle_returns, hurdle_metadata = classification_hurdle_expected_returns(
            train[regression_target],
            train[classification_target],
            probabilities,
        )
        hurdle_metrics = {
            **regression_metrics(test[regression_target], hurdle_returns),
            **hurdle_metadata,
        }
        if panel_mode:
            hurdle_metrics.update(
                panel_ranking_metrics(
                    test[regression_target],
                    hurdle_returns,
                    test["decision_date"],
                    test["entity"],
                )
            )
        result["classification_hurdle_return"][model_name] = hurdle_metrics
        groups = test["group"] if "group" in test else [None] * len(test)
        for date, entity, group, target, probability in zip(
            test["decision_date"],
            test["entity"],
            groups,
            test[classification_target],
            probabilities,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "window": window.name,
                    "task": "classification",
                    "model": model_name,
                    "decision_date": date.date().isoformat(),
                    "entity": str(entity),
                    "group": str(group) if group is not None else None,
                    "target": int(target),
                    "prediction": float(probability),
                }
            )
        for date, entity, group, target, expected_return in zip(
            test["decision_date"],
            test["entity"],
            groups,
            test[regression_target],
            hurdle_returns,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "window": window.name,
                    "task": "classification_expected_return",
                    "model": model_name,
                    "decision_date": date.date().isoformat(),
                    "entity": str(entity),
                    "group": str(group) if group is not None else None,
                    "target": float(target),
                    "prediction": float(expected_return),
                }
            )
    return result, prediction_rows


def _aggregate_development(windows: Sequence[dict[str, Any]], task: str) -> dict[str, Any]:
    metric = "rmse" if task == "regression" else "brier_score"
    models = sorted(windows[0][task])
    aggregate: dict[str, Any] = {}
    for model in models:
        metrics = windows[0][task][model].keys()
        aggregate[model] = {}
        for name in metrics:
            values = [window[task][model][name] for window in windows]
            if all(isinstance(value, (int, float)) for value in values):
                aggregate[model][f"mean_{name}"] = float(np.mean(values))
        aggregate[model]["development_folds"] = len(windows)
    ranking = sorted(models, key=lambda model: aggregate[model][f"mean_{metric}"])
    for index, model in enumerate(ranking, start=1):
        aggregate[model]["selection_rank"] = index
    result = {
        "selection_metric": f"mean_{metric}",
        "selected_model": ranking[0],
        "models": aggregate,
    }
    if task == "regression" and all(
        "mean_daily_rank_correlation" in aggregate[model] for model in models
    ):
        rank_order = sorted(
            models,
            key=lambda model: aggregate[model]["mean_daily_rank_correlation"],
            reverse=True,
        )
        result.update(
            {
                "ranking_selection_metric": "mean_daily_rank_correlation",
                "selected_ranking_model": rank_order[0],
            }
        )
        for index, model in enumerate(rank_order, start=1):
            aggregate[model]["ranking_selection_rank"] = index
    if task == "classification" and all(
        "mean_daily_roc_auc" in aggregate[model] for model in models
    ):
        rank_order = sorted(
            models,
            key=lambda model: aggregate[model]["mean_daily_roc_auc"],
            reverse=True,
        )
        result.update(
            {
                "ranking_selection_metric": "mean_daily_roc_auc",
                "selected_ranking_model": rank_order[0],
            }
        )
        for index, model in enumerate(rank_order, start=1):
            aggregate[model]["ranking_selection_rank"] = index
    return result


def evaluate_experiment(
    dataset: pd.DataFrame,
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one entity/target/horizon or return a structured unavailable result."""
    regression_target = str(experiment["regression_target_template"]).format(horizon=horizon)
    classification_target = str(experiment["classification_target_template"]).format(
        horizon=horizon
    )
    feature_columns = resolve_feature_columns(experiment["feature_sets"])
    missing = [
        column
        for column in [*feature_columns, regression_target, classification_target]
        if column not in dataset
    ]
    identity = {
        "name": str(experiment["name"]),
        "mode": str(experiment.get("mode", "entity")),
        "scope": str(experiment["scope"]),
        "entity": str(experiment["entity"]),
        "horizon_days": int(horizon),
        "regression_target": regression_target,
        "classification_target": classification_target,
        "feature_columns": feature_columns,
    }
    if missing:
        return {**identity, "status": "unavailable", "reason": f"Missing columns: {missing}"}, []
    panel_mode = identity["mode"] == "panel"
    if panel_mode:
        frame = dataset[dataset["scope"] == experiment["scope"]].copy()
        if experiment.get("group_filter"):
            frame = frame[frame["group"] == experiment["group_filter"]]
        if experiment.get("entities"):
            frame = frame[frame["entity"].isin(experiment["entities"])]
    else:
        frame = dataset[
            (dataset["scope"] == experiment["scope"]) & (dataset["entity"] == experiment["entity"])
        ].copy()
    required = [*feature_columns, regression_target, classification_target]
    frame = frame.dropna(subset=required).sort_values("decision_date")
    if panel_mode and not frame.empty:
        minimum_entities = int(experiment["minimum_panel_entities"])
        entity_counts = frame.groupby("decision_date")["entity"].nunique()
        complete_dates = entity_counts[entity_counts >= minimum_entities].index
        frame = frame[frame["decision_date"].isin(complete_dates)]
    if frame.empty:
        return {**identity, "status": "unavailable", "reason": "No complete causal rows"}, []
    first_date = frame["decision_date"].min()
    last_date = frame["decision_date"].max()
    final_window = build_final_window(
        first_date,
        last_date,
        minimum_training_days=int(config["minimum_training_days"]),
        calibration_days=int(config["calibration_days"]),
        final_holdout_days=int(config["final_holdout_days"]),
        purge_days=int(config["purge_days"]),
    )
    if final_window is None:
        return {
            **identity,
            "status": "unavailable",
            "reason": "History cannot satisfy minimum train, calibration, purges, and final holdout",
            "usable_rows": int(len(frame)),
            "usable_first_date": first_date.date().isoformat(),
            "usable_last_date": last_date.date().isoformat(),
        }, []
    development_end = final_window.calibration_start - pd.Timedelta(
        days=int(config["purge_days"]) + 1
    )
    development_windows = build_development_windows(
        first_date,
        development_end,
        minimum_training_days=int(config["minimum_training_days"]),
        calibration_days=int(config["calibration_days"]),
        test_days=int(config["test_days"]),
        step_days=int(config["step_days"]),
        purge_days=int(config["purge_days"]),
    )
    if not development_windows:
        return {
            **identity,
            "status": "unavailable",
            "reason": "No complete development fold remains before the reserved final holdout",
            "usable_rows": int(len(frame)),
            "usable_first_date": first_date.date().isoformat(),
            "usable_last_date": last_date.date().isoformat(),
        }, []
    window_results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    try:
        for window in development_windows:
            result, rows = _evaluate_window(
                frame,
                window,
                config=config,
                regression_target=regression_target,
                classification_target=classification_target,
                feature_columns=feature_columns,
                horizon=horizon,
                panel_mode=panel_mode,
            )
            window_results.append(result)
            predictions.extend(rows)
        regression_selection = _aggregate_development(window_results, "regression")
        classification_selection = _aggregate_development(window_results, "classification")
        final_result, final_predictions = _evaluate_window(
            frame,
            final_window,
            config=config,
            regression_target=regression_target,
            classification_target=classification_target,
            feature_columns=feature_columns,
            horizon=horizon,
            panel_mode=panel_mode,
        )
        predictions.extend(final_predictions)
    except ValueError as exc:
        return {
            **identity,
            "status": "unavailable",
            "reason": str(exc),
            "usable_rows": int(len(frame)),
            "usable_first_date": first_date.date().isoformat(),
            "usable_last_date": last_date.date().isoformat(),
        }, []
    for row in predictions:
        row.update({"experiment": identity["name"], "horizon_days": horizon})
    return {
        **identity,
        "status": "complete",
        "usable_rows": int(len(frame)),
        "usable_first_date": first_date.date().isoformat(),
        "usable_last_date": last_date.date().isoformat(),
        "usable_entities": sorted(str(value) for value in frame["entity"].unique()),
        "development": {
            "windows": window_results,
            "regression_selection": regression_selection,
            "classification_selection": classification_selection,
        },
        "final_confirmation": {
            **final_result,
            "selected_regression_model": regression_selection["selected_model"],
            "selected_classification_model": classification_selection["selected_model"],
            "selection_was_frozen_before_final_confirmation": True,
        },
    }, predictions


def assess_rotation_availability(
    dataset: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed when the frozen dataset cannot support rotation claims."""
    minimum_days = int(config["minimum_training_days"])
    longest_horizon = max(int(value) for value in config["forecast_horizons_days"])
    availability_target = f"target_return_{longest_horizon}d"
    required_span = (
        minimum_days
        + int(config["calibration_days"])
        + int(config["test_days"])
        + int(config["final_holdout_days"])
        + 3 * int(config["purge_days"])
    )
    complete_assets: list[str] = []
    asset_rows = dataset[dataset["scope"] == "asset"].copy()
    for entity, rows in asset_rows.groupby("entity"):
        dates = rows.dropna(subset=list(BASE_FEATURE_COLUMNS) + [availability_target])[
            "decision_date"
        ]
        if not dates.empty and (dates.max() - dates.min()).days + 1 >= required_span:
            complete_assets.append(str(entity))
    complete_groups: list[str] = []
    group_rows = dataset[dataset["scope"] == "group"].copy()
    for entity, rows in group_rows.groupby("entity"):
        dates = rows.dropna(subset=list(BASE_FEATURE_COLUMNS) + [availability_target])[
            "decision_date"
        ]
        if not dates.empty and (dates.max() - dates.min()).days + 1 >= required_span:
            complete_groups.append(str(entity))
    asset_minimum = int(config["unavailable_scope_checks"]["cross_sectional_minimum_assets"])
    group_minimum = int(config["unavailable_scope_checks"]["group_rotation_minimum_groups"])
    return {
        "cross_sectional_asset_rotation": {
            "status": "available" if len(complete_assets) >= asset_minimum else "unavailable",
            "required_entities": asset_minimum,
            "qualifying_entities": complete_assets,
            "reason": (
                None
                if len(complete_assets) >= asset_minimum
                else "Too few assets satisfy the unchanged long-history protocol"
            ),
        },
        "group_rotation": {
            "status": "available" if len(complete_groups) >= group_minimum else "unavailable",
            "required_entities": group_minimum,
            "qualifying_entities": complete_groups,
            "reason": (
                None
                if len(complete_groups) >= group_minimum
                else "Too few groups satisfy the unchanged long-history protocol"
            ),
        },
        "required_calendar_span_days": required_span,
    }


def run_experiments(
    dataset: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    if config.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment schema: {config.get('schema_version')}")
    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for experiment in config["experiments"]:
        for horizon in config["forecast_horizons_days"]:
            result, rows = evaluate_experiment(dataset, config, experiment, int(horizon))
            results.append(result)
            predictions.extend(rows)
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "protocol": {
            key: config[key]
            for key in (
                "random_seed",
                "purge_days",
                "minimum_training_days",
                "calibration_days",
                "test_days",
                "step_days",
                "final_holdout_days",
                "minimum_partition_rows",
                "probability_calibration",
                "selection_policy",
            )
        },
        "rotation_availability": assess_rotation_availability(dataset, config),
        "experiments": results,
    }, pd.DataFrame(predictions)
