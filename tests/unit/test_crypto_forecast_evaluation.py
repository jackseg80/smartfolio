import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.forecasting.evaluation import (
    apply_platt_calibrator,
    build_development_windows,
    build_final_window,
    classification_metrics,
    classification_hurdle_expected_returns,
    expected_calibration_error,
    fit_platt_calibrator,
    load_verified_dataset,
    panel_classification_ranking_metrics,
    panel_ranking_metrics,
    regression_metrics,
    run_experiments,
)


def test_development_windows_keep_both_purges_and_fixed_durations():
    windows = build_development_windows(
        "2020-01-01",
        "2024-12-31",
        minimum_training_days=730,
        calibration_days=183,
        test_days=183,
        step_days=183,
        purge_days=30,
    )

    assert len(windows) >= 2
    for window in windows:
        assert (window.train_end - window.train_start).days + 1 >= 730
        assert (window.calibration_start - window.train_end).days == 31
        assert (window.calibration_end - window.calibration_start).days + 1 == 183
        assert (window.test_start - window.calibration_end).days == 31
        assert (window.test_end - window.test_start).days + 1 == 183
    assert (windows[1].test_start - windows[0].test_start).days == 183
    assert windows[1].train_start == windows[0].train_start


def test_final_window_reserves_trailing_year_without_shortening_training():
    window = build_final_window(
        "2020-01-01",
        "2025-12-31",
        minimum_training_days=730,
        calibration_days=183,
        final_holdout_days=365,
        purge_days=30,
    )

    assert window is not None
    assert window.test_end == pd.Timestamp("2025-12-31")
    assert (window.test_end - window.test_start).days + 1 == 365
    assert (window.test_start - window.calibration_end).days == 31
    assert (window.calibration_start - window.train_end).days == 31


def test_final_window_returns_none_when_protocol_does_not_fit():
    assert (
        build_final_window(
            "2024-01-01",
            "2025-12-31",
            minimum_training_days=730,
            calibration_days=183,
            final_holdout_days=365,
            purge_days=30,
        )
        is None
    )


def test_platt_calibrator_returns_bounded_probabilities_and_requires_two_classes():
    probabilities = np.linspace(0.05, 0.95, 20)
    targets = np.array([0] * 8 + [1] * 12)
    calibrator = fit_platt_calibrator(targets, probabilities)
    calibrated = apply_platt_calibrator(calibrator, [0.01, 0.5, 0.99])

    assert np.all((calibrated > 0.0) & (calibrated < 1.0))
    assert calibrated[0] < calibrated[1] < calibrated[2]
    with pytest.raises(ValueError, match="both target classes"):
        fit_platt_calibrator(np.ones(4), [0.2, 0.3, 0.4, 0.5])


def test_metrics_are_exact_for_perfect_predictions():
    regression = regression_metrics([-1.0, 1.0], [-1.0, 1.0])
    classification = classification_metrics([0, 1], [0.0, 1.0])

    assert regression["rmse"] == pytest.approx(0.0)
    assert regression["directional_accuracy"] == pytest.approx(1.0)
    assert classification["brier_score"] == pytest.approx(0.0)
    assert classification["accuracy_at_0_5"] == pytest.approx(1.0)
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == pytest.approx(0.0)


def test_hurdle_expected_return_uses_training_class_means():
    expected, metadata = classification_hurdle_expected_returns(
        [-0.10, -0.20, 0.20, 0.40],
        [0, 0, 1, 1],
        [0.0, 0.5, 1.0],
    )

    assert expected.tolist() == pytest.approx([-0.15, 0.075, 0.30])
    assert metadata == {
        "training_negative_mean_return": pytest.approx(-0.15),
        "training_positive_mean_return": pytest.approx(0.30),
    }


def test_panel_ranking_metrics_average_daily_cross_sectional_ranks():
    metrics = panel_ranking_metrics(
        [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        [10.0, 20.0, 30.0, 30.0, 20.0, 10.0],
        ["2025-01-01"] * 3 + ["2025-01-02"] * 3,
        ["A", "B", "C"] * 2,
    )

    assert metrics["daily_rank_correlation"] == pytest.approx(1.0)
    assert metrics["ranked_dates"] == 2


def test_panel_classification_ranking_metrics_average_daily_auc():
    metrics = panel_classification_ranking_metrics(
        [0, 1, 0, 1, 0, 1],
        [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        ["2025-01-01"] * 3 + ["2025-01-02"] * 3,
        ["A", "B", "C"] * 2,
    )

    assert metrics["daily_roc_auc"] == pytest.approx(1.0)
    assert metrics["ranked_dates"] == 2


def test_verified_dataset_rejects_content_hash_mismatch(tmp_path: Path):
    dataset_path = tmp_path / "forecast_dataset.csv"
    dataset_path.write_text("decision_date,value\n2025-01-01,1\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"dataset_file": dataset_path.name, "dataset_sha256": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_dataset(tmp_path)


def test_run_experiments_keeps_final_holdout_out_of_model_selection():
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    signal = np.sin(np.arange(len(dates)) / 13.0)
    frame = pd.DataFrame(
        {
            "decision_date": dates,
            "scope": "asset",
            "entity": "BTC",
            "target_return_7d": signal * 0.05,
            "label_up_7d": (signal >= 0.0).astype(int),
        }
    )
    for index, column in enumerate(
        (
            "past_return_7d",
            "past_return_30d",
            "past_return_90d",
            "distance_sma_30d",
            "distance_sma_90d",
            "distance_sma_200d",
            "past_volatility_7d",
            "past_volatility_30d",
            "past_volatility_60d",
            "drawdown_from_90d_peak",
        ),
        start=1,
    ):
        frame[column] = signal + index * np.arange(len(dates)) / 10000.0
    config = {
        "schema_version": "crypto-forecast-experiment-v1",
        "random_seed": 42,
        "purge_days": 7,
        "minimum_training_days": 100,
        "calibration_days": 30,
        "test_days": 30,
        "step_days": 30,
        "final_holdout_days": 60,
        "minimum_partition_rows": 20,
        "probability_calibration": "platt_on_distinct_chronological_calibration_period",
        "selection_policy": {
            "regression": "lowest_mean_development_rmse",
            "classification": "lowest_mean_development_brier_score",
        },
        "regression_models": {"zero": {}, "momentum": {}},
        "classification_models": {"train_prevalence": {}},
        "experiments": [
            {
                "name": "btc_test",
                "scope": "asset",
                "entity": "BTC",
                "regression_target_template": "target_return_{horizon}d",
                "classification_target_template": "label_up_{horizon}d",
                "feature_sets": ["base"],
            }
        ],
        "forecast_horizons_days": [7],
        "unavailable_scope_checks": {
            "cross_sectional_minimum_assets": 5,
            "group_rotation_minimum_groups": 3,
        },
    }

    results, predictions = run_experiments(frame, config)

    experiment = results["experiments"][0]
    assert experiment["status"] == "complete"
    assert experiment["development"]["regression_selection"]["selected_model"] in {
        "zero",
        "momentum",
    }
    assert experiment["final_confirmation"]["selection_was_frozen_before_final_confirmation"]
    last_development_end = pd.Timestamp(
        experiment["development"]["windows"][-1]["window"]["test_end"]
    )
    final_calibration_start = pd.Timestamp(
        experiment["final_confirmation"]["window"]["calibration_start"]
    )
    assert last_development_end < final_calibration_start
    assert set(predictions["window"]) >= {"development_01", "final_confirmation"}
