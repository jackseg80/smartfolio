import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import pytest

from services.forecasting.funding_model_comparison import (
    _evaluate_variant,
    assess_horizon,
    build_windows,
    validate_config,
    write_comparison_artifact,
)
from services.forecasting.evaluation import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict[str, object]:
    return json.loads(
        (PROJECT_ROOT / "config" / "crypto_forecast_funding_model_comparison.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_config_and_windows_match_the_preregistered_dates():
    config = _config()
    baseline, funding = validate_config(config)
    development, final = build_windows(config)

    assert len(baseline) == 17
    assert len(funding) == 16
    assert [window.to_dict() for window in development] == [
        {
            "name": "development_01",
            "train_start": "2022-07-01",
            "train_end": "2023-06-30",
            "calibration_start": "2023-07-31",
            "calibration_end": "2023-10-28",
            "test_start": "2023-11-28",
            "test_end": "2024-05-27",
            "purge_days": 30,
        },
        {
            "name": "development_02",
            "train_start": "2022-07-01",
            "train_end": "2023-12-29",
            "calibration_start": "2024-01-29",
            "calibration_end": "2024-04-27",
            "test_start": "2024-05-28",
            "test_end": "2024-11-25",
            "purge_days": 30,
        },
    ]
    assert final.to_dict()["test_start"] == "2025-08-02"
    assert final.to_dict()["test_end"] == "2026-08-01"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("variants", ["enriched"], "Variants"),
        ("forecast_horizons_days", [7], "7/30-day"),
        ("probability_calibration", "none", "probability_calibration"),
        ("network_policy", "online", "network_policy"),
    ],
)
def test_rejects_changes_to_the_frozen_protocol(field: str, value: object, message: str):
    config = _config()
    config[field] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_horizon_gate_requires_all_improvements_and_fold_stability():
    thresholds = _config()["go_thresholds"]
    development = [
        {"brier_improvement": 0.003, "daily_auc_improvement": 0.02},
        {"brier_improvement": 0.004, "daily_auc_improvement": 0.03},
    ]
    final = {"brier_improvement": 0.003, "daily_auc_improvement": 0.02}

    passing = assess_horizon(development, final, thresholds)
    unstable = assess_horizon(
        [
            {"brier_improvement": -0.002, "daily_auc_improvement": 0.02},
            {"brier_improvement": 0.008, "daily_auc_improvement": 0.03},
        ],
        final,
        thresholds,
    )

    assert passing["pass"] is True
    assert unstable["pass"] is False
    assert unstable["gates"]["no_material_single_fold_brier_degradation"] is False


def test_variant_preprocessing_is_fitted_on_training_only():
    config = _config()
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    frame = pd.DataFrame(
        [
            {
                "decision_date": day,
                "entity": entity,
                "x": float(index + offset),
                "label": (index + offset) % 2,
            }
            for index, day in enumerate(dates)
            for offset, entity in enumerate(("A", "B"))
        ]
    )
    train = frame[frame["decision_date"] <= "2024-01-20"]
    calibration = frame[frame["decision_date"].between("2024-01-21", "2024-01-30")]
    test = frame[frame["decision_date"] >= "2024-01-31"]

    metrics, probabilities = _evaluate_variant(
        train,
        calibration,
        test,
        target="label",
        feature_columns=["x"],
        config=config,
    )

    assert len(probabilities) == len(test)
    assert np.isfinite(probabilities).all()
    assert metrics["preprocessing"]["fit_end"] == "2024-01-20"
    assert metrics["preprocessing"]["fit_rows"] == len(train)


def test_comparison_artifact_is_reproducible(tmp_path: Path):
    result = {
        "schema_version": "crypto-forecast-funding-model-comparison-v1",
        "decision": "NO_GO_PREDICTIVE_FUNDING",
        "final_holdout_used_for_selection": False,
    }
    predictions = pd.DataFrame(
        [
            {
                "horizon_days": 7,
                "window": "development_01",
                "variant": "baseline",
                "decision_date": "2024-01-01",
                "entity": "BTC",
                "target": 0,
                "probability": 0.4,
            }
        ]
    )
    first = write_comparison_artifact(
        result,
        predictions,
        config_sha256="1" * 64,
        comparison_code_sha256="2" * 64,
        output_root=tmp_path / "first",
    )
    second = write_comparison_artifact(
        result,
        predictions,
        config_sha256="1" * 64,
        comparison_code_sha256="2" * 64,
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert file_sha256(first / "funding_model_comparison_result.json") == file_sha256(
        second / "funding_model_comparison_result.json"
    )
    assert file_sha256(first / "predictions.csv") == file_sha256(second / "predictions.csv")
