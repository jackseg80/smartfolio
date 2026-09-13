import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.forecasting.okx_l2_evaluation_feasibility import (
    TARGET_CONTRACT,
    evaluate_l2_evaluation_feasibility,
    write_l2_evaluation_feasibility_artifact,
)
from services.forecasting.okx_l2_pilot import DAY_MS, file_sha256

DATES = ["2024-07-01", "2024-07-02"]
INSTRUMENTS = ["BTC-USDT", "ETH-USDT"]
SLOTS = 2
INTERVAL_MS = DAY_MS // SLOTS


def _day_start_ms(date_utc: str) -> int:
    value = datetime.strptime(date_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def _config() -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-okx-l2-evaluation-feasibility-v1",
        "source_artifact_id": "features-fixture",
        "source_result_sha256": "",
        "source_table_path": "l2_features.csv",
        "source_table_bytes": 0,
        "source_table_sha256": "",
        "dates_utc": DATES,
        "instruments": INSTRUMENTS,
        "grid_interval_ms": INTERVAL_MS,
        "expected_grid_slots_per_group": SLOTS,
        "minimum_available_slots_per_group": SLOTS,
        "expected_total_rows": len(DATES) * len(INSTRUMENTS) * SLOTS,
        "expected_available_rows": len(DATES) * len(INSTRUMENTS) * SLOTS,
        "expected_missing_rows": 0,
        "forecast_horizons_days": [7, 30],
        "target_contract": TARGET_CONTRACT,
        "evaluation_unit": "utc_observation_date",
        "entity_date_eligibility": "available_slots_at_least_configured_minimum",
        "partition_rule": "all_rows_from_same_utc_date_remain_in_same_partition",
        "intraday_rows_count_as_independent_samples": False,
        "feature_reduction_policy": "not_selected_in_feasibility_lot",
        "purge_days": 0,
        "minimum_training_calendar_days": 1,
        "calibration_calendar_days": 1,
        "development_test_calendar_days": 1,
        "final_holdout_calendar_days": 1,
        "required_purge_boundaries": 3,
        "minimum_calendar_span_days": 4,
        "minimum_independent_dates_per_partition": 1,
        "required_partition_count": 4,
        "minimum_total_independent_dates": 4,
        "minimum_panel_entities": 2,
        "model_policy": "forbidden_unless_all_feasibility_checks_pass",
        "target_materialization_policy": "forbidden_in_this_lot",
        "network_policy": "offline_only",
    }


def _write_source(
    root: Path, *, unavailable_keys: set[tuple[str, str, int]] | None = None
) -> tuple[Path, dict[str, object]]:
    unavailable = unavailable_keys or set()
    source_root = root / "source"
    source_root.mkdir(parents=True)
    table_path = source_root / "l2_features.csv"
    rows = []
    for date in DATES:
        day_start = _day_start_ms(date)
        for instrument in INSTRUMENTS:
            for slot in range(SLOTS):
                key = (date, instrument, day_start + slot * INTERVAL_MS)
                rows.append(
                    {
                        "date_utc": date,
                        "instrument": instrument,
                        "grid_timestamp_ms": key[2],
                        "available": key not in unavailable,
                    }
                )
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date_utc", "instrument", "grid_timestamp_ms", "available"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema_version": "crypto-forecast-okx-l2-features-v1",
        "decision": "GO_TECHNICAL_L2_FEATURES",
        "future_values_used": False,
        "targets_created": False,
        "model_trained": False,
    }
    result_path = source_root / "l2_feature_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    config = _config()
    config["source_result_sha256"] = file_sha256(result_path)
    config["source_table_bytes"] = table_path.stat().st_size
    config["source_table_sha256"] = file_sha256(table_path)
    config["expected_available_rows"] = len(rows) - len(unavailable)
    config["expected_missing_rows"] = len(unavailable)
    (source_root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "features-fixture",
                "result_sha256": config["source_result_sha256"],
                "feature_table_sha256": config["source_table_sha256"],
            }
        ),
        encoding="utf-8",
    )
    return source_root, config


def test_intraday_rows_do_not_inflate_independent_dates(tmp_path: Path):
    source_root, config = _write_source(tmp_path)

    result = evaluate_l2_evaluation_feasibility(source_root, config)

    assert result["decision"] == "NO_GO_PREDICTIVE_EVALUATION"
    assert result["integrity_passed"] is True
    assert result["observed"]["raw_grid_rows"] == 8
    assert result["observed"]["eligible_entity_date_units"] == 4
    assert result["observed"]["independent_utc_dates"] == 2
    assert result["observed"]["calendar_span_days"] == 2
    checks = {item["name"]: item for item in result["checks"]}
    assert checks["independent_observed_dates"]["shortfall"] == 2


def test_ineligible_entity_date_fails_closed_without_corrupting_integrity(tmp_path: Path):
    missing_key = (DATES[0], INSTRUMENTS[1], _day_start_ms(DATES[0]))
    source_root, config = _write_source(tmp_path, unavailable_keys={missing_key})

    result = evaluate_l2_evaluation_feasibility(source_root, config)

    assert result["integrity_passed"] is True
    assert result["evaluation_feasible"] is False
    assert result["observed"]["eligible_entity_date_units"] == 3
    assert result["observed"]["common_eligible_dates"] == [DATES[1]]


def test_rejects_source_provenance_mismatch(tmp_path: Path):
    source_root, config = _write_source(tmp_path)
    (source_root / "l2_feature_result.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="result SHA-256"):
        evaluate_l2_evaluation_feasibility(source_root, config)


def test_rejects_intraday_pseudoreplication_policy(tmp_path: Path):
    source_root, config = _write_source(tmp_path)
    config["intraday_rows_count_as_independent_samples"] = True

    with pytest.raises(ValueError, match="cannot count as independent"):
        evaluate_l2_evaluation_feasibility(source_root, config)


def test_feasibility_artifact_is_reproducible(tmp_path: Path):
    source_root, config = _write_source(tmp_path)
    result = evaluate_l2_evaluation_feasibility(source_root, config)

    first = write_l2_evaluation_feasibility_artifact(
        result,
        config_sha256="1" * 64,
        feasibility_code_sha256="2" * 64,
        output_root=tmp_path / "first",
    )
    second = write_l2_evaluation_feasibility_artifact(
        result,
        config_sha256="1" * 64,
        feasibility_code_sha256="2" * 64,
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert file_sha256(first / "evaluation_feasibility_result.json") == file_sha256(
        second / "evaluation_feasibility_result.json"
    )
