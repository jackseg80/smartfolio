"""Offline feasibility gate for causal evaluation of OKX L2 features."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.forecasting.okx_l2_features import OKX_L2_FEATURES_SCHEMA_VERSION
from services.forecasting.okx_l2_pilot import DAY_MS, file_sha256

OKX_L2_EVALUATION_FEASIBILITY_SCHEMA_VERSION = "crypto-forecast-okx-l2-evaluation-feasibility-v1"

TARGET_CONTRACT = {
    "price_source": "pinned_okx_spot_1Dutc_close_required_before_target_materialization",
    "entry_price": "utc_close_at_end_of_observation_day",
    "exit_price": "utc_close_exactly_horizon_calendar_days_after_entry",
    "availability": "strictly_after_exit_daily_bar_close",
    "absolute_return": "exit_price_divided_by_entry_price_minus_one",
    "defensive_excess_return": "absolute_return_minus_zero_return_usd_cash",
    "relative_btc_return": "instrument_return_minus_btc_return_same_dates",
    "classification_labels": (
        "derived_only_after_returns_exist_and_calibrated_on_distinct_chronological_period"
    ),
    "missing_policy": "explicit_unavailable_without_nearest_fill_or_interpolation",
}


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


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _day_start_ms(date_utc: str) -> int:
    day = datetime.strptime(date_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(day.timestamp() * 1000)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != OKX_L2_EVALUATION_FEASIBILITY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported L2 evaluation feasibility schema: {config.get('schema_version')}"
        )
    dates = [str(value) for value in config["dates_utc"]]
    instruments = [str(value) for value in config["instruments"]]
    if dates != sorted(set(dates)) or len(instruments) != len(set(instruments)):
        raise ValueError("Frozen dates must be sorted and dates/instruments must be unique")
    for date in dates:
        _day_start_ms(date)
    interval_ms = int(config["grid_interval_ms"])
    slots = int(config["expected_grid_slots_per_group"])
    if interval_ms <= 0 or DAY_MS % interval_ms or DAY_MS // interval_ms != slots:
        raise ValueError("Grid interval and expected slot count are inconsistent")
    minimum_slots = int(config["minimum_available_slots_per_group"])
    if not 1 <= minimum_slots <= slots:
        raise ValueError("Minimum group coverage must fit inside the frozen grid")
    expected_rows = len(dates) * len(instruments) * slots
    if int(config["expected_total_rows"]) != expected_rows:
        raise ValueError("Expected row count does not match the frozen grid")
    if (
        int(config["expected_available_rows"]) + int(config["expected_missing_rows"])
        != expected_rows
    ):
        raise ValueError("Expected available and missing rows are inconsistent")
    if list(config["forecast_horizons_days"]) != [7, 30]:
        raise ValueError("Forecast horizons must remain frozen at 7 and 30 days")
    if config.get("target_contract") != TARGET_CONTRACT:
        raise ValueError("Target contract differs from the frozen causal definitions")
    if config.get("evaluation_unit") != "utc_observation_date":
        raise ValueError("Evaluation unit must be the UTC observation date")
    if config.get("entity_date_eligibility") != "available_slots_at_least_configured_minimum":
        raise ValueError("Entity-date eligibility differs from the frozen rule")
    if config.get("partition_rule") != "all_rows_from_same_utc_date_remain_in_same_partition":
        raise ValueError("All rows from one UTC date must remain in one partition")
    if config.get("intraday_rows_count_as_independent_samples") is not False:
        raise ValueError("Intraday rows cannot count as independent samples")
    if config.get("feature_reduction_policy") != "not_selected_in_feasibility_lot":
        raise ValueError("Feature reduction cannot be selected in this feasibility lot")

    partition_calendar_days = [
        int(config["minimum_training_calendar_days"]),
        int(config["calibration_calendar_days"]),
        int(config["development_test_calendar_days"]),
        int(config["final_holdout_calendar_days"]),
    ]
    purge_days = int(config["purge_days"])
    purge_boundaries = int(config["required_purge_boundaries"])
    if (
        any(value <= 0 for value in partition_calendar_days)
        or purge_days < 0
        or purge_boundaries < 0
    ):
        raise ValueError("Partition durations must be positive and purges non-negative")
    if int(config["minimum_calendar_span_days"]) != sum(partition_calendar_days) + (
        purge_days * purge_boundaries
    ):
        raise ValueError("Minimum calendar span must include every frozen purge")
    minimum_independent = int(config["minimum_independent_dates_per_partition"])
    partition_count = int(config["required_partition_count"])
    if minimum_independent <= 0 or partition_count != 4:
        raise ValueError("Independent-date requirements must cover four positive partitions")
    if int(config["minimum_total_independent_dates"]) != minimum_independent * partition_count:
        raise ValueError("Minimum independent dates must equal the frozen partition total")
    if int(config["minimum_panel_entities"]) <= 0:
        raise ValueError("Minimum panel entity count must be positive")
    if config.get("model_policy") != "forbidden_unless_all_feasibility_checks_pass":
        raise ValueError("Model policy must fail closed")
    if config.get("target_materialization_policy") != "forbidden_in_this_lot":
        raise ValueError("Target materialization is forbidden in this lot")
    if config.get("network_policy") != "offline_only":
        raise ValueError("Feasibility analysis must remain offline")
    table_name = str(config["source_table_path"])
    if Path(table_name).name != table_name:
        raise ValueError("Source table path must be a single safe filename")


def _validate_source_artifact(source_root: Path, config: Mapping[str, Any]) -> Path:
    result_path = source_root / "l2_feature_result.json"
    manifest_path = source_root / "manifest.json"
    table_path = source_root / str(config["source_table_path"])
    if file_sha256(result_path) != str(config["source_result_sha256"]):
        raise ValueError("L2 feature result SHA-256 does not match the frozen input")
    result = _load_json_object(result_path, "L2 feature result")
    if (
        result.get("schema_version") != OKX_L2_FEATURES_SCHEMA_VERSION
        or result.get("decision") != "GO_TECHNICAL_L2_FEATURES"
        or result.get("future_values_used") is not False
        or result.get("targets_created") is not False
        or result.get("model_trained") is not False
    ):
        raise ValueError("L2 feature result is not an approved target-free input")
    manifest = _load_json_object(manifest_path, "L2 feature manifest")
    if (
        manifest.get("artifact_id") != config["source_artifact_id"]
        or manifest.get("result_sha256") != config["source_result_sha256"]
        or manifest.get("feature_table_sha256") != config["source_table_sha256"]
    ):
        raise ValueError("L2 feature manifest does not match the frozen artifact")
    if table_path.stat().st_size != int(config["source_table_bytes"]):
        raise ValueError("L2 feature table size does not match the frozen input")
    if file_sha256(table_path) != str(config["source_table_sha256"]):
        raise ValueError("L2 feature table SHA-256 does not match the frozen input")
    return table_path


def _load_grid(
    table_path: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dates = [str(value) for value in config["dates_utc"]]
    instruments = [str(value) for value in config["instruments"]]
    slots = int(config["expected_grid_slots_per_group"])
    interval_ms = int(config["grid_interval_ms"])
    expected_keys = [
        (date, instrument, _day_start_ms(date) + slot * interval_ms)
        for date in dates
        for instrument in instruments
        for slot in range(slots)
    ]
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date_utc", "instrument", "grid_timestamp_ms", "available"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("L2 feature table is missing required metadata fields")
        for index, raw in enumerate(reader):
            key = (raw["date_utc"], raw["instrument"], int(raw["grid_timestamp_ms"]))
            if key in seen:
                raise ValueError(f"Duplicate L2 feature grid key: {key}")
            seen.add(key)
            availability = raw["available"].lower()
            if availability not in {"true", "false"}:
                raise ValueError(f"Invalid availability at table row {index + 2}")
            rows.append(
                {
                    "date_utc": key[0],
                    "instrument": key[1],
                    "grid_timestamp_ms": key[2],
                    "available": availability == "true",
                }
            )
    actual_keys = [
        (str(row["date_utc"]), str(row["instrument"]), int(row["grid_timestamp_ms"]))
        for row in rows
    ]
    if actual_keys != expected_keys:
        raise ValueError("L2 feature table keys or order differ from the frozen grid")

    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "available_rows": 0}
    )
    for row in rows:
        group = (str(row["date_utc"]), str(row["instrument"]))
        counts[group]["rows"] += 1
        counts[group]["available_rows"] += int(bool(row["available"]))
    group_summaries: list[dict[str, object]] = []
    minimum_available = int(config["minimum_available_slots_per_group"])
    for date in dates:
        for instrument in instruments:
            count = counts[(date, instrument)]
            group_summaries.append(
                {
                    "date_utc": date,
                    "instrument": instrument,
                    "rows": count["rows"],
                    "available_rows": count["available_rows"],
                    "missing_rows": count["rows"] - count["available_rows"],
                    "eligible_entity_date": (
                        count["rows"] == slots and count["available_rows"] >= minimum_available
                    ),
                }
            )
    return rows, group_summaries


def _check(name: str, actual: int, required: int, *, operator: str = "at_least") -> dict[str, Any]:
    if operator == "equal":
        passed = actual == required
    elif operator == "at_least":
        passed = actual >= required
    else:
        raise ValueError(f"Unsupported feasibility comparison: {operator}")
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "actual": actual,
        "required": required,
        "shortfall": max(required - actual, 0),
    }


def evaluate_l2_evaluation_feasibility(
    source_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate data sufficiency without loading targets or training a model."""

    _validate_config(config)
    table_path = _validate_source_artifact(source_root, config)
    rows, groups = _load_grid(table_path, config)
    available_rows = sum(bool(row["available"]) for row in rows)
    missing_rows = len(rows) - available_rows
    eligible_groups = [group for group in groups if group["eligible_entity_date"]]
    eligible_by_date: dict[str, set[str]] = defaultdict(set)
    for group in eligible_groups:
        eligible_by_date[str(group["date_utc"])].add(str(group["instrument"]))
    instruments = [str(value) for value in config["instruments"]]
    common_eligible_dates = [
        date for date in config["dates_utc"] if eligible_by_date[str(date)] == set(instruments)
    ]
    if common_eligible_dates:
        first_day = datetime.strptime(str(common_eligible_dates[0]), "%Y-%m-%d")
        last_day = datetime.strptime(str(common_eligible_dates[-1]), "%Y-%m-%d")
        calendar_span_days = (last_day - first_day).days + 1
    else:
        calendar_span_days = 0

    checks = [
        _check("total_rows", len(rows), int(config["expected_total_rows"]), operator="equal"),
        _check(
            "available_rows",
            available_rows,
            int(config["expected_available_rows"]),
            operator="equal",
        ),
        _check(
            "missing_rows",
            missing_rows,
            int(config["expected_missing_rows"]),
            operator="equal",
        ),
        _check(
            "eligible_entity_dates",
            len(eligible_groups),
            len(config["dates_utc"]) * len(instruments),
            operator="equal",
        ),
        _check(
            "independent_observed_dates",
            len(common_eligible_dates),
            int(config["minimum_total_independent_dates"]),
        ),
        _check(
            "calendar_span_days",
            calendar_span_days,
            int(config["minimum_calendar_span_days"]),
        ),
        _check("panel_entities", len(instruments), int(config["minimum_panel_entities"])),
    ]
    integrity_check_names = {
        "total_rows",
        "available_rows",
        "missing_rows",
    }
    integrity_passed = all(
        check["status"] == "PASS" for check in checks if check["name"] in integrity_check_names
    )
    feasibility_passed = all(check["status"] == "PASS" for check in checks)
    return {
        "schema_version": OKX_L2_EVALUATION_FEASIBILITY_SCHEMA_VERSION,
        "source": {
            "artifact_id": config["source_artifact_id"],
            "result_sha256": config["source_result_sha256"],
            "table_sha256": config["source_table_sha256"],
        },
        "targets": {
            "horizons_days": list(config["forecast_horizons_days"]),
            "contract": dict(config["target_contract"]),
            "materialized": False,
            "future_prices_read": False,
        },
        "evaluation_protocol": {
            "unit": config["evaluation_unit"],
            "partition_rule": config["partition_rule"],
            "intraday_rows_count_as_independent_samples": False,
            "purge_days": int(config["purge_days"]),
            "minimum_training_calendar_days": int(config["minimum_training_calendar_days"]),
            "calibration_calendar_days": int(config["calibration_calendar_days"]),
            "development_test_calendar_days": int(config["development_test_calendar_days"]),
            "final_holdout_calendar_days": int(config["final_holdout_calendar_days"]),
            "minimum_calendar_span_days": int(config["minimum_calendar_span_days"]),
            "minimum_independent_dates_per_partition": int(
                config["minimum_independent_dates_per_partition"]
            ),
            "required_partition_count": int(config["required_partition_count"]),
            "minimum_total_independent_dates": int(config["minimum_total_independent_dates"]),
            "minimum_panel_entities": int(config["minimum_panel_entities"]),
        },
        "observed": {
            "raw_grid_rows": len(rows),
            "available_grid_rows": available_rows,
            "missing_grid_rows": missing_rows,
            "eligible_entity_date_units": len(eligible_groups),
            "independent_utc_dates": len(common_eligible_dates),
            "common_eligible_dates": [str(value) for value in common_eligible_dates],
            "calendar_span_days": calendar_span_days,
            "panel_entities": len(instruments),
            "groups": groups,
        },
        "checks": checks,
        "integrity_passed": integrity_passed,
        "evaluation_feasible": feasibility_passed,
        "decision": (
            "GO_TECHNICAL_PREDICTIVE_EVALUATION"
            if feasibility_passed
            else "NO_GO_PREDICTIVE_EVALUATION"
        ),
        "reason": (
            "All frozen integrity and independent-sample requirements pass"
            if feasibility_passed
            else "Frozen independent-day, calendar-span, or panel requirements fail"
        ),
        "feature_reduction_selected": False,
        "targets_created": False,
        "future_values_used": False,
        "model_trained": False,
        "network_used": False,
        "orders_placed": False,
        "production_touched": False,
    }


def write_l2_evaluation_feasibility_artifact(
    result: Mapping[str, Any],
    *,
    config_sha256: str,
    feasibility_code_sha256: str,
    output_root: Path,
) -> Path:
    identity = {
        **result,
        "config_sha256": config_sha256,
        "feasibility_code_sha256": feasibility_code_sha256,
    }
    artifact_id = (
        f"{OKX_L2_EVALUATION_FEASIBILITY_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"L2 evaluation feasibility artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        result_path = temporary_path / "evaluation_feasibility_result.json"
        result_path.write_bytes(_json_bytes(identity, pretty=True))
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_file": result_path.name,
            "result_sha256": file_sha256(result_path),
            "targets_created": False,
            "model_trained": False,
            "network_used": False,
            "credentials_used": False,
            "orders_placed": False,
            "production_touched": False,
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
