"""Causal time-grid normalization for pinned OKX L2 snapshot metrics."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.forecasting.okx_l2_pilot import DAY_MS, file_sha256

OKX_L2_NORMALIZATION_SCHEMA_VERSION = "crypto-forecast-okx-l2-normalization-v1"
SOURCE_ID_FIELDS = {"date_utc", "instrument", "timestamp_ms"}


@dataclass(frozen=True)
class SnapshotMetric:
    date_utc: str
    instrument: str
    timestamp_ms: int
    values: Mapping[str, str]


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


def _day_start_ms(date_utc: str) -> int:
    day = datetime.strptime(date_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(day.timestamp() * 1000)


def _validate_numeric_values(values: Mapping[str, str], row_number: int) -> None:
    if values.get("valid") != "True" or values.get("reason") != "ok":
        raise ValueError(f"Source snapshot is not valid at row {row_number}")
    for field, value in values.items():
        if field in {"valid", "reason"}:
            continue
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for {field} at row {row_number}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Non-finite value for {field} at row {row_number}")
    for field in ("bid_levels", "ask_levels"):
        level_count = int(values[field])
        if not 1 <= level_count <= 400:
            raise ValueError(f"Invalid {field} at row {row_number}")


def load_snapshot_metrics(
    metrics_path: Path, config: Mapping[str, Any]
) -> tuple[list[SnapshotMetric], list[str]]:
    if config.get("schema_version") != OKX_L2_NORMALIZATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported normalization schema: {config.get('schema_version')}")
    if file_sha256(metrics_path) != str(config["source_metrics_sha256"]):
        raise ValueError("Source snapshot metrics SHA-256 does not match the frozen input")

    expected_groups = {
        (str(date), str(instrument))
        for date in config["dates_utc"]
        for instrument in config["instruments"]
    }
    rows: list[SnapshotMetric] = []
    seen: set[tuple[str, str, int]] = set()
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "date_utc",
            "instrument",
            "timestamp_ms",
            "valid",
            "reason",
            "bid_levels",
            "ask_levels",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Source snapshot metrics are missing required columns")
        metric_fields = [field for field in reader.fieldnames if field not in SOURCE_ID_FIELDS]
        for row_number, row in enumerate(reader, start=2):
            if len(rows) >= int(config["maximum_source_rows"]):
                raise ValueError("Source snapshot metrics exceed the frozen row limit")
            date_utc = str(row["date_utc"])
            instrument = str(row["instrument"])
            group = (date_utc, instrument)
            if group not in expected_groups:
                raise ValueError(f"Unexpected source group at row {row_number}: {group}")
            try:
                timestamp_ms = int(str(row["timestamp_ms"]))
            except ValueError as exc:
                raise ValueError(f"Invalid source timestamp at row {row_number}") from exc
            day_start = _day_start_ms(date_utc)
            if not day_start <= timestamp_ms < day_start + DAY_MS:
                raise ValueError(f"Source timestamp outside UTC day at row {row_number}")
            key = (date_utc, instrument, timestamp_ms)
            if key in seen:
                raise ValueError(f"Duplicate source snapshot at row {row_number}")
            seen.add(key)
            values = {field: str(row[field]) for field in metric_fields}
            _validate_numeric_values(values, row_number)
            rows.append(SnapshotMetric(date_utc, instrument, timestamp_ms, values))

    observed_groups = {(row.date_utc, row.instrument) for row in rows}
    if observed_groups != expected_groups:
        raise ValueError("Source snapshot metrics do not contain every frozen group")
    return (
        sorted(rows, key=lambda row: (row.date_utc, row.instrument, row.timestamp_ms)),
        metric_fields,
    )


def _percentile(values: Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def normalize_snapshot_metrics(
    source_rows: Sequence[SnapshotMetric],
    metric_fields: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[SnapshotMetric]] = defaultdict(list)
    for row in source_rows:
        grouped[(row.date_utc, row.instrument)].append(row)

    expected_groups = [
        (str(date), str(instrument))
        for date in config["dates_utc"]
        for instrument in config["instruments"]
    ]
    interval_ms = int(config["grid_interval_ms"])
    expected_slots = int(config["expected_grid_slots_per_archive"])
    if interval_ms <= 0 or DAY_MS % interval_ms or DAY_MS // interval_ms != expected_slots:
        raise ValueError("Frozen grid interval and expected slot count are inconsistent")
    minimum_lag = int(config["minimum_source_lag_ms"])
    maximum_lag = int(config["maximum_source_lag_ms"])
    if minimum_lag < 0 or maximum_lag < minimum_lag:
        raise ValueError("Frozen source lag window is invalid")

    normalized: list[dict[str, object]] = []
    group_summaries: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, int]] = set()
    for date_utc, instrument in expected_groups:
        native_rows = sorted(
            grouped.get((date_utc, instrument), []), key=lambda row: row.timestamp_ms
        )
        pointer = 0
        available_count = 0
        lags: list[int] = []
        missing_grid_timestamps: list[int] = []
        day_start = _day_start_ms(date_utc)
        for slot in range(expected_slots):
            grid_timestamp = day_start + slot * interval_ms
            while pointer < len(native_rows) and native_rows[pointer].timestamp_ms < grid_timestamp:
                pointer += 1
            selected: SnapshotMetric | None = None
            if pointer < len(native_rows):
                candidate = native_rows[pointer]
                lag = candidate.timestamp_ms - grid_timestamp
                if minimum_lag <= lag <= maximum_lag:
                    selected = candidate
                    pointer += 1
            if selected is None:
                missing_grid_timestamps.append(grid_timestamp)
                normalized.append(
                    {
                        "date_utc": date_utc,
                        "instrument": instrument,
                        "grid_timestamp_ms": grid_timestamp,
                        "source_timestamp_ms": "",
                        "source_lag_ms": "",
                        "available": False,
                        **{
                            field: (
                                "False"
                                if field == "valid"
                                else "no_snapshot_within_lag_window" if field == "reason" else ""
                            )
                            for field in metric_fields
                        },
                    }
                )
                continue

            selected_key = (date_utc, instrument, selected.timestamp_ms)
            if selected_key in selected_keys:
                raise ValueError("A source snapshot was selected for more than one grid slot")
            selected_keys.add(selected_key)
            lag = selected.timestamp_ms - grid_timestamp
            lags.append(lag)
            available_count += 1
            normalized.append(
                {
                    "date_utc": date_utc,
                    "instrument": instrument,
                    "grid_timestamp_ms": grid_timestamp,
                    "source_timestamp_ms": selected.timestamp_ms,
                    "source_lag_ms": lag,
                    "available": True,
                    **selected.values,
                }
            )

        group_summaries.append(
            {
                "date_utc": date_utc,
                "instrument": instrument,
                "native_snapshots": len(native_rows),
                "grid_slots": expected_slots,
                "available_slots": available_count,
                "missing_slots": expected_slots - available_count,
                "coverage": available_count / expected_slots,
                "missing_grid_timestamps_ms": missing_grid_timestamps,
                "selected_native_snapshots": available_count,
                "ignored_native_snapshots": len(native_rows) - available_count,
                "source_lag_ms": {
                    "minimum": min(lags) if lags else None,
                    "median": _percentile(lags, 0.5),
                    "p95": _percentile(lags, 0.95),
                    "maximum": max(lags) if lags else None,
                },
            }
        )

    minimum_available = int(config["minimum_available_slots_per_archive"])
    expected_total = int(config["expected_total_grid_slots"])
    groups_pass = all(
        summary["grid_slots"] == expected_slots
        and summary["available_slots"] >= minimum_available
        and (
            summary["source_lag_ms"]["maximum"] is None
            or summary["source_lag_ms"]["maximum"] <= maximum_lag
        )
        for summary in group_summaries
    )
    total_available = sum(summary["available_slots"] for summary in group_summaries)
    go = (
        len(group_summaries) == int(config["expected_archive_count"])
        and len(normalized) == expected_total
        and groups_pass
    )
    result = {
        "schema_version": OKX_L2_NORMALIZATION_SCHEMA_VERSION,
        "provider": config["provider"],
        "source": {
            "artifact_id": config["source_artifact_id"],
            "result_sha256": config["source_result_sha256"],
            "metrics_sha256": config["source_metrics_sha256"],
            "native_snapshot_rows": len(source_rows),
        },
        "grid": {
            "selection_rule": config["selection_rule"],
            "missing_policy": config["missing_policy"],
            "interval_ms": interval_ms,
            "source_lag_window_ms": [minimum_lag, maximum_lag],
            "total_slots": len(normalized),
            "available_slots": total_available,
            "missing_slots": len(normalized) - total_available,
            "coverage": total_available / len(normalized) if normalized else 0.0,
            "selected_source_snapshots": len(selected_keys),
            "ignored_source_snapshots": len(source_rows) - len(selected_keys),
        },
        "groups": group_summaries,
        "decision": "GO_TECHNICAL_NORMALIZATION" if go else "NO_GO_TECHNICAL_NORMALIZATION",
        "updates_used_for_features": False,
        "interpolation_used": False,
        "forward_fill_used": False,
        "network_used": False,
        "model_trained": False,
        "orders_placed": False,
        "production_touched": False,
        "limitations": [
            "Normalization feasibility on three instruments and three UTC days only",
            "The 95-of-96 threshold was defined after Lot 5E exposed one missing native slot",
            "No predictive or economic validation",
        ],
    }
    return result, normalized


def _normalized_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_normalization_artifact(
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
    *,
    config_sha256: str,
    normalization_code_sha256: str,
    output_root: Path,
) -> Path:
    csv_payload = _normalized_csv(rows)
    identity = {
        **result,
        "config_sha256": config_sha256,
        "normalization_code_sha256": normalization_code_sha256,
        "normalized_metrics_sha256": hashlib.sha256(csv_payload).hexdigest(),
    }
    artifact_id = (
        f"{OKX_L2_NORMALIZATION_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Normalization artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "normalization_result.json").write_bytes(
            _json_bytes(identity, pretty=True)
        )
        (temporary_path / "normalized_snapshot_metrics.csv").write_bytes(csv_payload)
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "network_used": False,
            "credentials_used": False,
            "orders_placed": False,
            "updates_used_for_features": False,
            "interpolation_used": False,
            "model_trained": False,
            "production_touched": False,
            "result_file": "normalization_result.json",
            "result_sha256": file_sha256(temporary_path / "normalization_result.json"),
            "metrics_file": "normalized_snapshot_metrics.csv",
            "metrics_sha256": file_sha256(temporary_path / "normalized_snapshot_metrics.csv"),
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
