"""Symmetric, causally timestamped alignment of pinned OKX L2 snapshots."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tempfile
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.forecasting.okx_l2_normalization import (
    OKX_L2_NORMALIZATION_SCHEMA_VERSION,
    SnapshotMetric,
    load_snapshot_metrics,
)
from services.forecasting.okx_l2_pilot import DAY_MS, file_sha256

OKX_L2_SYMMETRIC_ALIGNMENT_SCHEMA_VERSION = "crypto-forecast-okx-l2-symmetric-alignment-v1"


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


def load_alignment_source(
    metrics_path: Path, config: Mapping[str, Any]
) -> tuple[list[SnapshotMetric], list[str]]:
    if config.get("schema_version") != OKX_L2_SYMMETRIC_ALIGNMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported alignment schema: {config.get('schema_version')}")
    loader_config = {**config, "schema_version": OKX_L2_NORMALIZATION_SCHEMA_VERSION}
    return load_snapshot_metrics(metrics_path, loader_config)


def align_snapshot_metrics(
    source_rows: Sequence[SnapshotMetric],
    metric_fields: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    if config.get("schema_version") != OKX_L2_SYMMETRIC_ALIGNMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported alignment schema: {config.get('schema_version')}")
    grouped: dict[tuple[str, str], list[SnapshotMetric]] = defaultdict(list)
    for row in source_rows:
        grouped[(row.date_utc, row.instrument)].append(row)

    expected_groups = [
        (str(date), str(instrument))
        for date in config["dates_utc"]
        for instrument in config["instruments"]
    ]
    interval_ms = int(config["grid_interval_ms"])
    window_ms = int(config["maximum_absolute_offset_ms"])
    expected_slots = int(config["expected_grid_slots_per_archive"])
    if interval_ms <= 2 * window_ms:
        raise ValueError("Alignment windows must not overlap")
    if interval_ms <= 0 or DAY_MS % interval_ms or DAY_MS // interval_ms != expected_slots:
        raise ValueError("Frozen grid interval and expected slot count are inconsistent")

    aligned: list[dict[str, object]] = []
    group_summaries: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, int]] = set()
    for date_utc, instrument in expected_groups:
        native_rows = sorted(
            grouped.get((date_utc, instrument), []), key=lambda row: row.timestamp_ms
        )
        timestamps = [row.timestamp_ms for row in native_rows]
        available_count = 0
        before_count = 0
        exact_count = 0
        after_count = 0
        signed_offsets: list[int] = []
        missing_grid_timestamps: list[int] = []
        day_start = _day_start_ms(date_utc)
        for slot in range(expected_slots):
            grid_timestamp = day_start + slot * interval_ms
            first_candidate = bisect_left(timestamps, grid_timestamp - window_ms)
            candidates: list[SnapshotMetric] = []
            position = first_candidate
            while position < len(native_rows):
                candidate = native_rows[position]
                if candidate.timestamp_ms > grid_timestamp + window_ms:
                    break
                candidates.append(candidate)
                position += 1
            selected = min(
                candidates,
                key=lambda row: (
                    abs(row.timestamp_ms - grid_timestamp),
                    0 if row.timestamp_ms <= grid_timestamp else 1,
                    row.timestamp_ms,
                ),
                default=None,
            )
            if selected is None:
                missing_grid_timestamps.append(grid_timestamp)
                aligned.append(
                    {
                        "date_utc": date_utc,
                        "instrument": instrument,
                        "grid_timestamp_ms": grid_timestamp,
                        "source_timestamp_ms": "",
                        "signed_offset_ms": "",
                        "absolute_offset_ms": "",
                        "availability_timestamp_ms": "",
                        "available": False,
                        **{
                            field: (
                                "False"
                                if field == "valid"
                                else (
                                    "no_snapshot_within_symmetric_window"
                                    if field == "reason"
                                    else ""
                                )
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
            signed_offset = selected.timestamp_ms - grid_timestamp
            availability_timestamp = max(grid_timestamp, selected.timestamp_ms)
            if (
                availability_timestamp < grid_timestamp
                or availability_timestamp < selected.timestamp_ms
            ):
                raise ValueError("Causal availability timestamp is inconsistent")
            available_count += 1
            signed_offsets.append(signed_offset)
            before_count += signed_offset < 0
            exact_count += signed_offset == 0
            after_count += signed_offset > 0
            aligned.append(
                {
                    "date_utc": date_utc,
                    "instrument": instrument,
                    "grid_timestamp_ms": grid_timestamp,
                    "source_timestamp_ms": selected.timestamp_ms,
                    "signed_offset_ms": signed_offset,
                    "absolute_offset_ms": abs(signed_offset),
                    "availability_timestamp_ms": availability_timestamp,
                    "available": True,
                    **selected.values,
                }
            )

        absolute_offsets = [abs(value) for value in signed_offsets]
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
                "selected_before_grid": before_count,
                "selected_exactly_on_grid": exact_count,
                "selected_after_grid": after_count,
                "selected_native_snapshots": available_count,
                "ignored_native_snapshots": len(native_rows) - available_count,
                "absolute_offset_ms": {
                    "minimum": min(absolute_offsets) if absolute_offsets else None,
                    "median": _percentile(absolute_offsets, 0.5),
                    "p95": _percentile(absolute_offsets, 0.95),
                    "maximum": max(absolute_offsets) if absolute_offsets else None,
                },
            }
        )

    minimum_available = int(config["minimum_available_slots_per_archive"])
    expected_total = int(config["expected_total_grid_slots"])
    groups_pass = all(
        summary["grid_slots"] == expected_slots
        and summary["available_slots"] >= minimum_available
        and (
            summary["absolute_offset_ms"]["maximum"] is None
            or summary["absolute_offset_ms"]["maximum"] <= window_ms
        )
        for summary in group_summaries
    )
    total_available = sum(summary["available_slots"] for summary in group_summaries)
    go = (
        len(group_summaries) == int(config["expected_archive_count"])
        and len(aligned) == expected_total
        and groups_pass
    )
    result = {
        "schema_version": OKX_L2_SYMMETRIC_ALIGNMENT_SCHEMA_VERSION,
        "provider": config["provider"],
        "source": {
            "artifact_id": config["source_artifact_id"],
            "result_sha256": config["source_result_sha256"],
            "metrics_sha256": config["source_metrics_sha256"],
            "native_snapshot_rows": len(source_rows),
        },
        "grid": {
            "selection_rule": config["selection_rule"],
            "availability_rule": config["availability_rule"],
            "missing_policy": config["missing_policy"],
            "interval_ms": interval_ms,
            "maximum_absolute_offset_ms": window_ms,
            "total_slots": len(aligned),
            "available_slots": total_available,
            "missing_slots": len(aligned) - total_available,
            "coverage": total_available / len(aligned) if aligned else 0.0,
            "selected_source_snapshots": len(selected_keys),
            "ignored_source_snapshots": len(source_rows) - len(selected_keys),
        },
        "groups": group_summaries,
        "decision": "GO_TECHNICAL_ALIGNMENT" if go else "NO_GO_TECHNICAL_ALIGNMENT",
        "updates_used_for_features": False,
        "interpolation_used": False,
        "forward_fill_used": False,
        "network_used": False,
        "model_trained": False,
        "orders_placed": False,
        "production_touched": False,
        "limitations": [
            "Alignment feasibility on three instruments and three UTC days only",
            "The symmetric rule was defined after Lot 5F exposed boundary-side sensitivity",
            "No predictive or economic validation",
        ],
    }
    return result, aligned


def _aligned_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_alignment_artifact(
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
    *,
    config_sha256: str,
    alignment_code_sha256: str,
    output_root: Path,
) -> Path:
    csv_payload = _aligned_csv(rows)
    identity = {
        **result,
        "config_sha256": config_sha256,
        "alignment_code_sha256": alignment_code_sha256,
        "aligned_metrics_sha256": hashlib.sha256(csv_payload).hexdigest(),
    }
    artifact_id = (
        f"{OKX_L2_SYMMETRIC_ALIGNMENT_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Alignment artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "alignment_result.json").write_bytes(_json_bytes(identity, pretty=True))
        (temporary_path / "aligned_snapshot_metrics.csv").write_bytes(csv_payload)
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
            "result_file": "alignment_result.json",
            "result_sha256": file_sha256(temporary_path / "alignment_result.json"),
            "metrics_file": "aligned_snapshot_metrics.csv",
            "metrics_sha256": file_sha256(temporary_path / "aligned_snapshot_metrics.csv"),
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
