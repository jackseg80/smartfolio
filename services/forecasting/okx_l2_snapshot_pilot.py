"""Independent-snapshot analysis for one pre-registered OKX L2 archive."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.forecasting.okx_l2_pilot import DAY_MS, OrderBook, file_sha256

OKX_L2_SNAPSHOT_PILOT_SCHEMA_VERSION = "crypto-forecast-okx-l2-snapshot-pilot-v1"


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


def _utc_day_bounds(date_text: str) -> tuple[int, int]:
    day = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(day.timestamp() * 1000)
    return start_ms, start_ms + DAY_MS


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(snapshots: Sequence[Mapping[str, object]], field: str) -> dict[str, object]:
    values = [float(row[field]) for row in snapshots if row.get("valid") and field in row]
    return {
        "observations": len(values),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _metrics_are_finite(metrics: Mapping[str, object]) -> bool:
    for key, value in metrics.items():
        if key in {"valid", "reason"}:
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def analyze_snapshot_archive(
    archive_path: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    if config.get("schema_version") != OKX_L2_SNAPSHOT_PILOT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported snapshot pilot schema: {config.get('schema_version')}")
    archive_size = archive_path.stat().st_size
    if archive_size > int(config["download_limit_bytes"]):
        raise ValueError("Compressed archive exceeds the pre-registered byte limit")
    archive_hash = file_sha256(archive_path)
    if archive_hash != str(config["expected_archive_sha256"]):
        raise ValueError("Archive SHA-256 does not match the pinned snapshot input")

    day_start_ms, day_end_ms = _utc_day_bounds(str(config["date_utc"]))
    bands = [int(value) for value in config["depth_bands_bps"]]
    maximum_levels = int(config["maximum_levels_per_side"])
    action_counts: Counter[str] = Counter()
    record_keys: set[str] = set()
    record_count = 0
    previous_record_ts: int | None = None
    previous_snapshot_ts: int | None = None
    snapshot_gaps_ms: list[int] = []
    snapshots: list[dict[str, object]] = []

    with tarfile.open(archive_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1:
            raise ValueError("The snapshot archive must contain exactly one regular member")
        member = members[0]
        if member.size > int(config["uncompressed_limit_bytes"]):
            raise ValueError("Uncompressed member exceeds the pre-registered byte limit")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Unable to open the snapshot archive member")

        for raw_line in stream:
            record_count += 1
            if record_count > int(config["max_records"]):
                raise ValueError("Record count exceeds the pre-registered limit")
            if len(raw_line) > int(config["max_line_bytes"]):
                raise ValueError("A JSON record exceeds the pre-registered line-size limit")
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSON record at line {record_count}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record {record_count} is not an object")
            record_keys.update(str(key) for key in record)
            if record.get("instId") != config["instrument"]:
                raise ValueError(f"Unexpected instrument at line {record_count}")
            try:
                timestamp_ms = int(str(record["ts"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid timestamp at line {record_count}") from exc
            if not day_start_ms <= timestamp_ms < day_end_ms:
                raise ValueError(f"Timestamp outside the frozen UTC day at line {record_count}")
            if previous_record_ts is not None and timestamp_ms < previous_record_ts:
                raise ValueError(f"Timestamps moved backwards at line {record_count}")
            previous_record_ts = timestamp_ms

            action = str(record.get("action", ""))
            action_counts[action] += 1
            if action == "update":
                continue
            if action != "snapshot":
                raise ValueError(
                    f"Unsupported order-book action at line {record_count}: {action!r}"
                )
            if previous_snapshot_ts is not None:
                if timestamp_ms <= previous_snapshot_ts:
                    raise ValueError("Snapshot timestamps must be strictly increasing")
                snapshot_gaps_ms.append(timestamp_ms - previous_snapshot_ts)
            previous_snapshot_ts = timestamp_ms

            raw_bids = record.get("bids")
            raw_asks = record.get("asks")
            if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
                raise ValueError(f"Snapshot sides must be lists at line {record_count}")
            book = OrderBook()
            book.apply("snapshot", raw_bids, raw_asks)
            if not 1 <= len(book.bids) <= maximum_levels:
                raise ValueError(f"Invalid bid depth at line {record_count}")
            if not 1 <= len(book.asks) <= maximum_levels:
                raise ValueError(f"Invalid ask depth at line {record_count}")
            if len(book.bids) != len(raw_bids) or len(book.asks) != len(raw_asks):
                raise ValueError(
                    f"Snapshot contains zero-size or duplicate levels at line {record_count}"
                )
            metrics = book.metrics(bands)
            if not _metrics_are_finite(metrics):
                raise ValueError(f"Snapshot contains non-finite metrics at line {record_count}")
            snapshots.append({"timestamp_ms": timestamp_ms, **metrics})

    expected_count = int(config["expected_snapshot_count"])
    target_gap = int(config["snapshot_interval_ms"])
    tolerance = int(config["snapshot_interval_tolerance_ms"])
    count_ok = len(snapshots) == expected_count
    first_ok = bool(snapshots) and snapshots[0]["timestamp_ms"] == day_start_ms
    last_ok = bool(snapshots) and day_end_ms - int(snapshots[-1]["timestamp_ms"]) < target_gap
    cadence_ok = bool(snapshot_gaps_ms) and all(
        abs(gap - target_gap) <= tolerance for gap in snapshot_gaps_ms
    )
    valid_count = sum(bool(row.get("valid")) for row in snapshots)
    valid_ratio = valid_count / len(snapshots) if snapshots else 0.0
    go_technical = count_ok and first_ok and last_ok and cadence_ok and valid_ratio == 1.0
    decision_reasons: list[str] = []
    if not count_ok:
        decision_reasons.append("Snapshot count does not match the pre-registered count")
    if not first_ok or not last_ok:
        decision_reasons.append("Snapshots do not cover the frozen UTC day boundaries")
    if not cadence_ok:
        decision_reasons.append("Snapshot cadence exceeds the pre-registered tolerance")
    if valid_ratio != 1.0:
        decision_reasons.append("At least one snapshot is empty, crossed, or otherwise invalid")

    metric_summaries = {"spread_bps": _summary(snapshots, "spread_bps")}
    for band in bands:
        for field in (
            f"bid_depth_{band}bps_usdt",
            f"ask_depth_{band}bps_usdt",
            f"imbalance_{band}bps",
        ):
            metric_summaries[field] = _summary(snapshots, field)

    result: dict[str, Any] = {
        "schema_version": OKX_L2_SNAPSHOT_PILOT_SCHEMA_VERSION,
        "provider": config["provider"],
        "instrument": config["instrument"],
        "date_utc": config["date_utc"],
        "archive": {
            "filename": archive_path.name,
            "compressed_bytes": archive_size,
            "sha256": archive_hash,
            "member_name": member.name,
            "member_bytes": member.size,
        },
        "records": {
            "total": record_count,
            "actions": dict(sorted(action_counts.items())),
            "fields": sorted(record_keys),
            "updates_ignored_for_features": action_counts["update"],
        },
        "snapshots": {
            "total": len(snapshots),
            "valid_uncrossed": valid_count,
            "valid_ratio": valid_ratio,
            "first_timestamp_ms": snapshots[0]["timestamp_ms"] if snapshots else None,
            "last_timestamp_ms": snapshots[-1]["timestamp_ms"] if snapshots else None,
            "minimum_gap_ms": min(snapshot_gaps_ms) if snapshot_gaps_ms else None,
            "median_gap_ms": _percentile(snapshot_gaps_ms, 0.5),
            "maximum_gap_ms": max(snapshot_gaps_ms) if snapshot_gaps_ms else None,
            "calculation_semantics": "each snapshot measured independently; updates ignored",
        },
        "metric_summaries": metric_summaries,
        "decision": "GO_TECHNICAL" if go_technical else "NO_GO_TECHNICAL",
        "decision_reasons": decision_reasons,
        "limitations": [
            "One instrument and one UTC day only",
            "No update event contributes to any feature",
            "No predictive model or economic evaluation",
            "No interpolation or state propagation between snapshots",
        ],
    }
    return result, snapshots


def _snapshots_csv(snapshots: Sequence[Mapping[str, object]]) -> bytes:
    if not snapshots:
        return b""
    fieldnames = list(snapshots[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(snapshots)
    return buffer.getvalue().encode("utf-8")


def write_snapshot_artifact(
    result: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, object]],
    *,
    config_sha256: str,
    analysis_code_sha256: str,
    output_root: Path,
) -> Path:
    identity = {
        **result,
        "config_sha256": config_sha256,
        "analysis_code_sha256": analysis_code_sha256,
    }
    artifact_id = (
        f"{OKX_L2_SNAPSHOT_PILOT_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Snapshot pilot artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "snapshot_result.json").write_bytes(_json_bytes(identity, pretty=True))
        (temporary_path / "snapshot_metrics.csv").write_bytes(_snapshots_csv(snapshots))
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "credentials_used": False,
            "orders_placed": False,
            "raw_member_extracted_to_disk": False,
            "updates_used_for_features": False,
            "model_trained": False,
            "production_touched": False,
            "result_file": "snapshot_result.json",
            "result_sha256": file_sha256(temporary_path / "snapshot_result.json"),
            "metrics_file": "snapshot_metrics.csv",
            "metrics_sha256": file_sha256(temporary_path / "snapshot_metrics.csv"),
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
