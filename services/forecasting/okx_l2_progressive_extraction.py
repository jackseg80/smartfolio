"""Progressive extraction of compact, aligned OKX L2 snapshots."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import tarfile
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.forecasting.okx_l2_pilot import DAY_MS, OrderBook, file_sha256

OKX_L2_PROGRESSIVE_EXTRACTION_SCHEMA_VERSION = "crypto-forecast-okx-l2-progressive-extraction-v1"


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


def _json_line_bytes(value: object) -> bytes:
    return _json_bytes(value) + b"\n"


def _day_bounds(date_utc: str) -> tuple[int, int]:
    start = datetime.strptime(date_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    return start_ms, start_ms + DAY_MS


def _gzip_bytes(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as stream:
        stream.write(payload)
    return buffer.getvalue()


def _is_finite(metrics: Mapping[str, object]) -> bool:
    return all(
        not isinstance(value, (int, float)) or math.isfinite(float(value))
        for key, value in metrics.items()
        if key not in {"valid", "reason"}
    )


def _validate_config(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if config.get("schema_version") != OKX_L2_PROGRESSIVE_EXTRACTION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported progressive extraction schema: {config.get('schema_version')}"
        )
    instruments = [str(value) for value in config["instruments"]]
    if len(instruments) != len(set(instruments)):
        raise ValueError("The frozen instrument list contains duplicates")
    archives = config.get("archives")
    if not isinstance(archives, list) or len(archives) != int(config["expected_archive_count"]):
        raise ValueError("The frozen archive count is inconsistent")
    archive_instruments = [str(item.get("instrument")) for item in archives]
    if archive_instruments != instruments:
        raise ValueError("Archive order must match the frozen instrument order")
    if sum(int(item["compressed_bytes"]) for item in archives) != int(
        config["expected_total_compressed_bytes"]
    ):
        raise ValueError("The frozen total archive size is inconsistent")
    interval_ms = int(config["grid_interval_ms"])
    window_ms = int(config["maximum_absolute_offset_ms"])
    slots = int(config["expected_grid_slots_per_archive"])
    if interval_ms <= 2 * window_ms:
        raise ValueError("Alignment windows must not overlap")
    if interval_ms <= 0 or DAY_MS % interval_ms or DAY_MS // interval_ms != slots:
        raise ValueError("The grid interval and frozen slot count are inconsistent")
    if int(config["expected_total_selected_snapshots"]) != len(archives) * slots:
        raise ValueError("The frozen total selected snapshot count is inconsistent")
    return archives


def _safe_archive_path(raw_root: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError("Archive filenames must not contain a path")
    resolved_root = raw_root.resolve()
    resolved_path = (resolved_root / filename).resolve()
    if resolved_path.parent != resolved_root:
        raise ValueError("Archive path escapes the frozen raw directory")
    return resolved_path


def _validated_snapshot(record: Mapping[str, object], config: Mapping[str, Any]) -> bool:
    raw_bids = record.get("bids")
    raw_asks = record.get("asks")
    if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
        raise ValueError("Snapshot sides must be lists")
    book = OrderBook()
    book.apply("snapshot", raw_bids, raw_asks)
    maximum_levels = int(config["maximum_levels_per_side"])
    if not 1 <= len(book.bids) <= maximum_levels:
        raise ValueError("Snapshot bid depth violates the frozen limit")
    if not 1 <= len(book.asks) <= maximum_levels:
        raise ValueError("Snapshot ask depth violates the frozen limit")
    if len(book.bids) != len(raw_bids) or len(book.asks) != len(raw_asks):
        raise ValueError("Snapshot contains zero-size or duplicate levels")
    metrics = book.metrics([int(value) for value in config["depth_bands_bps"]])
    if not _is_finite(metrics):
        raise ValueError("Snapshot contains non-finite metrics")
    return bool(metrics.get("valid"))


def _candidate_slot(timestamp_ms: int, day_start_ms: int, interval_ms: int) -> int:
    return (timestamp_ms - day_start_ms + interval_ms // 2) // interval_ms


def extract_archive(
    archive_path: Path,
    archive_spec: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Read one archive and retain only the aligned, independent snapshots."""

    expected_size = int(archive_spec["compressed_bytes"])
    actual_size = archive_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError("Archive size does not match the frozen input")
    if actual_size > int(config["maximum_single_archive_bytes"]):
        raise ValueError("Archive exceeds the frozen single-file limit")
    started = time.perf_counter()
    archive_hash = file_sha256(archive_path)
    if archive_hash != str(archive_spec["sha256"]):
        raise ValueError("Archive SHA-256 does not match the frozen input")

    instrument = str(archive_spec["instrument"])
    day_start_ms, day_end_ms = _day_bounds(str(config["date_utc"]))
    interval_ms = int(config["grid_interval_ms"])
    window_ms = int(config["maximum_absolute_offset_ms"])
    expected_slots = int(config["expected_grid_slots_per_archive"])
    selected: dict[int, tuple[tuple[int, int, int], dict[str, object]]] = {}
    action_counts: Counter[str] = Counter()
    record_count = 0
    snapshot_count = 0
    valid_snapshot_count = 0
    invalid_snapshot_count = 0
    previous_record_ts: int | None = None
    previous_snapshot_ts: int | None = None

    with tarfile.open(archive_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1:
            raise ValueError("The archive must contain exactly one regular member")
        member = members[0]
        if member.size > int(config["maximum_uncompressed_member_bytes"]):
            raise ValueError("Uncompressed member exceeds the frozen byte limit")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Unable to open the archive member")
        for raw_line in stream:
            record_count += 1
            if record_count > int(config["max_records_per_archive"]):
                raise ValueError("Record count exceeds the frozen limit")
            if len(raw_line) > int(config["max_line_bytes"]):
                raise ValueError("A JSON record exceeds the frozen line-size limit")
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSON record at line {record_count}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record {record_count} is not an object")
            if record.get("instId") != instrument:
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
                raise ValueError(f"Unsupported action at line {record_count}: {action!r}")
            snapshot_count += 1
            if previous_snapshot_ts is not None and timestamp_ms <= previous_snapshot_ts:
                raise ValueError("Snapshot timestamps must be strictly increasing")
            previous_snapshot_ts = timestamp_ms
            if not _validated_snapshot(record, config):
                invalid_snapshot_count += 1
                continue
            valid_snapshot_count += 1

            slot = _candidate_slot(timestamp_ms, day_start_ms, interval_ms)
            if not 0 <= slot < expected_slots:
                continue
            grid_timestamp_ms = day_start_ms + slot * interval_ms
            signed_offset_ms = timestamp_ms - grid_timestamp_ms
            if abs(signed_offset_ms) > window_ms:
                continue
            key = (
                abs(signed_offset_ms),
                0 if signed_offset_ms <= 0 else 1,
                timestamp_ms,
            )
            candidate = {
                "schema_version": OKX_L2_PROGRESSIVE_EXTRACTION_SCHEMA_VERSION,
                "provider": config["provider"],
                "instrument": instrument,
                "date_utc": config["date_utc"],
                "grid_timestamp_ms": grid_timestamp_ms,
                "source_timestamp_ms": timestamp_ms,
                "signed_offset_ms": signed_offset_ms,
                "absolute_offset_ms": abs(signed_offset_ms),
                "availability_timestamp_ms": max(grid_timestamp_ms, timestamp_ms),
                "bids": record["bids"],
                "asks": record["asks"],
            }
            current = selected.get(slot)
            if current is None or key < current[0]:
                selected[slot] = (key, candidate)

    rows = [selected[slot][1] for slot in sorted(selected)]
    missing_grid_timestamps_ms = [
        day_start_ms + slot * interval_ms for slot in range(expected_slots) if slot not in selected
    ]
    signed_offsets = [int(row["signed_offset_ms"]) for row in rows]
    before_count = sum(value < 0 for value in signed_offsets)
    exact_count = sum(value == 0 for value in signed_offsets)
    after_count = sum(value > 0 for value in signed_offsets)
    maximum_offset = max((abs(value) for value in signed_offsets), default=None)
    uncompressed_payload = b"".join(_json_line_bytes(row) for row in rows)
    compact_payload = _gzip_bytes(uncompressed_payload)
    elapsed_seconds = time.perf_counter() - started

    expected_match = (
        snapshot_count == int(archive_spec["expected_native_snapshots"])
        and invalid_snapshot_count == 0
        and len(rows) >= int(config["minimum_available_slots_per_archive"])
        and before_count == int(archive_spec["expected_selected_before_grid"])
        and exact_count == int(archive_spec["expected_selected_exactly_on_grid"])
        and after_count == int(archive_spec["expected_selected_after_grid"])
        and maximum_offset == int(archive_spec["expected_maximum_absolute_offset_ms"])
    )
    summary = {
        "instrument": instrument,
        "date_utc": config["date_utc"],
        "archive": {
            "filename": archive_path.name,
            "compressed_bytes": actual_size,
            "sha256": archive_hash,
            "member_name": member.name,
            "member_bytes": member.size,
        },
        "records": {
            "total": record_count,
            "actions": dict(sorted(action_counts.items())),
            "updates_ignored": action_counts["update"],
        },
        "snapshots": {
            "native": snapshot_count,
            "valid_uncrossed": valid_snapshot_count,
            "invalid_or_crossed": invalid_snapshot_count,
            "selected": len(rows),
            "missing": expected_slots - len(rows),
            "missing_grid_timestamps_ms": missing_grid_timestamps_ms,
            "selected_before_grid": before_count,
            "selected_exactly_on_grid": exact_count,
            "selected_after_grid": after_count,
            "maximum_absolute_offset_ms": maximum_offset,
        },
        "compact_file": {
            "filename": f"{instrument}.jsonl.gz",
            "uncompressed_bytes": len(uncompressed_payload),
            "compressed_bytes": len(compact_payload),
            "sha256": hashlib.sha256(compact_payload).hexdigest(),
        },
        "matches_frozen_alignment": expected_match,
    }
    performance = {
        "instrument": instrument,
        "archive_bytes_read": actual_size,
        "elapsed_seconds_including_sha256_and_extraction": elapsed_seconds,
        "throughput_mib_per_second": actual_size / (1024 * 1024) / elapsed_seconds,
    }
    return summary, compact_payload, performance


def extract_progressive_dataset(
    raw_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Extract the frozen archive set sequentially and build its deterministic identity."""

    archives = _validate_config(config)
    summaries: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    measurements: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for archive_spec in archives:
        archive_path = _safe_archive_path(raw_root, str(archive_spec["filename"]))
        summary, payload, performance = extract_archive(archive_path, archive_spec, config)
        filename = str(summary["compact_file"]["filename"])
        summaries.append(summary)
        payloads[filename] = payload
        measurements.append(performance)

    total_compact_bytes = sum(len(payload) for payload in payloads.values())
    total_selected = sum(int(item["snapshots"]["selected"]) for item in summaries)
    go = (
        len(summaries) == int(config["expected_archive_count"])
        and total_selected == int(config["expected_total_selected_snapshots"])
        and total_compact_bytes <= int(config["maximum_total_compact_bytes"])
        and all(bool(item["matches_frozen_alignment"]) for item in summaries)
    )
    result = {
        "schema_version": OKX_L2_PROGRESSIVE_EXTRACTION_SCHEMA_VERSION,
        "provider": config["provider"],
        "source": {
            "sample_artifact_id": config["source_sample_artifact_id"],
            "sample_result_sha256": config["source_sample_result_sha256"],
            "alignment_artifact_id": config["source_alignment_artifact_id"],
            "alignment_result_sha256": config["source_alignment_result_sha256"],
        },
        "date_utc": config["date_utc"],
        "processing": {
            "archive_order": [item["instrument"] for item in summaries],
            "mode": config["processing_order"],
            "raw_members_extracted_to_disk": False,
            "raw_archives_modified": False,
            "updates_used": False,
            "interpolation_used": False,
            "forward_fill_used": False,
        },
        "archives": summaries,
        "totals": {
            "archives": len(summaries),
            "source_compressed_bytes": sum(
                int(item["archive"]["compressed_bytes"]) for item in summaries
            ),
            "native_snapshots": sum(int(item["snapshots"]["native"]) for item in summaries),
            "selected_snapshots": total_selected,
            "missing_snapshots": sum(int(item["snapshots"]["missing"]) for item in summaries),
            "updates_ignored": sum(int(item["records"]["updates_ignored"]) for item in summaries),
            "compact_compressed_bytes": total_compact_bytes,
        },
        "decision": (
            "GO_TECHNICAL_PROGRESSIVE_EXTRACTION"
            if go
            else "NO_GO_TECHNICAL_PROGRESSIVE_EXTRACTION"
        ),
        "network_used": False,
        "credentials_used": False,
        "model_trained": False,
        "orders_placed": False,
        "production_touched": False,
        "limitations": [
            "One historical UTC day and three spot instruments only",
            "Storage and extraction feasibility only; no predictive validation",
            "Source archives remain retained until separate cleanup authorization",
        ],
    }
    performance = {
        "measurements_are_excluded_from_artifact_identity": True,
        "archives": measurements,
        "total_elapsed_seconds": time.perf_counter() - total_started,
    }
    return result, payloads, performance


def write_progressive_artifact(
    result: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    performance: Mapping[str, Any],
    *,
    config_sha256: str,
    extraction_code_sha256: str,
    output_root: Path,
) -> Path:
    expected_payloads = {
        str(item["compact_file"]["filename"]): item["compact_file"] for item in result["archives"]
    }
    if set(payloads) != set(expected_payloads):
        raise ValueError("Compact payload set does not match the extraction result")
    for filename, payload in payloads.items():
        if Path(filename).name != filename:
            raise ValueError("Compact payload filenames must not contain a path")
        expected = expected_payloads[filename]
        if len(payload) != int(expected["compressed_bytes"]):
            raise ValueError(f"Compact payload size mismatch: {filename}")
        if hashlib.sha256(payload).hexdigest() != str(expected["sha256"]):
            raise ValueError(f"Compact payload SHA-256 mismatch: {filename}")
    identity = {
        **result,
        "config_sha256": config_sha256,
        "extraction_code_sha256": extraction_code_sha256,
    }
    artifact_id = (
        f"{OKX_L2_PROGRESSIVE_EXTRACTION_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Progressive extraction artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        data_path = temporary_path / "data"
        data_path.mkdir()
        for filename, payload in sorted(payloads.items()):
            (data_path / filename).write_bytes(payload)
        result_path = temporary_path / "progressive_extraction_result.json"
        result_path.write_bytes(_json_bytes(identity, pretty=True))
        performance_path = temporary_path / "performance.json"
        performance_path.write_bytes(_json_bytes(performance, pretty=True))
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_file": result_path.name,
            "result_sha256": file_sha256(result_path),
            "performance_file": performance_path.name,
            "performance_excluded_from_identity": True,
            "data_files": [
                {
                    "path": f"data/{filename}",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for filename, payload in sorted(payloads.items())
            ],
            "raw_archive_policy": "retained_without_modification",
            "network_used": False,
            "credentials_used": False,
            "orders_placed": False,
            "model_trained": False,
            "production_touched": False,
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
