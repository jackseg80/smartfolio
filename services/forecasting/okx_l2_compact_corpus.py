"""Build a deterministic multi-period corpus from pinned OKX L2 archives."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from services.forecasting.okx_l2_pilot import DAY_MS, file_sha256
from services.forecasting.okx_l2_progressive_extraction import extract_archive

OKX_L2_COMPACT_CORPUS_SCHEMA_VERSION = "crypto-forecast-okx-l2-compact-corpus-v1"


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


def _validate_config(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if config.get("schema_version") != OKX_L2_COMPACT_CORPUS_SCHEMA_VERSION:
        raise ValueError(f"Unsupported compact corpus schema: {config.get('schema_version')}")
    dates = [str(value) for value in config["dates_utc"]]
    instruments = [str(value) for value in config["instruments"]]
    if len(dates) != len(set(dates)) or len(instruments) != len(set(instruments)):
        raise ValueError("Frozen dates and instruments must be unique")
    day_starts = {
        date: int(
            datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        )
        for date in dates
    }
    if any(Path(value).name != value for value in instruments):
        raise ValueError("Instrument names must not contain a path")
    archives = config.get("archives")
    if not isinstance(archives, list) or len(archives) != int(config["expected_archive_count"]):
        raise ValueError("The frozen archive count is inconsistent")
    expected_groups = [(date, instrument) for date in dates for instrument in instruments]
    actual_groups = [(str(item.get("date_utc")), str(item.get("instrument"))) for item in archives]
    if actual_groups != expected_groups:
        raise ValueError("Archive order must match the frozen date and instrument grid")
    if sum(int(item["compressed_bytes"]) for item in archives) != int(
        config["expected_total_compressed_bytes"]
    ):
        raise ValueError("The frozen total archive size is inconsistent")
    if sum(int(item["expected_native_snapshots"]) for item in archives) != int(
        config["expected_total_native_snapshots"]
    ):
        raise ValueError("The frozen native snapshot total is inconsistent")
    if sum(int(item["expected_available_snapshots"]) for item in archives) != int(
        config["expected_total_selected_snapshots"]
    ):
        raise ValueError("The frozen selected snapshot total is inconsistent")
    expected_slots = int(config["expected_grid_slots_per_archive"])
    interval_ms = int(config["grid_interval_ms"])
    window_ms = int(config["maximum_absolute_offset_ms"])
    expected_missing = 0
    for item in archives:
        filename = str(item["filename"])
        if Path(filename).name != filename:
            raise ValueError("Archive filenames must not contain a path")
        available = int(item["expected_available_snapshots"])
        missing = [int(value) for value in item["expected_missing_grid_timestamps_ms"]]
        if available + len(missing) != expected_slots:
            raise ValueError("Expected available and missing slots are inconsistent")
        date_utc = str(item["date_utc"])
        valid_grid = {day_starts[date_utc] + slot * interval_ms for slot in range(expected_slots)}
        if len(missing) != len(set(missing)) or any(value not in valid_grid for value in missing):
            raise ValueError("Expected missing timestamps must be unique frozen grid slots")
        if int(item["expected_maximum_absolute_offset_ms"]) > window_ms:
            raise ValueError("Expected maximum offset exceeds the frozen alignment window")
        selected_by_side = sum(
            int(item[field])
            for field in (
                "expected_selected_before_grid",
                "expected_selected_exactly_on_grid",
                "expected_selected_after_grid",
            )
        )
        if selected_by_side != available:
            raise ValueError("Frozen alignment side counts are inconsistent")
        expected_missing += len(missing)
    if expected_missing != int(config["expected_total_missing_snapshots"]):
        raise ValueError("The frozen missing snapshot total is inconsistent")
    if interval_ms <= 2 * window_ms:
        raise ValueError("Alignment windows must not overlap")
    if interval_ms <= 0 or DAY_MS % interval_ms or DAY_MS // interval_ms != expected_slots:
        raise ValueError("The grid interval and frozen slot count are inconsistent")
    return archives


def _safe_archive_path(raw_root: Path, filename: str) -> Path:
    resolved_root = raw_root.resolve()
    resolved_path = (resolved_root / filename).resolve()
    if resolved_path.parent != resolved_root:
        raise ValueError("Archive path escapes the frozen raw directory")
    return resolved_path


def _data_filename(date_utc: str, instrument: str) -> str:
    return f"{date_utc}/{instrument}.jsonl.gz"


def _safe_data_path(data_root: Path, filename: str) -> Path:
    if "\\" in filename:
        raise ValueError("Compact payload paths must use safe POSIX separators")
    relative = PurePosixPath(filename)
    if relative.is_absolute() or len(relative.parts) != 2 or ".." in relative.parts:
        raise ValueError("Compact payload path must contain exactly a date and filename")
    if any(Path(part).name != part for part in relative.parts):
        raise ValueError("Compact payload path contains an unsafe component")
    return data_root.joinpath(*relative.parts)


def build_compact_corpus(
    raw_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Read each pinned archive sequentially and retain its aligned full snapshots."""

    archives = _validate_config(config)
    summaries: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    measurements: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for archive_spec in archives:
        archive_config = {
            **config,
            "date_utc": archive_spec["date_utc"],
        }
        archive_path = _safe_archive_path(raw_root, str(archive_spec["filename"]))
        summary, payload, performance = extract_archive(archive_path, archive_spec, archive_config)
        expected_missing = list(archive_spec["expected_missing_grid_timestamps_ms"])
        expected_available = int(archive_spec["expected_available_snapshots"])
        summary["matches_frozen_corpus"] = bool(summary["matches_frozen_alignment"]) and (
            summary["snapshots"]["selected"] == expected_available
            and summary["snapshots"]["missing_grid_timestamps_ms"] == expected_missing
        )
        filename = _data_filename(str(archive_spec["date_utc"]), str(archive_spec["instrument"]))
        summary["compact_file"]["filename"] = filename
        summaries.append(summary)
        payloads[filename] = payload
        measurements.append(performance)

    total_compact_bytes = sum(len(payload) for payload in payloads.values())
    total_native = sum(int(item["snapshots"]["native"]) for item in summaries)
    total_selected = sum(int(item["snapshots"]["selected"]) for item in summaries)
    total_missing = sum(int(item["snapshots"]["missing"]) for item in summaries)
    go = (
        len(summaries) == int(config["expected_archive_count"])
        and total_native == int(config["expected_total_native_snapshots"])
        and total_selected == int(config["expected_total_selected_snapshots"])
        and total_missing == int(config["expected_total_missing_snapshots"])
        and total_compact_bytes <= int(config["maximum_total_compact_bytes"])
        and all(bool(item["matches_frozen_corpus"]) for item in summaries)
    )
    result = {
        "schema_version": OKX_L2_COMPACT_CORPUS_SCHEMA_VERSION,
        "provider": config["provider"],
        "source": {
            "sample_artifact_id": config["source_sample_artifact_id"],
            "sample_result_sha256": config["source_sample_result_sha256"],
            "alignment_artifact_id": config["source_alignment_artifact_id"],
            "alignment_result_sha256": config["source_alignment_result_sha256"],
            "method_artifact_id": config["source_method_artifact_id"],
            "method_result_sha256": config["source_method_result_sha256"],
        },
        "dates_utc": list(config["dates_utc"]),
        "instruments": list(config["instruments"]),
        "processing": {
            "archive_order": [
                {"date_utc": item["date_utc"], "instrument": item["instrument"]}
                for item in summaries
            ],
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
            "native_snapshots": total_native,
            "selected_snapshots": total_selected,
            "missing_snapshots": total_missing,
            "updates_ignored": sum(int(item["records"]["updates_ignored"]) for item in summaries),
            "compact_compressed_bytes": total_compact_bytes,
        },
        "decision": "GO_TECHNICAL_COMPACT_CORPUS" if go else "NO_GO_TECHNICAL_COMPACT_CORPUS",
        "network_used": False,
        "credentials_used": False,
        "model_trained": False,
        "orders_placed": False,
        "production_touched": False,
        "limitations": [
            "Three isolated UTC days and three spot instruments only",
            "The corpus is not a continuous historical time series",
            "Storage and format validation only; no predictive evaluation",
            "Source archives remain retained until separate cleanup authorization",
        ],
    }
    performance = {
        "measurements_are_excluded_from_artifact_identity": True,
        "archives": measurements,
        "total_elapsed_seconds": time.perf_counter() - total_started,
    }
    return result, payloads, performance


def write_compact_corpus_artifact(
    result: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    performance: Mapping[str, Any],
    *,
    config_sha256: str,
    corpus_code_sha256: str,
    extraction_code_sha256: str,
    output_root: Path,
) -> Path:
    expected_payloads = {
        str(item["compact_file"]["filename"]): item["compact_file"] for item in result["archives"]
    }
    if set(payloads) != set(expected_payloads):
        raise ValueError("Compact payload set does not match the corpus result")
    for filename, payload in payloads.items():
        expected = expected_payloads[filename]
        if len(payload) != int(expected["compressed_bytes"]):
            raise ValueError(f"Compact payload size mismatch: {filename}")
        if hashlib.sha256(payload).hexdigest() != str(expected["sha256"]):
            raise ValueError(f"Compact payload SHA-256 mismatch: {filename}")

    identity = {
        **result,
        "config_sha256": config_sha256,
        "corpus_code_sha256": corpus_code_sha256,
        "extraction_code_sha256": extraction_code_sha256,
    }
    artifact_id = (
        f"{OKX_L2_COMPACT_CORPUS_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Compact corpus artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        data_root = temporary_path / "data"
        data_root.mkdir()
        for filename, payload in sorted(payloads.items()):
            data_path = _safe_data_path(data_root, filename)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_bytes(payload)
        result_path = temporary_path / "compact_corpus_result.json"
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
