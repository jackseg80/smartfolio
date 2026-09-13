"""Atomic local storage and recovery for prospective OKX L2 snapshots."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.forecasting.okx_l2_collection_feasibility import (
    BookProbe,
    PublicBookClient,
    validate_book_probe,
)
from services.forecasting.okx_l2_pilot import DAY_MS, file_sha256

OKX_L2_PROSPECTIVE_COLLECTOR_SCHEMA_VERSION = "crypto-forecast-okx-l2-prospective-collector-v1"


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


def _date_utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).date().isoformat()


def _slot_utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat()


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != OKX_L2_PROSPECTIVE_COLLECTOR_SCHEMA_VERSION:
        raise ValueError(f"Unsupported collector schema: {config.get('schema_version')}")
    interval = int(config["grid_interval_ms"])
    slots = int(config["expected_slots_per_day"])
    if interval <= 0 or slots <= 0 or interval * slots != DAY_MS:
        raise ValueError("Collector grid must cover one UTC day exactly")
    instruments = [str(value) for value in config["instruments"]]
    if len(set(instruments)) != len(instruments):
        raise ValueError("Collector instruments must be unique")
    if any(re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)+", item) is None for item in instruments):
        raise ValueError("Collector instruments must be safe canonical identifiers")
    for key in ("payload_suffix", "temporary_suffix"):
        suffix = str(config[key])
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError(f"Unsafe collector suffix: {key}")


async def capture_pilot_probes(
    client: PublicBookClient, config: Mapping[str, Any]
) -> list[BookProbe]:
    """Capture exactly one public book for every frozen collector instrument."""
    _validate_config(config)
    instruments = [str(value) for value in config["instruments"]]
    probes: list[BookProbe] = []
    for index, instrument in enumerate(instruments):
        probes.append(
            await client.get_book(
                str(config["endpoint"]), instrument, int(config["book_depth_per_side"])
            )
        )
        if index + 1 < len(instruments) and float(config["request_delay_seconds"]) > 0:
            await asyncio.sleep(float(config["request_delay_seconds"]))
    return probes


def _capture_filename(slot_timestamp_ms: int, instrument: str, config: Mapping[str, Any]) -> str:
    return f"{slot_timestamp_ms}__{instrument}{config['payload_suffix']}"


def _atomic_write(path: Path, payload: bytes, temporary_suffix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=temporary_suffix, dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_create(path: Path, payload: bytes, temporary_suffix: str) -> bool:
    """Create a file atomically without ever replacing an existing capture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=temporary_suffix, dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def _decode_capture(path: Path) -> tuple[dict[str, Any], BookProbe]:
    try:
        envelope = json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid capture file: {path.name}") from exc
    if not isinstance(envelope, dict):
        raise ValueError(f"Invalid capture envelope: {path.name}")
    response_body = str(envelope["response_body_utf8"]).encode("utf-8")
    if hashlib.sha256(response_body).hexdigest() != str(envelope["response_sha256"]):
        raise ValueError(f"Capture response hash mismatch: {path.name}")
    probe = BookProbe(
        instrument=str(envelope["instrument"]),
        requested_at_ms=int(envelope["requested_at_ms"]),
        completed_at_ms=int(envelope["completed_at_ms"]),
        response_body=response_body,
    )
    return envelope, probe


def load_stored_probe(path: Path) -> BookProbe:
    """Load and hash-check one durable collector capture."""
    _envelope, probe = _decode_capture(path)
    return probe


def record_probe(
    storage_root: Path,
    *,
    slot_timestamp_ms: int,
    probe: BookProbe,
    config: Mapping[str, Any],
    pilot_mode: bool,
) -> dict[str, Any]:
    """Validate and atomically persist one unique slot/instrument capture."""
    _validate_config(config)
    interval = int(config["grid_interval_ms"])
    if slot_timestamp_ms % interval:
        raise ValueError("Capture slot is not aligned to the frozen UTC grid")
    if probe.instrument not in {str(value) for value in config["instruments"]}:
        raise ValueError(f"Unexpected collector instrument: {probe.instrument}")
    lag_ms = probe.completed_at_ms - slot_timestamp_ms
    maximum_lag = int(
        config["pilot_maximum_capture_lag_ms" if pilot_mode else "scheduled_capture_tolerance_ms"]
    )
    if lag_ms < 0 or lag_ms > maximum_lag:
        raise ValueError(f"Capture is outside the slot tolerance: {probe.instrument}")
    validation = validate_book_probe(probe, config)
    response_sha256 = hashlib.sha256(probe.response_body).hexdigest()
    envelope = {
        "schema_version": OKX_L2_PROSPECTIVE_COLLECTOR_SCHEMA_VERSION,
        "provider": config["provider"],
        "instrument": probe.instrument,
        "slot_timestamp_ms": slot_timestamp_ms,
        "slot_utc": _slot_utc(slot_timestamp_ms),
        "requested_at_ms": probe.requested_at_ms,
        "completed_at_ms": probe.completed_at_ms,
        "capture_lag_ms": lag_ms,
        "provider_timestamp_ms": validation["provider_timestamp_ms"],
        "response_sha256": response_sha256,
        "response_body_utf8": probe.response_body.decode("utf-8"),
    }
    canonical = _json_bytes(envelope)
    compressed = gzip.compress(canonical, compresslevel=int(config["gzip_compress_level"]), mtime=0)
    date_utc = _date_utc(slot_timestamp_ms)
    day_directory = storage_root / date_utc
    destination = day_directory / _capture_filename(slot_timestamp_ms, probe.instrument, config)
    if destination.exists():
        existing, _existing_probe = _decode_capture(destination)
        if str(existing["response_sha256"]) == response_sha256:
            return {
                "status": "duplicate_identical",
                "path": str(destination.relative_to(storage_root)).replace("\\", "/"),
                "file_sha256": file_sha256(destination),
                "response_sha256": response_sha256,
                "capture_lag_ms": int(existing["capture_lag_ms"]),
            }
        raise FileExistsError(f"Conflicting capture already exists: {destination.name}")
    if not _atomic_create(destination, compressed, str(config["temporary_suffix"])):
        existing, _existing_probe = _decode_capture(destination)
        if str(existing["response_sha256"]) == response_sha256:
            return {
                "status": "duplicate_identical",
                "path": str(destination.relative_to(storage_root)).replace("\\", "/"),
                "file_sha256": file_sha256(destination),
                "response_sha256": response_sha256,
                "capture_lag_ms": int(existing["capture_lag_ms"]),
            }
        raise FileExistsError(f"Conflicting capture already exists: {destination.name}")
    return {
        "status": "written",
        "path": str(destination.relative_to(storage_root)).replace("\\", "/"),
        "file_sha256": file_sha256(destination),
        "response_sha256": response_sha256,
        "capture_lag_ms": lag_ms,
    }


def build_daily_manifest(
    storage_root: Path,
    *,
    date_utc: str,
    as_of_ms: int,
    config: Mapping[str, Any],
    final: bool = False,
) -> dict[str, Any]:
    """Reconstruct one daily manifest from durable capture files only."""
    _validate_config(config)
    day_start = _day_start_ms(date_utc)
    interval = int(config["grid_interval_ms"])
    instruments = [str(value) for value in config["instruments"]]
    day_directory = storage_root / date_utc
    captures: list[dict[str, Any]] = []
    observed: set[tuple[int, str]] = set()
    if day_directory.exists():
        for path in sorted(day_directory.glob(f"*{config['payload_suffix']}")):
            envelope, probe = _decode_capture(path)
            if envelope.get("schema_version") != OKX_L2_PROSPECTIVE_COLLECTOR_SCHEMA_VERSION:
                raise ValueError(f"Unexpected capture schema: {path.name}")
            slot_timestamp_ms = int(envelope["slot_timestamp_ms"])
            expected_name = _capture_filename(slot_timestamp_ms, probe.instrument, config)
            if (
                _date_utc(slot_timestamp_ms) != date_utc
                or probe.instrument not in instruments
                or path.name != expected_name
            ):
                raise ValueError(f"Capture identity mismatch: {path.name}")
            validation = validate_book_probe(probe, config)
            key = (slot_timestamp_ms, probe.instrument)
            if key in observed:
                raise ValueError(f"Duplicate capture key: {path.name}")
            observed.add(key)
            captures.append(
                {
                    "slot_timestamp_ms": slot_timestamp_ms,
                    "slot_utc": _slot_utc(slot_timestamp_ms),
                    "instrument": probe.instrument,
                    "path": str(path.relative_to(storage_root)).replace("\\", "/"),
                    "file_sha256": file_sha256(path),
                    "response_sha256": validation["response_sha256"],
                    "capture_lag_ms": int(envelope["capture_lag_ms"]),
                    "provider_timestamp_ms": validation["provider_timestamp_ms"],
                }
            )
    all_expected = [
        (day_start + slot * interval, instrument)
        for slot in range(int(config["expected_slots_per_day"]))
        for instrument in instruments
    ]
    tolerance = int(config["scheduled_capture_tolerance_ms"])
    due = all_expected if final else [key for key in all_expected if key[0] + tolerance <= as_of_ms]
    missing = [
        {
            "slot_timestamp_ms": slot,
            "slot_utc": _slot_utc(slot),
            "instrument": instrument,
        }
        for slot, instrument in due
        if (slot, instrument) not in observed
    ]
    future_or_not_due = [key for key in all_expected if key not in due]
    orphan_files = []
    if day_directory.exists():
        orphan_files = [
            str(path.relative_to(storage_root)).replace("\\", "/")
            for path in sorted(day_directory.glob(f"*{config['temporary_suffix']}"))
        ]
    return {
        "schema_version": OKX_L2_PROSPECTIVE_COLLECTOR_SCHEMA_VERSION,
        "date_utc": date_utc,
        "as_of_ms": as_of_ms,
        "final": final,
        "expected_total_captures": len(all_expected),
        "due_captures": len(due),
        "observed_captures": len(observed),
        "missing_due_captures": missing,
        "not_yet_due_captures": len(future_or_not_due),
        "orphan_temporary_files": orphan_files,
        "captures": sorted(
            captures, key=lambda item: (item["slot_timestamp_ms"], item["instrument"])
        ),
        "complete": final and not missing and len(observed) == len(all_expected),
        "missing_policy": config["missing_policy"],
        "orphan_policy": config["orphan_policy"],
    }


def write_daily_manifest(
    storage_root: Path, manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> Path:
    destination = storage_root / str(manifest["date_utc"]) / "daily_manifest.json"
    _atomic_write(destination, _json_bytes(manifest, pretty=True), str(config["temporary_suffix"]))
    return destination


def write_pilot_artifact(
    probes: Sequence[BookProbe],
    *,
    slot_timestamp_ms: int,
    config: Mapping[str, Any],
    config_sha256: str,
    collector_code_sha256: str,
    output_root: Path,
) -> Path:
    _validate_config(config)
    if sorted(probe.instrument for probe in probes) != sorted(
        str(value) for value in config["instruments"]
    ):
        raise ValueError("Pilot probes do not match the frozen instrument set")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".collector-pilot-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        storage_root = temporary_path / "days"
        records = [
            record_probe(
                storage_root,
                slot_timestamp_ms=slot_timestamp_ms,
                probe=probe,
                config=config,
                pilot_mode=True,
            )
            for probe in probes
        ]
        as_of_ms = max(probe.completed_at_ms for probe in probes)
        date_utc = _date_utc(slot_timestamp_ms)
        manifest = build_daily_manifest(
            storage_root,
            date_utc=date_utc,
            as_of_ms=as_of_ms,
            config=config,
        )
        manifest_path = write_daily_manifest(storage_root, manifest, config)
        identity = {
            "schema_version": OKX_L2_PROSPECTIVE_COLLECTOR_SCHEMA_VERSION,
            "source_feasibility_artifact_id": config["source_feasibility_artifact_id"],
            "source_feasibility_result_sha256": config["source_feasibility_result_sha256"],
            "config_sha256": config_sha256,
            "collector_code_sha256": collector_code_sha256,
            "slot_timestamp_ms": slot_timestamp_ms,
            "slot_utc": _slot_utc(slot_timestamp_ms),
            "capture_records": records,
            "daily_manifest_sha256": file_sha256(manifest_path),
            "daily_manifest": manifest,
            "decision": (
                "GO_TECHNICAL_COLLECTOR"
                if len(manifest["captures"]) == len(probes)
                and not manifest["orphan_temporary_files"]
                else "NO_GO_TECHNICAL_COLLECTOR"
            ),
            "long_running_collection_started": False,
            "credentials_used": False,
            "orders_placed": False,
            "model_trained": False,
            "production_touched": False,
        }
        artifact_id = (
            f"{OKX_L2_PROSPECTIVE_COLLECTOR_SCHEMA_VERSION}-"
            f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
        )
        (temporary_path / "pilot_result.json").write_bytes(_json_bytes(identity, pretty=True))
        (temporary_path / "manifest.json").write_bytes(
            _json_bytes(
                {
                    "artifact_id": artifact_id,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "result_file": "pilot_result.json",
                    "result_sha256": file_sha256(temporary_path / "pilot_result.json"),
                    "daily_manifest_file": str(manifest_path.relative_to(temporary_path)).replace(
                        "\\", "/"
                    ),
                    "daily_manifest_sha256": file_sha256(manifest_path),
                    "public_read_only_endpoint": f"GET {config['endpoint']}",
                    "long_running_collection_started": False,
                    "credentials_used": False,
                    "orders_placed": False,
                    "model_trained": False,
                    "production_touched": False,
                },
                pretty=True,
            )
        )
        destination = output_root / artifact_id
        if destination.exists():
            raise FileExistsError(f"Collector artifact already exists: {destination}")
        Path(temporary).replace(destination)
    return destination


def load_pilot_probes(artifact: Path) -> tuple[int, list[BookProbe]]:
    result = json.loads((artifact / "pilot_result.json").read_text(encoding="utf-8"))
    slot_timestamp_ms = int(result["slot_timestamp_ms"])
    probes: list[BookProbe] = []
    for capture in result["daily_manifest"]["captures"]:
        relative = Path(str(capture["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe stored capture path")
        storage_root = (artifact / "days").resolve()
        path = (storage_root / relative).resolve()
        try:
            path.relative_to(storage_root)
        except ValueError as exc:
            raise ValueError("Stored capture path escapes the artifact") from exc
        _envelope, probe = _decode_capture(path)
        probes.append(probe)
    return slot_timestamp_ms, probes
