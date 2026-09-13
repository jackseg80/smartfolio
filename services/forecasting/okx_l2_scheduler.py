"""One-shot, UTC-aligned scheduler for the prospective OKX L2 collector."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from services.forecasting.okx_l2_collection_feasibility import BookProbe, PublicBookClient
from services.forecasting.okx_l2_pilot import file_sha256
from services.forecasting.okx_l2_prospective_collector import (
    OKX_L2_PROSPECTIVE_COLLECTOR_SCHEMA_VERSION,
    build_daily_manifest,
    load_stored_probe,
    record_probe,
    write_daily_manifest,
)

OKX_L2_ONE_SHOT_SCHEDULER_SCHEMA_VERSION = "crypto-forecast-okx-l2-one-shot-scheduler-v1"


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


def validate_scheduler_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != OKX_L2_ONE_SHOT_SCHEDULER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported scheduler schema: {config.get('schema_version')}")
    lock_filename = str(config["lock_filename"])
    if Path(lock_filename).name != lock_filename or lock_filename in {"", ".", ".."}:
        raise ValueError("Scheduler lock filename must not contain a path")


def _collector_config(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scheduler_config(config)
    return {**config, "schema_version": OKX_L2_PROSPECTIVE_COLLECTOR_SCHEMA_VERSION}


def next_slot_ms(now_ms: int, interval_ms: int) -> int:
    if now_ms < 0 or interval_ms <= 0:
        raise ValueError("Scheduler timestamps and intervals must be positive")
    return (now_ms // interval_ms + 1) * interval_ms


def slot_state(now_ms: int, slot_timestamp_ms: int, tolerance_ms: int) -> str:
    if now_ms < slot_timestamp_ms:
        return "WAITING"
    if now_ms <= slot_timestamp_ms + tolerance_ms:
        return "DUE"
    return "MISSED"


class SingleInstanceLock:
    """Exclusive lock that only its owning context may remove."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.token = uuid.uuid4().hex
        self._owned = False

    def __enter__(self) -> "SingleInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"Collector lock already exists: {self.path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(self.token)
            handle.flush()
            os.fsync(handle.fileno())
        self._owned = True
        return self

    def __exit__(self, *_args: object) -> None:
        if not self._owned:
            return
        try:
            current = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._owned = False
            return
        if current == self.token:
            self.path.unlink()
        self._owned = False


async def execute_due_slot(
    client: PublicBookClient,
    *,
    slot_timestamp_ms: int,
    execution_started_ms: int,
    storage_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    collector_config = _collector_config(config)
    tolerance_ms = int(config["scheduled_capture_tolerance_ms"])
    instruments = [str(value) for value in config["instruments"]]
    state = slot_state(execution_started_ms, slot_timestamp_ms, tolerance_ms)
    if state == "WAITING":
        raise ValueError("A due-slot execution cannot start before its target")
    if state == "MISSED":
        return {
            "status": "SKIPPED_LATE",
            "slot_timestamp_ms": slot_timestamp_ms,
            "execution_started_ms": execution_started_ms,
            "start_lag_ms": execution_started_ms - slot_timestamp_ms,
            "captures": [],
            "errors": [],
            "missing_instruments": instruments,
            "requests_attempted": 0,
            "backfill_attempted": False,
        }

    captures: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for instrument in instruments:
        try:
            probe = await client.get_book(
                str(config["endpoint"]), instrument, int(config["book_depth_per_side"])
            )
            record = record_probe(
                storage_root,
                slot_timestamp_ms=slot_timestamp_ms,
                probe=probe,
                config=collector_config,
                pilot_mode=False,
            )
            captures.append({"instrument": instrument, **record})
        except Exception as exc:  # noqa: BLE001 - each instrument must remain isolated
            errors.append(
                {
                    "instrument": instrument,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    captured_instruments = {str(item["instrument"]) for item in captures}
    missing = [item for item in instruments if item not in captured_instruments]
    return {
        "status": "COMPLETE" if not missing else "PARTIAL_FAILURE",
        "slot_timestamp_ms": slot_timestamp_ms,
        "execution_started_ms": execution_started_ms,
        "start_lag_ms": execution_started_ms - slot_timestamp_ms,
        "captures": captures,
        "errors": errors,
        "missing_instruments": missing,
        "requests_attempted": len(captures) + len(errors),
        "backfill_attempted": False,
    }


async def write_scheduler_artifact(
    client: PublicBookClient,
    *,
    slot_timestamp_ms: int,
    execution_started_ms: int,
    config: Mapping[str, Any],
    config_sha256: str,
    scheduler_code_sha256: str,
    collector_code_sha256: str,
    output_root: Path,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".one-shot-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        storage_root = temporary_path / "days"
        run = await execute_due_slot(
            client,
            slot_timestamp_ms=slot_timestamp_ms,
            execution_started_ms=execution_started_ms,
            storage_root=storage_root,
            config=config,
        )
        as_of_ms = max(
            [execution_started_ms]
            + [
                load_stored_probe(storage_root / str(item["path"])).completed_at_ms
                for item in run["captures"]
            ]
        )
        date_utc = datetime.fromtimestamp(slot_timestamp_ms / 1000, timezone.utc).date().isoformat()
        manifest = build_daily_manifest(
            storage_root,
            date_utc=date_utc,
            as_of_ms=as_of_ms,
            config=_collector_config(config),
        )
        manifest_path = write_daily_manifest(storage_root, manifest, _collector_config(config))
        identity = {
            "schema_version": OKX_L2_ONE_SHOT_SCHEDULER_SCHEMA_VERSION,
            "source_collector_artifact_id": config["source_collector_artifact_id"],
            "source_collector_result_sha256": config["source_collector_result_sha256"],
            "config_sha256": config_sha256,
            "scheduler_code_sha256": scheduler_code_sha256,
            "collector_code_sha256": collector_code_sha256,
            "run": run,
            "daily_manifest": manifest,
            "daily_manifest_sha256": file_sha256(manifest_path),
            "decision": (
                "GO_OPERATIONAL_ONE_SHOT" if run["status"] == "COMPLETE" else "NO_GO_ONE_SHOT"
            ),
            "long_running_collection_started": False,
            "model_trained": False,
            "production_touched": False,
            "orders_placed": False,
            "credentials_used": False,
        }
        artifact_id = (
            f"{OKX_L2_ONE_SHOT_SCHEDULER_SCHEMA_VERSION}-"
            f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
        )
        (temporary_path / "run_result.json").write_bytes(_json_bytes(identity, pretty=True))
        (temporary_path / "manifest.json").write_bytes(
            _json_bytes(
                {
                    "artifact_id": artifact_id,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "result_file": "run_result.json",
                    "result_sha256": file_sha256(temporary_path / "run_result.json"),
                    "daily_manifest_sha256": file_sha256(manifest_path),
                    "long_running_collection_started": False,
                    "model_trained": False,
                    "production_touched": False,
                    "orders_placed": False,
                    "credentials_used": False,
                },
                pretty=True,
            )
        )
        destination = output_root / artifact_id
        if destination.exists():
            raise FileExistsError(f"Scheduler artifact already exists: {destination}")
        Path(temporary).replace(destination)
    return destination


class ReplayBookClient:
    def __init__(self, probes: Sequence[BookProbe]) -> None:
        self._probes = {probe.instrument: probe for probe in probes}

    async def get_book(self, _endpoint: str, instrument: str, _depth: int) -> BookProbe:
        return self._probes[instrument]


def load_scheduler_replay(artifact: Path) -> tuple[int, int, list[BookProbe]]:
    result = json.loads((artifact / "run_result.json").read_text(encoding="utf-8"))
    run = result["run"]
    storage_root = (artifact / "days").resolve()
    probes: list[BookProbe] = []
    for capture in run["captures"]:
        relative = Path(str(capture["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe scheduler capture path")
        path = (storage_root / relative).resolve()
        try:
            path.relative_to(storage_root)
        except ValueError as exc:
            raise ValueError("Scheduler capture path escapes the artifact") from exc
        probes.append(load_stored_probe(path))
    return int(run["slot_timestamp_ms"]), int(run["execution_started_ms"]), probes
