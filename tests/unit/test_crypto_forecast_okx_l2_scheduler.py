import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.forecasting.okx_l2_collection_feasibility import BookProbe
from services.forecasting.okx_l2_scheduler import (
    ReplayBookClient,
    SingleInstanceLock,
    execute_due_slot,
    load_scheduler_replay,
    next_slot_ms,
    slot_state,
    validate_scheduler_config,
    write_scheduler_artifact,
)

DAY_START_MS = int(datetime(2026, 9, 13, tzinfo=timezone.utc).timestamp() * 1000)


def _config() -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-okx-l2-one-shot-scheduler-v1",
        "provider": "fixture",
        "endpoint": "/books",
        "source_collector_artifact_id": "collector",
        "source_collector_result_sha256": "collector-hash",
        "instruments": ["BTC-USDT", "ETH-USDT"],
        "book_depth_per_side": 2,
        "grid_interval_ms": 21_600_000,
        "expected_slots_per_day": 4,
        "maximum_snapshot_age_ms": 60_000,
        "maximum_future_lead_ms": 5_000,
        "maximum_response_bytes_per_instrument": 131_072,
        "minimum_levels_per_side": 2,
        "maximum_levels_per_side": 2,
        "scheduled_capture_tolerance_ms": 120_000,
        "pilot_maximum_capture_lag_ms": 900_000,
        "gzip_compress_level": 9,
        "payload_suffix": ".json.gz",
        "temporary_suffix": ".pending",
        "missing_policy": "preserve_missing_without_fill_or_interpolation",
        "duplicate_policy": "same_slot_same_hash_is_idempotent_otherwise_conflict",
        "orphan_policy": "report_without_deleting",
        "late_policy": "skip_without_backfill",
        "partial_failure_policy": "retain_successes_and_report_missing",
        "lock_filename": ".collector.lock",
    }


def _probe(instrument: str) -> BookProbe:
    payload = {
        "code": "0",
        "msg": "",
        "data": [
            {
                "asks": [["101", "2", "0", "1"], ["102", "3", "0", "2"]],
                "bids": [["100", "1", "0", "2"], ["99", "4", "0", "1"]],
                "ts": str(DAY_START_MS + 1000),
                "seqId": 7,
            }
        ],
    }
    return BookProbe(
        instrument=instrument,
        requested_at_ms=DAY_START_MS + 500,
        completed_at_ms=DAY_START_MS + 1100,
        response_body=json.dumps(payload, separators=(",", ":")).encode(),
    )


class _FakeClient:
    def __init__(self, *, failing: str | None = None) -> None:
        self.failing = failing
        self.calls: list[str] = []

    async def get_book(self, _endpoint: str, instrument: str, _depth: int) -> BookProbe:
        self.calls.append(instrument)
        if instrument == self.failing:
            raise TimeoutError("frozen network failure")
        return _probe(instrument)


def test_next_slot_is_strictly_future_and_state_never_backfills():
    interval = 900_000
    assert next_slot_ms(900_000, interval) == 1_800_000
    assert next_slot_ms(1_799_999, interval) == 1_800_000
    assert slot_state(1_799_999, 1_800_000, 120_000) == "WAITING"
    assert slot_state(1_920_000, 1_800_000, 120_000) == "DUE"
    assert slot_state(1_920_001, 1_800_000, 120_000) == "MISSED"


def test_lock_rejects_second_instance_and_only_owner_can_remove(tmp_path: Path):
    path = tmp_path / ".collector.lock"
    with SingleInstanceLock(path):
        with pytest.raises(RuntimeError, match="already exists"):
            with SingleInstanceLock(path):
                pass
        assert path.exists()
    assert not path.exists()


def test_scheduler_rejects_lock_path_escape():
    config = {**_config(), "lock_filename": "../outside.lock"}

    with pytest.raises(ValueError, match="must not contain a path"):
        validate_scheduler_config(config)


@pytest.mark.asyncio
async def test_late_run_skips_without_request_or_backfill(tmp_path: Path):
    client = _FakeClient()
    result = await execute_due_slot(
        client,
        slot_timestamp_ms=DAY_START_MS,
        execution_started_ms=DAY_START_MS + 120_001,
        storage_root=tmp_path,
        config=_config(),
    )

    assert result["status"] == "SKIPPED_LATE"
    assert result["requests_attempted"] == 0
    assert result["backfill_attempted"] is False
    assert client.calls == []
    assert not list(tmp_path.rglob("*.json.gz"))


@pytest.mark.asyncio
async def test_partial_failure_keeps_success_and_reports_missing(tmp_path: Path):
    client = _FakeClient(failing="ETH-USDT")
    result = await execute_due_slot(
        client,
        slot_timestamp_ms=DAY_START_MS,
        execution_started_ms=DAY_START_MS + 100,
        storage_root=tmp_path,
        config=_config(),
    )

    assert result["status"] == "PARTIAL_FAILURE"
    assert [item["instrument"] for item in result["captures"]] == ["BTC-USDT"]
    assert result["missing_instruments"] == ["ETH-USDT"]
    assert result["errors"][0]["error_type"] == "TimeoutError"
    assert len(list(tmp_path.rglob("*.json.gz"))) == 1


@pytest.mark.asyncio
async def test_complete_artifact_replays_with_same_identity(tmp_path: Path):
    config = _config()
    first = await write_scheduler_artifact(
        _FakeClient(),
        slot_timestamp_ms=DAY_START_MS,
        execution_started_ms=DAY_START_MS + 100,
        config=config,
        config_sha256="config",
        scheduler_code_sha256="scheduler",
        collector_code_sha256="collector",
        output_root=tmp_path / "first",
    )
    slot, started, probes = load_scheduler_replay(first)
    second = await write_scheduler_artifact(
        ReplayBookClient(probes),
        slot_timestamp_ms=slot,
        execution_started_ms=started,
        config=config,
        config_sha256="config",
        scheduler_code_sha256="scheduler",
        collector_code_sha256="collector",
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert (first / "run_result.json").read_bytes() == (second / "run_result.json").read_bytes()
