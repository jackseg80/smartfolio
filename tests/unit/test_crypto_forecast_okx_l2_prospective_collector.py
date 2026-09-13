import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.forecasting.okx_l2_collection_feasibility import BookProbe
from services.forecasting.okx_l2_prospective_collector import (
    build_daily_manifest,
    capture_pilot_probes,
    load_pilot_probes,
    record_probe,
    write_daily_manifest,
    write_pilot_artifact,
)

DAY_START_MS = int(datetime(2026, 9, 13, tzinfo=timezone.utc).timestamp() * 1000)


def _body(best_bid: str = "100", *, provider_timestamp_ms: int | None = None) -> bytes:
    payload = {
        "code": "0",
        "msg": "",
        "data": [
            {
                "asks": [["101", "2", "0", "1"], ["102", "3", "0", "2"]],
                "bids": [[best_bid, "1", "0", "2"], ["99", "4", "0", "1"]],
                "ts": str(provider_timestamp_ms or DAY_START_MS + 1000),
                "seqId": 7,
            }
        ],
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def _probe(
    instrument: str, best_bid: str = "100", *, slot_timestamp_ms: int = DAY_START_MS
) -> BookProbe:
    return BookProbe(
        instrument=instrument,
        requested_at_ms=slot_timestamp_ms + 900,
        completed_at_ms=slot_timestamp_ms + 1100,
        response_body=_body(best_bid, provider_timestamp_ms=slot_timestamp_ms + 1000),
    )


def _config() -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-okx-l2-prospective-collector-v1",
        "provider": "fixture",
        "endpoint": "/books",
        "source_feasibility_artifact_id": "source",
        "source_feasibility_result_sha256": "source-hash",
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
        "request_delay_seconds": 0,
    }


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    async def get_book(self, endpoint: str, instrument: str, depth: int) -> BookProbe:
        self.calls.append((endpoint, instrument, depth))
        return _probe(instrument)


@pytest.mark.asyncio
async def test_pilot_capture_uses_collector_schema_and_frozen_universe():
    client = _FakeClient()

    probes = await capture_pilot_probes(client, _config())

    assert [probe.instrument for probe in probes] == ["BTC-USDT", "ETH-USDT"]
    assert client.calls == [("/books", "BTC-USDT", 2), ("/books", "ETH-USDT", 2)]


def test_identical_retry_is_idempotent_and_conflict_preserves_original(tmp_path: Path):
    config = _config()
    first = record_probe(
        tmp_path,
        slot_timestamp_ms=DAY_START_MS,
        probe=_probe("BTC-USDT"),
        config=config,
        pilot_mode=False,
    )
    original = (tmp_path / first["path"]).read_bytes()

    duplicate = record_probe(
        tmp_path,
        slot_timestamp_ms=DAY_START_MS,
        probe=_probe("BTC-USDT"),
        config=config,
        pilot_mode=False,
    )
    with pytest.raises(FileExistsError, match="Conflicting capture"):
        record_probe(
            tmp_path,
            slot_timestamp_ms=DAY_START_MS,
            probe=_probe("BTC-USDT", "100.5"),
            config=config,
            pilot_mode=False,
        )

    assert first["status"] == "written"
    assert duplicate["status"] == "duplicate_identical"
    assert list((tmp_path / "2026-09-13").glob("*.json.gz")) == [tmp_path / first["path"]]
    assert (tmp_path / first["path"]).read_bytes() == original


def test_manifest_reports_due_gap_and_orphan_without_touching_future(tmp_path: Path):
    config = _config()
    record_probe(
        tmp_path,
        slot_timestamp_ms=DAY_START_MS,
        probe=_probe("BTC-USDT"),
        config=config,
        pilot_mode=False,
    )
    orphan = tmp_path / "2026-09-13" / ".interrupted.pending"
    orphan.write_bytes(b"partial")
    as_of_ms = DAY_START_MS + int(config["grid_interval_ms"]) + 119_999

    manifest = build_daily_manifest(
        tmp_path,
        date_utc="2026-09-13",
        as_of_ms=as_of_ms,
        config=config,
    )

    assert manifest["due_captures"] == 2
    assert [item["instrument"] for item in manifest["missing_due_captures"]] == ["ETH-USDT"]
    assert manifest["not_yet_due_captures"] == 6
    assert manifest["orphan_temporary_files"] == ["2026-09-13/.interrupted.pending"]
    assert orphan.exists()


def test_recovery_manifest_is_stable_and_later_slot_does_not_mutate_first(tmp_path: Path):
    config = _config()
    first = record_probe(
        tmp_path,
        slot_timestamp_ms=DAY_START_MS,
        probe=_probe("BTC-USDT"),
        config=config,
        pilot_mode=False,
    )
    first_path = tmp_path / first["path"]
    original = first_path.read_bytes()
    as_of_ms = DAY_START_MS + 120_000
    before = build_daily_manifest(tmp_path, date_utc="2026-09-13", as_of_ms=as_of_ms, config=config)
    manifest_path = write_daily_manifest(tmp_path, before, config)
    first_manifest = manifest_path.read_bytes()
    later_slot = DAY_START_MS + int(config["grid_interval_ms"])
    record_probe(
        tmp_path,
        slot_timestamp_ms=later_slot,
        probe=_probe("BTC-USDT", "100.25", slot_timestamp_ms=later_slot),
        config=config,
        pilot_mode=False,
    )
    second_manifest = write_daily_manifest(
        tmp_path,
        build_daily_manifest(tmp_path, date_utc="2026-09-13", as_of_ms=as_of_ms, config=config),
        config,
    )

    assert second_manifest.read_bytes() != first_manifest
    assert first_path.read_bytes() == original


def test_pilot_artifact_replays_with_same_identity(tmp_path: Path):
    config = _config()
    probes = [_probe(instrument) for instrument in config["instruments"]]
    first = write_pilot_artifact(
        probes,
        slot_timestamp_ms=DAY_START_MS,
        config=config,
        config_sha256="config",
        collector_code_sha256="code",
        output_root=tmp_path / "first",
    )
    replay_slot, replay_probes = load_pilot_probes(first)
    second = write_pilot_artifact(
        replay_probes,
        slot_timestamp_ms=replay_slot,
        config=config,
        config_sha256="config",
        collector_code_sha256="code",
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert (first / "pilot_result.json").read_bytes() == (second / "pilot_result.json").read_bytes()


def test_replay_rejects_path_escape(tmp_path: Path):
    config = _config()
    probes = [_probe(instrument) for instrument in config["instruments"]]
    artifact = write_pilot_artifact(
        probes,
        slot_timestamp_ms=DAY_START_MS,
        config=config,
        config_sha256="config",
        collector_code_sha256="code",
        output_root=tmp_path / "artifact",
    )
    result_path = artifact / "pilot_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["daily_manifest"]["captures"][0]["path"] = "../../outside.json.gz"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe stored capture path"):
        load_pilot_probes(artifact)
