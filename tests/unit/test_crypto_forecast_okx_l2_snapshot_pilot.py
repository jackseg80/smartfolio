import io
import json
import tarfile
from pathlib import Path

import pytest

from services.forecasting.okx_l2_pilot import file_sha256
from services.forecasting.okx_l2_snapshot_pilot import analyze_snapshot_archive

DAY_START_MS = 1_788_998_400_000


def _snapshot(timestamp_ms: int, bid: str = "99", ask: str = "101") -> dict[str, object]:
    return {
        "instId": "SOL-USDT",
        "action": "snapshot",
        "ts": str(timestamp_ms),
        "asks": [[ask, "2", "1"]],
        "bids": [[bid, "3", "1"]],
    }


def _update(timestamp_ms: int, bid: str = "100") -> dict[str, object]:
    return {
        "instId": "SOL-USDT",
        "action": "update",
        "ts": str(timestamp_ms),
        "asks": [],
        "bids": [[bid, "500", "9"]],
    }


def _records(update_bid: str = "100") -> list[dict[str, object]]:
    records = []
    for index in range(96):
        timestamp_ms = DAY_START_MS + index * 900_000 + (4 if index == 95 else 0)
        records.append(_snapshot(timestamp_ms))
        records.append(_update(timestamp_ms + 10, update_bid))
    return records


def _write_archive(path: Path, records: list[dict[str, object]]) -> None:
    payload = b"".join(
        json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n" for record in records
    )
    info = tarfile.TarInfo("fixture.data")
    info.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))


def _config(path: Path) -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-okx-l2-snapshot-pilot-v1",
        "provider": "fixture",
        "instrument": "SOL-USDT",
        "date_utc": "2026-09-10",
        "expected_archive_sha256": file_sha256(path),
        "download_limit_bytes": 1_000_000,
        "uncompressed_limit_bytes": 1_000_000,
        "max_records": 1_000,
        "max_line_bytes": 100_000,
        "expected_snapshot_count": 96,
        "snapshot_interval_ms": 900_000,
        "snapshot_interval_tolerance_ms": 1_000,
        "maximum_levels_per_side": 400,
        "depth_bands_bps": [10, 25, 50],
    }


def test_updates_are_counted_but_never_used_for_snapshot_features(tmp_path: Path):
    first_archive = tmp_path / "first.tar.gz"
    second_archive = tmp_path / "second.tar.gz"
    _write_archive(first_archive, _records(update_bid="100"))
    _write_archive(second_archive, _records(update_bid="10000"))

    first_result, first_snapshots = analyze_snapshot_archive(first_archive, _config(first_archive))
    second_result, second_snapshots = analyze_snapshot_archive(
        second_archive, _config(second_archive)
    )

    assert first_snapshots == second_snapshots
    assert first_result["records"]["updates_ignored_for_features"] == 96
    assert second_result["records"]["updates_ignored_for_features"] == 96
    assert first_result["decision"] == "GO_TECHNICAL"
    assert second_result["decision"] == "GO_TECHNICAL"


def test_snapshot_gap_failure_is_published_as_no_go(tmp_path: Path):
    records = _records()
    records[2]["ts"] = str(DAY_START_MS + 901_001)
    records[3]["ts"] = str(DAY_START_MS + 901_011)
    archive = tmp_path / "bad-gap.tar.gz"
    _write_archive(archive, records)

    result, _snapshots = analyze_snapshot_archive(archive, _config(archive))

    assert result["decision"] == "NO_GO_TECHNICAL"
    assert "cadence" in " ".join(result["decision_reasons"]).lower()


def test_snapshot_hash_and_depth_limits_are_enforced(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, _records())
    config = _config(archive)
    config["expected_archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        analyze_snapshot_archive(archive, config)

    config = _config(archive)
    config["maximum_levels_per_side"] = 0
    with pytest.raises(ValueError, match="bid depth"):
        analyze_snapshot_archive(archive, config)
