import io
import json
import tarfile
from pathlib import Path

import pytest

from services.forecasting.okx_l2_pilot import analyze_archive, file_sha256


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
        "schema_version": "crypto-forecast-okx-l2-pilot-v1",
        "provider": "fixture",
        "instrument": "SOL-USDT",
        "date_utc": "2026-09-10",
        "expected_archive_sha256": file_sha256(path),
        "download_limit_bytes": 1_000_000,
        "uncompressed_limit_bytes": 1_000_000,
        "max_records": 100,
        "max_line_bytes": 100_000,
        "sample_interval_ms": 60_000,
        "required_sequence_coverage": 0.99,
        "depth_bands_bps": [10, 25, 50],
    }


def test_archive_reconstruction_is_causal_and_deletes_zero_size_levels(tmp_path: Path):
    start = 1_788_998_400_000
    records = [
        {
            "instId": "SOL-USDT",
            "action": "snapshot",
            "ts": str(start),
            "asks": [["101", "2", "1"]],
            "bids": [["99", "3", "1"]],
        },
        {
            "instId": "SOL-USDT",
            "action": "update",
            "ts": str(start + 60_000),
            "asks": [["101", "0", "0"], ["102", "4", "1"]],
            "bids": [],
        },
    ]
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, records)

    result, samples = analyze_archive(archive, _config(archive))

    assert samples[0]["timestamp_ms"] == start
    assert samples[0]["best_ask"] == 101.0
    assert samples[1]["timestamp_ms"] == start + 60_000
    assert samples[1]["best_ask"] == 102.0
    assert result["records"]["actions"] == {"snapshot": 1, "update": 1}
    assert result["decision"] == "NO_GO_TECHNICAL"
    assert result["sequence_integrity"]["field_coverage"] == 0.0
    assert result["snapshot_cadence"]["count"] == 1
    assert result["snapshot_cadence"]["snapshot_only_follow_up_candidate"] is False


def test_sequence_links_are_required_for_go(tmp_path: Path):
    start = 1_788_998_400_000
    records = [
        {
            "instId": "SOL-USDT",
            "action": "snapshot",
            "ts": str(start),
            "seqId": "10",
            "prevSeqId": "-1",
            "asks": [["101", "2", "1"]],
            "bids": [["99", "3", "1"]],
        },
        {
            "instId": "SOL-USDT",
            "action": "update",
            "ts": str(start + 10),
            "seqId": "11",
            "prevSeqId": "10",
            "asks": [],
            "bids": [["100", "1", "1"]],
        },
    ]
    archive = tmp_path / "fixture-sequence.tar.gz"
    _write_archive(archive, records)

    result, _samples = analyze_archive(archive, _config(archive))

    assert result["sequence_integrity"]["link_coverage"] == 1.0
    assert result["sequence_integrity"]["verifiable_at_required_threshold"] is True
    assert result["decision"] == "GO_TECHNICAL"


def test_archive_hash_and_limits_are_enforced(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, [])
    config = _config(archive)
    config["expected_archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        analyze_archive(archive, config)

    config["expected_archive_sha256"] = file_sha256(archive)
    config["download_limit_bytes"] = 1
    with pytest.raises(ValueError, match="Compressed archive"):
        analyze_archive(archive, config)
