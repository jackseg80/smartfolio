import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

from services.forecasting.okx_l2_pilot import DAY_MS, file_sha256
from services.forecasting.okx_l2_progressive_extraction import (
    extract_progressive_dataset,
    write_progressive_artifact,
)

DAY_START_MS = 1_719_792_000_000
INTERVAL_MS = DAY_MS // 2


def _snapshot(timestamp_ms: int, bid: str = "99", ask: str = "101") -> dict[str, object]:
    return {
        "instId": "BTC-USDT",
        "action": "snapshot",
        "ts": str(timestamp_ms),
        "asks": [[ask, "2", "1"]],
        "bids": [[bid, "3", "1"]],
    }


def _update(timestamp_ms: int, bid: str) -> dict[str, object]:
    return {
        "instId": "BTC-USDT",
        "action": "update",
        "ts": str(timestamp_ms),
        "asks": [],
        "bids": [[bid, "500", "9"]],
    }


def _write_archive(path: Path, records: list[dict[str, object]]) -> None:
    payload = b"".join(
        json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n" for record in records
    )
    member = tarfile.TarInfo("fixture.data")
    member.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(payload))


def _config(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    digest = file_sha256(path)
    return {
        "schema_version": "crypto-forecast-okx-l2-progressive-extraction-v1",
        "provider": "fixture",
        "source_sample_artifact_id": "sample",
        "source_sample_result_sha256": "1" * 64,
        "source_alignment_artifact_id": "alignment",
        "source_alignment_result_sha256": "2" * 64,
        "date_utc": "2024-07-01",
        "instruments": ["BTC-USDT"],
        "archives": [
            {
                "instrument": "BTC-USDT",
                "filename": path.name,
                "compressed_bytes": size,
                "sha256": digest,
                "expected_native_snapshots": 3,
                "expected_selected_before_grid": 1,
                "expected_selected_exactly_on_grid": 1,
                "expected_selected_after_grid": 0,
                "expected_maximum_absolute_offset_ms": 100,
            }
        ],
        "expected_archive_count": 1,
        "expected_total_compressed_bytes": size,
        "maximum_single_archive_bytes": 1_000_000,
        "maximum_uncompressed_member_bytes": 1_000_000,
        "max_records_per_archive": 100,
        "max_line_bytes": 100_000,
        "maximum_levels_per_side": 400,
        "depth_bands_bps": [10, 25, 50],
        "grid_interval_ms": INTERVAL_MS,
        "maximum_absolute_offset_ms": 1_000,
        "expected_grid_slots_per_archive": 2,
        "minimum_available_slots_per_archive": 2,
        "expected_total_selected_snapshots": 2,
        "maximum_total_compact_bytes": 1_000_000,
        "selection_rule": "nearest_valid_snapshot_within_symmetric_window_prefer_prior_on_tie",
        "availability_rule": "maximum_of_grid_and_source_timestamp",
        "missing_policy": "preserve_missing_without_fill_or_interpolation",
        "processing_order": "one_archive_at_a_time",
        "raw_archive_policy": "retain_without_modification_until_explicit_cleanup_authorization",
    }


def _records(update_bid: str = "100") -> list[dict[str, object]]:
    second_grid = DAY_START_MS + INTERVAL_MS
    return [
        _snapshot(DAY_START_MS),
        _update(DAY_START_MS + 10, update_bid),
        _snapshot(second_grid - 100, bid="98"),
        _snapshot(second_grid + 100, bid="97"),
    ]


def test_extracts_only_aligned_snapshots_and_prefers_prior_tie(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, _records())

    result, payloads, _performance = extract_progressive_dataset(tmp_path, _config(archive))

    assert result["decision"] == "GO_TECHNICAL_PROGRESSIVE_EXTRACTION"
    assert result["totals"]["selected_snapshots"] == 2
    assert result["totals"]["updates_ignored"] == 1
    summary = result["archives"][0]
    assert summary["snapshots"]["selected_before_grid"] == 1
    assert summary["snapshots"]["selected_after_grid"] == 0
    assert summary["snapshots"]["maximum_absolute_offset_ms"] == 100
    assert list(payloads) == ["BTC-USDT.jsonl.gz"]
    decompressed = gzip.decompress(payloads["BTC-USDT.jsonl.gz"])
    lines = decompressed.splitlines()
    assert decompressed.endswith(b"\n")
    assert len(lines) == 2
    assert [json.loads(line)["grid_timestamp_ms"] for line in lines] == [
        DAY_START_MS,
        DAY_START_MS + INTERVAL_MS,
    ]


def test_update_mutation_cannot_change_compact_snapshots(tmp_path: Path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_archive = first_root / "fixture.tar.gz"
    second_archive = second_root / "fixture.tar.gz"
    _write_archive(first_archive, _records(update_bid="100"))
    _write_archive(second_archive, _records(update_bid="999999"))

    _first_result, first_payloads, _ = extract_progressive_dataset(
        first_root, _config(first_archive)
    )
    _second_result, second_payloads, _ = extract_progressive_dataset(
        second_root, _config(second_archive)
    )

    assert first_payloads == second_payloads


def test_missing_slot_is_explicit_and_published_as_no_go(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, [_snapshot(DAY_START_MS)])
    config = _config(archive)
    spec = config["archives"][0]
    spec["expected_selected_before_grid"] = 0
    spec["expected_maximum_absolute_offset_ms"] = 0

    result, _payloads, _performance = extract_progressive_dataset(tmp_path, config)

    assert result["decision"] == "NO_GO_TECHNICAL_PROGRESSIVE_EXTRACTION"
    assert result["totals"]["missing_snapshots"] == 1
    assert result["archives"][0]["snapshots"]["missing_grid_timestamps_ms"] == [
        DAY_START_MS + INTERVAL_MS
    ]


def test_result_and_compact_payload_are_reproducible(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, _records())
    config = _config(archive)

    first_result, first_payloads, first_performance = extract_progressive_dataset(tmp_path, config)
    second_result, second_payloads, second_performance = extract_progressive_dataset(
        tmp_path, config
    )
    first_artifact = write_progressive_artifact(
        first_result,
        first_payloads,
        first_performance,
        config_sha256="3" * 64,
        extraction_code_sha256="4" * 64,
        output_root=tmp_path / "first-output",
    )
    second_artifact = write_progressive_artifact(
        second_result,
        second_payloads,
        second_performance,
        config_sha256="3" * 64,
        extraction_code_sha256="4" * 64,
        output_root=tmp_path / "second-output",
    )

    assert first_result == second_result
    assert first_payloads == second_payloads
    assert first_artifact.name == second_artifact.name
    assert file_sha256(first_artifact / "progressive_extraction_result.json") == file_sha256(
        second_artifact / "progressive_extraction_result.json"
    )
    assert file_sha256(first_artifact / "data" / "BTC-USDT.jsonl.gz") == file_sha256(
        second_artifact / "data" / "BTC-USDT.jsonl.gz"
    )


def test_rejects_archive_path_escape_and_hash_mismatch(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, _records())
    config = _config(archive)
    config["archives"][0]["filename"] = "../fixture.tar.gz"
    with pytest.raises(ValueError, match="must not contain a path"):
        extract_progressive_dataset(tmp_path, config)

    config = _config(archive)
    config["archives"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        extract_progressive_dataset(tmp_path, config)


def test_artifact_refuses_payload_that_does_not_match_result(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, _records())
    result, payloads, performance = extract_progressive_dataset(tmp_path, _config(archive))
    payloads["BTC-USDT.jsonl.gz"] += b"tampered"

    with pytest.raises(ValueError, match="size mismatch"):
        write_progressive_artifact(
            result,
            payloads,
            performance,
            config_sha256="3" * 64,
            extraction_code_sha256="4" * 64,
            output_root=tmp_path / "output",
        )
