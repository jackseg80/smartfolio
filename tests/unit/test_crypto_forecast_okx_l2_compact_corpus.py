import io
import json
import tarfile
from pathlib import Path

import pytest

from services.forecasting.okx_l2_compact_corpus import (
    build_compact_corpus,
    write_compact_corpus_artifact,
)
from services.forecasting.okx_l2_pilot import DAY_MS, file_sha256

DAY_START_MS = 1_719_792_000_000
INTERVAL_MS = DAY_MS // 2


def _snapshot(timestamp_ms: int, bid: str = "99") -> dict[str, object]:
    return {
        "instId": "BTC-USDT",
        "action": "snapshot",
        "ts": str(timestamp_ms),
        "asks": [["101", "2", "1"]],
        "bids": [[bid, "3", "1"]],
    }


def _update(timestamp_ms: int) -> dict[str, object]:
    return {
        "instId": "BTC-USDT",
        "action": "update",
        "ts": str(timestamp_ms),
        "asks": [],
        "bids": [["100", "500", "9"]],
    }


def _records() -> list[dict[str, object]]:
    second_grid = DAY_START_MS + INTERVAL_MS
    return [
        _snapshot(DAY_START_MS),
        _update(DAY_START_MS + 10),
        _snapshot(second_grid - 100, bid="98"),
        _snapshot(second_grid + 100, bid="97"),
    ]


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
    archive = {
        "date_utc": "2024-07-01",
        "instrument": "BTC-USDT",
        "filename": path.name,
        "compressed_bytes": size,
        "sha256": file_sha256(path),
        "expected_native_snapshots": 3,
        "expected_available_snapshots": 2,
        "expected_missing_grid_timestamps_ms": [],
        "expected_selected_before_grid": 1,
        "expected_selected_exactly_on_grid": 1,
        "expected_selected_after_grid": 0,
        "expected_maximum_absolute_offset_ms": 100,
    }
    return {
        "schema_version": "crypto-forecast-okx-l2-compact-corpus-v1",
        "provider": "fixture",
        "source_sample_artifact_id": "sample",
        "source_sample_result_sha256": "1" * 64,
        "source_alignment_artifact_id": "alignment",
        "source_alignment_result_sha256": "2" * 64,
        "source_method_artifact_id": "method",
        "source_method_result_sha256": "3" * 64,
        "dates_utc": ["2024-07-01"],
        "instruments": ["BTC-USDT"],
        "archives": [archive],
        "expected_archive_count": 1,
        "expected_total_compressed_bytes": size,
        "expected_total_native_snapshots": 3,
        "expected_total_selected_snapshots": 2,
        "expected_total_missing_snapshots": 0,
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
        "maximum_total_compact_bytes": 1_000_000,
        "selection_rule": "nearest_valid_snapshot_within_symmetric_window_prefer_prior_on_tie",
        "availability_rule": "maximum_of_grid_and_source_timestamp",
        "missing_policy": "preserve_missing_without_fill_or_interpolation",
        "processing_order": "date_then_instrument_one_archive_at_a_time",
        "raw_archive_policy": "retain_without_modification_until_explicit_cleanup_authorization",
    }


def test_builds_nested_compact_corpus_with_frozen_totals(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, _records())

    result, payloads, _performance = build_compact_corpus(tmp_path, _config(archive))

    assert result["decision"] == "GO_TECHNICAL_COMPACT_CORPUS"
    assert result["totals"]["selected_snapshots"] == 2
    assert result["totals"]["updates_ignored"] == 1
    assert list(payloads) == ["2024-07-01/BTC-USDT.jsonl.gz"]
    assert result["archives"][0]["matches_frozen_corpus"] is True


def test_unexpected_missing_slot_is_explicit_no_go(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, [_snapshot(DAY_START_MS)])
    config = _config(archive)
    config["archives"][0]["expected_native_snapshots"] = 1
    config["expected_total_native_snapshots"] = 1

    result, _payloads, _performance = build_compact_corpus(tmp_path, config)

    assert result["decision"] == "NO_GO_TECHNICAL_COMPACT_CORPUS"
    assert result["totals"]["missing_snapshots"] == 1
    assert result["archives"][0]["snapshots"]["missing_grid_timestamps_ms"] == [
        DAY_START_MS + INTERVAL_MS
    ]


def test_rejects_archive_path_and_inconsistent_frozen_totals(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, _records())
    config = _config(archive)
    config["archives"][0]["filename"] = "../fixture.tar.gz"
    with pytest.raises(ValueError, match="must not contain a path"):
        build_compact_corpus(tmp_path, config)

    config = _config(archive)
    config["expected_total_selected_snapshots"] = 3
    with pytest.raises(ValueError, match="selected snapshot total"):
        build_compact_corpus(tmp_path, config)

    config = _config(archive)
    config["archives"][0]["expected_available_snapshots"] = 1
    config["archives"][0]["expected_missing_grid_timestamps_ms"] = [DAY_START_MS + 1]
    config["archives"][0]["expected_selected_before_grid"] = 0
    config["expected_total_selected_snapshots"] = 1
    config["expected_total_missing_snapshots"] = 1
    with pytest.raises(ValueError, match="frozen grid slots"):
        build_compact_corpus(tmp_path, config)


def test_artifact_is_reproducible_and_rejects_tampering(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_archive(archive, _records())
    config = _config(archive)
    first_result, first_payloads, first_performance = build_compact_corpus(tmp_path, config)
    second_result, second_payloads, second_performance = build_compact_corpus(tmp_path, config)

    first_artifact = write_compact_corpus_artifact(
        first_result,
        first_payloads,
        first_performance,
        config_sha256="4" * 64,
        corpus_code_sha256="5" * 64,
        extraction_code_sha256="6" * 64,
        output_root=tmp_path / "first-output",
    )
    second_artifact = write_compact_corpus_artifact(
        second_result,
        second_payloads,
        second_performance,
        config_sha256="4" * 64,
        corpus_code_sha256="5" * 64,
        extraction_code_sha256="6" * 64,
        output_root=tmp_path / "second-output",
    )

    assert first_result == second_result
    assert first_payloads == second_payloads
    assert first_artifact.name == second_artifact.name
    assert file_sha256(first_artifact / "compact_corpus_result.json") == file_sha256(
        second_artifact / "compact_corpus_result.json"
    )

    first_payloads["2024-07-01/BTC-USDT.jsonl.gz"] += b"tampered"
    with pytest.raises(ValueError, match="size mismatch"):
        write_compact_corpus_artifact(
            first_result,
            first_payloads,
            first_performance,
            config_sha256="4" * 64,
            corpus_code_sha256="5" * 64,
            extraction_code_sha256="6" * 64,
            output_root=tmp_path / "tampered-output",
        )
