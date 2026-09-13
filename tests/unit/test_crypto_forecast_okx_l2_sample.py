import csv
import hashlib
import json
from pathlib import Path

import pytest

from services.forecasting.okx_l2_sample import (
    ArchiveSpec,
    build_sample_result,
    load_candidate_specs,
    load_verified_archives,
    validate_actual_size,
    write_sample_artifact,
)


def _metadata(path: Path) -> str:
    rows = []
    for date in ("2023-04-01", "2024-07-01"):
        for instrument in ("BTC-USDT", "SOL-USDT"):
            filename = f"{instrument}-L2orderbook-400lv-{date}.tar.gz"
            rows.append(
                {
                    "date_utc": date,
                    "date_timestamp_ms": "0",
                    "instrument": instrument,
                    "filename": filename,
                    "size_mb": "1.00",
                    "url": f"https://static.okx.com/example/{filename}",
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(metadata_hash: str) -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-okx-l2-sample-v1",
        "provider": "fixture",
        "coverage_artifact_id": "coverage-fixture",
        "coverage_metadata_sha256": metadata_hash,
        "instruments": ["BTC-USDT", "SOL-USDT"],
        "dates_utc": ["2023-04-01", "2024-07-01"],
        "expected_archive_count": 4,
        "expected_total_snapshot_count": 8,
        "total_download_limit_bytes": 10 * 1024 * 1024,
        "maximum_single_file_bytes": 2 * 1024 * 1024,
        "advertised_size_tolerance_bytes": 1024,
    }


def test_candidate_metadata_is_exact_and_pinned(tmp_path: Path):
    path = tmp_path / "metadata.csv"
    config = _config(_metadata(path))

    specs = load_candidate_specs(path, config)

    assert len(specs) == 4
    assert specs[0].date_utc == "2023-04-01"
    assert specs[0].instrument == "BTC-USDT"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_candidate_specs(path, config)


def test_downloaded_size_must_match_advertised_metadata():
    spec = ArchiveSpec(
        date_utc="2023-04-01",
        instrument="BTC-USDT",
        filename="BTC-USDT-L2orderbook-400lv-2023-04-01.tar.gz",
        advertised_size_mib="1.00",
        url="https://static.okx.com/example.tar.gz",
    )
    config = {
        "maximum_single_file_bytes": 2 * 1024 * 1024,
        "advertised_size_tolerance_bytes": 1024,
    }

    validate_actual_size(spec, 1024 * 1024 + 100, config)
    with pytest.raises(ValueError, match="frozen metadata"):
        validate_actual_size(spec, 1024 * 1024 + 1025, config)


def test_global_decision_requires_every_archive_and_snapshot():
    config = _config("hash")
    manifest = {
        "schema_version": "crypto-forecast-okx-l2-sample-v1",
        "provider": "fixture",
        "coverage_artifact_id": "coverage-fixture",
        "coverage_metadata_sha256": "hash",
        "archive_count": 4,
        "total_compressed_bytes": 4,
        "files": [],
        "credentials_used": False,
        "orders_placed": False,
    }
    result_template = {
        "decision": "GO_TECHNICAL",
        "snapshots": {"valid_uncrossed": 2},
        "records": {"updates_ignored_for_features": 3},
    }
    metrics = [{"timestamp_ms": index} for index in range(8)]

    result = build_sample_result([result_template] * 4, metrics, manifest, config)
    assert result["decision"] == "GO_TECHNICAL"
    assert result["sample"]["updates_ignored_for_features"] == 12

    result = build_sample_result([result_template] * 3, metrics, manifest, config)
    assert result["decision"] == "NO_GO_TECHNICAL"


def test_reproduction_rejects_duplicate_manifest_entries(tmp_path: Path):
    spec = ArchiveSpec(
        date_utc="2023-04-01",
        instrument="BTC-USDT",
        filename="BTC-USDT-L2orderbook-400lv-2023-04-01.tar.gz",
        advertised_size_mib="1.00",
        url="https://static.okx.com/example.tar.gz",
    )
    entry = {
        **spec.__dict__,
        "compressed_bytes": 1024 * 1024,
        "sha256": "0" * 64,
    }
    manifest = {
        "schema_version": "crypto-forecast-okx-l2-sample-v1",
        "provider": "fixture",
        "coverage_artifact_id": "coverage-fixture",
        "coverage_metadata_sha256": "hash",
        "archive_count": 2,
        "total_compressed_bytes": 2 * 1024 * 1024,
        "files": [entry, entry],
    }
    (tmp_path / "acquisition_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        load_verified_archives([spec], tmp_path, _config("hash"))


def test_normalized_artifact_is_reproducible(tmp_path: Path):
    archive_result = {
        "date_utc": "2023-04-01",
        "instrument": "BTC-USDT",
        "archive": {"filename": "a.tar.gz", "compressed_bytes": 1, "sha256": "abc"},
        "records": {"total": 2, "updates_ignored_for_features": 1},
        "snapshots": {"total": 1, "valid_uncrossed": 1},
        "decision": "GO_TECHNICAL",
    }
    result = {
        "schema_version": "crypto-forecast-okx-l2-sample-v1",
        "archive_results": [archive_result],
        "decision": "GO_TECHNICAL",
    }
    metrics = [
        {
            "date_utc": "2023-04-01",
            "instrument": "BTC-USDT",
            "timestamp_ms": 1,
            "valid": True,
        }
    ]

    first = write_sample_artifact(
        result,
        metrics,
        config_sha256="config",
        sample_code_sha256="sample",
        snapshot_code_sha256="snapshot",
        output_root=tmp_path / "first",
    )
    second = write_sample_artifact(
        result,
        metrics,
        config_sha256="config",
        sample_code_sha256="sample",
        snapshot_code_sha256="snapshot",
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert (first / "sample_result.json").read_bytes() == (
        second / "sample_result.json"
    ).read_bytes()
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["result_sha256"] == second_manifest["result_sha256"]
