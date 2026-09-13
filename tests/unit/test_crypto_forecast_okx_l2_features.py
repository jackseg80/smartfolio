import csv
import gzip
import io
import json
from pathlib import Path

import pytest

from services.forecasting.okx_l2_features import (
    META_FIELDS,
    build_l2_feature_table,
    compute_snapshot_features,
    write_l2_feature_artifact,
)
from services.forecasting.okx_l2_pilot import DAY_MS, file_sha256

DAY_START_MS = 1_719_792_000_000
INTERVAL_MS = DAY_MS // 2
FEATURE_FIELDS = [
    "best_bid",
    "best_ask",
    "mid",
    "spread_bps",
    "bid_levels",
    "ask_levels",
    "best_bid_size",
    "best_ask_size",
    "top_level_imbalance",
    "microprice",
    "microprice_deviation_bps",
    "bid_depth_5bps_usdt",
    "ask_depth_5bps_usdt",
    "imbalance_5bps",
    "bid_depth_10bps_usdt",
    "ask_depth_10bps_usdt",
    "imbalance_10bps",
    "bid_depth_25bps_usdt",
    "ask_depth_25bps_usdt",
    "imbalance_25bps",
    "bid_depth_50bps_usdt",
    "ask_depth_50bps_usdt",
    "imbalance_50bps",
    "bid_top5_50bps_share",
    "ask_top5_50bps_share",
]
REFERENCE_FIELDS = [
    "best_bid",
    "best_ask",
    "mid",
    "spread_bps",
    "bid_levels",
    "ask_levels",
    "bid_depth_10bps_usdt",
    "ask_depth_10bps_usdt",
    "imbalance_10bps",
    "bid_depth_25bps_usdt",
    "ask_depth_25bps_usdt",
    "imbalance_25bps",
    "bid_depth_50bps_usdt",
    "ask_depth_50bps_usdt",
    "imbalance_50bps",
]


def _snapshot(grid_timestamp_ms: int, bid: str) -> dict[str, object]:
    source_timestamp_ms = grid_timestamp_ms + 10
    bid_value = float(bid)
    ask = f"{bid_value + 0.02:.2f}"
    second_ask = f"{bid_value + 0.03:.2f}"
    return {
        "schema_version": "crypto-forecast-okx-l2-progressive-extraction-v1",
        "provider": "fixture",
        "instrument": "BTC-USDT",
        "date_utc": "2024-07-01",
        "grid_timestamp_ms": grid_timestamp_ms,
        "source_timestamp_ms": source_timestamp_ms,
        "signed_offset_ms": 10,
        "absolute_offset_ms": 10,
        "availability_timestamp_ms": source_timestamp_ms,
        "bids": [[bid, "3", "1"], [f"{bid_value - 0.01:.2f}", "2", "1"]],
        "asks": [[ask, "2", "1"], [second_ask, "1", "1"]],
    }


def _write_gzip_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write(payload)


def _base_config() -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-okx-l2-features-v1",
        "provider": "fixture",
        "source_artifact_id": "corpus-fixture",
        "source_result_sha256": "",
        "source_files": [],
        "reference_alignment_artifact_id": "alignment-fixture",
        "reference_alignment_result_sha256": "",
        "reference_metrics_sha256": "",
        "dates_utc": ["2024-07-01"],
        "instruments": ["BTC-USDT"],
        "grid_interval_ms": INTERVAL_MS,
        "maximum_absolute_offset_ms": 1_000,
        "expected_grid_slots_per_group": 2,
        "expected_total_rows": 2,
        "expected_available_rows": 2,
        "expected_missing_rows": 0,
        "expected_missing_keys": [],
        "max_compressed_bytes_per_file": 1_000_000,
        "max_uncompressed_bytes_per_file": 1_000_000,
        "max_line_bytes": 100_000,
        "max_rows_per_file": 2,
        "depth_bands_bps": [5, 10, 25, 50],
        "concentration_band_bps": 50,
        "concentration_levels": 5,
        "feature_fields": FEATURE_FIELDS,
        "reference_fields": REFERENCE_FIELDS,
        "reference_relative_tolerance": 1e-12,
        "reference_absolute_tolerance": 1e-9,
        "feature_time_semantics": "independent_snapshot_only_at_causal_availability_timestamp",
        "missing_policy": "explicit_row_without_feature_values",
        "target_policy": "no_target_or_future_return_in_this_lot",
    }


def _create_workspace(
    root: Path, source_rows: list[dict[str, object]], *, missing_second: bool = False
) -> tuple[Path, Path, dict[str, object]]:
    corpus_root = root / "corpus"
    source_relative = "data/2024-07-01/BTC-USDT.jsonl.gz"
    source_path = corpus_root / "data" / "2024-07-01" / "BTC-USDT.jsonl.gz"
    _write_gzip_jsonl(source_path, source_rows)
    config = _base_config()
    if missing_second:
        config["expected_available_rows"] = 1
        config["expected_missing_rows"] = 1
        config["expected_missing_keys"] = [
            {
                "date_utc": "2024-07-01",
                "instrument": "BTC-USDT",
                "grid_timestamp_ms": DAY_START_MS + INTERVAL_MS,
            }
        ]
    source_entry = {
        "path": source_relative,
        "bytes": source_path.stat().st_size,
        "sha256": file_sha256(source_path),
    }
    config["source_files"] = [source_entry]
    corpus_result = {
        "schema_version": "crypto-forecast-okx-l2-compact-corpus-v1",
        "decision": "GO_TECHNICAL_COMPACT_CORPUS",
    }
    corpus_result_path = corpus_root / "compact_corpus_result.json"
    corpus_result_path.write_text(json.dumps(corpus_result), encoding="utf-8")
    config["source_result_sha256"] = file_sha256(corpus_result_path)
    (corpus_root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "corpus-fixture",
                "result_sha256": config["source_result_sha256"],
                "data_files": [source_entry],
            }
        ),
        encoding="utf-8",
    )

    reference_root = root / "reference"
    reference_root.mkdir()
    reference_path = reference_root / "aligned_snapshot_metrics.csv"
    reference_rows: list[dict[str, object]] = []
    by_grid = {int(row["grid_timestamp_ms"]): row for row in source_rows}
    for slot in range(2):
        grid_timestamp = DAY_START_MS + slot * INTERVAL_MS
        source = by_grid.get(grid_timestamp)
        if source is None:
            reference_rows.append(
                {
                    "date_utc": "2024-07-01",
                    "instrument": "BTC-USDT",
                    "grid_timestamp_ms": grid_timestamp,
                    "source_timestamp_ms": "",
                    "signed_offset_ms": "",
                    "absolute_offset_ms": "",
                    "availability_timestamp_ms": "",
                    "available": False,
                    **{field: "" for field in REFERENCE_FIELDS},
                }
            )
        else:
            features = compute_snapshot_features(source, config)
            reference_rows.append(
                {
                    "date_utc": "2024-07-01",
                    "instrument": "BTC-USDT",
                    "grid_timestamp_ms": grid_timestamp,
                    "source_timestamp_ms": source["source_timestamp_ms"],
                    "signed_offset_ms": source["signed_offset_ms"],
                    "absolute_offset_ms": source["absolute_offset_ms"],
                    "availability_timestamp_ms": source["availability_timestamp_ms"],
                    "available": True,
                    **{field: features[field] for field in REFERENCE_FIELDS},
                }
            )
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date_utc",
                "instrument",
                "grid_timestamp_ms",
                "source_timestamp_ms",
                "signed_offset_ms",
                "absolute_offset_ms",
                "availability_timestamp_ms",
                "available",
                *REFERENCE_FIELDS,
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(reference_rows)
    alignment_result = {
        "schema_version": "crypto-forecast-okx-l2-symmetric-alignment-v1",
        "decision": "GO_TECHNICAL_ALIGNMENT",
    }
    alignment_result_path = reference_root / "alignment_result.json"
    alignment_result_path.write_text(json.dumps(alignment_result), encoding="utf-8")
    config["reference_alignment_result_sha256"] = file_sha256(alignment_result_path)
    config["reference_metrics_sha256"] = file_sha256(reference_path)
    (reference_root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "alignment-fixture",
                "result_sha256": config["reference_alignment_result_sha256"],
                "metrics_sha256": config["reference_metrics_sha256"],
            }
        ),
        encoding="utf-8",
    )
    return corpus_root, reference_path, config


def test_builds_independent_features_and_matches_reference(tmp_path: Path):
    rows = [_snapshot(DAY_START_MS, "99"), _snapshot(DAY_START_MS + INTERVAL_MS, "98")]
    corpus_root, reference_path, config = _create_workspace(tmp_path, rows)

    result, features = build_l2_feature_table(corpus_root, reference_path, config)

    assert result["decision"] == "GO_TECHNICAL_L2_FEATURES"
    assert result["table"]["rows"] == 2
    assert result["table"]["feature_count"] == 25
    assert result["reference"]["mismatch_count"] == 0
    assert features[0]["best_bid"] <= features[0]["microprice"] <= features[0]["best_ask"]


def test_preserves_expected_missing_row_without_features(tmp_path: Path):
    corpus_root, reference_path, config = _create_workspace(
        tmp_path, [_snapshot(DAY_START_MS, "99")], missing_second=True
    )

    result, features = build_l2_feature_table(corpus_root, reference_path, config)

    assert result["decision"] == "GO_TECHNICAL_L2_FEATURES"
    assert result["table"]["missing_rows"] == 1
    assert features[1]["available"] is False
    assert features[1]["best_bid"] == ""
    assert features[1]["missing_reason"] == "no_snapshot_within_symmetric_window"


def test_future_snapshot_mutation_cannot_change_earlier_features(tmp_path: Path):
    first_rows = [_snapshot(DAY_START_MS, "99"), _snapshot(DAY_START_MS + INTERVAL_MS, "98")]
    second_rows = [_snapshot(DAY_START_MS, "99"), _snapshot(DAY_START_MS + INTERVAL_MS, "90")]
    first_corpus, first_reference, first_config = _create_workspace(tmp_path / "first", first_rows)
    second_corpus, second_reference, second_config = _create_workspace(
        tmp_path / "second", second_rows
    )

    _first_result, first_features = build_l2_feature_table(
        first_corpus, first_reference, first_config
    )
    _second_result, second_features = build_l2_feature_table(
        second_corpus, second_reference, second_config
    )

    assert first_features[0] == second_features[0]
    assert first_features[1] != second_features[1]


def test_rejects_broken_artifact_provenance(tmp_path: Path):
    rows = [_snapshot(DAY_START_MS, "99"), _snapshot(DAY_START_MS + INTERVAL_MS, "98")]
    corpus_root, reference_path, config = _create_workspace(tmp_path, rows)
    (corpus_root / "compact_corpus_result.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="result SHA-256"):
        build_l2_feature_table(corpus_root, reference_path, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("concentration_band_bps", 25, "top5_50bps"),
        ("concentration_levels", 3, "top5_50bps"),
        ("feature_time_semantics", "future_aware", "causal snapshot"),
        ("missing_policy", "interpolate", "explicit and empty"),
        ("target_policy", "future_return", "future returns are forbidden"),
        ("reference_absolute_tolerance", -1.0, "finite and non-negative"),
    ],
)
def test_rejects_semantically_mislabeled_feature_contract(
    tmp_path: Path, field: str, value: object, message: str
):
    rows = [_snapshot(DAY_START_MS, "99"), _snapshot(DAY_START_MS + INTERVAL_MS, "98")]
    corpus_root, reference_path, config = _create_workspace(tmp_path, rows)
    config[field] = value

    with pytest.raises(ValueError, match=message):
        build_l2_feature_table(corpus_root, reference_path, config)


def test_feature_artifact_is_reproducible(tmp_path: Path):
    rows = [_snapshot(DAY_START_MS, "99"), _snapshot(DAY_START_MS + INTERVAL_MS, "98")]
    corpus_root, reference_path, config = _create_workspace(tmp_path, rows)
    result, features = build_l2_feature_table(corpus_root, reference_path, config)

    first = write_l2_feature_artifact(
        result,
        features,
        config_sha256="4" * 64,
        feature_code_sha256="5" * 64,
        output_root=tmp_path / "first-output",
    )
    second = write_l2_feature_artifact(
        result,
        features,
        config_sha256="4" * 64,
        feature_code_sha256="5" * 64,
        output_root=tmp_path / "second-output",
    )

    assert first.name == second.name
    assert file_sha256(first / "l2_feature_result.json") == file_sha256(
        second / "l2_feature_result.json"
    )
    assert file_sha256(first / "l2_features.csv") == file_sha256(second / "l2_features.csv")
