import csv
import hashlib
import json
from pathlib import Path

import pytest

from services.forecasting.okx_l2_normalization import (
    SnapshotMetric,
    load_snapshot_metrics,
    normalize_snapshot_metrics,
    write_normalization_artifact,
)

DAY_START_MS = 1_680_307_200_000
METRIC_FIELDS = ["valid", "reason", "best_bid", "best_ask", "bid_levels", "ask_levels"]


def _values(best_bid: str = "99") -> dict[str, str]:
    return {
        "valid": "True",
        "reason": "ok",
        "best_bid": best_bid,
        "best_ask": "101",
        "bid_levels": "100",
        "ask_levels": "100",
    }


def _config(*, expected_slots: int = 2, minimum_available: int = 2) -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-okx-l2-normalization-v1",
        "provider": "fixture",
        "source_artifact_id": "source",
        "source_result_sha256": "result",
        "source_metrics_sha256": "metrics",
        "instruments": ["BTC-USDT"],
        "dates_utc": ["2023-04-01"],
        "expected_archive_count": 1,
        "maximum_source_rows": 10,
        "grid_interval_ms": 86_400_000 // expected_slots,
        "minimum_source_lag_ms": 0,
        "maximum_source_lag_ms": 1000,
        "expected_grid_slots_per_archive": expected_slots,
        "minimum_available_slots_per_archive": minimum_available,
        "expected_total_grid_slots": expected_slots,
        "selection_rule": "earliest_valid_snapshot_at_or_after_grid_within_lag_window",
        "missing_policy": "preserve_missing_without_fill_or_interpolation",
    }


def test_selects_first_snapshot_after_grid_and_ignores_other_rows():
    rows = [
        SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + 5, _values("99")),
        SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + 60_000, _values("1")),
        SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + 43_200_000 + 6, _values("98")),
    ]

    result, normalized = normalize_snapshot_metrics(rows, METRIC_FIELDS, _config())

    assert result["decision"] == "GO_TECHNICAL_NORMALIZATION"
    assert [row["best_bid"] for row in normalized] == ["99", "98"]
    assert [row["source_lag_ms"] for row in normalized] == [5, 6]
    assert result["grid"]["ignored_source_snapshots"] == 1

    mutated = [
        rows[0],
        SnapshotMetric(
            rows[1].date_utc,
            rows[1].instrument,
            rows[1].timestamp_ms,
            _values("1000000"),
        ),
        rows[2],
    ]
    mutated_result, mutated_normalized = normalize_snapshot_metrics(
        mutated, METRIC_FIELDS, _config()
    )
    assert normalized == mutated_normalized
    assert result["grid"] == mutated_result["grid"]


def test_missing_slot_is_explicit_and_never_filled():
    rows = [SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + 60_000, _values())]

    result, normalized = normalize_snapshot_metrics(
        rows, METRIC_FIELDS, _config(expected_slots=1, minimum_available=1)
    )

    assert result["decision"] == "NO_GO_TECHNICAL_NORMALIZATION"
    assert normalized[0]["available"] is False
    assert normalized[0]["source_timestamp_ms"] == ""
    assert normalized[0]["reason"] == "no_snapshot_within_lag_window"


def test_source_metrics_are_hash_pinned_and_finite(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    row = {
        "date_utc": "2023-04-01",
        "instrument": "BTC-USDT",
        "timestamp_ms": str(DAY_START_MS + 5),
        **_values(),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    config = _config(expected_slots=1, minimum_available=1)
    config["source_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    rows, fields = load_snapshot_metrics(path, config)
    assert len(rows) == 1
    assert fields == METRIC_FIELDS

    path.write_text(path.read_text(encoding="utf-8").replace("99", "nan"), encoding="utf-8")
    config["source_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Non-finite"):
        load_snapshot_metrics(path, config)


def test_normalized_artifact_is_reproducible(tmp_path: Path):
    result = {
        "schema_version": "crypto-forecast-okx-l2-normalization-v1",
        "decision": "GO_TECHNICAL_NORMALIZATION",
    }
    rows = [{"date_utc": "2023-04-01", "available": True}]

    first = write_normalization_artifact(
        result,
        rows,
        config_sha256="config",
        normalization_code_sha256="code",
        output_root=tmp_path / "first",
    )
    second = write_normalization_artifact(
        result,
        rows,
        config_sha256="config",
        normalization_code_sha256="code",
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert (first / "normalization_result.json").read_bytes() == (
        second / "normalization_result.json"
    ).read_bytes()
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["result_sha256"] == second_manifest["result_sha256"]
