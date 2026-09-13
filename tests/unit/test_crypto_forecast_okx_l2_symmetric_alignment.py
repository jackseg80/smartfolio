import json
from pathlib import Path

from services.forecasting.okx_l2_normalization import SnapshotMetric
from services.forecasting.okx_l2_symmetric_alignment import (
    align_snapshot_metrics,
    write_alignment_artifact,
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


def _config(*, expected_slots: int = 3, minimum_available: int = 3) -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-okx-l2-symmetric-alignment-v1",
        "provider": "fixture",
        "source_artifact_id": "source",
        "source_result_sha256": "result",
        "source_metrics_sha256": "metrics",
        "instruments": ["BTC-USDT"],
        "dates_utc": ["2023-04-01"],
        "expected_archive_count": 1,
        "grid_interval_ms": 86_400_000 // expected_slots,
        "maximum_absolute_offset_ms": 1000,
        "expected_grid_slots_per_archive": expected_slots,
        "minimum_available_slots_per_archive": minimum_available,
        "expected_total_grid_slots": expected_slots,
        "selection_rule": "nearest_valid_snapshot_within_symmetric_window_prefer_prior_on_tie",
        "availability_rule": "maximum_of_grid_and_source_timestamp",
        "missing_policy": "preserve_missing_without_fill_or_interpolation",
    }


def test_nearest_selection_prefers_prior_on_tie_and_preserves_availability():
    interval = 86_400_000 // 3
    rows = [
        SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + 5, _values("99")),
        SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + interval - 7, _values("98")),
        SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + interval + 7, _values("97")),
        SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + 2 * interval - 4, _values("96")),
    ]

    result, aligned = align_snapshot_metrics(rows, METRIC_FIELDS, _config())

    assert result["decision"] == "GO_TECHNICAL_ALIGNMENT"
    assert [row["best_bid"] for row in aligned] == ["99", "98", "96"]
    assert [row["signed_offset_ms"] for row in aligned] == [5, -7, -4]
    assert aligned[0]["availability_timestamp_ms"] == DAY_START_MS + 5
    assert aligned[1]["availability_timestamp_ms"] == DAY_START_MS + interval


def test_missing_slot_is_explicit_and_not_filled():
    rows = [SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + 60_000, _values())]

    result, aligned = align_snapshot_metrics(
        rows, METRIC_FIELDS, _config(expected_slots=1, minimum_available=1)
    )

    assert result["decision"] == "NO_GO_TECHNICAL_ALIGNMENT"
    assert aligned[0]["available"] is False
    assert aligned[0]["availability_timestamp_ms"] == ""
    assert aligned[0]["reason"] == "no_snapshot_within_symmetric_window"


def test_unselected_snapshot_mutation_cannot_change_alignment():
    interval = 43_200_000
    first = SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + 5, _values("99"))
    ignored = SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + 60_000, _values("1"))
    second = SnapshotMetric("2023-04-01", "BTC-USDT", DAY_START_MS + interval + 5, _values("98"))
    config = _config(expected_slots=2, minimum_available=2)

    _result, baseline = align_snapshot_metrics([first, ignored, second], METRIC_FIELDS, config)
    mutated_ignored = SnapshotMetric(
        ignored.date_utc, ignored.instrument, ignored.timestamp_ms, _values("1000000")
    )
    _mutated_result, mutated = align_snapshot_metrics(
        [first, mutated_ignored, second], METRIC_FIELDS, config
    )

    assert baseline == mutated


def test_alignment_artifact_is_reproducible(tmp_path: Path):
    result = {
        "schema_version": "crypto-forecast-okx-l2-symmetric-alignment-v1",
        "decision": "GO_TECHNICAL_ALIGNMENT",
    }
    rows = [{"date_utc": "2023-04-01", "available": True}]

    first = write_alignment_artifact(
        result,
        rows,
        config_sha256="config",
        alignment_code_sha256="code",
        output_root=tmp_path / "first",
    )
    second = write_alignment_artifact(
        result,
        rows,
        config_sha256="config",
        alignment_code_sha256="code",
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert (first / "alignment_result.json").read_bytes() == (
        second / "alignment_result.json"
    ).read_bytes()
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["metrics_sha256"] == second_manifest["metrics_sha256"]
