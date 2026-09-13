import csv
import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from services.forecasting.binance_funding_features import (
    FEATURE_FIELDS,
    FundingEvent,
    _parse_event_file,
    build_feature_rows,
    build_funding_feature_result,
    file_sha256,
    load_funding_events,
    run_future_mutation_tests,
    validate_config,
    write_funding_feature_artifact,
)


def _timestamp(day: date, hour: int) -> int:
    return int(datetime.combine(day, time(hour), tzinfo=timezone.utc).timestamp() * 1000)


def _config() -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-binance-funding-features-v1",
        "provider": "fixture",
        "source_artifact_id": "funding-fixture",
        "source_manifest_sha256": "",
        "start_date": "2024-01-01",
        "end_date": "2024-02-09",
        "expected_calendar_days": 40,
        "expected_instruments": 1,
        "expected_rows": 40,
        "warmup_days": 30,
        "expected_model_eligible_rows": 11,
        "maximum_event_file_bytes": 1_000_000,
        "decision_time_semantics": "end_of_utc_day_with_events_strictly_before_next_midnight",
        "missing_policy": "explicit_empty_value_until_full_window",
        "normalization_policy": "none_in_this_lot",
        "target_policy": "no_target_or_future_return_in_this_lot",
        "rolling_windows_days": [3, 7, 30],
        "future_mutation_cutoff_dates": ["2024-01-10", "2024-01-20", "2024-01-30"],
        "feature_fields": FEATURE_FIELDS,
        "source_files": [
            {
                "symbol": "BTC",
                "market_symbol": "BTCUSDT",
                "path": "events/BTCUSDT_funding.csv",
                "sha256": "a" * 64,
                "observations": 80,
            }
        ],
    }


def _histories() -> dict[str, tuple[FundingEvent, ...]]:
    start = date(2024, 1, 1)
    events = []
    for offset in range(40):
        day = start + timedelta(days=offset)
        events.extend(
            [
                FundingEvent(_timestamp(day, 0), day.isoformat(), 8, "0.0001"),
                FundingEvent(_timestamp(day, 8), day.isoformat(), 8, "-0.00005"),
            ]
        )
    return {"BTCUSDT": tuple(events)}


def _write_source(root: Path, config: dict[str, object]) -> None:
    event_path = root / "events" / "BTCUSDT_funding.csv"
    event_path.parent.mkdir(parents=True)
    with event_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "timestamp_utc",
                "calc_time_ms",
                "utc_date",
                "funding_interval_hours",
                "funding_rate",
            ]
        )
        for event in _histories()["BTCUSDT"]:
            timestamp = datetime.fromtimestamp(event.calc_time_ms / 1000, timezone.utc).isoformat()
            writer.writerow(
                [
                    timestamp,
                    event.calc_time_ms,
                    event.utc_date,
                    event.funding_interval_hours,
                    event.funding_rate,
                ]
            )
    source = config["source_files"][0]
    source["sha256"] = file_sha256(event_path)
    manifest = {
        "schema_version": "crypto-forecast-binance-funding-history-v1",
        "artifact_id": "funding-fixture",
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "targets_created": False,
        "predictive_features_created": False,
        "inputs": [
            {
                "symbol": "BTC",
                "market_symbol": "BTCUSDT",
                "events_file": source["path"],
                "events_file_sha256": source["sha256"],
                "observations": source["observations"],
            }
        ],
    }
    manifest_path = root / "acquisition_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["source_manifest_sha256"] = file_sha256(manifest_path)


def test_feature_windows_are_explicit_and_causal():
    rows = build_feature_rows(_histories(), _config())

    assert len(rows) == 40
    assert rows[0]["funding_daily_sum"] == pytest.approx(0.00005)
    assert rows[0]["funding_sum_3d"] == ""
    assert rows[2]["funding_sum_3d"] == pytest.approx(0.00015)
    assert rows[28]["model_eligible"] is False
    assert rows[29]["model_eligible"] is True
    assert all(rows[29][field] != "" for field in FEATURE_FIELDS)
    assert rows[29]["funding_sum_zscore_30d"] == 0.0
    assert rows[0]["latest_source_timestamp_ms"] < rows[0]["decision_timestamp_ms"]


def test_future_mutation_never_changes_protected_rows():
    config = _config()
    histories = _histories()
    rows = build_feature_rows(histories, config)

    results = run_future_mutation_tests(histories, rows, config)

    assert len(results) == 3
    assert all(item["protected_divergences"] == 0 for item in results)
    assert all(item["future_mutation_effective"] is True for item in results)
    assert all(item["future_rows_changed"] > 0 for item in results)


def test_loads_only_the_frozen_source_artifact(tmp_path: Path):
    config = _config()
    _write_source(tmp_path, config)

    histories = load_funding_events(tmp_path, config)

    assert len(histories["BTCUSDT"]) == 80
    (tmp_path / "events" / "BTCUSDT_funding.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_funding_events(tmp_path, config)


def test_event_timestamp_must_be_explicit_utc(tmp_path: Path):
    path = tmp_path / "events.csv"
    instant_ms = _timestamp(date(2024, 1, 1), 0)
    path.write_text(
        "timestamp_utc,calc_time_ms,utc_date,funding_interval_hours,funding_rate\n"
        f"2024-01-01T01:00:00+01:00,{instant_ms},2024-01-01,8,0.0001\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid funding semantics"):
        _parse_event_file(
            path,
            {"maximum_bytes": 10_000, "observations": 1},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("decision_time_semantics", "future_aware", "next UTC midnight"),
        ("missing_policy", "fill", "explicit and empty"),
        ("normalization_policy", "global", "Normalization is forbidden"),
        ("target_policy", "future_return", "future returns are forbidden"),
        ("rolling_windows_days", [7, 30], "3/7/30-day"),
        ("future_mutation_cutoff_dates", ["2024-01-20"], "cutoffs"),
    ],
)
def test_rejects_noncausal_or_unfrozen_contract(field: str, value: object, message: str):
    config = _config()
    config[field] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_result_and_artifact_are_reproducible(tmp_path: Path):
    config = _config()
    source_root = tmp_path / "source"
    _write_source(source_root, config)
    result, rows = build_funding_feature_result(source_root, config)

    first = write_funding_feature_artifact(
        result,
        rows,
        config_sha256="1" * 64,
        feature_code_sha256="2" * 64,
        output_root=tmp_path / "first",
    )
    second = write_funding_feature_artifact(
        result,
        rows,
        config_sha256="1" * 64,
        feature_code_sha256="2" * 64,
        output_root=tmp_path / "second",
    )

    assert result["decision"] == "GO_CAUSAL_FUNDING_FEATURES"
    assert first.name == second.name
    assert file_sha256(first / "funding_feature_result.json") == file_sha256(
        second / "funding_feature_result.json"
    )
    assert file_sha256(first / "funding_features.csv") == file_sha256(
        second / "funding_features.csv"
    )
