import json
from pathlib import Path

import pytest

from services.forecasting.okx_l2_collection_feasibility import (
    BookProbe,
    build_feasibility_result,
    load_book_probes,
    validate_book_probe,
    write_feasibility_artifact,
)


def _body(*, crossed: bool = False) -> bytes:
    best_ask = "99" if crossed else "101"
    payload = {
        "code": "0",
        "msg": "",
        "data": [
            {
                "asks": [[best_ask, "2", "0", "1"], ["102", "3", "0", "2"]],
                "bids": [["100", "1", "0", "2"], ["99", "4", "0", "1"]],
                "ts": "2000",
                "seqId": 7,
            }
        ],
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def _probe(instrument: str = "BTC-USDT", *, crossed: bool = False) -> BookProbe:
    return BookProbe(
        instrument=instrument,
        requested_at_ms=2005,
        completed_at_ms=2010,
        response_body=_body(crossed=crossed),
    )


def _config() -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-okx-l2-collection-feasibility-v1",
        "provider": "fixture",
        "endpoint": "/books",
        "instruments": ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        "book_depth_per_side": 2,
        "snapshots_per_day_per_instrument": 96,
        "minimum_continuous_days": 10,
        "chronological_design_days": {
            "training": 2,
            "purge_after_training": 1,
            "validation": 2,
            "purge_after_validation": 1,
            "calibration": 1,
            "purge_after_calibration": 1,
            "final_test": 2,
        },
        "maximum_target_horizon_days": 1,
        "historical_complete_day_size_mb": {
            "median": "1000",
            "p95": "1500",
            "maximum": "2000",
        },
        "maximum_local_archive_history_gib": "1",
        "maximum_snapshot_age_ms": 60_000,
        "maximum_future_lead_ms": 5_000,
        "maximum_response_bytes_per_instrument": 131_072,
        "minimum_levels_per_side": 2,
        "maximum_levels_per_side": 2,
        "maximum_projected_raw_collection_gib": "10",
        "gzip_storage_safety_multiplier": "3",
        "maximum_projected_retained_collection_gib": "5",
        "official_rate_limit_requests": 40,
        "official_rate_limit_window_seconds": 2,
        "maximum_planned_burst_requests": 3,
    }


def test_valid_probe_preserves_payload_size_hash_and_book_shape():
    result = validate_book_probe(_probe(), _config())

    assert result["valid"] is True
    assert result["response_bytes"] == len(_body())
    assert result["bid_levels"] == 2
    assert result["ask_levels"] == 2
    assert result["best_bid"] == "100"
    assert result["best_ask"] == "101"


def test_crossed_book_is_rejected():
    with pytest.raises(ValueError, match="Crossed order book"):
        validate_book_probe(_probe(crossed=True), _config())


def test_result_separates_historical_no_go_from_prospective_go():
    probes = [_probe(instrument) for instrument in _config()["instruments"]]

    result = build_feasibility_result(probes, _config())

    assert result["historical_archives"]["decision"] == "NO_GO_LOCAL_ARCHIVE_HISTORY"
    assert result["prospective_collection"]["decision"] == "GO_PROSPECTIVE_COLLECTION_FEASIBILITY"
    assert result["prospective_collection"]["calls_per_day"] == 288
    assert result["decision"] == "GO_PROSPECTIVE_COLLECTION_FEASIBILITY"


def test_artifact_replays_with_same_identity(tmp_path: Path):
    config = _config()
    probes = [_probe(instrument) for instrument in config["instruments"]]
    result = build_feasibility_result(probes, config)

    first = write_feasibility_artifact(
        result,
        probes,
        config_sha256="config",
        feasibility_code_sha256="code",
        output_root=tmp_path / "first",
    )
    replayed = load_book_probes(first)
    second = write_feasibility_artifact(
        build_feasibility_result(replayed, config),
        replayed,
        config_sha256="config",
        feasibility_code_sha256="code",
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert (first / "feasibility_result.json").read_bytes() == (
        second / "feasibility_result.json"
    ).read_bytes()
