import copy
import json
from pathlib import Path

import pytest

from services.forecasting.compact_signal_feasibility import (
    evaluate_compact_signal_feasibility,
    write_compact_signal_feasibility_artifact,
)
from services.forecasting.okx_l2_pilot import file_sha256


def _config() -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-compact-signal-feasibility-v1",
        "required_start_date_on_or_before": "2022-06-16",
        "minimum_calendar_span_days": 1551,
        "minimum_independent_daily_dates": 480,
        "minimum_covered_assets": 5,
        "maximum_estimated_raw_bytes": 256 * 1024 * 1024,
        "hard_gates": ["official", "coverage"],
        "candidate_families": [{"id": "first"}, {"id": "second"}],
        "forbidden_actions": [
            "dataset_download",
            "future_target_read",
            "feature_materialization",
            "model_training",
            "backtest",
            "production_change",
            "real_order",
        ],
    }


def _evidence() -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-compact-signal-evidence-v1",
        "candidates": [
            {
                "family_id": "first",
                "priority": 1,
                "pilot_allowed": True,
                "eligible_signals": ["funding"],
                "gates": {"official": "pass", "coverage": "unverified"},
            },
            {
                "family_id": "second",
                "priority": 2,
                "pilot_allowed": False,
                "eligible_signals": [],
                "gates": {"official": "pass", "coverage": "fail"},
            },
        ],
    }


def test_fail_closed_decisions_and_recommendation():
    result = evaluate_compact_signal_feasibility(_config(), _evidence())

    assert result["recommended_family_id"] == "first"
    assert result["recommended_decision"] == "GO_PILOT"
    assert [item["decision"] for item in result["candidates"]] == ["GO_PILOT", "NO_GO"]
    assert result["dataset_downloaded"] is False
    assert result["future_targets_read"] is False


def test_all_pass_is_primary():
    evidence = _evidence()
    evidence["candidates"][0]["gates"]["coverage"] = "pass"

    result = evaluate_compact_signal_feasibility(_config(), evidence)

    assert result["recommended_decision"] == "GO_PRIMARY"
    assert result["next_step"] == "pre_register_bounded_acquisition_for_recommended_family"


def test_unverified_gate_without_bounded_resolution_is_no_go():
    evidence = _evidence()
    evidence["candidates"][0]["pilot_allowed"] = False

    result = evaluate_compact_signal_feasibility(_config(), evidence)

    assert result["candidates"][0]["decision"] == "NO_GO"


def test_rejects_gate_order_drift():
    evidence = _evidence()
    evidence["candidates"][0]["gates"] = {"coverage": "unverified", "official": "pass"}

    with pytest.raises(ValueError, match="Gate order differs"):
        evaluate_compact_signal_feasibility(_config(), evidence)


def test_artifact_identity_is_reproducible(tmp_path: Path):
    result = evaluate_compact_signal_feasibility(_config(), _evidence())
    first = write_compact_signal_feasibility_artifact(
        result,
        config_sha256="1" * 64,
        evidence_sha256="2" * 64,
        feasibility_code_sha256="3" * 64,
        output_root=tmp_path / "first",
    )
    second = write_compact_signal_feasibility_artifact(
        copy.deepcopy(result),
        config_sha256="1" * 64,
        evidence_sha256="2" * 64,
        feasibility_code_sha256="3" * 64,
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert file_sha256(first / "compact_signal_feasibility_result.json") == file_sha256(
        second / "compact_signal_feasibility_result.json"
    )
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_downloaded"] is False
    assert manifest["network_used_by_evaluator"] is False
