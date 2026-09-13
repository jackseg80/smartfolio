"""Deterministic offline gate for compact historical signal candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from services.forecasting.okx_l2_pilot import file_sha256

COMPACT_SIGNAL_FEASIBILITY_SCHEMA_VERSION = "crypto-forecast-compact-signal-feasibility-v1"
COMPACT_SIGNAL_EVIDENCE_SCHEMA_VERSION = "crypto-forecast-compact-signal-evidence-v1"
ALLOWED_GATE_STATUSES = {"pass", "fail", "unverified"}


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + ("\n" if pretty else "")).encode("utf-8")


def _validate_inputs(config: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    if config.get("schema_version") != COMPACT_SIGNAL_FEASIBILITY_SCHEMA_VERSION:
        raise ValueError("Unsupported compact-signal feasibility schema")
    if evidence.get("schema_version") != COMPACT_SIGNAL_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported compact-signal evidence schema")

    hard_gates = [str(value) for value in config.get("hard_gates", [])]
    if not hard_gates or len(hard_gates) != len(set(hard_gates)):
        raise ValueError("Hard gates must be non-empty and unique")
    family_ids = [str(item["id"]) for item in config.get("candidate_families", [])]
    candidates = list(evidence.get("candidates", []))
    evidence_ids = [str(item["family_id"]) for item in candidates]
    if sorted(family_ids) != sorted(evidence_ids) or len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("Evidence must contain every frozen candidate family exactly once")

    priorities = [int(item["priority"]) for item in candidates]
    if sorted(priorities) != list(range(1, len(candidates) + 1)):
        raise ValueError("Candidate priorities must form a unique one-based sequence")
    for candidate in candidates:
        gates = candidate.get("gates")
        if not isinstance(gates, dict) or list(gates) != hard_gates:
            raise ValueError(f"Gate order differs for {candidate.get('family_id')}")
        invalid = set(gates.values()) - ALLOWED_GATE_STATUSES
        if invalid:
            raise ValueError(f"Invalid gate status for {candidate.get('family_id')}: {invalid}")
        if not isinstance(candidate.get("pilot_allowed"), bool):
            raise ValueError("pilot_allowed must be boolean")

    if int(config["minimum_calendar_span_days"]) <= 0:
        raise ValueError("Minimum calendar span must be positive")
    if int(config["minimum_independent_daily_dates"]) <= 0:
        raise ValueError("Minimum independent-date count must be positive")
    if int(config["minimum_covered_assets"]) <= 0:
        raise ValueError("Minimum asset coverage must be positive")
    if int(config["maximum_estimated_raw_bytes"]) <= 0:
        raise ValueError("Maximum estimated size must be positive")
    forbidden = set(config.get("forbidden_actions", []))
    required_forbidden = {
        "dataset_download",
        "future_target_read",
        "feature_materialization",
        "model_training",
        "backtest",
        "production_change",
        "real_order",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("Safety boundary is incomplete")


def evaluate_compact_signal_feasibility(
    config: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen fail-closed policy to a documented evidence snapshot."""

    _validate_inputs(config, evidence)
    hard_gates = [str(value) for value in config["hard_gates"]]
    results: list[dict[str, Any]] = []
    for candidate in sorted(evidence["candidates"], key=lambda item: int(item["priority"])):
        gates = dict(candidate["gates"])
        failed = [name for name in hard_gates if gates[name] == "fail"]
        unverified = [name for name in hard_gates if gates[name] == "unverified"]
        if failed:
            decision = "NO_GO"
            reason = "one_or_more_hard_gates_failed"
        elif unverified and bool(candidate["pilot_allowed"]):
            decision = "GO_PILOT"
            reason = "bounded_pilot_required_to_close_unverified_gates"
        elif unverified:
            decision = "NO_GO"
            reason = "unverified_gates_not_resolvable_by_authorized_bounded_pilot"
        else:
            decision = "GO_PRIMARY"
            reason = "all_hard_gates_passed"
        results.append(
            {
                "family_id": candidate["family_id"],
                "priority": int(candidate["priority"]),
                "decision": decision,
                "reason": reason,
                "passed_gate_count": sum(gates[name] == "pass" for name in hard_gates),
                "failed_gates": failed,
                "unverified_gates": unverified,
                "eligible_signals": list(candidate.get("eligible_signals", [])),
                "rejected_signals": dict(candidate.get("rejected_signals", {})),
                "observations": dict(candidate.get("observations", {})),
                "unverified_resolution": candidate.get("unverified_resolution"),
                "official_sources": list(candidate.get("official_sources", [])),
                "gates": gates,
            }
        )

    go_primary = [item for item in results if item["decision"] == "GO_PRIMARY"]
    go_pilot = [item for item in results if item["decision"] == "GO_PILOT"]
    recommended = (go_primary or go_pilot or [None])[0]
    return {
        "schema_version": COMPACT_SIGNAL_FEASIBILITY_SCHEMA_VERSION,
        "evidence_schema_version": COMPACT_SIGNAL_EVIDENCE_SCHEMA_VERSION,
        "frozen_thresholds": {
            "required_start_date_on_or_before": config["required_start_date_on_or_before"],
            "minimum_calendar_span_days": int(config["minimum_calendar_span_days"]),
            "minimum_independent_daily_dates": int(config["minimum_independent_daily_dates"]),
            "minimum_covered_assets": int(config["minimum_covered_assets"]),
            "maximum_estimated_raw_bytes": int(config["maximum_estimated_raw_bytes"]),
        },
        "hard_gates": hard_gates,
        "candidates": results,
        "recommended_family_id": recommended["family_id"] if recommended else None,
        "recommended_decision": recommended["decision"] if recommended else "NO_GO",
        "next_step": (
            "pre_register_bounded_acquisition_for_recommended_family"
            if recommended and recommended["decision"] in {"GO_PRIMARY", "GO_PILOT"}
            else "no_collection_authorized"
        ),
        "dataset_downloaded": False,
        "future_targets_read": False,
        "features_materialized": False,
        "model_trained": False,
        "backtest_run": False,
        "orders_placed": False,
        "production_touched": False,
    }


def write_compact_signal_feasibility_artifact(
    result: Mapping[str, Any],
    *,
    config_sha256: str,
    evidence_sha256: str,
    feasibility_code_sha256: str,
    output_root: Path,
) -> Path:
    """Write a content-addressed artifact without mutating the evidence."""

    identity = {
        **result,
        "config_sha256": config_sha256,
        "evidence_sha256": evidence_sha256,
        "feasibility_code_sha256": feasibility_code_sha256,
    }
    artifact_id = (
        f"{COMPACT_SIGNAL_FEASIBILITY_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Compact-signal feasibility artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        result_path = temporary_path / "compact_signal_feasibility_result.json"
        result_path.write_bytes(_json_bytes(identity, pretty=True))
        matrix_path = temporary_path / "candidate_matrix.csv"
        with matrix_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "priority",
                "family_id",
                "decision",
                "passed_gate_count",
                "failed_gates",
                "unverified_gates",
                "eligible_signals",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for candidate in result["candidates"]:
                writer.writerow(
                    {
                        "priority": candidate["priority"],
                        "family_id": candidate["family_id"],
                        "decision": candidate["decision"],
                        "passed_gate_count": candidate["passed_gate_count"],
                        "failed_gates": ";".join(candidate["failed_gates"]),
                        "unverified_gates": ";".join(candidate["unverified_gates"]),
                        "eligible_signals": ";".join(candidate["eligible_signals"]),
                    }
                )
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_file": result_path.name,
            "result_sha256": file_sha256(result_path),
            "matrix_file": matrix_path.name,
            "matrix_sha256": file_sha256(matrix_path),
            "dataset_downloaded": False,
            "future_targets_read": False,
            "model_trained": False,
            "network_used_by_evaluator": False,
            "credentials_used": False,
            "orders_placed": False,
            "production_touched": False,
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
