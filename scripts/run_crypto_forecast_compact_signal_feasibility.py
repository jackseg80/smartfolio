"""Evaluate the frozen compact-signal evidence without downloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.compact_signal_feasibility import (  # noqa: E402
    evaluate_compact_signal_feasibility,
    write_compact_signal_feasibility_artifact,
)
from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_compact_signal_feasibility.json",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_compact_signal_evidence.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-compact-signal-feasibility",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    result = evaluate_compact_signal_feasibility(config, evidence)
    artifact = write_compact_signal_feasibility_artifact(
        result,
        config_sha256=file_sha256(args.config),
        evidence_sha256=file_sha256(args.evidence),
        feasibility_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "compact_signal_feasibility.py"
        ),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "recommended_family_id": result["recommended_family_id"],
                "recommended_decision": result["recommended_decision"],
                "candidate_decisions": {
                    item["family_id"]: item["decision"] for item in result["candidates"]
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
