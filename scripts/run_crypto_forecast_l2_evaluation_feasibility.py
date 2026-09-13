"""Evaluate whether the frozen OKX L2 features can support causal prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_l2_evaluation_feasibility import (  # noqa: E402
    evaluate_l2_evaluation_feasibility,
    write_l2_evaluation_feasibility_artifact,
)
from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_evaluation_feasibility.json"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "crypto-forecast-l2-features-reviewed"
            / "crypto-forecast-okx-l2-features-v1-de3238a8dcf7a722"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-evaluation-feasibility",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate_l2_evaluation_feasibility(args.source_root, config)
    artifact = write_l2_evaluation_feasibility_artifact(
        result,
        config_sha256=file_sha256(args.config),
        feasibility_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_evaluation_feasibility.py"
        ),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "raw_grid_rows": result["observed"]["raw_grid_rows"],
                "independent_utc_dates": result["observed"]["independent_utc_dates"],
                "calendar_span_days": result["observed"]["calendar_span_days"],
                "panel_entities": result["observed"]["panel_entities"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
