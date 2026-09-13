"""Build the frozen causal feature table from the compact OKX L2 corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_l2_features import (  # noqa: E402
    build_l2_feature_table,
    write_l2_feature_artifact,
)
from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_features.json",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "crypto-forecast-l2-compact-corpus-corrected"
            / "crypto-forecast-okx-l2-compact-corpus-v1-1df479332e50e3d1"
        ),
    )
    parser.add_argument(
        "--reference-metrics",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "crypto-forecast-l2-symmetric-alignment"
            / "crypto-forecast-okx-l2-symmetric-alignment-v1-f201b14037ec95c9"
            / "aligned_snapshot_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-features",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result, rows = build_l2_feature_table(args.corpus_root, args.reference_metrics, config)
    artifact = write_l2_feature_artifact(
        result,
        rows,
        config_sha256=file_sha256(args.config),
        feature_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_features.py"
        ),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "rows": result["table"]["rows"],
                "available_rows": result["table"]["available_rows"],
                "missing_rows": result["table"]["missing_rows"],
                "reference_mismatches": result["reference"]["mismatch_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
