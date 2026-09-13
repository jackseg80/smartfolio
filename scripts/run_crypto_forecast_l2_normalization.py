"""Normalize the pinned OKX L2 snapshot sample onto a causal 15-minute grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_l2_normalization import (  # noqa: E402
    load_snapshot_metrics,
    normalize_snapshot_metrics,
    write_normalization_artifact,
)
from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_normalization.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "crypto-forecast-l2-sample"
            / "analysis-final"
            / "crypto-forecast-okx-l2-sample-v1-973d02b919af2d1c"
            / "snapshot_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-normalization",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_rows, metric_fields = load_snapshot_metrics(args.metrics, config)
    result, normalized = normalize_snapshot_metrics(source_rows, metric_fields, config)
    artifact = write_normalization_artifact(
        result,
        normalized,
        config_sha256=file_sha256(args.config),
        normalization_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_normalization.py"
        ),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "grid_slots": len(normalized),
                "available_slots": result["grid"]["available_slots"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
