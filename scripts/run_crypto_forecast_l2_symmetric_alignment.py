"""Align the pinned OKX L2 snapshots around a causal 15-minute grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402
from services.forecasting.okx_l2_symmetric_alignment import (  # noqa: E402
    align_snapshot_metrics,
    load_alignment_source,
    write_alignment_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_symmetric_alignment.json",
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
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-symmetric-alignment",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_rows, metric_fields = load_alignment_source(args.metrics, config)
    result, aligned = align_snapshot_metrics(source_rows, metric_fields, config)
    artifact = write_alignment_artifact(
        result,
        aligned,
        config_sha256=file_sha256(args.config),
        alignment_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_symmetric_alignment.py"
        ),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "grid_slots": len(aligned),
                "available_slots": result["grid"]["available_slots"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
