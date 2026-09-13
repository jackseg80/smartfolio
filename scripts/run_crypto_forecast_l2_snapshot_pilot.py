"""Run the pre-registered OKX L2 snapshot-only pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402
from services.forecasting.okx_l2_snapshot_pilot import (  # noqa: E402
    analyze_snapshot_archive,
    write_snapshot_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_snapshot_pilot.json",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "crypto-forecast-l2-pilot"
            / "raw"
            / "SOL-USDT-L2orderbook-400lv-2026-09-10.tar.gz"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-snapshot-pilot",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result, snapshots = analyze_snapshot_archive(args.archive, config)
    artifact = write_snapshot_artifact(
        result,
        snapshots,
        config_sha256=file_sha256(args.config),
        analysis_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_snapshot_pilot.py"
        ),
        output_root=args.output_root,
    )
    print(
        json.dumps({"artifact": str(artifact.resolve()), "decision": result["decision"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
