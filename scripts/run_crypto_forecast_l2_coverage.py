"""Collect OKX L2 archive metadata only and estimate bounded coverage costs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_history_acquisition import OkxPublicMarketClient  # noqa: E402
from services.forecasting.okx_l2_coverage import (  # noqa: E402
    build_coverage_result,
    collect_coverage,
    file_sha256,
    write_coverage_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_coverage.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-coverage",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> Path:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    async with OkxPublicMarketClient(str(config["base_url"])) as client:
        records, missing = await collect_coverage(client, config)
    result = build_coverage_result(records, missing, config=config)
    return write_coverage_artifact(
        result,
        records,
        config_sha256=file_sha256(args.config),
        acquisition_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_coverage.py"
        ),
        output_root=args.output_root,
    )


def main() -> int:
    args = parse_args()
    artifact = asyncio.run(run(args))
    result = json.loads((artifact / "coverage_result.json").read_text(encoding="utf-8"))
    print(
        json.dumps({"artifact": str(artifact.resolve()), "decision": result["decision"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
