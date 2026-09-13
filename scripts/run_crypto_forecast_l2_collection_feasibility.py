"""Measure or replay the bounded OKX L2 collection feasibility probe."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_l2_collection_feasibility import (  # noqa: E402
    OkxPublicBookClient,
    build_feasibility_result,
    capture_book_probes,
    load_book_probes,
    write_feasibility_artifact,
)
from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_collection_feasibility.json",
    )
    parser.add_argument("--probe-artifact", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-collection-feasibility",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> Path:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.probe_artifact:
        probes = load_book_probes(args.probe_artifact)
    else:
        async with OkxPublicBookClient(
            str(config["base_url"]), timeout_seconds=float(config["request_timeout_seconds"])
        ) as client:
            probes = await capture_book_probes(client, config)
    result = build_feasibility_result(probes, config)
    return write_feasibility_artifact(
        result,
        probes,
        config_sha256=file_sha256(args.config),
        feasibility_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_collection_feasibility.py"
        ),
        output_root=args.output_root,
    )


def main() -> int:
    artifact = asyncio.run(run(parse_args()))
    result = json.loads((artifact / "feasibility_result.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "historical_decision": result["historical_archives"]["decision"],
                "projected_retained_gib": result["prospective_collection"][
                    "projected_retained_gib_with_safety"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
