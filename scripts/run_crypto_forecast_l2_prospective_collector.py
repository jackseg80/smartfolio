"""Run one bounded OKX L2 collector pilot or replay its stored captures."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_l2_collection_feasibility import (  # noqa: E402
    OkxPublicBookClient,
)
from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402
from services.forecasting.okx_l2_prospective_collector import (  # noqa: E402
    capture_pilot_probes,
    load_pilot_probes,
    write_pilot_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_prospective_collector.json",
    )
    parser.add_argument("--replay-artifact", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-prospective-collector",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> Path:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.replay_artifact:
        slot_timestamp_ms, probes = load_pilot_probes(args.replay_artifact)
    else:
        now_ms = time.time_ns() // 1_000_000
        interval_ms = int(config["grid_interval_ms"])
        slot_timestamp_ms = now_ms - now_ms % interval_ms
        async with OkxPublicBookClient(
            str(config["base_url"]), timeout_seconds=float(config["request_timeout_seconds"])
        ) as client:
            probes = await capture_pilot_probes(client, config)
    return write_pilot_artifact(
        probes,
        slot_timestamp_ms=slot_timestamp_ms,
        config=config,
        config_sha256=file_sha256(args.config),
        collector_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_prospective_collector.py"
        ),
        output_root=args.output_root,
    )


def main() -> int:
    artifact = asyncio.run(run(parse_args()))
    result = json.loads((artifact / "pilot_result.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "captures": len(result["daily_manifest"]["captures"]),
                "missing_due": len(result["daily_manifest"]["missing_due_captures"]),
                "long_running_collection_started": result["long_running_collection_started"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
