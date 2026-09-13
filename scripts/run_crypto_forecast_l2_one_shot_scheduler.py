"""Wait for one UTC grid slot, capture three OKX books, then stop."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_l2_collection_feasibility import OkxPublicBookClient  # noqa: E402
from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402
from services.forecasting.okx_l2_scheduler import (  # noqa: E402
    ReplayBookClient,
    SingleInstanceLock,
    load_scheduler_replay,
    next_slot_ms,
    validate_scheduler_config,
    write_scheduler_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_scheduler.json",
    )
    parser.add_argument("--replay-artifact", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-one-shot-scheduler",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> Path:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_scheduler_config(config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / str(config["lock_filename"])
    with SingleInstanceLock(lock_path):
        if args.replay_artifact:
            slot_timestamp_ms, execution_started_ms, probes = load_scheduler_replay(
                args.replay_artifact
            )
            client = ReplayBookClient(probes)
            return await write_scheduler_artifact(
                client,
                slot_timestamp_ms=slot_timestamp_ms,
                execution_started_ms=execution_started_ms,
                config=config,
                config_sha256=file_sha256(args.config),
                scheduler_code_sha256=file_sha256(
                    PROJECT_ROOT / "services" / "forecasting" / "okx_l2_scheduler.py"
                ),
                collector_code_sha256=file_sha256(
                    PROJECT_ROOT / "services" / "forecasting" / "okx_l2_prospective_collector.py"
                ),
                output_root=args.output_root,
            )

        now_ms = time.time_ns() // 1_000_000
        slot_timestamp_ms = next_slot_ms(now_ms, int(config["grid_interval_ms"]))
        print(
            json.dumps(
                {
                    "waiting_for_slot_utc": datetime.fromtimestamp(
                        slot_timestamp_ms / 1000, timezone.utc
                    ).isoformat(),
                    "maximum_lag_seconds": int(config["scheduled_capture_tolerance_ms"]) / 1000,
                }
            ),
            flush=True,
        )
        while True:
            now_ms = time.time_ns() // 1_000_000
            if now_ms >= slot_timestamp_ms:
                break
            await asyncio.sleep(
                min(float(config["wait_poll_seconds"]), (slot_timestamp_ms - now_ms) / 1000)
            )
        execution_started_ms = time.time_ns() // 1_000_000
        async with OkxPublicBookClient(
            str(config["base_url"]), timeout_seconds=float(config["request_timeout_seconds"])
        ) as client:
            return await write_scheduler_artifact(
                client,
                slot_timestamp_ms=slot_timestamp_ms,
                execution_started_ms=execution_started_ms,
                config=config,
                config_sha256=file_sha256(args.config),
                scheduler_code_sha256=file_sha256(
                    PROJECT_ROOT / "services" / "forecasting" / "okx_l2_scheduler.py"
                ),
                collector_code_sha256=file_sha256(
                    PROJECT_ROOT / "services" / "forecasting" / "okx_l2_prospective_collector.py"
                ),
                output_root=args.output_root,
            )


def main() -> int:
    artifact = asyncio.run(run(parse_args()))
    result = json.loads((artifact / "run_result.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "status": result["run"]["status"],
                "start_lag_ms": result["run"]["start_lag_ms"],
                "captures": len(result["run"]["captures"]),
                "long_running_collection_started": result["long_running_collection_started"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
