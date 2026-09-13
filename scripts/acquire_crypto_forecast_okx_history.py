"""Acquire an isolated, provenance-rich OKX Spot daily history artifact."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_history_acquisition import (  # noqa: E402
    OkxPublicMarketClient,
    acquire_instrument,
    load_instrument_specs,
    write_acquisition_artifact,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_history.json",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Last fully closed UTC day to acquire (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-okx-history",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> Path:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    specs = load_instrument_specs(config)
    acquired = []
    async with OkxPublicMarketClient(str(config["base_url"])) as client:
        for spec in specs:
            instrument = await acquire_instrument(
                client,
                spec,
                start_date=config["start_date"],
                end_date=args.end_date,
                page_limit=int(config["page_limit"]),
                request_delay_seconds=float(config["request_delay_seconds"]),
            )
            acquired.append(instrument)
            print(
                f"{spec.symbol}: {len(instrument.observations)} observations "
                f"({instrument.observations[0].date} to {instrument.observations[-1].date})"
            )
    return write_acquisition_artifact(
        acquired,
        config=config,
        config_sha256=file_sha256(args.config),
        acquisition_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_history_acquisition.py"
        ),
        end_date=args.end_date,
        output_root=args.output_root,
    )


def main() -> int:
    args = parse_args()
    artifact = asyncio.run(run(args))
    print(json.dumps({"artifact": str(artifact.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
