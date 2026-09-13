"""Acquire checksum-verified Binance USD-M monthly funding archives."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.binance_funding_acquisition import (  # noqa: E402
    InstrumentSpec,
    ValidatedArchive,
    iter_months,
    load_instrument_specs,
    validate_archive,
    write_acquisition_artifact,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_bundle_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative_name = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(relative_name)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_binance_funding_history.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-binance-funding-history",
    )
    return parser.parse_args()


async def _get_bytes(
    client: httpx.AsyncClient, url: str, attempts: int, maximum_bytes: int
) -> bytes:
    for attempt in range(attempts):
        try:
            async with client.stream("GET", url) as response:
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == attempts - 1:
                        response.raise_for_status()
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > maximum_bytes:
                    raise ValueError(f"Public response exceeds the frozen byte limit: {url}")
                chunks = []
                received = 0
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    if received > maximum_bytes:
                        raise ValueError(f"Public response exceeds the frozen byte limit: {url}")
                    chunks.append(chunk)
                return b"".join(chunks)
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError("Public download exhausted its retries")


async def _acquire_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    base_url: str,
    spec: InstrumentSpec,
    month: str,
    config: dict[str, object],
) -> ValidatedArchive:
    filename = f"{spec.market_symbol}-fundingRate-{month}.zip"
    url = f"{base_url.rstrip('/')}/{spec.market_symbol}/{filename}"
    async with semaphore:
        archive_payload, checksum_payload = await asyncio.gather(
            _get_bytes(
                client,
                url,
                int(config["retry_attempts"]),
                int(config["maximum_archive_bytes"]),
            ),
            _get_bytes(client, f"{url}.CHECKSUM", int(config["retry_attempts"]), 4096),
        )
    return validate_archive(
        spec,
        month,
        archive_payload,
        checksum_payload,
        maximum_archive_bytes=int(config["maximum_archive_bytes"]),
        maximum_uncompressed_bytes=int(config["maximum_uncompressed_bytes"]),
    )


async def run(args: argparse.Namespace) -> Path:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    specs = load_instrument_specs(config)
    months = iter_months(str(config["first_month"]), str(config["last_month"]))
    semaphore = asyncio.Semaphore(int(config["concurrency"]))
    async with httpx.AsyncClient(
        timeout=float(config["timeout_seconds"]),
        headers={"User-Agent": "SmartFolio-Offline-Research/1.0"},
        follow_redirects=True,
    ) as client:
        archives = await asyncio.gather(
            *(
                _acquire_one(client, semaphore, str(config["base_url"]), spec, month, config)
                for spec in specs
                for month in months
            )
        )
    for spec in specs:
        selected = [item for item in archives if item.spec == spec]
        print(
            f"{spec.market_symbol}: {len(selected)} archives, "
            f"{sum(len(item.observations) for item in selected)} observations, "
            f"{sum(item.archive_bytes for item in selected)} compressed bytes"
        )
    return write_acquisition_artifact(
        archives,
        config=config,
        config_sha256=file_sha256(args.config),
        acquisition_code_sha256=code_bundle_sha256(
            [
                PROJECT_ROOT / "services" / "forecasting" / "binance_funding_acquisition.py",
                PROJECT_ROOT / "scripts" / "acquire_crypto_forecast_binance_funding.py",
            ]
        ),
        output_root=args.output_root,
    )


def main() -> int:
    args = parse_args()
    artifact = asyncio.run(run(args))
    print(json.dumps({"artifact": str(artifact.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
