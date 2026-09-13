"""Bounded feasibility probe for a prospective OKX L2 snapshot collection."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import httpx

from services.forecasting.okx_l2_pilot import file_sha256

OKX_L2_COLLECTION_FEASIBILITY_SCHEMA_VERSION = "crypto-forecast-okx-l2-collection-feasibility-v1"
BYTES_PER_GIB = Decimal(1024**3)
BYTES_PER_MB = Decimal(1_000_000)


@dataclass(frozen=True)
class BookProbe:
    instrument: str
    requested_at_ms: int
    completed_at_ms: int
    response_body: bytes


class PublicBookClient(Protocol):
    async def get_book(self, endpoint: str, instrument: str, depth: int) -> BookProbe: ...


class OkxPublicBookClient:
    """Public GET-only client that explicitly requests an uncompressed response."""

    def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": "SmartFolio-Offline-Research/1.0",
            },
        )

    async def __aenter__(self) -> "OkxPublicBookClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self._client.aclose()

    async def get_book(self, endpoint: str, instrument: str, depth: int) -> BookProbe:
        for attempt in range(3):
            requested_at_ms = time.time_ns() // 1_000_000
            response = await self._client.get(
                endpoint, params={"instId": instrument, "sz": str(depth)}
            )
            completed_at_ms = time.time_ns() // 1_000_000
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    response.raise_for_status()
                retry_after = float(response.headers.get("Retry-After", "1"))
                await asyncio.sleep(max(retry_after, 0.25) * (attempt + 1))
                continue
            response.raise_for_status()
            if response.headers.get("Content-Encoding", "identity").lower() not in {
                "",
                "identity",
            }:
                raise ValueError("OKX did not honor the identity content encoding")
            return BookProbe(
                instrument=instrument,
                requested_at_ms=requested_at_ms,
                completed_at_ms=completed_at_ms,
                response_body=response.content,
            )
        raise RuntimeError("Public order-book request exhausted its retries")


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + ("\n" if pretty else "")).encode("utf-8")


def _positive_decimal(value: object, field: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"Invalid {field}: {value!r}")
    return number


def _validate_side(
    rows: object,
    *,
    side: str,
    minimum_levels: int,
    maximum_levels: int,
) -> list[Decimal]:
    if not isinstance(rows, list) or not minimum_levels <= len(rows) <= maximum_levels:
        raise ValueError(f"Unexpected {side} level count")
    prices: list[Decimal] = []
    for index, level in enumerate(rows):
        if not isinstance(level, list) or len(level) < 2:
            raise ValueError(f"Invalid {side} level at index {index}")
        prices.append(_positive_decimal(level[0], f"{side} price"))
        _positive_decimal(level[1], f"{side} quantity")
        if len(level) >= 4:
            order_count = _positive_decimal(level[3], f"{side} order count")
            if order_count != order_count.to_integral_value():
                raise ValueError(f"Invalid {side} order count at index {index}")
    expected = sorted(prices, reverse=side == "bids")
    if prices != expected or len(set(prices)) != len(prices):
        raise ValueError(f"{side} prices are not strictly ordered and unique")
    return prices


def validate_book_probe(probe: BookProbe, config: Mapping[str, Any]) -> dict[str, Any]:
    if len(probe.response_body) > int(config["maximum_response_bytes_per_instrument"]):
        raise ValueError(f"Response is too large for {probe.instrument}")
    try:
        payload = json.loads(probe.response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON response for {probe.instrument}") from exc
    if not isinstance(payload, Mapping) or str(payload.get("code")) != "0":
        raise ValueError(f"OKX returned an error for {probe.instrument}")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], Mapping):
        raise ValueError(f"Expected one order book for {probe.instrument}")
    book = data[0]
    minimum_levels = int(config["minimum_levels_per_side"])
    maximum_levels = int(config["maximum_levels_per_side"])
    bids = _validate_side(
        book.get("bids"),
        side="bids",
        minimum_levels=minimum_levels,
        maximum_levels=maximum_levels,
    )
    asks = _validate_side(
        book.get("asks"),
        side="asks",
        minimum_levels=minimum_levels,
        maximum_levels=maximum_levels,
    )
    if bids[0] >= asks[0]:
        raise ValueError(f"Crossed order book for {probe.instrument}")
    timestamp_ms = int(str(book.get("ts")))
    age_ms = probe.completed_at_ms - timestamp_ms
    if age_ms > int(config["maximum_snapshot_age_ms"]):
        raise ValueError(f"Stale order book for {probe.instrument}")
    if timestamp_ms - probe.completed_at_ms > int(config["maximum_future_lead_ms"]):
        raise ValueError(f"Future-dated order book for {probe.instrument}")
    gzip_bytes = len(gzip.compress(probe.response_body, compresslevel=9, mtime=0))
    return {
        "instrument": probe.instrument,
        "requested_at_ms": probe.requested_at_ms,
        "completed_at_ms": probe.completed_at_ms,
        "provider_timestamp_ms": timestamp_ms,
        "snapshot_age_ms_at_completion": age_ms,
        "response_bytes": len(probe.response_body),
        "gzip_bytes": gzip_bytes,
        "response_sha256": hashlib.sha256(probe.response_body).hexdigest(),
        "bid_levels": len(bids),
        "ask_levels": len(asks),
        "best_bid": str(bids[0]),
        "best_ask": str(asks[0]),
        "sequence_id": book.get("seqId"),
        "valid": True,
    }


async def capture_book_probes(
    client: PublicBookClient, config: Mapping[str, Any]
) -> list[BookProbe]:
    if config.get("schema_version") != OKX_L2_COLLECTION_FEASIBILITY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported feasibility schema: {config.get('schema_version')}")
    instruments = [str(value) for value in config["instruments"]]
    if len(set(instruments)) != len(instruments):
        raise ValueError("Probe instruments must be unique")
    probes: list[BookProbe] = []
    for index, instrument in enumerate(instruments):
        probes.append(
            await client.get_book(
                str(config["endpoint"]), instrument, int(config["book_depth_per_side"])
            )
        )
        if index + 1 < len(instruments) and float(config["request_delay_seconds"]) > 0:
            await asyncio.sleep(float(config["request_delay_seconds"]))
    return probes


def _gib(byte_count: Decimal | int) -> float:
    return float(Decimal(byte_count) / BYTES_PER_GIB)


def build_feasibility_result(
    probes: Sequence[BookProbe], config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("schema_version") != OKX_L2_COLLECTION_FEASIBILITY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported feasibility schema: {config.get('schema_version')}")
    instruments = [str(value) for value in config["instruments"]]
    if sorted(probe.instrument for probe in probes) != sorted(instruments):
        raise ValueError("Probe set does not match the frozen instruments")
    design = {str(key): int(value) for key, value in config["chronological_design_days"].items()}
    minimum_days = int(config["minimum_continuous_days"])
    maximum_horizon = int(config["maximum_target_horizon_days"])
    if sum(design.values()) != minimum_days:
        raise ValueError("Chronological design does not sum to the frozen continuous window")
    purge_days = [value for key, value in design.items() if key.startswith("purge_")]
    if len(purge_days) != 3 or any(value < maximum_horizon for value in purge_days):
        raise ValueError("Every chronological boundary needs a full target-horizon purge")

    probe_results = [validate_book_probe(probe, config) for probe in probes]
    daily_sizes = {
        str(key): _positive_decimal(value, f"historical {key} size")
        for key, value in config["historical_complete_day_size_mb"].items()
    }
    historical_projection = {
        key: {
            "projected_mb": float(value * minimum_days),
            "projected_gib": _gib(value * minimum_days * BYTES_PER_MB),
        }
        for key, value in daily_sizes.items()
    }
    historical_limit = _positive_decimal(
        config["maximum_local_archive_history_gib"], "historical storage limit"
    )
    historical_feasible = all(
        Decimal(str(item["projected_gib"])) <= historical_limit
        for item in historical_projection.values()
    )

    snapshots_per_day = int(config["snapshots_per_day_per_instrument"])
    calls_per_day = snapshots_per_day * len(instruments)
    raw_cycle_bytes = sum(int(item["response_bytes"]) for item in probe_results)
    gzip_cycle_bytes = sum(int(item["gzip_bytes"]) for item in probe_results)
    raw_projection_bytes = raw_cycle_bytes * snapshots_per_day * minimum_days
    gzip_projection_bytes = gzip_cycle_bytes * snapshots_per_day * minimum_days
    safety_multiplier = _positive_decimal(
        config["gzip_storage_safety_multiplier"], "gzip storage safety multiplier"
    )
    retained_projection_bytes = Decimal(gzip_projection_bytes) * safety_multiplier
    raw_limit = _positive_decimal(
        config["maximum_projected_raw_collection_gib"], "raw collection limit"
    )
    retained_limit = _positive_decimal(
        config["maximum_projected_retained_collection_gib"], "retained collection limit"
    )
    rate_limit_ok = int(config["maximum_planned_burst_requests"]) <= int(
        config["official_rate_limit_requests"]
    )
    prospective_feasible = (
        len(probe_results) == len(instruments)
        and calls_per_day == 288
        and _gib(raw_projection_bytes) <= float(raw_limit)
        and _gib(retained_projection_bytes) <= float(retained_limit)
        and rate_limit_ok
    )
    return {
        "schema_version": OKX_L2_COLLECTION_FEASIBILITY_SCHEMA_VERSION,
        "provider": config["provider"],
        "minimum_history": {
            "continuous_days": minimum_days,
            "maximum_target_horizon_days": maximum_horizon,
            "chronological_design_days": design,
            "interpretation": "Engineering floor only; statistical sufficiency is not proven",
        },
        "historical_archives": {
            "source": "Lot 5D frozen complete-day size distribution for BTC, ETH and SOL",
            "projection": historical_projection,
            "maximum_local_gib": float(historical_limit),
            "decision": (
                "GO_LOCAL_ARCHIVE_HISTORY" if historical_feasible else "NO_GO_LOCAL_ARCHIVE_HISTORY"
            ),
            "downloads_performed": False,
        },
        "prospective_collection": {
            "endpoint": config["endpoint"],
            "book_depth_per_side": int(config["book_depth_per_side"]),
            "snapshots_per_day_per_instrument": snapshots_per_day,
            "calls_per_day": calls_per_day,
            "total_calls": calls_per_day * minimum_days,
            "raw_bytes_per_three_instrument_cycle": raw_cycle_bytes,
            "gzip_bytes_per_three_instrument_cycle": gzip_cycle_bytes,
            "projected_raw_gib": _gib(raw_projection_bytes),
            "projected_gzip_gib": _gib(gzip_projection_bytes),
            "gzip_storage_safety_multiplier": float(safety_multiplier),
            "projected_retained_gib_with_safety": _gib(retained_projection_bytes),
            "maximum_projected_raw_gib": float(raw_limit),
            "maximum_projected_retained_gib": float(retained_limit),
            "planned_burst_requests": int(config["maximum_planned_burst_requests"]),
            "official_rate_limit_requests": int(config["official_rate_limit_requests"]),
            "official_rate_limit_window_seconds": int(config["official_rate_limit_window_seconds"]),
            "rate_limit_ok": rate_limit_ok,
            "probes": probe_results,
            "decision": (
                "GO_PROSPECTIVE_COLLECTION_FEASIBILITY"
                if prospective_feasible
                else "NO_GO_PROSPECTIVE_COLLECTION_FEASIBILITY"
            ),
        },
        "decision": (
            "GO_PROSPECTIVE_COLLECTION_FEASIBILITY"
            if prospective_feasible
            else "NO_GO_COLLECTION_FEASIBILITY"
        ),
        "credentials_used": False,
        "orders_placed": False,
        "model_trained": False,
        "production_touched": False,
        "limitations": [
            "Three live public responses measure one point in time only",
            "Projected sizes assume the measured payload shape remains representative",
            "No long-running collector was started",
            "No predictive or economic validity was tested",
        ],
    }


def write_feasibility_artifact(
    result: Mapping[str, Any],
    probes: Sequence[BookProbe],
    *,
    config_sha256: str,
    feasibility_code_sha256: str,
    output_root: Path,
) -> Path:
    raw_files = {
        f"probes/{probe.instrument.lower()}.json": probe.response_body
        for probe in sorted(probes, key=lambda item: item.instrument)
    }
    probe_metadata = [
        {
            "instrument": probe.instrument,
            "requested_at_ms": probe.requested_at_ms,
            "completed_at_ms": probe.completed_at_ms,
            "file": f"probes/{probe.instrument.lower()}.json",
            "sha256": hashlib.sha256(probe.response_body).hexdigest(),
        }
        for probe in sorted(probes, key=lambda item: item.instrument)
    ]
    identity = {
        **result,
        "config_sha256": config_sha256,
        "feasibility_code_sha256": feasibility_code_sha256,
        "probe_metadata": probe_metadata,
    }
    artifact_id = (
        f"{OKX_L2_COLLECTION_FEASIBILITY_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Feasibility artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        for relative_name, payload in raw_files.items():
            path = temporary_path / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        (temporary_path / "probe_metadata.json").write_bytes(
            _json_bytes(probe_metadata, pretty=True)
        )
        (temporary_path / "feasibility_result.json").write_bytes(_json_bytes(identity, pretty=True))
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "public_read_only_endpoint": f"GET {result['prospective_collection']['endpoint']}",
            "credentials_used": False,
            "orders_placed": False,
            "archive_downloads_performed": False,
            "long_running_collection_started": False,
            "model_trained": False,
            "production_touched": False,
            "result_file": "feasibility_result.json",
            "result_sha256": file_sha256(temporary_path / "feasibility_result.json"),
            "probe_metadata_file": "probe_metadata.json",
            "probe_metadata_sha256": file_sha256(temporary_path / "probe_metadata.json"),
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination


def load_book_probes(artifact: Path) -> list[BookProbe]:
    metadata = json.loads((artifact / "probe_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise ValueError("Invalid probe metadata")
    probes: list[BookProbe] = []
    for item in metadata:
        if not isinstance(item, Mapping):
            raise ValueError("Invalid probe metadata item")
        relative = Path(str(item["file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe probe path")
        body = (artifact / relative).read_bytes()
        if hashlib.sha256(body).hexdigest() != str(item["sha256"]):
            raise ValueError("Probe hash mismatch")
        probes.append(
            BookProbe(
                instrument=str(item["instrument"]),
                requested_at_ms=int(item["requested_at_ms"]),
                completed_at_ms=int(item["completed_at_ms"]),
                response_body=body,
            )
        )
    return probes
