"""Isolated acquisition of confirmed OKX Spot daily candles for offline research."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import httpx
import pandas as pd

from shared.asset_groups import get_asset_group

OKX_ACQUISITION_SCHEMA_VERSION = "crypto-forecast-okx-history-v1"
DAY_MS = 86_400_000


class PublicMarketClient(Protocol):
    async def get_json(self, path: str, params: Mapping[str, object]) -> object: ...


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    market_symbol: str


@dataclass(frozen=True)
class DailyObservation:
    open_time_ms: int
    open: str
    high: str
    low: str
    close: str
    base_volume: str
    quote_volume: str
    quote_volume_native: str
    confirmed: bool

    @property
    def date(self) -> str:
        return datetime.fromtimestamp(self.open_time_ms / 1000, timezone.utc).date().isoformat()


@dataclass(frozen=True)
class AcquiredInstrument:
    spec: InstrumentSpec
    exchange_status: str
    base_asset: str
    quote_asset: str
    listing_time_ms: int
    observations: tuple[DailyObservation, ...]


class OkxPublicMarketClient:
    """Small retrying client restricted to public market-data GET requests."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers={"User-Agent": "SmartFolio-Offline-Research/1.0"},
        )

    async def __aenter__(self) -> "OkxPublicMarketClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self._client.aclose()

    async def get_json(self, path: str, params: Mapping[str, object]) -> object:
        for attempt in range(3):
            response = await self._client.get(path, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    response.raise_for_status()
                retry_after = float(response.headers.get("Retry-After", "1"))
                await asyncio.sleep(max(retry_after, 0.25) * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("Public market-data request exhausted its retries")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _date_to_open_ms(value: object) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.normalize().timestamp() * 1000)


def _parse_decimal(value: object, field: str, *, strictly_positive: bool = False) -> str:
    text = str(value)
    number = float(text)
    if not math.isfinite(number) or number < 0.0 or (strictly_positive and number <= 0.0):
        raise ValueError(f"Invalid {field}: {value!r}")
    return text


def _payload_data(payload: object, context: str) -> list[object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid OKX response for {context}")
    if str(payload.get("code")) != "0":
        raise ValueError(f"OKX returned an error for {context}: {payload.get('msg')!r}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Invalid OKX data for {context}")
    return data


def parse_daily_candles(rows: Sequence[Sequence[object]]) -> tuple[DailyObservation, ...]:
    """Validate completed OKX 1Dutc candles and return them oldest first."""
    parsed: list[DailyObservation] = []
    for row in rows:
        if len(row) < 9:
            raise ValueError("An OKX candle row must contain at least 9 fields")
        open_time = int(row[0])
        if open_time % DAY_MS != 0:
            raise ValueError(f"Daily candle is not aligned to UTC midnight: {open_time}")
        if str(row[8]) != "1":
            raise ValueError(f"Unconfirmed OKX candle is not admissible: {open_time}")
        parsed.append(
            DailyObservation(
                open_time_ms=open_time,
                open=_parse_decimal(row[1], "open", strictly_positive=True),
                high=_parse_decimal(row[2], "high", strictly_positive=True),
                low=_parse_decimal(row[3], "low", strictly_positive=True),
                close=_parse_decimal(row[4], "close", strictly_positive=True),
                base_volume=_parse_decimal(row[5], "base volume"),
                quote_volume_native=_parse_decimal(row[6], "quote volume"),
                quote_volume=_parse_decimal(row[7], "quote volume explicit"),
                confirmed=True,
            )
        )
    parsed.sort(key=lambda item: item.open_time_ms)
    timestamps = [item.open_time_ms for item in parsed]
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("Candle open times must be unique")
    return tuple(parsed)


async def acquire_instrument(
    client: PublicMarketClient,
    spec: InstrumentSpec,
    *,
    start_date: object,
    end_date: object,
    page_limit: int,
    request_delay_seconds: float,
) -> AcquiredInstrument:
    if not 1 <= page_limit <= 300:
        raise ValueError("page_limit must be between 1 and 300")
    instrument_payload = await client.get_json(
        "/api/v5/public/instruments",
        {"instType": "SPOT", "instId": spec.market_symbol},
    )
    instruments = _payload_data(instrument_payload, spec.market_symbol)
    if len(instruments) != 1 or not isinstance(instruments[0], Mapping):
        raise ValueError(f"No unique instrument record for {spec.market_symbol}")
    instrument = instruments[0]
    if str(instrument.get("instId")) != spec.market_symbol:
        raise ValueError(f"OKX returned the wrong instrument for {spec.market_symbol}")
    listing_time_ms = int(instrument.get("listTime", 0))
    if listing_time_ms <= 0:
        raise ValueError(f"Missing listing time for {spec.market_symbol}")

    requested_start = _date_to_open_ms(start_date)
    listing_day = _date_to_open_ms(pd.Timestamp(listing_time_ms, unit="ms", tz="UTC"))
    start_open = max(requested_start, listing_day)
    end_open = _date_to_open_ms(end_date)
    if start_open > end_open:
        raise ValueError(f"Requested period precedes the listing of {spec.market_symbol}")

    cursor_after = end_open + DAY_MS
    observations_by_time: dict[int, DailyObservation] = {}
    while cursor_after > start_open:
        candle_payload = await client.get_json(
            "/api/v5/market/history-candles",
            {
                "instId": spec.market_symbol,
                "bar": "1Dutc",
                "after": cursor_after,
                "limit": page_limit,
            },
        )
        rows = _payload_data(candle_payload, spec.market_symbol)
        if not rows:
            break
        sequences = [row for row in rows if isinstance(row, Sequence)]
        batch = parse_daily_candles(sequences)
        oldest = batch[0].open_time_ms
        if oldest >= cursor_after:
            raise ValueError(f"Candle pagination did not move backwards for {spec.market_symbol}")
        for observation in batch:
            if start_open <= observation.open_time_ms <= end_open:
                observations_by_time[observation.open_time_ms] = observation
        if oldest <= start_open or len(rows) < page_limit:
            break
        cursor_after = oldest
        if request_delay_seconds > 0:
            await asyncio.sleep(request_delay_seconds)

    observations = tuple(observations_by_time[key] for key in sorted(observations_by_time))
    if not observations:
        raise ValueError(f"No daily observations acquired for {spec.market_symbol}")
    return AcquiredInstrument(
        spec=spec,
        exchange_status=str(instrument.get("state", "")),
        base_asset=str(instrument.get("baseCcy", "")),
        quote_asset=str(instrument.get("quoteCcy", "")),
        listing_time_ms=listing_time_ms,
        observations=observations,
    )


def _ohlcv_csv_bytes(acquired: AcquiredInstrument) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "date",
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "base_volume",
            "quote_volume_native",
            "quote_volume",
            "confirmed",
        ]
    )
    for observation in acquired.observations:
        writer.writerow(
            [
                observation.date,
                observation.open_time_ms,
                observation.open,
                observation.high,
                observation.low,
                observation.close,
                observation.base_volume,
                observation.quote_volume_native,
                observation.quote_volume,
                "1",
            ]
        )
    return buffer.getvalue().encode("utf-8")


def _price_json_bytes(acquired: AcquiredInstrument) -> bytes:
    payload = [
        [observation.open_time_ms // 1000, float(observation.close)]
        for observation in acquired.observations
    ]
    return _json_bytes(payload)


def _missing_calendar_days(observations: Sequence[DailyObservation]) -> list[str]:
    if len(observations) < 2:
        return []
    observed = {observation.open_time_ms for observation in observations}
    missing = []
    cursor = observations[0].open_time_ms
    final = observations[-1].open_time_ms
    while cursor <= final:
        if cursor not in observed:
            missing.append(datetime.fromtimestamp(cursor / 1000, timezone.utc).date().isoformat())
        cursor += DAY_MS
    return missing


def write_acquisition_artifact(
    acquired_instruments: Sequence[AcquiredInstrument],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    acquisition_code_sha256: str,
    end_date: str,
    output_root: str | Path,
) -> Path:
    if not acquired_instruments:
        raise ValueError("At least one acquired instrument is required")
    end_open = _date_to_open_ms(end_date)
    files: dict[str, bytes] = {}
    input_records: list[dict[str, Any]] = []
    universe: list[dict[str, Any]] = []
    minimum_observations = int(config["minimum_observations"])
    expected_quote = str(config["quote_asset"])
    for acquired in sorted(acquired_instruments, key=lambda item: item.spec.symbol):
        if acquired.base_asset != acquired.spec.symbol:
            raise ValueError(f"Unexpected base asset for {acquired.spec.market_symbol}")
        if acquired.quote_asset != expected_quote:
            raise ValueError(f"Unexpected quote asset for {acquired.spec.market_symbol}")
        if acquired.exchange_status != "live":
            raise ValueError(f"Market is not live: {acquired.spec.market_symbol}")
        if acquired.observations[-1].open_time_ms != end_open:
            raise ValueError(f"Latest complete day is missing for {acquired.spec.market_symbol}")
        if len(acquired.observations) < minimum_observations:
            raise ValueError(
                f"{acquired.spec.market_symbol} has {len(acquired.observations)} observations; "
                f"minimum is {minimum_observations}"
            )
        missing_days = _missing_calendar_days(acquired.observations)
        price_name = f"prices/{acquired.spec.symbol}_1d.json"
        ohlcv_name = f"ohlcv/{acquired.spec.market_symbol}_1d.csv"
        price_payload = _price_json_bytes(acquired)
        ohlcv_payload = _ohlcv_csv_bytes(acquired)
        files[price_name] = price_payload
        files[ohlcv_name] = ohlcv_payload
        input_records.append(
            {
                "symbol": acquired.spec.symbol,
                "market_symbol": acquired.spec.market_symbol,
                "base_asset": acquired.base_asset,
                "quote_asset": acquired.quote_asset,
                "exchange_status_at_acquisition": acquired.exchange_status,
                "listing_time_ms": acquired.listing_time_ms,
                "listing_time_utc": datetime.fromtimestamp(
                    acquired.listing_time_ms / 1000, timezone.utc
                ).isoformat(),
                "observations": len(acquired.observations),
                "first_date": acquired.observations[0].date,
                "last_date": acquired.observations[-1].date,
                "missing_calendar_days": missing_days,
                "price_file": price_name,
                "price_file_sha256": _sha256_bytes(price_payload),
                "ohlcv_file": ohlcv_name,
                "ohlcv_file_sha256": _sha256_bytes(ohlcv_payload),
                "provider_provenance": str(config["provider"]),
            }
        )
        universe.append(
            {
                "symbol": acquired.spec.symbol,
                "group": get_asset_group(acquired.spec.symbol),
                "known_from": acquired.observations[0].date,
                "known_until": None,
                "delisted": False,
                "membership_provenance": "okx_spot_listing_time_and_first_observation",
                "eligible_for_cross_exchange_validation": True,
                "exclusion_reason": None,
            }
        )
    identity = {
        "schema_version": OKX_ACQUISITION_SCHEMA_VERSION,
        "provider": config["provider"],
        "base_url": config["base_url"],
        "bar": config["bar"],
        "quote_asset": expected_quote,
        "start_date": config["start_date"],
        "end_date": end_date,
        "config_sha256": config_sha256,
        "acquisition_code_sha256": acquisition_code_sha256,
        "inputs": input_records,
        "universe": universe,
    }
    artifact_id = f"{OKX_ACQUISITION_SCHEMA_VERSION}-{_sha256_bytes(_json_bytes(identity))[:16]}"
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    final_directory = root / artifact_id
    if final_directory.exists():
        raise FileExistsError(f"Acquisition artifact already exists: {final_directory}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=root) as temporary:
        temporary_directory = Path(temporary)
        for relative_name, payload in files.items():
            destination = temporary_directory / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        (temporary_directory / "universe.json").write_bytes(_json_bytes(universe, pretty=True))
        manifest = {
            **identity,
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "public_read_only_endpoints": [
                "GET /api/v5/public/instruments",
                "GET /api/v5/market/history-candles",
            ],
            "credentials_used": False,
            "raw_provider_payload_retained": False,
            "normalization": (
                "confirmed OKX 1Dutc candles; exact decimal strings retained; "
                "rows sorted oldest first; missing calendar days reported without filling"
            ),
            "trade_count_availability": "unavailable_in_okx_daily_candle_response",
            "quote_asset_limitation": (
                "Prices are quoted in USDT; equivalence with risk-free USD is not assumed or proven"
            ),
        }
        (temporary_directory / "acquisition_manifest.json").write_bytes(
            _json_bytes(manifest, pretty=True)
        )
        Path(temporary).replace(final_directory)
    return final_directory


def load_instrument_specs(config: Mapping[str, Any]) -> list[InstrumentSpec]:
    if config.get("schema_version") != OKX_ACQUISITION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported acquisition schema: {config.get('schema_version')}")
    if config.get("bar") != "1Dutc":
        raise ValueError("Only the 1Dutc OKX bar is supported")
    specs = [InstrumentSpec(**item) for item in config["instruments"]]
    symbols = [item.symbol for item in specs]
    markets = [item.market_symbol for item in specs]
    if len(set(symbols)) != len(symbols) or len(set(markets)) != len(markets):
        raise ValueError("Instrument symbols and market symbols must be unique")
    quote = str(config["quote_asset"])
    if any(not item.market_symbol.endswith(f"-{quote}") for item in specs):
        raise ValueError("Every market symbol must use the configured quote asset")
    return specs
