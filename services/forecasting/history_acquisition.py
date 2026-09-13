"""Isolated, provenance-rich acquisition of Binance Spot daily histories."""

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

ACQUISITION_SCHEMA_VERSION = "crypto-forecast-history-acquisition-v1"
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
    volume: str
    close_time_ms: int
    quote_asset_volume: str
    trades: int
    taker_buy_base_volume: str
    taker_buy_quote_volume: str

    @property
    def date(self) -> str:
        return datetime.fromtimestamp(self.open_time_ms / 1000, timezone.utc).date().isoformat()


@dataclass(frozen=True)
class AcquiredInstrument:
    spec: InstrumentSpec
    exchange_status: str
    base_asset: str
    quote_asset: str
    observations: tuple[DailyObservation, ...]


class BinancePublicMarketClient:
    """Small retrying client for public market-data GET requests only."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers={"User-Agent": "SmartFolio-Offline-Research/1.0"},
        )

    async def __aenter__(self) -> "BinancePublicMarketClient":
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
        options.update({"indent": 2})
    else:
        options.update({"separators": (",", ":")})
    return (json.dumps(value, **options) + ("\n" if pretty else "")).encode("utf-8")


def _date_to_open_ms(value: object) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    timestamp = timestamp.normalize()
    return int(timestamp.timestamp() * 1000)


def _parse_decimal(value: object, field: str) -> str:
    text = str(value)
    number = float(text)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"Invalid {field}: {value!r}")
    return text


def parse_daily_klines(rows: Sequence[Sequence[object]]) -> tuple[DailyObservation, ...]:
    """Validate normalized 1d UTC kline rows without inventing missing dates."""
    observations: list[DailyObservation] = []
    previous_open: int | None = None
    for row in rows:
        if len(row) < 11:
            raise ValueError("A Binance kline row must contain at least 11 fields")
        open_time = int(row[0])
        close_time = int(row[6])
        if open_time % DAY_MS != 0:
            raise ValueError(f"Daily kline is not aligned to UTC midnight: {open_time}")
        if close_time < open_time or close_time > open_time + DAY_MS - 1:
            raise ValueError(
                f"Daily close time is outside its UTC day for {open_time}: {close_time}"
            )
        if previous_open is not None and open_time <= previous_open:
            raise ValueError("Kline open times must be strictly increasing and unique")
        prices = {
            "open": _parse_decimal(row[1], "open"),
            "high": _parse_decimal(row[2], "high"),
            "low": _parse_decimal(row[3], "low"),
            "close": _parse_decimal(row[4], "close"),
        }
        if min(float(value) for value in prices.values()) <= 0.0:
            raise ValueError("OHLC prices must be strictly positive")
        observation = DailyObservation(
            open_time_ms=open_time,
            **prices,
            volume=_parse_decimal(row[5], "volume"),
            close_time_ms=close_time,
            quote_asset_volume=_parse_decimal(row[7], "quote asset volume"),
            trades=int(row[8]),
            taker_buy_base_volume=_parse_decimal(row[9], "taker buy base volume"),
            taker_buy_quote_volume=_parse_decimal(row[10], "taker buy quote volume"),
        )
        if observation.trades < 0:
            raise ValueError("Trade count must be non-negative")
        observations.append(observation)
        previous_open = open_time
    return tuple(observations)


async def acquire_instrument(
    client: PublicMarketClient,
    spec: InstrumentSpec,
    *,
    start_date: object,
    end_date: object,
    page_limit: int,
    request_delay_seconds: float,
) -> AcquiredInstrument:
    if not 1 <= page_limit <= 1000:
        raise ValueError("page_limit must be between 1 and 1000")
    exchange_payload = await client.get_json("/api/v3/exchangeInfo", {"symbol": spec.market_symbol})
    if not isinstance(exchange_payload, Mapping):
        raise ValueError(f"Invalid exchangeInfo response for {spec.market_symbol}")
    symbols = exchange_payload.get("symbols")
    if not isinstance(symbols, list) or len(symbols) != 1:
        raise ValueError(f"No unique exchangeInfo record for {spec.market_symbol}")
    exchange = symbols[0]
    if not isinstance(exchange, Mapping):
        raise ValueError(f"Invalid exchangeInfo symbol record for {spec.market_symbol}")
    if str(exchange.get("symbol")) != spec.market_symbol:
        raise ValueError(f"exchangeInfo returned the wrong market for {spec.market_symbol}")

    cursor = _date_to_open_ms(start_date)
    end_open = _date_to_open_ms(end_date)
    rows: list[Sequence[object]] = []
    while cursor <= end_open:
        payload = await client.get_json(
            "/api/v3/klines",
            {
                "symbol": spec.market_symbol,
                "interval": "1d",
                "startTime": cursor,
                "endTime": end_open + DAY_MS - 1,
                "limit": page_limit,
            },
        )
        if not isinstance(payload, list):
            raise ValueError(f"Invalid kline response for {spec.market_symbol}")
        if not payload:
            break
        batch = [row for row in payload if isinstance(row, Sequence)]
        parsed_batch = parse_daily_klines(batch)
        rows.extend(batch)
        next_cursor = parsed_batch[-1].open_time_ms + DAY_MS
        if next_cursor <= cursor:
            raise ValueError(f"Kline pagination did not advance for {spec.market_symbol}")
        cursor = next_cursor
        if len(payload) < page_limit:
            break
        if request_delay_seconds > 0:
            await asyncio.sleep(request_delay_seconds)

    observations = parse_daily_klines(rows)
    filtered = tuple(
        observation
        for observation in observations
        if _date_to_open_ms(start_date) <= observation.open_time_ms <= end_open
    )
    if not filtered:
        raise ValueError(f"No daily observations acquired for {spec.market_symbol}")
    return AcquiredInstrument(
        spec=spec,
        exchange_status=str(exchange.get("status", "UNKNOWN")),
        base_asset=str(exchange.get("baseAsset", "")),
        quote_asset=str(exchange.get("quoteAsset", "")),
        observations=filtered,
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
            "volume",
            "close_time_ms",
            "quote_asset_volume",
            "trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
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
                observation.volume,
                observation.close_time_ms,
                observation.quote_asset_volume,
                observation.trades,
                observation.taker_buy_base_volume,
                observation.taker_buy_quote_volume,
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
        if acquired.exchange_status != "TRADING":
            raise ValueError(f"Market is not trading: {acquired.spec.market_symbol}")
        if acquired.observations[-1].open_time_ms != end_open:
            raise ValueError(f"Latest complete day is missing for {acquired.spec.market_symbol}")
        if len(acquired.observations) < minimum_observations:
            raise ValueError(
                f"{acquired.spec.market_symbol} has {len(acquired.observations)} observations; "
                f"minimum is {minimum_observations}"
            )
        missing_days = _missing_calendar_days(acquired.observations)
        if missing_days:
            raise ValueError(
                f"{acquired.spec.market_symbol} has {len(missing_days)} missing calendar days"
            )
        price_name = f"prices/{acquired.spec.symbol}_1d.json"
        ohlcv_name = f"ohlcv/{acquired.spec.market_symbol}_1d.csv"
        irregular_close_days = [
            observation.date
            for observation in acquired.observations
            if observation.close_time_ms != observation.open_time_ms + DAY_MS - 1
        ]
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
                "observations": len(acquired.observations),
                "first_date": acquired.observations[0].date,
                "last_date": acquired.observations[-1].date,
                "missing_calendar_days": 0,
                "irregular_close_duration_days": irregular_close_days,
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
                "membership_provenance": (
                    "binance_spot_first_observation_and_current_exchange_info_snapshot"
                ),
                "eligible_for_forecasting": True,
                "exclusion_reason": None,
            }
        )
    identity = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "provider": config["provider"],
        "base_url": config["base_url"],
        "interval": config["interval"],
        "quote_asset": expected_quote,
        "start_date": config["start_date"],
        "end_date": end_date,
        "config_sha256": config_sha256,
        "acquisition_code_sha256": acquisition_code_sha256,
        "inputs": input_records,
        "universe": universe,
    }
    artifact_id = f"{ACQUISITION_SCHEMA_VERSION}-{_sha256_bytes(_json_bytes(identity))[:16]}"
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
            "public_read_only_endpoints": ["GET /api/v3/exchangeInfo", "GET /api/v3/klines"],
            "credentials_used": False,
            "raw_provider_payload_retained": False,
            "normalization": (
                "validated Binance 1d UTC klines; provider-published shortened daily candles are "
                "retained and listed; exact decimal strings retained in OHLCV CSV; "
                "legacy-compatible close series derived without interpolation"
            ),
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
    if config.get("schema_version") != ACQUISITION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported acquisition schema: {config.get('schema_version')}")
    specs = [InstrumentSpec(**item) for item in config["instruments"]]
    symbols = [item.symbol for item in specs]
    markets = [item.market_symbol for item in specs]
    if len(set(symbols)) != len(symbols) or len(set(markets)) != len(markets):
        raise ValueError("Instrument symbols and market symbols must be unique")
    quote = str(config["quote_asset"])
    if any(not item.market_symbol.endswith(quote) for item in specs):
        raise ValueError("Every market symbol must use the configured quote asset")
    return specs
