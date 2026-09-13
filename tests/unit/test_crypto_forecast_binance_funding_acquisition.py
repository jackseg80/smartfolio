import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import httpx

from services.forecasting.binance_funding_acquisition import (
    FundingObservation,
    InstrumentSpec,
    ValidatedArchive,
    iter_months,
    parse_official_checksum,
    validate_archive,
    validate_history,
    write_acquisition_artifact,
)
from scripts.acquire_crypto_forecast_binance_funding import _get_bytes


def _zip_payload(filename: str, rows: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            filename,
            "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(rows) + "\n",
        )
    return buffer.getvalue()


def _timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


def test_iter_months_and_checksum_parser_are_strict():
    assert iter_months("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    filename = "BTCUSDT-fundingRate-2022-06.zip"
    assert parse_official_checksum(f"{'a' * 64}  {filename}\n".encode(), filename) == "a" * 64
    with pytest.raises(ValueError, match="wrong archive"):
        parse_official_checksum(f"{'a' * 64}  other.zip\n".encode(), filename)


def test_validate_archive_verifies_checksum_schema_and_month():
    spec = InstrumentSpec(symbol="BTC", market_symbol="BTCUSDT")
    csv_name = "BTCUSDT-fundingRate-2022-06.csv"
    payload = _zip_payload(
        csv_name,
        [
            f"{_timestamp('2022-06-01T00:00:00')},8,0.00010000",
            f"{_timestamp('2022-06-01T08:00:00') + 11},8,-0.00002000",
        ],
    )
    digest = hashlib.sha256(payload).hexdigest()

    validated = validate_archive(
        spec,
        "2022-06",
        payload,
        f"{digest}  BTCUSDT-fundingRate-2022-06.zip\n".encode(),
        maximum_archive_bytes=65_536,
        maximum_uncompressed_bytes=1_048_576,
    )

    assert len(validated.observations) == 2
    assert validated.observations[1].funding_rate == "-0.00002000"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_archive(
            spec,
            "2022-06",
            payload,
            f"{'0' * 64}  BTCUSDT-fundingRate-2022-06.zip\n".encode(),
            maximum_archive_bytes=65_536,
            maximum_uncompressed_bytes=1_048_576,
        )


def test_validate_archive_rejects_duplicate_and_unsafe_members():
    spec = InstrumentSpec(symbol="BTC", market_symbol="BTCUSDT")
    timestamp = _timestamp("2022-06-01T00:00:00")
    duplicate = _zip_payload(
        "BTCUSDT-fundingRate-2022-06.csv",
        [f"{timestamp},8,0.0001", f"{timestamp},8,0.0002"],
    )
    digest = hashlib.sha256(duplicate).hexdigest()
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_archive(
            spec,
            "2022-06",
            duplicate,
            f"{digest}  BTCUSDT-fundingRate-2022-06.zip".encode(),
            maximum_archive_bytes=65_536,
            maximum_uncompressed_bytes=1_048_576,
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../BTCUSDT-fundingRate-2022-06.csv", "bad")
    unsafe = buffer.getvalue()
    digest = hashlib.sha256(unsafe).hexdigest()
    with pytest.raises(ValueError, match="unsafe path"):
        validate_archive(
            spec,
            "2022-06",
            unsafe,
            f"{digest}  BTCUSDT-fundingRate-2022-06.zip".encode(),
            maximum_archive_bytes=65_536,
            maximum_uncompressed_bytes=1_048_576,
        )


def _small_config() -> dict[str, object]:
    instruments = [
        {"symbol": symbol, "market_symbol": f"{symbol}USDT"}
        for symbol in ("BTC", "ETH", "SOL", "ADA", "XRP")
    ]
    return {
        "schema_version": "crypto-forecast-binance-funding-history-v1",
        "provider": "test",
        "base_url": "https://example.test",
        "start_date": "2022-06-01",
        "end_date": "2022-06-02",
        "first_month": "2022-06",
        "last_month": "2022-06",
        "expected_archive_count": 5,
        "maximum_total_archive_bytes": 10_000,
        "minimum_calendar_days": 2,
        "minimum_observations_per_day": 1,
        "maximum_gap_hours": 24,
        "instruments": instruments,
    }


def _small_archives(config: dict[str, object]) -> list[ValidatedArchive]:
    observations = (
        FundingObservation(_timestamp("2022-06-01T00:00:00"), 24, "0.0001"),
        FundingObservation(_timestamp("2022-06-02T00:00:00"), 24, "0.0002"),
    )
    return [
        ValidatedArchive(
            spec=InstrumentSpec(**item),
            month="2022-06",
            filename=f"{item['market_symbol']}-fundingRate-2022-06.zip",
            archive_bytes=100,
            uncompressed_bytes=200,
            official_sha256="a" * 64,
            calculated_sha256="a" * 64,
            observations=observations,
        )
        for item in config["instruments"]
    ]


def test_history_and_artifact_require_full_daily_coverage(tmp_path: Path):
    config = _small_config()
    archives = _small_archives(config)
    history = validate_history(archives, config)
    assert set(history) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"}

    artifact = write_acquisition_artifact(
        archives,
        config=config,
        config_sha256="config-hash",
        acquisition_code_sha256="code-hash",
        output_root=tmp_path,
    )
    manifest = json.loads((artifact / "acquisition_manifest.json").read_text(encoding="utf-8"))
    assert manifest["targets_created"] is False
    assert manifest["predictive_features_created"] is False
    assert manifest["inputs"][0]["calendar_days"] == 2
    assert (artifact / manifest["inputs"][0]["events_file"]).exists()

    broken = list(archives)
    broken[0] = ValidatedArchive(
        **{**broken[0].__dict__, "observations": broken[0].observations[:1]}
    )
    with pytest.raises(ValueError, match="Missing funding days"):
        validate_history(broken, config)


@pytest.mark.asyncio
async def test_public_download_stops_at_the_frozen_byte_limit():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 100)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="frozen byte limit"):
            await _get_bytes(client, "https://example.test/archive.zip", 1, 20)
