import json
from pathlib import Path

import pytest

from services.forecasting.dataset import load_price_cache
from services.forecasting.history_acquisition import (
    DAY_MS,
    AcquiredInstrument,
    DailyObservation,
    InstrumentSpec,
    parse_daily_klines,
    write_acquisition_artifact,
)


def _row(day: int, close: str = "101.5") -> list[object]:
    open_time = day * DAY_MS
    return [
        open_time,
        "100.0",
        "102.0",
        "99.0",
        close,
        "12.5",
        open_time + DAY_MS - 1,
        "1268.75",
        42,
        "6.0",
        "608.0",
        "0",
    ]


def test_parse_daily_klines_preserves_decimals_and_rejects_duplicates():
    shortened = _row(1)
    shortened[6] = DAY_MS + 3_600_000
    observations = parse_daily_klines([shortened, _row(2, "103.25")])

    assert observations[1].close == "103.25"
    assert observations[1].trades == 42
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_daily_klines([_row(1), _row(1)])
    outside_day = _row(1)
    outside_day[6] = 2 * DAY_MS
    with pytest.raises(ValueError, match="outside its UTC day"):
        parse_daily_klines([outside_day])


def test_write_artifact_is_loadable_with_verified_provider_provenance(tmp_path: Path):
    observations = tuple(
        DailyObservation(
            open_time_ms=day * DAY_MS,
            open="100.0",
            high="102.0",
            low="99.0",
            close=str(100 + day),
            volume="12.5",
            close_time_ms=(day + 1) * DAY_MS - 1,
            quote_asset_volume="1268.75",
            trades=42,
            taker_buy_base_volume="6.0",
            taker_buy_quote_volume="608.0",
        )
        for day in range(1, 4)
    )
    acquired = AcquiredInstrument(
        spec=InstrumentSpec(symbol="BTC", market_symbol="BTCUSDT"),
        exchange_status="TRADING",
        base_asset="BTC",
        quote_asset="USDT",
        observations=observations,
    )
    config = {
        "schema_version": "crypto-forecast-history-acquisition-v1",
        "provider": "binance_spot_public_market_data",
        "base_url": "https://data-api.binance.vision",
        "interval": "1d",
        "quote_asset": "USDT",
        "start_date": "1970-01-02",
        "minimum_observations": 3,
    }

    artifact = write_acquisition_artifact(
        [acquired],
        config=config,
        config_sha256="config-hash",
        acquisition_code_sha256="code-hash",
        end_date="1970-01-04",
        output_root=tmp_path,
    )
    manifest = json.loads((artifact / "acquisition_manifest.json").read_text(encoding="utf-8"))
    loaded = load_price_cache(artifact / "prices")

    assert manifest["credentials_used"] is False
    assert manifest["inputs"][0]["ohlcv_file_sha256"]
    assert loaded.inputs[0]["provider_provenance"] == "binance_spot_public_market_data"
    assert loaded.inputs[0]["market_symbol"] == "BTCUSDT"
