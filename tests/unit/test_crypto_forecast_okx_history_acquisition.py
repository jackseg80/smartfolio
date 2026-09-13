import json
from pathlib import Path

import pytest

from services.forecasting.okx_history_acquisition import (
    DAY_MS,
    AcquiredInstrument,
    DailyObservation,
    InstrumentSpec,
    acquire_instrument,
    parse_daily_candles,
    write_acquisition_artifact,
)


def _row(day: int, close: str = "101.5", confirmed: str = "1") -> list[object]:
    return [
        str(day * DAY_MS),
        "100.0",
        "102.0",
        "99.0",
        close,
        "12.5",
        "1268.75",
        "1268.75",
        confirmed,
    ]


class FakeClient:
    def __init__(self) -> None:
        self.candle_calls = 0

    async def get_json(self, path: str, params: dict[str, object]) -> object:
        if path.endswith("/instruments"):
            return {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "instId": "BTC-USDT",
                        "state": "live",
                        "baseCcy": "BTC",
                        "quoteCcy": "USDT",
                        "listTime": str(DAY_MS + 1_000),
                    }
                ],
            }
        self.candle_calls += 1
        rows = [_row(4, "104"), _row(3, "103")] if self.candle_calls == 1 else [_row(2)]
        return {"code": "0", "msg": "", "data": rows}


def test_parse_daily_candles_orders_rows_and_rejects_unconfirmed():
    observations = parse_daily_candles([_row(2, "102"), _row(1, "101")])

    assert [item.open_time_ms for item in observations] == [DAY_MS, 2 * DAY_MS]
    assert observations[0].quote_volume == "1268.75"
    with pytest.raises(ValueError, match="Unconfirmed"):
        parse_daily_candles([_row(1, confirmed="0")])
    with pytest.raises(ValueError, match="unique"):
        parse_daily_candles([_row(1), _row(1)])


@pytest.mark.asyncio
async def test_acquire_instrument_paginates_backwards_and_clamps_to_listing_day():
    client = FakeClient()

    acquired = await acquire_instrument(
        client,
        InstrumentSpec(symbol="BTC", market_symbol="BTC-USDT"),
        start_date="1970-01-01",
        end_date="1970-01-05",
        page_limit=2,
        request_delay_seconds=0.0,
    )

    assert client.candle_calls == 2
    assert [item.date for item in acquired.observations] == [
        "1970-01-03",
        "1970-01-04",
        "1970-01-05",
    ]
    assert acquired.listing_time_ms == DAY_MS + 1_000


def test_write_artifact_preserves_missing_days_and_explicit_unavailable_trade_count(
    tmp_path: Path,
):
    observations = tuple(
        DailyObservation(
            open_time_ms=day * DAY_MS,
            open="100.0",
            high="102.0",
            low="99.0",
            close=str(100 + day),
            base_volume="12.5",
            quote_volume="1268.75",
            quote_volume_native="1268.75",
            confirmed=True,
        )
        for day in (1, 3, 4)
    )
    acquired = AcquiredInstrument(
        spec=InstrumentSpec(symbol="BTC", market_symbol="BTC-USDT"),
        exchange_status="live",
        base_asset="BTC",
        quote_asset="USDT",
        listing_time_ms=DAY_MS + 1_000,
        observations=observations,
    )
    config = {
        "schema_version": "crypto-forecast-okx-history-v1",
        "provider": "okx_spot_public_market_data",
        "base_url": "https://www.okx.com",
        "bar": "1Dutc",
        "quote_asset": "USDT",
        "start_date": "1970-01-02",
        "minimum_observations": 3,
    }

    artifact = write_acquisition_artifact(
        [acquired],
        config=config,
        config_sha256="config-hash",
        acquisition_code_sha256="code-hash",
        end_date="1970-01-05",
        output_root=tmp_path,
    )
    manifest = json.loads((artifact / "acquisition_manifest.json").read_text(encoding="utf-8"))

    assert manifest["credentials_used"] is False
    assert manifest["trade_count_availability"] == "unavailable_in_okx_daily_candle_response"
    assert manifest["inputs"][0]["missing_calendar_days"] == ["1970-01-03"]
    assert (artifact / manifest["inputs"][0]["ohlcv_file"]).exists()
