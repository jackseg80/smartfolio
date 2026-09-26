"""Financial outputs must fail closed when their selected data is unavailable."""

from datetime import datetime
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

from adapters.saxo_adapter import _load_snapshot
from services.risk.bourse.calculator import (
    BourseRiskCalculator,
    IncompleteBourseMarketData,
)
from services.risk.bourse.data_fetcher import (
    BourseDataFetcher,
    MarketDataUnavailableError,
)


@pytest.mark.asyncio
async def test_yahoo_failure_never_creates_synthetic_prices(monkeypatch, tmp_path):
    def fail_download(*args, **kwargs):
        raise RuntimeError("market provider unavailable")

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda symbol: SimpleNamespace(history=fail_download)))
    fetcher = BourseDataFetcher(cache_dir=str(tmp_path))

    with pytest.raises(MarketDataUnavailableError, match="Market prices unavailable"):
        await fetcher._fetch_yahoo_finance(
            "AAPL", datetime(2026, 1, 1), datetime(2026, 2, 1)
        )


@pytest.mark.asyncio
async def test_explicit_synthetic_source_is_not_available(monkeypatch, tmp_path):
    fetcher = BourseDataFetcher(cache_dir=str(tmp_path))
    monkeypatch.setattr(
        fetcher.currency_detector,
        "detect_currency_and_exchange",
        lambda **kwargs: ("AAPL", "USD", "NASDAQ"),
    )

    with pytest.raises(ValueError, match="Unknown data source"):
        await fetcher.fetch_historical_prices(
            "AAPL", datetime(2026, 1, 1), datetime(2026, 2, 1), source="manual"
        )


@pytest.mark.asyncio
async def test_risk_score_is_unavailable_when_one_holding_has_no_prices():
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    prices = pd.DataFrame({"close": range(100, 130)}, index=dates)

    class Fetcher:
        async def fetch_historical_prices(self, ticker, *args, **kwargs):
            if ticker == "BBB":
                raise MarketDataUnavailableError("missing")
            return prices

    calculator = object.__new__(BourseRiskCalculator)
    calculator.data_fetcher = Fetcher()
    calculator.data_source = "yahoo"
    positions = [
        {"symbol": "AAA", "market_value_usd": 60.0},
        {"symbol": "BBB", "market_value_usd": 40.0},
    ]

    with pytest.raises(IncompleteBourseMarketData) as exc:
        await calculator.calculate_portfolio_risk(positions)
    assert exc.value.symbols == ["BBB"]
    assert exc.value.coverage == pytest.approx(0.6)


def test_missing_selected_saxo_csv_does_not_load_latest(monkeypatch):
    monkeypatch.setattr(
        "api.services.user_fs.UserScopedFS.glob_files",
        lambda self, pattern: ["data/users/jack/saxobank/data/latest.csv"],
    )

    with pytest.raises(FileNotFoundError, match="Selected Saxo CSV"):
        _load_snapshot(user_id="jack", file_key="selected.csv")


def test_concentration_uses_selected_holdings_and_cash():
    calculator = object.__new__(BourseRiskCalculator)
    result = calculator._calculate_concentration_metrics(
        [
            {"symbol": "AAA", "market_value_usd": 50.0},
            {"symbol": "BBB", "market_value_usd": 30.0},
        ],
        portfolio_value=100.0,
        cash_amount=20.0,
    )

    assert result["top5_pct"] == 100.0
    assert result["largest_position_pct"] == 50.0
    assert result["herfindahl_index"] == pytest.approx(0.38)
    assert result["sector_max_pct"] is None


def test_current_decision_rejects_stale_price_series():
    old = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp.now() - pd.Timedelta(days=9)])
    with pytest.raises(MarketDataUnavailableError, match="latest observation"):
        BourseDataFetcher._require_fresh_prices(old, "AAPL", datetime.now())


@pytest.mark.asyncio
async def test_usd_risk_rejects_native_currency_returns_without_fx():
    calculator = object.__new__(BourseRiskCalculator)
    class Fetcher:
        async def fetch_historical_prices(self, *args, **kwargs):
            return pd.DataFrame({'close': [100.0]}, index=[pd.Timestamp('2026-01-01')])
        async def fetch_historical_fx(self, *args, **kwargs):
            raise MarketDataUnavailableError('No historical FX')
    calculator.data_fetcher = Fetcher()
    calculator.data_source = 'yahoo'
    with pytest.raises(IncompleteBourseMarketData) as exc:
        await calculator.calculate_portfolio_risk(
            [{"symbol": "NESN.SW", "currency": "CHF", "market_value_usd": 100.0}]
        )
    assert exc.value.coverage == 0.0
    assert exc.value.symbols == ['NESN.SW']
