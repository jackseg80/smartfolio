from types import SimpleNamespace
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from services.risk.bourse.calculator import BourseRiskCalculator
from services.risk.bourse.data_fetcher import BourseDataFetcher
from services.ml.bourse.currency_detector import CurrencyExchangeDetector


def test_explicit_exchange_wins_over_symbol_and_domicile():
    detector = CurrencyExchangeDetector()
    assert detector.detect_currency_and_exchange('SAP', isin='DE0007164600', exchange_hint='NYSE')[0] == 'SAP'
    assert detector.detect_currency_and_exchange('SMH', exchange_hint='NASDAQ')[0] == 'SMH'
    assert detector.detect_currency_and_exchange('IWDA', exchange_hint='LSE')[0] == 'IWDA.L'
    with pytest.raises(ValueError, match='Unsupported exchange'):
        detector.detect_currency_and_exchange('ABC', exchange_hint='UNKNOWN')


def test_usd_returns_include_historical_fx_movement():
    dates = pd.bdate_range('2026-01-01', periods=3)
    prices = pd.DataFrame({'close': [100., 110., 110.]}, index=dates)
    fx = pd.Series([1., 1.1, 1.], index=dates)
    result = BourseRiskCalculator._convert_prices_to_usd(prices, fx)
    assert result['close'].tolist() == pytest.approx([100., 121., 110.])
    assert result['close'].pct_change().iloc[1] == pytest.approx(0.21)
    assert prices['close'].tolist() == [100., 110., 110.]


def test_fx_alignment_never_uses_future_or_unlimited_stale_rates():
    prices = pd.DataFrame({'close': [100.]}, index=[pd.Timestamp('2026-01-05')])
    for rate_date in ('2026-01-06', '2025-12-31'):
        fx = pd.Series([1.2], index=[pd.Timestamp(rate_date)])
        with pytest.raises(ValueError, match='cover every'):
            BourseRiskCalculator._convert_prices_to_usd(prices, fx)
    fx = pd.Series([1.2], index=[pd.Timestamp('2026-01-02')])
    assert BourseRiskCalculator._convert_prices_to_usd(prices, fx)['close'].iloc[0] == 120.


def test_return_alignment_rejects_different_start_dates():
    dates = pd.bdate_range('2026-01-01', periods=4)
    data = {
        'A': {'prices': pd.DataFrame({'close': [100., 110., 121., 133.1]}, index=dates), 'weight': 0.5},
        'B': {'prices': pd.DataFrame({'close': [100., 121., 133.1]}, index=dates[[0, 2, 3]]), 'weight': 0.5},
    }
    result = BourseRiskCalculator._calculate_portfolio_returns(object.__new__(BourseRiskCalculator), data)
    assert list(result.index) == [dates[3]]
    assert result.iloc[0] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_quote_currency_is_verified_and_pence_are_normalized(monkeypatch, tmp_path):
    dates = pd.date_range('2026-01-01', periods=3)
    frame = pd.DataFrame({'Open': [100., 110., np.nan], 'High': [100., 110., np.nan],
                          'Low': [100., 110., np.nan], 'Close': [100., 110., np.nan], 'Volume': [10., 20., 0.]}, index=dates)
    class Instrument:
        history_metadata = {'currency': 'GBp'}
        def history(self, **kwargs):
            assert kwargs['auto_adjust'] is True
            return frame
    monkeypatch.setitem(sys.modules, 'yfinance', SimpleNamespace(Ticker=lambda symbol: Instrument()))
    result = await BourseDataFetcher(str(tmp_path))._fetch_yahoo_finance('ABC.L', datetime(2026, 1, 1), datetime(2026, 2, 1))
    assert result.attrs['native_currency'] == 'GBP'
    assert result['close'].tolist() == pytest.approx([1., 1.1])
    assert result['volume'].tolist() == [10., 20.]
    assert result.attrs['dropped_observations'] == 1
@pytest.mark.asyncio
@pytest.mark.parametrize('end_date, expected', [('2026-03-16', 'ROG.SW'), ('2026-05-20', 'ROP.SW')])
async def test_roche_history_follows_verified_exchange_date(monkeypatch, tmp_path, end_date, expected):
    from datetime import datetime
    from unittest.mock import AsyncMock
    from services.risk.bourse.data_fetcher import BourseDataFetcher
    fetcher = BourseDataFetcher(cache_dir=str(tmp_path))
    prices = pd.DataFrame({'close': [100., 101.]}, index=pd.bdate_range('2026-03-10', periods=2))
    prices.attrs['native_currency'] = 'CHF'
    provider = AsyncMock(return_value=prices)
    monkeypatch.setattr(fetcher, '_fetch_yahoo_finance', provider)
    result = await fetcher.fetch_historical_prices('ROG:xvtx', datetime(2026, 1, 1), datetime.fromisoformat(end_date))
    assert provider.call_args.args[0] == expected
    assert result.attrs['history_symbol'] == expected
    if expected == 'ROP.SW':
        assert result.attrs['corporate_action']['exchange_ratio'] == 1.0
