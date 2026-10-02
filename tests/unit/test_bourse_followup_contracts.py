"""Regression checks for verified listings, explicit horizons and sector policies."""
from datetime import datetime
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from services.ml.bourse.currency_detector import CurrencyExchangeDetector
from services.ml.bourse.horizons import OPPORTUNITY_HORIZONS, RECOMMENDATION_HORIZONS
from services.ml.bourse.opportunity_scanner import OpportunityScanner, parse_sector_targets
from services.ml.bourse.sector_analyzer import SectorAnalyzer
from services.risk.bourse.data_fetcher import BourseDataFetcher, MarketDataUnavailableError


def test_currency_qualified_listing_requires_verified_venue_and_identity():
    detector = CurrencyExchangeDetector()
    assert detector.detect_currency_and_exchange('WRDUSW_CHF', isin='IE00BD4TXV59', exchange_hint='SWX')[0:2] == ('WRDUSW.SW', 'CHF')
    for symbol, kwargs in [
        ('WRDUSW_CHF', {}),
        ('UNKNOWN_CHF', {'exchange_hint': 'SWX'}),
        ('WRDUSW_CHF', {'exchange_hint': 'SWX', 'isin': 'WRONG'}),
        ('WRDUSW_CHF', {'exchange_hint': 'NASDAQ'}),
    ]:
        with pytest.raises(ValueError):
            detector.detect_currency_and_exchange(symbol, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('currency', ['CHF', 'USD', None])
async def test_qualified_listing_checks_provider_currency_before_caching(tmp_path, monkeypatch, currency):
    fetcher = BourseDataFetcher(str(tmp_path))
    prices = pd.DataFrame({'close': [100., 101.]}, index=pd.bdate_range('2026-01-01', periods=2))
    prices.attrs['native_currency'] = currency
    provider = AsyncMock(return_value=prices)
    monkeypatch.setattr(fetcher, '_fetch_yahoo_finance', provider)
    args = ('WRDUSW_CHF:xswx', datetime(2026, 1, 1), datetime(2026, 1, 5))
    if currency == 'CHF':
        assert (await fetcher.fetch_historical_prices(*args)).attrs['native_currency'] == 'CHF'
        assert provider.call_args.args[0] == 'WRDUSW.SW'
        # The same provider symbol must not satisfy a different currency line from memory.
        with pytest.raises(MarketDataUnavailableError, match='currency mismatch'):
            await fetcher.fetch_historical_prices('WRDUSW_USD:xswx', *args[1:])
        assert provider.await_count == 1
        # The persistent cache is also rechecked, including after process restart.
        second = BourseDataFetcher(str(tmp_path))
        monkeypatch.setattr(second, '_fetch_yahoo_finance', provider)
        with pytest.raises(MarketDataUnavailableError, match='currency mismatch'):
            await second.fetch_historical_prices('WRDUSW_USD:xswx', *args[1:])
    else:
        with pytest.raises(MarketDataUnavailableError, match='Quote currency'):
            await fetcher.fetch_historical_prices(*args)
        assert not fetcher.cache


@pytest.mark.parametrize('raw', [None, {'Technology': 60, 'Healthcare': 40}, '{"Technology":100}'])
def test_sector_targets_accept_valid_explicit_policy(raw):
    targets = parse_sector_targets(raw)
    if raw is None:
        assert targets is None
    else:
        assert sum(targets.values()) == pytest.approx(100)
        assert targets['Utilities'] == 0


@pytest.mark.parametrize('raw', [{}, [], {'Europe': 100}, {'Technology': True}, {'Technology': float('nan')},
    {'Technology': float('inf')}, {'Technology': -1, 'Healthcare': 101}, {'Technology': 90}, 'invalid'])
def test_sector_targets_reject_ambiguous_or_invalid_values(raw):
    with pytest.raises(ValueError, match='totaling 100%'):
        parse_sector_targets(raw)


def test_unknown_exposure_is_not_counted_as_a_sector_gap():
    scanner = object.__new__(OpportunityScanner)
    targets = parse_sector_targets({'Technology': 60, 'Healthcare': 40})
    # Technology may already occupy all the unclassified 50%.
    assert scanner._detect_gaps({'Technology': 50}, 5, targets, 50) == []
    gaps = scanner._detect_gaps({'Technology': 70, 'Healthcare': 20}, 5, targets, 10)
    assert len(gaps) == 1
    assert gaps[0]['sector'] == 'Healthcare'
    assert gaps[0]['gap_pct'] == 10
    assert gaps[0]['gap_kind'] == 'minimum_verified_gap'
    assert scanner._detect_gaps({'Technology': 60, 'Healthcare': 40}, 0, targets, 0) == []


def test_long_opportunity_requires_multiyear_observations():
    analyzer = object.__new__(SectorAnalyzer)
    dates = pd.bdate_range('2020-01-01', periods=800)
    prices = pd.DataFrame({'close': 100 * np.cumprod(1 + .0005 + .005 * np.sin(np.arange(800)))}, index=dates)
    assert analyzer._calculate_momentum_score(prices.tail(200), prices.tail(200), 'long') is None
    assert analyzer._calculate_momentum_score(prices, prices, 'long') is not None
    assert analyzer._get_lookback_days('long') >= 3 * 365


def test_holding_horizon_is_distinct_from_signal_and_not_a_performance_claim():
    short = OPPORTUNITY_HORIZONS['short'].metadata()
    assert (short['holding_sessions_min'], short['holding_sessions_max'], short['signal_sessions']) == (21, 63, 42)
    for horizon in [*OPPORTUNITY_HORIZONS.values(), *RECOMMENDATION_HORIZONS.values()]:
        assert horizon.metadata()['forward_validated'] is False


@pytest.mark.asyncio
async def test_opportunities_forward_selected_csv_and_auth_before_scanning(monkeypatch):
    import json
    from starlette.requests import Request
    import api.ml_bourse_endpoints as endpoints
    from services.ml.bourse import market_snapshot, market_analysis
    loader = AsyncMock(return_value={'positions': [], 'snapshot_id': 'test'})
    analyst = AsyncMock(return_value={'version': 2, 'context': {'user_id': 'alice'}})
    monkeypatch.setattr(market_snapshot, 'load_snapshot', loader)
    monkeypatch.setattr(market_analysis, 'analyze', analyst)
    request = Request({'type': 'http', 'headers': [(b'authorization', b'Bearer test-token')]})
    result = await endpoints.get_market_opportunities(request=request, user='alice', horizon='short', source='saxobank_csv',
        file_key='selected.csv', min_gap_pct=5, sector_targets=None, candidate_sector='all')
    assert json.loads(result.body)['data']['version'] == 2
    assert loader.call_args.args[:3] == ('alice', 'saxobank_csv', 'selected.csv')
    assert loader.call_args.args[3] is request
    assert endpoints._forward_authenticated_headers(request, 'alice')['Authorization'] == 'Bearer test-token'
    assert endpoints._forward_authenticated_headers(request, 'alice')['X-User'] == 'alice'
    loader.reset_mock()
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        await endpoints.get_market_opportunities(request=request, user='alice', horizon='short', source='saxobank_csv',
            file_key='selected.csv', min_gap_pct=5, sector_targets='{"Technology":101}', candidate_sector='all')
    assert error.value.status_code == 422
    loader.assert_not_called()
