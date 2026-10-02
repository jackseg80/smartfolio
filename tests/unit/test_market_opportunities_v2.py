"""Synthetic fixtures only: contracts, missing evidence and conservation regressions."""
import copy
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from services.ml.bourse.fund_exposure import parse_ishares, SECTORS
from services.ml.bourse.market_snapshot import load_csv_snapshot, normalize_rows, csv_value_currency
from services.ml.bourse.market_analysis import exposure_summary, simulate, historical_screen, screen_candidate, scenario_history, PublicMarketData, listing


def issuer_html(rows='<tr><td>Information Technology</td><td>60</td></tr><tr><td>Financials</td><td>25</td></tr>', observed='29/Sept/2026', headers='<th>Type</th><th>Fund</th>', isin='TEST00000001'):
    return f'<html><p>ISIN {isin}</p><div>NAV as of 01/Oct/2026</div><div id="tabpanel-exposureBreakdowns-sector">% of Market Value as of {observed}<table data-id="exposurebreakdowns-sector-table"><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div></html>'


def test_issuer_partial_breakdown_preserves_unknown_weight_and_own_date():
    result = parse_ishares(issuer_html(), 'TEST00000001', 'https://www.ishares.com/test', date(2026, 10, 1))
    assert result['known_fraction'] == .85
    assert result['weights'] == {'Technology': .6, 'Financials': .25}
    assert result['as_of'] == '2026-09-29'


@pytest.mark.parametrize('html', [issuer_html(isin='OTHER'), issuer_html(observed='01/Jan/2026'),
    issuer_html(observed='03/Oct/2026'), issuer_html(headers='<th>Type</th><th>Benchmark</th>'),
    issuer_html(rows='<tr><td>Technology</td><td>101</td></tr>'),
    issuer_html(rows='<tr><td>Technology</td><td>NaN</td></tr>'),
    issuer_html(rows='<tr><td>Technology</td><td>60</td></tr><tr><td>Technology</td><td>20</td></tr>'),
    issuer_html(rows='<tr><td>Technology</td><td>80</td></tr><tr><td>Cash</td><td>30</td></tr>')])
def test_issuer_rejects_wrong_identity_staleness_index_and_invalid_weights(html):
    with pytest.raises(ValueError):
        parse_ishares(html, 'TEST00000001', 'https://www.ishares.com/test', date(2026, 10, 1))


def test_geographic_label_does_not_become_an_industry():
    result = parse_ishares(issuer_html(rows='<tr><td>Europe</td><td>40</td></tr><tr><td>Technology</td><td>30</td></tr>'), 'TEST00000001', 'https://www.ishares.com/test', date(2026, 10, 1))
    assert result['weights'] == {'Technology': .3}
    assert result['known_fraction'] == .3


def row(symbol='ABC:xvtx', isin='TEST00000001', value=100):
    return dict(symbol=symbol, isin=isin, quantity=2, market_value=value, asset_class='Stock', currency='CHF')


def selected_source(tmp_path, key='selected.csv', cash=None):
    folder = tmp_path / 'data/users/test/saxobank/data'
    folder.mkdir(parents=True)
    (folder / key).write_text('Symbol,Valeur actuelle (EUR)\nABC,100\n')
    (folder / 'other.csv').write_text('Symbol,Valeur actuelle (USD)\nBAD,999\n')
    (folder.parents[1] / 'config.json').write_text(json.dumps({'sources': {'bourse': {'active_source': 'saxobank_csv', 'selected_csv_file': key}}}))
    if cash:
        cash_folder = folder.parent / 'cash'; cash_folder.mkdir()
        (cash_folder / (key + '_cash.json')).write_text(json.dumps(cash))
    return folder


def test_selected_csv_and_cash_share_one_key_with_explicit_value_currency(tmp_path):
    selected_source(tmp_path, cash={'cash_amount': 10, 'currency': 'CHF', 'last_updated': '2026-09-22T10:01:15Z'})
    calls = []
    result = load_csv_snapshot('test', root=tmp_path, rate_getter=lambda c: {'EUR': 2, 'CHF': 3}[c], row_loader=lambda u, k: calls.append((u, k)) or [row()])
    assert calls == [('test', 'selected.csv')]
    assert result['positions'][0]['value_usd'] == 200
    assert result['positions'][0]['quote_currency'] == 'CHF'
    assert result['positions'][0]['valuation_currency'] == 'EUR'
    assert result['positions'][0]['isin'] == 'TEST00000001'
    assert result['cash']['value_usd'] == 30
    assert result['context']['valuation_as_of'] is None


def test_missing_selected_cash_is_not_zero_or_default_cash(tmp_path):
    folder = selected_source(tmp_path)
    cash = folder.parent / 'cash'; cash.mkdir()
    (cash / 'default_cash.json').write_text('{"cash_amount": 1000000, "currency":"USD"}')
    result = load_csv_snapshot('test', root=tmp_path, rate_getter=lambda _: 1, row_loader=lambda *_: [row()])
    assert result['cash']['value_usd'] is None


def test_real_connector_path_handles_string_dtype_and_explicit_usd_header(tmp_path, monkeypatch):
    folder = selected_source(tmp_path)
    (folder / 'selected.csv').write_text('Instrument,Symbol,ISIN,Quantity,Market Value (USD),Currency,Asset Class\nTest,TEST:xnas,US9999999999,2,100,USD,Stock\n')
    from services import instruments_registry
    monkeypatch.setattr(instruments_registry, 'resolve', lambda *_, **__: {})
    result = load_csv_snapshot('test', root=tmp_path, rate_getter=lambda _: 1)
    assert len(result['positions']) == 1
    assert result['positions'][0]['value_usd'] == 100
    assert result['positions'][0]['valuation_currency'] == 'USD'


def test_missing_csv_value_retains_position_and_blocks_full_portfolio_conclusions(tmp_path, monkeypatch):
    folder = selected_source(tmp_path)
    (folder / 'selected.csv').write_text('Instrument,Symbol,ISIN,Quantity,Market Value (USD),Currency,Asset Class\nTest,TEST:xnas,US9999999999,2,-,USD,Stock\nSummary,-,-,-,-,-,-\n')
    from services import instruments_registry
    monkeypatch.setattr(instruments_registry, 'resolve', lambda *_, **__: {})
    result = load_csv_snapshot('test', root=tmp_path, rate_getter=lambda _: 1)
    assert len(result['positions']) == 1
    assert result['positions'][0]['value_usd'] is None
    assert result['context']['unvalued_positions'] == 1
    data = analysis(); data['positions'][1]['value_usd'] = None
    exposure = exposure_summary(data['positions'], data['targets'])
    assert exposure['coverage_scope'] == 'valued_subset_only'
    assert all(s['status'] == 'Indeterminate' and s['lower_pct'] is None and s['certain_deficit_pct'] is None for s in exposure['sector_assessment'])
    with pytest.raises(ValueError, match='no source valuation'):
        simulate(data, scenario())


@pytest.mark.parametrize('file_key', ['other.csv', '../other.csv', 'C:\\private.csv'])
def test_request_cannot_substitute_another_csv(tmp_path, file_key):
    selected_source(tmp_path)
    with pytest.raises(ValueError):
        load_csv_snapshot('test', file_key, root=tmp_path, rate_getter=lambda _: 1, row_loader=lambda *_: [row()])


def test_missing_selection_never_uses_the_latest_csv(tmp_path):
    folder = selected_source(tmp_path)
    (folder / 'selected.csv').unlink()
    with pytest.raises(ValueError, match='unavailable'):
        load_csv_snapshot('test', root=tmp_path, rate_getter=lambda _: 1, row_loader=lambda *_: [row()])


def test_header_currency_is_required_and_ambiguous_headers_are_rejected():
    assert csv_value_currency(b'Symbol,Market Value (USD)\nABC,1') == 'USD'
    assert csv_value_currency(b'Symbol,Market Value\nABC,1') is None
    assert csv_value_currency(b'Symbol,Market Value (USD),Market Value (EUR)\nABC,1,1') is None


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True])
def test_invalid_positions_are_not_silently_dropped(value):
    with pytest.raises(ValueError):
        normalize_rows([row(value=value)], 'EUR', 1)


def test_duplicate_position_identity_is_not_silently_double_counted():
    with pytest.raises(ValueError, match='duplicate'):
        normalize_rows([row(), row()], 'EUR', 1)


def test_swiss_currency_qualified_listing_uses_isin_and_never_substitutes_usd():
    position = {'symbol': 'WRDUSW_CHF:xswx', 'isin': 'IE00BD4TXV59', 'quote_currency': 'CHF'}
    assert listing(position) == 'WRDUSW.SW'
    for change in [{'isin': None}, {'isin': 'WRONG'}, {'quote_currency': 'USD'}, {'symbol': 'WRDUSW_USD:xswx'}]:
        with pytest.raises(ValueError): listing({**position, **change})


def test_usd_conversion_overflow_is_rejected_before_serialization():
    with pytest.raises(ValueError): normalize_rows([row(value=1e308)], 'EUR', 10)


def test_missing_source_quote_currency_is_not_the_connectors_default_usd(tmp_path, monkeypatch):
    folder = selected_source(tmp_path)
    (folder / 'selected.csv').write_text('Instrument,Symbol,ISIN,Quantity,Market Value (USD),Asset Class\nTest,TEST:xnas,US9999999999,2,100,Stock\n')
    from services import instruments_registry
    monkeypatch.setattr(instruments_registry, 'resolve', lambda *_, **__: {})
    result = load_csv_snapshot('test', root=tmp_path, rate_getter=lambda _: 1)
    assert result['positions'][0]['quote_currency'] is None


def positions():
    return [dict(id='held-a', symbol='A', name='A', isin='TESTA', value_usd=400, quantity=4,
                 acquisition_date=None, history_symbol='A', quote_currency='USD',
                 classification=dict(weights={'Technology': 1}, known_fraction=1, reason='Test')),
            dict(id='held-b', symbol='B', name='B', isin='TESTB', value_usd=600, quantity=6,
                 acquisition_date=None, history_symbol='B', quote_currency='USD',
                 classification=dict(weights={}, known_fraction=0, reason='Test unknown'))]


def analysis():
    targets = {s: 0 for s in SECTORS}; targets['Technology'] = 50; targets['Healthcare'] = 50
    rows = positions()
    return dict(snapshot_id='scope', positions=rows, cash={'value_usd': 100}, targets=targets, min_gap_pct=5,
                exposure=exposure_summary(rows, targets), horizon='short',
                candidates=[dict(id='C', symbol='C', name='C', kind='ETF', status='screened', isin=None)])


def scenario(trades=None, **overrides):
    return dict(snapshot_id='scope', trades=trades or [{'side': 'sell', 'id': 'held-a', 'amount_usd': 100}, {'side': 'buy', 'id': 'C', 'amount_usd': 150}],
                costs_usd=5, slippage_pct=1, acknowledge_dated_values=True, **overrides)


def test_bounds_retain_uncertainty_even_when_no_gap_is_certain():
    result = analysis()['exposure']
    assert result['coverage'] == .4
    tech = next(r for r in result['sector_assessment'] if r['sector'] == 'Technology')
    assert (tech['lower_pct'], tech['upper_pct'], tech['status']) == (40, 100, 'Indeterminate')


def test_manual_scenario_sells_by_stable_identity_buys_and_conserves_cash_after_friction():
    original = analysis(); copy_before = copy.deepcopy(original)
    result = simulate(original, scenario())
    assert result['cash_after_usd'] == 42.5
    assert result['after_total_usd'] == 1092.5
    assert result['costs_and_slippage_usd'] == 7.5
    assert next(p for p in result['positions_after'] if p['id'] == 'held-a')['value_usd'] == 300
    assert next(p for p in result['positions_after'] if p['id'] == 'candidate:C')['classification']['known_fraction'] == 0
    assert original == copy_before


@pytest.mark.parametrize('change', [
    {'snapshot_id': 'stale'}, {'acknowledge_dated_values': False}, {'costs_usd': float('nan')},
    {'slippage_pct': -1}, {'trades': []}, {'trades': [{'side':'sell','id':'WRONG','amount_usd':50}]},
    {'trades': [{'side':'sell','id':'held-a','amount_usd':401}]},
    {'trades': [{'side':'buy','id':'C','amount_usd':101}]},
    {'trades': [{'side':'buy','id':'C','amount_usd':True}]},
    {'trades': [{'side':'buy','id':'C','amount_usd':1}, {'side':'buy','id':'C','amount_usd':1}]},
])
def test_invalid_or_underfunded_scenarios_fail_closed(change):
    value = scenario(); value.update(change)
    with pytest.raises(ValueError):
        simulate(analysis(), value)


def test_cash_missing_blocks_funding_simulation():
    data = analysis(); data['cash']['value_usd'] = None
    with pytest.raises(ValueError, match='cash is unavailable'):
        simulate(data, scenario())


def test_historical_score_requires_full_window_and_actual_variability():
    index = pd.bdate_range('2026-01-01', periods=44)
    close = pd.Series(100 * np.cumprod(1 + np.resize([.01, -.003], 44)), index=index)
    result = historical_screen(close, 42)
    assert result['sessions'] == 42
    assert result['currency'] == 'USD' and result['annualized_volatility_pct'] > 0
    with pytest.raises(ValueError): historical_screen(close.iloc[:20], 42)
    with pytest.raises(ValueError): historical_screen(pd.Series(100., index=index), 42)


class Provider:
    async def profile(self, symbol): return {'sector': 'Technology', 'isin': 'TESTA'}
    async def history(self, *args): raise ValueError('Exact history missing')


@pytest.mark.asyncio
async def test_candidate_same_isin_on_another_listing_is_excluded():
    candidate = dict(symbol='A.OTHER', name='A alternate', kind='STOCK', intended_sector='Technology')
    result = await screen_candidate(candidate, positions(), Provider(), 'short')
    assert result['status'] == 'held' and 'ISIN' in result['exclusions'][0]


@pytest.mark.asyncio
async def test_history_gaps_block_full_portfolio_risk_instead_of_proxying():
    data = analysis(); result = simulate(data, scenario())
    risk = await scenario_history(data, result, Provider())
    assert risk['status'] == 'unavailable' and risk['volatility_before_pct'] is None
    assert risk['histories_available'] == 0 and risk['positions_required'] == 3


@pytest.mark.asyncio
async def test_full_actual_histories_calculate_both_volatilities_and_candidate_correlation():
    index = pd.bdate_range('2026-01-01', periods=90)
    class Histories:
        async def history(self, symbol, *_):
            scale = {'A': 1, 'B': 2, 'C': 0.5}[symbol]
            return pd.Series(100 * np.cumprod(1 + np.resize([.01, -.005, .001], 90) * scale), index=index)
    data = analysis(); result = simulate(data, scenario())
    risk = await scenario_history(data, result, Histories())
    assert risk['status'] == 'calculated'
    assert risk['volatility_before_pct'] > risk['volatility_after_pct'] > 0
    assert risk['correlation_changes'][0]['correlation_to_before'] == pytest.approx(1)


@pytest.mark.asyncio
@pytest.mark.parametrize('missing_session,expected_currency', [(False, 'USD'), (True, 'USD'), (False, 'CHF')])
async def test_public_provider_excludes_today_and_rejects_missing_sessions_or_wrong_currency(monkeypatch, missing_session, expected_currency):
    import asyncio
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    import yfinance
    from services.ml.live_observations import completed_stock_sessions
    from services.ml.bourse.data_sources import StocksDataSource
    today = datetime.now().date()
    index = completed_stock_sessions('XNYS', today - timedelta(days=100), today).tz_localize(None)
    values = np.arange(len(index)) + 100.
    frame = pd.DataFrame({'close': values}, index=index)
    if missing_session: frame = frame.drop(index[-3])
    # An uncompleted observation must never affect a historical screen.
    frame.loc[pd.Timestamp(today), 'close'] = 999999.
    frame.attrs = dict(price_origin='yahoo', price_adjustment='split_and_dividend_adjusted', native_currency='USD')
    monkeypatch.setattr(yfinance, 'Ticker', lambda *_: SimpleNamespace(info={'exchange': 'PCX', 'currency': 'USD'}))
    async def data(*_, **__): return frame.copy()
    monkeypatch.setattr(StocksDataSource, 'get_ohlcv_data', data)
    provider = PublicMarketData()
    if missing_session or expected_currency != 'USD':
        with pytest.raises(ValueError):
            await asyncio.wait_for(provider.history('TEST', 'short', expected_currency), 5)
    else:
        close = await asyncio.wait_for(provider.history('TEST', 'short', expected_currency), 5)
        assert close.index[-1] == index[-1] and len(close) == len(index)
        assert close.iloc[-1] != 999999
