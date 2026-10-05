"""Contracts: historical fidelity, current valuation, isolation and degraded data."""
import csv
import json
from copy import deepcopy
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from services import saxo_valuation_service as valuation
from services import saxo_quote_service as quotes
from services import portfolio_export_service as exports
from services.export_formatter import ExportFormatter
from connectors.saxo_import import SaxoImportConnector


@pytest.fixture
def reference():
    return {'file_key': 'Positions_18-janv.-2021.csv', 'export_date': '2021-01-18', 'imported_at': '2026-10-05',
            'cash': {'amount': 40, 'currency': 'EUR', 'asof': '2026-10-05', 'known': True},
            'positions': [{'symbol': 'AAPL:xnas', 'position_id': '1', 'instrument': 'Apple', 'quantity': 2,
                           'market_value': 100, 'account_base_currency': 'EUR', 'asset_class': 'Stock',
                           'currency': 'USD', 'avg_price': 20, 'export_price': 50}]}


@pytest.fixture
def current_quotes(monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    quote = {'price': 80, 'currency': 'USD', 'quote_at': now, 'fetched_at': now,
             'split_factor': 4, 'quantity_verified': True, 'retrieval_status': 'fresh',
             'source': 'yahoo', 'quote_type': 'latest_available'}
    monkeypatch.setattr(valuation, 'get_quote', lambda *_a: deepcopy(quote))
    def fx(source, target):
        usd = {'USD': 1, 'EUR': 2, 'CHF': 4}
        if source not in usd or target not in usd:
            return None, {'fresh': False, 'source': 'unavailable'}
        return usd[source] / usd[target], {'fresh': True, 'source': 'test'}
    monkeypatch.setattr(valuation, 'current_fx', fx)
    return quote


def test_export_never_calls_live_prices_or_fx(reference, monkeypatch):
    def forbidden(*_args):
        pytest.fail('Historical values must not use current prices/FX')
    monkeypatch.setattr(valuation, 'get_quote', forbidden)
    monkeypatch.setattr(valuation, 'current_fx', forbidden)
    before = deepcopy(reference)
    result = valuation.value_reference(reference, 'export')
    assert result['currency'] == 'EUR'
    assert result['summary']['total_value'] == 100
    assert result['summary']['total_value_usd'] is None
    assert result['cash']['included'] is False
    assert result['positions'][0]['quantity'] == 2
    assert reference == before


def test_current_prices_splits_fx_and_cash_share_one_total(reference, current_quotes):
    result = valuation.value_reference(reference)
    assert result['positions'][0]['quantity'] == 8
    assert result['positions'][0]['market_value_usd'] == 640
    assert result['summary']['cash_value'] == 80
    assert result['summary']['total_value'] == 720
    assert sum(result['summary']['asset_allocation'].values()) == pytest.approx(100)
    assert result['coverage']['updated'] == 1
    assert not result['coverage']['partial']
    assert result['comparison']['currency'] == 'EUR'
    assert result['comparison']['change'] == 220  # Positions only, at current EUR/USD.


@pytest.mark.parametrize('degradation', ['stale_cache', 'old_price', 'quantity', 'missing', 'fx'])
def test_degraded_quote_is_visible_and_comparison_is_suppressed(reference, current_quotes, monkeypatch, degradation):
    quote = current_quotes
    if degradation == 'stale_cache': quote['retrieval_status'] = 'stale'
    if degradation == 'old_price': quote['quote_at'] = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    if degradation == 'quantity': quote['quantity_verified'] = False
    if degradation == 'missing': monkeypatch.setattr(valuation, 'get_quote', lambda *_a: None)
    if degradation == 'fx': monkeypatch.setattr(valuation, 'current_fx', lambda *_a: (None, {'fresh': False}))
    result = valuation.value_reference(reference)
    assert result['coverage']['partial']
    assert result['comparison'] is None
    assert result['warnings']
    if degradation == 'missing':
        assert result['positions'][0]['valuation_status'] == 'export_fallback'
        assert result['positions'][0]['market_value_usd'] == 200
    if degradation == 'fx':
        assert result['positions'][0]['market_value_usd'] is None
        assert result['coverage']['missing_values'] == 2


def test_cash_at_export_requires_same_date_and_currency(reference):
    reference['cash']['asof'] = '2021-01-18T12:00:00'
    assert valuation.value_reference(reference, 'export')['summary']['total_value'] == 140
    reference['cash']['currency'] = 'CHF'
    assert valuation.value_reference(reference, 'export')['summary']['total_value'] == 100


def test_date_uses_original_filename_not_upload_prefix():
    assert valuation.export_date_from_name('20251213_100557_Positions_08-déc.-2025_11_35_09.csv') == '2025-12-08'
    assert valuation.export_date_from_name('20251213_100557_legacy.csv') is None


def test_unknown_date_does_not_promote_unverified_quantity(reference, current_quotes):
    reference['export_date'] = None
    current_quotes['quantity_verified'] = False
    result = valuation.value_reference(reference)
    assert result['coverage']['partial']
    assert result['positions'][0]['quantity'] == 2


def test_csv_selection_and_cash_are_isolated_and_duplicates_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(valuation, 'ROOT', tmp_path)
    monkeypatch.setattr('services.instruments_registry.resolve', lambda *_a, **_kw: {})
    for user in ('alice', 'bob'):
        data = tmp_path / 'data/users' / user / 'saxobank/data'
        data.mkdir(parents=True)
        path = data / 'Positions_18-janv.-2021.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Instrument', 'Symbol', 'Quantity', 'Market Value EUR', 'Currency', 'Position ID'])
            writer.writerows([['Apple', 'AAPL:xnas', 3, 150, 'USD', ''],
                              ['Apple', 'AAPL:xnas', 1, 50, 'USD', '1'],
                              ['Apple', 'AAPL:xnas', 2, 100, 'USD', '2']])
        (data.parents[1] / 'config.json').write_text(json.dumps({'sources': {'bourse': {'selected_csv_file': path.name}}}))
        cash_dir = data.parent / 'cash'
        cash_dir.mkdir()
        (cash_dir / f'{path.name}_cash.json').write_text(json.dumps({'cash_amount': 12 if user == 'alice' else 99, 'currency': 'EUR'}))
    ref = valuation.read_reference('alice', None)
    assert len(ref['positions']) == 2
    assert sum(p['market_value'] for p in ref['positions']) == 150
    assert ref['cash']['amount'] == 12
    assert valuation.read_reference('bob', None)['cash']['amount'] == 99
    with pytest.raises(FileNotFoundError): valuation.resolve_csv('nobody', None)
    with pytest.raises(FileNotFoundError): valuation.resolve_csv('alice', 'missing.csv')
    with pytest.raises(ValueError): valuation.resolve_csv('alice', '../bob.csv')
    with pytest.raises(ValueError): valuation.resolve_csv('../bob', None)


def test_currency_from_value_column_and_raw_export_price_are_preserved(monkeypatch):
    monkeypatch.setattr('services.instruments_registry.resolve', lambda *_a, **_kw: {})
    connector = SaxoImportConnector()
    frame = connector._normalize_dataframe(pd.DataFrame([{'Instrument': 'Apple', 'Symbol': 'AAPL', 'Quantity': 2,
        'Market Value USD': 100, 'Currency': 'CHF', 'Prix actuel': 50}]))
    row = frame.iloc[0].copy()
    row['Account Currency'] = frame.attrs['value_currency']
    item = connector._process_position(row, convert_values=False)
    assert item['account_base_currency'] == 'USD'
    assert item['market_value'] == 100
    assert item['export_price'] == 50
    assert item['market_value_usd'] is None


def test_current_and_export_downloads_reuse_valuation(reference, current_quotes, monkeypatch):
    monkeypatch.setattr(valuation, 'read_reference', lambda *_a: deepcopy(reference))
    current = exports.build_saxo_export_data('alice', valuation_mode='current')
    historical = exports.build_saxo_export_data('alice', valuation_mode='export')
    assert current['summary']['total_value'] == 720
    assert historical['summary']['total_value'] == 100
    csv_text = ExportFormatter('saxo').to_csv(historical)
    assert 'Value EUR' in csv_text
    assert 'Market Value USD' not in csv_text
    assert '2021-01-18' in csv_text
    assert 'EUR' in ExportFormatter('saxo').to_markdown(historical)


def test_quote_cache_coalesces_manual_refresh_and_retains_stale_real_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(quotes, 'CACHE_DIR', tmp_path)
    clock = [10000]
    monkeypatch.setattr(quotes.time, 'time', lambda: clock[0])
    calls = []
    def fetch(*args):
        calls.append(args)
        return {'price': 50, 'quote_at': '2021-01-18', 'currency': 'EUR'}
    monkeypatch.setattr(quotes, 'fetch_yahoo_quote', fetch)
    assert quotes.get_quote('AAPL', '2021-01-18')['price'] == 50
    assert quotes.get_quote('AAPL', '2021-01-18', force=True)['price'] == 50
    assert len(calls) == 1
    clock[0] += 2000
    def failed(*_a): raise ValueError('Provider offline')
    monkeypatch.setattr(quotes, 'fetch_yahoo_quote', failed)
    result = quotes.get_quote('AAPL', '2021-01-18')
    assert result['price'] == 50
    assert result['retrieval_status'] == 'stale'


def test_exchange_suffix_takes_precedence_over_isin_country():
    assert quotes.yahoo_symbol('SLHn:xvtx') == 'SLHN.SW'
    assert quotes.yahoo_symbol('AGGS:xvtx') == 'AGGS.SW'
    assert quotes.yahoo_symbol('MSFT:xnas') == 'MSFT'
    with pytest.raises(ValueError): quotes.yahoo_symbol('AAPL:unknown')


def test_roche_replacement_is_dated_documented_and_quantity_preserving(monkeypatch):
    import yfinance
    calls = []
    history = pd.DataFrame({'Stock Splits': [0, 0]}, index=pd.to_datetime(['2026-01-19', '2026-10-05']))
    stamp = int(datetime.now(timezone.utc).timestamp())
    class Ticker:
        def __init__(self, symbol): calls.append(symbol)
        def history(self, **kwargs): return history
        def get_history_metadata(self):
            return {'currency': 'CHF', 'regularMarketPrice': 348.9, 'regularMarketTime': stamp}
    monkeypatch.setattr(yfinance, 'Ticker', Ticker)
    quote = quotes.fetch_yahoo_quote('ROG.SW', '2026-01-18')
    assert calls == ['ROP.SW']
    assert quote['split_factor'] == 1
    assert quote['quantity_verified']
    assert quote['corporate_actions'][0]['original_symbol'] == 'ROG.SW'
    assert quote['corporate_actions'][0]['quantity_ratio'] == 1
    assert quote['corporate_actions'][0]['effective_date'] == '2026-03-17'
    assert quote['corporate_actions'][0]['source'].startswith('https://www.roche.com/')
    class BeforeExchange(datetime):
        @classmethod
        def now(cls, tz=None): return cls(2026, 3, 16, tzinfo=timezone.utc)
    monkeypatch.setattr(quotes, 'datetime', BeforeExchange)
    quote = quotes.fetch_yahoo_quote('ROG.SW', '2026-01-18')
    assert calls[-1] == 'ROG.SW'
    assert quote['corporate_actions'] == []


def test_provider_retains_market_timestamp_unadjusted_price_and_pence(monkeypatch):
    import yfinance
    history = pd.DataFrame({'Stock Splits': [0, 4, 0]}, index=pd.to_datetime(['2021-01-18', '2022-01-01', '2026-01-01']))
    calls = []
    stamp = int(datetime.now(timezone.utc).timestamp())
    class Ticker:
        def history(self, **kwargs):
            calls.append(kwargs)
            return history
        def get_history_metadata(self):
            return {'currency': 'GBp', 'regularMarketPrice': 1200, 'regularMarketTime': stamp,
                    'currentTradingPeriod': {'regular': {'start': stamp - 100, 'end': stamp + 100}}}
    monkeypatch.setattr(yfinance, 'Ticker', lambda _symbol: Ticker())
    quote = quotes.fetch_yahoo_quote('TEST.L', '2021-01-18')
    assert quote['price'] == 12
    assert quote['currency'] == 'GBP'
    assert quote['split_factor'] == 4
    assert quote['quantity_verified'] is False  # Le CSV de 2021 nécessite d'autres contrôles d'opérations sur titres.
    assert quote['quote_type'] == 'intraday'
    assert calls[0]['auto_adjust'] is False
    assert calls[0]['actions'] is True
    assert quote['quote_at'] == datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def test_provider_failure_never_generates_a_quote(monkeypatch):
    import yfinance
    class Ticker:
        def history(self, **_kw):
            return pd.DataFrame()
    monkeypatch.setattr(yfinance, 'Ticker', lambda _symbol: Ticker())
    with pytest.raises(ValueError, match='No real quote'):
        quotes.fetch_yahoo_quote('TEST', None)


def test_bonds_are_not_valued_as_quantity_times_equity_quote(reference, current_quotes):
    reference['positions'][0]['asset_class'] = 'Bond'
    result = valuation.value_reference(reference)
    assert result['positions'][0]['valuation_status'] == 'export_fallback'
    assert result['coverage']['partial']
    assert any('instrument-specific' in warning for warning in result['warnings'])


def test_mixed_export_currencies_are_not_silently_added(reference):
    other = deepcopy(reference['positions'][0])
    other['account_base_currency'] = 'CHF'
    reference['positions'].append(other)
    with pytest.raises(ValueError, match='multiple valuation currencies'):
        valuation.value_reference(reference, 'export')


@pytest.mark.parametrize('missing', [False, True])
def test_current_fx_requires_verified_currency_quotes(monkeypatch, missing):
    calls = []
    def verified(currency):
        calls.append(currency)
        if missing and currency == 'EUR':
            raise ValueError('Verified FX unavailable')
        return {'EUR': 1.2, 'CHF': 1.5}[currency]
    monkeypatch.setattr(valuation.fx_service, 'get_verified_rate', verified)
    monkeypatch.setattr(valuation.fx_service, 'get_cache_info', lambda: {
        'source': 'exchange-rate-api', 'last_update': '2026-10-06', 'source_updated': '2026-10-05'})
    rate, info = valuation.current_fx('EUR', 'CHF')
    if missing:
        assert rate is None
        assert not info['fresh']
    else:
        assert rate == pytest.approx(0.8)
        assert info['fresh']
        assert calls == ['EUR', 'CHF']
