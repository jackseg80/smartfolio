import time

import pytest

from api.saxo_auth_router import _normalize_positions
from services.saxo_valuation import value_saxo_portfolio_usd
from services import fx_service


def test_live_account_values_and_instrument_prices_use_distinct_currencies():
    raw = [{
        'PositionBase': {'Uic': 1, 'Amount': 10, 'AssetType': 'Stock', 'OpenPrice': 80},
        'PositionView': {'MarketValue': 1000, 'MarketValueInBaseCurrency': 800,
                         'CurrentPrice': 100, 'ExposureCurrency': 'USD',
                         'ProfitLossOnTradeInBaseCurrency': 160},
    }]
    positions = _normalize_positions(raw, {1: {'symbol': 'AAPL:xnas', 'currency': 'USD'}}, account_currency='CHF')
    valued, cash, total = value_saxo_portfolio_usd(
        positions, {'Currency': 'CHF', 'CashBalance': 100, 'TotalValue': 900},
        rate_getter=lambda currency: {'CHF': 1.25, 'USD': 1.0}[currency],
    )
    assert valued[0]['market_value_usd'] == 1000
    assert valued[0]['current_price'] == 100
    assert valued[0]['avg_price'] == 80
    assert valued[0]['currency'] == 'USD'
    assert valued[0]['pnl'] == 200
    assert (cash, total) == (125, 1125)


def test_native_value_is_converted_by_instrument_currency_if_base_value_missing():
    raw = [{'DisplayAndFormat': {'Symbol': 'NESN:xvtx'}, 'MarketValue': 100,
            'Amount': 1, 'Currency': 'CHF', 'SinglePositionBase': {'OpenPrice': 90}}]
    positions = _normalize_positions(raw, account_currency='EUR')
    valued, cash, total = value_saxo_portfolio_usd(
        positions, {'Currency': 'EUR', 'CashBalance': 10, 'TotalValue': 100},
        rate_getter=lambda currency: {'CHF': 1.2, 'EUR': 1.1}[currency],
    )
    assert valued[0]['symbol'] == 'NESN:xvtx'
    assert valued[0]['market_value_usd'] == 120
    assert cash == 11
    assert total == pytest.approx(110)


def test_missing_saxo_account_currency_is_not_assumed_to_be_eur():
    with pytest.raises(ValueError, match='account currency'):
        value_saxo_portfolio_usd([], {'CashBalance': 10, 'TotalValue': 10})


def test_verified_fx_rejects_reference_fallback_rates(monkeypatch):
    monkeypatch.setattr(fx_service, '_ensure_rates_fresh', lambda: None)
    monkeypatch.setattr(fx_service, '_RATES_CACHE_TIMESTAMP', time.time())
    monkeypatch.setattr(fx_service, '_RATES_SOURCE_TIMESTAMP', time.time())
    monkeypatch.setattr(fx_service, '_VERIFIED_CURRENCIES', set())
    with pytest.raises(ValueError, match='Verified FX rate unavailable'):
        fx_service.get_verified_rate('CHF')
    assert fx_service.get_verified_rate('USD') == 1
