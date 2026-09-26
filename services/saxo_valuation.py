"""Explicit currency contracts for Saxo account values and instrument prices."""

import math

from services.fx_service import get_verified_rate


def value_saxo_portfolio_usd(positions, balances, rate_getter=None):
    rate_getter = rate_getter or get_verified_rate
    account_currency = balances.get('Currency')
    if not account_currency:
        raise ValueError('Saxo account currency is missing')

    def converted(value, currency):
        if value is None or not currency:
            raise ValueError('Saxo amount or currency is missing')
        number = float(value)
        rate = float(rate_getter(currency.upper()))
        if not math.isfinite(number) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('Invalid Saxo amount or FX rate')
        return number * rate

    cash = converted(balances.get('CashBalance'), account_currency)
    total = converted(balances.get('TotalValue'), account_currency)
    valued = []
    for position in positions:
        result = dict(position)
        result['market_value_native'] = position['market_value']
        result['market_value'] = converted(position['market_value'], position.get('market_value_currency'))
        result['market_value_usd'] = result['market_value']
        result['valuation_currency'] = 'USD'
        result['pnl'] = (converted(position['pnl'], position.get('pnl_currency'))
                         if position.get('pnl') is not None else None)
        # Open/current prices stay in instrument currency, matching price histories.
        valued.append(result)
    return valued, cash, total
