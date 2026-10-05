"""Shared Saxo CSV valuation: immutable export amounts and dated market quotes."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

from connectors.saxo_import import SaxoImportConnector
from services import fx_service
from services.saxo_quote_service import LISTING_ALIASES, get_quote, yahoo_symbol

ROOT = Path(__file__).resolve().parents[1]
MONTHS = dict(zip(('janv', 'fevr', 'mars', 'avr', 'mai', 'juin', 'juil', 'aout', 'sept', 'oct', 'nov', 'dec'), range(1, 13)))


def resolve_csv(user: str, file_key: str | None) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', user):
        raise ValueError('Invalid user identifier')
    directory = ROOT / 'data/users' / user / 'saxobank/data'
    if directory.resolve() != directory.absolute():
        raise ValueError('CSV directory must belong to the selected user')
    if not file_key:
        try:
            config = json.loads((ROOT / 'data/users' / user / 'config.json').read_text(encoding='utf-8'))
            file_key = config.get('sources', {}).get('bourse', {}).get('selected_csv_file')
        except (OSError, ValueError):
            pass
    if file_key:
        if Path(file_key).name != file_key or '/' in file_key or '\\' in file_key:
            raise ValueError('Invalid CSV file key')
        path = directory / file_key
        if path.suffix.lower() != '.csv' or not path.is_file() or path.resolve().parent != directory.resolve():
            raise FileNotFoundError('Selected Saxo CSV not found')
        return path
    files = list(directory.glob('*.csv'))
    if not files:
        raise FileNotFoundError('No Saxo CSV found')
    path = max(files, key=lambda p: p.stat().st_mtime)
    if path.resolve().parent != directory.resolve():
        raise ValueError('Invalid CSV path')
    return path


def export_date_from_name(name: str) -> str | None:
    # Le préfixe YYYYMMDD_HHMMSS correspond à l'import, pas à l'export.
    text = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    match = re.search(r'positions[_ -](\d{1,2})-([a-z]+)\.?-(\d{4})', text)
    if not match or match[2] not in MONTHS:
        return None
    try:
        return date(int(match[3]), MONTHS[match[2]], int(match[1])).isoformat()
    except ValueError:
        return None


def read_reference(user: str, file_key: str | None) -> dict:
    path = resolve_csv(user, file_key)
    parsed = SaxoImportConnector().process_saxo_file(path, user_id=user, convert_values=False)
    if parsed.get('errors'):
        raise ValueError('Some CSV rows could not be parsed; valuation unavailable')
    positions = parsed.get('positions', [])
    # Un agrégat sans Position ID est supprimé seulement si des lots détaillés existent.
    groups = defaultdict(list)
    for position in positions:
        groups[(position['symbol'], position['account_base_currency'])].append(position)
    deduped = []
    for group in groups.values():
        detailed = [p for p in group if p['position_id'] != p['symbol']]
        deduped.extend(detailed or group)
    if not deduped:
        raise ValueError('CSV contains no usable positions')
    cash_path = path.parent.parent / 'cash' / f'{path.name}_cash.json'
    cash = {'amount': 0.0, 'currency': 'USD', 'asof': None, 'known': False}
    if cash_path.is_file():
        if cash_path.resolve() != cash_path.absolute():
            raise ValueError('Cash file must belong to the selected CSV')
        raw = json.loads(cash_path.read_text(encoding='utf-8'))
        cash.update(amount=float(raw.get('cash_amount', 0) or 0), currency=str(raw.get('currency') or 'USD').upper(),
                    asof=raw.get('last_updated'), known=True)
    return {'file_key': path.name, 'positions': deduped, 'cash': cash,
            'export_date': export_date_from_name(path.name),
            'imported_at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def current_fx(currency: str, target: str) -> tuple[float | None, dict]:
    if currency == target:
        return 1.0, {'source': 'identity', 'fresh': True}
    try:
        rate = fx_service.get_verified_rate(currency) / fx_service.get_verified_rate(target)
    except ValueError:
        return None, {'source': 'unavailable', 'fresh': False}
    metadata = fx_service.get_cache_info()
    if not math.isfinite(rate) or rate <= 0:
        return None, {'source': 'unavailable', 'fresh': False}
    return rate, {'source': metadata['source'], 'fetched_at': metadata['last_update'],
                  'source_updated': metadata['source_updated'], 'fresh': True}



def value_reference(reference: dict, mode: str = 'current', currency: str = 'USD', force: bool = False) -> dict:
    if mode not in {'current', 'export'}:
        raise ValueError('Unknown valuation mode')
    positions = reference['positions']
    export_date = reference['export_date']
    base_currencies = {p['account_base_currency'] for p in positions}
    if mode == 'export':
        if len(base_currencies) != 1:
            raise ValueError('Export contains multiple valuation currencies; no comparable total available')
        currency = next(iter(base_currencies))
    if not re.fullmatch('[A-Z]{3}', currency):
        raise ValueError('Invalid display currency')
    quotes = {}
    if mode == 'current':
        supported_symbols = {p['symbol'] for p in positions if p['asset_class'] not in {'Bond', 'CFD', 'Option', 'Warrant'}}
        def fetch(symbol):
            try:
                alias = LISTING_ALIASES.get(symbol)
                if alias and any(p.get('isin') != alias['isin'] for p in positions if p['symbol'] == symbol):
                    return symbol, None
                quote = get_quote(yahoo_symbol(symbol), export_date, force)
                if alias and quote and quote['currency'] != alias['currency']:
                    return symbol, None
                return symbol, quote
            except (ValueError, OSError, TimeoutError):
                return symbol, None
        with ThreadPoolExecutor(max_workers=4) as pool:
            quotes = dict(pool.map(fetch, sorted(supported_symbols)))
    fx_cache = {}
    def convert(amount, source):
        if source not in fx_cache:
            fx_cache[source] = current_fx(source, currency)
        rate, metadata = fx_cache[source]
        return (amount * rate if rate is not None else None), metadata
    result = []
    fresh_count = 0
    warnings = []
    if not export_date:
        warnings.append('Export valuation date is unknown; imported quantities cannot be verified against splits.')
    for raw in positions:
        item = dict(raw)
        item['export_quantity'] = raw['quantity']
        item['export_value'] = raw['market_value']
        item['export_currency'] = raw['account_base_currency']
        quote = quotes.get(raw['symbol'])
        if mode == 'export':
            value = raw['market_value']
            item.update(valuation_status='export', quote_at=export_date, price_source='csv', current_price=raw.get('export_price'))
        elif quote:
            factor = quote['split_factor'] if export_date else 1.0
            item['quantity'] = raw['quantity'] * factor
            item['currency'] = quote['currency']
            value, fx = convert(item['quantity'] * quote['price'], quote['currency'])
            quote_age = (datetime.now(timezone.utc) - datetime.fromisoformat(quote['quote_at'])).total_seconds()
            complete = quote['retrieval_status'] == 'fresh' and -300 <= quote_age <= 7 * 86400 and quote['quantity_verified'] and fx['fresh'] and value is not None
            fresh_count += int(complete)
            item.update(valuation_status='current' if complete else 'unverified_or_stale',
                        quote_at=quote['quote_at'], fetched_at=quote['fetched_at'], price_source=quote['source'],
                        current_price=quote['price'], quote_currency=quote['currency'], quote_type=quote['quote_type'],
                        split_factor=factor, fx=fx, quantity_verified=quote['quantity_verified'],
                        quote_symbol=quote.get('symbol'), corporate_actions=quote.get('corporate_actions', []))
            if quote.get('refresh_failed'):
                warnings.append(f"Price refresh failed for {raw['symbol']}; the previous dated quote is retained.")
        else:
            # Référence CSV visible comme secours, jamais présentée comme cours actuel.
            value, fx = convert(raw['market_value'], raw['account_base_currency'])
            item.update(valuation_status='export_fallback', quote_at=export_date, price_source='csv', current_price=None, fx=fx)
            if raw['asset_class'] in {'Bond', 'CFD', 'Option', 'Warrant'}:
                warnings.append(f"{raw['symbol']} requires an instrument-specific valuation; its export value is retained.")
        item['market_value_display'] = value
        item['market_value_usd'] = value if currency == 'USD' else None
        item['display_currency'] = currency
        result.append(item)
    cash = dict(reference['cash'])
    cash_value = 0.0
    cash['included'] = False
    if mode == 'export':
        # Un solde saisi après l'export ne constitue pas un solde historique.
        if cash['known'] and cash['asof'] and export_date and str(cash['asof'])[:10] == export_date and cash['currency'] == currency:
            cash_value = cash['amount']
            cash['included'] = True
        else:
            warnings.append('Cash at export is unverified; the historical total covers positions only.')
    elif cash['known']:
        cash_value, fx = convert(cash['amount'], cash['currency'])
        cash.update(included=cash_value is not None, fx=fx)
        if not fx['fresh']:
            warnings.append('Cash conversion uses stale or reference FX rates.')
    else:
        warnings.append('Cash balance is unavailable; the total covers positions only.')
    cash['value_display'] = cash_value
    missing_values = sum(p['market_value_display'] is None for p in result) + int(cash_value is None)
    subtotal = sum(p['market_value_display'] or 0 for p in result)
    total = subtotal + (cash_value or 0)
    allocation = defaultdict(float)
    exposure = defaultdict(float)
    for item in result:
        value = item['market_value_display'] or 0
        allocation[item['asset_class']] += value
        exposure[item['currency']] += value
        item['weight'] = value / total * 100 if total else 0
    if cash_value:
        allocation['Cash'] += cash_value
    partial = missing_values > 0 or (mode == 'current' and (
        fresh_count != len(result) or not cash['known'] or not cash.get('fx', {'fresh': True})['fresh']))
    if partial:
        warnings.append('Valuation is partial: some positions use old export values, stale quotes or unverified quantities/FX.')
    comparison = None
    if mode == 'current' and not partial and len(base_currencies) == 1 and export_date:
        baseline_currency = next(iter(base_currencies))
        original = sum(p['market_value'] for p in positions)
        rate, fx = current_fx(currency, baseline_currency)
        if rate is not None and fx['fresh'] and original > 0:
            difference = subtotal * rate - original
            comparison = {'currency': baseline_currency, 'change': difference,
                          'change_pct': difference / original * 100, 'scope': 'positions_only'}
    timestamps = [p['quote_at'] for p in result if p.get('quote_at')]
    return {
        'portfolio_id': 'saxo_csv_valuation', 'file_key': reference['file_key'], 'source': 'csv',
        'mode': mode, 'currency': currency, 'export_date': export_date,
        'export_date_provenance': 'filename_inferred' if export_date else 'unknown',
        'imported_at': reference['imported_at'], 'valued_at': datetime.now(timezone.utc).isoformat(),
        'oldest_quote_at': min(timestamps) if timestamps else None,
        'positions': result, 'cash': cash, 'totalValueIncludesCash': True,
        'coverage': {'updated': fresh_count, 'positions': len(result), 'missing_values': missing_values, 'partial': partial},
        'warnings': warnings,
        'comparison': comparison,
        'summary': {'total_value': total, 'total_value_usd': total if currency == 'USD' else None,
                    'positions_value': subtotal, 'cash_value': cash_value, 'total_positions': len({p['symbol'] for p in result}),
                    'asset_allocation': {k: v / total * 100 if total else 0 for k, v in allocation.items()},
                    'currency_exposure': {k: v / total * 100 if total else 0 for k, v in exposure.items()},
                    'top_holdings': sorted(result, key=lambda p: p['market_value_display'] or 0, reverse=True)[:10]},
    }


def get_valuation(user: str, file_key: str | None = None, mode: str = 'current', currency: str = 'USD', force: bool = False) -> dict:
    return value_reference(read_reference(user, file_key), mode, currency, force)
