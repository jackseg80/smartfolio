"""A strict, private, request-scoped snapshot for manual decision support."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def number(value, label):
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} is missing or invalid")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} is missing or invalid") from None
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def csv_value_currency(content: bytes):
    """Read the original header before the legacy connector removes its unit."""
    for encoding in ('utf-8-sig', 'cp1252', 'latin1'):
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            continue
        for delimiter in (',', ';', '\t'):
            header = next(csv.reader(text.splitlines(), delimiter=delimiter), [])
            if len(header) < 2:
                continue
            currencies = []
            for label in header:
                match = re.fullmatch(r'(?:Valeur actuelle|Market Value)\s*\(([A-Z]{3})\)', label.strip(), re.I)
                if match:
                    currencies.append(match[1].upper())
            if len(currencies) == 1:
                return currencies[0]
            return None
    return None


def normalize_rows(rows, valuation_currency, rate):
    """Keep ISIN, exact listing and quote currency separate from value currency."""
    normalized = []
    identities = set()
    for row in rows:
        symbol = str(row.get('symbol') or row.get('instrument_id') or '').strip()
        isin = str(row.get('isin') or '').upper().strip()
        identity = str(row.get('position_id') or '') + '|' + isin + '|' + symbol
        if not symbol or identity in identities:
            raise ValueError('Missing or duplicate position identity; holdings cannot be silently merged')
        identities.add(identity)
        quantity = number(row.get('quantity'), 'Position quantity')
        native = None if row.get('_valuation_missing') is True else number(row.get('market_value'), 'Position value')
        if quantity <= 0 or (native is not None and native < 0):
            raise ValueError('This analysis requires long positions with non-negative values')
        usd = number(native * rate, 'USD position value') if native is not None else None
        normalized.append(dict(
            id=hashlib.sha256(identity.encode()).hexdigest()[:20], symbol=symbol, isin=isin or None,
            name=str(row.get('name') or row.get('instrument') or row.get('asset_name') or symbol),
            asset_class=str(row.get('asset_class') or 'Unknown'),
            quote_currency=row.get('currency'), exchange=row.get('exchange'), quantity=quantity,
            value_native=native, value_usd=usd, valuation_currency=valuation_currency,
            valuation_reason='Selected CSV valuation is missing; never treated as zero' if native is None else 'Explicit selected CSV value',
            acquisition_date=row.get('purchase_date') or row.get('opened_at'),
        ))
    return normalized


def load_csv_snapshot(user, file_key=None, root=None, rate_getter=None, row_loader=None):
    from api.services.user_fs import UserScopedFS
    from services.fx_service import get_verified_rate
    root = root or Path(__file__).resolve().parents[3]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', user):
        raise ValueError('Invalid authenticated user identifier')
    fs = UserScopedFS(str(root), user)
    config = json.loads(Path(fs.get_path('config.json')).read_text(encoding='utf-8'))
    selected = config.get('sources', {}).get('bourse', {})
    active = selected.get('active_source')
    if active not in ('saxobank', 'saxobank_csv'):
        raise ValueError('The active stock source is not Saxo CSV')
    effective = selected.get('selected_csv_file')
    if not isinstance(effective, str) or not effective or '/' in effective or '\\' in effective or Path(effective).suffix.lower() != '.csv':
        raise ValueError('No valid selected Saxo CSV; no other file is substituted')
    if file_key and effective != file_key:
        raise ValueError('The requested CSV differs from the active selection; refresh the source')
    path = Path(fs.get_path('saxobank/data/' + effective))
    if not path.is_file():
        raise ValueError('The selected Saxo CSV is unavailable; no other file is substituted')
    content = path.read_bytes()
    currency = csv_value_currency(content)
    if not currency:
        raise ValueError('The selected CSV value currency is not explicit in its header')
    rate = number((rate_getter or get_verified_rate)(currency), 'Verified value FX rate')
    if rate <= 0:
        raise ValueError('Verified value FX rate must be positive')
    if row_loader is None:
        from connectors.saxo_import import SaxoImportConnector
        connector = SaxoImportConnector()
        frame = connector._load_file(path)
        # Request-local connector: replace its indicative conversion, not a global service.
        connector._convert_to_usd = lambda amount, unused_currency: amount * rate
        rows = []
        for _, original in frame.iterrows():
            placeholders = {'', 'nan', 'none', 'n/a', '-', '—'}
            identifiers = [str(original.get(k, '')).strip() for k in ('Symbol', 'ISIN')]
            if not any(value.lower() not in placeholders for value in identifiers):
                continue  # Report summary lines do not identify a tradable position.
            # pandas 3 infers a strict string dtype; numeric normalization needs object cells.
            row = original.astype(object).copy()
            missing_value = False
            for field in ('Quantity', 'Market Value'):
                raw = str(row.get(field, '')).replace('\u00a0', '').replace(' ', '').replace("'", '').replace(',', '.')
                if field == 'Market Value' and raw.lower() in placeholders:
                    missing_value = True
                    row[field] = 0  # Only to retain the identity through the old connector.
                else:
                    row[field] = number(raw, 'CSV ' + field)
            if row['Quantity'] <= 0 or row['Market Value'] < 0:
                raise ValueError('Zero/short quantities and negative values require a separate analysis; no row is dropped')
            row['Account Currency'] = currency
            parsed = connector._process_position(row, user_id=user)
            if not parsed:
                raise ValueError('An identified CSV position was rejected; no partial portfolio conclusion')
            source_currency = str(original.get('Currency', '')).strip()
            parsed['currency'] = source_currency.upper() if source_currency.lower() not in placeholders else None
            if missing_value:
                parsed['market_value'] = None
                parsed['_valuation_missing'] = True
            rows.append(parsed)
    else:
        rows = row_loader(user, effective)
    positions = normalize_rows(rows, currency, rate)
    # Do not display the connector's old indicative conversion as verified data.
    cash_path = Path(fs.get_path('saxobank/cash/' + effective + '_cash.json'))
    cash = dict(value_usd=None, currency=None, as_of=None, status='unavailable', reason='No cash record for the selected CSV')
    if cash_path.is_file():
        saved = json.loads(cash_path.read_text(encoding='utf-8'))
        try:
            cash_currency = saved['currency']
            amount = number(saved.get('cash_amount'), 'Cash amount')
            cash_rate = number((rate_getter or get_verified_rate)(cash_currency), 'Cash FX rate')
            if cash_rate <= 0:
                raise ValueError('Cash FX rate must be positive')
            cash = dict(value_usd=amount * cash_rate, currency=cash_currency, as_of=saved.get('last_updated'),
                        status='available', reason='Saved cash for the same selected CSV; not a live bank balance')
        except (ValueError, KeyError, TypeError):
            cash['reason'] = 'Saved cash cannot be valued with an explicit currency and verified FX rate'
    # A file timestamp is not a financial valuation date.
    imported = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    match = re.match(r'^(\d{8}_\d{6})_', effective)
    if match:
        try:
            imported = datetime.strptime(match[1], '%Y%m%d_%H%M%S').isoformat()
        except ValueError:
            pass
    fx_as_of = None
    if currency != 'USD' or cash.get('currency') not in (None, 'USD'):
        from services import fx_service
        timestamp = getattr(fx_service, '_RATES_SOURCE_TIMESTAMP', 0)
        fx_as_of = datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp else None
    metadata = dict(user_id=user, source='saxobank_csv', file_key=effective, imported_at=imported,
                    valuation_as_of=None, valuation_currency=currency, reporting_currency='USD',
                    fx_as_of=fx_as_of, fx_rate=rate,
                    warnings=['The import date is not a confirmed valuation date. Source values are converted with current verified FX, not historical FX.',
                              'Saved cash and positions are dated separately. Confirm balances before using a scenario.'])
    missing_count = sum(p['value_usd'] is None for p in positions)
    metadata['unvalued_positions'] = missing_count
    if missing_count:
        metadata['warnings'].append(f'{missing_count} positions have no source valuation. Coverage describes the valued subset only; full-portfolio sector conclusions and scenarios are unavailable.')
    if path.read_bytes() != content:
        raise ValueError('The selected CSV changed during analysis; scan again')
    scope = hashlib.sha256(content + json.dumps(cash, sort_keys=True).encode() + str(rate).encode() + user.encode() + effective.encode()).hexdigest()
    return dict(positions=positions, cash=cash, context=metadata, snapshot_id=scope)


async def load_snapshot(user, source, file_key, request, api_base):
    """No source/CSV fallback and no global cache of private snapshots."""
    root = Path(__file__).resolve().parents[3]
    from api.services.user_fs import UserScopedFS
    if not re.fullmatch(r'[A-Za-z0-9_-]+', user):
        raise ValueError('Invalid authenticated user identifier')
    fs = UserScopedFS(str(root), user)
    config = json.loads(Path(fs.get_path('config.json')).read_text(encoding='utf-8'))
    active = config.get('sources', {}).get('bourse', {}).get('active_source')
    active = 'saxobank_csv' if active == 'saxobank' else active
    source = 'saxobank_csv' if source == 'saxobank' else source
    if source and source != active:
        raise ValueError('The requested source differs from the active stock source; refresh the source')
    if active == 'saxobank_csv':
        import asyncio
        return await asyncio.to_thread(load_csv_snapshot, user, file_key)
    if active not in ('manual_bourse', 'saxobank_api') or file_key:
        raise ValueError('Select an explicit supported stock source')
    if active == 'manual_bourse':
        from services.sources import source_registry
        provider = source_registry.get_source(active, user, root)
        if not provider:
            raise ValueError('The selected manual source is unavailable')
        items = await provider.get_balances()
        rows = [dict(symbol=i.symbol, quantity=i.amount, market_value=i.value_usd,
                     currency=i.currency, asset_class=i.asset_class, name=i.alias) for i in items]
    else:
        import httpx
        from api.ml_bourse_endpoints import _forward_authenticated_headers
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(api_base + '/api/saxo/api-positions', headers=_forward_authenticated_headers(request, user))
            if response.status_code == 401:
                raise ValueError('Saxo Bank is not connected')
            response.raise_for_status()
            rows = response.json().get('data', {}).get('positions', [])
        # Only accept an explicit value-currency contract from the API.
        if any(r.get('valuation_currency') != 'USD' for r in rows):
            raise ValueError('The Saxo API positions do not expose an explicit USD valuation contract')
    positions = normalize_rows(rows, 'USD', 1)
    context = dict(user_id=user, source=active, file_key=None, imported_at=None, valuation_as_of=None,
                   valuation_currency='USD', reporting_currency='USD', fx_as_of=None,
                   warnings=['Cash and financial valuation dates are unavailable for this source.'])
    return dict(positions=positions, context=context,
                cash=dict(value_usd=None, status='unavailable', reason='No source-consistent cash record', currency=None, as_of=None),
                snapshot_id=hashlib.sha256(json.dumps(positions, sort_keys=True).encode()).hexdigest())
