"""Authentic, dated stock quotes. No generated data or crypto fallback."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from filelock import FileLock

logger = logging.getLogger(__name__)
CACHE_DIR = Path('data/cache/saxo_quotes')
QUOTE_TTL = 1800
RETRY_DELAY = 60

# Le MIC du CSV est prioritaire sur le pays de domiciliation de l'ISIN.
MIC_SUFFIX = {
    'xnas': '', 'xnys': '', 'arcx': '', 'xase': '',
    'xvtx': '.SW', 'xswx': '.SW', 'xetr': '.DE', 'xfra': '.F',
    'xlon': '.L', 'xpar': '.PA', 'xams': '.AS', 'xmil': '.MI',
    'xbru': '.BR', 'xmad': '.MC', 'xsto': '.ST', 'xosl': '.OL',
    'xcse': '.CO', 'xhel': '.HE', 'xtse': '.TO', 'xhkg': '.HK',
}

# Remplacement officiel 1:1, sans changer les quantités du portefeuille.
# Garder la date et la source : un simple alias permanent masquerait l'opération.
LISTING_REPLACEMENTS = {
    'ROG.SW': {
        'symbol': 'ROP.SW', 'effective_date': '2026-03-17', 'quantity_ratio': 1.0,
        'source': 'https://www.roche.com/investors/updates/inv-update-2026-03-16',
    },
}


def yahoo_symbol(symbol: str) -> str:
    base, sep, mic = symbol.partition(':')
    if not base or any(ch in base for ch in '/\\'):
        raise ValueError('Invalid instrument identifier')
    if sep:
        if mic.lower() not in MIC_SUFFIX:
            raise ValueError('Unsupported listing exchange')
        suffix = MIC_SUFFIX[mic.lower()]
        from services.ml.bourse.currency_detector import CurrencyExchangeDetector
        base = CurrencyExchangeDetector.SYMBOL_TRANSFORMATIONS.get(base, base)
        return base.upper() + suffix
    return base.upper()


def fetch_yahoo_quote(symbol: str, export_date: str | None) -> dict:
    import yfinance as yf

    corporate_actions = []
    replacement = LISTING_REPLACEMENTS.get(symbol)
    if replacement and datetime.now(timezone.utc).date() >= date.fromisoformat(replacement['effective_date']):
        corporate_actions.append(dict(replacement, original_symbol=symbol))
        symbol = replacement['symbol']
    ticker = yf.Ticker(symbol)
    # Historique non ajusté ; les splits servent uniquement à corriger les quantités.
    kwargs = {'interval': '1d', 'auto_adjust': False, 'actions': True, 'timeout': 10}
    if export_date:
        start = date.fromisoformat(export_date)
        if start > datetime.now(timezone.utc).date():
            raise ValueError('Export date is in the future')
        kwargs['start'] = start.isoformat()
    else:
        kwargs['period'] = '5d'
    history = ticker.history(**kwargs)
    if history.empty:
        raise ValueError('No real quote available')
    metadata = ticker.get_history_metadata()
    currency = metadata.get('currency')
    price = metadata.get('regularMarketPrice')
    stamp = metadata.get('regularMarketTime')
    if not currency or not stamp or price is None or not math.isfinite(float(price)) or float(price) <= 0:
        raise ValueError('Quote currency, price or market timestamp unavailable')
    factor = 1.0
    if export_date:
        if 'Stock Splits' not in history:
            raise ValueError('Split history unavailable')
        for ts, ratio in history['Stock Splits'].items():
            if date.fromisoformat(export_date) < ts.date() <= datetime.fromtimestamp(int(stamp), timezone.utc).date() and ratio:
                if not math.isfinite(float(ratio)) or ratio <= 0:
                    raise ValueError('Invalid split ratio')
                factor *= float(ratio)
    # Yahoo peut exprimer les cotations londoniennes en pence.
    scale = 0.01 if currency in {'GBp', 'GBX', 'ZAc'} else 1.0
    currency = {'GBp': 'GBP', 'GBX': 'GBP', 'ZAc': 'ZAR'}.get(currency, currency)
    session = metadata.get('currentTradingPeriod', {}).get('regular', {})
    quote_type = 'latest_available'
    if session.get('start') and session.get('end'):
        quote_type = 'intraday' if session['start'] <= int(stamp) < session['end'] else 'last_close'
    # Les splits ne prouvent pas l'absence de fusion, scission ou changement de titre
    # sur plusieurs années. Ces quantités restent des estimations explicites.
    quantity_verified = bool(export_date)
    if export_date:
        age_days = (datetime.now(timezone.utc).date() - date.fromisoformat(export_date)).days
        history_covers_export = history.index[0].date() <= date.fromisoformat(export_date) + timedelta(days=7)
        quantity_verified = age_days <= 730 and history_covers_export
    return {
        'symbol': symbol, 'price': float(price) * scale, 'currency': currency.upper(),
        'quote_at': datetime.fromtimestamp(int(stamp), timezone.utc).isoformat(),
        'fetched_at': datetime.now(timezone.utc).isoformat(), 'source': 'yahoo',
        'quote_type': quote_type, 'split_factor': factor,
        'quantity_verified': quantity_verified,
        'corporate_actions': corporate_actions,
    }


def get_quote(symbol: str, export_date: str | None, force: bool = False) -> dict | None:
    """Cache public quotes, coalesce concurrent fetches and retain dated stale data."""
    key = hashlib.sha256(f'v1:{symbol}:{export_date}'.encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f'{key}.json'
    with FileLock(str(path) + '.lock', timeout=45):
        cached = {}
        try:
            cached = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
        now = time.time()
        quote = cached.get('quote')
        age = now - cached.get('stored_at', 0)
        cooldown = now - cached.get('attempt_at', 0) < RETRY_DELAY
        if (quote and not force and age < QUOTE_TTL) or cooldown:
            return dict(quote, retrieval_status='fresh' if age < QUOTE_TTL else 'stale') if quote else None
        cached['attempt_at'] = now
        try:
            quote = fetch_yahoo_quote(symbol, export_date)
            cached.update(quote=quote, stored_at=now, refresh_failed=False)
        except Exception as exc:
            cached['refresh_failed'] = True
            logger.warning('Cours Saxo indisponible pour %s: %s', symbol, exc)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(cached), encoding='utf-8')
        os.replace(temporary, path)
        quote = cached.get('quote')
        return dict(quote, retrieval_status='fresh' if now - cached.get('stored_at', 0) < QUOTE_TTL else 'stale', refresh_failed=cached.get('refresh_failed', False)) if quote else None
