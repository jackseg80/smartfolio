"""Explainable screening and cash-conserving scenarios; never an order engine."""
from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from services.ml.bourse.fund_exposure import LABELS, SECTORS, issuer_exposure
from services.ml.bourse.horizons import OPPORTUNITY_HORIZONS
from services.ml.bourse.market_snapshot import number

SECTOR_ETFS = dict(zip(SECTORS, ('XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLU', 'XLRE', 'XLB')))
CALENDARS = {'NMS': 'XNYS', 'NGM': 'XNYS', 'NCM': 'XNYS', 'NYQ': 'XNYS', 'PCX': 'XNYS',
             'ASE': 'XNYS', 'BTS': 'XNYS', 'EBS': 'XSWX', 'SWX': 'XSWX', 'LSE': 'XLON',
             'GER': 'XETR', 'AMS': 'XAMS', 'MIL': 'XMIL', 'PAR': 'XPAR', 'HKG': 'XHKG',
             'JPX': 'XTKS', 'KSC': 'XKRX'}


def exposure_summary(positions, targets, min_gap=5):
    missing_values = sum(p['value_usd'] is None for p in positions)
    total = sum(p['value_usd'] for p in positions if p['value_usd'] is not None)
    known = {s: 0.0 for s in SECTORS}
    for p in positions:
        if p['value_usd'] is None:
            continue
        for sector, weight in p['classification']['weights'].items():
            known[sector] += p['value_usd'] * weight
    classified = sum(known.values())
    unknown = max(0, total - classified)
    if total <= 0:
        return dict(coverage=None, unclassified_pct=None, sector_assessment=[], denominator_usd=total,
                    unvalued_positions=missing_values, valued_positions=len(positions) - missing_values,
                    coverage_scope='valued_subset_only' if missing_values else 'all_valued_securities',
                    reason='No positive valued securities; no allocation conclusion')
    missing = unknown / total * 100
    assessments = []
    for sector in SECTORS:
        lower = known[sector] / total * 100
        upper = min(100, lower + missing)
        target = targets[sector]
        deficit = target - upper
        excess = lower - target
        status = 'Underweight' if deficit >= min_gap and deficit > 0 else 'Overweight' if excess >= min_gap and excess > 0 else 'Indeterminate' if lower <= target <= upper and missing > 1e-8 else 'Within threshold'
        assessments.append(dict(sector=sector, lower_pct=lower if not missing_values else None,
                                upper_pct=upper if not missing_values else None, target_pct=target,
                                status=status if not missing_values else 'Indeterminate',
                                certain_deficit_pct=max(0, deficit) if not missing_values else None,
                                certain_excess_pct=max(0, excess) if not missing_values else None))
    return dict(coverage=classified / total, unclassified_pct=missing, sector_assessment=assessments,
                unvalued_positions=missing_values, valued_positions=len(positions) - missing_values,
                coverage_scope='valued_subset_only' if missing_values else 'all_valued_securities',
                denominator_usd=total,
                reason=('Some position values are missing. Coverage is for the known-value subset; no full-portfolio industry intervals or conclusions can be computed. ' if missing_values else '') + 'Percentages use valued securities, excluding saved cash. Unknown and non-industry fund weights stay outside industry coverage. No equity-only denominator is assumed.')


def holding_reviews(positions, summary):
    total = summary['denominator_usd']
    reviews = []
    for p in positions:
        weight = p['value_usd'] / total * 100 if total and p['value_usd'] is not None and not summary.get('unvalued_positions') else None
        reasons = []
        if p['value_usd'] is None:
            reasons.append('Source valuation missing; position is retained, not counted as zero')
        if weight is not None and weight > 10:
            reasons.append('Position exceeds the generic 10% concentration review threshold')
        if p['classification']['known_fraction'] < 1 - 1e-8:
            reasons.append(p['classification']['reason'])
        if not p['acquisition_date']:
            reasons.append('Acquisition date is unavailable; holding-period and tax eligibility are not assessed')
        reviews.append(dict(position_id=p['id'], symbol=p['symbol'], name=p['name'], weight_pct=weight,
                            review_reasons=reasons, action='REVIEW' if reasons else 'NO_RULE_TRIGGER',
                            sale_eligibility='unassessed',
                            sale_reason='No automatic sale: acquisition dates, tax lots, stop orders and personal limits are not verified'))
    return reviews


class PublicMarketData:
    """Bounded per-request work. Only the existing public price cache persists."""
    def __init__(self):
        self.limit = asyncio.Semaphore(4)
        self.profiles = {}
        self.histories = {}
        self.source = None

    async def profile(self, symbol):
        if symbol not in self.profiles:
            async def fetch():
                import yfinance as yf
                async with self.limit:
                    try:
                        info = await asyncio.wait_for(asyncio.to_thread(lambda: yf.Ticker(symbol).info), 12)
                        return info if isinstance(info, dict) else {}
                    except Exception:
                        return {}
            self.profiles[symbol] = asyncio.create_task(fetch())
        return await self.profiles[symbol]

    async def history(self, symbol, horizon, expected_currency=None):
        key = (symbol, horizon, expected_currency)
        if key not in self.histories:
            async def fetch():
                from services.ml.bourse.data_sources import StocksDataSource
                from services.ml.live_observations import completed_stock_sessions
                from services.risk.bourse.calculator import BourseRiskCalculator
                info = await self.profile(symbol)
                async with self.limit:
                    if self.source is None:
                        self.source = StocksDataSource()
                    calendar = CALENDARS.get(info.get('exchange'))
                    if not calendar:
                        raise ValueError('Exchange calendar cannot be verified for this listing')
                    end = datetime.now(timezone.utc).replace(tzinfo=None)
                    frame = await self.source.get_ohlcv_data(symbol, OPPORTUNITY_HORIZONS[horizon].history_calendar_days, end)
                    native = frame.attrs.get('native_currency')
                    if expected_currency and native != expected_currency:
                        raise ValueError('Provider quote currency differs from the exact held listing')
                    if frame.attrs.get('price_origin') != 'yahoo' or frame.attrs.get('price_adjustment') != 'split_and_dividend_adjusted':
                        raise ValueError('Adjusted prices with verified provider provenance are required')
                    # Drop today's bar on every exchange, even if the market has closed.
                    frame = frame.loc[frame.index.normalize() < pd.Timestamp(end.date())].copy()
                    if frame.empty or frame.index.duplicated().any():
                        raise ValueError('Completed prices are empty or contain duplicate sessions')
                    frame = frame.sort_index()
                    sessions = completed_stock_sessions(calendar, frame.index[0].date(), end.date()).tz_localize(None)
                    if not frame.index.equals(sessions) or frame.attrs.get('dropped_observations', 0):
                        raise ValueError('History is incomplete or stale relative to completed exchange sessions')
                    if not native:
                        raise ValueError('Provider quote currency is unavailable')
                    if native != 'USD':
                        fx = await self.source.fetcher.fetch_historical_fx(native, frame.index[0].to_pydatetime() - timedelta(days=7), end)
                        frame = BourseRiskCalculator._convert_prices_to_usd(frame, fx)
                    close = frame['close']
                    if not np.isfinite(close).all() or (close <= 0).any():
                        raise ValueError('Historical prices contain invalid observations')
                    close.attrs = dict(frame.attrs, currency='USD', calendar=calendar, source_url=f'https://finance.yahoo.com/quote/{symbol}/history/')
                    return close
            self.histories[key] = asyncio.create_task(fetch())
        return await self.histories[key]

def listing(position):
    from services.ml.portfolio_context import stock_symbol
    base, _, mic = position['symbol'].partition(':')
    # The common helper has no ISIN argument. Qualified Saxo aliases need it.
    if re.search(r'_(?:CHF|EUR|GBP|JPY|USD)(?:\.[A-Z]+)?$', base.upper()):
        if not position.get('isin'):
            raise ValueError('The qualified listing needs its exact ISIN')
        if base.upper() in ('WRDUSW_USD', 'WRDUSW_USD.SW'):
            raise ValueError('The exact USD listing has no verified provider mapping')
        hints = {'xnas': 'NASDAQ', 'xnys': 'NYSE', 'arcx': 'NYSE', 'xswx': 'SWX', 'xvtx': 'SWX',
                 'xams': 'AMS', 'xetr': 'XETRA', 'xlon': 'LSE', 'xmil': 'MIL'}
        if mic.lower() not in hints:
            raise ValueError('The qualified listing has no verified venue')
        from services.ml.bourse.currency_detector import CurrencyExchangeDetector
        mapped, currency, _ = CurrencyExchangeDetector().detect_currency_and_exchange(base, isin=position['isin'], exchange_hint=hints[mic.lower()])
        if currency != position.get('quote_currency'):
            raise ValueError('Qualified ticker currency differs from the source quote currency')
        from services.ml.reliability import valid_symbol
        return valid_symbol(mapped)
    return stock_symbol(position['symbol'])


async def classify(position, provider, client):
    p = dict(position)
    try:
        p['history_symbol'] = listing(p)
        p['mapping_reason'] = 'Explicit source listing; quote currency will be checked against history'
    except ValueError:
        p['history_symbol'] = None
        p['mapping_reason'] = 'Exact listing mapping is unavailable; no replacement series'
    if p['asset_class'].upper() in ('ETF', 'FUND', 'MUTUAL FUND'):
        p['classification'] = await issuer_exposure(p['isin'], client)
        p['geography'] = dict(status='unavailable', reason='No verified geography adapter; geography is separate from economic sectors')
    elif p['asset_class'].upper() in ('STOCK', 'EQUITY', 'EQUITIES') and p['history_symbol']:
        info = await provider.profile(p['history_symbol'])
        sector = LABELS.get(info.get('sector'))
        currency = {'GBp': 'GBP', 'GBX': 'GBP'}.get(info.get('currency'), info.get('currency'))
        if not p.get('quote_currency') or currency != p['quote_currency']:
            sector = None
        p['classification'] = dict(weights={sector: 1} if sector else {}, known_fraction=1 if sector else 0,
                                   as_of=None, retrieved_at=datetime.now(timezone.utc).isoformat(),
                                   source_url=f"https://finance.yahoo.com/quote/{p['history_symbol']}/profile/",
                                   source_kind='secondary_company_profile' if sector else 'unavailable',
                                   reason='Company sector from a secondary provider; effective classification date unavailable' if sector else 'Company profile, industry or exact quote currency unavailable')
        p['geography'] = dict(status='company_domicile_only', country=info.get('country'),
                              reason='Company domicile does not measure revenue geography')
    else:
        p['classification'] = dict(weights={}, known_fraction=0, as_of=None, source_url=None, source_kind='unavailable',
                                   reason='Asset class or exact listing is not verified as an equity industry exposure')
        p['geography'] = dict(status='unavailable', reason='Geography is unavailable')
    return p


def historical_screen(close, sessions):
    if len(close) < sessions + 1:
        raise ValueError(f'History needs {sessions + 1} complete observations for this window')
    window = close.iloc[-sessions - 1:]
    returns = window.pct_change(fill_method=None).dropna()
    deviation = float(returns.std(ddof=1))
    ratio = float(returns.mean()) / deviation * math.sqrt(sessions) if deviation > 0 else None
    if ratio is None or not math.isfinite(ratio):
        raise ValueError('Historical variability is insufficient to calculate a score')
    score = max(0, min(100, 50 + 15 * ratio))
    return dict(score=score, return_pct=(float(window.iloc[-1]) / float(window.iloc[0]) - 1) * 100,
                annualized_volatility_pct=deviation * math.sqrt(252) * 100,
                sessions=sessions, start=window.index[0].date().isoformat(), end=window.index[-1].date().isoformat(),
                currency='USD', method='clip(50 + 15 * mean(daily returns) / sample std * sqrt(window sessions), 0, 100)',
                meaning='Historical risk-adjusted price screen. No expected return, probability or validated holding-period forecast.')


def candidate_universe(sector):
    if sector not in (*SECTORS, 'all'):
        raise ValueError('Unknown candidate sector')
    if sector == 'all':
        return [dict(symbol=e, name=f'{s} sector ETF', intended_sector=s, kind='ETF') for s, e in SECTOR_ETFS.items()]
    from services.ml.bourse.sector_analyzer import SECTOR_TOP_STOCKS
    etf = SECTOR_ETFS[sector]
    return [dict(symbol=etf, name=f'{sector} sector ETF', intended_sector=sector, kind='ETF')] + [
        dict(symbol=s, name=n, intended_sector=sector, kind='STOCK') for s, n, _ in SECTOR_TOP_STOCKS[etf]]


async def screen_candidate(candidate, positions, provider, horizon):
    row = dict(candidate, id=candidate['symbol'], status='unavailable', exclusions=[], metrics=None, rank=None)
    held = {p['history_symbol'] for p in positions if p['history_symbol']}
    if row['symbol'] in held:
        row['status'] = 'held'
        row['exclusions'] = ['The exact mapped listing is already held']
        return row
    info = await provider.profile(row['symbol'])
    row['reported_sector'] = LABELS.get(info.get('sector'))
    row['isin'] = info.get('isin')
    if row['isin'] and any(p['isin'] == row['isin'] for p in positions):
        row['status'] = 'held'
        row['exclusions'] = ['The same ISIN is already held on another listing']
        return row
    if row['kind'] == 'STOCK' and row['reported_sector'] != row['intended_sector']:
        row['exclusions'] = ['Current provider sector differs from the curated sector or is unavailable']
        return row
    try:
        close = await provider.history(row['symbol'], horizon)
        row['metrics'] = historical_screen(close, OPPORTUNITY_HORIZONS[horizon].signal_sessions)
        row.update(status='screened', price_source=close.attrs.get('source_url'), price_retrieved_at=close.attrs.get('retrieved_at'))
    except Exception as exc:
        row['exclusions'] = [str(exc) if isinstance(exc, ValueError) else 'Verified completed price history is unavailable']
    row['limits'] = ['Curated universe, not the whole market. No personal suitability or trading eligibility check.',
                     'Fund constituents may overlap held funds. No constituent-level overlap calculation.',
                     'ISIN deduplication is incomplete when the candidate provider does not expose an ISIN.',
                     'ETF sector name describes its mandate; it is not a verified 100% sector decomposition.',
                     'No valuation or diversification composite is invented.']
    return row


async def analyze(snapshot, horizon='medium', sector='all', targets=None, min_gap=5, provider=None):
    if horizon not in OPPORTUNITY_HORIZONS:
        raise ValueError('Invalid holding horizon')
    from services.ml.bourse.opportunity_scanner import default_sector_targets, parse_sector_targets
    allocations = parse_sector_targets(targets) if targets is not None else default_sector_targets()
    provider = provider or PublicMarketData()
    import httpx
    async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
        # Issuer calls also bounded. No private symbols/ISINs are written to a cache.
        semaphore = asyncio.Semaphore(4)
        async def one(p):
            async with semaphore:
                return await classify(p, provider, client)
        positions = await asyncio.gather(*(one(p) for p in snapshot['positions']))
    summary = exposure_summary(positions, allocations, min_gap)
    candidates = await asyncio.gather(*(screen_candidate(c, positions, provider, horizon) for c in candidate_universe(sector)))
    ranked = sorted((c for c in candidates if c['status'] == 'screened'), key=lambda c: c['metrics']['score'], reverse=True)
    for rank, row in enumerate(ranked, 1):
        row['rank'] = rank
    # Scores across different observation dates are not a comparable ranking.
    dates = {c['metrics']['end'] for c in ranked}
    if len(dates) > 1:
        for row in ranked:
            row['rank'] = None
    result = dict(version=2, nature='manual_decision_support', generated_at=datetime.now(timezone.utc).isoformat(),
                  context=snapshot['context'], snapshot_id=snapshot['snapshot_id'], cash=snapshot['cash'],
                  horizon=horizon, horizon_details=OPPORTUNITY_HORIZONS[horizon].metadata(), candidate_sector=sector,
                  target_source='personal_policy' if targets is not None else 'generic_reference',
                  target_note='Generic reference: normalized midpoint of the legacy static industry ranges, not current index weights or a recommended personal policy.' if targets is None else 'User-entered industry targets for all valued securities, excluding saved cash.',
                  targets=allocations, min_gap_pct=min_gap, positions=positions, exposure=summary,
                  candidates=candidates, holding_reviews=holding_reviews(positions, summary),
                  automatic_sales=[], automatic_sale_reason='No automated sale decision without personal constraints and verified acquisition/tax/stop-order data.',
                  candidate_summary=dict(total=len(candidates), screened=len(ranked), held=sum(c['status'] == 'held' for c in candidates),
                                         unavailable=sum(c['status'] == 'unavailable' for c in candidates),
                                         ranking_comparable=len(dates) <= 1),
                  risk_score=None, risk_reason='No validated portfolio robustness score is implemented here. Optional scenarios report actual historical volatility separately.',
                  conclusion='Zero screened candidates or certain gaps does not establish that this portfolio is balanced.',
                  scenario=None)
    return result


def simulate(analysis, scenario):
    """Estimated dollar amounts at snapshot values, never quantities or broker orders."""
    if not isinstance(scenario, dict) or set(scenario) - {'snapshot_id', 'trades', 'costs_usd', 'slippage_pct', 'acknowledge_dated_values', 'include_history'}:
        raise ValueError('Invalid scenario fields')
    if scenario.get('snapshot_id') != analysis['snapshot_id']:
        raise ValueError('The source snapshot changed; scan again before simulating')
    if scenario.get('acknowledge_dated_values') is not True:
        raise ValueError('Confirm that the scenario uses dated position values and saved cash')
    if any(p['value_usd'] is None for p in analysis['positions']):
        raise ValueError('Some selected positions have no source valuation; a full-portfolio scenario cannot be valued')
    cash = analysis['cash']['value_usd']
    if cash is None:
        raise ValueError('Source-consistent verified cash is unavailable; a funding scenario cannot be calculated')
    cash = number(cash, 'Saved cash')
    costs = number(scenario.get('costs_usd'), 'Total estimated costs in USD')
    slippage = number(scenario.get('slippage_pct'), 'Estimated slippage percentage')
    if costs < 0 or not 0 <= slippage <= 10:
        raise ValueError('Costs must be non-negative and slippage between 0 and 10%')
    trades = scenario.get('trades')
    if not isinstance(trades, list) or not 1 <= len(trades) <= 50:
        raise ValueError('Choose between 1 and 50 scenario changes')
    after = {p['id']: dict(p) for p in analysis['positions']}
    purchases = {c['id']: c for c in analysis['candidates'] if c['status'] == 'screened'}
    seen = set()
    gross = 0.0
    for trade in trades:
        if not isinstance(trade, dict) or set(trade) != {'side', 'id', 'amount_usd'}:
            raise ValueError('Each change needs side, id and amount_usd')
        side, identity = trade['side'], trade['id']
        if not isinstance(identity, str) or side not in ('sell', 'buy') or (side, identity) in seen:
            raise ValueError('Unknown side, invalid identity or duplicate change')
        seen.add((side, identity))
        amount = number(trade['amount_usd'], 'Scenario amount')
        if amount <= 0:
            raise ValueError('Scenario amounts must be positive')
        gross += amount
        if side == 'sell':
            if identity not in after or amount > after[identity]['value_usd']:
                raise ValueError('A sale must identify a held position and cannot exceed its snapshot value')
            after[identity]['value_usd'] -= amount
            cash += amount
        else:
            if identity not in purchases:
                raise ValueError('A purchase must identify a screened candidate in the current scan')
            c = purchases[identity]
            after['candidate:' + identity] = dict(id='candidate:' + identity, symbol=c['symbol'], name=c['name'],
                isin=c.get('isin'), history_symbol=c['symbol'], quote_currency=None, value_usd=amount,
                classification=dict(weights={}, known_fraction=0, reason='Candidate fund composition or dated company classification is not added to verified allocation by assumption'))
            # For screened individual stocks, the observed secondary sector is available.
            if c['kind'] == 'STOCK' and c.get('reported_sector'):
                after['candidate:' + identity]['classification'] = dict(weights={c['reported_sector']: 1}, known_fraction=1, reason='Secondary company profile')
            cash -= amount
    friction = costs + gross * slippage / 100
    cash -= friction
    if cash < -1e-8:
        raise ValueError('The combined changes exceed saved cash plus sale proceeds after costs and slippage')
    before_total = sum(p['value_usd'] for p in analysis['positions']) + analysis['cash']['value_usd']
    after_positions = list(after.values())
    after_total = sum(p['value_usd'] for p in after_positions) + cash
    if not math.isclose(after_total, before_total - friction, abs_tol=1e-6):
        raise ValueError('Scenario cash conservation check failed')
    exposure = exposure_summary(after_positions, analysis['targets'], analysis['min_gap_pct'])
    return dict(status='calculated', before_total_usd=before_total, after_total_usd=after_total,
                cash_before_usd=analysis['cash']['value_usd'], cash_after_usd=max(0, cash), costs_and_slippage_usd=friction,
                exposure_before=analysis['exposure'], exposure_after=exposure,
                positions_after=after_positions, trades=trades,
                historical_risk=None, risk_score=None,
                limits=['Amounts are manual USD estimates at dated snapshot values, not executable quantities.',
                        'Fees and slippage are user assumptions. Taxes, FX execution spreads, liquidity, lot sizes and stop orders are not modeled.',
                        'Purchased ETFs remain unclassified until an exact dated composition is verified.',
                        'No financial orders are created or submitted.'])


async def scenario_history(analysis, scenario, provider=None):
    provider = provider or PublicMarketData()
    risk = dict(status='unavailable', volatility_before_pct=None, volatility_after_pct=None, correlation_changes=[],
                reason='All valued holdings need exact USD histories and common completed return intervals; no partial-portfolio proxy')
    combined = {p['id']: p for p in analysis['positions'] + scenario['positions_after'] if p['value_usd'] > 0}
    async def prices(p):
        if not p['history_symbol']:
            raise ValueError('An exact held listing cannot be mapped')
        if not p['id'].startswith('candidate:') and not p.get('quote_currency'):
            raise ValueError('Held quote currency is missing; exact currency matching is unavailable')
        return p['id'], await provider.history(p['history_symbol'], analysis['horizon'], p.get('quote_currency'))
    outcomes = await asyncio.gather(*(prices(p) for p in combined.values()), return_exceptions=True)
    failed = sum(isinstance(x, Exception) for x in outcomes)
    risk.update(positions_required=len(outcomes), histories_available=len(outcomes) - failed)
    if failed:
        return risk
    closes = dict(outcomes)
    returns = pd.concat({k: v.pct_change(fill_method=None) for k, v in closes.items()}, axis=1).dropna()
    # Same start AND end dates for each daily return: no mixed holiday intervals.
    starts = pd.concat({k: pd.Series(v.index, index=v.index).shift(1) for k, v in closes.items()}, axis=1).reindex(returns.index)
    aligned = starts.nunique(axis=1) == 1
    returns = returns.loc[aligned]
    required = max(60, OPPORTUNITY_HORIZONS[analysis['horizon']].signal_sessions)
    if len(returns) < required or not np.isfinite(returns.to_numpy()).all():
        risk['reason'] = f'Fewer than {required} common completed daily USD return intervals'
        return risk
    def weighted(positions, cash):
        total = sum(p['value_usd'] for p in positions) + cash
        if total <= 0:
            raise ValueError('Portfolio total must be positive')
        # Cash return is held at zero; no invented interest or FX account return.
        return sum((returns[p['id']] * (p['value_usd'] / total) for p in positions if p['value_usd'] > 0), pd.Series(0., index=returns.index))
    before = weighted(analysis['positions'], scenario['cash_before_usd'])
    after = weighted(scenario['positions_after'], scenario['cash_after_usd'])
    risk.update(status='calculated', volatility_before_pct=float(before.std(ddof=1) * math.sqrt(252) * 100),
                volatility_after_pct=float(after.std(ddof=1) * math.sqrt(252) * 100),
                sessions=len(returns), start=starts.loc[returns.index[0]].iloc[0].date().isoformat(), end=returns.index[-1].date().isoformat(),
                reason='Historical fixed USD weights including zero-return cash; daily-rebalanced approximation. No validated Risk Score or forward forecast.')
    for p in scenario['positions_after']:
        if p['id'].startswith('candidate:') and p['value_usd'] > 0:
            correlation = returns[p['id']].corr(before)
            risk['correlation_changes'].append(dict(symbol=p['symbol'], correlation_to_before=float(correlation) if pd.notna(correlation) else None))
    return risk
