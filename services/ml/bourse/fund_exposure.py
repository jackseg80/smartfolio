"""Dated issuer sector look-through. Unknown weights are never renormalized."""
from __future__ import annotations

import hashlib
import math
import re
from datetime import date, datetime, timezone
from html.parser import HTMLParser

SECTORS = ('Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
           'Communication Services', 'Industrials', 'Consumer Staples', 'Energy',
           'Utilities', 'Real Estate', 'Materials')
LABELS = {**{s: s for s in SECTORS}, 'Information Technology': 'Technology',
          'Health Care': 'Healthcare', 'Communication': 'Communication Services',
          'Financial Services': 'Financials', 'Consumer Cyclical': 'Consumer Discretionary',
          'Consumer Defensive': 'Consumer Staples', 'Basic Materials': 'Materials'}
# Public product identities, never a manifest of an account's holdings.
ISSUER_PRODUCTS = {
    'IE00B4L5Y983': 'https://www.ishares.com/uk/individual/en/products/251882/ishares-msci-world-ucits-etf-acc-fund',
    'IE00BKM4GZ66': 'https://www.ishares.com/uk/individual/en/products/264659/ishares-core-msci-em-imi-ucits-etf',
}
MAX_EXPOSURE_AGE_DAYS = 45


class IssuerTableParser(HTMLParser):
    """Read Fund cells and the date inside the sector tab, never a nearby NAV date."""
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.panel_depth = None
        self.panel_text = []
        self.table_depth = None
        self.table_headers = []
        self.rows = []
        self.row = None
        self.cell = None
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag in ('br', 'img', 'input', 'meta', 'link', 'hr', 'source', 'wbr'):
            return
        self.depth += 1
        attrs = dict(attrs)
        if attrs.get('id') == 'tabpanel-exposureBreakdowns-sector':
            self.panel_depth = self.depth
        if self.panel_depth is not None and tag == 'table' and attrs.get('data-id') == 'exposurebreakdowns-sector-table':
            self.table_depth = self.depth
        if self.table_depth is not None:
            if tag == 'tr':
                self.row = []
            elif tag in ('td', 'th'):
                self.cell = []
                self.cell_tag = tag

    def handle_endtag(self, tag):
        if tag in ('br', 'img', 'input', 'meta', 'link', 'hr', 'source', 'wbr'):
            return
        if tag in ('td', 'th') and self.cell is not None:
            value = ' '.join(' '.join(self.cell).split())
            if self.cell_tag == 'th':
                self.table_headers.append(value)
            elif self.row is not None:
                self.row.append(value)
            self.cell = None
        if tag == 'tr' and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None
        if self.table_depth == self.depth:
            self.table_depth = None
        if self.panel_depth == self.depth:
            self.panel_depth = None
        self.depth = max(0, self.depth - 1)

    def handle_data(self, data):
        self.text.append(data)
        if self.panel_depth is not None:
            self.panel_text.append(data)
        if self.cell is not None:
            self.cell.append(data)


def parse_ishares(html, isin, url, today=None):
    today = today or date.today()
    parser = IssuerTableParser()
    parser.feed(html)
    if isin not in ' '.join(parser.text):
        raise ValueError('Issuer page does not confirm the exact fund ISIN')
    if parser.table_headers != ['Type', 'Fund']:
        raise ValueError('The issuer table is not an unambiguous Fund sector breakdown')
    panel = ' '.join(parser.panel_text)
    dates = re.findall(r'as of\s+(\d{1,2}/[A-Za-z]+/\d{4})', panel, re.I)
    if len(set(dates)) != 1:
        raise ValueError('The issuer sector breakdown has no unambiguous observation date')
    observed = datetime.strptime(dates[0].replace('/Sept/', '/Sep/'), '%d/%b/%Y').date()
    if not 0 <= (today - observed).days <= MAX_EXPOSURE_AGE_DAYS:
        raise ValueError('The issuer sector breakdown is stale or dated in the future')
    weights = {}
    unassigned = 0.0
    for row in parser.rows:
        if len(row) != 2:
            raise ValueError('Unexpected issuer table shape')
        pct = float(row[1].replace('%', '').replace(',', ''))
        if not math.isfinite(pct) or not 0 <= pct <= 100:
            raise ValueError('Invalid issuer sector weight')
        sector = LABELS.get(row[0])
        if sector:
            if sector in weights:
                raise ValueError('Duplicate issuer sector row')
            weights[sector] = pct / 100
        else:
            unassigned += pct / 100
    if not weights or sum(weights.values()) + unassigned > 1 + 1e-9:
        raise ValueError('Issuer weights do not form a valid partial breakdown')
    return dict(weights=weights, known_fraction=sum(weights.values()), as_of=observed.isoformat(),
                retrieved_at=datetime.now(timezone.utc).isoformat(), source_url=url,
                source_kind='issuer_fund_breakdown', source_sha256=hashlib.sha256(html.encode()).hexdigest(),
                identity_basis='exact_isin', reason='Partial Fund table; omitted and non-industry rows remain unclassified')


async def issuer_exposure(isin, client):
    url = ISSUER_PRODUCTS.get(isin)
    if not url:
        return dict(weights={}, known_fraction=0, as_of=None, source_url=None, source_kind='unavailable',
                    reason='No verified dated fund-sector adapter for this ISIN')
    try:
        response = await client.get(url)
        response.raise_for_status()
        return parse_ishares(response.text, isin, url)
    except Exception:
        # Never include URLs with query tokens or provider exception payloads in UI/logs.
        return dict(weights={}, known_fraction=0, as_of=None, source_url=url, source_kind='unavailable',
                    reason='Issuer breakdown unavailable, stale, or rejected by identity/date/weight checks')
