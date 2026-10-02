from pathlib import Path
from datetime import date
import pandas as pd
import exchange_calendars as xcals
from services.ml.live_observations import completed_stock_sessions

def test_public_calendar_cache_reuses_exact_schedule_without_caching_prices(monkeypatch):
    completed_stock_sessions.cache_clear()
    real = xcals.get_calendar
    calls = []
    def tracked(name, **kwargs):
        calls.append((name, kwargs))
        return real(name, **kwargs)
    monkeypatch.setattr(xcals, 'get_calendar', tracked)
    start, end = date(2026,9,1), date(2026,10,1)
    first = completed_stock_sessions('XNYS', start, end)
    second = completed_stock_sessions('XNYS', start, end)
    assert first.equals(second) and len(calls) == 1
    expected = pd.DatetimeIndex(real('XNYS', start=start, end=end).sessions)
    if expected.tz is None: expected = expected.tz_localize('UTC')
    assert first.equals(expected[expected < pd.Timestamp(end,tz='UTC')])
    tomorrow = completed_stock_sessions('XNYS', start, date(2026,10,2))
    assert len(calls) == 2 and tomorrow[-1] > first[-1]
    completed_stock_sessions('XSWX', start, end)
    assert len(calls) == 3  # A distinct exchange cannot reuse the NYSE schedule.
    completed_stock_sessions.cache_clear()

def test_stock_loading_source_is_utf8_and_failed_sections_remain_retryable():
    root = Path(__file__).resolve().parents[2]
    source = (root/'static/bourse-analytics.html').read_text(encoding='utf-8')
    loader = (root/'static/components/stock-ml-insights.js').read_text(encoding='utf-8')
    assert '<meta charset="UTF-8"' in source
    assert all(c not in source for c in ('\u00e2\u20ac', '\u00c3', '\ufffd'))
    assert 'return await loadStockMLInsights' in source
    assert 'Session expired' in loader and 'Retry this tab' in loader


def test_visible_ml_fallback_messages_have_no_corrupted_punctuation():
    root = Path(__file__).resolve().parents[2]
    for name in ('ai-dashboard.html', 'cycle-analysis.html'):
        lines = (root/'static'/name).read_text(encoding='utf-8').splitlines()
        for line in lines:
            if 'addAdminLog(' in line or 'addResult(' in line:
                assert all(c not in line for c in ('\u00e2\u0161', '\u00e2\u20ac', '\u00f0\u0178', '\ufffd'))
