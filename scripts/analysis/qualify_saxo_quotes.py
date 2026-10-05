"""Read-only qualification of three public listings; no Saxo account access."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.saxo_quote_service import fetch_yahoo_quote, yahoo_symbol

if __name__ == '__main__':
    for instrument in ('MSFT:xnas', 'SLHn:xvtx', 'AGGS:xvtx'):
        try:
            print(json.dumps({'instrument': instrument, 'quote': fetch_yahoo_quote(yahoo_symbol(instrument), '2026-01-18')}))
        except Exception as exc:
            print(json.dumps({'instrument': instrument, 'error': str(exc)}))
