"""Script de test pour analyser le parsing du CSV Saxo"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from connectors.saxo_import import SaxoImportConnector

# Test file path
csv_path = r"data\users\jack\saxobank\data\20260114_185404_Positions_13-janv.-2026_10_38_00.csv"

print(f"Testing Saxo CSV parsing for: {csv_path}")
print("=" * 80)

connector = SaxoImportConnector()
result = connector.process_saxo_file(csv_path, user_id="jack")

print(f"\n[OK] Total positions parsed: {result.get('total_positions', 0)}")
print(f"[OK] Total market value USD: ${result.get('total_market_value_usd', 0):,.2f}")
print(f"[OK] Errors: {len(result.get('errors', []))}")

if result.get('errors'):
    print("\n[WARN] Parsing errors:")
    for error in result['errors']:
        print(f"  - {error}")

print("\n[INFO] Positions parsed:")
for i, pos in enumerate(result.get('positions', []), 1):
    symbol = pos.get('symbol', 'N/A')
    instrument = pos.get('instrument', 'N/A')
    qty = pos.get('quantity', 0)
    mv = pos.get('market_value_usd', 0)
    print(f"{i:3d}. {symbol:20s} - {instrument[:40]:40s} | Qty: {qty:10.2f} | Value: ${mv:12,.2f}")

# Check for WFRD specifically
wfrd_found = any('WFRD' in pos.get('symbol', '').upper() or 'WEATHERFORD' in pos.get('instrument', '').upper()
                 for pos in result.get('positions', []))
print(f"\n[{'OK' if wfrd_found else 'FAIL'}] Weatherford (WFRD) found: {wfrd_found}")
