"""Shared, deterministic builders for portfolio exports.

Exports must remain useful offline: classifications therefore come from the
position itself and the maintained local taxonomy, never from a live quote
provider during a download.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from adapters import saxo_adapter
from services.fx_service import convert as fx_convert


GICS_SECTORS = [
    "Technology",
    "Healthcare",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]

# A sector is meaningful for a single operating company.  Funds, bonds and
# commodities instead use the exposure category shown below.
EQUITY_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
    "IFX": "Technology", "CDR": "Technology",
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "TSLA": "Consumer Discretionary", "AMZN": "Consumer Discretionary",
    "BABA": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "UHRN": "Consumer Discretionary",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BLK": "Financials", "SCHW": "Financials", "UBSG": "Financials",
    "BRKB": "Financials", "SLHN": "Financials", "COIN": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "LLY": "Healthcare", "MRK": "Healthcare", "BAX": "Healthcare",
    "ROG": "Healthcare",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "WFRD": "Energy",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "MMM": "Industrials", "GLEN": "Materials",
}

ETF_EXPOSURES = {
    "IWDA": "Diversified Equity", "WORLD": "Diversified Equity",
    "CSPX": "Diversified Equity", "ACWI": "Diversified Equity",
    "VT": "Diversified Equity", "VWO": "Emerging Markets Equity",
    "VGK": "Europe Equity", "FLXI": "Emerging Markets Equity",
    "SMH": "Technology", "XLI": "Industrials", "XLU": "Utilities",
    "XLP": "Consumer Staples", "AGGS": "Fixed Income",
    "XGDU": "Commodities", "GLD": "Commodities", "SLV": "Commodities",
}


def _base_symbol(symbol: str) -> str:
    return (symbol or "").split(":", 1)[0].upper()


def classify_saxo_position(position: dict[str, Any]) -> tuple[str, str]:
    """Return (classification, classification_basis) for an exported position."""
    asset_class = str(position.get("asset_class") or "").upper()
    if asset_class in {"CASH", "MONEY MARKET", "FX CASH"}:
        return "Cash", "Cash"

    provided_sector = position.get("sector")
    if provided_sector:
        return str(provided_sector), "Provider classification"

    symbol = _base_symbol(str(position.get("symbol") or ""))
    if asset_class == "ETF":
        return ETF_EXPOSURES.get(symbol, "Unclassified ETF"), "ETF exposure"
    return EQUITY_SECTORS.get(symbol, "Unclassified Equity"), "GICS"


def read_saxo_cash(user_id: str, file_key: Optional[str]) -> dict[str, Any]:
    """Read one saved cash balance and normalise it to USD for aggregation."""
    cash_key = file_key or "default"
    cash_file = Path(f"data/users/{user_id}/saxobank/cash/{cash_key}_cash.json")
    if not cash_file.exists():
        return {"amount": 0.0, "currency": "USD", "value_usd": 0.0, "last_updated": None}

    with cash_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    amount = float(data.get("cash_amount", 0.0) or 0.0)
    currency = str(data.get("currency") or "USD").upper()
    return {
        "amount": amount,
        "currency": currency,
        "value_usd": float(fx_convert(amount, currency, "USD")),
        "last_updated": data.get("last_updated"),
    }


def build_saxo_export_data(user_id: str, file_key: Optional[str] = None) -> dict[str, Any]:
    """Build positions, cash and classifications for Saxo exports."""
    positions: list[dict[str, Any]] = []
    classification_totals: dict[str, dict[str, float | int]] = {}

    for raw in saxo_adapter._iter_positions(user_id=user_id, file_key=file_key):
        classification, basis = classify_saxo_position(raw)
        value_usd = float(raw.get("market_value_usd", 0.0) or 0.0)
        position = {
            "symbol": raw.get("symbol", ""),
            "instrument": raw.get("instrument") or raw.get("instrument_name") or "",
            "asset_class": raw.get("asset_class", "Unknown"),
            "quantity": float(raw.get("quantity", 0.0) or 0.0),
            "market_value_usd": value_usd,
            "currency": raw.get("currency", "USD"),
            "classification": classification,
            "classification_basis": basis,
            "entry_price": float(raw.get("avg_price", 0.0) or raw.get("entry_price", 0.0) or 0.0),
        }
        positions.append(position)
        _add_total(classification_totals, classification, value_usd)

    cash = read_saxo_cash(user_id, file_key)
    if cash["amount"]:
        positions.append({
            "symbol": f"CASH:{cash['currency']}",
            "instrument": "Saxo cash balance",
            "asset_class": "Cash",
            "quantity": cash["amount"],
            "market_value_usd": cash["value_usd"],
            "currency": cash["currency"],
            "classification": "Cash",
            "classification_basis": "Cash",
            "entry_price": 0.0,
        })
        _add_total(classification_totals, "Cash", cash["value_usd"])

    total_value_usd = sum(float(position["market_value_usd"]) for position in positions)
    classifications = [
        {
            "name": name,
            "value_usd": totals["value_usd"],
            "percentage": (float(totals["value_usd"]) / total_value_usd * 100) if total_value_usd else 0.0,
            "asset_count": totals["count"],
        }
        for name, totals in sorted(classification_totals.items(), key=lambda item: float(item[1]["value_usd"]), reverse=True)
    ]
    return {
        "positions": positions,
        "classifications": classifications,
        "summary": {
            "total_value_usd": total_value_usd,
            "positions_value_usd": total_value_usd - float(cash["value_usd"]),
            "cash_value_usd": cash["value_usd"],
            "cash_currency": cash["currency"],
            "cash_amount": cash["amount"],
            "positions_count": len(positions),
        },
    }


def _add_total(totals: dict[str, dict[str, float | int]], name: str, value_usd: float) -> None:
    current = totals.setdefault(name, {"value_usd": 0.0, "count": 0})
    current["value_usd"] = float(current["value_usd"]) + value_usd
    current["count"] = int(current["count"]) + 1
