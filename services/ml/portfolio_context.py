"""Read the authenticated selection; never substitute a benchmark for a holding."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from services.ml.reliability import ROOT, valid_symbol


def stock_symbol(symbol: str) -> str:
    """Explicit Saxo venue to vendor suffix mapping, preserving instrument identity."""
    base, _, venue = symbol.partition(":")
    suffixes = {"xnas": "", "xnys": "", "arcx": "", "xswx": ".SW", "xvtx": ".SW", "xams": ".AS", "xetr": ".DE", "xlon": ".L", "xmil": ".MI"}
    if venue and venue.lower() not in suffixes:
        raise ValueError("The instrument venue has no verified ticker mapping")
    from services.ml.bourse.currency_detector import CurrencyExchangeDetector
    hints={'xnas':'NASDAQ','xnys':'NYSE','arcx':'NYSE','xswx':'SWX','xvtx':'SWX','xams':'AMS','xetr':'XETRA','xlon':'LSE','xmil':'MIL'}
    # The known Yahoo WRDUSW line is CHF. Do not treat a declared USD listing
    # as the same quote series when no corresponding USD line is verified.
    if base.upper() in ('WRDUSW_USD','WRDUSW_USD.SW'):
        raise ValueError('The exact USD listing has no verified provider mapping')
    mapped, _, _ = CurrencyExchangeDetector().detect_currency_and_exchange("BRKb" if base.lower()=="brkb" else base.upper(), exchange_hint=hints.get(venue.lower()))
    return valid_symbol(mapped)



async def read_context(user: str, source: str, market: str, file_key: str | None = None) -> dict:
    # Cette capture privée sert uniquement à la prévisualisation locale. Elle
    # n'est jamais incluse dans le paquet de livraison ni utilisée en production.
    snapshot_path = os.getenv("ML_PORTFOLIO_SNAPSHOT")
    if snapshot_path:
        if os.getenv("ENVIRONMENT", "").lower() == "production":
            raise ValueError("A private preview capture cannot be used in production")
        capture = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        entry = capture.get(market, {})
        if capture.get("user_id") != user or entry.get("source") != source or (file_key and file_key != entry.get("file_key")):
            raise ValueError("No captured portfolio matches this authenticated user and selected source")
        return {**entry, "observation_mode": "dated_read_only_production_copy", "captured_at": capture["observed_at"]}
    if market == "stocks" and source in ("saxobank", "saxobank_csv"):
        from adapters.saxo_adapter import _load_snapshot
        config_path = ROOT / 'data/users' / user / 'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
        effective_file = file_key or config.get('sources', {}).get('bourse', {}).get('selected_csv_file')
        if not effective_file:
            raise ValueError("No selected Saxo CSV is available for this user")
        path = ROOT / 'data/users' / user / 'saxobank/data' / effective_file
        if Path(effective_file).name != effective_file or path.suffix != '.csv' or not path.is_file():
            raise ValueError('The selected Saxo CSV is missing or invalid; no different file is substituted')
        snapshot = _load_snapshot(user_id=user, file_key=effective_file)
        items = []
        dates = []
        for portfolio in snapshot.get("portfolios", []):
            dates.append(portfolio.get("last_updated") or portfolio.get("updated_at"))
            for item in portfolio.get("positions", []):
                items.append({"symbol": item.get("symbol") or item.get("instrument_id"), "value_usd": item.get("market_value_usd")})
        path = ROOT / "data/users" / user / "saxobank/data" / effective_file
        return dict(source=source, source_used="saxobank_csv", file_key=effective_file, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), items=items, data_as_of=max((d for d in dates if d), default=None), observation_mode="selected_user_source")
    from services.balance_service import balance_service
    if source.startswith("stub") or source in ("auto", "all", "category_based"):
        raise ValueError("Select an explicit portfolio source before requesting personal ML results")
    result = await balance_service.resolve_current_balances(user_id=user, source=source)
    if result.get("error") or result.get("source_used") in ("none", None):
        raise ValueError("The selected authenticated portfolio source is unavailable")
    expected = {"cointracking_csv": "cointracking"}
    actual = result["source_used"].removesuffix("_cached")
    if actual != expected.get(source, source):
        raise ValueError("The resolved source differs from the selected source; no fallback is displayed")
    return dict(source=source, source_used=result["source_used"], items=result.get("items", []), data_as_of=result.get("asof") or result.get("timestamp"), consulted_at=datetime.now(timezone.utc).isoformat(), observation_mode="selected_user_source")


def select_assets(context: dict, market: str, limit: int) -> tuple[list[str], dict]:
    items = context.get("items", [])
    ranked = sorted(items, key=lambda r: float(r.get("value_usd") or 0), reverse=True)
    mappings = []
    for item in ranked:
        symbol = item.get("symbol")
        if not symbol:
            continue
        try:
            asset = stock_symbol(symbol) if market == "stocks" else valid_symbol(symbol)
        except ValueError:
            mappings.append(dict(source_symbol=symbol, asset=None, reason="Unverified instrument mapping; no proxy substitution"))
            continue
        if not any(row.get("asset") == asset for row in mappings):
            mappings.append(dict(source_symbol=symbol, asset=asset, reason="Exact source symbol" if asset == symbol else "Explicit Saxo venue mapping"))
    selected = [row["asset"] for row in mappings if row["asset"]][:limit]
    metadata = {k: v for k, v in context.items() if k != "items"}
    metadata.update(held_positions=len(items), selected_assets=len(selected), omitted_assets=max(0, len([m for m in mappings if m["asset"]])-len(selected)), limit=limit, instrument_mapping=mappings, reason="Selected holdings ordered by source value; no benchmark, wrapped-token or ETF proxy substitution. No portfolio-level forecasting validation.")
    metadata["scope_sha256"] = hashlib.sha256(json.dumps(dict(context=context, selected=selected), sort_keys=True).encode()).hexdigest()
    return selected, metadata
