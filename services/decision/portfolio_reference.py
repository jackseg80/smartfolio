"""Deterministic portfolio reference and crypto risk coverage calculations.

This module deliberately accepts already resolved balances and price histories.
The caller remains responsible for resolving them with the authenticated user
and the explicitly selected source through ``balance_service``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _finite_non_negative(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "locked"}:
            return True
        if normalized in {"false", "0", "no", "n", "", "unlocked"}:
            return False
    return False


def _utc_iso(observed_at: Optional[datetime]) -> str:
    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat()


def build_portfolio_snapshot(
    rows: Iterable[Mapping[str, Any]],
    *,
    user_id: str,
    source_id: str,
    observed_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build an immutable, serializable reference from resolved balance rows."""
    if not str(user_id).strip() or not str(source_id).strip():
        raise ValueError("user_id and source_id are required")

    timestamp = _utc_iso(observed_at)
    positions = []
    total_value = 0.0

    for index, raw in enumerate(rows):
        symbol = str(raw.get("symbol") or raw.get("asset") or "").strip().upper()
        if not symbol:
            raise ValueError(f"Position {index} has no symbol")

        quantity = _finite_non_negative(raw.get("amount", raw.get("quantity")))
        value_usd = _finite_non_negative(raw.get("value_usd", raw.get("usd_value")))
        locked_quantity = _finite_non_negative(raw.get("locked_amount", raw.get("locked_quantity")))
        is_locked = _coerce_bool(raw.get("is_locked", raw.get("locked", False)))
        if value_usd is not None:
            total_value += value_usd

        positions.append({
            "position_id": str(raw.get("position_id") or f"{index}:{symbol}"),
            "symbol": symbol,
            "quantity": quantity,
            "value_usd": value_usd,
            "location": str(raw.get("location") or "Unknown"),
            "is_locked": is_locked,
            "locked_quantity": locked_quantity,
            "valuation_status": "valued" if value_usd is not None else "unavailable",
        })

    for position in positions:
        value = position["value_usd"]
        position["weight_pct"] = (value / total_value * 100.0) if value is not None and total_value > 0 else None

    by_symbol: Dict[str, float] = defaultdict(float)
    by_location: Dict[str, float] = defaultdict(float)
    for position in positions:
        if position["value_usd"] is not None:
            by_symbol[position["symbol"]] += position["value_usd"]
            by_location[position["location"]] += position["value_usd"]

    concentrations = {
        "by_symbol_pct": {
            key: value / total_value * 100.0 for key, value in sorted(by_symbol.items())
        } if total_value > 0 else {},
        "by_location_pct": {
            key: value / total_value * 100.0 for key, value in sorted(by_location.items())
        } if total_value > 0 else {},
    }
    concentrations["largest_symbol_pct"] = max(concentrations["by_symbol_pct"].values(), default=None)
    concentrations["symbol_hhi"] = (
        sum((weight / 100.0) ** 2 for weight in concentrations["by_symbol_pct"].values())
        if concentrations["by_symbol_pct"] else None
    )

    canonical = {
        "user_id": str(user_id),
        "source_id": str(source_id),
        "observed_at": timestamp,
        "positions": positions,
    }
    snapshot_id = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    return {
        "contract_version": "portfolio-reference-v1",
        "snapshot_id": snapshot_id,
        **canonical,
        "total_value_usd": total_value,
        "valued_position_count": sum(position["valuation_status"] == "valued" for position in positions),
        "unvalued_symbols": sorted({
            position["symbol"] for position in positions if position["valuation_status"] == "unavailable"
        }),
        "concentrations": concentrations,
    }


def calculate_risk_reference(
    snapshot: Mapping[str, Any],
    price_history: pd.DataFrame,
    *,
    window_days: int = 365,
    min_common_observations: int = 90,
) -> Dict[str, Any]:
    """Estimate covariance risk without hiding assets that lack history.

    Risk contributions are percentages of the covered sleeve's variance. The
    coverage weight always uses original portfolio weights, so it cannot imply
    that omitted assets represent the whole portfolio.
    """
    if window_days < min_common_observations or min_common_observations < 2:
        raise ValueError("Invalid history window")
    if not isinstance(price_history, pd.DataFrame):
        raise TypeError("price_history must be a pandas DataFrame")

    positions = list(snapshot.get("positions") or [])
    symbol_values: Dict[str, float] = defaultdict(float)
    for position in positions:
        value = _finite_non_negative(position.get("value_usd"))
        if value is not None and value > 0:
            symbol_values[str(position.get("symbol", "")).upper()] += value

    total_value = _finite_non_negative(snapshot.get("total_value_usd")) or 0.0
    columns = {str(column).upper(): column for column in price_history.columns}
    history_observations: Dict[str, int] = {}
    eligible = []
    for symbol in sorted(symbol_values):
        original_column = columns.get(symbol)
        observations = int(price_history[original_column].dropna().shape[0]) if original_column is not None else 0
        history_observations[symbol] = observations
        if observations >= min_common_observations + 1:
            eligible.append(symbol)

    covered_value = sum(symbol_values[symbol] for symbol in eligible)
    coverage_weight_pct = covered_value / total_value * 100.0 if total_value > 0 else 0.0
    base = {
        "contract_version": "portfolio-risk-reference-v1",
        "snapshot_id": snapshot.get("snapshot_id"),
        "method": "ledoit_wolf_daily_returns",
        "annualization_days": 365,
        "window_days": window_days,
        "minimum_common_observations": min_common_observations,
        "history_observations": history_observations,
        "eligible_symbols": eligible,
        "uncovered_symbols": sorted(set(symbol_values) - set(eligible)),
        "coverage_weight_pct": coverage_weight_pct,
        "stablecoin_risk_scope": "market_price_history_only",
    }

    if not eligible:
        return {
            **base,
            "status": "unavailable",
            "common_observations": 0,
            "covered_sleeve_volatility_annualized": None,
            "covered_contribution_to_total_portfolio_volatility_annualized": None,
            "risk_contribution_pct_of_covered_risk": {},
            "reason": "No valued asset has sufficient price history",
        }

    selected = pd.DataFrame({
        symbol: pd.to_numeric(price_history[columns[symbol]], errors="coerce") for symbol in eligible
    }).tail(window_days + 1)
    selected = selected.where(selected > 0)
    returns = selected.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    common_observations = int(len(returns))
    base["common_observations"] = common_observations

    if common_observations < min_common_observations:
        return {
            **base,
            "status": "unavailable",
            "covered_sleeve_volatility_annualized": None,
            "covered_contribution_to_total_portfolio_volatility_annualized": None,
            "risk_contribution_pct_of_covered_risk": {},
            "reason": "Insufficient common return observations",
        }

    covariance = LedoitWolf().fit(returns.to_numpy()).covariance_ * 365.0
    original_weights = np.array([symbol_values[symbol] / total_value for symbol in eligible], dtype=float)
    covered_weight = float(original_weights.sum())
    sleeve_weights = original_weights / covered_weight

    sleeve_variance = float(sleeve_weights @ covariance @ sleeve_weights)
    portfolio_covered_variance = float(original_weights @ covariance @ original_weights)
    marginal = covariance @ sleeve_weights
    components = sleeve_weights * marginal
    contribution_denominator = float(components.sum())
    contributions = {
        symbol: float(component / contribution_denominator * 100.0)
        for symbol, component in zip(eligible, components)
    } if contribution_denominator > 0 else {symbol: None for symbol in eligible}

    complete = coverage_weight_pct >= 99.999 and len(eligible) == len(symbol_values)
    return {
        **base,
        "status": "complete" if complete else "partial",
        "covered_sleeve_volatility_annualized": math.sqrt(max(0.0, sleeve_variance)),
        "covered_contribution_to_total_portfolio_volatility_annualized": math.sqrt(max(0.0, portfolio_covered_variance)),
        "risk_contribution_pct_of_covered_risk": contributions,
        "reason": None if complete else "Risk estimate covers only the reported portfolio share",
    }
