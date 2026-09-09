from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import pytest

from services.decision.portfolio_reference import build_portfolio_snapshot, calculate_risk_reference
from api.risk_endpoints import get_portfolio_reference


OBSERVED_AT = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


def test_snapshot_is_deterministic_and_keeps_unvalued_positions():
    rows = [
        {"symbol": "btc", "amount": 1, "value_usd": 60_000, "location": "Cold"},
        {"symbol": "ETH", "amount": 10, "value_usd": 30_000, "location": "Exchange", "locked": True},
        {"symbol": "MYSTERY", "amount": 5, "value_usd": None, "location": "Wallet"},
    ]

    first = build_portfolio_snapshot(rows, user_id="jack", source_id="cointracking", observed_at=OBSERVED_AT)
    second = build_portfolio_snapshot(rows, user_id="jack", source_id="cointracking", observed_at=OBSERVED_AT)

    assert first == second
    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["total_value_usd"] == 90_000
    assert first["unvalued_symbols"] == ["MYSTERY"]
    assert first["positions"][2]["weight_pct"] is None
    assert first["concentrations"]["largest_symbol_pct"] == pytest.approx(66.6666667)


def test_snapshot_does_not_treat_false_string_as_locked():
    snapshot = build_portfolio_snapshot(
        [{"symbol": "BTC", "amount": 1, "value_usd": 100, "locked": "false"}],
        user_id="jack",
        source_id="cointracking",
    )
    assert snapshot["positions"][0]["is_locked"] is False


def test_risk_reference_reports_partial_coverage_without_renormalizing_it():
    snapshot = build_portfolio_snapshot(
        [
            {"symbol": "BTC", "value_usd": 60},
            {"symbol": "ETH", "value_usd": 30},
            {"symbol": "NOHISTORY", "value_usd": 10},
        ],
        user_id="jack",
        source_id="cointracking",
        observed_at=OBSERVED_AT,
    )
    dates = pd.date_range("2025-01-01", periods=121, freq="D")
    prices = pd.DataFrame({
        "BTC": 100 * np.exp(np.linspace(0, 0.30, len(dates)) + np.sin(np.arange(len(dates))) * 0.01),
        "ETH": 100 * np.exp(np.linspace(0, 0.20, len(dates)) + np.cos(np.arange(len(dates))) * 0.015),
    }, index=dates)

    result = calculate_risk_reference(snapshot, prices, window_days=120, min_common_observations=90)

    assert result["status"] == "partial"
    assert result["coverage_weight_pct"] == pytest.approx(90.0)
    assert result["uncovered_symbols"] == ["NOHISTORY"]
    assert result["common_observations"] == 120
    assert sum(result["risk_contribution_pct_of_covered_risk"].values()) == pytest.approx(100.0)
    assert result["covered_contribution_to_total_portfolio_volatility_annualized"] < result["covered_sleeve_volatility_annualized"]


def test_risk_reference_is_unavailable_below_common_history_minimum():
    snapshot = build_portfolio_snapshot(
        [{"symbol": "BTC", "value_usd": 100}],
        user_id="jack",
        source_id="cointracking",
        observed_at=OBSERVED_AT,
    )
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    prices = pd.DataFrame({"BTC": np.linspace(100, 120, len(dates))}, index=dates)

    result = calculate_risk_reference(snapshot, prices, min_common_observations=90)

    assert result["status"] == "unavailable"
    assert result["coverage_weight_pct"] == 0
    assert result["risk_contribution_pct_of_covered_risk"] == {}


@pytest.mark.asyncio
async def test_api_reference_uses_authenticated_user_and_resolved_source(monkeypatch):
    calls = {}

    async def resolve_current_balances(*, user_id, source):
        calls.update(user_id=user_id, source=source)
        return {
            "source_used": "cointracking_csv",
            "items": [
                {"symbol": "BTC", "amount": 1, "value_usd": 60, "location": "Cold"},
                {"symbol": "UNKNOWN", "amount": 2, "value_usd": 40, "location": "Wallet"},
            ],
        }

    start = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    btc_history = [(start + day * 86400, 100 + day) for day in range(100)]

    monkeypatch.setattr("services.balance_service.balance_service.resolve_current_balances", resolve_current_balances)
    monkeypatch.setattr(
        "services.price_history.get_cached_history",
        lambda symbol, days: btc_history if symbol == "BTC" else [],
    )

    response = await get_portfolio_reference(source="cointracking", history_days=365, user="jack")
    body = json.loads(response.body)

    assert calls == {"user_id": "jack", "source": "cointracking"}
    assert body["data"]["portfolio"]["user_id"] == "jack"
    assert body["data"]["portfolio"]["source_id"] == "cointracking_csv"
    assert body["data"]["risk_reference"]["status"] == "partial"
    assert body["data"]["risk_reference"]["coverage_weight_pct"] == pytest.approx(60.0)
