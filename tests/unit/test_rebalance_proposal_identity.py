from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

import api.main as main_module


@pytest.fixture
def resolved_balances(monkeypatch):
    async def fake_resolve(**_kwargs):
        return {"items": [{"symbol": "BTC", "value_usd": 60}, {"symbol": "USDC", "value_usd": 40}]}

    monkeypatch.setattr("api.unified_data.get_unified_filtered_balances", fake_resolve)


def proposal_payload(**updates):
    payload = {
        "dynamic_targets_pct": {"BTC": 50, "Stablecoins": 50},
        "target_origin": "unified_suggested_allocation",
        "portfolio_user_id": "jack",
        "portfolio_source_id": "cointracking",
        "allocation_snapshot": {"total_usd": 100, "weights_pct": {"BTC": 60, "Stablecoins": 40}},
        "proposal_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(updates)
    return payload


@pytest.mark.asyncio
async def test_rejects_suggested_allocation_for_another_user(resolved_balances):
    with pytest.raises(HTTPException) as error:
        await main_module.rebalance_plan(
            source="cointracking", dynamic_targets=True,
            payload=proposal_payload(portfolio_user_id="other"), user="jack",
        )
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_rejects_dynamic_allocation_without_verified_origin(resolved_balances):
    with pytest.raises(HTTPException) as error:
        await main_module.rebalance_plan(
            source="cointracking", dynamic_targets=True,
            payload={"dynamic_targets_pct": {"BTC": 50, "Stablecoins": 50}}, user="jack",
        )
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_rejects_when_resolver_uses_another_source(monkeypatch):
    async def fake_resolve(**_kwargs):
        return {"source_used": "manual_crypto", "items": [{"symbol": "BTC", "value_usd": 100}]}

    monkeypatch.setattr("api.unified_data.get_unified_filtered_balances", fake_resolve)
    with pytest.raises(HTTPException) as error:
        await main_module.rebalance_plan(
            source="cointracking", dynamic_targets=True,
            payload=proposal_payload(), user="jack",
        )
    assert error.value.status_code == 409
    assert "source" in error.value.detail.lower()


@pytest.mark.asyncio
async def test_rejects_stale_suggested_allocation(resolved_balances):
    stale = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    with pytest.raises(HTTPException) as error:
        await main_module.rebalance_plan(
            source="cointracking", dynamic_targets=True,
            payload=proposal_payload(proposal_timestamp=stale), user="jack",
        )
    assert error.value.status_code == 409
    assert "stale" in error.value.detail.lower()


@pytest.mark.asyncio
async def test_rejects_suggested_allocation_after_portfolio_value_changes(resolved_balances):
    with pytest.raises(HTTPException) as error:
        await main_module.rebalance_plan(
            source="cointracking", dynamic_targets=True,
            payload=proposal_payload(allocation_snapshot={"total_usd": 90}), user="jack",
        )
    assert error.value.status_code == 409
    assert "changed" in error.value.detail.lower()


@pytest.mark.asyncio
async def test_rejects_suggested_allocation_after_composition_changes(monkeypatch):
    async def fake_resolve(**_kwargs):
        return {
            "source_used": "cointracking",
            "items": [{"symbol": "BTC", "value_usd": 40}, {"symbol": "USDC", "value_usd": 60}],
        }

    monkeypatch.setattr("api.unified_data.get_unified_filtered_balances", fake_resolve)
    with pytest.raises(HTTPException) as error:
        await main_module.rebalance_plan(
            source="cointracking", dynamic_targets=True,
            payload=proposal_payload(), user="jack",
        )
    assert error.value.status_code == 409
    assert "composition" in error.value.detail.lower()


@pytest.mark.asyncio
async def test_rejects_negative_targets(resolved_balances):
    payload = {"group_targets_pct": {"BTC": 110, "Stablecoins": -10}}
    with pytest.raises(HTTPException) as error:
        await main_module.rebalance_plan(
            source="cointracking", dynamic_targets=False, payload=payload, user="jack",
        )
    assert error.value.status_code == 422
