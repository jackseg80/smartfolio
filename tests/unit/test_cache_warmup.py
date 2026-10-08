import asyncio
import ast
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from api.cache_warmup import warm_user_caches


@pytest.mark.asyncio
async def test_users_and_sources_are_explicit_and_never_mixed():
    operations = [AsyncMock(return_value={"ok": True}) for _ in range(3)]
    with patch("api.cache_warmup._warm_balances", operations[0]), patch("api.cache_warmup._warm_metrics", operations[1]), patch("api.cache_warmup._warm_risk", operations[2]):
        result = await warm_user_caches(["jack", "demo"], source="cointracking_api")
    assert result == {"attempted": 6, "succeeded": 6, "failures": []}
    for operation in operations:
        assert {tuple(c.args) for c in operation.await_args_list} == {("jack", "cointracking_api"), ("demo", "cointracking_api")}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [JSONResponse({"ok": False}, status_code=401), JSONResponse({"ok": False}), {"ok": False}, {"success": False}, None, HTTPException(401), OSError("secret=do-not-export")])
async def test_real_failures_are_counted_without_response_or_secret_leaks(failure):
    bad = AsyncMock(side_effect=failure) if isinstance(failure, Exception) else AsyncMock(return_value=failure)
    with patch("api.cache_warmup._warm_balances", AsyncMock(return_value={"items": []})), patch("api.cache_warmup._warm_metrics", bad), patch("api.cache_warmup._warm_risk", AsyncMock(return_value={"ok": True})):
        result = await warm_user_caches(["jack"], source="cointracking")
    assert result["attempted"] == 3 and result["succeeded"] == 2
    assert len(result["failures"]) == 1
    assert "jack/cointracking/metrics:" in result["failures"][0]
    assert "do-not-export" not in str(result)


@pytest.mark.asyncio
async def test_timeout_is_reported_and_other_operations_finish():
    async def slow(*args):
        await asyncio.sleep(1)
    with patch("api.cache_warmup._warm_balances", slow), patch("api.cache_warmup._warm_metrics", AsyncMock(return_value={})), patch("api.cache_warmup._warm_risk", AsyncMock(return_value={})):
        result = await warm_user_caches(["demo"], source="cointracking", timeout=0.01)
    assert result["succeeded"] == 2
    assert "TimeoutError" in result["failures"][0]


@pytest.mark.asyncio
async def test_bad_identity_is_rejected_before_any_operation():
    with patch("api.cache_warmup._warm_balances", new_callable=AsyncMock) as operation:
        with pytest.raises(Exception):
            await warm_user_caches(["../other"], source="cointracking")
    operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_portfolio_calculation_keeps_user_source_and_response_contract():
    from api import portfolio_endpoints as api
    resolve = AsyncMock(return_value={"source_used": "cointracking", "items": [{"symbol": "BTC", "amount": 2, "value_usd": 10}]})
    with patch.object(api, "_get_resolve_balances", return_value=resolve), patch.object(api.portfolio_analytics, "calculate_portfolio_metrics", return_value={"total": 10}), patch.object(api.portfolio_analytics, "calculate_performance_metrics", return_value={"pnl": 1}) as performance:
        result = await api.build_portfolio_metrics(user="demo", source="cointracking")
    resolve.assert_awaited_once_with(source="cointracking", user_id="demo", min_usd=1.0)
    assert performance.call_args.kwargs["user_id"] == "demo"
    assert result.status_code == 200
    body = json.loads(result.body)
    assert body["ok"] is True and body["data"]["metrics"] == {"total": 10}


@pytest.mark.asyncio
async def test_shared_risk_function_keeps_same_user_source_cache():
    from api import risk_endpoints as api
    with patch.object(api, "cache_get", return_value={"cache": "existing"}) as get:
        result = await api.build_risk_dashboard(user="demo", source="cointracking")
    assert result == {"cache": "existing"}
    namespace, values = ast.literal_eval(get.call_args.args[1])
    assert namespace == "risk_dashboard_v3"
    assert dict(values)["user"] == "demo" and dict(values)["source"] == "cointracking"


def test_http_routes_still_require_an_authenticated_session(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.portfolio_endpoints import router as portfolio
    from api.risk_endpoints import router as risk
    monkeypatch.setenv("AUTH_MODE", "dual")
    app = FastAPI()
    app.include_router(portfolio)
    app.include_router(risk)
    with TestClient(app) as client:
        for path in ["/api/portfolio/metrics?source=cointracking", "/api/risk/dashboard?source=cointracking"]:
            assert client.get(path, headers={"X-User": "jack"}).status_code == 401


@pytest.mark.asyncio
async def test_crypto_route_and_scheduler_use_same_cache_operation():
    from api import crypto_toolbox_endpoints as api
    with patch.object(api, "_get_data", new_callable=AsyncMock, return_value={"fresh": True}) as shared:
        assert await api.get_crypto_toolbox_data(force=True) == {"fresh": True}
        assert await api.get_cached_crypto_toolbox_data(force=True) == {"fresh": True}
    assert shared.await_count == 2
    assert all(c.kwargs == {"force": True} for c in shared.await_args_list)


@pytest.mark.asyncio
async def test_successful_json_response_is_a_successful_cache_operation():
    with patch("api.cache_warmup._warm_balances", AsyncMock(return_value={})), patch("api.cache_warmup._warm_metrics", AsyncMock(return_value=JSONResponse({"ok": True, "data": {}}))), patch("api.cache_warmup._warm_risk", AsyncMock(return_value={})):
        result = await warm_user_caches(["demo"], source="cointracking")
    assert result == {"attempted": 3, "succeeded": 3, "failures": []}
