"""Regression tests for actual user-reported preview failures."""

import pandas as pd
import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from api.deps import get_required_user
from services.ml.cycle_diagnostics import describe_cycles


def test_cycle_comparisons_do_not_invent_missing_halving_closes():
    close = pd.Series([100.0, 80.0, 120.0], index=pd.date_range("2020-05-10", periods=3, tz="UTC"))
    result = describe_cycles(close, {"provider": "verified_fixture", "dataset_id": "fixture"})
    cycle = result["cycles"][-1]
    assert cycle["anchor_available"]
    assert cycle["return_since_halving"] == 0.5
    assert cycle["points"][0]["drawdown"] == 0
    assert result["cycles"][0]["return_since_halving"] is None


def test_cycle_drawdowns_at_a_date_do_not_use_later_peaks():
    index = pd.date_range("2024-04-20", periods=5, tz="UTC")
    original = describe_cycles(
        pd.Series([100.0, 90.0, 110.0, 50.0, 500.0], index=index),
        {"provider": "fixture", "dataset_id": "fixture"},
    )
    mutated = describe_cycles(
        pd.Series([100.0, 90.0, 110.0, 9000.0, 1.0], index=index),
        {"provider": "fixture", "dataset_id": "fixture"},
    )
    assert original["cycles"][-1]["points"][:3] == mutated["cycles"][-1]["points"][:3]


@pytest.mark.parametrize("path", ["/optimize", "/optimize-advanced", "/analyze"])
def test_optimizer_reads_authenticated_user_and_exact_selected_source(monkeypatch, path):
    from api import portfolio_optimization_endpoints as endpoints

    calls = []

    async def balances(**kwargs):
        calls.append(kwargs)
        raise HTTPException(404, "No eligible selected holdings")

    monkeypatch.setattr(endpoints, "get_unified_filtered_balances", balances)
    app = FastAPI()
    app.include_router(endpoints.router)

    def identity(x_user: str = Header(None, alias="X-User")):
        if not x_user:
            raise HTTPException(401, "Session required")
        return x_user

    app.dependency_overrides[get_required_user] = identity
    client = TestClient(app)
    for user in ["alice", "bob"]:
        for source in ["source-a", "source-b"]:
            url = "/api/portfolio/optimization" + path + "?source=" + source
            response = (
                client.get(url, headers={"X-User": user})
                if path == "/analyze"
                else client.post(url, headers={"X-User": user}, json={})
            )
            assert response.status_code == 404
            assert calls[-1]["user_id"] == user and calls[-1]["source"] == source
    assert (
        client.get("/api/portfolio/optimization" + path)
        if path == "/analyze"
        else client.post("/api/portfolio/optimization" + path, json={})
    ).status_code == 401


def test_unimplemented_optimizer_backtest_never_returns_plausible_metrics():
    from api import portfolio_optimization_endpoints as endpoints

    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[get_required_user] = lambda: "alice"
    result = TestClient(app).post("/api/portfolio/optimization/backtest", json={})
    assert result.status_code == 501 and "backtest_summary" not in result.json()


@pytest.mark.asyncio
async def test_external_diagnostic_recovers_closed_browser_before_restart(monkeypatch):
    from types import SimpleNamespace
    import api.crypto_toolbox_endpoints as collector
    import asyncio
    events = []
    monkeypatch.setattr(collector, "_browser_start_lock", asyncio.Lock())
    monkeypatch.setattr(collector, "_browser", SimpleNamespace(is_connected=lambda: False))
    async def stop():
        events.append("stop")
        monkeypatch.setattr(collector, "_browser", None)
    async def start():
        assert collector._browser is None
        events.append("start")
        monkeypatch.setattr(collector, "_browser", SimpleNamespace(is_connected=lambda: True))
    monkeypatch.setattr(collector, "shutdown_playwright", stop)
    monkeypatch.setattr(collector, "_startup_playwright_locked", start)
    await collector.startup_playwright()
    await collector.startup_playwright()
    assert events == ["stop", "start"]


@pytest.mark.asyncio
async def test_external_diagnostic_serializes_concurrent_browser_starts(monkeypatch):
    from types import SimpleNamespace
    import api.crypto_toolbox_endpoints as collector
    import asyncio
    calls = []
    monkeypatch.setattr(collector, "_browser_start_lock", asyncio.Lock())
    monkeypatch.setattr(collector, "_browser", None)
    async def start():
        calls.append("start")
        await asyncio.sleep(0)
        monkeypatch.setattr(collector, "_browser", SimpleNamespace(is_connected=lambda: True))
    monkeypatch.setattr(collector, "_startup_playwright_locked", start)
    await asyncio.gather(collector.startup_playwright(),collector.startup_playwright())
    assert calls == ["start"]


@pytest.mark.asyncio
async def test_failed_external_browser_start_cleans_up_without_fake_data(monkeypatch):
    import api.crypto_toolbox_endpoints as collector
    import asyncio
    calls = []
    monkeypatch.setattr(collector, "_browser_start_lock", asyncio.Lock())
    monkeypatch.setattr(collector, "_browser", None)
    async def start():
        raise RuntimeError("Driver unavailable")
    async def stop():
        calls.append("stop")
    monkeypatch.setattr(collector, "_startup_playwright_locked", start)
    monkeypatch.setattr(collector, "shutdown_playwright", stop)
    with pytest.raises(RuntimeError,match="Driver unavailable"):
        await collector.startup_playwright()
    assert calls == ["stop"]
