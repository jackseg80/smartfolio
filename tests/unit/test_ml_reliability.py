"""Meaningful failure, isolation and causality gates for the ML overhaul."""
import asyncio
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.schemas.ml_contract import Availability, Horizon, ModelType, UnifiedPrediction
from services.ml.reliability import CapabilityService, FEATURES, PROTOCOL_ID, daily_features, future_targets, infer_estimator, digest, code_version
from services.ml.risk_evaluation import fit_estimator, partitions, evaluate

NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_external_provider_network(monkeypatch):
    async def missing(self):
        return UnifiedPrediction(asset="CRYPTO_MARKET", nature="diagnostic", target="external_fear_greed", reason="Provider unavailable in isolated test")
    monkeypatch.setattr(CapabilityService, "external_sentiment", missing)


def history(root, asset="BTC", market="crypto", *, constant=False, stale=False, days=3000):
    end = pd.Timestamp(NOW)-pd.Timedelta(days=20 if stale else 1)
    dates = pd.date_range(end=end, periods=days, freq="D", tz="UTC")
    close = np.ones(days)*100 if constant else 100*np.exp(np.cumsum(np.random.default_rng(3).normal(0, .02, days)))
    directory = root / "data/ml_verified" / market
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{asset}.csv"
    pd.DataFrame(dict(date=dates.strftime("%Y-%m-%d"), close=close)).to_csv(path, index=False)
    receipt = dict(file=path.name, sha256=digest(path), dataset_id="test-daily-v1", provider="verified_test_fixture", adjustment_policy="test adjusted closes")
    (directory / f"{asset}.json").write_text(json.dumps(receipt))
    return pd.Series(close, index=dates), receipt


def artifact(root, *, method="persistence", asset="BTC", market="crypto", horizon=7):
    directory = root / "models/validated_risk"
    directory.mkdir(parents=True, exist_ok=True)
    model = dict(schema_version=1, method=method, market=market, asset=asset, horizon=horizon,
        annualization=365 if market == "crypto" else 252, features=FEATURES,
        dataset_id="test-daily-v1", dataset_sha256=digest(root / "data/ml_verified" / market / f"{asset}.csv"), code_version=code_version(), training_start="2018-01-01T00:00:00Z", training_end="2024-01-01T00:00:00Z",
        validation=dict(state="retrospectively_validated", reason="Fixture validated in isolated tests", protocol_id=PROTOCOL_ID))
    path = directory / f"{market}_{asset}_{horizon}d.json"
    path.write_text(json.dumps(model))
    registry_path = directory / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    registry[path.name] = {"sha256": digest(path)}
    registry_path.write_text(json.dumps(registry))
    return path, model


@pytest.mark.asyncio
async def test_missing_stale_zero_and_incompatible_artifacts(tmp_path):
    service = CapabilityService(tmp_path, now=lambda: NOW)
    missing = await service.result("BTC", "crypto", ModelType.VOLATILITY, Horizon.D7)
    assert missing.value is None and missing.availability == Availability.UNAVAILABLE
    history(tmp_path, stale=True)
    artifact(tmp_path)
    stale = await service.result("BTC", "crypto", ModelType.VOLATILITY, Horizon.D7)
    assert stale.value is None and "stale" in stale.reason
    history(tmp_path, constant=True)
    artifact(tmp_path)
    zero = await service.result("BTC", "crypto", ModelType.VOLATILITY, Horizon.D7)
    assert zero.value == 0 and zero.availability == Availability.AVAILABLE
    assert zero.quality.confidence is None and zero.uncertainty is None
    path, model = artifact(tmp_path)
    model["features"] = ["future_return"]
    path.write_text(json.dumps(model))
    incompatible = await service.result("BTC", "crypto", ModelType.VOLATILITY, Horizon.D7)
    assert incompatible.value is None and "incompatible" in incompatible.reason
    path.write_text("broken JSON")
    broken = await service.result("BTC", "crypto", ModelType.VOLATILITY, Horizon.D7)
    assert broken.value is None and broken.availability == Availability.UNAVAILABLE


@pytest.mark.asyncio
async def test_partial_overview_and_no_training_on_reads(tmp_path, monkeypatch):
    service = CapabilityService(tmp_path, now=lambda: NOW)
    history(tmp_path)
    artifact(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("A read attempted training")
    monkeypatch.setattr("services.ml.risk_evaluation.fit_estimator", forbidden)
    overview = await service.overview("alice", "source-a", ["BTC", "ETH"], "crypto")
    assert overview["user_id"] == "alice" and overview["source"] == "source-a"
    assert any(r["value"] is not None for r in overview["results"])
    assert any(r["value"] is None for r in overview["results"])
    assert overview["counts"]["files_present"] == 1
    assert overview["counts"]["models_loaded"] == 1
    assert overview["counts"]["successful_inferences"] == 1
    assert overview["governance_integration"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("horizon", [Horizon.H1, Horizon.H4, Horizon.D1, Horizon.D90, None])
async def test_unsupported_forecast_horizons_are_not_substituted(tmp_path, horizon):
    result = await CapabilityService(tmp_path).result("BTC", "crypto", ModelType.VOLATILITY, horizon)
    assert result.value is None and result.target_date is None


def test_features_and_historical_inference_survive_future_mutation(tmp_path):
    close, _ = history(tmp_path)
    original = daily_features(close, "crypto").dropna()
    boundary = original.index[1700]
    changed = close.copy()
    changed.loc[changed.index > boundary] *= np.linspace(1, 5, (changed.index > boundary).sum())
    altered = daily_features(changed, "crypto").dropna()
    pd.testing.assert_frame_equal(original.loc[:boundary], altered.loc[:boundary])
    for method in ("persistence", "ewma", "ridge"):
        train = original.iloc[:1000].join(future_targets(close, "crypto", 7)).dropna()
        model = fit_estimator(method, train)
        np.testing.assert_array_equal(infer_estimator(model, original.loc[:boundary], 7), infer_estimator(model, altered.loc[:boundary], 7))


def test_future_volatility_definition_and_calendar_targets():
    dates = pd.date_range("2020-01-01", periods=50, tz="UTC")
    close = pd.Series(np.exp(np.arange(50)*.03 + np.sin(np.arange(50))*.1), index=dates)
    result = future_targets(close, "crypto", 7)
    returns = np.log(close.iloc[1:8].to_numpy()/close.iloc[:7].to_numpy())
    assert result.iloc[0].target == pytest.approx(np.std(returns, ddof=0)*np.sqrt(365))
    assert result.iloc[0].target_end == dates[7]
    stock = close[close.index.weekday < 5]
    target = future_targets(stock, "stocks", 7).iloc[0]
    assert target.target_end >= stock.index[0]+pd.Timedelta(days=7)
    j = stock.index.get_loc(target.target_end)
    actual = np.log(stock.iloc[1:j+1].to_numpy()/stock.iloc[:j].to_numpy())
    assert target.target == pytest.approx(np.std(actual, ddof=0)*np.sqrt(252))


def test_target_end_purge_and_train_only_transform(tmp_path):
    close, _ = history(tmp_path)
    frame = daily_features(close, "crypto").join(future_targets(close, "crypto", 30)).dropna()
    calib = frame.index[1100]
    test = calib+pd.Timedelta(days=183)
    train, calibration, validation = partitions(frame, calib, test, test+pd.Timedelta(days=182))
    assert train.target_end.max() < calib and calibration.target_end.max() < test
    assert validation.target_end.max() <= test+pd.Timedelta(days=182)
    original = fit_estimator("ridge", train)
    frame.loc[frame.index >= calib, FEATURES] *= 100
    same = fit_estimator("ridge", frame.loc[train.index])
    assert original == same


def test_atomic_artifact_evaluation_is_reproducible(tmp_path, monkeypatch):
    # Bounded meaningful end-to-end protocol test; the neural candidate is
    # exercised separately to keep this regression test fast.
    close, _ = history(tmp_path, days=2400)
    service = CapabilityService(tmp_path, now=lambda: NOW)
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)
    first = evaluate(service, "crypto", "BTC", 7, publish=True)
    second = evaluate(service, "crypto", "BTC", 7, publish=True)
    assert first["state"] == "retrospectively_validated"
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert len(first["folds"]) >= 3
    assert all(f["train_end"] < f["calibration_start"] < f["test_start"] for f in first["folds"])


def test_authenticated_adapters_two_users_two_sources_and_expiration(tmp_path, monkeypatch):
    from api.ml import unified_contract_endpoints as endpoints
    from api.deps import get_required_user
    from fastapi import Header, HTTPException
    service = CapabilityService(tmp_path, now=lambda: NOW)
    history(tmp_path)
    artifact(tmp_path)
    monkeypatch.setattr(endpoints, "capability_service", service)
    app = FastAPI()
    app.include_router(endpoints.router, prefix="/api/ml")
    def session(x_user: str = Header(None, alias="X-User"), authorization: str = Header(None)):
        if not authorization:
            raise HTTPException(401, "Expired session")
        if authorization != f"Bearer {x_user}":
            raise HTTPException(403, "Identity mismatch")
        return x_user
    app.dependency_overrides[get_required_user] = session
    client = TestClient(app)
    for user in ("alice", "bob"):
        for source in ("source-a", "source-b"):
            response = client.get(f"/api/ml/overview?assets=BTC&source={source}", headers={"X-User": user, "Authorization": f"Bearer {user}"})
            assert response.status_code == 200
            assert response.json()["user_id"] == user and response.json()["source"] == source
    assert client.get("/api/ml/overview").status_code == 401
    assert client.get("/api/ml/overview", headers={"X-User": "bob", "Authorization": "Bearer alice"}).status_code == 403
    partial = client.post("/api/ml/unified/predict", headers={"X-User": "alice", "Authorization": "Bearer alice"}, json={"assets": ["BTC", "ETH"], "model_type": "volatility", "horizon": "7d", "source": "source-b"}).json()
    assert partial["success"] and partial["failed_assets"] == ["ETH"]
    assert partial["predictions"][1]["value"] is None


def test_no_pickle_loading_and_input_paths_are_bounded(tmp_path):
    service = CapabilityService(tmp_path, now=lambda: NOW)
    with pytest.raises(ValueError, match="Invalid asset"):
        service.history("crypto", "../alice")
    history(tmp_path)
    receipt_path = tmp_path / "data/ml_verified/crypto/BTC.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["file"] = "../../outside.csv"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        service.history("crypto", "BTC")

def test_lstm_is_reproducible_and_uses_only_past_context(tmp_path):
    pytest.importorskip("torch")
    close, _ = history(tmp_path, days=700)
    original = daily_features(close, "crypto").join(future_targets(close, "crypto", 7)).dropna()
    train = original.iloc[:300]
    model = fit_estimator("lstm", train)
    assert model == fit_estimator("lstm", train)
    changed = original.copy()
    changed.iloc[400:, changed.columns.get_indexer(FEATURES)] *= 50
    a = infer_estimator(model, original, 7)
    b = infer_estimator(model, changed, 7)
    np.testing.assert_array_equal(a[:400], b[:400])
    assert model["mean"] == train[FEATURES].mean().tolist()
    assert not np.array_equal(a[400:], b[400:])


@pytest.mark.asyncio
async def test_intervals_require_confirmation_coverage_and_sealed_artifact(tmp_path):
    service = CapabilityService(tmp_path, now=lambda: NOW)
    history(tmp_path)
    path, model = artifact(tmp_path)
    for coverage, published in ((.80, False), (.90, True), (.98, False)):
        model["interval"] = dict(level=.9, width=.1, confirmation_coverage=coverage)
        path.write_text(json.dumps(model))
        (path.parent / "registry.json").write_text(json.dumps({path.name: {"sha256": digest(path)}}))
        result = await service.result("BTC", "crypto", ModelType.VOLATILITY, Horizon.D7)
        assert (result.uncertainty is not None) == published
        if published:
            assert result.uncertainty.nominal_coverage == .9
            assert result.uncertainty.confirmation_coverage == coverage
    model["interval"]["width"] = .2
    path.write_text(json.dumps(model))
    tampered = await service.result("BTC", "crypto", ModelType.VOLATILITY, Horizon.D7)
    assert tampered.value is None and tampered.availability == Availability.UNAVAILABLE


@pytest.mark.asyncio
async def test_opt_in_journal_isolated_and_deduplicated(tmp_path, monkeypatch):
    import sqlite3
    service = CapabilityService(tmp_path, now=lambda: NOW)
    history(tmp_path)
    artifact(tmp_path)
    result = await service.result("BTC", "crypto", ModelType.VOLATILITY, Horizon.D7)
    monkeypatch.delenv("ML_INFERENCE_JOURNAL", raising=False)
    service.journal("alice", "source-a", [result])
    assert not (tmp_path / "data/users").exists()
    monkeypatch.setenv("ML_INFERENCE_JOURNAL", "1")
    for user in ("alice", "bob"):
        for source in ("source-a", "source-b"):
            service.journal(user, source, [result])
            service.journal(user, source, [result])
        with sqlite3.connect(tmp_path / f"data/users/{user}/ml/inferences.sqlite") as connection:
            values = [json.loads(row[0]) for row in connection.execute("SELECT result_json FROM inferences")]
        assert len(values) == 2
        assert {row["user_id"] for row in values} == {user}
        assert {row["source"] for row in values} == {"source-a", "source-b"}


def test_real_jwt_identity_and_admin_role_on_ml_routes(tmp_path, monkeypatch):
    from api.ml import unified_contract_endpoints as endpoints, training_endpoints
    import api.deps as deps
    from api.auth_router import create_access_token
    from datetime import timedelta
    monkeypatch.setenv("AUTH_MODE", "legacy")
    monkeypatch.setenv("DEV_OPEN_API", "0")
    monkeypatch.setenv("REQUIRE_JWT", "1")
    monkeypatch.setenv("JWT_SECRET_KEY", "isolated-ml-test-secret-with-sufficient-length")
    monkeypatch.setattr(deps, "is_allowed_user", lambda user: user in ("alice", "bob", "admin"))
    monkeypatch.setattr(deps, "get_user_info", lambda user: {"status": "active", "roles": ["admin"] if user == "admin" else ["viewer"]})
    monkeypatch.setattr(deps, "is_access_token_revoked", lambda payload: False)
    monkeypatch.setattr(endpoints, "capability_service", CapabilityService(tmp_path, now=lambda: NOW))
    calls = []
    async def explicit(*args, **kwargs):
        calls.append(args)
    monkeypatch.setattr(training_endpoints, "_train_models_background", explicit)
    app = FastAPI()
    app.include_router(endpoints.router, prefix="/api/ml")
    app.include_router(training_endpoints.router, prefix="/api/ml")
    client = TestClient(app)
    token = create_access_token({"sub": "alice"})
    assert client.get("/api/ml/overview", headers={"X-User": "alice", "Authorization": "Bearer "+token}).status_code == 200
    assert client.get("/api/ml/overview", headers={"X-User": "bob", "Authorization": "Bearer "+token}).status_code == 403
    expired = create_access_token({"sub": "alice"}, expires_delta=timedelta(seconds=-1))
    assert client.get("/api/ml/overview", headers={"X-User": "alice", "Authorization": "Bearer "+expired}).status_code == 401
    payload = {"assets": ["BTC"], "market": "crypto"}
    assert client.post("/api/ml/train", json=payload, headers={"X-User": "alice", "Authorization": "Bearer "+token}).status_code == 403
    assert not calls
    admin_token = create_access_token({"sub": "admin"})
    assert client.post("/api/ml/train", json=payload, headers={"X-User": "admin", "Authorization": "Bearer "+admin_token}).status_code == 200
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_legacy_stock_reads_do_not_train_even_with_force_retrain(monkeypatch):
    from services.ml.bourse.stocks_adapter import StocksMLAdapter
    from services.ml.auto_trainer import MLAutoTrainer
    from services.ml.reliability import capability_service
    async def missing(*args, **kwargs):
        return UnifiedPrediction(asset="SPY", market="stocks", nature="diagnostic", reason="Verified observations unavailable")
    monkeypatch.setattr(capability_service, "result", missing)
    adapter = StocksMLAdapter()
    def forbidden(*args, **kwargs):
        raise AssertionError("Implicit training")
    monkeypatch.setattr(adapter.regime_detector, "train_regime_model", forbidden)
    result = await adapter.detect_market_regime(force_retrain=True)
    assert result["confidence"] is None and result["regime_probabilities"] == {}
    trainer = MLAutoTrainer()
    monkeypatch.setattr(trainer.scheduler, "start", forbidden)
    assert trainer.start() is False
    assert trainer.get_status()["running"] is False

@pytest.mark.asyncio
async def test_stock_target_session_date_has_explicit_utc_timezone(tmp_path):
    import exchange_calendars as xcals
    dates = xcals.get_calendar("XNYS").sessions_in_range("2018-01-01", "2026-09-29").tz_localize("UTC")
    directory = tmp_path / "data/ml_verified/stocks"
    directory.mkdir(parents=True)
    path = directory / "SPY.csv"
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": 100*np.exp(np.arange(len(dates))*.001+np.sin(np.arange(len(dates)))*.01)}).to_csv(path,index=False)
    (directory/"SPY.json").write_text(json.dumps(dict(file="SPY.csv",sha256=digest(path),dataset_id="test-daily-v1",provider="verified_test_fixture",exchange_calendar="XNYS",adjustment_policy="test adjusted closes")))
    artifact(tmp_path,market="stocks",asset="SPY")
    result = await CapabilityService(tmp_path,now=lambda:NOW).result("SPY","stocks",ModelType.VOLATILITY,Horizon.D7)
    assert result.availability == Availability.AVAILABLE
    assert result.target_date.isoformat() == "2026-10-06T00:00:00+00:00"

@pytest.mark.asyncio
async def test_personal_selection_is_bound_to_user_source_and_exact_instrument(tmp_path, monkeypatch):
    from services.ml import portfolio_context as context
    from api.ml import unified_contract_endpoints as endpoints
    from api.deps import get_required_user
    calls=[]
    async def read(user,source,market,file_key=None):
        calls.append((user,source,market,file_key))
        return dict(source=source,data_as_of="2026-09-22",items=[{"symbol":"WSTETH","value_usd":400},{"symbol":"ETH","value_usd":300},{"symbol":"SOL2","value_usd":100}])
    monkeypatch.setattr(context,"read_context",read)
    monkeypatch.setattr(endpoints,"capability_service",CapabilityService(tmp_path,now=lambda:NOW))
    history(tmp_path,asset="ETH")
    artifact(tmp_path,asset="ETH")
    app=FastAPI()
    app.include_router(endpoints.router,prefix="/api/ml")
    app.dependency_overrides[get_required_user]=lambda:"alice"
    response=TestClient(app).get("/api/ml/overview?mode=portfolio&source=selected-api&limit=3").json()
    assert calls==[("alice","selected-api","crypto",None)]
    assert response["scope"]=="selected_authenticated_portfolio"
    assert response["portfolio_context"]["held_positions"]==3
    assert response["portfolio_context"]["selected_assets"]==3
    assert all(r["value"] is None for r in response["results"] if r["asset"] in ("SOL2","WSTETH"))
    assert any(r["value"] is not None for r in response["results"] if r["asset"]=="ETH")
    assert context.stock_symbol("SMH:xlon")=="SMH.L"
    assert context.stock_symbol("SMH:arcx")=="SMH"
    assert context.stock_symbol("BRKb:xnys")=="BRK-B"
    with pytest.raises(ValueError):
        context.stock_symbol("AAA:unknown")


@pytest.mark.asyncio
async def test_private_preview_snapshot_never_crosses_identity_or_source(tmp_path,monkeypatch):
    from services.ml.portfolio_context import read_context
    path=tmp_path/"fixture.json"
    path.write_text(json.dumps(dict(user_id="alice",observed_at="2026-09-30",crypto=dict(source="a",items=[{"symbol":"BTC"}]))))
    monkeypatch.setenv("ML_PORTFOLIO_SNAPSHOT",str(path))
    assert (await read_context("alice","a","crypto"))["items"][0]["symbol"]=="BTC"
    for user,source in (("bob","a"),("alice","b")):
        with pytest.raises(ValueError,match="matches"):
            await read_context(user,source,"crypto")


@pytest.mark.asyncio
async def test_selected_saxo_file_never_falls_back_to_newest(tmp_path, monkeypatch):
    from services.ml import portfolio_context as context
    import adapters.saxo_adapter as adapter
    monkeypatch.delenv("ML_PORTFOLIO_SNAPSHOT", raising=False)
    monkeypatch.setattr(context, "ROOT", tmp_path)
    calls = []
    def snapshot(user_id, file_key):
        calls.append((user_id, file_key))
        return dict(portfolios=[dict(last_updated="2026-09-22", positions=[dict(symbol="MSFT:xnas", market_value_usd=100)])])
    monkeypatch.setattr(adapter, "_load_snapshot", snapshot)
    for user in ("alice", "bob"):
        directory = tmp_path / "data/users" / user / "saxobank/data"
        directory.mkdir(parents=True)
        (directory/"selected.csv").write_text("different-"+user)
        (directory/"newest.csv").write_text("other")
        (directory.parent.parent/"config.json").write_text(json.dumps(dict(sources=dict(bourse=dict(selected_csv_file="selected.csv")))))
        result=await context.read_context(user,"saxobank","stocks")
        assert result["file_key"]=="selected.csv"
        assert result["source_sha256"]==hashlib.sha256(("different-"+user).encode()).hexdigest()
        assert result["data_as_of"]=="2026-09-22"
    assert calls==[("alice","selected.csv"),("bob","selected.csv")]
    (tmp_path/"data/users/alice/saxobank/data/selected.csv").unlink()
    with pytest.raises(ValueError,match="missing"):
        await context.read_context("alice","saxobank","stocks")
    assert len(calls)==2


@pytest.mark.asyncio
async def test_resolved_source_cannot_silently_change(tmp_path, monkeypatch):
    from services.ml.portfolio_context import read_context
    from services.balance_service import balance_service
    monkeypatch.delenv("ML_PORTFOLIO_SNAPSHOT", raising=False)
    calls=[]
    async def resolve(user_id,source):
        calls.append((user_id,source))
        return dict(source_used="manual_crypto",items=[dict(symbol="BTC",value_usd=1)])
    monkeypatch.setattr(balance_service,"resolve_current_balances",resolve)
    with pytest.raises(ValueError,match="differs"):
        await read_context("alice","cointracking_api","crypto")
    assert calls==[("alice","cointracking_api")]


@pytest.mark.asyncio
async def test_missing_diagnostic_and_forecast_keep_their_target(tmp_path):
    service=CapabilityService(tmp_path,now=lambda:NOW)
    diagnostic=await service.result("BTC","crypto",ModelType.REGIME)
    forecast=await service.result("BTC","crypto",ModelType.VOLATILITY,Horizon.D7)
    assert diagnostic.nature=="diagnostic" and diagnostic.target=="economic_rule_regime"
    assert forecast.nature=="forecast" and forecast.target=="future_realized_volatility"
    assert diagnostic.value is None and forecast.value is None


@pytest.mark.asyncio
async def test_loaded_catalog_is_bound_to_the_current_artifact_hash(tmp_path):
    history(tmp_path)
    artifact(tmp_path)
    service=CapabilityService(tmp_path,now=lambda:NOW)
    result=await service.result("BTC","crypto",ModelType.VOLATILITY,Horizon.D7)
    assert result.value is not None
    assert any(row["loaded"] and row["inference_succeeded"] for row in service.catalog())
    path=tmp_path/"models/validated_risk/crypto_BTC_7d.json"
    path.write_text(path.read_text()+" ")
    assert not any(row["loaded"] or row["inference_succeeded"] for row in service.catalog())
    failed=await service.result("BTC","crypto",ModelType.VOLATILITY,Horizon.D7)
    assert failed.value is None


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream_status, expected", [(401, 502), (403, 502), (503, 503)])
async def test_provider_auth_error_does_not_expire_portfolio_session(monkeypatch, upstream_status, expected):
    import httpx
    from types import SimpleNamespace
    from fastapi import HTTPException
    import api.coingecko_proxy_router as proxy

    request = httpx.Request("GET", "https://provider.invalid/global")
    response = httpx.Response(upstream_status, request=request)

    class ProviderClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return response

    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: ProviderClient())
    monkeypatch.setattr(proxy, "_cache", {})
    monkeypatch.setattr(proxy, "coingecko_circuit", SimpleNamespace(
        is_available=lambda: True, record_failure=lambda: None))
    with pytest.raises(HTTPException) as error:
        await proxy._fetch_with_cache_and_fallback(str(request.url), "isolated-provider", 30)
    assert error.value.status_code == expected
    assert str(upstream_status) in error.value.detail
