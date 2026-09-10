"""Security regression tests for the staged cookie authentication mode."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest
from types import SimpleNamespace

from api import auth_router
from api.auth_security import AuthenticatedUser
from api.deps import require_any_role, resolve_authenticated_user
from api.middleware_setup import _is_public_path, setup_middlewares


@pytest.fixture
def auth_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AUTH_MODE", "cookie")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-that-is-longer-than-thirty-two-characters")
    monkeypatch.setattr(auth_router, "ensure_login_allowed", lambda *_: None)
    monkeypatch.setattr(auth_router, "record_login_failure", lambda *_: None)
    monkeypatch.setattr(auth_router, "clear_login_failures", lambda *_: None)
    monkeypatch.setattr(auth_router, "is_allowed_user", lambda user_id: user_id in {"jack", "demo"})
    monkeypatch.setattr(
        auth_router,
        "get_user_info",
        lambda user_id: {
            "id": user_id,
            "label": user_id.title(),
            "roles": ["admin", "governance_admin", "ml_admin"] if user_id == "jack" else ["viewer"],
            "status": "active",
            "password_hash": "hash",
        },
    )
    monkeypatch.setattr(auth_router, "verify_password", lambda *_: True)
    monkeypatch.setattr(
        auth_router,
        "create_refresh_session",
        lambda user_id, roles: ("opaque-refresh-token", "session-id"),
    )
    app = FastAPI()
    app.include_router(auth_router.router)
    return TestClient(app, base_url="https://testserver")


def test_cookie_login_does_not_return_jwt_to_javascript(auth_client: TestClient):
    response = auth_client.post(
        "/auth/login",
        data={"username": "jack", "password": "a-valid-password"},
    )
    assert response.status_code == 200
    assert "token" not in response.json()["data"]
    cookies = "\n".join(response.headers.get_list("set-cookie"))
    assert "smartfolio_access=" in cookies
    assert "smartfolio_refresh=" in cookies
    assert "smartfolio_csrf=" in cookies
    assert "HttpOnly" in cookies
    assert "Secure" in cookies
    assert "SameSite=strict" in cookies


def test_x_user_cannot_authenticate_in_cookie_mode(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "cookie")
    with pytest.raises(HTTPException) as exc_info:
        resolve_authenticated_user(x_user="jack")
    assert exc_info.value.status_code == 401


def test_bearer_cannot_authenticate_in_cookie_mode(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "cookie")
    with pytest.raises(HTTPException) as exc_info:
        resolve_authenticated_user(authorization="Bearer legacy-token")
    assert exc_info.value.status_code == 401


def test_session_and_x_user_mismatch_is_forbidden(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "cookie")
    monkeypatch.setattr("api.deps._extract_cookie_user", lambda _: "jack")
    with pytest.raises(HTTPException) as exc_info:
        resolve_authenticated_user(access_cookie="valid", x_user="demo")
    assert exc_info.value.status_code == 403


def test_viewer_is_rejected_from_governance_role(monkeypatch):
    dependency = require_any_role("governance_admin")
    monkeypatch.setattr(
        "api.deps.get_user_info",
        lambda _: {"id": "demo", "roles": ["viewer"], "status": "active"},
    )
    with pytest.raises(HTTPException) as exc_info:
        dependency(user_id="demo")
    assert exc_info.value.status_code == 403


def test_admin_qualifies_for_governance_role(monkeypatch):
    dependency = require_any_role("governance_admin")
    monkeypatch.setattr(
        "api.deps.get_user_info",
        lambda _: {"id": "jack", "roles": ["admin"], "status": "active"},
    )
    result = dependency(user_id="jack")
    assert result == AuthenticatedUser(username="jack", roles=["admin"])


def test_refresh_requires_csrf(auth_client: TestClient):
    auth_client.cookies.set("smartfolio_refresh", "opaque-refresh-token")
    auth_client.cookies.set("smartfolio_csrf", "csrf-token")
    response = auth_client.post("/auth/refresh")
    assert response.status_code == 403


def test_refresh_rotates_cookie(auth_client: TestClient, monkeypatch):
    monkeypatch.setattr(
        auth_router,
        "rotate_refresh_session",
        lambda _: (
            "replacement-refresh-token",
            {"user_id": "jack", "session_id": "session-id"},
        ),
    )
    auth_client.cookies.set("smartfolio_refresh", "opaque-refresh-token")
    auth_client.cookies.set("smartfolio_csrf", "csrf-token")
    response = auth_client.post(
        "/auth/refresh",
        headers={"X-CSRF-Token": "csrf-token"},
    )
    assert response.status_code == 200
    assert "token" not in response.json()["data"]
    assert "replacement-refresh-token" in "\n".join(response.headers.get_list("set-cookie"))


def test_dual_refresh_returns_replacement_bearer(auth_client: TestClient, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "dual")
    monkeypatch.setattr(
        auth_router,
        "rotate_refresh_session",
        lambda _: (
            "replacement-refresh-token",
            {"user_id": "jack", "session_id": "session-id"},
        ),
    )
    auth_client.cookies.set("smartfolio_refresh", "opaque-refresh-token")
    auth_client.cookies.set("smartfolio_csrf", "csrf-token")

    response = auth_client.post(
        "/auth/refresh",
        headers={"X-CSRF-Token": "csrf-token"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["token_type"] == "bearer"
    assert response.json()["data"]["token"]


def test_replayed_refresh_is_rejected(auth_client: TestClient, monkeypatch):
    def reject_replay(_):
        raise HTTPException(status_code=401, detail="Invalid or replayed refresh token")

    monkeypatch.setattr(auth_router, "rotate_refresh_session", reject_replay)
    auth_client.cookies.set("smartfolio_refresh", "replayed-token")
    auth_client.cookies.set("smartfolio_csrf", "csrf-token")
    response = auth_client.post(
        "/auth/refresh",
        headers={"X-CSRF-Token": "csrf-token"},
    )
    assert response.status_code == 401


def test_cookie_mode_requires_authentication_by_default(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "cookie")
    app = FastAPI()

    @app.get("/private")
    async def private_route():
        return {"ok": True}

    @app.get("/healthz")
    async def health_route():
        return {"ok": True}

    setup_middlewares(
        app,
        settings=SimpleNamespace(security=SimpleNamespace(force_https=False)),
        debug=True,
        environment="development",
        cors_origins=["https://testserver"],
    )
    client = TestClient(app, base_url="https://testserver")

    assert client.get("/private").status_code == 401
    assert client.get("/healthz").status_code == 200


def test_login_entrypoint_is_public_but_private_routes_are_not():
    assert _is_public_path("/")
    assert _is_public_path("/static/login.html")
    assert not _is_public_path("/balances/current")


def test_cookie_mutation_requires_csrf(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "cookie")
    monkeypatch.setattr(
        "api.middleware_setup.resolve_authenticated_user",
        lambda **_: "jack",
    )
    app = FastAPI()

    @app.post("/private")
    async def private_mutation():
        return {"ok": True}

    setup_middlewares(
        app,
        settings=SimpleNamespace(security=SimpleNamespace(force_https=False)),
        debug=True,
        environment="development",
        cors_origins=["https://testserver"],
    )
    client = TestClient(app, base_url="https://testserver")
    client.cookies.set("smartfolio_access", "access-token")
    client.cookies.set("smartfolio_csrf", "csrf-token")

    assert client.post("/private").status_code == 403
    assert (
        client.post("/private", headers={"X-CSRF-Token": "csrf-token"}).status_code
        == 200
    )
