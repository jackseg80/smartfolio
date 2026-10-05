from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.deps import get_required_user, get_current_user_jwt
from api.saxo_endpoints import router
from services import saxo_valuation_service


def app_for(user='alice', authenticated='alice'):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_required_user] = lambda: user
    app.dependency_overrides[get_current_user_jwt] = lambda: authenticated
    return app


def test_valuation_rejects_identity_mismatch_before_loading_data(monkeypatch):
    def forbidden(*_a): raise AssertionError('Data must not be loaded')
    monkeypatch.setattr(saxo_valuation_service, 'get_valuation', forbidden)
    response = TestClient(app_for(authenticated='bob')).get('/api/saxo/valuation')
    assert response.status_code == 403


def test_valuation_validates_mode_and_returns_standard_response(monkeypatch):
    calls = []
    def value(*args):
        calls.append(args)
        return {'currency': 'EUR', 'mode': 'export'}
    monkeypatch.setattr(saxo_valuation_service, 'get_valuation', value)
    client = TestClient(app_for())
    assert client.get('/api/saxo/valuation?mode=unknown').status_code == 422
    response = client.get('/api/saxo/valuation?mode=export&file_key=old.csv')
    assert response.status_code == 200
    assert response.json()['ok']
    assert calls == [('alice', 'old.csv', 'export', 'USD', False)]


def test_missing_csv_returns_404(monkeypatch):
    def missing(*_a): raise FileNotFoundError('Selected Saxo CSV not found')
    monkeypatch.setattr(saxo_valuation_service, 'get_valuation', missing)
    assert TestClient(app_for()).get('/api/saxo/valuation?file_key=missing.csv').status_code == 404


def test_missing_authentication_is_rejected(monkeypatch):
    monkeypatch.setenv('AUTH_MODE', 'cookie')
    monkeypatch.setenv('DEV_SKIP_AUTH', '0')
    app = app_for()
    app.dependency_overrides.pop(get_current_user_jwt)
    assert TestClient(app).get('/api/saxo/valuation').status_code == 401
