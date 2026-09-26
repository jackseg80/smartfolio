"""Stock ML reads require the same user identity as portfolio routes."""

from fastapi.testclient import TestClient

from api.main import app


def test_stock_ml_forecast_requires_user_header():
    response = TestClient(app).get("/api/ml/bourse/forecast")
    assert response.status_code == 422


def test_stock_ml_regime_requires_user_header():
    response = TestClient(app).get("/api/ml/bourse/regime")
    assert response.status_code == 422
