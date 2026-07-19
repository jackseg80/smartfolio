"""Regression tests for wealth item API endpoints."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.deps import get_required_user
from api.wealth_endpoints import router
from models.wealth import WealthItemOutput


def test_create_wealth_item_returns_created_response():
    """The API must not fail response validation after persisting an item."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_required_user] = lambda: "test_user"
    client = TestClient(app)

    created_item = WealthItemOutput(
        id="item-123",
        name="Emergency Fund",
        category="liquidity",
        type="bank_account",
        value=1000.0,
        currency="USD",
        value_usd=1000.0,
        metadata={},
    )
    payload = {
        "name": "Emergency Fund",
        "category": "liquidity",
        "type": "bank_account",
        "value": 1000.0,
        "currency": "USD",
        "acquisition_date": None,
        "notes": None,
        "metadata": {},
    }

    with patch(
        "services.wealth.wealth_service.create_item",
        return_value=created_item,
    ):
        response = client.post("/api/wealth/items", json=payload)

    assert response.status_code == 201
    assert response.json() == created_item.model_dump()
