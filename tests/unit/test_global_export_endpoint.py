import json
from types import SimpleNamespace

import pytest

from api.wealth_endpoints import export_global_lists
from services import portfolio_export_service
from services.wealth import wealth_service


@pytest.mark.asyncio
async def test_global_export_concatenates_crypto_stock_cash_and_wealth(monkeypatch):
    async def fake_crypto_balances(**_kwargs):
        return {"items": [{
            "symbol": "BTC", "amount": 1.25, "value_usd": 125.0, "location": "Ledger",
        }]}

    monkeypatch.setattr(
        "services.balance_service.balance_service.resolve_current_balances", fake_crypto_balances
    )
    monkeypatch.setattr(
        portfolio_export_service,
        "build_saxo_export_data",
        lambda **_kwargs: {
            "positions": [
                {
                    "symbol": "AAPL:xnas", "instrument": "Apple Inc.", "asset_class": "Stock",
                    "quantity": 2, "market_value_usd": 200.0, "currency": "USD",
                    "classification": "Technology", "classification_basis": "GICS",
                },
                {
                    "symbol": "CASH:EUR", "instrument": "Saxo cash balance", "asset_class": "Cash",
                    "quantity": 50, "market_value_usd": 55.0, "currency": "EUR",
                    "classification": "Cash", "classification_basis": "Cash",
                },
            ]
        },
    )
    monkeypatch.setattr(
        wealth_service,
        "list_items",
        lambda _user: [SimpleNamespace(
            category="liability", name="Mortgage", type="mortgage", value=-25.0,
            currency="USD", value_usd=-25.0, notes="Home loan",
        )],
    )

    response = await export_global_lists(
        user="test-user", source="cointracking", min_usd_threshold=1.0, format="json"
    )
    payload = json.loads(response.body)["data"]

    assert payload["summary"] == {
        "by_source_usd": {"Crypto": 125.0, "Stock Market": 255.0, "Wealth": -25.0},
        "total_value_usd": 355.0,
        "items_count": 4,
    }
    assert [item["asset"] for item in payload["items"]] == ["BTC", "AAPL:xnas", "CASH:EUR", "Mortgage"]
    assert payload["items"][2]["classification"] == "Cash"
