import csv
import io

from services import portfolio_export_service as export_service
from services.export_formatter import ExportFormatter


def test_saxo_export_adds_cash_and_uses_correct_classifications(monkeypatch):
    positions = [
        {
            "symbol": "TSLA:xnas", "instrument": "Tesla Inc.", "asset_class": "Stock",
            "quantity": 2, "market_value_usd": 500.0, "currency": "USD",
        },
        {
            "symbol": "AGGS:xswx", "instrument": "Global Aggregate Bond ETF", "asset_class": "ETF",
            "quantity": 3, "market_value_usd": 300.0, "currency": "CHF",
        },
    ]
    monkeypatch.setattr(export_service.saxo_adapter, "_iter_positions", lambda **_kwargs: iter(positions))
    monkeypatch.setattr(
        export_service,
        "read_saxo_cash",
        lambda *_args, **_kwargs: {"amount": 100.0, "currency": "EUR", "value_usd": 110.0, "last_updated": None},
    )

    exported = export_service.build_saxo_export_data("user", "selected.csv")

    assert exported["summary"]["cash_value_usd"] == 110.0
    assert exported["summary"]["total_value_usd"] == 910.0
    assert [(row["symbol"], row["classification"]) for row in exported["positions"]] == [
        ("TSLA:xnas", "Consumer Discretionary"),
        ("AGGS:xswx", "Fixed Income"),
        ("CASH:EUR", "Cash"),
    ]


def test_wealth_csv_escapes_notes_and_keeps_one_item_table():
    data = {
        "summary": {"net_worth": 100.0, "total_assets": 100.0, "total_liabilities": 0.0},
        "items_by_category": {
            "liquidity": [{
                "id": "cash", "name": "Joint, savings", "type": "bank_account", "value": 100.0,
                "currency": "CHF", "value_usd": 110.0, "acquisition_date": None,
                "notes": 'He said "shared"',
            }],
        },
    }

    rows = list(csv.reader(io.StringIO(ExportFormatter("wealth").to_csv(data))))

    assert rows[2] == ["Category", "ID", "Name", "Type", "Value", "Currency", "Value USD", "Acquisition Date", "Notes"]
    assert rows[3][2] == "Joint, savings"
    assert rows[3][8] == 'He said "shared"'


def test_global_csv_is_a_single_traceable_holdings_table():
    data = {
        "items": [{
            "source": "Stock Market", "category": "Cash", "asset": "CASH:USD", "type": "Saxo cash balance",
            "quantity": 100, "original_value": 100, "currency": "USD", "value_usd": 100,
            "classification": "Cash", "notes": "Cash",
        }],
        "summary": {"by_source_usd": {"Crypto": 10, "Stock Market": 100, "Wealth": -5}, "total_value_usd": 105},
    }

    rows = list(csv.reader(io.StringIO(ExportFormatter("global").to_csv(data))))

    assert rows[2][0] == "Source"
    assert rows[3][0:4] == ["Stock Market", "Cash", "CASH:USD", "Saxo cash balance"]
    assert rows[-1] == ["Global Total", "105.00"]
