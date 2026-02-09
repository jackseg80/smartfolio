"""
Unit tests for services.export_formatter module.

Tests the ExportFormatter class which converts portfolio data to JSON, CSV,
and Markdown formats for crypto, saxo, banks, and wealth modules.
"""
import json
import pytest
from datetime import datetime

from services.export_formatter import ExportFormatter


# ============================================================
# Fixtures - Realistic test data for each module
# ============================================================

@pytest.fixture
def crypto_data():
    """Realistic crypto portfolio data."""
    return {
        "items": [
            {
                "symbol": "BTC",
                "group": "Layer 1",
                "amount": 1.50000000,
                "value_usd": 148500.00,
                "location": "Ledger"
            },
            {
                "symbol": "ETH",
                "group": "Layer 1",
                "amount": 25.00000000,
                "value_usd": 87500.00,
                "location": "Binance"
            },
            {
                "symbol": "SOL",
                "group": "Layer 1",
                "amount": 200.00000000,
                "value_usd": 32000.00,
                "location": "Phantom"
            },
        ],
        "groups": [
            {
                "name": "Layer 1",
                "symbols": ["BTC", "ETH", "SOL"],
                "portfolio_total_usd": 268000.00,
                "portfolio_percentage": 85.50
            },
            {
                "name": "DeFi",
                "symbols": ["AAVE", "UNI"],
                "portfolio_total_usd": 45500.00,
                "portfolio_percentage": 14.50
            },
        ],
    }


@pytest.fixture
def saxo_data():
    """Realistic Saxo Bank positions data."""
    return {
        "positions": [
            {
                "symbol": "AAPL",
                "instrument": "Apple Inc.",
                "asset_class": "Stock",
                "quantity": 50,
                "market_value": 12500.00,
                "currency": "USD",
                "sector": "Technology",
                "entry_price": 185.5000
            },
            {
                "symbol": "MSFT",
                "instrument": "Microsoft Corporation",
                "asset_class": "Stock",
                "quantity": 30,
                "market_value": 13200.00,
                "currency": "USD",
                "sector": "Technology",
                "entry_price": 380.0000
            },
        ],
        "sectors": [
            {
                "name": "Technology",
                "value_usd": 25700.00,
                "percentage": 65.50,
                "asset_count": 2
            },
            {
                "name": "Healthcare",
                "value_usd": 13500.00,
                "percentage": 34.50,
                "asset_count": 1
            },
        ],
    }


@pytest.fixture
def banks_data():
    """Realistic bank accounts data."""
    return {
        "accounts": [
            {
                "bank_name": "UBS",
                "account_type": "Savings",
                "balance": 50000.00,
                "currency": "CHF",
                "balance_usd": 56250.00
            },
            {
                "bank_name": "Revolut",
                "account_type": "Current",
                "balance": 3500.00,
                "currency": "EUR",
                "balance_usd": 3815.00
            },
        ],
    }


@pytest.fixture
def wealth_data():
    """Realistic wealth / patrimoine data."""
    return {
        "summary": {
            "net_worth": 650000.00,
            "total_assets": 750000.00,
            "total_liabilities": 100000.00,
            "breakdown": {
                "liquidity": 120000.00,
                "tangible": 500000.00,
                "insurance": 130000.00,
                "liability": 100000.00,
            },
            "counts": {
                "liquidity": 3,
                "tangible": 2,
                "insurance": 1,
                "liability": 1,
            },
        },
        "items_by_category": {
            "liquidity": [
                {
                    "id": "liq-001",
                    "name": "UBS Savings",
                    "type": "bank_account",
                    "value": 50000.00,
                    "currency": "CHF",
                    "value_usd": 56250.00,
                    "acquisition_date": "2020-01-15",
                    "notes": "Main savings account"
                },
            ],
            "tangible": [
                {
                    "id": "tan-001",
                    "name": "Apartment Zurich",
                    "type": "real_estate",
                    "value": 450000.00,
                    "currency": "CHF",
                    "value_usd": 506250.00,
                    "acquisition_date": "2018-06-01",
                    "notes": "Primary residence"
                },
            ],
            "liability": [
                {
                    "id": "lia-001",
                    "name": "Mortgage",
                    "type": "mortgage",
                    "value": 100000.00,
                    "currency": "CHF",
                    "value_usd": 112500.00,
                    "acquisition_date": "2018-06-01",
                    "notes": "Fixed rate 1.5%"
                },
            ],
            "insurance": [
                {
                    "id": "ins-001",
                    "name": "Life Insurance AXA",
                    "type": "life_insurance",
                    "value": 130000.00,
                    "currency": "CHF",
                    "value_usd": 146250.00,
                    "acquisition_date": "2019-03-10",
                    "notes": "Matures 2045"
                },
            ],
        },
    }


# ============================================================
# Constructor tests
# ============================================================

class TestExportFormatterInit:
    """Tests for ExportFormatter constructor."""

    def test_module_type_stored(self):
        """Module type is stored on the instance."""
        formatter = ExportFormatter("crypto")
        assert formatter.module == "crypto"

    def test_module_type_saxo(self):
        """Saxo module type is stored correctly."""
        formatter = ExportFormatter("saxo")
        assert formatter.module == "saxo"

    def test_module_type_banks(self):
        """Banks module type is stored correctly."""
        formatter = ExportFormatter("banks")
        assert formatter.module == "banks"

    def test_module_type_wealth(self):
        """Wealth module type is stored correctly."""
        formatter = ExportFormatter("wealth")
        assert formatter.module == "wealth"

    def test_timestamp_set(self):
        """Timestamp is set on init and ends with Z."""
        formatter = ExportFormatter("crypto")
        assert formatter.timestamp.endswith("Z")
        # Verify it's a valid ISO timestamp (strip trailing Z)
        datetime.fromisoformat(formatter.timestamp[:-1])

    def test_timestamp_is_recent(self):
        """Timestamp should be very close to current UTC time."""
        before = datetime.utcnow()
        formatter = ExportFormatter("crypto")
        after = datetime.utcnow()

        ts = datetime.fromisoformat(formatter.timestamp[:-1])
        assert before <= ts <= after


# ============================================================
# to_json() tests
# ============================================================

class TestToJson:
    """Tests for to_json() method."""

    def test_json_has_required_fields(self, crypto_data):
        """JSON output contains module, exported_at, and data fields."""
        formatter = ExportFormatter("crypto")
        result = json.loads(formatter.to_json(crypto_data))

        assert "module" in result
        assert "exported_at" in result
        assert "data" in result

    def test_json_module_field(self, crypto_data):
        """JSON module field matches the formatter module."""
        formatter = ExportFormatter("saxo")
        result = json.loads(formatter.to_json(crypto_data))
        assert result["module"] == "saxo"

    def test_json_exported_at_is_timestamp(self, crypto_data):
        """JSON exported_at field is the formatter timestamp."""
        formatter = ExportFormatter("crypto")
        result = json.loads(formatter.to_json(crypto_data))
        assert result["exported_at"] == formatter.timestamp

    def test_json_data_preserved(self, crypto_data):
        """JSON data field contains the original data dict."""
        formatter = ExportFormatter("crypto")
        result = json.loads(formatter.to_json(crypto_data))
        assert result["data"] == crypto_data

    def test_json_pretty_print(self, crypto_data):
        """Pretty JSON has indentation (newlines and spaces)."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_json(crypto_data, pretty=True)
        # Pretty output has newlines and 2-space indentation
        assert "\n" in output
        assert '  "module"' in output

    def test_json_compact(self, crypto_data):
        """Compact JSON has no indentation."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_json(crypto_data, pretty=False)
        # Compact JSON is a single line (no indentation newlines between fields)
        # It might still have newlines inside string values, but the top-level
        # structure should not have pretty-print newlines after opening brace
        lines = output.strip().split("\n")
        assert len(lines) == 1

    def test_json_ensure_ascii_false(self):
        """Non-ASCII characters are preserved (not escaped)."""
        formatter = ExportFormatter("crypto")
        data = {"items": [{"symbol": "BTC", "notes": "Achat a Zurich"}]}
        output = formatter.to_json(data)
        assert "Zurich" in output
        # ensure_ascii=False means no \\uXXXX escaping for common chars
        parsed = json.loads(output)
        assert parsed["data"]["items"][0]["notes"] == "Achat a Zurich"

    def test_json_empty_data(self):
        """JSON works with empty data dict."""
        formatter = ExportFormatter("crypto")
        result = json.loads(formatter.to_json({}))
        assert result["data"] == {}


# ============================================================
# to_csv() tests - Crypto
# ============================================================

class TestCryptoCsv:
    """Tests for to_csv() with crypto module."""

    def test_csv_header_comment(self, crypto_data):
        """CSV starts with a comment line containing module name and timestamp."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_csv(crypto_data)
        first_line = output.split("\n")[0]
        assert first_line.startswith("# Crypto Portfolio Export")
        assert formatter.timestamp in first_line

    def test_csv_assets_header_row(self, crypto_data):
        """CSV contains the assets column header row."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_csv(crypto_data)
        assert "Symbol,Group,Amount,Value USD,Location" in output

    def test_csv_asset_data_rows(self, crypto_data):
        """CSV contains properly formatted asset data rows."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_csv(crypto_data)
        # BTC row: 8 decimal places for amount, 2 decimal for value
        assert "BTC,Layer 1,1.50000000,148500.00,Ledger" in output
        assert "ETH,Layer 1,25.00000000,87500.00,Binance" in output

    def test_csv_groups_header_row(self, crypto_data):
        """CSV contains the groups column header row."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_csv(crypto_data)
        assert "Group Name,Symbols Count,Portfolio Value USD,Portfolio Percentage" in output

    def test_csv_group_data_rows(self, crypto_data):
        """CSV contains properly formatted group data rows."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_csv(crypto_data)
        # Layer 1 group has 3 symbols
        assert "Layer 1,3,268000.00,85.50%" in output
        assert "DeFi,2,45500.00,14.50%" in output

    def test_csv_empty_items(self):
        """CSV with empty items list still has headers."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_csv({"items": [], "groups": []})
        assert "Symbol,Group,Amount,Value USD,Location" in output
        assert "Group Name,Symbols Count,Portfolio Value USD,Portfolio Percentage" in output


# ============================================================
# to_csv() tests - Saxo
# ============================================================

class TestSaxoCsv:
    """Tests for to_csv() with saxo module."""

    def test_csv_header_comment(self, saxo_data):
        """CSV starts with Saxo Bank header comment."""
        formatter = ExportFormatter("saxo")
        output = formatter.to_csv(saxo_data)
        first_line = output.split("\n")[0]
        assert "Saxo Bank Portfolio Export" in first_line

    def test_csv_positions_header_row(self, saxo_data):
        """CSV contains positions column header row."""
        formatter = ExportFormatter("saxo")
        output = formatter.to_csv(saxo_data)
        assert "Symbol,Instrument,Asset Class,Quantity,Market Value,Currency,Sector,Entry Price" in output

    def test_csv_position_data_rows(self, saxo_data):
        """CSV contains properly formatted position data rows."""
        formatter = ExportFormatter("saxo")
        output = formatter.to_csv(saxo_data)
        # Quantity has 4 decimal places, market_value 2, entry_price 4
        assert "AAPL,Apple Inc.,Stock,50.0000,12500.00,USD,Technology,185.5000" in output

    def test_csv_sectors_section(self, saxo_data):
        """CSV contains sectors section with proper formatting."""
        formatter = ExportFormatter("saxo")
        output = formatter.to_csv(saxo_data)
        assert "Sector,Value USD,Percentage,Asset Count" in output
        assert "Technology,25700.00,65.50%,2" in output

    def test_csv_empty_positions(self):
        """CSV with empty positions still has header rows."""
        formatter = ExportFormatter("saxo")
        output = formatter.to_csv({"positions": [], "sectors": []})
        assert "Symbol,Instrument,Asset Class" in output
        assert "Sector,Value USD,Percentage,Asset Count" in output


# ============================================================
# to_csv() tests - Banks
# ============================================================

class TestBanksCsv:
    """Tests for to_csv() with banks module."""

    def test_csv_header_comment(self, banks_data):
        """CSV starts with Bank Accounts header comment."""
        formatter = ExportFormatter("banks")
        output = formatter.to_csv(banks_data)
        first_line = output.split("\n")[0]
        assert "Bank Accounts Export" in first_line

    def test_csv_accounts_header_row(self, banks_data):
        """CSV contains accounts column header row."""
        formatter = ExportFormatter("banks")
        output = formatter.to_csv(banks_data)
        assert "Bank Name,Account Type,Balance,Currency,Balance USD" in output

    def test_csv_account_data_rows(self, banks_data):
        """CSV contains properly formatted account data rows."""
        formatter = ExportFormatter("banks")
        output = formatter.to_csv(banks_data)
        assert "UBS,Savings,50000.00,CHF,56250.00" in output
        assert "Revolut,Current,3500.00,EUR,3815.00" in output

    def test_csv_total_summary(self, banks_data):
        """CSV contains total balance summary row."""
        formatter = ExportFormatter("banks")
        output = formatter.to_csv(banks_data)
        # Total = 56250 + 3815 = 60065
        assert "Total Balance USD,60065.00" in output

    def test_csv_empty_accounts(self):
        """CSV with empty accounts has header and zero total."""
        formatter = ExportFormatter("banks")
        output = formatter.to_csv({"accounts": []})
        assert "Bank Name,Account Type,Balance,Currency,Balance USD" in output
        assert "Total Balance USD,0.00" in output


# ============================================================
# to_csv() tests - Wealth
# ============================================================

class TestWealthCsv:
    """Tests for to_csv() with wealth module."""

    def test_csv_header_comment(self, wealth_data):
        """CSV starts with Patrimoine header comment."""
        formatter = ExportFormatter("wealth")
        output = formatter.to_csv(wealth_data)
        first_line = output.split("\n")[0]
        assert "Patrimoine Export" in first_line

    def test_csv_summary_section(self, wealth_data):
        """CSV contains summary with net worth, assets, liabilities."""
        formatter = ExportFormatter("wealth")
        output = formatter.to_csv(wealth_data)
        assert "Net Worth USD,650000.00" in output
        assert "Total Assets USD,750000.00" in output
        assert "Total Liabilities USD,100000.00" in output

    def test_csv_breakdown_section(self, wealth_data):
        """CSV contains breakdown by category with counts."""
        formatter = ExportFormatter("wealth")
        output = formatter.to_csv(wealth_data)
        assert "Category,Total USD,Count" in output
        assert "Liquidity,120000.00,3" in output
        assert "Tangible Assets,500000.00,2" in output
        assert "Insurance,130000.00,1" in output
        assert "Liabilities,100000.00,1" in output

    def test_csv_items_per_category(self, wealth_data):
        """CSV contains item rows for each category present."""
        formatter = ExportFormatter("wealth")
        output = formatter.to_csv(wealth_data)
        # Liquidity items
        assert "liq-001,UBS Savings,bank_account,50000.00,CHF,56250.00,2020-01-15" in output
        # Tangible items
        assert "tan-001,Apartment Zurich,real_estate,450000.00,CHF,506250.00,2018-06-01" in output
        # Liability items
        assert "lia-001,Mortgage,mortgage,100000.00,CHF,112500.00,2018-06-01" in output
        # Insurance items
        assert "ins-001,Life Insurance AXA,life_insurance,130000.00,CHF,146250.00,2019-03-10" in output

    def test_csv_notes_quoted(self, wealth_data):
        """CSV wraps notes field in double quotes."""
        formatter = ExportFormatter("wealth")
        output = formatter.to_csv(wealth_data)
        assert '"Main savings account"' in output
        assert '"Primary residence"' in output

    def test_csv_empty_categories(self):
        """CSV with empty categories still has summary section."""
        formatter = ExportFormatter("wealth")
        data = {
            "summary": {
                "net_worth": 0,
                "total_assets": 0,
                "total_liabilities": 0,
                "breakdown": {},
                "counts": {},
            },
            "items_by_category": {},
        }
        output = formatter.to_csv(data)
        assert "Net Worth USD,0.00" in output
        assert "Category,Total USD,Count" in output
        # Default 0 from .get() when keys missing in breakdown
        assert "Liquidity,0.00,0" in output


# ============================================================
# to_markdown() tests - Crypto
# ============================================================

class TestCryptoMarkdown:
    """Tests for to_markdown() with crypto module."""

    def test_md_title(self, crypto_data):
        """Markdown starts with crypto portfolio title."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_markdown(crypto_data)
        assert "# " in output.split("\n")[0]
        assert "Crypto Portfolio Export" in output.split("\n")[0]

    def test_md_exported_timestamp(self, crypto_data):
        """Markdown contains exported timestamp."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_markdown(crypto_data)
        assert f"**Exported:** {formatter.timestamp}" in output

    def test_md_total_value_calculated(self, crypto_data):
        """Markdown calculates total portfolio value from items."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_markdown(crypto_data)
        # Total = 148500 + 87500 + 32000 = 268000
        assert "$268,000.00" in output

    def test_md_assets_count(self, crypto_data):
        """Markdown shows correct assets count."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_markdown(crypto_data)
        assert "**Assets Count:** 3" in output

    def test_md_assets_table(self, crypto_data):
        """Markdown contains assets table with pipe-delimited rows."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_markdown(crypto_data)
        assert "| Symbol | Group | Amount | Value USD | Location |" in output
        assert "| BTC | Layer 1 | 1.50000000 | $148,500.00 | Ledger |" in output

    def test_md_groups_table(self, crypto_data):
        """Markdown contains groups table with allocation percentages."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_markdown(crypto_data)
        assert "| Group | Symbols | Value USD | Allocation % |" in output
        assert "**Layer 1**" in output
        assert "85.50%" in output

    def test_md_group_symbols_truncated(self):
        """Markdown truncates group symbols list to 50 chars and appends '...'."""
        formatter = ExportFormatter("crypto")
        data = {
            "items": [],
            "groups": [
                {
                    "name": "DeFi",
                    "symbols": ["AAVE", "UNI", "COMP", "MKR", "SNX", "YFI", "CRV", "SUSHI", "BAL", "DYDX"],
                    "portfolio_total_usd": 10000.00,
                    "portfolio_percentage": 5.00
                }
            ],
        }
        output = formatter.to_markdown(data)
        # The symbols string gets [:50] then "..." appended
        assert "..." in output

    def test_md_footer(self, crypto_data):
        """Markdown ends with the generator footer."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_markdown(crypto_data)
        assert "*Generated by Crypto Rebalancer - Export System*" in output


# ============================================================
# to_markdown() tests - Saxo
# ============================================================

class TestSaxoMarkdown:
    """Tests for to_markdown() with saxo module."""

    def test_md_title(self, saxo_data):
        """Markdown starts with Saxo Bank title."""
        formatter = ExportFormatter("saxo")
        output = formatter.to_markdown(saxo_data)
        assert "Saxo Bank Portfolio Export" in output

    def test_md_total_value(self, saxo_data):
        """Markdown calculates total from market values."""
        formatter = ExportFormatter("saxo")
        output = formatter.to_markdown(saxo_data)
        # 12500 + 13200 = 25700
        assert "$25,700.00" in output

    def test_md_positions_table(self, saxo_data):
        """Markdown contains positions table."""
        formatter = ExportFormatter("saxo")
        output = formatter.to_markdown(saxo_data)
        assert "| Symbol | Instrument | Asset Class | Quantity | Market Value | Currency | Sector |" in output
        assert "| AAPL |" in output

    def test_md_instrument_name_truncated(self):
        """Markdown truncates instrument names to 30 chars."""
        formatter = ExportFormatter("saxo")
        data = {
            "positions": [
                {
                    "symbol": "TEST",
                    "instrument": "A" * 50,  # 50-char name
                    "asset_class": "Stock",
                    "quantity": 10,
                    "market_value": 1000.00,
                    "currency": "USD",
                    "sector": "Technology",
                    "entry_price": 100.0000
                }
            ],
            "sectors": [],
        }
        output = formatter.to_markdown(data)
        # Instrument should be truncated to 30 chars in the table
        assert "A" * 30 in output
        assert "A" * 50 not in output

    def test_md_sectors_table(self, saxo_data):
        """Markdown contains sectors table with GICS heading."""
        formatter = ExportFormatter("saxo")
        output = formatter.to_markdown(saxo_data)
        assert "Sectors (GICS Classification)" in output
        assert "| **Technology** | $25,700.00 | 65.50% | 2 |" in output

    def test_md_footer(self, saxo_data):
        """Markdown has the generator footer."""
        formatter = ExportFormatter("saxo")
        output = formatter.to_markdown(saxo_data)
        assert "*Generated by Crypto Rebalancer - Export System*" in output


# ============================================================
# to_markdown() tests - Banks
# ============================================================

class TestBanksMarkdown:
    """Tests for to_markdown() with banks module."""

    def test_md_title(self, banks_data):
        """Markdown starts with Bank Accounts title."""
        formatter = ExportFormatter("banks")
        output = formatter.to_markdown(banks_data)
        assert "Bank Accounts Export" in output

    def test_md_total_balance(self, banks_data):
        """Markdown shows total balance calculated from accounts."""
        formatter = ExportFormatter("banks")
        output = formatter.to_markdown(banks_data)
        # 56250 + 3815 = 60065
        assert "$60,065.00" in output

    def test_md_accounts_count(self, banks_data):
        """Markdown shows correct accounts count."""
        formatter = ExportFormatter("banks")
        output = formatter.to_markdown(banks_data)
        assert "**Accounts Count:** 2" in output

    def test_md_accounts_table(self, banks_data):
        """Markdown contains accounts table with formatted values."""
        formatter = ExportFormatter("banks")
        output = formatter.to_markdown(banks_data)
        assert "| Bank Name | Account Type | Balance | Currency | Balance USD |" in output
        assert "| UBS | Savings | 50,000.00 | CHF | $56,250.00 |" in output
        assert "| Revolut | Current | 3,500.00 | EUR | $3,815.00 |" in output


# ============================================================
# to_markdown() tests - Wealth
# ============================================================

class TestWealthMarkdown:
    """Tests for to_markdown() with wealth module."""

    def test_md_title(self, wealth_data):
        """Markdown starts with Patrimoine title."""
        formatter = ExportFormatter("wealth")
        output = formatter.to_markdown(wealth_data)
        assert "Patrimoine Export" in output

    def test_md_summary_section(self, wealth_data):
        """Markdown contains net worth, assets, and liabilities."""
        formatter = ExportFormatter("wealth")
        output = formatter.to_markdown(wealth_data)
        assert "**Net Worth:** $650,000.00" in output
        assert "**Total Assets:** $750,000.00" in output
        assert "**Total Liabilities:** $100,000.00" in output

    def test_md_breakdown_table(self, wealth_data):
        """Markdown contains breakdown by category table."""
        formatter = ExportFormatter("wealth")
        output = formatter.to_markdown(wealth_data)
        assert "| Category | Total USD | Count |" in output
        assert "$120,000.00" in output  # Liquidity
        assert "$500,000.00" in output  # Tangible

    def test_md_items_by_category_tables(self, wealth_data):
        """Markdown contains item tables for each category."""
        formatter = ExportFormatter("wealth")
        output = formatter.to_markdown(wealth_data)
        # Check that category section headings appear
        assert "Liquidit" in output  # Liquidites
        assert "Biens Tangibles" in output
        assert "Assurances" in output
        assert "Passifs" in output
        # Check item data
        assert "UBS Savings" in output
        assert "Apartment Zurich" in output

    def test_md_notes_truncated_to_50_chars(self):
        """Markdown truncates notes to 50 characters."""
        formatter = ExportFormatter("wealth")
        long_note = "A" * 80
        data = {
            "summary": {"net_worth": 0, "total_assets": 0, "total_liabilities": 0,
                        "breakdown": {}, "counts": {}},
            "items_by_category": {
                "liquidity": [
                    {"id": "x", "name": "Test", "type": "bank_account",
                     "value": 100, "currency": "USD", "value_usd": 100,
                     "acquisition_date": "", "notes": long_note}
                ],
            },
        }
        output = formatter.to_markdown(data)
        # Notes are [:50], so 50 A's should appear but not 80
        assert "A" * 50 in output
        assert "A" * 80 not in output

    def test_md_empty_category_skipped(self):
        """Markdown skips categories with no items."""
        formatter = ExportFormatter("wealth")
        data = {
            "summary": {"net_worth": 0, "total_assets": 0, "total_liabilities": 0,
                        "breakdown": {}, "counts": {}},
            "items_by_category": {
                "liquidity": [],
                "tangible": [],
                "liability": [],
                "insurance": [],
            },
        }
        output = formatter.to_markdown(data)
        # Category item tables should not appear (only the breakdown table)
        assert "| Name | Type | Value |" not in output


# ============================================================
# Invalid module type
# ============================================================

class TestInvalidModule:
    """Tests for invalid module type handling."""

    def test_csv_invalid_module_raises_value_error(self):
        """to_csv raises ValueError for unknown module type."""
        formatter = ExportFormatter.__new__(ExportFormatter)
        formatter.module = "invalid_module"
        formatter.timestamp = "2026-01-01T00:00:00Z"
        with pytest.raises(ValueError, match="Unknown module"):
            formatter.to_csv({})

    def test_markdown_invalid_module_raises_value_error(self):
        """to_markdown raises ValueError for unknown module type."""
        formatter = ExportFormatter.__new__(ExportFormatter)
        formatter.module = "stocks"
        formatter.timestamp = "2026-01-01T00:00:00Z"
        with pytest.raises(ValueError, match="Unknown module"):
            formatter.to_markdown({})


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_values_in_crypto(self):
        """Zero-value items are formatted correctly."""
        formatter = ExportFormatter("crypto")
        data = {
            "items": [
                {"symbol": "DUST", "group": "Others", "amount": 0,
                 "value_usd": 0, "location": "Exchange"}
            ],
            "groups": [],
        }
        output = formatter.to_csv(data)
        assert "DUST,Others,0.00000000,0.00,Exchange" in output

    def test_missing_fields_use_defaults(self):
        """Items with missing fields use default values."""
        formatter = ExportFormatter("crypto")
        data = {"items": [{}], "groups": [{}]}
        output = formatter.to_csv(data)
        # Default symbol='', group='Others', amount=0, value_usd=0, location=''
        assert ",Others,0.00000000,0.00," in output
        # Default group: name='', symbols=[], portfolio_total_usd=0
        assert ",0,0.00,0.00%" in output

    def test_saxo_missing_fields_use_defaults(self):
        """Saxo positions with missing fields use defaults."""
        formatter = ExportFormatter("saxo")
        data = {"positions": [{}], "sectors": [{}]}
        output = formatter.to_csv(data)
        # Default currency='USD', sector='Unknown'
        assert "USD" in output
        assert "Unknown" in output

    def test_banks_missing_fields_use_defaults(self):
        """Bank accounts with missing fields use defaults."""
        formatter = ExportFormatter("banks")
        data = {"accounts": [{}]}
        output = formatter.to_csv(data)
        # Default currency='USD', balance=0
        assert ",,0.00,USD,0.00" in output

    def test_special_characters_in_data(self):
        """Special characters (commas, quotes) in data fields."""
        formatter = ExportFormatter("banks")
        data = {
            "accounts": [
                {
                    "bank_name": "Credit Suisse",
                    "account_type": "Pilier 3a",
                    "balance": 25000.00,
                    "currency": "CHF",
                    "balance_usd": 28125.00
                }
            ]
        }
        output = formatter.to_csv(data)
        # Note: the current implementation does not escape commas in field values
        # This test documents current behavior
        assert "Credit Suisse" in output
        assert "Pilier 3a" in output

    def test_unicode_characters_in_json(self):
        """Unicode characters preserved in JSON output."""
        formatter = ExportFormatter("banks")
        data = {"accounts": [{"bank_name": "Banque Cantonale", "note": "Compte epargne"}]}
        output = formatter.to_json(data)
        parsed = json.loads(output)
        assert parsed["data"]["accounts"][0]["bank_name"] == "Banque Cantonale"

    def test_large_values_formatted(self):
        """Large monetary values are formatted correctly."""
        formatter = ExportFormatter("banks")
        data = {
            "accounts": [
                {
                    "bank_name": "BigBank",
                    "account_type": "Investment",
                    "balance": 1234567.89,
                    "currency": "USD",
                    "balance_usd": 1234567.89
                }
            ]
        }
        # CSV: no comma grouping
        csv_output = formatter.to_csv(data)
        assert "1234567.89" in csv_output

        # Markdown: comma-grouped
        md_output = formatter.to_markdown(data)
        assert "1,234,567.89" in md_output

    def test_data_without_expected_keys(self):
        """Formatter handles data dict missing expected top-level keys."""
        formatter = ExportFormatter("crypto")
        output = formatter.to_csv({})
        # Should still produce headers even with no 'items' or 'groups' keys
        assert "Symbol,Group,Amount,Value USD,Location" in output

    def test_wealth_csv_skips_empty_categories(self):
        """Wealth CSV skips category item sections when list is empty."""
        formatter = ExportFormatter("wealth")
        data = {
            "summary": {"net_worth": 1000, "total_assets": 1000, "total_liabilities": 0,
                        "breakdown": {"liquidity": 1000}, "counts": {"liquidity": 1}},
            "items_by_category": {
                "liquidity": [
                    {"id": "x", "name": "Cash", "type": "cash", "value": 1000,
                     "currency": "USD", "value_usd": 1000, "acquisition_date": "", "notes": ""}
                ],
                "tangible": [],  # Empty - should not produce a section header
            },
        }
        output = formatter.to_csv(data)
        # Liquidity section should exist
        assert "# Liquidit" in output
        # Tangible section should NOT exist (empty list)
        assert "# Biens Tangibles" not in output

    def test_multiple_formatters_independent(self):
        """Different formatter instances have independent timestamps and modules."""
        f1 = ExportFormatter("crypto")
        f2 = ExportFormatter("saxo")
        assert f1.module != f2.module
        assert f1.module == "crypto"
        assert f2.module == "saxo"
        # Timestamps should both end with Z
        assert f1.timestamp.endswith("Z")
        assert f2.timestamp.endswith("Z")
