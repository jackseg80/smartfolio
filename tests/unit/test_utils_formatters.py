"""
Unit tests for api.utils.formatters module.

Tests standard API response formatters and legacy formatters.
"""
import pytest
from datetime import datetime
from fastapi.responses import JSONResponse

from api.utils.formatters import (
    success_response,
    error_response,
    paginated_response,
    legacy_response,
    StandardResponse,
    to_csv,
    format_currency,
    format_percentage,
    format_action_summary,
    sanitize_for_json,
)
import math


class TestSuccessResponse:
    """Tests for success_response()"""

    def test_success_response_basic(self):
        """Test basic success response with data only"""
        data = {"balance": 1000, "currency": "USD"}
        response = success_response(data)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 200

        # Parse body
        import json
        body = json.loads(response.body.decode())

        assert body["ok"] is True
        assert body["data"] == data
        assert body["meta"] == {}
        assert "timestamp" in body
        # Verify timestamp is ISO 8601 format
        datetime.fromisoformat(body["timestamp"])

    def test_success_response_with_meta(self):
        """Test success response with metadata"""
        data = [1, 2, 3]
        meta = {"count": 3, "source": "test"}
        response = success_response(data, meta=meta)

        import json
        body = json.loads(response.body.decode())

        assert body["ok"] is True
        assert body["data"] == data
        assert body["meta"] == meta

    def test_success_response_custom_status_code(self):
        """Test success response with custom status code"""
        data = {"created": True}
        response = success_response(data, status_code=201)

        assert response.status_code == 201

        import json
        body = json.loads(response.body.decode())
        assert body["ok"] is True

    def test_success_response_empty_data(self):
        """Test success response with empty data"""
        response = success_response([])

        import json
        body = json.loads(response.body.decode())

        assert body["ok"] is True
        assert body["data"] == []


class TestErrorResponse:
    """Tests for error_response()"""

    def test_error_response_basic(self):
        """Test basic error response"""
        response = error_response("Something went wrong")

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        import json
        body = json.loads(response.body.decode())

        assert body["ok"] is False
        assert body["error"] == "Something went wrong"
        assert body["details"] == {}
        assert "timestamp" in body

    def test_error_response_with_code(self):
        """Test error response with custom status code"""
        response = error_response("Not found", code=404)

        assert response.status_code == 404

        import json
        body = json.loads(response.body.decode())
        assert body["ok"] is False
        assert body["error"] == "Not found"

    def test_error_response_with_details(self):
        """Test error response with details"""
        details = {"field": "email", "reason": "invalid format"}
        response = error_response("Validation failed", code=400, details=details)

        assert response.status_code == 400

        import json
        body = json.loads(response.body.decode())

        assert body["ok"] is False
        assert body["error"] == "Validation failed"
        assert body["details"] == details


class TestPaginatedResponse:
    """Tests for paginated_response()"""

    def test_paginated_response_basic(self):
        """Test basic paginated response"""
        items = [1, 2, 3, 4, 5]
        response = paginated_response(items=items, total=100, page=1, page_size=5)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 200

        import json
        body = json.loads(response.body.decode())

        assert body["ok"] is True
        assert body["data"] == items
        assert "pagination" in body["meta"]

        pagination = body["meta"]["pagination"]
        assert pagination["total"] == 100
        assert pagination["page"] == 1
        assert pagination["page_size"] == 5
        assert pagination["total_pages"] == 20
        assert pagination["has_next"] is True
        assert pagination["has_prev"] is False

    def test_paginated_response_last_page(self):
        """Test paginated response on last page"""
        items = [96, 97, 98, 99, 100]
        response = paginated_response(items=items, total=100, page=20, page_size=5)

        import json
        body = json.loads(response.body.decode())

        pagination = body["meta"]["pagination"]
        assert pagination["has_next"] is False
        assert pagination["has_prev"] is True

    def test_paginated_response_middle_page(self):
        """Test paginated response on middle page"""
        items = [11, 12, 13, 14, 15]
        response = paginated_response(items=items, total=100, page=3, page_size=5)

        import json
        body = json.loads(response.body.decode())

        pagination = body["meta"]["pagination"]
        assert pagination["has_next"] is True
        assert pagination["has_prev"] is True

    def test_paginated_response_with_extra_meta(self):
        """Test paginated response with additional metadata"""
        items = [1, 2, 3]
        extra_meta = {"source": "test", "cached": True}
        response = paginated_response(
            items=items,
            total=100,
            page=1,
            page_size=3,
            meta=extra_meta
        )

        import json
        body = json.loads(response.body.decode())

        assert "pagination" in body["meta"]
        assert body["meta"]["source"] == "test"
        assert body["meta"]["cached"] is True

    def test_paginated_response_total_pages_calculation(self):
        """Test ceiling division for total_pages"""
        # 100 items, 7 per page = 15 pages (ceiling of 100/7 = 14.28)
        response = paginated_response(items=[], total=100, page=1, page_size=7)

        import json
        body = json.loads(response.body.decode())

        pagination = body["meta"]["pagination"]
        assert pagination["total_pages"] == 15  # ceil(100/7) = 15


class TestLegacyResponse:
    """Tests for legacy_response()"""

    def test_legacy_response_basic(self):
        """Test basic legacy response"""
        items = [{"symbol": "BTC", "balance": 1.5}]
        response = legacy_response("cointracking", items)

        assert isinstance(response, dict)
        assert response["source_used"] == "cointracking"
        assert response["items"] == items
        assert "warnings" not in response
        assert "error" not in response

    def test_legacy_response_with_warnings(self):
        """Test legacy response with warnings"""
        items = [{"symbol": "ETH"}]
        warnings = ["Data is stale", "Some prices missing"]
        response = legacy_response("saxobank", items, warnings=warnings)

        assert response["source_used"] == "saxobank"
        assert response["items"] == items
        assert response["warnings"] == warnings

    def test_legacy_response_with_error(self):
        """Test legacy response with error"""
        items = []
        response = legacy_response("cointracking_api", items, error="API timeout")

        assert response["source_used"] == "cointracking_api"
        assert response["items"] == []
        assert response["error"] == "API timeout"

    def test_legacy_response_with_warnings_and_error(self):
        """Test legacy response with both warnings and error"""
        items = []
        warnings = ["Fallback data used"]
        response = legacy_response(
            "stub_conservative",
            items,
            warnings=warnings,
            error="No real data available"
        )

        assert response["source_used"] == "stub_conservative"
        assert response["warnings"] == warnings
        assert response["error"] == "No real data available"


class TestStandardResponseModel:
    """Tests for StandardResponse Pydantic model"""

    def test_standard_response_model_success(self):
        """Test StandardResponse model for success case"""
        response = StandardResponse(
            ok=True,
            data={"balance": 1000},
            meta={"currency": "USD"},
            timestamp="2025-10-20T10:30:00"
        )

        assert response.ok is True
        assert response.data == {"balance": 1000}
        assert response.meta == {"currency": "USD"}
        assert response.error is None
        assert response.details is None

    def test_standard_response_model_error(self):
        """Test StandardResponse model for error case"""
        response = StandardResponse(
            ok=False,
            error="Not found",
            details={"user_id": "invalid"},
            timestamp="2025-10-20T10:30:00"
        )

        assert response.ok is False
        assert response.error == "Not found"
        assert response.details == {"user_id": "invalid"}
        assert response.data is None
        assert response.meta is None


# ---------------------------------------------------------------------------
# to_csv — Lines 25-47 (uncovered)
# ---------------------------------------------------------------------------
class TestToCsv:
    def test_basic_csv(self):
        actions = [{"symbol": "BTC", "action": "buy", "amount": 0.5, "value_usd": 25000}]
        result = to_csv(actions)
        assert "symbol" in result  # header
        assert "BTC" in result
        assert "buy" in result

    def test_empty_list(self):
        assert to_csv([]) == ""

    def test_multiple_rows(self):
        actions = [
            {"symbol": "BTC", "action": "buy", "amount": 0.5},
            {"symbol": "ETH", "action": "sell", "amount": 2.0},
        ]
        result = to_csv(actions)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_missing_fields_default_empty(self):
        actions = [{"symbol": "SOL"}]
        result = to_csv(actions)
        assert "SOL" in result

    def test_all_fieldnames_in_header(self):
        actions = [{"symbol": "BTC"}]
        result = to_csv(actions)
        header = result.split("\n")[0]
        for field in ["symbol", "action", "amount", "value_usd", "location",
                      "current_allocation", "target_allocation", "drift", "priority", "notes"]:
            assert field in header


# ---------------------------------------------------------------------------
# format_currency — Lines 49-54 (uncovered)
# ---------------------------------------------------------------------------
class TestFormatCurrency:
    def test_usd_default(self):
        assert format_currency(1000.0) == "$1,000.00"

    def test_usd_explicit(self):
        assert format_currency(1234.56, "USD") == "$1,234.56"

    def test_other_currency(self):
        assert format_currency(500.0, "EUR") == "500.00 EUR"

    def test_zero(self):
        assert format_currency(0.0) == "$0.00"

    def test_large_number(self):
        result = format_currency(1000000.50)
        assert "$1,000,000.50" == result

    def test_negative(self):
        result = format_currency(-100.0)
        assert result == "$-100.00"


# ---------------------------------------------------------------------------
# format_percentage — Line 58 (uncovered)
# ---------------------------------------------------------------------------
class TestFormatPercentage:
    def test_default_decimals(self):
        assert format_percentage(42.567) == "42.57%"

    def test_zero_decimals(self):
        assert format_percentage(42.567, decimals=0) == "43%"

    def test_one_decimal(self):
        assert format_percentage(3.14, decimals=1) == "3.1%"

    def test_zero_value(self):
        assert format_percentage(0.0) == "0.00%"

    def test_negative(self):
        assert format_percentage(-5.5) == "-5.50%"


# ---------------------------------------------------------------------------
# format_action_summary — Lines 60-83 (uncovered)
# ---------------------------------------------------------------------------
class TestFormatActionSummary:
    def test_empty_actions(self):
        result = format_action_summary([])
        assert result["total_actions"] == 0
        assert result["total_value"] == 0
        assert result["summary"] == "No actions required"

    def test_buy_and_sell_actions(self):
        actions = [
            {"action": "buy", "value_usd": 1000},
            {"action": "sell", "value_usd": 500},
            {"action": "buy", "value_usd": 200},
        ]
        result = format_action_summary(actions)
        assert result["total_actions"] == 3
        assert result["buy_actions"] == 2
        assert result["sell_actions"] == 1
        assert result["total_value"] == 1700.0
        assert "$1,700.00" in result["total_value_formatted"]
        assert "3 actions" in result["summary"]

    def test_only_buys(self):
        actions = [{"action": "buy", "value_usd": 100}]
        result = format_action_summary(actions)
        assert result["buy_actions"] == 1
        assert result["sell_actions"] == 0


# ---------------------------------------------------------------------------
# sanitize_for_json — Lines 90-130 (uncovered)
# ---------------------------------------------------------------------------
class TestSanitizeForJson:
    def test_normal_dict(self):
        data = {"a": 1, "b": "hello"}
        assert sanitize_for_json(data) == data

    def test_nan_replaced_with_none(self):
        assert sanitize_for_json(float("nan")) is None

    def test_inf_replaced_with_none(self):
        assert sanitize_for_json(float("inf")) is None

    def test_neg_inf_replaced_with_none(self):
        assert sanitize_for_json(float("-inf")) is None

    def test_inf_with_replacement_value(self):
        assert sanitize_for_json(float("inf"), replace_inf_with=0) == 0
        assert sanitize_for_json(float("-inf"), replace_inf_with=0) == 0

    def test_nested_dict(self):
        data = {"score": float("inf"), "nested": {"value": float("nan")}}
        result = sanitize_for_json(data)
        assert result["score"] is None
        assert result["nested"]["value"] is None

    def test_list_with_special_values(self):
        data = [1.5, float("nan"), float("-inf"), 42]
        result = sanitize_for_json(data, replace_inf_with=0)
        assert result == [1.5, None, 0, 42]

    def test_normal_float_unchanged(self):
        assert sanitize_for_json(3.14) == 3.14

    def test_non_float_passthrough(self):
        assert sanitize_for_json("hello") == "hello"
        assert sanitize_for_json(42) == 42
        assert sanitize_for_json(True) is True
        assert sanitize_for_json(None) is None

    def test_empty_structures(self):
        assert sanitize_for_json({}) == {}
        assert sanitize_for_json([]) == []
