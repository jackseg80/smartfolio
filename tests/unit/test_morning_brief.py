"""
Tests for Morning Brief Service + API endpoints.

Tests cover:
- MorningBriefService.generate() with mocked dependencies
- Partial failure tolerance
- P&L aggregation
- Decision Index extraction
- Alert filtering (24h)
- Top movers calculation
- Telegram formatting
- Email HTML formatting
- API endpoints
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from services.morning_brief_service import MorningBriefService, morning_brief_service


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def service():
    return MorningBriefService()


def _make_mock_balance_result(items=None, total=50000):
    """Create a mock balance result."""
    if items is None:
        items = [
            {"symbol": "BTC", "amount": 0.5, "value_usd": 30000},
            {"symbol": "ETH", "amount": 10, "value_usd": 15000},
            {"symbol": "SOL", "amount": 50, "value_usd": 5000},
        ]
    return {"source_used": "cointracking", "items": items}


def _make_mock_performance(window, change=500, pct=1.5, available=True):
    """Create a mock performance metrics result."""
    return {
        "performance_available": available,
        "absolute_change_usd": change,
        "percentage_change": pct,
        "performance_status": "gain" if change >= 0 else "loss",
        "current_value_usd": 50000,
        "historical_value_usd": 50000 - change,
    }


def _make_mock_signals():
    """Create mock MLSignals object."""
    signals = MagicMock()
    signals.blended_score = 72.5
    signals.decision_score = 0.725
    signals.confidence = 0.85
    signals.contradiction_index = 0.15
    signals.sources_used = ["volatility", "regime", "sentiment"]
    signals.regime = {"bull": 0.6, "bear": 0.2, "neutral": 0.2}
    signals.volatility = {"BTC": 0.35, "ETH": 0.45}
    signals.sentiment = {"fear_greed": 65}
    signals.correlation = {"avg_correlation": 0.5}
    signals.as_of = datetime.now()
    return signals


def _make_mock_state(signals=None):
    """Create mock GovernanceEngine state."""
    state = MagicMock()
    state.signals = signals or _make_mock_signals()
    state.governance_mode = "ai_assisted"
    return state


def _make_mock_alert(severity="S2", title="Test Alert", hours_ago=2):
    """Create a mock Alert object."""
    alert = MagicMock()
    alert.id = f"alert-{title.lower().replace(' ', '-')}"
    alert.severity = MagicMock()
    alert.severity.value = severity
    alert.alert_type = MagicMock()
    alert.alert_type.value = "VOLATILITY_SPIKE"
    alert.title = title
    alert.message = f"Test message for {title}"
    alert.created_at = datetime.now() - timedelta(hours=hours_ago)
    return alert


# ============================================================================
# Test MorningBriefService.generate()
# ============================================================================

class TestMorningBriefGenerate:
    """Tests for the main generate() method."""

    @pytest.mark.asyncio
    async def test_generate_returns_all_sections(self, service):
        """Should return all expected sections."""
        with patch.object(service, '_fetch_pnl', new_callable=AsyncMock, return_value={"total_value_usd": 50000}), \
             patch.object(service, '_fetch_decision_index', new_callable=AsyncMock, return_value={"blended_score": 72}), \
             patch.object(service, '_fetch_alerts_24h', new_callable=AsyncMock, return_value={"total": 0}), \
             patch.object(service, '_fetch_top_movers', new_callable=AsyncMock, return_value={"movers": []}), \
             patch.object(service, '_fetch_signals', new_callable=AsyncMock, return_value={"regime": {}}):

            brief = await service.generate("jack")

            assert "pnl" in brief
            assert "decision_index" in brief
            assert "alerts" in brief
            assert "top_movers" in brief
            assert "signals" in brief
            assert "generated_at" in brief
            assert "user_id" in brief
            assert brief["user_id"] == "jack"
            assert "warnings" in brief
            assert "duration_ms" in brief

    @pytest.mark.asyncio
    async def test_generate_partial_failure(self, service):
        """Should handle partial failures gracefully."""
        with patch.object(service, '_fetch_pnl', new_callable=AsyncMock, return_value={"total_value_usd": 50000}), \
             patch.object(service, '_fetch_decision_index', new_callable=AsyncMock, side_effect=Exception("DI unavailable")), \
             patch.object(service, '_fetch_alerts_24h', new_callable=AsyncMock, return_value={"total": 0}), \
             patch.object(service, '_fetch_top_movers', new_callable=AsyncMock, side_effect=Exception("No data")), \
             patch.object(service, '_fetch_signals', new_callable=AsyncMock, return_value={"regime": {}}):

            brief = await service.generate("jack")

            assert brief["pnl"] is not None
            assert brief["decision_index"] is None
            assert brief["alerts"] is not None
            assert brief["top_movers"] is None
            assert brief["signals"] is not None
            assert len(brief["warnings"]) == 2
            assert "decision_index" in brief["warnings"][0]
            assert "top_movers" in brief["warnings"][1]

    @pytest.mark.asyncio
    async def test_generate_all_failures(self, service):
        """Should handle all sections failing."""
        for method in ['_fetch_pnl', '_fetch_decision_index', '_fetch_alerts_24h', '_fetch_top_movers', '_fetch_signals']:
            setattr(service, method, AsyncMock(side_effect=Exception(f"{method} down")))

        brief = await service.generate("jack")

        assert len(brief["warnings"]) == 5
        for section in ["pnl", "decision_index", "alerts", "top_movers", "signals"]:
            assert brief[section] is None

    @pytest.mark.asyncio
    async def test_generate_includes_source(self, service):
        """Should pass source parameter through."""
        with patch.object(service, '_fetch_pnl', new_callable=AsyncMock, return_value={}) as mock_pnl, \
             patch.object(service, '_fetch_decision_index', new_callable=AsyncMock, return_value={}), \
             patch.object(service, '_fetch_alerts_24h', new_callable=AsyncMock, return_value={}), \
             patch.object(service, '_fetch_top_movers', new_callable=AsyncMock, return_value={}) as mock_movers, \
             patch.object(service, '_fetch_signals', new_callable=AsyncMock, return_value={}):

            brief = await service.generate("jack", source="saxobank")
            assert brief["source"] == "saxobank"
            mock_pnl.assert_called_once_with("jack", "saxobank")
            mock_movers.assert_called_once_with("jack", "saxobank")


# ============================================================================
# Test _fetch_pnl
# ============================================================================

class TestFetchPnl:
    """Tests for P&L fetching."""

    @pytest.mark.asyncio
    async def test_pnl_all_windows(self, service):
        """Should fetch P&L for 24h, 7d, 30d."""
        mock_balance = _make_mock_balance_result()

        with patch("services.balance_service.balance_service") as mock_bs, \
             patch("services.portfolio.portfolio_analytics") as mock_pa:

            mock_bs.resolve_current_balances = AsyncMock(return_value=mock_balance)
            mock_pa.calculate_performance_metrics = MagicMock(
                side_effect=lambda **kw: _make_mock_performance(kw.get("window", "24h"))
            )

            result = await service._fetch_pnl("jack", "cointracking")

            assert result["total_value_usd"] == 50000
            assert result["24h"]["available"] is True
            assert result["7d"]["available"] is True
            assert result["30d"]["available"] is True
            assert result["24h"]["absolute_change"] == 500
            assert result["24h"]["percentage_change"] == 1.5

    @pytest.mark.asyncio
    async def test_pnl_partial_window_failure(self, service):
        """Should handle individual window failures."""
        mock_balance = _make_mock_balance_result()

        def mock_perf(**kwargs):
            if kwargs.get("window") == "30d":
                raise Exception("No 30d data")
            return _make_mock_performance(kwargs.get("window", "24h"))

        with patch("services.balance_service.balance_service") as mock_bs, \
             patch("services.portfolio.portfolio_analytics") as mock_pa:

            mock_bs.resolve_current_balances = AsyncMock(return_value=mock_balance)
            mock_pa.calculate_performance_metrics = MagicMock(side_effect=mock_perf)

            result = await service._fetch_pnl("jack", "cointracking")

            assert result["24h"]["available"] is True
            assert result["7d"]["available"] is True
            assert result["30d"]["available"] is False
            assert "error" in result["30d"]

    @pytest.mark.asyncio
    async def test_pnl_empty_portfolio(self, service):
        """Should handle empty portfolio."""
        with patch("services.balance_service.balance_service") as mock_bs, \
             patch("services.portfolio.portfolio_analytics") as mock_pa:

            mock_bs.resolve_current_balances = AsyncMock(return_value={"items": []})
            mock_pa.calculate_performance_metrics = MagicMock(
                return_value={"performance_available": False}
            )

            result = await service._fetch_pnl("jack", "cointracking")
            assert result["total_value_usd"] == 0

    @pytest.mark.asyncio
    async def test_pnl_negative_change(self, service):
        """Should handle negative P&L correctly."""
        mock_balance = _make_mock_balance_result()

        with patch("services.balance_service.balance_service") as mock_bs, \
             patch("services.portfolio.portfolio_analytics") as mock_pa:

            mock_bs.resolve_current_balances = AsyncMock(return_value=mock_balance)
            mock_pa.calculate_performance_metrics = MagicMock(
                return_value=_make_mock_performance("24h", change=-2000, pct=-3.8)
            )

            result = await service._fetch_pnl("jack", "cointracking")
            assert result["24h"]["absolute_change"] == -2000
            assert result["24h"]["percentage_change"] == -3.8
            assert result["24h"]["status"] == "loss"


# ============================================================================
# Test _fetch_decision_index
# ============================================================================

class TestFetchDecisionIndex:
    """Tests for Decision Index fetching."""

    @pytest.mark.asyncio
    async def test_decision_index_normal(self, service):
        """Should extract DI fields correctly."""
        mock_state = _make_mock_state()

        with patch("services.execution.governance.governance_engine") as mock_ge:
            mock_ge.get_current_state = AsyncMock(return_value=mock_state)

            result = await service._fetch_decision_index()

            assert result["blended_score"] == 72.5
            assert result["decision_score"] == 72.5
            assert result["confidence"] == 85.0
            assert result["contradiction_index"] == 15.0
            assert result["governance_mode"] == "ai_assisted"

    @pytest.mark.asyncio
    async def test_decision_index_no_blended(self, service):
        """Should handle missing blended_score."""
        signals = _make_mock_signals()
        signals.blended_score = None
        mock_state = _make_mock_state(signals)

        with patch("services.execution.governance.governance_engine") as mock_ge:
            mock_ge.get_current_state = AsyncMock(return_value=mock_state)

            result = await service._fetch_decision_index()
            assert result["blended_score"] is None
            assert result["decision_score"] == 72.5


# ============================================================================
# Test _fetch_alerts_24h
# ============================================================================

class TestFetchAlerts:
    """Tests for 24h alert fetching."""

    @pytest.mark.asyncio
    async def test_alerts_filters_24h(self, service):
        """Should only return alerts from last 24h."""
        recent = _make_mock_alert("S2", "Recent Alert", hours_ago=2)
        old = _make_mock_alert("S1", "Old Alert", hours_ago=48)

        with patch("services.alerts.alert_storage.AlertStorage") as MockStorage:
            instance = MockStorage.return_value
            instance.get_active_alerts.return_value = [recent, old]

            result = await service._fetch_alerts_24h()

            assert result["total"] == 1
            assert len(result["alerts"]) == 1
            assert result["alerts"][0]["title"] == "Recent Alert"

    @pytest.mark.asyncio
    async def test_alerts_groups_by_severity(self, service):
        """Should group alerts by severity."""
        alerts = [
            _make_mock_alert("S3", "Critical 1", hours_ago=1),
            _make_mock_alert("S3", "Critical 2", hours_ago=2),
            _make_mock_alert("S2", "Warning 1", hours_ago=3),
            _make_mock_alert("S1", "Info 1", hours_ago=4),
        ]

        with patch("services.alerts.alert_storage.AlertStorage") as MockStorage:
            instance = MockStorage.return_value
            instance.get_active_alerts.return_value = alerts

            result = await service._fetch_alerts_24h()

            assert result["total"] == 4
            assert result["by_severity"]["S3"] == 2
            assert result["by_severity"]["S2"] == 1
            assert result["by_severity"]["S1"] == 1

    @pytest.mark.asyncio
    async def test_alerts_empty(self, service):
        """Should handle no alerts."""
        with patch("services.alerts.alert_storage.AlertStorage") as MockStorage:
            instance = MockStorage.return_value
            instance.get_active_alerts.return_value = []

            result = await service._fetch_alerts_24h()

            assert result["total"] == 0
            assert result["alerts"] == []

    @pytest.mark.asyncio
    async def test_alerts_limits_to_10(self, service):
        """Should limit alerts list to 10."""
        alerts = [_make_mock_alert("S2", f"Alert {i}", hours_ago=i % 20 + 1) for i in range(15)]

        with patch("services.alerts.alert_storage.AlertStorage") as MockStorage:
            instance = MockStorage.return_value
            instance.get_active_alerts.return_value = alerts

            result = await service._fetch_alerts_24h()
            assert len(result["alerts"]) <= 10

    @pytest.mark.asyncio
    async def test_alerts_sorted_by_severity(self, service):
        """Should sort S3 before S2 before S1."""
        alerts = [
            _make_mock_alert("S1", "Info", hours_ago=1),
            _make_mock_alert("S3", "Critical", hours_ago=2),
            _make_mock_alert("S2", "Warning", hours_ago=3),
        ]

        with patch("services.alerts.alert_storage.AlertStorage") as MockStorage:
            instance = MockStorage.return_value
            instance.get_active_alerts.return_value = alerts

            result = await service._fetch_alerts_24h()
            severities = [a["severity"] for a in result["alerts"]]
            assert severities == ["S3", "S2", "S1"]


# ============================================================================
# Test _fetch_top_movers
# ============================================================================

class TestFetchTopMovers:
    """Tests for top movers calculation."""

    @pytest.mark.asyncio
    async def test_top_movers_with_history(self, service):
        """Should calculate movers from current vs historical."""
        items = [
            {"symbol": "BTC", "amount": 0.5, "value_usd": 30000},
            {"symbol": "ETH", "amount": 10, "value_usd": 15000},
        ]
        mock_balance = _make_mock_balance_result(items)

        historical = [
            {
                "date": (datetime.now() - timedelta(hours=25)).isoformat(),
                "items": [
                    {"symbol": "BTC", "amount": 0.5, "value_usd": 28000},
                    {"symbol": "ETH", "amount": 10, "value_usd": 16000},
                ]
            }
        ]

        with patch("services.balance_service.balance_service") as mock_bs, \
             patch("services.portfolio.portfolio_analytics") as mock_pa:

            mock_bs.resolve_current_balances = AsyncMock(return_value=mock_balance)
            mock_pa._load_historical_data.return_value = historical

            result = await service._fetch_top_movers("jack", "cointracking")

            assert result["period"] == "24h"
            assert len(result["movers"]) == 2
            # BTC: 60000 now vs 56000 then = +7.14%
            btc_mover = next(m for m in result["movers"] if m["symbol"] == "BTC")
            assert btc_mover["change_pct"] > 0

    @pytest.mark.asyncio
    async def test_top_movers_no_history(self, service):
        """Should fallback to current positions when no history."""
        items = [
            {"symbol": "BTC", "amount": 0.5, "value_usd": 30000},
            {"symbol": "ETH", "amount": 10, "value_usd": 15000},
        ]
        mock_balance = _make_mock_balance_result(items)

        with patch("services.balance_service.balance_service") as mock_bs, \
             patch("services.portfolio.portfolio_analytics") as mock_pa:

            mock_bs.resolve_current_balances = AsyncMock(return_value=mock_balance)
            mock_pa._load_historical_data.return_value = []

            result = await service._fetch_top_movers("jack", "cointracking")

            assert result["period"] == "current"
            assert len(result["movers"]) == 2

    @pytest.mark.asyncio
    async def test_top_movers_empty_portfolio(self, service):
        """Should handle empty portfolio."""
        with patch("services.balance_service.balance_service") as mock_bs:
            mock_bs.resolve_current_balances = AsyncMock(return_value={"items": []})

            result = await service._fetch_top_movers("jack", "cointracking")
            assert result["movers"] == []

    @pytest.mark.asyncio
    async def test_top_movers_sorted_by_abs_change(self, service):
        """Should sort by absolute change percentage."""
        items = [
            {"symbol": "BTC", "amount": 1, "value_usd": 60000},
            {"symbol": "ETH", "amount": 10, "value_usd": 15000},
            {"symbol": "SOL", "amount": 100, "value_usd": 10000},
        ]
        mock_balance = _make_mock_balance_result(items)

        historical = [
            {
                "date": (datetime.now() - timedelta(hours=25)).isoformat(),
                "items": [
                    {"symbol": "BTC", "amount": 1, "value_usd": 59000},   # +1.7%
                    {"symbol": "ETH", "amount": 10, "value_usd": 16000},  # -6.25%
                    {"symbol": "SOL", "amount": 100, "value_usd": 9500},  # +5.26%
                ]
            }
        ]

        with patch("services.balance_service.balance_service") as mock_bs, \
             patch("services.portfolio.portfolio_analytics") as mock_pa:

            mock_bs.resolve_current_balances = AsyncMock(return_value=mock_balance)
            mock_pa._load_historical_data.return_value = historical

            result = await service._fetch_top_movers("jack", "cointracking")

            # Sorted by abs change: ETH (-6.25%) > SOL (+5.26%) > BTC (+1.7%)
            symbols = [m["symbol"] for m in result["movers"]]
            assert symbols[0] == "ETH"  # Biggest absolute change

    @pytest.mark.asyncio
    async def test_top_movers_limits_to_5(self, service):
        """Should limit to top 5 movers."""
        items = [
            {"symbol": f"COIN{i}", "amount": 10, "value_usd": 1000 + i * 100}
            for i in range(10)
        ]
        mock_balance = _make_mock_balance_result(items)

        historical = [
            {
                "date": (datetime.now() - timedelta(hours=25)).isoformat(),
                "items": [
                    {"symbol": f"COIN{i}", "amount": 10, "value_usd": 900 + i * 100}
                    for i in range(10)
                ]
            }
        ]

        with patch("services.balance_service.balance_service") as mock_bs, \
             patch("services.portfolio.portfolio_analytics") as mock_pa:

            mock_bs.resolve_current_balances = AsyncMock(return_value=mock_balance)
            mock_pa._load_historical_data.return_value = historical

            result = await service._fetch_top_movers("jack", "cointracking")
            assert len(result["movers"]) <= 5


# ============================================================================
# Test _fetch_signals
# ============================================================================

class TestFetchSignals:
    """Tests for ML signals fetching."""

    @pytest.mark.asyncio
    async def test_signals_extraction(self, service):
        """Should extract key signal fields."""
        mock_state = _make_mock_state()

        with patch("services.execution.governance.governance_engine") as mock_ge:
            mock_ge.get_current_state = AsyncMock(return_value=mock_state)

            result = await service._fetch_signals()

            assert "regime" in result
            assert "volatility" in result
            assert "sentiment" in result
            assert "correlation" in result
            assert "as_of" in result


# ============================================================================
# Test Telegram Formatting
# ============================================================================

class TestTelegramFormat:
    """Tests for Telegram message formatting."""

    def test_format_full_brief(self, service):
        """Should format a complete brief."""
        brief = {
            "generated_at": "2026-02-10T07:30:00",
            "pnl": {
                "total_value_usd": 50000,
                "24h": {"available": True, "absolute_change": 500, "percentage_change": 1.0},
                "7d": {"available": True, "absolute_change": 2000, "percentage_change": 4.0},
                "30d": {"available": True, "absolute_change": -1000, "percentage_change": -2.0},
            },
            "decision_index": {"blended_score": 72, "decision_score": 72, "confidence": 85},
            "alerts": {"total": 2, "alerts": [
                {"severity": "S3", "title": "Volatility Spike"},
                {"severity": "S2", "title": "Position Size Warning"},
            ]},
            "top_movers": {"movers": [
                {"symbol": "BTC", "change_pct": 5.2},
                {"symbol": "ETH", "change_pct": -3.1},
            ], "period": "24h"},
            "warnings": [],
        }

        text = service.format_telegram(brief)

        assert "*Morning Brief*" in text
        assert "Portfolio: $50,000" in text
        assert "24h" in text
        assert "Decision Index" in text
        assert "72" in text
        assert "Alerts" in text
        assert "Top Movers" in text
        assert "BTC" in text

    def test_format_with_warnings(self, service):
        """Should mention unavailable sections."""
        brief = {
            "generated_at": "2026-02-10T07:30:00",
            "pnl": None,
            "decision_index": None,
            "alerts": None,
            "top_movers": None,
            "warnings": ["pnl: timeout", "decision_index: unavailable"],
        }

        text = service.format_telegram(brief)
        assert "2 sections unavailable" in text

    def test_format_negative_pnl(self, service):
        """Should show correct emoji for negative P&L."""
        brief = {
            "generated_at": "2026-02-10T07:30:00",
            "pnl": {
                "total_value_usd": 50000,
                "24h": {"available": True, "absolute_change": -500, "percentage_change": -1.0},
                "7d": {"available": False},
                "30d": {"available": False},
            },
            "decision_index": None,
            "alerts": None,
            "top_movers": None,
            "warnings": [],
        }

        text = service.format_telegram(brief)
        # Should contain cross mark emoji for negative
        assert "-$500" in text or "-500" in text

    def test_format_no_alerts(self, service):
        """Should not show alerts section when total=0."""
        brief = {
            "generated_at": "2026-02-10T07:30:00",
            "pnl": None,
            "decision_index": None,
            "alerts": {"total": 0, "alerts": []},
            "top_movers": None,
            "warnings": [],
        }

        text = service.format_telegram(brief)
        assert "Alerts" not in text


# ============================================================================
# Test Email HTML Formatting
# ============================================================================

class TestEmailFormat:
    """Tests for email HTML formatting."""

    def test_email_contains_html_tags(self, service):
        """Should produce valid HTML."""
        brief = {
            "generated_at": "2026-02-10T07:30:00",
            "pnl": {
                "total_value_usd": 50000,
                "24h": {"available": True, "absolute_change": 500, "percentage_change": 1.0},
                "7d": {"available": False},
                "30d": {"available": False},
            },
            "decision_index": {"blended_score": 72, "decision_score": 72},
            "alerts": {"total": 1, "alerts": [{"severity": "S2", "title": "Test"}]},
            "top_movers": None,
            "warnings": [],
        }

        html = service.format_email_html(brief)

        assert "<h2>Morning Brief</h2>" in html
        assert "<table" in html
        assert "Decision Index" in html
        assert "72" in html
        assert "<li>" in html

    def test_email_color_coding(self, service):
        """Should use green for positive, red for negative."""
        brief = {
            "generated_at": "2026-02-10",
            "pnl": {
                "total_value_usd": 50000,
                "24h": {"available": True, "absolute_change": 500, "percentage_change": 1.0},
                "7d": {"available": True, "absolute_change": -200, "percentage_change": -0.4},
                "30d": {"available": False},
            },
            "decision_index": None,
            "alerts": None,
            "top_movers": None,
            "warnings": [],
        }

        html = service.format_email_html(brief)
        assert "color:green" in html
        assert "color:red" in html


# ============================================================================
# Test Singleton
# ============================================================================

class TestSingleton:
    """Test singleton instance."""

    def test_singleton_exists(self):
        """Should have a module-level singleton."""
        assert morning_brief_service is not None
        assert isinstance(morning_brief_service, MorningBriefService)


# ============================================================================
# Test API Endpoints
# ============================================================================

class TestMorningBriefEndpoints:
    """Tests for the API endpoint module."""

    def test_router_has_correct_prefix(self):
        """Router should have /api/morning-brief prefix."""
        from api.morning_brief_endpoints import router
        assert router.prefix == "/api/morning-brief"

    def test_router_has_3_routes(self):
        """Router should have 3 endpoints."""
        from api.morning_brief_endpoints import router
        assert len(router.routes) == 3

    def test_cache_dict_exists(self):
        """Cache dict should exist."""
        from api.morning_brief_endpoints import _latest_briefs
        assert isinstance(_latest_briefs, dict)

    def test_router_tags(self):
        """Router should have correct tags."""
        from api.morning_brief_endpoints import router
        assert "morning-brief" in router.tags


# ============================================================================
# Test Scheduler Job
# ============================================================================

class TestSchedulerJob:
    """Tests for the morning brief scheduler job."""

    def test_job_function_exists(self):
        """Should have job_morning_brief function."""
        from api.scheduler import job_morning_brief
        assert callable(job_morning_brief)
        assert asyncio.iscoroutinefunction(job_morning_brief)
