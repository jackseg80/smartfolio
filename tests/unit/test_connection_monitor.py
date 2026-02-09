"""Tests unitaires pour services/monitoring/connection_monitor.py"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from services.monitoring.connection_monitor import (
    ConnectionStatus,
    AlertLevel,
    ConnectionMetrics,
    Alert,
    ConnectionMonitor,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def monitor(tmp_path):
    """Fresh ConnectionMonitor with tmp storage."""
    mon = ConnectionMonitor()
    mon.storage_path = tmp_path / "monitoring"
    mon.storage_path.mkdir()
    return mon


def _make_metrics(
    exchange="binance",
    connected=True,
    response_time_ms=100.0,
    success_rate_1h=100.0,
    success_rate_24h=100.0,
    uptime_percentage=100.0,
    error_count_1h=0,
    last_error=None,
    api_calls_count=3,
    status=ConnectionStatus.HEALTHY,
    timestamp=None,
):
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    return ConnectionMetrics(
        exchange=exchange,
        timestamp=timestamp,
        connected=connected,
        response_time_ms=response_time_ms,
        success_rate_1h=success_rate_1h,
        success_rate_24h=success_rate_24h,
        uptime_percentage=uptime_percentage,
        error_count_1h=error_count_1h,
        last_error=last_error,
        api_calls_count=api_calls_count,
        status=status,
    )


# ── Enum Tests ──────────────────────────────────────────────────────────

class TestEnums:
    def test_connection_status_values(self):
        assert ConnectionStatus.HEALTHY.value == "healthy"
        assert ConnectionStatus.DEGRADED.value == "degraded"
        assert ConnectionStatus.UNSTABLE.value == "unstable"
        assert ConnectionStatus.CRITICAL.value == "critical"
        assert ConnectionStatus.OFFLINE.value == "offline"

    def test_alert_level_values(self):
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"


# ── Dataclass Tests ─────────────────────────────────────────────────────

class TestConnectionMetrics:
    def test_to_dict_converts_status_enum(self):
        m = _make_metrics(status=ConnectionStatus.DEGRADED)
        d = m.to_dict()
        assert d["status"] == "degraded"
        assert d["exchange"] == "binance"
        assert d["connected"] is True

    def test_to_dict_includes_all_fields(self):
        m = _make_metrics(last_error="timeout")
        d = m.to_dict()
        expected_keys = {
            "exchange", "timestamp", "connected", "response_time_ms",
            "success_rate_1h", "success_rate_24h", "uptime_percentage",
            "error_count_1h", "last_error", "api_calls_count", "status",
        }
        assert set(d.keys()) == expected_keys
        assert d["last_error"] == "timeout"


class TestAlert:
    def test_to_dict_converts_level_enum(self):
        alert = Alert(
            id="test_1", exchange="binance", level=AlertLevel.WARNING,
            message="High latency", timestamp=datetime.now(timezone.utc).isoformat(),
        )
        d = alert.to_dict()
        assert d["level"] == "warning"
        assert d["resolved"] is False
        assert d["resolution_time"] is None

    def test_alert_with_resolution(self):
        now = datetime.now(timezone.utc).isoformat()
        alert = Alert(
            id="test_2", exchange="kraken", level=AlertLevel.CRITICAL,
            message="Offline", timestamp=now, resolved=True, resolution_time=now,
        )
        d = alert.to_dict()
        assert d["resolved"] is True
        assert d["resolution_time"] == now


# ── ConnectionMonitor._determine_status ─────────────────────────────────

class TestDetermineStatus:
    def test_offline_when_not_connected(self, monitor):
        status = monitor._determine_status(
            connected=False, response_time=50.0,
            success_rate_1h=100.0, success_rate_24h=100.0, uptime=100.0,
        )
        assert status == ConnectionStatus.OFFLINE

    def test_healthy_normal_conditions(self, monitor):
        status = monitor._determine_status(
            connected=True, response_time=200.0,
            success_rate_1h=99.0, success_rate_24h=99.0, uptime=99.5,
        )
        assert status == ConnectionStatus.HEALTHY

    def test_critical_high_response_time(self, monitor):
        status = monitor._determine_status(
            connected=True, response_time=6000.0,
            success_rate_1h=100.0, success_rate_24h=100.0, uptime=100.0,
        )
        assert status == ConnectionStatus.CRITICAL

    def test_critical_low_success_rate(self, monitor):
        status = monitor._determine_status(
            connected=True, response_time=100.0,
            success_rate_1h=85.0, success_rate_24h=95.0, uptime=99.0,
        )
        assert status == ConnectionStatus.CRITICAL

    def test_critical_low_uptime(self, monitor):
        status = monitor._determine_status(
            connected=True, response_time=100.0,
            success_rate_1h=99.0, success_rate_24h=99.0, uptime=90.0,
        )
        assert status == ConnectionStatus.CRITICAL

    def test_degraded_warning_response_time(self, monitor):
        status = monitor._determine_status(
            connected=True, response_time=3000.0,
            success_rate_1h=99.0, success_rate_24h=99.0, uptime=99.0,
        )
        assert status == ConnectionStatus.DEGRADED

    def test_unstable_low_24h_rate(self, monitor):
        status = monitor._determine_status(
            connected=True, response_time=3000.0,
            success_rate_1h=96.0, success_rate_24h=90.0, uptime=99.0,
        )
        assert status == ConnectionStatus.UNSTABLE


# ── ConnectionMonitor metric calculations ───────────────────────────────

class TestMetricCalculations:
    def test_success_rate_no_history(self, monitor):
        assert monitor._calculate_success_rate("unknown", hours=1) == 100.0

    def test_success_rate_with_metrics(self, monitor):
        now = datetime.now(timezone.utc)
        monitor.metrics_history["binance"] = [
            _make_metrics(connected=True, api_calls_count=3, timestamp=(now - timedelta(minutes=i)).isoformat())
            for i in range(5)
        ]
        # Add 2 failed
        monitor.metrics_history["binance"].append(
            _make_metrics(connected=False, api_calls_count=0, timestamp=(now - timedelta(minutes=6)).isoformat())
        )
        monitor.metrics_history["binance"].append(
            _make_metrics(connected=True, api_calls_count=0, timestamp=(now - timedelta(minutes=7)).isoformat())
        )
        rate = monitor._calculate_success_rate("binance", hours=1)
        # 5 successful out of 7
        assert abs(rate - (5.0 / 7.0) * 100.0) < 0.1

    def test_uptime_no_history(self, monitor):
        assert monitor._calculate_uptime("unknown", hours=24) == 100.0

    def test_uptime_with_metrics(self, monitor):
        now = datetime.now(timezone.utc)
        monitor.metrics_history["kraken"] = [
            _make_metrics(exchange="kraken", connected=True, timestamp=(now - timedelta(minutes=i)).isoformat())
            for i in range(8)
        ]
        monitor.metrics_history["kraken"].append(
            _make_metrics(exchange="kraken", connected=False, timestamp=(now - timedelta(minutes=9)).isoformat())
        )
        monitor.metrics_history["kraken"].append(
            _make_metrics(exchange="kraken", connected=False, timestamp=(now - timedelta(minutes=10)).isoformat())
        )
        uptime = monitor._calculate_uptime("kraken", hours=24)
        assert abs(uptime - 80.0) < 0.1

    def test_count_errors_no_history(self, monitor):
        assert monitor._count_errors("unknown", hours=1) == 0

    def test_count_errors_with_metrics(self, monitor):
        now = datetime.now(timezone.utc)
        monitor.metrics_history["binance"] = [
            _make_metrics(last_error=None, timestamp=(now - timedelta(minutes=1)).isoformat()),
            _make_metrics(last_error="timeout", timestamp=(now - timedelta(minutes=2)).isoformat()),
            _make_metrics(last_error="connection refused", timestamp=(now - timedelta(minutes=3)).isoformat()),
            _make_metrics(last_error=None, timestamp=(now - timedelta(minutes=4)).isoformat()),
        ]
        assert monitor._count_errors("binance", hours=1) == 2


# ── get_current_status ──────────────────────────────────────────────────

class TestGetCurrentStatus:
    def test_empty_history(self, monitor):
        assert monitor.get_current_status() == {}

    def test_returns_latest_metrics(self, monitor):
        now = datetime.now(timezone.utc).isoformat()
        monitor.metrics_history["binance"] = [
            _make_metrics(response_time_ms=100.0, timestamp=now),
            _make_metrics(response_time_ms=250.0, timestamp=now),
        ]
        status = monitor.get_current_status()
        assert "binance" in status
        assert status["binance"]["response_time_ms"] == 250.0
        assert status["binance"]["status"] == "healthy"


# ── get_alerts ──────────────────────────────────────────────────────────

class TestGetAlerts:
    def test_empty_alerts(self, monitor):
        assert monitor.get_alerts() == []

    def test_filter_resolved(self, monitor):
        now = datetime.now(timezone.utc).isoformat()
        monitor.alerts = [
            Alert(id="1", exchange="a", level=AlertLevel.WARNING, message="x", timestamp=now, resolved=False),
            Alert(id="2", exchange="b", level=AlertLevel.CRITICAL, message="y", timestamp=now, resolved=True),
        ]
        unresolved = monitor.get_alerts(resolved=False)
        assert len(unresolved) == 1
        assert unresolved[0]["id"] == "1"

        resolved = monitor.get_alerts(resolved=True)
        assert len(resolved) == 1
        assert resolved[0]["id"] == "2"

    def test_all_alerts(self, monitor):
        now = datetime.now(timezone.utc).isoformat()
        monitor.alerts = [
            Alert(id="1", exchange="a", level=AlertLevel.INFO, message="x", timestamp=now),
            Alert(id="2", exchange="b", level=AlertLevel.WARNING, message="y", timestamp=now),
        ]
        all_alerts = monitor.get_alerts()
        assert len(all_alerts) == 2


# ── get_performance_summary ─────────────────────────────────────────────

class TestGetPerformanceSummary:
    def test_empty_history(self, monitor):
        summary = monitor.get_performance_summary()
        assert summary == {"message": "No metrics available"}

    def test_summary_counts(self, monitor):
        now = datetime.now(timezone.utc).isoformat()
        monitor.metrics_history["exchange_a"] = [
            _make_metrics(exchange="exchange_a", status=ConnectionStatus.HEALTHY, response_time_ms=100.0, timestamp=now),
        ]
        monitor.metrics_history["exchange_b"] = [
            _make_metrics(exchange="exchange_b", status=ConnectionStatus.DEGRADED, response_time_ms=3000.0, timestamp=now),
        ]
        monitor.metrics_history["exchange_c"] = [
            _make_metrics(exchange="exchange_c", status=ConnectionStatus.OFFLINE, connected=False, timestamp=now),
        ]

        summary = monitor.get_performance_summary()
        assert summary["total_exchanges"] == 3
        assert summary["healthy_exchanges"] == 1
        assert summary["degraded_exchanges"] == 1
        assert summary["offline_exchanges"] == 1
        assert summary["active_alerts"] == 0
        # Only connected exchanges contribute to avg response time
        assert summary["average_response_time"] == pytest.approx((100.0 + 3000.0) / 2, abs=1)


# ── _store_metrics ──────────────────────────────────────────────────────

class TestStoreMetrics:
    def test_stores_in_memory(self, monitor):
        m = _make_metrics(exchange="test_exchange")
        with patch("asyncio.create_task"):
            monitor._store_metrics(m)
        assert "test_exchange" in monitor.metrics_history
        assert len(monitor.metrics_history["test_exchange"]) == 1

    def test_appends_to_existing(self, monitor):
        m1 = _make_metrics(exchange="test_exchange")
        m2 = _make_metrics(exchange="test_exchange", response_time_ms=200.0)
        with patch("asyncio.create_task"):
            monitor._store_metrics(m1)
            monitor._store_metrics(m2)
        assert len(monitor.metrics_history["test_exchange"]) == 2
