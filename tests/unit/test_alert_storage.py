"""
Tests unitaires pour services/alerts/alert_storage.py

Couvre: AlertStorage (file mode, in-memory mode, cascade fallback),
       _serialize_for_json, dedup, rate limit, purge, metrics, ping,
       scheduler lock, acknowledge, snooze, mark_applied.
"""

import json
import time
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from services.alerts.alert_storage import AlertStorage, _serialize_for_json
from services.alerts.alert_types import Alert, AlertType, AlertSeverity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert(
    alert_type=AlertType.VOL_Q90_CROSS,
    severity=AlertSeverity.S1,
    alert_id="test-001",
    created_at=None,
    data=None,
    acknowledged_at=None,
    resolved_at=None,
    snooze_until=None,
):
    return Alert(
        id=alert_id,
        alert_type=alert_type,
        severity=severity,
        created_at=created_at or datetime.now(),
        data=data or {"current_value": 0.25, "adaptive_threshold": 0.20},
        acknowledged_at=acknowledged_at,
        resolved_at=resolved_at,
        snooze_until=snooze_until,
    )


# ---------------------------------------------------------------------------
# _serialize_for_json
# ---------------------------------------------------------------------------

class TestSerializeForJson:
    def test_enum_serialized_to_value(self):
        assert _serialize_for_json(AlertSeverity.S1) == "S1"
        assert _serialize_for_json(AlertType.REGIME_FLIP) == "REGIME_FLIP"

    def test_datetime_serialized_to_iso(self):
        dt = datetime(2026, 1, 15, 10, 30, 0)
        assert _serialize_for_json(dt) == "2026-01-15T10:30:00"

    def test_dict_recursion(self):
        d = {"severity": AlertSeverity.S2, "ts": datetime(2026, 1, 1)}
        result = _serialize_for_json(d)
        assert result["severity"] == "S2"
        assert "2026" in result["ts"]

    def test_list_recursion(self):
        lst = [AlertSeverity.S3, datetime(2026, 2, 1)]
        result = _serialize_for_json(lst)
        assert result[0] == "S3"

    def test_plain_values_passthrough(self):
        assert _serialize_for_json(42) == 42
        assert _serialize_for_json("hello") == "hello"
        assert _serialize_for_json(None) is None

    def test_tuple_converted_to_list(self):
        result = _serialize_for_json((AlertSeverity.S1, 1))
        assert isinstance(result, list)
        assert result[0] == "S1"


# ---------------------------------------------------------------------------
# AlertStorage — File Mode (forced via storage_mode override)
# ---------------------------------------------------------------------------

class TestAlertStorageFileMode:
    """Tests avec storage_mode='file' forcé (no Redis → default in_memory, on force file)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.json_file = tmp_path / "alerts.json"
        self.storage = AlertStorage(
            redis_url=None,
            json_file=str(self.json_file),
            max_alerts=100,
            purge_days=30,
            enable_fallback_cascade=True,
        )
        # Force file mode for these tests (no Redis = default in_memory)
        self.storage.storage_mode = "file"

    def test_init_creates_json_file(self):
        assert self.json_file.exists()
        data = json.loads(self.json_file.read_text())
        assert "alerts" in data
        assert "metadata" in data

    def test_storage_mode_is_file(self):
        assert self.storage.storage_mode == "file"

    def test_store_alert_file_mode(self):
        alert = _make_alert()
        result = self.storage.store_alert(alert)
        assert result is True

        data = json.loads(self.json_file.read_text())
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["id"] == "test-001"

    def test_store_multiple_alerts(self):
        for i in range(5):
            alert = _make_alert(
                alert_id=f"test-{i:03d}",
                created_at=datetime.now() + timedelta(minutes=i * 6),
            )
            self.storage.store_alert(alert)

        data = json.loads(self.json_file.read_text())
        assert len(data["alerts"]) == 5

    def test_auto_rotation_keeps_max_alerts(self):
        self.storage.max_alerts = 5
        for i in range(10):
            alert = _make_alert(
                alert_id=f"rot-{i:03d}",
                created_at=datetime.now() + timedelta(minutes=i * 6),
            )
            self.storage.store_alert(alert)

        data = json.loads(self.json_file.read_text())
        assert len(data["alerts"]) <= 5
        # Should keep the LAST 5
        ids = [a["id"] for a in data["alerts"]]
        assert "rot-009" in ids

    def test_get_active_alerts_returns_active_only(self):
        # Store 3 alerts with different time buckets to avoid dedup
        self.storage.store_alert(_make_alert(
            alert_id="active-1",
            created_at=datetime.now(),
        ))
        self.storage.store_alert(_make_alert(
            alert_id="ack-1",
            alert_type=AlertType.REGIME_FLIP,
            created_at=datetime.now(),
        ))
        self.storage.store_alert(_make_alert(
            alert_id="resolved-1",
            alert_type=AlertType.CORR_HIGH,
            created_at=datetime.now(),
        ))

        # Acknowledge and resolve
        self.storage.acknowledge_alert("ack-1", "jack")
        self.storage._update_alert_field("resolved-1", {"resolved_at": datetime.now().isoformat()})

        active = self.storage.get_active_alerts()
        ids = [a.id for a in active]
        assert "active-1" in ids
        assert "ack-1" not in ids
        assert "resolved-1" not in ids

    def test_get_active_alerts_skips_snoozed(self):
        self.storage.store_alert(_make_alert(alert_id="snz-1"))
        # Snooze for 60 minutes
        self.storage.snooze_alert("snz-1", 60)

        active = self.storage.get_active_alerts(include_snoozed=False)
        ids = [a.id for a in active]
        assert "snz-1" not in ids

        # With snoozed included
        active_incl = self.storage.get_active_alerts(include_snoozed=True)
        ids_incl = [a.id for a in active_incl]
        assert "snz-1" in ids_incl

    def test_acknowledge_alert(self):
        self.storage.store_alert(_make_alert(alert_id="ack-test"))
        result = self.storage.acknowledge_alert("ack-test", "jack")
        assert result is True

        data = json.loads(self.json_file.read_text())
        alert = [a for a in data["alerts"] if a["id"] == "ack-test"][0]
        assert alert["acknowledged_by"] == "jack"
        assert alert["acknowledged_at"] is not None

    def test_acknowledge_nonexistent_alert(self):
        result = self.storage.acknowledge_alert("nonexistent", "jack")
        assert result is False

    def test_snooze_alert(self):
        self.storage.store_alert(_make_alert(alert_id="snooze-test"))
        result = self.storage.snooze_alert("snooze-test", 30)
        assert result is True

        data = json.loads(self.json_file.read_text())
        alert = [a for a in data["alerts"] if a["id"] == "snooze-test"][0]
        assert alert["snooze_until"] is not None

    def test_mark_alert_applied(self):
        self.storage.store_alert(_make_alert(alert_id="apply-test"))
        result = self.storage.mark_alert_applied("apply-test", "jack")
        assert result is True

        data = json.loads(self.json_file.read_text())
        alert = [a for a in data["alerts"] if a["id"] == "apply-test"][0]
        assert alert["applied_by"] == "jack"

    def test_purge_old_alerts(self):
        # Store old alert (40 days ago) and recent alert
        old_alert = _make_alert(
            alert_id="old-1",
            created_at=datetime.now() - timedelta(days=40),
        )
        recent_alert = _make_alert(
            alert_id="recent-1",
            alert_type=AlertType.REGIME_FLIP,  # Different type to avoid dedup
        )

        self.storage.store_alert(old_alert)
        self.storage.store_alert(recent_alert)

        purged = self.storage.purge_old_alerts()
        assert purged == 1

        data = json.loads(self.json_file.read_text())
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["id"] == "recent-1"

    def test_purge_no_old_alerts(self):
        self.storage.store_alert(_make_alert(alert_id="recent-only"))
        purged = self.storage.purge_old_alerts()
        assert purged == 0

    def test_ping_file_mode(self):
        assert self.storage.ping() is True

    def test_get_metrics_file_mode(self):
        self.storage.store_alert(_make_alert(
            alert_id="m-1", severity=AlertSeverity.S1,
        ))
        self.storage.store_alert(_make_alert(
            alert_id="m-2", severity=AlertSeverity.S2,
            alert_type=AlertType.REGIME_FLIP,
            created_at=datetime.now() + timedelta(minutes=6),
        ))

        metrics = self.storage.get_metrics()
        assert metrics["storage_mode"] == "file"
        assert metrics["total_alerts"] == 2
        assert metrics["fallback_cascade_enabled"] is True
        assert metrics["is_degraded"] is False

    def test_load_json_data_corrupt_file(self):
        # Corrupt the JSON file
        self.json_file.write_text("not-json{{{")
        data = self.storage._load_json_data()
        assert data == {"alerts": [], "metadata": {}}


# ---------------------------------------------------------------------------
# AlertStorage — Legacy Mode (no cascade, uses file directly)
# ---------------------------------------------------------------------------

class TestAlertStorageLegacyMode:
    """Tests avec enable_fallback_cascade=False."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.json_file = tmp_path / "alerts_legacy.json"
        self.storage = AlertStorage(
            redis_url=None,
            json_file=str(self.json_file),
            max_alerts=50,
            enable_fallback_cascade=False,
        )

    def test_store_uses_legacy(self):
        alert = _make_alert(alert_id="legacy-1")
        result = self.storage.store_alert(alert)
        assert result is True

        data = json.loads(self.json_file.read_text())
        assert len(data["alerts"]) == 1

    def test_get_active_uses_legacy(self):
        self.storage.store_alert(_make_alert(alert_id="leg-active"))
        active = self.storage.get_active_alerts()
        assert len(active) >= 1

    def test_rate_limit_legacy_allows(self):
        result = self.storage.check_rate_limit(
            AlertType.VOL_Q90_CROSS, AlertSeverity.S1, window_minutes=30
        )
        assert result is True

    def test_legacy_auto_rotation(self):
        self.storage.max_alerts = 3
        for i in range(5):
            self.storage.store_alert(_make_alert(
                alert_id=f"leg-{i}",
                created_at=datetime.now() + timedelta(minutes=i * 6),
            ))

        data = json.loads(self.json_file.read_text())
        assert len(data["alerts"]) <= 3

    def test_legacy_get_active_skips_ack(self):
        self.storage.store_alert(_make_alert(alert_id="leg-ack"))
        self.storage.acknowledge_alert("leg-ack", "jack")
        active = self.storage.get_active_alerts()
        assert len(active) == 0

    def test_legacy_get_active_skips_resolved(self):
        self.storage.store_alert(_make_alert(alert_id="leg-res"))
        self.storage._update_alert_field("leg-res", {"resolved_at": datetime.now().isoformat()})
        active = self.storage.get_active_alerts()
        assert len(active) == 0


# ---------------------------------------------------------------------------
# AlertStorage — In-Memory Mode (degraded)
# ---------------------------------------------------------------------------

class TestAlertStorageInMemoryMode:
    """Tests avec storage_mode='in_memory' (default sans Redis)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.json_file = tmp_path / "alerts_mem.json"
        self.storage = AlertStorage(
            redis_url=None,
            json_file=str(self.json_file),
            max_alerts=10,
            enable_fallback_cascade=True,
        )
        # Default without Redis is in_memory
        assert self.storage.storage_mode == "in_memory"

    def test_store_in_memory(self):
        alert = _make_alert(alert_id="mem-1")
        result = self.storage.store_alert(alert)
        assert result is True
        assert len(self.storage._degraded_alerts) == 1

    def test_memory_dedup(self):
        ts = datetime(2026, 2, 1, 12, 0, 0)
        alert1 = _make_alert(alert_id="dup-1", created_at=ts)
        alert2 = _make_alert(alert_id="dup-2", created_at=ts)

        self.storage.store_alert(alert1)
        result = self.storage.store_alert(alert2)
        # Same type + same minute → duplicate
        assert result is False
        assert len(self.storage._degraded_alerts) == 1

    def test_memory_auto_rotation(self):
        for i in range(15):
            alert = _make_alert(
                alert_id=f"mem-{i:03d}",
                created_at=datetime.now() + timedelta(minutes=i * 6),
            )
            self.storage.store_alert(alert)

        assert len(self.storage._degraded_alerts) <= 10

    def test_get_active_from_memory(self):
        alert = _make_alert(alert_id="mem-active")
        self.storage.store_alert(alert)

        active = self.storage.get_active_alerts()
        assert len(active) >= 1
        assert active[0].id == "mem-active"

    def test_memory_get_active_skips_acknowledged(self):
        self.storage.store_alert(_make_alert(alert_id="mem-ack"))
        # Manually set ack in memory
        self.storage._degraded_alerts[0]["acknowledged_at"] = datetime.now().isoformat()

        active = self.storage.get_active_alerts()
        assert len(active) == 0

    def test_memory_get_active_skips_resolved(self):
        self.storage.store_alert(_make_alert(alert_id="mem-res"))
        self.storage._degraded_alerts[0]["resolved_at"] = datetime.now().isoformat()

        active = self.storage.get_active_alerts()
        assert len(active) == 0

    def test_memory_get_active_skips_snoozed(self):
        self.storage.store_alert(_make_alert(alert_id="mem-snz"))
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        self.storage._degraded_alerts[0]["snooze_until"] = future

        active = self.storage.get_active_alerts(include_snoozed=False)
        assert len(active) == 0

        active_incl = self.storage.get_active_alerts(include_snoozed=True)
        assert len(active_incl) == 1

    def test_get_memory_metrics(self):
        self.storage.store_alert(_make_alert(alert_id="metr-1"))
        metrics = self.storage.get_metrics()
        assert metrics["storage_mode"] == "in_memory"
        assert metrics["is_degraded"] is True
        assert metrics["total_alerts"] == 1
        assert metrics["memory_alerts_count"] == 1


# ---------------------------------------------------------------------------
# Cascade Fallback
# ---------------------------------------------------------------------------

class TestCascadeFallback:
    """Tests du mécanisme de cascade: file → in_memory."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.json_file = tmp_path / "alerts_cascade.json"
        self.storage = AlertStorage(
            redis_url=None,
            json_file=str(self.json_file),
            max_alerts=50,
            enable_fallback_cascade=True,
        )
        # Start in file mode to test cascade
        self.storage.storage_mode = "file"

    def test_file_failure_degrades_to_memory(self):
        """Si le stockage fichier échoue, cascade vers in_memory."""
        alert = _make_alert(alert_id="cascade-1")

        with patch.object(self.storage, "_try_file_store_alert", return_value=(False, "file_error: disk full")):
            result = self.storage.store_alert(alert)

        assert result is True
        assert self.storage.storage_mode == "in_memory"
        assert len(self.storage._degraded_alerts) == 1

    def test_file_duplicate_does_not_cascade(self):
        """Un doublon fichier ne doit pas cascader vers memory."""
        alert = _make_alert(alert_id="dup-cascade")

        with patch.object(self.storage, "_try_file_store_alert", return_value=(False, "duplicate")):
            result = self.storage.store_alert(alert)

        assert result is False
        assert self.storage.storage_mode == "file"  # Not degraded

    def test_get_active_cascade_file_to_memory(self):
        """Si file get échoue, cascade vers memory."""
        # Store in memory directly
        self.storage._degraded_alerts.append({
            "id": "mem-fallback",
            "alert_type": AlertType.VOL_Q90_CROSS.value,
            "severity": AlertSeverity.S1.value,
            "created_at": datetime.now().isoformat(),
            "data": {},
            "acknowledged_at": None,
            "resolved_at": None,
            "snooze_until": None,
            "applied_at": None,
            "applied_by": None,
            "acknowledged_by": None,
            "suggested_action": {},
            "escalation_sources": [],
            "escalation_count": 0,
        })

        with patch.object(self.storage, "_try_file_get_active_alerts", return_value=(None, "file_error")):
            active = self.storage.get_active_alerts()

        assert self.storage.storage_mode == "in_memory"
        assert len(active) == 1

    def test_total_store_cascade_failure(self):
        """Si file ET memory échouent, retourne False."""
        with patch.object(self.storage, "_try_file_store_alert", return_value=(False, "file_error")):
            with patch.object(self.storage, "_try_memory_store_alert", return_value=(False, "memory_error")):
                result = self.storage.store_alert(_make_alert(alert_id="total-fail"))
        assert result is False

    def test_total_get_active_cascade_failure(self):
        """Si tout échoue pour get_active, retourne []."""
        with patch.object(self.storage, "_try_file_get_active_alerts", return_value=(None, "err")):
            with patch.object(self.storage, "_try_memory_get_active_alerts", return_value=(None, "err")):
                active = self.storage.get_active_alerts()
        assert active == []


# ---------------------------------------------------------------------------
# Rate Limiting (File-based legacy fallback)
# ---------------------------------------------------------------------------

class TestRateLimiting:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.json_file = tmp_path / "alerts_rate.json"
        # Use legacy mode for rate limit tests (file-based counting)
        self.storage = AlertStorage(
            redis_url=None,
            json_file=str(self.json_file),
            max_alerts=100,
            enable_fallback_cascade=False,
        )

    def test_rate_limit_allows_within_budget(self):
        result = self.storage.check_rate_limit(
            AlertType.VOL_Q90_CROSS, AlertSeverity.S1, window_minutes=30
        )
        assert result is True

    def test_rate_limit_blocks_s3_after_one(self):
        # Store 1 S3 alert via legacy mode (directly to file)
        self.storage.store_alert(_make_alert(
            alert_id="s3-1",
            severity=AlertSeverity.S3,
        ))

        result = self.storage.check_rate_limit(
            AlertType.VOL_Q90_CROSS, AlertSeverity.S3, window_minutes=30
        )
        # Budget S3 = 1 → already used
        assert result is False

    def test_rate_limit_allows_old_alerts(self):
        # Store 1 S3 alert 40 minutes ago (outside window)
        old = _make_alert(
            alert_id="s3-old",
            severity=AlertSeverity.S3,
            created_at=datetime.now() - timedelta(minutes=40),
        )
        self.storage.store_alert(old)

        result = self.storage.check_rate_limit(
            AlertType.VOL_Q90_CROSS, AlertSeverity.S3, window_minutes=30
        )
        assert result is True

    def test_rate_limit_s2_budget_is_two(self):
        for i in range(2):
            self.storage.store_alert(_make_alert(
                alert_id=f"s2-{i}",
                severity=AlertSeverity.S2,
                created_at=datetime.now() + timedelta(minutes=i * 6),
            ))

        result = self.storage.check_rate_limit(
            AlertType.VOL_Q90_CROSS, AlertSeverity.S2, window_minutes=30
        )
        assert result is False

    def test_rate_limit_s1_budget_is_five(self):
        for i in range(5):
            self.storage.store_alert(_make_alert(
                alert_id=f"s1-{i}",
                severity=AlertSeverity.S1,
                created_at=datetime.now() + timedelta(minutes=i * 6),
            ))

        result = self.storage.check_rate_limit(
            AlertType.VOL_Q90_CROSS, AlertSeverity.S1, window_minutes=30
        )
        assert result is False

    def test_rate_limit_error_returns_true(self):
        """En cas d'erreur, le rate limit doit autoriser."""
        with patch.object(self.storage, "_load_json_data", side_effect=Exception("disk error")):
            result = self.storage.check_rate_limit(
                AlertType.VOL_Q90_CROSS, AlertSeverity.S1, window_minutes=30
            )
        assert result is True

    def test_rate_limit_different_type_not_counted(self):
        """Alertes d'un type différent ne comptent pas."""
        self.storage.store_alert(_make_alert(
            alert_id="rt-1",
            severity=AlertSeverity.S3,
            alert_type=AlertType.REGIME_FLIP,
        ))

        result = self.storage.check_rate_limit(
            AlertType.VOL_Q90_CROSS, AlertSeverity.S3, window_minutes=30
        )
        assert result is True  # Different type → not counted


# ---------------------------------------------------------------------------
# Scheduler Lock (File-based)
# ---------------------------------------------------------------------------

class TestSchedulerLock:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.json_file = tmp_path / "alerts_lock.json"
        self.storage = AlertStorage(
            redis_url=None,
            json_file=str(self.json_file),
        )

    def test_acquire_lock(self):
        result = self.storage.acquire_scheduler_lock("host-1", ttl_seconds=90)
        assert result is True

    def test_lock_prevents_second_acquisition(self):
        self.storage.acquire_scheduler_lock("host-1", ttl_seconds=90)
        result = self.storage.acquire_scheduler_lock("host-2", ttl_seconds=90)
        assert result is False

    def test_release_lock(self):
        self.storage.acquire_scheduler_lock("host-1", ttl_seconds=90)
        self.storage.release_scheduler_lock("host-1")
        # Now another host can acquire
        result = self.storage.acquire_scheduler_lock("host-2", ttl_seconds=90)
        assert result is True

    def test_release_lock_wrong_host_noop(self):
        self.storage.acquire_scheduler_lock("host-1", ttl_seconds=90)
        self.storage.release_scheduler_lock("host-wrong")
        # Lock still held by host-1
        result = self.storage.acquire_scheduler_lock("host-2", ttl_seconds=90)
        assert result is False

    def test_expired_lock_can_be_reacquired(self):
        self.storage.acquire_scheduler_lock("host-1", ttl_seconds=1)
        time.sleep(2)  # Wait for expiry (generous margin on Windows)
        # TTL must match the original — the check uses the caller's ttl_seconds
        result = self.storage.acquire_scheduler_lock("host-2", ttl_seconds=1)
        assert result is True

    def test_acquire_lock_error_returns_false(self):
        """Si erreur fichier, acquire retourne False."""
        with patch.object(Path, "exists", side_effect=Exception("perm")):
            result = self.storage.acquire_scheduler_lock("host-err")
        assert result is False


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.json_file = tmp_path / "alerts_dedup.json"
        self.storage = AlertStorage(
            redis_url=None,
            json_file=str(self.json_file),
            enable_fallback_cascade=False,  # Legacy mode for simpler dedup test
        )

    def test_same_alert_is_duplicate(self):
        ts = datetime(2026, 2, 1, 12, 0, 0)
        alert = _make_alert(alert_id="d-1", created_at=ts)
        self.storage.store_alert(alert)

        # Same type, same time bucket, same values → duplicate
        alert2 = _make_alert(alert_id="d-2", created_at=ts)
        result = self.storage.store_alert(alert2)
        assert result is False

    def test_different_bucket_is_not_duplicate(self):
        ts1 = datetime(2026, 2, 1, 12, 0, 0)
        ts2 = datetime(2026, 2, 1, 12, 10, 0)  # Different 5-min bucket
        alert1 = _make_alert(alert_id="d-1", created_at=ts1)
        alert2 = _make_alert(alert_id="d-2", created_at=ts2)

        self.storage.store_alert(alert1)
        result = self.storage.store_alert(alert2)
        assert result is True

    def test_different_type_is_not_duplicate(self):
        ts = datetime(2026, 2, 1, 12, 0, 0)
        alert1 = _make_alert(alert_id="d-1", alert_type=AlertType.VOL_Q90_CROSS, created_at=ts)
        alert2 = _make_alert(alert_id="d-2", alert_type=AlertType.REGIME_FLIP, created_at=ts)

        self.storage.store_alert(alert1)
        result = self.storage.store_alert(alert2)
        assert result is True

    def test_different_direction_is_not_duplicate(self):
        ts = datetime(2026, 2, 1, 12, 0, 0)
        alert1 = _make_alert(alert_id="d-1", created_at=ts,
                             data={"current_value": 0.25, "adaptive_threshold": 0.20})  # up
        alert2 = _make_alert(alert_id="d-2", created_at=ts,
                             data={"current_value": 0.15, "adaptive_threshold": 0.20})  # down

        self.storage.store_alert(alert1)
        result = self.storage.store_alert(alert2)
        assert result is True


# ---------------------------------------------------------------------------
# Edge Cases & Error Handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.json_file = tmp_path / "alerts_edge.json"
        self.storage = AlertStorage(
            redis_url=None,
            json_file=str(self.json_file),
            enable_fallback_cascade=True,
        )
        self.storage.storage_mode = "file"

    def test_store_alert_with_enum_data(self):
        """Enums dans data doivent être sérialisés correctement."""
        alert = _make_alert(
            alert_id="enum-data",
            data={"severity": AlertSeverity.S2, "type": AlertType.REGIME_FLIP},
        )
        result = self.storage.store_alert(alert)
        assert result is True

        data = json.loads(self.json_file.read_text())
        stored = data["alerts"][0]["data"]
        assert stored["severity"] == "S2"
        assert stored["type"] == "REGIME_FLIP"

    def test_concurrent_store_does_not_corrupt(self):
        """Deux store séquentiels ne corrompent pas le fichier."""
        self.storage.store_alert(_make_alert(
            alert_id="c-1",
            created_at=datetime.now(),
        ))
        self.storage.store_alert(_make_alert(
            alert_id="c-2",
            alert_type=AlertType.REGIME_FLIP,
            created_at=datetime.now(),
        ))

        data = json.loads(self.json_file.read_text())
        assert len(data["alerts"]) == 2

    def test_empty_storage_get_active_returns_empty(self):
        active = self.storage.get_active_alerts()
        assert active == []

    def test_ping_when_file_missing(self, tmp_path):
        """Ping retourne True si le parent existe même si le fichier est absent."""
        missing = tmp_path / "nonexistent" / "alerts.json"
        storage = AlertStorage(
            redis_url=None,
            json_file=str(missing),
            enable_fallback_cascade=True,
        )
        # _ensure_storage_exists creates parent + file
        assert storage.ping() is True

    def test_get_metrics_error_handling(self):
        """get_metrics ne crash pas si load échoue."""
        with patch.object(self.storage, "_load_json_data", side_effect=Exception("oops")):
            metrics = self.storage.get_metrics()
        assert "error" in metrics or "storage_mode" in metrics

    def test_update_alert_field_nonexistent(self):
        result = self.storage._update_alert_field("ghost-id", {"acknowledged_at": "now"})
        assert result is False

    def test_save_json_data_serializes_enums(self):
        """_save_json_data doit sérialiser les Enums correctement."""
        data = {
            "alerts": [{
                "id": "sj-1",
                "severity": AlertSeverity.S2,
                "alert_type": AlertType.CORR_HIGH,
                "created_at": datetime.now(),
                "data": {},
            }],
            "metadata": {}
        }
        self.storage._save_json_data(data)
        loaded = json.loads(self.json_file.read_text())
        assert loaded["alerts"][0]["severity"] == "S2"

    def test_degraded_metrics_counter(self):
        """Les compteurs de dégradation s'incrémentent."""
        self.storage._degraded_metrics["redis_failures"] = 0
        self.storage._degraded_metrics["file_failures"] = 0

        # Simulate file failure counter
        self.storage._degraded_metrics["file_failures"] += 3
        metrics = self.storage.get_metrics()
        assert metrics["file_failures"] == 3

    def test_no_redis_lua_scripts_not_loaded(self):
        """Sans Redis, lua_scripts_loaded doit être False."""
        metrics = self.storage.get_metrics()
        assert metrics["lua_scripts_loaded"] is False
