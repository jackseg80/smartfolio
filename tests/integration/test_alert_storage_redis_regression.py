"""Real Lua regressions. Uses only a dedicated local Redis and UUID key namespace."""
import json
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from urllib.parse import urlparse

import pytest
import redis

from services.alerts.alert_storage import AlertStorage
from services.alerts.alert_types import Alert, AlertType, AlertSeverity


@pytest.fixture
def storage(tmp_path):
    url = os.getenv("SMARTFOLIO_TEST_REDIS_URL")
    if not url:
        pytest.skip("Dedicated test Redis is not configured")
    parsed = urlparse(url)
    assert parsed.hostname in ("127.0.0.1", "localhost") and parsed.port == 46379
    client = redis.from_url(url, decode_responses=True)
    client.ping()
    prefix = "repair-test:" + uuid4().hex + ":"
    instance = AlertStorage(redis_url=url, json_file=str(tmp_path / "alerts.json"))
    assert instance.redis_available
    instance.ALERTS_ZSET = prefix + "timeline"
    instance.ALERTS_HASH_PREFIX = prefix + "data:"
    instance.ACTIVE_ALERTS_SET = prefix + "active"
    instance.DEDUP_PREFIX = prefix + "dedup:"
    instance.RATE_LIMIT_PREFIX = prefix + "rate:"
    try:
        yield instance
    finally:
        keys = list(client.scan_iter(match=prefix + "*", count=100))
        if keys:
            client.delete(*keys)
        instance.redis_client.close()
        client.close()


def alert(id="new", **kwargs):
    return Alert(id=id, alert_type=AlertType.CONTRADICTION_SPIKE, severity=AlertSeverity.S2,
                 data={"current_value": 0.37, "adaptive_threshold": 0.2, "nested": {"enabled": True, "assets": [], "weights": [1, 2]}}, **kwargs)


def test_real_lua_round_trip_keeps_nulls_nested_data_and_empty_lists(storage):
    original = alert(suggested_action={"kind": "review"})
    assert storage._try_redis_store_alert(original)[0]
    raw = storage.redis_client.hgetall(storage.ALERTS_HASH_PREFIX + original.id)
    assert raw["acknowledged_at"] == "" and raw["resolved_at"] == ""
    assert json.loads(raw["escalation_sources"]) == []
    loaded = storage.get_active_alerts()
    assert len(loaded) == 1
    assert loaded[0].data == original.data
    assert loaded[0].suggested_action == original.suggested_action
    assert loaded[0].escalation_sources == []
    assert loaded[0].acknowledged_at is None and loaded[0].applied_by is None


def test_legacy_nulls_are_read_without_writing_or_dropping_real_dates(storage):
    original = alert(applied_at=datetime(2026, 1, 1))
    assert storage._try_redis_store_alert(original)[0]
    key = storage.ALERTS_HASH_PREFIX + original.id
    changes = {k: "userdata: 0x7f123abc" for k in ["acknowledged_at", "resolved_at", "snooze_until", "acknowledged_by", "applied_by"]}
    changes["escalation_sources"] = "{}"
    storage.redis_client.hset(key, mapping=changes)
    before = storage.redis_client.hgetall(key)
    loaded = storage.get_active_alerts()
    assert len(loaded) == 1 and loaded[0].applied_at == original.applied_at
    assert loaded[0].acknowledged_by is None
    assert storage.redis_client.hgetall(key) == before


@pytest.mark.parametrize("field", ["acknowledged_at", "resolved_at"])
def test_true_acknowledgements_and_resolutions_are_excluded(storage, field):
    assert storage._try_redis_store_alert(alert(**{field: datetime.now()}))[0]
    assert storage.get_active_alerts() == []


@pytest.mark.parametrize("aware", [True, False])
def test_snooze_dates_are_compared_as_dates_not_epoch_strings(storage, aware):
    now = datetime.now(timezone.utc) if aware else datetime.now()
    assert storage._try_redis_store_alert(alert(snooze_until=now + timedelta(hours=1)))[0]
    assert storage.get_active_alerts() == []
    assert len(storage.get_active_alerts(include_snoozed=True)) == 1
    storage._update_alert_field("new", {"snooze_until": (now - timedelta(hours=1)).isoformat()})
    assert len(storage.get_active_alerts()) == 1


def test_ack_and_snooze_update_redis_record_and_active_index(storage):
    assert storage._try_redis_store_alert(alert())[0]
    assert storage.snooze_alert("new", 30)
    assert storage.get_active_alerts() == []
    assert storage.acknowledge_alert("new", "reviewer")
    assert not storage.redis_client.sismember(storage.ACTIVE_ALERTS_SET, "new")
    raw = storage.redis_client.hgetall(storage.ALERTS_HASH_PREFIX + "new")
    assert raw["acknowledged_by"] == "reviewer"
    assert storage._update_alert_field("new", {"acknowledged_at": None, "snooze_until": None})
    assert storage.redis_client.sismember(storage.ACTIVE_ALERTS_SET, "new")
    assert len(storage.get_active_alerts()) == 1
    assert not storage.acknowledge_alert("missing", "reviewer")


def test_one_bad_record_does_not_hide_valid_alerts(storage):
    assert storage._try_redis_store_alert(alert())[0]
    raw = storage.redis_client.hgetall(storage.ALERTS_HASH_PREFIX + "new")
    raw.update(id="bad", created_at="invalid-date")
    storage.redis_client.hset(storage.ALERTS_HASH_PREFIX + "bad", mapping=raw)
    storage.redis_client.sadd(storage.ACTIVE_ALERTS_SET, "bad")
    assert [a.id for a in storage.get_active_alerts()] == ["new"]


def test_inventory_is_bounded_read_only_and_does_not_export_alert_content(storage):
    from scripts.ops.inspect_alert_redis_nulls import inspect_alerts
    assert storage._try_redis_store_alert(alert())[0]
    key = storage.ALERTS_HASH_PREFIX + "new"
    storage.redis_client.hset(key, "acknowledged_at", "userdata: 0x7f123abc")
    before = storage.redis_client.hgetall(key)
    report = inspect_alerts(storage.redis_client, data_prefix=storage.ALERTS_HASH_PREFIX,
                           timeline=storage.ALERTS_ZSET, active=storage.ACTIVE_ALERTS_SET, limit=1)
    assert report["records_examined"] == 1
    assert report["legacy_null_fields_in_sample"] == {"acknowledged_at": 1}
    assert report["timeline_entries"] == 1
    assert storage.redis_client.hgetall(key) == before
    assert "0x7f123abc" not in json.dumps(report)
    assert "current_value" not in json.dumps(report)


def test_literal_null_username_is_preserved(storage):
    assert storage._try_redis_store_alert(alert(applied_by="null"))[0]
    assert storage.get_active_alerts()[0].applied_by == "null"


def test_fully_malformed_redis_result_triggers_degraded_fallback(storage):
    assert storage._try_redis_store_alert(alert())[0]
    storage.redis_client.hset(storage.ALERTS_HASH_PREFIX + "new", "created_at", "invalid")
    alerts, reason = storage._try_redis_get_active_alerts()
    assert alerts is None and reason == "redis_decode_error"
    assert storage._degraded_metrics["redis_failures"] == 1
