"""Tests for services/notifications/notification_sender.py"""

from __future__ import annotations
import json, logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from services.notifications.alert_manager import Alert, AlertLevel, AlertType
from services.notifications.notification_sender import (
    ConsoleNotifier, EmailNotifier, NotificationConfig, NotificationSender, WebhookNotifier,
)


def _make_alert(level=AlertLevel.WARNING, alert_type=AlertType.PORTFOLIO_DRIFT,
    title="Test Alert", message="Something happened", data=None, actions=None, source="test"):
    return Alert(id="alert-001", level=level, type=alert_type, title=title,
        message=message, data=data or {}, actions=actions or [], source=source,
        created_at=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc))


class TestConsoleNotifier:

    @pytest.mark.asyncio
    async def test_send_info_level(self, caplog):
        notifier = ConsoleNotifier()
        alert = _make_alert(level=AlertLevel.INFO, title="INFO msg")
        with caplog.at_level(logging.INFO):
            result = await notifier.send(alert, {})
        assert result is True

    @pytest.mark.asyncio
    async def test_send_warning_level(self, caplog):
        notifier = ConsoleNotifier()
        alert = _make_alert(level=AlertLevel.WARNING, title="WARNING msg")
        with caplog.at_level(logging.WARNING):
            result = await notifier.send(alert, {})
        assert result is True

    @pytest.mark.asyncio
    async def test_send_error_level(self, caplog):
        notifier = ConsoleNotifier()
        alert = _make_alert(level=AlertLevel.ERROR, title="ERROR msg")
        with caplog.at_level(logging.ERROR):
            result = await notifier.send(alert, {})
        assert result is True

    @pytest.mark.asyncio
    async def test_send_critical_level(self, caplog):
        notifier = ConsoleNotifier()
        alert = _make_alert(level=AlertLevel.CRITICAL, title="CRITICAL msg")
        with caplog.at_level(logging.CRITICAL):
            result = await notifier.send(alert, {})
        assert result is True

    @pytest.mark.asyncio
    async def test_send_includes_message_body(self, caplog):
        notifier = ConsoleNotifier()
        alert = _make_alert(message="detailed body text")
        with caplog.at_level(logging.WARNING):
            await notifier.send(alert, {})
        assert "detailed body text" in caplog.text

    @pytest.mark.asyncio
    async def test_send_includes_data_json(self, caplog):
        notifier = ConsoleNotifier()
        alert = _make_alert(data={"drift_pct": 5.2})
        with caplog.at_level(logging.WARNING):
            await notifier.send(alert, {})
        assert "drift_pct" in caplog.text

    @pytest.mark.asyncio
    async def test_send_includes_actions(self, caplog):
        notifier = ConsoleNotifier()
        alert = _make_alert(actions=["rebalance", "check API"])
        with caplog.at_level(logging.WARNING):
            await notifier.send(alert, {})
        assert "rebalance" in caplog.text
        assert "check API" in caplog.text


class TestWebhookPayloads:

    def setup_method(self):
        self.notifier = WebhookNotifier()
        self.alert = _make_alert(level=AlertLevel.ERROR, title="High drift",
            message="Portfolio drifted 8pct", data={"drift_pct": 8.0, "threshold": 5.0},
            actions=["Rebalance now"])

    def test_discord_payload_structure(self):
        payload = self.notifier._format_discord_payload(self.alert)
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert embed["title"] == "[ERROR] High drift"
        assert embed["description"] == "Portfolio drifted 8pct"
        assert embed["color"] == 0xDC3545

    def test_discord_payload_fields_from_data(self):
        payload = self.notifier._format_discord_payload(self.alert)
        field_names = [f["name"] for f in payload["embeds"][0]["fields"]]
        assert "Drift Pct" in field_names
        assert "Threshold" in field_names

    def test_discord_limits_fields_to_5(self):
        alert = _make_alert(data={"k"+str(i): i for i in range(10)})
        payload = self.notifier._format_discord_payload(alert)
        data_fields = [f for f in payload["embeds"][0]["fields"] if f.get("inline", False)]
        assert len(data_fields) <= 5

    def test_discord_payload_actions_field(self):
        payload = self.notifier._format_discord_payload(self.alert)
        action_fields = [f for f in payload["embeds"][0]["fields"] if not f.get("inline", True)]
        assert len(action_fields) >= 1
        assert "Rebalance now" in action_fields[0]["value"]

    def test_slack_payload_structure(self):
        payload = self.notifier._format_slack_payload(self.alert)
        assert "attachments" in payload
        att = payload["attachments"][0]
        assert att["title"] == "[ERROR] High drift"
        assert att["color"] == "#dc3545"

    def test_slack_payload_fields(self):
        payload = self.notifier._format_slack_payload(self.alert)
        att = payload["attachments"][0]
        field_titles = [f["title"] for f in att["fields"]]
        assert "Drift Pct" in field_titles

    def test_generic_payload_structure(self):
        payload = self.notifier._format_generic_payload(self.alert)
        assert payload["alert_id"] == "alert-001"
        assert payload["level"] == "error"
        assert payload["type"] == "portfolio_drift"
        assert payload["title"] == "High drift"
        assert payload["data"]["drift_pct"] == 8.0
        assert payload["actions"] == ["Rebalance now"]

    @pytest.mark.asyncio
    async def test_webhook_send_no_url(self):
        result = await self.notifier.send(self.alert, {})
        assert result is False

    @pytest.mark.asyncio
    @patch("services.notifications.notification_sender.requests.post")
    async def test_webhook_send_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        result = await self.notifier.send(self.alert, {"webhook_url": "https://h.com/x", "webhook_type": "generic"})
        assert result is True
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch("services.notifications.notification_sender.requests.post")
    async def test_webhook_send_failure(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500, text="Error")
        result = await self.notifier.send(self.alert, {"webhook_url": "https://h.com/x", "webhook_type": "discord"})
        assert result is False

    @pytest.mark.asyncio
    @patch("services.notifications.notification_sender.requests.post", side_effect=Exception("timeout"))
    async def test_webhook_send_exception(self, mock_post):
        result = await self.notifier.send(self.alert, {"webhook_url": "https://h.com/x"})
        assert result is False

class TestEmailNotifier:

    def setup_method(self):
        self.notifier = EmailNotifier()
        self.alert = _make_alert(level=AlertLevel.CRITICAL, title="System down",
            message="API unreachable", data={"endpoint": "/api/ml", "status": 503},
            actions=["Restart server", "Check logs"], source="health_check")

    def test_format_email_body_html(self):
        body = self.notifier._format_email_body(self.alert)
        assert "<html>" in body
        assert "[CRITICAL]" in body
        assert "System down" in body
        assert "API unreachable" in body
        assert "health_check" in body

    def test_format_email_body_color(self):
        body = self.notifier._format_email_body(self.alert)
        assert "#343a40" in body

    def test_format_data_section(self):
        section = self.notifier._format_data_section({"key1": "val1", "key2": 42})
        assert "key1" in section and "val1" in section and "42" in section

    def test_format_actions_section(self):
        section = self.notifier._format_actions_section(["Do A", "Do B"])
        assert "Do A" in section and "Do B" in section

    @pytest.mark.asyncio
    async def test_send_no_recipients(self):
        result = await self.notifier.send(self.alert, {"to_emails": []})
        assert result is False

    @pytest.mark.asyncio
    async def test_send_email_exception(self):
        result = await self.notifier.send(self.alert, {"smtp_host": "invalid", "smtp_port": 9999, "to_emails": ["t@t.com"]})
        assert result is False


class TestNotificationConfig:

    def test_defaults(self):
        config = NotificationConfig(channel_type="console")
        assert config.enabled is True
        assert config.min_level == AlertLevel.INFO
        assert config.alert_types is None
        assert config.config is None

    def test_custom_config(self):
        config = NotificationConfig(channel_type="webhook", enabled=False,
            min_level=AlertLevel.ERROR, alert_types=["portfolio_drift"],
            config={"webhook_url": "https://example.com"})
        assert config.enabled is False
        assert config.min_level == AlertLevel.ERROR
        assert config.alert_types == ["portfolio_drift"]

class TestNotificationSender:

    def test_default_console_config(self):
        sender = NotificationSender()
        assert len(sender.configurations) == 1
        assert sender.configurations[0].channel_type == "console"

    def test_add_config(self):
        sender = NotificationSender()
        sender.add_config(NotificationConfig(channel_type="webhook"))
        assert len(sender.configurations) == 2

    def test_remove_config_existing(self):
        sender = NotificationSender()
        sender.add_config(NotificationConfig(channel_type="webhook"))
        assert sender.remove_config("webhook") is True
        assert len(sender.configurations) == 1

    def test_remove_config_nonexistent(self):
        sender = NotificationSender()
        assert sender.remove_config("sms") is False

    def test_should_send_min_level(self):
        sender = NotificationSender()
        config = NotificationConfig(channel_type="console", min_level=AlertLevel.ERROR)
        assert sender._should_send(_make_alert(level=AlertLevel.INFO), config) is False
        assert sender._should_send(_make_alert(level=AlertLevel.WARNING), config) is False
        assert sender._should_send(_make_alert(level=AlertLevel.ERROR), config) is True
        assert sender._should_send(_make_alert(level=AlertLevel.CRITICAL), config) is True

    def test_should_send_alert_types(self):
        sender = NotificationSender()
        config = NotificationConfig(channel_type="console", alert_types=["portfolio_drift", "execution_failure"])
        assert sender._should_send(_make_alert(alert_type=AlertType.PORTFOLIO_DRIFT), config) is True
        assert sender._should_send(_make_alert(alert_type=AlertType.API_CONNECTIVITY), config) is False

    def test_should_send_no_filter(self):
        sender = NotificationSender()
        config = NotificationConfig(channel_type="console", alert_types=None)
        assert sender._should_send(_make_alert(alert_type=AlertType.SYSTEM_ERROR), config) is True

    @pytest.mark.asyncio
    async def test_disabled_skipped(self):
        sender = NotificationSender()
        sender.configurations = [NotificationConfig(channel_type="console", enabled=False)]
        assert await sender.send_alert(_make_alert()) == {}

    @pytest.mark.asyncio
    async def test_unknown_channel(self):
        sender = NotificationSender()
        sender.configurations = [NotificationConfig(channel_type="pigeon", enabled=True)]
        results = await sender.send_alert(_make_alert())
        assert results.get("pigeon") is False

    @pytest.mark.asyncio
    async def test_console_success(self):
        sender = NotificationSender()
        results = await sender.send_alert(_make_alert(level=AlertLevel.WARNING))
        assert results.get("console") is True

    @pytest.mark.asyncio
    async def test_exception_in_notifier(self):
        sender = NotificationSender()
        mock_notifier = MagicMock()
        mock_notifier.send = AsyncMock(side_effect=Exception("boom"))
        sender.channels["console"] = mock_notifier
        results = await sender.send_alert(_make_alert())
        assert results.get("console") is False

    @pytest.mark.asyncio
    async def test_multiple_channels(self):
        sender = NotificationSender()
        mock_c = MagicMock()
        mock_c.send = AsyncMock(return_value=True)
        mock_w = MagicMock()
        mock_w.send = AsyncMock(return_value=True)
        sender.channels["console"] = mock_c
        sender.channels["webhook"] = mock_w
        sender.configurations = [
            NotificationConfig(channel_type="console", enabled=True),
            NotificationConfig(channel_type="webhook", enabled=True, config={"webhook_url": "http://x"}),
        ]
        results = await sender.send_alert(_make_alert())
        assert results["console"] is True
        assert results["webhook"] is True

    def test_get_config_status(self):
        sender = NotificationSender()
        sender.add_config(NotificationConfig(channel_type="webhook", enabled=False))
        status = sender.get_config_status()
        assert status["total_configs"] == 2
        assert status["enabled_configs"] == 1
        assert "console" in status["channels"]
        assert "webhook" in status["channels"]

    @pytest.mark.asyncio
    async def test_filtered_by_level(self):
        sender = NotificationSender()
        sender.configurations = [NotificationConfig(channel_type="console", enabled=True, min_level=AlertLevel.CRITICAL)]
        assert await sender.send_alert(_make_alert(level=AlertLevel.INFO)) == {}

    @pytest.mark.asyncio
    async def test_filtered_by_type(self):
        sender = NotificationSender()
        sender.configurations = [NotificationConfig(channel_type="console", enabled=True, alert_types=["execution_failure"])]
        assert await sender.send_alert(_make_alert(alert_type=AlertType.PORTFOLIO_DRIFT)) == {}


class TestWebhookColorMaps:

    def test_discord_colors_all_levels(self):
        notifier = WebhookNotifier()
        for level in AlertLevel:
            payload = notifier._format_discord_payload(_make_alert(level=level))
            assert isinstance(payload["embeds"][0]["color"], int)

    def test_slack_colors_all_levels(self):
        notifier = WebhookNotifier()
        for level in AlertLevel:
            payload = notifier._format_slack_payload(_make_alert(level=level))
            c = payload["attachments"][0]["color"]
            assert isinstance(c, str) and c.startswith("#")
