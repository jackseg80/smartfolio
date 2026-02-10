"""Tests for services/notifications/notification_sender.py"""

from __future__ import annotations
import json, logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from services.notifications.alert_manager import Alert, AlertLevel, AlertType
from services.notifications.notification_sender import (
    ConsoleNotifier, EmailNotifier, NotificationConfig, NotificationSender, WebhookNotifier,
    TelegramNotifier, convert_engine_alert, SEVERITY_TO_LEVEL,
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
    async def test_webhook_send_success_async(self):
        """Webhook now uses httpx.AsyncClient."""
        mock_response = MagicMock(status_code=200, text="OK")
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await self.notifier.send(
                self.alert, {"webhook_url": "https://h.com/x", "webhook_type": "generic"}
            )
        assert result is True
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_send_failure_async(self):
        mock_response = MagicMock(status_code=500, text="Error")
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await self.notifier.send(
                self.alert, {"webhook_url": "https://h.com/x", "webhook_type": "discord"}
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_webhook_send_exception_async(self):
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await self.notifier.send(
                self.alert, {"webhook_url": "https://h.com/x"}
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_webhook_discord_204_is_success(self):
        """Discord returns 204 No Content on success."""
        mock_response = MagicMock(status_code=204, text="")
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await self.notifier.send(
                self.alert, {"webhook_url": "https://discord.com/api/webhooks/123", "webhook_type": "discord"}
            )
        assert result is True


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

    def test_has_telegram_channel(self):
        sender = NotificationSender()
        assert "telegram" in sender.channels
        assert isinstance(sender.channels["telegram"], TelegramNotifier)

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


# ─── TelegramNotifier ────────────────────────────────────────────────


class TestTelegramNotifier:

    @pytest.mark.asyncio
    async def test_missing_config_returns_false(self):
        notifier = TelegramNotifier()
        result = await notifier.send(_make_alert(), {})
        assert result is False

    @pytest.mark.asyncio
    async def test_missing_chat_id(self):
        notifier = TelegramNotifier()
        result = await notifier.send(_make_alert(), {"bot_token": "123"})
        assert result is False

    @pytest.mark.asyncio
    async def test_missing_bot_token(self):
        notifier = TelegramNotifier()
        result = await notifier.send(_make_alert(), {"chat_id": "456"})
        assert result is False

    @pytest.mark.asyncio
    async def test_successful_send(self):
        notifier = TelegramNotifier()
        config = {"chat_id": "123456", "bot_token": "FAKE_TOKEN"}

        mock_response = MagicMock(status_code=200)
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await notifier.send(_make_alert(), config)
        assert result is True
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_error_returns_false(self):
        notifier = TelegramNotifier()
        config = {"chat_id": "123456", "bot_token": "FAKE_TOKEN"}

        mock_response = MagicMock(status_code=401, text="Unauthorized")
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await notifier.send(_make_alert(), config)
        assert result is False

    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        import httpx
        notifier = TelegramNotifier()
        config = {"chat_id": "123456", "bot_token": "FAKE_TOKEN"}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await notifier.send(_make_alert(), config)
        assert result is False

    def test_format_text_contains_title(self):
        notifier = TelegramNotifier()
        text = notifier._format_telegram_text(_make_alert(title="My Title"))
        assert "My Title" in text

    def test_format_text_contains_message(self):
        notifier = TelegramNotifier()
        text = notifier._format_telegram_text(_make_alert(message="Action required"))
        assert "Action required" in text

    def test_format_text_with_bridge_data(self):
        notifier = TelegramNotifier()
        alert = Alert(
            level=AlertLevel.WARNING,
            title="[S2] VOL_Q90_CROSS",
            message="Reduction exposition",
            data={
                "details": "Volatilite 5.2% tres elevee",
                "reasons": "Volatilite critique; Risque drawdown",
                "severity": "S2",
                "metric": "vol_90d",
            },
        )
        text = notifier._format_telegram_text(alert)
        assert "Reasons" in text
        assert "Details" in text
        assert "metric" in text


# ─── Bridge: convert_engine_alert ─────────────────────────────────────


class TestConvertEngineAlert:

    def _make_engine_alert(self, severity="S2", alert_type="VOL_Q90_CROSS"):
        alert = MagicMock()
        alert.id = "ALR-20260210-abc12345"

        sev_mock = MagicMock()
        sev_mock.value = severity
        alert.severity = sev_mock

        type_mock = MagicMock()
        type_mock.value = alert_type
        alert.alert_type = type_mock

        alert.data = {"current_vol": 0.052, "threshold": 0.04}
        alert.suggested_action = {"type": "slow", "ttl_minutes": 360}
        alert.format_unified_message.return_value = {
            "action": "Reduction exposition (mode Slow)",
            "impact": "2.0% estimated",
            "reasons": ["Volatilite critique", "Risque drawdown"],
            "details": "Volatilite 5.2% tres elevee. Mode Slow recommande.",
        }
        return alert

    def test_converts_s1_to_info(self):
        converted = convert_engine_alert(self._make_engine_alert(severity="S1"))
        assert converted.level == AlertLevel.INFO

    def test_converts_s2_to_warning(self):
        converted = convert_engine_alert(self._make_engine_alert(severity="S2"))
        assert converted.level == AlertLevel.WARNING

    def test_converts_s3_to_critical(self):
        converted = convert_engine_alert(self._make_engine_alert(severity="S3"))
        assert converted.level == AlertLevel.CRITICAL

    def test_preserves_alert_id(self):
        converted = convert_engine_alert(self._make_engine_alert())
        assert converted.id == "ALR-20260210-abc12345"

    def test_title_includes_severity_and_type(self):
        converted = convert_engine_alert(self._make_engine_alert(severity="S2", alert_type="REGIME_FLIP"))
        assert "S2" in converted.title
        assert "REGIME_FLIP" in converted.title

    def test_message_contains_action(self):
        converted = convert_engine_alert(self._make_engine_alert())
        assert "Reduction" in converted.message

    def test_data_includes_details_and_reasons(self):
        converted = convert_engine_alert(self._make_engine_alert())
        assert "details" in converted.data
        assert "reasons" in converted.data

    def test_data_preserves_original(self):
        converted = convert_engine_alert(self._make_engine_alert())
        assert converted.data["current_vol"] == 0.052

    def test_handles_format_error(self):
        alert = self._make_engine_alert()
        alert.format_unified_message.side_effect = Exception("format error")
        converted = convert_engine_alert(alert)
        assert converted.id == "ALR-20260210-abc12345"
        assert converted.message == "VOL_Q90_CROSS"


# ─── send_engine_alert integration ────────────────────────────────────


class TestSendEngineAlert:

    @pytest.mark.asyncio
    async def test_send_engine_alert_calls_bridge(self):
        sender = NotificationSender()
        engine_alert = MagicMock()
        engine_alert.id = "ALR-test"
        sev = MagicMock()
        sev.value = "S2"
        engine_alert.severity = sev
        atype = MagicMock()
        atype.value = "VOL_Q90_CROSS"
        engine_alert.alert_type = atype
        engine_alert.data = {}
        engine_alert.suggested_action = {}
        engine_alert.format_unified_message.return_value = {
            "action": "Test action",
            "impact": "1%",
            "reasons": ["Test"],
            "details": "Test details",
        }
        results = await sender.send_engine_alert(engine_alert)
        assert "console" in results


# ─── SEVERITY_TO_LEVEL mapping ───────────────────────────────────────


class TestSeverityMapping:

    def test_s1_maps_to_info(self):
        assert SEVERITY_TO_LEVEL["S1"] == AlertLevel.INFO

    def test_s2_maps_to_warning(self):
        assert SEVERITY_TO_LEVEL["S2"] == AlertLevel.WARNING

    def test_s3_maps_to_critical(self):
        assert SEVERITY_TO_LEVEL["S3"] == AlertLevel.CRITICAL

    def test_all_severities_mapped(self):
        assert len(SEVERITY_TO_LEVEL) == 3
