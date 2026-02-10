"""
Notification Sender - Systeme d'envoi de notifications

Ce module gere l'envoi des notifications via differents canaux :
- Email
- Webhooks (Discord, Slack, Teams)
- Telegram (Bot API)
- Console/Logs
- WebSockets pour UI temps reel
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass, field
import logging
import smtplib
import json
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback pour certaines versions Python
    MimeText = None
    MimeMultipart = None

from .alert_manager import Alert, AlertLevel

logger = logging.getLogger(__name__)

# Mapping severity (new system) -> AlertLevel (old system)
SEVERITY_TO_LEVEL = {
    "S1": AlertLevel.INFO,
    "S2": AlertLevel.WARNING,
    "S3": AlertLevel.CRITICAL,
}


def convert_engine_alert(engine_alert) -> Alert:
    """
    Bridge: convertit une Alert du nouveau systeme (alert_types.py)
    vers l'ancien format (alert_manager.py) utilise par les notifiers.
    """
    from services.alerts.alert_types import Alert as EngineAlert

    # Mapper severity S1/S2/S3 -> AlertLevel
    severity_str = engine_alert.severity.value if hasattr(engine_alert.severity, 'value') else str(engine_alert.severity)
    level = SEVERITY_TO_LEVEL.get(severity_str, AlertLevel.WARNING)

    # Extraire action/message depuis format_unified_message
    formatted = {}
    try:
        formatted = engine_alert.format_unified_message()
    except Exception:
        pass

    action_text = formatted.get("action", engine_alert.alert_type.value)
    details_text = formatted.get("details", "")
    reasons = formatted.get("reasons", [])
    if isinstance(reasons, list):
        reasons_text = "; ".join(reasons)
    else:
        reasons_text = str(reasons)

    return Alert(
        id=engine_alert.id,
        type=_map_alert_type(engine_alert.alert_type.value),
        level=level,
        source="alert_engine",
        title=f"[{severity_str}] {engine_alert.alert_type.value}",
        message=action_text,
        data={
            **engine_alert.data,
            "details": details_text,
            "reasons": reasons_text,
            "severity": severity_str,
            "suggested_action": engine_alert.suggested_action,
        },
        actions=[action_text] if action_text else [],
    )


def _map_alert_type(alert_type_value: str):
    """Mappe le type d'alerte du nouveau systeme vers l'ancien."""
    from .alert_manager import AlertType as OldAlertType
    # Legacy types ont le meme nom dans les deux systemes
    try:
        return OldAlertType(alert_type_value)
    except ValueError:
        # Nouveaux types (VOL_Q90_CROSS, etc.) -> THRESHOLD_BREACH comme fallback
        return OldAlertType.THRESHOLD_BREACH


class NotificationChannel(Protocol):
    """Interface pour les canaux de notification"""

    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Envoyer une notification"""
        ...

@dataclass
class NotificationConfig:
    """Configuration d'un canal de notification"""
    channel_type: str
    enabled: bool = True
    config: Dict[str, Any] = None

    # Filtres
    min_level: AlertLevel = AlertLevel.INFO
    alert_types: Optional[List[str]] = None

class ConsoleNotifier:
    """Notification via console/logs"""

    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Afficher dans la console"""

        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }

        emoji = level_emoji.get(alert.level, "📢")

        message = f"{emoji} [{alert.level.value.upper()}] {alert.title}"
        if alert.message:
            message += f"\n   {alert.message}"

        if alert.data:
            message += f"\n   Data: {json.dumps(alert.data, indent=2)}"

        if alert.actions:
            message += f"\n   Actions: {', '.join(alert.actions)}"

        # Log selon le niveau
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(message)
        elif alert.level == AlertLevel.ERROR:
            logger.error(message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(message)
        else:
            logger.info(message)

        return True

class EmailNotifier:
    """Notification par email"""

    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Envoyer par email"""

        if not MimeText or not MimeMultipart:
            logger.warning("Email functionality not available - missing email.mime imports")
            return False

        try:
            smtp_host = config.get("smtp_host", "localhost")
            smtp_port = config.get("smtp_port", 587)
            username = config.get("username")
            password = config.get("password")
            from_email = config.get("from_email", "crypto-rebalancer@localhost")
            to_emails = config.get("to_emails", [])

            if not to_emails:
                logger.warning("No email recipients configured")
                return False

            # Creer le message
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ", ".join(to_emails)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"

            # Corps du message
            body = self._format_email_body(alert)
            msg.attach(MimeText(body, 'html'))

            # Envoyer
            server = smtplib.SMTP(smtp_host, smtp_port)
            if username and password:
                server.starttls()
                server.login(username, password)

            server.send_message(msg)
            server.quit()

            logger.info(f"Email sent for alert {alert.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _format_email_body(self, alert: Alert) -> str:
        """Formater le corps de l'email en HTML"""

        level_colors = {
            AlertLevel.INFO: "#17a2b8",
            AlertLevel.WARNING: "#ffc107",
            AlertLevel.ERROR: "#dc3545",
            AlertLevel.CRITICAL: "#343a40"
        }

        color = level_colors.get(alert.level, "#6c757d")

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="border-left: 4px solid {color}; padding: 20px; margin: 10px;">
                <h2 style="color: {color}; margin-top: 0;">
                    [{alert.level.value.upper()}] {alert.title}
                </h2>

                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Heure:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

                {self._format_data_section(alert.data) if alert.data else ""}
                {self._format_actions_section(alert.actions) if alert.actions else ""}
            </div>
        </body>
        </html>
        """

        return html

    def _format_data_section(self, data: Dict[str, Any]) -> str:
        """Formater la section des donnees"""
        items = []
        for key, value in data.items():
            items.append(f"<li><strong>{key}:</strong> {value}</li>")

        return f"""
        <div style="margin-top: 15px;">
            <strong>Donnees detaillees:</strong>
            <ul>
                {"".join(items)}
            </ul>
        </div>
        """

    def _format_actions_section(self, actions: List[str]) -> str:
        """Formater la section des actions"""
        items = [f"<li>{action}</li>" for action in actions]

        return f"""
        <div style="margin-top: 15px;">
            <strong>Actions recommandees:</strong>
            <ul>
                {"".join(items)}
            </ul>
        </div>
        """


class TelegramNotifier:
    """Notification via Telegram Bot API (async httpx)"""

    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Envoyer une notification via Telegram Bot API."""
        import httpx

        chat_id = config.get("chat_id")
        bot_token = config.get("bot_token")
        timeout = config.get("timeout_seconds", 10)

        if not chat_id or not bot_token:
            logger.warning("Telegram config missing: chat_id or bot_token required")
            return False

        text = self._format_telegram_text(alert)
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)

            if response.status_code == 200:
                logger.info(f"Telegram notification sent for alert {alert.id}")
                return True
            else:
                logger.error(f"Telegram API error {response.status_code}: {response.text}")
                return False
        except httpx.TimeoutException:
            logger.error(f"Telegram timeout sending alert {alert.id}")
            return False
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

    def _format_telegram_text(self, alert: Alert) -> str:
        """Format alerte pour Telegram (HTML parse mode)."""
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }
        emoji = level_emoji.get(alert.level, "📢")

        # Extraire details et reasons si presents (bridge depuis engine alert)
        details = alert.data.get("details", "")
        reasons = alert.data.get("reasons", "")
        severity = alert.data.get("severity", alert.level.value.upper())

        lines = [
            f"{emoji} <b>{alert.title}</b>",
            "",
            f"<b>Action:</b> {alert.message}" if alert.message else "",
        ]

        if reasons:
            lines.append(f"<b>Reasons:</b> {reasons}")
        if details:
            lines.append(f"<b>Details:</b> {details}")

        # Ajouter quelques data cles (pas tout le dict)
        skip_keys = {"details", "reasons", "severity", "suggested_action"}
        extra = {k: v for k, v in alert.data.items() if k not in skip_keys}
        if extra:
            items = [f"  {k}: {v}" for k, v in list(extra.items())[:5]]
            lines.append("")
            lines.extend(items)

        return "\n".join(line for line in lines if line is not None)


class WebhookNotifier:
    """Notification via webhooks (Discord, Slack, etc.) — async httpx"""

    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Envoyer via webhook (async)"""
        import httpx

        try:
            webhook_url = config.get("webhook_url")
            webhook_type = config.get("webhook_type", "generic")
            timeout = config.get("timeout_seconds", 10)

            if not webhook_url:
                logger.warning("No webhook URL configured")
                return False

            # Formater selon le type de webhook
            if webhook_type == "discord":
                payload = self._format_discord_payload(alert)
            elif webhook_type == "slack":
                payload = self._format_slack_payload(alert)
            else:
                payload = self._format_generic_payload(alert)

            # Envoyer (async)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

            # Discord renvoie 204 No Content sur succes
            if response.status_code in (200, 204):
                logger.info(f"Webhook sent for alert {alert.id}")
                return True
            else:
                logger.error(f"Webhook failed: {response.status_code} - {response.text}")
                return False

        except httpx.TimeoutException:
            logger.error(f"Webhook timeout for alert {alert.id}")
            return False
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False

    def _format_discord_payload(self, alert: Alert) -> Dict[str, Any]:
        """Formater pour Discord"""

        color_map = {
            AlertLevel.INFO: 0x17a2b8,
            AlertLevel.WARNING: 0xffc107,
            AlertLevel.ERROR: 0xdc3545,
            AlertLevel.CRITICAL: 0x343a40
        }

        embed = {
            "title": f"[{alert.level.value.upper()}] {alert.title}",
            "description": alert.message,
            "color": color_map.get(alert.level, 0x6c757d),
            "timestamp": alert.created_at.isoformat(),
            "fields": []
        }

        # Ajouter les donnees importantes
        if alert.data:
            skip_keys = {"details", "reasons", "severity", "suggested_action"}
            for key, value in list(alert.data.items())[:5]:
                if key in skip_keys:
                    continue
                embed["fields"].append({
                    "name": key.replace("_", " ").title(),
                    "value": str(value)[:256],
                    "inline": True
                })

        # Details et reasons du bridge
        reasons = alert.data.get("reasons", "")
        if reasons:
            embed["fields"].append({
                "name": "Reasons",
                "value": str(reasons)[:1024],
                "inline": False
            })

        # Actions
        if alert.actions:
            actions_text = "\n".join(f"• {action}" for action in alert.actions[:3])
            embed["fields"].append({
                "name": "Recommended Actions",
                "value": actions_text,
                "inline": False
            })

        return {"embeds": [embed]}

    def _format_slack_payload(self, alert: Alert) -> Dict[str, Any]:
        """Formater pour Slack"""

        color_map = {
            AlertLevel.INFO: "#17a2b8",
            AlertLevel.WARNING: "#ffc107",
            AlertLevel.ERROR: "#dc3545",
            AlertLevel.CRITICAL: "#343a40"
        }

        attachment = {
            "color": color_map.get(alert.level, "#6c757d"),
            "title": f"[{alert.level.value.upper()}] {alert.title}",
            "text": alert.message,
            "ts": int(alert.created_at.timestamp()),
            "fields": []
        }

        # Donnees importantes
        if alert.data:
            for key, value in list(alert.data.items())[:5]:
                attachment["fields"].append({
                    "title": key.replace("_", " ").title(),
                    "value": str(value),
                    "short": True
                })

        return {"attachments": [attachment]}

    def _format_generic_payload(self, alert: Alert) -> Dict[str, Any]:
        """Formater generique"""
        return {
            "alert_id": alert.id,
            "level": alert.level.value,
            "type": alert.type.value,
            "title": alert.title,
            "message": alert.message,
            "source": alert.source,
            "timestamp": alert.created_at.isoformat(),
            "data": alert.data,
            "actions": alert.actions
        }

class NotificationSender:
    """Gestionnaire principal d'envoi de notifications"""

    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {
            "console": ConsoleNotifier(),
            "email": EmailNotifier(),
            "webhook": WebhookNotifier(),
            "telegram": TelegramNotifier(),
        }

        self.configurations: List[NotificationConfig] = []

        # Configuration par defaut (console)
        self.add_config(NotificationConfig(
            channel_type="console",
            enabled=True,
            min_level=AlertLevel.INFO
        ))

    def add_config(self, config: NotificationConfig) -> None:
        """Ajouter une configuration de notification"""
        self.configurations.append(config)
        logger.info(f"Notification config added: {config.channel_type}")

    def remove_config(self, channel_type: str) -> bool:
        """Supprimer une configuration"""
        original_count = len(self.configurations)
        self.configurations = [c for c in self.configurations if c.channel_type != channel_type]
        return len(self.configurations) < original_count

    async def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """Envoyer une alerte via tous les canaux configures"""

        results = {}

        for config in self.configurations:
            if not config.enabled:
                continue

            # Verifier les filtres
            if not self._should_send(alert, config):
                continue

            # Obtenir le notifier approprie
            notifier = self.channels.get(config.channel_type)
            if not notifier:
                logger.error(f"Unknown notification channel: {config.channel_type}")
                results[config.channel_type] = False
                continue

            # Envoyer
            try:
                success = await notifier.send(alert, config.config or {})
                results[config.channel_type] = success

                if success:
                    logger.debug(f"Alert {alert.id} sent via {config.channel_type}")
                else:
                    logger.warning(f"Failed to send alert {alert.id} via {config.channel_type}")

            except Exception as e:
                logger.error(f"Error sending alert via {config.channel_type}: {e}")
                results[config.channel_type] = False

        return results

    async def send_engine_alert(self, engine_alert) -> Dict[str, bool]:
        """
        Envoyer une alerte du nouveau systeme (alert_engine.py).
        Convertit automatiquement vers le format notification.
        """
        converted = convert_engine_alert(engine_alert)
        return await self.send_alert(converted)

    def _should_send(self, alert: Alert, config: NotificationConfig) -> bool:
        """Verifier si l'alerte doit etre envoyee selon les filtres"""

        # Filtre par niveau minimum
        level_order = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.ERROR: 2,
            AlertLevel.CRITICAL: 3
        }

        if level_order.get(alert.level, 0) < level_order.get(config.min_level, 0):
            return False

        # Filtre par types d'alertes
        if config.alert_types and alert.type.value not in config.alert_types:
            return False

        return True

    def get_config_status(self) -> Dict[str, Any]:
        """Obtenir le statut des configurations"""

        status = {
            "total_configs": len(self.configurations),
            "enabled_configs": sum(1 for c in self.configurations if c.enabled),
            "channels": {}
        }

        for config in self.configurations:
            channel_info = {
                "enabled": config.enabled,
                "min_level": config.min_level.value,
                "alert_types": config.alert_types
            }

            if config.channel_type not in status["channels"]:
                status["channels"][config.channel_type] = []

            status["channels"][config.channel_type].append(channel_info)

        return status

# Instance globale du sender
notification_sender = NotificationSender()
