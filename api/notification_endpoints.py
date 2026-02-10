"""
Notification Endpoints — Configuration and test notification channels.

Endpoints:
- GET  /api/notifications/config   — Get user notification configuration
- POST /api/notifications/config   — Update notification config
- POST /api/notifications/test     — Send a test notification
- GET  /api/notifications/status   — Get notification system status
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.deps import get_required_user
from api.utils import success_response, error_response

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/notifications",
    tags=["notifications"],
)


class ChannelConfigUpdate(BaseModel):
    """Update for a single notification channel."""
    enabled: bool = Field(..., description="Enable/disable this channel")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Channel-specific config (chat_id, bot_token, webhook_url, etc.)"
    )


class NotificationConfigUpdate(BaseModel):
    """Update notification configuration for a user."""
    enabled: Optional[bool] = Field(None, description="Enable/disable all notifications")
    channels: Optional[Dict[str, ChannelConfigUpdate]] = Field(
        None, description="Channel configurations (telegram, discord, email, slack)"
    )


def _load_user_config(user_id: str) -> Dict[str, Any]:
    """Load user config.json."""
    path = Path(f"data/users/{user_id}/config.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_user_config(user_id: str, config: Dict[str, Any]) -> None:
    """Save user config.json (atomic write)."""
    import os
    import tempfile

    path = Path(f"data/users/{user_id}/config.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


@router.get("/config")
async def get_notification_config(user: str = Depends(get_required_user)):
    """
    Get notification configuration for the current user.

    Returns:
        dict: Notification settings with channel configs
    """
    user_config = _load_user_config(user)
    notif_config = user_config.get("notifications", {
        "enabled": False,
        "channels": {}
    })
    return success_response(notif_config)


@router.post("/config")
async def update_notification_config(
    update: NotificationConfigUpdate,
    user: str = Depends(get_required_user),
):
    """
    Update notification configuration for the current user.

    Merges with existing config — only provided fields are updated.
    """
    user_config = _load_user_config(user)
    notif = user_config.setdefault("notifications", {"enabled": False, "channels": {}})

    if update.enabled is not None:
        notif["enabled"] = update.enabled

    if update.channels:
        existing_channels = notif.setdefault("channels", {})
        for channel_name, ch_update in update.channels.items():
            existing_channels[channel_name] = {
                "enabled": ch_update.enabled,
                **ch_update.config,
            }

    user_config["notifications"] = notif
    _save_user_config(user, user_config)

    logger.info(f"Notification config updated for user '{user}'")
    return success_response(notif)


@router.post("/test")
async def test_notification(
    channel: str = Query(..., description="Channel to test: telegram, discord, webhook, email"),
    user: str = Depends(get_required_user),
):
    """
    Send a test notification through a specific channel.

    Requires channel to be configured in user's notification config.
    """
    from services.notifications.notification_sender import notification_sender, NotificationConfig
    from services.notifications.alert_manager import Alert, AlertLevel, AlertType

    # Load user's channel config
    user_config = _load_user_config(user)
    notif = user_config.get("notifications", {})
    channels = notif.get("channels", {})
    channel_config = channels.get(channel, {})

    if not channel_config:
        return error_response(
            f"Channel '{channel}' not configured. Update /api/notifications/config first.",
            code=400,
        )

    # Create test alert
    test_alert = Alert(
        type=AlertType.SYSTEM_ERROR,
        level=AlertLevel.INFO,
        source="test",
        title="SmartFolio Test Notification",
        message=f"This is a test notification from SmartFolio for user '{user}'. If you see this, the channel is working correctly.",
        data={"test": True, "user": user, "channel": channel},
        actions=["No action required — this is a test"],
    )

    # Map channel type to notifier config
    config_for_send = dict(channel_config)
    config_for_send.pop("enabled", None)

    # For webhook/discord, map properly
    if channel == "discord":
        config_for_send["webhook_type"] = "discord"
        config_for_send.setdefault("webhook_url", config_for_send.pop("url", None))

    notifier = notification_sender.channels.get(channel)
    if not notifier:
        return error_response(f"Unknown channel: {channel}", code=400)

    try:
        success = await notifier.send(test_alert, config_for_send)
        if success:
            return success_response({"channel": channel, "sent": True})
        else:
            return error_response(f"Test notification failed via {channel}", code=502)
    except Exception as e:
        logger.error(f"Test notification error: {e}")
        return error_response(f"Error: {str(e)}", code=500)


@router.get("/status")
async def get_notification_status(user: str = Depends(get_required_user)):
    """
    Get notification system status and configured channels.
    """
    from services.notifications.notification_sender import notification_sender

    # System status
    system_status = notification_sender.get_config_status()

    # User config
    user_config = _load_user_config(user)
    notif = user_config.get("notifications", {})

    return success_response({
        "system": system_status,
        "user_config": {
            "enabled": notif.get("enabled", False),
            "channels_configured": list(notif.get("channels", {}).keys()),
        },
        "available_channels": list(notification_sender.channels.keys()),
    })
