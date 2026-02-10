"""
Morning Brief Endpoints — Daily summary API.

Endpoints:
- GET  /api/morning-brief          — Generate morning brief for current user
- GET  /api/morning-brief/latest   — Get cached latest brief (fast)
- POST /api/morning-brief/send     — Trigger notification send (Telegram/email)
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.deps import get_required_user
from api.utils import success_response, error_response

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/morning-brief",
    tags=["morning-brief"],
)

# In-memory cache for latest brief per user
_latest_briefs: dict[str, dict] = {}


@router.get("")
async def get_morning_brief(
    user: str = Depends(get_required_user),
    source: str = Query("cointracking", description="Data source"),
    force: bool = Query(False, description="Force regeneration (skip cache)"),
):
    """
    Generate a morning brief for the current user.

    Aggregates P&L, Decision Index, alerts, top movers and signals.
    Results are cached per user for 15 minutes.
    """
    from services.morning_brief_service import morning_brief_service
    from datetime import datetime, timedelta

    # Check cache (15 min)
    if not force and user in _latest_briefs:
        cached = _latest_briefs[user]
        cached_at = cached.get("generated_at", "")
        try:
            gen_time = datetime.fromisoformat(cached_at)
            if datetime.now() - gen_time < timedelta(minutes=15):
                return success_response(cached, meta={"cached": True})
        except (ValueError, TypeError):
            pass

    try:
        brief = await morning_brief_service.generate(user_id=user, source=source)
        _latest_briefs[user] = brief
        return success_response(brief, meta={"cached": False})
    except Exception as e:
        logger.error(f"Morning brief generation failed for {user}: {e}")
        return error_response(f"Failed to generate morning brief: {str(e)[:100]}", code=500)


@router.get("/latest")
async def get_latest_brief(user: str = Depends(get_required_user)):
    """
    Get the most recently cached morning brief for the user.

    Returns 404 if no brief has been generated yet this session.
    """
    cached = _latest_briefs.get(user)
    if not cached:
        return error_response("No morning brief available yet. Call GET /api/morning-brief first.", code=404)
    return success_response(cached, meta={"cached": True})


@router.post("/send")
async def send_morning_brief(
    user: str = Depends(get_required_user),
    channel: str = Query("telegram", description="Notification channel: telegram, webhook, email"),
    source: str = Query("cointracking", description="Data source"),
):
    """
    Generate and send the morning brief via a notification channel.

    Requires the channel to be configured in user's notification config.
    """
    from services.morning_brief_service import morning_brief_service

    try:
        brief = await morning_brief_service.generate(user_id=user, source=source)
        _latest_briefs[user] = brief
    except Exception as e:
        logger.error(f"Morning brief generation failed: {e}")
        return error_response(f"Brief generation failed: {str(e)[:100]}", code=500)

    # Format based on channel
    if channel == "telegram":
        message = morning_brief_service.format_telegram(brief)
    elif channel == "email":
        message = morning_brief_service.format_email_html(brief)
    else:
        message = morning_brief_service.format_telegram(brief)

    # Send via notification system
    try:
        from services.notifications.notification_sender import notification_sender
        from services.notifications.alert_manager import Alert, AlertLevel, AlertType

        notif_alert = Alert(
            type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.INFO,
            source="morning_brief",
            title="Morning Brief",
            message=message,
            data={"morning_brief": True, "user": user},
            actions=[],
        )

        # Load user channel config
        import json
        from pathlib import Path

        config_path = Path(f"data/users/{user}/config.json")
        channel_config = {}
        if config_path.exists():
            user_config = json.loads(config_path.read_text(encoding="utf-8"))
            channels = user_config.get("notifications", {}).get("channels", {})
            channel_config = channels.get(channel, {})

        if not channel_config:
            return error_response(
                f"Channel '{channel}' not configured. Update /api/notifications/config first.",
                code=400,
            )

        config_for_send = dict(channel_config)
        config_for_send.pop("enabled", None)

        notifier = notification_sender.channels.get(channel)
        if not notifier:
            return error_response(f"Unknown channel: {channel}", code=400)

        success = await notifier.send(notif_alert, config_for_send)
        if success:
            return success_response({"channel": channel, "sent": True})
        else:
            return error_response(f"Send failed via {channel}", code=502)

    except Exception as e:
        logger.error(f"Morning brief send error: {e}")
        return error_response(f"Send error: {str(e)[:100]}", code=500)
