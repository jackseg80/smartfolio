"""Authentication primitives shared by routes, dependencies and middleware."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import HTTPException, Request, Response, status

logger = logging.getLogger(__name__)

AUTH_MODES = {"legacy", "dual", "cookie"}
ACCESS_COOKIE = "smartfolio_access"
REFRESH_COOKIE = "smartfolio_refresh"
CSRF_COOKIE = "smartfolio_csrf"


def _read_duration_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded positive authentication duration at process startup."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


# A short-lived access token limits exposure. The refresh session controls how
# long the user can remain signed in and is renewed during active use.
ACCESS_TOKEN_MINUTES = _read_duration_setting(
    "AUTH_ACCESS_TOKEN_MINUTES", 15, 5, 1440
)
REFRESH_TOKEN_DAYS = _read_duration_setting("AUTH_SESSION_DAYS", 7, 1, 90)
LEGACY_TOKEN_DAYS = 7
DEFAULT_INSECURE_SECRET = "your-secret-key-change-in-production-please"
SESSION_TTL_SECONDS = int(timedelta(days=REFRESH_TOKEN_DAYS).total_seconds())


def get_auth_mode() -> str:
    mode = os.environ.get("AUTH_MODE", "legacy").strip().lower()
    if mode not in AUTH_MODES:
        raise RuntimeError(f"Invalid AUTH_MODE: {mode}")
    return mode


def get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", DEFAULT_INSECURE_SECRET)
    production = os.getenv("ENVIRONMENT", "development").strip().lower() == "production"
    if production and (not secret or secret == DEFAULT_INSECURE_SECRET or len(secret) < 32):
        raise RuntimeError("JWT_SECRET_KEY must be a random value of at least 32 characters in production")
    return secret


def get_redis_client():
    """Return the shared Redis client used for sessions and login throttling."""
    try:
        import redis

        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0").strip()
        if redis_url.lower() in {"", "disabled", "none", "off"}:
            logger.debug("Redis is explicitly disabled")
            return None
        return redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    except Exception as exc:
        logger.error("Redis client initialization failed: %s", exc)
        return None


def require_session_redis():
    client = get_redis_client()
    try:
        if client is None or not client.ping():
            raise RuntimeError("Redis unavailable")
    except Exception as exc:
        logger.error("Redis is required for secure sessions: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Secure session storage is temporarily unavailable",
        ) from exc
    return client


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_refresh_session(user_id: str, roles: list[str]) -> tuple[str, str]:
    """Create an opaque, Redis-backed refresh token and return token + session id."""
    client = require_session_redis()
    token = secrets.token_urlsafe(48)
    session_id = secrets.token_urlsafe(24)
    payload = {
        "user_id": user_id,
        "roles": roles,
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    token_hash = hash_token(token)
    pipeline = client.pipeline()
    pipeline.setex(f"auth:refresh:{token_hash}", SESSION_TTL_SECONDS, json.dumps(payload))
    pipeline.setex(
        f"auth:session:{session_id}",
        SESSION_TTL_SECONDS,
        json.dumps({"user_id": user_id, "refresh_hash": token_hash}),
    )
    pipeline.sadd(f"auth:user-sessions:{user_id}", session_id)
    pipeline.expire(f"auth:user-sessions:{user_id}", SESSION_TTL_SECONDS)
    pipeline.execute()
    return token, session_id


def rotate_refresh_session(token: str) -> tuple[str, dict[str, Any]]:
    """Atomically consume a refresh token and issue a replacement."""
    client = require_session_redis()
    key = f"auth:refresh:{hash_token(token)}"
    pipeline = client.pipeline()
    while True:
        try:
            pipeline.watch(key)
            raw = pipeline.get(key)
            if not raw:
                pipeline.unwatch()
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or replayed refresh token",
                )
            session = json.loads(raw)
            new_token = secrets.token_urlsafe(48)
            new_hash = hash_token(new_token)
            new_key = f"auth:refresh:{new_hash}"
            session_id = session.get("session_id")
            pipeline.multi()
            pipeline.delete(key)
            pipeline.setex(
                new_key,
                SESSION_TTL_SECONDS,
                json.dumps(session),
            )
            if session_id:
                pipeline.setex(
                    f"auth:session:{session_id}",
                    SESSION_TTL_SECONDS,
                    json.dumps(
                        {
                            "user_id": session.get("user_id"),
                            "refresh_hash": new_hash,
                        }
                    ),
                )
                pipeline.expire(
                    f"auth:user-sessions:{session.get('user_id')}",
                    SESSION_TTL_SECONDS,
                )
            pipeline.execute()
            return new_token, session
        except HTTPException:
            raise
        except Exception as exc:
            try:
                pipeline.reset()
            except Exception:
                pass
            if exc.__class__.__name__ == "WatchError":
                continue
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Secure session storage is temporarily unavailable",
            ) from exc


def revoke_refresh_session(token: Optional[str]) -> None:
    if not token:
        return
    client = get_redis_client()
    try:
        if client is not None:
            refresh_key = f"auth:refresh:{hash_token(token)}"
            raw = client.get(refresh_key)
            session = json.loads(raw) if raw else {}
            session_id = session.get("session_id")
            user_id = session.get("user_id")
            pipeline = client.pipeline()
            pipeline.delete(refresh_key)
            if session_id:
                pipeline.delete(f"auth:session:{session_id}")
            if session_id and user_id:
                pipeline.srem(f"auth:user-sessions:{user_id}", session_id)
            pipeline.execute()
    except Exception as exc:
        logger.warning("Unable to revoke refresh session: %s", exc)


def revoke_all_user_sessions(user_id: str) -> None:
    """Revoke every access and refresh session for a user."""
    client = require_session_redis()
    index_key = f"auth:user-sessions:{user_id}"
    session_ids = client.smembers(index_key)
    pipeline = client.pipeline()
    for session_id in session_ids:
        raw = client.get(f"auth:session:{session_id}")
        session = json.loads(raw) if raw else {}
        refresh_hash = session.get("refresh_hash")
        if refresh_hash:
            pipeline.delete(f"auth:refresh:{refresh_hash}")
        pipeline.delete(f"auth:session:{session_id}")
    pipeline.delete(index_key)
    pipeline.execute()


def revoke_access_token(payload: Optional[dict[str, Any]]) -> None:
    if not payload or not payload.get("jti"):
        return
    exp = int(payload.get("exp", 0))
    ttl = max(1, exp - int(datetime.now(timezone.utc).timestamp()))
    client = get_redis_client()
    try:
        if client is not None:
            client.setex(f"auth:revoked:{payload['jti']}", ttl, "1")
    except Exception as exc:
        logger.warning("Unable to revoke access token: %s", exc)


def is_access_token_revoked(payload: dict[str, Any]) -> bool:
    jti = payload.get("jti")
    if not jti:
        return False
    client = get_redis_client()
    try:
        if not client:
            return False
        if client.exists(f"auth:revoked:{jti}"):
            return True
        session_id = payload.get("sid")
        if session_id and get_auth_mode() in {"dual", "cookie"}:
            return not bool(client.exists(f"auth:session:{session_id}"))
        return False
    except Exception as exc:
        logger.warning("Unable to check token revocation: %s", exc)
        return get_auth_mode() == "cookie"


def record_login_failure(username: str, client_ip: str) -> None:
    client = get_redis_client()
    if client is None:
        return
    key = f"auth:login-fail:{hash_token(username + '|' + client_ip)}"
    try:
        count = client.incr(key)
        if count == 1:
            client.expire(key, 15 * 60)
    except Exception as exc:
        logger.warning("Unable to record login failure: %s", exc)


def clear_login_failures(username: str, client_ip: str) -> None:
    client = get_redis_client()
    if client is None:
        return
    try:
        client.delete(f"auth:login-fail:{hash_token(username + '|' + client_ip)}")
    except Exception as exc:
        logger.warning("Unable to clear login failures: %s", exc)


def ensure_login_allowed(username: str, client_ip: str) -> None:
    client = get_redis_client()
    if client is None:
        if get_auth_mode() in {"dual", "cookie"}:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Login protection is temporarily unavailable",
            )
        return
    try:
        count = int(client.get(f"auth:login-fail:{hash_token(username + '|' + client_ip)}") or 0)
        if count >= 5:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Try again later.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        if get_auth_mode() in {"dual", "cookie"}:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Login protection is temporarily unavailable",
            ) from exc


def set_session_cookies(response: Response, access_token: str, refresh_token: str, csrf_token: str) -> None:
    common = {"secure": True, "samesite": "strict", "path": "/"}
    response.set_cookie(
        ACCESS_COOKIE,
        access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_MINUTES * 60,
        **common,
    )
    response.set_cookie(
        REFRESH_COOKIE,
        refresh_token,
        httponly=True,
        max_age=REFRESH_TOKEN_DAYS * 24 * 60 * 60,
        **common,
    )
    response.set_cookie(
        CSRF_COOKIE,
        csrf_token,
        httponly=False,
        max_age=REFRESH_TOKEN_DAYS * 24 * 60 * 60,
        **common,
    )


def clear_session_cookies(response: Response) -> None:
    for cookie_name in (ACCESS_COOKIE, REFRESH_COOKIE, CSRF_COOKIE):
        response.delete_cookie(cookie_name, path="/", secure=True, samesite="strict")


def validate_csrf(request: Request) -> None:
    cookie_token = request.cookies.get(CSRF_COOKIE)
    header_token = request.headers.get("X-CSRF-Token")
    if not cookie_token or not header_token or not secrets.compare_digest(cookie_token, header_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")


@dataclass(frozen=True)
class AuthenticatedUser:
    username: str
    roles: list[str]
