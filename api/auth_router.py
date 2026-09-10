"""Authentication routes with a staged legacy, dual and cookie migration."""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from jose import JWTError, jwt

from api.auth_security import (
    ACCESS_COOKIE,
    ACCESS_TOKEN_MINUTES,
    CSRF_COOKIE,
    LEGACY_TOKEN_DAYS,
    REFRESH_COOKIE,
    clear_login_failures,
    clear_session_cookies,
    create_refresh_session,
    ensure_login_allowed,
    get_auth_mode,
    get_jwt_secret,
    is_access_token_revoked,
    record_login_failure,
    revoke_access_token,
    revoke_all_user_sessions,
    revoke_refresh_session,
    rotate_refresh_session,
    set_session_cookies,
    validate_csrf,
)
from api.config.users import (
    get_user_info,
    is_allowed_user,
    update_user_password,
)
from api.deps import get_current_user_jwt
from api.utils import error_response, success_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])
ALGORITHM = "HS256"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception as exc:
        logger.error("Password verification error: %s", exc)
        return False


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
    *,
    session_id: Optional[str] = None,
) -> str:
    mode = get_auth_mode()
    default_delta = (
        timedelta(days=LEGACY_TOKEN_DAYS)
        if mode == "legacy"
        else timedelta(minutes=ACCESS_TOKEN_MINUTES)
    )
    now = datetime.now(timezone.utc)
    payload = {
        **data,
        "iat": now,
        "exp": now + (expires_delta or default_delta),
        "jti": secrets.token_urlsafe(24),
    }
    if session_id:
        payload["sid"] = session_id
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
        if is_access_token_revoked(payload):
            return None
        return payload
    except (JWTError, RuntimeError) as exc:
        logger.debug("JWT decode error: %s", exc)
        return None


def _user_payload(user_info: dict) -> dict:
    return {
        "id": user_info.get("id"),
        "label": user_info.get("label"),
        "roles": user_info.get("roles", []),
    }


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    username = username.lower().strip()
    client_ip = _client_ip(request)
    ensure_login_allowed(username, client_ip)

    user_info = get_user_info(username) if is_allowed_user(username) else None
    if (
        not user_info
        or user_info.get("status") != "active"
        or not user_info.get("password_hash")
        or not verify_password(password, user_info["password_hash"])
    ):
        record_login_failure(username, client_ip)
        logger.warning("Failed login for user=%s ip=%s", username, client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    clear_login_failures(username, client_ip)
    roles = list(user_info.get("roles", []))
    mode = get_auth_mode()
    refresh_token = None
    csrf_token = None
    session_id = None
    if mode in {"dual", "cookie"}:
        refresh_token, session_id = create_refresh_session(username, roles)
        csrf_token = secrets.token_urlsafe(32)

    access_token = create_access_token(
        {"sub": username, "roles": roles, "label": user_info.get("label", username)},
        session_id=session_id,
    )
    logger.info("Successful login for user=%s mode=%s", username, mode)
    result = {
        "expires_in": (
            LEGACY_TOKEN_DAYS * 24 * 60 * 60
            if mode == "legacy"
            else ACCESS_TOKEN_MINUTES * 60
        ),
        "user": _user_payload(user_info),
    }
    if mode in {"legacy", "dual"}:
        result.update({"token": access_token, "token_type": "bearer"})
    response = success_response(result)
    if refresh_token and csrf_token:
        set_session_cookies(response, access_token, refresh_token, csrf_token)
    return response


@router.post("/refresh")
async def refresh(
    request: Request,
    refresh_token: Optional[str] = Cookie(None, alias=REFRESH_COOKIE),
):
    if get_auth_mode() == "legacy":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    validate_csrf(request)
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")

    replacement, session = rotate_refresh_session(refresh_token)
    user_id = session.get("user_id")
    user_info = get_user_info(user_id) if user_id else None
    if not user_info or user_info.get("status") != "active":
        revoke_refresh_session(replacement)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive")

    roles = list(user_info.get("roles", []))
    access_token = create_access_token(
        {"sub": user_id, "roles": roles, "label": user_info.get("label", user_id)},
        session_id=session.get("session_id"),
    )
    csrf_token = secrets.token_urlsafe(32)
    result = {"expires_in": ACCESS_TOKEN_MINUTES * 60, "user": _user_payload(user_info)}
    # Dual mode still supports legacy JavaScript callers. Keep their bearer in
    # sync with the rotated cookie session until the migration reaches cookie mode.
    if get_auth_mode() == "dual":
        result.update({"token": access_token, "token_type": "bearer"})
    response = success_response(result)
    set_session_cookies(response, access_token, replacement, csrf_token)
    return response


@router.post("/logout")
async def logout(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    access_cookie: Optional[str] = Cookie(None, alias=ACCESS_COOKIE),
    refresh_cookie: Optional[str] = Cookie(None, alias=REFRESH_COOKIE),
):
    token = access_cookie
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(None, 1)[1]
    revoke_access_token(decode_access_token(token) if token else None)
    revoke_refresh_session(refresh_cookie)
    response = success_response({"message": "Logged out successfully"})
    clear_session_cookies(response)
    return response


@router.get("/session")
async def get_session(user_id: str = Depends(get_current_user_jwt)):
    user_info = get_user_info(user_id)
    if not user_info:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return success_response({"authenticated": True, "user": _user_payload(user_info)})


@router.get("/verify")
async def verify_token(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    access_cookie: Optional[str] = Cookie(None, alias=ACCESS_COOKIE),
    token: Optional[str] = None,
):
    mode = get_auth_mode()
    if mode == "cookie" and token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query-string tokens are not accepted",
        )
    candidate = access_cookie
    if authorization and authorization.lower().startswith("bearer "):
        candidate = authorization.split(None, 1)[1]
    if mode != "cookie" and token:
        candidate = token
    payload = decode_access_token(candidate) if candidate else None
    if not payload:
        return error_response("Invalid or expired token", code=401)
    return success_response(
        {
            "valid": True,
            "user_id": payload.get("sub"),
            "roles": payload.get("roles", []),
            "expires_at": datetime.fromtimestamp(
                payload["exp"], timezone.utc
            ).isoformat(),
        }
    )


@router.post("/change-password")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    access_cookie: Optional[str] = Cookie(None, alias=ACCESS_COOKIE),
):
    from api.deps import resolve_authenticated_user

    user_id = resolve_authenticated_user(
        authorization=authorization,
        access_cookie=access_cookie,
        x_user=request.headers.get("X-User"),
    )
    if get_auth_mode() in {"dual", "cookie"}:
        validate_csrf(request)

    user_info = get_user_info(user_id)
    if not user_info or not verify_password(current_password, user_info.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )
    if len(new_password) < 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 12 characters",
        )
    update_user_password(user_id, get_password_hash(new_password))
    revoke_all_user_sessions(user_id)
    logger.info("Password changed and all sessions revoked for user=%s", user_id)
    return success_response({"message": "Password updated successfully"})
