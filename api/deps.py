"""
Dépendances FastAPI réutilisables.
Gestion des utilisateurs avec header X-User (legacy) ou JWT token (nouveau).
Redis client pour caching et persistence.
Common dependency factories for endpoints.
"""
from __future__ import annotations
from typing import Any, Callable, Optional, Tuple
from fastapi import Cookie, Depends, Header, HTTPException, status, Query
import logging
import os

import jwt
from jwt import InvalidTokenError

from api.config.users import (
    get_default_user,
    is_allowed_user,
    validate_user_id,
    get_user_info
)
from api.auth_security import (
    ACCESS_COOKIE,
    AuthenticatedUser,
    get_auth_mode,
    get_jwt_secret,
    is_access_token_revoked,
)

logger = logging.getLogger(__name__)

# ============================================================================
# JWT Token Validation (imported from auth_router)
# ============================================================================

def decode_access_token(token: str) -> Optional[dict]:
    """
    Décode et valide un JWT token.

    Args:
        token: JWT token à décoder

    Returns:
        dict: Payload du token si valide, None sinon
    """
    try:
        ALGORITHM = "HS256"

        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
        if is_access_token_revoked(payload):
            return None
        return payload
    except InvalidTokenError as e:
        logger.debug(f"JWT decode error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected JWT decode error: {e}")
        return None

def _extract_jwt_user(authorization: Optional[str]) -> Optional[str]:
    """
    Extract and validate user_id from a JWT Authorization header.

    Returns user_id if JWT is valid, None if no JWT present.
    Raises HTTPException(401) if JWT is present but invalid/expired.
    """
    if not authorization or not isinstance(authorization, str):
        return None

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None  # Not a Bearer token — skip silently

    token = parts[1]
    payload = decode_access_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check user still exists and is active
    if not is_allowed_user(user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_info = get_user_info(user_id)
    if user_info and user_info.get("status") != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user_id


def _extract_cookie_user(access_cookie: Optional[str]) -> Optional[str]:
    if not isinstance(access_cookie, str) or not access_cookie:
        return None
    payload = decode_access_token(access_cookie)
    if not payload or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
        )
    user_id = payload["sub"]
    if not is_allowed_user(user_id):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    user_info = get_user_info(user_id)
    if not user_info or user_info.get("status") != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive")
    return user_id


def resolve_authenticated_user(
    *,
    authorization: Optional[str] = None,
    access_cookie: Optional[str] = None,
    x_user: Optional[str] = None,
) -> str:
    """Resolve identity from a session; X-User is only a consistency assertion."""
    mode = get_auth_mode()
    if mode == "cookie" and authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer authentication is disabled in cookie mode",
        )
    bearer_user = _extract_jwt_user(authorization)
    cookie_user = _extract_cookie_user(access_cookie)
    if bearer_user and cookie_user and bearer_user != cookie_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session identity mismatch")
    session_user = cookie_user or bearer_user

    normalized_header = validate_user_id(x_user) if isinstance(x_user, str) and x_user else None
    if mode == "legacy":
        if not normalized_header:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="X-User header is required in legacy mode",
            )
        if not is_allowed_user(normalized_header):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Unknown user: {normalized_header}",
            )
        if session_user and session_user != normalized_header:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User identity mismatch between token and header",
            )
        return session_user or normalized_header

    if not session_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication session required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if normalized_header and normalized_header != session_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User identity mismatch between session and header",
        )
    return session_user


# Redis client singleton
_redis_client = None

def get_required_user(
    x_user: Optional[str] = Header(None, alias="X-User"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    access_cookie: Optional[str] = Cookie(None, alias=ACCESS_COOKIE),
) -> str:
    """
    Dépendance FastAPI qui FORCE le header X-User et valide le JWT si présent.

    En mode soft (REQUIRE_JWT=0, défaut) : valide le JWT si présent, fallback X-User.
    En mode strict (REQUIRE_JWT=1) : rejette les requêtes sans JWT valide.

    Args:
        x_user: Header X-User REQUIS
        authorization: Header Authorization optionnel (Bearer token)

    Returns:
        str: ID utilisateur validé

    Raises:
        HTTPException: 422 si header X-User manquant, 401 si JWT invalide,
                       403 si utilisateur inconnu ou mismatch JWT/X-User

    Example:
        @router.get("/endpoint")
        async def endpoint(user: str = Depends(get_required_user)):
            # user est garanti non-None, JWT validé si présent
    """
    try:
        mode = get_auth_mode()
        if mode == "legacy":
            if x_user is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="X-User header is required in legacy mode",
                )
            normalized_user = validate_user_id(x_user)
            if os.getenv("DEV_OPEN_API", "0") == "1":
                return normalized_user
            jwt_user = _extract_jwt_user(authorization)
            if os.getenv("REQUIRE_JWT", "0") == "1" and not jwt_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication token required",
                )
            if jwt_user and jwt_user != normalized_user:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User identity mismatch between token and header",
                )
            effective_user = jwt_user or normalized_user
            if not is_allowed_user(effective_user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Unknown user: {effective_user}",
                )
            return effective_user
        return resolve_authenticated_user(
            authorization=authorization,
            access_cookie=access_cookie,
            x_user=x_user,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user ID format: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error in get_required_user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


def require_admin_role(
    x_user: Optional[str] = Header(None, alias="X-User"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    access_cookie: Optional[str] = Cookie(None, alias=ACCESS_COOKIE),
) -> str:
    """
    Dépendance FastAPI qui FORCE le rôle admin et valide le JWT si présent.

    Usage: Pour endpoints admin uniquement (user management, logs, cache, ML, API keys).

    Args:
        x_user: Header X-User REQUIS
        authorization: Header Authorization optionnel (Bearer token)

    Returns:
        str: ID utilisateur validé avec rôle admin

    Raises:
        HTTPException: 401 si JWT invalide, 403 si pas admin ou mismatch

    Example:
        @router.get("/admin/users")
        async def list_users(user: str = Depends(require_admin_role)):
            # user est garanti avoir le rôle "admin", JWT validé si présent
    """
    try:
        user_id = get_required_user(x_user, authorization, access_cookie)
        if get_auth_mode() == "legacy" and os.getenv("DEV_OPEN_API", "0") == "1":
            return user_id
        user_info = get_user_info(user_id)
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User info not found: {user_id}",
            )
        if "admin" in user_info.get("roles", []):
            return user_id
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error in require_admin_role: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


def get_current_user_jwt(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    access_cookie: Optional[str] = Cookie(None, alias=ACCESS_COOKIE),
) -> str:
    """
    Dépendance FastAPI qui extrait et valide le JWT token.

    Usage: Pour nouveaux endpoints nécessitant authentification JWT.

    Args:
        authorization: Header Authorization avec format "Bearer <token>"

    Returns:
        str: ID utilisateur extrait du token JWT

    Raises:
        HTTPException: 401 si token manquant/invalide/expiré

    Example:
        @router.get("/endpoint")
        async def endpoint(user: str = Depends(get_current_user_jwt)):
            # user est garanti authentifié via JWT
    """
    if not isinstance(authorization, str):
        authorization = None
    if not isinstance(access_cookie, str):
        access_cookie = None

    if get_auth_mode() == "cookie" and authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer authentication is disabled in cookie mode",
        )

    # Mode développement : bypass uniquement en mode legacy
    dev_skip_auth = (
        get_auth_mode() == "legacy"
        and os.getenv("ENVIRONMENT", "development").lower() != "production"
        and os.getenv("DEV_SKIP_AUTH", "0") == "1"
    )
    if dev_skip_auth:
        default_user = get_default_user()
        logger.info(f"DEV MODE: Bypassing JWT auth, using default user: {default_user}")
        return default_user

    if authorization and not access_cookie:
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token format",
                headers={"WWW-Authenticate": "Bearer"},
            )

    bearer_user = _extract_jwt_user(authorization)
    cookie_user = _extract_cookie_user(access_cookie)
    if bearer_user and cookie_user and bearer_user != cookie_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session identity mismatch",
        )
    user_id = cookie_user or bearer_user
    if not user_id:
        logger.warning("Missing Authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    logger.debug(f"JWT authenticated user: {user_id}")
    return user_id


def require_any_role(*allowed_roles: str) -> Callable:
    """FastAPI dependency requiring at least one current role (admin always qualifies)."""
    allowed = set(allowed_roles)

    def dependency(user_id: str = Depends(get_current_user_jwt)) -> AuthenticatedUser:
        user_info = get_user_info(user_id)
        roles = set(user_info.get("roles", [])) if user_info else set()
        if "admin" not in roles and roles.isdisjoint(allowed):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of these roles is required: {', '.join(sorted(allowed))}",
            )
        return AuthenticatedUser(username=user_id, roles=sorted(roles))

    return dependency


def require_admin_role_jwt(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    access_cookie: Optional[str] = Cookie(None, alias=ACCESS_COOKIE),
) -> str:
    """
    Dépendance FastAPI qui FORCE le rôle admin via JWT token.

    Usage: Pour endpoints admin avec authentification JWT.

    Args:
        authorization: Header Authorization avec format "Bearer <token>"

    Returns:
        str: ID utilisateur validé avec rôle admin

    Raises:
        HTTPException: 401 si token invalide, 403 si pas admin

    Example:
        @router.get("/admin/users")
        async def list_users(user: str = Depends(require_admin_role_jwt)):
            # user est garanti avoir le rôle "admin" via JWT
    """
    # Valider le JWT d'abord
    if isinstance(access_cookie, str):
        user_id = get_current_user_jwt(authorization, access_cookie)
    else:
        user_id = get_current_user_jwt(authorization)

    # Récupérer les infos utilisateur pour vérifier le rôle
    user_info = get_user_info(user_id)
    if not user_info:
        logger.warning(f"User info not found for admin access: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User info not found: {user_id}"
        )

    # Vérifier le rôle admin
    user_roles = user_info.get("roles", [])
    if "admin" not in user_roles:
        logger.warning(f"User {user_id} attempted admin access without admin role (roles: {user_roles})")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation"
        )

    # Log pour audit
    logger.info(f"Admin access granted via JWT for user: {user_id}")
    return user_id


def get_redis_client() -> Optional[Any]:
    """
    Dépendance FastAPI pour obtenir le client Redis.

    Retourne un client Redis singleton partagé entre toutes les requêtes.
    Si Redis n'est pas disponible, retourne None (graceful degradation).

    Returns:
        Redis client ou None si indisponible

    Usage:
        @app.get("/endpoint")
        async def endpoint(redis = Depends(get_redis_client)):
            if redis:
                # Utiliser Redis
                redis.set(key, value)
    """
    global _redis_client

    # Return existing client if available
    if _redis_client is not None:
        try:
            # Test connection (with timeout already configured on client)
            _redis_client.ping()
            return _redis_client
        except Exception as e:
            logger.debug(f"Redis client lost connection: {e}")
            _redis_client = None

    # Try to create new client
    try:
        from redis import Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # Skip if REDIS_URL is empty
        if not redis_url or redis_url.strip() == "":
            logger.debug("Redis disabled (REDIS_URL is empty)")
            return None

        # Create client with fast timeout (0.5s) to avoid blocking
        _redis_client = Redis.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=0.5,  # Fast connection timeout
            socket_timeout=0.5             # Fast operation timeout
        )

        # Test connection (will timeout quickly if unavailable)
        _redis_client.ping()
        logger.info(f"Redis client connected: {redis_url}")
        return _redis_client

    except Exception as e:
        logger.debug(f"Redis not available: {e}")
        return None


# ============================================================================
# Common Dependency Factories
# ============================================================================

def get_user_and_source(
    user: str = Header(None, alias="X-User"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    access_cookie: Optional[str] = Cookie(None, alias=ACCESS_COOKIE),
    source: str = Query("auto", description="Data source (auto, cointracking, saxobank)")
) -> Tuple[str, str]:
    """
    Dependency factory for endpoints that need both user_id and source.

    This consolidates the common pattern of extracting user from header
    and source from query parameters.

    Args:
        user: Optional consistency header; it never authenticates in secure modes
        source: Data source from query parameter (default: 'auto')

    Returns:
        Tuple[str, str]: (user_id, source)

    Usage:
        from api.deps import get_user_and_source
        from fastapi import Depends

        @app.get("/endpoint")
        async def endpoint(
            user_source: Tuple[str, str] = Depends(get_user_and_source)
        ):
            user_id, source = user_source
            # ... use user_id and source

        # Or with unpacking (Python 3.10+)
        @app.get("/endpoint")
        async def endpoint(
            user_id: str = Depends(lambda x=Depends(get_user_and_source): x[0]),
            source: str = Depends(lambda x=Depends(get_user_and_source): x[1])
        ):
            # ... use user_id and source

        # Or simplest (extract from dict):
        @app.get("/endpoint")
        async def endpoint(params: dict = Depends(get_user_and_source_dict)):
            user_id = params["user_id"]
            source = params["source"]
    """
    try:
        if get_auth_mode() == "legacy" and not user:
            return get_default_user(), source
        user_id = resolve_authenticated_user(
            authorization=authorization,
            access_cookie=access_cookie,
            x_user=user,
        )
        return user_id, source
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user ID format: {exc}",
        ) from exc


def get_user_and_source_dict(
    user_source: Optional[Tuple[str, str]] = None
) -> dict:
    """
    Alternative dependency that returns user and source as a dict.

    This is a convenience wrapper around get_user_and_source that returns
    a dict instead of a tuple for easier unpacking.

    Returns:
        dict: {"user_id": str, "source": str}

    Usage:
        @app.get("/endpoint")
        async def endpoint(params: dict = Depends(get_user_and_source_dict)):
            user_id = params["user_id"]
            source = params["source"]
    """
    if user_source is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication session required",
        )
    else:
        user_id, source = user_source

    return {"user_id": user_id, "source": source}
