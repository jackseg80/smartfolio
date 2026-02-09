"""
Dépendances FastAPI réutilisables.
Gestion des utilisateurs avec header X-User (legacy) ou JWT token (nouveau).
Redis client pour caching et persistence.
Common dependency factories for endpoints.
"""
from __future__ import annotations
from typing import Optional, Tuple
from fastapi import Header, HTTPException, status, Query
import logging
import os

from api.config.users import (
    get_default_user,
    is_allowed_user,
    validate_user_id,
    get_user_info
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
        from jose import jwt, JWTError
        SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-please")
        ALGORITHM = "HS256"

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
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


# Redis client singleton
_redis_client = None

def get_required_user(
    x_user: str = Header(..., alias="X-User"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
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
        # Validation et normalisation
        normalized_user = validate_user_id(x_user)

        # Mode développement : bypass de l'autorisation si DEV_OPEN_API=1
        dev_mode = os.getenv("DEV_OPEN_API", "0") == "1"
        if dev_mode:
            logger.info(f"DEV MODE: Bypassing authorization for user: {normalized_user}")
            return normalized_user

        # JWT validation (if present)
        jwt_user = _extract_jwt_user(authorization)

        # Strict mode: reject requests without valid JWT
        require_jwt = os.getenv("REQUIRE_JWT", "0") == "1"
        if require_jwt and not jwt_user:
            logger.warning(f"JWT required but not provided for user: {normalized_user}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Anti-spoofing: ensure JWT user matches X-User header
        if jwt_user and jwt_user != normalized_user:
            logger.warning(f"JWT/X-User mismatch: JWT={jwt_user}, X-User={normalized_user}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User identity mismatch between token and header",
            )

        effective_user = jwt_user or normalized_user

        # Vérification autorisation normale
        if not is_allowed_user(effective_user):
            logger.warning(f"Unknown user attempted access (required): {effective_user}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Unknown user: {effective_user}",
            )

        # Log pour audit
        auth_mode = "[JWT]" if jwt_user else "[X-User]"
        logger.info(f"Active user (required): {effective_user} {auth_mode}")
        return effective_user

    except ValueError as e:
        logger.warning(f"Invalid user ID format: {x_user} - {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user ID format: {e}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_required_user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


def require_admin_role(
    x_user: str = Header(..., alias="X-User"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
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
        normalized_user = validate_user_id(x_user)

        # Mode développement : bypass de l'autorisation si DEV_OPEN_API=1
        dev_mode = os.getenv("DEV_OPEN_API", "0") == "1"
        if dev_mode:
            logger.info(f"DEV MODE: Bypassing admin role check for user: {normalized_user}")
            return normalized_user

        # JWT validation (if present)
        jwt_user = _extract_jwt_user(authorization)

        # Strict mode: reject requests without valid JWT
        require_jwt = os.getenv("REQUIRE_JWT", "0") == "1"
        if require_jwt and not jwt_user:
            logger.warning(f"JWT required but not provided for admin user: {normalized_user}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Anti-spoofing: ensure JWT user matches X-User header
        if jwt_user and jwt_user != normalized_user:
            logger.warning(f"JWT/X-User mismatch in admin endpoint: JWT={jwt_user}, X-User={normalized_user}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User identity mismatch between token and header",
            )

        effective_user = jwt_user or normalized_user

        # Vérification autorisation normale
        if not is_allowed_user(effective_user):
            logger.warning(f"Unknown user attempted admin access: {effective_user}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Unknown user: {effective_user}",
            )

        # Récupérer les infos utilisateur pour vérifier le rôle
        user_info = get_user_info(effective_user)
        if not user_info:
            logger.warning(f"User info not found for admin access: {effective_user}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User info not found: {effective_user}",
            )

        # Vérifier le rôle admin
        user_roles = user_info.get("roles", [])
        if "admin" not in user_roles:
            logger.warning(f"User {effective_user} attempted admin access without admin role (roles: {user_roles})")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin role required for this operation",
            )

        # Log pour audit
        auth_mode = "[JWT]" if jwt_user else "[X-User]"
        logger.info(f"Admin access granted for user: {effective_user} {auth_mode}")
        return effective_user

    except ValueError as e:
        logger.warning(f"Invalid user ID format in admin endpoint: {x_user} - {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user ID format: {e}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in require_admin_role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


def get_current_user_jwt(authorization: Optional[str] = Header(None, alias="Authorization")) -> str:
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
    # Mode développement : bypass si DEV_SKIP_AUTH=1
    dev_skip_auth = os.getenv("DEV_SKIP_AUTH", "0") == "1"
    if dev_skip_auth:
        default_user = get_default_user()
        logger.info(f"DEV MODE: Bypassing JWT auth, using default user: {default_user}")
        return default_user

    # Vérifier présence du header
    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Extraire le token du header "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        logger.warning(f"Invalid Authorization header format: {authorization[:20]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token format",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = parts[1]

    # Décoder et valider le token
    payload = decode_access_token(token)
    if not payload:
        logger.warning("Invalid or expired JWT token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Extraire l'user_id du payload
    user_id = payload.get("sub")
    if not user_id:
        logger.error("JWT payload missing 'sub' claim")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Vérifier que l'utilisateur existe toujours
    if not is_allowed_user(user_id):
        logger.warning(f"JWT token for unknown/deleted user: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Vérifier le status
    user_info = get_user_info(user_id)
    if user_info and user_info.get("status") != "active":
        logger.warning(f"JWT token for inactive user: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    # Log pour audit
    logger.debug(f"JWT authenticated user: {user_id}")
    return user_id


def require_admin_role_jwt(authorization: str = Header(..., alias="Authorization")) -> str:
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


def get_redis_client() -> Optional[any]:
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
    source: str = Query("auto", description="Data source (auto, cointracking, saxobank)")
) -> Tuple[str, str]:
    """
    Dependency factory for endpoints that need both user_id and source.

    This consolidates the common pattern of extracting user from header
    and source from query parameters.

    Args:
        user: User ID from X-User header (optional, defaults to 'demo')
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
    # Use get_active_user logic for user extraction
    if not user:
        user_id = get_default_user()
        logger.debug(f"No X-User header, using default: {user_id}")
    else:
        try:
            user_id = validate_user_id(user)

            # Dev mode bypass
            dev_mode = os.getenv("DEV_OPEN_API", "0") == "1"
            if dev_mode:
                logger.info(f"DEV MODE: Bypassing authorization for user: {user_id}")
            elif not is_allowed_user(user_id):
                logger.warning(f"Unknown user attempted access: {user}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Unknown user: {user}"
                )

            logger.info(f"Active user: {user_id}")

        except ValueError as e:
            logger.warning(f"Invalid user ID format: {user} - {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid user ID format: {e}"
            )

    return user_id, source


def get_user_and_source_dict(
    user_source: Tuple[str, str] = None
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
        # This should not happen if used as a dependency
        user_id, source = get_default_user(), "auto"
    else:
        user_id, source = user_source

    return {"user_id": user_id, "source": source}