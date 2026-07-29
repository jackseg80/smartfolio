"""
Middleware configuration for SmartFolio API

Extracted from api/main.py for better maintainability.
Configures all application middlewares: CORS, security, compression, rate limiting, etc.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from api.middleware import RateLimitMiddleware
from api.auth_security import ACCESS_COOKIE, get_auth_mode, validate_csrf
from api.deps import resolve_authenticated_user
from api.middlewares import (
    add_security_headers_middleware,
    no_cache_dev_middleware,
    request_logger_middleware,
    request_timing_middleware,
)

logger = logging.getLogger(__name__)

PUBLIC_EXACT_PATHS = {
    "/",
    "/auth/login",
    "/auth/refresh",
    "/healthz",
    "/favicon.ico",
    "/api/saxo/callback",
}


def _is_public_path(path: str) -> bool:
    return path in PUBLIC_EXACT_PATHS or path.startswith("/static/")


def setup_middlewares(
    app: FastAPI, settings, debug: bool, environment: str, cors_origins: Optional[List[str]] = None
) -> None:
    """
    Configure all application middlewares in the correct order.

    Middleware order matters in FastAPI - they are applied in reverse order:
    - Last added middleware runs first on requests
    - First added middleware runs last on requests

    Args:
        app: FastAPI application instance
        settings: Application settings from config
        debug: Debug mode flag
        environment: Environment name (development/production)
        cors_origins: List of allowed CORS origins (optional)
    """
    # ========== CORS Configuration ==========
    # Sécurisé avec configuration dynamique
    # Note: file:// et null retirés pour sécurité (risque CSRF)
    # Pour fichiers HTML locaux, utiliser un serveur HTTP local (ex: python -m http.server)
    default_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=(cors_origins or default_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Authorization",
            "Content-Type",
            "Idempotency-Key",
            "X-CSRF-Token",
            "X-User",
        ],
    )
    logger.info(f"✅ CORS configured with {len(cors_origins or default_origins)} allowed origins")

    # ========== HTTPS Redirect (Configurable) ==========
    # HTTPS redirect contrôlé via SECURITY_FORCE_HTTPS
    # Pour serveur LAN sans SSL : laisser à False
    # Pour serveur public avec SSL : mettre à True
    if settings.security.force_https:
        app.add_middleware(HTTPSRedirectMiddleware)
        logger.info("🔒 HTTPSRedirectMiddleware activé (SECURITY_FORCE_HTTPS=true)")
    else:
        logger.info("⚠️  HTTPSRedirectMiddleware désactivé (SECURITY_FORCE_HTTPS=false)")

    # ========== Trusted Host Configuration ==========
    # TrustedHost config selon l'environnement
    # Lecture depuis ALLOWED_HOSTS (env var) pour flexibilité production
    allowed_hosts_env = os.getenv("ALLOWED_HOSTS", "")

    if allowed_hosts_env:
        # Si ALLOWED_HOSTS défini, utiliser la liste (comma-separated)
        allowed_hosts = [h.strip() for h in allowed_hosts_env.split(",") if h.strip()]
        logger.info(f"🔒 TrustedHostMiddleware: custom allowed_hosts={allowed_hosts}")
    elif debug:
        # En développement, plus permissif pour les tests
        allowed_hosts = ["*"]
        logger.info("🔒 TrustedHostMiddleware: dev mode (allow all hosts)")
    else:
        raise RuntimeError("ALLOWED_HOSTS must be configured in production")

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    # ========== GZip Compression ==========
    # Compression GZip pour améliorer les performances
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    logger.info("✅ GZip compression enabled (minimum_size=1000)")

    # ========== Rate Limiting (Production Only) ==========
    # Rate limiting (production only)
    if environment == "production" or not debug:
        app.add_middleware(RateLimitMiddleware)
        logger.info("✅ Rate limiting middleware enabled (production mode)")
    else:
        logger.info("⚠️  Rate limiting middleware disabled (development mode)")

    # ========== Custom HTTP Middlewares ==========
    # Middleware order (applied in reverse):
    # 1. No-cache for dev (last to run, first in stack)
    # 2. Request logger
    # 3. Request timing
    # 4. Security headers (first to run, last in stack)

    # Security headers (CSP, HSTS, etc.)
    app.middleware("http")(add_security_headers_middleware)
    logger.info("✅ Security headers middleware registered")

    # Request timing and structured logging
    app.middleware("http")(request_timing_middleware)
    logger.info("✅ Request timing middleware registered")

    # Request logger (debug mode)
    app.middleware("http")(request_logger_middleware)
    logger.info("✅ Request logger middleware registered")

    # No-cache for static files (development only)
    app.middleware("http")(no_cache_dev_middleware)
    logger.info("✅ No-cache dev middleware registered")

    @app.middleware("http")
    async def default_authentication_middleware(request: Request, call_next):
        """Require a coherent authenticated identity on every non-public route."""
        mode = get_auth_mode()
        if (
            mode == "legacy"
            or request.method == "OPTIONS"
            or _is_public_path(request.url.path)
        ):
            return await call_next(request)
        try:
            resolve_authenticated_user(
                authorization=request.headers.get("Authorization"),
                access_cookie=request.cookies.get(ACCESS_COOKIE),
                x_user=request.headers.get("X-User"),
            )
            uses_cookie = bool(request.cookies.get(ACCESS_COOKIE))
            bearer_compat = mode == "dual" and bool(request.headers.get("Authorization"))
            if request.method not in {"GET", "HEAD", "OPTIONS"} and uses_cookie and not bearer_compat:
                validate_csrf(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )
        return await call_next(request)

    logger.info("🔒 Default authentication middleware registered (dual/cookie modes)")

    logger.info("🎯 All middlewares configured successfully")
