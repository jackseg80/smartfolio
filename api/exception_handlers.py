"""
Exception handlers for SmartFolio API

Extracted from api/main.py for better maintainability.
Configures global exception handlers for custom and generic exceptions.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.exceptions import (
    APIException,
    ConfigurationException,
    CryptoRebalancerException,
    DataException,
    ErrorCodes,
    TradingException,
    ValidationException,
)

logger = logging.getLogger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Configure global exception handlers for the FastAPI application.

    Handles:
    - Custom CryptoRebalancerException and subclasses
    - Generic Python exceptions as fallback

    Args:
        app: FastAPI application instance
    """

    @app.exception_handler(CryptoRebalancerException)
    async def crypto_exception_handler(request: Request, exc: CryptoRebalancerException):
        """Gestionnaire pour toutes les exceptions personnalisées"""
        status_code = 400
        if isinstance(exc, APIException):
            status_code = exc.status_code or 500
        elif isinstance(exc, ValidationException):
            status_code = ErrorCodes.INVALID_INPUT
        elif isinstance(exc, ConfigurationException):
            status_code = ErrorCodes.INVALID_CONFIG
        elif isinstance(exc, TradingException):
            status_code = ErrorCodes.INSUFFICIENT_BALANCE
        elif isinstance(exc, DataException):
            status_code = ErrorCodes.DATA_NOT_FOUND

        return JSONResponse(
            status_code=status_code,
            content={
                "ok": False,
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Gestionnaire pour toutes les autres exceptions"""
        # Log l'exception complète avec stacktrace pour debugging
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": str(exc) if app.debug else None,
                "path": request.url.path,
            },
        )

    logger.info("✅ Exception handlers configured successfully")
