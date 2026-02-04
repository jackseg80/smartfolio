"""
Exceptions personnalisées pour l'API crypto-rebalancer

NOTE: Les classes d'exceptions sont maintenant définies dans shared/exceptions.py
Ce fichier réexporte pour backward compatibility et ajoute les helpers FastAPI.
"""
from typing import Any, Optional
from fastapi import HTTPException

# Re-export from shared for backward compatibility
from shared.exceptions import (
    CryptoRebalancerException,
    APIException,
    DataException,
    ExchangeException,
    ConfigurationException,
    ValidationException,
    TradingException,
    StorageException,
    GovernanceException,
    MonitoringException,
    ConfigurationError,
    PricingException,
    NetworkException,
    TimeoutException,
    RateLimitException,
    DataNotFoundException,
    InsufficientBalanceException,
    ErrorCode,
    convert_standard_exception,
    handle_exceptions,
    handle_exceptions_async,
)

# Export all for `from api.exceptions import *`
__all__ = [
    "CryptoRebalancerException",
    "APIException",
    "DataException",
    "ExchangeException",
    "ConfigurationException",
    "ValidationException",
    "TradingException",
    "StorageException",
    "GovernanceException",
    "MonitoringException",
    "ConfigurationError",
    "PricingException",
    "NetworkException",
    "TimeoutException",
    "RateLimitException",
    "DataNotFoundException",
    "InsufficientBalanceException",
    "ErrorCode",
    "convert_standard_exception",
    "handle_exceptions",
    "handle_exceptions_async",
    "create_http_exception",
    "ErrorCodes",
]


# === FastAPI-specific helpers (not in shared/) ===

def create_http_exception(exc: CryptoRebalancerException, status_code: int = 400) -> HTTPException:
    """Convertit une exception personnalisée en HTTPException"""
    return HTTPException(
        status_code=status_code,
        detail={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": getattr(exc, 'details', None)
        }
    )


# Codes d'erreur HTTP standards (legacy - prefer ErrorCode enum from shared)
class ErrorCodes:
    """Legacy HTTP error codes - prefer ErrorCode enum from shared.exceptions"""
    # API Errors (500-599)
    API_UNAVAILABLE = 500
    API_TIMEOUT = 503
    API_RATE_LIMIT = 429

    # Validation Errors (400-499)
    INVALID_INPUT = 400
    MISSING_PARAMETER = 422
    INVALID_FORMAT = 400

    # Configuration Errors (500-599)
    MISSING_CONFIG = 500
    INVALID_CONFIG = 500

    # Trading Errors (400-499)
    INSUFFICIENT_BALANCE = 400
    INVALID_SYMBOL = 400
    MARKET_CLOSED = 400

    # Data Errors (500-599)
    DATA_NOT_FOUND = 404
    DATA_CORRUPT = 500
