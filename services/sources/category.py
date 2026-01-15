"""
Source categories and modes - Core enums for the modular sources system.

This module defines the fundamental categorization of data sources:
- SourceCategory: Asset class (crypto, bourse)
- SourceMode: Data acquisition method (manual, csv, api)
- SourceStatus: Runtime operational status
"""
from enum import Enum


class SourceCategory(str, Enum):
    """Asset category for source selection."""

    CRYPTO = "crypto"  # Cryptocurrencies (BTC, ETH, etc.)
    BOURSE = "bourse"  # Stocks, ETFs, bonds


class SourceMode(str, Enum):
    """Data acquisition mode."""

    MANUAL = "manual"  # Manual entry (default for new users)
    CSV = "csv"  # File import
    API = "api"  # Real-time API connection


class SourceStatus(str, Enum):
    """Runtime status of a source."""

    ACTIVE = "active"  # Source is working and has data
    INACTIVE = "inactive"  # Source exists but not selected
    ERROR = "error"  # Source has configuration/connection error
    NOT_CONFIGURED = "not_configured"  # Source needs setup


# Mapping of source IDs to their categories
SOURCE_CATEGORY_MAP = {
    # Crypto sources
    "manual_crypto": SourceCategory.CRYPTO,
    "cointracking_csv": SourceCategory.CRYPTO,
    "cointracking_api": SourceCategory.CRYPTO,
    # Bourse sources
    "manual_bourse": SourceCategory.BOURSE,
    "saxobank_csv": SourceCategory.BOURSE,
    "saxobank_api": SourceCategory.BOURSE,
}


def get_category_for_source(source_id: str) -> SourceCategory | None:
    """Get the category for a source ID."""
    return SOURCE_CATEGORY_MAP.get(source_id)
