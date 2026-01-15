"""
Modular Sources System for SmartFolio.

This package provides a plugin-based architecture for data sources,
supporting multiple categories (crypto, bourse) with various modes
(manual, csv, api).

Usage:
    from services.sources import source_registry, SourceCategory

    # List available crypto sources
    crypto_sources = source_registry.list_sources(SourceCategory.CRYPTO)

    # Get source for user
    source = source_registry.get_source("manual_crypto", user_id, project_root)
    balances = await source.get_balances()
"""
from services.sources.base import BalanceItem, SourceBase, SourceInfo
from services.sources.category import SourceCategory, SourceMode, SourceStatus
from services.sources.registry import source_registry

__all__ = [
    # Enums
    "SourceCategory",
    "SourceMode",
    "SourceStatus",
    # Base classes
    "SourceBase",
    "SourceInfo",
    "BalanceItem",
    # Registry
    "source_registry",
]
