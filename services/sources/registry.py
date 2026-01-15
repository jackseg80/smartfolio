"""
Source Registry - Central registry for all source implementations.

Provides:
- Auto-registration of built-in sources
- Dynamic source discovery by category
- Factory method to instantiate sources for users

Usage:
    from services.sources.registry import source_registry

    # List available sources
    sources = source_registry.list_sources(SourceCategory.CRYPTO)

    # Get source instance for user
    source = source_registry.get_source("manual_crypto", user_id, project_root)
    balances = await source.get_balances()
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

from services.sources.base import SourceBase, SourceInfo
from services.sources.category import SourceCategory, SourceMode

logger = logging.getLogger(__name__)


class SourceRegistry:
    """
    Registry for all available source implementations.

    Singleton pattern ensures single source of truth for available sources.
    """

    _instance: Optional[SourceRegistry] = None
    _sources: Dict[str, Type[SourceBase]]
    _initialized: bool

    def __new__(cls) -> SourceRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sources = {}
            cls._instance._initialized = False
        return cls._instance

    def _auto_register(self) -> None:
        """Auto-register built-in sources on first access."""
        if self._initialized:
            return

        self._initialized = True
        logger.info("Initializing SourceRegistry with built-in sources...")

        # Import and register sources lazily to avoid circular imports
        try:
            from services.sources.crypto.manual import ManualCryptoSource

            self.register(ManualCryptoSource)
        except ImportError as e:
            logger.debug(f"ManualCryptoSource not available: {e}")

        try:
            from services.sources.bourse.manual import ManualBourseSource

            self.register(ManualBourseSource)
        except ImportError as e:
            logger.debug(f"ManualBourseSource not available: {e}")

        try:
            from services.sources.crypto.cointracking_csv import CoinTrackingCSVSource

            self.register(CoinTrackingCSVSource)
        except ImportError as e:
            logger.debug(f"CoinTrackingCSVSource not available: {e}")

        try:
            from services.sources.crypto.cointracking_api import CoinTrackingAPISource

            self.register(CoinTrackingAPISource)
        except ImportError as e:
            logger.debug(f"CoinTrackingAPISource not available: {e}")

        try:
            from services.sources.bourse.saxobank_csv import SaxoBankCSVSource

            self.register(SaxoBankCSVSource)
        except ImportError as e:
            logger.debug(f"SaxoBankCSVSource not available: {e}")

        logger.info(f"SourceRegistry initialized with {len(self._sources)} sources")

    def register(self, source_class: Type[SourceBase]) -> None:
        """
        Register a source implementation.

        Args:
            source_class: SourceBase subclass to register
        """
        info = source_class.get_source_info()
        self._sources[info.id] = source_class
        logger.info(
            f"Registered source: {info.id} ({info.category.value}/{info.mode.value})"
        )

    def unregister(self, source_id: str) -> bool:
        """
        Unregister a source (mainly for testing).

        Args:
            source_id: ID of source to remove

        Returns:
            True if removed, False if not found
        """
        if source_id in self._sources:
            del self._sources[source_id]
            return True
        return False

    def get_source(
        self, source_id: str, user_id: str, project_root: str
    ) -> Optional[SourceBase]:
        """
        Get a source instance by ID for a specific user.

        Args:
            source_id: Source identifier (e.g., "manual_crypto")
            user_id: User ID for multi-tenant isolation
            project_root: Project root directory

        Returns:
            Instantiated source or None if not found
        """
        self._auto_register()

        source_class = self._sources.get(source_id)
        if source_class:
            return source_class(user_id, project_root)
        return None

    def get_source_class(self, source_id: str) -> Optional[Type[SourceBase]]:
        """Get source class without instantiation."""
        self._auto_register()
        return self._sources.get(source_id)

    def list_sources(
        self, category: Optional[SourceCategory] = None
    ) -> List[SourceInfo]:
        """
        List all registered sources, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of SourceInfo for matching sources
        """
        self._auto_register()

        sources = []
        for source_class in self._sources.values():
            info = source_class.get_source_info()
            if category is None or info.category == category:
                sources.append(info)
        return sources

    def get_sources_by_category(
        self, category: SourceCategory
    ) -> Dict[SourceMode, List[SourceInfo]]:
        """
        Get sources grouped by mode for a category.

        Args:
            category: Category to filter by

        Returns:
            Dict mapping SourceMode to list of SourceInfo
        """
        self._auto_register()

        result: Dict[SourceMode, List[SourceInfo]] = {mode: [] for mode in SourceMode}

        for source_class in self._sources.values():
            info = source_class.get_source_info()
            if info.category == category:
                result[info.mode].append(info)

        return result

    def get_default_source(self, category: SourceCategory) -> Optional[str]:
        """
        Get the default source ID for a category (manual).

        Args:
            category: Category to get default for

        Returns:
            Source ID or None
        """
        self._auto_register()

        # Manual sources are default
        default_id = f"manual_{category.value}"
        if default_id in self._sources:
            return default_id

        # Fallback to first available
        for source_class in self._sources.values():
            info = source_class.get_source_info()
            if info.category == category:
                return info.id

        return None

    def is_registered(self, source_id: str) -> bool:
        """Check if a source is registered."""
        self._auto_register()
        return source_id in self._sources

    @property
    def source_ids(self) -> List[str]:
        """Get all registered source IDs."""
        self._auto_register()
        return list(self._sources.keys())


# Singleton instance
source_registry = SourceRegistry()
