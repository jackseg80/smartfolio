"""
Base classes for the modular sources system.

Defines:
- SourceInfo: Metadata about a source implementation
- BalanceItem: Standardized balance/position item
- SourceBase: Abstract base class all sources must implement
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.sources.category import SourceCategory, SourceMode, SourceStatus


@dataclass
class SourceInfo:
    """Metadata about a source implementation."""

    id: str  # Unique identifier (e.g., "manual_crypto")
    name: str  # Display name for UI
    category: SourceCategory  # CRYPTO or BOURSE
    mode: SourceMode  # MANUAL, CSV, or API
    description: str  # Short description for UI
    icon: str  # Icon identifier (e.g., "pencil", "upload", "api")
    supports_transactions: bool = False  # Whether it tracks history
    requires_credentials: bool = False  # Whether API credentials needed
    file_patterns: List[str] = field(default_factory=list)  # For CSV sources


@dataclass
class BalanceItem:
    """Standardized balance/position item returned by all sources."""

    symbol: str  # Asset symbol (BTC, AAPL, etc.)
    amount: float  # Quantity held
    value_usd: float  # Current value in USD
    source_id: str  # Source that provided this data

    # Optional fields
    alias: Optional[str] = None  # Display name
    location: Optional[str] = None  # Where held (wallet, exchange, broker)
    price_usd: Optional[float] = None  # Unit price in USD
    currency: str = "USD"  # Original currency
    asset_class: Optional[str] = None  # CRYPTO, EQUITY, ETF, BOND, etc.

    # Bourse-specific
    isin: Optional[str] = None  # ISIN for stocks
    instrument_name: Optional[str] = None  # Full instrument name
    avg_price: Optional[float] = None  # Average purchase price

    # Manual entry specific
    entry_id: Optional[str] = None  # UUID for manual entries

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.alias is None:
            self.alias = self.symbol

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "symbol": self.symbol,
            "alias": self.alias,
            "amount": self.amount,
            "value_usd": self.value_usd,
            "location": self.location,
            "price_usd": self.price_usd,
            "currency": self.currency,
            "asset_class": self.asset_class,
            "isin": self.isin,
            "instrument_name": self.instrument_name,
            "avg_price": self.avg_price,
            "source_id": self.source_id,
        }


class SourceBase(ABC):
    """
    Abstract base class for all data sources.

    Each source implementation must:
    1. Define get_source_info() classmethod returning SourceInfo
    2. Implement get_balances() to return list of BalanceItem
    3. Implement validate_config() to check configuration
    4. Implement get_status() to report operational status

    Multi-tenant: Each instance is scoped to a specific user_id.
    """

    def __init__(self, user_id: str, project_root: str):
        """
        Initialize source for a specific user.

        Args:
            user_id: User identifier for multi-tenant isolation
            project_root: Project root directory path
        """
        self.user_id = user_id
        self.project_root = project_root

    @classmethod
    @abstractmethod
    def get_source_info(cls) -> SourceInfo:
        """
        Return metadata about this source.

        This is a classmethod so it can be called without instantiation
        for source discovery and registration.
        """
        pass

    @abstractmethod
    async def get_balances(self) -> List[BalanceItem]:
        """
        Fetch current balances/positions from this source.

        Returns:
            List of BalanceItem with standardized format
        """
        pass

    @abstractmethod
    async def validate_config(self) -> tuple[bool, Optional[str]]:
        """
        Validate source configuration.

        Returns:
            Tuple of (is_valid, error_message)
            error_message is None if valid
        """
        pass

    @abstractmethod
    def get_status(self) -> SourceStatus:
        """
        Get current operational status.

        Returns:
            SourceStatus enum value
        """
        pass

    # Optional methods with default implementations

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get detailed positions (default: convert balances to dicts).

        Override for sources that provide additional position details.
        """
        balances = await self.get_balances()
        return [b.to_dict() for b in balances]

    async def get_transactions(
        self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history.

        Default implementation returns empty list.
        Override for sources that support transaction history.
        """
        return []

    def supports_feature(self, feature: str) -> bool:
        """
        Check if source supports a specific feature.

        Features: "transactions", "real_time", "auto_price", etc.
        """
        info = self.get_source_info()
        if feature == "transactions":
            return info.supports_transactions
        return False
