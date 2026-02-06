"""
Manual Bourse Source - CRUD operations for manually entered stock/ETF positions.

Pattern based on patrimoine_service.py:
- JSON storage per user
- Atomic writes (temp file + rename)
- UUID for each position
- USD conversion via fx_service
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from services.fx_service import convert as fx_convert
from services.sources.base import BalanceItem, SourceBase, SourceInfo
from services.sources.category import SourceCategory, SourceMode, SourceStatus

logger = logging.getLogger(__name__)


class ManualBourseSource(SourceBase):
    """
    Manual entry source for stock/ETF positions.

    Stores data in: data/users/{user_id}/manual_bourse/positions.json
    """

    STORAGE_DIR = "manual_bourse"
    STORAGE_FILE = "positions.json"

    @classmethod
    def get_source_info(cls) -> SourceInfo:
        return SourceInfo(
            id="manual_bourse",
            name="Manual entry",
            category=SourceCategory.BOURSE,
            mode=SourceMode.MANUAL,
            description="Manually enter your stock positions",
            icon="pencil",
            supports_transactions=False,
            requires_credentials=False,
        )

    def __init__(self, user_id: str, project_root: str):
        super().__init__(user_id, project_root)
        self._storage_path = Path(project_root) / "data" / "users" / user_id / self.STORAGE_DIR / self.STORAGE_FILE

    def _ensure_storage(self) -> None:
        """Ensure storage directory and file exist."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._storage_path.exists():
            self._storage_path.write_text(json.dumps({"positions": [], "version": 1}), encoding="utf-8")

    def _load_data(self) -> dict:
        """Load positions from storage."""
        self._ensure_storage()
        try:
            with self._storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("positions", [])
                    return data
        except Exception as e:
            logger.warning(f"[manual_bourse] failed to load for user={self.user_id}: {e}")
        return {"positions": [], "version": 1}

    def _save_data(self, data: dict) -> None:
        """Save positions to storage (atomic write)."""
        self._ensure_storage()
        temp_path = self._storage_path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(self._storage_path)
            logger.info(f"[manual_bourse] saved {len(data.get('positions', []))} positions for user={self.user_id}")
        except Exception as e:
            logger.error(f"[manual_bourse] failed to save for user={self.user_id}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    # ============ SourceBase Implementation ============

    async def get_balances(self) -> List[BalanceItem]:
        """Return current manual positions as balance items."""
        data = self._load_data()
        positions = data.get("positions", [])
        items = []

        for pos in positions:
            value = pos.get("value", 0)
            currency = pos.get("currency", "USD").upper()

            # Convert to USD
            if currency == "USD":
                value_usd = value
            else:
                value_usd = fx_convert(value, currency, "USD")

            items.append(
                BalanceItem(
                    symbol=pos.get("symbol", "???"),
                    alias=pos.get("name", pos.get("symbol", "???")),
                    amount=pos.get("quantity", 0),
                    value_usd=value_usd,
                    location=pos.get("broker", "Manual Entry"),
                    price_usd=pos.get("price_usd"),
                    currency=currency,
                    asset_class=pos.get("asset_class", "EQUITY"),
                    isin=pos.get("isin"),
                    instrument_name=pos.get("name"),
                    avg_price=pos.get("avg_price"),
                    source_id="manual_bourse",
                    entry_id=pos.get("id"),
                )
            )

        return items

    async def validate_config(self) -> tuple[bool, Optional[str]]:
        """Manual source is always valid (no external config needed)."""
        return True, None

    def get_status(self) -> SourceStatus:
        """Check if we have any data."""
        try:
            data = self._load_data()
            positions = data.get("positions", [])
            return SourceStatus.ACTIVE if positions else SourceStatus.NOT_CONFIGURED
        except Exception:
            return SourceStatus.ERROR

    # ============ CRUD Methods ============

    def list_positions(self) -> List[dict]:
        """List all manual bourse positions."""
        data = self._load_data()
        return data.get("positions", [])

    def get_position(self, position_id: str) -> Optional[dict]:
        """Get a specific position by ID."""
        data = self._load_data()
        for pos in data.get("positions", []):
            if pos.get("id") == position_id:
                return pos
        return None

    def add_position(
        self,
        symbol: str,
        quantity: float,
        value: float,
        currency: str = "USD",
        name: Optional[str] = None,
        isin: Optional[str] = None,
        asset_class: str = "EQUITY",
        broker: str = "Manual Entry",
        avg_price: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> dict:
        """
        Add a new bourse position entry.

        Args:
            symbol: Ticker symbol (AAPL, MSFT, etc.)
            quantity: Number of shares/units
            value: Total position value
            currency: Currency of the value
            name: Full instrument name
            isin: ISIN code (optional)
            asset_class: Type (EQUITY, ETF, BOND, etc.)
            broker: Where held
            avg_price: Average purchase price
            notes: Optional notes

        Returns:
            Created position dict with ID
        """
        data = self._load_data()
        positions = data.get("positions", [])

        # Calculate price_usd if we have value and quantity
        price_usd = None
        if quantity and value:
            price = value / quantity
            if currency.upper() == "USD":
                price_usd = price
            else:
                price_usd = fx_convert(price, currency.upper(), "USD")

        new_position = {
            "id": str(uuid.uuid4()),
            "symbol": symbol.upper(),
            "name": name or symbol.upper(),
            "isin": isin,
            "quantity": quantity,
            "value": value,
            "currency": currency.upper(),
            "price_usd": price_usd,
            "asset_class": asset_class.upper(),
            "broker": broker,
            "avg_price": avg_price,
            "notes": notes,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        positions.append(new_position)
        data["positions"] = positions
        self._save_data(data)

        logger.info(f"[manual_bourse] added {symbol} for user={self.user_id}")
        return new_position

    def update_position(self, position_id: str, **kwargs) -> Optional[dict]:
        """
        Update an existing bourse position.

        Args:
            position_id: ID of position to update
            **kwargs: Fields to update

        Returns:
            Updated position dict or None if not found
        """
        data = self._load_data()
        positions = data.get("positions", [])

        for i, pos in enumerate(positions):
            if pos.get("id") == position_id:
                # Update allowed fields
                allowed_fields = {
                    "symbol", "name", "isin", "quantity", "value",
                    "currency", "asset_class", "broker", "avg_price", "notes"
                }
                for key, value in kwargs.items():
                    if key in allowed_fields:
                        if key in ("symbol", "currency", "asset_class"):
                            pos[key] = value.upper() if value else pos.get(key)
                        else:
                            pos[key] = value

                # Recalculate price_usd if value/quantity changed
                if pos.get("quantity") and pos.get("value"):
                    price = pos["value"] / pos["quantity"]
                    currency = pos.get("currency", "USD")
                    if currency == "USD":
                        pos["price_usd"] = price
                    else:
                        pos["price_usd"] = fx_convert(price, currency, "USD")

                pos["updated_at"] = datetime.utcnow().isoformat()
                positions[i] = pos
                data["positions"] = positions
                self._save_data(data)

                logger.info(f"[manual_bourse] updated position {position_id} for user={self.user_id}")
                return pos

        logger.warning(f"[manual_bourse] position not found for update id={position_id} user={self.user_id}")
        return None

    def delete_position(self, position_id: str) -> bool:
        """
        Delete a bourse position.

        Args:
            position_id: ID of position to delete

        Returns:
            True if deleted, False if not found
        """
        data = self._load_data()
        positions = data.get("positions", [])

        initial_count = len(positions)
        filtered = [p for p in positions if p.get("id") != position_id]

        if len(filtered) == initial_count:
            logger.warning(f"[manual_bourse] position not found for deletion id={position_id} user={self.user_id}")
            return False

        data["positions"] = filtered
        self._save_data(data)

        logger.info(f"[manual_bourse] deleted position {position_id} for user={self.user_id}")
        return True

    def get_summary(self) -> dict:
        """Get summary of manual bourse holdings."""
        data = self._load_data()
        positions = data.get("positions", [])

        total_usd = 0
        by_asset_class = {}

        for pos in positions:
            value = pos.get("value", 0)
            currency = pos.get("currency", "USD")
            value_usd = fx_convert(value, currency, "USD") if currency != "USD" else value

            total_usd += value_usd

            asset_class = pos.get("asset_class", "OTHER")
            by_asset_class[asset_class] = by_asset_class.get(asset_class, 0) + value_usd

        return {
            "source_id": "manual_bourse",
            "position_count": len(positions),
            "total_value_usd": total_usd,
            "by_asset_class": by_asset_class,
            "user_id": self.user_id,
        }
