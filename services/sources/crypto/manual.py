"""
Manual Crypto Source - CRUD operations for manually entered crypto balances.

Pattern based on patrimoine_service.py:
- JSON storage per user
- Atomic writes (temp file + rename)
- UUID for each asset
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


class ManualCryptoSource(SourceBase):
    """
    Manual entry source for cryptocurrency balances.

    Stores data in: data/users/{user_id}/manual_crypto/balances.json
    """

    STORAGE_DIR = "manual_crypto"
    STORAGE_FILE = "balances.json"

    @classmethod
    def get_source_info(cls) -> SourceInfo:
        return SourceInfo(
            id="manual_crypto",
            name="Saisie manuelle",
            category=SourceCategory.CRYPTO,
            mode=SourceMode.MANUAL,
            description="Entrez manuellement vos soldes crypto",
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
            self._storage_path.write_text(json.dumps({"assets": [], "version": 1}), encoding="utf-8")

    def _load_data(self) -> dict:
        """Load assets from storage."""
        self._ensure_storage()
        try:
            with self._storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("assets", [])
                    return data
        except Exception as e:
            logger.warning(f"[manual_crypto] failed to load for user={self.user_id}: {e}")
        return {"assets": [], "version": 1}

    def _save_data(self, data: dict) -> None:
        """Save assets to storage (atomic write)."""
        self._ensure_storage()
        temp_path = self._storage_path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(self._storage_path)
            logger.info(f"[manual_crypto] saved {len(data.get('assets', []))} assets for user={self.user_id}")
        except Exception as e:
            logger.error(f"[manual_crypto] failed to save for user={self.user_id}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    # ============ SourceBase Implementation ============

    async def get_balances(self) -> List[BalanceItem]:
        """Return current manual balances."""
        data = self._load_data()
        assets = data.get("assets", [])
        items = []

        for asset in assets:
            value_usd = asset.get("value_usd", 0)
            currency = asset.get("currency", "USD").upper()

            # Convert to USD if not already
            if currency != "USD" and value_usd == 0:
                value = asset.get("amount", 0) * asset.get("price", 0)
                value_usd = fx_convert(value, currency, "USD")

            items.append(
                BalanceItem(
                    symbol=asset.get("symbol", "???"),
                    alias=asset.get("alias", asset.get("symbol", "???")),
                    amount=asset.get("amount", 0),
                    value_usd=value_usd,
                    location=asset.get("location", "Manual Entry"),
                    price_usd=asset.get("price_usd"),
                    currency=currency,
                    asset_class="CRYPTO",
                    source_id="manual_crypto",
                    entry_id=asset.get("id"),
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
            assets = data.get("assets", [])
            return SourceStatus.ACTIVE if assets else SourceStatus.NOT_CONFIGURED
        except Exception:
            return SourceStatus.ERROR

    # ============ CRUD Methods ============

    def list_assets(self) -> List[dict]:
        """List all manual crypto assets."""
        data = self._load_data()
        return data.get("assets", [])

    def get_asset(self, asset_id: str) -> Optional[dict]:
        """Get a specific asset by ID."""
        data = self._load_data()
        for asset in data.get("assets", []):
            if asset.get("id") == asset_id:
                return asset
        return None

    def add_asset(
        self,
        symbol: str,
        amount: float,
        location: str = "Manual Entry",
        value_usd: Optional[float] = None,
        price_usd: Optional[float] = None,
        alias: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict:
        """
        Add a new crypto asset entry.

        Args:
            symbol: Asset symbol (BTC, ETH, etc.)
            amount: Quantity held
            location: Where stored (wallet, exchange)
            value_usd: Total value in USD (optional, can be calculated)
            price_usd: Unit price in USD (optional)
            alias: Display name (defaults to symbol)
            notes: Optional notes

        Returns:
            Created asset dict with ID
        """
        data = self._load_data()
        assets = data.get("assets", [])

        new_asset = {
            "id": str(uuid.uuid4()),
            "symbol": symbol.upper(),
            "alias": alias or symbol.upper(),
            "amount": amount,
            "location": location,
            "value_usd": value_usd or 0,
            "price_usd": price_usd,
            "currency": "USD",
            "notes": notes,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        assets.append(new_asset)
        data["assets"] = assets
        self._save_data(data)

        logger.info(f"[manual_crypto] added {symbol} for user={self.user_id}")
        return new_asset

    def update_asset(self, asset_id: str, **kwargs) -> Optional[dict]:
        """
        Update an existing crypto asset.

        Args:
            asset_id: ID of asset to update
            **kwargs: Fields to update (symbol, amount, location, value_usd, etc.)

        Returns:
            Updated asset dict or None if not found
        """
        data = self._load_data()
        assets = data.get("assets", [])

        for i, asset in enumerate(assets):
            if asset.get("id") == asset_id:
                # Update allowed fields
                allowed_fields = {"symbol", "alias", "amount", "location", "value_usd", "price_usd", "notes"}
                for key, value in kwargs.items():
                    if key in allowed_fields:
                        if key == "symbol":
                            asset[key] = value.upper() if value else asset.get(key)
                        else:
                            asset[key] = value

                asset["updated_at"] = datetime.utcnow().isoformat()
                assets[i] = asset
                data["assets"] = assets
                self._save_data(data)

                logger.info(f"[manual_crypto] updated asset {asset_id} for user={self.user_id}")
                return asset

        logger.warning(f"[manual_crypto] asset not found for update id={asset_id} user={self.user_id}")
        return None

    def delete_asset(self, asset_id: str) -> bool:
        """
        Delete a crypto asset.

        Args:
            asset_id: ID of asset to delete

        Returns:
            True if deleted, False if not found
        """
        data = self._load_data()
        assets = data.get("assets", [])

        initial_count = len(assets)
        filtered = [a for a in assets if a.get("id") != asset_id]

        if len(filtered) == initial_count:
            logger.warning(f"[manual_crypto] asset not found for deletion id={asset_id} user={self.user_id}")
            return False

        data["assets"] = filtered
        self._save_data(data)

        logger.info(f"[manual_crypto] deleted asset {asset_id} for user={self.user_id}")
        return True

    def get_summary(self) -> dict:
        """Get summary of manual crypto holdings."""
        data = self._load_data()
        assets = data.get("assets", [])

        total_usd = sum(a.get("value_usd", 0) for a in assets)

        return {
            "source_id": "manual_crypto",
            "asset_count": len(assets),
            "total_value_usd": total_usd,
            "user_id": self.user_id,
        }
