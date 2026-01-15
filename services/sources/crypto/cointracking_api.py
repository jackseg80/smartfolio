"""
CoinTracking API Source - Wrapper for existing API connector.

Delegates to connectors.cointracking_api for data fetching.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from services.sources.base import BalanceItem, SourceBase, SourceInfo
from services.sources.category import SourceCategory, SourceMode, SourceStatus

logger = logging.getLogger(__name__)


class CoinTrackingAPISource(SourceBase):
    """
    API source for CoinTracking real-time data.

    Requires API credentials in: data/users/{user_id}/config/secrets.json
    """

    @classmethod
    def get_source_info(cls) -> SourceInfo:
        return SourceInfo(
            id="cointracking_api",
            name="CoinTracking API",
            category=SourceCategory.CRYPTO,
            mode=SourceMode.API,
            description="Synchronisation temps r\u00e9el via API",
            icon="api",
            supports_transactions=True,
            requires_credentials=True,
        )

    def __init__(self, user_id: str, project_root: str):
        super().__init__(user_id, project_root)
        self._secrets_path = Path(project_root) / "data" / "users" / user_id / "config" / "secrets.json"

    def _get_credentials(self) -> tuple[Optional[str], Optional[str]]:
        """Load API credentials from secrets file."""
        if not self._secrets_path.exists():
            return None, None

        try:
            with open(self._secrets_path, "r", encoding="utf-8") as f:
                secrets = json.load(f)
                ct_secrets = secrets.get("cointracking", {})
                return ct_secrets.get("api_key"), ct_secrets.get("api_secret")
        except Exception as e:
            logger.error(f"[cointracking_api] Error loading credentials: {e}")
            return None, None

    async def get_balances(self) -> List[BalanceItem]:
        """Fetch balances from CoinTracking API."""
        api_key, api_secret = self._get_credentials()

        if not (api_key and api_secret):
            logger.warning(f"[cointracking_api] No API credentials for user {self.user_id}")
            return []

        try:
            from connectors.cointracking_api import get_current_balances as ct_get_balances

            logger.info(f"[cointracking_api] Fetching balances for user {self.user_id}")
            api_result = await ct_get_balances(api_key=api_key, api_secret=api_secret)

            items = []
            for r in api_result.get("items", []):
                items.append(
                    BalanceItem(
                        symbol=r.get("symbol", "???"),
                        alias=r.get("alias", r.get("symbol", "???")),
                        amount=r.get("amount", 0),
                        value_usd=r.get("value_usd", 0),
                        location=r.get("location", "CoinTracking"),
                        asset_class="CRYPTO",
                        source_id="cointracking_api",
                    )
                )

            logger.info(f"[cointracking_api] Fetched {len(items)} items for user {self.user_id}")
            return items

        except Exception as e:
            logger.error(f"[cointracking_api] API error for user {self.user_id}: {e}")
            return []

    async def validate_config(self) -> tuple[bool, Optional[str]]:
        """Check if API credentials are configured and valid."""
        api_key, api_secret = self._get_credentials()

        if not api_key:
            return False, "Cl\u00e9 API CoinTracking non configur\u00e9e"
        if not api_secret:
            return False, "Secret API CoinTracking non configur\u00e9"

        # Optionally test the connection
        try:
            from connectors.cointracking_api import get_current_balances as ct_get_balances

            await ct_get_balances(api_key=api_key, api_secret=api_secret)
            return True, None
        except Exception as e:
            return False, f"Erreur de connexion API: {str(e)}"

    def get_status(self) -> SourceStatus:
        """Check operational status."""
        api_key, api_secret = self._get_credentials()

        if not (api_key and api_secret):
            return SourceStatus.NOT_CONFIGURED

        return SourceStatus.ACTIVE

    def has_credentials(self) -> bool:
        """Check if credentials are configured (without validating them)."""
        api_key, api_secret = self._get_credentials()
        return bool(api_key and api_secret)
