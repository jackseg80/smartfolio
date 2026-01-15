"""
CoinTracking CSV Source - Wrapper for existing CSV import functionality.

Delegates to existing csv_helpers and sources_resolver logic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from services.sources.base import BalanceItem, SourceBase, SourceInfo
from services.sources.category import SourceCategory, SourceMode, SourceStatus

logger = logging.getLogger(__name__)


class CoinTrackingCSVSource(SourceBase):
    """
    CSV import source for CoinTracking exports.

    Reads data from: data/users/{user_id}/cointracking/data/*.csv
    """

    MODULE_NAME = "cointracking"

    @classmethod
    def get_source_info(cls) -> SourceInfo:
        return SourceInfo(
            id="cointracking_csv",
            name="CoinTracking CSV",
            category=SourceCategory.CRYPTO,
            mode=SourceMode.CSV,
            description="Import depuis fichier CoinTracking",
            icon="upload",
            supports_transactions=True,
            requires_credentials=False,
            file_patterns=["cointracking/data/*.csv"],
        )

    def __init__(self, user_id: str, project_root: str):
        super().__init__(user_id, project_root)
        self._data_dir = Path(project_root) / "data" / "users" / user_id / "cointracking" / "data"

    def _get_csv_files(self) -> List[Path]:
        """Get list of CSV files in data directory."""
        if not self._data_dir.exists():
            return []
        return sorted(self._data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    def _get_selected_file(self) -> Optional[Path]:
        """Get the user-selected or most recent CSV file."""
        import json

        # Check user config for explicit selection
        config_path = Path(self.project_root) / "data" / "users" / self.user_id / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    selected = config.get("csv_selected_file")
                    if selected:
                        # Check in new category-based config
                        sources_config = config.get("sources", {}).get("crypto", {})
                        csv_config = sources_config.get("cointracking_csv", {})
                        selected = csv_config.get("selected_file") or selected

                        selected_path = self._data_dir / selected
                        if selected_path.exists():
                            return selected_path
            except Exception:
                pass

        # Fall back to most recent
        files = self._get_csv_files()
        return files[0] if files else None

    async def get_balances(self) -> List[BalanceItem]:
        """Load balances from CSV file."""
        csv_file = self._get_selected_file()
        if not csv_file:
            logger.warning(f"[cointracking_csv] No CSV files found for user {self.user_id}")
            return []

        try:
            from api.services.csv_helpers import load_csv_balances

            items_raw = await load_csv_balances(str(csv_file))

            items = []
            for r in items_raw:
                items.append(
                    BalanceItem(
                        symbol=r.get("symbol", "???"),
                        alias=r.get("alias", r.get("symbol", "???")),
                        amount=r.get("amount", 0),
                        value_usd=r.get("value_usd", 0),
                        location=r.get("location", "CoinTracking"),
                        asset_class="CRYPTO",
                        source_id="cointracking_csv",
                    )
                )

            logger.info(f"[cointracking_csv] Loaded {len(items)} items for user {self.user_id}")
            return items

        except Exception as e:
            logger.error(f"[cointracking_csv] Error loading CSV for user {self.user_id}: {e}")
            return []

    async def validate_config(self) -> tuple[bool, Optional[str]]:
        """Check if we have valid CSV files."""
        files = self._get_csv_files()
        if not files:
            return False, "Aucun fichier CSV trouvé. Uploadez un export CoinTracking."
        return True, None

    def get_status(self) -> SourceStatus:
        """Check operational status."""
        files = self._get_csv_files()
        if not files:
            return SourceStatus.NOT_CONFIGURED
        return SourceStatus.ACTIVE

    # Additional methods for file management

    def list_files(self) -> List[dict]:
        """List available CSV files with metadata."""
        files = self._get_csv_files()
        result = []
        for f in files:
            stat = f.stat()
            result.append({
                "name": f.name,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
                "is_selected": f == self._get_selected_file(),
            })
        return result

    def get_selected_filename(self) -> Optional[str]:
        """Get the name of the currently selected file."""
        selected = self._get_selected_file()
        return selected.name if selected else None
