"""
SaxoBank CSV Source - Wrapper for existing Saxo Bank CSV import.

Reads position data from Saxo Bank export files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from services.fx_service import convert as fx_convert
from services.sources.base import BalanceItem, SourceBase, SourceInfo
from services.sources.category import SourceCategory, SourceMode, SourceStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bilingual header mapping (French Saxo Bank CSV exports → canonical English)
# Keys = French column names, Values = canonical English equivalents.
# English headers map to themselves so the dict is idempotent.
# ---------------------------------------------------------------------------
HEADER_ALIASES: dict[str, str] = {
    # --- identity / instrument ---
    "Instruments": "Instruments",
    "Instrument": "Instrument",
    "Symbole": "Symbol",
    "Symbol": "Symbol",
    "Ticker": "Ticker",
    "ISIN": "ISIN",
    "Isin": "Isin",
    "Description": "Description",
    "Name": "Name",
    "Émetteur": "Issuer",
    "Issuer": "Issuer",
    # --- status ---
    "Statut": "Status",
    "Status": "Status",
    # --- quantities & prices ---
    "Quantité": "Quantity",
    "Quantity": "Quantity",
    "Qty": "Qty",
    "Amount": "Amount",
    "Prix entrée": "Entry Price",
    "Entry Price": "Entry Price",
    # --- values ---
    "Valeur actuelle (EUR)": "Current Value (EUR)",
    "Current Value (EUR)": "Current Value (EUR)",
    "Valeur actuelle (USD)": "Current Value (USD)",
    "Current Value (USD)": "Current Value (USD)",
    "Val. actuelle": "Current Value",
    "Current Value": "Current Value",
    "Market Value": "Market Value",
    "MarketValue": "MarketValue",
    "Value": "Value",
    "Valeur": "Value",
    # --- currency ---
    "Devise": "Currency",
    "Currency": "Currency",
    # --- interest ---
    "Intérêts courus": "Accrued Interest",
    "Accrued Interest": "Accrued Interest",
    # --- asset type ---
    "AssetType": "AssetType",
    "Type": "Type",
}


def _normalize_headers(row: dict) -> dict:
    """Normalize CSV row keys: accept both French and English headers.

    Unknown keys are passed through unchanged so that no data is lost.
    """
    return {HEADER_ALIASES.get(k, k): v for k, v in row.items()}


class SaxoBankCSVSource(SourceBase):
    """
    CSV import source for SaxoBank position exports.

    Reads data from: data/users/{user_id}/saxobank/data/*.csv
    """

    MODULE_NAME = "saxobank"

    @classmethod
    def get_source_info(cls) -> SourceInfo:
        return SourceInfo(
            id="saxobank_csv",
            name="Saxo Bank CSV",
            category=SourceCategory.BOURSE,
            mode=SourceMode.CSV,
            description="Import from Saxo Bank file",
            icon="upload",
            supports_transactions=False,
            requires_credentials=False,
            file_patterns=["saxobank/data/*.csv"],
        )

    def __init__(self, user_id: str, project_root: str):
        super().__init__(user_id, project_root)
        self._data_dir = Path(project_root) / "data" / "users" / user_id / "saxobank" / "data"

    def _get_csv_files(self) -> List[Path]:
        """Get list of CSV files in data directory."""
        if not self._data_dir.exists():
            return []
        # Include both .csv and .json files (Saxo exports can be in either format)
        csv_files = list(self._data_dir.glob("*.csv"))
        json_files = list(self._data_dir.glob("*.json"))
        all_files = csv_files + json_files
        return sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)

    def _get_selected_file(self) -> Optional[Path]:
        """Get the user-selected or most recent file."""
        # Check user config for explicit selection
        config_path = Path(self.project_root) / "data" / "users" / self.user_id / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                    # Check in Sources V2 category-based config first
                    sources_config = config.get("sources", {}).get("bourse", {})
                    selected = sources_config.get("selected_csv_file")

                    # Fall back to legacy config locations
                    if not selected:
                        csv_config = sources_config.get("saxobank_csv", {})
                        selected = csv_config.get("selected_file")

                    if selected:
                        selected_path = self._data_dir / selected
                        if selected_path.exists():
                            return selected_path
            except Exception:
                pass

        # Fall back to most recent
        files = self._get_csv_files()
        return files[0] if files else None

    async def get_balances(self) -> List[BalanceItem]:
        """Load positions from CSV/JSON file."""
        data_file = self._get_selected_file()
        if not data_file:
            logger.warning(f"[saxobank_csv] No data files found for user {self.user_id}")
            return []

        try:
            items = []

            if data_file.suffix == ".json":
                items = await self._load_json_positions(data_file)
            else:
                items = await self._load_csv_positions(data_file)

            logger.info(f"[saxobank_csv] Loaded {len(items)} positions for user {self.user_id}")
            return items

        except Exception as e:
            logger.error(f"[saxobank_csv] Error loading data for user {self.user_id}: {e}")
            return []

    async def _load_json_positions(self, file_path: Path) -> List[BalanceItem]:
        """Load positions from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = []
        positions = data if isinstance(data, list) else data.get("positions", [])

        for pos in positions:
            # Handle various Saxo JSON formats
            symbol = pos.get("Symbol") or pos.get("symbol") or pos.get("AssetType", "???")
            name = pos.get("Description") or pos.get("name") or pos.get("InstrumentDescription", symbol)
            quantity = float(pos.get("Amount") or pos.get("quantity") or pos.get("Quantity", 0))
            value = float(pos.get("MarketValue") or pos.get("value") or pos.get("CurrentValue", 0))
            currency = pos.get("Currency") or pos.get("currency") or "USD"

            # Convert to USD
            value_usd = fx_convert(value, currency.upper(), "USD") if currency.upper() != "USD" else value

            items.append(
                BalanceItem(
                    symbol=symbol,
                    alias=name,
                    amount=quantity,
                    value_usd=value_usd,
                    location="Saxo Bank",
                    currency=currency.upper(),
                    asset_class=self._detect_asset_class(pos),
                    isin=pos.get("Isin") or pos.get("isin"),
                    instrument_name=name,
                    source_id="saxobank_csv",
                )
            )

        return items

    async def _load_csv_positions(self, file_path: Path) -> List[BalanceItem]:
        """Load positions from CSV file."""
        import csv

        items = []

        with open(file_path, "r", encoding="utf-8-sig") as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            reader = csv.DictReader(f, dialect=dialect)

            for raw_row in reader:
                # Normalize headers: translate French → canonical English
                row = _normalize_headers(raw_row)

                # Handle various CSV column names (now all in canonical English)
                symbol = (
                    row.get("Symbol")
                    or row.get("Ticker")
                    or row.get("ISIN", "???")
                )
                name = (
                    row.get("Description")
                    or row.get("Instrument")
                    or row.get("Name")
                    or symbol
                )
                quantity_str = (
                    row.get("Amount")
                    or row.get("Quantity")
                    or row.get("Qty")
                    or "0"
                )
                # Get value and detect currency from column name
                value_column_eur = row.get("Current Value (EUR)")
                value_column_usd = row.get("Current Value (USD)")
                value_column_generic = (
                    row.get("Current Value")
                    or row.get("Market Value")
                    or row.get("MarketValue")
                    or row.get("Value")
                )

                # Determine value and currency based on which column has data
                if value_column_eur:
                    value_str = value_column_eur
                    value_currency = "EUR"
                elif value_column_usd:
                    value_str = value_column_usd
                    value_currency = "USD"
                else:
                    value_str = value_column_generic or "0"
                    # Fallback to Currency column for generic value columns
                    value_currency = row.get("Currency") or "USD"

                # Parse numeric values (handle European number format)
                quantity = self._parse_number(quantity_str)
                value = self._parse_number(value_str)

                # Convert to USD using the correct currency
                value_usd = fx_convert(value, value_currency.upper(), "USD") if value_currency.upper() != "USD" else value

                if quantity != 0:  # Skip zero positions
                    # Get instrument currency (different from value currency)
                    instrument_currency = row.get("Currency") or value_currency

                    items.append(
                        BalanceItem(
                            symbol=symbol,
                            alias=name,
                            amount=quantity,
                            value_usd=value_usd,
                            location="Saxo Bank",
                            currency=instrument_currency.upper(),
                            asset_class=row.get("AssetType", "EQUITY"),
                            isin=row.get("ISIN") or row.get("Isin"),
                            instrument_name=name,
                            source_id="saxobank_csv",
                        )
                    )

        return items

    def _parse_number(self, value: str) -> float:
        """Parse number handling both US and European formats."""
        if not value:
            return 0.0
        try:
            # Remove currency symbols and whitespace
            cleaned = value.strip().replace("$", "").replace("€", "").replace(" ", "")
            # Handle European format (1.234,56 -> 1234.56)
            if "," in cleaned and "." in cleaned:
                if cleaned.index(",") > cleaned.index("."):
                    cleaned = cleaned.replace(".", "").replace(",", ".")
            elif "," in cleaned:
                cleaned = cleaned.replace(",", ".")
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0

    def _detect_asset_class(self, pos: dict) -> str:
        """Detect asset class from position data."""
        asset_type = (
            pos.get("AssetType")
            or pos.get("asset_class")
            or pos.get("Type")
            or ""
        ).upper()

        if "ETF" in asset_type:
            return "ETF"
        elif "BOND" in asset_type or "OBLIGATION" in asset_type:
            return "BOND"
        elif "FUND" in asset_type or "FCP" in asset_type:
            return "FUND"
        elif "STOCK" in asset_type or "EQUITY" in asset_type or "ACTION" in asset_type:
            return "EQUITY"
        elif "CFD" in asset_type:
            return "CFD"
        else:
            return "EQUITY"  # Default

    async def validate_config(self) -> tuple[bool, Optional[str]]:
        """Check if we have valid data files."""
        files = self._get_csv_files()
        if not files:
            return False, "Aucun fichier trouvé. Uploadez un export Saxo Bank."
        return True, None

    def get_status(self) -> SourceStatus:
        """Check operational status."""
        files = self._get_csv_files()
        if not files:
            return SourceStatus.NOT_CONFIGURED
        return SourceStatus.ACTIVE

    def list_files(self) -> List[dict]:
        """List available data files with metadata."""
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
