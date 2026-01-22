"""
Sources V2 API - Category-based source management endpoints.

New modular sources system with:
- Independent crypto/bourse source selection
- Manual entry CRUD operations
- Source discovery and status
- CSV file management and preview
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.deps import get_active_user
from api.utils import error_response, success_response
from services.sources import SourceCategory, SourceMode, SourceStatus, source_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sources/v2", tags=["Sources V2"])


# ============ Pydantic Models ============


class SourceInfoResponse(BaseModel):
    """Source metadata for API response."""

    id: str
    name: str
    category: str
    mode: str
    description: str
    icon: str
    requires_credentials: bool


class CategorySourcesResponse(BaseModel):
    """Sources available for a category."""

    category: str
    sources: Dict[str, List[SourceInfoResponse]]  # Grouped by mode


class ActiveSourceResponse(BaseModel):
    """Active source for a category."""

    category: str
    active_source: str
    status: str


class SetActiveSourceRequest(BaseModel):
    """Request to change active source."""

    source_id: str = Field(..., description="Source ID to activate")


class ManualCryptoAssetInput(BaseModel):
    """Input for creating/updating a manual crypto asset."""

    symbol: str = Field(..., min_length=1, max_length=10)
    amount: float = Field(..., ge=0)
    location: str = Field("Manual Entry", max_length=100)
    value_usd: Optional[float] = Field(None, ge=0)
    price_usd: Optional[float] = Field(None, ge=0)
    alias: Optional[str] = Field(None, max_length=50)
    notes: Optional[str] = Field(None, max_length=500)


class ManualBoursePositionInput(BaseModel):
    """Input for creating/updating a manual bourse position."""

    symbol: str = Field(..., min_length=1, max_length=20)
    quantity: float = Field(..., ge=0)
    value: float = Field(..., ge=0)
    currency: str = Field("USD", max_length=3)
    name: Optional[str] = Field(None, max_length=100)
    isin: Optional[str] = Field(None, max_length=12)
    asset_class: str = Field("EQUITY", max_length=20)
    broker: str = Field("Manual Entry", max_length=100)
    avg_price: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = Field(None, max_length=500)


class CSVFileInfo(BaseModel):
    """Information about a CSV file."""

    filename: str
    filepath: str
    size_bytes: int
    modified_at: str
    is_active: bool
    row_count: Optional[int] = None
    total_value_usd: Optional[float] = None


class CSVPreviewResponse(BaseModel):
    """Preview of CSV data."""

    filename: str
    rows: List[Dict[str, Any]]
    total_rows: int
    columns: List[str]
    validation: Dict[str, Any]
    summary: Dict[str, Any]


# ============ Helper Functions ============


def _get_project_root() -> str:
    """Get project root path."""
    return str(Path(__file__).parent.parent)


def _load_user_sources_config(user_id: str) -> dict:
    """Load user's sources configuration."""
    import json

    config_path = Path(_get_project_root()) / "data" / "users" / user_id / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_user_sources_config(user_id: str, config: dict) -> None:
    """Save user's sources configuration."""
    import json

    config_path = Path(_get_project_root()) / "data" / "users" / user_id / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = config_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    temp_path.replace(config_path)


def _ensure_category_config(config: dict, category: str) -> dict:
    """Ensure sources config structure exists for a category."""
    if "sources" not in config:
        config["sources"] = {}
    if category not in config["sources"]:
        default_source = f"manual_{category}"
        config["sources"][category] = {
            "active_source": default_source,
        }
    return config


def _get_csv_directory(user_id: str, source_type: str) -> Path:
    """Get CSV directory for a source type (cointracking or saxobank)."""
    root = Path(_get_project_root())
    return root / "data" / "users" / user_id / source_type / "data"


def _list_csv_files(user_id: str, source_type: str) -> List[Dict[str, Any]]:
    """List all CSV files for a source type, sorted by modification time (newest first)."""
    csv_dir = _get_csv_directory(user_id, source_type)

    if not csv_dir.exists():
        return []

    files = []
    for file in csv_dir.glob("*.csv"):
        stat = file.stat()
        files.append({
            "filename": file.name,
            "filepath": str(file),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_active": False,  # Will be updated below
        })

    # Sort by modification time (newest first)
    files.sort(key=lambda f: f["modified_at"], reverse=True)

    # Mark the most recent file as active (if any)
    if files:
        files[0]["is_active"] = True

    return files


def _preview_csv_file(filepath: str, max_rows: int = 10) -> Dict[str, Any]:
    """Preview a CSV file with validation and summary."""
    import csv

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []

            rows = []
            total_value = 0.0
            for i, row in enumerate(reader):
                if i < max_rows:
                    rows.append(dict(row))

                # Try to sum total value (for summary)
                if "Total" in row:
                    try:
                        total_value += float(row["Total"])
                    except (ValueError, TypeError):
                        pass

            total_rows = i + 1  # Total rows processed

            # Validation
            validation = {
                "is_valid": len(columns) > 0,
                "has_headers": len(columns) > 0,
                "warnings": [],
            }

            # Check for required columns (basic validation)
            if "Currency" not in columns:
                validation["warnings"].append("Missing 'Currency' column")
            if "Amount" not in columns and "Quantity" not in columns:
                validation["warnings"].append("Missing 'Amount' or 'Quantity' column")

            # Summary
            summary = {
                "total_rows": total_rows,
                "total_value_usd": round(total_value, 2) if total_value > 0 else None,
                "preview_rows": len(rows),
            }

            return {
                "filename": Path(filepath).name,
                "rows": rows,
                "total_rows": total_rows,
                "columns": columns,
                "validation": validation,
                "summary": summary,
            }

    except Exception as e:
        logger.error(f"Error previewing CSV {filepath}: {e}")
        return {
            "filename": Path(filepath).name,
            "rows": [],
            "total_rows": 0,
            "columns": [],
            "validation": {"is_valid": False, "warnings": [str(e)]},
            "summary": {},
        }


# ============ Source Discovery Endpoints ============


@router.get("/available")
async def list_available_sources(
    category: Optional[str] = Query(None, description="Filter by category (crypto, bourse)"),
    user: str = Depends(get_active_user),
):
    """
    List all available sources, optionally filtered by category.

    Returns source metadata for UI display.
    """
    cat_enum = None
    if category:
        try:
            cat_enum = SourceCategory(category)
        except ValueError:
            return error_response(f"Invalid category: {category}. Use 'crypto' or 'bourse'", code=400)

    sources = source_registry.list_sources(cat_enum)

    return success_response(
        [
            {
                "id": s.id,
                "name": s.name,
                "category": s.category.value,
                "mode": s.mode.value,
                "description": s.description,
                "icon": s.icon,
                "requires_credentials": s.requires_credentials,
            }
            for s in sources
        ]
    )


@router.get("/categories")
async def list_categories(user: str = Depends(get_active_user)):
    """
    List source categories with their available sources grouped by mode.

    Useful for building the settings UI.
    """
    result = {}

    for category in SourceCategory:
        sources_by_mode = source_registry.get_sources_by_category(category)
        result[category.value] = {
            mode.value: [{"id": s.id, "name": s.name, "icon": s.icon} for s in source_list]
            for mode, source_list in sources_by_mode.items()
            if source_list  # Only include modes with sources
        }

    return success_response(result)


# ============ Active Source Management ============


@router.get("/{category}/active")
async def get_active_source(
    category: str,
    user: str = Depends(get_active_user),
):
    """
    Get the currently active source for a category.

    Returns the active source ID and its current status.
    """
    try:
        cat_enum = SourceCategory(category)
    except ValueError:
        return error_response(f"Invalid category: {category}", code=400)

    config = _load_user_sources_config(user)
    config = _ensure_category_config(config, category)

    active_source_id = config["sources"][category].get("active_source", f"manual_{category}")

    # Get source status
    source = source_registry.get_source(active_source_id, user, _get_project_root())
    status = source.get_status().value if source else "not_found"

    return success_response(
        {
            "category": category,
            "active_source": active_source_id,
            "status": status,
        }
    )


@router.put("/{category}/active")
async def set_active_source(
    category: str,
    request: SetActiveSourceRequest,
    user: str = Depends(get_active_user),
):
    """
    Set the active source for a category (mutually exclusive).

    Only one source can be active per category at a time.
    """
    try:
        cat_enum = SourceCategory(category)
    except ValueError:
        return error_response(f"Invalid category: {category}", code=400)

    # Validate source exists and belongs to this category
    source_class = source_registry.get_source_class(request.source_id)
    if not source_class:
        return error_response(f"Source not found: {request.source_id}", code=404)

    source_info = source_class.get_source_info()
    if source_info.category != cat_enum:
        return error_response(
            f"Source {request.source_id} belongs to category {source_info.category.value}, not {category}",
            code=400,
        )

    # Update config
    config = _load_user_sources_config(user)
    config = _ensure_category_config(config, category)

    # Get old source for history logging
    old_source = config["sources"][category].get("active_source", "none")

    config["sources"][category]["active_source"] = request.source_id

    # ✅ FIX: Only force category_based mode for manual sources
    # For CSV/API sources, preserve the existing data_source setting to maintain V1 compatibility
    if request.source_id.startswith("manual_"):
        # Manual sources require category_based mode
        if config.get("data_source") != "category_based":
            config["data_source"] = "category_based"
    # For non-manual sources (CSV/API), keep data_source as-is to allow V1 mode to work

    _save_user_sources_config(user, config)

    # Log source change to history (only if actually changed)
    if old_source != request.source_id:
        _log_source_change(user, category, old_source, request.source_id)

    logger.info(f"[sources_v2] User {user} set {category} source to {request.source_id}")

    return success_response(
        {
            "category": category,
            "active_source": request.source_id,
            "message": f"Source changed to {request.source_id}",
        }
    )


# ============ Manual Crypto CRUD ============


@router.get("/crypto/manual/assets")
async def list_manual_crypto_assets(user: str = Depends(get_active_user)):
    """List all manual crypto asset entries."""
    source = source_registry.get_source("manual_crypto", user, _get_project_root())
    if not source:
        return error_response("Manual crypto source not available", code=500)

    # Access the ManualCryptoSource methods
    from services.sources.crypto.manual import ManualCryptoSource

    if isinstance(source, ManualCryptoSource):
        assets = source.list_assets()
        return success_response({"assets": assets, "count": len(assets)})

    return error_response("Invalid source type", code=500)


@router.post("/crypto/manual/assets")
async def add_manual_crypto_asset(
    asset: ManualCryptoAssetInput,
    user: str = Depends(get_active_user),
):
    """Add a new manual crypto asset entry."""
    source = source_registry.get_source("manual_crypto", user, _get_project_root())
    if not source:
        return error_response("Manual crypto source not available", code=500)

    from services.sources.crypto.manual import ManualCryptoSource

    if isinstance(source, ManualCryptoSource):
        result = source.add_asset(
            symbol=asset.symbol,
            amount=asset.amount,
            location=asset.location,
            value_usd=asset.value_usd,
            price_usd=asset.price_usd,
            alias=asset.alias,
            notes=asset.notes,
        )
        return success_response({"asset": result, "message": "Asset added successfully"})

    return error_response("Invalid source type", code=500)


@router.put("/crypto/manual/assets/{asset_id}")
async def update_manual_crypto_asset(
    asset_id: str,
    asset: ManualCryptoAssetInput,
    user: str = Depends(get_active_user),
):
    """Update an existing manual crypto asset."""
    source = source_registry.get_source("manual_crypto", user, _get_project_root())
    if not source:
        return error_response("Manual crypto source not available", code=500)

    from services.sources.crypto.manual import ManualCryptoSource

    if isinstance(source, ManualCryptoSource):
        result = source.update_asset(
            asset_id,
            symbol=asset.symbol,
            amount=asset.amount,
            location=asset.location,
            value_usd=asset.value_usd,
            price_usd=asset.price_usd,
            alias=asset.alias,
            notes=asset.notes,
        )
        if result:
            return success_response({"asset": result, "message": "Asset updated successfully"})
        return error_response("Asset not found", code=404)

    return error_response("Invalid source type", code=500)


@router.delete("/crypto/manual/assets/{asset_id}")
async def delete_manual_crypto_asset(
    asset_id: str,
    user: str = Depends(get_active_user),
):
    """Delete a manual crypto asset."""
    source = source_registry.get_source("manual_crypto", user, _get_project_root())
    if not source:
        return error_response("Manual crypto source not available", code=500)

    from services.sources.crypto.manual import ManualCryptoSource

    if isinstance(source, ManualCryptoSource):
        if source.delete_asset(asset_id):
            return success_response({"message": "Asset deleted successfully"})
        return error_response("Asset not found", code=404)

    return error_response("Invalid source type", code=500)


# ============ Manual Bourse CRUD ============


@router.get("/bourse/manual/positions")
async def list_manual_bourse_positions(user: str = Depends(get_active_user)):
    """List all manual bourse position entries."""
    source = source_registry.get_source("manual_bourse", user, _get_project_root())
    if not source:
        return error_response("Manual bourse source not available", code=500)

    from services.sources.bourse.manual import ManualBourseSource

    if isinstance(source, ManualBourseSource):
        positions = source.list_positions()
        return success_response({"positions": positions, "count": len(positions)})

    return error_response("Invalid source type", code=500)


@router.post("/bourse/manual/positions")
async def add_manual_bourse_position(
    position: ManualBoursePositionInput,
    user: str = Depends(get_active_user),
):
    """Add a new manual bourse position entry."""
    source = source_registry.get_source("manual_bourse", user, _get_project_root())
    if not source:
        return error_response("Manual bourse source not available", code=500)

    from services.sources.bourse.manual import ManualBourseSource

    if isinstance(source, ManualBourseSource):
        result = source.add_position(
            symbol=position.symbol,
            quantity=position.quantity,
            value=position.value,
            currency=position.currency,
            name=position.name,
            isin=position.isin,
            asset_class=position.asset_class,
            broker=position.broker,
            avg_price=position.avg_price,
            notes=position.notes,
        )
        return success_response({"position": result, "message": "Position added successfully"})

    return error_response("Invalid source type", code=500)


@router.put("/bourse/manual/positions/{position_id}")
async def update_manual_bourse_position(
    position_id: str,
    position: ManualBoursePositionInput,
    user: str = Depends(get_active_user),
):
    """Update an existing manual bourse position."""
    source = source_registry.get_source("manual_bourse", user, _get_project_root())
    if not source:
        return error_response("Manual bourse source not available", code=500)

    from services.sources.bourse.manual import ManualBourseSource

    if isinstance(source, ManualBourseSource):
        result = source.update_position(
            position_id,
            symbol=position.symbol,
            quantity=position.quantity,
            value=position.value,
            currency=position.currency,
            name=position.name,
            isin=position.isin,
            asset_class=position.asset_class,
            broker=position.broker,
            avg_price=position.avg_price,
            notes=position.notes,
        )
        if result:
            return success_response({"position": result, "message": "Position updated successfully"})
        return error_response("Position not found", code=404)

    return error_response("Invalid source type", code=500)


@router.delete("/bourse/manual/positions/{position_id}")
async def delete_manual_bourse_position(
    position_id: str,
    user: str = Depends(get_active_user),
):
    """Delete a manual bourse position."""
    source = source_registry.get_source("manual_bourse", user, _get_project_root())
    if not source:
        return error_response("Manual bourse source not available", code=500)

    from services.sources.bourse.manual import ManualBourseSource

    if isinstance(source, ManualBourseSource):
        if source.delete_position(position_id):
            return success_response({"message": "Position deleted successfully"})
        return error_response("Position not found", code=404)

    return error_response("Invalid source type", code=500)


# ============ Balances Endpoint ============


@router.get("/{category}/balances")
async def get_category_balances(
    category: str,
    source: Optional[str] = Query(None, description="Optional source ID to override active source"),
    user: str = Depends(get_active_user),
):
    """
    Get balances from the active source for a category (or override with specific source).

    Returns standardized balance items.
    """
    try:
        cat_enum = SourceCategory(category)
    except ValueError:
        return error_response(f"Invalid category: {category}", code=400)

    # Get source ID (use override if provided, otherwise active source)
    if source:
        source_id = source
    else:
        config = _load_user_sources_config(user)
        config = _ensure_category_config(config, category)
        source_id = config["sources"][category].get("active_source", f"manual_{category}")

    # Get source instance
    source_instance = source_registry.get_source(source_id, user, _get_project_root())
    if not source_instance:
        return error_response(f"Source not found: {source_id}", code=404)

    try:
        balances = await source_instance.get_balances()
        return success_response(
            {
                "category": category,
                "source_id": source_id,
                "items": [b.to_dict() for b in balances],
                "count": len(balances),
            }
        )
    except Exception as e:
        logger.error(f"[sources_v2] Error getting balances from {source_id}: {e}")
        return error_response(f"Error fetching balances: {str(e)}", code=500)


# ============ Summary Endpoint ============


@router.get("/summary")
async def get_sources_summary(user: str = Depends(get_active_user)):
    """
    Get summary of all source categories.

    Returns active source and status for each category.
    """
    config = _load_user_sources_config(user)

    result = {}
    for category in SourceCategory:
        cat_key = category.value
        config = _ensure_category_config(config, cat_key)

        active_source_id = config["sources"][cat_key].get("active_source", f"manual_{cat_key}")
        source = source_registry.get_source(active_source_id, user, _get_project_root())

        result[cat_key] = {
            "active_source": active_source_id,
            "status": source.get_status().value if source else "not_found",
            "available_sources": [s.id for s in source_registry.list_sources(category)],
        }

    return success_response(result)


# ============ CSV File Management Endpoints ============


@router.get("/{category}/csv/files")
async def list_csv_files(
    category: str,
    user: str = Depends(get_active_user),
):
    """
    List all CSV files for a category (cointracking for crypto, saxobank for bourse).

    Returns file metadata including size, modification time, and active status.
    """
    # Map category to source type
    source_type_map = {
        "crypto": "cointracking",
        "bourse": "saxobank",
    }

    source_type = source_type_map.get(category)
    if not source_type:
        return error_response(f"Invalid category: {category}. Use 'crypto' or 'bourse'", code=400)

    files = _list_csv_files(user, source_type)

    return success_response({
        "category": category,
        "source_type": source_type,
        "files": files,
        "count": len(files),
    })


@router.get("/{category}/csv/preview")
async def preview_csv_file(
    category: str,
    filename: str = Query(..., description="Filename to preview"),
    max_rows: int = Query(10, ge=1, le=50, description="Maximum rows to preview"),
    user: str = Depends(get_active_user),
):
    """
    Preview a CSV file with validation and summary.

    Returns first N rows, column headers, validation status, and summary statistics.
    """
    # Map category to source type
    source_type_map = {
        "crypto": "cointracking",
        "bourse": "saxobank",
    }

    source_type = source_type_map.get(category)
    if not source_type:
        return error_response(f"Invalid category: {category}", code=400)

    # Get file path
    csv_dir = _get_csv_directory(user, source_type)
    filepath = csv_dir / filename

    # Security: Ensure file is within expected directory
    if not filepath.is_relative_to(csv_dir):
        return error_response("Invalid file path", code=400)

    if not filepath.exists():
        return error_response(f"File not found: {filename}", code=404)

    # Preview the file
    preview = _preview_csv_file(str(filepath), max_rows)

    return success_response(preview)


@router.delete("/{category}/csv/files/{filename}")
async def delete_csv_file(
    category: str,
    filename: str,
    user: str = Depends(get_active_user),
):
    """
    Delete a CSV file.

    Warning: This action is irreversible. The active file cannot be deleted.
    """
    # Map category to source type
    source_type_map = {
        "crypto": "cointracking",
        "bourse": "saxobank",
    }

    source_type = source_type_map.get(category)
    if not source_type:
        return error_response(f"Invalid category: {category}", code=400)

    # Get file path
    csv_dir = _get_csv_directory(user, source_type)
    filepath = csv_dir / filename

    # Security: Ensure file is within expected directory
    if not filepath.is_relative_to(csv_dir):
        return error_response("Invalid file path", code=400)

    if not filepath.exists():
        return error_response(f"File not found: {filename}", code=404)

    # Check if it's the most recent file (active)
    files = _list_csv_files(user, source_type)
    if files and files[0]["filename"] == filename:
        return error_response("Cannot delete the active file. Upload a newer file first.", code=400)

    # Delete the file
    try:
        filepath.unlink()
        logger.info(f"[sources_v2] User {user} deleted CSV file: {filename}")
        return success_response({"message": f"File {filename} deleted successfully"})
    except Exception as e:
        logger.error(f"[sources_v2] Error deleting file {filename}: {e}")
        return error_response(f"Error deleting file: {str(e)}", code=500)


@router.get("/{category}/csv/download/{filename}")
async def download_csv_file(
    category: str,
    filename: str,
    user: str = Depends(get_active_user),
):
    """
    Download a CSV file.

    Returns the file as a downloadable response.
    """
    # Map category to source type
    source_type_map = {
        "crypto": "cointracking",
        "bourse": "saxobank",
    }

    source_type = source_type_map.get(category)
    if not source_type:
        return error_response(f"Invalid category: {category}", code=400)

    # Get file path
    csv_dir = _get_csv_directory(user, source_type)
    filepath = csv_dir / filename

    # Security: Ensure file is within expected directory
    if not filepath.is_relative_to(csv_dir):
        return error_response("Invalid file path", code=400)

    if not filepath.exists():
        return error_response(f"File not found: {filename}", code=404)

    # Return file response
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type="text/csv",
    )


# ============ Source Change History Endpoints ============


def _get_history_file(user_id: str) -> Path:
    """Get path to source change history file."""
    root = Path(_get_project_root())
    history_dir = root / "data" / "users" / user_id / "sources"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / "change_history.json"


def _log_source_change(user_id: str, category: str, old_source: str, new_source: str) -> None:
    """Log a source change to history."""
    import json

    history_file = _get_history_file(user_id)

    # Load existing history
    history = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"Error reading history file: {e}")
            history = []

    # Add new entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "old_source": old_source,
        "new_source": new_source,
    }
    history.insert(0, entry)  # Add to beginning

    # Keep only last 50 entries
    history = history[:50]

    # Save updated history
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error writing history file: {e}")


@router.get("/{category}/history")
async def get_source_history(
    category: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of entries"),
    user: str = Depends(get_active_user),
):
    """
    Get source change history for a category.

    Returns the most recent source changes with timestamps.
    """
    import json

    history_file = _get_history_file(user)

    if not history_file.exists():
        return success_response([])

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)

        # Filter by category and apply limit
        filtered = [entry for entry in history if entry.get("category") == category]
        filtered = filtered[:limit]

        return success_response(filtered)

    except Exception as e:
        logger.error(f"Error reading history: {e}")
        return error_response(f"Error reading history: {str(e)}", code=500)
