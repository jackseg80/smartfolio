"""
Sources V2 API - Category-based source management endpoints.

New modular sources system with:
- Independent crypto/bourse source selection
- Manual entry CRUD operations
- Source discovery and status
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
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
    config["sources"][category]["active_source"] = request.source_id

    # ✅ FIX: Only force category_based mode for manual sources
    # For CSV/API sources, preserve the existing data_source setting to maintain V1 compatibility
    if request.source_id.startswith("manual_"):
        # Manual sources require category_based mode
        if config.get("data_source") != "category_based":
            config["data_source"] = "category_based"
    # For non-manual sources (CSV/API), keep data_source as-is to allow V1 mode to work

    _save_user_sources_config(user, config)

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
    user: str = Depends(get_active_user),
):
    """
    Get balances from the active source for a category.

    Returns standardized balance items.
    """
    try:
        cat_enum = SourceCategory(category)
    except ValueError:
        return error_response(f"Invalid category: {category}", code=400)

    # Get active source for category
    config = _load_user_sources_config(user)
    config = _ensure_category_config(config, category)
    active_source_id = config["sources"][category].get("active_source", f"manual_{category}")

    # Get source instance
    source = source_registry.get_source(active_source_id, user, _get_project_root())
    if not source:
        return error_response(f"Source not found: {active_source_id}", code=404)

    try:
        balances = await source.get_balances()
        return success_response(
            {
                "category": category,
                "source_id": active_source_id,
                "items": [b.to_dict() for b in balances],
                "count": len(balances),
            }
        )
    except Exception as e:
        logger.error(f"[sources_v2] Error getting balances from {active_source_id}: {e}")
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
