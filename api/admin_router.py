"""
Admin Router - Endpoints pour administration système
Gestion utilisateurs, logs, cache, ML models, API keys.

PROTECTION RBAC: Tous les endpoints nécessitent le rôle "admin"
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Dict, Any, Optional
import logging

from api.deps import require_admin_role
from api.utils import success_response, error_response
from api.config.users import get_all_users, clear_users_cache

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={
        403: {"description": "Admin role required"},
        404: {"description": "Resource not found"}
    }
)


# ============================================================================
# Health & Status
# ============================================================================

@router.get("/health")
async def admin_health(user: str = Depends(require_admin_role)):
    """
    Health check pour admin dashboard.
    Vérifie que l'utilisateur a le rôle admin.

    Returns:
        dict: Status admin OK avec user_id
    """
    return success_response({
        "status": "ok",
        "admin_user": user,
        "modules": [
            "user_management",
            "logs_viewer",
            "cache_management",
            "ml_models",
            "api_keys"
        ]
    })


@router.get("/status")
async def admin_status(user: str = Depends(require_admin_role)):
    """
    Status système complet pour admin dashboard.

    Returns:
        dict: Statistiques système (users, logs, cache, ML)
    """
    # Récupérer stats basiques
    all_users = get_all_users()

    return success_response({
        "system": {
            "total_users": len(all_users),
            "active_users": len([u for u in all_users if u.get("status") == "active"]),
            "admin_users": len([u for u in all_users if "admin" in u.get("roles", [])])
        },
        "logs": {
            "available": True,
            "files": ["app.log", "app.log.1", "app.log.2", "app.log.3"]
        },
        "cache": {
            "enabled": True,
            "types": ["in_memory", "coingecko", "crypto_toolbox", "redis"]
        },
        "ml": {
            "models_count": 0,  # TODO: Implémenter comptage réel
            "training_jobs": 0
        }
    })


# ============================================================================
# User Management (Placeholder - Phase 2)
# ============================================================================

@router.get("/users")
async def list_users(user: str = Depends(require_admin_role)):
    """
    Liste tous les utilisateurs du système.

    Returns:
        dict: Liste des utilisateurs avec rôles et statut
    """
    try:
        all_users = get_all_users()

        # Enrichir avec info comptage rôles
        for u in all_users:
            u["roles_count"] = len(u.get("roles", []))

        return success_response(
            all_users,
            meta={"total": len(all_users)}
        )

    except Exception as e:
        logger.error(f"Error listing users: {e}")
        return error_response(
            f"Failed to list users: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============================================================================
# Logs Viewer (Placeholder - Phase 2)
# ============================================================================

@router.get("/logs/list")
async def list_log_files(user: str = Depends(require_admin_role)):
    """
    Liste les fichiers de logs disponibles.

    Returns:
        dict: Liste des fichiers logs
    """
    import os
    from pathlib import Path

    try:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return success_response([])

        log_files = []
        for f in logs_dir.glob("*.log*"):
            stat = f.stat()
            log_files.append({
                "name": f.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime
            })

        # Trier par nom (app.log en premier)
        log_files.sort(key=lambda x: x["name"])

        return success_response(
            log_files,
            meta={"count": len(log_files)}
        )

    except Exception as e:
        logger.error(f"Error listing log files: {e}")
        return error_response(
            f"Failed to list log files: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============================================================================
# Cache Management (Placeholder - Phase 3)
# ============================================================================

@router.get("/cache/stats")
async def get_cache_stats(user: str = Depends(require_admin_role)):
    """
    Statistiques de tous les caches système.

    Returns:
        dict: Stats par type de cache
    """
    # TODO: Implémenter cache manager unifié
    return success_response({
        "in_memory": {
            "keys": 0,
            "size_mb": 0,
            "hit_rate": 0.0
        },
        "coingecko": {
            "keys": 0,
            "ttl_seconds": 900,
            "hit_rate": 0.0
        },
        "redis": {
            "connected": False,
            "memory_mb": 0
        }
    })


@router.delete("/cache/clear")
async def clear_cache(
    cache_type: str = Query("all", description="Cache type (all, in_memory, coingecko, redis)"),
    user: str = Depends(require_admin_role)
):
    """
    Clear cache par type ou tous les caches.

    Args:
        cache_type: Type de cache à clear

    Returns:
        dict: Confirmation clear
    """
    try:
        cleared = []

        if cache_type in ["all", "users"]:
            clear_users_cache()
            cleared.append("users")

        # TODO: Implémenter clear autres caches

        return success_response({
            "cleared_caches": cleared,
            "cache_type": cache_type
        })

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return error_response(
            f"Failed to clear cache: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============================================================================
# ML Models (Placeholder - Phase 3)
# ============================================================================

@router.get("/ml/models")
async def list_ml_models(user: str = Depends(require_admin_role)):
    """
    Liste tous les modèles ML avec versions et statut.

    Returns:
        dict: Liste des modèles ML
    """
    # TODO: Implémenter via ModelRegistry
    return success_response([
        {
            "name": "volatility_forecaster",
            "version": "v2.1",
            "status": "DEPLOYED",
            "accuracy": 0.89,
            "last_trained": "2025-01-15T10:30:00Z"
        }
    ])


# ============================================================================
# API Keys (Placeholder - Phase 4)
# ============================================================================

@router.get("/apikeys")
async def list_api_keys(user: str = Depends(require_admin_role)):
    """
    Liste toutes les clés API (masquées) pour tous les users.

    Returns:
        dict: Keys par user et service
    """
    # TODO: Implémenter lecture secrets.json avec masking
    return success_response({
        "jack": {
            "coingecko": {"status": "valid", "masked_key": "cg_1234...abcd"},
            "cointracking": {"status": "valid", "masked_key": "ct_5678...efgh"}
        }
    })
