"""
Admin Router - Endpoints pour administration système
Gestion utilisateurs, logs, cache, ML models, API keys.

PROTECTION RBAC: Tous les endpoints nécessitent le rôle "admin"
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query, status, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from api.deps import require_admin_role
from api.utils import success_response, error_response
from api.config.users import get_all_users, clear_users_cache
from services.user_management import get_user_management_service
from services.log_reader import get_log_reader
from services.cache_manager import cache_manager
from services.ml.training_executor import training_executor, JobStatus

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
# Pydantic Models
# ============================================================================

class CreateUserRequest(BaseModel):
    """Request model pour création utilisateur"""
    user_id: str = Field(..., min_length=1, max_length=50, description="User ID (alphanumeric + underscore)")
    label: str = Field(..., min_length=1, max_length=100, description="Display label")
    password: Optional[str] = Field(None, min_length=8, description="Initial password (optional, min 8 characters)")
    roles: List[str] = Field(default=["viewer"], description="User roles")


class UpdateUserRequest(BaseModel):
    """Request model pour mise à jour utilisateur"""
    label: Optional[str] = Field(None, min_length=1, max_length=100)
    roles: Optional[List[str]] = None
    status: Optional[str] = Field(None, pattern="^(active|inactive)$")


class ResetPasswordRequest(BaseModel):
    """Request model pour reset password (admin uniquement)"""
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")


class AssignRolesRequest(BaseModel):
    """Request model pour assignation rôles"""
    roles: List[str] = Field(..., min_items=1, description="Roles to assign")


class TrainingConfig(BaseModel):
    """Configuration pour training ML models (Phase 2)"""
    # Data configuration
    days: int = Field(730, ge=90, le=7300, description="Data history (days): 90-7300 (20 years max)")
    train_val_split: float = Field(0.8, ge=0.6, le=0.9, description="Train/Val split: 0.6-0.9")

    # Common hyperparameters
    epochs: int = Field(100, ge=10, le=500, description="Number of epochs: 10-500")
    patience: int = Field(15, ge=5, le=50, description="Early stopping patience: 5-50")
    batch_size: int = Field(32, ge=8, le=256, description="Batch size: 8-256")
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1, description="Learning rate: 0.0001-0.1")

    # Volatility-specific params
    hidden_size: Optional[int] = Field(64, ge=32, le=256, description="Hidden layer size (volatility): 32-256")
    min_r2: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Min R² threshold (volatility): 0.0-1.0")

    # Model-specific override
    symbols: Optional[List[str]] = Field(None, description="Symbols for volatility models (ex: ['BTC', 'ETH', 'SOL'])")


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
# User Management - CRUD Operations (Phase 2)
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


@router.post("/users")
async def create_user(
    request: CreateUserRequest,
    user: str = Depends(require_admin_role)
):
    """
    Créer un nouvel utilisateur avec structure de dossiers complète.

    Args:
        request: Données utilisateur (user_id, label, password, roles)

    Returns:
        dict: Utilisateur créé
    """
    try:
        service = get_user_management_service()

        # Créer l'utilisateur avec la structure de base
        new_user = service.create_user(
            user_id=request.user_id,
            label=request.label,
            roles=request.roles,
            admin_user=user
        )

        # Si un password est fourni, le hasher et l'ajouter immédiatement
        if request.password:
            from api.auth_router import get_password_hash

            password_hash = get_password_hash(request.password)

            # Mettre à jour users.json avec le password_hash
            import json
            from pathlib import Path

            users_path = Path(__file__).parent.parent / "config" / "users.json"
            with open(users_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Trouver et mettre à jour l'utilisateur nouvellement créé
            for u in config.get("users", []):
                if u.get("id") == request.user_id:
                    u["password_hash"] = password_hash
                    break

            # Sauvegarder
            with open(users_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # Vider le cache pour forcer reload
            clear_users_cache()

            logger.info(f"Password set for new user '{request.user_id}'")

        return success_response(
            new_user,
            meta={"message": f"User '{request.user_id}' created successfully" + (" with password" if request.password else "")}
        )

    except ValueError as e:
        logger.warning(f"Invalid user creation request: {e}")
        return error_response(
            str(e),
            code=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return error_response(
            f"Failed to create user: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    user: str = Depends(require_admin_role)
):
    """
    Mettre à jour un utilisateur existant.

    Args:
        user_id: ID utilisateur à modifier
        request: Données à mettre à jour (label, roles, status)

    Returns:
        dict: Utilisateur modifié
    """
    try:
        service = get_user_management_service()

        # Construire dict des champs à mettre à jour
        update_data = {}
        if request.label is not None:
            update_data["label"] = request.label
        if request.roles is not None:
            update_data["roles"] = request.roles
        if request.status is not None:
            update_data["status"] = request.status

        if not update_data:
            return error_response(
                "No fields to update",
                code=status.HTTP_400_BAD_REQUEST
            )

        updated_user = service.update_user(
            user_id=user_id,
            data=update_data,
            admin_user=user
        )

        return success_response(
            updated_user,
            meta={"message": f"User '{user_id}' updated successfully"}
        )

    except ValueError as e:
        logger.warning(f"Invalid user update request: {e}")
        return error_response(
            str(e),
            code=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        return error_response(
            f"Failed to update user: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    hard_delete: bool = Query(False, description="If True, permanent deletion (hard delete). If False, soft delete (default)"),
    user: str = Depends(require_admin_role)
):
    """
    Supprimer un utilisateur (soft ou hard delete).

    Soft delete (défaut):
    1. Marquer status = "inactive" dans config
    2. Renommer dossier: data/users/{user_id} → data/users/{user_id}_deleted_{timestamp}

    Hard delete:
    1. Supprimer complètement l'utilisateur de users.json
    2. Supprimer le dossier data/users/{user_id}

    Args:
        user_id: ID utilisateur à supprimer
        hard_delete: Si True, suppression permanente

    Returns:
        dict: Confirmation suppression
    """
    try:
        service = get_user_management_service()

        result = service.delete_user(
            user_id=user_id,
            admin_user=user,
            hard_delete=hard_delete
        )

        delete_type = "HARD (permanent)" if hard_delete else "soft (désactivation)"
        return success_response(
            result,
            meta={"message": f"User '{user_id}' deleted successfully ({delete_type})"}
        )

    except ValueError as e:
        logger.warning(f"Invalid user deletion request: {e}")
        return error_response(
            str(e),
            code=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return error_response(
            f"Failed to delete user: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/users/{user_id}/roles")
async def assign_roles(
    user_id: str,
    request: AssignRolesRequest,
    user: str = Depends(require_admin_role)
):
    """
    Assigner des rôles à un utilisateur (remplace les rôles existants).

    Args:
        user_id: ID utilisateur
        request: Liste de rôles à assigner

    Returns:
        dict: Utilisateur avec rôles mis à jour
    """
    try:
        service = get_user_management_service()

        updated_user = service.assign_roles(
            user_id=user_id,
            roles=request.roles,
            admin_user=user
        )

        return success_response(
            updated_user,
            meta={"message": f"Roles assigned to user '{user_id}'"}
        )

    except ValueError as e:
        logger.warning(f"Invalid role assignment request: {e}")
        return error_response(
            str(e),
            code=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error assigning roles: {e}")
        return error_response(
            f"Failed to assign roles: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    request: ResetPasswordRequest,
    user: str = Depends(require_admin_role)
):
    """
    Reset le password d'un utilisateur (admin uniquement).

    Args:
        user_id: ID utilisateur dont le password sera reset
        request: Nouveau password

    Returns:
        dict: Confirmation du reset
    """
    try:
        import json
        from pathlib import Path
        from api.auth_router import get_password_hash
        from api.config.users import get_user_info

        # Vérifier que l'utilisateur existe
        target_user_info = get_user_info(user_id)
        if not target_user_info:
            raise ValueError(f"User '{user_id}' not found")

        # Hasher le nouveau password
        new_hash = get_password_hash(request.new_password)

        # Mettre à jour users.json
        users_path = Path(__file__).parent.parent / "config" / "users.json"
        with open(users_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Trouver et mettre à jour l'utilisateur
        updated = False
        for u in config.get("users", []):
            if u.get("id") == user_id:
                u["password_hash"] = new_hash
                updated = True
                break

        if not updated:
            raise ValueError(f"User '{user_id}' not found in config")

        # Sauvegarder
        with open(users_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # CRITICAL: Vider le cache pour forcer reload de la config
        clear_users_cache()

        logger.info(f"Password reset by admin '{user}' for user '{user_id}'")

        return success_response(
            {"user_id": user_id, "updated_by": user},
            meta={"message": f"Password reset successfully for user '{user_id}'"}
        )

    except ValueError as e:
        logger.warning(f"Invalid password reset request: {e}")
        return error_response(
            str(e),
            code=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error resetting password: {e}")
        return error_response(
            f"Failed to reset password: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/roles")
async def list_roles(user: str = Depends(require_admin_role)):
    """
    Liste tous les rôles disponibles.

    Returns:
        dict: {role_name: role_description}
    """
    try:
        service = get_user_management_service()
        roles = service.get_all_roles()

        return success_response(
            roles,
            meta={"total": len(roles)}
        )

    except Exception as e:
        logger.error(f"Error listing roles: {e}")
        return error_response(
            f"Failed to list roles: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============================================================================
# Logs Viewer (Phase 2)
# ============================================================================

@router.get("/logs/list")
async def list_log_files(user: str = Depends(require_admin_role)):
    """
    Liste les fichiers de logs disponibles.

    Returns:
        dict: Liste des fichiers logs avec métadonnées
    """
    try:
        log_reader = get_log_reader()
        log_files = log_reader.list_log_files()

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


@router.get("/logs/read")
async def read_logs(
    filename: str = Query("app.log", description="Log file name"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(100, ge=1, le=1000, description="Max number of lines"),
    level: Optional[str] = Query(None, description="Filter by level (INFO, WARNING, ERROR)"),
    search: Optional[str] = Query(None, description="Search text in messages"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    sort_by: str = Query("timestamp", description="Sort by column (timestamp, level, module)"),
    sort_order: str = Query("desc", description="Sort order (asc, desc)"),
    user: str = Depends(require_admin_role)
):
    """
    Lit les logs avec filtres, tri et pagination.

    Args:
        filename: Nom du fichier log
        offset: Ligne de départ
        limit: Nombre de lignes max
        level: Filtre par niveau
        search: Recherche texte
        start_date: Date début
        end_date: Date fin
        sort_by: Colonne de tri
        sort_order: Ordre de tri

    Returns:
        dict: Logs filtrés, triés et paginés
    """
    try:
        log_reader = get_log_reader()

        result = log_reader.read_logs(
            filename=filename,
            offset=offset,
            limit=limit,
            level=level,
            search=search,
            start_date=start_date,
            end_date=end_date,
            sort_by=sort_by,
            sort_order=sort_order
        )

        return success_response(
            result["logs"],
            meta={
                "total": result["total"],
                "offset": result["offset"],
                "limit": result["limit"],
                "has_more": result["has_more"],
                "filename": filename,
                "filters": {
                    "level": level,
                    "search": search,
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
        )

    except FileNotFoundError as e:
        logger.warning(f"Log file not found: {e}")
        return error_response(
            str(e),
            code=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return error_response(
            f"Failed to read logs: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/logs/stats")
async def get_log_stats(
    filename: str = Query("app.log", description="Log file name"),
    user: str = Depends(require_admin_role)
):
    """
    Calcule des statistiques sur les logs.

    Args:
        filename: Nom du fichier log

    Returns:
        dict: Statistiques (total, by_level, top_modules, recent_errors)
    """
    try:
        log_reader = get_log_reader()
        stats = log_reader.get_log_stats(filename=filename)

        return success_response(stats)

    except FileNotFoundError as e:
        logger.warning(f"Log file not found: {e}")
        return error_response(
            str(e),
            code=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Error getting log stats: {e}")
        return error_response(
            f"Failed to get log stats: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============================================================================
# Cache Management (Phase 3)
# ============================================================================

@router.get("/cache/stats")
async def get_cache_stats(
    cache_name: Optional[str] = Query(None, description="Cache name (optional, returns all if not specified)"),
    user: str = Depends(require_admin_role)
):
    """
    Statistiques de tous les caches système ou d'un cache spécifique.

    Args:
        cache_name: Nom du cache (optionnel)

    Returns:
        dict: Stats par type de cache
    """
    try:
        stats = cache_manager.get_cache_stats(cache_name)
        logger.info(f"✅ Cache stats retrieved by {user}" + (f" for {cache_name}" if cache_name else ""))
        return success_response(stats)

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return error_response(
            f"Failed to get cache stats: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.delete("/cache/clear")
async def clear_cache(
    cache_name: str = Query("all", description="Cache name to clear (all, or specific cache name)"),
    user: str = Depends(require_admin_role)
):
    """
    Clear cache par nom ou tous les caches.

    Args:
        cache_name: Nom du cache à clear (all pour tous les caches)

    Returns:
        dict: Confirmation clear avec nombre d'entrées supprimées
    """
    try:
        # Clear users cache si demandé
        if cache_name in ["all", "users"]:
            clear_users_cache()
            logger.info(f"✅ Users cache cleared by {user}")

        # Clear cache via CacheManager
        result = cache_manager.clear_cache(cache_name, admin_user=user)

        if not result.get("ok", False):
            return error_response(
                result.get("error", "Unknown error"),
                details=result,
                code=status.HTTP_400_BAD_REQUEST
            )

        return success_response(result)

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return error_response(
            f"Failed to clear cache: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/cache/list")
async def list_caches(user: str = Depends(require_admin_role)):
    """
    Liste tous les caches disponibles.

    Returns:
        dict: Liste des noms de caches disponibles
    """
    try:
        available_caches = cache_manager.get_available_caches()
        logger.info(f"✅ Cache list retrieved by {user}")
        return success_response({
            "caches": available_caches,
            "count": len(available_caches)
        })

    except Exception as e:
        logger.error(f"Error listing caches: {e}")
        return error_response(
            f"Failed to list caches: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/cache/clear-expired")
async def clear_expired_entries(
    cache_name: Optional[str] = Query(None, description="Cache name (optional, clears all if not specified)"),
    user: str = Depends(require_admin_role)
):
    """
    Clear expired entries from cache(s).

    Args:
        cache_name: Nom du cache (optionnel, tous si non spécifié)

    Returns:
        dict: Nombre d'entrées expirées supprimées
    """
    try:
        result = cache_manager.clear_expired(cache_name)
        logger.info(f"✅ Expired entries cleared by {user}" + (f" from {cache_name}" if cache_name else " from all caches"))
        return success_response(result)

    except Exception as e:
        logger.error(f"Error clearing expired entries: {e}")
        return error_response(
            f"Failed to clear expired entries: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============================================================================
# ML Models (Phase 3)
# ============================================================================

@router.get("/ml/models")
async def list_ml_models(user: str = Depends(require_admin_role)):
    """
    Liste tous les modèles ML avec versions et statut.

    Returns:
        dict: Liste des modèles ML avec statut training jobs
    """
    try:
        models = training_executor.list_available_models()
        logger.info(f"✅ ML models listed by {user}")
        return success_response(models)

    except Exception as e:
        logger.error(f"Error listing ML models: {e}")
        return error_response(
            f"Failed to list ML models: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/ml/models/{model_name}/default-params")
async def get_model_default_params(
    model_name: str,
    user: str = Depends(require_admin_role)
):
    """
    Retourne les paramètres par défaut pour un modèle donné.

    Utilise le model_type détecté dans le registry pour retourner
    les bons paramètres (regime vs volatility).

    Args:
        model_name: Name of model

    Returns:
        dict: Default TrainingConfig pour ce modèle
    """
    try:
        from services.ml.model_registry import ModelRegistry

        registry = ModelRegistry()
        manifest = registry.get_manifest(model_name)

        # Déterminer le type de modèle
        if manifest:
            model_type = manifest.model_type
        else:
            # Fallback: deviner depuis le nom
            if "regime" in model_name.lower():
                model_type = "regime"
            elif "volatility" in model_name.lower():
                model_type = "volatility"
            else:
                model_type = "unknown"

        # Retourner les defaults selon le MODÈLE SPÉCIFIQUE (pas juste le type)
        # Ceci permet d'adapter les defaults à la disponibilité des données historiques

        if model_name == "stock_regime_detector":
            # Stocks: Besoin de 20 ans pour capturer 2008-2009 et 2020 bear markets
            config = TrainingConfig(
                days=7300,          # 20 years (includes 2008 -57%, 2020 -35%)
                epochs=100,
                patience=15,
                batch_size=32,
                learning_rate=0.001,
                train_val_split=0.8
            )

        elif model_name in ["btc_regime_detector", "btc_regime_hmm"]:
            # Bitcoin: Données fiables depuis ~2012, max ~14 ans disponibles
            config = TrainingConfig(
                days=5000,          # ~13.7 years (max reliable BTC data, from 2012)
                epochs=200,         # Plus d'epochs pour crypto (plus volatile)
                patience=25,
                batch_size=32,
                learning_rate=0.001,
                train_val_split=0.8
            )

        elif model_name == "volatility_forecaster":
            # Volatility: 2 ans suffisent pour capturer différents régimes de vol
            config = TrainingConfig(
                days=730,           # 2 years (sufficient for volatility patterns)
                epochs=100,
                patience=15,
                batch_size=32,
                learning_rate=0.001,
                train_val_split=0.8,
                hidden_size=64,
                min_r2=0.5,
                symbols=['BTC', 'ETH', 'SOL']
            )

        elif model_type == "regime":
            # Fallback pour autres regime models: 2 ans par défaut
            config = TrainingConfig(
                days=730,
                epochs=100,
                patience=15,
                batch_size=32,
                learning_rate=0.001,
                train_val_split=0.8
            )

        elif model_type == "volatility":
            # Fallback pour autres volatility models
            config = TrainingConfig(
                days=730,
                epochs=100,
                patience=15,
                batch_size=32,
                learning_rate=0.001,
                train_val_split=0.8,
                hidden_size=64,
                min_r2=0.5,
                symbols=['BTC', 'ETH', 'SOL']
            )

        else:
            # Generic defaults
            config = TrainingConfig()

        logger.info(f"✅ Default params returned for {model_name} (type: {model_type})")
        return success_response({
            "model_name": model_name,
            "model_type": model_type,
            "config": config.dict()
        })

    except Exception as e:
        logger.error(f"Error getting default params for {model_name}: {e}")
        return error_response(
            f"Failed to get default params: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/ml/train/{model_name}")
async def trigger_model_training(
    model_name: str,
    model_type: str = Query("unknown", description="Model type"),
    config: Optional[TrainingConfig] = Body(None, description="Training configuration (optional)"),
    user: str = Depends(require_admin_role)
):
    """
    Trigger manual model retraining.

    Phase 2 (Dec 2025): Accepte maintenant un body TrainingConfig optionnel
    pour personnaliser les paramètres de training.

    Args:
        model_name: Name of model to retrain
        model_type: Type of model (regime, volatility, etc.)
        config: Training configuration (optional, uses defaults if None)

    Returns:
        dict: Training job info
    """
    try:
        # Convert config to dict if provided
        config_dict = config.dict() if config else None

        result = training_executor.trigger_training(
            model_name=model_name,
            model_type=model_type,
            config=config_dict,  # ← NEW: Pass config to executor
            admin_user=user
        )

        if not result.get("ok", False):
            return error_response(
                result.get("error", "Unknown error"),
                code=status.HTTP_400_BAD_REQUEST
            )

        logger.info(f"✅ Training triggered for {model_name} by {user} (custom_config={config is not None})")
        return success_response(result)

    except Exception as e:
        logger.error(f"Error triggering training for {model_name}: {e}")
        return error_response(
            f"Failed to trigger training: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/ml/jobs")
async def list_training_jobs(
    status_filter: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=200),
    user: str = Depends(require_admin_role)
):
    """
    Liste tous les training jobs.

    Args:
        status_filter: Filter by status (optional)
        limit: Max jobs to return

    Returns:
        dict: List of training jobs
    """
    try:
        # Parse status filter if provided
        status_enum = None
        if status_filter:
            try:
                status_enum = JobStatus(status_filter.lower())
            except ValueError:
                return error_response(
                    f"Invalid status filter: {status_filter}. Valid: {[s.value for s in JobStatus]}",
                    code=status.HTTP_400_BAD_REQUEST
                )

        jobs = training_executor.list_jobs(status_filter=status_enum, limit=limit)
        logger.info(f"✅ Training jobs listed by {user}")
        return success_response(jobs)

    except Exception as e:
        logger.error(f"Error listing training jobs: {e}")
        return error_response(
            f"Failed to list training jobs: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/ml/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    user: str = Depends(require_admin_role)
):
    """
    Get training job status.

    Args:
        job_id: Job ID

    Returns:
        dict: Job status and details
    """
    try:
        job = training_executor.get_job_status(job_id)

        if not job:
            return error_response(
                f"Job not found: {job_id}",
                code=status.HTTP_404_NOT_FOUND
            )

        logger.info(f"✅ Job status retrieved for {job_id} by {user}")
        return success_response(job)

    except Exception as e:
        logger.error(f"Error getting job status {job_id}: {e}")
        return error_response(
            f"Failed to get job status: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.delete("/ml/jobs/{job_id}")
async def cancel_training_job(
    job_id: str,
    user: str = Depends(require_admin_role)
):
    """
    Cancel a training job.

    Args:
        job_id: Job ID

    Returns:
        dict: Cancellation result
    """
    try:
        result = training_executor.cancel_job(job_id)

        if not result.get("ok", False):
            return error_response(
                result.get("error", "Unknown error"),
                code=status.HTTP_400_BAD_REQUEST
            )

        logger.info(f"✅ Job {job_id} cancelled by {user}")
        return success_response(result)

    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        return error_response(
            f"Failed to cancel job: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============================================================================
# ML Auto-Trainer (Automatic Periodic Training)
# ============================================================================

@router.get("/ml/auto-trainer/status")
async def get_auto_trainer_status(user: str = Depends(require_admin_role)):
    """
    Get ML Auto-Trainer status and schedule.

    Returns:
        dict: Auto-trainer status, running state, and next run times
    """
    try:
        from services.ml.auto_trainer import ml_auto_trainer

        status_info = ml_auto_trainer.get_status()

        logger.info(f"✅ Auto-trainer status retrieved by {user}")
        return success_response(status_info)

    except Exception as e:
        logger.error(f"Error getting auto-trainer status: {e}")
        return error_response(
            f"Failed to get auto-trainer status: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/ml/auto-trainer/start")
async def start_auto_trainer(user: str = Depends(require_admin_role)):
    """
    Start the ML Auto-Trainer scheduler.

    Returns:
        dict: Success message
    """
    try:
        from services.ml.auto_trainer import ml_auto_trainer

        ml_auto_trainer.start()

        logger.info(f"✅ Auto-trainer started by {user}")
        return success_response({
            "message": "ML Auto-Trainer started successfully",
            "schedule": {
                "regime_models": "Daily at 3am",
                "volatility_models": "Daily at midnight",
                "correlation_models": "Every Sunday at 4am"
            }
        })

    except Exception as e:
        logger.error(f"Error starting auto-trainer: {e}")
        return error_response(
            f"Failed to start auto-trainer: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/ml/auto-trainer/stop")
async def stop_auto_trainer(user: str = Depends(require_admin_role)):
    """
    Stop the ML Auto-Trainer scheduler.

    Returns:
        dict: Success message
    """
    try:
        from services.ml.auto_trainer import ml_auto_trainer

        ml_auto_trainer.stop()

        logger.info(f"✅ Auto-trainer stopped by {user}")
        return success_response({
            "message": "ML Auto-Trainer stopped successfully"
        })

    except Exception as e:
        logger.error(f"Error stopping auto-trainer: {e}")
        return error_response(
            f"Failed to stop auto-trainer: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/ml/auto-trainer/trigger/{job_id}")
async def trigger_auto_trainer_job(
    job_id: str,
    user: str = Depends(require_admin_role)
):
    """
    Manually trigger a scheduled auto-trainer job immediately.

    Args:
        job_id: Job ID (regime_training_daily, volatility_training_daily, correlation_training_weekly)

    Returns:
        dict: Success message
    """
    try:
        from services.ml.auto_trainer import ml_auto_trainer

        valid_jobs = ['regime_training_daily', 'volatility_training_daily', 'correlation_training_weekly']
        if job_id not in valid_jobs:
            return error_response(
                f"Invalid job_id. Valid: {valid_jobs}",
                code=status.HTTP_400_BAD_REQUEST
            )

        result = ml_auto_trainer.trigger_now(job_id)

        if not result.get("ok", False):
            return error_response(
                result.get("error", "Unknown error"),
                code=status.HTTP_400_BAD_REQUEST
            )

        logger.info(f"✅ Auto-trainer job {job_id} triggered by {user}")
        return success_response(result)

    except Exception as e:
        logger.error(f"Error triggering auto-trainer job {job_id}: {e}")
        return error_response(
            f"Failed to trigger job: {str(e)}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


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
