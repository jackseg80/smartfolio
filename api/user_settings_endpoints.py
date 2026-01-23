"""
Endpoints pour la gestion des settings utilisateur.
Chaque utilisateur a ses propres settings dans data/users/{user_id}/config.json
Supports: CoinTracking CSV/API + Saxo CSV sources
"""
from __future__ import annotations
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
import logging

from api.deps import get_active_user
from api.services.user_fs import UserScopedFS
from services.user_secrets import user_secrets_manager
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/users", tags=["User Settings"])

class UserSettings(BaseModel):
    """Modèle pour les settings utilisateur"""
    data_source: str = "csv"
    api_base_url: str = "http://localhost:8080"
    display_currency: str = "USD"
    min_usd_threshold: float = 1.0
    csv_glob: str = "csv/*.csv"
    # Nom du fichier CSV sélectionné explicitement (dans data/users/<user>/csv)
    csv_selected_file: str | None = None
    cointracking_api_key: str = ""
    cointracking_api_secret: str = ""
    coingecko_api_key: str = ""
    fred_api_key: str = ""
    groq_api_key: str = ""  # Groq API for AI chat (free tier)
    pricing: str = "auto"  # 🔧 FIX: Changed default from "local" to "auto" (real-time prices)
    refresh_interval: int = 5
    enable_coingecko_classification: bool = True
    enable_portfolio_snapshots: bool = True
    enable_performance_tracking: bool = True
    theme: str = "auto"
    debug_mode: bool = False

@router.get("/settings", response_model=Dict[str, Any])
async def get_user_settings(user: str = Depends(get_active_user)) -> Dict[str, Any]:
    """
    Récupère les settings de l'utilisateur actuel.

    Fusionne config.json (UI settings) + secrets.json (API keys)

    Returns:
        Dict[str, Any]: Settings de l'utilisateur (UI + clés API masquées)

    Raises:
        HTTPException: 404 si les settings n'existent pas
    """
    try:
        # Utiliser UserScopedFS pour lire les settings
        project_root = str(Path(__file__).parent.parent)
        user_fs = UserScopedFS(project_root, user)

        # Load UI settings from config.json
        if not user_fs.exists("config.json"):
            logger.warning(f"No config.json found for user {user}, using defaults")
            config_settings = {}
        else:
            config_settings = user_fs.read_json("config.json")
            logger.debug(f"Loaded config.json for user {user}: {len(config_settings)} keys")

        # Load API keys from secrets.json (modern system)
        secrets = user_secrets_manager.get_user_secrets(user)

        # Fusionner: defaults + config.json + secrets.json (API keys)
        default_settings = UserSettings()
        merged_settings = default_settings.dict()
        merged_settings.update(config_settings)

        # Add API keys from secrets.json
        merged_settings["cointracking_api_key"] = secrets.get("cointracking", {}).get("api_key", "")
        merged_settings["cointracking_api_secret"] = secrets.get("cointracking", {}).get("api_secret", "")
        merged_settings["coingecko_api_key"] = secrets.get("coingecko", {}).get("api_key", "")
        merged_settings["groq_api_key"] = secrets.get("groq", {}).get("api_key", "")
        merged_settings["fred_api_key"] = secrets.get("fred", {}).get("api_key", "")

        logger.debug(f"Settings loaded for user {user}: config.json + secrets.json merged")
        return merged_settings

    except Exception as e:
        logger.error(f"Error loading settings for user {user}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading user settings: {str(e)}"
        )

@router.put("/settings")
async def save_user_settings(
    settings: Dict[str, Any],
    user: str = Depends(get_active_user)
) -> Dict[str, str]:
    """
    Sauvegarde les settings de l'utilisateur actuel.

    Sépare automatiquement :
    - Clés API → secrets.json
    - UI settings → config.json

    Args:
        settings: Nouveau settings à sauvegarder
        user: ID utilisateur (injecté par dependency)

    Returns:
        Dict[str, str]: Statut de la sauvegarde

    Raises:
        HTTPException: 400 si les settings sont invalides, 500 si erreur serveur
    """
    try:
        # Valider les settings avec le modèle Pydantic
        try:
            validated_settings = UserSettings(**settings)
        except Exception as validation_error:
            logger.warning(f"Settings validation failed for user {user}: {validation_error}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid settings: {str(validation_error)}"
            )

        # Utiliser UserScopedFS pour écrire les settings
        project_root = str(Path(__file__).parent.parent)
        user_fs = UserScopedFS(project_root, user)

        # Convertir en dict
        new_settings = validated_settings.dict()

        # Séparer les clés API des autres settings
        api_keys = {
            "cointracking_api_key": new_settings.pop("cointracking_api_key", ""),
            "cointracking_api_secret": new_settings.pop("cointracking_api_secret", ""),
            "coingecko_api_key": new_settings.pop("coingecko_api_key", ""),
            "fred_api_key": new_settings.pop("fred_api_key", ""),
            "groq_api_key": new_settings.pop("groq_api_key", "")
        }

        # 1. Sauvegarder les clés API dans secrets.json
        try:
            secrets = user_secrets_manager.get_user_secrets(user)

            # Mettre à jour les secrets, y compris les chaînes vides pour permettre la suppression.
            secrets.setdefault("coingecko", {})["api_key"] = api_keys["coingecko_api_key"]
            secrets.setdefault("cointracking", {})["api_key"] = api_keys["cointracking_api_key"]
            secrets.setdefault("cointracking", {})["api_secret"] = api_keys["cointracking_api_secret"]
            secrets.setdefault("fred", {})["api_key"] = api_keys["fred_api_key"]
            secrets.setdefault("groq", {})["api_key"] = api_keys["groq_api_key"]

            # Save secrets.json
            user_fs.write_json("secrets.json", secrets)
            logger.debug(f"API keys saved to secrets.json for user {user}")

            # Vider le cache pour cet utilisateur pour forcer un rechargement
            user_secrets_manager.clear_cache(user)
            logger.debug(f"Secrets cache cleared for user {user}")
        except Exception as e:
            logger.warning(f"Failed to save API keys to secrets.json: {e}")

        # 2. Merger UI settings avec config.json existant
        old_config = {}
        try:
            old_config = user_fs.read_json("config.json")
            logger.debug(f"Merging with existing config: {len(old_config)} keys")
        except (FileNotFoundError, ValueError):
            logger.debug("No existing config found, creating new one")

        merged_settings = {**old_config, **new_settings}

        # ✅ FIX: Synchroniser V1 → V2 pour csv_selected_file
        # Évite la race condition où V1 est mis à jour mais V2 non
        csv_selected = merged_settings.get("csv_selected_file")
        if csv_selected:
            # S'assurer que la structure V2 existe
            if "sources" not in merged_settings:
                merged_settings["sources"] = {}
            if "crypto" not in merged_settings["sources"]:
                merged_settings["sources"]["crypto"] = {}
            # Synchroniser V1 → V2
            merged_settings["sources"]["crypto"]["selected_csv_file"] = csv_selected
            logger.debug(f"Synced V1 csv_selected_file to V2 for user {user}: {csv_selected}")

        # Sauvegarder config.json (SANS les clés API)
        user_fs.write_json("config.json", merged_settings)

        logger.info(f"✅ Settings saved for user {user}: config.json + secrets.json")

        return {
            "status": "success",
            "message": f"Settings saved (config.json + secrets.json)",
            "user": user
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving settings for user {user}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving user settings: {str(e)}"
        )

@router.delete("/settings")
async def reset_user_settings(user: str = Depends(get_active_user)) -> Dict[str, str]:
    """
    Remet les settings utilisateur aux valeurs par défaut.

    Args:
        user: ID utilisateur (injecté par dependency)

    Returns:
        Dict[str, str]: Statut de la remise à zéro
    """
    try:
        project_root = str(Path(__file__).parent.parent)
        user_fs = UserScopedFS(project_root, user)

        # Créer les settings par défaut
        default_settings = UserSettings()
        user_fs.write_json("config.json", default_settings.dict())

        logger.info(f"Settings reset to defaults for user {user}")

        return {
            "status": "success",
            "message": f"Settings reset to defaults for user {user}",
            "user": user
        }

    except Exception as e:
        logger.error(f"Error resetting settings for user {user}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resetting user settings: {str(e)}"
        )

@router.get("/sources")
async def get_user_data_sources(user: str = Depends(get_active_user)) -> Dict[str, Any]:
    """
    Récupère les sources de données disponibles pour l'utilisateur depuis data/ (nouveau système).

    Returns:
        Dict[str, Any]: Sources disponibles (CSV + API conditionnelle)
    """
    try:
        from api.services.data_router import UserDataRouter

        project_root = str(Path(__file__).parent.parent)
        data_router = UserDataRouter(project_root, user)

        sources = []

        # ========== SOURCES V2 STRUCTURE ==========
        # Instead of listing each CSV file, return generic source types

        # 1. Manual Crypto Source
        sources.append({
            "key": "manual_crypto",
            "label": "📝 Saisie Manuelle (Crypto)",
            "type": "manual",
            "module": "manual_crypto",
            "file_path": None
        })

        # 2. CoinTracking CSV (generic, single entry)
        ct_data_dir = data_router.user_fs.get_path("cointracking/data")
        ct_has_csv = Path(ct_data_dir).exists() and any(Path(ct_data_dir).glob("*.csv"))

        if ct_has_csv:
            sources.append({
                "key": "cointracking_csv",
                "label": "📄 Import CSV (CoinTracking)",
                "type": "csv",
                "module": "cointracking",
                "file_path": None  # File selection is handled by Sources V2 UI
            })

        # API CoinTracking: seulement si l'utilisateur a des clés API
        # Note: API keys sont dans data_router.api_credentials (secrets.json), pas dans settings (config.json)
        has_ct_credentials = (
            data_router.api_credentials.get("api_key") and
            data_router.api_credentials.get("api_secret")
        )

        if has_ct_credentials:
            sources.append({
                "key": "cointracking_api",
                "label": "🌐 CoinTracking API",
                "type": "api",
                "module": "cointracking",
                "file_path": None
            })

        # 4. Manual Bourse Source
        sources.append({
            "key": "manual_bourse",
            "label": "📝 Saisie Manuelle (Bourse)",
            "type": "manual",
            "module": "manual_bourse",
            "file_path": None
        })

        # 5. Saxo CSV (generic, single entry)
        saxo_data_dir = data_router.user_fs.get_path("saxobank/data")
        saxo_has_csv = Path(saxo_data_dir).exists() and any(Path(saxo_data_dir).glob("*.csv"))

        if saxo_has_csv:
            sources.append({
                "key": "saxobank_csv",
                "label": "📄 Import CSV (Saxo)",
                "type": "csv",
                "module": "saxobank",
                "file_path": None  # File selection is handled by Sources V2 UI
            })

        # 6. API SaxoBank: seulement si l'utilisateur est connecté via OAuth2
        from services.saxo_auth_service import SaxoAuthService

        saxo_auth = SaxoAuthService(user)
        if saxo_auth.is_connected():
            connection_status = saxo_auth.get_connection_status()
            env_label = "Live" if connection_status.get("environment") == "live" else "Simulation"

            sources.append({
                "key": "saxobank_api",
                "label": f"🌐 Saxo Bank API ({env_label})",
                "type": "api",
                "module": "saxobank",
                "file_path": None,
                "connected": True,
                "environment": connection_status.get("environment")
            })

        # Count CSV files (for statistics)
        cointracking_count = len(list(Path(ct_data_dir).glob("*.csv"))) if Path(ct_data_dir).exists() else 0
        saxo_count = len(list(Path(saxo_data_dir).glob("*.csv"))) if Path(saxo_data_dir).exists() else 0

        return {
            "user": user,
            "sources": sources,
            "current_source": data_router.settings.get("data_source", "csv"),
            "total_csv_files": cointracking_count + saxo_count,
            "cointracking_files": cointracking_count,
            "saxo_files": saxo_count
        }

    except Exception as e:
        logger.error(f"Error getting data sources for user {user}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting user data sources: {str(e)}"
        )

@router.get("/settings/info")
async def get_user_settings_info(user: str = Depends(get_active_user)) -> Dict[str, Any]:
    """
    Récupère des informations de debug sur les settings utilisateur.

    Returns:
        Dict[str, Any]: Informations de debug
    """
    try:
        project_root = str(Path(__file__).parent.parent)
        user_fs = UserScopedFS(project_root, user)

        config_exists = user_fs.exists("config.json")
        user_root = user_fs.get_user_root()

        info = {
            "user": user,
            "user_root": user_root,
            "config_exists": config_exists,
            "config_path": user_fs.get_path("config.json")
        }

        if config_exists:
            settings = user_fs.read_json("config.json")
            info["settings_keys"] = list(settings.keys())
            info["data_source"] = settings.get("data_source", "unknown")

        return info

    except Exception as e:
        logger.error(f"Error getting settings info for user {user}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting user settings info: {str(e)}"
        )

@router.get("/sources/debug")
async def debug_user_sources(user: str = Depends(get_active_user)) -> Dict[str, Any]:
    """DEBUG: Vérifie directement les fichiers dans data/ vs imports/"""
    try:
        project_root = str(Path(__file__).parent.parent)
        user_fs = UserScopedFS(project_root, user)

        # Scanner tous les dossiers
        ct_data = list(Path(user_fs.get_path("cointracking/data")).glob("*.csv")) if Path(user_fs.get_path("cointracking/data")).exists() else []
        ct_imports = list(Path(user_fs.get_path("cointracking/imports")).glob("*.csv")) if Path(user_fs.get_path("cointracking/imports")).exists() else []
        saxo_data = list(Path(user_fs.get_path("saxobank/data")).glob("*.csv")) if Path(user_fs.get_path("saxobank/data")).exists() else []
        saxo_imports = list(Path(user_fs.get_path("saxobank/imports")).glob("*.csv")) if Path(user_fs.get_path("saxobank/imports")).exists() else []

        return {
            "cointracking": {
                "data": [str(p.name) for p in ct_data],
                "imports": [str(p.name) for p in ct_imports]
            },
            "saxobank": {
                "data": [str(p.name) for p in saxo_data],
                "imports": [str(p.name) for p in saxo_imports]
            },
            "timestamp": "2025-10-13T18:50:00Z"  # Force reload marker
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

