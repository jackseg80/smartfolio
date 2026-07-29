"""
Configuration et validation des utilisateurs multi-tenants.
LRU cache pour performances, validation stricte.
"""
from __future__ import annotations
import json
import os
import shutil
from functools import lru_cache
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Type hints
UserConfig = Dict[str, Any]
UsersDatabase = Dict[str, Any]


def get_users_config_path() -> Path:
    """Return the selected user store without ever mutating the legacy source."""
    legacy_path = Path("config/users.json")
    store_mode = os.getenv("AUTH_USER_STORE", "legacy").strip().lower()
    if store_mode == "legacy":
        return legacy_path
    if store_mode != "persistent":
        raise RuntimeError(f"Invalid AUTH_USER_STORE: {store_mode}")

    persistent_path = Path(os.getenv("AUTH_USERS_PATH", "data/auth/users.json"))
    if not persistent_path.exists():
        persistent_path.parent.mkdir(parents=True, exist_ok=True)
        if legacy_path.exists():
            shutil.copy2(legacy_path, persistent_path)
            logger.info("Persistent authentication store initialized from legacy users")
        elif os.getenv("AUTH_ALLOW_LEGACY_USER_STORE", "false").lower() != "true":
            raise RuntimeError("Persistent authentication store is missing")
    return persistent_path


@lru_cache(maxsize=1)
def _load_users_config() -> UsersDatabase:
    """Charge la configuration des utilisateurs avec cache LRU."""
    config_path = get_users_config_path()

    if not config_path.exists():
        raise RuntimeError(f"Users config not found at {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.debug(f"Loaded users config: {len(config.get('users', []))} users")
            return config
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load users config: {e}")
        raise RuntimeError("Unable to load users configuration") from e

def get_default_user() -> str:
    """Retourne l'utilisateur par défaut."""
    config = _load_users_config()
    configured = config.get("default")
    production = os.getenv("ENVIRONMENT", "development").strip().lower() == "production"
    if production and configured == "demo":
        active_named = next(
            (
                user.get("id")
                for user in config.get("users", [])
                if user.get("id") != "demo" and user.get("status") == "active"
            ),
            None,
        )
        if not active_named:
            raise RuntimeError("No active named user is configured")
        return active_named
    if not configured:
        raise RuntimeError("No default user is configured")
    return configured

def get_all_users() -> List[UserConfig]:
    """Retourne la liste de tous les utilisateurs configurés."""
    config = _load_users_config()
    return config.get("users", [])

def is_allowed_user(user_id: str) -> bool:
    """Vérifie si un utilisateur est autorisé."""
    if not user_id or not isinstance(user_id, str):
        return False

    if (
        user_id == "demo"
        and os.getenv("ENVIRONMENT", "development").strip().lower() == "production"
    ):
        return False
    users = get_all_users()
    allowed_ids = {user.get("id") for user in users}
    return user_id in allowed_ids

def get_user_info(user_id: str) -> Optional[UserConfig]:
    """Retourne les informations d'un utilisateur ou None si non trouvé."""
    if not is_allowed_user(user_id):
        return None

    users = get_all_users()
    for user in users:
        if user.get("id") == user_id:
            return user

    return None

def get_user_mode(user_id: str) -> str:
    """Retourne le mode de l'utilisateur (csv/api) ou 'csv' par défaut."""
    user_info = get_user_info(user_id)
    return user_info.get("mode", "csv") if user_info else "csv"

def clear_users_cache() -> None:
    """Vide le cache des utilisateurs (utile pour les tests)."""
    _load_users_config.cache_clear()
    logger.debug("Users config cache cleared")


def update_user_password(user_id: str, password_hash: str) -> None:
    """Atomically update a password in the selected user store."""
    config_path = get_users_config_path()
    config = _load_users_config()
    updated = False
    for user in config.get("users", []):
        if user.get("id") == user_id:
            user["password_hash"] = password_hash
            updated = True
            break
    if not updated:
        raise ValueError("User not found")

    temp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp_path, config_path)
    clear_users_cache()

# Validation stricte pour sécurité
def validate_user_id(user_id: str) -> str:
    """Valide et normalise un user_id, lève une exception si invalide."""
    if not user_id or not isinstance(user_id, str):
        raise ValueError("User ID must be a non-empty string")

    # Normalisation basique
    normalized = user_id.strip().lower()

    # Validation caractères (alphanumériques + underscores seulement)
    if not normalized.replace('_', '').replace('-', '').isalnum():
        raise ValueError("User ID must contain only alphanumeric characters, hyphens and underscores")

    # Vérification longueur
    if len(normalized) > 50:
        raise ValueError("User ID too long (max 50 characters)")

    return normalized
