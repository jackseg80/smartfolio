"""
User Management Service - CRUD operations pour utilisateurs multi-tenants
Gestion complète: création, modification, suppression, rôles.
Auto-création structure de dossiers.
"""
from __future__ import annotations
import json
import os
import shutil
import tempfile
from filelock import FileLock
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging

# Import from config/ (not api/) to respect layered architecture
from config.users import clear_users_cache, get_user_info, validate_user_id

logger = logging.getLogger(__name__)

# Type hints
UserData = Dict[str, Any]


class UserManagementService:
    """Service centralisé pour gestion utilisateurs"""

    def __init__(self):
        self.users_config_path = Path("config/users.json")
        self.data_users_path = Path("data/users")

    def _load_users_config(self) -> Dict[str, Any]:
        """Charge la configuration users.json"""
        if not self.users_config_path.exists():
            raise FileNotFoundError(f"Users config not found: {self.users_config_path}")

        with open(self.users_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_users_config(self, config: Dict[str, Any]) -> None:
        """Sauvegarde la configuration users.json (écriture atomique)"""
        path = self.users_config_path
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent)
        )
        try:
            with FileLock(str(path) + ".lock", timeout=5):
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, path)
        except (OSError, PermissionError, ValueError):
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

        # Clear cache pour forcer reload
        clear_users_cache()
        logger.info("Users config saved and cache cleared")

    def _create_user_folder_structure(self, user_id: str) -> None:
        """
        Crée la structure de dossiers pour un nouvel utilisateur.

        Structure créée:
        data/users/{user_id}/
          config.json          # UI settings
          secrets.json         # API keys (empty template)
          audit.log            # Audit log file (empty)
          cointracking/
            data/
            api_cache/
          saxobank/
            data/
          config/
            sources.json
        """
        user_path = self.data_users_path / user_id

        # Créer dossier principal
        user_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created user directory: {user_path}")

        # Créer config.json (settings UI par défaut)
        config_file = user_path / "config.json"
        if not config_file.exists():
            default_config = {
                "data_source": "category_based",  # V2 mode: empty sources by default
                "theme": "dark",
                "refresh_interval": 60,
                "language": "fr",
                "sources": {
                    "crypto": {
                        "active_source": "manual_crypto",
                        "manual_crypto": {"enabled": True},
                        "cointracking_csv": {"enabled": False},
                        "cointracking_api": {"enabled": False}
                    },
                    "bourse": {
                        "active_source": "manual_bourse",
                        "manual_bourse": {"enabled": True},
                        "saxobank_csv": {"enabled": False}
                    }
                }
            }
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config.json for user: {user_id}")

        # Créer secrets.json (template vide)
        secrets_file = user_path / "secrets.json"
        if not secrets_file.exists():
            default_secrets = {
                "coingecko": {"api_key": "", "pro": False},
                "cointracking": {"api_key": "", "api_secret": ""},
                "fred": {"api_key": ""},
                "binance": {"api_key": "", "api_secret": "", "testnet": False},
                "kraken": {"api_key": "", "api_secret": ""},
                "saxo": {
                    "sim_client_id": "",
                    "sim_client_secret": "",
                    "live_client_id": "",
                    "live_client_secret": "",
                    "environment": "simulation",
                    "redirect_uri": "http://localhost:8080/api/saxo/callback"
                },
                "exchanges": {"default": "binance"}
            }
            with open(secrets_file, 'w', encoding='utf-8') as f:
                json.dump(default_secrets, f, indent=2)
            logger.info(f"Created default secrets.json for user: {user_id}")

        # Créer audit.log (vide)
        audit_file = user_path / "audit.log"
        if not audit_file.exists():
            audit_file.touch()
            logger.info(f"Created audit.log for user: {user_id}")

        # Créer structure cointracking
        (user_path / "cointracking" / "data").mkdir(parents=True, exist_ok=True)
        (user_path / "cointracking" / "api_cache").mkdir(parents=True, exist_ok=True)

        # Créer structure saxobank
        (user_path / "saxobank" / "data").mkdir(parents=True, exist_ok=True)

        # Créer structure config
        config_dir = user_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Créer sources.json (configuration modules sources)
        sources_file = config_dir / "sources.json"
        if not sources_file.exists():
            default_sources = {
                "modules": {
                    "cointracking": {"enabled": True, "mode": "csv"},
                    "saxobank": {"enabled": False, "mode": "csv"}
                }
            }
            with open(sources_file, 'w', encoding='utf-8') as f:
                json.dump(default_sources, f, indent=2)
            logger.info(f"Created sources.json for user: {user_id}")

        logger.info(f"✅ User folder structure created for: {user_id}")

    def create_user(
        self,
        user_id: str,
        label: str,
        roles: Optional[List[str]] = None,
        admin_user: str = "system"
    ) -> UserData:
        """
        Crée un nouvel utilisateur avec structure de dossiers complète.

        Args:
            user_id: ID utilisateur (alphanumeric + underscore)
            label: Label d'affichage
            roles: Liste de rôles (défaut: ["viewer"])
            admin_user: User qui crée (pour audit log)

        Returns:
            UserData: Données utilisateur créé

        Raises:
            ValueError: Si user_id invalide ou user existe déjà
        """
        # Validation user_id
        normalized_user_id = validate_user_id(user_id)

        # Charger config
        config = self._load_users_config()

        # Vérifier si user existe déjà
        existing_users = {u["id"] for u in config.get("users", [])}
        if normalized_user_id in existing_users:
            raise ValueError(f"User already exists: {normalized_user_id}")

        # Préparer données user
        if roles is None:
            roles = ["viewer"]

        new_user = {
            "id": normalized_user_id,
            "label": label,
            "roles": roles,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # Ajouter à la config
        config["users"].append(new_user)

        # Sauvegarder config
        self._save_users_config(config)

        # Créer structure de dossiers
        self._create_user_folder_structure(normalized_user_id)

        # Audit log
        logger.info(f"✅ User created: {normalized_user_id} by {admin_user}")

        return new_user

    def update_user(
        self,
        user_id: str,
        data: Dict[str, Any],
        admin_user: str = "system"
    ) -> UserData:
        """
        Met à jour les données d'un utilisateur.

        Args:
            user_id: ID utilisateur
            data: Données à mettre à jour (label, roles, status)
            admin_user: User qui modifie (pour audit log)

        Returns:
            UserData: Données utilisateur modifiées

        Raises:
            ValueError: Si user n'existe pas
        """
        # Validation
        normalized_user_id = validate_user_id(user_id)

        # Charger config
        config = self._load_users_config()

        # Trouver user
        user_index = None
        for i, user in enumerate(config["users"]):
            if user["id"] == normalized_user_id:
                user_index = i
                break

        if user_index is None:
            raise ValueError(f"User not found: {normalized_user_id}")

        # Mettre à jour champs autorisés
        allowed_fields = {"label", "roles", "status"}
        for field, value in data.items():
            if field in allowed_fields:
                config["users"][user_index][field] = value

        # Sauvegarder
        self._save_users_config(config)

        # Audit log
        logger.info(f"✅ User updated: {normalized_user_id} by {admin_user}, fields: {list(data.keys())}")

        return config["users"][user_index]

    def delete_user(
        self,
        user_id: str,
        admin_user: str = "system",
        hard_delete: bool = False
    ) -> Dict[str, Any]:
        """
        Supprime un utilisateur (soft ou hard delete).

        Soft delete process:
        1. Marquer status = "inactive" dans config
        2. Renommer dossier: data/users/{user_id} → data/users/{user_id}_deleted_{timestamp}

        Hard delete process:
        1. Supprimer complètement l'utilisateur de users.json
        2. Supprimer le dossier data/users/{user_id}

        Args:
            user_id: ID utilisateur
            admin_user: User qui supprime (pour audit log)
            hard_delete: Si True, suppression complète. Si False, soft delete (défaut)

        Returns:
            dict: Confirmation suppression

        Raises:
            ValueError: Si user n'existe pas ou user par défaut
        """
        # Validation
        normalized_user_id = validate_user_id(user_id)

        # Charger config
        config = self._load_users_config()

        # Empêcher suppression user par défaut
        if normalized_user_id == config.get("default", "demo"):
            raise ValueError(f"Cannot delete default user: {normalized_user_id}")

        # Trouver user
        user_index = None
        for i, user in enumerate(config["users"]):
            if user["id"] == normalized_user_id:
                user_index = i
                break

        if user_index is None:
            raise ValueError(f"User not found: {normalized_user_id}")

        if hard_delete:
            # HARD DELETE: Supprimer complètement de users.json
            deleted_user = config["users"].pop(user_index)
            self._save_users_config(config)

            # Supprimer dossier complètement
            user_path = self.data_users_path / normalized_user_id
            if user_path.exists():
                try:
                    shutil.rmtree(str(user_path))
                    logger.info(f"✅ User folder deleted permanently: {user_path}")
                except Exception as e:
                    logger.error(f"Failed to delete user folder: {e}")

            logger.info(f"✅ User deleted (HARD): {normalized_user_id} by {admin_user}")

            return {
                "user_id": normalized_user_id,
                "deleted": True,
                "delete_type": "hard",
                "deleted_at": datetime.now(timezone.utc).isoformat(),
                "deleted_by": admin_user
            }

        else:
            # SOFT DELETE: Marquer comme inactive
            config["users"][user_index]["status"] = "inactive"
            self._save_users_config(config)

            # Renommer dossier
            user_path = self.data_users_path / normalized_user_id
            if user_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                deleted_path = self.data_users_path / f"{normalized_user_id}_deleted_{timestamp}"

                try:
                    shutil.move(str(user_path), str(deleted_path))
                    logger.info(f"✅ User folder renamed: {user_path} → {deleted_path}")
                except Exception as e:
                    logger.error(f"Failed to rename user folder: {e}")

            logger.info(f"✅ User deleted (soft): {normalized_user_id} by {admin_user}")

            return {
                "user_id": normalized_user_id,
                "deleted": True,
                "delete_type": "soft",
                "deleted_at": datetime.now(timezone.utc).isoformat(),
                "deleted_by": admin_user
            }

    def assign_roles(
        self,
        user_id: str,
        roles: List[str],
        admin_user: str = "system"
    ) -> UserData:
        """
        Assigne des rôles à un utilisateur (remplace les rôles existants).

        Args:
            user_id: ID utilisateur
            roles: Liste de rôles à assigner
            admin_user: User qui assigne (pour audit log)

        Returns:
            UserData: Données utilisateur modifiées

        Raises:
            ValueError: Si user n'existe pas ou rôles invalides
        """
        # Charger config pour validation rôles
        config = self._load_users_config()
        valid_roles = set(config.get("roles", {}).keys())

        # Valider rôles
        invalid_roles = set(roles) - valid_roles
        if invalid_roles:
            raise ValueError(f"Invalid roles: {invalid_roles}. Valid roles: {valid_roles}")

        # Utiliser update_user
        return self.update_user(
            user_id,
            {"roles": roles},
            admin_user
        )

    def get_all_roles(self) -> Dict[str, str]:
        """
        Retourne tous les rôles disponibles.

        Returns:
            dict: {role_name: role_description}
        """
        config = self._load_users_config()
        return config.get("roles", {})


# Singleton
_user_management_service = None


def get_user_management_service() -> UserManagementService:
    """Retourne l'instance singleton du service"""
    global _user_management_service
    if _user_management_service is None:
        _user_management_service = UserManagementService()
    return _user_management_service
