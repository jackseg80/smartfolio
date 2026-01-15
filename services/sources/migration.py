"""
Source Migration Service - Migrate users from V1 to V2 category-based system.

Handles:
- Detection of existing sources (cointracking, saxobank data)
- Migration of user config to category-based format
- Preservation of existing data and selections
- Auto-migration on first access
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SourceMigration:
    """
    Migrate existing users to the category-based source system.

    Migration strategy:
    1. Detect existing data sources (cointracking CSV/API, saxobank CSV)
    2. Map to new category-based config structure
    3. Preserve csv_selected_file and other user preferences
    4. Set data_source to "category_based"
    5. Data files remain unchanged
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.users_dir = self.project_root / "data" / "users"

    def needs_migration(self, user_id: str) -> bool:
        """
        Check if a user needs migration.

        Returns True if user config doesn't have category_based data_source.
        """
        config_path = self.users_dir / user_id / "config.json"
        if not config_path.exists():
            return True  # New user, will get default config

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("data_source") != "category_based"
        except Exception:
            return True

    def migrate_user(self, user_id: str) -> dict:
        """
        Migrate a single user to the category-based source system.

        Args:
            user_id: User identifier

        Returns:
            Migration report dict
        """
        report = {
            "user_id": user_id,
            "migrated": False,
            "crypto_source": None,
            "bourse_source": None,
            "preserved_settings": [],
            "errors": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        user_dir = self.users_dir / user_id
        config_path = user_dir / "config.json"

        # Load existing config or create new
        old_config = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    old_config = json.load(f)
            except Exception as e:
                report["errors"].append(f"Error loading config: {e}")
                old_config = {}

        # Skip if already migrated
        if old_config.get("data_source") == "category_based":
            report["migrated"] = True
            report["message"] = "Already migrated"
            return report

        # Detect existing sources
        detected = self._detect_existing_sources(user_dir, old_config)

        # Build new config
        new_config = self._build_new_config(old_config, detected, report)

        # Save new config (atomic write)
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = config_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=2, ensure_ascii=False)
            temp_path.replace(config_path)

            report["migrated"] = True
            report["crypto_source"] = new_config["sources"]["crypto"]["active_source"]
            report["bourse_source"] = new_config["sources"]["bourse"]["active_source"]

            logger.info(
                f"[migration] Migrated user {user_id}: "
                f"crypto={report['crypto_source']}, bourse={report['bourse_source']}"
            )

        except Exception as e:
            report["errors"].append(f"Error saving config: {e}")
            logger.error(f"[migration] Failed to migrate user {user_id}: {e}")

        return report

    def _detect_existing_sources(self, user_dir: Path, old_config: dict) -> dict:
        """
        Detect what sources the user was using.

        Returns dict with detected sources per category.
        """
        detected = {
            "crypto": {
                "type": None,
                "has_csv": False,
                "has_api": False,
                "csv_files": [],
                "selected_file": None,
            },
            "bourse": {
                "type": None,
                "has_csv": False,
                "csv_files": [],
                "selected_file": None,
            },
        }

        old_source = old_config.get("data_source", "")
        old_selected_file = old_config.get("csv_selected_file")

        # Check CoinTracking CSV
        ct_data_dir = user_dir / "cointracking" / "data"
        if ct_data_dir.exists():
            csv_files = list(ct_data_dir.glob("*.csv"))
            if csv_files:
                detected["crypto"]["has_csv"] = True
                detected["crypto"]["csv_files"] = [f.name for f in csv_files]
                if old_selected_file:
                    detected["crypto"]["selected_file"] = old_selected_file

        # Check CoinTracking API credentials
        secrets_path = user_dir / "config" / "secrets.json"
        if secrets_path.exists():
            try:
                with open(secrets_path, "r", encoding="utf-8") as f:
                    secrets = json.load(f)
                    ct_secrets = secrets.get("cointracking", {})
                    if ct_secrets.get("api_key") and ct_secrets.get("api_secret"):
                        detected["crypto"]["has_api"] = True
            except Exception:
                pass

        # Determine crypto source type based on old config
        if old_source == "cointracking_api" and detected["crypto"]["has_api"]:
            detected["crypto"]["type"] = "cointracking_api"
        elif detected["crypto"]["has_csv"]:
            detected["crypto"]["type"] = "cointracking_csv"
        # else: will default to manual

        # Check SaxoBank CSV
        saxo_data_dir = user_dir / "saxobank" / "data"
        if saxo_data_dir.exists():
            csv_files = list(saxo_data_dir.glob("*.csv")) + list(saxo_data_dir.glob("*.json"))
            if csv_files:
                detected["bourse"]["has_csv"] = True
                detected["bourse"]["csv_files"] = [f.name for f in csv_files]
                detected["bourse"]["type"] = "saxobank_csv"

        return detected

    def _build_new_config(self, old_config: dict, detected: dict, report: dict) -> dict:
        """
        Build new category-based config from old config and detected sources.
        """
        # Preserve all non-source settings
        preserved_keys = {
            "display_currency",
            "min_usd_threshold",
            "theme",
            "show_small_positions",
            "risk_profile",
        }

        new_config = {}
        for key in preserved_keys:
            if key in old_config:
                new_config[key] = old_config[key]
                report["preserved_settings"].append(key)

        # Set new data_source
        new_config["data_source"] = "category_based"

        # Build sources config
        new_config["sources"] = {
            "crypto": self._build_crypto_config(detected["crypto"]),
            "bourse": self._build_bourse_config(detected["bourse"]),
        }

        # Preserve old config reference for debugging
        new_config["_migration"] = {
            "migrated_at": datetime.utcnow().isoformat(),
            "old_data_source": old_config.get("data_source"),
            "old_csv_selected_file": old_config.get("csv_selected_file"),
        }

        return new_config

    def _build_crypto_config(self, detected: dict) -> dict:
        """Build crypto category config."""
        config = {
            "active_source": detected["type"] or "manual_crypto",
            "manual_crypto": {"enabled": True},
            "cointracking_csv": {
                "enabled": detected["has_csv"],
            },
            "cointracking_api": {
                "enabled": detected["has_api"],
            },
        }

        # Preserve selected file
        if detected["selected_file"]:
            config["cointracking_csv"]["selected_file"] = detected["selected_file"]

        return config

    def _build_bourse_config(self, detected: dict) -> dict:
        """Build bourse category config."""
        config = {
            "active_source": detected["type"] or "manual_bourse",
            "manual_bourse": {"enabled": True},
            "saxobank_csv": {
                "enabled": detected["has_csv"],
            },
        }

        return config

    def migrate_all_users(self) -> dict:
        """
        Migrate all users in the system.

        Returns summary report.
        """
        results = {
            "migrated": 0,
            "skipped": 0,
            "errors": 0,
            "details": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        if not self.users_dir.exists():
            results["errors"] = 1
            results["details"].append({"error": "Users directory not found"})
            return results

        for user_dir in self.users_dir.iterdir():
            if user_dir.is_dir():
                user_id = user_dir.name
                try:
                    if self.needs_migration(user_id):
                        report = self.migrate_user(user_id)
                        results["details"].append(report)
                        if report["migrated"]:
                            results["migrated"] += 1
                        else:
                            results["errors"] += 1
                    else:
                        results["skipped"] += 1
                        results["details"].append({
                            "user_id": user_id,
                            "migrated": False,
                            "message": "Already migrated",
                        })
                except Exception as e:
                    results["errors"] += 1
                    results["details"].append({
                        "user_id": user_id,
                        "error": str(e),
                    })

        logger.info(
            f"[migration] Complete: {results['migrated']} migrated, "
            f"{results['skipped']} skipped, {results['errors']} errors"
        )

        return results


def ensure_user_migrated(user_id: str, project_root: str) -> bool:
    """
    Ensure a user is migrated to V2 (called on first access).

    Returns True if user is now on V2 system.
    """
    migration = SourceMigration(project_root)

    if migration.needs_migration(user_id):
        report = migration.migrate_user(user_id)
        return report.get("migrated", False)

    return True


def get_effective_sources(user_id: str, project_root: str) -> dict:
    """
    Get the effective sources for a user (ensuring migration first).

    Returns dict with crypto_source and bourse_source.
    """
    # Ensure migrated
    ensure_user_migrated(user_id, project_root)

    # Load config
    config_path = Path(project_root) / "data" / "users" / user_id / "config.json"
    if not config_path.exists():
        return {
            "crypto_source": "manual_crypto",
            "bourse_source": "manual_bourse",
        }

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            sources = config.get("sources", {})
            return {
                "crypto_source": sources.get("crypto", {}).get("active_source", "manual_crypto"),
                "bourse_source": sources.get("bourse", {}).get("active_source", "manual_bourse"),
            }
    except Exception:
        return {
            "crypto_source": "manual_crypto",
            "bourse_source": "manual_bourse",
        }
