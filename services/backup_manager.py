"""
Backup Manager — Sauvegarde et restauration des donnees utilisateur.

Strategie de retention:
- 7 daily backups
- 4 weekly backups (dimanche)
- 12 monthly backups (1er du mois)

Stockage: data/backups/{user_id}/backup_YYYYMMDD_HHMMSS.zip
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Retention policy
RETENTION_DAILY = 7
RETENTION_WEEKLY = 4
RETENTION_MONTHLY = 12

# Dossiers a sauvegarder par utilisateur
BACKUP_SUBDIRS = [
    "config",
    "cointracking/data",
    "cointracking/snapshots",
    "saxobank/data",
    "saxobank/cash",
    "manual_crypto",
    "manual_bourse",
    "banks",
    "wealth",
]

# Fichiers individuels a sauvegarder
BACKUP_FILES = [
    "config.json",
]

# Fichiers exclus (secrets)
EXCLUDED_FILES = {"secrets.json"}


class BackupManager:
    """Gere la creation, rotation et restauration des backups."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
    ):
        self.data_dir = data_dir or Path("data/users")
        self.backup_dir = backup_dir or Path("data/backups")

    def get_user_ids(self) -> List[str]:
        """Liste les user_ids disponibles."""
        if not self.data_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def create_backup(
        self,
        user_ids: Optional[List[str]] = None,
        include_secrets: bool = False,
    ) -> Dict[str, Any]:
        """
        Cree un backup ZIP pour chaque utilisateur.

        Args:
            user_ids: Liste d'utilisateurs (None = tous)
            include_secrets: Inclure secrets.json (defaut: non)

        Returns:
            Dict avec resultats par utilisateur
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        targets = user_ids or self.get_user_ids()
        results = {}

        for uid in targets:
            user_dir = self.data_dir / uid
            if not user_dir.exists():
                results[uid] = {"ok": False, "error": "User directory not found"}
                continue

            try:
                result = self._backup_user(uid, user_dir, timestamp, include_secrets)
                results[uid] = result
                logger.info(f"Backup created for user '{uid}': {result['file']}")
            except Exception as e:
                results[uid] = {"ok": False, "error": str(e)}
                logger.error(f"Backup failed for user '{uid}': {e}")

        return {
            "timestamp": timestamp,
            "users": results,
            "total_ok": sum(1 for r in results.values() if r.get("ok")),
            "total_failed": sum(1 for r in results.values() if not r.get("ok")),
        }

    def _backup_user(
        self,
        user_id: str,
        user_dir: Path,
        timestamp: str,
        include_secrets: bool,
    ) -> Dict[str, Any]:
        """Cree un backup ZIP pour un utilisateur."""
        dest_dir = self.backup_dir / user_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        zip_name = f"backup_{timestamp}.zip"
        zip_path = dest_dir / zip_name
        file_count = 0
        total_size = 0

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Fichiers individuels
            for fname in BACKUP_FILES:
                fpath = user_dir / fname
                if fpath.exists() and fpath.is_file():
                    zf.write(fpath, fname)
                    file_count += 1
                    total_size += fpath.stat().st_size

            # Secrets (optionnel)
            if include_secrets:
                secrets_path = user_dir / "secrets.json"
                if secrets_path.exists():
                    zf.write(secrets_path, "secrets.json")
                    file_count += 1
                    total_size += secrets_path.stat().st_size

            # Sous-dossiers
            for subdir_name in BACKUP_SUBDIRS:
                subdir = user_dir / subdir_name
                if not subdir.exists():
                    continue
                for root, _dirs, files in os.walk(subdir):
                    for f in files:
                        fpath = Path(root) / f
                        if fpath.name in EXCLUDED_FILES and not include_secrets:
                            continue
                        arcname = str(fpath.relative_to(user_dir))
                        zf.write(fpath, arcname)
                        file_count += 1
                        total_size += fpath.stat().st_size

        # Checksum
        checksum = self._compute_checksum(zip_path)
        zip_size = zip_path.stat().st_size

        return {
            "ok": True,
            "file": str(zip_path),
            "zip_size": zip_size,
            "original_size": total_size,
            "file_count": file_count,
            "checksum_sha256": checksum,
        }

    def apply_retention(self, user_id: Optional[str] = None) -> Dict[str, int]:
        """
        Applique la politique de retention. Garde:
        - 7 daily les plus recents
        - 4 weekly (dimanche)
        - 12 monthly (1er du mois)

        Returns:
            Nombre de fichiers supprimes par utilisateur
        """
        targets = [user_id] if user_id else self.get_user_ids()
        deleted_counts = {}

        for uid in targets:
            backup_dir = self.backup_dir / uid
            if not backup_dir.exists():
                continue
            deleted_counts[uid] = self._apply_retention_for_user(backup_dir)

        return deleted_counts

    def _apply_retention_for_user(self, backup_dir: Path) -> int:
        """Applique la retention pour un dossier de backups."""
        backups = self._list_backup_files(backup_dir)
        if not backups:
            return 0

        now = datetime.now()
        keep = set()

        # Trier par date decroissante
        dated = []
        for bp in backups:
            dt = self._parse_backup_date(bp.name)
            if dt:
                dated.append((dt, bp))
        dated.sort(key=lambda x: x[0], reverse=True)

        # 7 daily les plus recents
        for dt, bp in dated[:RETENTION_DAILY]:
            keep.add(bp)

        # 4 weekly (dimanche le plus recent de chaque semaine)
        weekly_kept = 0
        seen_weeks = set()
        for dt, bp in dated:
            week_key = dt.isocalendar()[:2]  # (year, week)
            if week_key not in seen_weeks and dt.weekday() == 6:  # dimanche
                keep.add(bp)
                seen_weeks.add(week_key)
                weekly_kept += 1
                if weekly_kept >= RETENTION_WEEKLY:
                    break

        # 12 monthly (1er du mois le plus recent)
        monthly_kept = 0
        seen_months = set()
        for dt, bp in dated:
            month_key = (dt.year, dt.month)
            if month_key not in seen_months:
                keep.add(bp)
                seen_months.add(month_key)
                monthly_kept += 1
                if monthly_kept >= RETENTION_MONTHLY:
                    break

        # Supprimer les non-gardes
        deleted = 0
        for bp in backups:
            if bp not in keep:
                bp.unlink()
                deleted += 1
                logger.info(f"Retention: deleted {bp.name}")

        return deleted

    def list_backups(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Liste les backups existants."""
        targets = [user_id] if user_id else self.get_user_ids()
        result = []

        for uid in targets:
            backup_dir = self.backup_dir / uid
            if not backup_dir.exists():
                continue
            for bp in self._list_backup_files(backup_dir):
                dt = self._parse_backup_date(bp.name)
                result.append({
                    "user_id": uid,
                    "file": bp.name,
                    "size": bp.stat().st_size,
                    "date": dt.isoformat() if dt else None,
                    "path": str(bp),
                })

        result.sort(key=lambda x: x.get("date") or "", reverse=True)
        return result

    def get_status(self) -> Dict[str, Any]:
        """Retourne le status global des backups."""
        all_backups = self.list_backups()
        total_size = sum(b["size"] for b in all_backups)

        users_status = {}
        for b in all_backups:
            uid = b["user_id"]
            if uid not in users_status:
                users_status[uid] = {"count": 0, "total_size": 0, "latest": None}
            users_status[uid]["count"] += 1
            users_status[uid]["total_size"] += b["size"]
            if users_status[uid]["latest"] is None or (b["date"] and b["date"] > users_status[uid]["latest"]):
                users_status[uid]["latest"] = b["date"]

        return {
            "backup_dir": str(self.backup_dir),
            "total_backups": len(all_backups),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2) if total_size else 0,
            "users": users_status,
            "retention_policy": {
                "daily": RETENTION_DAILY,
                "weekly": RETENTION_WEEKLY,
                "monthly": RETENTION_MONTHLY,
            },
        }

    def restore_backup(
        self,
        zip_path: str,
        user_id: str,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Restaure un backup.

        Args:
            zip_path: Chemin vers le fichier ZIP
            user_id: Utilisateur cible
            dry_run: Si True, simule sans ecrire

        Returns:
            Dict avec la liste des fichiers restaures
        """
        zip_file = Path(zip_path)
        if not zip_file.exists():
            return {"ok": False, "error": "Backup file not found"}

        # Verifier integrite
        try:
            with zipfile.ZipFile(zip_file, "r") as zf:
                bad = zf.testzip()
                if bad:
                    return {"ok": False, "error": f"Corrupted file in archive: {bad}"}
                file_list = zf.namelist()
        except zipfile.BadZipFile:
            return {"ok": False, "error": "Invalid ZIP file"}

        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "files_to_restore": file_list,
                "file_count": len(file_list),
                "target_dir": str(self.data_dir / user_id),
            }

        # Restauration reelle
        target_dir = self.data_dir / user_id
        target_dir.mkdir(parents=True, exist_ok=True)

        restored = []
        with zipfile.ZipFile(zip_file, "r") as zf:
            for member in file_list:
                dest = target_dir / member
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                restored.append(member)

        logger.info(f"Restored {len(restored)} files for user '{user_id}' from {zip_file.name}")

        return {
            "ok": True,
            "dry_run": False,
            "restored_files": restored,
            "file_count": len(restored),
            "target_dir": str(target_dir),
        }

    def verify_backup(self, zip_path: str) -> Dict[str, Any]:
        """Verifie l'integrite d'un backup."""
        zp = Path(zip_path)
        if not zp.exists():
            return {"ok": False, "error": "File not found"}

        try:
            with zipfile.ZipFile(zp, "r") as zf:
                bad = zf.testzip()
                if bad:
                    return {"ok": False, "error": f"Corrupted: {bad}"}
                return {
                    "ok": True,
                    "file_count": len(zf.namelist()),
                    "checksum_sha256": self._compute_checksum(zp),
                }
        except zipfile.BadZipFile:
            return {"ok": False, "error": "Invalid ZIP file"}

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_checksum(path: Path) -> str:
        """SHA-256 checksum d'un fichier."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _list_backup_files(directory: Path) -> List[Path]:
        """Liste les fichiers backup_*.zip dans un dossier."""
        if not directory.exists():
            return []
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.name.startswith("backup_") and p.suffix == ".zip"
        )

    @staticmethod
    def _parse_backup_date(filename: str) -> Optional[datetime]:
        """Extrait la date d'un nom de fichier backup_YYYYMMDD_HHMMSS.zip."""
        try:
            # backup_20260210_153045.zip -> 20260210_153045
            parts = filename.replace("backup_", "").replace(".zip", "")
            return datetime.strptime(parts, "%Y%m%d_%H%M%S")
        except (ValueError, IndexError):
            return None


# Singleton
backup_manager = BackupManager()
