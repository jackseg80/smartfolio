"""
Log Reader Service - Lecture et parsing des logs système
Filtrage, pagination, statistiques.
"""
from __future__ import annotations
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Type hints
LogEntry = Dict[str, Any]


class LogReader:
    """Service de lecture et parsing des logs"""

    # Format log: "2025-01-19 10:30:45,123 INFO module.name: message"
    LOG_PATTERN = re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+'
        r'(?P<level>\w+)\s+'
        r'(?P<module>[\w.-]+):\s+'
        r'(?P<message>.*)'
    )

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)

    def list_log_files(self) -> List[Dict[str, Any]]:
        """
        Liste tous les fichiers de logs disponibles.

        Returns:
            List[dict]: Liste des fichiers avec métadonnées
        """
        if not self.logs_dir.exists():
            logger.warning(f"Logs directory not found: {self.logs_dir}")
            return []

        log_files = []
        for log_file in self.logs_dir.glob("*.log*"):
            if log_file.is_file():
                stat = log_file.stat()
                log_files.append({
                    "name": log_file.name,
                    "path": str(log_file),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "modified_str": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        # Trier par nom (app.log en premier)
        log_files.sort(key=lambda x: x["name"])

        return log_files

    def parse_log_line(self, line: str) -> Optional[LogEntry]:
        """
        Parse une ligne de log.

        Args:
            line: Ligne de log brute

        Returns:
            dict ou None si parsing échoue
        """
        match = self.LOG_PATTERN.match(line)
        if not match:
            return None

        return {
            "timestamp": match.group("timestamp"),
            "level": match.group("level"),
            "module": match.group("module"),
            "message": match.group("message").strip()
        }

    def read_logs(
        self,
        filename: str = "app.log",
        offset: int = 0,
        limit: int = 100,
        level: Optional[str] = None,
        search: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sort_by: str = "timestamp",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """
        Lit les logs avec filtres, tri et pagination.

        Args:
            filename: Nom du fichier log
            offset: Ligne de départ (pagination)
            limit: Nombre de lignes max
            level: Filtre par niveau (INFO, WARNING, ERROR)
            search: Recherche texte dans message
            start_date: Date début (YYYY-MM-DD)
            end_date: Date fin (YYYY-MM-DD)
            sort_by: Colonne de tri (timestamp, level, module)
            sort_order: Ordre de tri (asc, desc)

        Returns:
            dict: {logs: List[LogEntry], total: int, has_more: bool}
        """
        log_file = self.logs_dir / filename

        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {filename}")

        # Lire toutes les lignes du fichier
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read log file {filename}: {e}")
            raise

        # Parser et filtrer les logs
        parsed_logs = []
        for line_num, line in enumerate(lines, start=1):
            log_entry = self.parse_log_line(line.strip())
            if not log_entry:
                continue

            # Filtre par niveau
            if level and log_entry["level"] != level.upper():
                continue

            # Filtre par recherche texte
            if search and search.lower() not in log_entry["message"].lower():
                continue

            # Filtre par date
            if start_date or end_date:
                try:
                    log_date = datetime.strptime(log_entry["timestamp"][:10], "%Y-%m-%d")

                    if start_date:
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                        if log_date < start_dt:
                            continue

                    if end_date:
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                        if log_date > end_dt:
                            continue
                except ValueError:
                    # Skip si parsing date échoue
                    continue

            # Ajouter line_num pour référence
            log_entry["line_num"] = line_num
            parsed_logs.append(log_entry)

        # Tri sur tous les logs
        if sort_by in ["timestamp", "level", "module"]:
            reverse = (sort_order == "desc")

            if sort_by == "timestamp":
                # Tri par timestamp (parsing datetime pour comparaison correcte)
                def timestamp_key(log):
                    try:
                        # Format: "2025-12-20 14:32:09,413" -> parse as datetime
                        ts_str = log["timestamp"].replace(',', '.')
                        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                    except Exception:
                        return datetime.min
                parsed_logs.sort(key=timestamp_key, reverse=reverse)

            elif sort_by == "level":
                # Tri par niveau de gravité: ERROR > WARNING > INFO > DEBUG
                level_order = {"ERROR": 4, "WARNING": 3, "INFO": 2, "DEBUG": 1}
                parsed_logs.sort(
                    key=lambda log: level_order.get(log["level"], 0),
                    reverse=reverse
                )

            elif sort_by == "module":
                # Tri alphabétique par module
                parsed_logs.sort(key=lambda log: log["module"], reverse=reverse)

        # Pagination
        total = len(parsed_logs)
        paginated_logs = parsed_logs[offset:offset + limit]
        has_more = (offset + limit) < total

        return {
            "logs": paginated_logs,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": has_more
        }

    def get_log_stats(self, filename: str = "app.log") -> Dict[str, Any]:
        """
        Calcule des statistiques sur les logs.

        Args:
            filename: Nom du fichier log

        Returns:
            dict: Statistiques (total lignes, count par level, modules, erreurs récentes)
        """
        log_file = self.logs_dir / filename

        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {filename}")

        # Lire et parser toutes les lignes
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read log file {filename}: {e}")
            raise

        # Compteurs
        total_lines = len(lines)
        by_level = {}
        by_module = {}
        recent_errors = []

        for line in lines:
            log_entry = self.parse_log_line(line.strip())
            if not log_entry:
                continue

            # Compter par level
            level = log_entry["level"]
            by_level[level] = by_level.get(level, 0) + 1

            # Compter par module
            module = log_entry["module"]
            by_module[module] = by_module.get(module, 0) + 1

            # Garder erreurs récentes (dernières 50)
            if level == "ERROR":
                recent_errors.append({
                    "timestamp": log_entry["timestamp"],
                    "module": log_entry["module"],
                    "message": log_entry["message"][:200]  # Tronquer message
                })

        # Garder seulement les 50 dernières erreurs
        recent_errors = recent_errors[-50:]

        # Top 10 modules par nombre de logs
        top_modules = sorted(
            by_module.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "filename": filename,
            "total_lines": total_lines,
            "by_level": by_level,
            "top_modules": [{"module": m, "count": c} for m, c in top_modules],
            "recent_errors": recent_errors
        }


# Singleton
_log_reader = None


def get_log_reader() -> LogReader:
    """Retourne l'instance singleton du service"""
    global _log_reader
    if _log_reader is None:
        _log_reader = LogReader()
    return _log_reader
