"""
ML Cache Utilities - Fonctions partagées pour la gestion du cache ML

Ce module centralise:
- Instance du cache ML global
- Fonctions utilitaires de cache (get, set, clear)

Extrait de unified_ml_endpoints.py pour modularité (Fév 2026).
"""

from typing import Any, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Cache global pour les endpoints ML unifiés
# NOTE: Cache global (modèles, prédictions, sentiment) - données publiques
# EXCEPTION: alias_correlation_matrix() utilise portfolio user-specific (isolé via user_id)
_unified_ml_cache: Dict[str, Any] = {}


def get_ml_cache() -> Dict[str, Any]:
    """Retourne l'instance du cache ML global."""
    return _unified_ml_cache


def cache_get(cache: Dict[str, Any], key: str, ttl_seconds: int) -> Optional[Any]:
    """
    Récupère une valeur du cache si elle existe et n'a pas expiré.

    Args:
        cache: Dictionnaire de cache
        key: Clé de cache
        ttl_seconds: TTL en secondes

    Returns:
        Valeur cachée ou None si expirée/inexistante
    """
    if key not in cache:
        return None

    entry = cache[key]
    if not isinstance(entry, dict) or "timestamp" not in entry:
        # Entrée sans timestamp, la retourner directement
        return entry

    cached_time = entry.get("timestamp")
    if cached_time:
        try:
            if isinstance(cached_time, str):
                cached_time = datetime.fromisoformat(cached_time)
            age = (datetime.now() - cached_time).total_seconds()
            if age <= ttl_seconds:
                return entry.get("data", entry)
        except Exception as e:
            logger.debug(f"Error checking cache timestamp for {key}: {e}")

    return None


def cache_set(cache: Dict[str, Any], key: str, value: Any) -> None:
    """
    Stocke une valeur dans le cache avec timestamp.

    Args:
        cache: Dictionnaire de cache
        key: Clé de cache
        value: Valeur à stocker
    """
    cache[key] = {
        "data": value,
        "timestamp": datetime.now()
    }


def cache_clear_expired(cache: Dict[str, Any], ttl_seconds: int) -> int:
    """
    Nettoie les entrées expirées du cache.

    Args:
        cache: Dictionnaire de cache
        ttl_seconds: TTL en secondes

    Returns:
        Nombre d'entrées supprimées
    """
    now = datetime.now()
    keys_to_remove = []

    for key, entry in cache.items():
        if isinstance(entry, dict) and "timestamp" in entry:
            try:
                cached_time = entry["timestamp"]
                if isinstance(cached_time, str):
                    cached_time = datetime.fromisoformat(cached_time)
                age = (now - cached_time).total_seconds()
                if age > ttl_seconds:
                    keys_to_remove.append(key)
            except Exception:
                pass

    for key in keys_to_remove:
        del cache[key]

    return len(keys_to_remove)


def cache_clear_all() -> int:
    """
    Vide entièrement le cache ML.

    Returns:
        Nombre d'entrées supprimées
    """
    global _unified_ml_cache
    size = len(_unified_ml_cache)
    _unified_ml_cache.clear()
    return size
