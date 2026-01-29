"""
Module pour la gestion unifiée des données de portfolio
Assure la cohérence entre tous les endpoints
"""

from typing import Dict, Any


async def get_unified_filtered_balances(user_id: str = None, source: str = "cointracking", min_usd: float = 1.0, pricing: str = "local"):
    """
    Fonction helper unifiée utilisée par tous les endpoints pour garantir
    des données identiques avec le même filtrage.
    Retourne: {source_used: str, items: [...]}

    NOTE: Le paramètre pricing est ignoré pour l'instant car resolve_current_balances
    ne le supporte pas, mais on le garde pour compatibilité future.

    Args:
        user_id: ID utilisateur (FORTEMENT RECOMMANDÉ) - passer via Depends(get_required_user) dans endpoints

    WARNING: Default None avec fallback "demo" est TEMPORAIRE pour fonctions helper internes.
             Tous les endpoints doivent passer user_id explicitement.
    """
    # Import local pour éviter les imports circulaires
    from services.balance_service import balance_service
    from api.services.utils import to_rows
    import logging
    logger = logging.getLogger(__name__)

    # Fallback temporaire avec warning visible
    if user_id is None:
        logger.warning("⚠️  DEPRECATED: get_unified_filtered_balances called without user_id - using 'demo' fallback")
        user_id = "demo"

    # Appel direct avec source explicite
    res = await balance_service.resolve_current_balances(user_id, source)
    rows = [r for r in to_rows(res.get("items", [])) if float(r.get("value_usd") or 0.0) >= float(min_usd)]
    return {"source_used": res.get("source_used"), "items": rows}