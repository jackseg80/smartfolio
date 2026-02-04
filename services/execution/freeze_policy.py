"""
Freeze Policy Module - Sémantique Freeze claire pour Governance Engine

Ce module définit les types de freeze et leurs opérations autorisées:
- FULL_FREEZE: Tout bloqué (urgence)
- S3_ALERT_FREEZE: Achats bloqués, rotations stables OK, hedge OK
- ERROR_FREEZE: Freeze prudent, réductions risque autorisées

Phase 1C: Sémantique claire pour éviter confusion opérationnelle
"""

from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FreezeType:
    """Types de freeze avec sémantique précise"""
    FULL_FREEZE = "full_freeze"       # Tout bloqué (urgence)
    S3_ALERT_FREEZE = "s3_freeze"     # Freeze achats, rotations↓ stables OK, hedge OK
    ERROR_FREEZE = "error_freeze"     # Freeze prudent, réductions risque autorisées


class FreezeSemantics:
    """
    Définit précisément ce qui est autorisé/bloqué selon le type de freeze
    Phase 1C: Sémantique claire pour éviter confusion opérationnelle
    """

    @staticmethod
    def get_allowed_operations(freeze_type: str) -> Dict[str, bool]:
        """Retourne les opérations autorisées selon le type de freeze"""

        if freeze_type == FreezeType.FULL_FREEZE:
            return {
                "new_purchases": False,     # Pas de nouveaux achats
                "sell_to_stables": False,   # Pas de ventes vers stables
                "asset_rotations": False,   # Pas de rotations BTC→ETH, etc.
                "hedge_operations": False,  # Pas de hedge
                "risk_reductions": False,   # Pas de réductions risque
                "emergency_exits": True,    # Seules sorties d'urgence autorisées
            }
        elif freeze_type == FreezeType.S3_ALERT_FREEZE:
            return {
                "new_purchases": False,     # Pas de nouveaux achats
                "sell_to_stables": True,    # Rotations↓ stables OK
                "asset_rotations": False,   # Pas de rotations entre risky assets
                "hedge_operations": True,   # Hedge autorisé (protection)
                "risk_reductions": True,    # Réductions risque autorisées
                "emergency_exits": True,    # Sorties d'urgence toujours OK
            }
        elif freeze_type == FreezeType.ERROR_FREEZE:
            return {
                "new_purchases": False,     # Pas de nouveaux achats
                "sell_to_stables": True,    # Ventes vers stables OK
                "asset_rotations": False,   # Pas de rotations risquées
                "hedge_operations": True,   # Hedge OK
                "risk_reductions": True,    # Réductions risque prioritaires
                "emergency_exits": True,    # Sorties d'urgence toujours OK
            }
        else:
            # Mode normal : tout autorisé
            return {
                "new_purchases": True,
                "sell_to_stables": True,
                "asset_rotations": True,
                "hedge_operations": True,
                "risk_reductions": True,
                "emergency_exits": True,
            }

    @staticmethod
    def validate_operation(freeze_type: str, operation_type: str) -> Tuple[bool, str]:
        """
        Valide si une opération est autorisée selon le freeze actuel

        Returns:
            (allowed: bool, reason: str)
        """
        allowed_ops = FreezeSemantics.get_allowed_operations(freeze_type)

        if operation_type not in allowed_ops:
            return False, f"Operation type '{operation_type}' not recognized"

        is_allowed = allowed_ops[operation_type]

        if not is_allowed:
            reason = f"{operation_type} blocked by {freeze_type}"
        else:
            reason = f"{operation_type} allowed under {freeze_type}"

        return is_allowed, reason

    @staticmethod
    def get_freeze_caps(freeze_type: str) -> Tuple[float, int]:
        """
        Retourne les caps et ramp_hours selon le type de freeze

        Returns:
            (cap_daily: float, ramp_hours: int)
        """
        if freeze_type == FreezeType.FULL_FREEZE:
            return 0.01, 1  # Très restrictif
        elif freeze_type == FreezeType.S3_ALERT_FREEZE:
            return 0.03, 6  # Permet réductions risque
        elif freeze_type == FreezeType.ERROR_FREEZE:
            return 0.05, 12  # Permet hedge et réductions
        else:
            return 0.08, 12  # Normal

    @staticmethod
    def get_status_message(freeze_type: str) -> str:
        """Retourne le message de statut selon le type de freeze"""
        if freeze_type == FreezeType.S3_ALERT_FREEZE:
            return "S3 Alert Freeze: purchases blocked, risk reductions allowed"
        elif freeze_type == FreezeType.ERROR_FREEZE:
            return "Error Freeze: purchases blocked, hedge & reductions allowed"
        elif freeze_type == FreezeType.FULL_FREEZE:
            return "Full Freeze: only emergency exits allowed"
        else:
            return f"Freeze active: {freeze_type}"

    @staticmethod
    def infer_freeze_type(reason: str) -> str:
        """
        Infère le type de freeze depuis la raison

        Args:
            reason: Raison du freeze

        Returns:
            Type de freeze approprié
        """
        reason_lower = reason.lower()
        if "s3" in reason_lower or "alert" in reason_lower:
            return FreezeType.S3_ALERT_FREEZE
        elif "error" in reason_lower or "backend" in reason_lower:
            return FreezeType.ERROR_FREEZE
        else:
            return FreezeType.FULL_FREEZE
