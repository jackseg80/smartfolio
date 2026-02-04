"""
Phase Context Module - Gestion de la phase laggée avec persistance

Ce module gère:
- PhaseSnapshot: Snapshot de phase avec timestamp pour le lag
- PhaseAwareContext: Gestionnaire de phase laggée avec persistance anti-oscillation

Extrait de alert_engine.py pour modularité.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from ..execution.phase_engine import Phase, PhaseState

logger = logging.getLogger(__name__)


@dataclass
class PhaseSnapshot:
    """Snapshot de phase avec timestamp pour le lag"""
    phase: Phase
    confidence: float
    persistence_count: int
    captured_at: datetime
    contradiction_index: float = 0.0


class PhaseAwareContext:
    """
    Gestionnaire de phase laggée avec persistance pour anti-oscillation

    Responsabilités:
    - Gère l'historique des phases
    - Calcule la phase laggée (avec délai configurable)
    - Vérifie la stabilité de la phase
    """

    def __init__(self, lag_minutes: int = 15, persistence_ticks: int = 3, metrics=None):
        """
        Initialise le gestionnaire de phase laggée.

        Args:
            lag_minutes: Délai de lag pour la phase (défaut: 15 min)
            persistence_ticks: Nombre de ticks requis pour persistance (défaut: 3)
            metrics: Instance de métriques Prometheus (optionnel)
        """
        self.lag_minutes = lag_minutes
        self.persistence_ticks = persistence_ticks
        self.phase_history: List[PhaseSnapshot] = []
        self.current_lagged_phase: Optional[PhaseSnapshot] = None
        self.metrics = metrics

    def update_phase(self, phase_state: PhaseState, contradiction_index: float = 0.0) -> Optional[PhaseSnapshot]:
        """
        Met à jour l'historique de phase et calcule la phase laggée.

        Args:
            phase_state: État de phase actuel depuis PhaseEngine
            contradiction_index: Index de contradiction actuel

        Returns:
            PhaseSnapshot laggée ou None si pas encore stable
        """
        now = datetime.utcnow()

        # Ajouter le snapshot actuel
        snapshot = PhaseSnapshot(
            phase=phase_state.phase_now,
            confidence=phase_state.confidence,
            persistence_count=phase_state.persistence_count,
            captured_at=now,
            contradiction_index=contradiction_index
        )

        self.phase_history.append(snapshot)

        # Nettoyer l'historique > 2 * lag_minutes
        cutoff = now - timedelta(minutes=self.lag_minutes * 2)
        self.phase_history = [s for s in self.phase_history if s.captured_at > cutoff]

        # Calculer la phase laggée
        lag_cutoff = now - timedelta(minutes=self.lag_minutes)
        lagged_snapshots = [s for s in self.phase_history if s.captured_at <= lag_cutoff]

        if lagged_snapshots:
            # Prendre le plus récent dans la fenêtre laggée
            candidate = max(lagged_snapshots, key=lambda x: x.captured_at)

            # Vérifier la persistance: phases similaires consécutives
            if candidate.persistence_count >= self.persistence_ticks:
                # Record phase transition if phase changed
                if self.current_lagged_phase and self.current_lagged_phase.phase != candidate.phase:
                    if self.metrics:
                        self.metrics.record_phase_transition(
                            self.current_lagged_phase.phase.value.lower(),
                            candidate.phase.value.lower()
                        )

                self.current_lagged_phase = candidate

                # Update current phase metrics
                if self.metrics:
                    self.metrics.update_current_lagged_phase(
                        candidate.phase.value.lower(),
                        candidate.persistence_count
                    )

                logger.debug(f"Phase laggée mise à jour: {candidate.phase.value} "
                           f"(persistance: {candidate.persistence_count}, "
                           f"contradiction: {candidate.contradiction_index:.2f})")

        return self.current_lagged_phase

    def get_lagged_phase(self) -> Optional[PhaseSnapshot]:
        """Retourne la phase laggée actuelle."""
        return self.current_lagged_phase

    def is_phase_stable(self) -> bool:
        """Vérifie si la phase laggée est stable (persistance suffisante)."""
        if not self.current_lagged_phase:
            return False
        return self.current_lagged_phase.persistence_count >= self.persistence_ticks

    def get_phase_history_length(self) -> int:
        """Retourne la longueur de l'historique des phases."""
        return len(self.phase_history)

    def reset(self) -> None:
        """Réinitialise l'historique des phases."""
        self.phase_history = []
        self.current_lagged_phase = None
        logger.info("Phase context reset")
