"""
Circuit Breaker — Protection des appels API externes

Pattern: CLOSED → OPEN (après N échecs) → HALF_OPEN (après timeout) → CLOSED (si succès)

Instances pré-configurées:
- coingecko_circuit: 5 failures, 60s recovery
- fred_circuit: 3 failures, 120s recovery
- saxo_circuit: 5 failures, 60s recovery
"""
from __future__ import annotations

import time
import threading
import logging
from enum import Enum
from typing import Optional, Dict

log = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, circuit_name: str, recovery_remaining: float = 0):
        self.circuit_name = circuit_name
        self.recovery_remaining = recovery_remaining
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN — call rejected "
            f"(recovery in {recovery_remaining:.0f}s)"
        )


class CircuitBreaker:
    """
    Lightweight circuit breaker for external API resilience.

    States:
    - CLOSED: Normal operation. Tracks consecutive failures.
    - OPEN: All calls fail-fast. Transitions to HALF_OPEN after recovery_timeout.
    - HALF_OPEN: Allows ONE test call. Success → CLOSED, Failure → OPEN.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN and self._last_failure_time:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    log.info(f"Circuit '{self.name}': OPEN -> HALF_OPEN (recovery timeout elapsed)")
            return self._state

    def is_available(self) -> bool:
        """Check if calls are allowed through this circuit."""
        return self.state != CircuitState.OPEN

    def record_success(self):
        """Record a successful call. Resets failure count."""
        with self._lock:
            prev = self._state
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            if prev == CircuitState.HALF_OPEN:
                log.info(f"Circuit '{self.name}': HALF_OPEN -> CLOSED (test call succeeded)")

    def record_failure(self):
        """Record a failed call. May trip the circuit to OPEN."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                log.warning(f"Circuit '{self.name}': HALF_OPEN -> OPEN (test call failed)")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                log.warning(
                    f"Circuit '{self.name}': CLOSED -> OPEN "
                    f"({self._failure_count} consecutive failures, "
                    f"recovery in {self.recovery_timeout}s)"
                )

    def raise_if_open(self):
        """Raise CircuitOpenError if circuit is OPEN. Convenience method."""
        if not self.is_available():
            remaining = 0.0
            if self._last_failure_time:
                remaining = max(
                    0.0,
                    self.recovery_timeout - (time.monotonic() - self._last_failure_time),
                )
            raise CircuitOpenError(self.name, remaining)

    def get_status(self) -> Dict:
        """Get current status for monitoring/health-check endpoints."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_s": self.recovery_timeout,
        }


# ── Pre-configured instances ──────────────────────────────────────────

coingecko_circuit = CircuitBreaker("coingecko", failure_threshold=5, recovery_timeout=60)
fred_circuit = CircuitBreaker("fred", failure_threshold=3, recovery_timeout=120)
saxo_circuit = CircuitBreaker("saxo", failure_threshold=5, recovery_timeout=60)
