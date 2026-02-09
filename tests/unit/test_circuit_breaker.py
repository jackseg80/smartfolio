"""
Unit tests for shared.circuit_breaker module.

Tests the CircuitBreaker pattern: CLOSED -> OPEN -> HALF_OPEN -> CLOSED,
including state transitions, error handling, and pre-configured instances.
"""
import pytest
from unittest.mock import patch

from shared.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    coingecko_circuit,
    fred_circuit,
    saxo_circuit,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def cb():
    """A fresh CircuitBreaker with threshold=3, recovery=30s for easy testing."""
    return CircuitBreaker("test", failure_threshold=3, recovery_timeout=30.0)


@pytest.fixture
def cb_default():
    """A fresh CircuitBreaker with default parameters."""
    return CircuitBreaker("default")


# ── 1. Initial state ─────────────────────────────────────────────────


class TestInitialState:
    def test_initial_state_is_closed(self, cb):
        assert cb.state == CircuitState.CLOSED

    def test_initial_is_available(self, cb):
        assert cb.is_available() is True

    def test_initial_failure_count_zero(self, cb):
        status = cb.get_status()
        assert status["failure_count"] == 0

    def test_initial_name_stored(self, cb):
        assert cb.name == "test"

    def test_default_parameters(self, cb_default):
        assert cb_default.failure_threshold == 5
        assert cb_default.recovery_timeout == 60.0


# ── 2. Failures below threshold keep CLOSED ──────────────────────────


class TestBelowThreshold:
    def test_one_failure_stays_closed(self, cb):
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_two_failures_stays_closed(self, cb):
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_below_threshold_still_available(self, cb):
        for _ in range(cb.failure_threshold - 1):
            cb.record_failure()
        assert cb.is_available() is True

    def test_failure_count_increments(self, cb):
        cb.record_failure()
        cb.record_failure()
        assert cb.get_status()["failure_count"] == 2


# ── 3. Failures at threshold trip to OPEN ─────────────────────────────


class TestTripsToOpen:
    def test_exact_threshold_opens_circuit(self, cb):
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_over_threshold_stays_open(self, cb):
        for _ in range(cb.failure_threshold + 2):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_failure_count_at_threshold(self, cb):
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.get_status()["failure_count"] == cb.failure_threshold


# ── 4. OPEN state rejects calls ───────────────────────────────────────


class TestOpenRejects:
    def test_open_not_available(self, cb):
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.is_available() is False

    def test_open_state_value(self, cb):
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN


# ── 5. raise_if_open raises CircuitOpenError when OPEN ────────────────


class TestRaiseIfOpen:
    def test_raises_when_open(self, cb):
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        with pytest.raises(CircuitOpenError):
            cb.raise_if_open()

    def test_does_not_raise_when_closed(self, cb):
        # Should not raise
        cb.raise_if_open()

    def test_does_not_raise_after_success(self, cb):
        cb.record_failure()
        cb.record_success()
        cb.raise_if_open()

    def test_raise_includes_recovery_remaining(self, cb):
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()

            # Advance 10s — still OPEN, recovery_remaining ~ 20s
            mock_time.monotonic.return_value = base_time + 10.0
            with pytest.raises(CircuitOpenError) as exc_info:
                cb.raise_if_open()
            assert exc_info.value.recovery_remaining == pytest.approx(20.0, abs=1.0)

    def test_raise_if_open_in_half_open_does_not_raise(self, cb):
        """HALF_OPEN allows a test call, so raise_if_open should not raise."""
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()

            # Advance past recovery timeout -> HALF_OPEN
            mock_time.monotonic.return_value = base_time + cb.recovery_timeout + 1
            assert cb.state == CircuitState.HALF_OPEN
            # Should not raise since HALF_OPEN is available
            cb.raise_if_open()


# ── 6. OPEN -> HALF_OPEN after recovery_timeout ──────────────────────


class TestRecoveryTimeout:
    def test_transitions_to_half_open_after_timeout(self, cb):
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()
            assert cb.state == CircuitState.OPEN

            # Just before timeout — still OPEN
            mock_time.monotonic.return_value = base_time + cb.recovery_timeout - 0.1
            assert cb.state == CircuitState.OPEN

            # Exactly at timeout — transitions to HALF_OPEN
            mock_time.monotonic.return_value = base_time + cb.recovery_timeout
            assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_is_available(self, cb):
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()

            mock_time.monotonic.return_value = base_time + cb.recovery_timeout + 1
            assert cb.is_available() is True

    def test_open_not_available_before_timeout(self, cb):
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()

            mock_time.monotonic.return_value = base_time + 5.0
            assert cb.is_available() is False


# ── 7. HALF_OPEN + success -> CLOSED ─────────────────────────────────


class TestHalfOpenSuccess:
    def test_success_in_half_open_closes_circuit(self, cb):
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()

            # Transition to HALF_OPEN
            mock_time.monotonic.return_value = base_time + cb.recovery_timeout + 1
            assert cb.state == CircuitState.HALF_OPEN

            # Record success
            cb.record_success()
            assert cb.state == CircuitState.CLOSED

    def test_success_in_half_open_resets_failures(self, cb):
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()

            mock_time.monotonic.return_value = base_time + cb.recovery_timeout + 1
            assert cb.state == CircuitState.HALF_OPEN

            cb.record_success()
            assert cb.get_status()["failure_count"] == 0

    def test_circuit_fully_operational_after_half_open_success(self, cb):
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()

            mock_time.monotonic.return_value = base_time + cb.recovery_timeout + 1
            cb.record_success()

            # Should be fully operational — can record failures again before tripping
            mock_time.monotonic.return_value = base_time + cb.recovery_timeout + 2
            cb.record_failure()
            assert cb.state == CircuitState.CLOSED
            assert cb.is_available() is True


# ── 8. HALF_OPEN + failure -> OPEN ───────────────────────────────────


class TestHalfOpenFailure:
    def test_failure_in_half_open_opens_circuit_immediately(self, cb):
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()

            # Transition to HALF_OPEN
            mock_time.monotonic.return_value = base_time + cb.recovery_timeout + 1
            assert cb.state == CircuitState.HALF_OPEN

            # Single failure in HALF_OPEN trips back to OPEN immediately
            cb.record_failure()
            # Need to check state without another time advance that would re-trigger HALF_OPEN
            assert cb._state == CircuitState.OPEN

    def test_half_open_failure_resets_recovery_timer(self, cb):
        """After HALF_OPEN fails, a new recovery period starts from the failure time."""
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()

            # Transition to HALF_OPEN at base_time + 31
            half_open_time = base_time + cb.recovery_timeout + 1
            mock_time.monotonic.return_value = half_open_time
            assert cb.state == CircuitState.HALF_OPEN

            # Fail in HALF_OPEN — this records a new _last_failure_time
            cb.record_failure()

            # The old recovery timeout from base_time would have passed, but
            # we need to wait recovery_timeout from the NEW failure time
            mock_time.monotonic.return_value = half_open_time + cb.recovery_timeout - 1
            assert cb.state == CircuitState.OPEN

            # Now past the new recovery timeout
            mock_time.monotonic.return_value = half_open_time + cb.recovery_timeout + 1
            assert cb.state == CircuitState.HALF_OPEN


# ── 9. record_success resets failure count ────────────────────────────


class TestRecordSuccessResets:
    def test_success_resets_failure_count(self, cb):
        cb.record_failure()
        cb.record_failure()
        assert cb.get_status()["failure_count"] == 2
        cb.record_success()
        assert cb.get_status()["failure_count"] == 0

    def test_success_keeps_closed(self, cb):
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_success_without_prior_failures(self, cb):
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.get_status()["failure_count"] == 0

    def test_success_prevents_threshold_accumulation(self, cb):
        """Two failures, reset, two more failures should not trip (total < threshold)."""
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED


# ── 10. get_status returns correct dict ───────────────────────────────


class TestGetStatus:
    def test_status_keys(self, cb):
        status = cb.get_status()
        expected_keys = {
            "name",
            "state",
            "failure_count",
            "failure_threshold",
            "recovery_timeout_s",
        }
        assert set(status.keys()) == expected_keys

    def test_status_closed_values(self, cb):
        status = cb.get_status()
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 3
        assert status["recovery_timeout_s"] == 30.0

    def test_status_reflects_open(self, cb):
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        status = cb.get_status()
        assert status["state"] == "open"
        assert status["failure_count"] == cb.failure_threshold

    def test_status_reflects_half_open(self, cb):
        base_time = 1000.0
        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            for _ in range(cb.failure_threshold):
                cb.record_failure()

            mock_time.monotonic.return_value = base_time + cb.recovery_timeout + 1
            status = cb.get_status()
            assert status["state"] == "half_open"

    def test_status_state_is_string(self, cb):
        """State value in status dict is the string value, not the enum."""
        status = cb.get_status()
        assert isinstance(status["state"], str)


# ── 11. Pre-configured instances ──────────────────────────────────────


class TestPreConfiguredInstances:
    def test_coingecko_circuit_exists(self):
        assert isinstance(coingecko_circuit, CircuitBreaker)

    def test_coingecko_circuit_params(self):
        assert coingecko_circuit.name == "coingecko"
        assert coingecko_circuit.failure_threshold == 5
        assert coingecko_circuit.recovery_timeout == 60

    def test_fred_circuit_exists(self):
        assert isinstance(fred_circuit, CircuitBreaker)

    def test_fred_circuit_params(self):
        assert fred_circuit.name == "fred"
        assert fred_circuit.failure_threshold == 3
        assert fred_circuit.recovery_timeout == 120

    def test_saxo_circuit_exists(self):
        assert isinstance(saxo_circuit, CircuitBreaker)

    def test_saxo_circuit_params(self):
        assert saxo_circuit.name == "saxo"
        assert saxo_circuit.failure_threshold == 5
        assert saxo_circuit.recovery_timeout == 60


# ── 12. CircuitOpenError attributes ──────────────────────────────────


class TestCircuitOpenError:
    def test_error_is_exception(self):
        err = CircuitOpenError("test_api", recovery_remaining=42.0)
        assert isinstance(err, Exception)

    def test_error_circuit_name(self):
        err = CircuitOpenError("my_api", recovery_remaining=10.0)
        assert err.circuit_name == "my_api"

    def test_error_recovery_remaining(self):
        err = CircuitOpenError("my_api", recovery_remaining=25.5)
        assert err.recovery_remaining == 25.5

    def test_error_default_recovery_remaining(self):
        err = CircuitOpenError("my_api")
        assert err.recovery_remaining == 0

    def test_error_message_contains_circuit_name(self):
        err = CircuitOpenError("coingecko", recovery_remaining=30.0)
        assert "coingecko" in str(err)

    def test_error_message_contains_open(self):
        err = CircuitOpenError("fred", recovery_remaining=10.0)
        assert "OPEN" in str(err)

    def test_error_can_be_raised_and_caught(self):
        with pytest.raises(CircuitOpenError) as exc_info:
            raise CircuitOpenError("test", recovery_remaining=5.0)
        assert exc_info.value.circuit_name == "test"
        assert exc_info.value.recovery_remaining == 5.0


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_threshold_of_one(self):
        """A circuit with threshold=1 trips on the very first failure."""
        cb = CircuitBreaker("fragile", failure_threshold=1, recovery_timeout=10.0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_full_cycle_closed_open_half_open_closed(self):
        """Test the complete lifecycle: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
        cb = CircuitBreaker("lifecycle", failure_threshold=2, recovery_timeout=20.0)
        base_time = 500.0

        with patch("shared.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time

            # CLOSED
            assert cb.state == CircuitState.CLOSED

            # Trip to OPEN
            cb.record_failure()
            cb.record_failure()
            assert cb.state == CircuitState.OPEN
            assert cb.is_available() is False

            # Wait for recovery -> HALF_OPEN
            mock_time.monotonic.return_value = base_time + 21.0
            assert cb.state == CircuitState.HALF_OPEN
            assert cb.is_available() is True

            # Success -> CLOSED
            cb.record_success()
            assert cb.state == CircuitState.CLOSED
            assert cb.is_available() is True
            assert cb.get_status()["failure_count"] == 0

    def test_multiple_successes_idempotent(self, cb):
        """Multiple successes in a row keep the circuit closed and healthy."""
        cb.record_success()
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.get_status()["failure_count"] == 0

    def test_circuit_state_enum_values(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"
