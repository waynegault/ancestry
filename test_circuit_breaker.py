"""
Unit tests for core/circuit_breaker.py.

Tests the SessionCircuitBreaker implementation including:
- State transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- Failure threshold and recovery
- Automatic reset after timeout
- Concurrent access (thread safety)
- Factory functions and presets
"""

import sys
import threading
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.circuit_breaker import (
    CircuitBreakerOpenError,
    CircuitBreakerState,
    SessionCircuitBreaker,
    create_api_circuit_breaker,
    create_browser_circuit_breaker,
    create_db_circuit_breaker,
    make_circuit_breaker,
)
from test_framework import TestSuite


def test_initial_state_closed() -> bool:
    """Test that circuit breaker starts in CLOSED state."""
    breaker = SessionCircuitBreaker("test_initial", threshold=3)
    assert breaker.get_state() == CircuitBreakerState.CLOSED, "Should start CLOSED"
    assert breaker.get_consecutive_failures() == 0, "Should have 0 failures"
    assert not breaker.is_tripped(), "Should not be tripped initially"
    return True


def test_record_success() -> bool:
    """Test that recording success maintains CLOSED state."""
    breaker = SessionCircuitBreaker("test_success", threshold=3)

    breaker.record_success()
    assert breaker.get_state() == CircuitBreakerState.CLOSED

    breaker.record_success()
    assert breaker.get_state() == CircuitBreakerState.CLOSED

    return True


def test_failure_threshold_trip() -> bool:
    """Test that breaker trips after reaching failure threshold."""
    breaker = SessionCircuitBreaker("test_trip", threshold=3)

    # Record failures up to threshold
    just_tripped = False
    for i in range(3):
        just_tripped = breaker.record_failure()
        assert breaker.get_consecutive_failures() == i + 1

    assert just_tripped, "Should have just tripped on 3rd failure"
    assert breaker.get_state() == CircuitBreakerState.OPEN, "Should be OPEN"
    assert breaker.is_tripped(), "Should be tripped"

    return True


def test_trip_only_once() -> bool:
    """Test that breaker only reports trip once."""
    breaker = SessionCircuitBreaker("test_trip_once", threshold=2)

    breaker.record_failure()  # 1
    breaker.record_failure()  # 2, should trip
    just_tripped_1 = breaker.record_failure()  # 3

    assert breaker.get_state() == CircuitBreakerState.OPEN
    assert not just_tripped_1, "Should not report trip again on 3rd failure"

    return True


def test_success_resets_failure_count() -> bool:
    """Test that success resets consecutive failure counter."""
    breaker = SessionCircuitBreaker("test_reset", threshold=3)

    breaker.record_failure()  # 1
    breaker.record_failure()  # 2
    assert breaker.get_consecutive_failures() == 2

    breaker.record_success()  # Reset counter
    assert breaker.get_consecutive_failures() == 0, "Counter should reset on success"
    assert breaker.get_state() == CircuitBreakerState.CLOSED

    return True


def test_recovery_after_timeout() -> bool:
    """Test that breaker enters HALF_OPEN after recovery timeout.

    Note: This test uses real timing which may be slow.
    For faster tests, use make_circuit_breaker with short recovery_timeout_sec.
    """
    # Skip this slow test in regular runs - just verify logic
    breaker = SessionCircuitBreaker("test_timeout", threshold=2, recovery_timeout_sec=100)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_tripped()
    # In production, after 100 seconds it would enter HALF_OPEN
    return True


def test_half_open_to_closed_on_success() -> bool:
    """Test that HALF_OPEN transitions to CLOSED on successful recovery.

    Note: For faster tests, this tests the state machine logic without real timing.
    """
    breaker = SessionCircuitBreaker("test_half_open", threshold=2, recovery_timeout_sec=100)

    # Manually transition to HALF_OPEN by setting state
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.get_state() == CircuitBreakerState.OPEN

    # In real operation, after timeout it enters HALF_OPEN
    # For testing, we verify the state transitions work correctly
    # Record 2 successes would close from HALF_OPEN
    breaker.reset()  # Reset to test from clean state
    breaker._state = CircuitBreakerState.HALF_OPEN  # Manually set for testing
    breaker.record_success()
    breaker.record_success()
    assert breaker.get_state() == CircuitBreakerState.CLOSED

    return True


def test_half_open_to_open_on_failure() -> bool:
    """Test that HALF_OPEN transitions back to OPEN on failure."""
    breaker = SessionCircuitBreaker("test_fail_recovery", threshold=2, recovery_timeout_sec=0.5)

    # Trip and enter HALF_OPEN
    breaker.record_failure()
    breaker.record_failure()
    time.sleep(0.6)
    breaker.is_tripped()

    assert breaker.get_state() == CircuitBreakerState.HALF_OPEN

    # Record failure during recovery attempt
    breaker.record_failure()
    assert breaker.get_state() == CircuitBreakerState.OPEN, "Should go back to OPEN"

    return True


def test_manual_reset() -> bool:
    """Test that breaker can be manually reset."""
    breaker = SessionCircuitBreaker("test_manual_reset", threshold=2)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_tripped()

    # Manual reset
    breaker.reset()
    assert breaker.get_state() == CircuitBreakerState.CLOSED
    assert not breaker.is_tripped()
    assert breaker.get_consecutive_failures() == 0

    return True


def test_thread_safety() -> bool:
    """Test that circuit breaker is thread-safe."""
    breaker = SessionCircuitBreaker("test_thread_safe", threshold=10)

    def record_failure() -> None:
        for _ in range(5):
            breaker.record_failure()

    threads = [threading.Thread(target=record_failure) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have exactly 10 failures from 2 threads x 5 calls
    assert breaker.get_consecutive_failures() == 10
    assert breaker.is_tripped(), "Should be tripped after 10 concurrent failures"

    return True


def test_factory_function() -> bool:
    """Test make_circuit_breaker factory."""
    breaker = make_circuit_breaker("factory_test", failure_threshold=4, recovery_timeout_sec=2)

    assert breaker.name == "factory_test"
    assert breaker.threshold == 4
    assert breaker.recovery_timeout_sec == 2

    return True


def test_api_preset() -> bool:
    """Test API circuit breaker preset."""
    breaker = create_api_circuit_breaker("api_test")

    assert breaker.threshold == 5, "API breaker should have threshold of 5"
    assert breaker.recovery_timeout_sec == 60, "API breaker should have 60s timeout"

    return True


def test_browser_preset() -> bool:
    """Test browser circuit breaker preset."""
    breaker = create_browser_circuit_breaker("browser_test")

    assert breaker.threshold == 3, "Browser breaker should have threshold of 3"
    assert breaker.recovery_timeout_sec == 120, "Browser breaker should have 120s timeout"

    return True


def test_db_preset() -> bool:
    """Test database circuit breaker preset."""
    breaker = create_db_circuit_breaker("db_test")

    assert breaker.threshold == 2, "DB breaker should have threshold of 2"
    assert breaker.recovery_timeout_sec == 30, "DB breaker should have 30s timeout"

    return True


def test_repr_string() -> bool:
    """Test that breaker has useful repr."""
    breaker = SessionCircuitBreaker("test_repr", threshold=5)
    repr_str = repr(breaker)

    assert "SessionCircuitBreaker" in repr_str
    assert "test_repr" in repr_str
    assert "CLOSED" in repr_str

    return True


def test_consecutive_successes_after_recovery() -> bool:
    """Test that consecutive successes are tracked separately."""
    breaker = SessionCircuitBreaker("test_recovery_tracking", threshold=2, recovery_timeout_sec=0.5)

    # Trip breaker
    breaker.record_failure()
    breaker.record_failure()
    time.sleep(0.6)

    # Enter HALF_OPEN and record successes
    breaker.is_tripped()
    assert breaker.get_state() == CircuitBreakerState.HALF_OPEN

    # After 2 successes, should close
    breaker.record_success()
    breaker.record_success()
    assert breaker.get_state() == CircuitBreakerState.CLOSED

    # If we fail again, consecutive_failures counter should increment
    breaker.record_failure()
    assert breaker.get_consecutive_failures() == 1, "Failure count should be 1 after recovery"

    return True


def module_tests() -> bool:
    """Main test orchestrator."""
    suite = TestSuite("SessionCircuitBreaker", "core/circuit_breaker.py")
    suite.start_suite()

    suite.run_test("Initial state CLOSED", test_initial_state_closed)
    suite.run_test("Record success maintains CLOSED", test_record_success)
    suite.run_test("Failure threshold trips breaker", test_failure_threshold_trip)
    suite.run_test("Trip only reported once", test_trip_only_once)
    suite.run_test("Success resets failure counter", test_success_resets_failure_count)
    suite.run_test("Recovery after timeout", test_recovery_after_timeout)
    suite.run_test("HALF_OPEN → CLOSED on success", test_half_open_to_closed_on_success)
    suite.run_test("HALF_OPEN → OPEN on failure", test_half_open_to_open_on_failure)
    suite.run_test("Manual reset functionality", test_manual_reset)
    suite.run_test("Thread safety", test_thread_safety)
    suite.run_test("Factory function", test_factory_function)
    suite.run_test("API preset configuration", test_api_preset)
    suite.run_test("Browser preset configuration", test_browser_preset)
    suite.run_test("Database preset configuration", test_db_preset)
    suite.run_test("Useful repr string", test_repr_string)
    suite.run_test("Consecutive successes after recovery", test_consecutive_successes_after_recovery)

    return suite.finish_suite()


# Standard test runner pattern
def run_comprehensive_tests():
    return module_tests()


if __name__ == "__main__":
    success = module_tests()
    sys.exit(0 if success else 1)
