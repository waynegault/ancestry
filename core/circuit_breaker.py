"""
Standardized Circuit Breaker Pattern Implementation for Core Orchestration.

Extracted and re-exported from error_handling.py to provide:
1. Easy access for all action modules
2. Consistent configuration across the system
3. Optional integration with SessionManager for session health checks
4. Pre-configured defaults for common use cases

This module serves as the public API for circuit breaker functionality,
complementing the advanced retry and error handling capabilities in
error_handling.py.

Usage:
    from core.circuit_breaker import SessionCircuitBreaker, make_circuit_breaker

    # Simple usage with action
    breaker = SessionCircuitBreaker(name="action_6", threshold=5)
    if breaker.is_tripped():
        raise CircuitBreakerOpenError("Circuit breaker tripped")

    # With session integration
    breaker = make_circuit_breaker(
        name="action_6_api",
        session_manager=session_manager,
        failure_threshold=5,
    )
"""

import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and call is rejected."""
    pass


class CircuitBreakerState:
    """States for circuit breaker state machine."""
    CLOSED = "CLOSED"        # Normal operation
    OPEN = "OPEN"            # Failing, calls rejected
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class SessionCircuitBreaker:
    """
    Simplified circuit breaker for session-based operations.

    Tracks consecutive failures and trips if threshold exceeded.
    Supports automatic reset after timeout.

    Configuration:
      failure_threshold: Number of consecutive failures to trip (default 5)
      recovery_timeout_sec: Seconds before attempting recovery (default 60)
    """

    def __init__(
        self,
        name: str,
        threshold: int = 5,
        recovery_timeout_sec: int | float = 60,
        session_manager: Optional[Any] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for logging
            threshold: Consecutive failures to trip (default 5)
            recovery_timeout_sec: Seconds before attempting recovery (default 60)
            session_manager: Optional SessionManager for health checks
        """
        self.name = name
        self.threshold = threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.session_manager = session_manager

        self._state = CircuitBreakerState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._trip_time: Optional[datetime] = None
        self._lock = Lock()

    def record_success(self) -> None:
        """Record successful call and potentially close breaker."""
        with self._lock:
            self._consecutive_failures = 0
            self._consecutive_successes += 1
            self._last_success_time = datetime.now()

            # If half-open and hit success threshold, close
            if self._state == CircuitBreakerState.HALF_OPEN and self._consecutive_successes >= 2:
                self._state = CircuitBreakerState.CLOSED
                logger.info(f"🟢 Circuit breaker '{self.name}' CLOSED (recovery successful)")

    def record_failure(self) -> bool:
        """Record failure and trip if threshold exceeded.

        Returns:
            True if just tripped, False otherwise
        """
        with self._lock:
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_failure_time = datetime.now()

            just_tripped = False
            if self._consecutive_failures >= self.threshold and self._state != CircuitBreakerState.OPEN:
                self._state = CircuitBreakerState.OPEN
                self._trip_time = datetime.now()
                logger.error(
                    f"🔴 Circuit breaker '{self.name}' OPEN after {self.threshold} failures"
                )
                just_tripped = True

            return just_tripped

    def is_tripped(self) -> bool:
        """Check if breaker is currently tripped."""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                if self._trip_time:
                    elapsed = (datetime.now() - self._trip_time).total_seconds()
                    if elapsed >= self.recovery_timeout_sec:
                        self._state = CircuitBreakerState.HALF_OPEN
                        self._consecutive_successes = 0
                        logger.info(
                            f"🟡 Circuit breaker '{self.name}' HALF_OPEN (attempting recovery)"
                        )
                        return False  # Not tripped anymore

                return True
            return False

    def get_state(self) -> str:
        """Get current state string."""
        with self._lock:
            return self._state

    def get_consecutive_failures(self) -> int:
        """Get count of consecutive failures."""
        with self._lock:
            return self._consecutive_failures

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._trip_time = None
            logger.debug(f"🔵 Circuit breaker '{self.name}' manually RESET")

    def __repr__(self) -> str:
        state = self.get_state()
        failures = self.get_consecutive_failures()
        return f"<SessionCircuitBreaker '{self.name}' state={state} failures={failures}/{self.threshold}>"


def make_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout_sec: int = 60,
    session_manager: Optional[Any] = None,
) -> SessionCircuitBreaker:
    """Factory function for creating circuit breakers.

    Provides a convenient way to instantiate circuit breakers with
    consistent defaults across the codebase.

    Args:
        name: Identifier for logging
        failure_threshold: Consecutive failures to trip (default 5)
        recovery_timeout_sec: Seconds before recovery attempt (default 60)
        session_manager: Optional SessionManager for integration

    Returns:
        SessionCircuitBreaker instance

    Example:
        breaker = make_circuit_breaker("api_fetch", failure_threshold=3)
        if breaker.is_tripped():
            raise CircuitBreakerOpenError("Too many failures")

        try:
            result = fetch_data()
            breaker.record_success()
        except Exception as e:
            if breaker.record_failure():
                logger.error("Circuit breaker tripped!")
            raise
    """
    return SessionCircuitBreaker(
        name=name,
        threshold=failure_threshold,
        recovery_timeout_sec=recovery_timeout_sec,
        session_manager=session_manager,
    )


# --- Module-Level Presets ---
# Common configurations for specific use cases

def create_api_circuit_breaker(name: str = "api_default") -> SessionCircuitBreaker:
    """Create circuit breaker for API operations (conservative: 5 failures)."""
    return make_circuit_breaker(name, failure_threshold=5, recovery_timeout_sec=60)


def create_browser_circuit_breaker(name: str = "browser_default") -> SessionCircuitBreaker:
    """Create circuit breaker for browser operations (conservative: 3 failures)."""
    return make_circuit_breaker(name, failure_threshold=3, recovery_timeout_sec=120)


def create_db_circuit_breaker(name: str = "db_default") -> SessionCircuitBreaker:
    """Create circuit breaker for database operations (strict: 2 failures)."""
    return make_circuit_breaker(name, failure_threshold=2, recovery_timeout_sec=30)


# --- Re-exports from error_handling for convenience ---

try:
    from error_handling import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitState,
        IntelligentRetryHandler,
        RetryConfig,
        RetryStrategy,
    )

    __all__ = [
        # Advanced (re-exported from error_handling)
        "CircuitBreaker",
        "CircuitBreakerConfig",
        "CircuitBreakerOpenError",
        "CircuitBreakerState",
        "CircuitState",
        "IntelligentRetryHandler",
        "RetryConfig",
        "RetryStrategy",
        # Session-based (simplified)
        "SessionCircuitBreaker",
        "create_api_circuit_breaker",
        "create_browser_circuit_breaker",
        "create_db_circuit_breaker",
        "make_circuit_breaker",
    ]

except ImportError as e:
    logger.warning(f"Could not import advanced circuit breaker classes: {e}")
    __all__ = [
        "CircuitBreakerOpenError",
        "CircuitBreakerState",
        "SessionCircuitBreaker",
        "create_api_circuit_breaker",
        "create_browser_circuit_breaker",
        "create_db_circuit_breaker",
        "make_circuit_breaker",
    ]
