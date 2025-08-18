#!/usr/bin/env python3
"""
Circuit breaker pattern implementation for Action 6.
Prevents cascading failures by temporarily stopping API calls when failure rate is high.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker to prevent cascade failures."""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("🔄 Circuit breaker: Attempting reset (HALF_OPEN)")
            else:
                raise Exception("Circuit breaker OPEN - blocking call to prevent cascade failure")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("✅ Circuit breaker: Reset successful (CLOSED)")
            self.state = CircuitState.CLOSED
        self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.critical(
                    f"🚨 Circuit breaker OPENED: {self.failure_count} failures "
                    f"(threshold: {self.failure_threshold}). Blocking further calls."
                )
                self.state = CircuitState.OPEN

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state.value

    def force_open(self):
        """Manually open the circuit breaker."""
        self.state = CircuitState.OPEN
        self.last_failure_time = datetime.now()
        logger.warning("⚠️ Circuit breaker manually OPENED")

    def force_close(self):
        """Manually close the circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("✅ Circuit breaker manually CLOSED")

# Global circuit breaker for API calls
api_circuit_breaker = CircuitBreaker(
    failure_threshold=3,  # Open after 3 failures
    recovery_timeout=30,  # Try again after 30 seconds
    expected_exception=Exception
)
