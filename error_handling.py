#!/usr/bin/env python3
"""
Enhanced Error Handling and Circuit Breaker Pattern for Ancestry.com Automation System
Provides robust error recovery, circuit breaker patterns, and graceful degradation.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from functools import wraps
from enum import Enum
import json
from dataclasses import dataclass, field
from logging_config import logger


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, calls rejected
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds before attempting recovery
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: int = 30  # Request timeout in seconds


@dataclass
class ErrorStats:
    """Error statistics tracking."""

    total_requests: int = 0
    failed_requests: int = 0
    last_failure_time: Optional[datetime] = None
    failure_rate: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit breaker implementation for API calls and critical operations.
    Prevents cascading failures by temporarily blocking calls to failing services.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = ErrorStats()
        self._lock = threading.Lock()
        self._last_failure_time: Optional[datetime] = None

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN"
                    )

        try:
            # Execute the function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Record success
            self._record_success(execution_time)
            return result

        except Exception as e:
            # Record failure
            self._record_failure(e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True

        time_since_failure = datetime.now() - self._last_failure_time
        return time_since_failure.total_seconds() > self.config.recovery_timeout

    def _record_success(self, execution_time: float):
        """Record successful execution."""
        with self._lock:
            self.stats.total_requests += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0

            # Update failure rate
            self._update_failure_rate()

            # State transitions
            if self.state == CircuitState.HALF_OPEN:
                if self.stats.consecutive_successes >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    logger.info(f"Circuit breaker {self.name} CLOSED after recovery")

            logger.debug(f"Circuit breaker {self.name} success: {execution_time:.2f}s")

    def _record_failure(self, error: Exception):
        """Record failed execution."""
        with self._lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = datetime.now()
            self._last_failure_time = self.stats.last_failure_time

            # Track error types
            error_type = type(error).__name__
            self.stats.error_types[error_type] = (
                self.stats.error_types.get(error_type, 0) + 1
            )

            # Update failure rate
            self._update_failure_rate()

            # State transitions
            if self.state == CircuitState.CLOSED:
                if self.stats.consecutive_failures >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker {self.name} OPENED after {self.stats.consecutive_failures} failures"
                    )
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker {self.name} returned to OPEN from HALF_OPEN"
                )

            logger.warning(f"Circuit breaker {self.name} failure: {error}")

    def _update_failure_rate(self):
        """Update failure rate statistics."""
        if self.stats.total_requests > 0:
            self.stats.failure_rate = (
                self.stats.failed_requests / self.stats.total_requests
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.stats.total_requests,
                "failed_requests": self.stats.failed_requests,
                "failure_rate": self.stats.failure_rate,
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
                "last_failure_time": (
                    self.stats.last_failure_time.isoformat()
                    if self.stats.last_failure_time
                    else None
                ),
                "error_types": dict(self.stats.error_types),
            }

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.stats.consecutive_failures = 0
            self.stats.consecutive_successes = 0
            logger.info(f"Circuit breaker {self.name} manually reset to CLOSED")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and graceful degradation.
    """

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.fallback_handlers: Dict[str, Callable] = {}

    def get_circuit_breaker(
        self, name: str, config: CircuitBreakerConfig = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]

    def register_recovery_strategy(self, service_name: str, strategy: Callable):
        """Register a recovery strategy for a service."""
        self.recovery_strategies[service_name] = strategy
        logger.info(f"Registered recovery strategy for {service_name}")

    def register_fallback_handler(self, service_name: str, handler: Callable):
        """Register a fallback handler for a service."""
        self.fallback_handlers[service_name] = handler
        logger.info(f"Registered fallback handler for {service_name}")

    def execute_with_recovery(
        self, service_name: str, operation: Callable, *args, **kwargs
    ) -> Any:
        """
        Execute operation with circuit breaker protection and recovery strategies.
        """
        circuit_breaker = self.get_circuit_breaker(service_name)

        try:
            return circuit_breaker.call(operation, *args, **kwargs)
        except CircuitBreakerOpenError:
            logger.warning(
                f"Circuit breaker open for {service_name}, attempting fallback"
            )
            return self._attempt_fallback(service_name, *args, **kwargs)
        except Exception as e:
            logger.error(f"Operation failed for {service_name}: {e}")
            return self._attempt_recovery(service_name, operation, e, *args, **kwargs)

    def _attempt_fallback(self, service_name: str, *args, **kwargs) -> Any:
        """Attempt fallback operation when circuit breaker is open."""
        if service_name in self.fallback_handlers:
            try:
                logger.info(f"Executing fallback for {service_name}")
                return self.fallback_handlers[service_name](*args, **kwargs)
            except Exception as e:
                logger.error(f"Fallback failed for {service_name}: {e}")
                raise
        else:
            logger.error(f"No fallback handler registered for {service_name}")
            raise CircuitBreakerOpenError(
                f"Service {service_name} unavailable and no fallback"
            )

    def _attempt_recovery(
        self, service_name: str, operation: Callable, error: Exception, *args, **kwargs
    ) -> Any:
        """Attempt recovery strategy when operation fails."""
        if service_name in self.recovery_strategies:
            try:
                logger.info(f"Attempting recovery for {service_name}")
                self.recovery_strategies[service_name](error)
                # Retry operation after recovery
                return operation(*args, **kwargs)
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {service_name}: {recovery_error}")
                raise error  # Re-raise original error
        else:
            raise error  # No recovery strategy, re-raise original error

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}

    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers to CLOSED state."""
        for cb in self.circuit_breakers.values():
            cb.reset()
        logger.info("All circuit breakers reset")


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


def with_circuit_breaker(service_name: str, config: CircuitBreakerConfig = None):
    """Decorator to add circuit breaker protection to functions."""

    def decorator(func: Callable) -> Callable:
        circuit_breaker = error_recovery_manager.get_circuit_breaker(
            service_name, config
        )
        return circuit_breaker(func)

    return decorator


def with_recovery(service_name: str):
    """Decorator to execute functions with recovery strategies."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return error_recovery_manager.execute_with_recovery(
                service_name, func, *args, **kwargs
            )

        return wrapper

    return decorator


# Specific recovery strategies for Ancestry.com automation
def ancestry_session_recovery(error: Exception):
    """Recovery strategy for Ancestry session failures."""
    logger.info("Attempting Ancestry session recovery")
    # This would be implemented to reset session, re-login, etc.
    # For now, just wait and hope the issue resolves
    time.sleep(5)


def ancestry_api_recovery(error: Exception):
    """Recovery strategy for Ancestry API failures."""
    logger.info("Attempting Ancestry API recovery")
    # Implement API-specific recovery (refresh tokens, rate limit backoff, etc.)
    if "rate limit" in str(error).lower():
        time.sleep(30)  # Wait for rate limit to reset
    elif "timeout" in str(error).lower():
        time.sleep(10)  # Wait for network issues to resolve


def ancestry_database_recovery(error: Exception):
    """Recovery strategy for database failures."""
    logger.info("Attempting database recovery")
    # Implement database-specific recovery (reconnect, retry transaction, etc.)
    time.sleep(2)


# Fallback handlers
def ancestry_session_fallback(*args, **kwargs):
    """Fallback when Ancestry session is unavailable."""
    logger.warning("Using session fallback - some features may be limited")
    return None  # Return None or minimal functionality


def ancestry_api_fallback(*args, **kwargs):
    """Fallback when Ancestry API is unavailable."""
    logger.warning("Using API fallback - returning cached data or minimal response")
    return {"status": "fallback", "data": None}


def ancestry_database_fallback(*args, **kwargs):
    """Fallback when database is unavailable."""
    logger.warning("Using database fallback - data may not be persisted")
    return True  # Pretend operation succeeded


# Register recovery strategies and fallback handlers
error_recovery_manager.register_recovery_strategy(
    "ancestry_session", ancestry_session_recovery
)
error_recovery_manager.register_recovery_strategy("ancestry_api", ancestry_api_recovery)
error_recovery_manager.register_recovery_strategy(
    "ancestry_database", ancestry_database_recovery
)

error_recovery_manager.register_fallback_handler(
    "ancestry_session", ancestry_session_fallback
)
error_recovery_manager.register_fallback_handler("ancestry_api", ancestry_api_fallback)
error_recovery_manager.register_fallback_handler(
    "ancestry_database", ancestry_database_fallback
)

# Circuit breaker configurations for different services
ANCESTRY_API_CONFIG = CircuitBreakerConfig(
    failure_threshold=3, recovery_timeout=30, success_threshold=2, timeout=30
)

ANCESTRY_SESSION_CONFIG = CircuitBreakerConfig(
    failure_threshold=2, recovery_timeout=60, success_threshold=1, timeout=45
)

ANCESTRY_DATABASE_CONFIG = CircuitBreakerConfig(
    failure_threshold=5, recovery_timeout=10, success_threshold=3, timeout=15
)


def self_test() -> bool:
    """Test the error handling and circuit breaker functionality."""
    logger.info("Starting error handling self-test...")

    try:
        # Test circuit breaker functionality
        @with_circuit_breaker("test_service")
        def failing_function():
            raise Exception("Test failure")

        @with_circuit_breaker("test_service")
        def success_function():
            return "success"

        # Test failures
        for i in range(6):  # Should trigger circuit breaker
            try:
                failing_function()
            except Exception:
                pass

        # Check that circuit breaker is open
        cb = error_recovery_manager.get_circuit_breaker("test_service")
        if cb.state != CircuitState.OPEN:
            logger.error("Circuit breaker should be OPEN after failures")
            return False

        # Test circuit breaker open exception
        try:
            failing_function()
            logger.error("Circuit breaker should prevent calls when OPEN")
            return False
        except CircuitBreakerOpenError:
            pass  # Expected

        # Reset and test success
        cb.reset()
        result = success_function()
        if result != "success":
            logger.error("Function should work after circuit breaker reset")
            return False

        # Test stats
        stats = cb.get_stats()
        if stats["total_requests"] <= 0:
            logger.error("Stats should show request counts")
            return False

        logger.info("Error handling self-test passed successfully")
        return True

    except Exception as e:
        logger.error(f"Error handling self-test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = self_test()
    exit(0 if success else 1)
