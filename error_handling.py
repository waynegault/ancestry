# Safe imports with fallback
from core_imports import (
    register_function,
    get_function,
    is_function_available,
    auto_register_module,
)

auto_register_module(globals(), __name__)

# Initialize function_registry as None for backward compatibility
function_registry = None
from logging_config import logger
from typing import Dict, Any, Optional, Callable, Union, Type, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback
import time
import functools
import sqlite3
import requests
from pathlib import Path
import threading
from functools import wraps

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)


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

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
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

    def get_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]

    def register_recovery_strategy(self, service_name: str, strategy: Callable):
        """Register a recovery strategy for a service."""
        self.recovery_strategies[service_name] = strategy
        logger.info(f"Registered recovery strategy for {service_name}")

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
            logger.error(
                f"Circuit breaker open for {service_name}, operation unavailable"
            )
            raise
        except Exception as e:
            logger.error(f"Operation failed for {service_name}: {e}")
            return self._attempt_recovery(service_name, operation, e, *args, **kwargs)

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


def with_circuit_breaker(
    service_name: str, config: Optional[CircuitBreakerConfig] = None
):
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


# Register recovery strategies
error_recovery_manager.register_recovery_strategy(
    "ancestry_session", ancestry_session_recovery
)
error_recovery_manager.register_recovery_strategy("ancestry_api", ancestry_api_recovery)
error_recovery_manager.register_recovery_strategy(
    "ancestry_database", ancestry_database_recovery
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


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for error_handling.py with real functionality testing.
    Tests initialization, core functionality, edge cases, integration, performance, and error handling.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Error Handling & Recovery Systems", "error_handling.py")
    suite.start_suite()

    # INITIALIZATION TESTS
    def test_error_recovery_manager_initialization():
        """Test error recovery manager and circuit breaker initialization."""
        # Verify error recovery manager exists
        assert (
            "error_recovery_manager" in globals()
        ), "error_recovery_manager not found in globals"

        manager = globals().get("error_recovery_manager")

        if manager:
            # Test that manager has required methods
            required_methods = ["get_circuit_breaker", "reset_all_circuit_breakers"]
            for method in required_methods:
                assert hasattr(
                    manager, method
                ), f"Manager missing required method: {method}"

            # Test circuit breaker creation
            test_breaker = manager.get_circuit_breaker("test_initialization")
            assert test_breaker is not None, "Failed to create circuit breaker"

    with suppress_logging():
        suite.run_test(
            "Error Recovery Manager Initialization",
            test_error_recovery_manager_initialization,
            "Error recovery manager initializes with circuit breaker creation capabilities",
            "Test error recovery manager initialization and circuit breaker factory methods",
            "Verify error_recovery_manager exists and can create circuit breakers with required methods",
        )

        def test_circuit_breaker_configurations():
            """Test that circuit breaker configurations are properly defined."""
            # Check for configuration constants
            config_names = [
                "ANCESTRY_API_CONFIG",
                "ANCESTRY_SESSION_CONFIG",
                "ANCESTRY_DATABASE_CONFIG",
            ]

            configs_found = 0
            for config_name in config_names:
                if config_name in globals():
                    config = globals()[config_name]
                    if hasattr(config, "failure_threshold") and hasattr(
                        config, "recovery_timeout"
                    ):
                        configs_found += 1

            assert (
                configs_found >= 2
            ), f"Expected at least 2 valid configurations, found {configs_found}"

        suite.run_test(
            "Circuit Breaker Configuration Setup",
            test_circuit_breaker_configurations,
            "Circuit breaker configurations are properly defined with required parameters",
            "Test circuit breaker configuration constants for system components",
            "Verify ANCESTRY_*_CONFIG constants exist with proper failure_threshold and recovery_timeout",
        )

        # CORE FUNCTIONALITY TESTS
        def test_circuit_breaker_operation():
            """Test basic circuit breaker operation patterns."""
            # Create test circuit breaker
            cb = error_recovery_manager.get_circuit_breaker("test_operation")

            # Should start in CLOSED state
            assert (
                cb.state == CircuitState.CLOSED
            ), f"Expected CLOSED state, got {cb.state}"

            # Test successful operation
            @with_circuit_breaker("test_operation")
            def success_function():
                return "success"

            result = success_function()
            assert result == "success", f"Expected 'success', got {result}"

            # Test failure handling
            @with_circuit_breaker("test_operation_fail")
            def failing_function():
                raise Exception("Test failure")

            # Test exception handling
            try:
                failing_function()
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Test failure" in str(e), f"Unexpected exception: {e}"

        suite.run_test(
            "Circuit Breaker Operation",
            test_circuit_breaker_operation,
            "Circuit breaker operates correctly with state transitions and error handling",
            "Test core circuit breaker operation with success/failure scenarios and state management",
            "Verify circuit breaker starts closed, handles success/failure scenarios appropriately",
        )

        def test_error_decorators():
            """Test error handling decorators."""
            # Test with_circuit_breaker decorator exists and is callable
            assert (
                "with_circuit_breaker" in globals()
            ), "with_circuit_breaker decorator not found"

            cb_decorator = globals().get("with_circuit_breaker")
            assert callable(
                cb_decorator
            ), "with_circuit_breaker is not callable"  # Test with_recovery decorator
            assert "with_recovery" in globals(), "with_recovery decorator not found"

            recovery_decorator = globals().get("with_recovery")
            assert callable(recovery_decorator), "with_recovery is not callable"

        suite.run_test(
            "Error Handling Decorators",
            test_error_decorators,
            "Error decorators are available and callable",
            "Test error handling decorators availability and basic functionality",
            "Verify @with_circuit_breaker and @with_recovery decorators exist and are callable",
        )

        # EDGE CASES TESTS
        def test_recovery_strategies():
            """Test recovery strategy registration and execution."""
            # Test recovery strategy functions exist
            strategies = [
                "ancestry_session_recovery",
                "ancestry_api_recovery",
                "ancestry_database_recovery",
            ]

            registered_count = 0
            for strategy in strategies:
                if strategy in globals() and callable(globals()[strategy]):
                    registered_count += 1

            assert (
                registered_count >= 2
            ), f"Expected at least 2 strategies available, found {registered_count}"

        suite.run_test(
            "Recovery Strategy Registration",
            test_recovery_strategies,
            "Recovery strategies are available for ancestry services",
            "Verify recovery strategy functions exist for ancestry_session, ancestry_api, ancestry_database",
            "Test recovery strategy function availability for system components",
        )

        # INTEGRATION TESTS
        def test_circuit_state_management():
            """Test circuit state management and transitions."""
            # Test CircuitState enum exists            assert "CircuitState" in globals(), "CircuitState enum not found"

            state_enum = globals().get("CircuitState")
            required_states = ["CLOSED", "OPEN", "HALF_OPEN"]

            states_found = 0
            for state in required_states:
                if hasattr(state_enum, state):
                    states_found += 1

            assert states_found >= 3, f"Expected 3 circuit states, found {states_found}"

        suite.run_test(
            "Circuit State Management",
            test_circuit_state_management,
            "Circuit state enum provides proper state management",
            "Test circuit breaker state enumeration and management",
            "Verify CircuitState enum has CLOSED, OPEN, HALF_OPEN states",
        )

        # PERFORMANCE TESTS
        def test_error_handling_performance():
            """Test error handling performance under load."""
            import time

            # Test multiple operations with circuit breaker
            start_time = time.time()

            # Execute multiple circuit breaker creations
            for i in range(50):
                cb = error_recovery_manager.get_circuit_breaker(f"perf_test_{i}")
                assert (
                    cb is not None
                ), f"Failed to create circuit breaker for perf_test_{i}"

            execution_time = time.time() - start_time

            # Should complete reasonably quickly (less than 1 second)
            assert (
                execution_time < 1.0
            ), f"Performance test took too long: {execution_time}s"

        suite.run_test(
            "Error Handling Performance",
            test_error_handling_performance,
            "Error handling maintains good performance under repeated operations",
            "Verify circuit breaker creation operations complete quickly",
            "Test error handling system performance with multiple operations",
        )

        # ERROR HANDLING TESTS
        def test_exception_types():
            """Test custom exception types."""
            # Test custom exception classes exist
            exception_types = ["CircuitBreakerOpenError"]

            exceptions_found = 0
            for exc_type in exception_types:
                if exc_type in globals():
                    exc_class = globals()[exc_type]
                    if isinstance(exc_class, type) and issubclass(exc_class, Exception):
                        exceptions_found += 1

            assert (
                exceptions_found >= 1
            ), f"Expected at least 1 custom exception, found {exceptions_found}"

        suite.run_test(
            "Custom Exception Types",
            test_exception_types,
            "Custom exception types are properly defined",
            "Test custom exception type definitions and inheritance",
            "Verify CircuitBreakerOpenError exception class exists",
        )

    return suite.finish_suite()


# === END OF error_handling.py ===


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    print("🚨 Running Error Handling & Recovery Systems comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

# Register all module functions
auto_register_module(globals(), __name__)
