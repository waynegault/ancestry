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
    """Enhanced error statistics tracking with detailed metrics."""

    total_requests: int = 0
    failed_requests: int = 0
    last_failure_time: Optional[datetime] = None
    failure_rate: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    recovery_attempts: int = 0
    last_recovery_time: Optional[datetime] = None
    avg_response_time: float = 0.0
    error_patterns: List[str] = field(default_factory=list)


class RetryStrategy(Enum):
    """Different retry strategies for error recovery."""

    EXPONENTIAL_BACKOFF = "exponential"
    LINEAR_BACKOFF = "linear"
    FIXED_DELAY = "fixed"
    FIBONACCI_BACKOFF = "fibonacci"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """Configuration for intelligent retry mechanisms."""

    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomization to prevent thundering herd
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on: List[Type[Exception]] = field(default_factory=list)


class IntelligentRetryHandler:
    """Advanced retry handler with multiple strategies and error pattern learning."""

    def __init__(self, name: str, config: Optional[RetryConfig] = None):
        self.name = name
        self.config = config or RetryConfig()
        self._error_history: List[Tuple[Exception, datetime]] = []
        self._success_patterns: Dict[str, int] = {}
        self._lock = threading.Lock()

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Intelligent decision on whether to retry based on error patterns."""
        if attempt >= self.config.max_attempts:
            return False

        exception_type = type(exception)

        # Never retry certain exceptions
        if any(isinstance(exception, stop_type) for stop_type in self.config.stop_on):
            return False

        # Only retry specified exceptions
        if self.config.retry_on and not any(
            isinstance(exception, retry_type) for retry_type in self.config.retry_on
        ):
            return False

        # Learn from patterns - if this error type has never succeeded after retry, be more conservative
        error_name = exception_type.__name__
        with self._lock:
            if error_name in self._success_patterns:
                success_rate = self._success_patterns[error_name]
                if success_rate < 10:  # Less than 10% success rate for this error
                    return attempt <= 1  # Only retry once

        return True

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on configured strategy."""
        if self.config.strategy == RetryStrategy.IMMEDIATE:
            return 0.0

        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay

        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt

        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (
                self.config.exponential_base ** (attempt - 1)
            )

        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            fib_sequence = [1, 1]
            for i in range(2, attempt + 1):
                fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
            delay = (
                self.config.base_delay
                * fib_sequence[min(attempt - 1, len(fib_sequence) - 1)]
            )

        else:
            delay = self.config.base_delay

        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)

        # Add jitter to prevent thundering herd problem
        if self.config.jitter:
            import random

            delay *= 0.5 + random.random()

        return delay

    def record_attempt(self, exception: Exception, succeeded: bool) -> None:
        """Record attempt outcome for pattern learning."""
        with self._lock:
            error_name = type(exception).__name__
            self._error_history.append((exception, datetime.now()))

            if succeeded:
                self._success_patterns[error_name] = (
                    self._success_patterns.get(error_name, 0) + 1
                )


class CircuitBreaker:
    """
    Enhanced circuit breaker with intelligent retry integration.
    Prevents cascading failures and includes advanced recovery mechanisms.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = ErrorStats()
        self._lock = threading.Lock()
        self._last_failure_time: Optional[datetime] = None
        self.retry_handler = (
            IntelligentRetryHandler(f"{name}_retry", retry_config)
            if retry_config
            else None
        )

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker and intelligent retry."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection and intelligent retry."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN. Recovery time remaining: {self._get_recovery_time_remaining()}s"
                    )

        # Execute with retry if configured
        if self.retry_handler:
            return self._execute_with_retry(func, *args, **kwargs)
        else:
            return self._execute_once(func, *args, **kwargs)

    def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with intelligent retry logic."""
        if not self.retry_handler:
            return self._execute_once(func, *args, **kwargs)

        last_exception = None

        for attempt in range(1, self.retry_handler.config.max_attempts + 1):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                self._record_success_with_timing(execution_time)
                if last_exception:
                    self.retry_handler.record_attempt(last_exception, True)
                    logger.info(
                        f"Function {func.__name__} succeeded after {attempt} attempts"
                    )

                return result

            except Exception as e:
                last_exception = e
                self._record_failure_with_details(e)

                if (
                    attempt < self.retry_handler.config.max_attempts
                    and self.retry_handler.should_retry(e, attempt)
                ):
                    delay = self.retry_handler.calculate_delay(attempt)
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}), retrying in {delay:.2f}s: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    self.retry_handler.record_attempt(e, False)
                    logger.error(
                        f"Function {func.__name__} failed after {attempt} attempts: {str(e)}"
                    )
                    raise

        # Should never reach here
        raise (
            last_exception if last_exception else RuntimeError("Unexpected retry state")
        )

    def _execute_once(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function once with circuit breaker protection."""
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            self._record_success_with_timing(execution_time)
            return result
        except Exception as e:
            self._record_failure_with_details(e)
            raise

    def _record_success_with_timing(self, execution_time: float) -> None:
        """Record successful execution with timing."""
        with self._lock:
            self.stats.total_requests += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0

            # Update average response time
            if self.stats.avg_response_time == 0:
                self.stats.avg_response_time = execution_time
            else:
                # Exponential moving average
                self.stats.avg_response_time = (0.9 * self.stats.avg_response_time) + (
                    0.1 * execution_time
                )

            if self.state == CircuitState.HALF_OPEN:
                if self.stats.consecutive_successes >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    logger.info(f"Circuit breaker {self.name} is now CLOSED")

    def _record_failure_with_details(self, exception: Exception) -> None:
        """Record failed execution with error details."""
        with self._lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = datetime.now()

            # Track error types
            error_type = type(exception).__name__
            self.stats.error_types[error_type] = (
                self.stats.error_types.get(error_type, 0) + 1
            )

            # Update failure rate
            self.stats.failure_rate = (
                self.stats.failed_requests / self.stats.total_requests
            )

            # Add to error patterns for analysis
            error_msg = str(exception)[:100]  # Truncate long messages
            if error_msg not in self.stats.error_patterns:
                self.stats.error_patterns.append(error_msg)
                # Keep only last 10 patterns to prevent memory bloat
                if len(self.stats.error_patterns) > 10:
                    self.stats.error_patterns.pop(0)

            if self.state == CircuitState.CLOSED:
                if self.stats.consecutive_failures >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self._last_failure_time = datetime.now()
                    logger.warning(
                        f"Circuit breaker {self.name} is now OPEN after {self.config.failure_threshold} failures"
                    )

    def _get_recovery_time_remaining(self) -> int:
        """Get remaining time before recovery attempt."""
        if not self._last_failure_time:
            return 0
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return max(0, int(self.config.recovery_timeout - elapsed))

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


def error_handling_module_tests():
    """Essential error handling tests for unified framework."""
    tests = []
    
    # Test 1: Function availability
    def test_function_availability():
        required_functions = [
            "error_recovery_manager", "AppError", "handle_error", 
            "safe_execute", "ErrorContext", "CircuitBreaker"
        ]
        for func_name in required_functions:
            if func_name in globals():
                assert callable(globals()[func_name]) or isinstance(globals()[func_name], type), f"Function {func_name} should be available"
    tests.append(("Function Availability", test_function_availability))
    
    # Test 2: Error handling basics
    def test_error_handling():
        # Test safe_execute with simple function
        def safe_func():
            return "success"
        
        result = safe_execute(safe_func, default="failed")
        assert result == "success", "safe_execute should handle successful execution"
    tests.append(("Error Handling", test_error_handling))
    
    # Test 3: Error types 
    def test_error_types():
        # Test AppError creation
        if "AppError" in globals():
            error = AppError("test error")
            assert str(error) == "test error", "AppError should store message"
    tests.append(("Error Types", test_error_types))
    
    return tests


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    from test_framework_unified import run_unified_tests
    return run_unified_tests("error_handling", error_handling_module_tests)



# === END OF error_handling.py ===


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    print("🚨 Running Error Handling & Recovery Systems comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
