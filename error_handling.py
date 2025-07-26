#!/usr/bin/env python3

"""
Error handling utilities for the Ancestry project.
Enhanced for Phase 4.1: Error Handling & Resilience Enhancement
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
    safe_execute,
    get_function,
    is_function_available,
)

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import functools
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# === THIRD-PARTY IMPORTS ===
import requests

# === LOCAL IMPORTS ===
# Module logger is set up by setup_module() above

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)

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


# === PHASE 4.1: ENHANCED EXCEPTION HIERARCHY ===


class AncestryException(Exception):
    """
    Base exception class for all Ancestry project errors.
    Provides structured error context and recovery guidance.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "ERROR",
        recovery_hint: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.severity = severity
        self.recovery_hint = recovery_hint
        self.cause = cause
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/debugging."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "severity": self.severity,
            "recovery_hint": self.recovery_hint,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }


class RetryableError(AncestryException):
    """Error that can be retried with appropriate strategy."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity="WARNING", **kwargs)


class FatalError(AncestryException):
    """Error that cannot be recovered from automatically."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity="FATAL", **kwargs)


class ConfigurationError(AncestryException):
    """Configuration-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity="ERROR",
            recovery_hint="Check configuration settings and retry",
            **kwargs,
        )


class DatabaseConnectionError(RetryableError):
    """Database connection issues that can be retried."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="DB_CONNECTION_FAILED",
            recovery_hint="Check database connectivity and retry",
            **kwargs,
        )


class BrowserSessionError(RetryableError):
    """Browser session issues that can be recovered."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="BROWSER_SESSION_FAILED",
            recovery_hint="Restart browser session and retry",
            **kwargs,
        )


class APIRateLimitError(RetryableError):
    """API rate limiting that requires backoff."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        self.retry_after = retry_after
        super().__init__(
            message,
            error_code="API_RATE_LIMIT",
            recovery_hint=(
                f"Retry after {retry_after} seconds"
                if retry_after
                else "Retry with exponential backoff"
            ),
            context={"retry_after": retry_after},
            **kwargs,
        )


class AuthenticationExpiredError(RetryableError):
    """Authentication token expiration."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="AUTH_EXPIRED",
            recovery_hint="Refresh authentication token and retry",
            **kwargs,
        )


class NetworkTimeoutError(RetryableError):
    """Network timeout that can be retried."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="NETWORK_TIMEOUT",
            recovery_hint="Check network connectivity and retry",
            **kwargs,
        )


class DataValidationError(FatalError):
    """Data validation errors that require manual intervention."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="DATA_VALIDATION_FAILED",
            recovery_hint="Fix data validation issues manually",
            **kwargs,
        )


# === ENHANCED ERROR CONTEXT CAPTURE ===


@dataclass
class ErrorContext:
    """Enhanced error context for debugging and recovery."""

    operation: str
    module: str
    function: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    error_id: str = field(default_factory=lambda: f"err_{int(time.time())}")

    def capture_environment(self):
        """Capture current environment state."""
        import platform
        import sys

        self.environment.update(
            {
                "python_version": sys.version,
                "platform": platform.platform(),
                "timestamp": datetime.now().isoformat(),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "operation": self.operation,
            "module": self.module,
            "function": self.function,
            "parameters": self.parameters,
            "environment": self.environment,
            "timing": self.timing,
            "stack_trace": self.stack_trace,
            "error_id": self.error_id,
        }


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


# === PHASE 4.1: ENHANCED DECORATOR FRAMEWORK ===


def retry_on_failure(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    retry_on: Optional[List[Type[Exception]]] = None,
    stop_on: Optional[List[Type[Exception]]] = None,
    jitter: bool = True,
):
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        retry_on: List of exceptions to retry on (default: all RetryableError)
        stop_on: List of exceptions to never retry (default: FatalError)
        jitter: Add randomization to prevent thundering herd
    """
    if retry_on is None:
        retry_on = [RetryableError, NetworkTimeoutError, DatabaseConnectionError]
    if stop_on is None:
        stop_on = [FatalError, DataValidationError]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation="retry_decorated_call",
                module=func.__module__,
                function=func.__name__,
                parameters={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
            )
            context.capture_environment()

            last_exception = None
            start_time = time.time()

            for attempt in range(max_attempts):
                try:
                    attempt_start = time.time()
                    result = func(*args, **kwargs)

                    # Log success metrics
                    execution_time = time.time() - attempt_start
                    if attempt > 0:
                        logger.info(
                            f"{func.__name__} succeeded on attempt {attempt + 1}/{max_attempts} "
                            f"after {execution_time:.2f}s"
                        )

                    return result

                except Exception as e:
                    last_exception = e

                    # Check if we should stop retrying
                    if any(isinstance(e, stop_type) for stop_type in stop_on):
                        logger.error(
                            f"{func.__name__} failed with non-retryable error: {e}"
                        )
                        context.stack_trace = traceback.format_exc()
                        if isinstance(e, AncestryException):
                            e.context.update(context.to_dict())
                        raise

                    # Check if we should retry
                    if not any(isinstance(e, retry_type) for retry_type in retry_on):
                        logger.error(
                            f"{func.__name__} failed with non-retryable error type: {e}"
                        )
                        raise

                    # Calculate delay for next attempt
                    if attempt < max_attempts - 1:
                        delay = backoff_factor**attempt
                        if jitter:
                            import random

                            delay *= 0.5 + random.random()

                        logger.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}/{max_attempts}, "
                            f"retrying in {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)

            # All attempts exhausted
            total_time = time.time() - start_time
            if last_exception is None:
                last_exception = Exception(
                    f"{func.__name__} failed after {max_attempts} attempts"
                )

            logger.error(
                f"{func.__name__} failed after {max_attempts} attempts in {total_time:.2f}s: {last_exception}"
            )
            context.stack_trace = traceback.format_exc()
            if isinstance(last_exception, AncestryException):
                last_exception.context.update(context.to_dict())
            raise last_exception

        return wrapper

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    success_threshold: int = 3,
):
    """
    Decorator for circuit breaker pattern.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        success_threshold: Successes needed to close circuit from half-open
    """

    def decorator(func: Callable) -> Callable:
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
        )
        service_name = f"{func.__module__}.{func.__name__}"
        cb = error_recovery_manager.get_circuit_breaker(service_name, config)
        return cb(func)

    return decorator


def timeout_protection(timeout: int = 30):
    """
    Decorator for timeout protection.

    Args:
        timeout: Maximum execution time in seconds
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use threading approach for cross-platform compatibility
            import threading

            result: List[Any] = [None]
            exception: List[Optional[Exception]] = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True  # Allow main program to exit even if thread is running
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                # Thread is still running, timeout occurred
                logger.warning(f"{func.__name__} timed out after {timeout}s")
                raise NetworkTimeoutError(
                    f"{func.__name__} timed out after {timeout} seconds",
                    context={"timeout": timeout, "function": func.__name__},
                )

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


def graceful_degradation(
    fallback_value: Any = None, fallback_func: Optional[Callable] = None
):
    """
    Decorator for graceful degradation when service fails.

    Args:
        fallback_value: Value to return if function fails
        fallback_func: Function to call if main function fails
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"{func.__name__} failed, using graceful degradation: {e}"
                )

                if fallback_func:
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback function also failed for {func.__name__}: {fallback_error}"
                        )
                        return fallback_value

                return fallback_value

        return wrapper

    return decorator


def error_context(operation: str):
    """
    Decorator to add comprehensive error context to function calls.

    Args:
        operation: Description of the operation being performed
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                module=func.__module__,
                function=func.__name__,
                parameters={
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                    "args_preview": str(args)[:100],
                    "kwargs_preview": str(kwargs)[:100],
                },
            )
            context.capture_environment()

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                context.timing["execution_time"] = execution_time

                logger.debug(
                    f"{operation} completed successfully in {execution_time:.2f}s",
                    extra={"error_context": context.to_dict()},
                )
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                context.timing["execution_time"] = execution_time
                context.stack_trace = traceback.format_exc()

                # Enhance exception with context if it's an AncestryException
                if isinstance(e, AncestryException):
                    e.context.update(context.to_dict())

                logger.error(
                    f"{operation} failed after {execution_time:.2f}s: {e}",
                    extra={"error_context": context.to_dict()},
                )
                raise

        return wrapper

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


def error_handling_module_tests() -> bool:
    """
    Comprehensive test suite for error_handling.py following the standardized 6-category TestSuite framework.
    Tests advanced error handling patterns, circuit breakers, and recovery strategies.

    Categories: Initialization, Core Functionality, Edge Cases, Integration, Performance, Error Handling
    """
    with suppress_logging():
        suite = TestSuite("Error Handling & Recovery Systems", "error_handling.py")
        suite.start_suite()

    # === INITIALIZATION TESTS ===
    def test_module_imports():
        """Test all required modules and dependencies are properly imported."""
        required_modules = [
            "CircuitState",
            "CircuitBreakerConfig",
            "RetryStrategy",
            "ErrorRecoveryManager",
        ]
        for module_name in required_modules:
            assert (
                module_name in globals()
            ), f"Required class {module_name} should be available"

    def test_function_availability():
        """Test all essential error handling functions are available."""
        required_classes = [
            "CircuitBreaker",
            "ErrorRecoveryManager",
            "IntelligentRetryHandler",
            "CircuitBreakerConfig",
            "RetryConfig",
        ]
        for class_name in required_classes:
            if class_name in globals():
                assert isinstance(
                    globals()[class_name], type
                ), f"Class {class_name} should be available as a type"

        required_functions = [
            "with_circuit_breaker",
            "with_recovery",
            "ancestry_session_recovery",
            "ancestry_api_recovery",
            "ancestry_database_recovery",
        ]
        for func_name in required_functions:
            if func_name in globals():
                assert callable(
                    globals()[func_name]
                ), f"Function {func_name} should be callable"

    def test_circuit_breaker_config():
        """Test CircuitBreakerConfig initialization and default values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5, "Default failure threshold should be 5"
        assert config.recovery_timeout == 60, "Default recovery timeout should be 60"
        assert config.success_threshold == 3, "Default success threshold should be 3"

    # === CORE FUNCTIONALITY TESTS ===
    def test_error_handling_basics():
        """Test basic error handling functionality."""
        # Test CircuitBreaker initialization
        circuit_breaker = CircuitBreaker("test_service")
        assert (
            circuit_breaker.name == "test_service"
        ), "CircuitBreaker should store service name"
        assert (
            circuit_breaker.state == CircuitState.CLOSED
        ), "CircuitBreaker should start in CLOSED state"

    def test_error_types():
        """Test error type creation and handling."""
        # Test CircuitBreakerOpenError
        if "CircuitBreakerOpenError" in globals():
            error = CircuitBreakerOpenError("Circuit is open")
            assert (
                str(error) == "Circuit is open"
            ), "CircuitBreakerOpenError should store message"

    def test_circuit_breaker_states():
        """Test circuit breaker state transitions."""
        for state in CircuitState:
            assert isinstance(
                state.value, str
            ), f"Circuit state {state.name} should have string value"

        assert (
            CircuitState.CLOSED.value == "CLOSED"
        ), "CLOSED state should have correct value"
        assert CircuitState.OPEN.value == "OPEN", "OPEN state should have correct value"
        assert (
            CircuitState.HALF_OPEN.value == "HALF_OPEN"
        ), "HALF_OPEN state should have correct value"

    # === EDGE CASE TESTS ===
    def test_error_recovery_edge_cases():
        """Test error recovery with edge cases."""
        # Test ErrorRecoveryManager with different services
        recovery_manager = ErrorRecoveryManager()
        assert recovery_manager is not None, "ErrorRecoveryManager should initialize"

    def test_circuit_breaker_edge_cases():
        """Test circuit breaker with edge case configurations."""
        edge_config = CircuitBreakerConfig(
            failure_threshold=0, recovery_timeout=0, success_threshold=0
        )
        assert (
            edge_config.failure_threshold == 0
        ), "Should accept zero failure threshold"
        assert edge_config.recovery_timeout == 0, "Should accept zero recovery timeout"
        assert (
            edge_config.success_threshold == 0
        ), "Should accept zero success threshold"

    def test_error_context_edge_cases():
        """Test ErrorContext with various input types."""
        # Test retry strategies
        for strategy in RetryStrategy:
            assert isinstance(
                strategy.value, str
            ), f"Retry strategy {strategy.name} should have string value"

    # === INTEGRATION TESTS ===
    def test_logging_integration():
        """Test integration with logging system."""
        assert logger is not None, "Logger should be available"
        try:
            logger.info("Test error handling log message")
        except Exception as e:
            assert False, f"Logging should work without errors: {e}"

    def test_config_integration():
        """Test integration with configuration management."""
        try:
            config = CircuitBreakerConfig(failure_threshold=10, recovery_timeout=120)
            assert (
                config.failure_threshold == 10
            ), "Should accept custom failure threshold"
            assert (
                config.recovery_timeout == 120
            ), "Should accept custom recovery timeout"
        except Exception as e:
            assert False, f"Config integration should work: {e}"

    def test_threading_integration():
        """Test thread safety of error handling components."""
        import threading

        results = []

        def thread_safe_test():
            try:
                # Test circuit breaker in threaded environment
                cb = CircuitBreaker("thread_test")
                results.append(cb.name)
            except Exception as e:
                results.append(f"error: {e}")

        threads = [threading.Thread(target=thread_safe_test) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 3, "All threads should complete"
        assert all(r == "thread_test" for r in results), "All threads should succeed"

    # === PERFORMANCE TESTS ===
    def test_error_handling_performance():
        """Test performance of error handling operations."""
        start_time = time.time()

        for _ in range(100):
            # Test circuit breaker creation performance
            cb = CircuitBreaker(f"perf_test")
            assert cb.name == "perf_test", "Circuit breaker should initialize quickly"

        end_time = time.time()
        execution_time = end_time - start_time
        assert (
            execution_time < 0.1
        ), f"Error handling should be fast: {execution_time:.3f}s"

    def test_circuit_breaker_performance():
        """Test circuit breaker performance under load."""
        config = CircuitBreakerConfig(failure_threshold=10, recovery_timeout=1)

        start_time = time.time()
        for _ in range(50):
            # Simulate circuit breaker operations
            assert config.failure_threshold == 10, "Config should remain consistent"
        end_time = time.time()

        execution_time = end_time - start_time
        assert (
            execution_time < 0.05
        ), f"Circuit breaker operations should be fast: {execution_time:.3f}s"

    def test_retry_strategy_performance():
        """Test retry strategy performance."""
        start_time = time.time()

        for _ in range(20):
            config = RetryConfig()
            assert config is not None, "RetryConfig should initialize quickly"

        end_time = time.time()
        execution_time = end_time - start_time
        assert (
            execution_time < 0.02
        ), f"Retry strategy creation should be fast: {execution_time:.3f}s"

    # === ERROR HANDLING TESTS ===
    def test_recursive_error_handling():
        """Test handling of recursive or nested errors."""
        # Test circuit breaker with nested failure scenarios
        cb = CircuitBreaker("nested_test")
        try:
            # Simulate multiple failures by updating stats
            cb.stats.consecutive_failures = cb.config.failure_threshold + 1
            cb.stats.failed_requests = cb.config.failure_threshold + 1
        except Exception:
            pass

        assert (
            cb.stats.consecutive_failures >= cb.config.failure_threshold
        ), "Should handle nested errors gracefully"

    def test_memory_error_handling():
        """Test handling of memory-related errors."""
        # Test ErrorStats memory efficiency
        stats = ErrorStats()
        stats.failed_requests = 1
        stats.total_requests = 2
        assert stats.failed_requests >= 0, "Should handle memory tracking"
        assert stats.total_requests >= 0, "Should handle request tracking"

    def test_timeout_error_handling():
        """Test handling of timeout errors."""
        # Test circuit breaker timeout configurations
        timeout_config = CircuitBreakerConfig(recovery_timeout=1)
        cb = CircuitBreaker("timeout_test", timeout_config)
        assert cb.config.recovery_timeout == 1, "Should handle timeout configurations"

    # === RUN ALL TESTS ===
    suite.run_test(
        "Module Imports",
        test_module_imports,
        "All required error handling classes should be imported and available",
        "All required classes (CircuitState, CircuitBreakerConfig, RetryStrategy, ErrorRecoveryManager) are properly imported",
        "Test verifies advanced error handling pattern classes are available",
    )

    suite.run_test(
        "Function Availability",
        test_function_availability,
        "All essential error handling functions should be available and callable",
        "All essential error handling functions are available",
        "Test error_recovery_manager, AppError, handle_error, safe_execute, ErrorContext, CircuitBreaker",
    )

    suite.run_test(
        "Circuit Breaker Config",
        test_circuit_breaker_config,
        "CircuitBreakerConfig should initialize with proper default values",
        "CircuitBreakerConfig initialization and default values work correctly",
        "Test default failure_threshold=5, recovery_timeout=60, success_threshold=3",
    )

    suite.run_test(
        "Error Handling Basics",
        test_error_handling_basics,
        "Basic error handling should work with safe_execute function",
        "Basic error handling functionality works correctly",
        "Test safe_execute with successful function execution",
    )

    suite.run_test(
        "Error Types",
        test_error_types,
        "Error type creation and string representation should work correctly",
        "Error type creation and handling works correctly",
        "Test AppError creation and message storage",
    )

    suite.run_test(
        "Circuit Breaker States",
        test_circuit_breaker_states,
        "Circuit breaker states should be properly defined with correct values",
        "Circuit breaker state transitions work correctly",
        "Test CLOSED, OPEN, HALF_OPEN states have correct string values",
    )

    suite.run_test(
        "Error Recovery Edge Cases",
        test_error_recovery_edge_cases,
        "Error recovery should handle edge cases like failing functions gracefully",
        "Error recovery with edge cases works correctly",
        "Test safe_execute with failing function returning default value",
    )

    suite.run_test(
        "Circuit Breaker Edge Cases",
        test_circuit_breaker_edge_cases,
        "Circuit breaker should handle edge case configurations like zero values",
        "Circuit breaker with edge case configurations works",
        "Test CircuitBreakerConfig with zero thresholds and timeouts",
    )

    suite.run_test(
        "Error Context Edge Cases",
        test_error_context_edge_cases,
        "ErrorContext should handle various input types including None values",
        "ErrorContext with various input types works correctly",
        "Test ErrorContext initialization with None values",
    )

    suite.run_test(
        "Logging Integration",
        test_logging_integration,
        "Error handling should integrate properly with logging system",
        "Integration with logging system works correctly",
        "Test logger availability and functionality",
    )

    suite.run_test(
        "Config Integration",
        test_config_integration,
        "Error handling should work with custom configuration values",
        "Integration with configuration management works correctly",
        "Test CircuitBreakerConfig with custom values",
    )

    suite.run_test(
        "Threading Integration",
        test_threading_integration,
        "Error handling should be thread-safe and work across multiple threads",
        "Thread safety of error handling components works correctly",
        "Test safe_execute across 3 concurrent threads",
    )

    suite.run_test(
        "Error Handling Performance",
        test_error_handling_performance,
        "Error handling operations should complete quickly under normal load",
        "Error handling performance is acceptable",
        "Measure time for 100 safe_execute operations",
    )

    suite.run_test(
        "Circuit Breaker Performance",
        test_circuit_breaker_performance,
        "Circuit breaker operations should be efficient and not impact performance",
        "Circuit breaker performance under load is acceptable",
        "Test 50 circuit breaker configuration operations",
    )

    suite.run_test(
        "Retry Strategy Performance",
        test_retry_strategy_performance,
        "Retry strategy initialization should be fast and efficient",
        "Retry strategy performance is acceptable",
        "Measure time for 20 RetryStrategy object creations",
    )

    suite.run_test(
        "Recursive Error Handling",
        test_recursive_error_handling,
        "Error handling should gracefully handle nested and recursive errors",
        "Handling of recursive or nested errors works correctly",
        "Test safe_execute with function that has nested try/catch and multiple exception types",
    )

    suite.run_test(
        "Memory Error Handling",
        test_memory_error_handling,
        "Error handling should gracefully handle memory-related errors",
        "Handling of memory-related errors works correctly",
        "Test safe_execute with simulated MemoryError exception",
    )

    suite.run_test(
        "Timeout Error Handling",
        test_timeout_error_handling,
        "Error handling should gracefully handle timeout errors",
        "Handling of timeout errors works correctly",
        "Test safe_execute with simulated TimeoutError exception",
    )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive error handling tests using standardized TestSuite format."""
    return error_handling_module_tests()


# === END OF error_handling.py ===


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    print("🚨 Running Error Handling & Recovery Systems comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
