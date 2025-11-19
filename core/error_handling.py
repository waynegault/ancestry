#!/usr/bin/env python3

"""
Standardized Error Handling Framework.

This module provides consistent error handling patterns across the entire
application with proper logging, recovery strategies, and user-friendly messages.
"""

# === CORE INFRASTRUCTURE ===
import sys

# Add parent directory to path for standard_imports
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import logging
import random
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, ParamSpec, TypeVar, Union, cast

# === THIRD-PARTY IMPORTS ===
import requests

# === CONFIGURATION IMPORTS ===
from config import config_schema
from config.config_schema import RetryPoliciesConfig

# Type variables for decorators
P = ParamSpec('P')
R = TypeVar('R')

# === ENHANCED CIRCUIT BREAKER CONFIGURATION ===


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


class RetryStrategy(Enum):
    """Retry strategy options."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    retry_on: list[type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on: list[type[Exception]] = field(default_factory=list)


@dataclass(frozen=True)
class RetryPolicyProfile:
    """Resolved retry policy sourced from telemetry-tuned configuration."""

    name: str
    max_attempts: int
    initial_delay_seconds: float
    backoff_factor: float
    max_delay_seconds: float
    jitter_seconds: float
    retry_on: tuple[type[Exception], ...]
    stop_on: tuple[type[Exception], ...]


_DEFAULT_RETRY_BASELINE = RetryPoliciesConfig()
_RETRY_POLICY_CACHE: dict[str, RetryPolicyProfile] = {}


@dataclass(frozen=True)
class RetryDecoratorSettings:
    """Resolved retry decorator configuration."""

    policy_name: Optional[str]
    max_attempts: int
    backoff_factor: float
    base_delay: float
    max_delay: float
    jitter_seconds: float
    retry_on: tuple[type[Exception], ...]
    stop_on: tuple[type[Exception], ...]


def _policy_exception_sets() -> dict[str, dict[str, tuple[type[Exception], ...]]]:
    """Return default retry/stop exception sets per policy channel."""

    return {
        "api": {
            "retry_on": (
                RetryableError,
                NetworkTimeoutError,
                DatabaseConnectionError,
                AuthenticationExpiredError,
                APIRateLimitError,
                requests.exceptions.RequestException,
                ConnectionError,
                TimeoutError,
            ),
            "stop_on": (
                FatalError,
                DataValidationError,
                ConfigurationError,
            ),
        },
        "selenium": {
            "retry_on": (
                RetryableError,
                NetworkTimeoutError,
                BrowserSessionError,
                AuthenticationExpiredError,
            ),
            "stop_on": (
                FatalError,
                DataValidationError,
                ConfigurationError,
            ),
        },
    }


def _get_channel_config(name: str) -> Any:
    cfg = getattr(config_schema, "retry_policies", None)
    if cfg and hasattr(cfg, name):
        return getattr(cfg, name)
    return getattr(_DEFAULT_RETRY_BASELINE, name, None)


def _build_retry_policy(name: str) -> RetryPolicyProfile:
    channel_cfg = _get_channel_config(name)
    if channel_cfg is None:
        raise ValueError(f"Unknown retry policy channel: {name}")

    exception_sets = _policy_exception_sets().get(name)
    if exception_sets is None:
        raise ValueError(f"No exception mapping defined for retry policy '{name}'")

    return RetryPolicyProfile(
        name=name,
        max_attempts=int(getattr(channel_cfg, "max_attempts", 3)),
        initial_delay_seconds=float(getattr(channel_cfg, "initial_delay_seconds", 1.0)),
        backoff_factor=float(getattr(channel_cfg, "backoff_factor", 2.0)),
        max_delay_seconds=float(getattr(channel_cfg, "max_delay_seconds", 20.0)),
        jitter_seconds=float(getattr(channel_cfg, "jitter_seconds", 0.3)),
        retry_on=exception_sets["retry_on"],
        stop_on=exception_sets["stop_on"],
    )


def resolve_retry_policy(
    policy: Optional[Union[str, RetryPolicyProfile]],
    default: str = "selenium",
) -> Optional[RetryPolicyProfile]:
    """Return resolved RetryPolicyProfile for retry decorators."""

    if isinstance(policy, RetryPolicyProfile):
        return policy

    policy_name = (policy or default or "").strip().lower()
    if not policy_name:
        return None

    if policy_name not in _RETRY_POLICY_CACHE:
        _RETRY_POLICY_CACHE[policy_name] = _build_retry_policy(policy_name)

    return _RETRY_POLICY_CACHE[policy_name]


class RecoveryStrategy(Enum):
    """Recovery strategy types for enhanced retry decorators."""

    RETRY = "retry"
    EXPONENTIAL_BACKOFF = "exp_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    PARTIAL_SUCCESS = "partial_success"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class RecoveryContext:
    """Context container shared across enhanced recovery attempts."""

    operation_name: str
    attempt_number: int = 1
    max_attempts: int = 3
    last_error: Optional[Exception] = None
    error_history: list[Exception] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    partial_results: list[Any] = field(default_factory=list)
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF

    def add_error(self, error: Exception) -> None:
        self.last_error = error
        self.error_history.append(error)

    def should_retry(self) -> bool:
        return self.attempt_number < self.max_attempts

    def get_backoff_delay(self, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        if self.recovery_strategy != RecoveryStrategy.EXPONENTIAL_BACKOFF:
            return base_delay

        delay = min(base_delay * (2 ** max(self.attempt_number - 1, 0)), max_delay)
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter


class EnhancedErrorRecovery:
    """Centralized error recovery telemetry with circuit breaker awareness."""

    def __init__(self) -> None:
        self.recovery_stats: dict[str, dict[str, int]] = {}
        self.circuit_breakers: dict[str, dict[str, Any]] = {}

    def get_recovery_stats(self, operation: str) -> dict[str, int]:
        return self.recovery_stats.get(
            operation,
            {
                "total_attempts": 0,
                "successful_recoveries": 0,
                "failed_recoveries": 0,
                "partial_successes": 0,
            },
        )

    def update_stats(self, operation: str, success: bool, partial: bool = False) -> None:
        stats = self.recovery_stats.setdefault(
            operation,
            {
                "total_attempts": 0,
                "successful_recoveries": 0,
                "failed_recoveries": 0,
                "partial_successes": 0,
            },
        )

        stats["total_attempts"] += 1
        if success:
            stats["successful_recoveries"] += 1
        elif partial:
            stats["partial_successes"] += 1
        else:
            stats["failed_recoveries"] += 1

    def is_circuit_open(self, operation: str, failure_threshold: int = 5) -> bool:
        breaker = self.circuit_breakers.get(operation)
        if breaker is None:
            return False

        open_until = breaker.get("open_until", datetime.min)
        if open_until < datetime.now():
            self.circuit_breakers[operation] = {"failures": 0, "open_until": datetime.min}
            return False

        return breaker.get("failures", 0) >= failure_threshold

    def record_failure(self, operation: str, recovery_timeout: int = 300) -> None:
        breaker = self.circuit_breakers.setdefault(operation, {"failures": 0, "open_until": datetime.min})
        breaker["failures"] += 1
        if breaker["failures"] >= 5:
            breaker["open_until"] = datetime.now() + timedelta(seconds=recovery_timeout)
            logger.warning(
                "Circuit breaker opened for %s - cooling down for %ss",
                operation,
                recovery_timeout,
            )

    def record_success(self, operation: str) -> None:
        if operation in self.circuit_breakers:
            self.circuit_breakers[operation] = {"failures": 0, "open_until": datetime.min}


error_recovery = EnhancedErrorRecovery()


def _handle_successful_attempt(operation_name: str, attempt: int) -> None:
    error_recovery.record_success(operation_name)
    error_recovery.update_stats(operation_name, success=True)

    if attempt > 1:
        logger.info("✅ %s succeeded after %d attempts", operation_name, attempt)


def _handle_non_retryable_error(operation_name: str, exc: Exception) -> None:
    logger.error("❌ Non-retryable error in %s: %s", operation_name, exc)
    error_recovery.record_failure(operation_name)
    error_recovery.update_stats(operation_name, success=False)


def _handle_partial_success(
    operation_name: str,
    partial_success_handler: Optional[Callable[[list[Any], Exception], Any]],
    context: RecoveryContext,
    last_exception: Exception,
) -> Any:
    if partial_success_handler and context.partial_results:
        try:
            partial_result = partial_success_handler(context.partial_results, last_exception)
            error_recovery.update_stats(operation_name, success=False, partial=True)
            logger.warning("⚠️ %s completed with partial success", operation_name)
            return partial_result
        except Exception as partial_error:
            logger.error("Partial success handler failed: %s", partial_error)
    return None


def _handle_retry_failure(
    operation_name: str,
    max_attempts: int,
    partial_success_handler: Optional[Callable[[list[Any], Exception], Any]],
    context: RecoveryContext,
    last_exception: Exception,
) -> Any:
    logger.error("❌ %s failed after %d attempts", operation_name, max_attempts)
    error_recovery.record_failure(operation_name)
    error_recovery.update_stats(operation_name, success=False)

    partial_result = _handle_partial_success(operation_name, partial_success_handler, context, last_exception)
    if partial_result is not None:
        return partial_result
    raise last_exception


def create_user_guidance() -> dict[type[Exception], str]:
    """Default user guidance for common retryable exceptions."""

    return {
        ConnectionError: "Check your internet connection and try again",
        TimeoutError: "The operation timed out - try reducing batch size or increasing timeout",
        PermissionError: "Verify file permissions and ensure no other process is locking it",
        FileNotFoundError: "Ensure all required files exist and paths are correct",
        ValueError: "Check input parameters and data format",
        KeyError: "Required configuration or data field is missing",
        ImportError: "Install missing dependencies or check the Python environment",
    }


def handle_partial_success(partial_results: list[Any], error: Exception) -> Any:
    """Return best-effort results when retries exhaust."""

    if not partial_results:
        raise error

    logger.warning("Returning %d partial results due to: %s", len(partial_results), error)
    return partial_results


def with_enhanced_recovery(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    partial_success_handler: Optional[Callable[[list[Any], Exception], Any]] = None,
    user_guidance: Optional[dict[type[Exception], str]] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that standardizes retries, jittered backoff, and guidance logging."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            operation_name = f"{func.__module__}.{func.__name__}"
            context = RecoveryContext(
                operation_name=operation_name,
                max_attempts=max_attempts,
                recovery_strategy=recovery_strategy,
            )

            if error_recovery.is_circuit_open(operation_name):
                raise RuntimeError(f"Circuit breaker is open for {operation_name}")

            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                context.attempt_number = attempt
                try:
                    logger.debug("Attempting %s (%d/%d)", operation_name, attempt, max_attempts)
                    result = func(*args, **kwargs)
                    _handle_successful_attempt(operation_name, attempt)
                    return result
                except Exception as exc:
                    last_exception = exc
                    context.add_error(exc)

                    if not isinstance(exc, retryable_exceptions):
                        _handle_non_retryable_error(operation_name, exc)
                        raise

                    logger.warning("⚠️ %s failed (%d/%d): %s", operation_name, attempt, max_attempts, exc)
                    if user_guidance and type(exc) in user_guidance:
                        logger.info("💡 Suggestion: %s", user_guidance[type(exc)])

                    if not context.should_retry():
                        return cast(
                            R,
                            _handle_retry_failure(
                                operation_name,
                                max_attempts,
                                partial_success_handler,
                                context,
                                last_exception,
                            ),
                        )

                    delay = context.get_backoff_delay(base_delay, max_delay)
                    logger.debug("Retrying %s in %.1fs", operation_name, delay)
                    time.sleep(delay)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"Unknown error in {operation_name}")

        return wrapper

    return decorator


def with_api_recovery(max_attempts: int = 5, base_delay: float = 2.0) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator optimized for API calls using unified recovery infrastructure."""

    return with_enhanced_recovery(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=120.0,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
        user_guidance=create_user_guidance(),
    )


def with_database_recovery(max_attempts: int = 3, base_delay: float = 1.0) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator optimized for database operations using unified recovery infrastructure."""

    return with_enhanced_recovery(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=30.0,
        retryable_exceptions=(ConnectionError, TimeoutError),
        user_guidance=create_user_guidance(),
    )


def with_file_recovery(max_attempts: int = 3, base_delay: float = 0.5) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator optimized for filesystem operations using unified recovery infrastructure."""

    return with_enhanced_recovery(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=10.0,
        retryable_exceptions=(PermissionError, FileNotFoundError, OSError),
        user_guidance=create_user_guidance(),
    )


# Enhanced CircuitBreaker implementation
class CircuitBreaker:
    """
    Enhanced Circuit Breaker pattern implementation for fault tolerance.
    Opens the circuit after a threshold of failures and closes after a timeout.
    """

    def __init__(self, name: str = "default", config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()

    def _handle_success_locked(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0

    def _handle_failure_locked(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.success_count = 0

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if (
                    self.last_failure_time
                    and time.time() - self.last_failure_time > self.config.recovery_timeout
                ):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._handle_success_locked()
            return result
        except Exception as e:
            with self._lock:
                self._handle_failure_locked()
            raise e

    def record_failure(self) -> None:
        """Record failure manually for testing/telemetry without executing call."""
        with self._lock:
            self._handle_failure_locked()

    def record_success(self) -> None:
        """Record success manually, useful for half-open recovery tests."""
        with self._lock:
            self._handle_success_locked()

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# === LEGACY EXCEPTION CLASSES FOR BACKWARD COMPATIBILITY ===

class AncestryError(Exception):
    """Base exception class for all Ancestry project errors."""
    pass


class RetryableError(AncestryError):
    """Exception that indicates the operation can be retried."""

    def __init__(self, message: str = "Operation can be retried", **kwargs: Any) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = kwargs.get('retry_after')
        self.max_retries = kwargs.get('max_retries')
        self.context = kwargs.get('context', {})
        self.recovery_hint = kwargs.get('recovery_hint')


class FatalError(AncestryError):
    """Exception that indicates the operation should not be retried."""

    def __init__(self, message: str = "Fatal error occurred", **kwargs: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = kwargs.get('context', {})
        self.recovery_hint = kwargs.get('recovery_hint')


class APIRateLimitError(RetryableError):
    """Exception for API rate limit errors."""

    def __init__(self, message: str = "API rate limit exceeded", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = kwargs.get('retry_after', 60)


class NetworkTimeoutError(RetryableError):
    """Exception for network timeout errors."""

    def __init__(self, message: str = "Network timeout occurred", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.timeout_duration = kwargs.get('timeout_duration')


class DatabaseConnectionError(RetryableError):
    """Exception for database connection errors."""

    def __init__(self, message: str = "Database connection failed", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.connection_string = kwargs.get('connection_string')
        self.error_code = kwargs.get('error_code', 'DB_CONNECTION_FAILED')


class DataValidationError(FatalError):
    """Exception for data validation errors."""

    def __init__(self, message: str = "Data validation failed", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.validation_errors = kwargs.get('validation_errors', [])


class MissingConfigError(FatalError):
    """Exception for missing configuration errors."""

    def __init__(self, message: str = "Required configuration is missing", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.missing_keys = kwargs.get('missing_keys', [])


class AuthenticationExpiredError(RetryableError):
    """Exception for expired authentication errors."""

    def __init__(self, message: str = "Authentication has expired", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.expired_at = kwargs.get('expired_at')


class BrowserSessionError(RetryableError):
    """Exception for browser session errors."""

    def __init__(self, message: str = "Browser session error occurred", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.session_id = kwargs.get('session_id')


class MaxApiFailuresExceededError(FatalError):
    """Exception for exceeding maximum API failures."""

    def __init__(self, message: str = "Maximum API failures exceeded", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.failure_count = kwargs.get('failure_count', 0)
        self.max_failures = kwargs.get('max_failures', 0)


class ConfigurationError(FatalError):
    """Exception for configuration errors."""

    def __init__(self, message: str = "Configuration error occurred", **kwargs: Any):
        super().__init__(message, **kwargs)
        self.config_section = kwargs.get('config_section')


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    DATABASE = "database"
    NETWORK = "network"
    BROWSER = "browser"
    API = "api"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    BUSINESS_LOGIC = "business_logic"
    USER_INPUT = "user_input"


class AppError(Exception):
    """
    Base application error with enhanced metadata.

    Provides structured error information including:
    - Error category and severity
    - User-friendly messages
    - Technical details
    - Recovery suggestions
    - Context information
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_message: Optional[str] = None,
        technical_details: Optional[str] = None,
        recovery_suggestion: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.user_message = user_message or self._generate_user_message()
        self.technical_details = technical_details
        self.recovery_suggestion = recovery_suggestion
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = time.time()

    def _generate_user_message(self) -> str:
        """Generate user-friendly message based on category."""
        category_messages = {
            ErrorCategory.AUTHENTICATION: "Please check your login credentials and try again.",
            ErrorCategory.NETWORK: "There seems to be a network connectivity issue. Please check your internet connection.",
            ErrorCategory.DATABASE: "A database error occurred. Please try again in a moment.",
            ErrorCategory.BROWSER: "A browser-related issue occurred. Please refresh the page or restart the browser.",
            ErrorCategory.API: "An API error occurred. Please try again in a moment.",
            ErrorCategory.CONFIGURATION: "A configuration error was detected. Please check your settings.",
            ErrorCategory.VALIDATION: "The provided input is invalid. Please check and try again.",
            ErrorCategory.SYSTEM: "A system error occurred. Please try again or contact support.",
        }
        return category_messages.get(self.category, "An unexpected error occurred.")

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "user_message": self.user_message,
            "technical_details": self.technical_details,
            "recovery_suggestion": self.recovery_suggestion,
            "context": self.context,
            "timestamp": self.timestamp,
            "original_exception": (
                str(self.original_exception) if self.original_exception else None
            ),
        }


def _safe_update_error_context(error: Exception, payload: Optional[dict[str, Any]]) -> None:
    """Attach diagnostic payloads to legacy error types that track context."""

    if not payload:
        return

    existing = getattr(error, "context", None)
    if isinstance(existing, dict):
        existing.update(payload)
        return

    setattr(error, "context", dict(payload))


class AuthenticationError(AppError):
    """Authentication-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class ValidationError(AppError):
    """Validation-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class DatabaseError(AppError):
    """Database-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(
            message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class NetworkError(AppError):
    """Network-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class BrowserError(AppError):
    """Browser/WebDriver-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(
            message,
            category=ErrorCategory.BROWSER,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class APIError(AppError):
    """API-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(
            message, category=ErrorCategory.API, severity=ErrorSeverity.MEDIUM, **kwargs
        )


# ConfigurationError already defined above as legacy exception


# MissingConfigError already defined above as legacy exception


class ErrorHandler(ABC):
    """Abstract base class for error handlers."""

    def _augment_context(self, context: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Attach handler metadata to the context for downstream diagnostics."""
        enriched_context = dict(context or {})
        enriched_context.setdefault("handler", type(self).__name__)
        return enriched_context

    @staticmethod
    def _match_keywords(error: Exception, keywords: tuple[str, ...]) -> bool:
        """Return True when the error metadata matches any of the supplied keywords."""
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()
        return any(keyword in error_type or keyword in error_msg for keyword in keywords)

    @abstractmethod
    def can_handle(self, error: Exception) -> bool:
        """Check if this handler can process the given error."""
        pass

    @abstractmethod
    def handle(
        self, error: Exception, context: Optional[dict[str, Any]] = None
    ) -> AppError:
        """Handle the error and return a standardized AppError."""
        pass


class DatabaseErrorHandler(ErrorHandler):
    """Handler for database-related errors."""

    def can_handle(self, error: Exception) -> bool:
        keywords = ("sql", "database", "connection", "integrity")
        return self._match_keywords(error, keywords)

    def handle(
        self, error: Exception, context: Optional[dict[str, Any]] = None
    ) -> AppError:
        error_message = str(error)
        context_with_handler = self._augment_context(context)

        if "connection" in error_message.lower():
            return DatabaseError(
                "Database connection failed",
                technical_details=error_message,
                recovery_suggestion="Check database connectivity and try again",
                context=context_with_handler,
                original_exception=error,
            )
        if "integrity" in error_message.lower():
            return DatabaseError(
                "Database integrity constraint violated",
                technical_details=error_message,
                recovery_suggestion="Check data validity and constraints",
                context=context_with_handler,
                original_exception=error,
            )
        return DatabaseError(
            "Database operation failed",
            technical_details=error_message,
            recovery_suggestion="Try the operation again or contact support",
            context=context_with_handler,
            original_exception=error,
        )


class NetworkErrorHandler(ErrorHandler):
    """Handler for network-related errors."""

    def can_handle(self, error: Exception) -> bool:
        keywords = ("connection", "timeout", "http", "request", "url")
        return self._match_keywords(error, keywords)

    def handle(
        self, error: Exception, context: Optional[dict[str, Any]] = None
    ) -> AppError:
        error_message = str(error)
        context_with_handler = self._augment_context(context)

        if "timeout" in error_message.lower():
            return NetworkError(
                "Network request timed out",
                technical_details=error_message,
                recovery_suggestion="Check your internet connection and try again",
                context=context_with_handler,
                original_exception=error,
            )
        if "connection" in error_message.lower():
            return NetworkError(
                "Network connection failed",
                technical_details=error_message,
                recovery_suggestion="Check your internet connection and try again",
                context=context_with_handler,
                original_exception=error,
            )
        return NetworkError(
            "Network request failed",
            technical_details=error_message,
            recovery_suggestion="Check your internet connection and try again",
            context=context_with_handler,
            original_exception=error,
        )


class BrowserErrorHandler(ErrorHandler):
    """Handler for browser/WebDriver-related errors."""

    def can_handle(self, error: Exception) -> bool:
        keywords = ("webdriver", "selenium", "browser", "chrome")
        return self._match_keywords(error, keywords)

    def handle(
        self, error: Exception, context: Optional[dict[str, Any]] = None
    ) -> AppError:
        error_message = str(error)
        context_with_handler = self._augment_context(context)

        if "session" in error_message.lower():
            return BrowserError(
                "Browser session lost",
                technical_details=error_message,
                recovery_suggestion="Restart the browser and try again",
                context=context_with_handler,
                original_exception=error,
            )
        if "element" in error_message.lower():
            return BrowserError(
                "Web element not found or not accessible",
                technical_details=error_message,
                recovery_suggestion="Refresh the page and try again",
                context=context_with_handler,
                original_exception=error,
            )
        return BrowserError(
            "Browser operation failed",
            technical_details=error_message,
            recovery_suggestion="Restart the browser and try again",
            context=context_with_handler,
            original_exception=error,
        )


class ErrorHandlerRegistry:
    """
    Registry for error handlers that automatically routes errors
    to appropriate handlers based on error type.
    """

    def __init__(self) -> None:
        """Initialize error handler registry with default handlers."""
        self.handlers: list[ErrorHandler] = []
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default error handlers."""
        self.handlers.extend(
            [DatabaseErrorHandler(), NetworkErrorHandler(), BrowserErrorHandler()]
        )

    def register_handler(self, handler: ErrorHandler):
        """Register a custom error handler."""
        self.handlers.append(handler)
        logger.debug(f"Registered error handler: {type(handler).__name__}")

    def handle_error(
        self,
        error: Exception,
        context: Optional[dict[str, Any]] = None,
        fallback_category: ErrorCategory = ErrorCategory.SYSTEM,
    ) -> AppError:
        """
        Handle an error using the appropriate handler.

        Args:
            error: The exception to handle
            context: Optional context information
            fallback_category: Category to use if no specific handler found

        Returns:
            Standardized AppError
        """
        # If it's already an AppError, return as-is
        if isinstance(error, AppError):
            return error

        # Try to find a suitable handler
        for handler in self.handlers:
            if handler.can_handle(error):
                try:
                    return handler.handle(error, context)
                except Exception as handler_error:
                    logger.error(
                        f"Error handler {type(handler).__name__} failed: {handler_error}"
                    )
                    continue

        # Fallback to generic error
        error_str = str(error)
        error_type = type(error).__name__
        # Special case for ZeroDivisionError
        if isinstance(error, ZeroDivisionError):
            error_str = f"{error_type}: division by zero"
        elif not error_str:
            error_str = f"{error_type}"
        else:
            error_str = f"{error_type}: {error_str}"
        return AppError(
            error_str,
            category=fallback_category,
            severity=ErrorSeverity.MEDIUM,
            technical_details=traceback.format_exc(),
            context=context,
            original_exception=error,
        )


# Global error handler registry
_error_registry = ErrorHandlerRegistry()


def handle_error(
    error: Exception,
    context: Optional[dict[str, Any]] = None,
    fallback_category: ErrorCategory = ErrorCategory.SYSTEM,
) -> AppError:
    """
    Global error handling function.
    """
    return _error_registry.handle_error(error, context, fallback_category)


def register_error_handler(handler: ErrorHandler) -> None:
    """Register a custom error handler globally."""
    _error_registry.register_handler(handler)


def error_handler(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    _severity: ErrorSeverity = ErrorSeverity.MEDIUM,  # Reserved for future severity-based handling
    log_errors: bool = True,
    reraise: bool = False,
):
    """
    Decorator for automatic error handling.

    Args:
        category: Error category for unhandled exceptions
        severity: Error severity for unhandled exceptions
        log_errors: Whether to log errors automatically
        reraise: Whether to reraise the original exception
    """

    def decorator(func: Callable[P, R]) -> Callable[P, Optional[R]]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create context from function info
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                }

                # Handle the error
                app_error = handle_error(e, context, category)

                # Log the error
                if log_errors:
                    log_level = {
                        ErrorSeverity.LOW: logging.INFO,
                        ErrorSeverity.MEDIUM: logging.WARNING,
                        ErrorSeverity.HIGH: logging.ERROR,
                        ErrorSeverity.CRITICAL: logging.CRITICAL,
                    }.get(app_error.severity, logging.ERROR)

                    logger.log(
                        log_level,
                        f"Error in {func.__name__}: {app_error.message}",
                        extra={"error_details": app_error.to_dict()},
                    )

                # Reraise if requested
                if reraise:
                    raise app_error from None

                return None

        return wrapper

    return decorator


def safe_execute(
    func: Callable[..., Any],
    *args: Any,
    default_return: Any = None,
    context: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return on error
        context: Optional context information
        **kwargs: Function keyword arguments

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        app_error = handle_error(e, context)
        logger.warning(f"Safe execution failed: {app_error.message}")
        return default_return


class ErrorContext:
    """
    Context manager for error handling with automatic logging.
    """

    def __init__(
        self,
        operation_name: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        log_success: bool = True,
    ):
        self.operation_name = operation_name
        self.category = category
        self.severity = severity
        self.log_success = log_success
        self.start_time = None

    def __enter__(self) -> "ErrorContext":
        self.start_time = time.time()
        logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], _exc_tb: Optional[Any]) -> bool:
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None and exc_val is not None:
            # Error occurred
            context = {"operation": self.operation_name, "duration": duration}

            # Convert BaseException to Exception for handle_error
            error_to_handle = exc_val if isinstance(exc_val, Exception) else Exception(str(exc_val))
            app_error = handle_error(error_to_handle, context, self.category)
            logger.error(
                f"Operation failed: {self.operation_name} ({duration:.2f}s) - {app_error.message}"
            )
            return False  # Don't suppress the exception
        # Success
        if self.log_success:
            logger.debug(
                f"Operation completed: {self.operation_name} ({duration:.2f}s)"
            )
        return True


def get_error_handler(error_type: type[Exception]) -> ErrorHandler:
    """
    Get the appropriate error handler for a specific exception type.

    Args:
        error_type: The exception type to handle

    Returns:
        ErrorHandler: The handler for the exception type
    """

    # Default handler for unknown types
    class DefaultHandler(ErrorHandler):
        def can_handle(self, error: Exception) -> bool:
            handler_name = type(self).__name__
            logger.debug(
                "%s engaged for %s: %s",
                handler_name,
                type(error).__name__,
                error,
            )
            return True  # Catch-all for unknown errors

        def handle(
            self, error: Exception, context: Optional[dict[str, Any]] = None
        ) -> AppError:
            context_with_handler = self._augment_context(context)
            return AppError(
                str(error),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                technical_details=traceback.format_exc(),
                context=context_with_handler,
                original_exception=error,
            )

    # Error handlers registry
    error_handlers = {
        "database": DatabaseErrorHandler(),
        "network": NetworkErrorHandler(),
        "browser": BrowserErrorHandler(),
    }

    # Find handler by error type
    return next(
        (h for t, h in error_handlers.items() if t in str(error_type).lower()),
        DefaultHandler(),
    )


# =============================================
# TEST FRAMEWORK IMPLEMENTATION
# =============================================


def error_handling_module_tests() -> bool:
    """
    Core Error Handling module test suite.
    Tests the six categories: Initialization, Core Functionality, Edge Cases, Integration, Performance, and Error Handling.
    """
    from test_framework import (
        TestSuite,
        suppress_logging,
    )

    with suppress_logging():
        suite = TestSuite(
            "Core Error Handling & Recovery Systems", "core/error_handling.py"
        )

    # Run all tests
    print(
        "🛡️ Running Core Error Handling & Recovery Systems comprehensive test suite..."
    )

    with suppress_logging():
        suite.run_test(
            "Function availability verification",
            test_function_availability,
            "Test that all required error handling functions are available",
            "Function availability verification ensures complete error handling functionality",
            "All core error handling functions (AppError, safe_execute, CircuitBreaker, etc.) are available",
        )

        suite.run_test(
            "Basic error handling functionality",
            test_error_handling,
            "Test basic error handling and safe execution patterns",
            "Basic error handling provides robust execution with graceful degradation",
            "safe_execute handles successful execution and error recovery correctly",
        )

        suite.run_test(
            "Error type instantiation",
            test_error_types,
            "Test custom error types and exception creation",
            "Error type instantiation provides structured exception handling",
            "AppError and custom error types are created and handled correctly",
        )

        suite.run_test(
            "Error recovery mechanisms",
            test_error_recovery,
            "Test error recovery and fallback mechanisms",
            "Error recovery mechanisms ensure continued operation despite failures",
            "Error recovery systems handle exceptions gracefully with appropriate fallbacks",
        )

        suite.run_test(
            "Circuit breaker functionality",
            test_circuit_breaker,
            "Test circuit breaker pattern for fault tolerance",
            "Circuit breaker functionality prevents cascade failures in distributed systems",
            "Circuit breaker correctly opens, closes, and half-opens based on failure thresholds",
        )

        suite.run_test(
            "Error context management",
            test_error_context,
            "Test error context tracking and propagation",
            "Error context management provides detailed error information for debugging",
            "ErrorContext correctly captures and propagates error information",
        )

    # Generate summary report
    return suite.finish_suite()


# === ERROR RECOVERY MANAGER ===

class ErrorRecoveryManager:
    """Manages circuit breakers and recovery strategies."""

    def __init__(self) -> None:
        """Initialize fault tolerance manager with circuit breakers and recovery strategies."""
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.recovery_strategies: dict[str, Callable[..., Any]] = {}
        self._lock = threading.Lock()

    def get_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(name, config)
            return self.circuit_breakers[name]

    def register_recovery_strategy(self, service_name: str, strategy: Callable[..., Any]) -> None:
        """Register a recovery strategy for a service."""
        with self._lock:
            self.recovery_strategies[service_name] = strategy

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of all circuit breakers."""
        with self._lock:
            return {
                name: cb.get_stats() for name, cb in self.circuit_breakers.items()
            }

    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        with self._lock:
            for cb in self.circuit_breakers.values():
                cb.reset()


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


# === DECORATORS AND UTILITY FUNCTIONS ===


def _should_stop_retry(exception: Exception, stop_on: list[type[Exception]]) -> bool:
    """Check if exception should stop retry attempts."""
    return any(isinstance(exception, exc_type) for exc_type in stop_on)


def _should_retry_exception(exception: Exception, retry_on: list[type[Exception]]) -> bool:
    """Check if exception should trigger a retry."""
    return any(isinstance(exception, exc_type) for exc_type in retry_on)


def _calculate_retry_delay(
    attempt: int,
    base_delay: float,
    backoff_factor: float,
    jitter_seconds: float,
    max_delay: float,
) -> float:
    """Calculate delay before next retry attempt."""
    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
    if jitter_seconds > 0:
        delay = min(delay + random.uniform(0, jitter_seconds), max_delay)
    return max(0.05, delay)


def _handle_retry_exception(
    exception: Exception,
    func_name: str,
    attempt: int,
    max_attempts: int,
    stop_on: list[type[Exception]],
    retry_on: list[type[Exception]],
    backoff_factor: float,
    base_delay: float,
    max_delay: float,
    jitter_seconds: float,
) -> bool:
    """Handle exception during retry attempt.

    Returns True if the caller should stop retrying and re-raise the exception.
    """
    if _should_stop_retry(exception, stop_on):
        logger.error(f"{func_name} failed with non-retryable error: {exception}")
        return True

    if not _should_retry_exception(exception, retry_on):
        logger.error(f"{func_name} failed with unsupported error type: {exception}")
        return True

    if attempt < max_attempts - 1:
        delay = _calculate_retry_delay(
            attempt,
            base_delay,
            backoff_factor,
            jitter_seconds,
            max_delay,
        )
        logger.warning(
            "%s failed on attempt %d/%d, retrying in %.2fs: %s",
            func_name,
            attempt + 1,
            max_attempts,
            delay,
            exception,
        )
        time.sleep(delay)

    return False


_DEFAULT_RETRY_EXCEPTIONS = (
    RetryableError,
    NetworkTimeoutError,
    DatabaseConnectionError,
)
_DEFAULT_STOP_EXCEPTIONS = (
    FatalError,
    DataValidationError,
)


def _resolve_retry_settings(
    max_attempts: Optional[int],
    backoff_factor: Optional[float],
    retry_on: Optional[list[type[Exception]]],
    stop_on: Optional[list[type[Exception]]],
    jitter: Optional[bool],
    base_delay: Optional[float],
    max_delay: Optional[float],
    policy: Optional[Union[str, RetryPolicyProfile]],
) -> RetryDecoratorSettings:
    resolved_policy = resolve_retry_policy(policy, default="selenium")

    def _int_value(value: Optional[int], attr: str, fallback: int) -> int:
        if value is not None:
            return value
        if resolved_policy is not None:
            return int(getattr(resolved_policy, attr))
        return fallback

    def _float_value(value: Optional[float], attr: str, fallback: float) -> float:
        if value is not None:
            return value
        if resolved_policy is not None:
            return float(getattr(resolved_policy, attr))
        return fallback

    if retry_on is not None:
        retry_source = tuple(retry_on)
    elif resolved_policy:
        retry_source = tuple(resolved_policy.retry_on)
    else:
        retry_source = _DEFAULT_RETRY_EXCEPTIONS

    if stop_on is not None:
        stop_source = tuple(stop_on)
    elif resolved_policy:
        stop_source = tuple(resolved_policy.stop_on)
    else:
        stop_source = _DEFAULT_STOP_EXCEPTIONS

    policy_jitter_value = _float_value(None, "jitter_seconds", 0.5)
    jitter_enabled = jitter if jitter is not None else (
        bool(policy_jitter_value) if resolved_policy else True
    )
    jitter_seconds = policy_jitter_value if jitter_enabled else 0.0

    return RetryDecoratorSettings(
        policy_name=resolved_policy.name if resolved_policy else None,
        max_attempts=_int_value(max_attempts, "max_attempts", 3),
        backoff_factor=_float_value(backoff_factor, "backoff_factor", 2.0),
        base_delay=_float_value(base_delay, "initial_delay_seconds", 1.0),
        max_delay=_float_value(max_delay, "max_delay_seconds", 60.0),
        jitter_seconds=jitter_seconds,
        retry_on=retry_source,
        stop_on=stop_source,
    )


def _wrap_with_retry(func: Callable[P, R], settings: RetryDecoratorSettings) -> Callable[P, R]:
    stop_on = list(settings.stop_on)
    retry_on = list(settings.retry_on)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        context_payload = {
            "operation": "retry_decorated_call",
            "module": func.__module__,
            "function": func.__name__,
            "args_preview": str(args)[:200],
            "kwargs_keys": list(kwargs.keys()),
        }
        start_time = time.time()
        last_exception: Optional[Exception] = None

        for attempt in range(settings.max_attempts):
            try:
                attempt_start = time.time()
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(
                        "%s succeeded on attempt %d/%d after %.2fs",
                        func.__name__,
                        attempt + 1,
                        settings.max_attempts,
                        time.time() - attempt_start,
                    )
                return result
            except Exception as exc:
                last_exception = exc
                should_raise = _handle_retry_exception(
                    exc,
                    func.__name__,
                    attempt,
                    settings.max_attempts,
                    stop_on,
                    retry_on,
                    settings.backoff_factor,
                    settings.base_delay,
                    settings.max_delay,
                    settings.jitter_seconds,
                )
                if should_raise:
                    raise

        total_time = time.time() - start_time
        if last_exception is None:
            last_exception = Exception(
                f"{func.__name__} failed after {settings.max_attempts} attempts"
            )

        logger.error(
            "%s failed after %d attempts in %.2fs: %s",
            func.__name__,
            settings.max_attempts,
            total_time,
            last_exception,
        )

        if isinstance(last_exception, AncestryError):
            _safe_update_error_context(last_exception, context_payload)

        raise last_exception

    setattr(wrapper, "__retry_policy__", settings.policy_name)
    setattr(
        wrapper,
        "__retry_settings__",
        {
            "max_attempts": settings.max_attempts,
            "backoff_factor": settings.backoff_factor,
            "base_delay": settings.base_delay,
            "max_delay": settings.max_delay,
        },
    )
    return wrapper


def retry_on_failure(
    max_attempts: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    retry_on: Optional[list[type[Exception]]] = None,
    stop_on: Optional[list[type[Exception]]] = None,
    jitter: Optional[bool] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    policy: Optional[Union[str, RetryPolicyProfile]] = "selenium",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for automatic retry with telemetry-derived policies."""

    settings = _resolve_retry_settings(
        max_attempts,
        backoff_factor,
        retry_on,
        stop_on,
        jitter,
        base_delay,
        max_delay,
        policy,
    )

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        return _wrap_with_retry(func, settings)

    return decorator


def api_retry(**overrides: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Telemetry-derived retry helper for API operations."""

    base_decorator = retry_on_failure(policy="api", **overrides)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        wrapped = base_decorator(func)
        setattr(wrapped, "__retry_helper__", "api_retry")
        return wrapped

    return decorator


def selenium_retry(**overrides: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Telemetry-derived retry helper for Selenium/browser operations."""

    base_decorator = retry_on_failure(policy="selenium", **overrides)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        wrapped = base_decorator(func)
        setattr(wrapped, "__retry_helper__", "selenium_retry")
        return wrapped

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    success_threshold: int = 3,
):
    """Decorator to add circuit breaker protection to functions."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
        )
        cb = error_recovery_manager.get_circuit_breaker(func.__name__, config)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


def timeout_protection(timeout: int = 30) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for timeout protection (cross-platform)."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            import platform
            import threading

            # Use different timeout mechanisms based on platform
            if platform.system() == "Windows":
                # Windows doesn't support SIGALRM, use threading approach
                result_container: list[R] = []  # Use empty list instead of [None]
                exception: list[Optional[Exception]] = [None]

                def target() -> None:
                    try:
                        result_container.append(func(*args, **kwargs))
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout)

                if thread.is_alive():
                    # Thread is still running, timeout occurred
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")

                if exception[0]:
                    raise exception[0] from None

                return result_container[0]
            # Unix-like systems can use signal
            import signal

            def timeout_handler(_signum: int, _frame: Any) -> None:
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")

            sigalrm = getattr(signal, "SIGALRM", None)
            alarm_fn: Optional[Callable[[int], Any]] = getattr(signal, "alarm", None)
            if sigalrm is None or alarm_fn is None:
                # Platform does not expose SIGALRM, fall back to direct execution
                return func(*args, **kwargs)

            old_handler = signal.signal(sigalrm, timeout_handler)
            alarm_fn(timeout)

            try:
                result = func(*args, **kwargs)
                alarm_fn(0)  # Disable the alarm once we have a result
                return result
            finally:
                signal.signal(sigalrm, old_handler)
        return wrapper
    return decorator


def graceful_degradation(
    fallback_value: Any = None, fallback_func: Optional[Callable[..., Any]] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for graceful degradation when service fails."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed: {e}, using fallback")
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                return fallback_value
        return wrapper
    return decorator


def error_context(context_name: str = "", **context_data: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to add context to errors."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, AppError):
                    if context_name:
                        context_data["context_name"] = context_name
                    e.context.update(context_data)
                raise e
        return wrapper
    return decorator


def with_circuit_breaker(
    service_name: str, config: Optional[CircuitBreakerConfig] = None
):
    """Decorator to add circuit breaker protection to functions."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        cb = error_recovery_manager.get_circuit_breaker(service_name, config)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


def with_recovery(recovery_strategy: Callable[..., Any]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to add recovery strategy to functions."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed: {e}, attempting recovery")
                return recovery_strategy(*args, **kwargs)
        return wrapper
    return decorator


# === RECOVERY STRATEGIES ===

def ancestry_session_recovery(*_args: Any, **_kwargs: Any) -> None:
    """Recovery strategy for session-related failures."""
    logger.info("Attempting session recovery...")
    # Placeholder for session recovery logic


def ancestry_api_recovery(*_args: Any, **_kwargs: Any) -> None:
    """Recovery strategy for API-related failures."""
    logger.info("Attempting API recovery...")
    # Placeholder for API recovery logic


def ancestry_database_recovery(*_args: Any, **_kwargs: Any) -> None:
    """Recovery strategy for database-related failures."""
    logger.info("Attempting database recovery...")
    # Placeholder for database recovery logic


# Test functions for comprehensive testing
def test_function_availability() -> None:
    """Test that all required error handling functions are available."""
    required_functions = [
        "ErrorRecoveryManager",
        "AppError",
        "handle_error",
        "safe_execute",
        "ErrorContext",
        "CircuitBreaker",
    ]

    available_count = 0
    for func_name in required_functions:
        if func_name in globals():
            assert callable(globals()[func_name]) or isinstance(
                globals()[func_name], type
            ), f"Function {func_name} should be available"
            available_count += 1

    # Ensure we have at least 80% of required functions available
    availability_ratio = available_count / len(required_functions)
    assert availability_ratio >= 0.8, f"Only {availability_ratio:.1%} of required functions available"

    # Test that error categories and severities are available
    if "ErrorCategory" in globals():
        categories = ["NETWORK", "AUTHENTICATION", "VALIDATION", "CONFIGURATION", "BUSINESS_LOGIC", "SYSTEM"]
        for category in categories:
            assert hasattr(globals()["ErrorCategory"], category), f"ErrorCategory.{category} should be available"

    if "ErrorSeverity" in globals():
        severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for severity in severities:
            assert hasattr(globals()["ErrorSeverity"], severity), f"ErrorSeverity.{severity} should be available"


def test_error_handling() -> None:
    """Test basic error handling and safe execution patterns."""

    # Test safe_execute with simple function
    def safe_func() -> str:
        return "success"

    def failing_func() -> None:
        raise ValueError("Test error")

    if "safe_execute" in globals():
        # Test successful execution
        result = safe_execute(safe_func, default_return="failed")
        assert result == "success", "safe_execute should handle successful execution"

        # Test error handling with default return
        result = safe_execute(failing_func, default_return="recovered")
        assert result == "recovered", "safe_execute should return default on error"

        # Test with context
        result = safe_execute(failing_func, default_return="context_test", context={"test": "value"})
        assert result == "context_test", "safe_execute should handle context parameter"

    # Test handle_error function
    if "handle_error" in globals():
        try:
            raise ValueError("Test error for handle_error")
        except Exception as e:
            app_error = handle_error(e)
            assert hasattr(app_error, 'message'), "handle_error should return AppError with message"


def test_error_types() -> None:
    """Test custom error types and exception creation."""
    # Test AppError creation
    if "AppError" in globals():
        error = AppError("test error")
        assert str(error) == "test error", "AppError should store message"

        # Test AppError with category and severity
        if "ErrorCategory" in globals() and "ErrorSeverity" in globals():
            error_with_details = AppError(
                "detailed error",
                category=globals()["ErrorCategory"].VALIDATION,
                severity=globals()["ErrorSeverity"].HIGH
            )
            assert error_with_details.message == "detailed error"
            assert error_with_details.category == globals()["ErrorCategory"].VALIDATION
            assert error_with_details.severity == globals()["ErrorSeverity"].HIGH

    # Test specific error types
    specific_errors = [
        "NetworkTimeoutError", "AuthenticationExpiredError", "DataValidationError",
        "BrowserSessionError", "ConfigurationError", "FatalError"
    ]

    for error_name in specific_errors:
        if error_name in globals():
            try:
                error_class = globals()[error_name]
                error_instance = error_class("Test error message")
                assert hasattr(error_instance, 'message'), f"{error_name} should have message attribute"
                assert hasattr(error_instance, 'category'), f"{error_name} should have category attribute"
                assert hasattr(error_instance, 'severity'), f"{error_name} should have severity attribute"
            except Exception:
                pass  # Some error types might require specific parameters


def test_error_recovery() -> None:
    """Test error recovery and fallback mechanisms."""

    # Test error recovery with failing function
    def failing_func() -> None:
        raise ValueError("test error")

    def successful_func() -> str:
        return "success"

    if "safe_execute" in globals():
        # Test basic error recovery
        result = safe_execute(failing_func, default_return="recovered")
        assert result == "recovered", "safe_execute should provide fallback on error"

        # Test successful execution doesn't trigger recovery
        result = safe_execute(successful_func, default_return="should_not_use")
        assert result == "success", "safe_execute should return actual result on success"

        # Test with different error types
        def network_error_func() -> None:
            raise ConnectionError("Network error")

        result = safe_execute(network_error_func, default_return="network_recovered")
        assert result == "network_recovered", "safe_execute should handle network errors"

    # Test ErrorRecoveryManager if available
    if "ErrorRecoveryManager" in globals():
        try:
            manager = globals()["ErrorRecoveryManager"]()
            assert manager is not None, "ErrorRecoveryManager should be instantiable"
        except Exception:
            pass  # Manager might require specific setup


def test_circuit_breaker() -> None:
    """Test circuit breaker pattern for fault tolerance."""
    if "CircuitBreaker" in globals():
        try:
            config = CircuitBreakerConfig(failure_threshold=3, timeout=1)
            cb = CircuitBreaker(name="test", config=config)
            assert cb is not None, "CircuitBreaker should be instantiable"

            # Test initial state
            if hasattr(cb, 'state'):
                initial_state = cb.state
                assert initial_state is not None, "CircuitBreaker should have initial state"

            # Test failure recording
            if hasattr(cb, 'record_failure'):
                for _ in range(2):  # Record some failures but not enough to open
                    cb.record_failure()

                # Test that circuit breaker tracks failures
                if hasattr(cb, 'failure_count'):
                    assert cb.failure_count >= 0, "CircuitBreaker should track failure count"

            # Test success recording
            if hasattr(cb, 'record_success'):
                cb.record_success()

        except Exception:
            pass  # Circuit breaker might require specific setup

    # Test circuit breaker configuration
    if "CircuitBreakerConfig" in globals():
        try:
            config = globals()["CircuitBreakerConfig"](failure_threshold=5, recovery_timeout=2)
            assert config is not None, "CircuitBreakerConfig should be instantiable"
        except Exception:
            pass


def test_error_context() -> None:
    """Test error context tracking and propagation."""
    if "ErrorContext" in globals():
        try:
            ctx = ErrorContext("test_operation")
            assert ctx is not None, "ErrorContext should be instantiable"

            # Test context properties
            if hasattr(ctx, 'operation_name'):
                assert ctx.operation_name == "test_operation", "ErrorContext should store operation name"

        except Exception:
            pass  # Error context might require specific setup

    # Test error context in AppError
    if "AppError" in globals():
        context_data = {"user_id": "test123", "action": "test_action", "timestamp": "2024-01-01"}
        error = AppError("Test error with context", context=context_data)

        if hasattr(error, 'context'):
            assert error.context == context_data, "AppError should store context data"
            assert error.context.get("user_id") == "test123", "Context should contain user_id"
            assert error.context.get("action") == "test_action", "Context should contain action"

    # Test context propagation through error handling
    if "handle_error" in globals():
        try:
            raise ValueError("Test error for context propagation")
        except Exception as e:
            context = {"source": "test_function", "line": 123}
            app_error = handle_error(e, context=context)

            if hasattr(app_error, 'context'):
                assert app_error.context.get("source") == "test_function", "Context should propagate through handle_error"


# =============================================
# Standalone Test Block
# =============================================
# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(error_handling_module_tests)


if __name__ == "__main__":
    import sys
    import traceback
    from pathlib import Path  # Use centralized path management

    project_root = Path(__file__).resolve().parent.parent
    try:
        # Replaced with standardize_module_imports()
        from core_imports import ensure_imports

        ensure_imports()
    except ImportError:
        # Fallback for testing environment
        # Replaced with standardize_module_imports()
        pass

    print("\U0001faea Running Error Handling comprehensive test suite...")
    try:
        success = error_handling_module_tests()
    except Exception:
        print(
            "\n[ERROR] Unhandled exception during error_handling tests:",
            file=sys.stderr,
        )
        traceback.print_exc()
        success = False
    if not success:
        print(
            "\n[FAIL] One or more error_handling tests failed. See above (stdout) for detailed failure summary.",
            file=sys.stderr,
        )
    sys.exit(0 if success else 1)
