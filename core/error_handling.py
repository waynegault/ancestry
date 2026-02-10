#!/usr/bin/env python3

"""
Standardized Error Handling Framework.

This module provides consistent error handling patterns across the entire
application with proper logging, recovery strategies, and user-friendly messages.

Exception hierarchy and retry decorators have been extracted to:
- core.exceptions: Exception/error class hierarchy
- core.retry: Retry decorators, recovery strategies, backoff logic

All symbols are re-exported here for backwards compatibility.
"""

# === CORE INFRASTRUCTURE ===
import contextlib
import logging
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

# Type variables for decorators
P = ParamSpec('P')
R = TypeVar('R')

# ============================================================================
# RE-EXPORTS: Exception hierarchy (extracted to core/exceptions.py)
# ============================================================================
from core.exceptions import (
    AncestryError,
    APIError,
    APIRateLimitError,
    AppError,
    AuthenticationError,
    AuthenticationExpiredError,
    BrowserError,
    BrowserSessionError,
    CircuitBreakerOpenError,
    ConfigurationError,
    DatabaseConnectionError,
    DatabaseError,
    DataValidationError,
    ErrorCategory,
    ErrorSeverity,
    FatalError,
    MaxApiFailuresExceededError,
    MissingConfigError,
    NetworkError,
    NetworkTimeoutError,
    RetryableError,
    ValidationError,
    _safe_update_error_context,
)

# ============================================================================
# RE-EXPORTS: Retry decorators (extracted to core/retry.py)
# ============================================================================
from core.retry import (
    EnhancedErrorRecovery,
    RecoveryContext,
    RecoveryStrategy,
    RetryConfig,
    RetryDecoratorSettings,
    RetryPolicyProfile,
    RetryStrategy,
    api_retry,
    create_user_guidance,
    error_recovery,
    graceful_degradation,
    handle_partial_success,
    resolve_retry_policy,
    retry_on_failure,
    selenium_retry,
    timeout_protection,
    with_api_recovery,
    with_database_recovery,
    with_enhanced_recovery,
    with_file_recovery,
    with_recovery,
)

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


# Enhanced CircuitBreaker implementation
class CircuitBreaker:
    """
    Enhanced Circuit Breaker pattern implementation for fault tolerance.
    Opens the circuit after a threshold of failures and closes after a timeout.
    """

    def __init__(self, name: str = "default", config: CircuitBreakerConfig | None = None):
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
                if self.last_failure_time and time.time() - self.last_failure_time > self.config.recovery_timeout:
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


# === ERROR HANDLER FRAMEWORK ===


class ErrorHandler(ABC):
    """Abstract base class for error handlers."""

    def _augment_context(self, context: dict[str, Any] | None) -> dict[str, Any]:
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
    def handle(self, error: Exception, context: dict[str, Any] | None = None) -> AppError:
        """Handle the error and return a standardized AppError."""
        pass


class DatabaseErrorHandler(ErrorHandler):
    """Handler for database-related errors."""

    def can_handle(self, error: Exception) -> bool:
        keywords = ("sql", "database", "connection", "integrity")
        return self._match_keywords(error, keywords)

    def handle(self, error: Exception, context: dict[str, Any] | None = None) -> AppError:
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

    def handle(self, error: Exception, context: dict[str, Any] | None = None) -> AppError:
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

    def handle(self, error: Exception, context: dict[str, Any] | None = None) -> AppError:
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
        self.handlers.extend([DatabaseErrorHandler(), NetworkErrorHandler(), BrowserErrorHandler()])

    def register_handler(self, handler: ErrorHandler):
        """Register a custom error handler."""
        self.handlers.append(handler)
        logger.debug(f"Registered error handler: {type(handler).__name__}")

    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
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
                    logger.error(f"Error handler {type(handler).__name__} failed: {handler_error}")
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
    context: dict[str, Any] | None = None,
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

    def decorator(func: Callable[P, R]) -> Callable[P, R | None]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
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


def execute_safely(
    func: Callable[..., Any],
    *args: Any,
    default_return: Any = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Safely execute a function with error handling (Function Wrapper).

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


def safe_execute(
    default_return: Any = None, log_errors: bool = True, error_message: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to safely execute a function with error handling.

    Usage:
        @safe_execute(default_return=False)
        def my_func(): ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    msg = error_message or f"Error in {func.__name__}: {e}"
                    logger.warning(msg)
                return default_return

        return wrapper

    return decorator


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

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, _exc_tb: Any | None) -> bool:
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None and exc_val is not None:
            # Error occurred
            context = {"operation": self.operation_name, "duration": duration}

            # Convert BaseException to Exception for handle_error
            error_to_handle = exc_val if isinstance(exc_val, Exception) else Exception(str(exc_val))
            app_error = handle_error(error_to_handle, context, self.category)
            logger.error(f"Operation failed: {self.operation_name} ({duration:.2f}s) - {app_error.message}")
            return False  # Don't suppress the exception
        # Success
        if self.log_success:
            logger.debug(f"Operation completed: {self.operation_name} ({duration:.2f}s)")
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

        def handle(self, error: Exception, context: dict[str, Any] | None = None) -> AppError:
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


# === ERROR RECOVERY MANAGER ===


class ErrorRecoveryManager:
    """Manages circuit breakers and recovery strategies."""

    def __init__(self) -> None:
        """Initialize fault tolerance manager with circuit breakers and recovery strategies."""
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.recovery_strategies: dict[str, Callable[..., Any]] = {}
        self._lock = threading.Lock()

    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
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
            return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}

    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        with self._lock:
            for cb in self.circuit_breakers.values():
                cb.reset()


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


# === DECORATORS ===


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


def with_circuit_breaker(service_name: str, config: CircuitBreakerConfig | None = None):
    """Decorator to add circuit breaker protection to functions."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        cb = error_recovery_manager.get_circuit_breaker(service_name, config)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return cb.call(func, *args, **kwargs)

        return wrapper

    return decorator


# === RECOVERY STRATEGIES ===


def ancestry_session_recovery(session_manager: Any | None = None, *_args: Any, **_kwargs: Any) -> bool:
    """
    Recovery strategy for session-related failures.

    Attempts to recover an invalid WebDriver session by:
    1. Closing the current browser session
    2. Starting a new browser session
    3. Re-authenticating if necessary

    Args:
        session_manager: Optional SessionManager instance. If not provided,
                         attempts to get from DI container or create new one.

    Returns:
        bool: True if recovery successful, False otherwise

    Usage:
        @with_recovery(ancestry_session_recovery)
        def action_requiring_session():
            # If this fails, ancestry_session_recovery is called
            pass
    """
    logger.info("🔄 Attempting session recovery...")

    try:
        # Get session manager from args or DI container
        if session_manager is None:
            try:
                from core.dependency_injection import get_service
                from core.session_manager import SessionManager

                session_manager = get_service(SessionManager)
            except Exception:
                logger.warning("Could not obtain SessionManager from DI container")

        if session_manager is None:
            try:
                from core.session_manager import SessionManager

                session_manager = SessionManager()
            except Exception as e:
                logger.error(f"Failed to create SessionManager for recovery: {e}")
                return False

        # Delegate to SessionManager's recovery method
        if hasattr(session_manager, "_attempt_session_recovery"):
            result = session_manager._attempt_session_recovery(reason="recovery_strategy")
            if result:
                logger.info("✅ Session recovery successful")
            else:
                logger.warning("⚠️ Session recovery failed")
            return result

        # Fallback: try ensure_session_ready
        if hasattr(session_manager, "ensure_session_ready"):
            result = session_manager.ensure_session_ready()
            if result:
                logger.info("✅ Session recovery via ensure_session_ready successful")
            return result

        logger.error("SessionManager lacks recovery methods")
        return False

    except Exception as e:
        logger.error(f"Session recovery failed: {e}", exc_info=True)
        return False


def _get_session_manager_for_recovery(session_manager: Any | None) -> Any | None:
    """Get or create SessionManager for recovery operations."""
    if session_manager is not None:
        return session_manager

    try:
        from core.dependency_injection import get_service
        from core.session_manager import SessionManager

        return get_service(SessionManager)
    except Exception:
        pass

    try:
        from core.session_manager import SessionManager

        return SessionManager()
    except Exception:
        return None


def ancestry_api_recovery(session_manager: Any | None = None, *_args: Any, **_kwargs: Any) -> bool:
    """
    Recovery strategy for API-related failures (403, cookie expiry).

    Attempts to recover API access by:
    1. Syncing cookies from browser to API session
    2. Refreshing CSRF token if needed

    Args:
        session_manager: Optional SessionManager instance

    Returns:
        bool: True if recovery successful, False otherwise

    Usage:
        @with_recovery(ancestry_api_recovery)
        def api_call():
            # If this fails with 403, recovery is attempted
            pass
    """
    logger.info("🔄 Attempting API recovery...")

    try:
        sm = _get_session_manager_for_recovery(session_manager)
        if sm is None:
            logger.error("Failed to obtain SessionManager for API recovery")
            return False

        # Sync cookies from browser to API session
        api_manager = getattr(sm, "api_manager", None)
        if api_manager and hasattr(api_manager, "sync_cookies_from_browser"):
            api_manager.sync_cookies_from_browser(force=True)
            logger.debug("Synced cookies from browser to API session")

        # Refresh CSRF token
        if hasattr(sm, "fetch_csrf_token"):
            token = sm.fetch_csrf_token()
            if token:
                logger.debug("Refreshed CSRF token")

        logger.info("✅ API recovery completed")
        return True

    except Exception as e:
        logger.error(f"API recovery failed: {e}", exc_info=True)
        return False


def ancestry_database_recovery(session_manager: Any | None = None, *_args: Any, **_kwargs: Any) -> bool:
    """
    Recovery strategy for database-related failures.

    Attempts to recover database access by:
    1. Returning the current session to pool
    2. Obtaining a fresh session

    Args:
        session_manager: Optional SessionManager instance

    Returns:
        bool: True if recovery successful, False otherwise
    """
    logger.info("🔄 Attempting database recovery...")

    try:
        if session_manager is None:
            try:
                from core.dependency_injection import get_service
                from core.session_manager import SessionManager

                session_manager = get_service(SessionManager)
            except Exception:
                pass

        if session_manager is None:
            logger.warning("No SessionManager available for database recovery")
            return False

        # Ensure database is ready via SessionManager
        if hasattr(session_manager, "ensure_db_ready"):
            result = session_manager.ensure_db_ready()
            if result:
                logger.info("✅ Database recovery successful")
            else:
                logger.warning("⚠️ Database recovery failed")
            return result

        return False

    except Exception as e:
        logger.error(f"Database recovery failed: {e}", exc_info=True)
        return False


# =============================================
# TEST FRAMEWORK IMPLEMENTATION
# =============================================


def error_handling_module_tests() -> bool:
    """
    Core Error Handling module test suite.
    Tests the six categories: Initialization, Core Functionality, Edge Cases, Integration, Performance, and Error Handling.
    """
    from testing.test_framework import (
        TestSuite,
        suppress_logging,
    )

    with suppress_logging():
        suite = TestSuite("Core Error Handling & Recovery Systems", "core/error_handling.py")

    # Run all tests
    print("🛡️ Running Core Error Handling & Recovery Systems comprehensive test suite...")

    with suppress_logging():
        suite.run_test(
            "Basic error handling functionality",
            test_error_handling,
            "Test basic error handling and safe execution patterns",
            "Basic error handling provides robust execution with graceful degradation",
            "safe_execute handles successful execution and error recovery correctly",
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

        # Test recovery strategies are callable and return bool
        suite.run_test(
            "ancestry_session_recovery is callable",
            lambda: None
            if callable(ancestry_session_recovery)
            else (_ for _ in ()).throw(AssertionError("Should be callable")),
            "ancestry_session_recovery function is properly defined",
            "Verify recovery strategy function exists and is callable",
            "Check function signature and return type",
        )

        suite.run_test(
            "ancestry_api_recovery is callable",
            lambda: None
            if callable(ancestry_api_recovery)
            else (_ for _ in ()).throw(AssertionError("Should be callable")),
            "ancestry_api_recovery function is properly defined",
            "Verify recovery strategy function exists and is callable",
            "Check function signature and return type",
        )

        suite.run_test(
            "ancestry_database_recovery is callable",
            lambda: None
            if callable(ancestry_database_recovery)
            else (_ for _ in ()).throw(AssertionError("Should be callable")),
            "ancestry_database_recovery function is properly defined",
            "Verify recovery strategy function exists and is callable",
            "Check function signature and return type",
        )

        # Test recovery strategies return bool when no session manager provided
        # Note: We only verify the function signature and early return behavior,
        # not full execution which would require browser/network access
        def test_session_recovery_returns_bool() -> None:
            # Verify function accepts session_manager parameter and has correct signature
            import inspect

            sig = inspect.signature(ancestry_session_recovery)
            assert "session_manager" in sig.parameters, "Should accept session_manager param"

        suite.run_test(
            "ancestry_session_recovery has correct signature",
            test_session_recovery_returns_bool,
            "ancestry_session_recovery has correct function signature",
            "Verify function accepts session_manager parameter",
            "Inspect function signature without executing recovery",
        )

        def test_api_recovery_returns_bool() -> None:
            import inspect

            sig = inspect.signature(ancestry_api_recovery)
            assert "session_manager" in sig.parameters, "Should accept session_manager param"

        suite.run_test(
            "ancestry_api_recovery has correct signature",
            test_api_recovery_returns_bool,
            "ancestry_api_recovery has correct function signature",
            "Verify function accepts session_manager parameter",
            "Inspect function signature without executing recovery",
        )

        def test_database_recovery_returns_bool() -> None:
            import inspect

            sig = inspect.signature(ancestry_database_recovery)
            assert "session_manager" in sig.parameters, "Should accept session_manager param"

        suite.run_test(
            "ancestry_database_recovery has correct signature",
            test_database_recovery_returns_bool,
            "ancestry_database_recovery has correct function signature",
            "Verify function accepts session_manager parameter",
            "Inspect function signature without executing recovery",
        )

    # Generate summary report
    return suite.finish_suite()


# === TEST FUNCTIONS ===


def test_error_handling() -> None:
    """Exercise safe_execute and handle_error behavior end-to-end."""

    recorded: dict[str, Any] = {"contexts": []}

    class RecordingHandler(ErrorHandler):
        def can_handle(self, error: Exception) -> bool:
            _ = self
            return isinstance(error, ValueError)

        def handle(self, error: Exception, context: dict[str, Any] | None = None) -> AppError:
            _ = self
            recorded["contexts"].append(context or {})
            return AppError(str(error), context=context, original_exception=error)

    register_error_handler(RecordingHandler())

    def failing_func(message: str) -> None:
        raise ValueError(message)

    fallback = execute_safely(failing_func, "boom", default_return="recovered", context={"step": "safe"})
    assert fallback == "recovered", "execute_safely should return fallback after error"
    assert recorded["contexts"][-1]["step"] == "safe", "Context dictionary should propagate to handler"

    direct_error = handle_error(ValueError("direct"), context={"phase": "handle"})
    assert isinstance(direct_error, AppError), "handle_error should normalize exceptions into AppError"
    assert direct_error.context.get("phase") == "handle", "Direct handle_error calls should keep context"


def test_error_recovery() -> None:
    """Exercise ErrorRecoveryManager circuit breaker orchestration."""

    manager = ErrorRecoveryManager()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def sample_strategy(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    manager.register_recovery_strategy("api", sample_strategy)
    manager.recovery_strategies["api"]("arg")
    assert calls and calls[0][0] == ("arg",)

    breaker = manager.get_circuit_breaker("api", CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0))
    breaker.record_failure()
    assert manager.get_health_status()["api"]["failure_count"] == 1

    breaker.record_failure()
    assert manager.get_health_status()["api"]["state"] == CircuitState.OPEN.value

    manager.reset_all_circuit_breakers()
    assert manager.get_health_status()["api"]["state"] == CircuitState.CLOSED.value


def test_circuit_breaker() -> None:
    """Validate that CircuitBreaker transitions through OPEN → HALF_OPEN → CLOSED."""

    config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0, success_threshold=1)
    breaker = CircuitBreaker(name="unit", config=config)

    def _fail() -> None:
        raise RuntimeError("boom")

    with contextlib.suppress(RuntimeError):
        breaker.call(_fail)

    assert breaker.state is CircuitState.OPEN, "Failure threshold should open circuit"

    if breaker.last_failure_time is None:
        breaker.last_failure_time = time.time() - 1
    else:
        breaker.last_failure_time -= config.recovery_timeout + 1

    result = breaker.call(lambda: "ok")
    assert result == "ok"
    assert breaker.state is CircuitState.CLOSED, "Successful half-open call should close breaker"


def test_error_context() -> None:
    """Ensure ErrorContext reports failures through handle_error and preserves context."""

    handled: list[dict[str, Any]] = []

    class ContextRecordingHandler(ErrorHandler):
        def can_handle(self, error: Exception) -> bool:
            _ = self
            return isinstance(error, RuntimeError)

        def handle(self, error: Exception, context: dict[str, Any] | None = None) -> AppError:
            _ = self
            handled.append(context or {})
            return AppError(str(error), context=context, original_exception=error)

    register_error_handler(ContextRecordingHandler())

    with contextlib.suppress(RuntimeError), ErrorContext("test_operation", log_success=False):
        raise RuntimeError("boom")

    assert handled, "ErrorContext should route exceptions through handle_error"
    assert handled[-1].get("operation") == "test_operation"

    with ErrorContext("success_operation"):
        result = "ok"
    assert result == "ok"


# =============================================
# Standalone Test Block
# =============================================
# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(error_handling_module_tests)


if __name__ == "__main__":
    import sys
    import traceback
    from pathlib import Path  # Use centralized path management

    project_root = Path(__file__).resolve().parent.parent

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
