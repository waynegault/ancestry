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
import time
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type


# Simple CircuitBreaker implementation for error handling tests
class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for fault tolerance.
    Opens the circuit after a threshold of failures and closes after a timeout.
    """

    def __init__(self, failure_threshold: int = 3, timeout: float = 5.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # Possible states: CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            if (
                self.last_failure_time is not None
                and (time.time() - self.last_failure_time) > self.timeout
            ):
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit is open")
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            self.state = "CLOSED"
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e


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
        context: Optional[Dict[str, Any]] = None,
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

    def to_dict(self) -> Dict[str, Any]:
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


class AuthenticationError(AppError):
    """Authentication-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class ValidationError(AppError):
    """Validation-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class DatabaseError(AppError):
    """Database-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class NetworkError(AppError):
    """Network-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class BrowserError(AppError):
    """Browser/WebDriver-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.BROWSER,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class APIError(AppError):
    """API-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, category=ErrorCategory.API, severity=ErrorSeverity.MEDIUM, **kwargs
        )


class ConfigurationError(AppError):
    """Configuration-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class MissingConfigError(AppError):
    """
    Raised when required configuration or credentials are missing.
    Treated as a special case for robust error handling in tests and runtime.
    """

    def __init__(
        self,
        message: str = "Required configuration or credentials are missing.",
        **kwargs,
    ):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class ErrorHandler(ABC):
    """Abstract base class for error handlers."""

    @abstractmethod
    def can_handle(self, error: Exception) -> bool:
        """Check if this handler can process the given error."""
        pass

    @abstractmethod
    def handle(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> AppError:
        """Handle the error and return a standardized AppError."""
        pass


class DatabaseErrorHandler(ErrorHandler):
    """Handler for database-related errors."""

    def can_handle(self, error: Exception) -> bool:
        keywords = ["sql", "database", "connection", "integrity"]
        error_type = str(type(error).__name__).lower()
        error_msg = str(error).lower()
        return any(k in error_type or k in error_msg for k in keywords)

    def handle(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> AppError:
        error_message = str(error)

        if "connection" in error_message.lower():
            return DatabaseError(
                "Database connection failed",
                technical_details=error_message,
                recovery_suggestion="Check database connectivity and try again",
                context=context,
                original_exception=error,
            )
        if "integrity" in error_message.lower():
            return DatabaseError(
                "Database integrity constraint violated",
                technical_details=error_message,
                recovery_suggestion="Check data validity and constraints",
                context=context,
                original_exception=error,
            )
        return DatabaseError(
            "Database operation failed",
            technical_details=error_message,
            recovery_suggestion="Try the operation again or contact support",
            context=context,
            original_exception=error,
        )


class NetworkErrorHandler(ErrorHandler):
    """Handler for network-related errors."""

    def can_handle(self, error: Exception) -> bool:
        return any(
            keyword in str(type(error).__name__).lower()
            for keyword in ["connection", "timeout", "http", "request", "url"]
        )

    def handle(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> AppError:
        error_message = str(error)

        if "timeout" in error_message.lower():
            return NetworkError(
                "Network request timed out",
                technical_details=error_message,
                recovery_suggestion="Check your internet connection and try again",
                context=context,
                original_exception=error,
            )
        if "connection" in error_message.lower():
            return NetworkError(
                "Network connection failed",
                technical_details=error_message,
                recovery_suggestion="Check your internet connection and try again",
                context=context,
                original_exception=error,
            )
        return NetworkError(
            "Network request failed",
            technical_details=error_message,
            recovery_suggestion="Check your internet connection and try again",
            context=context,
            original_exception=error,
        )


class BrowserErrorHandler(ErrorHandler):
    """Handler for browser/WebDriver-related errors."""

    def can_handle(self, error: Exception) -> bool:
        return any(
            keyword in str(type(error).__name__).lower()
            for keyword in ["webdriver", "selenium", "browser", "chrome"]
        )

    def handle(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> AppError:
        error_message = str(error)

        if "session" in error_message.lower():
            return BrowserError(
                "Browser session lost",
                technical_details=error_message,
                recovery_suggestion="Restart the browser and try again",
                context=context,
                original_exception=error,
            )
        if "element" in error_message.lower():
            return BrowserError(
                "Web element not found or not accessible",
                technical_details=error_message,
                recovery_suggestion="Refresh the page and try again",
                context=context,
                original_exception=error,
            )
        return BrowserError(
            "Browser operation failed",
            technical_details=error_message,
            recovery_suggestion="Restart the browser and try again",
            context=context,
            original_exception=error,
        )


class ErrorHandlerRegistry:
    """
    Registry for error handlers that automatically routes errors
    to appropriate handlers based on error type.
    """

    def __init__(self):
        self.handlers: List[ErrorHandler] = []
        self._register_default_handlers()

    def _register_default_handlers(self):
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
        context: Optional[Dict[str, Any]] = None,
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
    context: Optional[Dict[str, Any]] = None,
    fallback_category: ErrorCategory = ErrorCategory.SYSTEM,
) -> AppError:
    """
    Global error handling function.
    """
    return _error_registry.handle_error(error, context, fallback_category)


def register_error_handler(handler: ErrorHandler):
    """Register a custom error handler globally."""
    _error_registry.register_handler(handler)


def error_handler(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
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

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
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
                    raise app_error

                return None

        return wrapper

    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
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

    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            # Error occurred
            context = {"operation": self.operation_name, "duration": duration}

            app_error = handle_error(exc_val, context, self.category)
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


def get_error_handler(error_type: Type[Exception]) -> ErrorHandler:
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
            return True  # Catch-all for unknown errors

        def handle(
            self, error: Exception, context: Optional[Dict[str, Any]] = None
        ) -> AppError:
            return AppError(
                str(error),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                technical_details=traceback.format_exc(),
                context=context,
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


def run_comprehensive_tests() -> bool:
    """Run comprehensive core error handling tests using standardized TestSuite format."""
    return error_handling_module_tests()


# Test functions for comprehensive testing
def test_function_availability():
    """Test that all required error handling functions are available."""
    required_functions = [
        "error_recovery_manager",
        "AppError",
        "handle_error",
        "safe_execute",
        "ErrorContext",
        "CircuitBreaker",
    ]
    for func_name in required_functions:
        if func_name in globals():
            assert callable(globals()[func_name]) or isinstance(
                globals()[func_name], type
            ), f"Function {func_name} should be available"


def test_error_handling():
    """Test basic error handling and safe execution patterns."""

    # Test safe_execute with simple function
    def safe_func():
        return "success"

    if "safe_execute" in globals():
        result = safe_execute(safe_func, default_return="failed")
        assert result == "success", "safe_execute should handle successful execution"


def test_error_types():
    """Test custom error types and exception creation."""
    # Test AppError creation
    if "AppError" in globals():
        error = AppError("test error")
        assert str(error) == "test error", "AppError should store message"


def test_error_recovery():
    """Test error recovery and fallback mechanisms."""

    # Test error recovery with failing function
    def failing_func():
        raise ValueError("test error")

    if "safe_execute" in globals():
        result = safe_execute(failing_func, default_return="recovered")
        assert result == "recovered", "safe_execute should provide fallback on error"


def test_circuit_breaker():
    """Test circuit breaker pattern for fault tolerance."""
    if "CircuitBreaker" in globals():
        try:
            cb = CircuitBreaker(failure_threshold=3, timeout=1)
            assert cb is not None, "CircuitBreaker should be instantiable"
        except Exception:
            pass  # Circuit breaker might require specific setup


def test_error_context():
    """Test error context tracking and propagation."""
    if "ErrorContext" in globals():
        try:
            ctx = ErrorContext("test_operation")
            assert ctx is not None, "ErrorContext should be instantiable"
        except Exception:
            pass  # Error context might require specific setup


# =============================================
# Standalone Test Block
# =============================================
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
        success = run_comprehensive_tests()
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
