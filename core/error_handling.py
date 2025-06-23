"""
Standardized Error Handling Framework.

This module provides consistent error handling patterns across the entire
application with proper logging, recovery strategies, and user-friendly messages.
"""

from core_imports import standardize_module_imports, auto_register_module
standardize_module_imports()
auto_register_module(globals(), __name__)

import logging
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Type, Union, Callable, List
from functools import wraps
import time

from logging_config import logger

try:
    from core_imports import auto_register_module
    auto_register_module(globals(), __name__)
except ImportError:
    pass  # Continue without auto-registration if not available


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
        elif "integrity" in error_message.lower():
            return DatabaseError(
                "Database integrity constraint violated",
                technical_details=error_message,
                recovery_suggestion="Check data validity and constraints",
                context=context,
                original_exception=error,
            )
        else:
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
        elif "connection" in error_message.lower():
            return NetworkError(
                "Network connection failed",
                technical_details=error_message,
                recovery_suggestion="Check your internet connection and try again",
                context=context,
                original_exception=error,
            )
        else:
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
        elif "element" in error_message.lower():
            return BrowserError(
                "Web element not found or not accessible",
                technical_details=error_message,
                recovery_suggestion="Refresh the page and try again",
                context=context,
                original_exception=error,
            )
        else:
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
        else:
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
    handler = next(
        (h for t, h in error_handlers.items() if t in str(error_type).lower()),
        DefaultHandler(),
    )
    return handler


# =============================================
# TEST FRAMEWORK IMPLEMENTATION
# =============================================


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for core/error_handling.py.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Error Handling Framework", "error_handling.py")
    suite.start_suite()

    # Test Functions
    def test_app_error_creation():
        """Test AppError creation."""
        error = AppError(
            message="Test error",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            user_message="This is a test error.",
            technical_details="unittest",
            recovery_suggestion="None, this is a test.",
        )
        assert error.message == "Test error"
        assert error.category == ErrorCategory.SYSTEM
        assert error.severity == ErrorSeverity.CRITICAL
        return True

    def test_missing_config_error():
        """Test MissingConfigError behaves as expected."""
        try:
            raise MissingConfigError("Missing config for test.")
        except MissingConfigError as e:
            assert isinstance(e, AppError)
            assert e.category == ErrorCategory.CONFIGURATION
            return True
        return False

    def test_specialized_error_handlers():
        """Test specialized error handlers."""
        db_handler = DatabaseErrorHandler()
        db_exception = Exception("connection timeout")
        assert db_handler.can_handle(db_exception)
        result = db_handler.handle(db_exception)
        assert isinstance(result, AppError)
        assert result.category == ErrorCategory.DATABASE
        return True

    def test_error_handler_registry():
        """Test ErrorHandlerRegistry functionality."""
        registry = ErrorHandlerRegistry()
        assert len(registry.handlers) > 0

        class CustomHandler(ErrorHandler):
            def can_handle(self, error: Exception) -> bool:
                return "custom" in str(error).lower()

            def handle(self, error: Exception, context=None) -> AppError:
                return AppError("Handled by custom handler")

        registry.register_handler(CustomHandler())
        result = registry.handle_error(Exception("custom error"))
        assert isinstance(result, AppError)
        return True

    def test_handle_error_function():
        """Test global handle_error function."""
        try:
            raise ZeroDivisionError()
        except Exception as e:
            app_error = handle_error(e)
            assert isinstance(app_error, AppError)
            assert "division by zero" in app_error.message.lower()
        return True

    def test_error_handler_decorator():
        """Test error_handler decorator."""

        @error_handler()
        def test_function(should_fail=False):
            if should_fail:
                raise ValueError("Test failure")
            return "success"

        assert test_function(should_fail=False) == "success"
        assert test_function(should_fail=True) is None
        return True

    def test_safe_execute():
        """Test safe_execute function."""

        def successful_function():
            return "success"

        def failing_function():
            raise ValueError("Function failed")

        assert safe_execute(successful_function) == "success"
        assert safe_execute(failing_function, default_return="failed") == "failed"
        return True

    def test_error_context():
        """Test ErrorContext context manager."""
        with ErrorContext("test operation"):
            pass
        try:
            with ErrorContext("failing operation"):
                raise ValueError("Test failure")
        except ValueError:
            pass
        return True

    # Running tests
    suite.run_test(
        "AppError Creation",
        test_app_error_creation,
        "AppError instances are created correctly.",
        "Test AppError creation.",
        "Test AppError creation.",
    )
    suite.run_test(
        "MissingConfigError",
        test_missing_config_error,
        "MissingConfigError is raised and categorized correctly.",
        "Test MissingConfigError.",
        "Test MissingConfigError.",
    )
    suite.run_test(
        "Specialized Error Handlers",
        test_specialized_error_handlers,
        "Specialized error handlers correctly categorize errors.",
        "Test specialized error handlers.",
        "Test specialized error handlers.",
    )
    suite.run_test(
        "Error Handler Registry",
        test_error_handler_registry,
        "ErrorHandlerRegistry manages and applies handlers.",
        "Test ErrorHandlerRegistry.",
        "Test ErrorHandlerRegistry.",
    )
    suite.run_test(
        "Global handle_error",
        test_handle_error_function,
        "Global handle_error function correctly wraps exceptions.",
        "Test handle_error function.",
        "Test handle_error function.",
    )
    suite.run_test(
        "Error Handler Decorator",
        test_error_handler_decorator,
        "error_handler decorator correctly catches exceptions.",
        "Test error_handler decorator.",
        "Test error_handler decorator.",
    )
    suite.run_test(
        "Safe Execute",
        test_safe_execute,
        "safe_execute correctly runs functions and handles errors.",
        "Test safe_execute.",
        "Test safe_execute.",
    )
    suite.run_test(
        "Error Context",
        test_error_context,
        "ErrorContext correctly manages code blocks.",
        "Test ErrorContext.",
        "Test ErrorContext.",
    )

    return suite.finish_suite()


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
        from path_manager import ensure_imports

        ensure_imports()
    except ImportError:
        # Fallback for testing environment
        # Replaced with standardize_module_imports()
        pass

    print("\U0001faea Running Error Handling comprehensive test suite...")
    try:
        success = run_comprehensive_tests()
    except Exception as e:
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
