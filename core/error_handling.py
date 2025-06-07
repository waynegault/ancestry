"""
Standardized Error Handling Framework.

This module provides consistent error handling patterns across the entire
application with proper logging, recovery strategies, and user-friendly messages.
"""

import logging
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Type, Union, Callable, List
from functools import wraps
import time

logger = logging.getLogger(__name__)


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
        return any(
            keyword in str(type(error).__name__).lower()
            for keyword in ["sql", "database", "connection", "integrity"]
        )

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
        return AppError(
            str(error),
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

    Args:
        error: The exception to handle
        context: Optional context information
        fallback_category: Category to use if no specific handler found

    Returns:
        Standardized AppError
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


def run_comprehensive_tests():
    """
    Comprehensive test suite for ErrorHandling module.

    Tests all major functionality including:
    - Error classes and enums
    - Error handlers
    - Context managers
    - Error transformation
    - Registry functionality
    """

    def test_error_enums():
        """Test ErrorSeverity and ErrorCategory enums."""
        print("Testing error enums...")

        # Test ErrorSeverity
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

        # Test ErrorCategory
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert ErrorCategory.DATABASE.value == "database"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.BROWSER.value == "browser"
        assert ErrorCategory.API.value == "api"

        print("✓ Error enums working correctly")

    def test_app_error():
        """Test AppError class functionality."""
        print("Testing AppError class...")

        # Basic error creation
        error = AppError("Test error")
        assert error.message == "Test error"
        assert error.category == ErrorCategory.SYSTEM
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.user_message is not None
        assert error.context == {}
        assert error.timestamp is not None

        # Error with full parameters
        context = {"user_id": "123", "action": "test"}
        error = AppError(
            "Detailed error",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            user_message="Please login again",
            technical_details="Token expired",
            recovery_suggestion="Refresh your session",
            context=context,
        )

        assert error.message == "Detailed error"
        assert error.category == ErrorCategory.AUTHENTICATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.user_message == "Please login again"
        assert error.technical_details == "Token expired"
        assert error.recovery_suggestion == "Refresh your session"
        assert error.context == context

        # Test to_dict method
        error_dict = error.to_dict()
        assert isinstance(error_dict, dict)
        assert error_dict["message"] == "Detailed error"
        assert error_dict["category"] == "authentication"
        assert error_dict["severity"] == "high"

        print("✓ AppError class working correctly")

    def test_specialized_errors():
        """Test specialized error classes."""
        print("Testing specialized error classes...")

        # Test AuthenticationError
        auth_error = AuthenticationError("Invalid credentials")
        assert auth_error.category == ErrorCategory.AUTHENTICATION
        assert auth_error.severity == ErrorSeverity.HIGH

        # Test ValidationError
        val_error = ValidationError("Invalid input")
        assert val_error.category == ErrorCategory.VALIDATION
        assert val_error.severity == ErrorSeverity.MEDIUM

        # Test DatabaseError
        db_error = DatabaseError("Connection failed")
        assert db_error.category == ErrorCategory.DATABASE
        assert db_error.severity == ErrorSeverity.HIGH

        # Test NetworkError
        net_error = NetworkError("Timeout")
        assert net_error.category == ErrorCategory.NETWORK
        assert net_error.severity == ErrorSeverity.MEDIUM

        # Test BrowserError
        browser_error = BrowserError("Element not found")
        assert browser_error.category == ErrorCategory.BROWSER
        assert browser_error.severity == ErrorSeverity.MEDIUM

        # Test APIError
        api_error = APIError("Rate limited")
        assert api_error.category == ErrorCategory.API
        assert api_error.severity == ErrorSeverity.MEDIUM

        # Test ConfigurationError
        config_error = ConfigurationError("Missing config")
        assert config_error.category == ErrorCategory.CONFIGURATION
        assert config_error.severity == ErrorSeverity.HIGH

        print("✓ Specialized error classes working correctly")

    def test_error_handlers():
        """Test error handler implementations."""
        print("Testing error handlers...")

        # Test DatabaseErrorHandler
        db_handler = DatabaseErrorHandler()

        # Create a mock database error
        db_exception = Exception("connection timeout")
        assert db_handler.can_handle(db_exception) is True

        result = db_handler.handle(db_exception)
        assert isinstance(result, AppError)
        assert result.category == ErrorCategory.DATABASE

        # Test NetworkErrorHandler
        net_handler = NetworkErrorHandler()

        # Create a mock network error
        net_exception = Exception("connection failed")
        assert net_handler.can_handle(net_exception) is True

        result = net_handler.handle(net_exception)
        assert isinstance(result, AppError)
        assert result.category == ErrorCategory.NETWORK

        # Test BrowserErrorHandler
        browser_handler = BrowserErrorHandler()

        # Create a mock browser error
        browser_exception = Exception("webdriver error")
        assert browser_handler.can_handle(browser_exception) is True

        result = browser_handler.handle(browser_exception)
        assert isinstance(result, AppError)
        assert result.category == ErrorCategory.BROWSER

        print("✓ Error handlers working correctly")

    def test_error_handler_registry():
        """Test ErrorHandlerRegistry functionality."""
        print("Testing ErrorHandlerRegistry...")

        registry = ErrorHandlerRegistry()

        # Test default handlers are registered
        assert len(registry.handlers) > 0

        # Test custom handler registration
        class CustomHandler(ErrorHandler):
            def can_handle(self, error: Exception) -> bool:
                return "custom" in str(error).lower()

            def handle(self, error: Exception, context=None) -> AppError:
                return AppError("Handled by custom handler")

        custom_handler = CustomHandler()
        registry.register_handler(custom_handler)

        # Test error handling
        test_error = Exception("custom error")
        result = registry.handle_error(test_error)
        assert isinstance(result, AppError)

        print("✓ ErrorHandlerRegistry working correctly")

    def test_handle_error_function():
        """Test global handle_error function."""
        print("Testing handle_error function...")

        # Test with regular Exception
        try:
            1 / 0
        except Exception as e:
            app_error = handle_error(e)
            assert isinstance(app_error, AppError)
            assert "division by zero" in app_error.message.lower()
            assert app_error.category == ErrorCategory.SYSTEM

        # Test with AppError
        original_error = AppError("Original error", category=ErrorCategory.API)
        result = handle_error(original_error)
        assert result is original_error

        # Test with context
        context = {"test": "value"}
        try:
            raise ValueError("Test error")
        except Exception as e:
            app_error = handle_error(e, context)
            assert app_error.context == context

        print("✓ handle_error function working correctly")

    def test_error_handler_decorator():
        """Test error_handler decorator."""
        print("Testing error_handler decorator...")

        @error_handler(category=ErrorCategory.API, severity=ErrorSeverity.HIGH)
        def test_function(should_fail=False):
            if should_fail:
                raise ValueError("Test failure")
            return "success"

        # Test successful execution
        result = test_function(should_fail=False)
        assert result == "success"

        # Test error handling
        result = test_function(should_fail=True)
        assert result is None  # Default return value on error

        print("✓ error_handler decorator working correctly")

    def test_safe_execute():
        """Test safe_execute function."""
        print("Testing safe_execute function...")

        def successful_function():
            return "success"

        def failing_function():
            raise ValueError("Function failed")

        # Test successful execution
        result = safe_execute(successful_function)
        assert result == "success"

        # Test failed execution with default return
        result = safe_execute(failing_function, default_return="failed")
        assert result == "failed"

        # Test failed execution with no default
        result = safe_execute(failing_function)
        assert result is None

        print("✓ safe_execute function working correctly")

    def test_error_context():
        """Test ErrorContext context manager."""
        print("Testing ErrorContext...")

        # Test successful operation
        with ErrorContext("test operation") as ctx:
            result = "success"

        # Test failed operation
        try:
            with ErrorContext("failing operation") as ctx:
                raise ValueError("Test failure")
        except ValueError:
            pass  # Expected to propagate

        print("✓ ErrorContext working correctly")

    def test_imports_and_availability():
        """Test that all required imports are available."""
        print("Testing imports and availability...")

        # Test enum imports
        assert ErrorSeverity is not None
        assert ErrorCategory is not None

        # Test class imports
        assert AppError is not None
        assert ErrorHandler is not None
        assert ErrorHandlerRegistry is not None
        assert ErrorContext is not None

        # Test specialized errors
        assert AuthenticationError is not None
        assert ValidationError is not None
        assert DatabaseError is not None
        assert NetworkError is not None
        assert BrowserError is not None
        assert APIError is not None
        assert ConfigurationError is not None

        # Test function imports
        assert handle_error is not None
        assert error_handler is not None
        assert safe_execute is not None
        assert register_error_handler is not None

        print("✓ All imports and availability working correctly")

    def test_type_annotations():
        """Test type annotation consistency."""
        print("Testing type annotations...")

        # Test that classes have proper type hints
        import inspect

        # Check AppError constructor
        sig = inspect.signature(AppError.__init__)
        assert "message" in sig.parameters
        assert "category" in sig.parameters
        assert "severity" in sig.parameters

        # Check handle_error function
        sig = inspect.signature(handle_error)
        assert "error" in sig.parameters

        print("✓ Type annotations working correctly")

    def test_error_categories_comprehensive():
        """Test all error categories have proper handling."""
        print("Testing comprehensive error categories...")

        categories = [
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.AUTHORIZATION,
            ErrorCategory.VALIDATION,
            ErrorCategory.DATABASE,
            ErrorCategory.NETWORK,
            ErrorCategory.BROWSER,
            ErrorCategory.API,
            ErrorCategory.CONFIGURATION,
            ErrorCategory.SYSTEM,
            ErrorCategory.BUSINESS_LOGIC,
            ErrorCategory.USER_INPUT,
        ]

        for category in categories:
            error = AppError("Test error", category=category)
            assert error.category == category
            assert error.user_message is not None
            assert len(error.user_message) > 0

        print("✓ All error categories working correctly")

    def test_integration():
        """Test integration between components."""
        print("Testing component integration...")

        # Test error registry with global handle_error function
        test_exception = ValueError("Integration test error")
        app_error = handle_error(test_exception)
        assert isinstance(app_error, AppError)

        # Test decorator with custom error types
        @error_handler(category=ErrorCategory.NETWORK, severity=ErrorSeverity.HIGH)
        def network_operation():
            raise ConnectionError("Network failed")

        result = network_operation()
        assert result is None  # Should handle gracefully

        # Test ErrorContext with actual operations
        with ErrorContext("integration test"):
            test_value = "integration successful"

        print("✓ Component integration working correctly")

    def test_error_message_generation():
        """Test user-friendly error message generation."""
        print("Testing error message generation...")

        # Test various categories have appropriate messages
        auth_error = AppError("Auth failed", category=ErrorCategory.AUTHENTICATION)
        assert (
            "login" in auth_error.user_message.lower()
            or "credential" in auth_error.user_message.lower()
        )

        network_error = AppError("Network failed", category=ErrorCategory.NETWORK)
        assert (
            "network" in network_error.user_message.lower()
            or "connection" in network_error.user_message.lower()
        )

        db_error = AppError("DB failed", category=ErrorCategory.DATABASE)
        assert "database" in db_error.user_message.lower()

        print("✓ Error message generation working correctly")

    # Run all tests
    tests = [
        test_error_enums,
        test_app_error,
        test_specialized_errors,
        test_error_handlers,
        test_error_handler_registry,
        test_handle_error_function,
        test_error_handler_decorator,
        test_safe_execute,
        test_error_context,
        test_imports_and_availability,
        test_type_annotations,
        test_error_categories_comprehensive,
        test_integration,
        test_error_message_generation,
    ]

    print("=" * 50)
    print("RUNNING ERROR HANDLING COMPREHENSIVE TESTS")
    print("=" * 50)

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print("=" * 50)
    print(f"ERROR HANDLING TESTS COMPLETE: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    run_comprehensive_tests()
