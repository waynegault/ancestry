#!/usr/bin/env python3

"""
Exception hierarchy and error classification for the Ancestry application.

This module defines the base exception classes, error severity levels,
error categories, and structured error types used throughout the application.
"""

import logging
import time
from enum import Enum
from typing import Any, cast

logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


# === APPLICATION EXCEPTION HIERARCHY ===


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
        user_message: str | None = None,
        technical_details: str | None = None,
        recovery_suggestion: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
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
            "original_exception": (str(self.original_exception) if self.original_exception else None),
        }


def _safe_update_error_context(error: Exception, payload: dict[str, Any] | None) -> None:
    """Attach diagnostic payloads to legacy error types that track context."""

    if not payload:
        return

    existing = getattr(error, "context", None)
    if isinstance(existing, dict):
        cast(dict[str, Any], existing).update(payload)
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
        super().__init__(message, category=ErrorCategory.API, severity=ErrorSeverity.MEDIUM, **kwargs)


# =============================================
# TEST IMPLEMENTATION
# =============================================


def test_error_types() -> None:
    """Validate specialized AncestryError subclasses capture metadata."""

    error = AppError(
        "test error",
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.HIGH,
        context={"field": "name"},
    )
    assert error.category is ErrorCategory.VALIDATION
    assert error.severity is ErrorSeverity.HIGH
    assert error.context == {"field": "name"}

    retryable = APIRateLimitError(retry_after=42)
    assert isinstance(retryable, RetryableError)
    assert retryable.retry_after == 42, "APIRateLimitError should expose retry_after value"

    validation_error = DataValidationError(validation_errors=["missing birth date"])
    assert validation_error.validation_errors == ["missing birth date"]

    auth_error = AuthenticationExpiredError(context={"session": "abc"})
    assert auth_error.context == {"session": "abc"}


def module_tests() -> bool:
    """Exception hierarchy module test suite."""
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Exception Hierarchy", "core/exceptions.py")

    print("🛡️ Running Exception Hierarchy test suite...")

    with suppress_logging():
        suite.run_test(
            "Error type instantiation",
            test_error_types,
            "Test custom error types and exception creation",
            "Error type instantiation provides structured exception handling",
            "AppError and custom error types are created and handled correctly",
        )

    return suite.finish_suite()


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    import sys
    import traceback

    print("\U0001faea Running Exception Hierarchy test suite...")
    try:
        success = module_tests()
    except Exception:
        print(
            "\n[ERROR] Unhandled exception during exceptions tests:",
            file=sys.stderr,
        )
        traceback.print_exc()
        success = False
    if not success:
        print(
            "\n[FAIL] One or more exception tests failed. See above for details.",
            file=sys.stderr,
        )
    sys.exit(0 if success else 1)
