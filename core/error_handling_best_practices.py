#!/usr/bin/env python3

"""
error_handling_best_practices.py - Guidelines and helpers for consistent error handling

This module provides:
1. Decorators for common error handling patterns
2. Logging standards for exception handlers
3. Migration guide for improving existing code

Usage:
    from core.error_handling_best_practices import (
        log_exceptions,
        graceful_fallback,
        ErrorHandlingContext,
    )
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def log_exceptions(
    level: int = logging.ERROR,
    message: str | None = None,
    include_exc_info: bool = True,
    reraise: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that logs exceptions before re-raising or returning fallback.

    Args:
        level: Log level (default: ERROR)
        message: Custom message (default: auto-generated from function name)
        include_exc_info: Include traceback in logs (default: True)
        reraise: Re-raise exception after logging (default: True)

    Example:
        @log_exceptions(level=logging.WARNING, reraise=False)
        def risky_operation():
            return 1 / 0  # Will log and return None
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_message = message or f"Exception in {func.__name__}"
                logger.log(level, f"{log_message}: {e}", exc_info=include_exc_info)
                if reraise:
                    raise
                return None  # type: ignore[return-value]

        return wrapper

    return decorator


def graceful_fallback(
    fallback_value: Any = None,
    log_level: int = logging.WARNING,
    log_message: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that returns a fallback value on exception instead of raising.

    Args:
        fallback_value: Value to return on exception (default: None)
        log_level: Log level for the exception (default: WARNING)
        log_message: Custom log message (default: auto-generated)

    Example:
        @graceful_fallback(fallback_value=[], log_level=logging.DEBUG)
        def fetch_optional_data():
            return risky_api_call()  # Returns [] on failure
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = log_message or f"Using fallback for {func.__name__}"
                logger.log(log_level, f"{msg}: {e}", exc_info=True)
                return fallback_value  # type: ignore[return-value]

        return wrapper

    return decorator


class ErrorHandlingContext:
    """
    Context manager for structured error handling with logging.

    Example:
        with ErrorHandlingContext("Loading config", fallback={}):
            config = load_config_file()
        # If load_config_file() raises, config = {} and warning is logged
    """

    def __init__(
        self,
        operation: str,
        fallback: Any = None,
        log_level: int = logging.WARNING,
        reraise: bool = False,
    ) -> None:
        self.operation = operation
        self.fallback = fallback
        self.log_level = log_level
        self.reraise = reraise
        self.exception: Exception | None = None

    def __enter__(self) -> Any:
        return self.fallback

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> bool:
        if exc_val is not None:
            self.exception = exc_val
            logger.log(
                self.log_level,
                f"Error in {self.operation}: {exc_val}",
                exc_info=True,
            )
            return not self.reraise  # True suppresses, False propagates
        return False


# ============================================================================
# ERROR HANDLING STANDARDS
# ============================================================================

ERROR_HANDLING_GUIDELINES = """
Error Handling Standards for Ancestry Project
==============================================

1. NEVER swallow exceptions silently - ALWAYS log at appropriate level:
   - ERROR: Real failures that affect functionality
   - WARNING: Degraded functionality, system still works
   - DEBUG: Expected failures for optional features

2. Use specific exception types when possible:
   - except ValueError: instead of except Exception:
   - except KeyError: instead of except Exception:
   - Reserve broad except Exception: for truly optional subsystems

3. Always include exc_info=True in logger calls:
   - logger.error("Failed", exc_info=True)  # ✅ Shows traceback
   - logger.error("Failed")  # ❌ No context for debugging

4. Use decorators for common patterns:
   - @log_exceptions for functions that should log and re-raise
   - @graceful_fallback for optional features with defaults

5. Distinguish between:
   - CRITICAL PATHS: Database, authentication, core workflow
     → Must log ERROR and often re-raise
   - OPTIONAL FEATURES: Analytics, metrics, caching
     → Can log DEBUG/WARNING and return fallback

6. Migration strategy for existing code:
   a. Identify silent exception handlers (no logging)
   b. Add logger.debug() at minimum
   c. Evaluate if error should be higher severity
   d. Consider if exception should be re-raised
   e. Add tests to verify error handling behavior
"""


def error_handling_module_tests() -> bool:
    """Test error handling helpers."""
    print("🔧 Testing Error Handling Best Practices...")
    print()

    all_passed = True

    # Test 1: log_exceptions decorator
    print("Test 1: log_exceptions decorator")
    try:

        @log_exceptions(level=logging.ERROR, reraise=False)
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()
        if result is None:
            print("✅ PASSED: Returns None on exception")
        else:
            print(f"❌ FAILED: Expected None, got {result}")
            all_passed = False
    except Exception as e:
        print(f"❌ FAILED: Unexpected exception: {e}")
        all_passed = False

    # Test 2: graceful_fallback decorator
    print("Test 2: graceful_fallback decorator")
    try:

        @graceful_fallback(fallback_value="fallback")
        def failing_function_with_fallback():
            raise RuntimeError("Test")

        result = failing_function_with_fallback()
        if result == "fallback":
            print("✅ PASSED: Returns fallback value")
        else:
            print(f"❌ FAILED: Expected 'fallback', got {result}")
            all_passed = False
    except Exception as e:
        print(f"❌ FAILED: Unexpected exception: {e}")
        all_passed = False

    # Test 3: ErrorHandlingContext
    print("Test 3: ErrorHandlingContext")
    try:
        ctx = ErrorHandlingContext("test operation", fallback="default", reraise=False)
        with ctx:
            raise KeyError("missing")

        if ctx.exception is not None and isinstance(ctx.exception, KeyError):
            print("✅ PASSED: Context captured exception")
        else:
            print(f"❌ FAILED: Expected KeyError, got {ctx.exception}")
            all_passed = False
    except Exception as e:
        print(f"❌ FAILED: Exception leaked: {e}")
        all_passed = False

    if all_passed:
        print("\n🎉 All error handling tests PASSED")
    else:
        print("\n❌ Some tests FAILED")

    return all_passed


# Use centralized test runner utility from test_utilities
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(error_handling_module_tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
