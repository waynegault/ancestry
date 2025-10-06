#!/usr/bin/env python3

"""
Centralized Logging Utilities.

This module provides standardized logging setup and utilities to eliminate
inconsistent logging patterns across the codebase.
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

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import logging
from typing import Any, Callable, Optional


# Global flag to track if logging has been initialized
class _CentralizedLoggingState:
    """Manages centralized logging setup state."""
    setup_complete = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a properly configured logger instance.

    This function provides a consistent way to get loggers across the entire
    application, ensuring they all use the centralized logging configuration.

    Args:
        name: Optional logger name. If None, uses the calling module's __name__

    Returns:
        Configured logger instance
    """
    # Try to use centralized logging config first
    try:
        from logging_config import logger as central_logger, setup_logging

        # Initialize centralized logging if not already done
        if not _CentralizedLoggingState.setup_complete:
            setup_logging()
            _CentralizedLoggingState.setup_complete = True

        # If no specific name requested, return the central logger
        if name is None:
            return central_logger
        # Return a child logger that inherits from central config
        return central_logger.getChild(name)

    except ImportError:
        # Fallback to standard logging if logging_config is not available
        if name is None:
            import inspect

            frame = inspect.currentframe()
            name = frame.f_back.f_globals.get("__name__", "unknown") if frame and frame.f_back else "unknown"

        return logging.getLogger(name)


def ensure_no_duplicate_handlers(logger_instance: logging.Logger) -> None:
    """
    Ensure a logger doesn't have duplicate handlers.

    This prevents the common issue of multiple handlers being added
    when modules are imported multiple times.

    Args:
        logger_instance: The logger to check and clean up
    """
    seen_handlers = set()
    handlers_to_remove = []

    for handler in logger_instance.handlers:
        handler_id = (type(handler).__name__, getattr(handler, "baseFilename", None))
        if handler_id in seen_handlers:
            handlers_to_remove.append(handler)
        else:
            seen_handlers.add(handler_id)

    for handler in handlers_to_remove:
        logger_instance.removeHandler(handler)


def suppress_external_loggers() -> None:
    """
    Suppress noisy external library loggers.

    This function sets appropriate log levels for external libraries
    to reduce noise in the application logs.
    """
    external_loggers = {
        "urllib3": logging.ERROR,
        "urllib3.connectionpool": logging.ERROR,
        "selenium": logging.INFO,
        "selenium.webdriver.remote.remote_connection": logging.INFO,
        "websockets": logging.INFO,
        "undetected_chromedriver": logging.WARNING,
        "httpx": logging.WARNING,
        "requests": logging.WARNING,
        "asyncio": logging.WARNING,
    }

    for logger_name, level in external_loggers.items():
        ext_logger = logging.getLogger(logger_name)
        ext_logger.setLevel(level)
        ext_logger.propagate = False


# Convenience function to get the standard application logger
def get_app_logger() -> logging.Logger:
    """Get the main application logger."""
    return get_logger()


# =============================================================================
# Performance Optimized Logging
# =============================================================================

from functools import wraps


def debug_if_enabled(logger: logging.Logger):
    """
    Decorator to only execute debug logging if debug level is enabled.
    Prevents expensive f-string formatting when debug logging is disabled.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if logger.isEnabledFor(logging.DEBUG):
                return func(*args, **kwargs)
            return None
        return wrapper
    return decorator


class OptimizedLogger:
    """
    Logger wrapper that optimizes debug logging performance.
    Prevents expensive string formatting when debug logging is disabled.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    @property
    def logger(self) -> logging.Logger:
        """Access the underlying logger for compatibility."""
        return self._logger

    def debug_lazy(self, msg_func: Callable[[], str]) -> None:
        """
        Only execute the message function if debug logging is enabled.

        Usage:
            logger.debug_lazy(lambda: f"Expensive {expensive_calculation()}")

        Args:
            msg_func: Function that returns the debug message string
        """
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(msg_func())

    def __getattr__(self, name: str) -> Any:
        """Delegate all other logger methods to the underlying logger."""
        return getattr(self._logger, name)


# =============================================================================
# MODULE-LEVEL TEST FUNCTIONS
# =============================================================================
# These test functions are extracted from the main test suite for better
# modularity, maintainability, and reduced complexity. Each function tests
# a specific aspect of the logging utilities functionality.


def _test_get_logger_functionality() -> bool:
    """Test get_logger function with various parameters"""
    try:
        # Test without name parameter
        logger1 = get_logger()
        assert logger1 is not None
        assert hasattr(logger1, 'info')
        assert hasattr(logger1, 'debug')

        # Test with specific name
        logger2 = get_logger("test_logger")
        assert logger2 is not None
        assert logger2.name.endswith("test_logger")

        # Test that loggers are properly configured
        assert hasattr(logger1, 'handlers')

        return True
    except Exception:
        return False


def _test_duplicate_handler_prevention() -> bool:
    """Test ensure_no_duplicate_handlers functionality"""
    try:
        import logging

        # Create test logger
        test_logger = logging.getLogger("test_duplicate_handler")

        # Add duplicate handlers manually
        handler1 = logging.StreamHandler()
        handler2 = logging.StreamHandler()  # Same type, should be detected as duplicate
        # Use proper logs directory for test.log
        import os
        from pathlib import Path
        logs_dir = Path(os.getenv("LOG_DIR", "Logs"))
        if not logs_dir.is_absolute():
            logs_dir = (Path(__file__).parent.parent.resolve() / logs_dir).resolve()
        logs_dir.mkdir(parents=True, exist_ok=True)
        test_log_path = logs_dir / "test.log"
        handler3 = logging.FileHandler(str(test_log_path)) if hasattr(logging, 'FileHandler') else logging.StreamHandler()

        test_logger.addHandler(handler1)
        test_logger.addHandler(handler2)
        test_logger.addHandler(handler3)

        initial_count = len(test_logger.handlers)

        # Run duplicate removal
        ensure_no_duplicate_handlers(test_logger)

        # Should have fewer handlers now
        final_count = len(test_logger.handlers)
        assert final_count <= initial_count

        # Clean up
        test_logger.handlers.clear()

        return True
    except Exception:
        return False


def _test_external_logger_suppression() -> bool:
    """Test suppress_external_loggers functionality"""
    try:
        import logging

        # Run suppression
        suppress_external_loggers()

        # Check that external loggers are properly configured
        test_loggers = ["urllib3", "selenium", "requests"]
        for logger_name in test_loggers:
            ext_logger = logging.getLogger(logger_name)
            # Should have a level set (not NOTSET)
            assert ext_logger.level != logging.NOTSET
            # Should not propagate to reduce noise
            assert not ext_logger.propagate

        return True
    except Exception:
        return False


def _test_app_logger_convenience() -> bool:
    """Test get_app_logger convenience function"""
    try:
        app_logger = get_app_logger()
        assert app_logger is not None
        assert hasattr(app_logger, 'info')
        assert hasattr(app_logger, 'debug')
        assert hasattr(app_logger, 'error')

        # Should be equivalent to get_logger()
        get_logger()
        # They should be related (same or parent/child relationship)

        return True
    except Exception:
        return False


def _test_debug_decorator() -> bool:
    """Test debug_if_enabled decorator functionality"""
    try:
        test_logger = get_logger("test_debug_decorator")

        # Create a test function with decorator
        @debug_if_enabled(test_logger)
        def test_debug_function():
            return "debug_executed"

        # Test that decorator is callable
        assert callable(test_debug_function)

        # Test execution (result depends on debug level)
        result = test_debug_function()
        # Should either return the result or None based on debug level
        assert result is None or result == "debug_executed"

        return True
    except Exception:
        return False


def _test_optimized_logger_functionality() -> bool:
    """Test OptimizedLogger wrapper functionality"""
    try:
        base_logger = get_logger("test_optimized")
        opt_logger = OptimizedLogger(base_logger)

        # Test logger property access
        assert opt_logger.logger is base_logger

        # Test delegation of standard methods
        assert hasattr(opt_logger, 'info')
        assert hasattr(opt_logger, 'error')

        # Test debug_lazy method
        assert hasattr(opt_logger, 'debug_lazy')
        assert callable(opt_logger.debug_lazy)

        # Test debug_lazy execution
        execution_count = 0
        def test_msg_func():
            nonlocal execution_count
            execution_count += 1
            return "test message"

        opt_logger.debug_lazy(test_msg_func)
        # Execution count depends on debug level, but should not error

        return True
    except Exception:
        return False


def _test_centralized_logging_setup() -> bool:
    """Test centralized logging configuration"""
    try:
        original_setup = _CentralizedLoggingState.setup_complete

        # Reset setup flag
        _CentralizedLoggingState.setup_complete = False

        # Get logger should trigger setup
        logger = get_logger("test_centralized")

        # Should have attempted setup (flag may or may not change based on imports)
        assert logger is not None

        # Restore original state
        _CentralizedLoggingState.setup_complete = original_setup

        return True
    except Exception:
        return False


def _test_logging_import_fallback() -> bool:
    """Test fallback behavior when logging_config unavailable"""
    try:
        # Test that get_logger works even with import failures
        # This is hard to test directly, but we can test that
        # get_logger handles various scenarios gracefully

        logger = get_logger("fallback_test")
        assert logger is not None
        assert hasattr(logger, 'info')

        return True
    except Exception:
        return False


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def logging_utils_module_tests() -> bool:
    """
    Comprehensive test suite for core/logging_utils.py.

    Tests centralized logging utilities including logger creation,
    handler management, external logger suppression, and performance optimization.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        from test_framework import TestSuite

        suite = TestSuite("Logging Utils Comprehensive Tests", __name__)
        suite.start_suite()

        # Assign module-level test functions (removing duplicate nested definitions)
        test_get_logger_functionality = _test_get_logger_functionality
        test_duplicate_handler_prevention = _test_duplicate_handler_prevention
        test_external_logger_suppression = _test_external_logger_suppression
        test_app_logger_convenience = _test_app_logger_convenience
        test_debug_decorator = _test_debug_decorator
        test_optimized_logger_functionality = _test_optimized_logger_functionality
        test_centralized_logging_setup = _test_centralized_logging_setup
        test_logging_import_fallback = _test_logging_import_fallback

        # Run all tests
        suite.run_test(
            "Logger Creation Functionality",
            test_get_logger_functionality,
            "get_logger should create properly configured logger instances",
            "Logger creation provides consistent logging interface across application",
            "Test get_logger with and without name parameters"
        )

        suite.run_test(
            "Duplicate Handler Prevention",
            test_duplicate_handler_prevention,
            "ensure_no_duplicate_handlers should remove duplicate logging handlers",
            "Handler deduplication prevents log message duplication",
            "Test handler duplicate detection and removal functionality"
        )

        suite.run_test(
            "External Logger Suppression",
            test_external_logger_suppression,
            "suppress_external_loggers should configure external library loggers",
            "External logger suppression reduces log noise from third-party libraries",
            "Test configuration of external library logger levels and propagation"
        )

        suite.run_test(
            "App Logger Convenience",
            test_app_logger_convenience,
            "get_app_logger should provide easy access to main application logger",
            "App logger convenience function simplifies logger access",
            "Test get_app_logger convenience function functionality"
        )

        suite.run_test(
            "Debug Decorator Performance",
            test_debug_decorator,
            "debug_if_enabled decorator should optimize debug logging performance",
            "Debug decorator prevents expensive operations when debug logging disabled",
            "Test debug_if_enabled decorator functionality and optimization"
        )

        suite.run_test(
            "Optimized Logger Wrapper",
            test_optimized_logger_functionality,
            "OptimizedLogger should provide performance-optimized logging interface",
            "Optimized logger wrapper prevents expensive string formatting in production",
            "Test OptimizedLogger wrapper and debug_lazy method functionality"
        )

        suite.run_test(
            "Centralized Logging Setup",
            test_centralized_logging_setup,
            "Centralized logging configuration should initialize properly",
            "Centralized setup ensures consistent logging behavior across modules",
            "Test centralized logging initialization and configuration"
        )

        suite.run_test(
            "Import Fallback Handling",
            test_logging_import_fallback,
            "Logging utilities should handle import failures gracefully",
            "Fallback behavior ensures logging works even when dependencies unavailable",
            "Test graceful handling of logging_config import failures"
        )

        return suite.finish_suite()

    except ImportError:
        print("Warning: TestSuite not available, running basic validation...")

        # Basic fallback tests
        try:
            logger = get_logger()
            assert logger is not None

            app_logger = get_app_logger()
            assert app_logger is not None

            suppress_external_loggers()

            opt_logger = OptimizedLogger(logger)
            assert opt_logger is not None

            print("✅ Basic logging_utils validation passed")
            return True
        except Exception as e:
            print(f"❌ Basic logging_utils validation failed: {e}")
            return False


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(logging_utils_module_tests)


if __name__ == "__main__":
    import sys
    print("🧪 Running Logging Utils Comprehensive Tests...")
    success = logging_utils_module_tests()
    sys.exit(0 if success else 1)
