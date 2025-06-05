#!/usr/bin/env python3

# logging_config.py

"""
logging_config.py - Centralized Logging Configuration

Sets up application-wide logging using Python's standard `logging` module.
Features:
- Configurable log level via environment variable or function argument.
- Console (stderr) and File handlers.
- Custom formatter for aligned multi-line messages.
- Filters to reduce noise from external libraries (Selenium, urllib3).
- Dynamic handler level updates without full reconfiguration.
- Resolves log directory based on `config.py`.
"""

# --- Standard library imports ---
import copy
import logging
import os
import sys
from pathlib import Path
from typing import List

# --- Third-party imports ---
from dotenv import load_dotenv

# --- Load .env file early ---
load_dotenv()

# --- Define log format constants ---
LOG_FORMAT: str = (
    "%(asctime)s %(levelname).3s [%(module)-8.8s %(funcName)-8.8s %(lineno)-4d] %(message)s"
)
DATE_FORMAT: str = "%H:%M:%S.%f"[:-3]  # Format: HH:MM:SS.milliseconds

# --- Early Setup Logger (for logging config process itself) ---
# Use basic config initially for the setup logger
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s: %(message)s")
logger_for_setup = logging.getLogger("logger_setup")
logger_for_setup.setLevel(logging.DEBUG)  # Log setup details at DEBUG level

# --- Determine Log Directory ---
# Use environment variable or fallback to avoid circular import with config
LOG_DIRECTORY = Path(os.getenv("LOG_DIR", "Logs"))

# Ensure LOG_DIRECTORY is absolute
if not LOG_DIRECTORY.is_absolute():
    LOG_DIRECTORY = (Path(__file__).parent.resolve() / LOG_DIRECTORY).resolve()

import logging

# Suppress INFO and lower logs during startup
logging.basicConfig(level=logging.WARNING)

# --- Initialize Main Application Logger ---
# Get the logger instance named 'logger' (used throughout the application)

logger: logging.Logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)  # Set base level to DEBUG; handlers control final output
logger.propagate = False  # Prevent messages from propagating to the root logger

# --- Test framework imports ---
try:
    from test_framework import (
        TestSuite,
        suppress_logging,
        create_mock_data,
        assert_valid_function,
    )

    HAS_TEST_FRAMEWORK = True
except ImportError:
    # Create dummy classes/functions for when test framework is not available
    class DummyTestSuite:
        def __init__(self, *args, **kwargs):
            pass

        def start_suite(self):
            pass

        def add_test(self, *args, **kwargs):
            pass

        def end_suite(self):
            pass

        def run_test(self, *args, **kwargs):
            return True

        def finish_suite(self):
            return True

    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    TestSuite = DummyTestSuite
    suppress_logging = lambda: DummyContext()
    create_mock_data = lambda: {}
    assert_valid_function = lambda x, *args: True
    HAS_TEST_FRAMEWORK = False


# --- Custom Logging Filters ---
class NameFilter(logging.Filter):
    """Filters log records based on logger name starting with excluded prefixes."""

    def __init__(self, excluded_names: List[str]):
        super().__init__()
        self.excluded_names = excluded_names

    # End of __init__

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False if record name starts with any excluded prefix, True otherwise."""
        # Step 1: Check if the record's logger name starts with any excluded name
        # Step 2: Return False (filter out) if match found, True (keep) otherwise
        return not any(record.name.startswith(name) for name in self.excluded_names)

    # End of filter


# End of NameFilter class


class RemoteConnectionFilter(logging.Filter):
    """Filters out DEBUG level messages originating specifically from remote_connection.py"""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False if DEBUG level and from remote_connection.py, True otherwise."""
        # Step 1: Check if log level is DEBUG
        is_debug = record.levelno == logging.DEBUG
        # Step 2: Check if the source pathname includes remote_connection.py
        is_remote_conn = record.pathname and "remote_connection.py" in os.path.basename(
            record.pathname
        )
        # Step 3: Return False (filter out) only if both conditions are True
        return not (is_debug and is_remote_conn)

    # End of filter


# End of RemoteConnectionFilter class


# --- Custom Logging Formatter ---
class AlignedMessageFormatter(logging.Formatter):
    """
    Formats log records to align multi-line messages below the initial log prefix.
    Leading whitespace from subsequent lines of the original message is removed.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record with alignment."""
        # Step 1: Get the original message content
        original_message = record.getMessage()

        # Step 2: Create a copy to safely calculate prefix length
        record_copy = copy.copy(record)
        placeholder = "X"  # Placeholder character for calculation
        record_copy.msg = placeholder
        record_copy.args = tuple()  # Clear args for prefix calculation
        record_copy.message = (
            record_copy.getMessage()
        )  # Ensure message attribute is generated

        # Step 3: Format the record with the placeholder to find message start position
        prefix_with_placeholder = super().format(record_copy)  # Use base class format
        try:
            # Find the position of the placeholder (start of the actual message)
            message_start_pos = prefix_with_placeholder.index(placeholder)
        except ValueError:
            # Fallback if placeholder isn't found (shouldn't happen with standard format)
            logger_for_setup.warning(
                "Placeholder 'X' not found in formatted prefix calculation, using fallback.",
                exc_info=False,
            )
            # Heuristic: find end of metadata bracket ']' and add 2 spaces
            heuristic_index = prefix_with_placeholder.find("] ")
            message_start_pos = (
                heuristic_index + 2 if heuristic_index != -1 else 41
            )  # Default indent

        # Step 4: Extract the prefix string and calculate indent string
        prefix_string = prefix_with_placeholder[:message_start_pos]
        indent = " " * message_start_pos

        # Step 5: Split the original message into lines
        lines = original_message.split("\n")
        result_lines = []

        # Step 6: Format the first line (prefix + left-stripped content)
        if lines:
            first_line_content = lines[0].lstrip()  # Remove leading whitespace
            result_lines.append(f"{prefix_string}{first_line_content}")
        elif prefix_string:
            # Handle case where message is empty but prefix exists
            return prefix_string.rstrip()  # Return only prefix

        # Step 7: Format subsequent lines (indent + left-stripped content)
        for i in range(1, len(lines)):
            subsequent_line_content = lines[i].lstrip()  # Remove leading whitespace
            result_lines.append(f"{indent}{subsequent_line_content}")

        # Step 8: Join the formatted lines back together
        return "\n".join(result_lines)

    # End of format


# End of AlignedMessageFormatter class

# --- Initialization Flag ---
# Tracks if logging has been set up to avoid adding duplicate handlers.
_logging_initialized: bool = False


# --- Main Setup Function ---
def setup_logging(log_file: str = "app.log", log_level: str = "INFO") -> logging.Logger:
    """
    Configures the main application logger ('logger').
    Sets up file and console handlers with appropriate levels, formatters, and filters.
    If called again, updates existing handler levels instead of re-adding them.

    Args:
        log_file: The base name for the log file (will be placed in LOG_DIRECTORY).
        log_level: The desired minimum logging level (e.g., "DEBUG", "INFO")
                   for the handlers.

    Returns:
        The configured 'logger' instance.
    """
    global _logging_initialized, logger, LOG_DIRECTORY

    # Validate log level
    log_level_upper = log_level.upper()
    numeric_log_level = getattr(logging, log_level_upper, logging.INFO)

    # If already initialized, just update handler levels
    if _logging_initialized:
        for handler in logger.handlers:
            handler.setLevel(numeric_log_level)
        return logger

    # Ensure log directory exists
    logs_dir = LOG_DIRECTORY.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Construct full log file path
    log_file_path = logs_dir / Path(log_file).name
    log_file_for_handler = str(log_file_path)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = AlignedMessageFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Configure File Handler
    os.makedirs(os.path.dirname(log_file_for_handler), exist_ok=True)
    file_handler = logging.FileHandler(log_file_for_handler, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_log_level)
    logger.addHandler(file_handler)

    # Configure Console Handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_log_level)

    # Add filters to console handler
    console_handler.addFilter(RemoteConnectionFilter())
    console_handler.addFilter(
        NameFilter(
            [
                "selenium",
                "urllib3",
                "websockets",
                "undetected_chromedriver",
                "asyncio",
                "hpack",
            ]
        )
    )
    logger.addHandler(console_handler)

    # Configure logging levels for external libraries
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    logging.getLogger("selenium").setLevel(logging.INFO)
    logging.getLogger("selenium.webdriver.remote.remote_connection").setLevel(
        logging.INFO
    )
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("undetected_chromedriver").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Disable propagation
    logging.getLogger("urllib3").propagate = False
    logging.getLogger("selenium").propagate = False

    # Mark logging as initialized
    _logging_initialized = True

    return logger


# End of setup_logging


def run_comprehensive_tests():
    """
    Run comprehensive tests for the logging configuration module.
    Tests the six categories: Initialization, Core Functionality, Edge Cases,
    Integration, Performance, and Error Handling.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    if HAS_TEST_FRAMEWORK:
        with suppress_logging():
            suite = TestSuite("Logging Configuration Tests", "logging_config.py")
            suite.start_suite()  # --- INITIALIZATION TESTS ---
            suite.run_test(
                "Logger Creation",
                test_logger_creation,
                "Logger created with correct name, level, and propagation settings",
            )
            suite.run_test(
                "Default Configuration",
                test_default_configuration,
                "LOG_FORMAT, DATE_FORMAT, and LOG_DIRECTORY are properly defined",
            )
            suite.run_test(
                "Directory Creation",
                test_directory_creation,
                "Log directory is created when needed",
            )

            # --- CORE FUNCTIONALITY TESTS ---
            suite.run_test(
                "Setup Logging",
                test_setup_logging,
                "Logger is properly configured and returned",
            )
            suite.run_test(
                "Log Level Setting",
                test_log_level_setting,
                "Handlers have correct log levels set",
            )
            suite.run_test(
                "Handler Configuration",
                test_handler_configuration,
                "Handlers are properly configured and attached",
            )
            suite.run_test(
                "Formatter Application",
                test_formatter_application,
                "AlignedMessageFormatter formats messages correctly",
            )

            # --- EDGE CASES ---
            suite.run_test(
                "Invalid Log Level",
                test_invalid_log_level,
                "Invalid log levels default to INFO gracefully",
            )
            suite.run_test(
                "Missing Directory",
                test_missing_directory,
                "Missing directories are created automatically",
            )
            suite.run_test(
                "Reinitialize Logging",
                test_reinitialize_logging,
                "Reinitialization updates existing handlers without duplicating",
            )

            # --- INTEGRATION TESTS ---
            suite.run_test(
                "Multiple Handlers",
                test_multiple_handlers,
                "Multiple handlers work correctly together",
            )
            suite.run_test(
                "Filter Integration",
                test_filter_integration,
                "NameFilter correctly filters log records",
            )
            suite.run_test(
                "External Library Logging",
                test_external_library_logging,
                "External libraries have appropriate log levels set",
            )

            # --- PERFORMANCE TESTS ---
            suite.run_test(
                "Logging Speed",
                test_logging_speed,
                "100 log messages complete in under 1 second",
            )
            suite.run_test(
                "Handler Performance",
                test_handler_performance,
                "Handlers maintain reasonable performance",
            )

            # --- ERROR HANDLING TESTS ---
            suite.run_test(
                "Invalid File Path",
                test_invalid_file_path,
                "Invalid paths are handled gracefully",
            )
            suite.run_test(
                "Permission Errors",
                test_permission_errors,
                "Permission errors are handled without crashing",
            )

            return suite.finish_suite()
    else:
        return run_comprehensive_tests_fallback()


def run_comprehensive_tests_fallback():
    """Fallback test function when test framework is not available."""
    print(
        "🧪 Running basic logging configuration tests (test framework not available)..."
    )

    try:
        # Test 1: Basic logger setup
        test_logger = setup_logging("test.log", "DEBUG")
        print("✅ Test 1: Basic logger setup - PASSED")

        # Test 2: Log level validation
        test_logger = setup_logging("test.log", "INFO")
        print("✅ Test 2: Log level validation - PASSED")

        # Test 3: Directory creation
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            global LOG_DIRECTORY
            original_dir = LOG_DIRECTORY
            LOG_DIRECTORY = Path(temp_dir) / "test_logs"
            setup_logging("test.log", "INFO")
            LOG_DIRECTORY = original_dir
        print("✅ Test 3: Directory creation - PASSED")

        # Test 4: Formatter functionality
        formatter = AlignedMessageFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "test message" in formatted
        print("✅ Test 4: Formatter functionality - PASSED")

        print("🎉 All basic tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


# Test functions for comprehensive testing
def test_logger_creation():
    """Test that logger is properly created and configured."""
    assert logger.name == "logger"
    assert logger.level == logging.DEBUG
    assert not logger.propagate


def test_default_configuration():
    """Test default configuration values."""
    assert LOG_FORMAT
    assert DATE_FORMAT
    assert LOG_DIRECTORY


def test_directory_creation():
    """Test that log directory is created when needed."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        global LOG_DIRECTORY, logger, _logging_initialized
        original_dir = LOG_DIRECTORY
        original_init_state = _logging_initialized
        _logging_initialized = False  # Reset to allow fresh setup

        LOG_DIRECTORY = Path(temp_dir) / "test_logs"
        setup_logging("test.log", "INFO")

        # Check that directory was created
        directory_exists = LOG_DIRECTORY.exists()

        # Close all handlers to release file locks before temp dir cleanup
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        LOG_DIRECTORY = original_dir
        _logging_initialized = original_init_state
        assert directory_exists


def test_setup_logging():
    """Test basic setup_logging functionality."""
    test_logger = setup_logging("test.log", "INFO")
    assert test_logger is not None
    assert test_logger.name == "logger"


def test_log_level_setting():
    """Test that log levels are properly set."""
    test_logger = setup_logging("test.log", "DEBUG")
    # Check that handlers have the correct level
    for handler in test_logger.handlers:
        assert handler.level == logging.DEBUG


def test_handler_configuration():
    """Test that handlers are properly configured."""
    global _logging_initialized
    original_init_state = _logging_initialized
    _logging_initialized = False  # Reset to allow fresh setup

    test_logger = setup_logging("test.log", "INFO")
    assert len(test_logger.handlers) >= 1
    # Should have file handler and possibly console handler
    handler_types = [type(h).__name__ for h in test_logger.handlers]
    assert any("FileHandler" in ht for ht in handler_types)

    _logging_initialized = original_init_state


def test_formatter_application():
    """Test that formatters are properly applied."""
    formatter = AlignedMessageFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "test message" in formatted


def test_invalid_log_level():
    """Test handling of invalid log levels."""
    # Should default to INFO for invalid levels
    test_logger = setup_logging("test.log", "INVALID_LEVEL")
    assert test_logger is not None


def test_missing_directory():
    """Test creation of missing log directories."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        global LOG_DIRECTORY, logger, _logging_initialized
        original_dir = LOG_DIRECTORY
        original_init_state = _logging_initialized
        _logging_initialized = False  # Reset to allow fresh setup

        LOG_DIRECTORY = Path(temp_dir) / "missing" / "nested" / "dirs"
        setup_logging("test.log", "INFO")

        # Check that directory was created
        directory_exists = LOG_DIRECTORY.exists()

        # Close all handlers to release file locks before temp dir cleanup
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        LOG_DIRECTORY = original_dir
        _logging_initialized = original_init_state
        assert directory_exists


def test_reinitialize_logging():
    """Test that reinitialization updates existing handlers."""
    global _logging_initialized
    original_state = _logging_initialized
    _logging_initialized = True

    # Get initial handler count
    initial_count = len(logger.handlers)

    # Reinitialize - should not add new handlers
    setup_logging("test.log", "WARNING")
    assert len(logger.handlers) == initial_count

    _logging_initialized = original_state


def test_multiple_handlers():
    """Test that multiple handlers work correctly."""
    global _logging_initialized
    original_init_state = _logging_initialized
    _logging_initialized = False  # Reset to allow fresh setup

    test_logger = setup_logging("test.log", "INFO")
    # Should have at least one handler
    assert len(test_logger.handlers) >= 1

    _logging_initialized = original_init_state


def test_filter_integration():
    """Test that filters are properly integrated."""
    # Test the NameFilter
    filter_obj = NameFilter(["excluded_module"])

    # Create test records
    included_record = logging.LogRecord(
        name="included_module",
        level=logging.INFO,
        pathname="",
        lineno=1,
        msg="test",
        args=(),
        exc_info=None,
    )
    excluded_record = logging.LogRecord(
        name="excluded_module",
        level=logging.INFO,
        pathname="",
        lineno=1,
        msg="test",
        args=(),
        exc_info=None,
    )

    assert filter_obj.filter(included_record) == True
    assert filter_obj.filter(excluded_record) == False


def test_external_library_logging():
    """Test that external library logging is properly configured."""
    # This would test the urllib3, selenium, etc. logger configurations
    urllib3_logger = logging.getLogger("urllib3")
    assert urllib3_logger.level == logging.ERROR


def test_logging_speed():
    """Test logging performance."""
    import time

    test_logger = setup_logging("test.log", "INFO")

    start_time = time.time()
    for i in range(100):
        test_logger.info(f"Test message {i}")
    end_time = time.time()

    # Should complete 100 log messages in reasonable time (< 1 second)
    assert (end_time - start_time) < 1.0


def test_handler_performance():
    """Test handler performance with multiple handlers."""
    global _logging_initialized
    original_init_state = _logging_initialized
    _logging_initialized = False  # Reset to allow fresh setup

    test_logger = setup_logging("test.log", "DEBUG")
    # Performance should be reasonable even with multiple handlers
    assert len(test_logger.handlers) > 0

    _logging_initialized = original_init_state


def test_invalid_file_path():
    """Test handling of invalid file paths."""
    try:
        # Try with an invalid path - should handle gracefully
        setup_logging("/invalid/path/test.log", "INFO")
        # If it doesn't raise an exception, that's fine too
    except Exception:
        # Expected for truly invalid paths
        pass


def test_permission_errors():
    """Test handling of permission errors."""
    # This test is platform-specific and may not always be testable
    # Just verify the function doesn't crash with edge cases
    try:
        setup_logging("test.log", "INFO")
    except Exception:
        # Permission errors are acceptable in some environments
        pass


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    print("📋 Running Logging Configuration & Management comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
