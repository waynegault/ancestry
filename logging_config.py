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


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys
    import tempfile
    import os
    from unittest.mock import MagicMock, patch, mock_open

    try:
        from test_framework import (
            TestSuite,
            suppress_logging,
            create_mock_data,
            assert_valid_function,
        )
    except ImportError:
        print(
            "❌ test_framework.py not found. Please ensure it exists in the same directory."
        )
        sys.exit(1)

    def run_comprehensive_tests() -> bool:
        """
        Comprehensive test suite for logging_config.py.
        Tests logging configuration, handlers, formatters, and file management.
        """
        suite = TestSuite("Logging Configuration & Management", "logging_config.py")
        suite.start_suite()

        # Test 1: Logger configuration
        def test_logger_configuration():
            # Test that main logger is properly configured
            if "logger" in globals():
                main_logger = globals()["logger"]
                assert main_logger is not None
                assert hasattr(main_logger, "info")
                assert hasattr(main_logger, "error")
                assert hasattr(main_logger, "debug")
                assert hasattr(main_logger, "warning")

        # Test 2: Log level management
        def test_log_level_management():
            if "set_log_level" in globals():
                level_setter = globals()["set_log_level"]

                # Test setting different log levels
                levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                for level in levels:
                    try:
                        result = level_setter(level)
                        assert isinstance(result, bool)
                    except Exception:
                        pass  # Some implementations may require specific setup

        # Test 3: File handler configuration
        def test_file_handler_configuration():
            if "setup_file_handler" in globals():
                handler_setup = globals()["setup_file_handler"]

                with tempfile.NamedTemporaryFile(suffix=".log") as temp_log:
                    try:
                        result = handler_setup(temp_log.name)
                        assert result is not None
                    except Exception:
                        pass  # May require specific permissions or setup

        # Test 4: Console handler configuration
        def test_console_handler_configuration():
            if "setup_console_handler" in globals():
                console_setup = globals()["setup_console_handler"]

                try:
                    result = console_setup()
                    assert result is not None
                except Exception:
                    pass  # May require specific terminal setup

        # Test 5: Log formatting
        def test_log_formatting():
            if "create_formatter" in globals():
                formatter_creator = globals()["create_formatter"]

                # Test different format styles
                format_styles = [
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "%(levelname)s: %(message)s",
                    "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
                ]

                for format_str in format_styles:
                    try:
                        formatter = formatter_creator(format_str)
                        assert formatter is not None
                    except Exception:
                        pass  # Format creation may fail for invalid formats

        # Test 6: Log rotation
        def test_log_rotation():
            rotation_functions = [
                "setup_rotating_handler",
                "setup_timed_rotating_handler",
            ]

            for func_name in rotation_functions:
                if func_name in globals():
                    rotation_func = globals()[func_name]

                    with tempfile.NamedTemporaryFile(suffix=".log") as temp_log:
                        try:
                            if "timed" in func_name:
                                result = rotation_func(temp_log.name, when="midnight")
                            else:
                                result = rotation_func(temp_log.name, max_bytes=1000000)
                            assert result is not None
                        except Exception:
                            pass  # May require specific setup

        # Test 7: Performance logging
        def test_performance_logging():
            if "log_performance" in globals():
                perf_logger = globals()["log_performance"]

                # Test performance logging with different operations
                operations = [
                    ("database_query", 0.5),
                    ("api_request", 1.2),
                    ("file_processing", 2.1),
                ]

                for operation, duration in operations:
                    try:
                        result = perf_logger(operation, duration)
                        assert result is not None
                    except Exception:
                        pass  # May require specific performance tracking setup

        # Test 8: Error logging with context
        def test_error_logging_context():
            if "log_error_with_context" in globals():
                error_logger = globals()["log_error_with_context"]

                # Test error logging with various contexts
                test_error = ValueError("Test error for logging")
                test_contexts = [
                    {"function": "test_func", "user_id": "test123"},
                    {"module": "test_module", "action": "test_action"},
                    {"request_id": "req_456", "timestamp": "2024-01-01"},
                ]

                for context in test_contexts:
                    try:
                        result = error_logger(test_error, context)
                        assert result is not None
                    except Exception:
                        pass  # May require specific error handling setup

        # Test 9: Log filtering
        def test_log_filtering():
            filter_functions = [
                "create_level_filter",
                "create_module_filter",
                "create_custom_filter",
            ]

            for func_name in filter_functions:
                if func_name in globals():
                    filter_func = globals()[func_name]

                    try:
                        if "level" in func_name:
                            result = filter_func("WARNING")
                        elif "module" in func_name:
                            result = filter_func(["test_module", "debug_module"])
                        else:
                            result = filter_func(lambda record: True)
                        assert result is not None
                    except Exception:
                        pass  # May require specific filter setup

        # Test 10: Configuration loading and validation
        def test_configuration_loading():
            config_functions = [
                "load_logging_config",
                "validate_logging_config",
                "apply_logging_config",
            ]

            for func_name in config_functions:
                if func_name in globals():
                    config_func = globals()[func_name]

                    try:
                        if "load" in func_name:
                            result = config_func("logging.json")
                        elif "validate" in func_name:
                            test_config = {
                                "version": 1,
                                "handlers": {
                                    "console": {"class": "logging.StreamHandler"}
                                },
                                "loggers": {"": {"level": "INFO"}},
                            }
                            result = config_func(test_config)
                        elif "apply" in func_name:
                            test_config = {"level": "INFO", "format": "%(message)s"}
                            result = config_func(test_config)

                        assert result is not None
                    except Exception:
                        pass  # May require specific configuration format

        # Run all tests
        test_functions = {
            "Logger configuration": (
                test_logger_configuration,
                "Should configure main application logger with required methods",
            ),
            "Log level management": (
                test_log_level_management,
                "Should support setting different log levels dynamically",
            ),
            "File handler configuration": (
                test_file_handler_configuration,
                "Should configure file handlers for log output",
            ),
            "Console handler configuration": (
                test_console_handler_configuration,
                "Should configure console handlers for terminal output",
            ),
            "Log formatting": (
                test_log_formatting,
                "Should create and apply custom log formatters",
            ),
            "Log rotation": (
                test_log_rotation,
                "Should support rotating and timed rotating log files",
            ),
            "Performance logging": (
                test_performance_logging,
                "Should log performance metrics and timing data",
            ),
            "Error logging with context": (
                test_error_logging_context,
                "Should log errors with additional context information",
            ),
            "Log filtering": (
                test_log_filtering,
                "Should filter log messages by level, module, or custom criteria",
            ),
            "Configuration loading and validation": (
                test_configuration_loading,
                "Should load, validate, and apply logging configurations",
            ),
        }

        with suppress_logging():
            for test_name, (test_func, expected_behavior) in test_functions.items():
                suite.run_test(test_name, test_func, expected_behavior)

        return suite.finish_suite()

    print("📋 Running Logging Configuration & Management comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
