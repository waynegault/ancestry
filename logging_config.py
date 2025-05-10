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
try:
    from config import config_instance

    LOG_DIRECTORY = Path(getattr(config_instance, "LOG_DIR", "Logs"))
except ImportError:
    LOG_DIRECTORY = Path("Logs")

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


# --- Standalone Test Block ---
if __name__ == "__main__":
    print(f"\n--- Running {__file__} standalone test ---")
    if LOG_DIRECTORY:
        print(f"Using Log Directory: {LOG_DIRECTORY}")
    else:
        print("ERROR: LOG_DIRECTORY not set.")

    # Test with INFO level first
    test_log_file = "test_logging_config.log"
    main_logger = setup_logging(log_level="INFO", log_file=test_log_file)

    print(f"\n--- Initial setup (INFO Level) ---")
    if LOG_DIRECTORY:
        main_logger.info(f"Log file should be: {LOG_DIRECTORY / test_log_file}")
    main_logger.debug("Test DEBUG log (1) - Should NOT appear on console/file.")
    main_logger.info("Test INFO log (1) - Should appear on console/file.")
    main_logger.warning("Test WARNING log (1) - Should appear on console/file.")
    main_logger.info("Multi-line\n  test message\n    with different indents.")
    logging.getLogger("urllib3.connectionpool").warning(
        "Urllib3 Pool WARNING - Should NOT appear (set to ERROR)."
    )
    logging.getLogger("urllib3.connectionpool").error(
        "Urllib3 Pool ERROR - Should appear in file ONLY."
    )

    # Test toggling to DEBUG
    print("\n--- Toggling Log Level to DEBUG ---")
    main_logger = setup_logging(log_level="DEBUG")  # Call again to update levels
    main_logger.info("--- Logging state after toggle to DEBUG ---")
    main_logger.debug("Test DEBUG log (2) - Should NOW appear on console/file.")
    main_logger.info("Test INFO log (2)")
    logging.getLogger("urllib3.connectionpool").warning(
        "Urllib3 Pool WARNING - Should NOT appear (still ERROR)."
    )
    logging.getLogger("urllib3.connectionpool").error(
        "Urllib3 Pool ERROR - Should appear in file ONLY."
    )

    # Test toggling back to INFO
    print("\n--- Toggling Log Level back to INFO ---")
    main_logger = setup_logging(log_level="INFO")  # Call again
    main_logger.info("--- Logging state after toggle back to INFO ---")
    main_logger.debug("Test DEBUG log (3) - Should NOT appear on console/file again.")
    main_logger.info("Test INFO log (3)")

    print(f"\n--- Standalone Test Complete ---")
    if LOG_DIRECTORY:
        log_file_path_test = LOG_DIRECTORY / test_log_file
        print(f"Verify log messages in: {log_file_path_test}")
    else:
        print("Log directory was not set, cannot verify file.")
# End of standalone test block

# End of logging_config.py
