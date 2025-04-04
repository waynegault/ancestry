# logging_config.py

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import io
from typing import Optional, cast, List, Any, Dict
from datetime import datetime
import copy
import threading

# --- Load .env file at the top ---
load_dotenv()

# --- Define log format strings ---
LOG_FORMAT: str = (
    "%(asctime)s %(levelname).3s [%(module)-8.8s %(funcName)-8.8s %(lineno)-4d] %(message)s"
)
DATE_FORMAT: str = "%H:%M:%S.%f"[:-3] # Format as HH:MM:SS.mmm (milliseconds)

# --- Early Setup Logger ---
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s: %(message)s")
logger_for_setup = logging.getLogger("logger_setup")

# --- Attempt to import config_instance ---
try:
    from config import config_instance
    logger_for_setup.info("Successfully imported config_instance.")
except ImportError as e:
    logger_for_setup.error(f"Failed to import config_instance from config.py: {e}")
    class DummyConfig:
        LOG_DIR = Path(__file__).parent.resolve() / "logs_fallback"
        DATABASE_FILE = Path("dummy_db.db") # Example attribute
    config_instance = DummyConfig()
    logger_for_setup.warning("Using dummy config due to import failure.")

# Determine Log Directory using config_instance or fallback
try:
    _log_dir_path = Path(config_instance.LOG_DIR)
    LOG_DIRECTORY: Path = _log_dir_path.resolve() # Resolve to absolute path
    logger_for_setup.info(f"Using resolved LOG_DIRECTORY from config: {LOG_DIRECTORY}")
except AttributeError:
    logger_for_setup.warning("config_instance.LOG_DIR not found. Using default Logs directory.")
    LOG_DIRECTORY = (Path(__file__).parent.resolve() / "Logs").resolve()
except (TypeError, ValueError) as e:
    logger_for_setup.warning(f"config_instance.LOG_DIR ('{getattr(config_instance, 'LOG_DIR', 'N/A')}') is not a valid path ({e}). Using default Logs directory.")
    LOG_DIRECTORY = (Path(__file__).parent.resolve() / "Logs").resolve()
except Exception as e:
    logger_for_setup.critical(f"Error resolving LOG_DIRECTORY: {e}. Using fallback.", exc_info=True)
    LOG_DIRECTORY = (Path(__file__).parent.resolve() / "Logs").resolve()


# --- Initialize Main Logger ---
logger: logging.Logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# --- Custom Filters ---
class NameFilter(logging.Filter):
    """Filters log records based on logger name starting with excluded prefixes."""
    def __init__(self, excluded_names: List[str]):
        super().__init__()
        self.excluded_names = excluded_names
    def filter(self, record: logging.LogRecord) -> bool:
        return not any(record.name.startswith(name) for name in self.excluded_names)

class RemoteConnectionFilter(logging.Filter):
    """Filters out DEBUG messages specifically from remote_connection.py"""
    def filter(self, record: logging.LogRecord) -> bool:
        is_debug = record.levelno == logging.DEBUG
        is_remote_conn = record.pathname and "remote_connection.py" in os.path.basename(record.pathname)
        return not (is_debug and is_remote_conn)

# --- Custom Formatter for aligning multi-line messages ---
class AlignedMessageFormatter(logging.Formatter):
    """
    Formats log records, aligning continued lines of multi-line messages
    to the start of the message content, ignoring original leading whitespace.
    """
    def format(self, record: logging.LogRecord) -> str:
        original_message = record.getMessage()
        record_copy = copy.copy(record)
        placeholder = "X"
        record_copy.msg = placeholder
        record_copy.args = tuple()
        # Ensure message attribute is generated before formatting
        record_copy.message = record_copy.getMessage()
        prefix_with_placeholder = logging.Formatter.format(self, record_copy)
        try:
            message_start_pos = prefix_with_placeholder.index(placeholder)
        except ValueError:
            logger_for_setup.warning("Placeholder 'X' not found in formatted prefix calculation, using fallback.", exc_info=False)
            heuristic_index = prefix_with_placeholder.find("] ")
            message_start_pos = heuristic_index + 2 if heuristic_index != -1 else 41
        prefix_string = prefix_with_placeholder[:message_start_pos]
        indent = " " * message_start_pos
        lines = original_message.split("\n")
        result_lines = []
        if lines:
            first_line_content = lines[0].lstrip()
            result_lines.append(f"{prefix_string}{first_line_content}")
        # Handle case where prefix exists but message is empty
        elif prefix_string:
            return prefix_string.rstrip() # Return just the prefix (remove trailing space/newline)

        for i in range(1, len(lines)):
            subsequent_line_content = lines[i].lstrip()
            result_lines.append(f"{indent}{subsequent_line_content}")
        return "\n".join(result_lines)


# --- Initialization Flag ---
_logging_initialized: bool = False

# --- Main Setup Function ---
def setup_logging(log_file: str = "app.log", log_level: str = "INFO") -> logging.Logger:
    """
    V1.1 REVISED: Configures logging with console and standard file handlers.
    - Uses AlignedMessageFormatter for multi-line alignment.
    - Console/File handlers set dynamically based on log_level parameter.
    - Logger base level is always DEBUG; handlers control output.
    - Explicitly sets external library levels (urllib3, selenium, etc.)
    - Suppresses urllib3.connectionpool warnings at INFO level console output.
    """
    global _logging_initialized, logger, LOG_DIRECTORY, logger_for_setup

    # --- Validate and Determine Log Level ---
    log_level_upper = log_level.upper()
    numeric_log_level = getattr(logging, log_level_upper, None)
    if numeric_log_level is None:
        logger_for_setup.warning(f"Invalid log level '{log_level}'. Using default level INFO.")
        log_level_upper = "INFO"
        numeric_log_level = logging.INFO

    # --- Update Handler Levels if Already Initialized ---
    if _logging_initialized:
        updated_console = False
        updated_file = False
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                if handler.level != numeric_log_level:
                    handler.setLevel(numeric_log_level)
                    updated_console = True
            elif isinstance(handler, logging.FileHandler):
                if handler.level != numeric_log_level:
                    handler.setLevel(numeric_log_level)
                    updated_file = True
        if updated_console or updated_file:
             logger_for_setup.info(f"Updated handler log levels to: {log_level_upper}")
        return logger

    # --- First-Time Initialization ---
    logger_for_setup.debug("Performing first-time logging setup...")
    try:
        logs_dir = LOG_DIRECTORY.resolve()
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger_for_setup.debug(f"Log directory ensured: {logs_dir}")
    except OSError as e:
        logger_for_setup.critical(f"Failed to create log directory '{logs_dir}': {e}", exc_info=True)

    log_file_path = logs_dir / Path(log_file).name
    log_file_for_handler = str(log_file_path)

    # Clear existing handlers
    if logger.hasHandlers():
        logger_for_setup.debug(f"Clearing {len(logger.handlers)} existing handlers from logger '{logger.name}'...")
        for handler in logger.handlers[:]:
            try:
                if hasattr(handler, "flush"): handler.flush()
                if hasattr(handler, "close"): handler.close()
            except Exception as close_err:
                logger_for_setup.warning(f"Error closing handler {handler}: {close_err}", exc_info=False)
            logger.removeHandler(handler)
    if logger_for_setup.hasHandlers():
        for handler in logger_for_setup.handlers[:]:
            logger_for_setup.removeHandler(handler)

    # Create formatter
    formatter = AlignedMessageFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    logger_for_setup.debug(f"Formatter created: fmt='{LOG_FORMAT}', datefmt='{DATE_FORMAT}'")

    # --- Configure Handlers ---
    # File Handler
    try:
        os.makedirs(os.path.dirname(log_file_for_handler), exist_ok=True)
        file_handler = logging.FileHandler(log_file_for_handler, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_log_level) # Set based on input
        logger.addHandler(file_handler)
        logger_for_setup.debug(f"Added standard FileHandler for: {log_file_for_handler} (Level: {log_level_upper})")
    except Exception as e:
        logger_for_setup.critical(f"Failed to create/add file handler for '{log_file_for_handler}': {e}", exc_info=True)

    # Console Handler
    try:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_log_level) # Set based on input

        # Add Filters to Console Handler
        console_handler.addFilter(RemoteConnectionFilter())
        # Filter external libraries more aggressively for console
        console_handler.addFilter(NameFilter(excluded_names=["selenium", "urllib3", "websockets", "undetected_chromedriver"]))
        logger.addHandler(console_handler)
        logger_for_setup.debug(f"Added StreamHandler for console (Level: {log_level_upper})")
    except Exception as e:
        logger_for_setup.critical(f"Failed to create/add console handler: {e}", exc_info=True)

    # --- Suppress External Library Noise MORE AGGRESSIVELY ---
    try:
        # Set higher levels globally for these loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING) # Target the specific pool logger
        logging.getLogger("selenium.webdriver.remote.remote_connection").setLevel(logging.INFO)
        logging.getLogger("selenium").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
        logging.getLogger("undetected_chromedriver").setLevel(logging.WARNING) # Added undetected_chromedriver

        # Prevent propagation to avoid double logging if root handler exists
        logging.getLogger("urllib3").propagate = False
        logging.getLogger("selenium").propagate = False
        logging.getLogger("websockets").propagate = False
        logging.getLogger("undetected_chromedriver").propagate = False

        logger_for_setup.debug("Set external library log levels and disabled propagation.")

    except Exception as e:
        logger_for_setup.error(f"Error setting library log levels: {e}", exc_info=True)

    _logging_initialized = True
    logger_for_setup.info(f"Logging setup complete. Logger '{logger.name}' configured. Console/File Level: {log_level_upper}")

    return logger
# end setup_logging


# --- Standalone Test Block ---
if __name__ == "__main__":
    print(f"\n--- Running {__file__} standalone test ---")
    print(f"Using Log Directory: {LOG_DIRECTORY}")
    # Test with INFO level first
    main_logger = setup_logging(
        log_level="INFO", log_file="test_logging_config_sync_handler.log"
    )

    main_logger.info("--- Standalone Test Start (INFO level) ---")
    main_logger.debug("This is a DEBUG test log (should NOT appear anywhere).")
    main_logger.info("This is an INFO test log (should appear in console and file).")
    main_logger.warning(
        "This is a WARNING test log (should appear in console and file)."
    )

    # Test toggling to DEBUG
    print("\n--- Toggling Log Level to DEBUG ---")
    main_logger = setup_logging(log_level="DEBUG")  # Call again to update levels
    main_logger.info("--- Logging After Toggle to DEBUG ---")
    main_logger.debug(
        "This is a DEBUG test log (should NOW appear in console and file)."
    )
    main_logger.info("This is another INFO test log.")

    # Test toggling back to INFO
    print("\n--- Toggling Log Level back to INFO ---")
    main_logger = setup_logging(log_level="INFO")  # Call again to update levels
    main_logger.info("--- Logging After Toggle back to INFO ---")
    main_logger.debug("This is a DEBUG test log (should NOT appear anywhere again).")
    main_logger.info("This is a final INFO test log.")

    print("\n--- Standalone Test Complete ---")
    log_file_path_test = LOG_DIRECTORY / "test_logging_config_sync_handler.log"
    print(f"Check log file '{log_file_path_test.name}' content.")
    print(
        "Expected file content: INFO/WARN logs from start, then DEBUG/INFO, then final INFO logs."
    )
