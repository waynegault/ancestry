#!/usr/bin/env python3

"""
Centralized Logging Utilities.

This module provides standardized logging setup and utilities to eliminate
inconsistent logging patterns across the codebase.
"""

# === CORE INFRASTRUCTURE ===
import sys
import os

# Add parent directory to path for core_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core_imports import (
    standardize_module_imports,
    auto_register_module,
    get_logger,
)

standardize_module_imports()
auto_register_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import logging
from typing import Optional

# === MODULE LOGGER ===
logger = get_logger(__name__)

# Global flag to track if logging has been initialized
_centralized_logging_setup = False


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
    global _centralized_logging_setup

    # Try to use centralized logging config first
    try:
        from logging_config import logger as central_logger, setup_logging

        # Initialize centralized logging if not already done
        if not _centralized_logging_setup:
            setup_logging()
            _centralized_logging_setup = True

        # If no specific name requested, return the central logger
        if name is None:
            return central_logger
        else:
            # Return a child logger that inherits from central config
            return central_logger.getChild(name)

    except ImportError:
        # Fallback to standard logging if logging_config is not available
        if name is None:
            import inspect

            frame = inspect.currentframe()
            if frame and frame.f_back:
                name = frame.f_back.f_globals.get("__name__", "unknown")
            else:
                name = "unknown"

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
