"""
Legacy Configuration Compatibility Layer

This module provides backward compatibility for the legacy configuration system
while the codebase transitions to the new modular configuration architecture.
"""

import os
import sys
import importlib.util
from typing import Dict, Any, Optional

# Add the parent directory to the path to access config.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    # Import from the main config.py file in the parent directory
    import importlib.util

    config_path = os.path.join(parent_dir, "config.py")
    spec = importlib.util.spec_from_file_location("main_config", config_path)
    if spec and spec.loader:
        main_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_config)
        Config_Class = main_config.Config_Class
        SeleniumConfig = getattr(main_config, "SeleniumConfig", None)
        _config_available = True
    else:
        raise ImportError("Could not create module spec")
except Exception as e:
    print(f"Warning: Could not import Config_Class: {e}")
    _config_available = False
    Config_Class = None
    SeleniumConfig = None

# Try to import ConfigManager for new architecture compatibility
try:
    from .config_manager import ConfigManager

    _config_manager_available = True
except Exception as e:
    print(f"Warning: Could not import ConfigManager: {e}")
    _config_manager_available = False
    ConfigManager = None


class LegacyConfigInstance:
    """
    Legacy configuration instance that provides backward compatibility
    for modules still using the old configuration system.
    """

    def __init__(self):
        """Initialize the legacy configuration instance."""
        self._config = None
        self._initialize_config()

    def _initialize_config(self):
        """Initialize the configuration system."""
        if _config_available and Config_Class:
            try:
                self._config = Config_Class()
            except Exception as e:
                print(f"Warning: Could not initialize Config_Class: {e}")
                self._config = None
        else:
            print("Warning: Config_Class not available, using fallback configuration")

    def _get_config_value(self, attr_name: str, default_value: Any) -> Any:
        """
        Get a configuration value with fallback to default.

        Args:
            attr_name: The attribute name to retrieve
            default_value: The default value if not found

        Returns:
            The configuration value or default
        """
        if self._config:
            try:
                value = getattr(self._config, attr_name, None)
                if value is not None:
                    return value
            except Exception:
                pass
        return default_value

    # Configuration attributes with proper defaults
    @property
    def MAX_INBOX(self) -> int:
        """Maximum inbox messages to process (0 = unlimited)."""
        return self._get_config_value("MAX_INBOX", 0)

    @property
    def BATCH_SIZE(self) -> int:
        """Batch size for processing operations."""
        return self._get_config_value("BATCH_SIZE", 50)

    @property
    def AI_CONTEXT_MESSAGES_COUNT(self) -> int:
        """Number of context messages for AI processing."""
        return self._get_config_value("AI_CONTEXT_MESSAGES_COUNT", 7)

    @property
    def AI_CONTEXT_MESSAGE_MAX_WORDS(self) -> int:
        """Maximum words per AI context message."""
        return self._get_config_value("AI_CONTEXT_MESSAGE_MAX_WORDS", 500)

    @property
    def MESSAGE_TRUNCATION_LENGTH(self) -> int:
        """Maximum length for message truncation."""
        return self._get_config_value("MESSAGE_TRUNCATION_LENGTH", 300)

    @property
    def API_CONTEXTUAL_HEADERS(self) -> Dict[str, str]:
        """API contextual headers configuration."""
        return self._get_config_value("API_CONTEXTUAL_HEADERS", {})

    # Additional commonly used configuration attributes
    @property
    def DEBUG(self) -> bool:
        """Debug mode flag."""
        return self._get_config_value("DEBUG", False)

    @property
    def LOGGING_LEVEL(self) -> str:
        """Logging level."""
        return self._get_config_value("LOGGING_LEVEL", "INFO")

    @property
    def DATABASE_PATH(self) -> str:
        """Database file path."""
        return self._get_config_value("DATABASE_PATH", "ancestry.db") @ property

    def CACHE_ENABLED(self) -> bool:
        """Cache enabled flag."""
        return self._get_config_value("CACHE_ENABLED", True)

    def __getattr__(self, name: str) -> Any:
        """
        Fallback for any other configuration attributes.

        Args:
            name: The attribute name

        Returns:
            The configuration value or raises AttributeError
        """
        if self._config:
            try:
                return getattr(self._config, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class MinimalConfig:
    """
    Minimal fallback configuration for when the main config system is unavailable.
    """

    # Core configuration values with safe defaults
    MAX_INBOX = 0
    BATCH_SIZE = 50
    AI_CONTEXT_MESSAGES_COUNT = 7
    AI_CONTEXT_MESSAGE_MAX_WORDS = 500
    MESSAGE_TRUNCATION_LENGTH = 300
    API_CONTEXTUAL_HEADERS = {}
    DEBUG = False
    LOGGING_LEVEL = "INFO"
    DATABASE_PATH = "ancestry.db"
    CACHE_ENABLED = True

    def __getattr__(self, name: str) -> Any:
        """Return None for any undefined attributes."""
        return None


# Create the global configuration instance
try:
    config_instance = LegacyConfigInstance()
    # Test if the instance works by accessing a key attribute
    _ = config_instance.BATCH_SIZE
except Exception as e:
    print(f"Warning: LegacyConfigInstance failed ({e}), using MinimalConfig")
    config_instance = MinimalConfig()

# Create the selenium configuration instance
try:
    if _config_available and SeleniumConfig:
        selenium_config = SeleniumConfig()
    else:
        # Create a minimal selenium config fallback
        selenium_config = MinimalConfig()
except Exception as e:
    print(f"Warning: SeleniumConfig failed ({e}), using MinimalConfig")
    selenium_config = MinimalConfig()

# Export the configuration instance for backward compatibility
__all__ = [
    "config_instance",
    "selenium_config",
    "ConfigManager",
    "LegacyConfigInstance",
    "MinimalConfig",
]
