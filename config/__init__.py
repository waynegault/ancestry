"""
Configuration Package - Modular Configuration System

This package provides the new modular configuration system that replaces
the legacy config.py with a robust, schema-based configuration management.

Main components:
- ConfigManager: Handles configuration loading, validation, and management
- ConfigSchema: Type-safe configuration schemas with validation
- CredentialManager: Secure credential management integration
"""

import os
import sys

# Add parent directory to path for core_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .config_manager import ConfigManager
    from .config_schema import ConfigSchema
    from .credential_manager import CredentialManager
except ImportError:
    # If relative imports fail, try absolute imports
    from config_manager import ConfigManager
    from config_schema import ConfigSchema
    from credential_manager import CredentialManager

# Create the main configuration manager instance
config_manager = ConfigManager()

# Load the configuration
config_schema = config_manager.get_config()

# Export the configuration instances
__all__ = [
    "ConfigManager",
    "ConfigSchema",
    "CredentialManager",
    "config_manager",
    "config_schema",
]
