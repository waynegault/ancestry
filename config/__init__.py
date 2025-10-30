"""
Configuration Package - Modular Configuration System

This package provides the new modular configuration system that replaces
the legacy config.py with a robust, schema-based configuration management.

Main components:
- ConfigManager: Handles configuration loading, validation, and management
- ConfigSchema: Type-safe configuration schemas with validation
"""

import sys

# Add parent directory to path for core_imports
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .config_manager import ConfigManager
    from .config_schema import ConfigSchema
except ImportError:
    # If relative imports fail, try absolute imports
    from config_manager import ConfigManager
    from config_schema import ConfigSchema

# Create the main configuration manager instance
config_manager = ConfigManager()

# Load the configuration
config_schema = config_manager.get_config()

# Export the configuration instances
__all__ = [
    "ConfigManager",
    "ConfigSchema",
    "config_manager",
    "config_schema",
]
