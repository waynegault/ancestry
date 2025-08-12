#!/usr/bin/env python3

"""
Minimal config shim re-exporting essential config components.
"""

from config.config_manager import ConfigManager

# Create global instances like database.py does
config_manager = ConfigManager()
config_schema = config_manager.get_config()

__all__ = ["config_manager", "config_schema", "ConfigManager"]
