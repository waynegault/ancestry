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


# ---------------------------------------------------------------------------
# Module Tests
# ---------------------------------------------------------------------------

from test_framework import TestSuite
from test_utilities import create_standard_test_runner


def _test_config_manager_instance() -> bool:
    """Ensure the exported config_manager is a ConfigManager and returns cached configs."""

    assert isinstance(config_manager, ConfigManager), "config_manager should be a ConfigManager instance"
    second_call = config_manager.get_config()
    assert second_call is config_schema, "get_config() should return the exported config_schema"
    return True


def _test_config_schema_instance() -> bool:
    """Verify config_schema is materialized and exposes expected attributes."""

    assert isinstance(config_schema, ConfigSchema), "config_schema should be a ConfigSchema instance"
    assert hasattr(config_schema, "api"), "config_schema should expose the api section"
    assert hasattr(config_schema, "app"), "config_schema should expose the app section"
    return True


def _test_all_exports() -> bool:
    """Confirm __all__ tracks the public contract for downstream imports."""

    expected_names = {"ConfigManager", "ConfigSchema", "config_manager", "config_schema"}
    assert set(__all__) == expected_names, "__all__ should match documented exports"
    return True


def module_tests() -> bool:
    suite = TestSuite("config package exports", "config/__init__.py")

    suite.run_test(
        "ConfigManager singleton",
        _test_config_manager_instance,
        "Ensures config_manager is a ConfigManager instance and returns the cached schema.",
    )

    suite.run_test(
        "ConfigSchema instance",
        _test_config_schema_instance,
        "Ensures config_schema materializes and exposes expected sections.",
    )

    suite.run_test(
        "Export contract",
        _test_all_exports,
        "Ensures __all__ stays aligned with documented exports.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    sys.exit(0 if success else 1)
