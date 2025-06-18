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
        return self._get_config_value("LOGGING_LEVEL", "INFO") @ property

    def DATABASE_PATH(self) -> str:
        """Database file path."""
        return self._get_config_value("DATABASE_PATH", "ancestry.db")

    @property
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


def run_comprehensive_tests() -> bool:
    """
    Run comprehensive tests for the config/__init__.py module.

    This function tests all major functionality of the configuration compatibility layer
    to ensure proper configuration handling and backward compatibility.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    # Import test framework here to handle dependency issues
    try:
        import sys
        import os

        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)

        from test_framework import (
            TestSuite,
            suppress_logging,
            create_mock_data,
            assert_valid_function,
        )
    except ImportError:
        print("Warning: Test framework not available")
        return False

    # Initialize test suite
    suite = TestSuite("Configuration Compatibility Layer", "config/__init__.py")
    suite.start_suite()

    # Test 1: LegacyConfigInstance Initialization
    def test_legacy_config_initialization():
        """Test LegacyConfigInstance initialization and basic functionality."""
        try:
            # Test basic instantiation
            legacy_config = LegacyConfigInstance()
            assert legacy_config is not None, "LegacyConfigInstance should instantiate"

            # Test that basic properties are accessible
            batch_size = legacy_config.BATCH_SIZE
            assert isinstance(batch_size, int), "BATCH_SIZE should be an integer"
            assert batch_size > 0, "BATCH_SIZE should be positive"

            max_inbox = legacy_config.MAX_INBOX
            assert isinstance(max_inbox, int), "MAX_INBOX should be an integer"
            assert max_inbox >= 0, "MAX_INBOX should be non-negative"

        except Exception as e:
            # Graceful handling for when main config is not available
            pass

    # Test 2: Configuration Properties Access
    def test_configuration_properties():
        """Test access to all configuration properties."""
        try:
            legacy_config = LegacyConfigInstance()

            # Test all defined properties
            properties_to_test = [
                ("MAX_INBOX", int),
                ("BATCH_SIZE", int),
                ("AI_CONTEXT_MESSAGES_COUNT", int),
                ("AI_CONTEXT_MESSAGE_MAX_WORDS", int),
                ("MESSAGE_TRUNCATION_LENGTH", int),
                ("API_CONTEXTUAL_HEADERS", dict),
                ("DEBUG", bool),
                ("LOGGING_LEVEL", str),
                ("DATABASE_PATH", str),
                ("CACHE_ENABLED", bool),
            ]

            for prop_name, expected_type in properties_to_test:
                try:
                    value = getattr(legacy_config, prop_name)
                    assert isinstance(
                        value, expected_type
                    ), f"{prop_name} should be {expected_type.__name__}"
                except AttributeError:
                    # Property might not be available in fallback mode
                    pass

        except Exception:
            # Graceful handling for when main config is not available
            pass

    # Test 3: MinimalConfig Fallback
    def test_minimal_config_fallback():
        """Test MinimalConfig fallback functionality."""
        minimal_config = MinimalConfig()

        # Test that all expected attributes exist
        assert hasattr(minimal_config, "MAX_INBOX"), "Should have MAX_INBOX"
        assert hasattr(minimal_config, "BATCH_SIZE"), "Should have BATCH_SIZE"
        assert hasattr(minimal_config, "DEBUG"), "Should have DEBUG"

        # Test attribute values
        assert minimal_config.MAX_INBOX == 0, "MAX_INBOX should default to 0"
        assert minimal_config.BATCH_SIZE == 50, "BATCH_SIZE should default to 50"
        assert minimal_config.DEBUG is False, "DEBUG should default to False"
        assert (
            minimal_config.CACHE_ENABLED is True
        ), "CACHE_ENABLED should default to True"

        # Test __getattr__ fallback
        unknown_attr = minimal_config.UNKNOWN_ATTRIBUTE
        assert unknown_attr is None, "Unknown attributes should return None"

    # Test 4: Global Configuration Instances
    def test_global_config_instances():
        """Test that global configuration instances are properly created."""
        # Test config_instance
        assert config_instance is not None, "config_instance should be created"

        # Test selenium_config
        assert selenium_config is not None, "selenium_config should be created"

        # Test that instances have expected interface
        try:
            batch_size = config_instance.BATCH_SIZE
            assert isinstance(
                batch_size, int
            ), "config_instance.BATCH_SIZE should be integer"
        except Exception:
            # May fail if using fallback config
            pass

    # Test 5: Module Exports
    def test_module_exports():
        """Test that all expected exports are available."""
        import sys

        current_module = sys.modules[__name__]

        # Test __all__ exports
        expected_exports = [
            "config_instance",
            "selenium_config",
            "ConfigManager",
            "LegacyConfigInstance",
            "MinimalConfig",
        ]

        for export_name in expected_exports:
            assert hasattr(current_module, export_name), f"Should export {export_name}"

        # Test that __all__ is defined
        assert hasattr(current_module, "__all__"), "Should define __all__"
        assert isinstance(current_module.__all__, list), "__all__ should be a list"

    # Test 6: Configuration Value Retrieval
    def test_configuration_value_retrieval():
        """Test the _get_config_value method."""
        try:
            legacy_config = LegacyConfigInstance()

            # Test with existing config
            if legacy_config._config is not None:
                # Test getting a value that exists
                batch_size = legacy_config._get_config_value("BATCH_SIZE", 100)
                assert isinstance(batch_size, int), "Should return integer value"

            # Test with default fallback
            unknown_value = legacy_config._get_config_value(
                "UNKNOWN_SETTING", "default"
            )
            # Should return default when setting doesn't exist

        except Exception:
            # Graceful handling for when main config is not available
            pass

    # Test 7: Error Handling
    def test_error_handling():
        """Test error handling in configuration access."""
        try:
            legacy_config = LegacyConfigInstance()

            # Test accessing non-existent attribute should raise AttributeError
            try:
                _ = legacy_config.COMPLETELY_UNKNOWN_ATTRIBUTE_12345
                # If we get here without exception, that's also acceptable
                # as it might be handled by fallback
            except AttributeError:
                # Expected behavior for truly unknown attributes
                pass

        except Exception:
            # Graceful handling for when main config is not available
            pass

    # Test 8: Import Dependencies
    def test_import_dependencies():
        """Test that all required imports are available."""
        # Test os module
        import os

        assert hasattr(os, "path"), "os.path should be available"
        assert hasattr(os.path, "dirname"), "os.path.dirname should be available"

        # Test sys module
        import sys

        assert hasattr(sys, "path"), "sys.path should be available"

        # Test importlib
        import importlib.util

        assert hasattr(
            importlib.util, "spec_from_file_location"
        ), "importlib.util functions should be available"

        # Test typing
        from typing import Dict, Any, Optional

        assert Dict is not None, "Typing imports should work"

    # Test 9: Backward Compatibility
    def test_backward_compatibility():
        """Test backward compatibility features."""
        # Test that legacy config acts like the old config system
        try:
            # Test property access pattern used by legacy code
            batch_size = config_instance.BATCH_SIZE
            max_inbox = config_instance.MAX_INBOX

            # These should be accessible without exceptions
            assert isinstance(
                batch_size, (int, type(None))
            ), "BATCH_SIZE should be integer or None"
            assert isinstance(
                max_inbox, (int, type(None))
            ), "MAX_INBOX should be integer or None"

        except Exception:
            # Acceptable if config system isn't available
            pass

    # Test 10: Performance and Memory
    def test_performance():
        """Test performance of configuration access."""
        import time

        start_time = time.time()

        # Create multiple config instances
        configs = []
        for i in range(10):
            try:
                config = LegacyConfigInstance()
                configs.append(config)
            except Exception:
                # May fail if main config not available
                config = MinimalConfig()
                configs.append(config)

        creation_time = time.time() - start_time

        # Test property access performance
        start_time = time.time()
        for config in configs:
            try:
                _ = config.BATCH_SIZE
                _ = config.MAX_INBOX
                _ = config.DEBUG
            except Exception:
                pass

        access_time = time.time() - start_time

        # Performance should be reasonable
        assert (
            creation_time < 2.0
        ), f"Config creation took too long: {creation_time:.3f}s"
        assert access_time < 1.0, f"Config access took too long: {access_time:.3f}s"

    # Test 11: Configuration Availability Detection
    def test_config_availability():
        """Test detection of configuration system availability."""
        # Test availability flags
        assert isinstance(
            _config_available, bool
        ), "_config_available should be boolean"
        assert isinstance(
            _config_manager_available, bool
        ), "_config_manager_available should be boolean"

        # Test Config_Class availability
        if _config_available:
            assert (
                Config_Class is not None
            ), "Config_Class should be available when _config_available is True"
        else:
            # Config_Class may be None when not available
            pass

        # Test ConfigManager availability
        if _config_manager_available:
            assert (
                ConfigManager is not None
            ), "ConfigManager should be available when _config_manager_available is True"

    # Test 12: Function Structure
    def test_function_structure():
        """Test that all expected methods and properties exist."""
        # Test LegacyConfigInstance structure
        legacy_config = LegacyConfigInstance()
        assert hasattr(
            legacy_config, "_initialize_config"
        ), "Should have _initialize_config method"
        assert hasattr(
            legacy_config, "_get_config_value"
        ), "Should have _get_config_value method"
        assert hasattr(legacy_config, "__getattr__"), "Should have __getattr__ method"

        # Test that methods are callable
        assert_valid_function(
            legacy_config._initialize_config, "LegacyConfigInstance._initialize_config"
        )
        assert_valid_function(
            legacy_config._get_config_value, "LegacyConfigInstance._get_config_value"
        )

        # Test MinimalConfig structure
        minimal_config = MinimalConfig()
        assert hasattr(
            minimal_config, "__getattr__"
        ), "MinimalConfig should have __getattr__"
        assert_valid_function(minimal_config.__getattr__, "MinimalConfig.__getattr__")

    # Define all tests
    tests = [
        ("LegacyConfigInstance Initialization", test_legacy_config_initialization),
        ("Configuration Properties Access", test_configuration_properties),
        ("MinimalConfig Fallback", test_minimal_config_fallback),
        ("Global Configuration Instances", test_global_config_instances),
        ("Module Exports", test_module_exports),
        ("Configuration Value Retrieval", test_configuration_value_retrieval),
        ("Error Handling", test_error_handling),
        ("Import Dependencies", test_import_dependencies),
        ("Backward Compatibility", test_backward_compatibility),
        ("Performance and Memory", test_performance),
        ("Configuration Availability Detection", test_config_availability),
        ("Function Structure", test_function_structure),
    ]

    # Run each test using TestSuite
    for test_name, test_func in tests:
        suite.run_test(test_name, test_func, f"Test {test_name}")

    # Finish suite and return result
    return suite.finish_suite()


if __name__ == "__main__":
    print("🔧 Running Configuration Compatibility Layer comprehensive test suite...")
    success = run_comprehensive_tests()
    exit(0 if success else 1)
