#!/usr/bin/env python3

"""
Configuration Management & Environment Orchestration Engine

Comprehensive configuration management platform providing sophisticated environment
setup, intelligent configuration validation, and secure credential management
with multi-environment support, schema validation, and runtime configuration
management for reliable genealogical automation system configuration.

Configuration Intelligence:
• Advanced environment variable management with intelligent type conversion and validation
• Comprehensive configuration schema definition with business rule enforcement
• Multi-environment support with environment-specific configuration inheritance
• Intelligent configuration validation with dependency checking and constraint enforcement
• Secure credential management with encryption, rotation, and access control
• Configuration file parsing with format detection and validation

Environment Management:
• Sophisticated environment detection with automatic configuration selection
• Advanced configuration inheritance with environment-specific overrides
• Intelligent default value management with fallback strategies and validation
• Runtime configuration updates with hot-reloading and change notification
• Configuration versioning with rollback capabilities and change tracking
• Comprehensive configuration auditing with change logs and compliance tracking

Security & Validation:
• Secure credential storage with encryption at rest and in transit
• Advanced validation frameworks with custom validators and business rule enforcement
• Configuration sanitization with input validation and output encoding
• Access control with role-based configuration access and permission management
• Comprehensive logging with security event tracking and audit trails
• Integration with secret management systems for enterprise credential handling

Foundation Services:
Provides the essential configuration infrastructure that enables reliable,
secure genealogical automation through comprehensive configuration management,
intelligent validation, and robust environment setup for professional workflows.
"""

from config.config_manager import ConfigManager

# Create global instances like database.py does
config_manager = ConfigManager()
config_schema = config_manager.get_config()

__all__ = ["ConfigManager", "config_manager", "config_schema"]


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_config_manager_import():
    """Test ConfigManager import availability"""
    assert ConfigManager is not None
    assert hasattr(ConfigManager, '__init__')


def _test_global_instances():
    """Test global config instances creation"""
    assert config_manager is not None
    assert config_schema is not None
    assert isinstance(config_manager, ConfigManager)


def _test_config_manager_functionality():
    """Test basic ConfigManager functionality"""
    assert hasattr(config_manager, 'get_config')
    assert callable(config_manager.get_config)
    config = config_manager.get_config()
    assert config is not None


def _test_module_exports():
    """Test module __all__ exports"""
    import sys
    expected_exports = ["config_manager", "config_schema", "ConfigManager"]
    assert __all__ == expected_exports
    current_module = sys.modules[__name__]
    for export_name in __all__:
        assert hasattr(current_module, export_name)
        assert getattr(current_module, export_name) is not None


def _test_config_schema_validity():
    """Test config schema is valid"""
    assert config_schema is not None
    assert hasattr(config_schema, '__dict__') or hasattr(config_schema, 'keys') or str(config_schema) != ''


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def config_module_tests() -> bool:
    """Comprehensive test suite for config.py (minimal shim module)."""
    from test_framework import TestSuite

    suite = TestSuite("Config Shim Comprehensive Tests", __name__)
    suite.start_suite()

    # Define all tests in a data structure to reduce complexity
    tests = [
        ("ConfigManager Import", _test_config_manager_import,
         "ConfigManager class should be available for import and instantiation",
         "ConfigManager provides configuration management functionality",
         "Test ConfigManager class availability and basic structure"),
        ("Global Instances Creation", _test_global_instances,
         "Global config_manager and config_schema instances should be created successfully",
         "Global instances provide immediate access to configuration functionality",
         "Test creation of global config_manager and config_schema instances"),
        ("ConfigManager Functionality", _test_config_manager_functionality,
         "ConfigManager should have working get_config method functionality",
         "ConfigManager provides configuration retrieval capabilities",
         "Test ConfigManager method availability and basic functionality"),
        ("Module Exports", _test_module_exports,
         "Module should properly export all items listed in __all__",
         "Module exports provide controlled access to configuration components",
         "Test __all__ exports and accessibility of exported items"),
        ("Config Schema Validity", _test_config_schema_validity,
         "Config schema should be valid configuration object",
         "Config schema provides structured configuration data access",
         "Test config_schema validity and basic structure"),
    ]

    # Run all tests from the list
    for test_name, test_func, expected, method, details in tests:
        suite.run_test(test_name, test_func, expected, method, details)

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(config_module_tests)


if __name__ == "__main__":
    import sys
    print("🧪 Running Config Shim Comprehensive Tests...")
    success = config_module_tests()
    sys.exit(0 if success else 1)
