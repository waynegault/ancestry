#!/usr/bin/env python3

"""
Minimal config shim re-exporting essential config components.
"""

from config.config_manager import ConfigManager

# Create global instances like database.py does
config_manager = ConfigManager()
config_schema = config_manager.get_config()

__all__ = ["config_manager", "config_schema", "ConfigManager"]


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for config.py (minimal shim module).
    
    This module is a lightweight shim that re-exports configuration components.
    Tests focus on import availability and basic functionality.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        from test_framework import TestSuite
        
        suite = TestSuite("Config Shim Comprehensive Tests", __name__)
        suite.start_suite()
        
        def test_config_manager_import():
            """Test ConfigManager import availability"""
            try:
                assert ConfigManager is not None
                assert hasattr(ConfigManager, '__init__')
                return True
            except Exception:
                return False
        
        def test_global_instances():
            """Test global config instances creation"""
            try:
                assert config_manager is not None
                assert config_schema is not None
                assert isinstance(config_manager, ConfigManager)
                return True
            except Exception:
                return False
        
        def test_config_manager_functionality():
            """Test basic ConfigManager functionality"""
            try:
                # Test that config_manager has expected methods
                assert hasattr(config_manager, 'get_config')
                assert callable(config_manager.get_config)
                
                # Test that get_config returns something
                config = config_manager.get_config()
                assert config is not None
                
                return True
            except Exception:
                return False
        
        def test_module_exports():
            """Test module __all__ exports"""
            try:
                expected_exports = ["config_manager", "config_schema", "ConfigManager"]
                assert __all__ == expected_exports
                
                # Test that all exported items are accessible
                import sys
                current_module = sys.modules[__name__]
                for export_name in __all__:
                    assert hasattr(current_module, export_name)
                    assert getattr(current_module, export_name) is not None
                
                return True
            except Exception:
                return False
        
        def test_config_schema_validity():
            """Test config schema is valid"""
            try:
                # config_schema should be the result of get_config()
                assert config_schema is not None
                
                # Should be some kind of configuration object/dict
                # At minimum, should be iterable or have attributes
                try:
                    # Try dict-like access or attribute access
                    assert hasattr(config_schema, '__dict__') or hasattr(config_schema, 'keys')
                except:
                    # If neither, at least should be a non-empty object
                    assert str(config_schema) != ''
                
                return True
            except Exception:
                return False
        
        # Run all tests
        suite.run_test(
            "ConfigManager Import",
            test_config_manager_import,
            "ConfigManager class should be available for import and instantiation",
            "ConfigManager provides configuration management functionality",
            "Test ConfigManager class availability and basic structure"
        )
        
        suite.run_test(
            "Global Instances Creation", 
            test_global_instances,
            "Global config_manager and config_schema instances should be created successfully",
            "Global instances provide immediate access to configuration functionality",
            "Test creation of global config_manager and config_schema instances"
        )
        
        suite.run_test(
            "ConfigManager Functionality",
            test_config_manager_functionality, 
            "ConfigManager should have working get_config method functionality",
            "ConfigManager provides configuration retrieval capabilities",
            "Test ConfigManager method availability and basic functionality"
        )
        
        suite.run_test(
            "Module Exports",
            test_module_exports,
            "Module should properly export all items listed in __all__",
            "Module exports provide controlled access to configuration components", 
            "Test __all__ exports and accessibility of exported items"
        )
        
        suite.run_test(
            "Config Schema Validity",
            test_config_schema_validity,
            "Config schema should be valid configuration object",
            "Config schema provides structured configuration data access",
            "Test config_schema validity and basic structure"
        )
        
        return suite.finish_suite()
        
    except ImportError:
        print("Warning: TestSuite not available, running basic validation...")
        
        # Basic fallback tests
        try:
            assert ConfigManager is not None
            assert config_manager is not None  
            assert config_schema is not None
            assert callable(config_manager.get_config)
            print("✅ Basic config.py validation passed")
            return True
        except Exception as e:
            print(f"❌ Basic config.py validation failed: {e}")
            return False


if __name__ == "__main__":
    import sys
    print("🧪 Running Config Shim Comprehensive Tests...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
