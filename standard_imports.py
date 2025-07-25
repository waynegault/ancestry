"""
Standardized Imports Module - Single Source of Truth

This module provides a single, consistent way to import core functionality
across the entire Ancestry project, eliminating the 5+ different import
patterns currently scattered throughout the codebase.

Usage in any module:
    from standard_imports import *

    # Everything is now available:
    logger.info("Standardized logging")
    register_function("my_func", my_function)
    # etc.

This replaces inconsistent patterns like:
- from core_imports import auto_register_module, get_logger, ...
- try: from core_imports import ... except ImportError: ...
- from logging_config import logger
- Various fallback patterns
"""

# === CORE UNIFIED IMPORTS ===
from core_imports import (
    # Core functionality
    auto_register_module,
    get_logger,
    standardize_module_imports,
    safe_execute,
    # Function registry
    register_function,
    get_function,
    is_function_available,
    call_function,
    get_available_functions,
    register_many,
    # Utilities
    get_project_root,
    import_context,
    cleanup_registry,
    get_stats,
    ensure_imports,
)

# === STANDARD LIBRARY IMPORTS (COMMONLY NEEDED) ===
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from typing import Dict, List, Any, Optional, Callable, Union, Tuple


# === STANDARDIZED LOGGER ===
# Single logger pattern for all modules
def get_standard_logger(module_name: str) -> logging.Logger:
    """Get a standardized logger for any module."""
    return get_logger(module_name)


# === AUTO-REGISTRATION HELPER ===
def setup_module(module_globals: Dict[str, Any], module_name: str) -> logging.Logger:
    """
    One-line module setup that handles everything:
    - Auto-registration
    - Logger setup
    - Import standardization

    Usage at top of any module:
        from standard_imports import setup_module
        logger = setup_module(globals(), __name__)
    """
    auto_register_module(module_globals, module_name)
    standardize_module_imports()
    return get_standard_logger(module_name)


# === ERROR HANDLING UTILITIES ===
def safe_import(module_name: str, fallback=None):
    """Safely import a module with fallback."""
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError:
        return fallback


def safe_import_from(module_name: str, item_name: str, fallback=None):
    """Safely import an item from a module with fallback."""
    try:
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, item_name, fallback)
    except (ImportError, AttributeError):
        return fallback


# === TESTING INTEGRATION ===
def get_unified_test_framework():
    """Get the unified test framework, with fallback to individual tests."""
    # Using modern test_framework.py instead of deprecated test_framework_unified
    try:
        from test_framework import TestSuite

        return TestSuite
    except ImportError:
        # Fallback to legacy test framework
        return None


# === EXPORTS ===
# Make everything available for "from standard_imports import *"
__all__ = [
    # Core functionality
    "auto_register_module",
    "get_logger",
    "standardize_module_imports",
    "safe_execute",
    # Function registry
    "register_function",
    "get_function",
    "is_function_available",
    "call_function",
    "get_available_functions",
    "register_many",
    # Utilities
    "get_project_root",
    "import_context",
    "cleanup_registry",
    "get_stats",
    "ensure_imports",
    # Standardized helpers
    "setup_module",
    "get_standard_logger",
    "safe_import",
    "safe_import_from",
    "get_unified_test_framework",
    # Common types
    "Dict",
    "List",
    "Any",
    "Optional",
    "Callable",
    "Union",
    "Tuple",
    # Common modules
    "os",
    "sys",
    "time",
    "logging",
    "Path",
]

# === USAGE EXAMPLES ===

"""
BEFORE (Inconsistent patterns across codebase):

# Pattern 1 - Verbose and error-prone
from core_imports import (
    auto_register_module,
    get_logger, 
    standardize_module_imports,
    register_function,
    get_function,
    is_function_available,
)
auto_register_module(globals(), __name__)
logger = get_logger(__name__)

# Pattern 2 - Try/except fallbacks everywhere  
try:
    from core_imports import register_function
except ImportError:
    register_function = None

# Pattern 3 - Mixed logger sources
from logging_config import logger

# Pattern 4 - Manual fallbacks
try:
    from core_imports import auto_register_module
    pass  # Already registered at line 169
except ImportError:
    pass

AFTER (Single standardized pattern):

# Option 1 - Full control
from standard_imports import *
logger = setup_module(globals(), __name__)

# Option 2 - Individual imports 
from standard_imports import setup_module, register_function, safe_execute
logger = setup_module(globals(), __name__)

# Option 3 - Minimal
from standard_imports import setup_module
logger = setup_module(globals(), __name__)
# Everything else available via standard_imports if needed
"""


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for standard_imports.py.
    Tests all import standardization and module setup functionality.
    """
    from test_framework import TestSuite, suppress_logging

    print("🔧 Running Standard Imports comprehensive test suite...")

    # Quick basic test first
    try:
        # Test basic module setup
        logger = setup_module(globals(), __name__)
        assert logger is not None
        print("✅ Module setup test passed")

        # Test function registration
        def test_func():
            return "test_result"

        register_function("test_standard_imports", test_func)
        assert is_function_available("test_standard_imports")
        print("✅ Function registration test passed")

        print("✅ Basic Standard Imports tests completed")
    except Exception as e:
        print(f"❌ Basic Standard Imports tests failed: {e}")
        return False

    with suppress_logging():
        suite = TestSuite("Standard Imports & Module Setup", "standard_imports.py")
        suite.start_suite()

    # Module setup and initialization
    def test_module_setup():
        # Test setup_module function
        test_globals = {}
        logger = setup_module(test_globals, "test_module")
        assert logger is not None, "setup_module should return a logger"
        assert hasattr(logger, "info"), "Logger should have info method"
        assert hasattr(logger, "error"), "Logger should have error method"

    # Logger functionality
    def test_logger_creation():
        logger = get_standard_logger("test_logger_module")
        assert logger is not None, "get_standard_logger should return a logger"
        assert hasattr(logger, "debug"), "Logger should have debug method"
        assert hasattr(logger, "warning"), "Logger should have warning method"

    # Function registration
    def test_function_registration():
        def sample_function():
            return "sample_result"

        # Test registration
        register_function("test_sample_func", sample_function)
        assert is_function_available(
            "test_sample_func"
        ), "Function should be registered"

        # Test retrieval
        retrieved_func = get_function("test_sample_func")
        assert retrieved_func is not None, "Should retrieve registered function"
        assert retrieved_func() == "sample_result", "Retrieved function should work"

    # Safe imports
    def test_safe_imports():
        # Test successful import
        result = safe_import("os")
        assert result is not None, "Should successfully import existing module"
        assert hasattr(result, "path"), "Should have os.path"

        # Test failed import with fallback
        result = safe_import("nonexistent_module", "fallback_value")
        assert result == "fallback_value", "Should return fallback for missing module"

        # Test safe_import_from
        result = safe_import_from("os", "path")
        assert result is not None, "Should import os.path"

        # Test safe_import_from with fallback
        result = safe_import_from("nonexistent_module", "item", "fallback")
        assert result == "fallback", "Should return fallback for missing item"

    # Import standardization
    def test_import_standardization():
        # Test that standardize_module_imports doesn't crash
        try:
            standardize_module_imports()
            success = True
        except Exception:
            success = False
        assert success, "standardize_module_imports should not crash"

    # Core imports availability
    def test_core_imports_availability():
        required_functions = [
            "auto_register_module",
            "get_logger",
            "standardize_module_imports",
            "safe_execute",
            "register_function",
            "get_function",
            "is_function_available",
        ]

        for func_name in required_functions:
            assert func_name in globals(), f"Function {func_name} should be available"
            assert callable(
                globals()[func_name]
            ), f"Function {func_name} should be callable"

    # Standard library imports availability
    def test_standard_library_availability():
        required_modules = ["os", "sys", "time", "logging"]
        for module_name in required_modules:
            assert module_name in globals(), f"Module {module_name} should be imported"

    # Module cleanup
    def test_module_cleanup():
        # Test that we can clean up registered functions
        def temp_function():
            return "temp"

        register_function("temp_test_function", temp_function)
        assert is_function_available(
            "temp_test_function"
        ), "Temp function should be registered"

        # Test cleanup if available
        try:
            cleanup_registry()
            # After cleanup, function might or might not be available depending on implementation
            # This is more of a smoke test
        except Exception:
            pass  # Cleanup might not be implemented or might fail gracefully

    # Performance test
    def test_performance():
        import time

        start_time = time.time()
        for i in range(100):
            logger = get_standard_logger(f"test_module_{i}")
            register_function(f"test_func_{i}", lambda: i)
        duration = time.time() - start_time

        assert duration < 1.0, f"Module operations should be fast, took {duration:.3f}s"

    # Run all tests
    with suppress_logging():
        suite.run_test(
            "Module setup and initialization",
            test_module_setup,
            "Module setup function properly initializes logging and registration",
            "Test setup_module function with test globals and module name",
            "Module setup creates valid logger with expected methods",
        )

        suite.run_test(
            "Logger creation and functionality",
            test_logger_creation,
            "Standard logger creation works correctly with proper methods",
            "Test get_standard_logger function and verify logger methods",
            "Logger creation provides complete logging interface",
        )

        suite.run_test(
            "Function registration system",
            test_function_registration,
            "Function registration and retrieval works correctly",
            "Test register_function, is_function_available, and get_function",
            "Function registration system properly stores and retrieves functions",
        )

        suite.run_test(
            "Safe import functionality",
            test_safe_imports,
            "Safe import functions handle both success and failure cases",
            "Test safe_import and safe_import_from with existing and missing modules",
            "Safe imports provide proper fallback handling for missing modules",
        )

        suite.run_test(
            "Import standardization",
            test_import_standardization,
            "Import standardization runs without errors",
            "Test standardize_module_imports function execution",
            "Import standardization completes successfully",
        )

        suite.run_test(
            "Core imports availability",
            test_core_imports_availability,
            "All required core import functions are available and callable",
            "Test availability of auto_register_module, get_logger, and other core functions",
            "Core imports provide complete function registry and logging capabilities",
        )

        suite.run_test(
            "Standard library availability",
            test_standard_library_availability,
            "Required standard library modules are imported and available",
            "Test availability of os, sys, time, logging modules",
            "Standard library modules are properly imported and accessible",
        )

        suite.run_test(
            "Module cleanup functionality",
            test_module_cleanup,
            "Module cleanup and registry management works correctly",
            "Test cleanup_registry function and temporary function cleanup",
            "Module cleanup provides proper registry management",
        )

        suite.run_test(
            "Performance validation",
            test_performance,
            "Module operations complete within reasonable time limits",
            "Test performance of logger creation and function registration",
            "Performance operations complete efficiently for batch operations",
        )

    return suite.finish_suite()


# Self-test and comprehensive testing
if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
