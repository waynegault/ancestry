# Allows running `python -m core` without error.
# You can add test or main logic here if needed.

import sys

# Add parent directory to path for core_imports
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ==============================================
# Comprehensive Test Suite
# ==============================================


def _test_core_package_imports() -> bool:
    """Test that core package modules can be imported."""
    try:
        import core.api_manager
        import core.browser_manager
        import core.cancellation
        import core.database_manager
        import core.error_handling
        import core.progress_indicators
        import core.session_manager

        # Verify modules are loaded
        modules = [
            core.api_manager,
            core.browser_manager,
            core.cancellation,
            core.database_manager,
            core.error_handling,
            core.progress_indicators,
            core.session_manager
        ]
        assert all(m is not None for m in modules), "All modules should be loaded"
        return True
    except ImportError as e:
        print(f"Failed to import core modules: {e}")
        return False


def _test_core_package_structure() -> bool:
    """Test that core package has expected structure."""
    import core

    # Check that core is a package
    assert hasattr(core, '__path__'), "core should be a package"

    # Check that key modules exist
    expected_modules = [
        'api_manager',
        'browser_manager',
        'database_manager',
        'session_manager',
        'error_handling',
        'cancellation',
        'progress_indicators'
    ]

    for module_name in expected_modules:
        module_path = Path(__file__).parent / f"{module_name}.py"
        assert module_path.exists(), f"core/{module_name}.py should exist"

    return True


def _test_core_session_manager_availability() -> bool:
    """Test that SessionManager is available from core."""
    try:
        from core.session_manager import SessionManager
        assert SessionManager is not None, "SessionManager should be available"
        return True
    except ImportError:
        return False


def _test_core_database_manager_availability() -> bool:
    """Test that DatabaseManager is available from core."""
    try:
        from core.database_manager import DatabaseManager
        assert DatabaseManager is not None, "DatabaseManager should be available"
        return True
    except ImportError:
        return False


def _test_core_browser_manager_availability() -> bool:
    """Test that BrowserManager is available from core."""
    try:
        from core.browser_manager import BrowserManager
        assert BrowserManager is not None, "BrowserManager should be available"
        return True
    except ImportError:
        return False


def core_main_module_tests() -> bool:
    """
    Comprehensive test suite for core/__main__.py.
    Tests core package structure and module availability.
    """
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Core Package - Modular Session Management Architecture",
            "core/__main__.py"
        )
        suite.start_suite()

        suite.run_test(
            "Core Package Imports",
            _test_core_package_imports,
            "All core package modules can be imported",
            "Test importing all core modules",
            "Test core package availability",
        )

        suite.run_test(
            "Core Package Structure",
            _test_core_package_structure,
            "Core package has expected module structure",
            "Test core package file structure",
            "Test core package organization",
        )

        suite.run_test(
            "SessionManager Availability",
            _test_core_session_manager_availability,
            "SessionManager is available from core package",
            "Test SessionManager import",
            "Test session management availability",
        )

        suite.run_test(
            "DatabaseManager Availability",
            _test_core_database_manager_availability,
            "DatabaseManager is available from core package",
            "Test DatabaseManager import",
            "Test database management availability",
        )

        suite.run_test(
            "BrowserManager Availability",
            _test_core_browser_manager_availability,
            "BrowserManager is available from core package",
            "Test BrowserManager import",
            "Test browser management availability",
        )

        return suite.finish_suite()


# Use centralized test runner utility from test_utilities
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(core_main_module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()

    if success:
        print("\nCore Package - Modular Session Management Architecture")
        print("Version: 2.0.0")
        print("Available modules: session_manager, database_manager, browser_manager,")
        print("                  api_manager, error_handling, dependency_injection,")
        print("                  cancellation, progress_indicators, session_validator")
        print("Note: This is a package init file. Import individual modules as needed.")
        print("\nExample usage:")
        print("  from core.session_manager import SessionManager")
        print("  from core.database_manager import DatabaseManager")
        print("  from core.browser_manager import BrowserManager")

    sys.exit(0 if success else 1)
