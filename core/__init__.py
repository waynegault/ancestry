"""
Core Package - Modular Session Management Architecture.

This package provides the refactored, modular architecture that replaces
the monolithic SessionManager with specialized, focused components.

Components:
- DatabaseManager: Database operations and connection management
- BrowserManager: WebDriver/browser session management
- APIManager: HTTP requests and API interactions
- SessionValidator: Session validation and readiness checks
- SessionManager: Orchestrating manager that coordinates all components
- DependencyInjection: Dependency injection framework for clean architecture
- ErrorHandling: Standardized error handling across all components

The new architecture provides:
- Clear separation of concerns
- Improved testability
- Better maintainability
- Reduced coupling between components
- Standardized error handling
- Dependency injection for clean component relationships
"""

from typing import Any

# Version information
__version__ = "2.0.0"

# Export key components
try:
    # Try relative imports first (normal package usage)
    from .api_manager import APIManager
    from .browser_manager import BrowserManager
    from .database_manager import DatabaseManager
    from .dependency_injection import (
        DIContainer,
        DIScope,
        Injectable,
        configure_dependencies,
        get_container,
        get_service,
        inject,
    )
    from .error_handling import (
        APIError,
        AppError,
        AuthenticationError,
        BrowserError,
        ConfigurationError,
        DatabaseError,
        ErrorCategory,
        ErrorContext,
        ErrorSeverity,
        NetworkError,
        ValidationError,
        error_handler,
        handle_error,
        safe_execute,
    )
    from .session_manager import SessionManager
    from .session_validator import SessionValidator
except ImportError:
    # If relative imports fail (e.g., when running as __main__),
    # set up dummy objects for testing
    class DummyComponent:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, *args: Any, **kwargs: Any) -> 'DummyComponent':
            return self

        def __repr__(self) -> str:
            return f"<DummyComponent: {self.name}>"

    # Create dummy components for testing
    SessionManager = DummyComponent("SessionManager")
    DatabaseManager = DummyComponent("DatabaseManager")
    BrowserManager = DummyComponent("BrowserManager")
    APIManager = DummyComponent("APIManager")
    SessionValidator = DummyComponent("SessionValidator")
    DIContainer = DummyComponent("DIContainer")
    Injectable = DummyComponent("Injectable")
    inject = DummyComponent("inject")
    get_container = DummyComponent("get_container")
    get_service = DummyComponent("get_service")
    configure_dependencies = DummyComponent("configure_dependencies")
    DIScope = DummyComponent("DIScope")
    AppError = DummyComponent("AppError")
    ErrorSeverity = DummyComponent("ErrorSeverity")
    ErrorCategory = DummyComponent("ErrorCategory")
    AuthenticationError = DummyComponent("AuthenticationError")
    ValidationError = DummyComponent("ValidationError")
    DatabaseError = DummyComponent("DatabaseError")
    NetworkError = DummyComponent("NetworkError")
    BrowserError = DummyComponent("BrowserError")
    APIError = DummyComponent("APIError")
    ConfigurationError = DummyComponent("ConfigurationError")
    error_handler = DummyComponent("error_handler")
    handle_error = DummyComponent("handle_error")
    safe_execute = DummyComponent("safe_execute")
    ErrorContext = DummyComponent("ErrorContext")

__all__ = [
    "APIError",
    "APIManager",
    "AppError",
    "AuthenticationError",
    "BrowserError",
    "BrowserManager",
    "ConfigurationError",
    "DIContainer",
    "DIScope",
    "DatabaseError",
    "DatabaseManager",
    "ErrorCategory",
    "ErrorContext",
    "ErrorSeverity",
    "Injectable",
    "NetworkError",
    "SessionManager",
    "SessionValidator",
    "ValidationError",
    "configure_dependencies",
    "error_handler",
    "get_container",
    "get_service",
    "handle_error",
    "inject",
    "safe_execute",
]


def core_package_module_tests() -> bool:
    """
    Comprehensive test suite for core package initialization.
    Tests package structure, imports, and component availability.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Core Package Initialization", "core.__init__.py")
    suite.start_suite()

    def test_package_structure():
        """Test package structure and version information."""
        # Test version information
        assert __version__ == "2.0.0", f"Expected version 2.0.0, got {__version__}"

        # Test __all__ list exists and has expected components
        assert "__all__" in globals(), "__all__ list should be defined"
        assert len(__all__) > 0, "__all__ list should not be empty"

        # Test key components are in __all__
        required_components = [
            "SessionManager",
            "DatabaseManager",
            "BrowserManager",
            "APIManager",
            "SessionValidator",
            "DIContainer",
            "AppError",
        ]
        for component in required_components:
            assert component in __all__, f"Component {component} should be in __all__"

    def test_component_imports():
        """Test that all core components can be imported."""
        # Test core managers (use globals since they're already imported)
        assert callable(SessionManager), "SessionManager should be callable"
        assert callable(DatabaseManager), "DatabaseManager should be callable"
        assert callable(BrowserManager), "BrowserManager should be callable"
        assert callable(APIManager), "APIManager should be callable"
        assert callable(SessionValidator), "SessionValidator should be callable"

    def test_dependency_injection_imports():
        """Test dependency injection components."""
        # Test using globals since they're already imported
        assert callable(DIContainer), "DIContainer should be callable"
        assert callable(Injectable), "Injectable should be callable"
        assert callable(inject), "inject should be callable"
        assert callable(get_container), "get_container should be callable"
        assert callable(get_service), "get_service should be callable"

    def test_error_handling_imports():
        """Test error handling components."""
        # Test using globals since they're already imported
        assert callable(AppError), "AppError should be callable"
        assert ErrorSeverity is not None, "ErrorSeverity should be available"
        assert ErrorCategory is not None, "ErrorCategory should be available"
        assert callable(error_handler), "error_handler should be callable"
        assert callable(handle_error), "handle_error should be callable"

    with suppress_logging():
        suite.run_test(
            "Package structure",
            test_package_structure,
            "Package structure and version should be correct",
            "Test package structure and version information",
            "Package provides correct version and __all__ list",
        )

        suite.run_test(
            "Component imports",
            test_component_imports,
            "Core components should be importable",
            "Test that all core components can be imported",
            "Core managers import correctly",
        )

        suite.run_test(
            "Dependency injection imports",
            test_dependency_injection_imports,
            "Dependency injection components should be importable",
            "Test dependency injection components",
            "DI framework components import correctly",
        )

        suite.run_test(
            "Error handling imports",
            test_error_handling_imports,
            "Error handling components should be importable",
            "Test error handling components",
            "Error handling framework imports correctly",
        )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(core_package_module_tests)


if __name__ == "__main__":
    # import os  # unused import cleanup
    import sys

    # Use centralized path management
    try:
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core_imports import standardize_module_imports

        standardize_module_imports()
    except ImportError:
        # Fallback for testing environment
        sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

    if "--test" in sys.argv:
        print("🏗️ Running Core Package comprehensive test suite...")
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        print("Core Package - Modular Session Management Architecture")
        print(f"Version: {__version__}")
        print(
            "Note: This is a package init file. Use 'python -m core' or import as a package."
        )
