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

# Version information
__version__ = "2.0.0"

# Export key components
from .session_manager import SessionManager
from .database_manager import DatabaseManager
from .browser_manager import BrowserManager
from .api_manager import APIManager
from .session_validator import SessionValidator
from .dependency_injection import (
    DIContainer,
    Injectable,
    inject,
    get_container,
    get_service,
    configure_dependencies,
    DIScope,
)
from .error_handling import (
    AppError,
    ErrorSeverity,
    ErrorCategory,
    AuthenticationError,
    ValidationError,
    DatabaseError,
    NetworkError,
    BrowserError,
    APIError,
    ConfigurationError,
    error_handler,
    handle_error,
    safe_execute,
    ErrorContext,
)

__all__ = [
    # Core managers
    "SessionManager",
    "DatabaseManager",
    "BrowserManager",
    "APIManager",
    "SessionValidator",
    # Dependency injection
    "DIContainer",
    "Injectable",
    "inject",
    "get_container",
    "get_service",
    "configure_dependencies",
    "DIScope",
    # Error handling
    "AppError",
    "ErrorSeverity",
    "ErrorCategory",
    "AuthenticationError",
    "ValidationError",
    "DatabaseError",
    "NetworkError",
    "BrowserError",
    "APIError",
    "ConfigurationError",
    "error_handler",
    "handle_error",
    "safe_execute",
    "ErrorContext",
]
