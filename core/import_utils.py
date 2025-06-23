"""
Common Import Utilities.

This module provides standardized import patterns and fallbacks to eliminate
code duplication across the codebase.
"""

from typing import Any, Optional, Callable


class DummyFunctionRegistry:
    """
    Dummy function registry for when path_manager is not available.

    This provides a consistent fallback pattern across all modules
    that need to handle missing path_manager imports.
    """

    def register(self, name: str, func: Callable) -> None:
        """Register a function (no-op in dummy implementation)."""
        pass

    def get(self, name: str, default: Any = None) -> Any:
        """Get a function (returns default in dummy implementation)."""
        return default

    def is_available(self, name: str) -> bool:
        """Check if a function is available (always False in dummy implementation)."""
        return False


def get_function_registry():
    """
    Get function registry with safe fallback.

    Returns:
        Either the real function_registry from path_manager, or a dummy implementation
    """
    try:
        from core_imports import register_function, get_function, is_function_available

        return function_registry
    except ImportError:
        return DummyFunctionRegistry()


def safe_import(module_name: str, fallback_value: Any = None) -> Any:
    """
    Safely import a module with fallback.

    Args:
        module_name: Name of module to import
        fallback_value: Value to return if import fails

    Returns:
        Imported module or fallback value
    """
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError:
        return fallback_value


def safe_import_from(
    module_name: str, item_name: str, fallback_value: Any = None
) -> Any:
    """
    Safely import an item from a module with fallback.

    Args:
        module_name: Name of module to import from
        item_name: Name of item to import
        fallback_value: Value to return if import fails

    Returns:
        Imported item or fallback value
    """
    try:
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, item_name)
    except (ImportError, AttributeError):
        return fallback_value


# Pre-instantiated function registry for convenience
function_registry = get_function_registry()
