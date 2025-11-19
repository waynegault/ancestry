#!/usr/bin/env python3

"""
Core Import System - Unified Module and Function Registry

Provides centralized import management, function registry, and module coordination
with high-performance caching, thread-safe operations, and comprehensive error
handling for the Ancestry automation project.

Features:
- Thread-safe function registry with performance caching
- Project root detection and path management
- Import standardization and error tracking
- Performance metrics and statistics
"""

import logging
import sys
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, ParamSpec, TypeVar, cast, overload

# Thread-safe locks for concurrent access
_lock = threading.RLock()


# Global state tracking with enhanced metrics
class _ImportSystemState:
    """Manages import system state."""
    initialized = False
    project_root: Optional[Path] = None


_registry: dict[str, Any] = {}
_import_cache: dict[str, bool] = {}
_error_log: list[dict[str, Any]] = []


class _ImportStats:
    """Manages import system statistics."""

    data: ClassVar[dict[str, Any]] = {
        "functions_registered": 0,
        "imports_resolved": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "errors_encountered": 0,
        "initialization_time": 0.0,
        "last_cleanup": 0.0,
    }


def get_import_stats() -> dict[str, Any]:
    """Get comprehensive import system statistics."""
    with _lock:
        return {
            **_ImportStats.data.copy(),
            "registry_size": len(_registry),
            "cache_size": len(_import_cache),
            "error_count": len(_error_log),
            "hit_ratio": (
                _ImportStats.data["cache_hits"]
                / max(1, _ImportStats.data["cache_hits"] + _ImportStats.data["cache_misses"])
            )
            * 100,
        }


def get_project_root() -> Path:
    """Get the project root directory with caching and enhanced error handling."""
    with _lock:
        if _ImportSystemState.project_root is None:
            try:
                _ImportSystemState.project_root = _detect_project_root()
            except Exception as e:
                _ImportSystemState.project_root = Path.cwd()
                _log_error(f"Error determining project root, using CWD: {e}")
                _ImportStats.data["errors_encountered"] += 1

        return _ImportSystemState.project_root


def _detect_project_root() -> Path:
    """Detect project root by walking parents and checking markers."""
    current = Path(__file__).resolve()
    markers = {"main.py", "requirements.txt", ".git", "setup.py", "pyproject.toml"}

    for parent in (current.parent, current.parent.parent, current.parent.parent.parent):
        for marker in markers:
            if (parent / marker).exists():
                _log_info(f"Project root identified: {parent} (marker: {marker})")
                return parent

    fallback = current.parent
    _log_warning(f"Using fallback project root: {fallback}")
    return fallback


def _log_info(message: str) -> None:
    """Internal logging function that safely handles early initialization."""
    try:
        logging.getLogger(__name__).info(message)
    except Exception:
        print(f"INFO: {message}")


def _log_warning(message: str) -> None:
    """Internal logging function that safely handles early initialization."""
    try:
        logging.getLogger(__name__).warning(message)
    except Exception:
        print(f"WARNING: {message}")


def _log_error(message: str) -> None:
    """Internal logging function that safely handles early initialization."""
    try:
        logging.getLogger(__name__).error(message)
        _error_log.append(
            {
                "timestamp": time.time(),
                "message": message,
                "function": "get_project_root",
            }
        )
    except Exception:
        print(f"ERROR: {message}")


def ensure_imports() -> None:
    """Ensure all imports are properly configured. Call once per module."""
    if _ImportSystemState.initialized:
        return

    start_time = time.time()
    project_root = str(get_project_root())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    _ImportStats.data["initialization_time"] = time.time() - start_time
    _ImportStats.data["imports_resolved"] += 1
    _ImportSystemState.initialized = True


@contextmanager
def import_context() -> Any:
    """Context manager for safe import operations."""
    original_path = sys.path.copy()
    try:
        ensure_imports()
        yield
    finally:
        sys.path[:] = original_path


def register_function(name: str, func: Callable[..., Any]) -> None:
    """Register a function in the unified registry with performance tracking."""
    if callable(func):
        _registry[name] = func
        _ImportStats.data["functions_registered"] += 1
        # Clear cache when registry changes
        _import_cache.clear()


def register_many(**functions: Callable[..., Any]) -> None:
    """Register multiple functions efficiently."""
    for name, func in functions.items():
        if callable(func):
            _registry[name] = func
            _ImportStats.data["functions_registered"] += 1
    _import_cache.clear()


def get_function(name: str, default: Any = None) -> Any:
    """Get a function from the registry with caching."""
    return _registry.get(name, default)


def is_function_available(name: str) -> bool:
    """Check if a function is available with performance caching."""
    if name in _import_cache:
        _ImportStats.data["cache_hits"] += 1
        return _import_cache[name]

    result = name in _registry and callable(_registry[name])
    _import_cache[name] = result
    return result


def call_function(name: str, *args: Any, **kwargs: Any) -> Any:
    """Safely call a function from the registry."""
    if is_function_available(name):
        return _registry[name](*args, **kwargs)
    raise ValueError(f"Function '{name}' not available in registry")


def get_available_functions() -> list[str]:
    """Get list of all available function names."""
    return [name for name in _registry if callable(_registry[name])]


def auto_register_module(module_globals: dict[str, Any], module_name: str) -> None:
    """
    Automatically register all callable objects from a module.
    Consolidates the functionality from core.registry_utils.auto_register_module.
    """
    registered_count = 0
    for name, obj in module_globals.items():
        if callable(obj) and not name.startswith("_"):
            register_function(f"{module_name}.{name}", obj)
            registered_count += 1

    # Also register with simple names for commonly used functions
    for name, obj in module_globals.items():
        if (
            callable(obj)
            and not name.startswith("_")
            and name
            in {
                "run_comprehensive_tests",
                "main",
                "SessionManager",
                "DatabaseManager",
                "BrowserManager",
                "APIManager",
            }
        ):
            register_function(name, obj)


def standardize_module_imports() -> bool:
    """
    Standardized import pattern that replaces scattered fallback patterns.
    Consolidates functionality from path_manager.standardize_module_imports.
    """
    try:
        ensure_imports()
        return True
    except Exception:
        # Try common fallback patterns
        fallback_patterns = [
            str(Path(__file__).parent.parent),
            str(Path(__file__).resolve().parent.parent),
        ]

        for pattern in fallback_patterns:
            try:
                if pattern not in sys.path:
                    sys.path.insert(0, pattern)
                return True
            except Exception:
                continue
        return False


def get_stats() -> dict[str, Any]:
    """Get performance statistics for the unified system."""
    return {
        **_ImportStats.data,
        "registry_size": len(_registry),
        "cache_size": len(_import_cache),
        "cache_hit_rate": (
            _ImportStats.data["cache_hits"] / max(1, _ImportStats.data["cache_hits"] + len(_import_cache))
        )
        * 100,
    }


# Smart logger setup with fallback
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a properly configured logger with smart fallback."""
    try:
        from logging_config import logger

        return logger
    except ImportError:
        # Fallback logger with basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        return logging.getLogger(name or __name__)


P = ParamSpec("P")
R = TypeVar("R")


@overload
def safe_execute(
    func: Callable[P, R],
    *,
    default_return: Optional[R] = None,
    suppress_errors: bool = ...,
    log_errors: bool = ...,
) -> Callable[P, R]:
    ...


@overload
def safe_execute(
    func: None = ...,
    *,
    default_return: Optional[R] = None,
    suppress_errors: bool = ...,
    log_errors: bool = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    ...


def safe_execute(
    func: Optional[Callable[P, R]] = None,
    *,
    default_return: Optional[R] = None,
    suppress_errors: bool = True,
    log_errors: bool = True,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Unified safe-execution decorator that preserves wrapped signatures."""

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = get_logger()
                    logger.error(f"Error in {f.__name__}: {e}", exc_info=True)
                if not suppress_errors:
                    raise
                return cast(R, default_return)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def cleanup_registry() -> None:
    """Clean up the registry and reset caches."""
    _registry.clear()
    _import_cache.clear()
    _ImportStats.data = {
        "functions_registered": 0,
        "imports_resolved": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "errors_encountered": 0,
        "initialization_time": 0.0,
        "last_cleanup": 0.0,
    }


# ==============================================
# Module Tests
# ==============================================


def _test_function_registration() -> None:
    """Test function registration and retrieval."""
    from test_utilities import mock_func_with_param as test_func

    register_function("test_func", test_func)
    assert is_function_available(
        "test_func"
    ), "Function should be available after registration"
    assert call_function("test_func", 5) == 10, "Function should execute correctly"


def _test_auto_registration() -> None:
    """Test auto-registration of module functions."""
    test_globals = {"test_function": lambda: "test", "_private": lambda: "private"}
    auto_register_module(test_globals, "test_module")
    assert is_function_available(
        "test_module.test_function"
    ), "Auto-registered function should be available"


def _test_performance_caching() -> None:
    """Test performance caching."""
    start_time = time.time()
    for _ in range(1000):
        is_function_available("test_func")
    duration = time.time() - start_time

    stats = get_stats()
    cache_hit_rate = stats["cache_hit_rate"]

    assert duration < 0.1, f"1000 lookups should be fast, took {duration:.3f}s"
    assert (
        cache_hit_rate > 50
    ), f"Cache hit rate should be high, got {cache_hit_rate:.1f}%"


def _test_import_standardization() -> None:
    """Test import standardization."""
    result = standardize_module_imports()
    assert result, "Import standardization should succeed"


def _test_context_manager() -> None:
    """Test import context manager."""
    original_path = sys.path.copy()
    with import_context():
        pass
    assert sys.path == original_path, "Context manager should restore sys.path"


def core_imports_module_tests() -> bool:
    """Module-specific tests for core_imports.py functionality."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Core Imports", "core_imports.py")

    tests = [
        ("Function registration and retrieval", _test_function_registration, "Test function registry", "direct", "Test function registry"),
        ("Auto-registration", _test_auto_registration, "Test module auto-registration", "direct", "Test module auto-registration"),
        ("Performance caching", _test_performance_caching, "Test lookup caching", "direct", "Test lookup caching"),
        ("Import standardization", _test_import_standardization, "Test import standardization", "direct", "Test import standardization"),
        ("Context manager", _test_context_manager, "Test import context manager", "direct", "Test import context manager"),
    ]

    with suppress_logging():
        for test_name, test_func, expected_behavior, test_description, method_description in tests:
            suite.run_test(test_name, test_func, expected_behavior, test_description, method_description)

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

# Use centralized test runner utility
run_comprehensive_tests = create_standard_test_runner(core_imports_module_tests)


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    print("🔧 Running Core Imports comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)


# Initialize immediately when imported
ensure_imports()

# Export clean, unified interface
__all__ = [
    "auto_register_module",
    "call_function",
    "cleanup_registry",
    "ensure_imports",
    "get_available_functions",
    "get_function",
    "get_logger",
    "get_project_root",
    "get_stats",
    "import_context",
    "is_function_available",
    "register_function",
    "register_many",
    "run_comprehensive_tests",
    "safe_execute",
    "standardize_module_imports",
]
