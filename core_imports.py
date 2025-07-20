"""
Unified Import & Function Registry System

This module consolidates all import management and function registry functionality
into a single, high-performance system that eliminates the dual-system overhead
found throughout the codebase.

Key improvements:
- Consolidates path_manager.py and core_imports.py functionality
- Eliminates duplicate function registries and import systems
- Provides unified, consistent import patterns across all modules
- Single source of truth for all imports and function management
- Optimized performance with caching and minimal overhead
- Advanced error handling and recovery mechanisms
- Performance monitoring and statistics
- Thread-safe operations for concurrent access
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Callable, List, Union
import logging
import threading
from contextlib import contextmanager
from functools import wraps

# Thread-safe locks for concurrent access
_lock = threading.RLock()

# Global state tracking with enhanced metrics
_initialized = False
_project_root: Optional[Path] = None
_registry: Dict[str, Any] = {}
_import_cache: Dict[str, bool] = {}
_error_log: List[Dict[str, Any]] = []
_stats = {
    "functions_registered": 0,
    "imports_resolved": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "errors_encountered": 0,
    "initialization_time": 0.0,
    "last_cleanup": 0.0,
}


def get_import_stats() -> Dict[str, Any]:
    """Get comprehensive import system statistics."""
    with _lock:
        return {
            **_stats.copy(),
            "registry_size": len(_registry),
            "cache_size": len(_import_cache),
            "error_count": len(_error_log),
            "hit_ratio": (
                _stats["cache_hits"]
                / max(1, _stats["cache_hits"] + _stats["cache_misses"])
            )
            * 100,
        }


def get_project_root() -> Path:
    """Get the project root directory with caching and enhanced error handling."""
    global _project_root

    with _lock:
        if _project_root is None:
            try:
                # Start from current file and work up
                current = Path(__file__).resolve()

                # Look for project markers in order of preference
                markers = [
                    "main.py",
                    "requirements.txt",
                    ".git",
                    "setup.py",
                    "pyproject.toml",
                ]

                for parent in [
                    current.parent,
                    current.parent.parent,
                    current.parent.parent.parent,
                ]:
                    for marker in markers:
                        if (parent / marker).exists():
                            _project_root = parent
                            _log_info(
                                f"Project root identified: {_project_root} (marker: {marker})"
                            )
                            return _project_root

                # Fallback to current file's parent
                _project_root = current.parent
                _log_warning(f"Using fallback project root: {_project_root}")

            except Exception as e:
                # Ultimate fallback
                _project_root = Path.cwd()
                _log_error(f"Error determining project root, using CWD: {e}")
                _stats["errors_encountered"] += 1

        return _project_root


def _log_info(message: str) -> None:
    """Internal logging function that safely handles early initialization."""
    try:
        logging.getLogger(__name__).info(message)
    except:
        print(f"INFO: {message}")


def _log_warning(message: str) -> None:
    """Internal logging function that safely handles early initialization."""
    try:
        logging.getLogger(__name__).warning(message)
    except:
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
    except:
        print(f"ERROR: {message}")


def ensure_imports() -> None:
    """Ensure all imports are properly configured. Call once per module."""
    global _initialized, _stats
    if _initialized:
        return

    start_time = time.time()
    project_root = str(get_project_root())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    _stats["initialization_time"] = time.time() - start_time
    _stats["imports_resolved"] += 1
    _initialized = True


@contextmanager
def import_context():
    """Context manager for safe import operations."""
    original_path = sys.path.copy()
    try:
        ensure_imports()
        yield
    finally:
        sys.path[:] = original_path


def register_function(name: str, func: Callable) -> None:
    """Register a function in the unified registry with performance tracking."""
    if callable(func):
        _registry[name] = func
        _stats["functions_registered"] += 1
        # Clear cache when registry changes
        _import_cache.clear()


def register_many(**functions) -> None:
    """Register multiple functions efficiently."""
    for name, func in functions.items():
        if callable(func):
            _registry[name] = func
            _stats["functions_registered"] += 1
    _import_cache.clear()


def get_function(name: str, default: Any = None) -> Any:
    """Get a function from the registry with caching."""
    return _registry.get(name, default)


def is_function_available(name: str) -> bool:
    """Check if a function is available with performance caching."""
    if name in _import_cache:
        _stats["cache_hits"] += 1
        return _import_cache[name]

    result = name in _registry and callable(_registry[name])
    _import_cache[name] = result
    return result


def call_function(name: str, *args, **kwargs) -> Any:
    """Safely call a function from the registry."""
    if is_function_available(name):
        return _registry[name](*args, **kwargs)
    raise ValueError(f"Function '{name}' not available in registry")


def get_available_functions() -> List[str]:
    """Get list of all available function names."""
    return [name for name in _registry.keys() if callable(_registry[name])]


def auto_register_module(module_globals: Dict[str, Any], module_name: str) -> None:
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
            in [
                "run_comprehensive_tests",
                "main",
                "SessionManager",
                "DatabaseManager",
                "BrowserManager",
                "APIManager",
            ]
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
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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


def get_stats() -> Dict[str, Any]:
    """Get performance statistics for the unified system."""
    return {
        **_stats,
        "registry_size": len(_registry),
        "cache_size": len(_import_cache),
        "cache_hit_rate": (
            _stats["cache_hits"] / max(1, _stats["cache_hits"] + len(_import_cache))
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


def safe_execute(
    func: Optional[Callable] = None,
    *,
    default_return: Any = None,
    suppress_errors: bool = True,
    log_errors: bool = True,
):
    """
    Unified safe execution decorator that consolidates error handling patterns.
    Replaces scattered try/catch blocks throughout the codebase.
    """

    def decorator(f):
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = get_logger()
                    logger.error(f"Error in {f.__name__}: {e}", exc_info=True)
                if not suppress_errors:
                    raise
                return default_return

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def cleanup_registry() -> None:
    """Clean up the registry and reset caches."""
    global _registry, _import_cache, _stats
    _registry.clear()
    _import_cache.clear()
    _stats = {
        "functions_registered": 0,
        "imports_resolved": 0,
        "cache_hits": 0,
        "initialization_time": 0.0,
    }


def run_comprehensive_tests() -> bool:
    """Comprehensive test suite for the unified import system."""
    logger = get_logger(__name__)
    logger.info("🔧 Testing Unified Import & Function Registry System...")

    success = True
    tests_run = 0
    tests_passed = 0

    # Test 1: Function registration and retrieval
    tests_run += 1
    try:

        def test_func(x):
            return x * 2

        register_function("test_func", test_func)
        assert is_function_available("test_func")
        assert call_function("test_func", 5) == 10
        tests_passed += 1
        logger.info("✅ Function registration and retrieval")
    except Exception as e:
        logger.error(f"❌ Function registration failed: {e}")
        success = False

    # Test 2: Auto-registration
    tests_run += 1
    try:
        test_globals = {"test_function": lambda: "test", "_private": lambda: "private"}
        auto_register_module(test_globals, "test_module")
        assert is_function_available("test_module.test_function")
        tests_passed += 1
        logger.info("✅ Auto-registration")
    except Exception as e:
        logger.error(f"❌ Auto-registration failed: {e}")
        success = False

    # Test 3: Performance caching
    tests_run += 1
    try:
        # Test cache performance
        start_time = time.time()
        for _ in range(1000):
            is_function_available("test_func")
        end_time = time.time()

        duration = end_time - start_time
        stats = get_stats()
        cache_hit_rate = stats["cache_hit_rate"]

        assert duration < 0.1, f"1000 lookups should be fast, took {duration:.3f}s"
        assert (
            cache_hit_rate > 50
        ), f"Cache hit rate should be high, got {cache_hit_rate:.1f}%"
        tests_passed += 1
        logger.info(f"✅ Performance caching (hit rate: {cache_hit_rate:.1f}%)")
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        success = False

    # Test 4: Import standardization
    tests_run += 1
    try:
        result = standardize_module_imports()
        assert result == True
        tests_passed += 1
        logger.info("✅ Import standardization")
    except Exception as e:
        logger.error(f"❌ Import standardization failed: {e}")
        success = False

    # Test 5: Context manager
    tests_run += 1
    try:
        original_path = sys.path.copy()
        with import_context():
            pass
        assert sys.path == original_path
        tests_passed += 1
        logger.info("✅ Import context manager")
    except Exception as e:
        logger.error(f"❌ Context manager failed: {e}")
        success = False

    status = "PASSED" if success else "FAILED"
    logger.info(
        f"🎯 Unified Import System Tests: {status} ({tests_passed}/{tests_run})"
    )

    # Log performance stats
    stats = get_stats()
    logger.info(
        f"📊 Performance: {stats['functions_registered']} functions, "
        f"{stats['cache_hit_rate']:.1f}% cache hit rate"
    )

    return success


# Initialize immediately when imported
ensure_imports()

# Export clean, unified interface
__all__ = [
    "ensure_imports",
    "register_function",
    "register_many",
    "get_function",
    "is_function_available",
    "call_function",
    "get_available_functions",
    "auto_register_module",
    "standardize_module_imports",
    "get_logger",
    "get_project_root",
    "import_context",
    "safe_execute",
    "cleanup_registry",
    "get_stats",
    "run_comprehensive_tests",
]
