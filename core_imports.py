#!/usr/bin/env python3

"""
Core Import System - Unified Module and Function Registry

Provides centralized import management, function registry, and module coordination
with high-performance caching, thread-safe operations, and comprehensive error
handling for the entire Ancestry automation project infrastructure.
"""

import logging
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

# Thread-safe locks for concurrent access
_lock = threading.RLock()

# Global state tracking with enhanced metrics
_initialized = False
_project_root: Optional[Path] = None
_registry: dict[str, Any] = {}
_import_cache: dict[str, bool] = {}
_error_log: list[dict[str, Any]] = []
_stats = {
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
    global _initialized
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
    return decorator(func)


def cleanup_registry() -> None:
    """Clean up the registry and reset caches."""
    global _stats
    _registry.clear()
    _import_cache.clear()
    _stats = {
        "functions_registered": 0,
        "imports_resolved": 0,
        "cache_hits": 0,
        "initialization_time": 0.0,
    }


def core_imports_module_tests() -> bool:
    """Module-specific tests for core_imports.py functionality."""
    try:
        # Test 1: Function registration and retrieval
        def test_func(x):
            return x * 2

        register_function("test_func", test_func)
        assert is_function_available(
            "test_func"
        ), "Function should be available after registration"
        assert call_function("test_func", 5) == 10, "Function should execute correctly"

        # Test 2: Auto-registration
        test_globals = {"test_function": lambda: "test", "_private": lambda: "private"}
        auto_register_module(test_globals, "test_module")
        assert is_function_available(
            "test_module.test_function"
        ), "Auto-registered function should be available"

        # Test 3: Performance caching
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

        # Test 4: Import standardization
        result = standardize_module_imports()
        assert result, "Import standardization should succeed"

        # Test 5: Context manager
        original_path = sys.path.copy()
        with import_context():
            pass
        assert sys.path == original_path, "Context manager should restore sys.path"

        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Core imports module tests failed: {e}")
        return False


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for core imports functionality.

    Tests all core import system functionality including function registry,
    module management, performance caching, and error handling.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        from test_framework import TestSuite
    except ImportError:
        print("⚠️  TestSuite not available - falling back to basic testing")
        return core_imports_module_tests()

    suite = TestSuite("Core Imports", "core_imports")

    def test_function_registry():
        """Test function registration and retrieval"""
        # Test basic registration
        def test_func():
            return "test_result"

        register_function("test_func", test_func)
        assert is_function_available("test_func")

        retrieved_func = get_function("test_func")
        assert retrieved_func is not None
        assert retrieved_func() == "test_result"

        # Test function call
        result = call_function("test_func")
        assert result == "test_result"

    def test_bulk_registration():
        """Test bulk function registration"""
        # Test using **kwargs format
        register_many(
            bulk_test1=lambda: "result1",
            bulk_test2=lambda: "result2"
        )

        assert is_function_available("bulk_test1")
        assert is_function_available("bulk_test2")
        assert call_function("bulk_test1") == "result1"
        assert call_function("bulk_test2") == "result2"

    def test_module_auto_registration():
        """Test automatic module registration"""
        # Create mock module globals with test functions
        mock_globals = {
            "test_auto_func": lambda: "auto_result",
            "another_test_func": lambda: "another_result",
            "__name__": "mock_test_module"
        }

        # Test auto registration
        auto_register_module(mock_globals, "mock_test_module")

        # Functions are registered with module prefix
        assert is_function_available("mock_test_module.test_auto_func")
        assert call_function("mock_test_module.test_auto_func") == "auto_result"
        assert is_function_available("mock_test_module.another_test_func")
        assert call_function("mock_test_module.another_test_func") == "another_result"

    def test_project_root_detection():
        """Test project root detection"""
        root = get_project_root()
        assert root is not None
        assert isinstance(root, Path)
        assert root.exists()

    def test_logger_functionality():
        """Test logger creation and functionality"""
        logger = get_logger("test_logger")
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')

        # Test logger naming
        logger2 = get_logger("test.module.name")
        assert logger2 is not None

    def test_import_context_manager():
        """Test import context manager"""
        original_path = sys.path.copy()

        with import_context():
            # Context manager should preserve sys.path
            pass

        assert sys.path == original_path

    def test_safe_execution():
        """Test safe function execution wrapper"""
        def safe_func():
            return "safe_result"

        def error_func():
            raise ValueError("Test error")

        # Test successful execution using decorator
        @safe_execute
        def decorated_safe_func():
            return "safe_result"

        result = decorated_safe_func()
        assert result == "safe_result"

        # Test error handling using decorator with default return
        @safe_execute(default_return="default")
        def decorated_error_func():
            raise ValueError("Test error")

        result = decorated_error_func()
        assert result == "default"

    def test_performance_caching():
        """Test performance caching functionality"""
        # Clear cache first
        _import_cache.clear()
        _stats["cache_hits"] = 0
        _stats["cache_misses"] = 0

        # Register a test function
        register_function("cache_test", lambda: "cached")

        # Multiple lookups should hit cache
        for _ in range(5):
            assert is_function_available("cache_test")

        # Should have cache hits
        stats = get_import_stats()
        assert stats["cache_hits"] > 0

    def test_statistics_tracking():
        """Test import statistics tracking"""
        initial_stats = get_import_stats()
        assert isinstance(initial_stats, dict)
        assert "functions_registered" in initial_stats
        assert "imports_resolved" in initial_stats
        assert "cache_hits" in initial_stats
        assert "registry_size" in initial_stats

    def test_cleanup_functionality():
        """Test registry cleanup functionality"""
        # Register some test functions
        register_function("cleanup_test1", lambda: "test1")
        register_function("cleanup_test2", lambda: "test2")

        # Verify functions are available before cleanup
        assert is_function_available("cleanup_test1")
        assert is_function_available("cleanup_test2")

        initial_size = len(_registry)
        assert initial_size >= 2  # Should have at least our test functions

        # Test cleanup
        cleanup_registry()

        # Registry should be empty after cleanup
        assert len(_registry) == 0
        assert not is_function_available("cleanup_test1")
        assert not is_function_available("cleanup_test2")

        # Registry should still work after cleanup - register new functions
        register_function("post_cleanup_test", lambda: "post_cleanup")
        assert is_function_available("post_cleanup_test")

    def test_error_handling():
        """Test error handling and recovery"""
        # Test with non-existent function
        assert not is_function_available("nonexistent_function")

        # Test that call_function raises ValueError for non-existent function
        try:
            call_function("nonexistent_function")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "not available in registry" in str(e)

        # Test with invalid module registration (empty globals)
        try:
            auto_register_module({}, "empty_module")
            # Should handle gracefully without error
        except Exception:
            raise AssertionError("Should handle empty module gracefully")

    def test_function_availability():
        """Test that all required functions are available"""
        required_functions = [
            "ensure_imports", "register_function", "get_function", "is_function_available",
            "call_function", "auto_register_module", "get_logger", "get_project_root",
            "safe_execute", "cleanup_registry"
        ]

        from test_framework import test_function_availability
        test_function_availability(required_functions, globals(), "Core Imports")

    # Run all tests
    suite.run_test(
        "Function registry operations",
        test_function_registry,
        "Function registration and retrieval works correctly with registry system",
        "Test register_function, get_function, and is_function_available operations",
        "Verify function registry enables proper registration and retrieval of callables"
    )

    suite.run_test(
        "Bulk function registration",
        test_bulk_registration,
        "Bulk function registration processes multiple functions efficiently",
        "Test register_many function with dictionary of functions",
        "Verify bulk registration handles multiple function registrations properly"
    )

    suite.run_test(
        "Module auto-registration",
        test_module_auto_registration,
        "Automatic module registration discovers and registers module functions",
        "Test auto_register_module with mock module and function discovery",
        "Verify auto-registration scans modules and registers available functions"
    )

    suite.run_test(
        "Project root detection",
        test_project_root_detection,
        "Project root detection finds correct project directory structure",
        "Test get_project_root function with filesystem path detection",
        "Verify project root detection locates correct directory hierarchy"
    )

    suite.run_test(
        "Logger functionality",
        test_logger_functionality,
        "Logger creation provides proper logging interface for modules",
        "Test get_logger function with various module name formats",
        "Verify logger creation provides standard logging interface"
    )

    suite.run_test(
        "Import context management",
        test_import_context_manager,
        "Import context manager preserves system state during operations",
        "Test import_context context manager with sys.path preservation",
        "Verify context manager maintains system import path state"
    )

    suite.run_test(
        "Safe execution wrapper",
        test_safe_execution,
        "Safe execution wrapper handles errors gracefully with fallbacks",
        "Test safe_execute function with both successful and error cases",
        "Verify safe execution provides error handling with default returns"
    )

    suite.run_test(
        "Performance caching",
        test_performance_caching,
        "Performance caching improves function lookup efficiency",
        "Test caching system with repeated function availability checks",
        "Verify caching system reduces lookup overhead for repeated operations"
    )

    suite.run_test(
        "Statistics tracking",
        test_statistics_tracking,
        "Statistics tracking provides comprehensive system metrics",
        "Test get_import_stats function with various operational metrics",
        "Verify statistics provide insights into import system performance"
    )

    suite.run_test(
        "Cleanup functionality",
        test_cleanup_functionality,
        "Registry cleanup maintains system health without losing functionality",
        "Test cleanup_registry function with registered functions",
        "Verify cleanup operations maintain registry functionality"
    )

    suite.run_test(
        "Error handling and recovery",
        test_error_handling,
        "Error conditions are handled gracefully with proper fallbacks",
        "Test error handling with missing functions and invalid operations",
        "Verify robust error handling provides safe defaults for failures"
    )

    suite.run_test(
        "Function availability verification",
        test_function_availability,
        "All required core import functions are available and callable",
        "Test availability of ensure_imports, register_function, and utility functions",
        "Verify function availability ensures complete core import interface"
    )

    return suite.finish_suite()


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
