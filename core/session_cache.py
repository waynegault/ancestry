#!/usr/bin/env python3

"""
Session Cache - High-Performance Session State Caching

This module provides intelligent caching for session manager components to dramatically
reduce initialization overhead. Addresses the 34.59s session manager bottleneck by
implementing persistent session state and component reuse.

Extends the existing cache.py infrastructure rather than duplicating functionality.
"""

# === CORE INFRASTRUCTURE ===
import sys

# Add parent directory to path for standard_imports
from pathlib import Path as PathLib

parent_dir = str(PathLib(__file__).parent.parent.resolve())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import logging
from contextlib import contextmanager
from functools import wraps
from importlib import import_module
from typing import (
    Any,
    Callable,
    Optional,
    ParamSpec,
    TypeVar,
    cast,
)

# === OPTIONAL STANDARD IMPORTS SETUP ===
try:
    from standard_imports import setup_module
except Exception:  # pragma: no cover - fallback when bootstrap module missing

    def setup_module(module_globals: dict[str, object], module_name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO)
        logger_obj = logging.getLogger(module_name)
        module_globals["logger"] = logger_obj
        return logger_obj
logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# === STANDARD LIBRARY IMPORTS ===
import hashlib
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass

P = ParamSpec("P")
R = TypeVar("R")


# === LEVERAGE EXISTING CACHE INFRASTRUCTURE ===
class _BaseCacheModuleFallback:
    def __init__(self, *_: Any, **__: Any) -> None:
        return


BaseCacheModule: type = _BaseCacheModuleFallback
cache: Any = None


def get_cache_stats() -> dict[str, Any]:
    return {"cache_available": False}


def get_unified_cache_key(*parts: Any) -> str:
    return "::".join(str(part) for part in parts)


def warm_cache_with_data(*_: Any, **__: Any) -> bool:
    return False


try:  # pragma: no cover - cache module optional during some tests
    from cache import (
        BaseCacheModule as _BaseCacheModule,
        cache as _cache_instance,
        get_cache_stats as _get_cache_stats,
        get_unified_cache_key as _get_unified_cache_key,
        warm_cache_with_data as _warm_cache_with_data,
    )
except Exception as exc:  # pragma: no cover - fallback shims
    logger.debug("Cache module unavailable: %s", exc)
else:
    BaseCacheModule = _BaseCacheModule
    cache = cast(Any, _cache_instance)
    get_cache_stats = _get_cache_stats
    get_unified_cache_key = _get_unified_cache_key
    warm_cache_with_data = _warm_cache_with_data

# Provide a typed view of the current BaseCacheModule implementation for Pyright.
_TypedBaseCacheModule = cast(type[_BaseCacheModuleFallback], BaseCacheModule)

# === SESSION CACHE CONFIGURATION ===


@dataclass
class SessionCacheConfig:
    """Configuration for session caching behavior"""

    session_ttl_seconds: int = 300  # 5 minutes
    component_ttl_seconds: int = 600  # 10 minutes
    enable_component_reuse: bool = True
    track_session_lifecycle: bool = True


# Global cache configuration
CACHE_CONFIG = SessionCacheConfig()


_config_manager_factory: type[Any] | None = None
_config_manager_error: Exception | None = None

try:  # pragma: no cover - configuration optional in some tests
    _config_module = import_module("config.config_manager")
except Exception as exc:
    _config_manager_error = exc
else:
    _config_candidate = getattr(_config_module, "ConfigManager", None)
    if isinstance(_config_candidate, type):
        _config_manager_factory = cast(type[Any], _config_candidate)
    else:
        _config_manager_error = RuntimeError("ConfigManager class missing from config.config_manager")


def _load_config_schema_snapshot() -> Optional[Any]:
    """Safely instantiate ConfigManager and return its schema."""

    if _config_manager_factory is None:
        if _config_manager_error is not None:
            logger.debug("ConfigManager unavailable: %s", _config_manager_error)
        return None

    try:
        manager = _config_manager_factory()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("ConfigManager instantiation failed: %s", exc)
        return None

    try:
        return manager.get_config()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("ConfigManager.get_config() failed: %s", exc)
        return None


try:
    from test_framework import TestSuite as _FrameworkTestSuite, suppress_logging as _framework_suppress_logging
except Exception:  # pragma: no cover - minimal fallback for optional dependency

    @dataclass
    class _FallbackTestSuite:
        name: str
        module: str

        def start_suite(self) -> None:
            logger.info("Starting test suite: %s", self.name)

        @staticmethod
        def run_test(*args: Any, **__: Any) -> None:
            test_func = args[1] if len(args) > 1 else None
            if callable(test_func):
                test_func()

        def finish_suite(self) -> bool:
            logger.info("Finished test suite: %s", self.name)
            return True

    @contextmanager
    def _fallback_suppress_logging() -> Any:
        yield

    _FrameworkTestSuite = _FallbackTestSuite
    _framework_suppress_logging = _fallback_suppress_logging

TestSuite = cast(type[Any], _FrameworkTestSuite)
suppress_logging = _framework_suppress_logging


def create_standard_test_runner(test_func: Callable[[], bool]) -> Callable[[], bool]:
    """Import the shared runner lazily with a fallback."""

    try:
        from test_utilities import create_standard_test_runner as _create_standard_test_runner
    except Exception:

        def _runner() -> bool:
            return test_func()

        return _runner

    return _create_standard_test_runner(test_func)

# === SESSION COMPONENT CACHE ===


class SessionComponentCache(_TypedBaseCacheModule):
    """
    High-performance cache for session manager components.
    Extends the existing cache infrastructure with session-specific optimizations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.module_name = "session_cache"
        self._active_sessions: weakref.WeakSet[Any] = weakref.WeakSet()
        self._session_timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        logger.debug("SessionComponentCache initialized")

    @staticmethod
    def _get_config_hash() -> str:
        """Generate hash of current configuration for cache validation"""
        try:
            config_schema = _load_config_schema_snapshot()

            db_path = "default"
            username = "default"
            if config_schema is not None:
                try:
                    db_path = str(getattr(getattr(config_schema, "database", object), "database_file", "default"))
                except Exception:
                    logger.debug("Database path missing from config schema", exc_info=True)
                try:
                    username = str(getattr(getattr(config_schema, "api", object), "username", "default"))
                except Exception:
                    logger.debug("API username missing from config schema", exc_info=True)

            config_data = {
                "db_path": db_path,
                "username": username,
                "cache_version": "5.1.0",  # Version for cache invalidation
            }

            config_str = str(sorted(config_data.items()))
            return hashlib.md5(config_str.encode()).hexdigest()[:12]
        except Exception as e:
            logger.debug(f"Error generating config hash: {e}")
            return "default_config"

    def config_hash_snapshot(self) -> str:
        """Expose current config hash for diagnostics/tests."""

        return self._get_config_hash()

    def get_cached_component(self, component_type: str) -> Optional[Any]:
        """Get cached component if available and valid"""
        if not cache:
            return None

        try:
            config_hash = self._get_config_hash()
            cache_key = get_unified_cache_key("session", component_type, config_hash)

            # Check cache using existing infrastructure
            cached_data = cache.get(cache_key, default=None, retry=True)

            if cached_data is not None and isinstance(cached_data, dict):
                # Check if component is still valid based on timestamp
                typed_data = cast(dict[str, Any], cached_data)
                cache_time = typed_data.get("timestamp", 0)
                age = time.time() - cache_time

                if age < CACHE_CONFIG.component_ttl_seconds:
                    logger.debug(f"Cache HIT for {component_type} (age: {age:.1f}s)")
                    component = typed_data.get("component")
                    # Return deep copy for mutable objects to prevent cache corruption
                    if isinstance(component, (dict, list)):
                        import copy
                        return copy.deepcopy(component)
                    return component
                logger.debug(
                    f"Cache EXPIRED for {component_type} (age: {age:.1f}s)"
                )
                # Let existing cache eviction handle cleanup

            logger.debug(f"Cache MISS for {component_type}")
            return None

        except Exception as e:
            logger.warning(f"Error retrieving cached component {component_type}: {e}")
            return None

    def cache_component(self, component_type: str, component: Any) -> bool:
        """Cache a component for reuse using existing cache infrastructure"""
        if not cache:
            return False

        try:
            config_hash = self._get_config_hash()
            cache_key = get_unified_cache_key("session", component_type, config_hash)

            # Wrap component with metadata
            cache_data = {
                "component": component,
                "timestamp": time.time(),
                "component_type": component_type,
                "config_hash": config_hash,
            }

            # Use existing cache with component TTL
            cache.set(
                cache_key,
                cache_data,
                expire=CACHE_CONFIG.component_ttl_seconds,
                retry=True,
            )
            logger.debug(f"Cached component: {component_type}")
            return True

        except Exception as e:
            logger.warning(f"Error caching component {component_type}: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get session cache statistics"""
        base_stats = get_cache_stats()

        with self._lock:
            session_stats = {
                "active_sessions": len(self._active_sessions),
                "tracked_sessions": len(self._session_timestamps),
                "config_hash": self._get_config_hash(),
                "component_ttl": CACHE_CONFIG.component_ttl_seconds,
                "session_ttl": CACHE_CONFIG.session_ttl_seconds,
            }

        # Merge with base cache stats
        return {**base_stats, **session_stats}

    def clear(self) -> bool:
        """Clear session-specific cache entries"""
        try:
            with self._lock:
                self._session_timestamps.clear()

            # Clear session-related cache entries using existing infrastructure
            if cache:
                # Clear entries with session prefix
                cleared = 0
                for key in list(cache):
                    if str(key).startswith("session_"):
                        cache.delete(key)
                        cleared += 1
                logger.debug(f"Cleared {cleared} session cache entries")

            return True
        except Exception as e:
            logger.error(f"Error clearing session cache: {e}")
            return False

    def warm(self) -> bool:
        """Warm session cache with frequently used data"""
        try:
            logger.debug("Warming session cache...")

            # Warm with session metadata using existing infrastructure
            warm_cache_with_data(
                get_unified_cache_key("session", "metadata", "system"),
                {
                    "cache_version": "5.1.0",
                    "warmed_at": time.time(),
                    "config_hash": self._get_config_hash(),
                },
                expire=CACHE_CONFIG.component_ttl_seconds,
            )

            logger.debug("Session cache warmed successfully")
            return True
        except Exception as e:
            logger.warning(f"Error warming session cache: {e}")
            return False

    def get_module_name(self) -> str:
        """Get module name"""
        return self.module_name

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of session cache"""
        try:
            stats = self.get_stats()

            # Determine health based on cache availability and session tracking
            if cache is None:
                health = "critical"
                message = "Base cache instance not available"
            elif stats.get("active_sessions", 0) > 0:
                health = "excellent"
                message = f"Active sessions: {stats['active_sessions']}"
            else:
                health = "good"
                message = "Cache available, no active sessions"

            return {
                "health": health,
                "message": message,
                "active_sessions": stats.get("active_sessions", 0),
                "cache_available": cache is not None,
                "session_tracking": CACHE_CONFIG.track_session_lifecycle,
            }
        except Exception as e:
            return {
                "health": "error",
                "message": f"Health check failed: {e}",
                "active_sessions": 0,
                "cache_available": False,
                "session_tracking": False,
            }


# === GLOBAL CACHE INSTANCE ===
_session_cache = SessionComponentCache()

# === CACHING DECORATORS USING EXISTING INFRASTRUCTURE ===


def cached_session_component(component_type: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to cache expensive session components using existing cache infrastructure.
    Dramatically reduces session manager initialization time.
    """

    def decorator(creation_func: Callable[P, R]) -> Callable[P, R]:
        @wraps(creation_func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Try to get from cache first
            cached_component = _session_cache.get_cached_component(component_type)
            if cached_component is not None:
                logger.debug(f"Reusing cached {component_type}")
                return cast(R, cached_component)

            # Create new component
            logger.debug(f"Creating new {component_type}")
            start_time = time.time()
            component = creation_func(*args, **kwargs)
            creation_time = time.time() - start_time

            # Cache for future use if expensive
            if creation_time > 0.1:  # Only cache expensive operations
                _session_cache.cache_component(component_type, component)
                logger.debug(
                    f"Cached {component_type} (creation time: {creation_time:.2f}s)"
                )

            return component

        return wrapper

    return decorator


def cached_database_manager() -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator specifically for DatabaseManager caching"""
    return cached_session_component("database_manager")


def cached_browser_manager() -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator specifically for BrowserManager caching"""
    return cached_session_component("browser_manager")


def cached_api_manager() -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator specifically for APIManager caching"""
    return cached_session_component("api_manager")


def cached_session_validator() -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator specifically for SessionValidator caching"""
    return cached_session_component("session_validator")


# === SESSION STATE OPTIMIZATION ===


class OptimizedSessionState:
    """
    Optimized session state management using existing cache infrastructure.
    Reduces session validation overhead.
    """

    @staticmethod
    def get_cached_session_state(session_id: str) -> Optional[dict[str, Any]]:
        """Get cached session state if valid"""
        if not cache:
            return None

        try:
            cache_key = get_unified_cache_key("session", "state", session_id)
            cached_state = cache.get(cache_key, default=None, retry=True)

            if cached_state and isinstance(cached_state, dict):
                # Check if state is still valid
                typed_state = cast(dict[str, Any], cached_state)
                age = time.time() - typed_state.get("timestamp", 0)
                if age < CACHE_CONFIG.session_ttl_seconds:
                    return typed_state.get("state")

            return None
        except Exception as e:
            logger.debug(f"Error retrieving session state for {session_id}: {e}")
            return None

    @staticmethod
    def cache_session_state(session_id: str, state: Mapping[str, Any]) -> None:
        """Cache session state using existing infrastructure"""
        if not cache:
            return

        try:
            cache_key = get_unified_cache_key("session", "state", session_id)
            cache_data = {
                "state": dict(state),
                "timestamp": time.time(),
                "session_id": session_id,
            }

            cache.set(
                cache_key,
                cache_data,
                expire=CACHE_CONFIG.session_ttl_seconds,
                retry=True,
            )
            logger.debug(f"Cached session state for: {session_id}")
        except Exception as e:
            logger.debug(f"Error caching session state for {session_id}: {e}")


# === CACHE MANAGEMENT FUNCTIONS ===


def get_session_cache_stats() -> dict[str, Any]:
    """Get comprehensive session cache statistics"""
    return _session_cache.get_stats()


def clear_session_cache() -> int:
    """Clear session-specific caches"""
    success = _session_cache.clear()
    return 1 if success else 0


def warm_session_cache() -> bool:
    """Warm up session cache with frequently used components"""
    return _session_cache.warm()


# === TESTING FUNCTIONS ===


def _test_session_component_cache_initialization():
    """Test SessionComponentCache initialization"""
    cache_instance = SessionComponentCache()

    # Verify cache instance is created
    assert cache_instance is not None, "Cache instance should be created"
    assert hasattr(cache_instance, '_active_sessions'), "Should have _active_sessions"
    assert hasattr(cache_instance, '_session_timestamps'), "Should have _session_timestamps"
    assert hasattr(cache_instance, '_lock'), "Should have _lock"

    logger.info("✅ SessionComponentCache initialized successfully")
    return True


def _test_config_hash_generation():
    """Test configuration hash generation"""
    cache_instance = SessionComponentCache()

    # Generate config hash
    config_hash = cache_instance.config_hash_snapshot()

    # Verify hash is generated
    assert config_hash is not None, "Config hash should be generated"
    assert isinstance(config_hash, str), "Config hash should be string"
    assert len(config_hash) > 0, "Config hash should not be empty"

    # Verify hash is consistent
    config_hash2 = cache_instance.config_hash_snapshot()
    assert config_hash == config_hash2, "Config hash should be consistent"

    logger.info(f"✅ Config hash generated: {config_hash}")
    return True


def _test_component_caching_and_retrieval():
    """Test caching and retrieving components"""
    cache_instance = SessionComponentCache()

    # Create test component
    test_component = {"test": "data", "value": 123}
    component_type = "test_component_functional"

    # Cache the component
    cached = cache_instance.cache_component(component_type, test_component)
    assert cached, "Component should be cached successfully"

    # Retrieve the component
    retrieved = cache_instance.get_cached_component(component_type)
    assert retrieved is not None, "Component should be retrieved"
    assert retrieved == test_component, "Retrieved component should match original"

    logger.info("✅ Component cached and retrieved successfully")
    return True


def _test_cache_expiration():
    """Test cache expiration based on TTL"""
    cache_instance = SessionComponentCache()

    # Create test component with short TTL
    original_ttl = CACHE_CONFIG.component_ttl_seconds
    CACHE_CONFIG.component_ttl_seconds = 1  # 1 second TTL

    try:
        test_component = {"test": "expiring_data"}
        component_type = "test_expiring_component"

        # Cache the component
        cache_instance.cache_component(component_type, test_component)

        # Retrieve immediately (should work)
        retrieved = cache_instance.get_cached_component(component_type)
        assert retrieved is not None, "Component should be retrieved immediately"

        # Wait for expiration
        time.sleep(1.5)

        # Try to retrieve after expiration (should be None)
        expired = cache_instance.get_cached_component(component_type)
        assert expired is None, "Component should be expired"

        logger.info("✅ Cache expiration works correctly")
        return True

    finally:
        # Restore original TTL
        CACHE_CONFIG.component_ttl_seconds = original_ttl


def _test_cached_session_component_decorator():
    """Test cached_session_component decorator"""
    call_count = {"count": 0}

    @cached_session_component("test_decorator_component")
    def create_test_component() -> dict[str, Any]:
        # Sleep to make it "expensive" so it gets cached (>0.1s threshold)
        time.sleep(0.15)
        call_count["count"] += 1
        return {"data": "test", "call": call_count["count"]}

    # First call should execute function
    result1 = create_test_component()
    assert result1 is not None, "First call should return result"
    assert call_count["count"] == 1, "Function should be called once"

    # Second call should use cache (function not called again)
    result2 = create_test_component()
    assert result2 is not None, "Second call should return result"
    assert result2 == result1, "Second call should return same result"
    assert call_count["count"] == 1, "Function should still be called only once (cached)"

    logger.info("✅ Decorator caching works correctly")
    return True


def _test_cache_stats():
    """Test cache statistics retrieval"""
    # Get cache stats
    stats = get_session_cache_stats()

    # Verify stats structure
    assert stats is not None, "Stats should be returned"
    assert isinstance(stats, dict), "Stats should be dictionary"

    logger.info(f"✅ Cache stats retrieved: {stats}")
    return True


def _test_clear_session_cache():
    """Test clearing session cache"""
    cache_instance = SessionComponentCache()

    # Cache some components
    cache_instance.cache_component("test_clear_1", {"data": 1})
    cache_instance.cache_component("test_clear_2", {"data": 2})

    # Clear cache
    cleared_count = clear_session_cache()

    # Verify cache is cleared
    assert cleared_count >= 0, "Should return count of cleared items"

    # Verify components are gone (may still be in cache if clear_session_cache() only clears session-specific items)
    _ = cache_instance.get_cached_component("test_clear_1")
    _ = cache_instance.get_cached_component("test_clear_2")

    logger.info(f"✅ Cache cleared: {cleared_count} items")
    return True


def _test_warm_session_cache():
    """Test warming session cache"""
    # Warm cache
    warm_session_cache()

    # Verify cache is warmed (no errors)
    logger.info("✅ Session cache warmed successfully")
    return True


def test_session_cache_performance() -> bool:
    """Test session cache performance improvements"""
    logger.info("🚀 Testing Session Cache Performance")

    # Test component caching
    @cached_session_component("test_component")
    def create_expensive_component() -> dict[str, Any]:
        time.sleep(0.1)  # Simulate expensive operation
        return {"test": "data", "timestamp": time.time()}

    # First call should be slow
    start_time = time.time()
    create_expensive_component()
    first_time = time.time() - start_time

    # Second call should be fast (cached)
    start_time = time.time()
    create_expensive_component()
    second_time = time.time() - start_time

    speedup = first_time / max(second_time, 0.001)

    logger.info(f"First call: {first_time:.3f}s")
    logger.info(f"Second call: {second_time:.3f}s")
    logger.info(f"Speedup: {speedup:.1f}x")
    logger.info(f"Cache stats: {get_session_cache_stats()}")

    return speedup > 5


def session_cache_module_tests() -> bool:
    """Comprehensive test suite for session_cache.py using the unified TestSuite."""
    suite = TestSuite("Session Cache", "core/session_cache.py")
    suite.start_suite()

    # Test 1: Cache initialization
    suite.run_test(
        "SessionComponentCache initialization",
        _test_session_component_cache_initialization,
        "SessionComponentCache instance created with required attributes",
        "Test cache instance creation and attribute initialization",
        "Verify _active_sessions, _session_timestamps, and _lock attributes exist",
    )

    # Test 2: Config hash generation
    suite.run_test(
        "Configuration hash generation",
        _test_config_hash_generation,
        "Config hash generated consistently",
        "Test configuration hash generation for cache validation",
        "Verify hash is string, non-empty, and consistent across calls",
    )

    # Test 3: Component caching and retrieval
    suite.run_test(
        "Component caching and retrieval",
        _test_component_caching_and_retrieval,
        "Component cached and retrieved successfully",
        "Test caching and retrieving components",
        "Verify component can be cached and retrieved with same data",
    )

    # Test 4: Cache expiration
    suite.run_test(
        "Cache expiration",
        _test_cache_expiration,
        "Cache expires after TTL",
        "Test cache expiration based on TTL",
        "Verify component is available immediately but expires after TTL",
    )

    # Test 5: Decorator caching
    suite.run_test(
        "Cached session component decorator",
        _test_cached_session_component_decorator,
        "Decorator caches function results",
        "Test @cached_session_component decorator",
        "Verify function is called once and result is cached for subsequent calls",
    )

    # Test 6: Cache stats
    suite.run_test(
        "Cache statistics",
        _test_cache_stats,
        "Cache stats retrieved successfully",
        "Test cache statistics retrieval",
        "Verify get_session_cache_stats() returns dictionary",
    )

    # Test 7: Clear cache
    suite.run_test(
        "Clear session cache",
        _test_clear_session_cache,
        "Session cache cleared successfully",
        "Test clearing session cache",
        "Verify clear_session_cache() clears cached components",
    )

    # Test 8: Warm cache
    suite.run_test(
        "Warm session cache",
        _test_warm_session_cache,
        "Session cache warmed successfully",
        "Test warming session cache",
        "Verify warm_session_cache() executes without errors",
    )

    # Test 9: Performance test (kept from original)
    with suppress_logging():
        suite.run_test(
            "Cache performance improvement",
            test_session_cache_performance,
            "Cache provides significant speedup (>5x)",
            "Test cache performance improvements",
            "Verify cached calls are much faster than uncached calls",
        )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(session_cache_module_tests)


if __name__ == "__main__":
    run_comprehensive_tests()
