#!/usr/bin/env python3

"""
Session Cache - High-Performance Session State Caching

This module provides intelligent caching for session manager components to dramatically
reduce initialization overhead. Addresses the 34.59s session manager bottleneck by
implementing persistent session state and component reuse.

Extends the existing cache.py infrastructure rather than duplicating functionality.
"""

# === CORE INFRASTRUCTURE ===
import os
import sys

# Add parent directory to path for standard_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# === STANDARD LIBRARY IMPORTS ===
import hashlib
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# === LEVERAGE EXISTING CACHE INFRASTRUCTURE ===
from cache import (
    BaseCacheModule,  # Base cache interface
    cache,  # Global cache instance
    cache_result,  # Existing cache decorator
    clear_cache,  # Cache clearing
    get_cache_stats,  # Statistics
    get_unified_cache_key,  # Unified key generation
    warm_cache_with_data,  # Cache warming
)
from error_handling import (
    circuit_breaker,
    error_context,
    graceful_degradation,
    retry_on_failure,
    timeout_protection,
)

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

# === SESSION COMPONENT CACHE ===


class SessionComponentCache(BaseCacheModule):
    """
    High-performance cache for session manager components.
    Extends the existing cache infrastructure with session-specific optimizations.
    """

    def __init__(self):
        self._active_sessions = weakref.WeakSet()
        self._session_timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        logger.debug("SessionComponentCache initialized")

    def _get_config_hash(self) -> str:
        """Generate hash of current configuration for cache validation"""
        try:
            from config.config_manager import ConfigManager

            config_manager = ConfigManager()
            config_schema = config_manager.get_config()

            # Create hash from relevant config values
            config_data = {
                "db_path": (
                    str(config_schema.database.database_file)
                    if config_schema
                    else "default"
                ),
                "username": config_schema.api.username if config_schema else "default",
                "cache_version": "5.1.0",  # Version for cache invalidation
            }

            config_str = str(sorted(config_data.items()))
            return hashlib.md5(config_str.encode()).hexdigest()[:12]
        except Exception as e:
            logger.debug(f"Error generating config hash: {e}")
            return "default_config"

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
                cache_time = cached_data.get("timestamp", 0)
                age = time.time() - cache_time

                if age < CACHE_CONFIG.component_ttl_seconds:
                    logger.debug(f"Cache HIT for {component_type} (age: {age:.1f}s)")
                    return cached_data.get("component")
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

    def get_stats(self) -> Dict[str, Any]:
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
        return "session_cache"

    def get_health_status(self) -> Dict[str, Any]:
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


def cached_session_component(component_type: str):
    """
    Decorator to cache expensive session components using existing cache infrastructure.
    Dramatically reduces session manager initialization time.
    """

    def decorator(creation_func):
        def wrapper(*args, **kwargs):
            # Try to get from cache first
            cached_component = _session_cache.get_cached_component(component_type)
            if cached_component is not None:
                logger.debug(f"Reusing cached {component_type}")
                return cached_component

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


def cached_database_manager():
    """Decorator specifically for DatabaseManager caching"""
    return cached_session_component("database_manager")


def cached_browser_manager():
    """Decorator specifically for BrowserManager caching"""
    return cached_session_component("browser_manager")


def cached_api_manager():
    """Decorator specifically for APIManager caching"""
    return cached_session_component("api_manager")


def cached_session_validator():
    """Decorator specifically for SessionValidator caching"""
    return cached_session_component("session_validator")


# === SESSION STATE OPTIMIZATION ===


class OptimizedSessionState:
    """
    Optimized session state management using existing cache infrastructure.
    Reduces session validation overhead.
    """

    def get_cached_session_state(self, session_id: str) -> Optional[Dict]:
        """Get cached session state if valid"""
        if not cache:
            return None

        try:
            cache_key = get_unified_cache_key("session", "state", session_id)
            cached_state = cache.get(cache_key, default=None, retry=True)

            if cached_state and isinstance(cached_state, dict):
                # Check if state is still valid
                age = time.time() - cached_state.get("timestamp", 0)
                if age < CACHE_CONFIG.session_ttl_seconds:
                    return cached_state.get("state")

            return None
        except Exception as e:
            logger.debug(f"Error retrieving session state for {session_id}: {e}")
            return None

    def cache_session_state(self, session_id: str, state: Dict):
        """Cache session state using existing infrastructure"""
        if not cache:
            return

        try:
            cache_key = get_unified_cache_key("session", "state", session_id)
            cache_data = {
                "state": state.copy(),
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


def get_session_cache_stats() -> Dict[str, Any]:
    """Get comprehensive session cache statistics"""
    return _session_cache.get_stats()


def clear_session_cache() -> int:
    """Clear session-specific caches"""
    success = _session_cache.clear()
    return 1 if success else 0


def warm_session_cache():
    """Warm up session cache with frequently used components"""
    return _session_cache.warm()


# === TESTING FUNCTIONS ===


def test_session_cache_performance():
    """Test session cache performance improvements"""
    logger.info("🚀 Testing Session Cache Performance")

    # Test component caching
    @cached_session_component("test_component")
    def create_expensive_component():
        time.sleep(0.1)  # Simulate expensive operation
        return {"test": "data", "timestamp": time.time()}

    # First call should be slow
    start_time = time.time()
    result1 = create_expensive_component()
    first_time = time.time() - start_time

    # Second call should be fast (cached)
    start_time = time.time()
    result2 = create_expensive_component()
    second_time = time.time() - start_time

    speedup = first_time / max(second_time, 0.001)

    logger.info(f"First call: {first_time:.3f}s")
    logger.info(f"Second call: {second_time:.3f}s")
    logger.info(f"Speedup: {speedup:.1f}x")
    logger.info(f"Cache stats: {get_session_cache_stats()}")

    return speedup > 5  # Should be much faster


if __name__ == "__main__":
    # === COMPREHENSIVE SESSION CACHE TESTING ===
    print("🚀 Session Cache - Phase 5.1 Optimization Test")
    print("=" * 60)

    # Test 1: Basic cache functionality
    print("\n📋 Test 1: Basic Cache Performance")
    success1 = test_session_cache_performance()
    print(f"Result: {'✅ PASS' if success1 else '❌ FAIL'}")

    # Test 2: Component caching with real session manager
    print("\n� Test 2: Session Manager Integration")
    try:
        import time

        from core.session_manager import SessionManager

        times = []
        for i in range(3):
            start = time.time()
            sm = SessionManager()
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Initialization {i+1}: {elapsed:.3f}s")

        avg_time = sum(times) / len(times)
        success2 = avg_time < 1.0  # Should be under 1 second with caching
        print(f"  Average: {avg_time:.3f}s")
        print(f"Result: {'✅ PASS' if success2 else '❌ FAIL'} (Target: <1.0s)")

    except Exception as e:
        print(f"  Error: {e}")
        success2 = False
        print("Result: ❌ FAIL")

    # Test 3: Cache statistics
    print("\n📋 Test 3: Cache Health & Statistics")
    stats = get_session_cache_stats()
    health = _session_cache.get_health_status()

    print(f"  Cache hit rate: {stats.get('hit_rate', 0):.1f}%")
    print(f"  Cache entries: {stats.get('entries', 0)}")
    print(f"  Health status: {health.get('health', 'unknown')}")

    success3 = (
        stats.get('hit_rate', 0) > 0 and
        health.get('health') in ['good', 'excellent']
    )
    print(f"Result: {'✅ PASS' if success3 else '❌ FAIL'}")

    # Overall results
    print("\n🎯 Phase 5.1 Optimization Summary:")
    print("=" * 60)
    all_passed = success1 and success2 and success3
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"Cache Performance: {'✅ OPTIMIZED' if success1 else '❌ NEEDS WORK'}")
    print(f"Session Integration: {'✅ WORKING' if success2 else '❌ ISSUES'}")
    print(f"Cache Health: {'✅ HEALTHY' if success3 else '❌ DEGRADED'}")

    # Performance target validation
    if success2:
        avg_time = sum(times) / len(times) if 'times' in locals() else 0
        if avg_time < 0.1:
            print("🚀 EXCELLENT: Sub-100ms session initialization!")
        elif avg_time < 0.5:
            print("🚀 GREAT: Sub-500ms session initialization!")
        else:
            print("✅ GOOD: Under 1s session initialization")

    print(f"\nDetailed Cache Stats: {stats}")
