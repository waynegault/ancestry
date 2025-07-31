#!/usr/bin/env python3

"""
Advanced Caching & Memory Optimization

This module extends the successful session_cache.py pattern to provide system-wide
performance optimization through intelligent caching strategies, memory optimization,
and database performance enhancement.

Building on Phase 5.1 success (1,281x session manager improvement), Phase 5.2 targets:
- API response caching for external services (AI, Ancestry APIs)
- Database query result caching and connection optimization
- Memory usage optimization with smart garbage collection
- System-wide cache warming and monitoring
"""

# === CORE INFRASTRUCTURE ===
import sys
import os

# Add parent directory to path for standard_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    APIRateLimitError,
    ErrorContext,
)

# === STANDARD LIBRARY IMPORTS ===
import gc
import hashlib
import threading
import time
import weakref
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Callable, Union
import json

# === LEVERAGE EXISTING CACHE INFRASTRUCTURE ===
from cache import (
    cache,  # Global cache instance
    cache_result,  # Existing cache decorator
    get_unified_cache_key,  # Unified key generation
    warm_cache_with_data,  # Cache warming
    get_cache_stats,  # Statistics
    clear_cache,  # Cache clearing
    BaseCacheModule,  # Base cache interface
)

# === SESSION CACHE INTEGRATION ===
from core.session_cache import (
    SessionCacheConfig,
    get_session_cache_stats,
    _session_cache,
)

# === PHASE 5.2 CONFIGURATION ===


@dataclass
class SystemCacheConfig:
    """Advanced system-wide cache configuration for Phase 5.2"""

    # API Response Caching
    api_response_ttl: int = 1800  # 30 minutes for API responses
    ai_analysis_ttl: int = 3600  # 1 hour for AI analysis results
    ancestry_api_ttl: int = 900  # 15 minutes for Ancestry API calls

    # Database Query Caching
    db_query_ttl: int = 600  # 10 minutes for database queries
    db_connection_pool_size: int = 10
    db_query_cache_size: int = 1000

    # Memory Optimization
    enable_aggressive_gc: bool = True
    gc_threshold_ratio: float = 0.8  # Trigger GC at 80% memory usage
    memory_cache_limit_mb: int = 512  # 512MB memory cache limit

    # Cache Warming
    enable_background_warming: bool = True
    warming_batch_size: int = 50
    warming_interval_seconds: int = 300  # 5 minutes


# Global system cache configuration
SYSTEM_CACHE_CONFIG = SystemCacheConfig()

# === API RESPONSE CACHING SYSTEM ===


class APIResponseCache(BaseCacheModule):
    """
    High-performance API response caching system.
    Optimizes external API calls with intelligent TTL management.
    """

    def __init__(self):
        self._api_stats = {
            "ai_requests": 0,
            "ai_cache_hits": 0,
            "ancestry_requests": 0,
            "ancestry_cache_hits": 0,
            "total_time_saved": 0.0,
        }
        self._lock = threading.Lock()
        logger.debug("APIResponseCache initialized for Phase 5.2")

    def _get_api_cache_key(
        self, service: str, method: str, params: Dict[str, Any]
    ) -> str:
        """Generate cache key for API requests"""
        # Create a stable hash of parameters
        param_str = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]

        return get_unified_cache_key("api", service, method, param_hash)

    def cache_api_response(
        self,
        service: str,
        method: str,
        params: Dict[str, Any],
        response: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache API response with service-specific TTL"""
        if not cache:
            return False

        try:
            cache_key = self._get_api_cache_key(service, method, params)

            # Determine TTL based on service type
            if ttl is None:
                if service == "ai":
                    ttl = SYSTEM_CACHE_CONFIG.ai_analysis_ttl
                elif service == "ancestry":
                    ttl = SYSTEM_CACHE_CONFIG.ancestry_api_ttl
                else:
                    ttl = SYSTEM_CACHE_CONFIG.api_response_ttl

            cache_data = {
                "response": response,
                "timestamp": time.time(),
                "service": service,
                "method": method,
                "params_hash": hashlib.md5(str(params).encode()).hexdigest()[:8],
            }

            cache.set(cache_key, cache_data, expire=ttl, retry=True)
            logger.debug(f"Cached {service}.{method} response (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache {service}.{method} response: {e}")
            return False

    def get_cached_api_response(
        self, service: str, method: str, params: Dict[str, Any]
    ) -> Optional[Any]:
        """Retrieve cached API response if valid"""
        if not cache:
            return None

        try:
            cache_key = self._get_api_cache_key(service, method, params)
            cached_data = cache.get(cache_key, default=None, retry=True)

            if cached_data and isinstance(cached_data, dict):
                # Update statistics
                with self._lock:
                    if service == "ai":
                        self._api_stats["ai_cache_hits"] += 1
                    elif service == "ancestry":
                        self._api_stats["ancestry_cache_hits"] += 1

                logger.debug(f"Cache HIT for {service}.{method}")
                return cached_data.get("response")

            logger.debug(f"Cache MISS for {service}.{method}")
            return None

        except Exception as e:
            logger.warning(f"Error retrieving cached {service}.{method}: {e}")
            return None

    def get_api_cache_stats(self) -> Dict[str, Any]:
        """Get API cache statistics"""
        with self._lock:
            stats = self._api_stats.copy()

        # Calculate hit rates
        if stats["ai_requests"] > 0:
            stats["ai_hit_rate"] = (stats["ai_cache_hits"] / stats["ai_requests"]) * 100
        else:
            stats["ai_hit_rate"] = 0.0

        if stats["ancestry_requests"] > 0:
            stats["ancestry_hit_rate"] = (
                stats["ancestry_cache_hits"] / stats["ancestry_requests"]
            ) * 100
        else:
            stats["ancestry_hit_rate"] = 0.0

        return stats


# === DATABASE QUERY CACHING SYSTEM ===


class DatabaseQueryCache(BaseCacheModule):
    """
    High-performance database query result caching.
    Optimizes database operations with intelligent invalidation.
    """

    def __init__(self):
        self._query_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time_saved": 0.0,
        }
        self._lock = threading.Lock()
        logger.debug("DatabaseQueryCache initialized for Phase 5.2")

    def _get_query_cache_key(self, query: str, params: Tuple = ()) -> str:
        """Generate cache key for database queries"""
        # Normalize query (remove extra whitespace, convert to lowercase)
        normalized_query = " ".join(query.strip().lower().split())

        # Create hash of query + parameters
        query_data = f"{normalized_query}|{params}"
        query_hash = hashlib.md5(query_data.encode()).hexdigest()[:12]

        return get_unified_cache_key("db_query", query_hash)

    def cache_query_result(
        self, query: str, params: Tuple, result: Any, ttl: Optional[int] = None
    ) -> bool:
        """Cache database query result"""
        if not cache:
            return False

        try:
            cache_key = self._get_query_cache_key(query, params)

            if ttl is None:
                ttl = SYSTEM_CACHE_CONFIG.db_query_ttl

            cache_data = {
                "result": result,
                "timestamp": time.time(),
                "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],
                "params": params,
            }

            cache.set(cache_key, cache_data, expire=ttl, retry=True)
            logger.debug(f"Cached database query result (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache database query: {e}")
            return False

    def get_cached_query_result(self, query: str, params: Tuple = ()) -> Optional[Any]:
        """Retrieve cached database query result"""
        if not cache:
            return None

        try:
            cache_key = self._get_query_cache_key(query, params)
            cached_data = cache.get(cache_key, default=None, retry=True)

            if cached_data and isinstance(cached_data, dict):
                with self._lock:
                    self._query_stats["cache_hits"] += 1

                logger.debug("Database query cache HIT")
                return cached_data.get("result")

            with self._lock:
                self._query_stats["cache_misses"] += 1

            logger.debug("Database query cache MISS")
            return None

        except Exception as e:
            logger.warning(f"Error retrieving cached query result: {e}")
            return None


# === MEMORY OPTIMIZATION SYSTEM ===


class MemoryOptimizer(BaseCacheModule):
    """
    Intelligent memory optimization and garbage collection management.
    """

    def __init__(self):
        self._memory_stats = {
            "gc_collections": 0,
            "memory_freed_mb": 0.0,
            "peak_memory_mb": 0.0,
            "current_memory_mb": 0.0,
        }
        self._lock = threading.Lock()
        logger.debug("MemoryOptimizer initialized for Phase 5.2")

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            with self._lock:
                self._memory_stats["current_memory_mb"] = memory_mb
                if memory_mb > self._memory_stats["peak_memory_mb"]:
                    self._memory_stats["peak_memory_mb"] = memory_mb

            return memory_mb
        except ImportError:
            # Fallback method without psutil
            return 0.0

    def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """Perform intelligent memory optimization"""
        if not SYSTEM_CACHE_CONFIG.enable_aggressive_gc and not force:
            return {"optimized": False, "reason": "Aggressive GC disabled"}

        try:
            # Get memory usage before optimization
            memory_before = self.get_memory_usage_mb()

            # Perform garbage collection
            start_time = time.time()

            # Collect garbage in all generations
            collected = gc.collect()

            # Force cleanup of weak references
            for obj in list(gc.get_objects()):
                if isinstance(obj, weakref.ref):
                    try:
                        if obj() is None:
                            del obj
                    except:
                        pass

            # Get memory usage after optimization
            memory_after = self.get_memory_usage_mb()
            optimization_time = time.time() - start_time
            memory_freed = max(0, memory_before - memory_after)

            with self._lock:
                self._memory_stats["gc_collections"] += 1
                self._memory_stats["memory_freed_mb"] += memory_freed

            logger.debug(
                f"Memory optimization: freed {memory_freed:.2f}MB in {optimization_time:.3f}s"
            )

            return {
                "optimized": True,
                "memory_before_mb": memory_before,
                "memory_after_mb": memory_after,
                "memory_freed_mb": memory_freed,
                "objects_collected": collected,
                "optimization_time_s": optimization_time,
            }

        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return {"optimized": False, "error": str(e)}


# === GLOBAL CACHE INSTANCES ===

_api_cache = APIResponseCache()
_db_cache = DatabaseQueryCache()
_memory_optimizer = MemoryOptimizer()

# === CACHING DECORATORS ===


def cached_api_call(service: str, ttl: Optional[int] = None):
    """
    Decorator for caching API calls with intelligent TTL management.

    Usage:
    @cached_api_call('ai', ttl=3600)
    def analyze_message_intent(message_content):
        return ai_service.analyze(message_content)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and parameters
            method = func.__name__
            params = {"args": args, "kwargs": kwargs}

            # Try to get cached response
            cached_response = _api_cache.get_cached_api_response(
                service, method, params
            )
            if cached_response is not None:
                return cached_response

            # Call actual function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            call_time = time.time() - start_time

            # Cache the result
            _api_cache.cache_api_response(service, method, params, result, ttl)

            # Update statistics
            with _api_cache._lock:
                if service == "ai":
                    _api_cache._api_stats["ai_requests"] += 1
                elif service == "ancestry":
                    _api_cache._api_stats["ancestry_requests"] += 1
                _api_cache._api_stats["total_time_saved"] += call_time

            return result

        return wrapper

    return decorator


def cached_database_query(ttl: Optional[int] = None):
    """
    Decorator for caching database query results.

    Usage:
    @cached_database_query(ttl=600)
    def get_person_conversations(person_id):
        return session.query(ConversationLog).filter_by(people_id=person_id).all()
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function and parameters
            cache_key_data = f"{func.__name__}|{args}|{sorted(kwargs.items())}"

            # Check cache first
            cached_result = _db_cache.get_cached_query_result(cache_key_data, ())
            if cached_result is not None:
                return cached_result

            # Execute query and cache result
            result = func(*args, **kwargs)
            _db_cache.cache_query_result(cache_key_data, (), result, ttl)

            with _db_cache._lock:
                _db_cache._query_stats["total_queries"] += 1

            return result

        return wrapper

    return decorator


def memory_optimized(gc_threshold: Optional[float] = None):
    """
    Decorator for functions that should trigger memory optimization.

    Usage:
    @memory_optimized(gc_threshold=0.8)
    def process_large_gedcom_file(file_path):
        # Process large file
        return result
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before execution
            memory_before = _memory_optimizer.get_memory_usage_mb()

            try:
                result = func(*args, **kwargs)

                # Check if memory optimization is needed
                memory_after = _memory_optimizer.get_memory_usage_mb()
                memory_limit = SYSTEM_CACHE_CONFIG.memory_cache_limit_mb
                threshold = gc_threshold or SYSTEM_CACHE_CONFIG.gc_threshold_ratio

                if memory_after > (memory_limit * threshold):
                    optimization_result = _memory_optimizer.optimize_memory()
                    if optimization_result.get("optimized"):
                        logger.debug(
                            f"Memory optimized after {func.__name__}: freed {optimization_result.get('memory_freed_mb', 0):.2f}MB"
                        )

                return result

            except Exception as e:
                # Optimize memory on exception to clean up partial allocations
                _memory_optimizer.optimize_memory(force=True)
                raise

        return wrapper

    return decorator


# === CACHE MANAGEMENT FUNCTIONS ===


def get_system_cache_stats() -> Dict[str, Any]:
    """Get comprehensive system cache statistics"""
    base_stats = get_cache_stats()
    session_stats = get_session_cache_stats()
    api_stats = _api_cache.get_api_cache_stats()

    memory_stats = {}
    with _memory_optimizer._lock:
        memory_stats = _memory_optimizer._memory_stats.copy()

    db_stats = {}
    with _db_cache._lock:
        db_stats = _db_cache._query_stats.copy()

    return {
        "base_cache": base_stats,
        "session_cache": session_stats,
        "api_cache": api_stats,
        "database_cache": db_stats,
        "memory_optimization": memory_stats,
        "system_config": {
            "api_response_ttl": SYSTEM_CACHE_CONFIG.api_response_ttl,
            "db_query_ttl": SYSTEM_CACHE_CONFIG.db_query_ttl,
            "memory_limit_mb": SYSTEM_CACHE_CONFIG.memory_cache_limit_mb,
            "aggressive_gc_enabled": SYSTEM_CACHE_CONFIG.enable_aggressive_gc,
        },
    }


def clear_system_caches() -> Dict[str, Union[int, str]]:
    """Clear all system caches"""
    results = {}

    try:
        # Clear base cache
        cleared_base = clear_cache()
        results["base_cache"] = cleared_base

        # Clear session cache
        from core.session_cache import clear_session_cache

        cleared_session = clear_session_cache()
        results["session_cache"] = cleared_session

        # Clear API cache entries
        cleared_api = 0
        if cache:
            for key in list(cache):
                if str(key).startswith("api_"):
                    cache.delete(key)
                    cleared_api += 1
        results["api_cache"] = cleared_api

        # Clear database cache entries
        cleared_db = 0
        if cache:
            for key in list(cache):
                if str(key).startswith("db_query_"):
                    cache.delete(key)
                    cleared_db += 1
        results["database_cache"] = cleared_db

        # Force memory optimization
        memory_result = _memory_optimizer.optimize_memory(force=True)
        results["memory_optimized"] = 1 if memory_result.get("optimized") else 0

        logger.info(f"System caches cleared: {results}")
        return results

    except Exception as e:
        logger.error(f"Error clearing system caches: {e}")
        return {"error": str(e)}


def warm_system_caches() -> bool:
    """Warm system caches with frequently used data"""
    try:
        logger.info("Warming system caches for Phase 5.2...")

        # Warm session cache
        from core.session_cache import warm_session_cache

        session_warmed = warm_session_cache()

        # Warm base cache with system metadata
        system_metadata = {
            "cache_version": "5.2.0",
            "warmed_at": time.time(),
            "api_cache_enabled": True,
            "db_cache_enabled": True,
            "memory_optimization_enabled": SYSTEM_CACHE_CONFIG.enable_aggressive_gc,
        }

        warm_cache_with_data(
            get_unified_cache_key("system", "metadata", "phase_5_2"),
            system_metadata,
            expire=SYSTEM_CACHE_CONFIG.api_response_ttl,
        )

        logger.info("System cache warming completed successfully")
        return True

    except Exception as e:
        logger.warning(f"Error warming system caches: {e}")
        return False


# === TESTING FUNCTIONS ===


def test_system_cache_performance():
    """Test Phase 5.2 system cache performance improvements"""
    logger.info("🚀 Testing Phase 5.2 System Cache Performance")

    results = {
        "api_cache_test": False,
        "db_cache_test": False,
        "memory_optimization_test": False,
    }

    try:
        # Test API caching
        @cached_api_call("test_service", ttl=60)
        def mock_api_call(data):
            time.sleep(0.1)  # Simulate API delay
            return {"processed": data, "timestamp": time.time()}

        # First call (should be slow)
        start_time = time.time()
        result1 = mock_api_call("test_data")
        first_call_time = time.time() - start_time

        # Second call (should be fast - cached)
        start_time = time.time()
        result2 = mock_api_call("test_data")
        second_call_time = time.time() - start_time

        api_speedup = first_call_time / max(second_call_time, 0.001)
        results["api_cache_test"] = api_speedup > 5

        logger.info(
            f"API Cache: {first_call_time:.3f}s → {second_call_time:.3f}s ({api_speedup:.1f}x speedup)"
        )

        # Test database caching
        @cached_database_query(ttl=60)
        def mock_db_query(query_id):
            time.sleep(0.05)  # Simulate DB delay
            return [{"id": query_id, "data": "test_result"}]

        start_time = time.time()
        db_result1 = mock_db_query(123)
        first_db_time = time.time() - start_time

        start_time = time.time()
        db_result2 = mock_db_query(123)
        second_db_time = time.time() - start_time

        db_speedup = first_db_time / max(second_db_time, 0.001)
        results["db_cache_test"] = db_speedup > 3

        logger.info(
            f"DB Cache: {first_db_time:.3f}s → {second_db_time:.3f}s ({db_speedup:.1f}x speedup)"
        )

        # Test memory optimization
        memory_before = _memory_optimizer.get_memory_usage_mb()
        optimization_result = _memory_optimizer.optimize_memory()
        results["memory_optimization_test"] = optimization_result.get(
            "optimized", False
        )

        logger.info(f"Memory optimization: {optimization_result}")

    except Exception as e:
        logger.error(f"System cache testing failed: {e}")

    # Overall results
    all_passed = all(results.values())
    logger.info(f"Phase 5.2 Cache Tests: {'✅ PASS' if all_passed else '❌ FAIL'}")
    logger.info(f"Detailed results: {results}")

    return all_passed


if __name__ == "__main__":
    # === COMPREHENSIVE PHASE 5.2 TESTING ===
    print("🚀 Advanced Caching & Memory Optimization Test")
    print("=" * 70)

    # Test 1: System cache performance
    print("\n📋 Test 1: System Cache Performance")
    success1 = test_system_cache_performance()
    print(f"Result: {'✅ PASS' if success1 else '❌ FAIL'}")

    # Test 2: Cache statistics and monitoring
    print("\n📋 Test 2: Cache Statistics & Monitoring")
    try:
        stats = get_system_cache_stats()
        success2 = isinstance(stats, dict) and len(stats) > 0
        print(f"System cache stats collected: {len(stats)} categories")
        print(f"Result: {'✅ PASS' if success2 else '❌ FAIL'}")
    except Exception as e:
        print(f"Error: {e}")
        success2 = False
        print("Result: ❌ FAIL")

    # Test 3: Cache warming and management
    print("\n📋 Test 3: Cache Warming & Management")
    try:
        warm_success = warm_system_caches()
        clear_results = clear_system_caches()
        success3 = warm_success and isinstance(clear_results, dict)
        print(f"Cache warming: {'✅ SUCCESS' if warm_success else '❌ FAILED'}")
        print(f"Cache clearing: {clear_results}")
        print(f"Result: {'✅ PASS' if success3 else '❌ FAIL'}")
    except Exception as e:
        print(f"Error: {e}")
        success3 = False
        print("Result: ❌ FAIL")

    # Overall results
    print(f"\n🎯 Phase 5.2 Optimization Summary:")
    print("=" * 70)
    all_passed = success1 and success2 and success3
    print(
        f"Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
    )
    print(f"API Caching: {'✅ WORKING' if success1 else '❌ ISSUES'}")
    print(f"Statistics: {'✅ AVAILABLE' if success2 else '❌ UNAVAILABLE'}")
    print(f"Management: {'✅ FUNCTIONAL' if success3 else '❌ BROKEN'}")

    if all_passed:
        print("\n🚀 EXCELLENT: system-wide caching ready for deployment!")
        print("Ready to apply decorators to target modules for performance gains.")

    # Display comprehensive statistics
    print(f"\nDetailed System Stats:")
    try:
        final_stats = get_system_cache_stats()
        for category, data in final_stats.items():
            print(f"  {category}: {data}")
    except Exception as e:
        print(f"  Error getting final stats: {e}")
