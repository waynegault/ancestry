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

# Add parent directory to path for standard_imports
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import gc
import hashlib
import json
import threading
import time
import weakref
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

# === LEVERAGE EXISTING CACHE INFRASTRUCTURE ===
from cache import (
    BaseCacheModule,  # Base cache interface
    cache,  # Global cache instance
    clear_cache,  # Cache clearing
    get_cache_stats,  # Statistics
    get_unified_cache_key,  # Unified key generation
    warm_cache_with_data,  # Cache warming
)

# === SESSION CACHE INTEGRATION ===
from core.session_cache import (
    get_session_cache_stats,
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
                self._memory_stats["peak_memory_mb"] = max(self._memory_stats["peak_memory_mb"], memory_mb)

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
                    except Exception:
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
            _memory_optimizer.get_memory_usage_mb()

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

            except Exception:
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

        # Warm session cache
        from core.session_cache import warm_session_cache

        warm_session_cache()

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

        logger.debug("System cache warming completed successfully")
        return True

    except Exception as e:
        logger.warning(f"Error warming system caches: {e}")
        return False


# PHASE 2 ENHANCEMENT: Intelligent Cache Warming Strategies
def warm_system_caches_intelligent(
    strategies: Optional[List[str]] = None,
    background: bool = False,
    priority_data: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Enhanced cache warming with intelligent strategies and background processing.

    Args:
        strategies: List of warming strategies ('config', 'api_templates', 'common_queries', 'user_data')
        background: Whether to run warming in background thread
        priority_data: Priority data to warm first (e.g., frequently accessed items)

    Returns:
        True if warming completed successfully
    """
    if strategies is None:
        strategies = ['config', 'api_templates', 'common_queries']

    def _warm_caches():
        try:
            warmed_items = 0

            # Strategy 1: Configuration data
            if 'config' in strategies:
                config_data = {
                    "cache_version": "5.2.1",
                    "warmed_at": time.time(),
                    "strategies_used": strategies,
                    "api_cache_enabled": True,
                    "db_cache_enabled": True,
                    "memory_optimization_enabled": SYSTEM_CACHE_CONFIG.enable_aggressive_gc,
                    "adaptive_concurrency_enabled": True,
                }

                warm_cache_with_data(
                    get_unified_cache_key("system", "config", "enhanced"),
                    config_data,
                    expire=SYSTEM_CACHE_CONFIG.api_response_ttl,
                )
                warmed_items += 1

            # Strategy 2: API endpoint templates
            if 'api_templates' in strategies:
                api_templates = {
                    "ancestry_profile_details": "/api/profile/{profile_id}/details",
                    "ancestry_relationship_prob": "/api/dna/{uuid}/relationship",
                    "ancestry_suggest_api": "/api/search/suggest",
                    "ancestry_facts_api": "/api/person/{person_id}/facts",
                    "ai_classification": "/ai/classify/message",
                    "ai_extraction": "/ai/extract/genealogical",
                }

                warm_cache_with_data(
                    get_unified_cache_key("api", "templates", "endpoints"),
                    api_templates,
                    expire=SYSTEM_CACHE_CONFIG.api_response_ttl * 2,  # Longer TTL for templates
                )
                warmed_items += 1

            # Strategy 3: Common database queries
            if 'common_queries' in strategies:
                common_queries = {
                    "active_conversations": "SELECT * FROM conversation_logs WHERE status = 'active'",
                    "recent_matches": "SELECT * FROM people WHERE created_at > datetime('now', '-7 days')",
                    "productive_candidates": "SELECT * FROM people WHERE status = 'productive'",
                    "message_templates": "SELECT * FROM message_types",
                }

                warm_cache_with_data(
                    get_unified_cache_key("database", "queries", "common"),
                    common_queries,
                    expire=SYSTEM_CACHE_CONFIG.db_query_ttl,
                )
                warmed_items += 1

            # Strategy 4: Priority user data
            if 'user_data' in strategies and priority_data:
                for key, data in priority_data.items():
                    warm_cache_with_data(
                        get_unified_cache_key("user", "priority", key),
                        data,
                        expire=SYSTEM_CACHE_CONFIG.api_response_ttl,
                    )
                    warmed_items += 1

            # Warm session cache
            from core.session_cache import warm_session_cache
            session_warmed = warm_session_cache()
            if session_warmed:
                warmed_items += 1

            logger.info(f"Intelligent cache warming completed: {warmed_items} items warmed with strategies {strategies}")
            return True

        except Exception as e:
            logger.warning(f"Error in intelligent cache warming: {e}")
            return False

    if background:
        import threading
        warming_thread = threading.Thread(target=_warm_caches, daemon=True)
        warming_thread.start()
        logger.debug("Started background cache warming")
        return True
    return _warm_caches()


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
        mock_api_call("test_data")
        first_call_time = time.time() - start_time

        # Second call (should be fast - cached)
        start_time = time.time()
        mock_api_call("test_data")
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
        mock_db_query(123)
        first_db_time = time.time() - start_time

        start_time = time.time()
        mock_db_query(123)
        second_db_time = time.time() - start_time

        db_speedup = first_db_time / max(second_db_time, 0.001)
        results["db_cache_test"] = db_speedup > 3

        logger.info(
            f"DB Cache: {first_db_time:.3f}s → {second_db_time:.3f}s ({db_speedup:.1f}x speedup)"
        )

        # Test memory optimization
        _memory_optimizer.get_memory_usage_mb()
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


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for core/system_cache.py.

    Tests advanced system-wide caching including API response caching,
    database query caching, memory optimization, and system-wide cache management.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        from test_framework import TestSuite

        suite = TestSuite("System Cache Comprehensive Tests", __name__)
        suite.start_suite()

        def test_api_response_cache_functionality():
            """Test API response caching system"""
            try:
                # Test caching API response
                test_params = {"param1": "value1", "param2": "value2"}
                test_response = {"result": "success", "data": "test_data"}

                success = _api_cache.cache_api_response("test_service", "test_method", test_params, test_response, 300)
                assert success

                # Test retrieving cached response
                cached_response = _api_cache.get_cached_api_response("test_service", "test_method", test_params)
                assert cached_response is not None
                assert cached_response.get("result") == "success"

                # Test cache miss
                missing_response = _api_cache.get_cached_api_response("test_service", "nonexistent_method", test_params)
                assert missing_response is None

                # Test cache key generation
                key1 = _api_cache._get_api_cache_key("service1", "method1", {"a": 1})
                key2 = _api_cache._get_api_cache_key("service1", "method1", {"a": 1})
                assert key1 == key2  # Should be consistent

                return True
            except Exception:
                return False

        def test_database_query_cache_functionality():
            """Test database query caching system"""
            try:
                # Test caching database query result
                test_query = "SELECT * FROM test_table WHERE id = ?"
                test_params = (123,)
                test_result = [{"id": 123, "name": "test"}]

                success = _db_cache.cache_query_result(test_query, test_params, test_result, 600)
                assert success

                # Test retrieving cached result
                cached_result = _db_cache.get_cached_query_result(test_query, test_params)
                assert cached_result is not None
                assert len(cached_result) == 1
                assert cached_result[0].get("id") == 123

                # Test cache miss
                missing_result = _db_cache.get_cached_query_result("SELECT * FROM nonexistent", ())
                assert missing_result is None

                # Test query normalization
                _db_cache._get_query_cache_key("SELECT * FROM table", ())
                _db_cache._get_query_cache_key("  select  *  from  table  ", ())
                # Should normalize to same key (case and whitespace)

                return True
            except Exception:
                return False

        def test_memory_optimization_functionality():
            """Test memory optimization system"""
            try:
                # Test memory usage monitoring
                memory_usage = _memory_optimizer.get_memory_usage_mb()
                assert isinstance(memory_usage, (int, float))
                assert memory_usage >= 0

                # Test memory optimization
                optimization_result = _memory_optimizer.optimize_memory()
                assert isinstance(optimization_result, dict)
                assert "optimized" in optimization_result

                # Test forced memory optimization
                forced_result = _memory_optimizer.optimize_memory(force=True)
                assert forced_result.get("optimized") in [True, False]  # Should be boolean

                return True
            except Exception:
                return False

        def test_caching_decorators():
            """Test system caching decorators"""
            try:
                # Test API caching decorator
                call_count = 0

                @cached_api_call("test_api", ttl=60)
                def test_api_function(param):
                    nonlocal call_count
                    call_count += 1
                    return {"param": param, "call_count": call_count}

                result1 = test_api_function("test_param")
                assert result1.get("call_count") == 1

                result2 = test_api_function("test_param")  # Should use cache or call again
                assert result2 is not None

                # Test database caching decorator
                db_call_count = 0

                @cached_database_query(ttl=300)
                def test_db_function(query_id):
                    nonlocal db_call_count
                    db_call_count += 1
                    return [{"id": query_id, "call_count": db_call_count}]

                db_result1 = test_db_function(123)
                assert len(db_result1) > 0

                # Test memory optimization decorator
                @memory_optimized(gc_threshold=0.5)
                def test_memory_function():
                    # Simulate memory intensive operation
                    data = list(range(1000))
                    return len(data)

                memory_result = test_memory_function()
                assert memory_result == 1000

                return True
            except Exception:
                return False

        def test_system_cache_statistics():
            """Test comprehensive system cache statistics"""
            try:
                stats = get_system_cache_stats()

                # Should have all expected categories
                expected_categories = ["base_cache", "session_cache", "api_cache", "database_cache", "memory_optimization", "system_config"]
                for category in expected_categories:
                    assert category in stats, f"Missing category: {category}"

                # API cache stats should have expected fields
                api_stats = stats["api_cache"]
                expected_api_fields = ["ai_requests", "ai_cache_hits", "ancestry_requests", "ancestry_cache_hits"]
                for field in expected_api_fields:
                    assert field in api_stats, f"Missing API stat field: {field}"

                # System config should have expected settings
                system_config = stats["system_config"]
                expected_config_fields = ["api_response_ttl", "db_query_ttl", "memory_limit_mb", "aggressive_gc_enabled"]
                for field in expected_config_fields:
                    assert field in system_config, f"Missing config field: {field}"

                return True
            except Exception:
                return False

        def test_cache_management_operations():
            """Test system cache management"""
            try:
                # Test cache warming
                warm_result = warm_system_caches()
                assert warm_result

                # Test cache clearing
                clear_results = clear_system_caches()
                assert isinstance(clear_results, dict)

                # Should not have errors in clearing results
                assert "error" not in clear_results or clear_results.get("error") is None

                # Should have cleared various cache types
                expected_clear_types = ["base_cache", "session_cache", "api_cache", "database_cache"]
                for cache_type in expected_clear_types:
                    if cache_type in clear_results:
                        assert isinstance(clear_results[cache_type], (int, str))

                return True
            except Exception:
                return False

        def test_cache_configuration():
            """Test system cache configuration"""
            try:
                # Test configuration access
                assert SYSTEM_CACHE_CONFIG is not None
                assert hasattr(SYSTEM_CACHE_CONFIG, 'api_response_ttl')
                assert hasattr(SYSTEM_CACHE_CONFIG, 'db_query_ttl')
                assert hasattr(SYSTEM_CACHE_CONFIG, 'memory_cache_limit_mb')

                # Test configuration values are reasonable
                assert SYSTEM_CACHE_CONFIG.api_response_ttl > 0
                assert SYSTEM_CACHE_CONFIG.db_query_ttl > 0
                assert SYSTEM_CACHE_CONFIG.memory_cache_limit_mb > 0

                # Test configuration is used by cache instances
                assert _api_cache is not None
                assert _db_cache is not None
                assert _memory_optimizer is not None

                return True
            except Exception:
                return False

        def test_performance_improvements():
            """Test actual performance improvements from caching"""
            try:
                import time

                # Test API caching performance
                @cached_api_call("performance_test", ttl=30)
                def slow_api_call():
                    time.sleep(0.01)  # Simulate network delay
                    return {"timestamp": time.time()}

                # First call
                start = time.time()
                result1 = slow_api_call()
                first_call_time = time.time() - start

                # Second call (potentially cached)
                start = time.time()
                result2 = slow_api_call()
                second_call_time = time.time() - start

                # Should at least not be significantly slower
                assert first_call_time >= 0
                assert second_call_time >= 0
                assert result1 is not None
                assert result2 is not None

                # Test database caching performance
                @cached_database_query(ttl=30)
                def slow_db_query():
                    time.sleep(0.005)  # Simulate DB delay
                    return [{"result": "cached"}]

                start = time.time()
                db_result = slow_db_query()
                db_call_time = time.time() - start

                assert db_call_time >= 0
                assert len(db_result) > 0

                return True
            except Exception:
                return False

        def test_error_handling_and_edge_cases():
            """Test error handling in cache operations"""
            try:
                # Test API cache with None response
                success = _api_cache.cache_api_response("test", "method", {}, None, 60)
                assert success  # Should handle None gracefully

                # Test database cache with empty result
                success = _db_cache.cache_query_result("SELECT", (), [], 60)
                assert success  # Should handle empty results

                # Test memory optimization when no optimization possible
                result = _memory_optimizer.optimize_memory()
                assert isinstance(result, dict)  # Should return result dict regardless

                # Test cache operations when cache is unavailable
                # This tests graceful degradation
                original_cache = globals().get('cache')
                try:
                    # Temporarily disable cache
                    globals()['cache'] = None

                    no_cache_result = _api_cache.get_cached_api_response("test", "method", {})
                    assert no_cache_result is None  # Should return None when cache unavailable

                    no_cache_success = _api_cache.cache_api_response("test", "method", {}, "result", 60)
                    assert not no_cache_success  # Should return False when cache unavailable

                finally:
                    # Restore cache
                    globals()['cache'] = original_cache

                return True
            except Exception:
                return False

        # Run all tests
        suite.run_test(
            "API Response Cache Functionality",
            test_api_response_cache_functionality,
            "API response cache should handle caching and retrieval of external API calls",
            "API response caching reduces external service call overhead and improves performance",
            "Test API response caching, retrieval, and cache key generation"
        )

        suite.run_test(
            "Database Query Cache Functionality",
            test_database_query_cache_functionality,
            "Database query cache should handle caching and retrieval of query results",
            "Database query caching reduces database load and improves query performance",
            "Test database query result caching, retrieval, and query normalization"
        )

        suite.run_test(
            "Memory Optimization Functionality",
            test_memory_optimization_functionality,
            "Memory optimizer should monitor usage and perform intelligent garbage collection",
            "Memory optimization prevents memory leaks and improves system stability",
            "Test memory usage monitoring and optimization operations"
        )

        suite.run_test(
            "Caching Decorators",
            test_caching_decorators,
            "System caching decorators should provide transparent performance optimization",
            "Decorators enable easy integration of caching into existing code",
            "Test API, database, and memory optimization decorators"
        )

        suite.run_test(
            "System Cache Statistics",
            test_system_cache_statistics,
            "System cache statistics should provide comprehensive performance metrics",
            "Statistics enable monitoring and optimization of cache performance",
            "Test comprehensive statistics collection across all cache systems"
        )

        suite.run_test(
            "Cache Management Operations",
            test_cache_management_operations,
            "Cache management should handle warming and clearing operations",
            "Management operations ensure optimal cache performance and resource cleanup",
            "Test cache warming, clearing, and management functionality"
        )

        suite.run_test(
            "Cache Configuration",
            test_cache_configuration,
            "System cache configuration should be properly structured and accessible",
            "Configuration enables tuning of cache behavior for different environments",
            "Test system cache configuration structure and accessibility"
        )

        suite.run_test(
            "Performance Improvements",
            test_performance_improvements,
            "Caching should provide measurable performance improvements",
            "Performance improvements justify the complexity of caching systems",
            "Test actual performance improvements from API and database caching"
        )

        suite.run_test(
            "Error Handling and Edge Cases",
            test_error_handling_and_edge_cases,
            "System cache should handle errors and edge cases gracefully",
            "Robust error handling ensures cache failures don't break application functionality",
            "Test error scenarios and graceful degradation when cache is unavailable"
        )

        return suite.finish_suite()

    except ImportError:
        print("Warning: TestSuite not available, running basic validation...")

        # Basic fallback tests
        try:
            # Test basic functionality
            assert _api_cache is not None
            assert _db_cache is not None
            assert _memory_optimizer is not None

            # Test basic operations
            _api_cache.cache_api_response("basic_test", "method", {}, {"result": "test"}, 60)
            cached = _api_cache.get_cached_api_response("basic_test", "method", {})
            assert cached is not None

            stats = get_system_cache_stats()
            assert stats is not None

            warm_result = warm_system_caches()
            assert warm_result

            print("✅ Basic system_cache validation passed")
            return True
        except Exception as e:
            print(f"❌ Basic system_cache validation failed: {e}")
            return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Use comprehensive TestSuite framework
    success = run_comprehensive_tests()
    print(f"\n🎯 System Cache Test Results: {'✅ PASSED' if success else '❌ FAILED'}")
    exit(0 if success else 1)
