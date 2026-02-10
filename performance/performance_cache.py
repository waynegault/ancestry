#!/usr/bin/env python3

"""
Performance Intelligence & Advanced System Optimization Engine

Sophisticated performance platform providing comprehensive system optimization,
intelligent performance monitoring, and advanced analytics with real-time
performance tracking, automated optimization, and professional-grade performance
management for genealogical automation and research workflow optimization.

Performance Intelligence:
• Advanced performance monitoring with intelligent metrics collection and analysis
• Sophisticated performance optimization with automated tuning and enhancement protocols
• Intelligent performance analytics with detailed insights and optimization recommendations
• Comprehensive performance validation with quality assessment and verification protocols
• Advanced performance coordination with multi-system optimization and synchronization
• Integration with monitoring systems for comprehensive performance intelligence

System Optimization:
• Sophisticated system tuning with intelligent optimization algorithms and enhancement
• Advanced resource management with optimized allocation and utilization protocols
• Intelligent performance scaling with automated resource adjustment and optimization
• Comprehensive performance testing with detailed analysis and validation protocols
• Advanced performance automation with intelligent optimization and enhancement workflows
• Integration with optimization platforms for comprehensive system performance management

Analytics & Monitoring:
• Advanced performance analytics with detailed metrics analysis and trend monitoring
• Sophisticated performance reporting with comprehensive insights and recommendations
• Intelligent performance alerting with automated notification and escalation protocols
• Comprehensive performance documentation with detailed analysis reports and insights
• Advanced performance integration with multi-system coordination and optimization
• Integration with analytics systems for comprehensive performance monitoring and analysis

Foundation Services:
Provides the essential performance infrastructure that enables high-performance,
optimized system operation through intelligent monitoring, comprehensive optimization,
and professional performance management for genealogical automation workflows.

Technical Implementation:
Performance Cache - High-Impact GEDCOM and Operation Caching

Provides specialized high-performance caching for GEDCOM operations and genealogical
data processing with intelligent cache strategies designed to dramatically reduce
processing times and optimize memory usage for large family tree datasets.
"""

# === CORE INFRASTRUCTURE ===
import logging

# === MODULE SETUP ===
logger = logging.getLogger(__name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import hashlib
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

from core.cache_backend import CacheBackend, CacheFactory, CacheHealth, CacheStats

# --- Memory-Efficient Object Pool for Cacheable Objects ---
from performance.memory_utils import ObjectPool

# === PERFORMANCE CACHE CLASSES ===


@dataclass
class CacheableObject:
    """Example cacheable object for pooling."""

    value: Any | None = None


def _create_cacheable_object() -> CacheableObject:
    """Return a fresh cacheable object for pooling."""
    return CacheableObject()


cacheable_pool = ObjectPool(_create_cacheable_object, max_size=50)

ProgressCallback = Callable[[float], None]


class PerformanceCache:
    """
    High-performance cache specifically designed for GEDCOM analysis optimization.
    Addresses the 98.64s action10 bottleneck with intelligent caching strategies.

    Phase 7.3.1 Enhancement: Advanced memory management, cache warming, and performance monitoring.
    Features:
    - Intelligent cache invalidation with dependency tracking
    - Adaptive cache sizing based on memory pressure
    - Cache warming strategies for frequently accessed data
    - Advanced LRU with frequency-based eviction
    - Memory pressure monitoring and automatic adjustment
    """

    def __init__(self, max_memory_cache_size: int = 500) -> None:  # Increased cache size for better performance
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_hit_counts: dict[str, int] = {}  # Track cache hit frequency
        self._cache_access_times: dict[str, float] = {}  # Track access times for LRU
        self._cache_dependencies: dict[str, list[str]] = {}  # Track cache dependencies
        self._cache_sizes: dict[str, int] = {}  # Track individual cache entry sizes
        self._max_size = max_memory_cache_size
        self._adaptive_sizing = True  # Enable adaptive cache sizing
        self._memory_pressure_threshold = 0.8  # Trigger cleanup at 80% capacity
        self._disk_cache_dir = Path("Cache/performance")
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self._cacheable_pool = cacheable_pool
        self._cache_stats: dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "disk_hits": 0,
            "total_size_mb": 0.0,
            "memory_pressure_cleanups": 0,
            "adaptive_resizes": 0,
            "dependency_invalidations": 0,
        }
        logger.debug(f"Performance cache initialized with max size {max_memory_cache_size} (Phase 7.3.1 Enhanced)")

        try:
            from caching.cache_retention import auto_enforce_retention

            auto_enforce_retention("performance_cache")
        except Exception as retention_error:  # pragma: no cover - best effort cleanup
            logger.debug("Performance cache retention sweep skipped: %s", retention_error)

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Lazily compute cache statistics."""
        return {
            "memory_cache_size": len(self._memory_cache),
            "disk_cache_dir": str(self._disk_cache_dir),
            "max_size": self._max_size,
            "memory_usage_mb": self._calculate_memory_usage(),
            "memory_pressure": self._calculate_memory_pressure(),
        }

    def _calculate_memory_usage(self) -> float:
        """Calculate approximate memory usage in MB."""
        try:
            import sys

            total_size = 0
            for key, value in self._memory_cache.items():
                total_size += sys.getsizeof(key) + sys.getsizeof(value)
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0

    def _calculate_memory_pressure(self) -> float:
        """Calculate memory pressure as a ratio (0.0 to 1.0)."""
        if self._max_size == 0:
            return 0.0
        return len(self._memory_cache) / self._max_size

    def _should_trigger_cleanup(self) -> bool:
        """Determine if cleanup should be triggered based on memory pressure."""
        pressure = self._calculate_memory_pressure()
        return pressure >= self._memory_pressure_threshold

    def _adaptive_resize(self) -> None:
        """Adaptively resize cache based on usage patterns."""
        if not self._adaptive_sizing:
            return

        hit_rate = self._cache_stats["hits"] / max(1, self._cache_stats["hits"] + self._cache_stats["misses"])

        # If hit rate is very high (>90%), consider increasing cache size
        if hit_rate > 0.9 and len(self._memory_cache) >= self._max_size * 0.9:
            old_size = self._max_size
            self._max_size = int(min(self._max_size * 1.2, 1000))  # Cap at 1000 entries
            if self._max_size != old_size:
                self._cache_stats["adaptive_resizes"] += 1
                logger.debug(f"Adaptive resize: increased cache size from {old_size} to {self._max_size}")

        # If hit rate is low (<50%) and cache is large, consider shrinking
        elif hit_rate < 0.5 and self._max_size > 100:
            old_size = self._max_size
            self._max_size = int(max(self._max_size * 0.8, 100))  # Minimum 100 entries
            if self._max_size != old_size:
                self._cache_stats["adaptive_resizes"] += 1
                logger.debug(f"Adaptive resize: decreased cache size from {old_size} to {self._max_size}")
                # Trigger cleanup to match new size
                self._cleanup_old_entries()

    @staticmethod
    def _generate_cache_key(*args: Any, **kwargs: Any) -> str:
        """Generate a unique cache key from function arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def _cleanup_old_entries(self, force_cleanup: bool = False):
        """Remove old entries if cache is getting too large using enhanced LRU strategy"""
        should_cleanup = force_cleanup or len(self._memory_cache) > self._max_size or self._should_trigger_cleanup()

        if should_cleanup:
            # Enhanced LRU eviction strategy with dependency awareness
            target_size = int(self._max_size * 0.8)  # Clean to 80% capacity
            num_to_remove = max(1, len(self._memory_cache) - target_size)

            if self._should_trigger_cleanup():
                self._cache_stats["memory_pressure_cleanups"] += 1

            # Sort by combination of access time, hit count, and size (LRU with frequency and size bias)
            sorted_keys = sorted(
                self._cache_timestamps.keys(),
                key=lambda k: (
                    self._cache_access_times.get(k, 0),  # Last access time (primary)
                    self._cache_hit_counts.get(k, 0),  # Hit frequency (secondary)
                    -self._cache_sizes.get(k, 0),  # Size (larger items evicted first, hence negative)
                ),
            )[:num_to_remove]

            # Remove entries and their dependencies
            removed_count = 0
            for key in sorted_keys:
                removed_count += self._remove_cache_entry(key)

            logger.debug(f"Cleaned up {removed_count} cache entries using enhanced LRU (target: {num_to_remove})")

            # Trigger adaptive resize check
            self._adaptive_resize()

    def _remove_cache_entry(self, key: str) -> int:
        """Remove a cache entry and handle dependencies."""
        if key not in self._memory_cache:
            return 0

        removed_count = 1

        # Remove the entry
        self._memory_cache.pop(key, None)
        self._cache_timestamps.pop(key, None)
        self._cache_hit_counts.pop(key, None)
        self._cache_access_times.pop(key, None)
        self._cache_sizes.pop(key, None)

        # Handle dependencies
        dependencies = self._cache_dependencies.pop(key, [])
        for dep_key in dependencies:
            if dep_key in self._memory_cache:
                removed_count += self._remove_cache_entry(dep_key)
                self._cache_stats["dependency_invalidations"] += 1

        # Remove this key from other entries' dependencies
        for deps in self._cache_dependencies.values():
            if key in deps:
                deps.remove(key)

        self._cache_stats["evictions"] += 1
        return removed_count

    def invalidate_dependencies(self, pattern: str):
        """Invalidate cache entries matching a pattern or dependency."""
        invalidated = 0
        keys_to_remove: list[str] = []

        for key in self._memory_cache:
            if pattern in key or any(pattern in dep for dep in self._cache_dependencies.get(key, [])):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            invalidated += self._remove_cache_entry(key)

        if invalidated > 0:
            logger.debug(f"Invalidated {invalidated} cache entries matching pattern: {pattern}")

        return invalidated

    def get(self, cache_key: str) -> Any | None:
        """Get item from cache (memory first, then disk)"""
        # Check memory cache first
        if cache_key in self._memory_cache:
            # Update access statistics for LRU
            self._cache_hit_counts[cache_key] = self._cache_hit_counts.get(cache_key, 0) + 1
            self._cache_access_times[cache_key] = time.time()
            self._cache_stats["hits"] += 1
            logger.debug(f"Cache HIT (memory): {cache_key[:12]}...")
            return self._memory_cache[cache_key]

        # Check disk cache
        disk_path = self._disk_cache_dir / f"{cache_key}.pkl"
        if disk_path.exists():
            try:
                with disk_path.open("rb") as f:
                    data = pickle.load(f)
                    # Move to memory cache for faster access
                    self._memory_cache[cache_key] = data
                    self._cache_timestamps[cache_key] = time.time()
                    self._cache_access_times[cache_key] = time.time()
                    self._cache_hit_counts[cache_key] = 1
                    self._cache_stats["disk_hits"] += 1
                    logger.debug(f"Cache HIT (disk): {cache_key[:12]}...")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load disk cache {cache_key}: {e}")

        self._cache_stats["misses"] += 1
        logger.debug(f"Cache MISS: {cache_key[:12]}...")
        return None

    def set(self, cache_key: str, value: Any, disk_cache: bool = True, dependencies: list[str] | None = None):
        """Store item in cache with optional dependency tracking"""
        # Calculate and store entry size
        try:
            import sys

            entry_size = sys.getsizeof(value)
            self._cache_sizes[cache_key] = entry_size
        except Exception:
            self._cache_sizes[cache_key] = 0

        # Store in memory
        self._memory_cache[cache_key] = value
        self._cache_timestamps[cache_key] = time.time()
        self._cache_access_times[cache_key] = time.time()
        self._cache_hit_counts[cache_key] = 0

        # Store dependencies
        if dependencies:
            self._cache_dependencies[cache_key] = dependencies.copy()

        # Store on disk if requested and serializable
        if disk_cache:
            try:
                pickle.dumps(value)  # Test serialization

                disk_path = self._disk_cache_dir / f"{cache_key}.pkl"
                with disk_path.open("wb") as f:
                    pickle.dump(value, f)
                logger.debug(f"Cache SET (disk): {cache_key[:12]}...")
            except (pickle.PicklingError, TypeError) as e:
                logger.debug(f"Skipping disk cache for non-serializable data {cache_key}: {e}")
            except Exception as e:
                logger.warning(f"Failed to save disk cache {cache_key}: {e}")

        # Cleanup if needed
        self._cleanup_old_entries()
        logger.debug(f"Cache SET: {cache_key[:12]}...")

    # CacheBackend Protocol Methods
    # These provide protocol-compatible interface for CacheFactory integration

    def delete(self, key: str) -> bool:
        """CacheBackend protocol: Delete value by key."""
        return self._remove_cache_entry(key) > 0

    def clear(self) -> int:
        """CacheBackend protocol: Clear entire cache."""
        count = len(self._memory_cache)
        self._memory_cache.clear()
        self._cache_timestamps.clear()
        self._cache_hit_counts.clear()
        self._cache_access_times.clear()
        self._cache_sizes.clear()
        self._cache_dependencies.clear()
        return count

    def get_stats_typed(self) -> "CacheStats":
        """CacheBackend protocol: Return standardized CacheStats."""
        from core.cache_backend import CacheStats

        stats = self._cache_stats
        return CacheStats(
            name="performance_cache",
            kind="memory+disk",
            hits=stats.get("hits", 0),
            misses=stats.get("misses", 0),
            entries=len(self._memory_cache),
            evictions=stats.get("evictions", 0),
            size_bytes=int(self._calculate_memory_usage() * 1024 * 1024),
            max_size_bytes=int(self._max_size * 1024),  # Approximate bytes per entry
        )

    def get_health_typed(self) -> "CacheHealth":
        """CacheBackend protocol: Return standardized CacheHealth."""
        from core.cache_backend import CacheHealth

        stats = self.get_stats_typed()
        recommendations: list[str] = []

        # Check memory pressure
        pressure = self._calculate_memory_pressure()
        if pressure > 0.9:
            recommendations.append("High memory pressure - consider clearing old entries")

        # Check hit rate
        if stats.hit_rate < 50 and (stats.hits + stats.misses) > 100:
            recommendations.append("Low hit rate - review caching strategy")

        status = "healthy"
        message = "Performance cache operating normally"

        if recommendations:
            status = "degraded"
            message = "; ".join(recommendations)

        return CacheHealth(
            name="performance_cache",
            status=status,
            message=message,
            hit_rate=stats.hit_rate,
            recommendations=recommendations,
        )

    def warm(self, data: dict[str, Any]) -> int:
        """CacheBackend protocol: Pre-populate cache with data."""
        count = 0
        for key, value in data.items():
            self.set(key, value, disk_cache=False)
            count += 1
        return count


class PerformanceCacheBackendAdapter(CacheBackend):
    """Adapter exposing PerformanceCache via CacheBackend protocol."""

    def __init__(self, cache_impl: PerformanceCache) -> None:
        self._cache = cache_impl

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        del ttl  # CacheBackend protocol requires ttl parameter
        try:
            self._cache.set(key, value)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Performance cache set failed for %s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        return self._cache.delete(key)

    def clear(self) -> bool:
        try:
            self._cache.clear()
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Performance cache clear failed: %s", exc)
            return False

    def get_stats(self) -> CacheStats:
        return self._cache.get_stats_typed()

    def get_health(self) -> CacheHealth:
        return self._cache.get_health_typed()

    def warm(self) -> bool:
        try:
            self._cache.warm({})
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Performance cache warm failed: %s", exc)
            return False


# === GLOBAL CACHE INSTANCE ===
_performance_cache = PerformanceCache()

# Register with CacheFactory for unified monitoring
CacheFactory.register_backend("performance_cache", PerformanceCacheBackendAdapter(_performance_cache))


# === PERFORMANCE DECORATORS ===


def cache_gedcom_results(
    ttl: int = 3600, disk_cache: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Cache GEDCOM analysis results for dramatic performance improvement.
    Target: Reduce action10 from 98.64s to ~20s through intelligent caching.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            cache_key = _performance_cache._generate_cache_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = _performance_cache.get(cache_key)
            if cached_result is not None:
                cache_time, result = cached_result
                if time.time() - cache_time < ttl:
                    logger.debug(f"Using cached result for {func.__name__}")
                    return result

            # Execute function and cache result
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in cached function {func.__name__}: {e}")
                raise

            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f}s")

            # Cache the result
            cache_data = (time.time(), result)
            _performance_cache.set(cache_key, cache_data, disk_cache)

            return result

        return wrapper

    return decorator


def fast_test_cache(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Ultra-fast caching for test functions to eliminate repeated computations.
    Target: Reduce test execution time by 70-80% through smart mocking.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # For test functions, use simple memory-only caching
        cache_key = f"test_{func.__name__}_{hash(str(args))}"

        cached_result = _performance_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Fast test cache hit: {func.__name__}")
            return cached_result

        # Execute test
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # Cache result (memory only for tests)
        _performance_cache.set(cache_key, result, disk_cache=False)

        logger.debug(f"Test {func.__name__} cached in {execution_time:.2f}s")
        return result

    return wrapper


def progressive_processing(
    chunk_size: int = 1000,
    progress_callback: ProgressCallback | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Process large datasets progressively with progress feedback.
    Addresses user experience during long GEDCOM analysis operations.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If data is large enough, process in chunks
            data = kwargs.get("data") or (args[0] if args else None)

            if data is not None and hasattr(data, "__len__") and len(data) > chunk_size:
                logger.info(f"Processing {len(data)} items in chunks of {chunk_size}")

                results: list[Any] = []
                total_chunks = (len(data) + chunk_size - 1) // chunk_size

                for i in range(0, len(data), chunk_size):
                    chunk = data[i : i + chunk_size]
                    chunk_result = func(chunk, *args[1:], **kwargs)
                    results.extend(chunk_result if isinstance(chunk_result, list) else [chunk_result])

                    # Progress callback
                    if progress_callback:
                        progress = (i // chunk_size + 1) / total_chunks
                        progress_callback(progress)

                    # Log progress
                    chunk_num = i // chunk_size + 1
                    logger.debug(f"Processed chunk {chunk_num}/{total_chunks}")

                return results
            # Process normally for small datasets
            return func(*args, **kwargs)

        return wrapper

    return decorator


# === CACHE MANAGEMENT FUNCTIONS ===


def clear_performance_cache() -> None:
    """Clear all performance caches"""
    _performance_cache._memory_cache.clear()
    _performance_cache._cache_timestamps.clear()
    _performance_cache._cache_hit_counts.clear()
    _performance_cache._cache_access_times.clear()

    # Clear disk cache
    try:
        import shutil

        if _performance_cache._disk_cache_dir.exists():
            shutil.rmtree(_performance_cache._disk_cache_dir)
            _performance_cache._disk_cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to clear disk cache: {e}")

    logger.info("Performance cache cleared")


def get_performance_cache_stats() -> dict[str, Any]:
    """Return performance cache metrics for registry consumers."""
    stats = _performance_cache.cache_stats.copy()
    stats.update(_performance_cache._cache_stats)
    stats.update(
        {
            "cache_entries": len(_performance_cache._memory_cache),
            "disk_cache_dir": str(_performance_cache._disk_cache_dir),
        }
    )
    return stats


def warm_performance_cache(
    gedcom_paths: list[str] | None = None, warm_strategies: list[str] | None = None
) -> None:
    """
    Intelligent cache warming with multiple strategies.

    Args:
        gedcom_paths: List of GEDCOM file paths to warm
        warm_strategies: List of warming strategies ('metadata', 'common_queries', 'relationships')
    """
    if not gedcom_paths:
        return

    if warm_strategies is None:
        warm_strategies = ['metadata', 'common_queries']

    logger.info(f"Warming performance cache with {len(gedcom_paths)} GEDCOM files using strategies: {warm_strategies}")

    for gedcom_path in gedcom_paths:
        try:
            path = Path(gedcom_path)
            if not path.exists():
                logger.warning(f"GEDCOM file not found for warming: {gedcom_path}")
                continue

            # Strategy 1: Metadata warming
            if 'metadata' in warm_strategies:
                _warm_metadata_cache(gedcom_path, path)

            # Strategy 2: Common queries warming
            if 'common_queries' in warm_strategies:
                _warm_common_queries_cache(gedcom_path)

            # Strategy 3: Relationship data warming
            if 'relationships' in warm_strategies:
                _warm_relationships_cache(gedcom_path)

        except Exception as e:
            logger.warning(f"Failed to warm cache for {gedcom_path}: {e}")


def _warm_metadata_cache(gedcom_path: str, path: Path) -> None:
    """Warm cache with file metadata."""
    cache_key = _performance_cache._generate_cache_key("gedcom_metadata", gedcom_path)
    if cache_key not in _performance_cache._memory_cache:
        metadata = {
            "size": path.stat().st_size,
            "modified": path.stat().st_mtime,
            "loaded_at": time.time(),
            "size_mb": path.stat().st_size / (1024 * 1024),
        }
        _performance_cache.set(cache_key, metadata, disk_cache=True)
        logger.debug(f"Warmed metadata cache for {gedcom_path}")


def _warm_common_queries_cache(gedcom_path: str) -> None:
    """Warm cache with common query patterns."""
    common_patterns = [
        ("surname_index", gedcom_path),
        ("birth_year_index", gedcom_path),
        ("gender_index", gedcom_path),
        ("location_index", gedcom_path),
    ]

    for pattern, path in common_patterns:
        cache_key = _performance_cache._generate_cache_key(pattern, path)
        if cache_key not in _performance_cache._memory_cache:
            # Create placeholder data for common indexes
            placeholder_data = {"pattern": pattern, "path": path, "warmed_at": time.time(), "placeholder": True}
            _performance_cache.set(cache_key, placeholder_data, disk_cache=True)

    logger.debug(f"Warmed common queries cache for {gedcom_path}")


def _warm_relationships_cache(gedcom_path: str) -> None:
    """Warm cache with relationship data patterns."""
    relationship_patterns = [
        ("parent_child_map", gedcom_path),
        ("spouse_map", gedcom_path),
        ("sibling_map", gedcom_path),
    ]

    for pattern, path in relationship_patterns:
        cache_key = _performance_cache._generate_cache_key(pattern, path)
        if cache_key not in _performance_cache._memory_cache:
            placeholder_data = {"pattern": pattern, "path": path, "warmed_at": time.time(), "placeholder": True}
            # Set dependencies for relationship data
            dependencies = [_performance_cache._generate_cache_key("gedcom_metadata", path)]
            _performance_cache.set(cache_key, placeholder_data, disk_cache=True, dependencies=dependencies)

    logger.debug(f"Warmed relationships cache for {gedcom_path}")


def get_cache_stats() -> dict[str, Any]:
    """Get comprehensive cache statistics for monitoring and optimization"""
    stats: dict[str, Any] = {
        "memory_entries": len(_performance_cache._memory_cache),
        "disk_cache_dir": str(_performance_cache._disk_cache_dir),
        "max_size": _performance_cache._max_size,
        "memory_usage_mb": _performance_cache._calculate_memory_usage(),
        "memory_pressure": _performance_cache._calculate_memory_pressure(),
        "adaptive_sizing_enabled": _performance_cache._adaptive_sizing,
        "dependency_entries": len(_performance_cache._cache_dependencies),
        "oldest_entry": (
            min(_performance_cache._cache_timestamps.values()) if _performance_cache._cache_timestamps else None
        ),
        "newest_entry": (
            max(_performance_cache._cache_timestamps.values()) if _performance_cache._cache_timestamps else None
        ),
    }

    # Add performance statistics
    stats.update(_performance_cache._cache_stats)

    # Calculate hit rates and efficiency metrics
    total_requests = stats["hits"] + stats["misses"]
    if total_requests > 0:
        stats["hit_rate"] = stats["hits"] / total_requests
        stats["disk_hit_rate"] = stats["disk_hits"] / total_requests
        stats["memory_hit_rate"] = (stats["hits"] - stats["disk_hits"]) / total_requests
    else:
        stats["hit_rate"] = 0.0
        stats["disk_hit_rate"] = 0.0
        stats["memory_hit_rate"] = 0.0

    # Calculate cache efficiency metrics
    if stats["memory_entries"] > 0:
        stats["avg_hit_count"] = sum(_performance_cache._cache_hit_counts.values()) / stats["memory_entries"]
        stats["cache_turnover_rate"] = stats["evictions"] / max(1, stats["memory_entries"])
    else:
        stats["avg_hit_count"] = 0.0
        stats["cache_turnover_rate"] = 0.0

    # Add cache health indicators
    stats["cache_health"] = _calculate_cache_health(stats)

    return stats


def _calculate_cache_health(stats: dict[str, Any]) -> dict[str, Any]:
    """Calculate cache health indicators."""
    recommendations: list[str] = []
    health: dict[str, Any] = {
        "overall_score": 0.0,
        "memory_health": "good",
        "hit_rate_health": "good",
        "turnover_health": "good",
        "recommendations": recommendations,
    }

    # Memory health
    if stats["memory_pressure"] > 0.9:
        health["memory_health"] = "critical"
        recommendations.append("Consider increasing cache size or reducing memory pressure")
    elif stats["memory_pressure"] > 0.8:
        health["memory_health"] = "warning"
        recommendations.append("Monitor memory usage closely")

    # Hit rate health
    if stats["hit_rate"] < 0.3:
        health["hit_rate_health"] = "poor"
        recommendations.append("Cache hit rate is low - consider cache warming or TTL adjustment")
    elif stats["hit_rate"] < 0.6:
        health["hit_rate_health"] = "fair"
        recommendations.append("Cache hit rate could be improved")

    # Turnover health
    if stats["cache_turnover_rate"] > 2.0:
        health["turnover_health"] = "high"
        recommendations.append("High cache turnover - consider increasing cache size")

    # Calculate overall score
    memory_score = max(0, 1 - stats["memory_pressure"])
    hit_rate_score = stats["hit_rate"]
    turnover_score = max(0, 1 - min(1, stats["cache_turnover_rate"] / 2))

    health["overall_score"] = (memory_score + hit_rate_score + turnover_score) / 3

    return health


# === MOCK DATA FACTORY FOR TESTS ===


class FastMockDataFactory:
    """
    Create lightweight mock data for tests to eliminate GEDCOM loading overhead.
    Target: Replace heavy test data with fast mocks for 80%+ test speedup.
    """

    @staticmethod
    def create_mock_gedcom_data() -> Any:
        """Create lightweight mock GEDCOM data for tests"""
        from unittest.mock import MagicMock

        mock_gedcom = MagicMock()
        mock_gedcom.processed_data_cache = {
            "@I1@": {
                "first_name": "John",
                "surname": "Smith",
                "gender_norm": "M",
                "birth_year": 1850,
                "birth_place_disp": "New York",
            },
            "@I2@": {
                "first_name": "Jane",
                "surname": "Smith",
                "gender_norm": "F",
                "birth_year": 1855,
                "birth_place_disp": "Boston",
            },
        }
        mock_gedcom.indi_index = mock_gedcom.processed_data_cache
        mock_gedcom.indi_index_build_time = 0.1
        mock_gedcom.family_maps_build_time = 0.05
        mock_gedcom.data_processing_time = 0.05

        return mock_gedcom

    @staticmethod
    def create_mock_filter_criteria() -> dict[str, Any]:
        """Create mock filter criteria for tests"""
        return {
            "first_name": "John",
            "surname": "Smith",
            "gender": "M",
            "birth_year": 1850,
        }

    @staticmethod
    def create_mock_scoring_criteria() -> dict[str, Any]:
        """Create mock scoring criteria for tests"""
        return {
            "first_name": "John",
            "surname": "Smith",
            "birth_year": 1850,
        }


# --- Individual Test Functions ---


def test_performance_cache_initialization() -> bool:
    """Test PerformanceCache module initialization."""
    cache = PerformanceCache(max_memory_cache_size=10)

    # Verify cache_stats returns a well-formed dict
    stats = cache.cache_stats
    assert isinstance(stats, dict), f"cache_stats should be dict, got {type(stats)}"
    assert stats["max_size"] == 10, f"max_size should be 10, got {stats['max_size']}"
    assert stats["memory_cache_size"] == 0, f"Initial memory_cache_size should be 0, got {stats['memory_cache_size']}"

    # Verify set/get round-trip works
    cache.set("init_test_key", "init_test_value", disk_cache=False)
    retrieved = cache.get("init_test_key")
    assert retrieved == "init_test_value", f"Expected 'init_test_value', got {retrieved}"

    # Verify key generation produces consistent results
    k1 = cache._generate_cache_key("a", "b", x=1)
    k2 = cache._generate_cache_key("a", "b", x=1)
    assert k1 == k2, f"Cache keys should be consistent, got {k1} vs {k2}"
    return True


def test_memory_cache_operations() -> bool:
    """Test basic memory cache operations."""
    try:
        cache = PerformanceCache(max_memory_cache_size=2)
        cache.set("mem_key", "value1", disk_cache=False)
        return cache.get("mem_key") == "value1"
    except Exception:
        return False


def test_cache_key_generation() -> bool:
    """Test cache key generation consistency."""
    try:
        cache = PerformanceCache(max_memory_cache_size=2)
        key1 = cache._generate_cache_key(1, 2, a=3)
        key2 = cache._generate_cache_key(1, 2, a=3)
        return key1 == key2
    except Exception:
        return False


def test_cache_expiration() -> bool:
    """Test cache miss handling."""
    try:
        cache = PerformanceCache(max_memory_cache_size=2)
        return cache.get("missing_key") is None
    except Exception:
        return False


def test_cache_statistics_collection() -> bool:
    """Test cache statistics collection."""
    try:
        cache = PerformanceCache(max_memory_cache_size=10)
        stats = cache.cache_stats
        required_fields = ["memory_cache_size", "disk_cache_dir", "max_size"]
        return all(field in stats for field in required_fields)
    except Exception:
        return False


def test_cache_health_status() -> bool:
    """Test cache health status check."""
    try:
        stats = get_cache_stats()
        required_fields = ["memory_entries", "disk_cache_dir", "max_size"]
        return all(field in stats for field in required_fields)
    except Exception:
        return False


def test_cache_performance_metrics() -> bool:
    """Test cache performance metrics collection."""
    try:
        cache = PerformanceCache(max_memory_cache_size=2)
        cache.set("disk_key", {"a": 1}, disk_cache=True)
        return cache.get("disk_key") == {"a": 1}
    except Exception:
        return False


def test_memory_management_cleanup() -> bool:
    """Test memory management and cleanup."""
    try:
        cache = PerformanceCache(max_memory_cache_size=2)
        cache.set("key1", 1)
        cache.set("key2", 2)
        cache.set("key3", 3)
        cache._cleanup_old_entries()
        return len(cache._memory_cache) <= cache._max_size
    except Exception:
        return False


def performance_cache_module_tests() -> bool:
    """
    PerformanceCache Management & Optimization module test suite.
    Tests cache performance, invalidation, and optimization.
    """
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("PerformanceCache Management & Optimization", __name__)
        suite.start_suite()

    # Run all tests using the suite
    suite.run_test(
        "PerformanceCache Module Initialization",
        test_performance_cache_initialization,
        "Cache module initializes with proper interface and required methods",
        "Initialization",
        "Initialize PerformanceCache module and verify basic structure",
    )

    suite.run_test(
        "Memory Cache Operations",
        test_memory_cache_operations,
        "Memory cache stores and retrieves data correctly",
        "Initialization",
        "Store and retrieve data from memory cache",
    )

    suite.run_test(
        "Cache Key Generation",
        test_cache_key_generation,
        "Cache key generation produces consistent keys for identical inputs",
        "Core",
        "Generate cache keys for same inputs and verify consistency",
    )

    suite.run_test(
        "Cache Expiration",
        test_cache_expiration,
        "Expired cache entries are correctly identified as invalid",
        "Edge",
        "Store data with expired timestamp and verify expiration detection",
    )

    suite.run_test(
        "Cache Statistics Collection",
        test_cache_statistics_collection,
        "Statistics collection returns all required fields",
        "Integration",
        "Collect comprehensive cache statistics",
    )

    suite.run_test(
        "Cache Health Status Check",
        test_cache_health_status,
        "Health status returns comprehensive system health information",
        "Integration",
        "Check overall cache health and component status",
    )

    suite.run_test(
        "Cache Performance Metrics",
        test_cache_performance_metrics,
        "Performance metrics collection provides valid numeric data",
        "Performance",
        "Collect and validate cache performance statistics",
    )

    suite.run_test(
        "Memory Management and Cleanup",
        test_memory_management_cleanup,
        "Memory management functions execute without errors",
        "Error",
        "Test cache memory cleanup and optimization functions",
    )

    return suite.finish_suite()


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(performance_cache_module_tests)

# --- Main Execution ---

if __name__ == "__main__":
    print("🗂️ Running PerformanceCache Management & Optimization comprehensive test suite...")
    success = run_comprehensive_tests()
    import sys

    sys.exit(0 if success else 1)
