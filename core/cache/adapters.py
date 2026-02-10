#!/usr/bin/env python3
"""
core/cache/adapters.py - Concrete Cache Implementations

Provides ready-to-use cache implementations that conform to the Cache protocol:
- MemoryCache: Fast in-memory cache with LRU eviction
- DiskCacheAdapter: Persistent disk-based cache using diskcache
- TTLCache: Memory cache with automatic TTL-based expiration
- NullCache: No-op cache for testing and disabling caching

All adapters provide consistent statistics and health reporting.
"""


import sys
import threading
import time
from collections import OrderedDict
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

# Add parent directory for imports when running as script
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import logging

from core.cache.interface import (
    CacheConfig,
    CacheEntry,
    CacheKey,
    CacheTTL,
    InvalidationPattern,
)
from core.cache_backend import CacheHealth, CacheStats
from core.registry_utils import auto_register_module

logger = logging.getLogger(__name__)
auto_register_module(globals(), __name__)


# =============================================================================
# MEMORY CACHE
# =============================================================================


class MemoryCache:
    """Fast in-memory cache with LRU eviction and TTL support.

    Thread-safe implementation using OrderedDict for LRU ordering.
    Suitable for single-process applications with moderate cache sizes.

    Features:
    - O(1) get/set operations
    - LRU eviction when max_entries exceeded
    - Optional TTL per entry
    - Pattern-based invalidation
    - Comprehensive statistics
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize memory cache.

        Args:
            config: Cache configuration. Uses defaults if None.
        """
        self._config = config or CacheConfig(name="memory_cache")
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0
        self._errors = 0

    @property
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config

    def _make_key(self, key: CacheKey) -> str:
        """Create namespaced key if namespace configured."""
        if self._config.namespace:
            return f"{self._config.namespace}{key}"
        return key

    def _strip_namespace(self, full_key: str) -> str:
        """Remove namespace prefix from key."""
        if self._config.namespace and full_key.startswith(self._config.namespace):
            return full_key[len(self._config.namespace) :]
        return full_key

    def _evict_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        expired_keys = [k for k, v in self._entries.items() if v.is_expired]
        for key in expired_keys:
            del self._entries[key]
        self._expired += len(expired_keys)
        return len(expired_keys)

    def _evict_lru(self, count: int = 1) -> int:
        """Remove least recently used entries. Returns count removed."""
        removed = 0
        while removed < count and self._entries:
            self._entries.popitem(last=False)
            removed += 1
        self._evictions += removed
        return removed

    def _enforce_limits(self) -> None:
        """Enforce max_entries limit via LRU eviction."""
        if self._config.max_entries > 0:
            excess = len(self._entries) - self._config.max_entries
            if excess > 0:
                self._evict_lru(excess)

    def get(self, key: CacheKey) -> Any | None:
        """Retrieve a value from the cache."""
        full_key = self._make_key(key)
        with self._lock:
            entry = self._entries.get(full_key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._entries[full_key]
                self._expired += 1
                self._misses += 1
                return None

            # Move to end for LRU tracking
            self._entries.move_to_end(full_key)
            entry.hit_count += 1
            self._hits += 1
            return entry.value

    def set(self, key: CacheKey, value: Any, ttl: CacheTTL | None = None) -> bool:
        """Store a value in the cache."""
        full_key = self._make_key(key)
        effective_ttl = ttl if ttl is not None else self._config.default_ttl

        with self._lock:
            try:
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl=effective_ttl,
                    version=self._config.version,
                    size_bytes=sys.getsizeof(value),
                )

                # Store and move to end
                self._entries[full_key] = entry
                self._entries.move_to_end(full_key)

                # Enforce limits
                self._enforce_limits()
                return True

            except Exception as exc:
                logger.warning(f"Failed to set cache key {key}: {exc}")
                self._errors += 1
                return False

    def delete(self, key: CacheKey) -> bool:
        """Remove a key from the cache."""
        full_key = self._make_key(key)
        with self._lock:
            if full_key in self._entries:
                del self._entries[full_key]
                return True
            return False

    def exists(self, key: CacheKey) -> bool:
        """Check if a key exists and is not expired."""
        full_key = self._make_key(key)
        with self._lock:
            entry = self._entries.get(full_key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._entries[full_key]
                self._expired += 1
                return False
            return True

    def invalidate(self, key: CacheKey) -> bool:
        """Invalidate a specific cache key."""
        return self.delete(key)

    def invalidate_pattern(self, pattern: str, pattern_type: str = "glob") -> int:
        """Invalidate all keys matching a pattern."""
        inv_pattern = InvalidationPattern(pattern=pattern, pattern_type=pattern_type)
        count = 0

        with self._lock:
            keys_to_remove = []
            for full_key in self._entries:
                key = self._strip_namespace(full_key)
                if inv_pattern.matches(key):
                    keys_to_remove.append(full_key)

            for full_key in keys_to_remove:
                del self._entries[full_key]
                count += 1

        return count

    def invalidate_all(self) -> bool:
        """Invalidate all entries in the cache."""
        with self._lock:
            self._entries.clear()
            return True

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            total_size = sum(e.size_bytes for e in self._entries.values())
            return CacheStats(
                name=self._config.name,
                kind="memory",
                hits=self._hits,
                misses=self._misses,
                entries=len(self._entries),
                size_bytes=total_size,
                max_size_bytes=self._config.max_size_bytes,
                evictions=self._evictions,
                expired=self._expired,
                errors=self._errors,
            )

    def get_health(self) -> CacheHealth:
        """Get cache health status."""
        stats = self.get_stats()
        recommendations: list[str] = []

        # Determine status based on hit rate and errors
        if stats.errors > 0:
            status = "degraded"
            message = f"{stats.errors} errors encountered"
            recommendations.append("Check error logs for cache issues")
        elif stats.hit_rate < 50 and (stats.hits + stats.misses) > 100:
            status = "degraded"
            message = f"Low hit rate: {stats.hit_rate:.1f}%"
            recommendations.append("Consider increasing cache TTL or size")
        else:
            status = "healthy"
            message = f"Operating normally ({len(self._entries)} entries)"

        return CacheHealth(
            name=self._config.name,
            status=status,
            message=message,
            hit_rate=stats.hit_rate,
            is_available=True,
            recommendations=recommendations,
        )

    def keys(self, pattern: str = "*") -> list[CacheKey]:
        """Get all keys matching a pattern."""
        with self._lock:
            result = []
            for full_key in self._entries:
                key = self._strip_namespace(full_key)
                if fnmatch(key, pattern):
                    result.append(key)
            return result

    def get_entry(self, key: CacheKey) -> CacheEntry | None:
        """Get cache entry with metadata."""
        full_key = self._make_key(key)
        with self._lock:
            entry = self._entries.get(full_key)
            if entry is None or entry.is_expired:
                return None
            return entry


# =============================================================================
# TTL CACHE (Memory with stricter TTL enforcement)
# =============================================================================


class TTLCache(MemoryCache):
    """Memory cache with stricter TTL enforcement.

    Extends MemoryCache with periodic cleanup of expired entries.
    Better for caches where TTL accuracy is important.
    """

    def __init__(self, config: CacheConfig | None = None, cleanup_interval: float = 60.0) -> None:
        """Initialize TTL cache.

        Args:
            config: Cache configuration.
            cleanup_interval: Seconds between automatic cleanup runs.
        """
        super().__init__(config or CacheConfig(name="ttl_cache"))
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup >= self._cleanup_interval:
            self._evict_expired()
            self._last_cleanup = now

    def get(self, key: CacheKey) -> Any | None:
        """Retrieve value, running cleanup if needed."""
        self._maybe_cleanup()
        return super().get(key)

    def set(self, key: CacheKey, value: Any, ttl: CacheTTL | None = None) -> bool:
        """Store value, running cleanup if needed."""
        self._maybe_cleanup()
        return super().set(key, value, ttl)


# =============================================================================
# DISK CACHE ADAPTER
# =============================================================================


class DiskCacheAdapter:
    """Adapter wrapping diskcache.Cache with unified interface.

    Provides persistent disk-based caching with the standard Cache protocol.
    Suitable for large caches that need to survive process restarts.
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize disk cache adapter.

        Args:
            config: Cache configuration.
            cache_dir: Directory for disk cache storage.
        """
        self._config = config or CacheConfig(name="disk_cache")
        self._cache_dir = cache_dir or Path("Cache") / self._config.name

        # Ensure directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize diskcache
        try:
            from diskcache import Cache as DiskCache

            self._cache: DiskCache | None = DiskCache(
                str(self._cache_dir),
                size_limit=self._config.max_size_bytes if self._config.max_size_bytes > 0 else int(2e9),
                eviction_policy="least-recently-used",
                statistics=self._config.enable_stats,
            )
        except ImportError:
            logger.warning("diskcache not available, using NullCache fallback")
            self._cache = None

        # Statistics (for when diskcache stats not available)
        self._hits = 0
        self._misses = 0
        self._errors = 0

    @property
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config

    def _make_key(self, key: CacheKey) -> str:
        """Create versioned, namespaced key."""
        parts = [self._config.version]
        if self._config.namespace:
            parts.append(self._config.namespace)
        parts.append(key)
        return ":".join(parts)

    def get(self, key: CacheKey) -> Any | None:
        """Retrieve a value from the cache."""
        if self._cache is None:
            self._misses += 1
            return None

        full_key = self._make_key(key)
        try:
            from diskcache.core import ENOVAL

            value = self._cache.get(full_key, default=ENOVAL)
            if value is ENOVAL:
                self._misses += 1
                return None
            self._hits += 1
            return value
        except Exception as exc:
            logger.warning(f"Disk cache get error for {key}: {exc}")
            self._errors += 1
            self._misses += 1
            return None

    def set(self, key: CacheKey, value: Any, ttl: CacheTTL | None = None) -> bool:
        """Store a value in the cache."""
        if self._cache is None:
            return False

        full_key = self._make_key(key)
        effective_ttl = ttl if ttl is not None else self._config.default_ttl

        try:
            expire = effective_ttl if effective_ttl > 0 else None
            self._cache.set(full_key, value, expire=expire)
            return True
        except Exception as exc:
            logger.warning(f"Disk cache set error for {key}: {exc}")
            self._errors += 1
            return False

    def delete(self, key: CacheKey) -> bool:
        """Remove a key from the cache."""
        if self._cache is None:
            return False

        full_key = self._make_key(key)
        try:
            return self._cache.delete(full_key)
        except Exception as exc:
            logger.warning(f"Disk cache delete error for {key}: {exc}")
            self._errors += 1
            return False

    def exists(self, key: CacheKey) -> bool:
        """Check if a key exists."""
        if self._cache is None:
            return False

        full_key = self._make_key(key)
        try:
            return full_key in self._cache
        except Exception:
            return False

    def invalidate(self, key: CacheKey) -> bool:
        """Invalidate a specific cache key."""
        return self.delete(key)

    def invalidate_pattern(self, pattern: str, pattern_type: str = "glob") -> int:
        """Invalidate all keys matching a pattern."""
        if self._cache is None:
            return 0

        inv_pattern = InvalidationPattern(pattern=pattern, pattern_type=pattern_type)
        count = 0

        try:
            # Iterate all keys and check pattern
            for full_key in list(self._cache):
                # Extract key part after version:namespace:
                parts = str(full_key).split(":", 2)
                key = parts[-1] if parts else str(full_key)

                if inv_pattern.matches(key):
                    self._cache.delete(full_key)
                    count += 1
        except Exception as exc:
            logger.warning(f"Disk cache pattern invalidation error: {exc}")
            self._errors += 1

        return count

    def invalidate_all(self) -> bool:
        """Invalidate all entries in the cache."""
        if self._cache is None:
            return False

        try:
            self._cache.clear()
            return True
        except Exception as exc:
            logger.warning(f"Disk cache clear error: {exc}")
            self._errors += 1
            return False

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        if self._cache is None:
            return CacheStats(
                name=self._config.name,
                kind="disk",
                errors=self._errors,
            )

        try:
            volume = self._cache.volume()
            return CacheStats(
                name=self._config.name,
                kind="disk",
                hits=self._hits,
                misses=self._misses,
                entries=len(self._cache),
                size_bytes=volume,
                max_size_bytes=self._config.max_size_bytes,
                errors=self._errors,
            )
        except Exception:
            return CacheStats(
                name=self._config.name,
                kind="disk",
                hits=self._hits,
                misses=self._misses,
                errors=self._errors,
            )

    def get_health(self) -> CacheHealth:
        """Get cache health status."""
        if self._cache is None:
            return CacheHealth(
                name=self._config.name,
                status="critical",
                message="Disk cache not available",
                is_available=False,
            )

        stats = self.get_stats()
        recommendations: list[str] = []

        if stats.errors > 0:
            status = "degraded"
            message = f"{stats.errors} errors encountered"
            recommendations.append("Check disk space and permissions")
        else:
            status = "healthy"
            message = f"Operating normally ({stats.entries} entries, {stats.size_bytes / 1024 / 1024:.1f}MB)"

        return CacheHealth(
            name=self._config.name,
            status=status,
            message=message,
            hit_rate=stats.hit_rate,
            is_available=True,
            recommendations=recommendations,
        )

    def keys(self, pattern: str = "*") -> list[CacheKey]:
        """Get all keys matching a pattern."""
        if self._cache is None:
            return []

        result = []
        try:
            for full_key in self._cache:
                parts = str(full_key).split(":", 2)
                key = parts[-1] if parts else str(full_key)
                if fnmatch(key, pattern):
                    result.append(key)
        except Exception:
            pass
        return result

    def get_entry(self, key: CacheKey) -> CacheEntry | None:
        """Get cache entry with metadata."""
        value = self.get(key)
        if value is None:
            return None

        # DiskCache doesn't store metadata, create minimal entry
        return CacheEntry(
            key=key,
            value=value,
            version=self._config.version,
        )

    def close(self) -> None:
        """Close the disk cache."""
        if self._cache is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._cache.close()


# =============================================================================
# NULL CACHE
# =============================================================================


class NullCache:
    """No-op cache that never stores anything.

    Useful for:
    - Testing without actual caching
    - Disabling caching in specific contexts
    - Placeholder when cache system unavailable
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize null cache."""
        self._config = config or CacheConfig(name="null_cache")
        self._calls = 0

    @property
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config

    def get(self, key: CacheKey) -> Any | None:  # noqa: ARG002
        """Always returns None."""
        self._calls += 1
        return None

    def set(self, key: CacheKey, value: Any, ttl: CacheTTL | None = None) -> bool:  # noqa: ARG002
        """Always succeeds but stores nothing."""
        self._calls += 1
        return True

    def delete(self, key: CacheKey) -> bool:  # noqa: ARG002
        """Always returns False (nothing to delete)."""
        self._calls += 1
        return False

    def exists(self, key: CacheKey) -> bool:  # noqa: ARG002
        """Always returns False."""
        self._calls += 1
        return False

    def invalidate(self, key: CacheKey) -> bool:
        """Always returns False."""
        return self.delete(key)

    def invalidate_pattern(self, pattern: str, pattern_type: str = "glob") -> int:
        """Always returns 0."""
        _ = pattern, pattern_type  # Unused
        self._calls += 1
        return 0

    def invalidate_all(self) -> bool:
        """Always succeeds."""
        self._calls += 1
        return True

    def get_stats(self) -> CacheStats:
        """Return empty statistics."""
        return CacheStats(
            name=self._config.name,
            kind="null",
            extra={"calls": self._calls},
        )

    def get_health(self) -> CacheHealth:
        """Always healthy (nothing can fail)."""
        return CacheHealth(
            name=self._config.name,
            status="healthy",
            message="Null cache (no-op)",
            is_available=True,
        )

    @staticmethod
    def keys(pattern: str = "*") -> list[CacheKey]:  # noqa: ARG004
        """Always returns empty list."""
        return []

    @staticmethod
    def get_entry(key: CacheKey) -> CacheEntry | None:  # noqa: ARG004
        """Always returns None."""
        return None


# =============================================================================
# TESTS
# =============================================================================


def _test_memory_cache_basic_operations() -> None:
    """Test basic get/set/delete operations."""
    cache = MemoryCache(CacheConfig(name="test"))

    # Set and get
    assert cache.set("key1", "value1") is True
    assert cache.get("key1") == "value1"

    # Exists
    assert cache.exists("key1") is True
    assert cache.exists("nonexistent") is False

    # Delete
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    assert cache.delete("nonexistent") is False


def _test_memory_cache_ttl() -> None:
    """Test TTL-based expiration."""
    cache = MemoryCache(CacheConfig(name="test", default_ttl=1))

    cache.set("expires", "value")
    assert cache.get("expires") == "value"

    # Wait for expiration
    time.sleep(1.1)
    assert cache.get("expires") is None


def _test_memory_cache_lru_eviction() -> None:
    """Test LRU eviction when max_entries exceeded."""
    cache = MemoryCache(CacheConfig(name="test", max_entries=3))

    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    # Access key1 to make it recently used
    cache.get("key1")

    # Add key4, should evict key2 (least recently used)
    cache.set("key4", "value4")

    assert cache.get("key1") == "value1"  # Still present (recently accessed)
    assert cache.get("key2") is None  # Evicted
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"


def _test_memory_cache_pattern_invalidation() -> None:
    """Test pattern-based invalidation."""
    cache = MemoryCache()

    cache.set("user:123:profile", "data1")
    cache.set("user:456:profile", "data2")
    cache.set("user:123:settings", "data3")
    cache.set("session:abc", "data4")

    # Invalidate all user profiles
    count = cache.invalidate_pattern("user:*:profile", "glob")
    assert count == 2

    assert cache.get("user:123:profile") is None
    assert cache.get("user:456:profile") is None
    assert cache.get("user:123:settings") == "data3"  # Not invalidated
    assert cache.get("session:abc") == "data4"  # Not invalidated


def _test_memory_cache_statistics() -> None:
    """Test statistics tracking."""
    cache = MemoryCache(CacheConfig(name="stats_test"))

    cache.set("key1", "value1")
    cache.get("key1")  # Hit
    cache.get("key1")  # Hit
    cache.get("missing")  # Miss

    stats = cache.get_stats()
    assert stats.hits == 2
    assert stats.misses == 1
    assert stats.entries == 1
    assert stats.hit_rate > 60  # Should be ~66%


def _test_memory_cache_namespace() -> None:
    """Test namespace prefixing."""
    cache = MemoryCache(CacheConfig(name="test", namespace="app:"))

    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

    # Keys should return without namespace
    keys = cache.keys()
    assert "key1" in keys


def _test_ttl_cache_cleanup() -> None:
    """Test TTLCache automatic cleanup."""
    cache = TTLCache(
        CacheConfig(name="test", default_ttl=1),
        cleanup_interval=0.1,  # Fast cleanup for testing
    )

    cache.set("expires", "value")
    assert cache.get("expires") == "value"

    time.sleep(1.2)

    # Next access should trigger cleanup
    cache.get("anything")

    stats = cache.get_stats()
    assert stats.expired >= 1


def _test_null_cache_operations() -> None:
    """Test NullCache always returns None/False."""
    cache = NullCache()

    assert cache.set("key", "value") is True  # Accepts but doesn't store
    assert cache.get("key") is None
    assert cache.exists("key") is False
    assert cache.delete("key") is False
    assert cache.invalidate_pattern("*") == 0
    assert cache.keys() == []


def _test_disk_cache_adapter_fallback() -> None:
    """Test DiskCacheAdapter handles missing diskcache gracefully."""
    # This test verifies the adapter doesn't crash without diskcache
    # In practice, diskcache is installed, so this mainly tests error handling
    cache = DiskCacheAdapter(CacheConfig(name="test_disk"))

    # Should work regardless of diskcache availability
    stats = cache.get_stats()
    assert stats.name == "test_disk"
    assert stats.kind == "disk"

    health = cache.get_health()
    assert health.name == "test_disk"


def _test_cache_entry_metadata() -> None:
    """Test get_entry returns proper metadata."""
    cache = MemoryCache(CacheConfig(name="test", version="v2"))

    cache.set("key1", {"data": "value"})
    entry = cache.get_entry("key1")

    assert entry is not None
    assert entry.key == "key1"
    assert entry.value == {"data": "value"}
    assert entry.version == "v2"
    assert entry.hit_count == 0  # get_entry doesn't increment hits


def module_tests() -> bool:
    """Run module tests for cache adapters."""
    from testing.test_framework import TestSuite

    suite = TestSuite("core.cache.adapters", "core/cache/adapters.py")

    suite.run_test(
        "MemoryCache basic operations",
        _test_memory_cache_basic_operations,
        "Validates get/set/delete/exists work correctly",
    )

    suite.run_test(
        "MemoryCache TTL expiration",
        _test_memory_cache_ttl,
        "Validates TTL-based entry expiration",
    )

    suite.run_test(
        "MemoryCache LRU eviction",
        _test_memory_cache_lru_eviction,
        "Validates LRU eviction when max_entries exceeded",
    )

    suite.run_test(
        "MemoryCache pattern invalidation",
        _test_memory_cache_pattern_invalidation,
        "Validates glob pattern-based invalidation",
    )

    suite.run_test(
        "MemoryCache statistics",
        _test_memory_cache_statistics,
        "Validates hit/miss tracking and hit rate calculation",
    )

    suite.run_test(
        "MemoryCache namespace",
        _test_memory_cache_namespace,
        "Validates namespace prefixing for keys",
    )

    suite.run_test(
        "TTLCache automatic cleanup",
        _test_ttl_cache_cleanup,
        "Validates periodic expired entry cleanup",
    )

    suite.run_test(
        "NullCache operations",
        _test_null_cache_operations,
        "Validates no-op cache behavior",
    )

    suite.run_test(
        "DiskCacheAdapter fallback",
        _test_disk_cache_adapter_fallback,
        "Validates graceful handling without diskcache",
    )

    suite.run_test(
        "Cache entry metadata",
        _test_cache_entry_metadata,
        "Validates get_entry returns proper CacheEntry",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    from testing.test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
