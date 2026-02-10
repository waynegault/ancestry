#!/usr/bin/env python3
"""
core/cache - Unified Cache Interface Package

Provides a consistent, type-safe cache abstraction layer with:
- Protocol-based Cache interface for dependency inversion
- Concrete adapters: MemoryCache, DiskCache, TTLCache
- Explicit invalidation patterns and TTL management
- Cache versioning for safe deployments
- Consistent statistics and health reporting

Usage:
    from core.cache import Cache, MemoryCache, DiskCache, CacheConfig

    # Create a memory cache with 5-minute TTL
    cache = MemoryCache(config=CacheConfig(default_ttl=300))

    # Store and retrieve values
    cache.set("key", {"data": "value"})
    result = cache.get("key")

    # Invalidation patterns
    cache.invalidate("key")              # Single key
    cache.invalidate_pattern("user:*")   # Pattern-based
    cache.invalidate_all()               # Full clear

    # Statistics
    stats = cache.get_stats()
    print(f"Hit rate: {stats.hit_rate}%")
"""

from core.cache.adapters import (
    DiskCacheAdapter,
    MemoryCache,
    NullCache,
    TTLCache,
)
from core.cache.interface import (
    Cache,
    CacheConfig,
    CacheEntry,
    CacheKey,
    InvalidationPattern,
)

__all__ = [
    "Cache",
    "CacheConfig",
    "CacheEntry",
    "CacheKey",
    "DiskCacheAdapter",
    "InvalidationPattern",
    "MemoryCache",
    "NullCache",
    "TTLCache",
]

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    from testing.test_framework import TestSuite

    suite = TestSuite("core.cache __init__", "core/cache/__init__.py")

    def test_cache_is_class():
        assert isinstance(Cache, type), "Cache should be a class/protocol"

    def test_cache_config_is_class():
        assert isinstance(CacheConfig, type), "CacheConfig should be a class"

    def test_memory_cache_is_class():
        assert isinstance(MemoryCache, type), "MemoryCache should be a class"

    def test_null_cache_is_class():
        assert isinstance(NullCache, type), "NullCache should be a class"

    def test_ttl_cache_is_class():
        assert isinstance(TTLCache, type), "TTLCache should be a class"

    def test_disk_cache_adapter_is_class():
        assert isinstance(DiskCacheAdapter, type), "DiskCacheAdapter should be a class"

    def test_memory_cache_instantiation():
        cache = MemoryCache()
        assert isinstance(cache, MemoryCache), "Should create a MemoryCache instance"
        assert hasattr(cache, "get"), "MemoryCache should have get method"
        assert hasattr(cache, "set"), "MemoryCache should have set method"

    def test_all_exports():
        assert isinstance(__all__, list), "__all__ should be a list"
        assert "Cache" in __all__, "__all__ should contain Cache"
        assert "MemoryCache" in __all__, "__all__ should contain MemoryCache"
        assert "NullCache" in __all__, "__all__ should contain NullCache"

    suite.run_test("Cache protocol is a class", test_cache_is_class)
    suite.run_test("CacheConfig is a class", test_cache_config_is_class)
    suite.run_test("MemoryCache is a class", test_memory_cache_is_class)
    suite.run_test("NullCache is a class", test_null_cache_is_class)
    suite.run_test("TTLCache is a class", test_ttl_cache_is_class)
    suite.run_test("DiskCacheAdapter is a class", test_disk_cache_adapter_is_class)
    suite.run_test("MemoryCache can be instantiated with default config", test_memory_cache_instantiation)
    suite.run_test("__all__ contains expected exports", test_all_exports)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
