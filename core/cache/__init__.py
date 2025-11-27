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
