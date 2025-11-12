"""
core/unified_cache_manager.py - Centralized Cache Management System

Provides a unified, thread-safe cache for API responses and data across all
action modules. Replaces scattered caching logic with a single source of truth.

Phase 5 Implementation - Cache Hit Rate Optimization (Opportunity #3)
"""

import copy
import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for standard_imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)


@dataclass
class CacheEntry:
    """Individual cache entry with metadata and statistics."""

    key: str
    data: Any
    timestamp: float
    ttl_seconds: int
    hit_count: int = 0
    service: str = ""
    endpoint: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        elapsed = time.time() - self.timestamp
        return elapsed > self.ttl_seconds


class UnifiedCacheManager:
    """
    Centralized, thread-safe cache for API responses and shared data.

    Features:
    - Thread-safe with Lock-based synchronization
    - Service and endpoint-aware statistics
    - TTL-based automatic expiration
    - Optional size limits with LRU eviction
    - Per-service hit/miss tracking
    """

    def __init__(self, max_entries: int = 10000):
        """
        Initialize the unified cache manager.

        Args:
            max_entries: Maximum number of cache entries before eviction (default: 10,000)
        """
        self._entries: dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._created_at = time.time()
        self._session_id = ""

        # Statistics tracking
        self._stats: dict[str, dict[str, Any]] = {
            "ancestry": {
                "hits": 0,
                "misses": 0,
                "endpoints": {},
            },
            "ai": {
                "hits": 0,
                "misses": 0,
                "endpoints": {},
            },
            "global": {
                "hits": 0,
                "misses": 0,
                "total_entries": 0,
            },
        }

    def get(self, service: str, endpoint: str, key: str) -> Optional[Any]:
        """
        Retrieve a cached value if it exists and hasn't expired.

        Args:
            service: Service name (ancestry, ai, etc.)
            endpoint: Endpoint name (combined_details, rel_prob, etc.)
            key: Cache key for the value

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._entries.get(key)

            # Initialize service stats if needed
            if service not in self._stats:
                self._stats[service] = {"hits": 0, "misses": 0, "endpoints": {}}

            # Initialize endpoint stats if needed
            if endpoint not in self._stats[service]["endpoints"]:
                self._stats[service]["endpoints"][endpoint] = {"hits": 0, "misses": 0}

            if entry is None or entry.is_expired:
                # Cache miss
                self._stats[service]["misses"] += 1
                self._stats[service]["endpoints"][endpoint]["misses"] += 1
                self._stats["global"]["misses"] += 1
                if entry and entry.is_expired:
                    del self._entries[key]  # Clean up expired entry
                logger.debug(f"Cache MISS: {service}.{endpoint}.{key[:20]}")
                return None

            # Cache hit
            entry.hit_count += 1
            self._stats[service]["hits"] += 1
            self._stats[service]["endpoints"][endpoint]["hits"] += 1
            self._stats["global"]["hits"] += 1

            logger.debug(
                f"Cache HIT: {service}.{endpoint}.{key[:20]} "
                f"(age: {time.time() - entry.timestamp:.1f}s, hits: {entry.hit_count})"
            )

            # Return deep copy to prevent external modification
            return copy.deepcopy(entry.data)

    def set(
        self,
        service: str,
        endpoint: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache a value with optional TTL override.

        Args:
            service: Service name (ancestry, ai, etc.)
            endpoint: Endpoint name (combined_details, rel_prob, etc.)
            key: Cache key for the value
            value: Value to cache
            ttl: Time-to-live in seconds (uses service default if None)
        """
        with self._lock:
            # Enforce size limit before adding new entry
            if len(self._entries) >= self._max_entries and key not in self._entries:
                self._enforce_size_limit()

            # Determine TTL
            if ttl is None:
                ttl = self._get_default_ttl(service)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=copy.deepcopy(value),  # Store copy to prevent mutation
                timestamp=time.time(),
                ttl_seconds=ttl,
                service=service,
                endpoint=endpoint,
            )

            self._entries[key] = entry
            self._stats["global"]["total_entries"] = len(self._entries)

            # Initialize service stats if needed
            if service not in self._stats:
                self._stats[service] = {"hits": 0, "misses": 0, "endpoints": {}}

            if endpoint not in self._stats[service]["endpoints"]:
                self._stats[service]["endpoints"][endpoint] = {"hits": 0, "misses": 0}

            logger.debug(f"Cache SET: {service}.{endpoint}.{key[:20]} (TTL: {ttl}s)")

    def invalidate(
        self,
        service: Optional[str] = None,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries by service, endpoint, or specific key.

        Args:
            service: Optional service to invalidate (None = all services)
            endpoint: Optional endpoint to invalidate (requires service)
            key: Optional specific key to invalidate

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            count = 0

            if key is not None:
                # Invalidate specific key
                if key in self._entries:
                    del self._entries[key]
                    count = 1
            elif service is not None and endpoint is not None:
                # Invalidate all entries for specific service + endpoint
                keys_to_delete = [
                    k for k, v in self._entries.items()
                    if v.service == service and v.endpoint == endpoint
                ]
                for k in keys_to_delete:
                    del self._entries[k]
                count = len(keys_to_delete)
            elif service is not None:
                # Invalidate all entries for service
                keys_to_delete = [
                    k for k, v in self._entries.items()
                    if v.service == service
                ]
                for k in keys_to_delete:
                    del self._entries[k]
                count = len(keys_to_delete)
            else:
                # Invalidate entire cache
                count = len(self._entries)
                self._entries.clear()

            self._stats["global"]["total_entries"] = len(self._entries)
            logger.info(f"Cache invalidated: {count} entries")
            return count

    def get_stats(self, endpoint: Optional[str] = None) -> dict[str, Any]:
        """
        Get cache statistics.

        Args:
            endpoint: Optional specific endpoint to filter by

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            if endpoint is not None:
                # Return stats for specific endpoint across all services
                stats = {
                    "endpoint": endpoint,
                    "total_hits": 0,
                    "total_misses": 0,
                    "by_service": {},
                }
                for service, service_stats in self._stats.items():
                    if service == "global":
                        continue
                    if endpoint in service_stats.get("endpoints", {}):
                        ep_stats = service_stats["endpoints"][endpoint]
                        stats["by_service"][service] = ep_stats
                        stats["total_hits"] += ep_stats["hits"]
                        stats["total_misses"] += ep_stats["misses"]

                total = stats["total_hits"] + stats["total_misses"]
                stats["hit_rate_percent"] = (
                    (stats["total_hits"] / total * 100) if total > 0 else 0.0
                )
                return stats

            # Return overall statistics
            stats = {
                "total_entries": len(self._entries),
                "max_entries": self._max_entries,
                "uptime_seconds": time.time() - self._created_at,
                "by_service": {},
            }

            for service, service_stats in self._stats.items():
                if service == "global":
                    continue

                hits = service_stats["hits"]
                misses = service_stats["misses"]
                total = hits + misses

                stats["by_service"][service] = {
                    "hits": hits,
                    "misses": misses,
                    "hit_rate_percent": (hits / total * 100) if total > 0 else 0.0,
                    "endpoints": service_stats.get("endpoints", {}),
                }

            # Global statistics
            global_hits = self._stats["global"]["hits"]
            global_misses = self._stats["global"]["misses"]
            global_total = global_hits + global_misses
            stats["global"] = {
                "hits": global_hits,
                "misses": global_misses,
                "hit_rate_percent": (global_hits / global_total * 100) if global_total > 0 else 0.0,
            }

            return stats

    def clear(self) -> int:
        """Clear entire cache and reset statistics."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._stats = {
                "ancestry": {"hits": 0, "misses": 0, "endpoints": {}},
                "ai": {"hits": 0, "misses": 0, "endpoints": {}},
                "global": {"hits": 0, "misses": 0, "total_entries": 0},
            }
            logger.info(f"Cache cleared: {count} entries")
            return count

    # Private methods

    def _get_default_ttl(self, service: str) -> int:
        """Get default TTL for service."""
        ttls = {
            "ancestry": 2400,  # 40 minutes
            "ai": 3600,  # 1 hour
            "default": 300,  # 5 minutes
        }
        return ttls.get(service, ttls["default"])

    def _enforce_size_limit(self) -> None:
        """Evict least-recently-used entries if cache exceeds max size."""
        if len(self._entries) <= self._max_entries:
            return

        # Sort by hit_count (LRU proxy: entries with fewer hits are less valuable)
        entries_by_value = sorted(
            self._entries.items(),
            key=lambda item: item[1].hit_count,
        )

        # Remove entries until we're below max
        target_size = self._max_entries * 90 // 100  # Remove to 90% capacity
        entries_to_remove = len(self._entries) - target_size

        for i in range(entries_to_remove):
            key, _ = entries_by_value[i]
            del self._entries[key]
            logger.debug(f"Cache evicted (LRU): {key[:20]}")

    # Special methods

    def __len__(self) -> int:
        """Return current number of cache entries."""
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        """Return string representation."""
        with self._lock:
            stats = self.get_stats()
            hit_rate = stats["global"]["hit_rate_percent"]
            return (
                f"UnifiedCacheManager("
                f"entries={len(self._entries)}/{self._max_entries}, "
                f"hit_rate={hit_rate:.1f}%, "
                f"uptime={stats['uptime_seconds']:.0f}s)"
            )


# Global cache instance (singleton pattern)
_unified_cache: Optional[UnifiedCacheManager] = None


def get_unified_cache() -> UnifiedCacheManager:
    """Get or create the global unified cache instance (singleton)."""
    global _unified_cache  # noqa: PLW0603 - Singleton pattern requires global
    if _unified_cache is None:
        _unified_cache = UnifiedCacheManager()
        logger.info("🚀 Unified cache manager initialized")
    return _unified_cache


def create_ancestry_cache_config() -> dict[str, int]:
    """Create recommended cache TTL configuration for Action 6 endpoints."""
    return {
        "combined_details": 2400,  # 40 min
        "relationship_prob": 2400,
        "ethnicity_regions": 2400,
        "badge_details": 2400,
        "ladder_details": 2400,
        "tree_search": 2400,
    }


def generate_cache_key(
    service: str,
    endpoint: str,
    params: dict[str, Any] | str,
) -> str:
    """
    Generate a consistent cache key from service, endpoint, and parameters.

    Args:
        service: Service name (ancestry, ai, etc.)
        endpoint: Endpoint name (combined_details, rel_prob, etc.)
        params: Parameters dict (will be JSON-serialized) or string UUID

    Returns:
        Cache key string
    """
    if isinstance(params, str):
        # Simple string parameter (e.g., UUID)
        return f"cache:{service}:{endpoint}:{params.upper()}"

    # Complex parameters - hash them
    try:
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        return f"cache:{service}:{endpoint}:{params_hash}"
    except (TypeError, ValueError):
        # Fallback for non-serializable objects
        params_hash = hashlib.sha256(str(params).encode()).hexdigest()[:16]
        return f"cache:{service}:{endpoint}:{params_hash}"


def test_unified_cache_manager() -> bool:
    """Test unified cache manager functionality."""
    try:
        logger.info("Testing UnifiedCacheManager...")

        # Test 1: Basic get/set
        cache = get_unified_cache()
        cache.set("ancestry", "combined_details", "cache:test:uuid1", {"id": "test"})
        result = cache.get("ancestry", "combined_details", "cache:test:uuid1")
        assert result == {"id": "test"}, "Basic get/set failed"

        # Test 2: Cache miss
        result = cache.get("ancestry", "combined_details", "cache:nonexistent")
        assert result is None, "Cache miss should return None"

        # Test 3: Statistics
        stats = cache.get_stats()
        assert "by_service" in stats, "Statistics missing"
        assert stats["by_service"]["ancestry"]["hits"] > 0, "Hit count not tracked"

        # Test 4: Invalidation
        count = cache.invalidate(key="cache:test:uuid1")
        assert count == 1, "Invalidation failed"
        result = cache.get("ancestry", "combined_details", "cache:test:uuid1")
        assert result is None, "Invalidated entry still returned"

        # Test 5: Key generation
        key1 = generate_cache_key("ancestry", "combined_details", "UUID123")
        assert key1 == "cache:ancestry:combined_details:UUID123", "Simple key generation failed"

        key2 = generate_cache_key("ancestry", "rel_prob", {"uuid1": "A", "uuid2": "B"})
        assert key2.startswith("cache:ancestry:rel_prob:"), "Complex key generation failed"

        logger.info("✅ UnifiedCacheManager tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ UnifiedCacheManager tests failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    success = test_unified_cache_manager()
    sys.exit(0 if success else 1)
