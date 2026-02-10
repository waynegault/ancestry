#!/usr/bin/env python3
"""
core/cache/interface.py - Unified Cache Protocol and Types

Defines the standard Cache protocol that all cache implementations must conform to.
Enables dependency inversion and consistent behavior across cache backends.

Features:
- Type-safe Protocol-based interface
- Explicit invalidation patterns (single key, pattern, namespace, all)
- TTL management with default and per-key overrides
- Cache versioning for safe migrations
- Standardized statistics via CacheStats from cache_backend
"""


import logging
import re
import sys
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from core.cache_backend import CacheHealth, CacheStats

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE ALIASES
# =============================================================================

CacheKey = str
CacheTTL = int  # Seconds


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================


@dataclass
class CacheConfig:
    """Configuration for cache instances.

    Attributes:
        name: Human-readable cache name for logging/stats
        default_ttl: Default time-to-live in seconds (0 = no expiry)
        max_entries: Maximum number of entries (0 = unlimited)
        max_size_bytes: Maximum cache size in bytes (0 = unlimited)
        version: Cache version for invalidation on schema changes
        namespace: Optional namespace prefix for all keys
        enable_stats: Whether to track hit/miss statistics
    """

    name: str = "cache"
    default_ttl: CacheTTL = 300  # 5 minutes
    max_entries: int = 10000
    max_size_bytes: int = 0  # Unlimited by default
    version: str = "v1"
    namespace: str = ""
    enable_stats: bool = True


# =============================================================================
# CACHE ENTRY
# =============================================================================


@dataclass
class CacheEntry:
    """Individual cache entry with metadata.

    Attributes:
        key: The cache key (without namespace prefix)
        value: The cached value
        created_at: Unix timestamp when entry was created
        ttl: Time-to-live in seconds (0 = no expiry)
        version: Cache version at time of creation
        hit_count: Number of times this entry was accessed
        size_bytes: Approximate size of the cached value
    """

    key: CacheKey
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: CacheTTL = 0
    version: str = "v1"
    hit_count: int = 0
    size_bytes: int = 0

    @property
    def expires_at(self) -> float | None:
        """Get expiration timestamp, or None if no expiry."""
        if self.ttl <= 0:
            return None
        return self.created_at + self.ttl

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        expires = self.expires_at
        if expires is None:
            return False
        return time.time() > expires

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at

    @property
    def ttl_remaining(self) -> float | None:
        """Get remaining TTL in seconds, or None if no expiry."""
        expires = self.expires_at
        if expires is None:
            return None
        remaining = expires - time.time()
        return max(0.0, remaining)


# =============================================================================
# INVALIDATION PATTERN
# =============================================================================


@dataclass
class InvalidationPattern:
    """Defines a pattern for cache invalidation.

    Supports multiple invalidation strategies:
    - exact: Invalidate a specific key
    - prefix: Invalidate all keys starting with prefix
    - glob: Invalidate keys matching glob pattern (e.g., "user:*:profile")
    - regex: Invalidate keys matching regex pattern
    - namespace: Invalidate all keys in a namespace
    - all: Invalidate entire cache

    Attributes:
        pattern: The pattern string
        pattern_type: Type of pattern matching
        namespace: Optional namespace to scope the pattern
    """

    pattern: str
    pattern_type: str = "exact"  # exact, prefix, glob, regex, namespace, all
    namespace: str = ""

    def matches(self, key: str) -> bool:
        """Check if a key matches this invalidation pattern.

        Args:
            key: The cache key to check (without namespace prefix)

        Returns:
            True if the key matches the pattern
        """
        if self.pattern_type in {"all", "namespace"}:
            # namespace matching handled at cache level
            return True

        if self.pattern_type == "exact":
            return key == self.pattern

        if self.pattern_type == "prefix":
            return key.startswith(self.pattern)

        if self.pattern_type == "glob":
            return fnmatch(key, self.pattern)

        if self.pattern_type == "regex":
            try:
                return bool(re.match(self.pattern, key))
            except re.error:
                logger.warning(f"Invalid regex pattern: {self.pattern}")

        return False


# =============================================================================
# CACHE PROTOCOL
# =============================================================================


@runtime_checkable
class Cache(Protocol):
    """Protocol defining the unified cache interface.

    All cache implementations must conform to this protocol to enable:
    - Consistent API across memory, disk, and distributed caches
    - Type-safe dependency injection
    - Unified statistics and health monitoring
    - Explicit invalidation patterns

    Example implementation:
        class MyCache:
            def get(self, key: str) -> Optional[Any]: ...
            def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool: ...
            def delete(self, key: str) -> bool: ...
            def exists(self, key: str) -> bool: ...
            def invalidate(self, key: str) -> bool: ...
            def invalidate_pattern(self, pattern: str, pattern_type: str = "glob") -> int: ...
            def invalidate_all(self) -> bool: ...
            def get_stats(self) -> CacheStats: ...
            def get_health(self) -> CacheHealth: ...
    """

    @property
    @abstractmethod
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        ...

    @abstractmethod
    def get(self, key: CacheKey) -> Any | None:
        """Retrieve a value from the cache.

        Args:
            key: The cache key to look up.

        Returns:
            The cached value if found and not expired, None otherwise.
        """
        ...

    @abstractmethod
    def set(self, key: CacheKey, value: Any, ttl: CacheTTL | None = None) -> bool:
        """Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to store.
            ttl: Time-to-live in seconds. None uses config default.

        Returns:
            True if stored successfully, False otherwise.
        """
        ...

    @abstractmethod
    def delete(self, key: CacheKey) -> bool:
        """Remove a key from the cache.

        Args:
            key: The cache key to remove.

        Returns:
            True if key was removed, False if not found or error.
        """
        ...

    @abstractmethod
    def exists(self, key: CacheKey) -> bool:
        """Check if a key exists and is not expired.

        Args:
            key: The cache key to check.

        Returns:
            True if key exists and is valid, False otherwise.
        """
        ...

    @abstractmethod
    def invalidate(self, key: CacheKey) -> bool:
        """Invalidate a specific cache key.

        Alias for delete() but semantically indicates intentional invalidation.

        Args:
            key: The cache key to invalidate.

        Returns:
            True if key was invalidated, False otherwise.
        """
        ...

    @abstractmethod
    def invalidate_pattern(self, pattern: str, pattern_type: str = "glob") -> int:
        """Invalidate all keys matching a pattern.

        Args:
            pattern: The pattern to match against keys.
            pattern_type: Type of pattern - "glob", "prefix", "regex".

        Returns:
            Number of keys invalidated.
        """
        ...

    @abstractmethod
    def invalidate_all(self) -> bool:
        """Invalidate all entries in the cache.

        Returns:
            True if cleared successfully, False otherwise.
        """
        ...

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get current cache statistics.

        Returns:
            CacheStats with hits, misses, entries, size, etc.
        """
        ...

    @abstractmethod
    def get_health(self) -> CacheHealth:
        """Get cache health status.

        Returns:
            CacheHealth with status, availability, recommendations.
        """
        ...

    @abstractmethod
    def keys(self, pattern: str = "*") -> list[CacheKey]:
        """Get all keys matching a pattern.

        Args:
            pattern: Glob pattern to match keys (default: all keys).

        Returns:
            List of matching keys.
        """
        ...

    @abstractmethod
    def get_entry(self, key: CacheKey) -> CacheEntry | None:
        """Get cache entry with metadata.

        Args:
            key: The cache key to look up.

        Returns:
            CacheEntry with value and metadata, or None if not found.
        """
        ...


# =============================================================================
# TESTS
# =============================================================================


def _test_cache_config_defaults() -> None:
    """Test CacheConfig default values."""
    config = CacheConfig()
    assert config.name == "cache"
    assert config.default_ttl == 300
    assert config.max_entries == 10000
    assert config.version == "v1"
    assert config.enable_stats is True


def _test_cache_config_custom() -> None:
    """Test CacheConfig with custom values."""
    config = CacheConfig(
        name="test_cache",
        default_ttl=600,
        max_entries=5000,
        version="v2",
        namespace="test:",
    )
    assert config.name == "test_cache"
    assert config.default_ttl == 600
    assert config.namespace == "test:"


def _test_cache_entry_expiration() -> None:
    """Test CacheEntry expiration logic."""
    # Entry with TTL
    entry = CacheEntry(key="test", value="data", ttl=1)
    assert entry.is_expired is False
    assert entry.expires_at is not None
    assert entry.ttl_remaining is not None
    assert entry.ttl_remaining > 0

    # Entry without TTL (no expiry)
    entry_no_ttl = CacheEntry(key="test", value="data", ttl=0)
    assert entry_no_ttl.is_expired is False
    assert entry_no_ttl.expires_at is None
    assert entry_no_ttl.ttl_remaining is None


def _test_cache_entry_age() -> None:
    """Test CacheEntry age calculation."""
    entry = CacheEntry(key="test", value="data", created_at=time.time() - 10)
    assert entry.age_seconds >= 10


def _test_invalidation_pattern_exact() -> None:
    """Test exact pattern matching."""
    pattern = InvalidationPattern(pattern="user:123", pattern_type="exact")
    assert pattern.matches("user:123") is True
    assert pattern.matches("user:456") is False
    assert pattern.matches("user:123:profile") is False


def _test_invalidation_pattern_prefix() -> None:
    """Test prefix pattern matching."""
    pattern = InvalidationPattern(pattern="user:", pattern_type="prefix")
    assert pattern.matches("user:123") is True
    assert pattern.matches("user:456:profile") is True
    assert pattern.matches("session:abc") is False


def _test_invalidation_pattern_glob() -> None:
    """Test glob pattern matching."""
    pattern = InvalidationPattern(pattern="user:*:profile", pattern_type="glob")
    assert pattern.matches("user:123:profile") is True
    assert pattern.matches("user:456:profile") is True
    assert pattern.matches("user:123:settings") is False


def _test_invalidation_pattern_regex() -> None:
    """Test regex pattern matching."""
    pattern = InvalidationPattern(pattern=r"user:\d+$", pattern_type="regex")
    assert pattern.matches("user:123") is True
    assert pattern.matches("user:456") is True
    assert pattern.matches("user:abc") is False


def _test_invalidation_pattern_all() -> None:
    """Test 'all' pattern matching."""
    pattern = InvalidationPattern(pattern="", pattern_type="all")
    assert pattern.matches("anything") is True
    assert pattern.matches("user:123") is True


def _test_cache_protocol_structural() -> None:
    """Test that Cache protocol is runtime checkable."""
    # Cache should be a runtime_checkable Protocol
    assert hasattr(Cache, "__protocol_attrs__") or hasattr(Cache, "_is_protocol")


def module_tests() -> bool:
    """Run module tests for cache interface."""
    from testing.test_framework import TestSuite

    suite = TestSuite("core.cache.interface", "core/cache/interface.py")

    suite.run_test(
        "CacheConfig default values",
        _test_cache_config_defaults,
        "Validates default configuration values",
    )

    suite.run_test(
        "CacheConfig custom values",
        _test_cache_config_custom,
        "Validates custom configuration override",
    )

    suite.run_test(
        "CacheEntry expiration logic",
        _test_cache_entry_expiration,
        "Validates TTL-based expiration calculation",
    )

    suite.run_test(
        "CacheEntry age calculation",
        _test_cache_entry_age,
        "Validates age_seconds property",
    )

    suite.run_test(
        "InvalidationPattern exact match",
        _test_invalidation_pattern_exact,
        "Validates exact key matching",
    )

    suite.run_test(
        "InvalidationPattern prefix match",
        _test_invalidation_pattern_prefix,
        "Validates prefix-based matching",
    )

    suite.run_test(
        "InvalidationPattern glob match",
        _test_invalidation_pattern_glob,
        "Validates glob pattern matching",
    )

    suite.run_test(
        "InvalidationPattern regex match",
        _test_invalidation_pattern_regex,
        "Validates regex pattern matching",
    )

    suite.run_test(
        "InvalidationPattern all match",
        _test_invalidation_pattern_all,
        "Validates 'all' pattern matches everything",
    )

    suite.run_test(
        "Cache protocol is structural",
        _test_cache_protocol_structural,
        "Validates Protocol can be used for isinstance checks",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    from testing.test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
