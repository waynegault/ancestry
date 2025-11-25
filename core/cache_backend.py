#!/usr/bin/env python3
"""
core/cache_backend.py - Unified Cache Backend Protocol

Defines the standard interface for all cache implementations in the project.
This protocol enables dependency inversion: consumers depend on the abstract
protocol rather than concrete implementations, breaking import cycles.

Track 4 Step 2: Cache Stack Unification
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Protocol, runtime_checkable

# Add parent directory for imports when running as script
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)


# =============================================================================
# CACHE STATISTICS
# =============================================================================


@dataclass
class CacheStats:
    """Standardized cache statistics across all cache implementations.

    All cache backends should populate these fields for consistent monitoring
    and telemetry across Prometheus/Grafana dashboards.
    """

    name: str
    kind: str  # "disk", "memory", "session", "performance"
    hits: int = 0
    misses: int = 0
    entries: int = 0
    size_bytes: int = 0
    max_size_bytes: int = 0
    evictions: int = 0
    expired: int = 0
    errors: int = 0
    last_access_time: Optional[float] = None
    last_write_time: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as percentage (0-100)."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def utilization(self) -> float:
        """Calculate size utilization as percentage (0-100)."""
        if self.max_size_bytes <= 0:
            return 0.0
        return self.size_bytes / self.max_size_bytes * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization and telemetry."""
        return {
            "name": self.name,
            "kind": self.kind,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 2),
            "entries": self.entries,
            "size_bytes": self.size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "utilization": round(self.utilization, 2),
            "evictions": self.evictions,
            "expired": self.expired,
            "errors": self.errors,
            "last_access_time": self.last_access_time,
            "last_write_time": self.last_write_time,
            **self.extra,
        }


@dataclass
class CacheHealth:
    """Standardized cache health status.

    Used by monitoring systems to determine cache operational status
    and trigger alerts when caches degrade.
    """

    name: str
    status: str  # "healthy", "degraded", "critical", "unknown"
    message: str
    hit_rate: float = 0.0
    is_available: bool = True
    last_error: Optional[str] = None
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "hit_rate": round(self.hit_rate, 2),
            "is_available": self.is_available,
            "last_error": self.last_error,
            "recommendations": self.recommendations,
        }


# =============================================================================
# CACHE BACKEND PROTOCOL
# =============================================================================


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol defining the standard cache interface.

    All cache implementations must conform to this protocol to enable:
    - Unified statistics collection
    - Consistent health monitoring
    - Interchangeable cache backends
    - Dependency inversion (break import cycles)

    Example implementation:
        class MyCache:
            def get(self, key: str) -> Optional[Any]: ...
            def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool: ...
            def delete(self, key: str) -> bool: ...
            def clear(self) -> bool: ...
            def get_stats(self) -> CacheStats: ...
            def get_health(self) -> CacheHealth: ...
            def warm(self) -> bool: ...

    Usage with isinstance checks:
        if isinstance(my_cache, CacheBackend):
            stats = my_cache.get_stats()
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache.

        Args:
            key: The cache key to look up.

        Returns:
            The cached value if found and not expired, None otherwise.
        """
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to store.
            ttl: Time-to-live in seconds. None uses backend default.

        Returns:
            True if stored successfully, False otherwise.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove a key from the cache.

        Args:
            key: The cache key to remove.

        Returns:
            True if key was removed, False if not found or error.
        """
        ...

    @abstractmethod
    def clear(self) -> bool:
        """Clear all entries from the cache.

        Returns:
            True if cleared successfully, False otherwise.
        """
        ...

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get current cache statistics.

        Returns:
            CacheStats dataclass with current metrics.
        """
        ...

    @abstractmethod
    def get_health(self) -> CacheHealth:
        """Get cache health status.

        Returns:
            CacheHealth dataclass with operational status.
        """
        ...

    @abstractmethod
    def warm(self) -> bool:
        """Pre-populate cache with frequently accessed data.

        Returns:
            True if warming completed successfully, False otherwise.
        """
        ...


# =============================================================================
# SCOPED CACHE BACKEND PROTOCOL
# =============================================================================


@runtime_checkable
class ScopedCacheBackend(CacheBackend, Protocol):
    """Extended protocol for caches that support service/endpoint scoping.

    Used by UnifiedCacheManager and similar multi-tenant caches that
    track statistics per service (ancestry, ai) and endpoint.
    """

    @abstractmethod
    def get_scoped(self, service: str, endpoint: str, key: str) -> Optional[Any]:
        """Retrieve a scoped value from the cache.

        Args:
            service: Service name (e.g., "ancestry", "ai").
            endpoint: Endpoint name (e.g., "combined_details").
            key: The cache key within the scope.

        Returns:
            The cached value if found and not expired, None otherwise.
        """
        ...

    @abstractmethod
    def set_scoped(
        self,
        service: str,
        endpoint: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store a scoped value in the cache.

        Args:
            service: Service name (e.g., "ancestry", "ai").
            endpoint: Endpoint name (e.g., "combined_details").
            key: The cache key within the scope.
            value: The value to store.
            ttl: Time-to-live in seconds.

        Returns:
            True if stored successfully, False otherwise.
        """
        ...

    @abstractmethod
    def get_scoped_stats(self, service: str) -> dict[str, Any]:
        """Get statistics for a specific service.

        Args:
            service: Service name to get stats for.

        Returns:
            Dictionary with service-specific statistics.
        """
        ...


# =============================================================================
# CACHE FACTORY
# =============================================================================


class CacheFactory:
    """Factory for creating and managing cache backend instances.

    Provides a centralized location for cache instantiation, enabling
    easy swapping of implementations and consistent configuration.

    Future: Will be the single entry point for all cache creation,
    replacing scattered cache initialization across modules.
    """

    _instances: ClassVar[dict[str, CacheBackend]] = {}

    @classmethod
    def get_backend(cls, name: str) -> Optional[CacheBackend]:
        """Get a registered cache backend by name.

        Args:
            name: The registered name of the cache backend.

        Returns:
            The cache backend instance, or None if not registered.
        """
        return cls._instances.get(name)

    @classmethod
    def register_backend(cls, name: str, backend: CacheBackend) -> None:
        """Register a cache backend instance.

        Args:
            name: The name to register the backend under.
            backend: The cache backend instance.
        """
        cls._instances[name] = backend
        logger.debug(f"Registered cache backend: {name}")

    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backend names.

        Returns:
            Sorted list of registered backend names.
        """
        return sorted(cls._instances.keys())

    @classmethod
    def clear_all(cls) -> dict[str, bool]:
        """Clear all registered cache backends.

        Returns:
            Dictionary mapping backend names to clear success status.
        """
        results: dict[str, bool] = {}
        for name, backend in cls._instances.items():
            try:
                results[name] = backend.clear()
            except Exception as exc:
                logger.warning(f"Failed to clear cache {name}: {exc}")
                results[name] = False
        return results

    @classmethod
    def get_all_stats(cls) -> dict[str, CacheStats]:
        """Get statistics from all registered backends.

        Tries get_stats_typed() first (protocol method), then get_stats().
        This allows for gradual migration of cache implementations.

        Returns:
            Dictionary mapping backend names to their CacheStats.
        """
        stats: dict[str, CacheStats] = {}
        for name, backend in cls._instances.items():
            try:
                # Try the typed method first (new protocol)
                if hasattr(backend, "get_stats_typed"):
                    result = backend.get_stats_typed()
                    if isinstance(result, CacheStats):
                        stats[name] = result
                        continue

                # Fall back to get_stats() if it returns CacheStats
                if hasattr(backend, "get_stats"):
                    result = backend.get_stats()
                    if isinstance(result, CacheStats):
                        stats[name] = result
                    else:
                        # Legacy dict format - create minimal CacheStats
                        stats[name] = CacheStats(
                            name=name,
                            kind=result.get("kind", "unknown") if isinstance(result, dict) else "unknown",
                            hits=result.get("hits", 0) if isinstance(result, dict) else 0,
                            misses=result.get("misses", 0) if isinstance(result, dict) else 0,
                            entries=result.get("entries", 0) if isinstance(result, dict) else 0,
                        )
                else:
                    stats[name] = CacheStats(name=name, kind="unknown")
            except Exception as exc:
                logger.warning(f"Failed to get stats for {name}: {exc}")
                stats[name] = CacheStats(name=name, kind="unknown")
        return stats


# =============================================================================
# TESTS
# =============================================================================


def _test_cache_stats_hit_rate() -> bool:
    """Test CacheStats hit rate calculation."""
    stats = CacheStats(name="test", kind="memory", hits=80, misses=20)
    assert abs(stats.hit_rate - 80.0) < 0.01, f"Expected 80%, got {stats.hit_rate}%"

    # Test zero division case
    empty_stats = CacheStats(name="empty", kind="memory")
    assert empty_stats.hit_rate == 0.0, "Empty cache should have 0% hit rate"

    return True


def _test_cache_stats_utilization() -> bool:
    """Test CacheStats utilization calculation."""
    stats = CacheStats(name="test", kind="disk", size_bytes=500_000_000, max_size_bytes=1_000_000_000)
    assert abs(stats.utilization - 50.0) < 0.01, f"Expected 50%, got {stats.utilization}%"

    # Test zero max size
    zero_max = CacheStats(name="zero", kind="memory", size_bytes=100)
    assert zero_max.utilization == 0.0, "Zero max size should have 0% utilization"

    return True


def _test_cache_stats_to_dict() -> bool:
    """Test CacheStats serialization."""
    stats = CacheStats(
        name="test",
        kind="memory",
        hits=100,
        misses=50,
        entries=75,
        extra={"custom_field": "value"},
    )
    d = stats.to_dict()
    assert d["name"] == "test"
    assert d["kind"] == "memory"
    assert d["hits"] == 100
    assert d["misses"] == 50
    assert d["entries"] == 75
    assert d["custom_field"] == "value"
    assert "hit_rate" in d
    return True


def _test_cache_health_to_dict() -> bool:
    """Test CacheHealth serialization."""
    health = CacheHealth(
        name="test",
        status="healthy",
        message="All systems operational",
        hit_rate=85.5,
        recommendations=["Consider increasing cache size"],
    )
    d = health.to_dict()
    assert d["name"] == "test"
    assert d["status"] == "healthy"
    assert d["is_available"] is True
    assert len(d["recommendations"]) == 1
    return True


def _test_cache_backend_protocol_check() -> bool:
    """Test that protocol can be used for isinstance checks."""

    class MockCache:
        """Mock cache that implements the protocol."""

        def __init__(self) -> None:
            self._data: dict[str, Any] = {}

        def get(self, key: str) -> Optional[Any]:
            return self._data.get(key)

        def set(self, key: str, value: Any, _ttl: Optional[int] = None) -> bool:
            self._data[key] = value
            return True

        def delete(self, key: str) -> bool:
            return self._data.pop(key, None) is not None

        def clear(self) -> bool:
            self._data.clear()
            return True

        def get_stats(self) -> CacheStats:
            return CacheStats(name="mock", kind="memory", entries=len(self._data))

        def get_health(self) -> CacheHealth:
            return CacheHealth(
                name="mock",
                status="healthy",
                message="OK",
                is_available=len(self._data) >= 0,  # Always True, uses self
            )

        def warm(self) -> bool:
            self._data["warmed"] = True
            return True

    mock = MockCache()
    assert isinstance(mock, CacheBackend), "MockCache should implement CacheBackend"
    return True


def _test_cache_factory_operations() -> bool:
    """Test CacheFactory registration and retrieval."""

    class TestCache:
        """Simple test cache for factory testing."""

        def __init__(self) -> None:
            self._data: dict[str, Any] = {}

        def get(self, key: str) -> Optional[Any]:
            return self._data.get(key)

        def set(self, key: str, value: Any, _ttl: Optional[int] = None) -> bool:
            self._data[key] = value
            return True

        def delete(self, key: str) -> bool:
            return self._data.pop(key, None) is not None

        def clear(self) -> bool:
            self._data.clear()
            return True

        def get_stats(self) -> CacheStats:
            return CacheStats(name="factory_test", kind="memory", entries=len(self._data))

        def get_health(self) -> CacheHealth:
            return CacheHealth(
                name="factory_test",
                status="healthy",
                message="OK",
                is_available=len(self._data) >= 0,  # Always True, uses self
            )

        def warm(self) -> bool:
            self._data["warmed"] = True
            return True

    # Clear any existing registrations for clean test
    original_instances = CacheFactory._instances.copy()
    CacheFactory._instances.clear()

    try:
        cache = TestCache()
        CacheFactory.register_backend("test_cache", cache)

        assert "test_cache" in CacheFactory.list_backends()
        retrieved = CacheFactory.get_backend("test_cache")
        assert retrieved is cache

        # Test non-existent backend
        assert CacheFactory.get_backend("nonexistent") is None

        return True
    finally:
        # Restore original instances
        CacheFactory._instances = original_instances


def module_tests() -> bool:
    """Run module tests for cache_backend."""
    from test_framework import TestSuite

    suite = TestSuite("core.cache_backend", "core/cache_backend.py")

    suite.run_test(
        "CacheStats hit rate calculation",
        _test_cache_stats_hit_rate,
        "Validates hit rate percentage calculation with edge cases.",
    )

    suite.run_test(
        "CacheStats utilization calculation",
        _test_cache_stats_utilization,
        "Validates size utilization percentage with edge cases.",
    )

    suite.run_test(
        "CacheStats serialization",
        _test_cache_stats_to_dict,
        "Validates CacheStats.to_dict() includes all fields.",
    )

    suite.run_test(
        "CacheHealth serialization",
        _test_cache_health_to_dict,
        "Validates CacheHealth.to_dict() includes all fields.",
    )

    suite.run_test(
        "CacheBackend protocol check",
        _test_cache_backend_protocol_check,
        "Validates runtime_checkable protocol works with isinstance.",
    )

    suite.run_test(
        "CacheFactory operations",
        _test_cache_factory_operations,
        "Validates backend registration and retrieval.",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    from test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
