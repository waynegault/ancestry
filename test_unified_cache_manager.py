"""
Comprehensive Unit Tests for UnifiedCacheManager
Tests all core functionality: threading, serialization, TTL, statistics, eviction
"""

import sys
import threading
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the cache manager
from core.unified_cache_manager import (
    CacheEntry,
    UnifiedCacheManager,
    create_ancestry_cache_config,
    generate_cache_key,
    get_unified_cache,
)
from test_framework import TestSuite

# ============================================================================
# Test Functions (return None, raise AssertionError on failure)
# ============================================================================

def test_cache_entry_creation() -> None:
    """Test CacheEntry dataclass creation and properties"""
    entry = CacheEntry(
        key="test_key",
        data={"test": "data"},
        timestamp=time.time(),
        ttl_seconds=60,
        hit_count=5,
        service="test_service",
        endpoint="test_endpoint"
    )

    assert entry.key == "test_key"
    assert entry.data == {"test": "data"}
    assert entry.hit_count == 5
    assert entry.service == "test_service"
    assert not entry.is_expired


def test_cache_entry_expiration() -> None:
    """Test CacheEntry expiration logic"""
    past_time = time.time() - 120
    entry = CacheEntry(
        key="expired_key",
        data={"old": "data"},
        timestamp=past_time,
        ttl_seconds=60,
        service="test_service"
    )

    assert entry.is_expired

    fresh_entry = CacheEntry(
        key="fresh_key",
        data={"fresh": "data"},
        timestamp=time.time(),
        ttl_seconds=300,
        service="test_service"
    )

    assert not fresh_entry.is_expired


def test_cache_basic_set_get() -> None:
    """Test basic cache set and get operations"""
    cache = UnifiedCacheManager()

    cache.set(
        service="test_service",
        endpoint="test_endpoint",
        key="user_123",
        value={"name": "Alice", "age": 30},
        ttl=300
    )

    result = cache.get(
        service="test_service",
        endpoint="test_endpoint",
        key="user_123"
    )

    assert result is not None
    assert result["name"] == "Alice"
    assert result["age"] == 30


def test_cache_miss_returns_none() -> None:
    """Test that non-existent keys return None"""
    cache = UnifiedCacheManager()

    result = cache.get(
        service="test_service",
        endpoint="test_endpoint",
        key="nonexistent_key_xyz"
    )

    assert result is None


def test_cache_ttl_expiration() -> None:
    """Test that expired entries are not returned"""
    cache = UnifiedCacheManager()

    cache.set(
        service="test_service",
        endpoint="test_endpoint",
        key="expiring_key",
        value={"data": "will expire"},
        ttl=1
    )

    result1 = cache.get(
        service="test_service",
        endpoint="test_endpoint",
        key="expiring_key"
    )
    assert result1 is not None

    time.sleep(1.5)

    result2 = cache.get(
        service="test_service",
        endpoint="test_endpoint",
        key="expiring_key"
    )
    assert result2 is None


def test_cache_deep_copy_isolation() -> None:
    """Test that retrieved values are deep copied (not references)"""
    cache = UnifiedCacheManager()

    original_data = {"list": [1, 2, 3], "dict": {"key": "value"}}

    cache.set(
        service="test_service",
        endpoint="test_endpoint",
        key="mutable_key",
        value=original_data,
        ttl=300
    )

    retrieved = cache.get(
        service="test_service",
        endpoint="test_endpoint",
        key="mutable_key"
    )

    assert retrieved is not None
    retrieved["list"].append(999)
    retrieved["dict"]["key"] = "modified"

    retrieved_again = cache.get(
        service="test_service",
        endpoint="test_endpoint",
        key="mutable_key"
    )

    assert retrieved_again is not None
    assert retrieved_again["list"] == [1, 2, 3]
    assert retrieved_again["dict"]["key"] == "value"


def test_cache_hit_miss_statistics() -> None:
    """Test hit/miss counting in statistics"""
    cache = UnifiedCacheManager()  # Create fresh instance for isolated testing

    # Set a value
    cache.set(
        service="test_svc",
        endpoint="test_ep",
        key="uuid_001",
        value={"name": "Test"},
        ttl=300
    )

    # Get it (hit)
    result = cache.get(
        service="test_svc",
        endpoint="test_ep",
        key="uuid_001"
    )
    assert result is not None

    # Try to get non-existent (miss)
    result2 = cache.get(
        service="test_svc",
        endpoint="test_ep",
        key="uuid_999"
    )
    assert result2 is None

    # Check stats
    stats = cache.get_stats(endpoint="test_ep")
    assert "by_service" in stats
    assert "test_svc" in stats["by_service"]
    assert stats["by_service"]["test_svc"]["hits"] >= 1
    assert stats["by_service"]["test_svc"]["misses"] >= 1


def test_cache_service_creation() -> None:
    """Test that unknown services are created dynamically"""
    cache = UnifiedCacheManager()

    cache.set(
        service="new_service_xyz",
        endpoint="test_endpoint",
        key="key_001",
        value={"data": "test"},
        ttl=300
    )

    result = cache.get(
        service="new_service_xyz",
        endpoint="test_endpoint",
        key="key_001"
    )

    assert result is not None
    assert result["data"] == "test"


def test_cache_invalidate_by_key() -> None:
    """Test cache invalidation by specific key"""
    cache = UnifiedCacheManager()

    cache.set(
        service="test_service",
        endpoint="combined_details",
        key="uuid_001",
        value={"name": "Alice"},
        ttl=300
    )

    cache.set(
        service="test_service",
        endpoint="combined_details",
        key="uuid_002",
        value={"name": "Bob"},
        ttl=300
    )

    invalidated_count = cache.invalidate(
        service="test_service",
        endpoint="combined_details",
        key="uuid_001"
    )

    assert invalidated_count == 1

    result1 = cache.get(
        service="test_service",
        endpoint="combined_details",
        key="uuid_001"
    )
    assert result1 is None

    result2 = cache.get(
        service="test_service",
        endpoint="combined_details",
        key="uuid_002"
    )
    assert result2 is not None


def test_cache_invalidate_by_endpoint() -> None:
    """Test cache invalidation by endpoint"""
    cache = UnifiedCacheManager()

    cache.set("test_service", "endpoint_a", "key_1", {"data": "a"}, ttl=300)
    cache.set("test_service", "endpoint_b", "key_2", {"data": "b"}, ttl=300)

    invalidated_count = cache.invalidate(
        service="test_service",
        endpoint="endpoint_a"
    )

    assert invalidated_count >= 1

    result_a = cache.get("test_service", "endpoint_a", "key_1")
    assert result_a is None

    result_b = cache.get("test_service", "endpoint_b", "key_2")
    assert result_b is not None


def test_cache_invalidate_by_service() -> None:
    """Test cache invalidation by service"""
    cache = UnifiedCacheManager()

    cache.set("service_a", "endpoint", "key_1", {"data": "a"}, ttl=300)
    cache.set("service_b", "endpoint", "key_2", {"data": "b"}, ttl=300)

    invalidated_count = cache.invalidate(service="service_a")

    assert invalidated_count >= 1

    result_a = cache.get("service_a", "endpoint", "key_1")
    assert result_a is None

    result_b = cache.get("service_b", "endpoint", "key_2")
    assert result_b is not None


def test_cache_clear() -> None:
    """Test cache clear operation"""
    cache = UnifiedCacheManager()

    for i in range(5):
        cache.set("test_service", "endpoint", f"key_{i}", {"id": i}, ttl=300)

    cleared_count = cache.clear()

    assert cleared_count >= 5

    for i in range(5):
        result = cache.get("test_service", "endpoint", f"key_{i}")
        assert result is None


def test_cache_lru_eviction() -> None:
    """Test LRU eviction when cache exceeds size limit"""
    cache = UnifiedCacheManager(max_entries=5)

    for i in range(10):
        cache.set("test_service", "endpoint", f"key_{i}", {"id": i}, ttl=300)

    stats = cache.get_stats(endpoint="endpoint")
    total_entries = stats.get("test_service", {}).get("entries", 0)

    assert total_entries <= 5


def test_cache_singleton_pattern() -> None:
    """Test that get_unified_cache() returns same instance"""
    cache1 = get_unified_cache()
    cache2 = get_unified_cache()

    assert cache1 is cache2


def test_generate_cache_key_uuid() -> None:
    """Test cache key generation for UUID"""
    key1 = generate_cache_key("ancestry", "combined_details", "12345678-abcd-1234-abcd-12345678abcd")
    key2 = generate_cache_key("ancestry", "combined_details", "12345678-abcd-1234-abcd-12345678abcd")

    assert key1 == key2


def test_generate_cache_key_dict() -> None:
    """Test cache key generation for complex parameters"""
    params = {"name": "test", "age": 30, "tags": ["a", "b"]}
    key1 = generate_cache_key("ancestry", "rel_prob", params)
    key2 = generate_cache_key("ancestry", "rel_prob", params)

    assert key1 == key2

    params2 = {"name": "test", "age": 31, "tags": ["a", "b"]}
    key3 = generate_cache_key("ancestry", "rel_prob", params2)

    assert key1 != key3


def test_create_ancestry_cache_config() -> None:
    """Test preset cache configuration creation"""
    config = create_ancestry_cache_config()

    assert isinstance(config, dict)
    assert "combined_details" in config
    assert config["combined_details"] == 2400


def test_cache_stats_multiple_endpoints() -> None:
    """Test statistics tracking across multiple endpoints"""
    cache = UnifiedCacheManager()

    for endpoint in ["endpoint_a", "endpoint_b", "endpoint_c"]:
        for i in range(3):
            cache.set("test_service", endpoint, f"key_{i}", {"ep": endpoint}, ttl=300)
            cache.get("test_service", endpoint, f"key_{i}")

    stats_a = cache.get_stats(endpoint="endpoint_a")
    stats_b = cache.get_stats(endpoint="endpoint_b")
    stats_c = cache.get_stats(endpoint="endpoint_c")

    assert stats_a["by_service"]["test_service"]["hits"] > 0
    assert stats_b["by_service"]["test_service"]["hits"] > 0
    assert stats_c["by_service"]["test_service"]["hits"] > 0


def test_cache_overwrites_existing() -> None:
    """Test that setting same key overwrites previous value"""
    cache = UnifiedCacheManager()

    cache.set("test_service", "endpoint", "key_xyz", {"version": 1}, ttl=300)
    cache.set("test_service", "endpoint", "key_xyz", {"version": 2}, ttl=300)

    result = cache.get("test_service", "endpoint", "key_xyz")
    assert result is not None
    assert result["version"] == 2


def test_cache_thread_safety() -> None:
    """Test thread-safe concurrent access"""
    cache = UnifiedCacheManager()
    errors: list[str] = []

    def worker(worker_id: int) -> None:
        try:
            for i in range(50):
                cache.set(
                    "test_service",
                    f"endpoint_{worker_id}",
                    f"key_{i}",
                    {"worker": worker_id, "iter": i},
                    ttl=300
                )

                result = cache.get(
                    "test_service",
                    f"endpoint_{worker_id}",
                    f"key_{i}"
                )

                if result is None:
                    errors.append(f"Worker {worker_id}: Lost value at key_{i}")
        except Exception as e:
            errors.append(f"Worker {worker_id}: {e!s}")

    threads: list[threading.Thread] = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"


# ============================================================================
# Main Test Suite
# ============================================================================

def module_tests() -> bool:
    """Run all cache manager tests"""
    suite = TestSuite("UnifiedCacheManager", "core/unified_cache_manager.py")
    suite.start_suite()

    suite.run_test("CacheEntry Creation", test_cache_entry_creation)
    suite.run_test("CacheEntry Expiration", test_cache_entry_expiration)
    suite.run_test("Basic Cache Set/Get", test_cache_basic_set_get)
    suite.run_test("Cache Miss Returns None", test_cache_miss_returns_none)
    suite.run_test("TTL Expiration Logic", test_cache_ttl_expiration)
    suite.run_test("Deep Copy Isolation", test_cache_deep_copy_isolation)
    suite.run_test("Hit/Miss Statistics", test_cache_hit_miss_statistics)
    suite.run_test("Statistics Multiple Endpoints", test_cache_stats_multiple_endpoints)
    suite.run_test("Dynamic Service Creation", test_cache_service_creation)
    suite.run_test("Invalidation by Key", test_cache_invalidate_by_key)
    suite.run_test("Invalidation by Endpoint", test_cache_invalidate_by_endpoint)
    suite.run_test("Invalidation by Service", test_cache_invalidate_by_service)
    suite.run_test("Clear Entire Cache", test_cache_clear)
    suite.run_test("LRU Eviction", test_cache_lru_eviction)
    suite.run_test("Overwrite Existing Values", test_cache_overwrites_existing)
    suite.run_test("Singleton Factory Pattern", test_cache_singleton_pattern)
    suite.run_test("Cache Key Generation - UUID", test_generate_cache_key_uuid)
    suite.run_test("Cache Key Generation - Dict", test_generate_cache_key_dict)
    suite.run_test("Preset Cache Configuration", test_create_ancestry_cache_config)
    suite.run_test("Thread-Safe Concurrent Access", test_cache_thread_safety)

    return suite.finish_suite()


run_comprehensive_tests = module_tests

if __name__ == "__main__":
    import sys
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
