#!/usr/bin/env python3

"""
Memory Utilities - Lightweight memory management utilities.

Provides ObjectPool and fast_json_loads for performance optimization.
"""

import json
from typing import Any, Callable, Optional
from collections import deque


class ObjectPool:
    """
    Simple object pool for reusing objects to reduce memory allocation overhead.

    Args:
        factory: Callable that creates new objects
        max_size: Maximum number of objects to keep in the pool
    """

    def __init__(self, factory: Callable[[], Any], max_size: int = 10):
        """Initialize the object pool."""
        self.factory = factory
        self.max_size = max_size
        self._pool: deque[Any] = deque(maxlen=max_size)
    
    def acquire(self) -> Any:
        """Get an object from the pool or create a new one."""
        if self._pool:
            return self._pool.popleft()
        return self.factory()
    
    def release(self, obj: Any) -> None:
        """Return an object to the pool."""
        if len(self._pool) < self.max_size:
            self._pool.append(obj)


def fast_json_loads(json_str: str) -> Optional[dict[str, Any]]:
    """
    Fast JSON loading with error handling.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# Test functions
def run_comprehensive_tests() -> bool:
    """Run comprehensive tests for memory utilities."""
    print("\n" + "=" * 70)
    print("MEMORY UTILITIES - COMPREHENSIVE TESTS")
    print("=" * 70)
    
    all_passed = True
    
    # Test ObjectPool
    print("\n[1/2] Testing ObjectPool...")
    try:
        pool = ObjectPool(lambda: {"value": 0}, max_size=5)
        
        # Test acquire
        obj1 = pool.acquire()
        assert obj1 is not None, "Should create new object"
        assert "value" in obj1, "Object should have expected structure"
        
        # Test release and reacquire
        pool.release(obj1)
        obj2 = pool.acquire()
        assert obj2 is obj1, "Should reuse released object"
        
        print("   ✅ ObjectPool tests passed")
    except AssertionError as e:
        print(f"   ❌ ObjectPool test failed: {e}")
        all_passed = False
    
    # Test fast_json_loads
    print("\n[2/2] Testing fast_json_loads...")
    try:
        # Test valid JSON
        result = fast_json_loads('{"key": "value"}')
        assert result is not None, "Should parse valid JSON"
        assert result["key"] == "value", "Should parse correctly"
        
        # Test invalid JSON
        result = fast_json_loads("invalid json")
        assert result is None, "Should return None for invalid JSON"
        
        print("   ✅ fast_json_loads tests passed")
    except AssertionError as e:
        print(f"   ❌ fast_json_loads test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)

