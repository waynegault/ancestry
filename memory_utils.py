#!/usr/bin/env python3

"""
Memory Utilities - Lightweight memory management utilities.

Provides ObjectPool and fast_json_loads for performance optimization.
"""

import json
from collections import deque
from collections.abc import Callable
from typing import Any, Optional


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


def fast_json_loads(json_str: str) -> dict[str, Any] | None:
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


# ==============================================
# Module Tests
# ==============================================


def _test_object_pool() -> None:
    """Test ObjectPool functionality."""
    pool = ObjectPool(lambda: {"value": 0}, max_size=5)

    # Test acquire
    obj1 = pool.acquire()
    assert obj1 is not None, "Should create new object"
    assert "value" in obj1, "Object should have expected structure"

    # Test release and reacquire
    pool.release(obj1)
    obj2 = pool.acquire()
    assert obj2 is obj1, "Should reuse released object"


def _test_fast_json_loads() -> None:
    """Test fast_json_loads functionality."""
    # Test valid JSON
    result = fast_json_loads('{"key": "value"}')
    assert result is not None, "Should parse valid JSON"
    assert result["key"] == "value", "Should parse correctly"

    # Test invalid JSON
    result = fast_json_loads("invalid json")
    assert result is None, "Should return None for invalid JSON"


def memory_utils_module_tests() -> bool:
    """Module-specific tests for memory_utils.py functionality."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Memory Utilities", "memory_utils.py")

    tests = [
        ("ObjectPool functionality", _test_object_pool, "Test object pooling", "Test object pooling"),
        ("fast_json_loads functionality", _test_fast_json_loads, "Test JSON parsing", "Test JSON parsing"),
    ]

    with suppress_logging():
        for test_name, test_func, expected_behavior, test_description in tests:
            suite.run_test(test_name, test_func, expected_behavior, test_description)

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(memory_utils_module_tests)


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys
    print("🔧 Running Memory Utilities comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

