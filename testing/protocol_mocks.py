#!/usr/bin/env python3

"""
Protocol-Based Mock Implementations for Testing.

Phase 13.3: Provides mock implementations that satisfy core/protocols.py
interfaces for use in unit tests without concrete dependencies.

Usage:
    from testing.protocol_mocks import (
        MockRateLimiter,
        MockDatabaseSession,
        MockSessionManager,
        MockCache,
        MockLogger,
    )

    def test_api_call():
        limiter = MockRateLimiter()
        result = make_api_call(limiter, "https://example.com")
        assert limiter.wait_called
        assert limiter.wait_count == 1
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.protocols import (
    CacheProtocol,
    DatabaseSessionProtocol,
    LoggerProtocol,
    RateLimiterProtocol,
    SessionManagerProtocol,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Mock Implementations
# =============================================================================


class MockRateLimiter:
    """
    Mock implementation of RateLimiterProtocol for testing.

    Tracks all method calls for test assertions.
    """

    def __init__(self) -> None:
        self.wait_count = 0
        self.success_count = 0
        self.error_429_count = 0
        self.last_endpoint: str | None = None
        self.last_retry_after: float | None = None
        self._wait_time = 0.0

    def wait(self, endpoint: str | None = None) -> float:
        """Record wait call, return configured wait time."""
        self.wait_count += 1
        self.last_endpoint = endpoint
        return self._wait_time

    def on_success(self, endpoint: str | None = None) -> None:
        """Record success for endpoint."""
        self.success_count += 1
        self.last_endpoint = endpoint

    def on_429_error(self, endpoint: str | None = None, retry_after: float | None = None) -> None:
        """Record 429 error."""
        self.error_429_count += 1
        self.last_endpoint = endpoint
        self.last_retry_after = retry_after

    def configure_wait_time(self, wait_time: float) -> None:
        """Configure the wait time to return."""
        self._wait_time = wait_time

    def reset(self) -> None:
        """Reset all counters for fresh test."""
        self.wait_count = 0
        self.success_count = 0
        self.error_429_count = 0
        self.last_endpoint = None
        self.last_retry_after = None


class MockDatabaseSession:
    """
    Mock implementation of DatabaseSessionProtocol for testing.

    Tracks all operations and allows configuring query results.
    """

    def __init__(self) -> None:
        self.added_instances: list[Any] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0
        self.query_count = 0
        self._query_results: list[Any] = []
        self._should_raise_on_commit = False
        self._commit_exception: Exception | None = None

    def add(self, instance: Any) -> None:
        """Track added instance."""
        self.added_instances.append(instance)

    def commit(self) -> None:
        """Record commit, optionally raise exception."""
        self.commit_count += 1
        if self._should_raise_on_commit and self._commit_exception:
            raise self._commit_exception

    def rollback(self) -> None:
        """Record rollback."""
        self.rollback_count += 1

    def query(self, *_args: Any) -> "MockQuery":
        """Return a mock query object."""
        self.query_count += 1
        return MockQuery(self._query_results)

    def close(self) -> None:
        """Record close."""
        self.close_count += 1

    def configure_query_results(self, results: list[Any]) -> None:
        """Configure results to return from queries."""
        self._query_results = results

    def configure_commit_failure(self, exception: Exception) -> None:
        """Configure commit to raise an exception."""
        self._should_raise_on_commit = True
        self._commit_exception = exception

    def reset(self) -> None:
        """Reset all state."""
        self.added_instances.clear()
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0
        self.query_count = 0
        self._query_results.clear()
        self._should_raise_on_commit = False
        self._commit_exception = None


class MockQuery:
    """Mock query object for chained query operations."""

    def __init__(self, results: list[Any]) -> None:
        self._results = results
        self._filters: list[Any] = []

    def filter(self, *args: Any) -> "MockQuery":
        """Record filter and return self for chaining."""
        self._filters.extend(args)
        return self

    def filter_by(self, **kwargs: Any) -> "MockQuery":
        """Record filter_by and return self for chaining."""
        self._filters.append(kwargs)
        return self

    def first(self) -> Any | None:
        """Return first result or None."""
        return self._results[0] if self._results else None

    def all(self) -> list[Any]:
        """Return all results."""
        return list(self._results)

    def count(self) -> int:
        """Return count of results."""
        return len(self._results)

    def order_by(self, *_args: Any) -> "MockQuery":
        """Return self for chaining."""
        return self

    def limit(self, limit: int) -> "MockQuery":
        """Apply limit and return self."""
        self._results = self._results[:limit]
        return self


class MockSessionManager:
    """
    Mock implementation of SessionManagerProtocol for testing.

    Provides configurable session validity and readiness.
    """

    def __init__(self) -> None:
        self._is_valid = True
        self._session_ready = True
        self._db_ready = True
        self._session_age = 0.0
        self.driver: Any = None
        self.db_manager: Any | None = None

    def is_sess_valid(self) -> bool:
        """Return configured validity."""
        return self._is_valid

    def ensure_session_ready(self) -> bool:
        """Return configured session readiness."""
        return self._session_ready

    def ensure_db_ready(self) -> bool:
        """Return configured DB readiness."""
        return self._db_ready

    def session_age_seconds(self) -> float:
        """Return configured session age."""
        return self._session_age

    def configure_valid(self, is_valid: bool) -> None:
        """Configure session validity."""
        self._is_valid = is_valid

    def configure_session_ready(self, ready: bool) -> None:
        """Configure session readiness."""
        self._session_ready = ready

    def configure_db_ready(self, ready: bool) -> None:
        """Configure DB readiness."""
        self._db_ready = ready

    def configure_session_age(self, age: float) -> None:
        """Configure session age in seconds."""
        self._session_age = age


class MockCache:
    """
    Mock implementation of CacheProtocol for testing.

    Uses in-memory dict with optional TTL tracking.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._ttls: dict[str, int | None] = {}
        self.get_count = 0
        self.set_count = 0
        self.delete_count = 0
        self.clear_count = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        self.get_count += 1
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        self.set_count += 1
        self._cache[key] = value
        self._ttls[key] = ttl

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        self.delete_count += 1
        if key in self._cache:
            del self._cache[key]
            self._ttls.pop(key, None)
            return True
        return False

    def clear(self) -> None:
        """Clear all cached values."""
        self.clear_count += 1
        self._cache.clear()
        self._ttls.clear()

    def reset(self) -> None:
        """Reset all state including counters."""
        self._cache.clear()
        self._ttls.clear()
        self.get_count = 0
        self.set_count = 0
        self.delete_count = 0
        self.clear_count = 0


class MockLogger:
    """
    Mock implementation of LoggerProtocol for testing.

    Captures all log messages for assertion.
    """

    def __init__(self) -> None:
        self.debug_messages: list[str] = []
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.error_messages: list[str] = []

    def debug(self, msg: str, *args: Any, **_kwargs: Any) -> None:
        """Capture debug message."""
        self.debug_messages.append(msg % args if args else msg)

    def info(self, msg: str, *args: Any, **_kwargs: Any) -> None:
        """Capture info message."""
        self.info_messages.append(msg % args if args else msg)

    def warning(self, msg: str, *args: Any, **_kwargs: Any) -> None:
        """Capture warning message."""
        self.warning_messages.append(msg % args if args else msg)

    def error(self, msg: str, *args: Any, **_kwargs: Any) -> None:
        """Capture error message."""
        self.error_messages.append(msg % args if args else msg)

    def all_messages(self) -> list[str]:
        """Return all messages in order."""
        return self.debug_messages + self.info_messages + self.warning_messages + self.error_messages

    def reset(self) -> None:
        """Clear all captured messages."""
        self.debug_messages.clear()
        self.info_messages.clear()
        self.warning_messages.clear()
        self.error_messages.clear()


# =============================================================================
# Protocol Verification
# =============================================================================


def verify_protocol_compliance() -> dict[str, bool]:
    """
    Verify that all mock implementations satisfy their protocols.

    Returns:
        Dict mapping mock class name to compliance status.
    """
    results: dict[str, bool] = {}

    # Check RateLimiterProtocol
    results["MockRateLimiter"] = isinstance(MockRateLimiter(), RateLimiterProtocol)

    # Check DatabaseSessionProtocol
    results["MockDatabaseSession"] = isinstance(MockDatabaseSession(), DatabaseSessionProtocol)

    # Check SessionManagerProtocol
    results["MockSessionManager"] = isinstance(MockSessionManager(), SessionManagerProtocol)

    # Check CacheProtocol
    results["MockCache"] = isinstance(MockCache(), CacheProtocol)

    # Check LoggerProtocol
    results["MockLogger"] = isinstance(MockLogger(), LoggerProtocol)

    return results


# =============================================================================
# Test Functions
# =============================================================================


def _test_mock_rate_limiter() -> bool:
    """Test MockRateLimiter functionality."""
    limiter = MockRateLimiter()

    # Test wait tracking
    wait_time = limiter.wait("test_endpoint")
    assert limiter.wait_count == 1, "Should track wait count"
    assert limiter.last_endpoint == "test_endpoint", "Should track endpoint"
    assert wait_time == 0.0, "Default wait time should be 0"

    # Test configurable wait time
    limiter.configure_wait_time(0.5)
    wait_time = limiter.wait()
    assert wait_time == 0.5, "Should return configured wait time"

    # Test success tracking
    limiter.on_success()
    assert limiter.success_count == 1, "Should track success count"

    # Test error tracking
    limiter.on_429_error("api", 30.0)
    assert limiter.error_429_count == 1, "Should track 429 count"
    assert limiter.last_retry_after == 30.0, "Should track retry_after"

    # Test reset
    limiter.reset()
    assert limiter.wait_count == 0, "Reset should clear wait count"

    logger.info("✓ MockRateLimiter tests passed")
    return True


def _test_mock_database_session() -> bool:
    """Test MockDatabaseSession functionality."""
    session = MockDatabaseSession()

    # Test add tracking
    session.add({"id": 1})
    assert len(session.added_instances) == 1, "Should track added instances"

    # Test commit/rollback tracking
    session.commit()
    assert session.commit_count == 1, "Should track commits"

    session.rollback()
    assert session.rollback_count == 1, "Should track rollbacks"

    # Test query results
    session.configure_query_results([{"name": "Test"}])
    result = session.query("TestModel").first()
    assert result == {"name": "Test"}, "Should return configured results"

    # Test commit failure
    session.configure_commit_failure(RuntimeError("DB error"))
    try:
        session.commit()
        raise AssertionError("Should raise exception")
    except RuntimeError:
        pass  # Expected

    logger.info("✓ MockDatabaseSession tests passed")
    return True


def _test_mock_session_manager() -> bool:
    """Test MockSessionManager functionality."""
    sm = MockSessionManager()

    # Test defaults
    assert sm.is_sess_valid() is True, "Default should be valid"
    assert sm.ensure_session_ready() is True, "Default should be ready"
    assert sm.ensure_db_ready() is True, "Default DB should be ready"
    assert sm.session_age_seconds() == 0.0, "Default age should be 0"

    # Test configuration
    sm.configure_valid(False)
    assert sm.is_sess_valid() is False, "Should reflect configured validity"

    sm.configure_session_age(300.0)
    assert sm.session_age_seconds() == 300.0, "Should reflect configured age"

    logger.info("✓ MockSessionManager tests passed")
    return True


def _test_mock_cache() -> bool:
    """Test MockCache functionality."""
    cache = MockCache()

    # Test set/get
    cache.set("key1", "value1", ttl=60)
    assert cache.get("key1") == "value1", "Should retrieve set value"
    assert cache.set_count == 1, "Should track set count"
    assert cache.get_count == 1, "Should track get count"

    # Test delete
    assert cache.delete("key1") is True, "Should return True for existing key"
    assert cache.delete("key1") is False, "Should return False for missing key"

    # Test clear
    cache.set("key2", "value2")
    cache.clear()
    assert cache.get("key2") is None, "Clear should remove all entries"
    assert cache.clear_count == 1, "Should track clear count"

    logger.info("✓ MockCache tests passed")
    return True


def _test_mock_logger() -> bool:
    """Test MockLogger functionality."""
    mock_logger = MockLogger()

    mock_logger.debug("Debug message")
    mock_logger.info("Info message")
    mock_logger.warning("Warning message")
    mock_logger.error("Error message")

    assert len(mock_logger.debug_messages) == 1, "Should capture debug"
    assert len(mock_logger.info_messages) == 1, "Should capture info"
    assert len(mock_logger.warning_messages) == 1, "Should capture warning"
    assert len(mock_logger.error_messages) == 1, "Should capture error"

    assert len(mock_logger.all_messages()) == 4, "Should have 4 total messages"

    mock_logger.reset()
    assert len(mock_logger.all_messages()) == 0, "Reset should clear all"

    logger.info("✓ MockLogger tests passed")
    return True


def _test_protocol_compliance() -> bool:
    """Test that all mocks satisfy their protocols."""
    results = verify_protocol_compliance()

    for mock_name, complies in results.items():
        assert complies, f"{mock_name} should satisfy its protocol"

    logger.info(f"✓ All {len(results)} mocks satisfy their protocols")
    return True


def protocol_mocks_module_tests() -> bool:
    """Run all protocol mocks tests."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("Protocol-Based Mock Implementations", "protocol_mocks.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "MockRateLimiter",
            _test_mock_rate_limiter,
            "MockRateLimiter satisfies RateLimiterProtocol",
            "Test MockRateLimiter implementation",
            "Verify wait, success, error tracking",
        )

        suite.run_test(
            "MockDatabaseSession",
            _test_mock_database_session,
            "MockDatabaseSession satisfies DatabaseSessionProtocol",
            "Test MockDatabaseSession implementation",
            "Verify add, commit, rollback, query tracking",
        )

        suite.run_test(
            "MockSessionManager",
            _test_mock_session_manager,
            "MockSessionManager satisfies SessionManagerProtocol",
            "Test MockSessionManager implementation",
            "Verify validity and readiness configuration",
        )

        suite.run_test(
            "MockCache",
            _test_mock_cache,
            "MockCache satisfies CacheProtocol",
            "Test MockCache implementation",
            "Verify get, set, delete, clear operations",
        )

        suite.run_test(
            "MockLogger",
            _test_mock_logger,
            "MockLogger satisfies LoggerProtocol",
            "Test MockLogger implementation",
            "Verify message capture at all levels",
        )

        suite.run_test(
            "Protocol compliance",
            _test_protocol_compliance,
            "All mocks satisfy their respective protocols",
            "Verify runtime_checkable protocol satisfaction",
            "Use isinstance() with Protocol types",
        )

    return suite.finish_suite()


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(protocol_mocks_module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    import sys

    sys.exit(0 if success else 1)
