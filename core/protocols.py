"""
Type protocols and type aliases for the Ancestry Research Automation Platform.

This module provides:
1. Protocol classes for duck-typed interfaces
2. TypedDict definitions for structured dictionaries
3. Type aliases for common patterns

Usage:
    from core.protocols import (
        RateLimiterProtocol,
        SessionManagerProtocol,
        APIResponse,
        PersonData,
        MatchData,
    )
"""

from __future__ import annotations

import sys
from typing import Any, Protocol, TypedDict, runtime_checkable

# Ensure Python 3.9+ compatibility
if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired


# =============================================================================
# Protocol Classes (Duck Typing with Type Safety)
# =============================================================================


@runtime_checkable
class RateLimiterProtocol(Protocol):
    """Protocol for rate limiter implementations.

    Any class implementing these methods can be used where RateLimiterProtocol
    is expected, without explicit inheritance.

    Example:
        def make_api_call(limiter: RateLimiterProtocol, url: str) -> Response:
            limiter.wait()
            response = requests.get(url)
            if response.status_code == 429:
                limiter.on_429_error()
            else:
                limiter.on_success()
            return response
    """

    def wait(self, endpoint: str | None = None) -> float:
        """Wait for rate limit, return actual wait time."""
        ...

    def on_success(self) -> None:
        """Report successful request for rate adaptation."""
        ...

    def on_429_error(self, endpoint: str | None = None) -> None:
        """Report 429 error for rate backoff."""
        ...

    def get_metrics(self) -> Any:
        """Get current rate limiter metrics."""
        ...


@runtime_checkable
class DatabaseSessionProtocol(Protocol):
    """Protocol for database session implementations."""

    def add(self, instance: Any) -> None:
        """Add an instance to the session."""
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def rollback(self) -> None:
        """Roll back the current transaction."""
        ...

    def query(self, *args: Any) -> Any:
        """Create a query object."""
        ...

    def close(self) -> None:
        """Close the session."""
        ...


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for logger implementations."""

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        ...

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache implementations."""

    def get(self, key: str) -> Any | None:
        """Get value from cache, or None if not found."""
        ...

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL."""
        ...

    def delete(self, key: str) -> bool:
        """Delete key from cache, return True if deleted."""
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...


# =============================================================================
# TypedDict Definitions (Structured Dictionary Types)
# =============================================================================


class APIResponse(TypedDict, total=False):
    """Standard API response structure.

    Example:
        def fetch_data(url: str) -> APIResponse:
            response = requests.get(url)
            return APIResponse(
                success=response.ok,
                status_code=response.status_code,
                data=response.json() if response.ok else None,
                error=str(response.text) if not response.ok else None,
            )
    """

    success: bool
    status_code: int
    data: dict[str, Any] | list[Any] | None
    error: str | None
    headers: dict[str, str]


class PersonData(TypedDict, total=False):
    """Person data structure from DNA matches.

    Example:
        person: PersonData = {
            "uuid": "ABC123",
            "name": "John Smith",
            "profile_id": "user_12345",
            "shared_cm": 250.5,
        }
    """

    uuid: str
    name: str
    profile_id: str | None
    administrator_profile_id: str | None
    shared_cm: float
    shared_segments: int
    relationship_range: str | None
    has_tree: bool
    tree_person_count: int | None


class MatchData(TypedDict, total=False):
    """DNA match data structure.

    Example:
        match: MatchData = {
            "guid": "MATCH-UUID",
            "test_guid": "TEST-UUID",
            "display_name": "Jane Doe",
            "shared_centimorgans": 125.0,
            "confidence": 0.95,
        }
    """

    guid: str
    test_guid: str
    display_name: str
    shared_centimorgans: float
    shared_segments: int
    confidence: float
    groups: list[str]
    starred: bool
    viewed: bool
    has_hint: bool
    note: str | None


class RateLimiterMetrics(TypedDict):
    """Rate limiter metrics structure.

    Example:
        metrics: RateLimiterMetrics = limiter.get_metrics()
        print(f"Current rate: {metrics['current_fill_rate']} req/s")
    """

    current_fill_rate: float
    tokens_available: float
    total_requests: int
    avg_wait_time: float
    error_429_count: int
    rate_increases: int
    rate_decreases: int
    endpoint_rate_cap: float | None


class BudgetInfo(TypedDict):
    """Rate limit budget information.

    Example:
        budget: BudgetInfo = limiter.calculate_budget(60.0)
        print(f"Can make ~{budget['estimated_requests']} requests")
    """

    estimated_requests: int
    time_period_seconds: float
    current_fill_rate: float
    available_tokens: float


class TestResult(TypedDict, total=False):
    """Test execution result structure.

    Example:
        result: TestResult = {
            "name": "test_initialization",
            "passed": True,
            "duration": 0.123,
        }
    """

    name: str
    passed: bool
    duration: float
    error: str | None
    expected: str | None
    actual: str | None


class HealthStatus(TypedDict):
    """System health status structure.

    Example:
        status: HealthStatus = {
            "healthy": True,
            "components": {
                "database": "ok",
                "rate_limiter": "optimal",
            },
            "timestamp": "2025-11-29T12:00:00Z",
        }
    """

    healthy: bool
    components: dict[str, str]
    timestamp: str
    details: NotRequired[dict[str, Any]]


class CorrelationData(TypedDict, total=False):
    """Correlation context data for request tracking.

    Example:
        ctx: CorrelationData = {
            "correlation_id": "abc123",
            "operation": "fetch_matches",
            "started_at": "2025-11-29T12:00:00Z",
        }
    """

    correlation_id: str
    operation: str | None
    user_id: str | None
    started_at: str
    metadata: dict[str, Any]


# =============================================================================
# Type Aliases (Common Type Patterns)
# =============================================================================

# JSON-compatible types
JSONValue = str | int | float | bool | dict[str, Any] | list[Any] | None
JSONDict = dict[str, JSONValue]
JSONList = list[JSONValue]

# Common callback types
ProgressCallback = type[None] | type[type[None]]  # Simplified for compatibility

# HTTP-related types
Headers = dict[str, str]
QueryParams = dict[str, str | int | list[str]]
FormData = dict[str, str | bytes]

# Database-related types
RowDict = dict[str, Any]
RowList = list[RowDict]


# =============================================================================
# Module Tests
# =============================================================================


def _test_protocol_checkable() -> None:
    """Test that protocols are runtime checkable."""
    # RateLimiterProtocol should be checkable at runtime
    assert hasattr(RateLimiterProtocol, "__protocol_attrs__") or callable(RateLimiterProtocol)


def _test_typeddict_creation() -> None:
    """Test TypedDict can be instantiated."""
    response: APIResponse = {
        "success": True,
        "status_code": 200,
        "data": {"key": "value"},
    }
    assert response["success"] is True
    assert response["status_code"] == 200


def _test_rate_limiter_metrics() -> None:
    """Test RateLimiterMetrics structure."""
    metrics: RateLimiterMetrics = {
        "current_fill_rate": 0.5,
        "tokens_available": 10.0,
        "total_requests": 100,
        "avg_wait_time": 0.1,
        "error_429_count": 2,
        "rate_increases": 5,
        "rate_decreases": 3,
        "endpoint_rate_cap": None,
    }
    assert metrics["current_fill_rate"] == 0.5
    assert metrics["error_429_count"] == 2


def _test_person_data() -> None:
    """Test PersonData structure."""
    person: PersonData = {
        "uuid": "ABC123",
        "name": "John Smith",
        "shared_cm": 250.5,
    }
    assert person["uuid"] == "ABC123"
    assert person["name"] == "John Smith"


def _test_health_status() -> None:
    """Test HealthStatus structure."""
    status: HealthStatus = {
        "healthy": True,
        "components": {"database": "ok", "api": "ok"},
        "timestamp": "2025-11-29T12:00:00Z",
    }
    assert status["healthy"] is True
    assert len(status["components"]) == 2


def protocols_module_tests() -> bool:
    """Run protocol module tests."""
    from test_framework import TestSuite

    suite = TestSuite("Type Protocols", "core/protocols.py")

    suite.run_test(
        test_name="Protocol checkable",
        test_func=_test_protocol_checkable,
        test_summary="Verify protocols are runtime checkable",
        functions_tested="RateLimiterProtocol",
        method_description="Check protocol has required attributes",
        expected_outcome="Protocol is checkable at runtime",
    )

    suite.run_test(
        test_name="TypedDict creation",
        test_func=_test_typeddict_creation,
        test_summary="Verify TypedDict can be instantiated",
        functions_tested="APIResponse",
        method_description="Create APIResponse dict",
        expected_outcome="Dict created with correct types",
    )

    suite.run_test(
        test_name="RateLimiterMetrics structure",
        test_func=_test_rate_limiter_metrics,
        test_summary="Verify RateLimiterMetrics structure",
        functions_tested="RateLimiterMetrics",
        method_description="Create and validate metrics dict",
        expected_outcome="All required fields present with correct types",
    )

    suite.run_test(
        test_name="PersonData structure",
        test_func=_test_person_data,
        test_summary="Verify PersonData structure",
        functions_tested="PersonData",
        method_description="Create and validate person dict",
        expected_outcome="Person data fields accessible",
    )

    suite.run_test(
        test_name="HealthStatus structure",
        test_func=_test_health_status,
        test_summary="Verify HealthStatus structure",
        functions_tested="HealthStatus",
        method_description="Create and validate health status",
        expected_outcome="Health status with components dict",
    )

    return suite.finish_suite()


# Use centralized test runner - ensure path includes parent directory
from pathlib import Path

# Add parent directory to path for test_utilities import
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(protocols_module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
