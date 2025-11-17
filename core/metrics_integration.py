"""
Metrics Integration Helpers

Integration layer for recording metrics in core managers (APIManager,
BrowserManager, DatabaseManager, SessionManager).

Provides context managers and helper functions for:
- API call timing and response tracking
- Cache hit/miss recording
- Session uptime tracking
- Database operation performance

Thread-safe wrappers around the central MetricRegistry.

Usage:
    from core.metrics_integration import record_api_call, record_cache_hit

    with record_api_call("combined_details") as metrics_ctx:
        response = api_manager.fetch_person_combined(uuid)
        metrics_ctx.record_success()
        # or metrics_ctx.record_error("429")

    record_cache_hit("CacheManager", "combined_details", hit=True)
"""

import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Support standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from core.metrics_collector import get_metrics_registry


class APICallMetricsContext:
    """Context manager for recording API call metrics."""

    def __init__(self, service_name: str, endpoint_name: str) -> None:
        self.service_name = service_name
        self.endpoint_name = endpoint_name
        self.start_time = time.time()
        self.registry = get_metrics_registry()
        self.labels: dict[str, str] = {
            "endpoint": endpoint_name,
            "service": service_name
        }

    def __enter__(self) -> "APICallMetricsContext":
        """Enter context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Record elapsed time on exit."""
        elapsed_seconds = time.time() - self.start_time
        metric_name = f"{self.endpoint_name}_response_time_ms"
        self.registry.record_timer(self.service_name, metric_name, elapsed_seconds)

    def record_success(self) -> None:
        """Record successful API call."""
        self.registry.record_metric(
            self.service_name,
            f"{self.endpoint_name}_success",
            1.0,
            self.labels
        )

    def record_error(self, error_code: str) -> None:
        """Record API error."""
        self.registry.record_metric(
            self.service_name,
            f"{self.endpoint_name}_error_{error_code}",
            1.0,
            self.labels
        )

    def record_response_size(self, size_bytes: int) -> None:
        """Record response size."""
        self.registry.record_metric(
            self.service_name,
            f"{self.endpoint_name}_response_size_bytes",
            float(size_bytes),
            self.labels
        )


@contextmanager
def record_api_call(endpoint_name: str, service_name: str = "APIManager") -> Generator[APICallMetricsContext, None, None]:
    """
    Context manager for recording API call metrics.

    Usage:
        with record_api_call("combined_details") as ctx:
            response = api_call()
            ctx.record_success()
            ctx.record_response_size(len(response))
    """
    ctx = APICallMetricsContext(service_name, endpoint_name)
    try:
        yield ctx
    except Exception as e:
        ctx.record_error(type(e).__name__)
        raise


def record_cache_hit(service_name: str, endpoint_name: str, hit: bool) -> None:
    """Record cache hit or miss."""
    registry = get_metrics_registry()
    value = 1.0 if hit else 0.0
    registry.record_metric(
        service_name,
        f"{endpoint_name}_cache_hit",
        value,
        {"endpoint": endpoint_name, "hit": "yes" if hit else "no"}
    )


def record_session_event(event_name: str, value: float = 1.0) -> None:
    """Record session-related event."""
    registry = get_metrics_registry()
    registry.record_metric("SessionManager", event_name, value)


def record_database_operation(operation_name: str, duration_seconds: float, rows_affected: int = 0) -> None:
    """Record database operation."""
    registry = get_metrics_registry()
    registry.record_timer("DatabaseManager", f"{operation_name}_duration_ms", duration_seconds)
    if rows_affected > 0:
        registry.record_metric(
            "DatabaseManager",
            f"{operation_name}_rows_affected",
            float(rows_affected)
        )


def record_browser_operation(operation_name: str, duration_seconds: float) -> None:
    """Record browser operation."""
    registry = get_metrics_registry()
    registry.record_timer("BrowserManager", f"{operation_name}_duration_ms", duration_seconds)


def get_service_health_summary(service_name: str) -> dict[str, Any]:
    """Get health summary for a service."""
    registry = get_metrics_registry()
    service_metrics = registry.get_service_metrics(service_name)

    if not service_metrics:
        return {"status": "no_data", "service": service_name}

    snapshot = registry.get_snapshot()
    service_data = snapshot.services.get(service_name, {})

    # Calculate overall health score
    error_count = 0
    success_count = 0

    for metric_name, stats in service_data.items():
        if "error" in metric_name:
            error_count += stats.get("window_1min", {}).get("count", 0)
        if "success" in metric_name:
            success_count += stats.get("window_1min", {}).get("count", 0)

    success_rate = (success_count / (success_count + error_count) * 100) if (success_count + error_count) > 0 else 0

    return {
        "service": service_name,
        "status": "healthy" if success_rate > 95 else "degraded" if success_rate > 80 else "unhealthy",
        "success_rate": success_rate,
        "recent_errors": error_count,
        "recent_successes": success_count,
        "metric_count": len(service_data)
    }


def core_metrics_integration_module_tests() -> bool:
    """Run tests for metrics integration helpers."""
    from test_framework import TestSuite

    suite = TestSuite("MetricsIntegration", "core/metrics_integration.py")
    suite.start_suite()

    # Test 1: Record API call with context manager
    def test_api_call_context() -> None:
        with record_api_call("test_endpoint") as ctx:
            ctx.record_success()
            ctx.record_response_size(1024)

        registry = get_metrics_registry()
        snapshot = registry.get_snapshot()
        assert "APIManager" in snapshot.services

    suite.run_test("API call context manager", test_api_call_context)

    # Test 2: Record cache hit/miss
    def test_cache_recording() -> None:
        record_cache_hit("CacheManager", "endpoint1", hit=True)
        record_cache_hit("CacheManager", "endpoint1", hit=False)

        registry = get_metrics_registry()
        service_metrics = registry.get_service_metrics("CacheManager")
        assert service_metrics is not None

    suite.run_test("Cache hit/miss recording", test_cache_recording)

    # Test 3: Record session event
    def test_session_event() -> None:
        record_session_event("session_refresh", 1.0)
        record_session_event("session_timeout", 1.0)

        registry = get_metrics_registry()
        service_metrics = registry.get_service_metrics("SessionManager")
        assert service_metrics is not None

    suite.run_test("Session event recording", test_session_event)

    # Test 4: Record database operation
    def test_database_operation() -> None:
        record_database_operation("select_person", 0.05, rows_affected=10)
        record_database_operation("insert_match", 0.02, rows_affected=1)

        registry = get_metrics_registry()
        service_metrics = registry.get_service_metrics("DatabaseManager")
        assert service_metrics is not None

    suite.run_test("Database operation recording", test_database_operation)

    # Test 5: Get service health
    def test_service_health() -> None:
        # Record some successful and failed operations
        with record_api_call("health_test") as ctx:
            ctx.record_success()

        health = get_service_health_summary("APIManager")
        assert health["service"] == "APIManager"
        assert "status" in health

    suite.run_test("Service health summary", test_service_health)

    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(core_metrics_integration_module_tests)


if __name__ == "__main__":
    import sys
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
