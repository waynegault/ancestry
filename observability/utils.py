#!/usr/bin/env python3

"""
Observability Utilities.

This module provides helper functions for metrics recording and sanitization,
extracted from utils.py.
"""

# === STANDARD LIBRARY IMPORTS ===
import re
from urllib.parse import urlparse

# === THIRD-PARTY IMPORTS ===
from requests import Response as RequestsResponse

# === LOCAL IMPORTS ===
from observability.metrics_registry import metrics

# === CONSTANTS ===
_UUID_PATH_SEGMENT_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def sanitize_metric_segment(segment: str) -> str:
    """Reduce high-cardinality path segments for metrics labeling."""
    if segment.isdigit():
        return ":id"
    if _UUID_PATH_SEGMENT_PATTERN.match(segment):
        return ":uuid"
    if segment.startswith("person_"):
        return ":person_id"
    if segment.startswith("tree_"):
        return ":tree_id"
    return segment


def derive_metrics_endpoint(url: str) -> str:
    """Generate a normalized endpoint label from a URL."""
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return "root"
        segments = path.split("/")
        # Keep first 2 segments, sanitize rest
        sanitized = [sanitize_metric_segment(s) if i >= 2 else s for i, s in enumerate(segments)]
        return "/".join(sanitized)
    except Exception:
        return "unknown"


def metrics_status_family(status: int | None) -> str:
    """Convert HTTP status code to status family string."""
    if status is None:
        return "error"
    if 200 <= status < 300:
        return "2xx"
    if 300 <= status < 400:
        return "3xx"
    if 400 <= status < 500:
        return "4xx"
    if 500 <= status < 600:
        return "5xx"
    return "other"


def resolve_request_duration(
    response: RequestsResponse | None,
    fallback_duration: float,
) -> float:
    """Prefer requests' elapsed timing when available."""
    if response is not None and hasattr(response, "elapsed"):
        return response.elapsed.total_seconds()
    return fallback_duration


def record_api_metrics(
    endpoint: str,
    method: str,
    result: str,
    status_family: str,
    duration: float,
) -> None:
    """Emit API metrics via Prometheus registry helpers."""
    try:
        bundle = metrics()
        bundle.api_requests.inc(endpoint, method, result)
        bundle.api_latency.observe(endpoint, status_family, duration)
    except Exception:
        # Metrics recording should never crash the app
        pass


# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    from unittest.mock import MagicMock

    from testing.test_framework import TestSuite

    suite = TestSuite("Observability Utils", "observability/utils.py")
    suite.start_suite()

    def test_sanitize_metric_segment():
        assert sanitize_metric_segment("12345") == ":id"
        assert sanitize_metric_segment("abcdef01-2345-6789-abcd-ef0123456789") == ":uuid"
        assert sanitize_metric_segment("person_12345") == ":person_id"
        assert sanitize_metric_segment("tree_abc") == ":tree_id"
        assert sanitize_metric_segment("matches") == "matches"
        return True

    suite.run_test("sanitize_metric_segment reduces high-cardinality segments", test_sanitize_metric_segment)

    def test_derive_metrics_endpoint():
        result = derive_metrics_endpoint("https://example.com/api/v1/person_123/details")
        assert isinstance(result, str)
        assert ":person_id" in result
        assert derive_metrics_endpoint("") == "root"
        assert derive_metrics_endpoint("https://example.com/") == "root"
        return True

    suite.run_test("derive_metrics_endpoint normalizes URLs", test_derive_metrics_endpoint)

    def test_metrics_status_family():
        assert metrics_status_family(200) == "2xx"
        assert metrics_status_family(201) == "2xx"
        assert metrics_status_family(301) == "3xx"
        assert metrics_status_family(404) == "4xx"
        assert metrics_status_family(429) == "4xx"
        assert metrics_status_family(500) == "5xx"
        assert metrics_status_family(None) == "error"
        assert metrics_status_family(100) == "other"
        return True

    suite.run_test("metrics_status_family maps status codes correctly", test_metrics_status_family)

    def test_resolve_request_duration_with_response():
        mock_resp = MagicMock()
        mock_resp.elapsed.total_seconds.return_value = 1.25
        result = resolve_request_duration(mock_resp, fallback_duration=5.0)
        assert result == 1.25
        return True

    suite.run_test("resolve_request_duration prefers response.elapsed", test_resolve_request_duration_with_response)

    def test_resolve_request_duration_fallback():
        result = resolve_request_duration(None, fallback_duration=3.5)
        assert result == 3.5
        return True

    suite.run_test("resolve_request_duration uses fallback when no response", test_resolve_request_duration_fallback)

    def test_record_api_metrics_calls_bundle():
        import sys
        obs_utils = sys.modules[__name__]
        mock_bundle = MagicMock()
        original_metrics = obs_utils.metrics
        obs_utils.metrics = lambda: mock_bundle
        try:
            record_api_metrics("test_endpoint", "GET", "success", "2xx", 0.5)
            mock_bundle.api_requests.inc.assert_called_once_with("test_endpoint", "GET", "success")
            mock_bundle.api_latency.observe.assert_called_once_with("test_endpoint", "2xx", 0.5)
        finally:
            obs_utils.metrics = original_metrics
        return True

    suite.run_test("record_api_metrics emits to metrics bundle", test_record_api_metrics_calls_bundle)

    def test_all_functions_are_callable():
        assert callable(sanitize_metric_segment)
        assert callable(derive_metrics_endpoint)
        assert callable(metrics_status_family)
        assert callable(resolve_request_duration)
        assert callable(record_api_metrics)
        return True

    suite.run_test("All utility functions are callable", test_all_functions_are_callable)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
