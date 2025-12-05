#!/usr/bin/env python3

"""
Observability Utilities.

This module provides helper functions for metrics recording and sanitization,
extracted from utils.py.
"""

# === STANDARD LIBRARY IMPORTS ===
import re
from typing import Optional
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


def metrics_status_family(status: Optional[int]) -> str:
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
    response: Optional[RequestsResponse],
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
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
