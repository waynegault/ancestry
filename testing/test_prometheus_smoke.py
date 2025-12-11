"""Prometheus ingestion smoke tests for Grafana readiness."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast

import requests

# Allow running as a script without -m by adding project root to sys.path
if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

PROM_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
REQUEST_TIMEOUT = 5
# Strict by default: these tests should fail when Prometheus isn’t ready unless explicitly relaxed.
REQUIRE_PROM_SAMPLES = os.getenv("PROM_REQUIRE_SAMPLES", "true").lower() == "true"
REQUIRE_PROM_TARGET = os.getenv("PROM_REQUIRE_TARGET", "true").lower() == "true"


def _get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{PROM_URL}{path}"
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus API returned non-success status for {path}: {payload}")
    return payload.get("data")


def _any_target_up(job_name: str | None = None) -> bool:
    data = _get_json("/api/v1/targets")
    data_dict: dict[str, Any] = data if isinstance(data, dict) else {}
    active = cast(list[dict[str, Any]], data_dict.get("activeTargets", []))
    candidates: list[dict[str, Any]] = []
    for target in active:
        target_dict = target if isinstance(target, dict) else {}
        labels = cast(dict[str, Any], target_dict.get("labels", {}))
        if job_name and labels.get("job") != job_name:
            continue
        candidates.append(target_dict)
    if job_name and not candidates:
        return False
    if not candidates:
        candidates = active
    return any(cast(str, t.get("health", "down")).lower() == "up" for t in candidates)


def _request_samples(window: str = "5m") -> float:
    """Return summed ancestry_api_requests_total over a PromQL window."""

    query = f"sum by(job)(increase(ancestry_api_requests_total[{window}]))"
    data = _get_json("/api/v1/query", params={"query": query})
    if not isinstance(data, dict):
        return 0.0
    data_dict: dict[str, Any] = data
    result = cast(list[dict[str, Any]], data_dict.get("result", []))
    if not result:
        return 0.0
    total = 0.0
    for series in result:
        if not isinstance(series, dict):
            continue
        value = series.get("value")
        if not isinstance(value, list) or len(value) < 2:
            continue
        try:
            total += float(value[1])
        except (TypeError, ValueError):
            continue
    return total


def test_prometheus_targets_up() -> bool:
    """Ensure Prometheus sees at least one healthy target (ancestry job preferred)."""
    if not _any_target_up("ancestry"):
        if not REQUIRE_PROM_TARGET:
            print("⚠️  Prometheus target for job=ancestry is not UP. Test would fail, but PROM_REQUIRE_TARGET=false.")
            return True
        assert False, "Prometheus target for job=ancestry is not UP"
    return True


def test_prometheus_has_recent_samples() -> bool:
    """Ensure recent samples exist so Grafana can plot data."""
    total = _request_samples("5m")
    if total <= 0.0:
        total = _request_samples("60m")
    if total <= 0.0:
        if not REQUIRE_PROM_SAMPLES:
            print(
                "⚠️  No ancestry_api_requests_total samples in the last 60m. "
                "Test would fail, but PROM_REQUIRE_SAMPLES=false."
            )
            return True
        assert False, (
            "No ancestry_api_requests_total samples in the last 60m. "
            "Generate traffic (any API action) so Prometheus records metrics."
        )
    return True


def module_tests() -> bool:
    suite = TestSuite("Prometheus Smoke", "testing/test_prometheus_smoke.py")
    suite.start_suite()

    suite.run_test(
        "Prometheus target up",
        test_prometheus_targets_up,
        "Prometheus should report ancestry target as UP",
        functions_tested="_get_json, _any_target_up",
        method_description="Query /api/v1/targets and verify job=ancestry health",
        expected_outcome="At least one ancestry target is UP",
    )

    suite.run_test(
        "Recent request samples available",
        test_prometheus_has_recent_samples,
        "Prometheus should have ancestry_api_requests_total samples in last 5m",
        functions_tested="_get_json, _request_samples",
        method_description="Run increase() over 5m window for ancestry_api_requests_total (fallback 60m)",
        expected_outcome=">0 samples recorded in last 5-60m",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    raise SystemExit(0 if success else 1)
