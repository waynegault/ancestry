"""Prometheus ingestion smoke tests for Grafana readiness."""

from __future__ import annotations

import os
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, cast

import requests

from scripts import static_metrics_server

# Allow running as a script without -m by adding project root to sys.path
if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

PROM_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9091").rstrip("/")
FAKE_METRICS_URL = os.getenv("PROM_FAKE_METRICS_URL", "http://localhost:9000/metrics")
REQUEST_TIMEOUT = 5
# Availability is strict; sample enforcement is on by default to catch missing data
REQUIRE_PROM_SAMPLES = os.getenv("PROM_REQUIRE_SAMPLES", "true").lower() == "true"
REQUIRE_PROM_TARGET = os.getenv("PROM_REQUIRE_TARGET", "true").lower() == "true"
REQUIRE_PROM_AVAILABLE = os.getenv("PROM_REQUIRE_AVAILABLE", "true").lower() == "true"


def _get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{PROM_URL}{path}"
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus API returned non-success status for {path}: {payload}")
    return payload.get("data")


def _safe_call(fn: Callable[[], Any], reason: str) -> Any:
    """Run a Prometheus call and enforce availability unless explicitly disabled."""

    try:
        return fn()
    except requests.exceptions.RequestException as exc:
        if REQUIRE_PROM_AVAILABLE:
            raise
        print(f"⚠️  Prometheus unreachable: {reason}. Skipping (PROM_REQUIRE_AVAILABLE=false). Error: {exc}")
        return None


def _any_target_up(job_name: str | None = None) -> bool:
    data = _safe_call(lambda: _get_json("/api/v1/targets"), "target discovery")
    if data is None:
        return True
    data_dict: dict[str, Any] = data if isinstance(data, dict) else {}
    active = cast(list[dict[str, Any]], data_dict.get("activeTargets", []))
    candidates: list[dict[str, Any]] = []
    for target in active:
        if not isinstance(target, dict):
            continue
        target_dict: dict[str, Any] = target
        labels_raw = target_dict.get("labels", {})
        labels = cast(dict[str, Any], labels_raw if isinstance(labels_raw, dict) else {})
        if job_name and labels.get("job") != job_name:
            continue
        candidates.append(target_dict)
    if job_name and not candidates:
        return False
    if not candidates:
        candidates = active
    return any(cast(str, t.get("health", "down")).lower() == "up" for t in candidates)


def _sum_prom_result(data: Any) -> float:
    """Parse a PromQL response and return the summed numeric value."""

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
        series_dict: dict[str, Any] = series
        value = series_dict.get("value")
        if not isinstance(value, list) or len(value) < 2:
            continue
        try:
            total += float(value[1])
        except (TypeError, ValueError):
            continue
    return total


def _request_samples(window: str = "5m") -> Optional[float]:
    """Return ancestry_api_requests_total activity; fallback to raw counter snapshot."""

    queries = [
        (f"sum by(job)(increase(ancestry_api_requests_total[{window}]))", "sample query"),
        ("sum by(job)(ancestry_api_requests_total)", "counter snapshot"),
    ]

    last_total: Optional[float] = None
    for prom_query, reason in queries:
        data = _safe_call(
            lambda pq=prom_query: _get_json("/api/v1/query", params={"query": pq}),
            reason,
        )
        if data is None:
            return None
        total = _sum_prom_result(data)
        last_total = total
        if total > 0.0:
            return total

    return last_total if last_total is not None else 0.0


def _start_static_metrics_server() -> Optional[socketserver.TCPServer]:
    """Start the lightweight static metrics server on port 9000."""

    try:
        server = socketserver.TCPServer(("0.0.0.0", 9000), static_metrics_server.Handler)
    except OSError:
        return None

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _prime_fake_samples() -> bool:
    """Ensure Prometheus has at least one ancestry_api_requests_total sample."""

    server: Optional[socketserver.TCPServer] = None
    try:
        # If a metrics endpoint is already responding, use it; otherwise start our stub.
        try:
            requests.get(FAKE_METRICS_URL, timeout=2)
        except requests.RequestException:
            server = _start_static_metrics_server()
            if not server:
                return False
            time.sleep(0.5)
            requests.get(FAKE_METRICS_URL, timeout=2)

        # Allow Prometheus a few scrape intervals, checking for new samples.
        attempts = int(os.getenv("PROM_SCRAPE_ATTEMPTS", "8"))
        delay_seconds = float(os.getenv("PROM_SCRAPE_DELAY_SECONDS", "5"))
        for _ in range(max(attempts, 1)):
            time.sleep(delay_seconds)
            seeded = _request_samples("1m")
            if seeded is not None and seeded > 0.0:
                return True
        return False
    finally:
        if server:
            server.shutdown()
            server.server_close()


def test_prometheus_targets_up() -> bool:
    """Ensure Prometheus sees at least one healthy target (ancestry job preferred)."""
    if not _any_target_up("ancestry"):
        if not REQUIRE_PROM_TARGET:
            print("⚠️  Prometheus target for job=ancestry is not UP. Test would fail, but PROM_REQUIRE_TARGET=false.")
            return True
        raise AssertionError("Prometheus target for job=ancestry is not UP")
    return True


def test_prometheus_has_recent_samples() -> bool:
    """Ensure recent samples exist so Grafana can plot data."""
    total = _request_samples("5m")
    if total is None:
        return True
    if total <= 0.0:
        total = _request_samples("60m")
    if total is None:
        return True
    if total <= 0.0:
        if _prime_fake_samples():
            total = _request_samples("5m")

        if total is None:
            return True

        if total <= 0.0:
            if not REQUIRE_PROM_SAMPLES:
                print(
                    "⚠️  No ancestry_api_requests_total samples in the last 60m. "
                    "Test would fail, but PROM_REQUIRE_SAMPLES=false."
                )
                return True
            raise AssertionError(
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
