"""Observability utilities and Prometheus integration helpers."""

import sys
from pathlib import Path

# Ensure repo root is in path for standalone execution
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from observability.metrics_exporter import (
    get_metrics_exporter_address,
    is_metrics_exporter_running,
    start_metrics_exporter,
    stop_metrics_exporter,
)
from observability.metrics_registry import (
    PROMETHEUS_AVAILABLE,
    configure_metrics,
    disable_metrics,
    get_metrics_registry,
    is_metrics_enabled,
    metrics,
    reset_metrics,
)

__all__ = [
    "PROMETHEUS_AVAILABLE",
    "configure_metrics",
    "disable_metrics",
    "get_metrics_exporter_address",
    "get_metrics_registry",
    "is_metrics_enabled",
    "is_metrics_exporter_running",
    "metrics",
    "reset_metrics",
    "start_metrics_exporter",
    "stop_metrics_exporter",
]


# === TESTS ===


def _test_prometheus_available_is_bool() -> bool:
    """Test that PROMETHEUS_AVAILABLE is a boolean."""
    assert isinstance(PROMETHEUS_AVAILABLE, bool), "PROMETHEUS_AVAILABLE should be a boolean"
    return True


def _test_metrics_registry_functions_exist() -> bool:
    """Test that core metrics registry functions are callable."""
    assert callable(configure_metrics), "configure_metrics should be callable"
    assert callable(disable_metrics), "disable_metrics should be callable"
    assert callable(get_metrics_registry), "get_metrics_registry should be callable"
    assert callable(is_metrics_enabled), "is_metrics_enabled should be callable"
    assert callable(reset_metrics), "reset_metrics should be callable"
    return True


def _test_metrics_exporter_functions_exist() -> bool:
    """Test that core metrics exporter functions are callable."""
    assert callable(start_metrics_exporter), "start_metrics_exporter should be callable"
    assert callable(stop_metrics_exporter), "stop_metrics_exporter should be callable"
    assert callable(is_metrics_exporter_running), "is_metrics_exporter_running should be callable"
    assert callable(get_metrics_exporter_address), "get_metrics_exporter_address should be callable"
    return True


def _test_is_metrics_enabled_returns_bool() -> bool:
    """Test that is_metrics_enabled returns a boolean."""
    result = is_metrics_enabled()
    assert isinstance(result, bool), "is_metrics_enabled should return a boolean"
    return True


def _test_is_metrics_exporter_running_returns_bool() -> bool:
    """Test that is_metrics_exporter_running returns a boolean."""
    result = is_metrics_exporter_running()
    assert isinstance(result, bool), "is_metrics_exporter_running should return a boolean"
    return True


def module_tests() -> bool:
    """Run module tests for observability package."""
    from test_framework import TestSuite

    suite = TestSuite("observability", "observability/__init__.py")

    suite.run_test(
        "PROMETHEUS_AVAILABLE is bool",
        _test_prometheus_available_is_bool,
        "Ensures PROMETHEUS_AVAILABLE is a boolean indicating prometheus availability.",
    )

    suite.run_test(
        "Metrics registry functions exist",
        _test_metrics_registry_functions_exist,
        "Ensures core metrics registry functions are callable.",
    )

    suite.run_test(
        "Metrics exporter functions exist",
        _test_metrics_exporter_functions_exist,
        "Ensures core metrics exporter functions are callable.",
    )

    suite.run_test(
        "is_metrics_enabled returns bool",
        _test_is_metrics_enabled_returns_bool,
        "Ensures is_metrics_enabled returns a boolean.",
    )

    suite.run_test(
        "is_metrics_exporter_running returns bool",
        _test_is_metrics_exporter_running_returns_bool,
        "Ensures is_metrics_exporter_running returns a boolean.",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    from test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
