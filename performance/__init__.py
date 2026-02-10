"""Performance Monitoring Package.

Provides performance monitoring and optimization including:
- performance_monitor: System performance monitoring
- performance_profiling: Performance profiling utilities
- performance_orchestrator: Performance orchestration
- performance_cache: Performance-related caching
- health_monitor: System health monitoring
- grafana_checker: Grafana integration checker
"""

from typing import Any

_SUBMODULES = frozenset(
    [
        "grafana_checker",
        "health_monitor",
        "performance_cache",
        "performance_monitor",
        "performance_orchestrator",
        "performance_profiling",
    ]
)


def __getattr__(name: str) -> Any:
    """Lazy import submodules on attribute access."""
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available submodules."""
    return list(_SUBMODULES)


# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    import importlib

    from testing.test_framework import TestSuite

    suite = TestSuite("performance __init__", "performance/__init__.py")

    def _make_import_test(sub: str):
        def test():
            mod = importlib.import_module(f"performance.{sub}")
            assert mod is not None, f"performance.{sub} should import successfully"
            assert hasattr(mod, "__name__"), f"performance.{sub} should have __name__"
        return test

    for submodule in sorted(_SUBMODULES):
        suite.run_test(f"{submodule} submodule imports successfully", _make_import_test(submodule))

    def test_dir_lists_submodules():
        entries = __dir__()
        assert isinstance(entries, list), "__dir__ should return a list"
        assert set(entries) == _SUBMODULES, f"__dir__ should list all submodules, got {entries}"

    suite.run_test("__dir__ lists all submodules", test_dir_lists_submodules)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
