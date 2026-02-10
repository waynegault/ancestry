"""Testing Infrastructure Package.

Provides testing utilities including:
- test_framework: Core testing framework
- test_utilities: Test helper utilities
- test_integration_workflow: Integration workflow tests
"""

from typing import Any

_SUBMODULES = frozenset(["test_framework", "test_utilities", "test_integration_workflow"])


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

    suite = TestSuite("testing __init__", "testing/__init__.py")

    def test_testsuite_is_class():
        assert isinstance(TestSuite, type), "TestSuite should be a class"

    def test_create_standard_test_runner_callable():
        assert callable(create_standard_test_runner)
        assert create_standard_test_runner.__name__ == "create_standard_test_runner"

    def test_create_standard_test_runner_returns_callable():
        runner = create_standard_test_runner(lambda: True)
        assert callable(runner), "create_standard_test_runner should return a callable"

    def test_testsuite_instantiation():
        ts = TestSuite("test", "test.py")
        assert isinstance(ts, TestSuite), "Should create a TestSuite instance"
        assert hasattr(ts, "run_test"), "TestSuite should have run_test method"
        assert hasattr(ts, "finish_suite"), "TestSuite should have finish_suite method"

    def test_lazy_submodule_imports():
        tf = importlib.import_module("testing.test_framework")
        assert hasattr(tf, "TestSuite"), "test_framework should export TestSuite"
        tu = importlib.import_module("testing.test_utilities")
        assert hasattr(tu, "create_standard_test_runner"), "test_utilities should export create_standard_test_runner"

    suite.run_test("TestSuite is importable and is a class", test_testsuite_is_class)
    suite.run_test("create_standard_test_runner is callable", test_create_standard_test_runner_callable)
    suite.run_test("create_standard_test_runner returns a callable", test_create_standard_test_runner_returns_callable)
    suite.run_test("TestSuite can be instantiated", test_testsuite_instantiation)
    suite.run_test("Lazy submodule imports work", test_lazy_submodule_imports)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
