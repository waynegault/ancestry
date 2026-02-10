"""External service integrations package.

This package contains integrations with external services:
- ms_graph_utils: Microsoft Graph API for Office 365 task management
"""

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    import importlib

    from testing.test_framework import TestSuite

    suite = TestSuite("integrations __init__", "integrations/__init__.py")

    def test_ms_graph_utils_imports():
        mod = importlib.import_module("integrations.ms_graph_utils")
        assert mod is not None, "ms_graph_utils should import successfully"

    def test_ms_graph_utils_exports():
        mod = importlib.import_module("integrations.ms_graph_utils")
        assert hasattr(mod, "acquire_token_device_flow"), "should export acquire_token_device_flow"
        assert callable(mod.acquire_token_device_flow), "acquire_token_device_flow should be callable"
        assert hasattr(mod, "get_todo_list_id"), "should export get_todo_list_id"
        assert callable(mod.get_todo_list_id), "get_todo_list_id should be callable"
        assert hasattr(mod, "create_todo_task"), "should export create_todo_task"
        assert callable(mod.create_todo_task), "create_todo_task should be callable"

    suite.run_test("ms_graph_utils submodule imports successfully", test_ms_graph_utils_imports)
    suite.run_test("ms_graph_utils exports key functions", test_ms_graph_utils_exports)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
