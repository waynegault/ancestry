"""Research Utilities Package.

Provides research-related utilities including:
- relationship_utils: Relationship calculations and pathfinding
- relationship_diagram: Relationship diagram generation
- research_suggestions: Research suggestion generation
- research_guidance_prompts: AI prompts for research guidance
- research_prioritization: Research task prioritization
"""

from typing import Any

_SUBMODULES = frozenset(
    [
        "relationship_diagram",
        "relationship_utils",
        "research_guidance_prompts",
        "research_prioritization",
        "research_suggestions",
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

    suite = TestSuite("research __init__", "research/__init__.py")

    def _make_import_test(sub: str):
        def test():
            mod = importlib.import_module(f"research.{sub}")
            assert mod is not None, f"research.{sub} should import successfully"
            assert hasattr(mod, "__name__"), f"research.{sub} should have __name__"
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
