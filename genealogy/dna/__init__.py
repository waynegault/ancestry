"""DNA Analysis Package.

Provides DNA match analysis including:
- dna_utils: Universal DNA match utilities
- dna_ethnicity_utils: DNA ethnicity region utilities
- dna_gedcom_crossref: DNA-GEDCOM cross-referencing
"""

from typing import Any

_SUBMODULES = frozenset(["dna_ethnicity_utils", "dna_gedcom_crossref", "dna_utils"])


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

    suite = TestSuite("genealogy.dna __init__", "genealogy/dna/__init__.py")

    def _make_import_test(sub: str):
        def test():
            mod = importlib.import_module(f"genealogy.dna.{sub}")
            assert mod is not None, f"genealogy.dna.{sub} should import successfully"
            assert hasattr(mod, "__name__"), f"genealogy.dna.{sub} should have __name__"
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
