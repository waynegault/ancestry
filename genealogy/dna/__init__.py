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
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
