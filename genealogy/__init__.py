"""Genealogy Processing Package.

Provides genealogical data processing including:
- gedcom/: GEDCOM file parsing and analysis
- dna/: DNA match utilities and analysis
- normalization: Data normalization helpers
- presenter: Genealogy presentation utilities
- scoring: Universal scoring for genealogical data
"""

_SUBMODULES = frozenset(["dna", "gedcom"])


def __getattr__(name: str):
    """Lazy import submodules on attribute access."""
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available submodules."""
    return list(_SUBMODULES)
