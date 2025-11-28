"""DNA Analysis Package.

Provides DNA match analysis including:
- dna_utils: Universal DNA match utilities
- dna_ethnicity_utils: DNA ethnicity region utilities
- dna_gedcom_crossref: DNA-GEDCOM cross-referencing
"""

_SUBMODULES = frozenset(["dna_ethnicity_utils", "dna_gedcom_crossref", "dna_utils"])


def __getattr__(name: str):
    """Lazy import submodules on attribute access."""
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available submodules."""
    return list(_SUBMODULES)
