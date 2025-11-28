"""GEDCOM Processing Package.

Provides GEDCOM file parsing and analysis including:
- gedcom_utils: Core GEDCOM processing utilities
- gedcom_cache: GEDCOM data caching
- gedcom_intelligence: GEDCOM analysis and insights
- gedcom_search_utils: GEDCOM search functionality
"""

_SUBMODULES = frozenset(["gedcom_cache", "gedcom_intelligence", "gedcom_search_utils", "gedcom_utils"])


def __getattr__(name: str):
    """Lazy import submodules on attribute access."""
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available submodules."""
    return list(_SUBMODULES)
