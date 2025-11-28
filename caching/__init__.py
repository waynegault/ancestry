"""Cache Management Package.

Provides caching infrastructure including:
- cache: High-performance disk caching
- cache_manager: Centralized cache management
- cache_retention: Retention policies for cache directories
"""

_SUBMODULES = frozenset(["cache", "cache_manager", "cache_retention"])


def __getattr__(name: str):
    """Lazy import submodules on attribute access."""
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available submodules."""
    return list(_SUBMODULES)
