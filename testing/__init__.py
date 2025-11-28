"""Testing Infrastructure Package.

Provides testing utilities including:
- test_framework: Core testing framework
- test_utilities: Test helper utilities
- test_integration_workflow: Integration workflow tests
"""

_SUBMODULES = frozenset(["test_framework", "test_utilities", "test_integration_workflow"])


def __getattr__(name: str):
    """Lazy import submodules on attribute access."""
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available submodules."""
    return list(_SUBMODULES)
