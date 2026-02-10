"""Browser Automation Package.

Provides browser automation infrastructure including:
- chromedriver: ChromeDriver management and configuration
- selenium_utils: Selenium WebDriver utilities
- css_selectors: CSS selectors for Ancestry website automation
"""

_SUBMODULES = frozenset(["chromedriver", "css_selectors", "navigation", "selenium_utils"])


def __getattr__(name: str):
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
    "Test that all browser submodules are importable."
    missing: list[str] = []
    pkg = __package__ or __name__
    for name in _SUBMODULES:
        try:
            import importlib
            importlib.import_module(f"{pkg}.{name}")
        except ImportError as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        print(f"  FAIL  {pkg}: {', '.join(missing)}")
        return False
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
