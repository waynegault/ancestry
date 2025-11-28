"""Browser Automation Package.

Provides browser automation infrastructure including:
- chromedriver: ChromeDriver management and configuration
- selenium_utils: Selenium WebDriver utilities
- css_selectors: CSS selectors for Ancestry website automation
"""

_SUBMODULES = frozenset(["chromedriver", "css_selectors", "selenium_utils"])


def __getattr__(name: str):
    """Lazy import submodules on attribute access."""
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available submodules."""
    return list(_SUBMODULES)
