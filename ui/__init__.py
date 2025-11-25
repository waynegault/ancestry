"""UI helpers for the Ancestry automation toolkit."""

import sys
from pathlib import Path

# Ensure repo root is in path for standalone execution
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ui.menu import render_main_menu

__all__ = [
    "render_main_menu",
]


# === TESTS ===


def _test_render_main_menu_exists() -> bool:
    """Test that render_main_menu is importable and callable."""
    assert callable(render_main_menu), "render_main_menu should be callable"
    return True


def _test_ui_package_exports() -> bool:
    """Test that __all__ exports expected items."""
    assert "render_main_menu" in __all__, "render_main_menu should be in __all__"
    return True


def module_tests() -> bool:
    """Run module tests for ui package."""
    from test_framework import TestSuite

    suite = TestSuite("ui", "ui/__init__.py")

    suite.run_test(
        "render_main_menu exists",
        _test_render_main_menu_exists,
        "Ensures render_main_menu is importable and callable.",
    )

    suite.run_test(
        "UI package exports",
        _test_ui_package_exports,
        "Ensures __all__ exports expected items.",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    from test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
