"""UI helpers for the Ancestry automation toolkit."""

import sys

from ui.menu import render_main_menu

__all__ = [
    "render_main_menu",
]


# === TESTS ===


def _test_render_main_menu_exists() -> bool:
    """Test that render_main_menu is importable and has correct signature."""
    import inspect

    assert callable(render_main_menu), "render_main_menu should be callable"
    sig = inspect.signature(render_main_menu)
    params = list(sig.parameters.keys())
    assert "logger" in params, "Should accept logger"
    assert "config" in params, "Should accept config"
    assert "registry" in params, "Should accept registry"
    return True


def _test_ui_package_exports() -> bool:
    """Test that __all__ exports expected items."""
    assert "render_main_menu" in __all__, "render_main_menu should be in __all__"
    return True


def module_tests() -> bool:
    """Run module tests for ui package."""
    from testing.test_framework import TestSuite

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
    from testing.test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
