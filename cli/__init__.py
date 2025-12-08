"""CLI utility package metadata and discovery helpers."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest import mock

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from testing.test_framework import TestSuite, create_standard_test_runner

CLI_ENTRY_POINTS: dict[str, str] = {
    "maintenance": "cli.maintenance:MainCLIHelpers",
}


def discover_cli_modules() -> dict[str, str]:
    """Return a copy of the registered CLI entry points."""

    return dict(CLI_ENTRY_POINTS)


def list_cli_entry_points() -> list[str]:
    """List the available CLI entry point names in sorted order."""

    return sorted(CLI_ENTRY_POINTS.keys())


def resolve_cli_entry_point(name: str) -> Any:
    """Resolve and return the CLI helper object for *name*."""

    spec = CLI_ENTRY_POINTS.get(name)
    if spec is None:
        known = ", ".join(sorted(CLI_ENTRY_POINTS)) or "(none registered)"
        raise KeyError(f"Unknown CLI entry point '{name}'. Known: {known}")

    module_name, attr_name = spec.split(":", 1)
    module = import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(f"Module '{module_name}' does not define attribute '{attr_name}'") from exc


def load_cli_entry_points() -> dict[str, Any]:
    """Eagerly resolve all registered entry points and return a mapping."""

    return {name: resolve_cli_entry_point(name) for name in CLI_ENTRY_POINTS}


__all__ = [
    "CLI_ENTRY_POINTS",
    "discover_cli_modules",
    "list_cli_entry_points",
    "load_cli_entry_points",
    "resolve_cli_entry_point",
]


# ------------------------------------------------------------------
# Embedded regression tests
# ------------------------------------------------------------------


def _test_discover_cli_modules_returns_copy() -> bool:
    modules = discover_cli_modules()
    assert modules == CLI_ENTRY_POINTS
    modules["maintenance"] = "overridden"
    assert CLI_ENTRY_POINTS["maintenance"] != "overridden"
    return True


def _test_list_cli_entry_points_sorted() -> bool:
    expected = sorted(CLI_ENTRY_POINTS.keys())
    assert list_cli_entry_points() == expected
    return True


def _test_resolve_cli_entry_point_imports_target() -> bool:
    sentinel = object()
    fake_module = mock.MagicMock()
    fake_module.MainCLIHelpers = sentinel

    patch_target = f"{__name__}.import_module"
    with mock.patch(patch_target, return_value=fake_module) as patched_import:
        result = resolve_cli_entry_point("maintenance")

    patched_import.assert_called_once_with("cli.maintenance")
    assert result is sentinel
    return True


def _test_resolve_cli_entry_point_missing_name() -> bool:
    try:
        resolve_cli_entry_point("unknown")
    except KeyError as exc:
        assert "unknown" in str(exc)
    else:  # pragma: no cover - sanity
        raise AssertionError("Expected KeyError for unknown entry point")
    return True


def _test_load_cli_entry_points_uses_resolve() -> bool:
    sentinel = object()
    patch_target = f"{__name__}.resolve_cli_entry_point"
    with mock.patch(patch_target, return_value=sentinel) as patched:
        entries = load_cli_entry_points()

    patched.assert_called()
    assert entries == dict.fromkeys(CLI_ENTRY_POINTS, sentinel)
    return True


def module_tests() -> bool:
    suite = TestSuite("cli.__init__", "cli/__init__.py")

    suite.run_test(
        "Discover returns copy",
        _test_discover_cli_modules_returns_copy,
        "Ensures discover_cli_modules returns a defensive copy.",
    )

    suite.run_test(
        "Entry list sorted",
        _test_list_cli_entry_points_sorted,
        "Ensures names are presented in deterministic order.",
    )

    suite.run_test(
        "Resolve imports module",
        _test_resolve_cli_entry_point_imports_target,
        "Ensures resolving triggers importlib and returns target attribute.",
    )

    suite.run_test(
        "Resolve missing name",
        _test_resolve_cli_entry_point_missing_name,
        "Ensures unknown names raise a descriptive KeyError.",
    )

    suite.run_test(
        "Load entry points",
        _test_load_cli_entry_points_uses_resolve,
        "Ensures bulk loader delegates to resolve function.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    sys.exit(0 if success else 1)
