#!/usr/bin/env python3
"""Menu rendering helpers for the CLI interface."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from logging import StreamHandler
from pathlib import Path
from typing import Any, Optional, TextIO

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.action_registry import ActionCategory, ActionMetadata, ActionRegistry, ActionRequirement


def _format_menu_line(action: ActionMetadata) -> str:
    hint = f" {action.input_hint}" if action.input_hint else ""
    return f"{action.id}. {action.name}{hint}"


def _determine_log_level_name(logger: Optional[logging.Logger], config: Any) -> str:
    level_name = "UNKNOWN"

    if logger and logger.handlers:
        console_handler: Optional[StreamHandler[TextIO]] = None
        for handler in logger.handlers:
            if isinstance(handler, StreamHandler):
                stream = getattr(handler, "stream", None)
                if stream is sys.stderr:
                    console_handler = handler
                    break
        if console_handler is not None:
            level_name = logging.getLevelName(int(console_handler.level))
        else:
            level = logger.getEffectiveLevel()
            level_names = {v: k for k, v in logging.getLevelNamesMapping().items()}
            level_name = level_names.get(level, str(level))
    elif hasattr(config, "logging") and hasattr(config.logging, "log_level"):
        level_name = str(config.logging.log_level).upper()

    return level_name


def _print_action_block(actions: Iterable[ActionMetadata]) -> None:
    for action in actions:
        print(_format_menu_line(action))


def render_main_menu(
    logger: Optional[logging.Logger],
    config: Any,
    registry: ActionRegistry,
) -> str:
    """Print the menu and return the normalized user choice."""

    print("\nMain Menu")
    print("=" * 17)
    level_name = _determine_log_level_name(logger, config)
    print(f"(Log Level: {level_name})\n")

    _print_action_block(registry.get_menu_actions())

    meta_actions = registry.get_meta_actions()
    analytics_meta = [action for action in meta_actions if action.category == ActionCategory.ANALYTICS]
    graph_meta = [action for action in meta_actions if action.id == "graph"]
    system_meta = [
        action for action in meta_actions if action.category != ActionCategory.ANALYTICS and action.id != "graph"
    ]

    if analytics_meta:
        print("")
        _print_action_block(analytics_meta)

    if graph_meta:
        print("")
        _print_action_block(graph_meta)

    test_actions = registry.get_test_actions()
    if test_actions:
        print("")
        _print_action_block(test_actions)

    if system_meta:
        print("")
        _print_action_block(system_meta)

    return input("\nEnter choice: ").strip().lower()


# === TESTS ===


def _test_format_menu_line_basic() -> bool:
    """Test _format_menu_line with basic action."""
    action = ActionMetadata(
        id="1",
        name="Test Action",
        description="A test action",
        category=ActionCategory.WORKFLOW,
        browser_requirement=ActionRequirement.NONE,
        input_hint=None,
    )
    result = _format_menu_line(action)
    assert result == "1. Test Action", f"Expected '1. Test Action', got '{result}'"
    return True


def _test_format_menu_line_with_hint() -> bool:
    """Test _format_menu_line with input hint."""
    action = ActionMetadata(
        id="6",
        name="Gather Matches",
        description="Gather DNA matches",
        category=ActionCategory.WORKFLOW,
        browser_requirement=ActionRequirement.FULL_SESSION,
        input_hint="[start_page]",
    )
    result = _format_menu_line(action)
    assert result == "6. Gather Matches [start_page]", f"Expected '6. Gather Matches [start_page]', got '{result}'"
    return True


def _test_determine_log_level_unknown_fallback() -> bool:
    """Test _determine_log_level_name returns UNKNOWN when no logger."""
    from types import SimpleNamespace

    config = SimpleNamespace()
    result = _determine_log_level_name(None, config)
    assert result == "UNKNOWN", f"Expected 'UNKNOWN', got '{result}'"
    return True


def _test_determine_log_level_from_config() -> bool:
    """Test _determine_log_level_name extracts from config."""
    from types import SimpleNamespace

    config = SimpleNamespace(logging=SimpleNamespace(log_level="debug"))
    result = _determine_log_level_name(None, config)
    assert result == "DEBUG", f"Expected 'DEBUG', got '{result}'"
    return True


def _test_print_action_block_prints_actions() -> bool:
    """Test _print_action_block prints correct lines."""
    import contextlib
    import io

    actions = [
        ActionMetadata(
            id="1",
            name="Action One",
            description="First action",
            category=ActionCategory.WORKFLOW,
            browser_requirement=ActionRequirement.NONE,
            input_hint=None,
        ),
        ActionMetadata(
            id="2",
            name="Action Two",
            description="Second action",
            category=ActionCategory.WORKFLOW,
            browser_requirement=ActionRequirement.NONE,
            input_hint="[opt]",
        ),
    ]

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        _print_action_block(actions)

    result = output.getvalue()
    assert "1. Action One" in result, "Should contain '1. Action One'"
    assert "2. Action Two [opt]" in result, "Should contain '2. Action Two [opt]'"
    return True


def _test_render_main_menu_callable() -> bool:
    """Test that render_main_menu is callable."""
    assert callable(render_main_menu), "render_main_menu should be callable"
    return True


def module_tests() -> bool:
    """Run module tests for ui.menu."""
    from test_framework import TestSuite

    suite = TestSuite("ui.menu", "ui/menu.py")

    suite.run_test(
        "Format menu line basic",
        _test_format_menu_line_basic,
        "Ensures _format_menu_line formats basic actions correctly.",
    )

    suite.run_test(
        "Format menu line with hint",
        _test_format_menu_line_with_hint,
        "Ensures _format_menu_line includes input hints.",
    )

    suite.run_test(
        "Log level unknown fallback",
        _test_determine_log_level_unknown_fallback,
        "Ensures _determine_log_level_name returns UNKNOWN when no logger.",
    )

    suite.run_test(
        "Log level from config",
        _test_determine_log_level_from_config,
        "Ensures _determine_log_level_name extracts level from config.",
    )

    suite.run_test(
        "Print action block",
        _test_print_action_block_prints_actions,
        "Ensures _print_action_block prints formatted action lines.",
    )

    suite.run_test(
        "render_main_menu callable",
        _test_render_main_menu_callable,
        "Ensures render_main_menu is callable.",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    from test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
