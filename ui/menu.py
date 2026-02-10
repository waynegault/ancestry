#!/usr/bin/env python3
"""Menu rendering helpers for the CLI interface."""


import logging
import sys
from collections.abc import Iterable
from logging import StreamHandler
from typing import Any, TextIO

from core.action_registry import ActionCategory, ActionMetadata, ActionRegistry, ActionRequirement


def _format_menu_line(action: ActionMetadata) -> str:
    hint = f" {action.input_hint}" if action.input_hint else ""
    return f"{action.id}. {action.name}{hint}"


def _find_console_handler(logger: logging.Logger | None) -> StreamHandler[TextIO] | None:
    """Find the console handler in the logger hierarchy."""
    # Check root logger first as that's where handlers usually are
    root_logger = logging.getLogger()
    target_loggers = [root_logger]
    if logger:
        target_loggers.insert(0, logger)

    for target in target_loggers:
        if not target.handlers:
            continue

        for handler in target.handlers:
            if isinstance(handler, StreamHandler):
                # Robust console handler detection matching cli/maintenance.py
                stream = getattr(handler, "stream", None)
                is_stderr = stream == sys.stderr
                is_stdout = stream == sys.stdout
                is_console = is_stderr or is_stdout or getattr(stream, "name", "") in {"<stderr>", "<stdout>"}

                if is_console:
                    return handler
    return None


def _determine_log_level_name(logger: logging.Logger | None, config: Any) -> str:
    level_name = "UNKNOWN"

    console_handler = _find_console_handler(logger)

    if console_handler is not None:
        level_name = logging.getLevelName(int(console_handler.level))
    elif hasattr(config, "logging") and hasattr(config.logging, "log_level"):
        level_name = str(config.logging.log_level).upper()

    return level_name


def _print_action_block(actions: Iterable[ActionMetadata]) -> None:
    for action in actions:
        print(_format_menu_line(action))


def render_main_menu(
    logger: logging.Logger | None,
    config: Any,
    registry: ActionRegistry,
) -> str:
    """Print the menu and return the normalized user choice."""

    print("\nMain Menu")
    print("=" * 17)
    level_name = _determine_log_level_name(logger, config)
    print(f"(Log Level: {level_name})\n")

    _print_action_block(registry.get_menu_actions())

    # Collect all other actions (meta + test) and sort by menu_order
    other_actions = registry.get_meta_actions() + registry.get_test_actions()
    other_actions.sort(key=lambda x: x.menu_order)

    # Group 1: Analytics & Visualization (100-199)
    group1 = [a for a in other_actions if 100 <= a.menu_order < 200]
    if group1:
        print("")
        _print_action_block(group1)

    # Group 2: Testing & Maintenance (200-299)
    group2 = [a for a in other_actions if 200 <= a.menu_order < 300]
    if group2:
        print("")
        _print_action_block(group2)

    # Group 3: System (300+)
    group3 = [a for a in other_actions if a.menu_order >= 300]
    if group3:
        print("")
        _print_action_block(group3)

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
    from unittest.mock import MagicMock, patch

    config = SimpleNamespace()

    # Mock logging.getLogger to return a logger with no handlers
    mock_root = MagicMock()
    mock_root.handlers = []

    with patch("logging.getLogger", return_value=mock_root):
        result = _determine_log_level_name(None, config)

    assert result == "UNKNOWN", f"Expected 'UNKNOWN', got '{result}'"
    return True


def _test_determine_log_level_from_config() -> bool:
    """Test _determine_log_level_name extracts from config."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    config = SimpleNamespace(logging=SimpleNamespace(log_level="debug"))

    # Mock logging.getLogger to return a logger with no handlers
    mock_root = MagicMock()
    mock_root.handlers = []

    with patch("logging.getLogger", return_value=mock_root):
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


def _test_render_main_menu_output() -> bool:
    """Test that render_main_menu produces expected output."""
    import contextlib
    import io
    from unittest.mock import MagicMock, patch

    # Setup mocks
    logger = MagicMock()
    config = MagicMock()
    registry = MagicMock()

    # Mock registry actions
    action = ActionMetadata(
        id="1",
        name="Test Action",
        description="Description",
        category=ActionCategory.WORKFLOW,
        browser_requirement=ActionRequirement.NONE,
        input_hint=None,
    )
    registry.get_menu_actions.return_value = [action]
    registry.get_meta_actions.return_value = []
    registry.get_test_actions.return_value = []

    # Capture output and mock input
    output = io.StringIO()
    with contextlib.redirect_stdout(output), patch('builtins.input', return_value='1'):
        result = render_main_menu(logger, config, registry)

    # Verify
    assert result == '1', "Should return normalized input"
    captured = output.getvalue()
    assert "Main Menu" in captured, "Should print title"
    assert "1. Test Action" in captured, "Should print action"
    return True


def module_tests() -> bool:
    """Run module tests for ui.menu."""
    from testing.test_framework import TestSuite

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
        "render_main_menu output",
        _test_render_main_menu_output,
        "Ensures render_main_menu prints menu and handles input.",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    from testing.test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
