#!/usr/bin/env python3
"""Menu rendering helpers for the CLI interface."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from logging import StreamHandler
from typing import Any, Optional, TextIO

from core.action_registry import ActionCategory, ActionMetadata, ActionRegistry


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
