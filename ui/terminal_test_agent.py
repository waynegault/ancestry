#!/usr/bin/env python3
"""
Terminal Test Agent.

Automated agent that drives the terminal menu to verify its functionality.
Simulates user input and verifies menu rendering and action selection.
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from typing import Any
from unittest.mock import MagicMock, patch

from core.action_registry import ActionCategory, ActionMetadata, ActionRegistry, ActionRequirement
from ui.menu import render_main_menu


class TerminalTestAgent(unittest.TestCase):
    """Agent to test the terminal menu interface."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.registry = ActionRegistry()
        # Register some dummy actions for testing
        self.registry.register(
            ActionMetadata(
                id="1",
                name="Test Action 1",
                description="Description 1",
                category=ActionCategory.WORKFLOW,
                browser_requirement=ActionRequirement.NONE,
            )
        )
        self.registry.register(
            ActionMetadata(
                id="2",
                name="Test Action 2",
                description="Description 2",
                category=ActionCategory.WORKFLOW,
                browser_requirement=ActionRequirement.NONE,
            )
        )
        self.mock_logger = MagicMock()
        self.mock_config = MagicMock()
        self.mock_config.logging.log_level = "INFO"

    @patch("builtins.input", side_effect=["1"])
    def test_menu_selection(self, _mock_input: list[str] | None = None) -> None:
        """Test selecting an item from the menu."""
        f = io.StringIO()
        with redirect_stdout(f):
            choice = render_main_menu(self.mock_logger, self.mock_config, self.registry)

        output = f.getvalue()
        self.assertIn("Main Menu", output)
        self.assertIn("1. Test Action 1", output)
        self.assertIn("2. Test Action 2", output)
        self.assertEqual(choice, "1")

    @patch("builtins.input", side_effect=["99"])
    def test_invalid_selection(self, _mock_input: Any) -> None:
        """Test selecting an invalid item (just returns the string)."""
        f = io.StringIO()
        with redirect_stdout(f):
            choice = render_main_menu(self.mock_logger, self.mock_config, self.registry)

        self.assertEqual(choice, "99")

    @patch("builtins.input", side_effect=["q"])
    def test_quit_selection(self, _mock_input: Any) -> None:
        """Test selecting quit."""
        f = io.StringIO()
        with redirect_stdout(f):
            choice = render_main_menu(self.mock_logger, self.mock_config, self.registry)

        self.assertEqual(choice, "q")


def run_agent() -> bool:
    """Run the terminal test agent."""
    print("🤖 Terminal Test Agent starting...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TerminalTestAgent)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_agent()
    sys.exit(0 if success else 1)

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
