#!/usr/bin/env python3

"""
Action 14: Research Tools & Innovation Features

Provides access to advanced research tools including:
- Ethnicity Analysis
- Relationship Diagrams
- Research Prioritization
- Triangulation Intelligence
- Gap Detection
- Sentiment Analysis
"""

import logging
import sys
from typing import Any

from cli.research_tools import run_interactive_menu
from core.session_manager import SessionManager

logger = logging.getLogger(__name__)


def run_research_tools(session_manager: SessionManager, *_: Any) -> bool:
    """
    Execute the Research Tools interactive menu.

    Args:
        session_manager: The active session manager (not strictly required for all tools,
                        but passed for consistency with action signature)

    Returns:
        bool: True if execution completed successfully
    """
    logger.info("Starting Action 14: Research Tools")

    if session_manager is None:
        logger.debug("Session manager not provided for research tools")

    try:
        # The CLI handles its own menu loop and error catching
        run_interactive_menu()
        return True
    except Exception as e:
        logger.error(f"Error in Research Tools: {e}", exc_info=True)
        return False


def action14_module_tests() -> bool:
    """Test Action 14 module functionality."""
    from unittest.mock import MagicMock, patch

    from testing.test_framework import TestSuite

    suite = TestSuite("Action 14: Research Tools", __file__)

    def test_delegates_to_interactive_menu() -> None:
        """Test that run_research_tools delegates to cli.research_tools.run_interactive_menu."""
        mock_sm = MagicMock()
        with patch("actions.action14_research_tools.run_interactive_menu") as mock_menu:
            result = run_research_tools(mock_sm)
            mock_menu.assert_called_once()
            assert result is True, "Should return True on successful execution"

    def test_returns_false_on_error() -> None:
        """Test that run_research_tools returns False when menu raises."""
        mock_sm = MagicMock()
        with patch("actions.action14_research_tools.run_interactive_menu", side_effect=RuntimeError("test")):
            result = run_research_tools(mock_sm)
            assert result is False, "Should return False on exception"

    def test_handles_none_session_manager() -> None:
        """Test that run_research_tools handles None session_manager gracefully."""
        with patch("actions.action14_research_tools.run_interactive_menu"):
            result = run_research_tools(None)  # type: ignore[arg-type]
            assert result is True, "Should handle None session_manager without crashing"

    suite.run_test("Delegates to interactive menu", test_delegates_to_interactive_menu)
    suite.run_test("Returns False on error", test_returns_false_on_error)
    suite.run_test("Handles None session_manager", test_handles_none_session_manager)

    return suite.finish_suite()


if __name__ == "__main__":
    # Run tests if executed directly
    if action14_module_tests():
        print("Action 14 tests passed!")
    else:
        print("Action 14 tests failed!")
        sys.exit(1)
