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

    try:
        # The CLI handles its own menu loop and error catching
        run_interactive_menu()
        return True
    except Exception as e:
        logger.error(f"Error in Research Tools: {e}", exc_info=True)
        return False


def action14_module_tests() -> bool:
    """Test Action 14 module functionality."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Action 14: Research Tools", __file__)

    def test_imports() -> bool:
        """Test that required modules can be imported."""
        import cli.research_tools

        return hasattr(cli.research_tools, "run_interactive_menu")

    def test_function_signature() -> bool:
        """Test run_research_tools signature."""
        import inspect

        sig = inspect.signature(run_research_tools)
        return "session_manager" in sig.parameters

    suite.run_test("Import validation", test_imports, "Verify cli.research_tools is available")
    suite.run_test("Signature validation", test_function_signature, "Verify run_research_tools signature")

    return suite.finish_suite()


if __name__ == "__main__":
    # Run tests if executed directly
    if action14_module_tests():
        print("Action 14 tests passed!")
    else:
        print("Action 14 tests failed!")
        sys.exit(1)
