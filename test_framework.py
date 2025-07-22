from core_imports import (
    register_function,
    get_function,
    is_function_available,
    standardize_module_imports,
    auto_register_module,
)

auto_register_module(globals(), __name__)

standardize_module_imports()
#!/usr/bin/env python3
"""
Standardized Test Framework for Ancestry Project
Provides consistent test output formatting, colors, and icons across all scripts.
"""

import sys
import time
import traceback
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch


# Export commonly used testing utilities
__all__ = [
    "TestSuite",
    "suppress_logging",
    "create_mock_data",
    "assert_valid_function",
    "mock_logger_context",
    "MockLogger",
    "MagicMock",
    "patch",
]


# ANSI Color Codes for consistent formatting
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


# Icons for consistent visual indicators
class Icons:
    PASS = "✅"
    FAIL = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    GEAR = "⚙️"
    ROCKET = "🚀"
    BUG = "🐛"
    CLOCK = "⏰"
    MAGNIFY = "🔍"


class TestSuite:
    """Standardized test suite with consistent formatting and reporting."""

    def __init__(self, suite_name: str, module_name: str):
        self.suite_name = suite_name
        self.module_name = module_name
        self.start_time = None
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.warnings = 0
        self.test_results: List[Dict[str, Any]] = []

    def start_suite(self):
        """Initialize the test suite with formatted header."""
        self.start_time = time.time()
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.CYAN}{Icons.ROCKET} Testing: {self.suite_name}{Colors.RESET}"
        )
        print(f"{Colors.GRAY}Module: {self.module_name}{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

    def run_test(
        self,
        test_name: str,
        test_func: Callable,
        expected_behavior: str = "",
        test_description: str = "",
        method_description: str = "",
    ) -> bool:
        """
        Run a single test with standardized output and error handling.

        Args:
            test_name: Functions being tested
            test_func: Test function to execute
            expected_behavior: Expected outcome if functions work correctly
            test_description: What is being tested
            method_description: How the functions are being tested

        Returns:
            True if test passed, False if failed
        """
        self.tests_run += 1
        test_start = time.time()

        print(
            f"{Colors.BLUE}{Icons.GEAR} Test {self.tests_run}: {test_name}{Colors.RESET}"
        )
        if test_description:
            print(f"Test: {test_description}")
        if method_description:
            print(f"Method: {method_description}")
        if expected_behavior:
            print(f"Expected: {expected_behavior}")

        try:
            outcome_description = ""
            test_func()
            duration = time.time() - test_start
            outcome_description = (
                "Test executed successfully with all assertions passing"
            )
            print(f"Outcome: {outcome_description}")
            print(f"Duration: {duration:.3f}s")
            print(f"Conclusion: {Colors.GREEN}{Icons.PASS} PASSED{Colors.RESET}")
            print()  # Add blank line between tests

            self.tests_passed += 1
            self.test_results.append(
                {
                    "name": test_name,
                    "status": "PASSED",
                    "duration": duration,
                    "expected": expected_behavior,
                    "outcome": outcome_description,
                }
            )
            return True

        except AssertionError as e:
            import traceback

            duration = time.time() - test_start
            outcome_description = f"Assertion failed: {str(e)}"
            print(f"Outcome: {outcome_description}")
            traceback.print_exc()
            print(f"Duration: {duration:.3f}s")
            print(f"Conclusion: {Colors.RED}{Icons.FAIL} FAILED{Colors.RESET}")
            print()  # Add blank line between tests

            self.tests_failed += 1
            self.test_results.append(
                {
                    "name": test_name,
                    "status": "FAILED",
                    "duration": duration,
                    "error": str(e),
                    "expected": expected_behavior,
                    "outcome": outcome_description,
                }
            )
            return False

        except Exception as e:
            duration = time.time() - test_start
            outcome_description = f"Exception occurred: {type(e).__name__}: {str(e)}"
            print(f"Outcome: {outcome_description}")
            print(f"Duration: {duration:.3f}s")
            print(f"Conclusion: {Colors.RED}{Icons.FAIL} FAILED{Colors.RESET}")
            print()  # Add blank line between tests

            self.tests_failed += 1
            self.test_results.append(
                {
                    "name": test_name,
                    "status": "ERROR",
                    "duration": duration,
                    "error": f"{type(e).__name__}: {str(e)}",
                    "expected": expected_behavior,
                    "outcome": outcome_description,
                }
            )
            return False

    def add_warning(self, message: str):
        """Add a warning message to the test output."""
        self.warnings += 1
        print(f"  {Colors.YELLOW}{Icons.WARNING} WARNING: {message}{Colors.RESET}")

    def finish_suite(self) -> bool:
        """Complete the test suite and print summary."""
        total_duration = time.time() - self.start_time if self.start_time else 0

        print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.CYAN}{Icons.MAGNIFY} Test Summary: {self.suite_name}{Colors.RESET}"
        )
        print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")

        # Overall results
        if self.tests_failed == 0:
            status_color = Colors.GREEN
            status_icon = Icons.PASS
            status_text = "ALL TESTS PASSED"
        else:
            status_color = Colors.RED
            status_icon = Icons.FAIL
            status_text = "SOME TESTS FAILED"

        print(
            f"{status_color}{Icons.CLOCK} Duration: {total_duration:.3f}s{Colors.RESET}"
        )
        print(f"{status_color}{status_icon} Status: {status_text}{Colors.RESET}")
        print(f"{Colors.GREEN}{Icons.PASS} Passed: {self.tests_passed}{Colors.RESET}")
        print(f"{Colors.RED}{Icons.FAIL} Failed: {self.tests_failed}{Colors.RESET}")
        if self.warnings > 0:
            print(
                f"{Colors.YELLOW}{Icons.WARNING} Warnings: {self.warnings}{Colors.RESET}"
            )

        # Detailed results for failed tests
        failed_tests = [
            r for r in self.test_results if r["status"] in ["FAILED", "ERROR"]
        ]
        if failed_tests:
            print(f"\n{Colors.YELLOW}{Icons.INFO} Failed Test Details:{Colors.RESET}")
            for test in failed_tests:
                print(f"  {Colors.RED}• {test['name']}{Colors.RESET}")
                if "error" in test:
                    print(f"    {Colors.GRAY}{test['error']}{Colors.RESET}")
                if test.get("expected"):
                    print(
                        f"    {Colors.GRAY}Expected: {test['expected']}{Colors.RESET}"
                    )

        print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

        return self.tests_failed == 0


@contextmanager
def suppress_logging():
    """Context manager to suppress logging during tests."""
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def create_mock_data():
    """Create standard mock data for testing."""
    return {
        "mock_session_manager": MagicMock(),
        "mock_db_session": MagicMock(),
        "mock_config": MagicMock(),
        "sample_uuid": "TEST-UUID-1234-5678-ABCD",
        "sample_dna_data": {
            "uuid": "TEST-UUID-1234-5678-ABCD",
            "cM_DNA": 85,
            "shared_segments": 4,
            "username": "Test User",
        },
    }


def assert_valid_function(func: Any, func_name: str):
    """Assert that a function exists and is callable."""
    assert func is not None, f"Function {func_name} should exist"
    assert callable(func), f"Function {func_name} should be callable"


def assert_valid_config(config: Any, required_attrs: List[str]):
    """Assert that a config object has required attributes."""
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config should have attribute {attr}"


def test_framework_module_tests():
    """
    Comprehensive test suite for the test framework module.
    Tests all core functionality including colors, icons, test suite operations, and mock data.
    """
    print(f"{Icons.ROCKET} Running comprehensive tests for test_framework.py...")

    suite = TestSuite("Test Framework Comprehensive Tests", "test_framework.py")
    suite.start_suite()

    def test_colors():
        """Test that all color constants are properly defined."""
        assert Colors.RED == "\033[91m"
        assert Colors.GREEN == "\033[92m"
        assert Colors.YELLOW == "\033[93m"
        assert Colors.BLUE == "\033[94m"
        assert Colors.MAGENTA == "\033[95m"
        assert Colors.CYAN == "\033[96m"
        assert Colors.WHITE == "\033[97m"
        assert Colors.GRAY == "\033[90m"
        assert Colors.BOLD == "\033[1m"
        assert Colors.UNDERLINE == "\033[4m"
        assert Colors.RESET == "\033[0m"

    def test_icons():
        """Test that all icon constants are properly defined."""
        assert Icons.PASS == "✅"
        assert Icons.FAIL == "❌"
        assert Icons.WARNING == "⚠️"
        assert Icons.INFO == "ℹ️"
        assert Icons.GEAR == "⚙️"
        assert Icons.ROCKET == "🚀"
        assert Icons.BUG == "🐛"
        assert Icons.CLOCK == "⏰"
        assert Icons.MAGNIFY == "🔍"

    def test_mock_data():
        """Test mock data creation functionality."""
        data = create_mock_data()
        assert isinstance(data, dict)
        assert "mock_session_manager" in data
        assert "sample_dna_data" in data
        assert data["sample_dna_data"]["cM_DNA"] == 85
        assert isinstance(data["mock_session_manager"], MagicMock)

    def test_test_suite_creation():
        """Test that TestSuite can be created and initialized properly."""
        test_suite = TestSuite("Test Suite", "test_module.py")
        assert test_suite.suite_name == "Test Suite"
        assert test_suite.module_name == "test_module.py"
        assert test_suite.start_time is None

    def test_context_managers():
        """Test that context managers work properly."""
        with suppress_logging():
            logging.critical("This logging should be suppressed")

        # Test that it doesn't raise an exception
        import os  # Should work fine

        assert os.path.exists(".")

    suite.run_test(
        "Color constants", test_colors, "Should define all standard ANSI color codes"
    )
    suite.run_test(
        "Icon constants", test_icons, "Should define all standard Unicode icons"
    )
    suite.run_test(
        "Mock data creation",
        test_mock_data,
        "Should create valid test data structures",
    )
    suite.run_test(
        "TestSuite creation",
        test_test_suite_creation,
        "Should create TestSuite instances properly",
    )
    suite.run_test(
        "Context managers",
        test_context_managers,
        "Should provide working context managers",
    )

    return suite.finish_suite()


def run_comprehensive_tests():
    """Run comprehensive tests using the unified test framework."""
    from test_framework_unified import run_unified_tests

    return run_unified_tests("test_framework", test_framework_module_tests)


if __name__ == "__main__":
    # Test the framework itself
    def demo_tests():
        suite = TestSuite("Test Framework Demo", "test_framework.py")
        suite.start_suite()

        def test_colors():
            assert Colors.RED == "\033[91m"
            assert Colors.GREEN == "\033[92m"
            assert Colors.RESET == "\033[0m"

        def test_icons():
            assert Icons.PASS == "✅"
            assert Icons.FAIL == "❌"

        def test_mock_data():
            data = create_mock_data()
            assert "mock_session_manager" in data
            assert data["sample_dna_data"]["cM_DNA"] == 85

        suite.run_test(
            "Color constants", test_colors, "Should define standard ANSI color codes"
        )
        suite.run_test(
            "Icon constants", test_icons, "Should define standard Unicode icons"
        )
        suite.run_test(
            "Mock data creation",
            test_mock_data,
            "Should create valid test data structures",
        )

        return suite.finish_suite()

    print(f"{Icons.ROCKET} Testing the test framework itself...")
    success = demo_tests()
    sys.exit(0 if success else 1)


# Test utility classes for eliminating code duplication
class MockLogger:
    """
    Reusable mock logger for testing.
    Eliminates the need for DummyLogger class duplication across test modules.
    """

    def __init__(self):
        self.lines = []
        self.messages = {
            "debug": [],
            "info": [],
            "warning": [],
            "error": [],
            "critical": [],
        }

    def debug(self, msg, **kwargs):
        self.lines.append(msg)
        self.messages["debug"].append(msg)

    def info(self, msg, **kwargs):
        self.lines.append(msg)
        self.messages["info"].append(msg)

    def warning(self, msg, **kwargs):
        self.lines.append(msg)
        self.messages["warning"].append(msg)

    def error(self, msg, **kwargs):
        self.lines.append(msg)
        self.messages["error"].append(msg)

    def critical(self, msg, **kwargs):
        self.lines.append(msg)
        self.messages["critical"].append(msg)

    def get_messages(self, level: Optional[str] = None) -> List[str]:
        """Get messages by level, or all messages if level is None"""
        if level:
            return self.messages.get(level, [])
        return self.lines

    def clear(self):
        """Clear all logged messages"""
        self.lines.clear()
        for level_msgs in self.messages.values():
            level_msgs.clear()


@contextmanager
def mock_logger_context(module_globals: dict, logger_name: str = "logger"):
    """
    Context manager for temporarily replacing a module's logger with MockLogger.

    Usage:
        with mock_logger_context(globals()) as mock_logger:
            # Test code that uses logger
            pass
    """
    original_logger = module_globals.get(logger_name)
    mock_logger = MockLogger()
    module_globals[logger_name] = mock_logger
    try:
        yield mock_logger
    finally:
        if original_logger:
            module_globals[logger_name] = original_logger
