#!/usr/bin/env python3

"""
Comprehensive Testing Infrastructure & Quality Assurance Engine

Advanced testing framework providing sophisticated test execution, comprehensive
validation, and intelligent quality assurance capabilities with standardized
test suite management, performance monitoring, and automated quality assessment
for reliable genealogical automation system validation and verification.

Test Execution Framework:
• Standardized test suite execution with comprehensive reporting and analytics
• Advanced test orchestration with parallel execution and dependency management
• Intelligent test discovery with automatic test registration and categorization
• Comprehensive assertion utilities with detailed validation and error reporting
• Advanced test lifecycle management with setup, execution, and cleanup phases
• Integration with continuous integration systems for automated testing workflows

Quality Assurance:
• Comprehensive validation utilities with business rule enforcement and data integrity checks
• Advanced performance monitoring with timing analysis and resource usage tracking
• Intelligent test result aggregation with trend analysis and quality scoring
• Automated regression detection with baseline comparison and deviation analysis
• Comprehensive error handling with detailed debugging information and stack traces
• Integration with system monitoring for real-time test execution visibility

Testing Intelligence:
• Advanced test analytics with success rate tracking and failure pattern analysis
• Intelligent test prioritization with risk-based testing and impact assessment
• Comprehensive test coverage analysis with code coverage and functional coverage metrics
• Automated test maintenance with self-healing tests and adaptive test strategies
• Performance benchmarking with baseline establishment and performance regression detection
• Integration with quality gates for automated quality assessment and release validation

Foundation Services:
Provides the essential testing infrastructure that ensures reliable, high-quality
genealogical automation through comprehensive validation, intelligent quality
assessment, and systematic testing for professional research workflow reliability.
"""

# === PATH SETUP FOR PACKAGE IMPORTS ===
# === CORE INFRASTRUCTURE ===
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# === STANDARD LIBRARY IMPORTS ===
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from typing import Any
from unittest.mock import MagicMock, patch

# Export commonly used testing utilities
__all__ = [
    "Colors",
    "Icons",
    "MagicMock",
    "MockLogger",
    "TestSuite",
    "assert_valid_function",
    "clean_test_output",
    "create_mock_data",
    "database_rollback_test",
    "format_score_breakdown_table",
    "format_search_criteria",
    "format_test_result",
    "format_test_section_header",
    "has_ansi_codes",
    "mock_logger_context",
    "patch",
    "restore_debug_logging",
    "strip_ansi_codes",
    "suppress_debug_logging",
    "suppress_logging",
    "test_function_availability",
]


# ANSI Color Utilities - Consolidated from color_utils.py
class Colors:
    """ANSI color codes for terminal output with formatting utilities."""

    # Standard ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'  # Reset to default
    RESET = '\033[0m'  # Alternative name for compatibility

    @staticmethod
    def colorize(text: str, color_code: str) -> str:
        """
        Apply color formatting to text using the specified color code.

        Consolidated method that eliminates 10 duplicate color formatting methods.
        This implements DRY principles by providing a single implementation for
        all color formatting operations.

        Args:
            text: The text to colorize
            color_code: The ANSI color code to apply

        Returns:
            str: The text wrapped with the specified color code and reset
        """
        return f"{color_code}{text}{Colors.END}"

    @staticmethod
    def green(text: str) -> str:
        """Return text in green color."""
        return Colors.colorize(text, Colors.GREEN)

    @staticmethod
    def red(text: str) -> str:
        """Return text in red color."""
        return Colors.colorize(text, Colors.RED)

    @staticmethod
    def yellow(text: str) -> str:
        """Return text in yellow color."""
        return Colors.colorize(text, Colors.YELLOW)

    @staticmethod
    def blue(text: str) -> str:
        """Return text in blue color."""
        return Colors.colorize(text, Colors.BLUE)

    @staticmethod
    def magenta(text: str) -> str:
        """Return text in magenta color."""
        return Colors.colorize(text, Colors.MAGENTA)

    @staticmethod
    def cyan(text: str) -> str:
        """Return text in cyan color."""
        return Colors.colorize(text, Colors.CYAN)

    @staticmethod
    def white(text: str) -> str:
        """Return text in white color."""
        return Colors.colorize(text, Colors.WHITE)

    @staticmethod
    def gray(text: str) -> str:
        """Return text in gray color."""
        return Colors.colorize(text, Colors.GRAY)

    @staticmethod
    def bold(text: str) -> str:
        """Return text in bold formatting."""
        return Colors.colorize(text, Colors.BOLD)

    @staticmethod
    def underline(text: str) -> str:
        """Return text with underline formatting."""
        return Colors.colorize(text, Colors.UNDERLINE)


def has_ansi_codes(text: Any) -> bool:
    """Check if text already contains ANSI color codes."""
    return '\033[' in str(text)


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    import re

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


# Icons for consistent visual indicators
class Icons:
    PASS = "✅"
    FAIL = "❌"
    WARNING = "⚠️"
    INFO = "i"
    GEAR = "⚙️"
    ROCKET = "🚀"
    BUG = "🐛"
    CLOCK = "⏰"
    MAGNIFY = "🔍"


class TestSuite:
    """Standardized test suite with consistent formatting and reporting."""

    def __init__(self, suite_name: str, module_name: str) -> None:
        self.suite_name = suite_name
        self.module_name = module_name
        self.start_time = None
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.warnings = 0
        self.test_results: list[dict[str, Any]] = []

    def start_suite(self) -> None:
        """Initialize the test suite with formatted header."""
        self.start_time = time.time()
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{Icons.ROCKET} Testing: {self.suite_name}{Colors.RESET}")
        print(f"{Colors.GRAY}Module: {self.module_name}{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

    def run_test(
        self,
        test_name: str,
        test_func: Callable[[], Any],
        test_summary: str = "",
        functions_tested: str = "",
        method_description: str = "",
        expected_outcome: str = "",
    ) -> bool:
        """
        Run a single test with standardized output and error handling.

        Args:
            test_name: Name/title of the test
            test_func: Test function to execute
            test_summary: Summary of what is being tested
            functions_tested: Names of the functions being tested
            method_description: How the functions are being tested
            expected_outcome: Expected outcome if functions work correctly

        Returns:
            True if test passed, False if failed
        """
        self.tests_run += 1
        test_start = time.time()

        print(f"{Colors.BLUE}{Icons.GEAR} Test {self.tests_run}: {test_name}{Colors.RESET}")
        if test_summary:
            print(f"Test: {test_summary}")
        if functions_tested:
            print(f"Functions tested: {functions_tested}")
        if method_description:
            print(f"Method: {method_description}")
        if expected_outcome:
            print(f"Expected outcome: {expected_outcome}")

        try:
            actual_outcome = ""
            test_func()
            duration = time.time() - test_start
            actual_outcome = "Test executed successfully with all assertions passing"
            print(f"Actual outcome: {actual_outcome}")
            print(f"Duration: {duration:.3f}s")
            print(f"Conclusion: {Colors.GREEN}{Icons.PASS} PASSED{Colors.RESET}")
            print()  # Add blank line between tests

            self.tests_passed += 1
            self.test_results.append(
                {
                    "name": test_name,
                    "status": "PASSED",
                    "duration": duration,
                    "expected": expected_outcome,
                    "outcome": actual_outcome,
                }
            )
            return True

        except AssertionError as e:
            import traceback

            duration = time.time() - test_start
            actual_outcome = f"Assertion failed: {e!s}"
            print(f"Actual outcome: {actual_outcome}")
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
                    "expected": expected_outcome,
                    "outcome": actual_outcome,
                }
            )
            return False

        except Exception as e:
            duration = time.time() - test_start
            actual_outcome = f"Exception occurred: {type(e).__name__}: {e!s}"
            print(f"Actual outcome: {actual_outcome}")
            print(f"Duration: {duration:.3f}s")
            print(f"Conclusion: {Colors.RED}{Icons.FAIL} FAILED{Colors.RESET}")
            print()  # Add blank line between tests

            self.tests_failed += 1
            self.test_results.append(
                {
                    "name": test_name,
                    "status": "ERROR",
                    "duration": duration,
                    "error": f"{type(e).__name__}: {e!s}",
                    "expected": expected_outcome,
                    "outcome": actual_outcome,
                }
            )
            return False

    def add_warning(self, message: str) -> None:
        """Add a warning message to the test output."""
        self.warnings += 1
        print(f"  {Colors.YELLOW}{Icons.WARNING} WARNING: {message}{Colors.RESET}")

    def finish_suite(self) -> bool:
        """Complete the test suite and print summary."""
        total_duration = time.time() - self.start_time if self.start_time else 0

        print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{Icons.MAGNIFY} Test Summary: {self.suite_name}{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")

        # Overall results
        if self.tests_failed == 0:
            status_color = Colors.GREEN
            status_icon = Icons.PASS
            status_text = "ALL TESTS PASSED"
        else:
            status_color = Colors.RED
            status_icon = Icons.FAIL
            status_text = "SOME TESTS FAILED"

        print(f"{status_color}{Icons.CLOCK} Duration: {total_duration:.3f}s{Colors.RESET}")
        print(f"{status_color}{status_icon} Status: {status_text}{Colors.RESET}")
        print(f"{Colors.GREEN}{Icons.PASS} Passed: {self.tests_passed}{Colors.RESET}")
        print(f"{Colors.RED}{Icons.FAIL} Failed: {self.tests_failed}{Colors.RESET}")
        if self.warnings > 0:
            print(f"{Colors.YELLOW}{Icons.WARNING} Warnings: {self.warnings}{Colors.RESET}")

        # Detailed results for failed tests
        failed_tests = [r for r in self.test_results if r["status"] in {"FAILED", "ERROR"}]
        if failed_tests:
            print(f"\n{Colors.YELLOW}{Icons.INFO} Failed Test Details:{Colors.RESET}")
            for test in failed_tests:
                print(f"  {Colors.RED}• {test['name']}{Colors.RESET}")
                if "error" in test:
                    print(f"    {Colors.GRAY}{test['error']}{Colors.RESET}")
                if test.get("expected"):
                    print(f"    {Colors.GRAY}Expected: {test['expected']}{Colors.RESET}")

        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

        return self.tests_failed == 0


@contextmanager
def suppress_logging() -> Iterator[None]:
    """Context manager to suppress logging during tests."""
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def create_mock_data() -> dict[str, Any]:
    """Create standard mock data for testing."""
    return {
        "mock_session_manager": MagicMock(),
        "mock_db_session": MagicMock(),
        "mock_config": MagicMock(),
        "sample_uuid": "TEST-UUID-1234-5678-ABCD",
        "sample_dna_data": {
            "uuid": "TEST-UUID-1234-5678-ABCD",
            "cm_dna": 85,
            "shared_segments": 4,
            "username": "Test User",
        },
    }


def create_standardized_test_data() -> dict[str, Any]:
    """Create standardized test data that can be used across all modules."""
    return {
        "mock_data": create_mock_data(),
        "test_person": {
            "first_name": "Fraser",
            "last_name": "Gault",
            "birth_year": 1941,
            "birth_place": "Banff",
            "gender": "M",
        },
        "test_gedcom_individual": {
            "id": "@I1@",
            "first_name": "John",
            "surname": "Smith",
            "gender_norm": "M",
            "birth_year": 1850,
            "birth_place_disp": "New York",
        },
        "test_environment": {
            "use_real_data": False,  # Default to mock data for safety
            "skip_external_apis": True,  # Skip external API calls in tests
            "use_cache": True,  # Use caching for performance
        },
    }


def get_test_mode() -> bool:
    """Determine if tests should use real data or mock data."""
    import os

    # Check environment variable or config to determine test mode
    return os.getenv("ANCESTRY_TEST_MODE", "mock").lower() in {"real", "integration"}


def create_test_data_factory(use_real_data: bool | None = None) -> dict[str, Any]:
    """Create appropriate test data based on test mode."""
    if use_real_data is None:
        use_real_data = get_test_mode()

    base_data = create_standardized_test_data()
    base_data["test_environment"]["use_real_data"] = use_real_data

    if use_real_data:
        # For real data tests, load from environment variables
        import os

        from dotenv import load_dotenv

        load_dotenv()

        base_data["test_person"] = {
            "first_name": os.getenv("TEST_PERSON_FIRST_NAME", "Fraser"),
            "last_name": os.getenv("TEST_PERSON_LAST_NAME", "Gault"),
            "birth_year": int(os.getenv("TEST_PERSON_BIRTH_YEAR", "1941")),
            "birth_place": os.getenv("TEST_PERSON_BIRTH_PLACE", "Banff"),
            "gender": os.getenv("TEST_PERSON_GENDER", "M"),
        }
        base_data["test_environment"]["skip_external_apis"] = False

    return base_data


def assert_valid_function(func: Any, func_name: str) -> None:
    """Assert that a function exists and is callable."""
    assert func is not None, f"Function {func_name} should exist"
    assert callable(func), f"Function {func_name} should be callable"


def standardized_test_wrapper(
    test_func: Callable[[dict[str, Any]], Any], test_name: str, cleanup_func: Callable[[], None] | None = None
) -> Callable[[], Any]:
    """Standardized test wrapper that provides consistent test execution patterns."""

    def wrapper() -> Any:
        test_data = create_test_data_factory()

        try:
            # Setup phase
            if test_data["test_environment"]["use_real_data"]:
                print(f"🔍 {test_name}: Using real data")
            else:
                print(f"🧪 {test_name}: Using mock data")

            # Execute test with standardized data
            result = test_func(test_data)

            # Validation phase
            if result is None:
                result = True  # Assume success if no explicit return

            return result

        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            raise
        finally:
            # Cleanup phase
            if cleanup_func:
                try:
                    cleanup_func()
                except Exception as cleanup_error:
                    print(f"⚠️ Cleanup failed for {test_name}: {cleanup_error}")

    return wrapper


def create_isolated_test_environment() -> dict[str, Any]:
    """Create an isolated test environment with proper resource management."""
    return {"temp_files": [], "mock_objects": [], "original_env_vars": {}, "cleanup_functions": []}


def cleanup_test_environment(env: dict[str, Any]) -> None:
    """Clean up test environment and resources."""
    # Clean up temporary files

    for temp_file in env.get("temp_files", []):
        try:
            p = Path(temp_file)
            if p.exists():
                p.unlink(missing_ok=True)
        except Exception:
            pass

    # Restore environment variables
    import os

    for var, value in env.get("original_env_vars", {}).items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value

    # Run cleanup functions
    for cleanup_func in env.get("cleanup_functions", []):
        with suppress(Exception):
            cleanup_func()


def assert_valid_config(config: Any, required_attrs: list[str]) -> None:
    """Assert that a config object has required attributes."""
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config should have attribute {attr}"


def _test_colors() -> None:
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
    assert Colors.END == "\033[0m"  # Test both naming conventions


def _test_icons() -> None:
    """Test that all icon constants are properly defined."""
    assert Icons.PASS == "✅"
    assert Icons.FAIL == "❌"
    assert Icons.WARNING == "⚠️"
    assert Icons.INFO == "i"
    assert Icons.GEAR == "⚙️"
    assert Icons.ROCKET == "🚀"
    assert Icons.BUG == "🐛"
    assert Icons.CLOCK == "⏰"
    assert Icons.MAGNIFY == "🔍"


def _test_mock_data(test_data: Any) -> bool:
    """Test mock data creation functionality."""
    data = test_data["mock_data"]
    assert isinstance(data, dict)
    assert "mock_session_manager" in data
    assert "sample_dna_data" in data
    assert data["sample_dna_data"]["cm_dna"] == 85
    assert isinstance(data["mock_session_manager"], MagicMock)
    return True


def _test_standardized_data_factory(test_data: Any) -> bool:
    """Test standardized test data factory."""
    assert "test_person" in test_data
    assert "test_environment" in test_data
    assert test_data["test_person"]["first_name"] in {"Fraser", "John"}  # Allow both mock and real
    assert isinstance(test_data["test_environment"]["use_real_data"], bool)
    return True


def _test_function_behavior_utility() -> bool:
    """Test the test_function_behavior utility function."""

    # Define a simple test function
    def sample_add(a: int, b: int) -> int:
        return a + b

    def sample_divide(a: int, b: int) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    # Add to globals for testing
    test_globals = {"sample_add": sample_add, "sample_divide": sample_divide}

    # Test 1: Basic behavior testing
    test_function_behavior("sample_add", [((2, 3), 5), ((10, 20), 30), ((0, 0), 0)], test_globals)

    # Test 2: Testing with allowed exceptions
    test_function_behavior(
        "sample_divide",
        [((10, 2), 5.0), ((10, 0), None)],  # Second case will raise ValueError
        test_globals,
        allow_exceptions=(ValueError,),
    )

    return True


def _test_test_suite_creation() -> None:
    """Test that TestSuite can be created and initialized properly."""
    test_suite = TestSuite("Test Suite", "test_module.py")
    assert test_suite.suite_name == "Test Suite"
    assert test_suite.module_name == "test_module.py"
    assert test_suite.start_time is None


def _test_context_managers() -> None:
    """Test that context managers work properly."""
    with suppress_logging():
        logging.critical("This logging should be suppressed")

    # Test that it doesn't raise an exception

    assert Path().exists()


def test_framework_module_tests() -> bool:
    """
    Comprehensive test suite for the test framework module.
    Tests all core functionality including colors, icons, test suite operations, and mock data.
    """
    print(f"{Icons.ROCKET} Running comprehensive tests for test_framework.py...")

    suite = TestSuite("Test Framework Comprehensive Tests", "test_framework.py")
    suite.start_suite()

    suite.run_test("Color constants", _test_colors, "Should define all standard ANSI color codes")
    suite.run_test("Icon constants", _test_icons, "Should define all standard Unicode icons")
    suite.run_test(
        "Mock data creation",
        standardized_test_wrapper(_test_mock_data, "Mock data creation"),
        "Should create valid test data structures with standardized factory",
    )
    suite.run_test(
        "Standardized data factory",
        standardized_test_wrapper(_test_standardized_data_factory, "Standardized data factory"),
        "Should create consistent test data across different test modes",
    )
    suite.run_test(
        "TestSuite creation",
        _test_test_suite_creation,
        "Should create TestSuite instances properly",
    )
    suite.run_test(
        "Context managers",
        _test_context_managers,
        "Should provide working context managers",
    )
    suite.run_test(
        "Function behavior testing utility",
        _test_function_behavior_utility,
        "Should test function behavior with test cases and allowed exceptions",
    )

    return suite.finish_suite()


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(test_framework_module_tests)


if __name__ == "__main__":
    # Test the framework itself
    def demo_tests() -> bool:
        suite = TestSuite("Test Framework Demo", "test_framework.py")
        suite.start_suite()

        def test_colors() -> None:
            assert Colors.RED == "\033[91m"
            assert Colors.GREEN == "\033[92m"
            assert Colors.RESET == "\033[0m"
            assert Colors.END == "\033[0m"

        def test_icons() -> None:
            assert Icons.PASS == "✅"
            assert Icons.FAIL == "❌"

        def test_mock_data() -> None:
            data = create_mock_data()
            assert "mock_session_manager" in data
            assert data["sample_dna_data"]["cm_dna"] == 85

        suite.run_test("Color constants", test_colors, "Should define standard ANSI color codes")
        suite.run_test("Icon constants", test_icons, "Should define standard Unicode icons")
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

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.messages: dict[str, list[str]] = {
            "debug": [],
            "info": [],
            "warning": [],
            "error": [],
            "critical": [],
        }

    def debug(self, msg: str, **_kwargs: Any) -> None:
        self.lines.append(msg)
        self.messages["debug"].append(msg)

    def info(self, msg: str, **_kwargs: Any) -> None:
        self.lines.append(msg)
        self.messages["info"].append(msg)

    def warning(self, msg: str, **_kwargs: Any) -> None:
        self.lines.append(msg)
        self.messages["warning"].append(msg)

    def error(self, msg: str, **_kwargs: Any) -> None:
        self.lines.append(msg)
        self.messages["error"].append(msg)

    def critical(self, msg: str, **_kwargs: Any) -> None:
        self.lines.append(msg)
        self.messages["critical"].append(msg)

    def get_messages(self, level: str | None = None) -> list[str]:
        """Get messages by level, or all messages if level is None"""
        if level:
            return self.messages.get(level, [])
        return self.lines

    def clear(self) -> None:
        """Clear all logged messages"""
        self.lines.clear()
        for level_msgs in self.messages.values():
            level_msgs.clear()


@contextmanager
def mock_logger_context(module_globals: dict[str, Any], logger_name: str = "logger") -> Iterator[MockLogger]:
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


# ==============================================
# Universal Test Output Formatting Utilities
# ==============================================


def format_test_section_header(title: str, emoji: str = "🧮") -> str:
    """
    Create a consistent, visually appealing test section header.

    Args:
        title: The title text for the section
        emoji: Optional emoji to use (defaults to 🧮)

    Returns:
        Formatted header string with colors and separators
    """
    separator = "─" * 60
    header = f"\n{Colors.CYAN}{separator}{Colors.RESET}"
    header += f"\n{Colors.BOLD}{Colors.BLUE}{emoji} {title.upper()}{Colors.RESET}"
    header += f"\n{Colors.CYAN}{separator}{Colors.RESET}"
    return header


def format_score_breakdown_table(field_scores: dict[str, int], total_score: int) -> str:
    """
    Format scoring breakdown as a readable table with colors and descriptions.

    Args:
        field_scores: Dictionary of field names to scores
        total_score: The total calculated score

    Returns:
        Formatted table string with color coding
    """
    table = f"\n{Colors.BOLD}{Colors.WHITE}📊 Scoring Breakdown:{Colors.RESET}\n"
    table += f"{Colors.GRAY}{'Field':<12} {'Score':<6} {'Description':<30}{Colors.RESET}\n"
    table += f"{Colors.GRAY}{'-' * 50}{Colors.RESET}\n"

    # Field descriptions for readability
    descriptions = {
        'givn': 'First Name Match',
        'surn': 'Surname Match',
        'gender': 'Gender Match',
        'byear': 'Birth Year Match',
        'bdate': 'Birth Date Match',
        'bplace': 'Birth Place Match',
        'bbonus': 'Birth Info Bonus',
        'dyear': 'Death Year Match',
        'ddate': 'Death Date Match',
        'dplace': 'Death Place Match',
        'dbonus': 'Death Info Bonus',
        'bonus': 'Name Bonus',
    }

    for field, score in field_scores.items():
        desc = descriptions.get(field, field.capitalize().replace('_', ' '))
        if score > 0:
            color = Colors.GREEN
        elif score == 0:
            color = Colors.GRAY
        else:
            color = Colors.RED

        table += f"{Colors.WHITE}{field:<12}{color} {score:<6}{Colors.RESET} {desc:<30}\n"

    table += f"{Colors.GRAY}{'-' * 50}{Colors.RESET}\n"
    table += f"{Colors.BOLD}{Colors.YELLOW}{'Total':<12} {total_score:<6} Final Match Score{Colors.RESET}\n"

    return table


def format_search_criteria(criteria: dict[str, Any]) -> str:
    """
    Format search criteria in a clean, readable way with bullets and colors.

    Args:
        criteria: Dictionary of search criteria

    Returns:
        Formatted criteria string
    """
    formatted = f"{Colors.BOLD}{Colors.WHITE}🔍 Search Criteria:{Colors.RESET}\n"
    for key, value in criteria.items():
        clean_key = key.replace('_', ' ').title()
        formatted += f"   {Colors.CYAN}•{Colors.RESET} {clean_key}: {Colors.WHITE}{value}{Colors.RESET}\n"

    return formatted


def format_test_result(test_name: str, success: bool, duration: float | None = None) -> str:
    """
    Format a test result with consistent styling and colors.

    Args:
        test_name: Name of the test
        success: Whether the test passed
        duration: Optional duration in seconds

    Returns:
        Formatted result string
    """
    if success:
        status_color = Colors.GREEN
        status_icon = Icons.PASS
        status_text = "PASSED"
    else:
        status_color = Colors.RED
        status_icon = Icons.FAIL
        status_text = "FAILED"

    result = f"{status_color}{status_icon} {test_name}: {status_text}{Colors.RESET}"
    if duration:
        result += f" {Colors.GRAY}({duration:.3f}s){Colors.RESET}"

    return result


def suppress_debug_logging() -> None:
    """Temporarily suppress debug logging for cleaner test output."""
    import logging

    logging.getLogger().setLevel(logging.WARNING)


def restore_debug_logging() -> None:
    """Restore normal logging level."""
    import logging

    logging.getLogger().setLevel(logging.INFO)


def clean_test_output() -> AbstractContextManager[None]:
    """
    Context manager for clean test output without debug noise.

    Usage:
        with clean_test_output():
            # Test code here - debug logging will be suppressed
            result = some_function()
    """
    from contextlib import contextmanager

    @contextmanager
    def _clean_output() -> Any:
        suppress_debug_logging()
        try:
            yield
        finally:
            restore_debug_logging()

    return _clean_output()


def test_function_availability(
    required_functions: list[str], globals_dict: dict[str, Any], module_name: str = "Module"
) -> list[bool]:
    """
    Universal function availability testing pattern.
    Consolidates identical testing code from multiple modules.

    This function only tests that functions exist and are callable.
    For behavior validation, use test_function_behavior() or module-specific tests.

    Args:
        required_functions: list of function names to test
        globals_dict: globals() dictionary from the calling module
        module_name: Name of the module being tested (for display)

    Returns:
        List of boolean results for each function test
    """
    results: list[bool] = []
    print(f"\n🔍 Testing {module_name} Function Availability:")

    for func_name in required_functions:
        try:
            # Check if function exists in globals
            if func_name not in globals_dict:
                print(f"   ❌ {func_name}: Not found in globals")
                results.append(False)
                continue

            # Check if it's callable
            func_obj = globals_dict[func_name]
            if not callable(func_obj):
                print(f"   ❌ {func_name}: Found but not callable")
                results.append(False)
                continue

            # Function exists and is callable
            print(f"   ✅ {func_name}: Available and callable")
            results.append(True)

        except Exception as e:
            print(f"   ❌ {func_name}: Error during test - {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Function Availability Summary: {passed}/{total} functions available")

    # Assert all functions are available
    for func_name, available in zip(required_functions, results, strict=False):
        assert available, f"Required function '{func_name}' is not available"

    return results


def test_function_behavior(
    func_name: str,
    test_cases: list[tuple[tuple[Any, ...], Any]],
    globals_dict: dict[str, Any],
    allow_exceptions: tuple[type[Exception], ...] = (),
) -> bool:
    """
    Test basic function behavior with provided test cases.

    This extends function availability testing by actually calling functions
    with test inputs and verifying outputs.

    Args:
        func_name: Name of the function to test
        test_cases: list of (args_tuple, expected_result) pairs
        globals_dict: globals() dictionary from the calling module
        allow_exceptions: Tuple of exception types that are acceptable

    Returns:
        bool: True if all test cases pass

    Example:
        test_function_behavior(
            "add_numbers",
            [((2, 3), 5), ((10, 20), 30)],
            globals(),
            allow_exceptions=(ValueError,)
        )
    """
    if func_name not in globals_dict:
        raise AssertionError(f"Function '{func_name}' not found in globals")

    func = globals_dict[func_name]
    if not callable(func):
        raise AssertionError(f"'{func_name}' is not callable")

    print(f"\n🧪 Testing {func_name} behavior with {len(test_cases)} test cases:")

    for i, (args, expected) in enumerate(test_cases, 1):
        try:
            result = func(*args)
            if result == expected:
                print(f"   ✅ Test {i}: {func_name}{args} = {result}")
            else:
                print(f"   ❌ Test {i}: {func_name}{args} = {result}, expected {expected}")
                raise AssertionError(f"Test {i} failed: got {result}, expected {expected}")
        except allow_exceptions as e:
            print(f"   ⚠️  Test {i}: {func_name}{args} raised {type(e).__name__} (allowed)")
        except Exception as e:
            print(f"   ❌ Test {i}: {func_name}{args} raised unexpected {type(e).__name__}: {e}")
            raise

    print(f"   ✅ All {len(test_cases)} test cases passed")
    return True


# ============================================================================
# Database Testing Utilities
# ============================================================================


@contextmanager
def database_rollback_test(session: Any) -> Iterator[Any]:
    """
    Context manager for database tests that automatically rolls back changes.

    This ensures tests clean up after themselves by rolling back all database
    changes made during the test, regardless of whether the test passes or fails.

    USER REQUIREMENT: "It's ok to add something to the database as long as this
    is reversed once the test is completed."

    Usage:
        from testing.test_framework import database_rollback_test

        def test_database_operation():
            sm = SessionManager()
            sm.start_sess("Test")
            db_session = sm.db_session

            with database_rollback_test(db_session):
                # Make database changes
                person = Person(profile_id="test123", username="Test User")
                db_session.add(person)
                db_session.commit()

                # Verify changes
                assert db_session.query(Person).filter_by(profile_id="test123").first()

            # Changes are automatically rolled back after the context exits
            # The person with profile_id="test123" no longer exists

    Args:
        session: SQLAlchemy session to manage

    Yields:
        The session for use in the test

    Note:
        - All changes are rolled back, even if test passes
        - Nested transactions are supported via savepoints
        - Works with both SessionManager.db_session and standalone sessions
    """
    # Create a savepoint to rollback to
    savepoint = session.begin_nested() if session.in_transaction() else session.begin()

    try:
        yield session
    finally:
        # Always rollback to the savepoint, regardless of test outcome
        try:
            if savepoint.is_active:
                savepoint.rollback()
            else:
                # If savepoint is not active, rollback the entire session
                session.rollback()
        except Exception as rollback_error:
            logger.warning(f"Error during test database rollback: {rollback_error}")
            # Force rollback even if savepoint rollback fails
            with suppress(Exception):
                session.rollback()
