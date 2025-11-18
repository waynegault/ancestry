"""
Example: Basic Test Pattern

Demonstrates simple function testing with clear assertions.
"""

import sys
from pathlib import Path

# Ensure repository root is importable when run in isolation
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def add_numbers(a: int, b: int) -> int:
    """Example function: adds two numbers."""
    return a + b


def is_even(n: int) -> bool:
    """Example function: checks if number is even."""
    return n % 2 == 0


def reverse_string(s: str) -> str:
    """Example function: reverses a string."""
    return s[::-1]


# ============================================================================
# TESTS
# ============================================================================


def _test_add_positive_numbers() -> bool:
    """Test adding two positive numbers."""
    result = add_numbers(2, 3)
    return result == 5


def _test_add_negative_numbers() -> bool:
    """Test adding negative numbers."""
    result = add_numbers(-5, -3)
    return result == -8


def _test_add_zero() -> bool:
    """Test adding zero."""
    result = add_numbers(10, 0)
    return result == 10


def _test_is_even_with_even_number() -> bool:
    """Test is_even with even input."""
    return is_even(4) is True


def _test_is_even_with_odd_number() -> bool:
    """Test is_even with odd input."""
    return is_even(7) is False


def _test_is_even_with_zero() -> bool:
    """Test is_even with zero (edge case)."""
    return is_even(0) is True


def _test_reverse_string_normal() -> bool:
    """Test reversing a normal string."""
    result = reverse_string("hello")
    return result == "olleh"


def _test_reverse_string_empty() -> bool:
    """Test reversing empty string (edge case)."""
    result = reverse_string("")
    return not result


def _test_reverse_string_single_char() -> bool:
    """Test reversing single character (edge case)."""
    result = reverse_string("a")
    return result == "a"


def module_tests() -> bool:
    """Test suite for basic examples."""
    # Import here to avoid circular dependencies
    from test_framework import TestSuite

    suite = TestSuite("Basic Test Examples", "example_basic_test.py")

    suite.start_suite()

    # Test add_numbers
    suite.run_test("Add positive numbers", _test_add_positive_numbers)
    suite.run_test("Add negative numbers", _test_add_negative_numbers)
    suite.run_test("Add with zero", _test_add_zero)

    # Test is_even
    suite.run_test("Even number returns True", _test_is_even_with_even_number)
    suite.run_test("Odd number returns False", _test_is_even_with_odd_number)
    suite.run_test("Zero is even", _test_is_even_with_zero)

    # Test reverse_string
    suite.run_test("Reverse normal string", _test_reverse_string_normal)
    suite.run_test("Reverse empty string", _test_reverse_string_empty)
    suite.run_test("Reverse single character", _test_reverse_string_single_char)

    return suite.finish_suite()


if __name__ == "__main__":
    # Import here to avoid circular dependencies
    from test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
