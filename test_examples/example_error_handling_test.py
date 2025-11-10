"""
Example: Error Handling Test Pattern

Demonstrates testing exception handling and error conditions.
"""

import sys


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_age(age: int) -> bool:
    """Validates age is within reasonable bounds."""
    if age < 0:
        raise ValidationError("Age cannot be negative")
    if age > 150:
        raise ValidationError("Age cannot exceed 150")
    return True


def divide_numbers(a: float, b: float) -> float:
    """Divides two numbers with error handling."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def parse_year(year_str: str) -> int:
    """Parses year string to integer."""
    if not year_str:
        raise ValueError("Year string cannot be empty")

    try:
        year = int(year_str)
    except ValueError:
        raise ValueError(f"Invalid year format: {year_str}")

    if year < 1000 or year > 9999:
        raise ValueError(f"Year must be 4 digits: {year}")

    return year


# ============================================================================
# TESTS
# ============================================================================


def _test_validate_age_valid() -> bool:
    """Test that valid age passes validation."""
    try:
        result = validate_age(30)
        return result is True
    except Exception:
        return False


def _test_validate_age_negative() -> bool:
    """Test that negative age raises ValidationError."""
    try:
        validate_age(-5)
        return False  # Should have raised exception
    except ValidationError as e:
        return "negative" in str(e).lower()
    except Exception:
        return False  # Wrong exception type


def _test_validate_age_too_high() -> bool:
    """Test that age over 150 raises ValidationError."""
    try:
        validate_age(200)
        return False  # Should have raised exception
    except ValidationError as e:
        return "150" in str(e)
    except Exception:
        return False


def _test_divide_normal() -> bool:
    """Test normal division operation."""
    result = divide_numbers(10, 2)
    return result == 5.0


def _test_divide_by_zero() -> bool:
    """Test that dividing by zero raises ZeroDivisionError."""
    try:
        divide_numbers(10, 0)
        return False  # Should have raised exception
    except ZeroDivisionError:
        return True
    except Exception:
        return False


def _test_parse_year_valid() -> bool:
    """Test parsing valid year string."""
    result = parse_year("1995")
    return result == 1995


def _test_parse_year_empty() -> bool:
    """Test that empty string raises ValueError."""
    try:
        parse_year("")
        return False
    except ValueError as e:
        return "empty" in str(e).lower()
    except Exception:
        return False


def _test_parse_year_invalid_format() -> bool:
    """Test that non-numeric string raises ValueError."""
    try:
        parse_year("not a year")
        return False
    except ValueError as e:
        has_error_message = "invalid" in str(e).lower() or "format" in str(e).lower()
        return has_error_message
    except Exception:
        return False


def _test_parse_year_out_of_range() -> bool:
    """Test that 3-digit year raises ValueError."""
    try:
        parse_year("999")
        return False
    except ValueError as e:
        return "digit" in str(e).lower() or "1000" in str(e)
    except Exception:
        return False


def _test_exception_message_content() -> bool:
    """Test that exception messages contain helpful information."""
    try:
        validate_age(-10)
        return False
    except ValidationError as e:
        message = str(e)
        has_context = "negative" in message.lower()
        has_helpful_text = len(message) > 5  # Not just empty message
        return has_context and has_helpful_text


def module_tests() -> bool:
    """Test suite for error handling examples."""
    # Import here to avoid circular dependencies
    from test_framework import TestSuite

    suite = TestSuite("Error Handling Examples", "example_error_handling_test.py")

    suite.start_suite()

    # Test validate_age
    suite.run_test("Valid age passes", _test_validate_age_valid)
    suite.run_test("Negative age raises error", _test_validate_age_negative)
    suite.run_test("Age over 150 raises error", _test_validate_age_too_high)

    # Test divide_numbers
    suite.run_test("Normal division works", _test_divide_normal)
    suite.run_test("Division by zero raises error", _test_divide_by_zero)

    # Test parse_year
    suite.run_test("Parse valid year", _test_parse_year_valid)
    suite.run_test("Empty string raises error", _test_parse_year_empty)
    suite.run_test("Invalid format raises error", _test_parse_year_invalid_format)
    suite.run_test("Out of range year raises error", _test_parse_year_out_of_range)

    # Test error messages
    suite.run_test("Exception messages are helpful", _test_exception_message_content)

    return suite.finish_suite()


if __name__ == "__main__":
    # Import here to avoid circular dependencies
    from test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
