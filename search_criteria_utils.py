#!/usr/bin/env python3
"""
Unified search criteria collection for genealogical research actions.

Provides consistent user interaction for both Action 10 (GEDCOM) and Action 11 (API)
to ensure identical search criteria collection and validation.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def get_unified_search_criteria(
    get_input_func: Optional[Callable[[str], str]] = None,
) -> Optional[dict[str, Any]]:
    """
    Collect unified search criteria from user input.

    This function provides consistent search criteria collection for both
    Action 10 (GEDCOM) and Action 11 (API) to ensure identical user experience.

    Args:
        get_input_func: Optional function to get user input (for testing).
                       If None, uses built-in input().

    Returns:
        Dictionary with standardized search criteria, or None if cancelled.
    """
    if get_input_func is None:
        get_input_func = input

    print("--- Search Criteria ---\n")

    # Collect basic criteria
    first_name = _sanitize_input(get_input_func("  First Name Contains: "))
    surname = _sanitize_input(get_input_func("  Surname Contains: "))

    # Validate required fields
    if not (first_name or surname):
        logger.warning("Search requires First Name or Surname. Search cancelled.")
        print("\nSearch requires First Name or Surname. Search cancelled.")
        return None

    # Gender
    gender_input = _sanitize_input(get_input_func("  Gender (M/F): "))
    gender = _parse_gender_input(gender_input) if gender_input else None

    # Birth year
    birth_year = _parse_year_input(get_input_func("  Birth Year (YYYY): "))

    # Birth place
    birth_place = _sanitize_input(get_input_func("  Birth Place Contains: "))

    # Death year (optional)
    death_year = _parse_year_input(get_input_func("  Death Year (YYYY) [Optional]: "))

    # Death place (optional)
    death_place = _sanitize_input(get_input_func("  Death Place Contains [Optional]: "))

    # Create date objects
    birth_date_obj = _create_date_object(birth_year, "birth")
    death_date_obj = _create_date_object(death_year, "death")

    # Build standardized criteria dictionary
    criteria = {
        "first_name": first_name,
        "surname": surname,
        "gender": gender,
        "birth_year": birth_year,
        "birth_date_obj": birth_date_obj,
        "birth_place": birth_place,
        "death_year": death_year,
        "death_date_obj": death_date_obj,
        "death_place": death_place,
    }

    # Log criteria
    logger.debug("\n--- Search Criteria Collected ---")
    for key, value in criteria.items():
        if value is not None and key not in ["birth_date_obj", "death_date_obj"]:
            logger.debug(f"  {key.replace('_', ' ').title()}: {value}")

    return criteria


def _sanitize_input(value: str) -> Optional[str]:
    """Sanitize user input by stripping whitespace."""
    if not value:
        return None
    sanitized = value.strip()
    return sanitized if sanitized else None


def _parse_year_input(year_str: str) -> Optional[int]:
    """Parse year input string to integer."""
    return int(year_str) if year_str.strip().isdigit() else None


def _create_date_object(year: Optional[int], date_type: str) -> Optional[datetime]:
    """Create datetime object from year, with error handling."""
    if not year:
        return None
    try:
        return datetime(year, 1, 1, tzinfo=timezone.utc)
    except ValueError:
        logger.warning(f"Cannot create date object for {date_type} year {year}.")
        return None


def _parse_gender_input(gender_input: str) -> Optional[str]:
    """Parse gender input to standardized format."""
    if gender_input and gender_input[0].lower() in ["m", "f"]:
        return gender_input[0].lower()
    return None


def _format_years_display(birth_year: Optional[int], death_year: Optional[int]) -> str:
    """Format birth and death years for display."""
    if birth_year and death_year:
        return f" ({birth_year}-{death_year})"
    if birth_year:
        return f" (b. {birth_year})"
    if death_year:
        return f" (d. {death_year})"
    return ""


def _print_section_header(label: str, is_first: bool) -> None:
    """Print section header with appropriate spacing."""
    if is_first:
        print(f"{label}:")
    else:
        print(f"\n{label}:")


def _print_family_member(member: dict) -> None:
    """Print a single family member's information."""
    name = member.get("name", "Unknown")
    birth_year = member.get("birth_year")
    death_year = member.get("death_year")
    years_display = _format_years_display(birth_year, death_year)
    print(f"   - {name}{years_display}")


def display_family_members(
    family_data: dict[str, list],
    relation_labels: Optional[dict[str, str]] = None
) -> None:
    """
    Display family members in a consistent format for both Action 10 and Action 11.

    Args:
        family_data: Dictionary with keys like 'parents', 'siblings', 'spouses', 'children'
                    and values as lists of family member dictionaries
        relation_labels: Optional custom labels for each relation type
    """
    if relation_labels is None:
        relation_labels = {
            "parents": "📋 Parents",
            "siblings": "📋 Siblings",
            "spouses": "💕 Spouses",
            "children": "👶 Children",
        }

    first_section = True
    for relation_key, label in relation_labels.items():
        members = family_data.get(relation_key, [])

        _print_section_header(label, first_section)
        first_section = False

        if not members:
            print("   - None found")
            continue

        for member in members:
            if member:
                _print_family_member(member)


# === TESTS ===
def _test_sanitize_input() -> None:
    """Test input sanitization."""
    # Test with normal input
    assert _sanitize_input("John") == "John", "Should return trimmed input"

    # Test with whitespace
    assert _sanitize_input("  John  ") == "John", "Should strip whitespace"

    # Test with empty string
    assert _sanitize_input("") is None, "Should return None for empty string"

    # Test with only whitespace
    assert _sanitize_input("   ") is None, "Should return None for whitespace-only"

    # Test with None-like input
    assert _sanitize_input("") is None, "Should handle empty input"


def _test_parse_year_input() -> None:
    """Test year parsing."""
    # Test valid year
    assert _parse_year_input("1950") == 1950, "Should parse valid year"

    # Test with whitespace
    assert _parse_year_input("  1950  ") == 1950, "Should handle whitespace"

    # Test invalid input
    assert _parse_year_input("abc") is None, "Should return None for non-numeric"
    assert _parse_year_input("") is None, "Should return None for empty"
    assert _parse_year_input("19.50") is None, "Should return None for decimal"


def _test_parse_gender_input() -> None:
    """Test gender parsing."""
    # Test valid inputs
    assert _parse_gender_input("M") == "m", "Should parse M to m"
    assert _parse_gender_input("F") == "f", "Should parse F to f"
    assert _parse_gender_input("m") == "m", "Should parse lowercase m"
    assert _parse_gender_input("f") == "f", "Should parse lowercase f"
    assert _parse_gender_input("Male") == "m", "Should parse Male to m"
    assert _parse_gender_input("Female") == "f", "Should parse Female to f"

    # Test invalid inputs
    assert _parse_gender_input("X") is None, "Should return None for invalid"
    assert _parse_gender_input("") is None, "Should return None for empty"


def _test_create_date_object() -> None:
    """Test date object creation."""
    # Test valid year
    date_obj = _create_date_object(1950, "birth")
    assert date_obj is not None, "Should create date object"
    assert date_obj.year == 1950, "Should have correct year"
    assert date_obj.month == 1, "Should default to January"
    assert date_obj.day == 1, "Should default to day 1"

    # Test None year
    assert _create_date_object(None, "birth") is None, "Should return None for None year"

    # Test invalid year (too large)
    assert _create_date_object(99999, "birth") is None, "Should return None for invalid year"


def _test_format_years_display() -> None:
    """Test years display formatting."""
    # Test both years
    assert _format_years_display(1950, 2020) == " (1950-2020)", "Should format both years"

    # Test birth year only
    assert _format_years_display(1950, None) == " (b. 1950)", "Should format birth year"

    # Test death year only
    assert _format_years_display(None, 2020) == " (d. 2020)", "Should format death year"

    # Test no years
    assert _format_years_display(None, None) == "", "Should return empty for no years"


def _test_get_unified_search_criteria_valid() -> None:
    """Test unified search criteria collection with valid input."""
    # Mock input function
    inputs = ["John", "Smith", "M", "1950", "London", "2020", "Manchester"]
    input_iter = iter(inputs)

    def mock_input(prompt: str) -> str:
        _ = prompt  # Acknowledge prompt parameter
        return next(input_iter)

    criteria = get_unified_search_criteria(mock_input)

    assert criteria is not None, "Should return criteria"
    assert criteria["first_name"] == "John", "Should have first name"
    assert criteria["surname"] == "Smith", "Should have surname"
    assert criteria["gender"] == "m", "Should have gender"
    assert criteria["birth_year"] == 1950, "Should have birth year"
    assert criteria["birth_place"] == "London", "Should have birth place"
    assert criteria["death_year"] == 2020, "Should have death year"
    assert criteria["death_place"] == "Manchester", "Should have death place"
    assert criteria["birth_date_obj"] is not None, "Should have birth date object"
    assert criteria["death_date_obj"] is not None, "Should have death date object"


def _test_get_unified_search_criteria_minimal() -> None:
    """Test unified search criteria with minimal input."""
    # Only provide first name
    inputs = ["John", "", "", "", "", "", ""]
    input_iter = iter(inputs)

    def mock_input(prompt: str) -> str:
        _ = prompt  # Acknowledge prompt parameter
        return next(input_iter)

    criteria = get_unified_search_criteria(mock_input)

    assert criteria is not None, "Should return criteria"
    assert criteria["first_name"] == "John", "Should have first name"
    assert criteria["surname"] is None, "Should have None surname"
    assert criteria["gender"] is None, "Should have None gender"
    assert criteria["birth_year"] is None, "Should have None birth year"


def _test_get_unified_search_criteria_cancelled() -> None:
    """Test unified search criteria when cancelled (no name provided)."""
    # Provide no name
    inputs = ["", "", "", "", "", "", ""]
    input_iter = iter(inputs)

    def mock_input(prompt: str) -> str:
        _ = prompt  # Acknowledge prompt parameter
        return next(input_iter)

    criteria = get_unified_search_criteria(mock_input)

    assert criteria is None, "Should return None when no name provided"


def _test_display_family_members() -> None:
    """Test family members display."""
    # Test with full family data
    family_data = {
        "parents": [
            {"name": "John Smith", "birth_year": 1920, "death_year": 1990},
            {"name": "Jane Smith", "birth_year": 1925, "death_year": 1995}
        ],
        "siblings": [
            {"name": "Bob Smith", "birth_year": 1950, "death_year": None}
        ],
        "spouses": [
            {"name": "Mary Jones", "birth_year": 1952, "death_year": None}
        ],
        "children": [
            {"name": "Alice Smith", "birth_year": 1975, "death_year": None},
            {"name": "Charlie Smith", "birth_year": 1978, "death_year": None}
        ]
    }

    # Should not raise exception
    display_family_members(family_data)

    # Test with empty family data
    display_family_members({})

    # Test with custom labels
    custom_labels = {
        "parents": "Parents",
        "siblings": "Siblings",
        "spouses": "Spouses",
        "children": "Children"
    }
    display_family_members(family_data, custom_labels)


def _test_print_functions() -> None:
    """Test print helper functions."""
    # Test section header
    _print_section_header("Test Section", True)
    _print_section_header("Test Section", False)

    # Test family member printing
    member = {"name": "John Smith", "birth_year": 1950, "death_year": 2020}
    _print_family_member(member)

    # Test with minimal data
    member = {"name": "Jane Doe"}
    _print_family_member(member)

    # Test with birth year only
    member = {"name": "Bob Jones", "birth_year": 1960}
    _print_family_member(member)


def search_criteria_utils_module_tests() -> bool:
    """Comprehensive test suite for search_criteria_utils.py"""
    from test_framework import TestSuite

    suite = TestSuite("Search Criteria Utils", "search_criteria_utils.py")
    suite.start_suite()

    # Category 1: Input Sanitization Tests
    suite.run_test(
        "Input sanitization",
        _test_sanitize_input,
        "Sanitizes user input correctly",
        "_sanitize_input()",
        "Tests whitespace trimming and empty string handling"
    )

    suite.run_test(
        "Year parsing",
        _test_parse_year_input,
        "Parses year input correctly",
        "_parse_year_input()",
        "Tests valid years, whitespace, and invalid inputs"
    )

    suite.run_test(
        "Gender parsing",
        _test_parse_gender_input,
        "Parses gender input correctly",
        "_parse_gender_input()",
        "Tests M/F parsing and invalid inputs"
    )

    # Category 2: Date Handling Tests
    suite.run_test(
        "Date object creation",
        _test_create_date_object,
        "Creates date objects correctly",
        "_create_date_object()",
        "Tests valid years, None, and invalid years"
    )

    suite.run_test(
        "Years display formatting",
        _test_format_years_display,
        "Formats years for display",
        "_format_years_display()",
        "Tests birth/death year combinations"
    )

    # Category 3: Search Criteria Collection Tests
    suite.run_test(
        "Unified search criteria - valid input",
        _test_get_unified_search_criteria_valid,
        "Collects complete search criteria",
        "get_unified_search_criteria()",
        "Tests full criteria collection with all fields"
    )

    suite.run_test(
        "Unified search criteria - minimal input",
        _test_get_unified_search_criteria_minimal,
        "Handles minimal input",
        "get_unified_search_criteria()",
        "Tests criteria collection with only required fields"
    )

    suite.run_test(
        "Unified search criteria - cancelled",
        _test_get_unified_search_criteria_cancelled,
        "Returns None when cancelled",
        "get_unified_search_criteria()",
        "Tests cancellation when no name provided"
    )

    # Category 4: Display Tests
    suite.run_test(
        "Family members display",
        _test_display_family_members,
        "Displays family members correctly",
        "display_family_members()",
        "Tests display with full data, empty data, and custom labels"
    )

    suite.run_test(
        "Print helper functions",
        _test_print_functions,
        "Print functions don't crash",
        "_print_section_header(), _print_family_member()",
        "Tests all print helper functions"
    )

    return suite.finish_suite()


# Create standard test runner
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(search_criteria_utils_module_tests)

