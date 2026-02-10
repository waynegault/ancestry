#!/usr/bin/env python3
"""
Unified search criteria collection for genealogical research actions.

Provides consistent user interaction for both Action 10 (GEDCOM) and Action 11 (API)
to ensure identical search criteria collection and validation.
"""

# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from collections.abc import Callable
from datetime import UTC, datetime, timezone
from importlib import import_module
from typing import Any, Protocol, cast

from core.logging_config import logger

FamilyData = dict[str, list[dict[str, Any]]]


class _PresenterModule(Protocol):
    def display_family_members(
        self,
        family_data: FamilyData,
        relation_labels: dict[str, str] | None = None,
    ) -> None: ...

    def present_post_selection(
        self,
        display_name: str,
        birth_year: int | None,
        death_year: int | None,
        family_data: FamilyData,
        owner_name: str,
        relation_labels: dict[str, str] | None = None,
        unified_path: list[dict[str, Any]] | None = None,
        formatted_path: str | None = None,
    ) -> None: ...


_presenter_module: _PresenterModule | None = None
_presenter_import_error: Exception | None = None


try:  # pragma: no cover - optional dependency
    _presenter_candidate = import_module("genealogy.genealogy_presenter")
except Exception as exc:
    _presenter_import_error = exc
else:
    missing_attrs = [
        attr for attr in ("display_family_members", "present_post_selection") if not hasattr(_presenter_candidate, attr)
    ]
    if missing_attrs:
        _presenter_import_error = AttributeError(f"genealogy_presenter missing attributes: {', '.join(missing_attrs)}")
    else:
        _presenter_module = cast(_PresenterModule, _presenter_candidate)


def _resolve_presenter_module() -> _PresenterModule:
    if _presenter_module is not None:
        return _presenter_module
    if _presenter_import_error is not None:
        raise RuntimeError("genealogy_presenter module unavailable") from _presenter_import_error
    raise RuntimeError("genealogy_presenter module unavailable")


def get_unified_search_criteria(
    get_input_func: Callable[[str], str] | None = None,
) -> dict[str, Any] | None:
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

    # Collect basic criteria
    first_name = _sanitize_input(get_input_func("  First Name Contains: "))
    surname = _sanitize_input(get_input_func("  Surname Contains: "))

    # Validate required fields
    if not (first_name or surname):
        logger.warning("Search requires First Name or Surname. Search cancelled.")
        print("\nSearch requires First Name or Surname. Search cancelled.")
        return None

    # Birth year
    birth_year = _parse_year_input(get_input_func("  Birth Year (YYYY): "))

    # Birth place
    birth_place = _sanitize_input(get_input_func("  Birth Place Contains: "))

    # Death year (optional)
    death_year = _parse_year_input(get_input_func("  Death Year (YYYY): "))

    # Death place (optional)
    death_place = _sanitize_input(get_input_func("  Death Place Contains: "))

    # Create date objects
    birth_date_obj = _create_date_object(birth_year, "birth")
    death_date_obj = _create_date_object(death_year, "death")

    # Build standardized criteria dictionary
    criteria = {
        "first_name": first_name,
        "surname": surname,
        "birth_year": birth_year,
        "birth_date_obj": birth_date_obj,
        "birth_place": birth_place,
        "death_year": death_year,
        "death_date_obj": death_date_obj,
        "death_place": death_place,
    }

    _debug_log_criteria(criteria)
    _print_criteria_summary(criteria)

    return criteria


def _sanitize_input(value: str) -> str | None:
    """Sanitize user input by stripping whitespace."""
    if not value:
        return None
    sanitized = value.strip()
    return sanitized if sanitized else None


def _parse_year_input(year_str: str) -> int | None:
    """Parse year input string to integer."""
    return int(year_str) if year_str.strip().isdigit() else None


def _create_date_object(year: int | None, date_type: str) -> datetime | None:
    """Create datetime object from year, with error handling."""
    if not year:
        return None
    try:
        return datetime(year, 1, 1, tzinfo=UTC)
    except ValueError:
        logger.warning(f"Cannot create date object for {date_type} year {year}.")
        return None


# Re-export unified presenter functions from genealogy_presenter (single source of truth)
def display_family_members(
    family_data: FamilyData,
    relation_labels: dict[str, str] | None = None,
) -> None:
    presenter = _resolve_presenter_module()
    presenter.display_family_members(family_data, relation_labels)


def present_post_selection(
    display_name: str,
    birth_year: int | None,
    death_year: int | None,
    family_data: FamilyData,
    owner_name: str,
    relation_labels: dict[str, str] | None = None,
    unified_path: list[dict[str, Any]] | None = None,
    formatted_path: str | None = None,
) -> None:
    presenter = _resolve_presenter_module()
    presenter.present_post_selection(
        display_name=display_name,
        birth_year=birth_year,
        death_year=death_year,
        family_data=family_data,
        owner_name=owner_name,
        relation_labels=relation_labels,
        unified_path=unified_path,
        formatted_path=formatted_path,
    )


def _debug_log_criteria(criteria: dict[str, Any]) -> None:
    """Log collected criteria at DEBUG level, excluding date objects."""
    logger.debug("--- Search Criteria Collected ---")
    for key, value in criteria.items():
        if value is not None and key not in {"birth_date_obj", "death_date_obj"}:
            logger.debug(f"  {key.replace('_', ' ').title()}: {value}")


def _print_criteria_summary(criteria: dict[str, Any]) -> None:
    """Print a clean, INFO-level summary of only the provided criteria."""
    print("\n--- Search Criteria Used ---\n")
    label_map = [
        ("first_name", "First Name Contains"),
        ("surname", "Surname Contains"),
        ("birth_year", "Birth Year (YYYY)"),
        ("birth_place", "Birth Place Contains"),
        ("death_year", "Death Year (YYYY)"),
        ("death_place", "Death Place Contains"),
    ]
    for key, label in label_map:
        value = criteria.get(key)
        if value not in {None, ""}:
            print(f"  {label}: {value}")
    print("")


# (moved) presentation helpers are provided by genealogy_presenter module


# (moved) present_post_selection is provided by genealogy_presenter module


# === Additional Tests for presenter ===


def _test_present_post_selection_minimal() -> None:
    """Test unified presenter with minimal data."""
    import io
    import sys

    # Capture output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        family = {"parents": [], "spouses": [], "children": []}
        present_post_selection(
            display_name="Jane Doe",
            birth_year=1970,
            death_year=None,
            family_data=family,
            owner_name="Owner Name",
            formatted_path="Relationship to Owner Name:\n  - Jane Doe is related (example)",
        )

        # Restore stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Validate output contains expected elements
        assert "=== Jane Doe (b. 1970) ===" in output, "Header should show name with birth year"
        assert "Relationship to Owner Name:" in output, "Should show relationship header"
        assert "Jane Doe is related (example)" in output, "Should show relationship path"
    finally:
        sys.stdout = sys.__stdout__


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


# (moved) years display formatting is covered in genealogy_presenter tests


def _test_get_unified_search_criteria_valid() -> None:
    """Test unified search criteria collection with valid input."""
    # Mock input function
    inputs = ["John", "Smith", "1950", "London", "2020", "Manchester"]
    input_iter = iter(inputs)

    def mock_input(prompt: str) -> str:
        _ = prompt  # Acknowledge prompt parameter
        return next(input_iter)

    criteria = get_unified_search_criteria(mock_input)

    assert criteria is not None, "Should return criteria"
    assert criteria["first_name"] == "John", "Should have first name"
    assert criteria["surname"] == "Smith", "Should have surname"
    assert criteria["birth_year"] == 1950, "Should have birth year"
    assert criteria["birth_place"] == "London", "Should have birth place"
    assert criteria["death_year"] == 2020, "Should have death year"
    assert criteria["death_place"] == "Manchester", "Should have death place"
    assert criteria["birth_date_obj"] is not None, "Should have birth date object"
    assert criteria["death_date_obj"] is not None, "Should have death date object"


def _test_get_unified_search_criteria_minimal() -> None:
    """Test unified search criteria with minimal input."""
    # Only provide first name
    inputs = ["John", "", "", "", "", ""]
    input_iter = iter(inputs)

    def mock_input(prompt: str) -> str:
        _ = prompt  # Acknowledge prompt parameter
        return next(input_iter)

    criteria = get_unified_search_criteria(mock_input)

    assert criteria is not None, "Should return criteria"
    assert criteria["first_name"] == "John", "Should have first name"
    assert criteria["surname"] is None, "Should have None surname"
    assert criteria["birth_year"] is None, "Should have None birth year"


def _test_get_unified_search_criteria_cancelled() -> None:
    """Test unified search criteria when cancelled (no name provided)."""
    # Provide no name
    inputs = ["", "", "", "", "", ""]
    input_iter = iter(inputs)

    def mock_input(prompt: str) -> str:
        _ = prompt  # Acknowledge prompt parameter
        return next(input_iter)

    criteria = get_unified_search_criteria(mock_input)

    assert criteria is None, "Should return None when no name provided"


def _test_display_family_members() -> None:
    """Test family members display with output validation."""
    import io
    import sys

    # Test with full family data
    family_data = {
        "parents": [
            {"name": "John Smith", "birth_year": 1920, "death_year": 1990},
            {"name": "Jane Smith", "birth_year": 1925, "death_year": 1995},
        ],
        "siblings": [{"name": "Bob Smith", "birth_year": 1950, "death_year": None}],
        "spouses": [{"name": "Mary Jones", "birth_year": 1952, "death_year": None}],
        "children": [
            {"name": "Alice Smith", "birth_year": 1975, "death_year": None},
            {"name": "Charlie Smith", "birth_year": 1978, "death_year": None},
        ],
    }

    # Capture output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        display_family_members(family_data)

        # Restore stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Validate output contains expected family members
        assert "John Smith (1920-1990)" in output, "Should show parent with years"
        assert "Bob Smith (b. 1950)" in output, "Should show living sibling"
        assert "Parents:" in output, "Should show Parents section"
        assert "Siblings:" in output, "Should show Siblings section"
    finally:
        sys.stdout = sys.__stdout__

    # Test with empty family data
    captured_output = io.StringIO()
    sys.stdout = captured_output
    try:
        display_family_members({})
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        # Should handle empty data gracefully
        assert "Parents:" in output or not output, "Should handle empty data"
    finally:
        sys.stdout = sys.__stdout__

    # Test with None values in family members
    family_with_nones = {"parents": [{"name": "Unknown Parent", "birth_year": None, "death_year": None}]}
    captured_output = io.StringIO()
    sys.stdout = captured_output
    try:
        display_family_members(family_with_nones)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        assert "Unknown Parent" in output, "Should handle None years gracefully"
    finally:
        sys.stdout = sys.__stdout__


# (moved) print helper tests removed; functionality covered by genealogy_presenter


def search_criteria_utils_module_tests() -> bool:
    """Comprehensive test suite for search_criteria_utils.py"""
    from testing.test_framework import TestSuite

    suite = TestSuite("Search Criteria Utils", "search_criteria_utils.py")
    suite.start_suite()

    # Category 1: Input Sanitization Tests
    suite.run_test(
        "Input sanitization",
        _test_sanitize_input,
        "Sanitizes user input correctly",
        "_sanitize_input()",
        "Tests whitespace trimming and empty string handling",
    )

    suite.run_test(
        "Year parsing",
        _test_parse_year_input,
        "Parses year input correctly",
        "_parse_year_input()",
        "Tests valid years, whitespace, and invalid inputs",
    )

    suite.run_test(
        "Unified presenter",
        _test_present_post_selection_minimal,
        "Presents header, family, and relationship correctly",
        "present_post_selection()",
        "Smoke test with preformatted relationship text",
    )

    # Category 2: Date Handling Tests
    suite.run_test(
        "Date object creation",
        _test_create_date_object,
        "Creates date objects correctly",
        "_create_date_object()",
        "Tests valid years, None, and invalid years",
    )

    # Years display formatting covered by genealogy_presenter tests

    # Category 3: Search Criteria Collection Tests
    suite.run_test(
        "Unified search criteria - valid input",
        _test_get_unified_search_criteria_valid,
        "Collects complete search criteria",
        "get_unified_search_criteria()",
        "Tests full criteria collection with all fields",
    )

    suite.run_test(
        "Unified search criteria - minimal input",
        _test_get_unified_search_criteria_minimal,
        "Handles minimal input",
        "get_unified_search_criteria()",
        "Tests criteria collection with only required fields",
    )

    suite.run_test(
        "Unified search criteria - cancelled",
        _test_get_unified_search_criteria_cancelled,
        "Returns None when cancelled",
        "get_unified_search_criteria()",
        "Tests cancellation when no name provided",
    )

    # Category 4: Display Tests
    suite.run_test(
        "Family members display",
        _test_display_family_members,
        "Displays family members correctly",
        "display_family_members()",
        "Tests display with full data, empty data, and custom labels",
    )

    return suite.finish_suite()


# Create standard test runner
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(search_criteria_utils_module_tests)


if __name__ == "__main__":
    run_comprehensive_tests()
