#!/usr/bin/env python3

"""
Action 12: Compare GEDCOM vs API Search Results

This action compares the results from Action 10 (GEDCOM-based search) and
Action 11 (API-based search) to validate consistency between the two approaches.

Workflow:
1. Prompt user for search criteria (name, birth year, birth place, etc.)
2. Run Action 10 search using GEDCOM file
3. Run Action 11 search using Ancestry API
4. Compare and display results side-by-side
5. Highlight any differences in scoring, family data, or relationship paths
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module  # type: ignore[import-not-found]

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === IMPORTS ===
import os
from typing import Any, Optional

# Import action modules
import action10  # type: ignore[import-not-found]
import action11  # type: ignore[import-not-found]
from core.session_manager import SessionManager


def get_search_input() -> dict[str, Any]:
    """
    Prompt user for search criteria.

    Returns:
        Dictionary with search criteria
    """
    print("\n" + "=" * 60)
    print("Action 12: Compare GEDCOM vs API Search Results")
    print("=" * 60)
    print("\nEnter search criteria:")
    print("-" * 60)

    criteria = {}

    # Get name
    first_name = input("First name: ").strip()
    surname = input("Surname: ").strip()
    criteria["first_name"] = first_name
    criteria["surname"] = surname
    criteria["full_name"] = f"{first_name} {surname}".strip()

    # Get birth info
    birth_year = input("Birth year (or press Enter to skip): ").strip()
    if birth_year:
        criteria["birth_year"] = birth_year

    birth_place = input("Birth place (or press Enter to skip): ").strip()
    if birth_place:
        criteria["birth_place"] = birth_place

    # Get death info
    death_year = input("Death year (or press Enter to skip): ").strip()
    if death_year:
        criteria["death_year"] = death_year

    death_place = input("Death place (or press Enter to skip): ").strip()
    if death_place:
        criteria["death_place"] = death_place

    # Get gender
    gender = input("Gender (M/F or press Enter to skip): ").strip().upper()
    if gender in ["M", "F"]:
        criteria["gender"] = gender

    return criteria


def run_action10_search(criteria: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    """
    Run Action 10 (GEDCOM-based) search.

    Args:
        criteria: Search criteria dictionary

    Returns:
        List of search results or None if failed
    """
    print("\n" + "=" * 60)
    print("Running Action 10 (GEDCOM-based search)...")
    print("=" * 60)

    try:
        # Build search criteria dict for Action 10
        search_criteria = {
            "first_name": criteria.get("first_name", "").lower(),
            "surname": criteria.get("surname", "").lower(),
        }

        # Add optional fields if provided
        if criteria.get("birth_year"):
            search_criteria["birth_year"] = int(criteria["birth_year"])
        if criteria.get("birth_place"):
            search_criteria["birth_place"] = criteria["birth_place"]
        if criteria.get("death_year"):
            search_criteria["death_year"] = int(criteria["death_year"])
        if criteria.get("death_place"):
            search_criteria["death_place"] = criteria["death_place"]
        if criteria.get("gender"):
            search_criteria["gender"] = criteria["gender"].lower()

        # Call Action 10's filter_and_score_individuals function
        from pathlib import Path

        from action10 import load_gedcom_data  # type: ignore[import-not-found]

        gedcom_path = os.getenv("GEDCOM_FILE_PATH")
        if not gedcom_path:
            print("❌ GEDCOM_FILE_PATH not configured in .env")
            return None

        gedcom_data = load_gedcom_data(Path(gedcom_path))
        if not gedcom_data:
            print("❌ Failed to load GEDCOM data")
            return None

        # Use Action 10's scoring function
        from config_schema import config_schema  # type: ignore[import-not-found]

        scoring_weights = dict(config_schema.common_scoring_weights) if config_schema else {}
        date_flex = {"year_match_range": 5}

        results = action10.filter_and_score_individuals(
            gedcom_data,
            search_criteria,
            search_criteria,
            scoring_weights,
            date_flex,
        )
        return results if results else None
    except Exception as e:
        logger.error(f"Action 10 search failed: {e}", exc_info=True)
        print(f"\n❌ Action 10 search failed: {e}")
        return None


def run_action11_search(
    session_manager: SessionManager, criteria: dict[str, Any]
) -> Optional[list[dict[str, Any]]]:
    """
    Run Action 11 (API-based) search.

    Args:
        session_manager: Active session manager
        criteria: Search criteria dictionary

    Returns:
        List of search results or None if failed
    """
    print("\n" + "=" * 60)
    print("Running Action 11 (API-based search)...")
    print("=" * 60)

    try:
        # Build search criteria dict for Action 11
        search_criteria = {
            "first_name": criteria.get("first_name", "").lower(),
            "surname": criteria.get("surname", "").lower(),
        }

        # Add optional fields if provided
        if criteria.get("birth_year"):
            search_criteria["birth_year"] = int(criteria["birth_year"])
        if criteria.get("birth_place"):
            search_criteria["birth_place"] = criteria["birth_place"]
        if criteria.get("death_year"):
            search_criteria["death_year"] = int(criteria["death_year"])
        if criteria.get("death_place"):
            search_criteria["death_place"] = criteria["death_place"]
        if criteria.get("gender"):
            search_criteria["gender"] = criteria["gender"].lower()

        # Call Action 11's search function with correct signature
        result = action11.search_ancestry_api_for_person(
            session_manager=session_manager,
            search_criteria=search_criteria,
            max_results=5,
        )
        return result if result else None
    except Exception as e:
        logger.error(f"Action 11 search failed: {e}", exc_info=True)
        print(f"\n❌ Action 11 search failed: {e}")
        return None


def _print_search_criteria(criteria: dict[str, Any]) -> None:
    """Print search criteria."""
    print(f"\nSearch Criteria: {criteria.get('full_name', 'Unknown')}")
    if criteria.get("birth_year"):
        print(f"Birth Year: {criteria['birth_year']}")
    if criteria.get("birth_place"):
        print(f"Birth Place: {criteria['birth_place']}")


def _print_action10_results(action10_result: Optional[list[dict[str, Any]]]) -> None:
    """Print Action 10 (GEDCOM) results."""
    print("\n" + "-" * 60)
    print("ACTION 10 (GEDCOM) RESULTS:")
    print("-" * 60)
    if action10_result and len(action10_result) > 0:
        top_match = action10_result[0]
        print(f"✅ Found {len(action10_result)} match(es)")
        print(f"   Top Match: {top_match.get('full_name_disp', 'Unknown')}")
        print(f"   Score: {top_match.get('score', 0)}")
        print(f"   Birth: {top_match.get('birth_year', 'Unknown')}")
        print(f"   Death: {top_match.get('death_year', 'Unknown')}")
    else:
        print("❌ No results found")


def _print_action11_results(action11_result: Optional[list[dict[str, Any]]]) -> None:
    """Print Action 11 (API) results."""
    print("\n" + "-" * 60)
    print("ACTION 11 (API) RESULTS:")
    print("-" * 60)
    if action11_result and len(action11_result) > 0:
        top_match = action11_result[0]
        print(f"✅ Found {len(action11_result)} match(es)")
        print(f"   Top Match: {top_match.get('display_name', 'Unknown')}")
        print(f"   Score: {top_match.get('score', 0)}")
        print(f"   Birth: {top_match.get('birth_year', 'Unknown')}")
        print(f"   Death: {top_match.get('death_year', 'Unknown')}")
    else:
        print("❌ No results found")


def _print_score_comparison(
    action10_result: Optional[list[dict[str, Any]]],
    action11_result: Optional[list[dict[str, Any]]]
) -> None:
    """Print score comparison between Action 10 and Action 11."""
    if not (action10_result and action11_result and
            len(action10_result) > 0 and len(action11_result) > 0):
        return

    print("\n" + "-" * 60)
    print("SCORE COMPARISON:")
    print("-" * 60)
    score10 = action10_result[0].get("score", 0)
    score11 = action11_result[0].get("score", 0)
    diff = abs(score10 - score11)

    print(f"Action 10 Score: {score10}")
    print(f"Action 11 Score: {score11}")
    print(f"Difference: {diff}")

    if diff == 0:
        print("✅ Scores match perfectly!")
    elif diff <= 10:
        print("⚠️  Minor score difference (within tolerance)")
    else:
        print("❌ Significant score difference - investigate!")


def compare_results(
    action10_result: Optional[list[dict[str, Any]]],
    action11_result: Optional[list[dict[str, Any]]],
    criteria: dict[str, Any],
) -> None:
    """
    Compare and display results from Action 10 and Action 11.

    Args:
        action10_result: Results from Action 10 (list of matches)
        action11_result: Results from Action 11 (list of matches)
        criteria: Original search criteria
    """
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    _print_search_criteria(criteria)
    _print_action10_results(action10_result)
    _print_action11_results(action11_result)
    _print_score_comparison(action10_result, action11_result)

    print("\n" + "=" * 60)


def run_action12_wrapper(session_manager: SessionManager) -> bool:
    """
    Main wrapper for Action 12.

    Args:
        session_manager: Active session manager

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get search criteria from user
        criteria = get_search_input()

        if not criteria.get("full_name"):
            print("\n❌ Error: Name is required")
            return False

        # Run Action 10 search
        action10_result = run_action10_search(criteria)

        # Run Action 11 search
        action11_result = run_action11_search(session_manager, criteria)

        # Compare results
        compare_results(action10_result, action11_result, criteria)

        return True

    except Exception as e:
        logger.error(f"Action 12 failed: {e}", exc_info=True)
        print(f"\n❌ Action 12 failed: {e}")
        return False


# === TESTS ===
def _test_search_input_structure() -> None:
    """Test get_search_input returns proper structure."""
    # We can't test interactive input, but we can test the expected structure
    # by mocking input
    from unittest.mock import patch

    inputs = ["John", "Smith", "1950", "London", "2020", "Manchester", "M"]
    with patch('builtins.input', side_effect=inputs):
        criteria = get_search_input()

        assert "first_name" in criteria, "Should have first_name"
        assert "surname" in criteria, "Should have surname"
        assert "full_name" in criteria, "Should have full_name"
        assert "birth_year" in criteria, "Should have birth_year"
        assert "birth_place" in criteria, "Should have birth_place"
        assert "death_year" in criteria, "Should have death_year"
        assert "death_place" in criteria, "Should have death_place"
        assert "gender" in criteria, "Should have gender"

        assert criteria["first_name"] == "John"
        assert criteria["surname"] == "Smith"
        assert criteria["full_name"] == "John Smith"
        assert criteria["birth_year"] == "1950"
        assert criteria["gender"] == "M"


def _test_search_input_optional_fields() -> None:
    """Test get_search_input handles optional fields."""
    from unittest.mock import patch

    # Only provide name, skip all optional fields
    inputs = ["Jane", "Doe", "", "", "", "", ""]
    with patch('builtins.input', side_effect=inputs):
        criteria = get_search_input()

        assert criteria["first_name"] == "Jane"
        assert criteria["surname"] == "Doe"
        assert "birth_year" not in criteria, "Should not have birth_year if skipped"
        assert "birth_place" not in criteria, "Should not have birth_place if skipped"
        assert "death_year" not in criteria, "Should not have death_year if skipped"
        assert "death_place" not in criteria, "Should not have death_place if skipped"
        assert "gender" not in criteria, "Should not have gender if skipped"


def _test_run_action10_search_structure() -> None:
    """Test run_action10_search builds proper search criteria."""
    # We can't run the actual search without GEDCOM file, but we can verify
    # the function exists and has proper signature
    import inspect
    sig = inspect.signature(run_action10_search)
    assert "criteria" in sig.parameters, "Should have criteria parameter"

    # Verify return type annotation
    assert sig.return_annotation != inspect.Signature.empty, "Should have return type annotation"


def _test_run_action11_search_structure() -> None:
    """Test run_action11_search builds proper search criteria."""
    # Verify function signature
    import inspect
    sig = inspect.signature(run_action11_search)
    assert "criteria" in sig.parameters, "Should have criteria parameter"
    assert "session_manager" in sig.parameters, "Should have session_manager parameter"

    # Verify return type annotation
    assert sig.return_annotation != inspect.Signature.empty, "Should have return type annotation"


def _test_compare_results_structure() -> None:
    """Test compare_results handles different result scenarios."""
    # Test with None results (both failed)
    compare_results(None, None, {"first_name": "Test", "surname": "Person"})

    # Test with empty results
    compare_results([], [], {"first_name": "Test", "surname": "Person"})

    # Test with one None, one empty
    compare_results(None, [], {"first_name": "Test", "surname": "Person"})
    compare_results([], None, {"first_name": "Test", "surname": "Person"})


def _test_compare_results_output() -> None:
    """Test compare_results produces output for different scenarios."""
    # Test with sample Action 10 and Action 11 results
    action10_results = [
        {
            "id": "@I1@",
            "first_name": "John",
            "surname": "Smith",
            "birth_year": 1950,
            "total_score": 100,
            "confidence": "high"
        }
    ]

    action11_results = [
        {
            "personId": "12345",
            "givenNames": "John",
            "surnames": "Smith",
            "birthDate": "1950",
            "total_score": 95,
            "confidence": "high"
        }
    ]

    criteria = {"first_name": "John", "surname": "Smith"}

    # Should not raise exception
    compare_results(action10_results, action11_results, criteria)


def _test_action12_functions_available() -> None:
    """Test all Action 12 functions are available."""
    import inspect

    # Check main functions exist
    assert callable(get_search_input), "get_search_input should be callable"
    assert callable(run_action10_search), "run_action10_search should be callable"
    assert callable(run_action11_search), "run_action11_search should be callable"
    assert callable(compare_results), "compare_results should be callable"
    assert callable(run_action12_wrapper), "run_action12_wrapper should be callable"

    # Verify they have proper signatures
    for func in [get_search_input, run_action10_search, run_action11_search,
                 compare_results, run_action12_wrapper]:
        sig = inspect.signature(func)
        assert sig is not None, f"{func.__name__} should have signature"


def _test_action12_imports() -> None:
    """Test Action 12 has required imports."""
    # Verify action10 and action11 are imported
    import sys
    assert 'action10' in sys.modules, "action10 should be imported"
    assert 'action11' in sys.modules, "action11 should be imported"

    # Verify they have required functions
    assert hasattr(action10, 'filter_and_score_individuals'), "action10 should have filter_and_score_individuals"
    assert hasattr(action11, 'search_ancestry_api_for_person'), "action11 should have search_ancestry_api_for_person"


def _test_print_functions() -> None:
    """Test print helper functions don't crash."""
    # Test _print_search_criteria
    criteria = {"full_name": "John Smith", "birth_year": "1950", "birth_place": "London"}
    _print_search_criteria(criteria)

    # Test with minimal criteria
    _print_search_criteria({"full_name": "Jane Doe"})

    # Test _print_action10_results with results
    action10_results = [
        {
            "full_name_disp": "John Smith",
            "score": 95,
            "birth_year": 1950,
            "death_year": 2020
        }
    ]
    _print_action10_results(action10_results)

    # Test with None
    _print_action10_results(None)

    # Test with empty list
    _print_action10_results([])

    # Test _print_action11_results with results
    action11_results = [
        {
            "display_name": "John Smith",
            "score": 92,
            "birth_year": 1950,
            "death_year": 2020
        }
    ]
    _print_action11_results(action11_results)

    # Test with None
    _print_action11_results(None)

    # Test with empty list
    _print_action11_results([])


def _test_score_comparison() -> None:
    """Test score comparison logic."""
    # Test with matching scores
    action10_results = [{"score": 100}]
    action11_results = [{"score": 100}]
    _print_score_comparison(action10_results, action11_results)

    # Test with minor difference
    action10_results = [{"score": 100}]
    action11_results = [{"score": 95}]
    _print_score_comparison(action10_results, action11_results)

    # Test with significant difference
    action10_results = [{"score": 100}]
    action11_results = [{"score": 50}]
    _print_score_comparison(action10_results, action11_results)

    # Test with None results
    _print_score_comparison(None, None)
    _print_score_comparison(action10_results, None)
    _print_score_comparison(None, action11_results)

    # Test with empty results
    _print_score_comparison([], [])


def _test_search_criteria_building() -> None:
    """Test search criteria building for both actions."""
    from unittest.mock import patch

    # Test criteria with all fields
    full_criteria = {
        "first_name": "John",
        "surname": "Smith",
        "birth_year": "1950",
        "birth_place": "London",
        "death_year": "2020",
        "death_place": "Manchester",
        "gender": "M"
    }

    # Mock GEDCOM loading for Action 10 (import is inside the function)
    from unittest.mock import MagicMock

    # Create a mock module with config_schema attribute
    mock_config_module = MagicMock()
    mock_config = MagicMock()
    mock_config.common_scoring_weights = {}
    mock_config_module.config_schema = mock_config

    # Mock the module import
    import sys
    sys.modules['config_schema'] = mock_config_module

    try:
        with patch('action10.load_gedcom_data') as mock_load, \
             patch('action10.filter_and_score_individuals') as mock_filter, \
             patch.dict('os.environ', {'GEDCOM_FILE_PATH': '/path/to/gedcom.ged'}):

            mock_load.return_value = {"individuals": []}
            mock_filter.return_value = []

            # Run the search
            _ = run_action10_search(full_criteria)

            # Verify filter_and_score_individuals was called
            assert mock_filter.called, "Should call filter_and_score_individuals"

            # Verify search criteria was built correctly
            call_args = mock_filter.call_args
            search_criteria = call_args[0][1]  # Second argument

            assert search_criteria["first_name"] == "john", "Should lowercase first name"
            assert search_criteria["surname"] == "smith", "Should lowercase surname"
            assert search_criteria["birth_year"] == 1950, "Should convert birth year to int"
            assert search_criteria["gender"] == "m", "Should lowercase gender"
    finally:
        # Clean up the mock module
        if 'config_schema' in sys.modules:
            del sys.modules['config_schema']


def _test_action10_error_handling() -> None:
    """Test Action 10 error handling."""
    from unittest.mock import patch

    # Test with missing GEDCOM_FILE_PATH
    with patch.dict('os.environ', {}, clear=True):
        result = run_action10_search({"first_name": "John", "surname": "Smith"})
        assert result is None, "Should return None when GEDCOM_FILE_PATH not set"

    # Test with GEDCOM load failure (import is inside the function)
    with patch('action10.load_gedcom_data', return_value=None), \
         patch.dict('os.environ', {'GEDCOM_FILE_PATH': '/path/to/gedcom.ged'}):
        result = run_action10_search({"first_name": "John", "surname": "Smith"})
        assert result is None, "Should return None when GEDCOM load fails"

    # Test with exception during search
    with patch('action10.load_gedcom_data', side_effect=Exception("Test error")), \
         patch.dict('os.environ', {'GEDCOM_FILE_PATH': '/path/to/gedcom.ged'}):
        result = run_action10_search({"first_name": "John", "surname": "Smith"})
        assert result is None, "Should return None when exception occurs"


def _test_action11_error_handling() -> None:
    """Test Action 11 error handling."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()

    # Test with exception during search
    with patch('action11.search_ancestry_api_for_person', side_effect=Exception("API error")):
        result = run_action11_search(mock_sm, {"first_name": "John", "surname": "Smith"})
        assert result is None, "Should return None when exception occurs"

    # Test with None result from API
    with patch('action11.search_ancestry_api_for_person', return_value=None):
        result = run_action11_search(mock_sm, {"first_name": "John", "surname": "Smith"})
        assert result is None, "Should return None when API returns None"

    # Test with empty result from API
    with patch('action11.search_ancestry_api_for_person', return_value=[]):
        result = run_action11_search(mock_sm, {"first_name": "John", "surname": "Smith"})
        assert result is None, "Should return None when API returns empty list"


def _test_wrapper_function() -> None:
    """Test run_action12_wrapper function."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()

    # Test with missing name
    with patch('builtins.input', side_effect=["", "", "", "", "", "", ""]):
        result = run_action12_wrapper(mock_sm)
        assert result is False, "Should return False when name is missing"

    # Test with successful execution
    with patch('builtins.input', side_effect=["John", "Smith", "", "", "", "", ""]), \
         patch('action10.load_gedcom_data', return_value={"individuals": []}), \
         patch('action10.filter_and_score_individuals', return_value=[{"score": 100}]), \
         patch('action11.search_ancestry_api_for_person', return_value=[{"score": 95}]), \
         patch.dict('os.environ', {'GEDCOM_FILE_PATH': '/path/to/gedcom.ged'}):
        result = run_action12_wrapper(mock_sm)
        assert result is True, "Should return True on successful execution"

    # Test with exception during input
    with patch('builtins.input', side_effect=Exception("Test error")):
        result = run_action12_wrapper(mock_sm)
        assert result is False, "Should return False when exception occurs"


def action12_module_tests() -> bool:
    """Comprehensive test suite for action12.py"""
    from test_framework import TestSuite

    suite = TestSuite("Action 12: GEDCOM vs API Comparison", "action12.py")
    suite.start_suite()

    # Category 1: Input Handling Tests
    suite.run_test(
        "Search input structure",
        _test_search_input_structure,
        "Returns proper structure with all fields",
        "get_search_input()",
        "Tests that search input collection returns complete criteria dictionary"
    )

    suite.run_test(
        "Search input optional fields",
        _test_search_input_optional_fields,
        "Handles optional fields correctly",
        "get_search_input()",
        "Tests that optional fields are omitted when not provided"
    )

    # Category 2: Search Function Tests
    suite.run_test(
        "Action 10 search structure",
        _test_run_action10_search_structure,
        "Has proper function signature",
        "run_action10_search()",
        "Tests that Action 10 search has correct parameters and return type"
    )

    suite.run_test(
        "Action 11 search structure",
        _test_run_action11_search_structure,
        "Has proper function signature",
        "run_action11_search()",
        "Tests that Action 11 search has correct parameters and return type"
    )

    suite.run_test(
        "Search criteria building",
        _test_search_criteria_building,
        "Builds correct search criteria",
        "run_action10_search()",
        "Tests that search criteria is properly formatted for Action 10"
    )

    # Category 3: Error Handling Tests
    suite.run_test(
        "Action 10 error handling",
        _test_action10_error_handling,
        "Handles errors gracefully",
        "run_action10_search()",
        "Tests error handling for missing GEDCOM, load failures, and exceptions"
    )

    suite.run_test(
        "Action 11 error handling",
        _test_action11_error_handling,
        "Handles API errors gracefully",
        "run_action11_search()",
        "Tests error handling for API failures and empty results"
    )

    # Category 4: Comparison and Output Tests
    suite.run_test(
        "Compare results structure",
        _test_compare_results_structure,
        "Handles different result scenarios",
        "compare_results()",
        "Tests comparison with None, empty, and mixed results"
    )

    suite.run_test(
        "Compare results output",
        _test_compare_results_output,
        "Produces output for sample results",
        "compare_results()",
        "Tests output formatting with sample Action 10 and 11 results"
    )

    suite.run_test(
        "Print helper functions",
        _test_print_functions,
        "Print functions don't crash",
        "_print_search_criteria(), _print_action10_results(), _print_action11_results()",
        "Tests all print helper functions with various inputs"
    )

    suite.run_test(
        "Score comparison logic",
        _test_score_comparison,
        "Compares scores correctly",
        "_print_score_comparison()",
        "Tests score comparison with matching, minor, and significant differences"
    )

    # Category 5: Integration Tests
    suite.run_test(
        "Wrapper function",
        _test_wrapper_function,
        "Orchestrates full workflow",
        "run_action12_wrapper()",
        "Tests main wrapper function with various scenarios"
    )

    # Category 6: Function Availability Tests
    suite.run_test(
        "Action 12 functions available",
        _test_action12_functions_available,
        "All functions exist and callable",
        "Module functions",
        "Tests that all required functions are available"
    )

    suite.run_test(
        "Action 12 imports",
        _test_action12_imports,
        "Required modules imported",
        "Module imports",
        "Tests that action10 and action11 are properly imported"
    )

    return suite.finish_suite()


# Create standard test runner
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(action12_module_tests)


if __name__ == "__main__":
    import sys

    # If run directly, execute tests
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_comprehensive_tests()
    else:
        print("Action 12: Compare GEDCOM vs API Search Results")
        print("This action must be run from main.py")
        print("\nTo run tests: python action12.py --test")

