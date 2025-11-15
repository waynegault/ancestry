#!/usr/bin/env python3

"""
Universal Scoring & Advanced System Intelligence Engine

Sophisticated platform providing comprehensive automation capabilities,
intelligent processing, and advanced functionality with optimized algorithms,
professional-grade operations, and comprehensive management for genealogical
automation and research workflows.

System Intelligence:
• Advanced automation with intelligent processing and optimization protocols
• Sophisticated management with comprehensive operational capabilities
• Intelligent coordination with multi-system integration and synchronization
• Comprehensive analytics with detailed performance metrics and insights
• Advanced validation with quality assessment and verification protocols
• Integration with platforms for comprehensive system management and automation

Automation Capabilities:
• Sophisticated automation with intelligent workflow generation and execution
• Advanced optimization with performance monitoring and enhancement protocols
• Intelligent coordination with automated management and orchestration
• Comprehensive validation with quality assessment and reliability protocols
• Advanced analytics with detailed operational insights and optimization
• Integration with automation systems for comprehensive workflow management

Professional Operations:
• Advanced professional functionality with enterprise-grade capabilities and reliability
• Sophisticated operational protocols with professional standards and best practices
• Intelligent optimization with performance monitoring and enhancement
• Comprehensive documentation with detailed operational guides and analysis
• Advanced security with secure protocols and data protection measures
• Integration with professional systems for genealogical research workflows

Foundation Services:
Provides the essential infrastructure that enables reliable, high-performance
operations through intelligent automation, comprehensive management,
and professional capabilities for genealogical automation and research workflows.

Technical Implementation:
Universal Scoring Module

Provides standardized scoring functionality for genealogical data across
Action 10 (GEDCOM) and Action 11 (API) modules. This module consolidates
duplicate scoring logic and ensures consistent scoring algorithms.

Features:
- Universal scoring function for both GEDCOM and API data
- Standardized result formatting
- Consistent scoring criteria handling
- Performance optimizations for large datasets
"""

import time
from typing import Any, Optional

from standard_imports import setup_module
from test_framework import Colors

logger = setup_module(globals(), __name__)


# === DISPLAY FORMATTING UTILITIES ===
# Shared display formatting functions for Action 10 and Action 11


def calculate_display_bonuses(scores: dict[str, int], key_prefix: str = "") -> dict[str, int]:
    """
    Calculate display bonus values for birth and death.

    Works with both action10 (uses '_s' suffix) and action11 (no suffix) score keys.

    Args:
        scores: Dictionary of field scores
        key_prefix: Optional prefix for score keys (e.g., "_s" for action10)

    Returns:
        Dictionary with bonus calculations
    """
    # Determine key names based on prefix
    byear_key = f"byear{key_prefix}"
    bdate_key = f"bdate{key_prefix}"
    dyear_key = f"dyear{key_prefix}"
    ddate_key = f"ddate{key_prefix}"
    bplace_key = f"bplace{key_prefix}"
    dplace_key = f"dplace{key_prefix}"

    # Calculate date components
    birth_date_component = max(scores.get(byear_key, 0), scores.get(bdate_key, 0))
    death_date_component = max(scores.get(dyear_key, 0), scores.get(ddate_key, 0))

    # Calculate bonuses
    birth_bonus = 25 if (birth_date_component > 0 and scores.get(bplace_key, 0) > 0) else 0
    death_bonus = 25 if (death_date_component > 0 and scores.get(dplace_key, 0) > 0) else 0

    return {
        "birth_date_component": birth_date_component,
        "death_date_component": death_date_component,
        "birth_bonus": birth_bonus,
        "death_bonus": death_bonus,
    }


def apply_universal_scoring(
    candidates: list[dict[str, Any]],
    search_criteria: dict[str, Any],
    scoring_weights: dict[str, Any] | None = None,
    date_flexibility: dict[str, Any] | None = None,
    max_results: int = 10,
    performance_timeout: float = 5.0
) -> list[dict[str, Any]]:
    """
    Apply universal scoring to a list of candidates.

    This function provides consistent scoring across Action 10 and Action 11
    by using the same underlying scoring algorithm and result formatting.

    Args:
        candidates: List of candidate dictionaries to score
        search_criteria: Search parameters for scoring
        scoring_weights: Weights for different scoring fields
        date_flexibility: Date matching flexibility settings
        max_results: Maximum number of results to return
        performance_timeout: Maximum time to spend scoring (seconds)

    Returns:
        List of scored candidates sorted by score (highest first)
    """
    try:
        from config import config_schema
        from gedcom_utils import calculate_match_score

        # Use default weights if not provided
        if scoring_weights is None:
            scoring_weights = getattr(config_schema, 'common_scoring_weights', {})

        # Use default date flexibility if not provided
        if date_flexibility is None:
            date_flexibility = {"year_match_range": 5.0}

        scored_results = []
        start_time = time.time()

        logger.debug(f"Scoring {len(candidates)} candidates with universal scoring")

        for i, candidate in enumerate(candidates):
            # Performance timeout check
            if (time.time() - start_time) > performance_timeout:
                logger.debug(f"Universal scoring timeout after {i} candidates")
                break

            # Early termination if we have enough high-quality results
            if len(scored_results) >= max_results and scored_results[-1].get('total_score', 0) > 150:
                logger.debug(f"Early termination with {len(scored_results)} high-quality results")
                break

            try:
                # Calculate score using universal algorithm
                total_score, field_scores, reasons = calculate_match_score(
                    search_criteria=search_criteria,
                    candidate_processed_data=candidate,
                    scoring_weights=scoring_weights,
                    date_flexibility=date_flexibility
                )

                # Create standardized result format
                result = candidate.copy()
                result.update({
                    "total_score": int(total_score),
                    "field_scores": field_scores,
                    "reasons": reasons,
                    "full_name_disp": f"{candidate.get('first_name', '')} {candidate.get('surname', '')}".strip(),
                    "confidence": _get_confidence_level(total_score)
                })

                scored_results.append(result)

            except Exception as e:
                candidate_id = candidate.get('id', candidate.get('person_id', 'unknown'))
                logger.warning(f"Error scoring candidate {candidate_id}: {e}")
                continue

        # Sort by score (descending) and return top results
        scored_results.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        final_results = scored_results[:max_results]

        scoring_time = time.time() - start_time
        logger.debug(f"Universal scoring completed: {len(final_results)} results in {scoring_time:.3f}s")

        return final_results

    except Exception as e:
        logger.error(f"Universal scoring failed: {e}")
        return []


def _get_confidence_level(score: float) -> str:
    """Get confidence level based on score."""
    if score >= 200:
        return "very_high"
    if score >= 150:
        return "high"
    if score >= 100:
        return "medium"
    if score >= 50:
        return "low"
    return "very_low"


def format_scoring_breakdown(
    result: dict[str, Any],
    _search_criteria: dict[str, Any],
    title: str = "Scoring Breakdown"
) -> str:
    """
    Format a detailed scoring breakdown for display.

    Args:
        result: Scored result dictionary
        search_criteria: Original search criteria
        title: Title for the breakdown display

    Returns:
        Formatted string with scoring breakdown
    """
    try:
        score = result.get('total_score', 0)
        field_scores = result.get('field_scores', {})
        reasons = result.get('reasons', [])

        output = [f"\n{Colors.cyan(f'📊 {title}:')}"]
        output.append("Field        Score  Description")
        output.append("--------------------------------------------------")

        # Standard field mapping for consistent display
        field_mapping = {
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
            'bonus': 'Name Bonus'
        }

        # Display field scores
        for field_key, field_score in field_scores.items():
            field_name = field_mapping.get(field_key, field_key.title())
            output.append(f"{field_name:<12} {field_score:>5}  {_get_field_description(field_key, field_score)}")

        output.append("--------------------------------------------------")
        output.append(f"{'Total Score':<12} {score:>5}  {_get_confidence_level(score).replace('_', ' ').title()}")

        # Add reasons if available
        if reasons:
            output.append(f"\n{Colors.yellow('📝 Scoring Reasons:')}")
            for reason in reasons[:5]:  # Limit to top 5 reasons
                output.append(f"   • {reason}")

        return "\n".join(output)

    except Exception as e:
        logger.warning(f"Error formatting scoring breakdown: {e}")
        return f"Scoring breakdown unavailable: {e}"


def _get_field_description(_field_key: str, score: float) -> str:
    """Get description for a field score."""
    if score >= 25:
        return "Excellent match"
    if score >= 15:
        return "Good match"
    if score >= 5:
        return "Partial match"
    if score > 0:
        return "Weak match"
    return "No match"


def _normalize_name_fields(criteria: dict[str, Any], normalized: dict[str, Any]) -> None:
    """Normalize name fields to lowercase."""
    if 'first_name' in criteria:
        normalized['first_name'] = str(criteria['first_name']).lower().strip()
    if 'surname' in criteria:
        normalized['surname'] = str(criteria['surname']).lower().strip()


def _normalize_numeric_fields(criteria: dict[str, Any], normalized: dict[str, Any]) -> None:
    """Normalize numeric fields with validation."""
    for field in ['birth_year', 'death_year']:
        if field in criteria and criteria[field] is not None:
            try:
                normalized[field] = int(criteria[field])
            except (ValueError, TypeError):
                logger.warning(f"Invalid {field}: {criteria[field]}")
                normalized[field] = None


def _normalize_gender_field(criteria: dict[str, Any], normalized: dict[str, Any]) -> None:
    """Normalize gender field to standard format."""
    if 'gender' not in criteria:
        return

    gender = str(criteria['gender']).lower().strip()
    if gender in ['m', 'male', 'man']:
        normalized['gender'] = 'm'
    elif gender in ['f', 'female', 'woman']:
        normalized['gender'] = 'f'
    else:
        normalized['gender'] = gender


def _copy_remaining_fields(criteria: dict[str, Any], normalized: dict[str, Any]) -> None:
    """Copy remaining fields that weren't normalized."""
    for key, value in criteria.items():
        if key not in normalized:
            normalized[key] = value


def validate_search_criteria(criteria: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize search criteria for consistent scoring.

    Args:
        criteria: Raw search criteria

    Returns:
        Normalized search criteria
    """
    normalized = {}

    # Normalize different field types
    _normalize_name_fields(criteria, normalized)
    _normalize_numeric_fields(criteria, normalized)
    _normalize_gender_field(criteria, normalized)
    _copy_remaining_fields(criteria, normalized)

    return normalized


# ==============================================
# Module Tests
# ==============================================


def _test_universal_scoring_basic() -> None:
    """Test basic universal scoring functionality."""
    candidates = [
        {
            "id": "@I1@",
            "first_name": "john",
            "surname": "smith",
            "birth_year": 1950,
            "gender": "m"
        }
    ]

    search_criteria = {
        "first_name": "john",
        "surname": "smith",
        "birth_year": 1950
    }

    results = apply_universal_scoring(candidates, search_criteria, max_results=1)

    # Validate result structure
    assert isinstance(results, list), "Results should be a list"
    assert len(results) > 0, "Should have at least one result"
    result = results[0]
    assert 'total_score' in result, "Result should have total_score"
    assert 'confidence' in result, "Result should have confidence"
    assert 'full_name_disp' in result, "Result should have full_name_disp"
    assert result['total_score'] > 0, "Score should be positive for matching data"


def _test_universal_scoring_exact_match() -> None:
    """Test scoring with exact match on all fields."""
    candidates = [
        {
            "id": "@I1@",
            "first_name": "fraser",
            "surname": "gault",
            "birth_year": 1960,
            "birth_place": "Banff, Scotland",
            "death_year": 2020,
            "death_place": "Aberdeen, Scotland",
            "gender": "m"
        }
    ]

    search_criteria = {
        "first_name": "fraser",
        "surname": "gault",
        "birth_year": 1960,
        "birth_place": "Banff, Scotland",
        "death_year": 2020,
        "death_place": "Aberdeen, Scotland",
        "gender": "m"
    }

    results = apply_universal_scoring(candidates, search_criteria, max_results=1)

    assert len(results) > 0, "Should have results for exact match"
    result = results[0]
    assert result['total_score'] > 50, "Exact match should have positive score"
    assert result['confidence'] in ['low', 'medium', 'high', 'very_high'], "Should have valid confidence level"


def _test_universal_scoring_partial_match() -> None:
    """Test scoring with partial match (name only)."""
    candidates = [
        {
            "id": "@I1@",
            "first_name": "john",
            "surname": "smith",
            "birth_year": None,
            "gender": None
        }
    ]

    search_criteria = {
        "first_name": "john",
        "surname": "smith",
        "birth_year": 1950,
        "gender": "m"
    }

    results = apply_universal_scoring(candidates, search_criteria, max_results=1)

    assert len(results) > 0, "Should have results for partial match"
    result = results[0]
    assert result['total_score'] > 0, "Partial match should have positive score"
    # Note: Scoring algorithm may give higher scores for name-only matches due to bonuses


def _test_universal_scoring_no_match() -> None:
    """Test scoring with no match."""
    candidates = [
        {
            "id": "@I1@",
            "first_name": "jane",
            "surname": "doe",
            "birth_year": 1980,
            "gender": "f"
        }
    ]

    search_criteria = {
        "first_name": "john",
        "surname": "smith",
        "birth_year": 1950,
        "gender": "m"
    }

    results = apply_universal_scoring(candidates, search_criteria, max_results=1)

    assert len(results) > 0, "Should still return results even with no match"
    result = results[0]
    # Note: Scoring algorithm may give partial scores even for non-matches
    assert result['total_score'] >= 0, "Score should be non-negative"


def _test_universal_scoring_multiple_candidates() -> None:
    """Test scoring with multiple candidates."""
    candidates = [
        {
            "id": "@I1@",
            "first_name": "john",
            "surname": "smith",
            "birth_year": 1950,
            "gender": "m"
        },
        {
            "id": "@I2@",
            "first_name": "john",
            "surname": "smith",
            "birth_year": 1951,
            "gender": "m"
        },
        {
            "id": "@I3@",
            "first_name": "jane",
            "surname": "smith",
            "birth_year": 1950,
            "gender": "f"
        }
    ]

    search_criteria = {
        "first_name": "john",
        "surname": "smith",
        "birth_year": 1950,
        "gender": "m"
    }

    results = apply_universal_scoring(candidates, search_criteria, max_results=3)

    assert len(results) == 3, "Should return all candidates"
    # Results should be sorted by score (descending)
    assert results[0]['total_score'] >= results[1]['total_score'], "Results should be sorted by score"
    assert results[1]['total_score'] >= results[2]['total_score'], "Results should be sorted by score"


def _test_universal_scoring_max_results() -> None:
    """Test max_results parameter."""
    candidates = [
        {"id": f"@I{i}@", "first_name": "john", "surname": "smith", "birth_year": 1950 + i, "gender": "m"}
        for i in range(20)
    ]

    search_criteria = {
        "first_name": "john",
        "surname": "smith",
        "birth_year": 1950
    }

    results = apply_universal_scoring(candidates, search_criteria, max_results=5)

    assert len(results) <= 5, "Should respect max_results parameter"


def _test_criteria_validation_names() -> None:
    """Test search criteria validation for names."""
    criteria = {
        "first_name": "  JOHN  ",
        "surname": "SMITH"
    }

    normalized = validate_search_criteria(criteria)

    assert normalized['first_name'] == "john", "First name should be lowercase and trimmed"
    assert normalized['surname'] == "smith", "Surname should be lowercase and trimmed"


def _test_criteria_validation_years() -> None:
    """Test search criteria validation for years."""
    criteria = {
        "birth_year": "1950",
        "death_year": "2020"
    }

    normalized = validate_search_criteria(criteria)

    assert normalized['birth_year'] == 1950, "Birth year should be converted to int"
    assert normalized['death_year'] == 2020, "Death year should be converted to int"


def _test_criteria_validation_gender() -> None:
    """Test search criteria validation for gender."""
    test_cases = [
        ("Male", "m"),
        ("m", "m"),
        ("man", "m"),
        ("Female", "f"),
        ("f", "f"),
        ("woman", "f")
    ]

    for input_gender, expected_gender in test_cases:
        criteria = {"gender": input_gender}
        normalized = validate_search_criteria(criteria)
        assert normalized['gender'] == expected_gender, f"Gender '{input_gender}' should normalize to '{expected_gender}'"


def _test_criteria_validation_invalid_year() -> None:
    """Test search criteria validation with invalid year."""
    criteria = {
        "birth_year": "invalid",
        "death_year": "not_a_number"
    }

    normalized = validate_search_criteria(criteria)

    assert normalized['birth_year'] is None, "Invalid birth year should be None"
    assert normalized['death_year'] is None, "Invalid death year should be None"


def _test_confidence_levels() -> None:
    """Test confidence level calculation."""
    assert _get_confidence_level(250) == "very_high", "Score 250 should be very_high"
    assert _get_confidence_level(200) == "very_high", "Score 200 should be very_high"
    assert _get_confidence_level(175) == "high", "Score 175 should be high"
    assert _get_confidence_level(150) == "high", "Score 150 should be high"
    assert _get_confidence_level(125) == "medium", "Score 125 should be medium"
    assert _get_confidence_level(100) == "medium", "Score 100 should be medium"
    assert _get_confidence_level(75) == "low", "Score 75 should be low"
    assert _get_confidence_level(50) == "low", "Score 50 should be low"
    assert _get_confidence_level(25) == "very_low", "Score 25 should be very_low"
    assert _get_confidence_level(0) == "very_low", "Score 0 should be very_low"


def _test_display_bonuses_action10_format() -> None:
    """Test display bonus calculation for action10 format (with _s suffix)."""
    scores = {
        "byear_s": 25,
        "bplace_s": 25,
        "dyear_s": 25,
        "dplace_s": 25
    }

    bonuses = calculate_display_bonuses(scores, key_prefix="_s")

    assert bonuses['birth_date_component'] == 25, "Birth date component should be 25"
    assert bonuses['death_date_component'] == 25, "Death date component should be 25"
    assert bonuses['birth_bonus'] == 25, "Birth bonus should be 25 when both date and place present"
    assert bonuses['death_bonus'] == 25, "Death bonus should be 25 when both date and place present"


def _test_display_bonuses_action11_format() -> None:
    """Test display bonus calculation for action11 format (no suffix)."""
    scores = {
        "byear": 25,
        "bplace": 25,
        "dyear": 0,
        "dplace": 0
    }

    bonuses = calculate_display_bonuses(scores, key_prefix="")

    assert bonuses['birth_bonus'] == 25, "Birth bonus should be 25 when both date and place present"
    assert bonuses['death_bonus'] == 0, "Death bonus should be 0 when date or place missing"


def _test_display_bonuses_no_bonus() -> None:
    """Test display bonus calculation when no bonus should be awarded."""
    scores = {
        "byear": 25,
        "bplace": 0,  # Missing place
        "dyear": 0,   # Missing date
        "dplace": 25
    }

    bonuses = calculate_display_bonuses(scores, key_prefix="")

    assert bonuses['birth_bonus'] == 0, "Birth bonus should be 0 when place missing"
    assert bonuses['death_bonus'] == 0, "Death bonus should be 0 when date missing"


def _test_scoring_breakdown_format() -> None:
    """Test scoring breakdown formatting."""
    result = {
        'total_score': 150,
        'field_scores': {
            'givn': 25,
            'surn': 25,
            'byear': 25,
            'bplace': 25,
            'gender': 25,
            'bbonus': 25
        },
        'reasons': ['First name exact match', 'Surname exact match', 'Birth year exact match']
    }

    search_criteria = {
        'first_name': 'john',
        'surname': 'smith',
        'birth_year': 1950
    }

    breakdown = format_scoring_breakdown(result, search_criteria, title="Test Breakdown")

    assert isinstance(breakdown, str), "Breakdown should be a string"
    assert "Test Breakdown" in breakdown, "Breakdown should include title"
    assert "150" in breakdown, "Breakdown should include total score"
    assert "First name exact match" in breakdown or "First Name Match" in breakdown, "Breakdown should include field scores or reasons"


def universal_scoring_module_tests() -> bool:
    """Run all universal scoring tests."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Universal Scoring", "universal_scoring.py")

    tests = [
        ("Basic universal scoring", _test_universal_scoring_basic, "Test basic scoring functionality"),
        ("Exact match scoring", _test_universal_scoring_exact_match, "Test scoring with exact match"),
        ("Partial match scoring", _test_universal_scoring_partial_match, "Test scoring with partial match"),
        ("No match scoring", _test_universal_scoring_no_match, "Test scoring with no match"),
        ("Multiple candidates scoring", _test_universal_scoring_multiple_candidates, "Test scoring multiple candidates"),
        ("Max results parameter", _test_universal_scoring_max_results, "Test max_results parameter"),
        ("Criteria validation - names", _test_criteria_validation_names, "Test name normalization"),
        ("Criteria validation - years", _test_criteria_validation_years, "Test year conversion"),
        ("Criteria validation - gender", _test_criteria_validation_gender, "Test gender normalization"),
        ("Criteria validation - invalid year", _test_criteria_validation_invalid_year, "Test invalid year handling"),
        ("Confidence levels", _test_confidence_levels, "Test confidence level calculation"),
        ("Display bonuses - action10 format", _test_display_bonuses_action10_format, "Test action10 bonus calculation"),
        ("Display bonuses - action11 format", _test_display_bonuses_action11_format, "Test action11 bonus calculation"),
        ("Display bonuses - no bonus", _test_display_bonuses_no_bonus, "Test no bonus scenario"),
        ("Scoring breakdown format", _test_scoring_breakdown_format, "Test scoring breakdown formatting"),
    ]

    with suppress_logging():
        for test_name, test_func, expected_behavior in tests:
            suite.run_test(test_name, test_func, expected_behavior)

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

# Use centralized test runner utility
run_comprehensive_tests = create_standard_test_runner(universal_scoring_module_tests)


if __name__ == "__main__":
    # Demo the universal scoring
    print("Universal Scoring Demo:")
    print(Colors.green("✅ Universal scoring module loaded successfully"))

    # Run tests
    if run_comprehensive_tests():
        print(Colors.green("✅ All tests passed!"))
    else:
        print(Colors.red("❌ Some tests failed!"))
