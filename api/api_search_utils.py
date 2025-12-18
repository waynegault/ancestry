#!/usr/bin/env python3

"""
API Search Utilities - Ancestry API Search and Retrieval

Provides comprehensive Ancestry API search capabilities with person and family
information retrieval, relationship path analysis, and intelligent search
criteria matching for genealogical research and family tree analysis.
"""

# TESTING:
# Run `python api_search_utils.py` to execute the comprehensive test suite.
# All tests pass with 100% success rate.

# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# === CORE INFRASTRUCTURE ===
import logging

from core.registry_utils import auto_register_module

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# Imports removed - not used in this module

auto_register_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import re
from typing import Any, Callable, Optional, Union, cast

# === THIRD-PARTY IMPORTS ===
# (none currently needed)
# === LOCAL IMPORTS ===
from api.api_utils import (
    call_getladder_api,
    call_suggest_api,
    call_treesui_list_api,
)
from config import config_schema
from core.session_manager import SessionManager
from research.relationship_utils import format_api_relationship_path
from testing.test_utilities import create_standard_test_runner

# === MODULE LOGGER ===
logger = logging.getLogger(__name__)

# === MODULE CONSTANTS ===
API_UTILS_AVAILABLE = True
RELATIONSHIP_UTILS_AVAILABLE = True


def _extract_year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Extract year from a date string."""
    if not date_str or date_str == "Unknown":
        return None

    # Try to extract a 4-digit year
    year_match = re.search(r"\b(\d{4})\b", date_str)
    if year_match:
        try:
            return int(year_match.group(1))
        except ValueError:
            pass

    return None


# Helper functions for _run_simple_suggestion_scoring


def _get_scoring_weights(weights: Optional[dict[str, Union[int, float]]]) -> dict[str, Union[int, float]]:
    """Get scoring weights with defaults."""
    if weights is None:
        return dict(config_schema.common_scoring_weights)
    return weights


def _get_year_range(date_flex: Optional[dict[str, Any]]) -> int:
    """Get year range for flexible matching."""
    if date_flex:
        return date_flex.get("year_match_range", 10)
    return 10


def _extract_search_criteria(search_criteria: dict[str, Any]) -> dict[str, Any]:
    """Extract and clean search criteria."""

    def clean_param(p: Any) -> Optional[str]:
        return p.strip().lower() if p and isinstance(p, str) else None

    return {
        "first_name": clean_param(search_criteria.get("first_name")),
        "surname": clean_param(search_criteria.get("surname")),
        "birth_year": search_criteria.get("birth_year"),
        "birth_place": clean_param(search_criteria.get("birth_place")),
        "death_year": search_criteria.get("death_year"),
        "death_place": clean_param(search_criteria.get("death_place")),
    }


def _extract_candidate_data(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extract and clean candidate data - handle both camelCase and Title Case field names."""

    def clean_param(p: Any) -> Optional[str]:
        return p.strip().lower() if p and isinstance(p, str) else None

    return {
        "first_name": clean_param(candidate.get("first_name", candidate.get("firstName", candidate.get("First Name")))),
        "surname": clean_param(candidate.get("surname", candidate.get("lastName", candidate.get("Surname")))),
        "birth_year": candidate.get("birth_year", candidate.get("birthYear", candidate.get("Birth Year"))),
        "birth_place": clean_param(
            candidate.get("birth_place", candidate.get("birthPlace", candidate.get("Birth Place")))
        ),
        "death_year": candidate.get("death_year", candidate.get("deathYear", candidate.get("Death Year"))),
        "death_place": clean_param(
            candidate.get("death_place", candidate.get("deathPlace", candidate.get("Death Place")))
        ),
    }


def _places_requirements_satisfied(search_criteria: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Require birth/death place presence and match if provided in search criteria.

    - If search includes birth_place, candidate must have a birth_place and it must contain the search value (case-insensitive).
    - If search includes death_place, candidate must have a death_place and it must contain the search value (case-insensitive).
    """
    search = _extract_search_criteria(search_criteria)
    cand = _extract_candidate_data(candidate)

    # Evaluate violations for birth/death place when provided
    violates_birth = bool(search.get("birth_place")) and (
        not cand.get("birth_place") or search["birth_place"] not in cand["birth_place"]
    )
    violates_death = bool(search.get("death_place")) and (
        not cand.get("death_place") or search["death_place"] not in cand["death_place"]
    )
    return not (violates_birth or violates_death)


def _score_name_match(
    search_name: Optional[str],
    cand_name: Optional[str],
    field_name: str,
    score_value: Union[int, float],
    total_score: int,
    field_scores: dict[str, int],
    reasons: list[str],
) -> int:
    """Score name matching (first name or surname)."""
    if search_name and cand_name and search_name in cand_name:
        score_int = int(score_value)
        total_score += score_int
        field_scores[field_name] = score_int
        reasons.append(f"{field_name.replace('_', ' ').title()} '{search_name}' found in '{cand_name}'")
    return total_score


def _score_year_match(
    search_year: Any,
    cand_year: Any,
    field_name: str,
    exact_score: Union[int, float],
    close_score: Union[int, float],
    year_range: int,
    total_score: int,
    field_scores: dict[str, int],
    reasons: list[str],
) -> int:
    """Score year matching (birth or death year)."""
    if search_year and cand_year:
        try:
            search_year_int = int(search_year)
            cand_year_int = int(cand_year)

            if search_year_int == cand_year_int:
                exact_int = int(exact_score)
                total_score += exact_int
                field_scores[field_name] = exact_int
                reasons.append(f"{field_name.replace('_', ' ').title()} {search_year} matched exactly")
            elif abs(search_year_int - cand_year_int) <= year_range:
                close_int = int(close_score)
                total_score += close_int
                field_scores[field_name] = close_int
                reasons.append(f"{field_name.replace('_', ' ').title()} {search_year} close to {cand_year}")
        except (ValueError, TypeError) as e:
            logger.debug(
                f"{field_name} comparison failed - search: {search_year} ({type(search_year)}), candidate: {cand_year} ({type(cand_year)}), error: {e}"
            )
    return total_score


def _score_place_match(
    search_place: Optional[str],
    cand_place: Optional[str],
    field_name: str,
    score_value: Union[int, float],
    total_score: int,
    field_scores: dict[str, int],
    reasons: list[str],
) -> int:
    """Score place matching (birth or death place)."""
    if search_place and cand_place and search_place in cand_place:
        score_int = int(score_value)
        total_score += score_int
        field_scores[field_name] = score_int
        reasons.append(f"{field_name.replace('_', ' ').title()} '{search_place}' found in '{cand_place}'")
    return total_score


def _apply_bonus_scores(
    field_scores: dict[str, int], weights: dict[str, Union[int, float]], total_score: int, reasons: list[str]
) -> int:
    """Apply bonus scores for multiple matching fields."""
    # Bonus for both names matching
    if "first_name" in field_scores and "surname" in field_scores:
        bonus = int(weights.get("bonus_both_names_contain", 25))
        total_score += bonus
        field_scores["name_bonus"] = bonus
        reasons.append("Both first name and surname matched")

    # Bonus for both birth date and place matching
    if "birth_year" in field_scores and "birth_place" in field_scores:
        bonus = int(weights.get("bonus_birth_date_and_place", 15))
        total_score += bonus
        field_scores["birth_bonus"] = bonus
        reasons.append("Both birth year and place matched")

    # Bonus for both death date and place matching
    if "death_year" in field_scores and "death_place" in field_scores:
        bonus = int(weights.get("bonus_death_date_and_place", 15))
        total_score += bonus
        field_scores["death_bonus"] = bonus
        reasons.append("Both death year and place matched")

    return total_score


def _run_simple_suggestion_scoring(
    search_criteria: dict[str, Any],
    candidate: dict[str, Any],
    weights: Optional[dict[str, Union[int, float]]] = None,
    date_flex: Optional[dict[str, Any]] = None,
) -> tuple[int, dict[str, int], list[str]]:
    """
    Simple scoring function for API suggestions when gedcom_utils is not available.

    Args:
        search_criteria: Dictionary of search criteria
        candidate: Dictionary of candidate data
        weights: Optional dictionary of scoring weights
        date_flex: Optional dictionary of date flexibility settings

    Returns:
        Tuple of (total_score, field_scores, reasons)
    """
    # Get weights and year range
    weights = _get_scoring_weights(weights)
    if date_flex is None:
        date_flex = {"year_match_range": config_schema.date_flexibility}
    year_range = _get_year_range(date_flex)

    # Extract search criteria and candidate data
    search = _extract_search_criteria(search_criteria)
    cand = _extract_candidate_data(candidate)

    # Initialize scoring variables
    total_score = 0
    field_scores = {}
    reasons = []

    # Score name matches
    total_score = _score_name_match(
        search["first_name"],
        cand["first_name"],
        "first_name",
        weights.get("contains_first_name", 25),
        total_score,
        field_scores,
        reasons,
    )
    total_score = _score_name_match(
        search["surname"],
        cand["surname"],
        "surname",
        weights.get("contains_surname", 25),
        total_score,
        field_scores,
        reasons,
    )

    # Score birth year and place
    total_score = _score_year_match(
        search["birth_year"],
        cand["birth_year"],
        "birth_year",
        weights.get("birth_year_match", 20),
        weights.get("birth_year_close", 10),
        year_range,
        total_score,
        field_scores,
        reasons,
    )
    total_score = _score_place_match(
        search["birth_place"],
        cand["birth_place"],
        "birth_place",
        weights.get("birth_place_match", 20),
        total_score,
        field_scores,
        reasons,
    )

    # Score death year and place
    total_score = _score_year_match(
        search["death_year"],
        cand["death_year"],
        "death_year",
        weights.get("death_year_match", 20),
        weights.get("death_year_close", 10),
        year_range,
        total_score,
        field_scores,
        reasons,
    )
    total_score = _score_place_match(
        search["death_place"],
        cand["death_place"],
        "death_place",
        weights.get("death_place_match", 20),
        total_score,
        field_scores,
        reasons,
    )

    # Apply bonus scores
    total_score = _apply_bonus_scores(field_scores, weights, total_score, reasons)

    return int(total_score), field_scores, reasons


# Helper functions for search_api_for_criteria


def _build_search_query(search_criteria: dict[str, Any]) -> str:
    """Build search query string from criteria."""
    search_query = ""
    if search_criteria.get("first_name"):
        search_query += search_criteria["first_name"] + " "
    if search_criteria.get("surname"):
        search_query += search_criteria["surname"] + " "
    if search_criteria.get("birth_year"):
        search_query += f"b. {search_criteria['birth_year']} "
    if search_criteria.get("birth_place"):
        search_query += search_criteria["birth_place"] + " "
    if search_criteria.get("death_year"):
        search_query += f"d. {search_criteria['death_year']} "
    if search_criteria.get("death_place"):
        search_query += search_criteria["death_place"] + " "
    return search_query.strip()


def _get_tree_id(session_manager: SessionManager) -> Optional[str]:
    """Get tree ID from session manager or config."""
    tree_id = session_manager.my_tree_id
    if not tree_id:
        tree_id = getattr(config_schema.test, "test_tree_id", "")

    return tree_id


def _parse_hyphenated_lifespan(lifespan: str) -> tuple[Optional[int], Optional[int]]:
    """Parse hyphenated lifespan format (e.g., '1850-1920')."""
    parts = lifespan.split("-")
    if len(parts) == 2:
        try:
            birth_year = int(parts[0].strip()) if parts[0].strip() else None
            death_year = int(parts[1].strip()) if parts[1].strip() else None
            return birth_year, death_year
        except ValueError:
            pass
    return None, None


def _parse_birth_notation(lifespan: str) -> Optional[int]:
    """Parse birth notation format (e.g., 'b. 1850')."""
    match = re.search(r"b\.\s*(\d{4})", lifespan.lower())
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def _parse_death_notation(lifespan: str) -> Optional[int]:
    """Parse death notation format (e.g., 'd. 1920')."""
    match = re.search(r"d\.\s*(\d{4})", lifespan.lower())
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def _parse_lifespan(lifespan: str) -> tuple[Optional[int], Optional[int]]:
    """Parse lifespan string to extract birth and death years."""
    if not lifespan:
        return None, None

    if "-" in lifespan:
        return _parse_hyphenated_lifespan(lifespan)
    if "b." in lifespan.lower():
        return _parse_birth_notation(lifespan), None
    if "d." in lifespan.lower():
        return None, _parse_death_notation(lifespan)

    return None, None


def _process_suggest_result(
    suggestion: dict[str, Any],
    search_criteria: dict[str, Any],
    scoring_weights: dict[str, Union[int, float]],
    date_flex: dict[str, Union[int, float]],
) -> Optional[dict[str, Any]]:
    """Process a single suggestion result and return match record if score > 0."""
    try:
        person_id = suggestion.get("id")
        if not person_id:
            return None

        # Extract name components
        full_name = suggestion.get("name", "")
        name_parts = full_name.split()
        first_name = name_parts[0] if name_parts else ""
        surname = name_parts[-1] if len(name_parts) > 1 else ""

        # Parse lifespan
        lifespan = suggestion.get("lifespan", "")
        birth_year, death_year = _parse_lifespan(lifespan)

        # Extract location
        location = suggestion.get("location", "")

        # Create candidate data for scoring
        candidate = {
            "first_name": first_name,
            "surname": surname,
            "birth_year": birth_year,
            "death_year": death_year,
            "birth_place": location,
            "death_place": None,
            "gender": None,
        }

        # Score the candidate
        total_score, field_scores, reasons = _run_simple_suggestion_scoring(
            search_criteria, candidate, scoring_weights, date_flex
        )

        # Only include if score is above threshold and place requirements satisfied when provided
        allowed = _places_requirements_satisfied(search_criteria, candidate)
        if total_score > 0 and allowed:
            return {
                "id": person_id,
                "display_id": person_id,
                "first_name": first_name,
                "surname": surname,
                "gender": None,
                "birth_year": birth_year,
                "birth_place": location,
                "death_year": death_year,
                "death_place": None,
                "total_score": total_score,
                "field_scores": field_scores,
                "reasons": reasons,
                "source": "API",
            }
        return None
    except Exception as e:
        logger.error(f"Error processing suggestion result: {e}")
        return None


def _extract_event_details(event_data: dict[str, Any]) -> tuple[Optional[int], str]:
    """Extract normalized year and place from an event dictionary."""
    date_value = event_data.get("date", {}).get("normalized", "")
    place_value = event_data.get("place", {}).get("normalized", "")
    return _extract_year_from_date(date_value), place_value


def _process_treesui_person(
    person: dict[str, Any],
    search_criteria: dict[str, Any],
    scoring_weights: dict[str, Union[int, float]],
    date_flex: dict[str, Union[int, float]],
    scored_matches: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Process a single treesui-list person and return match record if score > 0 and not duplicate."""
    try:
        person_id = person.get("id")
        if not person_id:
            return None

        # Extract name components
        first_name = person.get("firstName", "")
        surname = person.get("lastName", "")

        # Extract birth/death details
        birth_year, birth_place = _extract_event_details(person.get("birth", {}))
        death_year, death_place = _extract_event_details(person.get("death", {}))

        # Extract gender
        gender = person.get("gender", "")
        if gender == "Male":
            gender = "M"
        elif gender == "Female":
            gender = "F"

        # Create candidate data for scoring
        candidate = {
            "first_name": first_name,
            "surname": surname,
            "birth_year": birth_year,
            "death_year": death_year,
            "birth_place": birth_place,
            "death_place": death_place,
        }

        # Score the candidate
        total_score, field_scores, reasons = _run_simple_suggestion_scoring(
            search_criteria, candidate, scoring_weights, date_flex
        )

        # Only include if score is above threshold, place requirements satisfied, and not duplicate
        allowed = _places_requirements_satisfied(search_criteria, candidate)
        if total_score > 0 and allowed and not any(match["id"] == person_id for match in scored_matches):
            return {
                "id": person_id,
                "display_id": person_id,
                "first_name": first_name,
                "surname": surname,
                "birth_year": birth_year,
                "birth_place": birth_place,
                "death_year": death_year,
                "death_place": death_place,
                "total_score": total_score,
                "field_scores": field_scores,
                "reasons": reasons,
                "source": "API",
            }
        return None
    except Exception as e:
        logger.error(f"Error processing treesui-list result: {e}")
        return None


# Helper functions for search_api_for_criteria


def _validate_session(session_manager: SessionManager) -> bool:
    """Validate session manager state for API access (browserless capable)."""
    try:
        # Prefer API identifiers readiness over browser session validity
        if not getattr(session_manager.api_manager, "has_essential_identifiers", False):
            logger.error("Essential API identifiers not available (not logged in)")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking session validity: {e}")
        return False


def _call_suggest_api_for_search(
    session_manager: SessionManager, search_criteria: dict[str, Any], tree_id: str
) -> list[dict[str, Any]]:
    """Call suggest API and return results."""
    base_url = config_schema.api.base_url

    api_search_criteria = {
        "first_name_raw": search_criteria.get("first_name", ""),
        "surname_raw": search_criteria.get("surname", ""),
        "birth_year": search_criteria.get("birth_year"),
    }

    suggest_results = call_suggest_api(
        session_manager=session_manager,
        owner_tree_id=tree_id,
        base_url=base_url,
        search_criteria=api_search_criteria,
    )

    if not suggest_results:
        logger.warning(f"No results from suggest API for query: {_build_search_query(search_criteria)}")
        return []

    return suggest_results


def _process_suggest_results(
    suggest_results: list[dict[str, Any]], search_criteria: dict[str, Any], max_suggestions: int
) -> list[dict[str, Any]]:
    """Process suggest API results and return scored matches."""
    scoring_weights = config_schema.common_scoring_weights
    date_flex = {"year_match_range": config_schema.date_flexibility}
    scored_matches: list[dict[str, Any]] = []

    for suggestion in suggest_results[:max_suggestions]:
        match_record = _process_suggest_result(suggestion, search_criteria, scoring_weights, date_flex)
        if match_record:
            scored_matches.append(match_record)

    return scored_matches


def _build_treesui_search_params(search_criteria: dict[str, Any]) -> dict[str, Any]:
    """Build search parameters for treesui-list API."""
    search_params = {}

    field_mapping = {
        "first_name": "firstName",
        "surname": "lastName",
        "birth_year": "birthYear",
        "birth_place": "birthLocation",
        "death_year": "deathYear",
        "death_place": "deathLocation",
    }

    for criteria_key, api_key in field_mapping.items():
        if search_criteria.get(criteria_key):
            search_params[api_key] = search_criteria[criteria_key]

    return search_params


def _call_treesui_api_for_search(
    session_manager: SessionManager, search_params: dict[str, Any], tree_id: str
) -> Optional[list[dict[str, Any]]]:
    """Call treesui-list API and return results."""
    if not search_params:
        return None

    logger.info(f"Calling treesui-list API with params: {search_params}")
    base_url = config_schema.api.base_url

    return call_treesui_list_api(
        session_manager=session_manager,
        owner_tree_id=tree_id,
        base_url=base_url,
        search_criteria=search_params,
    )


def _process_treesui_results(
    treesui_results: Optional[list[dict[str, Any]]],
    search_criteria: dict[str, Any],
    scored_matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Process treesui-list API results and add to scored matches."""
    if not treesui_results:
        return scored_matches

    scoring_weights = config_schema.common_scoring_weights
    date_flex = {"year_match_range": config_schema.date_flexibility}

    for person in treesui_results:
        match_record = _process_treesui_person(person, search_criteria, scoring_weights, date_flex, scored_matches)
        if match_record:
            scored_matches.append(match_record)

    return scored_matches


def search_api_for_criteria(
    session_manager: SessionManager,
    search_criteria: dict[str, Any],
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """
    Search Ancestry API for individuals matching the given criteria.

    Args:
        session_manager: SessionManager instance with active session
        search_criteria: Dictionary of search criteria (first_name, surname, gender, birth_year, etc.)
        max_results: Maximum number of results to return (default: 10)

    Returns:
        List of dictionaries containing match information, sorted by score (highest first)
    """
    # Validate session
    if not _validate_session(session_manager):
        return []

    # Prepare search parameters
    search_query = _build_search_query(search_criteria)
    if not search_query:
        logger.error("No search criteria provided")
        return []

    logger.info(f"Searching API with query: {search_query}")

    # Get tree ID
    tree_id = _get_tree_id(session_manager)
    if not tree_id:
        logger.error("No tree ID available for API search")
        return []

    # Call suggest API and process results
    suggest_results = _call_suggest_api_for_search(session_manager, search_criteria, tree_id)
    max_suggestions = config_schema.max_suggestions_to_score
    scored_matches = _process_suggest_results(suggest_results, search_criteria, max_suggestions)

    # Try treesui-list API if not enough results
    if len(scored_matches) < max_results:
        try:
            search_params = _build_treesui_search_params(search_criteria)
            treesui_results = _call_treesui_api_for_search(session_manager, search_params, tree_id)
            scored_matches = _process_treesui_results(treesui_results, search_criteria, scored_matches)
        except Exception as e:
            logger.error(f"Error calling treesui-list API: {e}")

    # Sort and return top matches
    scored_matches.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    return scored_matches[:max_results] if scored_matches else []


# Helper functions for get_api_family_details


def _validate_api_session(session_manager: SessionManager) -> bool:
    """Validate that session manager is active and valid (browserless capable)."""
    if not getattr(session_manager.api_manager, "has_essential_identifiers", False):
        logger.error("Essential API identifiers not available (not logged in)")
        return False
    return True


def _resolve_tree_id(session_manager: SessionManager, tree_id: Optional[str]) -> Optional[str]:
    """Resolve tree ID from session manager or config."""
    if tree_id:
        return tree_id

    tree_id = session_manager.my_tree_id
    if tree_id:
        return tree_id

    tree_id = getattr(config_schema.test, "test_tree_id", "")
    if not tree_id:
        logger.error("No tree ID available for API family details")
        return None

    return tree_id


def _resolve_owner_profile_id(session_manager: SessionManager) -> str:
    """Resolve owner profile ID from session manager or config."""
    owner_profile_id = session_manager.my_profile_id
    if owner_profile_id:
        return owner_profile_id

    owner_profile_id = getattr(config_schema.test, "test_profile_id", "")
    if not owner_profile_id:
        logger.warning("No owner profile ID available for API family details")

    return owner_profile_id


def _get_facts_data_from_api(
    session_manager: SessionManager,
    person_id: str,
    tree_id: str,
    owner_profile_id: str,
) -> Optional[dict[str, Any]]:
    """
    Call Edit Relationships API and return family relationship data.

    This uses the /family-tree/person/addedit/user/{user_id}/tree/{tree_id}/person/{person_id}/editrelationships
    endpoint which returns fathers[], mothers[], spouses[], and children[][] arrays.

    Note: The API returns a 'data' field containing a JSON STRING (not an object) that must be parsed.
    """
    logger.debug(f"Getting family relationship data for person {person_id} in tree {tree_id}")

    # Import the API function and json module
    import json

    from api.api_utils import call_edit_relationships_api

    # Call the Edit Relationships API
    api_response: Optional[dict[str, Any]] = call_edit_relationships_api(
        session_manager=session_manager,
        user_id=owner_profile_id,
        tree_id=tree_id,
        person_id=person_id,
    )

    if api_response is None:
        logger.warning(f"No data returned from Edit Relationships API for person {person_id}")
        return None

    # DEBUG: Summarize API response structure (concise)
    try:
        top_keys = list(api_response.keys())
        logger.debug(
            f"Edit Relationships API response summary for person {person_id}: type={type(api_response).__name__}, keys={top_keys}"
        )
        if isinstance(api_response.get("res"), dict):
            logger.debug(f"'res' keys: {list(api_response['res'].keys())}")
    except Exception:
        logger.debug("Edit Relationships API response summary unavailable (logging error)")
    # The API can return data in different formats:
    # Format 1: {"data": "JSON_STRING_HERE"} - data is a JSON string
    # Format 2: {"userId": "...", "treeId": "...", "res": {...}} - res contains the data
    # Try both formats

    data_str = api_response.get("data")
    if data_str and isinstance(data_str, str):
        # Format 1: Parse the JSON string
        try:
            family_data = json.loads(data_str)
            logger.debug(f"Successfully parsed family data from 'data' field for person {person_id}")
            logger.debug(
                f"Family data keys: {list(family_data.keys()) if isinstance(family_data, dict) else 'Not a dict'}"
            )
            return family_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Edit Relationships API 'data' field: {e}")
            logger.debug(f"Data string (first 200 chars): {data_str[:200]}")
            return None

    # Format 2: Check for 'res' field (THIS IS THE CORRECT FORMAT)
    # The Edit Relationships API returns: {userId, treeId, personId, person, urls, res}
    # The 'res' field contains the actual family data: {fathers[], mothers[], spouses[], children[][]}
    res_data = api_response.get("res")
    if res_data and isinstance(res_data, dict):
        logger.debug(f"Using 'res' field from Edit Relationships API for person {person_id}")
        logger.debug(f"Res data keys: {list(res_data.keys())}")
        # IMPORTANT: Return ONLY the res field, not the whole response
        # This is what the extraction logic expects
        return res_data

    # If neither format works, log the response structure and return None
    logger.warning(f"Edit Relationships API returned unexpected format for person {person_id}")
    logger.debug(f"API response keys: {list(api_response.keys())}")
    return None


def _initialize_family_result(person_id: str) -> dict[str, Any]:
    """Initialize empty family details result structure."""
    return {
        "id": person_id,
        "name": "",
        "first_name": "",
        "surname": "",
        "gender": "",
        "birth_year": None,
        "birth_date": "Unknown",
        "birth_place": "Unknown",
        "death_year": None,
        "death_date": "Unknown",
        "death_place": "Unknown",
        "parents": [],
        "spouses": [],
        "children": [],
        "siblings": [],
    }


def _extract_person_info_from_target(target_person: dict[str, Any], result: dict[str, Any]) -> None:
    """
    Extract person information from targetPerson object in edit relationships API response.

    The targetPerson object has this structure:
    {
        "name": {"given": "Fraser", "surname": "Gault", "suffix": ""},
        "gender": "Male",
        "bDate": {"day": null, "month": null, "year": null},
        "dDate": {},
        "id": "102281560744",
        "isLiving": false,
        ...
    }
    """
    # Extract name
    name_obj = target_person.get("name", {})
    given = name_obj.get("given", "")
    surname = name_obj.get("surname", "")
    suffix = name_obj.get("suffix", "")

    result["first_name"] = given
    result["surname"] = surname
    result["name"] = f"{given} {surname}".strip()
    if suffix:
        result["name"] += f" {suffix}"

    # Extract gender
    result["gender"] = target_person.get("gender", "")

    # Extract birth date
    bdate = target_person.get("bDate", {})
    if bdate and isinstance(bdate, dict):
        bdate_dict = cast(dict[str, Any], bdate)
        year = bdate_dict.get("year")
        month = bdate_dict.get("month")
        day = bdate_dict.get("day")
        if year:
            result["birth_year"] = year
            result["birth_date"] = _format_date(day, month, year)

    # Extract death date
    ddate = target_person.get("dDate", {})
    if ddate and isinstance(ddate, dict):
        ddate_dict = cast(dict[str, Any], ddate)
        year = ddate_dict.get("year")
        month = ddate_dict.get("month")
        day = ddate_dict.get("day")
        if year:
            result["death_year"] = year
            result["death_date"] = _format_date(day, month, year)


def _format_person_from_relationship_data(person_obj: dict[str, Any]) -> dict[str, Any]:
    """
    Format a person object from the edit relationships API into a standardized dict.

    Person objects have this structure:
    {
        "name": {"given": "James", "surname": "Gault", "suffix": ""},
        "gender": "Male",
        "bDate": {"day": 26, "month": 3, "year": 1906},
        "dDate": {"day": 16, "month": 6, "year": 1988},
        "id": "102281560741",
        ...
    }

    Returns:
        Dict with keys: name, birth_year, death_year (matching display_family_members format)
    """
    # Extract name
    name_obj = person_obj.get("name", {})
    given = name_obj.get("given", "")
    surname = name_obj.get("surname", "")
    suffix = name_obj.get("suffix", "")

    name = f"{given} {surname}".strip()
    if suffix:
        name += f" {suffix}"

    # Extract dates
    bdate = person_obj.get("bDate", {})
    ddate = person_obj.get("dDate", {})

    birth_year = cast(dict[str, Any], bdate).get("year") if isinstance(bdate, dict) else None
    death_year = cast(dict[str, Any], ddate).get("year") if isinstance(ddate, dict) else None

    # Return standardized dict format (matches display_family_members expectations)
    return {
        "name": name,
        "birth_year": birth_year,
        "death_year": death_year,
    }


def _format_date(day: Optional[int], month: Optional[int], year: Optional[int]) -> str:
    """Format a date from day/month/year components."""
    if not year:
        return "Unknown"

    if month and day:
        # Convert month number to name (1=Jan, 2=Feb, etc.)
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_name = month_names[month] if 1 <= month <= 12 else str(month)
        return f"{day} {month_name} {year}"
    if month:
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_name = month_names[month] if 1 <= month <= 12 else str(month)
        return f"{month_name} {year}"
    return str(year)


def _debug_log_facts_structure(facts_data: Any) -> None:
    """Log top-level structure of the facts data for debugging."""
    if not isinstance(facts_data, dict):
        logger.debug("Family data is not a dict; skipping structure log")
        return
    logger.debug(f"Family data keys: {list(facts_data.keys())}")
    for key, value in facts_data.items():
        if isinstance(value, dict):
            logger.debug(f"  {key}: dict with keys {list(value.keys())[:10]}")
        elif isinstance(value, list):
            logger.debug(f"  {key}: list with {len(value)} items")
        else:
            logger.debug(f"  {key}: {type(value).__name__}")


def _get_data_section_from_facts(facts_data: Any) -> dict[str, Any]:
    """Return the primary data section containing relationship arrays."""
    if isinstance(facts_data, dict):
        facts_dict = cast(dict[str, Any], facts_data)
        person_section = facts_dict.get("person")
        if isinstance(person_section, dict):
            return person_section
        return facts_dict
    return {}


def _extract_target_person_info_if_available(data_section: dict[str, Any], result: dict[str, Any]) -> None:
    """Extract target person's own info if present."""
    try:
        target_person = data_section.get("targetPerson")
        if isinstance(target_person, dict):
            _extract_person_info_from_target(target_person, result)
    except Exception:
        # Best-effort extraction; safe to ignore errors
        pass


def _extract_parents_from_data_section(data_section: dict[str, Any], result: dict[str, Any]) -> None:
    """Append parents (fathers/mothers) to result['parents']."""
    fathers = data_section.get("fathers", [])
    for father in fathers:
        if isinstance(father, dict):
            result["parents"].append(_format_person_from_relationship_data(father))

    mothers = data_section.get("mothers", [])
    for mother in mothers:
        if isinstance(mother, dict):
            result["parents"].append(_format_person_from_relationship_data(mother))


def _extract_spouses_from_data_section(data_section: dict[str, Any], result: dict[str, Any]) -> None:
    """Append spouses to result['spouses']."""
    spouses = data_section.get("spouses", [])
    for spouse in spouses:
        if isinstance(spouse, dict):
            result["spouses"].append(_format_person_from_relationship_data(spouse))


def _extract_children_from_data_section(data_section: dict[str, Any], result: dict[str, Any]) -> None:
    """Append children (flattened) to result['children']."""
    children_field = data_section.get("children", [])
    if not isinstance(children_field, list):
        return
    for item in children_field:
        if isinstance(item, list):
            for child in item:
                if isinstance(child, dict):
                    result["children"].append(_format_person_from_relationship_data(child))
        elif isinstance(item, dict):
            result["children"].append(_format_person_from_relationship_data(item))


# Keep legacy relationship helpers reachable for diagnostic scripts and tests.
LEGACY_FAMILY_DETAIL_HELPERS: tuple[Callable[..., Any], ...] = (
    _validate_api_session,
    _resolve_tree_id,
    _resolve_owner_profile_id,
    _get_facts_data_from_api,
    _debug_log_facts_structure,
    _get_data_section_from_facts,
    _extract_target_person_info_if_available,
    _extract_parents_from_data_section,
    _extract_spouses_from_data_section,
    _extract_children_from_data_section,
)


def _find_target_person_in_list(persons: list[Any], person_id: str) -> Optional[dict[str, Any]]:
    """Find target person in persons list by matching person_id in gid."""
    for person in persons:
        person_gid = person.get("gid", {}).get("v", "")
        if person_gid.startswith(f"{person_id}:"):
            return person
    return None


def _create_persons_lookup(persons: list[Any]) -> dict[str, dict[str, Any]]:
    """Create lookup dictionary mapping gid to person dict."""
    persons_by_gid = {}
    for person in persons:
        gid = person.get("gid", {}).get("v", "")
        if gid:
            persons_by_gid[gid] = person
    return persons_by_gid


def _extract_direct_family(
    family_relationships: list[Any],
    persons_by_gid: dict[str, dict[str, Any]],
    result: dict[str, list[dict[str, Any]]],
) -> None:
    """Extract parents, spouses, and children from family relationships."""
    for rel in family_relationships:
        rel_type = rel.get("t", "")
        target_gid = rel.get("tgid", {}).get("v", "")

        if not target_gid or target_gid not in persons_by_gid:
            continue

        related_person = persons_by_gid[target_gid]
        person_dict = _parse_person_from_newfamilyview(related_person)

        if rel_type in {"F", "M"}:  # Father or Mother
            result["parents"].append(person_dict)
        elif rel_type in {"W", "H"}:  # Wife or Husband
            result["spouses"].append(person_dict)
        elif rel_type == "C":  # Child
            result["children"].append(person_dict)


def _extract_siblings(
    family_relationships: list[Any],
    persons_by_gid: dict[str, dict[str, Any]],
    target_person_gid: str,
    result: dict[str, list[dict[str, Any]]],
) -> None:
    """Extract siblings by finding parents and getting their children."""
    parent_gids = [rel.get("tgid", {}).get("v", "") for rel in family_relationships if rel.get("t") in {"F", "M"}]

    for parent_gid in parent_gids:
        if not parent_gid or parent_gid not in persons_by_gid:
            continue

        parent = persons_by_gid[parent_gid]
        parent_family = parent.get("Family", [])

        for parent_rel in parent_family:
            if parent_rel.get("t") != "C":  # Not a child
                continue

            sibling_gid = parent_rel.get("tgid", {}).get("v", "")
            if not sibling_gid or sibling_gid == target_person_gid:
                continue

            if sibling_gid not in persons_by_gid:
                continue

            sibling_dict = _parse_person_from_newfamilyview(persons_by_gid[sibling_gid])

            # Avoid duplicates
            if not any(s.get("id") == sibling_dict.get("id") for s in result["siblings"]):
                result["siblings"].append(sibling_dict)


def get_api_family_details(
    session_manager: SessionManager,
    person_id: str,
    tree_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get family details for a specific individual from Ancestry New Family View API.

    Returns a dict with keys: parents, spouses, children, siblings.
    """
    from api.api_utils import call_newfamilyview_api

    # Validate session and resolve identifiers
    if not session_manager or not hasattr(session_manager, "driver"):
        logger.error("Invalid session manager provided")
        return _initialize_family_result(person_id)

    tree_id = tree_id or getattr(session_manager, "my_tree_id", None)
    if not tree_id:
        logger.error("No tree_id available for family details")
        return _initialize_family_result(person_id)

    base_url = getattr(config_schema.api, "base_url", "https://www.ancestry.co.uk")

    # Call New Family View API
    family_data = call_newfamilyview_api(session_manager, tree_id, person_id, base_url, timeout=20)
    if not family_data or "Persons" not in family_data:
        logger.warning(f"No family data returned from New Family View API for person {person_id}")
        return _initialize_family_result(person_id)

    # Parse the response
    result = _initialize_family_result(person_id)
    try:
        persons = family_data.get("Persons", [])
        logger.debug(f"New Family View API returned {len(persons)} persons")

        target_person = _find_target_person_in_list(persons, person_id)
        if not target_person:
            logger.warning(f"Target person {person_id} not found in New Family View response")
            return result

        family_relationships = target_person.get("Family", [])
        persons_by_gid = _create_persons_lookup(persons)

        # Extract direct family (parents, spouses, children)
        _extract_direct_family(family_relationships, persons_by_gid, result)

        # Extract siblings
        target_person_gid = target_person.get("gid", {}).get("v", "")
        _extract_siblings(family_relationships, persons_by_gid, target_person_gid, result)

        logger.debug(
            f"Extracted family: {len(result['parents'])} parents, "
            f"{len(result['siblings'])} siblings, {len(result['spouses'])} spouses, {len(result['children'])} children"
        )
    except Exception as e:
        logger.error(f"Error extracting family details from New Family View API data: {e}", exc_info=True)

    return result


def _extract_person_id_from_gid(gid_dict: dict[str, Any]) -> str:
    """Extract person ID from gid dictionary."""
    gid = gid_dict.get("v", "")
    return gid.split(":")[0] if ":" in gid else "Unknown"


def _extract_full_name_from_names(names_list: list[Any]) -> str:
    """Extract full name from Names array."""
    if not names_list:
        return "Unknown"
    given_name = names_list[0].get("g", "")
    surname = names_list[0].get("s", "")
    return f"{given_name} {surname}".strip() if given_name or surname else "Unknown"


def _extract_year_from_event_type(events: list[Any], event_type: str) -> Optional[int]:
    """Extract year from specific event type (Birth or Death)."""
    for event in events:
        if event.get("t") == event_type:
            nd = event.get("nd", "")
            if nd and isinstance(nd, str):
                parts = nd.split("-")
                if parts and parts[0].isdigit():
                    return int(parts[0])
            break
    return None


def _parse_person_from_newfamilyview(person: dict[str, Any]) -> dict[str, Any]:
    """Parse a person dict from New Family View API response into standard format."""
    person_id = _extract_person_id_from_gid(person.get("gid", {}))
    full_name = _extract_full_name_from_names(person.get("Names", []))
    events = person.get("Events", [])
    birth_year = _extract_year_from_event_type(events, "Birth")
    death_year = _extract_year_from_event_type(events, "Death")

    return {
        "id": person_id,
        "name": full_name,
        "birth_year": birth_year,
        "death_year": death_year,
    }


def _get_tree_id_for_relationship(session_manager: SessionManager, tree_id: Optional[str]) -> Optional[str]:
    """Get tree ID from session manager or config."""
    if tree_id:
        return tree_id

    tree_id = session_manager.my_tree_id
    if tree_id:
        return tree_id

    tree_id = getattr(config_schema.test, "test_tree_id", "")
    return tree_id if tree_id else None


def _get_reference_id_for_relationship(reference_id: Optional[str]) -> Optional[str]:
    """Get reference ID from parameter or config."""
    if reference_id:
        return reference_id

    reference_id = config_schema.reference_person_id
    return reference_id if reference_id else None


# =============================================================================
# DEAD CODE - Commented out 2025-12-18 (Technical Debt)
# Reason: Function defined but never called in production
# See: todo.md "Technical Debt" section
# =============================================================================
# def get_api_relationship_path(
#     session_manager: SessionManager,
#     person_id: str,
#     reference_id: Optional[str] = None,
#     reference_name: Optional[str] = "Reference Person",
#     tree_id: Optional[str] = None,
# ) -> str:
#     """
#     Get the relationship path between an individual and the reference person using Ancestry API.
#
#     Args:
#         session_manager: SessionManager instance with active session
#         person_id: Ancestry API person ID
#         reference_id: Optional reference person ID (default: from config)
#         reference_name: Optional reference person name (default: "Reference Person")
#         tree_id: Optional tree ID (default: from session_manager or config)
#
#     Returns:
#         Formatted relationship path string
#     """
#     # Step 1: Check if session has identifiers (browserless capable)
#     if not getattr(session_manager.api_manager, "has_essential_identifiers", False):
#         logger.error("Essential API identifiers not available (not logged in)")
#         return "(Session not valid)"
#
#     # Step 2: Get tree ID
#     tree_id = _get_tree_id_for_relationship(session_manager, tree_id)
#     if not tree_id:
#         logger.error("No tree ID available for API relationship path")
#         return "(Tree ID not available)"
#
#     # Step 3: Get reference ID
#     reference_id = _get_reference_id_for_relationship(reference_id)
#     if not reference_id:
#         logger.error("Reference person ID not provided and not found in config")
#         return "(Reference person ID not available)"
#
#     # Step 4: Get base URL
#     base_url = config_schema.api.base_url
#
#     # Step 5: Call the getladder API to get relationship path
#     logger.info(f"Getting relationship path from {person_id} to {reference_id} in tree {tree_id}")
#     ladder_data = call_getladder_api(
#         session_manager=session_manager,
#         owner_tree_id=tree_id,
#         target_person_id=person_id,
#         base_url=base_url,
#     )
#
#     if not ladder_data:
#         logger.warning(f"No ladder data returned for person {person_id}")
#         return f"(No relationship path found to {reference_name})"
#
#     try:
#         # Format the relationship path directly using the API formatter
#         return format_api_relationship_path(ladder_data, reference_name or "Reference Person", "Individual")
#     except Exception as e:
#         logger.error(f"Error formatting relationship path: {e}", exc_info=True)
#         return f"(Error formatting relationship path: {e!s})"
# =============================================================================


def _test_module_initialization() -> None:
    """Test module initialization and configuration."""
    # Test configuration access
    result = getattr(config_schema, "TEST_KEY_12345", "default_value")
    assert isinstance(result, str), "Should return string value"
    assert result == "default_value", "Should return default value for missing keys"

    # Ensure legacy helper registry remains populated for external diagnostics.
    assert LEGACY_FAMILY_DETAIL_HELPERS, "Legacy helper registry should not be empty"

    # Test configuration structure
    assert isinstance(config_schema.common_scoring_weights, dict), "common_scoring_weights should be a dictionary"
    assert "contains_first_name" in config_schema.common_scoring_weights, "Should have contains_first_name weight"
    assert isinstance(config_schema.date_flexibility, (int, float)), "Should have date_flexibility as number"


def _test_core_functionality() -> None:
    """Test all core API search and scoring functions."""
    # Test _extract_year_from_date function
    result = _extract_year_from_date("15 Jan 1985")
    assert result == 1985, "Should extract year from simple date"

    result = _extract_year_from_date("Born circa 1850, died 1920")
    assert result == 1850, "Should extract first year from complex date"

    result = _extract_year_from_date("no year here")
    assert result is None, "Should return None for dates without years"

    # Test _run_simple_suggestion_scoring function
    search_criteria = {
        "first_name": "John",
        "surname": "Smith",
        "birth_year": 1985,
        "birth_place": "New York",
    }
    suggestion = {
        "First Name": "John",
        "Surname": "Smith",
        "Birth Year": "1985",
        "Birth Place": "New York, NY",
    }

    score, field_scores, reasons = _run_simple_suggestion_scoring(search_criteria, suggestion)
    assert isinstance(score, (int, float)), "Should return numeric score"
    assert isinstance(field_scores, dict), "Should return field scores dictionary"
    assert isinstance(reasons, list), "Should return reasons list"
    assert score > 0, "Should have positive score for matching data"


def _test_edge_cases() -> None:
    """Test edge cases and boundary conditions."""
    # Test _extract_year_from_date with edge cases
    result = _extract_year_from_date("")
    assert result is None, "Should handle empty string"

    result = _extract_year_from_date(None)
    assert result is None, "Should handle None input"

    result = _extract_year_from_date("1800-2000")
    assert result == 1800, "Should extract first year from range"

    # Test scoring with empty data
    score, field_scores, _ = _run_simple_suggestion_scoring({}, {})
    assert score == 0, "Should return zero score for empty inputs"
    assert len(field_scores) == 0, "Should return empty field scores"


def _test_integration() -> None:
    """Test integration with mocked external dependencies."""
    import sys
    from unittest.mock import MagicMock, patch

    # Get reference to current module to patch functions imported into its namespace
    current_module = sys.modules[__name__]

    # Test search_api_for_criteria with mock session
    mock_session = MagicMock()
    mock_session.is_sess_valid.return_value = True
    # Ensure validation passes
    mock_session.api_manager.has_essential_identifiers = True
    # Ensure tree ID is a string
    mock_session.my_tree_id = "tree-123"

    with (
        patch.object(current_module, "call_suggest_api") as mock_suggest,
        patch.object(current_module, "call_treesui_list_api") as mock_treesui,
    ):
        mock_suggest.return_value = [{"First Name": "Test User 12345"}]
        mock_treesui.return_value = []

        result = search_api_for_criteria(mock_session, {"first_name": "Test", "surname": "User"})
        assert isinstance(result, list), "Should return list of results"


def _test_performance() -> None:
    """Test performance of scoring operations."""
    import time

    # Test multiple scoring operations
    start_time = time.time()
    for i in range(100):
        _run_simple_suggestion_scoring({"first_name": f"Test{i}_12345"}, {"First Name": f"Test{i}_12345"})
    duration = time.time() - start_time

    assert duration < 1.0, f"100 scoring operations should be fast, took {duration:.3f}s"


def _test_error_handling() -> None:
    """Test error handling scenarios."""
    from unittest.mock import MagicMock

    # Test configuration access with error
    result = getattr(config_schema, "NONEXISTENT_KEY_12345", "fallback")
    assert result == "fallback", "Should return fallback value"

    # Test search_api_for_criteria with invalid session
    mock_session = MagicMock()
    # Simulate missing identifiers to trigger validation failure
    mock_session.api_manager.has_essential_identifiers = False

    result = search_api_for_criteria(mock_session, {"first_name": "Test"})
    assert result == [], "Should return empty list on error"


def api_search_utils_module_tests() -> bool:
    """Comprehensive test suite for api_search_utils.py."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("API Search Utilities & GEDCOM Processing System", "api_search_utils.py")
    suite.start_suite()

    # Run all tests with suppress_logging
    with suppress_logging():
        # INITIALIZATION TESTS
        suite.run_test(
            "Module initialization and DEFAULT_CONFIG",
            _test_module_initialization,
            test_summary="Validates module initialization and configuration access",
            functions_tested="_test_module_initialization()",
            method_description="Testing configuration access and DEFAULT_CONFIG structure validation",
            expected_outcome="Module initializes correctly with proper configuration access and valid DEFAULT_CONFIG structure",
        )

        # CORE FUNCTIONALITY TESTS
        suite.run_test(
            "_extract_year_from_date(), _run_simple_suggestion_scoring()",
            _test_core_functionality,
            test_summary="Validates core API search and scoring functionality",
            functions_tested="_extract_year_from_date(), _run_simple_suggestion_scoring()",
            method_description="Testing year extraction from various date formats and suggestion scoring with matching criteria",
            expected_outcome="All core functions execute correctly, extracting years properly and generating accurate scores",
        )

        # EDGE CASE TESTS
        suite.run_test(
            "ALL functions with edge case inputs",
            _test_edge_cases,
            test_summary="Validates edge case handling across all module functions",
            functions_tested="All module functions",
            method_description="Testing functions with empty, None, and boundary condition inputs",
            expected_outcome="All functions handle edge cases gracefully without crashes or unexpected behavior",
        )

        # INTEGRATION TESTS
        suite.run_test(
            "search_api_for_criteria() with mocked dependencies",
            _test_integration,
            test_summary="Validates integration with external API dependencies",
            functions_tested="search_api_for_criteria()",
            method_description="Testing API search functionality with mocked session and API call responses",
            expected_outcome="Integration functions work correctly with mocked external dependencies",
        )

        # PERFORMANCE TESTS
        suite.run_test(
            "_run_simple_suggestion_scoring() performance testing",
            _test_performance,
            test_summary="Validates performance characteristics of scoring operations",
            functions_tested="_run_simple_suggestion_scoring()",
            method_description="Testing execution speed of multiple scoring operations in sequence",
            expected_outcome="Scoring operations complete within acceptable time limits",
        )

        # ERROR HANDLING TESTS
        suite.run_test(
            "search_api_for_criteria() error handling",
            _test_error_handling,
            test_summary="Validates error handling and recovery functionality",
            functions_tested="search_api_for_criteria()",
            method_description="Testing error scenarios with invalid inputs and failed dependencies",
            expected_outcome="All error conditions handled gracefully with appropriate fallback responses",
        )

    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
run_comprehensive_tests = create_standard_test_runner(api_search_utils_module_tests)


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
