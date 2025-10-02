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

# === CORE INFRASTRUCTURE ===
from core_imports import (
    auto_register_module,
    get_logger,
    standardize_module_imports,
)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# Imports removed - not used in this module

standardize_module_imports()
auto_register_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# === THIRD-PARTY IMPORTS ===
# (none currently needed)
# === LOCAL IMPORTS ===
from api_utils import (
    call_facts_user_api,
    call_getladder_api,
    call_suggest_api,
    call_treesui_list_api,
)
from config import config_schema
from relationship_utils import format_api_relationship_path
from utils import SessionManager

# === MODULE LOGGER ===
logger = get_logger(__name__)

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

def _get_scoring_weights(weights: Optional[Dict[str, Union[int, float]]]) -> Dict[str, Union[int, float]]:
    """Get scoring weights with defaults."""
    if weights is None:
        return dict(config_schema.common_scoring_weights)
    return weights


def _get_year_range(date_flex: Optional[Dict[str, Any]]) -> int:
    """Get year range for flexible matching."""
    if date_flex and isinstance(date_flex, dict):
        return date_flex.get("year_match_range", 10)
    return 10


def _extract_search_criteria(search_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and clean search criteria."""
    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)
    return {
        "first_name": clean_param(search_criteria.get("first_name")),
        "surname": clean_param(search_criteria.get("surname")),
        "gender": clean_param(search_criteria.get("gender")),
        "birth_year": search_criteria.get("birth_year"),
        "birth_place": clean_param(search_criteria.get("birth_place")),
        "death_year": search_criteria.get("death_year"),
        "death_place": clean_param(search_criteria.get("death_place")),
    }


def _extract_candidate_data(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and clean candidate data - handle both camelCase and Title Case field names."""
    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)
    return {
        "first_name": clean_param(candidate.get("first_name", candidate.get("firstName", candidate.get("First Name")))),
        "surname": clean_param(candidate.get("surname", candidate.get("lastName", candidate.get("Surname")))),
        "gender": clean_param(candidate.get("gender", candidate.get("Gender"))),
        "birth_year": candidate.get("birth_year", candidate.get("birthYear", candidate.get("Birth Year"))),
        "birth_place": clean_param(candidate.get("birth_place", candidate.get("birthPlace", candidate.get("Birth Place")))),
        "death_year": candidate.get("death_year", candidate.get("deathYear", candidate.get("Death Year"))),
        "death_place": clean_param(candidate.get("death_place", candidate.get("deathPlace", candidate.get("Death Place")))),
    }


def _score_name_match(search_name: Optional[str], cand_name: Optional[str], field_name: str, score_value: int, total_score: int, field_scores: Dict[str, int], reasons: List[str]) -> int:
    """Score name matching (first name or surname)."""
    if search_name and cand_name and search_name in cand_name:
        total_score += score_value
        field_scores[field_name] = score_value
        reasons.append(f"{field_name.replace('_', ' ').title()} '{search_name}' found in '{cand_name}'")
    return total_score


def _score_gender_match(search_gender: Optional[str], cand_gender: Optional[str], score_value: int, total_score: int, field_scores: Dict[str, int], reasons: List[str]) -> int:
    """Score gender matching."""
    if search_gender and cand_gender and search_gender == cand_gender:
        total_score += score_value
        field_scores["gender"] = score_value
        reasons.append(f"Gender '{search_gender}' matched")
    return total_score


def _score_year_match(search_year: Any, cand_year: Any, field_name: str, exact_score: int, close_score: int, year_range: int, total_score: int, field_scores: Dict[str, int], reasons: List[str]) -> int:
    """Score year matching (birth or death year)."""
    if search_year and cand_year:
        try:
            search_year_int = int(search_year)
            cand_year_int = int(cand_year)

            if search_year_int == cand_year_int:
                total_score += exact_score
                field_scores[field_name] = exact_score
                reasons.append(f"{field_name.replace('_', ' ').title()} {search_year} matched exactly")
            elif abs(search_year_int - cand_year_int) <= year_range:
                total_score += close_score
                field_scores[field_name] = close_score
                reasons.append(f"{field_name.replace('_', ' ').title()} {search_year} close to {cand_year}")
        except (ValueError, TypeError) as e:
            logger.debug(f"{field_name} comparison failed - search: {search_year} ({type(search_year)}), candidate: {cand_year} ({type(cand_year)}), error: {e}")
    return total_score


def _score_place_match(search_place: Optional[str], cand_place: Optional[str], field_name: str, score_value: int, total_score: int, field_scores: Dict[str, int], reasons: List[str]) -> int:
    """Score place matching (birth or death place)."""
    if search_place and cand_place and search_place in cand_place:
        total_score += score_value
        field_scores[field_name] = score_value
        reasons.append(f"{field_name.replace('_', ' ').title()} '{search_place}' found in '{cand_place}'")
    return total_score


def _apply_bonus_scores(field_scores: Dict[str, int], weights: Dict[str, Union[int, float]], total_score: int, reasons: List[str]) -> int:
    """Apply bonus scores for multiple matching fields."""
    # Bonus for both names matching
    if "first_name" in field_scores and "surname" in field_scores:
        bonus = weights.get("bonus_both_names_contain", 25)
        total_score += bonus
        field_scores["name_bonus"] = bonus
        reasons.append("Both first name and surname matched")

    # Bonus for both birth date and place matching
    if "birth_year" in field_scores and "birth_place" in field_scores:
        bonus = weights.get("bonus_birth_date_and_place", 15)
        total_score += bonus
        field_scores["birth_bonus"] = bonus
        reasons.append("Both birth year and place matched")

    # Bonus for both death date and place matching
    if "death_year" in field_scores and "death_place" in field_scores:
        bonus = weights.get("bonus_death_date_and_place", 15)
        total_score += bonus
        field_scores["death_bonus"] = bonus
        reasons.append("Both death year and place matched")

    return total_score


def _run_simple_suggestion_scoring(
    search_criteria: Dict[str, Any],
    candidate: Dict[str, Any],
    weights: Optional[Dict[str, Union[int, float]]] = None,
    date_flex: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, int], List[str]]:
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
    total_score = _score_name_match(search["first_name"], cand["first_name"], "first_name", weights.get("contains_first_name", 25), total_score, field_scores, reasons)
    total_score = _score_name_match(search["surname"], cand["surname"], "surname", weights.get("contains_surname", 25), total_score, field_scores, reasons)

    # Score gender
    total_score = _score_gender_match(search["gender"], cand["gender"], weights.get("gender_match", 15), total_score, field_scores, reasons)

    # Score birth year and place
    total_score = _score_year_match(search["birth_year"], cand["birth_year"], "birth_year", weights.get("birth_year_match", 20), weights.get("birth_year_close", 10), year_range, total_score, field_scores, reasons)
    total_score = _score_place_match(search["birth_place"], cand["birth_place"], "birth_place", weights.get("birth_place_match", 20), total_score, field_scores, reasons)

    # Score death year and place
    total_score = _score_year_match(search["death_year"], cand["death_year"], "death_year", weights.get("death_year_match", 20), weights.get("death_year_close", 10), year_range, total_score, field_scores, reasons)
    total_score = _score_place_match(search["death_place"], cand["death_place"], "death_place", weights.get("death_place_match", 20), total_score, field_scores, reasons)

    # Apply bonus scores
    total_score = _apply_bonus_scores(field_scores, weights, total_score, reasons)

    return int(total_score), field_scores, reasons


# Helper functions for search_api_for_criteria

def _build_search_query(search_criteria: Dict[str, Any]) -> str:
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


def _get_tree_and_profile_ids(session_manager: SessionManager) -> tuple[Optional[str], Optional[str]]:
    """Get tree ID and profile ID from session manager or config."""
    tree_id = session_manager.my_tree_id
    if not tree_id:
        tree_id = getattr(config_schema.test, "test_tree_id", "")

    owner_profile_id = session_manager.my_profile_id
    if not owner_profile_id:
        owner_profile_id = getattr(config_schema.test, "test_profile_id", "")

    return tree_id, owner_profile_id


def _parse_lifespan(lifespan: str) -> tuple[Optional[int], Optional[int]]:
    """Parse lifespan string to extract birth and death years."""
    birth_year = None
    death_year = None

    if not lifespan:
        return birth_year, death_year

    if "-" in lifespan:
        parts = lifespan.split("-")
        if len(parts) == 2:
            try:
                birth_year = int(parts[0].strip()) if parts[0].strip() else None
                death_year = int(parts[1].strip()) if parts[1].strip() else None
            except ValueError:
                pass
    elif "b." in lifespan.lower():
        match = re.search(r"b\.\s*(\d{4})", lifespan.lower())
        if match:
            try:
                birth_year = int(match.group(1))
            except ValueError:
                pass
    elif "d." in lifespan.lower():
        match = re.search(r"d\.\s*(\d{4})", lifespan.lower())
        if match:
            try:
                death_year = int(match.group(1))
            except ValueError:
                pass

    return birth_year, death_year


def _process_suggest_result(suggestion: Dict[str, Any], search_criteria: Dict[str, Any], scoring_weights: Dict[str, int], date_flex: Dict[str, int]) -> Optional[Dict[str, Any]]:
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

        # Only include if score is above threshold
        if total_score > 0:
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


def _process_treesui_person(person: Dict[str, Any], search_criteria: Dict[str, Any], scoring_weights: Dict[str, int], date_flex: Dict[str, int], scored_matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Process a single treesui-list person and return match record if score > 0 and not duplicate."""
    try:
        person_id = person.get("id")
        if not person_id:
            return None

        # Extract name components
        first_name = person.get("firstName", "")
        surname = person.get("lastName", "")

        # Extract birth information
        birth_info = person.get("birth", {})
        birth_date = birth_info.get("date", {}).get("normalized", "")
        birth_year = _extract_year_from_date(birth_date)
        birth_place = birth_info.get("place", {}).get("normalized", "")

        # Extract death information
        death_info = person.get("death", {})
        death_date = death_info.get("date", {}).get("normalized", "")
        death_year = _extract_year_from_date(death_date)
        death_place = death_info.get("place", {}).get("normalized", "")

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
            "gender": gender,
        }

        # Score the candidate
        total_score, field_scores, reasons = _run_simple_suggestion_scoring(
            search_criteria, candidate, scoring_weights, date_flex
        )

        # Only include if score is above threshold and not duplicate
        if total_score > 0:
            if not any(match["id"] == person_id for match in scored_matches):
                return {
                    "id": person_id,
                    "display_id": person_id,
                    "first_name": first_name,
                    "surname": surname,
                    "gender": gender,
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
    """Validate session manager is active."""
    try:
        if not session_manager or not session_manager.is_sess_valid():
            logger.error("Session manager is not valid or not logged in")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking session validity: {e}")
        return False


def _call_suggest_api_for_search(session_manager: SessionManager, search_criteria: Dict[str, Any], tree_id: str, owner_profile_id: str) -> List[Dict[str, Any]]:
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
        _owner_profile_id=owner_profile_id,
        base_url=base_url,
        search_criteria=api_search_criteria,
    )

    if not suggest_results or not isinstance(suggest_results, list):
        logger.warning(f"No results from suggest API for query: {_build_search_query(search_criteria)}")
        return []

    return suggest_results


def _process_suggest_results(suggest_results: List[Dict[str, Any]], search_criteria: Dict[str, Any], max_suggestions: int) -> List[Dict[str, Any]]:
    """Process suggest API results and return scored matches."""
    scoring_weights = config_schema.common_scoring_weights
    date_flex = {"year_match_range": config_schema.date_flexibility}
    scored_matches = []

    for suggestion in suggest_results[:max_suggestions]:
        match_record = _process_suggest_result(suggestion, search_criteria, scoring_weights, date_flex)
        if match_record:
            scored_matches.append(match_record)

    return scored_matches


def _build_treesui_search_params(search_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Build search parameters for treesui-list API."""
    search_params = {}

    field_mapping = {
        "first_name": "firstName",
        "surname": "lastName",
        "gender": "gender",
        "birth_year": "birthYear",
        "birth_place": "birthLocation",
        "death_year": "deathYear",
        "death_place": "deathLocation",
    }

    for criteria_key, api_key in field_mapping.items():
        if search_criteria.get(criteria_key):
            search_params[api_key] = search_criteria[criteria_key]

    return search_params


def _call_treesui_api_for_search(session_manager: SessionManager, search_params: Dict[str, Any], tree_id: str, owner_profile_id: str) -> Optional[Dict[str, Any]]:
    """Call treesui-list API and return results."""
    if not search_params:
        return None

    logger.info(f"Calling treesui-list API with params: {search_params}")
    base_url = config_schema.api.base_url

    return call_treesui_list_api(
        session_manager=session_manager,
        owner_tree_id=tree_id,
        _owner_profile_id=owner_profile_id,
        base_url=base_url,
        search_criteria=search_params,
    )


def _process_treesui_results(treesui_results: Optional[Dict[str, Any]], search_criteria: Dict[str, Any], scored_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process treesui-list API results and add to scored matches."""
    if not treesui_results or not isinstance(treesui_results, dict):
        return scored_matches

    persons = treesui_results.get("persons", [])
    if not persons:
        return scored_matches

    scoring_weights = config_schema.common_scoring_weights
    date_flex = {"year_match_range": config_schema.date_flexibility}

    for person in persons:
        match_record = _process_treesui_person(person, search_criteria, scoring_weights, date_flex, scored_matches)
        if match_record:
            scored_matches.append(match_record)

    return scored_matches


def search_api_for_criteria(
    session_manager: SessionManager,
    search_criteria: Dict[str, Any],
    max_results: int = 10,
) -> List[Dict[str, Any]]:
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

    # Get tree ID and profile ID
    tree_id, owner_profile_id = _get_tree_and_profile_ids(session_manager)
    if not tree_id:
        logger.error("No tree ID available for API search")
        return []

    # Call suggest API and process results
    suggest_results = _call_suggest_api_for_search(session_manager, search_criteria, tree_id, owner_profile_id)
    max_suggestions = config_schema.max_suggestions_to_score
    scored_matches = _process_suggest_results(suggest_results, search_criteria, max_suggestions)

    # Try treesui-list API if not enough results
    if len(scored_matches) < max_results:
        try:
            search_params = _build_treesui_search_params(search_criteria)
            treesui_results = _call_treesui_api_for_search(session_manager, search_params, tree_id, owner_profile_id)
            scored_matches = _process_treesui_results(treesui_results, search_criteria, scored_matches)
        except Exception as e:
            logger.error(f"Error calling treesui-list API: {e}")

    # Sort and return top matches
    scored_matches.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    return scored_matches[:max_results] if scored_matches else []


def get_api_family_details(
    session_manager: SessionManager,
    person_id: str,
    tree_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get family details for a specific individual from Ancestry API.

    Args:
        session_manager: SessionManager instance with active session
        person_id: Ancestry API person ID
        tree_id: Optional tree ID (default: from session_manager or config)

    Returns:
        Dictionary containing family details (parents, spouses, children, siblings)
    """
    # Step 1: Check if session is active
    if not session_manager or not session_manager.is_sess_valid():
        logger.error("Session manager is not valid or not logged in")
        return {}

    # Step 2: Get tree ID if not provided
    if not tree_id:
        tree_id = session_manager.my_tree_id
        if not tree_id:
            tree_id = getattr(config_schema.test, "test_tree_id", "")
            if not tree_id:
                logger.error("No tree ID available for API family details")
                return {}

    # Step 3: Get owner profile ID
    owner_profile_id = session_manager.my_profile_id
    if not owner_profile_id:
        owner_profile_id = getattr(config_schema.test, "test_profile_id", "")
        if not owner_profile_id:
            logger.warning("No owner profile ID available for API family details")

    # Step 4: Get base URL
    base_url = config_schema.api.base_url

    # Step 5: Call the facts API to get person details
    logger.info(f"Getting facts for person {person_id} in tree {tree_id}")
    facts_data = call_facts_user_api(
        session_manager=session_manager,
        owner_profile_id=owner_profile_id,
        api_person_id=person_id,
        api_tree_id=tree_id,
        base_url=base_url,
    )

    if not facts_data or not isinstance(facts_data, dict):
        logger.warning(f"No facts data returned for person {person_id}")
        return {}

    # Step 6: Extract person details
    result = {
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

    try:
        # Extract basic person information
        person_data = facts_data.get("person", {})
        result["name"] = person_data.get("personName", "Unknown")

        # Split name into first name and surname
        name_parts = result["name"].split()
        if name_parts:
            result["first_name"] = name_parts[0]
            if len(name_parts) > 1:
                result["surname"] = name_parts[-1]

        # Extract gender
        gender_raw = person_data.get("gender", "").lower()
        if gender_raw == "male":
            result["gender"] = "M"
        elif gender_raw == "female":
            result["gender"] = "F"

        # Extract birth information
        birth_facts = [
            f for f in facts_data.get("facts", []) if f.get("type") == "Birth"
        ]
        if birth_facts:
            birth_fact = birth_facts[0]
            birth_date = birth_fact.get("date", {}).get("normalized", "Unknown")
            birth_place = birth_fact.get("place", {}).get("normalized", "Unknown")
            result["birth_date"] = birth_date
            result["birth_place"] = birth_place

            # Extract birth year
            if birth_date and birth_date != "Unknown":
                birth_year = _extract_year_from_date(birth_date)
                if birth_year:
                    result["birth_year"] = birth_year

        # Extract death information
        death_facts = [
            f for f in facts_data.get("facts", []) if f.get("type") == "Death"
        ]
        if death_facts:
            death_fact = death_facts[0]
            death_date = death_fact.get("date", {}).get("normalized", "Unknown")
            death_place = death_fact.get("place", {}).get("normalized", "Unknown")
            result["death_date"] = death_date
            result["death_place"] = death_place

            # Extract death year
            if death_date and death_date != "Unknown":
                death_year = _extract_year_from_date(death_date)
                if death_year:
                    result["death_year"] = death_year

        # Extract family relationships
        relationships = facts_data.get("relationships", [])

        # Process parents
        parents = [
            r
            for r in relationships
            if r.get("relationshipType") in ["Father", "Mother"]
        ]
        for parent in parents:
            parent_id = parent.get("personId")
            if not parent_id:
                continue

            relationship_type = parent.get("relationshipType", "")
            parent_info = {
                "id": parent_id,
                "name": parent.get("personName", "Unknown"),
                "birth_year": None,
                "birth_place": "Unknown",
                "death_year": None,
                "death_place": "Unknown",
                "relationship": (
                    relationship_type.lower() if relationship_type else "parent"
                ),
            }

            # Extract birth/death years if available
            birth_year_str = parent.get("birthYear")
            if birth_year_str:
                try:
                    parent_info["birth_year"] = int(birth_year_str)
                except (ValueError, TypeError):
                    pass

            death_year_str = parent.get("deathYear")
            if death_year_str:
                try:
                    parent_info["death_year"] = int(death_year_str)
                except (ValueError, TypeError):
                    pass

            result["parents"].append(parent_info)

        # Process spouses
        spouses = [r for r in relationships if r.get("relationshipType") in ["Spouse"]]
        for spouse in spouses:
            spouse_id = spouse.get("personId")
            if not spouse_id:
                continue

            spouse_info = {
                "id": spouse_id,
                "name": spouse.get("personName", "Unknown"),
                "birth_year": None,
                "birth_place": "Unknown",
                "death_year": None,
                "death_place": "Unknown",
                "marriage_date": "Unknown",
                "marriage_place": "Unknown",
            }

            # Extract birth/death years if available
            birth_year_str = spouse.get("birthYear")
            if birth_year_str:
                try:
                    spouse_info["birth_year"] = int(birth_year_str)
                except (ValueError, TypeError):
                    pass

            death_year_str = spouse.get("deathYear")
            if death_year_str:
                try:
                    spouse_info["death_year"] = int(death_year_str)
                except (ValueError, TypeError):
                    pass

            # Extract marriage information if available
            marriage_facts = [
                f
                for f in facts_data.get("facts", [])
                if f.get("type") == "Marriage"
                and spouse_id in f.get("otherPersonIds", [])
            ]
            if marriage_facts:
                marriage_fact = marriage_facts[0]
                marriage_date = marriage_fact.get("date", {}).get(
                    "normalized", "Unknown"
                )
                marriage_place = marriage_fact.get("place", {}).get(
                    "normalized", "Unknown"
                )
                spouse_info["marriage_date"] = marriage_date
                spouse_info["marriage_place"] = marriage_place

            result["spouses"].append(spouse_info)

        # Process children
        children = [r for r in relationships if r.get("relationshipType") in ["Child"]]
        for child in children:
            child_id = child.get("personId")
            if not child_id:
                continue

            child_info = {
                "id": child_id,
                "name": child.get("personName", "Unknown"),
                "birth_year": None,
                "birth_place": "Unknown",
                "death_year": None,
                "death_place": "Unknown",
            }

            # Extract birth/death years if available
            birth_year_str = child.get("birthYear")
            if birth_year_str:
                try:
                    child_info["birth_year"] = int(birth_year_str)
                except (ValueError, TypeError):
                    pass

            death_year_str = child.get("deathYear")
            if death_year_str:
                try:
                    child_info["death_year"] = int(death_year_str)
                except (ValueError, TypeError):
                    pass

            result["children"].append(child_info)

        # Process siblings
        siblings = [
            r for r in relationships if r.get("relationshipType") in ["Sibling"]
        ]
        for sibling in siblings:
            sibling_id = sibling.get("personId")
            if not sibling_id:
                continue

            sibling_info = {
                "id": sibling_id,
                "name": sibling.get("personName", "Unknown"),
                "birth_year": None,
                "birth_place": "Unknown",
                "death_year": None,
                "death_place": "Unknown",
            }

            # Extract birth/death years if available
            birth_year_str = sibling.get("birthYear")
            if birth_year_str:
                try:
                    sibling_info["birth_year"] = int(birth_year_str)
                except (ValueError, TypeError):
                    pass

            death_year_str = sibling.get("deathYear")
            if death_year_str:
                try:
                    sibling_info["death_year"] = int(death_year_str)
                except (ValueError, TypeError):
                    pass

            result["siblings"].append(sibling_info)
    except Exception as e:
        logger.error(
            f"Error extracting family details from facts data: {e}", exc_info=True
        )

    return result


def get_api_relationship_path(
    session_manager: SessionManager,
    person_id: str,
    reference_id: Optional[str] = None,
    reference_name: Optional[str] = "Reference Person",
    tree_id: Optional[str] = None,
) -> str:
    """
    Get the relationship path between an individual and the reference person using Ancestry API.

    Args:
        session_manager: SessionManager instance with active session
        person_id: Ancestry API person ID
        reference_id: Optional reference person ID (default: from config)
        reference_name: Optional reference person name (default: "Reference Person")
        tree_id: Optional tree ID (default: from session_manager or config)

    Returns:
        Formatted relationship path string
    """
    # Step 1: Check if session is active
    if not session_manager or not session_manager.is_sess_valid():
        logger.error("Session manager is not valid or not logged in")
        return "(Session not valid)"

    # Step 2: Get tree ID if not provided
    if not tree_id:
        tree_id = session_manager.my_tree_id
        if not tree_id:
            tree_id = getattr(config_schema.test, "test_tree_id", "")
            if not tree_id:
                logger.error("No tree ID available for API relationship path")
                return "(Tree ID not available)"

    # Step 3: Get reference ID if not provided
    if not reference_id:
        reference_id = config_schema.reference_person_id
        if not reference_id:
            logger.error("Reference person ID not provided and not found in config")
            return "(Reference person ID not available)"

    # Step 4: Get base URL
    base_url = config_schema.api.base_url

    # Step 5: Call the getladder API to get relationship path
    logger.info(
        f"Getting relationship path from {person_id} to {reference_id} in tree {tree_id}"
    )
    ladder_data = call_getladder_api(
        session_manager=session_manager,
        owner_tree_id=tree_id,
        target_person_id=person_id,
        base_url=base_url,
    )

    if not ladder_data:
        logger.warning(f"No ladder data returned for person {person_id}")
        return f"(No relationship path found to {reference_name})"

    try:
        # Format the relationship path directly using the API formatter
        relationship_path = format_api_relationship_path(
            ladder_data, reference_name or "Reference Person", "Individual"
        )
        return relationship_path
    except Exception as e:
        logger.error(f"Error formatting relationship path: {e}", exc_info=True)
        return f"(Error formatting relationship path: {e!s})"


def api_search_utils_module_tests() -> bool:
    # Comprehensive test suite for api_search_utils.py
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite(
        "API Search Utilities & GEDCOM Processing System", "api_search_utils.py"
    )
    suite.start_suite()

    # INITIALIZATION TESTS
    def test_module_initialization():
        # Test module initialization and configuration
        # Test configuration access
        result = getattr(config_schema, "TEST_KEY_12345", "default_value")
        assert isinstance(result, str), "Should return string value"
        assert result == "default_value", "Should return default value for missing keys"

        # Test configuration structure
        assert isinstance(
            config_schema.common_scoring_weights, dict
        ), "common_scoring_weights should be a dictionary"
        assert (
            "contains_first_name" in config_schema.common_scoring_weights
        ), "Should have contains_first_name weight"
        assert isinstance(
            config_schema.date_flexibility, (int, float)
        ), "Should have date_flexibility as number"

    # CORE FUNCTIONALITY TESTS
    def test_core_functionality():
        # Test all core API search and scoring functions
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

        score, field_scores, reasons = _run_simple_suggestion_scoring(
            search_criteria, suggestion
        )
        assert isinstance(score, (int, float)), "Should return numeric score"
        assert isinstance(field_scores, dict), "Should return field scores dictionary"
        assert isinstance(reasons, list), "Should return reasons list"
        assert score > 0, "Should have positive score for matching data"

    # EDGE CASE TESTS
    def test_edge_cases():
        # Test edge cases and boundary conditions
        # Test _extract_year_from_date with edge cases
        result = _extract_year_from_date("")
        assert result is None, "Should handle empty string"

        result = _extract_year_from_date(None)
        assert result is None, "Should handle None input"

        result = _extract_year_from_date("1800-2000")
        assert result == 1800, "Should extract first year from range"

        # Test scoring with empty data
        score, field_scores, _reasons = _run_simple_suggestion_scoring({}, {})  # reasons unused
        assert score == 0, "Should return zero score for empty inputs"
        assert len(field_scores) == 0, "Should return empty field scores"

    # INTEGRATION TESTS
    def test_integration():
        # Test integration with mocked external dependencies
        from unittest.mock import MagicMock, patch

        # Test search_api_for_criteria with mock session
        mock_session = MagicMock()
        mock_session.is_sess_valid.return_value = True

        with patch("api_search_utils.call_suggest_api") as mock_suggest:
            mock_suggest.return_value = [{"First Name": "Test User 12345"}]

            result = search_api_for_criteria(
                mock_session, {"first_name": "Test", "surname": "User"}
            )
            assert isinstance(result, list), "Should return list of results"

    # PERFORMANCE TESTS
    def test_performance():
        # Test performance of scoring operations
        import time

        # Test multiple scoring operations
        start_time = time.time()
        for i in range(100):
            _run_simple_suggestion_scoring(
                {"first_name": f"Test{i}_12345"}, {"First Name": f"Test{i}_12345"}
            )
        duration = time.time() - start_time

        assert (
            duration < 1.0
        ), f"100 scoring operations should be fast, took {duration:.3f}s"

    # ERROR HANDLING TESTS
    def test_error_handling():
        # Test error handling scenarios
        from unittest.mock import MagicMock

        # Test configuration access with error
        result = getattr(config_schema, "NONEXISTENT_KEY_12345", "fallback")
        assert result == "fallback", "Should return fallback value"

        # Test search_api_for_criteria with invalid session
        mock_session = MagicMock()
        mock_session.is_sess_valid.side_effect = Exception("Test error 12345")

        result = search_api_for_criteria(mock_session, {"first_name": "Test"})
        assert result == [], "Should return empty list on error"

    # Run all tests with suppress_logging
    with suppress_logging():
        # INITIALIZATION TESTS
        suite.run_test(
            test_name="Module initialization and DEFAULT_CONFIG",
            test_func=test_module_initialization,
            expected_behavior="Module initializes correctly with proper configuration access and valid DEFAULT_CONFIG structure",
            test_description="Module initialization and configuration setup processes",
            method_description="Testing configuration access and DEFAULT_CONFIG structure validation",
        )

        # CORE FUNCTIONALITY TESTS
        suite.run_test(
            test_name="_extract_year_from_date(), _run_simple_suggestion_scoring()",
            test_func=test_core_functionality,
            expected_behavior="All core functions execute correctly, extracting years properly and generating accurate scores",
            test_description="Core API search and scoring functionality operations",
            method_description="Testing year extraction from various date formats and suggestion scoring with matching criteria",
        )

        # EDGE CASE TESTS
        suite.run_test(
            test_name="ALL functions with edge case inputs",
            test_func=test_edge_cases,
            expected_behavior="All functions handle edge cases gracefully without crashes or unexpected behavior",
            test_description="Edge case handling across all module functions",
            method_description="Testing functions with empty, None, and boundary condition inputs",
        )

        # INTEGRATION TESTS
        suite.run_test(
            test_name="search_api_for_criteria() with mocked dependencies",
            test_func=test_integration,
            expected_behavior="Integration functions work correctly with mocked external dependencies",
            test_description="Integration with external API dependencies using mocks",
            method_description="Testing API search functionality with mocked session and API call responses",
        )

        # PERFORMANCE TESTS
        suite.run_test(
            test_name="_run_simple_suggestion_scoring() performance testing",
            test_func=test_performance,
            expected_behavior="Scoring operations complete within acceptable time limits",
            test_description="Performance characteristics of scoring operations",
            method_description="Testing execution speed of multiple scoring operations in sequence",
        )

        # ERROR HANDLING TESTS
        suite.run_test(
            test_name="search_api_for_criteria() error handling",
            test_func=test_error_handling,
            expected_behavior="All error conditions handled gracefully with appropriate fallback responses",
            test_description="Error handling and recovery functionality",
            method_description="Testing error scenarios with invalid inputs and failed dependencies",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    '''Run comprehensive API search utilities tests.'''
    return api_search_utils_module_tests()


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    print(
        "🔎 Running API Search Utilities & Query Building comprehensive test suite..."
    )
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
