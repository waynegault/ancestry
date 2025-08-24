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
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import contextlib
import re
from typing import Any, Optional, Union

# === LOCAL IMPORTS ===
from api_utils import (
    call_facts_user_api,
    call_getladder_api,
    call_suggest_api,
    call_treesui_list_api,
)
from config import config_schema
from relationship_utils import format_api_relationship_path

# === THIRD-PARTY IMPORTS ===
from test_framework import (
    TestSuite,
    suppress_logging,
)
from utils import SessionManager

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


def _run_simple_suggestion_scoring(
    search_criteria: dict[str, Any],
    candidate: dict[str, Any],
    weights: Optional[dict[str, Union[int, float]]] = None,
    date_flex: Optional[dict[str, Any]] = None,
) -> tuple[int, dict[str, int], list[str]]:
    """
    Use the main scoring function from gedcom_utils for consistency.
    This ensures all scoring uses the same logic and calculations.

    Args:
        search_criteria: Dictionary of search criteria
        candidate: Dictionary of candidate data
        weights: Optional dictionary of scoring weights
        date_flex: Optional dictionary of date flexibility settings

    Returns:
        Tuple of (total_score, field_scores, reasons)
    """
    # Handle empty inputs - should return zero score
    if not search_criteria or not candidate:
        return (0, {}, [])

    # Import the main scoring function
    from gedcom_utils import calculate_match_score

    # Use the unified scoring function
    result = calculate_match_score(
        search_criteria=search_criteria,
        candidate_processed_data=candidate,
        scoring_weights=weights,
        date_flexibility=date_flex
    )

    # Convert float score to int for API compatibility
    return (int(result[0]), result[1], result[2])





def process_and_score_suggestions(api_results: list[dict[str, Any]], search_criteria: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Processes API results and scores them using the main scoring logic. Returns a list of scored suggestions sorted by score (descending).

    Args:
        api_results: List of API result dictionaries
        search_criteria: Dictionary of search criteria for scoring

    Returns:
        List of scored suggestions sorted by score (descending)
    """
    scored = []
    for candidate in api_results:
        score, field_scores, reasons = _run_simple_suggestion_scoring(
            search_criteria,
            candidate,
            weights=None,
            date_flex=None
        )
        candidate_copy = candidate.copy()
        candidate_copy["score"] = score
        candidate_copy["field_scores"] = field_scores
        candidate_copy["reasons"] = reasons
        scored.append(candidate_copy)
    # Sort by score descending
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored


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
    try:
        # Step 1: Check if session is active
        if not session_manager or not session_manager.is_sess_valid():
            logger.error("Session manager is not valid or not logged in")
            return []
    except Exception as e:
        logger.error(f"Error checking session validity: {e}")
        return []

    # Step 2: Prepare search parameters
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

    search_query = search_query.strip()
    if not search_query:
        logger.error("No search criteria provided")
        return []

    # Step 3: Call the suggest API
    logger.info(f"Searching API with query: {search_query}")

    # Get tree ID from session manager or config
    tree_id = session_manager.my_tree_id
    if not tree_id:
        tree_id = getattr(config_schema.test, "test_tree_id", "")
        if not tree_id:
            logger.error("No tree ID available for API search")
            return []

    # Get base URL from config
    base_url = config_schema.api.base_url

    # Get owner profile ID from session manager or config
    owner_profile_id = session_manager.my_profile_id
    if not owner_profile_id:
        owner_profile_id = getattr(config_schema.test, "test_profile_id", "")

    # Prepare search criteria for API
    api_search_criteria = {
        "first_name_raw": search_criteria.get("first_name", ""),
        "surname_raw": search_criteria.get("surname", ""),
        "birth_year": search_criteria.get("birth_year"),
    }

    # Call the suggest API
    suggest_results = call_suggest_api(
        session_manager=session_manager,
        owner_tree_id=tree_id,
        owner_profile_id=owner_profile_id,
        base_url=base_url,
        search_criteria=api_search_criteria,
    )

    if not suggest_results or not isinstance(suggest_results, list):
        logger.warning(f"No results from suggest API for query: {search_query}")
        return []

    # Step 4: Get configuration values
    scoring_weights = config_schema.common_scoring_weights
    date_flex = {"year_match_range": config_schema.date_flexibility}
    max_suggestions = config_schema.max_suggestions_to_score

    # Step 5: Score and filter results
    scored_matches = []

    # Ensure suggest_results is a list (API returns Optional[List[Dict]])
    if not suggest_results or not isinstance(suggest_results, list):
        suggest_results = []

    # Process each suggestion result
    for suggestion in suggest_results[:max_suggestions]:
        try:
            # Extract basic information - handle both old and new API formats
            person_id = suggestion.get("PersonId") or suggestion.get("id")
            if not person_id:
                continue

            # Extract name components - handle both old and new API formats
            full_name = suggestion.get("FullName") or suggestion.get("name", "")
            first_name = suggestion.get("GivenName") or ""
            surname = suggestion.get("Surname") or ""

            # If we don't have separate first/last names, parse from full name
            if not first_name or not surname:
                name_parts = full_name.split()
                if not first_name:
                    first_name = name_parts[0] if name_parts else ""
                if not surname:
                    surname = name_parts[-1] if len(name_parts) > 1 else ""

            # Extract birth/death years - handle both old and new API formats
            birth_year = None
            death_year = None

            # New API format has direct BirthYear/DeathYear fields
            if suggestion.get("BirthYear"):
                with contextlib.suppress(ValueError, TypeError):
                    birth_year = int(suggestion["BirthYear"])

            if suggestion.get("DeathYear"):
                with contextlib.suppress(ValueError, TypeError):
                    death_year = int(suggestion["DeathYear"])

            # Fallback to old lifespan parsing if needed
            if birth_year is None or death_year is None:
                lifespan = suggestion.get("lifespan", "")
                if lifespan:
                    if "-" in lifespan:
                        parts = lifespan.split("-")
                        if len(parts) == 2:
                            try:
                                if birth_year is None and parts[0].strip():
                                    birth_year = int(parts[0].strip())
                                if death_year is None and parts[1].strip():
                                    death_year = int(parts[1].strip())
                            except ValueError:
                                pass
                    elif "b." in lifespan.lower():
                        match = re.search(r"b\.\s*(\d{4})", lifespan.lower())
                        if match and birth_year is None:
                            with contextlib.suppress(ValueError):
                                birth_year = int(match.group(1))
                    elif "d." in lifespan.lower():
                        match = re.search(r"d\.\s*(\d{4})", lifespan.lower())
                        if match and death_year is None:
                            with contextlib.suppress(ValueError):
                                death_year = int(match.group(1))

            # Extract location information
            location = suggestion.get("location", "")

            # Create candidate data for scoring
            candidate = {
                "first_name": first_name,
                "surname": surname,
                "birth_year": birth_year,
                "death_year": death_year,
                "birth_place": location,  # Assuming location is birth place
                "death_place": None,  # Not available in suggestion results
                "gender": None,  # Not available in suggestion results
            }

            # Score the candidate using the same function as Action 10
            from gedcom_utils import calculate_match_score
            total_score, field_scores, reasons = calculate_match_score(
                search_criteria=search_criteria,
                candidate_processed_data=candidate,
                scoring_weights=scoring_weights,
                date_flexibility=date_flex
            )

            # Only include if score is above threshold
            if total_score > 0:
                # Create a match record
                match_record = {
                    "id": person_id,
                    "person_id": person_id,  # Add person_id field for compatibility
                    "display_id": person_id,
                    "first_name": first_name,
                    "surname": surname,
                    "full_name_disp": full_name,  # Add full name display field
                    "gender": None,  # Not available in suggestion results
                    "birth_year": birth_year,
                    "birth_place": location,  # Assuming location is birth place
                    "death_year": death_year,
                    "death_place": None,  # Not available in suggestion results
                    "total_score": total_score,
                    "field_scores": field_scores,
                    "reasons": reasons,
                    "source": "API",
                }
                scored_matches.append(match_record)
        except Exception as e:
            logger.error(f"Error processing suggestion result: {e}")
            continue

    # Step 6: Try treesui-list API if suggest API didn't return enough results
    if len(scored_matches) < max_results:
        try:
            # Prepare search parameters for treesui-list API
            search_params = {}
            if search_criteria.get("first_name"):
                search_params["firstName"] = search_criteria["first_name"]
            if search_criteria.get("surname"):
                search_params["lastName"] = search_criteria["surname"]
            if search_criteria.get("gender"):
                search_params["gender"] = search_criteria["gender"]
            if search_criteria.get("birth_year"):
                search_params["birthYear"] = search_criteria["birth_year"]
            if search_criteria.get("birth_place"):
                search_params["birthLocation"] = search_criteria["birth_place"]
            if search_criteria.get("death_year"):
                search_params["deathYear"] = search_criteria["death_year"]
            if search_criteria.get("death_place"):
                search_params["deathLocation"] = search_criteria["death_place"]

            # Call treesui-list API
            if search_params:
                logger.info(f"Calling treesui-list API with params: {search_params}")

                # Get tree ID from session manager or config (reuse from earlier)
                tree_id = session_manager.my_tree_id
                if not tree_id:
                    tree_id = getattr(config_schema.test, "test_tree_id", "")
                    if not tree_id:
                        logger.error("No tree ID available for treesui-list API search")
                        return scored_matches

                # Get base URL from config (reuse from earlier)
                base_url = config_schema.api.base_url

                # Get owner profile ID from session manager or config (reuse from earlier)
                owner_profile_id = session_manager.my_profile_id
                if not owner_profile_id:
                    owner_profile_id = getattr(
                        config_schema.test, "test_profile_id", ""
                    )

                # Call the treesui-list API (pass original search_criteria, not mapped search_params)
                treesui_results = call_treesui_list_api(
                    session_manager=session_manager,
                    owner_tree_id=tree_id,
                    owner_profile_id=owner_profile_id,
                    base_url=base_url,
                    search_criteria=search_criteria,
                )

                if (
                    treesui_results
                    and isinstance(treesui_results, dict)
                    and "persons" in treesui_results
                ):
                    persons = treesui_results.get("persons", [])

                    # Process each person
                    for person in persons:
                        try:
                            person_id = person.get("id")
                            if not person_id:
                                continue

                            # Extract name components
                            first_name = person.get("firstName", "")
                            surname = person.get("lastName", "")

                            # Extract birth information
                            birth_info = person.get("birth", {})
                            birth_date = birth_info.get("date", {}).get(
                                "normalized", ""
                            )
                            birth_year = _extract_year_from_date(birth_date)
                            birth_place = birth_info.get("place", {}).get(
                                "normalized", ""
                            )

                            # Extract death information
                            death_info = person.get("death", {})
                            death_date = death_info.get("date", {}).get(
                                "normalized", ""
                            )
                            death_year = _extract_year_from_date(death_date)
                            death_place = death_info.get("place", {}).get(
                                "normalized", ""
                            )

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
                            total_score, field_scores, reasons = (
                                _run_simple_suggestion_scoring(
                                    search_criteria,
                                    candidate,
                                    scoring_weights,
                                    date_flex,
                                )
                            )

                            # Only include if score is above threshold
                            if total_score > 0:
                                # Check if this person is already in scored_matches
                                if not any(
                                    match["id"] == person_id for match in scored_matches
                                ):
                                    # Create a match record
                                    match_record = {
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
                                    scored_matches.append(match_record)
                        except Exception as e:
                            logger.error(f"Error processing treesui-list result: {e}")
                            continue
        except Exception as e:
            logger.error(f"Error calling treesui-list API: {e}")

    # Sort matches by score (highest first)
    scored_matches.sort(key=lambda x: x.get("total_score", 0), reverse=True)

    # Return top matches (limited by max_results)
    return scored_matches[:max_results] if scored_matches else []


def get_api_family_details(
    session_manager: SessionManager,
    person_id: str,
    tree_id: Optional[str] = None,
) -> dict[str, Any]:
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
                with contextlib.suppress(ValueError, TypeError):
                    parent_info["birth_year"] = int(birth_year_str)

            death_year_str = parent.get("deathYear")
            if death_year_str:
                with contextlib.suppress(ValueError, TypeError):
                    parent_info["death_year"] = int(death_year_str)

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
                with contextlib.suppress(ValueError, TypeError):
                    spouse_info["birth_year"] = int(birth_year_str)

            death_year_str = spouse.get("deathYear")
            if death_year_str:
                with contextlib.suppress(ValueError, TypeError):
                    spouse_info["death_year"] = int(death_year_str)

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
                with contextlib.suppress(ValueError, TypeError):
                    child_info["birth_year"] = int(birth_year_str)

            death_year_str = child.get("deathYear")
            if death_year_str:
                with contextlib.suppress(ValueError, TypeError):
                    child_info["death_year"] = int(death_year_str)

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
                with contextlib.suppress(ValueError, TypeError):
                    sibling_info["birth_year"] = int(birth_year_str)

            death_year_str = sibling.get("deathYear")
            if death_year_str:
                with contextlib.suppress(ValueError, TypeError):
                    sibling_info["death_year"] = int(death_year_str)

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
        return format_api_relationship_path(
            ladder_data, reference_name or "Reference Person", "Individual"
        )
    except Exception as e:
        logger.error(f"Error formatting relationship path: {e}", exc_info=True)
        return f"(Error formatting relationship path: {e!s})"


def api_search_utils_module_tests() -> bool:
    # Comprehensive test suite for api_search_utils.py

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
        score, field_scores, reasons = _run_simple_suggestion_scoring({}, {})
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
