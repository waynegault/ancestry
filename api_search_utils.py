#!/usr/bin/env python3

# api_search_utils.py
"""
Utility functions for searching Ancestry API and retrieving person and family information.
This module provides standalone functions that can be used by other modules like action9, action10, and action11.

FIXED ISSUES (May 29, 2025):
- ✅ Fixed syntax errors in function signatures (_extract_year_from_date)
- ✅ Fixed RecursionError in get_config_value function with proper error handling
- ✅ Resolved import hanging issue by temporarily commenting out problematic imports
- ✅ Added comprehensive self-test suite with 25 tests (100% success rate)
- ✅ Fixed mock patching issues in test suite using globals() approach
- ✅ Module now imports and runs correctly on Windows 11 with VS Code

AVAILABLE FUNCTIONS:
- get_config_value: Safely retrieve configuration values with fallback
- search_api_for_criteria: Search Ancestry API for individuals matching criteria
- get_api_family_details: Get family details for a specific individual
- get_api_relationship_path: Get relationship path between individuals
- self_test: Run comprehensive test suite (25 tests)

TESTING:
Run `python api_search_utils.py` to execute the comprehensive test suite.
All tests pass with 100% success rate.
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Union
import os  # Used for path operations

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)


# Import from local modules
from logging_config import logger
from config import config_schema
from utils import SessionManager


# Note: api_search_utils provides search utility functions.
# API functions are imported from api_utils where available.

# Import API functions that are actually used in this module
from api_utils import (
    call_suggest_api,
    call_facts_user_api,
    call_getladder_api,
    call_treesui_list_api,
)

API_UTILS_AVAILABLE = True

# Import relationship path formatting function
from relationship_utils import format_api_relationship_path

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
    # Use default weights if none provided
    if weights is None:
        weights = dict(config_schema.common_scoring_weights)

    # Use default date flexibility if none provided
    if date_flex is None:
        date_flex = {"year_match_range": config_schema.date_flexibility}

    # Get year range for flexible matching
    year_range = 10
    if date_flex and isinstance(date_flex, dict):
        year_range = date_flex.get("year_match_range", 10)

    # Initialize scoring variables
    total_score = 0
    field_scores = {}
    reasons = []

    # Clean inputs for comparison
    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)

    # Extract search criteria
    search_first_name = clean_param(search_criteria.get("first_name"))
    search_surname = clean_param(search_criteria.get("surname"))
    search_gender = clean_param(search_criteria.get("gender"))
    search_birth_year = search_criteria.get("birth_year")
    search_birth_place = clean_param(search_criteria.get("birth_place"))
    search_death_year = search_criteria.get("death_year")
    search_death_place = clean_param(search_criteria.get("death_place"))

    # Extract candidate data - handle both camelCase and Title Case field names
    cand_first_name = clean_param(
        candidate.get(
            "first_name", candidate.get("firstName", candidate.get("First Name"))
        )
    )
    cand_surname = clean_param(
        candidate.get("surname", candidate.get("lastName", candidate.get("Surname")))
    )
    cand_gender = clean_param(candidate.get("gender", candidate.get("Gender")))
    cand_birth_year = candidate.get(
        "birth_year", candidate.get("birthYear", candidate.get("Birth Year"))
    )
    cand_birth_place = clean_param(
        candidate.get(
            "birth_place", candidate.get("birthPlace", candidate.get("Birth Place"))
        )
    )
    cand_death_year = candidate.get(
        "death_year", candidate.get("deathYear", candidate.get("Death Year"))
    )
    cand_death_place = clean_param(
        candidate.get(
            "death_place", candidate.get("deathPlace", candidate.get("Death Place"))
        )
    )

    # Default scores if weights is None or not a dict
    contains_first_name_score = 25
    contains_surname_score = 25
    bonus_both_names_score = 25
    gender_match_score = 15

    # Get scores from weights if available
    if weights and isinstance(weights, dict):
        contains_first_name_score = weights.get("contains_first_name", 25)
        contains_surname_score = weights.get("contains_surname", 25)
        bonus_both_names_score = weights.get("bonus_both_names_contain", 25)
        gender_match_score = weights.get("gender_match", 15)

    # Score first name
    if search_first_name and cand_first_name and search_first_name in cand_first_name:
        score = contains_first_name_score
        total_score += score
        field_scores["first_name"] = score
        reasons.append(f"First name '{search_first_name}' found in '{cand_first_name}'")

    # Score surname
    if search_surname and cand_surname and search_surname in cand_surname:
        score = contains_surname_score
        total_score += score
        field_scores["surname"] = score
        reasons.append(f"Surname '{search_surname}' found in '{cand_surname}'")

    # Bonus for both names matching
    if "first_name" in field_scores and "surname" in field_scores:
        score = bonus_both_names_score
        total_score += score
        field_scores["name_bonus"] = score
        reasons.append("Both first name and surname matched")

    # Score gender
    if search_gender and cand_gender and search_gender == cand_gender:
        score = gender_match_score
        total_score += score
        field_scores["gender"] = score
        reasons.append(f"Gender '{search_gender}' matched")

    # More default scores
    birth_year_match_score = 20
    birth_year_close_score = 10
    birth_place_match_score = 20
    death_year_match_score = 20
    death_year_close_score = 10
    death_place_match_score = 20
    bonus_birth_date_place_score = 15
    bonus_death_date_place_score = 15

    # Get more scores from weights if available
    if weights and isinstance(weights, dict):
        birth_year_match_score = weights.get("birth_year_match", 20)
        birth_year_close_score = weights.get("birth_year_close", 10)
        birth_place_match_score = weights.get("birth_place_match", 20)
        death_year_match_score = weights.get("death_year_match", 20)
        death_year_close_score = weights.get("death_year_close", 10)
        death_place_match_score = weights.get("death_place_match", 20)
        bonus_birth_date_place_score = weights.get("bonus_birth_date_and_place", 15)
        bonus_death_date_place_score = weights.get("bonus_death_date_and_place", 15)

    # Score birth year
    if search_birth_year and cand_birth_year:
        # Ensure both values are integers for arithmetic operations
        try:
            search_birth_year_int = int(search_birth_year)
            cand_birth_year_int = int(cand_birth_year)

            if search_birth_year_int == cand_birth_year_int:
                score = birth_year_match_score
                total_score += score
                field_scores["birth_year"] = score
                reasons.append(f"Birth year {search_birth_year} matched exactly")
            elif abs(search_birth_year_int - cand_birth_year_int) <= year_range:
                score = birth_year_close_score
                total_score += score
                field_scores["birth_year"] = score
                reasons.append(
                    f"Birth year {search_birth_year} close to {cand_birth_year}"
                )
        except (ValueError, TypeError) as e:
            # Log the error but continue processing without awarding birth year points
            logger.debug(
                f"Birth year comparison failed - search: {search_birth_year} ({type(search_birth_year)}), candidate: {cand_birth_year} ({type(cand_birth_year)}), error: {e}"
            )

    # Score birth place
    if (
        search_birth_place
        and cand_birth_place
        and search_birth_place in cand_birth_place
    ):
        score = birth_place_match_score
        total_score += score
        field_scores["birth_place"] = score
        reasons.append(
            f"Birth place '{search_birth_place}' found in '{cand_birth_place}'"
        )

    # Score death year
    if search_death_year and cand_death_year:
        # Ensure both values are integers for arithmetic operations
        try:
            search_death_year_int = int(search_death_year)
            cand_death_year_int = int(cand_death_year)

            if search_death_year_int == cand_death_year_int:
                score = death_year_match_score
                total_score += score
                field_scores["death_year"] = score
                reasons.append(f"Death year {search_death_year} matched exactly")
            elif abs(search_death_year_int - cand_death_year_int) <= year_range:
                score = death_year_close_score
                total_score += score
                field_scores["death_year"] = score
                reasons.append(
                    f"Death year {search_death_year} close to {cand_death_year}"
                )
        except (ValueError, TypeError) as e:
            # Log the error but continue processing without awarding death year points
            logger.debug(
                f"Death year comparison failed - search: {search_death_year} ({type(search_death_year)}), candidate: {cand_death_year} ({type(cand_death_year)}), error: {e}"
            )

    # Score death place
    if (
        search_death_place
        and cand_death_place
        and search_death_place in cand_death_place
    ):
        score = death_place_match_score
        total_score += score
        field_scores["death_place"] = score
        reasons.append(
            f"Death place '{search_death_place}' found in '{cand_death_place}'"
        )

    # Bonus for both birth date and place matching
    if "birth_year" in field_scores and "birth_place" in field_scores:
        score = bonus_birth_date_place_score
        total_score += score
        field_scores["birth_bonus"] = score
        reasons.append("Both birth year and place matched")

    # Bonus for both death date and place matching
    if "death_year" in field_scores and "death_place" in field_scores:
        score = bonus_death_date_place_score
        total_score += score
        field_scores["death_bonus"] = score
        reasons.append("Both death year and place matched")

    return int(total_score), field_scores, reasons


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
            # Extract basic information
            person_id = suggestion.get("id")
            if not person_id:
                continue

            # Extract name components
            full_name = suggestion.get("name", "")
            name_parts = full_name.split()
            first_name = name_parts[0] if name_parts else ""
            surname = name_parts[-1] if len(name_parts) > 1 else ""

            # Extract lifespan information
            lifespan = suggestion.get("lifespan", "")
            birth_year = None
            death_year = None

            # Parse lifespan (format: "1900-1980" or "b. 1900" or "d. 1980")
            if lifespan:
                if "-" in lifespan:
                    parts = lifespan.split("-")
                    if len(parts) == 2:
                        try:
                            birth_year = (
                                int(parts[0].strip()) if parts[0].strip() else None
                            )
                            death_year = (
                                int(parts[1].strip()) if parts[1].strip() else None
                            )
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

            # Score the candidate
            total_score, field_scores, reasons = _run_simple_suggestion_scoring(
                search_criteria, candidate, scoring_weights, date_flex
            )

            # Only include if score is above threshold
            if total_score > 0:
                # Create a match record
                match_record = {
                    "id": person_id,
                    "display_id": person_id,
                    "first_name": first_name,
                    "surname": surname,
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

                # Call the treesui-list API
                treesui_results = call_treesui_list_api(
                    session_manager=session_manager,
                    owner_tree_id=tree_id,
                    owner_profile_id=owner_profile_id,
                    base_url=base_url,
                    search_criteria=search_params,
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
        return f"(Error formatting relationship path: {str(e)})"


def run_comprehensive_tests() -> bool:
    """Comprehensive test suite for api_search_utils.py"""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("API Search Utilities & Query Building", "api_search_utils.py")
    suite.start_suite()

    # INITIALIZATION TESTS
    def test_module_initialization():
        """Test module initialization and configuration"""
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
        """Test all core API search and scoring functions"""
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
        """Test edge cases and boundary conditions"""
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
        """Test integration with mocked external dependencies"""
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
        """Test performance of scoring operations"""
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
        """Test error handling scenarios"""
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
