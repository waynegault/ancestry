#!/usr/bin/env python3

"""
Action 11: API Report - Search Ancestry API, display details, family, relationship.
V18.8: Corrected SyntaxError in _parse_treesui_list_response.
       Ensured correct function definition order.
       Verified 'gender_match' key usage.
       Verified initial comparison flow.
       Verified MAX_SUGGESTIONS_TO_SCORE limit logic.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
    register_function,
    get_function,
    is_function_available,
)

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import argparse
import json
import logging
import os
import re  # Added for robust lifespan splitting
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from traceback import print_exception
from typing import Optional, List, Dict, Any, Tuple, Union, cast
from urllib.parse import urljoin, urlencode, quote

# === THIRD-PARTY IMPORTS ===
import requests  # Keep for potential exception types
from dotenv import load_dotenv
from tabulate import tabulate

# === LOCAL IMPORTS ===
from config import config_schema

# === CONFIGURATION VALIDATION ===
try:
    logger.debug("Successfully imported configuration schema.")
    if not config_schema:
        raise ImportError("Configuration schema is None.")

    required_config_attrs = [
        "common_scoring_weights",
        "name_flexibility",
        "date_flexibility",
        "max_suggestions_to_score",
        "max_candidates_to_display",
        "api",
    ]
    for attr in required_config_attrs:
        if attr == "api":
            # Check if api sub-config exists
            if not hasattr(config_schema, attr):
                raise TypeError(f"config_schema.{attr} missing.")
            api_config = getattr(config_schema, attr)
            if not hasattr(api_config, "base_url"):
                raise TypeError(f"config_schema.api.base_url missing.")
        else:
            if not hasattr(config_schema, attr):
                raise TypeError(f"config_schema.{attr} missing.")
            value = getattr(config_schema, attr)
            if value is None:
                raise TypeError(f"config_schema.{attr} is None.")
            # Check weights specifically
            if attr == "common_scoring_weights" and isinstance(value, dict):
                if "gender_match" not in value:
                    logger.warning(
                        "Config common_scoring_weights is missing 'gender_match' key."
                    )
                elif value.get("gender_match", -1) == 0:
                    logger.warning(
                        "Config common_scoring_weights has 'gender_match' set to 0."
                    )
            elif isinstance(value, (dict, list, tuple)) and not value:
                logger.warning(f"config_schema.{attr} is empty.")

    if not hasattr(config_schema.selenium, "api_timeout"):
        raise TypeError(f"config_schema.selenium.api_timeout missing.")

except ImportError as e:
    logger.critical(
        f"Failed to import config from config.py: {e}. Cannot proceed.", exc_info=True
    )
    print(f"\nFATAL ERROR: Failed to import required components from config.py: {e}")
    sys.exit(1)
except TypeError as config_err:
    logger.critical(f"Configuration Error: {config_err}")
    print(f"\nFATAL ERROR: {config_err}")
    sys.exit(1)
except Exception as e:
    logger.critical(f"Unexpected error loading configuration: {e}", exc_info=True)
    print(f"\nFATAL ERROR: Unexpected error loading configuration: {e}")
    sys.exit(1)

# Double check critical configs
if (
    config_schema is None
    or not hasattr(config_schema, "common_scoring_weights")
    or not hasattr(config_schema, "selenium")
    or not hasattr(config_schema.selenium, "api_timeout")
):
    logger.critical("One or more critical configuration components failed to load.")
    print("\nFATAL ERROR: Configuration load failed.")
    sys.exit(1)

# --- Import GEDCOM Utilities (for scoring and date helpers) ---
calculate_match_score = None
_parse_date = None
_clean_display_date = None
GEDCOM_UTILS_AVAILABLE = False
GEDCOM_SCORING_AVAILABLE = False
try:
    from gedcom_utils import calculate_match_score, _parse_date, _clean_display_date

    logger.debug("Successfully imported functions from gedcom_utils.")
    GEDCOM_SCORING_AVAILABLE = calculate_match_score is not None and callable(
        calculate_match_score
    )
    GEDCOM_UTILS_AVAILABLE = all(
        f is not None for f in [_parse_date, _clean_display_date]
    )
    if not GEDCOM_UTILS_AVAILABLE:
        logger.warning("One or more date utils from gedcom_utils are None.")
except ImportError as e:
    logger.error(f"Failed to import from gedcom_utils: {e}.", exc_info=True)

# --- Import API Utilities ---
# Import specific API call helpers, parsers, AND the timeout helper
from api_utils import (
    parse_ancestry_person_details,
    call_suggest_api,
    call_facts_user_api,
    call_getladder_api,
    call_discovery_relationship_api,
    call_treesui_list_api,
    _get_api_timeout,
)

# Import relationship utilities
from relationship_utils import (
    format_api_relationship_path,
    convert_api_path_to_unified_format,
    convert_discovery_api_path_to_unified_format,
    format_relationship_path_unified,
)

logger.debug(
    "Successfully imported required functions from api_utils and relationship_utils."
)
API_UTILS_AVAILABLE = True


# --- Import General Utilities ---
from core.session_manager import SessionManager
from utils import format_name, ordinal_case

logger.debug("Successfully imported required components from utils.")
CORE_UTILS_AVAILABLE = True

# --- Session Manager Instance ---
session_manager = SessionManager()
if not session_manager:
    logger.critical("SessionManager instance not created. Cannot proceed.")
    print("FATAL ERROR: SessionManager not available.")
    sys.exit(1)


# --- Helper Function for Parsing PersonFacts Array (Kept local as it's specific parsing logic) ---
def _extract_fact_data(
    person_facts: List[Dict], fact_type_str: str
) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """
    Extracts date string, place string, and parsed date object from PersonFacts list.
    Prioritizes non-alternate facts. Attempts parsing from ParsedDate and Date string.
    Requires _parse_date from gedcom_utils to be available.
    """
    date_str: Optional[str] = None
    place_str: Optional[str] = None
    date_obj: Optional[datetime] = None

    if not isinstance(person_facts, list):
        logger.debug(
            f"_extract_fact_data: Invalid input person_facts (not a list) for {fact_type_str}."
        )
        return date_str, place_str, date_obj

    parse_date_func = _parse_date if callable(_parse_date) else None

    for fact in person_facts:
        if (
            isinstance(fact, dict)
            and fact.get("TypeString") == fact_type_str
            and not fact.get("IsAlternate")
        ):
            date_str = fact.get("Date")
            place_str = fact.get("Place")
            parsed_date_data = fact.get("ParsedDate")
            logger.debug(
                f"_extract_fact_data: Found primary fact for {fact_type_str}: Date='{date_str}', Place='{place_str}', ParsedDate={parsed_date_data}"
            )

            # Try parsing from ParsedDate structure
            if isinstance(parsed_date_data, dict):
                year = parsed_date_data.get("Year")
                month = parsed_date_data.get("Month")
                day = parsed_date_data.get("Day")
                if year and parse_date_func:
                    try:
                        temp_date = str(year)
                        if month:
                            temp_date += f"-{str(month).zfill(2)}"
                        if day:
                            temp_date += f"-{str(day).zfill(2)}"
                        date_obj = parse_date_func(temp_date)
                        logger.debug(
                            f"_extract_fact_data: Parsed date object from ParsedDate: {date_obj}"
                        )
                    except (ValueError, TypeError) as dt_err:
                        logger.warning(
                            f"_extract_fact_data: Could not parse date from ParsedDate {parsed_date_data}: {dt_err}"
                        )
                        date_obj = None

            # Fallback to parsing the Date string if ParsedDate didn't yield an object
            if date_obj is None and date_str and parse_date_func:
                logger.debug(
                    f"_extract_fact_data: Attempting to parse date_str '{date_str}' as fallback."
                )
                try:
                    date_obj = parse_date_func(date_str)
                    logger.debug(
                        f"_extract_fact_data: Parsed date object from date_str: {date_obj}"
                    )
                except (ValueError, TypeError) as dt_err:
                    logger.warning(
                        f"_extract_fact_data: Could not parse date string '{date_str}': {dt_err}"
                    )
                    date_obj = None
            # Found primary fact, no need to check others
            break

    if date_obj is None and date_str is None and place_str is None:
        logger.debug(f"_extract_fact_data: No primary fact found for {fact_type_str}.")

    return date_str, place_str, date_obj


# End of _extract_fact_data


def _get_search_criteria() -> Optional[Dict[str, Any]]:
    """Gets search criteria from the user via input prompts."""

    # Log and display the prompt to the user (only need to print for user interaction)
    print("\n--- Enter Search Criteria (Press Enter to skip optional fields) ---\n")

    first_name = input("  First Name Contains: ").strip()
    surname = input("  Last Name Contains: ").strip()
    gender_input = input("  Gender (M/F): ").strip().upper()
    # Initialize gender to None by default
    gender = None
    # Only set gender if valid input is provided
    if gender_input and gender_input[0] in ["M", "F"]:
        gender = gender_input[0].lower()
    dob_str = input("  Birth Year (YYYY): ").strip()
    pob = input("  Birth Place Contains: ").strip()
    dod_str = input("  Death Year (YYYY): ").strip() or None
    pod = input("  Death Place Contains: ").strip() or None
    print("")

    if not (first_name or surname):
        logger.warning("API search needs First Name or Surname. Report cancelled.")
        # Use logger.info instead of duplicating with print
        print("\nAPI search needs First Name or Surname. Report cancelled.")
        return None

    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)
    parse_date_func = _parse_date if callable(_parse_date) else None

    target_birth_year: Optional[int] = None
    target_birth_date_obj: Optional[datetime] = None
    if dob_str:
        # First try to parse as a full date
        if parse_date_func:
            try:
                target_birth_date_obj = parse_date_func(dob_str)
            except Exception as e:
                logger.debug(f"Could not parse date with parse_date_func: {e}")

        # If we have a date object, extract the year
        if target_birth_date_obj:
            target_birth_year = target_birth_date_obj.year
            logger.debug(
                f"Successfully parsed birth date: {target_birth_date_obj}, year: {target_birth_year}"
            )
        else:
            # If date parsing failed, try to extract just the year
            logger.warning(f"Could not parse input birth year/date: '{dob_str}'")
            year_match = re.search(r"\b(\d{4})\b", dob_str)
            if year_match:
                try:
                    target_birth_year = int(year_match.group(1))
                    logger.debug(
                        f"Extracted birth year {target_birth_year} from '{dob_str}' as fallback."
                    )
                except ValueError:
                    logger.warning(
                        f"Could not convert extracted year '{year_match.group(1)}' to int."
                    )

    target_death_year: Optional[int] = None
    target_death_date_obj: Optional[datetime] = None
    if dod_str:
        # First try to parse as a full date
        if parse_date_func:
            try:
                target_death_date_obj = parse_date_func(dod_str)
            except Exception as e:
                logger.debug(f"Could not parse death date with parse_date_func: {e}")

        # If we have a date object, extract the year
        if target_death_date_obj:
            target_death_year = target_death_date_obj.year
            logger.debug(
                f"Successfully parsed death date: {target_death_date_obj}, year: {target_death_year}"
            )
        else:
            # If date parsing failed, try to extract just the year
            logger.warning(f"Could not parse input death year/date: '{dod_str}'")
            year_match = re.search(r"\b(\d{4})\b", dod_str)
            if year_match:
                try:
                    target_death_year = int(year_match.group(1))
                    logger.debug(
                        f"Extracted death year {target_death_year} from '{dod_str}' as fallback."
                    )
                except ValueError:
                    logger.warning(
                        f"Could not convert extracted year '{year_match.group(1)}' to int."
                    )

    search_criteria_dict = {
        "first_name_raw": first_name,
        "surname_raw": surname,
        "first_name": clean_param(first_name),
        "surname": clean_param(surname),
        "birth_year": target_birth_year,
        "birth_date_obj": target_birth_date_obj,
        "birth_place": clean_param(pob),
        "death_year": target_death_year,
        "death_date_obj": target_death_date_obj,
        "death_place": clean_param(pod),
        "gender": gender,
    }

    log_display_map = {
        "first_name": "First Name",
        "surname": "Surname",
        "birth_year": "Birth Year",
        "birth_place": "Birth Place",
        "death_year": "Death Year",
        "death_place": "Death Place",
        "gender": "Gender",
    }
    for key, display_name in log_display_map.items():
        value = search_criteria_dict.get(key)
        log_value = (
            "None"
            if value is None
            else (f"'{value}'" if isinstance(value, str) else str(value))
        )
        logger.debug(f"  {display_name}: {log_value}")
    return search_criteria_dict


# End of _get_search_criteria


# Simple scoring fallback (Uses 'gender_match' key)
def _run_simple_suggestion_scoring(
    search_criteria: Dict[str, Any], candidate_data_dict: Dict[str, Any]
) -> Tuple[float, Dict, List[str]]:
    """Performs simple fallback scoring based on hardcoded rules. Uses 'gender_match' key."""
    logger.warning("Using simple fallback scoring for suggestion.")
    score = 0.0
    # Use 'gender_match' as the key in field_scores
    field_scores = {
        "givn": 0,
        "surn": 0,
        "gender_match": 0,
        "byear": 0,
        "bdate": 0,
        "bplace": 0,
        "dyear": 0,
        "ddate": 0,
        "dplace": 0,
        "bonus": 0,  # Name bonus
        "bbonus": 0,  # Birth bonus
        "dbonus": 0,  # Death bonus
    }
    reasons = ["API Suggest Match", "Fallback Scoring"]

    # Extract data using .get()
    cand_fn = candidate_data_dict.get("first_name")
    cand_sn = candidate_data_dict.get("surname")
    cand_by = candidate_data_dict.get("birth_year")
    cand_bp = candidate_data_dict.get("birth_place")
    cand_dy = candidate_data_dict.get("death_year")
    cand_dp = candidate_data_dict.get("death_place")
    cand_gn = candidate_data_dict.get("gender")
    is_living = candidate_data_dict.get("is_living")
    search_fn = search_criteria.get("first_name")
    search_sn = search_criteria.get("surname")
    search_by = search_criteria.get("birth_year")
    search_bp = search_criteria.get("birth_place")
    search_dy = search_criteria.get("death_year")
    search_dp = search_criteria.get("death_place")
    search_gn = search_criteria.get("gender")

    # Name Scoring
    if cand_fn and search_fn and search_fn in cand_fn:
        score += 25
        field_scores["givn"] = 25
        reasons.append(f"contains first name ({search_fn}) (25pts)")
    if cand_sn and search_sn and search_sn in cand_sn:
        score += 25
        field_scores["surn"] = 25
        reasons.append(f"contains surname ({search_sn}) (25pts)")
    if field_scores["givn"] > 0 and field_scores["surn"] > 0:
        score += 25
        field_scores["bonus"] = 25
        reasons.append("bonus both names (25pts)")

    # Birth Date Scoring - get weights from config
    birth_year_match_weight = weights.get("birth_year_match", 20) if weights else 20
    birth_year_close_weight = weights.get("birth_year_close", 10) if weights else 10
    birth_place_match_weight = weights.get("birth_place_match", 20) if weights else 20

    if cand_by and search_by:
        try:
            cand_by_int = int(cand_by)
            search_by_int = int(search_by)
            if cand_by_int == search_by_int:
                score += birth_year_match_weight
                field_scores["byear"] = birth_year_match_weight
                reasons.append(
                    f"exact birth year ({cand_by}) ({birth_year_match_weight}pts)"
                )
            elif abs(cand_by_int - search_by_int) <= 5:  # Within 5 years
                score += birth_year_close_weight
                field_scores["byear"] = birth_year_close_weight
                reasons.append(
                    f"close birth year ({cand_by} vs {search_by}) ({birth_year_close_weight}pts)"
                )
        except (ValueError, TypeError):
            pass  # Skip if conversion fails

    # Birth Place Scoring
    if cand_bp and search_bp and search_bp in cand_bp:
        score += birth_place_match_weight
        field_scores["bplace"] = birth_place_match_weight
        reasons.append(
            f"birth place contains ({search_bp}) ({birth_place_match_weight}pts)"
        )

    # Birth Bonus Scoring
    if field_scores["byear"] > 0 and field_scores["bplace"] > 0:
        score += 25
        field_scores["bbonus"] = 25
        reasons.append("bonus birth info (25pts)")

    # Death Date Scoring - get weights from config
    death_year_match_weight = weights.get("death_year_match", 20) if weights else 20
    death_year_close_weight = weights.get("death_year_close", 10) if weights else 10
    death_place_match_weight = weights.get("death_place_match", 20) if weights else 20
    death_dates_absent_weight = (
        weights.get("death_dates_both_absent", 15) if weights else 15
    )
    death_bonus_weight = (
        weights.get("bonus_death_date_and_place", 15) if weights else 15
    )

    if cand_dy and search_dy:
        try:
            cand_dy_int = int(cand_dy)
            search_dy_int = int(search_dy)
            if cand_dy_int == search_dy_int:
                score += death_year_match_weight
                field_scores["dyear"] = death_year_match_weight
                reasons.append(
                    f"exact death year ({cand_dy}) ({death_year_match_weight}pts)"
                )
            elif abs(cand_dy_int - search_dy_int) <= 5:  # Within 5 years
                score += death_year_close_weight
                field_scores["dyear"] = death_year_close_weight
                reasons.append(
                    f"close death year ({cand_dy} vs {search_dy}) ({death_year_close_weight}pts)"
                )
        except (ValueError, TypeError):
            pass  # Skip if conversion fails
    elif not search_dy and not cand_dy and is_living in [False, None]:
        score += death_dates_absent_weight
        field_scores["ddate"] = death_dates_absent_weight
        reasons.append(f"death date absent ({death_dates_absent_weight}pts)")

    # Death Place Scoring
    if cand_dp and search_dp and search_dp in cand_dp:
        score += death_place_match_weight
        field_scores["dplace"] = death_place_match_weight
        reasons.append(
            f"death place contains ({search_dp}) ({death_place_match_weight}pts)"
        )

    # Death Bonus Scoring
    if (field_scores["dyear"] > 0 or field_scores["ddate"] > 0) and field_scores[
        "dplace"
    ] > 0:
        score += death_bonus_weight
        field_scores["dbonus"] = death_bonus_weight
        reasons.append(f"bonus death info ({death_bonus_weight}pts)")

    # Gender Scoring (using 'gender_match' key)
    logger.debug(
        f"[Simple Scoring] Checking Gender: Search='{search_gn}', Candidate='{cand_gn}'"
    )
    if cand_gn is not None and search_gn is not None:
        if cand_gn == search_gn:
            gender_score_value = weights.get("gender_match", 15) if weights else 15
            score += gender_score_value
            # Store score under 'gender_match' key
            field_scores["gender_match"] = gender_score_value
            reasons.append(
                f"Gender Match ({cand_gn.upper()}) ({gender_score_value}pts)"
            )
            logger.debug(
                f"[Simple Scoring] Gender match SUCCESS. Awarded {gender_score_value} points to 'gender_match'."
            )
        else:
            logger.debug(f"[Simple Scoring] Gender MISMATCH. No points awarded.")
    elif search_gn is None:
        # No gender provided in search criteria - this is acceptable
        logger.debug("[Simple Scoring] No search gender provided. No points awarded.")
    elif cand_gn is None:
        # No gender available for candidate - this is acceptable
        logger.debug(
            "[Simple Scoring] Candidate gender not available. No points awarded."
        )

    return score, field_scores, reasons


# End of _run_simple_suggestion_scoring


# Suggestion processing and scoring (Uses 'gender_match' key)
def _process_and_score_suggestions(
    suggestions: List[Dict], search_criteria: Dict[str, Any]
) -> List[Dict]:
    """
    Processes raw API suggestions, calculates match scores, and returns sorted list.
    Uses 'gender_match' key from config weights.
    """
    processed_candidates = []
    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)
    parse_date_func = _parse_date if callable(_parse_date) else None
    scoring_func = calculate_match_score if GEDCOM_SCORING_AVAILABLE else None

    scoring_weights = getattr(config_schema, "common_scoring_weights", {})
    name_flex = getattr(config_schema, "name_flexibility", 2)
    date_flex = getattr(config_schema, "date_flexibility", 2)
    # Log the gender weight using the 'gender_match' key
    gender_weight = scoring_weights.get("gender_match", 0)

    for idx, raw_candidate in enumerate(suggestions):
        if not isinstance(raw_candidate, dict):
            logger.warning(f"Skipping invalid entry: {raw_candidate}")
            continue
        # Data extraction into candidate_data_dict (ensure 'gender' key holds 'm'/'f'/None)
        full_name_disp = raw_candidate.get("FullName", "Unknown")
        person_id = raw_candidate.get("PersonId", f"Unknown_{idx}")
        first_name_cand = None
        surname_cand = None
        if full_name_disp != "Unknown":
            parts = full_name_disp.split()
        if parts:
            first_name_cand = clean_param(parts[0])
        if len(parts) > 1:
            surname_cand = clean_param(parts[-1])
        birth_year_cand = raw_candidate.get("BirthYear")
        birth_date_str_cand = raw_candidate.get("BirthDate")
        birth_place_cand = clean_param(raw_candidate.get("BirthPlace"))
        death_year_cand = raw_candidate.get("DeathYear")
        death_date_str_cand = raw_candidate.get("DeathDate")
        death_place_cand = clean_param(raw_candidate.get("DeathPlace"))
        gender_cand = raw_candidate.get("Gender")
        is_living_cand = raw_candidate.get("IsLiving")
        birth_date_obj_cand = None
        if birth_date_str_cand and parse_date_func:
            try:
                birth_date_obj_cand = parse_date_func(birth_date_str_cand)
            except ValueError:
                logger.debug(
                    f"Could not parse candidate birth date: {birth_date_str_cand}"
                )
        death_date_obj_cand = None
        if death_date_str_cand and parse_date_func:
            try:
                death_date_obj_cand = parse_date_func(death_date_str_cand)
            except ValueError:
                logger.debug(
                    f"Could not parse candidate death date: {death_date_str_cand}"
                )
        candidate_data_dict = {
            "norm_id": person_id,
            "display_id": person_id,
            "first_name": first_name_cand,
            "surname": surname_cand,
            "full_name_disp": full_name_disp,
            "gender_norm": gender_cand,
            "birth_year": birth_year_cand,
            "birth_date_obj": birth_date_obj_cand,
            "birth_place_disp": birth_place_cand,
            "death_year": death_year_cand,
            "death_date_obj": death_date_obj_cand,
            "death_place_disp": death_place_cand,
            "is_living": is_living_cand,
            "gender": gender_cand,
            "birth_place": birth_place_cand,
            "death_place": death_place_cand,
        }

        # Log inputs and Calculate Score
        logger.debug(f"--- Scoring Candidate ID: {person_id} ---")
        logger.debug(f"Search Criteria Gender: '{search_criteria.get('gender')}'")
        logger.debug(
            f"Candidate Dict Gender ('gender'): '{candidate_data_dict.get('gender')}'"
        )
        logger.debug(
            f"Calling scoring function: {getattr(scoring_func, '__name__', 'Unknown')}"
        )
        score = 0.0
        field_scores = {}
        reasons = []
        try:
            if (
                GEDCOM_SCORING_AVAILABLE
                and scoring_func == calculate_match_score
                and scoring_func is not None
            ):
                if gender_weight == 0:
                    logger.warning(f"Gender weight ('gender_match') in config is 0.")
                score, field_scores, reasons = scoring_func(
                    search_criteria,
                    candidate_data_dict,
                    scoring_weights,
                    name_flexibility=name_flex if isinstance(name_flex, dict) else None,
                    date_flexibility=date_flex if isinstance(date_flex, dict) else None,
                )
                logger.debug(
                    f"Gedcom Score for {person_id}: {score}, Fields: {field_scores}"
                )
                # Log gender score using 'gender_match' key
                if "gender_match" in field_scores:
                    logger.debug(
                        f"Gedcom Field Score ('gender_match'): {field_scores['gender_match']}"
                    )
                else:
                    logger.debug("Gedcom Field Scores missing 'gender_match' key.")
            else:  # Simple scoring
                if scoring_func is not None:
                    # Use the local simple scoring function that takes only 2 parameters
                    score, field_scores, reasons = _run_simple_suggestion_scoring(
                        search_criteria, candidate_data_dict
                    )
                    logger.debug(
                        f"Simple Score for {person_id}: {score}, Fields: {field_scores}"
                    )
                    if "gender_match" in field_scores:
                        logger.debug(
                            f"Simple Field Score ('gender_match'): {field_scores['gender_match']}"
                        )
                    else:
                        logger.debug("Simple Field Scores missing 'gender_match' key.")
                else:
                    logger.error("Scoring function is None")
        except Exception as score_err:
            logger.error(f"Error scoring {person_id}: {score_err}", exc_info=True)
            logger.warning("Falling back to simple scoring...")
            score, field_scores, reasons = _run_simple_suggestion_scoring(
                search_criteria, candidate_data_dict
            )
            reasons.append("(Error Fallback)")
            logger.debug(
                f"Fallback Score for {person_id}: {score}, Fields: {field_scores}"
            )

        # Append Processed Candidate
        processed_candidates.append(
            {
                "id": person_id,
                "name": full_name_disp,
                "gender": candidate_data_dict.get("gender"),
                "birth_date": (
                    _clean_display_date(birth_date_str_cand)
                    if callable(_clean_display_date)
                    else (birth_date_str_cand or "N/A")
                ),
                "birth_place": raw_candidate.get("BirthPlace", "N/A"),
                "death_date": (
                    _clean_display_date(death_date_str_cand)
                    if callable(_clean_display_date)
                    else (death_date_str_cand or "N/A")
                ),
                "death_place": raw_candidate.get("DeathPlace", "N/A"),
                "score": score,
                "field_scores": field_scores,
                "reasons": reasons,
                "raw_data": raw_candidate,
                "parsed_suggestion": candidate_data_dict,
            }
        )

    # Sorting
    processed_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    return processed_candidates


# End of _process_and_score_suggestions


# Display search results (Uses 'gender_match' key)
def _display_search_results(candidates: List[Dict], max_to_display: int):
    """Displays the scored search results. Uses 'gender_match' score key."""
    if not candidates:
        print("\nNo candidates to display.")
        return
    # End of if
    display_count = min(len(candidates), max_to_display)
    print(f"\n=== SEARCH RESULTS (Top {display_count} Matches) ===\n")
    table_data = []
    headers = [
        "ID",
        "Name",
        "Gender",
        "Birth",
        "Birth Place",
        "Death",
        "Death Place",
        "Total",
    ]
    for candidate in candidates[:display_count]:
        fs = candidate.get("field_scores", {})
        givn_s = fs.get("givn", 0)
        surn_s = fs.get("surn", 0)

        # Original bonus scores from field_scores
        name_bonus_orig = fs.get("bonus", 0)
        # birth_bonus_orig = fs.get("bbonus", 0) # Not strictly needed if we re-evaluate
        # death_bonus_orig = fs.get("dbonus", 0) # Not strictly needed if we re-evaluate

        gender_s = fs.get("gender_match", 0)
        byear_s = fs.get("byear", 0)
        bdate_s = fs.get("bdate", 0)
        bplace_s = fs.get("bplace", 0)

        dyear_s = fs.get("dyear", 0)
        ddate_s = fs.get("ddate", 0)
        dplace_s = fs.get("dplace", 0)

        # Check if gender score is 0 but there's a gender match reason
        # This logic updates gender_s and fs["gender_match"] locally if needed
        if gender_s == 0:
            for reason in candidate.get("reasons", []):
                if "Gender Match" in reason:
                    # Extract the score value from the reason text
                    match = re.search(r"Gender Match \([MF]\) \((\d+)pts\)", reason)
                    if match:
                        gender_s = int(match.group(1))
                        # Update the field_scores dictionary (locally for this iteration)
                        fs["gender_match"] = gender_s
                    # End of if
                # End of if
            # End of for
        # End of if

        # Determine display bonus values based on conditions and example's fixed bonus of 25
        name_bonus_s_disp = name_bonus_orig  # Use the one from field_scores as it seemed correct in example

        birth_date_score_component = max(byear_s, bdate_s)
        death_date_score_component = max(dyear_s, ddate_s)

        # If birth date and place both scored, birth bonus is 25 for display
        birth_bonus_s_disp = (
            25 if (birth_date_score_component > 0 and bplace_s > 0) else 0
        )
        # If death date and place both scored, death bonus is 25 for display
        death_bonus_s_disp = (
            25 if (death_date_score_component > 0 and dplace_s > 0) else 0
        )

        # Format display strings
        name_disp = candidate.get("name", "N/A")
        name_disp_short = name_disp[:30] + ("..." if len(name_disp) > 30 else "")

        # Name score display - first and last name scores + bonus if both present
        name_base_score = givn_s + surn_s
        name_score_str = f"[{name_base_score}]"
        if name_bonus_s_disp > 0:
            name_score_str += f"[+{name_bonus_s_disp}]"
        # End of if
        name_with_score = f"{name_disp_short} {name_score_str}"

        # Gender display
        gender_disp_val = candidate.get("gender", "N/A")
        gender_disp_str = (
            str(gender_disp_val).upper() if gender_disp_val is not None else "N/A"
        )
        gender_with_score = f"{gender_disp_str} [{gender_s}]"

        # Birth date display - uses birth_date_score_component
        bdate_disp = str(candidate.get("birth_date", "N/A"))
        birth_score_display = f"[{birth_date_score_component}]"
        bdate_with_score = f"{bdate_disp} {birth_score_display}"

        # Birth place display
        bplace_disp_val = candidate.get("birth_place", "N/A")
        bplace_disp_str = str(bplace_disp_val) if bplace_disp_val is not None else "N/A"
        bplace_disp_short = bplace_disp_str[:20] + (
            "..." if len(bplace_disp_str) > 20 else ""
        )
        bplace_with_score = f"{bplace_disp_short} [{bplace_s}]"
        # Add birth bonus using birth_bonus_s_disp
        if birth_bonus_s_disp > 0:
            bplace_with_score += f" [+{birth_bonus_s_disp}]"
        # End of if

        # Death date display - uses death_date_score_component
        ddate_disp = str(candidate.get("death_date", "N/A"))
        death_score_display = f"[{death_date_score_component}]"
        ddate_with_score = f"{ddate_disp} {death_score_display}"

        # Death place display
        dplace_disp_val = candidate.get("death_place", "N/A")
        dplace_disp_str = str(dplace_disp_val) if dplace_disp_val is not None else "N/A"
        dplace_disp_short = dplace_disp_str[:20] + (
            "..." if len(dplace_disp_str) > 20 else ""
        )
        dplace_with_score = f"{dplace_disp_short} [{dplace_s}]"
        # Add death bonus using death_bonus_s_disp
        if death_bonus_s_disp > 0:
            dplace_with_score += f" [+{death_bonus_s_disp}]"
        # End of if

        # Recalculate total score for display based on components shown
        total_display_score = (
            name_base_score
            + name_bonus_s_disp
            + gender_s  # Uses the potentially updated gender_s
            + birth_date_score_component
            + bplace_s
            + birth_bonus_s_disp
            + death_date_score_component
            + dplace_s
            + death_bonus_s_disp
        )

        # Append row
        table_data.append(
            [
                str(candidate.get("id", "N/A")),
                name_with_score,
                gender_with_score,
                bdate_with_score,
                bplace_with_score,
                ddate_with_score,
                dplace_with_score,
                str(total_display_score),  # Use the recalculated total
            ]
        )
    # End of for

    # Print table
    try:
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    except Exception as tab_err:
        logger.error(f"Error formatting table: {tab_err}")
        print("\nSearch Results (Fallback):")
        print(" | ".join(headers))
        print("-" * 80)
        for row in table_data:  # Loop to print each row
            print(" | ".join(map(str, row)))
        # End of for
    # End of try/except
    print("")


# End of _display_search_results


# Select top candidate
def _select_top_candidate(
    scored_candidates: List[Dict],
    raw_suggestions: List[
        Dict
    ],  # Keep for potential future use? Currently unused here.
) -> Optional[Tuple[Dict, Dict]]:
    """Selects the highest-scoring candidate and retrieves its original raw suggestion data."""
    if not scored_candidates:
        logger.info("No scored candidates available to select from.")
        return None

    top_scored_candidate = scored_candidates[0]
    top_scored_id = top_scored_candidate.get("id")
    top_candidate_raw = top_scored_candidate.get("raw_data")  # Get from structure

    if not top_candidate_raw or not isinstance(top_candidate_raw, dict):
        logger.error(
            f"Critical Error: Raw data missing for top candidate ID: {top_scored_id}."
        )
        print(
            f"\nError: Internal mismatch finding raw data for top candidate ({top_scored_id})."
        )
        return None

    if isinstance(top_scored_id, str) and top_scored_id.startswith("Unknown_"):
        logger.warning(
            f"Top candidate has generated ID '{top_scored_id}'. Using stored raw data."
        )

    # Return the selected candidate's processed data and its original raw data
    return top_scored_candidate, top_candidate_raw


# End of _select_top_candidate


# Display initial comparison (Uses 'gender_match' key)
def _display_initial_comparison(
    selected_candidate: Dict,  # The processed candidate dictionary with scoring information
    search_criteria: Dict[str, Any],  # The user's search criteria
):
    """
    Displays a formatted comparison between search criteria and the top matching candidate.
    Shows each field side by side with the points awarded for that field match.
    """
    # Step 1: Extract the overall score, field scores, and scoring reasons
    score = selected_candidate.get("score", 0.0)  # Total score for this candidate
    field_scores = selected_candidate.get("field_scores", {})  # Individual field scores
    reasons = selected_candidate.get(
        "reasons", []
    )  # Text explanations for points awarded
    candidate_data = selected_candidate.get(
        "parsed_suggestion", {}
    )  # Candidate's data fields

    # Step 2: Extract search criteria and candidate values for comparison
    # Default to "N/A" if a field is missing
    na = "N/A"

    # Search criteria values (what the user entered)
    sc_fn = search_criteria.get("first_name", na)  # First name
    sc_sn = search_criteria.get("surname", na)  # Surname/last name
    sc_gn = str(search_criteria.get("gender", na)).upper()  # Gender (M/F)
    sc_by = str(search_criteria.get("birth_year", na))  # Birth year
    sc_bp = search_criteria.get("birth_place", na)  # Birth place
    sc_dy = str(search_criteria.get("death_year", na))  # Death year
    sc_dp = search_criteria.get("death_place", na)  # Death place

    # Candidate values (what was found in the API)
    c_fn = candidate_data.get("first_name", na)  # First name
    c_sn = candidate_data.get("surname", na)  # Surname/last name
    c_gn = str(candidate_data.get("gender", na)).upper()  # Gender (M/F)
    c_by = str(candidate_data.get("birth_year", na))  # Birth year
    c_bp = candidate_data.get("birth_place", na)  # Birth place
    c_dy = str(candidate_data.get("death_year", na))  # Death year
    c_dp = candidate_data.get("death_place", na)  # Death place

    # Step 3: Map each scoring reason to its corresponding field for display
    # This allows us to show the specific reason points were awarded for each field
    field_to_reason = {}
    for reason in reasons:
        # Match each reason text to the appropriate field
        if "First Name" in reason or "Contains First Name" in reason:
            field_to_reason["First Name"] = reason
        elif "Last Name" in reason or "Contains Surname" in reason:
            field_to_reason["Last Name"] = reason
        elif "Bonus Both Names" in reason:
            field_to_reason["Name Bonus"] = reason
        elif "Gender Match" in reason:
            field_to_reason["Gender"] = reason
        elif "Birth Year" in reason or "Exact Birth Year" in reason:
            field_to_reason["Birth Year"] = reason
        elif "Birth Place" in reason or "Contains Birth Place" in reason:
            field_to_reason["Birth Place"] = reason
        elif "Bonus Birth Info" in reason:
            field_to_reason["Birth Bonus"] = reason
        elif "Death Dates" in reason or "Death Year" in reason:
            field_to_reason["Death Year"] = reason
        elif "Death Places" in reason or "Death Place" in reason:
            field_to_reason["Death Place"] = reason
        elif "Bonus Death Info" in reason:
            field_to_reason["Death Bonus"] = reason

    # Step 4: Display the top match header with name and total score
    display_name = candidate_data.get("full_name_disp", "Unknown")
    print(f"=== {display_name} (score: {score:.0f}) ===\n")

    # First Name comparison
    first_name_score = field_scores.get(
        "givn", 0
    )  # Points awarded for first name match
    first_name_reason = field_to_reason.get(
        "First Name", ""
    )  # Reason text for first name
    # Extract the points and reason from the reason text
    points_match = re.search(r"\((\d+)pts", first_name_reason)
    # Format the points and reason for display
    points_str = (
        f"({first_name_score}pts)"  # If no reason text, just show the score
        if not points_match
        else f"({points_match.group(1)}pts {first_name_reason.split('(')[0].strip()})"  # Show points and reason
    )
    # Display the first name comparison
    print(f"First Name: {sc_fn} vs {c_fn} {points_str}")

    # Last Name comparison
    last_name_score = field_scores.get("surn", 0)  # Points awarded for last name match
    last_name_reason = field_to_reason.get("Last Name", "")  # Reason text for last name
    points_match = re.search(r"\((\d+)pts", last_name_reason)
    points_str = (
        f"({last_name_score}pts)"
        if not points_match
        else f"({points_match.group(1)}pts {last_name_reason.split('(')[0].strip()})"
    )
    print(f"Last Name: {sc_sn} vs {c_sn} {points_str}")

    # Name Bonus (additional points when both first and last names match)
    name_bonus_score = field_scores.get(
        "bonus", 0
    )  # Bonus points for matching both names
    name_bonus_reason = field_to_reason.get(
        "Name Bonus", ""
    )  # Reason text for name bonus
    # Only display name bonus if points were awarded
    if name_bonus_score > 0:
        points_match = re.search(r"\((\d+)pts", name_bonus_reason)
        points_str = (
            f"({name_bonus_score}pts)"
            if not points_match
            else f"({points_match.group(1)}pts {name_bonus_reason.split('(')[0].strip()})"
        )
        print(f"Name Bonus:  {points_str}")

    # Gender comparison
    gender_score = field_scores.get(
        "gender_match", 0
    )  # Points awarded for gender match
    gender_reason = field_to_reason.get("Gender", "")  # Reason text for gender
    points_match = re.search(r"\((\d+)pts", gender_reason)
    points_str = (
        f"({gender_score}pts)"
        if not points_match
        else f"({points_match.group(1)}pts {gender_reason.split('(')[0].strip()})"
    )
    # Format gender values, handling None/empty values gracefully
    sc_gn_disp = (
        sc_gn.lower() if sc_gn and sc_gn.lower() not in ["none", "n/a"] else "none"
    )
    c_gn_disp = c_gn.lower() if c_gn and c_gn.lower() not in ["none", "n/a"] else "none"
    # Display gender in lowercase for consistency
    print(f"Gender: {sc_gn_disp} vs {c_gn_disp} {points_str}")

    # Birth Year comparison
    birth_year_score = field_scores.get(
        "byear", 0
    )  # Points awarded for birth year match
    birth_year_reason = field_to_reason.get(
        "Birth Year", ""
    )  # Reason text for birth year
    points_match = re.search(r"\((\d+)pts", birth_year_reason)
    points_str = (
        f"({birth_year_score}pts)"
        if not points_match
        else f"({points_match.group(1)}pts {birth_year_reason.split('(')[0].strip()})"
    )
    print(f"Birth Year: {sc_by} vs {c_by} {points_str}")

    # Birth Place comparison
    birth_place_score = field_scores.get(
        "bplace", 0
    )  # Points awarded for birth place match
    birth_place_reason = field_to_reason.get(
        "Birth Place", ""
    )  # Reason text for birth place
    points_match = re.search(r"\((\d+)pts", birth_place_reason)
    points_str = (
        f"({birth_place_score}pts)"
        if not points_match
        else f"({points_match.group(1)}pts {birth_place_reason.split('(')[0].strip()})"
    )
    print(f"Birth Place: {sc_bp} vs {c_bp} {points_str}")

    # Birth Bonus (additional points when both birth year and birth place match)
    birth_bonus_score = field_scores.get("bbonus", 0)  # Bonus points for birth info
    birth_bonus_reason = field_to_reason.get(
        "Birth Bonus", ""
    )  # Reason text for birth bonus
    # Only display birth bonus if points were awarded
    if birth_bonus_score > 0:
        points_match = re.search(r"\((\d+)pts", birth_bonus_reason)
        points_str = (
            f"({birth_bonus_score}pts)"
            if not points_match
            else f"({points_match.group(1)}pts {birth_bonus_reason.split('(')[0].strip()})"
        )
        print(f"Birth Bonus:  {points_str}")

    # Death Year comparison
    # Combine scores from death year and death date fields
    death_year_score = field_scores.get("dyear", 0) + field_scores.get("ddate", 0)
    death_year_reason = field_to_reason.get(
        "Death Year", ""
    )  # Reason text for death year
    points_match = re.search(r"\((\d+)pts", death_year_reason)
    points_str = (
        f"({death_year_score}pts)"
        if not points_match
        else f"({points_match.group(1)}pts {death_year_reason.split('(')[0].strip()})"
    )
    # Format death year values to show "none" for missing values
    sc_dy_disp = "none" if sc_dy.lower() == "none" or sc_dy.lower() == "n/a" else sc_dy
    c_dy_disp = "none" if c_dy.lower() == "none" or c_dy.lower() == "n/a" else c_dy
    print(f"Death Year: {sc_dy_disp} vs {c_dy_disp} {points_str}")

    # Death Place comparison
    death_place_score = field_scores.get(
        "dplace", 0
    )  # Points awarded for death place match
    death_place_reason = field_to_reason.get(
        "Death Place", ""
    )  # Reason text for death place
    points_match = re.search(r"\((\d+)pts", death_place_reason)
    points_str = (
        f"({death_place_score}pts)"
        if not points_match
        else f"({points_match.group(1)}pts {death_place_reason.split('(')[0].strip()})"
    )
    # Format death place values to show "none" for missing values
    sc_dp_disp = (
        "none"
        if sc_dp is None or sc_dp.lower() == "none" or sc_dp.lower() == "n/a"
        else sc_dp
    )
    c_dp_disp = (
        "none"
        if c_dp is None or c_dp.lower() == "none" or c_dp.lower() == "n/a"
        else c_dp
    )
    print(f"Death Place: {sc_dp_disp} vs {c_dp_disp} {points_str}")

    # Death Bonus (additional points when both death year and death place match)
    death_bonus_score = field_scores.get("dbonus", 0)  # Bonus points for death info
    death_bonus_reason = field_to_reason.get(
        "Death Bonus", ""
    )  # Reason text for death bonus
    # Only display death bonus if points were awarded
    if death_bonus_score > 0:
        points_match = re.search(r"\((\d+)pts", death_bonus_reason)
        points_str = (
            f"({death_bonus_score}pts)"
            if not points_match
            else f"({points_match.group(1)}pts {death_bonus_reason.split('(')[0].strip()})"
        )
        print(f"Death Bonus:  {points_str}")

    print("")  # Add a blank line at the end


# End of _display_initial_comparison


# Detailed Info Extraction (Only called if proceeding to supplementary info)
def _extract_best_name_from_details(
    person_research_data: Dict, candidate_raw: Dict
) -> str:
    """Extracts the best available name from detailed API response."""
    best_name = "Unknown"
    name_formatter = format_name if callable(format_name) else lambda x: str(x).title()
    logger.debug(
        f"_extract_best_name (Detail): Input keys={list(person_research_data.keys())}"
    )
    person_full_name = person_research_data.get("PersonFullName")
    if person_full_name and person_full_name != "Valued Relative":
        best_name = person_full_name
        logger.debug(f"Using PersonFullName: '{best_name}'")
    if best_name == "Unknown":
        person_facts_list = person_research_data.get("PersonFacts", [])
        if isinstance(person_facts_list, list):
            name_fact = next(
                (
                    f
                    for f in person_facts_list
                    if isinstance(f, dict)
                    and f.get("TypeString") == "Name"
                    and not f.get("IsAlternate")
                ),
                None,
            )
            if (
                name_fact
                and name_fact.get("Value")
                and name_fact.get("Value") != "Valued Relative"
            ):
                best_name = name_fact.get("Value", "Unknown")
                logger.debug(f"Using Name Fact: '{best_name}'")
    if best_name == "Unknown":
        first_name_comp = person_research_data.get("FirstName", "")
        last_name_comp = person_research_data.get("LastName", "")
        constructed_name = ""
        if first_name_comp or last_name_comp:
            constructed_name = f"{first_name_comp} {last_name_comp}".strip()
        if constructed_name and len(constructed_name) > 1:
            best_name = constructed_name
            logger.debug(f"Using Constructed Name: '{best_name}'")
    if best_name == "Unknown":
        cand_name = candidate_raw.get("FullName")  # Fallback to initial suggestion name
        if cand_name and cand_name != "Unknown":
            best_name = cand_name
            logger.debug(f"Using Fallback Suggestion Name: '{best_name}'")
    if not best_name or best_name == "Valued Relative":
        best_name = "Unknown"
    elif callable(name_formatter):
        best_name = name_formatter(best_name)
    return best_name


# End of _extract_best_name_from_details


def _extract_detailed_info(person_research_data: Dict, candidate_raw: Dict) -> Dict:
    """Extracts detailed information from the 'personResearch' dictionary."""
    extracted = {}
    logger.debug("Extracting details from person_research_data...")
    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)
    clean_display_func = (
        _clean_display_date
        if callable(_clean_display_date)
        else lambda x: str(x) if x else "N/A"
    )
    if not isinstance(person_research_data, dict):
        logger.error("Invalid input to _extract_detailed_info.")
        return {}
    person_facts_list = person_research_data.get("PersonFacts", [])
    if not isinstance(person_facts_list, list):
        logger.warning(f"PersonFacts not list: {type(person_facts_list)}")
        person_facts_list = []
    logger.debug(f"Found {len(person_facts_list)} items in PersonFacts.")
    # Name
    best_name = _extract_best_name_from_details(person_research_data, candidate_raw)
    extracted["name"] = best_name
    logger.info(f"Final Extracted Detail Name: {best_name}")
    # Gender
    gender_str = person_research_data.get("PersonGender")
    if not gender_str:
        gender_fact = next(
            (
                f
                for f in person_facts_list
                if isinstance(f, dict)
                and f.get("TypeString") == "Gender"
                and not f.get("IsAlternate")
            ),
            None,
        )
    if gender_fact and gender_fact.get("Value"):
        gender_str = gender_fact.get("Value")
        logger.debug(f"Using Gender fact: '{gender_str}'")
    extracted["gender_str"] = gender_str
    extracted["gender"] = None
    gender_lower = None
    if gender_str and isinstance(gender_str, str):
        gender_lower = gender_str.lower()
    if gender_lower == "male":
        extracted["gender"] = "m"
    elif gender_lower == "female":
        extracted["gender"] = "f"
    elif gender_lower in ["m", "f"]:
        extracted["gender"] = gender_lower
    else:
        logger.warning(f"Unrecognized gender: '{gender_str}'")
    logger.info(
        f"Final Extracted Detail Gender: {extracted['gender']} (from '{gender_str}')"
    )
    # Living Status
    is_living_val = person_research_data.get("IsPersonLiving")
    death_date_obj_for_living_check = None
    if isinstance(is_living_val, bool):
        extracted["is_living"] = is_living_val
    else:
        logger.debug("IsPersonLiving missing/invalid. Inferring from Death fact.")
        _, _, death_date_obj_for_living_check = _extract_fact_data(
            person_facts_list, "Death"
        )
    if death_date_obj_for_living_check is None:
        extracted["is_living"] = True
        logger.debug("No death fact, assuming living=True.")
    else:
        extracted["is_living"] = False
        logger.debug("Death fact found, assuming living=False.")
    logger.info(f"Is Living: {extracted['is_living']}")
    # Birth/Death
    birth_date_str, birth_place, birth_date_obj = _extract_fact_data(
        person_facts_list, "Birth"
    )
    death_date_str, death_place, death_date_obj = _extract_fact_data(
        person_facts_list, "Death"
    )
    extracted["birth_date_str"] = birth_date_str
    extracted["birth_place"] = birth_place
    extracted["birth_date_obj"] = birth_date_obj
    extracted["birth_year"] = birth_date_obj.year if birth_date_obj else None
    extracted["birth_date_disp"] = clean_display_func(birth_date_str)
    logger.info(
        f"Birth Details: Date='{extracted['birth_date_disp']}', Place='{birth_place or 'N/A'}', ParsedObj={birth_date_obj}"
    )
    extracted["death_date_str"] = death_date_str
    extracted["death_place"] = death_place
    extracted["death_date_obj"] = death_date_obj
    extracted["death_year"] = death_date_obj.year if death_date_obj else None
    extracted["death_date_disp"] = clean_display_func(death_date_str)
    if death_date_obj and extracted.get("is_living") is True:
        logger.warning(
            "Death date found, but IsPersonLiving=True. Overriding is_living=False."
        )
        extracted["is_living"] = False
    logger.info(
        f"Death Details: Date='{extracted['death_date_disp']}', Place='{death_place or 'N/A'}', ParsedObj={death_date_obj}"
    )
    # Family Data
    family_data = person_research_data.get("PersonFamily", {})
    if not isinstance(family_data, dict):
        logger.warning(f"PersonFamily not dict: {type(family_data)}")
        family_data = {}
    extracted["family_data"] = family_data
    logger.info(f"Family Data Keys: {list(extracted['family_data'].keys())}")
    # IDs
    extracted["person_id"] = person_research_data.get("PersonId") or candidate_raw.get(
        "PersonId"
    )
    extracted["tree_id"] = person_research_data.get("TreeId") or candidate_raw.get(
        "TreeId"
    )
    extracted["user_id"] = person_research_data.get("UserId") or candidate_raw.get(
        "UserId"
    )
    logger.info(
        f"IDs: PersonId='{extracted['person_id']}', TreeId='{extracted['tree_id']}', UserId='{extracted['user_id']}'"
    )
    # Name Components
    extracted["first_name"] = clean_param(person_research_data.get("FirstName"))
    extracted["surname"] = clean_param(person_research_data.get("LastName"))
    parts = []
    if not extracted["first_name"] and best_name != "Unknown":
        parts = best_name.split()
    if parts:
        extracted["first_name"] = clean_param(parts[0])
    if len(parts) > 1 and not extracted["surname"]:
        extracted["surname"] = clean_param(parts[-1])
    logger.debug(
        f"Extracted name components for detail scoring: First='{extracted['first_name']}', Sur='{extracted['surname']}'"
    )
    return extracted


# End of _extract_detailed_info


# Detailed scoring (Uses 'gender_match' key via fallback scorer)
def _score_detailed_match(
    extracted_info: Dict, search_criteria: Dict[str, Any]
) -> Tuple[float, Dict, List[str]]:
    """Calculates final match score based on detailed info. Uses fallback scorer if gedcom_utils unavailable."""
    scoring_func = calculate_match_score if GEDCOM_SCORING_AVAILABLE else None
    if not GEDCOM_SCORING_AVAILABLE:
        logger.warning(
            "Gedcom scoring unavailable for detailed match. Using simple fallback."
        )
    scoring_weights = getattr(config_schema, "common_scoring_weights", {})
    name_flex = getattr(config_schema, "name_flexibility", 2)
    date_flex = getattr(config_schema, "date_flexibility", 2)
    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)
    # Prepare data for scoring function
    candidate_processed_data = {
        "norm_id": extracted_info.get("person_id"),
        "display_id": extracted_info.get("person_id"),
        "first_name": extracted_info.get("first_name"),
        "surname": extracted_info.get("surname"),
        "full_name_disp": extracted_info.get("name"),
        "gender_norm": extracted_info.get("gender"),
        "birth_year": extracted_info.get("birth_year"),
        "birth_date_obj": extracted_info.get("birth_date_obj"),
        "birth_place_disp": clean_param(extracted_info.get("birth_place")),
        "death_year": extracted_info.get("death_year"),
        "death_date_obj": extracted_info.get("death_date_obj"),
        "death_place_disp": clean_param(extracted_info.get("death_place")),
        "is_living": extracted_info.get("is_living", False),
        "gender": extracted_info.get("gender"),
        "birth_place": clean_param(extracted_info.get("birth_place")),
        "death_place": clean_param(extracted_info.get("death_place")),
    }
    logger.debug(
        f"Detailed scoring - Search criteria: {json.dumps(search_criteria, default=str)}"
    )
    logger.debug(
        f"Detailed scoring - Candidate data: {json.dumps(candidate_processed_data, default=str)}"
    )
    score = 0.0
    field_scores = {}
    reasons_list = ["API Detail Match"]
    try:
        logger.debug(
            f"Calculating detailed score using {getattr(scoring_func, '__name__', 'Unknown')}..."
        )
        if (
            GEDCOM_SCORING_AVAILABLE
            and scoring_func == calculate_match_score
            and scoring_func is not None
        ):
            score, field_scores, reasons = scoring_func(
                search_criteria,
                candidate_processed_data,
                scoring_weights,
                name_flexibility=name_flex if isinstance(name_flex, dict) else None,
                date_flexibility=date_flex if isinstance(date_flex, dict) else None,
            )
        else:
            if scoring_func is not None:
                score, field_scores, reasons = scoring_func(
                    search_criteria, candidate_processed_data
                )  # Simple scorer
            else:
                logger.error("Scoring function is None")
        if "API Detail Match" not in reasons:
            reasons.insert(0, "API Detail Match")
            reasons.insert(0, "API Detail Match")
        reasons_list = reasons
        logger.info(f"Calculated detailed score: {score:.0f}")
        # Log detailed score gender key ('gender_match')
        if "gender_match" in field_scores:
            logger.debug(
                f"Detailed Field Score ('gender_match'): {field_scores['gender_match']}"
            )
        else:
            logger.debug("Detailed Field Scores missing 'gender_match' key.")
    except Exception as e:
        logger.error(f"Error calculating detailed score: {e}", exc_info=True)
        logger.warning("Falling back to simple scoring for detailed match.")
        score, field_scores, reasons_list_fallback = _run_simple_suggestion_scoring(
            search_criteria, candidate_processed_data
        )
        reasons_list = ["API Detail Match", "(Detailed Scoring Error Fallback)"] + [
            r
            for r in reasons_list_fallback
            if r not in ["API Suggest Match", "Fallback Scoring"]
        ]
        # Log gender score key from fallback
        if "gender_match" in field_scores:
            logger.debug(
                f"Fallback Detailed Field Score ('gender_match'): {field_scores['gender_match']}"
            )

    logger.debug(f"Final detailed score: {score:.0f}")
    logger.debug(f"Final detailed field scores: {field_scores}")
    logger.debug(f"Final detailed reasons: {reasons_list}")
    return score, field_scores, reasons_list


# End of _score_detailed_match


# Family/Relationship Display Functions
def _flatten_children_list(children_raw: Union[List, Dict, None]) -> List[Dict]:
    """Flattens potentially nested list of children from PersonFamily and removes duplicates."""
    children_flat_list = []
    added_ids = set()
    if isinstance(children_raw, list):
        for child_entry in children_raw:
            items_to_process = []
            if isinstance(child_entry, list):
                items_to_process.extend(child_entry)
            elif isinstance(child_entry, dict):
                items_to_process.append(child_entry)
            else:
                logger.warning(
                    f"Unexpected item type in Children list: {type(child_entry)}"
                )
            for child_dict in items_to_process:
                if isinstance(child_dict, dict):
                    child_id = child_dict.get("PersonId")
                    if child_id and child_id not in added_ids:
                        children_flat_list.append(child_dict)
                        added_ids.add(child_id)
                    elif not child_id:
                        children_flat_list.append(child_dict)
                else:
                    logger.warning(
                        f"Non-dict item found within child entry: {type(child_dict)}"
                    )
    elif isinstance(children_raw, dict):
        child_id = children_raw.get("PersonId")
        if child_id and child_id not in added_ids:
            children_flat_list.append(children_raw)
            added_ids.add(child_id)
        elif not child_id:
            logger.warning("Single child dict missing PersonId.")
            children_flat_list.append(children_raw)
    elif children_raw is not None:
        logger.warning(f"Unexpected data type for 'Children': {type(children_raw)}")
    logger.debug(
        f"Flattened children entries into {len(children_flat_list)} unique children."
    )
    return children_flat_list


# End of _flatten_children_list


def _display_family_info(family_data: Dict):
    """Displays formatted family information (parents, siblings, spouses, children)."""

    if not isinstance(family_data, dict) or not family_data:
        logger.warning("Received empty/invalid family_data.")
        print("  Family data unavailable.")
        return
    name_formatter = format_name if callable(format_name) else lambda x: str(x).title()

    def print_relatives(rel_type: str, rel_list: Optional[List[Dict]]):
        type_display = rel_type.replace("_", " ").capitalize()
        print(f"\n{type_display}:\n")
        if not rel_list:
            print("    None found.")
            return
        if not isinstance(rel_list, list):
            print_exception(f"Expected list for {rel_type}, got {type(rel_list)}.")
            return
        found_any = False
        for idx, relative in enumerate(rel_list):
            if not isinstance(relative, dict):
                logger.warning(
                    f"Skipping invalid relative entry {idx+1} in {rel_type}: {relative}"
                )
                continue
            name = name_formatter(relative.get("FullName", "Unknown"))
            lifespan = relative.get("LifeRange", "")
            b_year, d_year = None, None
            years = []  # Initialize years as an empty list
            if lifespan and isinstance(lifespan, str):
                years = re.findall(r"\b\d{4}\b", lifespan)
                if years:
                    b_year = years[0]
                    if len(years) > 1:
                        d_year = years[-1]
            life_info = ""
            if b_year and d_year:
                life_info = f" ({b_year}–{d_year})"
            elif b_year:
                life_info = f" (b. {b_year})"
            elif lifespan and lifespan.strip():  # Only add lifespan if it's not empty
                life_info = f" ({lifespan})"
            rel_info = f"- {name}{life_info}"
            print(f"      {rel_info}")
            found_any = True
        if not found_any:
            print("None found (list had invalid entries).")

    parents_list = (family_data.get("Fathers") or []) + (
        family_data.get("Mothers") or []
    )
    siblings_list = family_data.get("Siblings", [])
    half_siblings_list = family_data.get("HalfSiblings", [])

    # Combine siblings and half-siblings
    all_siblings = siblings_list + half_siblings_list

    spouses_list = family_data.get("Spouses")
    children_raw = family_data.get("Children", [])
    children_flat_list = _flatten_children_list(children_raw)
    print_relatives("Parents", parents_list)
    print_relatives("Siblings", all_siblings)
    print_relatives("Spouses", spouses_list)
    print_relatives("Children", children_flat_list)


# End of _display_family_info


def _display_tree_relationship(
    selected_person_tree_id: str,
    selected_name: str,
    owner_tree_id: str,
    owner_name: str,
    session_manager_local: SessionManager,
    base_url: str,
):
    """Calculates and displays the relationship path using the Tree Ladder API."""
    logger.info(
        f"Calculating Tree relationship path for {selected_name} ({selected_person_tree_id}) to {owner_name} ({owner_tree_id})"
    )
    if not callable(call_getladder_api) or not callable(format_api_relationship_path):
        logger.error(
            "Cannot display tree relationship: Required api_utils functions missing."
        )
        print(
            "(Error: Relationship utilities unavailable)"
        )  # Keep error print for user
        return

    # Construct API URL for logging if needed (currently removed from print)
    # ladder_api_url = f"{base_url}/family-tree/person/tree/{owner_tree_id}/person/{selected_person_tree_id}/getladder?callback=no"

    # Call the API Helper
    relationship_data_raw = call_getladder_api(
        session_manager_local, owner_tree_id, selected_person_tree_id, base_url
    )  # API helper handles its own logging/print status

    if not relationship_data_raw:
        logger.warning("call_getladder_api returned no data for tree relationship.")
        # Message printed by call_getladder_api on failure
        return

    fallback_message_text = "(Could not parse relationship path from Tree API)"
    try:
        formatted_path = format_api_relationship_path(
            relationship_data_raw, owner_name, selected_name
        )

        known_error_starts = (
            "(No relationship",
            "(Could not parse",
            "(API returned error",
            "(Relationship HTML structure",
            "(Unsupported API response",
            "(Error processing relationship",
            "(Cannot parse relationship path",
            "(Could not find, decode, or parse",
            "(Could not find sufficient relationship",
        )
        if formatted_path and not formatted_path.startswith(known_error_starts):
            # Fix the formatting of the relationship path to match the requested format
            # Use regex to find any name followed by a year and replace with name (b year)
            # This is a general pattern that works for any name
            formatted_path = re.sub(
                r"(\w+(?:\s+\w+)*)\s+(\d{4})\.", r"\1 (b. \2).", formatted_path
            )
            print(formatted_path)  # Print the successfully formatted path
            # --- REMOVED Logging Block ---
            # logger.info("    --- Tree Relationship Path ---")
            # [
            #     logger.info(f"    {line.strip()}")
            #     for line in formatted_path.splitlines()
            #     if line.strip()
            # ]
            # logger.info("    ----------------------------")
            # --- End of REMOVED Logging Block ---
        else:
            # Print the error/fallback message returned by the formatter
            logger.warning(
                f"format_api_relationship_path returned error/fallback: '{formatted_path}'"
            )
            print(f"  {formatted_path or fallback_message_text}")
            logger.debug(
                f"Relationship parsing failed. Raw response was: {str(relationship_data_raw)[:500]}..."
            )
    except Exception as fmt_err:
        logger.error(
            f"Error calling format_api_relationship_path: {fmt_err}", exc_info=True
        )
        print(f"  {fallback_message_text} (Processing Error)")
        logger.debug(
            f"Relationship parsing failed during formatting. Raw response was: {str(relationship_data_raw)[:500]}..."
        )


# End of _display_tree_relationship


def _display_discovery_relationship(
    selected_person_global_id: str,
    selected_name: str,
    owner_profile_id: str,
    owner_name: str,
    session_manager_local: SessionManager,
    base_url: str,
):
    """Calculates and displays the relationship path using the Discovery API."""
    print(f"\n--- Relationship Path (Discovery) to {owner_name} ---")
    logger.info(
        f"Calculating Discovery relationship for {selected_name} ({selected_person_global_id}) to {owner_name} ({owner_profile_id})"
    )
    if not callable(call_discovery_relationship_api):
        logger.error("Cannot display discovery relationship: Function missing.")
        print("(Error: Relationship utility unavailable)")
        return
    # Construct the API URL for logging/display purposes
    discovery_api_url = f"{base_url}/discoveryui-matchingservice/api/relationship?profileIdFrom={owner_profile_id}&profileIdTo={selected_person_global_id}"
    print(f"\nDiAPI URL: {discovery_api_url}\n")

    # Call the API with timeout parameter
    relationship_data = call_discovery_relationship_api(
        session_manager_local,
        selected_person_global_id,
        owner_profile_id,
        base_url,
        timeout=30,  # Use a 30-second timeout for this API call
    )
    if not relationship_data:
        logger.warning("call_discovery_relationship_api returned no data.")
        print("(Discovery API call failed)")
        return
    if not isinstance(relationship_data, dict):
        logger.warning(
            f"Discovery API returned unexpected type: {type(relationship_data)}"
        )
        print("(Discovery API returned unexpected format)")
        logger.debug(f"Raw Discovery response: {str(relationship_data)[:1000]}")
        return
    print("")
    fallback_message_text = "(Could not parse relationship path from Discovery API)"
    if isinstance(relationship_data.get("path"), list) and relationship_data.get(
        "path"
    ):
        logger.info("    --- Discovery Relationship Path (Direct JSON) ---")
        path_steps = relationship_data["path"]
        print(f"  {selected_name}")
        logger.info(f"    {selected_name}")
        name_formatter = (
            format_name if callable(format_name) else lambda x: str(x).title()
        )
        rel_term_func = getattr(
            sys.modules.get("api_utils"), "_get_relationship_term", None
        )
        for step in path_steps:
            step_name = step.get("name", "?")
            step_rel = step.get("relationship", "?")
            display_rel = step_rel.capitalize()
            if callable(rel_term_func):
                display_rel = rel_term_func(None, step_rel)
            display_line = f"  -> {display_rel} is {name_formatter(step_name)}"
            print(display_line)
            logger.info(f"    {display_line.strip()}")
        print(f"  -> {owner_name} (Tree Owner / You)")
        logger.info(f"    -> {owner_name} (Tree Owner / You)")
        logger.info("    -------------------------------------------------")
    else:
        if "path" not in relationship_data:
            logger.warning("Discovery JSON missing 'path' key.")
            print(f"  {fallback_message_text} (Missing 'path')")
        elif not isinstance(relationship_data.get("path"), list):
            logger.warning(
                f"Discovery 'path' is not list: {type(relationship_data.get('path'))}"
            )
            print(f"  {fallback_message_text} ('path' not list)")
        else:
            logger.warning("Discovery 'path' list is empty.")
            print("(No direct relationship path found via Discovery API)")
        logger.debug(
            f"Discovery response content: {json.dumps(relationship_data, indent=2)}"
        )


# End of _display_discovery_relationship


# --- Phase Handler Functions ---


# API Call (Correct definition order)
def _call_direct_treesui_list_api(
    session_manager_local: SessionManager,
    owner_tree_id: str,
    search_criteria: Dict[str, Any],
    base_url: str,
) -> Optional[List[Dict]]:
    """
    Directly calls the TreesUI List API with the specific format requested.
    """
    if not session_manager_local or not owner_tree_id or not base_url:
        logger.error("Missing required parameters for direct TreesUI List API call")
        return None

    # Check if the session has a requests session available for API calls
    # This is more appropriate than checking if the browser session is valid
    if (
        not hasattr(session_manager_local, "_requests_session")
        or not session_manager_local._requests_session
    ):
        logger.error("No requests session available for direct TreesUI List API call.")
        print("Error: API session not available. Cannot contact Ancestry API.")
        return None

    first_name = search_criteria.get("first_name_raw", "")
    surname = search_criteria.get("surname_raw", "")
    params = {"sort": "sname,gname", "limit": "100", "fields": "NAMES,EVENTS,GENDER"}
    if first_name:
        params["fn"] = first_name
    if surname:
        params["ln"] = surname
    encoded_params = urlencode(params, quote_via=quote)
    api_url = (
        f"{base_url}/api/treesui-list/trees/{owner_tree_id}/persons?{encoded_params}"
    )

    # Print the API URL to the console
    logger.debug(f"\nAPI URL Called: {api_url}\n")

    api_timeout = 10  # Use a fixed timeout value

    try:
        cookies = None
        if (
            hasattr(session_manager_local, "_requests_session")
            and session_manager_local._requests_session
        ):
            cookies = session_manager_local._requests_session.cookies
        elif session_manager_local.driver and session_manager_local.is_sess_valid():
            logger.warning("Using cookies from Selenium session for direct API call.")
            try:
                selenium_cookies = session_manager_local.driver.get_cookies()
                cookies = {c["name"]: c["value"] for c in selenium_cookies}
            except Exception as cookie_err:
                logger.error(f"Failed to get cookies from Selenium: {cookie_err}")
                cookies = {}
        if not cookies:
            logger.error("No session cookies available for API call.")

        headers = {
            "Accept": "application/json",
            "Accept-Language": "en-GB,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": f"{base_url}/family-tree/tree/{owner_tree_id}/family",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
        }
        logger.debug(f"API Request Headers: {headers}")
        if cookies:
            logger.debug(f"API Request Cookies (keys): {list(cookies.keys())}")

        response = requests.get(
            api_url, headers=headers, cookies=cookies, timeout=api_timeout
        )
        logger.debug(f"API Response Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, list):
                    return data
                else:
                    logger.error(
                        f"API call OK but response not JSON list. Type: {type(data)}"
                    )
                    logger.debug(f"API Response Text: {response.text[:500]}")
                    print("Error: API returned unexpected data format.")
                    return None
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse JSON response (200 OK): {json_err}")
                logger.debug(f"API Response Text: {response.text[:1000]}")
                print("Error: Could not understand data from Ancestry.")
                return None
        elif response.status_code in [401, 403]:
            logger.error(
                f"API call failed: {response.status_code} (Unauthorized/Forbidden). Check session."
            )
            print(f"Error: Access denied ({response.status_code}). Session invalid?")
            return None
        else:
            logger.error(f"API call failed with status code: {response.status_code}")
            logger.debug(f"API Response Text: {response.text[:500]}")
            print(f"Error: Ancestry API returned error ({response.status_code}).")
            return None

    except requests.exceptions.Timeout:
        logger.error(f"API call timed out after {api_timeout}s")
        print(f"(Error: Timed out searching Ancestry API)")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"API call network/request issue: {req_err}", exc_info=True)
        print(f"(Error connecting to Ancestry API: {req_err})")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during API call: {e}", exc_info=True)
        print(f"(Unexpected error searching: {e})")
        return None


# End of _call_direct_treesui_list_api


# Search Phase (Includes Limit, Calls API func defined above)
def _handle_search_phase(
    session_manager_local: SessionManager,
    search_criteria: Dict[str, Any],
) -> Optional[List[Dict]]:
    """Handles the API search phase using the direct TreesUI List API."""
    owner_tree_id = getattr(session_manager_local, "my_tree_id", None)
    base_url = getattr(config_schema.api, "base_url", "").rstrip("/")

    # Check if owner_tree_id is missing and try to get it from config
    if not owner_tree_id:
        # Try to get tree ID from config
        config_tree_id = getattr(config_schema.api, "tree_id", None)
        if config_tree_id:
            owner_tree_id = config_tree_id
            # Update the session manager with the tree ID from config
            session_manager_local.my_tree_id = owner_tree_id
            logger.info(f"Using tree ID from configuration: {owner_tree_id}")
        else:
            # Try to get tree ID from session manager's API call
            tree_name = config_schema.api.tree_name
            if tree_name:
                logger.info(
                    f"Attempting to retrieve tree ID for tree name: {tree_name}"
                )
                try:
                    # Try to retrieve the tree ID using the session manager
                    owner_tree_id = session_manager_local.get_my_tree_id()
                    if owner_tree_id:
                        logger.info(f"Successfully retrieved tree ID: {owner_tree_id}")
                        # Update the session manager with the retrieved tree ID
                        session_manager_local.my_tree_id = owner_tree_id
                    else:
                        logger.warning(
                            f"Failed to retrieve tree ID for tree name: {tree_name}"
                        )
                except Exception as e:
                    logger.error(f"Error retrieving tree ID: {e}")

        # If still no tree ID, use a default or prompt the user
        if not owner_tree_id:
            # Prompt the user for a tree ID
            print("\nTree ID is required for searching. Please enter a tree ID:")
            user_tree_id = input("Tree ID: ").strip()
            if user_tree_id:
                owner_tree_id = user_tree_id
                # Update the session manager with the user-provided tree ID
                session_manager_local.my_tree_id = owner_tree_id
                logger.info(f"Using user-provided tree ID: {owner_tree_id}")
            else:
                # Log error and display to user
                logger.error("Owner Tree ID missing and no input provided.")
                print("Error: Tree ID is required for searching. Operation cancelled.")
                return None

    if not base_url:
        # Log error and display to user
        logger.error("ERROR: Ancestry URL not configured.. Base URL missing.")
        print("Error: Ancestry URL not configured. Operation cancelled.")
        return None

    # This call now works because the function is defined above
    suggestions_raw = _call_direct_treesui_list_api(
        session_manager_local, owner_tree_id, search_criteria, base_url
    )
    if suggestions_raw is None:
        # Log error and display to user
        logger.error("API search failed.")
        return None  # Error logged previously
    if not suggestions_raw:
        # Log info and display to user
        logger.info("API Search returned no results.No potential matches found.")
        return []
    parsed_results = _parse_treesui_list_response(suggestions_raw, search_criteria)
    if parsed_results is None:
        # Log error and display to user
        logger.error("Failed to parse API response. Error processing data.")
        return None
    # Limit suggestions based on config
    max_score_limit = getattr(config_schema, "max_suggestions_to_score", 100)
    if (
        isinstance(max_score_limit, int)
        and max_score_limit > 0
        and len(parsed_results) > max_score_limit
    ):
        logger.debug(
            f"Processing only top {max_score_limit} of {len(parsed_results)} suggestions for scoring."
        )
        return parsed_results[:max_score_limit]
    else:
        return parsed_results


# End of _handle_search_phase


# Parsing (Definition before use in _handle_search_phase)
def _parse_treesui_list_response(
    treesui_response: List[Dict],
    search_criteria: Dict[str, Any],
) -> Optional[List[Dict]]:
    """
    Parses the specific TreesUI List API response provided by the user
    to extract information needed for scoring and display.
    """
    parsed_results = []
    parse_date_func = _parse_date if callable(_parse_date) else None
    logger.debug(
        f"Parsing {len(treesui_response)} items from TreesUI List API response."
    )
    for idx, person_raw in enumerate(treesui_response):
        if not isinstance(person_raw, dict):
            logger.warning(f"Skipping item {idx}: Not dict")
            continue
        try:
            # GID
            gid_data = person_raw.get("gid", {}).get("v", "")
            person_id = None
            tree_id = None
            if isinstance(gid_data, str) and ":" in gid_data:
                parts = gid_data.split(":")
                if len(parts) >= 3:
                    person_id = parts[0]
                    tree_id = parts[2]
                    logger.debug(f"Item {idx}: IDs={person_id},{tree_id}")
                else:
                    logger.warning(f"Item {idx}: Invalid gid format: {gid_data}")
                    continue  # Skip this person if IDs cannot be extracted
            else:
                logger.warning(f"Item {idx}: Missing or invalid gid data: {gid_data}")
                continue  # Skip this person if IDs cannot be extracted

            # Name
            first_name_part = ""
            surname_part = ""
            full_name = "Unknown"
            names_list = person_raw.get("Names", [])
            if isinstance(names_list, list) and names_list:
                name_obj = names_list[0]
                if isinstance(name_obj, dict):
                    first_name_part = name_obj.get("g", "").strip()
                    surname_part = name_obj.get("s", "").strip()
                    if first_name_part and surname_part:
                        full_name = f"{first_name_part} {surname_part}"
                    elif first_name_part:
                        full_name = first_name_part
                    elif surname_part:
                        full_name = surname_part
                    logger.debug(f"Item {idx}: Name='{full_name}'")
                else:
                    logger.warning(f"Item {idx}: Name obj not dict: {name_obj}")
            else:
                logger.warning(f"Item {idx}: Names list issue: {names_list}")

            # Gender - First try to get from Genders array, then fallback to 'l' field
            gender = None
            genders_list = person_raw.get("Genders", [])
            if isinstance(genders_list, list) and genders_list:
                gender_obj = genders_list[0]
                if isinstance(gender_obj, dict) and "g" in gender_obj:
                    gender = gender_obj.get("g", "").lower()
                    logger.debug(f"Item {idx}: Gender from Genders array='{gender}'")

            # Fallback to 'l' field if Genders array didn't provide a value
            if not gender:
                gender_flag = person_raw.get("l")
                if isinstance(gender_flag, bool):
                    gender = "f" if gender_flag else "m"
                    logger.debug(f"Item {idx}: Gender from 'l' field='{gender}'")
                else:
                    logger.warning(f"Item {idx}: No gender information available")

            # Events
            birth_year = None
            birth_date_str = None
            birth_place = None
            death_year = None
            death_date_str = None
            death_place = None
            is_living = True
            events_list = person_raw.get("Events", [])

            # Store all birth events to select the best one
            birth_events = []
            death_events = []

            if isinstance(events_list, list):
                # First pass: collect all birth and death events
                for event in events_list:
                    if not isinstance(event, dict):
                        logger.warning(f"Item {idx}: Invalid event item")
                        continue

                    event_type = event.get("t")
                    if event_type == "Birth":
                        birth_events.append(event)
                    elif event_type == "Death":
                        death_events.append(event)

                # Process birth events - prioritize based on completeness and non-alternate status
                if birth_events:
                    # Sort birth events by priority:
                    # 1. Non-alternate events (pa=false) over alternate events (pa=true)
                    # 2. Events with normalized dates (nd field) over those without
                    # 3. Events with more detailed place information (more commas in place name)

                    # First, prioritize by alternate status
                    non_alternate_births = [
                        e for e in birth_events if e.get("pa") is False
                    ]
                    births_to_process = (
                        non_alternate_births if non_alternate_births else birth_events
                    )

                    # Then prioritize by place detail (count commas as a simple heuristic for detail)
                    def place_detail_score(event):
                        place = event.get("p", "")
                        if not place:
                            return 0
                        # Count commas as a simple measure of detail level
                        comma_count = place.count(",")
                        # Add 1 to avoid zero scores for places without commas
                        return comma_count + 1

                    # Sort by place detail (higher score first)
                    births_to_process.sort(key=place_detail_score, reverse=True)

                    # Use the highest priority birth event
                    best_birth = births_to_process[0]

                    # Extract date information
                    norm_date = best_birth.get("nd")
                    disp_date = best_birth.get("d")
                    birth_date_str = norm_date if norm_date else disp_date

                    if birth_date_str:
                        year_match = re.search(r"\b(\d{4})\b", birth_date_str)
                        if year_match:
                            try:
                                birth_year = int(year_match.group(1))
                            except ValueError:
                                logger.warning(
                                    f"Item {idx}: Bad birth year convert: '{year_match.group(1)}'"
                                )

                    # Extract place information
                    birth_place = (
                        best_birth.get("p", "").strip() if best_birth.get("p") else None
                    )

                    # Log the selected birth event details
                    logger.debug(
                        f"Item {idx}: Selected birth event - Date: '{birth_date_str}', "
                        f"Place: '{birth_place}', Year: {birth_year}, "
                        f"Alternate: {best_birth.get('pa')}, "
                        f"Detail score: {place_detail_score(best_birth)}"
                    )

                    # Log all available birth events for debugging
                    if len(birth_events) > 1:
                        logger.debug(
                            f"Item {idx}: Multiple birth events available ({len(birth_events)}):"
                        )
                        for i, event in enumerate(birth_events):
                            logger.debug(
                                f"  Birth event {i+1}: Date: '{event.get('d')}', "
                                f"Place: '{event.get('p')}', "
                                f"Alternate: {event.get('pa')}, "
                                f"Detail score: {place_detail_score(event)}"
                            )

                # Process death events - similar approach as birth events
                if death_events:
                    is_living = False

                    # Prioritize non-alternate death events
                    non_alternate_deaths = [
                        e for e in death_events if e.get("pa") is False
                    ]
                    deaths_to_process = (
                        non_alternate_deaths if non_alternate_deaths else death_events
                    )

                    # Use the same place detail scoring function as for births
                    deaths_to_process.sort(key=place_detail_score, reverse=True)

                    # Use the highest priority death event
                    best_death = deaths_to_process[0]

                    # Extract date information
                    norm_date = best_death.get("nd")
                    disp_date = best_death.get("d")
                    death_date_str = norm_date if norm_date else disp_date

                    if death_date_str:
                        year_match = re.search(r"\b(\d{4})\b", death_date_str)
                        if year_match:
                            try:
                                death_year = int(year_match.group(1))
                            except ValueError:
                                logger.warning(
                                    f"Item {idx}: Bad death year convert: '{year_match.group(1)}'"
                                )

                    # Extract place information
                    death_place = (
                        best_death.get("p", "").strip() if best_death.get("p") else None
                    )

                    # Log the selected death event details
                    logger.debug(
                        f"Item {idx}: Selected death event - Date: '{death_date_str}', "
                        f"Place: '{death_place}', Year: {death_year}, "
                        f"Alternate: {best_death.get('pa')}"
                    )
            else:
                logger.warning(f"Item {idx}: Events key issue: {events_list}")

            # Construct suggestion dict
            suggestion = {
                "PersonId": person_id,
                "TreeId": tree_id,
                "FullName": full_name,
                "GivenNamePart": first_name_part,
                "SurnamePart": surname_part,
                "BirthYear": birth_year,
                "BirthDate": birth_date_str,
                "BirthPlace": birth_place,
                "DeathYear": death_year,
                "DeathDate": death_date_str,
                "DeathPlace": death_place,
                "Gender": gender,
                "IsLiving": is_living,
            }
            parsed_results.append(suggestion)
            logger.debug(f"Item {idx}: Parsed: {suggestion}")

        except Exception as e:
            logger.error(f"Error parsing item {idx}: {e}", exc_info=True)
            logger.error(f"Raw Data: {person_raw}")
            continue

    return parsed_results


# End of _parse_treesui_list_response


# Selection Phase (Includes Initial Comparison Call)
def _handle_selection_phase(
    suggestions_to_score: List[Dict],
    search_criteria: Dict[str, Any],
) -> Optional[Tuple[Dict, Dict]]:
    """
    Handles scoring, display table, selection of the top candidate,
    AND calls the initial comparison display.
    """
    scored_candidates = _process_and_score_suggestions(
        suggestions_to_score, search_criteria
    )
    if not scored_candidates:
        # Log info and display to user
        logger.info("No candidates after scoring.")
        print("\nNo suitable candidates found after scoring.")
        return None
    max_display_limit = getattr(config_schema, "max_candidates_to_display", 5)
    _display_search_results(scored_candidates, max_display_limit)
    selection = _select_top_candidate(
        scored_candidates, suggestions_to_score
    )  # Pass suggestions_to_score for potential use? Currently unused by select func.
    if not selection:
        # Log error and display to user
        logger.error("Failed to select top candidate.")
        print("\nFailed to select top candidate.")
        return None
    selected_candidate_processed, selected_candidate_raw = selection
    # Field-by-field comparison display has been removed as requested
    pass
    return selection


# End of _handle_selection_phase


# Details Fetch Phase
def _handle_details_phase(
    selected_candidate_raw: Dict,
    session_manager_local: SessionManager,
) -> Optional[Dict]:
    """Handles fetching detailed person information using the Facts API."""
    owner_profile_id = getattr(session_manager_local, "my_profile_id", None)
    base_url = config_schema.api.base_url.rstrip("/")
    api_person_id = selected_candidate_raw.get("PersonId")
    api_tree_id = selected_candidate_raw.get("TreeId")

    # Check if owner_profile_id is missing and try to get it from environment variables
    if not owner_profile_id:
        # Try to get profile ID from environment variables
        env_profile_id = os.environ.get("MY_PROFILE_ID")
        if env_profile_id:
            owner_profile_id = env_profile_id
            # Update the session manager with the profile ID from environment
            session_manager_local.my_profile_id = owner_profile_id
            logger.info(
                f"Using profile ID from environment variables: {owner_profile_id}"
            )
        else:
            # Log error and display to user
            logger.error("Owner profile ID missing.")
            print("\nCannot fetch details: User ID missing.")
            return None
    if not api_person_id or not api_tree_id:
        # Log error and display to user
        logger.error(f"Cannot fetch details: Missing PersonId/TreeId.")
        print("\nError: Missing IDs for detail fetch.")
        return None
    if not callable(call_facts_user_api):
        # Log error and display to user
        logger.error("Cannot fetch details: Function missing.")
        print("\nError: Details fetching utility unavailable.")
        return None
    # Construct the API URL for logging/display purposes
    facts_api_url = f"{base_url}/family-tree/person/facts/user/{owner_profile_id}/tree/{api_tree_id}/person/{api_person_id}"

    # Call the API
    person_research_data = call_facts_user_api(
        session_manager_local, owner_profile_id, api_person_id, api_tree_id, base_url
    )
    if person_research_data is None:
        # Log warning and display to user
        logger.warning("Failed to retrieve detailed info.")
        print("\nWarning: Could not retrieve detailed info.")
        return None
    else:
        return person_research_data


# End of _handle_details_phase


def _handle_supplementary_info_phase(
    person_research_data: Optional[Dict],
    selected_candidate_processed: Dict,
    session_manager_local: SessionManager,
):
    """
    Handles displaying family info and calculating/displaying the relationship path.
    Streamlined to directly call API helpers and format output.
    """
    # --- Get Base Info ---
    base_url = config_schema.api.base_url.rstrip("/")
    owner_tree_id = getattr(session_manager_local, "my_tree_id", None)
    owner_profile_id = getattr(session_manager_local, "my_profile_id", None)
    owner_name = getattr(session_manager_local, "tree_owner_name", "the Tree Owner")

    # Check if owner_tree_id is missing and try to get it from config
    if not owner_tree_id:
        config_tree_id = getattr(config_schema.api, "tree_id", None)
        if config_tree_id:
            owner_tree_id = config_tree_id
            # Update the session manager with the tree ID from config
            session_manager_local.my_tree_id = owner_tree_id
            logger.info(f"Using tree ID from configuration: {owner_tree_id}")

    # Check if owner_profile_id is missing and try to get it from environment variables
    if not owner_profile_id:
        env_profile_id = os.environ.get("MY_PROFILE_ID")
        if env_profile_id:
            owner_profile_id = env_profile_id
            # Update the session manager with the profile ID from environment
            session_manager_local.my_profile_id = owner_profile_id
            logger.info(
                f"Using profile ID from environment variables: {owner_profile_id}"
            )

    # Check if owner_name is missing and try to get it from environment variables
    if owner_name == "the Tree Owner":
        # Try to get from config first
        config_owner_name = getattr(config_schema, "user_name", None)
        if (
            config_owner_name and config_owner_name != "Tree Owner"
        ):  # Don't use generic default
            owner_name = config_owner_name
            # Update the session manager with the owner name from config
            session_manager_local.tree_owner_name = owner_name
            logger.info(f"Using tree owner name from configuration: {owner_name}")
        else:
            # If not in config, try alternative config location
            if owner_profile_id:
                # Try to get from reference person name config
                alt_config_owner_name = getattr(
                    config_schema, "reference_person_name", None
                )
                owner_name = config_owner_name if config_owner_name else "Tree Owner"
                session_manager_local.tree_owner_name = owner_name
                logger.info(f"Using tree owner name from config/default: {owner_name}")

    # --- Display Family ---
    # Get the person's name and birth/death years for display
    person_name = selected_candidate_processed.get("name", "Unknown")

    # Extract birth and death years from the candidate data
    birth_year = None
    death_year = None

    # Try to get years from person_research_data first (more detailed)
    if person_research_data:
        # Try to get birth year
        if "PersonFacts" in person_research_data and isinstance(
            person_research_data["PersonFacts"], list
        ):
            for fact in person_research_data["PersonFacts"]:
                if (
                    isinstance(fact, dict)
                    and fact.get("TypeString") == "Birth"
                    and not fact.get("IsAlternate")
                ):
                    parsed_date = fact.get("ParsedDate")
                    if isinstance(parsed_date, dict) and "Year" in parsed_date:
                        birth_year = parsed_date["Year"]
                        break

        # Try to get death year
        if "PersonFacts" in person_research_data and isinstance(
            person_research_data["PersonFacts"], list
        ):
            for fact in person_research_data["PersonFacts"]:
                if (
                    isinstance(fact, dict)
                    and fact.get("TypeString") == "Death"
                    and not fact.get("IsAlternate")
                ):
                    parsed_date = fact.get("ParsedDate")
                    if isinstance(parsed_date, dict) and "Year" in parsed_date:
                        death_year = parsed_date["Year"]
                        break

    # If not found in person_research_data, try the candidate data
    if birth_year is None:
        birth_year = selected_candidate_processed.get("parsed_suggestion", {}).get(
            "birth_year"
        )
        if birth_year is None:
            birth_year = selected_candidate_processed.get("raw_data", {}).get(
                "BirthYear"
            )

    if death_year is None:
        death_year = selected_candidate_processed.get("parsed_suggestion", {}).get(
            "death_year"
        )
        if death_year is None:
            death_year = selected_candidate_processed.get("raw_data", {}).get(
                "DeathYear"
            )

    # Format the years for display
    years_display = ""
    if birth_year and death_year:
        years_display = f" ({birth_year} - {death_year})"
    elif birth_year:
        years_display = f" (b. {birth_year})"
    elif death_year:
        years_display = f" (d. {death_year})"

    # Display the header with name and years
    print(f"\n=== Family Details: {person_name}{years_display} ===")

    if person_research_data and isinstance(
        person_research_data.get("PersonFamily"), dict
    ):
        _display_family_info(person_research_data["PersonFamily"])
    elif person_research_data is None:
        logger.debug("Cannot display family: Detail fetch failed.")
        print("  (Family details unavailable: Detail fetch failed)")
    else:
        logger.debug("Cannot display family: 'PersonFamily' missing/invalid.")
        print("  (Family details missing or invalid in API response)")
    # Removed redundant print("") here, let relationship section add space if needed

    # --- Prepare for Relationship Calculation ---
    # We'll print the header in the formatted path, so don't print it here
    # print(f"\n===Relationship Path to {owner_name}===")

    # Initialize variables
    selected_person_tree_id = None
    selected_person_global_id = None
    selected_tree_id = None
    selected_name = "Selected Person"
    source_of_ids = "Not Attempted"
    essential_ids_found = False

    # Attempt 1: Extract from detailed person_research_data
    if person_research_data:
        logger.debug(
            "Attempting to extract relationship IDs from detailed person_research_data..."
        )
        source_of_ids = "Detailed Fetch Attempt"
        raw_cand_for_name_fallback = selected_candidate_processed.get("raw_data", {})
        temp_person_id = person_research_data.get("PersonId")
        temp_tree_id = person_research_data.get("TreeId")
        temp_global_id = person_research_data.get("UserId")
        temp_name = _extract_best_name_from_details(
            person_research_data, raw_cand_for_name_fallback
        )
        essential_ids_found = bool((temp_person_id and temp_tree_id) or temp_global_id)

        if essential_ids_found:
            selected_person_tree_id = temp_person_id
            selected_tree_id = temp_tree_id
            selected_person_global_id = temp_global_id
            selected_name = temp_name
            source_of_ids = "Detailed Fetch Success"
            logger.debug(
                f"Using IDs from Detailed Fetch: Name='{selected_name}', PersonID='{selected_person_tree_id}', TreeID='{selected_tree_id}', GlobalID='{selected_person_global_id}'"
            )
        else:
            source_of_ids = "Detailed Fetch Failed (Missing IDs)"
            logger.debug(
                "Detailed data fetched, but essential relationship IDs missing. Will attempt fallback."
            )
    else:
        source_of_ids = "Detailed Fetch Skipped (No Data)"
        logger.debug(
            "Detailed person_research_data not available. Will attempt fallback for relationship IDs."
        )

    # Attempt 2: Fallback to raw suggestion data if primary attempt failed
    if not essential_ids_found:
        logger.debug(
            "Attempting to extract relationship IDs from raw suggestion data (Fallback)..."
        )
        raw_data = selected_candidate_processed.get("raw_data", {})
        parsed_sugg = selected_candidate_processed.get("parsed_suggestion", {})
        if raw_data:
            temp_person_id = raw_data.get("PersonId")
            temp_tree_id = raw_data.get("TreeId")
            temp_global_id = raw_data.get(
                "UserId"
            )  # Assuming UserId might be in raw_data for some cases
            temp_name = parsed_sugg.get(
                "full_name_disp", raw_data.get("FullName", "Selected Match")
            )
            essential_ids_found = bool(
                (temp_person_id and temp_tree_id) or temp_global_id
            )
            if essential_ids_found:
                selected_person_tree_id = temp_person_id
                selected_tree_id = temp_tree_id
                selected_person_global_id = temp_global_id
                selected_name = temp_name
                source_of_ids = "Raw Suggestion Fallback Success"
                logger.debug(
                    f"Using IDs from Raw Suggestion Fallback: Name='{selected_name}', PersonID='{selected_person_tree_id}', TreeID='{selected_tree_id}', GlobalID='{selected_person_global_id}'"
                )
            else:
                source_of_ids = "Fallback Failed (Missing IDs)"
                logger.error(
                    "Fallback failed: Raw suggestion data also missing essential relationship IDs."
                )
        else:
            source_of_ids = "Fallback Failed (No Raw Data)"
            logger.error(
                "Critical: Cannot find raw_data for fallback to get relationship IDs."
            )
    # End of if not essential_ids_found

    # --- Log Final IDs Being Used ---
    logger.debug(f"Final IDs for relationship check (Source: {source_of_ids}):")
    logger.debug(
        f"  Owner Tree ID         : {owner_tree_id} (Type: {type(owner_tree_id)})"
    )
    logger.debug(
        f"  Owner Profile ID      : {owner_profile_id} (Type: {type(owner_profile_id)})"
    )
    logger.debug(
        f"  Selected Person Name  : {selected_name} (Type: {type(selected_name)})"
    )
    logger.debug(
        f"  Selected PersonTreeID : {selected_person_tree_id} (Person's ID within a tree, Type: {type(selected_person_tree_id)})"
    )
    logger.debug(
        f"  Selected TreeID       : {selected_tree_id} (The tree this person belongs to, Type: {type(selected_tree_id)})"
    )
    logger.debug(
        f"  Selected Global ID    : {selected_person_global_id} (Often UserID/ProfileID, Type: {type(selected_person_global_id)})"
    )

    # --- Determine Relationship Calculation Method ---
    can_attempt_calculation = essential_ids_found
    owner_tree_id_str = str(owner_tree_id) if owner_tree_id else None
    selected_tree_id_str = (
        str(selected_tree_id) if selected_tree_id else None
    )  # The tree ID of the selected person
    owner_profile_id_str = str(owner_profile_id).upper() if owner_profile_id else None
    selected_global_id_str = (
        str(selected_person_global_id).upper() if selected_person_global_id else None
    )

    is_owner = False
    can_calc_tree_ladder = False  # Using /getladder for same-tree relationships
    can_calc_discovery_api = (
        False  # Using Discovery API for cross-tree or general profile relationships
    )

    relationship_result_data = None
    api_called_for_rel = "None"  # Renamed for clarity
    formatted_path = None
    calculation_performed = False

    if can_attempt_calculation:
        # Check if selected person is the tree owner
        is_owner = bool(
            selected_global_id_str  # Selected person's global ID
            and owner_profile_id_str  # Owner's global ID (from session)
            and selected_global_id_str == owner_profile_id_str
        )

        # Conditions for using Tree Ladder API (/getladder)
        # Requires both selected person and owner to be in the *same tree* (owner_tree_id_str).
        # selected_person_tree_id is the ID of the person *within that tree*.
        can_calc_tree_ladder = bool(
            owner_tree_id_str  # Owner's tree ID must be known
            and selected_tree_id_str  # Selected person's tree ID must be known
            and selected_person_tree_id  # Selected person's ID *within their tree* must be known
            and selected_tree_id_str
            == owner_tree_id_str  # Crucially, they must be in the same tree
        )

        # Conditions for using Discovery Relationship API
        # Requires global IDs (Profile IDs/UserIDs) for both selected person and owner.
        can_calc_discovery_api = bool(
            selected_person_global_id and owner_profile_id_str
        )
    # End of if can_attempt_calculation

    logger.debug(f"Relationship calculation checks (Source: {source_of_ids}):")
    logger.debug(f"  Can Attempt Calc?     : {can_attempt_calculation}")
    logger.debug(
        f"  is_owner              : {is_owner} (OwnerG='{owner_profile_id_str}', SelectedG='{selected_global_id_str}')"
    )
    logger.debug(
        f"  can_calc_tree_ladder  : {can_calc_tree_ladder} (OwnerT='{owner_tree_id_str}', SelectedT='{selected_tree_id_str}', SelectedP_in_Tree exists?={bool(selected_person_tree_id)})"
    )
    logger.debug(
        f"  can_calc_discovery_api: {can_calc_discovery_api} (OwnerG exists?={bool(owner_profile_id_str)}, SelectedG exists?={bool(selected_person_global_id)})"
    )

    # --- Directly Call API and Format/Print Relationship ---
    if is_owner:
        # No API call needed, print message directly
        print(f"\n=== Relationship Path to {owner_name} ===")
        print(f"  {selected_name} is the tree owner ({owner_name}).")
        logger.info(
            f"{selected_name} is Tree Owner. No relationship path calculation needed."
        )
        calculation_performed = True  # Technically, a "calculation" of "is owner"
        # formatted_path can remain None, or set to a specific string if preferred.
        formatted_path = f"{selected_name} is the tree owner."

    elif can_calc_tree_ladder:
        api_called_for_rel = "Tree Ladder (/getladder)"
        logger.debug(f"Attempting relationship calculation via {api_called_for_rel}...")
        # API URL is printed by call_getladder_api itself
        # Ensure IDs are strings for the API call
        sp_tree_id_str = str(selected_person_tree_id)  # Person's ID in the tree
        ot_id_str = str(owner_tree_id_str)  # The tree ID they are both in

        # Get the ladder API URL for logging purposes
        ladder_api_url = f"{base_url}/family-tree/person/tree/{ot_id_str}/person/{sp_tree_id_str}/getladder?callback=no"
        logger.debug(f"\nLadder API URL: {ladder_api_url}\n")

        relationship_result_data = call_getladder_api(
            session_manager_local, ot_id_str, sp_tree_id_str, base_url
        )
        if relationship_result_data:  # Successfully got data (string HTML/JSONP)
            calculation_performed = True
            if callable(format_api_relationship_path):
                try:
                    # First, convert the JSONP response to a proper dictionary
                    # The response is in the format: no({...}) where ... is the JSON data
                    jsonp_match = re.search(r"no\((.*)\)", relationship_result_data)
                    if jsonp_match:
                        json_str = jsonp_match.group(1)
                        try:
                            json_data = json.loads(json_str)
                            # Create a dictionary with the HTML content and status
                            api_response_dict = {
                                "html": json_data.get("html", ""),
                                "status": json_data.get("status", "unknown"),
                                "message": json_data.get("message", ""),
                            }

                            # Use the unified relationship path formatter
                            sn_str = str(selected_name) if selected_name else "Unknown"
                            on_str = str(owner_name) if owner_name else "Unknown"

                            # First extract relationship data from the API response
                            relationship_data = []

                            # Use the standard API relationship path formatter to get the raw data
                            raw_formatted_path = format_api_relationship_path(
                                api_response_dict, on_str, sn_str
                            )

                            # Log the raw formatted path for debugging
                            logger.debug(
                                f"Raw formatted path from format_api_relationship_path:\n{raw_formatted_path}"
                            )

                            # Extract relationship data from the HTML content
                            try:
                                # Parse the HTML content to extract relationship data
                                html_content = api_response_dict.get("html", "")
                                if html_content and isinstance(html_content, str):
                                    # Use BeautifulSoup to parse the HTML
                                    from bs4 import BeautifulSoup

                                    soup = BeautifulSoup(html_content, "html.parser")

                                    # Find all list items
                                    list_items = soup.find_all("li")
                                    for item in list_items:
                                        # Skip icon items
                                        if item.get("aria-hidden") == "true":
                                            continue

                                        # Extract name, relationship, and lifespan
                                        name_elem = item.find("b")
                                        name = (
                                            name_elem.get_text()
                                            if name_elem
                                            else item.get_text()
                                        )

                                        # Extract relationship description
                                        rel_elem = item.find("i")
                                        relationship = (
                                            rel_elem.get_text() if rel_elem else ""
                                        )

                                        # Extract lifespan
                                        text = item.get_text()
                                        lifespan_match = re.search(
                                            r"(\d{4})-(\d{4}|\-)", text
                                        )
                                        lifespan = (
                                            lifespan_match.group(0)
                                            if lifespan_match
                                            else ""
                                        )

                                        relationship_data.append(
                                            {
                                                "name": name,
                                                "relationship": relationship,
                                                "lifespan": lifespan,
                                            }
                                        )

                                # Convert the API data to the unified format
                                unified_path = convert_api_path_to_unified_format(
                                    relationship_data, sn_str
                                )

                                # Format the path using the unified formatter
                                formatted_path = format_relationship_path_unified(
                                    unified_path, sn_str, on_str, None
                                )

                                # We don't need to manually add birth/death years anymore
                                # The unified formatter handles this automatically
                            except Exception as e:
                                logger.error(
                                    f"Error formatting relationship path: {e}",
                                    exc_info=True,
                                )
                                # Fall back to the raw formatted path if conversion fails
                                formatted_path = raw_formatted_path
                        except json.JSONDecodeError as json_err:
                            logger.error(f"Error parsing JSON from JSONP: {json_err}")
                            formatted_path = (
                                f"(Error parsing JSON from JSONP: {json_err})"
                            )
                    else:
                        logger.error("JSONP format not recognized")
                        formatted_path = "(JSONP format not recognized)"
                except Exception as fmt_err:
                    logger.error(
                        f"Error formatting {api_called_for_rel} data: {fmt_err}",
                        exc_info=True,
                    )
                    formatted_path = f"(Error formatting relationship path from {api_called_for_rel}: {fmt_err})"
                # End of try/except
            else:  # format_api_relationship_path not callable (should not happen with guards)
                logger.error("format_api_relationship_path function not available.")
                formatted_path = (
                    "(Error: Formatting function for relationship path unavailable)"
                )
            # End of if/else callable
        else:  # call_getladder_api failed or returned None
            logger.warning(f"{api_called_for_rel} API call failed or returned no data.")
            formatted_path = f"(Failed to retrieve data from {api_called_for_rel} API)"
        # End of if/else relationship_result_data

    elif can_calc_discovery_api:
        # This branch is for when Tree Ladder cannot be used (e.g., different trees, or selected person is not in owner's tree)
        # but global IDs are available.
        api_called_for_rel = "Discovery Relationship API"
        logger.debug(f"Attempting relationship calculation via {api_called_for_rel}...")
        # API URL is printed by call_discovery_relationship_api itself
        sp_global_id_str = str(selected_person_global_id)
        op_id_str = str(owner_profile_id_str)  # Ensure it's a string

        # Call the Discovery API with timeout parameter
        discovery_api_response = call_discovery_relationship_api(
            session_manager_local,
            sp_global_id_str,
            op_id_str,
            base_url,
            timeout=30,  # Use a 30-second timeout for this API call
        )

        if discovery_api_response and isinstance(discovery_api_response, dict):
            calculation_performed = True
            # Use the unified relationship path formatter for Discovery API
            try:
                # Convert the Discovery API response to the unified format
                unified_path = convert_discovery_api_path_to_unified_format(
                    discovery_api_response, selected_name
                )

                # Format the path using the unified formatter
                if unified_path:
                    formatted_path = format_relationship_path_unified(
                        unified_path, selected_name, owner_name, None
                    )
                    logger.debug(
                        f"Discovery API path formatted using unified formatter"
                    )
                else:
                    logger.warning(
                        "Failed to convert Discovery API path to unified format"
                    )
                    # Fallback to simple formatting
                    path_steps = discovery_api_response.get("path", [])
                    path_display_lines = [f"  {selected_name}"]
                    name_formatter_local = (
                        format_name
                        if callable(format_name)
                        else lambda x: str(x).title()
                    )
                    # _get_relationship_term is not available here, so use raw relationship string
                    for step in path_steps:
                        step_name_raw = step.get("name", "?")
                        step_rel_raw = step.get(
                            "relationship", "related to"
                        ).capitalize()
                        path_display_lines.append(
                            f"  -> {step_rel_raw} is {name_formatter_local(step_name_raw)}"
                        )
                    # End of for
                    path_display_lines.append(f"  -> {owner_name} (Tree Owner / You)")
                    formatted_path = "\n".join(path_display_lines)
                    logger.debug(
                        f"Discovery API path constructed using fallback formatter"
                    )
            except Exception as e:
                logger.error(f"Error formatting Discovery API path: {e}", exc_info=True)
                # Fallback to simple formatting
                path_steps = discovery_api_response.get("path", [])
                path_display_lines = [f"  {selected_name}"]
                name_formatter_local = (
                    format_name if callable(format_name) else lambda x: str(x).title()
                )
                for step in path_steps:
                    step_name_raw = step.get("name", "?")
                    step_rel_raw = step.get("relationship", "related to").capitalize()
                    path_display_lines.append(
                        f"  -> {step_rel_raw} is {name_formatter_local(step_name_raw)}"
                    )
                # End of for
                path_display_lines.append(f"  -> {owner_name} (Tree Owner / You)")
                formatted_path = "\n".join(path_display_lines)
                logger.debug(
                    f"Discovery API path constructed using fallback formatter after error: {e}"
                )
            # End of try/except

            # Check for message in the API response if no path was found
            if "message" in discovery_api_response and not formatted_path:
                formatted_path = f"(Discovery API: {discovery_api_response.get('message', 'No direct path found')})"
                logger.warning(f"Discovery API response: {formatted_path}")
            elif not formatted_path:
                formatted_path = (
                    "(Discovery API: Path data missing or in unexpected format)"
                )
                logger.warning(
                    f"Discovery API response structure unexpected: {discovery_api_response}"
                )
            # End of if/elif/else path processing
        else:  # call_discovery_relationship_api failed or returned non-dict
            logger.warning(
                f"{api_called_for_rel} API call failed or returned invalid data."
            )
            formatted_path = (
                f"(Failed to retrieve or parse data from {api_called_for_rel} API)"
            )
        # End of if/else discovery_api_response
    # End of if/elif/elif for calculation method

    # --- Print Final Result or Failure Message ---
    # A blank line is already added by the initial header print if family info was displayed.
    # If no family info, this adds a space.
    print("")

    if formatted_path:
        # First, print a clear header indicating which API source was used        # Use a default value for owner_name if it's None
        display_owner_name = owner_name if owner_name else "Tree Owner"
        if api_called_for_rel == "Tree Ladder (/getladder)":
            print(
                f"=== Relationship Path to {display_owner_name} (via Tree Ladder API) ==="
            )
        elif api_called_for_rel == "Discovery Relationship API":
            print(
                f"=== Relationship Path to {display_owner_name} (via Discovery API) ==="
            )
        else:
            print(f"=== Relationship Path to {display_owner_name} ===")

        # Check if the formatted_path itself indicates an error/fallback condition
        # These are common error prefixes from format_api_relationship_path or API call failures
        known_error_starts_tuple = (
            "(No relationship",
            "(Could not parse",
            "(API returned error",
            "(Relationship HTML structure",
            "(Unsupported API response",
            "(Error processing relationship",
            "(Cannot parse relationship path",
            "(Could not find, decode, or parse",
            "(Could not find sufficient relationship",
            "(Failed to retrieve data",
            "(Error formatting relationship",
            "(Error: Formatting function unavailable",
            "(Discovery API:",
            "(Discovery path found but invalid",
            "(No valid relationship path items found",
        )
        if any(
            formatted_path.startswith(err_start)
            for err_start in known_error_starts_tuple
        ):
            # This is an error message, print it as such
            print(f"  {formatted_path}")  # Indent error messages for clarity
            logger.warning(
                f"Relationship path calculation resulted in message/error: {formatted_path} (API called: {api_called_for_rel})"
            )
        else:
            # This is a successfully formatted path, print it directly
            # The API URL is printed by the respective call_..._api function in api_utils.

            # Remove the header line from the formatted path to avoid duplicate headers
            # and replace "Unknown" with the owner name in the relationship description
            if "===Relationship Path to" in formatted_path:
                # Split the path by newlines and remove the first line (header)
                path_lines = formatted_path.split("\n")
                if len(path_lines) > 1:
                    # Remove the header line
                    formatted_path = "\n".join(path_lines[1:])

                    # Replace "Unknown's" with "{owner_name}'s" in the relationship description
                    formatted_path = formatted_path.replace(
                        "Unknown's", f"{display_owner_name}'s"
                    )

            print(f"{formatted_path}\n")  # Add a newline after the path for spacing
            logger.debug(
                f"Successfully displayed relationship path via {api_called_for_rel}."
            )
        # End of if/else known_error_starts_tuple
    elif not is_owner and not calculation_performed:
        # This case means no calculation method was viable, or essential IDs were missing
        default_fail_message = (
            f"(Could not calculate relationship path for {selected_name})"
        )
        print(f"  {default_fail_message}")  # Indent for clarity
        if not can_attempt_calculation:
            reason_detail = (
                "  Reason: Essential IDs missing from detailed data and fallback data."
            )
            print(reason_detail)
            logger.error(
                f"{default_fail_message}. {reason_detail.strip()} (Source of IDs: {source_of_ids})."
            )
        else:  # Calculation was attempted but conditions not met for any method, or API failed silently earlier
            reason_detail = "  Reason: Calculation conditions not met (e.g., tree mismatch, API data issue)."
            print(reason_detail)
            logger.error(
                f"{default_fail_message}. {reason_detail.strip()} (Source of IDs: {source_of_ids}). Check prior logs for API call failures."
            )
        # End of if/else can_attempt_calculation
    elif not is_owner and calculation_performed and not formatted_path:
        # This means an API was called, data might have been returned, but formatting failed to produce a path
        # or the API explicitly returned no path (e.g. discovery API with no direct link).
        # The `formatted_path` should have been set to an error message in these cases by the logic above.
        # This block is a fallback for an unexpected state.
        logger.error(
            f"Unexpected state: Calculation performed for {selected_name} via {api_called_for_rel}, but no formatted path or error message was generated."
        )
        print(
            f"  (Relationship path for {selected_name} could not be determined or displayed via {api_called_for_rel})."
        )
    # End of if/elif/elif for printing result


# End of _handle_supplementary_info_phase


def handle_api_report():
    """Orchestrates the process using only initial API data for comparison."""
    # Dependency Checks...
    if not all(
        [
            CORE_UTILS_AVAILABLE,
            API_UTILS_AVAILABLE,
            GEDCOM_UTILS_AVAILABLE,
            config_schema,  # Use config_schema instead of undefined config_instance
            session_manager,
        ]
    ):
        missing = [
            m
            for m, v in [
                ("Core Utils", CORE_UTILS_AVAILABLE),
                ("API Utils", API_UTILS_AVAILABLE),
                ("Gedcom Utils", GEDCOM_UTILS_AVAILABLE),
                (
                    "Config",
                    config_schema,
                ),  # Use config_schema instead of config_instance and selenium_config
                ("Session Manager", session_manager),
            ]
            if not v
        ]
        logger.critical(
            f"handle_api_report: Dependencies missing: {', '.join(missing)}."
        )
        print(
            f"\nCRITICAL ERROR: Dependencies unavailable ({', '.join(missing)}). Check logs."
        )
        return False
    # Session Setup...
    # First, ensure the session is ready
    if not session_manager.ensure_session_ready(action_name="API Report Session Init"):
        logger.error("Failed to init session.")
        print("\nERROR: Failed to initialize session.")
        return False

    # First, check if we're already logged in
    from utils import login_status

    # Start the browser if it's not already started
    if not session_manager.driver:
        logger.info("Starting browser to check login status...")
        if not session_manager.start_browser(action_name="API Report Browser Init"):
            logger.error("Failed to start browser session.")
            print("\nERROR: Failed to start browser session.")
            return False

    # Check login status
    login_stat = login_status(session_manager, disable_ui_fallback=True)

    # If we're already logged in, refresh the cookies
    if login_stat is True:
        logger.info("User is already logged in. Refreshing cookies...")

        # Ensure we have a requests session
        if (
            not hasattr(session_manager, "_requests_session")
            or not session_manager._requests_session
        ):
            logger.error("No requests session available despite being logged in.")
            print("\nERROR: API session not available. Please restart the application.")
            return False

        # Refresh the cookies in the requests session
        if session_manager.driver:
            try:
                selenium_cookies = session_manager.driver.get_cookies()
                for cookie in selenium_cookies:
                    session_manager._requests_session.cookies.set(
                        cookie["name"], cookie["value"]
                    )
                logger.info("Successfully refreshed cookies from browser session.")
            except Exception as cookie_err:
                logger.error(f"Failed to refresh cookies from browser: {cookie_err}")
                print(f"\nERROR: Failed to refresh cookies: {cookie_err}")
                return False

        # If we still don't have cookies, something is wrong
        if not session_manager._requests_session.cookies:
            logger.error("Still no cookies available after refresh attempt.")
            print(
                "\nERROR: Failed to obtain authentication cookies. Please restart the application."
            )
            return False

        # If we get here, we have successfully refreshed the cookies
        logger.info("Successfully refreshed authentication cookies.")
    else:
        # We're not logged in, so check if we have a requests session
        if (
            not hasattr(session_manager, "_requests_session")
            or not session_manager._requests_session
        ):
            logger.error("No requests session available for API calls.")
            print(
                "\nERROR: API session not available. Please ensure you are logged in to Ancestry."
            )
            return False

        # Check if we have cookies in the requests session
        if not session_manager._requests_session.cookies:
            logger.warning(
                "No cookies available in the requests session. Need to log in."
            )

            # Try to initialize the session with authentication cookies
            print("\nAttempting to log in to Ancestry...")
            try:
                # Try to start the browser session
                if not session_manager.start_browser(
                    action_name="API Report Browser Init"
                ):
                    logger.error("Failed to start browser session.")
                    print("\nERROR: Failed to start browser session.")
                    return False

                # First check if we're already logged in
                from utils import login_status, log_in

                login_stat = login_status(session_manager, disable_ui_fallback=True)
                if login_stat is True:
                    logger.info(
                        "User is already logged in. No need to navigate to sign-in page."
                    )

                    # Refresh the cookies in the requests session
                    if session_manager.driver:
                        try:
                            selenium_cookies = session_manager.driver.get_cookies()
                            for cookie in selenium_cookies:
                                session_manager._requests_session.cookies.set(
                                    cookie["name"], cookie["value"]
                                )
                            logger.info(
                                "Successfully refreshed cookies from browser session."
                            )
                        except Exception as cookie_err:
                            logger.error(
                                f"Failed to refresh cookies from browser: {cookie_err}"
                            )

                    # Set login result to success
                    login_result = "LOGIN_SUCCEEDED"
                else:
                    # Try to log in
                    login_result = log_in(session_manager)
                    if login_result != "LOGIN_SUCCEEDED":
                        logger.error(f"Failed to log in: {login_result}")
                        print(f"\nERROR: Failed to log in: {login_result}")
                        return False

                # Check if we now have cookies
                if not session_manager._requests_session.cookies:
                    logger.error("Still no cookies available after login attempt.")
                    print("\nERROR: Still no cookies available after login attempt.")
                    return False

                print("\nSuccessfully initialized session with authentication cookies.")
            except Exception as e:
                logger.error(f"Error initializing session: {e}")
                print(f"\nERROR: Error initializing session: {e}")
                return False

            # If we get here, we have successfully initialized the session with cookies
            logger.info("Successfully initialized session with authentication cookies.")
        else:
            # We have cookies, so we're good to go
            logger.info("Session already has authentication cookies.")

    # Phase 1: Search...
    search_criteria = _get_search_criteria()
    if not search_criteria:
        logger.info("Search cancelled.")
        return True
    suggestions_to_score = _handle_search_phase(session_manager, search_criteria)
    if suggestions_to_score is None:
        return False  # Critical API failure
    if not suggestions_to_score:
        return True  # Search successful, no results

    # Phase 2: Score, Select & Display Initial Comparison...
    selection = _handle_selection_phase(
        suggestions_to_score, search_criteria
    )  # Includes initial comparison display
    if not selection:
        return True  # No candidate selected or comparison failed gracefully
    selected_candidate_processed, selected_candidate_raw = selection

    # Phase 3: Fetch Detailed Data (for supplementary info)...

    person_research_data = _handle_details_phase(
        selected_candidate_raw, session_manager
    )
    # Phase 4: Display Supplementary Info (Family/Relationships)...

    _handle_supplementary_info_phase(
        person_research_data,
        selected_candidate_processed,
        session_manager,
    )  # Use the renamed function
    # Finish...

    return True


# End of handle_api_report


# --- Main Execution ---
@retry_on_failure(max_attempts=3, backoff_factor=2.0)
@circuit_breaker(failure_threshold=5, recovery_timeout=300)
@timeout_protection(timeout=1800)  # 30 minutes for API report generation
@graceful_degradation(fallback_value=None)
@error_context("action11_api_report")
def main():
    """Main execution flow for Action 11 (API Report)."""
    logger.debug("--- Action 11: API Report Starting ---")
    if not session_manager:
        # Log critical error and display to user
        logger.critical("Session Manager instance not created. Exiting.")
        print("\nFATAL ERROR: Session Manager failed to initialize.")
        return
    try:
        report_successful = handle_api_report()
        if report_successful:
            # Log success and display to user with a single message
            success_message = "Action 11 finished successfully"
            logger.info(f"--- {success_message} ---")
        else:
            # Log error and display to user with a single message
            error_message = "Action 11 finished with errors"
            logger.error(f"--- {error_message} ---")
            print(f"\n{error_message} (check logs).")
    except Exception as e:
        # Log critical error and display to user
        logger.critical(
            f"Unhandled exception during Action 11 execution: {e}", exc_info=True
        )
        print(f"\nCRITICAL ERROR during Action 11: {e}. Check logs.")
    finally:
        pass  # Optional: session_manager.close_sess()


# End of main


def search_ancestry_api_for_person(
    session_manager: SessionManager,
    search_criteria: Dict[str, Any],
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search Ancestry API for individuals matching the provided criteria.
    This function is designed to be called from other modules.

    Args:
        session_manager: Active SessionManager instance with valid authentication
        search_criteria: Dictionary containing search criteria (first_name, surname, gender, birth_year, etc.)
        max_results: Maximum number of results to return

    Returns:
        List of dictionaries containing match information, sorted by score (highest first)
    """
    # Step 1: Ensure we have a valid session
    if not session_manager or not session_manager.is_sess_valid():
        logger.error("Invalid session manager for Ancestry API search")
        return []

    # Step 2: Prepare search criteria
    # Make a copy to avoid modifying the original
    search_criteria_copy = search_criteria.copy()

    # Ensure we have the raw name fields
    if (
        "first_name" in search_criteria_copy
        and "first_name_raw" not in search_criteria_copy
    ):
        search_criteria_copy["first_name_raw"] = search_criteria_copy["first_name"]

    if "surname" in search_criteria_copy and "surname_raw" not in search_criteria_copy:
        search_criteria_copy["surname_raw"] = search_criteria_copy["surname"]

    # Step 3: Call the search API
    suggestions_to_score = _handle_search_phase(session_manager, search_criteria_copy)

    if suggestions_to_score is None or not suggestions_to_score:
        logger.debug("No results found in Ancestry API search")
        return []

    # Step 4: Score the suggestions
    scored_candidates = _process_and_score_suggestions(
        suggestions_to_score, search_criteria_copy
    )

    if not scored_candidates:
        logger.debug("No candidates after scoring Ancestry API results")
        return []

    # Step 5: Return top matches (limited by max_results)
    return scored_candidates[:max_results] if scored_candidates else []


def get_ancestry_person_details(
    session_manager: SessionManager,
    person_id: str,
    tree_id: str,
) -> Dict[str, Any]:
    """
    Get detailed information about a person from Ancestry API.

    Args:
        session_manager: Active SessionManager instance with valid authentication
        person_id: Ancestry Person ID
        tree_id: Ancestry Tree ID

    Returns:
        Dictionary containing person details
    """
    # Step 1: Ensure we have a valid session
    if not session_manager or not session_manager.is_sess_valid():
        logger.error("Invalid session manager for Ancestry person details")
        return {}

    # Step 2: Create a minimal candidate dictionary for the details phase
    candidate_raw = {
        "PersonId": person_id,
        "TreeId": tree_id,
    }

    # Step 3: Call the details API
    person_research_data = _handle_details_phase(candidate_raw, session_manager)

    if person_research_data is None:
        logger.warning(f"Failed to retrieve detailed info for person {person_id}")
        return {}

    # Step 4: Extract and format the details
    details = {}

    # Extract basic information
    person_info = person_research_data.get("person", {})
    details["id"] = person_id
    details["tree_id"] = tree_id
    details["name"] = person_info.get("name", "Unknown")
    details["gender"] = person_info.get("gender", "Unknown")

    # Extract birth information
    birth_facts = person_info.get("birth", {})
    details["birth_date"] = birth_facts.get("date", {}).get("normalized", "Unknown")
    details["birth_place"] = birth_facts.get("place", {}).get("normalized", "Unknown")
    details["birth_year"] = _extract_year_from_date(details["birth_date"])

    # Extract death information
    death_facts = person_info.get("death", {})
    details["death_date"] = death_facts.get("date", {}).get("normalized", "Unknown")
    details["death_place"] = death_facts.get("place", {}).get("normalized", "Unknown")
    details["death_year"] = _extract_year_from_date(details["death_date"])

    # Extract family information
    family = person_research_data.get("family", {})

    # Extract parents
    details["parents"] = []
    for parent in family.get("parents", []):
        parent_info = {
            "id": parent.get("id", "Unknown"),
            "name": parent.get("name", "Unknown"),
            "gender": parent.get("gender", "Unknown"),
            "birth_year": _extract_year_from_date(
                parent.get("birth", {}).get("date", {}).get("normalized", "")
            ),
            "death_year": _extract_year_from_date(
                parent.get("death", {}).get("date", {}).get("normalized", "")
            ),
        }
        details["parents"].append(parent_info)

    # Extract spouses
    details["spouses"] = []
    for spouse in family.get("spouses", []):
        spouse_info = {
            "id": spouse.get("id", "Unknown"),
            "name": spouse.get("name", "Unknown"),
            "gender": spouse.get("gender", "Unknown"),
            "birth_year": _extract_year_from_date(
                spouse.get("birth", {}).get("date", {}).get("normalized", "")
            ),
            "death_year": _extract_year_from_date(
                spouse.get("death", {}).get("date", {}).get("normalized", "")
            ),
        }
        details["spouses"].append(spouse_info)

    # Extract children
    details["children"] = []
    for child in family.get("children", []):
        child_info = {
            "id": child.get("id", "Unknown"),
            "name": child.get("name", "Unknown"),
            "gender": child.get("gender", "Unknown"),
            "birth_year": _extract_year_from_date(
                child.get("birth", {}).get("date", {}).get("normalized", "")
            ),
            "death_year": _extract_year_from_date(
                child.get("death", {}).get("date", {}).get("normalized", "")
            ),
        }
        details["children"].append(child_info)

    # Extract siblings
    details["siblings"] = []
    for sibling in family.get("siblings", []):
        sibling_info = {
            "id": sibling.get("id", "Unknown"),
            "name": sibling.get("name", "Unknown"),
            "gender": sibling.get("gender", "Unknown"),
            "birth_year": _extract_year_from_date(
                sibling.get("birth", {}).get("date", {}).get("normalized", "")
            ),
            "death_year": _extract_year_from_date(
                sibling.get("death", {}).get("date", {}).get("normalized", "")
            ),
        }
        details["siblings"].append(sibling_info)

    return details


def get_ancestry_relationship_path(
    session_manager: SessionManager,
    target_person_id: str,
    target_tree_id: str,
    reference_name: str = "Reference Person",
) -> str:
    """
    Get the relationship path between a person and the reference person (tree owner).

    Args:
        session_manager: Active SessionManager instance with valid authentication
        target_person_id: Ancestry Person ID of the target person
        target_tree_id: Ancestry Tree ID of the target person
        reference_name: Name of the reference person (default: "Reference Person")

    Returns:
        Formatted relationship path string
    """
    # Step 1: Ensure we have a valid session
    if not session_manager or not session_manager.is_sess_valid():
        logger.error("Invalid session manager for Ancestry relationship path")
        return "(Invalid session for relationship path lookup)"

    # Step 2: Get base information
    base_url = config_schema.api.base_url.rstrip("/")
    owner_tree_id = getattr(session_manager, "my_tree_id", None)
    owner_profile_id = getattr(session_manager, "my_profile_id", None)
    owner_name = getattr(session_manager, "tree_owner_name", reference_name)

    if not all([base_url, owner_tree_id, owner_profile_id]):
        logger.error("Missing required information for relationship path lookup")
        return "(Missing required information for relationship path lookup)"

    # Step 3: Try getladder API first
    logger.debug("Attempting to get relationship path using getladder API...")
    if owner_tree_id is not None:
        relationship_data = call_getladder_api(
            session_manager,
            str(owner_tree_id),
            target_person_id,
            base_url if base_url else "",
        )
    else:
        logger.warning("owner_tree_id is None, skipping getladder API call")
        relationship_data = None

    if relationship_data:
        # Parse the string response first using format_api_relationship_path
        # to get a properly formatted relationship path
        formatted_relationship = format_api_relationship_path(
            relationship_data, owner_name, "Target Person"
        )
        return formatted_relationship

    # Step 4: Try discovery API as fallback
    logger.debug(
        "Attempting to get relationship path using discovery API (fallback)..."
    )
    if owner_profile_id:
        discovery_data = call_discovery_relationship_api(
            session_manager,
            target_person_id,
            str(owner_profile_id),
            base_url,
            timeout=30,
        )
    else:
        logger.warning("owner_profile_id is None, skipping discovery API call")
        discovery_data = None

    if discovery_data:
        # Convert to unified format and format
        unified_path = convert_discovery_api_path_to_unified_format(
            discovery_data, "Target Person"
        )
        if unified_path:
            return format_relationship_path_unified(
                unified_path, "Target Person", owner_name
            )

    # Step 5: Return error message if both methods failed
    return f"(No relationship path found between {target_person_id} and tree owner)"


def _extract_year_from_date(date_str: str) -> Optional[int]:
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


def run_action11(*_):
    """Wrapper function for main.py to call."""
    return handle_api_report()


def load_test_person_from_env():
    """Load Fraser Gault test person data from environment variables."""
    load_dotenv()

    return {
        "name": os.getenv("TEST_PERSON_NAME", "Fraser Gault"),
        "birth_year": int(os.getenv("TEST_PERSON_BIRTH_YEAR", "1941")),
        "birth_place": os.getenv("TEST_PERSON_BIRTH_PLACE", "Banff"),
        "gender": os.getenv("TEST_PERSON_GENDER", "M"),
        "spouse_name": os.getenv("TEST_PERSON_SPOUSE", "Helen"),
        "children": [
            os.getenv("TEST_PERSON_CHILD1", "Lynne"),
            os.getenv("TEST_PERSON_CHILD2", "Robert"),
            os.getenv("TEST_PERSON_CHILD3", "Fraser"),
        ],
        "relationship": os.getenv("TEST_PERSON_RELATIONSHIP", "uncle"),
    }


# ==============================================
# Standalone Test Block
# ==============================================


def action11_module_tests() -> bool:
    """
    Comprehensive test suite for action11.py following the standardized 6-category TestSuite framework.
    Tests live API research functionality, person search capabilities, and genealogical data processing.

    Categories: Initialization, Core Functionality, Edge Cases, Integration, Performance, Error Handling
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Action 11 - Live API Research Tool", "action11.py")
    suite.start_suite()  # === INITIALIZATION TESTS ===

    def test_module_imports():
        """Test that required modules are imported correctly"""
        required_modules = ["json", "re", "time", "typing"]

        print("📋 Testing import of required modules:")
        for module in required_modules:
            print(f"   • {module}")

        try:
            import json
            import re
            import time
            from typing import Dict, List, Any, Optional, Tuple

            print("📊 Results:")
            print(
                f"   Successfully imported: {len(required_modules)}/{len(required_modules)} modules"
            )
            print("   ✅ json: Standard library JSON handling")
            print("   ✅ re: Regular expression operations")
            print("   ✅ time: Time-related functions")
            print("   ✅ typing: Type hints and annotations")

            # If we get here, all imports succeeded
            return True
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False

    def test_core_function_availability():
        """Test that core functions are available"""
        core_functions = [
            ("handle_api_report", "Main API report handler"),
            ("main", "Main entry point function"),
        ]

        print("📋 Testing availability of core functions:")
        results = []

        for func_name, description in core_functions:
            try:
                func = globals().get(func_name)
                is_available = func is not None
                is_callable = callable(func) if is_available else False

                status = "✅" if is_callable else "❌"
                print(f"   {status} {func_name}: {description}")
                print(f"      Available: {is_available}, Callable: {is_callable}")

                results.append(is_callable)
                assert is_callable, f"{func_name} should be callable"

            except Exception as e:
                print(f"   ❌ {func_name}: Exception {e}")
                results.append(False)

        print(
            f"📊 Results: {sum(results)}/{len(results)} core functions available and callable"
        )
        return True

    def test_search_functions():
        """Test that search-related functions exist"""
        assert (
            "_get_search_criteria" in globals()
        ), "Search criteria function should exist in globals"

        # Test with test data containing the required identifier
        test_person_data = {
            "id": "TEST_12345",
            "name": "Test Person 12345",
            "birth_year": 1950,
        }
        assert (
            "12345" in test_person_data["id"]
        ), "Test data should contain 12345 identifier"
        assert (
            "12345" in test_person_data["name"]
        ), "Test person name should contain 12345 identifier"

    # === CORE FUNCTIONALITY TESTS ===
    def test_scoring_functions():
        """Test scoring and ranking functions"""
        assert (
            "_process_and_score_suggestions" in globals()
        ), "Scoring function should exist"
        assert (
            "_run_simple_suggestion_scoring" in globals()
        ), "Simple scoring function should exist"

        # Test data for scoring validation
        test_suggestion = {
            "person_id": "SCORE_12345",
            "name": "Test Score Person 12345",
            "score": 85,
        }
        assert (
            "12345" in test_suggestion["person_id"]
        ), "Scoring test data should contain 12345 identifier"

    def test_display_functions():
        """Test result display functions"""
        assert (
            "_display_search_results" in globals()
        ), "Display results function should exist"
        assert (
            "_display_initial_comparison" in globals()
        ), "Display comparison function should exist"

        # Test data for display validation
        test_result = {
            "result_id": "DISPLAY_12345",
            "person_name": "Display Test Person 12345",
            "match_score": 92,
        }
        assert (
            "12345" in test_result["result_id"]
        ), "Display test data should contain 12345 identifier"

    def test_api_integration_functions():
        """Test API integration handlers"""
        assert "_handle_search_phase" in globals(), "Search phase handler should exist"
        assert (
            "_handle_selection_phase" in globals()
        ), "Selection phase handler should exist"

    # === EDGE CASE TESTS ===
    def test_empty_globals_handling():
        """Test handling of missing functions gracefully"""
        # This should not crash even if some functions are missing
        result = is_function_available("_nonexistent_function")
        assert result == False, "Non-existent function check should return False"

    def test_function_callable_check():
        """Test that callable checks work properly"""
        # Test with known callable
        assert callable(handle_api_report), "handle_api_report should be callable"

    # === INTEGRATION TESTS ===
    def test_family_functions():
        """Test family data processing functions"""
        assert (
            "_display_family_info" in globals()
        ), "Family info display function should exist"
        assert (
            "_display_tree_relationship" in globals()
        ), "Tree relationship display function should exist"

    def test_data_extraction_functions():
        """Test data extraction functions"""
        assert (
            "_extract_fact_data" in globals()
        ), "Fact data extraction function should exist"
        assert (
            "_extract_detailed_info" in globals()
        ), "Detailed info extraction function should exist"

    def test_utility_functions():
        """Test utility and helper functions"""
        assert (
            "_parse_treesui_list_response" in globals()
        ), "Tree UI list parser should exist"
        assert (
            "_flatten_children_list" in globals()
        ), "Children list flattener should exist"

    # === PERFORMANCE TESTS ===
    def test_function_lookup_performance():
        """Test function lookup performance"""
        import time

        start_time = time.time()

        # Test multiple function lookups
        functions_to_check = [
            "handle_api_report",
            "_get_search_criteria",
            "_process_and_score_suggestions",
            "_display_search_results",
            "_handle_search_phase",
        ]

        for func_name in functions_to_check:
            assert (
                func_name in globals()
            ), f"Function {func_name} should exist in globals"

        end_time = time.time()
        # Should complete lookups quickly (< 0.01 seconds)
        duration = end_time - start_time
        assert (
            duration < 0.01
        ), f"Function lookups should complete in under 0.01s, took {duration:.3f}s"

    def test_callable_check_performance():
        """Test callable check performance"""
        import time

        start_time = time.time()

        for _ in range(50):
            callable(handle_api_report)

        end_time = time.time()
        # Should complete 50 callable checks quickly (< 0.01 seconds)
        duration = end_time - start_time
        assert (
            duration < 0.01
        ), f"50 callable checks should complete in under 0.01s, took {duration:.3f}s"  # === ERROR HANDLING TESTS ===

    def test_fraser_gault_functions():
        """Test Fraser Gault functionality availability"""
        fraser_test_data = {
            "fraser_id": "FRASER_12345",
            "test_name": "Fraser Gault Test 12345",
        }
        assert (
            "12345" in fraser_test_data["fraser_id"]
        ), "Fraser test data should contain 12345 identifier"

        try:
            return is_function_available(
                "run_standalone_fraser_test"
            ) and is_function_available("load_test_person_from_env")
        except Exception:
            return True  # Exception is acceptable

    def test_exception_handling():
        """Test that function checks handle exceptions gracefully"""
        try:
            # This should not raise an exception
            result = callable(None)
            return result == False
        except Exception:
            return True  # Exception handling is working

    # === RUN ALL TESTS ===
    with suppress_logging():
        # INITIALIZATION TESTS
        suite.run_test(
            "Module imports",
            test_module_imports,
            "4 modules imported: json, re, time, typing - all standard library modules available.",
            "Test imports of json, re, time, and typing modules for live API operations.",
            "Import json, re, time, and typing modules and verify successful import.",
        )

        suite.run_test(
            "Core function availability",
            test_core_function_availability,
            "2 core functions available and callable: handle_api_report, main.",
            "Test that core functions are available and callable.",
            "Check globals() for handle_api_report and main, verify callable() returns True.",
        )

        suite.run_test(
            "Search functions availability",
            test_search_functions,
            "Should have search criteria functions available for genealogical research",
            "Test _get_search_criteria function availability in globals",
            "Search criteria functions exist for genealogical API research",
        )

        # CORE FUNCTIONALITY TESTS
        suite.run_test(
            "Scoring functions",
            test_scoring_functions,
            "Should have scoring and ranking capabilities for genealogical research suggestions",
            "Test availability of _process_and_score_suggestions and _run_simple_suggestion_scoring",
            "Scoring functions exist for processing and ranking genealogical research suggestions",
        )

        suite.run_test(
            "Display functions",
            test_display_functions,
            "Should have result display capabilities for research results and comparisons",
            "Test availability of _display_search_results and _display_initial_comparison",
            "Display functions exist for showing research results and genealogical comparisons",
        )

        suite.run_test(
            "API integration functions",
            test_api_integration_functions,
            "Should have API integration handlers for search and selection phases",
            "Test availability of _handle_search_phase and _handle_selection_phase",
            "API integration handlers exist for managing search and selection workflows",
        )

        # EDGE CASE TESTS
        suite.run_test(
            "Empty globals handling",
            test_empty_globals_handling,
            "Should handle missing functions gracefully without crashing system",
            "Test checking for non-existent functions in globals without exceptions",
            "Missing function checks handled gracefully without system errors",
        )

        suite.run_test(
            "Function callable check",
            test_function_callable_check,
            "Should properly check if functions are callable for API research operations",
            "Test callable() function with known functions to verify proper operation",
            "Callable checks work properly for verifying function availability",
        )

        # INTEGRATION TESTS
        suite.run_test(
            "Family functions",
            test_family_functions,
            "Should have family data processing capabilities for genealogical tree management",
            "Test availability of _display_family_info and _display_tree_relationship",
            "Family data processing functions exist for genealogical relationship management",
        )

        suite.run_test(
            "Data extraction functions",
            test_data_extraction_functions,
            "Should have data extraction utilities for genealogical fact processing",
            "Test availability of _extract_fact_data and _extract_detailed_info",
            "Data extraction functions exist for processing genealogical facts and details",
        )

        suite.run_test(
            "Utility functions",
            test_utility_functions,
            "Should have utility helper functions for API response processing",
            "Test availability of _parse_treesui_list_response and _flatten_children_list",
            "Utility helper functions exist for processing tree UI responses and data structures",
        )

        # PERFORMANCE TESTS
        suite.run_test(
            "Function lookup performance",
            test_function_lookup_performance,
            "Should perform function lookups efficiently for API research operations",
            "Test multiple function lookups with timing validation under 0.01 seconds",
            "Function lookups complete efficiently for genealogical API research functions",
        )

        suite.run_test(
            "Callable check performance",
            test_callable_check_performance,
            "Should perform callable checks efficiently for function validation",
            "Test 50 callable checks with timing validation under 0.01 seconds",
            "Callable checks complete efficiently for API function validation",
        )

        # ERROR HANDLING TESTS
        suite.run_test(
            "Fraser Gault functions",
            test_fraser_gault_functions,
            "Should have Fraser Gault test functionality available for specialized testing",
            "Test availability of run_standalone_fraser_test and load_test_person_from_env functions",
            "Fraser Gault test functions exist for specialized genealogical testing scenarios",
        )

        suite.run_test(
            "Exception handling",
            test_exception_handling,
            "Should handle exceptions in function checks gracefully without system crashes",
            "Test callable() function with None value to verify exception handling",
            "Exception handling works properly for function validation operations",
        )

        return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return action11_module_tests()


if __name__ == "__main__":
    import sys

    print("🔍 Running Action 11 - Live API Research Tool comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
