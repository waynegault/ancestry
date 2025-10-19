#!/usr/bin/env python3

"""
Action 11: Ancestry API Search and Family Analysis

Searches Ancestry API for individuals, displays detailed person information,
analyzes family relationships, and generates comprehensive genealogical reports
with relationship path calculations and family tree visualization.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from core.error_handling import (
    circuit_breaker,
    error_context,
    graceful_degradation,
    retry_on_failure,
    timeout_protection,
)

# === STANDARD LIBRARY IMPORTS ===
import json
import os
import re  # Added for robust lifespan splitting
import sys
from datetime import datetime
from typing import Any, Callable, Optional, Union
from urllib.parse import quote, urlencode

# === THIRD-PARTY IMPORTS ===
import requests
from bs4 import BeautifulSoup
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
                raise TypeError("config_schema.api.base_url missing.")
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
        raise TypeError("config_schema.selenium.api_timeout missing.")

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
    not hasattr(config_schema, "common_scoring_weights")
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
    from gedcom_utils import _clean_display_date, _parse_date, calculate_match_score

    logger.debug("Successfully imported functions from gedcom_utils.")
    GEDCOM_SCORING_AVAILABLE = calculate_match_score is not None and callable(  # type: ignore
        calculate_match_score
    )
    GEDCOM_UTILS_AVAILABLE = all(  # type: ignore
        f is not None for f in [_parse_date, _clean_display_date]
    )
    if not GEDCOM_UTILS_AVAILABLE:
        logger.warning("One or more date utils from gedcom_utils are None.")
except ImportError as e:
    logger.error(f"Failed to import from gedcom_utils: {e}.", exc_info=True)

# --- Import API Utilities ---
# Import specific API call helpers, parsers, AND the timeout helper
from api_utils import (
    call_discovery_relationship_api,
    call_facts_user_api,
    call_getladder_api,
)
from common_params import ApiIdentifiers, RelationshipCalcContext

# Import relationship utilities
from relationship_utils import (
    convert_api_path_to_unified_format,
    convert_discovery_api_path_to_unified_format,
    format_api_relationship_path,
    format_relationship_path_unified,
)

# Import universal scoring utilities
from universal_scoring import calculate_display_bonuses

logger.debug(
    "Successfully imported required functions from api_utils and relationship_utils."
)
API_UTILS_AVAILABLE = True


# --- Import General Utilities ---
from core.session_manager import SessionManager
from utils import format_name

logger.debug("Successfully imported required components from utils.")
CORE_UTILS_AVAILABLE = True

# --- Session Manager Instance ---
session_manager = SessionManager()
if not session_manager:
    logger.critical("SessionManager instance not created. Cannot proceed.")
    print("FATAL ERROR: SessionManager not available.")
    sys.exit(1)

# _extract_fact_data removed - unused 77-line function


# === SEARCH CRITERIA HELPER FUNCTIONS ===

def _parse_year_from_string(date_str: str, parse_date_func: Optional[Callable]) -> tuple[Optional[int], Optional[datetime]]:
    """Parse year from date string, trying full date parse first, then year extraction."""
    date_obj: Optional[datetime] = None
    year: Optional[int] = None

    # Try full date parsing
    if parse_date_func:
        try:
            date_obj = parse_date_func(date_str)
            if date_obj:
                year = date_obj.year
                logger.debug(f"Successfully parsed date: {date_obj}, year: {year}")
                return year, date_obj
        except Exception as e:
            logger.debug(f"Could not parse date with parse_date_func: {e}")

    # Fallback: extract year with regex
    logger.warning(f"Could not parse input year/date: '{date_str}'")
    year_match = re.search(r"\b(\d{4})\b", date_str)
    if year_match:
        try:
            year = int(year_match.group(1))
            logger.debug(f"Extracted year {year} from '{date_str}' as fallback.")
        except ValueError:
            logger.warning(f"Could not convert extracted year '{year_match.group(1)}' to int.")

    return year, date_obj


def _get_user_input() -> dict[str, str]:
    """Get search criteria input from user."""
    print("\n--- Enter Search Criteria (Press Enter to skip optional fields) ---\n")
    return {
        "first_name": input("  First Name Contains: ").strip(),
        "surname": input("  Last Name Contains: ").strip(),
        "gender_input": input("  Gender (M/F): ").strip().upper(),
        "dob_str": input("  Birth Year (YYYY): ").strip(),
        "pob": input("  Birth Place Contains: ").strip(),
        "dod_str": input("  Death Year (YYYY): ").strip() or None,
        "pod": input("  Death Place Contains: ").strip() or None,
    }


def _validate_required_fields(first_name: str, surname: str) -> bool:
    """Validate that at least first name or surname is provided."""
    if not (first_name or surname):
        logger.warning("API search needs First Name or Surname. Report cancelled.")
        print("\nAPI search needs First Name or Surname. Report cancelled.")
        return False
    return True


def _get_search_criteria() -> Optional[dict[str, Any]]:
    """Gets search criteria from the user via input prompts."""

    # Get user input
    user_input = _get_user_input()
    print("")

    # Validate required fields
    if not _validate_required_fields(user_input["first_name"], user_input["surname"]):
        return None

    # Parse gender
    gender = None
    if user_input["gender_input"] and user_input["gender_input"][0] in ["M", "F"]:
        gender = user_input["gender_input"][0].lower()

    # Parse dates
    def clean_param(p: Any) -> Optional[str]:
        return (p.strip().lower() if p and isinstance(p, str) else None)
    parse_date_func = _parse_date if callable(_parse_date) else None

    target_birth_year, target_birth_date_obj = (None, None)
    if user_input["dob_str"]:
        target_birth_year, target_birth_date_obj = _parse_year_from_string(user_input["dob_str"], parse_date_func)

    target_death_year, target_death_date_obj = (None, None)
    if user_input["dod_str"]:
        target_death_year, target_death_date_obj = _parse_year_from_string(user_input["dod_str"], parse_date_func)

    # Build search criteria dictionary
    search_criteria_dict = {
        "first_name_raw": user_input["first_name"],
        "surname_raw": user_input["surname"],
        "first_name": clean_param(user_input["first_name"]),
        "surname": clean_param(user_input["surname"]),
        "birth_year": target_birth_year,
        "birth_date_obj": target_birth_date_obj,
        "birth_place": clean_param(user_input["pob"]),
        "death_year": target_death_year,
        "death_date_obj": target_death_date_obj,
        "death_place": clean_param(user_input["pod"]),
        "gender": gender,
    }

    # Log search criteria
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
        log_value = "None" if value is None else (f"'{value}'" if isinstance(value, str) else str(value))
        logger.debug(f"  {display_name}: {log_value}")

    return search_criteria_dict


# End of _get_search_criteria


# === SIMPLE SCORING HELPER FUNCTIONS ===

def _get_scoring_weights() -> dict[str, int]:
    """Get scoring weights from config schema."""
    weights = getattr(config_schema, 'common_scoring_weights', {})
    return {
        "birth_year_match": weights.get("birth_year_match", 20) if weights else 20,
        "birth_year_close": weights.get("birth_year_close", 10) if weights else 10,
        "birth_place_match": weights.get("birth_place_match", 20) if weights else 20,
        "death_year_match": weights.get("death_year_match", 20) if weights else 20,
        "death_year_close": weights.get("death_year_close", 10) if weights else 10,
        "death_place_match": weights.get("death_place_match", 20) if weights else 20,
        "death_dates_absent": weights.get("death_dates_both_absent", 15) if weights else 15,
        "death_bonus": weights.get("bonus_death_date_and_place", 15) if weights else 15,
        "gender_match": weights.get("gender_match", 15) if weights else 15,
    }


def _score_name_match(search_fn: str, search_sn: str, cand_fn: str, cand_sn: str) -> tuple[int, dict[str, int], list[str]]:
    """Score name matching (first name, surname, and bonus)."""
    score = 0
    field_scores = {"givn": 0, "surn": 0, "bonus": 0}
    reasons = []

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

    return score, field_scores, reasons


def _score_birth_info(search_by: int, search_bp: str, cand_by: int, cand_bp: str, weights: dict[str, int]) -> tuple[int, dict[str, int], list[str]]:
    """Score birth year and place matching."""
    score = 0
    field_scores = {"byear": 0, "bplace": 0, "bbonus": 0}
    reasons = []

    # Birth year scoring
    if cand_by and search_by:
        try:
            cand_by_int = int(cand_by)
            search_by_int = int(search_by)
            if cand_by_int == search_by_int:
                score += weights["birth_year_match"]
                field_scores["byear"] = weights["birth_year_match"]
                reasons.append(f"exact birth year ({cand_by}) ({weights['birth_year_match']}pts)")
            elif abs(cand_by_int - search_by_int) <= 5:
                score += weights["birth_year_close"]
                field_scores["byear"] = weights["birth_year_close"]
                reasons.append(f"close birth year ({cand_by} vs {search_by}) ({weights['birth_year_close']}pts)")
        except (ValueError, TypeError):
            pass

    # Birth place scoring
    if cand_bp and search_bp and search_bp in cand_bp:
        score += weights["birth_place_match"]
        field_scores["bplace"] = weights["birth_place_match"]
        reasons.append(f"birth place contains ({search_bp}) ({weights['birth_place_match']}pts)")

    # Birth bonus
    if field_scores["byear"] > 0 and field_scores["bplace"] > 0:
        score += 25
        field_scores["bbonus"] = 25
        reasons.append("bonus birth info (25pts)")

    return score, field_scores, reasons


# Helper functions for _score_death_info

def _score_death_year(search_dy: int, cand_dy: int, weights: dict[str, int]) -> tuple[int, int, list[str]]:
    """Score death year matching."""
    if not cand_dy or not search_dy:
        return 0, 0, []

    try:
        cand_dy_int = int(cand_dy)
        search_dy_int = int(search_dy)

        if cand_dy_int == search_dy_int:
            return (
                weights["death_year_match"],
                weights["death_year_match"],
                [f"exact death year ({cand_dy}) ({weights['death_year_match']}pts)"]
            )
        if abs(cand_dy_int - search_dy_int) <= 5:
            return (
                weights["death_year_close"],
                weights["death_year_close"],
                [f"close death year ({cand_dy} vs {search_dy}) ({weights['death_year_close']}pts)"]
            )
    except (ValueError, TypeError):
        pass

    return 0, 0, []


def _score_death_date_absent(search_dy: int, cand_dy: int, is_living: bool, weights: dict[str, int]) -> tuple[int, int, list[str]]:
    """Score when both death dates are absent."""
    if not search_dy and not cand_dy and is_living in [False, None]:
        return (
            weights["death_dates_absent"],
            weights["death_dates_absent"],
            [f"death date absent ({weights['death_dates_absent']}pts)"]
        )
    return 0, 0, []


def _score_death_place(search_dp: str, cand_dp: str, weights: dict[str, int]) -> tuple[int, int, list[str]]:
    """Score death place matching."""
    if cand_dp and search_dp and search_dp in cand_dp:
        return (
            weights["death_place_match"],
            weights["death_place_match"],
            [f"death place contains ({search_dp}) ({weights['death_place_match']}pts)"]
        )
    return 0, 0, []


def _score_death_bonus(dyear_score: int, ddate_score: int, dplace_score: int, weights: dict[str, int]) -> tuple[int, int, list[str]]:
    """Score death bonus when both date and place are present."""
    if (dyear_score > 0 or ddate_score > 0) and dplace_score > 0:
        return (
            weights["death_bonus"],
            weights["death_bonus"],
            [f"bonus death info ({weights['death_bonus']}pts)"]
        )
    return 0, 0, []


def _score_death_info(search_dy: int, search_dp: str, cand_dy: int, cand_dp: str, is_living: bool, weights: dict[str, int]) -> tuple[int, dict[str, int], list[str]]:
    """Score death year and place matching."""
    score = 0
    field_scores = {"dyear": 0, "ddate": 0, "dplace": 0, "dbonus": 0}
    reasons = []

    # Score death year
    year_score, year_field_score, year_reasons = _score_death_year(search_dy, cand_dy, weights)
    score += year_score
    field_scores["dyear"] = year_field_score
    reasons.extend(year_reasons)

    # Score death date absent (if year scoring didn't apply)
    if year_score == 0:
        date_score, date_field_score, date_reasons = _score_death_date_absent(search_dy, cand_dy, is_living, weights)
        score += date_score
        field_scores["ddate"] = date_field_score
        reasons.extend(date_reasons)

    # Score death place
    place_score, place_field_score, place_reasons = _score_death_place(search_dp, cand_dp, weights)
    score += place_score
    field_scores["dplace"] = place_field_score
    reasons.extend(place_reasons)

    # Score death bonus
    bonus_score, bonus_field_score, bonus_reasons = _score_death_bonus(
        field_scores["dyear"], field_scores["ddate"], field_scores["dplace"], weights
    )
    score += bonus_score
    field_scores["dbonus"] = bonus_field_score
    reasons.extend(bonus_reasons)

    return score, field_scores, reasons


def _score_gender_match(search_gn: str, cand_gn: str, weights: dict[str, int]) -> tuple[int, int, list[str]]:
    """Score gender matching."""
    score = 0
    gender_score = 0
    reasons = []

    logger.debug(f"[Simple Scoring] Checking Gender: Search='{search_gn}', Candidate='{cand_gn}'")
    if cand_gn is not None and search_gn is not None and cand_gn == search_gn:
        score = weights["gender_match"]
        gender_score = weights["gender_match"]
        reasons.append(f"Gender Match ({cand_gn.upper()}) ({weights['gender_match']}pts)")
        logger.debug(f"[Simple Scoring] Gender Match! Adding {weights['gender_match']} points.")

    return score, gender_score, reasons


def _extract_candidate_fields(candidate_data_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract candidate fields from candidate data dictionary."""
    return {
        "first_name": candidate_data_dict.get("first_name") or "",
        "surname": candidate_data_dict.get("surname") or "",
        "birth_year": candidate_data_dict.get("birth_year") or 0,
        "birth_place": candidate_data_dict.get("birth_place") or "",
        "death_year": candidate_data_dict.get("death_year") or 0,
        "death_place": candidate_data_dict.get("death_place") or "",
        "gender": candidate_data_dict.get("gender") or "",
        "is_living": candidate_data_dict.get("is_living") or False,
    }


def _extract_search_fields(search_criteria: dict[str, Any]) -> dict[str, Any]:
    """Extract search fields from search criteria dictionary."""
    return {
        "first_name": search_criteria.get("first_name") or "",
        "surname": search_criteria.get("surname") or "",
        "birth_year": search_criteria.get("birth_year") or 0,
        "birth_place": search_criteria.get("birth_place") or "",
        "death_year": search_criteria.get("death_year") or 0,
        "death_place": search_criteria.get("death_place") or "",
        "gender": search_criteria.get("gender") or "",
    }


def _log_gender_mismatch(cand_gn: Any, search_gn: Any) -> None:
    """Log gender mismatch information."""
    if cand_gn is not None and search_gn is not None and cand_gn != search_gn:
        logger.debug("[Simple Scoring] Gender MISMATCH. No points awarded.")
    elif search_gn is None:
        logger.debug("[Simple Scoring] No search gender provided. No points awarded.")
    elif cand_gn is None:
        logger.debug("[Simple Scoring] Candidate gender not available. No points awarded.")


# Simple scoring fallback (Uses 'gender_match' key)
def _run_simple_suggestion_scoring(
    search_criteria: dict[str, Any], candidate_data_dict: dict[str, Any]
) -> tuple[float, dict[str, Any], list[str]]:
    """Performs simple fallback scoring based on hardcoded rules. Uses 'gender_match' key."""
    logger.warning("Using simple fallback scoring for suggestion.")

    # Initialize scoring
    total_score = 0.0
    field_scores = {
        "givn": 0, "surn": 0, "gender_match": 0,
        "byear": 0, "bdate": 0, "bplace": 0,
        "dyear": 0, "ddate": 0, "dplace": 0,
        "bonus": 0, "bbonus": 0, "dbonus": 0,
    }
    reasons = ["API Suggest Match", "Fallback Scoring"]

    # Get scoring weights
    weights = _get_scoring_weights()

    # Extract candidate and search data
    cand = _extract_candidate_fields(candidate_data_dict)
    search = _extract_search_fields(search_criteria)

    # Score name matching
    name_score, name_fields, name_reasons = _score_name_match(
        search["first_name"], search["surname"], cand["first_name"], cand["surname"]
    )
    total_score += name_score
    field_scores.update(name_fields)
    reasons.extend(name_reasons)

    # Score birth information
    birth_score, birth_fields, birth_reasons = _score_birth_info(
        search["birth_year"], search["birth_place"], cand["birth_year"], cand["birth_place"], weights
    )
    total_score += birth_score
    field_scores.update(birth_fields)
    reasons.extend(birth_reasons)

    # Score death information
    death_score, death_fields, death_reasons = _score_death_info(
        search["death_year"], search["death_place"], cand["death_year"], cand["death_place"], cand["is_living"], weights
    )
    total_score += death_score
    field_scores.update(death_fields)
    reasons.extend(death_reasons)

    # Score gender matching
    gender_score, gender_field_score, gender_reasons = _score_gender_match(search["gender"], cand["gender"], weights)
    total_score += gender_score
    field_scores["gender_match"] = gender_field_score
    reasons.extend(gender_reasons)

    # Handle gender mismatch logging
    _log_gender_mismatch(cand["gender"], search["gender"])

    return total_score, field_scores, reasons


# End of _run_simple_suggestion_scoring


# === SUGGESTION PROCESSING HELPER FUNCTIONS ===

def _extract_candidate_data(raw_candidate: dict, idx: int, clean_param: Callable, parse_date_func: Optional[Callable]) -> dict[str, Any]:
    """Extract and normalize candidate data from raw API response."""
    full_name_disp = raw_candidate.get("FullName", "Unknown")
    person_id = raw_candidate.get("PersonId", f"Unknown_{idx}")

    # Parse name
    first_name_cand = None
    surname_cand = None
    if full_name_disp != "Unknown":
        parts = full_name_disp.split()
        if parts:
            first_name_cand = clean_param(parts[0])
        if len(parts) > 1:
            surname_cand = clean_param(parts[-1])

    # Parse dates
    birth_date_obj_cand = None
    birth_date_str_cand = raw_candidate.get("BirthDate")
    if birth_date_str_cand and parse_date_func:
        try:
            birth_date_obj_cand = parse_date_func(birth_date_str_cand)
        except ValueError:
            logger.debug(f"Could not parse candidate birth date: {birth_date_str_cand}")

    death_date_obj_cand = None
    death_date_str_cand = raw_candidate.get("DeathDate")
    if death_date_str_cand and parse_date_func:
        try:
            death_date_obj_cand = parse_date_func(death_date_str_cand)
        except ValueError:
            logger.debug(f"Could not parse candidate death date: {death_date_str_cand}")

    # Build candidate data dictionary
    birth_place_cand = clean_param(raw_candidate.get("BirthPlace"))
    death_place_cand = clean_param(raw_candidate.get("DeathPlace"))
    gender_cand = raw_candidate.get("Gender")

    return {
        "norm_id": person_id,
        "display_id": person_id,
        "first_name": first_name_cand,
        "surname": surname_cand,
        "full_name_disp": full_name_disp,
        "gender_norm": gender_cand,
        "birth_year": raw_candidate.get("BirthYear"),
        "birth_date_obj": birth_date_obj_cand,
        "birth_place_disp": birth_place_cand,
        "death_year": raw_candidate.get("DeathYear"),
        "death_date_obj": death_date_obj_cand,
        "death_place_disp": death_place_cand,
        "is_living": raw_candidate.get("IsLiving"),
        "gender": gender_cand,
        "birth_place": birth_place_cand,
        "death_place": death_place_cand,
    }


def _calculate_candidate_score(
    candidate_data_dict: dict[str, Any],
    search_criteria: dict[str, Any],
    scoring_func: Optional[Callable],
    scoring_weights: dict[str, int],
    name_flex: int,
    date_flex: dict[str, int],
    gender_weight: int,
) -> tuple[float, dict[str, Any], list[str]]:
    """Calculate match score for a candidate using available scoring function."""
    person_id = candidate_data_dict.get("norm_id", "Unknown")

    # Log inputs
    logger.debug(f"--- Scoring Candidate ID: {person_id} ---")
    logger.debug(f"Search Criteria Gender: '{search_criteria.get('gender')}'")
    logger.debug(f"Candidate dict Gender ('gender'): '{candidate_data_dict.get('gender')}'")
    logger.debug(f"Calling scoring function: {getattr(scoring_func, '__name__', 'Unknown')}")

    try:
        # Try GEDCOM scoring first
        if GEDCOM_SCORING_AVAILABLE and scoring_func == calculate_match_score and scoring_func is not None:
            if gender_weight == 0:
                logger.warning("Gender weight ('gender_match') in config is 0.")

            # Debug logging for Fraser Gault
            if "fraser" in candidate_data_dict.get("first_name", "").lower():
                logger.info("=== ACTION 11 FRASER GAULT SCORING DEBUG ===")
                logger.info(f"Search criteria: {search_criteria}")
                logger.info(f"Candidate data: {candidate_data_dict}")
                logger.info(f"Scoring weights: {scoring_weights}")
                logger.info(f"Date flexibility: {date_flex}")
                logger.info(f"Name flexibility: {name_flex}")

            score, field_scores, reasons = scoring_func(
                search_criteria,
                candidate_data_dict,
                scoring_weights,
                _name_flexibility=name_flex if isinstance(name_flex, dict) else None,
                date_flexibility=date_flex if isinstance(date_flex, dict) else None,
            )

            # Debug logging for Fraser Gault results
            if "fraser" in candidate_data_dict.get("first_name", "").lower():
                logger.info(f"Score result: {score}")
                logger.info(f"Field scores: {field_scores}")
                logger.info(f"Reasons: {reasons}")
                logger.info("=== END ACTION 11 FRASER GAULT DEBUG ===")

            logger.debug(f"Gedcom Score for {person_id}: {score}, Fields: {field_scores}")
            if "gender_match" in field_scores:
                logger.debug(f"Gedcom Field Score ('gender_match'): {field_scores['gender_match']}")
            else:
                logger.debug("Gedcom Field Scores missing 'gender_match' key.")

            return score, field_scores, reasons

        # Fall back to simple scoring
        if scoring_func is not None:
            score, field_scores, reasons = _run_simple_suggestion_scoring(search_criteria, candidate_data_dict)
            logger.debug(f"Simple Score for {person_id}: {score}, Fields: {field_scores}")
            if "gender_match" in field_scores:
                logger.debug(f"Simple Field Score ('gender_match'): {field_scores['gender_match']}")
            else:
                logger.debug("Simple Field Scores missing 'gender_match' key.")

            return score, field_scores, reasons

        logger.error("Scoring function is None")
        return 0.0, {}, []

    except Exception as score_err:
        logger.error(f"Error scoring {person_id}: {score_err}", exc_info=True)
        logger.warning("Falling back to simple scoring...")
        score, field_scores, reasons = _run_simple_suggestion_scoring(search_criteria, candidate_data_dict)
        reasons.append("(Error Fallback)")
        logger.debug(f"Fallback Score for {person_id}: {score}, Fields: {field_scores}")
        return score, field_scores, reasons


def _build_processed_candidate(
    raw_candidate: dict,
    candidate_data_dict: dict[str, Any],
    score: float,
    field_scores: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    """Build processed candidate dictionary with all required fields."""
    person_id = candidate_data_dict.get("norm_id", "Unknown")
    full_name_disp = candidate_data_dict.get("full_name_disp", "Unknown")
    birth_date_str = raw_candidate.get("BirthDate")
    death_date_str = raw_candidate.get("DeathDate")

    return {
        "id": person_id,
        "name": full_name_disp,
        "gender": candidate_data_dict.get("gender"),
        "birth_date": (
            _clean_display_date(birth_date_str)
            if callable(_clean_display_date)
            else (birth_date_str or "N/A")
        ),
        "birth_place": raw_candidate.get("BirthPlace", "N/A"),
        "death_date": (
            _clean_display_date(death_date_str)
            if callable(_clean_display_date)
            else (death_date_str or "N/A")
        ),
        "death_place": raw_candidate.get("DeathPlace", "N/A"),
        "score": score,
        "field_scores": field_scores,
        "reasons": reasons,
        "raw_data": raw_candidate,
        "parsed_suggestion": candidate_data_dict,
    }


# Suggestion processing and scoring (Uses 'gender_match' key)
def _process_and_score_suggestions(
    suggestions: list[dict], search_criteria: dict[str, Any]
) -> list[dict]:
    """
    Processes raw API suggestions, calculates match scores, and returns sorted list.
    Uses 'gender_match' key from config weights.
    """
    processed_candidates = []

    # Setup
    def clean_param(p: Any) -> Optional[str]:
        return (p.strip().lower() if p and isinstance(p, str) else None)
    parse_date_func = _parse_date if callable(_parse_date) else None
    scoring_func = calculate_match_score if GEDCOM_SCORING_AVAILABLE else None

    # Get configuration
    scoring_weights_raw = dict(config_schema.common_scoring_weights) if config_schema else {}
    scoring_weights = {k: int(v) for k, v in scoring_weights_raw.items()}
    name_flex = getattr(config_schema, "name_flexibility", 2)
    date_flexibility_value = getattr(config_schema, "date_flexibility", 2)
    date_flex = {"year_match_range": int(date_flexibility_value)}
    gender_weight = int(scoring_weights.get("gender_match", 0))

    # Process each suggestion
    for idx, raw_candidate in enumerate(suggestions):
        if not isinstance(raw_candidate, dict):
            logger.warning(f"Skipping invalid entry: {raw_candidate}")
            continue

        # Extract candidate data
        candidate_data_dict = _extract_candidate_data(raw_candidate, idx, clean_param, parse_date_func)

        # Calculate score
        score, field_scores, reasons = _calculate_candidate_score(
            candidate_data_dict,
            search_criteria,
            scoring_func,
            scoring_weights,
            name_flex,
            date_flex,
            gender_weight,
        )

        # Build processed candidate
        processed_candidate = _build_processed_candidate(
            raw_candidate,
            candidate_data_dict,
            score,
            field_scores,
            reasons,
        )
        processed_candidates.append(processed_candidate)

    # Sort by score (highest first)
    processed_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    return processed_candidates


# End of _process_and_score_suggestions


# Helper functions for _display_search_results

def _extract_field_scores_for_display(candidate: dict) -> dict[str, int]:
    """Extract all field scores from candidate."""
    fs = candidate.get("field_scores", {})
    return {
        "givn": fs.get("givn", 0),
        "surn": fs.get("surn", 0),
        "name_bonus": fs.get("bonus", 0),
        "gender": fs.get("gender_match", 0),
        "byear": fs.get("byear", 0),
        "bdate": fs.get("bdate", 0),
        "bplace": fs.get("bplace", 0),
        "dyear": fs.get("dyear", 0),
        "ddate": fs.get("ddate", 0),
        "dplace": fs.get("dplace", 0),
    }


def _fix_gender_score_from_reasons(candidate: dict, gender_score: int) -> int:
    """Fix gender score by extracting from reasons if score is 0."""
    if gender_score != 0:
        return gender_score

    for reason in candidate.get("reasons", []):
        if "Gender Match" in reason:
            match = re.search(r"Gender Match \([MF]\) \((\d+)pts\)", reason)
            if match:
                return int(match.group(1))

    return gender_score


def _calculate_display_bonuses(scores: dict[str, int]) -> dict[str, int]:
    """Calculate display bonus values for birth and death."""
    # Use universal function (no key prefix for action11)
    bonuses = calculate_display_bonuses(scores, key_prefix="")

    # Add name bonus (action11-specific)
    bonuses["name"] = scores.get("name_bonus", 0)
    bonuses["birth"] = bonuses.pop("birth_bonus")  # Rename for action11 compatibility
    bonuses["death"] = bonuses.pop("death_bonus")  # Rename for action11 compatibility

    return bonuses


def _format_name_display_with_score(candidate: dict, scores: dict[str, int], bonuses: dict[str, int]) -> str:
    """Format name display with scores."""
    name_disp = candidate.get("name", "N/A")
    name_disp_short = name_disp[:30] + ("..." if len(name_disp) > 30 else "")

    name_base_score = scores["givn"] + scores["surn"]
    name_score_str = f"[{name_base_score}]"
    if bonuses["name"] > 0:
        name_score_str += f"[+{bonuses['name']}]"

    return f"{name_disp_short} {name_score_str}"


def _format_gender_display_with_score(candidate: dict, gender_score: int) -> str:
    """Format gender display with score."""
    gender_disp_val = candidate.get("gender", "N/A")
    gender_disp_str = str(gender_disp_val).upper() if gender_disp_val is not None else "N/A"
    return f"{gender_disp_str} [{gender_score}]"


def _format_birth_displays_with_scores(candidate: dict, scores: dict[str, int], bonuses: dict[str, int]) -> tuple[str, str]:
    """Format birth date and place displays with scores."""
    # Birth date
    bdate_disp = str(candidate.get("birth_date", "N/A"))
    birth_score_display = f"[{bonuses['birth_date_component']}]"
    bdate_with_score = f"{bdate_disp} {birth_score_display}"

    # Birth place
    bplace_disp_val = candidate.get("birth_place", "N/A")
    bplace_disp_str = str(bplace_disp_val) if bplace_disp_val is not None else "N/A"
    bplace_disp_short = bplace_disp_str[:20] + ("..." if len(bplace_disp_str) > 20 else "")
    bplace_with_score = f"{bplace_disp_short} [{scores['bplace']}]"
    if bonuses["birth"] > 0:
        bplace_with_score += f" [+{bonuses['birth']}]"

    return bdate_with_score, bplace_with_score


def _format_death_displays_with_scores(candidate: dict, scores: dict[str, int], bonuses: dict[str, int]) -> tuple[str, str]:
    """Format death date and place displays with scores."""
    # Death date
    ddate_disp = str(candidate.get("death_date", "N/A"))
    death_score_display = f"[{bonuses['death_date_component']}]"
    ddate_with_score = f"{ddate_disp} {death_score_display}"

    # Death place
    dplace_disp_val = candidate.get("death_place", "N/A")
    dplace_disp_str = str(dplace_disp_val) if dplace_disp_val is not None else "N/A"
    dplace_disp_short = dplace_disp_str[:20] + ("..." if len(dplace_disp_str) > 20 else "")
    dplace_with_score = f"{dplace_disp_short} [{scores['dplace']}]"
    if bonuses["death"] > 0:
        dplace_with_score += f" [+{bonuses['death']}]"

    return ddate_with_score, dplace_with_score


def _create_table_row_for_candidate(candidate: dict) -> list[str]:
    """Create a table row for a single candidate."""
    # Extract scores
    scores = _extract_field_scores_for_display(candidate)

    # Fix gender score from reasons if needed
    gender_score = _fix_gender_score_from_reasons(candidate, scores["gender"])

    # Calculate bonuses
    bonuses = _calculate_display_bonuses(scores)

    # Format displays
    name_with_score = _format_name_display_with_score(candidate, scores, bonuses)
    gender_with_score = _format_gender_display_with_score(candidate, gender_score)
    bdate_with_score, bplace_with_score = _format_birth_displays_with_scores(candidate, scores, bonuses)
    ddate_with_score, dplace_with_score = _format_death_displays_with_scores(candidate, scores, bonuses)

    total_display_score = int(candidate.get("score", 0))

    return [
        str(candidate.get("id", "N/A")),
        name_with_score,
        gender_with_score,
        bdate_with_score,
        bplace_with_score,
        ddate_with_score,
        dplace_with_score,
        str(total_display_score),
    ]


def _print_results_table(table_data: list[list[str]], headers: list[str]) -> None:
    """Print the results table using tabulate or fallback."""
    try:
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    except Exception as tab_err:
        logger.error(f"Error formatting table: {tab_err}")
        print("\nSearch Results (Fallback):")
        print(" | ".join(headers))
        print("-" * 80)
        for row in table_data:
            print(" | ".join(map(str, row)))
    print("")


# Display search results (Uses 'gender_match' key)
def _display_search_results(candidates: list[dict], max_to_display: int):
    """Displays the scored search results. Uses 'gender_match' score key."""
    if not candidates:
        print("\nNo candidates to display.")
        return

    display_count = min(len(candidates), max_to_display)
    print(f"\n=== SEARCH RESULTS (Top {display_count} Matches) ===\n")

    headers = ["ID", "Name", "Gender", "Birth", "Birth Place", "Death", "Death Place", "Total"]
    table_data = [_create_table_row_for_candidate(candidate) for candidate in candidates[:display_count]]

    _print_results_table(table_data, headers)


# End of _display_search_results


# Select top candidate
def _select_top_candidate(
    scored_candidates: list[dict],
) -> Optional[tuple[dict, dict]]:
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


# _display_initial_comparison removed - unused 242-line display function

# Helper functions for _extract_best_name_from_details

def _try_person_full_name(person_research_data: dict) -> Optional[str]:
    """Try to extract name from PersonFullName field."""
    person_full_name = person_research_data.get("PersonFullName")
    if person_full_name and person_full_name != "Valued Relative":
        logger.debug(f"Using PersonFullName: '{person_full_name}'")
        return person_full_name
    return None


def _try_name_fact(person_research_data: dict) -> Optional[str]:
    """Try to extract name from PersonFacts Name fact."""
    person_facts_list = person_research_data.get("PersonFacts", [])
    if not isinstance(person_facts_list, list):
        return None

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
        name = name_fact.get("Value", "Unknown")
        logger.debug(f"Using Name Fact: '{name}'")
        return name

    return None


def _try_constructed_name(person_research_data: dict) -> Optional[str]:
    """Try to construct name from FirstName and LastName fields."""
    first_name_comp = person_research_data.get("FirstName", "")
    last_name_comp = person_research_data.get("LastName", "")

    if first_name_comp or last_name_comp:
        constructed_name = f"{first_name_comp} {last_name_comp}".strip()
        if constructed_name and len(constructed_name) > 1:
            logger.debug(f"Using Constructed Name: '{constructed_name}'")
            return constructed_name

    return None


def _try_fallback_suggestion_name(candidate_raw: dict) -> Optional[str]:
    """Try to use fallback suggestion name from candidate."""
    cand_name = candidate_raw.get("FullName")
    if cand_name and cand_name != "Unknown":
        logger.debug(f"Using Fallback Suggestion Name: '{cand_name}'")
        return cand_name
    return None


def _format_extracted_name(name: str) -> str:
    """Format extracted name using name formatter."""
    name_formatter = format_name if callable(format_name) else lambda x: str(x).title()

    if not name or name == "Valued Relative":
        return "Unknown"
    if callable(name_formatter):
        return name_formatter(name)
    return name


# Detailed Info Extraction (Only called if proceeding to supplementary info)
def _extract_best_name_from_details(
    person_research_data: dict, candidate_raw: dict
) -> str:
    """Extracts the best available name from detailed API response."""
    logger.debug(
        f"_extract_best_name (Detail): Input keys={list(person_research_data.keys())}"
    )

    # Try extraction methods in order of preference
    best_name = (
        _try_person_full_name(person_research_data)
        or _try_name_fact(person_research_data)
        or _try_constructed_name(person_research_data)
        or _try_fallback_suggestion_name(candidate_raw)
        or "Unknown"
    )

    # Format and return the extracted name
    return _format_extracted_name(best_name)


# End of _extract_best_name_from_details


# _extract_detailed_info removed - unused 135-line detail extraction function


# Detailed scoring (Uses 'gender_match' key via fallback scorer)
# _score_detailed_match removed - unused 100-line detailed scoring function


# Family/Relationship Display Functions

# _convert_api_family_to_display_format removed - unused 22-line family converter


# _extract_family_from_relationship_calculation removed - unused 37-line family extractor


# _extract_family_from_tree_ladder_response removed - unused 37-line Tree Ladder API parser


# _parse_family_from_relationship_text removed - unused 83-line relationship text parser


# _parse_family_from_ladder_text removed - unused 75-line ladder text parser
# _extract_years_from_lifespan removed - unused 18-line helper function
# _extract_family_from_person_facts removed - unused 70-line function


# _fetch_facts_glue_data removed - unused 50-line factsgluenodata API fetcher
# _fetch_html_facts_page_data removed - unused 88-line HTML facts page fetcher
# _extract_family_data_from_html removed - unused 63-line HTML family data extractor
# _extract_json_from_script_tags removed - unused 54-line JSON script tag extractor
# _parse_html_family_sections removed - unused 733-line HTML family section parser (MASSIVE!)


# Removed duplicate _fetch_family_data_alternative function (was actually HTML parsing code)
# Removed massive HTML parsing function block (855 lines total):
#   - _extract_family_data_from_html
#   - _extract_json_from_script_tags
#   - _parse_html_family_sections
#   - _merge_family_data
#   - _extract_microdata_family_info
#   - _extract_family_from_text_patterns
#   - _extract_family_from_navigation_data
#   - _extract_person_from_html_element
#   - _extract_year_from_text
#   - And many other HTML parsing helper functions


# _fetch_family_data_alternative removed - unused 121-line function


# _flatten_children_list removed - unused 44-line function


# _display_family_info removed - unused 67-line function


# _display_tree_relationship removed - unused 88-line function
# _display_discovery_relationship removed - unused 85-line function


# --- Phase Handler Functions ---

# Helper functions for _call_direct_treesui_list_api

def _validate_treesui_api_parameters(
    session_manager_local: SessionManager,
    owner_tree_id: str,
    base_url: str,
) -> bool:
    """Validate required parameters for TreesUI API call."""
    if not session_manager_local or not owner_tree_id or not base_url:
        logger.error("Missing required parameters for direct TreesUI List API call")
        return False
    return True


def _validate_requests_session(session_manager_local: SessionManager) -> bool:
    """Validate that requests session is available for API calls."""
    if (
        not hasattr(session_manager_local, "_requests_session")
        or not session_manager_local._requests_session
    ):
        logger.error("No requests session available for direct TreesUI List API call.")
        print("Error: API session not available. Cannot contact Ancestry API.")
        return False
    return True


def _build_treesui_api_url(
    base_url: str,
    owner_tree_id: str,
    search_criteria: dict[str, Any],
) -> str:
    """Build TreesUI List API URL with search parameters."""
    first_name = search_criteria.get("first_name_raw", "")
    surname = search_criteria.get("surname_raw", "")
    params = {"sort": "sname,gname", "limit": "100", "fields": "NAMES,EVENTS,GENDER"}
    if first_name:
        params["fn"] = first_name
    if surname:
        params["ln"] = surname
    encoded_params = urlencode(params, quote_via=quote)
    return f"{base_url}/api/treesui-list/trees/{owner_tree_id}/persons?{encoded_params}"


def _get_api_cookies(session_manager_local: SessionManager) -> Optional[dict]:
    """Get cookies for API call from session manager."""
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

    return cookies


def _build_treesui_api_headers(base_url: str, owner_tree_id: str) -> dict[str, str]:
    """Build headers for TreesUI API request."""
    return {
        "Accept": "application/json",
        "Accept-Language": "en-GB,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": f"{base_url}/family-tree/tree/{owner_tree_id}/family",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
    }


def _handle_treesui_api_response(response: requests.Response) -> Optional[list[dict]]:
    """Handle TreesUI API response and extract data."""
    logger.debug(f"API Response Status Code: {response.status_code}")

    if response.status_code == 200:
        try:
            data = response.json()
            if isinstance(data, list):
                return data
            logger.error(f"API call OK but response not JSON list. Type: {type(data)}")
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


# API Call (Correct definition order)
def _call_direct_treesui_list_api(
    session_manager_local: SessionManager,
    owner_tree_id: str,
    search_criteria: dict[str, Any],
    base_url: str,
) -> Optional[list[dict]]:
    """
    Directly calls the TreesUI List API with the specific format requested.
    """
    # Validate parameters
    if not _validate_treesui_api_parameters(session_manager_local, owner_tree_id, base_url):
        return None

    # Validate requests session
    if not _validate_requests_session(session_manager_local):
        return None

    # Build API URL
    api_url = _build_treesui_api_url(base_url, owner_tree_id, search_criteria)
    logger.debug(f"\nAPI URL Called: {api_url}\n")

    api_timeout = 10

    try:
        # Get cookies
        cookies = _get_api_cookies(session_manager_local)

        # Build headers
        headers = _build_treesui_api_headers(base_url, owner_tree_id)
        logger.debug(f"API Request Headers: {headers}")
        if cookies:
            logger.debug(f"API Request Cookies (keys): {list(cookies.keys())}")

        # Execute request
        response = requests.get(
            api_url, headers=headers, cookies=cookies, timeout=api_timeout
        )

        # Handle response
        return _handle_treesui_api_response(response)

    except requests.exceptions.Timeout:
        logger.error(f"API call timed out after {api_timeout}s")
        print("(Error: Timed out searching Ancestry API)")
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


# Helper functions for _handle_search_phase

def _get_tree_id_from_config() -> Optional[str]:
    """Get tree ID from configuration."""
    config_tree_id = getattr(config_schema.api, "tree_id", None)
    if config_tree_id:
        logger.info(f"Using tree ID from configuration: {config_tree_id}")
        return config_tree_id
    return None


def _get_tree_id_from_api(session_manager_local: SessionManager) -> Optional[str]:
    """Get tree ID from API using tree name."""
    tree_name = config_schema.api.tree_name
    if not tree_name:
        return None

    logger.info(f"Attempting to retrieve tree ID for tree name: {tree_name}")
    try:
        owner_tree_id = session_manager_local.get_my_tree_id()
        if owner_tree_id:
            logger.info(f"Successfully retrieved tree ID: {owner_tree_id}")
            return owner_tree_id
        logger.warning(f"Failed to retrieve tree ID for tree name: {tree_name}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving tree ID: {e}")
        return None


def _get_tree_id_from_user() -> Optional[str]:
    """Prompt user for tree ID."""
    print("\nTree ID is required for searching. Please enter a tree ID:")
    user_tree_id = input("Tree ID: ").strip()
    if user_tree_id:
        logger.info(f"Using user-provided tree ID: {user_tree_id}")
        return user_tree_id
    logger.error("Owner Tree ID missing and no input provided.")
    print("Error: Tree ID is required for searching. Operation cancelled.")
    return None


def _resolve_owner_tree_id(session_manager_local: SessionManager) -> Optional[str]:
    """Resolve owner tree ID from session manager, config, API, or user input."""
    # First check session manager
    owner_tree_id = getattr(session_manager_local, "my_tree_id", None)
    if owner_tree_id:
        return owner_tree_id

    # Try to get from config
    owner_tree_id = _get_tree_id_from_config()
    if owner_tree_id:
        session_manager_local.api_manager.my_tree_id = owner_tree_id
        return owner_tree_id

    # Try to get from API
    owner_tree_id = _get_tree_id_from_api(session_manager_local)
    if owner_tree_id:
        session_manager_local.api_manager.my_tree_id = owner_tree_id
        return owner_tree_id

    # Finally, prompt user
    owner_tree_id = _get_tree_id_from_user()
    if owner_tree_id:
        session_manager_local.api_manager.my_tree_id = owner_tree_id
        return owner_tree_id

    return None


def _validate_base_url(base_url: str) -> bool:
    """Validate that base URL is configured."""
    if not base_url:
        logger.error("ERROR: Ancestry URL not configured.. Base URL missing.")
        print("Error: Ancestry URL not configured. Operation cancelled.")
        return False
    return True


def _limit_search_results(parsed_results: list[dict]) -> list[dict]:
    """Limit search results based on configuration."""
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
    return parsed_results


# Search Phase (Includes Limit, Calls API func defined above)
def _handle_search_phase(
    session_manager_local: SessionManager,
    search_criteria: dict[str, Any],
) -> Optional[list[dict]]:
    """Handles the API search phase using the direct TreesUI List API."""
    base_url = getattr(config_schema.api, "base_url", "").rstrip("/")

    # Resolve owner tree ID
    owner_tree_id = _resolve_owner_tree_id(session_manager_local)
    if not owner_tree_id:
        return None

    # Validate base URL
    if not _validate_base_url(base_url):
        return None

    # Call API to get suggestions
    suggestions_raw = _call_direct_treesui_list_api(
        session_manager_local, owner_tree_id, search_criteria, base_url
    )
    if suggestions_raw is None:
        logger.error("API search failed.")
        return None
    if not suggestions_raw:
        logger.info("API Search returned no results.No potential matches found.")
        return []

    # Parse API response
    parsed_results = _parse_treesui_list_response(suggestions_raw)
    if parsed_results is None:
        logger.error("Failed to parse API response. Error processing data.")
        return None

    # Limit results based on config
    return _limit_search_results(parsed_results)


# End of _handle_search_phase

# Helper functions for _parse_treesui_list_response

def _extract_gid_parts(person_raw: dict, idx: int) -> Optional[tuple[str, str]]:
    """Extract person_id and tree_id from gid field."""
    gid_data = person_raw.get("gid", {}).get("v", "")

    if not isinstance(gid_data, str) or ":" not in gid_data:
        logger.warning(f"Item {idx}: Missing or invalid gid data: {gid_data}")
        return None

    parts = gid_data.split(":")
    if len(parts) < 3:
        logger.warning(f"Item {idx}: Invalid gid format: {gid_data}")
        return None

    person_id = parts[0]
    tree_id = parts[2]
    logger.debug(f"Item {idx}: IDs={person_id},{tree_id}")
    return person_id, tree_id


def _extract_name_parts(person_raw: dict, idx: int) -> tuple[str, str, str]:
    """Extract first name, surname, and full name from Names field."""
    first_name_part = ""
    surname_part = ""
    full_name = "Unknown"

    names_list = person_raw.get("Names", [])
    if not isinstance(names_list, list) or not names_list:
        logger.warning(f"Item {idx}: Names list issue: {names_list}")
        return first_name_part, surname_part, full_name

    name_obj = names_list[0]
    if not isinstance(name_obj, dict):
        logger.warning(f"Item {idx}: Name obj not dict: {name_obj}")
        return first_name_part, surname_part, full_name

    first_name_part = name_obj.get("g", "").strip()
    surname_part = name_obj.get("s", "").strip()

    if first_name_part and surname_part:
        full_name = f"{first_name_part} {surname_part}"
    elif first_name_part:
        full_name = first_name_part
    elif surname_part:
        full_name = surname_part

    logger.debug(f"Item {idx}: Name='{full_name}'")
    return first_name_part, surname_part, full_name


def _extract_gender(person_raw: dict, idx: int) -> Optional[str]:
    """Extract gender from Genders array or 'l' field."""
    # Try Genders array first
    genders_list = person_raw.get("Genders", [])
    if isinstance(genders_list, list) and genders_list:
        gender_obj = genders_list[0]
        if isinstance(gender_obj, dict) and "g" in gender_obj:
            gender = gender_obj.get("g", "").lower()
            logger.debug(f"Item {idx}: Gender from Genders array='{gender}'")
            return gender

    # Fallback to 'l' field
    gender_flag = person_raw.get("l")
    if isinstance(gender_flag, bool):
        gender = "f" if gender_flag else "m"
        logger.debug(f"Item {idx}: Gender from 'l' field='{gender}'")
        return gender

    logger.warning(f"Item {idx}: No gender information available")
    return None


def _calculate_place_detail_score(event: dict) -> int:
    """Calculate place detail score based on comma count."""
    place = event.get("p", "")
    if not place:
        return 0
    # Count commas as a simple measure of detail level
    comma_count = place.count(",")
    # Add 1 to avoid zero scores for places without commas
    return comma_count + 1


def _select_best_event(events: list[dict]) -> Optional[dict]:
    """Select best event from list based on alternate status and place detail."""
    if not events:
        return None

    # Prioritize non-alternate events
    non_alternate_events = [e for e in events if e.get("pa") is False]
    events_to_process = non_alternate_events if non_alternate_events else events

    # Sort by place detail (higher score first)
    events_to_process.sort(key=_calculate_place_detail_score, reverse=True)

    return events_to_process[0]


def _extract_year_from_date(date_str: Optional[str], idx: int, event_type: str) -> Optional[int]:
    """Extract year from date string."""
    if not date_str:
        return None

    year_match = re.search(r"\b(\d{4})\b", date_str)
    if year_match:
        try:
            return int(year_match.group(1))
        except ValueError:
            logger.warning(
                f"Item {idx}: Bad {event_type} year convert: '{year_match.group(1)}'"
            )
    return None


def _extract_birth_info(events_list: list[dict], idx: int) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """Extract birth year, date, and place from events list."""
    birth_events = [e for e in events_list if isinstance(e, dict) and e.get("t") == "Birth"]

    if not birth_events:
        return None, None, None

    best_birth = _select_best_event(birth_events)
    if not best_birth:
        return None, None, None

    # Extract date information
    norm_date = best_birth.get("nd")
    disp_date = best_birth.get("d")
    birth_date_str = norm_date if norm_date else disp_date

    # Extract year
    birth_year = _extract_year_from_date(birth_date_str, idx, "birth")

    # Extract place
    birth_place = best_birth.get("p", "").strip() if best_birth.get("p") else None

    # Log selection
    logger.debug(
        f"Item {idx}: Selected birth event - Date: '{birth_date_str}', "
        f"Place: '{birth_place}', Year: {birth_year}, "
        f"Alternate: {best_birth.get('pa')}, "
        f"Detail score: {_calculate_place_detail_score(best_birth)}"
    )

    # Log all available birth events if multiple
    if len(birth_events) > 1:
        logger.debug(f"Item {idx}: Multiple birth events available ({len(birth_events)}):")
        for i, event in enumerate(birth_events):
            logger.debug(
                f"  Birth event {i+1}: Date: '{event.get('d')}', "
                f"Place: '{event.get('p')}', "
                f"Alternate: {event.get('pa')}, "
                f"Detail score: {_calculate_place_detail_score(event)}"
            )

    return birth_year, birth_date_str, birth_place


def _extract_death_info(events_list: list[dict], idx: int) -> tuple[Optional[int], Optional[str], Optional[str], bool]:
    """Extract death year, date, place, and living status from events list."""
    death_events = [e for e in events_list if isinstance(e, dict) and e.get("t") == "Death"]

    if not death_events:
        return None, None, None, True  # is_living = True

    best_death = _select_best_event(death_events)
    if not best_death:
        return None, None, None, True

    # Extract date information
    norm_date = best_death.get("nd")
    disp_date = best_death.get("d")
    death_date_str = norm_date if norm_date else disp_date

    # Extract year
    death_year = _extract_year_from_date(death_date_str, idx, "death")

    # Extract place
    death_place = best_death.get("p", "").strip() if best_death.get("p") else None

    # Log selection
    logger.debug(
        f"Item {idx}: Selected death event - Date: '{death_date_str}', "
        f"Place: '{death_place}', Year: {death_year}, "
        f"Alternate: {best_death.get('pa')}"
    )

    return death_year, death_date_str, death_place, False  # is_living = False


# Parsing (Definition before use in _handle_search_phase)
def _parse_treesui_list_response(  # type: ignore
    treesui_response: list[dict[str, Any]],
) -> Optional[list[dict[str, Any]]]:
    """
    Parses the specific TreesUI List API response provided by the user
    to extract information needed for scoring and display.
    """
    parsed_results = []
    logger.debug(
        f"Parsing {len(treesui_response)} items from TreesUI List API response."
    )

    for idx, person_raw in enumerate(treesui_response):
        if not isinstance(person_raw, dict):
            logger.warning(f"Skipping item {idx}: Not dict")
            continue

        try:
            # Extract GID parts
            gid_result = _extract_gid_parts(person_raw, idx)
            if not gid_result:
                continue
            person_id, tree_id = gid_result

            # Extract name parts
            first_name_part, surname_part, full_name = _extract_name_parts(person_raw, idx)

            # Extract gender
            gender = _extract_gender(person_raw, idx)

            # Extract events
            events_list = person_raw.get("Events", [])

            if isinstance(events_list, list):
                # Extract birth information
                birth_year, birth_date_str, birth_place = _extract_birth_info(events_list, idx)

                # Extract death information
                death_year, death_date_str, death_place, is_living = _extract_death_info(events_list, idx)
            else:
                logger.warning(f"Item {idx}: Events key issue: {events_list}")
                birth_year = birth_date_str = birth_place = None
                death_year = death_date_str = death_place = None
                is_living = True

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
    suggestions_to_score: list[dict],
    search_criteria: dict[str, Any],
) -> Optional[tuple[dict, dict]]:
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
    selection = _select_top_candidate(scored_candidates)
    if not selection:
        # Log error and display to user
        logger.error("Failed to select top candidate.")
        print("\nFailed to select top candidate.")
        return None
    # Field-by-field comparison display has been removed as requested
    return selection


# End of _handle_selection_phase


# Details Fetch Phase
def _handle_details_phase(
    selected_candidate_raw: dict,
    session_manager_local: SessionManager,
) -> Optional[dict]:
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
            session_manager_local.api_manager._my_profile_id = owner_profile_id  # type: ignore[attr-defined]
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
        logger.error("Cannot fetch details: Missing PersonId/TreeId.")
        print("\nError: Missing IDs for detail fetch.")
        return None
    if not callable(call_facts_user_api):
        # Log error and display to user
        logger.error("Cannot fetch details: Function missing.")
        print("\nError: Details fetching utility unavailable.")
        return None

    # Call the API
    api_ids = ApiIdentifiers(owner_profile_id=owner_profile_id, api_person_id=api_person_id, api_tree_id=api_tree_id)
    person_research_data = call_facts_user_api(
        session_manager_local, api_ids, base_url
    )
    if person_research_data is None:
        # Log warning and display to user
        logger.warning("Failed to retrieve detailed info.")
        print("\nWarning: Could not retrieve detailed info.")
        return None
    return person_research_data


# End of _handle_details_phase

# Helper functions for _handle_supplementary_info_phase - Phase 1: Base Info Retrieval

def _get_base_owner_info(session_manager_local: SessionManager) -> tuple[str, Optional[str], Optional[str], str]:
    """Get base URL and owner information from session manager."""
    base_url = config_schema.api.base_url.rstrip("/")
    owner_tree_id = getattr(session_manager_local, "my_tree_id", None)
    owner_profile_id = getattr(session_manager_local, "my_profile_id", None)
    owner_name = getattr(session_manager_local, "tree_owner_name", "the Tree Owner")
    return base_url, owner_tree_id, owner_profile_id, owner_name


def _resolve_owner_tree_id_from_config(session_manager_local: SessionManager, owner_tree_id: Optional[str]) -> Optional[str]:
    """Resolve owner tree ID from config if missing."""
    if not owner_tree_id:
        config_tree_id = getattr(config_schema.api, "tree_id", None)
        if config_tree_id:
            session_manager_local.api_manager.my_tree_id = config_tree_id
            logger.info(f"Using tree ID from configuration: {config_tree_id}")
            return config_tree_id
    return owner_tree_id


def _resolve_owner_profile_id(session_manager_local: SessionManager, owner_profile_id: Optional[str]) -> Optional[str]:
    """Resolve owner profile ID from environment if missing."""
    if not owner_profile_id:
        env_profile_id = os.environ.get("MY_PROFILE_ID")
        if env_profile_id:
            session_manager_local.api_manager._my_profile_id = env_profile_id  # type: ignore[attr-defined]
            logger.info(f"Using profile ID from environment variables: {env_profile_id}")
            return env_profile_id
    return owner_profile_id


def _resolve_owner_name(session_manager_local: SessionManager, owner_name: str, owner_profile_id: Optional[str]) -> str:
    """Resolve owner name from config if missing."""
    if owner_name == "the Tree Owner":
        config_owner_name = getattr(config_schema, "user_name", None)
        if config_owner_name and config_owner_name != "Tree Owner":
            session_manager_local.api_manager._tree_owner_name = config_owner_name  # type: ignore[attr-defined]
            logger.info(f"Using tree owner name from configuration: {config_owner_name}")
            return config_owner_name
        if owner_profile_id:
            resolved_name = config_owner_name if config_owner_name else "Tree Owner"
            session_manager_local.api_manager._tree_owner_name = resolved_name  # type: ignore[attr-defined]
            logger.info(f"Using tree owner name from config/default: {resolved_name}")
            return resolved_name
    return owner_name


# Helper functions for _handle_supplementary_info_phase - Phase 2: ID Extraction

def _extract_ids_from_detailed_data(
    person_research_data: dict,
    selected_candidate_processed: dict,
) -> tuple[Optional[str], Optional[str], Optional[str], str, bool, str]:
    """Extract IDs from detailed person research data."""
    logger.debug("Attempting to extract relationship IDs from detailed person_research_data...")

    raw_cand_for_name_fallback = selected_candidate_processed.get("raw_data", {})
    temp_person_id = person_research_data.get("PersonId")
    temp_tree_id = person_research_data.get("TreeId")
    temp_global_id = person_research_data.get("UserId")
    temp_name = _extract_best_name_from_details(person_research_data, raw_cand_for_name_fallback)

    essential_ids_found = bool((temp_person_id and temp_tree_id) or temp_global_id)

    if essential_ids_found:
        logger.debug(
            f"Using IDs from Detailed Fetch: Name='{temp_name}', PersonID='{temp_person_id}', "
            f"TreeID='{temp_tree_id}', GlobalID='{temp_global_id}'"
        )
        return temp_person_id, temp_tree_id, temp_global_id, temp_name, True, "Detailed Fetch Success"
    logger.debug("Detailed data fetched, but essential relationship IDs missing. Will attempt fallback.")
    return None, None, None, "Selected Person", False, "Detailed Fetch Failed (Missing IDs)"


def _extract_ids_from_raw_suggestion(
    selected_candidate_processed: dict,
) -> tuple[Optional[str], Optional[str], Optional[str], str, bool, str]:
    """Extract IDs from raw suggestion data as fallback."""
    logger.debug("Attempting to extract relationship IDs from raw suggestion data (Fallback)...")

    raw_data = selected_candidate_processed.get("raw_data", {})
    parsed_sugg = selected_candidate_processed.get("parsed_suggestion", {})

    if not raw_data:
        logger.error("Critical: Cannot find raw_data for fallback to get relationship IDs.")
        return None, None, None, "Selected Person", False, "Fallback Failed (No Raw Data)"

    temp_person_id = raw_data.get("PersonId")
    temp_tree_id = raw_data.get("TreeId")
    temp_global_id = raw_data.get("UserId")
    temp_name = parsed_sugg.get("full_name_disp", raw_data.get("FullName", "Selected Match"))

    essential_ids_found = bool((temp_person_id and temp_tree_id) or temp_global_id)

    if essential_ids_found:
        logger.debug(
            f"Using IDs from Raw Suggestion Fallback: Name='{temp_name}', PersonID='{temp_person_id}', "
            f"TreeID='{temp_tree_id}', GlobalID='{temp_global_id}'"
        )
        return temp_person_id, temp_tree_id, temp_global_id, temp_name, True, "Raw Suggestion Fallback Success"
    logger.error("Fallback failed: Raw suggestion data also missing essential relationship IDs.")
    return None, None, None, "Selected Person", False, "Fallback Failed (Missing IDs)"


def _extract_selected_person_ids(
    person_research_data: Optional[dict],
    selected_candidate_processed: dict,
) -> tuple[Optional[str], Optional[str], Optional[str], str, bool, str]:
    """Extract selected person IDs from detailed data or raw suggestion fallback."""
    # Attempt 1: Extract from detailed person_research_data
    if person_research_data:
        person_id, tree_id, global_id, name, found, source = _extract_ids_from_detailed_data(
            person_research_data, selected_candidate_processed
        )
        if found:
            return person_id, tree_id, global_id, name, found, source
    else:
        logger.debug("Detailed person_research_data not available. Will attempt fallback for relationship IDs.")
        source = "Detailed Fetch Skipped (No Data)"

    # Attempt 2: Fallback to raw suggestion data
    return _extract_ids_from_raw_suggestion(selected_candidate_processed)


def _log_final_ids(
    owner_tree_id: Optional[str],
    owner_profile_id: Optional[str],
    selected_name: str,
    selected_person_tree_id: Optional[str],
    selected_tree_id: Optional[str],
    selected_person_global_id: Optional[str],
    source_of_ids: str,
) -> None:
    """Log final IDs being used for relationship calculation."""
    logger.debug(f"Final IDs for relationship check (Source: {source_of_ids}):")
    logger.debug(f"  Owner Tree ID         : {owner_tree_id} (Type: {type(owner_tree_id)})")
    logger.debug(f"  Owner Profile ID      : {owner_profile_id} (Type: {type(owner_profile_id)})")
    logger.debug(f"  Selected Person Name  : {selected_name} (Type: {type(selected_name)})")
    logger.debug(
        f"  Selected PersonTreeID : {selected_person_tree_id} "
        f"(Person's ID within a tree, Type: {type(selected_person_tree_id)})"
    )
    logger.debug(
        f"  Selected TreeID       : {selected_tree_id} "
        f"(The tree this person belongs to, Type: {type(selected_tree_id)})"
    )
    logger.debug(
        f"  Selected Global ID    : {selected_person_global_id} "
        f"(Often UserID/ProfileID, Type: {type(selected_person_global_id)})"
    )


# Helper functions for _handle_supplementary_info_phase - Phase 3: Relationship Calculation Method

def _determine_relationship_calculation_method(
    essential_ids_found: bool,
    owner_tree_id: Optional[str],
    owner_profile_id: Optional[str],
    selected_person_tree_id: Optional[str],
    selected_tree_id: Optional[str],
    selected_person_global_id: Optional[str],
) -> tuple[bool, bool, bool, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Determine which relationship calculation method to use."""
    can_attempt_calculation = essential_ids_found
    owner_tree_id_str = str(owner_tree_id) if owner_tree_id else None
    selected_tree_id_str = str(selected_tree_id) if selected_tree_id else None
    owner_profile_id_str = str(owner_profile_id).upper() if owner_profile_id else None
    selected_global_id_str = str(selected_person_global_id).upper() if selected_person_global_id else None

    is_owner = False
    can_calc_tree_ladder = False
    can_calc_discovery_api = False

    if can_attempt_calculation:
        # Check if selected person is the tree owner
        is_owner = bool(
            selected_global_id_str
            and owner_profile_id_str
            and selected_global_id_str == owner_profile_id_str
        )

        # Conditions for using Tree Ladder API (/getladder)
        can_calc_tree_ladder = bool(
            owner_tree_id_str
            and selected_tree_id_str
            and selected_person_tree_id
            and selected_tree_id_str == owner_tree_id_str
        )

        # Conditions for using Discovery Relationship API
        can_calc_discovery_api = bool(
            selected_person_global_id and owner_profile_id_str
        )

    return (
        is_owner,
        can_calc_tree_ladder,
        can_calc_discovery_api,
        owner_tree_id_str,
        selected_tree_id_str,
        owner_profile_id_str,
        selected_global_id_str,
    )


def _log_relationship_calculation_checks(ctx: RelationshipCalcContext) -> None:
    """Log relationship calculation method checks."""
    logger.debug(f"Relationship calculation checks (Source: {ctx.source_of_ids}):")
    logger.debug(f"  Can Attempt Calc?     : {ctx.can_attempt_calculation}")
    logger.debug(
        f"  is_owner              : {ctx.is_owner} "
        f"(OwnerG='{ctx.owner_profile_id_str}', SelectedG='{ctx.selected_global_id_str}')"
    )
    logger.debug(
        f"  can_calc_tree_ladder  : {ctx.can_calc_tree_ladder} "
        f"(OwnerT='{ctx.owner_tree_id_str}', SelectedT='{ctx.selected_tree_id_str}', "
        f"SelectedP_in_Tree exists?={bool(ctx.selected_person_tree_id)})"
    )
    logger.debug(
        f"  can_calc_discovery_api: {ctx.can_calc_discovery_api} "
        f"(OwnerG exists?={bool(ctx.owner_profile_id_str)}, SelectedG exists?={bool(ctx.selected_global_id_str)})"
    )


# Helper functions for _handle_supplementary_info_phase - Phase 4: API Call Logic

def _handle_owner_relationship(selected_name: str, owner_name: str) -> tuple[str, bool]:
    """Handle case where selected person is the tree owner."""
    print(f"=== Relationship Path to {owner_name} ===")
    print(f"  {selected_name} is the tree owner ({owner_name}).")
    logger.info(f"{selected_name} is Tree Owner. No relationship path calculation needed.")
    formatted_path = f"{selected_name} is the tree owner."
    return formatted_path, True


def _parse_jsonp_response(relationship_result_data: str) -> Optional[dict]:
    """Parse JSONP response to extract JSON data."""
    jsonp_match = re.search(r"no\((.*)\)", relationship_result_data)
    if not jsonp_match:
        logger.error("JSONP format not recognized")
        return None

    json_str = jsonp_match.group(1)
    try:
        json_data = json.loads(json_str)
        return {
            "html": json_data.get("html", ""),
            "status": json_data.get("status", "unknown"),
            "message": json_data.get("message", ""),
        }
    except json.JSONDecodeError as json_err:
        logger.error(f"Error parsing JSON from JSONP: {json_err}")
        return None


def _extract_relationship_data_from_html(html_content: str) -> list[dict]:
    """Extract relationship data from HTML content using BeautifulSoup."""
    relationship_data = []

    if not html_content or not isinstance(html_content, str):
        return relationship_data

    soup = BeautifulSoup(html_content, "html.parser")
    list_items = soup.find_all("li")

    for item in list_items:
        # Skip icon items
        if hasattr(item, 'get') and item.get("aria-hidden") == "true":  # type: ignore[union-attr]
            continue

        # Extract name
        name_elem = item.find("b") if hasattr(item, 'find') else None  # type: ignore[union-attr]
        name = name_elem.get_text() if name_elem and hasattr(name_elem, 'get_text') else (item.get_text() if hasattr(item, 'get_text') else str(item))  # type: ignore[union-attr]

        # Extract relationship description
        rel_elem = item.find("i") if hasattr(item, 'find') else None  # type: ignore[union-attr]
        relationship = rel_elem.get_text() if rel_elem and hasattr(rel_elem, 'get_text') else ""  # type: ignore[union-attr]

        # Extract lifespan
        text = item.get_text() if hasattr(item, 'get_text') else str(item)  # type: ignore[union-attr]
        lifespan_match = re.search(r"(\d{4})-(\d{4}|\-)", text)
        lifespan = lifespan_match.group(0) if lifespan_match else ""

        relationship_data.append({
            "name": name,
            "relationship": relationship,
            "lifespan": lifespan,
        })

    return relationship_data


def _format_tree_ladder_path(
    api_response_dict: dict,
    selected_name: str,
    owner_name: str,
) -> Optional[str]:
    """Format relationship path from Tree Ladder API response."""
    try:
        sn_str = str(selected_name) if selected_name else "Unknown"
        on_str = str(owner_name) if owner_name else "Unknown"

        # Get raw formatted path
        raw_formatted_path = format_api_relationship_path(api_response_dict, on_str, sn_str)
        logger.debug(f"Raw formatted path from format_api_relationship_path:\n{raw_formatted_path}")

        # Extract relationship data from HTML
        html_content = api_response_dict.get("html", "")
        relationship_data = _extract_relationship_data_from_html(html_content)

        # Convert to unified format and format
        unified_path = convert_api_path_to_unified_format(relationship_data, sn_str)
        return format_relationship_path_unified(unified_path, sn_str, on_str, None)

    except Exception as e:
        logger.error(f"Error formatting relationship path: {e}", exc_info=True)
        # Fall back to raw formatted path
        # Variables on_str and sn_str are defined in try block before any exception can occur
        return format_api_relationship_path(api_response_dict, on_str, sn_str)  # type: ignore[possibly-unbound]


def _handle_tree_ladder_api_call(
    session_manager_local: SessionManager,
    base_url: str,
    owner_tree_id_str: str,
    selected_person_tree_id: str,
    selected_name: str,
    owner_name: str,
) -> tuple[Optional[str], bool, str]:
    """Handle Tree Ladder API call and formatting."""
    api_called_for_rel = "Tree Ladder (/getladder)"
    logger.debug(f"Attempting relationship calculation via {api_called_for_rel}...")

    sp_tree_id_str = str(selected_person_tree_id)
    ot_id_str = str(owner_tree_id_str)

    ladder_api_url = f"{base_url}/family-tree/person/tree/{ot_id_str}/person/{sp_tree_id_str}/getladder?callback=no"
    logger.debug(f"\nLadder API URL: {ladder_api_url}\n")

    relationship_result_data = call_getladder_api(session_manager_local, ot_id_str, sp_tree_id_str, base_url)

    if not relationship_result_data:
        logger.warning(f"{api_called_for_rel} API call failed or returned no data.")
        return f"(Failed to retrieve data from {api_called_for_rel} API)", False, api_called_for_rel

    if not callable(format_api_relationship_path):
        logger.error("format_api_relationship_path function not available.")
        return "(Error: Formatting function for relationship path unavailable)", True, api_called_for_rel

    # Parse JSONP response
    api_response_dict = _parse_jsonp_response(relationship_result_data)
    if not api_response_dict:
        return "(JSONP format not recognized)", True, api_called_for_rel

    # Format the path
    formatted_path = _format_tree_ladder_path(api_response_dict, selected_name, owner_name)
    if not formatted_path:
        return f"(Error formatting relationship path from {api_called_for_rel})", True, api_called_for_rel

    return formatted_path, True, api_called_for_rel


def _format_discovery_api_path_fallback(
    discovery_api_response: dict,
    selected_name: str,
    owner_name: str,
) -> str:
    """Format Discovery API path using fallback formatter."""
    path_steps = discovery_api_response.get("path", [])
    path_display_lines = [f"  {selected_name}"]
    name_formatter_local = format_name if callable(format_name) else lambda x: str(x).title()

    for step in path_steps:
        step_name_raw = step.get("name", "?")
        step_rel_raw = step.get("relationship", "related to").capitalize()
        path_display_lines.append(f"  -> {step_rel_raw} is {name_formatter_local(step_name_raw)}")

    path_display_lines.append(f"  -> {owner_name} (Tree Owner / You)")
    return "\n".join(path_display_lines)


def _format_discovery_api_path(
    discovery_api_response: dict,
    selected_name: str,
    owner_name: str,
) -> Optional[str]:
    """Format relationship path from Discovery API response."""
    try:
        # Convert to unified format
        unified_path = convert_discovery_api_path_to_unified_format(discovery_api_response, selected_name)

        if unified_path:
            formatted_path = format_relationship_path_unified(unified_path, selected_name, owner_name, None)
            logger.debug("Discovery API path formatted using unified formatter")
            return formatted_path
        logger.warning("Failed to convert Discovery API path to unified format")
        # Fallback to simple formatting
        formatted_path = _format_discovery_api_path_fallback(discovery_api_response, selected_name, owner_name)
        logger.debug("Discovery API path constructed using fallback formatter")
        return formatted_path
    except Exception as e:
        logger.error(f"Error formatting Discovery API path: {e}", exc_info=True)
        # Fallback to simple formatting
        formatted_path = _format_discovery_api_path_fallback(discovery_api_response, selected_name, owner_name)
        logger.debug(f"Discovery API path constructed using fallback formatter after error: {e}")
        return formatted_path


def _handle_discovery_api_call(
    session_manager_local: SessionManager,
    base_url: str,
    selected_person_global_id: str,
    owner_profile_id_str: str,
    selected_name: str,
    owner_name: str,
) -> tuple[Optional[str], bool, str]:
    """Handle Discovery Relationship API call and formatting."""
    api_called_for_rel = "Discovery Relationship API"
    logger.debug(f"Attempting relationship calculation via {api_called_for_rel}...")

    sp_global_id_str = str(selected_person_global_id)
    op_id_str = str(owner_profile_id_str)

    discovery_api_response = call_discovery_relationship_api(
        session_manager_local,
        sp_global_id_str,
        op_id_str,
        base_url,
        timeout=30,
    )

    if not discovery_api_response or not isinstance(discovery_api_response, dict):
        logger.warning(f"{api_called_for_rel} API call failed or returned invalid data.")
        return f"(Failed to retrieve or parse data from {api_called_for_rel} API)", False, api_called_for_rel

    # Format the path
    formatted_path = _format_discovery_api_path(discovery_api_response, selected_name, owner_name)

    # Check for message in API response if no path was found
    if "message" in discovery_api_response and not formatted_path:
        formatted_path = f"(Discovery API: {discovery_api_response.get('message', 'No direct path found')})"
        logger.warning(f"Discovery API response: {formatted_path}")
    elif not formatted_path:
        formatted_path = "(Discovery API: Path data missing or in unexpected format)"
        logger.warning(f"Discovery API response structure unexpected: {discovery_api_response}")

    return formatted_path, True, api_called_for_rel


# Helper functions for _handle_supplementary_info_phase - Phase 5: Formatting and Display

def _is_error_message(formatted_path: str) -> bool:
    """Check if formatted path is an error message."""
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
    return any(formatted_path.startswith(err_start) for err_start in known_error_starts_tuple)


def _clean_formatted_path(formatted_path: str, owner_name: str) -> str:
    """Clean formatted path by removing duplicate headers and replacing Unknown."""
    display_owner_name = owner_name if owner_name else "Tree Owner"

    # Remove the header line from the formatted path to avoid duplicate headers
    if "===Relationship Path to" in formatted_path:
        path_lines = formatted_path.split("\n")
        if len(path_lines) > 1:
            formatted_path = "\n".join(path_lines[1:])
            # Replace "Unknown's" with owner name
            formatted_path = formatted_path.replace("Unknown's", f"{display_owner_name}'s")

    return formatted_path


def _display_formatted_path(
    formatted_path: str,
    owner_name: str,
    api_called_for_rel: Union[str, bool],
) -> None:
    """Display formatted relationship path."""
    display_owner_name = owner_name if owner_name else "Tree Owner"
    print(f"=== Relationship Path to {display_owner_name} ===\n")

    if _is_error_message(formatted_path):
        # This is an error message, print it as such
        print(f"  {formatted_path}")
        logger.warning(
            f"Relationship path calculation resulted in message/error: {formatted_path} "
            f"(API called: {api_called_for_rel})"
        )
    else:
        # This is a successfully formatted path
        cleaned_path = _clean_formatted_path(formatted_path, owner_name)
        print(f"{cleaned_path}\n")
        logger.debug(f"Successfully displayed relationship path via {api_called_for_rel}.")


def _display_calculation_failure(
    can_attempt_calculation: bool,
    selected_name: str,
    source_of_ids: str,
) -> None:
    """Display failure message when calculation could not be performed."""
    default_fail_message = f"(Could not calculate relationship path for {selected_name})"
    print(f"  {default_fail_message}")

    if not can_attempt_calculation:
        reason_detail = "  Reason: Essential IDs missing from detailed data and fallback data."
        print(reason_detail)
        logger.error(
            f"{default_fail_message}. {reason_detail.strip()} (Source of IDs: {source_of_ids})."
        )
    else:
        reason_detail = "  Reason: Calculation conditions not met (e.g., tree mismatch, API data issue)."
        print(reason_detail)
        logger.error(
            f"{default_fail_message}. {reason_detail.strip()} (Source of IDs: {source_of_ids}). "
            f"Check prior logs for API call failures."
        )


def _display_unexpected_state(
    selected_name: str,
    api_called_for_rel: Union[str, bool],
) -> None:
    """Display message for unexpected state where calculation performed but no path generated."""
    logger.error(
        f"Unexpected state: Calculation performed for {selected_name} via {api_called_for_rel}, "
        f"but no formatted path or error message was generated."
    )
    print(
        f"  (Relationship path for {selected_name} could not be determined or displayed "
        f"via {api_called_for_rel})."
    )


def _display_relationship_result(
    formatted_path: Optional[str],
    is_owner: bool,
    calculation_performed: bool,
    can_attempt_calculation: bool,
    owner_name: str,
    selected_name: str,
    api_called_for_rel: Union[str, bool],
    source_of_ids: str,
) -> None:
    """Display final relationship result or failure message."""
    print("")  # Add spacing

    if formatted_path:
        _display_formatted_path(formatted_path, owner_name, api_called_for_rel)
    elif not is_owner and not calculation_performed:
        _display_calculation_failure(can_attempt_calculation, selected_name, source_of_ids)
    elif not is_owner and calculation_performed and not formatted_path:
        _display_unexpected_state(selected_name, api_called_for_rel)


# Helper functions for handle_api_report - Phase 1: Dependency and Session Validation

def _check_dependencies() -> tuple[bool, list[str]]:
    """Check if all required dependencies are available."""
    dependencies = [
        ("Core Utils", CORE_UTILS_AVAILABLE),
        ("API Utils", API_UTILS_AVAILABLE),
        ("Gedcom Utils", GEDCOM_UTILS_AVAILABLE),
        ("Config", config_schema),
        ("Session Manager", session_manager),
    ]

    missing = [name for name, available in dependencies if not available]
    return len(missing) == 0, missing


def _validate_browser_session() -> bool:
    """Validate that browser session is available."""
    logger.debug(f"Checking browser session. Session manager ID: {id(session_manager)}")
    logger.debug(f"Session manager driver: {session_manager.driver}")
    logger.debug(f"Session manager driver_live: {getattr(session_manager, 'driver_live', 'N/A')}")

    if not session_manager.driver:
        logger.error(f"Browser session not available. Session manager: {session_manager}, Driver: {session_manager.driver}")
        print("\nERROR: Browser session not available.")
        return False

    logger.debug("Browser session is available and ready.")
    return True


def _refresh_cookies_from_browser() -> bool:
    """Refresh cookies in requests session from browser session."""
    if not session_manager.driver:
        return False

    try:
        selenium_cookies = session_manager.driver.get_cookies()
        for cookie in selenium_cookies:
            session_manager._requests_session.cookies.set(
                cookie["name"], cookie["value"]
            )
        logger.debug("Successfully refreshed cookies from browser session.")
        return True
    except Exception as cookie_err:
        logger.error(f"Failed to refresh cookies from browser: {cookie_err}")
        print(f"\nERROR: Failed to refresh cookies: {cookie_err}")
        return False


def _validate_global_requests_session() -> bool:
    """Validate that requests session exists in global session_manager."""
    if not hasattr(session_manager, "_requests_session") or not session_manager._requests_session:
        logger.error("No requests session available for API calls.")
        print("\nERROR: API session not available. Please ensure you are logged in to Ancestry.")
        return False
    return True


def _validate_cookies_available() -> bool:
    """Validate that cookies are available in requests session."""
    if not session_manager._requests_session.cookies:
        logger.error("Still no cookies available after refresh attempt.")
        print("\nERROR: Failed to obtain authentication cookies. Please restart the application.")
        return False
    return True


# Helper functions for handle_api_report - Phase 2: Login and Cookie Management

def _handle_logged_in_user() -> bool:
    """Handle case where user is already logged in - refresh cookies."""
    logger.debug("User is already logged in. Refreshing cookies...")

    if not _validate_global_requests_session():
        return False

    if not _refresh_cookies_from_browser():
        return False

    if not _validate_cookies_available():
        return False

    logger.debug("Successfully refreshed authentication cookies.")
    return True


def _attempt_browser_login() -> bool:
    """Attempt to log in via browser and refresh cookies."""
    from utils import log_in, login_status

    # First check if we're already logged in
    login_stat = login_status(session_manager, disable_ui_fallback=True)
    if login_stat is True:
        logger.info("User is already logged in. No need to navigate to sign-in page.")

        # Refresh the cookies in the requests session
        return _refresh_cookies_from_browser()
    # Try to log in
    login_result = log_in(session_manager)
    if login_result != "LOGIN_SUCCEEDED":
        logger.error(f"Failed to log in: {login_result}")
        print(f"\nERROR: Failed to log in: {login_result}")
        return False

    return True


def _initialize_session_with_login() -> bool:
    """Initialize session with authentication cookies via login."""
    logger.warning("No cookies available in the requests session. Need to log in.")

    # Try to initialize the session with authentication cookies
    print("\nAttempting to log in to Ancestry...")
    try:
        # Try to start the browser session
        if not session_manager.start_browser(action_name="API Report Browser Init"):
            logger.error("Failed to start browser session.")
            print("\nERROR: Failed to start browser session.")
            return False

        # Attempt login
        if not _attempt_browser_login():
            return False

        # Check if we now have cookies
        if not session_manager._requests_session.cookies:
            logger.error("Still no cookies available after login attempt.")
            print("\nERROR: Still no cookies available after login attempt.")
            return False

        print("\nSuccessfully initialized session with authentication cookies.")
        logger.info("Successfully initialized session with authentication cookies.")
        return True
    except Exception as e:
        logger.error(f"Error initializing session: {e}")
        print(f"\nERROR: Error initializing session: {e}")
        return False


def _handle_not_logged_in_user() -> bool:
    """Handle case where user is not logged in."""
    if not _validate_global_requests_session():
        return False

    # Check if we have cookies in the requests session
    if not session_manager._requests_session.cookies:
        return _initialize_session_with_login()
    # We have cookies, so we're good to go
    logger.info("Session already has authentication cookies.")
    return True


def _ensure_authenticated_session() -> bool:
    """Ensure we have an authenticated session with valid cookies."""
    from utils import login_status

    # Check login status
    login_stat = login_status(session_manager, disable_ui_fallback=True)

    # If we're already logged in, refresh the cookies
    if login_stat is True:
        return _handle_logged_in_user()
    return _handle_not_logged_in_user()


def _handle_supplementary_info_phase(
    person_research_data: Optional[dict],
    selected_candidate_processed: dict,
    session_manager_local: SessionManager,
):
    """
    Simplified to only calculate and display the relationship path.
    Family details functionality removed to keep Action 11 focused and reliable.
    """
    # --- Get Base Info ---
    base_url, owner_tree_id, owner_profile_id, owner_name = _get_base_owner_info(session_manager_local)

    # Resolve missing owner information
    owner_tree_id = _resolve_owner_tree_id_from_config(session_manager_local, owner_tree_id)
    owner_profile_id = _resolve_owner_profile_id(session_manager_local, owner_profile_id)
    owner_name = _resolve_owner_name(session_manager_local, owner_name, owner_profile_id)

    # --- Skip Family Details Section ---
    # Action 11 simplified to focus on search, scoring, and relationship calculation only

    # --- Extract Selected Person IDs ---
    (
        selected_person_tree_id,
        selected_tree_id,
        selected_person_global_id,
        selected_name,
        essential_ids_found,
        source_of_ids,
    ) = _extract_selected_person_ids(person_research_data, selected_candidate_processed)

    # --- Log Final IDs Being Used ---
    _log_final_ids(
        owner_tree_id,
        owner_profile_id,
        selected_name,
        selected_person_tree_id,
        selected_tree_id,
        selected_person_global_id,
        source_of_ids,
    )

    # --- Determine Relationship Calculation Method ---
    (
        is_owner,
        can_calc_tree_ladder,
        can_calc_discovery_api,
        owner_tree_id_str,
        selected_tree_id_str,
        owner_profile_id_str,
        selected_global_id_str,
    ) = _determine_relationship_calculation_method(
        essential_ids_found,
        owner_tree_id,
        owner_profile_id,
        selected_person_tree_id,
        selected_tree_id,
        selected_person_global_id,
    )

    can_attempt_calculation = essential_ids_found

    rel_calc_ctx = RelationshipCalcContext(
        can_attempt_calculation=can_attempt_calculation,
        is_owner=is_owner,
        can_calc_tree_ladder=can_calc_tree_ladder,
        can_calc_discovery_api=can_calc_discovery_api,
        owner_tree_id_str=owner_tree_id_str,
        selected_tree_id_str=selected_tree_id_str,
        owner_profile_id_str=owner_profile_id_str,
        selected_global_id_str=selected_global_id_str,
        selected_person_tree_id=selected_person_tree_id,
        source_of_ids=source_of_ids,
    )
    _log_relationship_calculation_checks(rel_calc_ctx)

    # Initialize relationship calculation variables
    api_called_for_rel = "None"
    formatted_path = None
    calculation_performed = False

    # --- Directly Call API and Format/Print Relationship ---
    if is_owner:
        formatted_path, calculation_performed = _handle_owner_relationship(selected_name, owner_name)

    elif can_calc_tree_ladder:
        if owner_tree_id_str and selected_person_tree_id:
            formatted_path, calculation_performed, api_called_for_rel = _handle_tree_ladder_api_call(
                session_manager_local,
                base_url,
                owner_tree_id_str,
                selected_person_tree_id,
                selected_name,
                owner_name,
            )
        else:
            formatted_path = None
            calculation_performed = False
            api_called_for_rel = False

    elif can_calc_discovery_api:
        if selected_person_global_id and owner_profile_id_str:
            formatted_path, calculation_performed, api_called_for_rel = _handle_discovery_api_call(
                session_manager_local,
                base_url,
                selected_person_global_id,
                owner_profile_id_str,
                selected_name,
                owner_name,
            )
        else:
            formatted_path = None
            calculation_performed = False
            api_called_for_rel = False

    # --- Display Final Result or Failure Message ---
    _display_relationship_result(
        formatted_path,
        is_owner,
        calculation_performed,
        can_attempt_calculation,
        owner_name,
        selected_name,
        api_called_for_rel,
        source_of_ids,
    )


# End of _handle_supplementary_info_phase


def handle_api_report(session_manager_param: Optional[Any] = None) -> bool:
    """Orchestrates the process using only initial API data for comparison."""

    # Use the session_manager passed by the framework if available, otherwise use module-level instance
    active_session_manager = session_manager_param if session_manager_param else session_manager
    if session_manager_param:
        logger.debug("Using session_manager passed by framework.")

    result = False  # Default to failure

    # Dependency Checks
    dependencies_ok, missing = _check_dependencies()
    if not dependencies_ok:
        logger.critical(f"handle_api_report: Dependencies missing: {', '.join(missing)}.")
        print(f"\nCRITICAL ERROR: Dependencies unavailable ({', '.join(missing)}). Check logs.")
    # Browser Session Validation
    elif not _validate_browser_session() or not _ensure_authenticated_session():
        pass  # result already False
    else:
        # Phase 1: Search...
        search_criteria = _get_search_criteria()
        if not search_criteria:
            logger.info("Search cancelled.")
            result = True
        else:
            suggestions_to_score = _handle_search_phase(active_session_manager, search_criteria)
            if suggestions_to_score is None:
                result = False  # Critical API failure
            elif not suggestions_to_score:
                result = True  # Search successful, no results
            else:
                # Phase 2: Score, Select & Display Initial Comparison...
                selection = _handle_selection_phase(
                    suggestions_to_score, search_criteria
                )  # Includes initial comparison display
                if not selection:
                    result = True  # No candidate selected or comparison failed gracefully
                else:
                    selected_candidate_processed, _ = selection  # Raw data unused

                    # Phase 3: Skip detailed data fetching - Action 11 simplified
                    # Phase 4: Display Relationship Path Only...
                    _handle_supplementary_info_phase(
                        None,  # No detailed data needed for simplified Action 11
                        selected_candidate_processed,
                        active_session_manager,
                    )
                    result = True

    return result


# End of handle_api_report


# --- Main Execution ---
@retry_on_failure(max_attempts=3, backoff_factor=4.0)  # Increased from 2.0 to 4.0 for better API handling
@circuit_breaker(failure_threshold=10, recovery_timeout=300)  # Increased from 5 to 10 for better tolerance
@timeout_protection(timeout=1800)  # 30 minutes for API report generation
@graceful_degradation(fallback_value=None)
@error_context("action11_api_report")
def main() -> None:
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
    search_criteria: dict[str, Any],
    max_results: int = 5,
) -> list[dict[str, Any]]:
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
) -> dict[str, Any]:
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
    details["birth_year"] = _extract_year_simple(details["birth_date"])

    # Extract death information
    death_facts = person_info.get("death", {})
    details["death_date"] = death_facts.get("date", {}).get("normalized", "Unknown")
    details["death_place"] = death_facts.get("place", {}).get("normalized", "Unknown")
    details["death_year"] = _extract_year_simple(details["death_date"])

    # Extract family information
    family = person_research_data.get("family", {})

    # Extract parents
    details["parents"] = []
    for parent in family.get("parents", []):
        parent_info = {
            "id": parent.get("id", "Unknown"),
            "name": parent.get("name", "Unknown"),
            "gender": parent.get("gender", "Unknown"),
            "birth_year": _extract_year_simple(
                parent.get("birth", {}).get("date", {}).get("normalized", "")
            ),
            "death_year": _extract_year_simple(
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
            "birth_year": _extract_year_simple(
                spouse.get("birth", {}).get("date", {}).get("normalized", "")
            ),
            "death_year": _extract_year_simple(
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
            "birth_year": _extract_year_simple(
                child.get("birth", {}).get("date", {}).get("normalized", "")
            ),
            "death_year": _extract_year_simple(
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
                sibling.get("birth", {}).get("date", {}).get("normalized", ""), 0, "birth"
            ),
            "death_year": _extract_year_from_date(
                sibling.get("death", {}).get("date", {}).get("normalized", ""), 0, "death"
            ),
        }
        details["siblings"].append(sibling_info)

    return details


def get_ancestry_relationship_path(
    session_manager: SessionManager,
    target_person_id: str,
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
        return format_api_relationship_path(
            relationship_data, owner_name, "Target Person"
        )

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


def _extract_year_simple(date_str: str) -> Optional[int]:
    """Extract year from a date string (simple version)."""
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


def run_action11(session_manager_param: Optional[Any] = None, *_: Any) -> bool:
    """Wrapper function for main.py to call."""
    # Use the session_manager passed by the framework if available
    if session_manager_param:
        return handle_api_report(session_manager_param)
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


# === SESSION SETUP FOR TESTS ===
_test_session_manager: Optional[SessionManager] = None
_test_session_uuid: Optional[str] = None


def _create_and_start_session() -> SessionManager:
    """Create and start a new session manager."""
    logger.info("Step 1: Creating SessionManager...")
    sm = SessionManager()
    logger.info("✅ SessionManager created")

    logger.info("Step 2: Configuring browser requirement...")
    sm.browser_manager.browser_needed = True
    logger.info("✅ Browser marked as needed")

    logger.info("Step 3: Starting session (database + browser)...")
    started = sm.start_sess("Action 11 API Tests")
    if not started:
        sm.close_sess(keep_db=False)
        raise AssertionError("Failed to start session - browser initialization failed")
    logger.info("✅ Session started successfully")

    return sm


def _authenticate_session(sm: SessionManager) -> None:
    """Authenticate the session using cookies or login."""
    from utils import _load_login_cookies, log_in, login_status

    logger.info("Step 4: Attempting to load saved cookies...")
    cookies_loaded = _load_login_cookies(sm)
    logger.info("✅ Loaded saved cookies from previous session" if cookies_loaded else "⚠️  No saved cookies found")

    logger.info("Step 5: Checking login status...")
    login_check = login_status(sm, disable_ui_fallback=True)

    if login_check is True:
        logger.info("✅ Already logged in")
    elif login_check is False:
        logger.info("⚠️  Not logged in - attempting login...")
        login_result = log_in(sm)
        if login_result != "LOGIN_SUCCEEDED":
            sm.close_sess(keep_db=False)
            raise AssertionError(f"Login failed: {login_result}")
        logger.info("✅ Login successful")
    else:
        sm.close_sess(keep_db=False)
        raise AssertionError("Login status check failed critically (returned None)")


def _validate_session_ready(sm: SessionManager) -> None:
    """Validate session is ready with all identifiers."""
    logger.info("Step 6: Ensuring session is ready...")
    ready = sm.ensure_session_ready("api_research - API Tests", skip_csrf=True)
    if not ready:
        sm.close_sess(keep_db=False)
        raise AssertionError("Session not ready - cookies/identifiers missing")
    logger.info("✅ Session ready")

    logger.info("Step 7: Verifying UUID is available...")
    if not sm.my_uuid:
        sm.close_sess(keep_db=False)
        raise AssertionError("UUID not available - session initialization incomplete")
    logger.info(f"✅ UUID available: {sm.my_uuid}")

    logger.info("Step 8: Verifying Profile ID is available...")
    if not sm.my_profile_id:
        sm.close_sess(keep_db=False)
        raise AssertionError("Profile ID not available - session initialization incomplete")
    logger.info(f"✅ Profile ID available: {sm.my_profile_id}")

    logger.info("Step 9: Verifying Tree ID is available...")
    if not sm.my_tree_id:
        sm.close_sess(keep_db=False)
        raise AssertionError("Tree ID not available - check TREE_NAME in .env")
    logger.info(f"✅ Tree ID available: {sm.my_tree_id}")


def _ensure_session_for_api_tests(reuse_session: bool = True) -> tuple[SessionManager, str]:
    """Ensure session is ready for API tests. Returns (session_manager, my_uuid).

    This function establishes a valid Ancestry session by:
    1. Creating and initializing a SessionManager (or reusing existing one)
    2. Starting the session (database + browser)
    3. Loading saved cookies from previous session (if available)
    4. Checking login status and logging in if needed
    5. Ensuring session is ready with all identifiers
    6. Validating UUID is available

    Args:
        reuse_session: If True, reuse existing session from previous test (default: True)

    Returns:
        tuple: (SessionManager, UUID string)
    """
    global _test_session_manager, _test_session_uuid

    # Reuse session if available and requested
    if reuse_session and _test_session_manager and _test_session_uuid:
        logger.info("Reusing authenticated session from previous test")
        return _test_session_manager, _test_session_uuid

    logger.info("=" * 80)
    logger.info("Setting up authenticated session for API tests...")
    logger.info("=" * 80)

    # Create and start new session
    sm = _create_and_start_session()

    # Authenticate the session
    _authenticate_session(sm)

    # Validate session is ready
    _validate_session_ready(sm)

    logger.info("=" * 80)
    logger.info("✅ Valid authenticated session established for API tests")
    logger.info("=" * 80)

    # Cache session for reuse
    _test_session_manager = sm
    _test_session_uuid = sm.my_uuid

    return sm, sm.my_uuid


def _build_search_criteria_from_test_person(tp: dict[str, Any]) -> dict[str, Any]:
    """Build search criteria from test person data."""
    return {
        "first_name": tp.get("name", "Fraser Gault").split()[0].lower(),
        "surname": tp.get("name", "Fraser Gault").split()[-1].lower(),
        "gender": str(tp.get("gender", "M")).lower()[0],
        "birth_year": int(tp.get("birth_year", 1941)),
        "birth_place": str(tp.get("birth_place", "Banff")).lower(),
    }


# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_live_search_fraser(skip_live_tests: bool) -> bool:
    """Live API: search for Fraser Gault and ensure a scored match is returned."""
    if skip_live_tests:
        raise AssertionError("Live API tests require SKIP_LIVE_API_TESTS=false and valid .env credentials")
    sm, _ = _ensure_session_for_api_tests()
    tp = load_test_person_from_env()
    criteria = _build_search_criteria_from_test_person(tp)
    results = search_ancestry_api_for_person(sm, criteria, max_results=5)
    assert results, "No results returned from live API search"
    top = results[0]
    name_l = str(top.get("name", "")).lower()
    assert "fraser" in name_l and "gault" in name_l, f"Top match is not Fraser Gault: {top.get('name')}"
    assert float(top.get("score", 0)) > 0, "Top match has non-positive score"
    # Birth place 'contains' logic
    bp_disp = str(top.get("birth_place", "")).lower()
    assert criteria["birth_place"] in bp_disp, f"Birth place does not contain '{criteria['birth_place']}'"
    return True


def _test_live_family_matches_env(skip_live_tests: bool) -> bool:
    """Live API: fetch person details and validate spouse/children from .env test data."""
    if skip_live_tests:
        raise AssertionError("Live API tests require SKIP_LIVE_API_TESTS=false and valid .env credentials")
    sm, _ = _ensure_session_for_api_tests()
    tp = load_test_person_from_env()
    # Reuse search to pick id/tree
    criteria = _build_search_criteria_from_test_person(tp)
    results = search_ancestry_api_for_person(sm, criteria, max_results=3)
    assert results, "No results available for details test"
    raw = results[0].get("raw_data", {})
    person_id = raw.get("PersonId")
    tree_id = raw.get("TreeId") or sm.my_tree_id
    assert person_id and tree_id, "Missing person or tree id for details fetch"
    details = get_ancestry_person_details(sm, str(person_id), str(tree_id))
    assert details, "No details returned from Facts User API"
    # Validate spouse and at least one child per .env expectations
    spouse_expect = str(tp.get("spouse_name", "Helen")).lower()
    children_expect = [c.lower() for c in tp.get("children", []) if c]
    spouses = [str(s.get("name", "")).lower() for s in details.get("spouses", [])]
    children = [str(c.get("name", "")).lower() for c in details.get("children", [])]
    assert any(spouse_expect in s for s in spouses), f"Expected spouse '{spouse_expect}' not found in {spouses}"
    assert any(any(exp in ch for exp in children_expect) for ch in children), f"Expected one of children {children_expect} not found in {children}"
    return True


def _test_live_relationship_uncle(skip_live_tests: bool) -> bool:
    """Live API: format relationship path between Fraser Gault and owner; should include 'Uncle'."""
    if skip_live_tests:
        raise AssertionError("Live API tests require SKIP_LIVE_API_TESTS=false and valid .env credentials")
    sm, _ = _ensure_session_for_api_tests()
    tp = load_test_person_from_env()
    # Search to get ids
    criteria = _build_search_criteria_from_test_person(tp)
    results = search_ancestry_api_for_person(sm, criteria, max_results=3)
    assert results, "No results available for relationship test"
    top = results[0]
    raw = top.get("raw_data", {})
    person_id = str(raw.get("PersonId"))
    tree_id = str(raw.get("TreeId") or sm.my_tree_id)
    owner_name = sm.tree_owner_name or os.getenv("TREE_OWNER_NAME", "Wayne Gault")
    target_name = top.get("name", tp.get("name", "Fraser Gault"))
    # Call ladder API and format
    from api_utils import call_getladder_api
    from relationship_utils import format_api_relationship_path
    ladder_raw = call_getladder_api(sm, tree_id, person_id, config_schema.api.base_url, timeout=20)
    assert ladder_raw and isinstance(ladder_raw, str), "GetLadder API returned no/invalid data"
    formatted = format_api_relationship_path(ladder_raw, owner_name=owner_name, target_name=target_name, relationship_type="relative")
    fmt_lower = formatted.lower()
    assert "uncle" in fmt_lower, f"Formatted relationship does not show 'uncle': {formatted}"
    assert "fraser" in fmt_lower and "gault" in fmt_lower, "Target name missing in formatted relationship"
    assert owner_name.split()[0].lower() in fmt_lower, "Owner name missing in formatted relationship"
    return True


# ==============================================
# MAIN TEST SUITE
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

    # Removed obsolete test functions that checked for functions we've already removed:
    # - test_module_imports (26 lines)
    # - test_core_function_availability (32 lines)
    # - test_search_functions (19 lines)
    # - test_scoring_functions (19 lines)
    # - test_display_functions (19 lines)
    # - test_api_integration_functions (6 lines)
    # - test_empty_globals_handling (6 lines)
    # - test_function_callable_check (4 lines)
    # - test_family_functions (8 lines)
    # - test_data_extraction_functions (8 lines)
    # - test_utility_functions (8 lines)

    # Removed more obsolete test functions:
    # - test_function_lookup_performance (28 lines)
    # - test_callable_check_performance (15 lines)
    # - test_fraser_gault_functions (17 lines)
    # - test_exception_handling (9 lines)

    # === RUN ALL TESTS ===
    # Skip live API tests when running through test runner to avoid hanging
    skip_live_tests = os.getenv("SKIP_LIVE_API_TESTS", "false").lower() == "true"
    if skip_live_tests:
        print("INFO: Skipping live API tests (SKIP_LIVE_API_TESTS=true)")
        logger.info("Skipping all live API tests due to SKIP_LIVE_API_TESTS environment variable")

    # Create wrapper functions that pass skip_live_tests parameter
    def test_live_search_fraser():
        return _test_live_search_fraser(skip_live_tests)

    def test_live_family_matches_env():
        return _test_live_family_matches_env(skip_live_tests)

    def test_live_relationship_uncle():
        return _test_live_relationship_uncle(skip_live_tests)

    # Register the live tests (these are decisive, fail on real issues)
    with suppress_logging():
        suite.run_test(
            "Live API search: Fraser Gault",
            test_live_search_fraser,
            "Uses .env to search API and verifies top match and scoring are real.",
            "Start session, call search_ancestry_api_for_person, assert name/score/place.",
            "Live Suggest/List API reachable, results scored and contain Fraser Gault.",
        )

        suite.run_test(
            "Live API details: family matches .env",
            test_live_family_matches_env,
            "Fetch detailed person info and validate spouse/children against .env expectations.",
            "Call get_ancestry_person_details and assert spouse/children present.",
            "Facts User API reachable; parsed family contains expected relatives.",
        )

        suite.run_test(
            "Live API relationship: Uncle path formatting",
            test_live_relationship_uncle,
            "GetLadder API builds a readable path identifying 'Uncle' between Fraser and owner.",
            "Call call_getladder_api and format_api_relationship_path; assert 'Uncle' present.",
            "Relationship ladder parsed correctly; formatted output meets spec.",
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
