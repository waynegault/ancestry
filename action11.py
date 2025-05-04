# --- START OF FILE action11.py ---
# action11.py
"""
Action 11: API Report - Search Ancestry API, display details, family, relationship.
V18.5: Refactored API calls to use helpers from api_utils.
       Broke down handle_api_report into phase-based helpers.
       Improved Discovery relationship parsing (direct JSON).
       Moved retry logic to api_utils helpers.
"""
# --- Standard library imports ---
import logging
import sys
import time
import urllib.parse
import json
import re  # Added for robust lifespan splitting
from pathlib import Path
from urllib.parse import urljoin, urlencode, quote
from tabulate import tabulate
import requests  # Keep for potential exception types

# Import specific types needed locally
from typing import Optional, List, Dict, Any, Tuple, Union, cast
from datetime import datetime

# --- Third-party imports ---
# (BeautifulSoup not currently needed)

# --- Local application imports ---
# Use centralized logging config setup
try:
    from logging_config import setup_logging, logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("action11")
    logger.warning("Using fallback logger for Action 11.")

# --- Load Config (Mandatory - Direct Import) ---
config_instance = None
selenium_config = None
try:
    from config import config_instance, selenium_config

    logger.info("Successfully imported config_instance and selenium_config.")
    if not config_instance or not selenium_config:
        raise ImportError("Config instances are None.")

    required_config_attrs = [
        "COMMON_SCORING_WEIGHTS",
        "NAME_FLEXIBILITY",
        "DATE_FLEXIBILITY",
        "MAX_SUGGESTIONS_TO_SCORE",
        "MAX_CANDIDATES_TO_DISPLAY",
        "BASE_URL",
    ]
    for attr in required_config_attrs:
        if not hasattr(config_instance, attr):
            raise TypeError(f"config_instance.{attr} missing.")
        value = getattr(config_instance, attr)
        if value is None:
            raise TypeError(f"config_instance.{attr} is None.")
        if isinstance(value, (dict, list, tuple)) and not value:
            logger.warning(f"config_instance.{attr} is empty.")

    if not hasattr(selenium_config, "API_TIMEOUT"):
        raise TypeError(f"selenium_config.API_TIMEOUT missing.")

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
    config_instance is None
    or not hasattr(config_instance, "COMMON_SCORING_WEIGHTS")
    or selenium_config is None
    or not hasattr(selenium_config, "API_TIMEOUT")
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

    logger.info("Successfully imported functions from gedcom_utils.")
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
format_api_relationship_path = None
parse_ancestry_person_details = None  # Import the parser
call_suggest_api = None
call_facts_user_api = None
call_getladder_api = None
call_discovery_relationship_api = None
call_treesui_list_api = None
API_UTILS_AVAILABLE = False
try:
    # Import specific API call helpers and parsers
    from api_utils import (
        format_api_relationship_path,
        parse_ancestry_person_details,
        call_suggest_api,
        call_facts_user_api,
        call_getladder_api,
        call_discovery_relationship_api,
        call_treesui_list_api,
    )

    logger.info("Successfully imported required functions from api_utils.")
    API_UTILS_AVAILABLE = all(
        callable(f)
        for f in [
            format_api_relationship_path,
            parse_ancestry_person_details,
            call_suggest_api,
            call_facts_user_api,
            call_getladder_api,
            call_discovery_relationship_api,
            call_treesui_list_api,
        ]
    )
    if not API_UTILS_AVAILABLE:
        logger.error(
            "One or more required functions from api_utils are missing or not callable."
        )
except ImportError as e:
    logger.error(f"Failed to import from api_utils: {e}.", exc_info=True)

# --- Import General Utilities ---
SessionManager = None
_api_req = None  # No longer directly needed here, but keep check
format_name = None
ordinal_case = None
CORE_UTILS_AVAILABLE = False
try:
    # Only import SessionManager, format_name, ordinal_case from utils now
    from utils import SessionManager, format_name, ordinal_case

    logger.info("Successfully imported required components from utils.")
    CORE_UTILS_AVAILABLE = all(
        callable(f) for f in [SessionManager, format_name, ordinal_case]
    )
    if not CORE_UTILS_AVAILABLE:
        logger.error("One or more required core utils are missing or not callable.")
except ImportError as e:
    logger.critical(
        f"Failed to import critical components from 'utils' module: {e}. Cannot run.",
        exc_info=True,
    )
    print(f"FATAL ERROR: Failed to import required functions from utils.py: {e}")
    sys.exit(1)

# --- Session Manager Instance ---
session_manager: Optional[SessionManager] = None
if SessionManager:
    session_manager = SessionManager()
else:
    logger.critical("SessionManager class not loaded. Cannot proceed.")
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

    if not isinstance(person_facts, list) or not callable(_parse_date):
        logger.debug(
            f"_extract_fact_data: Invalid input or _parse_date unavailable for {fact_type_str}."
        )
        return date_str, place_str, date_obj  # Return defaults

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
                if year:
                    try:
                        temp_date = str(year)
                        if month:
                            temp_date += f"-{str(month).zfill(2)}"
                        if day:
                            temp_date += f"-{str(day).zfill(2)}"
                        date_obj = _parse_date(temp_date)
                        logger.debug(
                            f"_extract_fact_data: Parsed date object from ParsedDate: {date_obj}"
                        )
                    except (ValueError, TypeError) as dt_err:
                        logger.warning(
                            f"_extract_fact_data: Could not parse date from ParsedDate {parsed_date_data}: {dt_err}"
                        )
                        date_obj = None

            # Fallback to parsing the Date string if ParsedDate didn't yield an object
            if date_obj is None and date_str:
                logger.debug(
                    f"_extract_fact_data: Attempting to parse date_str '{date_str}' as fallback."
                )
                try:
                    date_obj = _parse_date(date_str)
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
    logger.info("--- Getting search criteria from user ---")
    print("\n--- Enter Search Criteria (Press Enter to skip optional fields) ---")

    first_name = input("  First Name Contains: ").strip()
    surname = input("  Last Name Contains: ").strip()
    dob_str = input("  Birth Year (YYYY): ").strip()
    pob = input("  Birth Place Contains: ").strip()
    dod_str = input("  Death Year (YYYY): ").strip() or None
    pod = input("  Death Place Contains: ").strip() or None
    gender_input = input("  Gender (M/F): ").strip().upper()

    gender = None
    if gender_input and gender_input[0] in ["M", "F"]:
        gender = gender_input[0].lower()  # Store as lowercase 'm' or 'f'

    logger.info(f"  Input - First Name: '{first_name}'")
    logger.info(f"  Input - Surname: '{surname}'")
    logger.info(f"  Input - Birth Date/Year: '{dob_str}'")
    logger.info(f"  Input - Birth Place: '{pob}'")
    logger.info(f"  Input - Death Date/Year: '{dod_str}'")
    logger.info(f"  Input - Death Place: '{pod}'")
    logger.info(f"  Input - Gender: '{gender}'")

    if not (first_name or surname):
        logger.warning("API search needs First Name or Surname. Report cancelled.")
        print("\nAPI search needs First Name or Surname. Report cancelled.")
        return None

    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)

    target_birth_year: Optional[int] = None
    target_birth_date_obj: Optional[datetime] = None
    if dob_str and callable(_parse_date):
        target_birth_date_obj = _parse_date(dob_str)
        if target_birth_date_obj:
            target_birth_year = target_birth_date_obj.year

    target_death_year: Optional[int] = None
    target_death_date_obj: Optional[datetime] = None
    if dod_str and callable(_parse_date):
        target_death_date_obj = _parse_date(dod_str)
        if target_death_date_obj:
            target_death_year = target_death_date_obj.year

    search_criteria_dict = {
        "first_name_raw": first_name,  # Keep raw for API params
        "surname_raw": surname,  # Keep raw for API params
        "first_name": clean_param(first_name),  # Cleaned for scoring
        "surname": clean_param(surname),  # Cleaned for scoring
        "birth_year": target_birth_year,
        "birth_date_obj": target_birth_date_obj,
        "birth_place": clean_param(pob),
        "death_year": target_death_year,
        "death_date_obj": target_death_date_obj,
        "death_place": clean_param(pod),
        "gender": gender,
    }

    logger.info("\n--- Final Search Criteria Prepared ---")
    for key, value in search_criteria_dict.items():
        if value is not None and key not in [
            "first_name_raw",
            "surname_raw",
            "birth_date_obj",
            "death_date_obj",
        ]:
            logger.info(f"  {key.replace('_', ' ').title()}: '{value}'")
        elif key == "death_place" and value is None:
            logger.info(f"  {key.replace('_', ' ').title()}: None")

    return search_criteria_dict


# End of _get_search_criteria


# Simple scoring remains local as it's a fallback specific to this action's workflow
def _run_simple_suggestion_scoring(
    search_criteria: Dict[str, Any], candidate_data_dict: Dict[str, Any]
) -> Tuple[float, Dict, List[str]]:
    """Performs simple fallback scoring based on hardcoded rules."""
    logger.warning("Using simple fallback scoring for suggestion.")
    score = 0.0
    field_scores = {
        "givn": 0,
        "surn": 0,
        "gender": 0,
        "byear": 0,
        "bdate": 0,
        "bplace": 0,
        "dyear": 0,
        "ddate": 0,
        "dplace": 0,
        "bonus": 0,
    }
    reasons = ["API Suggest Match", "Fallback Scoring"]

    cand_fn = candidate_data_dict.get("first_name")
    cand_sn = candidate_data_dict.get("surname")
    cand_by = candidate_data_dict.get("birth_year")
    cand_gn = candidate_data_dict.get("gender")  # Should be 'm'/'f'/None
    search_fn = search_criteria.get("first_name")
    search_sn = search_criteria.get("surname")
    search_by = search_criteria.get("birth_year")
    search_gn = search_criteria.get("gender")  # Should be 'm'/'f'/None

    if cand_fn and search_fn and search_fn in cand_fn:
        score += 25
        field_scores["givn"] = 25
        reasons.append("Contains First Name (25pts)")
    if cand_sn and search_sn and search_sn in cand_sn:
        score += 25
        field_scores["surn"] = 25
        reasons.append("Contains Surname (25pts)")
    if field_scores["givn"] > 0 and field_scores["surn"] > 0:
        score += 25
        field_scores["bonus"] = 25
        reasons.append("Bonus Both Names (25pts)")
    if cand_by and search_by and cand_by == search_by:
        score += 20
        field_scores["byear"] = 20
        reasons.append(f"Exact Birth Year ({cand_by}) (20pts)")
    cand_dy = candidate_data_dict.get("death_year")
    search_dy = search_criteria.get("death_year")
    if cand_dy and search_dy and cand_dy == search_dy:
        score += 15
        field_scores["dyear"] = 15
        reasons.append(f"Exact Death Year ({cand_dy}) (15pts)")
    is_living = candidate_data_dict.get("is_living")
    if not search_dy and not cand_dy and is_living in [False, None]:
        score += 15
        field_scores["ddate"] = 15
        reasons.append("Death Dates Absent (15pts)")
    if cand_gn and search_gn and cand_gn == search_gn:
        score += 25
        field_scores["gender"] = 25
        reasons.append(f"Gender Match ({cand_gn.upper()}) (25pts)")

    return score, field_scores, reasons


# End of _run_simple_suggestion_scoring


def _process_and_score_suggestions(
    suggestions: List[Dict], search_criteria: Dict[str, Any], config_instance_local: Any
) -> List[Dict]:
    """Processes raw API suggestions, extracts data, calculates match scores, returns sorted list."""
    processed_candidates = []
    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)
    logger.info(f"\n--- Filtering and Scoring {len(suggestions)} Candidates ---")

    scoring_func = (
        calculate_match_score
        if GEDCOM_SCORING_AVAILABLE
        else _run_simple_suggestion_scoring
    )
    if not GEDCOM_SCORING_AVAILABLE:
        logger.warning(
            "Gedcom scoring function unavailable. Using simple fallback scoring."
        )

    scoring_weights = getattr(config_instance_local, "COMMON_SCORING_WEIGHTS", {})
    name_flex = getattr(config_instance_local, "NAME_FLEXIBILITY", 2)
    date_flex = getattr(config_instance_local, "DATE_FLEXIBILITY", 2)

    for idx, raw_candidate in enumerate(suggestions):
        if not isinstance(raw_candidate, dict):
            logger.warning(
                f"Skipping invalid suggestion entry (not a dict): {raw_candidate}"
            )
            continue

        # Use api_utils parser to get standardized basic info from suggestion
        # Pass raw_candidate as both person_card and facts_data (parser handles overlaps)
        parsed_suggestion = parse_ancestry_person_details(raw_candidate, raw_candidate)

        # Prepare Candidate Data for Scoring using parsed info
        candidate_data_dict = {
            "first_name": clean_param(
                parsed_suggestion["name"].split()[0]
                if parsed_suggestion["name"] != "Unknown"
                else None
            ),  # Crude split
            "surname": clean_param(
                parsed_suggestion["name"].split()[-1]
                if parsed_suggestion["name"] != "Unknown"
                and len(parsed_suggestion["name"].split()) > 1
                else None
            ),  # Crude split
            "birth_year": (
                parsed_suggestion["api_birth_obj"].year
                if parsed_suggestion.get("api_birth_obj")
                else None
            ),
            "birth_date_obj": parsed_suggestion.get("api_birth_obj"),
            "birth_place": clean_param(parsed_suggestion.get("birth_place")),
            "death_year": (
                parsed_suggestion["api_death_obj"].year
                if parsed_suggestion.get("api_death_obj")
                else None
            ),
            "death_date_obj": parsed_suggestion.get("api_death_obj"),
            "death_place": clean_param(parsed_suggestion.get("death_place")),
            "gender": parsed_suggestion.get("gender"),  # Already 'm'/'f'/None
            "is_living": parsed_suggestion.get("is_living"),
            "norm_id": parsed_suggestion.get("person_id", f"Unknown_{idx}"),
            "display_id": parsed_suggestion.get("person_id", f"Unknown_{idx}"),
            "full_name_disp": parsed_suggestion.get("name", "Unknown"),
            "gender_norm": parsed_suggestion.get(
                "gender"
            ),  # Redundant but keep for compatibility
            "birth_place_disp": clean_param(
                parsed_suggestion.get("birth_place")
            ),  # Redundant but keep for compatibility
            "death_place_disp": clean_param(
                parsed_suggestion.get("death_place")
            ),  # Redundant but keep for compatibility
        }

        # Calculate Score
        score = 0.0
        field_scores = {}
        reasons = []
        try:
            if scoring_func == calculate_match_score:
                score, field_scores, reasons = scoring_func(
                    search_criteria,
                    candidate_data_dict,
                    scoring_weights,
                    name_flex,
                    date_flex,
                )
            else:  # Simple scoring
                score, field_scores, reasons = scoring_func(
                    search_criteria, candidate_data_dict
                )
        except Exception as score_err:
            logger.error(
                f"Error calculating score for suggestion {candidate_data_dict['display_id']}: {score_err}",
                exc_info=True,
            )
            score, field_scores, reasons = _run_simple_suggestion_scoring(
                search_criteria, candidate_data_dict
            )
            reasons.append("(Error Fallback)")

        # Append Processed Candidate
        processed_candidates.append(
            {
                "id": candidate_data_dict["display_id"],
                "name": parsed_suggestion.get("name", "Unknown"),
                "gender": candidate_data_dict.get(
                    "gender"
                ),  # Use 'm'/'f' for consistency
                "birth_date": parsed_suggestion.get(
                    "birth_date", "N/A"
                ),  # Display date
                "birth_place": parsed_suggestion.get("birth_place", "N/A"),
                "death_date": parsed_suggestion.get(
                    "death_date", "N/A"
                ),  # Display date
                "death_place": parsed_suggestion.get("death_place", "N/A"),
                "score": score,
                "field_scores": field_scores,
                "reasons": reasons,
                "raw_data": raw_candidate,  # Keep original raw data
                "parsed_suggestion": parsed_suggestion,  # Keep parsed data for details display fallback
            }
        )

    processed_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    logger.info(f"Scored {len(processed_candidates)} candidates.")
    return processed_candidates


# End of _process_and_score_suggestions


def _display_search_results(candidates: List[Dict], max_to_display: int):
    """Displays the scored search results in a formatted table."""
    if not candidates:
        print("\nNo candidates to display.")
        return

    display_count = min(len(candidates), max_to_display)
    print(f"\n=== SEARCH RESULTS (Top {display_count} Matches) ===")

    table_data = []
    headers = [
        "ID",
        "Name [Score]",
        "Gender [Score]",
        "Birth [Score]",
        "B.Place [Score]",
        "Death [Score]",
        "D.Place [Score]",
        "Total",
    ]

    for candidate in candidates[:display_count]:
        fs = candidate.get("field_scores", {})
        givn_s, surn_s, bonus_s = (
            fs.get("givn", 0),
            fs.get("surn", 0),
            fs.get("bonus", 0),
        )
        gender_s = fs.get("gender", 0)
        byear_s, bdate_s, bplace_s = (
            fs.get("byear", 0),
            fs.get("bdate", 0),
            fs.get("bplace", 0),
        )
        dyear_s, ddate_s, dplace_s = (
            fs.get("dyear", 0),
            fs.get("ddate", 0),
            fs.get("dplace", 0),
        )

        name_disp = candidate.get("name", "N/A")[:25] + (
            "..." if len(candidate.get("name", "")) > 25 else ""
        )
        name_score_str = f"[{givn_s+surn_s}]" + (f"[+{bonus_s}]" if bonus_s > 0 else "")
        name_with_score = f"{name_disp} {name_score_str}"

        gender_disp = str(candidate.get("gender", "N/A")).upper()  # 'm'/'f' -> 'M'/'F'
        gender_with_score = f"{gender_disp} [{gender_s}]"

        bdate_disp = str(candidate.get("birth_date", "N/A"))
        birth_score_display = (
            f"[{byear_s+bdate_s}]"
            if byear_s != bdate_s and bdate_s != 0
            else f"[{byear_s}]"
        )  # Show combined if different
        bdate_with_score = f"{bdate_disp} {birth_score_display}"

        bplace_disp = str(candidate.get("birth_place", "N/A"))
        bplace_disp = bplace_disp[:15] + ("..." if len(bplace_disp) > 15 else "")
        bplace_with_score = f"{bplace_disp} [{bplace_s}]"

        ddate_disp = str(candidate.get("death_date", "N/A"))
        death_score_display = (
            f"[{dyear_s+ddate_s}]"
            if dyear_s != ddate_s and ddate_s != 0
            else f"[{dyear_s}]"
        )  # Show combined if different
        ddate_with_score = f"{ddate_disp} {death_score_display}"

        dplace_disp = str(candidate.get("death_place", "N/A"))
        dplace_disp = dplace_disp[:15] + ("..." if len(dplace_disp) > 15 else "")
        dplace_with_score = f"{dplace_disp} [{dplace_s}]"

        table_data.append(
            [
                str(candidate.get("id", "N/A")),
                name_with_score,
                gender_with_score,
                bdate_with_score,
                bplace_with_score,
                ddate_with_score,
                dplace_with_score,
                f"{candidate.get('score', 0):.0f}",
            ]
        )
        logger.info(
            f"  ID: {candidate.get('id', 'N/A')}, Name: {candidate.get('name', 'N/A')}, Score: {candidate.get('score', 0):.0f}"
        )
        logger.debug(f"    Field Scores: {candidate.get('field_scores', {})}")
        logger.debug(f"    Reasons: {candidate.get('reasons', [])}")

    try:
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    except Exception as tab_err:
        logger.error(f"Error formatting results table with tabulate: {tab_err}")
        print("\nSearch Results (Fallback Format):")
        print(" | ".join(headers))
        print("-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))
        for row in table_data:
            print(" | ".join(map(str, row)))
    print("")


# End of _display_search_results


def _select_top_candidate(
    scored_candidates: List[Dict],
    raw_suggestions: List[Dict],  # raw_suggestions may not be needed now
) -> Optional[Tuple[Dict, Dict]]:
    """Selects the highest-scoring candidate and retrieves its original raw suggestion data."""
    if not scored_candidates:
        logger.info("No scored candidates available to select from.")
        return None

    top_scored_candidate = scored_candidates[0]
    top_scored_id = top_scored_candidate.get("id")
    top_candidate_raw = top_scored_candidate.get(
        "raw_data"
    )  # Already stored during scoring

    if not top_candidate_raw or not isinstance(top_candidate_raw, dict):
        logger.error(
            f"Critical Error: Raw data missing for top scored candidate ID: {top_scored_id}."
        )
        print(
            f"\nError: Internal mismatch finding details for top candidate ({top_scored_id})."
        )
        return None

    # Handle case where ID was generated
    if isinstance(top_scored_id, str) and top_scored_id.startswith("Unknown_"):
        logger.warning(
            f"Top candidate has generated ID '{top_scored_id}'. Using stored raw data."
        )

    logger.info(
        f"Highest scoring candidate selected: ID {top_scored_id}, Score {top_scored_candidate.get('score', 0):.0f}"
    )
    print(
        f"\n---> Auto-selecting top match: {top_scored_candidate.get('name', 'Unknown')}"
    )
    return top_scored_candidate, top_candidate_raw


# End of _select_top_candidate

# _fetch_person_details is removed, logic moved to api_utils.call_facts_user_api


def _extract_best_name_from_details(
    person_research_data: Dict, candidate_raw: Dict
) -> str:
    """Extracts the best available name from multiple potential sources in API details."""
    # This function now primarily uses fields standard in person_research_data
    # Fallback to candidate_raw fields if primary ones are missing

    best_name = "Unknown"
    logger.debug(f"_extract_best_name: Input keys: {list(person_research_data.keys())}")

    # Priority 1: PersonFullName
    person_full_name = person_research_data.get("PersonFullName")
    if person_full_name and person_full_name != "Valued Relative":
        best_name = person_full_name
        logger.debug(f"Using PersonFullName: '{best_name}'")

    # Priority 2: Name Fact
    if best_name == "Unknown":
        person_facts_list = person_research_data.get("PersonFacts", [])
        if isinstance(person_facts_list, list):
            name_fact = next(
                (
                    f
                    for f in person_facts_list
                    if isinstance(f, dict) and f.get("TypeString") == "Name"
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

    # Priority 3: Constructed from FirstName/LastName
    if best_name == "Unknown":
        first_name_comp = person_research_data.get("FirstName", "")
        last_name_comp = person_research_data.get("LastName", "")
        if first_name_comp or last_name_comp:
            constructed_name = f"{first_name_comp} {last_name_comp}".strip()
            if constructed_name:
                best_name = constructed_name
                logger.debug(f"Using Constructed Name: '{best_name}'")

    # Fallback to name from original suggestion if still Unknown
    if best_name == "Unknown":
        cand_name = candidate_raw.get("FullName", candidate_raw.get("Name"))
        if cand_name:
            best_name = cand_name
            logger.debug(f"Using Fallback Name from Suggestion: '{best_name}'")

    # Final formatting and validation
    if not best_name or best_name == "Valued Relative":
        best_name = "Unknown"
    elif callable(format_name):
        best_name = format_name(best_name)

    return best_name


# End of _extract_best_name_from_details


def _extract_detailed_info(person_research_data: Dict, candidate_raw: Dict) -> Dict:
    """Extracts detailed information from the 'personResearch' dictionary."""
    # Relies on _extract_fact_data helper for birth/death
    # Relies on _extract_best_name_from_details for name

    extracted = {}
    logger.debug("Extracting details from person_research_data...")

    if not isinstance(person_research_data, dict):
        logger.error("Invalid input to _extract_detailed_info.")
        return {}

    person_facts_list = person_research_data.get("PersonFacts", [])
    if not isinstance(person_facts_list, list):
        person_facts_list = []
    logger.debug(f"Found {len(person_facts_list)} items in PersonFacts.")

    # --- Name ---
    best_name = _extract_best_name_from_details(person_research_data, candidate_raw)
    extracted["name"] = best_name
    logger.info(f"Final Extracted Name: {best_name}")

    # --- Gender ---
    gender_str = person_research_data.get("PersonGender")
    if not gender_str:  # Check fact if main field missing
        gender_fact = next(
            (
                f
                for f in person_facts_list
                if isinstance(f, dict) and f.get("TypeString") == "Gender"
            ),
            None,
        )
        if gender_fact and gender_fact.get("Value"):
            gender_str = gender_fact.get("Value")
    extracted["gender_str"] = gender_str
    extracted["gender"] = (
        "m" if gender_str == "Male" else ("f" if gender_str == "Female" else None)
    )
    logger.info(f"Final Extracted Gender: {extracted['gender']} (from '{gender_str}')")

    # --- Living Status ---
    extracted["is_living"] = person_research_data.get(
        "IsPersonLiving", False
    )  # Boolean expected
    logger.info(f"Is Living: {extracted['is_living']}")

    # --- Birth/Death using helper ---
    logger.debug("Extracting birth info...")
    birth_date_str, birth_place, birth_date_obj = _extract_fact_data(
        person_facts_list, "Birth"
    )
    logger.debug("Extracting death info...")
    death_date_str, death_place, death_date_obj = _extract_fact_data(
        person_facts_list, "Death"
    )

    extracted["birth_date_str"] = birth_date_str
    extracted["birth_place"] = birth_place
    extracted["birth_date_obj"] = birth_date_obj
    extracted["birth_year"] = birth_date_obj.year if birth_date_obj else None
    extracted["birth_date_disp"] = (
        _clean_display_date(birth_date_str)
        if birth_date_str and callable(_clean_display_date)
        else "N/A"
    )
    logger.info(
        f"Birth: Date='{extracted['birth_date_disp']}', Place='{birth_place or 'N/A'}'"
    )

    extracted["death_date_str"] = death_date_str
    extracted["death_place"] = death_place
    extracted["death_date_obj"] = death_date_obj
    extracted["death_year"] = death_date_obj.year if death_date_obj else None
    extracted["death_date_disp"] = (
        _clean_display_date(death_date_str)
        if death_date_str and callable(_clean_display_date)
        else "N/A"
    )
    logger.info(
        f"Death: Date='{extracted['death_date_disp']}', Place='{death_place or 'N/A'}'"
    )

    # --- Family Data ---
    extracted["family_data"] = person_research_data.get("PersonFamily", {})
    if not isinstance(extracted["family_data"], dict):
        logger.warning("PersonFamily data not dict/missing.")
        extracted["family_data"] = {}
    logger.info(f"Family Data Keys: {list(extracted['family_data'].keys())}")

    # --- IDs ---
    extracted["person_id"] = person_research_data.get(
        "PersonId", candidate_raw.get("PersonId")
    )
    extracted["tree_id"] = person_research_data.get(
        "TreeId", candidate_raw.get("TreeId")
    )
    extracted["user_id"] = person_research_data.get(
        "UserId", candidate_raw.get("UserId")
    )  # Global ID
    logger.info(
        f"IDs: PersonId='{extracted['person_id']}', TreeId='{extracted['tree_id']}', UserId='{extracted['user_id']}'"
    )

    # --- Name Components for scoring ---
    extracted["first_name"] = person_research_data.get(
        "FirstName"
    )  # Use direct fields if available
    extracted["surname"] = person_research_data.get("LastName")
    if not extracted["first_name"] and best_name != "Unknown":  # Fallback crude split
        parts = best_name.split()
        if parts:
            extracted["first_name"] = parts[0]
        if len(parts) > 1:
            extracted["surname"] = parts[-1]
    logger.debug(
        f"Extracted name components for scoring: First='{extracted['first_name']}', Sur='{extracted['surname']}'"
    )

    return extracted


# End of _extract_detailed_info


def _score_detailed_match(
    extracted_info: Dict, search_criteria: Dict[str, Any], config_instance_local: Any
) -> Tuple[float, Dict, List[str]]:
    """Calculates final match score based on detailed info using scoring function or fallback."""
    scoring_func = (
        calculate_match_score
        if GEDCOM_SCORING_AVAILABLE
        else _run_simple_suggestion_scoring
    )
    if not GEDCOM_SCORING_AVAILABLE:
        logger.warning(
            "Gedcom scoring unavailable for detailed match. Using simple fallback."
        )

    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)
    scoring_weights = getattr(config_instance_local, "COMMON_SCORING_WEIGHTS", {})
    name_flex = getattr(config_instance_local, "NAME_FLEXIBILITY", 2)
    date_flex = getattr(config_instance_local, "DATE_FLEXIBILITY", 2)

    # Prepare data structure compatible with scoring function
    candidate_processed_data = {
        "norm_id": extracted_info.get("person_id"),
        "display_id": extracted_info.get("person_id"),
        "first_name": clean_param(extracted_info.get("first_name")),
        "surname": clean_param(extracted_info.get("surname")),
        "full_name_disp": extracted_info.get("name"),
        "gender_norm": extracted_info.get("gender"),  # Already 'm'/'f'/None
        "birth_year": extracted_info.get("birth_year"),
        "birth_date_obj": extracted_info.get("birth_date_obj"),
        "birth_place_disp": clean_param(extracted_info.get("birth_place")),
        "death_year": extracted_info.get("death_year"),
        "death_date_obj": extracted_info.get("death_date_obj"),
        "death_place_disp": clean_param(extracted_info.get("death_place")),
        "is_living": extracted_info.get("is_living", False),  # Needs to be bool
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
            f"Calculating detailed score using {'gedcom_utils' if scoring_func == calculate_match_score else 'simple fallback'}..."
        )
        if scoring_func == calculate_match_score:
            score, field_scores, reasons = scoring_func(
                search_criteria,
                candidate_processed_data,
                scoring_weights,
                name_flex,
                date_flex,
            )
        else:  # Simple scoring
            score, field_scores, reasons = scoring_func(
                search_criteria, candidate_processed_data
            )
        # Ensure base reason is present and combine
        if "API Detail Match" not in reasons:
            reasons.insert(0, "API Detail Match")
        reasons_list = reasons
        logger.info(f"Calculated detailed score: {score:.0f}")
    except Exception as e:
        logger.error(f"Error calculating detailed score: {e}", exc_info=True)
        logger.warning(
            "Falling back to simple scoring for detailed match due to error."
        )
        score, field_scores, reasons_list = _run_simple_suggestion_scoring(
            search_criteria, candidate_processed_data
        )  # Use extracted_info for simple scoring
        reasons_list.append("(Detailed Scoring Error Fallback)")

    logger.debug(f"Final detailed score: {score:.0f}")
    logger.debug(f"Final detailed field scores: {field_scores}")
    logger.debug(f"Final detailed reasons: {reasons_list}")
    return score, field_scores, reasons_list


# End of _score_detailed_match


def _display_detailed_match_info(
    extracted_info: Dict,
    score_info: Tuple[float, Dict, List[str]],
    search_criteria: Dict[str, Any],
    base_url: str,
):
    """Displays the final scoring details, comparison, and selected match summary."""
    score, field_scores, reasons = score_info
    print("\n=== DETAILED SCORING INFORMATION ===")
    print(f"Total Score: {score:.0f}")
    print("\nField-by-Field Comparison:")

    na = "N/A"
    sc = search_criteria
    ei = extracted_info
    # Prepare comparison dict using cleaned search criteria and extracted info
    cp = {  # Use extracted data directly
        "First Name": str(ei.get("first_name", na)),
        "Last Name": str(ei.get("surname", na)),
        "Gender": str(ei.get("gender", na)).upper(),
        "Birth Year": str(ei.get("birth_year", na)),
        "Birth Place": str(ei.get("birth_place", na)),
        "Death Year": str(ei.get("death_year", na)),
        "Death Place": str(ei.get("death_place", na)),
    }
    sc_comp = {  # Use cleaned search criteria
        "First Name": str(sc.get("first_name", na)),
        "Last Name": str(sc.get("surname", na)),
        "Gender": str(sc.get("gender", na)).upper(),
        "Birth Year": str(sc.get("birth_year", na)),
        "Birth Place": str(sc.get("birth_place", na)),
        "Death Year": str(sc.get("death_year", na)),
        "Death Place": str(sc.get("death_place", na)),
    }

    field_order = [
        "First Name",
        "Last Name",
        "Gender",
        "Birth Year",
        "Birth Place",
        "Death Year",
        "Death Place",
    ]
    max_len = max(len(f) for f in field_order)
    for field in field_order:
        print(f"  {field:<{max_len}} : {sc_comp[field]:<15} vs {cp[field]}")

    logger.debug("  Detailed Field Scores:")
    for field, score_value in field_scores.items():
        logger.debug(f"    {field}: {score_value}")

    print("\nScore Reasons:")
    for reason in reasons:
        print(f"  - {reason}")

    print(f"\n--- Top Match Details ---")
    person_link = "(Link unavailable)"
    # Generate link using api_utils helper (handles global vs tree ID)
    link_func = getattr(sys.modules.get("api_utils"), "_generate_person_link", None)
    if callable(link_func):
        person_link = link_func(
            ei.get("user_id") or ei.get("person_id"),
            ei.get("tree_id") if not ei.get("user_id") else None,
            base_url,
        )
    else:
        logger.error("Cannot generate link: _generate_person_link helper missing.")

    print(
        f"  Name : {ei.get('name', 'Unknown')} (ID: {ei.get('user_id') or ei.get('person_id') or 'N/A'})"
    )
    print(f"  Link : {person_link}")
    print(
        f"  Born : {ei.get('birth_date_disp', '?')} in {ei.get('birth_place') or '?'}"
    )
    if not ei.get("is_living", False):
        print(
            f"  Died : {ei.get('death_date_disp', '?')} in {ei.get('death_place') or '?'}"
        )
    else:
        print(f"  Died : (Living)")
    reason_summary = (
        reasons[0] + ("..." if len(reasons) > 1 else "") if reasons else "N/A"
    )
    print(f"  Score: {score:.0f} (Reason: {reason_summary})")


# End of _display_detailed_match_info


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
                    f"_flatten_children_list: Unexpected item type in Children list: {type(child_entry)}"
                )

            for child_dict in items_to_process:
                if isinstance(child_dict, dict):
                    child_id = child_dict.get("PersonId")
                    if child_id and child_id not in added_ids:
                        children_flat_list.append(child_dict)
                        added_ids.add(child_id)
                    elif not child_id:
                        logger.warning(
                            "_flatten_children_list: Child dict missing PersonId."
                        )
                        children_flat_list.append(child_dict)  # Add anyway?
                else:
                    logger.warning(
                        f"_flatten_children_list: Non-dict item found within child entry: {type(child_dict)}"
                    )
    elif isinstance(
        children_raw, dict
    ):  # Handle case where 'Children' might be a single dict
        child_id = children_raw.get("PersonId")
        if child_id and child_id not in added_ids:
            children_flat_list.append(children_raw)
            added_ids.add(child_id)
        elif not child_id:
            logger.warning(
                "_flatten_children_list: Single child dict missing PersonId."
            )
            children_flat_list.append(children_raw)
    elif children_raw is not None:
        logger.warning(
            f"_flatten_children_list: Unexpected data type for 'Children': {type(children_raw)}"
        )

    logger.debug(
        f"Flattened children entries into {len(children_flat_list)} unique children."
    )
    return children_flat_list


# End of _flatten_children_list


def _display_family_info(family_data: Dict):
    """Displays formatted family information (parents, siblings, spouses, children)."""
    print("\nRelatives:")
    logger.info("\n  Relatives:")
    if not isinstance(family_data, dict) or not family_data:
        logger.warning("_display_family_info: Received empty or invalid family_data.")
        print("  Family data unavailable.")
        return

    # Use consistent name formatting
    name_formatter = format_name if callable(format_name) else lambda x: str(x).title()

    def print_relatives(rel_type: str, rel_list: Optional[List[Dict]]):
        type_display = rel_type.replace("_", " ").capitalize()
        print(f"  {type_display}:")
        logger.info(f"    {type_display}:")
        if not rel_list:
            print("    None found.")
            logger.info("    None found.")
            return
        if not isinstance(rel_list, list):
            logger.warning(
                f"Expected list for {rel_type}, but got {type(rel_list)}. Skipping."
            )
            print("    (Data format error)")
            return

        found_any = False
        for idx, relative in enumerate(rel_list):
            if not isinstance(relative, dict):
                logger.warning(
                    f"Skipping invalid relative entry {idx+1} in {rel_type}: {relative}"
                )
                continue

            name = name_formatter(relative.get("FullName", "Unknown"))
            lifespan = relative.get("LifeRange", "")  # e.g., "1900-1950" or "1920"
            # Try to extract just years for display
            b_year, d_year = None, None
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
            elif lifespan:
                life_info = f" ({lifespan})"  # Fallback to full string

            rel_info = f"- {name}{life_info}"
            print(f"    {rel_info}")
            logger.info(f"      {rel_info}")
            found_any = True
        if not found_any:
            print("    None found.")
            logger.info("    None found (list contained invalid entries).")

    # End of print_relatives (nested function)

    parents_list = (family_data.get("Fathers") or []) + (
        family_data.get("Mothers") or []
    )
    siblings_list = family_data.get("Siblings")
    spouses_list = family_data.get("Spouses")
    children_raw = family_data.get("Children", [])
    children_flat_list = _flatten_children_list(children_raw)

    print_relatives("Parents", parents_list)
    print_relatives("Siblings", siblings_list)
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
    """Calculates and displays the relationship path using the Tree Ladder API (/getladder) via api_utils helper."""
    print(f"\n--- Relationship Path (within Tree) to {owner_name} ---")
    logger.info(
        f"Calculating Tree relationship path for {selected_name} (PersonID: {selected_person_tree_id}) to {owner_name} (TreeID: {owner_tree_id})"
    )

    if not callable(call_getladder_api) or not callable(format_api_relationship_path):
        logger.error(
            "Cannot display tree relationship: Required api_utils functions missing."
        )
        print("(Error: Required relationship utilities unavailable)")
        return

    # Call the API helper
    relationship_data_raw = call_getladder_api(
        session_manager_local, owner_tree_id, selected_person_tree_id, base_url
    )

    if not relationship_data_raw:
        logger.warning("call_getladder_api returned no data.")
        print("(Tree API call for relationship returned no response or failed)")
        return

    print("")  # Add space before output
    fallback_message_text = "(Could not parse relationship path from Tree API)"

    try:
        formatted_path = format_api_relationship_path(
            relationship_data_raw, owner_name, selected_name
        )
        # Check for known error/empty states from the formatter
        known_error_starts = (
            "(No relationship",
            "(Could not parse",
            "(API returned error",
            "(Relationship HTML structure",
            "(Unsupported API response",
            "(Error processing relationship",
        )
        if formatted_path and not formatted_path.startswith(known_error_starts):
            print(
                formatted_path
            )  # Assumes formatter includes necessary indentation/newlines
            logger.info("    --- Tree Relationship Path Interpretation ---")
            for line in formatted_path.splitlines():
                if line.strip():
                    logger.info(f"    {line.strip()}")
            logger.info("    ------------------------------------")
        else:
            logger.warning(
                f"format_api_relationship_path returned no path or error: '{formatted_path}'"
            )
            print(
                f"  {formatted_path or fallback_message_text}"
            )  # Display error from parser or fallback
            logger.warning(
                f"Relationship parsing failed/returned error. Raw response:\n{str(relationship_data_raw)[:1000]}"
            )
    except Exception as fmt_err:
        logger.error(
            f"Error calling format_api_relationship_path: {fmt_err}", exc_info=True
        )
        print(f"  {fallback_message_text} (Processing Error)")
        logger.warning(
            f"Relationship parsing failed during format call. Raw response:\n{str(relationship_data_raw)[:1000]}"
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
    """Calculates and displays the relationship path using the Discovery API (/relationshiptome) via api_utils helper."""
    print(f"\n--- Relationship Path (Discovery) to {owner_name} ---")
    logger.info(
        f"Calculating Discovery relationship path for {selected_name} (GlobalID: {selected_person_global_id}) to {owner_name} (ProfileID: {owner_profile_id})"
    )

    if not callable(call_discovery_relationship_api):
        logger.error(
            "Cannot display discovery relationship: call_discovery_relationship_api function missing."
        )
        print("(Error: Required relationship utility unavailable)")
        return

    # Call the API helper
    relationship_data = call_discovery_relationship_api(
        session_manager_local, selected_person_global_id, owner_profile_id, base_url
    )

    if not relationship_data:
        logger.warning("call_discovery_relationship_api returned no data.")
        print("(Discovery API call for relationship returned no response or failed)")
        return
    if not isinstance(relationship_data, dict):
        logger.warning(
            f"Discovery API call returned unexpected type: {type(relationship_data)}"
        )
        print("(Discovery API call returned data in unexpected format)")
        logger.debug(f"Raw Discovery response: {str(relationship_data)[:1000]}")
        return

    print("")  # Add space before output
    fallback_message_text = "(Could not parse relationship path from Discovery API)"

    # Direct JSON Parsing for 'path' array
    if isinstance(relationship_data.get("path"), list) and relationship_data.get(
        "path"
    ):
        logger.info(
            "    --- Discovery Relationship Path Interpretation (Direct JSON) ---"
        )
        path_steps = relationship_data["path"]
        print(f"  {selected_name}")  # Start with target
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
            display_rel = step_rel.capitalize()  # Default
            if callable(rel_term_func):
                display_rel = rel_term_func(
                    None, step_rel
                )  # Get specific term if possible

            display_line = f"  -> {display_rel} is {name_formatter(step_name)}"
            print(display_line)
            logger.info(f"    {display_line.strip()}")
        print(f"  -> {owner_name} (Tree Owner / You)")
        logger.info(f"    -> {owner_name} (Tree Owner / You)")
        logger.info("    ------------------------------------")
    else:
        # Log failure reason if path missing or invalid
        if "path" not in relationship_data:
            logger.warning("Discovery API response JSON missing 'path' key.")
            print(f"  {fallback_message_text} (Missing 'path')")
        elif not isinstance(relationship_data.get("path"), list):
            logger.warning(
                f"Discovery API response 'path' key is not a list: {type(relationship_data.get('path'))}"
            )
            print(f"  {fallback_message_text} ('path' not a list)")
        else:  # Path is likely an empty list
            logger.warning("Discovery API response 'path' list is empty.")
            print(
                "(No direct relationship path found via Discovery API)"
            )  # More specific message

        logger.debug(
            f"Discovery response content: {json.dumps(relationship_data, indent=2)}"
        )


# End of _display_discovery_relationship


# --- Phase Handler Functions ---


def _handle_search_phase(
    session_manager_local: SessionManager,
    search_criteria: Dict[str, Any],
    config_instance_local: Any,
) -> Optional[List[Dict]]:
    """Handles the API search phase, including fallbacks."""
    owner_tree_id = getattr(session_manager_local, "my_tree_id", None)
    owner_profile_id = getattr(session_manager_local, "my_profile_id", None)
    base_url = getattr(config_instance_local, "BASE_URL", "").rstrip("/")

    if not owner_tree_id:
        logger.error("Cannot perform API search: Owner Tree ID is missing.")
        print("\nERROR: Cannot perform API search because your Tree ID is unknown.")
        return None

    if not callable(call_suggest_api) or not callable(call_treesui_list_api):
        logger.error("Cannot perform API search: Required api_utils functions missing.")
        print("\nERROR: API search utilities are unavailable.")
        return None

    print("Searching Ancestry API...")  # Simplified message

    # Try Suggest API first
    suggestions_raw = call_suggest_api(
        session_manager_local,
        owner_tree_id,
        owner_profile_id,
        base_url,
        search_criteria,
    )

    # Fallback to TreesUI List API if Suggest failed AND birth year available
    if suggestions_raw is None and search_criteria.get("birth_year"):
        logger.warning("Suggest API failed, attempting TreesUI List API fallback...")
        print("\nTrying alternative API search method...")
        suggestions_raw = call_treesui_list_api(
            session_manager_local,
            owner_tree_id,
            owner_profile_id,
            base_url,
            search_criteria,
        )

    if suggestions_raw is None:
        logger.error("All API Search attempts failed critically.")
        print(
            "\nError during API search. No results found or API calls failed critically."
        )
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Verify Ancestry.com/.co.uk is accessible")
        print("3. Try again later (API issues)")
        print("4. Try a different search with fewer criteria")
        return None
    if not suggestions_raw:
        logger.info("API Search returned no results.")
        print("\nNo potential matches found in Ancestry API based on criteria.")
        return []  # Return empty list for no results

    # Limit suggestions to score
    max_score_limit = getattr(config_instance_local, "MAX_SUGGESTIONS_TO_SCORE", 10)
    if max_score_limit > 0 and len(suggestions_raw) > max_score_limit:
        logger.warning(
            f"Processing only top {max_score_limit} of {len(suggestions_raw)} suggestions for scoring."
        )
        return suggestions_raw[:max_score_limit]
    else:
        return suggestions_raw


# End of _handle_search_phase


def _handle_selection_phase(
    suggestions_to_score: List[Dict],
    search_criteria: Dict[str, Any],
    config_instance_local: Any,
) -> Optional[Tuple[Dict, Dict]]:
    """Handles scoring, display, and selection of the top candidate."""
    scored_candidates = _process_and_score_suggestions(
        suggestions_to_score, search_criteria, config_instance_local
    )
    if not scored_candidates:
        print("\nNo suitable candidates found after scoring.")
        logger.info("No candidates available after scoring process.")
        return None

    max_display_limit = getattr(config_instance_local, "MAX_CANDIDATES_TO_DISPLAY", 5)
    _display_search_results(scored_candidates, max_display_limit)

    selection = _select_top_candidate(
        scored_candidates, suggestions_to_score
    )  # Pass scored list for raw data retrieval
    if not selection:
        print("\nFailed to select a top candidate (check logs for errors).")
        logger.error("Failed to select top candidate.")
        return None

    return selection  # (selected_candidate_processed, selected_candidate_raw)


# End of _handle_selection_phase


def _handle_details_phase(
    selected_candidate_raw: Dict,
    session_manager_local: SessionManager,
    config_instance_local: Any,
) -> Optional[Dict]:
    """Handles fetching detailed person information using the Facts API."""
    owner_profile_id = getattr(session_manager_local, "my_profile_id", None)
    base_url = getattr(config_instance_local, "BASE_URL", "").rstrip("/")
    api_person_id = selected_candidate_raw.get("PersonId")
    api_tree_id = selected_candidate_raw.get("TreeId")

    if not owner_profile_id:
        print("\nCannot fetch details: Your User ID is required but missing.")
        logger.error("Cannot fetch details: Owner profile ID missing.")
        return None
    if not api_person_id or not api_tree_id:
        logger.error(
            f"Cannot fetch details: Missing PersonId ({api_person_id}) or TreeId ({api_tree_id}) for selected candidate."
        )
        print("\nError: Missing essential IDs for fetching person details.")
        return None
    if not callable(call_facts_user_api):
        logger.error("Cannot fetch details: call_facts_user_api function missing.")
        print("\nError: Details fetching utility unavailable.")
        return None

    # API call helper handles retries/fallbacks internally
    person_research_data = call_facts_user_api(
        session_manager_local, owner_profile_id, api_person_id, api_tree_id, base_url
    )

    if person_research_data is None:
        logger.warning("Failed to retrieve detailed information after all attempts.")
        print(
            "\nWarning: Could not retrieve detailed information for the selected match."
        )
        return None
    else:
        return person_research_data


# End of _handle_details_phase


def _handle_display_and_relationship_phase(
    person_research_data: Optional[Dict],  # Can be None if detail fetch failed
    selected_candidate_processed: Dict,  # Fallback info from initial scoring
    selected_candidate_raw: Dict,  # Raw suggestion data
    search_criteria: Dict[str, Any],
    session_manager_local: SessionManager,
    config_instance_local: Any,
):
    """Handles extracting, scoring, displaying details, family, and relationships."""
    extracted_info = None
    base_url = getattr(config_instance_local, "BASE_URL", "").rstrip("/")

    if person_research_data:  # Details were fetched successfully
        extracted_info = _extract_detailed_info(
            person_research_data, selected_candidate_raw
        )
        if not extracted_info:
            print("\nError: Failed to extract details from API response.")
            logger.error("Failed to extract details, using fallback display.")
            # Create minimal fallback from processed suggestion data
            extracted_info = selected_candidate_processed.get("parsed_suggestion", {})
            extracted_info["family_data"] = {}  # Ensure family_data exists
        else:
            # Only score and display full details if extraction succeeded
            final_score_info = _score_detailed_match(
                extracted_info, search_criteria, config_instance_local
            )
            _display_detailed_match_info(
                extracted_info, final_score_info, search_criteria, base_url
            )
            _display_family_info(extracted_info.get("family_data", {}))
    else:  # Handle failed detail fetch explicitly using processed suggestion data
        print("\n--- Top Match (From Initial Suggestion - Detailed Fetch Failed) ---")
        parsed_sugg = selected_candidate_processed.get(
            "parsed_suggestion", {}
        )  # Get pre-parsed data
        link_func = getattr(sys.modules.get("api_utils"), "_generate_person_link", None)
        person_link = "(Link unavailable)"
        if callable(link_func):
            person_link = link_func(
                parsed_sugg.get("user_id") or parsed_sugg.get("person_id"),
                parsed_sugg.get("tree_id") if not parsed_sugg.get("user_id") else None,
                base_url,
            )

        print(
            f"  Name : {parsed_sugg.get('name', 'Unknown')} (ID: {parsed_sugg.get('user_id') or parsed_sugg.get('person_id') or 'N/A'})"
        )
        print(f"  Link : {person_link}")
        print(
            f"  Born : {parsed_sugg.get('birth_date', '?')} in {parsed_sugg.get('birth_place') or '?'}"
        )
        if not parsed_sugg.get(
            "is_living", True
        ):  # Assume living if unknown and no death date
            print(
                f"  Died : {parsed_sugg.get('death_date', '?')} in {parsed_sugg.get('death_place') or '?'}"
            )
        elif (
            parsed_sugg.get("death_date", "N/A") != "N/A"
        ):  # Has death date but living status might be wrong
            print(
                f"  Died : {parsed_sugg.get('death_date', '?')} in {parsed_sugg.get('death_place') or '?'}"
            )
        else:
            print("  Died : (Assumed Living)")
        print(
            f"  Score: {selected_candidate_processed.get('score', 0):.0f} (Initial Suggestion Score)"
        )
        # Create minimal structure for relationship function from parsed suggestion
        extracted_info = parsed_sugg
        extracted_info["family_data"] = {}  # Ensure family_data exists

    # --- Display Relationship ---
    if not extracted_info:
        logger.error("Cannot display relationship path as extracted_info is missing.")
        print("\nError: Cannot determine relationship path due to previous errors.")
        return  # Cannot proceed

    selected_person_tree_id = extracted_info.get("person_id")  # Tree-specific ID
    selected_person_global_id = extracted_info.get("user_id")  # Global/Profile ID
    selected_tree_id = extracted_info.get("tree_id")
    selected_name = extracted_info.get("name", "Selected Person")
    owner_name = getattr(session_manager_local, "tree_owner_name", "the Tree Owner")
    owner_profile_id = getattr(session_manager_local, "my_profile_id", None)
    owner_tree_id = getattr(session_manager_local, "my_tree_id", None)

    if not owner_profile_id:
        logger.warning("Owner profile ID not found. Discovery relationship may fail.")
    if not owner_tree_id:
        logger.warning("Owner tree ID not found. Tree relationship may fail.")

    is_owner = bool(
        selected_person_global_id
        and owner_profile_id
        and selected_person_global_id.upper() == owner_profile_id.upper()
    )
    can_calc_tree = bool(
        owner_tree_id and selected_tree_id == owner_tree_id and selected_person_tree_id
    )
    can_calc_discovery = bool(selected_person_global_id and owner_profile_id)

    if is_owner:
        print(f"\n({selected_name} is the Tree Owner)")
        logger.info(
            f"Selected person ({selected_name}) is the Tree Owner. Skipping relationship path."
        )
    elif can_calc_tree:
        if all(
            isinstance(i, str)
            for i in [selected_person_tree_id, selected_name, owner_tree_id, owner_name]
        ):
            _display_tree_relationship(
                selected_person_tree_id,
                selected_name,
                owner_tree_id,
                owner_name,
                session_manager_local,
                base_url,
            )
        else:
            logger.error("Cannot calculate tree relationship: Invalid ID/Name types")
            print("\nCannot calculate relationship path: Invalid IDs")
    elif can_calc_discovery:
        if all(
            isinstance(i, str)
            for i in [
                selected_person_global_id,
                selected_name,
                owner_profile_id,
                owner_name,
            ]
        ):
            _display_discovery_relationship(
                selected_person_global_id,
                selected_name,
                owner_profile_id,
                owner_name,
                session_manager_local,
                base_url,
            )
        else:
            logger.error(
                "Cannot calculate discovery relationship: Invalid ID/Name types"
            )
            print("\nCannot calculate relationship path: Invalid IDs")
    else:
        # Log failure conditions
        conditions = []
        if not owner_profile_id:
            conditions.append("Owner Profile ID missing")
        if not selected_person_tree_id and not selected_person_global_id:
            conditions.append("Selected Person IDs missing")
        if not can_calc_tree and not can_calc_discovery:
            if owner_tree_id and selected_tree_id != owner_tree_id:
                conditions.append(
                    f"Target tree ({selected_tree_id}) != owner tree ({owner_tree_id})"
                )
            if not selected_person_global_id:
                conditions.append("Target global ID missing")
        elif not can_calc_tree:
            if not selected_person_global_id:
                conditions.append("Target global ID missing")
        elif not can_calc_discovery:
            if not owner_tree_id:
                conditions.append("Owner Tree ID missing")
            elif selected_tree_id != owner_tree_id:
                conditions.append("Target not in owner tree")
            elif not selected_person_tree_id:
                conditions.append("Target tree-person ID missing")
        logger.error(
            f"Cannot calculate relationship for {selected_name}: {'; '.join(conditions) or 'Unknown reason'}"
        )
        print(
            "\n(Cannot calculate relationship path: Necessary IDs or conditions not met. Check logs.)"
        )


# End of _handle_display_and_relationship_phase


# --- Main Handler ---
def handle_api_report():
    """Orchestrates the process of searching, selecting, detailing, and relating a person via API."""
    logger.info(
        "\n--- Action 11: Person Details & Relationship (Ancestry API Report) ---"
    )

    # --- Dependency Checks ---
    if not all(
        [
            CORE_UTILS_AVAILABLE,
            API_UTILS_AVAILABLE,
            GEDCOM_UTILS_AVAILABLE,
            config_instance,
            selenium_config,
            session_manager,
        ]
    ):
        logger.critical(
            "handle_api_report: One or more critical dependencies unavailable."
        )
        missing = [
            m
            for m, v in [
                ("Core Utils", CORE_UTILS_AVAILABLE),
                ("API Utils", API_UTILS_AVAILABLE),
                ("Gedcom Utils", GEDCOM_UTILS_AVAILABLE),
                ("Config", config_instance and selenium_config),
                ("Session Manager", session_manager),
            ]
            if not v
        ]
        logger.critical(f" - Missing: {', '.join(missing)}")
        print(
            "\nCRITICAL ERROR: Required libraries, utilities, or config unavailable. Check logs."
        )
        return False

    # --- Session Setup ---
    print("Initializing Ancestry session...")
    if not session_manager.ensure_session_ready(action_name="API Report Session Init"):
        logger.error("Failed to initialize Ancestry session for API report.")
        print("\nERROR: Failed to initialize session. Cannot proceed.")
        return False

    # --- Phase 1: Search ---
    logger.info("--- Action 11: Phase 1: Get Criteria & Search ---")
    search_criteria = _get_search_criteria()
    if not search_criteria:
        logger.info("Search criteria not provided. Exiting Action 11.")
        return True  # User cancelled is not an error

    suggestions_to_score = _handle_search_phase(
        session_manager, search_criteria, config_instance
    )
    if suggestions_to_score is None:
        return False  # Critical API failure
    if not suggestions_to_score:
        return True  # Search successful, no results

    # --- Phase 2: Score & Select ---
    logger.info("--- Action 11: Phase 2: Score & Select ---")
    selection = _handle_selection_phase(
        suggestions_to_score, search_criteria, config_instance
    )
    if not selection:
        return True  # Scoring/selection failed gracefully or no candidates
    selected_candidate_processed, selected_candidate_raw = selection

    # --- Phase 3: Fetch Details ---
    logger.info("--- Action 11: Phase 3: Fetch Details ---")
    person_research_data = _handle_details_phase(
        selected_candidate_raw, session_manager, config_instance
    )
    # Continue even if person_research_data is None (handled in next phase)

    # --- Phase 4 & 5: Process Details, Display Info & Relationship ---
    logger.info("--- Action 11: Phase 4/5: Display Info & Relationship ---")
    _handle_display_and_relationship_phase(
        person_research_data,  # Pass potentially None data
        selected_candidate_processed,  # Pass processed suggestion data for fallback
        selected_candidate_raw,  # Pass raw suggestion data for fallback name extraction etc.
        search_criteria,
        session_manager,
        config_instance,
    )

    # --- Finish ---
    logger.info("--- Action 11: Finished ---")
    return True  # Report completed workflow (even if some steps warned/failed non-critically)


# End of handle_api_report


# --- Main Execution ---
def main():
    """Main execution flow for Action 11 (API Report)."""
    logger.info("--- Action 11: API Report Starting ---")
    # Ensure session manager is available before calling handler
    if not session_manager:
        logger.critical("Session Manager instance not created. Exiting.")
        print("\nFATAL ERROR: Session Manager failed to initialize.")
        return

    try:
        report_successful = handle_api_report()
        if report_successful:
            logger.info("--- Action 11: API Report Finished Successfully ---")
            print("\nAction 11 finished.")
        else:
            logger.error("--- Action 11: API Report Finished with Errors ---")
            print("\nAction 11 finished with errors (check logs).")
    except Exception as e:
        logger.critical(
            f"Unhandled exception during Action 11 execution: {e}", exc_info=True
        )
        print(f"\nCRITICAL ERROR during Action 11: {e}. Check logs.")
    finally:
        # Optional: Close session if needed, though SessionManager might handle this
        # if session_manager: session_manager.close_sess()
        pass


# End of main

# Script entry point check
if __name__ == "__main__":
    if (
        CORE_UTILS_AVAILABLE
        and API_UTILS_AVAILABLE
        and GEDCOM_UTILS_AVAILABLE
        and config_instance
        and selenium_config
    ):
        main()
    else:
        print(
            "\nCRITICAL ERROR: Required utilities or configuration are not available."
        )
        print("Please check imports, dependencies, and config files.")
        logging.getLogger().critical("Exiting: Required components not loaded.")
        sys.exit(1)
# End of action11.py
# --- END OF FILE action11.py ---
