# --- START OF FILE action11.py ---

# action11.py

"""
Action 11: API Report - Search Ancestry API, display details, family, relationship.
V18.4: Added user feedback during potentially long API search calls.
       Corrected minor logic/logging issues identified during review.
       (Based on V18.3)
"""
# ... (imports and config loading remain the same) ...
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
import requests  # Keep for potential exception types from _api_req

# Import specific types needed locally
from typing import Optional, List, Dict, Any, Tuple, Union
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
    if not config_instance:
        raise ImportError("config_instance is None after import.")
    if not selenium_config:
        raise ImportError("selenium_config is None after import.")

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
            logger.warning(f"config_instance.{attr} dictionary/list is empty.")

    if not hasattr(selenium_config, "API_TIMEOUT"):
        raise TypeError(f"selenium_config.API_TIMEOUT missing.")

except ImportError as e:
    logger.critical(
        f"Failed to import config_instance/selenium_config from config.py: {e}. Cannot proceed.",
        exc_info=True,
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

# Double check critical configs are loaded
if (
    config_instance is None
    or not hasattr(config_instance, "COMMON_SCORING_WEIGHTS")
    or not hasattr(config_instance, "NAME_FLEXIBILITY")
    or not hasattr(config_instance, "DATE_FLEXIBILITY")
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
    GEDCOM_SCORING_AVAILABLE = calculate_match_score is not None
    GEDCOM_UTILS_AVAILABLE = all(
        f is not None for f in [_parse_date, _clean_display_date]
    )
except ImportError as e:
    logger.error(f"Failed to import from gedcom_utils: {e}.", exc_info=True)


# --- Import API Utilities ---
print_group = None
format_api_relationship_path = None
API_UTILS_AVAILABLE = False

try:
    from api_utils import print_group, format_api_relationship_path

    logger.info("Successfully imported required functions from api_utils.")
    API_UTILS_AVAILABLE = all(
        f is not None for f in [print_group, format_api_relationship_path]
    )
    if not API_UTILS_AVAILABLE:
        logger.error("One or more required functions from api_utils are None.")
except ImportError as e:
    logger.error(f"Failed to import from api_utils: {e}.", exc_info=True)


# --- Import General Utilities ---
try:
    from utils import (
        SessionManager,
        _api_req,
        nav_to_page,
        format_name,
        ordinal_case,
        make_ube,
        make_newrelic,
        make_traceparent,
        make_tracestate,
    )

    logger.info("Successfully imported required components from utils.")
    CORE_UTILS_AVAILABLE = True
except ImportError as e:
    logger.critical(
        f"Failed to import critical components from 'utils' module: {e}. Cannot run.",
        exc_info=True,
    )
    print(f"FATAL ERROR: Failed to import required functions from utils.py: {e}")
    CORE_UTILS_AVAILABLE = False
    sys.exit(1)


# --- Session Manager Instance ---
session_manager: SessionManager = SessionManager()


# --- Helper Function for Parsing PersonFacts Array ---
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

    if not isinstance(person_facts, list) or not _parse_date:
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

            if isinstance(parsed_date_data, dict):
                year = parsed_date_data.get("Year")
                month = parsed_date_data.get("Month")
                day = parsed_date_data.get("Day")
                if year:
                    try:
                        if month and day:
                            temp_date = (
                                f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
                            )
                        elif month:
                            temp_date = f"{year}-{str(month).zfill(2)}"
                        else:
                            temp_date = str(year)
                        date_obj = _parse_date(temp_date)
                        logger.debug(
                            f"_extract_fact_data: Parsed date object from ParsedDate: {date_obj}"
                        )
                    except (ValueError, TypeError) as dt_err:
                        logger.warning(
                            f"_extract_fact_data: Could not parse date from ParsedDate {parsed_date_data}: {dt_err}"
                        )
                        date_obj = None

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
    gender_input = input("  Gender (M/F): ").strip()

    gender = None
    if gender_input and gender_input[0].lower() in ["m", "f"]:
        gender = gender_input[0].lower()

    logger.info(f"  Input - First Name: {first_name}")
    logger.info(f"  Input - Surname: {surname}")
    logger.info(f"  Input - Birth Date/Year: {dob_str}")
    logger.info(f"  Input - Birth Place: {pob}")
    logger.info(f"  Input - Death Date/Year: {dod_str}")
    logger.info(f"  Input - Death Place: {pod}")
    logger.info(f"  Input - Gender: {gender}")

    if not (first_name or surname):
        logger.warning("API search needs First Name or Surname. Report cancelled.")
        print("\nAPI search needs First Name or Surname. Report cancelled.")
        return None

    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)

    target_birth_year: Optional[int] = None
    target_birth_date_obj: Optional[datetime] = None
    if dob_str and _parse_date:
        target_birth_date_obj = _parse_date(dob_str)
        if target_birth_date_obj:
            target_birth_year = target_birth_date_obj.year

    target_death_year: Optional[int] = None
    target_death_date_obj: Optional[datetime] = None
    if dod_str and _parse_date:
        target_death_date_obj = _parse_date(dod_str)
        if target_death_date_obj:
            target_death_year = target_death_date_obj.year

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

    logger.info("\n--- Final Search Criteria Prepared ---")
    for key, value in search_criteria_dict.items():
        # Display the cleaned/processed values used for scoring
        if value is not None and key not in [
            "first_name_raw",
            "surname_raw",
            "birth_date_obj",
            "death_date_obj",
        ]:
            logger.info(f"  {key.replace('_', ' ').title()}: '{value}'")
        elif key == "death_place" and value is None:  # Explicitly log if None
            logger.info(f"  {key.replace('_', ' ').title()}: None")

    return search_criteria_dict


def _search_suggest_api(
    search_criteria: Dict[str, Any],
    session_manager: SessionManager,
    owner_tree_id: str,
    owner_profile_id: Optional[str],
    base_url: str,
) -> Optional[List[Dict]]:
    """Calls the Ancestry Suggest API (/api/person-picker/suggest) with progressive timeouts and direct fallback."""
    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")
    suggest_params = []
    if first_name_raw:
        suggest_params.append(f"partialFirstName={quote(first_name_raw)}")
    if surname_raw:
        suggest_params.append(f"partialLastName={quote(surname_raw)}")
    suggest_params.append("isHideVeiledRecords=false")
    if birth_year:
        suggest_params.append(f"birthYear={birth_year}")

    suggest_url = f"{base_url}/api/person-picker/suggest/{owner_tree_id}?{'&'.join(suggest_params)}"

    owner_facts_referer = base_url
    if owner_profile_id and owner_tree_id:
        owner_facts_referer = urljoin(
            base_url,
            f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts",
        )
        logger.debug(
            f"Using owner facts page as referer for Suggest API: {owner_facts_referer}"
        )
    else:
        logger.warning(
            "Cannot construct specific owner facts referer for Suggest API. Using base URL."
        )

    logger.info(f"Attempting Suggest API search using _api_req: {suggest_url}")
    api_description = "Suggest API"

    # Use progressive timeouts to avoid waiting the full time on each attempt
    # Increased timeouts to handle slower API responses
    timeouts = [20, 30, 60]  # Increased progressive timeouts in seconds
    max_attempts = len(timeouts)

    # Get API timeout from selenium_config if available, otherwise use default
    api_timeout = 60  # Default timeout
    if selenium_config and hasattr(selenium_config, "API_TIMEOUT"):
        api_timeout = selenium_config.API_TIMEOUT

    print(f"Searching Ancestry API (Timeout: {api_timeout}s per attempt)")

    for attempt, timeout in enumerate(timeouts, 1):
        print(f"(Calling {api_description}... Timeout: {timeout}s)")

        try:
            # Create custom headers for the Suggest API
            # Enhanced headers based on successful API calls
            custom_headers = {
                "Accept": "application/json",
                "Referer": owner_facts_referer,
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "X-Requested-With": "XMLHttpRequest",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            }

            suggest_response = _api_req(
                url=suggest_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                api_description=api_description,
                headers=custom_headers,  # Use enhanced custom headers
                referer_url=owner_facts_referer,
                timeout=timeout,
                use_csrf_token=False,  # Don't use CSRF token for this API
            )

            if isinstance(suggest_response, list):
                logger.info(
                    f"Suggest API call successful on attempt {attempt}/{max_attempts}, found {len(suggest_response)} results."
                )
                return suggest_response
            elif suggest_response is None:
                # _api_req handles logging the specific error (timeout, connection etc)
                logger.warning(
                    f"{api_description} call using _api_req returned None on attempt {attempt}/{max_attempts}."
                )
                if attempt < max_attempts:
                    print(f"Retrying with longer timeout...")
                    continue  # Try next timeout
                else:
                    print(f"({api_description} call failed after all attempts.)")
                    # Don't return None yet, try direct request fallback
            else:
                logger.error(
                    f"Suggest API call using _api_req returned unexpected type: {type(suggest_response)}"
                )
                logger.debug(
                    f"Suggest API Response Content: {str(suggest_response)[:500]}"
                )
                print(f"({api_description} call returned unexpected data format.)")

                # If we got a response but it's not the expected format, don't retry
                # Try direct request fallback instead
                break

        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
            if attempt < max_attempts:
                print(f"Timeout occurred. Retrying with longer timeout...")
                continue  # Try next timeout
            else:
                logger.error(
                    f"{api_description} call failed after all timeout attempts."
                )
                print(
                    f"\nAll API search attempts timed out. Trying direct request fallback..."
                )
                # Fall through to try direct request fallback
                break
        except Exception as api_err:
            logger.error(
                f"{api_description} call failed with error on attempt {attempt}/{max_attempts}: {api_err}",
                exc_info=True,
            )
            if attempt < max_attempts:
                print(f"Error occurred. Retrying...")
                continue  # Try next timeout
            else:
                print(
                    f"\nAll API search attempts failed. Trying direct request fallback..."
                )
                break

    # If we reach here, all _api_req attempts failed
    # Try direct requests fallback before falling back to TreesUI List API
    print("\nAttempting direct request fallback...")
    try:
        # Get cookies from session manager's requests session
        cookies = {}
        if session_manager and session_manager._requests_session:
            cookies = session_manager._requests_session.cookies.get_dict()
            logger.debug(f"Got {len(cookies)} cookies from session manager")
        else:
            logger.warning("No cookies available from session manager")

        # Prepare headers for direct request with more complete headers
        direct_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": owner_facts_referer,
            "X-Requested-With": "XMLHttpRequest",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        }

        # Log the URL and headers for debugging
        logger.debug(f"Direct request URL: {suggest_url}")
        logger.debug(f"Direct request headers: {direct_headers}")
        logger.debug(f"Cookie count: {len(cookies)}")

        # Make direct request with a longer timeout
        print("Making direct API request (timeout: 30s)...")
        direct_response = requests.get(
            suggest_url, headers=direct_headers, cookies=cookies, timeout=30
        )

        # Check if successful
        if direct_response.status_code == 200:
            try:
                direct_data = direct_response.json()
                if isinstance(direct_data, list):
                    logger.info(
                        f"Direct request fallback successful! Found {len(direct_data)} results."
                    )
                    print(
                        f"Direct request successful! Found {len(direct_data)} potential matches."
                    )
                    return direct_data
                else:
                    logger.warning(
                        f"Direct request returned non-list data: {type(direct_data)}"
                    )
                    logger.debug(f"Response content: {str(direct_data)[:500]}")
            except Exception as json_err:
                logger.error(f"Error parsing direct request JSON: {json_err}")
                logger.debug(f"Response content: {direct_response.text[:500]}")
        else:
            logger.warning(
                f"Direct request failed with status code: {direct_response.status_code}"
            )
            logger.debug(f"Response content: {direct_response.text[:500]}")

    except requests.exceptions.Timeout:
        logger.error("Direct request timed out after 30 seconds")
        print("Direct request timed out. Trying alternative API...")
    except Exception as direct_err:
        logger.error(f"Direct request fallback failed: {direct_err}", exc_info=True)

    # If direct request fallback failed, try TreesUI List API as final fallback
    print("\nAttempting alternative API search method...")
    return _search_treesui_list_api(
        search_criteria, session_manager, owner_tree_id, owner_profile_id, base_url
    )


def _search_treesui_list_api(
    search_criteria: Dict[str, Any],
    session_manager: SessionManager,
    owner_tree_id: str,
    owner_profile_id: Optional[str],
    base_url: str,
) -> Optional[List[Dict]]:
    """Calls the Ancestry TreesUI List API (/api/treesui-list/trees/{tree}/persons) as a fallback with progressive timeouts."""
    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")

    if not birth_year:
        logger.warning("Cannot call TreesUI List API without birth year.")
        print("Cannot use alternative API search: birth year is required.")
        return None

    treesui_params = []
    if first_name_raw:
        treesui_params.append(f"fn={quote(first_name_raw)}")
    if surname_raw:
        treesui_params.append(f"ln={quote(surname_raw)}")
    treesui_params.extend([f"by={birth_year}", "limit=100", "fields=NAMES,BIRTH_DEATH"])

    treesui_url = f"{base_url}/api/treesui-list/trees/{owner_tree_id}/persons?{'&'.join(treesui_params)}"

    owner_facts_referer = base_url
    if owner_profile_id and owner_tree_id:
        owner_facts_referer = urljoin(
            base_url,
            f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts",
        )
        logger.debug(
            f"Using owner facts page as referer for TreesUI API: {owner_facts_referer}"
        )
    else:
        logger.warning(
            "Cannot construct specific owner facts referer for TreesUI API. Using base URL."
        )

    logger.info(f"Attempting TreesUI List API search using _api_req: {treesui_url}")
    api_description = "TreesUI List API"

    # Use progressive timeouts to avoid waiting the full time on each attempt
    # Start with a shorter timeout and increase if needed
    timeouts = [15, 25, 35]  # Progressive timeouts in seconds
    max_attempts = len(timeouts)

    print(
        f"Trying alternative API search (Progressive timeouts: {', '.join(map(str, timeouts))}s)"
    )

    for attempt, timeout in enumerate(timeouts, 1):
        print(
            f"(Attempt {attempt}/{max_attempts}: Calling {api_description}... Timeout: {timeout}s)"
        )

        try:
            # Create custom headers for the TreesUI List API
            custom_headers = {
                "Accept": "application/json",
                "Referer": owner_facts_referer,
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }

            treesui_response = _api_req(
                url=treesui_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                api_description=api_description,
                headers=custom_headers,  # Use custom headers
                referer_url=owner_facts_referer,
                timeout=timeout,
            )

            if treesui_response and isinstance(treesui_response, list):
                logger.info(
                    f"TreesUI List API call successful on attempt {attempt}/{max_attempts}! Found {len(treesui_response)} results."
                )
                print(
                    f"Alternative API search successful! Found {len(treesui_response)} potential matches."
                )
                return treesui_response
            elif treesui_response:
                logger.error(
                    f"TreesUI List API returned unexpected format: {type(treesui_response)}"
                )
                print("Alternative API search returned unexpected format.")
                # If we got a response but it's not the expected format, don't retry
                return None
            else:
                # _api_req handles logging the specific error (timeout, connection etc)
                logger.warning(
                    f"{api_description} call failed or returned None on attempt {attempt}/{max_attempts}."
                )
                if attempt < max_attempts:
                    print(f"Retrying with longer timeout...")
                    continue  # Try next timeout
                else:
                    print(f"Alternative API search failed after all attempts.")
                    return None

        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
            if attempt < max_attempts:
                print(f"Timeout occurred. Retrying with longer timeout...")
                continue  # Try next timeout
            else:
                logger.error(
                    f"{api_description} call failed after all timeout attempts."
                )
                print(f"\nAll alternative API search attempts timed out.")
                return None

        except Exception as treesui_err:
            logger.error(
                f"TreesUI List API call failed with error on attempt {attempt}/{max_attempts}: {treesui_err}",
                exc_info=True,
            )
            if attempt < max_attempts:
                print(f"Error occurred. Retrying...")
                continue  # Try next timeout
            else:
                print(f"\nAll alternative API search attempts failed.")
                return None

    # If we reach here, all attempts failed
    return None


# ... (rest of the functions _run_simple_suggestion_scoring to _display_discovery_relationship remain unchanged) ...
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
    cand_gn = candidate_data_dict.get("gender")
    search_fn = search_criteria.get("first_name")
    search_sn = search_criteria.get("surname")
    search_by = search_criteria.get("birth_year")
    search_gn = search_criteria.get("gender")

    # Name Scoring
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

    # Birth Year
    if cand_by and search_by and cand_by == search_by:
        score += 20
        field_scores["byear"] = 20
        reasons.append(f"Exact Birth Year ({cand_by}) (20pts)")

    # Death Year
    cand_dy = candidate_data_dict.get("death_year")
    search_dy = search_criteria.get("death_year")
    if cand_dy and search_dy and cand_dy == search_dy:
        score += 15
        field_scores["dyear"] = 15
        reasons.append(f"Exact Death Year ({cand_dy}) (15pts)")

    # Death Dates Absent
    is_living = candidate_data_dict.get("is_living")
    # Only score absent death if candidate is confirmed not living (or status unknown)
    # Avoid penalizing potentially living people if user didn't provide death year
    if not search_dy and not cand_dy and is_living in [False, None]:
        score += 15
        field_scores["ddate"] = 15
        reasons.append("Death Dates Absent (15pts)")

    # Gender
    if cand_gn and search_gn and cand_gn == search_gn:
        score += 25
        field_scores["gender"] = 25
        reasons.append(f"Gender Match ({cand_gn.upper()}) (25pts)")

    return score, field_scores, reasons


def _process_and_score_suggestions(
    suggestions: List[Dict], search_criteria: Dict[str, Any], config_instance: Any
) -> List[Dict]:
    """Processes raw API suggestions, extracts data, calculates match scores, returns sorted list."""
    processed_candidates = []
    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)
    logger.info(f"\n--- Filtering and Scoring {len(suggestions)} Candidates ---")

    # Check if the scoring function is available and callable
    scoring_function_available = (
        GEDCOM_SCORING_AVAILABLE
        and calculate_match_score is not None
        and callable(calculate_match_score)
    )

    if not scoring_function_available:
        logger.warning(
            "Gedcom scoring function unavailable or not callable. Using simple fallback scoring."
        )

    for idx, candidate in enumerate(suggestions):
        # Extract Name
        candidate_name = "Unknown"
        cand_given = candidate.get("GivenName", candidate.get("FirstName"))
        cand_sur = candidate.get("Surname", candidate.get("LastName"))
        if "FullName" in candidate:
            candidate_name = candidate.get("FullName")
        elif "Name" in candidate:
            candidate_name = candidate.get("Name")
        elif cand_given or cand_sur:
            candidate_name = f"{cand_given or ''} {cand_sur or ''}".strip()

        # Extract Other Details
        candidate_id = candidate.get("PersonId", f"Unknown_{idx}")
        birth_year_str = candidate.get("BirthYear")
        birth_date = candidate.get("BirthDate", "N/A")
        birth_place = candidate.get("BirthPlace", "N/A")
        death_year_str = candidate.get("DeathYear")
        death_date = candidate.get("DeathDate", "N/A")
        death_place = candidate.get("DeathPlace", "N/A")
        gender = candidate.get("Gender", "N/A")
        gender_norm = None
        if (
            gender
            and gender != "N/A"
            and isinstance(gender, str)
            and len(gender) > 0
            and gender[0].lower() in ["m", "f"]
        ):
            gender_norm = gender[0].lower()
        is_living_suggestion = candidate.get("IsLiving")  # May be None

        # Parse Years
        birth_year_int = None
        try:
            if birth_year_str and str(birth_year_str).isdigit():
                birth_year_int = int(birth_year_str)
        except (ValueError, TypeError):
            pass  # Keep None

        death_year_int = None
        try:
            if death_year_str and str(death_year_str).isdigit():
                death_year_int = int(death_year_str)
        except (ValueError, TypeError):
            pass  # Keep None

        # Prepare Candidate Data for Scoring
        candidate_data_dict = {
            "first_name": clean_param(cand_given),
            "surname": clean_param(cand_sur),
            "birth_year": birth_year_int,
            "birth_date_obj": None,
            "birth_place": clean_param(birth_place),
            "death_year": death_year_int,
            "death_date_obj": None,
            "death_place": clean_param(death_place),
            "gender": gender_norm,
            "is_living": is_living_suggestion,
            "norm_id": candidate_id,
            "display_id": candidate_id,
            "full_name_disp": candidate_name,
            "gender_norm": gender_norm,
            "birth_place_disp": clean_param(birth_place),
            "death_place_disp": clean_param(death_place),
        }

        # Calculate Score
        score = 0.0
        field_scores = {}
        reasons = []

        # Use the scoring function if available and callable
        if scoring_function_available:
            try:
                # We've already verified calculate_match_score is callable above
                score, field_scores, reasons = calculate_match_score(
                    search_criteria,
                    candidate_data_dict,
                    config_instance.COMMON_SCORING_WEIGHTS,
                    config_instance.NAME_FLEXIBILITY,
                    config_instance.DATE_FLEXIBILITY,
                )
            except Exception as score_err:
                logger.error(
                    f"Error calculating score for suggestion {candidate_id} using gedcom_utils: {score_err}",
                    exc_info=True,
                )
                # Fall back to simple scoring on error
                score, field_scores, reasons = _run_simple_suggestion_scoring(
                    search_criteria, candidate_data_dict
                )
                reasons.append("(Error Fallback)")
        else:
            # Use simple scoring if the main scoring function isn't available
            score, field_scores, reasons = _run_simple_suggestion_scoring(
                search_criteria, candidate_data_dict
            )
            reasons.append("(Using Simple Scoring)")

        # Append Processed Candidate
        processed_candidates.append(
            {
                "id": candidate_id,
                "name": candidate_name,
                "gender": gender,
                "birth_date": (
                    birth_date if birth_date != "N/A" else birth_year_str or "N/A"
                ),
                "birth_place": birth_place,
                "death_date": (
                    death_date if death_date != "N/A" else death_year_str or "N/A"
                ),
                "death_place": death_place,
                "score": score,
                "field_scores": field_scores,
                "reasons": reasons,
                "raw_data": candidate,
            }
        )

    # Sort and Log
    processed_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    logger.info(f"Scored {len(processed_candidates)} candidates.")
    return processed_candidates


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
        givn_s = fs.get("givn", 0)
        surn_s = fs.get("surn", 0)
        bonus_s = fs.get("bonus", 0)
        gender_s = fs.get("gender", 0)
        byear_s = fs.get("byear", 0)
        bdate_s = fs.get("bdate", 0)
        bplace_s = fs.get("bplace", 0)
        dyear_s = fs.get("dyear", 0)
        ddate_s = fs.get("ddate", 0)
        dplace_s = fs.get("dplace", 0)

        name_disp = candidate.get("name", "N/A")[:25] + (
            "..." if len(candidate.get("name", "")) > 25 else ""
        )
        name_score_str = f"[{givn_s+surn_s}]" + (f"[+{bonus_s}]" if bonus_s > 0 else "")
        name_with_score = f"{name_disp} {name_score_str}"

        raw_gender = candidate.get("gender", "")
        gender_disp = "N/A"
        if raw_gender and isinstance(raw_gender, str):
            # Handle cases like "Male", "Female", "M", "F", or other strings
            if raw_gender.lower().startswith("m"):
                gender_disp = "M"
            elif raw_gender.lower().startswith("f"):
                gender_disp = "F"
            else:
                gender_disp = str(raw_gender)  # Show unknown gender strings
        gender_with_score = f"{gender_disp} [{gender_s}]"

        bdate_disp = str(candidate.get("birth_date", "N/A"))
        birth_score_display = f"[{byear_s+bdate_s}]" if bdate_s else f"[{byear_s}]"
        bdate_with_score = f"{bdate_disp} {birth_score_display}"

        bplace_disp = str(candidate.get("birth_place")) or "N/A"
        bplace_disp = bplace_disp[:15] + ("..." if len(bplace_disp) > 15 else "")
        bplace_with_score = f"{bplace_disp} [{bplace_s}]"

        ddate_disp = str(candidate.get("death_date", "N/A"))
        death_score_display = f"[{dyear_s+ddate_s}]" if ddate_s else f"[{dyear_s}]"
        ddate_with_score = f"{ddate_disp} {death_score_display}"

        dplace_disp = str(candidate.get("death_place")) or "N/A"
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
        # Use a string format parameter that's recognized by tabulate
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    except Exception as tab_err:
        logger.error(f"Error formatting results table with tabulate: {tab_err}")
        # Fallback to manual table formatting
        print("\nSearch Results:")
        # Print headers
        print(" | ".join(headers))
        print("-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))
        # Print data rows
        for row in table_data:
            print(" | ".join(map(str, row)))
    print("")


def _select_top_candidate(
    scored_candidates: List[Dict], raw_suggestions: List[Dict]
) -> Optional[Tuple[Dict, Dict]]:
    """Selects the highest-scoring candidate and retrieves its original raw suggestion data."""
    if not scored_candidates:
        logger.info("No scored candidates available to select from.")
        return None

    top_scored_candidate = scored_candidates[0]
    top_scored_id = top_scored_candidate.get("id")

    if isinstance(top_scored_id, str) and top_scored_id.startswith("Unknown_"):
        logger.warning(
            f"Top candidate has generated ID '{top_scored_id}'. Using raw data directly."
        )
        top_candidate_raw = top_scored_candidate.get("raw_data")
        if top_candidate_raw and isinstance(top_candidate_raw, dict):
            logger.info(f"Using raw data associated with generated ID {top_scored_id}.")
            print(
                f"\n---> Auto-selecting top match: {top_scored_candidate.get('name', 'Unknown')}"
            )
            return top_scored_candidate, top_candidate_raw
        else:
            logger.error(
                f"Cannot proceed: Top candidate '{top_scored_id}' lacks valid PersonId and associated raw_data."
            )
            print(
                f"\nError: Cannot get details for top candidate {top_scored_candidate.get('name', 'Unknown')} due to missing ID."
            )
            return None

    logger.info(
        f"Highest scoring candidate selected: ID {top_scored_id}, Score {top_scored_candidate.get('score', 0):.0f}"
    )

    top_candidate_raw = None
    for suggestion in raw_suggestions:
        # Ensure comparison works even if one is int and other is str
        if str(suggestion.get("PersonId")) == str(top_scored_id):
            top_candidate_raw = suggestion
            logger.debug(f"Found raw data for top candidate ID {top_scored_id}")
            break

    if not top_candidate_raw:
        logger.error(
            f"Critical Error: Could not find raw suggestion data for top scored ID: {top_scored_id}."
        )
        print(
            f"\nError: Internal mismatch finding details for top candidate ({top_scored_id})."
        )
        return None

    print(
        f"\n---> Auto-selecting top match: {top_scored_candidate.get('name', 'Unknown')}"
    )
    return top_scored_candidate, top_candidate_raw


def _fetch_person_details(
    candidate_raw: Dict,
    owner_profile_id: str,
    session_manager: SessionManager,
    base_url: str,
) -> Optional[Dict]:
    """Fetches detailed person facts using the /facts/user/ API endpoint via _api_req with progressive timeouts."""
    api_person_id = candidate_raw.get("PersonId")
    api_tree_id = candidate_raw.get("TreeId")
    owner_tree_id = getattr(session_manager, "my_tree_id", None)

    if not all([api_person_id, api_tree_id, owner_profile_id]):
        logger.error(
            f"Facts API prerequisites missing (PersonId: {api_person_id}, TreeId: {api_tree_id}, OwnerUserId: {owner_profile_id})"
        )
        print("\nError: Missing essential IDs for fetching person details.")
        return None

    facts_api_url = f"{base_url}/family-tree/person/facts/user/{owner_profile_id.lower()}/tree/{api_tree_id}/person/{api_person_id}"
    api_description = "Person Facts API"
    logger.info(
        f"Attempting to fetch facts for PersonID {api_person_id} using _api_req: {facts_api_url}"
    )

    facts_referer = base_url
    if owner_tree_id and owner_profile_id:
        facts_referer = urljoin(
            base_url,
            f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts",
        )

    # Use progressive timeouts to avoid waiting the full time on each attempt
    # Start with a shorter timeout and increase if needed
    timeouts = [30, 45, 60]  # Increased timeouts in seconds
    max_attempts = len(timeouts)

    print(
        f"Fetching person details (Progressive timeouts: {', '.join(map(str, timeouts))}s)"
    )

    facts_data_raw = None

    # First try with direct requests approach (similar to the curl command)
    print(f"Attempting direct API request first...")
    try:
        # Get cookies from session manager's requests session
        cookies = {}
        if session_manager and session_manager._requests_session:
            cookies = session_manager._requests_session.cookies.get_dict()
            logger.debug(f"Got {len(cookies)} cookies from session manager")
        else:
            logger.warning("No cookies available from session manager")

        # Prepare headers for direct request with more complete headers (similar to curl)
        direct_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-GB,en;q=0.9",
            "Referer": facts_referer,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Content-Type": "application/json",
            "DNT": "1",
            "Connection": "keep-alive",
        }

        # Log the URL and headers for debugging
        logger.debug(f"Direct request URL: {facts_api_url}")
        logger.debug(f"Direct request headers: {direct_headers}")
        logger.debug(f"Cookie count: {len(cookies)}")

        # Make direct request with a longer timeout
        print("Making direct API request (timeout: 30s)...")
        direct_response = requests.get(
            facts_api_url, headers=direct_headers, cookies=cookies, timeout=30
        )

        # Check if successful
        if direct_response.status_code == 200:
            try:
                facts_data_raw = direct_response.json()
                if isinstance(facts_data_raw, dict):
                    logger.info(
                        f"Direct request successful! Response is a dictionary with {len(facts_data_raw)} top-level keys."
                    )
                    print(f"Direct request successful!")
                else:
                    logger.warning(
                        f"Direct request returned non-dict data: {type(facts_data_raw)}"
                    )
                    logger.debug(f"Response content: {str(facts_data_raw)[:500]}")
                    facts_data_raw = None
            except Exception as json_err:
                logger.error(f"Error parsing direct request JSON: {json_err}")
                logger.debug(f"Response content: {direct_response.text[:500]}")
                facts_data_raw = None
        else:
            logger.warning(
                f"Direct request failed with status code: {direct_response.status_code}"
            )
            logger.debug(f"Response content: {direct_response.text[:500]}")
            facts_data_raw = None

    except requests.exceptions.Timeout:
        logger.error("Direct request timed out after 30 seconds")
        print("Direct request timed out. Trying alternative approach...")
        facts_data_raw = None
    except Exception as direct_err:
        logger.error(f"Direct request failed: {direct_err}", exc_info=True)
        facts_data_raw = None

    # If direct request failed, try the original _api_req approach
    if facts_data_raw is None:
        print("Direct request failed. Trying original approach...")
        for attempt, timeout in enumerate(timeouts, 1):
            print(
                f"(Attempt {attempt}/{max_attempts}: Fetching details... Timeout: {timeout}s)"
            )

            try:
                # Create custom headers for the Facts API
                custom_headers = {
                    "Accept": "application/json",
                    "Referer": facts_referer,
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                    "X-Requested-With": "XMLHttpRequest",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
                }

                facts_data_raw = _api_req(
                    url=facts_api_url,
                    driver=session_manager.driver,
                    session_manager=session_manager,
                    method="GET",
                    api_description=api_description,
                    headers=custom_headers,  # Use enhanced custom headers
                    referer_url=facts_referer,
                    timeout=timeout,
                )

                # --- Post-Call Logging for _api_req ---
                logger.info(
                    f"--- _api_req response received for {api_description} (Attempt {attempt}/{max_attempts}) ---"
                )
                logger.info(f"Response Type: {type(facts_data_raw)}")

                if isinstance(
                    facts_data_raw, requests.Response
                ):  # Check if _api_req returned raw Response
                    logger.warning(
                        f"{api_description} _api_req returned raw Response object (Status: {getattr(facts_data_raw, 'status_code', 'N/A')} {getattr(facts_data_raw, 'reason', '')}). Expected dict."
                    )
                    try:
                        logger.debug(
                            f"Response Text (first 500): {facts_data_raw.text[:500]}"
                        )
                    except Exception as log_e:
                        logger.error(
                            f"Could not log response text from error Response: {log_e}"
                        )
                    facts_data_raw = None  # Treat as failure

                    if attempt < max_attempts:
                        print(f"Received error response. Retrying...")
                        continue  # Try next timeout

                elif (
                    facts_data_raw is None
                ):  # Check if _api_req returned None explicitly
                    logger.warning(
                        f"{api_description} response (_api_req) was None on attempt {attempt}/{max_attempts}."
                    )
                    if attempt < max_attempts:
                        print(f"No response received. Retrying with longer timeout...")
                        continue  # Try next timeout

                elif isinstance(
                    facts_data_raw, dict
                ):  # Check for expected success case
                    logger.info(
                        f"{api_description} response (_api_req) was a dictionary. Top-level keys: {list(facts_data_raw.keys())}"
                    )
                    logger.debug(
                        f"{api_description} Response Dict Preview (_api_req): {json.dumps(facts_data_raw, indent=2, default=str)[:1000]}"
                    )
                    # Success! Break out of the retry loop
                    break

                else:  # Handle any other unexpected type
                    logger.warning(
                        f"{api_description} response (_api_req) was unexpected type: {type(facts_data_raw)}. Value: {str(facts_data_raw)[:500]}"
                    )
                    facts_data_raw = None  # Treat as failure

                    if attempt < max_attempts:
                        print(f"Received unexpected response format. Retrying...")
                        continue  # Try next timeout

            except requests.exceptions.Timeout:
                logger.warning(
                    f"{api_description} call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
                )
                if attempt < max_attempts:
                    print(f"Timeout occurred. Retrying with longer timeout...")
                    continue  # Try next timeout
                else:
                    logger.error(
                        f"{api_description} call failed after all timeout attempts."
                    )
                    print(f"\nAll attempts to fetch person details timed out.")
                    facts_data_raw = None

            except Exception as api_req_err:
                logger.error(
                    f"{api_description} call using _api_req failed on attempt {attempt}/{max_attempts}: {api_req_err}",
                    exc_info=True,
                )
                if attempt < max_attempts:
                    print(f"Error occurred. Retrying...")
                    continue  # Try next timeout
                else:
                    print(f"\nError fetching person details: {api_req_err}")
                    facts_data_raw = None

    # --- Process Final Result ---
    if not isinstance(facts_data_raw, dict):
        logger.error(f"Failed to fetch facts data after all attempts.")
        print(f"\nError: Could not fetch valid person details from API. Check logs.")
        return None

    person_research_data = facts_data_raw.get("data", {}).get("personResearch")
    if not isinstance(person_research_data, dict) or not person_research_data:
        logger.error(f"Facts API response is missing 'data.personResearch' dictionary.")
        logger.debug(f"Full raw response keys: {list(facts_data_raw.keys())}")
        if "data" in facts_data_raw:
            logger.debug(f"'data' sub-keys: {list(facts_data_raw['data'].keys())}")
        print(
            f"\nError: API response format for person details was unexpected (Missing personResearch section)."
        )
        return None

    logger.info(
        f"Successfully fetched and parsed 'personResearch' data dictionary for PersonID {api_person_id}."
    )
    return person_research_data


def _extract_best_name_from_details(
    person_research_data: Dict, candidate_raw: Dict
) -> str:
    """Extracts the best available name from multiple potential sources in API details."""
    best_name = candidate_raw.get("FullName", candidate_raw.get("Name", "Unknown"))
    logger.debug(f"_extract_best_name: Initial name from suggestion: {best_name}")

    person_facts_list = person_research_data.get("PersonFacts", [])
    if not isinstance(person_facts_list, list):
        person_facts_list = []

    name_fact = next(
        (
            f
            for f in person_facts_list
            if isinstance(f, dict) and f.get("TypeString") == "Name"
        ),
        None,
    )
    if name_fact and name_fact.get("Value"):
        formatted_nf = format_name(name_fact.get("Value"))
        logger.debug(
            f"_extract_best_name: Found Name fact, formatted: '{formatted_nf}'"
        )
        if formatted_nf and formatted_nf != "Valued Relative":
            best_name = formatted_nf  # Ensure not empty after format

    person_full_name = person_research_data.get("PersonFullName")
    if person_full_name and person_full_name != "Unknown":
        formatted_full_name = format_name(person_full_name)
        logger.debug(
            f"_extract_best_name: Found PersonFullName fact, formatted: '{formatted_full_name}'"
        )
        if (
            formatted_full_name
            and formatted_full_name != "Valued Relative"
            and (best_name == "Unknown" or len(formatted_full_name) > len(best_name))
        ):
            best_name = formatted_full_name
            logger.debug("_extract_best_name: Using PersonFullName as best name.")

    first_name_comp = person_research_data.get("FirstName", "")
    last_name_comp = person_research_data.get("LastName", "")
    if first_name_comp or last_name_comp:
        constructed_name = f"{first_name_comp} {last_name_comp}".strip()
        logger.debug(f"_extract_best_name: Constructed='{constructed_name}'")
        if constructed_name and (
            best_name == "Unknown"
            or best_name == "Valued Relative"
            or len(constructed_name) > len(best_name)
        ):
            best_name = constructed_name
            logger.debug("_extract_best_name: Using constructed name as best name.")

    # Final check for empty or invalid name
    if not best_name or best_name == "Valued Relative":
        best_name = "Unknown"

    return best_name


def _extract_detailed_info(person_research_data: Dict, candidate_raw: Dict) -> Dict:
    """Extracts detailed information from the 'personResearch' dictionary."""
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
    logger.debug(f"Gender from PersonGender: {gender_str}")
    if not gender_str:  # Check if gender_str is None or empty
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
            logger.debug(f"Gender from Gender Fact: {gender_str}")

    extracted["gender_str"] = gender_str
    extracted["gender"] = (
        "m" if gender_str == "Male" else ("f" if gender_str == "Female" else None)
    )
    logger.info(f"Final Extracted Gender: {extracted['gender']} (from '{gender_str}')")

    # --- Living Status ---
    extracted["is_living"] = person_research_data.get("IsPersonLiving", False)
    logger.info(f"Is Living: {extracted['is_living']}")

    # --- Birth/Death ---
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
        if birth_date_str and _clean_display_date
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
        if death_date_str and _clean_display_date
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

    # --- Name Components ---
    extracted["first_name"] = None
    extracted["surname"] = None
    name_parts = best_name.split()
    if name_parts and best_name not in ["Unknown", "Valued Relative"]:
        extracted["first_name"] = name_parts[0]
        if len(name_parts) > 1:
            extracted["surname"] = name_parts[-1]  # Crude assumption
    logger.debug(
        f"Extracted name components for scoring: First='{extracted['first_name']}', Sur='{extracted['surname']}'"
    )

    return extracted


def _score_detailed_match(
    extracted_info: Dict, search_criteria: Dict[str, Any], config_instance: Any
) -> Tuple[float, Dict, List[str]]:
    """Calculates final match score based on detailed info."""
    # Check if the scoring function is available and callable
    scoring_function_available = (
        GEDCOM_SCORING_AVAILABLE
        and calculate_match_score is not None
        and callable(calculate_match_score)
    )
    if not scoring_function_available:
        logger.warning(
            "Gedcom scoring unavailable for detailed match. Using simple fallback."
        )

    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)

    candidate_processed_data = {
        "norm_id": extracted_info.get("person_id"),
        "display_id": extracted_info.get("person_id"),
        "first_name": clean_param(extracted_info.get("first_name")),
        "surname": clean_param(extracted_info.get("surname")),
        "full_name_disp": extracted_info.get("name"),
        "gender_norm": extracted_info.get("gender"),
        "birth_year": extracted_info.get("birth_year"),
        "birth_date_obj": extracted_info.get("birth_date_obj"),
        "birth_place_disp": clean_param(extracted_info.get("birth_place")),
        "death_year": extracted_info.get("death_year"),
        "death_date_obj": extracted_info.get("death_date_obj"),
        "death_place_disp": clean_param(extracted_info.get("death_place")),
        "is_living": extracted_info.get("is_living", False),
    }
    logger.debug(
        f"Detailed scoring - Search criteria: {json.dumps(search_criteria, default=str)}"
    )
    logger.debug(
        f"Detailed scoring - Candidate data: {json.dumps(candidate_processed_data, default=str)}"
    )

    score = 0.0
    field_scores = {}
    reasons_list = ["API Suggest Match"]

    if scoring_function_available:
        try:
            logger.debug(
                "Calculating detailed score using gedcom_utils.calculate_match_score..."
            )
            score, field_scores, reasons = calculate_match_score(
                search_criteria,
                candidate_processed_data,
                config_instance.COMMON_SCORING_WEIGHTS,
                config_instance.NAME_FLEXIBILITY,
                config_instance.DATE_FLEXIBILITY,
            )
            # Ensure base reason is present
            if "API Suggest Match" not in reasons:
                reasons.insert(0, "API Suggest Match")
            reasons_list = reasons
            logger.info(f"Calculated detailed score via gedcom_utils: {score:.0f}")
        except Exception as e:
            logger.error(
                f"Error calculating detailed score using gedcom_utils: {e}",
                exc_info=True,
            )
            logger.warning("Falling back to simple scoring for detailed match.")
            score, field_scores, reasons_list = _run_simple_suggestion_scoring(
                search_criteria, extracted_info
            )
            reasons_list.append("(Detailed Scoring Error Fallback)")
    else:
        logger.debug("Calculating detailed score using simple fallback scoring...")
        score, field_scores, reasons_list = _run_simple_suggestion_scoring(
            search_criteria, extracted_info
        )
        reasons_list.append("(Detailed Scoring Fallback)")

    logger.debug(f"Final detailed score: {score:.0f}")
    logger.debug(f"Final detailed field scores: {field_scores}")
    logger.debug(f"Final detailed reasons: {reasons_list}")
    return score, field_scores, reasons_list


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
    cp = {
        "first_name": str(ei.get("first_name", na)),
        "surname": str(ei.get("surname", na)),
        "birth_year": str(ei.get("birth_year", na)),
        "birth_place": str(ei.get("birth_place", na)),
        "death_year": str(ei.get("death_year", na)),
        "death_place": str(ei.get("death_place", na)),
        "gender": str(ei.get("gender", na)),  # 'm'/'f'/None converted to string
    }
    sc_fn = str(sc.get("first_name", na))
    sc_sn = str(sc.get("surname", na))
    sc_by = str(sc.get("birth_year", na))
    sc_bp = str(sc.get("birth_place", na))
    sc_dy = str(sc.get("death_year", na))
    sc_dp = str(sc.get("death_place", na))
    sc_gn = str(sc.get("gender", na))  # 'm'/'f'/None converted to string

    print(f"  First Name : {sc_fn:<15} vs {cp['first_name']}")
    print(f"  Last Name  : {sc_sn:<15} vs {cp['surname']}")
    print(f"  Gender     : {sc_gn:<15} vs {cp['gender']}")
    print(f"  Birth Year : {sc_by:<15} vs {cp['birth_year']}")
    print(f"  Birth Place: {sc_bp:<15} vs {cp['birth_place']}")
    print(f"  Death Year : {sc_dy:<15} vs {cp['death_year']}")
    print(f"  Death Place: {sc_dp:<15} vs {cp['death_place']}")

    logger.debug("  Detailed Field Scores:")
    for field, score_value in field_scores.items():
        logger.debug(f"    {field}: {score_value}")

    print("\nScore Reasons:")
    for reason in reasons:
        print(f"  - {reason}")

    print(f"\n--- Top Match Details ---")
    person_link = "(Link unavailable)"
    tree_id = ei.get("tree_id")
    person_id = ei.get("person_id")
    if tree_id and person_id and base_url:
        person_link = (
            f"{base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/facts"
        )

    print(f"  Name : {ei.get('name', 'Unknown')} (ID: {person_id or 'N/A'})")
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


def _flatten_children_list(children_raw: Union[List, Dict, None]) -> List[Dict]:
    """Flattens potentially nested list of children and removes duplicates."""
    children_flat_list = []
    added_ids = set()  # Track IDs to avoid duplicates
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
                        children_flat_list.append(child_dict)
                else:
                    logger.warning(
                        f"_flatten_children_list: Non-dict item found within child entry: {type(child_dict)}"
                    )
    elif isinstance(children_raw, dict):
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


def _display_family_info(family_data: Dict):
    """Displays formatted family information (parents, siblings, spouses, children)."""
    print("\nRelatives:")
    logger.info("\n  Relatives:")
    if not isinstance(family_data, dict) or not family_data:
        logger.warning("_display_family_info: Received empty or invalid family_data.")
        print("  Family data unavailable.")
        return

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

            name = format_name(relative.get("FullName", "Unknown"))
            lifespan = relative.get("LifeRange", "")
            b_date, d_date = "", ""
            if (
                lifespan
                and isinstance(lifespan, str)
                and ("-" in lifespan or "–" in lifespan)
            ):
                parts = re.split(r"\s*[-––]\s*", lifespan)
                b_date = parts[0].strip() if len(parts) >= 1 else ""
                d_date = parts[1].strip() if len(parts) == 2 else ""
                if len(parts) > 2:
                    logger.debug(f"Ambiguous lifespan '{lifespan}'.")
                    d_date = ""

            life_info = ""
            if b_date or d_date:
                life_info = (
                    f" (b. {b_date or '?'}" + (f", d. {d_date}" if d_date else "") + ")"
                )
            elif lifespan and isinstance(lifespan, str):
                life_info = f" ({lifespan})"

            rel_info = f"- {name}{life_info}"
            print(f"    {rel_info}")
            logger.info(f"      {rel_info}")
            found_any = True

        if not found_any:
            print("    None found.")
            logger.info("    None found (list contained invalid entries).")

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


def _display_tree_relationship(
    selected_person_tree_id: str,
    selected_name: str,
    owner_tree_id: str,
    owner_name: str,
    session_manager: SessionManager,
    base_url: str,
):
    """Calculates and displays the relationship path using the Tree Ladder API (/getladder)."""
    api_description = "Get Tree Ladder API"
    print(f"\n--- Relationship Path (within Tree) to {owner_name} ---")
    print(f"Calculating path for {selected_name}...")
    logger.info(
        f"Calculating Tree relationship path for {selected_name} (PersonID: {selected_person_tree_id}) to {owner_name} (TreeID: {owner_tree_id})"
    )

    ladder_api_url_base = f"{base_url}/family-tree/person/tree/{owner_tree_id}/person/{selected_person_tree_id}/getladder"
    callback_name = f"__ancestry_jsonp_{int(time.time()*1000)}"
    timestamp_ms = int(time.time() * 1000)
    query_params = urlencode({"callback": callback_name, "_": timestamp_ms})
    ladder_api_url = f"{ladder_api_url_base}?{query_params}"
    ladder_referer = urljoin(
        base_url,
        f"/family-tree/person/tree/{owner_tree_id}/person/{selected_person_tree_id}/facts",
    )
    api_timeout = 20  # Specific timeout
    relationship_data = None

    try:
        logger.info(f"Calling {api_description} at {ladder_api_url}")
        logger.debug(f" Referer: {ladder_referer}")
        # ** Add User Feedback **
        print(
            f"(Calculating relationship via {api_description}... Timeout: {api_timeout}s)"
        )
        relationship_data = _api_req(
            url=ladder_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            api_description=api_description,
            referer_url=ladder_referer,
            use_csrf_token=False,
            force_text_response=True,
            timeout=api_timeout,
        )
    except requests.exceptions.Timeout:
        logger.error(
            f"{api_description} call caught specific Timeout exception after {api_timeout}s."
        )
        print(f"(Error: Timed out fetching relationship path from Tree API)")
        return
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        print(f"(Error fetching relationship path from Tree API: {e})")
        return

    if not relationship_data:
        logger.warning(f"API call '{api_description}' returned no data.")
        print("(Tree API call for relationship returned no response or empty data)")
        return

    print("")  # Add space before output
    # print(f"  {selected_name}") # Name already clear from context
    # logger.info(f"    {selected_name}")

    fallback_message_text = "(Could not parse relationship path from Tree API)"

    if format_api_relationship_path:
        try:
            formatted_path = format_api_relationship_path(
                relationship_data, owner_name, selected_name
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
                # Print formatted path directly
                print(
                    formatted_path
                )  # Assumes formatter includes necessary indentation/newlines
                logger.info("    --- Tree Relationship Path Interpretation ---")
                for line in formatted_path.splitlines():
                    if line.strip():
                        logger.info(f"    {line.strip()}")
                # logger.info(f"    -> {owner_name} (Tree Owner)") # Implicit in formatter output
                logger.info("    ------------------------------------")
            else:
                logger.warning(
                    f"format_api_relationship_path returned no path or error: '{formatted_path}'"
                )
                print(
                    f"  {formatted_path or fallback_message_text}"
                )  # Display error from parser or fallback
                logger.warning(
                    f"Relationship parsing failed/returned error. Raw response:\n{str(relationship_data)[:1000]}"
                )
        except Exception as fmt_err:
            logger.error(
                f"Error calling format_api_relationship_path: {fmt_err}", exc_info=True
            )
            print(f"  {fallback_message_text} (Processing Error)")
            logger.warning(
                f"Relationship parsing failed during format call. Raw response:\n{str(relationship_data)[:1000]}"
            )
    else:
        logger.error(
            "format_api_relationship_path function not available from api_utils."
        )
        print(f"  {fallback_message_text} (Utility Missing)")


def _display_discovery_relationship(
    selected_person_global_id: str,
    selected_name: str,
    owner_profile_id: str,
    owner_name: str,
    session_manager: SessionManager,
    base_url: str,
):
    """Calculates and displays the relationship path using the Discovery API (/relationshiptome)."""
    api_description = "Discovery Relationship API"
    print(f"\n--- Relationship Path (Discovery) to {owner_name} ---")
    print(f"Calculating path for {selected_name}...")
    logger.info(
        f"Calculating Discovery relationship path for {selected_name} (GlobalID: {selected_person_global_id}) to {owner_name} (ProfileID: {owner_profile_id})"
    )

    ladder_api_url = f"{base_url}/discoveryui-matchesservice/api/samples/{selected_person_global_id}/relationshiptome/{owner_profile_id}"
    uuid_for_referer = getattr(session_manager, "my_uuid", None)
    ladder_referer = base_url
    if uuid_for_referer:
        ladder_referer = urljoin(
            base_url, f"/discoveryui-matches/list/{uuid_for_referer}"
        )
    api_timeout = 20  # Specific timeout
    relationship_data = None

    try:
        logger.info(f"Calling {api_description} at {ladder_api_url}")
        logger.debug(f" Referer: {ladder_referer}")
        # ** Add User Feedback **
        print(
            f"(Calculating relationship via {api_description}... Timeout: {api_timeout}s)"
        )
        relationship_data = _api_req(
            url=ladder_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            api_description=api_description,
            referer_url=ladder_referer,
            timeout=api_timeout,
        )
    except requests.exceptions.Timeout:
        logger.error(
            f"{api_description} call caught specific Timeout exception after {api_timeout}s."
        )
        print(f"(Error: Timed out fetching relationship path from Discovery API)")
        return
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        print(f"(Error fetching relationship path from Discovery API: {e})")
        return

    if not relationship_data:
        logger.warning(f"API call '{api_description}' returned no data.")
        print(
            "(Discovery API call for relationship returned no response or empty data)"
        )
        return
    if not isinstance(relationship_data, dict):
        logger.warning(
            f"API call '{api_description}' returned unexpected type: {type(relationship_data)}"
        )
        print("(Discovery API call returned data in unexpected format)")
        logger.debug(f"Raw Discovery response: {str(relationship_data)[:1000]}")
        return

    print("")  # Add space before output
    # print(f"  {selected_name}") # Name already clear
    # logger.info(f"    {selected_name}")

    fallback_message_text = "(Could not parse relationship path from Discovery API)"

    # Attempt direct parsing first
    if isinstance(relationship_data.get("path"), list) and relationship_data.get(
        "path"
    ):
        logger.info("    --- Discovery Relationship Path Interpretation ---")
        path_steps = relationship_data["path"]
        # Print target first for context
        print(f"  {selected_name}")
        logger.info(f"    {selected_name}")
        for step in path_steps:
            step_name = step.get("name", "?")
            step_rel = step.get("relationship", "?")
            # Just capitalize the relationship term since _get_relationship_term isn't available
            display_rel = step_rel.capitalize()
            display_line = (
                f"  -> {display_rel} is {format_name(step_name)}"  # Format name
            )
            print(display_line)
            logger.info(f"    {display_line.strip()}")
        print(f"  -> {owner_name} (Tree Owner / You)")
        logger.info(f"    -> {owner_name} (Tree Owner / You)")
        logger.info("    ------------------------------------")
    # Fallback to using format_api_relationship_path if direct parse fails
    elif format_api_relationship_path and callable(format_api_relationship_path):
        logger.warning(
            "Discovery API response JSON structure unknown/unparsed. Attempting format_api_relationship_path (may fail)."
        )
        try:
            # Convert relationship_data to string if needed
            relationship_data_to_pass = relationship_data
            if isinstance(relationship_data, bytes):
                try:
                    relationship_data_to_pass = relationship_data.decode("utf-8")
                except UnicodeDecodeError:
                    relationship_data_to_pass = str(relationship_data)

            # Pass the dict - format_api_relationship_path should ideally handle this gracefully
            formatted_path = format_api_relationship_path(
                relationship_data_to_pass, owner_name, selected_name
            )
            known_error_starts = (
                "(No relationship",
                "(Could not parse",
                "(API returned error",
                "(Relationship HTML structure",
                "(Unsupported API response",
                "(Error processing relationship",
                "(Received unhandled dictionary",
            )
            if formatted_path and not formatted_path.startswith(known_error_starts):
                # Print target first for context
                print(f"  {selected_name}")
                logger.info(f"    {selected_name}")
                # Print formatted path
                print(formatted_path)
                logger.info(
                    "    --- Discovery Relationship Path Interpretation (Fallback Parser) ---"
                )
                for line in formatted_path.splitlines():
                    if line.strip():
                        logger.info(f"    {line.strip()}")
                # logger.info(f"    -> {owner_name} (Tree Owner)") # Implicit
                logger.info("    ------------------------------------")
            else:
                logger.warning(
                    f"format_api_relationship_path failed on Discovery JSON: '{formatted_path}'"
                )
                print(f"  {fallback_message_text}")  # Use fallback message on error
                logger.warning(
                    f"Discovery relationship parsing failed. Raw response dict:\n{json.dumps(relationship_data, indent=2)}"
                )
        except Exception as fmt_err:
            logger.error(
                f"Error calling format_api_relationship_path on Discovery JSON: {fmt_err}",
                exc_info=True,
            )
            print(f"  {fallback_message_text} (Processing Error)")
            logger.warning(
                f"Discovery relationship parsing failed during format call. Raw response dict:\n{json.dumps(relationship_data, indent=2)}"
            )
    else:
        logger.error(
            "format_api_relationship_path function not available from api_utils."
        )
        print(f"  {fallback_message_text} (Utility Missing)")


# --- Main Handler ---
def handle_api_report():
    """Orchestrates the process of searching, selecting, detailing, and relating a person via API."""
    logger.info(
        "\n--- Action 11: Person Details & Relationship (Ancestry API Report) ---"
    )

    # --- Dependency Checks ---
    missing_core = not CORE_UTILS_AVAILABLE
    missing_api_utils = not API_UTILS_AVAILABLE
    missing_gedcom_utils = not GEDCOM_UTILS_AVAILABLE
    missing_libs = not requests
    missing_config = config_instance is None or selenium_config is None
    if (
        missing_core
        or missing_api_utils
        or missing_gedcom_utils
        or missing_libs
        or missing_config
    ):
        logger.critical(
            "handle_api_report: One or more critical dependencies unavailable."
        )
        if missing_core:
            logger.critical(" - Core utils (utils.py) missing.")
        if missing_api_utils:
            logger.critical(
                " - API utils (api_utils.py, format_api_relationship_path) missing."
            )
        if missing_gedcom_utils:
            logger.critical(" - GEDCOM date utils (gedcom_utils.py) missing.")
        if missing_libs:
            logger.critical(" - Required library (requests) missing.")
        if missing_config:
            logger.critical(" - Config instance(s) missing.")
        print(
            "\nCRITICAL ERROR: Required libraries or utilities unavailable. Check logs, imports and installations."
        )
        return False

    # --- Session Setup ---
    print("Initializing Ancestry session...")
    if not session_manager.ensure_session_ready(action_name="API Report Session Init"):
        logger.error("Failed to initialize Ancestry session for API report.")
        print("\nERROR: Failed to initialize session. Cannot proceed.")
        return False

    owner_name = getattr(session_manager, "tree_owner_name", "the Tree Owner")
    owner_profile_id = getattr(session_manager, "my_profile_id", None)
    owner_tree_id = getattr(session_manager, "my_tree_id", None)
    base_url = getattr(
        config_instance, "BASE_URL", "https://www.ancestry.co.uk"
    ).rstrip("/")
    if not owner_profile_id:
        logger.warning(
            "Owner profile ID not found in session. Relationship path calculation may fail."
        )
    if not owner_tree_id:
        logger.warning("Owner tree ID not found in session. Some API calls may fail.")
        print("\nWARNING: Your Tree ID could not be determined. API search may fail.")

    # --- Phase 1: Search ---
    logger.info("--- Action 11: Phase 1: Search ---")
    search_criteria = _get_search_criteria()
    if not search_criteria:
        logger.info("Search criteria not provided. Exiting Action 11.")
        return True  # User cancelled

    print(
        "Searching Ancestry API (Using progressive timeouts to avoid long waits)..."
    )  # User feedback

    # Track API search attempts
    api_search_attempts = 0
    max_api_search_attempts = 2  # Try main search method twice before giving up
    suggestions_raw = None

    while api_search_attempts < max_api_search_attempts and suggestions_raw is None:
        api_search_attempts += 1

        if api_search_attempts > 1:
            print(
                f"\nRetrying API search (Attempt {api_search_attempts}/{max_api_search_attempts})..."
            )
            # Add a small delay between attempts
            time.sleep(2)

        if owner_tree_id:
            # Try the Suggest API first
            suggestions_raw = _search_suggest_api(
                search_criteria,
                session_manager,
                owner_tree_id,
                owner_profile_id,
                base_url,
            )

            # Fallback if suggest failed/timed out AND birth year available
            if suggestions_raw is None and search_criteria.get("birth_year"):
                logger.warning(
                    "Suggest API failed or timed out, attempting TreesUI List API fallback..."
                )
                print("\nTrying alternative API endpoint...")  # User feedback
                suggestions_raw = _search_treesui_list_api(
                    search_criteria,
                    session_manager,
                    owner_tree_id,
                    owner_profile_id,
                    base_url,
                )
        else:
            logger.error("Cannot perform API search: Owner Tree ID is missing.")
            print("\nERROR: Cannot perform API search because your Tree ID is unknown.")
            return False

    # Check results after all attempts
    if suggestions_raw is None:
        logger.error(
            f"All API Search attempts ({api_search_attempts}) failed critically."
        )
        print(
            "\nError during API search. No results found or API calls failed critically."
        )
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Verify that Ancestry.co.uk is accessible in your browser")
        print("3. Try again later as the Ancestry API may be experiencing issues")
        print("4. Try a different search with fewer criteria")
        return False
    if not suggestions_raw:
        logger.info("API Search returned no results.")
        print("\nNo potential matches found in Ancestry API based on criteria.")
        return True  # Successful search, no results

    # Limit suggestions to score
    suggestions_to_score = suggestions_raw
    max_score_limit = getattr(config_instance, "MAX_SUGGESTIONS_TO_SCORE", 10)
    if max_score_limit > 0 and len(suggestions_raw) > max_score_limit:
        logger.warning(
            f"Processing only top {max_score_limit} of {len(suggestions_raw)} suggestions for scoring."
        )
        suggestions_to_score = suggestions_raw[:max_score_limit]

    # --- Phase 2: Score & Select ---
    logger.info("--- Action 11: Phase 2: Score & Select ---")
    scored_candidates = _process_and_score_suggestions(
        suggestions_to_score, search_criteria, config_instance
    )
    if not scored_candidates:
        print("\nNo suitable candidates found after scoring.")
        logger.info("No candidates available after scoring process.")
        return True  # Successful processing, no suitable candidates

    max_display_limit = getattr(config_instance, "MAX_CANDIDATES_TO_DISPLAY", 5)
    _display_search_results(scored_candidates, max_display_limit)

    selection = _select_top_candidate(scored_candidates, suggestions_raw)
    if not selection:
        print("\nFailed to select a top candidate (check logs for errors).")
        logger.error("Failed to select top candidate.")
        return False
    selected_candidate_processed, selected_candidate_raw = selection

    # --- Phase 3: Fetch Details ---
    logger.info("--- Action 11: Phase 3: Fetch Details ---")
    person_research_data = None

    if not owner_profile_id:
        print("\nCannot fetch details: Your User ID is required but missing.")
        logger.error("Cannot fetch details: Owner profile ID missing.")
    else:
        # Track person details fetch attempts
        details_fetch_attempts = 0
        max_details_fetch_attempts = 2  # Try fetching details twice before giving up

        while (
            details_fetch_attempts < max_details_fetch_attempts
            and person_research_data is None
        ):
            details_fetch_attempts += 1

            if details_fetch_attempts > 1:
                print(
                    f"\nRetrying person details fetch (Attempt {details_fetch_attempts}/{max_details_fetch_attempts})..."
                )
                # Add a small delay between attempts
                time.sleep(2)

            person_research_data = _fetch_person_details(
                selected_candidate_raw, owner_profile_id, session_manager, base_url
            )

        if (
            person_research_data is None
            and details_fetch_attempts >= max_details_fetch_attempts
        ):
            logger.warning(
                f"All person details fetch attempts ({details_fetch_attempts}) failed."
            )
            print(
                "\nWarning: Could not retrieve detailed information after multiple attempts."
            )

    # --- Phase 4: Process Details & Display ---
    logger.info("--- Action 11: Phase 4: Process Details & Display ---")
    extracted_info = None
    if person_research_data:
        extracted_info = _extract_detailed_info(
            person_research_data, selected_candidate_raw
        )
        if not extracted_info:
            print("\nError: Failed to extract details from API response.")
            logger.error("Failed to extract details.")
            # Create minimal fallback structure
            extracted_info = {
                "person_id": selected_candidate_raw.get("PersonId"),
                "tree_id": selected_candidate_raw.get("TreeId"),
                "user_id": selected_candidate_raw.get("UserId"),
                "name": selected_candidate_processed.get("name", "Selected Person"),
                "family_data": {},
            }
        else:
            # Only score and display full details if extraction succeeded
            final_score_info = _score_detailed_match(
                extracted_info, search_criteria, config_instance
            )
            _display_detailed_match_info(
                extracted_info, final_score_info, search_criteria, base_url
            )
            _display_family_info(extracted_info.get("family_data", {}))
    elif not person_research_data:  # Handle failed detail fetch explicitly
        print(
            "\nWarning: Failed to retrieve detailed information for the selected match."
        )
        logger.warning(
            "Proceeding to relationship path calculation despite failed detail fetch."
        )
        print(f"\n--- Top Match (Minimal Info) ---")
        print(
            f"  Name : {selected_candidate_processed.get('name', 'Unknown')} (ID: {selected_candidate_raw.get('PersonId') or 'N/A'})"
        )
        print(
            f"  Born : {selected_candidate_processed.get('birth_date', '?')} in {selected_candidate_processed.get('birth_place') or '?'}"
        )
        print(
            f"  Died : {selected_candidate_processed.get('death_date', '?')} in {selected_candidate_processed.get('death_place') or '?'}"
        )
        print(
            f"  Score: {selected_candidate_processed.get('score', 0):.0f} (Initial Suggestion Score)"
        )
        # Create minimal structure for relationship function
        extracted_info = {
            "person_id": selected_candidate_raw.get("PersonId"),
            "tree_id": selected_candidate_raw.get("TreeId"),
            "user_id": selected_candidate_raw.get("UserId"),  # Global ID
            "name": selected_candidate_processed.get("name", "Selected Person"),
            "family_data": {},  # Needs to exist for safety, even if empty
        }

    # --- Phase 5: Display Relationship ---
    logger.info("--- Action 11: Phase 5: Display Relationship ---")
    if not extracted_info:
        logger.error("Cannot display relationship path as extracted_info is missing.")
        print("\nError: Cannot determine relationship path due to previous errors.")
        return False  # Indicate failure if we somehow reach here without extracted_info

    selected_person_tree_id = extracted_info.get("person_id")  # Tree-specific ID
    selected_person_global_id = extracted_info.get("user_id")  # Global ID
    selected_tree_id = extracted_info.get("tree_id")
    selected_name = extracted_info.get("name", "Selected Person")

    # Ensure owner_profile_id exists before comparison/use
    is_owner = bool(
        selected_person_global_id
        and owner_profile_id
        and selected_person_global_id.upper() == owner_profile_id.upper()
    )

    # Determine calculation possibility
    can_calc_tree = bool(
        owner_tree_id and selected_tree_id == owner_tree_id and selected_person_tree_id
    )
    can_calc_discovery = bool(
        selected_person_global_id and owner_profile_id
    )  # Requires both IDs

    if is_owner:
        print(f"\n({selected_name} is the Tree Owner)")
        logger.info(
            f"Selected person ({selected_name}) is the Tree Owner ({owner_name}). Skipping relationship path."
        )
    elif can_calc_tree:
        # Only try if we have all required IDs as strings
        if (
            isinstance(selected_person_tree_id, str)
            and isinstance(owner_tree_id, str)
            and isinstance(selected_name, str)
            and isinstance(owner_name, str)
        ):
            _display_tree_relationship(
                selected_person_tree_id,
                selected_name,
                owner_tree_id,
                owner_name,
                session_manager,
                base_url,
            )
        else:
            logger.error(
                "Cannot calculate tree relationship: One or more required IDs are not valid strings"
            )
            print("\nCannot calculate relationship path: Missing or invalid IDs")
    elif can_calc_discovery:
        # Only try if we have all required IDs as strings
        if (
            isinstance(selected_person_global_id, str)
            and isinstance(owner_profile_id, str)
            and isinstance(selected_name, str)
            and isinstance(owner_name, str)
        ):
            _display_discovery_relationship(
                selected_person_global_id,
                selected_name,
                owner_profile_id,
                owner_name,
                session_manager,
                base_url,
            )
        else:
            logger.error(
                "Cannot calculate discovery relationship: One or more required IDs are not valid strings"
            )
            print("\nCannot calculate relationship path: Missing or invalid IDs")
    else:
        # Log conditions for failure more clearly
        log_msg = f"Cannot calculate relationship for {selected_name}: "
        conditions = []
        if not owner_profile_id:
            conditions.append("Owner Profile ID missing from session")
        if not selected_person_tree_id and not selected_person_global_id:
            conditions.append("Selected Person IDs (Tree or Global) missing")
        if not can_calc_tree and not can_calc_discovery:
            if owner_tree_id and selected_tree_id != owner_tree_id:
                conditions.append(
                    f"Target tree ({selected_tree_id}) differs from owner tree ({owner_tree_id})"
                )
            if not selected_person_global_id:
                conditions.append("Target global ID missing for Discovery API")
        elif not can_calc_tree:  # Discovery might have been possible but failed check?
            if not selected_person_global_id:
                conditions.append("Target global ID missing for Discovery API")
            elif not owner_profile_id:
                conditions.append("Owner Profile ID missing")  # Redundant but safe
        elif not can_calc_discovery:  # Tree might have been possible but failed check?
            if not owner_tree_id:
                conditions.append("Owner Tree ID missing")
            elif selected_tree_id != owner_tree_id:
                conditions.append("Target not in owner tree")
            elif not selected_person_tree_id:
                conditions.append("Target tree-person ID missing")

        log_msg += (
            "; ".join(conditions)
            if conditions
            else "Unknown reason (check available IDs and conditions)"
        )
        logger.error(log_msg)
        print(
            "\n(Cannot calculate relationship path: Necessary IDs or conditions not met. Check logs.)"
        )

    # --- Finish ---
    logger.info("--- Action 11: Finished ---")
    return True  # Report completed successfully


# --- Main Execution ---
def main():
    """Main execution flow for Action 11 (API Report)."""
    logger.info("--- Action 11: API Report Starting ---")
    report_successful = handle_api_report()
    if report_successful:
        logger.info("--- Action 11: API Report Finished Successfully ---")
        print("\nAction 11 finished.")
    else:
        logger.error("--- Action 11: API Report Finished with Errors ---")
        print("\nAction 11 finished with errors (check logs).")


# Script entry point check
if __name__ == "__main__":
    if CORE_UTILS_AVAILABLE:
        main()
    else:
        print(
            "\nCRITICAL ERROR: Required core utilities (utils.py) are not installed or failed to load."
        )
        print("Please check your Python environment and dependencies.")
        logging.getLogger().critical("Exiting: Required core utilities not loaded.")
        sys.exit(1)
# End of action11.py
# --- END OF FILE action11.py ---
