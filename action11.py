# action11.py
"""
Action 11: API Report - Search Ancestry API, display details, family, relationship.
V17.6: Corrected family data extraction in _extract_detailed_info.
       Refined family display in _display_family_info with better validation.
       Reordered field comparison display in _display_detailed_match_info.
       Removed unreliable keyword relationship deduction; improved parsing attempts & logging.
"""
# --- Standard library imports ---
import logging
import sys
import time
import urllib.parse
import json
from pathlib import Path
from urllib.parse import urljoin, urlencode, quote
from tabulate import tabulate

# Import specific types needed locally
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime

# --- Third-party imports ---
try:
    import requests

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        BeautifulSoup = None  # type: ignore

    try:
        import cloudscraper
    except ImportError:
        cloudscraper = None
except ImportError:
    requests = None
    cloudscraper = None
    BeautifulSoup = None  # type: ignore

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
try:
    from config import config_instance

    logger.info("Successfully imported config_instance.")
    if not config_instance:
        raise ImportError("config_instance is None after import.")
    if not hasattr(config_instance, "COMMON_SCORING_WEIGHTS") or not isinstance(
        config_instance.COMMON_SCORING_WEIGHTS, dict
    ):
        raise TypeError("config_instance.COMMON_SCORING_WEIGHTS missing/invalid.")
    if not hasattr(config_instance, "NAME_FLEXIBILITY") or not isinstance(
        config_instance.NAME_FLEXIBILITY, dict
    ):
        raise TypeError("config_instance.NAME_FLEXIBILITY missing/invalid.")
    if not hasattr(config_instance, "DATE_FLEXIBILITY") or not isinstance(
        config_instance.DATE_FLEXIBILITY, dict
    ):
        raise TypeError("config_instance.DATE_FLEXIBILITY missing/invalid.")
    if not config_instance.COMMON_SCORING_WEIGHTS:
        logger.warning("config_instance.COMMON_SCORING_WEIGHTS dictionary is empty.")

except ImportError as e:
    logger.critical(
        f"Failed to import config_instance from config.py: {e}. Cannot proceed.",
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

if (
    config_instance is None
    or not hasattr(config_instance, "COMMON_SCORING_WEIGHTS")
    or not hasattr(config_instance, "NAME_FLEXIBILITY")
    or not hasattr(config_instance, "DATE_FLEXIBILITY")
):
    logger.critical("One or more critical configuration components failed to load.")
    print("\nFATAL ERROR: Configuration load failed.")
    sys.exit(1)


# --- Import GEDCOM Utilities (for scoring and date helpers) ---
calculate_match_score = None
_parse_date = None
_clean_display_date = None
GEDCOM_UTILS_AVAILABLE = False
GEDCOM_SCORING_AVAILABLE = False  # Specific flag for scoring function

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
display_raw_relationship_ladder = None
format_api_relationship_path = None
API_UTILS_AVAILABLE = False

try:
    from api_utils import (
        print_group,
        display_raw_relationship_ladder,
        format_api_relationship_path,
    )

    logger.info("Successfully imported required functions from api_utils.")
    API_UTILS_AVAILABLE = all(
        f is not None
        for f in [
            print_group,
            display_raw_relationship_ladder,
            format_api_relationship_path,
        ]
    )
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
    sys.exit(1)  # Exit if core utils are missing


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
        return (
            date_str,
            place_str,
            date_obj,
        )  # Return defaults if input invalid or date util missing

    for fact in person_facts:
        if (
            isinstance(fact, dict)
            and fact.get("TypeString") == fact_type_str
            and not fact.get("IsAlternate")  # Prioritize primary facts
        ):
            date_str = fact.get("Date")
            place_str = fact.get("Place")
            parsed_date_data = fact.get("ParsedDate")
            logger.debug(
                f"_extract_fact_data: Found primary fact for {fact_type_str}: Date='{date_str}', Place='{place_str}', ParsedDate={parsed_date_data}"
            )

            # Try parsing the date object if available and valid
            if isinstance(parsed_date_data, dict):
                year = parsed_date_data.get("Year")
                month = parsed_date_data.get("Month")
                day = parsed_date_data.get("Day")
                if year:
                    try:
                        # Construct YYYY-MM-DD format for better parsing
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
                        date_obj = None  # Reset on error

            # If date_obj parsing failed or wasn't possible, try parsing the Date string directly
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

            break  # Found the primary fact, break the loop

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
    gender = (
        gender_input[0].lower()
        if gender_input and gender_input[0].lower() in ["m", "f"]
        else None
    )

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

    # Prepare search criteria for scoring and API calls
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
        "first_name_raw": first_name,  # Keep raw input for API params
        "surname_raw": surname,  # Keep raw input for API params
        "first_name": clean_param(first_name),
        "surname": clean_param(surname),
        "birth_year": target_birth_year,
        "birth_date_obj": target_birth_date_obj,
        "birth_place": clean_param(pob),
        "death_year": target_death_year,
        "death_date_obj": target_death_date_obj,
        "death_place": clean_param(pod),  # Value will be None if user entered nothing
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


def _search_ancestry_api(
    search_criteria: Dict[str, Any],
    session_manager: SessionManager,
    owner_tree_id: Optional[str],
    owner_profile_id: Optional[str],
    base_url: str,
) -> Optional[List[Dict]]:
    """
    Performs the Ancestry API search using /suggest (via Cloudscraper first)
    or /treesui-list as a fallback if birth year is provided. Implements
    logic from V16.5-V16.15 including fallbacks and cookie sync.

    Args:
        search_criteria: Dictionary containing processed search parameters.
        session_manager: The active SessionManager instance.
        owner_tree_id: The tree ID of the owner's tree.
        owner_profile_id: The profile ID of the tree owner.
        base_url: The base Ancestry URL.

    Returns:
        A list of suggestion dictionaries, or None if the search failed critically,
        or an empty list if no matches were found.
    """
    if not owner_tree_id:
        logger.error("Cannot perform API search: My tree ID is not available.")
        print("\nERROR: Cannot determine your tree ID for API search.")
        return None
    if not session_manager.scraper:
        logger.warning(
            "Cloudscraper instance not available. Suggest API will use _api_req directly."
        )
        # Allow fallback to _api_req below

    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")

    # --- Prepare Suggest API Call ---
    suggest_params = []
    if first_name_raw:
        suggest_params.append(f"partialFirstName={quote(first_name_raw)}")
    if surname_raw:
        suggest_params.append(f"partialLastName={quote(surname_raw)}")
    suggest_params.append("isHideVeiledRecords=false")  # As per V16.3
    if birth_year:
        suggest_params.append(f"birthYear={birth_year}")

    suggest_url = f"{base_url}/api/person-picker/suggest/{owner_tree_id}?{'&'.join(suggest_params)}"

    # Determine Referer
    owner_facts_referer = base_url  # Default referer
    if owner_profile_id and owner_tree_id:  # Use more specific referer if possible
        owner_facts_referer = urljoin(
            base_url,
            f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts",
        )
        logger.debug(f"Using owner facts page as referer: {owner_facts_referer}")
    else:
        logger.warning(
            "Cannot construct specific owner facts referer for Suggest API (owner profile/tree ID missing). Using base URL."
        )

    logger.info(f"Attempting Suggest API search: {suggest_url}")
    print("Searching Ancestry API...")

    suggest_response = None
    scraper_response = None

    # --- Attempt 1: Cloudscraper for Suggest API (If available) ---
    if session_manager.scraper:
        try:
            logger.info("Attempting Suggest API call using Cloudscraper...")
            scraper = session_manager.scraper
            logger.debug("Syncing cookies from requests session to Cloudscraper...")
            session_manager._sync_cookies()  # Ensure latest cookies are synced
            scraper.cookies.clear()
            synced_count = 0
            for cookie in session_manager._requests_session.cookies:
                try:
                    scraper.cookies.set(
                        name=cookie.name,
                        value=cookie.value,
                        domain=cookie.domain,
                        path=cookie.path,
                    )
                    synced_count += 1
                except Exception as set_cookie_err:
                    logger.warning(
                        f"Failed to set cookie '{cookie.name}' in cloudscraper: {set_cookie_err}"
                    )
            logger.debug(f"Synced {synced_count} cookies to Cloudscraper.")
            # Verify essential cookies
            if not any(c.name == "ANCSESSIONID" for c in scraper.cookies):
                logger.warning(
                    "ANCSESSIONID cookie potentially missing after sync to Cloudscraper!"
                )

            # Headers based on V16.14 successful curl for Suggest API
            scraper_headers = {
                "accept": "application/json",
                "accept-language": "en-GB,en;q=0.9",
                "cache-control": "no-cache",
                "content-type": "application/json",
                "dnt": "1",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "referer": owner_facts_referer,
                "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            }

            wait_time = session_manager.dynamic_rate_limiter.wait()
            if wait_time > 0.1:
                logger.debug(
                    f"[Suggest API (Cloudscraper)] Rate limit wait: {wait_time:.2f}s"
                )

            logger.debug(
                f"Making Cloudscraper request to {suggest_url} with timeout=15s and headers: {scraper_headers}"
            )
            scraper_response = scraper.get(
                suggest_url, headers=scraper_headers, timeout=15
            )
            logger.debug(
                f"Cloudscraper response status: {scraper_response.status_code}"
            )
            logger.debug(
                f"Cloudscraper response headers: {dict(scraper_response.headers)}"
            )
            scraper_response.raise_for_status()  # Check for HTTP errors

            content_type = scraper_response.headers.get("Content-Type", "")
            if (
                "application/json" not in content_type
                and "text/json" not in content_type
            ):
                logger.warning(
                    f"Suggest API (Cloudscraper) unexpected content type: {content_type}"
                )
                logger.debug(f"Response text: {scraper_response.text[:500]}")
                raise ValueError("Non-JSON response received from Cloudscraper")

            suggest_response = scraper_response.json()
            logger.info("Suggest API call successful using Cloudscraper.")
            session_manager.dynamic_rate_limiter.decrease_delay()

        except (
            cloudscraper.exceptions.CloudflareChallengeError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            requests.exceptions.RequestException,
            ValueError,
        ) as scrape_err:
            logger.warning(f"Cloudscraper Suggest API call failed: {scrape_err}")
            if (
                isinstance(scrape_err, requests.exceptions.HTTPError)
                and scrape_err.response is not None
            ):
                logger.warning(f"  Status Code: {scrape_err.response.status_code}")
                logger.debug(f"  Response Text: {scrape_err.response.text[:500]}")
                if scrape_err.response.status_code == 401:
                    logger.error(
                        "Suggest API (Cloudscraper) returned 401 Unauthorized (Cookie/Session issue?)"
                    )
            if isinstance(scrape_err, ValueError):  # JSON decode error likely
                logger.debug(
                    f"Cloudscraper Response Text (on error): {getattr(scraper_response, 'text', 'N/A')[:500]}"
                )
            suggest_response = None  # Ensure it's None for fallback logic
            session_manager.dynamic_rate_limiter.increase_delay()
        except Exception as scrape_err:
            logger.error(
                f"Unexpected error during Cloudscraper Suggest API call: {scrape_err}",
                exc_info=True,
            )
            suggest_response = None
            session_manager.dynamic_rate_limiter.increase_delay()

    # --- Attempt 2: Fallback using _api_req for Suggest API ---
    if suggest_response is None:
        logger.warning(
            "Cloudscraper failed or was skipped, attempting fallback Suggest API call using _api_req..."
        )
        try:
            fallback_headers = {
                "Accept": "application/json, text/plain, */*",
                "Referer": owner_facts_referer,
            }
            logger.debug(
                f"Making _api_req request to {suggest_url} with timeout=15s and headers: {fallback_headers}"
            )
            suggest_response = _api_req(
                url=suggest_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                api_description="Suggest API (Fallback)",
                headers=fallback_headers,
                timeout=15,
            )
            if isinstance(suggest_response, list):  # Success is usually a list
                logger.info("Suggest API fallback call successful using _api_req.")
            elif suggest_response is None:
                logger.error("Suggest API fallback call using _api_req returned None.")
            else:  # Received something unexpected
                logger.error(
                    f"Suggest API fallback call using _api_req returned unexpected type: {type(suggest_response)}"
                )
                logger.debug(
                    f"Fallback Response Content: {str(suggest_response)[:500]}"
                )
                suggest_response = None  # Treat unexpected as failure

        except Exception as fallback_err:
            logger.error(
                f"Fallback Suggest API call (_api_req) failed with error: {fallback_err}",
                exc_info=True,
            )
            suggest_response = None  # Ensure it's None for next fallback

    # --- Attempt 3: Fallback using _api_req for TreesUI List API (if birth year exists) ---
    if suggest_response is None and birth_year:
        logger.warning(
            "Suggest API methods failed. Trying treesui-list endpoint as final fallback..."
        )
        print("\nTrying alternative API endpoint with birth year parameter...")
        treesui_params = []
        if first_name_raw:
            treesui_params.append(f"fn={quote(first_name_raw)}")
        if surname_raw:
            treesui_params.append(f"ln={quote(surname_raw)}")
        treesui_params.extend(
            [f"by={birth_year}", "limit=100", "fields=NAMES,BIRTH_DEATH"]
        )  # Use user format
        treesui_url = f"{base_url}/api/treesui-list/trees/{owner_tree_id}/persons?{'&'.join(treesui_params)}"
        logger.info(f"Attempting TreesUI List API call: {treesui_url}")

        try:
            treesui_headers = {
                "Accept": "application/json, text/plain, */*",
                "Referer": owner_facts_referer,
            }
            logger.debug(
                f"Making _api_req request to {treesui_url} with timeout=15s and headers: {treesui_headers}"
            )
            treesui_response = _api_req(
                url=treesui_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                api_description="TreesUI List API (Fallback)",
                headers=treesui_headers,
                timeout=15,
            )

            if treesui_response and isinstance(treesui_response, list):
                logger.info(
                    f"TreesUI List API call successful! Found {len(treesui_response)} results."
                )
                print(
                    f"Alternative API search successful! Found {len(treesui_response)} potential matches."
                )
                suggest_response = treesui_response  # Use this response
            elif treesui_response:
                logger.error(
                    f"TreesUI List API returned unexpected format: {type(treesui_response)}"
                )
                print("Alternative API search returned unexpected format.")
                suggest_response = None
            else:
                logger.error("TreesUI List API call failed or returned None.")
                print("Alternative API search also failed.")
                suggest_response = None
        except Exception as treesui_err:
            logger.error(
                f"TreesUI List API call failed with error: {treesui_err}", exc_info=True
            )
            print(f"Alternative API search failed: {treesui_err}")
            suggest_response = None

    # --- Final Check and Return ---
    if suggest_response is None:
        logger.error("All API search attempts failed.")
        print(
            "\nError during API search. No results found or API calls failed critically."
        )
        return None  # Indicate critical failure
    elif not isinstance(suggest_response, list):
        logger.error(
            f"Final API search result was not a list, type: {type(suggest_response)}"
        )
        print("\nError: API search returned data in an unexpected format.")
        return None  # Indicate critical failure
    elif not suggest_response:
        logger.info("API search successful but returned no matches.")
        print("\nNo potential matches found in Ancestry API based on criteria.")
        return []  # Return empty list for no matches
    else:
        logger.info(
            f"API search successful, found {len(suggest_response)} raw suggestions."
        )
        logger.debug(
            f"Full Suggest API response (first 5): {json.dumps(suggest_response[:5], indent=2)}"
        )
        print(f"\nTotal API results: {len(suggest_response)}")
        return suggest_response


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
    # Add check for death year match (if both exist)
    cand_dy = candidate_data_dict.get("death_year")
    search_dy = search_criteria.get("death_year")
    if cand_dy and search_dy and cand_dy == search_dy:
        score += 15
        field_scores["dyear"] = 15
        reasons.append(f"Exact Death Year ({cand_dy}) (15pts)")
    # Score death dates both absent (consistent with gedcom_utils scoring)
    is_living = candidate_data_dict.get(
        "is_living"
    )  # May not be available in suggestions
    if (
        not search_dy and not cand_dy and is_living is False
    ):  # Only if confirmed not living or unknown
        score += 15
        field_scores["ddate"] = 15
        reasons.append(
            "Death Dates Absent (15pts)"
        )  # Or dyear? Use ddate consistent with detailed scoring
    if cand_gn and search_gn and cand_gn == search_gn:
        score += 25
        field_scores["gender"] = 25
        reasons.append(f"Gender Match ({cand_gn.upper()}) (25pts)")

    return score, field_scores, reasons


def _process_and_score_suggestions(
    suggestions: List[Dict],
    search_criteria: Dict[str, Any],
    config_instance: Any,  # Assuming config_instance has scoring attrs
) -> List[Dict]:
    """
    Processes raw API suggestions, extracts data, calculates match scores
    (using gedcom_utils function or simple fallback), and returns a list
    of candidate dictionaries sorted by score.

    Args:
        suggestions: List of raw suggestion dictionaries from the API.
        search_criteria: Dictionary of user's search criteria for scoring.
        config_instance: The loaded configuration object with scoring weights.

    Returns:
        A list of processed candidate dictionaries, sorted by score descending.
    """
    processed_candidates = []
    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)

    logger.info(f"\n--- Filtering and Scoring {len(suggestions)} Candidates ---")

    scoring_function_available = (
        GEDCOM_SCORING_AVAILABLE and calculate_match_score is not None
    )
    if not scoring_function_available:
        logger.warning(
            "Gedcom scoring function (calculate_match_score) not available. Using simple fallback scoring."
        )

    for idx, candidate in enumerate(suggestions):
        # Extract name robustly
        candidate_name = "Unknown"
        cand_given = candidate.get("GivenName", candidate.get("FirstName"))
        cand_sur = candidate.get("Surname", candidate.get("LastName"))
        if "FullName" in candidate:
            candidate_name = candidate.get("FullName")
        elif "Name" in candidate:
            candidate_name = candidate.get("Name")
        elif cand_given or cand_sur:
            candidate_name = f"{cand_given or ''} {cand_sur or ''}".strip()

        # Extract other details
        candidate_id = candidate.get("PersonId", f"Unknown_{idx}")
        birth_year_str = candidate.get("BirthYear")
        birth_date = candidate.get("BirthDate", "N/A")
        birth_place = candidate.get("BirthPlace", "N/A")
        death_year_str = candidate.get("DeathYear")
        death_date = candidate.get("DeathDate", "N/A")
        death_place = candidate.get("DeathPlace", "N/A")
        gender = candidate.get("Gender", "N/A")
        gender_norm = (
            gender[0].lower()
            if gender and gender != "N/A" and gender[0].lower() in ["m", "f"]
            else None
        )
        is_living_suggestion = candidate.get(
            "IsLiving"
        )  # Check if suggest API provides this

        # Prepare candidate data for scoring
        try:
            birth_year_int = (
                int(birth_year_str)
                if birth_year_str and str(birth_year_str).isdigit()
                else None
            )
        except (ValueError, TypeError):
            birth_year_int = None
        try:
            death_year_int = (
                int(death_year_str)
                if death_year_str and str(death_year_str).isdigit()
                else None
            )
        except (ValueError, TypeError):
            death_year_int = None

        # Structure for calculate_match_score OR fallback scoring
        candidate_data_dict = {
            "first_name": clean_param(cand_given),
            "surname": clean_param(cand_sur),
            "birth_year": birth_year_int,
            "birth_date_obj": None,  # Not available from suggest API
            "birth_place": clean_param(birth_place),
            "death_year": death_year_int,
            "death_date_obj": None,  # Not available from suggest API
            "death_place": clean_param(death_place),
            "gender": gender_norm,
            "is_living": is_living_suggestion,  # Pass if available
            # Fields expected by calculate_match_score (duplicates some above)
            "norm_id": candidate_id,
            "display_id": candidate_id,
            "full_name_disp": candidate_name,
            "gender_norm": gender_norm,
            "birth_place_disp": clean_param(birth_place),
            "death_place_disp": clean_param(death_place),
        }

        # Calculate score using gedcom_utils function or fallback
        score = 0.0
        field_scores = {}
        reasons = []
        if scoring_function_available:
            try:
                score, field_scores, reasons = calculate_match_score(
                    search_criteria,
                    candidate_data_dict,  # Pass the structure expected
                    config_instance.COMMON_SCORING_WEIGHTS,
                    config_instance.NAME_FLEXIBILITY,
                    config_instance.DATE_FLEXIBILITY,
                )
            except Exception as score_err:
                logger.error(
                    f"Error calculating score for suggestion {candidate_id} using gedcom_utils: {score_err}",
                    exc_info=True,
                )
                # Fallback to simple scoring on error
                score, field_scores, reasons = _run_simple_suggestion_scoring(
                    search_criteria, candidate_data_dict
                )
                reasons.append("(Error Fallback)")
        else:
            # Use simple scoring if function wasn't available from start
            score, field_scores, reasons = _run_simple_suggestion_scoring(
                search_criteria, candidate_data_dict
            )

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
                "raw_data": candidate,  # Keep raw data for detail fetch
            }
        )

    # Sort by score descending
    processed_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    logger.info(f"Scored {len(processed_candidates)} candidates.")
    return processed_candidates


def _display_search_results(candidates: List[Dict]):
    """Displays the scored search results in a formatted table."""
    if not candidates:
        print("\nNo candidates to display.")
        return

    print(f"\n=== SEARCH RESULTS (Top {len(candidates)} Matches) ===")

    # Prepare data for tabulate
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

    for candidate in candidates:
        # Get field scores for display formatting
        fs = candidate.get("field_scores", {})
        givn_s = fs.get("givn", 0)
        surn_s = fs.get("surn", 0)
        bonus_s = fs.get("bonus", 0)
        gender_s = fs.get("gender", 0)
        byear_s = fs.get("byear", 0)
        bdate_s = fs.get("bdate", 0)  # Often 0 for suggestions
        bplace_s = fs.get("bplace", 0)
        dyear_s = fs.get("dyear", 0)
        ddate_s = fs.get("ddate", 0)  # Often 0 for suggestions
        dplace_s = fs.get("dplace", 0)

        name_disp = candidate.get("name", "N/A")[:25] + (
            "..." if len(candidate.get("name", "")) > 25 else ""
        )
        name_score_str = f"[{givn_s+surn_s}]" + (f"[+{bonus_s}]" if bonus_s > 0 else "")
        name_with_score = f"{name_disp} {name_score_str}"

        raw_gender = candidate.get("gender", "")
        gender_disp = (
            raw_gender.upper()
            if raw_gender and raw_gender.lower() in ["m", "f"]
            else (str(raw_gender) if raw_gender else "N/A")
        )
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
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    except Exception as tab_err:
        logger.error(f"Error formatting results table with tabulate: {tab_err}")
        print("\nError displaying table. Raw results:")
        for row in table_data:
            print(" | ".join(map(str, row)))
    print("")


def _select_top_candidate(
    scored_candidates: List[Dict], raw_suggestions: List[Dict]
) -> Optional[Tuple[Dict, Dict]]:
    """
    Selects the highest-scoring candidate and retrieves its original raw suggestion data.

    Args:
        scored_candidates: List of processed candidates, sorted by score.
        raw_suggestions: The original list of suggestions from the API.

    Returns:
        A tuple containing (top_candidate_processed, top_candidate_raw), or None if no candidates
        or raw data cannot be found.
    """
    if not scored_candidates:
        logger.info("No scored candidates available to select from.")
        return None

    top_scored_candidate = scored_candidates[0]
    top_scored_id = top_scored_candidate.get("id")

    if isinstance(top_scored_id, str) and top_scored_id.startswith("Unknown_"):
        logger.warning(
            f"Top candidate has a generated ID '{top_scored_id}'. Attempting to use its raw data directly."
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
        if str(suggestion.get("PersonId")) == str(top_scored_id):
            top_candidate_raw = suggestion
            logger.debug(f"Found raw data for top candidate ID {top_scored_id}")
            break

    if not top_candidate_raw:
        logger.error(
            f"Critical Error: Could not find raw suggestion data for the top scored candidate ID: {top_scored_id}. This should not happen unless the ID was invalid or missing from original suggestions."
        )
        print(
            f"\nError: Internal mismatch finding details for the top candidate ({top_scored_id})."
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
    """
    Fetches detailed person facts using the /facts/user/ API endpoint.
    Tries Cloudscraper first (with V16.15 headers), falls back to _api_req
    if Cloudscraper fails, addressing historical issues with this endpoint.

    Args:
        candidate_raw: The raw suggestion dictionary for the selected person.
        owner_profile_id: The profile ID (UserId) of the tree owner.
        session_manager: The active SessionManager instance.
        base_url: The base Ancestry URL.

    Returns:
        The parsed JSON dictionary's 'personResearch' section from the facts API,
        or None on failure.
    """
    api_person_id = candidate_raw.get("PersonId")
    api_tree_id = candidate_raw.get("TreeId")
    owner_tree_id = getattr(session_manager, "my_tree_id", None)  # Get owner's tree id

    if not all([api_person_id, api_tree_id, owner_profile_id]):
        logger.error(
            f"Facts API prerequisites missing (PersonId: {api_person_id}, TreeId: {api_tree_id}, OwnerUserId: {owner_profile_id})"
        )
        print("\nError: Missing essential IDs for fetching person details.")
        return None

    facts_api_url = f"{base_url}/family-tree/person/facts/user/{owner_profile_id.lower()}/tree/{api_tree_id}/person/{api_person_id}"
    logger.info(
        f"Attempting to fetch facts for PersonID {api_person_id} from {facts_api_url}"
    )

    facts_referer = (
        urljoin(
            base_url,
            f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts",
        )
        if owner_tree_id and owner_profile_id
        else base_url
    )
    # Headers based on V16.15 analysis (successful curl)
    facts_headers = {
        "accept": "application/json",
        "accept-language": "en-GB,en;q=0.9",
        "ancestry-context-ube": "eyJldmVudElkIjoiMDAwMDAwMDAtMDAwMC0wMDAwLTAwMDAtMDAwMDAwMDAwMDAwIiwiY29ycmVsYXRlZFNjcmVlblZpZXdlZElkIjoiMDM0YmNhYTMtMDIxYS00YmYyLTg2OTItNTg5ZDczMTc1ZDZjIiwiY29ycmVsYXRlZFNlc3Npb25JZCI6ImY3ZjA0OTA3LTdhYzMtNGFjMC05ZTRjLTdiODUzOGFjOWY3NCIsInVzZXJDb25zZW50IjoibmVjZXNzYXJ5fHByZWZlcmVuY2V8cGVyZm9ybWFuY2V8YW5hbHl0aWNzMXN0fGFuYWx5dGljczNyZHxhZHZlcnRpc2luZzFzdHxhZHZlcnRpc2luZzNyZHxhdHRyaWJ1dGlvbjNyZCIsInZlbmRvcnMiOiIiLCJ2ZW5kb3JDb25maWd1cmF0aW9ucyI6Int9In0=",  # Static example
        "cache-control": "no-cache",
        "content-type": "application/json",
        "dnt": "1",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": facts_referer,
        "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",  # Kept from V16.15
    }
    facts_timeout = 60  # Keep 60s timeout

    facts_data_raw = None
    method_used = "None"

    # --- Attempt 1: Cloudscraper for Facts API ---
    if session_manager.scraper:
        try:
            logger.info("Attempting Facts API call using Cloudscraper (Method 1)...")
            method_used = "Cloudscraper"
            scraper = session_manager.scraper
            logger.debug("Syncing cookies to Cloudscraper for Facts API call...")
            session_manager._sync_cookies()
            scraper.cookies.clear()
            for cookie in session_manager._requests_session.cookies:
                try:
                    scraper.cookies.set(
                        name=cookie.name,
                        value=cookie.value,
                        domain=cookie.domain,
                        path=cookie.path,
                    )
                except Exception as set_cookie_err:
                    logger.warning(
                        f"Failed to set cookie '{cookie.name}' for Cloudscraper: {set_cookie_err}"
                    )
            if not any(c.name == "ANCSESSIONID" for c in scraper.cookies):
                logger.warning(
                    "ANCSESSIONID cookie potentially missing after sync to Cloudscraper (Facts API)!"
                )

            logger.debug(f"Making Cloudscraper GET request to {facts_api_url}")
            logger.debug(
                f" Cloudscraper Headers: {json.dumps(facts_headers, indent=2)}"
            )
            logger.debug(f" Cloudscraper Timeout: {facts_timeout}s")
            facts_response = scraper.get(
                facts_api_url, headers=facts_headers, timeout=facts_timeout
            )
            logger.debug(f"Cloudscraper response status: {facts_response.status_code}")
            logger.debug(
                f"Cloudscraper response headers: {dict(facts_response.headers)}"
            )
            facts_response.raise_for_status()  # Check for HTTP errors

            content_type = facts_response.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                logger.warning(
                    f"Facts API (Cloudscraper) unexpected content type: {content_type}"
                )
                logger.debug(f"Response text: {facts_response.text[:500]}")
                raise ValueError("Non-JSON response from Cloudscraper Facts API")

            facts_data_raw = facts_response.json()
            logger.info("Facts API call successful using Cloudscraper.")

        except (
            cloudscraper.exceptions.CloudflareChallengeError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            requests.exceptions.RequestException,
            ValueError,
        ) as cs_err:
            logger.warning(f"Cloudscraper Facts API call failed: {cs_err}")
            if isinstance(cs_err, requests.exceptions.Timeout):
                logger.error(" -> Cloudscraper request timed out.")
            elif (
                isinstance(cs_err, requests.exceptions.HTTPError)
                and cs_err.response is not None
            ):
                logger.warning(f" -> Status Code: {cs_err.response.status_code}")
                logger.debug(f" -> Response Text: {cs_err.response.text[:500]}")
                if cs_err.response.status_code == 401:
                    logger.error(
                        " -> Facts API (Cloudscraper) returned 401 Unauthorized (Cookie/Session issue likely)."
                    )
                elif cs_err.response.status_code == 403:
                    logger.error(" -> Facts API (Cloudscraper) returned 403 Forbidden.")
            elif isinstance(cs_err, ValueError):
                logger.error(" -> Failed to decode JSON response from Cloudscraper.")
                logger.debug(
                    f" -> Cloudscraper Response Text (on JSON error): {getattr(facts_response, 'text', 'N/A')[:500]}"
                )
            facts_data_raw = None  # Ensure it's None for fallback

        except Exception as cs_gen_err:
            logger.error(
                f"Unexpected error during Cloudscraper Facts API call: {cs_gen_err}",
                exc_info=True,
            )
            facts_data_raw = None
    else:
        logger.warning(
            "Cloudscraper instance not available, skipping Cloudscraper attempt for Facts API."
        )

    # --- Attempt 2: Fallback using _api_req for Facts API ---
    if facts_data_raw is None:
        logger.warning(
            "Cloudscraper failed or was skipped for Facts API, attempting fallback using _api_req (Method 2)..."
        )
        method_used = "_api_req"
        try:
            # --- Pre-Call Logging for _api_req ---
            logger.info(f"--- Preparing _api_req for Facts API ---")
            logger.info(f"URL: {facts_api_url}")
            logger.info(
                f"Headers passed to _api_req: {json.dumps(facts_headers, indent=2)}"
            )
            logger.info(f"Timeout: {facts_timeout}")
            logger.info(f"Referer: {facts_referer}")
            logger.info(f"Session Valid: {session_manager.is_sess_valid()}")
            logger.info(f"Owner Profile ID (for facts): {owner_profile_id}")
            logger.info(f"Owner UUID (session): {session_manager.my_uuid}")
            csrf_log = (
                session_manager.csrf_token[:10] + "..."
                if session_manager.csrf_token and len(session_manager.csrf_token) > 20
                else session_manager.csrf_token
            )
            logger.info(f"CSRF Token (session, partial): {csrf_log}")
            try:
                cookie_names = [
                    c.name for c in session_manager._requests_session.cookies
                ]
                logger.info(
                    f"Requests Session Cookies ({len(cookie_names)}) before call: {sorted(cookie_names)}"
                )
            except Exception as e:
                logger.error(f"Could not log requests session cookies: {e}")
            # --- End Pre-Call Logging ---

            facts_data_raw = _api_req(
                url=facts_api_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                api_description="Person Facts API (_api_req fallback)",
                referer_url=facts_referer,
                timeout=facts_timeout,
                headers=facts_headers,
            )

            # --- Post-Call Logging for _api_req ---
            logger.info(f"--- _api_req response received for Facts API ---")
            logger.info(f"Response Type: {type(facts_data_raw)}")
            if isinstance(facts_data_raw, requests.Response):
                logger.warning(
                    f"Facts API _api_req returned raw Response object (Status: {facts_data_raw.status_code} {facts_data_raw.reason}). Expected dict."
                )
                try:
                    logger.debug(
                        f"Response Text (first 500): {facts_data_raw.text[:500]}"
                    )
                except Exception as e:
                    logger.error(f"Could not log response text: {e}")
                facts_data_raw = None  # Treat as failure
            elif facts_data_raw is None:
                logger.warning(
                    "Facts API response (_api_req) was None (check previous errors/timeouts in _api_req logs)."
                )
            elif isinstance(facts_data_raw, dict):
                logger.info(
                    f"Facts API response (_api_req) was a dictionary (Potential Success). Top-level keys: {list(facts_data_raw.keys())}"
                )
                logger.debug(
                    f"Facts API Response Dict Preview (_api_req): {json.dumps(facts_data_raw, indent=2, default=str)[:1000]}"
                )
            else:
                logger.warning(
                    f"Facts API response (_api_req) was an unexpected type. Value (first 500 chars): {str(facts_data_raw)[:500]}"
                )
                facts_data_raw = None  # Treat as failure
            # --- End Post-Call Logging ---

            if isinstance(facts_data_raw, dict):
                logger.info("Facts API call successful using _api_req fallback.")
            elif facts_data_raw is None:
                logger.error(
                    "Facts API call failed using _api_req fallback (returned None)."
                )
            else:
                logger.error(
                    f"Facts API call using _api_req fallback returned unexpected type: {type(facts_data_raw)}"
                )
                facts_data_raw = None  # Ensure failure path

        except requests.exceptions.Timeout as timeout_err:
            logger.error(
                f"Facts API call using _api_req fallback timed out after {facts_timeout}s: {timeout_err}"
            )
            print(
                f"\nError: Timed out waiting for person details from Ancestry (>{facts_timeout}s)."
            )
            facts_data_raw = None
        except Exception as api_req_err:
            logger.error(
                f"Facts API call using _api_req fallback failed unexpectedly: {api_req_err}",
                exc_info=True,
            )
            print(f"\nError fetching person details: {api_req_err}")
            facts_data_raw = None

    # --- Process Final Result ---
    if not isinstance(facts_data_raw, dict):
        logger.error(f"Failed to fetch facts data using {method_used} method.")
        print(
            f"\nError: Could not fetch valid person details from API (Method: {method_used}). Check logs."
        )
        return None

    # Extract the relevant section ('personResearch' based on curl output)
    person_research_data = facts_data_raw.get("data", {}).get("personResearch")

    if not isinstance(person_research_data, dict) or not person_research_data:
        logger.error(
            f"Facts API response (via {method_used}) is missing 'data.personResearch' dictionary."
        )
        logger.debug(f"Full raw response keys: {list(facts_data_raw.keys())}")
        if "data" in facts_data_raw:
            logger.debug(f"'data' sub-keys: {list(facts_data_raw['data'].keys())}")
        print(
            f"\nError: API response format for person details was unexpected (Missing personResearch section via {method_used})."
        )
        return None

    logger.info(
        f"Successfully fetched and parsed 'personResearch' data dictionary for PersonID {api_person_id} using {method_used}."
    )
    return person_research_data  # Return the personResearch dictionary


def _extract_detailed_info(person_research_data: Dict, candidate_raw: Dict) -> Dict:
    """
    Extracts detailed information (name, dates, places, family, etc.)
    from the 'personResearch' dictionary obtained from the facts API.

    Args:
        person_research_data: The 'personResearch' dictionary from _fetch_person_details.
        candidate_raw: The raw suggestion data (used for fallback name/IDs).

    Returns:
        A dictionary containing the extracted details.
    """
    extracted = {}
    logger.debug("Extracting details from person_research_data...")

    if not isinstance(person_research_data, dict):
        logger.error("Invalid input to _extract_detailed_info: expected a dictionary.")
        return {}

    person_facts_list = person_research_data.get("PersonFacts", [])
    if not isinstance(person_facts_list, list):
        person_facts_list = []
    logger.debug(f"Found {len(person_facts_list)} items in PersonFacts.")

    # --- Extract Name (Robustly) ---
    best_name = candidate_raw.get("FullName", candidate_raw.get("Name", "Unknown"))
    logger.debug(f"Initial name from suggestion: {best_name}")
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
        logger.debug(f"Found Name fact, formatted: '{formatted_nf}'")
        if formatted_nf != "Valued Relative":
            best_name = formatted_nf
    person_full_name = person_research_data.get("PersonFullName")
    if person_full_name and person_full_name != "Unknown":
        formatted_full_name = format_name(person_full_name)
        logger.debug(f"Found PersonFullName fact, formatted: '{formatted_full_name}'")
        if formatted_full_name != "Valued Relative" and (
            best_name == "Unknown" or len(formatted_full_name) > len(best_name)
        ):
            best_name = formatted_full_name
            logger.debug("Using PersonFullName as best name.")
    first_name_comp = person_research_data.get("FirstName", "")
    last_name_comp = person_research_data.get("LastName", "")
    if first_name_comp or last_name_comp:
        constructed_name = f"{first_name_comp} {last_name_comp}".strip()
        logger.debug(
            f"Found name components: First='{first_name_comp}', Last='{last_name_comp}', Constructed='{constructed_name}'"
        )
        if constructed_name and (
            best_name == "Unknown"
            or best_name == "Valued Relative"
            or len(constructed_name) > len(best_name)
        ):
            best_name = constructed_name
            logger.debug("Using constructed name as best name.")
    extracted["name"] = best_name
    logger.info(f"Final Extracted Name: {best_name}")

    # --- Extract Gender ---
    gender_str = person_research_data.get("PersonGender")
    logger.debug(f"Gender from PersonGender: {gender_str}")
    if not gender_str:
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

    # --- Extract Living Status ---
    extracted["is_living"] = person_research_data.get("IsPersonLiving", False)
    logger.info(f"Is Living: {extracted['is_living']}")

    # --- Extract Birth/Death ---
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

    # --- Extract Family Data ---
    extracted["family_data"] = person_research_data.get("PersonFamily", {})
    if not isinstance(extracted["family_data"], dict):  # Add validation
        logger.warning(
            "PersonFamily data is not a dictionary or is missing. Setting to empty."
        )
        extracted["family_data"] = {}
    logger.info(f"Family Data Keys: {list(extracted['family_data'].keys())}")

    # --- Extract IDs ---
    extracted["person_id"] = person_research_data.get(
        "PersonId", candidate_raw.get("PersonId")
    )
    extracted["tree_id"] = person_research_data.get(
        "TreeId", candidate_raw.get("TreeId")
    )
    extracted["user_id"] = person_research_data.get(
        "UserId", candidate_raw.get("UserId")
    )
    logger.info(
        f"IDs: PersonId='{extracted['person_id']}', TreeId='{extracted['tree_id']}', UserId='{extracted['user_id']}'"
    )

    # Add name components based on final best_name
    extracted["first_name"] = None
    extracted["surname"] = None
    name_parts = best_name.split()
    if name_parts and best_name not in ["Unknown", "Valued Relative"]:
        extracted["first_name"] = name_parts[0]
        if len(name_parts) > 1:
            extracted["surname"] = name_parts[-1]
    logger.debug(
        f"Extracted name components for scoring: First='{extracted['first_name']}', Sur='{extracted['surname']}'"
    )

    return extracted


def _score_detailed_match(
    extracted_info: Dict, search_criteria: Dict[str, Any], config_instance: Any
) -> Tuple[float, Dict, List[str]]:
    """
    Calculates the final match score based on the detailed information
    extracted from the facts API, using gedcom_utils or a simple fallback.

    Args:
        extracted_info: Dictionary from _extract_detailed_info.
        search_criteria: User's search criteria.
        config_instance: Configuration object with scoring weights.

    Returns:
        A tuple: (score, field_scores, reasons_list).
    """
    scoring_function_available = (
        GEDCOM_SCORING_AVAILABLE and calculate_match_score is not None
    )
    if not scoring_function_available:
        logger.warning(
            "Gedcom scoring function not available for detailed match. Using simple fallback scoring."
        )

    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)

    # Prepare data structure matching calculate_match_score expectation
    candidate_processed_data = {
        "norm_id": extracted_info.get("person_id"),
        "display_id": extracted_info.get("person_id"),
        "first_name": clean_param(
            extracted_info.get("first_name")
        ),  # Use extracted components
        "surname": clean_param(
            extracted_info.get("surname")
        ),  # Use extracted components
        "full_name_disp": extracted_info.get("name"),
        "gender_norm": extracted_info.get("gender"),
        "birth_year": extracted_info.get("birth_year"),
        "birth_date_obj": extracted_info.get("birth_date_obj"),
        "birth_place_disp": clean_param(
            extracted_info.get("birth_place")
        ),  # Use cleaned place
        "death_year": extracted_info.get("death_year"),
        "death_date_obj": extracted_info.get("death_date_obj"),
        "death_place_disp": clean_param(
            extracted_info.get("death_place")
        ),  # Use cleaned place
        "is_living": extracted_info.get("is_living", False),  # Pass living status
    }

    logger.debug(
        f"Detailed scoring - Search criteria: {json.dumps(search_criteria, default=str)}"
    )
    logger.debug(
        f"Detailed scoring - Candidate data: {json.dumps(candidate_processed_data, default=str)}"
    )

    score = 0.0
    field_scores = {}
    reasons_list = ["API Suggest Match"]  # Start with baseline reason

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
            if "API Suggest Match" not in reasons:
                reasons.insert(0, "API Suggest Match")
            reasons_list = reasons
            logger.info(f"Calculated detailed score via gedcom_utils: {score:.0f}")

        except Exception as e:
            logger.error(
                f"Error calculating detailed score using gedcom_utils: {e}",
                exc_info=True,
            )
            logger.warning(
                "Falling back to simple scoring for detailed match due to error."
            )
            # Pass extracted_info to simple scoring as it expects that structure
            score, field_scores, reasons_list = _run_simple_suggestion_scoring(
                search_criteria, extracted_info
            )
            reasons_list.append("(Detailed Scoring Error Fallback)")
    else:
        logger.debug("Calculating detailed score using simple fallback scoring...")
        # Pass extracted_info structure here as well
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
    # Use cleaned/extracted values for comparison, ensuring they are strings for printing
    cp = {
        "first_name": str(ei.get("first_name", na)),
        "surname": str(ei.get("surname", na)),
        "birth_year": str(ei.get("birth_year", na)),
        "birth_place": str(ei.get("birth_place", na)),
        "death_year": str(ei.get("death_year", na)),
        "death_place": str(ei.get("death_place", na)),
        "gender": str(ei.get("gender", na)),  # Note: 'm'/'f'/None converted to string
    }

    # Ensure search criteria values are also strings before formatting
    sc_fn = str(sc.get("first_name", na))
    sc_sn = str(sc.get("surname", na))
    sc_by = str(sc.get("birth_year", na))
    sc_bp = str(sc.get("birth_place", na))
    sc_dy = str(sc.get("death_year", na))
    sc_dp = str(sc.get("death_place", na))
    sc_gn = str(sc.get("gender", na))  # Note: 'm'/'f'/None converted to string

    # Display comparison using the guaranteed string variables - **REORDERED**
    print(f"  First Name : {sc_fn:<15} vs {cp['first_name']}")
    print(f"  Last Name  : {sc_sn:<15} vs {cp['surname']}")
    print(f"  Gender     : {sc_gn:<15} vs {cp['gender']}")  # Moved Gender up
    print(f"  Birth Year : {sc_by:<15} vs {cp['birth_year']}")
    print(f"  Birth Place: {sc_bp:<15} vs {cp['birth_place']}")
    print(f"  Death Year : {sc_dy:<15} vs {cp['death_year']}")
    print(
        f"  Death Place: {sc_dp:<15} vs {cp['death_place']}"
    )  # Apply formatting to sc_dp (string)

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
        # Check if list is None or empty
        if not rel_list:  # Handles None and empty list []
            print("    None found.")
            logger.info("    None found.")
            return

        # Ensure it's actually a list before iterating
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
                and ("-" in lifespan or "&ndash;" in lifespan)
            ):
                parts = lifespan.replace("&ndash;", "-").split("-")
                if len(parts) == 2:
                    b_date = parts[0].strip()
                    d_date = parts[1].strip()

            life_info = ""
            if b_date or d_date:
                life_info = f" (b. {b_date or '?'}"
                if d_date:
                    life_info += f", d. {d_date}"
                life_info += ")"
            elif lifespan and isinstance(lifespan, str):
                life_info = f" ({lifespan})"

            rel_info = f"- {name}{life_info}"
            print(f"    {rel_info}")
            logger.info(f"      {rel_info}")  # Indent log further
            found_any = True
        # End for loop

        if not found_any:  # If loop finished without printing anything valid
            print("    None found.")
            logger.info("    None found (list contained invalid entries).")

    # Structure from API: Fathers, Mothers, Siblings, Spouses, Children
    # Combine Fathers and Mothers for display
    parents_list = (family_data.get("Fathers") or []) + (
        family_data.get("Mothers") or []
    )
    siblings_list = family_data.get("Siblings")  # Should be a list
    spouses_list = family_data.get("Spouses")  # Should be a list

    # Children can be nested [[{child1}], [{child2}]] or flat [{child1}, {child2}] ?
    # The example shows [[{wayne}], [{wayne}]] which is strange (duplicates). Let's flatten robustly.
    children_flat_list = []
    children_raw = family_data.get("Children", [])
    if isinstance(children_raw, list):
        for child_entry in children_raw:
            if isinstance(child_entry, list):  # Nested list case
                for child_dict in child_entry:
                    if (
                        isinstance(child_dict, dict)
                        and child_dict not in children_flat_list
                    ):  # Avoid duplicates
                        children_flat_list.append(child_dict)
            elif isinstance(child_entry, dict):  # Flat list case
                if child_entry not in children_flat_list:  # Avoid duplicates
                    children_flat_list.append(child_entry)
            else:
                logger.warning(
                    f"Unexpected item type in Children list: {type(child_entry)}"
                )
    elif isinstance(children_raw, dict):  # Unlikely, but handle single child object
        if children_raw not in children_flat_list:
            children_flat_list.append(children_raw)
    else:
        logger.warning(f"Unexpected data type for 'Children': {type(children_raw)}")

    print_relatives("Parents", parents_list)
    print_relatives("Siblings", siblings_list)
    print_relatives("Spouses", spouses_list)
    print_relatives("Children", children_flat_list)


def _display_relationship_path(
    selected_info: Dict,
    owner_profile_id: Optional[str],
    owner_tree_id: Optional[str],
    owner_name: str,
    session_manager: SessionManager,
    base_url: str,
):
    """
    Calculates and displays the relationship path between the selected person
    and the tree owner using the appropriate Ancestry API. Includes robust
    parsing with fallbacks (api_utils, BeautifulSoup) and enhanced logging.
    Removes unreliable keyword search.

    Args:
        selected_info: Dictionary with details of the selected person (must include person_id, tree_id, user_id, name).
        owner_profile_id: The profile ID (UserId) of the tree owner.
        owner_tree_id: The tree ID of the owner's tree.
        owner_name: Display name of the tree owner.
        session_manager: The active SessionManager instance.
        base_url: The base Ancestry URL.
    """
    selected_person_tree_id = selected_info.get("person_id")  # Tree-specific ID
    selected_person_global_id = selected_info.get("user_id")  # Global ID (may be None)
    selected_tree_id = selected_info.get("tree_id")
    selected_name = selected_info.get("name", "Selected Person")

    print(f"\n--- Relationship Path to {owner_name} (API) ---")

    if not owner_profile_id:
        print("(Skipping relationship calculation: Owner profile ID not found)")
        logger.warning("Cannot calculate relationship: Owner profile ID missing.")
        return
    if not selected_person_tree_id and not selected_person_global_id:
        print("(Skipping relationship calculation: Selected person's IDs are missing)")
        logger.warning(
            "Cannot calculate relationship: Selected person's tree and global IDs are missing."
        )
        return

    is_owner = bool(
        selected_person_global_id
        and owner_profile_id
        and owner_profile_id.upper() == selected_person_global_id.upper()
    )
    if is_owner:
        print(f"({selected_name} is the Tree Owner)")
        logger.info(
            f"Selected person ({selected_name}) is the Tree Owner ({owner_name})."
        )
        return

    print(f"Calculating relationship path for {selected_name} to {owner_name}...")
    logger.info(
        f"Calculating relationship path for {selected_name} (TreeID: {selected_tree_id}, PersonID: {selected_person_tree_id}, GlobalID: {selected_person_global_id}) to {owner_name} (ProfileID: {owner_profile_id}, TreeID: {owner_tree_id})"
    )

    ladder_api_url = ""
    api_description = ""
    ladder_headers = {}
    ladder_referer = ""
    use_csrf = False
    force_text = False
    can_calculate = False

    if owner_tree_id and selected_tree_id == owner_tree_id and selected_person_tree_id:
        id_for_ladder = selected_person_tree_id
        ladder_api_url = f"{base_url}/family-tree/person/tree/{owner_tree_id}/person/{id_for_ladder}/getladder"
        api_description = "Get Tree Ladder API"
        callback_name = f"__ancestry_jsonp_{int(time.time()*1000)}"
        timestamp_ms = int(time.time() * 1000)
        query_params = urlencode({"callback": callback_name, "_": timestamp_ms})
        ladder_api_url = f"{ladder_api_url}?{query_params}"
        ladder_headers = {
            "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        }
        ladder_referer = f"{base_url}/family-tree/person/tree/{owner_tree_id}/person/{id_for_ladder}/facts"
        force_text = True
        can_calculate = True
        logger.debug(
            f"Using Tree Ladder API (getladder) with PersonID: {id_for_ladder}, URL: {ladder_api_url}"
        )
    elif selected_person_global_id:
        id_for_ladder = selected_person_global_id
        ladder_api_url = f"{base_url}/discoveryui-matchesservice/api/samples/{id_for_ladder}/relationshiptome/{owner_profile_id}"
        api_description = "API Relationship Ladder (Discovery)"
        uuid_for_referer = getattr(session_manager, "my_uuid", None)
        ladder_referer = (
            urljoin(base_url, f"/discoveryui-matches/list/{uuid_for_referer}")
            if uuid_for_referer
            else base_url
        )
        ladder_headers = {"Accept": "application/json"}
        can_calculate = True
        logger.debug(
            f"Using Discovery Ladder API (relationshiptome) with GlobalID: {id_for_ladder}, URL: {ladder_api_url}"
        )
    else:
        logger.error(
            "Cannot calculate relationship: Target is outside owner's tree, but their global UserId is missing."
        )
        print(
            "(Cannot calculate path: Target is outside your tree and their global ID was not found)."
        )
        return

    if not can_calculate or not ladder_api_url:
        logger.error(
            "Failed to determine a valid API endpoint for relationship calculation."
        )
        print(
            "(Internal error: Could not determine how to calculate relationship path)."
        )
        return

    relationship_data = None
    try:
        logger.info(f"Calling {api_description} at {ladder_api_url}")
        logger.debug(f" Headers: {ladder_headers}")
        logger.debug(f" Referer: {ladder_referer}")
        relationship_data = _api_req(
            url=ladder_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            api_description=f"{api_description} (Action 11)",
            headers=ladder_headers,
            referer_url=ladder_referer,
            use_csrf_token=use_csrf,
            force_text_response=force_text,
            timeout=20,
        )
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        print(f"(Error fetching relationship path from API: {e})")
        return

    if not relationship_data:
        logger.warning(
            f"API call '{api_description}' returned no data (None or empty)."
        )
        print("(API call for relationship returned no response or empty data)")
        return

    print("")
    print(f"  {selected_name}")
    logger.info(f"    {selected_name}")

    parsed_ok = False
    processed_text = ""
    # Define fallback message text consistently
    fallback_message_text = "(Could not parse relationship path from API)"

    # Attempt 1: Use format_api_relationship_path from api_utils
    if format_api_relationship_path:
        try:
            logger.debug("Attempting parsing with format_api_relationship_path...")
            # This function should ideally handle JSONP or HTML internally
            formatted_path = format_api_relationship_path(
                relationship_data, owner_name, selected_name
            )
            if formatted_path and not formatted_path.startswith(
                ("(No relationship", "(Could not parse")
            ):  # Check for known failure messages
                processed_text = formatted_path
                for line in formatted_path.splitlines():
                    if line.strip():
                        print(f"  {line}")
                parsed_ok = True
                logger.debug("Parsed successfully using format_api_relationship_path.")
            else:
                logger.warning(
                    f"format_api_relationship_path returned no path or error message: '{formatted_path}'"
                )
                # Log the input data to format_api_relationship_path for debugging api_utils
                rel_data_str_preview = str(relationship_data)[:500] + "..."
                logger.debug(
                    f"Input to format_api_relationship_path was (Type: {type(relationship_data)}): {rel_data_str_preview}"
                )
        except Exception as fmt_err:
            logger.warning(
                f"Error using format_api_relationship_path: {fmt_err}", exc_info=True
            )
            rel_data_str_preview = str(relationship_data)[:500] + "..."
            logger.debug(
                f"Input to format_api_relationship_path (on error) was (Type: {type(relationship_data)}): {rel_data_str_preview}"
            )

    # Attempt 2: Use BeautifulSoup for HTML parsing (if needed and possible)
    if (
        not parsed_ok
        and BeautifulSoup
        and isinstance(relationship_data, str)
        and ("<li" in relationship_data or "<div" in relationship_data)
    ):
        try:
            logger.debug("Attempting parsing with BeautifulSoup as fallback...")
            soup = BeautifulSoup(relationship_data, "html.parser")
            # Add more potential selectors based on observation
            path_elements = (
                soup.select("ul.textCenter li")
                or soup.select(".relationshipLadder li")
                or soup.select("li.relationshipStep")
                or soup.select("div.rel-step")
                or soup.select("div.relationItem")
                or soup.select("li[class*='relation']")
            )  # More generic class selector

            if path_elements:
                logger.debug(
                    f"Found {len(path_elements)} potential path elements with BS4."
                )
                steps_found = []
                start_index = (
                    1
                    if path_elements[0]
                    .get_text(strip=True)
                    .startswith(selected_name[:10])
                    else 0
                )
                if start_index == 1:
                    logger.debug(
                        "Skipping first BS4 element as it may repeat start person."
                    )

                for i, elem in enumerate(path_elements):
                    if i < start_index:
                        continue
                    step_text = elem.get_text(strip=True)
                    # Filter out empty steps or just the owner name
                    if (
                        step_text
                        and step_text != selected_name
                        and step_text != owner_name
                    ):
                        # Basic check to filter out obviously wrong steps like just "mother" or "father"
                        if len(step_text.split()) > 1 or step_text.lower() not in [
                            "mother",
                            "father",
                            "son",
                            "daughter",
                            "brother",
                            "sister",
                            "spouse",
                        ]:
                            steps_found.append(step_text)
                            print(f"  -> {step_text}")
                        else:
                            logger.debug(
                                f"Skipping potential BS4 step - looks like just a role: '{step_text}'"
                            )

                if steps_found:
                    processed_text = "\n".join([f"-> {s}" for s in steps_found])
                    parsed_ok = True
                    logger.debug("Parsed successfully using BeautifulSoup.")
                else:
                    logger.debug(
                        "BS4 found elements but no usable step text after filtering."
                    )
            else:
                logger.debug(
                    "BeautifulSoup found no common relationship path elements."
                )
        except Exception as bs_err:
            logger.warning(
                f"Error parsing relationship HTML with BeautifulSoup: {bs_err}",
                exc_info=True,
            )

    # Attempt 3: Keyword search REMOVED

    # Final check if parsing ultimately failed
    if not parsed_ok or not processed_text.strip():
        print(f"  {fallback_message_text}")
        processed_text = fallback_message_text  # Set for logging
        # Ensure parsed_ok is False if we ended up here
        parsed_ok = False

    # Log the final interpretation
    logger.info("    --- Relationship Path Interpretation ---")
    if processed_text.strip() and processed_text != fallback_message_text:
        for line in processed_text.splitlines():
            if line.strip():
                logger.info(f"    {line.strip()}")
    elif processed_text == fallback_message_text:
        logger.warning(f"    {fallback_message_text}")
    else:
        logger.warning(
            "    No relationship steps could be parsed or deduced (empty result)."
        )

    logger.info(f"    -> {owner_name} (Tree Owner)")
    logger.info("    ------------------------------------")

    # Log raw data for debugging if parsing failed or produced fallback/no text
    if (
        not parsed_ok
        or not processed_text.strip()
        or processed_text == fallback_message_text
    ):
        log_level = logging.WARNING if not parsed_ok else logging.DEBUG
        logger.log(
            log_level,
            f"Relationship parsing failed or produced no steps (parsed_ok={parsed_ok}). Logging raw response:",
        )
        rel_data_str = str(relationship_data)
        if rel_data_str.startswith("__ancestry_jsonp_"):
            try:
                json_part = rel_data_str[
                    rel_data_str.find("(") + 1 : rel_data_str.rfind(")")
                ]
                parsed_jsonp = json.loads(json_part)
                pretty_jsonp = json.dumps(parsed_jsonp, indent=2)
                logger.log(
                    log_level,
                    f"Raw JSONP data received (decoded):\n{pretty_jsonp[:2500]}{'...' if len(pretty_jsonp)>2500 else ''}",
                )
            except Exception as jsonp_err:
                logger.log(log_level, f"Failed to decode JSONP: {jsonp_err}")
                logger.log(
                    log_level,
                    f"Raw relationship data received (Type: {type(relationship_data)}): {rel_data_str[:2500]}{'...' if len(rel_data_str)>2500 else ''}",
                )
        else:
            logger.log(
                log_level,
                f"Raw relationship data received (Type: {type(relationship_data)}): {rel_data_str[:2500]}{'...' if len(rel_data_str)>2500 else ''}",
            )


# --- Main Handler ---
def handle_api_report():
    """
    Main handler for Action 11 - API Report. Orchestrates the process of
    searching, selecting, detailing, and relating a person via the Ancestry API.
    """
    logger.info(
        "\n--- Action 11: Person Details & Relationship (Ancestry API Report) ---"
    )

    missing_core = not CORE_UTILS_AVAILABLE
    missing_api_utils = not API_UTILS_AVAILABLE
    missing_gedcom_utils = not GEDCOM_UTILS_AVAILABLE
    missing_libs = not all([cloudscraper, requests])
    missing_config = config_instance is None

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
            logger.critical(" - API utils (api_utils.py) missing.")
        if missing_gedcom_utils:
            logger.critical(" - GEDCOM date utils (gedcom_utils.py) missing.")
        if missing_libs:
            logger.critical(" - Required libraries (requests/cloudscraper) missing.")
        if missing_config:
            logger.critical(" - Config instance missing.")
        print(
            "\nCRITICAL ERROR: Required libraries or utilities unavailable. Check logs, imports and installations."
        )
        return False

    print("Initializing Ancestry session...")
    if not session_manager.ensure_session_ready(action_name="API Report Session Init"):
        logger.error("Failed to initialize Ancestry session for API report.")
        print(
            "\nERROR: Failed to initialize session. Cannot proceed with API operations."
        )
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
        logger.warning(
            "Owner tree ID not found in session. Search may be restricted or fail."
        )

    search_criteria = _get_search_criteria()
    if not search_criteria:
        logger.info("Search criteria not provided or invalid. Exiting Action 11.")
        return True

    suggestions_raw = _search_ancestry_api(
        search_criteria, session_manager, owner_tree_id, owner_profile_id, base_url
    )
    if suggestions_raw is None:
        logger.error("API Search failed critically.")
        return False
    if not suggestions_raw:
        logger.info("API Search returned no results.")
        return True

    MAX_SUGGESTIONS_TO_SCORE = 50
    if len(suggestions_raw) > MAX_SUGGESTIONS_TO_SCORE:
        logger.warning(
            f"Processing only the top {MAX_SUGGESTIONS_TO_SCORE} of {len(suggestions_raw)} suggestions."
        )
        suggestions_to_score = suggestions_raw[:MAX_SUGGESTIONS_TO_SCORE]
    else:
        suggestions_to_score = suggestions_raw

    scored_candidates = _process_and_score_suggestions(
        suggestions_to_score, search_criteria, config_instance
    )
    if not scored_candidates:
        print("\nNo suitable candidates found after scoring.")
        logger.info("No candidates available after scoring process.")
        return True

    MAX_CANDIDATES_TO_DISPLAY = 10
    _display_search_results(scored_candidates[:MAX_CANDIDATES_TO_DISPLAY])

    selection = _select_top_candidate(scored_candidates, suggestions_raw)
    if not selection:
        print("\nFailed to select a top candidate (check logs for errors).")
        logger.error(
            "Failed to select top candidate, likely due to ID mismatch or missing raw data."
        )
        return False
    selected_candidate_processed, selected_candidate_raw = selection

    if not owner_profile_id:
        print(
            "\nCannot fetch details: Your Ancestry User ID (Profile ID) is required but missing from the session."
        )
        logger.error("Cannot fetch details: Owner profile ID missing.")
        return False

    # Pass person_research_data dictionary, not the whole raw response
    person_research_data = _fetch_person_details(
        selected_candidate_raw, owner_profile_id, session_manager, base_url
    )

    if not person_research_data:
        # If facts call failed (e.g., timeout even with Cloudscraper), attempt relationship path anyway
        print(
            "\nWarning: Failed to retrieve detailed information for the selected match."
        )
        logger.warning(
            "Proceeding to relationship path calculation despite failed detail fetch."
        )
        extracted_info_minimal = {
            "person_id": selected_candidate_raw.get("PersonId"),
            "tree_id": selected_candidate_raw.get("TreeId"),
            "user_id": selected_candidate_raw.get("UserId"),
            "name": selected_candidate_processed.get("name", "Selected Person"),
        }
        _display_relationship_path(
            extracted_info_minimal,
            owner_profile_id,
            owner_tree_id,
            owner_name,
            session_manager,
            base_url,
        )
        logger.warning("Action 11 finished, but with errors retrieving person details.")
        return True

    # --- Process Successful Detail Fetch ---
    # Pass person_research_data to extraction function
    extracted_info = _extract_detailed_info(
        person_research_data, selected_candidate_raw
    )
    if not extracted_info:  # Handle case where extraction fails
        print("\nError: Failed to extract details from API response.")
        logger.error(
            "Failed to extract details even though API call seemed successful."
        )
        return False

    final_score_info = _score_detailed_match(
        extracted_info, search_criteria, config_instance
    )
    _display_detailed_match_info(
        extracted_info, final_score_info, search_criteria, base_url
    )
    _display_family_info(extracted_info.get("family_data", {}))
    _display_relationship_path(
        extracted_info,
        owner_profile_id,
        owner_tree_id,
        owner_name,
        session_manager,
        base_url,
    )

    return True  # Report completed successfully


# --- Main Execution ---
def main():
    """Main execution flow for Action 11 (API Report)."""
    logger.info("--- Action 11: API Report Starting ---")
    report_successful = handle_api_report()
    if report_successful:
        logger.info("--- Action 11: API Report Finished ---")
        print("\nAction 11 finished.")
    else:
        logger.error("--- Action 11: API Report Finished with Critical Errors ---")
        print("\nAction 11 finished with critical errors (check logs).")


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
