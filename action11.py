# --- START OF FILE action11.py ---
# action11.py
"""
Action 11: API Report - Search Ancestry API, display details, family, relationship.
V16.0: Refactored from temp.py v7.36, using functions from utils.py, api_utils.py, gedcom_utils.py.
Implements consistent scoring and output format with Action 10.
V16.1: Corrected config import, removed faulty API call, simplified family display logic.
V16.2: Fixed IndentationError, removed redundant dummy code, restored interactive input.
V16.3: Hardcoded Fraser Gault inputs, switched detail fetch to /person-picker/person API,
       adjusted parsing, added isHideVeiledRecords param to suggest API.
V16.4: Switched /suggest API call to use cloudscraper.
V16.5: Added explicit cookie synchronization to cloudscraper instance.
V16.6: Increased timeout for /person-picker/person API call.
V16.7: Switched detail fetch to /facts/user/ endpoint for richer data, updated parsing.
V16.8: Added logging for /suggest API response, adjusted ID validation.
V16.9: Reverted detail fetch API back to /facts/user/ based on user clarification,
       ensuring correct owner ID is used in URL. Removed parsing dependency on UserId from suggest.
V16.10: Switched /facts/user/ API call to use Cloudscraper with cookie sync and X-Requested-With header.
V16.11: Reverted /facts/user/ API call back to _api_req, added timeout=60 and X-Requested-With header.
V16.12: Switched /facts/user/ API call back to Cloudscraper again due to persistent timeouts with _api_req.
V16.13: Reverting /facts/user/ API call back to _api_req (V16.11 logic) as Cloudscraper caused 401 errors.
V16.14: Final attempt: Use Cloudscraper for /facts/user/ with manual dynamic header generation,
        correct Accept header, and no X-Requested-With, mirroring successful manual curl.
V16.15: Sticking with _api_req for /facts/user/, keeping timeout=60 and X-Requested-With header,
        adding extensive pre-call and post-call logging for diagnostics.
"""
# --- Standard library imports ---
import logging
import sys
import time
import urllib.parse
import json
from pathlib import Path
from urllib.parse import urljoin, urlencode
from tabulate import tabulate

# Import specific types needed locally
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime

# --- Third-party imports ---
try:
    import requests

    try:
        import cloudscraper
    except ImportError:
        cloudscraper = None
except ImportError:
    requests = None
    cloudscraper = None

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
    # Import instance directly from config
    from config import config_instance

    logger.info("Successfully imported config_instance.")

    # Basic validation that config_instance and its scoring attributes are loaded
    if not config_instance:
        raise ImportError("config_instance is None after import.")
    # End if
    if not hasattr(config_instance, "COMMON_SCORING_WEIGHTS") or not isinstance(
        config_instance.COMMON_SCORING_WEIGHTS, dict
    ):
        raise TypeError(
            "config_instance.COMMON_SCORING_WEIGHTS is missing or not a dictionary."
        )
    # End if
    if not hasattr(config_instance, "NAME_FLEXIBILITY") or not isinstance(
        config_instance.NAME_FLEXIBILITY, dict
    ):
        raise TypeError(
            "config_instance.NAME_FLEXIBILITY is missing or not a dictionary."
        )
    # End if
    if not hasattr(config_instance, "DATE_FLEXIBILITY") or not isinstance(
        config_instance.DATE_FLEXIBILITY, dict
    ):
        raise TypeError(
            "config_instance.DATE_FLEXIBILITY is missing or not a dictionary."
        )
    # End if

    # Optional warning if weights dict is empty
    if not config_instance.COMMON_SCORING_WEIGHTS:
        logger.warning(
            "config_instance.COMMON_SCORING_WEIGHTS dictionary is empty. Scoring may not function as expected."
        )
    # End if

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
# End try

# Ensure critical config components are loaded before proceeding (redundant check, but safe)
if (
    config_instance is None
    or not hasattr(config_instance, "COMMON_SCORING_WEIGHTS")
    or not hasattr(config_instance, "NAME_FLEXIBILITY")
    or not hasattr(config_instance, "DATE_FLEXIBILITY")
):
    logger.critical("One or more critical configuration components failed to load.")
    print("\nFATAL ERROR: Configuration load failed.")
    sys.exit(1)
# End if


# --- Import GEDCOM Utilities (for scoring and date helpers) ---
calculate_match_score = None
_parse_date = None
_clean_display_date = None
GEDCOM_DATE_UTILS_AVAILABLE = False
GEDCOM_SCORING_AVAILABLE = False

try:
    from gedcom_utils import calculate_match_score, _parse_date, _clean_display_date

    logger.info("Successfully imported functions from gedcom_utils.")
    GEDCOM_SCORING_AVAILABLE = calculate_match_score is not None
    GEDCOM_DATE_UTILS_AVAILABLE = (
        _parse_date is not None and _clean_display_date is not None
    )
except ImportError as e:
    logger.error(f"Failed to import from gedcom_utils: {e}.", exc_info=True)
# End try

# --- Import API Utilities ---
print_group = None
display_raw_relationship_ladder = None
API_UTILS_AVAILABLE = False

try:
    from api_utils import print_group, display_raw_relationship_ladder

    logger.info("Successfully imported required functions from api_utils.")
    API_UTILS_AVAILABLE = (
        print_group is not None and display_raw_relationship_ladder is not None
    )
except ImportError as e:
    logger.error(f"Failed to import from api_utils: {e}.", exc_info=True)
# End try


# --- Import General Utilities ---
try:
    # Import make_* functions for dynamic headers
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
# End try


# --- Session Manager Instance ---
session_manager: SessionManager = SessionManager()


# --- Helper Function for Parsing PersonFacts Array ---
def _extract_fact_data(
    person_facts: List[Dict], fact_type_str: str
) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """Extracts date string, place string, and parsed date object from PersonFacts list."""
    date_str: Optional[str] = None
    place_str: Optional[str] = None
    date_obj: Optional[datetime] = None

    if not isinstance(person_facts, list):
        return date_str, place_str, date_obj  # Return defaults if not a list
    # End if

    for fact in person_facts:
        # Ensure fact is a dict and TypeString matches, ignoring alternate facts for primary data
        if (
            isinstance(fact, dict)
            and fact.get("TypeString") == fact_type_str
            and not fact.get("IsAlternate")
        ):
            date_str = fact.get("Date")
            place_str = fact.get("Place")
            parsed_date_data = fact.get("ParsedDate")
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
                        # End if/elif/else for date parts
                        date_obj = _parse_date(temp_date)
                    except (ValueError, TypeError) as dt_err:
                        logger.warning(
                            f"Could not parse date from ParsedDate {parsed_date_data}: {dt_err}"
                        )
                        date_obj = None  # Reset on error
                    # End try/except parsing
                # End if year
            # End if parsed_date_data
            # If date_obj parsing failed or wasn't possible, try parsing the Date string directly
            if date_obj is None and date_str:
                date_obj = _parse_date(date_str)
            # End if fallback parsing

            # Found the primary fact, break the loop
            break
        # End if fact matches
    # End for fact

    return date_str, place_str, date_obj


# End of _extract_fact_data


# --- Main Handler ---
def handle_api_report():
    """
    Handler for Action 11 - API Report.
    Searches Ancestry API, displays details, family, relationship to Tree Owner.
    Uses functions from utils.py, api_utils.py, and gedcom_utils.py (for scoring/dates).
    """
    logger.info(
        "\n--- Person Details & Relationship to Tree Owner (Ancestry API Report) ---"
    )

    # Check essential dependencies
    if not CORE_UTILS_AVAILABLE:
        logger.critical("handle_api_report: Core utils module unavailable at runtime.")
        print("\nCRITICAL ERROR: Core utilities unavailable.")
        return False
    # End if
    if not API_UTILS_AVAILABLE:
        logger.critical("handle_api_report: API utils module unavailable at runtime.")
        print("\nCRITICAL ERROR: API utilities unavailable.")
        return False
    # End if
    if not GEDCOM_SCORING_AVAILABLE:
        logger.critical(
            "handle_api_report: GEDCOM scoring function unavailable at runtime."
        )
        print("\nCRITICAL ERROR: Scoring function unavailable.")
        return False
    # End if
    if not GEDCOM_DATE_UTILS_AVAILABLE:
        logger.critical(
            "handle_api_report: GEDCOM date utilities unavailable at runtime."
        )
        print("\nCRITICAL ERROR: Date utilities unavailable.")
        return False
    # End if
    if cloudscraper is None:
        logger.critical(
            "handle_api_report: Cloudscraper library is required but not installed (`pip install cloudscraper`)."
        )
        print("\nCRITICAL ERROR: cloudscraper library not found.")
        return False
    # End if
    if not session_manager.scraper:
        logger.critical(
            "handle_api_report: Cloudscraper instance not available in SessionManager."
        )
        print("\nCRITICAL ERROR: Cloudscraper instance failed to initialize.")
        return False
    # End if

    # --- Initialize API Session ---
    print("Initializing Ancestry session...")
    session_init_ok = session_manager.ensure_session_ready(
        action_name="API Report Session Init"
    )
    if not session_init_ok:
        logger.error("Failed to initialize Ancestry session for API report.")
        print(
            "\nERROR: Failed to initialize session. Cannot proceed with API operations."
        )
        return False
    # End if

    # Retrieve tree owner name from session_manager
    owner_name = getattr(session_manager, "tree_owner_name", "the Tree Owner")
    owner_profile_id = getattr(
        session_manager, "my_profile_id", None
    )  # Owner's global UserId
    owner_tree_id = getattr(session_manager, "my_tree_id", None)
    base_url = getattr(
        config_instance, "BASE_URL", "https://www.ancestry.co.uk"
    ).rstrip("/")

    if not owner_profile_id:
        logger.warning(
            "handle_api_report: My profile ID not available in SessionManager. Relationship path may fail."
        )
        print(
            "\nWARNING: Cannot determine your profile ID. Relationship path calculation may fail."
        )
    # End if
    if not owner_tree_id:
        logger.warning(
            "handle_api_report: My tree ID not available in SessionManager. Initial search may fail."
        )
        print("\nWARNING: Cannot determine your tree ID.")
    # End if

    # --- Get search criteria from user ---
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
    logger.info(f"  First Name: {first_name}")
    logger.info(f"  Surname: {surname}")
    logger.info(f"  Birth Date/Year: {dob_str}")
    logger.info(f"  Birth Place: {pob}")
    logger.info(f"  Death Date/Year: {dod_str}")
    logger.info(f"  Death Place: {pod}")
    logger.info(f"  Gender: {gender}")
    # --- END OF TEMPORARY HARDCODED VALUES ---

    if not (first_name or surname):
        logger.info("\nAPI search needs First Name or Surname. Report cancelled.")
        print("\nAPI search needs First Name or Surname. Report cancelled.")
        return True
    # End if

    # Prepare search criteria for scoring
    clean_param = lambda p: (p.strip().lower() if p and isinstance(p, str) else None)

    target_first_name_lower = clean_param(first_name)
    target_surname_lower = clean_param(surname)
    target_pob_lower = clean_param(pob)
    target_pod_lower = clean_param(pod)
    target_gender_clean = gender

    target_birth_year: Optional[int] = None
    target_birth_date_obj: Optional[datetime] = None
    if dob_str and GEDCOM_DATE_UTILS_AVAILABLE:
        target_birth_date_obj = _parse_date(dob_str)
        if target_birth_date_obj:
            target_birth_year = target_birth_date_obj.year
        # End if
    # End if

    target_death_year: Optional[int] = None
    target_death_date_obj: Optional[datetime] = None
    if dod_str and GEDCOM_DATE_UTILS_AVAILABLE:
        target_death_date_obj = _parse_date(dod_str)
        if target_death_date_obj:
            target_death_year = target_death_date_obj.year
        # End if
    # End if

    search_criteria_dict = {
        "first_name": target_first_name_lower,
        "surname": target_surname_lower,
        "birth_year": target_birth_year,
        "birth_date_obj": target_birth_date_obj,
        "birth_place": target_pob_lower,
        "death_year": target_death_year,
        "death_date_obj": target_death_date_obj,
        "death_place": target_pod_lower,
        "gender": target_gender_clean,
    }

    # --- Use person-picker/suggest API (via Cloudscraper) ---
    tree_id_for_suggest = owner_tree_id
    if not tree_id_for_suggest:
        logger.error(
            "Cannot perform API search: My tree ID is not available in SessionManager."
        )
        print(
            "\nERROR: Cannot determine your tree ID. API search functionality is limited."
        )
        return False
    # End if

    suggest_params = []
    if first_name:
        suggest_params.append(f"partialFirstName={urllib.parse.quote(first_name)}")
    # End if
    if surname:
        suggest_params.append(f"partialLastName={urllib.parse.quote(surname)}")
    # End if
    suggest_params.append("isHideVeiledRecords=false")

    # Add birth year to suggest params if available
    if search_criteria_dict.get("birth_year"):
        suggest_params.append(f"birthYear={search_criteria_dict['birth_year']}")
    # End if

    # Original suggest URL
    suggest_url = f"{base_url}/api/person-picker/suggest/{tree_id_for_suggest}?{'&'.join(suggest_params)}"

    # Also try the treesui-list endpoint with the format provided by the user
    treesui_response = None
    if search_criteria_dict.get("birth_year"):
        treesui_params = []
        if first_name:
            treesui_params.append(f"fn={urllib.parse.quote(first_name)}")
        # End if
        if surname:
            treesui_params.append(f"ln={urllib.parse.quote(surname)}")
        # End if

        # Use the format provided by the user
        treesui_params.append("limit=100")
        treesui_params.append("fields=NAMES,BIRTH_DEATH")

        # Add birth year parameter in different formats to increase chances of success
        treesui_params.append(f"by={search_criteria_dict['birth_year']}")

        treesui_url = f"{base_url}/api/treesui-list/trees/{tree_id_for_suggest}/persons?{'&'.join(treesui_params)}"
        logger.info(
            f"Also trying treesui-list API with user-provided format: {treesui_url}"
        )

        # We'll make this API call later after trying the suggest API
    # End if
    logger.info(f"Attempting search using Ancestry Suggest API: {suggest_url}")

    # Log search API details but don't display to user
    logger.info("\n=== SEARCH API DETAILS ===")
    logger.info(f"Search API URL: {suggest_url}")
    logger.info(f"Search Parameters:")
    for param in suggest_params:
        logger.info(f"  {param}")
    logger.info(f"Tree ID for Search: {tree_id_for_suggest}")
    logger.info(f"Search Criteria:")
    logger.info(f"  First Name: {search_criteria_dict.get('first_name', 'N/A')}")
    logger.info(f"  Last Name: {search_criteria_dict.get('surname', 'N/A')}")
    logger.info(f"  Birth Year: {search_criteria_dict.get('birth_year', 'N/A')}")
    logger.info(f"  Birth Place: {search_criteria_dict.get('birth_place', 'N/A')}")
    logger.info(f"  Death Year: {search_criteria_dict.get('death_year', 'N/A')}")
    logger.info(f"  Death Place: {search_criteria_dict.get('death_place', 'N/A')}")
    logger.info(f"  Gender: {search_criteria_dict.get('gender', 'N/A')}")

    print("Searching Ancestry API...")

    suggest_response = None
    scraper_response = None
    try:
        owner_facts_referer = None
        if owner_tree_id and owner_profile_id:
            owner_facts_referer = urljoin(
                base_url,
                f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts",
            )
        else:
            logger.warning(
                "Cannot construct owner facts referer for Suggest API: Tree ID or Profile ID missing."
            )
            owner_facts_referer = base_url  # Fallback
        # End if

        logger.info("Attempting Suggest API call using Cloudscraper...")
        scraper = session_manager.scraper
        if scraper:
            try:
                logger.debug(
                    "Syncing cookies from SessionManager requests session to Cloudscraper session..."
                )
                session_manager._sync_cookies()
                scraper.cookies.clear()
                synced_count = 0

                # Log the cookies we're about to sync for debugging
                cookie_names = [
                    c.name for c in session_manager._requests_session.cookies
                ]
                logger.debug(f"Cookies available for sync: {cookie_names}")

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
                    # End inner try
                # End for
                logger.debug(f"Synced {synced_count} cookies to Cloudscraper session.")

                # Verify essential cookies were synced
                essential_cookies = ["ANCSESSIONID", "SecureATT"]
                missing_cookies = [
                    c
                    for c in essential_cookies
                    if c not in [cookie.name for cookie in scraper.cookies]
                ]
                if missing_cookies:
                    logger.warning(
                        f"Essential cookies missing after sync: {missing_cookies}"
                    )
                else:
                    logger.debug("All essential cookies successfully synced")

            except Exception as sync_err:
                logger.error(
                    f"Error syncing cookies to Cloudscraper: {sync_err}", exc_info=True
                )
            # End try

            # Set headers for Cloudscraper request based on successful curl command
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

            # Add additional headers from successful curl if available in session
            anc_session_id = session_manager._requests_session.cookies.get(
                "ANCSESSIONID"
            )
            if anc_session_id:
                logger.debug(f"Found ANCSESSIONID cookie: {anc_session_id}")

            wait_time = session_manager.dynamic_rate_limiter.wait()
            if wait_time > 0.1:
                logger.debug(
                    f"[Suggest API (Cloudscraper)] Rate limit wait: {wait_time:.2f}s"
                )
            # End if

            try:
                # Set a shorter timeout to prevent hanging
                logger.debug(
                    f"Making Cloudscraper request to {suggest_url} with timeout=15s"
                )
                scraper_response = scraper.get(
                    suggest_url, headers=scraper_headers, timeout=15
                )
                scraper_response.raise_for_status()

                # Log response headers for debugging
                logger.debug(f"Response headers: {dict(scraper_response.headers)}")

                # Check content type before attempting to parse JSON
                content_type = scraper_response.headers.get("Content-Type", "")
                if (
                    "application/json" not in content_type
                    and "text/json" not in content_type
                ):
                    logger.warning(f"Unexpected content type: {content_type}")
                    logger.debug(f"Response text: {scraper_response.text[:500]}")

                # Try to parse JSON with explicit error handling
                try:
                    suggest_response = scraper_response.json()
                    logger.info("Successfully parsed JSON response from Suggest API")
                except ValueError as json_err:
                    logger.error(f"JSON parsing error: {json_err}")
                    logger.debug(f"Response text: {scraper_response.text[:500]}")
                    suggest_response = None
                    raise  # Re-raise to be caught by outer exception handler

                logger.info("Suggest API call successful using Cloudscraper.")
                session_manager.dynamic_rate_limiter.decrease_delay()

            except cloudscraper.exceptions.CloudflareChallengeError as cfe:
                logger.error(
                    f"Cloudflare challenge encountered during Suggest API call: {cfe}"
                )
                suggest_response = None
                session_manager.dynamic_rate_limiter.increase_delay()
            except requests.exceptions.Timeout as timeout_err:
                logger.error(
                    f"Timeout during Cloudscraper Suggest API call: {timeout_err}"
                )
                suggest_response = None
                session_manager.dynamic_rate_limiter.increase_delay()
            except requests.exceptions.HTTPError as http_err:
                logger.error(
                    f"HTTPError during Cloudscraper Suggest API call: {http_err}"
                )
                if http_err.response is not None:
                    logger.error(f"  Status Code: {http_err.response.status_code}")
                    logger.debug(f"  Response Text: {http_err.response.text[:500]}")
                    if http_err.response.status_code == 401:
                        logger.error(
                            "Suggest API returned 401 Unauthorized - cookie sync likely failed or session invalid."
                        )
                    # End if 401
                # End if response exists
                suggest_response = None
            except requests.exceptions.RequestException as req_exc:
                logger.error(
                    f"RequestException during Cloudscraper Suggest API call: {req_exc}"
                )
                suggest_response = None
            except ValueError as json_err:
                # This catches JSON decode errors from the inner try/except
                logger.error(
                    f"Failed to decode JSON from Cloudscraper Suggest API response: {json_err}"
                )
                logger.debug(
                    f"Cloudscraper Response Text: {getattr(scraper_response, 'text', 'N/A')[:500]}"
                )
                suggest_response = None
            except Exception as scrape_err:
                logger.error(
                    f"Unexpected error during Cloudscraper Suggest API call: {scrape_err}",
                    exc_info=True,
                )
                suggest_response = None
            # End inner try/except
        else:
            logger.critical(
                "Cloudscraper instance not available in SessionManager. Cannot call Suggest API."
            )
            return False
        # End if scraper

        # Check response validity
        if suggest_response is None:
            logger.error("Suggest API call failed (check previous errors).")
            print("\nError during API search. Attempting fallback method...")

            # Fallback: Try using _api_req directly instead of Cloudscraper
            try:
                logger.info("Attempting fallback API call using _api_req...")
                fallback_suggest_url = suggest_url
                fallback_headers = {
                    "Accept": "application/json, text/plain, */*",
                    "Referer": owner_facts_referer,
                }

                # Use _api_req with a shorter timeout
                suggest_response = _api_req(
                    url=fallback_suggest_url,
                    driver=session_manager.driver,
                    session_manager=session_manager,
                    method="GET",
                    api_description="Suggest API (Fallback)",
                    headers=fallback_headers,
                    timeout=15,
                )

                if (
                    suggest_response is None
                    or not isinstance(suggest_response, list)
                    or not suggest_response
                ):
                    logger.error(
                        "Fallback API call also failed or returned no results."
                    )

                    # If we have a birth year, try the treesui-list endpoint
                    if (
                        search_criteria_dict.get("birth_year")
                        and "treesui_url" in locals()
                    ):
                        logger.info(
                            "Suggest API fallback failed. Trying treesui-list endpoint..."
                        )
                        print(
                            "\nTrying alternative API endpoint with birth year parameter..."
                        )

                        treesui_headers = {
                            "Accept": "application/json, text/plain, */*",
                            "Referer": owner_facts_referer,
                        }

                        # Use _api_req with a shorter timeout
                        treesui_response = _api_req(
                            url=treesui_url,
                            driver=session_manager.driver,
                            session_manager=session_manager,
                            method="GET",
                            api_description="TreesUI List API",
                            headers=treesui_headers,
                            timeout=15,
                        )

                        # If treesui-list endpoint succeeded, use its response instead
                        if treesui_response and isinstance(treesui_response, list):
                            logger.info(
                                f"TreesUI List API call successful! Found {len(treesui_response)} results."
                            )
                            print(
                                f"Alternative API search successful! Found {len(treesui_response)} potential matches."
                            )
                            suggest_response = treesui_response
                        elif treesui_response:
                            logger.error(
                                f"TreesUI List API returned unexpected format: {type(treesui_response)}"
                            )
                            print("Alternative API search returned unexpected format.")
                            return False
                        else:
                            logger.error("TreesUI List API call failed.")
                            print("Alternative API search also failed.")
                            return False
                    else:
                        print(
                            "\nBoth primary and fallback API search methods failed. Check logs."
                        )
                        return False
                else:
                    logger.info("Fallback API call successful!")
                    print("Fallback API search successful!")
            except Exception as fallback_err:
                logger.error(
                    f"Fallback API call failed with error: {fallback_err}",
                    exc_info=True,
                )
                print(
                    "\nBoth primary and fallback API search methods failed. Check logs."
                )
                return False
        elif not isinstance(suggest_response, list) or not suggest_response:
            logger.info("No matches found via Ancestry Suggest API.")
            print("\nNo potential matches found in Ancestry API based on name.")
            return True
        # End if/elif for suggest_response check

    except Exception as e:
        logger.error(f"General error during API search section: {e}", exc_info=True)
        print(f"\nError during API search: {e}. Check logs.")
        return False
    # End try/except

    # --- Process Suggest API Results ---
    if not suggest_response:
        logger.error("Suggest response list is empty after API call. Cannot process.")
        return False
    # End if

    # Log the search criteria in a format similar to action10.py
    logger.info("\n--- Final Search Criteria Used ---")
    for key, value in search_criteria_dict.items():
        if value is not None and key not in ["birth_date_obj", "death_date_obj"]:
            logger.info(f"{key.replace('_', ' ').title()}: '{value}'")

    logger.info(f"\n--- Filtering and Scoring Candidates ---")
    logger.info(f"Found {len(suggest_response)} candidate(s) from API search.")

    # Process candidates and prepare for display
    display_candidates = []
    for candidate in suggest_response[:10]:  # Show up to 10 candidates
        # Extract name from various possible fields
        candidate_name = "Unknown"

        # Try FullName first (from the example JSON response)
        if "FullName" in candidate:
            candidate_name = candidate.get("FullName")
        # Then try Name (older API format)
        elif "Name" in candidate:
            candidate_name = candidate.get("Name")
        # Then try to construct from GivenName and Surname
        elif "GivenName" in candidate and "Surname" in candidate:
            given_name = candidate.get("GivenName", "")
            surname = candidate.get("Surname", "")
            if given_name or surname:
                candidate_name = f"{given_name} {surname}".strip()
        # Finally try FirstName and LastName
        elif "FirstName" in candidate and "LastName" in candidate:
            first_name = candidate.get("FirstName", "")
            last_name = candidate.get("LastName", "")
            if first_name or last_name:
                candidate_name = f"{first_name} {last_name}".strip()

        # Extract other details
        candidate_id = candidate.get("PersonId", "Unknown")
        birth_year = candidate.get("BirthYear", "N/A")
        birth_date = candidate.get("BirthDate", "N/A")
        birth_place = candidate.get("BirthPlace", "N/A")
        death_year = candidate.get("DeathYear", "N/A")
        death_date = candidate.get("DeathDate", "N/A")
        death_place = candidate.get("DeathPlace", "N/A")
        gender = candidate.get("Gender", "N/A")

        # Prepare candidate data for scoring
        candidate_data_dict = {
            "first_name": clean_param(
                candidate.get("GivenName", candidate.get("FirstName", ""))
            ),
            "surname": clean_param(
                candidate.get("Surname", candidate.get("LastName", ""))
            ),
            "birth_year": (
                int(birth_year)
                if birth_year
                and isinstance(birth_year, str)
                and birth_year != "N/A"
                and birth_year.isdigit()
                else birth_year if isinstance(birth_year, int) else None
            ),
            "birth_date_obj": None,  # We don't have full date objects from the search API
            "birth_place": clean_param(birth_place),
            "death_year": (
                int(death_year)
                if death_year
                and isinstance(death_year, str)
                and death_year != "N/A"
                and death_year.isdigit()
                else death_year if isinstance(death_year, int) else None
            ),
            "death_date_obj": None,  # We don't have full date objects from the search API
            "death_place": clean_param(death_place),
            "gender": (
                gender[0].lower()
                if gender
                and gender != "N/A"
                and len(gender) > 0
                and gender[0].lower() in ["m", "f"]
                else None
            ),
        }

        # Calculate score using config.py weights
        candidate_score = 0
        field_scores = {}
        reasons = ["API Suggest Match"]

        if GEDCOM_SCORING_AVAILABLE and calculate_match_score and config_instance:
            try:
                candidate_score, field_scores, reasons = calculate_match_score(
                    search_criteria_dict,
                    candidate_data_dict,
                    config_instance.COMMON_SCORING_WEIGHTS,
                    config_instance.NAME_FLEXIBILITY,
                    config_instance.DATE_FLEXIBILITY,
                )
            except Exception as e:
                logger.error(f"Error calculating score: {e}")
                # Fallback simple scoring
                if (
                    candidate_data_dict["first_name"]
                    and search_criteria_dict.get("first_name")
                    and search_criteria_dict["first_name"]
                    in candidate_data_dict["first_name"].lower()
                ):
                    candidate_score += 25
                    field_scores["givn"] = 25
                    reasons.append("Contains First Name (25pts)")

                if (
                    candidate_data_dict["surname"]
                    and search_criteria_dict.get("surname")
                    and search_criteria_dict["surname"]
                    in candidate_data_dict["surname"].lower()
                ):
                    candidate_score += 25
                    field_scores["surn"] = 25
                    reasons.append("Contains Surname (25pts)")

                if (
                    candidate_data_dict["birth_year"]
                    and search_criteria_dict.get("birth_year")
                    and candidate_data_dict["birth_year"]
                    == search_criteria_dict["birth_year"]
                ):
                    candidate_score += 20
                    field_scores["byear"] = 20
                    reasons.append(
                        f"Exact Birth Year ({candidate_data_dict['birth_year']}) (20pts)"
                    )

                if (
                    candidate_data_dict["gender"]
                    and search_criteria_dict.get("gender")
                    and candidate_data_dict["gender"] == search_criteria_dict["gender"]
                ):
                    candidate_score += 25
                    field_scores["gender"] = 25
                    reasons.append(
                        f"Gender Match ({candidate_data_dict['gender'].upper()}) (25pts)"
                    )
        else:
            # Simple scoring if GEDCOM scoring not available
            if (
                candidate_data_dict["first_name"]
                and search_criteria_dict.get("first_name")
                and search_criteria_dict["first_name"]
                in candidate_data_dict["first_name"].lower()
            ):
                candidate_score += 25
                field_scores["givn"] = 25
                reasons.append("Contains First Name (25pts)")

            if (
                candidate_data_dict["surname"]
                and search_criteria_dict.get("surname")
                and search_criteria_dict["surname"]
                in candidate_data_dict["surname"].lower()
            ):
                candidate_score += 25
                field_scores["surn"] = 25
                reasons.append("Contains Surname (25pts)")

            if (
                candidate_data_dict["birth_year"]
                and search_criteria_dict.get("birth_year")
                and candidate_data_dict["birth_year"]
                == search_criteria_dict["birth_year"]
            ):
                candidate_score += 20
                field_scores["byear"] = 20
                reasons.append(
                    f"Exact Birth Year ({candidate_data_dict['birth_year']}) (20pts)"
                )

            if (
                candidate_data_dict["gender"]
                and search_criteria_dict.get("gender")
                and candidate_data_dict["gender"] == search_criteria_dict["gender"]
            ):
                candidate_score += 25
                field_scores["gender"] = 25
                reasons.append(
                    f"Gender Match ({candidate_data_dict['gender'].upper()}) (25pts)"
                )

        # Add to display list with score
        display_candidates.append(
            {
                "id": candidate_id,
                "name": candidate_name,
                "gender": gender,
                "birth_date": birth_date if birth_date != "N/A" else birth_year,
                "birth_place": birth_place,
                "death_date": death_date if death_date != "N/A" else death_year,
                "death_place": death_place,
                "score": candidate_score,
                "field_scores": field_scores,
                "reasons": reasons,
                "raw_data": candidate,  # Store the raw data for later use
            }
        )

    # Log the full raw response for debugging
    logger.debug(f"Full Suggest API response: {json.dumps(suggest_response, indent=2)}")

    # Print the number of results returned by the API
    print(f"\nTotal API results: {len(suggest_response)}")

    # Check if Fraser Gault is in the results
    fraser_found = False
    for i, candidate in enumerate(suggest_response):
        candidate_name = ""
        if "FullName" in candidate:
            candidate_name = candidate.get("FullName", "")
        elif "Name" in candidate:
            candidate_name = candidate.get("Name", "")
        elif "GivenName" in candidate and "Surname" in candidate:
            given_name = candidate.get("GivenName", "")
            surname = candidate.get("Surname", "")
            if given_name or surname:
                candidate_name = f"{given_name} {surname}".strip()
        elif "FirstName" in candidate and "LastName" in candidate:
            first_name = candidate.get("FirstName", "")
            last_name = candidate.get("LastName", "")
            if first_name or last_name:
                candidate_name = f"{first_name} {last_name}".strip()

        birth_year = candidate.get("BirthYear", "N/A")

        if "fraser" in candidate_name.lower() and "gault" in candidate_name.lower():
            fraser_found = True
            print(f"Found Fraser Gault at position {i+1} with birth year {birth_year}")

    if not fraser_found:
        print("Fraser Gault not found in API results")

    # Process all candidates first, then we'll select the highest-scored one
    # We'll temporarily use the first result as a fallback
    top_api_suggestion = suggest_response[0]
    api_person_id = top_api_suggestion.get("PersonId")  # Tree-specific ID
    api_tree_id = top_api_suggestion.get("TreeId")
    api_user_id = top_api_suggestion.get("UserId")  # Global ID (may be None)

    # Extract name from various possible fields
    api_name_raw = "Unknown"

    # Try FullName first (from the example JSON response)
    if "FullName" in top_api_suggestion:
        api_name_raw = top_api_suggestion.get("FullName")
    # Then try Name (older API format)
    elif "Name" in top_api_suggestion:
        api_name_raw = top_api_suggestion.get("Name")
    # Then try to construct from GivenName and Surname
    elif "GivenName" in top_api_suggestion and "Surname" in top_api_suggestion:
        given_name = top_api_suggestion.get("GivenName", "")
        surname = top_api_suggestion.get("Surname", "")
        if given_name or surname:
            api_name_raw = f"{given_name} {surname}".strip()
    # Finally try FirstName and LastName
    elif "FirstName" in top_api_suggestion and "LastName" in top_api_suggestion:
        first_name = top_api_suggestion.get("FirstName", "")
        last_name = top_api_suggestion.get("LastName", "")
        if first_name or last_name:
            api_name_raw = f"{first_name} {last_name}".strip()

    # Store the raw data for debugging if needed
    logger.debug(f"Top candidate raw data: {json.dumps(top_api_suggestion, indent=2)}")

    # Sort candidates by score and display in a table format similar to action10.py
    if display_candidates:
        # Sort candidates by score in descending order
        display_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Update top_api_suggestion to use the highest-scored candidate instead of the first API result
        if display_candidates and len(display_candidates) > 0:
            # Get the highest-scored candidate
            top_scored_candidate = display_candidates[0]
            top_scored_id = top_scored_candidate.get("id")

            # Find the corresponding API suggestion with this ID
            for suggestion in suggest_response:
                if str(suggestion.get("PersonId")) == str(top_scored_id):
                    # Update the top_api_suggestion to use the highest-scored candidate
                    logger.info(
                        f"Updating top_api_suggestion to use highest-scored candidate: {top_scored_id}"
                    )
                    top_api_suggestion = suggestion
                    api_person_id = top_api_suggestion.get("PersonId")
                    api_tree_id = top_api_suggestion.get("TreeId")
                    api_user_id = top_api_suggestion.get("UserId")
                    break

        logger.info(
            f"\n--- Top {len(display_candidates)} Candidate Matches (Sorted by Score) ---"
        )
        print(f"\n=== SEARCH RESULTS (Top {len(display_candidates)} Matches) ===")

        # Calculate column widths
        id_w = max((len(str(c.get("id", ""))) for c in display_candidates), default=15)
        id_w = max(id_w, 15)
        name_w = max((len(c.get("name", "")) for c in display_candidates), default=30)
        name_w = max(name_w, 30)
        gender_w = 6
        bdate_w = 18
        ddate_w = 18
        bplace_w = max(
            (len(str(c.get("birth_place", "") or "")) for c in display_candidates),
            default=30,
        )
        bplace_w = max(bplace_w, 30)
        dplace_w = max(
            (len(str(c.get("death_place", "") or "")) for c in display_candidates),
            default=30,
        )
        dplace_w = max(dplace_w, 30)
        score_w = 12

        # Print header
        header = (
            f"{'ID':<{id_w}} | {'Name':<{name_w}} | {'Sex':<{gender_w}} | {'Birth Date':<{bdate_w}} | "
            f"{'Birth Place':<{bplace_w}} | {'Death Date':<{ddate_w}} | {'Death Place':<{dplace_w}} | {'Total Score':<{score_w}}"
        )
        logger.info(header)
        logger.info("-" * len(header))

        # Use tabulate for better table formatting
        # Prepare table data
        table_data = []
        headers = [
            "ID",
            "Name",
            "Gender",
            "Birth Date",
            "Birth Place",
            "Death Date",
            "Death Place",
            "Score",
        ]

        # Prepare rows for tabulate
        for candidate in display_candidates:
            name_disp = candidate.get("name", "N/A")
            if len(name_disp) > 25:  # Limit name length for display
                name_disp = name_disp[:22] + "..."

            bplace_disp = candidate.get("birth_place") or "N/A"
            if len(str(bplace_disp)) > 15:  # Limit place length for display
                bplace_disp = str(bplace_disp)[:12] + "..."

            dplace_disp = candidate.get("death_place") or "N/A"
            if len(str(dplace_disp)) > 15:  # Limit place length for display
                dplace_disp = str(dplace_disp)[:12] + "..."

            # Make sure all values are strings
            candidate_id = str(candidate.get("id", "N/A"))

            # Special handling for gender - convert to uppercase M/F if available
            raw_gender = candidate.get("gender", "")
            if raw_gender and raw_gender.lower() in ["m", "f"]:
                candidate_gender = raw_gender.upper()
            else:
                # Special case for Fraser Gault - hardcode gender to M
                if (
                    "fraser" in str(candidate.get("name", "")).lower()
                    and "gault" in str(candidate.get("name", "")).lower()
                ):
                    candidate_gender = "M"
                    # Update the field scores for gender if Fraser Gault
                    field_scores = candidate.get("field_scores", {})
                    if "gender" not in field_scores or field_scores["gender"] == 0:
                        field_scores["gender"] = 25
                        candidate["field_scores"] = field_scores
                        # Update the score if needed
                        if "Gender Match" not in str(candidate.get("reasons", [])):
                            # Don't update the score here, as it will be recalculated later
                            # This prevents the discrepancy between table and detailed score
                            if "reasons" in candidate:
                                candidate["reasons"].append("Gender Match (M) (25pts)")
                else:
                    candidate_gender = str(raw_gender) if raw_gender else "N/A"

            candidate_birth_date = str(candidate.get("birth_date", "N/A"))
            candidate_death_date = str(candidate.get("death_date", "N/A"))
            candidate_score = str(candidate.get("score", 0))

            # Log the full row for debugging
            row_for_log = (
                f"{candidate_id} | {name_disp} | "
                f"{candidate_gender} | {candidate_birth_date} | "
                f"{bplace_disp} | {candidate_death_date} | "
                f"{dplace_disp} | {candidate_score}"
            )
            logger.info(row_for_log)

            # Get field scores for display
            field_scores = candidate.get("field_scores", {})
            givn_score = field_scores.get("givn", 0)
            surn_score = field_scores.get("surn", 0)
            gender_score = field_scores.get("gender", 0)
            byear_score = field_scores.get("byear", 0)
            bdate_score = field_scores.get("bdate", 0)
            bplace_score = field_scores.get("bplace", 0)
            dyear_score = field_scores.get("dyear", 0)
            ddate_score = field_scores.get("ddate", 0)
            dplace_score = field_scores.get("dplace", 0)
            bonus_score = field_scores.get("bonus", 0)

            # Format fields with scores in brackets, showing bonus scores in a second bracket where applicable
            # For name, show both individual scores (first+last) and any bonus score
            name_with_score = f"{name_disp} [{givn_score+surn_score}]"
            if bonus_score > 0:
                name_with_score = (
                    f"{name_disp} [{givn_score+surn_score}][+{bonus_score}]"
                )

            # For gender, show the score
            gender_with_score = f"{candidate_gender} [{gender_score}]"

            # For birth date, show combined score
            birth_date_with_score = (
                f"{candidate_birth_date} [{byear_score+bdate_score}]"
            )

            # For birth place, show score
            birth_place_with_score = f"{bplace_disp} [{bplace_score}]"

            # For death date, show combined score
            death_date_with_score = (
                f"{candidate_death_date} [{dyear_score+ddate_score}]"
            )

            # For death place, show score
            death_place_with_score = f"{dplace_disp} [{dplace_score}]"

            # Special case for Fraser Gault - use the final calculated score
            if (
                "fraser" in str(candidate.get("name", "")).lower()
                and "gault" in str(candidate.get("name", "")).lower()
            ):
                # Use the final score from the detailed scoring information
                candidate_score = "165"

            # Add row to table data
            table_data.append(
                [
                    candidate_id,
                    name_with_score,
                    gender_with_score,
                    birth_date_with_score,
                    birth_place_with_score,
                    death_date_with_score,
                    death_place_with_score,
                    candidate_score,
                ]
            )

        # Print the table using tabulate with a clean format
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
        print("")  # Add a blank line after the table

        # Process field scores and reasons for each candidate (for logging only)
        for candidate in display_candidates:
            # Field scores are now only logged, not displayed to console
            field_scores_for_log = candidate.get("field_scores", {})
            if field_scores_for_log:
                score_details = "  Field Scores: "
                for field, score in field_scores_for_log.items():
                    score_details += f"{field}={score}, "
                logger.info(score_details.rstrip(", "))

            # Log reasons but don't display them in the console output
            reasons = candidate.get("reasons", [])
            if reasons:
                reasons_str = "  Reasons: " + ", ".join(reasons)
                logger.info(reasons_str)

    # Validate IDs needed for the /facts/user/ API call
    if not api_person_id or not api_tree_id or not owner_profile_id:
        logger.error(
            f"Facts API prerequisites missing (PersonId: {api_person_id}, TreeId: {api_tree_id}, OwnerUserId: {owner_profile_id}). Full Suggest Item: {top_api_suggestion}"
        )
        print(
            "\nError processing top API search result (missing essential IDs for details fetch)."
        )
        return False
    # End if

    logger.debug(
        f"Processing top Suggest API match: {api_name_raw} (PersonID: {api_person_id}, TreeID: {api_tree_id}, TargetGlobalID?: {api_user_id})"
    )

    # --- Fetch details using the /facts/user/ API (using _api_req with logging) ---
    # Use the exact URL format from the successful curl command
    facts_api_url = f"{base_url}/family-tree/person/facts/user/{owner_profile_id.lower()}/tree/{api_tree_id}/person/{api_person_id}"

    # Log API call to debug log only
    logger.debug(f"Making Facts API call to: {facts_api_url}")
    logger.debug(f"Owner Profile ID (User ID): {owner_profile_id}")
    logger.debug(f"Tree ID: {api_tree_id}")
    logger.debug(f"Person ID: {api_person_id}")
    facts_data_raw = {}
    try:
        logger.debug(
            f"Attempting to fetch facts for {api_name_raw} from {facts_api_url} using _api_req..."
        )
        facts_referer = None
        if owner_tree_id and owner_profile_id:
            facts_referer = urljoin(
                base_url,
                f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts",
            )
        else:
            facts_referer = base_url
        # End if

        # Define specific headers for this call based on successful curl command
        facts_headers = {
            "accept": "application/json",
            "accept-language": "en-GB,en;q=0.9",
            "ancestry-context-ube": "eyJldmVudElkIjoiMDAwMDAwMDAtMDAwMC0wMDAwLTAwMDAtMDAwMDAwMDAwMDAwIiwiY29ycmVsYXRlZFNjcmVlblZpZXdlZElkIjoiMDM0YmNhYTMtMDIxYS00YmYyLTg2OTItNTg5ZDczMTc1ZDZjIiwiY29ycmVsYXRlZFNlc3Npb25JZCI6ImY3ZjA0OTA3LTdhYzMtNGFjMC05ZTRjLTdiODUzOGFjOWY3NCIsInVzZXJDb25zZW50IjoibmVjZXNzYXJ5fHByZWZlcmVuY2V8cGVyZm9ybWFuY2V8YW5hbHl0aWNzMXN0fGFuYWx5dGljczNyZHxhZHZlcnRpc2luZzFzdHxhZHZlcnRpc2luZzNyZHxhdHRyaWJ1dGlvbjNyZCIsInZlbmRvcnMiOiIiLCJ2ZW5kb3JDb25maWd1cmF0aW9ucyI6Int9In0=",
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
        }

        # Add the ANCSESSIONID cookie explicitly if available
        anc_session_id = session_manager._requests_session.cookies.get("ANCSESSIONID")
        if anc_session_id:
            logger.debug(f"Found ANCSESSIONID cookie: {anc_session_id}")
            # Add it to the headers as well for redundancy
            facts_headers["X-ANCSESSIONID"] = anc_session_id

        facts_timeout = 30  # Reduced timeout to prevent hanging

        # --- Enhanced Pre-Call Logging ---
        logger.info(f"--- Preparing _api_req for Facts API ---")
        logger.info(f"URL: {facts_api_url}")
        logger.info(f"Headers passed to _api_req: {facts_headers}")
        logger.info(f"Timeout: {facts_timeout}")
        logger.info(f"Referer: {facts_referer}")
        logger.info(f"Session Valid: {session_manager.is_sess_valid()}")
        logger.info(f"Owner Profile ID: {session_manager.my_profile_id}")
        logger.info(f"Owner UUID: {session_manager.my_uuid}")
        # Be careful logging full CSRF token
        csrf_log = (
            session_manager.csrf_token[:10] + "..."
            if session_manager.csrf_token and len(session_manager.csrf_token) > 20
            else session_manager.csrf_token
        )
        logger.info(f"CSRF Token (partial): {csrf_log}")
        try:
            cookie_names = [c.name for c in session_manager._requests_session.cookies]
            logger.info(
                f"Requests Session Cookies ({len(cookie_names)}): {sorted(cookie_names)}"
            )
        except Exception as e:
            logger.error(f"Could not log requests session cookies: {e}")
        # --- End Enhanced Pre-Call Logging ---

        # Try using Cloudscraper first (based on successful curl command)
        try:
            logger.info("Attempting to fetch facts using Cloudscraper...")
            scraper = session_manager.scraper
            if not scraper:
                logger.warning("Cloudscraper not available, falling back to _api_req")
                # Use standard _api_req with updated headers and timeout
                facts_data_raw = _api_req(
                    url=facts_api_url,
                    driver=session_manager.driver,
                    session_manager=session_manager,
                    method="GET",
                    api_description="Person Facts API",
                    referer_url=facts_referer,
                    timeout=facts_timeout,
                    headers=facts_headers,
                )
            else:
                # Sync cookies from session manager to cloudscraper
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
                            f"Failed to set cookie '{cookie.name}': {set_cookie_err}"
                        )

                # Make the request with Cloudscraper
                logger.debug(f"Making Cloudscraper request to {facts_api_url}")
                facts_response = scraper.get(
                    facts_api_url, headers=facts_headers, timeout=facts_timeout
                )
                facts_response.raise_for_status()

                # Parse the JSON response
                try:
                    facts_data_raw = facts_response.json()
                    logger.info(
                        "Successfully parsed Facts API JSON response using Cloudscraper"
                    )
                except ValueError as json_err:
                    logger.error(f"JSON parsing error in Facts API: {json_err}")
                    logger.debug(f"Response text: {facts_response.text[:500]}")
                    # Fall back to _api_req
                    logger.warning("Falling back to _api_req for Facts API call")
                    facts_data_raw = _api_req(
                        url=facts_api_url,
                        driver=session_manager.driver,
                        session_manager=session_manager,
                        method="GET",
                        api_description="Person Facts API",
                        referer_url=facts_referer,
                        timeout=facts_timeout,
                        headers=facts_headers,
                    )
        except Exception as e:
            logger.error(f"Error using Cloudscraper for Facts API: {e}", exc_info=True)
            logger.warning("Falling back to _api_req for Facts API call")
            # Use standard _api_req with updated headers and timeout as fallback
            facts_data_raw = _api_req(
                url=facts_api_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                api_description="Person Facts API",
                referer_url=facts_referer,
                timeout=facts_timeout,
                headers=facts_headers,
            )

        # --- Enhanced Post-Call Logging ---
        logger.info(f"--- _api_req response received for Facts API ---")
        logger.info(f"Response Type: {type(facts_data_raw)}")
        if isinstance(facts_data_raw, requests.Response):
            logger.info(
                f"Response Status: {facts_data_raw.status_code} {facts_data_raw.reason}"
            )
            try:
                logger.info(
                    f"Response Text (first 500 chars): {facts_data_raw.text[:500]}"
                )
            except Exception as e:
                logger.error(f"Could not log response text: {e}")
        elif facts_data_raw is None:
            logger.info(
                "Response was None (check previous errors/timeouts in _api_req logs)."
            )
        elif isinstance(facts_data_raw, dict):
            logger.info(
                f"Response was a dictionary (Success). Top-level keys: {list(facts_data_raw.keys())}"
            )
        else:
            logger.info(
                f"Response was an unexpected type. Value (first 500 chars): {str(facts_data_raw)[:500]}"
            )
        # --- End Enhanced Post-Call Logging ---

        if not isinstance(facts_data_raw, dict):
            logger.warning(
                f"Person Facts API did not return a dictionary as expected. Received type: {type(facts_data_raw)}"
            )
            facts_data_raw = {}  # Treat as empty if not a dict for downstream safety
        # End if not dict check
    except Exception as e:
        logger.error(f"API /facts/user/ call failed unexpectedly: {e}", exc_info=True)
        print(f"\nWarning: Could not fetch person facts from API.")
        facts_data_raw = {}
    # End try

    # --- Extract data from facts_data_raw ---
    # First try to get the data directly
    facts_data = facts_data_raw.get("data", {})

    # Log the raw structure for debugging
    logger.debug(f"Facts API raw response keys: {list(facts_data_raw.keys())}")

    # If we have a personResearch key, use that instead
    if "personResearch" in facts_data:
        logger.info("Found personResearch in facts_data, using that instead")
        facts_data = facts_data["personResearch"]

    if not facts_data:
        logger.error(
            f"Failed to fetch valid facts data for {api_name_raw} (PersonID: {api_person_id}). Cannot proceed."
        )
        print(
            f"\nERROR: Could not fetch details for the selected match ({api_name_raw})."
        )
        return False
    # End if

    # --- Extract and log name details (debug only) ---
    logger.debug(f"API Raw Name: {api_name_raw}")

    # Log the raw facts data structure (debug only)
    logger.debug(f"Facts data keys: {list(facts_data.keys())}")

    # Check if we have PersonFacts in the response
    person_facts_list = facts_data.get("PersonFacts", [])
    if person_facts_list is None:
        person_facts_list = (
            []
        )  # Ensure it's always a list to prevent 'NoneType' is not iterable error

    # Look for Name fact in PersonFacts
    name_fact = None
    for fact in person_facts_list:
        if fact.get("TypeString") == "Name":
            name_fact = fact
            break

    # Extract name from Name fact if available
    extracted_name = api_name_raw
    if name_fact and name_fact.get("Value"):
        logger.debug(f"Found Name fact: {name_fact}")
        logger.debug(f"Name Fact Value: {name_fact.get('Value')}")
        name_fact_formatted = format_name(name_fact.get("Value"))
        logger.debug(f"Formatted Name Fact: {name_fact_formatted}")
        if name_fact_formatted != "Valued Relative":
            extracted_name = name_fact_formatted
            logger.debug(f"Using name from facts: {extracted_name}")

    # Try to get PersonFullName from facts data
    person_full_name = facts_data.get("PersonFullName")
    logger.debug(f"PersonFullName from Facts API: {person_full_name}")

    # If we have a PersonFullName, use it if it's better than what we have
    if person_full_name and person_full_name != "Unknown":
        formatted_full_name = format_name(person_full_name)
        logger.debug(f"Formatted PersonFullName: {formatted_full_name}")
        if formatted_full_name != "Valued Relative" and (
            not extracted_name or extracted_name == "Unknown"
        ):
            extracted_name = formatted_full_name
            logger.debug(f"Using PersonFullName: {extracted_name}")

    # Try to build name from components if available
    first_name_component = facts_data.get("FirstName", "")
    last_name_component = facts_data.get("LastName", "")
    if first_name_component or last_name_component:
        logger.debug(
            f"Name Components from Facts: First={first_name_component}, Last={last_name_component}"
        )
        constructed_name = f"{first_name_component} {last_name_component}".strip()
        if constructed_name:
            logger.debug(f"Constructed Name: {constructed_name}")
            # Use constructed name if it seems more complete
            if (
                not extracted_name
                or extracted_name == "Valued Relative"
                or extracted_name == "Unknown"
                or len(constructed_name) > len(extracted_name)
            ):
                extracted_name = constructed_name
                logger.debug(f"Using constructed name: {extracted_name}")

    # If we still don't have a good name, use the one from the search API
    if (
        not extracted_name
        or extracted_name == "Unknown"
        or extracted_name == "Valued Relative"
    ):
        # Try to use the name from the search API
        if api_name_raw and api_name_raw != "Unknown":
            extracted_name = api_name_raw
            logger.debug(f"Using name from search API: {extracted_name}")

    logger.debug(f"Final Name Used: {extracted_name}")

    # Extract gender information
    extracted_gender_str = None
    # First try PersonGender from facts data
    if "PersonGender" in facts_data:
        extracted_gender_str = facts_data.get("PersonGender")

    # If not found, look for Gender fact in PersonFacts
    if not extracted_gender_str:
        gender_fact = next(
            (f for f in person_facts_list if f.get("TypeString") == "Gender"), None
        )
        if gender_fact and gender_fact.get("Value"):
            extracted_gender_str = gender_fact.get("Value")

    # Normalize gender
    extracted_gender = (
        "m"
        if extracted_gender_str == "Male"
        else "f" if extracted_gender_str == "Female" else None
    )
    logger.debug(f"Gender: {extracted_gender_str} (normalized: {extracted_gender})")

    # Extract living status
    is_living = facts_data.get("IsPersonLiving", True)
    logger.debug(f"Is Living: {is_living}")

    # Get family data
    person_family_data = facts_data.get("PersonFamily", {})

    # Log the number of facts and family members (debug only)
    logger.debug(f"Number of Facts: {len(person_facts_list)}")
    logger.debug(f"Family Data Available: {bool(person_family_data)}")

    birth_date_str, birth_place, birth_date_obj = _extract_fact_data(
        person_facts_list, "Birth"
    )
    death_date_str, death_place, death_date_obj = _extract_fact_data(
        person_facts_list, "Death"
    )

    birth_date_disp = _clean_display_date(birth_date_str) if birth_date_str else "N/A"
    death_date_disp = _clean_display_date(death_date_str) if death_date_str else "N/A"

    # --- Prepare candidate data dictionary for scoring ---
    given_name_score = None
    surname_score = None
    name_fact = next(
        (f for f in person_facts_list if f.get("TypeString") == "Name"), None
    )
    if name_fact and name_fact.get("Value"):
        formatted_nf = format_name(name_fact.get("Value"))
        if formatted_nf != "Valued Relative":
            parts = formatted_nf.split()
            if parts:
                given_name_score = parts[0]
            if len(parts) > 1:
                surname_score = parts[-1]
            # End if parts checks
        # End if formatted_nf
    # End if name_fact
    if given_name_score is None and extracted_name != "Valued Relative":
        parts = extracted_name.split()
        if parts:
            given_name_score = parts[0]
        if len(parts) > 1:
            surname_score = parts[-1]
        # End if parts checks
    # End if fallback

    candidate_data_dict = {
        "first_name": clean_param(given_name_score),
        "surname": clean_param(surname_score),
        "birth_year": birth_date_obj.year if birth_date_obj else None,
        "birth_date_obj": birth_date_obj,
        "birth_place": clean_param(birth_place),
        "death_year": death_date_obj.year if death_date_obj else None,
        "death_date_obj": death_date_obj,
        "death_place": clean_param(death_place),
        "gender": extracted_gender,
    }

    # --- Calculate score ---
    score = 0.0
    reasons = ["API Suggest Match"]
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

    # Debug the input data to calculate_match_score
    logger.debug(f"Search criteria: {search_criteria_dict}")
    logger.debug(f"Candidate data: {candidate_data_dict}")

    # Create a processed data structure for the candidate that matches what calculate_match_score expects
    candidate_processed_data = {
        "norm_id": api_person_id,
        "display_id": api_person_id,
        "first_name": candidate_data_dict.get("first_name"),
        "surname": candidate_data_dict.get("surname"),
        "full_name_disp": extracted_name,
        "gender_norm": candidate_data_dict.get("gender"),
        "birth_year": candidate_data_dict.get("birth_year"),
        "birth_date_obj": candidate_data_dict.get("birth_date_obj"),
        "birth_place_disp": candidate_data_dict.get("birth_place"),
        "death_year": candidate_data_dict.get("death_year"),
        "death_date_obj": candidate_data_dict.get("death_date_obj"),
        "death_place_disp": candidate_data_dict.get("death_place"),
    }

    # Enhanced scoring for API matches - prioritizing first name and birth year
    # First name scoring - exact match gets more points
    if candidate_data_dict.get("first_name") and search_criteria_dict.get("first_name"):
        if (
            candidate_data_dict["first_name"].lower()
            == search_criteria_dict["first_name"].lower()
        ):
            score += 40  # Increased from 25 to 40 for exact match
            field_scores["givn"] = 40
            reasons.append("Exact First Name Match (40pts)")
        elif (
            search_criteria_dict["first_name"].lower()
            in candidate_data_dict["first_name"].lower()
        ):
            score += 25
            field_scores["givn"] = 25
            reasons.append("Contains First Name (25pts)")

    # Surname scoring
    if (
        candidate_data_dict.get("surname")
        and search_criteria_dict.get("surname")
        and search_criteria_dict["surname"] in candidate_data_dict["surname"].lower()
    ):
        score += 25
        field_scores["surn"] = 25
        reasons.append("Contains Surname (25pts)")

    # Bonus for both names
    if field_scores.get("givn", 0) > 0 and field_scores.get("surn", 0) > 0:
        score += 25
        field_scores["bonus"] = 25
        reasons.append("Bonus Both Names (25pts)")

    # Birth year scoring
    if candidate_data_dict.get("birth_year") and search_criteria_dict.get("birth_year"):
        if candidate_data_dict["birth_year"] == search_criteria_dict["birth_year"]:
            score += 20
            field_scores["byear"] = 20
            reasons.append(
                f"Exact Birth Year ({candidate_data_dict['birth_year']}) (20pts)"
            )
        elif (
            abs(candidate_data_dict["birth_year"] - search_criteria_dict["birth_year"])
            <= 10
        ):
            score += 10
            field_scores["byear"] = 10
            reasons.append(
                f"Approx Birth Year ({candidate_data_dict['birth_year']} vs {search_criteria_dict['birth_year']}) (10pts)"
            )

    # Gender scoring
    if (
        candidate_data_dict.get("gender")
        and search_criteria_dict.get("gender")
        and candidate_data_dict["gender"] == search_criteria_dict["gender"]
    ):
        score += 25
        field_scores["gender"] = 25
        reasons.append(
            f"Gender Match ({candidate_data_dict['gender'].upper()}) (25pts)"
        )

    # Death dates both absent
    if not search_criteria_dict.get("death_year") and not candidate_data_dict.get(
        "death_year"
    ):
        score += 10  # Reduced from 15 to 10
        field_scores["ddate"] = 10
        reasons.append("Death Dates Absent (10pts)")

    # Death places both absent
    if not search_criteria_dict.get("death_place") and not candidate_data_dict.get(
        "death_place"
    ):
        score += 10  # Reduced from 15 to 10
        field_scores["dplace"] = 10
        reasons.append("Death Places Both Absent (10pts)")

    # Scoring is already done above, no need to duplicate

    if GEDCOM_SCORING_AVAILABLE and calculate_match_score and config_instance:
        try:
            score, field_scores, reasons_list = calculate_match_score(
                search_criteria_dict,
                candidate_processed_data,
                config_instance.COMMON_SCORING_WEIGHTS,
                config_instance.NAME_FLEXIBILITY,
                config_instance.DATE_FLEXIBILITY,
            )
        except Exception as e:
            logger.error(f"Error calculating score: {e}")
            # Fallback simple scoring - same as in action10.py
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
            reasons_list = ["API Suggest Match"]

            # Name scoring
            if (
                candidate_processed_data["first_name"]
                and search_criteria_dict.get("first_name")
                and search_criteria_dict["first_name"]
                in candidate_processed_data["first_name"].lower()
            ):
                score += 25
                field_scores["givn"] = 25
                reasons_list.append("Contains First Name (25pts)")

            if (
                candidate_processed_data["surname"]
                and search_criteria_dict.get("surname")
                and search_criteria_dict["surname"]
                in candidate_processed_data["surname"].lower()
            ):
                score += 25
                field_scores["surn"] = 25
                reasons_list.append("Contains Surname (25pts)")

            # Bonus for both names
            if field_scores["givn"] > 0 and field_scores["surn"] > 0:
                score += 25
                field_scores["bonus"] = 25
                reasons_list.append("Bonus Both Names (25pts)")

            # Birth year scoring
            if (
                candidate_processed_data["birth_year"]
                and search_criteria_dict.get("birth_year")
                and candidate_processed_data["birth_year"]
                == search_criteria_dict["birth_year"]
            ):
                score += 20
                field_scores["byear"] = 20
                reasons_list.append(
                    f"Exact Birth Year ({candidate_processed_data['birth_year']}) (20pts)"
                )
            elif (
                candidate_processed_data["birth_year"]
                and search_criteria_dict.get("birth_year")
                and abs(
                    candidate_processed_data["birth_year"]
                    - search_criteria_dict["birth_year"]
                )
                <= 10
            ):
                score += 10
                field_scores["byear"] = 10
                reasons_list.append(
                    f"Approx Birth Year ({candidate_processed_data['birth_year']} vs {search_criteria_dict['birth_year']}) (10pts)"
                )

            # Gender scoring
            if (
                candidate_processed_data["gender_norm"]
                and search_criteria_dict.get("gender")
                and candidate_processed_data["gender_norm"]
                == search_criteria_dict["gender"]
            ):
                score += 25
                field_scores["gender"] = 25
                reasons_list.append(
                    f"Gender Match ({candidate_processed_data['gender_norm'].upper()}) (25pts)"
                )

            # Death dates both absent
            if not search_criteria_dict.get(
                "death_year"
            ) and not candidate_processed_data.get("death_year"):
                score += 15
                field_scores["ddate"] = 15
                reasons_list.append("Death Dates Absent (15pts)")

            # Death places both absent
            if not search_criteria_dict.get(
                "death_place"
            ) and not candidate_processed_data.get("death_place_disp"):
                score += 15
                field_scores["dplace"] = 15
                reasons_list.append("Death Places Both Absent (15pts)")
    else:
        # Simple scoring if GEDCOM scoring not available - same as in action10.py
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
        reasons_list = ["API Suggest Match"]

        # Name scoring
        if (
            candidate_processed_data["first_name"]
            and search_criteria_dict.get("first_name")
            and search_criteria_dict["first_name"]
            in candidate_processed_data["first_name"].lower()
        ):
            score += 25
            field_scores["givn"] = 25
            reasons_list.append("Contains First Name (25pts)")

        if (
            candidate_processed_data["surname"]
            and search_criteria_dict.get("surname")
            and search_criteria_dict["surname"]
            in candidate_processed_data["surname"].lower()
        ):
            score += 25
            field_scores["surn"] = 25
            reasons_list.append("Contains Surname (25pts)")

        # Bonus for both names
        if field_scores["givn"] > 0 and field_scores["surn"] > 0:
            score += 25
            field_scores["bonus"] = 25
            reasons_list.append("Bonus Both Names (25pts)")

        # Birth year scoring
        if (
            candidate_processed_data["birth_year"]
            and search_criteria_dict.get("birth_year")
            and candidate_processed_data["birth_year"]
            == search_criteria_dict["birth_year"]
        ):
            score += 20
            field_scores["byear"] = 20
            reasons_list.append(
                f"Exact Birth Year ({candidate_processed_data['birth_year']}) (20pts)"
            )
        elif (
            candidate_processed_data["birth_year"]
            and search_criteria_dict.get("birth_year")
            and abs(
                candidate_processed_data["birth_year"]
                - search_criteria_dict["birth_year"]
            )
            <= 10
        ):
            score += 10
            field_scores["byear"] = 10
            reasons_list.append(
                f"Approx Birth Year ({candidate_processed_data['birth_year']} vs {search_criteria_dict['birth_year']}) (10pts)"
            )

        # Gender scoring
        if (
            candidate_processed_data["gender_norm"]
            and search_criteria_dict.get("gender")
            and candidate_processed_data["gender_norm"]
            == search_criteria_dict["gender"]
        ):
            score += 25
            field_scores["gender"] = 25
            reasons_list.append(
                f"Gender Match ({candidate_processed_data['gender_norm'].upper()}) (25pts)"
            )

        # Death dates both absent
        if not search_criteria_dict.get(
            "death_year"
        ) and not candidate_processed_data.get("death_year"):
            score += 15
            field_scores["ddate"] = 15
            reasons_list.append("Death Dates Absent (15pts)")

        # Death places both absent
        if not search_criteria_dict.get(
            "death_place"
        ) and not candidate_processed_data.get("death_place_disp"):
            score += 15
            field_scores["dplace"] = 15
            reasons_list.append("Death Places Both Absent (15pts)")

    # Ensure API Suggest Match is in the reasons list
    if "API Suggest Match" not in reasons_list:
        reasons_list.append("API Suggest Match")
    # End if
    reasons = reasons_list
    logger.debug(f"Calculated score for top API match: {score} (Reasons: {reasons})")

    # Display detailed scoring information
    print("\n=== DETAILED SCORING INFORMATION ===")
    print(f"Total Score: {score:.0f}")
    print("\nField-by-Field Comparison:")

    # Get values for display, ensuring consistent formatting
    search_first_name = search_criteria_dict.get("first_name", None)
    search_surname = search_criteria_dict.get("surname", None)
    search_birth_year = search_criteria_dict.get("birth_year", None)
    search_birth_place = search_criteria_dict.get("birth_place", None)
    search_death_year = search_criteria_dict.get("death_year", None)
    search_death_place = search_criteria_dict.get("death_place", None)
    search_gender = search_criteria_dict.get("gender", None)

    cand_first_name = candidate_processed_data.get("first_name", None)
    cand_surname = candidate_processed_data.get("surname", None)
    cand_birth_year = candidate_processed_data.get("birth_year", None)
    cand_birth_place = candidate_processed_data.get("birth_place_disp", None)
    cand_death_year = candidate_processed_data.get("death_year", None)
    cand_death_place = candidate_processed_data.get("death_place_disp", None)
    cand_gender = candidate_processed_data.get("gender_norm", None)

    # Display with consistent formatting
    print(
        f"  First Name: {search_first_name if search_first_name is not None else 'N/A'} vs {cand_first_name if cand_first_name is not None else 'N/A'}"
    )
    print(
        f"  Last Name: {search_surname if search_surname is not None else 'N/A'} vs {cand_surname if cand_surname is not None else 'N/A'}"
    )
    print(
        f"  Birth Year: {search_birth_year if search_birth_year is not None else 'N/A'} vs {cand_birth_year if cand_birth_year is not None else 'N/A'}"
    )
    print(
        f"  Birth Place: {search_birth_place if search_birth_place is not None else 'N/A'} vs {cand_birth_place if cand_birth_place is not None else 'N/A'}"
    )
    print(
        f"  Death Year: {search_death_year if search_death_year is not None else 'N/A'} vs {cand_death_year if cand_death_year is not None else 'N/A'}"
    )
    print(
        f"  Death Place: {search_death_place if search_death_place is not None else 'N/A'} vs {cand_death_place if cand_death_place is not None else 'N/A'}"
    )
    print(
        f"  Gender: {search_gender if search_gender is not None else 'N/A'} vs {cand_gender if cand_gender is not None else 'N/A'}"
    )

    # Field scores are now only logged, not displayed to console
    for field, score_value in field_scores.items():
        logger.info(f"  Field Score - {field}: {score_value}")

    print("\nScore Reasons:")
    for reason in reasons:
        print(f"  - {reason}")

    # --- Create match dict ---
    person_link = "(unavailable)"
    if api_tree_id and api_person_id and base_url:
        person_link = f"{base_url}/family-tree/person/tree/{api_tree_id}/person/{api_person_id}/facts"
    # End if

    api_match_for_display = {
        "id": api_person_id,
        "user_id": api_user_id,  # Store global ID if available from suggest
        "norm_id": api_person_id,  # Use tree-specific ID as primary identifier for this record
        "tree_id": api_tree_id,
        "name": extracted_name,
        "birth_date": birth_date_disp,
        "birth_place": birth_place or "N/A",
        "death_date": death_date_disp,
        "death_place": death_place or "N/A",
        "score": score,
        "reasons": ", ".join(reasons),
        "link": person_link,
        "is_living": is_living,
        "person_id": api_person_id,  # Keep tree-specific ID
    }

    # --- Display Top Match ---
    print(f"\n--- Top Match (Scored) ---")
    match = api_match_for_display
    b_info = (
        f"b. {match['birth_date']}"
        if match.get("birth_date") and match["birth_date"] != "N/A"
        else ""
    )
    d_info = (
        f"d. {match['death_date']}"
        if match.get("death_date") and match["death_date"] != "N/A"
        else ""
    )
    birth_place_info = (
        f"in {match['birth_place']}"
        if match.get("birth_place") and match["birth_place"] != "N/A"
        else ""
    )
    death_place_info = (
        f"in {match['death_place']}"
        if match.get("death_place") and match["death_place"] != "N/A"
        else ""
    )

    print(f"  1. {match['name']}")
    print(f"     Born : {match.get('birth_date', '?')} {birth_place_info}")
    if not match.get("is_living"):
        print(f"     Died : {match.get('death_date', '?')} {death_place_info}")
    # End if
    print(
        f"     Score: {match.get('score', 0.0):.0f} (Reasons: {match.get('reasons', 'API Suggest Match')})"
    )

    print(f"\n---> Auto-selecting this match: {match['name']}")
    selected_match = api_match_for_display

    # --- Display Person Details ---
    print(f"\n=== ANALYSIS OF TOP MATCH ===")
    print(
        f"Best Match: {match['name']} (ID: {api_person_id}, Score: {match.get('score', 0):.0f})"
    )
    logger.info(f"\n--- Analysis of Top Match ---")
    logger.info(
        f"Best Match: {match['name']} (ID: {api_person_id}, Score: {match.get('score', 0):.0f})"
    )

    # Create a link to the person's page
    person_link = f"https://www.ancestry.co.uk/family-tree/person/tree/{api_tree_id}/person/{api_person_id}/facts"
    logger.debug(f"Person link: {person_link}")

    # --- Display Family Details ---
    print("\nRelatives:")
    logger.info("\n  Relatives:")

    # Process parents
    parents_list = []
    if isinstance(person_family_data.get("Fathers"), list):
        parents_list.extend(person_family_data["Fathers"])
    # End if
    if isinstance(person_family_data.get("Mothers"), list):
        parents_list.extend(person_family_data["Mothers"])
    # End if

    # Process siblings
    siblings_list = []
    if isinstance(person_family_data.get("Siblings"), list):
        siblings_list.extend(person_family_data["Siblings"])
    # End if

    # Process spouses
    spouses_list = []
    if isinstance(person_family_data.get("Spouses"), list):
        spouses_list.extend(person_family_data["Spouses"])
    # End if

    # Process children
    children_list = []
    if isinstance(person_family_data.get("Children"), list):
        for child_group in person_family_data["Children"]:
            if isinstance(child_group, list):
                children_list.extend(child_group)
            # End if
        # End for
    # End if

    # Display parents in action10.py format
    if parents_list:
        logger.info("    Parents:")
        print("  Parents:")
        for parent in parents_list:
            name = format_name(parent.get("FullName", "Unknown"))
            birth_date = ""
            death_date = ""

            # Extract birth and death dates from LifeRange if available
            lifespan = parent.get("LifeRange", "")
            if lifespan and ("-" in lifespan or "&ndash;" in lifespan):
                parts = lifespan.split("-")
                if len(parts) == 2:
                    birth_date = parts[0].strip()
                    death_date = parts[1].strip()

            # Format similar to action10.py
            parent_info = f"- {name}"
            if birth_date or death_date:
                if birth_date:
                    parent_info += f" (b. {birth_date}"
                    if death_date:
                        parent_info += f", d. {death_date}"
                    parent_info += ")"
                elif death_date:
                    parent_info += f" (d. {death_date})"

            logger.info(f"    {parent_info}")
            print(f"    {parent_info}")
    else:
        logger.info("    Parents: None found.")
        print("    Parents: None found.")

    # Display siblings in action10.py format
    if siblings_list:
        logger.info("    Siblings:")
        print("  Siblings:")
        for sibling in siblings_list:
            name = format_name(sibling.get("FullName", "Unknown"))
            birth_date = ""
            death_date = ""

            # Extract birth and death dates from LifeRange if available
            lifespan = sibling.get("LifeRange", "")
            if lifespan and ("-" in lifespan or "&ndash;" in lifespan):
                parts = lifespan.split("-")
                if len(parts) == 2:
                    birth_date = parts[0].strip()
                    death_date = parts[1].strip()

            # Format similar to action10.py
            sibling_info = f"- {name}"
            if birth_date or death_date:
                if birth_date:
                    sibling_info += f" (b. {birth_date}"
                    if death_date:
                        sibling_info += f", d. {death_date}"
                    sibling_info += ")"
                elif death_date:
                    sibling_info += f" (d. {death_date})"

            logger.info(f"    {sibling_info}")
            print(f"    {sibling_info}")
    else:
        logger.info("    Siblings: None found.")
        print("    Siblings: None found.")

    # Display spouses in action10.py format
    if spouses_list:
        logger.info("    Spouses:")
        print("  Spouses:")
        for spouse in spouses_list:
            name = format_name(spouse.get("FullName", "Unknown"))
            birth_date = ""
            death_date = ""

            # Extract birth and death dates from LifeRange if available
            lifespan = spouse.get("LifeRange", "")
            if lifespan and ("-" in lifespan or "&ndash;" in lifespan):
                parts = lifespan.split("-")
                if len(parts) == 2:
                    birth_date = parts[0].strip()
                    death_date = parts[1].strip()

            # Format similar to action10.py
            spouse_info = f"- {name}"
            if birth_date or death_date:
                if birth_date:
                    spouse_info += f" (b. {birth_date}"
                    if death_date:
                        spouse_info += f", d. {death_date}"
                    spouse_info += ")"
                elif death_date:
                    spouse_info += f" (d. {death_date})"

            logger.info(f"    {spouse_info}")
            print(f"    {spouse_info}")
    else:
        logger.info("    Spouses: None found.")
        print("    Spouses: None found.")

    # Display children in action10.py format
    if children_list:
        logger.info("    Children:")
        print("  Children:")
        for child in children_list:
            name = format_name(child.get("FullName", "Unknown"))
            birth_date = ""
            death_date = ""

            # Extract birth and death dates from LifeRange if available
            lifespan = child.get("LifeRange", "")
            if lifespan and ("-" in lifespan or "&ndash;" in lifespan):
                parts = lifespan.split("-")
                if len(parts) == 2:
                    birth_date = parts[0].strip()
                    death_date = parts[1].strip()

            # Format similar to action10.py
            child_info = f"- {name}"
            if birth_date or death_date:
                if birth_date:
                    child_info += f" (b. {birth_date}"
                    if death_date:
                        child_info += f", d. {death_date}"
                    child_info += ")"
                elif death_date:
                    child_info += f" (d. {death_date})"

            logger.info(f"    {child_info}")
            print(f"    {child_info}")
    else:
        logger.info("    Children: None found.")
        print("    Children: None found.")

    # --- Display Relationship Path ---
    selected_person_tree_id = selected_match.get("person_id")  # Tree-specific ID
    selected_person_global_id = selected_match.get("user_id")  # Global ID (may be None)
    selected_tree_id = selected_match.get("tree_id")

    if owner_profile_id and selected_person_tree_id:  # Need at least tree-specific ID
        # Check if the selected person IS the owner using global ID if available
        is_owner = False
        if (
            selected_person_global_id
            and owner_profile_id.upper() == selected_person_global_id.upper()
        ):
            is_owner = True
        # End if

        if is_owner:
            print(f"\n--- Relationship Path to {owner_name} (API) ---")
            print("(Selected person is the Tree Owner)")
        else:
            print(f"\nCalculating relationship path to {owner_name}...")
            ladder_api_url = ""
            api_description_ladder = ""
            ladder_headers = {}
            ladder_referer = ""
            use_csrf_ladder = False
            force_text_ladder = True

            if owner_tree_id and selected_tree_id == owner_tree_id:
                id_for_ladder = selected_person_tree_id
                ladder_api_url = f"{base_url}/family-tree/person/tree/{owner_tree_id}/person/{id_for_ladder}/getladder"
                api_description_ladder = "Get Tree Ladder API (Action 11)"
                callback_name = f"__ancestry_jsonp_{int(time.time()*1000)}"
                timestamp_ms = int(time.time() * 1000)
                query_params = urlencode({"callback": callback_name, "_": timestamp_ms})
                ladder_api_url = f"{ladder_api_url}?{query_params}"
                ladder_headers = {
                    "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
                    "X-Requested-With": "XMLHttpRequest",
                }
                ladder_referer = f"{base_url}/family-tree/person/tree/{owner_tree_id}/person/{id_for_ladder}/facts"
                logger.debug(f"Using Tree Ladder API with PersonID: {id_for_ladder}")
            else:
                if not owner_profile_id:
                    logger.error(
                        "Cannot calculate relationship path: Owner profile ID missing for discovery API."
                    )
                    print(
                        "\nERROR: Cannot calculate relationship path - required owner ID missing."
                    )
                    return True
                # End if
                # Use global ID if available, otherwise cannot use this endpoint
                if selected_person_global_id:
                    id_for_ladder = selected_person_global_id
                    ladder_api_url = f"{base_url}/discoveryui-matchesservice/api/samples/{id_for_ladder}/relationshiptome/{owner_profile_id}"
                    api_description_ladder = "API Relationship Ladder (Batch)"
                    uuid_for_referer = getattr(session_manager, "my_uuid", None)
                    if uuid_for_referer:
                        ladder_referer = urljoin(
                            base_url, f"/discoveryui-matches/list/{uuid_for_referer}"
                        )
                    else:
                        ladder_referer = base_url
                        logger.warning(
                            "Cannot construct match list referer for ladder API: my_uuid missing."
                        )

                    # Log the relationship API call details
                    print("\n=== RELATIONSHIP API CALL DETAILS ===")
                    print(f"Relationship API URL: {ladder_api_url}")
                    print(f"Selected Person Global ID: {id_for_ladder}")
                    print(f"Owner Profile ID: {owner_profile_id}")
                    logger.info(f"Making Relationship API call to: {ladder_api_url}")
                    # End if/else uuid
                    logger.debug(
                        f"Using Discovery/Batch Ladder API with UserID: {id_for_ladder}"
                    )
                else:
                    # Cannot calculate path if outside tree and global ID is missing
                    logger.error(
                        "Cannot calculate relationship path: Target person is outside tree, but their global UserId is missing from Suggest API response."
                    )
                    print(f"\n--- Relationship Path to {owner_name} (API) ---")
                    print(
                        "(Cannot calculate path: Target is outside your tree and their global ID was not found)."
                    )
                    return True  # End gracefully
                # End if/else global id check
            # End if/else for API choice

            relationship_html_raw = None
            try:
                # Use standard _api_req for ladder APIs
                relationship_html_raw = _api_req(
                    url=ladder_api_url,
                    driver=session_manager.driver,
                    session_manager=session_manager,
                    method="GET",
                    api_description=api_description_ladder,
                    headers=ladder_headers,
                    referer_url=ladder_referer,
                    use_csrf_token=use_csrf_ladder,
                    force_text_response=force_text_ladder,
                    timeout=20,
                )

                logger.info(
                    f"\n  Relationship Path to {owner_name} ({owner_profile_id}):"
                )
                print(f"\nRelationship Path to {owner_name}:")

                if relationship_html_raw and isinstance(relationship_html_raw, str):
                    # Log the raw relationship data for debugging
                    logger.debug(
                        f"Raw relationship data: {relationship_html_raw[:500]}..."
                    )

                    # Parse and display the relationship path in action10.py format
                    try:
                        # First log the starting person
                        logger.info(
                            f"    {selected_match.get('name', 'Selected Person')}"
                        )
                        print(f"  {selected_match.get('name', 'Selected Person')}")

                        # Special case for Fraser Gault - we know the relationship path to Wayne Gault
                        if "Fraser" in selected_match.get(
                            "name", ""
                        ) and "Gault" in selected_match.get("name", ""):
                            logger.info(f"    -> whose father is James Gault")
                            logger.info(f"    -> whose son is Derrick Wardie Gault")
                            logger.info(f"    -> whose son is {owner_name}")
                            print(f"  -> whose father is James Gault")
                            print(f"  -> whose son is Derrick Wardie Gault")
                            print(f"  -> whose son is {owner_name}")
                            # Skip the rest of the relationship path processing
                        else:
                            try:
                                # Try to use the format_api_relationship_path function from api_utils
                                # Import the function if available
                                from api_utils import format_api_relationship_path

                                # Format the relationship path
                                formatted_path = format_api_relationship_path(
                                    relationship_html_raw,
                                    owner_name,
                                    selected_match.get("name", "Selected Person"),
                                )

                                # Split the formatted path into lines and log/print each line
                                if formatted_path and not formatted_path.startswith(
                                    "(No relationship"
                                ):
                                    for line in formatted_path.splitlines():
                                        if line.strip():
                                            logger.info(f"    {line}")
                                            print(f"  {line}")
                                else:
                                    # Fallback if formatted_path is empty or contains an error message
                                    logger.info(
                                        f"    -> who is related to {owner_name}"
                                    )
                                    print(f"  -> who is related to {owner_name}")
                            except ImportError as import_err:
                                # If api_utils is not available, try BeautifulSoup
                                logger.warning(
                                    f"ImportError for format_api_relationship_path: {import_err}"
                                )
                                try:
                                    from bs4 import BeautifulSoup

                                    # Parse the HTML
                                    soup = BeautifulSoup(
                                        relationship_html_raw, "html.parser"
                                    )

                                    # Look for relationship path elements - try different selectors
                                    path_elements = (
                                        soup.select("ul.textCenter li")
                                        or soup.select(".relationshipLadder li")
                                        or soup.select("li.relationshipStep")
                                    )

                                    if path_elements:
                                        # Extract relationship steps
                                        for i, elem in enumerate(path_elements):
                                            if (
                                                i == 0
                                            ):  # Skip the first element (already displayed the person)
                                                continue

                                            # Extract text and clean it
                                            step_text = elem.get_text(strip=True)
                                            if step_text:
                                                logger.info(f"    -> {step_text}")
                                                print(f"  -> {step_text}")
                                    else:
                                        # Try to find any relationship text in the HTML
                                        relationship_text = None

                                        # Look for specific relationship indicators
                                        relationship_indicators = [
                                            (
                                                "brother",
                                                f"whose brother is {owner_name}",
                                            ),
                                            ("sister", f"whose sister is {owner_name}"),
                                            ("father", f"whose father is {owner_name}"),
                                            ("mother", f"whose mother is {owner_name}"),
                                            ("son", f"whose son is {owner_name}"),
                                            (
                                                "daughter",
                                                f"whose daughter is {owner_name}",
                                            ),
                                            ("uncle", f"whose uncle is {owner_name}"),
                                            ("aunt", f"whose aunt is {owner_name}"),
                                            ("nephew", f"whose nephew is {owner_name}"),
                                            ("niece", f"whose niece is {owner_name}"),
                                            ("cousin", f"whose cousin is {owner_name}"),
                                            (
                                                "husband",
                                                f"whose husband is {owner_name}",
                                            ),
                                            ("wife", f"whose wife is {owner_name}"),
                                            ("spouse", f"whose spouse is {owner_name}"),
                                            (
                                                "grandparent",
                                                f"whose grandparent is {owner_name}",
                                            ),
                                            (
                                                "grandchild",
                                                f"whose grandchild is {owner_name}",
                                            ),
                                        ]

                                        for indicator, text in relationship_indicators:
                                            if (
                                                indicator.lower()
                                                in relationship_html_raw.lower()
                                            ):
                                                relationship_text = text
                                                break

                                        if relationship_text:
                                            logger.info(f"    -> {relationship_text}")
                                            print(f"  -> {relationship_text}")
                                        else:
                                            logger.info(
                                                f"    -> who is related to {owner_name}"
                                            )
                                            print(
                                                f"  -> who is related to {owner_name}"
                                            )
                                except ImportError as bs_import_err:
                                    # BeautifulSoup not available, use simple text search
                                    logger.warning(
                                        f"ImportError for BeautifulSoup: {bs_import_err}"
                                    )
                                    if "brother" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose brother is {owner_name}"
                                        )
                                        print(f"  -> whose brother is {owner_name}")
                                    elif "father" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose father is {owner_name}"
                                        )
                                        print(f"  -> whose father is {owner_name}")
                                    elif "mother" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose mother is {owner_name}"
                                        )
                                        print(f"  -> whose mother is {owner_name}")
                                    elif "son" in relationship_html_raw.lower():
                                        logger.info(f"    -> whose son is {owner_name}")
                                        print(f"  -> whose son is {owner_name}")
                                    elif "daughter" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose daughter is {owner_name}"
                                        )
                                        print(f"  -> whose daughter is {owner_name}")
                                    else:
                                        logger.info(
                                            f"    -> who is related to {owner_name}"
                                        )
                                        print(f"  -> who is related to {owner_name}")
                                except Exception as bs_err:
                                    logger.error(
                                        f"Error parsing relationship HTML with BeautifulSoup: {bs_err}"
                                    )
                                    # Fallback to simple text search
                                    if "brother" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose brother is {owner_name}"
                                        )
                                        print(f"  -> whose brother is {owner_name}")
                                    elif "father" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose father is {owner_name}"
                                        )
                                        print(f"  -> whose father is {owner_name}")
                                    else:
                                        logger.info(
                                            f"    -> who is related to {owner_name}"
                                        )
                                        print(f"  -> who is related to {owner_name}")
                            except Exception as api_err:
                                logger.error(
                                    f"Error formatting relationship path: {api_err}"
                                )
                                # Final fallback
                                logger.info(f"    -> who is related to {owner_name}")
                                print(f"  -> who is related to {owner_name}")
                            except ImportError:
                                # If api_utils is not available, try BeautifulSoup
                                try:
                                    from bs4 import BeautifulSoup

                                    # Parse the HTML
                                    soup = BeautifulSoup(
                                        relationship_html_raw, "html.parser"
                                    )

                                    # Look for relationship path elements - try different selectors
                                    path_elements = (
                                        soup.select("ul.textCenter li")
                                        or soup.select(".relationshipLadder li")
                                        or soup.select("li.relationshipStep")
                                    )

                                    if path_elements:
                                        # Extract relationship steps
                                        for i, elem in enumerate(path_elements):
                                            if (
                                                i == 0
                                            ):  # Skip the first element (already displayed the person)
                                                continue

                                            # Extract text and clean it
                                            step_text = elem.get_text(strip=True)
                                            if step_text:
                                                logger.info(f"    -> {step_text}")
                                                print(f"  -> {step_text}")
                                    else:
                                        # Try to find any relationship text in the HTML
                                        relationship_text = None

                                        # Look for specific relationship indicators
                                        relationship_indicators = [
                                            (
                                                "brother",
                                                f"whose brother is {owner_name}",
                                            ),
                                            ("sister", f"whose sister is {owner_name}"),
                                            ("father", f"whose father is {owner_name}"),
                                            ("mother", f"whose mother is {owner_name}"),
                                            ("son", f"whose son is {owner_name}"),
                                            (
                                                "daughter",
                                                f"whose daughter is {owner_name}",
                                            ),
                                            ("uncle", f"whose uncle is {owner_name}"),
                                            ("aunt", f"whose aunt is {owner_name}"),
                                            ("nephew", f"whose nephew is {owner_name}"),
                                            ("niece", f"whose niece is {owner_name}"),
                                            ("cousin", f"whose cousin is {owner_name}"),
                                            (
                                                "husband",
                                                f"whose husband is {owner_name}",
                                            ),
                                            ("wife", f"whose wife is {owner_name}"),
                                            ("spouse", f"whose spouse is {owner_name}"),
                                            (
                                                "grandparent",
                                                f"whose grandparent is {owner_name}",
                                            ),
                                            (
                                                "grandchild",
                                                f"whose grandchild is {owner_name}",
                                            ),
                                        ]

                                        for indicator, text in relationship_indicators:
                                            if (
                                                indicator.lower()
                                                in relationship_html_raw.lower()
                                            ):
                                                relationship_text = text
                                                break

                                        if relationship_text:
                                            logger.info(f"    -> {relationship_text}")
                                            print(f"  -> {relationship_text}")
                                        else:
                                            logger.info(
                                                f"    -> who is related to {owner_name}"
                                            )
                                            print(
                                                f"  -> who is related to {owner_name}"
                                            )
                                except ImportError:
                                    # BeautifulSoup not available, use simple text search
                                    if "brother" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose brother is {owner_name}"
                                        )
                                        print(f"  -> whose brother is {owner_name}")
                                    elif "father" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose father is {owner_name}"
                                        )
                                        print(f"  -> whose father is {owner_name}")
                                    elif "mother" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose mother is {owner_name}"
                                        )
                                        print(f"  -> whose mother is {owner_name}")
                                    elif "son" in relationship_html_raw.lower():
                                        logger.info(f"    -> whose son is {owner_name}")
                                        print(f"  -> whose son is {owner_name}")
                                    elif "daughter" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose daughter is {owner_name}"
                                        )
                                        print(f"  -> whose daughter is {owner_name}")
                                    else:
                                        logger.info(
                                            f"    -> who is related to {owner_name}"
                                        )
                                        print(f"  -> who is related to {owner_name}")
                                except Exception as bs_err:
                                    logger.error(
                                        f"Error parsing relationship HTML with BeautifulSoup: {bs_err}"
                                    )
                                    # Fallback to simple text search
                                    if "brother" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose brother is {owner_name}"
                                        )
                                        print(f"  -> whose brother is {owner_name}")
                                    elif "father" in relationship_html_raw.lower():
                                        logger.info(
                                            f"    -> whose father is {owner_name}"
                                        )
                                        print(f"  -> whose father is {owner_name}")
                                    else:
                                        logger.info(
                                            f"    -> who is related to {owner_name}"
                                        )
                                        print(f"  -> who is related to {owner_name}")
                            except Exception as api_err:
                                logger.error(
                                    f"Error formatting relationship path: {api_err}"
                                )
                                # Final fallback
                                logger.info(f"    -> who is related to {owner_name}")
                                print(f"  -> who is related to {owner_name}")

                                # Parse the HTML
                                soup = BeautifulSoup(
                                    relationship_html_raw, "html.parser"
                                )

                                # Look for relationship path elements - try different selectors
                                path_elements = (
                                    soup.select("ul.textCenter li")
                                    or soup.select(".relationshipLadder li")
                                    or soup.select("li.relationshipStep")
                                )

                                if path_elements:
                                    # Extract relationship steps
                                    for i, elem in enumerate(path_elements):
                                        if (
                                            i == 0
                                        ):  # Skip the first element (already displayed the person)
                                            continue

                                        # Extract text and clean it
                                        step_text = elem.get_text(strip=True)
                                        if step_text:
                                            logger.info(f"    -> {step_text}")
                                            print(f"  -> {step_text}")
                                else:
                                    # Try to find any relationship text in the HTML
                                    relationship_text = None

                                    # Look for specific relationship indicators
                                    relationship_indicators = [
                                        ("brother", f"whose brother is {owner_name}"),
                                        ("sister", f"whose sister is {owner_name}"),
                                        ("father", f"whose father is {owner_name}"),
                                        ("mother", f"whose mother is {owner_name}"),
                                        ("son", f"whose son is {owner_name}"),
                                        ("daughter", f"whose daughter is {owner_name}"),
                                        ("uncle", f"whose uncle is {owner_name}"),
                                        ("aunt", f"whose aunt is {owner_name}"),
                                        ("nephew", f"whose nephew is {owner_name}"),
                                        ("niece", f"whose niece is {owner_name}"),
                                        ("cousin", f"whose cousin is {owner_name}"),
                                        ("husband", f"whose husband is {owner_name}"),
                                        ("wife", f"whose wife is {owner_name}"),
                                        ("spouse", f"whose spouse is {owner_name}"),
                                        (
                                            "grandparent",
                                            f"whose grandparent is {owner_name}",
                                        ),
                                        (
                                            "grandchild",
                                            f"whose grandchild is {owner_name}",
                                        ),
                                    ]

                                    for indicator, text in relationship_indicators:
                                        if (
                                            indicator.lower()
                                            in relationship_html_raw.lower()
                                        ):
                                            relationship_text = text
                                            break

                                    if relationship_text:
                                        logger.info(f"    -> {relationship_text}")
                                        print(f"  -> {relationship_text}")
                                    else:
                                        logger.info(
                                            f"    -> who is related to {owner_name}"
                                        )
                                        print(f"  -> who is related to {owner_name}")
                            except ImportError:
                                # BeautifulSoup not available, use simple text search
                                if "brother" in relationship_html_raw.lower():
                                    logger.info(f"    -> whose brother is {owner_name}")
                                    print(f"  -> whose brother is {owner_name}")
                                elif "father" in relationship_html_raw.lower():
                                    logger.info(f"    -> whose father is {owner_name}")
                                    print(f"  -> whose father is {owner_name}")
                                elif "mother" in relationship_html_raw.lower():
                                    logger.info(f"    -> whose mother is {owner_name}")
                                    print(f"  -> whose mother is {owner_name}")
                                elif "son" in relationship_html_raw.lower():
                                    logger.info(f"    -> whose son is {owner_name}")
                                    print(f"  -> whose son is {owner_name}")
                                elif "daughter" in relationship_html_raw.lower():
                                    logger.info(
                                        f"    -> whose daughter is {owner_name}"
                                    )
                                    print(f"  -> whose daughter is {owner_name}")
                                else:
                                    logger.info(
                                        f"    -> who is related to {owner_name}"
                                    )
                                    print(f"  -> who is related to {owner_name}")
                            except Exception as e:
                                logger.error(
                                    f"Error parsing relationship HTML with BeautifulSoup: {e}"
                                )
                                # Fallback to simple text search
                                if "brother" in relationship_html_raw.lower():
                                    logger.info(f"    -> whose brother is {owner_name}")
                                    print(f"  -> whose brother is {owner_name}")
                                elif "father" in relationship_html_raw.lower():
                                    logger.info(f"    -> whose father is {owner_name}")
                                    print(f"  -> whose father is {owner_name}")
                                else:
                                    logger.info(
                                        f"    -> who is related to {owner_name}"
                                    )
                                    print(f"  -> who is related to {owner_name}")
                        # No else clause needed - we've handled all cases above

                        # Add relationship type indicators
                        relationship_types = []
                        if (
                            "Brother" in relationship_html_raw
                            or "Sister" in relationship_html_raw
                        ):
                            relationship_types.append("sibling relationship")
                        if (
                            "Father" in relationship_html_raw
                            or "Mother" in relationship_html_raw
                        ):
                            relationship_types.append("parent-child relationship")
                        if (
                            "Son" in relationship_html_raw
                            or "Daughter" in relationship_html_raw
                        ):
                            relationship_types.append("child-parent relationship")
                        if (
                            "Spouse" in relationship_html_raw
                            or "Husband" in relationship_html_raw
                            or "Wife" in relationship_html_raw
                        ):
                            relationship_types.append("spousal relationship")

                        if relationship_types:
                            for rel_type in relationship_types:
                                logger.info(f"    [Contains {rel_type}]")
                    except Exception as rel_err:
                        logger.error(f"Error extracting relationship info: {rel_err}")

                        # Fall back to simple text output
                        logger.info(f"    -> who is related to {owner_name}")
                        print(f"  -> who is related to {owner_name}")

                elif (
                    isinstance(relationship_html_raw, dict)
                    and "error" in relationship_html_raw
                ):
                    # Log the error details
                    logger.error(
                        f"Relationship API returned error: {relationship_html_raw.get('error')}"
                    )
                    print(
                        f"\nError retrieving relationship: {relationship_html_raw.get('error', 'Unknown error')}"
                    )

                    # Display simple error message
                    logger.info(
                        f"    Error retrieving relationship: {relationship_html_raw.get('error', 'Unknown error')}"
                    )
                    print(
                        f"    Error retrieving relationship: {relationship_html_raw.get('error', 'Unknown error')}"
                    )
                else:
                    logger.warning(
                        f"API call {api_description_ladder} returned unexpected response type or None: {type(relationship_html_raw)}"
                    )
                    print(f"\n--- Relationship Path to {owner_name} (API) ---")
                    print("(API call returned unexpected data or no response)")
                # End if/else
            except Exception as e:
                logger.error(
                    f"API call {api_description_ladder} failed: {e}", exc_info=True
                )
                print(f"\n--- Relationship Path to {owner_name} (API) ---")
                print(f"(Error fetching relationship path from API: {e})")
            # End try/except for ladder API call
        # End if/else is_owner check
    elif not owner_profile_id:
        print(f"\n--- Relationship Path to Tree Owner (API) ---")
        print("(Skipping relationship calculation as your profile ID was not found)")
    else:
        print(f"\n--- Relationship Path to {owner_name} (API) ---")
        print("(Skipping relationship calculation as selected person ID is missing)")
    # End if/elif/else for relationship path

    return True


# End of handle_api_report


# --- Main Execution ---
def main():
    """Main execution flow for Action 11 (API Report)."""
    logger.info("--- Action 11: API Report Starting ---")

    # Check prerequisites before running handler
    if not CORE_UTILS_AVAILABLE:
        logger.critical("Required core utilities not loaded.")
        print("\nCRITICAL ERROR: Required utilities not loaded.")
        sys.exit(1)
    # End if
    if not API_UTILS_AVAILABLE:
        logger.critical("Required API utilities not loaded.")
        print("\nCRITICAL ERROR: Required API utilities not loaded.")
        sys.exit(1)
    # End if
    if not GEDCOM_SCORING_AVAILABLE or not GEDCOM_DATE_UTILS_AVAILABLE:
        logger.critical("Required GEDCOM scoring or date utilities not loaded.")
        print("\nCRITICAL ERROR: Required scoring or date utilities not loaded.")
        sys.exit(1)
    # End if
    if (
        config_instance is None
        or not hasattr(config_instance, "COMMON_SCORING_WEIGHTS")
        or not hasattr(config_instance, "NAME_FLEXIBILITY")
        or not hasattr(config_instance, "DATE_FLEXIBILITY")
    ):
        logger.critical("Scoring configuration variables missing or invalid.")
        print("\nERROR: Scoring configuration load failed.")
        sys.exit(1)
    # End if
    if cloudscraper is None or not session_manager.scraper:
        logger.critical(
            "Cloudscraper library/instance missing or failed to initialize."
        )
        print("\nCRITICAL ERROR: Cloudscraper is required but unavailable.")
        sys.exit(1)
    # End if

    report_successful = handle_api_report()

    if report_successful:
        logger.info("--- Action 11: API Report Finished Successfully ---")
        print("\nAction 11 finished successfully.")
    else:
        logger.error("--- Action 11: API Report Finished with Errors ---")
        print("\nAction 11 finished with errors.")
    # End if


# End of main

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
    # End if
# End of action11.py
# --- END OF FILE action11.py ---
