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
V16.7: Switched /person-picker/person API call to use cloudscraper.
"""
# --- Standard library imports ---
import logging
import sys
import time
import urllib.parse
from pathlib import Path
from urllib.parse import urljoin, urlencode

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
    from utils import SessionManager, _api_req, nav_to_page, format_name, ordinal_case

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


# --- Helper Function for Parsing Life Events ---
def _extract_life_event(
    events: List[Dict], event_type: str
) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """Extracts date string, location, and parsed date object for a specific event type."""
    date_str: Optional[str] = None
    location: Optional[str] = None
    date_obj: Optional[datetime] = None

    if not isinstance(events, list):
        return date_str, location, date_obj
    # End if

    for event in events:
        if isinstance(event, dict) and event.get("Type") == event_type:
            location = event.get("LocationText")
            date_data = event.get("Date")
            if isinstance(date_data, dict):
                year = date_data.get("Year")
                month = date_data.get("Month")
                day = date_data.get("Day")
                if year:
                    try:
                        if month and day:
                            month_str = str(month).zfill(2)
                            day_str = str(day).zfill(2)
                            temp_date_str = f"{year}-{month_str}-{day_str}"
                            date_obj = _parse_date(temp_date_str)
                            date_str = temp_date_str
                        elif month:
                            month_str = str(month).zfill(2)
                            temp_date_str = f"{year}-{month_str}"
                            date_obj = _parse_date(temp_date_str)
                            date_str = temp_date_str
                        else:
                            temp_date_str = str(year)
                            date_obj = _parse_date(temp_date_str)
                            date_str = temp_date_str
                        # End if/elif/else for date parts
                    except Exception as parse_e:
                        logger.warning(
                            f"Could not reconstruct/parse date from {date_data}: {parse_e}"
                        )
                        date_obj = None
                        date_str = None
                    # End try/except for parsing
                # End if year
            # End if date_data is dict
            break  # Found the event
        # End if
    # End for

    return date_str, location, date_obj


# End of _extract_life_event


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
    owner_profile_id = getattr(session_manager, "my_profile_id", None)
    owner_tree_id = getattr(session_manager, "my_tree_id", None)
    base_url = getattr(
        config_instance, "BASE_URL", "https://www.ancestry.co.uk"
    ).rstrip("/")

    if not owner_profile_id:
        logger.warning(
            "handle_api_report: My profile ID not available in SessionManager. Relationship path will fail."
        )
        print(
            "\nWARNING: Cannot determine your profile ID. Relationship path calculation will fail."
        )
    # End if

    # --- TEMPORARY HARDCODED VALUES FOR TESTING ---
    logger.info("--- Using TEMPORARY hardcoded input values for Fraser Gault ---")
    first_name = "Fraser"
    surname = "Gault"
    dob_str = "1941"  # Birth Date/Year
    pob = "Banff"  # Birth Place
    dod_str = None  # Death Date/Year
    pod = None  # Death Place
    gender_input = "m"  # Gender
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

    suggest_url = f"{base_url}/api/person-picker/suggest/{tree_id_for_suggest}?{'&'.join(suggest_params)}"
    logger.info(f"Attempting search using Ancestry Suggest API: {suggest_url}")
    print("Searching Ancestry API...")

    suggest_response = None
    scraper_response = None
    try:
        # Construct referer using owner's context
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

        # --- Use Cloudscraper for Suggest API ---
        logger.info("Attempting Suggest API call using Cloudscraper...")
        scraper = session_manager.scraper
        if scraper:
            # --- Explicitly sync cookies to Cloudscraper's session ---
            try:
                logger.debug(
                    "Syncing cookies from SessionManager requests session to Cloudscraper session..."
                )
                session_manager._sync_cookies()  # Ensure requests session is synced first
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
                    # End inner try
                # End for
                logger.debug(f"Synced {synced_count} cookies to Cloudscraper session.")
            except Exception as sync_err:
                logger.error(
                    f"Error syncing cookies to Cloudscraper: {sync_err}", exc_info=True
                )
            # --- End cookie sync ---

            # Prepare headers for scraper
            scraper_headers = {
                "Accept": "application/json, text/plain, */*",
                "Referer": owner_facts_referer,
            }

            # Apply rate limit wait before the call
            wait_time = session_manager.dynamic_rate_limiter.wait()
            if wait_time > 0.1:
                logger.debug(
                    f"[Suggest API (Cloudscraper)] Rate limit wait: {wait_time:.2f}s"
                )
            # End if

            try:
                scraper_response = scraper.get(
                    suggest_url, headers=scraper_headers, timeout=30
                )
                scraper_response.raise_for_status()  # Check for HTTP errors
                suggest_response = scraper_response.json()  # Parse JSON
                logger.info("Suggest API call successful using Cloudscraper.")
                session_manager.dynamic_rate_limiter.decrease_delay()  # Indicate success

            except cloudscraper.exceptions.CloudflareChallengeError as cfe:
                logger.error(
                    f"Cloudflare challenge encountered during Suggest API call: {cfe}"
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
            except requests.exceptions.JSONDecodeError as json_err:
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
            # End inner try/except for scraper call
        else:
            logger.critical(
                "Cloudscraper instance not available in SessionManager. Cannot call Suggest API."
            )
            return False
        # --- END: Use Cloudscraper for Suggest API ---

        # Check response validity
        if suggest_response is None:
            logger.error("Suggest API call failed (check previous errors).")
            print("\nError during API search. Check logs.")
            return False
        elif not isinstance(suggest_response, list) or not suggest_response:
            logger.info("No matches found via Ancestry Suggest API.")
            print("\nNo potential matches found in Ancestry API based on name.")
            return True
        # End if/elif for suggest_response check

    except Exception as e:  # Keep outer catch for broader issues
        logger.error(f"General error during API search section: {e}", exc_info=True)
        print(f"\nError during API search: {e}. Check logs.")
        return False
    # End try/except for suggest search

    # --- Process Suggest API Results and Score the Top Match ---
    top_api_suggestion = suggest_response[0]
    api_person_id = top_api_suggestion.get("PersonId")
    api_tree_id = top_api_suggestion.get("TreeId")
    api_name_raw = top_api_suggestion.get("Name", "Unknown")

    if not api_person_id or not api_tree_id:
        logger.error("Suggest API result missing critical IDs (PersonId or TreeId).")
        print("\nError processing top API search result (missing IDs).")
        return False
    # End if

    logger.debug(
        f"Processing top Suggest API match: {api_name_raw} (ID: {api_person_id}, TreeID: {api_tree_id})"
    )

    # --- Fetch details using the /person-picker/person API (via Cloudscraper) ---
    person_details_url = (
        f"{base_url}/api/person-picker/person/{api_tree_id}/{api_person_id}/"
    )
    person_details_data = {}
    details_scraper_response = None  # Initialize response var
    try:
        logger.debug(
            f"Fetching person details for {api_person_id} from {person_details_url} using Cloudscraper..."
        )
        details_referer = None
        if tree_id_for_suggest and owner_profile_id:
            details_referer = urljoin(
                base_url,
                f"/family-tree/tree/{tree_id_for_suggest}/person/{owner_profile_id}/facts",
            )
        else:
            details_referer = base_url
        # End if

        # --- Use Cloudscraper for Person Details API ---
        scraper = session_manager.scraper
        if scraper:
            # --- Sync Cookies Again (Important!) ---
            try:
                logger.debug("Re-syncing cookies before Person Details API call...")
                session_manager._sync_cookies()  # Ensure requests session is synced first
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
                            f"Failed to set cookie '{cookie.name}' in cloudscraper (details call): {set_cookie_err}"
                        )
                    # End inner try
                # End for
                logger.debug(
                    f"Synced {synced_count} cookies to Cloudscraper session (details call)."
                )
            except Exception as sync_err:
                logger.error(
                    f"Error syncing cookies to Cloudscraper (details call): {sync_err}",
                    exc_info=True,
                )
            # --- End cookie sync ---

            # Prepare headers for scraper
            details_scraper_headers = {
                "Accept": "application/json, text/plain, */*",
                "Referer": details_referer,
            }

            # Apply rate limit wait
            wait_time = session_manager.dynamic_rate_limiter.wait()
            if wait_time > 0.1:
                logger.debug(
                    f"[Person Details API (Cloudscraper)] Rate limit wait: {wait_time:.2f}s"
                )
            # End if

            try:
                details_scraper_response = scraper.get(
                    person_details_url,
                    headers=details_scraper_headers,
                    timeout=30,  # Use increased timeout
                )
                details_scraper_response.raise_for_status()
                person_details_data = details_scraper_response.json()
                logger.info("Person Details API call successful using Cloudscraper.")
                session_manager.dynamic_rate_limiter.decrease_delay()

            except cloudscraper.exceptions.CloudflareChallengeError as cfe:
                logger.error(
                    f"Cloudflare challenge encountered during Person Details API call: {cfe}"
                )
                person_details_data = {}
                session_manager.dynamic_rate_limiter.increase_delay()
            except requests.exceptions.HTTPError as http_err:
                logger.error(
                    f"HTTPError during Cloudscraper Person Details API call: {http_err}"
                )
                if http_err.response is not None:
                    logger.error(f"  Status Code: {http_err.response.status_code}")
                    logger.debug(f"  Response Text: {http_err.response.text[:500]}")
                person_details_data = {}
            except requests.exceptions.RequestException as req_exc:
                logger.error(
                    f"RequestException during Cloudscraper Person Details API call: {req_exc}"
                )
                person_details_data = {}
            except requests.exceptions.JSONDecodeError as json_err:
                logger.error(
                    f"Failed to decode JSON from Cloudscraper Person Details API response: {json_err}"
                )
                logger.debug(
                    f"Cloudscraper Response Text: {getattr(details_scraper_response, 'text', 'N/A')[:500]}"
                )
                person_details_data = {}
            except Exception as scrape_err:
                logger.error(
                    f"Unexpected error during Cloudscraper Person Details API call: {scrape_err}",
                    exc_info=True,
                )
                person_details_data = {}
            # End inner try/except for scraper details call
        else:
            logger.critical(
                "Cloudscraper instance not available in SessionManager. Cannot call Person Details API."
            )
            return False
        # --- END: Use Cloudscraper for Person Details API ---

    except Exception as e:  # Outer catch for broader issues during details fetch setup
        logger.error(f"Error preparing for Person Details API call: {e}", exc_info=True)
        print(f"\nWarning: Could not fetch person details from API.")
        person_details_data = {}
    # End try/except for details fetch

    # --- Extract data directly from person_details_data ---
    # Check if data was actually fetched
    if not person_details_data:
        logger.error(
            f"Failed to fetch details for {api_name_raw} (ID: {api_person_id}). Cannot proceed."
        )
        print(
            f"\nERROR: Could not fetch details for the selected match ({api_name_raw})."
        )
        return False  # Cannot proceed without details
    # End if

    extracted_name = format_name(
        f"{person_details_data.get('GivenName', '')} {person_details_data.get('Surname', '')}".strip()
    )
    extracted_gender = person_details_data.get("Gender")
    life_events = person_details_data.get("LifeEvents", [])

    birth_date_str, birth_place, birth_date_obj = _extract_life_event(life_events, "b")
    death_date_str, death_place, death_date_obj = _extract_life_event(life_events, "d")

    birth_date_disp = _clean_display_date(birth_date_str) if birth_date_str else "N/A"
    death_date_disp = _clean_display_date(death_date_str) if death_date_str else "N/A"

    # --- Prepare candidate data dictionary for scoring ---
    candidate_data_dict = {
        "first_name": clean_param(person_details_data.get("GivenName")),
        "surname": clean_param(person_details_data.get("Surname")),
        "birth_year": birth_date_obj.year if birth_date_obj else None,
        "birth_date_obj": birth_date_obj,
        "birth_place": clean_param(birth_place),
        "death_year": death_date_obj.year if death_date_obj else None,
        "death_date_obj": death_date_obj,
        "death_place": clean_param(death_place),
        "gender": extracted_gender if extracted_gender in ["m", "f"] else None,
    }

    # --- Calculate score ---
    score = 0.0
    reasons = ["API Suggest Match"]
    field_scores = {}

    score, field_scores, reasons_list = calculate_match_score(
        search_criteria_dict,
        candidate_data_dict,
        config_instance.COMMON_SCORING_WEIGHTS,
        config_instance.NAME_FLEXIBILITY,
        config_instance.DATE_FLEXIBILITY,
    )
    if "API Suggest Match" not in reasons_list:
        reasons_list.append("API Suggest Match")
    # End if
    reasons = reasons_list
    logger.debug(f"Calculated score for top API match: {score} (Reasons: {reasons})")

    # --- Create match dict ---
    person_link = "(unavailable)"
    if api_tree_id and api_person_id and base_url:
        person_link = f"{base_url}/family-tree/person/tree/{api_tree_id}/person/{api_person_id}/facts"
    # End if

    api_match_for_display = {
        "id": api_person_id,
        "norm_id": api_person_id,
        "tree_id": api_tree_id,
        "name": extracted_name if extracted_name != "Valued Relative" else api_name_raw,
        "birth_date": birth_date_disp,
        "birth_place": birth_place or "N/A",
        "death_date": death_date_disp,
        "death_place": death_place or "N/A",
        "score": score,
        "reasons": ", ".join(reasons),
        "link": person_link,
        "is_living": death_date_obj is None and death_date_str is None,
        "person_id": api_person_id,
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
    if (
        match.get("death_date") != "N/A"
        or match.get("death_place") != "N/A"
        or match.get("is_living") is False
    ):
        print(f"     Died : {match.get('death_date', '?')} {death_place_info}")
    # End if
    print(
        f"     Score: {match.get('score', 0.0):.0f} (Reasons: {match.get('reasons', 'API Suggest Match')})"
    )

    print(f"\n---> Auto-selecting this match: {match['name']}")
    selected_match = api_match_for_display

    # --- Display Details ---
    print("\n=== PERSON DETAILS (API) ===")
    print(f"Name: {match['name']}")
    gender_display = (
        "Male"
        if extracted_gender == "m"
        else "Female" if extracted_gender == "f" else "N/A"
    )
    print(f"Gender: {gender_display}")
    print(f"Born: {match['birth_date']} in {match['birth_place']}")
    if (
        match.get("death_date") != "N/A"
        or match.get("death_place") != "N/A"
        or match.get("is_living") is False
    ):
        print(f"Died: {match['death_date']} in {match['death_place']}")
    # End if
    print(f"Link: {match['link']}")

    # --- Display Family Details ---
    print("\n--- Family Details (API) ---")
    family_members = person_details_data.get("FamilyMembers", [])
    parents_list = []
    siblings_list = []
    spouses_list = []
    children_list = []

    for member in family_members:
        if isinstance(member, dict):
            rel_type = member.get("Relationship")
            member_name = format_name(
                f"{member.get('GivenName', '')} {member.get('Surname', '')}".strip()
            )
            if member_name == "Valued Relative":
                continue
            # End if
            member_info = {"name": member_name}

            if rel_type == "f" or rel_type == "m":
                parents_list.append(member_info)
            elif rel_type == "b":  # Brother
                siblings_list.append(member_info)
            elif rel_type == "s" and member.get("Gender") == "f":  # Sister
                siblings_list.append(member_info)
            elif rel_type == "s":  # Spouse
                spouses_list.append(member_info)
            elif rel_type == "c":
                children_list.append(member_info)
            # End if/elif
        # End if
    # End for

    print_group("Parents", parents_list)
    print_group("Siblings", siblings_list)
    print_group("Spouse(s)", spouses_list)
    print_group("Children", children_list)

    # --- Display Relationship Path ---
    selected_person_id = selected_match.get("person_id")
    selected_tree_id = selected_match.get("tree_id")

    if owner_profile_id and selected_person_id:
        if owner_profile_id.upper() == selected_person_id.upper():
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
                ladder_api_url = f"{base_url}/family-tree/person/tree/{owner_tree_id}/person/{selected_person_id}/getladder"
                api_description_ladder = "Get Tree Ladder API (Action 11)"
                callback_name = f"__ancestry_jsonp_{int(time.time()*1000)}"
                timestamp_ms = int(time.time() * 1000)
                query_params = urlencode({"callback": callback_name, "_": timestamp_ms})
                ladder_api_url = f"{ladder_api_url}?{query_params}"
                ladder_headers = {
                    "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
                    "X-Requested-With": "XMLHttpRequest",
                }
                ladder_referer = f"{base_url}/family-tree/person/tree/{owner_tree_id}/person/{selected_person_id}/facts"
                logger.debug(f"Using Tree Ladder API: {ladder_api_url}")
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

                ladder_api_url = f"{base_url}/discoveryui-matchesservice/api/samples/{selected_person_id}/relationshiptome/{owner_profile_id}"
                api_description_ladder = "API Relationship Ladder (Batch)"
                ladder_referer = urljoin(
                    base_url, f"/discoveryui-matches/list/{session_manager.my_uuid}"
                )
                logger.debug(f"Using Discovery/Batch Ladder API: {ladder_api_url}")
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

                if relationship_html_raw and isinstance(relationship_html_raw, str):
                    display_raw_relationship_ladder(
                        relationship_html_raw,
                        owner_name,
                        selected_match.get("name", "Selected Person"),
                    )
                elif (
                    isinstance(relationship_html_raw, dict)
                    and "error" in relationship_html_raw
                ):
                    display_raw_relationship_ladder(
                        relationship_html_raw,
                        owner_name,
                        selected_match.get("name", "Selected Person"),
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
        # End if/else for relationship calculation
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
