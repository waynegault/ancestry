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
V16.14: Switching /facts/user/ back to Cloudscraper (like V16.12) with robust cookie sync right before call.
"""
# --- Standard library imports ---
import logging
import sys
import time
import urllib.parse
import json
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

    # --- TEMPORARY HARDCODED VALUES FOR TESTING ---
    logger.info("--- Using TEMPORARY hardcoded input values for Fraser Gault ---")
    first_name = "Fraser"
    surname = "Gault"
    dob_str = "1941"
    pob = "Banff"
    dod_str = None
    pod = None
    gender_input = "m"
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
            # End try

            scraper_headers = {
                "Accept": "application/json, text/plain, */*",
                "Referer": owner_facts_referer,
            }
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
                scraper_response.raise_for_status()
                suggest_response = scraper_response.json()
                logger.info("Suggest API call successful using Cloudscraper.")
                session_manager.dynamic_rate_limiter.decrease_delay()

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
            # End inner try/except
        else:
            logger.critical(
                "Cloudscraper instance not available in SessionManager. Cannot call Suggest API."
            )
            return False
        # End if scraper

        # Check response validity *after* API call attempt
        if suggest_response is None:
            logger.error("Suggest API call failed (check previous errors).")
            print("\nError during API search. Check logs.")
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

    top_api_suggestion = suggest_response[0]
    api_person_id = top_api_suggestion.get("PersonId")  # Tree-specific ID
    api_tree_id = top_api_suggestion.get("TreeId")
    api_user_id = top_api_suggestion.get("UserId")  # Global ID (may be None)
    api_name_raw = top_api_suggestion.get("Name", "Unknown")

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

    # --- Fetch details using the /facts/user/ API (via Cloudscraper) ---
    facts_api_url = f"{base_url}/family-tree/person/facts/user/{owner_profile_id.upper()}/tree/{api_tree_id}/person/{api_person_id}"
    facts_data_raw = {}
    facts_scraper_response = None
    try:
        logger.debug(
            f"Fetching facts for {api_name_raw} from {facts_api_url} using Cloudscraper..."
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

        # --- Use Cloudscraper for Facts API ---
        scraper = session_manager.scraper
        if scraper:
            # --- Sync Cookies Again (Important!) ---
            try:
                logger.debug("Re-syncing cookies before Facts API call...")
                session_manager._sync_cookies()
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
                            f"Failed to set cookie '{cookie.name}' in cloudscraper (facts call): {set_cookie_err}"
                        )
                    # End inner try
                # End for
                logger.debug(
                    f"Synced {synced_count} cookies to Cloudscraper session (facts call)."
                )
            except Exception as sync_err:
                logger.error(
                    f"Error syncing cookies to Cloudscraper (facts call): {sync_err}",
                    exc_info=True,
                )
            # --- End cookie sync ---

            # Prepare headers for scraper
            facts_scraper_headers = {
                "Accept": "application/json, text/plain, */*",
                "Referer": facts_referer,
                "X-Requested-With": "XMLHttpRequest",  # Added header
            }

            # Apply rate limit wait
            wait_time = session_manager.dynamic_rate_limiter.wait()
            if wait_time > 0.1:
                logger.debug(
                    f"[Facts API (Cloudscraper)] Rate limit wait: {wait_time:.2f}s"
                )
            # End if

            # Make request
            try:
                facts_scraper_response = scraper.get(
                    facts_api_url,
                    headers=facts_scraper_headers,
                    timeout=60,  # Use increased timeout
                )
                facts_scraper_response.raise_for_status()
                facts_data_raw = facts_scraper_response.json()
                logger.info("Facts API call successful using Cloudscraper.")
                session_manager.dynamic_rate_limiter.decrease_delay()
            except cloudscraper.exceptions.CloudflareChallengeError as cfe:
                logger.error(
                    f"Cloudflare challenge encountered during Facts API call: {cfe}"
                )
                facts_data_raw = {}
                session_manager.dynamic_rate_limiter.increase_delay()
            except requests.exceptions.HTTPError as http_err:
                logger.error(
                    f"HTTPError during Cloudscraper Facts API call: {http_err}"
                )
                if http_err.response is not None:
                    logger.error(f"  Status Code: {http_err.response.status_code}")
                    logger.debug(f"  Response Text: {http_err.response.text[:500]}")
                # End if
                facts_data_raw = {}
            except requests.exceptions.RequestException as req_exc:
                logger.error(
                    f"RequestException during Cloudscraper Facts API call: {req_exc}"
                )
                facts_data_raw = {}
            except requests.exceptions.JSONDecodeError as json_err:
                logger.error(
                    f"Failed to decode JSON from Cloudscraper Facts API response: {json_err}"
                )
                logger.debug(
                    f"Cloudscraper Response Text: {getattr(facts_scraper_response, 'text', 'N/A')[:500]}"
                )
                facts_data_raw = {}
            except Exception as scrape_err:
                logger.error(
                    f"Unexpected error during Cloudscraper Facts API call: {scrape_err}",
                    exc_info=True,
                )
                facts_data_raw = {}
            # End inner try/except
        else:
            logger.critical(
                "Cloudscraper instance not available in SessionManager. Cannot call Facts API."
            )
            return False
        # --- END Use Cloudscraper for Facts API ---

    except Exception as e:  # Outer catch for broader issues during details fetch setup
        logger.error(f"Error preparing for Facts API call: {e}", exc_info=True)
        print(f"\nWarning: Could not fetch person facts from API.")
        facts_data_raw = {}
    # End try/except for details fetch

    # --- Extract data from facts_data_raw ---
    facts_data = facts_data_raw.get("data", {})
    if not facts_data:
        logger.error(
            f"Failed to fetch valid facts data for {api_name_raw} (PersonID: {api_person_id}). Cannot proceed."
        )
        print(
            f"\nERROR: Could not fetch details for the selected match ({api_name_raw})."
        )
        return False
    # End if

    extracted_name = format_name(facts_data.get("PersonFullName", api_name_raw))
    extracted_gender_str = facts_data.get("PersonGender")
    extracted_gender = (
        "m"
        if extracted_gender_str == "Male"
        else "f" if extracted_gender_str == "Female" else None
    )
    is_living = facts_data.get("IsPersonLiving", True)
    person_facts_list = facts_data.get("PersonFacts", [])
    person_family_data = facts_data.get("PersonFamily", {})

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
    if not match.get("is_living"):
        print(f"Died: {match['death_date']} in {match['death_place']}")
    # End if
    print(f"Link: {match['link']}")

    # --- Display Family Details ---
    print("\n--- Family Details (API) ---")
    parents_list = []
    if isinstance(person_family_data.get("Fathers"), list):
        parents_list.extend(person_family_data["Fathers"])
    # End if
    if isinstance(person_family_data.get("Mothers"), list):
        parents_list.extend(person_family_data["Mothers"])
    # End if

    siblings_list = []
    if isinstance(person_family_data.get("Siblings"), list):
        siblings_list.extend(person_family_data["Siblings"])
    # End if

    spouses_list = []
    if isinstance(person_family_data.get("Spouses"), list):
        spouses_list.extend(person_family_data["Spouses"])
    # End if

    children_list = []
    if isinstance(person_family_data.get("Children"), list):
        for child_group in person_family_data["Children"]:
            if isinstance(child_group, list):
                children_list.extend(child_group)
            # End if
        # End for
    # End if

    def format_family_member(member: Dict) -> Dict:
        name = format_name(member.get("FullName", "Unknown"))
        lifespan = member.get("LifeRange", "")
        return {"name": f"{name} ({lifespan})" if lifespan else name}

    # End of format_family_member

    print_group("Parents", [format_family_member(p) for p in parents_list])
    print_group("Siblings", [format_family_member(s) for s in siblings_list])
    print_group("Spouse(s)", [format_family_member(s) for s in spouses_list])
    print_group("Children", [format_family_member(c) for c in children_list])

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
