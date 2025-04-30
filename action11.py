# action11.py
"""
Action 11: API Report - Search Ancestry API, display details, family, relationship.
V16.0: Refactored from temp.py v7.36, using functions from utils.py, api_utils.py, gedcom_utils.py.
Implements consistent scoring and output format with Action 10.
"""
# --- Standard library imports ---
import logging
import sys
import os
import re
from pathlib import Path
import time
import json
import urllib.parse
import random  # Keep for retry jitter in utils (if not used there)
import html  # Keep if needed for parsing directly in action11

# from typing import Optional, List, Dict, Any, Tuple # Keep types needed locally

# Import specific types needed locally
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime  # Import datetime for date comparisons/objects

# import inspect # Keep for config path if needed

# --- Third-party imports ---
try:
    from bs4 import (
        BeautifulSoup,
    )  # Import BeautifulSoup if needed directly for any reason, otherwise api_utils uses it
except ImportError:
    # print("Warning: BeautifulSoup not found. Relationship ladder parsing might fail.") # This message should ideally come from api_utils
    BeautifulSoup = None  # Set to None if import fails

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


# --- Load Config and SCORING CONSTANTS (Mandatory - Direct Import) ---
config_instance = None
COMMON_SCORING_WEIGHTS = None
NAME_FLEXIBILITY = None
DATE_FLEXIBILITY = None
try:
    # Import instance and scoring dicts directly from config
    from config import (
        config_instance,
        COMMON_SCORING_WEIGHTS,
        NAME_FLEXIBILITY,
        DATE_FLEXIBILITY,
        USER_AGENTS,  # Import for potential use in headers
    )

    logger.info("Successfully imported config_instance and scoring dictionaries.")
    # Basic validation that scoring weights are dictionaries
    if (
        not isinstance(COMMON_SCORING_WEIGHTS, dict)
        or not isinstance(NAME_FLEXIBILITY, dict)
        or not isinstance(DATE_FLEXIBILITY, dict)
    ):
        raise TypeError(
            "One or more scoring configurations imported from config.py is not a dictionary."
        )
    # Optional warning if weights dict is empty
    if not COMMON_SCORING_WEIGHTS:
        logger.warning(
            "COMMON_SCORING_WEIGHTS dictionary imported from config.py is empty. Scoring may not function as expected."
        )

except ImportError as e:
    logger.critical(
        f"Failed to import config_instance or scoring dictionaries from config.py: {e}. Cannot proceed.",
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

# Ensure critical config components are loaded before proceeding
if (
    config_instance is None
    or COMMON_SCORING_WEIGHTS is None
    or NAME_FLEXIBILITY is None
    or DATE_FLEXIBILITY is None
):
    logger.critical("One or more critical configuration components failed to load.")
    print("\nFATAL ERROR: Configuration load failed.")
    sys.exit(1)


# --- Import GEDCOM Utilities (for scoring and date helpers) ---
# Import specific functions needed from gedcom_utils
calculate_match_score = None
_parse_date = None
_clean_display_date = None
GEDCOM_DATE_UTILS_AVAILABLE = False  # Flag for date helpers
GEDCOM_SCORING_AVAILABLE = False  # Flag for scoring function

try:
    # Attempt to import the specific functions
    from gedcom_utils import calculate_match_score, _parse_date, _clean_display_date

    logger.info(
        "Successfully imported calculate_match_score and date helpers from gedcom_utils."
    )

    # Check availability of imported functions
    if calculate_match_score is not None:
        GEDCOM_SCORING_AVAILABLE = True
    else:
        logger.error("calculate_match_score NOT FOUND in gedcom_utils after import.")

    if _parse_date is not None and _clean_display_date is not None:
        GEDCOM_DATE_UTILS_AVAILABLE = True
    else:
        logger.error(
            "_parse_date or _clean_display_date NOT FOUND in gedcom_utils after import."
        )

except ImportError as e:
    logger.error(
        f"Failed to import required scoring or date functions from gedcom_utils: {e}.",
        exc_info=True,
    )
    # Variables remain None/False as initialized

# --- Import API Utilities ---
parse_ancestry_person_details = None
print_group = None
display_raw_relationship_ladder = None  # This function handles parsing/displaying HTML
# format_api_relationship_path = None # display_raw_relationship_ladder calls this internally now

API_UTILS_AVAILABLE = False

try:
    # Attempt to import the specific functions
    from api_utils import (
        parse_ancestry_person_details,
        print_group,
        display_raw_relationship_ladder,
    )

    logger.info("Successfully imported required functions from api_utils.")

    # Check availability of imported functions
    if (
        parse_ancestry_person_details is not None
        and print_group is not None
        and display_raw_relationship_ladder is not None
    ):
        API_UTILS_AVAILABLE = True
    else:
        logger.error("One or more core functions NOT FOUND in api_utils after import.")

except ImportError as e:
    logger.error(
        f"Failed to import required functions from api_utils: {e}.", exc_info=True
    )
    # Variables remain None/False as initialized


# --- Import General Utilities ---
# SessionManager, _api_req, nav_to_page are critical for API actions
SessionManager = None
_api_req = None
nav_to_page = None
format_name = lambda x: str(x)  # Fallback for format_name
ordinal_case = lambda x: str(x)  # Fallback for ordinal_case

CORE_UTILS_AVAILABLE = False

try:
    # Import the core utility components
    from utils import SessionManager, _api_req, nav_to_page, format_name, ordinal_case

    logger.info("Successfully imported required functions from utils.")
    CORE_UTILS_AVAILABLE = True

except ImportError as e:
    logger.critical(
        f"Failed to import critical components from 'utils' module: {e}. API functions unavailable.",
        exc_info=True,
    )
    print(f"FATAL ERROR: Failed to import required functions from utils.py: {e}")

    # Define dummy SessionManager and _api_req to prevent crashes if import fails
    class SessionManager:
        def __init__(self):
            self.driver_live = False
            self.session_ready = False
            self.my_tree_id = None
            self.my_profile_id = None
            self.driver = None
            self.my_uuid = None
            self.tree_owner_name = "Unknown (Utils missing)"
            # Ensure necessary attributes are present even if dummy
            self.csrf_token = None

        # End of __init__
        def ensure_driver_live(self):
            logger.error("Dummy SessionManager: ensure_driver_live called.")
            return False

        # End of ensure_driver_live
        def ensure_session_ready(self):
            logger.error("Dummy SessionManager: ensure_session_ready called.")
            return False

        # End of ensure_session_ready
        def check_session_status(self):
            logger.error("Dummy SessionManager: check_session_status called.")
            pass

        # End of check_session_status
        def _retrieve_identifiers(self):
            logger.error("Dummy SessionManager: _retrieve_identifiers called.")
            return False

        # End of _retrieve_identifiers
        def _sync_cookies(self):
            logger.error("Dummy SessionManager: _sync_cookies called.")
            pass

        # End of _sync_cookies
        def get_csrf(self):
            logger.error("Dummy SessionManager: get_csrf called.")
            return None

        # End of get_csrf
        # Add dummy get_cookies to prevent crash in _api_req
        def get_cookies(self, cookie_names: List[str], timeout: int = 30) -> bool:
            logger.error("Dummy SessionManager: get_cookies called.")
            return False

        # Add dummy is_sess_valid to prevent crash in _api_req
        def is_sess_valid(self) -> bool:
            logger.error("Dummy SessionManager: is_sess_valid called.")
            return False

    # End of SessionManager dummy class
    def _api_req(*args, **kwargs):
        logger.error(
            "Dummy _api_req: API request attempted but core 'utils' module is missing."
        )

        # Return a response-like object with error info
        class DummyResponse:
            status_code = 500
            reason = "Utils Module Missing"
            text = "Error: Required utils module is missing."

            def json(self):
                return {"error": self.text}

            ok = False
            headers = {}

        return DummyResponse()  # Return dummy response

    # End of _api_req dummy function
    nav_to_page = None  # type: ignore
    # format_name and ordinal_case fallbacks already defined above

# --- Constants ---
# MAX_DISPLAY_MATCHES is handled by action10.handle_gedcom_report parameter,
# Action11 logic auto-selects top API match after searching, but we will
# adjust it to display the top scored match consistently.

# --- Session Manager Instance ---
# Create a standalone session_manager instance from utils if available
session_manager: SessionManager = (
    SessionManager()
)  # Instantiate the real or dummy SessionManager


# --- Main Handler ---
def handle_api_report():
    """
    Handler for Action 11 - API Report.
    Searches Ancestry API, displays details, family, relationship to Tree Owner.
    Uses functions from utils.py, api_utils.py, and gedcom_utils.py (for scoring/dates).
    """
    global COMMON_SCORING_WEIGHTS, NAME_FLEXIBILITY, DATE_FLEXIBILITY
    logger.info(
        "\n--- Person Details & Relationship to Tree Owner (Ancestry API Report) ---"
    )

    # Check essential dependencies
    if not CORE_UTILS_AVAILABLE:
        logger.error("handle_api_report: Core utils module unavailable.")
        print("\nERROR: Core utilities (SessionManager, API req) unavailable.")
        return False  # Indicate failure
    if not API_UTILS_AVAILABLE:
        logger.error("handle_api_report: API utils module unavailable.")
        print("\nERROR: API utilities (parsing, formatting) unavailable.")
        return False  # Indicate failure
    # Scoring and date helpers are not strictly fatal for API Report *if* user doesn't enter search criteria,
    # but required for consistent match display and scoring.
    if not GEDCOM_SCORING_AVAILABLE:
        logger.warning(
            "handle_api_report: GEDCOM scoring function unavailable. API match scoring will be skipped."
        )
        # Allow proceeding, but scoring info won't be displayed
    if not GEDCOM_DATE_UTILS_AVAILABLE:
        logger.warning(
            "handle_api_report: GEDCOM date utilities unavailable. API date formatting may be basic."
        )
        # Allow proceeding, but date formatting might be inconsistent

    # --- Initialize API Session ---
    # Use the global session_manager instance and ensure it's ready
    print("Initializing Ancestry session...")  # User feedback
    session_init_ok = session_manager.ensure_session_ready(
        action_name="API Report Session Init"
    )
    if not session_init_ok:
        logger.error("Failed to initialize Ancestry session for API report.")
        print(
            "\nERROR: Failed to initialize session. Cannot proceed with API operations."
        )
        return False  # Indicate failure

    # Retrieve tree owner name from session_manager
    owner_name = getattr(session_manager, "tree_owner_name", "the Tree Owner")
    owner_profile_id = getattr(session_manager, "my_profile_id", None)
    owner_tree_id = getattr(session_manager, "my_tree_id", None)

    if not owner_profile_id:
        logger.error(
            "handle_api_report: My profile ID not available in SessionManager."
        )
        print(
            "\nERROR: Cannot determine your profile ID. API features requiring it (like relationship path) will fail."
        )
        # Continue, but some features will be limited

    # --- Prompt for search criteria ---
    logger.info(
        "\nEnter search criteria for the person of interest (Ancestry API Search):"
    )
    # Use try/except for input to handle EOFError (Ctrl+D) or other input issues
    try:
        first_name = input(" First Name (optional): ").strip() or None
        surname = input(" Surname (optional): ").strip() or None
        dob_str = input(" Birth Date/Year (optional): ").strip() or None
        pob = input(" Birth Place (optional): ").strip() or None
        dod_str = (
            input(" Death Date/Year (optional): ").strip() or None
        )  # Added Death Date input
        pod = (
            input(" Death Place (optional): ").strip() or None
        )  # Added Death Place input
        gender = input(" Gender (M/F, optional): ").strip() or None
        if gender:
            gender = (
                gender[0].lower() if gender[0].lower() in ["m", "f"] else None
            )  # Normalize gender input
    except EOFError:
        print("\nInput cancelled.")
        return False  # Indicate operation cancelled
    except Exception as input_err:
        logger.error(f"Error reading input: {input_err}", exc_info=True)
        print("\nError reading input.")
        return False  # Indicate failure

    # API Suggest requires at least first name or surname
    if not (first_name or surname):
        logger.info("\nAPI search needs First Name or Surname. Report cancelled.")
        print("\nAPI search needs First Name or Surname. Report cancelled.")
        return True  # Indicate success (cancelled by user intent)

    # Prepare search criteria for scoring
    clean_param = lambda p: (
        p.strip().lower() if p and isinstance(p, str) else None
    )  # Clean and lowercase input strings

    target_first_name_lower = clean_param(first_name)
    target_surname_lower = clean_param(surname)
    target_pob_lower = clean_param(pob)
    target_pod_lower = clean_param(pod)
    target_gender_clean = gender  # m or f or None

    # Parse target dates/years using imported helpers
    target_birth_year: Optional[int] = None
    target_birth_date_obj: Optional[datetime] = None
    if dob_str and GEDCOM_DATE_UTILS_AVAILABLE:
        target_birth_date_obj = _parse_date(dob_str)
        if target_birth_date_obj:
            target_birth_year = target_birth_date_obj.year

    target_death_year: Optional[int] = None
    target_death_date_obj: Optional[datetime] = None
    if dod_str and GEDCOM_DATE_UTILS_AVAILABLE:
        target_death_date_obj = _parse_date(dod_str)
        if target_death_date_obj:
            target_death_year = target_death_date_obj.year

    # Prepare search criteria dictionary for scoring function
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

    # --- Use person-picker/suggest API to find candidates ---
    # This API is used to get candidate PersonIds based on name search.
    # It typically returns a limited number of results (often just the top one).
    tree_id_for_suggest = owner_tree_id  # Suggest API needs tree ID
    if not tree_id_for_suggest:
        logger.error(
            "Cannot perform API search: My tree ID is not available in SessionManager."
        )
        print(
            "\nERROR: Cannot determine your tree ID. API search functionality is limited."
        )
        return False  # Indicate failure

    base_url = getattr(
        config_instance, "BASE_URL", "https://www.ancestry.co.uk"
    ).rstrip("/")
    suggest_params = []
    if first_name:
        suggest_params.append(f"partialFirstName={urllib.parse.quote(first_name)}")
    if surname:
        suggest_params.append(f"partialLastName={urllib.parse.quote(surname)}")

    suggest_url = f"{base_url}/api/person-picker/suggest/{tree_id_for_suggest}?{'&'.join(suggest_params)}"
    logger.info(f"Attempting search using Ancestry Suggest API...")
    print("Searching Ancestry API...")  # User feedback

    suggest_response = None
    try:
        suggest_response = _api_req(
            url=suggest_url,
            driver=session_manager.driver,  # Pass the actual driver from the session manager
            session_manager=session_manager,  # Pass the session manager instance
            method="GET",
            api_description="Person Picker Suggest API",
            # Headers are handled automatically by _api_req, but can provide context-specific ones
            # Add specific referer if needed, otherwise _api_req uses BASE_URL
            # referer_url=urljoin(base_url, f"/family-tree/tree/{tree_id_for_suggest}/person/{owner_profile_id}/facts") # Example referer
            timeout=15,  # Shorter timeout for this quick API
        )
        # suggest_response is expected to be a list of person dictionaries or None on error/no result
        if not isinstance(suggest_response, list) or not suggest_response:
            logger.info("No matches found via Ancestry Suggest API.")
            print("\nNo potential matches found in Ancestry API based on name.")
            return True  # Indicate success (no matches is a valid outcome)

    except Exception as e:
        logger.error(f"API /suggest call failed: {e}", exc_info=True)
        print(f"\nError during API search: {e}. Check logs.")
        return False  # Indicate failure

    # --- Process Suggest API Results and Score the Top Match ---
    # We will only process and score the top suggestion from the API for now
    # To fully replicate GEDCOM report (multiple matches + selection), we would need
    # to fetch details and score *all* suggestions returned by the API, which can be slow.
    # Sticking to scoring/displaying the top one as implemented in temp.py API handler.

    top_api_suggestion = suggest_response[0]
    api_person_id = top_api_suggestion.get("PersonId")
    api_tree_id = top_api_suggestion.get(
        "TreeId"
    )  # Tree ID from the search result person (might be different from user's tree ID)
    api_name_raw = top_api_suggestion.get("Name", "Unknown")

    if not api_person_id or not api_tree_id:
        logger.error("Suggest API result missing critical IDs (PersonId or TreeId).")
        print("\nError processing top API search result (missing IDs).")
        return False  # Indicate failure

    logger.debug(
        f"Processing top Suggest API match: {api_name_raw} (ID: {api_person_id})"
    )

    # Fetch more details for scoring and display using Person Card API
    # The Person Card API gives birth/death dates/places and family summary
    person_card_url = f"{base_url}/api/search-results/person-card/tree/{api_tree_id}/person/{api_person_id}"
    person_card_data = {}  # Use empty dict as default
    try:
        logger.debug(f"Fetching person card for {api_person_id} from {person_card_url}")
        person_card_data = _api_req(
            url=person_card_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            api_description="Person Card API",
            # Referer needed for this API context
            referer_url=urljoin(
                base_url,
                f"/family-tree/tree/{tree_id_for_suggest}/person/{owner_profile_id}/facts",
            ),
            timeout=15,
        )
        # Ensure the response is a dictionary
        if not isinstance(person_card_data, dict):
            logger.warning(
                f"Person Card API did not return a dictionary for {api_person_id}. Received type: {type(person_card_data)}"
            )
            person_card_data = {}  # Reset to empty dict if unexpected format
    except Exception as e:
        logger.error(
            f"API /person-card call failed for {api_person_id}: {e}", exc_info=True
        )
        print(f"\nWarning: Could not fetch full details from API for scoring/display.")
        # person_card_data remains {}

    # Parse the fetched data to get structured details (using api_utils helper)
    # Pass empty dict/None if API calls failed
    parsed_api_details = parse_ancestry_person_details(
        person_card_data, {}
    )  # Pass empty facts for now if not fetched separately

    # Prepare candidate data dictionary for scoring (using parsed details)
    candidate_data_dict = {
        "first_name": (
            parsed_api_details.get("name", "").split(" ")[0].lower()
            if parsed_api_details.get("name")
            else None
        ),
        "surname": (
            parsed_api_details.get("name", "").split(" ")[-1].lower()
            if parsed_api_details.get("name")
            and len(parsed_api_details.get("name", "").split(" ")) > 1
            else None
        ),
        "birth_year": (
            parsed_api_details.get("api_birth_obj").year
            if parsed_api_details.get("api_birth_obj")
            else None
        ),
        "birth_date_obj": parsed_api_details.get("api_birth_obj"),
        "birth_place": (
            parsed_api_details.get("birth_place").lower()
            if parsed_api_details.get("birth_place")
            and parsed_api_details.get("birth_place") != "N/A"
            else None
        ),
        "death_year": (
            parsed_api_details.get("api_death_obj").year
            if parsed_api_details.get("api_death_obj")
            else None
        ),
        "death_date_obj": parsed_api_details.get("api_death_obj"),
        "death_place": (
            parsed_api_details.get("death_place").lower()
            if parsed_api_details.get("death_place")
            and parsed_api_details.get("death_place") != "N/A"
            else None
        ),
        "gender": (
            parsed_api_details.get("gender").lower()
            if parsed_api_details.get("gender")
            else None
        ),
    }

    # Calculate score using the common scoring function (if available)
    score = 0
    reasons = ["API Suggest Match"]  # Base reason
    if GEDCOM_SCORING_AVAILABLE:
        score, reasons_list = calculate_match_score(
            search_criteria_dict,
            candidate_data_dict,
            COMMON_SCORING_WEIGHTS,
            NAME_FLEXIBILITY,
            DATE_FLEXIBILITY,
        )
        # Add "API Suggest Match" if not already in reasons (unlikely if scoring uses this)
        if "API Suggest Match" not in reasons_list:
            reasons_list.append("API Suggest Match")
        reasons = reasons_list
        logger.debug(
            f"Calculated score for top API match: {score} (Reasons: {reasons})"
        )
    else:
        logger.warning("Scoring function unavailable. Cannot score API match.")
        # Score remains 0, reasons is just "API Suggest Match"

    # Create a match dict for display, similar to GEDCOM results
    api_match_for_display = {
        "id": api_person_id,  # Use Person ID as ID
        "norm_id": api_person_id,  # Use Person ID as normalized ID
        "tree_id": api_tree_id,
        "name": parsed_api_details.get(
            "name", api_name_raw
        ),  # Use parsed name if available, else raw
        "birth_date": parsed_api_details.get("birth_date", "N/A"),
        "birth_place": parsed_api_details.get("birth_place", "N/A"),
        "death_date": parsed_api_details.get("death_date", "N/A"),
        "death_place": parsed_api_details.get("death_place", "N/A"),
        "score": score,
        "reasons": ", ".join(reasons),
        "link": parsed_api_details.get("link"),
        "is_living": parsed_api_details.get("is_living"),
    }

    # --- Display the Top Scored Match (consistent with Action 10) ---
    print(f"\n--- Top Match (Scored) ---")
    # Always display the one scored match
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
    date_info = (
        f" ({', '.join(filter(None, [b_info, d_info]))}" if b_info or d_info else ""
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

    # Display using format consistent with Action 10
    print(f"  1. {match['name']}")
    print(f"     Born : {match.get('birth_date', '?')} {birth_place_info}")
    # Only print death line if there is death info or marked as not living
    if (
        match.get("death_date") != "N/A"
        or match.get("death_place") != "N/A"
        or match.get("is_living") is False
    ):
        print(f"     Died : {match.get('death_date', '?')} {death_place_info}")

    if GEDCOM_SCORING_AVAILABLE:
        print(
            f"     Score: {match.get('score', '?')} (Reasons: {match.get('reasons', 'API Suggest Match')})"
        )
    else:
        print(
            f"     Score: (Scoring unavailable) (Reasons: {match.get('reasons', 'API Suggest Match')})"
        )

    print(f"\n---> Auto-selecting this match: {match['name']}")
    selected_match = api_match_for_display  # Auto-select the only one

    # --- Fetch Full Details and Family using API (if not already done by person-card) ---
    # We already have person_card_data. We might need facts_data for more complete family details.
    # The 'facts' API endpoint often has a different structure and requires a CSRF token.
    # This endpoint is https://www.ancestry.co.uk/tree/{treeId}/person/{personId}/facts (navigating here gets the data via JS/internal APIs)
    # There might be a direct API endpoint for facts data, but the Person Card API already provides much of it.
    # Let's check if facts_data is significantly different from person_card_data. Temp.py v7.36 used person-card and facts API, let's replicate that pattern if facts adds value.
    # Re-checking temp.py: it fetches person-card AND facts data (`/tree/{treeId}/person/{personId}/facts` which is a web page, then tries to use it). Let's use the explicit facts API mentioned in action11.py provided initially.

    # Use the explicit facts API endpoint from the initial action11.py
    # This requires MyProfileID to be available from the SessionManager
    if owner_profile_id:
        facts_api_url = f"{base_url}/app-api/express/v1/profiles/details?userId={api_person_id.upper()}"  # This looks more like a profile details API, not facts
        # The initial action11.py used `/tree/{tree_id}/person/{person_id}/facts` which loads an HTML page that contains JSON data embedded in it, or fetched by JS.
        # Let's use the API endpoint identified in the initial action11.py that seems to give facts: `/app-api/express/v1/profiles/details?userId={profile_id.upper()}`
        facts_data = None
        api_description_facts = "API Profile Details (Facts-like)"
        logger.debug(
            f"Attempting to fetch facts-like data from: {facts_api_url} ({api_description_facts})"
        )
        try:
            # This API might not need CSRF, but requires session cookies
            facts_data = _api_req(
                url=facts_api_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                headers={},  # _api_req adds defaults + dynamic
                use_csrf_token=False,  # This API likely doesn't need CSRF
                api_description=api_description_facts,
                # Referer needed for context, use a likely page user would be on
                referer_url=urljoin(
                    base_url,
                    f"/family-tree/tree/{tree_id_for_suggest}/person/{owner_profile_id}/facts",
                ),
                timeout=15,
            )
            if not isinstance(facts_data, dict):
                logger.warning(
                    f"{api_description_facts} did not return a dictionary for {api_person_id}. Received type: {type(facts_data)}"
                )
                facts_data = None  # Reset to None if unexpected format
            else:
                logger.debug(
                    f"Successfully fetched {api_description_facts} for {api_person_id}."
                )
                # Need to check structure for family members if needed from here
                # The Person Card API already provides father, mother, selectedSpouse, children under spouse.
                # The profile details API might have a different structure.
                # Let's stick to Person Card API for family details as in temp.py v7.36 if it provides enough.
                # Reviewing temp.py v7.36 family display: it uses person_card_data's 'father', 'mother', 'selectedSpouse', 'children' under spouse. It *didn't* use a separate facts API for family display. Let's just use person_card_data for family display.

        except Exception as e:
            logger.error(
                f"API {api_description_facts} call failed for {api_person_id}: {e}",
                exc_info=True,
            )
            print(f"\nWarning: Could not fetch additional profile details from API.")
            facts_data = None  # Ensure it's None on failure

    # --- Display Details using api_utils functions ---
    # Use the already parsed details from the person_card
    print("\n=== PERSON DETAILS (API) ===")
    print(f"Name: {parsed_api_details.get('name', 'Unknown')}")
    print(f"Gender: {parsed_api_details.get('gender') or 'N/A'}")
    print(
        f"Born: {parsed_api_details.get('birth_date', '(Date unknown)')} in {parsed_api_details.get('birth_place', '(Place unknown)')}"
    )
    # Only print Died line if there is death info or marked as not living
    if (
        parsed_api_details.get("death_date", "N/A") != "N/A"
        or parsed_api_details.get("death_place", "N/A") != "N/A"
        or parsed_api_details.get("is_living") is False
    ):
        print(
            f"Died: {parsed_api_details.get('death_date', '(Date unknown)')} in {parsed_api_details.get('death_place', '(Place unknown)')}"
        )
    print(f"Link: {parsed_api_details.get('link', '(unavailable)')}")

    # --- Display Family Details using api_utils functions ---
    print("\n--- Family Details (API) ---")
    # Extract family members from person_card_data as done in temp.py v7.36
    parents_list = []
    father = person_card_data.get("father")
    mother = person_card_data.get("mother")
    if father and isinstance(father, dict):
        parents_list.append(father)
    if mother and isinstance(mother, dict):
        parents_list.append(mother)

    print_group("Parents", parents_list)

    # Siblings are not directly available in this Person Card response structure in a simple list.
    # We could potentially fetch them via a different API or parse the Facts API more deeply.
    # For now, matching temp.py v7.36, display as None found.
    print_group(
        "Siblings", []
    )  # Placeholder / Not available directly in this API response

    spouses_list = []
    spouse = person_card_data.get(
        "selectedSpouse"
    )  # The currently selected/primary spouse
    # The Person Card API only gives details for ONE selected spouse and their children.
    # If multiple spouses exist, they might be in the Facts API response structure, but
    # the specific `/app-api/express/v1/profiles/details` endpoint structure needs checking.
    # Temp.py v7.36's Person Card parsing only checks for 'selectedSpouse'. Let's add a check for 'spouses' list in facts_data if available, but prioritize the selected one.
    if spouse and isinstance(spouse, dict):
        spouses_list.append(spouse)  # Add the selected spouse

    # Add other spouses from facts_data if available (checking for the list structure)
    if facts_data and isinstance(facts_data.get("spouses"), list):
        selected_spouse_id = (
            spouse.get("personId") if spouse and isinstance(spouse, dict) else None
        )
        for sp_fact in facts_data["spouses"]:
            if (
                isinstance(sp_fact, dict)
                and sp_fact.get("personId") != selected_spouse_id
            ):
                spouses_list.append(
                    {"name": sp_fact.get("name", "Unknown Spouse")}
                )  # Append other spouses

    print_group("Spouse(s)", spouses_list)

    children_list = []
    # Children are listed under the selected spouse in the Person Card API
    if spouse and isinstance(spouse, dict) and isinstance(spouse.get("children"), list):
        children_list.extend(spouse["children"])

    # Also check for a 'children' list in the facts_data if available
    if facts_data and isinstance(facts_data.get("children"), list):
        # Avoid adding children already listed under the selected spouse
        existing_child_ids = {
            c.get("personId") for c in children_list if isinstance(c, dict)
        }
        for ch_fact in facts_data["children"]:
            if (
                isinstance(ch_fact, dict)
                and ch_fact.get("personId") not in existing_child_ids
            ):
                children_list.append(
                    {"name": ch_fact.get("name", "Unknown Child")}
                )  # Append other children

    print_group("Children", children_list)

    # --- Display Relationship Path to Tree Owner (WGG) ---
    # This requires the selected person's ID and the logged-in user's profile ID (Tree Owner)
    # The API endpoint for this is /discoveryui-matchesservice/api/samples/{personId}/relationshiptome/{meProfileId}
    # It returns a JSONP response containing HTML.
    selected_person_id = selected_match.get(
        "person_id"
    )  # Use the ID from the selected match
    if owner_profile_id and selected_person_id:
        # Ensure the Tree Owner's profile ID is not the same as the selected person's ID
        if owner_profile_id.upper() == selected_person_id.upper():
            print(f"\n--- Relationship Path to {owner_name} (API) ---")
            print("(Selected person is the Tree Owner)")
        else:
            print(
                f"\nCalculating relationship path to {owner_name}..."
            )  # Use actual owner name
            ladder_api_url = f"{base_url}/discoveryui-matchesservice/api/samples/{selected_person_id}/relationshiptome/{owner_profile_id}"
            api_description_ladder = "API Relationship Ladder (Batch)"  # Using the batch API description from config

            relationship_html_raw = None  # Store the raw response text

            try:
                # Fetch the raw response text (it's JSONP/text, not pure JSON)
                relationship_html_raw = _api_req(
                    url=ladder_api_url,
                    driver=session_manager.driver,
                    session_manager=session_manager,
                    method="GET",
                    api_description=api_description_ladder,
                    # Referer needed for context, match list is a good referer
                    referer_url=urljoin(
                        base_url, f"/discoveryui-matches/list/{session_manager.my_uuid}"
                    ),
                    is_json=False,  # Explicitly tell _api_req NOT to parse as JSON initially
                    force_text_response=True,  # Ensure we get text response
                    timeout=15,  # Short timeout
                )

                # display_raw_relationship_ladder handles extraction, decoding, parsing, and printing
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
                    # If _api_req returned an error dict, pass it to the display function
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

            except Exception as e:
                logger.error(
                    f"API call {api_description_ladder} failed: {e}", exc_info=True
                )
                print(f"\n--- Relationship Path to {owner_name} (API) ---")
                print(f"(Error fetching relationship path from API: {e})")

    elif not owner_profile_id:
        print(f"\n--- Relationship Path to Tree Owner (API) ---")
        print("(Skipping relationship calculation as your profile ID was not found)")

    else:  # selected_person_id is None (shouldn't happen if match was selected)
        print(f"\n--- Relationship Path to {owner_name} (API) ---")
        print("(Skipping relationship calculation as selected person ID is missing)")

    return True  # Indicate successful report completion


# End of handle_api_report


# --- Main Execution ---
def main():
    """Main execution flow for Action 11 (API Report)."""
    logger.info("--- Action 11: API Report Starting ---")

    # Check if required core utilities are available
    if not CORE_UTILS_AVAILABLE:
        logger.critical(
            "Required core utilities (utils.py) not loaded. Cannot run Action 11."
        )
        print("\nCRITICAL ERROR: Required utilities not loaded.")
        sys.exit(1)

    # Check if required API utilities are available
    if not API_UTILS_AVAILABLE:
        logger.critical(
            "Required API utilities (api_utils.py) not loaded. Cannot run Action 11."
        )
        print("\nCRITICAL ERROR: Required API utilities not loaded.")
        sys.exit(1)

    # Check if required GEDCOM scoring/date utilities are available (needed for consistent scoring and display)
    # These are not strictly fatal for the *entire* API report, but necessary for key parts.
    # We log warnings in handle_api_report if they are missing, but exit here if scoring is a critical requirement.
    # As per requirement 3, scoring mechanism must be used.
    if not GEDCOM_SCORING_AVAILABLE or not GEDCOM_DATE_UTILS_AVAILABLE:
        logger.critical(
            "Required GEDCOM scoring or date utilities (from gedcom_utils.py) not loaded."
        )
        print(
            "\nCRITICAL ERROR: Required scoring or date utilities not loaded. Cannot ensure consistent report format."
        )
        sys.exit(1)

    # Check if scoring configuration is available
    if (
        COMMON_SCORING_WEIGHTS is None
        or NAME_FLEXIBILITY is None
        or DATE_FLEXIBILITY is None
    ):
        logger.critical(
            "Scoring configuration variables are None after attempting load."
        )
        print("\nERROR: Scoring configuration load failed. Cannot run Action 11.")
        sys.exit(1)

    # Run the API Report Handler
    handle_api_report()

    logger.info("--- Action 11: API Report Finished ---")
    print("\nAction 11 finished.")


# End of main

# Script entry point check
if __name__ == "__main__":
    # Initial check for core utils availability before calling main
    # main function will perform more detailed checks
    if CORE_UTILS_AVAILABLE:
        main()
    else:
        # If CORE_UTILS_AVAILABLE is False (due to utils import error), exit immediately
        print(
            "\nCRITICAL ERROR: Required core utilities (utils.py) are not installed or failed to load."
        )
        print("Please check your Python environment and dependencies.")
        logging.getLogger().critical("Exiting: Required core utilities not loaded.")
        sys.exit(1)




# End of action11.py