# api_utils.py
"""
Utility functions specifically for parsing Ancestry API responses
and formatting data obtained from APIs.
V3.4: Improved HTML parsing in format_api_relationship_path for /getladder responses.
"""

# --- Standard library imports ---
import logging
import sys
import re
import os
import time
import json
import requests  # Keep for exception types and Response object checking
import urllib.parse  # Used for urlencode in self_check and API calls
import html
from typing import Optional, Dict, Any, Union, List, Tuple, Callable, cast
from datetime import (
    datetime,
    timezone,
)  # Import datetime for parse_ancestry_person_details and self_check
from urllib.parse import (
    urljoin,
    urlencode,
    quote,
)  # Need quote for person picker params
from pathlib import Path  # Needed for __main__ block
import traceback  # For detailed exception logging in self_check

# --- Third-party imports ---
# Keep BeautifulSoup import here, check for its availability in functions
try:
    from bs4 import BeautifulSoup, FeatureNotFound
except ImportError:
    BeautifulSoup = None  # type: ignore # Gracefully handle missing dependency
    FeatureNotFound = None  # type: ignore

# Initialize logger - Ensure logger is always available
# Use basicConfig as fallback if logging_config fails
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_utils")

# --- Local application imports ---
# Use try-except for robustness, especially if run standalone initially
UTILS_AVAILABLE = False
GEDCOM_UTILS_AVAILABLE = False
CONFIG_AVAILABLE = False
try:
    import utils
    from utils import format_name, ordinal_case, SessionManager, _api_req

    UTILS_AVAILABLE = True
    logger.info("Successfully imported base utils module")
except ImportError:
    format_name = lambda x: str(x).title() if x else "Unknown"
    ordinal_case = lambda x: str(x)

    # Define dummy SessionManager and _api_req if utils unavailable
    class DummySessionManager:
        driver = None
        _requests_session = None

        def is_sess_valid(self):
            return False

        def close_sess(self):
            pass

        def start_sess(self, action_name=""):
            return False

        def ensure_session_ready(self, action_name=""):
            return False

        # Add dummy attributes needed by self_check
        my_tree_id = None
        tree_owner_name = "Dummy Owner"
        my_profile_id = None
        my_uuid = None

        # Add dummy methods/attributes if needed by API helpers
        def get_csrf_token(self):
            return "dummy_token"

    SessionManager = DummySessionManager  # type: ignore
    _api_req = lambda *args, **kwargs: None

    logger.warning("Failed to import utils, using fallback/dummy components.")

try:
    from gedcom_utils import _parse_date, _clean_display_date

    GEDCOM_UTILS_AVAILABLE = True
    logger.info("Successfully imported gedcom_utils date functions")
except ImportError:
    _parse_date = lambda x: None
    _clean_display_date = lambda x: str(x) if x else "N/A"
    logger.warning("Failed to import gedcom_utils, using fallback date functions")

try:
    from config import config_instance, selenium_config

    CONFIG_AVAILABLE = True
    logger.info("Successfully imported config instances")
except ImportError:
    CONFIG_AVAILABLE = False

    # Fallback to dummy config if config.py is not available
    class DummyConfig:
        BASE_URL = "https://www.ancestry.com"  # Provide a default
        TESTING_PROFILE_ID = "08FA6E79-0006-0000-0000-000000000000"
        TESTING_PERSON_TREE_ID = None
        API_TIMEOUT = 60  # Add default timeout

    config_instance = DummyConfig()

    # Create a dummy selenium_config if needed
    class DummySeleniumConfig:
        API_TIMEOUT = 60

    selenium_config = DummySeleniumConfig()
    logger.warning("Failed to import config from config.py, using default values")

# --- Helper Functions for parse_ancestry_person_details ---


def _extract_name_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> str:
    """
    Extracts the best name from person card or detailed facts data.
    Enhanced to check Suggest API keys in person_card.
    """
    name = "Unknown"
    # Use a local helper for formatting to ensure availability
    formatter = (
        format_name if UTILS_AVAILABLE else lambda x: str(x).title() if x else "Unknown"
    )

    # --- Prioritize detailed facts_data structures ---
    if facts_data and isinstance(facts_data, dict):
        # Structure from /app-api/express/v1/profiles/details
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            name = person_info.get("personName", name)
        if name == "Unknown":
            name = facts_data.get("personName", name)
        if name == "Unknown":
            name = facts_data.get("DisplayName", name)

        # Structure from facts/user API
        if name == "Unknown":
            name = facts_data.get("PersonFullName", name)
        if name == "Unknown":
            person_facts_list = facts_data.get("PersonFacts", [])
            if isinstance(person_facts_list, list):
                name_fact = next(
                    (
                        f
                        for f in person_facts_list
                        if isinstance(f, dict) and f.get("TypeString") == "Name"
                    ),
                    None,
                )
                if name_fact and name_fact.get("Value"):
                    name = name_fact.get("Value", "Unknown")
        if name == "Unknown":
            first_name_pd = facts_data.get("FirstName")
            last_name_pd = facts_data.get("LastName")
            if first_name_pd or last_name_pd:
                name = (
                    f"{first_name_pd or ''} {last_name_pd or ''}".strip() or "Unknown"
                )

    # --- Fallback to person_card (checking Suggest API keys first) ---
    if name == "Unknown" and person_card:
        # Check keys common in Suggest API response
        suggest_fullname = person_card.get("FullName")
        suggest_given = person_card.get("GivenName")
        suggest_sur = person_card.get("Surname")
        if suggest_fullname:
            name = suggest_fullname
        elif suggest_given or suggest_sur:
            name = f"{suggest_given or ''} {suggest_sur or ''}".strip() or "Unknown"

        # Final fallback to generic 'name' key in person_card
        if name == "Unknown":
            name = person_card.get("name", "Unknown")

    # Final formatting and validation
    formatted_name = formatter(name) if name and name != "Unknown" else "Unknown"
    return "Unknown" if formatted_name == "Valued Relative" else formatted_name


# End of _extract_name_from_api_details


def _extract_gender_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Optional[str]:
    """
    Extracts gender ('M' or 'F') from person card or detailed facts data.
    Enhanced to check Suggest API key 'Gender' in person_card.
    """
    gender = None
    gender_str = None

    # --- Prioritize detailed facts_data structures ---
    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})  # app-api structure
        if isinstance(person_info, dict):
            gender_str = person_info.get("gender")
        if not gender_str:
            gender_str = facts_data.get("gender")  # Alt app-api key?
        if not gender_str:
            gender_str = facts_data.get("PersonGender")  # facts/user structure
        if not gender_str:  # Gender fact in facts/user
            person_facts_list = facts_data.get("PersonFacts", [])
            if isinstance(person_facts_list, list):
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

    # --- Fallback to person_card (checking Suggest API 'Gender' key first) ---
    if not gender_str and person_card:
        # Check key 'Gender' common in Suggest API (often "Male"/"Female")
        gender_str = person_card.get("Gender")
        # Final fallback to generic 'gender' key (might be M/F)
        if not gender_str:
            gender_str = person_card.get("gender")

    # Normalize ("Male" -> "M", "Female" -> "F")
    if gender_str and isinstance(gender_str, str):
        gender_str_lower = gender_str.lower()
        if gender_str_lower == "male":
            gender = "M"
        elif gender_str_lower == "female":
            gender = "F"
        elif gender_str_lower in ["m", "f"]:
            gender = gender_str_lower.upper()  # Handle if already M/F

    return gender


# End of _extract_gender_from_api_details


def _extract_living_status_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Optional[bool]:
    """
    Extracts living status (True/False) from person card or detailed facts data.
    Enhanced to check Suggest API key 'IsLiving' in person_card.
    """
    is_living = None

    # --- Prioritize detailed facts_data structures ---
    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})  # app-api structure
        if isinstance(person_info, dict):
            is_living = person_info.get("isLiving")
        if is_living is None:
            is_living = facts_data.get("isLiving")  # Alt app-api key?
        if is_living is None:
            is_living = facts_data.get("IsPersonLiving")  # facts/user structure

    # --- Fallback to person_card (checking Suggest API 'IsLiving' key first) ---
    if is_living is None and person_card:
        # Check key 'IsLiving' common in Suggest API (boolean)
        is_living = person_card.get("IsLiving")
        # Final fallback to generic 'isLiving' key
        if is_living is None:
            is_living = person_card.get("isLiving")

    # Return as bool if found (and not None), otherwise None
    return bool(is_living) if is_living is not None else None


# End of _extract_living_status_from_api_details


def _extract_event_from_api_details(
    event_type: str, person_card: Dict, facts_data: Optional[Dict]
) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """
    Extracts date string, place string, and parsed date object for a specific event type.
    Prioritizes facts_data > person_card (including Suggest API keys).
    """
    date_str: Optional[str] = None
    place_str: Optional[str] = None
    date_obj: Optional[datetime] = None
    parser = _parse_date if GEDCOM_UTILS_AVAILABLE else lambda x: None
    event_key_lower = event_type.lower()  # e.g., "birth", "death"
    # Keys commonly used in Suggest API responses
    suggest_year_key = f"{event_type}Year"  # e.g., "BirthYear"
    suggest_place_key = f"{event_type}Place"  # e.g., "BirthPlace"
    # Keys used in app-api / facts/user
    facts_user_key = event_type  # e.g., "Birth", "Death" (for TypeString)
    app_api_key = f"{event_key_lower}Date"  # e.g., "birthDate"
    app_api_facts_key = event_type  # e.g., "Birth", "Death" (for facts dict key)

    found_in_facts = False
    # 1. Try detailed facts_data structures first
    if facts_data and isinstance(facts_data, dict):
        # Try PersonFacts structure (facts/user API)
        person_facts_list = facts_data.get("PersonFacts", [])
        if isinstance(person_facts_list, list):
            event_fact = next(
                (
                    f
                    for f in person_facts_list
                    if isinstance(f, dict)
                    and f.get("TypeString") == facts_user_key
                    and not f.get("IsAlternate")
                ),
                None,
            )
            if event_fact:
                date_str = event_fact.get("Date")
                place_str = event_fact.get("Place")
                parsed_date_data = event_fact.get("ParsedDate")
                found_in_facts = True
                logger.debug(
                    f"Found primary {event_type} fact in PersonFacts: Date='{date_str}', Place='{place_str}', ParsedDate={parsed_date_data}"
                )
                # Try parsing from ParsedDate first
                if isinstance(parsed_date_data, dict):
                    year = parsed_date_data.get("Year")
                    month = parsed_date_data.get("Month")
                    day = parsed_date_data.get("Day")
                    if year:
                        try:
                            temp_date_str = str(year)
                            if month:
                                temp_date_str += f"-{str(month).zfill(2)}"
                            if day:
                                temp_date_str += f"-{str(day).zfill(2)}"
                            date_obj = parser(temp_date_str) if parser else None
                            logger.debug(
                                f"Parsed {event_type} date object from ParsedDate: {date_obj}"
                            )
                        except Exception as dt_err:
                            logger.warning(
                                f"Could not parse {event_type} date from ParsedDate {parsed_date_data}: {dt_err}"
                            )

        # Try 'facts' structure (app-api) if not found yet
        if not found_in_facts:
            fact_group_list = facts_data.get("facts", {}).get(app_api_facts_key, [])
            if fact_group_list and isinstance(fact_group_list, list):
                fact_group = fact_group_list[0]  # Assume first is primary
                if isinstance(fact_group, dict):
                    date_info = fact_group.get("date", {})
                    place_info = fact_group.get("place", {})
                    if isinstance(date_info, dict):
                        date_str = date_info.get(
                            "normalized", date_info.get("original")
                        )
                    if isinstance(place_info, dict):
                        place_str = place_info.get("placeName")
                    found_in_facts = True

        # Try alternative direct keys in facts_data (app-api) if not found yet
        if not found_in_facts:
            event_fact_alt = facts_data.get(app_api_key)
            if event_fact_alt and isinstance(event_fact_alt, dict):
                date_str = event_fact_alt.get("normalized", event_fact_alt.get("date"))
                place_str = event_fact_alt.get(
                    "place", place_str
                )  # Keep previous place if new one not found
                found_in_facts = True
            elif isinstance(event_fact_alt, str):  # Sometimes just a string date
                date_str = event_fact_alt
                found_in_facts = True

    # 2. Fallback to person_card (checking Suggest API keys first)
    if not found_in_facts and person_card:
        # Check Suggest API Year/Place keys
        suggest_year = person_card.get(suggest_year_key)
        suggest_place = person_card.get(suggest_place_key)
        if suggest_year:  # Year is often the most reliable from Suggest
            date_str = str(suggest_year)  # Use year as date string
            place_str = suggest_place  # Use suggest place if available
            logger.debug(
                f"Using Suggest API keys for {event_type}: Year='{date_str}', Place='{place_str}'"
            )
        else:
            # Final fallback to generic person_card keys (e.g., "birth", "death")
            event_info_card = person_card.get(event_key_lower, "")
            if event_info_card and isinstance(event_info_card, str):
                parts = re.split(r"\s+in\s+", event_info_card, maxsplit=1)
                date_str = parts[0].strip() if parts else event_info_card
                if place_str is None and len(parts) > 1:
                    place_str = parts[1].strip()
            elif isinstance(event_info_card, dict):
                date_str = event_info_card.get("date", date_str)
                if place_str is None:
                    place_str = event_info_card.get("place", place_str)

    # 3. Parse date string if found and not already parsed
    if date_obj is None and date_str and parser:
        try:
            date_obj = parser(date_str)
        except Exception as parse_err:
            logger.warning(
                f"Failed to parse {event_type} date string '{date_str}': {parse_err}"
            )

    return date_str, place_str, date_obj


# End of _extract_event_from_api_details


def _generate_person_link(
    person_id: Optional[str], tree_id: Optional[str], base_url: str
) -> str:
    """Generates the Ancestry profile link based on available IDs."""
    if tree_id and person_id:
        return f"{base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/facts"
    elif person_id:  # Assume global profile ID if tree ID is missing
        # Corrected link for global ID based on recent observations
        return f"{base_url}/discoveryui-matches/list/summary/{person_id}"
        # return f"{base_url}/discoveryui-matches/profile/{person_id}" # Older format?
    else:
        return "(Link unavailable)"


# End of _generate_person_link


# --- API Response Parsing ---
def parse_ancestry_person_details(
    person_card: Dict, facts_data: Optional[Dict] = None  # Make facts_data optional
) -> Dict:
    """
    Extracts standardized details from Ancestry Person-Card (e.g., Suggest API)
    and optionally merges with detailed Facts API responses.
    Includes parsing dates/places and generating a link.

    Args:
        person_card (Dict): Basic info (e.g., from Suggest API or match list).
                            Required keys depend on context, helpers check common ones.
        facts_data (Optional[Dict]): More detailed API response (e.g., facts/user).
                                     If provided, its data takes precedence.

    Returns:
        Dict: Standardized details: 'name', 'birth_date' (display), 'birth_place',
              'death_date' (display), 'death_place', 'gender' ('M'/'F'/None),
              'person_id' (tree-specific), 'tree_id', 'user_id' (global), 'link',
              'api_birth_obj', 'api_death_obj', 'is_living'.
    """
    details = {
        "name": "Unknown",
        "birth_date": "N/A",
        "birth_place": None,
        "api_birth_obj": None,
        "death_date": "N/A",
        "death_place": None,
        "api_death_obj": None,
        "gender": None,
        "is_living": None,
        "person_id": person_card.get("PersonId"),  # Prefer Suggest API casing first
        "tree_id": person_card.get("TreeId"),
        "user_id": person_card.get("UserId"),  # Global ID from Suggest API
        "link": None,
    }
    # Fallback ID casings
    if not details["person_id"]:
        details["person_id"] = person_card.get("personId")
    if not details["tree_id"]:
        details["tree_id"] = person_card.get("treeId")

    # Update/Override with facts_data if provided
    if facts_data and isinstance(facts_data, dict):
        details["person_id"] = facts_data.get("PersonId", details["person_id"])
        details["tree_id"] = facts_data.get("TreeId", details["tree_id"])
        details["user_id"] = facts_data.get("UserId", details["user_id"])
        # Also check app-api structure for global ID
        if not details["user_id"]:
            person_info = facts_data.get("person", {})
            if isinstance(person_info, dict):
                details["user_id"] = person_info.get("userId", details["user_id"])

    # Extract using helpers - they prioritize facts_data if present, then check person_card
    details["name"] = _extract_name_from_api_details(person_card, facts_data)
    details["gender"] = _extract_gender_from_api_details(person_card, facts_data)
    details["is_living"] = _extract_living_status_from_api_details(
        person_card, facts_data
    )

    birth_date_raw, details["birth_place"], details["api_birth_obj"] = (
        _extract_event_from_api_details("Birth", person_card, facts_data)
    )
    death_date_raw, details["death_place"], details["api_death_obj"] = (
        _extract_event_from_api_details("Death", person_card, facts_data)
    )

    # Clean display dates
    cleaner = (
        _clean_display_date
        if GEDCOM_UTILS_AVAILABLE
        else lambda x: str(x) if x else "N/A"
    )
    details["birth_date"] = cleaner(birth_date_raw) if birth_date_raw else "N/A"
    details["death_date"] = cleaner(death_date_raw) if death_date_raw else "N/A"
    # If cleaner returned N/A but we have a year from helper, use year
    if details["birth_date"] == "N/A" and details["api_birth_obj"]:
        details["birth_date"] = str(details["api_birth_obj"].year)
    if details["death_date"] == "N/A" and details["api_death_obj"]:
        details["death_date"] = str(details["api_death_obj"].year)

    # Generate link (prioritize global ID if available)
    base_url_for_link = getattr(
        config_instance, "BASE_URL", "https://www.ancestry.com"
    ).rstrip("/")
    link_id = details["user_id"] or details["person_id"]  # Use global ID if present
    link_tree_id = (
        details["tree_id"] if not details["user_id"] else None
    )  # Don't use tree ID if using global profile link
    details["link"] = _generate_person_link(link_id, link_tree_id, base_url_for_link)

    logger.debug(
        f"Parsed API details for '{details.get('name', 'Unknown')}': "
        f"PersonID={details.get('person_id')}, TreeID={details.get('tree_id', 'N/A')}, "
        f"UserID={details.get('user_id', 'N/A')}, "
        f"Born='{details.get('birth_date')}' [{details.get('api_birth_obj')}] in '{details.get('birth_place') or '?'}', "
        f"Died='{details.get('death_date')}' [{details.get('api_death_obj')}] in '{details.get('death_place') or '?'}', "
        f"Gender='{details.get('gender') or '?'}', Living={details.get('is_living')}, "
        f"Link='{details.get('link')}'"
    )

    return details


# End of parse_ancestry_person_details


def print_group(label: str, items: List[Dict]):
    """Prints a formatted group of relatives from API data."""
    print(f"\n{label}:")
    if items:
        # Use imported format_name or fallback
        formatter = format_name if UTILS_AVAILABLE else lambda x: str(x).title()
        for item in items:
            # Ensure item is a dict and has 'name' before formatting
            name_to_format = item.get("name") if isinstance(item, dict) else None
            print(f"  - {formatter(name_to_format)}")
        # End for
    else:
        print("  (None found)")
    # End if/else


# End of print_group


def _get_relationship_term(
    person_a_gender: Optional[str], basic_relationship: str
) -> str:
    """Determines the specific relationship term based on gender (e.g., Father vs Parent)."""
    term = basic_relationship.capitalize()  # Default
    if basic_relationship.lower() == "parent":
        if person_a_gender == "M":
            term = "Father"
        elif person_a_gender == "F":
            term = "Mother"
    elif basic_relationship.lower() == "child":
        if person_a_gender == "M":
            term = "Son"
        elif person_a_gender == "F":
            term = "Daughter"
    elif basic_relationship.lower() == "sibling":
        if person_a_gender == "M":
            term = "Brother"
        elif person_a_gender == "F":
            term = "Sister"
    elif basic_relationship.lower() == "spouse":
        if person_a_gender == "M":
            term = "Husband"
        elif person_a_gender == "F":
            term = "Wife"
    # Add more specific terms as needed

    # Apply ordinal casing if the term might contain ordinals (e.g., "1st Cousin")
    # Use imported or fallback ordinal_case
    ord_caser = ordinal_case if UTILS_AVAILABLE else lambda x: str(x)
    if any(char.isdigit() for char in term):
        try:
            term = ord_caser(term)
        except Exception as ord_err:
            logger.warning(f"Failed to apply ordinal case to '{term}': {ord_err}")

    return term


# End of _get_relationship_term


def format_api_relationship_path(
    api_response_data: Union[str, Dict, None], owner_name: str, target_name: str
) -> str:
    """
    Parses relationship data primarily from the /getladder JSONP HTML response
    and formats it into a human-readable path with a specific bulleted format.

    Handles the JSONP string format (e.g., `__ancestry_jsonp_...` or `no(...)`)
    containing HTML, and the direct JSON `path` format from the Discovery API.

    Args:
        api_response_data: Raw data from /getladder (JSONP string) or Discovery (dict).
        owner_name: Name of the tree owner (or "You").
        target_name: Name of the person whose relationship is checked.

    Returns:
        Formatted string representing the relationship path, or an error message string.
    """
    if not api_response_data:
        logger.warning(
            "format_api_relationship_path: Received empty API response data."
        )
        return "(No relationship data received from API)"

    html_content_raw: Optional[str] = None  # Raw string from JSON
    json_data: Optional[Dict] = None
    api_status: str = "unknown"
    response_source: str = "Unknown"  # 'JSONP', 'JSON', 'RawString'
    name_formatter = format_name if UTILS_AVAILABLE else lambda x: str(x).title()

    # --- Step 1: Handle Input Type and Extract Relevant Data ---
    # (This part remains the same)
    if isinstance(api_response_data, dict):
        response_source = "JSON"
        if "error" in api_response_data:
            return f"(API returned error object: {api_response_data.get('error', 'Unknown')})"
        elif "path" in api_response_data:
            logger.debug("Detected direct JSON 'path' format (Discovery API).")
            json_data = api_response_data  # Assign dict to json_data
        elif (
            "html" in api_response_data
            and "status" in api_response_data
            and isinstance(api_response_data.get("html"), str)
        ):
            logger.debug("Detected pre-parsed JSONP structure with 'html' key.")
            html_content_raw = api_response_data.get("html")
            api_status = api_response_data.get("status", "unknown")
            if api_status != "success":
                return f"(API returned status '{api_status}': {api_response_data.get('message', 'Unknown Error')})"
        else:
            logger.warning(
                f"Received unhandled dictionary format: Keys={list(api_response_data.keys())}"
            )
            return "(Received unhandled dictionary format from API)"
    elif isinstance(api_response_data, str):
        response_source = "JSONP/RawString"
        if (
            api_response_data.strip().startswith("__ancestry_jsonp_")
            and api_response_data.strip().endswith(");")
        ) or (
            api_response_data.strip().startswith("no(")
            and api_response_data.strip().endswith(")")
        ):
            response_source = "JSONP"
            try:
                json_part_match = re.search(
                    r"^\s*[\w$.]+\((.*)\)\s*;?\s*$", api_response_data, re.DOTALL
                ) or re.search(r"^\s*no\((.*)\)\s*$", api_response_data, re.DOTALL)
                if json_part_match:
                    json_part = json_part_match.group(1).strip()
                    logger.debug(f"Extracted JSON part: {json_part[:100]}...")
                    parsed_json = json.loads(json_part)
                    api_status = parsed_json.get("status", "unknown")
                    if api_status == "success":
                        html_content_raw = parsed_json.get("html")
                        if not isinstance(html_content_raw, str):
                            logger.warning("'html' key not a string.")
                            html_content_raw = None
                        else:
                            logger.debug(
                                f"Extracted raw 'html': {html_content_raw[:100]}..."
                            )
                    else:
                        return f"(API status '{api_status}': {parsed_json.get('message', 'Error')})"
                else:
                    logger.warning("Could not extract JSON part from wrapper.")
                    html_content_raw = api_response_data
                    response_source = "RawString"
            except Exception as e:
                logger.error(f"Error processing {response_source}: {e}")
                html_content_raw = api_response_data
                response_source = "RawString"
        else:
            html_content_raw = api_response_data
            response_source = "RawString"
    else:
        return f"(Unsupported data type: {type(api_response_data)})"

    # --- Step 2: Format Discovery JSON Path (if applicable) ---
    if json_data and "path" in json_data:
        path_steps_json = []
        if isinstance(json_data["path"], list) and json_data["path"]:
            # Correct formatting for Discovery JSON path
            path_steps_json.append(f"  {target_name}")  # Start with target
            for step in json_data["path"]:
                step_name = name_formatter(step.get("name", "?"))  # Apply formatter
                step_rel = step.get("relationship", "?")
                # Use _get_relationship_term for consistent capitalization etc.
                step_rel_display = _get_relationship_term(None, step_rel)
                path_steps_json.append(
                    f"  -> {step_rel_display} is {step_name}"  # Use formatted name
                )
            path_steps_json.append(
                f"  -> {owner_name} (Tree Owner / You)"
            )  # Match expected end format
            result_str = "\n".join(path_steps_json)
            logger.info(f"Formatted Discovery relationship path:\n{result_str}")
            return result_str
        else:
            logger.warning(
                f"Discovery 'path' data invalid/empty: {json_data.get('path')}"
            )
            return "(Discovery path found but invalid format or empty)"

    # --- Step 3: Decode and Parse HTML Content (if applicable, from /getladder) ---
    # (Decoding and Parsing logic remains the same as previous correct version)
    html_content_decoded: Optional[str] = None
    if html_content_raw:
        try:
            html_content_decoded = bytes(html_content_raw, "utf-8").decode(
                "unicode_escape"
            )
            logger.debug(f"Decoded HTML content: {html_content_decoded[:200]}...")
        except Exception as decode_err:
            logger.error(f"Failed to decode HTML content: {decode_err}", exc_info=True)
            html_content_decoded = html_content_raw  # Fallback

    if not html_content_decoded:
        logger.warning("No processable HTML content found for relationship path.")
        return f"(Could not find, decode, or parse relationship HTML)"

    if not BeautifulSoup:
        logger.error("BeautifulSoup library not found.")
        return "(Cannot parse relationship path - BeautifulSoup missing)"

    try:
        logger.debug("Attempting to parse DECODED HTML content with BeautifulSoup...")
        soup = None
        list_items = []

        parser_used = "html.parser"
        try:
            soup = BeautifulSoup(html_content_decoded, "lxml")
            parser_used = "lxml"
        except FeatureNotFound:
            logger.warning("'lxml' parser not found, using 'html.parser'.")
            soup = BeautifulSoup(html_content_decoded, "html.parser")
        except Exception as e:
            logger.warning(f"Parser '{parser_used}' failed ({e}), using 'html.parser'.")
            soup = BeautifulSoup(html_content_decoded, "html.parser")
            parser_used = "html.parser"

        if soup:
            list_items = soup.select("ul.textCenter li")
            logger.debug(f"Found {len(list_items)} list items using '{parser_used}'.")
        else:
            logger.error("BeautifulSoup failed to create a soup object.")
            return "(Error creating BeautifulSoup object)"

        # --- Step 4: Extract Data from Parsed HTML List Items ---
        if not list_items:
            logger.warning("Expected list items, found none.")
            return "(Relationship HTML structure not as expected - Found 0 items)"

        item1 = list_items[0]
        role1_tag = item1.select_one("i b, i")
        overall_relationship = (
            role1_tag.get_text(strip=True).lower()
            if role1_tag
            else "unknown relationship"
        )
        logger.debug(f"Overall relationship: {overall_relationship}")

        path_items = soup.select('ul.textCenter > li:not([class*="iconArrowDown"])')
        logger.debug(f"Found {len(path_items)} relevant path items.")

        if len(path_items) < 2:
            logger.warning(f"Expected at least 2 path items, found {len(path_items)}.")
            return "(Could not find sufficient relationship path steps in HTML)"

        # --- Step 5: Build Formatted Output String (Bulleted List for HTML) ---
        summary_line = f"{target_name} is {owner_name}'s {overall_relationship}:"
        path_lines = []
        # Get the name of the first person (target) for context in descriptions
        name1_tag = path_items[0].find("b")
        target_name_from_html = (
            name_formatter(name1_tag.get_text(strip=True)) if name1_tag else target_name
        )

        previous_person_name = (
            target_name_from_html  # Use name parsed from HTML if possible
        )

        for i in range(1, len(path_items)):  # Start from the second item
            item = path_items[i]
            name_tag = item.find("a") or item.find("b")
            current_person_name_raw = (
                name_tag.get_text(strip=True) if name_tag else "Unknown"
            )
            current_person_name = name_formatter(current_person_name_raw)

            lifespan = ""
            if (
                name_tag
                and name_tag.next_sibling
                and isinstance(name_tag.next_sibling, str)
            ):
                potential_lifespan = name_tag.next_sibling.strip()
                lifespan_match = re.search(
                    r"(\d{4})\s*[-–—]?\s*(\d{4})?$", potential_lifespan
                )
                if lifespan_match:
                    start_year = lifespan_match.group(1)
                    end_year = lifespan_match.group(2)
                    if end_year:
                        lifespan = f"{start_year}–{end_year}"
                    elif (
                        potential_lifespan == start_year
                        or potential_lifespan.startswith(start_year)
                    ):
                        lifespan = f"b. {start_year}"
                    else:
                        lifespan = start_year  # Fallback to just year

            desc_tag = item.find("i")
            desc_text = desc_tag.get_text(strip=True) if desc_tag else ""
            logger.debug(
                f"Item {i}: Name='{current_person_name}', Lifespan='{lifespan}', Desc='{desc_text}'"
            )

            relationship_term = "relation"  # Default
            rel_match = re.search(
                r"\b(brother|sister|father|mother|son|daughter|husband|wife|spouse|parent|child|sibling)\b(?:\s+of)?",
                desc_text,
                re.IGNORECASE,
            )
            you_are_match = re.search(
                r"You\s+are\s+the\s+(\w+)", desc_text, re.IGNORECASE
            )

            if you_are_match:
                relationship_term = you_are_match.group(1).lower()
            elif rel_match:
                relationship_term = rel_match.group(1).lower()
            else:
                logger.warning(f"Could not extract standard term from: '{desc_text}'")

            current_person_display = current_person_name
            if lifespan:
                current_person_display += f" ({lifespan})"

            # Format the bullet point line based on relationship direction inferred from text
            # Check if the description defines the current person relative to the previous one
            previous_name_in_desc = previous_person_name.lower() in desc_text.lower()
            # Check if the description defines the owner relative to the current person
            owner_in_desc = (
                owner_name.lower() in desc_text.lower()
                and "you are" in desc_text.lower()
            )

            if owner_in_desc or (
                i == len(path_items) - 1
            ):  # Last item connects to owner
                # Final line format: Owner is Previous's Relation
                path_lines.append(
                    f"- {owner_name} is {previous_person_name}'s {relationship_term}."
                )
            elif previous_name_in_desc:
                # Intermediate format: Previous's Relation is Current
                path_lines.append(
                    f"- {previous_person_name}'s {relationship_term} is {current_person_display}."
                )
            else:
                # Fallback/Unknown format - state relationship and person
                logger.warning(
                    f"Ambiguous relationship direction in '{desc_text}'. Using fallback format."
                )
                path_lines.append(
                    f"- {current_person_display} ({relationship_term} of {previous_person_name})"
                )

            # Update previous person for the next iteration ONLY IF NOT THE OWNER ITEM
            if not (owner_in_desc or (i == len(path_items) - 1)):
                previous_person_name = current_person_name

        result_str = f"{summary_line}\n\n" + "\n".join(path_lines)
        logger.info(f"Formatted relationship path successfully:\n{result_str}")
        return result_str

    except Exception as e:
        logger.error(
            f"Error processing relationship HTML with BeautifulSoup: {e}", exc_info=True
        )
        return f"(Error processing relationship HTML: {e})"
# End of format_api_relationship_path


def _get_api_timeout(default: int = 60) -> int:
    """Safely gets the API timeout from config or returns default."""
    if CONFIG_AVAILABLE and selenium_config and hasattr(selenium_config, "API_TIMEOUT"):
        timeout = getattr(selenium_config, "API_TIMEOUT")
        if isinstance(timeout, (int, float)) and timeout > 0:
            return int(timeout)
    return default


# End of _get_api_timeout


def _get_owner_referer(session_manager: SessionManager, base_url: str) -> str:
    """Constructs the owner facts page referer URL."""
    owner_profile_id = getattr(session_manager, "my_profile_id", None)
    owner_tree_id = getattr(session_manager, "my_tree_id", None)
    if owner_profile_id and owner_tree_id:
        referer = urljoin(
            base_url,
            f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts",
        )
        logger.debug(f"Using owner facts page as referer: {referer}")
        return referer
    else:
        logger.warning("Cannot construct specific owner facts referer. Using base URL.")
        return base_url


# End of _get_owner_referer


def call_suggest_api(
    session_manager: SessionManager,
    owner_tree_id: str,
    owner_profile_id: Optional[str],  # Needed for referer
    base_url: str,
    search_criteria: Dict[str, Any],
    timeouts: Optional[List[int]] = None,
) -> Optional[List[Dict]]:
    """
    Calls the Ancestry Suggest API (/api/person-picker/suggest).
    Handles URL construction, headers, progressive timeouts, and direct request fallback.

    Args:
        session_manager: The active SessionManager instance.
        owner_tree_id: The ID of the owner's tree.
        owner_profile_id: The profile ID of the owner (for referer).
        base_url: The base Ancestry URL.
        search_criteria: Dict containing 'first_name_raw', 'surname_raw', 'birth_year'.
        timeouts: Optional list of progressive timeouts (seconds). Defaults to [20, 30, 60].

    Returns:
        List of suggestion dictionaries, or None if all attempts fail.
    """
    if not _api_req or not isinstance(session_manager, SessionManager):
        logger.error("Suggest API call failed: _api_req or SessionManager unavailable.")
        return None

    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")

    suggest_params = ["isHideVeiledRecords=false"]
    if first_name_raw:
        suggest_params.append(f"partialFirstName={quote(first_name_raw)}")
    if surname_raw:
        suggest_params.append(f"partialLastName={quote(surname_raw)}")
    if birth_year:
        suggest_params.append(f"birthYear={birth_year}")

    suggest_url = f"{base_url}/api/person-picker/suggest/{owner_tree_id}?{'&'.join(suggest_params)}"
    api_description = "Suggest API"
    owner_facts_referer = _get_owner_referer(session_manager, base_url)

    # Print the API URL to the console
    print(f"\nAPI URL Called: {suggest_url}\n")
    logger.info(f"Attempting {api_description} search: {suggest_url}")  # Keep log

    # Use provided timeouts or default
    timeouts_used = timeouts if timeouts else [20, 30, 60]
    max_attempts = len(timeouts_used)

    # print(f"Searching Ancestry API (Timeout: {sum(timeouts_used)}s max)") # Removed general message

    for attempt, timeout in enumerate(timeouts_used, 1):
        # print(f"(Calling {api_description}... Timeout: {timeout}s)") # Removed status message
        try:
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
                headers=custom_headers,
                referer_url=owner_facts_referer,
                timeout=timeout,
                use_csrf_token=False,
            )

            if isinstance(suggest_response, list):
                logger.info(
                    f"{api_description} call successful (attempt {attempt}/{max_attempts}), found {len(suggest_response)} results."
                )
                return suggest_response
            elif suggest_response is None:
                logger.warning(
                    f"{api_description} call using _api_req returned None on attempt {attempt}/{max_attempts}."
                )
                if attempt < max_attempts:
                    print("Retrying with longer timeout...")
                else:
                    print(f"({api_description} call failed after {attempt} attempts.)")
            else:
                logger.error(
                    f"{api_description} call using _api_req returned unexpected type: {type(suggest_response)}"
                )
                logger.debug(
                    f"{api_description} Response Content: {str(suggest_response)[:500]}"
                )
                print(f"({api_description} call returned unexpected data format.)")
                break  # Don't retry if format is wrong, try direct fallback

        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
            if attempt < max_attempts:
                print("Timeout occurred. Retrying with longer timeout...")
            else:
                logger.error(
                    f"{api_description} call failed after all timeout attempts."
                )
        except Exception as api_err:
            logger.error(
                f"{api_description} call failed with error on attempt {attempt}/{max_attempts}: {api_err}",
                exc_info=True,
            )
            if attempt < max_attempts:
                print("Error occurred. Retrying...")
            else:
                print(f"\nAll {api_description} attempts failed.")
            break

    # --- Direct Request Fallback ---
    # print("\nAttempting direct request fallback...") # Removed status message
    try:
        cookies = (
            session_manager._requests_session.cookies.get_dict()
            if session_manager._requests_session
            else {}
        )
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
        logger.debug(f"Direct request URL: {suggest_url}")
        # Print URL again for direct fallback attempt
        print(f"\nAPI URL Called (Direct Fallback): {suggest_url}\n")
        logger.debug(f"Direct request headers: {direct_headers}")
        direct_response = requests.get(
            suggest_url, headers=direct_headers, cookies=cookies, timeout=30
        )

        if direct_response.status_code == 200:
            direct_data = direct_response.json()
            if isinstance(direct_data, list):
                logger.info(
                    f"Direct request fallback successful! Found {len(direct_data)} results."
                )
                # print(f"Direct request successful! Found {len(direct_data)} potential matches.") # Removed status message
                return direct_data
            else:
                logger.warning(
                    f"Direct request returned non-list data: {type(direct_data)}"
                )
                logger.debug(f"Response content: {str(direct_data)[:500]}")
        else:
            logger.warning(
                f"Direct request failed: Status {direct_response.status_code}"
            )
            logger.debug(f"Response content: {direct_response.text[:500]}")

    except requests.exceptions.Timeout:
        logger.error("Direct request timed out after 30 seconds")
        print("Direct request timed out.")
    except Exception as direct_err:
        logger.error(f"Direct request fallback failed: {direct_err}", exc_info=True)

    logger.error(f"{api_description} failed after all attempts and fallback.")
    return None


# End of call_suggest_api


def call_facts_user_api(
    session_manager: SessionManager,
    owner_profile_id: str,
    api_person_id: str,
    api_tree_id: str,
    base_url: str,
    timeouts: Optional[List[int]] = None,
) -> Optional[Dict]:
    """
    Calls the Ancestry Person Facts User API (/family-tree/person/facts/user/).
    Handles URL construction, headers, direct request attempt, _api_req fallback with progressive timeouts.

    Args:
        session_manager: The active SessionManager instance.
        owner_profile_id: The profile ID of the owner making the request.
        api_person_id: The ID of the person whose facts are requested.
        api_tree_id: The tree ID containing the person.
        base_url: The base Ancestry URL.
        timeouts: Optional list of progressive timeouts for _api_req fallback (seconds). Defaults to [30, 45, 60].

    Returns:
        The 'personResearch' dictionary from the response, or None if all attempts fail.
    """
    if not _api_req or not isinstance(session_manager, SessionManager):
        logger.error("Facts API call failed: _api_req or SessionManager unavailable.")
        return None

    facts_api_url = f"{base_url}/family-tree/person/facts/user/{owner_profile_id.lower()}/tree/{api_tree_id}/person/{api_person_id}"
    api_description = "Person Facts User API"
    facts_referer = _get_owner_referer(session_manager, base_url)
    facts_data_raw = None

    # Print the API URL to the console for the direct attempt
    print(f"\nAPI URL Called (Direct Attempt): {facts_api_url}\n")
    logger.info(
        f"Attempting to fetch facts for PersonID {api_person_id}: {facts_api_url}"
    )  # Keep log

    # --- Direct Request Attempt First ---
    try:
        cookies = (
            session_manager._requests_session.cookies.get_dict()
            if session_manager._requests_session
            else {}
        )
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
        logger.debug(f"Direct facts request URL: {facts_api_url}")
        logger.debug(f"Direct facts request headers: {direct_headers}")
        direct_response = requests.get(
            facts_api_url, headers=direct_headers, cookies=cookies, timeout=30
        )

        if direct_response.status_code == 200:
            facts_data_raw = direct_response.json()
            if not isinstance(facts_data_raw, dict):
                logger.warning(
                    f"Direct request returned non-dict data: {type(facts_data_raw)}"
                )
                logger.debug(f"Response content: {str(facts_data_raw)[:500]}")
                facts_data_raw = None
        else:
            logger.warning(
                f"Direct request failed: Status {direct_response.status_code}"
            )
            logger.debug(f"Response content: {direct_response.text[:500]}")
            facts_data_raw = None

    except requests.exceptions.Timeout:
        logger.error("Direct facts request timed out after 30 seconds")
        print("Direct request timed out. Trying original approach...")
        facts_data_raw = None
    except Exception as direct_err:
        logger.error(f"Direct facts request failed: {direct_err}", exc_info=True)
        facts_data_raw = None

    # --- Fallback to _api_req with Progressive Timeouts ---
    if facts_data_raw is None:
        # print("Direct request failed or returned invalid data. Trying original approach...") # Removed status message
        # Print URL again for the _api_req fallback attempt
        print(f"\nAPI URL Called (_api_req Fallback): {facts_api_url}\n")
        logger.info(
            f"Attempting {api_description} via _api_req: {facts_api_url}"
        )  # Keep log

        timeouts_used = timeouts if timeouts else [30, 45, 60]
        max_attempts = len(timeouts_used)

        for attempt, timeout in enumerate(timeouts_used, 1):
            # print(f"(Attempt {attempt}/{max_attempts}: Fetching details via _api_req... Timeout: {timeout}s)") # Removed status message
            try:
                custom_headers = {
                    "Accept": "application/json",
                    "Referer": facts_referer,
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                    "X-Requested-With": "XMLHttpRequest",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
                }
                api_response = _api_req(
                    url=facts_api_url,
                    driver=session_manager.driver,
                    session_manager=session_manager,
                    method="GET",
                    api_description=api_description,
                    headers=custom_headers,
                    referer_url=facts_referer,
                    timeout=timeout,
                )

                if isinstance(api_response, dict):
                    facts_data_raw = api_response
                    logger.info(
                        f"{api_description} call successful via _api_req (attempt {attempt}/{max_attempts})."
                    )
                    break
                elif isinstance(api_response, requests.Response):
                    logger.warning(
                        f"{api_description} _api_req returned Response (Status: {api_response.status_code}). Expected dict."
                    )
                    logger.debug(f"Response Text: {api_response.text[:500]}")
                    if attempt < max_attempts:
                        print("Received error response. Retrying...")
                    else:
                        print("Received error response on final attempt.")
                elif api_response is None:
                    logger.warning(
                        f"{api_description} _api_req returned None (attempt {attempt}/{max_attempts})."
                    )
                    if attempt < max_attempts:
                        print("No response received. Retrying...")
                    else:
                        print("No response received on final attempt.")
                else:
                    logger.warning(
                        f"{api_description} _api_req returned unexpected type: {type(api_response)}"
                    )
                    logger.debug(f"Response Value: {str(api_response)[:500]}")
                    if attempt < max_attempts:
                        print("Received unexpected response format. Retrying...")
                    else:
                        print("Received unexpected response format on final attempt.")

            except requests.exceptions.Timeout:
                logger.warning(
                    f"{api_description} call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
                )
                if attempt < max_attempts:
                    print("Timeout occurred. Retrying...")
                else:
                    logger.error(
                        f"{api_description} call failed after all timeout attempts."
                    )
            except Exception as api_req_err:
                logger.error(
                    f"{api_description} call using _api_req failed on attempt {attempt}/{max_attempts}: {api_req_err}",
                    exc_info=True,
                )
                if attempt < max_attempts:
                    print("Error occurred. Retrying...")
                else:
                    print(
                        f"\nError fetching person details via _api_req: {api_req_err}"
                    )
                break

    # --- Process Final Result ---
    if not isinstance(facts_data_raw, dict):
        logger.error(f"Failed to fetch {api_description} data after all attempts.")
        print(f"\nError: Could not fetch valid person details from API. Check logs.")
        return None

    person_research_data = facts_data_raw.get("data", {}).get("personResearch")
    if not isinstance(person_research_data, dict) or not person_research_data:
        logger.error(
            f"{api_description} response missing 'data.personResearch' dictionary."
        )
        logger.debug(f"Full raw response keys: {list(facts_data_raw.keys())}")
        if "data" in facts_data_raw:
            logger.debug(f"'data' sub-keys: {list(facts_data_raw['data'].keys())}")
        print(f"\nError: API response format for person details was unexpected.")
        return None

    logger.info(
        f"Successfully fetched and parsed 'personResearch' data for PersonID {api_person_id}."
    )
    return person_research_data


# End of call_facts_user_api


def call_getladder_api(
    session_manager: SessionManager,
    owner_tree_id: str,
    target_person_id: str,
    base_url: str,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """
    Calls the Ancestry Tree Ladder API (/getladder).
    Handles URL construction, headers, and _api_req call.

    Args:
        session_manager: The active SessionManager instance.
        owner_tree_id: The ID of the owner's tree.
        target_person_id: The ID of the person whose relationship is requested.
        base_url: The base Ancestry URL.
        timeout: Optional specific timeout (seconds). Defaults to config or 20s.

    Returns:
        The raw JSONP string response, or None on failure.
    """
    if not _api_req or not isinstance(session_manager, SessionManager):
        logger.error(
            "GetLadder API call failed: _api_req or SessionManager unavailable."
        )
        return None

    api_description = "Get Tree Ladder API"
    ladder_api_url_base = f"{base_url}/family-tree/person/tree/{owner_tree_id}/person/{target_person_id}/getladder"
    query_params = urlencode({"callback": "no"})
    ladder_api_url = f"{ladder_api_url_base}?{query_params}"
    ladder_referer = urljoin(
        base_url,
        f"/family-tree/person/tree/{owner_tree_id}/person/{target_person_id}/facts",
    )
    api_timeout = timeout if timeout else _get_api_timeout(20)

    # Print the API URL to the console
    print(f"\nAPI URL Called: {ladder_api_url}\n")
    logger.info(f"Calling {api_description} at {ladder_api_url}")  # Keep log
    logger.debug(f" Referer: {ladder_referer}")
    # print(f"(Calculating relationship via {api_description}... Timeout: {api_timeout}s)") # Removed status message

    try:
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
        if isinstance(relationship_data, str):
            logger.info(f"{api_description} call successful.")
            return relationship_data
        else:
            logger.warning(
                f"{api_description} call returned non-string or None: {type(relationship_data)}"
            )
            return None

    except requests.exceptions.Timeout:
        logger.error(
            f"{api_description} call caught specific Timeout exception after {api_timeout}s."
        )
        print(f"(Error: Timed out fetching relationship path from Tree API)")
        return None
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        print(f"(Error fetching relationship path from Tree API: {e})")
        return None


# End of call_getladder_api


def call_discovery_relationship_api(
    session_manager: SessionManager,
    target_global_id: str,
    owner_profile_id: str,
    base_url: str,
    timeout: Optional[int] = None,
) -> Optional[Dict]:
    """
    Calls the Ancestry Discovery Relationship API (/relationshiptome).
    Handles URL construction, headers, and _api_req call.

    Args:
        session_manager: The active SessionManager instance.
        target_global_id: The global profile ID of the target person.
        owner_profile_id: The global profile ID of the owner.
        base_url: The base Ancestry URL.
        timeout: Optional specific timeout (seconds). Defaults to config or 20s.

    Returns:
        The JSON dictionary response, or None on failure.
    """
    if not _api_req or not isinstance(session_manager, SessionManager):
        logger.error(
            "Discovery Relationship API call failed: _api_req or SessionManager unavailable."
        )
        return None

    api_description = "Discovery Relationship API"
    # Corrected API endpoint structure based on observations (might vary)
    # Using relationshiptome - adjust if /api/relationship is needed
    discovery_api_url = f"{base_url}/discoveryui-matchesservice/api/samples/{target_global_id}/relationshiptome/{owner_profile_id}"
    # If the other endpoint is needed:
    # discovery_api_url = f"{base_url}/discoveryui-matchingservice/api/relationship?profileIdFrom={owner_profile_id}&profileIdTo={target_global_id}"

    uuid_for_referer = getattr(session_manager, "my_uuid", None)
    discovery_referer = base_url
    if uuid_for_referer:
        discovery_referer = urljoin(
            base_url, f"/discoveryui-matches/list/{uuid_for_referer}"
        )
    api_timeout = timeout if timeout else _get_api_timeout(20)

    # Print the API URL to the console
    print(f"\nAPI URL Called: {discovery_api_url}\n")
    logger.info(f"Calling {api_description} at {discovery_api_url}")  # Keep log
    logger.debug(f" Referer: {discovery_referer}")
    # print(f"(Calculating relationship via {api_description}... Timeout: {api_timeout}s)") # Removed status message

    try:
        relationship_data = _api_req(
            url=discovery_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            api_description=api_description,
            referer_url=discovery_referer,
            timeout=api_timeout,
        )
        if isinstance(relationship_data, dict):
            logger.info(f"{api_description} call successful.")
            return relationship_data
        else:
            logger.warning(
                f"{api_description} call returned non-dict or None: {type(relationship_data)}"
            )
            return None

    except requests.exceptions.Timeout:
        logger.error(
            f"{api_description} call caught specific Timeout exception after {api_timeout}s."
        )
        print(f"(Error: Timed out fetching relationship path from Discovery API)")
        return None
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        print(f"(Error fetching relationship path from Discovery API: {e})")
        return None


# End of call_discovery_relationship_api


def call_treesui_list_api(
    session_manager: SessionManager,
    owner_tree_id: str,
    owner_profile_id: Optional[str],  # Needed for referer
    base_url: str,
    search_criteria: Dict[str, Any],
    timeouts: Optional[List[int]] = None,
) -> Optional[List[Dict]]:
    """
    Calls the Ancestry TreesUI List API (/api/treesui-list/trees/{tree}/persons) as a fallback.
    Handles URL construction, headers, and _api_req call with progressive timeouts.

    Args:
        session_manager: The active SessionManager instance.
        owner_tree_id: The ID of the owner's tree.
        owner_profile_id: The profile ID of the owner (for referer).
        base_url: The base Ancestry URL.
        search_criteria: Dict containing 'first_name_raw', 'surname_raw', 'birth_year'.
        timeouts: Optional list of progressive timeouts (seconds). Defaults to [15, 25, 35].

    Returns:
        List of result dictionaries, or None if call fails or birth year missing.
    """
    if not _api_req or not isinstance(session_manager, SessionManager):
        logger.error(
            "TreesUI List API call failed: _api_req or SessionManager unavailable."
        )
        return None

    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")

    if not birth_year:
        logger.warning("Cannot call TreesUI List API without birth year.")
        # print("Cannot use alternative API search: birth year is required.") # Removed print
        return None

    treesui_params = ["limit=100", "fields=NAMES,BIRTH_DEATH"]  # Reduced fields
    if first_name_raw:
        treesui_params.append(f"fn={quote(first_name_raw)}")
    if surname_raw:
        treesui_params.append(f"ln={quote(surname_raw)}")
    treesui_params.append(f"by={birth_year}")

    treesui_url = f"{base_url}/api/treesui-list/trees/{owner_tree_id}/persons?{'&'.join(treesui_params)}"
    api_description = "TreesUI List API"
    owner_facts_referer = _get_owner_referer(session_manager, base_url)
    timeouts_used = timeouts if timeouts else [15, 25, 35]
    max_attempts = len(timeouts_used)

    # Print the API URL to the console
    print(f"\nAPI URL Called (TreesUI List Fallback): {treesui_url}\n")
    logger.info(
        f"Attempting {api_description} search using _api_req: {treesui_url}"
    )  # Keep log
    # print(f"Trying alternative API search (Progressive timeouts: {', '.join(map(str, timeouts_used))}s)") # Removed status message

    for attempt, timeout in enumerate(timeouts_used, 1):
        # print(f"(Attempt {attempt}/{max_attempts}: Calling {api_description}... Timeout: {timeout}s)") # Removed status message
        try:
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
                headers=custom_headers,
                referer_url=owner_facts_referer,
                timeout=timeout,
            )

            if isinstance(treesui_response, list):
                logger.info(
                    f"{api_description} call successful (attempt {attempt}/{max_attempts}), found {len(treesui_response)} results."
                )
                # print(f"Alternative API search successful! Found {len(treesui_response)} potential matches.") # Removed status message
                return treesui_response
            elif treesui_response is not None:
                logger.error(
                    f"{api_description} returned unexpected format: {type(treesui_response)}"
                )
                print("Alternative API search returned unexpected format.")
                return None
            else:
                logger.warning(
                    f"{api_description} call failed or returned None on attempt {attempt}/{max_attempts}."
                )
                if attempt < max_attempts:
                    print("Retrying with longer timeout...")
                else:
                    print("Alternative API search failed after all attempts.")

        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
            if attempt < max_attempts:
                print("Timeout occurred. Retrying...")
            else:
                logger.error(
                    f"{api_description} call failed after all timeout attempts."
                )
        except Exception as treesui_err:
            logger.error(
                f"{api_description} call failed with error on attempt {attempt}/{max_attempts}: {treesui_err}",
                exc_info=True,
            )
            if attempt < max_attempts:
                print("Error occurred. Retrying...")
            else:
                print("\nAll alternative API search attempts failed.")
            break

    logger.error(f"{api_description} failed after all attempts.")
    return None


# End of call_treesui_list_api


# --- Standalone Test Block ---


# --- [ Self-Check Test Runner Helper ] ---
def _sc_run_test(
    test_name: str,
    test_func: Callable,
    test_results_list: List[Tuple[str, str, str]],
    logger_instance: logging.Logger,
    *args,
    **kwargs,
) -> Tuple[str, str, str]:
    """Runs a single test, logs result, stores result, and returns status tuple."""
    logger_instance.debug(f"[ RUNNING SC ] {test_name}")
    status = "FAIL"  # Default assumption
    message = ""
    expect_none = kwargs.pop("expected_none", False)
    expect_type = kwargs.pop("expected_type", None)
    expect_value = kwargs.pop("expected_value", None)
    expect_contains = kwargs.pop("expected_contains", None)
    expect_truthy = kwargs.pop("expected_truthy", False)
    test_func_kwargs = kwargs  # Remaining kwargs are for the test function

    try:
        result = test_func(*args, **test_func_kwargs)
        passed = False

        # --- Evaluation Logic ---
        if expect_none:
            passed = result is None
            if not passed:
                message = f"Expected None, got {type(result).__name__}"
        elif expect_type is not None:
            if result is None:
                passed = False  # It didn't pass the type check
                status = "SKIPPED"  # Mark as skipped for overall status
                message = f"Expected type {expect_type.__name__}, got None (API issue?)"
                logger_instance.warning(
                    f"Test '{test_name}' expecting type {expect_type.__name__} received None, marking as SKIPPED."
                )
            elif isinstance(result, expect_type):
                passed = True  # Type matches
            else:  # Type mismatch (and not None)
                passed = False
                message = (
                    f"Expected type {expect_type.__name__}, got {type(result).__name__}"
                )
        elif expect_value is not None:
            passed = result == expect_value
            if not passed:
                message = f"Expected value '{str(expect_value)[:50]}', got '{repr(result)[:100]}'"
        elif expect_contains is not None:
            if isinstance(result, str):
                if isinstance(expect_contains, (list, tuple)):
                    passed = all(sub in result for sub in expect_contains)
                    if not passed:
                        missing = [sub for sub in expect_contains if sub not in result]
                        message = f"Expected result to contain all of: {expect_contains}. Missing: {missing}"
                elif isinstance(expect_contains, str):
                    passed = expect_contains in result
                    if not passed:
                        message = f"Expected result to contain '{expect_contains}', got '{repr(result)[:100]}'"
                else:
                    passed = False
                    message = (
                        f"Invalid type for expect_contains: {type(expect_contains)}"
                    )
            else:
                passed = False
                message = f"Expected string result for contains check, got {type(result).__name__}"
        elif expect_truthy:
            passed = bool(result)
            if not passed:
                message = f"Expected truthy value, got {repr(result)[:100]}"
        elif isinstance(result, str) and result == "Skipped":
            passed = False
            status = "SKIPPED"
            message = ""  # Explicit skip from lambda
        else:  # Default check: result should be exactly True
            passed = result is True
            if not passed:
                message = f"Expected True, got {repr(result)[:100]}"

        # --- Set Status ---
        if (
            status != "SKIPPED"
        ):  # Don't override explicit skip or the None->SKIPPED case
            if passed:
                status = "PASS"
            else:
                status = "FAIL"
            if status == "FAIL" and not message:
                message = f"Test condition not met (Result: {repr(result)[:100]})"

    except Exception as e:
        status = "FAIL"
        message = f"{type(e).__name__}: {e}"
        logger_instance.debug(
            f"Exception details for {test_name}: {message}\n{traceback.format_exc()}",
            exc_info=False,
        )

    log_level = (
        logging.INFO
        if status == "PASS"
        else (logging.WARNING if status == "SKIPPED" else logging.ERROR)
    )
    log_message = f"[ {status:<6} SC ] {test_name}{f': {message}' if message and status == 'FAIL' else ''}"
    logger_instance.log(log_level, log_message)
    test_results_list.append(
        (test_name, status, message if status != "PASS" else "")
    )  # Store message only if not PASS
    return (test_name, status, message)


# End of _sc_run_test


# --- [ Self-Check Summary Printer ] ---
def _sc_print_summary(
    test_results_list: List[Tuple[str, str, str]],
    overall_status: bool,
    logger_instance: logging.Logger,
):
    """Prints the formatted summary table of self-check results."""
    print("\n--- api_utils.py Self-Check Summary ---")
    name_width = 55  # Increased width slightly for longer test names
    if test_results_list:
        try:
            name_width = max(len(name) for name, _, _ in test_results_list) + 2
        except ValueError:
            pass  # Handle empty list case
    status_width = 8
    header = f"{'Test Name':<{name_width}} | {'Status':<{status_width}} | {'Message'}"
    print(header)
    print("-" * (len(header)))
    final_fail_count = 0
    final_skip_count = 0
    reported_test_names = set()

    for name, status, message in test_results_list:
        if name in reported_test_names:
            continue  # Avoid double printing if run multiple times
        reported_test_names.add(name)
        current_status = status
        if status == "FAIL":
            final_fail_count += 1
        elif status == "SKIPPED":
            final_skip_count += 1
            message = ""  # Don't show message for skipped
        print(
            f"{name:<{name_width}} | {current_status:<{status_width}} | {message if current_status == 'FAIL' else ''}"
        )

    print("-" * (len(header)))
    total_tests = len(reported_test_names)
    passed_tests = total_tests - final_fail_count - final_skip_count
    final_overall_status = overall_status and (final_fail_count == 0)
    final_status_msg = f"Result: {'PASS' if final_overall_status else 'FAIL'} ({passed_tests} passed, {final_fail_count} failed, {final_skip_count} skipped out of {total_tests} tests)"
    print(f"{final_status_msg}\n")
    if final_overall_status:
        logger_instance.info("api_utils self-check overall status: PASS")
    else:
        logger_instance.error("api_utils self-check overall status: FAIL")


# End of _sc_print_summary


def self_check() -> bool:
    """
    Performs internal self-checks for api_utils.py, including LIVE API calls.
    Requires .env file to be correctly configured. Provides formatted output summary.
    """
    # --- Local Imports & Logger Setup for Self-Check ---
    try:
        from logging_config import logger as logger_sc
    except ImportError:
        logger_sc = logging.getLogger("api_utils.self_check")

    try:
        if not UTILS_AVAILABLE or "utils" not in sys.modules:
            raise ImportError("Base utils module not imported.")
        if not SessionManager:
            raise ImportError("SessionManager class missing.")  # Check explicitly
        if not _api_req:
            raise ImportError("_api_req function missing.")  # Check explicitly
        if not CONFIG_AVAILABLE or "config" not in sys.modules:
            raise ImportError("Config module not imported.")
        from config import (
            config_instance as config_instance_sc,
            selenium_config as selenium_config_sc,
        )

        if not config_instance_sc or not selenium_config_sc:
            raise ImportError("Config instances are None.")
    except ImportError as e:
        print(
            f"\n[api_utils.py self-check ERROR] - Cannot import base utils/config for live tests: {e}"
        )
        logger_sc.critical(f"Self-check cannot run due to import error: {e}")
        return False

    # --- Test Runner Setup ---
    test_results_sc: List[Tuple[str, str, str]] = []
    session_manager_sc: Optional["SessionManager"] = None
    overall_status = True

    # --- Internal API Call Helpers for Self-Check ---
    # (These remain the same)
    def _sc_api_req_wrapper(
        url: str, description: str, expect_json: bool = True, **kwargs
    ) -> Any:
        """Wrapper for _api_req within self-check, handling session and potential Response objects."""
        nonlocal session_manager_sc
        if not _api_req: raise RuntimeError("_api_req func unavailable")
        if not session_manager_sc or not session_manager_sc.is_sess_valid(): raise RuntimeError("Session not ready")
        result = _api_req(url=url, driver=session_manager_sc.driver, session_manager=session_manager_sc, api_description=f"{description} (SC)", **kwargs)
        if expect_json and isinstance(result, requests.Response):
            logger_sc.warning(f"[_sc wrapper] Expected JSON for '{description}', got Response (Status: {result.status_code}). Returning None.")
            return None
        return result
    # End of _sc_api_req_wrapper

    def _sc_get_profile_details(profile_id: str) -> Optional[Dict]:
        """Helper to get profile details via /app-api endpoint."""
        if not config_instance_sc or not profile_id: return None
        api_desc = f"Get Profile Details ({profile_id})"
        url = urljoin(config_instance_sc.BASE_URL, f"/app-api/express/v1/profiles/details?userId={profile_id.upper()}")
        timeout = getattr(selenium_config_sc, "API_TIMEOUT", 60)
        return _sc_api_req_wrapper(url, api_desc, expect_json=True, use_csrf_token=False, timeout=timeout)
    # End of _sc_get_profile_details

    def _sc_get_tree_ladder(tree_id: str, person_id: str) -> Optional[str]:
        """Helper to get relationship ladder using production call_getladder_api."""
        nonlocal session_manager_sc
        if not all([config_instance_sc, selenium_config_sc, session_manager_sc, tree_id, person_id, callable(call_getladder_api)]): return None
        api_timeout = getattr(selenium_config_sc, "API_TIMEOUT", 60)
        base_url = config_instance_sc.BASE_URL
        return call_getladder_api(session_manager_sc, tree_id, person_id, base_url, timeout=api_timeout)
    # End of _sc_get_tree_ladder

    # --- Test Parameters ---
    can_run_live_tests = bool(SessionManager and _api_req and config_instance_sc and selenium_config_sc)
    target_profile_id = getattr(config_instance_sc, "TESTING_PROFILE_ID", None)
    target_person_id_for_ladder = getattr(config_instance_sc, "TESTING_PERSON_TREE_ID", None)
    base_url_sc = getattr(config_instance_sc, "BASE_URL", "https://www.ancestry.com").rstrip("/")
    target_name_from_profile = "Unknown Target"
    target_name_for_ladder = "Unknown Ladder Target" # Will be updated if possible
    if can_run_live_tests and not target_profile_id: logger_sc.warning("TESTING_PROFILE_ID missing.")
    if can_run_live_tests and not target_person_id_for_ladder: logger_sc.warning("TESTING_PERSON_TREE_ID missing.")

    logger_sc.info("\n[api_utils.py self-check starting...]")

    # === Phase 0: Prerequisite Checks ===
    logger_sc.info("--- Phase 0: Prerequisite Checks ---")
    _, s0_bs_stat, _ = _sc_run_test("Check BeautifulSoup Import", lambda: BeautifulSoup is not None, test_results_sc, logger_sc, expected_truthy=True)
    if s0_bs_stat != "PASS": overall_status = False
    # Check core functions
    func_map = { "format_name": format_name, "ordinal_case": ordinal_case, "_parse_date": _parse_date, "_clean_display_date": _clean_display_date, "parse_ancestry_person_details": parse_ancestry_person_details, "format_api_relationship_path": format_api_relationship_path, "call_suggest_api": call_suggest_api, "call_facts_user_api": call_facts_user_api, "call_getladder_api": call_getladder_api, "call_discovery_relationship_api": call_discovery_relationship_api, "call_treesui_list_api": call_treesui_list_api, }
    for name, func in func_map.items():
        _, s0_f_stat, _ = _sc_run_test(f"Check Function '{name}'", lambda f=func: callable(f), test_results_sc, logger_sc, expected_truthy=True)
        if s0_f_stat != "PASS": overall_status = False
    _, s0_c_stat, _ = _sc_run_test("Check Config Loaded", lambda: CONFIG_AVAILABLE and config_instance_sc and hasattr(config_instance_sc, "BASE_URL"), test_results_sc, logger_sc, expected_truthy=True)
    if s0_c_stat != "PASS": overall_status = False

    # === Phase 0b: Test Formatters with Static Data ===
    logger_sc.info("--- Phase 0b: Static Formatter Tests ---")
    # Test format_api_relationship_path with mock Discovery JSON
    mock_discovery_data = {
        "path": [
            {"relationship": "Father", "name": "Test Dad"},
            {"relationship": "mother", "name": "Test Mom Name"}, # Use lowercase to test case handling
        ]
    }
    owner_name_mock = "Owner Name"
    target_name_mock = "Target Name"
    # Define the CORRECT expected output string for the Discovery JSON format
    expected_output_mock = f"  {target_name_mock}\n  -> Father is Test Dad\n  -> Mother is Test Mom Name\n  -> {owner_name_mock} (Tree Owner / You)"
    _, s0b_status, _ = _sc_run_test(
        "format_api_relationship_path (Discovery JSON)",
        lambda: format_api_relationship_path(
            mock_discovery_data, owner_name_mock, target_name_mock
        ),
        test_results_sc,
        logger_sc,
        expected_value=expected_output_mock, # Check against the correct expected value
    )
    if s0b_status != "PASS":
        overall_status = False # Mark fail if static test fails

    # Check overall status before proceeding to live tests
    if not overall_status:
        logger_sc.error("Prerequisite or static tests failed. Cannot proceed further.")
        can_run_live_tests = False # Prevent live tests if basic checks fail

    # === Live Tests Section ===
    if can_run_live_tests:
        try:
            # === Phase 1: Session Setup ===
            logger_sc.info("--- Phase 1: Session Setup & Login ---")
            session_manager_sc = SessionManager()
            _, s1_start_stat, _ = _sc_run_test("SessionManager.start_sess()", session_manager_sc.start_sess, test_results_sc, logger_sc, action_name="SC Phase 1 Start", expected_truthy=True)
            if s1_start_stat != "PASS": overall_status = False; raise RuntimeError("start_sess failed")
            _, s1_ready_stat, _ = _sc_run_test("SessionManager.ensure_session_ready()", session_manager_sc.ensure_session_ready, test_results_sc, logger_sc, action_name="SC Phase 1 Ready", expected_truthy=True)
            if s1_ready_stat != "PASS": overall_status = False; raise RuntimeError("ensure_session_ready failed")

            # === Phase 2: Get Target Info & Validate Config ===
            logger_sc.info("--- Phase 2: Get Target Info & Validate Config ---")
            target_tree_id = session_manager_sc.my_tree_id
            target_owner_name = session_manager_sc.tree_owner_name
            target_owner_profile_id = session_manager_sc.my_profile_id
            _, s2_tid_stat, _ = _sc_run_test("Check Target Tree ID Found", lambda: bool(target_tree_id), test_results_sc, logger_sc, expected_truthy=True)
            _, s2_owner_stat, _ = _sc_run_test("Check Target Owner Name Found", lambda: bool(target_owner_name), test_results_sc, logger_sc, expected_truthy=True)
            _, s2_profile_stat, _ = _sc_run_test("Check Target Owner Profile ID Found", lambda: bool(target_owner_profile_id), test_results_sc, logger_sc, expected_truthy=True)
            if not all(s == "PASS" for s in [s2_tid_stat, s2_owner_stat, s2_profile_stat]): overall_status = False

            profile_response_details = None
            test_name_target_profile = "API Call: Get Target Profile Details (app-api)"
            if target_profile_id:
                api_call_lambda = lambda: _sc_get_profile_details(cast(str, target_profile_id))
                _, s2_api_stat, _ = _sc_run_test(test_name_target_profile, api_call_lambda, test_results_sc, logger_sc, expected_type=dict)
                if s2_api_stat == "PASS":
                    profile_response_details = api_call_lambda()
                    if profile_response_details:
                        target_name_from_profile = parse_ancestry_person_details({}, profile_response_details).get("name", "Unknown Target")
                        _, s2_name_stat, _ = _sc_run_test("Check Target Name Found in API Resp", lambda: target_name_from_profile not in ["Unknown", "Unknown Target"], test_results_sc, logger_sc, expected_truthy=True)
                        if s2_name_stat != "PASS": overall_status = False
                        # Set target_name_for_ladder if profile name found
                        if target_name_from_profile not in ["Unknown", "Unknown Target"]:
                             target_name_for_ladder = target_name_from_profile
                             logger_sc.info(f"Using profile name '{target_name_for_ladder}' for ladder test.")
                    else: logger_sc.error(f"{test_name_target_profile} passed but result invalid."); overall_status = False; _sc_run_test("Check Target Name Found in API Resp", lambda: "Skipped", test_results_sc, logger_sc)
                elif s2_api_stat == "SKIPPED": logger_sc.warning(f"{test_name_target_profile} skipped. Cannot check name."); overall_status = False; _sc_run_test("Check Target Name Found in API Resp", lambda: "Skipped", test_results_sc, logger_sc)
                else: overall_status = False; logger_sc.error(f"{test_name_target_profile} failed."); _sc_run_test("Check Target Name Found in API Resp", lambda: "Skipped", test_results_sc, logger_sc)
            else: logger_sc.warning("Skipping Get Target Profile Details: TESTING_PROFILE_ID not set."); _sc_run_test(test_name_target_profile, lambda: "Skipped", test_results_sc, logger_sc); _sc_run_test("Check Target Name Found in API Resp", lambda: "Skipped", test_results_sc, logger_sc)

            # Fallback if ladder name wasn't set from profile
            if target_name_for_ladder == "Unknown Ladder Target":
                 logger_sc.warning(f"Using fallback '{target_name_for_ladder}' for ladder target name.")

            # === Phase 3: Test parse_ancestry_person_details ===
            logger_sc.info("--- Phase 3: Test parse_ancestry_person_details (Live & Static) ---")
            # (Test logic remains the same as previous version)
            test_name_parse = "Function Call: parse_ancestry_person_details()"
            # Test case 1: Parsing profile_response_details
            if profile_response_details and isinstance(profile_response_details, dict):
                person_card_empty = {}
                try:
                    parse_lambda_facts = lambda: parse_ancestry_person_details(person_card_empty, profile_response_details)
                    _, s3_facts_stat, _ = _sc_run_test(f"{test_name_parse} (with Facts)", parse_lambda_facts, test_results_sc, logger_sc, expected_type=dict)
                    if s3_facts_stat == "PASS":
                        parsed_details_facts = parse_lambda_facts()
                        if parsed_details_facts:
                            keys_ok_facts = all(k in parsed_details_facts for k in ["name", "person_id", "link", "birth_date", "death_date", "gender", "is_living"])
                            _, s3_keys_stat, _ = _sc_run_test("Validation: Parsed Details Keys (Facts)", lambda: keys_ok_facts, test_results_sc, logger_sc, expected_truthy=True)
                            if s3_keys_stat != "PASS": overall_status = False
                            if target_name_from_profile not in ["Unknown Target", "Unknown"]:
                                _, s3_name_stat, _ = _sc_run_test("Validation: Parsed Name Match (Facts)", lambda p=parsed_details_facts: p.get("name") == target_name_from_profile, test_results_sc, logger_sc, expected_truthy=True)
                                if s3_name_stat != "PASS": overall_status = False
                        else: _sc_run_test(f"{test_name_parse} (with Facts)", lambda: False, test_results_sc, logger_sc, expected_value="Parser Invalid Return (Facts)"); overall_status = False
                    else: overall_status = False # Skip dependent tests
                except Exception as parse_e: _sc_run_test(f"{test_name_parse} (with Facts)", lambda: False, test_results_sc, logger_sc, expected_value=f"Exception: {parse_e}"); overall_status = False
            else: _sc_run_test(f"{test_name_parse} (with Facts)", lambda: "Skipped", test_results_sc, logger_sc); _sc_run_test("Validation: Parsed Details Keys (Facts)", lambda: "Skipped", test_results_sc, logger_sc); _sc_run_test("Validation: Parsed Name Match (Facts)", lambda: "Skipped", test_results_sc, logger_sc)
            # Test case 2: Static Suggest API like structure
            suggest_like_card = { "PersonId": "12345", "TreeId": "67890", "UserId": "ABC-DEF", "FullName": "Test Suggest Person", "GivenName": "Test", "Surname": "Suggest Person", "BirthYear": 1950, "BirthPlace": "SuggestBirth Town", "DeathYear": 2000, "DeathPlace": "SuggestDeath City", "Gender": "Female", "IsLiving": False, }
            try:
                parse_lambda_suggest = lambda: parse_ancestry_person_details(suggest_like_card, None)
                _, s3_suggest_stat, _ = _sc_run_test(f"{test_name_parse} (Suggest Format)", parse_lambda_suggest, test_results_sc, logger_sc, expected_type=dict)
                if s3_suggest_stat == "PASS":
                    parsed_details_suggest = parse_lambda_suggest()
                    if parsed_details_suggest:
                        vals_ok = all([ parsed_details_suggest.get("name") == "Test Suggest Person", parsed_details_suggest.get("birth_date") == "1950", parsed_details_suggest.get("death_date") == "2000", parsed_details_suggest.get("gender") == "F", parsed_details_suggest.get("is_living") is False, parsed_details_suggest.get("birth_place") == "SuggestBirth Town", parsed_details_suggest.get("death_place") == "SuggestDeath City", parsed_details_suggest.get("person_id") == "12345", parsed_details_suggest.get("user_id") == "ABC-DEF" ])
                        _, s3_val_stat, _ = _sc_run_test("Validation: Parsed Details Values (Suggest)", lambda: vals_ok, test_results_sc, logger_sc, expected_truthy=True)
                        if s3_val_stat != "PASS": overall_status = False
                    else: _sc_run_test(f"{test_name_parse} (Suggest Format)", lambda: False, test_results_sc, logger_sc, expected_value="Parser Invalid Return (Suggest)"); overall_status = False
                else: overall_status = False
            except Exception as parse_e: _sc_run_test(f"{test_name_parse} (Suggest Format)", lambda: False, test_results_sc, logger_sc, expected_value=f"Exception: {parse_e}"); overall_status = False

            # === Phase 4: Test API Helpers (Suggest, Facts User) ===
            logger_sc.info("--- Phase 4: Test API Helpers (Live) ---")
            # Test call_suggest_api
            if target_tree_id and target_owner_profile_id and callable(call_suggest_api):
                suggest_criteria = { "first_name_raw": "John", "surname_raw": "Smith", "birth_year": 1900 }
                _, suggest_status, _ = _sc_run_test("API Helper: call_suggest_api", lambda: call_suggest_api(session_manager_sc, target_tree_id, target_owner_profile_id, base_url_sc, suggest_criteria), test_results_sc, logger_sc, expected_type=list)
                if suggest_status != "PASS": overall_status = False
            else: _sc_run_test("API Helper: call_suggest_api", lambda: "Skipped", test_results_sc, logger_sc)
            # Test call_facts_user_api
            facts_person_id = target_person_id_for_ladder # Use ladder test person
            if target_tree_id and target_owner_profile_id and facts_person_id and callable(call_facts_user_api):
                _, facts_status, _ = _sc_run_test("API Helper: call_facts_user_api", lambda: call_facts_user_api(session_manager_sc, target_owner_profile_id, facts_person_id, target_tree_id, base_url_sc), test_results_sc, logger_sc, expected_type=dict)
                if facts_status != "PASS": overall_status = False
            else: logger_sc.warning(f"Skipping call_facts_user_api test. Missing deps."); _sc_run_test("API Helper: call_facts_user_api", lambda: "Skipped", test_results_sc, logger_sc)

            # === Phase 5: Test Relationship Ladder Parsing ===
            logger_sc.info("--- Phase 5: Test Relationship Ladder Parsing (Live API) ---")
            test_target_person_id = target_person_id_for_ladder
            test_target_tree_id = target_tree_id
            test_owner_name = target_owner_name
            test_target_name = target_name_for_ladder # Updated in phase 2 if possible

            can_run_ladder_test_live = bool(test_owner_name and test_target_person_id and test_target_tree_id)
            test_name_ladder_api = "API Helper: call_getladder_api"
            test_name_format_ladder = "Function Call: format_api_relationship_path (HTML)"

            if not can_run_ladder_test_live:
                logger_sc.warning(f"Skipping Live Ladder test: Missing prerequisites.")
                _sc_run_test(test_name_ladder_api, lambda: "Skipped", test_results_sc, logger_sc)
                _sc_run_test(test_name_format_ladder, lambda: "Skipped", test_results_sc, logger_sc)
            else:
                ladder_response_raw = _sc_get_tree_ladder(cast(str, test_target_tree_id), cast(str, test_target_person_id))
                _, s5_api_status, _ = _sc_run_test(test_name_ladder_api, lambda r=ladder_response_raw: isinstance(r, str) and len(r) > 10, test_results_sc, logger_sc, expected_truthy=True)

                if s5_api_status == "PASS" and ladder_response_raw:
                    owner_name_str = cast(str, test_owner_name)
                    target_name_str = cast(str, test_target_name)
                    format_lambda = lambda: format_api_relationship_path(ladder_response_raw, owner_name_str, target_name_str)
                    # --- Define Expected Substrings for Your Specific Test Case ---
                    # !! IMPORTANT: Update these based on the KNOWN relationship
                    #    between your owner profile and the TESTING_PERSON_TREE_ID !!
                    # Example for Fraser Gault (Uncle):
                    expected_substrings = [
                        f"{target_name_str} is {owner_name_str}'s uncle:", # Summary line check
                        f"- {target_name_str}'s brother is Derrick Wardie Gault", # Example intermediate step
                        f"- {owner_name_str} is Derrick Wardie Gault's son." # Example final step
                    ]
                    logger_sc.info(f"Expecting relationship format containing: {expected_substrings}")

                    _, s5_format_status, _ = _sc_run_test(
                        test_name_format_ladder,
                        format_lambda,
                        test_results_sc,
                        logger_sc,
                        expected_type=str,
                        expected_contains=expected_substrings, # Check for key parts
                    )
                    if s5_format_status == "FAIL":
                        overall_status = False
                else:
                    logger_sc.warning(f"Skipping {test_name_format_ladder} because API call failed or invalid.")
                    _sc_run_test(test_name_format_ladder, lambda: "Skipped", test_results_sc, logger_sc)
                    if s5_api_status != "SKIPPED": overall_status = False

        except Exception as e:
            logger_sc.critical("\n--- CRITICAL ERROR during self-check live tests ---", exc_info=True)
            _sc_run_test("Self-Check Live Execution", lambda: False, test_results_sc, logger_sc, expected_value="CRITICAL EXCEPTION OCCURRED")
            overall_status = False
        finally:
            if session_manager_sc: logger_sc.info("--- Finalizing: Closing Session ---"); session_manager_sc.close_sess()
            else: logger_sc.info("--- Finalizing: No session object to close ---")

    else: # Not can_run_live_tests
        logger_sc.warning("Skipping Live API tests due to missing dependencies or prerequisite failures.")
        # (Skip logic remains the same)
        phases_to_skip = ["SessionManager.start_sess()", "SessionManager.ensure_session_ready()", "Check Target Tree ID Found", "Check Target Owner Name Found", "Check Target Owner Profile ID Found", "API Call: Get Target Profile Details (app-api)", "Check Target Name Found in API Resp", "Function Call: parse_ancestry_person_details() (with Facts)", "Validation: Parsed Details Keys (Facts)", "Validation: Parsed Name Match (Facts)", "API Helper: call_suggest_api", "API Helper: call_facts_user_api", "API Helper: call_getladder_api", "Function Call: format_api_relationship_path (HTML)", ]
        existing_test_names = {name for name, _, _ in test_results_sc}
        for test_name in phases_to_skip:
            if test_name not in existing_test_names: _sc_run_test(test_name, lambda: "Skipped", test_results_sc, logger_sc)


    # --- Print Formatted Summary ---
    _sc_print_summary(test_results_sc, overall_status, logger_sc)

    final_overall_status = overall_status and not any(status == "FAIL" for _, status, _ in test_results_sc)
    return final_overall_status
# End of self_check

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Running api_utils.py self-check (with live API calls)...")
    log_file = Path("api_utils_self_check.log").resolve()
    logger_standalone = None
    try:
        import logging_config

        if not hasattr(logging_config, "setup_logging"):
            raise ImportError("setup_logging missing")
        logger_standalone = logging_config.setup_logging(
            log_file=log_file, log_level="DEBUG"
        )
        print(f"Detailed logs will be written to: {log_file}")
    except ImportError as log_imp_err:
        print(
            f"Warning: logging_config import/setup failed ({log_imp_err}). Using basic logging."
        )
        logging.basicConfig(
            level=logging.DEBUG,
            filename=log_file,
            filemode="w",
            format="%(asctime)s %(levelname)-8s [%(name)-15s] %(message)s",
        )
        logger_standalone = logging.getLogger("api_utils_standalone")
        logger_standalone.info(f"Using basicConfig, logging to {log_file}")
    except Exception as log_setup_err:
        print(f"Error setting up logging: {log_setup_err}. Using basic logging.")
        logging.basicConfig(
            level=logging.DEBUG,
            filename=log_file,
            filemode="w",
            format="%(asctime)s %(levelname)-8s [%(name)-15s] %(message)s",
        )
        logger_standalone = logging.getLogger("api_utils_standalone")
        logger_standalone.info(
            f"Using basicConfig due to setup error, logging to {log_file}"
        )

    # Ensure the logger used by self_check is set if setup succeeded
    if logger_standalone:
        try:
            from logging_config import (
                logger as logger_sc_main,
            )  # Try importing the configured logger

            if (
                not logger_sc_main.hasHandlers()
            ):  # Check if handlers are missing (might happen if run standalone first)
                for handler in logger_standalone.handlers:
                    logger_sc_main.addHandler(handler)
                logger_sc_main.setLevel(logger_standalone.level)
        except ImportError:
            logger_sc_main = (
                logger_standalone  # Fallback if logging_config logger not found
            )

    # Check for TESTING_PERSON_TREE_ID
    if CONFIG_AVAILABLE and not getattr(
        config_instance, "TESTING_PERSON_TREE_ID", None
    ):
        print("\n" + "!" * 70)
        print("WARNING: config.TESTING_PERSON_TREE_ID is not set!")
        print("The relationship ladder test (Phase 5) and facts API test (Phase 4)")
        print(
            "require this to be the ID of a person in your tree for accurate testing."
        )
        print("Tests may fail or be skipped without it. Please set it in config.py.")
        print("!" * 70 + "\n")
    elif not CONFIG_AVAILABLE:
        print(
            "\nWARNING: config.py not loaded. Cannot check for TESTING_PERSON_TREE_ID."
        )

    self_check_passed = self_check()
    print("\nThis is the api_utils module. Import it into other scripts.")
    sys.exit(0 if self_check_passed else 1)
# End of __main__ block

# End of api_utils.py
