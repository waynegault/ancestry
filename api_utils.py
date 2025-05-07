# api_utils.py
"""
Utility functions for parsing Ancestry API responses and formatting API data.

Provides functions to:
- Parse person details from various Ancestry API responses.
- Format relationship paths from getladder API responses.
- Call specific Ancestry APIs (Suggest, Facts, Ladder, Discovery Relationship, TreesUI, Send Message, Profile Details, Header Trees, Tree Owner).
- Includes a self-check mechanism using live API calls (requires configuration).

Note: Relationship path formatting functions previously in test_relationship_path.py are now integrated here.
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
import html  # Used for unescaping HTML entities
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
import uuid  # For call_send_message_api

# --- Third-party imports ---
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None  # type: ignore
    BS4_AVAILABLE = False


# Initialize logger - Ensure logger is always available
# Use basicConfig as fallback if logging_config fails
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_utils")

# --- Local application imports ---
# Remove try-except blocks and fallbacks. Fail early if imports fail.
import utils  # noqa F401 # Keep utils import here if needed elsewhere in the module
from utils import SessionManager, _api_req, format_name, ordinal_case

logger.info(
    "Successfully imported base utils module (SessionManager, _api_req, format_name, ordinal_case)"
)

from gedcom_utils import _parse_date, _clean_display_date

logger.info("Successfully imported gedcom_utils date functions")

from config import config_instance, selenium_config

logger.info("Successfully imported config instances")

from database import Person  # Required for call_send_message_api

logger.info("Successfully imported Person from database module")


# --- Constants for moved/new functions ---

# For call_send_message_api
SEND_ERROR_INVALID_RECIPIENT = "send_error (invalid_recipient)"
SEND_ERROR_MISSING_OWN_ID = "send_error (missing_own_id)"
SEND_ERROR_INTERNAL_MODE = "send_error (internal_mode_error)"
SEND_ERROR_API_PREP_FAILED = "send_error (api_prep_failed)"
SEND_ERROR_UNEXPECTED_FORMAT = "send_error (unexpected_format)"
SEND_ERROR_VALIDATION_FAILED = "send_error (validation_failed)"
SEND_ERROR_POST_FAILED = "send_error (post_failed)"
SEND_ERROR_UNKNOWN = "send_error (unknown)"
SEND_SUCCESS_DELIVERED = "delivered OK"
SEND_SUCCESS_DRY_RUN = "typed (dry_run)"

API_PATH_SEND_MESSAGE_NEW = "app-api/express/v2/conversations/message"
API_PATH_SEND_MESSAGE_EXISTING = "app-api/express/v2/conversations/{conv_id}"
KEY_CONVERSATION_ID = "conversation_id"
KEY_MESSAGE = "message"
KEY_AUTHOR = "author"

# For call_profile_details_api
API_PATH_PROFILE_DETAILS = "/app-api/express/v1/profiles/details"
KEY_FIRST_NAME = "FirstName"
KEY_DISPLAY_NAME_APIUTILS = "displayName"
KEY_LAST_LOGIN_DATE = "LastLoginDate"
KEY_IS_CONTACTABLE = "IsContactable"

# For call_header_trees_api_for_tree_id
API_PATH_HEADER_TREES = "api/uhome/secure/rest/header/trees"
KEY_MENUITEMS = "menuitems"
KEY_URL = "url"
KEY_TEXT = "text"

# For call_tree_owner_api
API_PATH_TREE_OWNER_INFO = "api/uhome/secure/rest/user/tree-info"
KEY_OWNER = "owner"

# For Relationship Path Formatting Test in self_check
TEST_RELATIONSHIP_PATH_RAW_API_RESPONSE = r"""
no({
    "html": "\u003cul class=\"textCenter\"\u003e \u003cli\u003e\u003cb\u003eElizabeth \u0027Betty\u0027 Cruickshank\u003c/b\u003e 1839-1886\u003cbr /\u003e\u003ci\u003e\u003cb\u003e3rd great-grandmother\u003c/b\u003e\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560698\"\u003eMargaret Simpson 1865-1946\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eDaughter of Elizabeth \u0027Betty\u0027 Cruickshank\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560684\"\u003eAlexander Stables 1899-1948\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eSon of Margaret Simpson\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560677\"\u003eCatherine Margaret Stables 1924-2004\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eDaughter of Alexander Stables\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560544\"\u003eFrances Margaret Milne 1947-\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eDaughter of Catherine Margaret Stables\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"bottomName\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560836\"\u003e\u003cb\u003eWayne Gordon Gault\u003c/b\u003e\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eYou are the son of Frances Margaret Milne\u003c/i\u003e\u003c/li\u003e \u003c/ul\u003e ",
    "title": "Relationship to me",
    "printText": "Print",
    "status": "success"
})
"""

# Adjusted EXPECTED_FORMATTED_PATH_STRING for correct spacing and last line format
EXPECTED_FORMATTED_PATH_STRING = """Elizabeth 'Betty' Cruickshank (1839-1886) is Wayne Gordon Gault's 3rd Great-Grandmother:

* Margaret Simpson (1865-1946) is Elizabeth 'Betty' Cruickshank's daughter
* Alexander Stables (1899-1948) is Margaret Simpson's son
* Catherine Margaret Stables (1924-2004) is Alexander Stables's daughter
* Frances Margaret Milne (b. 1947) is Catherine Margaret Stables's daughter
Wayne Gordon Gault is Frances Margaret Milne's son"""


# --- Helper Functions for parse_ancestry_person_details ---
def _extract_name_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> str:
    name = "Unknown"
    formatter = format_name
    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            name = person_info.get("personName", name)
        # End of if
        if name == "Unknown":
            name = facts_data.get("personName", name)
        # End of if
        if name == "Unknown":
            name = facts_data.get("DisplayName", name)
        # End of if
        if name == "Unknown":
            name = facts_data.get("PersonFullName", name)
        # End of if
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
                # End of if
            # End of if
        # End of if
        if name == "Unknown":
            first_name_pd = facts_data.get("FirstName")
            last_name_pd = facts_data.get("LastName")
            if first_name_pd or last_name_pd:
                name = (
                    f"{first_name_pd or ''} {last_name_pd or ''}".strip() or "Unknown"
                )
            # End of if
        # End of if
    # End of if
    if name == "Unknown" and person_card:
        suggest_fullname = person_card.get("FullName")
        suggest_given = person_card.get("GivenName")
        suggest_sur = person_card.get("Surname")
        if suggest_fullname:
            name = suggest_fullname
        elif suggest_given or suggest_sur:
            name = f"{suggest_given or ''} {suggest_sur or ''}".strip() or "Unknown"
        # End of if/elif
        if name == "Unknown":
            name = person_card.get("name", "Unknown")
        # End of if
    # End of if
    formatted_name_val = formatter(name) if name and name != "Unknown" else "Unknown"
    return "Unknown" if formatted_name_val == "Valued Relative" else formatted_name_val


# End of _extract_name_from_api_details


def _extract_gender_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Optional[str]:
    gender = None
    gender_str = None
    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            gender_str = person_info.get("gender")
        # End of if
        if not gender_str:
            gender_str = facts_data.get("gender")
        # End of if
        if not gender_str:
            gender_str = facts_data.get("PersonGender")
        # End of if
        if not gender_str:
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
                # End of if
            # End of if
        # End of if
    # End of if
    if not gender_str and person_card:
        gender_str = person_card.get("Gender")
        if not gender_str:
            gender_str = person_card.get("gender")
        # End of if
    # End of if
    if gender_str and isinstance(gender_str, str):
        gender_str_lower = gender_str.lower()
        if gender_str_lower == "male":
            gender = "M"
        elif gender_str_lower == "female":
            gender = "F"
        elif gender_str_lower in ["m", "f"]:
            gender = gender_str_lower.upper()
        # End of if/elif
    # End of if
    return gender


# End of _extract_gender_from_api_details


def _extract_living_status_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Optional[bool]:
    is_living = None
    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            is_living = person_info.get("isLiving")
        # End of if
        if is_living is None:
            is_living = facts_data.get("isLiving")
        # End of if
        if is_living is None:
            is_living = facts_data.get("IsPersonLiving")
        # End of if
    # End of if
    if is_living is None and person_card:
        is_living = person_card.get("IsLiving")
        if is_living is None:
            is_living = person_card.get("isLiving")
        # End of if
    # End of if
    return bool(is_living) if is_living is not None else None


# End of _extract_living_status_from_api_details


def _extract_event_from_api_details(
    event_type: str, person_card: Dict, facts_data: Optional[Dict]
) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    date_str: Optional[str] = None
    place_str: Optional[str] = None
    date_obj: Optional[datetime] = None
    parser = _parse_date
    event_key_lower = event_type.lower()
    suggest_year_key = f"{event_type}Year"
    suggest_place_key = f"{event_type}Place"
    facts_user_key = event_type
    app_api_key = f"{event_key_lower}Date"
    app_api_facts_key = event_type
    found_in_facts = False
    if facts_data and isinstance(facts_data, dict):
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
                    f"Found primary {event_type} fact in PersonFacts: Date='{date_str}', Place='{place_str}'"
                )
                if isinstance(parsed_date_data, dict) and parser:
                    year = parsed_date_data.get("Year")
                    month = parsed_date_data.get("Month")
                    day = parsed_date_data.get("Day")
                    if year:
                        try:
                            temp_date_str = str(year)
                            if month:
                                temp_date_str += f"-{str(month).zfill(2)}"
                            # End of if
                            if day:
                                temp_date_str += f"-{str(day).zfill(2)}"
                            # End of if
                            date_obj = parser(temp_date_str)
                            logger.debug(
                                f"Parsed {event_type} date object from ParsedDate: {date_obj}"
                            )
                        except Exception as dt_err:
                            logger.warning(
                                f"Could not parse {event_type} date from ParsedDate {parsed_date_data}: {dt_err}"
                            )
                        # End of try/except
                    # End of if year
                # End of if parsed_date_data
            # End of if event_fact
        # End of if
        if not found_in_facts:
            fact_group_list = facts_data.get("facts", {}).get(app_api_facts_key, [])
            if fact_group_list and isinstance(fact_group_list, list):
                fact_group = fact_group_list[0]
                if isinstance(fact_group, dict):
                    date_info = fact_group.get("date", {})
                    place_info = fact_group.get("place", {})
                    if isinstance(date_info, dict):
                        date_str = date_info.get(
                            "normalized", date_info.get("original")
                        )
                    # End of if
                    if isinstance(place_info, dict):
                        place_str = place_info.get("placeName")
                    # End of if
                    found_in_facts = True
                # End of if
            # End of if
        # End of if
        if not found_in_facts:
            event_fact_alt = facts_data.get(app_api_key)
            if event_fact_alt and isinstance(event_fact_alt, dict):
                date_str = event_fact_alt.get("normalized", event_fact_alt.get("date"))
                place_str = event_fact_alt.get("place", place_str)
                found_in_facts = True
            elif isinstance(event_fact_alt, str):
                date_str = event_fact_alt
                found_in_facts = True
            # End of if/elif
        # End of if
    # End of if
    if not found_in_facts and person_card:
        suggest_year = person_card.get(suggest_year_key)
        suggest_place = person_card.get(suggest_place_key)
        if suggest_year:
            date_str = str(suggest_year)
            place_str = suggest_place
            logger.debug(
                f"Using Suggest API keys for {event_type}: Year='{date_str}', Place='{place_str}'"
            )
        else:
            event_info_card = person_card.get(event_key_lower, "")
            if event_info_card and isinstance(event_info_card, str):
                parts = re.split(r"\s+in\s+", event_info_card, maxsplit=1)
                date_str = parts[0].strip() if parts else event_info_card
                if place_str is None and len(parts) > 1:
                    place_str = parts[1].strip()
                # End of if
            elif isinstance(event_info_card, dict):
                date_str = event_info_card.get("date", date_str)
                if place_str is None:
                    place_str = event_info_card.get("place", place_str)
                # End of if
            # End of if/elif
        # End of if/else
    # End of if
    if date_obj is None and date_str and parser:
        try:
            date_obj = parser(date_str)
        except Exception as parse_err:
            logger.warning(
                f"Failed to parse {event_type} date string '{date_str}': {parse_err}"
            )
        # End of try/except
    # End of if
    return date_str, place_str, date_obj


# End of _extract_event_from_api_details


def _generate_person_link(
    person_id: Optional[str], tree_id: Optional[str], base_url: str
) -> str:
    if tree_id and person_id:
        return f"{base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/facts"
    elif person_id:
        return f"{base_url}/discoveryui-matches/list/summary/{person_id}"
    # End of if/elif
    return "(Link unavailable)"


# End of _generate_person_link


def parse_ancestry_person_details(
    person_card: Dict, facts_data: Optional[Dict] = None
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "name": "Unknown",
        "birth_date": "N/A",
        "birth_place": None,
        "api_birth_obj": None,
        "death_date": "N/A",
        "death_place": None,
        "api_death_obj": None,
        "gender": None,
        "is_living": None,
        "person_id": person_card.get("PersonId"),
        "tree_id": person_card.get("TreeId"),
        "user_id": person_card.get("UserId"),
        "link": "(Link unavailable)",
    }
    if not details["person_id"]:
        details["person_id"] = person_card.get("personId")
    # End of if
    if not details["tree_id"]:
        details["tree_id"] = person_card.get("treeId")
    # End of if

    if facts_data and isinstance(facts_data, dict):
        details["person_id"] = facts_data.get("PersonId", details["person_id"])
        details["tree_id"] = facts_data.get("TreeId", details["tree_id"])
        details["user_id"] = facts_data.get("UserId", details["user_id"])
        if not details["user_id"]:
            person_info = facts_data.get("person", {})
            if isinstance(person_info, dict):
                details["user_id"] = person_info.get("userId", details["user_id"])
            # End of if
        # End of if
    # End of if

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

    cleaner = _clean_display_date
    details["birth_date"] = cleaner(birth_date_raw) if birth_date_raw else "N/A"
    details["death_date"] = cleaner(death_date_raw) if death_date_raw else "N/A"

    if details["birth_date"] == "N/A" and details["api_birth_obj"]:
        details["birth_date"] = str(details["api_birth_obj"].year)
    # End of if
    if details["death_date"] == "N/A" and details["api_death_obj"]:
        details["death_date"] = str(details["api_death_obj"].year)
    # End of if

    base_url_for_link = getattr(
        config_instance, "BASE_URL", "https://www.ancestry.com"
    ).rstrip("/")

    link_id = details["user_id"] or details["person_id"]
    link_tree_id = details["tree_id"] if not details["user_id"] else None

    details["link"] = _generate_person_link(link_id, link_tree_id, base_url_for_link)

    logger.debug(
        f"Parsed API details for '{details.get('name', 'Unknown')}': PersonID={details.get('person_id')}, TreeID={details.get('tree_id', 'N/A')}, UserID={details.get('user_id', 'N/A')}, Born='{details.get('birth_date')}' [{details.get('api_birth_obj')}] in '{details.get('birth_place') or '?'}', Died='{details.get('death_date')}' [{details.get('api_death_obj')}] in '{details.get('death_place') or '?'}', Gender='{details.get('gender') or '?'}', Living={details.get('is_living')}, Link='{details.get('link')}'"
    )
    return details


# End of parse_ancestry_person_details


def format_api_relationship_path(
    api_response_data: Union[str, Dict, None], owner_name: str, target_name: str
) -> str:
    """
    Parses relationship data from Ancestry APIs and formats it into a readable path.
    Handles getladder API HTML/JSONP response.
    Uses format_name and ordinal_case from utils.py.
    """

    # --- Inner Helper Functions ---
    def _inner_get_relationship_term(
        person_a_gender: Optional[str], basic_relationship: str
    ) -> str:
        term = basic_relationship.capitalize()
        rel_lower = basic_relationship.lower()
        if rel_lower == "parent":
            if person_a_gender == "M":
                term = "Father"
            elif person_a_gender == "F":
                term = "Mother"
            # End of if/elif
        elif rel_lower == "child":
            if person_a_gender == "M":
                term = "Son"
            elif person_a_gender == "F":
                term = "Daughter"
            # End of if/elif
        elif rel_lower == "sibling":
            if person_a_gender == "M":
                term = "Brother"
            elif person_a_gender == "F":
                term = "Sister"
            # End of if/elif
        elif rel_lower == "spouse":
            if person_a_gender == "M":
                term = "Husband"
            elif person_a_gender == "F":
                term = "Wife"
            # End of if/elif
        # End of if/elif chain

        if any(char.isdigit() for char in term):
            try:
                term = ordinal_case(term)
            except Exception as ord_err:
                logger.warning(f"Failed to apply ordinal case to '{term}': {ord_err}")
            # End of try/except
        # End of if
        return term

    # End of _inner_get_relationship_term

    def _inner_extract_name_and_lifespan(text_content: str) -> Tuple[str, str]:
        name_part = text_content
        lifespan_str_formatted = ""

        match_yyyy_yyyy = re.search(r"\s+(\d{4}[–-]\d{4})$", text_content)
        if match_yyyy_yyyy:
            lifespan_raw = match_yyyy_yyyy.group(1)
            lifespan_str_formatted = f"({lifespan_raw.replace('–', '-')})"
            name_part = text_content[: match_yyyy_yyyy.start()].strip()
            return name_part, lifespan_str_formatted
        # End of if

        match_yyyy_living = re.search(r"\s+(\d{4})-$", text_content)
        if match_yyyy_living:
            birth_year = match_yyyy_living.group(1)
            lifespan_str_formatted = f"(b. {birth_year})"
            name_part = text_content[: match_yyyy_living.start()].strip()
            return name_part, lifespan_str_formatted
        # End of if

        return name_part.strip(), lifespan_str_formatted

    # End of _inner_extract_name_and_lifespan

    # --- Main Function Logic ---
    if not api_response_data:
        logger.warning(
            "format_api_relationship_path: Received empty API response data."
        )
        return "(No relationship data received from API)"
    # End of if

    html_content_raw: Optional[str] = None
    json_data: Optional[Dict] = None
    api_status: str = "unknown"

    if isinstance(api_response_data, dict):
        if "path" in api_response_data and isinstance(
            api_response_data.get("path"), list
        ):
            logger.debug("Detected direct JSON 'path' format (Discovery API).")
            json_data = api_response_data
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
            # End of if
        else:
            logger.warning(
                f"Received unhandled dictionary format: Keys={list(api_response_data.keys())}"
            )
            return "(Received unhandled dictionary format from API)"
        # End of if/elif/else
    elif isinstance(api_response_data, str):
        if api_response_data.strip().startswith(
            "no("
        ) and api_response_data.strip().endswith(")"):
            try:
                json_part_match = re.search(
                    r"^\s*no\((.*)\)\s*$", api_response_data, re.DOTALL
                )
                if json_part_match:
                    json_part_str_local = json_part_match.group(1).strip()
                    parsed_json = json.loads(json_part_str_local)
                    api_status = parsed_json.get("status", "unknown")
                    if api_status == "success":
                        html_content_raw = parsed_json.get("html")
                        if not isinstance(html_content_raw, str):
                            html_content_raw = None
                            logger.warning(
                                "JSONP status 'success', but 'html' key missing or not a string."
                            )
                        # End of if
                    else:
                        return f"(API status '{api_status}' in JSONP: {parsed_json.get('message', 'Error')})"
                    # End of if/else
                else:
                    logger.warning("Could not extract JSON part from JSONP wrapper.")
                    return "(Could not extract JSON from JSONP wrapper)"
                # End of if/else
            except json.JSONDecodeError as json_err:
                error_context = (
                    f" near: {json_part_str_local[:100]}..."
                    if "json_part_str_local" in locals()
                    else ""
                )
                logger.error(
                    f"Error decoding JSON part from JSONP: {json_err}{error_context}"
                )
                return f"(Error parsing JSONP data: {json_err}{error_context})"
            # End of try/except
            except Exception as e:
                logger.error(f"Error processing JSONP string: {e}", exc_info=True)
                return f"(Unexpected error processing JSONP: {e})"
            # End of try/except
        else:
            logger.warning(
                "Input string not in expected no(...) JSONP format, and not a dict."
            )
            return "(Input string not in expected JSONP format)"
        # End of if/else
    else:
        return f"(Unsupported data type received: {type(api_response_data)})"
    # End of if/elif/else

    if json_data and "path" in json_data:
        path_steps_json = []
        discovery_path = json_data["path"]
        if isinstance(discovery_path, list) and discovery_path:
            logger.info("Formatting relationship path from Discovery API JSON.")
            path_steps_json.append(f"*   {format_name(target_name)}")
            for i, step in enumerate(discovery_path):
                step_name = format_name(step.get("name", "?"))
                step_rel = step.get("relationship", "?")
                step_rel_display = _inner_get_relationship_term(
                    None, step_rel
                ).capitalize()
                path_steps_json.append(f"    -> is {step_rel_display} of")
                path_steps_json.append(f"*   {step_name}")
            # End of for
            path_steps_json.append(f"    -> leads to")
            path_steps_json.append(f"*   {owner_name} (You)")
            result_str = "\n".join(path_steps_json)
            logger.debug(f"Formatted Discovery relationship path:\n{result_str}")
            return result_str
        else:
            logger.warning(
                f"Discovery 'path' data invalid or empty: {json_data.get('path')}"
            )
            return "(Discovery path found but is empty or invalid)"
        # End of if/else
    # End of if json_data

    if not html_content_raw:
        logger.warning("No processable HTML content found for relationship path.")
        return "(Could not find or extract relationship HTML content)"
    # End of if

    html_content_decoded: Optional[str] = None
    try:
        html_content_intermediate = html.unescape(html_content_raw)
        html_content_decoded = bytes(html_content_intermediate, "utf-8").decode(
            "unicode_escape"
        )
    except Exception as decode_err:
        logger.error(f"Failed to decode HTML content: {decode_err}", exc_info=True)
        html_content_decoded = html_content_raw
    # End of try/except

    if not BS4_AVAILABLE or not BeautifulSoup:
        logger.error("BeautifulSoup library not available. Cannot parse HTML.")
        return "(Cannot parse relationship HTML - BeautifulSoup library missing)"
    # End of if

    try:
        soup = None
        for parser_name in ["lxml", "html.parser"]:
            try:
                soup = BeautifulSoup(html_content_decoded, parser_name)
                logger.info(f"Successfully parsed HTML using '{parser_name}'.")
                break
            except Exception as parse_err_bs:
                logger.warning(
                    f"Error using '{parser_name}' parser: {parse_err_bs}. Trying next."
                )
            # End of try/except
        # End of for

        if not soup:
            logger.error("BeautifulSoup failed to parse HTML with available parsers.")
            return "(Error parsing relationship HTML - BeautifulSoup failed)"
        # End of if

        list_items = soup.select("ul.textCenter li")
        path_items = [
            li for li in list_items if "iconArrowDown" not in (li.get("class") or [])
        ]
        logger.debug(f"Found {len(path_items)} relevant path items (summary + people).")

        if not path_items:
            logger.warning(
                "Expected list items ('ul.textCenter li'), found none in parsed HTML or no relevant items."
            )
            return "(Relationship HTML structure not as expected - Found 0 relevant list items)"
        # End of if

        target_li = path_items[0]
        target_b_tag = target_li.find("b")
        target_year_text_raw = ""
        if (
            target_b_tag
            and target_b_tag.next_sibling
            and isinstance(target_b_tag.next_sibling, str)
        ):
            target_year_text_raw = target_b_tag.next_sibling.strip()
        # End of if

        target_lifespan_formatted = ""
        if re.fullmatch(r"\d{4}[–-]\d{4}", target_year_text_raw):
            target_lifespan_formatted = f"({target_year_text_raw.replace('–','-')})"
        # End of if

        overall_relationship_text = "Unknown Relationship"
        summary_tag_html = target_li.select_one("i b, i")
        if summary_tag_html:
            raw_overall_rel = summary_tag_html.get_text(strip=True)
            ordinal_match = re.match(
                r"(\d+(?:st|nd|rd|th))\s*(.*)", raw_overall_rel, re.IGNORECASE
            )
            if ordinal_match:
                ordinal_part = ordinal_match.group(1)
                text_part = ordinal_match.group(2)
                overall_relationship_text = f"{ordinal_part} {format_name(text_part)}"
            else:
                overall_relationship_text = format_name(raw_overall_rel)
            # End of if/else
        # End of if

        # Use format_name on target_name argument for the summary line
        # This should use the version of format_name from utils.py
        formatted_target_name_for_summary = format_name(target_name)

        summary_line_parts = [formatted_target_name_for_summary]
        if target_lifespan_formatted:
            summary_line_parts.append(target_lifespan_formatted)
        # End of if
        summary_line_parts.append(
            f"is {format_name(owner_name)}'s {overall_relationship_text}:"
        )
        summary_line = " ".join(summary_line_parts)

        path_lines = []
        for i in range(1, len(path_items)):
            current_person_li = path_items[i]

            current_name_tag = current_person_li.find("a") or current_person_li.find(
                "b"
            )
            current_name_raw_from_tag = (
                current_name_tag.get_text(strip=True)
                if current_name_tag
                else "Unknown Current"
            )

            current_name_parsed, current_lifespan_formatted = (
                _inner_extract_name_and_lifespan(current_name_raw_from_tag)
            )
            current_name_display = format_name(current_name_parsed)

            if current_name_parsed.lower() == "you" or (
                i == len(path_items) - 1
                and format_name(owner_name).startswith(current_name_display)
            ):
                current_name_display = format_name(owner_name)
            # End of if

            prev_name_display_for_relation: str
            if i == 1:
                prev_name_display_for_relation = format_name(target_name)
            else:
                prev_li = path_items[i - 1]
                prev_name_tag = prev_li.find("a") or prev_li.find("b")
                prev_name_raw = (
                    prev_name_tag.get_text(strip=True)
                    if prev_name_tag
                    else "Unknown Previous"
                )
                parsed_prev_name, _ = _inner_extract_name_and_lifespan(prev_name_raw)
                prev_name_display_for_relation = format_name(parsed_prev_name)
                if parsed_prev_name.lower() == "you":
                    prev_name_display_for_relation = format_name(owner_name)
                # End of if
            # End of if/else

            relationship_term = "related"
            desc_tag_html = current_person_li.find("i")
            if desc_tag_html:
                desc_text_html = desc_tag_html.get_text(strip=True)
                rel_match_html = re.search(
                    r"\b(son|daughter|father|mother|husband|wife|spouse|brother|sister|parent|child|sibling)\b",
                    desc_text_html,
                    re.IGNORECASE,
                )
                if rel_match_html:
                    relationship_term = rel_match_html.group(1).lower()
                elif "you are the" in desc_text_html.lower():
                    you_match = re.search(
                        r"You\s+are\s+the\s+([\w\s]+)\s+of",
                        desc_text_html,
                        re.IGNORECASE,
                    )
                    if you_match:
                        relationship_term = you_match.group(1).strip().lower()
                    # End of if
                # End of if/elif
            # End of if

            line_parts = []
            line_parts.extend(["*", current_name_display]) 

            if current_lifespan_formatted:
                line_parts.append(current_lifespan_formatted)
            # End of if
            line_parts.append(
                f"is {prev_name_display_for_relation}'s {relationship_term}"
            )
            path_lines.append(" ".join(line_parts))
        # End of for

        result_str = f"{summary_line}\n\n" + "\n".join(path_lines)
        logger.info("Formatted relationship path from HTML successfully.")
        return result_str

    except Exception as e:
        logger.error(
            f"Error processing relationship HTML with BeautifulSoup: {e}", exc_info=True
        )
        error_context = (
            f" near HTML: {html_content_decoded[:200]}..."
            if html_content_decoded
            else ""  # pyright: ignore[reportUnboundVariable]
        )
        return f"(Error parsing relationship HTML: {e}{error_context})"


# End of format_api_relationship_path


def print_group(label: str, items: List[Dict]):
    print(f"\n{label}:")
    if items:
        formatter = format_name
        for item in items:
            name_to_format = item.get("name") if isinstance(item, dict) else None
            print(f"  - {formatter(name_to_format)}")
        # End of for
    else:
        print("  (None found)")
    # End of if/else


# End of print_group


def _get_api_timeout(default: int = 60) -> int:
    timeout_value = default
    if selenium_config and hasattr(selenium_config, "API_TIMEOUT"):
        config_timeout = getattr(selenium_config, "API_TIMEOUT")
        if isinstance(config_timeout, (int, float)) and config_timeout > 0:
            timeout_value = int(config_timeout)
        else:
            logger.warning(
                f"Invalid API_TIMEOUT value in config ({config_timeout}), using default {default}s."
            )
        # End of if/else
    # End of if
    return timeout_value


# End of _get_api_timeout


def _get_owner_referer(session_manager: "SessionManager", base_url: str) -> str:
    owner_profile_id = getattr(session_manager, "my_profile_id", None)
    owner_tree_id = getattr(session_manager, "my_tree_id", None)
    if owner_profile_id and owner_tree_id:
        referer_path = (
            f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts"
        )
        referer = urljoin(base_url.rstrip("/") + "/", referer_path.lstrip("/"))
        logger.debug(f"Using owner facts page as referer: {referer}")
        return referer
    else:
        logger.warning(
            "Owner profile/tree ID missing in session. Using base URL as referer."
        )
        return base_url.rstrip("/") + "/"
    # End of if/else


# End of _get_owner_referer


def call_suggest_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    owner_profile_id: Optional[str],
    base_url: str,
    search_criteria: Dict[str, Any],
    timeouts: Optional[List[int]] = None,
) -> Optional[List[Dict]]:
    if not callable(_api_req):
        logger.critical(
            "Suggest API call failed: _api_req function unavailable (Import Failed?)."
        )
        raise ImportError("_api_req function not available from utils")
    # End of if
    if not isinstance(session_manager, SessionManager):
        logger.error("Suggest API call failed: Invalid SessionManager passed.")
        return None
    # End of if
    if not owner_tree_id:
        logger.error("Suggest API call failed: owner_tree_id is required.")
        return None
    # End of if

    api_description = "Suggest API"
    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")
    suggest_params_list = ["isHideVeiledRecords=false"]
    if first_name_raw:
        suggest_params_list.append(f"partialFirstName={quote(first_name_raw)}")
    # End of if
    if surname_raw:
        suggest_params_list.append(f"partialLastName={quote(surname_raw)}")
    # End of if
    if birth_year:
        suggest_params_list.append(f"birthYear={birth_year}")
    # End of if
    suggest_params = "&".join(suggest_params_list)
    suggest_url = f"{base_url.rstrip('/')}/api/person-picker/suggest/{owner_tree_id}?{suggest_params}"
    owner_facts_referer = _get_owner_referer(session_manager, base_url)
    timeouts_used = timeouts if timeouts else [20, 30, 60]
    max_attempts = len(timeouts_used)
    logger.info(f"Attempting {api_description} search: {suggest_url}")

    suggest_response = None
    for attempt, timeout in enumerate(timeouts_used, 1):
        logger.debug(
            f"{api_description} attempt {attempt}/{max_attempts} with timeout {timeout}s"
        )
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
                    f"{api_description} call successful via _api_req (attempt {attempt}/{max_attempts}), found {len(suggest_response)} results."
                )
                return suggest_response
            elif suggest_response is None:
                logger.warning(
                    f"{api_description} call using _api_req returned None on attempt {attempt}/{max_attempts}."
                )
            else:
                logger.error(
                    f"{api_description} call using _api_req returned unexpected type: {type(suggest_response)}"
                )
                logger.debug(
                    f"Unexpected Response Content: {str(suggest_response)[:500]}"
                )
                suggest_response = None
                break
            # End of if/elif/else
        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} _api_req call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
        except Exception as api_err:
            logger.error(
                f"{api_description} _api_req call failed on attempt {attempt}/{max_attempts}: {api_err}",
                exc_info=True,
            )
            suggest_response = None
            break
        # End of try/except
    # End of for

    if suggest_response is None:
        logger.warning(
            f"{api_description} failed via _api_req. Attempting direct requests fallback."
        )
        direct_response_obj = None
        try:
            cookies = {}
            if session_manager._requests_session:
                cookies = session_manager._requests_session.cookies.get_dict()
            # End of if
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
            logger.debug(f"Direct request headers: {direct_headers}")
            logger.debug(f"Direct request cookies: {list(cookies.keys())}")
            direct_timeout = _get_api_timeout(30)
            direct_response_obj = requests.get(
                suggest_url,
                headers=direct_headers,
                cookies=cookies,
                timeout=direct_timeout,
            )
            if direct_response_obj.status_code == 200:
                direct_data = direct_response_obj.json()
                if isinstance(direct_data, list):
                    logger.info(
                        f"Direct request fallback successful! Found {len(direct_data)} results."
                    )
                    return direct_data
                else:
                    logger.warning(
                        f"Direct request succeeded (200 OK) but returned non-list data: {type(direct_data)}"
                    )
                    logger.debug(f"Direct Response content: {str(direct_data)[:500]}")
                # End of if/else
            else:
                logger.warning(
                    f"Direct request fallback failed: Status {direct_response_obj.status_code}"
                )
                logger.debug(
                    f"Direct Response content: {direct_response_obj.text[:500]}"
                )
            # End of if/else
        except requests.exceptions.Timeout:
            logger.error(
                f"Direct request fallback timed out after {direct_timeout} seconds"
            )
        except json.JSONDecodeError as json_err:
            logger.error(f"Direct request fallback failed to decode JSON: {json_err}")
            if direct_response_obj:
                logger.debug(
                    f"Direct Response content: {direct_response_obj.text[:500]}"
                )
            # End of if
        except Exception as direct_err:
            logger.error(
                f"Direct request fallback failed with error: {direct_err}",
                exc_info=True,
            )
        # End of try/except
    # End of if

    logger.error(f"{api_description} failed after all attempts and fallback.")
    return None


# End of call_suggest_api


def call_facts_user_api(
    session_manager: "SessionManager",
    owner_profile_id: str,
    api_person_id: str,
    api_tree_id: str,
    base_url: str,
    timeouts: Optional[List[int]] = None,
) -> Optional[Dict]:
    if not callable(_api_req):
        logger.critical(
            "Facts API call failed: _api_req function unavailable (Import Failed?)."
        )
        raise ImportError("_api_req function not available from utils")
    # End of if
    if not isinstance(session_manager, SessionManager):
        logger.error("Facts API call failed: Invalid SessionManager passed.")
        return None
    # End of if
    if not all([owner_profile_id, api_person_id, api_tree_id]):
        logger.error(
            "Facts API call failed: owner_profile_id, api_person_id, and api_tree_id are required."
        )
        return None
    # End of if

    api_description = "Person Facts User API"
    facts_api_url = f"{base_url.rstrip('/')}/family-tree/person/facts/user/{owner_profile_id.lower()}/tree/{api_tree_id.lower()}/person/{api_person_id.lower()}"
    facts_referer = _get_owner_referer(session_manager, base_url)
    facts_data_raw = None
    direct_timeout = _get_api_timeout(30)
    fallback_timeouts = timeouts if timeouts else [30, 45, 60]
    logger.info(f"Attempting {api_description} via direct request: {facts_api_url}")

    direct_response_obj = None
    try:
        cookies = {}
        if session_manager._requests_session:
            cookies = session_manager._requests_session.cookies.get_dict()
        # End of if
        direct_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": facts_referer,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Content-Type": "application/json",
            "DNT": "1",
            "Connection": "keep-alive",
        }
        logger.debug(f"Direct facts request headers: {direct_headers}")
        logger.debug(f"Direct facts request cookies: {list(cookies.keys())}")
        direct_response_obj = requests.get(
            facts_api_url,
            headers=direct_headers,
            cookies=cookies,
            timeout=direct_timeout,
        )
        if direct_response_obj.status_code == 200:
            facts_data_raw = direct_response_obj.json()
            if not isinstance(facts_data_raw, dict):
                logger.warning(
                    f"Direct facts request OK (200) but returned non-dict data: {type(facts_data_raw)}"
                )
                logger.debug(f"Response content: {direct_response_obj.text[:500]}")
                facts_data_raw = None
            else:
                logger.info(f"{api_description} call successful via direct request.")
            # End of if/else
        else:
            logger.warning(
                f"Direct facts request failed: Status {direct_response_obj.status_code}"
            )
            logger.debug(f"Response content: {direct_response_obj.text[:500]}")
            facts_data_raw = None
        # End of if/else
    except requests.exceptions.Timeout:
        logger.error(f"Direct facts request timed out after {direct_timeout} seconds")
        facts_data_raw = None
    except json.JSONDecodeError as json_err:
        logger.error(f"Direct facts request failed to decode JSON: {json_err}")
        if direct_response_obj:
            logger.debug(f"Response content: {direct_response_obj.text[:500]}")
        # End of if
        facts_data_raw = None
    except Exception as direct_err:
        logger.error(f"Direct facts request failed: {direct_err}", exc_info=True)
        facts_data_raw = None
    # End of try/except

    if facts_data_raw is None:
        logger.warning(
            f"{api_description} direct request failed. Trying _api_req fallback."
        )
        max_attempts = len(fallback_timeouts)
        for attempt, timeout in enumerate(fallback_timeouts, 1):
            logger.debug(
                f"{api_description} _api_req attempt {attempt}/{max_attempts} with timeout {timeout}s"
            )
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
                elif api_response is None:
                    logger.warning(
                        f"{api_description} _api_req returned None (attempt {attempt}/{max_attempts})."
                    )
                else:
                    logger.warning(
                        f"{api_description} _api_req returned unexpected type: {type(api_response)}"
                    )
                    logger.debug(
                        f"Unexpected Response Value: {str(api_response)[:500]}"
                    )
                # End of if/elif/else
            except requests.exceptions.Timeout:
                logger.warning(
                    f"{api_description} _api_req call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
                )
            except Exception as api_req_err:
                logger.error(
                    f"{api_description} call using _api_req failed on attempt {attempt}/{max_attempts}: {api_req_err}",
                    exc_info=True,
                )
                facts_data_raw = None
                break
            # End of try/except
        # End of for
    # End of if

    if not isinstance(facts_data_raw, dict):
        logger.error(
            f"Failed to fetch valid {api_description} data after all attempts."
        )
        return None
    # End of if
    person_research_data = facts_data_raw.get("data", {}).get("personResearch")
    if not isinstance(person_research_data, dict) or not person_research_data:
        logger.error(
            f"{api_description} response received, but missing 'data.personResearch' dictionary."
        )
        logger.debug(f"Full raw response keys: {list(facts_data_raw.keys())}")
        if "data" in facts_data_raw and isinstance(facts_data_raw["data"], dict):
            logger.debug(f"'data' sub-keys: {list(facts_data_raw['data'].keys())}")
        else:
            logger.debug(f"'data' key missing or not a dict in response.")
        # End of if/else
        return None
    # End of if

    logger.info(
        f"Successfully fetched and extracted 'personResearch' data for PersonID {api_person_id}."
    )
    return person_research_data


# End of call_facts_user_api


def call_getladder_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    target_person_id: str,
    base_url: str,
    timeout: Optional[int] = None,
) -> Optional[str]:
    if not callable(_api_req):
        logger.critical(
            "GetLadder API call failed: _api_req function unavailable (Import Failed?)."
        )
        raise ImportError("_api_req function not available from utils")
    # End of if
    if not isinstance(session_manager, SessionManager):
        logger.error("GetLadder API call failed: Invalid SessionManager passed.")
        return None
    # End of if
    if not all([owner_tree_id, target_person_id]):
        logger.error(
            "GetLadder API call failed: owner_tree_id and target_person_id are required."
        )
        return None
    # End of if

    api_description = "Get Tree Ladder API"
    ladder_api_url_base = f"{base_url.rstrip('/')}/family-tree/person/tree/{owner_tree_id}/person/{target_person_id}/getladder"
    query_params = urlencode({"callback": "no"})
    ladder_api_url = f"{ladder_api_url_base}?{query_params}"
    ladder_referer_path = (
        f"/family-tree/person/tree/{owner_tree_id}/person/{target_person_id}/facts"
    )
    ladder_referer = urljoin(
        base_url.rstrip("/") + "/", ladder_referer_path.lstrip("/")
    )
    api_timeout_val = timeout if timeout else _get_api_timeout(20)
    logger.info(f"Attempting {api_description} call: {ladder_api_url}")

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
            timeout=api_timeout_val,
        )
        if isinstance(relationship_data, str) and len(relationship_data) > 10:
            logger.info(f"{api_description} call successful, received string response.")
            return relationship_data
        elif isinstance(relationship_data, str):
            logger.warning(
                f"{api_description} call returned a very short string: '{relationship_data}'"
            )
            return None
        else:
            logger.warning(
                f"{api_description} call returned non-string or None: {type(relationship_data)}"
            )
            return None
        # End of if/elif/else
    except requests.exceptions.Timeout:
        logger.error(f"{api_description} call timed out after {api_timeout_val}s.")
        return None
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        return None
    # End of try/except


# End of call_getladder_api


def call_treesui_list_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    owner_profile_id: Optional[str],
    base_url: str,
    search_criteria: Dict[str, Any],
    timeouts: Optional[List[int]] = None,
) -> Optional[List[Dict]]:
    if not callable(_api_req):
        logger.critical(
            "TreesUI List API call failed: _api_req function unavailable (Import Failed?)."
        )
        raise ImportError("_api_req function not available from utils")
    # End of if
    if not isinstance(session_manager, SessionManager):
        logger.error("TreesUI List API call failed: Invalid SessionManager passed.")
        return None
    # End of if
    if not owner_tree_id:
        logger.error("TreesUI List API call failed: owner_tree_id is required.")
        return None
    # End of if

    api_description = "TreesUI List API (Alternative Search)"
    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")
    if not birth_year:
        logger.warning(
            "Cannot call TreesUI List API: 'birth_year' is missing in search criteria."
        )
        return None
    # End of if
    treesui_params_list = ["limit=100", "fields=NAMES,BIRTH_DEATH"]
    if first_name_raw:
        treesui_params_list.append(f"fn={quote(first_name_raw)}")
    # End of if
    if surname_raw:
        treesui_params_list.append(f"ln={quote(surname_raw)}")
    # End of if
    treesui_params_list.append(f"by={birth_year}")
    treesui_params = "&".join(treesui_params_list)
    treesui_url = f"{base_url.rstrip('/')}/api/treesui-list/trees/{owner_tree_id}/persons?{treesui_params}"
    owner_facts_referer = _get_owner_referer(session_manager, base_url)
    timeouts_used = timeouts if timeouts else [15, 25, 35]
    max_attempts = len(timeouts_used)
    logger.info(f"Attempting {api_description} search using _api_req: {treesui_url}")

    treesui_response = None
    for attempt, timeout in enumerate(timeouts_used, 1):
        logger.debug(
            f"{api_description} attempt {attempt}/{max_attempts} with timeout {timeout}s"
        )
        try:
            custom_headers = {
                "Accept": "application/json",
                "Referer": owner_facts_referer,
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
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
                use_csrf_token=False,
            )
            if isinstance(treesui_response, list):
                logger.info(
                    f"{api_description} call successful via _api_req (attempt {attempt}/{max_attempts}), found {len(treesui_response)} results."
                )
                return treesui_response
            elif treesui_response is None:
                logger.warning(
                    f"{api_description} _api_req returned None (attempt {attempt}/{max_attempts})."
                )
            else:
                logger.error(
                    f"{api_description} returned unexpected format via _api_req: {type(treesui_response)}"
                )
                logger.debug(f"Unexpected Response: {str(treesui_response)[:500]}")
                return None
            # End of if/elif/else
        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} _api_req call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
        except Exception as treesui_err:
            logger.error(
                f"{api_description} _api_req call failed on attempt {attempt}/{max_attempts}: {treesui_err}",
                exc_info=True,
            )
            treesui_response = None
            break
        # End of try/except
    # End of for

    logger.error(f"{api_description} failed after all attempts.")
    return None


# End of call_treesui_list_api


def call_send_message_api(
    session_manager: "SessionManager",
    person: "Person",
    message_text: str,
    existing_conv_id: Optional[str],
    log_prefix: str,
) -> Tuple[str, Optional[str]]:
    if not session_manager or not session_manager.my_profile_id:
        logger.error(
            f"{log_prefix}: Cannot send message - SessionManager or own profile ID missing."
        )
        return SEND_ERROR_MISSING_OWN_ID, None
    # End of if

    if not isinstance(person, Person) or not getattr(person, "profile_id", None):
        logger.error(
            f"{log_prefix}: Cannot send message - Invalid Person object (Type: {type(person)}) or missing profile ID."
        )
        return SEND_ERROR_INVALID_RECIPIENT, None
    # End of if
    if not isinstance(message_text, str) or not message_text.strip():
        logger.error(
            f"{log_prefix}: Cannot send message - Message text is empty or invalid."
        )
        return SEND_ERROR_API_PREP_FAILED, None
    # End of if

    app_mode = getattr(config_instance, "APP_MODE", "unknown")
    if app_mode == "dry_run":
        message_status = SEND_SUCCESS_DRY_RUN
        effective_conv_id = existing_conv_id or f"dryrun_{uuid.uuid4()}"
        logger.info(
            f"{log_prefix}: Dry Run - Simulated message send to {getattr(person, 'username', None) or getattr(person, 'profile_id', 'Unknown') }."
        )
        return message_status, effective_conv_id
    elif app_mode not in ["production", "testing"]:
        logger.error(
            f"{log_prefix}: Logic Error - Unexpected APP_MODE '{app_mode}' reached send logic."
        )
        return SEND_ERROR_INTERNAL_MODE, None
    # End of if/elif

    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()
    MY_PROFILE_ID_UPPER = session_manager.my_profile_id.upper()
    recipient_profile_id_upper = getattr(person, "profile_id", "").upper()

    is_initial = not existing_conv_id
    send_api_url: str = ""
    payload: Dict[str, Any] = {}
    send_api_desc: str = ""
    api_headers: Dict[str, Any] = {}

    try:
        base_url_cfg = getattr(config_instance, "BASE_URL", "https://www.ancestry.com")
        if is_initial:
            send_api_url = urljoin(
                base_url_cfg.rstrip("/") + "/", API_PATH_SEND_MESSAGE_NEW
            )
            send_api_desc = "Create Conversation API"
            payload = {
                "content": message_text,
                "author": MY_PROFILE_ID_LOWER,
                "index": 0,
                "created": 0,
                "conversation_members": [
                    {
                        "user_id": recipient_profile_id_upper.lower(),
                        "family_circles": [],
                    },
                    {"user_id": MY_PROFILE_ID_LOWER},
                ],
            }
        elif existing_conv_id:
            formatted_path = API_PATH_SEND_MESSAGE_EXISTING.format(
                conv_id=existing_conv_id
            )
            send_api_url = urljoin(base_url_cfg.rstrip("/") + "/", formatted_path)
            send_api_desc = "Send Message API (Existing Conv)"
            payload = {
                "content": message_text,
                "author": MY_PROFILE_ID_LOWER,
            }
        else:
            logger.error(
                f"{log_prefix}: Logic Error - Cannot determine API URL/payload (existing_conv_id issue?)."
            )
            return SEND_ERROR_API_PREP_FAILED, None
        # End of if/elif/else

        ctx_headers = getattr(config_instance, "API_CONTEXTUAL_HEADERS", {}).get(
            send_api_desc, {}
        )
        api_headers = ctx_headers.copy()

    except Exception as prep_err:
        logger.error(
            f"{log_prefix}: Error preparing API request data: {prep_err}", exc_info=True
        )
        return SEND_ERROR_API_PREP_FAILED, None
    # End of try/except

    api_response = _api_req(
        url=send_api_url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="POST",
        json_data=payload,
        use_csrf_token=False,
        headers=api_headers,
        api_description=send_api_desc,
    )

    message_status = SEND_ERROR_UNKNOWN
    new_conversation_id_from_api: Optional[str] = None
    post_ok = False

    if api_response is None:
        message_status = SEND_ERROR_POST_FAILED
        logger.error(
            f"{log_prefix}: API POST ({send_api_desc}) failed (No response/Retries exhausted)."
        )
    elif isinstance(api_response, requests.Response):
        message_status = f"send_error (http_{api_response.status_code})"
        logger.error(
            f"{log_prefix}: API POST ({send_api_desc}) failed with status {api_response.status_code}."
        )
        try:
            logger.debug(f"Error response body: {api_response.text[:500]}")
        except Exception:
            pass
        # End of try/except
    elif isinstance(api_response, dict):
        try:
            if is_initial:
                api_conv_id = str(api_response.get(KEY_CONVERSATION_ID, ""))
                msg_details = api_response.get(KEY_MESSAGE, {})
                api_author = (
                    str(msg_details.get(KEY_AUTHOR, "")).upper()
                    if isinstance(msg_details, dict)
                    else None
                )

                if api_conv_id and api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = api_conv_id
                else:
                    logger.error(
                        f"{log_prefix}: API initial response format invalid (ConvID: '{api_conv_id}', Author: '{api_author}', Expected Author: '{MY_PROFILE_ID_UPPER}')."
                    )
                    logger.debug(f"API Response: {api_response}")
                    message_status = SEND_ERROR_VALIDATION_FAILED
                # End of if/else
            else:
                api_author = str(api_response.get(KEY_AUTHOR, "")).upper()
                if api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = existing_conv_id
                else:
                    logger.error(
                        f"{log_prefix}: API follow-up author validation failed (Author: '{api_author}', Expected Author: '{MY_PROFILE_ID_UPPER}')."
                    )
                    logger.debug(f"API Response: {api_response}")
                    message_status = SEND_ERROR_VALIDATION_FAILED
                # End of if/else
            # End of if/else

            if post_ok:
                message_status = SEND_SUCCESS_DELIVERED
                logger.info(
                    f"{log_prefix}: Message send to {getattr(person, 'username', None) or getattr(person, 'profile_id', 'Unknown')} successful (ConvID: {new_conversation_id_from_api})."
                )
            # End of if

        except Exception as parse_err:
            logger.error(
                f"{log_prefix}: Error parsing successful API response ({send_api_desc}): {parse_err}",
                exc_info=True,
            )
            logger.debug(f"API Response received: {api_response}")
            message_status = SEND_ERROR_UNEXPECTED_FORMAT
        # End of try/except
    else:
        logger.error(
            f"{log_prefix}: API call ({send_api_desc}) unexpected success format. Type:{type(api_response)}, Resp:{str(api_response)[:200]}"
        )
        message_status = SEND_ERROR_UNEXPECTED_FORMAT
    # End of if/elif/else

    if not post_ok and message_status == SEND_ERROR_UNKNOWN:
        message_status = SEND_ERROR_VALIDATION_FAILED
        logger.warning(
            f"{log_prefix}: Message send attempt concluded with status: {message_status}"
        )
    # End of if

    return message_status, new_conversation_id_from_api


# End of call_send_message_api


def call_profile_details_api(
    session_manager: "SessionManager", profile_id: str
) -> Optional[Dict[str, Any]]:
    if not profile_id or not isinstance(profile_id, str):
        logger.warning("call_profile_details_api: Profile ID missing or invalid.")
        return None
    # End of if
    if not session_manager or not session_manager.my_profile_id:
        logger.error(
            "call_profile_details_api: SessionManager or own profile ID missing."
        )
        return None
    # End of if

    if not session_manager.is_sess_valid():
        logger.error(
            f"call_profile_details_api: Session invalid for Profile ID {profile_id}."
        )
        return None
    # End of if

    api_description = "Profile Details API (Single)"
    base_url_cfg = getattr(config_instance, "BASE_URL", "https://www.ancestry.com")
    profile_url = urljoin(
        base_url_cfg,
        f"{API_PATH_PROFILE_DETAILS}?userId={profile_id.upper()}",
    )
    referer_url = urljoin(base_url_cfg, "/messaging/")

    logger.debug(
        f"Fetching profile details ({api_description}) for Profile ID {profile_id}..."
    )

    try:
        profile_response = _api_req(
            url=profile_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers={},
            use_csrf_token=False,
            api_description=api_description,
            referer_url=referer_url,
        )

        if profile_response and isinstance(profile_response, dict):
            logger.debug(f"Successfully fetched profile details for {profile_id}.")
            result_data: Dict[str, Any] = {
                "first_name": None,
                "last_logged_in_dt": None,
                "contactable": False,
            }

            first_name_raw = profile_response.get(KEY_FIRST_NAME)
            if first_name_raw and isinstance(first_name_raw, str):
                result_data["first_name"] = format_name(first_name_raw)
            else:
                display_name_raw = profile_response.get(KEY_DISPLAY_NAME_APIUTILS)
                if display_name_raw and isinstance(display_name_raw, str):
                    formatted_dn = format_name(display_name_raw)
                    result_data["first_name"] = (
                        formatted_dn.split()[0]
                        if formatted_dn != "Valued Relative"
                        else None
                    )
                else:
                    logger.warning(
                        f"Could not extract FirstName or DisplayName for profile {profile_id}"
                    )
                # End of if/else
            # End of if/else

            contactable_val = profile_response.get(KEY_IS_CONTACTABLE)
            result_data["contactable"] = (
                bool(contactable_val) if contactable_val is not None else False
            )

            last_login_str = profile_response.get(KEY_LAST_LOGIN_DATE)
            if last_login_str and isinstance(last_login_str, str):
                try:
                    if last_login_str.endswith("Z"):
                        dt_aware = datetime.fromisoformat(
                            last_login_str.replace("Z", "+00:00")
                        )
                    elif "+" in last_login_str or "-" in last_login_str[10:]:
                        dt_aware = datetime.fromisoformat(last_login_str)
                    else:
                        dt_naive = datetime.fromisoformat(last_login_str)
                        dt_aware = dt_naive.replace(tzinfo=timezone.utc)
                    # End of if/elif/else
                    result_data["last_logged_in_dt"] = dt_aware.astimezone(timezone.utc)
                except (ValueError, TypeError) as date_parse_err:
                    logger.warning(
                        f"Could not parse LastLoginDate '{last_login_str}' for {profile_id}: {date_parse_err}"
                    )
                # End of try/except
            else:
                logger.debug(
                    f"LastLoginDate missing or invalid for profile {profile_id}"
                )
            # End of if/else
            return result_data
        elif isinstance(profile_response, requests.Response):
            logger.warning(
                f"Failed profile details fetch for {profile_id}. Status: {profile_response.status_code}."
            )
            return None
        elif profile_response is None:
            logger.warning(
                f"Failed profile details fetch for {profile_id} (_api_req returned None)."
            )
            return None
        else:
            logger.warning(
                f"Failed profile details fetch for {profile_id} (Invalid response type: {type(profile_response)})."
            )
            return None
        # End of if/elif/else
    except requests.exceptions.RequestException as req_e:
        logger.error(
            f"RequestException fetching profile details for {profile_id}: {req_e}",
            exc_info=False,
        )
        return None
    # End of try/except
    except Exception as e:
        logger.error(
            f"Unexpected error fetching profile details for {profile_id}: {e}",
            exc_info=True,
        )
        return None
    # End of try/except


# End of call_profile_details_api


def call_header_trees_api_for_tree_id(
    session_manager: "SessionManager", tree_name_config: str
) -> Optional[str]:
    if not tree_name_config:
        logger.debug("TREE_NAME not configured, skipping tree ID retrieval.")
        return None
    # End of if
    if not session_manager.is_sess_valid():
        logger.error("call_header_trees_api_for_tree_id: Session invalid.")
        return None
    # End of if
    if not callable(_api_req):
        logger.critical("call_header_trees_api_for_tree_id: _api_req is not callable!")
        raise ImportError("_api_req function not available from utils")
    # End of if

    base_url_cfg = getattr(config_instance, "BASE_URL", "https://www.ancestry.com")
    alternative_api_path = "api/navheaderdata/v1/header/data/trees"
    url = urljoin(base_url_cfg.rstrip("/") + "/", alternative_api_path)
    api_description = "Header Trees API (Nav Data)"
    referer_url = urljoin(base_url_cfg.rstrip("/") + "/", "family-tree/trees")

    logger.debug(
        f"Attempting to fetch tree ID for TREE_NAME='{tree_name_config}' via {api_description} ({alternative_api_path}). Referer: {referer_url}"
    )

    custom_headers = {
        "Accept": "application/json",
        "Accept-Language": "en-GB,en;q=0.9",
        "X-Requested-With": "XMLHttpRequest",
    }

    try:
        response_data = _api_req(
            url=url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers=custom_headers,
            use_csrf_token=False,
            api_description=api_description,
            referer_url=referer_url,
        )

        if (
            response_data
            and isinstance(response_data, dict)
            and KEY_MENUITEMS in response_data
            and isinstance(response_data[KEY_MENUITEMS], list)
        ):
            for item in response_data[KEY_MENUITEMS]:
                if isinstance(item, dict) and item.get(KEY_TEXT) == tree_name_config:
                    tree_url = item.get(KEY_URL)
                    if tree_url and isinstance(tree_url, str):
                        match = re.search(r"/tree/(\d+)", tree_url)
                        if match:
                            my_tree_id_val = match.group(1)
                            logger.debug(
                                f"Found tree ID '{my_tree_id_val}' for tree '{tree_name_config}'."
                            )
                            return my_tree_id_val
                        else:
                            logger.warning(
                                f"Found tree '{tree_name_config}', but URL format unexpected: {tree_url}"
                            )
                        # End of if/else
                    else:
                        logger.warning(
                            f"Found tree '{tree_name_config}', but '{KEY_URL}' key missing or invalid."
                        )
                    # End of if/else
                    break
                # End of if
            # End of for
            logger.warning(
                f"Could not find TREE_NAME '{tree_name_config}' in {api_description} response."
            )
            return None
        elif response_data is None:
            logger.warning(f"{api_description} call failed (_api_req returned None).")
            return None
        else:
            status = "N/A"
            if isinstance(response_data, requests.Response):
                status = response_data.status_code
            # End of if
            logger.warning(
                f"Unexpected response format from {api_description} (Type: {type(response_data)}, Status: {status})."
            )
            logger.debug(f"Full {api_description} response data: {response_data}")
            return None
        # End of if/elif/else
    except Exception as e:
        logger.error(f"Error during {api_description}: {e}", exc_info=True)
        return None
    # End of try/except


# End of call_header_trees_api_for_tree_id


def call_tree_owner_api(
    session_manager: "SessionManager", tree_id: str
) -> Optional[str]:
    if not tree_id:
        logger.warning("Cannot get tree owner: tree_id is missing.")
        return None
    # End of if
    if not isinstance(tree_id, str):
        logger.warning(
            f"Invalid tree_id type provided: {type(tree_id)}. Expected string."
        )
        return None
    # End of if
    if not session_manager.is_sess_valid():
        logger.error("call_tree_owner_api: Session invalid.")
        return None
    # End of if
    if not callable(_api_req):
        logger.critical("call_tree_owner_api: _api_req is not callable!")
        raise ImportError("_api_req function not available from utils")
    # End of if

    base_url_cfg = getattr(config_instance, "BASE_URL", "https://www.ancestry.com")
    url = urljoin(base_url_cfg, f"{API_PATH_TREE_OWNER_INFO}?tree_id={tree_id}")
    api_description = "Tree Owner Name API"
    logger.debug(
        f"Attempting to fetch tree owner name for tree ID: {tree_id} via {api_description}..."
    )

    try:
        response_data = _api_req(
            url=url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description=api_description,
        )

        if response_data and isinstance(response_data, dict):
            owner_data = response_data.get(KEY_OWNER)
            if owner_data and isinstance(owner_data, dict):
                display_name = owner_data.get(KEY_DISPLAY_NAME_APIUTILS)
                if display_name and isinstance(display_name, str):
                    logger.debug(
                        f"Found tree owner '{display_name}' for tree ID {tree_id}."
                    )
                    return display_name
                else:
                    logger.warning(
                        f"Could not find '{KEY_DISPLAY_NAME_APIUTILS}' in owner data for tree {tree_id}."
                    )
                # End of if/else
            else:
                logger.warning(
                    f"Could not find '{KEY_OWNER}' data in {api_description} response for tree {tree_id}."
                )
            # End of if/else
            logger.debug(f"Full {api_description} response data: {response_data}")
            return None
        elif response_data is None:
            logger.warning(f"{api_description} call failed (_api_req returned None).")
            return None
        else:
            status = "N/A"
            if isinstance(response_data, requests.Response):
                status = response_data.status_code
            # End of if
            logger.warning(
                f"{api_description} call returned unexpected data (Type: {type(response_data)}, Status: {status}) or None."
            )
            logger.debug(f"Response received: {response_data}")
            return None
        # End of if/elif/else
    except Exception as e:
        logger.error(
            f"Error during {api_description} for tree {tree_id}: {e}", exc_info=True
        )
        return None
    # End of try/except


# End of call_tree_owner_api


# --- Standalone Test Block ---
def _sc_run_test(
    test_name: str,
    test_func: Callable,
    test_results_list: List[Tuple[str, str, str]],
    logger_instance: logging.Logger,
    *args,
    **kwargs,
) -> Tuple[str, str, str]:
    logger_instance.debug(f"[ RUNNING SC ] {test_name}")
    status = "FAIL"
    message = kwargs.pop("message", "")
    expect_none = kwargs.pop("expected_none", False)
    expect_type = kwargs.pop("expected_type", None)
    expect_value = kwargs.pop("expected_value", None)
    expect_contains = kwargs.pop("expected_contains", None)
    expect_truthy = kwargs.pop("expected_truthy", False)
    test_func_kwargs = kwargs

    try:
        result = test_func(*args, **test_func_kwargs)
        passed = False
        explicit_skip = False

        if expect_none:
            passed = result is None
            if not passed:
                message = f"Expected None, got {type(result).__name__}"
            # End of if
        elif expect_type is not None:
            if result is None:
                passed = False
                message = f"Expected type {expect_type.__name__}, but function returned None (API/Parse issue?)"
                if not isinstance(result, str) or result != "Skipped":  # type: ignore
                    logger_instance.error(f"Test '{test_name}': {message}")
                # End of if
            elif isinstance(result, expect_type):
                passed = True
            else:
                passed = False
                message = (
                    f"Expected type {expect_type.__name__}, got {type(result).__name__}"
                )
            # End of if/elif/else
        elif expect_value is not None:
            passed = result == expect_value
            if not passed:
                message = f"Expected value '{str(expect_value)[:50]}...', got '{repr(result)[:100]}...'"
            # End of if
        elif expect_contains is not None:
            if isinstance(result, str):
                if isinstance(expect_contains, (list, tuple)):
                    missing = [sub for sub in expect_contains if sub not in result]
                    passed = not missing
                    if not passed:
                        message = f"Expected result to contain all of: {expect_contains}. Missing: {missing}"
                    # End of if
                elif isinstance(expect_contains, str):
                    passed = expect_contains in result
                    if not passed:
                        message = f"Expected result to contain '{expect_contains}', got '{repr(result)[:100]}...'"
                    # End of if
                else:
                    passed = False
                    message = (
                        f"Invalid type for expect_contains: {type(expect_contains)}"
                    )
                # End of if/elif/else
            else:
                passed = False
                message = f"Expected string result for contains check, got {type(result).__name__}"
            # End of if/else
        elif expect_truthy:
            passed = bool(result)
            if not passed:
                if result is None:
                    message = "Expected truthy value, but function returned None (Underlying call likely failed)"
                else:
                    message = f"Expected truthy value, got {repr(result)[:100]}"
                # End of if/else
            # End of if
        elif isinstance(result, str) and result == "Skipped":
            passed = False
            explicit_skip = True
            status = "SKIPPED"
        else:
            passed = result is True
            if not passed:
                message = (
                    f"Default check failed: Expected True, got {repr(result)[:100]}"
                )
            # End of if
        # End of if/elif chain

        if not explicit_skip:
            status = "PASS" if passed else "FAIL"
            if status == "FAIL" and not message:
                message = f"Test condition not met (Result: {repr(result)[:100]})"
            # End of if
        # End of if

    except Exception as e:
        status = "FAIL"
        message = f"EXCEPTION: {type(e).__name__}: {e}"
        logger_instance.error(
            f"Exception during self-check test '{test_name}': {message}\n{traceback.format_exc()}",
            exc_info=False,
        )
    # End of try/except

    log_level = (
        logging.INFO
        if status == "PASS"
        else (logging.WARNING if status == "SKIPPED" else logging.ERROR)
    )
    log_message = f"[ {status:<7} SC ] {test_name}{f': {message}' if message else ''}"
    logger_instance.log(log_level, log_message)

    test_results_list.append((test_name, status, message if status != "PASS" else ""))
    return (test_name, status, message)


# End of _sc_run_test


def _sc_print_summary(
    test_results_list: List[Tuple[str, str, str]],
    overall_status: bool,
    logger_instance: logging.Logger,
):
    print("\n--- api_utils.py Self-Check Summary ---")
    name_width = 55
    if test_results_list:
        try:
            name_width = max(len(name) for name, _, _ in test_results_list) + 2
            name_width = min(name_width, 70)
        except ValueError:
            pass
        # End of try/except
    # End of if
    status_width = 8
    header = f"{'Test Name':<{name_width}} | {'Status':<{status_width}} | {'Message / Details'}"
    print(header)
    print("-" * (len(header) + 5))

    final_fail_count = 0
    final_skip_count = 0
    final_pass_count = 0

    for name, status, message in test_results_list:
        print(
            f"{name:<{name_width}} | {status:<{status_width}} | {message if status != 'PASS' else ''}"
        )
        if status == "FAIL":
            final_fail_count += 1
        elif status == "SKIPPED":
            final_skip_count += 1
        elif status == "PASS":
            final_pass_count += 1
        # End of if/elif
    # End of for

    total_executed_tests = len(test_results_list)

    print("-" * (len(header) + 5))
    final_overall_status_from_tests = final_fail_count == 0
    final_overall_status = overall_status and final_overall_status_from_tests

    result_color = "\033[92m" if final_overall_status else "\033[91m"
    reset_color = "\033[0m"
    final_status_msg = (
        f"Result: {result_color}{'PASS' if final_overall_status else 'FAIL'}{reset_color} "
        f"({final_pass_count} passed, {final_fail_count} failed, {final_skip_count} skipped out of {total_executed_tests} executed tests)"
    )
    print(f"{final_status_msg}\n")
    logger_instance.log(
        logging.INFO if final_overall_status else logging.ERROR,
        f"api_utils self-check overall status: {'PASS' if final_overall_status else 'FAIL'}",
    )


# End of _sc_print_summary


def self_check() -> bool:
    logger_sc = logging.getLogger("api_utils.self_check")
    logger_sc.info("\n" + "=" * 30 + " api_utils.py Self-Check Starting " + "=" * 30)

    required_modules_ok = True
    config_instance_sc = config_instance
    selenium_config_sc = selenium_config

    if not BS4_AVAILABLE:
        logger_sc.warning(
            "BeautifulSoup (bs4) library not found. HTML parsing tests will be skipped."
        )
    # End of if

    test_results_sc: List[Tuple[str, str, str]] = []
    session_manager_sc: Optional["SessionManager"] = None
    overall_status = required_modules_ok

    def _sc_api_req_wrapper(
        url: str, description: str, expect_json: bool = True, **kwargs
    ) -> Any:
        nonlocal session_manager_sc, overall_status
        if not session_manager_sc:
            raise RuntimeError("SessionManager not initialized for SC")
        # End of if
        if not session_manager_sc.is_sess_valid():
            logger_sc.error(f"Session invalid before calling '{description}' (SC)")
            overall_status = False
            raise RuntimeError("Session not ready for API call")
        # End of if
        result = _api_req(
            url=url,
            driver=session_manager_sc.driver,
            session_manager=session_manager_sc,
            api_description=f"{description} (SC)",
            **kwargs,
        )
        if expect_json and isinstance(result, requests.Response):
            logger_sc.warning(
                f"[_sc wrapper] Expected JSON for '{description}', got Response object (Status: {result.status_code}). Returning None."
            )
            return None
        # End of if
        return result

    # End of _sc_api_req_wrapper

    def _sc_get_profile_details(profile_id: str) -> Optional[Dict]:
        nonlocal overall_status
        if not profile_id:
            return None
        # End of if
        api_desc = f"Get Profile Details ({profile_id})"
        url = urljoin(
            config_instance_sc.BASE_URL,
            f"{API_PATH_PROFILE_DETAILS}?userId={profile_id.upper()}",
        )
        timeout = _get_api_timeout(30)
        try:
            return _sc_api_req_wrapper(
                url, api_desc, expect_json=True, use_csrf_token=False, timeout=timeout
            )
        except RuntimeError as session_err:
            logger_sc.error(
                f"Failed to get profile details due to session issue: {session_err}"
            )
            overall_status = False
            return None
        # End of try/except
        except Exception as e:
            logger_sc.error(
                f"Unexpected error in _sc_get_profile_details: {e}", exc_info=True
            )
            overall_status = False
            return None
        # End of try/except

    # End of _sc_get_profile_details

    can_run_live_tests = overall_status
    target_profile_id_sc: Optional[str] = None
    target_person_id_for_ladder_sc: Optional[str] = None
    base_url_sc = "https://www.ancestry.com"

    target_profile_id_sc = getattr(config_instance_sc, "TESTING_PROFILE_ID", None)
    target_person_id_for_ladder_sc = getattr(
        config_instance_sc, "TESTING_PERSON_TREE_ID", None
    )
    base_url_sc = getattr(config_instance_sc, "BASE_URL", base_url_sc).rstrip("/")
    if not target_profile_id_sc:
        logger_sc.warning(
            "TESTING_PROFILE_ID not set in config. Some tests will be skipped."
        )
    # End of if
    if not target_person_id_for_ladder_sc:
        logger_sc.warning(
            "TESTING_PERSON_TREE_ID not set in config. Ladder/Facts tests will be skipped."
        )
    # End of if

    target_name_from_profile = "Unknown Target"
    target_name_for_ladder = "Unknown Ladder Target"

    logger_sc.info("--- Phase 0: Prerequisite & Static Function Checks ---")
    core_funcs = {
        "format_name": format_name,
        "ordinal_case": ordinal_case,
        "_parse_date": _parse_date,
        "_clean_display_date": _clean_display_date,
        "parse_ancestry_person_details": parse_ancestry_person_details,
        "format_api_relationship_path": format_api_relationship_path,
        "call_suggest_api": call_suggest_api,
        "call_facts_user_api": call_facts_user_api,
        "call_getladder_api": call_getladder_api,
        "call_treesui_list_api": call_treesui_list_api,
        "call_send_message_api": call_send_message_api,
        "call_profile_details_api": call_profile_details_api,
        "call_header_trees_api_for_tree_id": call_header_trees_api_for_tree_id,
        "call_tree_owner_api": call_tree_owner_api,
        "_api_req": _api_req,
    }
    for name, func in core_funcs.items():
        _, s0_f_stat, _ = _sc_run_test(
            f"Check Function '{name}' Callable",
            lambda f_param=func: callable(f_param),
            test_results_sc,
            logger_sc,
            expected_truthy=True,
        )
        if s0_f_stat != "PASS":
            overall_status = False
        # End of if
    # End of for
    _, s0_c_stat, _ = _sc_run_test(
        "Check Config Loaded (BASE_URL)",
        lambda: hasattr(config_instance_sc, "BASE_URL"),
        test_results_sc,
        logger_sc,
        expected_truthy=True,
    )
    if s0_c_stat != "PASS":
        overall_status = False
    # End of if

    logger_sc.info("--- Phase 0b: (Skipped - Mock Data Tests Removed) ---")

    target_owner_global_id = None
    if can_run_live_tests and overall_status:
        try:
            logger_sc.info("--- Phase 1: Session Setup & Login ---")
            session_manager_sc = SessionManager()
            _, s1_start_stat, _ = _sc_run_test(
                "SessionManager.start_sess()",
                session_manager_sc.start_sess,
                test_results_sc,
                logger_sc,
                action_name="SC Phase 1 Start",
                expected_truthy=True,
            )
            if s1_start_stat != "PASS":
                overall_status = False
                raise RuntimeError("start_sess failed")
            # End of if
            _, s1_ready_stat, _ = _sc_run_test(
                "SessionManager.ensure_session_ready()",
                session_manager_sc.ensure_session_ready,
                test_results_sc,
                logger_sc,
                action_name="SC Phase 1 Ready",
                expected_truthy=True,
            )
            if s1_ready_stat != "PASS":
                overall_status = False
                raise RuntimeError("ensure_session_ready failed")
            # End of if

            logger_sc.info("--- Phase 2: Get Target Info & Validate Config ---")
            target_tree_id = session_manager_sc.my_tree_id
            target_owner_name = session_manager_sc.tree_owner_name
            target_owner_profile_id = session_manager_sc.my_profile_id
            target_owner_global_id = session_manager_sc.my_uuid
            _, s2_tid_stat, _ = _sc_run_test(
                "Check Target Tree ID Found",
                lambda: bool(target_tree_id),
                test_results_sc,
                logger_sc,
                expected_truthy=True,
            )
            _, s2_owner_stat, _ = _sc_run_test(
                "Check Target Owner Name Found",
                lambda: bool(target_owner_name),
                test_results_sc,
                logger_sc,
                expected_truthy=True,
            )
            _, s2_profile_stat, _ = _sc_run_test(
                "Check Target Owner Profile ID Found",
                lambda: bool(target_owner_profile_id),
                test_results_sc,
                logger_sc,
                expected_truthy=True,
            )
            _, s2_uuid_stat, _ = _sc_run_test(
                "Check Target Owner Global ID (UUID) Found",
                lambda: bool(target_owner_global_id),
                test_results_sc,
                logger_sc,
                expected_truthy=True,
            )
            if not all(
                s == "PASS"
                for s in [s2_tid_stat, s2_owner_stat, s2_profile_stat, s2_uuid_stat]
            ):
                overall_status = False
                logger_sc.error(
                    "Essential IDs missing from session. Aborting most live tests."
                )
                raise RuntimeError("Essential IDs missing from session.")
            # End of if

            profile_response_details = None
            test_name_target_profile = "API Call: Get Target Profile Details (app-api via _sc_get_profile_details)"
            if target_profile_id_sc:
                api_call_lambda = lambda: _sc_get_profile_details(
                    cast(str, target_profile_id_sc)
                )
                _, s2_api_stat, s2_api_msg = _sc_run_test(
                    test_name_target_profile,
                    api_call_lambda,
                    test_results_sc,
                    logger_sc,
                    expected_type=dict,
                )
                if s2_api_stat == "PASS":
                    profile_response_details = api_call_lambda()
                    if profile_response_details:
                        parsed_api_details = parse_ancestry_person_details(
                            {}, profile_response_details
                        )
                        target_name_from_profile = parsed_api_details.get(
                            "name", "Unknown Target"
                        )
                        _, s2_name_stat, _ = _sc_run_test(
                            "Check Target Name Found in API Resp",
                            lambda: target_name_from_profile
                            not in ["Unknown", "Unknown Target"],
                            test_results_sc,
                            logger_sc,
                            expected_truthy=True,
                        )
                        if s2_name_stat == "FAIL":
                            overall_status = False
                        # End of if
                        if target_name_from_profile not in [
                            "Unknown",
                            "Unknown Target",
                        ]:
                            if target_person_id_for_ladder_sc == target_profile_id_sc:
                                target_name_for_ladder = target_name_from_profile
                            # End of if
                        # End of if
                    else:
                        overall_status = False
                        _sc_run_test(
                            "Check Target Name Found in API Resp",
                            lambda: "Skipped",
                            test_results_sc,
                            logger_sc,
                            message="API call passed but returned None",
                        )
                    # End of if/else profile_response_details
                elif s2_api_stat == "SKIPPED":
                    _sc_run_test(
                        "Check Target Name Found in API Resp",
                        lambda: "Skipped",
                        test_results_sc,
                        logger_sc,
                        message=s2_api_msg,
                    )
                else:
                    overall_status = False
                    _sc_run_test(
                        "Check Target Name Found in API Resp",
                        lambda: "Skipped",
                        test_results_sc,
                        logger_sc,
                        message=s2_api_msg,
                    )
                # End of if/elif/else s2_api_stat
            else:
                _sc_run_test(
                    test_name_target_profile,
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="TESTING_PROFILE_ID not set",
                )
                _sc_run_test(
                    "Check Target Name Found in API Resp",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="TESTING_PROFILE_ID not set",
                )
            # End of if target_profile_id_sc

            if (
                target_name_for_ladder == "Unknown Ladder Target"
                and target_person_id_for_ladder_sc
            ):
                logger_sc.warning(
                    f"Using default name '{target_name_for_ladder}' for ladder target name (TESTING_PERSON_TREE_ID may differ from TESTING_PROFILE_ID or name lookup failed)."
                )
            # End of if

            logger_sc.info(
                "--- Phase 3: Test parse_ancestry_person_details (Live & Static) ---"
            )
            test_name_parse = "Function Call: parse_ancestry_person_details()"
            if profile_response_details and isinstance(profile_response_details, dict):
                person_card_empty = {}
                try:
                    parse_lambda_facts = lambda: parse_ancestry_person_details(
                        person_card_empty, profile_response_details
                    )
                    _, s3_facts_stat, s3_facts_msg = _sc_run_test(
                        f"{test_name_parse} (with Live Facts)",
                        parse_lambda_facts,
                        test_results_sc,
                        logger_sc,
                        expected_type=dict,
                    )
                    if s3_facts_stat == "PASS":
                        parsed_details_facts = parse_lambda_facts()
                        if parsed_details_facts:
                            keys_ok_facts = all(
                                k in parsed_details_facts
                                for k in [
                                    "name",
                                    "person_id",
                                    "link",
                                    "birth_date",
                                    "death_date",
                                    "gender",
                                    "is_living",
                                ]
                            )
                            _, s3_keys_stat, _ = _sc_run_test(
                                "Validation: Parsed Details Keys (Live Facts)",
                                lambda: keys_ok_facts,
                                test_results_sc,
                                logger_sc,
                                expected_truthy=True,
                            )
                            if s3_keys_stat == "FAIL":
                                overall_status = False
                            # End of if
                            if target_name_from_profile not in [
                                "Unknown Target",
                                "Unknown",
                            ]:
                                _, s3_name_stat, _ = _sc_run_test(
                                    "Validation: Parsed Name Match (Live Facts)",
                                    lambda p=parsed_details_facts: p.get("name")
                                    == target_name_from_profile,
                                    test_results_sc,
                                    logger_sc,
                                    expected_truthy=True,
                                )
                                if s3_name_stat == "FAIL":
                                    overall_status = False
                                # End of if
                            else:
                                _sc_run_test(
                                    "Validation: Parsed Name Match (Live Facts)",
                                    lambda: "Skipped",
                                    test_results_sc,
                                    logger_sc,
                                    message="Target name unknown",
                                )
                            # End of if/else
                        else:
                            _sc_run_test(
                                f"{test_name_parse} (with Live Facts)",
                                lambda: False,
                                test_results_sc,
                                logger_sc,
                                message="Parser returned None/invalid",
                            )
                            overall_status = False
                        # End of if/else
                    elif s3_facts_stat == "SKIPPED":
                        _sc_run_test(
                            "Validation: Parsed Details Keys (Live Facts)",
                            lambda: "Skipped",
                            test_results_sc,
                            logger_sc,
                            message=s3_facts_msg,
                        )
                        _sc_run_test(
                            "Validation: Parsed Name Match (Live Facts)",
                            lambda: "Skipped",
                            test_results_sc,
                            logger_sc,
                            message=s3_facts_msg,
                        )
                    else:
                        overall_status = False
                        _sc_run_test(
                            "Validation: Parsed Details Keys (Live Facts)",
                            lambda: "Skipped",
                            test_results_sc,
                            logger_sc,
                            message=s3_facts_msg,
                        )
                        _sc_run_test(
                            "Validation: Parsed Name Match (Live Facts)",
                            lambda: "Skipped",
                            test_results_sc,
                            logger_sc,
                            message=s3_facts_msg,
                        )
                    # End of if/elif/else
                except Exception as parse_e:
                    _sc_run_test(
                        f"{test_name_parse} (with Live Facts)",
                        lambda: False,
                        test_results_sc,
                        logger_sc,
                        message=f"Exception: {parse_e}",
                    )
                    overall_status = False
                # End of try/except
            else:
                _sc_run_test(
                    f"{test_name_parse} (with Live Facts)",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="No live profile details",
                )
                _sc_run_test(
                    "Validation: Parsed Details Keys (Live Facts)",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="No live profile details",
                )
                _sc_run_test(
                    "Validation: Parsed Name Match (Live Facts)",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="No live profile details",
                )
            # End of if profile_response_details

            suggest_like_card = {
                "PersonId": "12345",
                "TreeId": "67890",
                "UserId": "ABC-DEF",
                "FullName": "Test Suggest Person",
                "GivenName": "Test",
                "Surname": "Suggest Person",
                "BirthYear": 1950,
                "BirthPlace": "SuggestBirth Town",
                "DeathYear": 2000,
                "DeathPlace": "SuggestDeath City",
                "Gender": "Female",
                "IsLiving": False,
            }
            try:
                parse_lambda_suggest = lambda: parse_ancestry_person_details(
                    suggest_like_card, None
                )
                _, s3_suggest_stat, s3_suggest_msg = _sc_run_test(
                    f"{test_name_parse} (Static Suggest Format)",
                    parse_lambda_suggest,
                    test_results_sc,
                    logger_sc,
                    expected_type=dict,
                )
                if s3_suggest_stat == "PASS":
                    parsed_details_suggest = parse_lambda_suggest()
                    if parsed_details_suggest:
                        vals_ok = (
                            parsed_details_suggest.get("name") == "Test Suggest Person"
                        )
                        _, s3_val_stat, _ = _sc_run_test(
                            "Validation: Parsed Details Values (Static Suggest)",
                            lambda: vals_ok,
                            test_results_sc,
                            logger_sc,
                            expected_truthy=True,
                        )
                        if s3_val_stat == "FAIL":
                            overall_status = False
                        # End of if
                    else:
                        _sc_run_test(
                            f"{test_name_parse} (Static Suggest Format)",
                            lambda: False,
                            test_results_sc,
                            logger_sc,
                            message="Parser returned None/invalid",
                        )
                        overall_status = False
                    # End of if/else
                elif s3_suggest_stat == "FAIL":
                    overall_status = False
                    _sc_run_test(
                        "Validation: Parsed Details Values (Static Suggest)",
                        lambda: "Skipped",
                        test_results_sc,
                        logger_sc,
                        message=s3_suggest_msg,
                    )
                else:
                    _sc_run_test(
                        "Validation: Parsed Details Values (Static Suggest)",
                        lambda: "Skipped",
                        test_results_sc,
                        logger_sc,
                        message=s3_suggest_msg,
                    )
                # End of if/elif/else
            except Exception as parse_e:
                _sc_run_test(
                    f"{test_name_parse} (Static Suggest Format)",
                    lambda: False,
                    test_results_sc,
                    logger_sc,
                    message=f"Exception: {parse_e}",
                )
                overall_status = False
            # End of try/except

            logger_sc.info("--- Phase 4: Test API Helpers (Live) ---")
            if (
                target_tree_id
                and target_owner_profile_id
                and callable(call_suggest_api)
            ):
                suggest_criteria = {
                    "first_name_raw": "John",
                    "surname_raw": "Smith",
                    "birth_year": 1900,
                }
                _, suggest_status, _ = _sc_run_test(
                    "API Helper: call_suggest_api",
                    lambda: call_suggest_api(
                        session_manager_sc,
                        target_tree_id,
                        target_owner_profile_id,
                        base_url_sc,
                        suggest_criteria,
                    ),
                    test_results_sc,
                    logger_sc,
                    expected_type=list,
                )
                if suggest_status == "FAIL":
                    overall_status = False
                # End of if
            else:
                _sc_run_test(
                    "API Helper: call_suggest_api",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="Missing tree/owner ID",
                )
            # End of if/else

            facts_person_id = target_person_id_for_ladder_sc
            if (
                target_tree_id
                and target_owner_profile_id
                and facts_person_id
                and callable(call_facts_user_api)
            ):
                _, facts_status, _ = _sc_run_test(
                    "API Helper: call_facts_user_api",
                    lambda: call_facts_user_api(
                        session_manager_sc,
                        target_owner_profile_id,
                        facts_person_id,
                        target_tree_id,
                        base_url_sc,
                    ),
                    test_results_sc,
                    logger_sc,
                    expected_type=dict,
                )
                if facts_status == "FAIL":
                    overall_status = False
                # End of if
            else:
                _sc_run_test(
                    "API Helper: call_facts_user_api",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="Missing TreeID, OwnerProfileID, or TESTING_PERSON_TREE_ID",
                )
            # End of if/else

            if (
                Person != type(None)
                and session_manager_sc
                and session_manager_sc.my_profile_id
            ):
                dummy_person = Person(
                    profile_id="DUMMY-RECIPIENT-ID", username="DummyRecipient"
                )
                original_app_mode = getattr(config_instance_sc, "APP_MODE", "unknown")
                setattr(config_instance_sc, "APP_MODE", "dry_run")
                _, send_msg_stat, _ = _sc_run_test(
                    "API Helper: call_send_message_api (dry_run)",
                    lambda: call_send_message_api(
                        session_manager_sc,
                        dummy_person,
                        "SC Test Message",
                        None,
                        "SC_SendMsg",
                    )[0]
                    == SEND_SUCCESS_DRY_RUN,
                    test_results_sc,
                    logger_sc,
                    expected_truthy=True,
                )
                if send_msg_stat == "FAIL":
                    overall_status = False
                # End of if
                setattr(config_instance_sc, "APP_MODE", original_app_mode)
            else:
                _sc_run_test(
                    "API Helper: call_send_message_api (dry_run)",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="Person unavailable or my_profile_id missing",
                )
            # End of if/else

            if (
                target_profile_id_sc
                and session_manager_sc
                and callable(call_profile_details_api)
            ):
                _, prof_details_stat, _ = _sc_run_test(
                    "API Helper: call_profile_details_api",
                    lambda: call_profile_details_api(
                        session_manager_sc, target_profile_id_sc
                    ),
                    test_results_sc,
                    logger_sc,
                    expected_type=dict,
                )
                if prof_details_stat == "FAIL":
                    overall_status = False
                # End of if
            else:
                _sc_run_test(
                    "API Helper: call_profile_details_api",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="TESTING_PROFILE_ID missing or prerequisites failed",
                )
            # End of if/else

            tree_name_cfg_sc = getattr(config_instance_sc, "TREE_NAME", None)
            if (
                tree_name_cfg_sc
                and session_manager_sc
                and callable(call_header_trees_api_for_tree_id)
            ):
                test_name_hdr_trees = "API Helper: call_header_trees_api_for_tree_id"
                _, hdr_trees_stat, _ = _sc_run_test(
                    test_name_hdr_trees,
                    lambda: call_header_trees_api_for_tree_id(
                        session_manager_sc, tree_name_cfg_sc
                    ),
                    test_results_sc,
                    logger_sc,
                    expected_type=str,
                )
                if hdr_trees_stat == "FAIL":
                    overall_status = False
                elif hdr_trees_stat == "PASS":
                    result = call_header_trees_api_for_tree_id(
                        session_manager_sc, tree_name_cfg_sc
                    )
                    if result is None:
                        logger_sc.error(
                            "Test 'call_header_trees_api_for_tree_id' PASSED type check but returned None when configured TREE_NAME exists."
                        )
                        for i_res, res_item in enumerate(test_results_sc):
                            if res_item[0] == test_name_hdr_trees:
                                test_results_sc[i_res] = (
                                    res_item[0],
                                    "FAIL",
                                    "Expected string tree ID, got None",
                                )
                                break
                            # End of if
                        # End of for
                        overall_status = False
                    # End of if
                # End of if/elif
            else:
                _sc_run_test(
                    "API Helper: call_header_trees_api_for_tree_id",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="TREE_NAME not configured or prerequisites failed",
                )
            # End of if/else

            tree_id_for_owner_test = (
                session_manager_sc.my_tree_id if session_manager_sc else None
            )
            if (
                tree_id_for_owner_test
                and session_manager_sc
                and callable(call_tree_owner_api)
            ):
                test_name_owner_api = "API Helper: call_tree_owner_api"
                _, tree_owner_stat, _ = _sc_run_test(
                    test_name_owner_api,
                    lambda: call_tree_owner_api(
                        session_manager_sc, tree_id_for_owner_test
                    ),
                    test_results_sc,
                    logger_sc,
                    expected_type=str,
                )
                if tree_owner_stat == "FAIL":
                    overall_status = False
                elif tree_owner_stat == "PASS":
                    result = call_tree_owner_api(
                        session_manager_sc, tree_id_for_owner_test
                    )
                    if result is None:
                        logger_sc.error(
                            "Test 'call_tree_owner_api' PASSED type check but returned None."
                        )
                        for i_res, res_item in enumerate(test_results_sc):
                            if res_item[0] == test_name_owner_api:
                                test_results_sc[i_res] = (
                                    res_item[0],
                                    "FAIL",
                                    "Expected string owner name, got None",
                                )
                                break
                            # End of if
                        # End of for
                        overall_status = False
                    # End of if
                # End of if/elif
            else:
                _sc_run_test(
                    "API Helper: call_tree_owner_api",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="Tree ID not available or prerequisites failed",
                )
            # End of if/else

            # --- Phase 5: Test Relationship Path Formatting ---
            logger_sc.info("--- Phase 5: Test Relationship Path Formatting ---")
            if BS4_AVAILABLE:
                target_rel_name = "Elizabeth 'Betty' Cruickshank"
                owner_rel_name_for_test = "Wayne Gordon Gault"

                def rel_path_test_lambda():
                    return format_api_relationship_path(
                        TEST_RELATIONSHIP_PATH_RAW_API_RESPONSE,
                        owner_name=owner_rel_name_for_test,
                        target_name=target_rel_name,
                    )

                # End of rel_path_test_lambda

                _, rel_path_status, rel_path_msg_detail = _sc_run_test(
                    "Format Relationship Path (HTML)",
                    rel_path_test_lambda,
                    test_results_sc,
                    logger_sc,
                    expected_value=EXPECTED_FORMATTED_PATH_STRING,
                )
                if rel_path_status == "FAIL":
                    overall_status = False
                    try:
                        actual_output = rel_path_test_lambda()
                        logger_sc.error(
                            f"Format Relationship Path (HTML) - Actual Output:\n{actual_output}"
                        )
                        logger_sc.error(
                            f"Format Relationship Path (HTML) - Expected Output:\n{EXPECTED_FORMATTED_PATH_STRING}"
                        )
                    except Exception as e_log:
                        logger_sc.error(
                            f"Error re-running lambda for logging failed test: {e_log}"
                        )
                    # End of try/except
                # End of if
            else:
                _sc_run_test(
                    "Format Relationship Path (HTML)",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="BeautifulSoup (bs4) not available.",
                )
            # End of if BS4_AVAILABLE

        except RuntimeError as rt_err:
            logger_sc.critical(f"RUNTIME ERROR during SC live tests: {rt_err}")
            _sc_run_test(
                "Self-Check Live Execution",
                lambda: False,
                test_results_sc,
                logger_sc,
                message=f"RUNTIME ERROR: {rt_err}",
            )
            overall_status = False
        except Exception as e:
            logger_sc.critical(
                "UNEXPECTED EXCEPTION during SC live tests", exc_info=True
            )
            _sc_run_test(
                "Self-Check Live Execution",
                lambda: False,
                test_results_sc,
                logger_sc,
                message="CRITICAL EXCEPTION",
            )
            overall_status = False
        finally:
            logger_sc.info("--- Phase 6: Finalizing - Closing Session ---")
            if session_manager_sc:
                try:
                    session_manager_sc.close_sess()
                    _sc_run_test(
                        "SessionManager.close_sess()",
                        lambda: True,
                        test_results_sc,
                        logger_sc,
                    )
                except Exception as close_err:
                    _sc_run_test(
                        "SessionManager.close_sess()",
                        lambda: False,
                        test_results_sc,
                        logger_sc,
                        message=f"Exception: {close_err}",
                    )
                    overall_status = False
                # End of try/except
            else:
                _sc_run_test(
                    "SessionManager.close_sess()",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="No session manager to close",
                )
            # End of if/else
        # End of try/except/finally
    else:
        logger_sc.warning(
            "Skipping ALL Live API tests due to unmet prerequisites (e.g., BS4 missing or initial checks failed)."
        )
        phases_to_skip = [
            "SessionManager.start_sess()",
            "SessionManager.ensure_session_ready()",
            "Check Target Tree ID Found",
            "Check Target Owner Name Found",
            "Check Target Owner Profile ID Found",
            "Check Target Owner Global ID (UUID) Found",
            "API Call: Get Target Profile Details (app-api via _sc_get_profile_details)",
            "Check Target Name Found in API Resp",
            "Function Call: parse_ancestry_person_details() (with Live Facts)",
            "Validation: Parsed Details Keys (Live Facts)",
            "Validation: Parsed Name Match (Live Facts)",
            "Validation: Parsed Details Values (Static Suggest)",
            "API Helper: call_suggest_api",
            "API Helper: call_facts_user_api",
            "API Helper: call_send_message_api (dry_run)",
            "API Helper: call_profile_details_api",
            "API Helper: call_header_trees_api_for_tree_id",
            "API Helper: call_tree_owner_api",
            "Format Relationship Path (HTML)",
            "SessionManager.close_sess()",
            "Self-Check Live Execution",
        ]
        existing_test_names = {name for name, _, _ in test_results_sc}
        for test_name in phases_to_skip:
            if test_name not in existing_test_names:
                _sc_run_test(
                    test_name,
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                    message="Prerequisites failed or BS4 missing",
                )
            # End of if
        # End of for
    # End of if/else can_run_live_tests and overall_status

    _sc_print_summary(test_results_sc, overall_status, logger_sc)
    final_overall_status_from_tests = not any(
        status == "FAIL" for _, status, _ in test_results_sc
    )
    final_overall_status = overall_status and final_overall_status_from_tests

    logger_sc.info(
        f"--- api_utils.py Self-Check Finished (Overall Status: {'PASS' if final_overall_status else 'FAIL'}) ---"
    )
    return final_overall_status


# End of self_check

# --- Main Execution Block ---
if __name__ == "__main__":
    print(
        "Running api_utils.py self-check (requires config, may perform live API calls)..."
    )
    log_file_path = Path("api_utils_self_check.log").resolve()
    logger_standalone = None

    try:
        import logging_config

        logger_standalone = logging_config.setup_logging(
            log_file=str(log_file_path), log_level="DEBUG"
        )
        print(f"Detailed logs (DEBUG level) will be written to: {log_file_path}")
        logger_standalone.info(
            f"Logging configured via logging_config.py to {log_file_path}"
        )

        logger_main_module = logging.getLogger("api_utils")
        if not logger_main_module.hasHandlers() and logger_standalone:
            for handler in logger_standalone.handlers:
                logger_main_module.addHandler(handler)
            # End of for
            logger_main_module.setLevel(logger_standalone.level)
        # End of if
    except Exception as log_setup_err:
        print(
            f"Error setting up logging via logging_config: {log_setup_err}. Using basic file logging."
        )
        logging.basicConfig(
            level=logging.DEBUG,
            filename=str(log_file_path),
            filemode="w",
            format="%(asctime)s %(levelname)-8s [%(name)-15s] %(message)s",
        )
        logger_standalone = logging.getLogger("api_utils_standalone_error")
        logger_standalone.exception(
            "Error setting up logging_config, using basicConfig."
        )
        logger_main_module = logging.getLogger("api_utils")
        logger_main_module.setLevel(logging.DEBUG)
    # End of try/except

    config_ok_for_tests = False
    test_person_id = getattr(config_instance, "TESTING_PERSON_TREE_ID", None)
    test_profile_id = getattr(config_instance, "TESTING_PROFILE_ID", None)
    tree_name_check = getattr(config_instance, "TREE_NAME", None)

    if not test_person_id or not test_profile_id or not tree_name_check:
        print("\n" + "=" * 70)
        print(" WARNING: Configuration Incomplete for Full Self-Check ".center(70, "="))
        print("=".center(70, "="))
        if not test_person_id:
            print("- config.TESTING_PERSON_TREE_ID is not set.")
        # End of if
        if not test_profile_id:
            print("- config.TESTING_PROFILE_ID is not set.")
        # End of if
        if not tree_name_check:
            print("- config.TREE_NAME is not set.")
        # End of if
        print("\nLive API tests requiring these IDs/Names may be skipped or fail.")
        print("Ensure these are set in config.py or .env for comprehensive testing.")
        print("=".ljust(70, "="))
        config_ok_for_tests = False
    else:
        config_ok_for_tests = True
        print(
            "\nConfiguration check: TESTING_PERSON_TREE_ID, TESTING_PROFILE_ID, and TREE_NAME found."
        )
    # End of if/else

    print("\nStarting self_check function...")
    self_check_passed = self_check()

    print("\napi_utils module self-check complete.")
    print("Import this module into other scripts to use its functions.")
    sys.exit(0 if self_check_passed else 1)
# End of __main__
