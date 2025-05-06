# api_utils.py
"""
Utility functions for parsing Ancestry API responses and formatting API data.

Provides functions to:
- Parse person details from various Ancestry API responses.
- Format relationship paths obtained from Tree Ladder or Discovery APIs.
- Call specific Ancestry APIs (Suggest, Facts, Ladder, Discovery Relationship, TreesUI).
- Includes a self-check mechanism using live API calls (requires configuration).
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

# --- Third-party imports ---
# Keep BeautifulSoup import here, check for its availability in functions
try:
    from bs4 import BeautifulSoup, FeatureNotFound

    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None  # type: ignore # Gracefully handle missing dependency
    FeatureNotFound = None  # type: ignore
    BS4_AVAILABLE = False

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
    # Fallback basic format_name
    def format_name(name_str: Optional[str]) -> str:
        return str(name_str).title() if name_str else "Unknown"

    # End of format_name fallback

    # Fallback basic ordinal_case
    def ordinal_case(text: str) -> str:
        return str(text)

    # End of ordinal_case fallback

    # Define dummy SessionManager and _api_req if utils unavailable
    class DummySessionManager:
        driver = None
        _requests_session = None
        my_tree_id = None
        tree_owner_name = "Dummy Owner"
        my_profile_id = None
        my_uuid = None

        def is_sess_valid(self) -> bool:
            return False

        def close_sess(self) -> None:
            pass

        def start_sess(self, action_name: str = "") -> bool:
            return False

        def ensure_session_ready(self, action_name: str = "") -> bool:
            return False

        def get_csrf_token(self) -> str:
            return "dummy_token"

    # End of DummySessionManager class

    SessionManager = DummySessionManager  # type: ignore
    _api_req = lambda *args, **kwargs: None  # type: ignore

    logger.warning("Failed to import utils, using fallback/dummy components.")
# End try/except utils import

try:
    from gedcom_utils import _parse_date, _clean_display_date

    GEDCOM_UTILS_AVAILABLE = True
    logger.info("Successfully imported gedcom_utils date functions")
except ImportError:
    # Fallback basic date parser (returns None)
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        return None

    # End of _parse_date fallback

    # Fallback basic date cleaner
    def _clean_display_date(date_str: Optional[str]) -> str:
        return str(date_str) if date_str else "N/A"

    # End of _clean_display_date fallback

    logger.warning("Failed to import gedcom_utils, using fallback date functions")
# End try/except gedcom_utils import

try:
    from config import config_instance, selenium_config

    CONFIG_AVAILABLE = True
    logger.info("Successfully imported config instances")
except ImportError:
    CONFIG_AVAILABLE = False

    # Fallback to dummy config if config.py is not available
    class DummyConfig:
        BASE_URL = "https://www.ancestry.com"
        TESTING_PROFILE_ID = (
            "00000000-0000-0000-0000-000000000000"  # Example placeholder
        )
        TESTING_PERSON_TREE_ID = None
        API_TIMEOUT = 60

    # End of DummyConfig class

    config_instance = DummyConfig()

    # Create a dummy selenium_config if needed
    class DummySeleniumConfig:
        API_TIMEOUT = 60

    # End of DummySeleniumConfig class

    selenium_config = DummySeleniumConfig()
    logger.warning("Failed to import config from config.py, using default values")
# End try/except config import


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
        if name == "Unknown":
            name = facts_data.get("personName", name)
        if name == "Unknown":
            name = facts_data.get("DisplayName", name)
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
    if name == "Unknown" and person_card:
        suggest_fullname = person_card.get("FullName")
        suggest_given = person_card.get("GivenName")
        suggest_sur = person_card.get("Surname")
        if suggest_fullname:
            name = suggest_fullname
        elif suggest_given or suggest_sur:
            name = f"{suggest_given or ''} {suggest_sur or ''}".strip() or "Unknown"
        if name == "Unknown":
            name = person_card.get("name", "Unknown")
    formatted_name = formatter(name) if name and name != "Unknown" else "Unknown"
    return "Unknown" if formatted_name == "Valued Relative" else formatted_name
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
        if not gender_str:
            gender_str = facts_data.get("gender")
        if not gender_str:
            gender_str = facts_data.get("PersonGender")
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
    if not gender_str and person_card:
        gender_str = person_card.get("Gender")
        if not gender_str:
            gender_str = person_card.get("gender")
    if gender_str and isinstance(gender_str, str):
        gender_str_lower = gender_str.lower()
        if gender_str_lower == "male":
            gender = "M"
        elif gender_str_lower == "female":
            gender = "F"
        elif gender_str_lower in ["m", "f"]:
            gender = gender_str_lower.upper()
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
        if is_living is None:
            is_living = facts_data.get("isLiving")
        if is_living is None:
            is_living = facts_data.get("IsPersonLiving")
    if is_living is None and person_card:
        is_living = person_card.get("IsLiving")
        if is_living is None:
            is_living = person_card.get("isLiving")
    return bool(is_living) if is_living is not None else None
# End of _extract_living_status_from_api_details


def _extract_event_from_api_details(
    event_type: str, person_card: Dict, facts_data: Optional[Dict]
) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    date_str: Optional[str] = None
    place_str: Optional[str] = None
    date_obj: Optional[datetime] = None
    parser = _parse_date if GEDCOM_UTILS_AVAILABLE else lambda x: None
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
                            if day:
                                temp_date_str += f"-{str(day).zfill(2)}"
                            date_obj = parser(temp_date_str)
                            logger.debug(
                                f"Parsed {event_type} date object from ParsedDate: {date_obj}"
                            )
                        except Exception as dt_err:
                            logger.warning(
                                f"Could not parse {event_type} date from ParsedDate {parsed_date_data}: {dt_err}"
                            )
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
                    if isinstance(place_info, dict):
                        place_str = place_info.get("placeName")
                    found_in_facts = True
        if not found_in_facts:
            event_fact_alt = facts_data.get(app_api_key)
            if event_fact_alt and isinstance(event_fact_alt, dict):
                date_str = event_fact_alt.get("normalized", event_fact_alt.get("date"))
                place_str = event_fact_alt.get("place", place_str)
                found_in_facts = True
            elif isinstance(event_fact_alt, str):
                date_str = event_fact_alt
                found_in_facts = True
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
            elif isinstance(event_info_card, dict):
                date_str = event_info_card.get("date", date_str)
                if place_str is None:
                    place_str = event_info_card.get("place", place_str)
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
    if tree_id and person_id:
        return f"{base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/facts"
    elif person_id:
        return f"{base_url}/discoveryui-matches/list/summary/{person_id}"
    else:
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
    if not details["tree_id"]:
        details["tree_id"] = person_card.get("treeId")
    if facts_data and isinstance(facts_data, dict):
        details["person_id"] = facts_data.get("PersonId", details["person_id"])
        details["tree_id"] = facts_data.get("TreeId", details["tree_id"])
        details["user_id"] = facts_data.get("UserId", details["user_id"])
        if not details["user_id"]:
            person_info = facts_data.get("person", {})
            if isinstance(person_info, dict):
                details["user_id"] = person_info.get("userId", details["user_id"])
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
    cleaner = (
        _clean_display_date
        if GEDCOM_UTILS_AVAILABLE
        else lambda x: str(x) if x else "N/A"
    )
    details["birth_date"] = cleaner(birth_date_raw) if birth_date_raw else "N/A"
    details["death_date"] = cleaner(death_date_raw) if death_date_raw else "N/A"
    if details["birth_date"] == "N/A" and details["api_birth_obj"]:
        details["birth_date"] = str(details["api_birth_obj"].year)
    if details["death_date"] == "N/A" and details["api_death_obj"]:
        details["death_date"] = str(details["api_death_obj"].year)
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


def print_group(label: str, items: List[Dict]):
    print(f"\n{label}:")
    if items:
        formatter = format_name
        for item in items:
            name_to_format = item.get("name") if isinstance(item, dict) else None
            print(f"  - {formatter(name_to_format)}")
    else:
        print("  (None found)")
# End of print_group


def _get_relationship_term(
    person_a_gender: Optional[str], basic_relationship: str
) -> str:
    term = basic_relationship.capitalize()
    rel_lower = basic_relationship.lower()
    if rel_lower == "parent":
        if person_a_gender == "M":
            term = "Father"
        elif person_a_gender == "F":
            term = "Mother"
    elif rel_lower == "child":
        if person_a_gender == "M":
            term = "Son"
        elif person_a_gender == "F":
            term = "Daughter"
    elif rel_lower == "sibling":
        if person_a_gender == "M":
            term = "Brother"
        elif person_a_gender == "F":
            term = "Sister"
    elif rel_lower == "spouse":
        if person_a_gender == "M":
            term = "Husband"
        elif person_a_gender == "F":
            term = "Wife"
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
    Parses relationship data from Ancestry APIs and formats it into a readable path.

    Handles:
    1.  Discovery API JSON response (expects `{"path": [...]}`).
    2.  Tree Ladder API HTML/JSONP response (`/getladder`, contains embedded HTML).

    Args:
        api_response_data: Raw data from the API (dict for Discovery, string for Ladder).
        owner_name: Name of the tree owner (often "You").
        target_name: Name of the person whose relationship is being displayed.

    Returns:
        Formatted string representing the relationship path, or an error message string.
    """
    if not api_response_data:
        logger.warning(
            "format_api_relationship_path: Received empty API response data."
        )
        return "(No relationship data received from API)"

    # --- Initialize variables ---
    html_content_raw: Optional[str] = None  # Raw HTML string from JSONP
    json_data: Optional[Dict] = None  # Parsed JSON data if input is dict
    api_status: str = "unknown"
    response_source: str = "Unknown"  # 'JSONP', 'JSON', 'RawString'
    name_formatter = format_name  # Use imported or fallback formatter

    # --- Step 1: Process Input Data Type ---
    # Determine if input is JSON, JSONP string, or other string, and extract key data.
    if isinstance(api_response_data, dict):
        response_source = "JSON"
        if "error" in api_response_data:
            # Handle direct error object from API
            return f"(API returned error object: {api_response_data.get('error', 'Unknown')})"
        elif "path" in api_response_data and isinstance(
            api_response_data.get("path"), list
        ):
            # Handle Discovery API JSON format (ensure path is a list)
            logger.debug("Detected direct JSON 'path' format (Discovery API).")
            json_data = api_response_data
        elif (
            "html" in api_response_data
            and "status" in api_response_data
            and isinstance(api_response_data.get("html"), str)
        ):
            # Handle pre-parsed JSONP structure (e.g., if wrapper already handled it)
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
        # Check for common JSONP wrappers (__ancestry_jsonp_...(...) or no(...))
        if (
            api_response_data.strip().startswith("__ancestry_jsonp_")
            and api_response_data.strip().endswith(");")
        ) or (
            api_response_data.strip().startswith("no(")
            and api_response_data.strip().endswith(")")
        ):
            response_source = "JSONP"
            try:
                # Extract the JSON part from the wrapper
                json_part_match = re.search(
                    r"^\s*[\w$.]+\((.*)\)\s*;?\s*$", api_response_data, re.DOTALL
                ) or re.search(r"^\s*no\((.*)\)\s*$", api_response_data, re.DOTALL)

                if json_part_match:
                    json_part_str = json_part_match.group(1).strip()
                    logger.debug(f"Extracted JSON part: {json_part_str[:100]}...")
                    # Parse the extracted JSON string
                    parsed_json = json.loads(json_part_str)
                    api_status = parsed_json.get("status", "unknown")

                    if api_status == "success":
                        html_content_raw = parsed_json.get("html")
                        if not isinstance(html_content_raw, str):
                            logger.warning(
                                "JSONP status 'success', but 'html' key missing or not a string."
                            )
                            html_content_raw = None
                        else:
                            logger.debug("Successfully extracted 'html' from JSONP.")
                    else:
                        # Return status message from JSONP if not successful
                        return f"(API status '{api_status}' in JSONP: {parsed_json.get('message', 'Error')})"
                else:
                    logger.warning("Could not extract JSON part from JSONP wrapper.")
                    # Fallback: Treat the original string as potential raw HTML/text
                    html_content_raw = api_response_data
                    response_source = "RawString"
            except json.JSONDecodeError as json_err:
                logger.error(
                    f"Error decoding JSON part from {response_source}: {json_err}"
                )
                # Include snippet of problematic string
                error_context = (
                    f" near: {json_part_str[:100]}..."
                    if "json_part_str" in locals()
                    else ""
                )
                return f"(Error parsing JSONP data: {json_err}{error_context})"
            except Exception as e:
                logger.error(f"Error processing {response_source}: {e}", exc_info=True)
                # Fallback: Treat the original string as potential raw HTML/text
                html_content_raw = api_response_data
                response_source = "RawString"
        else:
            # Input string doesn't look like JSONP, treat as raw content
            html_content_raw = api_response_data
            response_source = "RawString"
    else:
        # Handle unsupported input types
        return f"(Unsupported data type received: {type(api_response_data)})"

    # --- Step 2: Format Discovery API JSON Path (if applicable) ---
    if json_data and "path" in json_data:
        path_steps_json = []
        discovery_path = json_data["path"]
        if isinstance(discovery_path, list) and discovery_path:
            logger.info("Formatting relationship path from Discovery API JSON.")
            # Start with the target person's name (provided to function)
            path_steps_json.append(f"*   {name_formatter(target_name)}")
            # Iterate through the steps in the path from the API response
            for i, step in enumerate(discovery_path):
                step_name = name_formatter(step.get("name", "?"))
                step_rel = step.get("relationship", "?")
                # Use _get_relationship_term for consistent capitalization/ordinals
                step_rel_display = _get_relationship_term(None, step_rel).capitalize()

                # Add the relationship connector line based on the *current* step's relationship
                path_steps_json.append(f"    -> is {step_rel_display} of")  # Use " of"
                # Add the next person in the path
                path_steps_json.append(f"*   {step_name}")

            # Add the final connector indicating the end of the explicit path from API
            # Note: Discovery API path ends at the last known person before the owner.
            path_steps_json.append(f"    -> leads to")
            path_steps_json.append(
                f"*   {owner_name} (You)"
            )  # Represent owner at the end

            result_str = "\n".join(path_steps_json)
            logger.debug(f"Formatted Discovery relationship path:\n{result_str}")
            return result_str
        else:
            logger.warning(
                f"Discovery 'path' data invalid or empty: {json_data.get('path')}"
            )
            return "(Discovery path found but is empty or invalid)"
    # End if json_data (Discovery format)

    # --- Step 3: Process HTML Content (from /getladder JSONP) ---
    if not html_content_raw:
        logger.warning("No processable HTML content found for relationship path.")
        # Return specific message if JSONP was parsed but had no HTML
        if response_source == "JSONP" and api_status != "success":
            # Message already returned earlier in this case
            return f"(API status '{api_status}' in JSONP, no HTML content)"  # Fallback message
        return "(Could not find or extract relationship HTML content)"

    # Decode HTML entities (e.g., &lt; becomes <) and unicode escapes (\uXXXX)
    html_content_decoded: Optional[str] = None
    try:
        # First, handle standard HTML entities
        html_content_intermediate = html.unescape(html_content_raw)
        # Second, decode unicode escapes often used in JSON embedding
        html_content_decoded = bytes(html_content_intermediate, "utf-8").decode(
            "unicode_escape"
        )
        logger.debug(f"Decoded HTML content: {html_content_decoded[:250]}...")
    except Exception as decode_err:
        logger.error(f"Failed to decode HTML content: {decode_err}", exc_info=True)
        # Fallback to using the raw content, might fail parsing later
        html_content_decoded = html_content_raw  # Keep original if decode fails

    if not BS4_AVAILABLE or not BeautifulSoup:
        logger.error("BeautifulSoup library not available. Cannot parse HTML.")
        return "(Cannot parse relationship HTML - BeautifulSoup library missing)"

    # --- Step 4: Parse Decoded HTML with BeautifulSoup ---
    try:
        logger.debug("Attempting to parse DECODED HTML content with BeautifulSoup...")
        soup = None
        # Prefer lxml for speed/robustness, fall back to html.parser
        parser_to_try = ["lxml", "html.parser"]

        for parser_name in parser_to_try:
            try:
                soup = BeautifulSoup(html_content_decoded, parser_name)
                logger.info(f"Successfully parsed HTML using '{parser_name}'.")
                break  # Stop if parsing succeeds
            except FeatureNotFound:
                logger.warning(f"'{parser_name}' parser not found. Trying next.")
            except Exception as parse_err:
                # Catch potential parsing errors with a specific parser
                logger.warning(
                    f"Error using '{parser_name}' parser: {parse_err}. Trying next."
                )
        # End for parser_name

        if not soup:
            logger.error("BeautifulSoup failed to parse HTML with available parsers.")
            return "(Error parsing relationship HTML - BeautifulSoup failed)"

        # --- Step 5: Extract Path Information from Parsed HTML ---
        # Find the list items representing the relationship path
        # Target selector based on observed structure in getladder response
        list_items = soup.select("ul.textCenter li")
        if not list_items:
            logger.warning(
                "Expected list items ('ul.textCenter li'), found none in parsed HTML."
            )
            logger.debug(
                f"Parsed HTML structure (abbreviated):\n{soup.prettify()[:500]}"
            )
            return "(Relationship HTML structure not as expected - Found 0 list items)"

        # Extract overall relationship summary (usually the first item's italic/bold text)
        overall_relationship = "unknown relationship"
        if list_items:  # Check if list_items is not empty
            summary_tag = list_items[0].select_one("i")  # Look for <i> tag
            if summary_tag:
                # Extract the relationship part, removing the names
                summary_text = summary_tag.get_text(strip=True)
                # Find the relationship part after the possessive
                match = re.search(
                    r"is\s+.*?'s\s+(\w+(?:\s+\w+)*)",
                    summary_text,
                    re.IGNORECASE,
                )
                if match:
                    overall_relationship = match.group(1).lower()
                else:
                    overall_relationship = summary_text.lower()
        logger.debug(
            f"Extracted overall relationship summary: '{overall_relationship}'"
        )

        # Log the HTML content for debugging
        logger.debug(f"HTML content: {soup.prettify()[:1000]}")

        # Filter out items that are just decorative arrows (often have 'iconArrowDown' class)
        path_items = [
            li for li in list_items if "iconArrowDown" not in li.get("class", [])
        ]
        logger.debug(f"Found {len(path_items)} relevant path items (summary + people).")

        # The actual path starts from the second relevant item (index 1)
        if len(path_items) < 2:  # Need at least summary + one person
            logger.warning(
                f"Expected at least 2 relevant items (summary + 1 person), found {len(path_items)}."
            )
            return "(Could not find sufficient relationship path steps in HTML)"

        # --- Step 6: Build Formatted Output String (Bulleted List for HTML Path) ---
        # Use target_name provided to the function for the summary line
        summary_line = f"{target_name} is {owner_name}'s {overall_relationship}:"

        # Use the summary line from the API response
        path_lines = []

        # Collect all people and their relationships
        people = []
        relationships = []

        # Iterate through the actual people items, STARTING FROM INDEX 1 (skipping the summary item)
        # path_items[1] should be the first person in the path (target_name equivalent from HTML)
        # path_items[len-1] should be the owner/you
        for i in range(1, len(path_items)):  # Start loop from 1
            item = path_items[i]

            # a) Extract Name (usually in <a> or <b> tag)
            name_tag = item.find("a") or item.find("b")
            current_person_name_raw = (
                name_tag.get_text(strip=True) if name_tag else "Unknown"
            )

            # b) Extract Lifespan (heuristic based on text after name or year in name)
            lifespan = ""
            # Look for a year at the end of the name (e.g., "Frances Margaret Milne 1947")
            year_in_name_match = re.search(r"\s+(\d{4})$", current_person_name_raw)
            display_name = current_person_name_raw
            if year_in_name_match:
                # If name ends with a year, remove it and use for lifespan
                year = year_in_name_match.group(1)
                # Store the year with "b." prefix for birth year
                lifespan = f"b. {year}"
                # Create a display name without the year
                display_name = current_person_name_raw[
                    : year_in_name_match.start()
                ].strip()
            elif (
                name_tag
                and name_tag.next_sibling
                and isinstance(name_tag.next_sibling, str)
            ):
                # Check text immediately following the name tag for (YYYY-YYYY) or (YYYY)
                potential_lifespan = name_tag.next_sibling.strip()
                # Match common patterns like (YYYY-YYYY) or (YYYY)
                lifespan_match = re.match(
                    r"\(\s*(\d{4})\s*(?:[-–—]\s*(\d{4}))?\s*\)$", potential_lifespan
                )
                if lifespan_match:
                    start_year = lifespan_match.group(1)
                    end_year = lifespan_match.group(2)
                    lifespan = f"{start_year}–{end_year}" if end_year else start_year
            # End lifespan extraction

            # c) Format the extracted name
            current_person_name = name_formatter(current_person_name_raw)
            # Special case: If the name extracted is 'You', use the provided owner_name
            # Check against lowercase and strip potential "(You)" part from owner_name for comparison
            owner_name_base = owner_name.replace("(You)", "").strip()
            # Ensure comparison handles potential "(You)" suffix in owner_name
            if (
                current_person_name.lower() == "you"
                or name_formatter(current_person_name_raw) == owner_name_base
            ):
                current_person_name = f"{owner_name_base}"
                is_owner = True
            else:
                is_owner = False

            # Store the person with their lifespan
            # Use name without year for display in relationship descriptions
            display_name = current_person_name_raw
            # If there's a year in the name, remove it for display purposes
            year_in_display_match = re.search(r"\s+(\d{4})$", display_name)
            if year_in_display_match:
                display_name = display_name[: year_in_display_match.start()].strip()

            person_info = {
                "name": current_person_name,
                "lifespan": lifespan,
                "is_owner": is_owner,
                "display_name": display_name,
            }
            people.append(person_info)

            # e) Get relationship IF this is NOT the last person in the path
            if i < len(path_items) - 1:
                # Relationship description is usually in the *same* <li>'s <i> tag
                # This describes the relationship of the *current* person (i) to the *next* person (i+1)
                desc_tag = item.find("i")
                desc_text = desc_tag.get_text(strip=True) if desc_tag else ""

                # Extract the core relationship term (father, mother, son, daughter, etc.)
                relationship_term = "related to"  # Default if specific term not found
                # Regex to find common terms, ignoring surrounding words like "of X"
                rel_match = re.search(
                    r"\b(brother|sister|father|mother|son|daughter|husband|wife|spouse|parent|child|sibling)\b",
                    desc_text,
                    re.IGNORECASE,
                )
                if rel_match:
                    relationship_term = rel_match.group(1).lower()
                # Handle "You are the..." case specifically if needed (less common here)
                elif "you are the" in desc_text.lower():
                    you_match = re.search(
                        r"You\s+are\s+the\s+([\w\s]+)", desc_text, re.IGNORECASE
                    )
                    if you_match:
                        # This case is less likely in the intermediate steps' descriptions
                        relationship_term = you_match.group(1).strip().lower()

                relationships.append(relationship_term)
            # End if not last item
        # End for loop processing path items (indices 1 to end)

        # Now format the path in the desired format
        for i in range(len(people)):
            person = people[i]
            person_name = person["name"]
            person_lifespan = person["lifespan"]

            # Format the person line
            # Remove any year from the person's name for display
            clean_name = person_name
            name_parts = person_name.split()
            if len(name_parts) > 0 and name_parts[-1].isdigit():
                # Extract the year from the name
                year = name_parts[-1]
                # Remove the year from the name
                clean_name = " ".join(name_parts[:-1])
                # Update the person's name to not include the year
                person_name = clean_name
                # Add the birth year in the desired format
                person_line = f"*   {person_name} (b. {year})"
            else:
                person_line = f"*   {person_name}"
                if person_lifespan:
                    # Format birth year as (b. YYYY) instead of just the year
                    if person_lifespan.startswith("b. "):
                        person_line += f" (b. {person_lifespan.replace('b. ', '')})"
                    else:
                        person_line += f" ({person_lifespan})"

            # Store the clean name for relationship descriptions
            person["clean_name"] = clean_name

            # Add relationship description
            if i < len(people) - 1:
                # Not the last person - use the relationship to the next person
                next_person = people[i + 1]
                relationship = relationships[i]

                # Use the clean name for relationship descriptions
                next_person_name = next_person.get(
                    "clean_name", next_person.get("display_name", next_person["name"])
                )

                if relationship.lower() in ["father", "mother", "parent"]:
                    person_line += f" is {next_person_name}'s {relationship}"
                elif relationship.lower() in ["son", "daughter", "child"]:
                    prev_person_name = target_name
                    if i > 0:
                        prev_person = people[i - 1]
                        prev_person_name = prev_person.get(
                            "clean_name",
                            prev_person.get("display_name", prev_person["name"]),
                        )
                    person_line += f" is {prev_person_name}'s {relationship}"
                elif relationship.lower() in ["brother", "sister", "sibling"]:
                    person_line += f" is {next_person_name}'s {relationship}"
                else:
                    person_line += f" is {relationship} to {next_person_name}"
            elif i > 0:
                # Last person - use the relationship to the previous person
                prev_person = people[i - 1]

                # Use the clean name for relationship descriptions
                prev_person_name = prev_person.get(
                    "clean_name", prev_person.get("display_name", prev_person["name"])
                )

                # Determine the relationship based on the previous person's relationship
                if i > 0 and i - 1 < len(relationships):
                    prev_relationship = relationships[i - 1]
                    if prev_relationship.lower() == "daughter":
                        person_line += f" is {prev_person_name}'s son"
                    else:
                        # Default relationship for the last person
                        person_line += f" is {prev_person_name}'s son"

            path_lines.append(person_line)

        # --- Step 7: Combine and Return Formatted String ---
        result_str = f"{summary_line}\n\n" + "\n".join(path_lines)
        logger.info("Formatted relationship path from HTML successfully.")
        logger.debug(f"Formatted HTML relationship path:\n{result_str}")
        return result_str

    except Exception as e:
        logger.error(
            f"Error processing relationship HTML with BeautifulSoup: {e}", exc_info=True
        )
        # Include decoded HTML snippet in error if possible for debugging
        error_context = (
            f" near HTML: {html_content_decoded[:200]}..."
            if html_content_decoded
            else ""
        )
        return f"(Error parsing relationship HTML: {e}{error_context})"
# End of format_api_relationship_path


def _get_api_timeout(default: int = 60) -> int:
    """Safely gets the API timeout value from configuration or returns default."""
    timeout_value = default
    if CONFIG_AVAILABLE and selenium_config and hasattr(selenium_config, "API_TIMEOUT"):
        config_timeout = getattr(selenium_config, "API_TIMEOUT")
        if isinstance(config_timeout, (int, float)) and config_timeout > 0:
            timeout_value = int(config_timeout)
        else:
            logger.warning(
                f"Invalid API_TIMEOUT value in config ({config_timeout}), using default {default}s."
            )
    return timeout_value
# End of _get_api_timeout


def _get_owner_referer(session_manager: "SessionManager", base_url: str) -> str:
    """Constructs the referer URL for API calls, typically the owner's facts page."""
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
# End of _get_owner_referer


def call_suggest_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    owner_profile_id: Optional[str],  # Needed for referer construction
    base_url: str,
    search_criteria: Dict[str, Any],
    timeouts: Optional[List[int]] = None,
) -> Optional[List[Dict]]:
    """
    Calls the Ancestry Suggest API (/api/person-picker/suggest) to find people.

    Uses progressive timeouts via _api_req and includes a direct requests fallback.
    """
    # 1. Check prerequisites
    if not callable(_api_req) or not isinstance(session_manager, SessionManager):
        logger.error(
            "Suggest API call failed: _api_req function or SessionManager unavailable."
        )
        return None
    if not owner_tree_id:
        logger.error("Suggest API call failed: owner_tree_id is required.")
        return None

    # 2. Prepare API call parameters
    api_description = "Suggest API"
    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")
    suggest_params_list = ["isHideVeiledRecords=false"]
    if first_name_raw:
        suggest_params_list.append(f"partialFirstName={quote(first_name_raw)}")
    if surname_raw:
        suggest_params_list.append(f"partialLastName={quote(surname_raw)}")
    if birth_year:
        suggest_params_list.append(f"birthYear={birth_year}")
    suggest_params = "&".join(suggest_params_list)
    suggest_url = f"{base_url.rstrip('/')}/api/person-picker/suggest/{owner_tree_id}?{suggest_params}"
    owner_facts_referer = _get_owner_referer(session_manager, base_url)
    timeouts_used = timeouts if timeouts else [20, 30, 60]
    max_attempts = len(timeouts_used)
    logger.info(f"Attempting {api_description} search: {suggest_url}")
    print(f"\nSearch API URL: {suggest_url}\n")

    # 3. Attempt API call using _api_req with progressive timeouts
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
            # REMOVED expect_json=True
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
                print(f"({api_description} call returned unexpected data format.)")
                suggest_response = None
                break
        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} _api_req call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
            if attempt < max_attempts:
                print("Timeout occurred. Retrying with longer timeout...")
        except Exception as api_err:
            # Catch potential TypeError here if _api_req signature is wrong
            logger.error(
                f"{api_description} _api_req call failed on attempt {attempt}/{max_attempts}: {api_err}",
                exc_info=True,
            )
            suggest_response = None
            break
    # End for loop

    # 4. Direct Request Fallback (remains unchanged)
    if suggest_response is None:
        logger.warning(
            f"{api_description} failed via _api_req. Attempting direct requests fallback."
        )
        print("\nAttempting direct request fallback...")
        try:
            cookies = {}
            direct_response = None  # Initialize direct_response here
            if session_manager._requests_session:
                cookies = session_manager._requests_session.cookies.get_dict()
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
            print(f"API URL Called (Direct Fallback): {suggest_url}\n")
            logger.debug(f"Direct request headers: {direct_headers}")
            logger.debug(f"Direct request cookies: {list(cookies.keys())}")
            direct_timeout = _get_api_timeout(30)
            direct_response = requests.get(
                suggest_url,
                headers=direct_headers,
                cookies=cookies,
                timeout=direct_timeout,
            )
            if direct_response.status_code == 200:
                direct_data = direct_response.json()
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
            else:
                logger.warning(
                    f"Direct request fallback failed: Status {direct_response.status_code}"
                )
                logger.debug(f"Direct Response content: {direct_response.text[:500]}")
        except requests.exceptions.Timeout:
            logger.error(
                f"Direct request fallback timed out after {direct_timeout} seconds"
            )
            print("Direct request timed out.")
        except json.JSONDecodeError as json_err:
            logger.error(f"Direct request fallback failed to decode JSON: {json_err}")
            if direct_response:
                logger.debug(f"Direct Response content: {direct_response.text[:500]}")
            print("Direct request failed to parse response.")
        except Exception as direct_err:
            logger.error(
                f"Direct request fallback failed with error: {direct_err}",
                exc_info=True,
            )
            print(f"Direct request failed: {direct_err}")
    # End fallback

    # 5. Final failure
    logger.error(f"{api_description} failed after all attempts and fallback.")
    print(f"\nError: Could not retrieve suggestions from API. Check logs.")
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
    """
    Calls the Ancestry Person Facts User API to get detailed person data.

    Attempts direct request first, then falls back to _api_req with timeouts.
    """
    # 1. Check prerequisites
    if not callable(_api_req) or not isinstance(session_manager, SessionManager):
        logger.error(
            "Facts API call failed: _api_req function or SessionManager unavailable."
        )
        return None
    if not all([owner_profile_id, api_person_id, api_tree_id]):
        logger.error(
            "Facts API call failed: owner_profile_id, api_person_id, and api_tree_id are required."
        )
        return None

    # 2. Prepare API call parameters
    api_description = "Person Facts User API"
    facts_api_url = f"{base_url.rstrip('/')}/family-tree/person/facts/user/{owner_profile_id.lower()}/tree/{api_tree_id.lower()}/person/{api_person_id.lower()}"
    facts_referer = _get_owner_referer(session_manager, base_url)
    facts_data_raw = None
    direct_timeout = _get_api_timeout(30)
    fallback_timeouts = timeouts if timeouts else [30, 45, 60]
    logger.info(f"Attempting {api_description} via direct request: {facts_api_url}")
    print(f"\nFamily Facts API URL (Direct Attempt): {facts_api_url}\n")

    # 3. Direct Request Attempt First (remains unchanged)
    try:
        cookies = {}
        direct_response = None
        if session_manager._requests_session:
            cookies = session_manager._requests_session.cookies.get_dict()
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
        direct_response = requests.get(
            facts_api_url,
            headers=direct_headers,
            cookies=cookies,
            timeout=direct_timeout,
        )
        if direct_response.status_code == 200:
            facts_data_raw = direct_response.json()
            if not isinstance(facts_data_raw, dict):
                logger.warning(
                    f"Direct facts request OK (200) but returned non-dict data: {type(facts_data_raw)}"
                )
                logger.debug(f"Response content: {direct_response.text[:500]}")
                facts_data_raw = None
            else:
                logger.info(f"{api_description} call successful via direct request.")
        else:
            logger.warning(
                f"Direct facts request failed: Status {direct_response.status_code}"
            )
            logger.debug(f"Response content: {direct_response.text[:500]}")
            facts_data_raw = None
    except requests.exceptions.Timeout:
        logger.error(f"Direct facts request timed out after {direct_timeout} seconds")
        print("Direct facts request timed out. Trying _api_req fallback...")
        facts_data_raw = None
    except json.JSONDecodeError as json_err:
        logger.error(f"Direct facts request failed to decode JSON: {json_err}")
        if direct_response:
            logger.debug(f"Response content: {direct_response.text[:500]}")
        print(
            "Direct facts request failed to parse response. Trying _api_req fallback..."
        )
        facts_data_raw = None
    except Exception as direct_err:
        logger.error(f"Direct facts request failed: {direct_err}", exc_info=True)
        print(
            f"Direct facts request failed ({direct_err}). Trying _api_req fallback..."
        )
        facts_data_raw = None
    # End direct attempt

    # 4. Fallback to _api_req
    if facts_data_raw is None:
        logger.warning(
            f"{api_description} direct request failed. Trying _api_req fallback."
        )
        print(f"\nAPI URL Called (_api_req Fallback): {facts_api_url}\n")
        max_attempts = len(fallback_timeouts)
        for attempt, timeout in enumerate(fallback_timeouts, 1):
            logger.debug(
                f"{api_description} _api_req attempt {attempt}/{max_attempts} with timeout {timeout}s"
            )
            print(
                f"(Attempt {attempt}/{max_attempts}: Fetching details via _api_req... Timeout: {timeout}s)"
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
                # REMOVED expect_json=True
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
            except requests.exceptions.Timeout:
                logger.warning(
                    f"{api_description} _api_req call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
                )
                if attempt < max_attempts:
                    print("Timeout occurred. Retrying...")
            except Exception as api_req_err:
                logger.error(
                    f"{api_description} call using _api_req failed on attempt {attempt}/{max_attempts}: {api_req_err}",
                    exc_info=True,
                )
                facts_data_raw = None
                break
    # End fallback

    # 5. Process Final Result (remains unchanged)
    if not isinstance(facts_data_raw, dict):
        logger.error(
            f"Failed to fetch valid {api_description} data after all attempts."
        )
        print(
            f"\nError: Could not fetch valid person details from {api_description}. Check logs."
        )
        return None
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
        print(
            f"\nError: API response format for person details was unexpected (missing 'data.personResearch')."
        )
        return None

    # 6. Return extracted data
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
    """
    Calls the Ancestry Tree Ladder API (/getladder) to get relationship path HTML.

    Returns the raw JSONP string response.
    """
    # 1. Check prerequisites
    if not callable(_api_req) or not isinstance(session_manager, SessionManager):
        logger.error(
            "GetLadder API call failed: _api_req function or SessionManager unavailable."
        )
        return None
    if not all([owner_tree_id, target_person_id]):
        logger.error(
            "GetLadder API call failed: owner_tree_id and target_person_id are required."
        )
        return None

    # 2. Prepare API call parameters
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
    api_timeout = timeout if timeout else _get_api_timeout(20)
    logger.info(f"Attempting {api_description} call: {ladder_api_url}")
    print(f"\nRelationship Path API URL: {ladder_api_url}")

    # 3. Call API using _api_req
    try:
        # REMOVED expect_json=False (and force_text_response=True is kept)
        relationship_data = _api_req(
            url=ladder_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            api_description=api_description,
            referer_url=ladder_referer,
            use_csrf_token=False,
            force_text_response=True,  # Ensure we get the raw JSONP string
            timeout=api_timeout,
        )
        # 4. Validate response
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
    except requests.exceptions.Timeout:
        logger.error(f"{api_description} call timed out after {api_timeout}s.")
        print(f"(Error: Timed out fetching relationship path from {api_description})")
        return None
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        print(f"(Error fetching relationship path from {api_description}: {e})")
        return None
# End of call_getladder_api

def call_treesui_list_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    owner_profile_id: Optional[str],  # Needed for referer
    base_url: str,
    search_criteria: Dict[str, Any],
    timeouts: Optional[List[int]] = None,
) -> Optional[List[Dict]]:
    """
    Calls the Ancestry TreesUI List API as an alternative search method.

    Requires birth year. Uses _api_req with progressive timeouts.
    """
    # 1. Check prerequisites
    if not callable(_api_req) or not isinstance(session_manager, SessionManager):
        logger.error(
            "TreesUI List API call failed: _api_req or SessionManager unavailable."
        )
        return None
    if not owner_tree_id:
        logger.error("TreesUI List API call failed: owner_tree_id is required.")
        return None

    # 2. Prepare API call parameters
    api_description = "TreesUI List API (Alternative Search)"
    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")
    if not birth_year:
        logger.warning(
            "Cannot call TreesUI List API: 'birth_year' is missing in search criteria."
        )
        return None
    treesui_params_list = ["limit=100", "fields=NAMES,BIRTH_DEATH"]
    if first_name_raw:
        treesui_params_list.append(f"fn={quote(first_name_raw)}")
    if surname_raw:
        treesui_params_list.append(f"ln={quote(surname_raw)}")
    treesui_params_list.append(f"by={birth_year}")
    treesui_params = "&".join(treesui_params_list)
    treesui_url = f"{base_url.rstrip('/')}/api/treesui-list/trees/{owner_tree_id}/persons?{treesui_params}"
    owner_facts_referer = _get_owner_referer(session_manager, base_url)
    timeouts_used = timeouts if timeouts else [15, 25, 35]
    max_attempts = len(timeouts_used)
    logger.info(f"Attempting {api_description} search using _api_req: {treesui_url}")
    print(f"\nAPI URL Called (TreesUI List Fallback): {treesui_url}\n")

    # 3. Attempt API call using _api_req
    treesui_response = None
    for attempt, timeout in enumerate(timeouts_used, 1):
        logger.debug(
            f"{api_description} attempt {attempt}/{max_attempts} with timeout {timeout}s"
        )
        print(
            f"(Attempt {attempt}/{max_attempts}: Calling {api_description}... Timeout: {timeout}s)"
        )
        try:
            custom_headers = {
                "Accept": "application/json",
                "Referer": owner_facts_referer,
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            }
            # REMOVED expect_json=True
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
                print(
                    f"Alternative API search successful! Found {len(treesui_response)} potential matches."
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
                print("Alternative API search returned unexpected format.")
                return None
        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} _api_req call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
            if attempt < max_attempts:
                print("Timeout occurred. Retrying...")
        except Exception as treesui_err:
            logger.error(
                f"{api_description} _api_req call failed on attempt {attempt}/{max_attempts}: {treesui_err}",
                exc_info=True,
            )
            treesui_response = None
            break
    # End attempts

    # 4. Final failure
    logger.error(f"{api_description} failed after all attempts.")
    print(f"Alternative API search ({api_description}) failed. Check logs.")
    return None
# End of call_treesui_list_api

# --- Standalone Test Block ---

def _sc_run_test(
    test_name: str,
    test_func: Callable,
    test_results_list: List[Tuple[str, str, str]],
    logger_instance: logging.Logger,
    *args,
    **kwargs,
) -> Tuple[str, str, str]:
    """Runs a single self-check test, logs result, stores result, and returns status tuple."""
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
        explicit_skip = False  # Flag for explicit "Skipped" return

        # --- Evaluation Logic ---
        if expect_none:
            passed = result is None
            if not passed:
                message = f"Expected None, got {type(result).__name__}"
        elif expect_type is not None:
            # ***MODIFICATION START***
            # Change behavior when None is received but a type was expected
            if result is None:
                passed = False  # It failed the type check
                # status remains FAIL (default)
                message = f"Expected type {expect_type.__name__}, but function returned None (API/Parse issue?)"
                logger_instance.error(f"Test '{test_name}': {message}")  # Log as error
            elif isinstance(result, expect_type):
                passed = True  # Type matches
            else:  # Type mismatch (and not None)
                passed = False
                message = (
                    f"Expected type {expect_type.__name__}, got {type(result).__name__}"
                )
            # ***MODIFICATION END***
        elif expect_value is not None:
            passed = result == expect_value
            if not passed:
                message = f"Expected value '{str(expect_value)[:50]}...', got '{repr(result)[:100]}...'"
        elif expect_contains is not None:
            if isinstance(result, str):
                if isinstance(expect_contains, (list, tuple)):
                    missing = [sub for sub in expect_contains if sub not in result]
                    passed = not missing  # Pass if the missing list is empty
                    if not passed:
                        message = f"Expected result to contain all of: {expect_contains}. Missing: {missing}"
                elif isinstance(expect_contains, str):
                    passed = expect_contains in result
                    if not passed:
                        message = f"Expected result to contain '{expect_contains}', got '{repr(result)[:100]}...'"
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
                if result is None:
                    message = "Expected truthy value, but function returned None (Underlying call likely failed)"
                else:
                    message = f"Expected truthy value, got {repr(result)[:100]}"
        elif isinstance(result, str) and result == "Skipped":
            passed = False
            explicit_skip = True  # Mark as explicit skip
            status = "SKIPPED"  # Set status directly
            message = ""  # No message for explicit skip
        else:  # Default check if no specific expectation given (usually expect True)
            passed = result is True
            if not passed:
                message = (
                    f"Default check failed: Expected True, got {repr(result)[:100]}"
                )

        # --- Determine Final Status ---
        if not explicit_skip:  # Don't override explicit skip
            status = "PASS" if passed else "FAIL"
            if (
                status == "FAIL" and not message
            ):  # Add default message if failed without one
                message = f"Test condition not met (Result: {repr(result)[:100]})"

    except Exception as e:
        status = "FAIL"
        message = f"EXCEPTION: {type(e).__name__}: {e}"
        logger_instance.error(
            f"Exception during self-check test '{test_name}': {message}\n{traceback.format_exc()}",
            exc_info=False,
        )  # traceback included in message

    # Log result
    log_level = (
        logging.INFO
        if status == "PASS"
        else (logging.WARNING if status == "SKIPPED" else logging.ERROR)
    )
    log_message = f"[ {status:<7} SC ] {test_name}{f': {message}' if message and status != 'PASS' else ''}"
    logger_instance.log(log_level, log_message)

    # Store result tuple (name, status, message if not PASS)
    test_results_list.append((test_name, status, message if status != "PASS" else ""))
    return (test_name, status, message)
# End of _sc_run_test


# --- [ Self-Check Summary Printer ] ---
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
        except ValueError:
            pass
    status_width = 8
    header = f"{'Test Name':<{name_width}} | {'Status':<{status_width}} | {'Message / Details'}"
    print(header)
    print("-" * (len(header) + 5))
    final_fail_count = 0
    final_skip_count = 0
    final_pass_count = 0
    reported_test_names = set()
    for name, status, message in test_results_list:
        if name in reported_test_names:
            continue
        reported_test_names.add(name)
        if status == "FAIL":
            final_fail_count += 1
        elif status == "SKIPPED":
            final_skip_count += 1
            message = "(Skipped)"
        elif status == "PASS":
            final_pass_count += 1
        print(
            f"{name:<{name_width}} | {status:<{status_width}} | {message if status != 'PASS' else ''}"
        )
    print("-" * (len(header) + 5))
    total_tests = len(reported_test_names)
    passed_tests = final_pass_count
    final_overall_status = overall_status and (final_fail_count == 0)
    result_color = "\033[92m" if final_overall_status else "\033[91m"
    reset_color = "\033[0m"
    final_status_msg = (
        f"Result: {result_color}{'PASS' if final_overall_status else 'FAIL'}{reset_color} "
        f"({passed_tests} passed, {final_fail_count} failed, {final_skip_count} skipped out of {total_tests} tests)"
    )
    print(f"{final_status_msg}\n")
    logger_instance.log(
        logging.INFO if final_overall_status else logging.ERROR,
        f"api_utils self-check overall status: {'PASS' if final_overall_status else 'FAIL'}",
    )
# End of _sc_print_summary


def self_check() -> bool:
    """
    Performs internal self-checks for api_utils.py, including LIVE API calls.

    Requires configuration (config.py, .env) for live tests.
    Provides a formatted summary table of test results to the console and logs.
    Tests formatters using live API data only (no static mocks).

    Returns:
        bool: True if all checks pass (including prerequisites), False otherwise.
    """
    # --- Logger Setup for Self-Check ---
    logger_sc = logging.getLogger("api_utils.self_check")
    logger_sc.info("\n" + "=" * 30 + " api_utils.py Self-Check Starting " + "=" * 30)

    # --- Local Imports & Dependency Checks ---
    required_modules_ok = True
    try:
        if not UTILS_AVAILABLE or "utils" not in sys.modules:
            raise ImportError(
                "Base 'utils' module (SessionManager, _api_req) not available."
            )
        if not SessionManager or not callable(_api_req):
            raise ImportError(
                "SessionManager class or _api_req function missing from utils."
            )
        if not CONFIG_AVAILABLE or "config" not in sys.modules:
            raise ImportError(
                "'config' module (config_instance, selenium_config) not available."
            )
        from config import (
            config_instance as config_instance_sc,
            selenium_config as selenium_config_sc,
        )

        if not config_instance_sc or not selenium_config_sc:
            raise ImportError(
                "Config instances (config_instance, selenium_config) are None."
            )
        if not BS4_AVAILABLE:
            logger_sc.warning(
                "BeautifulSoup (bs4) library not found. HTML parsing tests will be skipped."
            )
    except ImportError as import_err:
        print(
            f"\n[SELF-CHECK ERROR] Cannot import dependencies for live tests: {import_err}"
        )
        logger_sc.critical(f"Self-check cannot run due to import error: {import_err}")
        required_modules_ok = False
    except Exception as general_err:
        print(
            f"\n[SELF-CHECK ERROR] Unexpected error during dependency check: {general_err}"
        )
        logger_sc.critical(
            f"Self-check cannot run due to unexpected error: {general_err}",
            exc_info=True,
        )
        required_modules_ok = False

    # --- Test Runner Setup ---
    test_results_sc: List[Tuple[str, str, str]] = []
    session_manager_sc: Optional["SessionManager"] = None
    overall_status = required_modules_ok  # Start with dependency status

    # --- Internal API Call Helpers for Self-Check (Wrappers) ---
    def _sc_api_req_wrapper(
        url: str, description: str, expect_json: bool = True, **kwargs
    ) -> Any:
        nonlocal session_manager_sc, overall_status
        if not callable(_api_req):
            raise RuntimeError("_api_req func unavailable")
        if not session_manager_sc:
            raise RuntimeError("SessionManager not initialized for SC")
        if not session_manager_sc.is_sess_valid():
            logger_sc.error(f"Session invalid before calling '{description}' (SC)")
            overall_status = False  # Mark failure if session invalid
            raise RuntimeError("Session not ready for API call")
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
        return result

    # End of _sc_api_req_wrapper

    def _sc_get_profile_details(profile_id: str) -> Optional[Dict]:
        nonlocal overall_status
        if not config_instance_sc or not profile_id:
            return None
        api_desc = f"Get Profile Details ({profile_id})"
        url = urljoin(
            config_instance_sc.BASE_URL,
            f"/app-api/express/v1/profiles/details?userId={profile_id.upper()}",
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
        except Exception as e:
            logger_sc.error(
                f"Unexpected error in _sc_get_profile_details: {e}", exc_info=True
            )
            overall_status = False
            return None

    # End of _sc_get_profile_details

    def _sc_get_tree_ladder(tree_id: str, person_id: str) -> Optional[str]:
        nonlocal session_manager_sc, overall_status
        if not all(
            [
                config_instance_sc,
                selenium_config_sc,
                session_manager_sc,
                tree_id,
                person_id,
                callable(call_getladder_api),
            ]
        ):
            logger_sc.error("Cannot run _sc_get_tree_ladder: Missing dependencies.")
            overall_status = False
            return None
        if not session_manager_sc.is_sess_valid():
            logger_sc.error("Cannot run _sc_get_tree_ladder: Session invalid.")
            overall_status = False
            return None
        api_timeout = _get_api_timeout(45)
        base_url = config_instance_sc.BASE_URL
        try:
            return call_getladder_api(
                session_manager_sc, tree_id, person_id, base_url, timeout=api_timeout
            )
        except Exception as e:
            logger_sc.error(
                f"Unexpected error calling call_getladder_api in self-check: {e}",
                exc_info=True,
            )
            overall_status = False
            return None

    # End of _sc_get_tree_ladder

    # --- Test Parameters & Configuration Values ---
    can_run_live_tests = overall_status
    target_profile_id = None  # Global ID for Discovery/App API
    target_person_id_for_ladder = None  # Tree-specific ID for Ladder/Facts API
    base_url_sc = "https://www.ancestry.com"  # Default
    if can_run_live_tests:
        target_profile_id = getattr(config_instance_sc, "TESTING_PROFILE_ID", None)
        target_person_id_for_ladder = getattr(
            config_instance_sc, "TESTING_PERSON_TREE_ID", None
        )
        base_url_sc = getattr(config_instance_sc, "BASE_URL", base_url_sc).rstrip("/")
        if not target_profile_id:
            logger_sc.warning(
                "TESTING_PROFILE_ID not set in config. Some tests will be skipped."
            )
        if not target_person_id_for_ladder:
            logger_sc.warning(
                "TESTING_PERSON_TREE_ID not set in config. Ladder/Facts tests will be skipped."
            )
    target_name_from_profile = (
        "Unknown Target"  # Name corresponding to TESTING_PROFILE_ID
    )
    target_name_for_ladder = (
        "Unknown Ladder Target"  # Name corresponding to TESTING_PERSON_TREE_ID
    )

    # ========================================
    # === Self-Check Test Execution Phases ===
    # ========================================

    # === Phase 0: Prerequisite & Static Function Checks ===
    logger_sc.info("--- Phase 0: Prerequisite & Static Function Checks ---")
    _, s0_bs_stat, _ = _sc_run_test(
        "Check BeautifulSoup Import (bs4)",
        lambda: BS4_AVAILABLE,
        test_results_sc,
        logger_sc,
        expected_truthy=True,
    )
    # Check core functions are callable
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
    }
    for name, func in core_funcs.items():
        _, s0_f_stat, _ = _sc_run_test(
            f"Check Function '{name}' Callable",
            lambda f=func: callable(f),
            test_results_sc,
            logger_sc,
            expected_truthy=True,
        )
        if s0_f_stat != "PASS":
            overall_status = False
    _, s0_c_stat, _ = _sc_run_test(
        "Check Config Loaded (BASE_URL)",
        lambda: CONFIG_AVAILABLE and hasattr(config_instance_sc, "BASE_URL"),
        test_results_sc,
        logger_sc,
        expected_truthy=True,
    )
    if s0_c_stat != "PASS":
        overall_status = False

    # === Phase 0b: (REMOVED) ===
    # Static mock data tests for format_api_relationship_path were removed as requested.
    # Testing now relies on live data in Phase 5 and 5b.
    logger_sc.info("--- Phase 0b: (Skipped - Mock Data Tests Removed) ---")

    # === Live Tests Section ===
    target_owner_global_id = None  # Initialize variable needed later
    if (
        can_run_live_tests and overall_status
    ):  # Proceed only if deps OK and static tests passed
        try:
            # === Phase 1: Session Setup ===
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

            # === Phase 2: Get Target Info & Validate Config ===
            logger_sc.info("--- Phase 2: Get Target Info & Validate Config ---")
            target_tree_id = session_manager_sc.my_tree_id
            target_owner_name = session_manager_sc.tree_owner_name
            target_owner_profile_id = session_manager_sc.my_profile_id
            target_owner_global_id = session_manager_sc.my_uuid  # Get global ID here
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

            # Fetch details for the TESTING_PROFILE_ID to get a name for Discovery test target
            profile_response_details = None
            test_name_target_profile = "API Call: Get Target Profile Details (app-api)"
            if target_profile_id:
                api_call_lambda = lambda: _sc_get_profile_details(
                    cast(str, target_profile_id)
                )
                _, s2_api_stat, _ = _sc_run_test(
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
                        if s2_name_stat != "PASS":
                            overall_status = False
                        if target_name_from_profile not in [
                            "Unknown",
                            "Unknown Target",
                        ]:
                            # Set name for ladder test IF target person ID matches profile ID
                            if target_person_id_for_ladder == target_profile_id:
                                target_name_for_ladder = target_name_from_profile
                                logger_sc.info(
                                    f"Using profile name '{target_name_for_ladder}' for ladder test target (IDs match)."
                                )
                            else:
                                logger_sc.info(
                                    f"Using profile name '{target_name_from_profile}' for Discovery test target."
                                )
                    else:
                        logger_sc.error(
                            f"{test_name_target_profile} test passed but result was None/invalid."
                        )
                        overall_status = False
                        _sc_run_test(
                            "Check Target Name Found in API Resp",
                            lambda: "Skipped",
                            test_results_sc,
                            logger_sc,
                        )
                elif s2_api_stat == "SKIPPED":
                    logger_sc.warning(
                        f"{test_name_target_profile} skipped (likely API issue). Cannot check name."
                    )
                    _sc_run_test(
                        "Check Target Name Found in API Resp",
                        lambda: "Skipped",
                        test_results_sc,
                        logger_sc,
                    )
                else:
                    overall_status = False
                    logger_sc.error(f"{test_name_target_profile} failed.")
                    _sc_run_test(
                        "Check Target Name Found in API Resp",
                        lambda: "Skipped",
                        test_results_sc,
                        logger_sc,
                    )
            else:
                logger_sc.warning(
                    "Skipping Get Target Profile Details: TESTING_PROFILE_ID not set."
                )
                _sc_run_test(
                    test_name_target_profile,
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                )
                _sc_run_test(
                    "Check Target Name Found in API Resp",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                )
            if target_name_for_ladder == "Unknown Ladder Target":
                logger_sc.warning(
                    f"Using default name '{target_name_for_ladder}' for ladder target name (TESTING_PERSON_TREE_ID may differ from TESTING_PROFILE_ID or name lookup failed)."
                )

            # === Phase 3: Test parse_ancestry_person_details (Live & Static) ===
            # (Code remains the same - tests core parsing logic)
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
                    _, s3_facts_stat, _ = _sc_run_test(
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
                            if s3_keys_stat != "PASS":
                                overall_status = False
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
                                if s3_name_stat != "PASS":
                                    overall_status = False
                            else:
                                _sc_run_test(
                                    "Validation: Parsed Name Match (Live Facts)",
                                    lambda: "Skipped",
                                    test_results_sc,
                                    logger_sc,
                                )
                        else:
                            _sc_run_test(
                                f"{test_name_parse} (with Live Facts)",
                                lambda: False,
                                test_results_sc,
                                logger_sc,
                                expected_value="Parser returned None/invalid",
                            )
                            overall_status = False
                    else:
                        overall_status = False
                        _sc_run_test(
                            "Validation: Parsed Details Keys (Live Facts)",
                            lambda: "Skipped",
                            test_results_sc,
                            logger_sc,
                        )
                        _sc_run_test(
                            "Validation: Parsed Name Match (Live Facts)",
                            lambda: "Skipped",
                            test_results_sc,
                            logger_sc,
                        )
                except Exception as parse_e:
                    _sc_run_test(
                        f"{test_name_parse} (with Live Facts)",
                        lambda: False,
                        test_results_sc,
                        logger_sc,
                        expected_value=f"Exception: {parse_e}",
                    )
                    overall_status = False
            else:
                _sc_run_test(
                    f"{test_name_parse} (with Live Facts)",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                )
                _sc_run_test(
                    "Validation: Parsed Details Keys (Live Facts)",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                )
                _sc_run_test(
                    "Validation: Parsed Name Match (Live Facts)",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                )

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
                _, s3_suggest_stat, _ = _sc_run_test(
                    f"{test_name_parse} (Static Suggest Format)",
                    parse_lambda_suggest,
                    test_results_sc,
                    logger_sc,
                    expected_type=dict,
                )
                if s3_suggest_stat == "PASS":
                    parsed_details_suggest = parse_lambda_suggest()
                    if parsed_details_suggest:
                        vals_ok = all(
                            [
                                parsed_details_suggest.get("name")
                                == "Test Suggest Person",
                                parsed_details_suggest.get("birth_date") == "1950",
                                parsed_details_suggest.get("death_date") == "2000",
                                parsed_details_suggest.get("gender") == "F",
                                parsed_details_suggest.get("is_living") is False,
                                parsed_details_suggest.get("birth_place")
                                == "SuggestBirth Town",
                                parsed_details_suggest.get("death_place")
                                == "SuggestDeath City",
                                parsed_details_suggest.get("person_id") == "12345",
                                parsed_details_suggest.get("user_id") == "ABC-DEF",
                                "ABC-DEF/facts"
                                not in parsed_details_suggest.get("link", ""),
                                "/summary/ABC-DEF"
                                in parsed_details_suggest.get("link", ""),
                            ]
                        )
                        _, s3_val_stat, _ = _sc_run_test(
                            "Validation: Parsed Details Values (Static Suggest)",
                            lambda: vals_ok,
                            test_results_sc,
                            logger_sc,
                            expected_truthy=True,
                        )
                        if s3_val_stat != "PASS":
                            overall_status = False
                    else:
                        _sc_run_test(
                            f"{test_name_parse} (Static Suggest Format)",
                            lambda: False,
                            test_results_sc,
                            logger_sc,
                            expected_value="Parser returned None/invalid",
                        )
                        overall_status = False
                else:
                    overall_status = False
            except Exception as parse_e:
                _sc_run_test(
                    f"{test_name_parse} (Static Suggest Format)",
                    lambda: False,
                    test_results_sc,
                    logger_sc,
                    expected_value=f"Exception: {parse_e}",
                )
                overall_status = False

            # === Phase 4: Test API Helpers (Suggest, Facts User) ===
            # (Code remains the same - tests API call functions)
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
            else:
                _sc_run_test(
                    "API Helper: call_suggest_api",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                )

            facts_person_id = target_person_id_for_ladder
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
                if facts_status != "PASS":
                    overall_status = False
            else:
                logger_sc.warning(
                    "Skipping call_facts_user_api test: Missing TreeID, OwnerProfileID, or TargetPersonID (TESTING_PERSON_TREE_ID)."
                )
                _sc_run_test(
                    "API Helper: call_facts_user_api",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                )

            # === Phase 5: Test Relationship Ladder Parsing (Live HTML) ===
            logger_sc.info(
                "--- Phase 5: Test Relationship Ladder Parsing (Live HTML) ---"
            )
            test_target_person_id_ladder = target_person_id_for_ladder
            test_target_tree_id_ladder = target_tree_id
            test_owner_name_ladder = target_owner_name
            test_target_name_ladder = (
                target_name_for_ladder  # Use name derived in Phase 2 if possible
            )

            can_run_ladder_test_live = bool(
                test_owner_name_ladder
                and test_target_person_id_ladder
                and test_target_tree_id_ladder
                and BS4_AVAILABLE
            )
            test_name_ladder_api = "API Helper: call_getladder_api"
            test_name_format_ladder = (
                "Function Call: format_api_relationship_path (Live HTML)"
            )

            if not can_run_ladder_test_live:
                reason = "Missing BS4" if not BS4_AVAILABLE else "Missing IDs/OwnerName"
                logger_sc.warning(f"Skipping Live Ladder test: {reason}.")
                _sc_run_test(
                    test_name_ladder_api, lambda: "Skipped", test_results_sc, logger_sc
                )
                _sc_run_test(
                    test_name_format_ladder,
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                )
            else:
                # Use the self-check helper to call the ladder API function
                ladder_response_raw = _sc_get_tree_ladder(
                    cast(str, test_target_tree_id_ladder),
                    cast(str, test_target_person_id_ladder),
                )
                _, s5_api_status, _ = _sc_run_test(
                    test_name_ladder_api,
                    lambda r=ladder_response_raw: isinstance(r, str) and len(r) > 10,
                    test_results_sc,
                    logger_sc,
                    expected_truthy=True,
                )
                if s5_api_status == "PASS" and ladder_response_raw:
                    owner_name_str = cast(str, test_owner_name_ladder)
                    target_name_str = cast(str, test_target_name_ladder)
                    format_lambda = lambda: format_api_relationship_path(
                        ladder_response_raw, owner_name_str, target_name_str
                    )

                    # --- Define Expected Substrings for the Live HTML Format ---
                    # !! USER ACTION REQUIRED !!
                    # !! Update these substrings based on the KNOWN relationship
                    #    between your owner profile and TESTING_PERSON_TREE_ID !!
                    expected_substrings_ladder = [
                        f"{target_name_str} is {owner_name_str}'s uncle:",  # Example: Summary line for uncle
                        f"*   {target_name_str}",  # Example: Start person
                        "-> is brother's",  # Example: Intermediate relationship
                        "*   Derrick Wardie Gault",  # Example: Intermediate person name
                        "-> is father's",  # Example: Final relationship to owner
                        f"*   {owner_name_str} (You)",  # Example: Owner name
                    ]
                    logger_sc.info(
                        f"(Ladder HTML) Expecting relationship format containing substrings like: {expected_substrings_ladder}"
                    )

                    _, s5_format_status, _ = _sc_run_test(
                        test_name_format_ladder,
                        format_lambda,
                        test_results_sc,
                        logger_sc,
                        expected_type=str,
                        expected_contains=expected_substrings_ladder,
                    )
                    if s5_format_status != "PASS":
                        overall_status = False
                        actual_output = format_lambda()  # Log actual output on failure
                        logger_sc.error(
                            f"Ladder Relationship format mismatch. Expected substrings: {expected_substrings_ladder}\nActual Output:\n{actual_output}"
                        )
                else:
                    logger_sc.warning(
                        f"Skipping {test_name_format_ladder} because {test_name_ladder_api} failed or returned invalid data."
                    )
                    _sc_run_test(
                        test_name_format_ladder,
                        lambda: "Skipped",
                        test_results_sc,
                        logger_sc,
                    )
                    if s5_api_status == "FAIL":
                        overall_status = False  # Mark failure if API call failed

            # === Phase 5b: Test Relationship Discovery Parsing (Live JSON) ===
            logger_sc.info(
                "--- Phase 5b: Test Relationship Discovery Parsing (Live JSON) ---"
            )
            # Discovery Relationship API tests have been removed
            # We now only use the getladder API endpoint

            # Discovery Relationship API tests have been completely removed
            # We now only use the getladder API endpoint through call_getladder_api

        except RuntimeError as rt_err:
            logger_sc.critical(
                f"\n--- RUNTIME ERROR during self-check live tests: {rt_err} ---",
                exc_info=False,
            )
            _sc_run_test(
                "Self-Check Live Execution",
                lambda: False,
                test_results_sc,
                logger_sc,
                expected_value=f"RUNTIME ERROR: {rt_err}",
            )
            overall_status = False
        except Exception as e:
            logger_sc.critical(
                "\n--- UNEXPECTED EXCEPTION during self-check live tests ---",
                exc_info=True,
            )
            _sc_run_test(
                "Self-Check Live Execution",
                lambda: False,
                test_results_sc,
                logger_sc,
                expected_value="CRITICAL EXCEPTION OCCURRED",
            )
            overall_status = False
        finally:
            # === Phase 6: Session Teardown ===
            logger_sc.info("--- Phase 6: Finalizing - Closing Session ---")
            if session_manager_sc:
                try:
                    session_manager_sc.close_sess()
                    logger_sc.info("Session closed successfully.")
                    _sc_run_test(
                        "SessionManager.close_sess()",
                        lambda: True,
                        test_results_sc,
                        logger_sc,
                    )
                except Exception as close_err:
                    logger_sc.error(
                        f"Error closing session: {close_err}", exc_info=True
                    )
                    _sc_run_test(
                        "SessionManager.close_sess()",
                        lambda: False,
                        test_results_sc,
                        logger_sc,
                        expected_value=f"Exception: {close_err}",
                    )
                    overall_status = False
            else:
                logger_sc.info("No session object was initialized to close.")
                _sc_run_test(
                    "SessionManager.close_sess()",
                    lambda: "Skipped",
                    test_results_sc,
                    logger_sc,
                )

    else:  # Not can_run_live_tests or initial overall_status is False
        if not required_modules_ok:
            logger_sc.warning(
                "Skipping ALL Live API tests due to missing dependencies."
            )
        elif not overall_status:
            logger_sc.warning(
                "Skipping Live API tests due to prerequisite/static test failures."
            )
        else:
            logger_sc.warning(
                "Skipping Live API tests (reason not specified, check logs)."
            )
        # Mark all potentially live tests as skipped
        phases_to_skip = [
            "SessionManager.start_sess()",
            "SessionManager.ensure_session_ready()",
            "Check Target Tree ID Found",
            "Check Target Owner Name Found",
            "Check Target Owner Profile ID Found",
            "Check Target Owner Global ID (UUID) Found",
            "API Call: Get Target Profile Details (app-api)",
            "Check Target Name Found in API Resp",
            "Function Call: parse_ancestry_person_details() (with Live Facts)",
            "Validation: Parsed Details Keys (Live Facts)",
            "Validation: Parsed Name Match (Live Facts)",
            "API Helper: call_suggest_api",
            "API Helper: call_facts_user_api",
            "API Helper: call_getladder_api",
            "Function Call: format_api_relationship_path (Live HTML)",
            "SessionManager.close_sess()",
        ]
        existing_test_names = {name for name, _, _ in test_results_sc}
        for test_name in phases_to_skip:
            if test_name not in existing_test_names:
                _sc_run_test(test_name, lambda: "Skipped", test_results_sc, logger_sc)

    # --- Print Formatted Summary ---
    _sc_print_summary(test_results_sc, overall_status, logger_sc)

    # Determine final boolean result based on failures
    final_overall_status = overall_status and not any(
        status == "FAIL" for _, status, _ in test_results_sc
    )

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
    log_file = Path("api_utils_self_check.log").resolve()
    logger_standalone = None

    # --- Setup Logging for Standalone Execution ---
    try:
        import logging_config

        if not hasattr(logging_config, "setup_logging"):
            raise ImportError("setup_logging func missing")
        logger_standalone = logging_config.setup_logging(
            log_file=str(log_file), log_level="DEBUG"
        )
        print(f"Detailed logs (DEBUG level) will be written to: {log_file}")
        logger_standalone.info(
            f"Logging configured via logging_config.py to {log_file}"
        )
        logger_main_module = logging.getLogger("api_utils")
        if not logger_main_module.hasHandlers():
            for handler in logger_standalone.handlers:
                logger_main_module.addHandler(handler)
            logger_main_module.setLevel(logger_standalone.level)
    except ImportError as log_imp_err:
        print(
            f"Warning: logging_config unavailable ({log_imp_err}). Using basic file logging."
        )
        logging.basicConfig(
            level=logging.DEBUG,
            filename=log_file,
            filemode="w",
            format="%(asctime)s %(levelname)-8s [%(name)-15s] %(message)s",
        )
        logger_standalone = logging.getLogger("api_utils_standalone")
        logger_standalone.info(
            f"Using basicConfig for logging, writing DEBUG logs to {log_file}"
        )
        logger_main_module = logging.getLogger("api_utils")
        if not logger_main_module.hasHandlers():
            pass  # basicConfig usually handles this implicitly
        logger_main_module.setLevel(logging.DEBUG)
    except Exception as log_setup_err:
        print(
            f"Error setting up logging via logging_config: {log_setup_err}. Using basic file logging."
        )
        logging.basicConfig(
            level=logging.DEBUG,
            filename=log_file,
            filemode="w",
            format="%(asctime)s %(levelname)-8s [%(name)-15s] %(message)s",
        )
        logger_standalone = logging.getLogger("api_utils_standalone_error")
        logger_standalone.exception(
            "Error setting up logging_config, using basicConfig."
        )
        logger_main_module = logging.getLogger("api_utils")
        logger_main_module.setLevel(logging.DEBUG)

    # --- Configuration Checks ---
    config_ok_for_tests = False
    if CONFIG_AVAILABLE:
        test_person_id = getattr(config_instance, "TESTING_PERSON_TREE_ID", None)
        test_profile_id = getattr(config_instance, "TESTING_PROFILE_ID", None)
        if not test_person_id or not test_profile_id:
            print("\n" + "=" * 70)
            print(
                " WARNING: Configuration Incomplete for Full Self-Check ".center(
                    70, "="
                )
            )
            print("=".center(70, "="))
            if not test_person_id:
                print(
                    "- config.TESTING_PERSON_TREE_ID is not set (needed for Phase 4/5)."
                )
            if not test_profile_id:
                print(
                    "- config.TESTING_PROFILE_ID is not set (needed for Phase 2/3/5b)."
                )
            print("\nLive API tests requiring these IDs may be skipped or fail.")
            print("Ensure these are set in config.py for comprehensive testing.")
            print("=".ljust(70, "="))
            config_ok_for_tests = False
        else:
            config_ok_for_tests = True
            print(
                "\nConfiguration check: TESTING_PERSON_TREE_ID and TESTING_PROFILE_ID found."
            )
    else:
        print(
            "\nWARNING: config.py module not loaded. Cannot check for necessary test IDs."
        )

    # --- Execute Self-Check ---
    print("\nStarting self_check function...")
    self_check_passed = self_check()  # Run the main self-check function

    # --- Exit ---
    print("\napi_utils module self-check complete.")
    print("Import this module into other scripts to use its functions.")
    sys.exit(0 if self_check_passed else 1)  # Exit with 0 on success, 1 on failure
# End of __main__ block

# End of api_utils.py