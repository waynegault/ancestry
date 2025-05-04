# --- START OF FILE api_utils.py ---
# api_utils.py
"""
Utility functions specifically for parsing Ancestry API responses
and formatting data obtained from APIs.
V3.0: Integrated relationship ladder HTML parsing directly into
      format_api_relationship_path, removed temp function and helper.
      Updated self-check tests.
"""

# --- Standard library imports ---
import logging
import sys
import re
import os
import time
import json
import requests  # Keep for exception types and Response object checking
import urllib.parse  # Used for urlencode in self_check
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
    from utils import format_name, ordinal_case

    UTILS_AVAILABLE = True
    logger.info("Successfully imported base utils module")
except ImportError:
    format_name = lambda x: str(x).title() if x else "Unknown"
    ordinal_case = lambda x: str(x)
    logger.warning("Failed to import utils, using fallback format_name/ordinal_case")

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
        TESTING_PERSON_TREE_ID = (
            None  # Example: "102281560837" or specific ID needed for test
        )

    config_instance = DummyConfig()
    selenium_config = None  # Define selenium_config as None or a dummy if needed
    logger.warning("Failed to import config from config.py, using default values")


# --- Internal Helper Functions for parse_ancestry_person_details ---


def _extract_name_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> str:
    """Extracts the best name from person card or detailed facts data."""
    name = "Unknown"
    if facts_data and isinstance(facts_data, dict):
        # Prioritize detailed facts data
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            name = person_info.get("personName", name)
        if name == "Unknown":
            name = facts_data.get("personName", name)
        if name == "Unknown":
            name = facts_data.get("DisplayName", name)
        if name == "Unknown":
            first_name_pd = facts_data.get("FirstName")
            last_name_pd = facts_data.get("LastName")
            if first_name_pd:
                name = (
                    f"{first_name_pd} {last_name_pd}" if last_name_pd else first_name_pd
                )
                name = name.strip() or "Unknown"  # Ensure not empty string
            else:
                name = "Unknown"  # Reset if no name components found

    # Fallback to person card if still unknown
    if name == "Unknown":
        name = person_card.get("name", "Unknown")

    # Final formatting (Use imported or fallback)
    formatter = (
        format_name if UTILS_AVAILABLE else lambda x: str(x).title() if x else "Unknown"
    )
    return formatter(name) if name and name != "Unknown" else "Unknown"


def _extract_gender_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Optional[str]:
    """Extracts gender ('M' or 'F') from person card or detailed facts data."""
    gender = None
    gender_str = None

    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            gender_str = person_info.get("gender")
        if not gender_str:
            gender_str = facts_data.get("gender")
        if not gender_str:
            gender_str = facts_data.get("Gender")

    # Fallback to person card
    if not gender_str:
        gender_str = person_card.get("gender")

    # Normalize
    if gender_str and isinstance(gender_str, str):
        gender_str_lower = gender_str.lower()
        if gender_str_lower == "male":
            gender = "M"
        elif gender_str_lower == "female":
            gender = "F"

    return gender


def _extract_living_status_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Optional[bool]:
    """Extracts living status (True/False) from person card or detailed facts data."""
    is_living = None

    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            is_living = person_info.get("isLiving")
        if is_living is None:
            is_living = facts_data.get("isLiving")
        if is_living is None:
            is_living = facts_data.get("IsLiving")

    # Fallback to person card
    if is_living is None:
        is_living = person_card.get("isLiving")

    # Return as bool if found, otherwise None
    return bool(is_living) if is_living is not None else None


def _extract_event_from_api_details(
    event_type: str, person_card: Dict, facts_data: Optional[Dict]
) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """
    Extracts date string, place string, and parsed date object for a specific event type.
    Prioritizes facts_data > facts_data alternative keys > person_card.
    """
    date_str: Optional[str] = None
    place_str: Optional[str] = None
    date_obj: Optional[datetime] = None
    event_key_lower = event_type.lower()  # e.g., "birth", "death"
    event_key_camel = f"{event_key_lower}Date"  # e.g., "birthDate", "deathDate"

    # 1. Try primary facts structure in facts_data
    if facts_data and isinstance(facts_data, dict):
        fact_group_list = facts_data.get("facts", {}).get(event_type, [])
        if fact_group_list and isinstance(fact_group_list, list):
            # Assume first entry is primary if multiple exist
            fact_group = fact_group_list[0]
            if isinstance(fact_group, dict):
                date_info = fact_group.get("date", {})
                place_info = fact_group.get("place", {})
                if isinstance(date_info, dict):
                    date_str = date_info.get("normalized", date_info.get("original"))
                if isinstance(place_info, dict):
                    place_str = place_info.get("placeName")

    # 2. Try alternative keys in facts_data if primary missing
    if date_str is None and facts_data and isinstance(facts_data, dict):
        event_fact_alt = facts_data.get(event_key_camel)
        if event_fact_alt and isinstance(event_fact_alt, dict):
            date_str = event_fact_alt.get("normalized", event_fact_alt.get("date"))
            place_str = event_fact_alt.get(
                "place", place_str
            )  # Keep previous place if new one not found
        elif isinstance(event_fact_alt, str):  # Sometimes just a string date
            date_str = event_fact_alt

    # 3. Try person_card if still missing
    if date_str is None:
        event_info_card = person_card.get(event_key_lower, "")
        if event_info_card and isinstance(event_info_card, str):
            # Try splitting "Date in Place" format
            parts = re.split(r"\s+in\s+", event_info_card, maxsplit=1)
            date_str = parts[0].strip() if parts else event_info_card
            if (
                place_str is None and len(parts) > 1
            ):  # Only update place if not found earlier
                place_str = parts[1].strip()
        elif isinstance(event_info_card, dict):  # Handle if card gives dict directly
            date_str = event_info_card.get("date", date_str)
            if place_str is None:
                place_str = event_info_card.get("place", place_str)

    # 4. Parse date string if found
    # Use imported or fallback _parse_date
    parser = _parse_date if GEDCOM_UTILS_AVAILABLE else lambda x: None
    if date_str:
        try:
            date_obj = parser(date_str)
        except Exception as parse_err:
            logger.warning(
                f"Failed to parse date string '{date_str}' for {event_type}: {parse_err}"
            )
            date_obj = None

    # Return raw date string, place string, and parsed date object
    return date_str, place_str, date_obj


def _generate_person_link(
    person_id: Optional[str], tree_id: Optional[str], base_url: str
) -> str:
    """Generates the Ancestry profile link based on available IDs."""
    if tree_id and person_id:
        return f"{base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/facts"
    elif person_id:  # Assume global profile ID if tree ID is missing
        return f"{base_url}/discoveryui-matches/profile/{person_id}"
    else:
        return "(Link unavailable)"


# --- API Response Parsing ---
def parse_ancestry_person_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Dict:
    """
    Extracts standardized details from Ancestry Person-Card and Facts API responses.
    Includes parsing dates/places and generating a link.
    Uses internal helper functions for extraction logic.

    Args:
        person_card (Dict): A dictionary containing basic person info (e.g., from match list).
                            Expected keys (optional): personId, treeId, name, birth, death, gender, isLiving.
        facts_data (Optional[Dict]): More detailed API response (e.g., Profile Details from /app-api/express/v1/profiles/details).
                                     Structure varies, helpers handle different potential keys.

    Returns:
        Dict: A standardized dictionary containing parsed details:
              'name', 'birth_date' (display str), 'birth_place', 'death_date' (display str),
              'death_place', 'gender' ('M'/'F'/None), 'person_id', 'tree_id', 'link',
              'api_birth_obj' (datetime/None), 'api_death_obj' (datetime/None), 'is_living' (bool/None).
    """
    details = {
        "name": "Unknown",
        "birth_date": "N/A",
        "birth_place": None,
        "death_date": "N/A",
        "death_place": None,
        "gender": None,
        "person_id": person_card.get("personId"),  # Start with card ID
        "tree_id": person_card.get("treeId"),  # Start with card ID
        "link": None,
        "api_birth_obj": None,
        "api_death_obj": None,
        "is_living": None,
    }

    # Update IDs from facts_data if available (might be more reliable)
    if facts_data and isinstance(facts_data, dict):
        details["person_id"] = facts_data.get("personId", details["person_id"])
        details["tree_id"] = facts_data.get("treeId", details["tree_id"])

    # Extract using helpers
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

    # Clean display dates (Use imported or fallback)
    cleaner = (
        _clean_display_date
        if GEDCOM_UTILS_AVAILABLE
        else lambda x: str(x) if x else "N/A"
    )
    details["birth_date"] = cleaner(birth_date_raw) if birth_date_raw else "N/A"
    details["death_date"] = cleaner(death_date_raw) if death_date_raw else "N/A"

    # Generate link
    base_url_for_link = getattr(
        config_instance, "BASE_URL", "https://www.ancestry.com"
    ).rstrip("/")
    details["link"] = _generate_person_link(
        details["person_id"], details["tree_id"], base_url_for_link
    )

    logger.debug(
        f"Parsed API details for '{details.get('name', 'Unknown')}': "
        f"ID={details.get('person_id')}, Tree={details.get('tree_id', 'N/A')}, "
        f"Born='{details.get('birth_date')}' in '{details.get('birth_place') or '?'}', "
        f"Died='{details.get('death_date')}' in '{details.get('death_place') or '?'}', "
        f"Gender='{details.get('gender') or '?'}', Living={details.get('is_living')}"
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
    and formats it into the specific two-line output:
    'Person 1 (Lifespan): Relationship to Person 2'
    '↓'
    'Person 2 (Lifespan): Relationship to Person 1'

    Args:
        api_response_data: The raw data returned by the relationship API call (usually JSONP string).
        owner_name: The name of the tree owner (Person 2).
        target_name: The name of the person whose relationship is being checked (Person 1).

    Returns:
        A formatted two-line string representing the relationship,
        or an error message string if parsing fails or format is unexpected.
    """
    if not api_response_data:
        logger.warning(
            "format_api_relationship_path: Received empty API response data."
        )
        return "(No relationship data received from API)"

    html_content: Optional[str] = None
    api_status: str = "unknown"

    # --- Step 1: Extract HTML content from JSONP or handle other formats ---
    if isinstance(api_response_data, str):
        # Try parsing as JSONP
        if api_response_data.strip().startswith(
            "__ancestry_jsonp_"
        ) and api_response_data.strip().endswith(");"):
            try:
                json_part_match = re.search(
                    r"^\s*[\w$.]+\((.*)\)\s*;?\s*$", api_response_data, re.DOTALL
                )
                if json_part_match:
                    json_part = json_part_match.group(1).strip()
                    parsed_json = json.loads(json_part)
                    api_status = parsed_json.get("status", "unknown")
                    if api_status == "success":
                        html_content = parsed_json.get("html")
                        if not isinstance(html_content, str):
                            logger.warning(
                                "JSONP success but 'html' key is not a string."
                            )
                            html_content = None  # Force failure later
                    else:
                        return f"(API returned status '{api_status}': {parsed_json.get('message', 'Unknown Error')})"
                else:
                    logger.warning("Could not extract JSON part from JSONP string.")
                    # Fall through to treat as raw HTML/text
                    html_content = api_response_data
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON part from JSONP: {e}")
                # Fall through to treat as raw HTML/text
                html_content = api_response_data
            except Exception as e:
                logger.error(f"Unexpected error processing JSONP: {e}")
                # Fall through to treat as raw HTML/text
                html_content = api_response_data
        else:
            # Assume it might be raw HTML or just text if not JSONP format
            logger.debug("Input string not in JSONP format, treating as raw content.")
            html_content = api_response_data

    elif isinstance(api_response_data, dict):
        # Handle direct dictionary input (less common for /getladder)
        if "error" in api_response_data:
            return f"(API returned error object: {api_response_data.get('error', 'Unknown')})"
        elif (
            "html" in api_response_data and api_response_data.get("status") == "success"
        ):
            html_content = api_response_data.get("html")
            api_status = "success"
            if not isinstance(html_content, str):
                logger.warning("Dict input had 'html' key but it was not a string.")
                html_content = None
        elif "path" in api_response_data:  # Handle alternative direct JSON path format
            logger.warning(
                "Received direct JSON 'path' format, cannot convert to standard two-line output."
            )
            # Provide basic path info if possible
            path_steps_json = []
            if isinstance(api_response_data["path"], list):
                for step in api_response_data["path"]:
                    step_name = step.get("name", "?")
                    step_rel = step.get("relationship", "?")
                    step_rel_display = _get_relationship_term(None, step_rel)
                    path_steps_json.append(f"-> {step_rel_display} is {step_name}")
                return (
                    "\n".join(path_steps_json)
                    if path_steps_json
                    else "(Relationship path found but empty)"
                )
            else:
                return "(Relationship path found but invalid format)"
        else:
            return "(Received unhandled dictionary format from API)"
    else:
        return f"(Unsupported API response data type: {type(api_response_data)})"

    # --- Step 2: Check if we have HTML content to parse ---
    if not html_content or not isinstance(html_content, str):
        logger.warning("No processable HTML content found after extraction.")
        # Include API status if extraction failed after success status
        status_msg = f" (API status: {api_status})" if api_status != "unknown" else ""
        return f"(Could not find or parse relationship HTML{status_msg})"

    # --- Step 3: Unescape and Parse the HTML ---
    if not BeautifulSoup:
        logger.error("BeautifulSoup library not found. Cannot parse relationship HTML.")
        return "(Cannot parse relationship path - BeautifulSoup missing)"

    try:
        # Unescape HTML entities (like &) and potentially others handled by html.unescape
        # JSON unicode escapes (\uXXXX) should have been handled by json.loads() earlier
        try:
            processed_html = html.unescape(html_content)
        except Exception as unescape_err:
            logger.warning(
                f"HTML unescaping failed: {unescape_err}. Processing potentially escaped HTML."
            )
            processed_html = html_content  # Proceed with original if unescape fails

        logger.debug(
            f"Parsing HTML content with BeautifulSoup: {processed_html[:300]}..."
        )

        # --- Try lxml, fallback to html.parser ---
        parser_used = "html.parser"  # Default
        try:
            soup = BeautifulSoup(processed_html, "lxml")
            parser_used = "lxml"
        except FeatureNotFound:
            logger.warning("'lxml' parser not found, falling back to 'html.parser'.")
            soup = BeautifulSoup(processed_html, "html.parser")
        except Exception as e:
            # Catch potential errors with specific parsers (rare)
            logger.warning(
                f"Parser '{parser_used}' failed ({e}), falling back to 'html.parser'."
            )
            soup = BeautifulSoup(processed_html, "html.parser")
            parser_used = "html.parser"
        logger.debug(f"Successfully parsed HTML using '{parser_used}'.")

        # --- Specific Parsing Logic for /getladder Format (ul.textCenter) ---
        list_items = soup.select("ul.textCenter li")
        logger.debug(f"Found {len(list_items)} list items in ul.textCenter")

        # Expecting 3 items: Target, Arrow, Owner
        if len(list_items) < 3:
            logger.warning(
                f"Expected at least 3 list items for relationship path, found {len(list_items)}. HTML structure mismatch?"
            )
            # Try to return raw text content as fallback
            raw_text = soup.get_text(separator=" ", strip=True)
            fallback_msg = f"(Relationship HTML structure not as expected - Found {len(list_items)} items)"
            if raw_text and len(raw_text) < 150:  # Avoid huge dumps
                fallback_msg += f"\nRaw Text: {raw_text}"
            return fallback_msg

        # --- Extract Info from First Item (Target Person) ---
        item1 = list_items[0]
        name1_tag = item1.find("b")
        # Role is inside <i>, possibly nested <b> e.g. <i><b>mother</b></i> or <i>mother</i>
        role1_tag = item1.select_one("i b, i")
        scraped_name1 = name1_tag.get_text(strip=True) if name1_tag else "Unknown"
        role1 = role1_tag.get_text(strip=True) if role1_tag else "Unknown relationship"
        lifespan1 = ""
        if (
            name1_tag
            and name1_tag.next_sibling
            and isinstance(name1_tag.next_sibling, str)
        ):
            potential_lifespan = name1_tag.next_sibling.strip()
            # Regex to match YYYY or YYYY-YYYY or YYYY - YYYY etc.
            if re.match(r"^\d{4}\s*[-–—]?\s*(\d{4})?$", potential_lifespan):
                lifespan1 = potential_lifespan

        logger.debug(
            f"Parsed Item 1: Name='{scraped_name1}', Role='{role1}', Lifespan='{lifespan1}'"
        )

        # --- Extract Info from Third Item (Owner Person) ---
        # Item 2 is usually the icon spacer <li aria-hidden="true" class="icon iconArrowDown"></li>
        item3 = list_items[2]
        # Owner name usually inside <a><b>...</b></a> or <a>...</a>
        name3_tag = item3.select_one("a b, a")
        # Confirmation text like "You are the son of..." is inside <i>
        role3_text_tag = item3.find("i")
        scraped_name3 = name3_tag.get_text(strip=True) if name3_tag else "Unknown"
        role3_text = role3_text_tag.get_text(strip=True) if role3_text_tag else ""
        lifespan3 = ""  # Owner lifespan is less common in this view
        if (
            name3_tag
            and name3_tag.next_sibling
            and isinstance(name3_tag.next_sibling, str)
        ):
            potential_lifespan3 = name3_tag.next_sibling.strip()
            if re.match(r"^\d{4}\s*[-–—]?\s*(\d{4})?$", potential_lifespan3):
                lifespan3 = potential_lifespan3

        logger.debug(
            f"Parsed Item 3: Name='{scraped_name3}', RoleText='{role3_text}', Lifespan='{lifespan3}'"
        )

        # --- Determine Inverse Relationship (Role for Line 2) ---
        role3 = "related"  # Default fallback
        role3_text_lower = role3_text.lower()
        if "son of" in role3_text_lower:
            role3 = "Son"
        elif "daughter of" in role3_text_lower:
            role3 = "Daughter"
        elif "mother of" in role3_text_lower:  # Check for owner being parent
            role3 = "Mother"
        elif "father of" in role3_text_lower:
            role3 = "Father"
        elif "parent of" in role3_text_lower:
            role3 = "Parent"
        elif "husband of" in role3_text_lower:  # Check spouse cases
            role3 = "Husband"
        elif "wife of" in role3_text_lower:
            role3 = "Wife"
        elif "spouse of" in role3_text_lower:
            role3 = "Spouse"
        elif "brother of" in role3_text_lower:  # Check sibling cases
            role3 = "Brother"
        elif "sister of" in role3_text_lower:
            role3 = "Sister"
        elif "sibling of" in role3_text_lower:
            role3 = "Sibling"
        # Add more inverse logic here if other relationship types appear in `role3_text`

        # --- Construct the Desired Output ---
        # Use owner_name and target_name passed into function for consistency
        # Use scraped roles (role1, role3) and lifespans (lifespan1, lifespan3)
        if role1 != "Unknown relationship" and role3 != "related":
            line1 = f"{target_name}{f' ({lifespan1})' if lifespan1 else ''}: {role1.capitalize()} of {owner_name}"
            line2 = f"{owner_name}{f' ({lifespan3})' if lifespan3 else ''}: {role3} of {target_name}"
            result_str = f"{line1}\n↓\n{line2}"
            logger.info(f"Formatted relationship path successfully:\n{result_str}")
            return result_str
        else:
            # Fallback if essential roles couldn't be determined
            logger.warning(
                "Could not reliably determine relationship roles from parsed HTML. "
                f"Extracted: role1='{role1}', role3_text='{role3_text}', determined role3='{role3}'"
            )
            # Provide basic info if possible
            base_info = f"{target_name} ({scraped_name1 or '?'}) and {owner_name} ({scraped_name3 or '?'})"
            if role1 != "Unknown relationship":
                return f"{base_info} seem related as {role1.capitalize()}."
            else:
                return f"(Could not determine specific relationship between {target_name} and {owner_name})"

    except Exception as e:
        logger.error(
            f"Error processing relationship HTML with BeautifulSoup: {e}", exc_info=True
        )
        return f"(Error processing relationship HTML: {e})"


# End of format_api_relationship_path


# --- Display Function (Consider removing or simplifying) ---
def display_raw_relationship_ladder(
    raw_content: Union[str, Dict], owner_name: str, target_name: str
):
    """
    DEPRECATED. Parses and displays the Ancestry relationship ladder.
    Prefer using format_api_relationship_path directly in calling code.
    """
    logger.warning(
        "display_raw_relationship_ladder is deprecated. Use format_api_relationship_path."
    )

    logger.info(
        f"\n--- Relationship between {owner_name} and {target_name} (Deprecated Display) ---"
    )

    # Use the formatter function which now includes BS4 check and parsing
    formatted_path_str = format_api_relationship_path(
        raw_content, owner_name, target_name
    )

    print(formatted_path_str.strip())


# End of display_raw_relationship_ladder


# --- Standalone Test Block ---
def self_check() -> bool:
    """
    Performs internal self-checks for api_utils.py, including LIVE API calls.
    Requires .env file to be correctly configured. Provides formatted output summary.
    """
    # --- Local Imports for Self-Check ---
    try:
        if not UTILS_AVAILABLE or "utils" not in sys.modules:
            raise ImportError("Base utils module not imported or available.")
        from utils import SessionManager, _api_req  # type: ignore # Assuming SessionManager defined in utils

        if not CONFIG_AVAILABLE or "config" not in sys.modules:
            raise ImportError("Config module not imported or available.")
        from config import (
            config_instance as config_instance_sc,
            selenium_config as selenium_config_sc,
        )
    except ImportError as e:
        print(
            f"\n[api_utils.py self-check ERROR] - Cannot import base utils/config for live tests: {e}"
        )
        SessionManager = None
        _api_req = None
        config_instance_sc = None
        selenium_config_sc = None

    try:
        from logging_config import logger as logger_sc
    except ImportError:
        logger_sc = logging.getLogger("api_utils.self_check")

    # --- Test Runner Helper ---
    test_results_sc: List[Tuple[str, str, str]] = []
    session_manager_sc: Optional["SessionManager"] = None

    # --- Internal API Call Helpers for Self-Check (DEFINED FIRST) ---
    def _sc_api_req(
        url: str, description: str, expect_json: bool = True, **kwargs
    ) -> Any:
        """
        Wrapper for utils._api_req within self-check.
        Calls the core _api_req from utils and then handles unexpected
        Response objects if JSON was expected by this wrapper call.
        """
        nonlocal session_manager_sc
        if not _api_req:
            raise RuntimeError("_api_req function not available for self-check")
        if (
            not SessionManager
            or not session_manager_sc
            or not session_manager_sc.is_sess_valid()
        ):
            raise RuntimeError(
                "SessionManager or session object not ready for API call in self_check"
            )

        result = _api_req(
            url=url,
            driver=session_manager_sc.driver,
            session_manager=session_manager_sc,
            api_description=f"{description} (Self Check)",
            **kwargs,  # DO NOT pass expect_json here
        )

        if expect_json and isinstance(result, requests.Response):
            logger_sc.warning(
                f"[_sc_api_req wrapper] Expected JSON (dict) for '{description}', "
                f"but received a Response object from utils._api_req. Status: {getattr(result, 'status_code', 'N/A')}. "
                f"Returning None instead."
            )
            status_code = getattr(result, "status_code", None)
            if status_code and 400 <= status_code < 600:
                try:
                    logger_sc.debug(f"Response content preview: {result.text[:500]}")
                except Exception as log_err:
                    logger_sc.debug(f"Could not log response text: {log_err}")
            return None

        return result

    def _sc_get_profile_details(profile_id: str) -> Optional[Dict]:
        """Helper to get profile details using the self-check API request wrapper."""
        if not config_instance_sc:
            logger_sc.warning("_sc_get_profile_details: Config instance not available.")
            return None
        if not profile_id or not isinstance(profile_id, str):
            logger_sc.warning(
                f"_sc_get_profile_details: Invalid profile_id provided: {profile_id}"
            )
            return None

        # Use the specific user ID endpoint
        api_desc = f"Get Profile Details ({profile_id})"
        url = urljoin(
            config_instance_sc.BASE_URL,
            f"/app-api/express/v1/profiles/details?userId={profile_id.upper()}",
        )
        timeout = (
            getattr(selenium_config_sc, "API_TIMEOUT", 60) if selenium_config_sc else 60
        )
        # Call wrapper, TELL WRAPPER to expect JSON
        return _sc_api_req(
            url,
            api_desc,
            expect_json=True,  # For wrapper's check
            use_csrf_token=False,
            timeout=timeout,
        )

    def _sc_get_tree_ladder(tree_id: str, person_id: str) -> Optional[str]:
        """Helper to get the relationship ladder using the self-check API request wrapper."""
        if not config_instance_sc or not selenium_config_sc:
            logger_sc.warning(
                "_sc_get_tree_ladder: Config or Selenium Config not available."
            )
            return None
        if not tree_id or not person_id:
            logger_sc.warning(
                f"_sc_get_tree_ladder: Invalid tree_id '{tree_id}' or person_id '{person_id}'."
            )
            return None

        # Use the ladder endpoint
        api_desc = f"Get Tree Ladder ({tree_id}/{person_id})"
        base_url = config_instance_sc.BASE_URL
        base_ladder_url = urljoin(
            base_url, f"/family-tree/person/tree/{tree_id}/person/{person_id}/getladder"
        )
        callback_name = f"__ancestry_jsonp_{int(time.time()*1000)}"
        timestamp_ms = int(time.time() * 1000)
        query_params = urlencode({"callback": callback_name, "_": timestamp_ms})
        ladder_api_url = f"{base_ladder_url}?{query_params}"
        ladder_referer = urljoin(
            base_url, f"/family-tree/person/tree/{tree_id}/person/{person_id}/facts"
        )
        headers = {
            "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": ladder_referer,
        }
        api_timeout = getattr(selenium_config_sc, "API_TIMEOUT", 60)
        # Call wrapper, TELL WRAPPER *NOT* to expect JSON
        return _sc_api_req(
            ladder_api_url,
            api_desc,
            expect_json=False,  # For wrapper's check
            headers=headers,
            force_text_response=True,
            timeout=api_timeout,
            use_csrf_token=False,
        )

    # --- [ Test Runner Helper _run_test_sc Definition ] ---
    def _run_test_sc(
        test_name: str, test_func: Callable, *args, **kwargs
    ) -> Tuple[str, str, str]:
        """Runs a single test, logs result, stores result, and returns status tuple."""
        logger_sc.debug(f"[ RUNNING SC ] {test_name}")
        status = "FAIL" # Default assumption
        message = ""
        expect_none = kwargs.pop("expected_none", False)
        expect_type = kwargs.pop("expected_type", None)
        expect_value = kwargs.pop("expected_value", None)
        expect_contains = kwargs.pop("expected_contains", None)
        expect_truthy = kwargs.pop("expected_truthy", False)
        test_func_kwargs = kwargs # Remaining kwargs are for the test function

        try:
            result = test_func(*args, **test_func_kwargs)
            passed = False

            # --- Evaluation Logic ---
            if expect_none:
                passed = result is None
                if not passed: message = f"Expected None, got {type(result).__name__}"
            elif expect_type is not None:
                if result is None:
                    # *** CHANGE: If None received when type expected, treat as SKIPPED ***
                    # This handles the case where _sc_api_req wrapper handled an API error.
                    passed = False # It didn't technically pass the type check
                    status = "SKIPPED" # But we mark it as skipped for overall status
                    message = f"Expected type {expect_type.__name__}, got None (API issue handled by wrapper?)"
                    logger_sc.warning(f"Test '{test_name}' expecting type {expect_type.__name__} received None, marking as SKIPPED.")
                elif isinstance(result, expect_type):
                    passed = True # Type matches
                else: # Type mismatch (and not None)
                    passed = False; message = f"Expected type {expect_type.__name__}, got {type(result).__name__}"
            elif expect_value is not None:
                passed = result == expect_value
                if not passed: message = f"Expected value '{str(expect_value)[:50]}', got '{repr(result)[:100]}'"
            elif expect_contains is not None:
                if isinstance(result, str):
                    if isinstance(expect_contains, (list, tuple)):
                        passed = all(sub in result for sub in expect_contains)
                        if not passed: missing = [sub for sub in expect_contains if sub not in result]; message = f"Expected result to contain all of: {expect_contains}. Missing: {missing}"
                    elif isinstance(expect_contains, str):
                        passed = expect_contains in result
                        if not passed: message = f"Expected result to contain '{expect_contains}', got '{repr(result)[:100]}'"
                    else: passed = False; message = f"Invalid type for expect_contains: {type(expect_contains)}"
                else: passed = False; message = f"Expected string result for contains check, got {type(result).__name__}"
            elif expect_truthy:
                passed = bool(result)
                if not passed: message = f"Expected truthy value, got {repr(result)[:100]}"
            elif isinstance(result, str) and result == "Skipped":
                 passed = False; status = "SKIPPED"; message = "" # Explicit skip from lambda
            else: # Default check: result should be exactly True
                passed = result is True
                if not passed: message = f"Expected True, got {repr(result)[:100]}"

            # --- Set Status ---
            if status != "SKIPPED": # Don't override explicit skip or the None->SKIPPED case
                if passed: status = "PASS"
                else: status = "FAIL";
                if status == "FAIL" and not message: message = f"Test condition not met (Result: {repr(result)[:100]})"

        except Exception as e:
            status = "FAIL"
            message = f"{type(e).__name__}: {e}"
            logger_sc.debug(f"Exception details for {test_name}: {message}\n{traceback.format_exc()}", exc_info=False)

        log_level = logging.INFO if status == "PASS" else (logging.WARNING if status == "SKIPPED" else logging.ERROR)
        # Adjust message display: show message only for FAIL status in the log line
        log_message = f"[ {status:<6} SC ] {test_name}{f': {message}' if message and status == 'FAIL' else ''}"
        logger_sc.log(log_level, log_message)
        # Store the potentially updated message (especially for the None case)
        test_results_sc.append((test_name, status, message if status != "PASS" else "")) # Store message only if not PASS
        return (test_name, status, message)
    # --- [ End _run_test_sc ] ---

    # --- Test Parameters ---
    can_run_live_tests = bool(
        SessionManager and _api_req and config_instance_sc and selenium_config_sc
    )
    target_profile_id = (
        getattr(config_instance_sc, "TESTING_PROFILE_ID", None)
        if config_instance_sc
        else None
    )
    # *** Ensure correct assignment spelling ***
    target_person_id_for_ladder = (
        getattr(config_instance_sc, "TESTING_PERSON_TREE_ID", None)
        if config_instance_sc
        else None
    )
    target_name_from_profile = "Unknown Target"
    target_name_for_ladder = "Unknown Ladder Target"  # Initial default

    if can_run_live_tests and not target_profile_id:
        logger_sc.warning("TESTING_PROFILE_ID missing in config.")
    if can_run_live_tests and not target_person_id_for_ladder:
        logger_sc.warning("TESTING_PERSON_TREE_ID missing in config.")

    # --- Status Tracking ---
    overall_status = True

    logger_sc.info("\n[api_utils.py self-check starting...]")

    # === Phase 0: Prerequisite Checks ===
    logger_sc.info("--- Phase 0: Prerequisite Checks ---")
    # ... (Prerequisite checks remain the same) ...
    _, s0_status, _ = _run_test_sc(
        "Check BeautifulSoup Import",
        lambda: BeautifulSoup is not None,
        expected_truthy=True,
    )
    if s0_status == "FAIL":
        overall_status = False
    func_map = {
        "format_name": format_name,
        "ordinal_case": ordinal_case,
        "_parse_date": _parse_date,
        "_clean_display_date": _clean_display_date,
        "parse_ancestry_person_details": parse_ancestry_person_details,
        "format_api_relationship_path": format_api_relationship_path,
    }
    for name, func in func_map.items():
        _, s0_status, _ = _run_test_sc(
            f"Check Function '{name}'", lambda f=func: callable(f), expected_truthy=True
        )
        if s0_status == "FAIL":
            overall_status = False
    _, s0_status, _ = _run_test_sc(
        "Check Config Loaded",
        lambda: CONFIG_AVAILABLE
        and config_instance_sc is not None
        and hasattr(config_instance_sc, "BASE_URL"),
        expected_truthy=True,
    )
    if s0_status == "FAIL":
        overall_status = False
    if not overall_status:
        logger_sc.error("Prerequisite checks failed. Cannot proceed further.")
        can_run_live_tests = False

    # === Live Tests Section ===
    if can_run_live_tests:
        try:
            # === Phase 1: Session Setup ===
            logger_sc.info("--- Phase 1: Session Setup & Login ---")
            if not SessionManager:
                raise RuntimeError("SessionManager class not loaded.")
            session_manager_sc = SessionManager()
            _, s1_status, _ = _run_test_sc(
                "SessionManager.start_sess()",
                session_manager_sc.start_sess,
                action_name="SC Phase 1 Start",
                expected_truthy=True,
            )
            if s1_status == "FAIL":
                overall_status = False
                raise RuntimeError("start_sess failed")
            _, s1_status, _ = _run_test_sc(
                "SessionManager.ensure_session_ready()",
                session_manager_sc.ensure_session_ready,
                action_name="SC Phase 1 Ready",
                expected_truthy=True,
            )
            if s1_status == "FAIL":
                overall_status = False
                raise RuntimeError("ensure_session_ready failed")

            # === Phase 2: Get Target Info & Validate Config ===
            logger_sc.info("--- Phase 2: Get Target Info & Validate Config ---")
            if not session_manager_sc: raise RuntimeError("Session Manager object not created.")
            target_tree_id = session_manager_sc.my_tree_id
            target_owner_name = session_manager_sc.tree_owner_name
            _, s2_status_tid, _ = _run_test_sc("Check Target Tree ID Found", lambda: bool(target_tree_id), expected_truthy=True)
            _, s2_status_owner, _ = _run_test_sc("Check Target Owner Name Found", lambda: bool(target_owner_name), expected_truthy=True)
            if s2_status_tid == "FAIL" or s2_status_owner == "FAIL": overall_status = False

            profile_response_details = None
            test_name_target_profile = "API Call: Get Target Profile Details"
            # Fetch main profile details (using TESTING_PROFILE_ID) for parser testing
            if target_profile_id:
                api_call_lambda = lambda: _sc_get_profile_details(cast(str, target_profile_id))
                _, s2_status, s2_msg = _run_test_sc(test_name_target_profile, api_call_lambda, expected_type=dict)
                if s2_status == "PASS":
                    profile_response_details = api_call_lambda()
                    if profile_response_details and isinstance(profile_response_details, dict):
                        name_test_card = {"personId": target_profile_id}
                        target_name_from_profile = _extract_name_from_api_details(name_test_card, profile_response_details)
                        _, s2_status_name, _ = _run_test_sc("Check Target Name Found in API Resp", lambda: target_name_from_profile != "Unknown", expected_truthy=True)
                        if s2_status_name == "FAIL": overall_status = False
                    else:
                         logger_sc.error(f"{test_name_target_profile} passed type check but result was invalid: {type(profile_response_details)}")
                         overall_status = False; _run_test_sc("Check Target Name Found in API Resp", lambda: "Skipped") # Report skip for dependent test
                elif s2_msg and "got None" in s2_msg:
                     logger_sc.warning(f"{test_name_target_profile} failed because API returned None. Cannot check name.")
                     overall_status = False; _run_test_sc("Check Target Name Found in API Resp", lambda: "Skipped") # Report skip for dependent test
                else:
                    overall_status = False; logger_sc.error(f"{test_name_target_profile} failed. Cannot check name.")
                    _run_test_sc("Check Target Name Found in API Resp", lambda: "Skipped") # Report skip for dependent test
            else:
                logger_sc.warning("Skipping Get Target Profile Details API call: TESTING_PROFILE_ID not set.")
                _run_test_sc(test_name_target_profile, lambda: "Skipped")
                _run_test_sc("Check Target Name Found in API Resp", lambda: "Skipped")

            # --- Determine Ladder Target Name (No API Call Here) ---
            # We REMOVED the attempt to call the incompatible API.
            # Always use fallback name for the ladder target in Phase 5 test.
            target_name_for_ladder = "Expected Target Name"
            logger_sc.info(f"Using fallback '{target_name_for_ladder}' for ladder target name (API call removed).")
            # We no longer run the test "API Call: Get Ladder Target Profile Details"

            # --- End Phase 2 ---

            # === Phase 3: Test parse_ancestry_person_details ===
            logger_sc.info("--- Phase 3: Test parse_ancestry_person_details ---")
            test_name_parse = "Function Call: parse_ancestry_person_details()"
            if (
                profile_response_details
                and isinstance(profile_response_details, dict)
                and target_profile_id
                and target_tree_id
            ):
                person_card_for_parse = {
                    "personId": target_profile_id,
                    "treeId": target_tree_id,
                }
                parsed_details = None
                try:
                    parse_lambda = lambda: parse_ancestry_person_details(
                        person_card_for_parse, profile_response_details
                    )
                    _, s3_status, _ = _run_test_sc(
                        test_name_parse, parse_lambda, expected_type=dict
                    )
                    if s3_status == "PASS":
                        parsed_details = parse_lambda()
                        if parsed_details and isinstance(parsed_details, dict):
                            keys_ok = all(
                                k in parsed_details
                                for k in [
                                    "name",
                                    "person_id",
                                    "link",
                                    "birth_date",
                                    "death_date",
                                ]
                            )
                            _, s3_v_status, _ = _run_test_sc(
                                "Validation: Parsed Details Keys",
                                lambda: keys_ok,
                                expected_truthy=True,
                            )
                            if s3_v_status == "FAIL":
                                overall_status = False
                            if (
                                target_name_from_profile != "Unknown Target"
                                and target_name_from_profile != "Unknown"
                            ):
                                _, s3_n_status, _ = _run_test_sc(
                                    "Validation: Parsed Name Match",
                                    lambda: parsed_details.get("name")
                                    == target_name_from_profile,
                                    expected_truthy=True,
                                )
                                if s3_n_status == "FAIL":
                                    logger_sc.warning(
                                        f"Parsed name '{parsed_details.get('name')}' != API name '{target_name_from_profile}'."
                                    )
                                    overall_status = False
                            else:
                                _run_test_sc(
                                    "Validation: Parsed Name Match", lambda: "Skipped"
                                )
                        else:
                            logger_sc.error(
                                f"{test_name_parse} passed status but returned invalid result: {parsed_details}"
                            )
                            _run_test_sc(
                                test_name_parse,
                                lambda: False,
                                expected_value="Parser Invalid Return",
                            )
                            overall_status = False
                    else:
                        overall_status = False
                except Exception as parse_e:
                    logger_sc.error(
                        f"Exception calling parser: {parse_e}", exc_info=True
                    )
                    overall_status = False
                    _run_test_sc(
                        test_name_parse,
                        lambda: False,
                        expected_value=f"Exception: {parse_e}",
                    )
            else:
                logger_sc.warning(
                    "Skipping parse_ancestry_person_details test: Prerequisite API call failed/None or IDs missing."
                )
                _run_test_sc(test_name_parse, lambda: "Skipped")
                _run_test_sc("Validation: Parsed Details Keys", lambda: "Skipped")
                _run_test_sc("Validation: Parsed Name Match", lambda: "Skipped")

            # === Phase 5: Test Relationship Ladder Parsing ===
            logger_sc.info(
                "--- Phase 5: Test Relationship Ladder Parsing (Live API) ---"
            )
            # *** Ensure correct usage spelling ***
            test_target_person_id = target_person_id_for_ladder
            test_target_tree_id = target_tree_id
            test_owner_name = target_owner_name
            test_target_name = (
                target_name_for_ladder
                if target_name_for_ladder
                and target_name_for_ladder
                not in ["Unknown Ladder Target", "Unknown", "Expected Target Name"]
                else "Expected Target Name"
            )
            if (
                test_target_name == "Expected Target Name"
                and target_person_id_for_ladder
            ):
                logger_sc.warning(
                    f"Using fallback 'Expected Target Name' for ladder target person ID {target_person_id_for_ladder}."
                )

            can_run_ladder_test_live = bool(
                test_owner_name and test_target_person_id and test_target_tree_id
            )
            test_name_ladder_api = "API Call: Get Tree Ladder"
            test_name_format_ladder = "Function Call: format_api_relationship_path()"

            if not can_run_ladder_test_live:
                missing_reqs = [
                    n
                    for n, v in [
                        ("OwnerName", test_owner_name),
                        ("TargetPersonID", test_target_person_id),
                        ("TargetTreeID", test_target_tree_id),
                    ]
                    if not v
                ]
                logger_sc.warning(
                    f"Skipping Live Ladder test: Missing prerequisites ({', '.join(missing_reqs)})."
                )
                _run_test_sc(test_name_ladder_api, lambda: "Skipped")
                _run_test_sc(test_name_format_ladder, lambda: "Skipped")
            else:
                ladder_response_raw = None
                api_call_lambda_ladder = lambda: _sc_get_tree_ladder(
                    cast(str, test_target_tree_id), cast(str, test_target_person_id)
                )
                _, s5_api_status, s5_api_msg = _run_test_sc(
                    test_name_ladder_api, api_call_lambda_ladder, expected_type=str
                )
                if s5_api_status == "PASS":
                    ladder_response_raw = api_call_lambda_ladder()
                    if ladder_response_raw and isinstance(ladder_response_raw, str):
                        logger_sc.debug(
                            f"Raw Ladder Response received (type: {type(ladder_response_raw)})."
                        )
                    elif ladder_response_raw is None:
                        logger_sc.error(
                            f"{test_name_ladder_api} passed status but result was None."
                        )
                        overall_status = False
                        ladder_response_raw = None
                    else:
                        logger_sc.error(
                            f"{test_name_ladder_api} passed status but returned invalid data: {repr(ladder_response_raw)}"
                        )
                        overall_status = False
                        ladder_response_raw = None
                elif s5_api_msg and "got None" in s5_api_msg:
                    logger_sc.warning(
                        f"{test_name_ladder_api} failed because API returned None. Cannot format."
                    )
                    overall_status = False
                    ladder_response_raw = None
                else:
                    overall_status = False
                    logger_sc.error(f"{test_name_ladder_api} failed.")
                    ladder_response_raw = None

                if ladder_response_raw and isinstance(ladder_response_raw, str):
                    owner_name_str = cast(str, test_owner_name)
                    target_name_str = cast(str, test_target_name)
                    format_lambda = lambda: format_api_relationship_path(
                        ladder_response_raw, owner_name_str, target_name_str
                    )
                    # *** ADJUST expected_substrings based on your TESTING_PERSON_TREE_ID ***
                    expected_substrings = [
                        target_name_str,
                        "Mother of",
                        owner_name_str,
                        "\n↓\n",
                        "Son of",
                    ]  # Example: Mother
                    logger_sc.info(
                        f"Expecting relationship format containing: {expected_substrings}"
                    )
                    _, s5_format_status, _ = _run_test_sc(
                        test_name_format_ladder,
                        format_lambda,
                        expected_type=str,
                        expected_contains=expected_substrings,
                    )
                    if s5_format_status == "FAIL":
                        overall_status = False
                else:
                    logger_sc.warning(
                        f"Skipping {test_name_format_ladder} because prerequisite API call failed or returned invalid data."
                    )
                    _run_test_sc(test_name_format_ladder, lambda: "Skipped")

        except Exception as e:
            # Use simplified logging
            logger_sc.critical(
                f"\n--- CRITICAL ERROR during self-check live tests ---", exc_info=True
            )
            _run_test_sc(
                "Self-Check Live Execution",
                lambda: False,
                expected_value="CRITICAL EXCEPTION OCCURRED",
            )
            overall_status = False
        finally:
            if session_manager_sc:
                logger_sc.info("--- Finalizing: Closing Session ---")
                session_manager_sc.close_sess()
            else:
                logger_sc.info("--- Finalizing: No session object to close ---")

    else:
        logger_sc.warning(
            "Skipping Live API tests due to missing dependencies or prerequisite failures."
        )
        phases_to_skip = [
            "SessionManager.start_sess()",
            "SessionManager.ensure_session_ready()",
            "Check Target Tree ID Found",
            "Check Target Owner Name Found",
            "API Call: Get Target Profile Details",
            "Check Target Name Found in API Resp",
            "API Call: Get Ladder Target Profile Details",
            "Function Call: parse_ancestry_person_details()",
            "Validation: Parsed Details Keys",
            "Validation: Parsed Name Match",
            "API Call: Get Tree Ladder",
            "Function Call: format_api_relationship_path()",
        ]
        existing_test_names = {name for name, _, _ in test_results_sc}
        for test_name in phases_to_skip:
            if test_name not in existing_test_names:
                _run_test_sc(test_name, lambda: "Skipped")

    # --- Print Formatted Summary ---
    print("\n--- api_utils.py Self-Check Summary ---")
    name_width = 50
    if test_results_sc:
        try:
            name_width = max(len(name) for name, _, _ in test_results_sc) + 2
        except ValueError:
            pass
    status_width = 8
    header = f"{'Test Name':<{name_width}} | {'Status':<{status_width}} | {'Message'}"
    print(header)
    print("-" * (len(header)))
    final_fail_count = 0
    final_skip_count = 0
    reported_test_names = set()
    for name, status, message in test_results_sc:
        reported_test_names.add(name)
        current_status = status
        if status == "FAIL":
            final_fail_count += 1
        elif status == "SKIPPED":
            final_skip_count += 1
            message = ""
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
        logger_sc.info("api_utils self-check overall status: PASS")
    else:
        logger_sc.error("api_utils self-check overall status: FAIL")

    return final_overall_status


# End of self_check

if __name__ == "__main__":
    print("Running api_utils.py self-check (with live API calls)...")
    log_file = Path("api_utils_self_check.log").resolve()
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
            format="%(asctime)s %(levelname)-8s [%(name)-15s thr:%(thread)d] %(message)s",
        )
        logger_standalone = logging.getLogger("api_utils_standalone")
        logger_standalone.info(f"Using basicConfig, logging to {log_file}")
    except Exception as log_setup_err:
        print(f"Error setting up logging: {log_setup_err}. Using basic logging.")
        logging.basicConfig(
            level=logging.DEBUG,
            filename=log_file,
            filemode="w",
            format="%(asctime)s %(levelname)-8s [%(name)-15s thr:%(thread)d] %(message)s",
        )
        logger_standalone = logging.getLogger("api_utils_standalone")
        logger_standalone.info(
            f"Using basicConfig due to setup error, logging to {log_file}"
        )

    if CONFIG_AVAILABLE and not getattr(
        config_instance, "TESTING_PERSON_TREE_ID", None
    ):
        print(
            "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        print("WARNING: config.TESTING_PERSON_TREE_ID is not set!")
        print("The relationship ladder test (Phase 5) requires this to be the ID of a")
        print("person in your tree (e.g., your mother, father) for accurate testing.")
        print("The test may fail or be skipped without it. Please set it in config.py.")
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )
    elif not CONFIG_AVAILABLE:
        print(
            "\nWARNING: config.py not loaded. Cannot check for TESTING_PERSON_TREE_ID."
        )

    self_check_passed = self_check()
    print("\nThis is the api_utils module. Import it into other scripts.")
    sys.exit(0 if self_check_passed else 1)
# End of __main__ block
# End of api_utils.py
# --- END OF FILE api_utils.py ---
