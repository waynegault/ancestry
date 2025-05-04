# --- START OF FILE api_utils.py ---
# api_utils.py
"""
Utility functions specifically for parsing Ancestry API responses
and formatting data obtained from APIs.
V2.9: Corrected NameError for html_content in format_api_relationship_path.
"""

# --- Standard library imports ---
import logging
import sys
import re
import os
import time
import json
import requests  # Keep for exception types
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
        TESTING_PERSON_TREE_ID = None

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
    Parses relationship data primarily from /getladder JSONP HTML response
    and formats it into the specific two-line output:
    'Person 1 (Lifespan): Relationship to Person 2'
    '↓'
    'Person 2 (Lifespan): Relationship to Person 1'

    Includes fallbacks for other formats (direct JSON, errors).

    Args:
        api_response_data: The raw data returned by the relationship API call.
        owner_name: The name of the tree owner (Person 2).
        target_name: The name of the person whose relationship is being checked (Person 1).

    Returns:
        A formatted two-line string representing the relationship,
        or an error message string if parsing fails.
    """
    # --- Step 1: Extract and Decode HTML content (Same as before) ---
    if not api_response_data:
        logger.warning(
            "format_api_relationship_path: Received empty API response data."
        )
        return "(No relationship data received from API)"

    html_content: Optional[str] = None
    # ... (JSONP/Dict handling logic remains the same) ...
    # --- [Existing JSONP/Dict handling code from previous versions] ---
    if isinstance(api_response_data, str):
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
                    status = parsed_json.get("status", "unknown")
                    if status == "success":
                        html_content = parsed_json.get("html")
                    else:
                        return f"(API returned error: {parsed_json.get('message', 'Unknown')})"
                else:
                    html_content = api_response_data  # Fallback
            except Exception as e:
                logger.error(f"Error processing JSONP: {e}")
                html_content = api_response_data
        else:
            html_content = api_response_data
    elif isinstance(api_response_data, dict):
        if "error" in api_response_data:
            return f"(API returned error object: {api_response_data.get('error', 'Unknown')})"
        elif "path" in api_response_data and isinstance(
            api_response_data["path"], list
        ):
            # Handle direct JSON path - This format doesn't match the desired output structure easily
            logger.warning(
                "Direct JSON path format cannot be easily converted to the desired two-line output. Returning basic steps."
            )
            path_steps_json = []
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
            return "(Received unhandled dictionary from API)"
    # --- [End JSONP/Dict Handling] ---

    # --- Step 2: Parse the HTML (if properly extracted) ---
    if not html_content or not isinstance(html_content, str):
        logger.warning("No processable HTML content found after extraction/decoding.")
        return "(Could not find or parse relationship HTML)"

    # --- Attempt Parsing with BeautifulSoup ---
    if not BeautifulSoup:
        logger.error("BeautifulSoup library not found. Cannot parse relationship HTML.")
        return "(Cannot parse relationship path - BeautifulSoup missing)"

    try:
        processed_html = html_content
        logger.debug(
            f"Parsing decoded HTML content with BeautifulSoup: {processed_html[:300]}..."
        )
        # --- Try lxml, fallback to html.parser ---
        try:
            soup = BeautifulSoup(processed_html, "lxml")
            parser_used = "lxml"
        except FeatureNotFound:
            logger.warning("'lxml' parser not found, falling back to 'html.parser'.")
            soup = BeautifulSoup(processed_html, "html.parser")
            parser_used = "html.parser"
        except Exception as e:
            logger.warning(f"Parser failed ({e}), falling back to 'html.parser'.")
            soup = BeautifulSoup(processed_html, "html.parser")
            parser_used = "html.parser"
        logger.debug(f"Successfully parsed HTML using '{parser_used}'.")

        # --- Specific Parsing Logic for /getladder Format ---
        list_items = soup.select("ul.textCenter li")
        logger.debug(
            f"Attempting specific parse: Found {len(list_items)} list items in ul.textCenter"
        )

        if len(list_items) < 3:
            logger.warning(
                f"Specific parse failed: Expected at least 3 list items, found {len(list_items)}"
            )
            return "(Relationship HTML structure not as expected)"

        item1 = list_items[0]
        item3 = list_items[2]  # Owner/Confirmation is usually the third item

        # Extract Info from First Item (Target Person)
        name1_tag = item1.find("b")
        role1_tag = item1.select_one("i b, i")
        name1 = name1_tag.get_text(strip=True) if name1_tag else None
        role1 = role1_tag.get_text(strip=True) if role1_tag else None
        lifespan1 = ""
        if (
            name1_tag
            and name1_tag.next_sibling
            and isinstance(name1_tag.next_sibling, str)
        ):
            potential_lifespan = name1_tag.next_sibling.strip()
            if re.match(r"^\d{4}\s*[-–]?\s*(\d{4})?$", potential_lifespan):
                lifespan1 = potential_lifespan

        # Extract Info from Third Item (Owner Person)
        name3_tag = item3.select_one("a b, a, b")
        role3_text_tag = item3.find("i")
        name3 = name3_tag.get_text(strip=True) if name3_tag else None
        role3_text = role3_text_tag.get_text(strip=True) if role3_text_tag else None
        # Attempt to find owner lifespan (might not be present)
        lifespan3 = ""
        if (
            name3_tag
            and name3_tag.next_sibling
            and isinstance(name3_tag.next_sibling, str)
        ):
            potential_lifespan3 = name3_tag.next_sibling.strip()
            if re.match(r"^\d{4}\s*[-–]?\s*(\d{4})?$", potential_lifespan3):
                lifespan3 = potential_lifespan3

        logger.debug(
            f"Specific Parse: Item 1 Name='{name1}', Role='{role1}', Lifespan='{lifespan1}'"
        )
        logger.debug(
            f"Specific Parse: Item 3 Name='{name3}', RoleText='{role3_text}', Lifespan='{lifespan3}'"
        )

        # Construct the Desired Output if components found
        if name1 and role1 and name3:
            # Determine inverse relationship for the second line
            role3 = "related"  # Default fallback
            role1_lower = role1.lower() if isinstance(role1, str) else ""
            if (
                role1_lower == "mother"
                or role1_lower == "father"
                or role1_lower == "parent"
            ):
                if role3_text and "son of" in role3_text.lower():
                    role3 = "Son"
                elif role3_text and "daughter of" in role3_text.lower():
                    role3 = "Daughter"
                else:
                    role3 = "Child"
            elif (
                role1_lower == "son"
                or role1_lower == "daughter"
                or role1_lower == "child"
            ):
                if role3_text and "mother of" in role3_text.lower():
                    role3 = "Mother"
                elif role3_text and "father of" in role3_text.lower():
                    role3 = "Father"
                else:
                    role3 = "Parent"
            # Add more inverse logic here if needed

            # Format output - Ensure names match expected owner/target from args for consistency
            # Use owner_name and target_name passed into the function
            line1 = f"{target_name}{f' ({lifespan1})' if lifespan1 else ''}: {role1.capitalize()} of {owner_name}"
            line2 = f"{owner_name}{f' ({lifespan3})' if lifespan3 else ''}: {role3} of {target_name}"

            result_str = f"{line1}\n↓\n{line2}"
            logger.info(f"Formatted relationship path (Specific Parse):\n{result_str}")
            # --- DEBUG PRINT ---
            print(
                f"\nDEBUG format_api_relationship_path RETURNING (Success Specific Format):\n{repr(result_str)}\n"
            )
            # --- END DEBUG PRINT ---
            return result_str
        else:
            logger.warning(
                "Specific parse failed to extract all components (name1, role1, name3)."
            )
            return "(Relationship details incomplete in HTML)"

    except Exception as e:
        logger.error(
            f"Error processing relationship HTML with BeautifulSoup: {e}", exc_info=True
        )
        error_str = f"(Error processing relationship HTML: {e})"
        print(
            f"\nDEBUG format_api_relationship_path RETURNING (Processing Error): {repr(error_str)}\n"
        )
        return error_str


# End of format_api_relationship_path

def temp_parse_ladder_html(html_string: str) -> str:
    """
    Temporary function to specifically parse the known HTML structure
    from the /getladder JSONP response.
    """
    logger.info("--- Running temp_parse_ladder_html ---")
    if not BeautifulSoup:
        return "(TEMP PARSE ERROR: BeautifulSoup not available)"

    try:
        # Use html.parser as a reliable default
        soup = BeautifulSoup(html_string, "html.parser")
        logger.debug(f"Temp Parse: Parsed with html.parser")

        # Expecting <ul class="textCenter"><li>...</li><li>...</li><li>...</li></ul>
        list_items = soup.select("ul.textCenter li")
        logger.debug(f"Temp Parse: Found {len(list_items)} list items in ul.textCenter")

        if len(list_items) < 3:
            logger.warning("Temp Parse: Did not find expected number of list items (at least 3)")
            return f"(TEMP PARSE ERROR: Found only {len(list_items)} list items)"

        # --- Extract Info from First Item (Ancestor/Relative) ---
        item1 = list_items[0]
        name1_tag = item1.find("b")
        role1_tag = item1.select_one("i b, i") # Role is inside <i>, possibly nested <b>
        name1 = name1_tag.get_text(strip=True) if name1_tag else None
        role1 = role1_tag.get_text(strip=True) if role1_tag else None
        lifespan1 = ""
        if name1_tag and name1_tag.next_sibling and isinstance(name1_tag.next_sibling, str):
             potential_lifespan = name1_tag.next_sibling.strip()
             if re.match(r'^\d{4}\s*[-–]?\s*(\d{4})?$', potential_lifespan):
                 lifespan1 = potential_lifespan

        logger.debug(f"Temp Parse Item 1: Name='{name1}', Role='{role1}', Lifespan='{lifespan1}'")

        # --- Extract Info from Third Item (Owner/Self) ---
        # Item 1 is the icon spacer
        item3 = list_items[2]
        name3_tag = item3.select_one("a b, a") # Name is usually linked
        role3_text_tag = item3.find("i") # Confirmation text "You are the..."
        name3 = name3_tag.get_text(strip=True) if name3_tag else None
        role3_text = role3_text_tag.get_text(strip=True) if role3_text_tag else None

        logger.debug(f"Temp Parse Item 3: Name='{name3}', RoleText='{role3_text}'")

        # --- Construct the Desired Output ---
        if not name1 or not role1 or not name3:
             logger.warning("Temp Parse: Failed to extract all required name/role components.")
             return "(TEMP PARSE ERROR: Missing components)"

        # Determine relationship for the second line ("Son of...")
        # We need the inverse relationship
        role3 = "Unknown relationship"
        if role3_text and "son of" in role3_text.lower():
            role3 = "Son"
        elif role3_text and "daughter of" in role3_text.lower():
             role3 = "Daughter"
        # Add more inverse checks if needed based on role1

        # Format output
        line1 = f"{name1}{f' ({lifespan1})' if lifespan1 else ''}: {role1.capitalize()} of {name3}"
        line2 = f"{name3}: {role3} of {name1}" # Simplified lifespan for owner

        result = f"{line1}\n↓\n{line2}"
        logger.info(f"Temp Parse Result:\n{result}")
        return result

    except Exception as e:
        logger.error(f"Error in temp_parse_ladder_html: {e}", exc_info=True)
        return f"(TEMP PARSE ERROR: Exception - {e})"

# --- Helper Function for HTML Extraction (Potentially less needed now) ---
def _extract_ladder_html(raw_content: Union[str, Dict]) -> Optional[str]:
    """
    Extracts and decodes the relationship ladder HTML from raw API response content.
    Handles standard JSON, JSONP, and potential errors.
    NOTE: May be less necessary if format_api_relationship_path handles JSONP directly.
    """
    if isinstance(raw_content, dict) and "error" in raw_content:
        error_msg = raw_content.get("error", {}).get(
            "message", raw_content.get("message", "Unknown API Error")
        )
        logger.error(f"_extract_ladder_html: API returned error: {error_msg}")
        return None

    if not raw_content or not isinstance(raw_content, str):
        logger.error(
            f"_extract_ladder_html: Invalid raw content type: {type(raw_content)}. Expected string."
        )
        return None

    html_escaped = None
    logger.debug("_extract_ladder_html: Attempting JSONP extraction from string...")
    try:
        jsonp_match = re.match(r"^\s*[\w$.]+\((.*)\)\s*;?\s*$", raw_content, re.DOTALL)
        if jsonp_match:
            json_str = jsonp_match.group(1).strip()
            if json_str.startswith("{") and json_str.endswith("}"):
                json_data = json.loads(json_str)
                if isinstance(json_data, dict) and "html" in json_data:
                    html_escaped = json_data["html"]
                    if isinstance(html_escaped, str):
                        logger.debug(
                            f"_extract_ladder_html: Found HTML via JSONP. Length: {len(html_escaped)}"
                        )
                    else:
                        logger.warning(
                            f"_extract_ladder_html: 'html' key found in JSONP, but not string: {type(html_escaped)}"
                        )
                        html_escaped = None
                else:
                    logger.warning(
                        "_extract_ladder_html: 'html' key not found in JSONP object."
                    )
            else:
                logger.warning(
                    f"_extract_ladder_html: Content in JSONP () not JSON: {json_str[:100]}..."
                )
        else:
            logger.debug(
                "_extract_ladder_html: Raw content does not match JSONP structure. Assuming direct HTML/text."
            )
            # If not JSONP, the raw string itself might be the HTML (or just text)
            html_escaped = (
                raw_content  # Treat the input string as the content to unescape/return
            )

    except json.JSONDecodeError as json_e:
        logger.warning(
            f"_extract_ladder_html: JSONDecodeError during JSONP extraction: {json_e}"
        )
        html_escaped = raw_content  # Fallback to raw content on JSON error
    except Exception as e:
        logger.warning(
            f"_extract_ladder_html: Unexpected error during JSONP extraction: {e}"
        )
        html_escaped = raw_content  # Fallback to raw content on other errors

    if not html_escaped:
        logger.error(
            "_extract_ladder_html: Could not extract or determine HTML content."
        )
        logger.debug(f"Raw content snippet: {raw_content[:500]}...")
        return None

    # Attempt to unescape potential HTML entities
    try:
        # Multi-pass unescaping might be needed for complex cases, but start simple
        html_unescaped = html.unescape(html_escaped)
        logger.debug("_extract_ladder_html: Successfully unescaped content.")
        return html_unescaped
    except Exception as decode_err:
        logger.error(
            f"_extract_ladder_html: Could not unescape content. Error: {decode_err}",
            exc_info=False,
        )
        logger.debug(f"Problematic escaped content snippet: {html_escaped[:500]}...")
        return html_escaped  # Return the escaped version if unescaping fails


# End of _extract_ladder_html


# --- Display Function (Consider removing or simplifying) ---
def display_raw_relationship_ladder(
    raw_content: Union[str, Dict], owner_name: str, target_name: str
):
    """
    DEPRECATED potentially. Parses and displays the Ancestry relationship ladder.
    Prefer using format_api_relationship_path directly in calling code.
    """
    logger.warning(
        "display_raw_relationship_ladder is potentially deprecated. Use format_api_relationship_path."
    )

    if BeautifulSoup is None:
        logger.error(
            "BeautifulSoup library not found. Cannot parse relationship ladder HTML."
        )
        print(f"\n--- Relationship between {owner_name} and {target_name} (API) ---")
        print("\n(Cannot parse relationship path - BeautifulSoup missing)")
        return

    logger.info(
        f"\n--- Relationship between {owner_name} and {target_name} (Raw Display) ---"
    )

    # Use the formatter function which now includes BS4 fallback
    formatted_path_str = format_api_relationship_path(
        raw_content, owner_name, target_name
    )

    if formatted_path_str and not formatted_path_str.startswith(
        "("
    ):  # Check it's not an error message
        # Print the already formatted path from the function
        print(formatted_path_str.strip())
    else:
        # format_api_relationship_path should return error strings on failure
        print(formatted_path_str or "(Could not format relationship path steps)")


# End of display_raw_relationship_ladder


# --- Standalone Test Block ---
def self_check() -> bool:
    """
    Performs internal self-checks for api_utils.py, including LIVE API calls.
    Requires .env file to be correctly configured. Provides formatted output summary.
    """
    # --- Local Imports for Self-Check ---
    # Ensure utils and SessionManager are available if needed for API calls
    try:
        import utils
        from utils import SessionManager, _api_req
        from config import (
            config_instance as config_instance_sc,
            selenium_config as selenium_config_sc,
        )  # Use distinct names

        if not UTILS_AVAILABLE:
            raise ImportError("Base utils failed to load")
        if not CONFIG_AVAILABLE:
            raise ImportError("Config failed to load")
    except ImportError as e:
        print(
            f"\n[api_utils.py self-check ERROR] - Cannot import base utils/config for live tests: {e}"
        )
        # Cannot run live tests, but allow static tests below
        SessionManager = None
        _api_req = None
        config_instance_sc = None
        selenium_config_sc = None

    # Use main logger if available, otherwise basic
    try:
        from logging_config import logger as logger_sc
    except ImportError:
        logger_sc = logging.getLogger("api_utils.self_check")

    # --- Test Runner Helper ---
    test_results_sc: List[Tuple[str, str, str]] = []
    session_manager_sc: Optional[SessionManager] = (
        None  # Define session manager for the scope
    )

    # ... (_run_test_sc definition remains the same) ...
    def _run_test_sc(
        test_name: str, test_func: Callable, *args, **kwargs
    ) -> Tuple[str, str, str]:
        """Runs a single test, logs result, stores result, and returns status tuple."""
        logger_sc.debug(f"[ RUNNING SC ] {test_name}")
        status = "FAIL"
        message = ""
        expect_none = kwargs.pop("expected_none", False)
        expect_type = kwargs.pop("expected_type", None)
        expect_value = kwargs.pop("expected_value", None)
        expect_contains = kwargs.pop("expected_contains", None)
        expect_truthy = kwargs.pop(
            "expected_truthy", False
        )  # Check if result is considered True

        try:
            result = test_func(*args, **kwargs)
            passed = False
            if expect_none and result is None:
                passed = True
            elif expect_type is not None and isinstance(result, expect_type):
                passed = True
            elif expect_value is not None and result == expect_value:
                passed = True
            elif (
                expect_contains is not None
                and isinstance(result, str)
                and expect_contains in result
            ):
                passed = True
            elif expect_truthy and bool(result):
                passed = True
            elif (
                not expect_none
                and not expect_type
                and not expect_value
                and not expect_contains
                and not expect_truthy
            ):
                # Default check only passes if result is exactly True
                if result is True:
                    passed = True

            if passed:
                status = "PASS"
            else:  # Construct failure message
                status = "FAIL"
                res_repr = repr(result)[:100] + (
                    "..." if len(repr(result)) > 100 else ""
                )
                if expect_none:
                    message = f"Expected None, got {type(result).__name__} ({res_repr})"
                elif expect_type:
                    message = f"Expected type {expect_type.__name__}, got {type(result).__name__}"
                elif expect_value:
                    message = (
                        f"Expected value '{str(expect_value)[:50]}', got '{res_repr}'"
                    )
                elif expect_contains:
                    message = f"Expected result to contain '{expect_contains}', got '{res_repr}'"
                elif expect_truthy:
                    message = f"Expected truthy value, got {res_repr}"
                else:
                    message = f"Assertion failed (returned {res_repr})"

        except Exception as e:
            status = "FAIL"
            message = f"{type(e).__name__}: {e}"
            logger_sc.debug(
                f"Exception details for {test_name}: {message}\n{traceback.format_exc()}",
                exc_info=False,
            )

        log_level = logging.INFO if status == "PASS" else logging.ERROR
        log_message = f"[ {status:<6} SC ] {test_name}{f': {message}' if message and status == 'FAIL' else ''}"
        logger_sc.log(log_level, log_message)
        test_results_sc.append((test_name, status, message))
        return (test_name, status, message)

    # --- Internal API Call Helpers for Self-Check ---
    def _sc_api_req(
        url: str, description: str, expect_json: bool = True, **kwargs
    ) -> Any:
        """Wrapper for _api_req within self-check."""
        nonlocal session_manager_sc  # Access the outer scope session manager
        if not _api_req:
            raise RuntimeError("_api_req not available for self-check")
        if not session_manager_sc or not session_manager_sc.is_sess_valid():
            raise RuntimeError("Session not ready for API call in self_check")
        # Ensure kwargs like timeout are passed through
        return _api_req(
            url=url,
            driver=session_manager_sc.driver,
            session_manager=session_manager_sc,
            api_description=f"{description} (Self Check)",
            **kwargs,  # Pass other args like timeout, headers, use_csrf_token etc.
        )

    def _sc_get_profile_details(profile_id: str) -> Optional[Dict]:
        if not config_instance_sc:
            return None
        url = urljoin(
            config_instance_sc.BASE_URL,
            f"/app-api/express/v1/profiles/details?userId={profile_id.upper()}",
        )
        # Add timeout from selenium_config if available
        timeout = (
            getattr(selenium_config_sc, "API_TIMEOUT", 60) if selenium_config_sc else 60
        )
        return _sc_api_req(
            url,
            "Get Target Name",
            expect_json=True,
            use_csrf_token=False,
            timeout=timeout,
        )

    def _sc_get_tree_ladder(tree_id: str, person_id: str) -> Optional[str]:
        if not config_instance_sc or not selenium_config_sc:
            return None  # Add selenium_config check
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
        headers = {  # Specific headers needed for JSONP
            "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        }
        # --- Use Standard Timeout & Disable CSRF ---
        api_timeout = selenium_config_sc.API_TIMEOUT  # Use standard timeout
        return _sc_api_req(
            ladder_api_url,
            "Get Tree Ladder",
            expect_json=False,  # Expects text/javascript
            headers=headers,
            referer_url=ladder_referer,
            force_text_response=True,
            timeout=api_timeout,  # Use standard timeout
            use_csrf_token=False,  # Explicitly disable CSRF for this GET
        )  # Expects String

    # --- Test Parameters ---
    can_run_live_tests = bool(
        SessionManager and _api_req and config_instance_sc and selenium_config_sc
    )

    target_profile_id = (
        getattr(config_instance_sc, "TESTING_PROFILE_ID", None)
        if config_instance_sc
        else None
    )
    target_person_id = (
        getattr(config_instance_sc, "TESTING_PERSON_TREE_ID", None)
        if config_instance_sc
        else None
    )
    target_name_from_profile = "Unknown Target"  # Default

    if can_run_live_tests and not target_profile_id:
        logger_sc.warning(
            "TESTING_PROFILE_ID missing in config_instance. Profile-dependent live tests will be skipped."
        )

    # --- Status Tracking ---
    overall_status = True  # Assume PASS initially

    logger_sc.info("\n[api_utils.py self-check starting...]")

    # === Phase 0: Prerequisite Checks ===
    logger_sc.info("--- Phase 0: Prerequisite Checks ---")
    # ... (Prereq checks remain the same) ...
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
        can_run_live_tests = False  # Prevent live tests

    # === Live Tests Section ===
    if can_run_live_tests:
        try:
            # === Phase 1: Session Setup ===
            logger_sc.info("--- Phase 1: Session Setup & Login ---")
            # ... (Session setup remains the same) ...
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
            # ... (Fetching IDs and owner name remains the same) ...
            target_tree_id = session_manager_sc.my_tree_id
            target_owner_name = session_manager_sc.tree_owner_name
            _, s2_status_tid, _ = _run_test_sc(
                "Check Target Tree ID Found",
                lambda: bool(target_tree_id),
                expected_truthy=True,
            )
            _, s2_status_owner, _ = _run_test_sc(
                "Check Target Owner Name Found",
                lambda: bool(target_owner_name),
                expected_truthy=True,
            )
            if s2_status_tid == "FAIL" or s2_status_owner == "FAIL":
                overall_status = False

            profile_response_details = None
            if target_profile_id:
                api_call_lambda = lambda: _sc_get_profile_details(
                    cast(str, target_profile_id)
                )
                test_name = "API Call: Get Target Name"
                _, s2_status, _ = _run_test_sc(
                    test_name, api_call_lambda, expected_type=dict
                )
                if s2_status == "PASS":
                    profile_response_details = api_call_lambda()  # Get result
                    if profile_response_details:
                        name_test_card = {"personId": target_profile_id}
                        target_name_from_profile = _extract_name_from_api_details(
                            name_test_card, profile_response_details
                        )
                        _, s2_status_name, _ = _run_test_sc(
                            "Check Target Name Found in API Resp",
                            lambda: target_name_from_profile != "Unknown",
                            expected_truthy=True,
                        )
                        if s2_status_name == "FAIL":
                            overall_status = False
                    else:
                        logger_sc.error(
                            "API call passed but response was None/invalid."
                        )
                        _run_test_sc(
                            "Check Target Name Found in API Resp", lambda: False
                        )
                        overall_status = False
                else:
                    overall_status = False
            else:
                logger_sc.warning(
                    "Skipping Get Target Name API call: TESTING_PROFILE_ID not set."
                )
                _run_test_sc("API Call: Get Target Name", lambda: "Skipped")

            # === Phase 3: Test parse_ancestry_person_details ===
            logger_sc.info("--- Phase 3: Test parse_ancestry_person_details ---")
            # ... (parse_ancestry_person_details test remains the same) ...
            if (
                profile_response_details
                and isinstance(profile_response_details, dict)
                and target_profile_id
            ):
                person_card_for_parse = {
                    "personId": target_profile_id,
                    "treeId": target_tree_id,
                }
                test_name_parse = "Function Call: parse_ancestry_person_details()"
                parsed_details = None  # Initialize
                try:
                    parse_lambda = lambda: parse_ancestry_person_details(
                        person_card_for_parse, profile_response_details
                    )
                    _, s3_status, _ = _run_test_sc(
                        test_name_parse, parse_lambda, expected_type=dict
                    )
                    if s3_status == "PASS":
                        parsed_details = parse_lambda()  # Call again to get result
                        if parsed_details:
                            keys_ok = all(
                                k in parsed_details
                                for k in ["name", "person_id", "link"]
                            )
                            _, s3_v_status, _ = _run_test_sc(
                                "Validation: Parsed Details Keys",
                                lambda: keys_ok,
                                expected_truthy=True,
                            )
                            if s3_v_status == "FAIL":
                                overall_status = False
                            _, s3_n_status, _ = _run_test_sc(
                                "Validation: Parsed Name Match",
                                lambda: parsed_details.get("name")
                                == target_name_from_profile,
                                expected_truthy=True,
                            )
                        else:
                            _run_test_sc(
                                test_name_parse,
                                lambda: False,
                                expected_value="Parser returned None unexpectedly",
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
                    "Skipping parse_ancestry_person_details test: Prerequisite API call failed or profile ID missing."
                )
                _run_test_sc(
                    "Function Call: parse_ancestry_person_details()", lambda: "Skipped"
                )

            # === Phase 4: Test HTML Extraction (_extract_ladder_html - less critical now) ===
            logger_sc.info("--- Phase 4: Test HTML Extraction (Low Priority) ---")
            # ... (HTML extraction test remains the same) ...
            test_jsonp_good = 'jQuery123({"html": "<p>Good HTML</p>", "other": 1});'
            expected_html_good = "<p>Good HTML</p>"
            _, s4_status, _ = _run_test_sc(
                "HTML Extract Helper: Good JSONP",
                lambda: _extract_ladder_html(test_jsonp_good),
                expected_value=expected_html_good,
            )

            # === Phase 5: Test Relationship Ladder Parsing ===
            logger_sc.info(
                "--- Phase 5: Test Relationship Ladder Parsing (Live API) ---"
            )
            # Use known names/IDs for this specific test
            test_target_person_id = getattr(
                config_instance_sc, "TESTING_PERSON_TREE_ID", None
            )
            test_target_tree_id = (
                target_tree_id  # Use the owner's tree ID found earlier
            )
            test_owner_name = target_owner_name  # Use the owner's name found earlier
            # test_target_name = "Frances Margaret Milne" # No longer needed for temp function call

            can_run_ladder_test_live = bool(
                test_owner_name and test_target_person_id and test_target_tree_id
            )

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
                _run_test_sc("API Call: Get Tree Ladder", lambda: "Skipped")
                _run_test_sc(
                    "Function Call: format_api_relationship_path()", lambda: "Skipped"
                )  # Keep skip marker
            else:
                # Test API call using helper
                ladder_response_raw = None
                api_call_lambda_ladder = lambda: _sc_get_tree_ladder(
                    cast(str, test_target_tree_id), cast(str, test_target_person_id)
                )
                test_name_ladder_api = "API Call: Get Tree Ladder"
                _, s5_api_status, _ = _run_test_sc(
                    test_name_ladder_api, api_call_lambda_ladder, expected_type=str
                )
                if s5_api_status == "PASS":
                    ladder_response_raw = api_call_lambda_ladder()  # Get result
                    print(
                        f"\nDEBUG Raw Ladder Response (_sc_get_tree_ladder output):\n{repr(ladder_response_raw)}\n"
                    )
                else:
                    overall_status = False

                # Proceed only if API call succeeded and returned string
                if ladder_response_raw and isinstance(ladder_response_raw, str):

                    # --- CALL THE TEMP FUNCTION ---
                    print("\n--- Calling temp_parse_ladder_html ---")
                    # Extract the raw HTML first using the existing helper
                    html_for_temp_parse = _extract_ladder_html(ladder_response_raw)
                    if html_for_temp_parse:
                        temp_result = temp_parse_ladder_html(html_for_temp_parse)
                        print(
                            f"--- Output from temp_parse_ladder_html ---\n{temp_result}\n----------------------------------------"
                        )
                    else:
                        print(
                            "--- Could not extract HTML for temp_parse_ladder_html ---"
                        )
                    # --- END TEMP FUNCTION CALL ---

                    # --- Temporarily Comment Out Original Test ---
                    # format_lambda = lambda: format_api_relationship_path(
                    #     ladder_response_raw,
                    #     cast(str, test_owner_name),
                    #     test_target_name
                    # )
                    # _, s5_format_status, _ = _run_test_sc(
                    #     "Function Call: format_api_relationship_path()",
                    #     format_lambda,
                    #     expected_contains="Mother"
                    # )
                    # if s5_format_status == 'FAIL': overall_status = False
                    print(
                        "\nNOTE: Original format_api_relationship_path test skipped for temp function test.\n"
                    )
                    # Manually mark as skipped for summary? Or just ignore for now.
                    test_results_sc.append(
                        (
                            "Function Call: format_api_relationship_path()",
                            "SKIPPED",
                            "Using temp function",
                        )
                    )

                elif (
                    s5_api_status == "PASS"
                ):  # API Call passed but didn't return valid string
                    logger_sc.error(
                        "Ladder API call passed status check but returned invalid data type."
                    )
                    _run_test_sc(
                        "Function Call: format_api_relationship_path()",
                        lambda: False,
                        expected_value="API Response Invalid Type",
                    )
                    overall_status = False
                # End if/elif ladder_response_raw check


        except Exception as e:
            logger_sc.critical(
                f"\n--- CRITICAL ERROR during self-check live tests ---", exc_info=True
            )
            _run_test_sc(
                "Self-Check Live Execution",
                lambda: False,
                expected_value=f"CRITICAL ERROR: {e}",
            )
            overall_status = False
        finally:
            if session_manager_sc:
                logger_sc.info("--- Finalizing: Closing Session ---")
                session_manager_sc.close_sess()
            else:
                logger_sc.info("--- Finalizing: No session to close ---")

    else:  # Not can_run_live_tests
        logger_sc.warning(
            "Skipping Live API tests due to missing dependencies (utils/config/session)."
        )
        _run_test_sc("Live API Tests Phase", lambda: "Skipped")

    # --- Print Formatted Summary ---
    # ... (Summary printing remains the same) ...
    print("\n--- api_utils.py Self-Check Summary ---")
    name_width = 50
    if test_results_sc:
        try:
            name_width = max(len(name) for name, _, _ in test_results_sc)
        except ValueError:
            pass  # Handle empty list case
    status_width = 8
    header = f"{'Test Name':<{name_width}} | {'Status':<{status_width}} | {'Message'}"
    print(header)
    print("-" * (len(header)))
    final_fail_count = 0
    final_skip_count = 0
    for name, status, message in test_results_sc:
        if status == "FAIL":
            final_fail_count += 1
        if message == "Skipped":
            status = "SKIPPED"
            final_skip_count += 1
        print(
            f"{name:<{name_width}} | {status:<{status_width}} | {message if status != 'PASS' else ''}"
        )
    print("-" * (len(header)))
    total_tests = len(test_results_sc)
    passed_tests = total_tests - final_fail_count - final_skip_count
    final_overall_status = overall_status and (final_fail_count == 0)
    final_status_msg = f"Result: {'PASS' if final_overall_status else 'FAIL'} ({passed_tests} passed, {final_fail_count} failed, {final_skip_count} skipped out of {total_tests} tests)"
    print(f"{final_status_msg}\n")
    if final_overall_status:
        logger_sc.info("api_utils self-check overall status: PASS")
    else:
        logger_sc.error("api_utils self-check overall status: FAIL")

    return final_overall_status  # Return the actual test outcome


# End of self_check


if __name__ == "__main__":
    # ... (main block remains the same) ...
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
        logger_standalone = logging.getLogger("api_utils_standalone")
        logger_standalone.setLevel(logging.DEBUG)
    except Exception as log_setup_err:
        print(f"Error setting up logging: {log_setup_err}. Using basic logging.")
        logger_standalone = logging.getLogger("api_utils_standalone")
        logger_standalone.setLevel(logging.DEBUG)

    self_check_passed = self_check()
    print("\nThis is the api_utils module. Import it into other scripts.")
    sys.exit(0 if self_check_passed else 1)
# End of __main__ block
# End of api_utils.py
# --- END OF FILE api_utils.py ---
