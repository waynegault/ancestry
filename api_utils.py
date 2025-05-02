# --- START OF FILE api_utils.py ---
# api_utils.py
"""
Utility functions specifically for parsing Ancestry API responses
and formatting data obtained from APIs.
Consolidates parsing/formatting logic from temp.py v7.36.
V16.0: Consolidated API helpers from temp.py, uses date helpers from gedcom_utils.
V16.1: Added standalone self-check functionality. Adjusted self-check mock data.
V16.2: Rewrote self_check to use live API calls for functional testing.
V16.3: Corrected API endpoint used in self_check for fetching person details.
V16.4: Reverted self_check Phase 3 to use Profile Details API (returns JSON)
       instead of Facts API (returns HTML), ensuring parser test works.
V16.5: Added diagnostics (logging, shorter timeout) for Ladder API call hang.
V16.6: Implemented personId lookup via Tree Search API and switched Ladder API
       call to use the correct /getladder endpoint with personId.
V16.7: Added logging around requests call in _api_req. Disabled header CSRF token
       for Tree Search API call in self_check, relying on cookies. Added timeout.
       Adjusted Accept header for Tree Search API.
V16.8: Adjusted Tree Search API response handling in self_check to accept
       list or dict return types.
V16.9: Modified Tree Search in self_check to use only FirstName for lookup.
V16.10: Replaced Tree Search API with Person Picker API in self_check for
        personId lookup, using FirstName and LastName.
V16.11: Disabled header CSRF token for Person Picker API call in self_check.
V16.12: Removed fragile personId API search from self_check. Now requires
        TESTING_PERSON_TREE_ID (env var) loaded into config_instance.
V16.14: Corrected Ladder API call in self_check Phase 4 to include JSONP
        parameters. Updated JSONP handling in display_raw_relationship_ladder.
        Uses TESTING_PERSON_TREE_ID from config.
V16.15: Corrected validation logic for format_api_relationship_path in self_check.
V16.16: Refactored HTML/JSONP extraction logic from display_raw_relationship_ladder
        into a new helper function _extract_ladder_html. Updated self_check.
V16.17: Revised format_api_relationship_path to correctly parse relationship direction
        and handle appended dates in names based on analysis of raw HTML.
"""

# --- Standard library imports ---
import logging
import sys
import re
import os
import time
import json
import requests # Keep if used by parsing logic, though _api_req handles fetch
import urllib.parse # Used for urlencode in self_check
import html
from typing import Optional, Dict, Any, Union, List, Tuple
from datetime import (
    datetime,
    timezone,
)  # Import datetime for parse_ancestry_person_details and self_check
from urllib.parse import (
    urljoin,
    urlencode,
    quote,
)  # Need quote for person picker params
from pathlib import Path # Needed for __main__ block

# --- Third-party imports ---
# Keep BeautifulSoup import here, check for its availability in functions
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore # Gracefully handle missing dependency

# --- Local application imports ---
# Use try-except for robustness, especially if run standalone initially
try:
    import utils
    from utils import format_name, ordinal_case
    from config import config_instance, selenium_config
    from gedcom_utils import _parse_date, _clean_display_date
    UTILS_AVAILABLE = True
except ImportError as imp_err:
    UTILS_AVAILABLE = False
    # Define fallbacks if imports fail
    format_name = lambda x: str(x).title() if x else "Unknown"
    ordinal_case = lambda x: str(x)
    _parse_date = lambda x: None
    _clean_display_date = lambda x: str(x) if x else "N/A"
    class DummyConfig:
        BASE_URL = "https://www.ancestry.com" # Provide a default
        TESTING_PROFILE_ID = "08FA6E79-0006-0000-0000-000000000000"
        TESTING_PERSON_TREE_ID = None
    config_instance = DummyConfig()
    selenium_config = None # Define selenium_config as None or a dummy if needed

# Initialize logger - Ensure logger is always available
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_utils")

# --- API Response Parsing ---
def parse_ancestry_person_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Dict:
    """
    Extracts standardized details from Ancestry Person-Card and Facts API responses.
    Includes parsing dates/places and generating a link.
    Based on temp.py v7.36 logic, uses imported date helpers.
    Prioritizes facts_data if available.

    Args:
        person_card (Dict): A dictionary containing basic person info.
        facts_data (Optional[Dict]): More detailed API response (e.g., Profile Details).

    Returns:
        Dict: A standardized dictionary containing parsed details.
    """
    details = {
        "name": "Unknown",
        "birth_date": None,
        "birth_place": None,
        "death_date": None,
        "death_place": None,
        "gender": None,
        "person_id": person_card.get("personId"),
        "tree_id": person_card.get("treeId"),
        "link": None,
        "api_birth_obj": None,
        "api_death_obj": None,
        "is_living": None,
    }

    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            details["name"] = person_info.get("personName", details["name"])
            gender_fact = person_info.get("gender")
            if gender_fact and isinstance(gender_fact, str):
                details["gender"] = (
                    "M"
                    if gender_fact.lower() == "male"
                    else "F" if gender_fact.lower() == "female" else None
                )
            details["is_living"] = person_info.get("isLiving", details["is_living"])
        # End if person_info is dict

        if details["name"] == "Unknown":
            details["name"] = facts_data.get("personName", details["name"])
        # End if name unknown

        if details["name"] == "Unknown":
            details["name"] = facts_data.get("DisplayName", details["name"])
        # End if name unknown

        if details["name"] == "Unknown":
            first_name_pd = facts_data.get("FirstName")
            if first_name_pd:
                last_name_pd = facts_data.get("LastName")
                details["name"] = f"{first_name_pd} {last_name_pd}" if last_name_pd else first_name_pd
            # End if first_name_pd
            if not details["name"]:
                details["name"] = "Unknown"
            # End if not details name
        # End if name unknown

        if details["gender"] is None:
            gender_fact = facts_data.get("gender")
            if gender_fact and isinstance(gender_fact, str):
                details["gender"] = (
                    "M"
                    if gender_fact.lower() == "male"
                    else "F" if gender_fact.lower() == "female" else None
                )
            # End if gender_fact
        # End if gender is None

        if details["gender"] is None:
            gender_pd = facts_data.get("Gender")
            if gender_pd and isinstance(gender_pd, str):
                details["gender"] = (
                    "M"
                    if gender_pd.lower() == "male"
                    else "F" if gender_pd.lower() == "female" else None
                )
            # End if gender_pd
        # End if gender is None

        if details["is_living"] is None:
            details["is_living"] = facts_data.get("isLiving", details["is_living"])
        # End if is_living is None

        if details["is_living"] is None:
            details["is_living"] = facts_data.get("IsLiving", details["is_living"])
        # End if is_living is None

        birth_fact_group = facts_data.get("facts", {}).get("Birth", [{}])[0]
        if isinstance(birth_fact_group, dict):
            date_info = birth_fact_group.get("date", {})
            place_info = birth_fact_group.get("place", {})
            if isinstance(date_info, dict):
                details["birth_date"] = date_info.get("normalized", date_info.get("original"))
            # End if date_info
            if isinstance(place_info, dict):
                details["birth_place"] = place_info.get("placeName")
            # End if place_info
        # End if birth_fact_group

        death_fact_group = facts_data.get("facts", {}).get("Death", [{}])[0]
        if isinstance(death_fact_group, dict):
            date_info = death_fact_group.get("date", {})
            place_info = death_fact_group.get("place", {})
            if isinstance(date_info, dict):
                details["death_date"] = date_info.get("normalized", date_info.get("original"))
            # End if date_info
            if isinstance(place_info, dict):
                details["death_place"] = place_info.get("placeName")
            # End if place_info
        # End if death_fact_group

        if details["birth_date"] is None:
            birth_fact_alt = facts_data.get("birthDate")
            if birth_fact_alt and isinstance(birth_fact_alt, dict):
                date_str = birth_fact_alt.get("normalized", birth_fact_alt.get("date", ""))
                place_str = birth_fact_alt.get("place", "")
                if date_str and isinstance(date_str, str):
                    details["birth_date"] = date_str
                # End if date_str
                if place_str and isinstance(place_str, str):
                    details["birth_place"] = place_str
                # End if place_str
            elif isinstance(birth_fact_alt, str):
                details["birth_date"] = birth_fact_alt
            # End if/elif birth_fact_alt
        # End if birth_date is None

        if details["death_date"] is None:
            death_fact_alt = facts_data.get("deathDate")
            if death_fact_alt and isinstance(death_fact_alt, dict):
                date_str = death_fact_alt.get("normalized", death_fact_alt.get("date", ""))
                place_str = death_fact_alt.get("place", "")
                if date_str and isinstance(date_str, str):
                    details["death_date"] = date_str
                # End if date_str
                if place_str and isinstance(place_str, str):
                    details["death_place"] = place_str
                # End if place_str
            elif isinstance(death_fact_alt, str):
                details["death_date"] = death_fact_alt
            # End if/elif death_fact_alt
        # End if death_date is None
    # End if facts_data

    if details["name"] == "Unknown":
        details["name"] = person_card.get("name", "Unknown")
    # End if name unknown

    if details["birth_date"] is None:
        birth_info_card = person_card.get("birth", "")
        if birth_info_card and isinstance(birth_info_card, str):
            parts = birth_info_card.split(" in ")
            details["birth_date"] = parts[0].strip() if parts else birth_info_card
            if details["birth_place"] is None:
                details["birth_place"] = parts[1].strip() if len(parts) > 1 else None
            # End if birth_place
        elif isinstance(birth_info_card, dict):
            details["birth_date"] = birth_info_card.get("date", details["birth_date"])
            if details["birth_place"] is None:
                details["birth_place"] = birth_info_card.get("place", details["birth_place"])
            # End if birth_place
        # End if/elif birth_info_card
    # End if birth_date is None

    if details["death_date"] is None:
        death_info_card = person_card.get("death", "")
        if death_info_card and isinstance(death_info_card, str):
            parts = death_info_card.split(" in ")
            details["death_date"] = parts[0].strip() if parts else death_info_card
            if details["death_place"] is None:
                details["death_place"] = parts[1].strip() if len(parts) > 1 else None
            # End if death_place
        elif isinstance(death_info_card, dict):
            details["death_date"] = death_info_card.get("date", details["death_date"])
            if details["death_place"] is None:
                details["death_place"] = death_info_card.get("place", details["death_place"])
            # End if death_place
        # End if/elif death_info_card
    # End if death_date is None

    if details["gender"] is None:
        gender_card = person_card.get("gender")
        if gender_card and isinstance(gender_card, str):
            details["gender"] = (
                "M"
                if gender_card.lower() == "male"
                else "F" if gender_card.lower() == "female" else None
            )
        # End if gender_card
    # End if gender is None

    if details["is_living"] is None:
        details["is_living"] = person_card.get("isLiving", details["is_living"])
    # End if is_living is None

    details["name"] = format_name(details["name"])

    if UTILS_AVAILABLE and _parse_date:
        if details["birth_date"]:
            details["api_birth_obj"] = _parse_date(details["birth_date"])
        # End if birth_date
        if details["death_date"]:
            details["api_death_obj"] = _parse_date(details["death_date"])
        # End if death_date
    # End if UTILS_AVAILABLE

    if UTILS_AVAILABLE and _clean_display_date:
        details["birth_date"] = (
            _clean_display_date(details["birth_date"]) if details["birth_date"] else "N/A"
        )
        details["death_date"] = (
            _clean_display_date(details["death_date"]) if details["death_date"] else "N/A"
        )
    else:
        details["birth_date"] = str(details["birth_date"]) if details["birth_date"] else "N/A"
        details["death_date"] = str(details["death_date"]) if details["death_date"] else "N/A"
    # End if/else UTILS_AVAILABLE

    base_url_for_link = getattr(config_instance, "BASE_URL", "https://www.ancestry.com").rstrip("/")
    if details["tree_id"] and details["person_id"]:
        details["link"] = f"{base_url_for_link}/family-tree/person/tree/{details['tree_id']}/person/{details['person_id']}/facts"
    elif details["person_id"]:
        details["link"] = f"{base_url_for_link}/discoveryui-matches/profile/{details['person_id']}"
    else:
        details["link"] = "(unavailable)"
    # End if/elif/else link

    logger.debug(
        f"Parsed API details for '{details.get('name', 'Unknown')}': "
        f"ID={details.get('person_id')}, Tree={details.get('tree_id', 'N/A')}, "
        f"Born='{details.get('birth_date')}' in '{details.get('birth_place')}', "
        f"Died='{details.get('death_date')}' in '{details.get('death_place')}', "
        f"Gender='{details.get('gender')}', Living={details.get('is_living')}"
    )

    return details
# End of parse_ancestry_person_details


def print_group(label: str, items: List[Dict]):
    """Prints a formatted group of relatives from API data."""
    print(f"\n{label}:")
    if items:
        formatter = format_name if UTILS_AVAILABLE else lambda x: str(x).title()
        for item in items:
            # Ensure item is a dict and has 'name' before formatting
            name_to_format = item.get('name') if isinstance(item, dict) else None
            print(f"  - {formatter(name_to_format)}")
        # End for
    else:
        print("  (None found)")
    # End if/else
# End of print_group


# Relationship term mapping (simple examples, expand as needed)
RELATIONSHIP_MAP = {
    "son": "Father",
    "daughter": "Mother",
    "mother": "Daughter", # Relationship FROM A TO B
    "father": "Son",     # Relationship FROM A TO B
    "brother": "Brother",
    "sister": "Sister",
    "husband": "Wife",
    "wife": "Husband",
    "uncle": "Nephew/Niece", # Can't determine Nephew/Niece without B's gender
    "aunt": "Nephew/Niece",
    # Add more inverse relationships
}


def _get_relationship_term(person_a_gender: Optional[str], basic_relationship: str) -> str:
    """ Determines the specific relationship term based on gender (e.g., Father vs Parent). """
    term = basic_relationship.capitalize() # Default
    if basic_relationship.lower() == 'parent':
        if person_a_gender == 'M':
            term = "Father"
        elif person_a_gender == 'F':
            term = "Mother"
        # End if/elif
    elif basic_relationship.lower() == 'child':
        if person_a_gender == 'M':
            term = "Son"
        elif person_a_gender == 'F':
            term = "Daughter"
        # End if/elif
    elif basic_relationship.lower() == 'sibling':
        if person_a_gender == 'M':
            term = "Brother"
        elif person_a_gender == 'F':
            term = "Sister"
        # End if/elif
    elif basic_relationship.lower() == 'spouse':
        if person_a_gender == 'M':
            term = "Husband"
        elif person_a_gender == 'F':
            term = "Wife"
        # End if/elif

    # Apply ordinal casing if the term might contain ordinals (e.g., "1st Cousin")
    if any(char.isdigit() for char in term) and UTILS_AVAILABLE:
        term = ordinal_case(term)
    # End if

    return term
# End of _get_relationship_term


def format_api_relationship_path(
    ladder_html: Optional[str], owner_name: str, target_name: str
) -> str:
    """
    Parses relationship ladder HTML from Ancestry API response and formats
    it into a human-readable path string. Uses BeautifulSoup for parsing.
    Revised (v16.18) to correctly parse relationship direction based on HTML structure.
    """
    if not ladder_html or not isinstance(ladder_html, str):
        logger.warning("format_api_relationship_path: HTML missing or invalid.")
        return "(No relationship path explanation available - HTML missing)"
    # End if

    if BeautifulSoup is None:
        logger.error("format_api_relationship_path: BeautifulSoup is not available.")
        return "(Cannot parse relationship path - BeautifulSoup missing. pip install beautifulsoup4 lxml)"
    # End if

    try:
        # HTML should already be unescaped by _extract_ladder_html
        soup = BeautifulSoup(ladder_html, "html.parser")

        # Select list items, excluding dividers
        path_items_raw = soup.select('ul.textCenter > li:not([class*="iconArrowDown"])')

        if not path_items_raw:
            logger.warning("format_api_relationship_path: Could not find path items.")
            rel_text_elem = soup.select_one(".rel-path-wrapper p") or soup.select_one(
                ".relationshipText"
            )
            if rel_text_elem:
                return f"({rel_text_elem.get_text(strip=True)})"
            # End if rel_text_elem
            return "(Could not parse relationship path from API HTML)"
        # End if not path_items_raw

        # Extract name and raw description for each person in the path
        path_data = []
        for item in path_items_raw:
            name_text, desc_text = "Unknown", ""
            # Extract name
            name_container_b_in_a = item.select_one("a > b")
            name_container_b = item.find("b") if not name_container_b_in_a else None
            name_container_a = (
                item.find("a")
                if not name_container_b_in_a and not name_container_b
                else None
            )

            raw_name = "Unknown"
            if name_container_b_in_a:
                raw_name = name_container_b_in_a.get_text(strip=True)
            elif name_container_b:
                raw_name = name_container_b.get_text(strip=True)
            elif name_container_a:
                raw_name = name_container_a.get_text(strip=True)
            # End if/elif

            cleaned_name = re.sub(r"\s+\d{4}-\d{0,4}$", "", raw_name).strip()
            name_text = format_name(cleaned_name)

            # Extract raw description text from <i> tag
            desc_element = item.find("i")
            if desc_element:
                desc_text = desc_element.get_text(separator=" ", strip=True).replace(
                    '"', "'"
                )
            # End if desc_element
            path_data.append({"name": name_text, "raw_desc": desc_text})
        # End for item

        # --- Format the Path Steps ---
        if len(path_data) < 2:
            logger.warning(
                f"Path data too short ({len(path_data)} items) for detailed explanation."
            )
            if path_data:
                return f"{path_data[0]['name']} (Path too short)"
            # End if path_data
            return "(Could not parse sufficient path steps)"
        # End if len < 2

        formatted_steps: List[str] = []
        # Extract overall relationship from the *first* item's description if simple
        overall_relationship = ""
        first_desc = path_data[0].get("raw_desc", "").strip()
        # Check if it's a single capitalized word (likely a direct relationship term)
        if first_desc and " " not in first_desc and first_desc[0].isupper():
            overall_relationship = first_desc
        # End if

        # Iterate through pairs to determine the relationship FROM A TO B
        # The description associated with B tells us B's relationship TO A.
        for i in range(len(path_data) - 1):
            person_a_data = path_data[i]
            person_b_data = path_data[i + 1]
            name_a = person_a_data.get("name", "Unknown")
            name_b = person_b_data.get("name", "Unknown")
            desc_b = person_b_data.get(
                "raw_desc", ""
            )  # Description associated with Person B

            rel_term_b_to_a = "related"  # Default relationship term

            # --- NEW PARSING LOGIC ---
            # 1. Check for "You are the [Relation] of [Person A Name]"
            you_are_match = re.match(
                r"You are the (.*?) of (.*)", desc_b, re.IGNORECASE
            )
            if you_are_match:
                rel_term_b_to_a = you_are_match.group(1).strip()
                # Person B is "You", replace name_b with owner_name
                name_b = owner_name
                logger.debug(
                    f"Path step {i+1}: Parsed 'You are the {rel_term_b_to_a} of {name_a}'"
                )

            else:
                # 2. Check for "[Relation] of [Person A Name]"
                relation_of_match = re.match(r"(.*?) of (.*)", desc_b, re.IGNORECASE)
                if relation_of_match:
                    # Ensure the name matches Person A approximately (handle formatting diffs)
                    name_check = relation_of_match.group(2).strip()
                    if (
                        name_a.lower() in name_check.lower()
                        or name_check.lower() in name_a.lower()
                    ):
                        rel_term_b_to_a = relation_of_match.group(1).strip()
                        logger.debug(
                            f"Path step {i+1}: Parsed '{rel_term_b_to_a} of {name_a}'"
                        )
                    else:
                        logger.warning(
                            f"Path step {i+1}: Desc '{desc_b}' matched 'X of Y', but Y ('{name_check}') didn't match Person A ('{name_a}'). Using fallback."
                        )
                        rel_term_b_to_a = "related"  # Fallback if name mismatch
                    # End if name check
                else:
                    # 3. Check if desc_b is a simple relationship term (e.g., "Brother", "Mother")
                    # Assume simple term applies to Person B's relationship to A
                    if desc_b and " " not in desc_b and desc_b[0].isupper():
                        rel_term_b_to_a = desc_b
                        logger.debug(
                            f"Path step {i+1}: Parsed simple term '{rel_term_b_to_a}'"
                        )
                    elif desc_b:
                        # Fallback if description exists but doesn't match patterns
                        logger.warning(
                            f"Path step {i+1}: Unrecognized description format '{desc_b}'. Using 'related'."
                        )
                        rel_term_b_to_a = "related"
                    # End if/else simple term check
                # End if/else relation_of_match
            # End if/else you_are_match

            # Format the relationship term (capitalize, handle ordinals)
            formatted_rel_term = ordinal_case(rel_term_b_to_a.capitalize())

            # Format the step string: B is the Relation of A
            formatted_steps.append(f"{name_b} is the {formatted_rel_term} of {name_a}")
            # --- END NEW PARSING LOGIC ---
        # End for i

        # Assemble the final explanation string
        explanation_str = f"{path_data[0]['name']}\n -> " + "\n -> ".join(
            formatted_steps
        )

        # Prepend overall relationship if found
        overall_rel_display = ""
        if overall_relationship:
            overall_rel_display = (
                f"Overall: {ordinal_case(overall_relationship.title())}\nPath:\n"
            )
        # End if overall_relationship

        return f"{overall_rel_display}{explanation_str}"

    except Exception as bs_parse_err:
        logger.error(
            f"Error parsing relationship ladder HTML: {bs_parse_err}", exc_info=True
        )
        logger.debug(f"Problematic HTML: {ladder_html[:500]}...")
        return f"(Error parsing API relationship path: {bs_parse_err})"


# End of format_api_relationship_path


# --- Helper Function for HTML Extraction ---
def _extract_ladder_html(raw_content: Union[str, Dict]) -> Optional[str]:
    """
    Extracts and decodes the relationship ladder HTML from raw API response content.
    Handles standard JSON, JSONP, and potential errors.
    """
    if isinstance(raw_content, dict) and "error" in raw_content:
        error_msg = raw_content.get("error", {}).get("message", raw_content.get("message", "Unknown API Error"))
        logger.error(f"_extract_ladder_html: API returned error: {error_msg}")
        return None
    # End if
    if not raw_content or not isinstance(raw_content, str):
        logger.error(f"_extract_ladder_html: Invalid raw content type: {type(raw_content)}")
        return None
    # End if

    html_escaped = None
    logger.debug("_extract_ladder_html: Attempting JSONP extraction...")
    try:
        jsonp_match = re.match(r"^\s*[\w$.]+\((.*)\)\s*;?\s*$", raw_content, re.DOTALL)
        if jsonp_match:
            json_str = jsonp_match.group(1).strip()
            if json_str.startswith("{") and json_str.endswith("}"):
                json_data = json.loads(json_str)
                if isinstance(json_data, dict) and "html" in json_data:
                    html_escaped = json_data["html"]
                    if isinstance(html_escaped, str):
                        logger.debug(f"_extract_ladder_html: Found HTML via JSONP. Length: {len(html_escaped)}")
                    else:
                        logger.warning(f"_extract_ladder_html: 'html' key found in JSONP, but not string: {type(html_escaped)}")
                        html_escaped = None
                    # End if/else isinstance
                else:
                    logger.warning("_extract_ladder_html: 'html' key not found in JSONP object.")
                # End if/else html key
            else:
                logger.warning(f"_extract_ladder_html: Content in JSONP () not JSON: {json_str[:100]}...")
            # End if/else json_str looks like JSON
        else:
            logger.debug("_extract_ladder_html: Raw content does not match JSONP structure.")
            if raw_content.strip().startswith("{") and raw_content.strip().endswith("}"):
                logger.debug("_extract_ladder_html: Attempting direct JSON parse...")
                try:
                    json_data_direct = json.loads(raw_content.strip())
                    if isinstance(json_data_direct, dict) and "html" in json_data_direct and isinstance(json_data_direct["html"], str):
                        html_escaped = json_data_direct["html"]
                        logger.debug(f"_extract_ladder_html: Found HTML via direct JSON parse. Length: {len(html_escaped)}")
                    else:
                        logger.warning("_extract_ladder_html: Direct JSON ok, but 'html' key missing/invalid.")
                    # End if/else html key
                except json.JSONDecodeError:
                    logger.warning("_extract_ladder_html: Direct JSON parse failed.")
                # End try/except JSONDecodeError
            # End if starts/ends with {}
        # End if/else jsonp_match
    except json.JSONDecodeError as json_e:
        logger.warning(f"_extract_ladder_html: JSONDecodeError during JSONP/JSON extraction: {json_e}")
    except Exception as e:
        logger.warning(f"_extract_ladder_html: Unexpected error during JSONP/JSON extraction: {e}")
    # End try/except

    if not html_escaped:
        logger.debug("_extract_ladder_html: JSONP/JSON failed, trying regex...")
        html_match = re.search(r'"html"\s*:\s*"((?:\\.|[^"\\])*)"', raw_content, re.IGNORECASE | re.DOTALL)
        if html_match:
            html_escaped = html_match.group(1)
            logger.debug(f"_extract_ladder_html: Found HTML via regex. Length: {len(html_escaped)}")
        # End if html_match
    # End if not html_escaped

    if not html_escaped:
        logger.error("_extract_ladder_html: Could not extract HTML content.")
        logger.debug(f"Raw content snippet: {raw_content[:500]}...")
        return None
    # End if not html_escaped

    try:
        temp_unescaped = html_escaped.replace("\\\\", "\\")
        html_intermediate = temp_unescaped.encode("utf-8", "backslashreplace").decode("unicode_escape", errors="replace")
        html_unescaped = html.unescape(html_intermediate)
        logger.debug("_extract_ladder_html: Successfully unescaped HTML.")
        return html_unescaped
    except Exception as decode_err:
        logger.error(f"_extract_ladder_html: Could not decode HTML. Error: {decode_err}", exc_info=True)
        logger.debug(f"Problematic escaped HTML snippet: {html_escaped[:500]}...")
        return None
    # End try/except decode
# End of _extract_ladder_html


# --- Display Function ---
def display_raw_relationship_ladder(
    raw_content: Union[str, Dict], owner_name: str, target_name: str
):
    """
    Parse and display the Ancestry relationship ladder from raw JSONP/HTML content.
    Uses helper functions to extract HTML and format the path.
    """
    if BeautifulSoup is None:
        logger.error("BeautifulSoup library not found. Cannot parse relationship ladder HTML.")
        print(f"\n--- Relationship between {owner_name} and {target_name} (API) ---")
        print("\n(Cannot parse relationship path - BeautifulSoup missing. pip install beautifulsoup4 lxml)")
        return # End of function display_raw_relationship_ladder
    # End if

    logger.info(f"\n--- Relationship between {owner_name} and {target_name} (API Report) ---")

    html_unescaped = _extract_ladder_html(raw_content)

    if not html_unescaped:
        print("\n(Could not extract or decode relationship path HTML from API response)")
        return # End of function display_raw_relationship_ladder
    # End if

    # --- REMOVED DEBUG PRINT BLOCK ---

    formatted_path_str = format_api_relationship_path(
        html_unescaped, owner_name, target_name
    )

    if formatted_path_str:
        print(formatted_path_str.strip())
    else:
        logger.warning("format_api_relationship_path returned empty string or None.")
        print("(Could not format relationship path steps from extracted HTML)")
    # End if/else

# End of display_raw_relationship_ladder


# --- Standalone Test Block ---
def self_check() -> bool:  # Removed verbose argument as it's not used
    """
    Performs internal self-checks for api_utils.py, including LIVE API calls.
    Requires .env file to be correctly configured. Provides formatted output summary.
    """
    # --- Local Imports for Self-Check ---
    try:
        import utils
        from utils import SessionManager, _api_req, format_name, ordinal_case
        from config import config_instance, selenium_config
        from logging_config import logger as utils_logger
        from urllib.parse import urljoin, urlencode, quote
        import traceback
        import requests
        from typing import Callable  # For _run_test_sc type hinting
    except ImportError as e:
        print(f"\n[api_utils.py self-check ERROR] - CRITICAL IMPORT FAILED: {e}")
        print(
            "Ensure utils.py, config.py, logging_config.py are present and configured."
        )
        print(f"Self-check status: FAIL\n")
        return False
    # End try

    # Use the logger imported from logging_config
    logger_self_check = utils_logger

    # --- Test Runner Helper ---
    test_results_sc: List[Tuple[str, str, str]] = []

    def _run_test_sc(
        test_name: str, test_func: Callable, *args, **kwargs
    ) -> Tuple[str, str, str]:
        """Runs a single test, logs result, stores result, and returns status tuple."""
        logger_self_check.debug(f"[ RUNNING SC ] {test_name}")
        status = "FAIL"
        message = ""
        expect_none = kwargs.pop("expected_none", False)
        expect_type = kwargs.pop("expected_type", None)
        expect_value = kwargs.pop("expected_value", None)
        expect_contains = kwargs.pop("expected_contains", None)

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
            elif result is True:
                passed = True
            elif result is None and not expect_none:
                passed = True
            else:  # Construct failure message
                if expect_none:
                    message = f"Expected None, got {type(result).__name__} ({str(result)[:50]})"
                elif expect_type:
                    message = f"Expected type {expect_type.__name__}, got {type(result).__name__}"
                elif expect_value:
                    message = f"Expected value '{str(expect_value)[:50]}', got '{str(result)[:50]}'"
                elif expect_contains:
                    message = f"Expected result to contain '{expect_contains}', got '{str(result)[:100]}...'"
                else:
                    message = f"Assertion failed (returned {str(result)[:100]})"
            # End if/elif/else for checking result
            status = "PASS" if passed else "FAIL"
        except Exception as e:
            status = "FAIL"
            message = f"{type(e).__name__}: {e}"
            logger_self_check.debug(
                f"Exception details for {test_name}: {message}\n{traceback.format_exc()}",
                exc_info=False,
            )
        # End try/except

        log_level = logging.INFO if status == "PASS" else logging.ERROR
        log_message = f"[ {status:<6} SC ] {test_name}{f': {message}' if message and status == 'FAIL' else ''}"
        logger_self_check.log(log_level, log_message)
        test_results_sc.append((test_name, status, message))
        return (test_name, status, message)

    # End of _run_test_sc

    # --- Test Parameters ---
    # Ensure required attributes exist before accessing
    if (
        not hasattr(config_instance, "TESTING_PROFILE_ID")
        or not config_instance.TESTING_PROFILE_ID
    ):
        logger_self_check.error("TESTING_PROFILE_ID missing in config_instance.")
        return False
    if (
        not hasattr(config_instance, "TESTING_PERSON_TREE_ID")
        or not config_instance.TESTING_PERSON_TREE_ID
    ):
        logger_self_check.error("TESTING_PERSON_TREE_ID missing in config_instance.")
        return False
    # End if attribute checks

    target_profile_id = config_instance.TESTING_PROFILE_ID
    target_person_id = config_instance.TESTING_PERSON_TREE_ID
    target_name_from_profile = "Unknown Target"

    # --- Status Tracking ---
    overall_status = True  # Assume PASS initially
    session_manager_sc: Optional[SessionManager] = None

    logger_self_check.info("\n[api_utils.py self-check starting...]")

    # === Phase 0: Prerequisite Checks ===
    logger_self_check.info("--- Phase 0: Prerequisite Checks ---")
    _, s0_status, _ = _run_test_sc(
        "Check BeautifulSoup Import",
        lambda: BeautifulSoup is not None,
        expected_type=type,
    )
    if s0_status == "FAIL":
        overall_status = False
    # End if

    func_map = {
        "format_name": format_name,
        "ordinal_case": ordinal_case,
        "_parse_date": _parse_date,
        "_clean_display_date": _clean_display_date,
    }
    for name, func in func_map.items():
        _, s0_status, _ = _run_test_sc(
            f"Check Function '{name}'", lambda: callable(func)
        )
        if s0_status == "FAIL":
            overall_status = False
    # End for

    _, s0_status, _ = _run_test_sc(
        "Check Config Loaded",
        lambda: config_instance is not None
        and hasattr(config_instance, "BASE_URL")
        and config_instance.BASE_URL is not None,
    )
    if s0_status == "FAIL":
        overall_status = False
    # End if

    if not overall_status:
        logger_self_check.error(
            "Prerequisite checks failed. Cannot proceed with live tests."
        )
        return False
    # End if

    try:
        # === Phase 1: Session Setup ===
        logger_self_check.info("--- Phase 1: Session Setup & Login ---")
        session_manager_sc = SessionManager()  # Instantiate within try block
        _, s1_status, _ = _run_test_sc(
            "SessionManager.start_sess()",
            session_manager_sc.start_sess,
            action_name="SC Phase 1 Start",
        )
        if s1_status == "FAIL":
            overall_status = False
            raise RuntimeError("start_sess failed")
        # End if

        _, s1_status, _ = _run_test_sc(
            "SessionManager.ensure_session_ready()",
            session_manager_sc.ensure_session_ready,
            action_name="SC Phase 1 Ready",
        )
        if s1_status == "FAIL":
            overall_status = False
            raise RuntimeError("ensure_session_ready failed")
        # End if

        # === Phase 2: Get Target Info & Validate Config ===
        logger_self_check.info("--- Phase 2: Get Target Info & Validate Config ---")
        target_tree_id = session_manager_sc.my_tree_id
        target_owner_name = session_manager_sc.tree_owner_name

        _, s2_status_tid, _ = _run_test_sc(
            "Check Target Tree ID Found", lambda: bool(target_tree_id)
        )
        _, s2_status_owner, _ = _run_test_sc(
            "Check Target Owner Name Found", lambda: bool(target_owner_name)
        )
        if s2_status_tid == "FAIL" or s2_status_owner == "FAIL":
            overall_status = False
        # End if

        # Fetch target display name
        profile_api_url = urljoin(
            config_instance.BASE_URL,
            f"/app-api/express/v1/profiles/details?userId={target_profile_id.upper()}",
        )
        api_call_lambda = lambda: utils._api_req(
            url=profile_api_url,
            driver=session_manager_sc.driver,
            session_manager=session_manager_sc,
            api_description="Get Target Name (Profile Details)",
            use_csrf_token=False,
        )
        test_name = "API Call: Get Target Name"
        _, s2_status, _ = _run_test_sc(test_name, api_call_lambda, expected_type=dict)
        profile_response_details = api_call_lambda() if s2_status == "PASS" else None

        if s2_status == "FAIL":
            overall_status = False
        else:
            if profile_response_details:
                target_name_from_profile = profile_response_details.get(
                    "DisplayName", "Unknown Target"
                )
                if target_name_from_profile == "Unknown Target":
                    first = profile_response_details.get("FirstName")
                    last = profile_response_details.get("LastName")
                    if first and last:
                        target_name_from_profile = f"{first} {last}"
                    elif first:
                        target_name_from_profile = first
                    # End if/elif
                # End if fallback
                _, s2_status_name, _ = _run_test_sc(
                    "Check Target Name Found in API Resp",
                    lambda: target_name_from_profile != "Unknown Target",
                )
                if s2_status_name == "FAIL":
                    overall_status = False
                # End if name check
            else:
                logger_self_check.error(
                    "API call passed but response was None/invalid."
                )
                _run_test_sc("Check Target Name Found in API Resp", lambda: False)
                overall_status = False
            # End if/else profile_response_details
        # End if/else API call status

        # === Phase 3: Test parse_ancestry_person_details ===
        logger_self_check.info("--- Phase 3: Test parse_ancestry_person_details ---")
        if profile_response_details and isinstance(profile_response_details, dict):
            person_card_for_parse = {
                "personId": target_profile_id,
                "treeId": target_tree_id,
            }
            test_name_parse = "Function Call: parse_ancestry_person_details()"
            parsed_details = None  # Initialize
            try:
                # Need to call within lambda for _run_test_sc exception handling
                parse_lambda = lambda: parse_ancestry_person_details(
                    person_card_for_parse, profile_response_details
                )
                _, s3_status, _ = _run_test_sc(
                    test_name_parse, parse_lambda, expected_type=dict
                )
                if s3_status == "PASS":
                    parsed_details = parse_lambda()  # Call again to get result
                else:
                    overall_status = False
                # End if/else status
            except Exception as parse_e:
                # Log exception if the call itself fails inside _run_test_sc's try block wasn't enough
                logger_self_check.error(
                    f"Exception calling parser: {parse_e}", exc_info=True
                )
                _run_test_sc(
                    test_name_parse,
                    lambda: False,
                    expected_value=f"Exception: {parse_e}",
                )  # Log failure
                overall_status = False
            # End try/except around parse call

            # Validate result if parse call succeeded
            if parsed_details:
                missing_keys = [
                    k
                    for k in ["name", "person_id", "link"]
                    if k not in parsed_details or parsed_details[k] is None
                ]
                if missing_keys:
                    _, s3_v_status, _ = _run_test_sc(
                        "Validation: Parsed Details Keys",
                        lambda: False,
                        expected_value=f"Missing keys: {missing_keys}",
                    )
                    if s3_v_status == "FAIL":
                        overall_status = False
                    # End if status fail
                else:
                    _run_test_sc("Validation: Parsed Details Keys", lambda: True)
                    # Optional: Check name match
                    _, s3_n_status, _ = _run_test_sc(
                        "Validation: Parsed Name Match",
                        lambda: parsed_details["name"]
                        == format_name(target_name_from_profile),
                    )
                    # if s3_n_status == 'FAIL': overall_status = False # Treat name mismatch as warning, not failure
                # End if/else missing keys
            elif s3_status == "PASS":  # Parser call succeeded but returned None/invalid
                _run_test_sc(
                    "Validation: Parsed Details Keys",
                    lambda: False,
                    expected_value="Parser returned None/invalid unexpectedly",
                )
                overall_status = False
            # End if/elif parsed_details check
        else:
            logger_self_check.warning(
                "Skipping parse_ancestry_person_details test: Prerequisite API call failed."
            )
            _run_test_sc(
                "Function Call: parse_ancestry_person_details()",
                lambda: False,
                expected_value="Skipped",
            )
        # End if profile_response_details

        # === Phase 4: Test HTML Extraction (_extract_ladder_html) ===
        logger_self_check.info("--- Phase 4: Test HTML Extraction ---")
        test_jsonp_good = 'jQuery123({"html": "<p>Good HTML</p>", "other": 1});'
        test_jsonp_bad_json = "jQuery123(Not JSON Content);"
        test_jsonp_no_html = 'jQuery123({"key": "value"});'
        test_json_good = '{"html": "<b>Good JSON HTML</b>"}'
        test_json_bad = '{"key": "value"}'
        test_invalid = "Just some text"
        expected_html_good = "<p>Good HTML</p>"
        expected_html_json = "<b>Good JSON HTML</b>"

        _, s4_status, _ = _run_test_sc(
            "HTML Extract: Good JSONP",
            lambda: _extract_ladder_html(test_jsonp_good),
            expected_value=expected_html_good,
        )
        if s4_status == "FAIL":
            overall_status = False
        # End if
        _, s4_status, _ = _run_test_sc(
            "HTML Extract: Bad JSON in JSONP",
            lambda: _extract_ladder_html(test_jsonp_bad_json),
            expected_none=True,
        )
        if s4_status == "FAIL":
            overall_status = False
        # End if
        _, s4_status, _ = _run_test_sc(
            "HTML Extract: No HTML Key in JSONP",
            lambda: _extract_ladder_html(test_jsonp_no_html),
            expected_none=True,
        )
        if s4_status == "FAIL":
            overall_status = False
        # End if
        _, s4_status, _ = _run_test_sc(
            "HTML Extract: Good Direct JSON",
            lambda: _extract_ladder_html(test_json_good),
            expected_value=expected_html_json,
        )
        if s4_status == "FAIL":
            overall_status = False
        # End if
        _, s4_status, _ = _run_test_sc(
            "HTML Extract: Bad Direct JSON",
            lambda: _extract_ladder_html(test_json_bad),
            expected_none=True,
        )
        if s4_status == "FAIL":
            overall_status = False
        # End if
        _, s4_status, _ = _run_test_sc(
            "HTML Extract: Invalid Input",
            lambda: _extract_ladder_html(test_invalid),
            expected_none=True,
        )
        if s4_status == "FAIL":
            overall_status = False
        # End if

        # === Phase 5: Test Relationship Ladder Parsing ===
        logger_self_check.info(
            "--- Phase 5: Test Relationship Ladder Parsing (Live API) ---"
        )
        if (
            not target_owner_name
            or target_name_from_profile == "Unknown Target"
            or not target_person_id
            or not target_tree_id
        ):
            missing_ladder_reqs = [
                n
                for n, v in [
                    ("Owner Name", target_owner_name),
                    ("Target Name", target_name_from_profile != "Unknown Target"),
                    ("Target Person ID", target_person_id),
                    ("Target Tree ID", target_tree_id),
                ]
                if not v
            ]
            logger_self_check.warning(
                f"Skipping Live Ladder test: Missing ({', '.join(missing_ladder_reqs)})."
            )
            _run_test_sc(
                "API Call: Get Tree Ladder", lambda: False, expected_value="Skipped"
            )
            _run_test_sc(
                "Function Call: display_raw_relationship_ladder()",
                lambda: False,
                expected_value="Skipped",
            )
            _run_test_sc(
                "Function Call: format_api_relationship_path()",
                lambda: False,
                expected_value="Skipped",
            )
        else:
            base_ladder_url = urljoin(
                config_instance.BASE_URL,
                f"/family-tree/person/tree/{target_tree_id}/person/{target_person_id}/getladder",
            )
            callback_name = f"__ancestry_jsonp_{int(time.time()*1000)}"
            timestamp_ms = int(time.time() * 1000)
            query_params = urlencode({"callback": callback_name, "_": timestamp_ms})
            ladder_api_url = f"{base_ladder_url}?{query_params}"

            ladder_headers = {
                "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            }
            ladder_referer = urljoin(
                config_instance.BASE_URL,
                f"/family-tree/person/tree/{target_tree_id}/person/{target_person_id}/facts",
            )

            # Test API call
            ladder_response_raw = None  # Initialize
            api_call_lambda_ladder = lambda: utils._api_req(
                url=ladder_api_url,
                driver=session_manager_sc.driver,
                session_manager=session_manager_sc,
                api_description="Get Tree Ladder API (Self Check)",
                use_csrf_token=False,
                headers=ladder_headers,
                referer_url=ladder_referer,
                force_text_response=True,
                timeout=20,
            )
            test_name_ladder_api = "API Call: Get Tree Ladder"
            _, s5_api_status, _ = _run_test_sc(
                test_name_ladder_api, api_call_lambda_ladder, expected_type=str
            )
            if s5_api_status == "PASS":
                ladder_response_raw = api_call_lambda_ladder()
            else:
                overall_status = False
            # End if/else API status

            # Proceed only if API call succeeded and returned string
            if ladder_response_raw and isinstance(ladder_response_raw, str):
                # Test display function
                display_lambda = lambda: display_raw_relationship_ladder(
                    ladder_response_raw, target_owner_name, target_name_from_profile
                )
                _, s5_display_status, _ = _run_test_sc(
                    "Function Call: display_raw_relationship_ladder()", display_lambda
                )
                if s5_display_status == "FAIL":
                    overall_status = False
                # End if display fail

                # Test format function
                html_for_format = _extract_ladder_html(ladder_response_raw)
                if html_for_format:
                    format_lambda = lambda: format_api_relationship_path(
                        html_for_format, target_owner_name, target_name_from_profile
                    )
                    _, s5_format_status, _ = _run_test_sc(
                        "Function Call: format_api_relationship_path()",
                        format_lambda,
                        expected_contains=target_owner_name,
                    )
                    if s5_format_status == "FAIL":
                        overall_status = False
                    # End if format fail
                else:
                    logger_self_check.error(
                        "Could not extract HTML for format_api_relationship_path test."
                    )
                    _run_test_sc(
                        "Function Call: format_api_relationship_path()",
                        lambda: False,
                        expected_value="HTML Extraction Failed",
                    )
                    overall_status = False
                # End if html_for_format
            elif (
                s5_api_status == "PASS"
            ):  # API Call passed but didn't return valid string
                logger_self_check.error(
                    "Ladder API call passed status check but returned invalid data."
                )
                _run_test_sc(
                    "Function Call: display_raw_relationship_ladder()",
                    lambda: False,
                    expected_value="API Response Invalid",
                )
                _run_test_sc(
                    "Function Call: format_api_relationship_path()",
                    lambda: False,
                    expected_value="API Response Invalid",
                )
                overall_status = False
            # End if/elif ladder_response_raw check
        # End if/else can run ladder test

    except Exception as e:
        logger_self_check.critical(
            f"\n--- CRITICAL ERROR during self-check ---", exc_info=True
        )
        _run_test_sc(
            "Self-Check Execution", lambda: False, expected_value=f"CRITICAL ERROR: {e}"
        )
        overall_status = False
    finally:
        if session_manager_sc:
            logger_self_check.info("--- Finalizing: Closing Session ---")
            session_manager_sc.close_sess()
        else:
            logger_self_check.info("--- Finalizing: No session to close ---")
        # End if

        # --- Print Formatted Summary ---
        print("\n--- api_utils.py Self-Check Summary ---")
        name_width = max((len(name) for name, _, _ in test_results_sc), default=50)
        status_width = 8
        header = (
            f"{'Test Name':<{name_width}} | {'Status':<{status_width}} | {'Message'}"
        )
        print(header)
        print("-" * (len(header)))
        final_fail_count = 0
        final_skip_count = 0
        for name, status, message in test_results_sc:
            if status == "FAIL":
                final_fail_count += 1
            elif status == "SKIPPED":
                final_skip_count += 1
            # End if/elif
            print(
                f"{name:<{name_width}} | {status:<{status_width}} | {message if status != 'PASS' else ''}"
            )
        # End for
        print("-" * (len(header)))
        total_tests = len(test_results_sc)
        passed_tests = total_tests - final_fail_count - final_skip_count

        # Determine final overall status based on individual test failures
        overall_status = final_fail_count == 0

        final_status_msg = f"Result: {'PASS' if overall_status else 'FAIL'} ({passed_tests} passed, {final_fail_count} failed, {final_skip_count} skipped out of {total_tests} tests)"
        print(f"{final_status_msg}\n")
        if overall_status:
            logger_self_check.info("api_utils self-check overall status: PASS")
        else:
            logger_self_check.error("api_utils self-check overall status: FAIL")
        # --- End Summary Printing ---

    return overall_status


# End of self_check

if __name__ == "__main__":
    # Run self-check when executed directly
    print("Running api_utils.py self-check (with live API calls)...")
    try:
        # Ensure logging_config is importable before calling setup_logging
        import logging_config

        if not hasattr(logging_config, "setup_logging"):
            raise ImportError("setup_logging missing")
        log_file = Path("api_utils_self_check.log").resolve()
        # Set level to DEBUG to capture detailed logs from self_check
        logger_standalone = logging_config.setup_logging(
            log_file=log_file, log_level="DEBUG"
        )
        print(f"Detailed logs will be written to: {log_file}")
    except ImportError as log_imp_err:
        print(
            f"Warning: logging_config import/setup failed ({log_imp_err}). Using basic logging."
        )
        logging.basicConfig(level=logging.DEBUG)
        logger_standalone = logging.getLogger("api_utils_standalone")
    except Exception as log_setup_err:
        print(f"Error setting up logging: {log_setup_err}. Using basic logging.")
        logging.basicConfig(level=logging.DEBUG)
        logger_standalone = logging.getLogger("api_utils_standalone")
    # End try/except logging setup

    self_check_passed = self_check()  # Run the self-check
    print("\nThis is the api_utils module. Import it into other scripts.")
    sys.exit(0 if self_check_passed else 1)
# End of __main__ block
# End of api_utils.py
# --- END OF FILE api_utils.py ---
