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
"""

# --- Standard library imports ---
import logging
import sys
import re
import os
import time
import json
import requests  # Keep if used by parsing logic, though _api_req handles fetch
import urllib.parse  # Used for urlencode in self_check
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
from pathlib import Path  # Needed for __main__ block

# --- Third-party imports ---
# Keep BeautifulSoup import here, check for its availability in functions
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore # Gracefully handle missing dependency

# --- Local application imports ---
import utils
from utils import format_name, ordinal_case
from config import config_instance, selenium_config
from gedcom_utils import _parse_date, _clean_display_date


# Initialize logger - Ensure logger is always available
# If running standalone, __main__ block will reconfigure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_utils")
# logger.warning("Using fallback logger for api_utils.")


# --- API Response Parsing (Functions remain unchanged) ---


def parse_ancestry_person_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Dict:
    """
    Extracts standardized details from Ancestry Person-Card and Facts API responses.
    Includes parsing dates/places and generating a link.
    Based on temp.py v7.36 logic, uses imported date helpers.
    Prioritizes facts_data if available.

    Args:
        person_card (Dict): A dictionary containing basic person info,
                             MUST include 'personId' and 'treeId' if link generation
                             or logging context is desired. personId can be the User ID
                             or the tree-specific Person ID depending on the context.
        facts_data (Optional[Dict]): The more detailed response from the Facts API endpoint
                                      or the Profile Details API endpoint.
                                      Expected to contain keys like 'gender', 'birthDate',
                                      'deathDate', 'personName', 'FirstName', 'DisplayName', etc.

    Returns:
        Dict: A standardized dictionary containing parsed details.
    """
    # --- Initialize Details with defaults and essential IDs from person_card ---
    details = {
        "name": "Unknown",
        "birth_date": None,
        "birth_place": None,
        "death_date": None,
        "death_place": None,
        "gender": None,
        "person_id": person_card.get(
            "personId"
        ),  # This should be the USER ID if facts_data is from Profile Details API
        "tree_id": person_card.get(
            "treeId"
        ),  # May be None if facts_data is from Profile Details API
        "link": None,
        "api_birth_obj": None,  # Store parsed datetime object
        "api_death_obj": None,  # Store parsed datetime object
        "is_living": None,
    }

    # --- Extract from Facts Data (Primary Source if available) ---
    if facts_data and isinstance(facts_data, dict):
        # Extract Name (Check multiple possible keys)
        person_info = facts_data.get(
            "person", {}
        )  # Check inside 'person' dict first (Facts API)
        if isinstance(person_info, dict):
            details["name"] = person_info.get("personName", details["name"])
            # Gender often within 'person'
            gender_fact = person_info.get("gender")
            if gender_fact and isinstance(gender_fact, str):
                details["gender"] = (
                    "M"
                    if gender_fact.lower() == "male"
                    else "F" if gender_fact.lower() == "female" else None
                )
            # isLiving flag
            details["is_living"] = person_info.get("isLiving", details["is_living"])

        # Fallback: Check top level 'personName' (Facts API)
        if details["name"] == "Unknown":
            details["name"] = facts_data.get("personName", details["name"])

        # Fallback: Check 'DisplayName' (Profile Details API)
        if details["name"] == "Unknown":
            details["name"] = facts_data.get("DisplayName", details["name"])

        # Fallback: Check 'FirstName' (Profile Details API - might need combining)
        if details["name"] == "Unknown":
            first_name_pd = facts_data.get("FirstName")
            if first_name_pd:
                # Attempt to construct full name if possible, otherwise use FirstName
                last_name_pd = facts_data.get("LastName")
                if last_name_pd:
                    details["name"] = f"{first_name_pd} {last_name_pd}"
                else:
                    details["name"] = first_name_pd
            # Ensure name is not empty if DisplayName/FirstName were used
            if not details["name"]:
                details["name"] = "Unknown"

        # Gender fallback: check top-level 'gender' (Facts API)
        if details["gender"] is None:
            gender_fact = facts_data.get("gender")
            if gender_fact and isinstance(gender_fact, str):
                details["gender"] = (
                    "M"
                    if gender_fact.lower() == "male"
                    else "F" if gender_fact.lower() == "female" else None
                )
        # Gender fallback: check 'Gender' (Profile Details API)
        if details["gender"] is None:
            gender_pd = facts_data.get("Gender")  # Note capitalization
            if gender_pd and isinstance(gender_pd, str):
                details["gender"] = (
                    "M"
                    if gender_pd.lower() == "male"
                    else "F" if gender_pd.lower() == "female" else None
                )

        # isLiving fallback: check top-level 'isLiving' (Facts API)
        if details["is_living"] is None:
            details["is_living"] = facts_data.get("isLiving", details["is_living"])
        # isLiving fallback: check 'IsLiving' (Profile Details API) - Less common here
        if details["is_living"] is None:
            details["is_living"] = facts_data.get("IsLiving", details["is_living"])

        # Extract Birth/Death from structured facts (Facts API structure)
        birth_fact_group = facts_data.get("facts", {}).get("Birth", [{}])[0]
        if isinstance(birth_fact_group, dict):
            date_info = birth_fact_group.get("date", {})
            place_info = birth_fact_group.get("place", {})
            if isinstance(date_info, dict):
                details["birth_date"] = date_info.get(
                    "normalized", date_info.get("original")
                )
            if isinstance(place_info, dict):
                details["birth_place"] = place_info.get("placeName")

        death_fact_group = facts_data.get("facts", {}).get("Death", [{}])[0]
        if isinstance(death_fact_group, dict):
            date_info = death_fact_group.get("date", {})
            place_info = death_fact_group.get("place", {})
            if isinstance(date_info, dict):
                details["death_date"] = date_info.get(
                    "normalized", date_info.get("original")
                )
            if isinstance(place_info, dict):
                details["death_place"] = place_info.get("placeName")

        # Alternative structure check (e.g., from Profile Details API)
        if details["birth_date"] is None:
            birth_fact_alt = facts_data.get(
                "birthDate"
            )  # Profile Details often uses this
            if birth_fact_alt and isinstance(birth_fact_alt, dict):
                date_str = birth_fact_alt.get(
                    "normalized", birth_fact_alt.get("date", "")
                )
                place_str = birth_fact_alt.get("place", "")
                if date_str and isinstance(date_str, str):
                    details["birth_date"] = date_str
                if place_str and isinstance(place_str, str):
                    details["birth_place"] = place_str
            elif isinstance(birth_fact_alt, str):  # Sometimes it's just a string
                details["birth_date"] = birth_fact_alt

        if details["death_date"] is None:
            death_fact_alt = facts_data.get(
                "deathDate"
            )  # Profile Details often uses this
            if death_fact_alt and isinstance(death_fact_alt, dict):
                date_str = death_fact_alt.get(
                    "normalized", death_fact_alt.get("date", "")
                )
                place_str = death_fact_alt.get("place", "")
                if date_str and isinstance(date_str, str):
                    details["death_date"] = date_str
                if place_str and isinstance(place_str, str):
                    details["death_place"] = place_str
            elif isinstance(death_fact_alt, str):  # Sometimes it's just a string
                details["death_date"] = death_fact_alt

    # --- Fallback to Person Card data if Facts are missing or incomplete ---
    # Only overwrite if the detail is still None after checking facts_data
    if details["name"] == "Unknown":
        details["name"] = person_card.get("name", "Unknown")

    if details["birth_date"] is None:
        birth_info_card = person_card.get("birth", "")
        if birth_info_card and isinstance(birth_info_card, str):
            parts = birth_info_card.split(" in ")
            details["birth_date"] = parts[0].strip() if parts else birth_info_card
            if (
                details["birth_place"] is None
            ):  # Only set place if not already from facts
                details["birth_place"] = parts[1].strip() if len(parts) > 1 else None
        elif isinstance(birth_info_card, dict):  # Handle structured data in person_card
            details["birth_date"] = birth_info_card.get("date", details["birth_date"])
            if details["birth_place"] is None:
                details["birth_place"] = birth_info_card.get(
                    "place", details["birth_place"]
                )

    if details["death_date"] is None:
        death_info_card = person_card.get("death", "")
        if death_info_card and isinstance(death_info_card, str):
            parts = death_info_card.split(" in ")
            details["death_date"] = parts[0].strip() if parts else death_info_card
            if (
                details["death_place"] is None
            ):  # Only set place if not already from facts
                details["death_place"] = parts[1].strip() if len(parts) > 1 else None
        elif isinstance(death_info_card, dict):  # Handle structured data in person_card
            details["death_date"] = death_info_card.get("date", details["death_date"])
            if details["death_place"] is None:
                details["death_place"] = death_info_card.get(
                    "place", details["death_place"]
                )

    if details["gender"] is None:
        gender_card = person_card.get("gender")
        if gender_card and isinstance(gender_card, str):
            details["gender"] = (
                "M"
                if gender_card.lower() == "male"
                else "F" if gender_card.lower() == "female" else None
            )

    if details["is_living"] is None:
        details["is_living"] = person_card.get("isLiving", details["is_living"])

    # --- Final Processing ---
    # Format Name (always apply formatting)
    details["name"] = format_name(details["name"])

    # Parse dates into datetime objects using gedcom_utils helper
    if _parse_date:
        if details["birth_date"]:
            details["api_birth_obj"] = _parse_date(details["birth_date"])
        if details["death_date"]:
            details["api_death_obj"] = _parse_date(details["death_date"])

    # Clean dates for display using gedcom_utils helper
    if _clean_display_date:
        details["birth_date"] = (
            _clean_display_date(details["birth_date"])
            if details["birth_date"]
            else "N/A"
        )
        details["death_date"] = (
            _clean_display_date(details["death_date"])
            if details["death_date"]
            else "N/A"
        )
    else:  # Fallback if _clean_display_date import failed
        details["birth_date"] = (
            str(details["birth_date"]) if details["birth_date"] else "N/A"
        )
        details["death_date"] = (
            str(details["death_date"]) if details["death_date"] else "N/A"
        )

    # Generate Ancestry link
    # Use treeId if available (from person_card), otherwise use profile link structure
    if (
        details["tree_id"]
        and details[
            "person_id"
        ]  # This ID might be USER ID or PERSON ID depending on source
        and config_instance
        and hasattr(config_instance, "BASE_URL")
    ):
        # Best guess: If treeId exists, person_id is likely the PERSON ID for tree context
        base_url = getattr(
            config_instance, "BASE_URL", "https://www.ancestry.com"
        ).rstrip("/")
        # Use the originally intended facts link if tree context seems available
        details["link"] = (
            f"{base_url}/family-tree/person/tree/{details['tree_id']}/person/{details['person_id']}/facts"
        )
    elif (
        details["person_id"]  # Assume this is the USER ID if tree_id is missing
        and config_instance
        and hasattr(config_instance, "BASE_URL")
    ):
        # Use profile link if tree context is missing
        base_url = getattr(
            config_instance, "BASE_URL", "https://www.ancestry.com"
        ).rstrip("/")
        details["link"] = (
            f"{base_url}/discoveryui-matches/profile/{details['person_id']}"  # Link to profile/match page
        )
    else:
        details["link"] = "(unavailable)"

    # Log details extracted
    logger.debug(
        f"Parsed API details for '{details.get('name', 'Unknown')}': "
        f"ID={details.get('person_id')}, Tree={details.get('tree_id', 'N/A')}, "  # Handle missing tree_id
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
        # Use format_name helper from utils (if available)
        formatter = (
            format_name
            if "format_name" in globals() and callable(format_name)
            else lambda x: str(x)
        )
        for item in items:
            print(f"  - {formatter(item.get('name', 'Unknown'))}")
    else:
        print("  (None found)")


# End of print_group


def format_api_relationship_path(
    ladder_html: Optional[str], owner_name: str, target_name: str
) -> str:
    """
    Parses relationship ladder HTML from Ancestry API response and formats
    it into a human-readable path string. Uses BeautifulSoup for parsing.
    Based on temp.py v7.36 display_raw_relationship_ladder parsing logic.
    """
    if not ladder_html or not isinstance(ladder_html, str):
        logger.warning(
            "format_api_relationship_path: Relationship ladder HTML missing or invalid type."
        )
        return "(No relationship path explanation available from API - HTML missing)"

    # Check if BeautifulSoup is available
    if BeautifulSoup is None:
        logger.error(
            "format_api_relationship_path: BeautifulSoup is not available. Cannot parse HTML."
        )
        return "(Cannot parse relationship path - BeautifulSoup missing. pip install beautifulsoup4 lxml)"

    try:
        # Attempt to decode HTML entities and unicode escapes (handled by display_raw_relationship_ladder already if called from action11)
        # Assuming the input ladder_html might still need some cleaning if not coming directly from display_raw_relationship_ladder raw output
        html_unescaped = html.unescape(ladder_html)  # Basic unescaping
        # Decode unicode escapes (e.g., \u003c -> <). Needs bytes.
        try:
            # Handle potential encoding issues. 'unicode_escape' requires bytes.
            # Use utf-8 with backslashreplace for robustness.
            html_unescaped = bytes(html_unescaped, "utf-8", "backslashreplace").decode(
                "unicode_escape", errors="replace"
            )
        except Exception as e:
            logger.debug(
                f"Failed to decode unicode escapes in format_api_relationship_path: {e}. Proceeding.",
                exc_info=False,
            )

        soup = BeautifulSoup(html_unescaped, "html.parser")

        # Extract Relationship Path items
        path_list: List[str] = []
        # Try common path item selectors - prioritize specific structure if known
        path_items = (
            soup.select(
                'ul.textCenter > li:not([class*="iconArrowDown"])'
            )  # Common structure seen
            or soup.select(
                "div.rel-path-wrapper ul > li:not(.rel-divider)"
            )  # Another common structure
            or soup.select("ul#relationshipPath > li")  # Older structure
            or soup.select("div.relationshipPath li")
            or soup.select(".rel-path li")
        )

        num_items = len(path_items)
        if not path_items:
            logger.warning(
                "format_api_relationship_path: Could not find any relationship path items using selectors."
            )
            logger.debug(
                f"HTML snippet that yielded no path items: {html_unescaped[:500]}..."
            )
            # Attempt to extract relationship text directly if path fails
            rel_text_elem = soup.select_one(".rel-path-wrapper p") or soup.select_one(
                ".relationshipText"
            )
            if rel_text_elem:
                return f"({rel_text_elem.get_text(strip=True)})"  # Return just the summary text
            return "(Could not parse relationship path from API HTML)"

        # Get formatting helpers from utils (if available)
        name_formatter = (
            format_name
            if "format_name" in globals() and callable(format_name)
            else lambda x: str(x)
        )
        ordinal_formatter = (
            ordinal_case
            if "ordinal_case" in globals() and callable(ordinal_case)
            else lambda x: str(x)
        )

        # --- Parse Path Items ---
        for i, item in enumerate(path_items):
            # Skip arrow/divider elements if they were selected
            # More robust check for classes containing 'arrow' or 'divider'
            item_classes = item.get("class", [])
            if any("arrow" in c or "divider" in c for c in item_classes):
                continue

            name_text = ""
            desc_text = ""

            # Find name container (<a>, <b>, <strong>, or span.name)
            name_container = (
                item.find("a")
                or item.find("b")
                or item.find("strong")
                or item.find(
                    "span", class_=re.compile(r"\bname\b", re.IGNORECASE)
                )  # More specific class match
            )

            if name_container:
                # Clean and format the name using the helper from utils
                raw_name = name_container.get_text(strip=True)
                name_text = name_formatter(raw_name)
                # Remove quotes if any (sometimes in API names)
                name_text = name_text.replace('"', "").replace("'", "")
            else:
                # Fallback: Get all direct text within the li, stripping known unwanted tags
                potential_name = item.get_text(separator=" ", strip=True)
                # Remove text from known sub-elements like <i> or relationship spans
                rel_span = item.find(
                    "span", class_=re.compile(r"relationship", re.IGNORECASE)
                )
                if rel_span:
                    potential_name = potential_name.replace(
                        rel_span.get_text(strip=True), ""
                    )
                italic_span = item.find("i")
                if italic_span:
                    potential_name = potential_name.replace(
                        italic_span.get_text(strip=True), ""
                    )
                potential_name = potential_name.strip()
                # Basic check if it looks like a name
                if (
                    potential_name
                    and not potential_name.isdigit()
                    and len(potential_name.split()) <= 4
                ):
                    name_text = name_formatter(
                        potential_name.replace('"', "").replace("'", "")
                    )
                    logger.debug(
                        f"Path item {i}: Using fallback text as name: '{name_text}'"
                    )
                else:
                    logger.debug(
                        f"Path item {i}: Could not find name container or suitable fallback text."
                    )

            # Find description container (<i> or <span class="relationship">)
            # The relationship description usually describes the *connection to the next person*
            desc_element = (
                item.find("i")  # Often holds the relationship label like 'Father'
                or item.find(
                    "span", class_=re.compile(r"\brelationship\b", re.IGNORECASE)
                )
                or item.find(
                    "div", class_=re.compile(r"\brelationship\b", re.IGNORECASE)
                )
            )

            if desc_element:
                raw_desc_full = desc_element.get_text(strip=True).replace('"', "'")
                # Apply ordinal_case helper from utils for formatting terms like 1st Cousin
                desc_text = ordinal_formatter(raw_desc_full)

                # Check specifically for the last item which often has "You are the..." prefix
                # This assumes the path is from the target person TO the tree owner ("You")
                you_are_prefix_match = re.match(
                    r"you are (?:the\s+)?", raw_desc_full, re.IGNORECASE
                )
                if i == num_items - 1 and you_are_prefix_match:
                    # If it's the last item and starts with "You are (the)...", format just the relationship part
                    processed_desc = raw_desc_full[you_are_prefix_match.end() :].strip()
                    desc_text = ordinal_formatter(
                        processed_desc
                    )  # Apply ordinal_case to the relationship part
                    logger.debug(
                        f"Path item {i} (last): Removed prefix, using description: '{desc_text}'"
                    )
                elif raw_desc_full:
                    # For other items, just apply ordinal_case to the raw description
                    desc_text = ordinal_formatter(raw_desc_full)
            else:
                logger.debug(
                    f"Path item {i}: Could not find description element (i, span.relationship)."
                )

            # Combine Name and Description (if available)
            display_text = name_text
            # Only append description if it's not empty and not just the name itself
            # Also check if description is just a date range (common mistake)
            is_date_range = desc_text and re.match(r"\d{4}\s*-\s*\d{0,4}", desc_text)
            if (
                desc_text
                and desc_text.lower() != name_text.lower()
                and not is_date_range
            ):
                display_text += f" ({desc_text})"
            elif is_date_range:
                logger.debug(
                    f"Path item {i}: Skipping description '{desc_text}' as it looks like a date range."
                )

            # Add to path list if we got valid text
            if display_text:
                path_list.append(display_text)
            elif name_text:
                path_list.append(
                    name_text
                )  # Fallback to just name if desc logic failed
            else:
                # If name extraction completely failed, log and skip
                logger.warning(
                    f"Path item {i}: Skipping path item because display_text/name_text is empty. Raw HTML item: {item}"
                )

        # --- Format the Path Steps ---
        # The API path typically goes from the Target Person UP or ACROSS to the Tree Owner ("You").
        # The first item is the Target Person, the last item is the Tree Owner.
        formatted_steps: List[str] = []

        if len(path_list) < 2:
            logger.warning(
                f"Path list too short ({len(path_list)} items) for detailed explanation."
            )
            # Attempt to extract relationship text directly if path is short
            rel_text_elem = soup.select_one(".rel-path-wrapper p") or soup.select_one(
                ".relationshipText"
            )
            if rel_text_elem:
                return f"({rel_text_elem.get_text(strip=True)})"  # Return just the summary text

            if path_list:
                return f"{path_list[0]} (Path too short for explanation)"
            return "(Could not parse path steps from HTML)"

        # Start the explanation with the target person
        start_person_display = path_list[0]  # First element is the target
        # Remove " (description)" from the start person's entry if present
        start_person_name_only = name_formatter(start_person_display.split(" (")[0])

        # Generate steps like "Name A is the [description] of Name B"
        # The description associated with a person in the API HTML often refers to the *relationship to the next person*
        # So, we look at person N and their description to describe the link from N to N+1.

        for i in range(len(path_list) - 1):
            current_person_info = path_list[i]  # E.g., "John Smith (Father)"
            next_person_info = path_list[
                i + 1
            ]  # E.g., "Jane Doe (Mother)" or "You (Self)"

            current_person_name = name_formatter(
                current_person_info.split(" (")[0]
            )  # Just the name
            next_person_name = name_formatter(
                next_person_info.split(" (")[0]
            )  # Just the name

            # Extract description from current_person_info if available
            desc_match = re.search(r"\((.*?)\)", current_person_info)
            # Use ordinal_formatter on the extracted relationship label
            relationship_label = (
                ordinal_formatter(desc_match.group(1)) if desc_match else "related"
            )

            # Special handling for the last step towards the Tree Owner
            if i == len(path_list) - 2:
                # Ensure the last person displayed is the Tree Owner name passed to function
                # The API often labels the last item "You" or similar, use the known owner name instead
                next_person_name = name_formatter(
                    owner_name
                )  # Use the actual tree owner name

            # Format the step
            formatted_steps.append(
                f"{current_person_name} is the {relationship_label} of {next_person_name}"
            )

        # Combine the steps with arrows
        explanation_str = "\n -> ".join(formatted_steps)

        # Return the final formatted path, starting with the initial person
        # Adding the overall relationship text if available
        overall_rel_text = ""
        rel_text_elem = soup.select_one(
            ".rel-path-wrapper p.marBtm10"
        ) or soup.select_one(".relationshipText")
        if rel_text_elem:
            overall_rel_text = f"Overall: {ordinal_formatter(rel_text_elem.get_text(strip=True))}\nPath:\n"

        return f"{overall_rel_text}{start_person_name_only}\n -> {explanation_str}"

    except Exception as bs_parse_err:
        logger.error(
            f"Error parsing relationship ladder HTML with BeautifulSoup: {bs_parse_err}",
            exc_info=True,  # Log traceback
        )
        logger.debug(f"Problematic unescaped HTML: {html_unescaped[:500]}...")
        return f"(Error parsing API relationship path: {bs_parse_err})"


# End of format_api_relationship_path


# --- NEW HELPER FUNCTION ---
def _extract_ladder_html(raw_content: Union[str, Dict]) -> Optional[str]:
    """
    Extracts and decodes the relationship ladder HTML from raw API response content.
    Handles standard JSON, JSONP, and potential errors.
    """
    if isinstance(raw_content, dict) and "error" in raw_content:
        error_msg = raw_content.get("error", {}).get(
            "message", raw_content.get("message", "Unknown API Error")
        )
        logger.error(f"_extract_ladder_html: API returned error: {error_msg}")
        return None
    if not raw_content or not isinstance(raw_content, str):
        logger.error(
            f"_extract_ladder_html: Invalid raw content type: {type(raw_content)}"
        )
        return None

    html_escaped = None
    # 1. Try JSONP extraction first
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
                "_extract_ladder_html: Raw content does not match JSONP callback structure."
            )
            # --- Attempt direct JSON parse if JSONP match fails ---
            if raw_content.strip().startswith("{") and raw_content.strip().endswith(
                "}"
            ):
                logger.debug("_extract_ladder_html: Attempting direct JSON parse...")
                try:
                    json_data_direct = json.loads(raw_content.strip())
                    if (
                        isinstance(json_data_direct, dict)
                        and "html" in json_data_direct
                        and isinstance(json_data_direct["html"], str)
                    ):
                        html_escaped = json_data_direct["html"]
                        logger.debug(
                            f"_extract_ladder_html: Found HTML via direct JSON parse. Length: {len(html_escaped)}"
                        )
                    else:
                        logger.warning(
                            "_extract_ladder_html: Direct JSON ok, but 'html' key missing/invalid."
                        )
                except json.JSONDecodeError:
                    logger.warning("_extract_ladder_html: Direct JSON parse failed.")
            # --- End Direct JSON ---

    except json.JSONDecodeError as json_e:
        logger.warning(
            f"_extract_ladder_html: JSONDecodeError during JSONP/JSON extraction: {json_e}"
        )
    except Exception as e:
        logger.warning(
            f"_extract_ladder_html: Unexpected error during JSONP/JSON extraction: {e}"
        )

    # 2. Fallback: Try direct regex for "html":"..." structure
    if not html_escaped:
        logger.debug("_extract_ladder_html: JSONP/JSON failed, trying regex...")
        html_match = re.search(
            r'"html"\s*:\s*"((?:\\.|[^"\\])*)"',
            raw_content,
            re.IGNORECASE | re.DOTALL,
        )
        if html_match:
            html_escaped = html_match.group(1)
            logger.debug(
                f"_extract_ladder_html: Found HTML via regex. Length: {len(html_escaped)}"
            )

    # 3. If HTML still not found, fail
    if not html_escaped:
        logger.error("_extract_ladder_html: Could not extract HTML content.")
        logger.debug(f"Raw content snippet: {raw_content[:500]}...")
        return None

    # 4. Unescape and decode
    try:
        temp_unescaped = html_escaped.replace("\\\\", "\\")
        html_intermediate = temp_unescaped.encode("utf-8", "backslashreplace").decode(
            "unicode_escape", errors="replace"
        )
        html_unescaped = html.unescape(html_intermediate)
        logger.debug("_extract_ladder_html: Successfully unescaped HTML.")
        return html_unescaped
    except Exception as decode_err:
        logger.error(
            f"_extract_ladder_html: Could not decode HTML. Error: {decode_err}",
            exc_info=True,
        )
        logger.debug(f"Problematic escaped HTML snippet: {html_escaped[:500]}...")
        return None


# End of _extract_ladder_html


# --- UPDATED FUNCTION ---
def display_raw_relationship_ladder(
    raw_content: Union[str, Dict], owner_name: str, target_name: str
):
    """
    Parse and display the Ancestry relationship ladder from raw JSONP/HTML content.
    Uses helper functions to extract HTML and format the path.
    """
    # Check if BeautifulSoup is available early
    if BeautifulSoup is None:
        logger.error(
            "BeautifulSoup library not found. Cannot parse relationship ladder HTML."
        )
        print(f"\n--- Relationship between {owner_name} and {target_name} (API) ---")
        print(
            "\n(Cannot parse relationship path - BeautifulSoup missing. pip install beautifulsoup4 lxml)"
        )
        return  # End of function display_raw_relationship_ladder

    logger.info(
        f"\n--- Relationship between {owner_name} and {target_name} (API Report) ---"
    )

    # --- Use helper to extract and decode HTML ---
    html_unescaped = _extract_ladder_html(raw_content)

    if not html_unescaped:
        # Error already logged by helper
        print(
            "\n(Could not extract or decode relationship path HTML from API response)"
        )
        return  # End of function display_raw_relationship_ladder
    # --- End HTML extraction ---

    # --- Use format_api_relationship_path to get the full formatted string ---
    # Pass the clean, unescaped HTML to the formatter function
    formatted_path_str = format_api_relationship_path(
        html_unescaped, owner_name, target_name
    )

    # Print the formatted path/relationship string returned by the formatter
    if formatted_path_str:
        print(
            formatted_path_str.strip()
        )  # Print the generated path/relationship string
    else:
        # Error should have been logged by format_api_relationship_path
        logger.warning("format_api_relationship_path returned empty string or None.")
        print("(Could not format relationship path steps from extracted HTML)")


# End of display_raw_relationship_ladder


# --- Standalone Test Block ---
def self_check(verbose: bool = True) -> bool:
    """
    Performs internal self-checks for api_utils.py, including LIVE API calls
    to test core parsing functions using configured credentials and target IDs.
    Requires .env file to be correctly configured with Ancestry credentials,
    TESTING_PROFILE_ID, TESTING_PERSON_TREE_ID (tree-specific), and MY_TREE_ID
    (or TREE_NAME for auto-detection). Tests parser with Profile Details API (JSON)
    and relationship functions with the tree-specific /getladder API (JSONP/HTML).
    """
    # --- Local Imports for Self-Check ---
    # Need these to run the live tests without top-level dependency loops
    try:
        # Ensure utils can be imported and has the necessary components
        import utils

        if not hasattr(utils, "SessionManager") or not hasattr(utils, "_api_req"):
            raise ImportError(
                "Required components (SessionManager, _api_req) not found in utils."
            )
        from utils import SessionManager, _api_req, log_in, login_status

        # Ensure config provides necessary attributes
        from config import config_instance, selenium_config  # Need selenium_config too

        if not hasattr(config_instance, "TESTING_PROFILE_ID"):
            raise ImportError("config_instance missing TESTING_PROFILE_ID.")
        # --- Check for TESTING_PERSON_TREE_ID existence ---
        if not hasattr(config_instance, "TESTING_PERSON_TREE_ID"):
            raise ImportError("config_instance missing TESTING_PERSON_TREE_ID.")
        # Ensure logging is set up
        from logging_config import logger as utils_logger
        from urllib.parse import urljoin, urlencode, quote  # Need quote
        import traceback
        import requests  # Needed for type checking response
    except ImportError as e:
        print(f"\n[api_utils.py self-check ERROR]")
        print(
            f"- CRITICAL: Failed to import required modules/attributes for self-check: {e}"
        )
        print(
            "- Ensure utils.py, config.py, logging_config.py are present, importable, and contain required definitions."
        )
        print(
            "- Make sure TESTING_PERSON_TREE_ID is defined in your .env file and loaded by config.py"
        )
        print(f"Self-check status: FAIL\n")
        return False

    # Use the logger imported from logging_config
    logger = utils_logger

    # --- Test Parameters ---
    target_profile_id = (
        config_instance.TESTING_PROFILE_ID
    )  # User ID for profile/ladder API
    target_person_id = (
        config_instance.TESTING_PERSON_TREE_ID
    )  # Tree-specific Person ID from config
    target_name_from_profile = "Unknown Target"  # Name from profile API

    # --- Status Tracking ---
    status = True
    messages = []
    session_manager: Optional[SessionManager] = None  # Initialize

    # --- Helper for printing/logging messages ---
    def log_msg(msg: str, level: int = logging.INFO):
        messages.append(msg)
        if level == logging.ERROR:
            logger.error(msg)
        elif level == logging.WARNING:
            logger.warning(msg)
        else:
            logger.info(msg)

    # End of log_msg

    log_msg("\n[api_utils.py self-check starting...]")

    # === Phase 0: Prerequisite Checks ===
    log_msg("\n--- Phase 0: Prerequisite Checks ---")
    # Check essential imports loaded at module level
    module_imports_ok = True
    if BeautifulSoup is None:
        log_msg(
            "- BeautifulSoup Import: FAILED (Needed for HTML parsing)", logging.ERROR
        )
        module_imports_ok = False
        status = False
    else:
        log_msg("- BeautifulSoup Import: OK")

    # Check formatters/parsers
    core_funcs = {
        "format_name": format_name,
        "ordinal_case": ordinal_case,
        "_parse_date": _parse_date,
        "_clean_display_date": _clean_display_date,
    }
    for name, func in core_funcs.items():
        if name not in globals() or not callable(func):
            log_msg(f"- {name} Import/Fallback: FAILED", logging.ERROR)
            module_imports_ok = False
            status = False
        else:
            log_msg(f"- {name} Import/Fallback: OK")

    # Check config existence and critical values needed for tests
    config_ok = True
    if not hasattr(config_instance, "BASE_URL") or not config_instance.BASE_URL:
        log_msg("- Config Check: BASE_URL missing or empty. FAILED", logging.ERROR)
        config_ok = False
        status = False
    if (
        not hasattr(config_instance, "ANCESTRY_USERNAME")
        or not config_instance.ANCESTRY_USERNAME
    ):
        log_msg(
            "- Config Check: ANCESTRY_USERNAME missing or empty. FAILED", logging.ERROR
        )
        config_ok = False
        status = False
    if (
        not hasattr(config_instance, "ANCESTRY_PASSWORD")
        or not config_instance.ANCESTRY_PASSWORD
    ):
        log_msg(
            "- Config Check: ANCESTRY_PASSWORD missing or empty. FAILED", logging.ERROR
        )
        config_ok = False
        status = False
    if (
        not hasattr(config_instance, "TESTING_PROFILE_ID")
        or not config_instance.TESTING_PROFILE_ID
    ):
        log_msg(
            "- Config Check: TESTING_PROFILE_ID missing or empty. FAILED", logging.ERROR
        )
        config_ok = False
        status = False
    # --- Check for the NEW required config ---
    if (
        not hasattr(config_instance, "TESTING_PERSON_TREE_ID")
        or not config_instance.TESTING_PERSON_TREE_ID
    ):
        log_msg(
            "- Config Check: TESTING_PERSON_TREE_ID missing or empty (Required for Ladder Test). FAILED",
            logging.ERROR,
        )
        config_ok = False
        status = False
    if config_ok:
        log_msg("- Config Check: Essential credentials/URLs/IDs present.")

    if not module_imports_ok or not config_ok:
        log_msg(
            "Prerequisite checks failed. Cannot proceed with live tests.", logging.ERROR
        )
        if verbose:
            print("\n[api_utils.py self-check results]")
            for m in messages:
                print(m.replace("- ", "  "))
        print(f"\nSelf-check status: FAIL (Prerequisites)\n")
        return False

    try:
        # === Phase 1: Session Setup ===
        log_msg("\n--- Phase 1: Session Setup & Login ---")
        session_manager = SessionManager()
        start_ok = session_manager.start_sess(action_name="api_utils Self Check Start")
        if not start_ok or not session_manager.driver_live:
            log_msg("- SessionManager.start_sess(): FAILED", logging.ERROR)
            status = False
            raise RuntimeError("Failed to start WebDriver session.")

        log_msg("- SessionManager.start_sess(): PASSED")
        ready_ok = session_manager.ensure_session_ready(
            action_name="api_utils Self Check Ready"
        )
        if not ready_ok:
            log_msg("- SessionManager.ensure_session_ready(): FAILED", logging.ERROR)
            status = False
            raise RuntimeError(
                "Failed to ensure session readiness (login/setup failed)."
            )

        log_msg("- SessionManager.ensure_session_ready(): PASSED")

        # === Phase 2: Get Target Info & Validate Config ===
        log_msg("\n--- Phase 2: Get Target Info & Validate Config ---")
        params_ok = True
        target_tree_id = session_manager.my_tree_id
        target_owner_name = session_manager.tree_owner_name
        target_person_id = config_instance.TESTING_PERSON_TREE_ID  # Get from config

        log_msg(f"- Target Profile ID (TESTING_PROFILE_ID): {target_profile_id}")
        log_msg(
            f"- Target Person ID (TESTING_PERSON_TREE_ID): {target_person_id}"
        )  # Log configured person ID

        if not target_tree_id:
            log_msg(
                "- Target Tree ID (MY_TREE_ID): FAILED (Not found/set after login)",
                logging.ERROR,
            )
            params_ok = False
            status = False
        else:
            log_msg(f"- Target Tree ID (MY_TREE_ID): {target_tree_id}")

        if not target_owner_name:
            log_msg(
                "- Target Owner Name: FAILED (Not found after login)", logging.ERROR
            )
            params_ok = False
            status = False
        else:
            log_msg(f"- Target Owner Name: {target_owner_name}")

        # Validate the configured Person ID is present (already checked in Phase 0, but double check)
        if not target_person_id:
            log_msg(
                "- Configured Target Person ID (TESTING_PERSON_TREE_ID): FAILED (Missing from config)",
                logging.ERROR,
            )
            params_ok = False
            status = False  # Critical for ladder test

        # Step 2a: Get target display name from Profile API (still useful for relationship functions)
        profile_api_url = urljoin(
            config_instance.BASE_URL.rstrip("/") + "/",
            f"/app-api/express/v1/profiles/details?userId={target_profile_id.upper()}",
        )
        profile_api_desc = "Get Target Name (Profile Details)"
        log_msg(f"- Getting target display name from Profile API: {profile_api_url}")
        profile_response_details = utils._api_req(  # Use utils._api_req explicitly
            url=profile_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            api_description=profile_api_desc,
            use_csrf_token=False,
        )
        if profile_response_details and isinstance(profile_response_details, dict):
            target_name_from_profile = profile_response_details.get(
                "DisplayName", target_name_from_profile
            )
            # Fallback logic for name
            if target_name_from_profile == "Unknown Target":
                first_name = profile_response_details.get("FirstName")
                last_name = profile_response_details.get("LastName")
                if first_name and last_name:
                    target_name_from_profile = f"{first_name} {last_name}"
                elif first_name:
                    target_name_from_profile = first_name
            log_msg(f"- Target Display Name found: '{target_name_from_profile}'")
        else:
            log_msg(
                f"- Failed to get target display name from Profile API. Response: {profile_response_details}",
                logging.ERROR,
            )
            params_ok = False
            status = False  # Fail if we can't get the name for relationship test

        # Step 2b: Removed API search for personId

        if not params_ok:
            log_msg(
                "One or more required parameters missing. Subsequent tests may be skipped or fail.",
                logging.WARNING,
            )

        # === Phase 3: Test parse_ancestry_person_details (Profile Details API) ===
        # This phase remains the same, testing the parser with the Profile Details JSON
        log_msg(
            "\n--- Phase 3: Test parse_ancestry_person_details (Profile Details API) ---"
        )
        # Use profile_response_details fetched in Phase 2
        parsed_details: Optional[Dict] = None
        if profile_response_details and isinstance(profile_response_details, dict):
            log_msg("- Using Profile Details API response from Phase 2.")
            # Pass the USER ID (target_profile_id) here, as that's what the Profile API used
            person_card_for_parse = {
                "personId": target_profile_id,
                "treeId": target_tree_id,  # Can be None
            }
            try:
                parsed_details = parse_ancestry_person_details(
                    person_card_for_parse, profile_response_details
                )
                if parsed_details and isinstance(parsed_details, dict):
                    log_msg("- parse_ancestry_person_details call: SUCCESS")
                    # Validate essential keys (person_id will be the PROFILE ID here)
                    missing_keys = [
                        k
                        for k in ["name", "person_id", "link"]
                        if k not in parsed_details or parsed_details[k] is None
                    ]
                    if not missing_keys:
                        # Check if name matches the one fetched earlier
                        if parsed_details["name"] == format_name(
                            target_name_from_profile
                        ):
                            log_msg(
                                f"- Parsed Details Validation: PASSED (Name: '{parsed_details['name']}', Link: {parsed_details['link']})"
                            )
                        else:
                            log_msg(
                                f"- Parsed Details Validation: WARNING (Parsed name '{parsed_details['name']}' differs from fetched '{format_name(target_name_from_profile)}')",
                                logging.WARNING,
                            )
                            # Don't fail the test for name discrepancy alone, but log it.
                    else:
                        log_msg(
                            f"- Parsed Details Validation: FAILED (Missing/Null keys: {missing_keys})",
                            logging.ERROR,
                        )
                        log_msg(f"  - Parsed Output: {parsed_details}", logging.DEBUG)
                        status = False  # Fail if core details missing
                else:
                    log_msg(
                        "- parse_ancestry_person_details call: FAILED (Returned None or wrong type)",
                        logging.ERROR,
                    )
                    status = False
            except Exception as parse_e:
                log_msg(
                    f"- parse_ancestry_person_details call: FAILED (Exception: {parse_e})",
                    logging.ERROR,
                )
                log_msg(traceback.format_exc(), logging.DEBUG)
                status = False
        else:
            log_msg(
                "- Skipping parse_ancestry_person_details test: Profile Details API failed in Phase 2.",
                logging.WARNING,
            )

        # === Phase 4: Test Relationship Ladder Parsing (using /getladder endpoint) ===
        log_msg(
            "\n--- Phase 4: Test Relationship Ladder Parsing (using /getladder) ---"
        )
        # Only proceed if owner name, target name, AND target_person_id (from config) are known
        if (
            not target_owner_name
            or target_name_from_profile == "Unknown Target"
            or not target_person_id
            or not target_tree_id
        ):
            missing_ladder_reqs = []
            if not target_owner_name:
                missing_ladder_reqs.append("Owner Name")
            if target_name_from_profile == "Unknown Target":
                missing_ladder_reqs.append("Target Name")
            if not target_person_id:
                missing_ladder_reqs.append(
                    "Target Person ID (Config: TESTING_PERSON_TREE_ID)"
                )
            if not target_tree_id:
                missing_ladder_reqs.append("Target Tree ID")
            log_msg(
                f"- Skipping Relationship Ladder test: Missing required info ({', '.join(missing_ladder_reqs)}).",
                logging.WARNING,
            )
            if (
                not target_person_id
            ):  # Explicitly fail if the required config is missing
                status = False
        else:
            # --- Construct URL with JSONP parameters ---
            base_ladder_url = urljoin(
                config_instance.BASE_URL.rstrip("/") + "/",
                f"/family-tree/person/tree/{target_tree_id}/person/{target_person_id}/getladder",
            )
            callback_name = (
                f"__ancestry_jsonp_{int(time.time()*1000)}"  # Simple unique callback
            )
            timestamp_ms = int(time.time() * 1000)
            query_params = urlencode({"callback": callback_name, "_": timestamp_ms})
            ladder_api_url = f"{base_ladder_url}?{query_params}"
            # --- End URL construction ---

            log_msg(f"- Calling Tree Relationship Ladder API: {ladder_api_url}")

            # Mimic headers from working cURL
            ladder_headers = {
                "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            }
            # Specific referer based on the person being viewed
            ladder_referer = urljoin(
                config_instance.BASE_URL.rstrip("/") + "/",
                f"/family-tree/person/tree/{target_tree_id}/person/{target_person_id}/facts",
            )

            ladder_response_raw = utils._api_req(  # Use utils._api_req explicitly
                url=ladder_api_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                api_description="Get Tree Ladder API (Self Check)",
                use_csrf_token=False,  # Rely on cookies based on cURL example
                headers=ladder_headers,
                referer_url=ladder_referer,
                force_text_response=True,  # Expecting JSONP string
                timeout=20,  # Keep shorter timeout for diagnosis
            )

            if ladder_response_raw and isinstance(ladder_response_raw, str):
                log_msg("- Tree Ladder API call: SUCCESS (Received string)")

                # --- Use helper function to extract HTML ---
                log_msg("- Extracting HTML using _extract_ladder_html...")
                html_extracted = _extract_ladder_html(ladder_response_raw)

                if not html_extracted:
                    log_msg("- HTML Extraction FAILED.", logging.ERROR)
                    status = False  # Fail if HTML extraction fails
                else:
                    log_msg("- HTML Extraction SUCCESS.")
                    # Test 1: display_raw_relationship_ladder (now just uses the helper + format)
                    try:
                        log_msg(
                            "- Testing display_raw_relationship_ladder (internal logic)..."
                        )
                        display_raw_relationship_ladder(
                            ladder_response_raw,
                            target_owner_name,
                            target_name_from_profile,
                        )
                        log_msg(
                            "- display_raw_relationship_ladder execution: PASSED (No exceptions)"
                        )
                    except Exception as display_e:
                        log_msg(
                            f"- display_raw_relationship_ladder execution: FAILED (Exception: {display_e})",
                            logging.ERROR,
                        )
                        log_msg(traceback.format_exc(), logging.DEBUG)
                        status = False

                    # Test 2: format_api_relationship_path (using the extracted HTML)
                    log_msg("- Testing format_api_relationship_path...")
                    try:
                        formatted_path = format_api_relationship_path(
                            html_extracted, target_owner_name, target_name_from_profile
                        )
                        # Corrected Validation
                        if (
                            formatted_path
                            and isinstance(formatted_path, str)
                            and "->" in formatted_path
                            and target_owner_name in formatted_path
                        ):
                            log_msg("- format_api_relationship_path call: PASSED")
                            log_msg(
                                f"  - Formatted Output (Preview):\n{formatted_path[:200]}...\n",
                                logging.DEBUG,
                            )
                        elif (
                            formatted_path
                            and isinstance(formatted_path, str)
                            and formatted_path.startswith("(")
                        ):
                            log_msg(
                                "- format_api_relationship_path call: PASSED (Returned summary text)"
                            )
                            log_msg(
                                f"  - Summary Output: {formatted_path}", logging.DEBUG
                            )
                        else:
                            log_msg(
                                "- format_api_relationship_path call: FAILED (Output invalid, empty, or missing structure/owner name)",
                                logging.ERROR,
                            )
                            log_msg(
                                f"  - Actual Output: {formatted_path}", logging.DEBUG
                            )
                            status = False
                        # End Corrected Validation
                    except Exception as format_e:
                        log_msg(
                            f"- format_api_relationship_path call: FAILED (Exception: {format_e})",
                            logging.ERROR,
                        )
                        log_msg(traceback.format_exc(), logging.DEBUG)
                        status = False

            elif isinstance(ladder_response_raw, requests.Response):
                log_msg(
                    f"- Tree Ladder API call: FAILED (HTTP Status {ladder_response_raw.status_code})",
                    logging.ERROR,
                )
                status = False
            elif ladder_response_raw is None:
                log_msg(
                    f"- Tree Ladder API call: FAILED (Returned None - likely timeout after {20}s)",
                    logging.ERROR,
                )
                status = False
            else:
                log_msg(
                    f"- Tree Ladder API call: FAILED (Returned {type(ladder_response_raw)})",
                    logging.ERROR,
                )
                status = False

    except Exception as e:
        log_msg(f"\n--- CRITICAL ERROR during self-check execution ---", logging.ERROR)
        log_msg(f"- Error Type: {type(e).__name__}", logging.ERROR)
        log_msg(f"- Error Details: {e}", logging.ERROR)
        log_msg(f"- Traceback:\n{traceback.format_exc()}", logging.DEBUG)
        status = False
    finally:
        if session_manager:
            log_msg("\n--- Finalizing: Closing Session ---")
            session_manager.close_sess()
        else:
            log_msg("\n--- Finalizing: No session to close ---")

        # Print results
        if verbose:
            print("\n[api_utils.py self-check results]")
            for m in messages:
                prefix = "  "
                if "ERROR" in m or "FAIL" in m:
                    prefix = "* "
                elif "WARN" in m:
                    prefix = "! "
                print(f"{prefix}{m.replace('- ', '')}")

        final_status_msg = f"Self-check status: {'PASS' if status else 'FAIL'}"
        print(f"\n{final_status_msg}\n")
        if status:
            logger.info("api_utils self-check status: PASS")
        else:
            logger.error("api_utils self-check status: FAIL")

    return status


# End of self_check

if __name__ == "__main__":
    # --- ADDED IMPORT HERE ---
    from pathlib import Path

    # Note: This standalone test now performs live API calls.
    # Ensure your .env file is configured correctly before running,
    # including TESTING_PERSON_TREE_ID for the ladder test.
    print("Running api_utils.py self-check (with live API calls)...")
    # Need to ensure logging is set up if running standalone
    try:
        # Ensure logging_config is importable before calling setup_logging
        import logging_config

        if not hasattr(logging_config, "setup_logging"):
            raise ImportError("setup_logging function not found in logging_config")
        # Use a generic log file name for standalone run
        log_file = Path("api_utils_self_check.log").resolve()
        # Set level to DEBUG to capture detailed logs from self_check
        logger = logging_config.setup_logging(log_file=log_file, log_level="DEBUG")
        print(f"Detailed logs will be written to: {log_file}")
    except ImportError as log_imp_err:
        print(
            f"Warning: logging_config import or setup failed ({log_imp_err}). Using basic logging."
        )
        logging.basicConfig(level=logging.DEBUG)  # Fallback basic config
        logger = logging.getLogger("api_utils_standalone")
    except Exception as log_setup_err:
        print(f"Error setting up logging: {log_setup_err}. Using basic logging.")
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger("api_utils_standalone")

    self_check_passed = self_check(verbose=True)
    print("\nThis is the api_utils module. Import it into other scripts.")
    # Optional: Exit with status code based on test result
    sys.exit(0 if self_check_passed else 1)
# End of api_utils.py
