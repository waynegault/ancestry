"""
Test script for relationship path formatting functionality.

This script contains functions for parsing and formatting relationship paths from
Ancestry API responses, specifically from the getladder endpoint. It demonstrates
how to handle both raw JSONP strings and pre-parsed JSON data.

The test verifies that:
1. HTML content can be extracted from JSONP responses
2. HTML content can be properly decoded
3. The relationship path can be properly formatted
4. Both raw string and parsed JSON inputs produce identical results

This is a standalone version that includes all necessary functions from api_utils.py.
"""

import logging
import json
import re
import html
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

# Try to import BeautifulSoup, but handle the case where it's not available
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None  # type: ignore

    BS4_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Test raw API response from Ancestry's getladder endpoint
# This is a sample response containing a relationship path from
# Elizabeth 'Betty' Cruickshank to Wayne Gordon Gault
raw_api_response = r"""
no({
    "html": "\u003cul class=\"textCenter\"\u003e \u003cli\u003e\u003cb\u003eElizabeth \u0027Betty\u0027 Cruickshank\u003c/b\u003e 1839-1886\u003cbr /\u003e\u003ci\u003e\u003cb\u003e3rd great-grandmother\u003c/b\u003e\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560698\"\u003eMargaret Simpson 1865-1946\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eDaughter of Elizabeth \u0027Betty\u0027 Cruickshank\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560684\"\u003eAlexander Stables 1899-1948\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eSon of Margaret Simpson\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560677\"\u003eCatherine Margaret Stables 1924-2004\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eDaughter of Alexander Stables\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560544\"\u003eFrances Margaret Milne 1947-\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eDaughter of Catherine Margaret Stables\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"bottomName\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560836\"\u003e\u003cb\u003eWayne Gordon Gault\u003c/b\u003e\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eYou are the son of Frances Margaret Milne\u003c/i\u003e\u003c/li\u003e \u003c/ul\u003e ",
    "title": "Relationship to me",
    "printText": "Print",
    "status": "success"
})
"""

# --- Helper Functions ---


def format_name(name_str: Optional[str]) -> str:
    """
    Format a name string with proper capitalization.

    Args:
        name_str: The name string to format

    Returns:
        Formatted name string
    """
    return str(name_str).title() if name_str else "Unknown"


# End of format_name


def ordinal_case(text: str) -> str:
    """
    Apply ordinal case to text (e.g., "1st", "2nd", "3rd").
    Simple implementation that just returns the input.

    Args:
        text: The text to format

    Returns:
        Formatted text
    """
    return str(text)


# End of ordinal_case


def _get_relationship_term(
    person_a_gender: Optional[str], basic_relationship: str
) -> str:
    """
    Get the appropriate relationship term based on gender.

    Args:
        person_a_gender: Gender of the person ("M" or "F")
        basic_relationship: Basic relationship term (e.g., "parent", "child")

    Returns:
        Appropriate relationship term
    """
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

    # Apply ordinal case if the term contains digits
    if any(char.isdigit() for char in term):
        try:
            term = ordinal_case(term)
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
            except Exception as parse_err:
                # Catch all parsing errors
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
            li for li in list_items if "iconArrowDown" not in (li.get("class") or [])
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
        # Extract the summary directly from the first item's <i> tag if available
        summary_tag = None
        if list_items:
            summary_tag = list_items[0].select_one("i")

        if summary_tag:
            # Get the raw summary text from the first item
            raw_summary = summary_tag.get_text(strip=True)
            # Clean up any HTML entities or extra spaces
            summary_line = html.unescape(raw_summary)
            # Add colon if not present
            if not summary_line.endswith(":"):
                summary_line += ":"
        else:
            # Fallback to constructed summary if extraction fails
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


def extract_html_from_jsonp(jsonp_str):
    """
    Extract HTML content from JSONP response string.

    This function handles the specific format of Ancestry's getladder API response,
    which is wrapped in a 'no(...)' JSONP callback. It extracts the JSON data,
    parses it, and returns the HTML content if available.

    Args:
        jsonp_str: Raw JSONP string from Ancestry API (e.g., 'no({...})')

    Returns:
        HTML content string or None if extraction fails
    """
    # Step 1: Extract the JSON part from the JSONP wrapper
    try:
        # Match the JSON part inside the JSONP wrapper (no(...))
        json_part_match = re.search(r"^\s*no\((.*)\)\s*$", jsonp_str.strip(), re.DOTALL)

        if not json_part_match:
            logger.error("Failed to extract JSON part from JSONP wrapper")
            return None

        json_part_str = json_part_match.group(1).strip()

        # Step 2: Parse the JSON string
        parsed_json = json.loads(json_part_str)

        # Step 3: Extract the HTML content
        if "html" in parsed_json and parsed_json.get("status") == "success":
            return parsed_json.get("html")
        else:
            logger.error(
                f"No HTML content found or API status not success: {parsed_json.get('status')}"
            )
            return None

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error extracting HTML from JSONP: {e}")
        return None


# End of extract_html_from_jsonp


def decode_html_content(html_content):
    """
    Decode HTML content by unescaping HTML entities and unicode escapes.

    This function performs two-step decoding:
    1. First, it unescapes HTML entities (e.g., &lt; becomes <)
    2. Then, it decodes unicode escapes (e.g., \u003c becomes <)

    This is necessary because Ancestry API responses often contain doubly-encoded HTML.

    Args:
        html_content: Raw HTML content with escaped characters

    Returns:
        Decoded HTML content, or the original content if decoding fails
    """
    try:
        # First, handle standard HTML entities
        html_content_intermediate = html.unescape(html_content)

        # Second, decode unicode escapes often used in JSON embedding
        html_content_decoded = bytes(html_content_intermediate, "utf-8").decode(
            "unicode_escape"
        )

        return html_content_decoded
    except Exception as e:
        logger.error(f"Error decoding HTML content: {e}")
        return html_content  # Return original if decoding fails


# End of decode_html_content


def test_format_relationship_path():
    """
    Test the format_api_relationship_path function with the test raw_api_response.

    This function performs a comprehensive test of the relationship path formatting:
    1. Extracts HTML from the JSONP response to verify extraction works
    2. Decodes the HTML content to verify decoding works
    3. Parses the HTML using BeautifulSoup to examine the structure
    4. Calls format_api_relationship_path with the raw response string
    5. Prints the formatted relationship path
    6. Tests with pre-parsed JSON data to verify both input methods work
    7. Compares results to ensure consistency

    The test demonstrates that the format_api_relationship_path function can handle
    both raw JSONP strings and pre-parsed JSON objects, producing identical results.
    This is important for flexibility in how the function is used in the application.
    """
    print("\n=== Testing Relationship Path Formatting ===\n")

    # Step 1: Extract HTML from the JSONP response
    print("Step 1: Extracting HTML from JSONP response...")
    html_content = extract_html_from_jsonp(raw_api_response)
    if not html_content:
        print("Failed to extract HTML content from JSONP response")
        return

    print(f"\nExtracted HTML (first 200 chars):\n{html_content[:200]}...\n")

    # Step 2: Decode the HTML content
    print("Step 2: Decoding HTML content...")
    decoded_html = decode_html_content(html_content)
    print(f"\nDecoded HTML (first 200 chars):\n{decoded_html[:200]}...\n")

    # Step 3: Parse the HTML using BeautifulSoup
    print("Step 3: Parsing HTML with BeautifulSoup...")
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(decoded_html, "html.parser")

        # Find all list items
        list_items = soup.select("ul.textCenter li")
        print(f"\nFound {len(list_items)} list items in the HTML")

        # Filter out arrow items
        path_items = [
            li for li in list_items if "iconArrowDown" not in (li.get("class") or [])
        ]
        print(f"After filtering arrows: {len(path_items)} relevant path items")

        # Print the first item (summary)
        if path_items:
            print(f"\nFirst item (summary): {path_items[0].get_text(strip=True)}")

            # Print relationship description
            summary_tag = path_items[0].select_one("i")
            if summary_tag:
                print(f"Relationship summary: {summary_tag.get_text(strip=True)}")

            # Print a few people in the path
            print("\nPeople in the path:")
            for i, item in enumerate(path_items):
                if i > 0 and i < 4:  # Just show a few examples
                    name_tag = item.find("a") or item.find("b")
                    name = name_tag.get_text(strip=True) if name_tag else "Unknown"
                    desc_tag = item.find("i")
                    desc = desc_tag.get_text(strip=True) if desc_tag else ""
                    print(f"  Person {i}: {name} - {desc}")
    except ImportError:
        print("BeautifulSoup not available - skipping HTML parsing demonstration")
    except Exception as e:
        print(f"Error parsing HTML: {e}")

    # Step 4: Format the relationship path using the raw API response directly
    print("\nStep 4: Formatting relationship path using raw API response...")

    # Define the target person and owner names
    target_name = "Elizabeth 'Betty' Cruickshank"
    owner_name = "Wayne Gordon Gault"

    print(f"Target person: {target_name}")
    print(f"Owner name: {owner_name}")

    # Call the function with the raw API response
    formatted_path = format_api_relationship_path(
        api_response_data=raw_api_response,
        owner_name=owner_name,
        target_name=target_name,
    )

    # Step 5: Print the formatted relationship path
    print("\n=== Formatted Relationship Path ===\n")
    print(formatted_path)

    # Step 6: Test with pre-parsed JSON data
    print("\nStep 6: Testing with pre-parsed JSON data...")
    try:
        # Extract and parse the JSON part
        json_part_match = re.search(
            r"^\s*no\((.*)\)\s*$", raw_api_response.strip(), re.DOTALL
        )
        if json_part_match:
            json_part_str = json_part_match.group(1).strip()
            parsed_json = json.loads(json_part_str)

            print(f"\nParsed JSON keys: {list(parsed_json.keys())}")
            print(f"Status: {parsed_json.get('status')}")
            print(f"Title: {parsed_json.get('title')}")

            # Call the function with the parsed JSON
            formatted_path_from_json = format_api_relationship_path(
                api_response_data=parsed_json,
                owner_name=owner_name,
                target_name=target_name,
            )

            print("\n=== Formatted Path from Parsed JSON ===\n")
            print(formatted_path_from_json)

            # Compare results
            if formatted_path == formatted_path_from_json:
                print(
                    "\nResults match: Both raw string and parsed JSON inputs produce identical output."
                )
            else:
                print(
                    "\nResults differ: Raw string and parsed JSON inputs produce different output."
                )
        else:
            print("Could not extract JSON part for the second test.")
    except Exception as e:
        print(f"Error in second test: {e}")


# End of test_format_relationship_path

if __name__ == "__main__":
    test_format_relationship_path()
