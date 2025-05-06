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
    Simple implementation that just returns the input, as API provides "3rd".

    Args:
        text: The text to format

    Returns:
        Formatted text
    """
    # For "3rd great-grandmother", the API already provides "3rd".
    # This function can be a passthrough if the input is already correct.
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

    # Apply ordinal case if the term contains digits
    if any(char.isdigit() for char in term):
        try:
            term = ordinal_case(term)
        except Exception as ord_err:
            logger.warning(f"Failed to apply ordinal case to '{term}': {ord_err}")
        # End of try/except
    # End of if
    return term


# End of _get_relationship_term


def _extract_name_and_lifespan_from_html_text(text_content: str) -> Tuple[str, str]:
    """
    Parses a raw text string to separate name and format lifespan.
    Example inputs: "Name YYYY-YYYY", "Name YYYY-", "Name"

    Returns:
        Tuple (name_part_str, formatted_lifespan_str)
        formatted_lifespan_str is like "(YYYY-YYYY)" or "(b. YYYY)" or ""
    """
    name_part = text_content
    lifespan_str_formatted = ""

    # Pattern 1: "Name YYYY-YYYY" or "Name YYYY–YYYY" (long dash)
    match_yyyy_yyyy = re.search(r"\s+(\d{4}[–-]\d{4})$", text_content)
    if match_yyyy_yyyy:
        lifespan_raw = match_yyyy_yyyy.group(1)
        # Normalize dash and format
        lifespan_str_formatted = f"({lifespan_raw.replace('–', '-')})"
        name_part = text_content[: match_yyyy_yyyy.start()].strip()
        return name_part, lifespan_str_formatted
    # End of if

    # Pattern 2: "Name YYYY-" (living, birth year known)
    match_yyyy_living = re.search(r"\s+(\d{4})-$", text_content)
    if match_yyyy_living:
        birth_year = match_yyyy_living.group(1)
        lifespan_str_formatted = f"(b. {birth_year})"
        name_part = text_content[: match_yyyy_living.start()].strip()
        return name_part, lifespan_str_formatted
    # End of if

    # Pattern 3: "Name YYYY" (could be birth or death year alone, assume birth for (b. YYYY) if ambiguous)
    # This pattern is less specific. For getladder, explicit "YYYY-" or "YYYY-YYYY" is more common.
    # If the HTML provides just "Name YYYY", and it's meant as birth, it would need context.
    # For now, this helper primarily focuses on explicit "YYYY-YYYY" and "YYYY-".
    # If no specific pattern matches, lifespan_str_formatted remains ""

    return name_part.strip(), lifespan_str_formatted


# End of _extract_name_and_lifespan_from_html_text


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
    # End of if

    # --- Initialize variables ---
    html_content_raw: Optional[str] = None  # Raw HTML string from JSONP
    json_data: Optional[Dict] = None  # Parsed JSON data if input is dict
    api_status: str = "unknown"
    response_source: str = "Unknown"  # 'JSONP', 'JSON', 'RawString'
    name_formatter = format_name  # Use imported or fallback formatter

    # --- Step 1: Process Input Data Type ---
    if isinstance(api_response_data, dict):
        response_source = "JSON"
        if "error" in api_response_data:
            return f"(API returned error object: {api_response_data.get('error', 'Unknown')})"
        # End of if
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
        # End of if
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
                    json_part_str = json_part_match.group(1).strip()
                    logger.debug(f"Extracted JSON part: {json_part_str[:100]}...")
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
                        # End of if
                    else:
                        return f"(API status '{api_status}' in JSONP: {parsed_json.get('message', 'Error')})"
                    # End of if/else
                else:
                    logger.warning("Could not extract JSON part from JSONP wrapper.")
                    html_content_raw = api_response_data
                    response_source = "RawString"
                # End of if/else
            except json.JSONDecodeError as json_err:
                logger.error(
                    f"Error decoding JSON part from {response_source}: {json_err}"
                )
                error_context = (
                    f" near: {json_part_str[:100]}..."
                    if "json_part_str" in locals()
                    else ""
                )
                return f"(Error parsing JSONP data: {json_err}{error_context})"
            # End of try/except
            except Exception as e:
                logger.error(f"Error processing {response_source}: {e}", exc_info=True)
                html_content_raw = api_response_data
                response_source = "RawString"
            # End of try/except
        else:
            html_content_raw = api_response_data
            response_source = "RawString"
        # End of if/else
    else:
        return f"(Unsupported data type received: {type(api_response_data)})"
    # End of if/elif/else

    # --- Step 2: Format Discovery API JSON Path (if applicable) ---
    if json_data and "path" in json_data:
        path_steps_json = []
        discovery_path = json_data["path"]
        if isinstance(discovery_path, list) and discovery_path:
            logger.info("Formatting relationship path from Discovery API JSON.")
            path_steps_json.append(f"*   {name_formatter(target_name)}")
            for i, step in enumerate(discovery_path):
                step_name = name_formatter(step.get("name", "?"))
                step_rel = step.get("relationship", "?")
                step_rel_display = _get_relationship_term(None, step_rel).capitalize()
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

    # --- Step 3: Process HTML Content (from /getladder JSONP) ---
    if not html_content_raw:
        logger.warning("No processable HTML content found for relationship path.")
        if response_source == "JSONP" and api_status != "success":
            return f"(API status '{api_status}' in JSONP, no HTML content)"
        # End of if
        return "(Could not find or extract relationship HTML content)"
    # End of if

    html_content_decoded: Optional[str] = None
    try:
        html_content_intermediate = html.unescape(html_content_raw)
        html_content_decoded = bytes(html_content_intermediate, "utf-8").decode(
            "unicode_escape"
        )
        logger.debug(f"Decoded HTML content: {html_content_decoded[:250]}...")
    except Exception as decode_err:
        logger.error(f"Failed to decode HTML content: {decode_err}", exc_info=True)
        html_content_decoded = html_content_raw
    # End of try/except

    if not BS4_AVAILABLE or not BeautifulSoup:
        logger.error("BeautifulSoup library not available. Cannot parse HTML.")
        return "(Cannot parse relationship HTML - BeautifulSoup library missing)"
    # End of if

    # --- Step 4: Parse Decoded HTML with BeautifulSoup ---
    try:
        logger.debug("Attempting to parse DECODED HTML content with BeautifulSoup...")
        soup = None
        parser_to_try = ["lxml", "html.parser"]
        for parser_name in parser_to_try:
            try:
                soup = BeautifulSoup(html_content_decoded, parser_name)
                logger.info(f"Successfully parsed HTML using '{parser_name}'.")
                break
            except Exception as parse_err:
                logger.warning(
                    f"Error using '{parser_name}' parser: {parse_err}. Trying next."
                )
            # End of try/except
        # End of for

        if not soup:
            logger.error("BeautifulSoup failed to parse HTML with available parsers.")
            return "(Error parsing relationship HTML - BeautifulSoup failed)"
        # End of if

        # --- Step 5: Extract Path Information from Parsed HTML ---
        list_items = soup.select("ul.textCenter li")
        path_items = [
            li for li in list_items if "iconArrowDown" not in (li.get("class") or [])
        ]
        logger.debug(f"Found {len(path_items)} relevant path items (summary + people).")

        if not path_items:  # Need at least the target person
            logger.warning(
                "Expected list items ('ul.textCenter li'), found none in parsed HTML or no relevant items."
            )
            logger.debug(
                f"Parsed HTML structure (abbreviated):\n{soup.prettify()[:500]}"
            )
            return "(Relationship HTML structure not as expected - Found 0 relevant list items)"
        # End of if

        # --- Step 6: Build Formatted Output String ---
        path_lines = []

        # 1. Extract Target Info and Overall Relationship for Summary Line
        target_li = path_items[0]
        target_name_tag = target_li.find("b")  # Target name usually in <b>
        target_name_raw_html = (
            target_name_tag.get_text(strip=True) if target_name_tag else target_name
        )

        # Use helper to get name and lifespan from the raw text of the first item
        # This raw text includes name and year e.g. "Elizabeth 'Betty' Cruickshank 1839-1886"
        _, target_lifespan_from_html = _extract_name_and_lifespan_from_html_text(
            target_name_raw_html
        )

        # Use the function argument `target_name` for display consistency in summary
        formatted_target_name_for_summary = name_formatter(target_name)

        overall_relationship_text = "Unknown Relationship"
        # Overall relationship is in <i><b>...</b></i> or <i>...</i> within the first <li>
        summary_tag_html = target_li.select_one("i b, i")
        if summary_tag_html:
            overall_relationship_text = summary_tag_html.get_text(strip=True)
            # The API provides "3rd great-grandmother", so ordinal_case might not be needed
            # if it's just a passthrough.
            overall_relationship_text = ordinal_case(overall_relationship_text)
        # End of if

        summary_line = f"{formatted_target_name_for_summary} {target_lifespan_from_html} is {name_formatter(owner_name)}'s {overall_relationship_text}:"

        # 2. Iterate for Path Lines
        for i in range(1, len(path_items)):
            current_person_li = path_items[i]

            # Determine the name of the "previous person" for the relationship string
            if i == 1:  # Current is the first person after target
                # The relationship is to the main target_name
                prev_person_name_for_rel_str = name_formatter(target_name)
            else:  # Current is second person onwards in the list
                prev_person_li_for_name = path_items[i - 1]
                prev_name_tag_for_rel = prev_person_li_for_name.find(
                    "a"
                ) or prev_person_li_for_name.find("b")
                prev_name_raw_for_rel = (
                    prev_name_tag_for_rel.get_text(strip=True)
                    if prev_name_tag_for_rel
                    else "Unknown Previous"
                )
                parsed_prev_name, _ = _extract_name_and_lifespan_from_html_text(
                    prev_name_raw_for_rel
                )
                prev_person_name_for_rel_str = name_formatter(parsed_prev_name)
                if (
                    prev_person_name_for_rel_str.lower() == "you"
                ):  # Handle if previous was owner
                    prev_person_name_for_rel_str = name_formatter(owner_name)
                # End of if
            # End of if/else

            # Extract current person's details
            current_name_tag = current_person_li.find("a") or current_person_li.find(
                "b"
            )
            current_name_raw_from_tag = (
                current_name_tag.get_text(strip=True)
                if current_name_tag
                else "Unknown Current"
            )

            parsed_current_name, current_lifespan_str = (
                _extract_name_and_lifespan_from_html_text(current_name_raw_from_tag)
            )
            current_name_formatted = name_formatter(parsed_current_name)

            # If the current person is the owner, ensure name matches owner_name argument
            if current_name_formatted.lower() == "you" or (
                i == len(path_items) - 1
                and name_formatter(owner_name).startswith(current_name_formatted)
            ):
                current_name_formatted = name_formatter(owner_name)
                # If owner name from HTML didn't have lifespan, and current_lifespan_str is empty, it stays empty.
            # End of if

            # Extract relationship term from current <li>'s <i> tag
            relationship_term_text = "related"  # Default
            desc_tag_html = current_person_li.find("i")
            if desc_tag_html:
                desc_text_html = desc_tag_html.get_text(strip=True)
                # Regex for "Son of Margaret Simpson", "Daughter of...", "You are the son of..."
                rel_match_html = re.search(
                    r"\b(son|daughter|father|mother|husband|wife|spouse|brother|sister|parent|child|sibling)\b",
                    desc_text_html,
                    re.IGNORECASE,
                )
                if rel_match_html:
                    relationship_term_text = rel_match_html.group(1).lower()
                elif "you are the" in desc_text_html.lower():
                    you_match = re.search(
                        r"You\s+are\s+the\s+([\w\s]+)\s+of",
                        desc_text_html,
                        re.IGNORECASE,
                    )
                    if you_match:
                        relationship_term_text = you_match.group(1).strip().lower()
                    # End of if
                # End of if/elif
            # End of if

            path_line = f"*   {current_name_formatted} {current_lifespan_str} is {prev_person_name_for_rel_str}'s {relationship_term_text}"
            path_lines.append(path_line.strip())
        # End of for

        # --- Step 7: Combine and Return Formatted String ---
        result_str = f"{summary_line}\n\n" + "\n".join(path_lines)
        logger.info("Formatted relationship path from HTML successfully.")
        logger.debug(f"Formatted HTML relationship path:\n{result_str}")
        return result_str

    except Exception as e:
        logger.error(
            f"Error processing relationship HTML with BeautifulSoup: {e}", exc_info=True
        )
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
        # End of if

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
        # End of if/else

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None
    # End of try/except
    except Exception as e:
        logger.error(f"Error extracting HTML from JSONP: {e}")
        return None
    # End of try/except


# End of extract_html_from_jsonp


def decode_html_content(html_content):
    """
    Decode HTML content by unescaping HTML entities and unicode escapes.

    This function performs two-step decoding:
    1. First, it unescapes HTML entities (e.g., < becomes <)
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
    # End of try/except


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
    # End of if

    print(f"\nExtracted HTML (first 200 chars):\n{html_content[:200]}...\n")

    # Step 2: Decode the HTML content
    print("Step 2: Decoding HTML content...")
    decoded_html = decode_html_content(html_content)
    print(f"\nDecoded HTML (first 200 chars):\n{decoded_html[:200]}...\n")

    # Step 3: Parse the HTML using BeautifulSoup
    print("Step 3: Parsing HTML with BeautifulSoup...")
    try:
        # from bs4 import BeautifulSoup # Already imported globally if available

        if not BS4_AVAILABLE or not BeautifulSoup:
            raise ImportError("BeautifulSoup not available")
        # End of if

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
            summary_tag = path_items[0].select_one("i b, i")  # More specific selector
            if summary_tag:
                print(f"Relationship summary: {summary_tag.get_text(strip=True)}")
            # End of if

            # Print a few people in the path
            print("\nPeople in the path:")
            for i_loop_var, item in enumerate(path_items):  # Renamed i to i_loop_var
                if i_loop_var > 0 and i_loop_var < 4:  # Just show a few examples
                    name_tag_local = item.find("a") or item.find(
                        "b"
                    )  # Renamed name_tag
                    name_local = (
                        name_tag_local.get_text(strip=True)
                        if name_tag_local
                        else "Unknown"
                    )  # Renamed name
                    desc_tag_local = item.find("i")  # Renamed desc_tag
                    desc_local = (
                        desc_tag_local.get_text(strip=True) if desc_tag_local else ""
                    )  # Renamed desc
                    print(f"  Person {i_loop_var}: {name_local} - {desc_local}")
                # End of if
            # End of for
        # End of if
    except ImportError:
        print("BeautifulSoup not available - skipping HTML parsing demonstration")
    # End of try/except
    except Exception as e:
        print(f"Error parsing HTML: {e}")
    # End of try/except

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
            # End of if/else
        else:
            print("Could not extract JSON part for the second test.")
        # End of if/else
    except Exception as e:
        print(f"Error in second test: {e}")
    # End of try/except


# End of test_format_relationship_path

if __name__ == "__main__":
    test_format_relationship_path()
# End of if
