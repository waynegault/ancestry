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

# Set up logging to only show CRITICAL errors, effectively silencing DEBUG/INFO for this script's run
logging.basicConfig(level=logging.CRITICAL)
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
        lifespan_str_formatted = f"({lifespan_raw.replace('–', '-')})"  # Normalize dash
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

    html_content_raw: Optional[str] = None
    json_data: Optional[Dict] = None
    api_status: str = "unknown"
    response_source: str = "Unknown"
    name_formatter = format_name

    if isinstance(api_response_data, dict):
        response_source = "JSON"
        if "error" in api_response_data:
            return f"(API returned error object: {api_response_data.get('error', 'Unknown')})"
        # End of if
        if "path" in api_response_data and isinstance(
            api_response_data.get("path"), list
        ):
            json_data = api_response_data
        elif (
            "html" in api_response_data
            and "status" in api_response_data
            and isinstance(api_response_data.get("html"), str)
        ):
            html_content_raw = api_response_data.get("html")
            api_status = api_response_data.get("status", "unknown")
            if api_status != "success":
                return f"(API returned status '{api_status}': {api_response_data.get('message', 'Unknown Error')})"
            # End of if
        else:
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
                    parsed_json = json.loads(json_part_str)
                    api_status = parsed_json.get("status", "unknown")
                    if api_status == "success":
                        html_content_raw = parsed_json.get("html")
                        if not isinstance(html_content_raw, str):
                            html_content_raw = None
                        # End of if
                    else:
                        return f"(API status '{api_status}' in JSONP: {parsed_json.get('message', 'Error')})"
                    # End of if/else
                else:
                    html_content_raw = api_response_data
                    response_source = "RawString"
                # End of if/else
            except json.JSONDecodeError as json_err:
                error_context = (
                    f" near: {json_part_str[:100]}..."
                    if "json_part_str" in locals()
                    else ""
                )
                return f"(Error parsing JSONP data: {json_err}{error_context})"
            # End of try/except
            except Exception as e:
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

    if json_data and "path" in json_data:  # Discovery API JSON Path
        path_steps_json = []
        discovery_path = json_data["path"]
        if isinstance(discovery_path, list) and discovery_path:
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
            return "\n".join(path_steps_json)
        else:
            return "(Discovery path found but is empty or invalid)"
        # End of if/else
    # End of if json_data

    if not html_content_raw:
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
    except Exception as decode_err:
        html_content_decoded = html_content_raw
    # End of try/except

    if not BS4_AVAILABLE or not BeautifulSoup:
        return "(Cannot parse relationship HTML - BeautifulSoup library missing)"
    # End of if

    try:
        soup = None
        for parser_name in ["lxml", "html.parser"]:
            try:
                soup = BeautifulSoup(html_content_decoded, parser_name)
                break
            except Exception:
                continue  # Try next parser
            # End of try/except
        # End of for

        if not soup:
            return "(Error parsing relationship HTML - BeautifulSoup failed)"
        # End of if

        list_items = soup.select("ul.textCenter li")
        path_items = [
            li for li in list_items if "iconArrowDown" not in (li.get("class") or [])
        ]

        if not path_items:
            return "(Relationship HTML structure not as expected - Found 0 relevant list items)"
        # End of if

        # --- Summary Line Construction ---
        target_li = path_items[0]
        target_b_tag = target_li.find("b")
        target_year_text_raw = ""
        if (
            target_b_tag
            and target_b_tag.next_sibling
            and isinstance(target_b_tag.next_sibling, str)
        ):
            target_year_text_raw = (
                target_b_tag.next_sibling.strip()
            )  # e.g., "1839-1886"
        # End of if

        target_lifespan_formatted = ""
        if re.fullmatch(r"\d{4}[–-]\d{4}", target_year_text_raw):
            target_lifespan_formatted = f"({target_year_text_raw.replace('–','-')})"
        elif re.fullmatch(
            r"\d{4}-", target_year_text_raw
        ):  # Should not happen for target
            target_lifespan_formatted = f"(b. {target_year_text_raw[:-1]})"
        # End of if/elif

        overall_relationship_text = "Unknown Relationship"
        summary_tag_html = target_li.select_one("i b, i")
        if summary_tag_html:
            overall_relationship_text = summary_tag_html.get_text(strip=True)
            overall_relationship_text = ordinal_case(overall_relationship_text)
        # End of if

        summary_line_parts = [name_formatter(target_name)]
        if target_lifespan_formatted:
            summary_line_parts.append(target_lifespan_formatted)
        # End of if
        summary_line_parts.append(
            f"is {name_formatter(owner_name)}'s {overall_relationship_text}:"
        )
        summary_line = " ".join(summary_line_parts)

        # --- Path Lines Construction ---
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
                _extract_name_and_lifespan_from_html_text(current_name_raw_from_tag)
            )
            current_name_display = name_formatter(current_name_parsed)

            # If current is owner, ensure name matches owner_name argument
            if current_name_parsed.lower() == "you" or (
                i == len(path_items) - 1
                and name_formatter(owner_name).startswith(current_name_display)
            ):
                current_name_display = name_formatter(owner_name)
            # End of if

            # Determine previous person's name for relationship string
            prev_name_display_for_relation: str
            if (
                i == 1
            ):  # Current is first person in path, relationship is to target_name
                prev_name_display_for_relation = name_formatter(target_name)
            else:  # Relationship is to the person from the previous <li>
                prev_li = path_items[i - 1]
                prev_name_tag = prev_li.find("a") or prev_li.find("b")
                prev_name_raw = (
                    prev_name_tag.get_text(strip=True)
                    if prev_name_tag
                    else "Unknown Previous"
                )
                parsed_prev_name, _ = _extract_name_and_lifespan_from_html_text(
                    prev_name_raw
                )
                prev_name_display_for_relation = name_formatter(parsed_prev_name)
                if parsed_prev_name.lower() == "you":
                    prev_name_display_for_relation = name_formatter(owner_name)
                # End of if
            # End of if/else

            relationship_term = "related"  # Default
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

            line_parts = ["*  ", current_name_display]  # Two spaces after asterisk
            if current_lifespan_formatted:
                line_parts.append(current_lifespan_formatted)
            # End of if
            line_parts.append(
                f"is {prev_name_display_for_relation}'s {relationship_term}"
            )
            path_lines.append(" ".join(line_parts))
        # End of for

        return f"{summary_line}\n\n" + "\n".join(path_lines)

    except Exception as e:
        error_context = (
            f" near HTML: {html_content_decoded[:200]}..."
            if html_content_decoded
            else ""
        )
        return f"(Error parsing relationship HTML: {e}{error_context})"


# End of format_api_relationship_path


# extract_html_from_jsonp and decode_html_content are effectively inlined or
# their logic is covered by format_api_relationship_path's initial processing.
# They are kept here for modularity if ever needed separately but aren't
# called directly by the streamlined test_format_relationship_path below.


def extract_html_from_jsonp(jsonp_str):
    """
    Extract HTML content from JSONP response string.
    """
    try:
        json_part_match = re.search(r"^\s*no\((.*)\)\s*$", jsonp_str.strip(), re.DOTALL)
        if not json_part_match:
            return None
        # End of if
        json_part_str = json_part_match.group(1).strip()
        parsed_json = json.loads(json_part_str)
        if "html" in parsed_json and parsed_json.get("status") == "success":
            return parsed_json.get("html")
        # End of if
        return None
    except Exception:
        return None
    # End of try/except


# End of extract_html_from_jsonp


def decode_html_content(html_content):
    """
    Decode HTML content by unescaping HTML entities and unicode escapes.
    """
    try:
        html_content_intermediate = html.unescape(html_content)
        return bytes(html_content_intermediate, "utf-8").decode("unicode_escape")
    except Exception:
        return html_content
    # End of try/except


# End of decode_html_content


def test_format_relationship_path():
    """
    Tests the format_api_relationship_path function and prints only the final formatted output.
    """
    target_name = "Elizabeth 'Betty' Cruickshank"
    owner_name = "Wayne Gordon Gault"

    formatted_path = format_api_relationship_path(
        api_response_data=raw_api_response,
        owner_name=owner_name,
        target_name=target_name,
    )
    print(formatted_path)


# End of test_format_relationship_path

if __name__ == "__main__":
    test_format_relationship_path()
# End of if
