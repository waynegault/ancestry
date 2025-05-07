"""
Test script for relationship path formatting functionality.
Processes a specific raw API response to a defined output format.
"""

import logging
import json
import re
import html
from typing import Dict, Optional, Union, Tuple  # type: ignore

# Try to import BeautifulSoup, but handle the case where it's not available
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None  # type: ignore
    BS4_AVAILABLE = False

# Assume utils.py is available in the PYTHONPATH or same directory for these imports
from utils import format_name, ordinal_case


# Set up logging to only show CRITICAL errors, effectively silencing other messages
logging.basicConfig(level=logging.CRITICAL)
# logger = logging.getLogger(__name__) # Not strictly needed if no logging calls remain

# Test raw API response
raw_api_response = r"""
no({
    "html": "\u003cul class=\"textCenter\"\u003e \u003cli\u003e\u003cb\u003eElizabeth \u0027Betty\u0027 Cruickshank\u003c/b\u003e 1839-1886\u003cbr /\u003e\u003ci\u003e\u003cb\u003e3rd great-grandmother\u003c/b\u003e\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560698\"\u003eMargaret Simpson 1865-1946\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eDaughter of Elizabeth \u0027Betty\u0027 Cruickshank\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560684\"\u003eAlexander Stables 1899-1948\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eSon of Margaret Simpson\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560677\"\u003eCatherine Margaret Stables 1924-2004\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eDaughter of Alexander Stables\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"relative\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560544\"\u003eFrances Margaret Milne 1947-\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eDaughter of Catherine Margaret Stables\u003c/i\u003e\u003c/li\u003e \u003cli aria-hidden=\"true\" class=\"icon iconArrowDown\"\u003e\u003c/li\u003e \u003cli\u003e\u003ca class=\"bottomName\" href=\"https://www.ancestry.co.uk/family-tree/person/tree/175946702/person/102281560836\"\u003e\u003cb\u003eWayne Gordon Gault\u003c/b\u003e\u003c/a\u003e\u003cbr /\u003e\u003ci\u003eYou are the son of Frances Margaret Milne\u003c/i\u003e\u003c/li\u003e \u003c/ul\u003e ",
    "title": "Relationship to me",
    "printText": "Print",
    "status": "success"
})
"""


def format_api_relationship_path(
    api_response_data: Union[str, Dict, None], owner_name: str, target_name: str
) -> str:
    """
    Parses relationship data from Ancestry getladder API HTML and formats it.
    Uses imported format_name and ordinal_case from utils.py.
    """

    # --- Inner Helper Function ---
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
        return "(No relationship data received from API)"
    # End of if

    html_content_raw: Optional[str] = None
    api_status: str = "unknown"

    if isinstance(
        api_response_data, str
    ):  # Expecting JSONP string for this specific task
        if api_response_data.strip().startswith(
            "no("
        ) and api_response_data.strip().endswith(")"):
            try:
                json_part_match = re.search(
                    r"^\s*no\((.*)\)\s*$", api_response_data, re.DOTALL
                )
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
                else:  # Should not happen with the given raw_api_response
                    return "(Could not extract JSON from JSONP wrapper)"
                # End of if/else
            except json.JSONDecodeError as json_err:
                error_context = (
                    f" near: {json_part_str[:100]}..."  # pyright: ignore[reportUnboundVariable]
                    if "json_part_str" in locals()
                    else ""
                )
                return f"(Error parsing JSONP data: {json_err}{error_context})"
            # End of try/except
            except Exception as e:
                return f"(Unexpected error processing JSONP: {e})"
            # End of try/except
        else:  # Not the expected JSONP format
            return "(Input string not in expected no(...) JSONP format)"
        # End of if/else
    elif isinstance(api_response_data, dict):  # Handle if pre-parsed JSON is passed
        if "html" in api_response_data and isinstance(
            api_response_data.get("html"), str
        ):
            html_content_raw = api_response_data.get("html")
            api_status = api_response_data.get("status", "unknown")
            if api_status != "success":
                return f"(API returned status '{api_status}': {api_response_data.get('message', 'Unknown Error')})"
            # End of if
        else:
            return (
                "(Input dictionary does not contain 'html' or status is not 'success')"
            )
        # End of if/else
    else:
        return f"(Unsupported data type received: {type(api_response_data)})"
    # End of if/elif/else

    if not html_content_raw:
        return "(Could not find or extract relationship HTML content)"
    # End of if

    html_content_decoded: Optional[str] = None
    try:
        html_content_intermediate = html.unescape(html_content_raw)
        html_content_decoded = bytes(html_content_intermediate, "utf-8").decode(
            "unicode_escape"
        )
    except Exception:  # Keep it simple, if decode fails, it fails.
        html_content_decoded = html_content_raw  # Fallback to raw if decode error
    # End of try/except

    if not BS4_AVAILABLE or not BeautifulSoup:
        return "(Cannot parse relationship HTML - BeautifulSoup library missing)"
    # End of if

    try:
        soup = None
        for parser_name in ["lxml", "html.parser"]:  # Try lxml first
            try:
                soup = BeautifulSoup(html_content_decoded, parser_name)
                break
            except Exception:
                continue
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
        summary_tag_html = target_li.select_one("i b, i")  # Handles <i><b> or just <i>
        if summary_tag_html:
            overall_relationship_text = summary_tag_html.get_text(strip=True)
            overall_relationship_text = ordinal_case(overall_relationship_text)  # type: ignore[possibly-undefined]
        # End of if

        summary_line_parts = [format_name(target_name)]
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
                if (
                    parsed_prev_name.lower() == "you"
                ):  # Handles if "You" (owner) is an intermediate step
                    prev_name_display_for_relation = format_name(owner_name)
                # End of if
            # End of if/else

            relationship_term = "related"  # Default
            desc_tag_html = current_person_li.find("i")
            if desc_tag_html:
                desc_text_html = desc_tag_html.get_text(strip=True)
                # Regex to find primary relationship term like son, daughter, etc.
                rel_match_html = re.search(
                    r"\b(son|daughter|father|mother|husband|wife|spouse|brother|sister|parent|child|sibling)\b",
                    desc_text_html,
                    re.IGNORECASE,
                )
                if rel_match_html:
                    relationship_term = rel_match_html.group(1).lower()
                elif (
                    "you are the" in desc_text_html.lower()
                ):  # Handles specific "You are the X of Y"
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
            f" near HTML: {html_content_decoded[:200]}..."  # pyright: ignore[reportUnboundVariable]
            if "html_content_decoded" in locals() and html_content_decoded
            else ""
        )
        return f"(Error parsing relationship HTML: {e}{error_context})"


# End of format_api_relationship_path


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


def test_gedcom_relationship_path():
    """
    Tests the get_relationship_path function in gedcom_utils.py.
    This tests the relationship path between Wayne Gault and Fraser Gault.
    """
    try:
        # Import the necessary modules
        from gedcom_utils import GedcomData
        from config import config_instance

        # Get the GEDCOM file path from the config
        gedcom_path = config_instance.GEDCOM_FILE_PATH
        if not gedcom_path:
            print("ERROR: GEDCOM_FILE_PATH not set in config.")
            return False

        # Create a GedcomData instance
        try:
            gedcom_data = GedcomData(gedcom_path)
        except Exception as e:
            print(f"ERROR: Failed to load GEDCOM file: {e}")
            return False

        # Define the IDs for Wayne Gault and Fraser Gault
        wayne_id = "I102281560836"  # Wayne Gordon Gault
        fraser_id = "I102281560744"  # Fraser Gault

        # Get the relationship path
        print(f"\nGetting relationship path between Wayne Gault and Fraser Gault...")
        relationship_path = gedcom_data.get_relationship_path(wayne_id, fraser_id)

        # Print the relationship path
        print("\n=== GEDCOM Relationship Path ===")
        print(relationship_path)
        print("===============================\n")

        # Check if the relationship path contains the expected names
        expected_names = ["Wayne", "Derrick", "James", "Fraser"]
        for name in expected_names:
            if name not in relationship_path:
                print(
                    f"WARNING: Expected name '{name}' not found in relationship path."
                )

        return True
    except Exception as e:
        print(f"ERROR: Unexpected error in test_gedcom_relationship_path: {e}")
        return False


if __name__ == "__main__":
    print("=== Testing API Relationship Path Formatting ===")
    test_format_relationship_path()

    print("\n=== Testing GEDCOM Relationship Path ===")
    test_gedcom_relationship_path()
# End of if
