# api_utils.py
"""
Utility functions for interacting with the Ancestry API.
Includes session management initialization, response parsing (like relationship ladder),
and general formatting helpers used by API display functions.
Relies on the external 'utils' module for core SessionManager.
_api_req is accessed as a method of the SessionManager instance.
"""

import logging
import sys
import re
import os
import time
import json
import requests
import urllib.parse
import html
from typing import Optional, Dict, Any, Union
from bs4 import BeautifulSoup

# Add parent directory to sys.path to import from sibling directories
# Adjust based on actual project structure
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules - Adjust path as necessary
try:
    from logging_config import setup_logging
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    def setup_logging(log_file="api_utils.log", log_level="INFO"):
        logger = logging.getLogger("api_utils_fallback")
        return logger


try:
    from config import config_instance
except ImportError:
    logging.error("Failed to import config_instance. API functionality may be limited.")

    class DummyConfig:
        GEDCOM_FILE_PATH = None
        BASE_URL = "https://www.ancestry.com"  # Default fallback
        USER_AGENTS = ["Mozilla/5.0"]

    config_instance = DummyConfig()

# Setup logging
logger = setup_logging(
    log_file="gedcom_processor.log", log_level="INFO"
)  # Use shared log file

# --- Attempt to import SessionManager from the original 'utils' module ---
API_UTILS_AVAILABLE = False
try:
    # ***** CORRECTED IMPORT LINE *****
    # Only import SessionManager, rename for clarity if needed
    from utils import SessionManager as UtilsSessionManager

    # ***********************************

    API_UTILS_AVAILABLE = True
    # Update the log message as we are no longer importing _api_req here
    logger.debug("Successfully imported SessionManager from 'utils'.")
except ImportError as e_utils:
    # This block should NOT execute if SessionManager imports correctly from utils.py
    logger.warning(f"Could not import SessionManager from 'utils' module: {e_utils}")
    logger.warning("API functionality will be disabled.")

    # Define dummy classes/functions if utils is missing
    class UtilsSessionManager:  # Renamed to avoid conflict if SessionManager is defined below
        def __init__(self):
            self.driver_live = False
            self.session_ready = False
            self.my_tree_id = None
            self.my_profile_id = None
            self.my_uuid = None
            self.driver = None

        def ensure_driver_live(self):
            pass

        def ensure_session_ready(self):
            return False

        def check_session_status(self):
            pass

        def _retrieve_identifiers(self):
            pass

        def _sync_cookies(self):
            pass

        def get_csrf(self):
            return None

        # Add dummy _api_req to the dummy class
        def _api_req(self, *args, **kwargs):
            logger.error(
                "API request attempted but 'utils' module/SessionManager failed to load."
            )
            return {"error": "'utils' module unavailable"}

        def quit_driver(self):
            pass

    # No need for a separate dummy _api_req function if UtilsSessionManager is defined above with the method
    API_UTILS_AVAILABLE = False


# --- Import Formatting Function from gedcom_utils ---
try:
    from gedcom_utils import format_name
except ImportError:
    logger.warning(
        "Could not import format_name from gedcom_utils. Using basic fallback."
    )

    def format_name(name: Optional[str]) -> str:
        return str(name).strip().title() if name else "Unknown"


# --- Global Session Manager Instance ---
# Use the imported SessionManager if available, otherwise the dummy one
session_manager = UtilsSessionManager()
logger.info(
    f"Session Manager initialized (type: {type(session_manager).__name__}). API Utils Available: {API_UTILS_AVAILABLE}"
)


# --- Authentication and Session Initialization ---


def initialize_session() -> bool:
    """Initialize the session with proper authentication for standalone usage."""
    global session_manager
    if not API_UTILS_AVAILABLE:
        logger.error(
            "Cannot initialize session: API utilities (SessionManager) are missing."
        )
        return False

    # Ensure driver is live
    if not session_manager.driver_live:
        logger.info("Initializing browser session...")
        try:
            session_manager.ensure_driver_live()
        except Exception as e:
            logger.error(f"Failed to start browser driver: {e}", exc_info=True)
            return False

    # Ensure session is ready (authenticated)
    if not session_manager.session_ready:
        # ensure_session_ready should handle login/checks internally
        success = session_manager.ensure_session_ready()
        if not success:
            logger.warning("Failed to ensure session ready automatically.")
            # session_manager.ensure_session_ready() might internally prompt for manual login
            # Re-check status after potential manual intervention (if applicable in ensure_session_ready logic)
            if not session_manager.session_ready:
                logger.error(
                    "Session still not ready after ensure_session_ready attempt."
                )
                return False
        else:
            logger.info("Session is ready.")

    # Ensure necessary identifiers are loaded (ensure_session_ready should handle this)
    if (
        not session_manager.my_tree_id
        or not session_manager.my_profile_id
        or not session_manager.my_uuid
    ):
        logger.info(
            "Loading tree/profile/uuid identifiers (might be redundant if ensure_session_ready handles it)..."
        )
        try:
            # _retrieve_identifiers is usually called within ensure_session_ready
            if hasattr(session_manager, "_retrieve_identifiers") and callable(
                session_manager._retrieve_identifiers
            ):
                session_manager._retrieve_identifiers()
            else:
                logger.warning(
                    "_retrieve_identifiers method not found on session_manager."
                )
        except Exception as e:
            logger.error(f"Failed to retrieve identifiers: {e}")
            # Decide if this is critical - may depend on API calls needed
            # return False # Optional: fail if identifiers are crucial

        if not session_manager.my_tree_id:
            logger.warning("Could not load tree ID.")
        else:
            logger.info(f"Tree ID loaded: {session_manager.my_tree_id}")
        if not session_manager.my_profile_id:
            logger.warning("Could not load profile ID.")
        else:
            logger.info(f"Profile ID loaded: {session_manager.my_profile_id}")
        if not session_manager.my_uuid:
            logger.warning("Could not load UUID.")
        else:
            logger.info(f"UUID loaded: {session_manager.my_uuid}")

    return session_manager.session_ready


# --- Formatting Helpers ---


def ordinal_case(value: Union[str, int]) -> str:
    """Converts number or string number to ordinal format. Also handles simple relationship capitalization."""
    if isinstance(value, str) and not value.isdigit():
        words = value.title().split()
        # Simple fixes for common prepositions/articles
        lc_words = {"Of", "The", "A", "An", "In", "On", "At", "For", "To", "With"}
        for i, word in enumerate(words):
            if i > 0 and word in lc_words:  # Don't lowercase the first word
                words[i] = word.lower()
        return " ".join(words)

    try:
        num = int(value)
        if 11 <= (num % 100) <= 13:
            suffix = "th"
        else:
            last_digit = num % 10
            if last_digit == 1:
                suffix = "st"
            elif last_digit == 2:
                suffix = "nd"
            elif last_digit == 3:
                suffix = "rd"
            else:
                suffix = "th"
        return str(num) + suffix
    except (ValueError, TypeError):
        return str(value)  # Return original as string if conversion fails


# --- API Response Parsing ---


def display_raw_relationship_ladder(raw_content: Union[str, Dict]):
    """
    Parse and display the Ancestry relationship ladder from raw JSONP/HTML content.
    """
    logger.info(
        "\n--- Relationship to Reference Person (API) ---"
    )  # Assuming WGG or similar is reference
    if isinstance(raw_content, dict) and "error" in raw_content:
        logger.error(f"Could not retrieve relationship data: {raw_content['error']}")
        return
    if not raw_content or not isinstance(raw_content, str):
        logger.error("No relationship content available or invalid format.")
        return

    html_escaped = None
    # More robust regex including single quotes for keys/values
    html_match = re.search(
        r'["\']html["\']\s*:\s*["\']((?:\\.|[^"\\])*)["\']',
        raw_content,
        re.IGNORECASE | re.DOTALL,
    )
    if html_match:
        html_escaped = html_match.group(1)
    else:  # Fallback
        # Look for 'html": "' or 'html":"' or 'html': ' etc.
        match = re.search(r'["\']html["\']\s*:\s*["\']', raw_content)
        if match:
            html_start = match.end()
            # Find the corresponding ending quote, careful about escaped quotes
            end_quote_index = -1
            current_index = html_start
            while current_index < len(raw_content):
                if raw_content[current_index] == '"':
                    if current_index == 0 or raw_content[current_index - 1] != "\\":
                        end_quote_index = current_index
                        break
                current_index += 1

            if end_quote_index != -1:
                html_escaped = raw_content[html_start:end_quote_index]
            else:
                logger.warning("Fallback HTML extraction failed to find end quote.")
        else:
            logger.warning("Fallback HTML extraction failed to find start pattern.")

    if not html_escaped:
        logger.error(
            "Could not extract relationship ladder HTML from the API response."
        )
        logger.debug(f"Raw content for failed HTML extraction: {raw_content[:500]}...")
        return

    html_unescaped = ""
    try:
        # Handle potential multiple layers of escaping carefully
        temp_unescaped = html_escaped.replace("\\\\", "\\")  # Handle \\u -> \u
        # Decode unicode escapes FIRST
        try:
            # Use 'raw_unicode_escape' if unicode escapes are like \uXXXX
            # Use 'unicode_escape' if they are like \\uXXXX (after replacing \\ above)
            html_intermediate = temp_unescaped.encode(
                "latin-1", "backslashreplace"
            ).decode("unicode_escape")
        except UnicodeDecodeError:
            logger.warning("Unicode decoding failed, trying latin-1 passthrough.")
            html_intermediate = (
                temp_unescaped  # Fallback: proceed without unicode decode
            )

        # Decode HTML entities AFTER unicode escapes
        html_unescaped = html.unescape(html_intermediate)

    except Exception as decode_err:
        logger.error(
            f"Could not decode relationship ladder HTML. Error: {decode_err}",
            exc_info=True,
        )
        logger.debug(f"Problematic escaped HTML snippet: {html_escaped[:500]}...")
        return

    try:
        soup = BeautifulSoup(html_unescaped, "html.parser")
        actual_relationship = "(not found)"
        # Combined and refined selectors
        rel_elem = soup.select_one(
            "ul.textCenter > li:first-child > i > b, "  # Original primary
            "ul > li > i > b, "  # Original secondary
            "div.relationshipText > span, "  # Observed structure 1
            "span.relationshipText, "  # Observed structure 2
            "p.relationshipText"  # Observed structure 3 (e.g., in <p>)
        )

        if rel_elem:
            actual_relationship = ordinal_case(rel_elem.get_text(strip=True))
        else:
            logger.warning(
                "Could not extract actual_relationship element using primary selectors."
            )
            # Fallback: Find any text that looks like a relationship term if primary fails
            possible_direct_rel = soup.find(
                string=re.compile(
                    r"^(Mother|Father|Son|Daughter|Spouse|Brother|Sister|Cousin|Aunt|Uncle|Grand\w+|Great-Grand\w+)$",
                    re.IGNORECASE,
                )
            )
            if possible_direct_rel:
                actual_relationship = ordinal_case(possible_direct_rel.strip())
                logger.info(
                    f"Found possible relationship text fallback: {actual_relationship}"
                )

        logger.info(f"Relationship: {actual_relationship}")

        path_list = []
        # Combined and refined selectors for path items
        path_items = soup.select(
            'ul.textCenter > li:not([class*="iconArrowDown"]), '  # Original primary
            "ul#relationshipPath > li, "  # Observed structure 1
            "div.relationshipPath li, "  # Observed structure 2
            ".rel-path li"  # Simplified class name
        )

        if not path_items:
            logger.warning(
                "Could not find any relationship path items using selectors."
            )
        else:
            for i, item in enumerate(path_items):
                # Skip items that are clearly just arrows or dividers
                if item.find(class_=re.compile("iconArrow|divider")):
                    continue

                name_text, desc_text = "", ""
                # Find name container more broadly
                name_container = (
                    item.find("a")
                    or item.find("b")
                    or item.find("strong")
                    or item.find(class_=re.compile("name", re.I))
                )
                if name_container:
                    name_text = format_name(
                        name_container.get_text(strip=True).replace('"', "'")
                    )  # Use imported format_name

                # Find description container more broadly
                desc_element = (
                    item.find("i")
                    or item.find("span", class_=re.compile("relationship|desc", re.I))
                    or item.find(class_=re.compile("relationship|desc", re.I))
                )
                if desc_element:
                    raw_desc_full = desc_element.get_text(strip=True).replace('"', "'")
                    # Check if desc text is just the name again (case-insensitive)
                    if format_name(raw_desc_full).lower() != name_text.lower():
                        desc_text = ordinal_case(
                            raw_desc_full
                        )  # Format relationship descriptors

                # Combine Name and Description for display
                display_text = name_text
                if desc_text:  # Add description if found and different from name
                    display_text += f" ({desc_text})"

                # Basic check to avoid adding empty items
                if display_text:
                    path_list.append(display_text)
                elif (
                    name_text
                ):  # Fallback: Use only name if description fails but name found
                    path_list.append(name_text)
                else:
                    logger.warning(
                        f"Path item {i}: Skipping path item because display_text is empty. Raw item: {item.get_text(strip=True)[:50]}..."
                    )

        if path_list:
            logger.info("Path:")
            # Display logic assumes path goes from target person up/across to WGG (you)
            # If path looks reversed, adjust this logic
            logger.info(f"  {path_list[0]} (Target Person)")
            for i, p in enumerate(path_list[1:-1]):
                logger.info("  ↓")
                logger.info(f"  {p}")
            if len(path_list) > 1:
                logger.info("  ↓")
                logger.info(
                    f"  {path_list[-1]} (You/Reference)"
                )  # Last item is usually the root/user
        else:
            logger.warning("No relationship path found in HTML.")
            logger.debug(
                f"Parsed HTML content that yielded no path: {html_unescaped[:500]}..."
            )

    except Exception as bs_parse_err:
        logger.error(
            f"Error parsing relationship ladder HTML with BeautifulSoup: {bs_parse_err}",
            exc_info=True,
        )
        logger.debug(f"Problematic unescaped HTML: {html_unescaped[:500]}...")


if __name__ == "__main__":
    print("This is the api_utils module. Import it into other scripts.")
    logger.info("api_utils loaded.")
    if API_UTILS_AVAILABLE:
        logger.info("Underlying 'utils' module (SessionManager) seems available.")
    else:
        logger.error(
            "Underlying 'utils' module (SessionManager) is NOT available. API functions will fail."
        )
