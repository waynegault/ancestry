# gedcom_interrogator.py (v7.36 - API Auto-Select Attempt)
"""
Standalone script to load and query a GEDCOM file or Ancestry API.
Provides two main actions:
1. GEDCOM Report: Fuzzy search local GEDCOM, display details, full family, relationship to WGG.
2. API Report: Fuzzy search Ancestry API (name-based), attempts auto-select, displays details, basic family (via /facts API), relationship to WGG.
Handles potential TypeError during record reads gracefully. Uses fuzzy match for user searches.
Integrated with Ancestry API search functionality for extended data access.
"""

import logging
import sys
import re
import traceback
import os
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple, Set, Deque, Callable, Union
from collections import deque
from datetime import datetime, timezone
import difflib  # For partial name matching
import time
import json  # Added for API response parsing
import requests
import urllib.parse
import html  # Added for unescaping
from bs4 import BeautifulSoup  # Added for HTML parsing

# Add parent directory to sys.path to import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules

# Global cache for family relationships and individual lookup
FAMILY_MAPS_CACHE = None
FAMILY_MAPS_BUILD_TIME = 0
INDI_INDEX = {}  # Index for ID -> Individual object
INDI_INDEX_BUILD_TIME = 0  # Track build time


# --- Add function to build the individual index ---
def build_indi_index(reader):
    """Builds a dictionary mapping normalized ID to Individual object."""
    global INDI_INDEX, INDI_INDEX_BUILD_TIME
    if INDI_INDEX:  # Avoid rebuilding
        return  # End of function build_indi_index
    start_time = time.time()
    logger.info("[Cache] Building INDI index...")
    count = 0
    for indi in reader.records0("INDI"):
        if _is_individual(indi) and indi.xref_id:
            norm_id = _normalize_id(indi.xref_id)
            if norm_id:
                INDI_INDEX[norm_id] = indi
                count += 1
    elapsed = time.time() - start_time
    INDI_INDEX_BUILD_TIME = elapsed
    logger.info(f"[Cache] INDI index built with {count} individuals in {elapsed:.2f}s.")


# End of function build_indi_index


# --- Function to build family relationship maps ---
def build_family_maps(reader):
    """Builds id_to_parents and id_to_children maps for all individuals."""
    start_time = time.time()
    id_to_parents = {}
    id_to_children = {}
    fam_count = 0
    indi_count = 0
    # Iterate through family records
    for fam in reader.records0("FAM"):
        fam_count += 1
        if not _is_record(fam):
            continue
        # Get husband and wife IDs
        husband = fam.sub_tag("HUSB")
        wife = fam.sub_tag("WIFE")
        parents = set()
        if _is_individual(husband) and husband.xref_id:
            parent_id_h = _normalize_id(husband.xref_id)
            if parent_id_h:
                parents.add(parent_id_h)
        if _is_individual(wife) and wife.xref_id:
            parent_id_w = _normalize_id(wife.xref_id)
            if parent_id_w:
                parents.add(parent_id_w)
        # Process children
        for child in fam.sub_tags("CHIL"):
            if _is_individual(child) and child.xref_id:
                child_id = _normalize_id(child.xref_id)
                if child_id:
                    # Map child to parents
                    id_to_parents.setdefault(child_id, set()).update(parents)
                    # Map parents to child
                    for parent_id in parents:
                        if parent_id:  # Ensure parent ID is valid
                            id_to_children.setdefault(parent_id, set()).add(child_id)
    # Count individuals (optional, for logging context)
    for indi in reader.records0("INDI"):
        indi_count += 1
    elapsed = time.time() - start_time
    logger.debug(
        f"[PROFILE] Family maps built: {fam_count} FAMs, {indi_count} INDI, {len(id_to_parents)} child->parents, {len(id_to_children)} parent->children in {elapsed:.2f}s"
    )
    global FAMILY_MAPS_BUILD_TIME
    FAMILY_MAPS_BUILD_TIME = elapsed
    return id_to_parents, id_to_children


# End of function build_family_maps


# --- Top-level helpers for ID extraction and lookup (used by all menu actions) ---
def extract_and_fix_id(raw_id):
    """
    Cleans and validates a raw ID string (e.g., '@I123@', 'F45').
    Returns the normalized ID (e.g., 'I123', 'F45') or None if invalid.
    """
    if not raw_id or not isinstance(raw_id, str):
        return None
    # Strip leading/trailing whitespace and '@' symbols, convert to uppercase
    id_clean = raw_id.strip().strip("@").upper()
    # Regex 1: Match standard GEDCOM IDs (I, F, S, T, N, M, C, X, O followed by numbers/letters/hyphens)
    m = re.match(r"^([IFSTNMCXO][0-9A-Z\-]+)$", id_clean)
    if m:
        return m.group(1)
    # Regex 2 (Fallback): Find first occurrence of a standard prefix followed by digits
    m2 = re.search(r"([IFSTNMCXO][0-9]+)", id_clean)
    if m2:
        logger.debug(
            f"extract_and_fix_id: Used fallback regex for '{raw_id}' -> '{m2.group(1)}'"
        )
        return m2.group(1)
    # If no match, return None
    logger.warning(f"extract_and_fix_id: Could not extract valid ID from '{raw_id}'")
    return None


# End of function extract_and_fix_id


def find_individual_by_id(reader, norm_id):
    """Finds an individual by normalized ID using the pre-built index."""
    global INDI_INDEX
    if not norm_id:
        logger.warning("find_individual_by_id called with invalid norm_id: None")
        return None
    if not INDI_INDEX:
        # Fallback if index isn't built (shouldn't happen in normal flow)
        logger.warning("INDI_INDEX not built, falling back to linear scan.")
        for indi in reader.records0("INDI"):
            if _is_individual(indi) and hasattr(indi, "xref_id"):
                current_norm_id = _normalize_id(indi.xref_id)
                if current_norm_id == norm_id:
                    return indi
        logger.warning(
            f"Individual with normalized ID {norm_id} not found via linear scan."
        )
        return None
    # Use the pre-built index for O(1) lookup
    found_indi = INDI_INDEX.get(norm_id)
    if not found_indi:
        logger.debug(
            f"Individual with normalized ID {norm_id} not found in INDI_INDEX."
        )
    return found_indi


# End of function find_individual_by_id

# --- Third-party Imports ---
# Setup logging handlers
log_file_handler = logging.FileHandler(
    "gedcom_processor.log", mode="a", encoding="utf-8"
)
log_stream_handler = logging.StreamHandler(sys.stderr)
# Configure root logger
logging.basicConfig(
    level=logging.INFO,  # Default level
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[log_file_handler, log_stream_handler],
)
logger = logging.getLogger(__name__)  # Get logger for this module

try:
    from ged4py.parser import GedcomReader
    from ged4py.model import Individual, Record, Name  # Use Name type

    GEDCOM_LIB_AVAILABLE = True
except ImportError:
    logger.error("`ged4py` library not found.")
    logger.error("Please install it: pip install ged4py")
    GedcomReader = None
    Individual = None
    Record = None
    Name = None
    GEDCOM_LIB_AVAILABLE = False  # type: ignore
except Exception as import_err:
    logger.error(f"ERROR importing ged4py: {type(import_err).__name__} - {import_err}")
    logger.error("!!!", exc_info=True)
    GedcomReader = None
    Individual = None
    Record = None
    Name = None
    GEDCOM_LIB_AVAILABLE = False  # type: ignore

# --- Local Application Imports ---
# Setup logging handlers
log_file_handler = logging.FileHandler(
    "gedcom_processor.log", mode="a", encoding="utf-8"
)
log_stream_handler = logging.StreamHandler(sys.stderr)
# Configure root logger
logging.basicConfig(
    level=logging.INFO,  # Default level
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[log_file_handler, log_stream_handler],
)
logger = logging.getLogger(__name__)  # Get logger for this module

try:
    # Attempt to import SessionManager and _api_req from utils
    from utils import (
        SessionManager as UtilsSessionManager,
        _api_req,
    )
except ImportError as e_utils:
    logger.warning(f"Could not import from 'utils' module: {e_utils}")
    logger.warning("API functionality will be disabled.")

    # Define dummy classes/functions if utils is missing, to avoid NameErrors later
    class SessionManager:
        def __init__(self):
            self.driver_live = False
            self.session_ready = False
            self.my_tree_id = None
            self.my_profile_id = None
            self.driver = None

        def ensure_driver_live(self):
            pass

        def ensure_session_ready(self):
            return False

        def check_session_status(self):
            pass

        def _retrieve_identifiers(self):
            pass

    def _api_req(*args, **kwargs):
        logger.error("API request attempted but 'utils' module is missing.")
        return {"error": "'utils' module unavailable"}

    # Set API_AVAILABLE flag or similar if needed later
    API_UTILS_AVAILABLE = False
else:
    API_UTILS_AVAILABLE = True

try:
    from config import config_instance

    # Setup logging handlers
    log_file_handler = logging.FileHandler(
        "gedcom_processor.log", mode="a", encoding="utf-8"
    )
    log_stream_handler = logging.StreamHandler(sys.stderr)
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,  # Default level
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[log_file_handler, log_stream_handler],
    )
    logger = logging.getLogger(__name__)  # Get logger for this module
except ImportError as e_config:
    logger.error(f"Failed to import local config module: {e_config}")
    logger.error("Ensure config.py exists.")
    # Fallback logging configuration if config fails
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("gedcom_processor_fallback")

    # Dummy config object to prevent crashes
    class DummyConfig:
        GEDCOM_FILE_PATH = None

    config_instance = DummyConfig()
    logger.warning("Using fallback logger and dummy config.")

# --- Helper Functions ---


def _is_individual(obj):
    """Checks if object is an Individual safely handling None values"""
    # Check if the object is not None and if its type name is 'Individual'
    return obj is not None and type(obj).__name__ == "Individual"


# End of function _is_individual


def _is_record(obj):
    """Checks if object is a Record safely handling None values"""
    # Check if the object is not None and if its type name is 'Record'
    return obj is not None and type(obj).__name__ == "Record"


# End of function _is_record


def _is_name(obj):
    """Checks if object is a Name safely handling None values"""
    # Check if the object is not None and if its type name is 'Name'
    return obj is not None and type(obj).__name__ == "Name"


# End of function _is_name


def _normalize_id(xref_id: Optional[str]) -> Optional[str]:
    """Normalizes INDI/FAM etc IDs (e.g., '@I123@' -> 'I123')."""
    if xref_id and isinstance(xref_id, str):
        # Match standard GEDCOM ID format (leading char + digits/chars/hyphen), allowing optional @ symbols
        match = re.match(r"^@?([IFSTNMCXO][0-9A-Z\-]+)@?$", xref_id.strip().upper())
        if match:
            return match.group(1)
    # Return None if input is invalid or doesn't match format
    return None


# End of function _normalize_id


def _get_full_name(indi) -> str:
    """Safely gets formatted name using Name.format(). Handles None/errors."""
    if not _is_individual(indi):
        return "Unknown (Not Individual)"
    try:
        name_rec = indi.name  # Access the name record
        if _is_name(name_rec):
            # Use the ged4py Name object's format method
            formatted_name = name_rec.format()
            # Clean up extra spaces and apply title case
            cleaned_name = " ".join(formatted_name.split()).title()
            # Remove trailing GEDCOM surname slashes (e.g., /Smith/)
            cleaned_name = re.sub(r"\s*/([^/]+)/\s*$", r" \1", cleaned_name).strip()
            # Return cleaned name or a placeholder if empty
            return cleaned_name if cleaned_name else "Unknown (Empty Name)"
        elif name_rec is None:
            return "Unknown (No Name Tag)"
        else:
            # Log warning if .name attribute is not a Name object
            indi_id_log = _normalize_id(indi.xref_id) if indi.xref_id else "Unknown ID"
            logger.warning(
                f"Indi @{indi_id_log}@ unexpected .name type: {type(name_rec)}"
            )
            return f"Unknown (Type {type(name_rec).__name__})"
    except AttributeError:
        # Handle cases where .name attribute might be missing
        return "Unknown (Attr Error)"
    except Exception as e:
        # Catch any other unexpected errors during name formatting
        indi_id_log = _normalize_id(indi.xref_id) if indi.xref_id else "Unknown ID"
        logger.error(
            f"Error formatting name for @{indi_id_log}@: {e}",
            exc_info=False,  # Set to True for full traceback if needed
        )
        return "Unknown (Error)"


# End of function _get_full_name


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Attempts to parse various GEDCOM date string formats into datetime objects."""
    if not date_str or not isinstance(date_str, str):
        return None
    original_date_str = date_str
    # Preprocessing: uppercase, remove qualifiers, ranges, slashes, parentheses
    date_str = date_str.strip().upper()
    clean_date_str = re.sub(r"^(ABT|EST|BEF|AFT|BET|FROM|TO)\s+", "", date_str).strip()
    clean_date_str = re.sub(
        r"(\s+(AND|&)\s+\d{4}.*|\s+TO\s+\d{4}.*)", "", clean_date_str
    ).strip()  # Remove date ranges
    clean_date_str = re.sub(
        r"/\d*$", "", clean_date_str
    ).strip()  # Remove trailing slashes
    clean_date_str = re.sub(
        r"\s*\(.*\)\s*", "", clean_date_str
    ).strip()  # Remove content in parentheses
    # Define supported date formats
    formats = ["%d %b %Y", "%d %B %Y", "%b %Y", "%B %Y", "%Y"]
    # Attempt parsing with each format
    for fmt in formats:
        try:
            # Special check for year-only format to avoid parsing non-digits
            if fmt == "%Y" and not clean_date_str.isdigit():
                continue
            dt = datetime.strptime(clean_date_str, fmt)
            # Return date as UTC timezone-aware object
            return dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            # Ignore parsing errors for this format and try the next
            continue
        except Exception as e:
            # Log unexpected errors during parsing attempt
            logger.debug(
                f"Date parsing err for '{original_date_str}' (clean:'{clean_date_str}', fmt:'{fmt}'): {e}"
            )
            continue
    # Return None if no format matched
    return None


# End of function _parse_date


def _clean_display_date(raw_date_str: str) -> str:
    """Removes surrounding brackets if date exists, handles empty brackets."""
    if raw_date_str == "N/A":
        return raw_date_str
    cleaned = raw_date_str.strip()
    # Remove surrounding parentheses only if they enclose the entire string
    cleaned = re.sub(r"^\((.+)\)$", r"\1", cleaned).strip()
    # Return "N/A" if the string is empty after cleaning
    return cleaned if cleaned else "N/A"


# End of function _clean_display_date


def _get_event_info(individual, event_tag: str) -> Tuple[Optional[datetime], str, str]:
    """Gets date/place for an event using tag.value. Handles non-string dates."""
    date_obj: Optional[datetime] = None
    date_str: str = "N/A"
    place_str: str = "N/A"
    indi_id_log = "Invalid/Unknown"
    # Validate input individual object
    if _is_individual(individual) and individual.xref_id:
        indi_id_log = (
            _normalize_id(individual.xref_id) or f"Unnormalized({individual.xref_id})"
        )
    else:
        logger.warning(f"_get_event_info invalid input: type {type(individual)}")
        return date_obj, date_str, place_str
    try:
        # Find the event record (e.g., BIRT, DEAT)
        event_record = individual.sub_tag(event_tag)
        if event_record:
            # Get Date sub-tag
            date_tag = event_record.sub_tag("DATE")
            if date_tag and hasattr(date_tag, "value"):
                raw_date_val = date_tag.value
                # Process date value based on its type
                if isinstance(raw_date_val, str):
                    processed_date_str = raw_date_val.strip()
                    date_str = processed_date_str if processed_date_str else "N/A"
                    date_obj = _parse_date(date_str)  # Attempt to parse into datetime
                elif raw_date_val is not None:  # Handle non-string but non-None values
                    date_str = str(raw_date_val)
                    date_obj = _parse_date(date_str)
            # Get Place sub-tag
            place_tag = event_record.sub_tag("PLAC")
            if place_tag and hasattr(place_tag, "value"):
                raw_place_val = place_tag.value
                # Process place value
                if isinstance(raw_place_val, str):
                    processed_place_str = raw_place_val.strip()
                    place_str = processed_place_str if processed_place_str else "N/A"
                elif raw_place_val is not None:
                    place_str = str(raw_place_val)
    except AttributeError:
        # Ignore errors if tags/attributes are missing
        pass
    except Exception as e:
        # Log unexpected errors
        logger.error(
            f"Unexpected error accessing event {event_tag} for @{indi_id_log}@: {e}",
            exc_info=True,
        )
    return date_obj, date_str, place_str


# End of function _get_event_info


# --- New helper functions for handling events and formatting ---
def format_life_dates(indi) -> str:
    """Returns a formatted string with birth and death dates."""
    # Get birth and death info
    b_date_obj, b_date_str, b_place = _get_event_info(indi, "BIRT")
    d_date_obj, d_date_str, d_place = _get_event_info(indi, "DEAT")
    # Clean the display strings (remove parentheses etc.)
    b_date_str_cleaned = _clean_display_date(b_date_str)
    d_date_str_cleaned = _clean_display_date(d_date_str)
    # Format birth and death parts
    birth_info = f"b. {b_date_str_cleaned}" if b_date_str_cleaned != "N/A" else ""
    death_info = f"d. {d_date_str_cleaned}" if d_date_str_cleaned != "N/A" else ""
    # Combine parts if they exist
    life_parts = [info for info in [birth_info, death_info] if info]
    # Return formatted string (e.g., "(b. 1900, d. 1980)") or empty string
    return f" ({', '.join(life_parts)})" if life_parts else ""


# End of function format_life_dates


def format_full_life_details(indi) -> Tuple[str, str]:
    """Returns formatted birth and death details (date and place) for display."""
    # Get birth event details
    b_date_obj, b_date_str, b_place = _get_event_info(indi, "BIRT")
    b_date_str_cleaned = _clean_display_date(b_date_str)
    # Format birth string
    birth_info = (
        f"Born: {b_date_str_cleaned if b_date_str_cleaned != 'N/A' else '(Date unknown)'} "
        f"in {b_place if b_place != 'N/A' else '(Place unknown)'}"
    )
    # Get death event details
    d_date_obj, d_date_str, d_place = _get_event_info(indi, "DEAT")
    d_date_str_cleaned = _clean_display_date(d_date_str)
    death_info = ""
    # Format death string only if death date exists
    if d_date_str_cleaned != "N/A":
        death_info = (
            f"   Died: {d_date_str_cleaned} "
            f"in {d_place if d_place != 'N/A' else '(Place unknown)'}"
        )
    return birth_info, death_info


# End of function format_full_life_details


def format_relative_info(relative) -> str:
    """Formats information about a relative (name and life dates) for display."""
    if not _is_individual(relative):
        return "  - (Invalid Relative Data)"
    # Get relative's full name
    rel_name = _get_full_name(relative)
    # Get formatted life dates (b. date, d. date)
    life_info = format_life_dates(relative)
    # Combine into a display string
    return f"  - {rel_name}{life_info}"


# End of function format_relative_info

# --- Core Data Retrieval Functions ---


def _find_family_records_where_individual_is_child(reader, target_id):
    """Helper function to find family records where an individual is listed as a child."""
    parent_families = []
    # Iterate through all FAM records
    for family_record in reader.records0("FAM"):
        if not _is_record(family_record):
            continue
        # Check children in the current family
        children_in_fam = family_record.sub_tags("CHIL")
        if children_in_fam:
            for child in children_in_fam:
                # If child matches target ID, add the family record and break inner loop
                if _is_individual(child) and _normalize_id(child.xref_id) == target_id:
                    parent_families.append(family_record)
                    break
    return parent_families


# End of function _find_family_records_where_individual_is_child


def _find_family_records_where_individual_is_parent(reader, target_id):
    """Helper function to find family records where an individual is listed as a parent (HUSB or WIFE)."""
    parent_families = []
    # Iterate through all FAM records
    for family_record in reader.records0("FAM"):
        if not _is_record(family_record) or not family_record.xref_id:
            continue
        # Get husband and wife records
        husband = family_record.sub_tag("HUSB")
        wife = family_record.sub_tag("WIFE")
        # Check if target ID matches either parent
        is_target_husband = (
            _is_individual(husband) and _normalize_id(husband.xref_id) == target_id
        )
        is_target_wife = (
            _is_individual(wife) and _normalize_id(wife.xref_id) == target_id
        )
        # If target is a parent in this family, add the record along with role flags
        if is_target_husband or is_target_wife:
            parent_families.append((family_record, is_target_husband, is_target_wife))
    return parent_families


# End of function _find_family_records_where_individual_is_parent


def get_related_individuals(reader, individual, relationship_type: str) -> List:
    """Gets parents, spouses, children, or siblings using family record lookups."""
    related_individuals: List = []
    unique_related_ids: Set[str] = set()

    # Validate inputs
    if not reader:
        logger.error("get_related_individuals: No reader.")
        return related_individuals
    if not _is_individual(individual) or not individual.xref_id:
        logger.warning(f"get_related_individuals: Invalid input individual.")
        return related_individuals

    # Normalize the target individual's ID
    target_id = _normalize_id(individual.xref_id)
    if not target_id:
        logger.warning(
            f"get_related_individuals: Cannot normalize target ID {individual.xref_id}"
        )
        return related_individuals
    # target_name = _get_full_name(individual) # For logging if needed

    try:
        if relationship_type == "parents":
            logger.debug(f"Finding parents for {target_id}...")
            # Find families where the target is a child
            parent_families = _find_family_records_where_individual_is_child(
                reader, target_id
            )
            potential_parents = []
            # Extract parents (HUSB, WIFE) from those families
            for family_record in parent_families:
                husband = family_record.sub_tag("HUSB")
                wife = family_record.sub_tag("WIFE")
                if _is_individual(husband):
                    potential_parents.append(husband)
                if _is_individual(wife):
                    potential_parents.append(wife)
            # Add unique parents to the result list
            for parent in potential_parents:
                if parent is not None and hasattr(parent, "xref_id") and parent.xref_id:
                    parent_id = _normalize_id(parent.xref_id)
                    if parent_id and parent_id not in unique_related_ids:
                        related_individuals.append(parent)
                        unique_related_ids.add(parent_id)
            logger.debug(
                f"Added {len(unique_related_ids)} unique parents for {target_id}."
            )

        elif relationship_type == "siblings":
            logger.debug(f"Finding siblings for {target_id}...")
            # Find families where the target is a child
            parent_families = _find_family_records_where_individual_is_child(
                reader, target_id
            )
            potential_siblings = []
            # Collect all children from those families
            for fam in parent_families:
                fam_children = fam.sub_tags("CHIL")
                if fam_children:
                    potential_siblings.extend(
                        c for c in fam_children if _is_individual(c)
                    )
            # Add unique siblings (excluding the target) to the result list
            for sibling in potential_siblings:
                if (
                    sibling is not None
                    and hasattr(sibling, "xref_id")
                    and sibling.xref_id
                ):
                    sibling_id = _normalize_id(sibling.xref_id)
                    if (
                        sibling_id
                        and sibling_id not in unique_related_ids
                        and sibling_id != target_id
                    ):
                        related_individuals.append(sibling)
                        unique_related_ids.add(sibling_id)
            logger.debug(
                f"Added {len(unique_related_ids)} unique siblings for {target_id}."
            )

        elif relationship_type in ["spouses", "children"]:
            # Find families where the target is a parent
            parent_families = _find_family_records_where_individual_is_parent(
                reader, target_id
            )
            if relationship_type == "spouses":
                logger.debug(f"Finding spouses for {target_id}...")
                # Extract the *other* parent from each family
                for family_record, is_target_husband, is_target_wife in parent_families:
                    other_spouse = None
                    if is_target_husband:
                        other_spouse = family_record.sub_tag("WIFE")
                    elif is_target_wife:
                        other_spouse = family_record.sub_tag("HUSB")
                    # Add unique spouses to the result list
                    if (
                        other_spouse is not None
                        and _is_individual(other_spouse)
                        and hasattr(other_spouse, "xref_id")
                        and other_spouse.xref_id
                    ):
                        spouse_id = _normalize_id(other_spouse.xref_id)
                        if spouse_id and spouse_id not in unique_related_ids:
                            related_individuals.append(other_spouse)
                            unique_related_ids.add(spouse_id)
                logger.debug(
                    f"Added {len(unique_related_ids)} unique spouses for {target_id}."
                )
            else:  # relationship_type == "children"
                logger.debug(f"Finding children for {target_id}...")
                # Extract all children from each family where target is a parent
                for family_record, _, _ in parent_families:
                    children_list = family_record.sub_tags("CHIL")
                    if children_list:
                        for child in children_list:
                            # Add unique children to the result list
                            if (
                                child is not None
                                and _is_individual(child)
                                and hasattr(child, "xref_id")
                                and child.xref_id
                            ):
                                child_id = _normalize_id(child.xref_id)
                                if child_id and child_id not in unique_related_ids:
                                    related_individuals.append(child)
                                    unique_related_ids.add(child_id)
                logger.debug(
                    f"Added {len(unique_related_ids)} unique children for {target_id}."
                )
        else:
            logger.warning(
                f"Unknown relationship type requested: '{relationship_type}'"
            )

    # Catch potential errors during record access
    except AttributeError as ae:
        logger.error(
            f"AttributeError finding {relationship_type} for {target_id}: {ae}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"Unexpected error finding {relationship_type} for {target_id}: {e}",
            exc_info=True,
        )

    # Sort the results by ID for consistent display ordering
    related_individuals.sort(key=lambda x: (_normalize_id(x.xref_id) or ""))
    return related_individuals


# End of function get_related_individuals

# --- Relationship Path Functions ---


# --- Robust Ancestor Map (no read_record) ---
def _get_ancestors_map(reader, start_id_norm):
    """Builds a map of ancestor IDs to their depth relative to the start ID."""
    ancestors = {}
    queue = deque([(start_id_norm, 0)])  # Queue stores (ID, depth)
    visited = {start_id_norm}
    if not reader or not start_id_norm:
        logger.error("_get_ancestors_map: Invalid input.")
        return ancestors
    logger.debug(f"_get_ancestors_map: Starting for {start_id_norm}")
    processed_count = 0
    while queue:
        current_id, depth = queue.popleft()
        ancestors[current_id] = depth
        processed_count += 1
        # Find the individual using the index
        current_indi = find_individual_by_id(reader, current_id)
        if not current_indi:
            logger.warning(
                f"Ancestor map: Could not find individual for ID {current_id}"
            )
            continue
        # Get parents directly using ged4py attributes if available
        father = getattr(current_indi, "father", None)
        mother = getattr(current_indi, "mother", None)
        parents = [father, mother]
        # Add valid parents to the queue if not visited
        for parent_indi in parents:
            if (
                _is_individual(parent_indi)
                and hasattr(parent_indi, "xref_id")
                and parent_indi.xref_id
            ):
                parent_id = _normalize_id(parent_indi.xref_id)
                if parent_id and parent_id not in visited:
                    visited.add(parent_id)
                    queue.append((parent_id, depth + 1))
    logger.debug(
        f"_get_ancestors_map for {start_id_norm} finished. Processed {processed_count} individuals. Found {len(ancestors)} ancestors (incl. self)."
    )
    return ancestors


# End of function _get_ancestors_map


# --- Robust Path Builder (no read_record) ---
def _build_relationship_path_str(reader, start_id_norm, end_id_norm):
    """Builds shortest ancestral path string from start up to end using BFS."""
    if not reader or not start_id_norm or not end_id_norm:
        logger.error("_build_relationship_path_str: Invalid input.")
        return []
    logger.debug(
        f"_build_relationship_path_str: Building path {start_id_norm} -> {end_id_norm}"
    )
    queue = deque(
        [(start_id_norm, [])]
    )  # Queue stores (current_id, path_list_of_names)
    visited = {start_id_norm}
    processed_count = 0
    while queue:
        current_id, current_path_names = queue.popleft()
        processed_count += 1
        # Get individual and their name
        current_indi = find_individual_by_id(reader, current_id)
        current_name = (
            _get_full_name(current_indi)
            if current_indi
            else f"Unknown/Error ({current_id})"
        )
        # Build the path including the current person's name
        current_full_path = current_path_names + [current_name]
        # Check if target reached
        if current_id == end_id_norm:
            logger.debug(
                f"Path build: Reached target {end_id_norm} after checking {processed_count} nodes. Path found."
            )
            return current_full_path
        # Add parents to queue if not visited
        if current_indi:
            father = getattr(current_indi, "father", None)
            mother = getattr(current_indi, "mother", None)
            parents = [father, mother]
            for parent_indi in parents:
                if _is_individual(parent_indi) and parent_indi.xref_id:
                    parent_id = _normalize_id(parent_indi.xref_id)
                    if parent_id and parent_id not in visited:
                        visited.add(parent_id)
                        queue.append((parent_id, current_full_path))
    # Path not found
    logger.warning(
        f"_build_relationship_path_str: Could not find ANCESTRAL path from {start_id_norm} to {end_id_norm} after checking {processed_count} nodes."
    )
    return []  # Return empty list if no path found


# End of function _build_relationship_path_str


def _find_lca_from_maps(
    ancestors1: Dict[str, int], ancestors2: Dict[str, int]
) -> Optional[str]:
    """Finds Lowest Common Ancestor (LCA) ID from two ancestor maps."""
    if not ancestors1 or not ancestors2:
        return None
    # Find IDs present in both ancestor sets
    common_ancestor_ids = set(ancestors1.keys()) & set(ancestors2.keys())
    if not common_ancestor_ids:
        logger.debug("_find_lca_from_maps: No common ancestors found.")
        return None
    # Calculate combined depth (sum of depths from each start node) for common ancestors
    lca_candidates: Dict[str, Union[int, float]] = {
        cid: ancestors1.get(cid, float("inf")) + ancestors2.get(cid, float("inf"))
        for cid in common_ancestor_ids
    }
    if not lca_candidates:
        return None
    # Find the common ancestor ID with the minimum combined depth
    lca_id = min(lca_candidates.keys(), key=lambda k: lca_candidates[k])
    logger.debug(
        f"_find_lca_from_maps: LCA ID: {lca_id} (Depth Sum: {lca_candidates[lca_id]})"
    )
    return lca_id


# End of function _find_lca_from_maps


# --- Helper to reconstruct path from predecessor maps ---
def _reconstruct_path(start_id, end_id, meeting_id, visited_fwd, visited_bwd):
    """Reconstructs the path from start to end via the meeting point using predecessor maps."""
    path_fwd = []
    curr = meeting_id
    # Trace path back from meeting point to start node using forward predecessors
    while curr is not None:
        path_fwd.append(curr)
        curr = visited_fwd.get(curr)
    path_fwd.reverse()  # Reverse to get path from start to meeting point

    path_bwd = []
    curr = visited_bwd.get(
        meeting_id
    )  # Start from the predecessor of the meeting point in the backward search
    # Trace path back from meeting point's predecessor to end node using backward predecessors
    while curr is not None:
        path_bwd.append(curr)
        curr = visited_bwd.get(curr)

    # Combine the forward and backward paths
    path = path_fwd + path_bwd

    # --- Sanity Checks ---
    if not path:
        logger.error("_reconstruct_path: Failed to reconstruct any path.")
        return []
    if path[0] != start_id:
        logger.warning(
            f"_reconstruct_path: Path doesn't start with start_id ({path[0]} != {start_id}). Prepending."
        )
        path.insert(0, start_id)  # Attempt fix
    if path[-1] != end_id:
        logger.warning(
            f"_reconstruct_path: Path doesn't end with end_id ({path[-1]} != {end_id}). Appending."
        )
        path.append(end_id)  # Attempt fix

    logger.debug(f"_reconstruct_path: Final reconstructed path IDs: {path}")
    return path


# End of function _reconstruct_path


def explain_relationship_path(path_ids, reader, id_to_parents, id_to_children):
    """Return a human-readable explanation of the relationship path with relationship labels."""
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"
    steps = []
    # Iterate through pairs of consecutive IDs in the path
    for i in range(len(path_ids) - 1):
        id_a, id_b = path_ids[i], path_ids[i + 1]
        indi_a = find_individual_by_id(reader, id_a)
        indi_b = find_individual_by_id(reader, id_b)
        # Get names, handling potential lookup failures
        name_a = _get_full_name(indi_a) if indi_a else f"Unknown ({id_a})"
        name_b = _get_full_name(indi_b) if indi_b else f"Unknown ({id_b})"

        # Determine relationship using pre-built maps
        rel = "related"  # Default relationship
        label = rel  # Default label
        if id_b in id_to_parents.get(id_a, set()):
            # B is a parent of A
            rel = "child"  # A is child of B
            sex_a = getattr(indi_a, "sex", None) if indi_a else None
            label = "daughter" if sex_a == "F" else "son" if sex_a == "M" else "child"
        elif id_b in id_to_children.get(id_a, set()):
            # B is a child of A
            rel = "parent"  # A is parent of B
            sex_a = getattr(indi_a, "sex", None) if indi_a else None
            label = "mother" if sex_a == "F" else "father" if sex_a == "M" else "parent"
        # No need for reverse check if maps are correctly built

        # Format the step explanation: "Person A is the [relationship] of Person B"
        steps.append(f"{name_a} is the {label} of {name_b}")

    # Join steps with arrows for display
    start_person_name = (
        _get_full_name(find_individual_by_id(reader, path_ids[0]))
        or f"Unknown ({path_ids[0]})"
    )
    explanation_str = "\n -> ".join(steps)
    # Return format: "Start Person\n -> Step 1\n -> Step 2..."
    return f"{start_person_name}\n -> {explanation_str}"


# End of function explain_relationship_path


# --- Optimized bidirectional BFS using pre-built maps and predecessors ---
def fast_bidirectional_bfs(
    start_id,
    end_id,
    id_to_parents,
    id_to_children,
    max_depth=20,
    node_limit=100000,
    timeout_sec=30,
    log_progress=False,
):
    """Performs bidirectional BFS using maps & predecessors. Returns path as list of IDs."""
    start_time = time.time()
    from collections import deque

    # Initialize forward search queue and visited map
    queue_fwd = deque([(start_id, 0)])  # (id, depth)
    visited_fwd = {start_id: None}  # {id: predecessor_id}
    # Initialize backward search queue and visited map
    queue_bwd = deque([(end_id, 0)])
    visited_bwd = {end_id: None}
    processed = 0
    meeting_id = None  # ID where forward and backward searches meet

    # Main BFS loop
    while queue_fwd and queue_bwd and meeting_id is None:
        # --- Check limits ---
        if time.time() - start_time > timeout_sec:
            logger.warning(f"  [FastBiBFS] Timeout after {timeout_sec} seconds.")
            return []
        if processed > node_limit:
            logger.warning(f"  [FastBiBFS] Node limit {node_limit} reached.")
            return []

        # --- Expand Forward Search ---
        if queue_fwd:
            current_id_fwd, depth_fwd = queue_fwd.popleft()
            processed += 1
            if log_progress and processed % 5000 == 0:
                logger.info(
                    f"  [FastBiBFS] FWD processed {processed}, Q:{len(queue_fwd)}, D:{depth_fwd}"
                )
            # Skip if max depth reached
            if depth_fwd >= max_depth:
                continue
            # Get neighbors (parents and children)
            neighbors_fwd = id_to_parents.get(
                current_id_fwd, set()
            ) | id_to_children.get(current_id_fwd, set())
            for neighbor_id in neighbors_fwd:
                # Check if neighbor was visited by backward search (intersection found)
                if neighbor_id in visited_bwd:
                    meeting_id = neighbor_id
                    visited_fwd[neighbor_id] = current_id_fwd  # Record predecessor
                    logger.debug(
                        f"  [FastBiBFS] Path found (FWD meets BWD) at {meeting_id} after {processed} nodes."
                    )
                    break  # Exit inner neighbor loop
                # If neighbor not visited by forward search, add it
                if neighbor_id not in visited_fwd:
                    visited_fwd[neighbor_id] = current_id_fwd
                    queue_fwd.append((neighbor_id, depth_fwd + 1))
            if meeting_id:
                break  # Exit main while loop if intersection found

        # --- Expand Backward Search (only if no intersection yet) ---
        if queue_bwd and meeting_id is None:
            current_id_bwd, depth_bwd = queue_bwd.popleft()
            processed += 1
            if log_progress and processed % 5000 == 0:
                logger.debug(
                    f"  [FastBiBFS] BWD processed {processed}, Q:{len(queue_bwd)}, D:{depth_bwd}"
                )
            # Skip if max depth reached
            if depth_bwd >= max_depth:
                continue
            # Get neighbors (parents and children)
            neighbors_bwd = id_to_parents.get(
                current_id_bwd, set()
            ) | id_to_children.get(current_id_bwd, set())
            for neighbor_id in neighbors_bwd:
                # Check if neighbor was visited by forward search (intersection found)
                if neighbor_id in visited_fwd:
                    meeting_id = neighbor_id
                    visited_bwd[neighbor_id] = current_id_bwd  # Record predecessor
                    logger.debug(
                        f"  [FastBiBFS] Path found (BWD meets FWD) at {meeting_id} after {processed} nodes."
                    )
                    break  # Exit inner neighbor loop
                # If neighbor not visited by backward search, add it
                if neighbor_id not in visited_bwd:
                    visited_bwd[neighbor_id] = current_id_bwd
                    queue_bwd.append((neighbor_id, depth_bwd + 1))
            # No need to break outer loop here, FWD check handles it

    # --- Reconstruct Path if intersection found --- #
    if meeting_id:
        path_ids = _reconstruct_path(
            start_id, end_id, meeting_id, visited_fwd, visited_bwd
        )
        return path_ids
    else:
        logger.warning(
            f"  [FastBiBFS] No path found between {start_id} and {end_id} after {processed} nodes."
        )
        return []  # Return empty list if no path found


# End of function fast_bidirectional_bfs


# --- Enhanced get_relationship_path using FastBiBFS ---
def get_relationship_path(reader, id1: str, id2: str) -> str:
    """Calculates and formats relationship path using fast bidirectional BFS with pre-built maps."""
    # Normalize input IDs
    id1_norm = _normalize_id(id1)
    id2_norm = _normalize_id(id2)
    if not reader:
        return "Error: GEDCOM Reader unavailable."
    if not id1_norm or not id2_norm:
        return "Invalid input IDs."
    if id1_norm == id2_norm:
        return "Individuals are the same."

    # Ensure caches/maps are built
    global FAMILY_MAPS_CACHE, FAMILY_MAPS_BUILD_TIME, INDI_INDEX, INDI_INDEX_BUILD_TIME
    if FAMILY_MAPS_CACHE is None:
        logger.debug(f"  [Cache] Building family maps (first time)...")
        FAMILY_MAPS_CACHE = build_family_maps(reader)
        logger.debug(
            f"  [Cache] Maps built and cached in {FAMILY_MAPS_BUILD_TIME:.2f}s."
        )
    if not INDI_INDEX:
        logger.debug(f"  [Cache] Building individual index (first time)...")
        build_indi_index(reader)
        logger.debug(
            f"  [Cache] Index built and cached in {INDI_INDEX_BUILD_TIME:.2f}s."
        )

    id_to_parents, id_to_children = FAMILY_MAPS_CACHE
    # Default search parameters
    max_depth = 20
    node_limit = 100000
    timeout_sec = 30

    logger.debug(
        f"Calculating relationship path (FastBiBFS): {id1_norm} <-> {id2_norm}"
    )
    logger.debug(f"  [FastBiBFS] Using cached maps & index. Starting search...")
    search_start = time.time()

    # --- Perform BFS Search (Returns list of IDs) ---
    path_ids = fast_bidirectional_bfs(
        id1_norm,
        id2_norm,
        id_to_parents,
        id_to_children,
        max_depth,
        node_limit,
        timeout_sec,
        log_progress=False,
    )
    search_time = time.time() - search_start
    logger.debug(f"[PROFILE] BFS search completed in {search_time:.2f}s.")

    # Handle case where no path is found
    if not path_ids:
        profile_info = f"[PROFILE] Search: {search_time:.2f}s, Maps: {FAMILY_MAPS_BUILD_TIME:.2f}s, Index: {INDI_INDEX_BUILD_TIME:.2f}s"
        return (
            f"No relationship path found (FastBiBFS could not connect).\n{profile_info}"
        )

    # --- Explain the found path ---
    explanation_start = time.time()
    explanation_str = explain_relationship_path(
        path_ids, reader, id_to_parents, id_to_children
    )
    explanation_time = time.time() - explanation_start
    logger.debug(f"[PROFILE] Path explanation built in {explanation_time:.2f}s.")

    # --- Format final output ---
    profile_info = (
        f"[PROFILE] Total Time: {search_time+explanation_time:.2f}s "
        f"(Search: {search_time:.2f}s, Explain: {explanation_time:.2f}s, "
        f"Maps: {FAMILY_MAPS_BUILD_TIME:.2f}s, Index: {INDI_INDEX_BUILD_TIME:.2f}s)"
    )
    logger.debug(profile_info)

    return f"{explanation_str}\n"


# End of function get_relationship_path


# --- REVISED v7.34: Fuzzy scoring with death date/place ---
def find_potential_matches(
    reader,
    first_name: Optional[str],
    surname: Optional[str],
    dob_str: Optional[str],  # Birth date string
    pob: Optional[str],  # Birth place
    dod_str: Optional[str],  # Death date string (NEW)
    pod: Optional[str],  # Death place (NEW)
    gender: Optional[str] = None,
) -> List[Dict]:
    """
    Finds potential matches in GEDCOM based on various criteria including death info.
    Prioritizes name matches.
    """
    if not reader:
        logger.error("find_potential_matches: No reader.")
        return []
    results: List[Dict] = []
    max_results = 3  # Limit displayed results
    year_score_range = 1  # Allow +/- 1 year for date scoring bonus
    year_filter_range = (
        30  # Pre-filter: only consider individuals within 30 years of target year
    )

    # Clean input parameters - remove non-alphanumeric except spaces
    clean_param = lambda p: re.sub(r"[^\w\s]", "", p).strip() if p else None
    first_name_clean = clean_param(first_name)
    surname_clean = clean_param(surname)
    pob_clean = clean_param(pob)
    pod_clean = clean_param(pod)  # Clean death place
    gender_clean = (
        gender.strip().lower()
        if gender and gender.strip().lower() in ("m", "f")
        else None
    )

    logger.debug(
        f"Fuzzy Search: FirstName='{first_name_clean}', Surname='{surname_clean}', "
        f"DOB='{dob_str}', POB='{pob_clean}', DOD='{dod_str}', POD='{pod_clean}', Gender='{gender_clean}'"
    )

    # Prepare target values for comparison
    target_first_name_lower = first_name_clean.lower() if first_name_clean else None
    target_surname_lower = surname_clean.lower() if surname_clean else None
    target_pob_lower = pob_clean.lower() if pob_clean else None
    target_pod_lower = pod_clean.lower() if pod_clean else None  # Target death place

    # Parse target birth/death years
    target_birth_year: Optional[int] = None
    birth_dt = _parse_date(dob_str) if dob_str else None
    if birth_dt:
        target_birth_year = birth_dt.year

    target_death_year: Optional[int] = None  # NEW
    death_dt = _parse_date(dod_str) if dod_str else None
    if death_dt:
        target_death_year = death_dt.year

    # Check if any search criteria provided
    if not any(
        [
            target_first_name_lower,
            target_surname_lower,
            target_birth_year,
            target_pob_lower,
            target_death_year,
            target_pod_lower,
            gender_clean,
        ]
    ):
        logger.warning("Fuzzy search called with no valid criteria.")
        return []

    candidate_count = 0
    exact_matches = []
    fuzzy_results = []
    for indi in reader.records0("INDI"):
        candidate_count += 1
        if not _is_individual(indi) or not hasattr(indi, "xref_id") or not indi.xref_id:
            continue

        indi_id = _normalize_id(indi.xref_id)
        indi_full_name = _get_full_name(indi)
        if indi_full_name.startswith("Unknown"):
            continue  # Skip unknown names

        # Get event info for the individual
        birth_date_obj, birth_date_str_ged, birth_place_str_ged = _get_event_info(
            indi, "BIRT"
        )
        death_date_obj, death_date_str_ged, death_place_str_ged = _get_event_info(
            indi, "DEAT"
        )  # NEW: Get death info
        birth_year_ged: Optional[int] = birth_date_obj.year if birth_date_obj else None
        death_year_ged: Optional[int] = (
            death_date_obj.year if death_date_obj else None
        )  # NEW: Get death year

        # --- Pre-filtering based on years ---
        # If target birth year exists, check if candidate is within filter range
        if target_birth_year and birth_year_ged is not None:
            if abs(birth_year_ged - target_birth_year) > year_filter_range:
                continue
        # NEW: If target death year exists, check if candidate is within filter range
        elif target_death_year and death_year_ged is not None:
            if abs(death_year_ged - target_death_year) > year_filter_range:
                continue
        # If only one year is provided, this filters based on that. If both, checks birth first.

        # --- Scoring ---
        score = 0
        match_reasons = []
        # Extract first/last name from full name
        indi_name_lower = indi_full_name.lower()
        indi_name_parts = indi_name_lower.split()
        indi_first_name = indi_name_parts[0] if indi_name_parts else None
        indi_surname = indi_name_parts[-1] if len(indi_name_parts) > 1 else None

        # Flags for matching components
        first_name_match = False
        surname_match = False
        birth_year_match = False
        death_year_match = False  # NEW
        gender_match = False

        # 1. Name Scoring (Prioritize heavily)
        if (
            target_first_name_lower
            and indi_first_name
            and target_first_name_lower == indi_first_name
        ):
            first_name_match = True
        if (
            target_surname_lower
            and indi_surname
            and target_surname_lower == indi_surname
        ):
            surname_match = True
        # If both exact, add to exact_matches for later strict filtering
        if first_name_match and surname_match:
            # Score and reasons as before
            # ...existing code for scoring...
            score = 30  # Always start with 30 for exact name match
            match_reasons = ["First & Surname"]
            # ...existing code for year/date scoring...
            # Birth Year
            birth_year_match = False
            if (
                target_birth_year
                and birth_year_ged is not None
                and abs(birth_year_ged - target_birth_year) <= year_score_range
            ):
                score += 8
                birth_year_match = True
                match_reasons.append(
                    f"Birth Year ~{target_birth_year} ({birth_year_ged})"
                )
            # Exact birth date match (if both have a date)
            if birth_date_obj and birth_dt and birth_date_obj.date() == birth_dt.date():
                score += 5
                match_reasons.append("Exact Birth Date")
            # Death Year
            if (
                target_death_year
                and death_year_ged is not None
                and abs(death_year_ged - target_death_year) <= year_score_range
            ):
                score += 8
                match_reasons.append(
                    f"Death Year ~{target_death_year} ({death_year_ged})"
                )
            # Gender
            indi_gender = getattr(indi, "sex", None)
            gender_match = False
            if gender_clean and indi_gender:
                indi_gender_lower = str(indi_gender).strip().lower()
                if indi_gender_lower and indi_gender_lower[0] in ("m", "f"):
                    if indi_gender_lower[0] == gender_clean:
                        score += 5
                        gender_match = True
                        match_reasons.append(
                            f"Gender Match ({indi_gender_lower.upper()})"
                        )
                    else:
                        score -= 5
                        match_reasons.append(
                            f"Gender Mismatch ({indi_gender_lower.upper()} vs {gender_clean.upper()})"
                        )
            # ...existing code for place scoring...
            if target_pob_lower and birth_place_str_ged != "N/A":
                place_lower = birth_place_str_ged.lower()
                if place_lower.startswith(target_pob_lower):
                    score += 3
                    match_reasons.append(f"POB starts '{pob_clean}'")
                elif target_pob_lower in place_lower:
                    score += 1
                    match_reasons.append(f"POB contains '{pob_clean}'")
            if target_pod_lower and death_place_str_ged != "N/A":
                place_lower = death_place_str_ged.lower()
                if place_lower.startswith(target_pod_lower):
                    score += 3
                    match_reasons.append(f"POD starts '{pod_clean}'")
                elif target_pod_lower in place_lower:
                    score += 1
                    match_reasons.append(f"POD contains '{pod_clean}'")
            # ...existing code for boosts...
            if birth_year_match:
                score += 2  # Small boost for name+year
            # Add result
            reasons_str = ", ".join(sorted(list(set(match_reasons))))
            raw_indi_id_str = f"@{indi.xref_id}@" if indi.xref_id else None
            exact_matches.append(
                {
                    "id": raw_indi_id_str,
                    "name": indi_full_name,
                    "birth_date": _clean_display_date(birth_date_str_ged),
                    "birth_place": birth_place_str_ged,
                    "death_date": _clean_display_date(death_date_str_ged),
                    "death_place": death_place_str_ged,
                    "score": score,
                    "reasons": reasons_str or "Overall match",
                }
            )
            continue  # Do not add to fuzzy_results
        # ...existing code for fuzzy scoring (as before, but only if not exact)...
        # ...existing code for fuzzy scoring, append to fuzzy_results if score > 0...
        # ...existing code...
    # If any exact matches, only consider those, sorted by score, then birth year/date
    if exact_matches:
        # Sort by score, then by birth year (descending), then by birth date string
        exact_matches.sort(key=lambda x: (x["score"], x["birth_date"]), reverse=True)
        limited_results = exact_matches[:max_results]
        logger.debug(
            f"Fuzzy search scanned {candidate_count} individuals. Found {len(exact_matches)} exact name matches. Showing top {len(limited_results)}."
        )
        return limited_results
    # Otherwise, fall back to fuzzy scoring as before
    # ...existing code for fuzzy_results...
    results = fuzzy_results
    results.sort(key=lambda x: x["score"], reverse=True)
    limited_results = results[:max_results]
    logger.debug(
        f"Fuzzy search scanned {candidate_count} individuals. Found {len(results)} potential matches. Showing top {len(limited_results)}."
    )
    return limited_results


# End of function find_potential_matches

# --- Menu and Main Execution ---


# REVISED: Simplified Menu
def menu() -> str:
    """Displays the simplified interactive menu and returns the user's choice."""
    logger.info("\n--- GEDCOM Interrogator (v7.36) ---")
    logger.info("===================================")
    logger.info("  1. GEDCOM Report (Local File)")
    logger.info("  2. API Report (Ancestry Online)")
    logger.info("----------------------------------")
    logger.info(
        f"  t. Toggle Log Level (Current: {logging.getLevelName(logger.level)})"
    )
    logger.info("  q. Quit")
    logger.info("===================================")
    return input("Enter choice: ").strip().lower()


# End of function menu


# REVISED: display_gedcom_family_details (was display_family_details)
def display_gedcom_family_details(reader, individual):
    """Helper function to display formatted family details from GEDCOM data."""
    if not reader or not _is_individual(individual):
        logger.error("  Error: Cannot display GEDCOM details for invalid input.")
        return  # End of function display_gedcom_family_details

    indi_name = _get_full_name(individual)

    logger.info(f"Name: {indi_name}")
    birth_info, death_info = format_full_life_details(individual)
    logger.info(birth_info)
    if death_info:
        logger.info(death_info)

    # Display Parents
    logger.info("\n Parents:")
    parents = get_related_individuals(reader, individual, "parents")
    if parents:
        [logger.info(format_relative_info(p)) for p in parents]
    else:
        logger.info("  (None found)")

    # Display Siblings
    logger.info("\n Siblings:")
    siblings = get_related_individuals(reader, individual, "siblings")
    if siblings:
        [logger.info(format_relative_info(s)) for s in siblings]
    else:
        logger.info("  (None found)")

    # Display Spouses
    logger.info("\n Spouse(s):")
    spouses = get_related_individuals(reader, individual, "spouses")
    if spouses:
        [logger.info(format_relative_info(s)) for s in spouses]
    else:
        logger.info("  (None found)")

    # Display Children
    logger.info("\n Children:")
    children = get_related_individuals(reader, individual, "children")
    if children:
        [logger.info(format_relative_info(c)) for c in children]
    else:
        logger.info("  (None found)")


# End of function display_gedcom_family_details


# Create a standalone session_manager that can authenticate itself
# Place this near where API functions are defined or used
if API_UTILS_AVAILABLE:
    session_manager = UtilsSessionManager()
else:
    # If utils isn't available, create a dummy session_manager
    # so the code doesn't crash trying to access it.
    session_manager = SessionManager()  # Uses the dummy class defined earlier
    logger.warning("API Utils unavailable, using dummy SessionManager.")


# Add standalone authentication function
def initialize_session():
    """Initialize the session with proper authentication for standalone usage"""
    if not API_UTILS_AVAILABLE:
        logger.error("Cannot initialize session: API utilities are missing.")
        return False

    global session_manager
    # Ensure Selenium driver is running
    if not session_manager.driver_live:
        logger.info("Initializing browser session...")
        session_manager.ensure_driver_live()  # Starts driver if needed

    # Ensure authenticated session is ready
    if not session_manager.session_ready:
        success = session_manager.ensure_session_ready()  # Handles login/cookie loading
        if not success:
            logger.warning("Failed to authenticate with Ancestry automatically.")
            logger.warning("Please login manually when the browser opens.")
            input("Press Enter after you've logged in manually...")
            # Re-check session readiness after manual login attempt
            if hasattr(session_manager, "check_session_status") and callable(
                getattr(session_manager, "check_session_status", None)
            ):
                session_manager.check_session_status()
            if not session_manager.session_ready:
                logger.error("Session still not ready after manual login attempt.")
                return False  # Authentication failed
        else:
            logger.info("Authentication successful.")

    # Ensure tree_id is loaded (needed for some API calls)
    if not session_manager.my_tree_id:
        logger.info("Loading tree information...")
        session_manager._retrieve_identifiers()  # Fetches tree ID, user ID etc.
        if not session_manager.my_tree_id:
            # Log warning but allow proceeding as not all calls need tree ID
            logger.warning(
                "Could not load tree ID. Some API functionality might be limited."
            )
        else:
            logger.info(f"Tree ID loaded successfully: {session_manager.my_tree_id}")
    # Ensure profile ID is loaded (needed for /facts)
    if not session_manager.my_profile_id:
        logger.info("Loading user profile ID...")
        session_manager._retrieve_identifiers()
        if not session_manager.my_profile_id:
            logger.error("Could not load user profile ID. Facts API will fail.")
            # Optionally return False here if facts are critical
        else:
            logger.info(f"User Profile ID loaded: {session_manager.my_profile_id}")

    # Return True if session appears ready
    return session_manager.session_ready


# End of function initialize_session


class AncestryAPISearch:
    """Class to handle searching the Ancestry API for person information."""

    def __init__(self):
        """Initialize with a SessionManager instance."""
        global session_manager
        self.session_manager = session_manager
        self.base_url = "https://www.ancestry.co.uk"  # TODO: Make configurable?

    # End of function __init__

    def _get_tree_id(self):
        """Ensures tree_id is loaded in the session_manager and returns it."""
        if not self.session_manager.my_tree_id:
            self.session_manager._retrieve_identifiers()
        return self.session_manager.my_tree_id

    # End of function _get_tree_id

    def _extract_person_id(self, person: dict) -> Optional[str]:
        """Extracts the primary Person ID (CFPID) from various API response fields."""
        # Prefer 'id' (often the CFPID in list/search results)
        person_id = person.get("id")
        if person_id:
            return str(person_id)
        # Fallback to 'pid'
        person_id = person.get("pid")
        if person_id:
            return str(person_id)
        # Fallback to 'gid.v' if present
        gid = person.get("gid")
        if isinstance(gid, dict):
            person_id = gid.get("v")
            if person_id:
                return str(person_id)
        # Fallback to AssertionId if this is a fact target person (unlikely needed here)
        person_id = person.get("AssertionId")
        if person_id:
            return str(person_id)

        logger.warning(
            f"Could not extract Person ID (CFPID) from person data: {person}"
        )
        return None

    # End of function _extract_person_id

    def _extract_display_name(self, person: dict) -> str:
        """Tries various fields to get a display name from API person data."""
        # Try preferredName structure first
        preferred_name = person.get("preferredName")
        if preferred_name and isinstance(preferred_name, dict):
            given = preferred_name.get("givenName", "")
            surname = preferred_name.get("surname", "")
            full_name = f"{given} {surname}".strip()
            if full_name:
                return format_name(full_name)

        # Try Names list (often used in DNA match contexts)
        names = person.get("Names") or person.get("names")
        if names and isinstance(names, list) and names:
            given = names[0].get("g") or ""
            surname = names[0].get("s") or ""
            full_name = f"{given} {surname}".strip()
            if full_name:
                return format_name(full_name)

        # Fallbacks to direct fields
        given = person.get("gname") or person.get("givenName") or ""
        surname = person.get("sname") or person.get("surname") or ""
        if given or surname:
            return format_name(f"{given} {surname}".strip())

        # Final fallback to FullName or ID
        full_name_field = person.get("FullName")  # Common in facts API
        if full_name_field:
            return format_name(full_name_field)

        # Fallback to ID if no name found
        person_id = self._extract_person_id(person)
        return f"Unknown ({person_id})" if person_id else "(Unknown)"

    # End of function _extract_display_name

    def suggest_persons(self, first_name: str, last_name: str, limit: int = 10) -> list:
        import difflib

        if not API_UTILS_AVAILABLE:
            return [{"error": "API utilities missing"}]
        tree_id = self._get_tree_id()
        if not tree_id:
            logger.error("API Search: No tree_id available in session_manager.")
            return []
        base_url = self.base_url
        # Use fn and ln parameters for best results
        persons_url = (
            f"{base_url}/api/treesui-list/trees/{tree_id}/persons"
            f"?fn={urllib.parse.quote(first_name)}"
            f"&ln={urllib.parse.quote(last_name)}"
            f"&limit=50&fields=NAMES,EVENTS"
        )
        referer = f"{base_url}/family-tree/tree/{tree_id}/family"
        try:
            persons_response = _api_req(
                url=persons_url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                api_description="API Search by fn/ln",
                referer_url=referer,
                timeout=15,
            )
            if not isinstance(persons_response, list):
                logger.info("API /persons returned no results.")
                return []
            # Fuzzy match using difflib if needed
            search_name = f"{first_name} {last_name}".lower().strip()
            candidates = [
                (self._extract_display_name(p), p)
                for p in persons_response
                if isinstance(p, dict)
            ]
            names = [n.lower() for n, _ in candidates]
            close = difflib.get_close_matches(search_name, names, n=limit, cutoff=0.6)
            results = (
                [p for n, p in candidates if n.lower() in close]
                if close
                else [p for _, p in candidates][:limit]
            )
            if results:
                logger.info(
                    f"API search found {len(results)} close matches for '{search_name}'."
                )
            else:
                logger.info(f"API search found no close matches for '{search_name}'.")
            return results
        except Exception as e:
            logger.error(f"API /persons failed: {e}")
            return []

    # ...existing code...

    def get_person_details(self, person_id: str) -> dict:
        """Fetch full person details from /persons endpoint using PersonId."""
        if not API_UTILS_AVAILABLE:
            return {"error": "API utilities missing"}
        tree_id = self._get_tree_id()
        if not tree_id or not person_id:
            return {"error": "Missing tree_id or person_id for details fetch."}
        base_url = self.base_url
        # Use the persons endpoint with name param as fallback, but prefer direct lookup if available
        persons_url = f"{base_url}/api/treesui-list/trees/{tree_id}/persons?name=&tags=&page=1&limit=10&fields=EVENTS,GENDERS,KINSHIP,NAMES,PHOTO,RELATIONS,TAGS&isGetFullPersonObject=false"
        referer = f"{base_url}/family-tree/tree/{tree_id}/family"
        try:
            response = _api_req(
                url=persons_url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                api_description="API Person Details",
                referer_url=referer,
                timeout=30,
            )
            if not response or not isinstance(response, list):
                if isinstance(response, dict) and "error" in response:
                    logger.error(f"API Person Details failed: {response['error']}")
                    return {}
                logger.warning(
                    f"API Person Details: Invalid response format. Type: {type(response)}"
                )
                return {}
            # Find the person with the matching pid
            for person in response:
                if isinstance(person, dict) and str(person.get("pid")) == str(
                    person_id
                ):
                    return person
            logger.warning(
                f"API Person Details: No match for pid {person_id} in response."
            )
            return {}
        except Exception as e:
            logger.error(
                f"API Person Details: Error fetching details: {e}", exc_info=True
            )
            return {}

    def search_by_name(self, name: str, limit: int = 10) -> list[dict]:
        """Searches for people in the user's tree by name via API."""
        if not API_UTILS_AVAILABLE:
            return [{"error": "API utilities missing"}]

        tree_id = self._get_tree_id()
        if not tree_id:
            logger.error("API Search: No tree_id available in session_manager.")
            return []
        base_url = self.base_url
        query = name.strip()
        tags = ""  # Tags not used currently
        persons_url = f"{base_url}/api/treesui-list/trees/{tree_id}/persons?name={urllib.parse.quote(query)}&tags={tags}&page=1&limit={limit}&fields=EVENTS,GENDERS,KINSHIP,NAMES,PHOTO,RELATIONS,TAGS&isGetFullPersonObject=false"
        referer = f"{base_url}/family-tree/tree/{tree_id}/family"
        try:
            persons_response = _api_req(
                url=persons_url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                api_description="API Search by Name",
                referer_url=referer,
                timeout=30,
            )
            if not persons_response or not isinstance(persons_response, list):
                if isinstance(persons_response, dict) and "error" in persons_response:
                    logger.error(f"API Search failed: {persons_response['error']}")
                    return []
                logger.warning(
                    f"API Search: Invalid response format for '{query}'. Type: {type(persons_response)}"
                )
                return []
            logger.info(
                f"API Search: Found {len(persons_response)} results for '{query}'."
            )
            return persons_response
        except Exception as search_err:
            logger.error(
                f"API Search: Error fetching persons for '{query}': {search_err}",
                exc_info=True,
            )
            return []

    def format_person_details(self, person: dict) -> str:
        """Formats core details from an Ancestry API person dict for display."""
        name = self._extract_display_name(person)
        gender = None
        # Try various gender fields
        if (
            "Genders" in person
            and person["Genders"]
            and isinstance(person["Genders"][0], dict)
        ):
            gender = person["Genders"][0].get("g")
        elif "gender" in person:
            gender = person["gender"]
        elif "Gender" in person:
            gender = person["Gender"]  # Check facts API field
        gender_str = f"Gender: {gender}" if gender else "Gender: Unknown"

        # Try to get birth event details
        birth_info = ""
        events = (
            person.get("Events")
            or person.get("events")
            or person.get("PersonFacts")
            or []
        )  # Check facts API list too
        for event in events:
            if isinstance(event, dict):
                type_str = (
                    event.get("t", "").lower()
                    or event.get("type", "").lower()
                    or event.get("TypeString", "").lower()
                )
                if type_str == "birth":
                    date_str = (
                        event.get("d") or event.get("date") or event.get("DateString")
                    )  # Raw string date
                    place_str = (
                        event.get("p") or event.get("place") or event.get("PlaceString")
                    )  # Raw string place

                    # Prefer normalized/structured date/place if available (from /facts mainly)
                    norm_date = event.get("Date")  # May be dict
                    norm_place = event.get("Place")  # May be dict

                    if isinstance(norm_date, dict):
                        date_display = (
                            norm_date.get("Display") or date_str or "(Date unknown)"
                        )
                    else:
                        date_display = date_str or "(Date unknown)"

                    if isinstance(norm_place, dict):
                        place_display = (
                            norm_place.get("Display") or place_str or "(Place unknown)"
                        )
                    else:
                        place_display = place_str or "(Place unknown)"

                    birth_info = f"Birth: {date_display} in {place_display}"
                    break  # Stop after finding first birth event

        # Try to get death event details (similar logic to birth)
        death_info = ""
        for event in events:
            if isinstance(event, dict):
                type_str = (
                    event.get("t", "").lower()
                    or event.get("type", "").lower()
                    or event.get("TypeString", "").lower()
                )
                if type_str == "death":
                    date_str = (
                        event.get("d") or event.get("date") or event.get("DateString")
                    )
                    place_str = (
                        event.get("p") or event.get("place") or event.get("PlaceString")
                    )
                    norm_date = event.get("Date")
                    norm_place = event.get("Place")

                    if isinstance(norm_date, dict):
                        date_display = (
                            norm_date.get("Display") or date_str or "(Date unknown)"
                        )
                    else:
                        date_display = date_str or "(Date unknown)"

                    if isinstance(norm_place, dict):
                        place_display = (
                            norm_place.get("Display") or place_str or "(Place unknown)"
                        )
                    else:
                        place_display = place_str or "(Place unknown)"

                    death_info = f"Death: {date_display} in {place_display}"
                    break

        # Get Person ID (CFPID)
        pid = self._extract_person_id(person)
        pid_str = f"Person ID: {pid}" if pid else ""

        # Compose output lines
        lines = [f" Name: {name}", f"   {gender_str}"]
        if birth_info:
            lines.append(f"   {birth_info}")
        if death_info:
            lines.append(f"   {death_info}")  # Add death info if found
        if pid_str:
            lines.append(f"   {pid_str}")
        return "\n".join(lines)

    # End of function format_person_details

    def get_person_facts(self, person_id: str, tree_id: str) -> Dict[str, Any]:
        """Fetches data from the /facts API endpoint for a specific person."""
        if not API_UTILS_AVAILABLE:
            return {"error": "API utilities missing"}
        if not person_id or not tree_id:
            return {"error": "Missing person_id or tree_id for facts fetch."}
        # Construct the specific URL - user id needed here
        if not self.session_manager.my_profile_id:
            logger.error(
                "Cannot fetch facts: Missing own profile ID (user ID) in session manager."
            )
            return {"error": "Missing required user ID for facts API call."}
        facts_url = f"{self.base_url}/family-tree/person/facts/user/{self.session_manager.my_profile_id}/tree/{tree_id}/person/{person_id}"
        referer = f"{self.base_url}/family-tree/person/tree/{tree_id}/person/{person_id}"  # Referer is person page

        logger.debug(f"Fetching /facts API for Person {person_id} in Tree {tree_id}...")
        try:
            response = _api_req(
                url=facts_url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                headers={},  # Contextual headers added by _api_req
                use_csrf_token=False,  # No CSRF usually needed
                api_description="Person Facts API",
                referer_url=referer,
                # No force_text_response needed, expect JSON
            )
            # Check response validity
            if isinstance(response, dict) and "data" in response:
                logger.debug(f"Successfully fetched /facts for Person {person_id}.")
                data = response.get("data", {})
                if isinstance(data, dict):
                    return data
                else:
                    logger.error("Facts API returned non-dict data.")
                    return {}
            elif (
                isinstance(response, dict) and "error" in response
            ):  # Handle error dict from dummy _api_req
                logger.error(f"Facts API failed: {response['error']}")
                return response
            elif isinstance(response, requests.Response):  # Handle HTTP errors
                logger.warning(
                    f"Facts API failed for {person_id}. Status: {response.status_code}. Body: {response.text[:200]}"
                )
                return {"error": f"Facts API failed with status {response.status_code}"}
            else:  # Handle None or other types
                logger.warning(
                    f"Facts API returned unexpected data type: {type(response)}"
                )
                return {
                    "error": f"Facts API returned unexpected data type: {type(response)}"
                }
        except Exception as e:
            logger.error(
                f"Error fetching facts for Person {person_id}: {e}", exc_info=True
            )
            return {"error": f"Exception during facts API call: {type(e).__name__}"}

    # End of function get_person_facts

    def get_relationship_ladder(self, person_id: str) -> Union[str, Dict]:
        """
        Fetches relationship ladder, robustly handling JSONP and returning raw text
        or dict on success/error.
        """
        if not API_UTILS_AVAILABLE:
            return {"error": "API utilities missing"}
        tree_id = self._get_tree_id()
        if not tree_id or not person_id:
            return {
                "error": "Missing tree_id or person_id."
            }  # Return dict for error clarity

        # Unique callback and timestamp for URL
        callback_param = f"jQuery{int(time.time() * 1000)}_{int(time.time() * 1000)}"
        timestamp_param = int(time.time() * 1000)
        url = f"{self.base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/getladder?callback={callback_param}&_={timestamp_param}"
        referer_url = f"{self.base_url}/family-tree/tree/{tree_id}/family"  # Generic family page referer

        logger.debug(f"Fetching /getladder API for Person {person_id}...")
        try:
            response = _api_req(
                url=url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                headers=None,  # Use default/contextual headers
                use_csrf_token=False,  # No CSRF usually needed
                api_description="Ancestry Relationship Ladder",  # Context for _api_req
                referer_url=referer_url,
                timeout=20,
                force_text_response=True,  # MUST be true to get raw JSONP
            )
            # Handle error dict from dummy _api_req
            if isinstance(response, dict) and "error" in response:
                logger.error(f"Ladder API failed: {response['error']}")
                return response

            # Return the raw response (string) or None/Response object on error
            # The caller (display_raw_relationship_ladder) will handle parsing
            return (
                response
                if response
                else {"error": "Ladder API call failed or returned empty."}
            )

        except Exception as e:
            logger.error(
                f"Exception fetching relationship ladder for {person_id}: {e}",
                exc_info=True,
            )
            return {"error": f"Exception during ladder API call: {type(e).__name__}"}

    # End of function get_relationship_ladder


# End of class AncestryAPISearch


# REVISED: Handler for API Report (v7.36 - with Auto-Select & Query Fix)
def handle_api_report():
    """Handler for Option 2 - API Report with auto-selection attempt."""
    logger.info("\n--- API Report ---")
    if not API_UTILS_AVAILABLE:
        logger.error(
            "API functionality is disabled because the 'utils' module could not be loaded."
        )
        return  # End of function handle_api_report

    # --- Initialize API Session ---
    if not initialize_session():
        logger.error(
            "Failed to initialize session. Cannot proceed with API operations."
        )
        return  # End of function handle_api_report

    # Ensure session_manager is passed to the API search class
    api_search = (
        AncestryAPISearch()
    )  # Remove argument, use global session_manager inside class

    # --- Prompt for search criteria (Aligned with GEDCOM) ---
    logger.info("\nEnter search criteria for the person of interest:")
    first_name = input(" First Name (optional): ").strip() or ""
    surname = input(" Surname (optional): ").strip() or ""
    dob_str = input(" Birth Date/Year (optional): ").strip() or None
    pob = input(" Birth Place (optional): ").strip() or None
    dod_str = input(" Death Date/Year (optional): ").strip() or None
    pod = input(" Death Place (optional): ").strip() or None
    target_gender = input(" Gender (M/F, optional): ").strip() or None
    if target_gender:
        target_gender = (
            target_gender[0].lower() if target_gender[0].lower() in ["m", "f"] else None
        )

    # --- Construct Name Query CORRECTLY ---
    query_parts = [
        p for p in [first_name, surname] if p
    ]  # Create list of non-empty parts
    query_name = " ".join(query_parts)  # Join parts with a space

    if not query_name:  # Check if name is still empty after combining
        logger.error(
            "API search requires at least First Name or Surname. Report cancelled."
        )
        return  # End of function handle_api_report

    # --- Explain API Search Limitation ---
    logger.info(f"INFO: Searching Ancestry API using Name: '{query_name}'.")
    logger.info(
        "      Other criteria (dates, places, gender) will be used to help select the best match."
    )

    # --- Find Matches using the Corrected Name Query ---
    persons = api_search.suggest_persons(first_name, surname)
    # Defensive: ensure persons is a list
    if not isinstance(persons, list):
        logger.error("API did not return a list of persons.")
        return
    if not persons:
        logger.info(f"No matches found in Ancestry API for '{query_name}'.")
        return  # End of function handle_api_report
    # Check if the only result is an error dict
    if len(persons) == 1 and isinstance(persons[0], dict) and "error" in persons[0]:
        logger.error(f"API Search Error: {persons[0]['error']}")
        return  # End of function handle_api_report

    # --- Auto-Selection Logic ---
    selected_person = None
    auto_selected = False
    max_results_api = 5  # Limit processing/display for selection
    # Defensive: check for None in top_persons
    top_persons = [p for p in persons[:max_results_api] if isinstance(p, dict)]

    if len(top_persons) == 1:
        selected_person = top_persons[0]
        auto_selected = True
        logger.info(
            f"Found 1 match. Automatically selected: {api_search._extract_display_name(selected_person)}"
        )
    elif len(top_persons) > 1:
        # Try to score based on birth year and gender
        target_birth_year: Optional[int] = None
        birth_dt = _parse_date(dob_str) if dob_str else None
        if birth_dt:
            target_birth_year = birth_dt.year

        scored_persons = []
        for i, person_data in enumerate(top_persons):
            score = 0
            person_birth_year = None
            person_gender = None

            # Extract birth year from API Events
            events = person_data.get("Events") or person_data.get("events") or []
            for event in events:
                # Check event is a dictionary before accessing keys
                if isinstance(event, dict) and (
                    event.get("t", "").lower() == "birth"
                    or event.get("type", "").lower() == "birth"
                ):
                    date_dict = event.get("d") or event.get("date")
                    # Check date_dict is a dictionary before accessing keys
                    if isinstance(date_dict, dict):
                        api_year_str = date_dict.get("normalizedDate") or date_dict.get(
                            "sortDate"
                        )  # Prefer normalized
                        if (
                            api_year_str
                            and isinstance(api_year_str, str)
                            and len(api_year_str) >= 4
                            and api_year_str[:4].isdigit()
                        ):
                            try:
                                person_birth_year = int(api_year_str[:4])
                                break  # Found year
                            except ValueError:
                                logger.debug(
                                    f"Could not convert potential year '{api_year_str[:4]}' to int."
                                )
                        elif date_dict.get("yearInt"):  # Fallback to yearInt
                            year_int_val = date_dict.get("yearInt")
                            if isinstance(year_int_val, int):
                                person_birth_year = year_int_val
                                break
                            elif (
                                isinstance(year_int_val, str) and year_int_val.isdigit()
                            ):
                                try:
                                    person_birth_year = int(year_int_val)
                                    break
                                except ValueError:
                                    logger.debug(
                                        f"Could not convert yearInt string '{year_int_val}' to int."
                                    )

            # Extract gender from API Genders
            genders = person_data.get("Genders") or []
            if genders and isinstance(genders[0], dict):
                person_gender = genders[0].get("g")  # Usually 'm' or 'f'

            # Calculate score
            if (
                target_birth_year
                and person_birth_year
                and abs(target_birth_year - person_birth_year) <= 1
            ):  # +/- 1 year match
                score += 5
            if (
                target_gender
                and person_gender
                and target_gender == person_gender.lower()
            ):
                score += 3

            scored_persons.append(
                {
                    "person": person_data,
                    "score": score,
                    "birth_year": person_birth_year,
                    "gender": person_gender,
                }
            )

        # Sort by score descending
        scored_persons.sort(key=lambda x: x["score"], reverse=True)

        # Check for a clear winner
        # Defensive: check scored_persons is not empty before accessing [0]
        if (
            scored_persons
            and scored_persons[0]["score"] > 0
            and (
                len(scored_persons) == 1
                or scored_persons[0]["score"] >= scored_persons[1]["score"] + 3
            )
        ):
            selected_person = scored_persons[0]["person"]
            auto_selected = True
            logger.info(
                f"Automatically selected best match based on criteria: {api_search._extract_display_name(selected_person)}"
            )
            logger.debug(
                f"Auto-select scores: Top={scored_persons[0]['score']}, Next={(scored_persons[1]['score'] if len(scored_persons)>1 else 'N/A')}"
            )
        else:
            # Fallback to manual selection
            logger.debug(
                "Auto-select conditions not met. Falling back to manual selection."
            )
            logger.info(
                f"Found {len(persons)} potential matches for '{query_name}'. Please select:"
            )
            for i, scored_p in enumerate(scored_persons):  # Display scored top N
                p_data = scored_p["person"]
                name_display = api_search._extract_display_name(p_data)
                year_info = (
                    f" (b. {scored_p['birth_year']})" if scored_p["birth_year"] else ""
                )
                gender_info = f" ({scored_p['gender']})" if scored_p["gender"] else ""
                score_info = f" (Score: {scored_p['score']})"
                logger.info(
                    f"  {i+1}. {name_display}{year_info}{gender_info}{score_info}"
                )

            try:
                choice = int(
                    input(
                        f"\nSelect person (1-{len(scored_persons)}, or 0 to cancel): "
                    )
                )
                if 0 < choice <= len(scored_persons):
                    selected_person = scored_persons[choice - 1]["person"]
                else:
                    logger.info("Selection cancelled or invalid.")
                    return  # End of function handle_api_report
            except ValueError:
                logger.error("Invalid selection. Please enter a number.")
                return  # End of function handle_api_report

    # --- Proceed with the selected person ---
    # Defensive: check selected_person is not None
    if not selected_person:
        logger.info("No person selected.")  # Should be handled above, but safeguard
        return  # End of function handle_api_report

    try:
        # Extract Person ID (CFPID) needed for facts/ladder APIs
        selected_person_id = api_search._extract_person_id(selected_person)
        if not selected_person_id:
            logger.error(
                "Could not determine Person ID (CFPID) for the selected individual."
            )
            return  # End of function handle_api_report

        # Display Core Person Details (API)
        logger.info(
            "--- Individual Details (API) ---\n"
            + api_search.format_person_details(selected_person)
        )

        # Fetch and Display Family Structure (API /facts)
        tree_id = api_search._get_tree_id()
        if not tree_id:
            logger.error("Could not determine Tree ID. Cannot fetch family details.")
        else:
            logger.info("--- Family Structure (API) ---")
            logger.info("(Note: Shows names and life ranges from API /facts data)")
            family_data = api_search.get_person_facts(selected_person_id, tree_id)
            if family_data and "error" not in family_data:
                family_section = family_data.get("personResearch", {}).get(
                    "PersonFamily", {}
                )

                # Helper to format relative display
                def _format_api_relative(relative_dict):
                    if not isinstance(relative_dict, dict):
                        return "  - (Invalid Relative Data)"  # Safeguard
                    name = format_name(relative_dict.get("FullName"))  # Use formatter
                    life = (
                        relative_dict.get("LifeRange", "").replace("–", "-").strip()
                    )  # Clean range
                    return f"  - {name} ({life})" if life else f"  - {name}"

                # Display Parents
                logger.info("\n Parents:")
                fathers = family_section.get("Fathers", [])
                mothers = family_section.get("Mothers", [])
                if fathers or mothers:
                    for p in fathers:
                        logger.info(_format_api_relative(p))
                    for p in mothers:
                        logger.info(_format_api_relative(p))
                else:
                    logger.info("  (None found in API data)")

                # Display Siblings (includes half-siblings if API separates them)
                logger.info("\n Siblings:")
                siblings = family_section.get("Siblings", [])
                half_siblings = family_section.get(
                    "HalfSiblings", []
                )  # Check if API provides this
                all_siblings = siblings + half_siblings
                if all_siblings:
                    # Sort siblings by birth date if available, otherwise by name
                    all_siblings.sort(
                        key=lambda s: (
                            s.get("BirthDate", "9999"),
                            s.get("FullName", ""),
                        )
                    )
                    for s in all_siblings:
                        logger.info(_format_api_relative(s))
                else:
                    logger.info("  (None found in API data)")

                # Display Spouses
                logger.info("\n Spouse(s):")
                spouses = family_section.get("Spouses", [])
                if spouses:
                    for s in spouses:
                        logger.info(_format_api_relative(s))
                else:
                    logger.info("  (None found in API data)")

                # Display Children (API nests children list)
                logger.info("\n Children:")
                children_nested = family_section.get("Children", [])  # List of lists
                all_children = [
                    child for sublist in children_nested for child in sublist
                ]
                if all_children:
                    # Sort children by birth date if available, otherwise by name
                    all_children.sort(
                        key=lambda c: (
                            c.get("BirthDate", "9999"),
                            c.get("FullName", ""),
                        )
                    )
                    for c in all_children:
                        logger.info(_format_api_relative(c))
                else:
                    logger.info("  (None found in API data)")

            elif family_data and "error" in family_data:
                logger.error(f"Error fetching family data: {family_data['error']}")
            else:
                logger.error("Could not retrieve family data from API.")

        # Display Relationship Ladder (API)
        ladder_response = api_search.get_relationship_ladder(selected_person_id)
        display_raw_relationship_ladder(ladder_response)

    except Exception as e:
        logger.error(f"Error in handle_api_report after selection: {e}", exc_info=True)


# End of function handle_api_report


# REVISED: Main function with simplified logic
def main():
    """Main execution flow: Load GEDCOM, process user choices for reports."""
    reader = None
    wayne_gault_indi = None  # Store the Individual object for GEDCOM
    wayne_gault_id_gedcom = None  # Store normalized GEDCOM ID
    fuzzy_max_results_display = 3

    try:
        logger.info("--- GEDCOM Interrogator Script Starting (v7.36) ---")
        # --- Initialization Phase: Load GEDCOM and check libraries ---
        if not GEDCOM_LIB_AVAILABLE or GedcomReader is None:
            raise ImportError("ged4py library unavailable.")

        gedcom_path_str = getattr(config_instance, "GEDCOM_FILE_PATH", None)
        if not gedcom_path_str:
            raise ValueError("GEDCOM_FILE_PATH not set.")
        gedcom_path = Path(gedcom_path_str)
        if not gedcom_path.is_file():
            raise FileNotFoundError(f"GEDCOM not found: {gedcom_path}")

        logger.info(f"Loading GEDCOM: {gedcom_path}...")
        reader = GedcomReader(str(gedcom_path))
        logger.info("GEDCOM loaded successfully.")

        # --- Phase 2: Pre-build cache and find reference person (WGG) in GEDCOM ---
        build_indi_index(reader)
        logger.info("Pre-searching for 'Wayne Gordon Gault' in GEDCOM...")
        wgg_search_name_lower = "wayne gordon gault"
        try:
            for indi in INDI_INDEX.values():  # Search using index
                name_rec = indi.name
                name_str = name_rec.format() if _is_name(name_rec) else ""
                if name_str.lower() == wgg_search_name_lower:
                    wayne_gault_indi = indi
                    wayne_gault_id_gedcom = _normalize_id(indi.xref_id)
                    logger.info(
                        f"Found WGG in GEDCOM: {_get_full_name(wayne_gault_indi)} [@{wayne_gault_id_gedcom}@]"
                    )
                    break
            if not wayne_gault_id_gedcom:
                logger.warning("WGG not found in GEDCOM during startup.")
        except Exception as e:
            logger.error(f"Error during WGG GEDCOM search: {e}", exc_info=True)

        # --- Phase 3: Interactive menu loop ---
        while True:
            if not reader:
                logger.critical("GEDCOM reader unavailable.")
                break  # Should not happen if init succeeds

            choice = menu()

            try:
                # --- ACTION 1: GEDCOM Report ---
                if choice == "1":
                    handle_gedcom_report(
                        reader,
                        wayne_gault_indi,
                        wayne_gault_id_gedcom,
                        fuzzy_max_results_display,
                    )
                # --- ACTION 2: API Report ---
                elif choice == "2":
                    if API_UTILS_AVAILABLE:
                        handle_api_report()  # API handler doesn't need GEDCOM reader/WGG object
                    else:
                        logger.info(
                            "\nAPI functionality is disabled due to missing dependencies ('utils' module)."
                        )

                # --- Toggle Log Level ---
                elif choice == "t":
                    current_level = logger.level
                    new_level = (
                        logging.DEBUG if current_level >= logging.INFO else logging.INFO
                    )
                    logger.setLevel(new_level)
                    # Update handlers' levels as well
                    for handler in logging.getLogger().handlers:
                        if hasattr(handler, "setLevel"):
                            handler.setLevel(new_level)
                    logger.info(f"\nLog level set to {logging.getLevelName(new_level)}")
                    logger.log(new_level, "Log level toggled.")  # Log the toggle itself

                # --- Exit ---
                elif choice == "q":
                    logger.info("\nExiting...")
                    break
                # --- Invalid Choice ---
                else:
                    logger.info("\nInvalid choice.")

                if choice != "q":
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                logger.info("\n\nOperation interrupted.")
                logger.warning("User interrupted operation.")
                input("Press Enter to continue...")
            except Exception as loop_err:
                logger.error(
                    f"Error during menu action (Choice: {choice}): {loop_err}",
                    exc_info=True,
                )
                logger.info(f"\nError occurred: {type(loop_err).__name__}: {loop_err}")
                logger.info("Check logs for details.")
                input("Press Enter to continue...")

    # Handle setup errors
    except (ValueError, FileNotFoundError, ImportError) as setup_err:
        logger.critical(f"Fatal Setup Error: {setup_err}")
        logger.info(f"\nCRITICAL ERROR: {setup_err}")
    # Handle any other unexpected errors in main
    except Exception as outer_e:
        logger.critical(f"Unexpected critical error in main: {outer_e}", exc_info=True)
        logger.info(f"\nCRITICAL ERROR: {outer_e}. Check logs.")
    finally:
        # Attempt to close Selenium driver if it was opened
        if (
            "session_manager" in globals()
            and session_manager
            and getattr(session_manager, "driver_live", False)
            and getattr(session_manager, "driver", None) is not None
        ):
            try:
                logger.info("Attempting to close Selenium driver...")
                driver_obj = getattr(session_manager, "driver", None)
                if (
                    driver_obj is not None
                    and hasattr(driver_obj, "quit")
                    and callable(driver_obj.quit)
                ):
                    driver_obj.quit()
                logger.info("Selenium driver closed.")
            except Exception as close_err:
                logger.error(
                    f"Error closing Selenium driver: {close_err}", exc_info=False
                )

        logger.info("--- GEDCOM Interrogator Script Finished ---")


# End of function main


# REVISED: Handler for GEDCOM Report
def handle_gedcom_report(
    reader, wayne_gault_indi, wayne_gault_id_gedcom, max_results=3
):
    """Handler for Option 1 - GEDCOM Report."""
    logger.info("\n--- GEDCOM Report ---")
    if not wayne_gault_indi or not wayne_gault_id_gedcom:
        logger.info(
            "ERROR: Wayne Gordon Gault (reference person) not found in local GEDCOM."
        )
        logger.info("Cannot calculate relationships accurately.")
        # Allow proceeding without relationship calculation? Or return? Returning for now.
        return  # End of function handle_gedcom_report

    # --- Prompt for search criteria ---
    logger.info("\nEnter search criteria for the person of interest:")
    first_name = input(" First Name (optional): ").strip() or None
    surname = input(" Surname (optional): ").strip() or None
    dob_str = input(" Birth Date/Year (optional): ").strip() or None
    pob = input(" Birth Place (optional): ").strip() or None
    dod_str = input(" Death Date/Year (optional): ").strip() or None
    pod = input(" Death Place (optional): ").strip() or None
    gender = input(" Gender (M/F, optional): ").strip() or None
    if gender:
        gender = gender[0].lower() if gender[0].lower() in ["m", "f"] else None

    if not any([first_name, surname, dob_str, pob, dod_str, pod, gender]):
        logger.info("\nNo search criteria entered. Report cancelled.")
        return  # End of function handle_gedcom_report

    # --- Find potential matches using fuzzy search ---
    matches = find_potential_matches(
        reader, first_name, surname, dob_str, pob, dod_str, pod, gender
    )

    if not matches:
        logger.info("\nNo potential matches found in GEDCOM based on criteria.")
        return  # End of function handle_gedcom_report

    # --- Auto-select if only one match ---
    if len(matches) == 1:
        selected_match = matches[0]
    else:
        for i, match in enumerate(matches[:max_results]):
            # Display name, dates, and reasons for better selection context
            b_info = f"b. {match['birth_date']}" if match["birth_date"] != "N/A" else ""
            d_info = f"d. {match['death_date']}" if match["death_date"] != "N/A" else ""
            date_info = (
                f" ({', '.join(filter(None, [b_info, d_info]))})"
                if b_info or d_info
                else ""
            )
            logger.info(
                f"  {i+1}. {match['name']}{date_info} (Score: {match['score']}, {match['reasons']})"
            )
        try:
            choice = int(input("\nSelect person (or 0 to cancel): "))
            if choice < 1 or choice > len(matches[:max_results]):
                logger.info("Selection cancelled or invalid.")
                return  # End of function handle_gedcom_report
            selected_match = matches[choice - 1]
        except ValueError:
            logger.error("Invalid selection. Please enter a number.")
            return
        except Exception as e:
            logger.error(f"Error in handle_gedcom_report: {e}", exc_info=True)
            logger.info(f"An error occurred: {type(e).__name__}: {e}")
            return

    selected_id = extract_and_fix_id(selected_match["id"])
    if not selected_id:
        logger.info("ERROR: Invalid ID in selected match.")
        return
    selected_indi = find_individual_by_id(reader, selected_id)
    if not selected_indi:
        logger.info("ERROR: Could not retrieve individual record from GEDCOM.")
        return
    display_gedcom_family_details(reader, selected_indi)
    relationship_path = get_relationship_path(
        reader, selected_id, wayne_gault_id_gedcom
    )
    logger.info(f"\n{relationship_path}")


# End of function handle_gedcom_report


def ordinal_case(value: Union[str, int]) -> str:
    """Converts number or string number to ordinal format (e.g., 1 -> 1st, 2 -> 2nd). Also handles simple relationship capitalization."""
    # Check if input is likely a relationship string first
    if isinstance(value, str) and not value.isdigit():
        # Apply title case and fix common relationship capitalizations
        words = value.title().split()
        # Example fixes (add more if needed)
        if "Of" in words:
            words[words.index("Of")] = "of"
        if "The" in words:
            words[words.index("The")] = "the"
        # Capitalize specific relationship terms if needed (e.g., Grandfather)
        # This simple title case might be sufficient for many common terms.
        return " ".join(words)
    # Proceed with ordinal number formatting
    try:
        num = int(value)
        # Handle 11th, 12th, 13th explicitly
        if 11 <= (num % 100) <= 13:
            suffix = "th"
        else:  # Determine suffix based on the last digit
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
    except (
        ValueError,
        TypeError,
    ):  # If conversion fails, return the original value as string
        return str(value)


# End of function ordinal_case


def format_name(name: Optional[str]) -> str:
    """Cleans and formats a person's name string."""
    if not name or not isinstance(name, str):
        return "Valued Relative"
    # Apply title casing and strip whitespace
    cleaned_name = name.strip().title()
    # Remove trailing GEDCOM-style surname slashes (e.g., "John /Doe/")
    cleaned_name = re.sub(r"\s*/([^/]+)/\s*$", r" \1", cleaned_name).strip()
    # Remove leading/trailing slashes if they exist without spaces
    cleaned_name = re.sub(r"^/", "", cleaned_name).strip()
    cleaned_name = re.sub(r"/$", "", cleaned_name).strip()
    # Replace multiple spaces with a single space
    cleaned_name = re.sub(r"\s+", " ", cleaned_name)
    # Return 'Valued Relative' if the name becomes empty after cleaning
    return cleaned_name if cleaned_name else "Valued Relative"


# End of function format_name


# REVISED: display_raw_relationship_ladder (incorporates parsing logic)
def display_raw_relationship_ladder(raw_content):
    """
    Parse and display the Ancestry relationship ladder from raw JSONP/HTML content.
    Robustly extracts the 'html' part, decodes, and parses the relationship and path.
    """
    logger.info("\n--- Relationship to Wayne Gordon Gault (API) ---")
    # Handle cases where API call failed (raw_content might be dict with error)
    if isinstance(raw_content, dict) and "error" in raw_content:
        logger.error(f"Could not retrieve relationship data: {raw_content['error']}")
        return  # End of function display_raw_relationship_ladder
    if not raw_content or not isinstance(raw_content, str):
        logger.error("No relationship content available or invalid format.")
        return  # End of function display_raw_relationship_ladder

    # Try to extract the 'html' part from the JSONP string
    html_escaped = None
    # More robust regex to handle variations in spacing and structure
    html_match = re.search(
        r'["\']html["\']\s*:\s*["\']((?:\\.|[^"\\])*)["\']', raw_content, re.IGNORECASE
    )
    if html_match:
        html_escaped = html_match.group(1)
    else:  # Fallback extraction (less reliable)
        html_start = raw_content.find('html":"')
        if html_start != -1:
            html_start += len('html":"')
            html_escaped = raw_content[html_start:]
            end_quote = -1
            end_seq1 = html_escaped.find('"},')
            end_seq2 = html_escaped.find('"}')
            if end_seq1 != -1 and end_seq2 != -1:
                end_quote = min(end_seq1, end_seq2)
            elif end_seq1 != -1:
                end_quote = end_seq1
            elif end_seq2 != -1:
                end_quote = end_seq2
            if end_quote != -1:
                html_escaped = html_escaped[:end_quote]
            else:
                logger.warning("Fallback HTML extraction might be incomplete.")

    if not html_escaped:
        logger.error(
            "Could not extract relationship ladder HTML from the API response."
        )
        logger.debug(f"Raw content for failed HTML extraction: {raw_content[:500]}...")
        return  # End of function display_raw_relationship_ladder

    # Unescape unicode and HTML entities
    html_unescaped = ""
    try:
        # Handle potential double escaping (e.g., \\uXXXX)
        html_intermediate = html_escaped.replace("\\\\", "\\")
        # Decode unicode escapes
        html_intermediate = bytes(html_intermediate, "utf-8").decode(
            "unicode_escape", errors="replace"
        )
        # Decode HTML entities
        html_unescaped = html.unescape(html_intermediate)
    except Exception as decode_err:
        logger.error(f"Could not decode relationship ladder HTML. Error: {decode_err}")
        logger.error(f"HTML decoding error: {decode_err}", exc_info=True)
        logger.debug(f"Problematic escaped HTML: {html_escaped[:500]}...")
        return  # End of function display_raw_relationship_ladder

    # --- Start of BS4 Parsing Logic ---
    try:
        soup = BeautifulSoup(html_unescaped, "html.parser")
        # Extract Actual Relationship
        actual_relationship = "(not found)"
        # Try multiple selectors for the relationship text
        rel_elem = (
            soup.select_one("ul.textCenter > li:first-child > i > b")
            or soup.select_one("ul > li > i > b")
            or soup.select_one("div.relationshipText > span")
        )  # Another possible structure

        if rel_elem:
            actual_relationship = ordinal_case(
                rel_elem.get_text(strip=True)
            )  # Use helper
        else:
            logger.warning(
                "Could not extract actual_relationship element using selectors."
            )
        logger.info(f"Relationship: {actual_relationship}")

        # Extract Relationship Path
        path_list = []
        # Try common path item selectors
        path_items = soup.select(
            'ul.textCenter > li:not([class*="iconArrowDown"])'
        ) or soup.select(
            "ul#relationshipPath > li"
        )  # Another possible structure

        num_items = len(path_items)
        if not path_items:
            logger.warning("Could not find any path items using selector.")
        else:
            for i, item in enumerate(path_items):
                name_text, desc_text = "", ""
                # Find name (usually in <a> or <b>)
                name_container = item.find("a") or item.find("b") or item.find("strong")
                if name_container:
                    name_text = format_name(
                        name_container.get_text(strip=True).replace('"', "'")
                    )  # Use helper
                else:
                    logger.debug(
                        f"Path item {i}: Could not find name container (a, b, strong). Item: {item}"
                    )

                # Find description (often in <i> or <span class="relationship">)
                # Relationship description is usually for the *next* person in the path relative to the current one
                # This needs careful handling - Ancestry's structure is complex.
                # Let's try to extract the primary text block first.
                item_text_cleaned = item.get_text(separator=" ", strip=True).replace(
                    '"', "'"
                )

                # Try to extract a relationship descriptor if present (often italicized)
                desc_element = item.find("i") or item.find(
                    "span", class_="relationship"
                )
                if desc_element:
                    raw_desc_full = desc_element.get_text(strip=True).replace('"', "'")
                    desc_text = format_name(raw_desc_full)  # Simple formatting for now
                else:
                    # Fallback: Use the whole item text minus the name if found
                    if name_text and item_text_cleaned.startswith(name_text):
                        potential_desc = item_text_cleaned[len(name_text) :].strip()
                        if potential_desc:
                            desc_text = potential_desc
                    logger.debug(
                        f"Path item {i}: Could not find specific description element (i, span.relationship). Item: {item}"
                    )

                # Combine Name and Description (if available)
                display_text = name_text
                if (
                    desc_text and desc_text.lower() != name_text.lower()
                ):  # Avoid repeating name if it's the only desc
                    display_text += f" ({desc_text})"

                if display_text:
                    path_list.append(display_text)
                else:
                    logger.warning(
                        f"Path item {i}: Skipping path item because display_text is empty. Raw item: {item}"
                    )

        # Print the formatted path
        if path_list:
            logger.info("Path:")
            # Display logic assumes path goes from target person up/across to WGG (you)
            # The last item often represents "You"
            # The first item is the target person
            # Items in between show the connection
            logger.info(f"  {path_list[0]} (Target Person)")  # First item is the target
            for i, p in enumerate(path_list[1:-1]):  # Middle items show the path
                logger.info("  ↓")
                logger.info(f"  {p}")
            if len(path_list) > 1:
                logger.info("  ↓")
                logger.info(
                    f"  {path_list[-1]} (You/WGG)"
                )  # Last item is usually the root/user
        else:
            logger.warning("No relationship path found in HTML.")
            logger.debug(
                f"Parsed HTML content that yielded no path: {html_unescaped[:500]}..."
            )

    except Exception as bs_parse_err:
        logger.error(
            f"Error parsing relationship ladder HTML with BeautifulSoup: {bs_parse_err}"
        )
        logger.error(f"BeautifulSoup parsing error: {bs_parse_err}", exc_info=True)
        logger.debug(f"Problematic unescaped HTML: {html_unescaped[:500]}...")
    # --- End of BS4 Parsing Logic ---


# End of function display_raw_relationship_ladder

# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure logger is set up
    if "logger" not in globals() or not isinstance(logger, logging.Logger):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logger = logging.getLogger("gedcom_main_fallback")
        logger.warning("Using fallback logger.")
    # Check if GEDCOM library is available before running main
    if GEDCOM_LIB_AVAILABLE:
        main()
    else:
        logger.critical("Exiting: ged4py library unavailable.")
        sys.exit(1)
# End of script
