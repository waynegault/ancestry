# gedcom_utils.py
"""
Utility functions for loading, parsing, and querying GEDCOM data using ged4py.
Includes relationship mapping, path calculation, and fuzzy matching/scoring.
Consolidates helper functions and core logic from temp.py v7.36 for use
by action10.py and action11.py.
V16.0: Consolidated utils from temp.py, added standalone scoring function.
V16.1: Added standalone self-check functionality.
V16.2: Fixed IndentationError in fast_bidirectional_bfs.
V16.3: Fixed additional IndentationError in fast_bidirectional_bfs.
"""

# --- Standard library imports ---
import logging
import sys
import re
import time
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set, Deque, Union, Any
from collections import deque
from datetime import datetime, timezone
import difflib

# third party imports
from ged4py.parser import GedcomReader
from ged4py.model import Individual, Record, Name

# --- Local application imports ---
from utils import (format_name, ordinal_case)

# Use centralized logging config setup in main scripts
# Centralized logging config setup (see main scripts for custom config)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gedcom_utils")

# Global cache for family relationships and individual lookup
FAMILY_MAPS_CACHE: Optional[Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]] = None
FAMILY_MAPS_BUILD_TIME: float = 0
INDI_INDEX: Dict[str, Any] = {} # Store Individual objects keyed by normalized ID
INDI_INDEX_BUILD_TIME: float = 0 # Track build time


def _is_individual(obj) -> bool:
    """Checks if object is an Individual safely handling None values"""
    # Check if the object is not None and if its type name is 'Individual'
    # Use isinstance with the imported Individual class if available
    return obj is not None and Individual is not None and isinstance(obj, Individual)
# End of _is_individual

def _is_record(obj) -> bool:
    """Checks if object is a Record safely handling None values"""
    # Check if the object is not None and if its type name is 'Record'
    # Use isinstance with the imported Record class if available
    return obj is not None and Record is not None and isinstance(obj, Record)
# End of _is_record

def _is_name(obj) -> bool:
    """Checks if object is a Name safely handling None values"""
    # Check if the object is not None and if its type name is 'Name'
    # Use isinstance with the imported Name class if available
    return obj is not None and Name is not None and isinstance(obj, Name)
# End of _is_name

def _normalize_id(xref_id: Optional[str]) -> Optional[str]:
    """Normalizes INDI/FAM etc IDs (e.g., '@I123@' -> 'I123')."""
    if xref_id and isinstance(xref_id, str):
        # Match standard GEDCOM ID format (leading char + digits/chars/hyphen), allowing optional @ symbols
        match = re.match(r"^@?([IFSTNMCXO][0-9A-Z\-]+)@?$", xref_id.strip().upper())
        if match:
            return match.group(1)
    # Return None if input is invalid or doesn't match format
    return None
# End of _normalize_id

def extract_and_fix_id(raw_id):
    """
    Cleans and validates a raw ID string (e.g., '@I123@', 'F45').
    Returns the normalized ID (e.g., 'I123', 'F45') or None if invalid.
    Matches temp.py v7.36 logic.
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
# End of extract_and_fix_id

def _get_full_name(indi) -> str:
    """Safely gets formatted name using Name.format(). Handles None/errors."""
    # Ensure the input is an Individual object before proceeding
    if not _is_individual(indi):
         # Attempt to access the underlying object if wrapped (e.g., from Record)
         if hasattr(indi, 'value') and _is_individual(indi.value):
              indi = indi.value
         else:
            logger.warning(f"_get_full_name called with non-Individual type: {type(indi)}")
            return "Unknown (Invalid Type)"

    try:
        name_rec = indi.name # Access the name record
        if _is_name(name_rec):
            # Use the ged4py Name object's format method
            formatted_name = name_rec.format()
            # Clean up extra spaces and apply title case - Use utils.format_name (if available)
            name_formatter = format_name if 'format_name' in globals() and callable(format_name) else lambda x: str(x)
            cleaned_name = name_formatter(formatted_name)
            # Return cleaned name or a placeholder if empty
            return cleaned_name if cleaned_name else "Unknown (Empty Name)"
        elif name_rec is None:
            return "Unknown (No Name Tag)"
        else:
            # Log warning if .name attribute is not a Name object
            indi_id_log = _normalize_id(getattr(indi, 'xref_id', None)) or "Unknown ID"
            logger.warning(
                f"Indi @{indi_id_log}@ unexpected .name type: {type(name_rec)}"
            )
            return f"Unknown (Type {type(name_rec).__name__})"
    except AttributeError:
        # Handle cases where .name attribute might be missing
        return "Unknown (Attr Error)"
    except Exception as e:
        # Catch any other unexpected errors during name formatting
        indi_id_log = _normalize_id(getattr(indi, 'xref_id', None)) or "Unknown ID"
        logger.error(
            f"Error formatting name for @{indi_id_log}@: {e}",
            exc_info=False, # Set to True for full traceback if needed
        )
        return "Unknown (Error)"
# End of _get_full_name


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
    clean_date_str = re.sub(r"\s*\(.*\)\s*", "", clean_date_str).strip()  # Remove content in parentheses
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
# End of _parse_date

def _clean_display_date(raw_date_str: Optional[str]) -> str:
    """Removes surrounding brackets if date exists, handles empty brackets."""
    if not raw_date_str or raw_date_str == "N/A":
        return "N/A"
    cleaned = raw_date_str.strip()
    # Remove surrounding parentheses only if they enclose the entire string
    cleaned = re.sub(r"^\((.+)\)$", r"\1", cleaned).strip()
    # Replace GEDCOM qualifiers with standard symbols
    cleaned = (
        cleaned.replace("ABT ", "~")
        .replace("EST ", "~")
        .replace("BEF ", "<")
        .replace("AFT ", ">")
    )
    # Replace GEDCOM date ranges
    cleaned = (
        cleaned.replace("BET ", "")
        .replace(" FROM ", "")
        .replace(" TO ", "-")
        .replace(" AND ", "-")
    )
    # Return "N/A" if the string is empty after cleaning
    return cleaned if cleaned else "N/A"
# End of _clean_display_date


def _get_event_info(individual, event_tag: str) -> Tuple[Optional[datetime], str, str]:
    """Gets date/place for an event using tag.value. Handles non-string dates."""
    date_obj: Optional[datetime] = None
    date_str: str = "N/A"
    place_str: str = "N/A"
    indi_id_log = "Invalid/Unknown"

    # Ensure the input is an Individual object before proceeding
    if _is_individual(individual):
        indi_id_log = _normalize_id(getattr(individual, 'xref_id', None)) or "Unknown ID"
    # Attempt to access the underlying object if wrapped (e.g., from Record or Link)
    elif hasattr(individual, 'value') and _is_individual(individual.value):
        individual = individual.value # Re-assign individual to the actual Individual object
        indi_id_log = _normalize_id(getattr(individual, 'xref_id', None)) or "Unknown ID"
    else:
        logger.warning(f"_get_event_info invalid input type: {type(individual)}")
        return date_obj, date_str, place_str

    try:
        # Find the event record (e.g., BIRT, DEAT)
        event_record = individual.sub_tag(event_tag.upper())
        if event_record:
            # Get Date sub-tag
            date_tag = event_record.sub_tag("DATE")
            if date_tag and hasattr(date_tag, "value"):
                raw_date_val = date_tag.value
                # Process date value based on its type
                if isinstance(raw_date_val, str):
                    processed_date_str = raw_date_val.strip()
                    date_str = processed_date_str if processed_date_str else "N/A"
                    # Use imported _parse_date if available, otherwise it's None
                    date_obj = _parse_date(date_str) if _parse_date else None
                elif raw_date_val is not None: # Handle non-string but non-None values
                    date_str = str(raw_date_val)
                    date_obj = _parse_date(date_str) if _parse_date else None
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
    except AttributeError as ae:
        logger.debug(f"Attr error getting event '{event_tag}' for {indi_id_log}: {ae}")
    except Exception as e:
        # Log unexpected errors
        logger.error(
            f"Error accessing event {event_tag} for @{indi_id_log}@: {e}",
            exc_info=True,
        )
    return date_obj, date_str, place_str
# End of _get_event_info


def format_life_dates(indi) -> str:
    """Returns a formatted string with birth and death dates."""
    # Ensure input is a valid Individual object
    if not _is_individual(indi):
        logger.warning(f"format_life_dates called with non-Individual type: {type(indi)}")
        return ""

    # Get birth and death info using _get_event_info
    b_date_obj, b_date_str, b_place = _get_event_info(indi, "BIRT")
    d_date_obj, d_date_str, d_place = _get_event_info(indi, "DEAT")

    # Clean the display strings (remove parentheses etc.) using _clean_display_date (if available)
    b_date_str_cleaned = _clean_display_date(b_date_str) if _clean_display_date else str(b_date_str)
    d_date_str_cleaned = _clean_display_date(d_date_str) if _clean_display_date else str(d_date_str)


    # Format birth and death parts
    birth_info = f"b. {b_date_str_cleaned}" if b_date_str_cleaned != "N/A" else ""
    death_info = f"d. {d_date_str_cleaned}" if d_date_str_cleaned != "N/A" else ""

    # Combine parts if they exist
    life_parts = [info for info in [birth_info, death_info] if info]

    # Return formatted string (e.g., "(b. 1900, d. 1980)") or empty string
    return f" ({', '.join(life_parts)})" if life_parts else ""
# End of format_life_dates


def format_full_life_details(indi) -> Tuple[str, str]:
    """Returns formatted birth and death details (date and place) for display."""
    # Ensure input is a valid Individual object
    if not _is_individual(indi):
        logger.warning(f"format_full_life_details called with non-Individual type: {type(indi)}")
        return "(Error: Invalid data)", ""

    # Get birth event details
    b_date_obj, b_date_str, b_place = _get_event_info(indi, "BIRT")
    # Clean the display string using _clean_display_date (if available)
    b_date_str_cleaned = _clean_display_date(b_date_str) if _clean_display_date else str(b_date_str)

    # Format birth string
    birth_info = (
        f"Born: {b_date_str_cleaned if b_date_str_cleaned != 'N/A' else '(Date unknown)'} "
        f"in {b_place if b_place != 'N/A' else '(Place unknown)'}"
    )

    # Get death event details
    d_date_obj, d_date_str, d_place = _get_event_info(indi, "DEAT")
    # Clean the display string using _clean_display_date (if available)
    d_date_str_cleaned = _clean_display_date(d_date_str) if _clean_display_date else str(d_date_str)

    death_info = ""
    # Format death string only if death date or place exists
    if d_date_str_cleaned != "N/A" or d_place != "N/A":
        death_info = (
            f"   Died: {d_date_str_cleaned if d_date_str_cleaned != 'N/A' else '(Date unknown)'} "
            f"in {d_place if d_place != 'N/A' else '(Place unknown)'}"
        )
    return birth_info, death_info
# End of format_full_life_details


def format_relative_info(relative) -> str:
    """Formats information about a relative (name and life dates) for display."""
    # Ensure input is a valid Individual object or a Link/Record containing one
    indi_obj = None
    if _is_individual(relative):
        indi_obj = relative
    elif hasattr(relative, 'value') and _is_individual(relative.value):
        indi_obj = relative.value
    else:
         # Handle cases where it might be a dead xref (just have xref_id)
         if hasattr(relative, 'xref_id'):
              raw_id = getattr(relative, 'xref_id', 'N/A')
              norm_id = _normalize_id(raw_id)
              return f"  - (Invalid Relative Data: ID={norm_id or 'N/A'}, Type={type(relative).__name__})"
         else:
              return f"  - (Invalid Relative Data: Type={type(relative).__name__})"

    # Get relative's full name using _get_full_name (if available)
    rel_name = _get_full_name(indi_obj) if _get_full_name else getattr(indi_obj, 'xref_id', 'Unknown Name') # Fallback name
    # Get formatted life dates (b. date, d. date) using format_life_dates (if available)
    life_info = format_life_dates(indi_obj) if format_life_dates else ""
    # Combine into a display string
    return f"  - {rel_name}{life_info}"
# End of format_relative_info


# --- Cache Building Functions (from temp.py v7.36) ---

def build_indi_index(reader):
    """Builds a dictionary mapping normalized ID to Individual object."""
    global INDI_INDEX, INDI_INDEX_BUILD_TIME
    if INDI_INDEX:  # Avoid rebuilding if already built
        logger.debug("[Cache] INDI index already built. Skipping build.")
        return
    if not reader:
        logger.error("Cannot build INDI index: GedcomReader not available.")
        return
    start_time = time.time()
    logger.info("[Cache] Building INDI index...")
    count = 0
    try:
        for indi in reader.records0("INDI"):
            # Ensure it's an individual object and has a usable xref_id
            if _is_individual(indi) and hasattr(indi, "xref_id") and indi.xref_id:
                norm_id = _normalize_id(indi.xref_id)
                if norm_id:
                    INDI_INDEX[norm_id] = indi
                    count += 1
                else:
                    logger.debug(f"Skipping INDI with unnormalizable xref_id: {indi.xref_id}")
            else:
                # Log skipping records that aren't recognized as Individuals
                if hasattr(indi, 'xref_id'):
                     logger.debug(f"Skipping non-Individual record during index build: Type={type(indi).__name__}, Xref={indi.xref_id}")
                else:
                     logger.debug(f"Skipping record during index build with no xref_id: Type={type(indi).__name__}")


    except Exception as e:
        logger.error(f"Error during INDI index build: {e}", exc_info=True)
        INDI_INDEX = {} # Clear index on error

    elapsed = time.time() - start_time
    INDI_INDEX_BUILD_TIME = elapsed
    logger.info(f"[Cache] INDI index built with {count} individuals in {elapsed:.2f}s.")
# End of build_indi_index


def build_family_maps(reader):
    """
    Builds id_to_parents and id_to_children maps for all individuals.
    Uses direct tag access logic from temp.py v7.36.
    Caches results globally.
    """
    global FAMILY_MAPS_CACHE, FAMILY_MAPS_BUILD_TIME
    if FAMILY_MAPS_CACHE:
        logger.debug("[Cache] Returning cached family maps.")
        return FAMILY_MAPS_CACHE
    if not reader:
        logger.error("Cannot build family maps: GedcomReader not available.")
        return None

    start_time = time.time()
    logger.info("[Cache] Building family maps (direct tag access)...")

    id_to_parents: Dict[str, Set[str]] = {}
    id_to_children: Dict[str, Set[str]] = {}
    fam_count = 0
    indi_count = 0 # This count is just for logging context, maps are built from FAMs
    processed_links = 0

    try:
        # Iterate through family records to build relationships
        for fam in reader.records0("FAM"):
            fam_count += 1
            if not _is_record(fam):
                continue

            # Get parents' IDs in this family
            parents = set()
            husband_tag = fam.sub_tag("HUSB")
            wife_tag = fam.sub_tag("WIFE")

            # Ensure tag exists and has xref_id before checking its value/id
            if husband_tag is not None and hasattr(husband_tag, "xref_id") and husband_tag.xref_id:
                 # Check if the tag itself is an individual or contains one via value
                 if _is_individual(husband_tag) or (hasattr(husband_tag, 'value') and _is_individual(husband_tag.value)):
                      parent_id_h = _normalize_id(husband_tag.xref_id)
                      if parent_id_h:
                           parents.add(parent_id_h)

            # Ensure tag exists and has xref_id before checking its value/id
            if wife_tag is not None and hasattr(wife_tag, "xref_id") and wife_tag.xref_id:
                 # Check if the tag itself is an individual or contains one via value
                 if _is_individual(wife_tag) or (hasattr(wife_tag, 'value') and _is_individual(wife_tag.value)):
                      parent_id_w = _normalize_id(wife_tag.xref_id)
                      if parent_id_w:
                           parents.add(parent_id_w)


            # Process children in this family
            children_tags = fam.sub_tags("CHIL")
            for child_tag in children_tags:
                 # Ensure the child tag holds a reference to an Individual and has an xref_id
                if _is_individual(child_tag) and hasattr(child_tag, "xref_id") and child_tag.xref_id:
                    child_id = _normalize_id(child_tag.xref_id)
                    if child_id:
                        processed_links += 1
                        # Map child to these parents
                        id_to_parents.setdefault(child_id, set()).update(parents)
                        # Map these parents to this child
                        for parent_id in parents:
                            if parent_id: # Ensure parent ID is valid before mapping child
                                id_to_children.setdefault(parent_id, set()).add(child_id)
                elif child_tag is not None and hasattr(child_tag, 'xref_id'):
                     # Log if a CHIL tag has xref_id but isn't recognized as Individual
                     logger.debug(f"Skipping non-Individual CHIL record in FAM {getattr(fam, 'xref_id', 'N/A')}: Type={type(child_tag).__name__}, Xref={child_tag.xref_id}")
                elif child_tag is not None:
                     logger.debug(f"Skipping CHIL record in FAM {getattr(fam, 'xref_id', 'N/A')} with no xref_id: Type={type(child_tag).__name__}")


        # Optional: Count individuals for logging context
        # This loop isn't strictly necessary for building the maps, but can provide stats
        try:
            for indi in reader.records0("INDI"):
                indi_count += 1
        except Exception as e:
             logger.warning(f"Error counting individuals after map build: {e}")
             indi_count = -1 # Indicate count failed


    except AttributeError as ae:
        # Catch specific attribute errors during traversal (e.g., unexpected structure)
        logger.error(
            f"AttributeError during family map build: {ae}.",
            exc_info=True, # Log traceback for debugging structure issues
        )
        FAMILY_MAPS_CACHE = None # Clear cache on error
        return None
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(
            f"Unexpected error during family map build: {e}",
            exc_info=True,
        )
        FAMILY_MAPS_CACHE = None # Clear cache on error
        return None

    elapsed = time.time() - start_time
    FAMILY_MAPS_BUILD_TIME = elapsed
    logger.info(
        f"[Cache] Family maps built: {fam_count} FAMs processed. Found {processed_links} parent/child links. "
        f"Map sizes: {len(id_to_parents)} child->parents entries, {len(id_to_children)} parent->children entries in {elapsed:.2f}s."
    )
    if indi_count >= 0:
         logger.info(f"[Cache] Processed {indi_count} INDI records (total count for context).")


    FAMILY_MAPS_CACHE = (id_to_parents, id_to_children) # Cache the results
    return id_to_parents, id_to_children
# End of build_family_maps


# --- ID Lookup and Extraction ---

def find_individual_by_id(reader, norm_id: Optional[str]):
    """Finds an individual by normalized ID using the pre-built index."""
    global INDI_INDEX
    if not norm_id or not isinstance(norm_id, str):
        logger.warning("find_individual_by_id called with invalid norm_id: None or not string")
        return None
    # Ensure index is built
    if not INDI_INDEX:
        logger.debug("INDI_INDEX not built, attempting build.")
        # Check if build_indi_index is available before calling
        if build_indi_index:
             build_indi_index(reader)
        else:
             logger.error("build_indi_index function is not available. Cannot build index.")
             return None

        if not INDI_INDEX: # Check again after build attempt
             logger.error("INDI_INDEX build failed or is empty. Cannot lookup individual by ID.")
             return None

    # Use the pre-built index for O(1) lookup
    found_indi = INDI_INDEX.get(norm_id)
    if not found_indi:
        logger.debug(f"Individual with normalized ID {norm_id} not found in INDI_INDEX.")
    return found_indi
# End of find_individual_by_id


# --- Core Data Retrieval Functions (from temp.py v7.36) ---

def _find_family_records_where_individual_is_child(reader, target_id):
    """Helper function to find family records where an individual is listed as a child."""
    parent_families = []
    if not reader:
        logger.error("_find_family_records_where_individual_is_child: No reader.")
        return parent_families
    if not target_id or not isinstance(target_id, str):
         logger.warning("_find_family_records_where_individual_is_child called with invalid target_id.")
         return parent_families

    try:
        # Iterate through all FAM records
        for family_record in reader.records0("FAM"):
            if not _is_record(family_record):
                continue

            # Check children in the current family
            children_in_fam = family_record.sub_tags("CHIL")
            if children_in_fam:
                for child_tag in children_in_fam:
                    # Check if child_tag links to an Individual and its normalized ID matches target_id
                    if (_is_individual(child_tag) or (hasattr(child_tag, 'value') and _is_individual(child_tag.value))) \
                       and hasattr(child_tag, "xref_id") and _normalize_id(child_tag.xref_id) == target_id:
                        parent_families.append(family_record)
                        # Assume one person is a child in a given family record only once
                        break
    except Exception as e:
        logger.error(
            f"Error in _find_family_records_where_individual_is_child for ID {target_id}: {e}",
            exc_info=True,
        )
        parent_families = [] # Return empty list on error
    return parent_families
# End of _find_family_records_where_individual_is_child


def _find_family_records_where_individual_is_parent(reader, target_id):
    """Helper function to find family records where an individual is listed as a parent (HUSB or WIFE)."""
    parent_families = [] # List of tuples: (family_record, is_husband, is_wife)
    if not reader:
        logger.error("_find_family_records_where_individual_is_parent: No reader.")
        return parent_families
    if not target_id or not isinstance(target_id, str):
         logger.warning("_find_family_records_where_individual_is_parent called with invalid target_id.")
         return parent_families

    try:
        # Iterate through all FAM records
        for family_record in reader.records0("FAM"):
            if not _is_record(family_record):
                continue

            # Get husband and wife records/tags
            husband_tag = family_record.sub_tag("HUSB")
            wife_tag = family_record.sub_tag("WIFE")

            # Check if target ID matches either parent's normalized ID
            is_target_husband = False
            # Ensure tag exists and has xref_id before checking its value/id
            if husband_tag is not None and hasattr(husband_tag, "xref_id") and husband_tag.xref_id:
                 # Check if the tag itself is an individual or contains one via value
                 if _is_individual(husband_tag) or (hasattr(husband_tag, 'value') and _is_individual(husband_tag.value)):
                      if _normalize_id(husband_tag.xref_id) == target_id:
                           is_target_husband = True

            is_target_wife = False
            # Ensure tag exists and has xref_id before checking its value/id
            if wife_tag is not None and hasattr(wife_tag, "xref_id") and wife_tag.xref_id:
                 # Check if the tag itself is an individual or contains one via value
                 if _is_individual(wife_tag) or (hasattr(wife_tag, 'value') and _is_individual(wife_tag.value)):
                       if _normalize_id(wife_tag.xref_id) == target_id:
                            is_target_wife = True

            # If target is a parent in this family, add the record along with role flags
            if is_target_husband or is_target_wife:
                parent_families.append((family_record, is_target_husband, is_target_wife))

    except Exception as e:
        logger.error(
            f"Error in _find_family_records_where_individual_is_parent for ID {target_id}: {e}",
            exc_info=True,
        )
        parent_families = [] # Return empty list on error

    return parent_families
# End of _find_family_records_where_individual_is_parent


def get_related_individuals(reader, individual, relationship_type: str) -> List[Any]:
    """
    Gets parents, spouses, children, or siblings using family record lookups (NOT maps).
    Based on temp.py v7.36 logic.
    Returns a list of Individual objects.
    """
    related_individuals: List[Any] = []
    unique_related_ids: Set[str] = set()

    # Validate inputs
    if not reader:
        logger.error("get_related_individuals: No reader.")
        return related_individuals
    if not _is_individual(individual) or not hasattr(individual, "xref_id") or not individual.xref_id:
        logger.warning(f"get_related_individuals: Invalid input individual object: {type(individual)}")
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
            # logger.debug(f"Finding parents for {target_id}...")
            # Find families where the target is a child
            parent_families = _find_family_records_where_individual_is_child(
                reader, target_id
            )
            potential_parents = []
            # Extract parents (HUSB, WIFE) from those families
            for family_record in parent_families:
                husband = family_record.sub_tag("HUSB")
                wife = family_record.sub_tag("WIFE")
                # Add the actual Individual object if the tag references one
                if _is_individual(husband): potential_parents.append(husband)
                elif husband is not None and hasattr(husband, 'value') and _is_individual(husband.value): potential_parents.append(husband.value) # Handle Link objects

                if _is_individual(wife): potential_parents.append(wife)
                elif wife is not None and hasattr(wife, 'value') and _is_individual(wife.value): potential_parents.append(wife.value) # Handle Link objects

            # Add unique parents to the result list
            for parent in potential_parents:
                # Ensure it's a valid Individual object before processing
                if parent is not None and _is_individual(parent) and hasattr(parent, "xref_id") and parent.xref_id:
                    parent_id = _normalize_id(parent.xref_id)
                    if parent_id and parent_id not in unique_related_ids:
                        related_individuals.append(parent) # Append the Individual object
                        unique_related_ids.add(parent_id)
            # logger.debug(f"Added {len(unique_related_ids)} unique parents for {target_id}.")

        elif relationship_type == "siblings":
            # logger.debug(f"Finding siblings for {target_id}...")
            # Find families where the target is a child
            parent_families = _find_family_records_where_individual_is_child(
                reader, target_id
            )
            potential_siblings = []
            # Collect all children from those families
            for fam in parent_families:
                fam_children = fam.sub_tags("CHIL")
                if fam_children:
                     # Extend list with Individual objects from CHIL tags, handling Link objects
                    potential_siblings.extend(c.value if hasattr(c, 'value') and _is_individual(c.value) else c for c in fam_children if (_is_individual(c) or (hasattr(c, 'value') and _is_individual(c.value))))

            # Add unique siblings (excluding the target) to the result list
            for sibling in potential_siblings:
                 # Ensure it's a valid Individual object before processing
                if sibling is not None and _is_individual(sibling) and hasattr(sibling, "xref_id") and sibling.xref_id:
                    sibling_id = _normalize_id(sibling.xref_id)
                    if (
                        sibling_id
                        and sibling_id not in unique_related_ids
                        and sibling_id != target_id # Exclude self
                    ):
                        related_individuals.append(sibling) # Append the Individual object
                        unique_related_ids.add(sibling_id)
            # logger.debug(f"Added {len(unique_related_ids)} unique siblings for {target_id}."
        elif relationship_type in ["spouses", "children"]:
            # logger.debug(f"Finding {relationship_type} for {target_id}...")
            # Find families where the target is a parent
            parent_families = _find_family_records_where_individual_is_parent(
                reader, target_id
            )

            if relationship_type == "spouses":
                # Extract the *other* parent from each family
                for family_record, is_target_husband, is_target_wife in parent_families:
                    other_spouse_tag = None
                    if is_target_husband:
                        other_spouse_tag = family_record.sub_tag("WIFE")
                    elif is_target_wife:
                        other_spouse_tag = family_record.sub_tag("HUSB")

                    # Add unique spouses to the result list
                    # Ensure other_spouse_tag links to an Individual object
                    if other_spouse_tag is not None and hasattr(other_spouse_tag, 'value') and _is_individual(other_spouse_tag.value):
                         other_spouse_indi = other_spouse_tag.value
                    elif _is_individual(other_spouse_tag):
                         other_spouse_indi = other_spouse_tag
                    else:
                         other_spouse_indi = None # Invalid spouse tag


                    if (
                        other_spouse_indi is not None
                        and _is_individual(other_spouse_indi)
                        and hasattr(other_spouse_indi, "xref_id")
                        and other_spouse_indi.xref_id
                    ):
                        spouse_id = _normalize_id(other_spouse_indi.xref_id)
                        if spouse_id and spouse_id not in unique_related_ids:
                            related_individuals.append(other_spouse_indi) # Append the Individual object
                            unique_related_ids.add(spouse_id)
                # logger.debug(f"Added {len(unique_related_ids)} unique spouses for {target_id}.")

            else:  # relationship_type == "children"
                # Extract all children from each family where target is a parent
                for family_record, _, _ in parent_families:
                    children_list = family_record.sub_tags("CHIL")
                    if children_list:
                        for child_tag in children_list:
                            # Add unique children to the result list
                            # Ensure child_tag links to an Individual object
                            if hasattr(child_tag, 'value') and _is_individual(child_tag.value):
                                 child_indi = child_tag.value
                            elif _is_individual(child_tag):
                                 child_indi = child_tag
                            else:
                                 child_indi = None # Invalid child tag


                            if (
                                child_indi is not None
                                and _is_individual(child_indi)
                                and hasattr(child_indi, "xref_id")
                                and child_indi.xref_id
                            ):
                                child_id = _normalize_id(child_indi.xref_id)
                                if child_id and child_id not in unique_related_ids:
                                    related_individuals.append(child_indi) # Append the Individual object
                                    unique_related_ids.add(child_id)
                # logger.debug(f"Added {len(unique_related_ids)} unique children for {target_id}.")

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
        related_individuals = [] # Clear on error
    except Exception as e:
        logger.error(
            f"Unexpected error finding {relationship_type} for {target_id}: {e}",
            exc_info=True,
        )
        related_individuals = [] # Clear on error

    # Sort the results by ID for consistent display ordering
    # Ensure objects have xref_id before sorting
    related_individuals.sort(key=lambda x: (_normalize_id(getattr(x, "xref_id", None)) or ""))

    return related_individuals
# End of get_related_individuals


# --- Relationship Path Functions (from temp.py v7.36) ---

def _reconstruct_path(
    start_id, end_id, meeting_id, visited_fwd, visited_bwd
) -> List[str]:
    """
    Reconstructs the path from start to end via the meeting point using predecessor maps.
    Returns a list of normalized IDs.
    Based on temp.py v7.36 logic.
    """
    path_fwd = []
    curr = meeting_id
    # Trace path back from meeting point to start node using forward predecessors
    while curr is not None:
        path_fwd.append(curr)
        curr = visited_fwd.get(curr)
    path_fwd.reverse() # Reverse to get path from start to meeting point

    path_bwd = []
    curr = visited_bwd.get(meeting_id) # Start from the predecessor of the meeting point in the backward search
    # Trace path back from meeting point's predecessor to end node using backward predecessors
    while curr is not None:
        path_bwd.append(curr)
        curr = visited_bwd.get(curr)

    # Combine the forward and backward paths
    # Check if meeting_id is already the first element of path_bwd (can happen if meeting point is start/end of one search)
    if path_bwd and path_fwd and path_bwd and path_fwd[-1] == path_bwd[0]:
        # If the last node of the forward path is the same as the first node of the backward path (the meeting point)
        # then exclude the first node from the backward path list to avoid duplication.
        path = path_fwd + path_bwd[1:]
        logger.debug(f"_reconstruct_path: Meeting ID {meeting_id} was last of FWD and start of BWD, excluded from BWD list.")
    else:
         path = path_fwd + path_bwd

    # --- Sanity Checks ---
    if not path:
        logger.error("_reconstruct_path: Failed to reconstruct any path.")
        return []
    # The BFS guarantees start_id is in visited_fwd and end_id in visited_bwd.
    # The reconstruction should naturally include start_id as the first element
    # of the reversed fwd path and end_id as the last element of the bwd path.
    # Checking and prepending/appending as fallback if needed.
    if path and path[0] != start_id: # Check if path is not empty before accessing index
        logger.warning(f"_reconstruct_path: Path doesn't start with start_id ({path[0]} != {start_id}). Prepending.")
        path.insert(0, start_id) # Attempt fix
    if path and path[-1] != end_id: # Check if path is not empty before accessing index
        logger.warning(f"_reconstruct_path: Path doesn't end with end_id ({path[-1]} != {end_id}). Appending.")
        path.append(end_id) # Attempt fix


    logger.debug(f"_reconstruct_path: Final reconstructed path IDs: {path}")
    return path
# End of _reconstruct_path


def explain_relationship_path(
    path_ids: List[str],
    reader,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
) -> str:
    """
    Return a human-readable explanation of the relationship path with relationship labels.
    Based on temp.py v7.36 logic.
    """
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"
    if id_to_parents is None or id_to_children is None:
        return "(Error: Relationship maps unavailable for explanation)"

    steps = []
    # Get ordinal_case helper from utils (if available)
    ordinal_formatter = ordinal_case if 'ordinal_case' in globals() and callable(ordinal_case) else lambda x: str(x)

    # Iterate through pairs of consecutive IDs in the path
    for i in range(len(path_ids) - 1):
        id_a, id_b = path_ids[i], path_ids[i + 1]
        indi_a = find_individual_by_id(reader, id_a)
        indi_b = find_individual_by_id(reader, id_b)

        # Get names, handling potential lookup failures (use _get_full_name if available)
        name_a = (_get_full_name(indi_a) if _get_full_name and indi_a else f"Unknown ({id_a})")
        name_b = (_get_full_name(indi_b) if _get_full_name and indi_b else f"Unknown ({id_b})")


        # Determine relationship using pre-built maps
        # Relationship is from Person A (current) to Person B (next)
        label = "related"  # Default relationship

        # Check if B is a parent of A (A is the child of B)
        if id_b in id_to_parents.get(id_a, set()):
             # A is the child of B. Describe relationship A -> B
             sex_a = getattr(indi_a, "sex", None) if indi_a else None
             sex_a_char = str(sex_a).upper()[0] if sex_a and isinstance(sex_a, str) and str(sex_a).upper() in ("M", "F") else None

             label = "daughter" if sex_a_char == "F" else "son" if sex_a_char == "M" else "child"

        # Check if B is a child of A (B is the child of A)
        elif id_b in id_to_children.get(id_a, set()):
            # B is the child of A. Describe relationship A -> B
            sex_b = getattr(indi_b, "sex", None) if indi_b else None
            sex_b_char = str(sex_b).upper()[0] if sex_b and isinstance(sex_b, str) and str(sex_b).upper() in ("M", "F") else None

            label = "father" if sex_b_char == "M" else "mother" if sex_b_char == "F" else "parent"

        # Check if A and B are siblings
        else:
             parents_a = id_to_parents.get(id_a, set())
             parents_b = id_to_parents.get(id_b, set())
             # They are siblings if they share at least one parent AND are different people
             if parents_a and parents_b and (parents_a & parents_b) and id_a != id_b:
                 sex_a = getattr(indi_a, "sex", None) if indi_a else None
                 sex_a_char = str(sex_a).upper()[0] if sex_a and isinstance(sex_a, str) and str(sex_a).upper() in ("M", "F") else None

                 label = "sister" if sex_a_char == "F" else "brother" if sex_a_char == "M" else "sibling"
             else:
                 logger.warning(f"Could not determine direct relation between {id_a} ({name_a}) and {id_b} ({name_b}) for path explanation.")
                 label = "connected to" # Fallback label

        # Format the step explanation: "Person A is the [relationship] of Person B"
        # Use ordinal_formatter on the label
        steps.append(f"{name_a} is the {ordinal_formatter(label)} of {name_b}")

    # Join steps with arrows for display
    # Get the name of the starting person using _get_full_name (if available)
    start_person_indi = find_individual_by_id(reader, path_ids[0])
    start_person_name = (
        _get_full_name(start_person_indi)
        if _get_full_name and start_person_indi
        else f"Unknown ({path_ids[0]})"
    )
    explanation_str = "\n -> ".join(steps)

    # Return format: "Start Person\n -> Step 1\n -> Step 2..."
    # Add the start person's name on a separate line before the steps
    return f"{start_person_name}\n -> {explanation_str}"
# End of explain_relationship_path


def fast_bidirectional_bfs(
    start_id: str,
    end_id: str,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
    max_depth=25,
    node_limit=150000,
    timeout_sec=45,
    log_progress=False,
) -> List[str]:
    """
    Performs bidirectional BFS using pre-built maps and predecessors.
    Returns path as list of normalized IDs.
    Based on temp.py v7.36 logic, adjusted depth/limit/timeout slightly for safety.
    """
    start_time = time.time()
    from collections import deque

    if start_id == end_id:
        return [start_id]

    if not isinstance(id_to_parents, dict) or not isinstance(id_to_children, dict):
        logger.error(
            f"[FastBiBFS] Invalid map inputs: Parents={type(id_to_parents).__name__}, Children={type(id_to_children).__name__}"
        )
        return []

    # Initialize forward search queue and visited map (stores predecessors)
    queue_fwd: Deque[Tuple[str, int]] = deque([(start_id, 0)])  # (id, depth)
    visited_fwd: Dict[str, Optional[str]] = {start_id: None}  # {id: predecessor_id}

    # Initialize backward search queue and visited map (stores predecessors)
    queue_bwd: Deque[Tuple[str, int]] = deque([(end_id, 0)])
    visited_bwd: Dict[str, Optional[str]] = {end_id: None} # {id: predecessor_id}

    processed = 0 # Counter for nodes processed (rough estimate)
    meeting_id: Optional[str] = None # ID where forward and backward searches meet

    logger.debug(f"[FastBiBFS] Starting BFS: {start_id} <-> {end_id}")

    # Main BFS loop: continue as long as both queues have items AND no meeting point found
    while queue_fwd and queue_bwd and meeting_id is None:
        # --- Check limits ---
        if time.time() - start_time > timeout_sec:
            logger.warning(f"  [FastBiBFS] Timeout after {timeout_sec} seconds.")
            return [] # Return empty path on timeout

        if processed > node_limit:
            logger.warning(f"  [FastBiBFS] Node limit {node_limit} reached. Processed: ~{processed}.")
            return [] # Return empty path on node limit

        if log_progress and processed > 0 and processed % 10000 == 0:
             logger.info(f"[FastBiBFS] Progress: ~{processed} nodes, QF:{len(queue_fwd)}, QB:{len(queue_bwd)}")

        # --- Expand Forward Search (from start_id towards end_id) ---
        if queue_fwd:
            current_id_fwd, depth_fwd = queue_fwd.popleft()
            # Check if current_id_fwd is valid (should always be from queue, but defensive)
            if current_id_fwd is None: continue

            processed += 1 # Increment processed count

            # Skip if max depth reached in this direction
            if depth_fwd >= max_depth:
                continue

            # Check if the current node from the forward search has already been visited by the backward search
            if current_id_fwd in visited_bwd:
                meeting_id = current_id_fwd
                # We found the meeting point! visited_fwd[neighbor_id] = current_id_fwd was already set when the meeting node was added to queue_fwd
                logger.debug(f"  [FastBiBFS] Path found (FWD meets BWD) at {meeting_id} (Depth FWD: {depth_fwd}).")
                break # Exit the main while loop

            # Get neighbors (parents and children) of the current node in the forward direction
            neighbors_fwd = id_to_parents.get(current_id_fwd, set()) | id_to_children.get(current_id_fwd, set())

            for neighbor_id in neighbors_fwd:
                # Check if neighbor_id is valid (should come from maps, but defensive)
                if neighbor_id is None:
                    continue

                if neighbor_id not in visited_fwd:
                    visited_fwd[neighbor_id] = current_id_fwd
                    queue_fwd.append((neighbor_id, depth_fwd + 1))

                    # Check for intersection *after* adding to this search's visited set
                    # This check should be *inside* the `if neighbor_id not in visited_fwd:` block
                    if neighbor_id in visited_bwd:
                        meeting_id = neighbor_id
                        # The predecessor for the backward path was already set when the meeting node was added to queue_bwd
                         # The predecessor for the backward path was already set when the meeting node was added to queue_bwd
                        logger.debug(f"  [FastBiBFS] Path found (FWD adds node visited by BWD) at {meeting_id} (Depth FWD: {depth_fwd+1}).")
                        break # Exit the inner neighbor loop

            if meeting_id: # Check meeting_id after iterating through neighbors in the forward step
                break # Exit main while loop if intersection found

        # --- Expand Backward Search (from end_id towards start_id) ---
        # Only do this if the meeting point hasn't been found yet in the forward step
        if queue_bwd and meeting_id is None:
            current_id_bwd, depth_bwd = queue_bwd.popleft()
             # Check if current_id_bwd is valid (should always be from queue, but defensive)
            if current_id_bwd is None: continue

            processed += 1 # Increment processed count

            # Skip if max depth reached in this direction
            if depth_bwd >= max_depth:
                continue

            # Check if the current node from the backward search has already been visited by the forward search
            if current_id_bwd in visited_fwd:
                meeting_id = current_id_bwd
                # We found the meeting point! visited_bwd[neighbor_id] = current_id_bwd was already set when the meeting node was added to queue_bwd
                logger.debug(f"  [FastBiBFS] Path found (BWD meets FWD) at {meeting_id} (Depth BWD: {depth_bwd}).")
                break # Exit the main while loop

            # Get neighbors (parents and children) of the current node in the backward direction
            # Note: Parent/child relationship is symmetric in the graph, so neighbors are the same
            neighbors_bwd = id_to_parents.get(current_id_bwd, set()) | id_to_children.get(current_id_bwd, set())

            for neighbor_id in neighbors_bwd:
                # Check if neighbor_id is valid (should come from maps, but defensive)
                if neighbor_id is None:
                    continue

                if neighbor_id not in visited_bwd:
                    visited_bwd[neighbor_id] = current_id_bwd
                    queue_bwd.append((neighbor_id, depth_bwd + 1))

                    # Check for intersection *after* adding to this search's visited set
                    # This check should be *inside* the `if neighbor_id not in visited_bwd:` block
                    if neighbor_id in visited_fwd:
                        meeting_id = neighbor_id
                        # The predecessor for the forward path was already set when the meeting node was added to queue_fwd
                        logger.debug(f"  [FastBiBFS] Path found (BWD adds node visited by FWD) at {meeting_id} (Depth BWD: {depth_bwd+1}).")
                        break # Exit the inner neighbor loop

            # No need to check meeting_id again here, the next iteration of the main while loop will catch it before the forward step

    # --- Reconstruct Path if intersection found --- #
    if meeting_id:
        logger.debug(f"[FastBiBFS] Intersection found at {meeting_id}. Reconstructing path...")
        path_ids = _reconstruct_path(
            start_id, end_id, meeting_id, visited_fwd, visited_bwd
        )
        logger.debug(f"[FastBiBFS] Path reconstruction complete. Length: {len(path_ids)}")
        return path_ids
    else:
        # No path found within limits or queues exhausted
        reason = "Queues Emptied"
        # Use find_individual_by_id (if available) to check if nodes exist in index for better reason
        start_node_in_index = find_individual_by_id and find_individual_by_id(None, start_id)
        end_node_in_index = find_individual_by_id and find_individual_by_id(None, end_id)
        start_node_in_maps = start_id in id_to_parents or start_id in id_to_children
        end_node_in_maps = end_id in id_to_parents or end_id in id_to_children


        if time.time() - start_time > timeout_sec:
            reason = "Timeout"
        elif processed > node_limit:
            reason = "Node Limit Reached"
        elif not start_node_in_maps and start_node_in_index:
             reason = f"Start Node {start_id} found in index but not in relationship maps"
        elif not end_node_in_maps and end_node_in_index:
             reason = f"End Node {end_id} found in index but not in relationship maps"
        elif not start_node_in_maps and not start_node_in_index:
             reason = f"Start Node {start_id} not found in maps or index"
        elif not end_node_in_maps and not end_node_in_index:
             reason = f"End Node {end_id} not found in maps or index"
        elif not start_node_in_maps:
             reason = f"Start Node {start_id} not found in relationship maps"
        elif not end_node_in_maps:
             reason = f"End Node {end_id} not found in relationship maps"


        logger.warning(
            f"[FastBiBFS] No path found between {start_id} and {end_id}. Reason: {reason}. Processed ~{processed} nodes."
        )
        return [] # Return empty list if no path found
# End of fast_bidirectional_bfs


def get_relationship_path(reader, id1: str, id2: str) -> str:
    """
    Calculates and formats relationship path using fast bidirectional BFS with pre-built maps.
    Based on temp.py v7.36 logic.
    """
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
    if not FAMILY_MAPS_CACHE:
        logger.debug(f"[Cache] Building family maps (first time)...")
        # Check if build_family_maps is available before calling
        if build_family_maps:
            FAMILY_MAPS_CACHE = build_family_maps(reader)
        else:
             logger.error("build_family_maps function is not available. Cannot build relationship maps.")
             return "Error: Relationship maps could not be built."

        # build_family_maps logs its build time
    # build_indi_index should ideally be called before build_family_maps or separately by the main script.
    # Ensure it's built for find_individual_by_id lookup within path explanation
    if not INDI_INDEX:
        logger.debug(f"[Cache] Building individual index (first time)...")
        # Check if build_indi_index is available before calling
        if build_indi_index:
            build_indi_index(reader)
        else:
             logger.error("build_indi_index function is not available. Cannot build index.")
             # Continue anyway, index is not strictly required for BFS but is for explanation


    # Ensure maps were successfully built
    if FAMILY_MAPS_CACHE is None:
        return "Error: Family relationship maps could not be built or retrieved."
    # INDI_INDEX is not strictly required for BFS but is needed for explain_relationship_path.
    # We'll proceed with BFS even if index is missing, and explanation might show IDs instead of names.


    id_to_parents, id_to_children = FAMILY_MAPS_CACHE

    # Check if start/end nodes exist in the maps (quick validation)
    # This check isn't critical as BFS handles missing nodes, but can provide early warning
    if INDI_INDEX and id1_norm and id1_norm in INDI_INDEX and not (id1_norm in id_to_parents or id1_norm in id_to_children):
         # Use _get_full_name if available
         name_1_debug = (_get_full_name(INDI_INDEX[id1_norm]) if _get_full_name else f"ID {id1_norm}")
         logger.warning(f"Start node {id1_norm} ({name_1_debug}) found in index but not in relationship maps.")
    elif id1_norm and not (id1_norm in id_to_parents or id1_norm in id_to_children):
          logger.debug(f"Start node {id1_norm} not found in relationship maps.")

    if INDI_INDEX and id2_norm and id2_norm in INDI_INDEX and not (id2_norm in id_to_parents or id2_norm in id_to_children):
         # Use _get_full_name if available
         name_2_debug = (_get_full_name(INDI_INDEX[id2_norm]) if _get_full_name else f"ID {id2_norm}")
         logger.warning(f"End node {id2_norm} ({name_2_debug}) found in index but not in relationship maps.")
    elif id2_norm and not (id2_norm in id_to_parents or id2_norm in id_to_children):
         logger.debug(f"End node {id2_norm} not found in relationship maps.")


    # Default search parameters (can potentially be moved to config)
    max_depth = 25 # Increased slightly from temp.py default
    node_limit = 150000 # Increased slightly from temp.py default
    timeout_sec = 45 # Increased slightly from temp.py default


    logger.debug(
        f"Calculating relationship path (FastBiBFS): {id1_norm} <-> {id2_norm}"
    )
    logger.debug(f"  [FastBiBFS] Using cached maps. Starting search...")
    search_start = time.time()

    # --- Perform BFS Search (Returns list of IDs) ---
    # Check if fast_bidirectional_bfs is available
    if fast_bidirectional_bfs:
        path_ids = fast_bidirectional_bfs(
            id1_norm,
            id2_norm,
            id_to_parents,
            id_to_children,
            max_depth,
            node_limit,
            timeout_sec,
            log_progress=False, # Set to True for detailed progress logs
        )
    else:
        logger.error("fast_bidirectional_bfs function is not available. Cannot perform BFS.")
        path_ids = [] # No path found

    search_time = time.time() - search_start
    logger.debug(f"[PROFILE] BFS search completed in {search_time:.2f}s.")

    # Handle case where no path is found
    if not path_ids:
        # Use the latest build times available (could be 0 if already cached)
        current_maps_build_time = FAMILY_MAPS_BUILD_TIME if FAMILY_MAPS_BUILD_TIME is not None else 0 # Handle None if build failed
        current_index_build_time = INDI_INDEX_BUILD_TIME if INDI_INDEX_BUILD_TIME is not None else 0 # Handle None if build failed

        profile_info = (
             f"[PROFILE] Search: {search_time:.2f}s, "
             f"Maps: {current_maps_build_time:.2f}s, "
             f"Index: {current_index_build_time:.2f}s"
        )
        return (
            f"No relationship path found (FastBiBFS could not connect).\n{profile_info}"
        )

    # --- Explain the found path ---
    explanation_start = time.time()
    # Check if explain_relationship_path is available
    if explain_relationship_path:
        explanation_str = explain_relationship_path(
            path_ids, reader, id_to_parents, id_to_children
        )
    else:
        logger.error("explain_relationship_path function is not available. Cannot explain path.")
        explanation_str = "Error: Cannot explain relationship path (function missing)."
        # Fallback to just listing IDs if explanation is missing
        if path_ids: explanation_str += "\nPath IDs: " + " -> ".join(path_ids)


    explanation_time = time.time() - explanation_start
    logger.debug(f"[PROFILE] Path explanation built in {explanation_time:.2f}s.")

    # --- Format final output ---
    # Combine times, accounting for potential 0 build times if cached on first run
    total_build_time = (FAMILY_MAPS_BUILD_TIME if FAMILY_MAPS_BUILD_TIME is not None else 0) + (INDI_INDEX_BUILD_TIME if INDI_INDEX_BUILD_TIME is not None else 0) # Handle None
    total_process_time = search_time + explanation_time + total_build_time # Include build time in total

    profile_info = (
        f"[PROFILE] Total Time: {total_process_time:.2f}s "
        f"(Maps: {FAMILY_MAPS_BUILD_TIME:.2f}s, Index: {INDI_INDEX_BUILD_TIME:.2f}s, " # Use global vars directly for profile info display
        f"Search: {search_time:.2f}s, Explain: {explanation_time:.2f}s)"
    )
    logger.debug(profile_info)

    return f"{explanation_str}\n{profile_info}" # Include profile info in output for now
# End of get_relationship_path


# --- Fuzzy Matching and Scoring ---

def calculate_match_score(
    search_criteria: Dict,
    candidate_data: Dict,
    scoring_weights: Dict,
    name_flexibility: Dict,
    date_flexibility: Dict
) -> Tuple[float, List[str]]:
    """
    Calculates a match score between search criteria and candidate data.
    Uses weights and flexibility settings from config.
    Returns score and list of reasons.
    Based on the scoring logic within temp.py v7.36 find_potential_matches.
    """
    score = 0.0
    match_reasons: List[str] = []

    # Use default empty dicts if None is passed, although the calling functions
    # should ensure these are Dicts from config.
    weights = scoring_weights if scoring_weights is not None else {}
    date_flex = date_flexibility if date_flexibility is not None else {"year_match_range": 1}
    name_flex = name_flexibility if name_flexibility is not None else {"fuzzy_threshold": 0.8} # check_starts_with not used in scoring function

    year_score_range = date_flex.get("year_match_range", 1)
    fuzzy_threshold = name_flex.get("fuzzy_threshold", 0.8)

    # Extract lowercased/parsed values for comparison
    # Ensure keys exist before accessing them and handle potential None values
    target_first_name_lower = search_criteria.get('first_name')
    target_surname_lower = search_criteria.get('surname')
    target_birth_year = search_criteria.get('birth_year')
    target_birth_date_obj = search_criteria.get('birth_date_obj')
    target_pob_lower = search_criteria.get('birth_place')
    target_death_year = search_criteria.get('death_year')
    target_death_date_obj = search_criteria.get('death_date_obj')
    target_pod_lower = search_criteria.get('death_place')
    target_gender_clean = search_criteria.get('gender') # m or f or None

    # Ensure keys exist before accessing them and handle potential None values
    c_first_name_lower = candidate_data.get('first_name')
    c_surname_lower = candidate_data.get('surname')
    c_birth_year = candidate_data.get('birth_year')
    c_birth_date_obj = candidate_data.get('birth_date_obj')
    c_birth_place_lower = candidate_data.get('birth_place') # Assumed already lowercased from GEDCOM/API parsing
    c_death_year = candidate_data.get('death_year')
    c_death_date_obj = candidate_data.get('death_date_obj')
    c_death_place_lower = candidate_data.get('death_place') # Assumed already lowercased from GEDCOM/API parsing
    c_gender_clean = candidate_data.get('gender') # m or f or None


    # --- Scoring Logic ---

    # 1. Name Scoring
    first_name_match = (target_first_name_lower is not None and c_first_name_lower is not None and target_first_name_lower == c_first_name_lower)
    surname_match = (target_surname_lower is not None and c_surname_lower is not None and target_surname_lower == c_surname_lower)

    if first_name_match and surname_match:
        # Exact Full Name Path (High Score)
        score += weights.get("exact_first_name", 20)
        match_reasons.append("Exact First")
        score += weights.get("exact_surname", 20)
        match_reasons.append("Exact Surname")
        score += weights.get("boost_exact_full_name", 20)
        match_reasons.append("Boost Exact Name")

        # Date/Year Scoring in Exact Name Path
        birth_year_bonus_match = False
        if target_birth_date_obj and c_birth_date_obj and target_birth_date_obj.date() == c_birth_date_obj.date():
            score += weights.get("exact_birth_date", 20)
            match_reasons.append("Exact Birth Date")
        # Check if years are integers before comparison
        elif target_birth_year is not None and c_birth_year is not None and isinstance(c_birth_year, int) and isinstance(target_birth_year, int) and abs(c_birth_year - target_birth_year) <= year_score_range:
            score += weights.get("year_birth", 15)
            birth_year_bonus_match = True
            match_reasons.append(f"Birth Year ~{target_birth_year} ({c_birth_year})")

        if target_death_date_obj and c_death_date_obj and target_death_date_obj.date() == c_death_date_obj.date():
            score += weights.get("exact_death_date", 20)
            match_reasons.append("Exact Death Date")
        # Check if years are integers before comparison
        elif target_death_year is not None and c_death_year is not None and isinstance(c_death_year, int) and isinstance(target_death_year, int) and abs(c_death_year - target_death_year) <= year_score_range:
            score += weights.get("year_death", 15)
            match_reasons.append(f"Death Year ~{target_death_year} ({c_death_year})")
        # Both absent check needs to be careful with None vs 'N/A' strings vs actual data
        # Check if both search and candidate have no parsed date objects and no year integers
        elif target_death_date_obj is None and c_death_date_obj is None and target_death_year is None and c_death_year is None:
            score += weights.get("death_dates_both_absent", 5)
            match_reasons.append("Death Dates Both Absent")


        # Place Scoring in Exact Name Path
        if target_pob_lower is not None and c_birth_place_lower is not None and target_pob_lower in c_birth_place_lower:
            score += weights.get("contains_pob", 15)
            match_reasons.append(f"POB contains '{target_pob_lower}'")
        if target_pod_lower is not None and c_death_place_lower is not None and target_pod_lower in c_death_place_lower:
            score += weights.get("contains_pod", 15)
            match_reasons.append(f"POD contains '{target_pod_lower}'")

        # Gender Scoring in Exact Name Path
        gender_bonus_match = False
        if target_gender_clean is not None and c_gender_clean is not None and target_gender_clean == c_gender_clean:
            score += weights.get("gender_match", 20)
            gender_bonus_match = True
            match_reasons.append(f"Gender ({target_gender_clean.upper()})")
        elif target_gender_clean is not None and c_gender_clean is not None and target_gender_clean != c_gender_clean:
            score += weights.get("gender_mismatch_penalty", -20)
            match_reasons.append(f"Gender Mismatch ({c_gender_clean.upper()} vs {target_gender_clean.upper()})")

        # Boosts in Exact Name Path
        if birth_year_bonus_match: # Boost for exact name + year match
             score += weights.get("boost_exact_name_year", 2)
             match_reasons.append("Boost Exact Name + Year")

    else:
        # Fuzzy/Other Match Path (Lower Scores)
        name_score = 0.0
        fuzzy_first = 0.0
        fuzzy_surname = 0.0

        # Calculate fuzzy similarity ratios if needed
        if target_first_name_lower is not None and c_first_name_lower is not None:
            fuzzy_first = difflib.SequenceMatcher(None, target_first_name_lower, c_first_name_lower).ratio()
        if target_surname_lower is not None and c_surname_lower is not None:
            fuzzy_surname = difflib.SequenceMatcher(None, target_surname_lower, c_surname_lower).ratio()

        # Score First Name (Exact preferred, then Fuzzy)
        if first_name_match:
            name_score += weights.get("exact_first_name", 20) # Use exact weight even in fuzzy path if it's an exact match
            match_reasons.append("Exact First")
        elif fuzzy_first >= fuzzy_threshold:
            delta = weights.get("fuzzy_first_name", 15) * fuzzy_first
            name_score += delta
            match_reasons.append(f"Fuzzy First ({fuzzy_first:.2f})")
        # Note: check_starts_with logic from temp.py is removed here, using only exact/fuzzy threshold

        # Score Surname (Exact preferred, then Fuzzy)
        if surname_match:
            name_score += weights.get("exact_surname", 20) # Use exact weight even in fuzzy path if it's an exact match
            match_reasons.append("Exact Surname")
        elif fuzzy_surname >= fuzzy_threshold:
            delta = weights.get("fuzzy_surname", 15) * fuzzy_surname
            name_score += delta
            match_reasons.append(f"Fuzzy Surname ({fuzzy_surname:.2f})")
        # Note: check_starts_with logic from temp.py is removed here

        # Only proceed with other scoring if there's at least some name match score
        # This prevents scoring people with completely different names just because their dates/places match
        if name_score > 0:
            score += name_score # Add the name score

            # Date/Year Scoring in Fuzzy Path (Lower Points)
            if target_birth_date_obj and c_birth_date_obj and target_birth_date_obj.date() == c_birth_date_obj.date():
                 score += weights.get("exact_birth_date", 20) # Exact date match is still strong
                 match_reasons.append("Exact Birth Date")
            # Check if years are integers before comparison
            elif target_birth_year is not None and c_birth_year is not None and isinstance(c_birth_year, int) and isinstance(target_birth_year, int) and abs(c_birth_year - target_birth_year) <= year_score_range:
                score += weights.get("year_birth_fuzzy", 5) # Use fuzzy year weight
                match_reasons.append(f"Birth Year ~{target_birth_year} ({c_birth_year})")

            if target_death_date_obj and c_death_date_obj and target_death_date_obj.date() == c_death_date_obj.date():
                 score += weights.get("exact_death_date", 20) # Exact date match is still strong
                 match_reasons.append("Exact Death Date")
            # Check if years are integers before comparison
            elif target_death_year is not None and c_death_year is not None and isinstance(c_death_year, int) and isinstance(target_death_year, int) and abs(c_death_year - target_death_year) <= year_score_range:
                score += weights.get("year_death_fuzzy", 5) # Use fuzzy year weight
                match_reasons.append(f"Death Year ~{target_death_year} ({c_death_year})")
            # Both absent check
            elif target_death_date_obj is None and c_death_date_obj is None and target_death_year is None and c_death_year is None:
                score += weights.get("death_dates_both_absent", 5)
                match_reasons.append("Death Dates Both Absent")


            # Place Scoring in Fuzzy Path (Lower Points)
            if target_pob_lower is not None and c_birth_place_lower is not None and target_pob_lower in c_birth_place_lower:
                score += weights.get("contains_pob_fuzzy", 1)
                match_reasons.append(f"POB contains '{target_pob_lower}'")
            if target_pod_lower is not None and c_death_place_lower is not None and target_pod_lower in c_death_place_lower:
                score += weights.get("contains_pod_fuzzy", 1)
                match_reasons.append(f"POD contains '{target_pod_lower}'")

            # Gender Scoring in Fuzzy Path (Lower Points)
            if target_gender_clean is not None and c_gender_clean is not None and target_gender_clean == c_gender_clean:
                score += weights.get("gender_match_fuzzy", 3)
                match_reasons.append(f"Gender ({target_gender_clean.upper()})")
            elif target_gender_clean is not None and c_gender_clean is not None and target_gender_clean != c_gender_clean:
                 score += weights.get("gender_mismatch_penalty_fuzzy", -3)
                 match_reasons.append(f"Gender Mismatch ({c_gender_clean.upper()} vs {target_gender_clean.upper()})")


        # If name_score was 0, score remains 0. No points for non-name matches alone.

    # Return rounded score and sorted unique reasons
    return round(score), sorted(list(set(match_reasons)))
# End of calculate_match_score


def find_potential_matches(
    reader,
    first_name: Optional[str],
    surname: Optional[str],
    dob_str: Optional[str], # Birth date string
    pob: Optional[str], # Birth place
    dod_str: Optional[str], # Death date string
    pod: Optional[str], # Death place
    gender: Optional[str] = None,
    max_results: int = 10, # Fetch more initially for internal sorting/filtering
    scoring_weights: Optional[Dict] = None,
    name_flexibility: Optional[Dict] = None,
    date_flexibility: Optional[Dict] = None,
) -> List[Dict]:
    """
    Finds potential matches in GEDCOM based on various criteria using fuzzy matching.
    Uses the common calculate_match_score function for scoring.
    Returns a list of match dictionaries.
    Based on temp.py v7.36 logic.
    """
    if not reader:
        logger.error("find_potential_matches: No reader.")
        return []

    # Ensure caches/indexes are built (build_indi_index and build_family_maps should be called by main script)
    global INDI_INDEX
    if not INDI_INDEX:
        # Fallback to building index if it hasn't been (less efficient if main didn't do it)
        logger.warning("INDI_INDEX not built, attempting build in find_potential_matches.")
        # Check if build_indi_index is available before calling
        if build_indi_index:
             build_indi_index(reader)
        else:
             logger.error("build_indi_index function is not available. Cannot build index.")
             return []

        if not INDI_INDEX:
             logger.error("INDI_INDEX build failed or is empty. Cannot search.")
             return []

    # Check if scoring function and configurations are available
    if calculate_match_score is None or scoring_weights is None or name_flexibility is None or date_flexibility is None:
        logger.error("Scoring function or configurations not available. Cannot perform fuzzy match scoring.")
        # We could potentially return matches with score 0 and no reasons, but if the core logic is missing,
        # it's safer to return an empty list.
        return []


    weights = scoring_weights
    date_flex = date_flexibility
    name_flex = name_flexibility

    year_filter_range = 30 # Pre-filter: only consider individuals within 30 years of target year
    clean_param = lambda p: p.strip().lower() if p and isinstance(p, str) else None # Clean and lowercase input strings

    target_first_name_lower = clean_param(first_name)
    target_surname_lower = clean_param(surname)
    target_pob_lower = clean_param(pob)
    target_pod_lower = clean_param(pod)
    target_gender_clean = (
        gender.strip().lower()[0]
        if gender and isinstance(gender, str) and gender.strip().lower() in ("m", "f")
        else None
    )

    # Parse target dates/years using imported helpers (if helpers are available)
    target_birth_year: Optional[int] = None
    target_birth_date_obj: Optional[datetime] = None
    if dob_str and _parse_date:
        target_birth_date_obj = _parse_date(dob_str)
        if target_birth_date_obj: target_birth_year = target_birth_date_obj.year

    target_death_year: Optional[int] = None
    target_death_date_obj: Optional[datetime] = None
    if dod_str and _parse_date:
        target_death_date_obj = _parse_date(dod_str)
        if target_death_date_obj: target_death_year = target_death_date_obj.year


    # Prepare search criteria dictionary for scoring function
    search_criteria_dict = {
        'first_name': target_first_name_lower,
        'surname': target_surname_lower,
        'birth_year': target_birth_year,
        'birth_date_obj': target_birth_date_obj,
        'birth_place': target_pob_lower,
        'death_year': target_death_year,
        'death_date_obj': target_death_date_obj,
        'death_place': target_pod_lower,
        'gender': target_gender_clean,
    }

    # Check if any search criteria provided (excluding gender alone)
    # A search requires at least one of name, date, or place. Gender alone is not sufficient.
    if not any([target_first_name_lower, target_surname_lower, target_birth_year, target_pob_lower, target_death_year, target_pod_lower]) and target_gender_clean is None:
         logger.warning("Fuzzy search called with no sufficient criteria.")
         return []
    # If only gender is provided, require at least one other criteria
    if target_gender_clean is not None and not any([target_first_name_lower, target_surname_lower, target_birth_year, target_pob_lower, target_death_year, target_pod_lower]):
         logger.warning("Fuzzy search called with only Gender criteria. No name, date, or place provided. Refusing to search.")
         return []


    candidate_count = 0
    scored_results = []

    # Iterate through individuals using the index
    individuals_to_check = INDI_INDEX.values()

    for indi in individuals_to_check:
        candidate_count += 1
        indi_id_str = "Unknown ID" # Default for logging if ID extraction fails
        try:
            # Basic validation
            if not _is_individual(indi) or not hasattr(indi, "xref_id") or not indi.xref_id:
                continue # Skip invalid records

            # Use extract_and_fix_id (if available)
            if extract_and_fix_id:
                 indi_id_norm = extract_and_fix_id(indi.xref_id)
            else:
                 indi_id_norm = _normalize_id(indi.xref_id) # Fallback to basic normalize
                 logger.warning("extract_and_fix_id not available. Using basic _normalize_id for search.")

            indi_id_raw = f"@{indi.xref_id}@" if indi.xref_id else None # Keep original raw ID for result dict
            indi_id_str = indi_id_raw or "Unknown ID" # Update for logging


            # Use _get_full_name (if available)
            if _get_full_name:
                indi_full_name = _get_full_name(indi)
                if indi_full_name.startswith("Unknown"):
                     continue # Skip unknown names
            else:
                 # Fallback if _get_full_name is not available
                 indi_full_name = getattr(indi, 'name', None)
                 if indi_full_name is None or not isinstance(indi_full_name, str):
                      indi_full_name = indi_id_str # Use ID as name fallback
                 else:
                      indi_full_name = str(indi_full_name) # Ensure it's a string
                 logger.warning("_get_full_name not available. Using basic name getter for search.")


            # Get event info for the individual using helper functions (_get_event_info needs to be available)
            if _get_event_info:
                birth_date_obj, birth_date_str_ged, birth_place_str_ged_raw = _get_event_info(indi, "BIRT")
                death_date_obj, death_date_str_ged, death_place_str_ged_raw = _get_event_info(indi, "DEAT")
            else:
                 # Fallback if _get_event_info is not available
                 birth_date_obj, birth_date_str_ged, birth_place_str_ged_raw = None, 'N/A', 'N/A'
                 death_date_obj, death_date_str_ged, death_place_str_ged_raw = None, 'N/A', 'N/A'
                 logger.warning("_get_event_info not available. Cannot extract date/place for scoring/display.")

            birth_year_ged: Optional[int] = birth_date_obj.year if birth_date_obj else None
            death_year_ged: Optional[int] = death_date_obj.year if death_date_obj else None

            # Pre-filtering based on years (broad range) - Only filter if year parsing was successful
            if birth_year_ged is not None and target_birth_year is not None and isinstance(birth_year_ged, int) and isinstance(target_birth_year, int) and abs(birth_year_ged - target_birth_year) > year_filter_range:
                continue
            elif death_year_ged is not None and target_death_year is not None and isinstance(death_year_ged, int) and isinstance(target_death_year, int) and abs(death_year_ged - target_death_year) > year_filter_range:
                 continue
            # If no year is provided in search or candidate, this pre-filter is skipped.

            # Prepare candidate data dictionary for scoring function
            indi_name_parts = indi_full_name.lower().split()
            c_first_name_lower = indi_name_parts[0] if indi_name_parts else None
            c_surname_lower = indi_name_parts[-1] if len(indi_name_parts) > 1 else None
            indi_gender_raw = getattr(indi, "sex", None)
            c_gender_clean = str(indi_gender_raw).strip().lower()[0] if indi_gender_raw and isinstance(indi_gender_raw, str) and str(indi_gender_raw).strip().lower() in ("m", "f") else None

            candidate_data_dict = {
                'first_name': c_first_name_lower,
                'surname': c_surname_lower,
                'birth_year': birth_year_ged,
                'birth_date_obj': birth_date_obj,
                'birth_place': birth_place_str_ged_raw.lower() if birth_place_str_ged_raw != 'N/A' else None,
                'death_year': death_year_ged,
                'death_date_obj': death_date_obj,
                'death_place': death_place_str_ged_raw.lower() if death_place_str_ged_raw != 'N/A' else None,
                'gender': c_gender_clean,
            }

            # Calculate score using the common scoring function
            # Check again if calculate_match_score is available
            if calculate_match_score:
                score, reasons = calculate_match_score(
                    search_criteria_dict,
                    candidate_data_dict,
                    weights,
                    name_flex,
                    date_flex
                )
            else:
                 # Fallback score if function is missing
                 score = 0
                 reasons = ["Scoring function unavailable"]
                 logger.warning("calculate_match_score not available. Skipping scoring.")


            # Only add results with a score greater than 0
            if score > 0:
                 # Clean display dates using _clean_display_date (if available)
                 display_birth_date = _clean_display_date(birth_date_str_ged) if _clean_display_date else str(birth_date_str_ged)
                 display_death_date = _clean_display_date(death_date_str_ged) if _clean_display_date else str(death_date_str_ged)

                 scored_results.append(
                     {
                         "id": indi_id_raw, # Store the original raw ID
                         "norm_id": indi_id_norm, # Store normalized ID
                         "name": indi_full_name,
                         "birth_date": display_birth_date,
                         "birth_place": (birth_place_str_ged_raw if birth_place_str_ged_raw != "N/A" else None),
                         "death_date": display_death_date,
                         "death_place": (death_place_str_ged_raw if death_place_str_ged_raw != "N/A" else None),
                         "score": round(score), # Round score for display
                         "reasons": ", ".join(reasons) if reasons else "Score > 0",
                     }
                 )

        except Exception as loop_err:
            # Log errors for specific individuals but continue with the rest
            logger.error(
                f"!!! ERROR processing individual {indi_id_str} in find_potential_matches !!! Error: {loop_err}",
                exc_info=True, # Log traceback
            )
            continue # Move to the next individual

    # --- Final Result Sorting and Limiting ---
    logger.debug(
        f"Finished processing {candidate_count} individuals. Found {len(scored_results)} potential matches with score > 0."
    )

    # Sort results primarily by score (descending), then by birth date (ascending)
    # Use a robust key for sorting by birth date, handling None dates (using _parse_date if available)
    if _parse_date:
        scored_results.sort(key=lambda x: (
            x["score"],
            _parse_date(x.get("birth_date")) or datetime.max.replace(tzinfo=timezone.utc) # None dates sort last
            ),
            reverse=True # Score descending
        )
    else:
        # Fallback sort if date parsing is unavailable - sort only by score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        logger.warning("Date parsing unavailable, sorting results by score only.")


    # Limit the final number of results to display
    limited_results = scored_results[:max_results]

    logger.info(
        f"find_potential_matches returning top {len(limited_results)} of {len(scored_results)} total scored matches."
    )
    return limited_results
# End of find_potential_matches


# --- REMOVED V15.3: find_gedcom_root_individual function ---

# --- calculate_match_score function is defined above ---

# --- Standalone Test Block ---
def self_check(verbose: bool = True) -> bool:
    """Performs internal self-checks for gedcom_utils.py."""
    status = True
    messages = []

    # Check if core dependencies loaded correctly
    messages.append(f"Logger: {'OK' if 'logger' in globals() and isinstance(logger, logging.Logger) else 'FAILED'}")
    if 'logger' not in globals() or not isinstance(logger, logging.Logger): status = False


    # Check imports from utils
    try:
        # Check if format_name and ordinal_case are imported from utils
        if (
            getattr(format_name, '__module__', None) == 'utils'
            and getattr(ordinal_case, '__module__', None) == 'utils'
        ):
            messages.append("Utils Imports (format_name, ordinal_case): OK")
        else:
            messages.append("Utils Imports (format_name, ordinal_case): FAILED (Using fallbacks)")
            status = False # Consider failed if fallbacks are active
    except NameError: # This happens if utils import failed entirely
         messages.append("Utils Imports (format_name, ordinal_case): FAILED (Module import failed)")
         status = False


    # Check if key ged4py classes are available 
    messages.append(f"GedcomReader Class: {'Available' if 'GedcomReader' in globals() and GedcomReader is not None else 'Unavailable'}")
    if 'GedcomReader' not in globals() or GedcomReader is None: status = False # Fails if ged4py failed

    messages.append(f"Individual Class: {'Available' if 'Individual' in globals() and Individual is not None else 'Unavailable'}")
    if 'Individual' not in globals() or Individual is None: status = False # Fails if ged4py failed

    messages.append(f"Record Class: {'Available' if 'Record' in globals() and Record is not None else 'Unavailable'}")
    if 'Record' not in globals() or Record is None: status = False # Fails if ged4py failed

    messages.append(f"Name Class: {'Available' if 'Name' in globals() and Name is not None else 'Unavailable'}")
    if 'Name' not in globals() or Name is None: status = False # Fails if ged4py failed


    # Check if key utility functions are defined
    # Note: Some functions might be defined but rely on ged4py classes/objects
    key_functions = [
        ('_is_individual', _is_individual),
        ('_normalize_id', _normalize_id),
        ('extract_and_fix_id', extract_and_fix_id),
        ('_get_full_name', _get_full_name),
        ('_parse_date', _parse_date), # Check if these date helpers are defined (even if fallbacks)
        ('_clean_display_date', _clean_display_date), # Check if these date helpers are defined (even if fallbacks)
        ('_get_event_info', _get_event_info),
        ('format_life_dates', format_life_dates),
        ('format_full_life_details', format_full_life_details),
        ('format_relative_info', format_relative_info),
        ('build_indi_index', build_indi_index), # Cache building
        ('build_family_maps', build_family_maps), # Cache building
        ('find_individual_by_id', find_individual_by_id), # Lookup
        ('_find_family_records_where_individual_is_child', _find_family_records_where_individual_is_child), # Relational helpers
        ('_find_family_records_where_individual_is_parent', _find_family_records_where_individual_is_parent), # Relational helpers
        ('get_related_individuals', get_related_individuals), # Relational retrieval
        ('_reconstruct_path', _reconstruct_path), # Path building helpers
        ('explain_relationship_path', explain_relationship_path), # Path explanation
        ('fast_bidirectional_bfs', fast_bidirectional_bfs), # Path search algorithm
        ('get_relationship_path', get_relationship_path), # Main path function
        ('calculate_match_score', calculate_match_score), # Scoring function
        ('find_potential_matches', find_potential_matches), # Fuzzy search
    ]

    messages.append("\n--- Function Definitions ---")
    for func_name, func_obj in key_functions:
        messages.append(f"  {func_name}: {'DEFINED' if func_obj is not None and callable(func_obj) else 'MISSING'}")
        if func_obj is None or not callable(func_obj): status = False


    messages.append("\n--- Note ---")
    messages.append("  Functional tests (requiring a GEDCOM file) are not performed in this self-check.")
    messages.append("  This check primarily verifies module imports, ged4py availability, and function definitions.")


    if verbose:
        print("\n[gedcom_utils.py self-check results]")
        for msg in messages:
            print("-", msg)
        print(f"Self-check status: {'PASS' if status else 'FAIL'}\n")

    # Log the final status
    if status:
        logger.info("gedcom_utils self-check status: PASS")
    else:
        logger.error("gedcom_utils self-check status: FAIL")

    return status
# End of self_check


if __name__ == "__main__":
    # Note: This standalone test only checks if functions are defined and imports work.
    # It does NOT perform functional tests that require a GEDCOM file or mock data.
    # A full test suite would be needed for comprehensive functional testing.
    print("Running gedcom_utils.py self-check...")
    self_check(verbose=True)
    print("\nThis is the gedcom_utils module. Import it into other scripts.")

    # --- Improved functional self-test using .env and GEDCOM file ---
    print("\n--- Functional Self-Test (using .env and GEDCOM) ---")
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        gedcom_path = os.getenv("GEDCOM_FILE_PATH")
        my_profile_id = os.getenv("MY_PROFILE_ID")
        tree_name = os.getenv("TREE_NAME")
        tree_owner = os.getenv("TREE_OWNER_NAME")
        print(f"Tree Name: {tree_name}, Owner: {tree_owner}")
        print(f"GEDCOM file: {gedcom_path}")
        if not gedcom_path or not os.path.exists(gedcom_path):
            print("[Functional Test] GEDCOM file not found. Skipping functional tests.")
        else:
            from ged4py.parser import GedcomReader
            reader = GedcomReader(gedcom_path)
            if my_profile_id:
                indi = find_individual_by_id(reader, my_profile_id)
                if indi:
                    print(f"[Functional Test] Profile ID {my_profile_id} found.")
                    birth, death = format_full_life_details(indi)
                    print(f"  Life details: {birth} {death}")
                    parents = get_related_individuals(reader, indi, "parents")
                    if parents:
                        print("  Parents:")
                        for p in parents:
                            pname, _ = format_full_life_details(p)
                            print(f"    - {pname}")
                    else:
                        print("  Parents: None found")
                else:
                    print(f"[Functional Test] Profile ID {my_profile_id} not found in GEDCOM.")
            else:
                print("[Functional Test] MY_PROFILE_ID is not set in .env.")
    except Exception as e:
        print(f"[Functional Test] Exception during functional test: {e}")
    print("--- End of Functional Self-Test ---\n")
