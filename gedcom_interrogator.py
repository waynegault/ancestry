# gedcom_interrogator.py (v7.33 - Added Ancestry API integration)
"""
Standalone script to load and query a GEDCOM file specified in the config.
Handles potential TypeError during record reads gracefully. Uses fuzzy match for user searches.
Integrated with Ancestry API search functionality for extended data access.
NOTE: Actions involving specific individuals (e.g., WGG, Hamish, Hannah, Fraser)
      may fail if their records trigger TypeError during read_record, likely due
      to GEDCOM file issues or ged4py library limitations. This script aims
      to function for *other* records while handling errors for affected ones.
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
import json
import requests
import urllib.parse

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
        return
    start = time.time()
    logger.info("[Cache] Building INDI index...")
    count = 0
    for indi in reader.records0("INDI"):
        if _is_individual(indi) and indi.xref_id:
            norm_id = _normalize_id(indi.xref_id)
            if norm_id:
                INDI_INDEX[norm_id] = indi
                count += 1
    elapsed = time.time() - start
    INDI_INDEX_BUILD_TIME = elapsed
    logger.info(f"[Cache] INDI index built with {count} individuals in {elapsed:.2f}s.")


# --- Function to build family relationship maps ---
def build_family_maps(reader):
    """Builds id_to_parents and id_to_children maps for all individuals."""
    start = time.time()
    id_to_parents = {}
    id_to_children = {}
    fam_count = 0
    indi_count = 0
    for fam in reader.records0("FAM"):
        fam_count += 1
        if not _is_record(fam):
            continue
        husband = fam.sub_tag("HUSB")
        wife = fam.sub_tag("WIFE")
        parents = set()
        if _is_individual(husband) and husband.xref_id:
            parents.add(_normalize_id(husband.xref_id))
        if _is_individual(wife) and wife.xref_id:
            parents.add(_normalize_id(wife.xref_id))
        for child in fam.sub_tags("CHIL"):
            if _is_individual(child) and child.xref_id:
                child_id = _normalize_id(child.xref_id)
                if child_id:
                    id_to_parents.setdefault(child_id, set()).update(parents)
                    for parent_id in parents:
                        id_to_children.setdefault(parent_id, set()).add(child_id)
    for indi in reader.records0("INDI"):
        indi_count += 1
    elapsed = time.time() - start
    logger.info(
        f"[PROFILE] Family maps built: {fam_count} FAMs, {indi_count} INDI, {len(id_to_parents)} child->parents, {len(id_to_children)} parent->children in {elapsed:.2f}s"
    )
    global FAMILY_MAPS_BUILD_TIME
    FAMILY_MAPS_BUILD_TIME = elapsed
    return id_to_parents, id_to_children


# --- Top-level helpers for ID extraction and lookup (used by all menu actions) ---
def extract_and_fix_id(raw_id):
    if not raw_id or not isinstance(raw_id, str):
        return None
    id_clean = raw_id.strip().strip("@").upper()
    m = re.match(r"^([IFSTNMCXO][0-9A-Z\-]+)$", id_clean)
    if m:
        return m.group(1)
    m2 = re.search(r"([IFSTNMCXO][0-9]+)", id_clean)
    if m2:
        return m2.group(1)
    return None


def find_individual_by_id(reader, norm_id):
    """Finds an individual by normalized ID using the pre-built index."""
    global INDI_INDEX
    if (
        not INDI_INDEX
    ):  # Fallback if index isn't built (shouldn't happen in normal flow)
        logger.warning("INDI_INDEX not built, falling back to linear scan.")
        for indi in reader.records0("INDI"):
            if hasattr(indi, "xref_id"):
                xref = str(indi.xref_id).strip().strip("@").upper()
                if xref == norm_id:
                    return indi
        return None
    return INDI_INDEX.get(norm_id)  # O(1) lookup


# --- Third-party Imports ---
try:
    from ged4py.parser import GedcomReader
    from ged4py.model import Individual, Record, Name  # Use Name type

    GEDCOM_LIB_AVAILABLE = True
except ImportError:
    print("ERROR: `ged4py` library not found.", file=sys.stderr)
    print("Please install it: pip install ged4py", file=sys.stderr)
    GedcomReader = None
    Individual = None
    Record = None
    Name = None
    GEDCOM_LIB_AVAILABLE = False  # type: ignore
except Exception as import_err:
    print(
        "\n!!! ERROR importing ged4py:",
        type(import_err).__name__,
        "-",
        import_err,
        file=sys.stderr,
    )
    traceback.print_exc(file=sys.stderr)
    print("!!!\n", file=sys.stderr)
    GedcomReader = None
    Individual = None
    Record = None
    Name = None
    GEDCOM_LIB_AVAILABLE = False  # type: ignore

# --- Local Application Imports ---
try:
    from config import config_instance

    log_file_handler = logging.FileHandler(
        "gedcom_processor.log", mode="a", encoding="utf-8"
    )
    log_stream_handler = logging.StreamHandler(sys.stderr)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[log_file_handler, log_stream_handler],
    )
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"ERROR: Failed to import local config module: {e}", file=sys.stderr)
    print("Ensure config.py exists.", file=sys.stderr)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("gedcom_processor_fallback")

    class DummyConfig:
        GEDCOM_FILE_PATH = None

    config_instance = DummyConfig()
    logger.warning("Using fallback logger and dummy config.")

# --- Helper Functions ---


def _is_individual(obj):
    """Checks if object is an Individual safely handling None values"""
    return obj is not None and type(obj).__name__ == "Individual"


def _is_record(obj):
    return obj is not None and type(obj).__name__ == "Record"


def _is_name(obj):
    return obj is not None and type(obj).__name__ == "Name"


def _normalize_id(xref_id: Optional[str]) -> Optional[str]:
    """Normalizes INDI/FAM etc IDs (e.g., '@I123@' -> 'I123')."""
    if xref_id and isinstance(xref_id, str):
        match = re.match(r"^@?([IFSTNMCXO][0-9]+)@?$", xref_id.strip().upper())
        if match:
            return match.group(1)
    return None


def _get_full_name(indi) -> str:
    """Safely gets formatted name using Name.format(). Handles None/errors."""
    if not _is_individual(indi):
        return "Unknown (Not Individual)"
    try:
        name_rec = indi.name
        if _is_name(name_rec):
            formatted_name = name_rec.format()
            cleaned_name = " ".join(formatted_name.split()).title()
            cleaned_name = re.sub(r" /([^/]+)/$", r" \1", cleaned_name).strip()
            return cleaned_name if cleaned_name else "Unknown (Empty Name)"
        elif name_rec is None:
            return "Unknown (No Name Tag)"
        else:
            logger.warning(
                f"Indi @{_normalize_id(indi.xref_id)}@ unexpected .name type: {type(name_rec)}"
            )
            return f"Unknown (Type {type(name_rec).__name__})"
    except AttributeError:
        return "Unknown (Attr Error)"
    except Exception as e:
        logger.error(
            f"Error formatting name for @{_normalize_id(indi.xref_id)}@: {e}",
            exc_info=False,
        )
        return "Unknown (Error)"


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Attempts to parse various GEDCOM date string formats into datetime objects."""
    if not date_str or not isinstance(date_str, str):
        return None
    original_date_str = date_str
    date_str = date_str.strip().upper()
    clean_date_str = re.sub(r"^(ABT|EST|BEF|AFT|BET|FROM|TO)\s+", "", date_str).strip()
    clean_date_str = re.sub(
        r"(\s+(AND|&)\s+\d{4}.*|\s+TO\s+\d{4}.*)", "", clean_date_str
    ).strip()
    clean_date_str = re.sub(r"/\d*$", "", clean_date_str).strip()
    clean_date_str = re.sub(r"\s*\(.*\)\s*", "", clean_date_str).strip()
    formats = ["%d %b %Y", "%d %B %Y", "%b %Y", "%B %Y", "%Y"]
    for fmt in formats:
        try:
            if fmt == "%Y" and not clean_date_str.isdigit():
                continue
            dt = datetime.strptime(clean_date_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            continue
        except Exception as e:
            logger.debug(
                f"Date parsing err for '{original_date_str}' (clean:'{clean_date_str}', fmt:'{fmt}'): {e}"
            )
            continue
    return None


def _clean_display_date(raw_date_str: str) -> str:
    """Removes surrounding brackets if date exists, handles empty brackets."""
    if raw_date_str == "N/A":
        return raw_date_str
    cleaned = raw_date_str.strip()
    cleaned = re.sub(r"^\((.+)\)$", r"\1", cleaned).strip()  # Remove brackets
    return cleaned if cleaned else "N/A"  # Return N/A if empty after cleaning


def _get_event_info(individual, event_tag: str) -> Tuple[Optional[datetime], str, str]:
    """Gets date/place for an event using tag.value. Handles non-string dates."""
    date_obj: Optional[datetime] = None
    date_str: str = "N/A"
    place_str: str = "N/A"
    indi_id_log = "Invalid/Unknown"
    if _is_individual(individual) and individual.xref_id:
        indi_id_log = (
            _normalize_id(individual.xref_id) or f"Unnormalized({individual.xref_id})"
        )
    else:
        logger.warning(f"_get_event_info invalid input: type {type(individual)}")
        return date_obj, date_str, place_str
    try:
        event_record = individual.sub_tag(event_tag)
        if event_record:
            date_tag = event_record.sub_tag("DATE")
            if date_tag and hasattr(date_tag, "value"):
                raw_date_val = date_tag.value
                if isinstance(raw_date_val, str):
                    processed_date_str = raw_date_val.strip()
                    date_str = processed_date_str if processed_date_str else "N/A"
                    date_obj = _parse_date(date_str)
                elif raw_date_val is not None:
                    date_str = str(raw_date_val)
                    date_obj = _parse_date(date_str)
            place_tag = event_record.sub_tag("PLAC")
            if place_tag and hasattr(place_tag, "value"):
                raw_place_val = place_tag.value
                if isinstance(raw_place_val, str):
                    processed_place_str = raw_place_val.strip()
                    place_str = processed_place_str if processed_place_str else "N/A"
                elif raw_place_val is not None:
                    place_str = str(raw_place_val)
    except AttributeError:
        pass
    except Exception as e:
        logger.error(
            f"Unexpected error accessing event {event_tag} for @{indi_id_log}@: {e}",
            exc_info=True,
        )
    return date_obj, date_str, place_str


# --- New helper functions for handling events and formatting ---
def format_life_dates(indi) -> str:
    """Returns a formatted string with birth and death dates."""
    b_date_obj, b_date_str, b_place = _get_event_info(indi, "BIRT")
    d_date_obj, d_date_str, d_place = _get_event_info(indi, "DEAT")
    b_date_str_cleaned = _clean_display_date(b_date_str)
    d_date_str_cleaned = _clean_display_date(d_date_str)

    birth_info = f"b. {b_date_str_cleaned}" if b_date_str_cleaned != "N/A" else ""
    death_info = f"d. {d_date_str_cleaned}" if d_date_str_cleaned != "N/A" else ""
    life_parts = [info for info in [birth_info, death_info] if info]
    return f" ({', '.join(life_parts)})" if life_parts else ""


def format_full_life_details(indi) -> Tuple[str, str]:
    """Returns formatted birth and death details for display."""
    b_date_obj, b_date_str, b_place = _get_event_info(indi, "BIRT")
    d_date_obj, d_date_str, d_place = _get_event_info(indi, "DEAT")
    b_date_str_cleaned = _clean_display_date(b_date_str)
    d_date_str_cleaned = _clean_display_date(d_date_str)

    birth_info = (
        f"   Born: {b_date_str_cleaned if b_date_str_cleaned != 'N/A' else '(Date unknown)'} "
        f"in {b_place if b_place != 'N/A' else '(Place unknown)'}"
    )

    death_info = ""
    if d_date_str_cleaned != "N/A":  # Only show death if date exists
        death_info = (
            f"   Died: {d_date_str_cleaned} "
            f"in {d_place if d_place != 'N/A' else '(Place unknown)'}"
        )

    return birth_info, death_info


def format_relative_info(relative) -> str:
    """Formats information about a relative for display."""
    if not _is_individual(relative):
        return "  - (Invalid Relative Data)"
    rel_name = _get_full_name(relative)
    life_info = format_life_dates(relative)
    return f"  - {rel_name}{life_info}"  # Removed ID


# --- Core Data Retrieval Functions ---


def _find_family_records_where_individual_is_child(reader, target_id, target_id_at):
    """Helper function to find family records where an individual is a child."""
    parent_families = []
    for family_record in reader.records0("FAM"):
        if not _is_record(family_record):
            continue
        children_in_fam = family_record.sub_tags("CHIL")
        if children_in_fam:
            for child in children_in_fam:
                if _is_individual(child) and _normalize_id(child.xref_id) == target_id:
                    parent_families.append(family_record)
                    break
    return parent_families


def _find_family_records_where_individual_is_parent(reader, target_id, target_id_at):
    """Helper function to find family records where an individual is a parent."""
    parent_families = []
    for family_record in reader.records0("FAM"):
        if not _is_record(family_record) or not family_record.xref_id:
            continue
        husband = family_record.sub_tag("HUSB")
        wife = family_record.sub_tag("WIFE")
        is_target_husband = (
            _is_individual(husband) and _normalize_id(husband.xref_id) == target_id
        )
        is_target_wife = (
            _is_individual(wife) and _normalize_id(wife.xref_id) == target_id
        )
        if is_target_husband or is_target_wife:
            parent_families.append((family_record, is_target_husband, is_target_wife))
    return parent_families


def get_related_individuals(reader, individual, relationship_type: str) -> List:
    """Gets parents, spouses, children, or siblings using a more modular approach."""
    related_individuals: List = []
    unique_related_ids: Set[str] = set()

    if not reader:
        logger.error("get_related_individuals: No reader.")
        return related_individuals
    if not _is_individual(individual) or not individual.xref_id:
        logger.warning(f"get_related_individuals: Invalid input.")
        return related_individuals
    target_id_at = f"@{individual.xref_id}@"
    target_id = _normalize_id(individual.xref_id)
    if not target_id:
        logger.warning(
            f"get_related_individuals: Cannot normalize target ID {individual.xref_id}"
        )
        return related_individuals
    target_name = _get_full_name(individual)

    try:
        if relationship_type == "parents":
            logger.debug(
                f"Finding parents for {target_id} by checking all FAM records..."
            )
            potential_parents = []

            # Find all families where this individual is a child
            parent_families = _find_family_records_where_individual_is_child(
                reader, target_id, target_id_at
            )

            # Extract parents from those families
            for family_record in parent_families:
                husband = family_record.sub_tag("HUSB")
                wife = family_record.sub_tag("WIFE")
                if _is_individual(husband):
                    potential_parents.append(husband)
                if _is_individual(wife):
                    potential_parents.append(wife)

            # Add unique parents to the result list
            for parent in potential_parents:
                if parent.xref_id:
                    parent_id = _normalize_id(parent.xref_id)
                    if parent_id and parent_id not in unique_related_ids:
                        related_individuals.append(parent)
                        unique_related_ids.add(parent_id)

            logger.debug(f"Added {len(unique_related_ids)} unique parents.")

        elif relationship_type == "siblings":
            logger.debug(f"Finding siblings for {target_id}...")

            # Find all families where this individual is a child
            parent_families = _find_family_records_where_individual_is_child(
                reader, target_id, target_id_at
            )

            # Collect other children from those families
            potential_siblings = []
            for fam in parent_families:
                fam_children = fam.sub_tags("CHIL")
                if fam_children:
                    potential_siblings.extend(
                        c for c in fam_children if _is_individual(c)
                    )

            # Add unique siblings to the result list, excluding the target individual
            for sibling in potential_siblings:
                if sibling.xref_id:
                    sibling_id = _normalize_id(sibling.xref_id)
                    if (
                        sibling_id
                        and sibling_id not in unique_related_ids
                        and sibling_id != target_id
                    ):
                        related_individuals.append(sibling)
                        unique_related_ids.add(sibling_id)

            logger.debug(f"Added {len(unique_related_ids)} unique siblings.")

        elif relationship_type in ["spouses", "children"]:
            # Find families where target is parent
            parent_families = _find_family_records_where_individual_is_parent(
                reader, target_id, target_id_at
            )

            if relationship_type == "spouses":
                # Process spouses
                for family_record, is_target_husband, is_target_wife in parent_families:
                    other_spouse = None
                    if is_target_husband and _is_individual(
                        family_record.sub_tag("WIFE")
                    ):
                        other_spouse = family_record.sub_tag("WIFE")
                    elif is_target_wife and _is_individual(
                        family_record.sub_tag("HUSB")
                    ):
                        other_spouse = family_record.sub_tag("HUSB")

                    if other_spouse and other_spouse.xref_id:
                        spouse_id = _normalize_id(other_spouse.xref_id)
                        if spouse_id and spouse_id not in unique_related_ids:
                            related_individuals.append(other_spouse)
                            unique_related_ids.add(spouse_id)
            else:  # relationship_type == "children"
                # Process children
                for family_record, _, _ in parent_families:
                    children_list = family_record.sub_tags("CHIL")
                    if children_list:
                        for child in children_list:
                            if _is_individual(child) and child.xref_id:
                                child_id = _normalize_id(child.xref_id)
                                if child_id and child_id not in unique_related_ids:
                                    related_individuals.append(child)
                                    unique_related_ids.add(child_id)
        else:
            logger.warning(f"Unknown relationship type: '{relationship_type}'")
    except AttributeError as ae:
        logger.error(
            f"AttributeError finding {relationship_type} for {target_id_at}: {ae}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"Unexpected error finding {relationship_type} for {target_id_at}: {e}",
            exc_info=True,
        )

    # Sort the results by ID for consistent ordering
    related_individuals.sort(key=lambda x: _normalize_id(x.xref_id) or "")
    return related_individuals


# --- Relationship Path Functions (Keep v7.17 logic + TypeError handling) ---
# --- Robust Individual Lookup for Ancestry GEDCOMs ---


# --- Robust Ancestor Map (no read_record) ---
def _get_ancestors_map(reader, start_id_norm):
    ancestors = {}
    queue = deque([(start_id_norm, 0)])
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
        current_indi = find_individual_by_id(reader, current_id)
        if not current_indi:
            continue
        father = getattr(current_indi, "father", None)
        mother = getattr(current_indi, "mother", None)
        parents = [("Father", father), ("Mother", mother)]
        for role, parent_indi in parents:
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


# --- Robust Path Builder (no read_record) ---
def _build_relationship_path_str(reader, start_id_norm, end_id_norm):
    """Builds shortest ancestral path from start up to end using BFS."""
    if not reader or not start_id_norm or not end_id_norm:
        logger.error("_build_path: Invalid input.")
        return []
    logger.debug(
        f"_build_relationship_path_str: Building path {start_id_norm} -> {end_id_norm}"
    )
    queue = deque([(start_id_norm, [])])
    visited = {start_id_norm}
    processed_count = 0
    while queue:
        current_id, current_path_nodes = queue.popleft()
        processed_count += 1
        current_indi = find_individual_by_id(reader, current_id)
        current_name = (
            _get_full_name(current_indi) if current_indi else f"Unknown/Error"
        )
        new_path_segment = f"{current_name}"
        current_full_path = current_path_nodes + [new_path_segment]
        if current_id == end_id_norm:
            logger.debug(
                f"  Path build: Reached target {end_id_norm} after checking {processed_count} nodes. Path found."
            )
            return current_full_path
        if current_indi:
            father = getattr(current_indi, "father", None)
            mother = getattr(current_indi, "mother", None)
            parents = [("Father", father), ("Mother", mother)]
            for role, parent_indi in parents:
                if _is_individual(parent_indi) and parent_indi.xref_id:
                    parent_id = _normalize_id(parent_indi.xref_id)
                    if parent_id and parent_id not in visited:
                        visited.add(parent_id)
                        queue.append((parent_id, current_full_path))
    logger.warning(
        f"_build_relationship_path_str: Could not find path from {start_id_norm} to {end_id_norm} after checking {processed_count} nodes."
    )
    return []


def _find_lca_from_maps(
    ancestors1: Dict[str, int], ancestors2: Dict[str, int]
) -> Optional[str]:
    """Finds LCA from two ancestor maps (assumed not to contain start nodes)."""
    if not ancestors1 or not ancestors2:
        return None
    common_ancestor_ids = set(ancestors1.keys()) & set(ancestors2.keys())
    if not common_ancestor_ids:
        return None
    # Fix: Correct the type annotation to allow float values
    lca_candidates: Dict[str, Union[int, float]] = {
        cid: ancestors1.get(cid, float("inf")) + ancestors2.get(cid, float("inf"))
        for cid in common_ancestor_ids
    }
    if not lca_candidates:
        return None
    # Fix: Use a lambda function to get the value for comparison
    lca_id = min(lca_candidates.keys(), key=lambda k: lca_candidates[k])
    logger.debug(
        f"_find_lca_from_maps: LCA ID: {lca_id} (Depth Sum: {lca_candidates[lca_id]})"
    )
    return lca_id


# --- Enhanced: Graph-based BFS for Relationship Path (traverse up and down) ---
def _bfs_relationship_path(
    reader, start_id_norm, end_id_norm, max_depth=20, log_progress=True
):
    """Finds the shortest relationship path between two individuals using BFS (parents/children)."""
    if not reader or not start_id_norm or not end_id_norm:
        logger.error("_bfs_relationship_path: Invalid input.")
        return []
    from collections import deque

    queue = deque([(start_id_norm, [start_id_norm])])
    visited = set([start_id_norm])
    processed = 0
    while queue:
        current_id, path = queue.popleft()
        processed += 1
        if log_progress and processed % 1000 == 0:
            logger.info(
                f"  [BFS] Processed {processed} nodes, queue size: {len(queue)}"
            )
        if current_id == end_id_norm:
            # Convert IDs to names for display
            name_path = []
            for pid in path:
                indi = find_individual_by_id(reader, pid)
                name_path.append(_get_full_name(indi) if indi else f"@{pid}@")
            logger.info(f"  [BFS] Found path after {processed} nodes.")
            return name_path
        indi = find_individual_by_id(reader, current_id)
        if not indi:
            continue
        # Traverse parents
        for rel in ["father", "mother"]:
            parent = getattr(indi, rel, None)
            if _is_individual(parent) and parent.xref_id:
                parent_id = _normalize_id(parent.xref_id)
                if parent_id and parent_id not in visited:
                    visited.add(parent_id)
                    queue.append((parent_id, path + [parent_id]))
        # Traverse children
        for fam in reader.records0("FAM"):
            if not _is_record(fam):
                continue
            husband = fam.sub_tag("HUSB")
            wife = fam.sub_tag("WIFE")
            is_parent = False
            if (
                _is_individual(husband) and _normalize_id(husband.xref_id) == current_id
            ) or (_is_individual(wife) and _normalize_id(wife.xref_id) == current_id):
                is_parent = True
            if is_parent:
                for child in fam.sub_tags("CHIL"):
                    if _is_individual(child) and child.xref_id:
                        child_id = _normalize_id(child.xref_id)
                        if child_id and child_id not in visited:
                            visited.add(child_id)
                            queue.append((child_id, path + [child_id]))
        if len(path) > max_depth:
            logger.warning(f"  [BFS] Max depth {max_depth} reached at {current_id}.")
            break
    logger.warning(f"  [BFS] No path found after {processed} nodes.")
    return []


# --- Helper to reconstruct path from predecessor maps ---
def _reconstruct_path(start_id, end_id, meeting_id, visited_fwd, visited_bwd):
    """Reconstructs the path from start to end via the meeting point."""
    path_fwd = []
    curr = meeting_id
    while curr is not None:
        path_fwd.append(curr)
        curr = visited_fwd.get(curr)
    path_fwd.reverse()

    path_bwd = []
    curr = meeting_id  # Start from the meeting point in the backward path
    while curr is not None and curr != end_id:  # Go all the way to the end_id
        curr = visited_bwd.get(curr)
        if curr is not None:  # Only append if not None
            path_bwd.append(curr)

    # Combine paths
    if start_id == meeting_id:
        path = [start_id] + path_bwd  # No need to reverse
    elif end_id == meeting_id:
        path = path_fwd
    else:
        path = path_fwd + path_bwd  # No need to reverse since we're building forward

    return path


def explain_relationship_path(path_ids, reader, id_to_parents, id_to_children):
    """Return a human-readable explanation of the relationship path with relationship labels."""
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"
    steps = []
    for i in range(len(path_ids) - 1):
        a, b = path_ids[i], path_ids[i + 1]
        indi_a = find_individual_by_id(reader, a)
        indi_b = find_individual_by_id(reader, b)
        name_a = _get_full_name(indi_a)
        name_b = _get_full_name(indi_b)

        # Determine relationship label using maps
        if b in id_to_parents.get(a, set()):
            rel = "child"
        elif b in id_to_children.get(a, set()):
            rel = "parent"
        elif a in id_to_parents.get(b, set()):  # Check reverse for consistency
            rel = "parent"
        elif a in id_to_children.get(b, set()):
            rel = "child"
        else:
            rel = "related"  # Fallback

        # Add gendered label using the already looked-up individual objects
        label = rel
        if rel == "parent":  # B is parent of A
            sex_b = getattr(indi_b, "sex", None) if indi_b else None
            if sex_b == "M":
                label = "father"
            elif sex_b == "F":
                label = "mother"
        elif rel == "child":  # B is child of A
            sex_b = getattr(indi_b, "sex", None) if indi_b else None
            if sex_b == "M":
                label = "son"
            elif sex_b == "F":
                label = "daughter"

        # For the reverse direction when checking consistency, make sure to use the correct gender
        if a in id_to_parents.get(b, set()) or a in id_to_children.get(b, set()):
            # Need to use gender of individual A, not B
            sex_a = getattr(indi_a, "sex", None) if indi_a else None

            if rel == "parent" and sex_a == "F":
                label = "daughter"
            elif rel == "parent" and sex_a == "M":
                label = "son"
            elif rel == "child" and sex_a == "F":
                label = "mother"
            elif rel == "child" and sex_a == "M":
                label = "father"

        # Format step: "Person A is the [relationship] of Person B"
        steps.append(f"{name_a} is the {label} of {name_b}")

    return "\n".join(steps)


# --- Optimized bidirectional BFS using pre-built maps and predecessors ---
def fast_bidirectional_bfs(
    start_id,
    end_id,
    id_to_parents,
    id_to_children,
    max_depth=20,
    node_limit=100000,
    timeout_sec=30,
    log_progress=True,
):
    """Performs bidirectional BFS using maps & predecessors. Returns path as list of IDs."""
    start_time = time.time()
    from collections import deque

    # Queue stores (id, depth)
    queue_fwd = deque([(start_id, 0)])
    queue_bwd = deque([(end_id, 0)])
    # Visited stores {id: predecessor_id}
    visited_fwd = {start_id: None}  # Start node has no predecessor
    visited_bwd = {end_id: None}  # End node has no predecessor
    processed = 0
    meeting_id = None

    while queue_fwd and queue_bwd and meeting_id is None:
        # --- Check limits ---
        if time.time() - start_time > timeout_sec:
            logger.warning(f"  [FastBiBFS] Timeout after {timeout_sec} seconds.")
            return []
        if processed > node_limit:
            logger.warning(f"  [FastBiBFS] Node limit {node_limit} reached.")
            return []

        # --- Expand Forward ---
        if queue_fwd:
            current_id, depth = queue_fwd.popleft()
            processed += 1
            if log_progress and processed % 5000 == 0:
                logger.info(
                    f"  [FastBiBFS] FWD processed {processed}, Q:{len(queue_fwd)}, D:{depth}"
                )

            if depth >= max_depth:
                continue

            neighbors = id_to_parents.get(current_id, set()) | id_to_children.get(
                current_id, set()
            )
            for neighbor_id in neighbors:
                if neighbor_id in visited_bwd:  # Intersection found!
                    meeting_id = neighbor_id
                    visited_fwd[neighbor_id] = (
                        current_id  # Record predecessor before breaking
                    )
                    logger.info(
                        f"  [FastBiBFS] Path found (FWD->BWD) at {meeting_id} after {processed} nodes."
                    )
                    break  # Exit inner loop
                if neighbor_id not in visited_fwd:
                    visited_fwd[neighbor_id] = current_id
                    queue_fwd.append((neighbor_id, depth + 1))
            if meeting_id:
                break  # Exit outer loop if intersection found

        # --- Expand Backward ---
        if queue_bwd and meeting_id is None:  # Only expand if no intersection yet
            current_id, depth = queue_bwd.popleft()
            processed += 1
            if log_progress and processed % 5000 == 0:
                logger.info(
                    f"  [FastBiBFS] BWD processed {processed}, Q:{len(queue_bwd)}, D:{depth}"
                )

            if depth >= max_depth:
                continue

            neighbors = id_to_parents.get(current_id, set()) | id_to_children.get(
                current_id, set()
            )
            for neighbor_id in neighbors:
                if neighbor_id in visited_fwd:  # Intersection found!
                    meeting_id = neighbor_id
                    visited_bwd[neighbor_id] = (
                        current_id  # Record predecessor before breaking
                    )
                    logger.info(
                        f"  [FastBiBFS] Path found (BWD->FWD) at {meeting_id} after {processed} nodes."
                    )
                    break  # Exit inner loop
                if neighbor_id not in visited_bwd:
                    visited_bwd[neighbor_id] = current_id
                    queue_bwd.append((neighbor_id, depth + 1))
            # No need to break outer loop here, FWD check will handle it

    # --- Reconstruct Path --- #
    if meeting_id:
        return _reconstruct_path(start_id, end_id, meeting_id, visited_fwd, visited_bwd)
    else:
        logger.warning(f"  [FastBiBFS] No path found after {processed} nodes.")
        return []


# --- Enhanced get_relationship_path using FastBiBFS ---
def get_relationship_path(reader, id1: str, id2: str) -> str:
    """Calculates and formats relationship path using fast bidirectional BFS with pre-built maps."""
    id1_norm = _normalize_id(id1)
    id2_norm = _normalize_id(id2)
    if not reader:
        return "Error: GEDCOM Reader unavailable."
    if not id1_norm or not id2_norm:
        return "Invalid input IDs."
    if id1_norm == id2_norm:
        return "Individuals are the same."

    # Ensure maps and index are built
    global FAMILY_MAPS_CACHE, FAMILY_MAPS_BUILD_TIME, INDI_INDEX, INDI_INDEX_BUILD_TIME
    if FAMILY_MAPS_CACHE is None:
        logger.info(f"  [Cache] Building family maps (first time)...")
        FAMILY_MAPS_CACHE = build_family_maps(reader)
        logger.info(
            f"  [Cache] Maps built and cached in {FAMILY_MAPS_BUILD_TIME:.2f}s."
        )
    if not INDI_INDEX:
        logger.info(f"  [Cache] Building individual index (first time)...")
        build_indi_index(reader)  # Build index if not already done
        logger.info(
            f"  [Cache] Index built and cached in {INDI_INDEX_BUILD_TIME:.2f}s."
        )

    id_to_parents, id_to_children = FAMILY_MAPS_CACHE

    # Use default search parameters, do not prompt user
    max_depth = 20
    node_limit = 100000
    timeout_sec = 30

    logger.info(f"Calculating relationship path (FastBiBFS): {id1_norm} <-> {id2_norm}")
    logger.info(f"  [FastBiBFS] Using cached maps & index. Starting search...")
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
        log_progress=False,  # Reduce log noise during search
    )
    search_time = time.time() - search_start
    logger.info(f"[PROFILE] BFS search completed in {search_time:.2f}s.")
    print(f"[TIMER] BFS search took {search_time:.2f} seconds.")

    if not path_ids:
        return f"No relationship path found (FastBiBFS could not connect).\n[PROFILE] Search: {search_time:.2f}s, Maps: {FAMILY_MAPS_BUILD_TIME:.2f}s, Index: {INDI_INDEX_BUILD_TIME:.2f}s"

    # --- Efficiently Explain Path using Index ---
    explanation_start = time.time()
    # Use the separate explanation function
    explanation_str = explain_relationship_path(
        path_ids, reader, id_to_parents, id_to_children
    )

    explanation_time = time.time() - explanation_start
    logger.info(f"[PROFILE] Path explanation built in {explanation_time:.2f}s.")

    # Get the name of the starting person for the first line
    start_person_name = _get_full_name(find_individual_by_id(reader, path_ids[0]))

    # Combine start person name with the detailed steps
    final_output = f"{start_person_name}\n -> {explanation_str.replace('\n', '\n -> ')}"

    return f"{final_output}\n\n[PROFILE] Total Time: {search_time+explanation_time:.2f}s (Search: {search_time:.2f}s, Explain: {explanation_time:.2f}s, Maps: {FAMILY_MAPS_BUILD_TIME:.2f}s, Index: {INDI_INDEX_BUILD_TIME:.2f}s)"


# --- REVISED v7.30: Fuzzy scoring prioritizing name matches ---
def find_potential_matches(
    reader,
    first_name: Optional[str],
    surname: Optional[str],
    dob_str: Optional[str],
    pob: Optional[str],
    gender: Optional[str] = None,  # New argument for gender
) -> List[Dict]:
    """Finds potential matches based on separate first/last names, DOB, POB, and gender.
    Prioritizes name matches over date matches. Adds gender scoring if provided."""
    if not reader:
        logger.error("find_potential_matches: No reader.")
        return []
    results: List[Dict] = []
    max_results = 3
    year_score_range = 1
    year_filter_range = 30

    # Clean input parameters - remove any non-alphanumeric chars except spaces
    if first_name:
        first_name = re.sub(r"[^\w\s]", "", first_name)
    if surname:
        surname = re.sub(r"[^\w\s]", "", surname)
    if gender:
        gender = gender.strip().lower()
        if gender and gender not in ("m", "f"):
            gender = None  # Only accept 'm' or 'f'

    logger.info(
        f"Fuzzy Search: FirstName='{first_name}', Surname='{surname}', "
        f"DOB='{dob_str}', POB='{pob}', Gender='{gender}'"
    )

    target_first_name_lower = first_name.lower().strip() if first_name else None
    target_surname_lower = surname.lower().strip() if surname else None
    target_year: Optional[int] = None
    dt = _parse_date(dob_str) if dob_str else None
    target_year = dt.year if dt else None
    target_pob_lower = pob.lower().strip() if pob else None

    if (
        not target_first_name_lower
        and not target_surname_lower
        and not target_year
        and not target_pob_lower
        and not gender
    ):
        logger.warning("Fuzzy search called with no valid criteria.")
        return []

    candidate_count = 0
    for indi in reader.records0("INDI"):
        candidate_count += 1
        if not _is_individual(indi) or not hasattr(indi, "xref_id") or not indi.xref_id:
            continue

        indi_id = _normalize_id(indi.xref_id)
        indi_full_name = _get_full_name(indi)
        if indi_full_name.startswith("Unknown"):
            continue

        birth_date_obj, birth_date_str_ged, birth_place_str_ged = _get_event_info(
            indi, "BIRT"
        )
        birth_year_ged: Optional[int] = birth_date_obj.year if birth_date_obj else None

        # Pre-filter by Year
        if target_year and birth_year_ged is not None:
            if abs(birth_year_ged - target_year) > year_filter_range:
                continue

        # --- Scoring ---
        score = 0
        match_reasons = []
        indi_name_lower = indi_full_name.lower()
        indi_name_parts = indi_name_lower.split()
        indi_first_name = indi_name_parts[0] if indi_name_parts else None
        indi_surname = indi_name_parts[-1] if len(indi_name_parts) > 1 else None

        name_score = 0
        date_score = 0
        place_score = 0
        gender_score = 0
        first_name_match = False
        surname_match = False
        year_match = False
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

        # Also check for partial name matches if no exact match
        if not first_name_match and target_first_name_lower and indi_first_name:
            if indi_first_name.startswith(
                target_first_name_lower
            ) or target_first_name_lower.startswith(indi_first_name):
                first_name_match = True
                match_reasons.append(f"Partial First Name Match ('{indi_first_name}')")

        if first_name_match and surname_match:
            name_score = 30
            if "Partial" not in "".join(match_reasons):
                match_reasons.append("First & Surname Match")
        elif surname_match:
            name_score = 10
            match_reasons.append(f"Surname Match ('{indi_surname}')")
        elif first_name_match and "Partial" not in "".join(match_reasons):
            name_score = 5
            match_reasons.append(f"First Name Match ('{indi_first_name}')")

        # 2. Date Scoring (Lower weight)
        if (
            target_year
            and birth_year_ged is not None
            and abs(birth_year_ged - target_year) <= year_score_range
        ):
            date_score = 8
            match_reasons.append(
                f"Birth Year ~{target_year} ({birth_year_ged}, range {year_score_range})"
            )
            year_match = True

        # 3. Place Scoring (Low weight)
        if target_pob_lower and birth_place_str_ged != "N/A":
            place_lower = birth_place_str_ged.lower()
            if place_lower.startswith(target_pob_lower):
                place_score = 2
                match_reasons.append(f"POB starts with '{pob}'")
            elif target_pob_lower in place_lower:
                place_score = 1
                match_reasons.append(f"POB contains '{pob}'")

        # 4. Gender Scoring (if provided)
        indi_gender = getattr(indi, "sex", None)
        if gender and indi_gender:
            indi_gender_lower = str(indi_gender).strip().lower()
            if indi_gender_lower and indi_gender_lower[0] in ("m", "f"):
                if indi_gender_lower[0] == gender:
                    gender_score = 5
                    gender_match = True
                    match_reasons.append(f"Gender Match ({indi_gender_lower.upper()})")
                else:
                    gender_score = -5
                    match_reasons.append(
                        f"Gender Mismatch ({indi_gender_lower.upper()} != {gender.upper()})"
                    )

        # --- Calculate Final Score with SINGLE Boost ---
        score = name_score + date_score + place_score + gender_score
        if first_name_match and surname_match and year_match:
            score += 10
            match_reasons.append("FullName+Year Boost")
        if gender_match and (first_name_match or surname_match):
            score += 2  # Small bonus for gender+name

        if score > 0:
            reasons_str = ", ".join(sorted(list(set(match_reasons))))
            raw_indi_id_str = f"@{indi.xref_id}@" if indi.xref_id else None
            results.append(
                {
                    "id": raw_indi_id_str,
                    "name": indi_full_name,
                    "birth_date": birth_date_str_ged,
                    "birth_place": birth_place_str_ged,
                    "score": score,
                    "reasons": reasons_str or "Overall match",
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    limited_results = results[:max_results]  # Apply limit
    logger.info(
        f"Fuzzy search scanned {candidate_count} individuals. Found {len(results)} potential matches. Showing top {len(limited_results)}."
    )
    return limited_results


# --- Menu and Main Execution ---


# --- REVISED v7.29: Update menu text ---
def menu() -> str:
    """Displays the interactive menu and returns the user's choice."""
    import time

    time.sleep(5)  # Pause for 5 seconds before clearing screen and showing menu
    os.system("cls" if os.name == "nt" else "clear")
    print("\n--- GEDCOM Interrogator (v7.31) ---")  # Version bump
    print("===================================")
    print("ACTIONS using Local GEDCOM File:")
    print("  1. Show Person Details & Relationship to WGG")  # Combined Action 1 & 3
    print("  2. Show Person Details Only")
    # Action 3 removed
    print("----------------------------------")
    print("ACTIONS using Ancestry API (Placeholders):")
    print("  4. Show Person Details & Relationship to WGG (API)")
    print("  5. Show Family (API)")
    print("  6. Show Family Specific (API)")
    print("  7. Show Relationship (API)")
    print("  8. Find Matches (API)")
    print("----------------------------------")
    print(f"  t. Toggle Log Level (Current: {logging.getLevelName(logger.level)})")
    print("  q. Quit")
    print("===================================")
    return input("Enter choice: ").strip().lower()


# --- REVISED v7.26: Update display formatting & add Siblings ---
def display_family_details(reader, individual):
    """Helper function to display formatted family details, including siblings."""
    if not reader or not individual:
        print("  Error: Cannot display details.")
        return

    indi_name = _get_full_name(individual)
    print(f" Individual: {indi_name}")  # Removed ID
    birth_info, death_info = format_full_life_details(individual)
    print(birth_info)
    if death_info:
        print(death_info)

    print("\n Parents:")
    parents = get_related_individuals(reader, individual, "parents")
    if parents:
        [print(format_relative_info(p)) for p in parents]
    else:
        print("  (None found)")

    print("\n Siblings:")
    siblings = get_related_individuals(reader, individual, "siblings")
    if siblings:
        [print(format_relative_info(s)) for s in siblings]
    else:
        print("  (None found)")

    print("\n Spouse(s):")
    spouses = get_related_individuals(reader, individual, "spouses")
    if spouses:
        [print(format_relative_info(s)) for s in spouses]
    else:
        print("  (None found)")
    print("\n Children:")
    children = get_related_individuals(reader, individual, "children")
    if children:
        [print(format_relative_info(c)) for c in children]
    else:
        print("  (None found)")


# --- REVISED v7.28: Revert startup search, handle TypeErrors ---
# --- REVISED v7.29: Use fuzzy match for action 2/3 targets ---
# --- REVISED v7.32: Refactored main function with improved error handling ---
def main():
    """Main execution flow: Load GEDCOM, process user choices."""
    reader = None
    wayne_gault_indi = None  # Store the Individual object
    wayne_gault_id = None  # Store normalized ID
    fuzzy_max_results_display = 3  # For fuzzy search result display

    try:
        logger.info("--- GEDCOM Interrogator Script Starting (v7.32) ---")
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

        # --- Phase 2: Pre-build cache and search for reference person ---
        build_indi_index(reader)  # Build the index immediately for faster lookups

        logger.info("Pre-searching for 'Wayne Gordon Gault' using exact match...")
        wgg_search_name_lower = "wayne gordon gault"

        # Look for WGG with robust error handling
        try:
            for indi in reader.records0("INDI"):
                if _is_individual(indi) and hasattr(
                    indi, "name"
                ):  # Check for name attribute
                    name_rec = indi.name
                    name_str = name_rec.format() if _is_name(name_rec) else ""
                    if name_str.lower() == wgg_search_name_lower:
                        wayne_gault_indi = indi
                        wayne_gault_id = _normalize_id(indi.xref_id)
                        logger.info(
                            f"Found 'Wayne Gordon Gault': {_get_full_name(wayne_gault_indi)} [@{wayne_gault_id}@]"
                        )
                        break

            if not wayne_gault_id:
                logger.warning("'Wayne Gordon Gault' not found during startup")
        except Exception as e:
            logger.error(f"Error during search for WGG: {e}", exc_info=True)

        # --- Phase 3: Interactive menu loop ---
        while True:
            if not reader:
                logger.critical("GEDCOM reader unavailable.")
                break

            choice = menu()

            try:
                # --- ACTION 1: Fuzzy Find + Details + Relationship ---
                if choice == "1":
                    handle_person_details_and_relationship(
                        reader,
                        wayne_gault_indi,
                        wayne_gault_id,
                        fuzzy_max_results_display,
                    )

                # --- ACTION 2: Fuzzy Find + Details Only ---
                elif choice == "2":
                    handle_person_details_only(reader, fuzzy_max_results_display)

                # --- ACTION 4: API Person Details & Relationship to WGG ---
                elif choice == "4":
                    handle_api_person_details_and_relationship_to_wgg()

                # --- ACTIONS 5-8: API Placeholders ---
                elif choice in ["5", "6", "7", "8"]:
                    print(
                        f"\n--- Action {choice} (API) ---\nPlaceholder for future API functionality."
                    )

                # --- Toggle Log Level ---
                elif choice == "t":
                    current_level = logger.level
                    new_level = (
                        logging.DEBUG if current_level >= logging.INFO else logging.INFO
                    )
                    logger.setLevel(new_level)
                    [
                        h.setLevel(new_level)
                        for h in logging.getLogger().handlers
                        if hasattr(h, "setLevel")
                    ]
                    print(f"\nLog level set to {logging.getLevelName(new_level)}")
                    logger.log(new_level, "Log level toggled.")

                # --- Exit ---
                elif choice == "q":
                    print("\nExiting...")
                    break

                # --- Invalid Choice ---
                else:
                    print("\nInvalid choice.")

                if choice != "q":
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\nOperation interrupted.")
                logger.warning("User interrupted.")
                input("Press Enter to continue...")

            except Exception as loop_err:
                logger.error(
                    f"Error during menu action (Choice: {choice}): {loop_err}",
                    exc_info=True,
                )
                print(f"\nError occurred: {type(loop_err).__name__}: {loop_err}")
                print("Check logs for details.")
                input("Press Enter to continue...")

    except (ValueError, FileNotFoundError, ImportError) as setup_err:
        logger.critical(f"Fatal Setup Error: {setup_err}")
        print(f"\nCRITICAL ERROR: {setup_err}", file=sys.stderr)

    except Exception as outer_e:
        logger.critical(f"Unexpected critical error in main: {outer_e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {outer_e}. Check logs.", file=sys.stderr)

    finally:
        logger.info("--- GEDCOM Interrogator Script Finished ---")


# --- REVISED v7.32: Handler functions for menu choices ---
def handle_person_details_and_relationship(
    reader, wayne_gault_indi, wayne_gault_id, max_results=3
):
    """Handler for Option 1 - Show person details and relationship to WGG."""
    print("\n--- Person Details & Relationship to WGG ---")
    if not wayne_gault_indi or not wayne_gault_id:
        print("ERROR: Wayne Gordon Gault (reference person) not found in database.")
        print("Cannot calculate relationships.")
        return

    # Prompt for search query
    query = input("\nEnter search (First name, Last name, or both): ").strip()
    if not query:
        print("Search cancelled.")
        return

    # Find potential matches using advanced fuzzy search
    parts = query.split(maxsplit=1)
    first_name = parts[0] if parts else None
    surname = parts[1] if len(parts) > 1 else None
    gender = input("\nGender (M/F, optional): ").strip() or None
    if gender:
        gender = gender[0].lower() if gender[0].lower() in ["m", "f"] else None

    matches = find_potential_matches(reader, first_name, surname, None, None, gender)

    # Display matches and select one
    if not matches:
        print("\nNo matches found.")
        return

    print(f"\nFound {len(matches)} potential matches:")
    for i, match in enumerate(matches[:max_results]):
        print(f"  {i+1}. {match['name']} ({match['reasons']})")

    try:
        choice = int(input("\nSelect person (or 0 to cancel): "))
        if choice < 1 or choice > len(matches[:max_results]):
            print("Selection cancelled or invalid.")
            return
        selected_match = matches[choice - 1]
        selected_id = extract_and_fix_id(selected_match["id"])
        if not selected_id:
            print("ERROR: Invalid ID in selected match.")
            return

        # Show detailed info
        selected_indi = find_individual_by_id(reader, selected_id)
        if not selected_indi:
            print("ERROR: Could not retrieve individual record.")
            return

        # Display family details
        print("\n=== INDIVIDUAL DETAILS ===")
        display_family_details(reader, selected_indi)

        # Calculate and display relationship path between selected person and reference person (WGG)
        print("\n=== RELATIONSHIP TO WAYNE GORDON GAULT ===")
        relationship_path = get_relationship_path(reader, selected_id, wayne_gault_id)
        print(f"\n{relationship_path}")

    except ValueError:
        print("Invalid selection. Please enter a number.")
    except Exception as e:
        logger.error(
            f"Error in handle_person_details_and_relationship: {e}", exc_info=True
        )
        print(f"Error: {type(e).__name__}: {e}")


def handle_person_details_only(reader, max_results=3):
    """Handler for Option 2 - Show person details only."""
    print("\n--- Person Details ---")

    # Prompt for search query
    query = input("\nEnter search (First name, Last name, or both): ").strip()
    if not query:
        print("Search cancelled.")
        return

    # Find potential matches using advanced fuzzy search
    parts = query.split(maxsplit=1)
    first_name = parts[0] if parts else None
    surname = parts[1] if len(parts) > 1 else None
    dob = input("\nYear of birth (optional): ").strip() or None
    gender = input("Gender (M/F, optional): ").strip() or None
    if gender:
        gender = gender[0].lower() if gender[0].lower() in ["m", "f"] else None

    matches = find_potential_matches(reader, first_name, surname, dob, None, gender)

    # Display matches and select one
    if not matches:
        print("\nNo matches found.")
        return

    print(f"\nFound {len(matches)} potential matches:")
    for i, match in enumerate(matches[:max_results]):
        print(f"  {i+1}. {match['name']} ({match['reasons']})")

    try:
        choice = int(input("\nSelect person (or 0 to cancel): "))
        if choice < 1 or choice > len(matches[:max_results]):
            print("Selection cancelled or invalid.")
            return
        selected_match = matches[choice - 1]
        selected_id = extract_and_fix_id(selected_match["id"])
        if not selected_id:
            print("ERROR: Invalid ID in selected match.")
            return

        # Show detailed info
        selected_indi = find_individual_by_id(reader, selected_id)
        if not selected_indi:
            print("ERROR: Could not retrieve individual record.")
            return

        # Display family details
        print("\n=== INDIVIDUAL DETAILS ===")
        display_family_details(reader, selected_indi)

    except ValueError:
        print("Invalid selection. Please enter a number.")
    except Exception as e:
        logger.error(f"Error in handle_person_details_only: {e}", exc_info=True)
        print(f"Error: {type(e).__name__}: {e}")


# --- Add Ancestry API Search Integration ---
from utils import (
    SessionManager,
    _api_req,
)  # Import dynamic session and API request logic

# Create a standalone session_manager that can authenticate itself
session_manager = SessionManager()


# Add standalone authentication function
def initialize_session():
    """Initialize the session with proper authentication for standalone usage"""
    global session_manager
    if not session_manager.driver_live:
        print("Initializing browser session...")
        session_manager.ensure_driver_live()

    if not session_manager.session_ready:
        print("Authenticating with Ancestry...")
        success = session_manager.ensure_session_ready()
        if not success:
            print("Failed to authenticate with Ancestry.")
            print("Please login manually when the browser opens.")
            input("Press Enter after you've logged in manually...")
        else:
            print("Authentication successful.")

    # Ensure tree_id is loaded
    if not session_manager.my_tree_id:
        print("Loading tree information...")
        session_manager._retrieve_identifiers()
        if not session_manager.my_tree_id:
            print("WARNING: Could not load tree ID. Some functionality may be limited.")
        else:
            print(f"Tree ID loaded successfully: {session_manager.my_tree_id}")

    return session_manager.session_ready


class AncestryAPISearch:
    """
    Class to handle searching the Ancestry API for person information.
    Uses SessionManager and _api_req for dynamic session/cookie/header management.
    """

    def __init__(self, session_manager: SessionManager):
        """
        Initialize the AncestryAPISearch with a SessionManager instance.
        Args:
            session_manager: Required SessionManager object for browser interaction and API requests.
        """
        self.session_manager = session_manager
        self.base_url = "https://www.ancestry.co.uk"

    def _get_tree_id(self):
        # Ensure tree_id is loaded in the session_manager
        if not self.session_manager.my_tree_id:
            self.session_manager._retrieve_identifiers()
        return self.session_manager.my_tree_id

    def _extract_display_name(self, person: dict) -> str:
        # Try to extract a display name from Ancestry API person dict
        # Prefer Names[0] (g: given, s: surname), fallback to gname/sname, fallback to pid
        names = person.get("Names") or person.get("names")
        if names and isinstance(names, list) and names:
            given = names[0].get("g") or ""
            surname = names[0].get("s") or ""
            full_name = f"{given} {surname}".strip()
            if full_name:
                return full_name
        # Fallbacks
        given = person.get("gname") or ""
        surname = person.get("sname") or ""
        if given or surname:
            return f"{given} {surname}".strip()
        # Fallback to pid or id
        return str(person.get("pid") or person.get("id") or "(Unknown)")

    def search_by_name(self, name: str, limit: int = 10) -> list[dict]:
        import urllib.parse

        tree_id = self._get_tree_id()
        if not tree_id:
            from logging import getLogger

            getLogger(__name__).error("No tree_id available in session_manager.")
            return []
        base_url = self.base_url
        query = name.strip()
        tags = ""
        # Remove debug prints for cleaner output
        count_url = f"{base_url}/api/treesui-list/trees/{tree_id}/personscount?name={urllib.parse.quote(query)}&tags={tags}"
        try:
            count_response = _api_req(
                url=count_url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                headers=None,
                use_csrf_token=False,
                api_description="Ancestry Person Count by Name",
                referer_url=f"{base_url}/family-tree/tree/{tree_id}/family",
                timeout=10,
            )
            if isinstance(count_response, int):
                count = count_response
            elif isinstance(count_response, dict):
                count = count_response.get("count")
            else:
                return []
            if not count or count < 1:
                return []
        except Exception:
            return []
        persons_url = f"{base_url}/api/treesui-list/trees/{tree_id}/persons?name={urllib.parse.quote(query)}&tags={tags}&page=1&limit={limit}&fields=EVENTS,GENDERS,KINSHIP,NAMES,PHOTO,RELATIONS,TAGS&isGetFullPersonObject=false"
        try:
            persons_response = _api_req(
                url=persons_url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                headers=None,
                use_csrf_token=False,
                api_description="Ancestry Search by Name",
                referer_url=f"{base_url}/family-tree/tree/{tree_id}/family",
                timeout=20,
            )
            if not persons_response or not isinstance(persons_response, list):
                return []
            return persons_response
        except Exception:
            return []

    def format_person_details(self, person: dict) -> str:
        # Extract and format key details from an Ancestry API person dict
        name = self._extract_display_name(person)
        gender = None
        # Try to get gender from Genders or gender/g fields
        if "Genders" in person and person["Genders"]:
            gender = person["Genders"][0].get("g")
        elif "gender" in person:
            gender = person["gender"]
        gender_str = f"Gender: {gender}" if gender else "Gender: Unknown"
        # Try to get birth event
        birth_info = ""
        events = person.get("Events") or person.get("events") or []
        for event in events:
            if (
                event.get("t", "").lower() == "birth"
                or event.get("type", "").lower() == "birth"
            ):
                date = event.get("d") or event.get("date") or event.get("nd")
                place = event.get("p") or event.get("place")
                birth_info = f"Birth: {date or '?'} in {place or '?'}"
                break
        # PID or ID
        pid = person.get("pid") or person.get("id") or person.get("gid", {}).get("v")
        pid_str = f"Person ID: {pid}" if pid else ""
        # Compose output
        lines = [
            f"Name: {name}",
            gender_str,
        ]
        if birth_info:
            lines.append(birth_info)
        if pid_str:
            lines.append(pid_str)
        return "\n".join(lines)

    def get_relationship_ladder(self, person_id: str):
        import re, json

        tree_id = self._get_tree_id()
        if not tree_id or not person_id:
            return {"error": "Missing tree_id or person_id."}
        url = f"{self.base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/getladder?callback=jQuery1234567890_1234567890&_={int(time.time()*1000)}"
        try:
            response = _api_req(
                url=url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                headers=None,
                use_csrf_token=False,
                api_description="Ancestry Relationship Ladder",
                referer_url=f"{self.base_url}/family-tree/tree/{tree_id}/family",
                timeout=20,
            )
            if isinstance(response, str):
                match = re.search(r"\((\{.*\})\)", response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    first_brace = json_str.find("{")
                    last_brace = json_str.rfind("}")
                    if (
                        first_brace != -1
                        and last_brace != -1
                        and last_brace > first_brace
                    ):
                        json_str = json_str[first_brace : last_brace + 1]
                        json_str = bytes(json_str, "utf-8").decode("unicode_escape")
                        try:
                            ladder_json = json.loads(json_str)
                        except Exception:
                            # Fallback: extract HTML manually and parse
                            html_match = re.search(
                                r'"html"\s*:\s*"(.*?)"\s*[,}]', json_str, re.DOTALL
                            )
                            if html_match:
                                html_escaped = html_match.group(1)
                                html = bytes(html_escaped, "utf-8").decode(
                                    "unicode_escape"
                                )
                                return self.parse_ancestry_ladder_html(html)
                            else:
                                return {
                                    "error": "JSON decode failed and no HTML found."
                                }
                        if isinstance(ladder_json, dict) and "html" in ladder_json:
                            return self.parse_ancestry_ladder_html(ladder_json["html"])
                        else:
                            return {"error": "No 'html' key in ladder JSON."}
                    else:
                        return {
                            "error": "Could not extract JSON object from JSONP response."
                        }
                else:
                    return {"error": "Could not parse JSONP response."}
            elif isinstance(response, dict):
                return response
            else:
                return {"error": "Unexpected response type."}
        except Exception as e:
            return {"error": str(e)}

    def format_ladder_details(self, ladder_data) -> str:
        # Format the relationship ladder details from the API response
        if not ladder_data:
            return "No relationship ladder data available."
        if isinstance(ladder_data, dict):
            if "error" in ladder_data:
                return f"Error: {ladder_data['error']}"
            # Try to extract relationship and path
            rel = ladder_data.get("actual_relationship")
            path = ladder_data.get("relationship_path")
            if rel or path:
                out = []
                if rel:
                    out.append(f"Relationship: {rel}")
                if path:
                    out.append(f"Path: {path}")
                return "\n".join(out)
        return str(ladder_data)

    # ...existing code...

    def parse_ancestry_ladder_html(self, html):
        from bs4 import BeautifulSoup

        ladder_data = {}
        soup = BeautifulSoup(html, "html.parser")
        # Extract actual relationship
        rel_elem = soup.select_one(
            "ul.textCenter > li:first-child > i > b"
        ) or soup.select_one("ul.textCenter > li > i > b")
        if rel_elem:
            ladder_data["actual_relationship"] = rel_elem.get_text(strip=True).title()
        # Extract relationship path
        path_items = soup.select('ul.textCenter > li:not([class*="iconArrowDown"])')
        path_list = []
        for i, item in enumerate(path_items):
            name_text = ""
            desc_text = ""
            name_container = item.find("a") or item.find("b")
            if name_container:
                name_text = name_container.get_text(strip=True)
            if i > 0:
                desc_element = item.find("i")
                if desc_element:
                    desc_text = desc_element.get_text(strip=True)
            if name_text:
                path_list.append(
                    f"{name_text} ({desc_text})" if desc_text else name_text
                )
        if path_list:
            ladder_data["relationship_path"] = "\n↓\n".join(path_list)
        return ladder_data

    def get_relationship_ladder(self, person_id: str):
        import re, json

        tree_id = self._get_tree_id()
        if not tree_id or not person_id:
            return {"error": "Missing tree_id or person_id."}
        url = f"{self.base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/getladder?callback=jQuery1234567890_1234567890&_={int(time.time()*1000)}"
        try:
            response = _api_req(
                url=url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                headers=None,
                use_csrf_token=False,
                api_description="Ancestry Relationship Ladder",
                referer_url=f"{self.base_url}/family-tree/tree/{tree_id}/family",
                timeout=20,
            )
            if isinstance(response, str):
                match = re.search(r"\((\{.*\})\)", response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    first_brace = json_str.find("{")
                    last_brace = json_str.rfind("}")
                    if (
                        first_brace != -1
                        and last_brace != -1
                        and last_brace > first_brace
                    ):
                        json_str = json_str[first_brace : last_brace + 1]
                        json_str = bytes(json_str, "utf-8").decode("unicode_escape")
                        try:
                            ladder_json = json.loads(json_str)
                        except Exception:
                            # Fallback: extract HTML manually and parse
                            html_match = re.search(
                                r'"html"\s*:\s*"(.*?)"\s*[,}]', json_str, re.DOTALL
                            )
                            if html_match:
                                html_escaped = html_match.group(1)
                                html = bytes(html_escaped, "utf-8").decode(
                                    "unicode_escape"
                                )
                                return self.parse_ancestry_ladder_html(html)
                            else:
                                return {
                                    "error": "JSON decode failed and no HTML found."
                                }
                        if isinstance(ladder_json, dict) and "html" in ladder_json:
                            return self.parse_ancestry_ladder_html(ladder_json["html"])
                        else:
                            return {"error": "No 'html' key in ladder JSON."}
                    else:
                        return {
                            "error": "Could not extract JSON object from JSONP response."
                        }
                else:
                    return {"error": "Could not parse JSONP response."}
            elif isinstance(response, dict):
                return response
            else:
                return {"error": "Unexpected response type."}
        except Exception as e:
            return {"error": str(e)}

    def format_ladder_details(self, ladder_data) -> str:
        if not ladder_data:
            return "No relationship ladder data available."
        if isinstance(ladder_data, dict):
            if "error" in ladder_data:
                return f"Error: {ladder_data['error']}"
            rel = ladder_data.get("actual_relationship")
            path = ladder_data.get("relationship_path")
            out = []
            if rel:
                out.append(f"Relationship: {rel}")
            if path:
                out.append(f"Path:\n{path}")
            return "\n".join(out) if out else str(ladder_data)
        return str(ladder_data)

    # ...existing code...


# --- API Menu Action Handlers ---
def handle_api_show_family(api_searcher):
    """Handler for API: Show Family option"""
    print("\n=== API: Show Family ===")

    iteration = 1
    while True:
        print(f"\nIteration {iteration}:")
        family_name = input("Enter family name to search: ")
        if not family_name:
            print("Search cancelled.")
            break

        location = input("Enter location (optional): ")

        results = api_searcher.search_by_name(family_name)

        if results:
            print(f"\nFound {len(results)} potential family matches:")
            for i, person in enumerate(results[:5]):
                preferred_name = person.get("preferredName", {})
                full_name = f"{preferred_name.get('givenName', '')} {preferred_name.get('surname', '')}".strip()
                print(f"  {i+1}. {full_name}")

            try:
                selection = int(
                    input("\nSelect a person to view family details (or 0 to skip): ")
                )
                if 1 <= selection <= len(results[:5]):
                    selected_person = results[selection - 1]
                    # Get person details
                    details = api_searcher.format_person_details(selected_person)
                    print(f"\n{details}")

                    # Get and display relationship ladder for family context
                    person_id = selected_person.get("id")
                    if person_id:
                        print("\nFetching family relationships...")
                        ladder_data = api_searcher.get_relationship_ladder(person_id)
                        ladder_details = api_searcher.format_ladder_details(ladder_data)
                        print(f"\n{ladder_details}")
            except ValueError:
                print("Invalid selection.")
        else:
            print(f"No matches found for family name '{family_name}'.")

        continue_iteration = input("\nContinue to iterate? (y/n): ").lower()
        if continue_iteration != "y":
            break

        iteration += 1


def handle_api_show_family_specific(api_searcher):
    """Handler for API: Show Family Specific option"""
    print("\n=== API: Show Family Specific ===")

    iteration = 1
    while True:
        print(f"\nIteration {iteration}:")
        first_name = input("Enter first name: ")
        last_name = input("Enter last name: ")

        if not first_name and not last_name:
            print("Search cancelled.")
            break

        # Build search query
        search_query = f"{first_name} {last_name}".strip()

        # Additional criteria
        birth_year = input("Enter birth year (optional): ")

        results = api_searcher.search_by_name(search_query)

        # Filter by birth year if provided
        if birth_year and birth_year.isdigit():
            birth_year = int(birth_year)
            filtered_results = []
            for person in results:
                for event in person.get("events", []):
                    if (
                        event.get("type") == "birth"
                        and event.get("date", {}).get("yearInt") == birth_year
                    ):
                        filtered_results.append(person)
                        break
            results = filtered_results

        if results:
            print(f"\nFound {len(results)} specific matches:")
            for i, person in enumerate(results[:5]):
                preferred_name = person.get("preferredName", {})
                full_name = f"{preferred_name.get('givenName', '')} {preferred_name.get('surname', '')}".strip()

                # Display birth year if available
                birth_year_display = ""
                for event in person.get("events", []):
                    if event.get("type") == "birth" and event.get("date", {}).get(
                        "yearInt"
                    ):
                        birth_year_display = (
                            f" (b. {event.get('date', {}).get('yearInt')})"
                        )
                        break

                print(f"  {i+1}. {full_name}{birth_year_display}")

            try:
                selection = int(
                    input("\nSelect a person to view details (or 0 to skip): ")
                )
                if 1 <= selection <= len(results[:5]):
                    selected_person = results[selection - 1]
                    # Get person details
                    details = api_searcher.format_person_details(selected_person)
                    print(f"\n{details}")

                    # Get and display relationship ladder for family context
                    person_id = selected_person.get("id")
                    if person_id:
                        print("\nFetching family relationships...")
                        ladder_data = api_searcher.get_relationship_ladder(person_id)
                        ladder_details = api_searcher.format_ladder_details(ladder_data)
                        print(f"\n{ladder_details}")
            except ValueError:
                print("Invalid selection.")
        else:
            print(f"No specific matches found for '{search_query}'.")

        continue_iteration = input("\nContinue to iterate? (y/n): ").lower()
        if continue_iteration != "y":
            break

        iteration += 1


def handle_api_show_relationship(api_searcher):
    """Handler for API: Show Relationship option"""
    print("\n=== API: Show Relationship ===")

    iteration = 1
    while True:
        print(f"\nIteration {iteration}:")
        # First person
        person1_name = input("Enter first person's name: ")
        if not person1_name:
            print("Search cancelled.")
            break

        person1_results = api_searcher.search_by_name(person1_name)
        if not person1_results:
            print(f"No matches found for '{person1_name}'.")
            continue

        print(f"\nFound {len(person1_results)} matches for first person:")
        for i, person in enumerate(person1_results[:5]):
            preferred_name = person.get("preferredName", {})
            full_name = f"{preferred_name.get('givenName', '')} {preferred_name.get('surname', '')}".strip()
            print(f"  {i+1}. {full_name}")

        try:
            selection1 = int(input("\nSelect first person (or 0 to cancel): "))
            if selection1 < 1 or selection1 > len(person1_results[:5]):
                print("Selection cancelled.")
                continue

            person1 = person1_results[selection1 - 1]
            person1_id = person1.get("id")

            # Second person
            person2_name = input("\nEnter second person's name: ")
            if not person2_name:
                print("Search cancelled.")
                continue

            person2_results = api_searcher.search_by_name(person2_name)
            if not person2_results:
                print(f"No matches found for '{person2_name}'.")
                continue

            print(f"\nFound {len(person2_results)} matches for second person:")
            for i, person in enumerate(person2_results[:5]):
                preferred_name = person.get("preferredName", {})
                full_name = f"{preferred_name.get('givenName', '')} {preferred_name.get('surname', '')}".strip()
                print(f"  {i+1}. {full_name}")

            selection2 = int(input("\nSelect second person (or 0 to cancel): "))
            if selection2 < 1 or selection2 > len(person2_results[:5]):
                print("Selection cancelled.")
                continue

            person2 = person2_results[selection2 - 1]
            person2_id = person2.get("id")

            # Show relationship information for both people
            print("\nFetching relationship information...")

            # Get details for person 1
            person1_details = api_searcher.format_person_details(person1)
            print(f"\nPerson 1:\n{person1_details}")

            # Get details for person 2
            person2_details = api_searcher.format_person_details(person2)
            print(f"\nPerson 2:\n{person2_details}")

            # Compare relationship ladders to find connections
            print("\nChecking relationship path...")
            ladder1 = api_searcher.get_relationship_ladder(person1_id)
            ladder2 = api_searcher.get_relationship_ladder(person2_id)

            # This is a simplified approach - in a real implementation, we would
            # analyze the ladders to find common ancestors and calculate relationship
            print("\nRelationship Analysis:")
            print("(Note: This is a simplified representation of relationships)")
            print("Person 1 Family Tree:")
            print(api_searcher.format_ladder_details(ladder1))
            print("\nPerson 2 Family Tree:")
            print(api_searcher.format_ladder_details(ladder2))

        except ValueError:
            print("Invalid selection.")

        continue_iteration = input("\nContinue to iterate? (y/n): ").lower()
        if continue_iteration != "y":
            break

        iteration += 1


# --- ADDED v7.33: Handler functions for Ancestry API options ---
def handle_api_show_family(reader):
    """Handler for Option 5 - Show Family (API search)."""
    api_search = AncestryAPISearch(session_manager)
    print("\n--- Show Family (Ancestry API) ---")

    continue_search = True
    while continue_search:
        # Get search query
        name = input("\nEnter name to search for: ").strip()
        if not name:
            print("Search cancelled.")
            return

        # Perform API search
        persons = api_search.search_by_name(name)

        if not persons:
            print("\nNo matches found in Ancestry.")
            continue_search = (
                input("\nContinue to iterate? (y/n): ").lower().startswith("y")
            )
            continue

        # Display search results
        print(f"\nFound {len(persons)} matches:")
        for i, person in enumerate(persons[:5]):  # Limit display to 5 results
            preferred_name = person.get("preferredName", {})
            name_display = f"{preferred_name.get('givenName', '')} {preferred_name.get('surname', '')}".strip()
            print(f"  {i+1}. {name_display}")

        # Select a person
        try:
            choice = int(input("\nSelect person (or 0 to cancel): "))
            if choice < 1 or choice > len(persons[:5]):
                print("Selection cancelled or invalid.")
                continue_search = (
                    input("\nContinue to iterate? (y/n): ").lower().startswith("y")
                )
                continue

            selected_person = persons[choice - 1]

            # Display detailed information
            print("\n=== PERSON DETAILS ===")
            print(api_search.format_person_details(selected_person))

        except ValueError:
            print("Invalid selection. Please enter a number.")

        # Ask if user wants to continue searching
        continue_search = (
            input("\nContinue to iterate? (y/n): ").lower().startswith("y")
        )


def handle_api_show_family_specific(reader):
    """Handler for Option 6 - Show Family Specific (API search)."""
    api_search = AncestryAPISearch(session_manager)
    print("\n--- Show Family Specific (Ancestry API) ---")

    continue_search = True
    while continue_search:
        # Get search parameters
        first_name = input("\nEnter first name: ").strip()
        last_name = input("Enter last name: ").strip()

        if not first_name and not last_name:
            print("Search cancelled.")
            return

        full_name = f"{first_name} {last_name}".strip()

        # Perform API search
        persons = api_search.search_by_name(full_name)

        if not persons:
            print(f"\nNo matches found for '{full_name}' in Ancestry.")
            continue_search = (
                input("\nContinue to iterate? (y/n): ").lower().startswith("y")
            )
            continue

        # Display search results
        print(f"\nFound {len(persons)} matches:")
        for i, person in enumerate(persons[:5]):  # Limit display to 5 results
            preferred_name = person.get("preferredName", {})
            name_display = f"{preferred_name.get('givenName', '')} {preferred_name.get('surname', '')}".strip()
            print(f"  {i+1}. {name_display}")

        # Select a person
        try:
            choice = int(input("\nSelect person (or 0 to cancel): "))
            if choice < 1 or choice > len(persons[:5]):
                print("Selection cancelled or invalid.")
                continue_search = (
                    input("\nContinue to iterate? (y/n): ").lower().startswith("y")
                )
                continue

            selected_person = persons[choice - 1]

            # Display detailed information
            print("\n=== PERSON DETAILS ===")
            print(api_search.format_person_details(selected_person))

        except ValueError:
            print("Invalid selection. Please enter a number.")

        # Ask if user wants to continue searching
        continue_search = (
            input("\nContinue to iterate? (y/n): ").lower().startswith("y")
        )


def handle_api_show_relationship(reader):
    """Handler for Option 7 - Show Relationship (API search)."""
    api_search = AncestryAPISearch(session_manager)
    print("\n--- Show Relationship (Ancestry API) ---")

    continue_search = True
    while continue_search:
        # Get first person
        name1 = input("\nEnter name of first person: ").strip()
        if not name1:
            print("Search cancelled.")
            return

        persons1 = api_search.search_by_name(name1)
        if not persons1:
            print(f"\nNo matches found for '{name1}' in Ancestry.")
            continue_search = (
                input("\nContinue to iterate? (y/n): ").lower().startswith("y")
            )
            continue

        print(f"\nFound {len(persons1)} matches for first person:")
        for i, person in enumerate(persons1[:5]):
            preferred_name = person.get("preferredName", {})
            name_display = f"{preferred_name.get('givenName', '')} {preferred_name.get('surname', '')}".strip()
            print(f"  {i+1}. {name_display}")

        try:
            choice1 = int(input("\nSelect first person (or 0 to cancel): "))
            if choice1 < 1 or choice1 > len(persons1[:5]):
                print("Selection cancelled or invalid.")
                continue_search = (
                    input("\nContinue to iterate? (y/n): ").lower().startswith("y")
                )
                continue

            selected_person1 = persons1[choice1 - 1]

            # Get second person
            name2 = input("\nEnter name of second person: ").strip()
            if not name2:
                print("Search cancelled.")
                continue_search = (
                    input("\nContinue to iterate? (y/n): ").lower().startswith("y")
                )
                continue

            persons2 = api_search.search_by_name(name2)
            if not persons2:
                print(f"\nNo matches found for '{name2}' in Ancestry.")
                continue_search = (
                    input("\nContinue to iterate? (y/n): ").lower().startswith("y")
                )
                continue

            print(f"\nFound {len(persons2)} matches for second person:")
            for i, person in enumerate(persons2[:5]):
                preferred_name = person.get("preferredName", {})
                name_display = f"{preferred_name.get('givenName', '')} {preferred_name.get('surname', '')}".strip()
                print(f"  {i+1}. {name_display}")

            choice2 = int(input("\nSelect second person (or 0 to cancel): "))
            if choice2 < 1 or choice2 > len(persons2[:5]):
                print("Selection cancelled or invalid.")
                continue_search = (
                    input("\nContinue to iterate? (y/n): ").lower().startswith("y")
                )
                continue

            selected_person2 = persons2[choice2 - 1]

            # Get relationship ladder for the first person
            ladder_data = api_search.get_relationship_ladder(selected_person1.get("id"))

            if ladder_data:
                print("\n=== RELATIONSHIP INFORMATION ===")
                print(api_search.format_ladder_details(ladder_data))
            else:
                print("\nNo relationship information available.")

        except ValueError:
            print("Invalid selection. Please enter a number.")

        # Ask if user wants to continue searching
        continue_search = (
            input("\nContinue to iterate? (y/n): ").lower().startswith("y")
        )


def handle_api_find_matches(reader):
    """Handler for Option 8 - Find Matches (API search)."""
    api_search = AncestryAPISearch(session_manager)
    print("\n--- Find Matches (Ancestry API) ---")

    continue_search = True
    while continue_search:
        # Get search parameters
        first_name = input("\nEnter first name: ").strip()
        last_name = input("Enter last name: ").strip()
        birth_year = input("Enter birth year (optional): ").strip()

        if not first_name and not last_name:
            print("Search cancelled.")
            return

        full_name = f"{first_name} {last_name}".strip()

        # Perform API search
        persons = api_search.search_by_name(full_name)

        if not persons:
            print(f"\nNo matches found for '{full_name}' in Ancestry.")
            continue_search = (
                input("\nContinue to iterate? (y/n): ").lower().startswith("y")
            )
            continue

        # Display search results
        print(f"\nFound {len(persons)} potential matches:")
        for i, person in enumerate(persons[:5]):  # Limit display to 5 results
            preferred_name = person.get("preferredName", {})
            name_display = f"{preferred_name.get('givenName', '')} {preferred_name.get('surname', '')}".strip()

            # Add birth information if available
            birth_info = ""
            for event in person.get("events", []):
                if event.get("type", "").lower() == "birth":
                    date = event.get("date", {}).get("dateDisplay", "")
                    if date:
                        birth_info = f" (b. {date})"
                    break

            print(f"  {i+1}. {name_display}{birth_info}")

        # Select a person
        try:
            choice = int(input("\nSelect person (or 0 to cancel): "))
            if choice < 1 or choice > len(persons[:5]):
                print("Selection cancelled or invalid.")
                continue_search = (
                    input("\nContinue to iterate? (y/n): ").lower().startswith("y")
                )
                continue

            selected_person = persons[choice - 1]

            # Display detailed information
            print("\n=== PERSON DETAILS ===")
            print(api_search.format_person_details(selected_person))

        except ValueError:
            print("Invalid selection. Please enter a number.")

        # Ask if user wants to continue searching
        continue_search = (
            input("\nContinue to iterate? (y/n): ").lower().startswith("y")
        )


def handle_api_person_details_and_relationship_to_wgg():
    """Action 4: Show Person Details & Relationship to WGG (API)"""
    print("\n--- Person Details & Relationship to WGG (API) ---")

    # Initialize session before proceeding with API operations
    if not initialize_session():
        print("Failed to initialize session. Cannot proceed with API operations.")
        return

    api_search = AncestryAPISearch(session_manager)
    # Prompt for search query
    query = input("\nEnter search (First name, Last name, or both): ").strip()
    if not query:
        print("Search cancelled.")
        return
    # Find potential matches using API
    persons = api_search.search_by_name(query)
    if not persons:
        print("\nNo matches found in Ancestry API.")
        return
    print(f"\nFound {len(persons)} potential matches:")
    for i, person in enumerate(persons[:5]):
        name_display = api_search._extract_display_name(person)
        print(f"  {i+1}. {name_display}")
    try:
        choice = int(input("\nSelect person (or 0 to cancel): "))
        if choice < 1 or choice > len(persons[:5]):
            print("Selection cancelled or invalid.")
            return
        selected_person = persons[choice - 1]
        print("\n=== PERSON DETAILS ===")
        print(api_search.format_person_details(selected_person))

        # Find WGG in API tree
        wgg_name = "Wayne Gordon Gault"
        wgg_results = api_search.search_by_name(wgg_name)
        if not wgg_results:
            print(f"\nReference person '{wgg_name}' not found in Ancestry API.")
            return
        wgg_person = wgg_results[0]

        # Extract ID using pid, id, or gid
        def extract_id(person):
            return (
                person.get("pid")
                or person.get("id")
                or (
                    person.get("gid", {}).get("v")
                    if isinstance(person.get("gid"), dict)
                    else None
                )
            )

        wgg_id = extract_id(wgg_person)
        selected_id = extract_id(selected_person)

        if not wgg_id or not selected_id:
            print("\nCould not determine IDs for relationship lookup.")
            return

        # Get the ladder API URL
        tree_id = api_search._get_tree_id()
        if not tree_id:
            print("Could not determine tree ID.")
            return

        url = f"{api_search.base_url}/family-tree/person/tree/{tree_id}/person/{selected_id}/getladder?callback=jQuery1234567890_1234567890&_={int(time.time()*1000)}"

        # Make the API request directly
        print("\nLooking up relationship information...")
        response = _api_req(
            url=url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers=None,
            use_csrf_token=False,
            api_description="Ancestry Relationship Ladder",
            referer_url=f"{api_search.base_url}/family-tree/tree/{tree_id}/family",
            timeout=20,
        )

        # Always use our robust display function
        display_raw_relationship_ladder(response)

    except ValueError:
        print("Invalid selection. Please enter a number.")
    except Exception as e:
        logger.error(
            f"Error in handle_api_person_details_and_relationship_to_wgg: {e}",
            exc_info=True,
        )
        print(f"Error: {type(e).__name__}: {e}")


def display_raw_relationship_ladder(raw_content):
    """
    Parse and display the Ancestry relationship ladder from raw JSONP/HTML content.
    Robustly extracts the 'html' part, decodes, and parses the relationship and path, even if the JSON is truncated or malformed.
    """
    import html
    import re
    from bs4 import BeautifulSoup

    print("\n=== RELATIONSHIP TO WAYNE GORDON GAULT (API) ===")
    if not raw_content or not isinstance(raw_content, str):
        print("No relationship content available.")
        return

    # Try to extract the 'html' part from the JSONP, even if truncated
    html_escaped = None
    # Use regex to robustly extract the html string, even if the JSON is truncated
    html_match = re.search(r'"html"\s*:\s*"((?:\\.|[^"\\])*)"', raw_content)
    if html_match:
        html_escaped = html_match.group(1)
    else:
        # Fallback: try to find 'html":"' and take everything after
        html_start = raw_content.find('html":"')
        if html_start != -1:
            html_start += len('html":"')
            html_escaped = raw_content[html_start:]
            # Truncate at the next unescaped quote
            end_quote = html_escaped.find('"}')
            if end_quote != -1:
                html_escaped = html_escaped[:end_quote]
    if not html_escaped:
        print("Could not extract relationship ladder HTML from the API response.")
        return

    # Unescape unicode and HTML entities
    try:
        html_unescaped = bytes(html_escaped, "utf-8").decode("unicode_escape")
        html_unescaped = html.unescape(html_unescaped)
    except Exception:
        print("Could not decode relationship ladder HTML.")
        return

    soup = BeautifulSoup(html_unescaped, "html.parser")

    # Extract actual relationship
    rel_elem = soup.select_one(
        "ul.textCenter > li:first-child > i > b"
    ) or soup.select_one("ul.textCenter > li > i > b")
    if rel_elem:
        actual_relationship = rel_elem.get_text(strip=True).title()
        print(f"Relationship: {actual_relationship}")
    else:
        print("Relationship: (not found)")

    # Extract relationship path
    path_items = soup.select('ul.textCenter > li:not([class*="iconArrowDown"])')
    path_list = []
    for item in path_items:
        name_container = item.find("a") or item.find("b")
        name_text = name_container.get_text(strip=True) if name_container else ""
        # Try to get description (e.g., relationship label)
        desc_elem = item.find("i")
        desc_text = desc_elem.get_text(strip=True) if desc_elem else ""
        if name_text:
            if desc_text:
                path_list.append(f"{name_text} ({desc_text})")
            else:
                path_list.append(name_text)
    if path_list:
        print("Path:")
        for i, p in enumerate(path_list):
            print(f"  {p}")
            if i < len(path_list) - 1:
                print("  ↓")
    else:
        print("No relationship path found.")


# --- Script Entry Point ---
if __name__ == "__main__":
    if "logger" not in globals() or not isinstance(logger, logging.Logger):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logger = logging.getLogger("gedcom_main_fallback")
        logger.warning("Using fallback logger.")
    if GEDCOM_LIB_AVAILABLE:
        main()
    else:
        logger.critical("Exiting: ged4py library unavailable.")
        sys.exit(1)
