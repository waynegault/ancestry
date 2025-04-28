# gedcom_utils.py
"""
Utility functions for loading, parsing, and querying GEDCOM data using ged4py.
Includes functions for indexing, finding individuals/families, calculating relationships,
and fuzzy searching within a GEDCOM file.
"""

import logging
import sys
import re
import time
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set, Deque, Union
from collections import deque
from datetime import datetime, timezone
import difflib  # For partial name matching

# Add parent directory to sys.path if needed (adjust if structure differs)
# Assuming this file might be in a subdirectory relative to logging_config
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules - Adjust path as necessary
try:
    from logging_config import setup_logging
except ImportError:
    # Fallback basic logging if setup_logging is not found
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    def setup_logging(log_file="gedcom_utils.log", log_level="INFO"):
        logger = logging.getLogger("gedcom_utils_fallback")
        # Basic setup already done by basicConfig
        return logger


# Setup logging
logger = setup_logging(log_file="gedcom_processor.log", log_level="INFO")

# Global cache for family relationships and individual lookup
FAMILY_MAPS_CACHE: Optional[Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]] = None
FAMILY_MAPS_BUILD_TIME: float = 0
INDI_INDEX: Dict[str, "Individual"] = {}  # Index for ID -> Individual object
INDI_INDEX_BUILD_TIME: float = 0  # Track build time

# --- Third-party Imports ---
GEDCOM_LIB_AVAILABLE = False
GedcomReader = None
Individual = None
Record = None
Name = None

try:
    from ged4py.parser import GedcomReader
    from ged4py.model import Individual, Record, Name  # Use Name type

    GEDCOM_LIB_AVAILABLE = True
except ImportError:
    logger.error("`ged4py` library not found.")
    logger.error("Please install it: pip install ged4py")
except Exception as import_err:
    logger.error(f"ERROR importing ged4py: {type(import_err).__name__} - {import_err}")
    logger.error("!!!", exc_info=True)

# --- Helper Functions ---


def _is_individual(obj) -> bool:
    """Checks if object is an Individual safely handling None values"""
    return obj is not None and type(obj).__name__ == "Individual"


def _is_record(obj) -> bool:
    """Checks if object is a Record safely handling None values"""
    return obj is not None and type(obj).__name__ == "Record"


def _is_name(obj) -> bool:
    """Checks if object is a Name safely handling None values"""
    return obj is not None and type(obj).__name__ == "Name"


def _normalize_id(xref_id: Optional[str]) -> Optional[str]:
    """Normalizes INDI/FAM etc IDs (e.g., '@I123@' -> 'I123')."""
    if xref_id and isinstance(xref_id, str):
        match = re.match(r"^@?([IFSTNMCXO][0-9A-Z\-]+)@?$", xref_id.strip().upper())
        if match:
            return match.group(1)
    return None


def format_name(name: Optional[str]) -> str:
    """Cleans and formats a person's name string."""
    if not name or not isinstance(name, str):
        return "Valued Relative"
    cleaned_name = name.strip().title()
    cleaned_name = re.sub(r"\s*/([^/]+)/\s*$", r" \1", cleaned_name).strip()
    cleaned_name = re.sub(r"^/", "", cleaned_name).strip()
    cleaned_name = re.sub(r"/$", "", cleaned_name).strip()
    cleaned_name = re.sub(r"\s+", " ", cleaned_name)
    return cleaned_name if cleaned_name else "Valued Relative"


def _get_full_name(indi) -> str:
    """Safely gets formatted name using Name.format(). Handles None/errors."""
    if not _is_individual(indi):
        return "Unknown (Not Individual)"
    try:
        name_rec = indi.name
        if _is_name(name_rec):
            formatted_name = name_rec.format()
            # Use the dedicated format_name function for cleaning
            return format_name(formatted_name)
        elif name_rec is None:
            return "Unknown (No Name Tag)"
        else:
            indi_id_log = _normalize_id(indi.xref_id) if indi.xref_id else "Unknown ID"
            logger.warning(
                f"Indi @{indi_id_log}@ unexpected .name type: {type(name_rec)}"
            )
            return f"Unknown (Type {type(name_rec).__name__})"
    except AttributeError:
        return "Unknown (Attr Error)"
    except Exception as e:
        indi_id_log = _normalize_id(indi.xref_id) if indi.xref_id else "Unknown ID"
        logger.error(f"Error formatting name for @{indi_id_log}@: {e}", exc_info=False)
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
    cleaned = re.sub(r"^\((.+)\)$", r"\1", cleaned).strip()
    return cleaned if cleaned else "N/A"


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
    """Returns formatted birth and death details (date and place) for display."""
    b_date_obj, b_date_str, b_place = _get_event_info(indi, "BIRT")
    b_date_str_cleaned = _clean_display_date(b_date_str)
    birth_info = (
        f"Born: {b_date_str_cleaned if b_date_str_cleaned != 'N/A' else '(Date unknown)'} "
        f"in {b_place if b_place != 'N/A' else '(Place unknown)'}"
    )
    d_date_obj, d_date_str, d_place = _get_event_info(indi, "DEAT")
    d_date_str_cleaned = _clean_display_date(d_date_str)
    death_info = ""
    if d_date_str_cleaned != "N/A":
        death_info = (
            f"   Died: {d_date_str_cleaned} "
            f"in {d_place if d_place != 'N/A' else '(Place unknown)'}"
        )
    return birth_info, death_info


def format_relative_info(relative) -> str:
    """Formats information about a relative (name and life dates) for display."""
    if not _is_individual(relative):
        return "  - (Invalid Relative Data)"
    rel_name = _get_full_name(relative)
    life_info = format_life_dates(relative)
    return f"  - {rel_name}{life_info}"


# --- Cache Building Functions ---


def build_indi_index(reader):
    """Builds a dictionary mapping normalized ID to Individual object."""
    global INDI_INDEX, INDI_INDEX_BUILD_TIME
    if INDI_INDEX:
        return  # Avoid rebuilding
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


def build_family_maps(reader):
    """Builds id_to_parents and id_to_children maps for all individuals."""
    global FAMILY_MAPS_CACHE, FAMILY_MAPS_BUILD_TIME
    if FAMILY_MAPS_CACHE:
        return FAMILY_MAPS_CACHE  # Avoid rebuilding

    start_time = time.time()
    id_to_parents = {}
    id_to_children = {}
    fam_count = 0
    indi_count = 0  # Just for logging context
    for fam in reader.records0("FAM"):
        fam_count += 1
        if not _is_record(fam):
            continue
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

        for child in fam.sub_tags("CHIL"):
            if _is_individual(child) and child.xref_id:
                child_id = _normalize_id(child.xref_id)
                if child_id:
                    id_to_parents.setdefault(child_id, set()).update(parents)
                    for parent_id in parents:
                        if parent_id:
                            id_to_children.setdefault(parent_id, set()).add(child_id)

    for indi in reader.records0("INDI"):
        indi_count += 1  # Log context

    elapsed = time.time() - start_time
    logger.debug(
        f"[PROFILE] Family maps built: {fam_count} FAMs, {indi_count} INDI, {len(id_to_parents)} child->parents, {len(id_to_children)} parent->children in {elapsed:.2f}s"
    )
    FAMILY_MAPS_BUILD_TIME = elapsed
    FAMILY_MAPS_CACHE = (id_to_parents, id_to_children)
    return id_to_parents, id_to_children


# --- ID Lookup and Extraction ---


def extract_and_fix_id(raw_id: Optional[str]) -> Optional[str]:
    """Cleans and validates a raw ID string (e.g., '@I123@', 'F45'). Returns normalized ID or None."""
    if not raw_id or not isinstance(raw_id, str):
        return None
    id_clean = raw_id.strip().strip("@").upper()
    m = re.match(r"^([IFSTNMCXO][0-9A-Z\-]+)$", id_clean)
    if m:
        return m.group(1)
    m2 = re.search(r"([IFSTNMCXO][0-9]+)", id_clean)
    if m2:
        logger.debug(
            f"extract_and_fix_id: Used fallback regex for '{raw_id}' -> '{m2.group(1)}'"
        )
        return m2.group(1)
    logger.warning(f"extract_and_fix_id: Could not extract valid ID from '{raw_id}'")
    return None


def find_individual_by_id(reader, norm_id: Optional[str]):
    """Finds an individual by normalized ID using the pre-built index."""
    global INDI_INDEX
    if not norm_id:
        logger.warning("find_individual_by_id called with invalid norm_id: None")
        return None
    if not INDI_INDEX:
        logger.warning("INDI_INDEX not built, attempting to build now.")
        build_indi_index(reader)  # Attempt to build if missing
        if not INDI_INDEX:
            logger.error("INDI_INDEX build failed. Cannot lookup by ID.")
            return None  # Fallback to linear scan if build fails? Maybe not safe.
    found_indi = INDI_INDEX.get(norm_id)
    if not found_indi:
        logger.debug(
            f"Individual with normalized ID {norm_id} not found in INDI_INDEX."
        )
    return found_indi


# --- Core Data Retrieval Functions ---


def _find_family_records_where_individual_is_child(reader, target_id: str) -> List:
    """Helper: Find FAM records where target_id is a child."""
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


def _find_family_records_where_individual_is_parent(
    reader, target_id: str
) -> List[Tuple]:
    """Helper: Find FAM records where target_id is HUSB or WIFE. Returns (fam_record, is_husband, is_wife)."""
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
    """Gets parents, spouses, children, or siblings using family record lookups."""
    related_individuals: List = []
    unique_related_ids: Set[str] = set()

    if not reader:
        logger.error("get_related_individuals: No reader.")
        return related_individuals
    if not _is_individual(individual) or not individual.xref_id:
        logger.warning(f"get_related_individuals: Invalid input individual.")
        return related_individuals

    target_id = _normalize_id(individual.xref_id)
    if not target_id:
        logger.warning(
            f"get_related_individuals: Cannot normalize target ID {individual.xref_id}"
        )
        return related_individuals

    try:
        if relationship_type == "parents":
            logger.debug(f"Finding parents for {target_id}...")
            parent_families = _find_family_records_where_individual_is_child(
                reader, target_id
            )
            potential_parents = []
            for family_record in parent_families:
                husband = family_record.sub_tag("HUSB")
                wife = family_record.sub_tag("WIFE")
                if _is_individual(husband):
                    potential_parents.append(husband)
                if _is_individual(wife):
                    potential_parents.append(wife)
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
            parent_families = _find_family_records_where_individual_is_child(
                reader, target_id
            )
            potential_siblings = []
            for fam in parent_families:
                fam_children = fam.sub_tags("CHIL")
                if fam_children:
                    potential_siblings.extend(
                        c for c in fam_children if _is_individual(c)
                    )
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
            parent_families = _find_family_records_where_individual_is_parent(
                reader, target_id
            )
            if relationship_type == "spouses":
                logger.debug(f"Finding spouses for {target_id}...")
                for family_record, is_target_husband, is_target_wife in parent_families:
                    other_spouse = None
                    if is_target_husband:
                        other_spouse = family_record.sub_tag("WIFE")
                    elif is_target_wife:
                        other_spouse = family_record.sub_tag("HUSB")
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
            else:  # children
                logger.debug(f"Finding children for {target_id}...")
                for family_record, _, _ in parent_families:
                    children_list = family_record.sub_tags("CHIL")
                    if children_list:
                        for child in children_list:
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

    related_individuals.sort(key=lambda x: (_normalize_id(x.xref_id) or ""))
    return related_individuals


# --- Relationship Path Functions ---


def _reconstruct_path(
    start_id, end_id, meeting_id, visited_fwd, visited_bwd
) -> List[str]:
    """Reconstructs the path from start to end via the meeting point using predecessor maps."""
    path_fwd = []
    curr = meeting_id
    while curr is not None:
        path_fwd.append(curr)
        curr = visited_fwd.get(curr)
    path_fwd.reverse()

    path_bwd = []
    curr = visited_bwd.get(meeting_id)
    while curr is not None:
        path_bwd.append(curr)
        curr = visited_bwd.get(curr)

    path = path_fwd + path_bwd

    if not path:
        logger.error("_reconstruct_path: Failed to reconstruct any path.")
        return []
    if path[0] != start_id:
        logger.warning(
            f"_reconstruct_path: Path doesn't start with start_id ({path[0]} != {start_id}). Prepending."
        )
        path.insert(0, start_id)
    if path[-1] != end_id:
        logger.warning(
            f"_reconstruct_path: Path doesn't end with end_id ({path[-1]} != {end_id}). Appending."
        )
        path.append(end_id)

    logger.debug(f"_reconstruct_path: Final reconstructed path IDs: {path}")
    return path


def explain_relationship_path(
    path_ids: List[str],
    reader,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
) -> str:
    """Return a human-readable explanation of the relationship path with relationship labels."""
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"
    steps = []
    for i in range(len(path_ids) - 1):
        id_a, id_b = path_ids[i], path_ids[i + 1]
        indi_a = find_individual_by_id(reader, id_a)
        indi_b = find_individual_by_id(reader, id_b)
        name_a = _get_full_name(indi_a) if indi_a else f"Unknown ({id_a})"
        name_b = _get_full_name(indi_b) if indi_b else f"Unknown ({id_b})"

        rel = "related"
        label = rel
        if id_b in id_to_parents.get(id_a, set()):  # B is parent of A
            rel = "child"  # A is child of B
            sex_a = getattr(indi_a, "sex", None) if indi_a else None
            label = "daughter" if sex_a == "F" else "son" if sex_a == "M" else "child"
        elif id_b in id_to_children.get(id_a, set()):  # B is child of A
            rel = "parent"  # A is parent of B
            sex_a = getattr(indi_a, "sex", None) if indi_a else None
            label = "mother" if sex_a == "F" else "father" if sex_a == "M" else "parent"

        steps.append(f"{name_a} is the {label} of {name_b}")

    start_person_name = (
        _get_full_name(find_individual_by_id(reader, path_ids[0]))
        or f"Unknown ({path_ids[0]})"
    )
    explanation_str = "\n -> ".join(steps)
    return f"{start_person_name}\n -> {explanation_str}"


def fast_bidirectional_bfs(
    start_id: str,
    end_id: str,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
    max_depth=20,
    node_limit=100000,
    timeout_sec=30,
    log_progress=False,
) -> List[str]:
    """Performs bidirectional BFS using maps & predecessors. Returns path as list of IDs."""
    start_time = time.time()

    queue_fwd: Deque[Tuple[str, int]] = deque([(start_id, 0)])
    visited_fwd: Dict[str, Optional[str]] = {start_id: None}
    queue_bwd: Deque[Tuple[str, int]] = deque([(end_id, 0)])
    visited_bwd: Dict[str, Optional[str]] = {end_id: None}
    processed = 0
    meeting_id: Optional[str] = None

    while queue_fwd and queue_bwd and meeting_id is None:
        if time.time() - start_time > timeout_sec:
            logger.warning(f"  [FastBiBFS] Timeout after {timeout_sec} seconds.")
            return []
        if processed > node_limit:
            logger.warning(f"  [FastBiBFS] Node limit {node_limit} reached.")
            return []

        # Expand Forward
        if queue_fwd:
            current_id_fwd, depth_fwd = queue_fwd.popleft()
            processed += 1
            if log_progress and processed % 5000 == 0:
                logger.info(
                    f"  [FastBiBFS] FWD processed {processed}, Q:{len(queue_fwd)}, D:{depth_fwd}"
                )
            if depth_fwd >= max_depth:
                continue

            neighbors_fwd = id_to_parents.get(
                current_id_fwd, set()
            ) | id_to_children.get(current_id_fwd, set())
            for neighbor_id in neighbors_fwd:
                if neighbor_id in visited_bwd:
                    meeting_id = neighbor_id
                    visited_fwd[neighbor_id] = current_id_fwd
                    logger.debug(
                        f"  [FastBiBFS] Path found (FWD meets BWD) at {meeting_id} after {processed} nodes."
                    )
                    break
                if neighbor_id not in visited_fwd:
                    visited_fwd[neighbor_id] = current_id_fwd
                    queue_fwd.append((neighbor_id, depth_fwd + 1))
            if meeting_id:
                break

        # Expand Backward
        if queue_bwd and meeting_id is None:
            current_id_bwd, depth_bwd = queue_bwd.popleft()
            processed += 1
            if log_progress and processed % 5000 == 0:
                logger.debug(
                    f"  [FastBiBFS] BWD processed {processed}, Q:{len(queue_bwd)}, D:{depth_bwd}"
                )
            if depth_bwd >= max_depth:
                continue

            neighbors_bwd = id_to_parents.get(
                current_id_bwd, set()
            ) | id_to_children.get(current_id_bwd, set())
            for neighbor_id in neighbors_bwd:
                if neighbor_id in visited_fwd:
                    meeting_id = neighbor_id
                    visited_bwd[neighbor_id] = current_id_bwd
                    logger.debug(
                        f"  [FastBiBFS] Path found (BWD meets FWD) at {meeting_id} after {processed} nodes."
                    )
                    break
                if neighbor_id not in visited_bwd:
                    visited_bwd[neighbor_id] = current_id_bwd
                    queue_bwd.append((neighbor_id, depth_bwd + 1))
            # No need to break outer loop here, FWD check handles it

    if meeting_id:
        path_ids = _reconstruct_path(
            start_id, end_id, meeting_id, visited_fwd, visited_bwd
        )
        return path_ids
    else:
        logger.warning(
            f"  [FastBiBFS] No path found between {start_id} and {end_id} after {processed} nodes."
        )
        return []


def get_relationship_path(reader, id1: str, id2: str) -> str:
    """Calculates and formats relationship path using fast bidirectional BFS with pre-built maps."""
    global FAMILY_MAPS_CACHE, FAMILY_MAPS_BUILD_TIME, INDI_INDEX, INDI_INDEX_BUILD_TIME
    id1_norm = _normalize_id(id1)
    id2_norm = _normalize_id(id2)
    if not reader:
        return "Error: GEDCOM Reader unavailable."
    if not id1_norm or not id2_norm:
        return "Invalid input IDs."
    if id1_norm == id2_norm:
        return "Individuals are the same."

    # Ensure caches/maps are built (call respective build functions)
    if FAMILY_MAPS_CACHE is None:
        logger.debug(f"  [Cache] Building family maps (first time)...")
        build_family_maps(reader)  # Builds and sets FAMILY_MAPS_CACHE
        logger.debug(
            f"  [Cache] Maps built and cached in {FAMILY_MAPS_BUILD_TIME:.2f}s."
        )
    if not INDI_INDEX:
        logger.debug(f"  [Cache] Building individual index (first time)...")
        build_indi_index(reader)  # Builds and sets INDI_INDEX
        logger.debug(
            f"  [Cache] Index built and cached in {INDI_INDEX_BUILD_TIME:.2f}s."
        )

    # Retrieve maps from cache after ensuring they are built
    if FAMILY_MAPS_CACHE is None:
        logger.error("Family maps cache failed to build or retrieve.")
        return "Error: Could not build family relationship maps."
    id_to_parents, id_to_children = FAMILY_MAPS_CACHE

    max_depth = 20
    node_limit = 100000
    timeout_sec = 30

    logger.debug(
        f"Calculating relationship path (FastBiBFS): {id1_norm} <-> {id2_norm}"
    )
    logger.debug(f"  [FastBiBFS] Using cached maps & index. Starting search...")
    search_start = time.time()

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

    if not path_ids:
        profile_info = f"[PROFILE] Search: {search_time:.2f}s, Maps: {FAMILY_MAPS_BUILD_TIME:.2f}s, Index: {INDI_INDEX_BUILD_TIME:.2f}s"
        return (
            f"No relationship path found (FastBiBFS could not connect).\n{profile_info}"
        )

    explanation_start = time.time()
    explanation_str = explain_relationship_path(
        path_ids, reader, id_to_parents, id_to_children
    )
    explanation_time = time.time() - explanation_start
    logger.debug(f"[PROFILE] Path explanation built in {explanation_time:.2f}s.")

    profile_info = (
        f"[PROFILE] Total Time: {search_time+explanation_time:.2f}s "
        f"(Search: {search_time:.2f}s, Explain: {explanation_time:.2f}s, "
        f"Maps: {FAMILY_MAPS_BUILD_TIME:.2f}s, Index: {INDI_INDEX_BUILD_TIME:.2f}s)"
    )
    logger.debug(profile_info)
    return f"{explanation_str}\n"


# --- Fuzzy Matching ---


def find_potential_matches(
    reader,
    first_name: Optional[str],
    surname: Optional[str],
    dob_str: Optional[str],
    pob: Optional[str],
    dod_str: Optional[str],
    pod: Optional[str],
    gender: Optional[str] = None,
    max_results: int = 3,
) -> List[Dict]:
    """
    Finds potential matches in GEDCOM based on various criteria including death info.
    Prioritizes name matches. Returns top `max_results`.
    """
    if not reader:
        logger.error("find_potential_matches: No reader.")
        return []
    results: List[Dict] = []
    year_score_range = 1
    year_filter_range = 30

    clean_param = lambda p: re.sub(r"[^\w\s]", "", p).strip() if p else None
    first_name_clean = clean_param(first_name)
    surname_clean = clean_param(surname)
    pob_clean = clean_param(pob)
    pod_clean = clean_param(pod)
    gender_clean = (
        gender.strip().lower()
        if gender and gender.strip().lower() in ("m", "f")
        else None
    )

    logger.debug(
        f"Fuzzy Search: FirstName='{first_name_clean}', Surname='{surname_clean}', "
        f"DOB='{dob_str}', POB='{pob_clean}', DOD='{dod_str}', POD='{pod_clean}', Gender='{gender_clean}'"
    )

    target_first_name_lower = first_name_clean.lower() if first_name_clean else None
    target_surname_lower = surname_clean.lower() if surname_clean else None
    target_pob_lower = pob_clean.lower() if pob_clean else None
    target_pod_lower = pod_clean.lower() if pod_clean else None

    target_birth_year: Optional[int] = None
    birth_dt = _parse_date(dob_str) if dob_str else None
    if birth_dt:
        target_birth_year = birth_dt.year

    target_death_year: Optional[int] = None
    death_dt = _parse_date(dod_str) if dod_str else None
    if death_dt:
        target_death_year = death_dt.year

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
    fuzzy_results = []  # Keep fuzzy results separate initially

    # Use the index if available and built
    individuals_to_check = (
        INDI_INDEX.values() if INDI_INDEX else reader.records0("INDI")
    )
    if not INDI_INDEX:
        logger.warning(
            "INDI_INDEX not available for fuzzy search, using slower linear scan."
        )

    for indi in individuals_to_check:
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
        death_date_obj, death_date_str_ged, death_place_str_ged = _get_event_info(
            indi, "DEAT"
        )
        birth_year_ged: Optional[int] = birth_date_obj.year if birth_date_obj else None
        death_year_ged: Optional[int] = death_date_obj.year if death_date_obj else None

        # Pre-filtering based on years
        birth_year_ok = (
            not target_birth_year
            or birth_year_ged is None
            or abs(birth_year_ged - target_birth_year) <= year_filter_range
        )
        death_year_ok = (
            not target_death_year
            or death_year_ged is None
            or abs(death_year_ged - target_death_year) <= year_filter_range
        )

        # If both target years are present, both must be ok (within range or None)
        if target_birth_year and target_death_year:
            if not (birth_year_ok and death_year_ok):
                continue
        # If only one target year, that one must be ok
        elif target_birth_year and not birth_year_ok:
            continue
        elif target_death_year and not death_year_ok:
            continue
        # If no target years, always passes this filter

        # Scoring
        score = 0
        match_reasons = []
        indi_name_lower = indi_full_name.lower()
        indi_name_parts = indi_name_lower.split()
        indi_first_name = indi_name_parts[0] if indi_name_parts else None
        indi_surname = indi_name_parts[-1] if len(indi_name_parts) > 1 else None

        first_name_match = False
        surname_match = False
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

        # Exact Name Match Path (High Priority)
        if first_name_match and surname_match:
            score = 30  # Base score for exact name
            match_reasons.append("First & Surname")

            # Birth Year Bonus
            birth_year_bonus_match = False
            if (
                target_birth_year
                and birth_year_ged is not None
                and abs(birth_year_ged - target_birth_year) <= year_score_range
            ):
                score += 8
                birth_year_bonus_match = True
                match_reasons.append(
                    f"Birth Year ~{target_birth_year} ({birth_year_ged})"
                )
            # Exact Birth Date Bonus
            if birth_date_obj and birth_dt and birth_date_obj.date() == birth_dt.date():
                score += 5
                match_reasons.append("Exact Birth Date")
            # Death Year Bonus
            death_year_bonus_match = False
            if (
                target_death_year
                and death_year_ged is not None
                and abs(death_year_ged - target_death_year) <= year_score_range
            ):
                score += 8
                death_year_bonus_match = True
                match_reasons.append(
                    f"Death Year ~{target_death_year} ({death_year_ged})"
                )

            # Gender Bonus/Penalty
            indi_gender = getattr(indi, "sex", None)
            gender_bonus_match = False
            if gender_clean and indi_gender:
                indi_gender_lower = str(indi_gender).strip().lower()
                if indi_gender_lower and indi_gender_lower[0] in ("m", "f"):
                    if indi_gender_lower[0] == gender_clean:
                        score += 5
                        gender_bonus_match = True
                        match_reasons.append(
                            f"Gender Match ({indi_gender_lower.upper()})"
                        )
                    else:
                        score -= 5  # Penalty for mismatch
                        match_reasons.append(
                            f"Gender Mismatch ({indi_gender_lower.upper()} vs {gender_clean.upper()})"
                        )

            # Place Bonus
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

            # Boosts
            if birth_year_bonus_match:
                score += 2  # Small boost for name+year

            # Add to exact matches list
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
            continue  # Skip adding to fuzzy results if exact name matched

        # Fuzzy Name Scoring Path (Lower Priority)
        score = 0  # Reset score for fuzzy path
        match_reasons = []

        # Fuzzy First Name
        if target_first_name_lower and indi_first_name:
            ratio = difflib.SequenceMatcher(
                None, target_first_name_lower, indi_first_name
            ).ratio()
            if ratio > 0.8:
                score += int(10 * ratio)
                match_reasons.append(f"First Name~ ({ratio:.2f})")
            elif target_first_name_lower.startswith(
                indi_first_name
            ) or indi_first_name.startswith(target_first_name_lower):
                score += 3
                match_reasons.append("First Name starts")  # Partial match bonus

        # Fuzzy Surname
        if target_surname_lower and indi_surname:
            ratio = difflib.SequenceMatcher(
                None, target_surname_lower, indi_surname
            ).ratio()
            if ratio > 0.8:
                score += int(12 * ratio)
                match_reasons.append(
                    f"Surname~ ({ratio:.2f})"
                )  # Higher weight for surname
            elif target_surname_lower.startswith(
                indi_surname
            ) or indi_surname.startswith(target_surname_lower):
                score += 4
                match_reasons.append("Surname starts")

        # Only proceed with other fuzzy bonuses if there's some name similarity
        if score > 0:
            # Birth Year Bonus (Fuzzy)
            birth_year_bonus_match = False
            if (
                target_birth_year
                and birth_year_ged is not None
                and abs(birth_year_ged - target_birth_year) <= year_score_range
            ):
                score += 5  # Lower bonus than exact match case
                birth_year_bonus_match = True
                match_reasons.append(
                    f"Birth Year ~{target_birth_year} ({birth_year_ged})"
                )
            # Death Year Bonus (Fuzzy)
            death_year_bonus_match = False
            if (
                target_death_year
                and death_year_ged is not None
                and abs(death_year_ged - target_death_year) <= year_score_range
            ):
                score += 5  # Lower bonus
                death_year_bonus_match = True
                match_reasons.append(
                    f"Death Year ~{target_death_year} ({death_year_ged})"
                )
            # Gender Bonus/Penalty (Fuzzy) - Same logic, maybe slightly lower weight? Keep same for now.
            indi_gender = getattr(indi, "sex", None)
            if gender_clean and indi_gender:
                indi_gender_lower = str(indi_gender).strip().lower()
                if indi_gender_lower and indi_gender_lower[0] in ("m", "f"):
                    if indi_gender_lower[0] == gender_clean:
                        score += 3
                        match_reasons.append(
                            f"Gender Match ({indi_gender_lower.upper()})"
                        )
                    else:
                        score -= 3
                        match_reasons.append(
                            f"Gender Mismatch ({indi_gender_lower.upper()} vs {gender_clean.upper()})"
                        )
            # Place Bonus (Fuzzy) - Slightly lower weights
            if target_pob_lower and birth_place_str_ged != "N/A":
                place_lower = birth_place_str_ged.lower()
                if place_lower.startswith(target_pob_lower):
                    score += 2
                    match_reasons.append(f"POB starts '{pob_clean}'")
                elif target_pob_lower in place_lower:
                    score += 1
                    match_reasons.append(f"POB contains '{pob_clean}'")
            if target_pod_lower and death_place_str_ged != "N/A":
                place_lower = death_place_str_ged.lower()
                if place_lower.startswith(target_pod_lower):
                    score += 2
                    match_reasons.append(f"POD starts '{pod_clean}'")
                elif target_pod_lower in place_lower:
                    score += 1
                    match_reasons.append(f"POD contains '{pod_clean}'")

            # Add to fuzzy results if score is positive
            if score > 0:
                reasons_str = ", ".join(sorted(list(set(match_reasons))))
                raw_indi_id_str = f"@{indi.xref_id}@" if indi.xref_id else None
                fuzzy_results.append(
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

    # Prioritize Exact Matches
    if exact_matches:
        exact_matches.sort(key=lambda x: (x["score"], x["birth_date"]), reverse=True)
        limited_results = exact_matches[:max_results]
        logger.debug(
            f"Fuzzy search scanned {candidate_count} individuals. Found {len(exact_matches)} exact name matches. Showing top {len(limited_results)}."
        )
        return limited_results

    # Fallback to Fuzzy Matches if no exact ones
    fuzzy_results.sort(key=lambda x: x["score"], reverse=True)
    limited_results = fuzzy_results[:max_results]
    logger.debug(
        f"Fuzzy search scanned {candidate_count} individuals. Found {len(fuzzy_results)} potential fuzzy matches. Showing top {len(limited_results)}."
    )
    return limited_results


# Example of how to use (if run directly, though unlikely)
if __name__ == "__main__":
    print("This is the gedcom_utils module. Import it into other scripts.")
    # Add basic test or example usage if desired
    # Example: Ensure logging works
    logger.info("gedcom_utils loaded.")
    # Example: Check ged4py status
    if GEDCOM_LIB_AVAILABLE:
        logger.info("ged4py library seems available.")
    else:
        logger.error("ged4py library is NOT available.")
