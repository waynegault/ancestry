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
from utils import SessionManager, _api_req
import html  # Added for unescaping
from bs4 import BeautifulSoup  # Added for HTML parsing
from dotenv import load_dotenv
import html
from bs4 import BeautifulSoup
from logging_config import setup_logging
from rapidfuzz import fuzz, process  # For robust fuzzy matching

# Create a standalone session_manager that can authenticate itself
global session_manager
session_manager = SessionManager()


# Add parent directory to sys.path to import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import local modules


# Setup logging using centralized config
logger = setup_logging(log_file="gedcom_processor.log", log_level="INFO")

# Global cache for family relationships and individual lookup
FAMILY_MAPS_CACHE = None
FAMILY_MAPS_BUILD_TIME = 0
INDI_INDEX = {}  # Index for ID -> Individual object
INDI_INDEX_BUILD_TIME = 0  # Track build time

# Always define GEDCOM_LIB_AVAILABLE at the top
GEDCOM_LIB_AVAILABLE = False

# Load environment variables
load_dotenv()
MY_TREE_ID = os.getenv("MY_TREE_ID")


# --- Third-party Imports ---
try:
    from ged4py.parser import GedcomReader
    from ged4py.model import Individual, Record, Name  # Use Name type

    GEDCOM_LIB_AVAILABLE = True  # Ensure this is set on successful import
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


try:
    from config import config_instance
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
from utils import (
    _is_individual,
    _is_record,
    _is_name,
    _normalize_id,
    _get_full_name,
    _parse_date,
    _clean_display_date,
    _get_event_info,
    build_indi_index,
    build_family_maps,
    extract_and_fix_id,
    find_individual_by_id,
    format_life_dates,
    format_full_life_details,
    format_relative_info,
    _find_family_records_where_individual_is_child,
    _find_family_records_where_individual_is_parent,
    get_related_individuals,
)


# --- Core Data Retrieval Functions ---


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
                parent_indi is not None
                and _is_individual(parent_indi)
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
                if (
                    parent_indi is not None
                    and _is_individual(parent_indi)
                    and hasattr(parent_indi, "xref_id")
                    and parent_indi.xref_id
                ):
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
        # --- Fuzzy Name Matching ---
        fuzzy_name_score = 0
        fuzzy_first = 0
        fuzzy_surname = 0
        FUZZY_NAME_THRESHOLD = 70  # Lowered from 80 to 70 for more tolerant matching
        FUZZY_PLACE_THRESHOLD = 70
        if target_first_name_lower and indi_first_name:
            fuzzy_first = fuzz.ratio(target_first_name_lower, indi_first_name)
        if target_surname_lower and indi_surname:
            fuzzy_surname = fuzz.ratio(target_surname_lower, indi_surname)
        if fuzzy_first >= FUZZY_NAME_THRESHOLD:
            score += 10
            match_reasons.append(f"Fuzzy First Name ({fuzzy_first})")
        if fuzzy_surname >= FUZZY_NAME_THRESHOLD:
            score += 10
            match_reasons.append(f"Fuzzy Surname ({fuzzy_surname})")
        if (
            fuzzy_first >= FUZZY_NAME_THRESHOLD
            and fuzzy_surname >= FUZZY_NAME_THRESHOLD
        ):
            score += 10
            match_reasons.append("Strong Fuzzy Name Match")
        # ...existing code for gender, year, place, etc...
        # --- Fuzzy Place Matching ---
        if target_pob_lower and birth_place_str_ged != "N/A":
            fuzzy_pob = fuzz.partial_ratio(
                target_pob_lower, birth_place_str_ged.lower()
            )
            if fuzzy_pob >= FUZZY_PLACE_THRESHOLD:
                score += 3
                match_reasons.append(f"Fuzzy POB ({fuzzy_pob})")
        if target_pod_lower and death_place_str_ged != "N/A":
            fuzzy_pod = fuzz.partial_ratio(
                target_pod_lower, death_place_str_ged.lower()
            )
            if fuzzy_pod >= FUZZY_PLACE_THRESHOLD:
                score += 3
                match_reasons.append(f"Fuzzy POD ({fuzzy_pod})")
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
            print(
                "Failed to authenticate with Ancestry. Please login manually when the browser opens."
            )
            input("Press Enter after you've logged in manually...")
        else:
            print("Authentication successful.")
    if not session_manager.my_tree_id:
        print("Loading tree information...")
        session_manager._retrieve_identifiers()
        if not session_manager.my_tree_id:
            print("WARNING: Could not load tree ID. Some functionality may be limited.")
        else:
            print(f"Tree ID loaded successfully: {session_manager.my_tree_id}")
    return session_manager.session_ready


# End of function initialize_session


class AncestryAPISearch:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.base_url = "https://www.ancestry.co.uk"

    def _get_tree_id(self):
        if not self.session_manager.my_tree_id:
            self.session_manager._retrieve_identifiers()
        return self.session_manager.my_tree_id

    def _extract_display_name(self, person: dict) -> str:
        names = person.get("Names") or person.get("names")
        if names and isinstance(names, list) and names:
            given = names[0].get("g") or ""
            surname = names[0].get("s") or ""
            full_name = f"{given} {surname}".strip()
            if full_name:
                return full_name
        given = person.get("gname") or ""
        surname = person.get("sname") or ""
        if given or surname:
            return f"{given} {surname}".strip()
        return str(person.get("pid") or person.get("id") or "(Unknown)")

    def search_by_name(self, name: str, limit: int = 10) -> list[dict]:
        import urllib.parse

        tree_id = self._get_tree_id()
        if not tree_id:
            return []
        base_url = self.base_url
        query = name.strip()
        tags = ""
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
        name = self._extract_display_name(person)
        gender = None
        if "Genders" in person and person["Genders"]:
            gender = person["Genders"][0].get("g")
        elif "gender" in person:
            gender = person["gender"]
        gender_str = f"Gender: {gender}" if gender else "Gender: Unknown"
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
        pid = person.get("pid") or person.get("id") or person.get("gid", {}).get("v")
        pid_str = f"Person ID: {pid}" if pid else ""
        lines = [f"Name: {name}", gender_str]
        if birth_info:
            lines.append(birth_info)
        if pid_str:
            lines.append(pid_str)
        return "\n".join(lines)

    def get_relationship_ladder(self, person_id: str):
        import re, json, time

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
                match = re.search(r"\\((\{.*\})\\)", response, re.DOTALL)
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
                            return {"error": "JSON decode failed."}
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

    def parse_ancestry_ladder_html(self, html):
        soup = BeautifulSoup(html, "html.parser")
        ladder_data = {}
        rel_elem = soup.select_one(
            "ul.textCenter > li:first-child > i > b"
        ) or soup.select_one("ul.textCenter > li > i > b")
        if rel_elem:
            ladder_data["actual_relationship"] = rel_elem.get_text(strip=True).title()
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


def handle_api_report():
    """Handler for Option 2 - API Report (Ancestry Online) with advanced search and polished output."""
    print("\n--- Person Details & Relationship to WGG (API) ---")
    if not initialize_session():
        print("Failed to initialize session. Cannot proceed with API operations.")
        return
    api_search = AncestryAPISearch(session_manager)
    print("\nEnter as many details as you know. Leave blank to skip a field.")
    first_name = input("First name: ").strip() or None
    surname = input("Surname (or maiden name): ").strip() or None
    dob_str = input("Date of birth (YYYY-MM-DD or year): ").strip() or None
    pob = input("Place of birth: ").strip() or None
    gender = input("Gender (M/F, optional): ").strip().lower() or None
    if gender and gender not in ("m", "f"):
        gender = None
    # Optionally allow death info (API may not always provide these)
    dod_str = input("Date of death (YYYY-MM-DD or year, optional): ").strip() or None
    pod = input("Place of death (optional): ").strip() or None

    # Build search query for API (use first_name + surname for best match)
    query = " ".join([x for x in [first_name, surname] if x]).strip()
    if not query:
        print("Search cancelled.")
        return
    persons = api_search.search_by_name(query)
    if not persons:
        print("\nNo matches found in Ancestry API.")
        return

    # Score and filter candidates using additional fields
    def score_api_candidate(person):
        score = 0
        reasons = []
        # Name matching
        display_name = api_search._extract_display_name(person).lower()
        if first_name and first_name.lower() in display_name:
            score += 10
            reasons.append("First name match")
        if surname and surname.lower() in display_name:
            score += 10
            reasons.append("Surname match")
        # Gender
        api_gender = None
        if "Genders" in person and person["Genders"]:
            api_gender = person["Genders"][0].get("g", "").lower()
        elif "gender" in person:
            api_gender = str(person["gender"]).lower()
        if gender and api_gender and gender == api_gender[0]:
            score += 5
            reasons.append(f"Gender match ({api_gender})")
        elif gender and api_gender and gender != api_gender[0]:
            score -= 5
            reasons.append(f"Gender mismatch ({api_gender})")
        # Birth info
        events = person.get("Events") or person.get("events") or []
        birth_date = None
        birth_place = None
        for event in events:
            if (
                event.get("t", "").lower() == "birth"
                or event.get("type", "").lower() == "birth"
            ):
                birth_date = event.get("d") or event.get("date") or event.get("nd")
                # Try all possible keys and handle nested dicts for place
                birth_place = (
                    event.get("p")
                    or event.get("place")
                    or (event.get("pl") if "pl" in event else None)
                )
                if isinstance(birth_place, dict):
                    birth_place = birth_place.get("v") or birth_place.get("name")
                if not birth_place:
                    birth_place = "?"
                break
        if dob_str and birth_date and dob_str in str(birth_date):
            score += 8
            reasons.append(f"Birth date match ({birth_date})")
        if pob and birth_place and pob.lower() in birth_place.lower():
            score += 4
            reasons.append(f"Place of birth match ({birth_place})")
        # Death info (rare in API, but check)
        death_date = None
        death_place = None
        for event in events:
            if (
                event.get("t", "").lower() == "death"
                or event.get("type", "").lower() == "death"
            ):
                death_date = event.get("d") or event.get("date") or event.get("nd")
                death_place = event.get("p") or event.get("place")
                break
        if dod_str and death_date and dod_str in str(death_date):
            score += 4
            reasons.append(f"Death date match ({death_date})")
        if pod and death_place and pod.lower() in death_place.lower():
            score += 2
            reasons.append(f"Place of death match ({death_place})")
        return score, ", ".join(reasons)

    # Score and sort candidates
    scored = []
    for person in persons:
        score, reasons = score_api_candidate(person)
        scored.append((score, reasons, person))
    scored.sort(reverse=True, key=lambda x: x[0])
    shortlist = scored[:5]

    print(
        f"\nFound {len(shortlist)} potential match{'es' if len(shortlist) != 1 else ''}:"
    )
    for i, (score, reasons, person) in enumerate(shortlist):
        display_name = api_search._extract_display_name(person)
        events = person.get("Events") or person.get("events") or []
        birth_date = birth_place = death_date = death_place = ""
        for event in events:
            if (
                event.get("t", "").lower() == "birth"
                or event.get("type", "").lower() == "birth"
            ):
                birth_date = (
                    event.get("d") or event.get("date") or event.get("nd") or "?"
                )
                birth_place = (
                    event.get("Place") or event.get("place") or event.get("p") or "?"
                )
            if (
                event.get("t", "").lower() == "death"
                or event.get("type", "").lower() == "death"
            ):
                death_date = (
                    event.get("d") or event.get("date") or event.get("nd") or "?"
                )
                death_place = event.get("p") or event.get("place") or "?"
        line = f"  {i+1}. {display_name}\n     Born : {birth_date} in {birth_place}"
        if (death_date and death_date != "?" and death_date.strip()) or (
            death_place and death_place != "?" and death_place.strip()
        ):
            line += f"\n     Died : {death_date} in {death_place}"
        if reasons:
            line += f"\n     Reasons: {reasons}"
        print(line)
    # Auto-select if only one match or if top match has a unique score
    if len(shortlist) == 1 or (
        len(shortlist) > 1 and shortlist[0][0] != shortlist[1][0]
    ):
        selected_person = shortlist[0][2]
        print(f"\nAuto-selected: {api_search._extract_display_name(selected_person)}")
    else:
        try:
            choice = int(input("\nSelect person (or 0 to cancel): "))
            if choice < 1 or choice > len(shortlist):
                print("Selection cancelled or invalid.")
                return
            selected_person = shortlist[choice - 1][2]
        except ValueError:
            print("Invalid selection. Please enter a number.")
            return
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(
                f"Error in handle_api_report: {e}", exc_info=True
            )
            print(f"Error: {type(e).__name__}: {e}")
            return
    print("\n=== PERSON DETAILS ===")
    print(f"Name: {api_search._extract_display_name(selected_person)}")
    # Gender
    api_gender = None
    if "Genders" in selected_person and selected_person["Genders"]:
        api_gender = selected_person["Genders"][0].get("g")
    elif "gender" in selected_person:
        api_gender = selected_person["gender"]
    print(f"Gender: {api_gender.upper() if api_gender else 'N/A'}")
    # Birth/Death info
    events = selected_person.get("Events") or selected_person.get("events") or []
    birth_date = birth_place = death_date = death_place = None
    for event in events:
        # Use the most complete birth event (with place)
        if (
            event.get("t", "").lower() == "birth"
            or event.get("type", "").lower() == "birth"
        ) or (event.get("TypeString", "").lower() == "birth"):
            birth_date = (
                event.get("d")
                or event.get("date")
                or event.get("nd")
                or event.get("Date")
            )
            birth_place = event.get("Place") or event.get("place") or event.get("p")
            if not birth_place and "PlaceGpids" in event and event["PlaceGpids"]:
                pass
            if birth_place:
                break
    if not birth_place:
        for event in events:
            if (
                event.get("t", "").lower() == "birth"
                or event.get("type", "").lower() == "birth"
            ) or (event.get("TypeString", "").lower() == "birth"):
                birth_place = event.get("Place") or event.get("place") or event.get("p")
                if birth_place:
                    break
    print(
        f"Birth : {birth_date if birth_date else '?'} in {birth_place if birth_place else '?'}"
    )
    # Only print death info if at least one is not missing/blank/?
    for event in events:
        if (
            event.get("t", "").lower() == "death"
            or event.get("type", "").lower() == "death"
        ):
            death_date = event.get("d") or event.get("date") or event.get("nd")
            death_place = event.get("p") or event.get("place")
            break
    if (death_date and str(death_date).strip() and death_date != "?") or (
        death_place and str(death_place).strip() and death_place != "?"
    ):
        print(
            f"Death : {death_date if death_date else '?'} in {death_place if death_place else '?'}"
        )
    # Build and display Ancestry link in tree
    tree_id = api_search._get_tree_id()
    selected_id = (
        selected_person.get("pid")
        or selected_person.get("id")
        or (
            selected_person.get("gid", {}).get("v")
            if isinstance(selected_person.get("gid"), dict)
            else None
        )
    )
    ancestry_id = (
        selected_id[1:]
        if selected_id
        and selected_id[0] in ("I", "F", "S", "T", "N", "M", "C", "X", "O")
        else selected_id
    )
    if tree_id and ancestry_id:
        print(
            f"Link in Tree: https://www.ancestry.co.uk/family-tree/person/tree/{tree_id}/person/{ancestry_id}/facts"
        )
    else:
        print(f"Link in Tree: (unavailable)")

    # --- Fetch and show family details (parents, spouses, children) if available ---
    # Use the browser facts page JSON endpoint for family details
    def fetch_facts_json(profile_id, tree_id, person_id):
        url = f"https://www.ancestry.co.uk/family-tree/person/facts/user/{profile_id}/tree/{tree_id}/person/{person_id}"
        try:
            return _api_req(
                url=url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                headers=None,
                use_csrf_token=False,
                api_description="Ancestry Facts JSON Endpoint",
                referer_url=f"https://www.ancestry.co.uk/family-tree/tree/{tree_id}/family",
                timeout=20,
            )
        except Exception:
            return None

    # Extract profile_id, tree_id, person_id
    profile_id = (
        getattr(session_manager, "my_profile_id", None)
        or "07bdd45e-0006-0000-0000-000000000000"
    )
    tree_id = api_search._get_tree_id()
    person_id = ancestry_id
    facts_json = None
    if profile_id and tree_id and person_id:
        facts_json = fetch_facts_json(profile_id, tree_id, person_id)
    if facts_json and isinstance(facts_json, dict):
        # Try to extract family details from the JSON structure
        family = facts_json.get("family") or facts_json.get("Family")
        if family:

            def print_family_section(label, people):
                print(f"\n{label}:")
                if people:
                    for rel in people:
                        name = (
                            rel.get("displayName")
                            or rel.get("fullName")
                            or rel.get("name")
                            or rel.get("gname")
                            or rel.get("sname")
                            or rel.get("id")
                            or "(Unknown)"
                        )
                        life = rel.get("birthDate") or rel.get("birth") or ""
                        if rel.get("deathDate") or rel.get("death"):
                            life += f" - {rel.get('deathDate') or rel.get('death')}"
                        print(f"  - {name}{f' ({life})' if life else ''}")
                else:
                    print("  (None found)")

            print_family_section("Parents", family.get("parents", []))
            print_family_section("Spouse(s)", family.get("spouses", []))
            print_family_section("Children", family.get("children", []))
        else:
            print("\n(Family details not available in facts JSON response.)")
    else:
        facts_url = f"https://www.ancestry.co.uk/family-tree/person/facts/user/{profile_id}/tree/{tree_id}/person/{person_id}"
        print(
            "\n(Family details could not be retrieved. You can view full family details in your browser:"
        )
        print(f"  {facts_url}\n")


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
                    handle_api_report()  # API handler doesn't need GEDCOM reader/WGG object

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
    reader, wayne_gault_indi, wayne_gault_id_gedcom, max_results=5
):
    """Handler for Option 1 - GEDCOM Report (now with unified input/output style and advanced search fields)."""
    print("\n--- Person Details & Relationship to WGG (GEDCOM) ---")
    if not wayne_gault_indi or not wayne_gault_id_gedcom:
        print("ERROR: Wayne Gordon Gault (reference person) not found in local GEDCOM.")
        print("Cannot calculate relationships accurately.")
        return

    # Advanced input prompts for fuzzy search
    print("\nEnter as many details as you know. Leave blank to skip a field.")
    first_name = input("First name: ").strip() or None
    surname = input("Surname (or maiden name): ").strip() or None
    dob_str = input("Date of birth (YYYY-MM-DD or year): ").strip() or None
    pob = input("Place of birth: ").strip() or None
    gender = input("Gender (M/F, optional): ").strip().lower() or None
    if gender and gender not in ("m", "f"):
        gender = None
    # Optionally allow death info
    dod_str = input("Date of death (YYYY-MM-DD or year, optional): ").strip() or None
    pod = input("Place of death (optional): ").strip() or None

    matches = find_potential_matches(
        reader, first_name, surname, dob_str, pob, dod_str, pod, gender
    )
    if not matches:
        print("\nNo matches found in GEDCOM.")
        return
    print(f"\nFound {len(matches)} potential matches:")
    for i, match in enumerate(matches[:max_results]):
        print(f"  {i+1}. {match['name']}")
        print(f"     Born : {match['birth_date']} in {match['birth_place']}")
        if (match["death_date"] and match["death_date"] != "N/A") or (
            match["death_place"] and match["death_place"] != "N/A"
        ):
            print(f"     Died : {match['death_date']} in {match['death_place']}")
        print(f"     Reasons: {match['reasons']}")
    # Auto-select if only one match or if top match has a unique score
    if len(matches) == 1 or (
        len(matches) > 1 and matches[0]["score"] != matches[1]["score"]
    ):
        selected_match = matches[0]
        print(f"\nAuto-selected: {selected_match['name']}")
    else:
        try:
            choice = int(input("\nSelect person (or 0 to cancel): "))
            if choice < 1 or choice > len(matches[:max_results]):
                print("Selection cancelled or invalid.")
                return
            selected_match = matches[choice - 1]
        except ValueError:
            print("Invalid selection. Please enter a number.")
            return
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(
                f"Error in handle_gedcom_report: {e}", exc_info=True
            )
            print(f"Error: {type(e).__name__}: {e}")
            return
    selected_id = extract_and_fix_id(selected_match["id"])
    if not selected_id:
        print("ERROR: Invalid ID in selected match.")
        return
    selected_indi = find_individual_by_id(reader, selected_id)
    if not selected_indi:
        print("ERROR: Could not retrieve individual record from GEDCOM.")
        return
    # --- Polished person details output ---
    print("\n=== PERSON DETAILS ===")
    print(f"Name: {selected_match['name']}")
    print(f"Gender: {gender.upper() if gender else 'N/A'}")
    print(f"Birth : {selected_match['birth_date']} in {selected_match['birth_place']}")
    if (selected_match["death_date"] and selected_match["death_date"] != "N/A") or (
        selected_match["death_place"] and selected_match["death_place"] != "N/A"
    ):
        print(
            f"Death : {selected_match['death_date']} in {selected_match['death_place']}"
        )
    # Build and display Ancestry link in tree
    tree_id = MY_TREE_ID or getattr(session_manager, "my_tree_id", None)
    ancestry_id = (
        selected_id[1:]
        if selected_id
        and selected_id[0] in ("I", "F", "S", "T", "N", "M", "C", "X", "O")
        else selected_id
    )
    if tree_id and ancestry_id:
        print(
            f"Link in Tree: https://www.ancestry.co.uk/family-tree/person/tree/{tree_id}/person/{ancestry_id}/facts"
        )
    else:
        print(f"Link in Tree: (unavailable)")
    # --- Show family details (parents, siblings, spouses, children) ---
    display_gedcom_family_details(reader, selected_indi)
    # --- Unified relationship path output ---
    print("\nLooking up relationship information...")
    relationship_path = get_relationship_path(
        reader, selected_id, wayne_gault_id_gedcom
    )
    print("\nRelationship Path:")
    print(relationship_path.strip())


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
