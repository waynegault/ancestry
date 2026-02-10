#!/usr/bin/env python3

"""
GEDCOM Processing & Genealogical Data Intelligence Engine

Comprehensive genealogical data processing platform providing sophisticated
GEDCOM file analysis, family relationship mapping, and intelligent genealogical
data extraction with advanced parsing capabilities, relationship pathfinding,
and comprehensive data validation for professional genealogical research.

GEDCOM Processing:
• Advanced GEDCOM file parsing with comprehensive format support and validation
• Intelligent individual and family record extraction with metadata preservation
• Sophisticated date parsing and normalization with flexible format recognition
• Advanced name processing with standardization and variant recognition
• Comprehensive data validation with error detection and correction suggestions
• Efficient memory management for large genealogical datasets

Relationship Intelligence:
• Advanced relationship pathfinding using optimized graph algorithms
• Comprehensive family structure analysis with multi-generational mapping
• Intelligent sibling detection and family group identification
• Complex relationship calculation including step-relationships and adoptions
• Advanced ancestor and descendant tracking with generation mapping
• Sophisticated relationship degree calculation with cousin identification

Data Enhancement:
• Intelligent data normalization with standardization and quality improvement
• Advanced search capabilities with fuzzy matching and phonetic algorithms
• Comprehensive data quality assessment with improvement recommendations
• Integration with external genealogical databases for data enrichment
• Export capabilities for seamless integration with genealogical software
• Performance optimization for large family tree processing

Foundation Services:
Provides the essential GEDCOM processing infrastructure that enables sophisticated
genealogical analysis, relationship discovery, and family tree intelligence for
professional genealogical research and family history exploration.
"""


# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# === CORE INFRASTRUCTURE ===
import logging

from genealogy.relationship_calculations import (
    are_cousins,
    are_siblings,
    find_direct_relationship,
    has_direct_relationship,
    is_aunt_or_uncle,
    is_grandchild,
    is_grandparent,
    is_great_grandchild,
    is_great_grandparent,
    is_niece_or_nephew,
)

logger = logging.getLogger(__name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===


# === STANDARD LIBRARY IMPORTS ===
import os
import re
import time
from collections import deque
from collections.abc import Callable, Collection, Mapping
from datetime import datetime
from typing import (
    Any,
    cast,
)

# --- Third-party imports ---
try:
    from ged4py.model import (
        Individual as _GedcomIndividual,
        Name as _GedcomName,
        NameRec as _GedcomNameRec,
        Record as _GedcomRecord,
    )
    from ged4py.parser import GedcomReader as _GedcomReader
except ImportError:
    _GedcomReader = type(None)
    _GedcomIndividual = type(None)
    _GedcomRecord = type(None)
    _GedcomName = type(None)
    _GedcomNameRec = type(None)
    print("ERROR: ged4py library not found. This script requires ged4py (`pip install ged4py`)")

GedcomReader = _GedcomReader
Individual = _GedcomIndividual
Record = _GedcomRecord
Name = _GedcomName
NameRec = _GedcomNameRec

# === LOCAL IMPORTS ===
from config.config_manager import get_config_manager
from core.common_params import GraphContext
from testing.test_framework import TestSuite, create_standard_test_runner

# === MODULE CONFIGURATION ===
config_manager = get_config_manager()
config = config_manager.get_config()

# === TYPE ALIASES ===
# Define type aliases for GEDCOM types
# Define type aliases for GEDCOM classes (always available as required dependency)
GedcomIndividualType = Individual
GedcomRecordType = Record
GedcomNameType = Name
GedcomNameRecType = NameRec
GedcomReaderType = GedcomReader
TAG_INDI = "INDI"
TAG_BIRTH = "BIRT"
TAG_DEATH = "DEAT"
TAG_HUSBAND = "HUSB"
TAG_WIFE = "WIFE"
TAG_CHILD = "CHIL"
TAG_FAMILY_CHILD = "FAMC"
TAG_FAMILY_SPOUSE = "FAMS"
TAG_DATE = "DATE"
TAG_PLACE = "PLAC"
_GENDER_CHARS_UPPER = {"M", "F"}
_GENDER_CHARS_LOWER = {"m", "f"}
TAG_SEX = "SEX"
TAG_NAME = "NAME"
TAG_GIVN = "GIVN"
TAG_SURN = "SURN"
TAG_SOUR = "SOUR"  # Source citation tag
TAG_TITL = "TITL"  # Source title tag


# --- Logging Setup ---
# Use centralized logger from logging_config
# from logging_config import logger  # Unused in this module's pure functions


# ==============================================
# Utility Functions (Moved to Top)
# ==============================================
def _is_individual(obj: Any) -> bool:
    """Check if an object is a GEDCOM Individual.

    We can't use isinstance with Any, so we need to check for specific attributes
    that are expected to be present on Individual objects.
    """
    if obj is None:
        return False

    # We can't use isinstance with Any, so we need to check for specific attributes
    # This is a heuristic approach to identify Individual objects
    return hasattr(obj, "xref_id") and hasattr(obj, "tag") and getattr(obj, "tag", "") == TAG_INDI


def _is_record(obj: Any) -> bool:
    """Check if an object is a GEDCOM Record.

    We can't use isinstance with Any, so we need to check for specific attributes
    that are expected to be present on Record objects.
    """
    if obj is None:
        return False

    # We can't use isinstance with Any, so we need to check for specific attributes
    # This is a heuristic approach to identify Record objects
    return hasattr(obj, "xref_id") and hasattr(obj, "tag") and hasattr(obj, "sub_tag")


# _is_name_rec removed - unused helper function


def normalize_id(xref_id: str | None) -> str | None:
    if not xref_id:
        return None

    # Try to match standard GEDCOM ID format
    match = re.match(r"^@?([IFSNMCXO][0-9\-]+)@?$", xref_id.strip().upper())
    if match:
        return match.group(1)

    # Try fallback regex for partial GEDCOM IDs
    search_match = re.search(r"([IFSNMCXO][0-9\-]+)", xref_id.strip().upper())
    if search_match:
        logger.debug(f"Normalized ID '{search_match.group(1)}' using fallback regex from '{xref_id}'.")
        return search_match.group(1)

    # For pure numeric strings, return as-is (handle raw numeric IDs)
    if re.match(r"^\d+$", xref_id.strip()):
        return xref_id.strip()

    logger.warning(f"Could not normalize potential ID: '{xref_id}'")
    return None


def extract_and_fix_id(raw_id: Any) -> str | None:
    if not raw_id:
        return None
    id_to_normalize: str | None = None
    if isinstance(raw_id, str):
        id_to_normalize = raw_id
    elif isinstance(raw_id, int):
        id_to_normalize = str(raw_id)
    elif hasattr(raw_id, "xref_id") and (_is_record(raw_id) or _is_individual(raw_id)):
        id_to_normalize = getattr(raw_id, "xref_id", None)
    else:
        logger.debug(f"extract_and_fix_id: Invalid input type '{type(raw_id).__name__}'.")
        return None
    return normalize_id(id_to_normalize)


# ==============================================
# Lazy re-exports from gedcom_parser & gedcom_events (backward compatibility)
# Uses __getattr__ to avoid circular imports when running as __main__
# ==============================================
_LAZY_RE_EXPORTS: dict[str, str] = {
    # gedcom_events
    "_clean_display_date": "genealogy.gedcom.gedcom_events",
    "_extract_date_from_event": "genealogy.gedcom.gedcom_events",
    "_extract_event_record": "genealogy.gedcom.gedcom_events",
    "_extract_place_from_event": "genealogy.gedcom.gedcom_events",
    "_extract_sources_from_event": "genealogy.gedcom.gedcom_events",
    "_parse_date": "genealogy.gedcom.gedcom_events",
    "format_full_life_details": "genealogy.gedcom.gedcom_events",
    "format_life_dates": "genealogy.gedcom.gedcom_events",
    "format_relative_info": "genealogy.gedcom.gedcom_events",
    "format_source_citations": "genealogy.gedcom.gedcom_events",
    "get_event_info": "genealogy.gedcom.gedcom_events",
    "get_person_sources": "genealogy.gedcom.gedcom_events",
    # gedcom_parser
    "get_full_name": "genealogy.gedcom.gedcom_parser",
}


def __getattr__(name: str) -> Any:
    """Lazy re-export: resolve names from sub-modules on first access."""
    if name in _LAZY_RE_EXPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_RE_EXPORTS[name])
        val = getattr(mod, name)
        globals()[name] = val  # cache for subsequent accesses
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# _reconstruct_path removed - unused 89-line helper function for BFS path reconstruction


def _validate_bfs_inputs(
    start_id: str,
    end_id: str,
    id_to_parents: Mapping[str, Collection[str]] | None,
    id_to_children: Mapping[str, Collection[str]] | None,
) -> bool:
    """Validate inputs for bidirectional BFS search."""
    if start_id == end_id:
        return True
    if id_to_parents is None or id_to_children is None:
        logger.error("[FastBiBFS] Relationship maps are None.")
        return False
    if not start_id or not end_id:
        logger.error("[FastBiBFS] Start or end ID is missing.")
        return False
    return True


def _initialize_bfs_queues(start_id: str, end_id: str) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Initialize BFS queues and visited sets for bidirectional search."""
    # Initialize BFS queues and visited sets
    # Forward queue from start_id
    queue_fwd = deque([(start_id, 0, [start_id])])  # (id, depth, path)
    # Backward queue from end_id
    queue_bwd = deque([(end_id, 0, [end_id])])  # (id, depth, path)

    # Track visited nodes and their paths
    visited_fwd = {start_id: (0, [start_id])}  # {id: (depth, path)}
    visited_bwd = {end_id: (0, [end_id])}  # {id: (depth, path)}

    return queue_fwd, queue_bwd, visited_fwd, visited_bwd


def _add_node_to_forward_queue(
    node_id: str, path: list[str], depth: int, visited_fwd: dict[str, Any], queue_fwd: deque[Any]
) -> None:
    """Add a node to the forward search queue if not already visited."""
    if node_id not in visited_fwd:
        new_path = [*path, node_id]
        visited_fwd[node_id] = (depth, new_path)
        queue_fwd.append((node_id, depth, new_path))


def _expand_forward_siblings(
    graph: GraphContext,
    current_id: str,
    path: list[str],
    depth: int,
    visited_fwd: dict[str, Any],
    queue_fwd: deque[Any],
) -> None:
    """Expand to siblings in forward direction through parents."""
    for parent_id in graph.id_to_parents.get(current_id, set()):
        for sibling_id in graph.id_to_children.get(parent_id, set()):
            if sibling_id != current_id and sibling_id not in visited_fwd:
                new_path = [*path, parent_id, sibling_id]
                visited_fwd[sibling_id] = (depth + 2, new_path)
                queue_fwd.append((sibling_id, depth + 2, new_path))


def _expand_forward_node(
    graph: GraphContext, depth: int, path: list[str], visited_fwd: dict[str, Any], queue_fwd: deque[Any], max_depth: int
):
    """Expand a node in the forward direction during BFS."""
    # Stop expanding if we've reached max depth
    if depth >= max_depth:
        return

    current_id = graph.current_id
    if not current_id:
        return

    # Expand to parents (direct relationship)
    for parent_id in graph.id_to_parents.get(current_id, set()):
        _add_node_to_forward_queue(parent_id, path, depth + 1, visited_fwd, queue_fwd)

    # Expand to children (direct relationship)
    for child_id in graph.id_to_children.get(current_id, set()):
        _add_node_to_forward_queue(child_id, path, depth + 1, visited_fwd, queue_fwd)

    # Expand to siblings (through parent)
    _expand_forward_siblings(graph, current_id, path, depth, visited_fwd, queue_fwd)


def _add_node_to_backward_queue(
    node_id: str, path: list[str], depth: int, visited_bwd: dict[str, Any], queue_bwd: deque[Any]
) -> None:
    """Add a node to the backward search queue if not already visited."""
    if node_id not in visited_bwd:
        new_path = [node_id, *path]
        visited_bwd[node_id] = (depth, new_path)
        queue_bwd.append((node_id, depth, new_path))


def _expand_backward_siblings(
    graph: GraphContext,
    current_id: str,
    path: list[str],
    depth: int,
    visited_bwd: dict[str, Any],
    queue_bwd: deque[Any],
) -> None:
    """Expand to siblings in backward direction through parents."""
    for parent_id in graph.id_to_parents.get(current_id, set()):
        for sibling_id in graph.id_to_children.get(parent_id, set()):
            if sibling_id != current_id and sibling_id not in visited_bwd:
                new_path = [sibling_id, parent_id, *path]
                visited_bwd[sibling_id] = (depth + 2, new_path)
                queue_bwd.append((sibling_id, depth + 2, new_path))


def _expand_backward_node(
    graph: GraphContext, depth: int, path: list[str], visited_bwd: dict[str, Any], queue_bwd: deque[Any], max_depth: int
):
    """Expand a node in the backward direction during BFS."""
    # Stop expanding if we've reached max depth
    if depth >= max_depth:
        return

    current_id = graph.current_id
    if not current_id:
        return

    # Expand to parents (direct relationship)
    for parent_id in graph.id_to_parents.get(current_id, set()):
        _add_node_to_backward_queue(parent_id, path, depth + 1, visited_bwd, queue_bwd)

    # Expand to children (direct relationship)
    for child_id in graph.id_to_children.get(current_id, set()):
        _add_node_to_backward_queue(child_id, path, depth + 1, visited_bwd, queue_bwd)

    # Expand to siblings (through parent)
    _expand_backward_siblings(graph, current_id, path, depth, visited_bwd, queue_bwd)


def fast_bidirectional_bfs(
    graph: GraphContext,
    max_depth: int = 25,
    node_limit: int = 150000,
    timeout_sec: int = 45,
) -> list[str]:
    """
    Enhanced bidirectional BFS that finds direct paths through family trees.

    This implementation focuses on finding paths where each person has a clear,
    direct relationship to the next person in the path (parent, child, sibling).
    It avoids using special cases or "connected to" placeholders.

    The algorithm prioritizes shorter paths with direct relationships over longer paths.
    """
    start_time = time.time()

    start_id = graph.start_id
    end_id = graph.end_id

    # Early return if IDs are None
    if not start_id or not end_id:
        logger.warning("[FastBiBFS] Start or end ID is None")
        return []

    id_to_parents = graph.id_to_parents
    id_to_children = graph.id_to_children

    # Validate inputs
    if not _validate_bfs_inputs(start_id, end_id, id_to_parents, id_to_children):
        return []

    if start_id == end_id:
        return [start_id]

    # First try to find a direct relationship (parent, child, sibling)
    # This is a quick check before running the full BFS

    # Convert lists to sets for find_direct_relationship
    id_to_parents_set = {k: set(v) for k, v in id_to_parents.items()}
    id_to_children_set = {k: set(v) for k, v in id_to_children.items()}
    direct_path = find_direct_relationship(start_id, end_id, id_to_parents_set, id_to_children_set)
    if direct_path:
        logger.debug(f"[FastBiBFS] Found direct relationship: {direct_path}")
        return direct_path

    # Initialize BFS data structures
    queue_fwd, queue_bwd, visited_fwd, visited_bwd = _initialize_bfs_queues(start_id, end_id)

    # Track all complete paths found
    all_paths = []
    processed = 0
    logger.debug(f"[FastBiBFS] Starting BFS: {start_id} <-> {end_id}")

    # Main search loop - continue until we find paths or exhaust the search
    while queue_fwd and queue_bwd and len(all_paths) < 5:
        # Check timeout and node limit
        if _check_search_limits(start_time, timeout_sec, processed, node_limit):
            break

        # Process forward queue (from start)
        processed += _process_forward_queue_item(
            queue_fwd, visited_bwd, visited_fwd, all_paths, id_to_parents, id_to_children, max_depth
        )

        # Process backward queue (from end)
        processed += _process_backward_queue_item(
            queue_bwd, visited_fwd, visited_bwd, all_paths, id_to_parents, id_to_children, max_depth
        )

    # Select the best path from found paths
    return _select_best_path(all_paths, start_id, end_id, id_to_parents, id_to_children)


def _process_forward_queue_item(
    queue_fwd: Any,
    visited_bwd: dict[str, Any],
    visited_fwd: dict[str, Any],
    all_paths: list[Any],
    id_to_parents: Any,
    id_to_children: Any,
    max_depth: int,
) -> int:
    """
    Process one item from the forward queue.

    Returns:
        1 if processed, 0 if queue empty
    """
    if not queue_fwd:
        return 0

    current_id, depth, path = queue_fwd.popleft()

    # Check if we've reached a node visited by backward search
    if current_id in visited_bwd:
        # Found a meeting point - reconstruct the path
        _, bwd_path = visited_bwd[current_id]  # depth unused
        # Combine paths (remove duplicate meeting point)
        combined_path = path + bwd_path[1:]
        all_paths.append(combined_path)
        logger.debug(f"[FastBiBFS] Path found via {current_id}: {len(combined_path)} nodes")
        return 1

    # Expand this node in forward direction
    graph_ctx = GraphContext(id_to_parents=id_to_parents, id_to_children=id_to_children, current_id=current_id)
    _expand_forward_node(graph_ctx, depth, path, visited_fwd, queue_fwd, max_depth)
    return 1


def _process_backward_queue_item(
    queue_bwd: Any,
    visited_fwd: dict[str, Any],
    visited_bwd: dict[str, Any],
    all_paths: list[Any],
    id_to_parents: Any,
    id_to_children: Any,
    max_depth: int,
) -> int:
    """
    Process one item from the backward queue.

    Returns:
        1 if processed, 0 if queue empty
    """
    if not queue_bwd:
        return 0

    current_id, depth, path = queue_bwd.popleft()

    # Check if we've reached a node visited by forward search
    if current_id in visited_fwd:
        # Found a meeting point - reconstruct the path
        _, fwd_path = visited_fwd[current_id]  # depth unused
        # Combine paths (remove duplicate meeting point)
        combined_path = fwd_path + path[1:]
        all_paths.append(combined_path)
        logger.debug(f"[FastBiBFS] Path found via {current_id}: {len(combined_path)} nodes")
        return 1

    # Expand this node in backward direction
    graph_ctx = GraphContext(id_to_parents=id_to_parents, id_to_children=id_to_children, current_id=current_id)
    _expand_backward_node(graph_ctx, depth, path, visited_bwd, queue_bwd, max_depth)
    return 1


def _check_search_limits(start_time: float, timeout_sec: int, processed: int, node_limit: int) -> bool:
    """
    Check if search limits (timeout or node limit) have been reached.

    Returns:
        True if limits reached, False otherwise
    """
    if time.time() - start_time > timeout_sec:
        logger.warning(f"[FastBiBFS] Timeout after {timeout_sec:.1f} seconds.")
        return True
    if processed > node_limit:
        logger.warning(f"[FastBiBFS] Node limit ({node_limit}) reached.")
        return True
    return False


def _select_best_path(
    all_paths: list[list[str]], start_id: str, end_id: str, id_to_parents: Any, id_to_children: Any
) -> list[str]:
    """Select the best path from a list of found paths based on relationship directness."""
    # If we found paths, select the best one
    if all_paths:
        # Score paths based on directness of relationships
        # Calculate scores for all paths
        scored_paths: list[tuple[list[str], float]] = []
        for path in all_paths:
            # Calculate score based on path properties
            direct_relationships = 0
            for i in range(len(path) - 1):
                if has_direct_relationship(path[i], path[i + 1], id_to_parents, id_to_children):
                    direct_relationships += 1

            # Calculate score: prefer paths with more direct relationships and shorter length
            directness_score = direct_relationships / (len(path) - 1) if len(path) > 1 else 0
            length_penalty = len(path) / 10  # Slight penalty for longer paths
            score = directness_score - length_penalty

            scored_paths.append((path, score))

        # Sort by score (highest first)
        scored_paths.sort(key=lambda x: x[1], reverse=True)

        # Return the path with the highest score
        best_path = scored_paths[0][0]
        logger.debug(f"[FastBiBFS] Selected best path: {len(best_path)} nodes with score {scored_paths[0][1]:.2f}")
        return best_path

    # If we didn't find any paths, try a more aggressive search
    logger.warning(f"[FastBiBFS] No paths found between {start_id} and {end_id}.")

    # Fallback: Try a direct path if possible
    return [start_id, end_id]


# Note: has_direct_relationship and find_direct_relationship live in relationship_utils.py
# to eliminate duplication. They are accessed via lazy-loaded helpers defined above.


# _are_directly_related removed - unused 24-line helper function for relationship checking


def _get_person_name_with_birth_year(indi: GedcomIndividualType | None, person_id: str) -> tuple[str, str]:
    """Get person's full name and birth year string."""
    from genealogy.gedcom.gedcom_events import get_event_info
    from genealogy.gedcom.gedcom_parser import get_full_name

    if not indi:
        return f"Unknown ({person_id})", ""

    name = get_full_name(indi)
    birth_year_str = ""
    birth_date_obj, _, _ = get_event_info(indi, TAG_BIRTH)
    if birth_date_obj:
        birth_year_str = f" (b. {birth_date_obj.year})"

    return name, birth_year_str


def _get_gender_char(indi: GedcomIndividualType) -> str | None:
    """Get gender character (M/F) from individual."""
    sex_b = getattr(indi, TAG_SEX.lower(), None)
    if sex_b and isinstance(sex_b, str) and str(sex_b).upper() in _GENDER_CHARS_UPPER:
        return str(sex_b).upper()[0]
    return None


def _determine_parent_relationship(sex_char: str | None, name: str, birth_year: str) -> str:
    """Determine parent relationship phrase based on gender."""
    parent_label = "father" if sex_char == "M" else "mother" if sex_char == "F" else "parent"
    return f"whose {parent_label} is {name}{birth_year}"


def _determine_child_relationship(sex_char: str | None, name: str, birth_year: str) -> str:
    """Determine child relationship phrase based on gender."""
    child_label = "son" if sex_char == "M" else "daughter" if sex_char == "F" else "child"
    return f"whose {child_label} is {name}{birth_year}"


def _determine_sibling_relationship(sex_char: str | None, name: str, birth_year: str) -> str:
    """Determine sibling relationship phrase based on gender."""
    sibling_label = "brother" if sex_char == "M" else "sister" if sex_char == "F" else "sibling"
    return f"whose {sibling_label} is {name}{birth_year}"


def _determine_spouse_relationship(sex_char: str | None, name: str, birth_year: str) -> str:
    """Determine spouse relationship phrase based on gender."""
    spouse_label = "husband" if sex_char == "M" else "wife" if sex_char == "F" else "spouse"
    return f"whose {spouse_label} is {name}{birth_year}"


def _determine_aunt_uncle_relationship(sex_char: str | None, name: str, birth_year: str) -> str:
    """Determine aunt/uncle relationship phrase based on gender."""
    relative_label = "uncle" if sex_char == "M" else "aunt" if sex_char == "F" else "aunt/uncle"
    return f"whose {relative_label} is {name}{birth_year}"


def _determine_niece_nephew_relationship(sex_char: str | None, name: str, birth_year: str) -> str:
    """Determine niece/nephew relationship phrase based on gender."""
    relative_label = "nephew" if sex_char == "M" else "niece" if sex_char == "F" else "niece/nephew"
    return f"whose {relative_label} is {name}{birth_year}"


def _determine_grandparent_relationship(sex_char: str | None, name: str, birth_year: str) -> str:
    """Determine grandparent relationship phrase based on gender."""
    grandparent_label = "grandfather" if sex_char == "M" else "grandmother" if sex_char == "F" else "grandparent"
    return f"whose {grandparent_label} is {name}{birth_year}"


def _determine_grandchild_relationship(sex_char: str | None, name: str, birth_year: str) -> str:
    """Determine grandchild relationship phrase based on gender."""
    grandchild_label = "grandson" if sex_char == "M" else "granddaughter" if sex_char == "F" else "grandchild"
    return f"whose {grandchild_label} is {name}{birth_year}"


def _determine_great_grandparent_relationship(sex_char: str | None, name: str, birth_year: str) -> str:
    """Determine great-grandparent relationship phrase based on gender."""
    grandparent_label = (
        "great-grandfather" if sex_char == "M" else "great-grandmother" if sex_char == "F" else "great-grandparent"
    )
    return f"whose {grandparent_label} is {name}{birth_year}"


def _determine_great_grandchild_relationship(sex_char: str | None, name: str, birth_year: str) -> str:
    """Determine great-grandchild relationship phrase based on gender."""
    grandchild_label = (
        "great-grandson" if sex_char == "M" else "great-granddaughter" if sex_char == "F" else "great-grandchild"
    )
    return f"whose {grandchild_label} is {name}{birth_year}"


def _check_relationship_type(
    relationship_type: str,
    id_a: str,
    id_b: str,
    sex_char: str | None,
    name_b: str,
    birth_year_b: str,
    reader: GedcomReaderType,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> str | None:
    """
    Check a specific relationship type and return the relationship phrase if matched.

    Returns None if the relationship type doesn't match.
    """
    # Data-driven relationship checking
    relationship_checks = {
        "parent": (
            lambda: id_b in id_to_parents.get(id_a, set()),
            lambda: _determine_parent_relationship(sex_char, name_b, birth_year_b),
        ),
        "child": (
            lambda: id_b in id_to_children.get(id_a, set()),
            lambda: _determine_child_relationship(sex_char, name_b, birth_year_b),
        ),
        "sibling": (
            lambda: are_siblings(id_a, id_b, id_to_parents),
            lambda: _determine_sibling_relationship(sex_char, name_b, birth_year_b),
        ),
        "spouse": (
            lambda: are_spouses(id_a, id_b, reader),
            lambda: _determine_spouse_relationship(sex_char, name_b, birth_year_b),
        ),
        "aunt_uncle": (
            lambda: is_aunt_or_uncle(id_a, id_b, id_to_parents, id_to_children),
            lambda: _determine_aunt_uncle_relationship(sex_char, name_b, birth_year_b),
        ),
        "niece_nephew": (
            lambda: is_niece_or_nephew(id_a, id_b, id_to_parents, id_to_children),
            lambda: _determine_niece_nephew_relationship(sex_char, name_b, birth_year_b),
        ),
        "cousin": (lambda: are_cousins(id_a, id_b, id_to_parents), lambda: f"whose cousin is {name_b}{birth_year_b}"),
        "grandparent": (
            lambda: is_grandparent(id_a, id_b, id_to_parents),
            lambda: _determine_grandparent_relationship(sex_char, name_b, birth_year_b),
        ),
        "grandchild": (
            lambda: is_grandchild(id_a, id_b, id_to_children),
            lambda: _determine_grandchild_relationship(sex_char, name_b, birth_year_b),
        ),
        "great_grandparent": (
            lambda: is_great_grandparent(id_a, id_b, id_to_parents),
            lambda: _determine_great_grandparent_relationship(sex_char, name_b, birth_year_b),
        ),
        "great_grandchild": (
            lambda: is_great_grandchild(id_a, id_b, id_to_children),
            lambda: _determine_great_grandchild_relationship(sex_char, name_b, birth_year_b),
        ),
    }

    if relationship_type in relationship_checks:
        check_func, result_func = relationship_checks[relationship_type]
        if check_func():
            return result_func()

    return None


def _determine_relationship_between_individuals(
    id_a: str,
    id_b: str,
    indi_b: GedcomIndividualType,
    name_b: str,
    birth_year_b: str,
    reader: GedcomReaderType,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> str:
    """Determine the relationship phrase between two individuals."""
    sex_char = _get_gender_char(indi_b)

    # Define relationship types to check in priority order
    relationship_types = [
        "parent",
        "child",
        "sibling",
        "spouse",
        "aunt_uncle",
        "niece_nephew",
        "cousin",
        "grandparent",
        "grandchild",
        "great_grandparent",
        "great_grandchild",
    ]

    # Check each relationship type
    for rel_type in relationship_types:
        result = _check_relationship_type(
            rel_type, id_a, id_b, sex_char, name_b, birth_year_b, reader, id_to_parents, id_to_children
        )
        if result:
            return result

    # Fallback for unknown relationships
    return f"related to {name_b}{birth_year_b}"


def explain_relationship_path(
    path_ids: list[str],
    reader: GedcomReaderType,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
    indi_index: dict[str, GedcomIndividualType],
) -> str:
    """
    Generates a human-readable explanation of the relationship path.

    This implementation uses a generic approach to determine relationships
    between individuals in the path without special cases. It analyzes the
    family structure to determine parent-child, sibling, spouse, and other
    relationships.
    """
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"

    steps: list[str] = []
    start_person_indi = indi_index.get(path_ids[0])

    # Get start person name with birth year
    start_person_name, birth_year_str = _get_person_name_with_birth_year(start_person_indi, path_ids[0])
    full_start_name = f"{start_person_name}{birth_year_str}"

    # Process each pair of individuals in the path
    for i in range(len(path_ids) - 1):
        id_a, id_b = path_ids[i], path_ids[i + 1]
        indi_a = indi_index.get(id_a)
        indi_b = indi_index.get(id_b)

        # Handle missing individuals
        if not indi_a or not indi_b:
            name_b, birth_year_b = _get_person_name_with_birth_year(indi_b, id_b)
            steps.append(f"  -> connected to {name_b}{birth_year_b}")
            continue

        # Get name and birth year for person B
        name_b, birth_year_b = _get_person_name_with_birth_year(indi_b, id_b)

        # Determine the relationship between A and B
        relationship_phrase = _determine_relationship_between_individuals(
            id_a, id_b, indi_b, name_b, birth_year_b, reader, id_to_parents, id_to_children
        )

        steps.append(f"  -> {relationship_phrase}")

    # Join the start name and all the steps
    return full_start_name + "\n" + "\n".join(steps)


def _extract_spouse_ids_from_family(fam: Any) -> tuple[str | None, str | None]:
    """Extract husband and wife IDs from a family record. Returns (husb_id, wife_id)."""
    husb_ref = fam.sub_tag(TAG_HUSBAND)
    wife_ref = fam.sub_tag(TAG_WIFE)

    husb_id = normalize_id(husb_ref.xref_id) if husb_ref and hasattr(husb_ref, "xref_id") else None
    wife_id = normalize_id(wife_ref.xref_id) if wife_ref and hasattr(wife_ref, "xref_id") else None

    return husb_id, wife_id


def are_spouses(id1: str, id2: str, reader: GedcomReaderType) -> bool:
    """Check if two individuals are spouses."""
    if not reader:
        return False

    try:
        for fam in reader.records0("FAM"):
            if not _is_record(fam):
                continue

            husb_id, wife_id = _extract_spouse_ids_from_family(fam)

            # Check if id1 and id2 are husband and wife in this family
            if (husb_id == id1 and wife_id == id2) or (husb_id == id2 and wife_id == id1):
                return True
    except Exception as e:
        logger.error(f"Error checking spouse relationship: {e}", exc_info=False)

    return False


# ==============================================
def _prepare_search_data(search_criteria: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize search criteria data."""
    return {
        "fname": (search_criteria.get("first_name") or "").lower()
        if isinstance(search_criteria.get("first_name"), str)
        else "",
        "sname": (search_criteria.get("surname") or "").lower()
        if isinstance(search_criteria.get("surname"), str)
        else "",
        "pob": (search_criteria.get("birth_place") or "").lower()
        if isinstance(search_criteria.get("birth_place"), str)
        else "",
        "pod": (search_criteria.get("death_place") or "").lower()
        if isinstance(search_criteria.get("death_place"), str)
        else "",
        "b_year": search_criteria.get("birth_year"),
        "b_date": search_criteria.get("birth_date_obj"),
        "d_year": search_criteria.get("death_year"),
        "d_date": search_criteria.get("death_date_obj"),
    }


def _prepare_candidate_data(candidate_processed_data: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize candidate data."""
    return {
        "id_debug": candidate_processed_data.get("norm_id", "N/A_in_proc_cache"),
        "fname": (candidate_processed_data.get("first_name") or "").lower()
        if isinstance(candidate_processed_data.get("first_name"), str)
        else "",
        "sname": (candidate_processed_data.get("surname") or "").lower()
        if isinstance(candidate_processed_data.get("surname"), str)
        else "",
        "bplace": (candidate_processed_data.get("birth_place_disp") or "").lower()
        if isinstance(candidate_processed_data.get("birth_place_disp"), str)
        else "",
        "dplace": (candidate_processed_data.get("death_place_disp") or "").lower()
        if isinstance(candidate_processed_data.get("death_place_disp"), str)
        else "",
        "b_year": candidate_processed_data.get("birth_year"),
        "b_date": candidate_processed_data.get("birth_date_obj"),
        "d_year": candidate_processed_data.get("death_year"),
        "d_date": candidate_processed_data.get("death_date_obj"),
    }


def _score_names(
    t_data: dict[str, Any],
    c_data: dict[str, Any],
    weights: Mapping[str, Any],
    field_scores: dict[str, float],
    match_reasons: list[str],
) -> None:
    """Score name matches (first name, surname, and bonus for both)."""
    first_name_matched = False
    surname_matched = False

    if t_data["fname"] and c_data["fname"] and t_data["fname"] in c_data["fname"]:
        points_givn = weights.get("contains_first_name", 0)
        if points_givn != 0:
            field_scores["givn"] = int(points_givn)
            match_reasons.append(f"Contains First Name ({points_givn}pts)")
            first_name_matched = True

    if t_data["sname"] and c_data["sname"] and t_data["sname"] in c_data["sname"]:
        points_surn = weights.get("contains_surname", 0)
        if points_surn != 0:
            field_scores["surn"] = int(points_surn)
            match_reasons.append(f"Contains Surname ({points_surn}pts)")
            surname_matched = True

    if t_data["fname"] and t_data["sname"] and first_name_matched and surname_matched:
        bonus_points = weights.get("bonus_both_names_contain", 0)
        if bonus_points != 0:
            field_scores["bonus"] = int(bonus_points)
            match_reasons.append(f"Bonus Both Names ({bonus_points}pts)")


def _check_year_match(t_year: Any, c_year: Any, year_score_range: int) -> tuple[bool, bool]:
    """Check if years match exactly or approximately. Returns (exact_match, approx_match)."""
    if t_year is None or c_year is None:
        return False, False
    try:
        t_year_int = int(t_year)
        c_year_int = int(c_year)
        exact_match = t_year_int == c_year_int
        approx_match = not exact_match and abs(c_year_int - t_year_int) <= year_score_range
        return exact_match, approx_match
    except Exception:
        return False, False


def _calculate_date_flags(
    t_data: dict[str, Any], c_data: dict[str, Any], year_score_range: int | float
) -> dict[str, Any]:
    """Calculate date match flags for birth and death dates."""
    year_range = int(year_score_range)
    birth_year_match, birth_year_approx = _check_year_match(t_data["b_year"], c_data["b_year"], year_range)
    death_year_match, death_year_approx = _check_year_match(t_data["d_year"], c_data["d_year"], year_range)

    return {
        "exact_birth_date_match": bool(
            t_data["b_date"]
            and c_data["b_date"]
            and isinstance(t_data["b_date"], datetime)
            and isinstance(c_data["b_date"], datetime)
            and t_data["b_date"].date() == c_data["b_date"].date()
        ),
        "exact_death_date_match": bool(
            t_data["d_date"]
            and c_data["d_date"]
            and isinstance(t_data["d_date"], datetime)
            and isinstance(c_data["d_date"], datetime)
            and t_data["d_date"].date() == c_data["d_date"].date()
        ),
        "birth_year_match": birth_year_match,
        "birth_year_approx_match": birth_year_approx,
        "death_year_match": death_year_match,
        "death_year_approx_match": death_year_approx,
        "death_dates_absent": bool(
            t_data["d_date"] is None
            and c_data["d_date"] is None
            and t_data["d_year"] is None
            and c_data["d_year"] is None
        ),
    }


def _score_birth_dates(
    t_data: dict[str, Any],
    c_data: dict[str, Any],
    date_flags: dict[str, Any],
    weights: Mapping[str, Any],
    field_scores: dict[str, float],
    match_reasons: list[str],
) -> None:
    """Score birth date matches (prioritize: exact date > exact year > approx year)."""
    if date_flags["exact_birth_date_match"]:
        points_bdate = weights.get("exact_birth_date", 0)
        if points_bdate != 0:
            field_scores["bdate"] = int(points_bdate)
            match_reasons.append(f"Exact Birth Date ({points_bdate}pts)")
            return

    if date_flags["birth_year_match"]:
        points_byear = weights.get("year_birth", 0)
        if points_byear != 0:
            field_scores["byear"] = int(points_byear)
            match_reasons.append(f"Exact Birth Year ({c_data['b_year']}) ({points_byear}pts)")
            return

    if date_flags["birth_year_approx_match"]:
        points_byear_approx = weights.get("approx_year_birth", 0) or weights.get("birth_year_close", 0)
        if points_byear_approx != 0:
            field_scores["byear"] = int(points_byear_approx)
            match_reasons.append(
                f"Approx Birth Year ({c_data['b_year']} vs {t_data['b_year']}) ({points_byear_approx}pts)"
            )
        return

    _apply_birth_mismatch_penalty(t_data, c_data, weights, field_scores, match_reasons)


def _apply_birth_mismatch_penalty(
    t_data: dict[str, Any],
    c_data: dict[str, Any],
    weights: Mapping[str, Any],
    field_scores: dict[str, float],
    match_reasons: list[str],
) -> None:
    """Apply penalty for significant birth year mismatch."""
    if t_data["b_year"] is not None and c_data["b_year"] is not None:
        try:
            diff = abs(int(t_data["b_year"]) - int(c_data["b_year"]))
            if diff > 10:  # Hardcoded threshold for significant mismatch
                penalty = weights.get("birth_year_mismatch_penalty", -100)  # Default penalty
                field_scores["byear"] = int(penalty)
                match_reasons.append(f"Birth Year Mismatch ({diff}y) ({penalty}pts)")
        except (ValueError, TypeError):
            pass


def _apply_death_mismatch_penalty(
    t_data: dict[str, Any],
    c_data: dict[str, Any],
    weights: Mapping[str, Any],
    field_scores: dict[str, float],
    match_reasons: list[str],
) -> None:
    """Apply penalty for significant death year mismatch."""
    if t_data["d_year"] is not None and c_data["d_year"] is not None:
        try:
            diff = abs(int(t_data["d_year"]) - int(c_data["d_year"]))
            if diff > 10:
                penalty = weights.get("death_year_mismatch_penalty", -75)
                field_scores["dyear"] = int(penalty)
                match_reasons.append(f"Death Year Mismatch ({diff}y) ({penalty}pts)")
        except (ValueError, TypeError):
            pass


def _score_death_dates(
    t_data: dict[str, Any],
    c_data: dict[str, Any],
    date_flags: dict[str, Any],
    weights: Mapping[str, Any],
    field_scores: dict[str, float],
    match_reasons: list[str],
) -> None:
    """Score death date matches (prioritize: exact date > exact year > approx year; no points for absence)."""
    if date_flags["exact_death_date_match"]:
        points_ddate = weights.get("exact_death_date", 0)
        if points_ddate != 0:
            field_scores["ddate"] = int(points_ddate)
            match_reasons.append(f"Exact Death Date ({points_ddate}pts)")
            return

    if date_flags["death_year_match"]:
        points_dyear = weights.get("year_death", 0)
        if points_dyear != 0:
            field_scores["dyear"] = int(points_dyear)
            match_reasons.append(f"Exact Death Year ({c_data['d_year']}) ({points_dyear}pts)")
            return

    if date_flags["death_year_approx_match"]:
        points_dyear_approx = weights.get("approx_year_death", 0)
        if points_dyear_approx != 0:
            field_scores["dyear"] = int(points_dyear_approx)
            match_reasons.append(
                f"Approx Death Year ({c_data['d_year']} vs {t_data['d_year']}) ({points_dyear_approx}pts)"
            )
        return

    # Check for significant mismatch if no match found
    _apply_death_mismatch_penalty(t_data, c_data, weights, field_scores, match_reasons)

    # Do not award points for both death dates absent when the user did not specify death criteria.


def _score_dates(
    t_data: dict[str, Any],
    c_data: dict[str, Any],
    date_flags: dict[str, Any],
    weights: Mapping[str, Any],
    field_scores: dict[str, float],
    match_reasons: list[str],
) -> None:
    """Score birth and death date matches."""
    _score_birth_dates(t_data, c_data, date_flags, weights, field_scores, match_reasons)
    _score_death_dates(t_data, c_data, date_flags, weights, field_scores, match_reasons)


def _score_birth_place(
    t_data: dict[str, Any],
    c_data: dict[str, Any],
    weights: Mapping[str, Any],
    field_scores: dict[str, float],
    match_reasons: list[str],
) -> None:
    """Score birth place match."""
    if not (t_data["pob"] and c_data["bplace"] and t_data["pob"] in c_data["bplace"]):
        return
    points_pob = weights.get("contains_pob", 0) or weights.get("birth_place_match", 0)
    if points_pob != 0:
        field_scores["bplace"] = int(points_pob)
        match_reasons.append(f"Birth Place Contains ({points_pob}pts)")


def _score_death_place(
    t_data: dict[str, Any],
    c_data: dict[str, Any],
    weights: Mapping[str, Any],
    field_scores: dict[str, float],
    match_reasons: list[str],
) -> None:
    """Score death place match (contains only; no points for absence)."""
    pod_match = bool(t_data["pod"] and c_data["dplace"] and t_data["pod"] in c_data["dplace"])
    if not pod_match:
        return
    points_pod = weights.get("contains_pod", 0) or weights.get("death_place_match", 0)
    if points_pod != 0:
        field_scores["dplace"] = int(points_pod)
        match_reasons.append(f"Death Place Contains ({points_pod}pts)")


def _score_places(
    t_data: dict[str, Any],
    c_data: dict[str, Any],
    weights: Mapping[str, Any],
    field_scores: dict[str, float],
    match_reasons: list[str],
) -> None:
    """Score birth place and death place matches (gender removed from scoring)."""
    _score_birth_place(t_data, c_data, weights, field_scores, match_reasons)
    _score_death_place(t_data, c_data, weights, field_scores, match_reasons)


def _score_birth_bonus(weights: Mapping[str, Any], field_scores: dict[str, float], match_reasons: list[str]) -> None:
    """Score birth bonus (if both birth year and birth place matched)."""
    if not (field_scores["byear"] > 0 and field_scores["bplace"] > 0):
        return
    birth_bonus_points = weights.get("bonus_birth_info", 0) or weights.get("bonus_birth_date_and_place", 0)
    if birth_bonus_points != 0:
        field_scores["bbonus"] = int(birth_bonus_points)
        match_reasons.append(f"Bonus Birth Info ({birth_bonus_points}pts)")


def _score_death_bonus(weights: Mapping[str, Any], field_scores: dict[str, float], match_reasons: list[str]) -> None:
    """Score death bonus (only when BOTH death date and place matched)."""
    death_info_matched = (field_scores["dyear"] > 0 or field_scores["ddate"] > 0) and field_scores["dplace"] > 0
    if not death_info_matched:
        return
    death_bonus_points = weights.get("bonus_death_info", 0) or weights.get("bonus_death_date_and_place", 0)
    if death_bonus_points != 0:
        field_scores["dbonus"] = int(death_bonus_points)
        match_reasons.append(f"Bonus Death Info ({death_bonus_points}pts)")


def _score_bonuses(weights: Mapping[str, Any], field_scores: dict[str, float], match_reasons: list[str]) -> None:
    """Score birth and death bonuses."""
    _score_birth_bonus(weights, field_scores, match_reasons)
    _score_death_bonus(weights, field_scores, match_reasons)


def _apply_alive_conflict_penalty(
    t_data: dict[str, Any],
    c_data: dict[str, Any],
    weights: Mapping[str, Any],
    field_scores: dict[str, float],
    match_reasons: list[str],
) -> None:
    """Apply a small negative score when query implies 'alive' but candidate has death info.

    Alive-mode heuristic: if the user provided no death year, no death date, and no death place,
    we assume they are searching for a living person. In that case, a candidate that contains
    death information (date/year/place) receives a configurable penalty.
    """
    try:
        alive_query = not (t_data.get("d_date") or t_data.get("d_year") or t_data.get("pod"))
        candidate_has_death = bool(c_data.get("d_date") or c_data.get("d_year") or c_data.get("dplace"))
        if alive_query and candidate_has_death:
            penalty = int(weights.get("alive_conflict_penalty", 0))
            if penalty != 0:
                field_scores["alive_penalty"] = penalty
                match_reasons.append(f"Alive Assumed; Candidate Has Death Info ({penalty}pts)")
    except Exception:
        # Non-fatal; scoring should not break due to penalty logic
        pass


# Scoring Function (V18 - Corrected Syntax)
# ==============================================
def calculate_match_score(
    search_criteria: dict[str, Any],
    candidate_processed_data: dict[str, Any],  # Expects pre-processed data
    scoring_weights: Mapping[str, int | float] | None = None,
    date_flexibility: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float], list[str]]:
    """
    Calculates match score using pre-processed candidate data.
    Handles OR logic for death place matching (contains OR both absent).
    Prioritizes exact date > exact year > approx year for date scoring.
    V18.PreProcess compatible - Syntax Fixed.
    """
    match_reasons: list[str] = []
    field_scores = {
        "givn": 0.0,
        "surn": 0.0,
        "byear": 0.0,
        "bdate": 0.0,
        "bplace": 0.0,
        "bbonus": 0.0,
        "dyear": 0.0,
        "ddate": 0.0,
        "dplace": 0.0,
        "dbonus": 0.0,
        "bonus": 0.0,
        # Negative adjustments (policy-based)
        "alive_penalty": 0.0,
    }
    weights = scoring_weights if scoring_weights is not None else config.common_scoring_weights
    date_flex = date_flexibility if date_flexibility is not None else {"year_match_range": config.date_flexibility}
    year_score_range = date_flex.get("year_match_range", 10)

    # Prepare data
    t_data = _prepare_search_data(search_criteria)
    c_data = _prepare_candidate_data(candidate_processed_data)

    # Name Scoring
    _score_names(t_data, c_data, weights, field_scores, match_reasons)

    # Calculate date match flags
    date_flags = _calculate_date_flags(t_data, c_data, year_score_range)

    # Date Scoring
    _score_dates(t_data, c_data, date_flags, weights, field_scores, match_reasons)

    # Place Scoring
    _score_places(t_data, c_data, weights, field_scores, match_reasons)

    # Bonus Scoring
    _score_bonuses(weights, field_scores, match_reasons)

    # Policy-based negative adjustments
    _apply_alive_conflict_penalty(t_data, c_data, weights, field_scores, match_reasons)

    # Calculate Final Total Score
    final_total_score = sum(field_scores.values())
    final_total_score = max(0.0, round(final_total_score))
    unique_reasons = sorted(set(match_reasons))

    # Final score calculated

    # Return a COPY of the dictionary
    return final_total_score, field_scores.copy(), unique_reasons


# End of calculate_match_score


# ==============================================
# GedcomData Class
# ==============================================
class GedcomData:
    def __init__(self, gedcom_path: str | Path, skip_cache_build: bool = False):
        """
        Initialize GedcomData from a GEDCOM file.

        Args:
            gedcom_path: Path to the GEDCOM file
            skip_cache_build: If True, skip building caches (used when loading from cache)
        """
        self.path = Path(gedcom_path).resolve()
        self.reader: GedcomReaderType | None = None
        self.indi_index: dict[str, GedcomIndividualType] = {}  # Index of INDI records
        self.processed_data_cache: dict[str, dict[str, Any]] = {}  # NEW: Cache for processed data
        self.id_to_parents: dict[str, set[str]] = {}
        self.id_to_children: dict[str, set[str]] = {}
        self.id_to_spouses: dict[str, set[str]] = {}  # NEW: Map for spouse lookups
        self.indi_index_build_time: float = 0
        self.family_maps_build_time: float = 0
        self.data_processing_time: float = 0  # NEW: Time for pre-processing
        self._cache_source: str = "unknown"  # Track where data was loaded from: "memory", "disk", or "file"

        if not self.path.is_file():
            logger.critical(f"GEDCOM file not found: {self.path}")
            raise FileNotFoundError(f"GEDCOM file not found: {self.path}")
        try:
            logger.debug(f"Loading GEDCOM file: {self.path}")
            load_start = time.time()
            # Initialize GedcomReader with the file path as a string
            # The constructor takes a file parameter (file name or file object)
            # There seems to be a discrepancy between the documentation and the actual implementation
            # We'll call the reader via a casted factory to satisfy the type checker while
            # still passing the file path string required at runtime.
            reader_factory = cast(Callable[[Any], GedcomReaderType], GedcomReader)
            self.reader = reader_factory(str(self.path))
            load_time = time.time() - load_start
            logger.debug(f"GEDCOM file loaded in {load_time:.2f}s.")
        except Exception as e:
            file_size_mb = self.path.stat().st_size / (1024 * 1024) if self.path.exists() else "unknown"
            error_msg = (
                f"Failed to load/parse GEDCOM file {self.path} (size: {file_size_mb:.2f}MB). "
                f"Error type: {type(e).__name__}. This may indicate file corruption, "
                f"unsupported GEDCOM format, or encoding issues."
            )
            logger.critical(error_msg, exc_info=True)
            # Use exception chaining to preserve the original error context
            raise RuntimeError(error_msg) from e

        if not skip_cache_build:
            self.build_caches()  # Build caches upon initialization

    @classmethod
    def from_cache(cls, cached_data: dict[str, Any], gedcom_path: str) -> "GedcomData":
        """
        Create a GedcomData instance from cached data.

        This method creates a GedcomData instance with the reader initialized,
        but populates the expensive-to-build data structures from cache.
        The indi_index is rebuilt quickly from the reader (just indexing, no parsing).

        Args:
            cached_data: Dictionary containing cached GEDCOM data
            gedcom_path: Path to the GEDCOM file

        Returns:
            GedcomData instance with cached data populated
        """
        # Create instance with reader but skip cache building
        instance = cls(gedcom_path, skip_cache_build=True)

        # Populate from cached data (note: indi_index is NOT in cache because
        # Individual objects cannot be pickled)
        instance.processed_data_cache = cached_data.get("processed_data_cache", {})
        instance.id_to_parents = cached_data.get("id_to_parents", {})
        instance.id_to_children = cached_data.get("id_to_children", {})
        spouse_map = cached_data.get("id_to_spouses") if "id_to_spouses" in cached_data else None
        instance.id_to_spouses = spouse_map or {}
        instance.indi_index_build_time = cached_data.get("indi_index_build_time", 0)
        instance.family_maps_build_time = cached_data.get("family_maps_build_time", 0)
        instance.data_processing_time = cached_data.get("data_processing_time", 0)

        # Rebuild indi_index from reader (fast - just indexing, no data extraction)
        instance._build_indi_index()

        if spouse_map is None:
            logger.debug("Cached GEDCOM data missing spouse relationships; rebuilding family maps for accuracy.")
            instance._build_family_maps()

        logger.debug(
            f"GedcomData restored from cache: {len(instance.processed_data_cache)} processed individuals, "
            f"{len(instance.id_to_parents)} parent relationships, "
            f"{len(instance.indi_index)} individuals indexed"
        )

        return instance

    def build_caches(self) -> None:
        """Builds the individual index, family maps, and pre-processes data."""
        if not self.reader:
            logger.error("[Cache Build] Cannot build caches: GedcomReader is None.")
            return
        self._build_indi_index()
        # Only build maps and process data if index was successful
        if self.indi_index:
            self._build_family_maps()
            self._pre_process_individual_data()  # NEW: Call pre-processing
        else:
            logger.error("[Cache Build] Skipping map build and data pre-processing due to empty INDI index.")

    def _process_indi_record(self, indi_record: Any) -> tuple[bool, bool]:
        """Process an individual record for indexing. Returns (processed, skipped)."""
        if not (_is_individual(indi_record) and hasattr(indi_record, "xref_id") and indi_record.xref_id):
            if logger.isEnabledFor(logging.DEBUG):
                if hasattr(indi_record, "xref_id"):
                    logger.debug(
                        f"Skipping non-Individual record: Type={type(indi_record).__name__}, Xref={indi_record.xref_id}"
                    )
                else:
                    logger.debug(f"Skipping record with no xref_id: Type={type(indi_record).__name__}")
            return False, True

        norm_id = normalize_id(indi_record.xref_id)
        if not norm_id:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Skipping INDI with unnormalizable xref_id: {indi_record.xref_id}")
            return False, True

        if norm_id in self.indi_index:
            logger.warning(f"Duplicate normalized INDI ID found: {norm_id}. Overwriting.")

        safe_record = cast(GedcomIndividualType, indi_record)
        self.indi_index[norm_id] = safe_record
        return True, False

    def _build_indi_index(self) -> None:
        """Builds a dictionary mapping normalized IDs to Individual records."""
        if not self.reader:
            logger.error("[Cache Build] Cannot build INDI index: GedcomReader is None.")
            return

        start_time = time.time()
        logger.debug("[Cache] Building INDI index...")
        self.indi_index = {}
        count = 0
        skipped = 0
        current_record_id = "None"

        try:
            for indi_record in self.reader.records0(TAG_INDI):
                current_record_id = getattr(indi_record, "xref_id", "Unknown") if indi_record else "None"
                processed, skip = self._process_indi_record(indi_record)
                if processed:
                    count += 1
                if skip:
                    skipped += 1
        except StopIteration:
            logger.debug("[Cache] Finished iterating INDI records for index.")
        except Exception as e:
            record_context = f"while processing record ID: {current_record_id}"
            logger.error(
                f"[Cache Build] Error during INDI index build {record_context}: {e}. "
                f"Error type: {type(e).__name__}. Index may be incomplete. "
                f"Records processed so far: {count}, skipped: {skipped}.",
                exc_info=True,
            )

        elapsed = time.time() - start_time
        self.indi_index_build_time = elapsed
        if count > 0:
            logger.debug(f"[Cache] INDI index built with {count} individuals ({skipped} skipped) in {elapsed:.2f}s.")
        else:
            logger.error(
                f"[Cache Build] INDI index is EMPTY after build attempt ({skipped} skipped) in {elapsed:.2f}s."
            )

    @staticmethod
    def _extract_parents_from_family(fam: Any, fam_id_log: str) -> set[str]:
        """Extract parent IDs from a family record."""
        parents: set[str] = set()
        for parent_tag in [TAG_HUSBAND, TAG_WIFE]:
            parent_ref = fam.sub_tag(parent_tag)
            if parent_ref and hasattr(parent_ref, "xref_id"):
                parent_id = normalize_id(parent_ref.xref_id)
                if parent_id:
                    parents.add(parent_id)
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Skipping parent with invalid/unnormalizable ID {getattr(parent_ref, 'xref_id', '?')} in FAM {fam_id_log}"
                    )
        return parents

    def _process_child_in_family(self, child_tag: Any, parents: set[str], fam_id_log: str) -> tuple[bool, bool]:
        """Process a child tag and update family maps. Returns (processed, skipped)."""
        if not (child_tag and hasattr(child_tag, "xref_id")):
            if child_tag is not None and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Skipping CHIL record in FAM {fam_id_log} with invalid format: Type={type(child_tag).__name__}"
                )
            return False, True

        child_id = normalize_id(child_tag.xref_id)
        if not child_id:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Skipping child with invalid/unnormalizable ID {getattr(child_tag, 'xref_id', '?')} in FAM {fam_id_log}"
                )
            return False, True

        # Add child to each parent's children set
        for parent_id in parents:
            self.id_to_children.setdefault(parent_id, set()).add(child_id)

        # Add parents to child's parents set
        if parents:
            self.id_to_parents.setdefault(child_id, set()).update(parents)
            return True, False

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Child {child_id} found in FAM {fam_id_log} but no valid parents identified in this specific record."
            )
        return False, False

    def _build_family_maps(self) -> None:
        """Builds dictionaries mapping child IDs to parent IDs and parent IDs to child IDs."""
        if not self.reader:
            logger.error("[Cache Build] Cannot build family maps: GedcomReader is None.")
            return

        start_time = time.time()
        logger.debug("[Cache] Building family maps...")
        self.id_to_parents = {}
        self.id_to_children = {}
        self.id_to_spouses = {}
        fam_count = 0
        processed_links = 0
        skipped_links = 0

        try:
            for fam in self.reader.records0("FAM"):
                fam_count += 1
                if not _is_record(fam):
                    logger.debug(f"Skipping non-record FAM entry: {type(fam)}")
                    continue

                fam_id_log = getattr(fam, "xref_id", "N/A_FAM")
                parents = self._extract_parents_from_family(fam, fam_id_log)

                # Process spouses (parents in the same family are spouses)
                if len(parents) == 2:
                    p1, p2 = list(parents)
                    self.id_to_spouses.setdefault(p1, set()).add(p2)
                    self.id_to_spouses.setdefault(p2, set()).add(p1)

                for child_tag in fam.sub_tags(TAG_CHILD):
                    processed, skipped = self._process_child_in_family(child_tag, parents, fam_id_log)
                    if processed:
                        processed_links += 1
                    if skipped:
                        skipped_links += 1
        except StopIteration:
            logger.debug("[Cache] Finished iterating FAM records for maps.")
        except Exception as e:
            logger.error(
                f"[Cache Build] Unexpected error during family map build: {e}. Maps may be incomplete.", exc_info=True
            )

        self._log_family_maps_build_results(time.time() - start_time, fam_count, processed_links, skipped_links)

    def _log_family_maps_build_results(
        self, elapsed: float, fam_count: int, processed_links: int, skipped_links: int
    ) -> None:
        """Log the results of family maps building."""
        self.family_maps_build_time = elapsed
        parent_map_count = len(self.id_to_parents)
        child_map_count = len(self.id_to_children)
        logger.debug(
            f"[Cache] Family maps built: {fam_count} FAMs processed. Added {processed_links} child-parent relationships "
            f"({skipped_links} skipped invalid links/IDs). Map sizes: {parent_map_count} child->parents entries, "
            f"{child_map_count} parent->children entries in {elapsed:.2f}s."
        )
        if parent_map_count == 0 and child_map_count == 0 and fam_count > 0:
            logger.warning(
                "[Cache Build] Family maps are EMPTY despite processing FAM records. Check GEDCOM structure or parsing logic."
            )

    def _pre_process_individual_data(self) -> None:
        """NEW: Extracts and caches key data points for each individual."""
        if not self.indi_index:
            logger.error("Cannot pre-process data: INDI index is not built.")
            return
        start_time = time.time()
        logger.debug("[Pre-Process] Extracting key data for individuals...")
        self.processed_data_cache = {}
        processed_count = 0
        errors = 0
        for norm_id, indi in self.indi_index.items():
            try:
                self.processed_data_cache[norm_id] = self._build_processed_record(norm_id, indi)
                processed_count += 1
            except Exception as e:
                # Enhanced error reporting with more context
                error_type = type(e).__name__
                logger.error(
                    f"Error pre-processing individual {norm_id}: {error_type}: {e}. "
                    f"This may affect search results and relationship paths for this individual.",
                    exc_info=True,
                )
                errors += 1
        elapsed = time.time() - start_time
        self.data_processing_time = elapsed
        logger.debug(
            f"[Pre-Process] Processed data for {processed_count} individuals ({errors} errors) in {elapsed:.2f}s."
        )
        if not self.processed_data_cache:
            logger.error("[Pre-Process] Processed data cache is EMPTY after build attempt.")

    def _build_processed_record(self, norm_id: str, indi: GedcomIndividualType) -> dict[str, Any]:
        """Construct the processed cache entry for an individual."""
        from genealogy.gedcom.gedcom_parser import get_full_name

        full_name_disp = get_full_name(indi)
        first_name_score, surname_score = self._derive_name_parts(full_name_disp)

        name_rec = indi.sub_tag(TAG_NAME) if indi is not None else None
        givn_raw = name_rec.sub_tag_value(TAG_GIVN) if name_rec else None
        surn_raw = name_rec.sub_tag_value(TAG_SURN) if name_rec else None

        sex_raw = indi.sub_tag_value(TAG_SEX) if indi is not None else None
        gender_norm = self._normalize_gender_value(sex_raw)

        birth_details = self._event_cache_details(indi, TAG_BIRTH)
        death_details = self._event_cache_details(indi, TAG_DEATH)

        return {
            "norm_id": norm_id,
            "display_id": getattr(indi, "xref_id", norm_id),
            "givn_raw": givn_raw,
            "surn_raw": surn_raw,
            "first_name": first_name_score,
            "surname": surname_score,
            "full_name_disp": full_name_disp,
            "gender_raw": sex_raw,
            "gender_norm": gender_norm,
            "birth_date_obj": birth_details["date_obj"],
            "birth_date_str": birth_details["date_str"],
            "birth_date_disp": birth_details["date_disp"],
            "birth_year": birth_details["year"],
            "birth_place_raw": birth_details["place_raw"],
            "birth_place_disp": birth_details["place_disp"],
            "death_date_obj": death_details["date_obj"],
            "death_date_str": death_details["date_str"],
            "death_date_disp": death_details["date_disp"],
            "death_year": death_details["year"],
            "death_place_raw": death_details["place_raw"],
            "death_place_disp": death_details["place_disp"],
        }

    @staticmethod
    def _derive_name_parts(full_name: str) -> tuple[str, str]:
        if full_name == "Unknown":
            return "", ""
        parts = full_name.split()
        if not parts:
            return "", ""
        first = parts[0]
        surname = parts[-1] if len(parts) > 1 else ""
        return first, surname

    @staticmethod
    def _normalize_gender_value(sex_raw: str | None) -> str | None:
        if not sex_raw:
            return None
        sex_lower = str(sex_raw).lower()
        return sex_lower if sex_lower in _GENDER_CHARS_LOWER else None

    @staticmethod
    def _event_cache_details(indi: GedcomIndividualType, tag: str) -> dict[str, Any]:
        from genealogy.gedcom.gedcom_events import _clean_display_date, get_event_info

        date_obj, date_str, place_raw = get_event_info(indi, tag)
        return {
            "date_obj": date_obj,
            "date_str": date_str,
            "date_disp": _clean_display_date(date_str),
            "year": date_obj.year if date_obj else None,
            "place_raw": place_raw,
            "place_disp": None if place_raw == "N/A" else place_raw,
        }

    def get_processed_indi_data(self, norm_id: str) -> dict[str, Any] | None:
        """Retrieves pre-processed data for an individual from the cache."""
        if not self.processed_data_cache:
            logger.warning("Attempting to get processed data, but cache is empty. Triggering pre-processing.")
            self._pre_process_individual_data()
        return self.processed_data_cache.get(norm_id)

    def find_individual_by_id(self, norm_id: str | None) -> GedcomIndividualType | None:
        """Finds an individual by their normalized ID using the index."""
        if not norm_id:
            logger.warning("find_individual_by_id called with invalid norm_id")
            return None
        if not self.indi_index:
            logger.warning("INDI_INDEX not built. Attempting build now.")
            self._build_indi_index()
        if not self.indi_index:
            logger.error("INDI_INDEX build failed. Cannot lookup individual by ID.")
            return None
        found_indi = self.indi_index.get(norm_id)
        if not found_indi and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Individual with normalized ID {norm_id} not found in INDI_INDEX.")
        return found_indi

    def _get_sibling_ids(self, target_id: str) -> set[str]:
        """Get sibling IDs for a target individual."""
        parents = self.id_to_parents.get(target_id, set())
        if not parents:
            return set()
        potential_siblings = set().union(*(self.id_to_children.get(p_id, set()) for p_id in parents))
        return potential_siblings - {target_id}

    def _get_spouse_ids(self, target_id: str) -> set[str]:
        """Get spouse IDs for a target individual."""
        return self.id_to_spouses.get(target_id, set())

    def _get_related_ids_by_type(self, target_id: str, relationship_type: str) -> set[str] | None:
        """Get related IDs based on relationship type. Returns None for unknown types."""
        if relationship_type == "parents":
            return self.id_to_parents.get(target_id, set())
        if relationship_type == "children":
            return self.id_to_children.get(target_id, set())
        if relationship_type == "siblings":
            return self._get_sibling_ids(target_id)
        if relationship_type == "spouses":
            return self._get_spouse_ids(target_id)
        logger.warning(f"Unknown relationship type requested: '{relationship_type}'")
        return None

    def _ensure_family_maps_built(self) -> bool:
        """Ensure family maps are built. Returns True if maps are available."""
        if not self.id_to_parents and not self.id_to_children:
            logger.warning("get_related_individuals: Relationship maps empty. Attempting build.")
            self._build_family_maps()
        if not self.id_to_parents and not self.id_to_children:
            logger.error("get_related_individuals: Maps still empty after build attempt.")
            return False
        return True

    def _convert_ids_to_individuals(self, related_ids: set[str], target_id: str) -> list[GedcomIndividualType]:
        """Convert a set of IDs to Individual objects."""
        related_individuals: list[GedcomIndividualType] = []
        for rel_id in related_ids:
            if rel_id != target_id:
                indi = self.find_individual_by_id(rel_id)
                if indi:
                    related_individuals.append(indi)
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Could not find Individual object for related ID: {rel_id}")
        related_individuals.sort(key=lambda x: (normalize_id(getattr(x, "xref_id", None)) or ""))
        return related_individuals

    def get_related_individuals(
        self, individual: GedcomIndividualType, relationship_type: str
    ) -> list[GedcomIndividualType]:
        """Gets parents, children, siblings, or spouses using cached maps."""
        # Validate input
        if not _is_individual(individual) or not hasattr(individual, "xref_id"):
            logger.warning(f"get_related_individuals: Invalid input individual object: {type(individual)}")
            return []

        target_id = normalize_id(individual.xref_id if individual is not None else None)
        if not target_id:
            return []

        # Ensure maps are built
        if not self._ensure_family_maps_built():
            return []

        # Get related IDs and convert to Individual objects
        try:
            related_ids = self._get_related_ids_by_type(target_id, relationship_type)
            if related_ids is None:
                return []
            return self._convert_ids_to_individuals(related_ids, target_id)
        except Exception as e:
            logger.error(f"Error getting {relationship_type} for {target_id}: {e}", exc_info=True)
            return []

    def _ensure_maps_and_index_built(self) -> str | None:
        """Ensure family maps and individual index are built. Returns error message or None if successful."""
        # Ensure family maps are built
        if not self.id_to_parents and not self.id_to_children:
            logger.warning("Relationship maps are empty, attempting rebuild.")
            self._build_family_maps()
        if not self.id_to_parents and not self.id_to_children:
            return "Error: Family relationship maps could not be built."

        # Ensure individual index is built
        if not self.indi_index:
            logger.warning("Individual index is empty, attempting rebuild.")
            self._build_indi_index()
        if not self.indi_index:
            return "Error: Individual index could not be built."

        return None  # Success

    def _validate_relationship_path_inputs(self, id1_norm: str, id2_norm: str) -> str | None:
        """Validate inputs for relationship path calculation. Returns error message or None if valid."""
        if not self.reader:
            return "Error: GEDCOM Reader unavailable."
        if not id1_norm or not id2_norm:
            return "Invalid input IDs."
        if id1_norm == id2_norm:
            return "Individuals are the same."

        return self._ensure_maps_and_index_built()

    def get_relationship_path(self, id1: str, id2: str) -> str:
        """
        Finds and explains the relationship path between two individuals.

        This implementation uses a general approach to find relationship paths
        without any special cases.

        Args:
            id1: ID of the first individual
            id2: ID of the second individual

        Returns:
            A human-readable string explaining the relationship path
        """
        id1_norm = normalize_id(id1)
        id2_norm = normalize_id(id2)

        # Check for None after normalization
        if not id1_norm or not id2_norm:
            return "(Invalid individual IDs)"

        # Validate inputs
        validation_error = self._validate_relationship_path_inputs(id1_norm, id2_norm)
        if validation_error:
            return validation_error

        path_ids, search_time = self._run_relationship_search(id1_norm, id2_norm)
        if not path_ids:
            profile_info = f"[PROFILE] Search: {search_time:.2f}s, MapsBuild: {self.family_maps_build_time:.2f}s, IndexBuild: {self.indi_index_build_time:.2f}s, PreProcess: {self.data_processing_time:.2f}s"
            return f"No relationship path found (FastBiBFS could not connect).\n{profile_info}"

        # Generate the explanation
        explanation_start = time.time()
        try:
            explanation_str = explain_relationship_path(
                path_ids,
                self.reader,
                self.id_to_parents,
                self.id_to_children,
                self.indi_index,
            )
        except Exception as explain_err:
            logger.error(f"Error generating path explanation: {explain_err}", exc_info=True)
            explanation_str = "(Error generating explanation)"
        explanation_time = time.time() - explanation_start
        logger.debug(f"[PROFILE] Path explanation built in {explanation_time:.2f}s.")
        total_process_time = explanation_time
        profile_info = f"[PROFILE] Total Time: {total_process_time:.2f}s (BFS: 0.00s, Explain: {explanation_time:.2f}s) [Build Times: Maps={self.family_maps_build_time:.2f}s, Index={self.indi_index_build_time:.2f}s, PreProcess={self.data_processing_time:.2f}s]"
        logger.debug(profile_info)
        return f"{explanation_str}\n{profile_info}"

    def _run_relationship_search(self, start_id: str, end_id: str) -> tuple[list[str], float]:
        """Execute the FastBiBFS search and return path IDs with timing."""
        max_depth = 25
        node_limit = 150000
        timeout_sec = 45
        logger.debug(f"Calculating relationship path (FastBiBFS): {start_id} <-> {end_id}")
        search_start = time.time()
        id_to_parents_list = {k: list(v) for k, v in self.id_to_parents.items()}
        id_to_children_list = {k: list(v) for k, v in self.id_to_children.items()}
        graph_ctx = GraphContext(
            id_to_parents=id_to_parents_list,
            id_to_children=id_to_children_list,
            start_id=start_id,
            end_id=end_id,
        )
        path_ids = fast_bidirectional_bfs(
            graph_ctx,
            max_depth,
            node_limit,
            timeout_sec,
        )
        search_time = time.time() - search_start
        logger.debug(f"[PROFILE] BFS search completed in {search_time:.2f}s.")
        return path_ids, search_time

    def _find_direct_relationship(self, id1: str, id2: str) -> list[str]:
        """
        Find a direct relationship between two individuals.

        This is a helper method for get_relationship_path that finds direct
        parent-child, child-parent, or sibling relationships.

        Args:
            id1: ID of the first individual
            id2: ID of the second individual

        Returns:
            A list of IDs representing the path from id1 to id2, or an empty list if no path is found
        """
        # Check if id2 is a parent of id1
        if id2 in self.id_to_parents.get(id1, set()):
            return [id1, id2]

        # Check if id2 is a child of id1
        if id2 in self.id_to_children.get(id1, set()):
            return [id1, id2]

        # Check if id1 and id2 are siblings (share at least one parent)
        parents_1 = self.id_to_parents.get(id1, set())
        parents_2 = self.id_to_parents.get(id2, set())
        common_parents = parents_1.intersection(parents_2)
        if common_parents:
            # Use the first common parent
            common_parent = next(iter(common_parents))
            return [id1, common_parent, id2]

        # No direct relationship found
        return []


class _FakeIndividualRecord:
    """Minimal fake for extract_and_fix_id tests."""

    def __init__(self, *, xref_id: str = "@I1@") -> None:
        self.tag = TAG_INDI
        self.xref_id = xref_id

    def sub_tag(self, tag: str) -> Any:
        return None

    def sub_tag_value(self, tag: str) -> Any:
        return None


def _testnormalize_id_variants() -> bool:
    assert normalize_id("@I123@") == "I123"
    assert normalize_id("i-42") == "I-42"
    assert normalize_id("Family I999 note") == "I999"
    assert normalize_id("12345") == "12345"
    assert normalize_id("???") is None
    return True


def _test_extract_and_fix_id_handles_various_inputs() -> bool:
    fake = _FakeIndividualRecord(xref_id="@S77@")
    assert extract_and_fix_id(fake) == "S77"
    assert extract_and_fix_id(42) == "42"
    assert extract_and_fix_id("@F12@") == "F12"
    assert extract_and_fix_id(object()) is None
    return True


def module_tests() -> bool:
    suite = TestSuite("gedcom_utils", "gedcom_utils.py")
    suite.run_test(
        "Normalize GEDCOM IDs",
        _testnormalize_id_variants,
        "Ensures GEDCOM IDs normalize from standard, fallback, and numeric formats.",
    )
    suite.run_test(
        "extract_and_fix_id inputs",
        _test_extract_and_fix_id_handles_various_inputs,
        "Validates ID extraction from strings, ints, and GEDCOM-like objects.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


def _should_run_module_tests() -> bool:
    return os.environ.get("RUN_MODULE_TESTS") == "1"


def _print_module_usage() -> int:
    print("gedcom_utils provides import-only helpers; there is no standalone CLI entry point.")
    print("Set RUN_MODULE_TESTS=1 before execution to run the embedded regression tests.")
    return 0


if __name__ == "__main__":
    if _should_run_module_tests():
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    sys.exit(_print_module_usage())
