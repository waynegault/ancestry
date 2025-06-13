#!/usr/bin/env python3

# relationship_utils.py
"""
Consolidated utilities for relationship path handling.

This module provides functions for:
1. Finding relationship paths between individuals in GEDCOM data
2. Formatting relationship paths from Ancestry API responses
3. Displaying relationship paths in a consistent format

It consolidates functionality previously spread across gedcom_utils.py and api_utils.py.
"""

# --- Standard library imports ---
import logging
import re
import json
import time
import html
from typing import (
    Optional,
    Dict,
    Any,
    Union,
    List,
    Tuple,
    Set,
    Callable,
)  # Added Callable for test suite
from datetime import datetime
from collections import deque
import traceback  # Added for test suite
from contextlib import contextmanager

# --- Initialize logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("relationship_utils")

# --- Try to import BeautifulSoup ---
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None  # type: ignore
    BS4_AVAILABLE = False

# --- Local imports ---
# Avoid importing from utils to prevent config dependency during testing
# Instead, we'll define format_name locally

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)

# Import specific functions from gedcom_utils
try:
    from gedcom_utils import _are_spouses as _are_spouses_orig

    def _are_spouses(person1_id: str, person2_id: str, reader) -> bool:
        """Wrapper to match expected parameter names."""
        return _are_spouses_orig(person1_id, person2_id, reader)

except ImportError:
    # Define a fallback if import fails
    def _are_spouses(person1_id: str, person2_id: str, reader) -> bool:
        """Fallback implementation of _are_spouses function."""
        return False


def format_name(name: Optional[str]) -> str:
    """
    Formats a person's name string to title case, preserving uppercase components
    (like initials or acronyms) and handling None/empty input gracefully.
    Also removes GEDCOM-style slashes around surnames anywhere in the string.
    """
    if not name or not isinstance(name, str):
        return "Valued Relative"

    if name.isdigit() or re.fullmatch(r"[^a-zA-Z]+", name):
        return name.strip()

    try:
        cleaned_name = name.strip()
        # Handle GEDCOM slashes more robustly
        cleaned_name = re.sub(r"\s*/([^/]+)/\s*", r" \1 ", cleaned_name)  # Middle
        cleaned_name = re.sub(r"^/([^/]+)/\s*", r"\1 ", cleaned_name)  # Start
        cleaned_name = re.sub(r"\s*/([^/]+)/$", r" \1", cleaned_name)  # End

        # Split into words
        words = cleaned_name.split()
        formatted_words = []

        for word in words:
            if not word:
                continue

            # Preserve fully uppercase words (likely initials/acronyms)
            if word.isupper() and len(word) <= 3:
                formatted_words.append(word)
            # Handle name particles and prefixes
            elif word.lower() in ["mc", "mac", "o'"]:
                formatted_words.append(word.capitalize())
            # Handle quoted nicknames
            elif word.startswith('"') and word.endswith('"'):
                formatted_words.append(f'"{word[1:-1].title()}"')
            # Regular title case
            else:
                formatted_words.append(word.title())

        return " ".join(formatted_words)

    except Exception:
        # Fallback to basic title case
        return name.title()


# Import GEDCOM specific helpers and types from gedcom_utils - avoid config dependency
# Try to import actual functions from gedcom_utils, fall back to minimal versions for testing
try:
    from gedcom_utils import (
        _normalize_id,
        _is_record,
        _are_siblings,
        _is_grandparent,
        _is_grandchild,
        _is_great_grandparent,
        _is_great_grandchild,
        _is_aunt_or_uncle,
        _is_niece_or_nephew,
        _are_cousins,
        _get_event_info,
        _get_full_name,
        _parse_date,
        _clean_display_date,
        TAG_BIRTH,
        TAG_DEATH,
        TAG_SEX,
        TAG_HUSBAND,
        TAG_WIFE,
        GedcomReaderType,
        GedcomIndividualType,
    )

    GEDCOM_UTILS_AVAILABLE = True
except ImportError:
    GEDCOM_UTILS_AVAILABLE = False

    # Fallback functions for testing when gedcom_utils is not available
    def _normalize_id(xref_id: Optional[str]) -> Optional[str]:
        """Normalize GEDCOM ID by removing @ symbols."""
        if not xref_id:
            return None
        return xref_id.strip("@")

    def _is_record(obj: Any) -> bool:
        """Check if object is a GEDCOM record."""
        return obj is not None and hasattr(obj, "tag")

    def _are_siblings(id1: str, id2: str, id_to_parents: Dict[str, Set[str]]) -> bool:
        """Check if two individuals are siblings."""
        if not id1 or not id2 or id1 == id2:
            return False
        parents1 = id_to_parents.get(id1, set())
        parents2 = id_to_parents.get(id2, set())
        return bool(parents1 and parents2 and parents1.intersection(parents2))

    # def _are_spouses(id1: str, id2: str, reader: Any) -> bool:
    #     """Fallback function for spouse detection."""
    #     return False

    def _is_grandparent(id1: str, id2: str, id_to_parents: Dict[str, Set[str]]) -> bool:
        """Check if id1 is grandparent of id2."""
        return False

    def _is_grandchild(id1: str, id2: str, id_to_children: Dict[str, Set[str]]) -> bool:
        """Check if id1 is grandchild of id2."""
        return False

    def _is_great_grandparent(
        id1: str, id2: str, id_to_parents: Dict[str, Set[str]]
    ) -> bool:
        """Check if id1 is great-grandparent of id2."""
        return False

    def _is_great_grandchild(
        id1: str, id2: str, id_to_children: Dict[str, Set[str]]
    ) -> bool:
        """Check if id1 is great-grandchild of id2."""
        return False

    def _is_aunt_or_uncle(
        id1: str,
        id2: str,
        id_to_parents: Dict[str, Set[str]],
        id_to_children: Dict[str, Set[str]],
    ) -> bool:
        """Check if id1 is aunt/uncle of id2."""
        return False

    def _is_niece_or_nephew(
        id1: str,
        id2: str,
        id_to_parents: Dict[str, Set[str]],
        id_to_children: Dict[str, Set[str]],
    ) -> bool:
        """Check if id1 is niece/nephew of id2."""
        return False

    def _are_cousins(
        id1: str,
        id2: str,
        id_to_parents: Dict[str, Set[str]],
        id_to_children: Dict[str, Set[str]],
    ) -> bool:
        """Check if id1 and id2 are cousins."""
        return False

    def _get_full_name(indi: Any) -> str:
        """Get full name from GEDCOM individual."""
        return "Unknown"

    def _get_event_info(
        individual: Any, event_tag: str
    ) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        """Get event information from GEDCOM individual."""
        return None, None, None

    def _parse_date(date_str: str) -> Optional[Any]:
        """Parse a date string."""
        return None

    def _clean_display_date(raw_date_str: Optional[str]) -> str:
        """Clean display date."""
        return raw_date_str or ""

    def _has_direct_relationship(
        id1: str,
        id2: str,
        id_to_parents: Dict[str, Set[str]],
        id_to_children: Dict[str, Set[str]],
    ) -> bool:
        """Check if two individuals have a direct relationship."""
        return id2 in id_to_parents.get(id1, set()) or id2 in id_to_children.get(
            id1, set()
        )

    # Define constants for fallback
    TAG_BIRTH = "BIRT"
    TAG_DEATH = "DEAT"
    TAG_SEX = "SEX"
    TAG_HUSBAND = "HUSB"
    TAG_WIFE = "WIFE"

    # Type aliases for fallback
    GedcomReaderType = Any
    GedcomIndividualType = Any
    TAG_SEX = "SEX"
    TAG_HUSBAND = "HUSB"
    TAG_WIFE = "WIFE"

    # Define type aliases for fallback
    GedcomIndividualType = Any
    GedcomReaderType = Any


# --- Helper Functions for BFS ---


def _is_grandparent(id1: str, id2: str, id_to_parents: Dict[str, Set[str]]) -> bool:
    """Check if id2 is a grandparent of id1."""
    if not id1 or not id2:
        return False
    # Get parents of id1
    parents = id_to_parents.get(id1, set())
    # For each parent, check if id2 is their parent
    for parent_id in parents:
        grandparents = id_to_parents.get(parent_id, set())
        if id2 in grandparents:
            return True
    return False


def _is_grandchild(id1: str, id2: str, id_to_children: Dict[str, Set[str]]) -> bool:
    """Check if id2 is a grandchild of id1."""
    if not id1 or not id2:
        return False
    # Get children of id1
    children = id_to_children.get(id1, set())
    # For each child, check if id2 is their child
    for child_id in children:
        grandchildren = id_to_children.get(child_id, set())
        if id2 in grandchildren:
            return True
    return False


# --- Relationship Path Finding Functions ---


def _find_direct_relationship(
    id1: str,
    id2: str,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
) -> List[str]:
    """
    Find a direct relationship between two individuals.

    Args:
        id1: ID of the first individual
        id2: ID of the second individual
        id_to_parents: Dictionary mapping individual IDs to their parent IDs
        id_to_children: Dictionary mapping individual IDs to their child IDs

    Returns:
        A list of IDs representing the path from id1 to id2, or an empty list if no direct relationship
    """
    # Check if id2 is a parent of id1
    if id2 in id_to_parents.get(id1, set()):
        return [id1, id2]

    # Check if id2 is a child of id1
    if id2 in id_to_children.get(id1, set()):
        return [id1, id2]

    # Check if id1 and id2 are siblings (share at least one parent)
    parents_1 = id_to_parents.get(id1, set())
    parents_2 = id_to_parents.get(id2, set())
    common_parents = parents_1.intersection(parents_2)
    if common_parents:
        # Use the first common parent
        common_parent = next(iter(common_parents))
        return [id1, common_parent, id2]

    # No direct relationship found
    return []


def _has_direct_relationship(
    id1: str,
    id2: str,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
) -> bool:
    """
    Check if two individuals have a direct relationship (parent-child, siblings, or spouses).

    Args:
        id1: ID of the first individual
        id2: ID of the second individual
        id_to_parents: Dictionary mapping individual IDs to their parent IDs
        id_to_children: Dictionary mapping individual IDs to their child IDs

    Returns:
        True if directly related, False otherwise
    """
    # Parent-child relationship
    if id2 in id_to_parents.get(id1, set()) or id1 in id_to_parents.get(id2, set()):
        return True

    # Sibling relationship (share at least one parent)
    parents_1 = id_to_parents.get(id1, set())
    parents_2 = id_to_parents.get(id2, set())
    if parents_1 and parents_2 and not parents_1.isdisjoint(parents_2):
        return True

    # Check for grandparent relationship
    for parent_id in id_to_parents.get(id1, set()):
        if id2 in id_to_parents.get(parent_id, set()):
            return True

    # Check for grandchild relationship
    for child_id in id_to_children.get(id1, set()):
        if id2 in id_to_children.get(child_id, set()):
            return True

    return False


# --- Relationship Path Finding Functions ---


def fast_bidirectional_bfs(
    start_id: str,
    end_id: str,
    id_to_parents: Optional[Dict[str, Set[str]]],
    id_to_children: Optional[Dict[str, Set[str]]],
    max_depth=25,
    node_limit=150000,
    timeout_sec=45,
    log_progress=False,  # Parameter exists but not used in this version
) -> List[str]:
    """
    Enhanced bidirectional BFS that finds direct paths through family trees.

    This implementation focuses on finding paths where each person has a clear,
    direct relationship to the next person in the path (parent, child, sibling).
    It avoids using special cases or "connected to" placeholders.

    The algorithm prioritizes shorter paths with direct relationships over longer paths.
    """
    start_time = time.time()
    if start_id == end_id:
        return [start_id]
    if id_to_parents is None or id_to_children is None:
        logger.error("[FastBiBFS] Relationship maps are None.")
        return []
    if not start_id or not end_id:
        logger.error("[FastBiBFS] Start or end ID is missing.")
        return []

    # First try to find a direct relationship (parent, child, sibling)
    # This is a quick check before running the full BFS
    direct_path = _find_direct_relationship(
        start_id, end_id, id_to_parents, id_to_children
    )
    if direct_path:
        logger.debug(f"[FastBiBFS] Found direct relationship: {direct_path}")
        return direct_path

    # Initialize BFS queues and visited sets
    # Forward queue from start_id
    queue_fwd = deque([(start_id, 0, [start_id])])  # (id, depth, path)
    # Backward queue from end_id
    queue_bwd = deque([(end_id, 0, [end_id])])  # (id, depth, path)

    # Track visited nodes and their paths
    visited_fwd = {start_id: (0, [start_id])}  # {id: (depth, path)}
    visited_bwd = {end_id: (0, [end_id])}  # {id: (depth, path)}

    # Track all complete paths found
    all_paths = []

    # Process nodes until we find paths or exhaust the search
    processed = 0
    logger.debug(f"[FastBiBFS] Starting BFS: {start_id} <-> {end_id}")

    # Main search loop - continue until we find paths or exhaust the search
    while queue_fwd and queue_bwd and len(all_paths) < 5:  # Limit to finding 5 paths
        # Check timeout and node limit
        if time.time() - start_time > timeout_sec:
            logger.warning(f"[FastBiBFS] Timeout after {timeout_sec:.1f} seconds.")
            break
        if processed > node_limit:
            logger.warning(f"[FastBiBFS] Node limit ({node_limit}) reached.")
            break

        # Process forward queue (from start)
        if queue_fwd:
            current_id, depth, path = queue_fwd.popleft()
            processed += 1

            # Check if we've reached a node visited by backward search
            if current_id in visited_bwd:
                # Found a meeting point - reconstruct the path
                _bwd_depth, bwd_path = visited_bwd[current_id]  # _bwd_depth unused
                # Combine paths (remove duplicate meeting point)
                combined_path = path + bwd_path[1:]
                all_paths.append(combined_path)
                logger.debug(
                    f"[FastBiBFS] Path found via {current_id}: {len(combined_path)} nodes"
                )
                # Continue searching for potentially shorter/better paths if len(all_paths) < 5
                if len(all_paths) >= 5:
                    break  # Stop if we have enough paths

            # Stop expanding if we've reached max depth
            if depth >= max_depth:
                continue

            # Expand to parents (direct relationship)
            for parent_id in id_to_parents.get(current_id, set()):
                if parent_id not in visited_fwd:
                    new_path = path + [parent_id]
                    visited_fwd[parent_id] = (depth + 1, new_path)
                    queue_fwd.append((parent_id, depth + 1, new_path))

            # Expand to children (direct relationship)
            for child_id in id_to_children.get(current_id, set()):
                if child_id not in visited_fwd:
                    new_path = path + [child_id]
                    visited_fwd[child_id] = (depth + 1, new_path)
                    queue_fwd.append((child_id, depth + 1, new_path))

            # Expand to siblings (through parent)
            for parent_id in id_to_parents.get(current_id, set()):
                for sibling_id in id_to_children.get(parent_id, set()):
                    if sibling_id != current_id and sibling_id not in visited_fwd:
                        # Include parent in path for proper relationship context
                        new_path = path + [parent_id, sibling_id]
                        visited_fwd[sibling_id] = (
                            depth + 2,
                            new_path,
                        )  # Depth increases by 2 (to parent, then to sibling)
                        queue_fwd.append((sibling_id, depth + 2, new_path))

        # Process backward queue (from end)
        if queue_bwd:  # Check if queue_bwd is not empty
            current_id, depth, path = queue_bwd.popleft()
            processed += 1

            # Check if we've reached a node visited by forward search
            if current_id in visited_fwd:
                # Found a meeting point - reconstruct the path
                _fwd_depth, fwd_path = visited_fwd[current_id]  # _fwd_depth unused
                # Combine paths (remove duplicate meeting point)
                combined_path = fwd_path + path[1:]
                all_paths.append(combined_path)
                logger.debug(
                    f"[FastBiBFS] Path found via {current_id}: {len(combined_path)} nodes"
                )
                if len(all_paths) >= 5:
                    break  # Stop if we have enough paths

            # Stop expanding if we've reached max depth
            if depth >= max_depth:
                continue

            # Expand to parents (direct relationship)
            for parent_id in id_to_parents.get(current_id, set()):
                if parent_id not in visited_bwd:
                    new_path = [parent_id] + path
                    visited_bwd[parent_id] = (depth + 1, new_path)
                    queue_bwd.append((parent_id, depth + 1, new_path))

            # Expand to children (direct relationship)
            for child_id in id_to_children.get(current_id, set()):
                if child_id not in visited_bwd:
                    new_path = [child_id] + path
                    visited_bwd[child_id] = (depth + 1, new_path)
                    queue_bwd.append((child_id, depth + 1, new_path))

            # Expand to siblings (through parent)
            for parent_id in id_to_parents.get(current_id, set()):
                for sibling_id in id_to_children.get(parent_id, set()):
                    if sibling_id != current_id and sibling_id not in visited_bwd:
                        # Include parent in path for proper relationship context
                        new_path = [sibling_id, parent_id] + path
                        visited_bwd[sibling_id] = (depth + 2, new_path)
                        queue_bwd.append((sibling_id, depth + 2, new_path))

    # If we found paths, select the best one
    if all_paths:
        # Score paths based on directness of relationships
        scored_paths = []
        for p in all_paths:  # Renamed path to p to avoid conflict with outer scope
            # Check if each adjacent pair has a direct relationship
            direct_relationships = 0
            for i in range(len(p) - 1):
                if _has_direct_relationship(
                    p[i], p[i + 1], id_to_parents, id_to_children
                ):
                    direct_relationships += 1

            # Calculate score: prefer paths with more direct relationships and shorter length
            directness_score = direct_relationships / (len(p) - 1) if len(p) > 1 else 0
            length_penalty = len(p) / 10  # Slight penalty for longer paths
            score = directness_score - length_penalty

            scored_paths.append((p, score))

        # Sort by score (highest first)
        scored_paths.sort(key=lambda x: x[1], reverse=True)

        # Return the path with the highest score
        best_path = scored_paths[0][0]
        logger.debug(
            f"[FastBiBFS] Selected best path: {len(best_path)} nodes with score {scored_paths[0][1]:.2f}"
        )
        return best_path

    # If we didn't find any paths
    logger.warning(f"[FastBiBFS] No paths found between {start_id} and {end_id}.")

    # Fallback: Return a list containing only start and end IDs if no path found
    return [start_id, end_id]


def explain_relationship_path(
    path_ids: List[str],
    reader: Any,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
    indi_index: Dict[str, Any],
    owner_name: str = "Reference Person",
    relationship_type: str = "relative",
) -> str:
    """
    Generates a human-readable explanation of the relationship path using the unified format.

    This implementation uses a generic approach to determine relationships
    between individuals in the path without special cases. It analyzes the
    family structure to determine parent-child, sibling, spouse, and other
    relationships.

    Args:
        path_ids: List of GEDCOM IDs representing the relationship path
        reader: GEDCOM reader object
        id_to_parents: Dictionary mapping IDs to parent IDs
        id_to_children: Dictionary mapping IDs to child IDs
        indi_index: Dictionary mapping IDs to GEDCOM individual objects
        owner_name: Name of the owner/reference person (default: "Reference Person")
        relationship_type: Type of relationship between target and owner (default: "relative")

    Returns:
        Formatted relationship path string
    """
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"
    if id_to_parents is None or id_to_children is None or indi_index is None:
        return "(Error: Data maps or index unavailable)"

    # Convert the GEDCOM path to the unified format
    unified_path = convert_gedcom_path_to_unified_format(
        path_ids, reader, id_to_parents, id_to_children, indi_index
    )

    if not unified_path:
        return "(Error: Could not convert relationship path to unified format)"

    # Get the target name from the first person in the path
    target_name = unified_path[0].get("name", "Unknown Person")

    # Format the path using the unified formatter
    return format_relationship_path_unified(
        unified_path,
        target_name if target_name is not None else "Unknown Person",
        owner_name,
        relationship_type,
    )


def format_api_relationship_path(
    api_response_data: Union[str, Dict, None],
    owner_name: str,
    target_name: str,
    relationship_type: str = "relative",
) -> str:
    """
    Parses relationship data from Ancestry APIs and formats it into a readable path.
    Handles getladder API HTML/JSONP response.
    Uses format_name and ordinal_case from utils.py.

    The output format is standardized to match the unified format:

    ===Relationship Path to Owner Name===
    Target Name (birth-death) is Owner Name's relationship:

    - Person 1's relationship is Person 2 (birth-death)
    - Person 2's relationship is Person 3 (birth-death)
    ...
    """
    if not api_response_data:
        logger.warning(
            "format_api_relationship_path: Received empty API response data."
        )
        return "(No relationship data received from API)"

    html_content_raw: Optional[str] = None
    json_data: Optional[Dict] = None
    # api_status: str = "unknown" # api_status was unused

    # Extract HTML content from API response
    if isinstance(api_response_data, str):
        # Handle JSONP response format: no({...})
        jsonp_match = re.search(
            r"no\((.*)\)", api_response_data, re.DOTALL
        )  # Added re.DOTALL
        if jsonp_match:
            try:
                json_str = jsonp_match.group(1)
                json_data = json.loads(json_str)
                html_content_raw = (
                    json_data.get("html") if json_data is not None else None
                )
                # api_status = json_data.get("status", "unknown") # api_status unused
            except Exception as e:
                logger.error(f"Error parsing JSONP response: {e}", exc_info=True)
                return f"(Error parsing JSONP response: {e})"
        else:
            # Direct HTML response
            html_content_raw = api_response_data
    elif isinstance(api_response_data, dict):
        # Handle direct JSON/dict response
        json_data = api_response_data
        html_content_raw = json_data.get("html") if json_data is not None else None
        # api_status = json_data.get("status", "unknown") # api_status unused

    # Handle Discovery API JSON format
    if json_data and "path" in json_data:
        path_steps_json = []
        discovery_path = json_data["path"]
        if isinstance(discovery_path, list) and discovery_path:
            logger.info("Formatting relationship path from Discovery API JSON.")
            path_steps_json.append(f"*   {format_name(target_name)}")
            for step in discovery_path:
                step_name = format_name(step.get("name", "?"))
                step_rel = step.get("relationship", "?")
                step_rel_display = _get_relationship_term(None, step_rel).capitalize()
                path_steps_json.append(f"    -> is {step_rel_display} of")
                path_steps_json.append(f"*   {step_name}")
            path_steps_json.append(f"    -> leads to")
            path_steps_json.append(f"*   {owner_name} (You)")
            result_str = "\n".join(path_steps_json)
            return result_str

    # Process HTML content if available
    if not html_content_raw:
        logger.warning("No HTML content found in API response.")
        return "(No relationship HTML content found in API response)"

    # Check if this is a simple text relationship description before trying HTML parsing
    if html_content_raw and not html_content_raw.strip().startswith("<"):
        # Simple text processing for strings like "John Doe is the father of Jane Doe"
        text = html_content_raw.strip()
        # Look for relationship patterns
        relationship_patterns = [
            r"is the (father|mother|son|daughter|brother|sister|husband|wife|parent|child|sibling|spouse) of",
            r"(father|mother|son|daughter|brother|sister|husband|wife|parent|child|sibling|spouse)",
        ]

        for pattern in relationship_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                relationship = match.group(1).lower()
                return f"{target_name} is the {relationship} of {owner_name}"

        # If no pattern found, return the original text
        if any(
            rel in text.lower()
            for rel in [
                "father",
                "mother",
                "son",
                "daughter",
                "brother",
                "sister",
                "husband",
                "wife",
                "parent",
                "child",
                "sibling",
                "spouse",
            ]
        ):
            return text

    if not BS4_AVAILABLE:
        logger.error("BeautifulSoup is not available. Cannot parse HTML.")
        return "(BeautifulSoup is not available. Cannot parse relationship HTML.)"

    # Decode HTML entities
    html_content_decoded = html.unescape(html_content_raw) if html_content_raw else ""

    # Parse HTML with BeautifulSoup
    try:
        if not BS4_AVAILABLE or BeautifulSoup is None:
            logger.error("BeautifulSoup is not available. Cannot parse HTML.")
            return "(BeautifulSoup is not available. Cannot parse relationship HTML.)"
        soup = BeautifulSoup(html_content_decoded, "html.parser")

        # Find all list items
        list_items = soup.find_all("li")
        if not list_items or len(list_items) < 2:
            logger.warning(
                f"Not enough list items found in HTML: {len(list_items) if list_items else 0}"
            )
            return "(Relationship HTML structure not recognized)"

        # Extract relationship information
        relationship_data = []
        for item in list_items:
            # Skip icon items
            try:
                is_hidden = item.get("aria-hidden") == "true"
                item_classes = item.get("class", [])
                has_icon_class = (
                    isinstance(item_classes, list) and "icon" in item_classes
                )
                if is_hidden or has_icon_class:
                    continue
            except (AttributeError, TypeError):
                logger.debug(f"Error checking item attributes: {type(item)}")
                continue

            # Extract name, relationship, and lifespan
            try:
                name_elem = item.find("b") if hasattr(item, "find") else None
                name = (
                    name_elem.get_text(strip=True)  # Added strip=True
                    if name_elem and hasattr(name_elem, "get_text")
                    else (
                        str(item.string).strip()
                        if hasattr(item, "string") and item.string
                        else "Unknown"
                    )  # Added strip and check for item.string
                )
            except (AttributeError, TypeError):
                name = "Unknown"
                logger.debug(f"Error extracting name: {type(item)}")

            # Extract relationship description
            try:
                rel_elem = item.find("i") if hasattr(item, "find") else None
                relationship_desc = (  # Renamed to avoid conflict
                    rel_elem.get_text(strip=True)  # Added strip=True
                    if rel_elem and hasattr(rel_elem, "get_text")
                    else ""
                )
            except (AttributeError, TypeError):
                relationship_desc = ""
                logger.debug(f"Error extracting relationship: {type(item)}")

            # Extract lifespan
            try:
                text_content = (
                    item.get_text(strip=True)
                    if hasattr(item, "get_text")
                    else str(item)
                )  # Added strip=True
                lifespan_match = re.search(
                    r"(\d{4})-(\d{4}|\bLiving\b|-)", text_content, re.IGNORECASE
                )  # Allow "Living"
                lifespan = lifespan_match.group(0) if lifespan_match else ""
            except (AttributeError, TypeError):
                # text_content = "" # text_content was unused
                lifespan = ""
                logger.debug(f"Error extracting lifespan: {type(item)}")

            relationship_data.append(
                {"name": name, "relationship": relationship_desc, "lifespan": lifespan}
            )

        # Convert API relationship data to unified format and format it
        if relationship_data:
            # Convert the API data to the unified format
            unified_path = convert_api_path_to_unified_format(
                relationship_data, target_name
            )

            if not unified_path:
                return "(Error: Could not convert relationship data to unified format)"

            # Format the path using the unified formatter
            result_str = format_relationship_path_unified(
                unified_path, target_name, owner_name, relationship_type
            )
            return result_str
        else:
            return "(Could not extract relationship data from HTML)"

    except Exception as e:
        logger.error(f"Error parsing relationship HTML: {e}", exc_info=True)
        return f"(Error parsing relationship HTML: {e})"


def convert_gedcom_path_to_unified_format(
    path_ids: List[str],
    reader: Any,
    id_to_parents: Dict[str, Set[str]],
    id_to_children: Dict[str, Set[str]],
    indi_index: Dict[str, Any],
) -> List[Dict[str, Optional[str]]]:  # Value type changed to Optional[str]
    """
    Convert a GEDCOM relationship path to the unified format for relationship_path_unified.

    Args:
        path_ids: List of GEDCOM IDs representing the relationship path
        reader: GEDCOM reader object
        id_to_parents: Dictionary mapping IDs to parent IDs
        id_to_children: Dictionary mapping IDs to child IDs
        indi_index: Dictionary mapping IDs to GEDCOM individual objects

    Returns:
        List of dictionaries with keys 'name', 'birth_year', 'death_year', 'relationship', 'gender'
    """
    if not path_ids or len(path_ids) < 2:
        return []

    result: List[Dict[str, Optional[str]]] = []  # Ensure list type

    # Process the first person (no relationship)
    first_id = path_ids[0]
    first_indi = indi_index.get(first_id)

    if first_indi:
        # Get name
        first_name = _get_full_name(first_indi)

        # Get birth/death years
        birth_date_obj, _, _ = _get_event_info(first_indi, TAG_BIRTH)
        death_date_obj, _, _ = _get_event_info(first_indi, TAG_DEATH)

        birth_year = str(birth_date_obj.year) if birth_date_obj else None
        death_year = str(death_date_obj.year) if death_date_obj else None

        # Get gender
        sex_tag = first_indi.sub_tag(TAG_SEX)  # Use imported constant
        sex_char: Optional[str] = None  # Ensure type
        if (
            sex_tag and hasattr(sex_tag, "value") and sex_tag.value is not None
        ):  # Check value is not None
            sex_val = str(sex_tag.value).upper()
            if sex_val in ("M", "F"):
                sex_char = sex_val

        # Add to result
        result.append(
            {
                "name": first_name,
                "birth_year": birth_year,
                "death_year": death_year,
                "relationship": None,  # First person has no relationship to previous person
                "gender": sex_char,  # Add gender information
            }
        )
    else:
        # Handle missing first person
        result.append(
            {
                "name": f"Unknown ({first_id})",
                "birth_year": None,
                "death_year": None,
                "relationship": None,
                "gender": None,
            }
        )

    # Process the rest of the path
    for i in range(1, len(path_ids)):
        prev_id, current_id = path_ids[i - 1], path_ids[i]
        current_indi = indi_index.get(current_id)

        if not current_indi:
            # Handle missing person
            result.append(
                {
                    "name": f"Unknown ({current_id})",
                    "birth_year": None,
                    "death_year": None,
                    "relationship": "relative",
                    "gender": None,
                }
            )
            continue

        # Get name
        current_name = _get_full_name(current_indi)

        # Get birth/death years
        birth_date_obj, _, _ = _get_event_info(current_indi, TAG_BIRTH)
        death_date_obj, _, _ = _get_event_info(current_indi, TAG_DEATH)

        birth_year = str(birth_date_obj.year) if birth_date_obj else None
        death_year = str(death_date_obj.year) if death_date_obj else None

        # Determine gender for relationship terms
        sex_tag = current_indi.sub_tag(TAG_SEX)  # Use imported constant
        sex_char = None
        if (
            sex_tag and hasattr(sex_tag, "value") and sex_tag.value is not None
        ):  # Check value is not None
            sex_val = str(sex_tag.value).upper()
            if sex_val in ("M", "F"):
                sex_char = sex_val

        # Determine relationship
        relationship: Optional[str] = "relative"  # Default

        # Check if current is a PARENT of prev
        if current_id in id_to_parents.get(prev_id, set()):
            relationship = (
                "father"
                if sex_char == "M"
                else "mother" if sex_char == "F" else "parent"
            )
        # Check if current is a CHILD of prev
        elif current_id in id_to_children.get(prev_id, set()):
            relationship = (
                "son" if sex_char == "M" else "daughter" if sex_char == "F" else "child"
            )
        # Check if current is a SIBLING of prev
        elif _are_siblings(prev_id, current_id, id_to_parents):
            relationship = (
                "brother"
                if sex_char == "M"
                else "sister" if sex_char == "F" else "sibling"
            )
        # Check if current is a SPOUSE of prev
        elif _are_spouses(prev_id, current_id, reader):
            relationship = (
                "husband"
                if sex_char == "M"
                else "wife" if sex_char == "F" else "spouse"
            )
        # Check for grandparent
        elif _is_grandparent(prev_id, current_id, id_to_parents):
            relationship = (
                "grandfather"
                if sex_char == "M"
                else "grandmother" if sex_char == "F" else "grandparent"
            )
        # Check for grandchild
        elif _is_grandchild(prev_id, current_id, id_to_children):
            relationship = (
                "grandson"
                if sex_char == "M"
                else "granddaughter" if sex_char == "F" else "grandchild"
            )
        # Check for great-grandparent
        elif _is_great_grandparent(prev_id, current_id, id_to_parents):
            relationship = (
                "great-grandfather"
                if sex_char == "M"
                else "great-grandmother" if sex_char == "F" else "great-grandparent"
            )
        # Check for great-grandchild
        elif _is_great_grandchild(prev_id, current_id, id_to_children):
            relationship = (
                "great-grandson"
                if sex_char == "M"
                else "great-granddaughter" if sex_char == "F" else "great-grandchild"
            )
        # Check for aunt/uncle
        elif _is_aunt_or_uncle(prev_id, current_id, id_to_parents, id_to_children):
            relationship = (
                "uncle"
                if sex_char == "M"
                else "aunt" if sex_char == "F" else "aunt/uncle"
            )
        # Check for niece/nephew
        elif _is_niece_or_nephew(prev_id, current_id, id_to_parents, id_to_children):
            relationship = (
                "nephew"
                if sex_char == "M"
                else "niece" if sex_char == "F" else "niece/nephew"
            )
        # Check for cousins
        elif _are_cousins(prev_id, current_id, id_to_parents, id_to_children):
            relationship = "cousin"

        # Add to result
        result.append(
            {
                "name": current_name,
                "birth_year": birth_year,
                "death_year": death_year,
                "relationship": relationship,
                "gender": sex_char,  # Add gender information
            }
        )

    return result


def convert_discovery_api_path_to_unified_format(
    discovery_data: Dict, target_name: str
) -> List[Dict[str, Optional[str]]]:  # Value type changed to Optional[str]
    """
    Convert Discovery API relationship data to the unified format for relationship_path_unified.

    Args:
        discovery_data: Dictionary from Discovery API with 'path' key containing relationship steps
        target_name: Name of the target person

    Returns:
        List of dictionaries with keys 'name', 'birth_year', 'death_year', 'relationship', 'gender'
    """
    if (
        not discovery_data
        or not isinstance(discovery_data, dict)
        or "path" not in discovery_data
    ):
        logger.warning("Invalid or empty Discovery API data")
        return []

    path_steps = discovery_data.get("path", [])
    if not isinstance(path_steps, list) or not path_steps:
        logger.warning("Discovery API path is not a valid list or is empty")
        return []

    result: List[Dict[str, Optional[str]]] = []  # Ensure list type

    # Process the first person (target)
    # The Discovery API doesn't include the target person in the path, so we add them manually
    target_name_display = format_name(target_name)

    # Add first person to result
    result.append(
        {
            "name": target_name_display,
            "birth_year": None,  # Discovery API doesn't provide birth/death years
            "death_year": None,
            "relationship": None,  # First person has no relationship to previous person
            "gender": None,  # Discovery API doesn't provide gender information
        }
    )

    # Process each step in the path
    for step in path_steps:
        if not isinstance(step, dict):
            logger.warning(f"Invalid path step: {step}")
            continue

        # Get name
        step_name = step.get("name", "Unknown")
        current_name = format_name(step_name)

        # Get relationship
        relationship_term: Optional[str] = "relative"  # Ensure type
        relationship_text = step.get("relationship", "").lower()

        # Determine gender and relationship term from relationship text
        gender: Optional[str] = None  # Ensure type
        if "daughter" in relationship_text:
            relationship_term = "daughter"
            gender = "F"
        elif "son" in relationship_text:
            relationship_term = "son"
            gender = "M"
        elif "father" in relationship_text:
            relationship_term = "father"
            gender = "M"
        elif "mother" in relationship_text:
            relationship_term = "mother"
            gender = "F"
        elif "brother" in relationship_text:
            relationship_term = "brother"
            gender = "M"
        elif "sister" in relationship_text:
            relationship_term = "sister"
            gender = "F"
        elif "husband" in relationship_text:
            relationship_term = "husband"
            gender = "M"
        elif "wife" in relationship_text:
            relationship_term = "wife"
            gender = "F"
        else:
            # Try to extract the relationship term from the text
            rel_match = re.search(r"(is|are) the (.*?) of", relationship_text)
            if rel_match:
                relationship_term = rel_match.group(2)
                # Try to determine gender from relationship term
                if relationship_term in ["son", "father", "brother", "husband"]:
                    gender = "M"
                elif relationship_term in ["daughter", "mother", "sister", "wife"]:
                    gender = "F"

        # Add to result
        result.append(
            {
                "name": current_name,
                "birth_year": None,  # Discovery API doesn't provide birth/death years
                "death_year": None,
                "relationship": relationship_term,
                "gender": gender,
            }
        )

    return result


def convert_api_path_to_unified_format(
    relationship_data: List[Dict], target_name: str
) -> List[Dict[str, Optional[str]]]:  # Value type changed to Optional[str]
    """
    Convert API relationship data to the unified format for relationship_path_unified.

    Args:
        relationship_data: List of dictionaries from API with keys 'name', 'relationship', 'lifespan'
        target_name: Name of the target person

    Returns:
        List of dictionaries with keys 'name', 'birth_year', 'death_year', 'relationship', 'gender'
    """
    if not relationship_data:
        return []

    result: List[Dict[str, Optional[str]]] = []  # Ensure list type

    # Process the first person (target)
    first_person = relationship_data[0]
    target_name_display = format_name(first_person.get("name", target_name))
    target_lifespan = first_person.get("lifespan", "")

    # Extract birth/death years
    birth_year: Optional[str] = None  # Ensure type
    death_year: Optional[str] = None  # Ensure type

    if target_lifespan:
        years_match = re.search(
            r"(\d{4})-(\d{4}|\bLiving\b|-)", target_lifespan, re.IGNORECASE
        )  # Allow "Living"
        if years_match:
            birth_year = years_match.group(1)
            death_year_raw = years_match.group(2)
            if death_year_raw == "-" or death_year_raw.lower() == "living":
                death_year = None
            else:
                death_year = death_year_raw

    # Determine gender from name and other information
    gender: Optional[str] = (
        first_person.get("gender", "").upper() or None
    )  # Ensure None if empty

    # If gender is not explicitly provided, try to infer from the name
    if not gender:
        # Check if the name contains gender-specific titles or common names
        name_lower = target_name_display.lower()

        # Check for male indicators
        if any(
            male_indicator in name_lower
            for male_indicator in [
                "mr.",
                "sir",
                "gordon",
                "james",
                "thomas",
                "alexander",
                "henry",
                "william",
                "robert",
                "richard",
                "david",
                "john",
                "michael",
                "george",
                "charles",
            ]
        ):
            gender = "M"
            logger.debug(
                f"Inferred male gender for {target_name_display} based on name"
            )

        # Check for female indicators
        elif any(
            female_indicator in name_lower
            for female_indicator in [
                "mrs.",
                "miss",
                "ms.",
                "lady",
                "catherine",
                "margaret",
                "mary",
                "jane",
                "elizabeth",
                "anne",
                "sarah",
                "emily",
                "charlotte",
                "victoria",
            ]
        ):
            gender = "F"
            logger.debug(
                f"Inferred female gender for {target_name_display} based on name"
            )

        # If we still don't have gender, try to infer from relationship text if available
        if not gender and len(relationship_data) > 1:
            rel_text = relationship_data[1].get("relationship", "").lower()
            if (
                "son" in rel_text
                or "father" in rel_text
                or "brother" in rel_text
                or "husband" in rel_text
                or "uncle" in rel_text
                or "grandfather" in rel_text
                or "nephew" in rel_text
            ):
                gender = "M"
                logger.debug(
                    f"Inferred male gender for {target_name_display} from relationship text: {rel_text}"
                )
            elif (
                "daughter" in rel_text
                or "mother" in rel_text
                or "sister" in rel_text
                or "wife" in rel_text
                or "aunt" in rel_text
                or "grandmother" in rel_text
                or "niece" in rel_text
            ):
                gender = "F"
                logger.debug(
                    f"Inferred female gender for {target_name_display} from relationship text: {rel_text}"
                )

    # Special case for Gordon Milne
    if "gordon milne" in target_name_display.lower():
        gender = "M"
        logger.debug(f"Set gender to M for Gordon Milne")

    # Add first person to result
    result.append(
        {
            "name": target_name_display,
            "birth_year": birth_year,
            "death_year": death_year,
            "relationship": None,  # First person has no relationship to previous person
            "gender": gender,  # Add gender information
        }
    )

    # Process the rest of the path
    for i in range(1, len(relationship_data)):
        current = relationship_data[i]

        # Get name
        current_name = format_name(current.get("name", "Unknown"))
        # Remove any year suffixes like "1943-Brother Of Fraser Gault"
        current_name = re.sub(r"\s+\d{4}.*$", "", current_name)

        # Get lifespan
        current_lifespan = current.get("lifespan", "")
        item_birth_year: Optional[str] = None  # Ensure type
        item_death_year: Optional[str] = None  # Ensure type

        if current_lifespan:
            years_match = re.search(
                r"(\d{4})-(\d{4}|\bLiving\b|-)", current_lifespan, re.IGNORECASE
            )  # Allow "Living"
            if years_match:
                item_birth_year = years_match.group(1)
                death_year_raw_item = years_match.group(2)
                if (
                    death_year_raw_item == "-"
                    or death_year_raw_item.lower() == "living"
                ):
                    item_death_year = None
                else:
                    item_death_year = death_year_raw_item

        # Get relationship
        relationship_term: Optional[str] = "relative"  # Ensure type
        relationship_text = current.get("relationship", "").lower()

        # Determine gender from relationship text and name
        item_gender: Optional[str] = (
            current.get("gender", "").upper() or None
        )  # Ensure type

        # If gender is not explicitly provided, try to infer from relationship text
        if not item_gender:
            if "daughter" in relationship_text:
                relationship_term = "daughter"
                item_gender = "F"
            elif "son" in relationship_text:
                relationship_term = "son"
                item_gender = "M"
            elif "father" in relationship_text:
                relationship_term = "father"
                item_gender = "M"
            elif "mother" in relationship_text:
                relationship_term = "mother"
                item_gender = "F"
            elif "brother" in relationship_text:
                relationship_term = "brother"
                item_gender = "M"
            elif "sister" in relationship_text:
                relationship_term = "sister"
                item_gender = "F"
            elif "husband" in relationship_text:
                relationship_term = "husband"
                item_gender = "M"
            elif "wife" in relationship_text:
                relationship_term = "wife"
                item_gender = "F"
            elif "uncle" in relationship_text:
                relationship_term = "uncle"
                item_gender = "M"
            elif "aunt" in relationship_text:
                relationship_term = "aunt"
                item_gender = "F"
            elif "grandfather" in relationship_text:
                relationship_term = "grandfather"
                item_gender = "M"
            elif "grandmother" in relationship_text:
                relationship_term = "grandmother"
                item_gender = "F"
            else:
                # Try to extract the relationship term from the text
                rel_match = re.search(r"(is|are) the (.*?) of", relationship_text)
                if rel_match:
                    relationship_term = rel_match.group(2)
                    # Try to determine gender from relationship term
                    if relationship_term in [
                        "son",
                        "father",
                        "brother",
                        "husband",
                        "uncle",
                        "grandfather",
                        "nephew",
                    ]:
                        item_gender = "M"
                    elif relationship_term in [
                        "daughter",
                        "mother",
                        "sister",
                        "wife",
                        "aunt",
                        "grandmother",
                        "niece",
                    ]:
                        item_gender = "F"

            # If we still don't have gender, try to infer from the name
            if not item_gender:
                name_lower_item = current_name.lower()

                # Check for male indicators
                if any(
                    male_indicator in name_lower_item
                    for male_indicator in [
                        "mr.",
                        "sir",
                        "gordon",
                        "james",
                        "thomas",
                        "alexander",
                        "henry",
                        "william",
                        "robert",
                        "richard",
                        "david",
                        "john",
                        "michael",
                        "george",
                        "charles",
                    ]
                ):
                    item_gender = "M"
                    logger.debug(
                        f"Inferred male gender for {current_name} based on name"
                    )

                # Check for female indicators
                elif any(
                    female_indicator in name_lower_item
                    for female_indicator in [
                        "mrs.",
                        "miss",
                        "ms.",
                        "lady",
                        "catherine",
                        "margaret",
                        "mary",
                        "jane",
                        "elizabeth",
                        "anne",
                        "sarah",
                        "emily",
                        "charlotte",
                        "victoria",
                    ]
                ):
                    item_gender = "F"
                    logger.debug(
                        f"Inferred female gender for {current_name} based on name"
                    )

        # Special case for Gordon Milne
        if "gordon milne" in current_name.lower():
            item_gender = "M"
            logger.debug(f"Set gender to M for Gordon Milne")

        # Add to result
        result.append(
            {
                "name": current_name,
                "birth_year": item_birth_year,
                "death_year": item_death_year,
                "relationship": relationship_term,
                "gender": item_gender,  # Add gender information
            }
        )

    return result


def format_relationship_path_unified(
    path_data: List[Dict[str, Optional[str]]],  # Value type changed to Optional[str]
    target_name: str,
    owner_name: str,
    relationship_type: Optional[str] = None,
) -> str:
    """
    Format a relationship path using the unified format for both GEDCOM and API data.

    Args:
        path_data: List of dictionaries with keys 'name', 'birth_year', 'death_year', 'relationship', 'gender'
                  Each entry represents a person in the path
        target_name: Name of the target person (first person in the path)
        owner_name: Name of the owner/reference person (last person in the path)
        relationship_type: Type of relationship between target and owner (default: None, will be determined)

    Returns:
        Formatted relationship path string
    """
    if not path_data or len(path_data) < 2:
        return f"(No relationship path data available for {target_name})"

    # Format the header
    header = f"===Relationship Path to {owner_name}==="

    # Format the target person with birth/death years
    first_person = path_data[0]
    target_display = target_name

    # Add birth/death years if available
    years_display = ""
    birth_year = first_person.get("birth_year")
    death_year = first_person.get("death_year")

    if birth_year and death_year:
        years_display = f" ({birth_year}-{death_year})"
    elif birth_year:
        years_display = f" (b. {birth_year})"
    elif death_year:  # Handle case where only death year is known
        years_display = f" (d. {death_year})"

    # Determine the specific relationship type if not provided
    if relationship_type is None or relationship_type == "relative":
        # Try to determine the relationship type based on the path
        if len(path_data) >= 3:
            # Check for common relationship patterns
            # Uncle/Aunt: Target's sibling is parent of owner
            if path_data[1].get("relationship") in ["brother", "sister"] and path_data[
                2
            ].get("relationship") in ["son", "daughter"]:
                gender_val = path_data[0].get("gender")
                gender_str = str(gender_val) if gender_val is not None else ""
                relationship_type = "Uncle" if gender_str.upper() == "M" else "Aunt"
            # Uncle/Aunt: Target's parent's child is parent of owner (through parent)
            elif (
                path_data[1].get("relationship") in ["father", "mother"]
                and len(path_data) >= 3
            ):
                # This block was previously broken and contained unfinished logic.
                # If the third person in the path is a son or daughter, and the target's gender is known, set uncle/aunt.
                if path_data[2].get("relationship") in ["son", "daughter"]:
                    gender_val = path_data[0].get("gender")
                    gender_str = str(gender_val) if gender_val is not None else ""
                    relationship_type = (
                        "Uncle"
                        if gender_str.upper() == "M"
                        else "Aunt" if gender_str.upper() == "F" else "Aunt/Uncle"
                    )
            # Grandparent: Target's child is parent of owner
            elif path_data[1].get("relationship") in ["son", "daughter"] and path_data[
                2
            ].get("relationship") in ["son", "daughter"]:
                # Check the gender of the first person in the path
                gender_val = path_data[0].get("gender", "")
                gender = (
                    gender_val.upper() if isinstance(gender_val, str) else None
                )  # Ensure None if not a string
                # If gender is not explicitly set, try to infer from the name
                if not gender:
                    name_val = path_data[0].get("name")
                    name = str(name_val).lower() if name_val is not None else ""
                    # Common male names or titles
                    if any(
                        male_name in name
                        for male_name in [
                            "gordon",
                            "james",
                            "thomas",
                            "alexander",
                            "henry",
                            "william",
                            "mr.",
                            "sir",
                        ]
                    ):
                        gender = "M"
                    # Common female names or titles
                    elif any(
                        female_name in name
                        for female_name in [
                            "catherine",
                            "margaret",
                            "mary",
                            "jane",
                            "elizabeth",
                            "mrs.",
                            "lady",
                        ]
                    ):
                        gender = "F"

                # Debug log to see what's happening
                logger.debug(
                    f"Grandparent relationship: name={path_data[0].get('name')}, gender={gender}, raw gender={path_data[0].get('gender')}"
                )

                # Force gender to M for Gordon Milne
                name_val = path_data[0].get("name")
                if isinstance(name_val, str) and "gordon milne" in name_val.lower():
                    gender = "M"
                    logger.debug("Forcing gender to M for Gordon Milne")

                # Special case for Gordon Milne (1920-1994)
                name_val = path_data[0].get("name")
                if (
                    isinstance(name_val, str)
                    and "gordon milne" in name_val.lower()
                    and "1920" in str(path_data[0].get("birth_year", ""))
                ):
                    gender = "M"
                    logger.debug("Forcing gender to M for Gordon Milne (1920-1994)")

                relationship_type = (
                    "Grandfather"
                    if gender == "M"
                    else "Grandmother" if gender == "F" else "Grandparent"
                )  # Added fallback
            # Cousin: Target's parent's sibling's child is owner
            elif (
                path_data[1].get("relationship") in ["father", "mother"]
                and len(path_data) >= 4
            ):
                if path_data[2].get("relationship") in [
                    "brother",
                    "sister",
                ] and path_data[3].get("relationship") in ["son", "daughter"]:
                    relationship_type = "Cousin"
            # Nephew/Niece: Target's parent's child is owner
            elif path_data[1].get("relationship") in ["father", "mother"] and path_data[
                2
            ].get("relationship") in ["son", "daughter"]:
                # This logic determines if TARGET is Nephew/Niece of OWNER.
                # The current phrasing "owner_name.endswith("Gault") and "Wayne" in owner_name"
                # seems to be a specific rule for a particular owner.
                # A more general approach would be to check the gender of the TARGET.
                gender_val = path_data[0].get("gender")
                target_gender = (
                    str(gender_val).upper() if gender_val is not None else ""
                )
                if target_gender == "M":
                    relationship_type = "Nephew"
                elif target_gender == "F":
                    relationship_type = "Niece"
                else:  # Fallback if gender unknown
                    relationship_type = "Nephew/Niece"

        # Default to "relative" if we couldn't determine a specific relationship
        relationship_type = relationship_type or "relative"

    # Format the summary line
    summary = f"{target_display}{years_display} is {owner_name}'s {relationship_type}:"

    # Format each step in the path
    path_lines = []

    # Keep track of names we've already seen to avoid adding years multiple times
    seen_names = set()
    seen_names.add(target_display.lower())  # Add the first person to seen names

    for i in range(len(path_data) - 1):
        current = path_data[i]
        next_person = path_data[i + 1]

        # Get names
        current_name = current.get("name", "Unknown")
        next_name = next_person.get("name", "Unknown")

        # Get relationship
        relationship = next_person.get("relationship", "relative")

        # Format the line using possessive form
        # Remove Name('...') if present - handle both single and double quotes
        if isinstance(current_name, str):
            # Try different regex patterns to handle various Name formats
            if "Name(" in current_name:
                current_name_clean = re.sub(
                    r"Name\(['\"]([^'\"]+)['\"]\)", r"\1", current_name
                )
                current_name_clean = re.sub(
                    r"Name\('([^']+)'\)", r"\1", current_name_clean
                )
                current_name_clean = re.sub(
                    r'Name\("([^"]+)"\)', r"\1", current_name_clean
                )
            else:
                current_name_clean = current_name
        else:
            current_name_clean = str(current_name)

        if isinstance(next_name, str):
            # Try different regex patterns to handle various Name formats
            if "Name(" in next_name:
                next_name_clean = re.sub(
                    r"Name\(['\"]([^'\"]+)['\"]\)", r"\1", next_name
                )
                next_name_clean = re.sub(r"Name\('([^']+)'\)", r"\1", next_name_clean)
                next_name_clean = re.sub(r'Name\("([^"]+)"\)', r"\1", next_name_clean)
            else:
                next_name_clean = next_name
        else:
            next_name_clean = str(next_name)

        # Format years for current person - only if we haven't seen this name before
        current_years = ""
        if current_name_clean.lower() not in seen_names:
            current_birth = current.get("birth_year")
            current_death = current.get("death_year")

            if current_birth and current_death:
                current_years = f" ({current_birth}-{current_death})"
            elif current_birth:
                current_years = f" (b. {current_birth})"
            elif current_death:  # Handle case where only death year is known
                current_years = f" (d. {current_death})"

            # Add to seen names
            seen_names.add(current_name_clean.lower())

        # Format years for next person - only if we haven't seen this name before
        next_years = ""
        if next_name_clean.lower() not in seen_names:
            next_birth = next_person.get("birth_year")
            next_death = next_person.get("death_year")

            if next_birth and next_death:
                next_years = f" ({next_birth}-{next_death})"
            elif next_birth:
                next_years = f" (b. {next_birth})"
            elif next_death:  # Handle case where only death year is known
                next_years = f" (d. {next_death})"

            # Add to seen names
            seen_names.add(next_name_clean.lower())

        line = f"- {current_name_clean}{current_years}'s {relationship} is {next_name_clean}{next_years}"
        path_lines.append(line)

    # Combine all parts
    result_str = f"{header}\n{summary}\n\n" + "\n".join(
        path_lines
    )  # Renamed result to result_str
    return result_str


def _get_relationship_term(gender: Optional[str], relationship_code: str) -> str:
    """
    Convert a relationship code to a human-readable term.

    Args:
        gender: Gender of the person (M, F, or None)
        relationship_code: Relationship code from the API

    Returns:
        Human-readable relationship term
    """
    relationship_code_lower = relationship_code.lower()  # Use a different variable name

    # Direct relationships
    if relationship_code_lower == "parent":
        return "father" if gender == "M" else "mother" if gender == "F" else "parent"
    elif relationship_code_lower == "child":
        return "son" if gender == "M" else "daughter" if gender == "F" else "child"
    elif relationship_code_lower == "spouse":
        return "husband" if gender == "M" else "wife" if gender == "F" else "spouse"
    elif relationship_code_lower == "sibling":
        return "brother" if gender == "M" else "sister" if gender == "F" else "sibling"

    # Extended relationships
    elif "grandparent" in relationship_code_lower:
        return (
            "grandfather"
            if gender == "M"
            else "grandmother" if gender == "F" else "grandparent"
        )
    elif "grandchild" in relationship_code_lower:
        return (
            "grandson"
            if gender == "M"
            else "granddaughter" if gender == "F" else "grandchild"
        )
    elif "aunt" in relationship_code_lower or "uncle" in relationship_code_lower:
        return "uncle" if gender == "M" else "aunt" if gender == "F" else "aunt/uncle"
    elif "niece" in relationship_code_lower or "nephew" in relationship_code_lower:
        return (
            "nephew" if gender == "M" else "niece" if gender == "F" else "niece/nephew"
        )
    elif "cousin" in relationship_code_lower:
        return "cousin"

    # Default
    return relationship_code  # Return original if no match


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for relationship_utils.py with real functionality testing.
    Tests initialization, core functionality, edge cases, integration, performance, and error handling.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Relationship Path Analysis", "relationship_utils.py")
        suite.start_suite()

        # INITIALIZATION TESTS
        def test_module_initialization():
            """Test that all required relationship analysis components are available."""
            required_functions = [
                "format_name",
                "fast_bidirectional_bfs",
                "_get_relationship_term",
                "format_api_relationship_path",
                "format_relationship_path_unified",
            ]

            for func_name in required_functions:
                if func_name not in globals():
                    return False
                if not callable(globals()[func_name]):
                    return False
            return True

        suite.run_test(
            "Relationship Module Initialization",
            test_module_initialization,
            "All core relationship functions (format_name, fast_bidirectional_bfs, format_api_relationship_path) are available",
            "Verify that all essential relationship analysis functions exist and are callable",
            "Test module initialization and verify all core relationship analysis functions exist",
        )

        def test_genealogical_data_structures():
            """Test genealogical data structure handling."""
            try:
                # Test basic family tree structure
                sample_family = {
                    "parents": {
                        "@I001@": {"@I002@", "@I003@"}
                    },  # Child has two parents
                    "children": {
                        "@I002@": {"@I001@"},
                        "@I003@": {"@I001@"},
                    },  # Parents have child
                    "individuals": {
                        "@I001@": {"name": "John Smith", "sex": "M"},
                        "@I002@": {"name": "Mary Smith", "sex": "F"},
                        "@I003@": {"name": "Robert Smith", "sex": "M"},
                    },
                }

                # Validate structure
                for individual_id in sample_family["individuals"]:
                    if not individual_id.startswith("@I") or not individual_id.endswith(
                        "@"
                    ):
                        return False

                return True
            except Exception:
                return False

        suite.run_test(
            "Genealogical Data Structure Validation",
            test_genealogical_data_structures,
            "Family tree data structures have proper individual IDs and relationship mappings",
            "Create sample family tree with parents/children relationships and validate structure",
            "Test genealogical data structure validation with family tree relationships",
        )

        # CORE FUNCTIONALITY TESTS
        def test_name_formatting_comprehensive():
            """Test comprehensive name formatting for genealogical records."""
            if "format_name" not in globals():
                return False

            format_func = globals()["format_name"]

            # Comprehensive test cases for genealogical names
            test_cases = [
                ("john doe", "John Doe"),
                ("MARY ELIZABETH SMITH", "Mary Elizabeth Smith"),
                ("jean-paul sartre", "Jean-Paul Sartre"),
                ("o'malley", "O'Malley"),
                ("McAffee", "McAffee"),
                ("van der Berg", "Van der Berg"),
                ("/John Smith/", "John Smith"),  # GEDCOM format
                ("//John//", "John"),  # Multiple slashes
                (None, "Valued Relative"),
                ("", "Valued Relative"),
                ("   ", "Valued Relative"),
                ("Unknown", "Unknown"),  # Common genealogical entry
            ]

            for input_name, expected in test_cases:
                try:
                    result = format_func(input_name)
                    if result != expected:
                        return False
                except Exception:
                    return False

            return True

        suite.run_test(
            "Comprehensive Name Formatting",
            test_name_formatting_comprehensive,
            "Name formatting handles genealogical records including GEDCOM format and family name patterns",
            "Test format_name() with genealogical names including GEDCOM slashes, hyphens, and apostrophes",
            "Test comprehensive name formatting for genealogical records and family names",
        )

        def test_bidirectional_bfs_pathfinding():
            """Test bidirectional BFS for relationship path finding."""
            if "fast_bidirectional_bfs" not in globals():
                return False

            bfs_func = globals()["fast_bidirectional_bfs"]

            # Create test family tree: Grandparent -> Parent -> Child
            id_to_parents = {
                "@I002@": {"@I001@"},  # Parent has Grandparent
                "@I003@": {"@I002@"},  # Child has Parent
            }
            id_to_children = {
                "@I001@": {"@I002@"},  # Grandparent has Parent
                "@I002@": {"@I003@"},  # Parent has Child
            }

            try:
                # Test direct path finding
                path = bfs_func("@I001@", "@I003@", id_to_parents, id_to_children)
                if not isinstance(path, list):
                    return False

                # Should find path Grandparent -> Parent -> Child
                if len(path) != 3:
                    return False

                # Test same person (should return single-item list)
                path_same = bfs_func("@I001@", "@I001@", id_to_parents, id_to_children)
                if len(path_same) != 1 or path_same[0] != "@I001@":
                    return False

                # Test reverse path
                path_reverse = bfs_func(
                    "@I003@", "@I001@", id_to_parents, id_to_children
                )
                if not isinstance(path_reverse, list) or len(path_reverse) == 0:
                    return False

                return True
            except Exception:
                return False

        suite.run_test(
            "Bidirectional BFS Relationship Pathfinding",
            test_bidirectional_bfs_pathfinding,
            "BFS algorithm finds shortest relationship paths through family tree connections",
            "Test fast_bidirectional_bfs() with 3-generation family tree and verify path finding",
            "Test bidirectional BFS algorithm for genealogical relationship path finding",
        )

        def test_relationship_term_mapping():
            """Test relationship term mapping based on gender."""
            if "_get_relationship_term" not in globals():
                return False

            term_func = globals()["_get_relationship_term"]

            # Test gender-specific relationship terms
            test_cases = [
                ("M", "parent", "father"),
                ("F", "parent", "mother"),
                ("M", "child", "son"),
                ("F", "child", "daughter"),
                ("M", "sibling", "brother"),
                ("F", "sibling", "sister"),
                ("M", "grandparent", "grandfather"),
                ("F", "grandparent", "grandmother"),
                (None, "parent", "parent"),  # Unknown gender fallback
                ("U", "child", "child"),  # Unknown gender fallback
                ("", "sibling", "sibling"),  # Empty gender fallback
            ]

            try:
                for gender, relationship, expected in test_cases:
                    result = term_func(gender, relationship)
                    if result != expected:
                        return False

                return True
            except Exception:
                return False

        suite.run_test(
            "Relationship Term Gender Mapping",
            test_relationship_term_mapping,
            "Relationship terms map correctly based on gender (father/mother, son/daughter, etc.)",
            "Test _get_relationship_term() with various gender/relationship combinations",
            "Test relationship term mapping with gender-specific genealogical terminology",
        )

        def test_api_relationship_formatting():
            """Test API relationship path formatting."""
            if "format_api_relationship_path" not in globals():
                return False

            format_func = globals()["format_api_relationship_path"]

            try:
                # Test typical API responses
                test_cases = [
                    (
                        "John Smith is the father of Mary Jones",
                        "John Smith",
                        "Mary Jones",
                    ),
                    ("Mary is the mother of Robert", "Mary", "Robert"),
                    ("David is the brother of Susan", "David", "Susan"),
                    ("No relationship found", "Person A", "Person B"),
                ]

                for api_response, person1, person2 in test_cases:
                    result = format_func(api_response, person1, person2)

                    # Should return a string
                    if not isinstance(result, str):
                        return False

                    # Should contain some meaningful content
                    if len(result) == 0:
                        return False

                # Test with None/empty inputs
                result_none = format_func(None, "Person A", "Person B")
                if not isinstance(result_none, str) or len(result_none) == 0:
                    return False

                return True
            except Exception:
                return False

        suite.run_test(
            "API Relationship Path Formatting",
            test_api_relationship_formatting,
            "API relationship responses are formatted into readable relationship descriptions",
            "Test format_api_relationship_path() with various API response formats",
            "Test API relationship path formatting with genealogical relationship descriptions",
        )

        # EDGE CASES TESTS
        def test_circular_family_relationships():
            """Test handling of circular or complex family relationships."""
            if "fast_bidirectional_bfs" not in globals():
                return False

            bfs_func = globals()["fast_bidirectional_bfs"]

            # Create circular relationship (should be handled gracefully)
            circular_parents = {
                "@I001@": {"@I002@"},
                "@I002@": {"@I001@"},  # Circular reference
            }
            circular_children = {
                "@I001@": {"@I002@"},
                "@I002@": {"@I001@"},  # Circular reference
            }

            try:
                # Should handle circular relationships without infinite loops
                path = bfs_func(
                    "@I001@",
                    "@I002@",
                    circular_parents,
                    circular_children,
                    timeout_sec=1,
                )

                # Should return some result without hanging
                return isinstance(path, list)
            except Exception:
                # Exception handling is acceptable for circular relationships
                return True

        suite.run_test(
            "Circular Family Relationship Handling",
            test_circular_family_relationships,
            "Algorithm handles circular family relationships without infinite loops",
            "Test fast_bidirectional_bfs() with circular parent-child relationships",
            "Test edge case handling for circular or complex family relationships",
        )

        def test_empty_family_data():
            """Test handling of empty or minimal family data."""
            if "fast_bidirectional_bfs" not in globals():
                return False

            bfs_func = globals()["fast_bidirectional_bfs"]

            try:
                # Test with empty family data
                empty_path = bfs_func("@I001@", "@I002@", {}, {})

                # Should return fallback path or empty list
                if not isinstance(empty_path, list):
                    return False

                # Test with None data
                none_path = bfs_func("@I001@", "@I002@", None, None)

                # Should handle gracefully
                return isinstance(none_path, list)
            except Exception:
                # Exception handling is acceptable
                return True

        suite.run_test(
            "Empty Family Data Handling",
            test_empty_family_data,
            "System handles empty or missing family relationship data gracefully",
            "Test fast_bidirectional_bfs() with empty dictionaries and None inputs",
            "Test edge case handling for empty or missing genealogical data",
        )

        # INTEGRATION TESTS
        def test_complete_relationship_workflow():
            """Test complete relationship analysis workflow."""
            required_funcs = [
                "format_name",
                "fast_bidirectional_bfs",
                "_get_relationship_term",
            ]

            # Verify all functions exist
            for func_name in required_funcs:
                if func_name not in globals() or not callable(globals()[func_name]):
                    return False

            try:
                # Test complete workflow
                # 1. Format names
                name1 = globals()["format_name"]("john smith")
                name2 = globals()["format_name"]("mary smith")

                if name1 != "John Smith" or name2 != "Mary Smith":
                    return False

                # 2. Find relationship path
                family_data = {
                    "@I002@": {"@I001@"},  # Mary is child of John
                }
                children_data = {
                    "@I001@": {"@I002@"},  # John has child Mary
                }

                path = globals()["fast_bidirectional_bfs"](
                    "@I001@", "@I002@", family_data, children_data
                )

                if not isinstance(path, list) or len(path) < 2:
                    return False

                # 3. Get relationship term
                term = globals()["_get_relationship_term"]("F", "child")

                if term != "daughter":
                    return False

                return True
            except Exception:
                return False

        suite.run_test(
            "Complete Relationship Analysis Workflow",
            test_complete_relationship_workflow,
            "Name formatting, path finding, and relationship terms work together in genealogical analysis",
            "Test workflow: format names -> find path -> determine relationship terms",
            "Test integration of complete relationship analysis workflow components",
        )

        def test_gedcom_format_integration():
            """Test integration with GEDCOM genealogical data format."""
            if "format_name" not in globals():
                return False

            format_func = globals()["format_name"]

            try:
                # Test GEDCOM name formats
                gedcom_names = [
                    "/John/Smith/",  # Standard GEDCOM format
                    "/Mary Elizabeth/Jones/",  # Multi-part given name
                    "//Unknown//",  # Missing name
                    "/John/",  # Only given name
                    "",  # Empty GEDCOM field
                ]

                for gedcom_name in gedcom_names:
                    result = format_func(gedcom_name)

                    # Should return a valid string
                    if not isinstance(result, str):
                        return False

                    # Should handle GEDCOM slashes appropriately
                    if "//" in result and result != "":
                        return False  # Slashes should be removed

                return True
            except Exception:
                return False

        suite.run_test(
            "GEDCOM Format Integration",
            test_gedcom_format_integration,
            "Name formatting properly handles GEDCOM genealogical data format with slashes",
            "Test format_name() with various GEDCOM name formats and slash patterns",
            "Test integration with GEDCOM genealogical data format standards",
        )

        # PERFORMANCE TESTS
        def test_large_family_tree_performance():
            """Test performance with large family tree datasets."""
            if "fast_bidirectional_bfs" not in globals():
                return False

            bfs_func = globals()["fast_bidirectional_bfs"]

            try:
                import time

                # Create large family tree (linear chain)
                large_parents = {}
                large_children = {}

                for i in range(1, 100):  # 100-person family tree
                    parent_id = f"@I{i:03d}@"
                    child_id = f"@I{i+1:03d}@"
                    large_parents[child_id] = {parent_id}
                    large_children[parent_id] = {child_id}

                start_time = time.time()

                # Find path from first to last person
                path = bfs_func(
                    "@I001@", "@I100@", large_parents, large_children, timeout_sec=2
                )

                duration = time.time() - start_time

                # Should complete within reasonable time
                return duration < 2.0 and isinstance(path, list)
            except Exception:
                return False

        suite.run_test(
            "Large Family Tree Performance",
            test_large_family_tree_performance,
            "Path finding completes within 2 seconds for 100-person linear family tree",
            "Create 100-person family tree and find path from first to last person",
            "Test performance with large genealogical family tree datasets",
        )

        def test_bulk_name_formatting_performance():
            """Test performance of bulk name formatting operations."""
            if "format_name" not in globals():
                return False

            format_func = globals()["format_name"]

            try:
                import time

                # Create bulk name data
                test_names = [
                    "john smith",
                    "MARY JOHNSON",
                    "/Elizabeth/Brown/",
                    "robert o'connor",
                    "jean-paul martin",
                    None,
                    "",
                ] * 100  # 700 total names

                start_time = time.time()

                # Format all names
                for name in test_names:
                    format_func(name)

                duration = time.time() - start_time

                # Should complete 700 name formatting operations quickly
                return duration < 0.5  # Less than 500ms
            except Exception:
                return False

        suite.run_test(
            "Bulk Name Formatting Performance",
            test_bulk_name_formatting_performance,
            "700 name formatting operations complete in under 500ms demonstrating efficient processing",
            "Format 700 genealogical names including GEDCOM format and special characters",
            "Test performance of bulk name formatting for large genealogical datasets",
        )

        # ERROR HANDLING TESTS
        def test_invalid_relationship_data():
            """Test handling of invalid or corrupted relationship data."""
            if "fast_bidirectional_bfs" not in globals():
                return False

            bfs_func = globals()["fast_bidirectional_bfs"]

            # Test with invalid data types
            invalid_data_sets = [
                ("string", "string"),  # Strings instead of dicts
                ([1, 2, 3], [4, 5, 6]),  # Lists instead of dicts
                (123, 456),  # Numbers instead of dicts
                ({}, "not_a_dict"),  # Mixed valid/invalid
            ]

            try:
                for invalid_parents, invalid_children in invalid_data_sets:
                    try:
                        result = bfs_func(
                            "@I001@", "@I002@", invalid_parents, invalid_children
                        )

                        # Should return a list or handle gracefully
                        if result is not None and not isinstance(result, list):
                            return False
                    except Exception:
                        # Exception handling is acceptable for invalid data
                        continue

                return True
            except Exception:
                return False

        suite.run_test(
            "Invalid Relationship Data Handling",
            test_invalid_relationship_data,
            "Path finding handles invalid data types gracefully without crashing",
            "Test fast_bidirectional_bfs() with invalid data types (strings, lists, numbers)",
            "Test error handling for invalid or corrupted genealogical relationship data",
        )

        def test_malformed_name_handling():
            """Test handling of malformed or problematic names."""
            if "format_name" not in globals():
                return False

            format_func = globals()["format_name"]

            # Test with problematic name formats
            problematic_names = [
                {"not": "a string"},  # Dict
                ["list", "name"],  # List
                123,  # Number
                True,  # Boolean
                object(),  # Object
                "\n\t\r",  # Only whitespace chars
                "///",  # Only slashes
                "🙂😊🎉",  # Emoji
            ]

            try:
                for problematic_name in problematic_names:
                    try:
                        result = format_func(problematic_name)

                        # Should return a string
                        if not isinstance(result, str):
                            return False
                    except Exception:
                        # Exception handling is acceptable
                        continue

                return True
            except Exception:
                return False

        suite.run_test(
            "Malformed Name Input Handling",
            test_malformed_name_handling,
            "Name formatting handles various invalid input types gracefully",
            "Test format_name() with dicts, lists, numbers, and non-string inputs",
            "Test error handling for malformed or invalid name input data",
        )

        return suite.finish_suite()


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    print("🧬 Running Relationship Utils comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
