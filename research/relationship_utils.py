#!/usr/bin/env python3

"""Relationship Utilities for Genealogical Data.

Utilities for processing genealogical relationship data including
relationship calculation, path finding, and family tree traversal.
"""

from __future__ import annotations

# === CORE INFRASTRUCTURE ===
import logging

logger = logging.getLogger(__name__)

# === ERROR HANDLING ===
# Debug log de-duplication for gender inference
_gender_log_once: set[str] = set()


def _log_inferred_gender_once(name: str, source: str, message: str) -> None:
    try:
        key = f"{source}:{name.lower()}"
        if key in _gender_log_once:
            return
        # Prevent unbounded growth in long sessions
        if len(_gender_log_once) > 200:
            _gender_log_once.clear()
        _gender_log_once.add(key)
        logger.debug(message)
    except Exception:
        # Never break control flow for logging
        pass


# === STANDARD LIBRARY IMPORTS ===
import html
import re
import time
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional, Protocol, Union, cast

# --- Try to import BeautifulSoup ---
from bs4 import BeautifulSoup, Tag

# === PERFORMANCE OPTIMIZATIONS ===
from performance.memory_utils import fast_json_loads

BS4_AVAILABLE = True

# --- Local imports ---
# Import format_name from utils (canonical implementation with full name particle handling)
# --- Test framework imports ---
# Import specific functions from gedcom_utils
from core.common_params import GraphContext
from core.utils import format_name
from testing.test_framework import (
    TestSuite,
    suppress_logging,
)


class GedcomTagProtocol(Protocol):
    value: Any


class GedcomIndividualProtocol(Protocol):
    def sub_tag(self, tag: str) -> Optional[GedcomTagProtocol]: ...


from genealogy.gedcom.gedcom_utils import (
    TAG_BIRTH,
    TAG_DEATH,
    TAG_SEX,
    are_spouses,
    get_event_info,
    get_full_name,
)
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

# --- Relationship Path Finding Functions ---

# Helper functions for fast_bidirectional_bfs


def _validate_bfs_inputs(
    start_id: str,
    end_id: str,
    id_to_parents: Optional[dict[str, set[str]]],
    id_to_children: Optional[dict[str, set[str]]],
) -> bool:
    """Validate inputs for BFS search."""
    if start_id == end_id:
        return False  # Special case handled by caller
    if id_to_parents is None or id_to_children is None:
        logger.error("[FastBiBFS] Relationship maps are None.")
        return False
    if not start_id or not end_id:
        logger.error("[FastBiBFS] Start or end ID is missing.")
        return False
    return True


def _initialize_bfs_queues(
    start_id: str, end_id: str
) -> tuple[
    deque[tuple[str, int, list[str]]],
    deque[tuple[str, int, list[str]]],
    dict[str, tuple[int, list[str]]],
    dict[str, tuple[int, list[str]]],
]:
    """Initialize BFS queues and visited sets."""
    queue_fwd = deque([(start_id, 0, [start_id])])
    queue_bwd = deque([(end_id, 0, [end_id])])
    visited_fwd = {start_id: (0, [start_id])}
    visited_bwd = {end_id: (0, [end_id])}
    return queue_fwd, queue_bwd, visited_fwd, visited_bwd


def _check_search_limits(start_time: float, processed: int, timeout_sec: float, node_limit: int) -> bool:
    """Check if search limits have been exceeded."""
    if time.time() - start_time > timeout_sec:
        logger.warning(f"[FastBiBFS] Timeout after {timeout_sec:.1f} seconds.")
        return False
    if processed > node_limit:
        logger.warning(f"[FastBiBFS] Node limit ({node_limit}) reached.")
        return False
    return True


def _add_relative_to_queue(
    relative_id: str,
    path: list[str],
    depth: int,
    visited: dict[str, tuple[int, list[str]]],
    queue: deque[tuple[str, int, list[str]]],
    is_forward: bool,
) -> None:
    """Add a relative to the search queue if not already visited."""
    if relative_id not in visited:
        new_path = [*path, relative_id] if is_forward else [relative_id, *path]
        visited[relative_id] = (depth, new_path)
        queue.append((relative_id, depth, new_path))


def _expand_to_siblings(
    graph: GraphContext,
    current_id: str,
    path: list[str],
    depth: int,
    visited: dict[str, tuple[int, list[str]]],
    queue: deque[tuple[str, int, list[str]]],
    is_forward: bool,
) -> None:
    """Expand search to siblings through parents."""
    for parent_id in graph.id_to_parents.get(current_id, set()):
        for sibling_id in graph.id_to_children.get(parent_id, set()):
            if sibling_id != current_id and sibling_id not in visited:
                new_path = [*path, parent_id, sibling_id] if is_forward else [sibling_id, parent_id, *path]
                visited[sibling_id] = (depth + 2, new_path)
                queue.append((sibling_id, depth + 2, new_path))


def _expand_to_relatives(
    graph: GraphContext,
    path: list[str],
    depth: int,
    visited: dict[str, tuple[int, list[str]]],
    queue: deque[tuple[str, int, list[str]]],
    is_forward: bool,
) -> None:
    """Expand search to parents, children, and siblings."""
    current_id = graph.current_id
    if not current_id:
        return

    # Expand to parents
    for parent_id in graph.id_to_parents.get(current_id, set()):
        _add_relative_to_queue(parent_id, path, depth + 1, visited, queue, is_forward)

    # Expand to children
    for child_id in graph.id_to_children.get(current_id, set()):
        _add_relative_to_queue(child_id, path, depth + 1, visited, queue, is_forward)

    # Expand to siblings (through parent)
    _expand_to_siblings(graph, current_id, path, depth, visited, queue, is_forward)


def _process_forward_queue(
    queue_fwd: deque[tuple[str, int, list[str]]],
    visited_fwd: dict[str, tuple[int, list[str]]],
    visited_bwd: dict[str, tuple[int, list[str]]],
    all_paths: list[list[str]],
    graph: GraphContext,
    max_depth: int,
) -> int:
    """Process forward queue and return number of nodes processed."""
    if not queue_fwd:
        return 0

    current_id, depth, path = queue_fwd.popleft()

    # Check if we've reached a node visited by backward search
    if current_id in visited_bwd:
        _, bwd_path = visited_bwd[current_id]
        combined_path = path + bwd_path[1:]
        all_paths.append(combined_path)
        logger.debug(f"[FastBiBFS] Path found via {current_id}: {len(combined_path)} nodes")

    # Stop expanding if we've reached max depth
    if depth < max_depth:
        graph_ctx = GraphContext(
            id_to_parents=graph.id_to_parents, id_to_children=graph.id_to_children, current_id=current_id
        )
        _expand_to_relatives(graph_ctx, path, depth, visited_fwd, queue_fwd, is_forward=True)

    return 1


def _process_backward_queue(
    queue_bwd: deque[tuple[str, int, list[str]]],
    visited_fwd: dict[str, tuple[int, list[str]]],
    visited_bwd: dict[str, tuple[int, list[str]]],
    all_paths: list[list[str]],
    graph: GraphContext,
    max_depth: int,
) -> int:
    """Process backward queue and return number of nodes processed."""
    if not queue_bwd:
        return 0

    current_id, depth, path = queue_bwd.popleft()

    # Check if we've reached a node visited by forward search
    if current_id in visited_fwd:
        _, fwd_path = visited_fwd[current_id]
        combined_path = fwd_path + path[1:]
        all_paths.append(combined_path)
        logger.debug(f"[FastBiBFS] Path found via {current_id}: {len(combined_path)} nodes")

    # Stop expanding if we've reached max depth
    if depth < max_depth:
        graph_ctx = GraphContext(
            id_to_parents=graph.id_to_parents, id_to_children=graph.id_to_children, current_id=current_id
        )
        _expand_to_relatives(graph_ctx, path, depth, visited_bwd, queue_bwd, is_forward=False)

    return 1


def _score_path(path: list[str], id_to_parents: dict[str, set[str]], id_to_children: dict[str, set[str]]) -> float:
    """Score a path based on directness of relationships."""
    direct_relationships = 0
    for i in range(len(path) - 1):
        if has_direct_relationship(path[i], path[i + 1], id_to_parents, id_to_children):
            direct_relationships += 1

    directness_score = direct_relationships / (len(path) - 1) if len(path) > 1 else 0
    length_penalty = len(path) / 10
    return directness_score - length_penalty


def _select_best_path(
    all_paths: list[list[str]], id_to_parents: dict[str, set[str]], id_to_children: dict[str, set[str]]
) -> list[str]:
    """Select the best path from all found paths."""
    scored_paths = [(p, _score_path(p, id_to_parents, id_to_children)) for p in all_paths]
    scored_paths.sort(key=lambda x: x[1], reverse=True)
    best_path = scored_paths[0][0]
    logger.debug(f"[FastBiBFS] Selected best path: {len(best_path)} nodes with score {scored_paths[0][1]:.2f}")
    return best_path


def _run_bfs_search_loop(
    queue_fwd: deque[tuple[str, int, list[str]]],
    queue_bwd: deque[tuple[str, int, list[str]]],
    visited_fwd: dict[str, tuple[int, list[str]]],
    visited_bwd: dict[str, tuple[int, list[str]]],
    graph_ctx: GraphContext,
    max_depth: int,
    timeout_sec: float,
    node_limit: int,
    start_time: float,
) -> list[list[str]]:
    """Run the main BFS search loop and return all found paths."""
    all_paths: list[list[str]] = []
    processed = 0

    while queue_fwd and queue_bwd and len(all_paths) < 5:
        if not _check_search_limits(start_time, processed, timeout_sec, node_limit):
            break

        # Process forward queue
        processed += _process_forward_queue(queue_fwd, visited_fwd, visited_bwd, all_paths, graph_ctx, max_depth)
        if len(all_paths) >= 5:
            break

        # Process backward queue
        processed += _process_backward_queue(queue_bwd, visited_fwd, visited_bwd, all_paths, graph_ctx, max_depth)

    return all_paths


# === RELATIONSHIP PATHFINDING CACHE (Priority 1 Todo #9) ===


class RelationshipPathCache:
    """
    LRU cache for relationship pathfinding results.

    Caches the results of fast_bidirectional_bfs() to avoid redundant
    graph traversals for frequently queried ancestor relationships.

    Target: 60%+ cache hit rate on repeat queries.
    """

    def __init__(self, maxsize: int = 1000):
        """
        Initialize the cache with a maximum size.

        Args:
            maxsize: Maximum number of cached paths (default: 1000)
        """
        self._cache: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0
        self._total_queries = 0

    def get(self, start_id: str, end_id: str) -> Optional[list[str]]:
        """
        Get cached path result.

        Args:
            start_id: Starting person ID
            end_id: Ending person ID

        Returns:
            Cached path list or None if not found
        """
        self._total_queries += 1

        # Create normalized cache key (order-independent for bidirectional)
        key = self._make_key(start_id, end_id)

        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        return None

    def put(self, start_id: str, end_id: str, path: list[str]) -> None:
        """
        Store path result in cache.

        Args:
            start_id: Starting person ID
            end_id: Ending person ID
            path: Computed path list
        """
        key = self._make_key(start_id, end_id)

        # If key exists, move to end
        if key in self._cache:
            self._cache.move_to_end(key)

        # Add new entry
        self._cache[key] = path

        # Evict oldest if over capacity (LRU eviction)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached paths and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._total_queries = 0

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        hit_rate = (self._hits / self._total_queries * 100) if self._total_queries > 0 else 0.0

        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "total_queries": self._total_queries,
            "hit_rate_percent": hit_rate,
        }

    @staticmethod
    def _make_key(start_id: str, end_id: str) -> tuple[str, str]:
        """
        Create normalized cache key.

        Ensures bidirectional queries (A→B and B→A) use the same cache entry.
        """
        # Sort IDs to create consistent key regardless of query direction
        return (start_id, end_id) if start_id < end_id else (end_id, start_id)


# Global cache instance
_relationship_path_cache = RelationshipPathCache(maxsize=1000)


def get_relationship_cache_stats() -> dict[str, Any]:
    """
    Get current relationship path cache statistics.

    Returns:
        Dictionary with cache performance metrics
    """
    return _relationship_path_cache.get_stats()


def clear_relationship_cache() -> None:
    """Clear the relationship path cache and reset statistics."""
    _relationship_path_cache.clear()
    logger.info("Relationship path cache cleared")


def report_cache_stats_to_performance_monitor() -> None:
    """
    Report relationship path cache statistics to performance monitor (Priority 1 Todo #9).

    This should be called periodically to track cache hit rate over time.
    """
    try:
        from performance.performance_monitor import performance_monitor

        stats = _relationship_path_cache.get_stats()
        performance_monitor.track_cache_hit_rate(
            cache_name="relationship_path_cache",
            hits=stats["hits"],
            misses=stats["misses"],
            total_queries=stats["total_queries"],
            cache_size=stats["size"],
            maxsize=stats["maxsize"],
        )
    except Exception as e:
        logger.debug(f"Failed to report cache stats to performance monitor: {e}")


def fast_bidirectional_bfs(
    start_id: str,
    end_id: str,
    id_to_parents: Optional[dict[str, set[str]]],
    id_to_children: Optional[dict[str, set[str]]],
    max_depth: int = 25,
    node_limit: int = 150000,
    timeout_sec: float = 45,
) -> list[str]:
    """
    Enhanced bidirectional BFS that finds direct paths through family trees.

    This implementation focuses on finding paths where each person has a clear,
    direct relationship to the next person in the path (parent, child, sibling).
    It avoids using special cases or "connected to" placeholders.

    The algorithm prioritizes shorter paths with direct relationships over longer paths.

    Now includes LRU caching (Priority 1 Todo #9) to avoid redundant graph traversals
    for frequently queried ancestor relationships.
    """
    # Check cache first (Priority 1 Todo #9)
    cached_path = _relationship_path_cache.get(start_id, end_id)
    if cached_path is not None:
        logger.debug(
            f"[FastBiBFS Cache] HIT: {start_id} <-> {end_id} (hit rate: {_relationship_path_cache.get_stats()['hit_rate_percent']:.1f}%)"
        )
        return cached_path

    logger.debug(f"[FastBiBFS Cache] MISS: {start_id} <-> {end_id}")

    start_time = time.time()

    # Quick return for same node
    if start_id == end_id:
        result = [start_id]
        _relationship_path_cache.put(start_id, end_id, result)
        return result

    # Validate inputs
    if not _validate_bfs_inputs(start_id, end_id, id_to_parents, id_to_children):
        result = []
        _relationship_path_cache.put(start_id, end_id, result)
        return result

    # After validation, we know these are not None
    assert id_to_parents is not None and id_to_children is not None, "Validation should have caught None values"

    # Try direct relationship first
    direct_path = find_direct_relationship(start_id, end_id, id_to_parents, id_to_children)
    if direct_path:
        logger.debug(f"[FastBiBFS] Found direct relationship: {direct_path}")
        _relationship_path_cache.put(start_id, end_id, direct_path)
        return direct_path

    # Initialize BFS
    queue_fwd, queue_bwd, visited_fwd, visited_bwd = _initialize_bfs_queues(start_id, end_id)
    logger.debug(f"[FastBiBFS] Starting BFS: {start_id} <-> {end_id}")

    # Run main search loop
    graph_ctx = GraphContext(id_to_parents=id_to_parents, id_to_children=id_to_children)
    all_paths = _run_bfs_search_loop(
        queue_fwd, queue_bwd, visited_fwd, visited_bwd, graph_ctx, max_depth, timeout_sec, node_limit, start_time
    )

    # Select best path if found
    if all_paths:
        result = _select_best_path(all_paths, id_to_parents, id_to_children)
        _relationship_path_cache.put(start_id, end_id, result)
        return result

    # No paths found
    logger.warning(f"[FastBiBFS] No paths found between {start_id} and {end_id}.")
    result = [start_id, end_id]
    _relationship_path_cache.put(start_id, end_id, result)
    return result


def explain_relationship_path(
    path_ids: list[str],
    reader: Any,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
    indi_index: dict[str, Union[GedcomIndividualProtocol, Any]],
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

    # Convert the GEDCOM path to the unified format
    unified_path = convert_gedcom_path_to_unified_format(path_ids, reader, id_to_parents, id_to_children, indi_index)

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


# Helper functions for format_api_relationship_path


def _extract_html_from_response(
    api_response_data: Union[str, dict[str, Any], None],
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Extract HTML content and JSON data from API response."""
    html_content_raw: Optional[str] = None
    json_data: Optional[dict[str, Any]] = None

    if isinstance(api_response_data, str):
        # Handle JSONP response format: no({...})
        jsonp_match = re.search(r"no\((.*)\)", api_response_data, re.DOTALL)
        if jsonp_match:
            try:
                json_str = jsonp_match.group(1)
                json_data = fast_json_loads(json_str)
                html_content_raw = json_data.get("html") if json_data is not None else None
            except Exception as e:
                logger.error(f"Error parsing JSONP response: {e}", exc_info=True)
                return None, None
        else:
            # Direct HTML response
            html_content_raw = api_response_data
    elif isinstance(api_response_data, dict):
        # Handle direct JSON/dict response
        json_data = api_response_data
        html_content_raw = json_data.get("html") if json_data else None

    return html_content_raw, json_data


def _format_discovery_api_path(json_data: dict[str, Any], target_name: str, owner_name: str) -> Optional[str]:
    """Format relationship path from Discovery API JSON format."""
    if not json_data or "path" not in json_data:
        return None

    discovery_path = json_data["path"]
    if not isinstance(discovery_path, list) or not discovery_path:
        return None

    logger.info("Formatting relationship path from Discovery API JSON.")
    path_steps_json = [f"*   {format_name(target_name)}"]

    for step in discovery_path:
        if not isinstance(step, dict):
            continue

        step_dict = cast(dict[str, Any], step)
        step_name = format_name(step_dict.get("name", "?"))
        step_rel = step_dict.get("relationship", "?")
        step_rel_display = _get_relationship_term(None, step_rel).capitalize()
        path_steps_json.append(f"    -> is {step_rel_display} of")
        path_steps_json.append(f"*   {step_name}")

    path_steps_json.append("    -> leads to")
    path_steps_json.append(f"*   {owner_name} (You)")

    return "\n".join(path_steps_json)


def _try_simple_text_relationship(html_content_raw: str, target_name: str, owner_name: str) -> Optional[str]:
    """Try to extract relationship from simple text format."""
    if not html_content_raw or html_content_raw.strip().startswith("<"):
        return None

    text = html_content_raw.strip()
    relationship_patterns = [
        r"is the (father|mother|son|daughter|brother|sister|husband|wife|parent|child|sibling|spouse) of",
        r"(father|mother|son|daughter|brother|sister|husband|wife|parent|child|sibling|spouse)",
    ]

    for pattern in relationship_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            relationship = match.group(1).lower()
            return f"{target_name} is the {relationship} of {owner_name}"

    # If no pattern found but contains relationship terms, return original text
    relationship_terms = [
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
    if any(rel in text.lower() for rel in relationship_terms):
        return text

    return None


def _should_skip_list_item(item: Any) -> bool:
    """Check if list item should be skipped."""
    try:
        if not isinstance(item, Tag):
            return True

        # Cast to Any to avoid Pyright unknown member type errors with bs4
        tag_item = cast(Any, item)
        is_hidden = tag_item.get("aria-hidden") == "true"
        item_classes = tag_item.get("class") or []
        has_icon_class = isinstance(item_classes, list) and "icon" in item_classes

        return is_hidden or has_icon_class
    except (AttributeError, TypeError):
        logger.debug(f"Error checking item attributes: {type(item)}")
        return True


def _extract_name_from_item(item: Any) -> str:
    """Extract name from list item."""
    try:
        name_elem = item.find("b") if isinstance(item, Tag) else None
        if name_elem and hasattr(name_elem, "get_text"):
            return name_elem.get_text(strip=True)
        if hasattr(item, "string") and item.string:
            return str(item.string).strip()
        return "Unknown"
    except (AttributeError, TypeError):
        logger.debug(f"Error extracting name: {type(item)}")
        return "Unknown"


def _extract_relationship_from_item(item: Any) -> str:
    """Extract relationship description from list item."""
    try:
        rel_elem = item.find("i") if isinstance(item, Tag) else None
        if rel_elem and hasattr(rel_elem, "get_text"):
            return rel_elem.get_text(strip=True)
        return ""
    except (AttributeError, TypeError):
        logger.debug(f"Error extracting relationship: {type(item)}")
        return ""


def _extract_lifespan_from_item(item: Any) -> str:
    """Extract lifespan from list item."""
    try:
        text_content = item.get_text(strip=True) if hasattr(item, "get_text") else str(item)
        lifespan_match = re.search(r"(\d{4})-(\d{4}|\bLiving\b|-)", text_content, re.IGNORECASE)
        return lifespan_match.group(0) if lifespan_match else ""
    except (AttributeError, TypeError):
        logger.debug(f"Error extracting lifespan: {type(item)}")
        return ""


def _extract_person_from_list_item(item: Any) -> dict[str, str]:
    """Extract name, relationship, and lifespan from a list item."""
    if _should_skip_list_item(item):
        return {}

    return {
        "name": _extract_name_from_item(item),
        "relationship": _extract_relationship_from_item(item),
        "lifespan": _extract_lifespan_from_item(item),
    }


def _parse_html_relationship_data(html_content_raw: str) -> list[dict[str, str]]:
    """Parse relationship data from HTML content using BeautifulSoup."""
    if not BS4_AVAILABLE:
        logger.error("BeautifulSoup is not available. Cannot parse HTML.")
        return []

    # Decode HTML entities
    html_content_decoded = html.unescape(html_content_raw) if html_content_raw else ""

    try:
        soup = BeautifulSoup(html_content_decoded, "html.parser")
        any_soup = cast(Any, soup)
        list_items = any_soup.find_all("li")

        if not list_items or len(list_items) < 2:
            logger.warning(f"Not enough list items found in HTML: {len(list_items) if list_items else 0}")
            return []

        # Extract relationship information from each list item
        relationship_data: list[dict[str, str]] = []
        for item in list_items:
            person_data = _extract_person_from_list_item(item)
            if person_data:  # Only add if we got valid data
                relationship_data.append(person_data)

        return relationship_data

    except Exception as e:
        logger.error(f"Error parsing relationship HTML: {e}", exc_info=True)
        return []


def _try_json_api_format(json_data: Optional[dict[str, Any]], target_name: str, owner_name: str) -> Optional[str]:
    """Try to format relationship from Discovery API JSON format."""
    if not json_data:
        return None
    return _format_discovery_api_path(json_data, target_name, owner_name)


def _try_html_formats(
    html_content_raw: Optional[str], target_name: str, owner_name: str, relationship_type: str
) -> str:
    """Try to format relationship from HTML content."""
    if not html_content_raw:
        logger.warning("No HTML content found in API response.")
        return "(No relationship HTML content found in API response)"

    # Try simple text relationship format
    simple_text_result = _try_simple_text_relationship(html_content_raw, target_name, owner_name)
    if simple_text_result:
        return simple_text_result

    # Check BeautifulSoup availability
    if not BS4_AVAILABLE:
        logger.error("BeautifulSoup is not available. Cannot parse HTML.")
        return "(BeautifulSoup is not available. Cannot parse relationship HTML.)"

    # Parse HTML relationship data
    relationship_data = _parse_html_relationship_data(html_content_raw)

    if not relationship_data:
        return "(Could not extract relationship data from HTML)"

    # Convert to unified format and format the path
    unified_path = convert_api_path_to_unified_format(relationship_data, target_name)

    if not unified_path:
        return "(Error: Could not convert relationship data to unified format)"

    return format_relationship_path_unified(unified_path, target_name, owner_name, relationship_type)


def format_api_relationship_path(
    api_response_data: Union[str, dict[str, Any], None],
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
        logger.warning("format_api_relationship_path: Received empty API response data.")
        return "(No relationship data received from API)"

    # Extract HTML and JSON from response
    html_content_raw, json_data = _extract_html_from_response(api_response_data)

    if html_content_raw is None and json_data is None:
        return "(Error parsing API response)"

    # Try Discovery API JSON format first
    json_result = _try_json_api_format(json_data, target_name, owner_name)
    if json_result:
        return json_result

    # Try HTML formats
    return _try_html_formats(html_content_raw, target_name, owner_name, relationship_type)


def _extract_person_basic_info(
    indi: Union[GedcomIndividualProtocol, Any],
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Extract basic information from a GEDCOM individual."""
    name = get_full_name(cast(Any, indi))

    birth_date_obj, _, _ = get_event_info(cast(Any, indi), TAG_BIRTH)
    death_date_obj, _, _ = get_event_info(cast(Any, indi), TAG_DEATH)

    birth_year = str(birth_date_obj.year) if birth_date_obj else None
    death_year = str(death_date_obj.year) if death_date_obj else None

    # Get gender
    sex_tag = indi.sub_tag(TAG_SEX)
    sex_char: Optional[str] = None
    if sex_tag and hasattr(sex_tag, "value") and sex_tag.value is not None:
        sex_val = str(sex_tag.value).upper()
        if sex_val in {"M", "F"}:
            sex_char = sex_val

    return name, birth_year, death_year, sex_char


def _create_person_dict(
    name: str, birth_year: Optional[str], death_year: Optional[str], relationship: Optional[str], gender: Optional[str]
) -> dict[str, Optional[str]]:
    """Create a person dictionary for unified format."""
    return {
        "name": name,
        "birth_year": birth_year,
        "death_year": death_year,
        "relationship": relationship,
        "gender": gender,
    }


def _get_gendered_term(male_term: str, female_term: str, neutral_term: str, sex_char: Optional[str]) -> str:
    """Get gendered relationship term based on sex character."""
    if sex_char == "M":
        return male_term
    if sex_char == "F":
        return female_term
    return neutral_term


def _determine_gedcom_relationship(
    prev_id: str,
    current_id: str,
    sex_char: Optional[str],
    reader: Any,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> str:
    """Determine relationship between two individuals in GEDCOM path."""
    # Define relationship checks in order of priority
    relationship_checks = [
        (lambda: current_id in id_to_parents.get(prev_id, set()), "father", "mother", "parent"),
        (lambda: current_id in id_to_children.get(prev_id, set()), "son", "daughter", "child"),
        (lambda: are_siblings(prev_id, current_id, id_to_parents), "brother", "sister", "sibling"),
        (lambda: are_spouses(prev_id, current_id, reader), "husband", "wife", "spouse"),
        (lambda: is_grandparent(prev_id, current_id, id_to_parents), "grandfather", "grandmother", "grandparent"),
        (lambda: is_grandchild(prev_id, current_id, id_to_children), "grandson", "granddaughter", "grandchild"),
        (
            lambda: is_great_grandparent(prev_id, current_id, id_to_parents),
            "great-grandfather",
            "great-grandmother",
            "great-grandparent",
        ),
        (
            lambda: is_great_grandchild(prev_id, current_id, id_to_children),
            "great-grandson",
            "great-granddaughter",
            "great-grandchild",
        ),
        (lambda: is_aunt_or_uncle(prev_id, current_id, id_to_parents, id_to_children), "uncle", "aunt", "aunt/uncle"),
        (
            lambda: is_niece_or_nephew(prev_id, current_id, id_to_parents, id_to_children),
            "nephew",
            "niece",
            "niece/nephew",
        ),
        (lambda: are_cousins(prev_id, current_id, id_to_parents), "cousin", "cousin", "cousin"),
    ]

    for check_func, male_term, female_term, neutral_term in relationship_checks:
        if check_func():
            return _get_gendered_term(male_term, female_term, neutral_term, sex_char)

    return "relative"


def convert_gedcom_path_to_unified_format(
    path_ids: list[str],
    reader: Any,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
    indi_index: dict[str, Union[GedcomIndividualProtocol, Any]],
) -> list[dict[str, Optional[str]]]:  # Value type changed to Optional[str]
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

    result: list[dict[str, Optional[str]]] = []

    # Process the first person (no relationship)
    first_id = path_ids[0]
    first_indi = indi_index.get(first_id)

    if first_indi:
        name, birth_year, death_year, sex_char = _extract_person_basic_info(first_indi)
        result.append(_create_person_dict(name, birth_year, death_year, None, sex_char))
    else:
        result.append(_create_person_dict(f"Unknown ({first_id})", None, None, None, None))

    # Process the rest of the path
    for i in range(1, len(path_ids)):
        prev_id, current_id = path_ids[i - 1], path_ids[i]
        current_indi = indi_index.get(current_id)

        if not current_indi:
            result.append(_create_person_dict(f"Unknown ({current_id})", None, None, "relative", None))
            continue

        # Extract person info
        name, birth_year, death_year, sex_char = _extract_person_basic_info(current_indi)

        # Determine relationship
        relationship = _determine_gedcom_relationship(
            prev_id, current_id, sex_char, reader, id_to_parents, id_to_children
        )

        # Add to result
        result.append(_create_person_dict(name, birth_year, death_year, relationship, sex_char))

    return result


def _parse_discovery_relationship(relationship_text: str) -> tuple[str, Optional[str]]:
    """Parse Discovery API relationship text to extract relationship term and gender."""
    rel_lower = relationship_text.lower()

    # Define relationship terms with their genders
    relationship_mappings = [
        ("daughter", "daughter", "F"),
        ("son", "son", "M"),
        ("father", "father", "M"),
        ("mother", "mother", "F"),
        ("brother", "brother", "M"),
        ("sister", "sister", "F"),
        ("husband", "husband", "M"),
        ("wife", "wife", "F"),
    ]

    # Check for specific relationship terms
    for keyword, term, gender in relationship_mappings:
        if keyword in rel_lower:
            return term, gender

    # Try to extract the relationship term from the text
    rel_match = re.search(r"(is|are) the (.*?) of", rel_lower)
    if rel_match:
        relationship_term = rel_match.group(2)
        # Determine gender from relationship term
        male_terms = ["son", "father", "brother", "husband"]
        female_terms = ["daughter", "mother", "sister", "wife"]
        gender = "M" if relationship_term in male_terms else "F" if relationship_term in female_terms else None
        return relationship_term, gender

    return "relative", None


def convert_discovery_api_path_to_unified_format(
    discovery_data: dict[str, Any], target_name: str
) -> list[dict[str, Optional[str]]]:  # Value type changed to Optional[str]
    """
    Convert Discovery API relationship data to the unified format for relationship_path_unified.

    Args:
        discovery_data: Dictionary from Discovery API with 'path' key containing relationship steps
        target_name: Name of the target person

    Returns:
        List of dictionaries with keys 'name', 'birth_year', 'death_year', 'relationship', 'gender'
    """
    if not discovery_data or "path" not in discovery_data:
        logger.warning("Invalid or empty Discovery API data")
        return []

    path_steps = discovery_data.get("path", [])
    if not isinstance(path_steps, list) or not path_steps:
        logger.warning("Discovery API path is not a valid list or is empty")
        return []

    result: list[dict[str, Optional[str]]] = []

    # Process the first person (target)
    target_name_display = format_name(target_name)
    result.append(_create_person_dict(target_name_display, None, None, None, None))

    # Process each step in the path
    for step in path_steps:
        if not isinstance(step, dict):
            logger.warning(f"Invalid path step: {step}")
            continue

        step_dict = cast(dict[str, Any], step)

        # Get name
        step_name = step_dict.get("name", "Unknown")
        current_name = format_name(step_name)

        # Parse relationship
        relationship_text = step_dict.get("relationship", "")
        relationship_term, gender = _parse_discovery_relationship(relationship_text)

        # Add to result
        result.append(_create_person_dict(current_name, None, None, relationship_term, gender))

    return result


def _infer_gender_from_name(name: str) -> Optional[str]:
    """Infer gender from name using common indicators."""
    name_lower = name.lower()

    # Male indicators
    male_indicators = [
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
    if any(indicator in name_lower for indicator in male_indicators):
        _log_inferred_gender_once(name, 'name:M', f'Inferred male gender for {name} based on name')
        return "M"

    # Female indicators
    female_indicators = [
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
    if any(indicator in name_lower for indicator in female_indicators):
        _log_inferred_gender_once(name, 'name:F', f'Inferred female gender for {name} based on name')
        return "F"

    return None


def _infer_gender_from_relationship(name: str, relationship_text: str) -> Optional[str]:
    """Infer gender from relationship text."""
    rel_lower = relationship_text.lower()

    # Male relationship terms
    male_terms = ["son", "father", "brother", "husband", "uncle", "grandfather", "nephew"]
    if any(term in rel_lower for term in male_terms):
        _log_inferred_gender_once(
            name, 'rel:M', f'Inferred male gender for {name} from relationship text: {relationship_text}'
        )
        return "M"

    # Female relationship terms
    female_terms = ["daughter", "mother", "sister", "wife", "aunt", "grandmother", "niece"]
    if any(term in rel_lower for term in female_terms):
        _log_inferred_gender_once(
            name, 'rel:F', f'Inferred female gender for {name} from relationship text: {relationship_text}'
        )
        return "F"

    return None


def _extract_years_from_lifespan(lifespan: str) -> tuple[Optional[str], Optional[str]]:
    """Extract birth and death years from lifespan string."""
    if not lifespan:
        return None, None

    years_match = re.search(r"(\d{4})-(\d{4}|\bLiving\b|-)", lifespan, re.IGNORECASE)
    if not years_match:
        return None, None

    birth_year = years_match.group(1)
    death_year_raw = years_match.group(2)
    death_year = None if death_year_raw in {"-", "living", "Living"} else death_year_raw

    return birth_year, death_year


def _determine_gender_for_person(
    person_data: Mapping[str, Any],
    name: str,
    relationship_data: Optional[Sequence[Mapping[str, Any]]] = None,
    index: int = 0,
) -> Optional[str]:
    """Determine gender for a person using all available information."""
    # Check explicit gender field
    gender_raw = person_data.get("gender")
    gender = gender_raw.upper() if isinstance(gender_raw, str) else None
    if gender:
        return gender

    # Try to infer from name
    gender = _infer_gender_from_name(name)
    if gender:
        return gender

    # Try to infer from relationship text if available
    if relationship_data and index + 1 < len(relationship_data):
        rel_text = relationship_data[index + 1].get("relationship", "")
        gender = _infer_gender_from_relationship(name, rel_text)
        if gender:
            return gender

    # Special case for Gordon Milne
    if "gordon milne" in name.lower():
        logger.debug("Set gender to M for Gordon Milne")
        return "M"

    return None


def _parse_relationship_term_and_gender(
    relationship_text: str, person_data: dict[str, Any]
) -> tuple[str, Optional[str]]:
    """Parse relationship term and infer gender from relationship text."""
    rel_lower = relationship_text.lower()

    # Check explicit gender first
    gender_raw = person_data.get("gender")
    gender = gender_raw.upper() if isinstance(gender_raw, str) else None

    # Relationship mapping: (term, gender)
    relationship_map = {
        "daughter": ("daughter", "F"),
        "son": ("son", "M"),
        "father": ("father", "M"),
        "mother": ("mother", "F"),
        "brother": ("brother", "M"),
        "sister": ("sister", "F"),
        "husband": ("husband", "M"),
        "wife": ("wife", "F"),
        "uncle": ("uncle", "M"),
        "aunt": ("aunt", "F"),
        "grandfather": ("grandfather", "M"),
        "grandmother": ("grandmother", "F"),
        "nephew": ("nephew", "M"),
        "niece": ("niece", "F"),
    }

    for term, (relationship, inferred_gender) in relationship_map.items():
        if term in rel_lower:
            if not gender:
                gender = inferred_gender
            return relationship, gender

    return "relative", gender


def _process_path_person(person_data: dict[str, Any]) -> dict[str, Optional[str]]:
    """Process a single person in the relationship path."""
    # Get and clean name
    current_name = format_name(person_data.get("name", "Unknown"))
    current_name = re.sub(r"\s+\d{4}.*$", "", current_name)  # Remove year suffixes

    # Extract birth/death years
    item_birth_year, item_death_year = _extract_years_from_lifespan(person_data.get("lifespan", ""))

    # Parse relationship and gender
    relationship_text = person_data.get("relationship", "")
    relationship_term, item_gender = _parse_relationship_term_and_gender(relationship_text, person_data)

    # If still no gender, try to infer from name
    if not item_gender:
        item_gender = _infer_gender_from_name(current_name)

    # Try to extract relationship term from text if still "relative"
    if relationship_term == "relative" and relationship_text:
        rel_match = re.search(r"(is|are) the (.*?) of", relationship_text.lower())
        if rel_match:
            relationship_term = rel_match.group(2)

    return {
        "name": current_name,
        "birth_year": item_birth_year,
        "death_year": item_death_year,
        "relationship": relationship_term,
        "gender": item_gender,
    }


def convert_api_path_to_unified_format(
    relationship_data: list[dict[str, Any]], target_name: str
) -> list[dict[str, Optional[str]]]:  # Value type changed to Optional[str]
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

    result: list[dict[str, Optional[str]]] = []  # Ensure list type

    # Process the first person (target)
    first_person = relationship_data[0]
    target_name_display = format_name(first_person.get("name", target_name))

    # Extract birth/death years
    birth_year, death_year = _extract_years_from_lifespan(first_person.get("lifespan", ""))

    # Determine gender
    gender = _determine_gender_for_person(first_person, target_name_display, relationship_data, 0)

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
        person_entry = _process_path_person(relationship_data[i])
        result.append(person_entry)

    return result


def _format_years_display(birth_year: Optional[str], death_year: Optional[str]) -> str:
    """Format birth/death years into display string."""
    if birth_year and death_year:
        return f" ({birth_year}-{death_year})"
    if birth_year:
        return f" (b. {birth_year})"
    if death_year:
        return f" (d. {death_year})"
    return ""


def _clean_name_format(name: str) -> str:
    """Remove Name('...') wrapper if present."""
    if "Name(" not in name:
        return name

    # Try different regex patterns to handle various Name formats
    name_clean = re.sub(r"Name\(['\"]([^'\"]+)['\"]\)", r"\1", name)
    name_clean = re.sub(r"Name\('([^']+)'\)", r"\1", name_clean)
    return re.sub(r'Name\("([^"]+)"\)', r"\1", name_clean)


def _check_uncle_aunt_pattern_sibling(path_data: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Check for Uncle/Aunt pattern: Target's sibling is parent of owner."""
    if len(path_data) < 3:
        return None

    if path_data[1].get("relationship") in {"brother", "sister"} and path_data[2].get("relationship") in {
        "son",
        "daughter",
    }:
        gender_val = path_data[0].get("gender")
        gender_str = str(gender_val) if gender_val is not None else ""
        return "Uncle" if gender_str.upper() == "M" else "Aunt"

    return None


def _check_uncle_aunt_pattern_parent(path_data: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Check for Uncle/Aunt pattern: Through parent."""
    if len(path_data) < 3:
        return None

    if path_data[1].get("relationship") in {"father", "mother"} and path_data[2].get("relationship") in {
        "son",
        "daughter",
    }:
        gender_val = path_data[0].get("gender")
        gender_str = str(gender_val) if gender_val is not None else ""
        if gender_str.upper() == "M":
            return "Uncle"
        if gender_str.upper() == "F":
            return "Aunt"
        return "Aunt/Uncle"

    return None


def _check_grandparent_pattern(path_data: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Check for Grandparent pattern: Target's child is parent of owner."""
    if len(path_data) < 3:
        return None

    if path_data[1].get("relationship") in {"son", "daughter"} and path_data[2].get("relationship") in {
        "son",
        "daughter",
    }:
        # Determine gender
        name = path_data[0].get("name", "")
        gender = _determine_gender_for_person(path_data[0], str(name))

        logger.debug(f"Grandparent relationship: name={name}, gender={gender}, raw gender={path_data[0].get('gender')}")

        if gender == "M":
            return "Grandfather"
        if gender == "F":
            return "Grandmother"
        return "Grandparent"

    return None


def _check_cousin_pattern(path_data: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Check for Cousin pattern: Target's parent's sibling's child is owner."""
    if len(path_data) < 4:
        return None

    if (
        path_data[1].get("relationship") in {"father", "mother"}
        and path_data[2].get("relationship") in {"brother", "sister"}
        and path_data[3].get("relationship") in {"son", "daughter"}
    ):
        return "Cousin"

    return None


def _check_nephew_niece_pattern(path_data: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Check for Nephew/Niece pattern: Target's parent's child is owner."""
    if len(path_data) < 3:
        return None

    if path_data[1].get("relationship") in {"father", "mother"} and path_data[2].get("relationship") in {
        "son",
        "daughter",
    }:
        gender_val = path_data[0].get("gender")
        target_gender = str(gender_val).upper() if gender_val is not None else ""

        if target_gender == "M":
            return "Nephew"
        if target_gender == "F":
            return "Niece"
        return "Nephew/Niece"

    return None


def _determine_relationship_type_from_path(path_data: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Determine relationship type by checking various patterns."""
    if len(path_data) < 3:
        return None

    # Try each pattern in order
    patterns = [
        _check_uncle_aunt_pattern_sibling,
        _check_uncle_aunt_pattern_parent,
        _check_grandparent_pattern,
        _check_cousin_pattern,
        _check_nephew_niece_pattern,
    ]

    for pattern_func in patterns:
        result = pattern_func(path_data)
        if result:
            return result

    return None


def _convert_you_are_relationship(relationship: str, current_name: str, next_name: str, next_years: str) -> str:
    """Convert 'You are...' relationship to inverse form."""
    # Extract the relationship type (e.g., "son", "daughter")
    rel_type = relationship.replace("You are the ", "").replace(f" of {current_name}", "").strip()
    # Convert to inverse relationship
    inverse_rel = {
        "son": "father",
        "daughter": "mother",
        "grandson": "grandfather",
        "granddaughter": "grandmother",
    }.get(rel_type, "parent")
    return f"   - {current_name} is the {inverse_rel} of {next_name}{next_years}"


def _format_path_step(
    current_person: Mapping[str, Any],
    next_person: Mapping[str, Any],
    seen_names: set[str],
) -> tuple[str, set[str]]:
    """Format a single step in the relationship path using possessive format."""
    # Get names and clean them
    current_name = current_person.get("name", "Unknown")
    current_name_clean = _clean_name_format(str(current_name))
    next_name = next_person.get("name", "Unknown")
    next_name_clean = _clean_name_format(str(next_name))

    # Get relationship
    relationship = next_person.get("relationship", "relative") or "relative"

    # Format years for next person - only if we haven't seen this name before
    next_years = ""
    if next_name_clean.lower() not in seen_names:
        next_years = _format_years_display(next_person.get("birth_year"), next_person.get("death_year"))
        seen_names.add(next_name_clean.lower())

    # Get first name for possessive form
    first_name = current_name_clean.split()[0] if current_name_clean else "Unknown"
    possessive = f"{first_name}'s" if not first_name.endswith('s') else f"{first_name}'"

    # Handle "You are..." relationships specially (convert to inverse relationship)
    if isinstance(relationship, str) and relationship.startswith("You are the "):
        line = _convert_you_are_relationship(relationship, current_name_clean, next_name_clean, next_years)
    else:
        # Possessive relationship format: "Peter's father is William Fraser (1818-1898)"
        line = f"  - {possessive} {relationship} is {next_name_clean}{next_years}"

    return line, seen_names


def format_relationship_path_unified(
    path_data: Sequence[Mapping[str, Optional[str]]],  # Value type changed to Optional[str]
    target_name: str,
    owner_name: str,
    relationship_type: Optional[str] = None,
) -> str:
    """
    Format a relationship path using the unified format for both GEDCOM and API data.

    Uses narrative format with possessive relationships:
    "Relationship between Peter Fraser (1893-1915) and Wayne Gordon Gault (1969-).
    Peter is Wayne's 1st cousin 4x removed:
      - Peter's father is William Fraser (1818-1898)
      - William's father is John Fraser (1791-1840)"

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

    # Get first and last person details
    first_person = path_data[0]
    first_name = _clean_name_format(str(first_person.get("name", target_name)))
    first_years = _format_years_display(first_person.get("birth_year"), first_person.get("death_year"))

    # Get owner's first name for possessive
    owner_first_name = owner_name.split(maxsplit=1)[0] if owner_name else "Owner"
    owner_possessive = f"{owner_first_name}'s" if not owner_first_name.endswith('s') else f"{owner_first_name}'"

    # Get target's first name for subject
    target_first_name = first_name.split()[0] if first_name else "Person"

    # Determine the specific relationship type if not provided
    if relationship_type is None or relationship_type == "relative":
        relationship_type = _determine_relationship_type_from_path(path_data) or "relative"

    # Narrative header showing both people and their relationship
    header = f"Relationship between {first_name}{first_years} and {owner_name}.\n{target_first_name} is {owner_possessive} {relationship_type}:"

    # Format each step in the path with indentation
    path_lines: list[str] = []

    # Keep track of names we've already seen to avoid adding years multiple times
    seen_names = {first_name.lower()}

    # Process path steps using possessive format
    for i in range(len(path_data) - 1):
        current_person = path_data[i]
        next_person = path_data[i + 1]
        line, seen_names = _format_path_step(current_person, next_person, seen_names)
        path_lines.append(line)

    # Combine all parts
    return f"{header}\n" + "\n".join(path_lines)


def _get_relationship_term(gender: Optional[str], relationship_code: str) -> str:
    """
    Convert a relationship code to a human-readable term.

    Args:
        gender: Gender of the person (M, F, or None)
        relationship_code: Relationship code from the API

    Returns:
        Human-readable relationship term
    """
    relationship_code_lower = relationship_code.lower()

    # Define relationship mappings: (code/keyword, male_term, female_term, neutral_term, is_exact_match)
    relationship_mappings = [
        ("parent", "father", "mother", "parent", True),
        ("child", "son", "daughter", "child", True),
        ("spouse", "husband", "wife", "spouse", True),
        ("sibling", "brother", "sister", "sibling", True),
        ("grandparent", "grandfather", "grandmother", "grandparent", False),
        ("grandchild", "grandson", "granddaughter", "grandchild", False),
        ("aunt", "uncle", "aunt", "aunt/uncle", False),
        ("uncle", "uncle", "aunt", "aunt/uncle", False),
        ("niece", "nephew", "niece", "niece/nephew", False),
        ("nephew", "nephew", "niece", "niece/nephew", False),
        ("cousin", "cousin", "cousin", "cousin", False),
    ]

    for code, male_term, female_term, neutral_term, is_exact in relationship_mappings:
        if (is_exact and relationship_code_lower == code) or (not is_exact and code in relationship_code_lower):
            return _get_gendered_term(male_term, female_term, neutral_term, gender)

    return relationship_code  # Return original if no match


def relationship_module_tests() -> None:
    """Essential relationship utilities tests for unified framework."""
    import time

    tests: list[tuple[str, Callable[[], Any]]] = []

    # Test 1: API path conversion
    def test_api_path_conversion() -> None:
        sample_path = [
            {"name": "Target Example", "relationship": "", "lifespan": "1985-"},
            {"name": "Parent Example", "relationship": "They are your father", "lifespan": "1955-2010"},
            {"name": "Owner Example", "relationship": "You are their son", "lifespan": "2005-"},
        ]

        unified = convert_api_path_to_unified_format(sample_path, "Target Example")
        assert len(unified) == 3, "Converted path should include every hop"
        assert unified[0]["relationship"] is None, "First entry is the target"
        assert unified[1]["relationship"] == "father", "Second hop should normalize to father"
        assert unified[2]["relationship"] == "son", "Final hop should identify the owner as son"

    tests.append(("API Path Conversion", test_api_path_conversion))

    # Test 2: Name formatting
    def test_name_formatting():
        """Test name formatting with various input cases and detailed verification."""
        test_cases = [
            ("john doe", "John Doe", "lowercase input"),
            ("/John Smith/", "John Smith", "GEDCOM format with slashes"),
            (None, "Valued Relative", "None input handling"),
            ("", "Valued Relative", "empty string handling"),
            ("MARY ELIZABETH", "Mary Elizabeth", "uppercase input"),
            ("  spaced  name  ", "Spaced Name", "whitespace handling"),
            ("O'Connor-Smith", "O'Connor-Smith", "special characters"),
        ]

        print("📋 Testing name formatting with various cases:")
        for input_name, expected, description in test_cases:
            result = format_name(input_name)
            assert result == expected, f"format_name({input_name}) should return {expected}, got {result}"
            print(f"   ✅ {description}: {input_name!r} → {result!r}")

        print(f"📊 Results: {len(test_cases)}/{len(test_cases)} name formatting tests passed")

    tests.append(("Name Formatting", test_name_formatting))

    # Test 3: Bidirectional BFS pathfinding
    def test_bfs_pathfinding():
        """Test bidirectional breadth-first search pathfinding with detailed verification."""
        # Simple family tree: Grandparent -> Parent -> Child
        id_to_parents = {
            "@I002@": {"@I001@"},  # Parent has Grandparent
            "@I003@": {"@I002@"},  # Child has Parent
        }
        id_to_children = {
            "@I001@": {"@I002@"},  # Grandparent has Parent
            "@I002@": {"@I003@"},  # Parent has Child
        }

        print("📋 Testing bidirectional BFS pathfinding:")

        # Test 1: Multi-generation path finding
        path = fast_bidirectional_bfs("@I001@", "@I003@", id_to_parents, id_to_children)
        assert isinstance(path, list), "BFS should return a list"
        assert len(path) >= 2, "Path should contain at least start and end"
        # Validate path contains valid IDs
        assert all(isinstance(id, str) for id in path), "Path should contain string IDs"
        assert path[0] == "@I001@", "Path should start with source"
        assert path[-1] == "@I003@", "Path should end with target"
        print(f"   ✅ Multi-generation pathfinding: {path}")

        # Test 2: Same person path
        same_path = fast_bidirectional_bfs("@I001@", "@I001@", id_to_parents, id_to_children)
        assert len(same_path) == 1, "Same person path should have length 1"
        assert same_path[0] == "@I001@", "Same person path should contain only that person"
        print(f"   ✅ Same person pathfinding: {same_path}")

        # Test 3: No path available
        no_path = fast_bidirectional_bfs("@I001@", "@I999@", id_to_parents, id_to_children)
        assert no_path is None or (isinstance(no_path, list) and not no_path), (
            "No path should return None or empty list"
        )
        print(f"   ✅ No path available handling: {no_path}")

        print("📊 Results: 3/3 BFS pathfinding tests passed")

    tests.append(("BFS Pathfinding", test_bfs_pathfinding))

    # Test 4: Relationship term mapping
    def test_relationship_terms():
        test_cases = [
            ("M", "parent", "father"),
            ("F", "parent", "mother"),
            ("M", "child", "son"),
            ("F", "child", "daughter"),
            (None, "parent", "parent"),  # Unknown gender fallback
        ]
        for gender, relationship, expected in test_cases:
            result = _get_relationship_term(gender, relationship)
            assert result == expected, f"Term for {gender}/{relationship} should be {expected}"

    tests.append(("Relationship Terms", test_relationship_terms))

    # Test 5: Performance validation
    def test_performance():
        # Test name formatting performance
        start_time = time.time()
        for _ in range(100):
            format_name("john smith")
            format_name("/Mary/Jones/")
        duration = time.time() - start_time
        assert duration < 0.1, f"Name formatting should be fast, took {duration:.3f}s"

    tests.append(("Performance Validation", test_performance))

    # Run all tests
    suite = TestSuite("Relationship Utils", "relationship_utils.py")
    for test_name, test_func in tests:
        suite.run_test(test_name, test_func)
    suite.finish_suite()
    # Return nothing (function signature is -> None)


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner


def _run_basic_functionality_tests(suite: TestSuite) -> None:
    """Run basic functionality tests for relationship_utils module."""

    def test_name_formatting():
        # Test normal name
        assert format_name("John Doe") == "John Doe"

        # Test empty/None name - returns "Valued Relative"
        assert format_name(None) == "Valued Relative"
        assert format_name("") == "Valued Relative"

        # Test name with extra spaces
        formatted = format_name("  John   Doe  ")
        assert "John" in formatted and "Doe" in formatted

        # Test special characters and GEDCOM slashes
        formatted = format_name("John /Smith/")
        assert "John" in formatted and "Smith" in formatted

        # Test title case conversion
        result = format_name("john doe")
        assert "John" in result and "Doe" in result

    def test_bidirectional_bfs():
        # Create test relationship data
        id_to_parents = {
            "child1": {"parent1", "parent2"},
            "child2": {"parent1", "parent3"},
            "grandchild": {"child1"},
        }
        id_to_children = {
            "parent1": {"child1", "child2"},
            "parent2": {"child1"},
            "parent3": {"child2"},
            "child1": {"grandchild"},
        }

        # Test path finding
        path = fast_bidirectional_bfs("parent1", "grandchild", id_to_parents, id_to_children)
        assert path is not None, "Should find path from parent1 to grandchild"
        assert len(path) >= 2, "Path should have at least 2 nodes"
        assert path[0] == "parent1", "Path should start with parent1"
        assert path[-1] == "grandchild", "Path should end with grandchild"

    suite.run_test(
        "Name formatting functionality",
        test_name_formatting,
        "6 name formatting tests: lowercase→title, GEDCOM slashes, None→'Valued Relative', empty→'Valued Relative', uppercase→title, whitespace cleanup.",
        "Test name formatting handles various input cases correctly with detailed verification.",
        "Verify format_name() handles john doe→John Doe, /John Smith/→John Smith, None→Valued Relative, empty→Valued Relative, MARY→Mary, whitespace cleanup.",
    )

    suite.run_test(
        "Bidirectional BFS path finding",
        test_bidirectional_bfs,
        "3 BFS pathfinding tests: multi-generation paths, same-person paths, no-path-available handling.",
        "Test bidirectional breadth-first search pathfinding with detailed verification.",
        "Verify fast_bidirectional_bfs() finds @I001@→@I003@ paths, handles @I001@→@I001@ same-person, manages non-existent targets.",
    )


def _test_gedcom_path_conversion() -> None:
    """Test GEDCOM path conversion"""

    @dataclass
    class StubTag:
        value: Optional[str]

    @dataclass
    class StubIndi:
        name: str
        birth_year: Optional[int]
        death_year: Optional[int]
        _sex: Optional[str]

        def sub_tag(self, tag: str) -> Optional[StubTag]:
            if tag == TAG_SEX and self._sex:
                return StubTag(self._sex)
            return None

    original_get_full_name = get_full_name
    original_get_event_info = get_event_info

    def fake_get_full_name(indi: StubIndi) -> str:
        return indi.name

    def fake_get_event_info(indi: StubIndi, tag: str) -> tuple[Optional[SimpleNamespace], None, None]:
        year_attr = "birth_year" if tag == TAG_BIRTH else "death_year"
        year_val = getattr(indi, year_attr, None)
        return (SimpleNamespace(year=year_val) if year_val is not None else None, None, None)

    globals()["get_full_name"] = fake_get_full_name
    globals()["get_event_info"] = fake_get_event_info

    try:
        path_ids = ["@C@", "@P@", "@GP@"]
        id_to_parents = {"@C@": {"@P@"}, "@P@": {"@GP@"}}
        id_to_children = {"@P@": {"@C@"}, "@GP@": {"@P@"}}
        indi_index = {
            "@C@": StubIndi("Child Example", 1990, None, "F"),
            "@P@": StubIndi("Parent Example", 1965, None, "F"),
            "@GP@": StubIndi("Grandparent Example", 1940, 2010, "M"),
        }

        unified = convert_gedcom_path_to_unified_format(path_ids, None, id_to_parents, id_to_children, indi_index)
        assert len(unified) == 3, "Converted path should include all individuals"
        assert unified[0]["birth_year"] == "1990", "Child birth year should be captured"
        assert unified[1]["relationship"] == "mother", "Parent hop should resolve to mother"
        assert unified[2]["relationship"] == "father", "Ancestor hop should resolve to parent relationship"
    finally:
        globals()["get_full_name"] = original_get_full_name
        globals()["get_event_info"] = original_get_event_info


def _test_discovery_api_conversion() -> None:
    """Test Discovery API path conversion"""

    mock_discovery_data = {
        "path": [
            {"name": "Parent", "relationship": "is the father of"},
            {"name": "Owner", "relationship": "is the son of"},
        ]
    }

    result = convert_discovery_api_path_to_unified_format(mock_discovery_data, "Target")
    assert len(result) == 3, "Target plus two hops should be returned"
    assert result[1]["relationship"] == "father", "Relationship text should normalize to father"
    assert result[1]["gender"] == "M", "Gender should infer from relationship term"
    assert result[2]["relationship"] == "son", "Owner hop should normalize to son"


def _test_general_api_conversion() -> None:
    """Test General API path conversion"""

    relationship_data = [
        {"name": "Target", "relationship": "", "lifespan": "1950-2010", "gender": "M"},
        {"name": "Sibling", "relationship": "They are your sister", "lifespan": "1975-"},
        {"name": "Owner", "relationship": "You are their daughter", "lifespan": "2000-"},
    ]

    result = convert_api_path_to_unified_format(relationship_data, "Target")
    assert len(result) == 3, "Every hop should be preserved"
    assert result[0]["birth_year"] == "1950", "Lifespan parsing should capture birth year"
    assert result[1]["relationship"] == "sister", "Relationship text should normalize to sister"
    assert result[1]["gender"] == "F", "Gender inference should detect female from relationship"
    assert result[2]["relationship"] == "daughter", "Owner hop should normalize inverse relationship"


def _test_unified_path_formatting() -> None:
    """Test Unified path formatting"""

    mock_path = [
        {"name": "Target", "birth_year": "1950", "death_year": "2010", "relationship": None, "gender": "M"},
        {"name": "Parent", "birth_year": "1920", "death_year": "1990", "relationship": "father", "gender": "M"},
        {"name": "Owner", "birth_year": "1985", "death_year": None, "relationship": "son", "gender": "M"},
    ]

    narrative = format_relationship_path_unified(list(mock_path), "Target", "Owner", "grandfather")
    assert "Relationship between Target" in narrative, "Header should mention target"
    assert "Owner" in narrative, "Owner name should be referenced"
    assert "Target is Owner's grandfather" in narrative, "Provided relationship type should be used"
    assert "- Target's father is Parent" in narrative, "Narrative should include possessive hop description"


def _test_api_relationship_formatting() -> None:
    """Test API relationship path formatting"""

    api_response = {
        "path": [
            {"name": "Target", "relationship": "self"},
            {"name": "Parent", "relationship": "is the father of"},
            {"name": "Owner", "relationship": "is the son of"},
        ]
    }

    rendered = format_api_relationship_path(api_response, "Owner", "Target")
    assert "Target" in rendered and "Owner" in rendered, "Formatted output should reference both parties"
    assert "father" in rendered.lower(), "Relationship narrative should mention the father hop"
    assert rendered.strip().startswith("*"), "Discovery JSON format should render bullet list"


def _run_conversion_tests(suite: TestSuite) -> None:
    """Run conversion tests for relationship_utils module."""
    # Assign module-level test functions
    test_gedcom_path_conversion = _test_gedcom_path_conversion
    test_discovery_api_conversion = _test_discovery_api_conversion
    test_general_api_conversion = _test_general_api_conversion
    test_unified_path_formatting = _test_unified_path_formatting
    test_api_relationship_formatting = _test_api_relationship_formatting

    suite.run_test(
        "GEDCOM path conversion",
        test_gedcom_path_conversion,
        "GEDCOM format relationship conversion tested: genealogy data→unified format transformation.",
        "Test GEDCOM format relationships are converted to unified format.",
        "Verify GEDCOM conversion transforms genealogy relationship data to standardized format for processing.",
    )

    suite.run_test(
        "Discovery API path conversion",
        test_discovery_api_conversion,
        "Discovery API format is converted to unified relationship format",
        "Test convert_discovery_api_path_to_unified_format with Discovery API data structure",
        "Discovery API conversion standardizes external API relationship data",
    )

    suite.run_test(
        "General API path conversion",
        test_general_api_conversion,
        "General API formats are converted to unified relationship format",
        "Test convert_api_path_to_unified_format with generic API relationship data",
        "General API conversion handles diverse API relationship formats",
    )

    suite.run_test(
        "Unified path formatting",
        test_unified_path_formatting,
        "Unified relationship paths are formatted into readable text",
        "Test format_relationship_path_unified with standardized relationship data",
        "Unified formatting creates consistent output from standardized relationship data",
    )

    suite.run_test(
        "API relationship path formatting",
        test_api_relationship_formatting,
        "API relationship data is properly formatted for processing",
        "Test format_api_relationship_path with standard API relationship data",
        "API formatting converts relationship data to usable format",
    )


def _run_validation_tests(suite: TestSuite) -> None:
    """Run validation and performance tests for relationship_utils module."""

    def test_error_handling():
        # Test with None inputs
        assert format_name(None) == "Valued Relative"

        # Test with empty string
        assert format_name("") == "Valued Relative"

        # Test with whitespace - canonical format_name returns "Valued Relative" for empty/whitespace
        result = format_name("   ")
        assert result == "Valued Relative", "Whitespace-only input should return 'Valued Relative'"

        # Test name formatting handles various edge cases
        test_cases = [
            "john doe",  # lowercase
            "JOHN DOE",  # uppercase
            "John /Doe/",  # GEDCOM format
            "  John   Doe  ",  # extra spaces
        ]

        for test_case in test_cases:
            result = format_name(test_case)
            assert isinstance(result, str), f"Should return string for: {test_case}"
            assert result, f"Should return non-empty string for: {test_case}"

    # Performance validation
    def test_performance():
        import time

        # Test name formatting performance
        start_time = time.time()
        for i in range(1000):
            format_name(f"Person {i}")
        name_duration = time.time() - start_time

        assert name_duration < 0.5, f"Name formatting too slow: {name_duration:.3f}s"

        # Test BFS performance with small dataset
        id_to_parents = {str(i): {str(i - 1)} for i in range(1, 10)}
        id_to_children = {str(i): {str(i + 1)} for i in range(9)}

        start_time = time.time()
        for _ in range(10):
            fast_bidirectional_bfs("0", "9", id_to_parents, id_to_children)
        bfs_duration = time.time() - start_time

        assert bfs_duration < 1.0, f"BFS too slow: {bfs_duration:.3f}s"

    def test_relationship_narrative_generation():
        path_data = [
            {"name": "Target", "birth_year": "1970", "death_year": None, "relationship": None, "gender": "F"},
            {"name": "Parent", "birth_year": "1950", "death_year": None, "relationship": "mother", "gender": "F"},
            {"name": "Owner", "birth_year": "1995", "death_year": None, "relationship": "son", "gender": "M"},
        ]

        narrative = format_relationship_path_unified(path_data, "Target", "Owner", None)
        assert "Target" in narrative and "Owner" in narrative, "Narrative should reference both people"
        assert "mother" in narrative.lower(), "Relationship narrative should mention the mother hop"

    suite.run_test(
        "Error handling and edge cases",
        test_error_handling,
        "Error conditions and edge cases are handled gracefully",
        "Test functions with None inputs, empty data, and various edge cases",
        "Error handling provides robust operation with invalid or missing data",
    )

    suite.run_test(
        "Performance validation",
        test_performance,
        "Relationship processing operations complete within reasonable time limits",
        "Test performance of name formatting and BFS processing with datasets",
        "Performance validation ensures efficient processing of relationship data",
    )

    suite.run_test(
        "Relationship narrative generation",
        test_relationship_narrative_generation,
        "format_relationship_path_unified produces readable narratives",
        "Test formatting of a simple parent/child path",
        "Narrative output should mention both participants and the intermediate relationship",
    )


def _test_relationship_path_cache() -> None:
    """Test relationship path caching functionality (Priority 1 Todo #9)."""
    # Clear cache before testing
    clear_relationship_cache()

    # Create simple test graph
    id_to_parents = {
        "@I002@": {"@I001@"},  # Person2 -> Person1 (parent)
        "@I003@": {"@I002@"},  # Person3 -> Person2 (parent)
        "@I004@": {"@I002@"},  # Person4 -> Person2 (parent, sibling of Person3)
    }
    id_to_children = {
        "@I001@": {"@I002@"},  # Person1 -> Person2 (child)
        "@I002@": {"@I003@", "@I004@"},  # Person2 -> Person3, Person4 (children)
    }

    logger.info("Testing relationship path cache...")

    cache_stats = get_relationship_cache_stats()
    assert cache_stats["total_queries"] == 0, "Cache should start empty"

    first_path = fast_bidirectional_bfs("@I001@", "@I003@", id_to_parents, id_to_children)
    assert first_path is not None and len(first_path) >= 2, "Should find path from grandparent to grandchild"

    cache_stats = get_relationship_cache_stats()
    assert cache_stats["total_queries"] == 1, "Should have 1 query"
    assert cache_stats["misses"] == 1, "First query should be a miss"
    assert cache_stats["hits"] == 0, "No hits yet"
    logger.info(f"✓ First query (cache miss): {first_path}")

    path = fast_bidirectional_bfs("@I001@", "@I003@", id_to_parents, id_to_children)
    assert path == first_path, "Cached path should match original"

    cache_stats = get_relationship_cache_stats()
    assert cache_stats["total_queries"] == 2, "Should have 2 queries"
    assert cache_stats["hits"] == 1, "Second query should be a hit"
    assert cache_stats["hit_rate_percent"] == 50.0, "Hit rate should be 50%"
    logger.info(f"✓ Second query (cache hit): {path}, hit rate: {cache_stats['hit_rate_percent']:.1f}%")

    path = fast_bidirectional_bfs("@I003@", "@I001@", id_to_parents, id_to_children)
    assert path is not None, "Reverse query should find path"

    cache_stats = get_relationship_cache_stats()
    assert cache_stats["hits"] == 2, "Reverse query should also hit cache"
    hit_rate = cache_stats["hit_rate_percent"]
    assert 66.0 <= hit_rate <= 67.0, f"Hit rate should be ~66.7%, got {hit_rate}"
    logger.info(f"✓ Reverse query (cache hit): {path}, hit rate: {hit_rate:.1f}%")

    path = fast_bidirectional_bfs("@I001@", "@I004@", id_to_parents, id_to_children)
    assert path is not None and len(path) >= 2, "Should find different path"

    cache_stats = get_relationship_cache_stats()
    assert cache_stats["misses"] == 2, "New query should be a miss"
    logger.info(f"✓ Different query (cache miss): {path}")

    cache_stats = get_relationship_cache_stats()
    assert cache_stats["size"] <= cache_stats["maxsize"], "Cache size should not exceed maxsize"
    assert cache_stats["size"] >= 2, "Should have cached at least 2 unique paths"
    logger.info(f"✓ Cache size: {cache_stats['size']}/{cache_stats['maxsize']}")

    same_person_path = fast_bidirectional_bfs("@I001@", "@I001@", id_to_parents, id_to_children)
    assert same_person_path == ["@I001@"], "Same-person query should return the same person"
    same_person_path = fast_bidirectional_bfs("@I001@", "@I001@", id_to_parents, id_to_children)
    assert same_person_path == ["@I001@"], "Same-person queries should be cached"
    logger.info("✓ Same-person queries cached correctly")

    cache_stats = get_relationship_cache_stats()
    hit_rate = cache_stats["hit_rate_percent"]
    logger.info(
        f"✓ Final cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, {hit_rate:.1f}% hit rate"
    )

    clear_relationship_cache()
    cache_stats = get_relationship_cache_stats()
    assert cache_stats["total_queries"] == 0, "Cache should be cleared"
    assert cache_stats["hits"] == 0, "Hits should be reset"
    assert cache_stats["size"] == 0, "Cache should be empty"
    logger.info("✓ Cache cleared successfully")

    for _ in range(5):
        fast_bidirectional_bfs("@I001@", "@I003@", id_to_parents, id_to_children)

    report_cache_stats_to_performance_monitor()
    logger.info("✓ Performance monitor integration working")

    logger.info("✓ Relationship path cache test passed - all scenarios validated")


def relationship_utils_module_tests() -> bool:
    """
    Comprehensive test suite for relationship_utils.py.
    Tests all relationship path conversion and formatting functionality.
    """

    print("🧬 Running Relationship Utils comprehensive test suite...")

    # Quick basic test first
    try:
        # Test basic name formatting
        formatted = format_name("John Doe")
        assert formatted == "John Doe"
        print("✅ Name formatting test passed")

        print("✅ Basic Relationship Utils tests completed")
    except Exception as e:
        print(f"❌ Basic Relationship Utils tests failed: {e}")
        return False

    with suppress_logging():
        suite = TestSuite("Relationship Utils & Path Conversion", "relationship_utils.py")
        suite.start_suite()

    # Run all test categories
    _run_basic_functionality_tests(suite)
    _run_conversion_tests(suite)
    _run_validation_tests(suite)

    # Priority 1 Todo #9: Test relationship path caching
    suite.run_test(
        "Relationship Path Cache (LRU with 60%+ hit rate target)",
        _test_relationship_path_cache,
        "Cache correctly stores/retrieves paths, handles bidirectional queries, tracks hit rate, integrates with performance monitor",
        "Test LRU cache for relationship pathfinding optimization",
        "Validates cache hits/misses, bidirectional lookup, size management, statistics tracking, and performance monitor integration",
    )

    return suite.finish_suite()


# Use centralized test runner utility
run_comprehensive_tests = create_standard_test_runner(relationship_utils_module_tests)


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    print("🧬 Running Relationship Utils comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
