#!/usr/bin/env python3
"""Relationship Graph - BFS Pathfinding and Cache.

BFS-based bidirectional pathfinding through family trees, path scoring,
and LRU caching for relationship path queries.

Split from research.relationship_utils to reduce module size.
"""

# === CORE INFRASTRUCTURE ===
import logging
import time
from collections import OrderedDict, deque
from typing import Any

from core.common_params import GraphContext
from genealogy.relationship_calculations import (
    find_direct_relationship,
    has_direct_relationship,
)

logger = logging.getLogger(__name__)


# --- Relationship Path Finding Functions ---

# Helper functions for fast_bidirectional_bfs


def _validate_bfs_inputs(
    start_id: str,
    end_id: str,
    id_to_parents: dict[str, set[str]] | None,
    id_to_children: dict[str, set[str]] | None,
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

    def get(self, start_id: str, end_id: str) -> list[str] | None:
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
    id_to_parents: dict[str, set[str]] | None,
    id_to_children: dict[str, set[str]] | None,
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


# ==============================================
# Module Tests
# ==============================================


def relationship_graph_module_tests() -> bool:
    """Test suite for relationship_graph.py - BFS pathfinding and cache."""
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Relationship Graph", "relationship_graph.py")
        suite.start_suite()

    def test_bfs_basic():
        """Test basic BFS pathfinding."""
        id_to_parents = {
            "child1": {"parent1", "parent2"},
            "grandchild": {"child1"},
        }
        id_to_children = {
            "parent1": {"child1"},
            "parent2": {"child1"},
            "child1": {"grandchild"},
        }

        path = fast_bidirectional_bfs("parent1", "grandchild", id_to_parents, id_to_children)
        assert path is not None, "Should find path from parent1 to grandchild"
        assert len(path) >= 2, "Path should have at least 2 nodes"
        assert path[0] == "parent1", "Path should start with parent1"
        assert path[-1] == "grandchild", "Path should end with grandchild"

    def test_bfs_same_person():
        """Test BFS with same start and end."""
        path = fast_bidirectional_bfs("@I001@", "@I001@", {}, {})
        assert len(path) == 1, "Same person path should have length 1"
        assert path[0] == "@I001@", "Same person path should contain only that person"

    def test_cache_functionality():
        """Test relationship path cache."""
        clear_relationship_cache()
        stats = get_relationship_cache_stats()
        assert stats["total_queries"] == 0, "Cache should start empty"
        assert stats["size"] == 0, "Cache should have zero entries"

        id_to_parents = {"@I002@": {"@I001@"}}
        id_to_children = {"@I001@": {"@I002@"}}

        fast_bidirectional_bfs("@I001@", "@I002@", id_to_parents, id_to_children)
        stats = get_relationship_cache_stats()
        assert stats["misses"] == 1, "First query should be a miss"

        fast_bidirectional_bfs("@I001@", "@I002@", id_to_parents, id_to_children)
        stats = get_relationship_cache_stats()
        assert stats["hits"] == 1, "Second query should be a hit"

        clear_relationship_cache()

    suite.run_test("BFS basic pathfinding", test_bfs_basic)
    suite.run_test("BFS same person", test_bfs_same_person)
    suite.run_test("Cache functionality", test_cache_functionality)

    return suite.finish_suite()


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(relationship_graph_module_tests)

# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    print("🔍 Running Relationship Graph test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
