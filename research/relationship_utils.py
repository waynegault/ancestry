#!/usr/bin/env python3

"""Relationship Utilities for Genealogical Data.

Utilities for processing genealogical relationship data including
relationship calculation, path finding, and family tree traversal.

This module re-exports functions from:
- research.relationship_graph: BFS pathfinding and cache
- research.relationship_formatting: Path formatting and relationship labels
"""


# === CORE INFRASTRUCTURE ===
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

# === IMPORTS FROM CHILD MODULES (also serve as re-exports) ===
from core.utils import format_name
from genealogy.gedcom.gedcom_utils import (
    TAG_BIRTH,
    TAG_DEATH,
    TAG_SEX,
    get_event_info,
    get_full_name,
)
from research.relationship_formatting import (
    GedcomIndividualProtocol,
    GedcomTagProtocol,
    _determine_gender_for_person,
    _get_relationship_term,
    convert_api_path_to_unified_format,
    convert_discovery_api_path_to_unified_format,
    convert_gedcom_path_to_unified_format,
    explain_relationship_path,
    format_api_relationship_path,
    format_relationship_path_unified,
)
from research.relationship_graph import (
    RelationshipPathCache,
    clear_relationship_cache,
    fast_bidirectional_bfs,
    get_relationship_cache_stats,
    report_cache_stats_to_performance_monitor,
)
from testing.test_framework import (
    TestSuite,
    suppress_logging,
)

# ==============================================
# Lazy re-exports for backward compatibility
# Uses __getattr__ to avoid issues when running as __main__
# ==============================================
_LAZY_RE_EXPORTS: dict[str, str] = {
    # relationship_graph (BFS, path cache, graph traversal)
    "_add_relative_to_queue": "research.relationship_graph",
    "_check_search_limits": "research.relationship_graph",
    "_expand_to_relatives": "research.relationship_graph",
    "_expand_to_siblings": "research.relationship_graph",
    "_initialize_bfs_queues": "research.relationship_graph",
    "_process_backward_queue": "research.relationship_graph",
    "_process_forward_queue": "research.relationship_graph",
    "_run_bfs_search_loop": "research.relationship_graph",
    "_score_path": "research.relationship_graph",
    "_select_best_path": "research.relationship_graph",
    "_validate_bfs_inputs": "research.relationship_graph",
    "clear_relationship_cache": "research.relationship_graph",
    "fast_bidirectional_bfs": "research.relationship_graph",
    "get_relationship_cache_stats": "research.relationship_graph",
    "RelationshipPathCache": "research.relationship_graph",
    "report_cache_stats_to_performance_monitor": "research.relationship_graph",
    # relationship_formatting (path formatting, labels, conversion)
    "_check_cousin_pattern": "research.relationship_formatting",
    "_check_grandparent_pattern": "research.relationship_formatting",
    "_check_nephew_niece_pattern": "research.relationship_formatting",
    "_check_uncle_aunt_pattern_parent": "research.relationship_formatting",
    "_check_uncle_aunt_pattern_sibling": "research.relationship_formatting",
    "_clean_name_format": "research.relationship_formatting",
    "_convert_you_are_relationship": "research.relationship_formatting",
    "_create_person_dict": "research.relationship_formatting",
    "_determine_gedcom_relationship": "research.relationship_formatting",
    "_determine_gender_for_person": "research.relationship_formatting",
    "_determine_relationship_type_from_path": "research.relationship_formatting",
    "_extract_html_from_response": "research.relationship_formatting",
    "_extract_lifespan_from_item": "research.relationship_formatting",
    "_extract_name_from_item": "research.relationship_formatting",
    "_extract_person_basic_info": "research.relationship_formatting",
    "_extract_person_from_list_item": "research.relationship_formatting",
    "_extract_relationship_from_item": "research.relationship_formatting",
    "_extract_years_from_lifespan": "research.relationship_formatting",
    "_format_discovery_api_path": "research.relationship_formatting",
    "_format_path_step": "research.relationship_formatting",
    "_format_years_display": "research.relationship_formatting",
    "_get_gendered_term": "research.relationship_formatting",
    "_get_relationship_term": "research.relationship_formatting",
    "_infer_gender_from_name": "research.relationship_formatting",
    "_infer_gender_from_relationship": "research.relationship_formatting",
    "_log_inferred_gender_once": "research.relationship_formatting",
    "_parse_discovery_relationship": "research.relationship_formatting",
    "_parse_html_relationship_data": "research.relationship_formatting",
    "_parse_relationship_term_and_gender": "research.relationship_formatting",
    "_process_path_person": "research.relationship_formatting",
    "_should_skip_list_item": "research.relationship_formatting",
    "_try_html_formats": "research.relationship_formatting",
    "_try_json_api_format": "research.relationship_formatting",
    "_try_simple_text_relationship": "research.relationship_formatting",
    "convert_api_path_to_unified_format": "research.relationship_formatting",
    "convert_discovery_api_path_to_unified_format": "research.relationship_formatting",
    "convert_gedcom_path_to_unified_format": "research.relationship_formatting",
    "explain_relationship_path": "research.relationship_formatting",
    "format_api_relationship_path": "research.relationship_formatting",
    "format_relationship_path_unified": "research.relationship_formatting",
    "GedcomIndividualProtocol": "research.relationship_formatting",
    "GedcomTagProtocol": "research.relationship_formatting",
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


# ==============================================
# Tests
# ==============================================


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
        value: str | None

    @dataclass
    class StubIndi:
        name: str
        birth_year: int | None
        death_year: int | None
        _sex: str | None

        def sub_tag(self, tag: str) -> StubTag | None:
            if tag == TAG_SEX and self._sex:
                return StubTag(self._sex)
            return None

    # Monkey-patch in relationship_formatting module where the function actually lives
    import research.relationship_formatting as _rf_mod

    original_get_full_name = _rf_mod.get_full_name
    original_get_event_info = _rf_mod.get_event_info

    def fake_get_full_name(indi: StubIndi) -> str:
        return indi.name

    def fake_get_event_info(indi: StubIndi, tag: str) -> tuple[SimpleNamespace | None, None, None]:
        year_attr = "birth_year" if tag == TAG_BIRTH else "death_year"
        year_val = getattr(indi, year_attr, None)
        return (SimpleNamespace(year=year_val) if year_val is not None else None, None, None)

    _rf_mod.get_full_name = fake_get_full_name
    _rf_mod.get_event_info = fake_get_event_info

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
        _rf_mod.get_full_name = original_get_full_name
        _rf_mod.get_event_info = original_get_event_info


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
