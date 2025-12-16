"""
Research Suggestion Generation for Genealogical Conversations.

Phase 5.2: Research Assistant Features
Generates relevant research suggestions based on conversation context, common ancestors,
geographic locations, time periods, and available Ancestry collections.

Author: Wayne Gault
Created: October 21, 2025
"""

from __future__ import annotations

# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Ancestry Collection Mappings by Location
ANCESTRY_COLLECTIONS = {
    "scotland": [
        "Scotland, Select Births and Baptisms, 1564-1950",
        "Scotland, Select Marriages, 1561-1910",
        "Scotland, Select Deaths and Burials, 1538-1854",
        "Scotland Census Collection (1841-1911)",
        "Scotland, Old Parish Registers, 1538-1854",
        "Scotland, Valuation Rolls, 1855-1955",
    ],
    "england": [
        "England & Wales, Civil Registration Birth Index, 1837-1915",
        "England & Wales, Civil Registration Marriage Index, 1837-1915",
        "England & Wales, Civil Registration Death Index, 1837-1915",
        "England Census Collection (1841-1911)",
        "England, Select Births and Christenings, 1538-1975",
        "England, Select Marriages, 1538-1973",
    ],
    "ireland": [
        "Ireland, Civil Registration Births Index, 1864-1958",
        "Ireland, Civil Registration Marriages Index, 1845-1958",
        "Ireland, Civil Registration Deaths Index, 1864-1958",
        "Ireland Census Collection (1901, 1911)",
        "Ireland, Catholic Parish Registers, 1655-1915",
        "Griffith's Valuation, 1847-1864",
    ],
    "canada": [
        "Canada Census Collection (1851-1921)",
        "Canada, Births and Baptisms, 1661-1959",
        "Canada, Marriages, 1661-1949",
        "Canada, Deaths and Burials, 1800-1999",
        "Canadian Passenger Lists, 1865-1935",
    ],
    "usa": [
        "U.S. Census Collection (1790-1950)",
        "U.S., Social Security Death Index, 1935-2014",
        "U.S., World War I Draft Registration Cards, 1917-1918",
        "U.S., World War II Draft Registration Cards, 1942",
        "U.S., Naturalization Records, 1840-1957",
    ],
}

# Time Period Specific Collections
TIME_PERIOD_COLLECTIONS = {
    "1800s": [
        "19th Century Census Records",
        "Parish Registers and Vital Records",
        "Immigration and Emigration Records",
    ],
    "1900s": [
        "20th Century Census Records",
        "Military Records (WWI, WWII)",
        "Social Security and Death Records",
    ],
}

# DNA Ethnicity Region to Research Area Mappings (Phase 5.1)
ETHNICITY_RESEARCH_SUGGESTIONS = {
    "scotland": {
        "regions": ["Scottish Highlands", "Scottish Lowlands", "Hebrides", "Shetland", "Orkney"],
        "collections": [
            "Scotland, Old Parish Registers, 1538-1854",
            "Scotland Census Collection (1841-1911)",
        ],
        "surname_clusters": ["MacLeod", "Campbell", "Stewart", "Robertson", "MacDonald"],
        "strategies": [
            "Research Scottish clan histories for surname connections",
            "Check Kirk Session records for family events",
        ],
    },
    "ireland": {
        "regions": ["Ulster", "Munster", "Leinster", "Connacht"],
        "collections": [
            "Ireland, Catholic Parish Registers, 1655-1915",
            "Griffith's Valuation, 1847-1864",
        ],
        "surname_clusters": ["Murphy", "Kelly", "O'Sullivan", "Walsh", "O'Brien"],
        "strategies": [
            "Research townland-level records for Irish ancestors",
            "Check Griffith's Valuation for property records",
        ],
    },
    "england": {
        "regions": ["Cornwall", "Devon", "Yorkshire", "Lancashire", "East Anglia"],
        "collections": [
            "England, Select Births and Christenings, 1538-1975",
            "England Census Collection (1841-1911)",
        ],
        "surname_clusters": ["Smith", "Jones", "Taylor", "Brown", "Wilson"],
        "strategies": [
            "Check quarter session records for English ancestors",
            "Research nonconformist church records if not in parish registers",
        ],
    },
    "wales": {
        "regions": ["North Wales", "South Wales", "Mid Wales"],
        "collections": [
            "Wales, Parish Registers, 1533-1911",
            "Wales Census Collection (1841-1911)",
        ],
        "surname_clusters": ["Jones", "Williams", "Davies", "Evans", "Thomas"],
        "strategies": [
            "Be aware of patronymic naming patterns (ap/ab = son of)",
            "Check Welsh-language records and place names",
        ],
    },
    "germany": {
        "regions": ["Bavaria", "Prussia", "Saxony", "Rhineland", "Westphalia"],
        "collections": [
            "Germany, Lutheran Baptisms, Marriages, and Burials, 1500-1971",
            "German Immigrants to the US, 1850-1897",
        ],
        "surname_clusters": ["Mueller", "Schmidt", "Schneider", "Fischer", "Weber"],
        "strategies": [
            "Research Kirchenbücher (church books) for vital records",
            "Check German emigration lists for US-bound ancestors",
        ],
    },
    "norway": {
        "regions": ["Bergen", "Oslo", "Trondheim", "Stavanger"],
        "collections": [
            "Norway, Parish Registers, 1812-1938",
            "Norway Census and Population Lists, 1664-1900",
        ],
        "surname_clusters": ["Hansen", "Olsen", "Johansen", "Larsen", "Andersen"],
        "strategies": [
            "Use bygdebøker (farm history books) for rural ancestors",
            "Be aware of patronymic naming (son/datter) until ~1900",
        ],
    },
    "sweden": {
        "regions": ["Stockholm", "Göteborg", "Malmö", "Uppsala"],
        "collections": [
            "Sweden, Church Records, 1451-1943",
            "Sweden, Emigration Records, 1783-1951",
        ],
        "surname_clusters": ["Andersson", "Johansson", "Karlsson", "Nilsson", "Eriksson"],
        "strategies": [
            "Check husförhörslängder (household examination records)",
            "Research soldier rolls for military ancestors",
        ],
    },
}


def _extract_ethnicity_suggestions(ethnicity_regions: list[str]) -> dict[str, Any]:
    """
    Extract research suggestions based on DNA ethnicity regions.

    Phase 5.1: Ethnicity-Based Suggestions
    Maps DNA ethnicity regions to relevant collections, surname clusters,
    and research strategies.

    Args:
        ethnicity_regions: List of shared ethnicity region names

    Returns:
        Dict with collections, surname_clusters, and strategies
    """
    result: dict[str, Any] = {
        "collections": [],
        "surname_clusters": [],
        "strategies": [],
        "regions": [],
    }

    for region in ethnicity_regions:
        region_lower = region.lower()
        # Match against known ethnicity mappings
        for eth_key, eth_data in ETHNICITY_RESEARCH_SUGGESTIONS.items():
            if eth_key in region_lower or any(r.lower() in region_lower for r in eth_data["regions"]):
                result["collections"].extend(eth_data["collections"])
                result["surname_clusters"].extend(eth_data["surname_clusters"][:3])
                result["strategies"].extend(eth_data["strategies"])
                result["regions"].append(eth_key.title())
                break

    # Deduplicate
    result["collections"] = list(dict.fromkeys(result["collections"]))[:4]
    result["surname_clusters"] = list(dict.fromkeys(result["surname_clusters"]))[:5]
    result["strategies"] = list(dict.fromkeys(result["strategies"]))[:2]
    result["regions"] = list(dict.fromkeys(result["regions"]))

    return result


def _extract_location_collections(locations: list[str]) -> list[str]:
    """Extract Ancestry collections based on geographic locations."""
    collections: list[str] = []
    for location in locations:
        location_lower = location.lower()
        for region, region_collections in ANCESTRY_COLLECTIONS.items():
            if region in location_lower:
                collections.extend(region_collections[:3])  # Top 3 per region
                break
    return collections


def _extract_time_period_collections(time_periods: list[str]) -> list[str]:
    """Extract Ancestry collections based on time periods."""
    collections: list[str] = []
    for period in time_periods:
        try:
            year = int(period) if isinstance(period, str) and period.isdigit() else None
            if year:
                if 1800 <= year < 1900:
                    collections.extend(TIME_PERIOD_COLLECTIONS["1800s"])
                elif 1900 <= year < 2000:
                    collections.extend(TIME_PERIOD_COLLECTIONS["1900s"])
        except (ValueError, TypeError):
            continue
    return collections


def _generate_record_types(common_ancestors: list[dict[str, Any]]) -> list[str]:
    """Generate specific record type suggestions based on common ancestors."""
    record_types: list[str] = []
    if common_ancestors:
        for ancestor in common_ancestors[:2]:  # Focus on top 2 ancestors
            name = ancestor.get("name", "Unknown")
            birth_year = ancestor.get("birth_year")
            birth_place = ancestor.get("birth_place", "")

            if birth_year:
                record_types.append(f"Birth/baptism records for {name} around {birth_year}")
            if birth_place:
                record_types.append(f"Census records in {birth_place}")
    return record_types


def _generate_strategies(
    relationship_context: Optional[str],
    locations: list[str],
    common_ancestors: list[dict[str, Any]],
) -> list[str]:
    """Generate research strategy suggestions."""
    strategies: list[str] = []

    if relationship_context:
        strategies.append(f"Focus on the common ancestor line connecting you as {relationship_context}")

    if locations:
        primary_location = locations[0]
        strategies.append(f"Search parish registers and civil records in {primary_location}")

    if common_ancestors:
        strategies.append("Use DNA matches to verify the relationship path to common ancestors")

    return strategies


def generate_research_suggestions(
    common_ancestors: list[dict[str, Any]],
    locations: list[str],
    time_periods: list[str],
    relationship_context: Optional[str] = None,
    ethnicity_regions: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Generate relevant research suggestions based on conversation context.

    Phase 5.1 & 5.2: Research Assistant Features
    Creates personalized research suggestions including relevant Ancestry collections,
    specific record types, research strategies, and ethnicity-based suggestions.

    Args:
        common_ancestors: List of common ancestor dicts with name, birth_year, birth_place
        locations: List of geographic locations mentioned in conversation
        time_periods: List of time periods (years or decades) mentioned
        relationship_context: Optional context about the relationship (e.g., "3rd cousin")
        ethnicity_regions: Optional list of shared DNA ethnicity regions

    Returns:
        Dictionary containing:
        - collections: List of relevant Ancestry collections
        - record_types: List of specific record types to search
        - research_strategies: List of research strategy suggestions
        - formatted_message: Ready-to-use message text
        - ethnicity_suggestions: Dict with ethnicity-specific research tips (Phase 5.1)
    """
    # Extract collections from locations and time periods
    collections = _extract_location_collections(locations)
    collections.extend(_extract_time_period_collections(time_periods))

    # Generate record types and strategies
    record_types = _generate_record_types(common_ancestors)
    strategies = _generate_strategies(relationship_context, locations, common_ancestors)

    # Phase 5.1: Add ethnicity-based suggestions
    ethnicity_suggestions: dict[str, Any] = {}
    if ethnicity_regions:
        ethnicity_suggestions = _extract_ethnicity_suggestions(ethnicity_regions)
        # Merge ethnicity collections with location collections
        collections.extend(ethnicity_suggestions.get("collections", []))
        # Add ethnicity-specific strategies
        strategies.extend(ethnicity_suggestions.get("strategies", []))

    # Remove duplicates and limit results
    collections = list(dict.fromkeys(collections))[:5]  # Top 5 unique collections
    record_types = list(dict.fromkeys(record_types))[:3]  # Top 3 unique record types
    strategies = list(dict.fromkeys(strategies))[:3]  # Top 3 unique strategies

    # Format message
    formatted_message = _format_research_suggestion_message(
        collections, record_types, strategies, ethnicity_suggestions
    )

    return {
        "collections": collections,
        "record_types": record_types,
        "research_strategies": strategies,
        "formatted_message": formatted_message,
        "ethnicity_suggestions": ethnicity_suggestions,
    }


def _format_research_suggestion_message(
    collections: list[str],
    record_types: list[str],
    strategies: list[str],
    ethnicity_suggestions: Optional[dict[str, Any]] = None,
) -> str:
    """
    Format research suggestions into a human-readable message.

    Args:
        collections: List of Ancestry collections
        record_types: List of specific record types
        strategies: List of research strategies
        ethnicity_suggestions: Optional ethnicity-based suggestions (Phase 5.1)

    Returns:
        Formatted message text
    """
    message_parts: list[str] = []

    def _append_collection_section() -> None:
        if not (collections or record_types):
            return
        message_parts.append("Based on our connection, you might find these records helpful:")
        if collections:
            message_parts.append("\n\nRelevant Ancestry Collections:")
            message_parts.extend([f"  • {collection}" for collection in collections])
        if record_types:
            message_parts.append("\n\nSpecific Records to Search:")
            message_parts.extend([f"  • {record_type}" for record_type in record_types])

    def _append_ethnicity_section() -> None:
        if not ethnicity_suggestions:
            return
        regions = ethnicity_suggestions.get("regions", [])
        surname_clusters = ethnicity_suggestions.get("surname_clusters", [])
        if regions:
            message_parts.append(f"\n\nShared Ethnicity Regions: {', '.join(regions)}")
        if surname_clusters:
            message_parts.append("\n\nCommon Surname Clusters to Watch For:")
            message_parts.extend([f"  • {surname}" for surname in surname_clusters[:3]])

    def _append_strategy_section() -> None:
        if not strategies:
            return
        message_parts.append("\n\nResearch Strategies:")
        message_parts.extend([f"  • {strategy}" for strategy in strategies])

    _append_collection_section()
    _append_ethnicity_section()
    _append_strategy_section()

    if message_parts:
        message_parts.append("\n\nWould you like me to share specific records from my tree?")
        return "".join(message_parts)

    return ""


# ==============================================
# TESTS
# ==============================================


def _test_basic_research_suggestions() -> bool:
    """Test basic research suggestion generation."""
    common_ancestors = [{"name": "John Smith", "birth_year": "1850", "birth_place": "Aberdeen, Scotland"}]
    locations = ["Aberdeen, Scotland"]
    time_periods = ["1850"]

    result = generate_research_suggestions(common_ancestors, locations, time_periods, "3rd cousin")

    assert isinstance(result, dict), "Should return dictionary"
    assert "collections" in result, "Should have collections"
    assert "record_types" in result, "Should have record_types"
    assert "research_strategies" in result, "Should have research_strategies"
    assert "formatted_message" in result, "Should have formatted_message"

    assert len(result["collections"]) > 0, "Should suggest at least one collection"
    assert isinstance(result["collections"], list), "Collections should be a list"
    assert all(isinstance(c, str) for c in result["collections"]), "Each collection should be a string"

    assert len(result["record_types"]) > 0, "Should suggest at least one record type"
    assert len(result["research_strategies"]) > 0, "Should suggest at least one strategy"

    assert "Scotland" in result["formatted_message"], "Should mention Scotland"
    assert "1850" in result["formatted_message"], "Should mention time period"

    logger.info("✓ Basic research suggestions work correctly")
    return True


def _test_multiple_locations() -> bool:
    """Test research suggestions with multiple locations."""
    common_ancestors = []
    locations = ["Scotland", "Ireland", "England"]
    time_periods = []

    result = generate_research_suggestions(common_ancestors, locations, time_periods)

    assert len(result["collections"]) > 0, "Should suggest collections for multiple locations"

    # Should include collections from all three regions
    message = result["formatted_message"]
    assert any(region in message for region in ["Scotland", "Ireland", "England"]), "Should mention at least one region"

    logger.info("✓ Multiple location suggestions work correctly")
    return True


def _test_time_period_collections() -> bool:
    """Test time period specific collection suggestions."""
    common_ancestors = []
    locations = []
    time_periods = ["1850", "1920"]

    result = generate_research_suggestions(common_ancestors, locations, time_periods)

    assert len(result["collections"]) > 0, "Should suggest time-period collections"
    assert any("Census" in col for col in result["collections"]), "Should suggest census records"

    logger.info("✓ Time period collection suggestions work correctly")
    return True


def _test_empty_input() -> bool:
    """Test research suggestions with empty input."""
    result = generate_research_suggestions([], [], [])

    assert isinstance(result, dict), "Should return dictionary even with empty input"
    assert not result["formatted_message"], "Should return empty message for no data"

    logger.info("✓ Empty input handling works correctly")
    return True


# ==============================================
# Test Suite
# ==============================================


def _test_location_collections_extraction() -> bool:
    """Test location-based collection extraction."""
    locations = ["Scotland", "Ireland"]
    collections = _extract_location_collections(locations)

    assert isinstance(collections, list), "Should return list"
    assert len(collections) > 0, "Should extract collections for Scotland and Ireland"
    assert any("Scotland" in col for col in collections), "Should include Scotland collections"

    logger.info("✓ Location collections extraction works correctly")
    return True


def _test_time_period_collections_extraction() -> bool:
    """Test time period-based collection extraction."""
    time_periods = ["1850", "1920"]
    collections = _extract_time_period_collections(time_periods)

    assert isinstance(collections, list), "Should return list"
    assert len(collections) > 0, "Should extract collections for time periods"

    logger.info("✓ Time period collections extraction works correctly")
    return True


def _test_record_types_generation() -> bool:
    """Test record type generation from common ancestors."""
    common_ancestors = [{"name": "John Smith", "birth_year": "1850", "birth_place": "Aberdeen, Scotland"}]
    record_types = _generate_record_types(common_ancestors)

    assert isinstance(record_types, list), "Should return list"
    assert len(record_types) > 0, "Should generate record types"

    logger.info("✓ Record types generation works correctly")
    return True


def _test_strategies_generation() -> bool:
    """Test research strategies generation."""
    common_ancestors = [{"name": "John Smith", "birth_year": "1850", "birth_place": "Aberdeen, Scotland"}]
    locations = ["Aberdeen, Scotland"]
    strategies = _generate_strategies("3rd cousin", locations, common_ancestors)

    assert isinstance(strategies, list), "Should return list"
    assert len(strategies) > 0, "Should generate strategies"

    logger.info("✓ Research strategies generation works correctly")
    return True


def _test_complete_research_suggestions() -> bool:
    """Test complete research suggestions with all parameters."""
    common_ancestors = [
        {"name": "John Smith", "birth_year": "1850", "birth_place": "Aberdeen, Scotland"},
        {"name": "Mary Brown", "birth_year": "1855", "birth_place": "Banff, Scotland"},
    ]
    locations = ["Aberdeen, Scotland", "Banff, Scotland"]
    time_periods = ["1850", "1855"]

    result = generate_research_suggestions(common_ancestors, locations, time_periods, "2nd cousin")

    # Verify all sections present
    assert isinstance(result, dict), "Should return dictionary"
    assert "collections" in result, "Should have collections"
    assert "record_types" in result, "Should have record_types"
    assert "research_strategies" in result, "Should have research_strategies"
    assert "formatted_message" in result, "Should have formatted_message"

    # Verify content
    assert len(result["collections"]) > 0, "Should have collections"
    assert len(result["record_types"]) > 0, "Should have record types"
    assert len(result["research_strategies"]) > 0, "Should have strategies"
    assert len(result["formatted_message"]) > 0, "Should have formatted message"

    logger.info("✓ Complete research suggestions work correctly")
    return True


def _test_result_limits() -> bool:
    """Test that results are limited to reasonable numbers."""
    # Create many locations to test limiting
    locations = [f"Location{i}" for i in range(20)]
    time_periods = [str(1800 + i) for i in range(20)]
    common_ancestors = [
        {"name": f"Person{i}", "birth_year": str(1800 + i), "birth_place": f"Place{i}"} for i in range(20)
    ]

    result = generate_research_suggestions(common_ancestors, locations, time_periods)

    # Verify limits are applied
    assert len(result["collections"]) <= 5, "Should limit collections to 5"
    assert len(result["record_types"]) <= 3, "Should limit record types to 3"
    assert len(result["research_strategies"]) <= 3, "Should limit strategies to 3"

    logger.info("✓ Result limits work correctly")
    return True


def _test_formatted_message_structure() -> bool:
    """Test formatted message structure."""
    common_ancestors = [{"name": "John Smith", "birth_year": "1850", "birth_place": "Scotland"}]
    locations = ["Scotland"]
    time_periods = ["1850"]

    result = generate_research_suggestions(common_ancestors, locations, time_periods)

    message = result["formatted_message"]
    assert isinstance(message, str), "Message should be string"
    assert len(message) > 0, "Message should not be empty"

    logger.info("✓ Formatted message structure works correctly")
    return True


def _test_ethnicity_suggestions_extraction() -> bool:
    """Test ethnicity-based suggestion extraction."""
    ethnicity_regions = ["Scotland", "Ireland"]
    result = _extract_ethnicity_suggestions(ethnicity_regions)

    assert isinstance(result, dict), "Should return dictionary"
    assert "collections" in result, "Should have collections"
    assert "surname_clusters" in result, "Should have surname_clusters"
    assert "strategies" in result, "Should have strategies"
    assert len(result["collections"]) > 0, "Should extract collections for Scotland/Ireland"
    assert len(result["surname_clusters"]) > 0, "Should extract surname clusters"

    logger.info("✓ Ethnicity suggestions extraction works correctly")
    return True


def _test_ethnicity_in_research_suggestions() -> bool:
    """Test that ethnicity regions are included in research suggestions."""
    common_ancestors = []
    locations = []
    time_periods = []
    ethnicity_regions = ["Scottish Highlands", "Ireland"]

    result = generate_research_suggestions(
        common_ancestors, locations, time_periods, ethnicity_regions=ethnicity_regions
    )

    assert "ethnicity_suggestions" in result, "Should include ethnicity_suggestions"
    assert len(result["ethnicity_suggestions"]) > 0, "Should have ethnicity suggestions"
    assert "collections" in result["ethnicity_suggestions"], "Should have ethnicity collections"

    # Formatted message should mention ethnicity
    message = result["formatted_message"]
    assert len(message) > 0, "Should have formatted message"

    logger.info("✓ Ethnicity in research suggestions works correctly")
    return True


def _test_ethnicity_surname_clusters() -> bool:
    """Test that surname clusters are extracted from ethnicity."""
    ethnicity_regions = ["Scotland"]
    result = _extract_ethnicity_suggestions(ethnicity_regions)

    assert len(result["surname_clusters"]) > 0, "Should have surname clusters"
    # Check for Scottish surnames
    scottish_surnames = ["MacLeod", "Campbell", "Stewart", "Robertson", "MacDonald"]
    assert any(surname in result["surname_clusters"] for surname in scottish_surnames), (
        "Should include Scottish surnames"
    )

    logger.info("✓ Ethnicity surname clusters work correctly")
    return True


def research_suggestions_tests() -> bool:
    """Run all research suggestion tests."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("Research Suggestion Generation", "research_suggestions.py")
    suite.start_suite()

    with suppress_logging():
        # Helper function tests
        suite.run_test(
            "Location collections extraction",
            _test_location_collections_extraction,
            "Location collections extraction works correctly",
            "Test location-based collection extraction",
            "Verify _extract_location_collections() returns appropriate collections",
        )

        suite.run_test(
            "Time period collections extraction",
            _test_time_period_collections_extraction,
            "Time period collections extraction works correctly",
            "Test time period-based collection extraction",
            "Verify _extract_time_period_collections() returns appropriate collections",
        )

        suite.run_test(
            "Record types generation",
            _test_record_types_generation,
            "Record types generation works correctly",
            "Test record type generation from common ancestors",
            "Verify _generate_record_types() returns appropriate record types",
        )

        suite.run_test(
            "Research strategies generation",
            _test_strategies_generation,
            "Research strategies generation works correctly",
            "Test research strategies generation",
            "Verify _generate_strategies() returns appropriate strategies",
        )

        # Main function tests
        suite.run_test(
            "Basic research suggestions",
            _test_basic_research_suggestions,
            "Test generation of research suggestions with common ancestor and location",
            "Research suggestions provide relevant collections and strategies",
            "generate_research_suggestions() creates appropriate suggestions",
        )

        suite.run_test(
            "Complete research suggestions",
            _test_complete_research_suggestions,
            "Complete research suggestions work correctly",
            "Test complete research suggestions with all parameters",
            "Verify all sections present with multiple ancestors and locations",
        )

        suite.run_test(
            "Multiple location handling",
            _test_multiple_locations,
            "Test suggestions with multiple geographic locations",
            "Multiple locations generate diverse collection suggestions",
            "Collections from all mentioned regions are included",
        )

        suite.run_test(
            "Time period collections",
            _test_time_period_collections,
            "Test time-period specific collection suggestions",
            "Time periods trigger appropriate historical collections",
            "Census and era-specific records are suggested",
        )

        suite.run_test(
            "Result limits",
            _test_result_limits,
            "Result limits work correctly",
            "Test that results are limited to reasonable numbers",
            "Verify collections limited to 5, record types to 3, strategies to 3",
        )

        suite.run_test(
            "Formatted message structure",
            _test_formatted_message_structure,
            "Formatted message structure works correctly",
            "Test formatted message structure",
            "Verify message is non-empty string with proper formatting",
        )

        suite.run_test(
            "Empty input handling",
            _test_empty_input,
            "Test graceful handling of empty input",
            "Empty input returns valid but empty result structure",
            "No errors with missing data",
        )

        # Phase 5.1: Ethnicity-based suggestion tests
        suite.run_test(
            "Ethnicity suggestions extraction",
            _test_ethnicity_suggestions_extraction,
            "Ethnicity-based suggestion extraction works correctly",
            "Test _extract_ethnicity_suggestions() returns collections and surnames",
            "Verify ethnicity regions map to research suggestions",
        )

        suite.run_test(
            "Ethnicity in research suggestions",
            _test_ethnicity_in_research_suggestions,
            "Ethnicity regions are included in research suggestions",
            "Test generate_research_suggestions() with ethnicity_regions parameter",
            "Verify ethnicity_suggestions added to result",
        )

        suite.run_test(
            "Ethnicity surname clusters",
            _test_ethnicity_surname_clusters,
            "Surname clusters extracted from ethnicity regions",
            "Test that Scottish regions return Scottish surnames",
            "Verify surname cluster mapping works correctly",
        )

    return suite.finish_suite()


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(research_suggestions_tests)


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
