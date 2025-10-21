#!/usr/bin/env python3
"""
Test Phase 1: Enhanced Message Content

Verifies that messages include:
- Relationship paths (in-tree matches)
- Tree statistics (all messages)
- Ethnicity commonality (out-of-tree matches)
"""

# === STANDARD IMPORTS ===
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from standard_imports import setup_module

logger = setup_module(globals(), __file__)

# === TEST FRAMEWORK ===
from test_framework import TestSuite


def test_tree_stats_module_import() -> None:
    """Test 1: Verify tree_stats_utils module can be imported."""
    from tree_stats_utils import calculate_ethnicity_commonality, calculate_tree_statistics

    assert calculate_tree_statistics is not None, "calculate_tree_statistics not found"
    assert calculate_ethnicity_commonality is not None, "calculate_ethnicity_commonality not found"
    logger.info("✓ tree_stats_utils module imported successfully")


def test_database_schema() -> None:
    """Test 2: Verify TreeStatisticsCache table exists in database."""
    import sqlite3

    from database import TreeStatisticsCache

    conn = sqlite3.connect("Data/ancestry_test.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tree_statistics_cache'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None, "tree_statistics_cache table NOT found in database"
    logger.info("✓ tree_statistics_cache table exists in database")

def test_tree_statistics_calculation() -> None:
    """Test 3: Calculate tree statistics for Frances Milne."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from tree_stats_utils import calculate_tree_statistics

    test_engine = create_engine("sqlite:///Data/ancestry_test.db")
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()

    frances_profile_id = "08FA6E79-0006-0000-0000-000000000000"
    stats = calculate_tree_statistics(session, frances_profile_id)

    assert stats is not None, "Tree statistics calculation returned None"
    assert 'total_matches' in stats, "Missing total_matches in statistics"
    assert 'in_tree_count' in stats, "Missing in_tree_count in statistics"
    assert 'ethnicity_regions' in stats, "Missing ethnicity_regions in statistics"

    logger.info(f"✓ Tree statistics calculated: {stats['total_matches']} total matches, {stats['in_tree_count']} in tree")

    # Test caching
    stats2 = calculate_tree_statistics(session, frances_profile_id)
    assert stats2 == stats, "Cache not working - different results returned"
    logger.info("✓ Cache working - same results returned")

    session.close()

def test_ethnicity_commonality() -> None:
    """Test 4: Test ethnicity commonality calculation."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database import Person
    from tree_stats_utils import calculate_ethnicity_commonality

    test_engine = create_engine("sqlite:///Data/ancestry_test.db")
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()

    frances_profile_id = "08FA6E79-0006-0000-0000-000000000000"
    frances = session.query(Person).filter(Person.profile_id == frances_profile_id).first()

    assert frances is not None, "Frances not found in database"

    ethnicity = calculate_ethnicity_commonality(session, frances_profile_id, frances.id)

    assert ethnicity is not None, "Ethnicity commonality calculation returned None"
    assert 'shared_regions' in ethnicity, "Missing shared_regions in ethnicity data"
    assert 'similarity_score' in ethnicity, "Missing similarity_score in ethnicity data"

    logger.info(f"✓ Ethnicity commonality calculated: {len(ethnicity['shared_regions'])} shared regions, {ethnicity['similarity_score']:.1f}% similarity")

    session.close()

def test_action8_integration() -> None:
    """Test 5: Verify Action 8 integration with tree statistics."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from action8_messaging import TREE_STATS_AVAILABLE, _prepare_message_format_data
    from database import Person

    assert TREE_STATS_AVAILABLE, "Tree statistics NOT available in Action 8"
    logger.info("✓ Tree statistics available in Action 8")

    # Test format data preparation
    test_engine = create_engine("sqlite:///Data/ancestry_test.db")
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()

    frances_profile_id = "08FA6E79-0006-0000-0000-000000000000"
    frances = session.query(Person).filter(Person.profile_id == frances_profile_id).first()

    assert frances is not None, "Frances not found in database"

    format_data = _prepare_message_format_data(
        frances,
        frances.family_tree,
        frances.dna_match,
        session
    )

    # Verify required fields are present
    required_fields = ['name', 'relationship_path', 'total_matches', 'matches_in_tree']
    missing_fields = [f for f in required_fields if f not in format_data]

    assert not missing_fields, f"Missing required fields: {', '.join(missing_fields)}"
    logger.info("✓ All required fields present in format data")

    session.close()

def test_message_template_compatibility() -> None:
    """Test 6: Verify message templates can use new placeholders."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database import MessageTemplate

    test_engine = create_engine("sqlite:///Data/ancestry_test.db")
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()

    # Get a sample template
    template = session.query(MessageTemplate).filter(
        MessageTemplate.template_key == "In_Tree-Initial"
    ).first()

    assert template is not None, "In_Tree-Initial template not found"

    # Check if template uses relationship_path
    assert "{relationship_path}" in template.message_content, "In-tree template does NOT use {relationship_path} placeholder"
    logger.info("✓ In-tree template uses {relationship_path} placeholder")

    # Try formatting with sample data
    sample_data = {
        "name": "Test Person",
        "predicted_relationship": "4th cousin",
        "actual_relationship": "4th cousin",
        "relationship_path": "Test → Parent → Grandparent → Great-grandparent",
        "total_rows": 100,
        "total_matches": 500,
        "matches_in_tree": 100,
        "ethnicity_commonality": "We both have Scottish ancestry"
    }

    formatted = template.message_content.format(**sample_data)
    assert formatted is not None, "Template formatting failed"
    logger.info("✓ Template formatted successfully with enhanced data")

    session.close()


def phase1_module_tests() -> bool:
    """Run all Phase 1 tests using TestSuite framework."""
    suite = TestSuite("Phase 1: Enhanced Message Content", "test_phase1_enhanced_messages.py")
    suite.start_suite()

    suite.run_test(
        "Tree Stats Module Import",
        test_tree_stats_module_import,
        "Verifies tree_stats_utils module can be imported",
        "Test tree_stats_utils module import",
        "Import calculate_tree_statistics and calculate_ethnicity_commonality functions"
    )

    suite.run_test(
        "Database Schema",
        test_database_schema,
        "Verifies TreeStatisticsCache table exists in database",
        "Test TreeStatisticsCache table exists",
        "Check tree_statistics_cache table in Data/ancestry_test.db"
    )

    suite.run_test(
        "Tree Statistics Calculation",
        test_tree_statistics_calculation,
        "Calculates tree statistics for Frances Milne and tests caching",
        "Test tree statistics calculation and caching",
        "Calculate stats for Frances Milne profile and verify cache works"
    )

    suite.run_test(
        "Ethnicity Commonality",
        test_ethnicity_commonality,
        "Tests ethnicity commonality calculation",
        "Test ethnicity commonality calculation",
        "Calculate ethnicity commonality for Frances Milne"
    )

    suite.run_test(
        "Action 8 Integration",
        test_action8_integration,
        "Verifies Action 8 integration with tree statistics",
        "Test Action 8 integration",
        "Verify TREE_STATS_AVAILABLE and _prepare_message_format_data works"
    )

    suite.run_test(
        "Message Template Compatibility",
        test_message_template_compatibility,
        "Verifies message templates can use new placeholders",
        "Test message template compatibility",
        "Verify In_Tree-Initial template uses {relationship_path} placeholder"
    )

    return suite.finish_suite()


if __name__ == "__main__":
    import sys

    logger.info("🧪 Running Phase 1 comprehensive test suite...")
    try:
        success = phase1_module_tests()
    except Exception:
        logger.error("\n[ERROR] Unhandled exception during Phase 1 tests:")
        import traceback
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)

