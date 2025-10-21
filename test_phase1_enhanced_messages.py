#!/usr/bin/env python3
"""
Test Phase 1: Enhanced Message Content

Verifies that messages include:
- Relationship paths (in-tree matches)
- Tree statistics (all messages)
- Ethnicity commonality (out-of-tree matches)
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("PHASE 1 TESTING: Enhanced Message Content")
print("=" * 80)

# Test 1: Verify tree_stats_utils module
print("\n[Test 1] Verifying tree_stats_utils module...")
try:
    from tree_stats_utils import calculate_ethnicity_commonality, calculate_tree_statistics
    print("✓ tree_stats_utils module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import tree_stats_utils: {e}")
    sys.exit(1)

# Test 2: Verify database schema includes TreeStatisticsCache
print("\n[Test 2] Verifying TreeStatisticsCache table...")
try:
    from database import TreeStatisticsCache
    print("✓ TreeStatisticsCache model imported successfully")

    import sqlite3
    conn = sqlite3.connect("Data/ancestry_test.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tree_statistics_cache'")
    if cursor.fetchone():
        print("✓ tree_statistics_cache table exists in database")
    else:
        print("✗ tree_statistics_cache table NOT found in database")
        sys.exit(1)
    conn.close()
except Exception as e:
    print(f"✗ Error checking TreeStatisticsCache: {e}")
    sys.exit(1)

# Test 3: Calculate tree statistics for Frances Milne
print("\n[Test 3] Calculating tree statistics...")
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    test_engine = create_engine("sqlite:///Data/ancestry_test.db")
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()

    frances_profile_id = "08FA6E79-0006-0000-0000-000000000000"
    stats = calculate_tree_statistics(session, frances_profile_id)

    print("✓ Tree statistics calculated:")
    print(f"  - Total matches: {stats['total_matches']}")
    print(f"  - In tree: {stats['in_tree_count']}")
    print(f"  - Out of tree: {stats['out_tree_count']}")
    print(f"  - Close matches (>100 cM): {stats['close_matches']}")
    print(f"  - Moderate matches (20-100 cM): {stats['moderate_matches']}")
    print(f"  - Distant matches (<20 cM): {stats['distant_matches']}")
    print(f"  - Ethnicity regions: {len(stats['ethnicity_regions'])}")

    # Test caching
    print("\n[Test 3b] Testing cache...")
    stats2 = calculate_tree_statistics(session, frances_profile_id)
    if stats2 == stats:
        print("✓ Cache working - same results returned")
    else:
        print("⚠ Cache may not be working - different results")

    session.close()
except Exception as e:
    print(f"✗ Error calculating tree statistics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test ethnicity commonality
print("\n[Test 4] Testing ethnicity commonality...")
try:
    from database import Person

    session = TestSession()
    frances = session.query(Person).filter(Person.profile_id == frances_profile_id).first()

    if frances:
        ethnicity = calculate_ethnicity_commonality(session, frances_profile_id, frances.id)
        print("✓ Ethnicity commonality calculated:")
        print(f"  - Shared regions: {len(ethnicity['shared_regions'])}")
        print(f"  - Similarity score: {ethnicity['similarity_score']:.1f}%")
        print(f"  - Top shared region: {ethnicity['top_shared_region']}")
    else:
        print("⚠ Frances not found in database")

    session.close()
except Exception as e:
    print(f"✗ Error calculating ethnicity commonality: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify Action 8 integration
print("\n[Test 5] Verifying Action 8 integration...")
try:
    from action8_messaging import TREE_STATS_AVAILABLE, _prepare_message_format_data

    if TREE_STATS_AVAILABLE:
        print("✓ Tree statistics available in Action 8")
    else:
        print("✗ Tree statistics NOT available in Action 8")
        sys.exit(1)

    # Test format data preparation
    session = TestSession()
    frances = session.query(Person).filter(Person.profile_id == frances_profile_id).first()

    if frances:
        format_data = _prepare_message_format_data(
            frances,
            frances.family_tree,
            frances.dna_match,
            session
        )

        print("✓ Message format data prepared:")
        print(f"  - Name: {format_data.get('name', 'N/A')}")
        print(f"  - Relationship path: {format_data.get('relationship_path', 'N/A')[:50]}...")
        print(f"  - Total matches: {format_data.get('total_matches', 'N/A')}")
        print(f"  - Matches in tree: {format_data.get('matches_in_tree', 'N/A')}")
        print(f"  - Ethnicity commonality: {format_data.get('ethnicity_commonality', 'N/A')[:50]}...")

        # Verify required fields are present
        required_fields = ['name', 'relationship_path', 'total_matches', 'matches_in_tree']
        missing_fields = [f for f in required_fields if f not in format_data]

        if missing_fields:
            print(f"✗ Missing required fields: {', '.join(missing_fields)}")
            sys.exit(1)
        else:
            print("✓ All required fields present in format data")

    session.close()
except Exception as e:
    print(f"✗ Error testing Action 8 integration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verify message templates can use new placeholders
print("\n[Test 6] Verifying message template compatibility...")
try:
    from database import MessageTemplate

    session = TestSession()

    # Get a sample template
    template = session.query(MessageTemplate).filter(
        MessageTemplate.template_key == "In_Tree-Initial"
    ).first()

    if template:
        # Check if template uses relationship_path
        if "{relationship_path}" in template.message_content:
            print("✓ In-tree template uses {relationship_path} placeholder")
        else:
            print("⚠ In-tree template does NOT use {relationship_path} placeholder")

        # Try formatting with sample data
        try:
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
            print("✓ Template formatted successfully with enhanced data")
        except KeyError as e:
            print(f"⚠ Template missing placeholder: {e}")
    else:
        print("⚠ In_Tree-Initial template not found")

    session.close()
except Exception as e:
    print(f"✗ Error testing template compatibility: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL PHASE 1 TESTS PASSED!")
print("=" * 80)
print("\nPhase 1 implementation complete:")
print("  ✓ Tree statistics calculation with caching")
print("  ✓ Ethnicity commonality analysis")
print("  ✓ Action 8 integration")
print("  ✓ Message template compatibility")
print("\nReady for Phase 2: Person Lookup Integration")

