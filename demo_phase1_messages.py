#!/usr/bin/env python3
"""
Phase 1 Demonstration: Enhanced Message Content

Shows actual messages with Phase 1 enhancements:
- Tree statistics
- Ethnicity commonality
- Relationship paths
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from action8_messaging import _prepare_message_format_data
from database import MessageTemplate, Person
from tree_stats_utils import calculate_ethnicity_commonality, calculate_tree_statistics

print("=" * 80)
print("PHASE 1 DEMONSTRATION: Enhanced Message Content")
print("=" * 80)

# Connect to test database
test_engine = create_engine("sqlite:///Data/ancestry_test.db")
TestSession = sessionmaker(bind=test_engine)
session = TestSession()

frances_profile_id = "08FA6E79-0006-0000-0000-000000000000"

# Get Frances from database
frances = session.query(Person).filter(
    Person.profile_id == frances_profile_id
).first()

if not frances:
    print("✗ Frances not found in test database")
    sys.exit(1)

print(f"\n📊 Testing with: {frances.username}")
print(f"   Profile ID: {frances.profile_id}")
print(f"   In my tree: {frances.in_my_tree}")

# ============================================================================
# PART 1: Tree Statistics
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: Tree Statistics Calculation")
print("=" * 80)

stats = calculate_tree_statistics(session, frances_profile_id)

print("\n✓ Tree statistics calculated:")
print(f"   Total DNA matches: {stats['total_matches']}")
print(f"   Matches in tree: {stats['in_tree_count']}")
print(f"   Matches out of tree: {stats['out_tree_count']}")
print(f"   Close matches (>100 cM): {stats['close_matches']}")
print(f"   Moderate matches (20-100 cM): {stats['moderate_matches']}")
print(f"   Distant matches (<20 cM): {stats['distant_matches']}")
print(f"   Ethnicity regions tracked: {len(stats['ethnicity_regions'])}")

if stats['ethnicity_regions']:
    region_list = list(stats['ethnicity_regions'].keys()) if isinstance(stats['ethnicity_regions'], dict) else stats['ethnicity_regions']
    print(f"\n   Ethnicity regions: {', '.join(region_list[:5])}")
    if len(region_list) > 5:
        print(f"   ... and {len(region_list) - 5} more")

# ============================================================================
# PART 2: Ethnicity Commonality
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: Ethnicity Commonality Analysis")
print("=" * 80)

ethnicity = calculate_ethnicity_commonality(session, frances_profile_id, frances.id)

print("\n✓ Ethnicity commonality calculated:")
print(f"   Shared regions: {len(ethnicity['shared_regions'])}")
print(f"   Similarity score: {ethnicity['similarity_score']:.1f}%")
print(f"   Top shared region: {ethnicity['top_shared_region']}")

if ethnicity['shared_regions']:
    shared_list = ethnicity['shared_regions'][:5] if isinstance(ethnicity['shared_regions'], list) else list(ethnicity['shared_regions'])[:5]
    print(f"\n   Shared regions: {', '.join(shared_list)}")
    if len(ethnicity['shared_regions']) > 5:
        print(f"   ... and {len(ethnicity['shared_regions']) - 5} more")

# ============================================================================
# PART 3: Message Format Data
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: Message Format Data Preparation")
print("=" * 80)

format_data = _prepare_message_format_data(
    frances,
    frances.family_tree,
    frances.dna_match,
    session
)

print(f"\n✓ Format data prepared with {len(format_data)} fields:")
print(f"   name: {format_data.get('name', 'N/A')}")
print(f"   predicted_relationship: {format_data.get('predicted_relationship', 'N/A')}")
print(f"   actual_relationship: {format_data.get('actual_relationship', 'N/A')}")
print(f"   total_rows: {format_data.get('total_rows', 'N/A')}")
print(f"   total_matches: {format_data.get('total_matches', 'N/A')}")
print(f"   matches_in_tree: {format_data.get('matches_in_tree', 'N/A')}")
print(f"   matches_out_tree: {format_data.get('matches_out_tree', 'N/A')}")

relationship_path = format_data.get('relationship_path', 'N/A')
if len(relationship_path) > 100:
    print(f"   relationship_path: {relationship_path[:100]}...")
else:
    print(f"   relationship_path: {relationship_path}")

ethnicity_text = format_data.get('ethnicity_commonality', 'N/A')
if len(ethnicity_text) > 100:
    print(f"   ethnicity_commonality: {ethnicity_text[:100]}...")
else:
    print(f"   ethnicity_commonality: {ethnicity_text}")

# ============================================================================
# PART 4: Sample Enhanced Messages
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: Sample Enhanced Messages")
print("=" * 80)

# Get in-tree template
in_tree_template = session.query(MessageTemplate).filter(
    MessageTemplate.template_key == "In_Tree-Initial"
).first()

if in_tree_template:
    print("\n" + "-" * 80)
    print("IN-TREE MESSAGE (with relationship path)")
    print("-" * 80)

    try:
        # Format the message
        message = in_tree_template.message_content.format(**format_data)

        # Show first 500 characters
        if len(message) > 500:
            print(message[:500])
            print(f"\n... ({len(message) - 500} more characters)")
        else:
            print(message)
    except KeyError as e:
        print(f"⚠ Template missing placeholder: {e}")
        print("\nTemplate content (first 300 chars):")
        print(in_tree_template.message_content[:300])

# Get out-of-tree template
out_tree_template = session.query(MessageTemplate).filter(
    MessageTemplate.template_key == "Out_Tree-Initial"
).first()

if out_tree_template:
    print("\n" + "-" * 80)
    print("OUT-OF-TREE MESSAGE (with tree statistics)")
    print("-" * 80)

    try:
        # Format the message
        message = out_tree_template.message_content.format(**format_data)

        # Show first 500 characters
        if len(message) > 500:
            print(message[:500])
            print(f"\n... ({len(message) - 500} more characters)")
        else:
            print(message)
    except KeyError as e:
        print(f"⚠ Template missing placeholder: {e}")
        print("\nTemplate content (first 300 chars):")
        print(out_tree_template.message_content[:300])

# ============================================================================
# PART 5: Before/After Comparison
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: Before/After Comparison")
print("=" * 80)

print("\n📋 BEFORE Phase 1 (basic message):")
print("-" * 80)
print("""
Dear Frances,

I'm Wayne, and I'm excited to connect with you as a DNA match on Ancestry!

I've been researching my family tree for some time. Ancestry thinks you are
my parent/child. I'd love to hear from you and compare our family trees.

Warmest regards,
Wayne
""")

print("\n📋 AFTER Phase 1 (enhanced message):")
print("-" * 80)
print(f"""
Dear {format_data['name']},

I'm Wayne, and I'm excited to connect with you as a DNA match on Ancestry!

I've been researching my family tree for some time and have successfully
connected {format_data['total_rows']} DNA matches to my tree so far. Each
connection has helped corroborate and improve the accuracy of my tree.

Ancestry thinks you are my {format_data['predicted_relationship']}. I've
added you because I think you are my {format_data['actual_relationship']}.
I believe our connection to be:

{format_data['relationship_path'][:200]}...

Does this align with your research, or do you see another connection?

Warmest regards,
Wayne
""")

print("\n" + "=" * 80)
print("✅ PHASE 1 ENHANCEMENTS DEMONSTRATED")
print("=" * 80)

print("\nKey improvements:")
print("  ✓ Relationship paths show exact genealogical connection")
print("  ✓ Tree statistics show research credibility")
print("  ✓ Ethnicity commonality creates personal connection")
print("  ✓ Messages are more informative and engaging")

print("\nNext steps:")
print("  1. Review message quality - do they look good?")
print("  2. Test with production database (dry_run mode)")
print("  3. Verify statistics accuracy")
print("  4. Proceed to Phase 2 (Person Lookup Integration)")

session.close()

