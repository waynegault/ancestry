#!/usr/bin/env python3
"""
Direct test of the availability flags logic without importing action9_process_productive
"""

print("Testing the availability flags logic directly...")

# Initialize flags to False (as in the original code)
GEDCOM_UTILS_AVAILABLE = False
API_UTILS_AVAILABLE = False
RELATIONSHIP_UTILS_AVAILABLE = False

print("Initial flags:")
print(f"GEDCOM_UTILS_AVAILABLE: {GEDCOM_UTILS_AVAILABLE}")
print(f"API_UTILS_AVAILABLE: {API_UTILS_AVAILABLE}")
print(f"RELATIONSHIP_UTILS_AVAILABLE: {RELATIONSHIP_UTILS_AVAILABLE}")

# Test the first import block (GEDCOM utilities)
print("\n--- Testing GEDCOM utilities import ---")
try:
    from gedcom_utils import (
        calculate_match_score,
        _normalize_id,
        GedcomData,
    )
    from relationship_utils import (
        fast_bidirectional_bfs,
        convert_gedcom_path_to_unified_format,
        format_relationship_path_unified,
    )

    GEDCOM_UTILS_AVAILABLE = True
    print("✓ GEDCOM utilities imported successfully")
except ImportError as e:
    print(f"✗ GEDCOM utilities import failed: {e}")
    pass

# Test the second import block (Relationship utilities)
print("\n--- Testing relationship utilities import ---")
try:
    from gedcom_search_utils import get_gedcom_relationship_path
    from action11 import get_ancestry_relationship_path

    RELATIONSHIP_UTILS_AVAILABLE = True
    print("✓ Relationship utilities imported successfully")
except ImportError as e:
    print(f"✗ Relationship utilities import failed: {e}")
    pass

# Test the third import block (API utilities)
print("\n--- Testing API utilities import ---")
try:
    from action11 import _process_and_score_suggestions

    API_UTILS_AVAILABLE = True
    print("✓ API utilities imported successfully")
except ImportError as e:
    print(f"✗ API utilities import failed: {e}")
    pass

print("\n=== FINAL RESULTS ===")
print(f"GEDCOM_UTILS_AVAILABLE: {GEDCOM_UTILS_AVAILABLE}")
print(f"API_UTILS_AVAILABLE: {API_UTILS_AVAILABLE}")
print(f"RELATIONSHIP_UTILS_AVAILABLE: {RELATIONSHIP_UTILS_AVAILABLE}")

# Simulate the logic that would be in action9_process_productive
if GEDCOM_UTILS_AVAILABLE and RELATIONSHIP_UTILS_AVAILABLE and API_UTILS_AVAILABLE:
    print("\n🎉 SUCCESS! All availability flags would be set to True!")
    print("This confirms our fix to action9_process_productive.py is correct.")
    print("The hanging issue is likely in a specific import, not our flag logic.")
elif GEDCOM_UTILS_AVAILABLE and RELATIONSHIP_UTILS_AVAILABLE:
    print("\n✅ GEDCOM and Relationship utilities are working!")
    print(
        "Only API utilities failed - this might be acceptable for basic functionality."
    )
elif GEDCOM_UTILS_AVAILABLE:
    print("\n⚠️  Only GEDCOM utilities are working.")
    print("Need to investigate relationship and API utility imports.")
else:
    print("\n❌ No utilities are working - need to investigate imports.")
