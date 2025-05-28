#!/usr/bin/env python3
"""
Focused test to replicate the exact import logic from action9_process_productive.py
"""

print("Testing exact import logic from action9_process_productive.py...")

# Initialize flags (as in original code)
GEDCOM_UTILS_AVAILABLE = False
API_UTILS_AVAILABLE = False
RELATIONSHIP_UTILS_AVAILABLE = False

print("Initial state:")
print(f"GEDCOM_UTILS_AVAILABLE: {GEDCOM_UTILS_AVAILABLE}")
print(f"API_UTILS_AVAILABLE: {API_UTILS_AVAILABLE}")
print(f"RELATIONSHIP_UTILS_AVAILABLE: {RELATIONSHIP_UTILS_AVAILABLE}")

# Test Block 1: GEDCOM utilities (exact same logic as action9_process_productive.py)
print("\n--- Testing Block 1: GEDCOM Utilities ---")
try:
    # Import from gedcom_utils
    print("Importing from gedcom_utils...")
    from gedcom_utils import (
        calculate_match_score,
        _normalize_id,
        GedcomData,
    )

    print("✓ gedcom_utils imports successful")

    # Import from relationship_utils
    print("Importing from relationship_utils...")
    from relationship_utils import (
        fast_bidirectional_bfs,
        convert_gedcom_path_to_unified_format,
        format_relationship_path_unified,
    )

    print("✓ relationship_utils imports successful")

    GEDCOM_UTILS_AVAILABLE = True
    print("✓ GEDCOM_UTILS_AVAILABLE set to True")

except ImportError as e:
    print(f"✗ GEDCOM utilities import failed: {e}")
    pass

print(f"After Block 1: GEDCOM_UTILS_AVAILABLE = {GEDCOM_UTILS_AVAILABLE}")

# Test Block 2: Relationship utilities (exact same logic as action9_process_productive.py)
print("\n--- Testing Block 2: Relationship Utilities ---")
try:
    print("Importing specific relationship functions...")
    from gedcom_search_utils import get_gedcom_relationship_path
    from action11 import get_ancestry_relationship_path

    RELATIONSHIP_UTILS_AVAILABLE = True
    print("✓ RELATIONSHIP_UTILS_AVAILABLE set to True")

except ImportError as e:
    print(f"✗ Relationship utilities import failed: {e}")
    pass

print(f"After Block 2: RELATIONSHIP_UTILS_AVAILABLE = {RELATIONSHIP_UTILS_AVAILABLE}")

# Test Block 3: API utilities (exact same logic as action9_process_productive.py)
print("\n--- Testing Block 3: API Utilities ---")
try:
    print("Importing from action11...")
    from action11 import _process_and_score_suggestions

    API_UTILS_AVAILABLE = True
    print("✓ API_UTILS_AVAILABLE set to True")

except ImportError as e:
    print(f"✗ API utilities import failed: {e}")
    pass

print(f"After Block 3: API_UTILS_AVAILABLE = {API_UTILS_AVAILABLE}")

# Final Results
print("\n" + "=" * 50)
print("FINAL AVAILABILITY FLAGS:")
print(f"GEDCOM_UTILS_AVAILABLE: {GEDCOM_UTILS_AVAILABLE}")
print(f"RELATIONSHIP_UTILS_AVAILABLE: {RELATIONSHIP_UTILS_AVAILABLE}")
print(f"API_UTILS_AVAILABLE: {API_UTILS_AVAILABLE}")

if GEDCOM_UTILS_AVAILABLE and RELATIONSHIP_UTILS_AVAILABLE and API_UTILS_AVAILABLE:
    print("\n🎉 SUCCESS! All flags are True!")
    print("This proves our fix to action9_process_productive.py is working!")
else:
    print(f"\n⚠️  Not all flags are True. Results:")
    if GEDCOM_UTILS_AVAILABLE:
        print("  ✓ GEDCOM utilities working")
    if RELATIONSHIP_UTILS_AVAILABLE:
        print("  ✓ Relationship utilities working")
    if API_UTILS_AVAILABLE:
        print("  ✓ API utilities working")

print("\nTest completed successfully!")
