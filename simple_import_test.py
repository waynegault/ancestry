#!/usr/bin/env python3

"""Simple test to verify import status after fixes"""

print("Testing imports manually...")

# Test individual imports that should work
print("\n1. Testing gedcom_utils imports...")
try:
    from gedcom_utils import calculate_match_score, _normalize_id, GedcomData

    print("✅ SUCCESS: gedcom_utils imports work")
except ImportError as e:
    print(f"❌ FAILED: gedcom_utils import error: {e}")

print("\n2. Testing relationship_utils imports...")
try:
    from relationship_utils import (
        fast_bidirectional_bfs,
        convert_gedcom_path_to_unified_format,
        format_relationship_path_unified,
    )

    print("✅ SUCCESS: relationship_utils imports work")
except ImportError as e:
    print(f"❌ FAILED: relationship_utils import error: {e}")

print("\n3. Testing relationship utilities...")
try:
    from gedcom_search_utils import get_gedcom_relationship_path
    from action11 import get_ancestry_relationship_path

    print("✅ SUCCESS: relationship utilities import work")
except ImportError as e:
    print(f"❌ FAILED: relationship utilities import error: {e}")

print("\n4. Testing that action9 loads without import errors...")
try:
    import action9_process_productive

    print("✅ SUCCESS: action9_process_productive loads without import errors")

    # Check the flags
    print(
        f"   GEDCOM_UTILS_AVAILABLE: {action9_process_productive.GEDCOM_UTILS_AVAILABLE}"
    )
    print(
        f"   RELATIONSHIP_UTILS_AVAILABLE: {action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE}"
    )
    print(f"   API_UTILS_AVAILABLE: {action9_process_productive.API_UTILS_AVAILABLE}")

except ImportError as e:
    print(f"❌ FAILED: action9_process_productive import error: {e}")
    import traceback

    traceback.print_exc()

print("\nImport test complete!")
