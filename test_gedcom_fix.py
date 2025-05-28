#!/usr/bin/env python3

print("Testing GEDCOM utility imports after fix...")

try:
    from gedcom_utils import load_gedcom_data

    print("❌ UNEXPECTED: load_gedcom_data imported from gedcom_utils")
except ImportError as e:
    print("✅ EXPECTED: load_gedcom_data not in gedcom_utils")

try:
    from gedcom_utils import calculate_match_score, GedcomData

    print("✅ SUCCESS: calculate_match_score and GedcomData imported from gedcom_utils")
except ImportError as e:
    print(f"❌ ERROR: {e}")

try:
    from gedcom_search_utils import load_gedcom_data

    print("✅ SUCCESS: load_gedcom_data imported from gedcom_search_utils")
except ImportError as e:
    print(f"❌ ERROR: {e}")

# Test the import that was causing issues
try:
    import action9_process_productive

    print("✅ SUCCESS: action9_process_productive imports without errors")
except ImportError as e:
    print(f"❌ ERROR importing action9_process_productive: {e}")

# Check GEDCOM_UTILS_AVAILABLE from test_imports
try:
    from test_imports import GEDCOM_UTILS_AVAILABLE

    print(f"GEDCOM_UTILS_AVAILABLE: {GEDCOM_UTILS_AVAILABLE}")
    if GEDCOM_UTILS_AVAILABLE:
        print("✅ SUCCESS: GEDCOM utilities are now available!")
    else:
        print("❌ ISSUE: GEDCOM utilities still not available")
except ImportError as e:
    print(f"❌ ERROR importing test_imports: {e}")

print("\nTest complete!")
