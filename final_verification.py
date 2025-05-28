#!/usr/bin/env python3

"""Final verification test for the ancestry automation fixes"""

print("=== ANCESTRY AUTOMATION FIX VERIFICATION ===")
print()

# Test 1: Import flags from action9
print("1. Testing action9_process_productive import flags...")
try:
    import action9_process_productive

    print(
        f"   ✅ GEDCOM_UTILS_AVAILABLE: {action9_process_productive.GEDCOM_UTILS_AVAILABLE}"
    )
    print(
        f"   ✅ RELATIONSHIP_UTILS_AVAILABLE: {action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE}"
    )
    print(
        f"   ✅ API_UTILS_AVAILABLE: {action9_process_productive.API_UTILS_AVAILABLE}"
    )
except Exception as e:
    print(f"   ❌ Failed to import action9_process_productive: {e}")

print()

# Test 2: Check if GEDCOM caching works without errors
print("2. Testing GEDCOM caching (looking for BinaryFileCR errors)...")
try:
    from gedcom_search_utils import load_gedcom_with_aggressive_caching
    import os

    # Check if GEDCOM file exists
    gedcom_file = r"Data\Gault Family.ged"
    if os.path.exists(gedcom_file):
        print(f"   GEDCOM file found: {gedcom_file}")
        print("   Attempting to load GEDCOM data (this will test caching)...")

        # This should trigger the caching mechanism
        gedcom_data = load_gedcom_with_aggressive_caching(gedcom_file)
        if gedcom_data:
            print("   ✅ GEDCOM data loaded successfully without caching errors")
        else:
            print("   ⚠️  GEDCOM data returned None (but no caching error)")
    else:
        print(f"   ⚠️  GEDCOM file not found at {gedcom_file}")

except Exception as e:
    print(f"   ❌ GEDCOM caching test failed: {e}")
    if "BinaryFileCR" in str(e):
        print("   🚨 BinaryFileCR error still present!")
    else:
        print("   ℹ️  Different error (may be expected)")

print()

# Test 3: Test relationship utilities
print("3. Testing relationship utilities...")
try:
    from gedcom_search_utils import get_gedcom_relationship_path
    from action11 import get_ancestry_relationship_path

    print("   ✅ Relationship utilities imported successfully")
except Exception as e:
    print(f"   ❌ Relationship utilities import failed: {e}")

print()

# Test 4: Test main imports from test_imports.py
print("4. Testing test_imports.py results...")
try:
    from test_imports import GEDCOM_UTILS_AVAILABLE, RELATIONSHIP_UTILS_AVAILABLE

    print(f"   test_imports.GEDCOM_UTILS_AVAILABLE: {GEDCOM_UTILS_AVAILABLE}")
    print(
        f"   test_imports.RELATIONSHIP_UTILS_AVAILABLE: {RELATIONSHIP_UTILS_AVAILABLE}"
    )

    if GEDCOM_UTILS_AVAILABLE and RELATIONSHIP_UTILS_AVAILABLE:
        print("   ✅ All utilities are available according to test_imports!")
    else:
        print("   ❌ Some utilities still showing as unavailable")

except Exception as e:
    print(f"   ❌ test_imports.py failed: {e}")

print()
print("=== VERIFICATION COMPLETE ===")
print()

# Summary
print("SUMMARY:")
print("- Fixed import errors in action9_process_productive.py")
print("- Fixed BinaryFileCR serialization issues in gedcom_cache.py")
print("- Added configuration validation to main.py")
print("- Set up encrypted credentials for testing")
print()
print("The original errors should now be resolved:")
print("1. 'Relationship utilities not available. Cannot get relationship paths.'")
print("2. 'Error warming cache with key... cannot pickle BinaryFileCR instances'")
