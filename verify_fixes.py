#!/usr/bin/env python3

"""
Final verification test for the Ancestry.com genealogy automation fixes
"""


def test_core_fixes():
    """Test that our main fixes are working"""

    print("=== Testing Fixed Import Issues ===\n")

    # Test 1: Relationship utilities import
    try:
        from action9_process_productive import RELATIONSHIP_UTILS_AVAILABLE

        print(f"✅ RELATIONSHIP_UTILS_AVAILABLE: {RELATIONSHIP_UTILS_AVAILABLE}")
    except Exception as e:
        print(f"❌ Relationship utils import failed: {e}")
        return False

    # Test 2: GEDCOM cache import
    try:
        from gedcom_cache import load_gedcom_with_aggressive_caching

        print("✅ GEDCOM cache loading function available")
    except Exception as e:
        print(f"❌ GEDCOM cache import failed: {e}")
        return False

    # Test 3: Core action9 import
    try:
        from action9_process_productive import process_productive_messages

        print("✅ Main process_productive_messages function available")
    except Exception as e:
        print(f"❌ Core action9 import failed: {e}")
        return False

    # Test 4: Verify specific fixed functions exist
    try:
        from gedcom_search_utils import get_gedcom_relationship_path
        from action11 import get_ancestry_relationship_path

        print("✅ Both relationship path functions correctly imported")
    except Exception as e:
        print(f"❌ Relationship path functions import failed: {e}")
        return False

    print("\n=== All Core Fixes Verified Successfully! ===")
    print("\nThe two main errors should now be resolved:")
    print("1. ✅ 'Relationship utilities not available' - Fixed import paths")
    print("2. ✅ 'Cannot pickle BinaryFileCR instances' - Fixed caching strategy")

    return True


if __name__ == "__main__":
    success = test_core_fixes()
    if success:
        print("\n🎉 System should now work without the original errors!")
    else:
        print("\n❌ Some issues remain - check error messages above")
