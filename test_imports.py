#!/usr/bin/env python3

"""Quick test to verify fixed imports"""

try:
    from action9_process_productive import (
        GEDCOM_UTILS_AVAILABLE,
        RELATIONSHIP_UTILS_AVAILABLE,
        get_gedcom_data,
    )

    print(f"GEDCOM_UTILS_AVAILABLE: {GEDCOM_UTILS_AVAILABLE}")
    print(f"RELATIONSHIP_UTILS_AVAILABLE: {RELATIONSHIP_UTILS_AVAILABLE}")
    print("✓ Import test successful - main errors fixed!")

    # Test if we can call the function without error
    print("\nTesting get_gedcom_data function...")
    result = get_gedcom_data()
    if result:
        print(f"✓ GEDCOM data loaded successfully")
    else:
        print("ℹ GEDCOM data not loaded (expected if file not found)")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback

    traceback.print_exc()
