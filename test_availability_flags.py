#!/usr/bin/env python3

"""
Simple test to verify availability flags without triggering configuration validation.
This test directly imports the module and checks the flags.
"""

import sys
import os

# Set environment variables to bypass configuration validation
os.environ["TESTING_MODE"] = "true"
os.environ["SKIP_CONFIG_VALIDATION"] = "true"

print("Testing availability flags in action9_process_productive.py...")

try:
    # Import the module directly
    import action9_process_productive

    print(f"✓ Successfully imported action9_process_productive")
    print(
        f"GEDCOM_UTILS_AVAILABLE: {action9_process_productive.GEDCOM_UTILS_AVAILABLE}"
    )
    print(
        f"RELATIONSHIP_UTILS_AVAILABLE: {action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE}"
    )
    print(f"API_UTILS_AVAILABLE: {action9_process_productive.API_UTILS_AVAILABLE}")

    # Check if all flags are True
    all_available = (
        action9_process_productive.GEDCOM_UTILS_AVAILABLE
        and action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE
        and action9_process_productive.API_UTILS_AVAILABLE
    )

    if all_available:
        print("🎉 SUCCESS: All availability flags are now True!")
    else:
        print("❌ ISSUE: Some availability flags are still False")

    # Test if we can use the get_gedcom_data function
    print("\nTesting get_gedcom_data function...")
    try:
        result = action9_process_productive.get_gedcom_data()
        if result:
            print("✓ GEDCOM data loaded successfully")
        else:
            print(
                "ℹ GEDCOM data not loaded (expected if file not found or unavailable)"
            )
    except Exception as e:
        print(f"⚠ Error calling get_gedcom_data: {e}")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback

    traceback.print_exc()

print("\nTest completed.")
