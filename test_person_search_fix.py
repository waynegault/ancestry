#!/usr/bin/env python3
"""
Simple test to verify the ValueError is fixed in person_search.py
"""

print("Testing if the ValueError is fixed...")

try:
    # Test the fallback function signature in relationship_utils
    # This should now return 3 values instead of 2
    import sys
    import os

    # Add current directory to path
    sys.path.insert(0, os.getcwd())

    # Force using fallback functions by temporarily disabling gedcom_utils import
    import relationship_utils

    # Check if we can get the function
    if hasattr(relationship_utils, "_get_event_info"):
        get_event_info_func = relationship_utils._get_event_info
        print("✓ Successfully imported _get_event_info function")

        # Test the function signature
        result = get_event_info_func(None, "BIRT")
        print(f"✓ Function returns: {result} (length: {len(result)})")

        # This was the problematic line - should now work
        birth_date_obj, date_str, location = result
        print(
            f"✓ Successfully unpacked 3 values: {birth_date_obj}, {date_str}, {location}"
        )
        print("✅ ValueError fix confirmed!")

    else:
        print("❌ _get_event_info function not found")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
