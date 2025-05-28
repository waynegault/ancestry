#!/usr/bin/env python3
"""
Quick test to verify the ValueError fix is still working
"""

print("Checking if ValueError fix is still in place...")

try:
    # Test the function signature directly
    import os
    import sys

    sys.path.insert(0, os.getcwd())

    # Import and test the _get_event_info function
    from relationship_utils import _get_event_info, GEDCOM_UTILS_AVAILABLE

    print(f"GEDCOM_UTILS_AVAILABLE: {GEDCOM_UTILS_AVAILABLE}")

    # Test the function
    result = _get_event_info(None, "BIRT")
    print(f"_get_event_info returns: {result}")
    print(f"Number of values: {len(result)}")

    # Test unpacking (this should work without ValueError)
    birth_date_obj, date_str, location = result
    print(f"Successfully unpacked: {birth_date_obj}, {date_str}, {location}")

    print("✅ ValueError fix is still working correctly!")

except ValueError as e:
    print(f"❌ ValueError still exists: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback

    traceback.print_exc()
