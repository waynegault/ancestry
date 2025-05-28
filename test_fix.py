#!/usr/bin/env python3
"""
Simple test to check if the ValueError fix worked
"""

print("Testing import and function signature fix...")

try:
    from relationship_utils import _get_event_info

    print("✓ Successfully imported _get_event_info from relationship_utils")

    # Test the function signature
    result = _get_event_info(None, "BIRT")
    print(f"✓ Function returns: {result} (type: {type(result)})")
    print(f"✓ Return value length: {len(result)}")

    # Test unpacking (this should work now)
    birth_date_obj, date_str, location = _get_event_info(None, "BIRT")
    print(f"✓ Successfully unpacked 3 values: {birth_date_obj}, {date_str}, {location}")

    print("✅ ValueError fix appears to be working!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
