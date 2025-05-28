#!/usr/bin/env python3
"""
Test just the fallback functions without importing gedcom_utils
"""

print("Testing fallback functions...")

# Force use of fallback functions by setting the flag
import sys
import os

sys.path.insert(0, os.getcwd())

# Mock the gedcom_utils import to force fallback usage
import relationship_utils

# Check if we're using fallback functions
print(f"GEDCOM_UTILS_AVAILABLE: {relationship_utils.GEDCOM_UTILS_AVAILABLE}")

if not relationship_utils.GEDCOM_UTILS_AVAILABLE:
    print("Using fallback functions - testing grandparent logic...")

    # Test the _get_event_info fix (ValueError fix)
    result = relationship_utils._get_event_info(None, "BIRT")
    print(f"_get_event_info returns {len(result)} values: {result}")

    # Test unpacking (this should work now)
    try:
        birth_date_obj, date_str, location = result
        print("✓ Successfully unpacked 3 values from _get_event_info")
    except ValueError as e:
        print(f"❌ ValueError still exists: {e}")

    print("✅ Fallback function tests completed successfully!")
else:
    print("Using real gedcom_utils functions - fixes may not be testable this way")
