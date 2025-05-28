#!/usr/bin/env python3

"""
Test script that bypasses configuration validation by directly testing imports.
"""

import sys
import importlib


def test_imports_directly():
    """Test importing modules directly without triggering configuration."""

    print("=== Direct Import Test ===")

    # Test individual module imports
    modules_to_test = ["gedcom_utils", "relationship_utils", "gedcom_search_utils"]

    import_results = {}

    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            import_results[module_name] = "SUCCESS"
            print(f"✓ {module_name}: Import successful")
        except Exception as e:
            import_results[module_name] = f"FAILED: {e}"
            print(f"✗ {module_name}: Import failed - {e}")

    return import_results


def test_flags_by_direct_edit():
    """Test if we can manually verify the flag setting logic."""

    print("\n=== Flag Logic Test ===")

    # Simulate the flag setting logic
    try:
        # Test importing gedcom_utils components
        from gedcom_utils import calculate_match_score, _normalize_id, GedcomData
        from relationship_utils import (
            fast_bidirectional_bfs,
            convert_gedcom_path_to_unified_format,
            format_relationship_path_unified,
        )

        gedcom_utils_available = True
        print("✓ GEDCOM utilities import successful - flag should be True")
    except ImportError as e:
        gedcom_utils_available = False
        print(f"✗ GEDCOM utilities import failed - flag should be False: {e}")

    try:
        from gedcom_search_utils import get_gedcom_relationship_path

        # Note: action11 import might fail due to config, so let's skip it for now
        relationship_utils_available = True
        print("✓ Relationship utilities import successful - flag should be True")
    except ImportError as e:
        relationship_utils_available = False
        print(f"✗ Relationship utilities import failed - flag should be False: {e}")

    try:
        # This will likely fail due to config dependency in action11
        from action11 import _process_and_score_suggestions

        api_utils_available = True
        print("✓ API utilities import successful - flag should be True")
    except ImportError as e:
        api_utils_available = False
        print(f"✗ API utilities import failed - flag should be False: {e}")

    print("\n=== Expected Flag Values ===")
    print(f"GEDCOM_UTILS_AVAILABLE should be: {gedcom_utils_available}")
    print(f"RELATIONSHIP_UTILS_AVAILABLE should be: {relationship_utils_available}")
    print(f"API_UTILS_AVAILABLE should be: {api_utils_available}")

    return gedcom_utils_available, relationship_utils_available, api_utils_available


if __name__ == "__main__":
    print("Testing imports and availability flags...\n")

    # Test direct imports
    import_results = test_imports_directly()

    # Test flag logic
    expected_flags = test_flags_by_direct_edit()

    print(f"\n=== Summary ===")
    print(
        f"Key modules imported successfully: {sum(1 for v in import_results.values() if v == 'SUCCESS')}/{len(import_results)}"
    )
    print(
        f"Expected flag values: GEDCOM={expected_flags[0]}, RELATIONSHIP={expected_flags[1]}, API={expected_flags[2]}"
    )

    if expected_flags[0] and expected_flags[1]:
        print(
            "🎉 Core utilities are working! The availability flags should now be set to True."
        )
    else:
        print("❌ Some core utilities have import issues.")
