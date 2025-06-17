#!/usr/bin/env python3
"""
Verification script to check TestSuite standardization across all Python files
"""

import sys
import importlib
import traceback
from pathlib import Path

# Files that should have run_comprehensive_tests
TARGET_FILES = [
    "main",
    "action6_gather",
    "action7_inbox",
    "action8_messaging",
    "action9_process_productive",
    "action10",
    "action11",
    "api_cache",
    "api_search_utils",
    "api_utils",
    "cache_manager",
    "chromedriver",
    'config', 
    "database",
    "error_handling",
    "gedcom_cache",
    "gedcom_search_utils",
    "gedcom_utils",
    "logging_config",
    "ms_graph_utils",
    "person_search",
    "relationship_utils",
    "selenium_utils",
    "test_framework",
    "utils",
]


def check_file_standards(module_name):
    """Check if a file follows the TestSuite standards"""
    result = {
        "module": module_name,
        "can_import": False,
        "has_run_comprehensive_tests": False,
        "function_is_callable": False,
        "error": None,
    }

    try:
        # Try to import the module
        module = importlib.import_module(module_name)
        result["can_import"] = True

        # Check if it has run_comprehensive_tests function
        if hasattr(module, "run_comprehensive_tests"):
            result["has_run_comprehensive_tests"] = True

            # Check if the function is callable
            func = getattr(module, "run_comprehensive_tests")
            if callable(func):
                result["function_is_callable"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    print("🔍 TestSuite Standardization Verification Report")
    print("=" * 60)

    results = []

    for module_name in TARGET_FILES:
        print(f"Checking {module_name}...", end=" ")
        result = check_file_standards(module_name)
        results.append(result)

        if (
            result["can_import"]
            and result["has_run_comprehensive_tests"]
            and result["function_is_callable"]
        ):
            print("✅ COMPLIANT")
        elif result["can_import"] and result["has_run_comprehensive_tests"]:
            print("⚠️  FUNCTION NOT CALLABLE")
        elif result["can_import"]:
            print("❌ MISSING run_comprehensive_tests")
        else:
            print(f"❌ IMPORT ERROR: {result['error']}")

    print("\n" + "=" * 60)
    print("📊 Summary Report")
    print("=" * 60)

    compliant = [
        r
        for r in results
        if r["can_import"]
        and r["has_run_comprehensive_tests"]
        and r["function_is_callable"]
    ]
    missing_function = [
        r for r in results if r["can_import"] and not r["has_run_comprehensive_tests"]
    ]
    import_errors = [r for r in results if not r["can_import"]]

    print(f"✅ Fully Compliant: {len(compliant)} files")
    print(f"❌ Missing Function: {len(missing_function)} files")
    print(f"🚫 Import Errors: {len(import_errors)} files")

    if missing_function:
        print(f"\n📝 Files needing run_comprehensive_tests:")
        for r in missing_function:
            print(f"   - {r['module']}.py")

    if import_errors:
        print(f"\n🚨 Files with import issues:")
        for r in import_errors:
            print(f"   - {r['module']}.py: {r['error']}")

    return len(compliant) == len(TARGET_FILES)


if __name__ == "__main__":
    success = main()
    print(f"\n🎯 Overall Status: {'FULLY STANDARDIZED' if success else 'NEEDS WORK'}")
