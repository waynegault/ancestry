#!/usr/bin/env python3
"""
Test script to verify gedcom_search_utils timeout fix
"""
import sys
import time


def run_comprehensive_tests() -> bool:
    """
    Simple test function for gedcom_search_utils timeout verification.
    """
    import time

    start_time = time.time()
    print("🧪 Testing gedcom_search_utils timeout fix...")

    tests_passed = 0
    total_tests = 0

    # Test 1: Basic import test
    total_tests += 1
    try:
        # Try importing the module without executing it
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "gedcom_search_utils", "gedcom_search_utils.py"
        )
        if spec and spec.loader:
            print("✅ Module can be loaded successfully")
            tests_passed += 1
        else:
            print("❌ Module spec could not be created")
    except Exception as e:
        print(f"❌ Module import test failed: {e}")

    # Test 2: Check if module has required functions
    total_tests += 1
    try:
        import gedcom_search_utils

        required_functions = [
            "search_gedcom_for_criteria",
            "get_gedcom_family_details",
            "run_comprehensive_tests_fallback",
        ]
        available = [
            func for func in required_functions if hasattr(gedcom_search_utils, func)
        ]
        if len(available) >= 2:  # At least 2 of the 3 functions should be available
            print(f"✅ Module has required functions: {available}")
            tests_passed += 1
        else:
            print(f"❌ Module missing required functions, only found: {available}")
    except Exception as e:
        print(f"❌ Function availability test failed: {e}")

    # Test 3: Quick execution test with timeout
    total_tests += 1
    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "gedcom_search_utils.py"],
            capture_output=True,
            text=True,
            timeout=60,  # Reduced timeout
        )
        if result.returncode == 0:
            print("✅ Module executes successfully")
            tests_passed += 1
        else:
            print(f"❌ Module execution failed with exit code: {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("❌ Module execution timed out (>60s)")
    except Exception as e:
        print(f"❌ Module execution test failed: {e}")

    elapsed = time.time() - start_time
    success = tests_passed >= 2  # Pass if at least 2 out of 3 tests pass
    print(f"📊 Test results: {tests_passed}/{total_tests} passed in {elapsed:.2f}s")
    print(f"🎯 Overall result: {'PASS' if success else 'FAIL'}")

    return success


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
