#!/usr/bin/env python3
"""
Comprehensive Test Runner for Ancestry Project
Runs all unit tests and integration tests across the entire project.
"""

import sys
import os
import time
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def discover_test_modules() -> List[str]:
    """Discover all Python modules that contain tests by scanning the project directory."""
    test_modules = []  # Scan all Python files in the current directory
    for python_file in current_dir.glob("*.py"):
        # Skip the test runner itself, __init__.py, and main.py (interactive application)
        if python_file.name in ["run_all_tests.py", "__init__.py", "main.py"]:
            continue

        module_name = python_file.stem

        try:
            # Check if module has test capabilities
            with open(python_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Look for test indicators
                has_main_block = 'if __name__ == "__main__"' in content
                has_test_content = any(
                    indicator in content.lower()
                    for indicator in [
                        "test_",
                        "def test",
                        "class test",
                        "unittest",
                        "assert",
                        "testsuite",
                        "run_test",
                        "comprehensive_test",
                    ]
                )

                if has_main_block and has_test_content:
                    test_modules.append(module_name)

        except Exception as e:
            # Log but don't fail - some files might be binary or have encoding issues
            print(f"⚠️  Skipping {python_file.name}: {e}")
            continue

    # Sort for consistent output
    return sorted(test_modules)


def run_module_test(module_name: str) -> Dict[str, Any]:
    """Run tests for a specific module using subprocess for safety."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {module_name}.py")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        import subprocess
        import sys

        # Use subprocess to run the module in isolation to avoid import loops
        cmd = [sys.executable, f"{module_name}.py"]

        # Run with timeout to prevent infinite loops
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout per module
            cwd=current_dir,
        )

        success = result.returncode == 0

        if success:
            print(f"✅ {module_name}.py tests passed")
        else:
            print(f"❌ {module_name}.py tests failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"   Error output: {result.stderr.strip()[:200]}")

        error = (
            None
            if success
            else f"Exit code {result.returncode}: {result.stderr.strip()[:500]}"
        )

    except subprocess.TimeoutExpired:
        success = False
        error = "Test timeout (30s) - possible infinite loop or hanging test"
        print(f"⏰ {module_name}.py timed out after 30 seconds")

    except FileNotFoundError:
        success = False
        error = f"Module file {module_name}.py not found"
        print(f"📁 {module_name}.py file not found")

    except ImportError as e:
        success = False
        error = f"Import error: {e}"
        print(f"📦 Import error in {module_name}.py: {e}")

    except Exception as e:
        success = False
        error = f"Unexpected error: {e}"
        print(f"💥 Unexpected error in {module_name}.py: {e}")

    duration = time.time() - start_time

    return {
        "module": module_name,
        "success": success,
        "duration": duration,
        "error": error,
    }


def run_unittest_suite() -> Dict[str, Any]:
    """Run the unittest suite from selenium_utils if available."""
    print(f"\n{'='*60}")
    print(f"🧪 Running unittest suite")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        import unittest
        from selenium_utils import TestSeleniumUtils

        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestSeleniumUtils)

        # Run tests with minimal output
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, "w"))
        result = runner.run(suite)

        success = result.wasSuccessful()
        error_count = len(result.errors) + len(result.failures)

        if success:
            print(f"✅ All {result.testsRun} unittest cases passed")
        else:
            print(f"❌ {error_count} unittest failures out of {result.testsRun} tests")

        return {
            "module": "unittest_suite",
            "success": success,
            "duration": time.time() - start_time,
            "error": None if success else f"{error_count} test failures",
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
        }

    except ImportError:
        return {
            "module": "unittest_suite",
            "success": False,
            "duration": time.time() - start_time,
            "error": "TestSeleniumUtils not available",
        }
    except Exception as e:
        return {
            "module": "unittest_suite",
            "success": False,
            "duration": time.time() - start_time,
            "error": str(e),
        }


def print_summary(results: List[Dict[str, Any]]):
    """Print a comprehensive test summary."""
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests
    total_duration = sum(r["duration"] for r in results)

    print(f"Total modules tested: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"⏱️  Total time: {total_duration:.2f}s")
    print()

    if failed_tests > 0:
        print("❌ FAILED TESTS:")
        for result in results:
            if not result["success"]:
                print(f"  • {result['module']}: {result.get('error', 'Unknown error')}")
        print()

    print("📋 DETAILED RESULTS:")
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = result["duration"]

        if "tests_run" in result:
            # Unittest suite result
            print(
                f"  {status} | {result['module']:<20} | {duration:>6.2f}s | "
                f"{result['tests_run']} tests"
            )
        else:
            # Module test result
            print(f"  {status} | {result['module']:<20} | {duration:>6.2f}s")


def main():
    """Main test runner function."""
    print("🚀 Ancestry Project - Comprehensive Test Suite")
    print("=" * 60)

    start_time = time.time()
    results = []

    # Discover and run module tests
    test_modules = discover_test_modules()

    if not test_modules:
        print("⚠️  No test modules discovered")
        print("🔍 Scanning for all Python files in project...")

        # Show what files were found
        all_py_files = [
            f.stem for f in current_dir.glob("*.py") if f.name != "run_all_tests.py"
        ]
        if all_py_files:
            print(f"📁 Found Python files: {', '.join(sorted(all_py_files))}")
            print("💡 None appear to have test capabilities (missing test indicators)")
        else:
            print("📁 No Python files found in project directory")
        return False

    print(f"🔍 Discovered {len(test_modules)} test modules: {', '.join(test_modules)}")

    # Show what files were skipped
    all_py_files = [
        f.stem for f in current_dir.glob("*.py") if f.name != "run_all_tests.py"
    ]
    skipped_files = [f for f in all_py_files if f not in test_modules]
    if skipped_files:
        print(
            f"📋 Skipped files (no test indicators): {', '.join(sorted(skipped_files))}"
        )  # Run individual module tests with progress tracking
    print(f"\n🔄 Running tests for {len(test_modules)} modules...")
    for i, module_name in enumerate(test_modules, 1):
        print(f"\n📍 Progress: {i}/{len(test_modules)} - Testing {module_name}.py")
        try:
            result = run_module_test(module_name)
            results.append(result)

            # Show immediate result
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"   {status} ({result['duration']:.1f}s)")

        except KeyboardInterrupt:
            print(f"\n⚠️  Test run interrupted by user at {module_name}")
            break
        except Exception as e:
            print(f"💥 Critical error testing {module_name}: {e}")
            results.append(
                {
                    "module": module_name,
                    "success": False,
                    "duration": 0,
                    "error": f"Critical test runner error: {e}",
                }
            )

    # Run unittest suite if available
    unittest_result = run_unittest_suite()
    results.append(unittest_result)

    # Print comprehensive summary
    print_summary(results)

    # Overall result
    total_duration = time.time() - start_time
    all_passed = all(r["success"] for r in results)

    print(f"\n{'='*60}")
    if all_passed:
        print(f"🎉 ALL TESTS PASSED! Total time: {total_duration:.2f}s")
    else:
        failed_count = sum(1 for r in results if not r["success"])
        print(
            f"💥 {failed_count} TEST SUITE(S) FAILED! Total time: {total_duration:.2f}s"
        )
    print(f"{'='*60}")

    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error in test runner: {e}")
        sys.exit(1)
