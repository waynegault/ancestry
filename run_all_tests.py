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
    test_modules = []

    # Scan all Python files in the current directory
    for python_file in current_dir.glob("*.py"):
        # Skip the test runner itself and __init__.py
        if python_file.name in ["run_all_tests.py", "__init__.py"]:
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
    """Run tests for a specific module."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {module_name}.py")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Import and run the module's tests
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)

        # Execute the module's test main block
        module = sys.modules[module_name]
        if hasattr(module, "__file__"):
            # Run the module as script to trigger __main__ block
            old_name = module.__name__
            module.__name__ = "__main__"

            try:
                # Ensure module.__file__ is not None before using it
                if module.__file__ is not None:
                    with open(module.__file__, encoding="utf-8") as f:  # Added encoding
                        exec(compile(f.read(), module.__file__, "exec"))
                    success = True
                    error = None
                else:
                    success = False
                    error = f"Module {module_name} has no __file__ attribute."
            except SystemExit as e:
                success = e.code == 0
                error = None if success else f"Test failed with exit code {e.code}"
            except Exception as e:
                success = False
                error = str(e)
            finally:
                module.__name__ = old_name
        else:
            success = False
            error = "Module file not found"

    except ImportError as e:
        success = False
        error = f"Import error: {e}"
    except Exception as e:
        success = False
        error = f"Unexpected error: {e}"

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
        )

    # Run individual module tests
    for module_name in test_modules:
        result = run_module_test(module_name)
        results.append(result)

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
