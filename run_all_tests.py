#!/usr/bin/env python3
"""
Comprehensive Test Runner for Ancestry Project
Runs all unit tests and integration tests across the entire project.

Usage:
    python run_all_tests.py           # Run all tests with standard timeouts
    python run_all_tests.py --fast    # Run all tests with reduced timeouts (faster but may miss slow tests)
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

    # Known modules with comprehensive test suites
    known_test_modules = [
        "action6_gather",
        "action7_inbox",
        "action8_messaging",
        "action9_process_productive",
        "action10",
        "action11",
        "ai_interface",
        "api_utils",
        "cache_manager",
        "config",
        "database",
        "error_handling",
        "gedcom_search_utils",
        "gedcom_utils",
        "my_selectors",
        "performance_monitor",
        "person_search",
        "relationship_utils",
        "selenium_utils",
        "utils",
        "check_db",
    ]

    # Scan all Python files in the current directory
    for python_file in current_dir.glob("*.py"):
        # Skip the test runner itself, __init__.py, main.py, and setup scripts
        skip_files = [
            "run_all_tests.py",
            "__init__.py",
            "main.py",
            "setup_credentials_helper.py",
            "setup_real_credentials.py",
            "setup_security.py",
            "check_test_coverage.py",
        ]

        if python_file.name in skip_files:
            continue

        module_name = python_file.stem

        try:
            # Check if module has test capabilities
            with open(python_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Enhanced test detection patterns
                has_main_block = 'if __name__ == "__main__"' in content
                has_test_content = any(
                    indicator in content.lower()
                    for indicator in [
                        "run_comprehensive_tests",
                        "def test_",
                        "class test",
                        "unittest",
                        "testsuite",
                        "test_framework",
                        "from test_framework import",
                        "import unittest",
                        "assert ",
                        "suite.run_test",
                        "comprehensive_test",
                    ]
                )

                # Also check for known test modules or modules with comprehensive test functions
                has_comprehensive_tests = "run_comprehensive_tests" in content
                is_known_test_module = module_name in known_test_modules

                if (
                    (has_main_block and has_test_content)
                    or has_comprehensive_tests
                    or is_known_test_module
                ):
                    test_modules.append(module_name)

        except Exception as e:
            # Log but don't fail - some files might be binary or have encoding issues
            print(f"⚠️  Skipping {python_file.name}: {e}")
            continue

    # Add any known test modules that weren't detected but exist as files
    for known_module in known_test_modules:
        if (
            known_module not in test_modules
            and (current_dir / f"{known_module}.py").exists()
        ):
            test_modules.append(known_module)

    # Sort for consistent output
    return sorted(test_modules)


def run_module_test(module_name: str, fast_mode: bool = False) -> Dict[str, Any]:
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

        # Adjust timeout for modules that process large datasets or have complex tests
        timeout_config = {
            "gedcom_search_utils": 180,  # 3 minutes for GEDCOM search (large data processing)
            "gedcom_utils": 120,  # 2 minutes for GEDCOM utilities
            "action6_gather": 120,  # 2 minutes for DNA match gathering
            "action7_inbox": 90,  # 1.5 minutes for inbox processing
            "action8_messaging": 90,  # 1.5 minutes for messaging
            "action10": 90,  # 1.5 minutes for GEDCOM analysis
            "action11": 90,  # 1.5 minutes for API research
            "database": 90,  # 1.5 minutes for database operations
            "selenium_utils": 90,  # 1.5 minutes for selenium operations
            "utils": 90,  # 1.5 minutes for core utilities
            "person_search": 60,  # 1 minute for person search
            "ai_interface": 60,  # 1 minute for AI interface
            "performance_monitor": 60,  # 1 minute for performance monitoring
            "relationship_utils": 45,  # 45 seconds for relationship utilities
            "api_utils": 45,  # 45 seconds for API utilities
            "cache_manager": 45,  # 45 seconds for cache management
            "error_handling": 45,  # 45 seconds for error handling
        }

        timeout = timeout_config.get(
            module_name, 30
        )  # Default 30 seconds for other modules

        # Reduce timeouts in fast mode
        if fast_mode:
            timeout = min(timeout // 2, 15)  # Half the timeout, but at least 15 seconds

        # Run with timeout to prevent infinite loops
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,  # Adjusted timeout per module
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
        error = f"Test timeout ({timeout}s) - possible infinite loop or hanging test"
        print(f"⏰ {module_name}.py timed out after {timeout} seconds")

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
    print(f"🧪 Running unittest suite (checking for TestSeleniumUtils)")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        import unittest
        import selenium_utils

        # Check if TestSeleniumUtils class exists in selenium_utils
        if hasattr(selenium_utils, "TestSeleniumUtils"):
            TestSeleniumUtils = getattr(selenium_utils, "TestSeleniumUtils")

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
                print(
                    f"❌ {error_count} unittest failures out of {result.testsRun} tests"
                )

            return {
                "module": "unittest_suite",
                "success": success,
                "duration": time.time() - start_time,
                "error": None if success else f"{error_count} test failures",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
            }
        else:
            print("ℹ️  TestSeleniumUtils class not found in selenium_utils module")
            print(
                "✅ Skipping unittest suite (module uses standardized test framework)"
            )
            return {
                "module": "unittest_suite",
                "success": True,
                "duration": time.time() - start_time,
                "error": "TestSeleniumUtils class not found - using standardized framework",
            }

    except ImportError as ie:
        return {
            "module": "unittest_suite",
            "success": False,
            "duration": time.time() - start_time,
            "error": f"Import error: {ie}",
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
    print(
        f"📈 Success rate: {(passed_tests/total_tests*100):.1f}%"
        if total_tests > 0
        else "📈 Success rate: N/A"
    )
    print()

    if failed_tests > 0:
        print("❌ FAILED TESTS:")
        failed_by_category = {}
        for result in results:
            if not result["success"]:
                module = result["module"]
                if module.startswith("action"):
                    category = "Action Modules"
                elif module in ["utils", "config", "database", "error_handling"]:
                    category = "Core Modules"
                elif "api" in module or module in ["selenium_utils", "cache_manager"]:
                    category = "API/Web Modules"
                elif "gedcom" in module:
                    category = "GEDCOM Modules"
                else:
                    category = "Other Modules"

                if category not in failed_by_category:
                    failed_by_category[category] = []
                failed_by_category[category].append(
                    f"{module}: {result.get('error', 'Unknown error')}"
                )

        for category, failures in failed_by_category.items():
            print(f"  🏷️ {category}:")
            for failure in failures:
                print(f"    • {failure}")
        print()

    print("📋 DETAILED RESULTS:")

    # Group results by category for better organization
    categories = {
        "Action Modules": [],
        "Core Modules": [],
        "API/Web Modules": [],
        "GEDCOM Modules": [],
        "Other Modules": [],
    }

    for result in results:
        module = result["module"]
        if module.startswith("action"):
            categories["Action Modules"].append(result)
        elif module in ["utils", "config", "database", "error_handling"]:
            categories["Core Modules"].append(result)
        elif "api" in module or module in ["selenium_utils", "cache_manager"]:
            categories["API/Web Modules"].append(result)
        elif "gedcom" in module:
            categories["GEDCOM Modules"].append(result)
        else:
            categories["Other Modules"].append(result)

    for category_name, category_results in categories.items():
        if category_results:
            print(f"\n  🏷️ {category_name}:")
            for result in category_results:
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                duration = result["duration"]

                if "tests_run" in result:
                    # Unittest suite result
                    print(
                        f"    {status} | {result['module']:<20} | {duration:>6.2f}s | "
                        f"{result['tests_run']} tests"
                    )
                else:
                    # Module test result
                    print(f"    {status} | {result['module']:<20} | {duration:>6.2f}s")


def main():
    """Main test runner function."""
    print("🚀 Ancestry Project - Comprehensive Test Suite")
    print("=" * 60)

    start_time = time.time()
    results = []

    # Check for fast mode argument
    fast_mode = len(sys.argv) > 1 and sys.argv[1] == "--fast"
    if fast_mode:
        print("⚡ Running in FAST MODE - reduced timeouts")

    # Discover and run module tests
    test_modules = discover_test_modules()

    if not test_modules:
        print("⚠️  No test modules discovered")
        print("🔍 Scanning for all Python files in project...")

        # Show what files were found
        all_py_files = [
            f.stem
            for f in current_dir.glob("*.py")
            if f.name
            not in [
                "run_all_tests.py",
                "__init__.py",
                "main.py",
                "setup_credentials_helper.py",
                "setup_real_credentials.py",
                "setup_security.py",
                "check_test_coverage.py",
            ]
        ]
        if all_py_files:
            print(f"📁 Found Python files: {', '.join(sorted(all_py_files))}")
            print("💡 None appear to have test capabilities (missing test indicators)")
        else:
            print("📁 No Python files found in project directory")
        return False

    print(f"🔍 Discovered {len(test_modules)} test modules:")

    # Group modules by category for better output
    action_modules = [m for m in test_modules if m.startswith("action")]
    core_modules = [
        m
        for m in test_modules
        if m in ["utils", "config", "database", "error_handling"]
    ]
    api_modules = [
        m
        for m in test_modules
        if "api" in m or m in ["selenium_utils", "cache_manager"]
    ]
    gedcom_modules = [m for m in test_modules if "gedcom" in m]
    other_modules = [
        m
        for m in test_modules
        if m not in action_modules + core_modules + api_modules + gedcom_modules
    ]

    if action_modules:
        print(f"  📝 Action modules: {', '.join(action_modules)}")
    if core_modules:
        print(f"  🏗️ Core modules: {', '.join(core_modules)}")
    if api_modules:
        print(f"  🌐 API/Web modules: {', '.join(api_modules)}")
    if gedcom_modules:
        print(f"  🌳 GEDCOM modules: {', '.join(gedcom_modules)}")
    if other_modules:
        print(f"  🔧 Other modules: {', '.join(other_modules)}")

    # Show what files were found but skipped
    all_py_files = [
        f.stem
        for f in current_dir.glob("*.py")
        if f.name
        not in [
            "run_all_tests.py",
            "__init__.py",
            "main.py",
            "setup_credentials_helper.py",
            "setup_real_credentials.py",
            "setup_security.py",
            "check_test_coverage.py",
        ]
    ]
    skipped_files = [f for f in all_py_files if f not in test_modules]
    if skipped_files:
        print(
            f"📋 Skipped files (no test indicators): {', '.join(sorted(skipped_files))}"
        )

    # Run individual module tests with progress tracking
    print(f"\n🔄 Running tests for {len(test_modules)} modules...")
    for i, module_name in enumerate(test_modules, 1):
        print(f"\n📍 Progress: {i}/{len(test_modules)} - Testing {module_name}.py")
        try:
            result = run_module_test(module_name, fast_mode)
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

    # Calculate timing statistics
    if results:
        durations = [r["duration"] for r in results]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        # Find slowest and fastest modules
        slowest_module = max(results, key=lambda r: r["duration"])
        fastest_module = min(results, key=lambda r: r["duration"])

    print(f"\n{'='*60}")
    if all_passed:
        print(f"🎉 ALL TESTS PASSED!")
        print(f"🕒 Total execution time: {total_duration:.2f}s")
        if results:
            print(f"📊 Test statistics:")
            print(f"   • Average module time: {avg_duration:.2f}s")
            print(
                f"   • Fastest module: {fastest_module['module']} ({min_duration:.2f}s)"
            )
            print(
                f"   • Slowest module: {slowest_module['module']} ({max_duration:.2f}s)"
            )
    else:
        failed_count = sum(1 for r in results if not r["success"])
        print(f"💥 {failed_count} TEST SUITE(S) FAILED!")
        print(f"🕒 Total execution time: {total_duration:.2f}s")
        print(
            f"📉 Success rate: {((len(results) - failed_count) / len(results) * 100):.1f}%"
            if results
            else "📉 Success rate: N/A"
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
