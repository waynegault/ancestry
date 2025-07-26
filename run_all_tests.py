#!/usr/bin/env python3

"""
Comprehensive Test Runner for Ancestry Project
Runs all unit tests and integration tests across the entire project.

Usage:
    python run_all_tests.py           # Run all tests with standard timeouts
    python run_all_tests.py --fast    # Run all tests with reduced timeouts (faster but may miss slow tests)
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module, safe_execute

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

# === STANDARD LIBRARY IMPORTS ===
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to Python path to allow imports from subdirectories
project_root = Path(__file__).parent

# Use centralized path management (already called at top of file)
# standardize_module_imports() was already called in the unified import system above


def discover_test_modules() -> List[str]:
    """Discover all Python modules that contain tests by recursively scanning the project directory."""
    test_modules = []

    for python_file in project_root.rglob("*.py"):
        # Skip the test runner itself, __init__.py, main.py, setup files, and temporary files
        if (
            python_file.name
            in [
                "run_all_tests.py",
                "__init__.py",
                "main.py",
                "setup.py",
            ]
            or "__pycache__" in str(python_file)
            or python_file.name.endswith("_backup.py")
            or "backup_before_migration" in str(python_file)
            or python_file.name.startswith("phase1_cleanup")
            or python_file.name.startswith("test_phase1")
            or python_file.name.startswith("cleanup_")
            or python_file.name.startswith("migration_")
            or python_file.name.startswith("fix_")
            or python_file.name.startswith("convert_")
            or python_file.name.startswith("test_")
            and python_file.name
            not in ["test_framework.py", "test_framework_unified.py"]
            or "temp" in python_file.name.lower()
            or "_old" in python_file.name
        ):
            continue

        # Construct module name from path (e.g., core.api_manager)
        relative_path = python_file.relative_to(project_root)
        module_name = ".".join(relative_path.with_suffix("").parts)

        try:
            with open(python_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Standardized TestSuite detection
                has_main_block = 'if __name__ == "__main__"' in content
                has_test_content = "run_comprehensive_tests()" in content

                if has_main_block and has_test_content:
                    test_modules.append(module_name)

        except Exception as e:
            print(f"⚠️  Skipping {python_file.name}: Could not read file ({e})")
            continue

    return sorted(test_modules)


def run_module_test(
    module_name: str, fast_mode: bool = False, verbose: bool = False
) -> Dict[str, Any]:
    """Run tests for a specific module using subprocess for safety and isolation."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {module_name}")
    print(f"{'='*60}")

    start_time = time.time()
    # Use the simple name for timeout configuration
    simple_module_name = module_name.split(".")[-1]

    try:
        import subprocess
        import sys

        # Use `python -m <module>` to run modules from subdirectories correctly
        cmd = [sys.executable, "-m", module_name]

        timeout_config = {
            "gedcom_search_utils": 180,
            "gedcom_utils": 120,
            "action6_gather": 120,
            "action7_inbox": 90,
            "action8_messaging": 90,
            "action10": 120,
            "action11": 90,
            "database": 90,
            "selenium_utils": 90,
            "utils": 90,
            "person_search": 60,
            "ai_interface": 60,
            "relationship_utils": 45,
            "api_utils": 45,
            "cache_manager": 45,
            "error_handling": 45,
            # Subdirectory modules
            "api_manager": 60,
            "browser_manager": 60,
            "database_manager": 60,
            "session_manager": 60,
            "session_validator": 60,
            "config_manager": 45,
            "config_schema": 45,
            "credential_manager": 45,
        }

        timeout = timeout_config.get(simple_module_name, 30)

        if fast_mode:
            timeout = min(timeout // 2, 15)

        if verbose:
            print(f"   ⚙️ Running command: {' '.join(cmd)}")
            print(f"   📁 Working directory: {project_root}")
            print(f"   ⏱️ Timeout: {timeout}s")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["RUNNING_ANCESTRY_TESTS"] = "1"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root,
            env=env,
        )

        success = result.returncode == 0

        if success:
            print(f"✅ PASSED: {module_name} tests completed successfully")
            if result.stdout:
                stdout_lines = result.stdout.strip().split("\n")
                test_summary_lines = [
                    line
                    for line in stdout_lines
                    if "✅ Passed:" in line or "Status:" in line
                ]
                if test_summary_lines:
                    for summary_line in test_summary_lines[-2:]:
                        print(f"   📊 {summary_line.strip()}")
        else:
            print(
                f"❌ FAILED: {module_name} tests failed (exit code: {result.returncode})"
            )
            if result.stderr:
                stderr_preview = result.stderr.strip().replace("\n", "\n   ")
                print(f"   🚨 Error: {stderr_preview[:2000]}")

        error = (
            None
            if success
            else f"Exit code {result.returncode}: {(result.stderr or '').strip()[:2000]}"
        )

    except subprocess.TimeoutExpired:
        success = False
        error = f"Test timeout ({timeout}s) - possible infinite loop or hanging test"
        print(f"⏰ {module_name} timed out after {timeout} seconds")

    except FileNotFoundError:
        success = False
        error = f"Module {module_name} not found. Check sys.path and module name."
        print(f"📁 Module not found: {module_name}")

    except Exception as e:
        success = False
        error = f"Unexpected error: {e}"
        print(f"💥 Unexpected error in {module_name}: {e}")

    duration = time.time() - start_time

    return {
        "module": module_name,
        "success": success,
        "duration": duration,
        "error": error,
    }


def get_module_category(module_name: str) -> str:
    """Categorize module based on its name and path."""
    if module_name.startswith("core."):
        return "Core Subsystem"
    if module_name.startswith("config."):
        return "Configuration Subsystem"
    if module_name.startswith("action"):
        return "Action Modules"
    if "gedcom" in module_name:
        return "GEDCOM Modules"
    if "api" in module_name or "selenium" in module_name or "browser" in module_name:
        return "API/Web Modules"
    if "db" in module_name or "database" in module_name or "cache" in module_name:
        return "Data Modules"
    return "Other Modules"


def print_summary(results: List[Dict[str, Any]]):
    """Print a comprehensive test summary with consistent formatting."""
    print(f"\n{'='*60}")
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests
    total_duration = sum(r["duration"] for r in results)

    print(f"📈 Overall Results:")
    print(f"   • Total modules tested: {total_tests}")
    print(f"   • ✅ Passed: {passed_tests}")
    print(f"   • ❌ Failed: {failed_tests}")
    print(f"   • ⏱️ Total time: {total_duration:.2f}s")
    print(
        f"   • 📊 Success rate: {(passed_tests/total_tests*100):.1f}%"
        if total_tests > 0
        else "   • 📊 Success rate: N/A"
    )

    if failed_tests > 0:
        print("\n❌ FAILED TESTS DETAILS:")
        failed_by_category = {}
        for result in results:
            if not result["success"]:
                category = get_module_category(result["module"])
                if category not in failed_by_category:
                    failed_by_category[category] = []
                failed_by_category[category].append(
                    f"{result['module']}: {result.get('error', 'Unknown error')[:100]}..."
                )

        for category, failures in failed_by_category.items():
            print(f"  🏷️ {category}:")
            for failure in failures:
                print(f"    • {failure}")

    print("\n📋 DETAILED RESULTS BY CATEGORY:")
    results_by_category = {}
    for r in results:
        category = get_module_category(r["module"])
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(r)

    for category_name, category_results in sorted(results_by_category.items()):
        if category_results:
            print(f"\n  🏷️ {category_name}:")
            for result in sorted(category_results, key=lambda x: x["module"]):
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                duration = result["duration"]
                print(f"    {status} | {result['module']:<30} | {duration:>6.2f}s")


def main():
    """Main test runner function."""
    print("🚀 Ancestry Project - Comprehensive Test Suite")
    print("=" * 60)

    start_time = time.time()
    results = []

    fast_mode = len(sys.argv) > 1 and sys.argv[1] == "--fast"
    if fast_mode:
        print("⚡ Running in FAST MODE - reduced timeouts")

    test_modules = discover_test_modules()

    if not test_modules:
        print("⚠️  No test modules discovered.")
        print(
            "Ensure modules have a `run_comprehensive_tests()` call inside a `if __name__ == '__main__'` block."
        )
        return False

    print(f"🔍 Discovered {len(test_modules)} test modules.")

    # Run individual module tests
    for i, module_name in enumerate(test_modules, 1):
        print(f"\n📍 Progress: [{i:2d}/{len(test_modules)}] - Executing {module_name}")
        try:
            result = run_module_test(module_name, fast_mode)
            results.append(result)
            status_emoji = "✅" if result["success"] else "❌"
            status_text = "PASSED" if result["success"] else "FAILED"
            print(
                f"   {status_emoji} {status_text} | Duration: {result['duration']:.2f}s"
            )

        except KeyboardInterrupt:
            print(f"\n⚠️  Test execution interrupted by user at {module_name}")
            break
        except Exception as e:
            print(f"💥 Critical error testing {module_name}: {e}")
            results.append(
                {
                    "module": module_name,
                    "success": False,
                    "duration": 0,
                    "error": f"Critical runner error: {e}",
                }
            )

    print_summary(results)

    total_duration = time.time() - start_time
    all_passed = all(r["success"] for r in results)

    print(f"\n{'='*60}")
    if all_passed:
        print(f"🎉 ALL TESTS PASSED!")
    else:
        failed_count = sum(1 for r in results if not r["success"])
        print(f"💥 {failed_count} TEST SUITE(S) FAILED!")
    print(f"🕒 Total execution time: {total_duration:.2f}s")
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
