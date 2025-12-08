#!/usr/bin/env python3
"""
Fast Test Runner - Run only unit tests for rapid feedback.

This script runs a subset of fast tests (<5s total) for quick validation
during development. Use the full test suite (run_all_tests.py) for comprehensive
validation before commits.

Usage:
    python testing/run_tests_fast.py           # Run fast unit tests
    python testing/run_tests_fast.py --all     # Run all tests (same as run_all_tests.py)
    python testing/run_tests_fast.py --list    # List available test modules
"""

import argparse
import importlib
import sys
import time

# Fast test modules - these complete in <2s each and provide good coverage
FAST_TEST_MODULES = [
    # Core infrastructure (fast, isolated tests)
    "test_utilities",
    "test_framework",
    "config.config_schema",
    "config.validator",
    # Database and caching (in-memory tests)
    "database",
    "cache",
    "cache_manager",
    # Utilities (pure functions, no I/O)
    "api_constants",
    "common_params",
    "genealogical_normalization",
    # AI and prompts (mock-based)
    "ai.prompts",
    "ai_prompt_utils",
    # Core modules (fast initialization)
    "core.error_handling",
    "core.session_cache",
    "core.action_registry",
]

# Integration test modules - slower but important
INTEGRATION_TEST_MODULES = [
    "core.session_manager",
    "core.health_check",
    "actions.action6_gather",
    "actions.action7_inbox",
    "actions.action8_messaging",
    "actions.action9_process_productive",
    "actions.action10",
]


def run_module_tests(module_name: str) -> tuple[bool, float, int]:
    """
    Run tests for a single module.

    Returns:
        Tuple of (passed: bool, duration: float, test_count: int)
    """
    try:
        start_time = time.time()
        module = importlib.import_module(module_name)

        # Try different test entry points
        test_runner = getattr(module, "run_comprehensive_tests", None)
        if test_runner is None:
            test_runner = getattr(module, "module_tests", None)
        if test_runner is None:
            # No tests found
            return True, 0.0, 0

        result = test_runner()
        duration = time.time() - start_time

        # Estimate test count from module
        test_count = 1  # Default
        if hasattr(module, "TEST_COUNT"):
            test_count = module.TEST_COUNT

        return bool(result), duration, test_count

    except Exception as e:
        print(f"  ❌ Error running {module_name}: {e}")
        return False, 0.0, 0


def list_modules() -> None:
    """List available test modules."""
    print("\n📋 Fast Test Modules (run by default):")
    for module in FAST_TEST_MODULES:
        print(f"  • {module}")

    print("\n📋 Integration Test Modules (run with --all):")
    for module in INTEGRATION_TEST_MODULES:
        print(f"  • {module}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fast test runner for rapid development feedback")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including integration tests",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test modules",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    args = parser.parse_args()

    if args.list:
        list_modules()
        return 0

    # Determine which modules to run
    modules = FAST_TEST_MODULES.copy()
    if args.all:
        modules.extend(INTEGRATION_TEST_MODULES)

    print("=" * 60)
    print("🚀 FAST TEST RUNNER")
    print("=" * 60)
    print(f"Running {len(modules)} test modules...")
    print()

    total_start = time.time()
    passed = 0
    failed = 0
    total_tests = 0
    failed_modules: list[str] = []

    for module_name in modules:
        if args.verbose:
            print(f"  Testing {module_name}...", end=" ", flush=True)

        success, duration, test_count = run_module_tests(module_name)
        total_tests += test_count

        if success:
            passed += 1
            if args.verbose:
                print(f"✅ ({duration:.2f}s)")
        else:
            failed += 1
            failed_modules.append(module_name)
            if args.verbose:
                print(f"❌ ({duration:.2f}s)")

    total_duration = time.time() - total_start

    # Summary
    print()
    print("=" * 60)
    print("📊 FAST TEST SUMMARY")
    print("=" * 60)
    print(f"⏰ Duration: {total_duration:.2f}s")
    print(f"🧪 Modules: {passed + failed} ({passed} passed, {failed} failed)")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed_modules:
        print()
        print("Failed modules:")
        for module in failed_modules:
            print(f"  • {module}")

    if failed == 0:
        print()
        print("🎉 ALL FAST TESTS PASSED!")
        if not args.all:
            print("💡 Run with --all for full test coverage")
    else:
        print()
        print("⚠️  Some tests failed. Run full suite for details:")
        print("   python run_all_tests.py")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
