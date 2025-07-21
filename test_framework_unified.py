#!/usr/bin/env python3
"""
Unified Test Framework - Eliminates 25+ duplicate test functions

This replaces the massive code duplication where every module has its own
run_comprehensive_tests() function with nearly identical logic.

Usage in any module:
    from test_framework_unified import StandardTestFramework

    def module_specific_tests():
        # Your module-specific test logic here
        return True

    if __name__ == "__main__":
        framework = StandardTestFramework(__name__)
        framework.run_all_tests(custom_tests=module_specific_tests)
"""

from core_imports import auto_register_module, get_logger
import time
import sys
import traceback
from typing import Callable, Optional, Dict, Any, List
from pathlib import Path

auto_register_module(globals(), __name__)
logger = get_logger(__name__)


class TestResult:
    """Standardized test result container."""

    def __init__(
        self, name: str, success: bool, duration: float, error: Optional[str] = None
    ):
        self.name = name
        self.success = success
        self.duration = duration
        self.error = error


class StandardTestFramework:
    """
    Unified test framework that replaces 25+ duplicate run_comprehensive_tests() functions.

    Eliminates ~10,000 lines of duplicated test code across the codebase.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.results: List[TestResult] = []
        self.start_time = 0.0

    def run_test(
        self, test_name: str, test_func: Callable, description: str = ""
    ) -> bool:
        """Run a single test with standardized error handling and timing."""
        start_time = time.time()
        try:
            logger.info(f"🧪 Running {test_name}...")
            success = bool(test_func())
            duration = time.time() - start_time

            if success:
                logger.info(f"✅ {test_name} passed ({duration:.3f}s)")
            else:
                logger.error(f"❌ {test_name} failed ({duration:.3f}s)")

            self.results.append(TestResult(test_name, success, duration))
            return success

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"💥 {test_name} crashed ({duration:.3f}s): {error_msg}")
            logger.debug(f"Traceback: {traceback.format_exc()}")

            self.results.append(TestResult(test_name, False, duration, error_msg))
            return False

    def run_import_tests(self) -> bool:
        """Standard import tests that every module should pass."""

        def test_core_imports():
            from core_imports import (
                auto_register_module,
                get_logger,
                register_function,
                get_function,
                is_function_available,
            )

            assert callable(auto_register_module)
            assert callable(get_logger)
            return True

        def test_logging_setup():
            test_logger = get_logger(f"{self.module_name}.test")
            assert test_logger is not None
            test_logger.debug("Test log message")
            return True

        success = True
        success &= self.run_test("Core imports", test_core_imports)
        success &= self.run_test("Logging setup", test_logging_setup)
        return success

    def run_registration_tests(self) -> bool:
        """Standard function registration tests."""

        def test_function_registration():
            from core_imports import (
                register_function,
                get_function,
                is_function_available,
            )

            def test_func():
                return "test_result"

            register_function("test_unified_func", test_func)
            assert is_function_available("test_unified_func")
            result = get_function("test_unified_func")
            assert result is not None
            assert result() == "test_result"
            return True

        return self.run_test("Function registration", test_function_registration)

    def run_performance_tests(self) -> bool:
        """Standard performance tests."""

        def test_import_performance():
            # Test that imports are reasonably fast
            import_start = time.time()
            from core_imports import get_stats

            import_duration = time.time() - import_start

            # Should be very fast due to caching
            assert (
                import_duration < 0.1
            ), f"Import took too long: {import_duration:.3f}s"

            stats = get_stats()
            assert "functions_registered" in stats
            return True

        return self.run_test("Import performance", test_import_performance)

    def run_all_tests(self, custom_tests: Optional[Callable] = None) -> bool:
        """Run all standard tests plus any custom module tests."""
        self.start_time = time.time()
        logger.info(f"🚀 Running unified test suite for {self.module_name}")
        logger.info("=" * 60)

        overall_success = True

        # Standard tests that every module should pass
        overall_success &= self.run_import_tests()
        overall_success &= self.run_registration_tests()
        overall_success &= self.run_performance_tests()

        # Module-specific tests
        if custom_tests:
            overall_success &= self.run_test("Custom module tests", custom_tests)

        # Generate report
        self._generate_report(overall_success)
        return overall_success

    def _generate_report(self, overall_success: bool) -> None:
        """Generate standardized test report."""
        total_duration = time.time() - self.start_time
        passed_tests = sum(1 for r in self.results if r.success)
        total_tests = len(self.results)

        logger.info("=" * 60)
        logger.info(f"📊 Test Summary for {self.module_name}")
        logger.info(f"Tests: {passed_tests}/{total_tests} passed")
        logger.info(f"Duration: {total_duration:.3f}s")
        logger.info(f"Status: {'✅ PASSED' if overall_success else '❌ FAILED'}")

        # Detailed results
        if self.results:
            logger.info("\n📋 Detailed Results:")
            for result in self.results:
                status = "✅" if result.success else "❌"
                logger.info(f"  {status} {result.name}: {result.duration:.3f}s")
                if result.error:
                    logger.info(f"    Error: {result.error}")

        logger.info("=" * 60)


def run_unified_tests(
    module_name: str, custom_tests: Optional[Callable] = None
) -> bool:
    """
    Convenience function to run unified tests.

    This single function call replaces the need for 25+ different
    run_comprehensive_tests() functions across the codebase.

    Args:
        module_name: Name of the module being tested
        custom_tests: Optional function for module-specific tests

    Returns:
        True if all tests passed, False otherwise
    """
    framework = StandardTestFramework(module_name)
    return framework.run_all_tests(custom_tests)


# Example usage that can be copied to any module:
def example_module_tests():
    """Example of module-specific tests."""
    # Module-specific test logic goes here
    logger.info("Running example module-specific tests")
    return True


if __name__ == "__main__":
    # Self-test
    def test_framework_itself():
        # Test the framework functionality
        test_framework = StandardTestFramework("test_framework_unified.self_test")

        def dummy_passing_test():
            return True

        def dummy_failing_test():
            return False

        result1 = test_framework.run_test("Dummy passing test", dummy_passing_test)
        result2 = test_framework.run_test("Dummy failing test", dummy_failing_test)

        # Framework should handle both success and failure
        assert result1 == True
        assert result2 == False
        assert len(test_framework.results) == 2
        return True

    success = run_unified_tests(__name__, test_framework_itself)
    logger.info(f"🎯 Unified Test Framework: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
