#!/usr/bin/env python3

# test_selenium_utils.py
"""
Unit tests for selenium_utils.py module.

This test module leverages the self-test functionality built into selenium_utils.py
to avoid code duplication and ensure consistent test coverage.
"""

# --- Standard library imports ---
import unittest
import time
import logging

# --- Local application imports ---
from logging_config import logger


def run_tests():
    """
    Run the selenium_utils self-tests and return the test results.

    This function is a wrapper around the self-test functionality in selenium_utils.py.
    """
    print("\n=== Running Selenium Utils Tests ===")

    # Save the original logger level to restore it later
    original_logger_level = logger.level

    try:
        # Temporarily set logger to CRITICAL level to suppress all expected messages during tests
        logger.setLevel(logging.CRITICAL)

        # Import the TestSeleniumUtils class from selenium_utils
        from selenium_utils import TestSeleniumUtils

        # Create a test suite and run the tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSeleniumUtils)
        runner = unittest.TextTestRunner(verbosity=2)

        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()

        # Display test summary
        print("\n=== Test Summary ===")
        print(f"Total tests run: {result.testsRun}")
        print(
            f"Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}"
        )
        print(f"Tests failed: {len(result.failures)}")
        print(f"Tests with errors: {len(result.errors)}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        # Display test coverage information
        print("\n=== Test Coverage ===")
        print("Functions tested:")
        print(
            "- force_user_agent: Tests for valid driver, None driver, and exception cases"
        )
        print(
            "- extract_text: Tests for successful extraction, None element, NoSuchElementException, empty text, and general exception"
        )
        print(
            "- extract_attribute: Tests for successful extraction, None element, href resolution, NoSuchElementException, and general exception"
        )
        print(
            "- is_elem_there: Tests for None driver case (limited due to WebDriverWait mocking complexity)"
        )
        print(
            "- is_browser_open: Tests for open browser, None driver, InvalidSessionIdException, NoSuchWindowException, WebDriverException, and general exception"
        )
        print(
            "- close_tabs: Tests for multiple tabs, None driver, single tab, NoSuchWindowException, WebDriverException, and general exception"
        )
        print(
            "- get_driver_cookies: Tests for valid cookies, None driver, WebDriverException, and general exception"
        )
        print(
            "- export_cookies: Tests for valid cookies, None driver, empty cookies, WebDriverException, file I/O error, and general exception"
        )

        return result.wasSuccessful()

    finally:
        # Restore the original logger level
        logger.setLevel(original_logger_level)


if __name__ == "__main__":
    success = run_tests()

    if success:
        print("\nAll selenium_utils tests passed successfully!")
    else:
        print("\nSome selenium_utils tests failed. See details above.")
# End of test_selenium_utils.py
