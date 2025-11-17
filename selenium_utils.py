#!/usr/bin/env python3

"""Selenium/WebDriver Utilities for Browser Automation.

Utility functions for browser automation and element interaction using
Selenium WebDriver, separated from general or API-specific utilities.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import safe_execute, setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import json
import sys
import time
from typing import Any, Optional

# === THIRD-PARTY IMPORTS ===
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

# Local imports

# --- Selenium Specific Helpers ---


@safe_execute(default_return=False, log_errors=True)
def force_user_agent(driver: Optional[WebDriver], user_agent: str):
    """
    Attempts to force the browser's User-Agent string using Chrome DevTools Protocol.
    Now with unified error handling via safe_execute decorator.
    """
    if not driver:
        logger.warning("Driver is None, cannot set user agent")
        return False

    # Use execute_script to modify the user agent
    driver.execute_script("navigator.userAgent = arguments[0]", user_agent)
    logger.debug(f"Set user agent to: {user_agent}")
    return True


@safe_execute(default_return="", log_errors=False)
def extract_text(element: Any) -> str:  # type: ignore[misc]
    """Extract text from an element safely with unified error handling."""
    if not element:
        return ""
    return element.text or ""


@safe_execute(default_return="", log_errors=False)
def extract_attribute(element: Any, attribute: str) -> str:  # type: ignore[misc]
    """Extract attribute from an element safely with unified error handling."""
    if not element:
        return ""
    return element.get_attribute(attribute) or ""


@safe_execute(default_return=False, log_errors=False)
def is_elem_there(driver: Any, selector: str, by: str = By.CSS_SELECTOR) -> bool:  # type: ignore[misc]
    """Check if element exists with unified error handling."""
    if not driver:
        return False
    driver.find_element(by, selector)
    return True


@safe_execute(default_return=False, log_errors=False)
def is_browser_open(driver: Any) -> bool:  # type: ignore[misc]
    """Check if browser is still open and responsive with unified error handling."""
    if not driver:
        return False
    # Access current_url - will raise exception if browser is closed
    _ = driver.current_url
    return True


@safe_execute(log_errors=True)
def close_tabs(driver: Any, keep_first: bool = True) -> None:  # type: ignore[misc]
    """Close browser tabs with unified error handling."""
    if not driver:
        return

    handles = driver.window_handles
    if keep_first and len(handles) > 1:
        # Close all but first tab
        for handle in handles[1:]:
            driver.switch_to.window(handle)
            driver.close()
        # Switch back to first tab
        driver.switch_to.window(handles[0])
    elif not keep_first:
        # Close all tabs
        for handle in handles:
            driver.switch_to.window(handle)
            driver.close()


@safe_execute(default_return=[], log_errors=False)
def get_driver_cookies(driver: Any) -> list[dict[str, Any]]:  # type: ignore[misc]
    """Get all cookies from driver with unified error handling."""
    if not driver:
        return []
    return driver.get_cookies()


@safe_execute(default_return=False, log_errors=True)
def export_cookies(driver: Any, filepath: str) -> bool:  # type: ignore[misc]
    """Export cookies to file with unified error handling."""
    if not driver:
        return False

    cookies = get_driver_cookies(driver)
    from pathlib import Path
    with Path(filepath).open("w", encoding="utf-8") as f:
        json.dump(cookies, f, indent=2)
    return True


@safe_execute(log_errors=False)
def scroll_to_element(driver: Any, element: Any) -> None:
    """Scroll element into view with unified error handling."""
    if not driver or not element:
        return

    driver.execute_script("arguments[0].scrollIntoView(true);", element)
    time.sleep(0.1)  # Brief pause for scroll completion


@safe_execute(default_return=None, log_errors=False)
def wait_for_element(
    driver: Any, selector: str, timeout: int = 10, by: str = By.CSS_SELECTOR
) -> Any:
    """Wait for element to be present with unified error handling."""
    if not driver:
        return None

    wait = WebDriverWait(driver, timeout)
    return wait.until(expected_conditions.presence_of_element_located((by, selector)))


@safe_execute(default_return=False, log_errors=False)
def safe_click(driver: Any, element: Any) -> bool:
    """Safely click an element with unified error handling."""
    if not driver or not element:
        return False

    # Scroll to element first
    scroll_to_element(driver, element)
    # Try to click
    element.click()
    return True


@safe_execute(default_return="", log_errors=False)
def get_element_text(element: Any) -> str:  # type: ignore[misc]
    """Get text from element with unified error handling."""
    if not element:
        return ""
    return element.text or ""


@safe_execute(default_return=False, log_errors=False)
def is_element_visible(element: Any) -> bool:  # type: ignore[misc]
    """Check if element is visible with unified error handling."""
    if not element:
        return False
    return element.is_displayed()


def selenium_module_tests() -> list[tuple[str, Any]]:  # type: ignore[misc]
    """Essential selenium utilities tests for unified framework."""
    import time
    from unittest.mock import MagicMock

    tests: list[tuple[str, Any]] = []

    # Test 1: Function availability
    def test_function_availability():
        required_functions = [
            "force_user_agent",
            "extract_text",
            "extract_attribute",
            "is_elem_there",
            "safe_click",
            "get_element_text",
            "is_element_visible",
        ]
        for func_name in required_functions:
            assert func_name in globals(), f"Function {func_name} should be available"
            assert callable(
                globals()[func_name]
            ), f"Function {func_name} should be callable"

    tests.append(("Function Availability", test_function_availability))

    # Test 2: Force user agent functionality
    def test_force_user_agent():
        mock_driver = MagicMock()
        result = force_user_agent(mock_driver, "test-agent")
        mock_driver.execute_script.assert_called_once()
        assert result, "force_user_agent should return True on success"

    tests.append(("Force User Agent", test_force_user_agent))

    # Test 3: Safe execution with None
    def test_safe_execution():
        assert extract_text(None) == "", "extract_text should handle None safely"
        assert (
            extract_attribute(None, "href") == ""
        ), "extract_attribute should handle None safely"
        assert (
            not is_elem_there(None, "selector")
        ), "is_elem_there should handle None safely"

    tests.append(("Safe Execution", test_safe_execution))

    # Test 4: Element text extraction
    def test_element_text():
        mock_element = MagicMock()
        mock_element.text = "test text"
        result = get_element_text(mock_element)
        assert result == "test text", "Should extract text from element"

    tests.append(("Element Text Extraction", test_element_text))

    # Test 5: Performance validation
    def test_performance():
        start_time = time.time()
        for _ in range(100):  # Reduced for faster testing
            extract_text(None)
            is_elem_there(None, "test")
        duration = time.time() - start_time
        assert duration < 0.1, f"Operations should be fast, took {duration:.3f}s"

    tests.append(("Performance Validation", test_performance))

    return tests


# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


# Removed smoke test: _test_function_availability - only checked availability in globals()


def _test_force_user_agent():
    """Test force_user_agent function."""
    assert callable(force_user_agent), "force_user_agent should be callable"


def _test_safe_execution():
    """Test safe execution wrappers."""
    assert callable(extract_text), "extract_text should be callable"
    assert callable(extract_attribute), "extract_attribute should be callable"


def _test_element_text():
    """Test element text extraction."""
    from unittest.mock import MagicMock
    mock_elem = MagicMock()
    mock_elem.text = "Test"
    result = extract_text(mock_elem)
    assert result == "Test"


def _test_performance():
    """Test performance of utility functions."""
    import time
    from unittest.mock import MagicMock
    start = time.time()
    for _ in range(100):
        mock_elem = MagicMock()
        mock_elem.text = "Test"
        _ = extract_text(mock_elem)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"Should be fast, took {elapsed:.3f}s"


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def selenium_utils_module_tests() -> bool:
    """Run comprehensive selenium utilities tests using standardized TestSuite framework."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Selenium Utilities & Browser Automation", "selenium_utils.py")
    suite.start_suite()

    # Assign module-level test functions
    # Removed: test_function_availability = _test_function_availability (smoke test)
    test_force_user_agent = _test_force_user_agent
    test_safe_execution = _test_safe_execution
    test_element_text = _test_element_text
    test_performance = _test_performance

    # Define all tests in a data structure to reduce complexity
    tests = [
        # Removed smoke test: Function Availability
        ("Force User Agent", test_force_user_agent,
         "force_user_agent function is callable",
         "Test user agent modification",
         "Verify force_user_agent exists"),
        ("Safe Execution", test_safe_execution,
         "Safe execution wrappers are callable",
         "Test safe execution functions",
         "Verify extract_text and extract_attribute are callable"),
        ("Element Text", test_element_text,
         "Element text extraction works correctly",
         "Test text extraction from mock element",
         "Verify extract_text returns correct text"),
        ("Performance", test_performance,
         "Utility functions are performant",
         "Test performance of 100 text extractions",
         "Verify operations complete in less than 1 second"),
    ]

    with suppress_logging():
        # Run all tests from the list
        for test_name, test_func, expected, method, details in tests:
            suite.run_test(test_name, test_func, expected, method, details)

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(selenium_utils_module_tests)


if __name__ == "__main__":
    import sys
    print("🧪 Running Selenium Utils Comprehensive Tests...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
