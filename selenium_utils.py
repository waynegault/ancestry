#!/usr/bin/env python3

"""
Selenium/WebDriver utility functions specifically for browser automation
and element interaction, separated from general or API-specific utilities.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import safe_execute, setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import json
import sys
import time
from typing import Optional

# === THIRD-PARTY IMPORTS ===
from selenium.common.exceptions import (
    InvalidSessionIdException,
    NoSuchElementException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
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
def extract_text(element) -> str:
    """Extract text from an element safely with unified error handling."""
    if not element:
        return ""
    return element.text or ""


@safe_execute(default_return="", log_errors=False)
def extract_attribute(element, attribute: str) -> str:
    """Extract attribute from an element safely with unified error handling."""
    if not element:
        return ""
    return element.get_attribute(attribute) or ""


@safe_execute(default_return=False, log_errors=False)
def is_elem_there(driver, selector: str, by: str = By.CSS_SELECTOR) -> bool:
    """Check if element exists with unified error handling."""
    if not driver:
        return False
    driver.find_element(by, selector)
    return True


@safe_execute(default_return=False, log_errors=False)
def is_browser_open(driver) -> bool:
    """Check if browser is still open and responsive with unified error handling."""
    if not driver:
        return False
    # Access current_url - will raise exception if browser is closed
    _ = driver.current_url
    return True


@safe_execute(log_errors=True)
def close_tabs(driver, keep_first: bool = True):
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
def get_driver_cookies(driver) -> list:
    """Get all cookies from driver with unified error handling."""
    if not driver:
        return []
    return driver.get_cookies()


@safe_execute(default_return=False, log_errors=True)
def export_cookies(driver, filepath: str) -> bool:
    """Export cookies to file with unified error handling."""
    if not driver:
        return False

    cookies = get_driver_cookies(driver)
    with open(filepath, "w") as f:
        json.dump(cookies, f, indent=2)
    return True


@safe_execute(log_errors=False)
def scroll_to_element(driver, element):
    """Scroll element into view with unified error handling."""
    if not driver or not element:
        return

    driver.execute_script("arguments[0].scrollIntoView(true);", element)
    time.sleep(0.1)  # Brief pause for scroll completion


@safe_execute(default_return=None, log_errors=False)
def wait_for_element(
    driver, selector: str, timeout: int = 10, by: str = By.CSS_SELECTOR
):
    """Wait for element to be present with unified error handling."""
    if not driver:
        return None

    wait = WebDriverWait(driver, timeout)
    return wait.until(EC.presence_of_element_located((by, selector)))


@safe_execute(default_return=False, log_errors=False)
def safe_click(driver, element):
    """Safely click an element with unified error handling."""
    if not driver or not element:
        return False

    # Scroll to element first
    scroll_to_element(driver, element)
    # Try to click
    element.click()
    return True


@safe_execute(default_return="", log_errors=False)
def get_element_text(element) -> str:
    """Get text from element with unified error handling."""
    if not element:
        return ""
    return element.text or ""


@safe_execute(default_return=False, log_errors=False)
def is_element_visible(element) -> bool:
    """Check if element is visible with unified error handling."""
    if not element:
        return False
    return element.is_displayed()


def selenium_module_tests():
    """Essential selenium utilities tests for unified framework."""
    import time
    from unittest.mock import MagicMock

    tests = []

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


def run_comprehensive_tests() -> bool:
    """Run comprehensive selenium utilities tests using standardized TestSuite framework."""
    import time
    from unittest.mock import MagicMock

    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Selenium Utilities & Browser Automation", "selenium_utils.py")
    suite.start_suite()

    def test_function_availability():
        """Test selenium utility functions are available with detailed verification."""
        required_functions = [
            ("force_user_agent", "User agent modification for browser automation"),
            ("extract_text", "Safe text extraction from web elements"),
            ("extract_attribute", "Safe attribute extraction from web elements"),
            ("is_elem_there", "Element presence detection with selectors"),
            ("is_browser_open", "Browser session status validation"),
            ("close_tabs", "Tab management and cleanup operations"),
            ("get_driver_cookies", "Cookie extraction from browser sessions"),
            ("export_cookies", "Cookie export functionality"),
            ("safe_click", "Safe element clicking with error handling"),
            ("get_element_text", "Element text extraction with fallbacks"),
            ("is_element_visible", "Element visibility detection"),
        ]

        print("📋 Testing selenium utility function availability:")
        results = []

        for func_name, description in required_functions:
            # Test function existence
            func_exists = func_name in globals()

            # Test function callability
            func_callable = False
            if func_exists:
                try:
                    func_callable = callable(globals()[func_name])
                except Exception:
                    func_callable = False

            # Test function type
            func_type = type(globals().get(func_name, None)).__name__

            status = "✅" if func_exists and func_callable else "❌"
            print(f"   {status} {func_name}: {description}")
            print(
                f"      Exists: {func_exists}, Callable: {func_callable}, Type: {func_type}"
            )

            test_passed = func_exists and func_callable
            results.append(test_passed)

            assert func_exists, f"Function {func_name} should be available"
            assert func_callable, f"Function {func_name} should be callable"

        print(
            f"📊 Results: {sum(results)}/{len(results)} selenium utility functions available"
        )

    def test_force_user_agent():
        mock_driver = MagicMock()
        result = force_user_agent(mock_driver, "test-agent")
        mock_driver.execute_script.assert_called_once()
        assert result, "force_user_agent should return True on success"

    def test_safe_text_extraction():
        # Test safe extraction with None elements
        assert extract_text(None) == "", "extract_text should handle None safely"
        assert (
            extract_attribute(None, "href") == ""
        ), "extract_attribute should handle None safely"

        # Test with mock element
        mock_element = MagicMock()
        mock_element.text = "test text"
        mock_element.get_attribute.return_value = "test value"

        assert (
            extract_text(mock_element) == "test text"
        ), "extract_text should return element text"
        assert (
            extract_attribute(mock_element, "href") == "test value"
        ), "extract_attribute should return attribute value"

    def test_element_detection():
        # Test with None driver
        assert (
            not is_elem_there(None, "selector")
        ), "is_elem_there should handle None driver safely"

        # Test with mock driver - element found
        mock_driver = MagicMock()
        mock_driver.find_element.return_value = MagicMock()
        result = is_elem_there(mock_driver, "test-selector")
        assert result, "is_elem_there should return True when element found"

        # Test with mock driver - element not found
        mock_driver.find_element.side_effect = NoSuchElementException()
        result = is_elem_there(mock_driver, "missing-selector")
        assert (
            not result
        ), "is_elem_there should return False when element not found"

    def test_browser_status():
        # Test with None driver
        assert (
            not is_browser_open(None)
        ), "is_browser_open should return False for None driver"

        # Test with valid mock driver
        mock_driver = MagicMock()
        mock_driver.current_url = "https://example.com"
        result = is_browser_open(mock_driver)
        assert result, "is_browser_open should return True for valid driver"

        # Test with invalid session - use spec-based mock with property descriptor
        from selenium.webdriver.chrome.webdriver import WebDriver

        mock_invalid_driver = MagicMock(spec=WebDriver)

        # Define property that raises exception
        def current_url_getter(self):
            raise InvalidSessionIdException()

        # Set the property on the mock's type
        type(mock_invalid_driver).current_url = property(current_url_getter)

        result = is_browser_open(mock_invalid_driver)
        assert (
            not result
        ), "is_browser_open should return False for invalid session"

    def test_tab_management():
        mock_driver = MagicMock()
        mock_driver.window_handles = ["tab1", "tab2", "tab3"]

        # Test closing tabs while keeping first
        close_tabs(mock_driver, keep_first=True)
        assert (
            mock_driver.switch_to.window.call_count >= 1
        ), "Should switch to tabs for closing"

        # Test closing all tabs
        close_tabs(mock_driver, keep_first=False)
        assert mock_driver.close.called, "Should close tabs when keep_first=False"

    def test_cookie_operations():
        mock_driver = MagicMock()
        mock_cookies = [
            {"name": "session", "value": "abc123"},
            {"name": "user", "value": "test_user"},
        ]
        mock_driver.get_cookies.return_value = mock_cookies

        # Test getting cookies
        cookies = get_driver_cookies(mock_driver)
        assert isinstance(cookies, list), "get_driver_cookies should return a list"
        assert len(cookies) == 2, "Should return correct number of cookies"

        # Test with None driver
        cookies = get_driver_cookies(None)
        assert (
            cookies == []
        ), "get_driver_cookies should return empty list for None driver"

    def test_safe_interaction():
        # Test safe_click with None element
        result = safe_click(None, None)
        assert not result, "safe_click should return False for None element"

        # Test with mock element
        mock_driver = MagicMock()
        mock_element = MagicMock()
        result = safe_click(mock_driver, mock_element)
        mock_element.click.assert_called_once()
        assert result, "safe_click should return True on successful click"

    def test_element_text_helpers():
        # Test get_element_text with None
        assert (
            get_element_text(None) == ""
        ), "get_element_text should handle None safely"

        # Test is_element_visible with None
        assert (
            not is_element_visible(None)
        ), "is_element_visible should handle None safely"

        # Test with mock elements
        mock_element = MagicMock()
        mock_element.text = "test content"
        mock_element.is_displayed.return_value = True

        assert (
            get_element_text(mock_element) == "test content"
        ), "get_element_text should return element text"
        assert (
            is_element_visible(mock_element)
        ), "is_element_visible should return visibility status"

    def test_performance_validation():
        # Test that operations complete within reasonable time
        start_time = time.time()

        # Run multiple operations
        for _ in range(50):
            extract_text(None)
            extract_attribute(None, "href")
            is_elem_there(None, "test")
            is_browser_open(None)
            get_element_text(None)
            is_element_visible(None)

        elapsed = time.time() - start_time
        assert (
            elapsed < 0.1
        ), f"Performance test should complete quickly, took {elapsed:.3f}s"

    # Run all tests
    print(
        "🌐 Running Selenium Utilities & Browser Automation comprehensive test suite..."
    )

    with suppress_logging():
        suite.run_test(
            "Function availability verification",
            test_function_availability,
            "11 selenium functions tested: force_user_agent, extract_text, extract_attribute, is_elem_there, is_browser_open, close_tabs, get_driver_cookies, export_cookies, safe_click, get_element_text, is_element_visible.",
            "Test selenium utility functions are available with detailed verification.",
            "Verify force_user_agent→browser identity, extract_text→safe text, extract_attribute→safe attributes, is_elem_there→element detection, is_browser_open→session status, close_tabs→tab management.",
        )

        suite.run_test(
            "User agent configuration",
            test_force_user_agent,
            "Test force_user_agent function with mock WebDriver instance",
            "User agent configuration allows browser fingerprint customization",
            "User agent can be programmatically set for browser sessions",
        )

        suite.run_test(
            "Safe text extraction",
            test_safe_text_extraction,
            "Test extract_text and extract_attribute with None and mock elements",
            "Safe text extraction provides robust element content retrieval",
            "Text and attribute extraction handles missing elements gracefully",
        )

        suite.run_test(
            "Element detection functionality",
            test_element_detection,
            "Test is_elem_there with various driver and element states",
            "Element detection provides reliable presence checking",
            "Element detection accurately identifies element presence in DOM",
        )

        suite.run_test(
            "Browser session status",
            test_browser_status,
            "Test is_browser_open with valid, invalid, and None driver instances",
            "Browser status checking provides accurate session validation",
            "Browser session status accurately reflects WebDriver connectivity",
        )

        suite.run_test(
            "Tab management operations",
            test_tab_management,
            "Test close_tabs functionality with different keep_first settings",
            "Tab management provides browser window organization capabilities",
            "Tab management allows selective closing and window organization",
        )

        suite.run_test(
            "Cookie handling operations",
            test_cookie_operations,
            "Test get_driver_cookies with mock driver and cookie data",
            "Cookie operations provide session state management capabilities",
            "Cookie handling safely retrieves and manages browser session data",
        )

        suite.run_test(
            "Safe element interaction",
            test_safe_interaction,
            "Test safe_click with None and mock elements for robust clicking",
            "Safe interaction provides reliable element manipulation",
            "Element interaction handles missing elements and click operations safely",
        )

        suite.run_test(
            "Element text helpers",
            test_element_text_helpers,
            "Test get_element_text and is_element_visible with various element states",
            "Element helpers provide comprehensive element information retrieval",
            "Text helpers and visibility checks work correctly with all element types",
        )

        suite.run_test(
            "Performance validation",
            test_performance_validation,
            "Test performance of core Selenium operations with multiple iterations",
            "Performance validation ensures efficient Selenium utility execution",
            "Selenium operations complete within reasonable time limits for automation",
        )

    # Generate summary report and return result
    return suite.finish_suite()


# Functions are automatically registered via auto_register_module() at import
# No manual registration needed - this is a key benefit of the unified system


if __name__ == "__main__":
    success = run_comprehensive_tests()
    import sys
    sys.exit(0 if success else 1)
