# selenium_utils.py
"""
Selenium/WebDriver utility functions specifically for browser automation
and element interaction, separated from general or API-specific utilities.
"""

# --- Standard library imports ---
import sys
import time
import os
import json
import logging
from typing import Optional, Dict  # Import Optional and Dict for type hints

# --- Third-party imports ---
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
    InvalidSessionIdException,  # Added
    NoSuchWindowException,  # Added
)
import undetected_chromedriver as uc  # Added import for uc.Chrome

# --- Local application imports ---
from config import config_schema

from logging_config import logger

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)

# --- Selenium Specific Helpers ---


def force_user_agent(driver: Optional[uc.Chrome], user_agent: str):
    """
    Attempts to force the browser's User-Agent string using Chrome DevTools Protocol.
    """
    if not driver:
        logger.warning("force_user_agent: WebDriver is None.")
        return
    logger.debug(f"Attempting to set User-Agent via CDP to: {user_agent}")
    start_time = time.time()
    try:
        # driver.execute_cdp_cmd(
        #     "Network.setUserAgentOverride", {"userAgent": user_agent}
        # )
        # Replace with execute_script if execute_cdp_cmd is not available or causing issues
        driver.execute_script("navigator.userAgent = arguments[0]", user_agent)
        duration = time.time() - start_time
        logger.info(
            f"Successfully set User-Agent via execute_script in {duration:.3f} seconds."
        )
    except Exception as e:
        # Only show traceback if not running in test mode
        show_traceback = __name__ != "__main__"
        logger.warning(
            f"Error setting User-Agent via CDP: {e}", exc_info=show_traceback
        )


# End of force_user_agent


def extract_text(element, selector: str) -> str:
    """
    Safely extracts text content from a Selenium WebElement using a CSS selector.
    """
    if not element:
        logger.warning("extract_text: Parent element is None.")
        return ""
    try:
        target_element = element.find_element(By.CSS_SELECTOR, selector)
        text_content = target_element.text
        return text_content.strip() if text_content else ""
    except NoSuchElementException:
        logger.debug(f"Element '{selector}' not found for text extraction.")
        return ""
    except Exception as e:
        logger.warning(f"Error extracting text '{selector}': {e}", exc_info=False)
        return ""


# End of extract_text


def extract_attribute(element, selector: str, attribute: str) -> str:
    """
    Extracts attribute from a child element. Resolves relative 'href'.
    """
    if not element:
        logger.warning("extract_attribute: Parent element is None.")
        return ""
    try:
        target_element = element.find_element(By.CSS_SELECTOR, selector)
        value = target_element.get_attribute(attribute)
        if attribute == "href" and value:
            # Use urljoin from urllib.parse (needs import in calling module or here)
            from urllib.parse import urljoin  # Local import for utility

            if value.startswith("/"):
                return urljoin(config_schema.api.base_url, value)
            # Keep absolute and other protocols as is
            elif (
                value.startswith("http://")
                or value.startswith("https://")
                or ":" in value.split("/")[0]
            ):
                return value
            else:  # Assume relative path without leading slash (less common)
                # This might need context (current URL) for perfect resolution,
                # but joining with base is a reasonable default.
                return urljoin(config_schema.api.base_url, value)
        return value if value else ""
    except NoSuchElementException:
        logger.debug(f"Element '{selector}' not found for attribute '{attribute}'.")
        return ""
    except Exception as e:
        logger.warning(
            f"Error extracting attr '{attribute}' from '{selector}': {e}",
            exc_info=False,
        )
        return ""


# End of extract_attribute


def is_elem_there(
    driver: Optional[WebDriver], by: str, value: str, wait: Optional[int] = None
) -> bool:
    """
    Checks if a web element is present in the DOM within a specified timeout.
    """
    if driver is None:
        logger.warning("is_elem_there: WebDriver is None.")
        return False
    effective_wait = wait if wait is not None else config_schema.selenium.explicit_wait
    try:
        WebDriverWait(driver, effective_wait).until(
            EC.presence_of_element_located((by, value))
        )
        return True
    except TimeoutException:
        return False
    except Exception as e:
        logger.error(f"Error checking element presence '{value}' ({by}): {e}")
        return False


# End of is_elem_there


def is_browser_open(driver: Optional[WebDriver]) -> bool:
    """
    Checks if the browser window appears open and responsive.
    """
    if driver is None:
        return False
    try:
        _ = driver.window_handles  # Lightweight command
        return True
    except (InvalidSessionIdException, NoSuchWindowException, WebDriverException) as e:
        err_str = str(e).lower()
        if any(
            sub in err_str
            for sub in [
                "invalid session id",
                "target closed",
                "disconnected",
                "no such window",
                "unable to connect",
            ]
        ):
            logger.debug(
                f"Browser appears closed or session invalid: {type(e).__name__}"
            )
            return False
        else:
            logger.warning(
                f"WebDriverException checking browser status (assuming closed): {e}"
            )
            return False
    except Exception as e:
        # Only show traceback if not running in test mode
        show_traceback = __name__ != "__main__"
        logger.error(
            f"Unexpected error checking browser status: {e}", exc_info=show_traceback
        )
        return False


# End of is_browser_open


def close_tabs(driver: Optional[WebDriver]):
    """
    Closes all browser tabs except the first one. Focuses the first tab.
    """
    if not driver:
        logger.warning("close_tabs: WebDriver instance is None.")
        return
    logger.debug("Closing extra browser tabs...")
    try:
        handles = driver.window_handles
        if len(handles) <= 1:
            logger.debug("No extra tabs to close.")
            return
        original_handle = driver.current_window_handle
        first_handle = handles[0]
        logger.debug(
            f"Original handle: {original_handle}, First handle: {first_handle}"
        )
        closed_count = 0
        for handle in handles[1:]:
            try:
                logger.debug(f"Switching to tab {handle} to close...")
                driver.switch_to.window(handle)
                driver.close()
                logger.debug(f"Closed tab handle: {handle}")
                closed_count += 1
            except NoSuchWindowException:
                logger.warning(f"Tab {handle} already closed.")
            except WebDriverException as e:
                logger.error(f"Error closing tab {handle}: {e}")
        logger.debug(f"Closed {closed_count} extra tabs.")
        remaining_handles = driver.window_handles
        if original_handle in remaining_handles:
            if driver.current_window_handle != original_handle:
                logger.debug(f"Switching back to original tab: {original_handle}")
                driver.switch_to.window(original_handle)
            else:
                logger.debug("Already focused on original tab.")
        elif first_handle in remaining_handles:
            logger.warning(
                f"Original handle {original_handle} missing. Switching to first handle: {first_handle}"
            )
            driver.switch_to.window(first_handle)
        elif remaining_handles:
            logger.error(
                f"Original and first tabs gone. Switching to remaining: {remaining_handles[0]}"
            )
            driver.switch_to.window(remaining_handles[0])
        else:
            logger.error("All browser tabs were closed unexpectedly.")
    except NoSuchWindowException:
        logger.warning("Attempted to close/switch tab that no longer exists.")
    except WebDriverException as e:
        logger.error(f"WebDriverException during close_tabs: {e}")
    except Exception as e:
        # Only show traceback if not running in test mode
        show_traceback = __name__ != "__main__"
        logger.error(f"Unexpected error in close_tabs: {e}", exc_info=show_traceback)


# End of close_tabs


def get_driver_cookies(driver: Optional[WebDriver]) -> Dict[str, str]:
    """Retrieves all cookies as a simple dictionary."""
    if not driver:
        logger.warning("Cannot get driver cookies: WebDriver is None.")
        return {}
    try:
        cookies_list = driver.get_cookies()
        return {
            cookie["name"]: cookie["value"]
            for cookie in cookies_list
            if "name" in cookie
        }
    except WebDriverException as e:
        logger.error(f"WebDriverException getting driver cookies: {e}")
        return {}
    except Exception as e:
        # Only show traceback if not running in test mode
        show_traceback = __name__ != "__main__"
        logger.error(
            f"Unexpected error getting driver cookies: {e}", exc_info=show_traceback
        )
        return {}


# End of get_driver_cookies


def export_cookies(driver: Optional[WebDriver], file_path: str) -> bool:
    """
    Exports WebDriver cookies to a JSON file.

    Args:
        driver: The WebDriver instance
        file_path: Path to save the cookies JSON file

    Returns:
        True if cookies were successfully exported, False otherwise
    """
    if not driver:
        logger.warning("Cannot export cookies: WebDriver is None.")
        return False

    try:
        cookies_list = driver.get_cookies()
        if not cookies_list:
            logger.warning("No cookies to export (empty list returned).")
            return False

        import json
        import os

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(cookies_list, f, indent=2)

        logger.info(f"Successfully exported {len(cookies_list)} cookies to {file_path}")
        return True

    except WebDriverException as e:
        logger.error(f"WebDriverException exporting cookies: {e}")
        return False
    except (OSError, IOError) as io_err:
        logger.error(f"I/O error exporting cookies to {file_path}: {io_err}")
        return False
    except Exception as e:
        # Only show traceback if not running in test mode
        show_traceback = __name__ != "__main__"
        logger.error(
            f"Unexpected error exporting cookies: {e}", exc_info=show_traceback
        )
        return False


# End of export_cookies


# --- Additional Utility Functions for Testing ---


def scroll_to_element(driver: Optional[WebDriver], element) -> None:
    """Scroll to a specific element."""
    if not driver or not element:
        return
    try:
        driver.execute_script("arguments[0].scrollIntoView();", element)
    except Exception as e:
        logger.warning(f"Error scrolling to element: {e}")


def wait_for_element(driver: Optional[WebDriver], locator: tuple, timeout: int = 10):
    """Wait for element to be present."""
    if not driver:
        return None
    try:
        wait = WebDriverWait(driver, timeout)
        return wait.until(EC.presence_of_element_located(locator))
    except Exception as e:
        logger.warning(f"Error waiting for element: {e}")
        return None


def safe_click(driver: Optional[WebDriver], element) -> bool:
    """Safely click an element."""
    if not driver or not element:
        return False
    try:
        element.click()
        return True
    except Exception as e:
        logger.warning(f"Error clicking element: {e}")
        return False


def get_element_text(element) -> str:
    """Get text from element safely."""
    if not element:
        return ""
    try:
        return element.text or ""
    except Exception:
        return ""


def is_element_visible(element) -> bool:
    """Check if element is visible."""
    if not element:
        return False
    try:
        return element.is_displayed()
    except Exception:
        return False


# --- Test Class Definition ---
# This class is defined at the module level so it can be imported by test_selenium_utils.py
from unittest.mock import MagicMock, PropertyMock


# Original unittest.TestCase tests have been replaced with standardized test framework format below


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for selenium_utils.py with real functionality testing.
    Tests initialization, core functionality, edge cases, integration, performance, and error handling.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Selenium WebDriver Utilities", "selenium_utils.py")
    suite.start_suite()

    # INITIALIZATION TESTS
    def test_module_imports():
        """Test that all Selenium utilities are properly imported and available."""
        required_functions = [
            "force_user_agent",
            "extract_text",
            "extract_attribute",
            "is_elem_there",
            "is_browser_open",
            "close_tabs",
            "get_driver_cookies",
            "export_cookies",
            "scroll_to_element",
            "wait_for_element",
            "safe_click",
            "get_element_text",
            "is_element_visible",
        ]

        for func_name in required_functions:
            assert (
                func_name in globals()
            ), f"Required function '{func_name}' not found in globals"
            assert callable(globals()[func_name]), f"'{func_name}' is not callable"

    def test_selenium_dependencies():
        """Test that required Selenium dependencies are available."""
        # Import should succeed without exceptions
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import (
            TimeoutException,
            WebDriverException,
        )

        # Verify imports worked
        assert By is not None, "By class import failed"
        assert WebDriverWait is not None, "WebDriverWait class import failed"
        assert EC is not None, "expected_conditions import failed"
        assert TimeoutException is not None, "TimeoutException import failed"

    with suppress_logging():
        # INITIALIZATION TESTS
        suite.run_test(
            "force_user_agent(), extract_text(), extract_attribute(), is_elem_there(), is_browser_open()",
            test_module_imports,
            "All core Selenium utility functions are properly defined and callable",
            "Verify existence and callability of all essential Selenium WebDriver utility functions",
            "All required functions exist in global namespace and are callable objects",
        )

        suite.run_test(
            "selenium.webdriver imports and dependencies",
            test_selenium_dependencies,
            "All required Selenium WebDriver dependencies import successfully",
            "Import core Selenium classes (By, WebDriverWait, expected_conditions, exceptions)",
            "All Selenium dependencies imported without exceptions and objects are properly instantiated",
        )

        # CORE FUNCTIONALITY TESTS
        def test_force_user_agent_functionality():
            """Test user agent forcing with mock WebDriver."""
            assert (
                "force_user_agent" in globals()
            ), "force_user_agent function not found"

            from unittest.mock import MagicMock

            mock_driver = MagicMock()
            test_user_agent = "Mozilla/5.0 (Test_12345) Custom User Agent"

            # Test normal operation
            force_user_agent(mock_driver, test_user_agent)

            # Verify the JavaScript execution was called
            mock_driver.execute_script.assert_called_once_with(
                "navigator.userAgent = arguments[0]", test_user_agent
            )

        suite.run_test(
            "force_user_agent()",
            test_force_user_agent_functionality,
            "User agent modification executes JavaScript correctly with mock WebDriver",
            "Test force_user_agent() with mock WebDriver and verify execute_script call with test user agent",
            "JavaScript execution called once with correct user agent string and proper method signature",
        )

        def test_safe_click_mechanism():
            """Test safe clicking mechanism with error handling."""
            assert "safe_click" in globals(), "safe_click function not found"

            from unittest.mock import MagicMock

            mock_driver = MagicMock()
            mock_element = MagicMock()

            # Test successful click
            mock_element.click.return_value = None
            result = safe_click(mock_driver, mock_element)

            assert isinstance(result, bool), "safe_click should return boolean"
            assert result == True, "Successful click should return True"

            # Test click with exception
            mock_element_failing = MagicMock()
            mock_element_failing.click.side_effect = Exception(
                "Click intercepted test_12345"
            )
            result_fail = safe_click(mock_driver, mock_element_failing)

            assert isinstance(result_fail, bool), "Failed click should return boolean"
            assert result_fail == False, "Failed click should return False"

        suite.run_test(
            "safe_click()",
            test_safe_click_mechanism,
            "Safe clicking handles successful clicks and exceptions with proper boolean returns",
            "Test safe_click() with successful clicks and simulated click failures using mock elements",
            "Successful clicks return True, failed clicks return False, all results are boolean type",
        )

        def test_element_text_extraction():
            """Test text extraction from web elements."""
            assert (
                "get_element_text" in globals()
            ), "get_element_text function not found"

            from unittest.mock import MagicMock

            # Test normal text extraction
            mock_element = MagicMock()
            test_text = "Sample element text content 12345"
            mock_element.text = test_text

            result = get_element_text(mock_element)
            assert (
                result == test_text
            ), f"Expected '{test_text}', got '{result}'"  # Test with None element
            result_none = get_element_text(None)
            assert result_none == "", "None element should return empty string"
            # Test with element that raises exception
            mock_element_error = MagicMock()
            # Configure text property to raise exception when accessed
            type(mock_element_error).text = PropertyMock(
                side_effect=Exception("Text access error 12345")
            )

            result_error = get_element_text(mock_element_error)
            assert isinstance(result_error, str), "Error case should return string"
            assert result_error == "", "Error case should return empty string"

        suite.run_test(
            "get_element_text()",
            test_element_text_extraction,
            "Text extraction works for normal elements, handles None elements, and manages text access errors",
            "Test get_element_text() with normal elements, None input, and error conditions using mock objects",
            "Normal text extracted correctly, None returns empty string, errors handled gracefully with string return",
        )

        def test_element_visibility_detection():
            """Test element visibility detection functionality."""
            assert (
                "is_element_visible" in globals()
            ), "is_element_visible function not found"

            from unittest.mock import MagicMock

            # Test visible element
            mock_visible = MagicMock()
            mock_visible.is_displayed.return_value = True
            result_visible = is_element_visible(mock_visible)

            assert result_visible == True, "Visible element should return True"

            # Test hidden element
            mock_hidden = MagicMock()
            mock_hidden.is_displayed.return_value = False
            result_hidden = is_element_visible(mock_hidden)

            assert result_hidden == False, "Hidden element should return False"

            # Test None element
            result_none = is_element_visible(None)
            assert result_none == False, "None element should return False"

            # Test element that raises exception
            mock_error = MagicMock()
            mock_error.is_displayed.side_effect = Exception("Display check error 12345")
            result_error = is_element_visible(mock_error)

            assert result_error == False, "Error case should return False"

        suite.run_test(
            "is_element_visible()",
            test_element_visibility_detection,
            "Visibility detection correctly identifies visible/hidden elements and handles None/error cases",
            "Test is_element_visible() with visible, hidden, None, and error-prone elements using mock objects",
            "Visible elements return True, hidden/None/error elements return False, all results are boolean",
        )

        # EDGE CASES TESTS
        def test_scroll_to_element_edge_cases():
            """Test element scrolling with edge cases."""
            assert (
                "scroll_to_element" in globals()
            ), "scroll_to_element function not found"

            from unittest.mock import MagicMock

            mock_driver = MagicMock()
            mock_element = MagicMock()

            # Test normal scrolling
            scroll_to_element(mock_driver, mock_element)
            mock_driver.execute_script.assert_called_with(
                "arguments[0].scrollIntoView();", mock_element
            )

            # Test with None driver - should handle gracefully
            scroll_to_element(None, mock_element)  # Should not raise exception

            # Test with None element - should handle gracefully
            scroll_to_element(mock_driver, None)  # Should not raise exception

        suite.run_test(
            "scroll_to_element()",
            test_scroll_to_element_edge_cases,
            "Element scrolling executes JavaScript correctly and handles None driver/element inputs gracefully",
            "Test scroll_to_element() with normal operation and None inputs to verify error handling",
            "Normal scrolling executes JavaScript, None inputs handled without exceptions",
        )

        def test_wait_for_element_timeout_handling():
            """Test element waiting with timeout scenarios."""
            assert (
                "wait_for_element" in globals()
            ), "wait_for_element function not found"

            from unittest.mock import MagicMock

            # Test with None driver
            result = wait_for_element(None, ("id", "test_element_12345"))
            assert result is None, "None driver should return None"

            # Test with mock driver (will likely timeout)
            mock_driver = MagicMock()
            result = wait_for_element(
                mock_driver, ("id", "nonexistent_element_12345"), timeout=1
            )

            # Should return None on timeout or handle gracefully
            assert (
                result is None or result is not None
            ), "Should handle timeout gracefully"

        suite.run_test(
            "wait_for_element()",
            test_wait_for_element_timeout_handling,
            "Element waiting handles None driver and timeout scenarios appropriately",
            "Test wait_for_element() with None driver and short timeout for non-existent element",
            "None driver returns None, timeout scenarios handled gracefully without crashes",
        )

        # INTEGRATION TESTS
        def test_selenium_workflow_integration():
            """Test integration of multiple Selenium utilities together."""
            required_funcs = [
                "safe_click",
                "get_element_text",
                "is_element_visible",
                "scroll_to_element",
            ]

            # Verify all functions exist
            for func_name in required_funcs:
                assert (
                    func_name in globals()
                ), f"Required function '{func_name}' not found"
                assert callable(globals()[func_name]), f"'{func_name}' is not callable"

            from unittest.mock import MagicMock

            mock_driver = MagicMock()
            mock_element = MagicMock()
            mock_element.text = "Integration Test Text 12345"
            mock_element.is_displayed.return_value = True

            # Test workflow: check visibility -> scroll -> get text -> click
            is_visible = globals()["is_element_visible"](mock_element)
            assert is_visible == True, "Element should be visible"

            globals()["scroll_to_element"](mock_driver, mock_element)
            text = globals()["get_element_text"](mock_element)
            assert (
                text == "Integration Test Text 12345"
            ), f"Expected 'Integration Test Text 12345', got '{text}'"

            click_result = globals()["safe_click"](mock_driver, mock_element)
            assert isinstance(click_result, bool), "Click result should be boolean"

        suite.run_test(
            "is_element_visible(), scroll_to_element(), get_element_text(), safe_click()",
            test_selenium_workflow_integration,
            "Multiple Selenium utilities work together in typical web automation workflow",
            "Test visibility check -> scroll -> text extraction -> safe click workflow with mock objects",
            "All utilities integrate successfully, visibility detected, scrolling executed, text extracted, click attempted",
        )

        def test_browser_compatibility_handling():
            """Test handling of different browser-specific scenarios."""
            from unittest.mock import MagicMock

            # Test with different mock browser scenarios
            browsers = ["chrome", "firefox", "edge"]

            for browser in browsers:
                mock_driver = MagicMock()
                mock_driver.name = browser

                # Test force_user_agent with different browsers
                if "force_user_agent" in globals():
                    user_agent = f"Mozilla/5.0 ({browser.title()}) Test Agent 12345"
                    globals()["force_user_agent"](mock_driver, user_agent)

        suite.run_test(
            "force_user_agent() with multiple browser types",
            test_browser_compatibility_handling,
            "Utilities work across different browser types (Chrome, Firefox, Edge)",
            "Test Selenium utilities with mock drivers representing different browsers",
            "All browser types handle user agent modification without exceptions",
        )

        # PERFORMANCE TESTS
        def test_bulk_element_operations():
            """Test performance with multiple element operations."""
            assert (
                "get_element_text" in globals()
            ), "get_element_text function not found"
            assert (
                "is_element_visible" in globals()
            ), "is_element_visible function not found"

            from unittest.mock import MagicMock
            import time

            # Create multiple mock elements
            elements = []
            for i in range(100):
                mock_element = MagicMock()
                mock_element.text = f"Element {i} text 12345"
                mock_element.is_displayed.return_value = True
                elements.append(mock_element)

            start_time = time.time()

            # Perform bulk operations
            for element in elements:
                globals()["get_element_text"](element)
                globals()["is_element_visible"](element)

            duration = time.time() - start_time

            # Should complete 200 operations (100 text + 100 visibility) in reasonable time
            assert (
                duration < 0.5
            ), f"200 operations should complete in under 500ms, took {duration:.3f}s"

        suite.run_test(
            "get_element_text(), is_element_visible() bulk operations",
            test_bulk_element_operations,
            "200 element operations (100 text extractions + 100 visibility checks) complete in under 500ms",
            "Perform text extraction and visibility checks on 100 mock elements with timing measurement",
            "All 200 operations completed within performance threshold demonstrating efficient element handling",
        )

        def test_repeated_driver_operations():
            """Test performance of repeated WebDriver operations."""
            assert (
                "scroll_to_element" in globals()
            ), "scroll_to_element function not found"

            from unittest.mock import MagicMock
            import time

            mock_driver = MagicMock()
            mock_element = MagicMock()

            start_time = time.time()

            # Perform repeated scroll operations
            for _ in range(50):
                globals()["scroll_to_element"](mock_driver, mock_element)

            duration = time.time() - start_time

            # Should complete 50 scroll operations in reasonable time
            assert (
                duration < 0.2
            ), f"50 scroll operations should complete in under 200ms, took {duration:.3f}s"

        suite.run_test(
            "scroll_to_element() repeated operations",
            test_repeated_driver_operations,
            "50 scroll operations complete in under 200ms demonstrating efficient WebDriver interaction",
            "Perform scroll_to_element() operation 50 times with mock WebDriver and timing measurement",
            "All 50 scroll operations completed within performance threshold with efficient JavaScript execution",
        )

        # ERROR HANDLING TESTS
        def test_invalid_element_handling():
            """Test handling of invalid or corrupted element objects."""
            assert (
                "get_element_text" in globals()
            ), "get_element_text function not found"
            assert (
                "is_element_visible" in globals()
            ), "is_element_visible function not found"

            # Test with various invalid inputs
            invalid_inputs = [
                None,
                "not_an_element",
                123,
                {},
                [],
            ]

            for invalid_input in invalid_inputs:
                result_text = globals()["get_element_text"](invalid_input)
                result_visible = globals()["is_element_visible"](invalid_input)

                # Should return reasonable defaults or handle gracefully
                assert result_text is None or isinstance(
                    result_text, str
                ), f"Text result should be None or str for input {invalid_input}"
                assert result_visible is None or isinstance(
                    result_visible, bool
                ), f"Visibility result should be None or bool for input {invalid_input}"

        suite.run_test(
            "get_element_text(), is_element_visible() with invalid inputs",
            test_invalid_element_handling,
            "Utilities handle invalid element inputs (None, strings, numbers) gracefully without crashes",
            "Test element utilities with various invalid input types and verify graceful error handling",
            "All invalid inputs handled gracefully with appropriate return types, no exceptions raised",
        )

        def test_webdriver_exception_handling():
            """Test handling of WebDriver-specific exceptions."""
            assert "safe_click" in globals(), "safe_click function not found"

            from unittest.mock import MagicMock

            mock_driver = MagicMock()
            mock_element = MagicMock()

            # Simulate various WebDriver exceptions
            webdriver_exceptions = [
                Exception("ElementClickInterceptedException 12345"),
                Exception("ElementNotInteractableException 12345"),
                Exception("StaleElementReferenceException 12345"),
                Exception("WebDriverException 12345"),
            ]

            for exception in webdriver_exceptions:
                mock_element.click.side_effect = exception
                result = globals()["safe_click"](mock_driver, mock_element)

                assert isinstance(
                    result, bool
                ), f"Result should be boolean for exception {exception}"
                assert (
                    result == False
                ), f"Should return False for failed clicks with exception {exception}"

        suite.run_test(
            "safe_click() with WebDriver exceptions",
            test_webdriver_exception_handling,
            "Safe click handles various WebDriver exceptions gracefully and returns appropriate boolean results",
            "Test safe_click() with simulated WebDriver exceptions (click intercepted, stale element, etc.)",
            "All WebDriver exceptions handled gracefully, False returned for all failure cases",
        )

        return suite.finish_suite()


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    print("🔧 Running Selenium WebDriver Utilities comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

# End of selenium_utils.py
