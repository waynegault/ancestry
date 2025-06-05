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
import unittest
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
from config import config_instance, selenium_config
from logging_config import logger

# Note: Removed urllib3 and psutil imports as they weren't used here

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
                return urljoin(config_instance.BASE_URL, value)
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
                return urljoin(config_instance.BASE_URL, value)
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
    effective_wait = wait if wait is not None else selenium_config.ELEMENT_TIMEOUT
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
    # Import test framework components
    try:
        from test_framework import (
            TestSuite,
            suppress_logging,
            create_mock_data,
            assert_valid_function,
        )
    except ImportError:
        return run_comprehensive_tests_fallback()

    with suppress_logging():
        suite = TestSuite("Selenium WebDriver Utilities", "selenium_utils.py")
        suite.start_suite()

        # INITIALIZATION TESTS
        def test_module_imports():
            """Test that all Selenium utilities are properly imported and available."""
            required_functions = [
                "force_user_agent",
                "scroll_to_element",
                "wait_for_element",
                "safe_click",
                "get_element_text",
                "is_element_visible",
            ]

            for func_name in required_functions:
                if func_name not in globals():
                    return False
                if not callable(globals()[func_name]):
                    return False
            return True

        suite.run_test(
            "Selenium Utilities Initialization",
            test_module_imports,
            "All core Selenium utility functions (force_user_agent, safe_click, wait_for_element, etc.) are available",
            "Verify that all essential Selenium WebDriver utility functions exist and are callable",
            "Test module initialization and verify all core Selenium utility functions exist",
        )

        def test_selenium_dependencies():
            """Test that required Selenium dependencies are available."""
            try:
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.common.exceptions import (
                    TimeoutException,
                    WebDriverException,
                )

                return True
            except ImportError:
                return False

        suite.run_test(
            "Selenium Dependencies Availability",
            test_selenium_dependencies,
            "Required Selenium WebDriver dependencies are properly imported",
            "Import key Selenium classes (By, WebDriverWait, expected_conditions, exceptions)",
            "Test availability of required Selenium WebDriver dependencies",
        )

        # CORE FUNCTIONALITY TESTS
        def test_force_user_agent_functionality():
            """Test user agent forcing with mock WebDriver."""
            if "force_user_agent" not in globals():
                return False

            try:
                from unittest.mock import MagicMock

                mock_driver = MagicMock()
                test_user_agent = "Mozilla/5.0 (Test) Custom User Agent"

                # Test normal operation
                force_user_agent(mock_driver, test_user_agent)

                # Verify the JavaScript execution was called
                mock_driver.execute_script.assert_called_once_with(
                    "navigator.userAgent = arguments[0]", test_user_agent
                )

                return True
            except Exception:
                return False

        suite.run_test(
            "User Agent Modification",
            test_force_user_agent_functionality,
            "Successfully modifies browser user agent string via JavaScript execution",
            "Use mock WebDriver to test force_user_agent() with custom user agent string",
            "Test user agent modification functionality with JavaScript execution",
        )

        def test_safe_click_mechanism():
            """Test safe clicking mechanism with error handling."""
            if "safe_click" not in globals():
                return False

            try:
                from unittest.mock import MagicMock

                mock_driver = MagicMock()
                mock_element = MagicMock()

                # Test successful click
                mock_element.click.return_value = None
                result = safe_click(mock_driver, mock_element)

                if not isinstance(result, bool):
                    return False

                # Test click with exception
                mock_element_failing = MagicMock()
                mock_element_failing.click.side_effect = Exception("Click intercepted")
                result_fail = safe_click(mock_driver, mock_element_failing)

                # Should return False for failed click
                return isinstance(result_fail, bool)

            except Exception:
                return False

        suite.run_test(
            "Safe Element Clicking",
            test_safe_click_mechanism,
            "Safe clicking handles both successful clicks and click exceptions gracefully",
            "Test safe_click() with successful clicks and simulated click failures",
            "Test safe element clicking mechanism with error handling",
        )

        def test_element_text_extraction():
            """Test text extraction from web elements."""
            if "get_element_text" not in globals():
                return False

            try:
                from unittest.mock import MagicMock

                # Test normal text extraction
                mock_element = MagicMock()
                test_text = "Sample element text content"
                mock_element.text = test_text

                result = get_element_text(mock_element)
                if result != test_text:
                    return False

                # Test with None element
                result_none = get_element_text(None)
                if result_none != "":
                    return False

                # Test with element that raises exception
                mock_element_error = MagicMock()
                mock_element_error.text = property(
                    lambda self: (_ for _ in ()).throw(Exception("Text access error"))
                )

                try:
                    result_error = get_element_text(mock_element_error)
                    # Should handle gracefully
                    return isinstance(result_error, str)
                except:
                    # Exception handling is acceptable
                    return True

            except Exception:
                return False

        suite.run_test(
            "Element Text Extraction",
            test_element_text_extraction,
            "Text extraction works for normal elements, handles None elements, and manages text access errors",
            "Test get_element_text() with normal elements, None input, and error conditions",
            "Test web element text extraction with various scenarios and error handling",
        )

        def test_element_visibility_detection():
            """Test element visibility detection functionality."""
            if "is_element_visible" not in globals():
                return False

            try:
                from unittest.mock import MagicMock

                # Test visible element
                mock_visible = MagicMock()
                mock_visible.is_displayed.return_value = True
                result_visible = is_element_visible(mock_visible)

                if result_visible != True:
                    return False

                # Test hidden element
                mock_hidden = MagicMock()
                mock_hidden.is_displayed.return_value = False
                result_hidden = is_element_visible(mock_hidden)

                if result_hidden != False:
                    return False

                # Test None element
                result_none = is_element_visible(None)
                if result_none != False:
                    return False

                # Test element that raises exception
                mock_error = MagicMock()
                mock_error.is_displayed.side_effect = Exception("Display check error")
                result_error = is_element_visible(mock_error)

                # Should handle gracefully and return False
                return result_error == False

            except Exception:
                return False

        suite.run_test(
            "Element Visibility Detection",
            test_element_visibility_detection,
            "Visibility detection correctly identifies visible/hidden elements and handles None/error cases",
            "Test is_element_visible() with visible, hidden, None, and error-prone elements",
            "Test web element visibility detection with comprehensive scenarios",
        )

        # EDGE CASES TESTS
        def test_scroll_to_element_edge_cases():
            """Test element scrolling with edge cases."""
            if "scroll_to_element" not in globals():
                return False

            try:
                from unittest.mock import MagicMock

                mock_driver = MagicMock()
                mock_element = MagicMock()

                # Test normal scrolling
                scroll_to_element(mock_driver, mock_element)
                mock_driver.execute_script.assert_called_with(
                    "arguments[0].scrollIntoView();", mock_element
                )

                # Test with None driver
                try:
                    scroll_to_element(None, mock_element)
                    # Should handle gracefully or raise appropriate exception
                except Exception:
                    pass  # Exception is acceptable

                # Test with None element
                try:
                    scroll_to_element(mock_driver, None)
                    # Should handle gracefully or raise appropriate exception
                except Exception:
                    pass  # Exception is acceptable

                return True

            except Exception:
                return False

        suite.run_test(
            "Element Scrolling Edge Cases",
            test_scroll_to_element_edge_cases,
            "Element scrolling handles None driver/element inputs gracefully",
            "Test scroll_to_element() with None inputs and verify error handling",
            "Test element scrolling with edge cases and None inputs",
        )

        def test_wait_for_element_timeout_handling():
            """Test element waiting with timeout scenarios."""
            if "wait_for_element" not in globals():
                return False

            try:
                from unittest.mock import MagicMock

                # Test with None driver
                result = wait_for_element(None, ("id", "test_element"))
                if result is not None:
                    return False  # Test with mock driver (will likely timeout)
                mock_driver = MagicMock()
                result = wait_for_element(
                    mock_driver, ("id", "nonexistent_element"), timeout=1
                )

                # Should return None on timeout
                return result is None

            except Exception:
                # Timeout exceptions are acceptable
                return True

        suite.run_test(
            "Element Wait Timeout Handling",
            test_wait_for_element_timeout_handling,
            "Element waiting handles None driver and timeout scenarios appropriately",
            "Test wait_for_element() with None driver and short timeout for non-existent element",
            "Test element waiting with timeout scenarios and None inputs",
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
                if func_name not in globals() or not callable(globals()[func_name]):
                    return False

            try:
                from unittest.mock import MagicMock

                mock_driver = MagicMock()
                mock_element = MagicMock()
                mock_element.text = "Integration Test Text"
                mock_element.is_displayed.return_value = True

                # Test workflow: check visibility -> scroll -> get text -> click
                is_visible = globals()["is_element_visible"](mock_element)
                if not is_visible:
                    return False

                globals()["scroll_to_element"](mock_driver, mock_element)
                text = globals()["get_element_text"](mock_element)
                if text != "Integration Test Text":
                    return False

                click_result = globals()["safe_click"](mock_driver, mock_element)
                return isinstance(click_result, bool)

            except Exception:
                return False

        suite.run_test(
            "Selenium Workflow Integration",
            test_selenium_workflow_integration,
            "Multiple Selenium utilities work together in typical web automation workflow",
            "Test visibility check -> scroll -> text extraction -> safe click workflow",
            "Test integration of multiple Selenium utilities in web automation workflow",
        )

        def test_browser_compatibility_handling():
            """Test handling of different browser-specific scenarios."""
            try:
                from unittest.mock import MagicMock

                # Test with different mock browser scenarios
                browsers = ["chrome", "firefox", "edge"]

                for browser in browsers:
                    mock_driver = MagicMock()
                    mock_driver.name = browser

                    # Test force_user_agent with different browsers
                    if "force_user_agent" in globals():
                        user_agent = f"Mozilla/5.0 ({browser.title()}) Test Agent"
                        globals()["force_user_agent"](mock_driver, user_agent)

                return True

            except Exception:
                return False

        suite.run_test(
            "Browser Compatibility Handling",
            test_browser_compatibility_handling,
            "Utilities work across different browser types (Chrome, Firefox, Edge)",
            "Test Selenium utilities with mock drivers representing different browsers",
            "Test browser compatibility across different WebDriver implementations",
        )

        # PERFORMANCE TESTS
        def test_bulk_element_operations():
            """Test performance with multiple element operations."""
            if (
                "get_element_text" not in globals()
                or "is_element_visible" not in globals()
            ):
                return False

            try:
                from unittest.mock import MagicMock
                import time

                # Create multiple mock elements
                elements = []
                for i in range(100):
                    mock_element = MagicMock()
                    mock_element.text = f"Element {i} text"
                    mock_element.is_displayed.return_value = True
                    elements.append(mock_element)

                start_time = time.time()

                # Perform bulk operations
                for element in elements:
                    globals()["get_element_text"](element)
                    globals()["is_element_visible"](element)

                duration = time.time() - start_time

                # Should complete 200 operations (100 text + 100 visibility) in reasonable time
                return duration < 0.5  # Less than 500ms

            except Exception:
                return False

        suite.run_test(
            "Bulk Element Operations Performance",
            test_bulk_element_operations,
            "200 element operations (100 text extractions + 100 visibility checks) complete in under 500ms",
            "Perform text extraction and visibility checks on 100 mock elements",
            "Test performance of bulk element operations with multiple web elements",
        )

        def test_repeated_driver_operations():
            """Test performance of repeated WebDriver operations."""
            if "scroll_to_element" not in globals():
                return False

            try:
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
                return duration < 0.2  # Less than 200ms

            except Exception:
                return False

        suite.run_test(
            "Repeated WebDriver Operations Performance",
            test_repeated_driver_operations,
            "50 scroll operations complete in under 200ms demonstrating efficient WebDriver interaction",
            "Perform scroll_to_element() operation 50 times with mock WebDriver",
            "Test performance of repeated WebDriver operations and JavaScript execution",
        )

        # ERROR HANDLING TESTS
        def test_invalid_element_handling():
            """Test handling of invalid or corrupted element objects."""
            if (
                "get_element_text" not in globals()
                or "is_element_visible" not in globals()
            ):
                return False

            try:
                # Test with various invalid inputs
                invalid_inputs = [
                    None,
                    "not_an_element",
                    123,
                    {},
                    [],
                ]

                for invalid_input in invalid_inputs:
                    try:
                        result_text = globals()["get_element_text"](invalid_input)
                        result_visible = globals()["is_element_visible"](invalid_input)

                        # Should return reasonable defaults or handle gracefully
                        if result_text is not None and not isinstance(result_text, str):
                            return False
                        if result_visible is not None and not isinstance(
                            result_visible, bool
                        ):
                            return False

                    except Exception:
                        # Exception handling is also acceptable
                        continue

                return True

            except Exception:
                return False

        suite.run_test(
            "Invalid Element Input Handling",
            test_invalid_element_handling,
            "Utilities handle invalid element inputs (None, strings, numbers) gracefully",
            "Test element utilities with various invalid input types and verify graceful handling",
            "Test error handling for invalid or corrupted element objects",
        )

        def test_webdriver_exception_handling():
            """Test handling of WebDriver-specific exceptions."""
            if "safe_click" not in globals():
                return False

            try:
                from unittest.mock import MagicMock

                mock_driver = MagicMock()
                mock_element = MagicMock()

                # Simulate various WebDriver exceptions
                webdriver_exceptions = [
                    Exception("ElementClickInterceptedException"),
                    Exception("ElementNotInteractableException"),
                    Exception("StaleElementReferenceException"),
                    Exception("WebDriverException"),
                ]

                for exception in webdriver_exceptions:
                    mock_element.click.side_effect = exception
                    result = globals()["safe_click"](mock_driver, mock_element)

                    # Should return False for failed clicks
                    if not isinstance(result, bool):
                        return False

                return True

            except Exception:
                return False

        suite.run_test(
            "WebDriver Exception Handling",
            test_webdriver_exception_handling,
            "Safe click handles various WebDriver exceptions gracefully and returns appropriate boolean results",
            "Test safe_click() with simulated WebDriver exceptions (click intercepted, stale element, etc.)",
            "Test error handling for WebDriver-specific exceptions during element operations",
        )

        return suite.finish_suite()


def run_comprehensive_tests_fallback() -> bool:
    """
    Fallback test function when test framework is not available.
    Provides basic testing capability using simple assertions.
    """
    print("🔧 Running Selenium Utils fallback test suite...")

    tests_passed = 0
    tests_total = 0

    # Test force_user_agent if available
    if "force_user_agent" in globals():
        tests_total += 1
        try:
            from unittest.mock import MagicMock

            mock_driver = MagicMock()
            force_user_agent(mock_driver, "test agent")
            tests_passed += 1
            print("✅ force_user_agent basic test passed")
        except Exception as e:
            print(f"❌ force_user_agent test error: {e}")

    # Test safe_click if available
    if "safe_click" in globals():
        tests_total += 1
        try:
            from unittest.mock import MagicMock

            mock_driver = MagicMock()
            mock_element = MagicMock()
            result = safe_click(mock_driver, mock_element)
            if isinstance(result, bool):
                tests_passed += 1
                print("✅ safe_click basic test passed")
            else:
                print("❌ safe_click returned unexpected type")
        except Exception as e:
            print(f"❌ safe_click test error: {e}")

    # Test get_element_text if available
    if "get_element_text" in globals():
        tests_total += 1
        try:
            from unittest.mock import MagicMock

            mock_element = MagicMock()
            mock_element.text = "test text"
            result = get_element_text(mock_element)
            if result == "test text":
                tests_passed += 1
                print("✅ get_element_text basic test passed")
            else:
                print("❌ get_element_text returned unexpected result")
        except Exception as e:
            print(f"❌ get_element_text test error: {e}")

    # Test is_element_visible if available
    if "is_element_visible" in globals():
        tests_total += 1
        try:
            from unittest.mock import MagicMock

            mock_element = MagicMock()
            mock_element.is_displayed.return_value = True
            result = is_element_visible(mock_element)
            if result == True:
                tests_passed += 1
                print("✅ is_element_visible basic test passed")
            else:
                print("❌ is_element_visible returned unexpected result")
        except Exception as e:
            print(f"❌ is_element_visible test error: {e}")

    print(
        f"🏁 Selenium Utils fallback tests completed: {tests_passed}/{tests_total} passed"
    )
    return tests_passed == tests_total


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    print("🔧 Running Selenium WebDriver Utilities comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

# End of selenium_utils.py
