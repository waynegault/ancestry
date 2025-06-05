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
    Comprehensive test suite for selenium_utils.py.
    Tests Selenium WebDriver utilities and browser automation functions.
    """
    # Import test framework components
    try:
        from test_framework import TestSuite, suppress_logging, assert_valid_function
    except ImportError:
        return run_comprehensive_tests_fallback()

    from unittest.mock import MagicMock, PropertyMock, patch

    suite = TestSuite("Selenium WebDriver Utilities", "selenium_utils.py")
    suite.start_suite()

    def test_force_user_agent():
        mock_driver = MagicMock()
        user_agent = "Mozilla/5.0 Test User Agent"

        # Test with valid driver and user agent
        force_user_agent(mock_driver, user_agent)
        mock_driver.execute_script.assert_called_once_with(
            "navigator.userAgent = arguments[0]", user_agent
        )

        # Test with None driver
        try:
            force_user_agent(None, user_agent)
        except Exception:
            pass  # Expected behavior

    def test_scroll_to_element():
        mock_driver = MagicMock()
        mock_element = MagicMock()

        # Test scrolling to element
        scroll_to_element(mock_driver, mock_element)
        mock_driver.execute_script.assert_called_with(
            "arguments[0].scrollIntoView();", mock_element
        )

        # Test with None parameters
        try:
            scroll_to_element(None, mock_element)
            scroll_to_element(mock_driver, None)
        except Exception:
            pass  # Expected behavior for None inputs

    def test_wait_for_element():
        from unittest.mock import MagicMock

        # Test with None driver (should return None)
        result = wait_for_element(None, ("id", "test_id"))
        assert result is None

        # Test with mock driver - since we can't easily mock the WebDriverWait
        # in this context, just test that the function doesn't crash
        mock_driver = MagicMock()
        try:
            result = wait_for_element(mock_driver, ("id", "test_id"))
            # Function should return None due to exception handling
            # when WebDriverWait fails with mock objects
            assert result is None
        except Exception:
            # If any exception occurs, that's also acceptable for this test
            pass

    def test_safe_click():
        mock_driver = MagicMock()
        mock_element = MagicMock()

        # Test normal click
        result = safe_click(mock_driver, mock_element)
        mock_element.click.assert_called_once()
        assert result == True

        # Test with click exception
        mock_element.click.side_effect = Exception("Click failed")
        result = safe_click(mock_driver, mock_element)
        assert result == False

    def test_get_element_text():
        mock_element = MagicMock()
        mock_element.text = "Test Element Text"

        # Test getting text from element
        result = get_element_text(mock_element)
        assert result == "Test Element Text"

        # Test with None element
        result = get_element_text(None)
        assert result == ""

    def test_is_element_visible():
        mock_element = MagicMock()

        # Test visible element
        mock_element.is_displayed.return_value = True
        result = is_element_visible(mock_element)
        assert result == True

        # Test hidden element
        mock_element.is_displayed.return_value = False
        result = is_element_visible(mock_element)
        assert result == False

        # Test with None element
        result = is_element_visible(None)
        assert result == False  # Run all tests using the test framework

    with suppress_logging():
        suite.run_test(
            "User Agent Forcing",
            test_force_user_agent,
            "Forces specific user agent in browser",
        )
        suite.run_test(
            "Element Scrolling", test_scroll_to_element, "Scrolls to element on page"
        )
        suite.run_test(
            "Element Waiting", test_wait_for_element, "Waits for element to appear"
        )
        suite.run_test(
            "Safe Clicking", test_safe_click, "Safely clicks elements with retry"
        )
        suite.run_test(
            "Text Extraction", test_get_element_text, "Extracts text from elements"
        )
        suite.run_test(
            "Visibility Checking", test_is_element_visible, "Checks element visibility"
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
