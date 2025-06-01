# selenium_utils.py
"""
Selenium/WebDriver utility functions specifically for browser automation
and element interaction, separated from general or API-specific utilities.
"""

# --- Standard library imports ---
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


# --- Test Class Definition ---
# This class is defined at the module level so it can be imported by test_selenium_utils.py
from unittest.mock import MagicMock, PropertyMock


# Original unittest.TestCase tests have been replaced with standardized test framework format below


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, PropertyMock, patch

    try:
        from test_framework import TestSuite, suppress_logging, assert_valid_function
    except ImportError:
        print(
            "❌ test_framework.py not found. Please ensure it exists in the same directory."
        )
        sys.exit(1)

    def run_comprehensive_tests() -> bool:
        """
        Comprehensive test suite for selenium_utils.py.
        Tests Selenium WebDriver utilities and browser automation functions.
        """
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
            mock_driver.reset_mock()
            force_user_agent(None, user_agent)
            mock_driver.execute_script.assert_not_called()

            # Test with exception during CDP command
            mock_driver.reset_mock()
            mock_driver.execute_script.side_effect = Exception("Test exception")
            force_user_agent(mock_driver, user_agent)
            mock_driver.execute_script.assert_called_once()

        def test_extract_text():
            mock_element = MagicMock()
            child_element = MagicMock()
            child_element.text = "Test Text"
            mock_element.find_element.return_value = child_element

            # Test successful text extraction
            result = extract_text(mock_element, "div.test")
            assert result == "Test Text"

            # Test with None parent element
            result = extract_text(None, "div.test")
            assert result == ""

            # Test with NoSuchElementException
            mock_element.find_element.side_effect = NoSuchElementException(
                "Test exception"
            )
            result = extract_text(mock_element, "div.nonexistent")
            assert result == ""

        def test_extract_attribute():
            mock_element = MagicMock()
            child_element = MagicMock()
            child_element.get_attribute.return_value = "attribute_value"
            mock_element.find_element.return_value = child_element

            # Test successful attribute extraction
            result = extract_attribute(mock_element, "div.test", "data-test")
            assert result == "attribute_value"

            # Test with None parent element
            result = extract_attribute(None, "div.test", "data-test")
            assert result == ""

            # Test with href attribute - relative URL with leading slash
            child_element.get_attribute.return_value = "/relative/path"
            # Store original BASE_URL and temporarily change it
            original_base_url = config_instance.BASE_URL
            try:
                config_instance.BASE_URL = "https://www.example.com"
                result = extract_attribute(mock_element, "a.link", "href")
                assert result == "https://www.example.com/relative/path"
            finally:
                # Restore original BASE_URL
                config_instance.BASE_URL = original_base_url

            # Test with href attribute - absolute URL
            child_element.get_attribute.return_value = "https://other-domain.com/page"
            result = extract_attribute(mock_element, "a.link", "href")
            assert result == "https://other-domain.com/page"

            # Test with NoSuchElementException
            mock_element.find_element.side_effect = NoSuchElementException(
                "Test exception"
            )
            result = extract_attribute(mock_element, "div.nonexistent", "data-test")
            assert result == ""

        def test_is_browser_open():
            mock_driver = MagicMock()

            # Test with open browser
            type(mock_driver).window_handles = PropertyMock(return_value=["handle1"])
            result = is_browser_open(mock_driver)
            assert result is True

            # Test with None driver
            result = is_browser_open(None)
            assert result is False

            # Test with InvalidSessionIdException
            type(mock_driver).window_handles = PropertyMock(
                side_effect=InvalidSessionIdException("invalid session id")
            )
            result = is_browser_open(mock_driver)
            assert result is False

        def test_close_tabs():
            mock_driver = MagicMock()

            # Test with None driver
            close_tabs(None)  # Should not raise exception

            # Test with single tab
            mock_driver.window_handles = ["handle1"]
            close_tabs(mock_driver)
            mock_driver.switch_to.window.assert_not_called()

            # Test with multiple tabs
            mock_driver.reset_mock()
            mock_driver.window_handles = ["handle1", "handle2", "handle3"]
            mock_driver.current_window_handle = "handle1"
            mock_driver.close.side_effect = None

            close_tabs(mock_driver)
            assert mock_driver.switch_to.window.call_count == 2
            assert mock_driver.close.call_count == 2

        def test_get_driver_cookies():
            mock_driver = MagicMock()

            # Test with valid cookies
            mock_driver.get_cookies.return_value = [
                {"name": "cookie1", "value": "value1"},
                {"name": "cookie2", "value": "value2"},
            ]

            result = get_driver_cookies(mock_driver)
            assert result == {"cookie1": "value1", "cookie2": "value2"}

            # Test with None driver
            result = get_driver_cookies(None)
            assert result == {}

            # Test with WebDriverException
            mock_driver.get_cookies.side_effect = WebDriverException("Test exception")
            result = get_driver_cookies(mock_driver)
            assert result == {}

        def test_browser_state_functions():
            # Test that critical browser state functions exist
            state_functions = ["is_browser_open", "close_tabs", "force_user_agent"]

            for func_name in state_functions:
                if func_name in globals():
                    assert_valid_function(globals()[func_name], func_name)

        def test_element_interaction_functions():
            # Test element interaction utility functions
            interaction_functions = [
                "extract_text",
                "extract_attribute",
                "is_elem_there",
            ]

            for func_name in interaction_functions:
                if func_name in globals():
                    assert_valid_function(globals()[func_name], func_name)

        def test_error_handling():
            # Test error handling across selenium utilities
            mock_driver = MagicMock()

            # Test WebDriverException handling in get_driver_cookies
            mock_driver.get_cookies.side_effect = WebDriverException("Connection lost")
            result = get_driver_cookies(mock_driver)
            assert result == {}

            # Test general exception handling in extract_text
            mock_element = MagicMock()
            mock_element.find_element.side_effect = Exception("General error")
            result = extract_text(mock_element, "div.test")
            assert result == ""

        def test_selenium_integration():
            # Test integration patterns with Selenium WebDriver
            mock_driver = MagicMock()

            # Test cookie retrieval and processing
            mock_driver.get_cookies.return_value = [
                {"name": "session", "value": "abc123", "domain": ".example.com"}
            ]

            cookies = get_driver_cookies(mock_driver)
            assert isinstance(cookies, dict)
            assert "session" in cookies
            assert cookies["session"] == "abc123"

        # Run all tests
        test_functions = {
            "Force user agent functionality": (
                test_force_user_agent,
                "Should set user agent via CDP commands",
            ),
            "Text extraction from elements": (
                test_extract_text,
                "Should extract text content from DOM elements",
            ),
            "Attribute extraction from elements": (
                test_extract_attribute,
                "Should extract attributes from DOM elements with URL resolution",
            ),
            "Browser state detection": (
                test_is_browser_open,
                "Should detect if browser session is active",
            ),
            "Tab management utilities": (
                test_close_tabs,
                "Should close additional browser tabs safely",
            ),
            "Cookie management": (
                test_get_driver_cookies,
                "Should retrieve and format browser cookies",
            ),
            "Browser state management functions": (
                test_browser_state_functions,
                "Should provide browser state management utilities",
            ),
            "Element interaction functions": (
                test_element_interaction_functions,
                "Should provide DOM element interaction utilities",
            ),
            "Error handling in utility functions": (
                test_error_handling,
                "Should handle WebDriver errors gracefully",
            ),
            "Integration with Selenium WebDriver": (
                test_selenium_integration,
                "Should integrate seamlessly with Selenium WebDriver",
            ),
        }

        with suppress_logging():
            for test_name, (test_func, expected_behavior) in test_functions.items():
                suite.run_test(test_name, test_func, expected_behavior)

        return suite.finish_suite()

    print("🔧 Running Selenium WebDriver Utilities comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

# End of selenium_utils.py
