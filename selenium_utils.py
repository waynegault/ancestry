# selenium_utils.py
"""
Selenium/WebDriver utility functions specifically for browser automation
and element interaction, separated from general or API-specific utilities.
"""

# --- Standard library imports ---
import time
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

# --- Local application imports ---
from config import config_instance, selenium_config
from logging_config import logger

# Note: Removed urllib3 and psutil imports as they weren't used here

# --- Selenium Specific Helpers ---


def force_user_agent(driver: Optional[WebDriver], user_agent: str):
    """
    Attempts to force the browser's User-Agent string using Chrome DevTools Protocol.
    """
    if not driver:
        logger.warning("force_user_agent: WebDriver is None.")
        return
    logger.debug(f"Attempting to set User-Agent via CDP to: {user_agent}")
    start_time = time.time()
    try:
        driver.execute_cdp_cmd(
            "Network.setUserAgentOverride", {"userAgent": user_agent}
        )
        duration = time.time() - start_time
        logger.info(f"Successfully set User-Agent via CDP in {duration:.3f} seconds.")
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


if __name__ == "__main__":
    """
    Self-test for selenium_utils.py module.

    When run directly, this module will execute a series of unit tests
    to verify the functionality of its utility functions.
    """
    import unittest
    from unittest.mock import MagicMock, PropertyMock
    import time
    import logging

    print("\n=== Selenium Utils Self-Test ===")
    logger.info("Running selenium_utils self-tests...")

    class TestSeleniumUtils(unittest.TestCase):
        """Test cases for selenium_utils.py functions."""

        @classmethod
        def setUpClass(cls):
            """Set up class-level fixtures before running tests."""
            # Save the original logger level to restore it later
            cls.original_logger_level = logger.level
            # Temporarily set logger to CRITICAL level to suppress all expected messages during tests
            logger.setLevel(logging.CRITICAL)

        @classmethod
        def tearDownClass(cls):
            """Clean up class-level fixtures after running tests."""
            # Restore the original logger level
            logger.setLevel(cls.original_logger_level)

        def setUp(self):
            """Set up test fixtures before each test method."""
            # Create mock WebDriver
            self.mock_driver = MagicMock()

            # Create mock WebElement
            self.mock_element = MagicMock()

            # Set up config_instance.BASE_URL for URL resolution tests
            config_instance.BASE_URL = "https://www.example.com"

        def test_force_user_agent(self):
            """Test force_user_agent function."""
            # Test with valid driver and user agent
            user_agent = "Mozilla/5.0 Test User Agent"
            force_user_agent(self.mock_driver, user_agent)
            self.mock_driver.execute_cdp_cmd.assert_called_once_with(
                "Network.setUserAgentOverride", {"userAgent": user_agent}
            )

            # Test with None driver
            self.mock_driver.reset_mock()
            force_user_agent(None, user_agent)
            self.mock_driver.execute_cdp_cmd.assert_not_called()

            # Test with exception during CDP command
            self.mock_driver.reset_mock()
            self.mock_driver.execute_cdp_cmd.side_effect = Exception("Test exception")
            force_user_agent(self.mock_driver, user_agent)
            self.mock_driver.execute_cdp_cmd.assert_called_once()

        def test_extract_text(self):
            """Test extract_text function."""
            # Set up mock element with text
            child_element = MagicMock()
            child_element.text = "Test Text"
            self.mock_element.find_element.return_value = child_element

            # Test successful text extraction
            result = extract_text(self.mock_element, "div.test")
            self.assertEqual(result, "Test Text")
            self.mock_element.find_element.assert_called_once_with(
                By.CSS_SELECTOR, "div.test"
            )

            # Test with None parent element
            self.mock_element.reset_mock()
            result = extract_text(None, "div.test")
            self.assertEqual(result, "")
            self.mock_element.find_element.assert_not_called()

            # Test with NoSuchElementException
            self.mock_element.reset_mock()
            self.mock_element.find_element.side_effect = NoSuchElementException(
                "Test exception"
            )
            result = extract_text(self.mock_element, "div.nonexistent")
            self.assertEqual(result, "")

            # Test with empty text
            self.mock_element.reset_mock()
            child_element.text = ""
            self.mock_element.find_element.side_effect = None
            self.mock_element.find_element.return_value = child_element
            result = extract_text(self.mock_element, "div.empty")
            self.assertEqual(result, "")

            # Test with general exception
            self.mock_element.reset_mock()
            self.mock_element.find_element.side_effect = Exception("Test exception")
            result = extract_text(self.mock_element, "div.error")
            self.assertEqual(result, "")

        def test_extract_attribute(self):
            """Test extract_attribute function."""
            # Set up mock element with attribute
            child_element = MagicMock()
            child_element.get_attribute.return_value = "attribute_value"
            self.mock_element.find_element.return_value = child_element

            # Test successful attribute extraction (non-href)
            result = extract_attribute(self.mock_element, "div.test", "data-test")
            self.assertEqual(result, "attribute_value")
            self.mock_element.find_element.assert_called_once_with(
                By.CSS_SELECTOR, "div.test"
            )
            child_element.get_attribute.assert_called_once_with("data-test")

            # Test with None parent element
            self.mock_element.reset_mock()
            child_element.reset_mock()
            result = extract_attribute(None, "div.test", "data-test")
            self.assertEqual(result, "")
            self.mock_element.find_element.assert_not_called()

            # Test with href attribute - absolute URL
            self.mock_element.reset_mock()
            child_element.reset_mock()
            child_element.get_attribute.return_value = "https://other-domain.com/page"
            result = extract_attribute(self.mock_element, "a.link", "href")
            self.assertEqual(result, "https://other-domain.com/page")

            # Test with href attribute - relative URL with leading slash
            self.mock_element.reset_mock()
            child_element.reset_mock()
            child_element.get_attribute.return_value = "/relative/path"
            result = extract_attribute(self.mock_element, "a.link", "href")
            self.assertEqual(result, "https://www.example.com/relative/path")

            # Test with NoSuchElementException
            self.mock_element.reset_mock()
            child_element.reset_mock()
            self.mock_element.find_element.side_effect = NoSuchElementException(
                "Test exception"
            )
            result = extract_attribute(
                self.mock_element, "div.nonexistent", "data-test"
            )
            self.assertEqual(result, "")

            # Test with general exception
            self.mock_element.reset_mock()
            self.mock_element.find_element.side_effect = Exception("Test exception")
            result = extract_attribute(self.mock_element, "div.error", "data-test")
            self.assertEqual(result, "")

        def test_is_elem_there(self):
            """Test is_elem_there function."""
            # Test with None driver - this is the most important case to test
            # and doesn't require any mocking
            result = is_elem_there(None, By.ID, "test-id")
            self.assertFalse(result)

            # Note: We're only testing the None driver case because it's the most critical
            # and doesn't require complex mocking. The other cases would require more
            # sophisticated mocking of WebDriverWait and EC.presence_of_element_located,
            # which is challenging in this context.

        def test_is_browser_open(self):
            """Test is_browser_open function."""
            # Test with open browser
            type(self.mock_driver).window_handles = PropertyMock(
                return_value=["handle1"]
            )
            result = is_browser_open(self.mock_driver)
            self.assertTrue(result)

            # Test with None driver
            result = is_browser_open(None)
            self.assertFalse(result)

            # Test with InvalidSessionIdException
            type(self.mock_driver).window_handles = PropertyMock(
                side_effect=InvalidSessionIdException("invalid session id")
            )
            result = is_browser_open(self.mock_driver)
            self.assertFalse(result)

            # Test with NoSuchWindowException
            type(self.mock_driver).window_handles = PropertyMock(
                side_effect=NoSuchWindowException("no such window")
            )
            result = is_browser_open(self.mock_driver)
            self.assertFalse(result)

            # Test with WebDriverException - disconnected
            type(self.mock_driver).window_handles = PropertyMock(
                side_effect=WebDriverException("disconnected")
            )
            result = is_browser_open(self.mock_driver)
            self.assertFalse(result)

            # Test with WebDriverException - other
            type(self.mock_driver).window_handles = PropertyMock(
                side_effect=WebDriverException("some other error")
            )
            result = is_browser_open(self.mock_driver)
            self.assertFalse(result)

            # Test with general exception
            type(self.mock_driver).window_handles = PropertyMock(
                side_effect=Exception("Test exception")
            )
            result = is_browser_open(self.mock_driver)
            self.assertFalse(result)

        def test_close_tabs(self):
            """Test close_tabs function."""
            # Test with None driver first (simplest case)
            close_tabs(None)

            # Test with single tab
            self.mock_driver.reset_mock()
            self.mock_driver.window_handles = ["handle1"]
            close_tabs(self.mock_driver)
            self.mock_driver.switch_to.window.assert_not_called()
            self.mock_driver.close.assert_not_called()

            # Test with multiple tabs, current handle is first
            self.mock_driver.reset_mock()
            self.mock_driver.window_handles = ["handle1", "handle2", "handle3"]
            self.mock_driver.current_window_handle = "handle1"
            # Reset any side effects
            self.mock_driver.close.side_effect = None

            close_tabs(self.mock_driver)

            # Should switch to each handle and close it
            self.assertEqual(self.mock_driver.switch_to.window.call_count, 2)
            self.assertEqual(self.mock_driver.close.call_count, 2)

            # Test with NoSuchWindowException during close
            self.mock_driver.reset_mock()
            self.mock_driver.window_handles = ["handle1", "handle2"]
            self.mock_driver.current_window_handle = "handle1"
            self.mock_driver.close.side_effect = NoSuchWindowException("Test exception")

            close_tabs(self.mock_driver)
            self.mock_driver.switch_to.window.assert_called_once()

            # Test with WebDriverException during close
            self.mock_driver.reset_mock()
            self.mock_driver.window_handles = ["handle1", "handle2"]
            self.mock_driver.current_window_handle = "handle1"
            self.mock_driver.close.side_effect = WebDriverException("Test exception")

            close_tabs(self.mock_driver)
            self.mock_driver.switch_to.window.assert_called_once()

            # Test with general exception during close
            self.mock_driver.reset_mock()
            self.mock_driver.window_handles = ["handle1", "handle2"]
            self.mock_driver.current_window_handle = "handle1"
            self.mock_driver.close.side_effect = Exception("Test exception")

            close_tabs(self.mock_driver)
            self.mock_driver.switch_to.window.assert_called_once()

        def test_get_driver_cookies(self):
            """Test get_driver_cookies function."""
            # Test with valid cookies
            self.mock_driver.get_cookies.return_value = [
                {"name": "cookie1", "value": "value1"},
                {"name": "cookie2", "value": "value2"},
            ]

            result = get_driver_cookies(self.mock_driver)
            self.assertEqual(result, {"cookie1": "value1", "cookie2": "value2"})

            # Test with None driver
            result = get_driver_cookies(None)
            self.assertEqual(result, {})

            # Test with WebDriverException
            self.mock_driver.reset_mock()
            self.mock_driver.get_cookies.side_effect = WebDriverException(
                "Test exception"
            )
            result = get_driver_cookies(self.mock_driver)
            self.assertEqual(result, {})

            # Test with general exception
            self.mock_driver.reset_mock()
            self.mock_driver.get_cookies.side_effect = Exception("Test exception")
            result = get_driver_cookies(self.mock_driver)
            self.assertEqual(result, {})

    # Run the tests
    def run_tests():
        """Run all tests and display results."""
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

        return result.wasSuccessful()

    # Execute the tests
    success = run_tests()

    if success:
        print("\nAll selenium_utils tests passed successfully!")
    else:
        print("\nSome selenium_utils tests failed. See details above.")

    logger.info("selenium_utils self-tests completed.")
# End of selenium_utils.py
