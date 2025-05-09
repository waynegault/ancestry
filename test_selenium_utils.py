#!/usr/bin/env python3

# test_selenium_utils.py
"""
Unit tests for selenium_utils.py module.

These tests verify the functionality of Selenium utility functions
using mocks to simulate browser behavior without requiring an actual browser.
"""

# --- Standard library imports ---
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# --- Third-party imports ---
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
    WebDriverException,
    InvalidSessionIdException,
    NoSuchWindowException,
)

# --- Local application imports ---
import selenium_utils
from config import config_instance


class TestSeleniumUtils(unittest.TestCase):
    """Test cases for selenium_utils.py functions."""

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
        selenium_utils.force_user_agent(self.mock_driver, user_agent)
        self.mock_driver.execute_cdp_cmd.assert_called_once_with(
            "Network.setUserAgentOverride", {"userAgent": user_agent}
        )

        # Test with None driver
        self.mock_driver.reset_mock()
        selenium_utils.force_user_agent(None, user_agent)
        self.mock_driver.execute_cdp_cmd.assert_not_called()

        # Test with exception during CDP command
        self.mock_driver.reset_mock()
        self.mock_driver.execute_cdp_cmd.side_effect = Exception("Test exception")
        selenium_utils.force_user_agent(self.mock_driver, user_agent)
        self.mock_driver.execute_cdp_cmd.assert_called_once()

    def test_extract_text(self):
        """Test extract_text function."""
        # Set up mock element with text
        child_element = MagicMock()
        child_element.text = "Test Text"
        self.mock_element.find_element.return_value = child_element

        # Test successful text extraction
        result = selenium_utils.extract_text(self.mock_element, "div.test")
        self.assertEqual(result, "Test Text")
        self.mock_element.find_element.assert_called_once_with(
            By.CSS_SELECTOR, "div.test"
        )

        # Test with None parent element
        self.mock_element.reset_mock()
        result = selenium_utils.extract_text(None, "div.test")
        self.assertEqual(result, "")
        self.mock_element.find_element.assert_not_called()

        # Test with NoSuchElementException
        self.mock_element.reset_mock()
        self.mock_element.find_element.side_effect = NoSuchElementException(
            "Test exception"
        )
        result = selenium_utils.extract_text(self.mock_element, "div.nonexistent")
        self.assertEqual(result, "")

        # Test with empty text
        self.mock_element.reset_mock()
        child_element.text = ""
        self.mock_element.find_element.side_effect = None
        self.mock_element.find_element.return_value = child_element
        result = selenium_utils.extract_text(self.mock_element, "div.empty")
        self.assertEqual(result, "")

        # Test with general exception
        self.mock_element.reset_mock()
        self.mock_element.find_element.side_effect = Exception("Test exception")
        result = selenium_utils.extract_text(self.mock_element, "div.error")
        self.assertEqual(result, "")

    def test_extract_attribute(self):
        """Test extract_attribute function."""
        # Set up mock element with attribute
        child_element = MagicMock()
        child_element.get_attribute.return_value = "attribute_value"
        self.mock_element.find_element.return_value = child_element

        # Test successful attribute extraction (non-href)
        result = selenium_utils.extract_attribute(
            self.mock_element, "div.test", "data-test"
        )
        self.assertEqual(result, "attribute_value")
        self.mock_element.find_element.assert_called_once_with(
            By.CSS_SELECTOR, "div.test"
        )
        child_element.get_attribute.assert_called_once_with("data-test")

        # Test with None parent element
        self.mock_element.reset_mock()
        child_element.reset_mock()
        result = selenium_utils.extract_attribute(None, "div.test", "data-test")
        self.assertEqual(result, "")
        self.mock_element.find_element.assert_not_called()

        # Test with href attribute - absolute URL
        self.mock_element.reset_mock()
        child_element.reset_mock()
        child_element.get_attribute.return_value = "https://other-domain.com/page"
        result = selenium_utils.extract_attribute(self.mock_element, "a.link", "href")
        self.assertEqual(result, "https://other-domain.com/page")

        # Test with href attribute - relative URL with leading slash
        self.mock_element.reset_mock()
        child_element.reset_mock()
        child_element.get_attribute.return_value = "/relative/path"
        result = selenium_utils.extract_attribute(self.mock_element, "a.link", "href")
        self.assertEqual(result, "https://www.example.com/relative/path")

        # Test with href attribute - relative URL without leading slash
        self.mock_element.reset_mock()
        child_element.reset_mock()
        child_element.get_attribute.return_value = "relative/path"
        result = selenium_utils.extract_attribute(self.mock_element, "a.link", "href")
        self.assertEqual(result, "https://www.example.com/relative/path")

        # Test with NoSuchElementException
        self.mock_element.reset_mock()
        child_element.reset_mock()
        self.mock_element.find_element.side_effect = NoSuchElementException(
            "Test exception"
        )
        result = selenium_utils.extract_attribute(
            self.mock_element, "div.nonexistent", "data-test"
        )
        self.assertEqual(result, "")

        # Test with general exception
        self.mock_element.reset_mock()
        self.mock_element.find_element.side_effect = Exception("Test exception")
        result = selenium_utils.extract_attribute(
            self.mock_element, "div.error", "data-test"
        )
        self.assertEqual(result, "")

    def test_is_elem_there(self):
        """Test is_elem_there function."""
        # Test with None driver - this is the most important case to test
        # and doesn't require any mocking
        result = selenium_utils.is_elem_there(None, By.ID, "test-id")
        self.assertFalse(result)

    def test_is_browser_open(self):
        """Test is_browser_open function."""
        # Test with open browser
        type(self.mock_driver).window_handles = PropertyMock(return_value=["handle1"])
        result = selenium_utils.is_browser_open(self.mock_driver)
        self.assertTrue(result)

        # Test with None driver
        result = selenium_utils.is_browser_open(None)
        self.assertFalse(result)

        # Test with InvalidSessionIdException
        type(self.mock_driver).window_handles = PropertyMock(
            side_effect=InvalidSessionIdException("invalid session id")
        )
        result = selenium_utils.is_browser_open(self.mock_driver)
        self.assertFalse(result)

        # Test with NoSuchWindowException
        type(self.mock_driver).window_handles = PropertyMock(
            side_effect=NoSuchWindowException("no such window")
        )
        result = selenium_utils.is_browser_open(self.mock_driver)
        self.assertFalse(result)

        # Test with WebDriverException - disconnected
        type(self.mock_driver).window_handles = PropertyMock(
            side_effect=WebDriverException("disconnected")
        )
        result = selenium_utils.is_browser_open(self.mock_driver)
        self.assertFalse(result)

        # Test with WebDriverException - other
        type(self.mock_driver).window_handles = PropertyMock(
            side_effect=WebDriverException("some other error")
        )
        result = selenium_utils.is_browser_open(self.mock_driver)
        self.assertFalse(result)

        # Test with general exception
        type(self.mock_driver).window_handles = PropertyMock(
            side_effect=Exception("Test exception")
        )
        result = selenium_utils.is_browser_open(self.mock_driver)
        self.assertFalse(result)

    def test_close_tabs(self):
        """Test close_tabs function."""
        # Test with multiple tabs, current handle is first
        self.mock_driver.window_handles = ["handle1", "handle2", "handle3"]
        self.mock_driver.current_window_handle = "handle1"

        selenium_utils.close_tabs(self.mock_driver)

        # Should switch to each handle and close it
        self.assertEqual(self.mock_driver.switch_to.window.call_count, 2)
        self.assertEqual(self.mock_driver.close.call_count, 2)

        # Test with None driver
        self.mock_driver.reset_mock()
        selenium_utils.close_tabs(None)
        self.mock_driver.switch_to.window.assert_not_called()

        # Test with single tab
        self.mock_driver.reset_mock()
        self.mock_driver.window_handles = ["handle1"]
        selenium_utils.close_tabs(self.mock_driver)
        self.mock_driver.switch_to.window.assert_not_called()
        self.mock_driver.close.assert_not_called()

        # Test with NoSuchWindowException during close
        self.mock_driver.reset_mock()
        self.mock_driver.window_handles = ["handle1", "handle2"]
        self.mock_driver.current_window_handle = "handle1"
        self.mock_driver.close.side_effect = NoSuchWindowException("Test exception")

        selenium_utils.close_tabs(self.mock_driver)
        self.mock_driver.switch_to.window.assert_called_once()

        # Test with WebDriverException during close
        self.mock_driver.reset_mock()
        self.mock_driver.window_handles = ["handle1", "handle2"]
        self.mock_driver.current_window_handle = "handle1"
        self.mock_driver.close.side_effect = WebDriverException("Test exception")

        selenium_utils.close_tabs(self.mock_driver)
        self.mock_driver.switch_to.window.assert_called_once()

    def test_get_driver_cookies(self):
        """Test get_driver_cookies function."""
        # Test with valid cookies
        self.mock_driver.get_cookies.return_value = [
            {"name": "cookie1", "value": "value1"},
            {"name": "cookie2", "value": "value2"},
        ]

        result = selenium_utils.get_driver_cookies(self.mock_driver)
        self.assertEqual(result, {"cookie1": "value1", "cookie2": "value2"})

        # Test with None driver
        result = selenium_utils.get_driver_cookies(None)
        self.assertEqual(result, {})

        # Test with WebDriverException
        self.mock_driver.reset_mock()
        self.mock_driver.get_cookies.side_effect = WebDriverException("Test exception")
        result = selenium_utils.get_driver_cookies(self.mock_driver)
        self.assertEqual(result, {})

        # Test with general exception
        self.mock_driver.reset_mock()
        self.mock_driver.get_cookies.side_effect = Exception("Test exception")
        result = selenium_utils.get_driver_cookies(self.mock_driver)
        self.assertEqual(result, {})

    @patch("builtins.open", create=True)
    @patch("json.dump")
    @patch("os.makedirs")
    def test_export_cookies(self, mock_makedirs, mock_json_dump, mock_open):
        """Test export_cookies function."""
        # Test with valid cookies
        self.mock_driver.get_cookies.return_value = [
            {"name": "cookie1", "value": "value1"},
            {"name": "cookie2", "value": "value2"},
        ]

        result = selenium_utils.export_cookies(
            self.mock_driver, "test/path/cookies.json"
        )
        self.assertTrue(result)
        mock_makedirs.assert_called_once()
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()

        # Test with None driver
        mock_makedirs.reset_mock()
        mock_open.reset_mock()
        mock_json_dump.reset_mock()
        result = selenium_utils.export_cookies(None, "test/path/cookies.json")
        self.assertFalse(result)
        mock_makedirs.assert_not_called()
        mock_open.assert_not_called()
        mock_json_dump.assert_not_called()

        # Test with empty cookies list
        self.mock_driver.reset_mock()
        mock_makedirs.reset_mock()
        mock_open.reset_mock()
        mock_json_dump.reset_mock()
        self.mock_driver.get_cookies.return_value = []
        result = selenium_utils.export_cookies(
            self.mock_driver, "test/path/cookies.json"
        )
        self.assertFalse(result)
        mock_makedirs.assert_not_called()
        mock_open.assert_not_called()
        mock_json_dump.assert_not_called()

        # Test with WebDriverException
        self.mock_driver.reset_mock()
        mock_makedirs.reset_mock()
        mock_open.reset_mock()
        mock_json_dump.reset_mock()
        self.mock_driver.get_cookies.side_effect = WebDriverException("Test exception")
        result = selenium_utils.export_cookies(
            self.mock_driver, "test/path/cookies.json"
        )
        self.assertFalse(result)
        mock_makedirs.assert_not_called()
        mock_open.assert_not_called()
        mock_json_dump.assert_not_called()

        # Test with file I/O error
        self.mock_driver.reset_mock()
        mock_makedirs.reset_mock()
        mock_open.reset_mock()
        mock_json_dump.reset_mock()
        self.mock_driver.get_cookies.side_effect = None
        self.mock_driver.get_cookies.return_value = [
            {"name": "cookie1", "value": "value1"}
        ]
        mock_open.side_effect = OSError("Test I/O exception")
        result = selenium_utils.export_cookies(
            self.mock_driver, "test/path/cookies.json"
        )
        self.assertFalse(result)

        # Test with general exception
        self.mock_driver.reset_mock()
        mock_makedirs.reset_mock()
        mock_open.reset_mock()
        mock_json_dump.reset_mock()
        mock_open.side_effect = None
        mock_json_dump.side_effect = Exception("Test exception")
        result = selenium_utils.export_cookies(
            self.mock_driver, "test/path/cookies.json"
        )
        self.assertFalse(result)


if __name__ == "__main__":
    print("=== Running Selenium Utils Tests ===")
    unittest.main()
# End of test_selenium_utils.py
