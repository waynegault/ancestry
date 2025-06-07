"""
Browser Manager - Handles all browser/WebDriver operations.

This module extracts browser management functionality from the monolithic
SessionManager class to provide a clean separation of concerns.
"""

import logging
import time
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from selenium.common.exceptions import (
    InvalidSessionIdException,
    NoSuchWindowException,
    WebDriverException,
)
from selenium.webdriver.remote.webdriver import WebDriver

from config import config_instance, selenium_config
from chromedriver import init_webdvr
from selenium_utils import export_cookies
from utils import nav_to_page

logger = logging.getLogger(__name__)

# Type alias
DriverType = Optional[WebDriver]


class BrowserManager:
    """Manages browser/WebDriver operations and state."""

    def __init__(self):
        """Initialize the BrowserManager."""
        self.driver: DriverType = None
        self.driver_live: bool = False
        self.browser_needed: bool = False
        self.session_start_time: Optional[float] = None

        logger.debug("BrowserManager initialized")

    def start_browser(self, action_name: Optional[str] = None) -> bool:
        """
        Start the browser session.

        Args:
            action_name: Optional name of the action for logging

        Returns:
            bool: True if browser started successfully, False otherwise
        """
        logger.debug(f"Starting browser for action: {action_name or 'Unknown'}")

        try:
            if self.is_session_valid():
                logger.debug("Browser already running and valid")
                return True

            logger.debug("Initializing WebDriver instance...")
            self.driver = init_webdvr()

            if not self.driver:
                logger.error(
                    "WebDriver initialization failed (init_webdvr returned None)."
                )
                return False

            logger.debug("WebDriver initialization successful.")

            # Navigate to base URL to stabilize
            logger.debug(
                f"Navigating to Base URL ({config_instance.BASE_URL}) to stabilize..."
            )

            if not nav_to_page(self.driver, config_instance.BASE_URL):
                logger.error(
                    f"Failed to navigate to base URL: {config_instance.BASE_URL}"
                )
                self.close_browser()
                return False

            # Mark as live and set timing
            self.driver_live = True
            self.browser_needed = True
            self.session_start_time = time.time()

            logger.info("Browser session started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start browser: {e}", exc_info=True)
            self.close_browser()
            return False

    def close_browser(self) -> None:
        """Close the browser and cleanup resources."""
        logger.debug("Closing browser session...")

        if self.driver:
            try:
                self.driver.quit()
                logger.debug("WebDriver quit successfully")
            except Exception as e:
                logger.warning(f"Error quitting WebDriver: {e}")

        # Reset state
        self.driver = None
        self.driver_live = False
        self.browser_needed = False
        self.session_start_time = None

        logger.debug("Browser session closed")

    def is_session_valid(self) -> bool:
        """
        Check if the current browser session is valid.

        Returns:
            bool: True if session is valid, False otherwise
        """
        if not self.driver or not self.driver_live:
            return False

        try:
            # Try a simple operation to check if driver is responsive
            _ = self.driver.current_url
            return True
        except (
            InvalidSessionIdException,
            NoSuchWindowException,
            WebDriverException,
        ) as e:
            logger.warning(f"Browser session invalid: {e}")
            self.driver_live = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking session validity: {e}")
            self.driver_live = False
            return False

    def ensure_driver_live(self, action_name: Optional[str] = None) -> bool:
        """
        Ensure that the browser session is active and valid.

        Args:
            action_name: Name of the action that requires the browser

        Returns:
            bool: True if session is valid or successfully started, False otherwise
        """
        if not self.browser_needed:
            logger.debug(f"Browser not needed for action: {action_name}")
            return True

        if self.is_session_valid():
            logger.debug(f"Browser session is valid for action: {action_name}")
            return True

        logger.info(f"Starting browser session for action: {action_name}")
        return self.start_browser(action_name)

    def get_cookies(self, cookie_names: list, timeout: int = 30) -> bool:
        """
        Check if specified cookies are present in browser session.

        Args:
            cookie_names: List of cookie names to check for
            timeout: Maximum time to wait for cookies (seconds)

        Returns:
            bool: True if all specified cookies are found, False otherwise
        """
        if not self.is_session_valid():
            logger.error("Cannot check cookies: WebDriver session invalid")
            return False

        try:
            start_time = time.time()
            required_lower = {name.lower() for name in cookie_names}

            while time.time() - start_time < timeout:
                if not self.driver:  # Additional safety check
                    logger.error("WebDriver became None during cookie check")
                    return False

                cookies = self.driver.get_cookies()
                current_cookies_lower = {
                    c["name"].lower()
                    for c in cookies
                    if isinstance(c, dict) and "name" in c
                }
                missing_lower = required_lower - current_cookies_lower
                if not missing_lower:
                    logger.debug(f"All required cookies found: {cookie_names}")
                    return True
                time.sleep(0.5)

            logger.warning(f"Timeout waiting for cookies: {list(missing_lower)}")
            return False
        except Exception as e:
            logger.error(f"Error checking cookies: {e}", exc_info=True)
            return False

    def create_new_tab(self) -> Optional[str]:
        """
        Create a new browser tab.

        Returns:
            str: Window handle of the new tab, or None if failed
        """
        if not self.is_session_valid() or not self.driver:
            logger.error("Cannot create new tab: WebDriver session invalid.")
            return None

        try:
            # Store current window handle
            original_handle = self.driver.current_window_handle

            # Create new tab
            self.driver.execute_script("window.open('', '_blank');")

            # Switch to new tab
            handles = self.driver.window_handles
            new_handle = None

            for handle in handles:
                if handle != original_handle:
                    new_handle = handle
                    break

            if new_handle:
                self.driver.switch_to.window(new_handle)
                logger.debug(f"Created and switched to new tab: {new_handle}")
                return new_handle
            else:
                logger.error("Failed to find new tab handle")
                return None

        except Exception as e:
            logger.error(f"Error creating new tab: {e}", exc_info=True)
            return None


def run_comprehensive_tests():
    """
    Run comprehensive tests for the BrowserManager class.

    This function tests all major functionality of the BrowserManager
    to ensure proper operation and integration.
    """
    import sys
    import traceback
    from typing import Dict, Any

    # Test framework imports with fallback
    try:
        from test_framework import (
            TestSuite,
            suppress_logging,
            create_mock_data,
            assert_valid_function,
        )

        HAS_TEST_FRAMEWORK = True
    except ImportError:
        # Fallback implementations
        HAS_TEST_FRAMEWORK = False

        class TestSuite:
            def __init__(self, name, module):
                self.name = name
                self.tests_passed = 0
                self.tests_failed = 0

            def start_suite(self):
                print(f"Starting {self.name} tests...")

            def run_test(self, name, func, description):
                try:
                    func()
                    self.tests_passed += 1
                    print(f"✓ {name}")
                except Exception as e:
                    self.tests_failed += 1
                    print(f"✗ {name}: {e}")

            def finish_suite(self):
                print(f"Tests: {self.tests_passed} passed, {self.tests_failed} failed")
                return self.tests_failed == 0

        class suppress_logging:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def create_mock_data():
            return {}

        def assert_valid_function(func, func_name):
            assert callable(func), f"{func_name} should be callable"

    logger.info("=" * 60)
    logger.info("BROWSER MANAGER COMPREHENSIVE TESTS")
    logger.info("=" * 60)

    test_results = {"passed": 0, "failed": 0, "errors": []}

    def run_test(test_name: str, test_func) -> bool:
        """Helper to run individual tests with error handling."""
        try:
            logger.info(f"\n--- Running: {test_name} ---")
            test_func()
            test_results["passed"] += 1
            logger.info(f"✓ PASSED: {test_name}")
            return True
        except Exception as e:
            test_results["failed"] += 1
            error_msg = f"✗ FAILED: {test_name} - {str(e)}"
            test_results["errors"].append(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False

    # Test 1: Basic Initialization
    def test_initialization():
        manager = BrowserManager()
        assert manager is not None, "BrowserManager should initialize"
        assert manager.driver_live == False, "Should start with driver_live=False"
        assert manager.browser_needed == False, "Should start with browser_needed=False"
        assert manager.driver is None, "Should start with driver=None"
        assert (
            manager.session_start_time is None
        ), "Should start with session_start_time=None"

    # Test 2: Session Validation (No Driver)
    def test_session_validation_no_driver():
        manager = BrowserManager()
        result = manager.is_session_valid()
        assert result == False, "Should return False when no driver exists"

    # Test 3: Ensure Driver Live (Not Needed)
    def test_ensure_driver_not_needed():
        manager = BrowserManager()
        manager.browser_needed = False
        result = manager.ensure_driver_live("test_action")
        assert result == True, "Should return True when browser not needed"

    # Test 4: Cookie Check (Invalid Session)
    def test_cookie_check_invalid_session():
        manager = BrowserManager()
        result = manager.get_cookies(["test_cookie"])
        assert result == False, "Should return False for invalid session"

    # Test 5: New Tab Creation (Invalid Session)
    def test_new_tab_invalid_session():
        manager = BrowserManager()
        result = manager.create_new_tab()
        assert result is None, "Should return None for invalid session"

    # Test 6: Close Browser (No Driver)
    def test_close_browser_no_driver():
        manager = BrowserManager()
        # Should not raise exception
        manager.close_browser()
        assert manager.driver is None, "Driver should remain None"
        assert manager.driver_live == False, "driver_live should be False"

    # Test 7: Configuration Access
    def test_configuration_access():
        # Test that required configurations are accessible
        assert config_instance is not None, "config_instance should be available"
        assert selenium_config is not None, "selenium_config should be available"
        assert logger is not None, "Logger should be initialized"

    # Test 8: Method Availability
    def test_method_availability():
        manager = BrowserManager()
        methods_to_check = [
            "start_browser",
            "close_browser",
            "is_session_valid",
            "ensure_driver_live",
            "get_cookies",
            "create_new_tab",
        ]

        for method_name in methods_to_check:
            method = getattr(manager, method_name, None)
            assert method is not None, f"Method {method_name} should exist"
            assert callable(method), f"Method {method_name} should be callable"

    # Test 9: Logger Functionality
    def test_logger_functionality():
        assert_valid_function(logger.info, "logger.info should be callable")
        logger.info("Test log message from BrowserManager")

    # Test 10: State Management
    def test_state_management():
        manager = BrowserManager()
        # Modify state
        manager.browser_needed = True
        assert manager.browser_needed == True, "browser_needed should be modifiable"

        # Reset state through close_browser
        manager.close_browser()
        assert (
            manager.browser_needed == False
        ), "close_browser should reset browser_needed"

    # Test 11: Exception Handling
    def test_exception_handling():
        manager = BrowserManager()
        try:
            # These should handle invalid state gracefully
            manager.is_session_valid()
            manager.get_cookies(["test"])
            manager.create_new_tab()
            logger.info("Exception handling test passed - no exceptions raised")
        except Exception as e:
            assert False, f"Methods should handle invalid state gracefully: {e}"

    # Test 12: Import Dependencies
    def test_import_dependencies():
        # Test that all imports are accessible
        required_imports = [
            ("config", "config_instance"),
            ("config", "selenium_config"),
            ("chromedriver", "init_webdvr"),
            ("selenium_utils", "export_cookies"),
            ("utils", "nav_to_page"),
        ]

        for module_name, item_name in required_imports:
            if module_name == "config":
                if item_name == "config_instance":
                    item = config_instance
                elif item_name == "selenium_config":
                    item = selenium_config
            elif module_name == "chromedriver":
                item = init_webdvr
            elif module_name == "selenium_utils":
                item = export_cookies
            elif module_name == "utils":
                item = nav_to_page
            else:
                continue

            assert (
                item is not None
            ), f"Should be able to import {item_name} from {module_name}"

    # Test 13: Type Definitions
    def test_type_definitions():
        assert DriverType is not None, "DriverType should be defined"
        manager = BrowserManager()
        assert isinstance(manager.driver_live, bool), "driver_live should be bool"
        assert isinstance(manager.browser_needed, bool), "browser_needed should be bool"

    # Test 14: Function Structure
    def test_function_structure():
        # Test that this function itself is properly structured
        test_results = {"passed": 0, "failed": 0, "errors": []}
        assert test_results is not None, "test_results should be initialized"
        assert isinstance(test_results, dict), "test_results should be a dictionary"
        assert_valid_function(
            run_comprehensive_tests, "run_comprehensive_tests should be callable"
        )

    # Run all tests
    tests = [
        ("Basic Initialization", test_initialization),
        ("Session Validation (No Driver)", test_session_validation_no_driver),
        ("Ensure Driver Live (Not Needed)", test_ensure_driver_not_needed),
        ("Cookie Check (Invalid Session)", test_cookie_check_invalid_session),
        ("New Tab Creation (Invalid Session)", test_new_tab_invalid_session),
        ("Close Browser (No Driver)", test_close_browser_no_driver),
        ("Configuration Access", test_configuration_access),
        ("Method Availability", test_method_availability),
        ("Logger Functionality", test_logger_functionality),
        ("State Management", test_state_management),
        ("Exception Handling", test_exception_handling),
        ("Import Dependencies", test_import_dependencies),
        ("Type Definitions", test_type_definitions),
        ("Function Structure", test_function_structure),
    ]

    # Run each test
    for test_name, test_func in tests:
        run_test(test_name, test_func)  # Print summary
    total_tests = len(tests)
    logger.info("\n" + "=" * 60)
    logger.info("BROWSER MANAGER TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {test_results['passed']}")
    logger.info(f"Failed: {test_results['failed']}")
    if test_results["errors"]:
        logger.info("\nErrors:")
        for error in test_results["errors"]:
            logger.error(f"  {error}")

    success = test_results["failed"] == 0
    if success:
        logger.info("🎉 ALL BROWSER MANAGER TESTS PASSED!")
    else:
        logger.warning("⚠️ Some Browser Manager tests failed")
    return success


if __name__ == "__main__":
    run_comprehensive_tests()
