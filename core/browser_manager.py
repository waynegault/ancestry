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

from config.config_manager import ConfigManager
from chromedriver import init_webdvr
from selenium_utils import export_cookies
from utils import nav_to_page

# Initialize config
config_manager = ConfigManager()
config_schema = config_manager.get_config()

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
                f"Navigating to Base URL ({config_schema.api.base_url}) to stabilize..."
            )

            if not nav_to_page(self.driver, config_schema.api.base_url):
                logger.error(
                    f"Failed to navigate to base URL: {config_schema.api.base_url}"
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


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for browser_manager.py with real functionality testing.
    Tests initialization, core functionality, edge cases, integration, performance, and error handling.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Browser Management & WebDriver Operations", "browser_manager.py"
        )
        suite.start_suite()

        # INITIALIZATION TESTS
        def test_browser_manager_initialization():
            """Test BrowserManager initialization and initial state."""
            manager = BrowserManager()
            assert manager is not None, "BrowserManager should initialize"
            assert manager.driver_live == False, "Should start with driver_live=False"
            assert (
                manager.browser_needed == False
            ), "Should start with browser_needed=False"
            assert manager.driver is None, "Should start with driver=None"
            assert (
                manager.session_start_time is None
            ), "Should start with session_start_time=None"
            return True

        suite.run_test(
            "BrowserManager Initialization",
            test_browser_manager_initialization,
            "BrowserManager creates successfully with proper initial state (no active driver)",
            "Instantiate BrowserManager and verify all attributes are properly initialized",
            "Test BrowserManager initialization and default state setup",
        )

        def test_method_availability():
            """Test that all required methods are available and callable."""
            manager = BrowserManager()
            required_methods = [
                "start_browser",
                "close_browser",
                "is_session_valid",
                "ensure_driver_live",
                "get_cookies",
                "create_new_tab",
            ]

            for method_name in required_methods:
                method = getattr(manager, method_name, None)
                assert method is not None, f"Method {method_name} should exist"
                assert callable(method), f"Method {method_name} should be callable"
            return True

        suite.run_test(
            "Method Availability",
            test_method_availability,
            "All essential browser management methods are available and callable",
            "Check that all required methods exist and are callable on BrowserManager instance",
            "Test method availability and callable status for essential browser operations",
        )

        # CORE FUNCTIONALITY TESTS
        def test_session_validation_no_driver():
            """Test session validation when no driver exists."""
            manager = BrowserManager()
            result = manager.is_session_valid()
            assert result == False, "Should return False when no driver exists"
            return True

        suite.run_test(
            "Session Validation Without Driver",
            test_session_validation_no_driver,
            "Session validation returns False when no WebDriver is active",
            "Call is_session_valid() on manager with no driver and verify it returns False",
            "Test session validation behavior when no WebDriver instance exists",
        )

        def test_ensure_driver_not_needed():
            """Test ensure_driver_live when browser is not needed."""
            manager = BrowserManager()
            manager.browser_needed = False
            result = manager.ensure_driver_live("test_action")
            assert result == True, "Should return True when browser not needed"
            return True

        suite.run_test(
            "Ensure Driver When Not Needed",
            test_ensure_driver_not_needed,
            "ensure_driver_live returns True when browser_needed is False",
            "Set browser_needed=False and call ensure_driver_live to verify it returns True",
            "Test driver management when browser is not required for the action",
        )

        # EDGE CASES TESTS
        def test_cookie_check_invalid_session():
            """Test cookie retrieval with invalid/missing session."""
            manager = BrowserManager()
            result = manager.get_cookies(["test_cookie"])
            assert result == False, "Should return False for invalid session"
            return True

        suite.run_test(
            "Cookie Check Invalid Session",
            test_cookie_check_invalid_session,
            "Cookie retrieval fails gracefully when no valid WebDriver session exists",
            "Call get_cookies() without valid driver session and verify it returns False",
            "Test edge case handling for cookie operations without valid session",
        )

        def test_close_browser_no_driver():
            """Test browser closure when no driver exists."""
            manager = BrowserManager()
            # Should not raise exception
            manager.close_browser()
            assert manager.driver is None, "Driver should remain None"
            assert manager.driver_live == False, "driver_live should be False"
            return True

        suite.run_test(
            "Close Browser Without Driver",
            test_close_browser_no_driver,
            "Browser closure handles case when no driver exists without errors",
            "Call close_browser() when no driver exists and verify no exceptions occur",
            "Test graceful handling of browser closure when no WebDriver is active",
        )

        # INTEGRATION TESTS
        def test_state_management():
            """Test browser state management and transitions."""
            manager = BrowserManager()
            # Modify state
            manager.browser_needed = True
            assert manager.browser_needed == True, "browser_needed should be modifiable"

            # Reset state through close_browser
            manager.close_browser()
            assert (
                manager.browser_needed == False
            ), "close_browser should reset browser_needed"
            return True

        suite.run_test(
            "Browser State Management",
            test_state_management,
            "Browser state transitions work correctly (needed→not needed via close_browser)",
            "Set browser_needed=True, then call close_browser() and verify state is reset",
            "Test state management and transitions in browser lifecycle",
        )

        def test_configuration_access():
            """Test access to required configuration objects."""
            # These imports should be available
            assert config_schema is not None, "config_schema should be available"
            assert logger is not None, "Logger should be initialized"
            return True

        suite.run_test(
            "Configuration Access",
            test_configuration_access,
            "Required configuration objects are accessible",
            "Verify that configuration objects and logger are properly imported and available",
            "Test configuration and dependency access for browser management",
        )

        # PERFORMANCE TESTS
        def test_initialization_performance():
            """Test BrowserManager initialization performance."""
            import time

            start_time = time.time()
            for _ in range(100):
                manager = BrowserManager()
            end_time = time.time()

            total_time = end_time - start_time
            assert (
                total_time < 1.0
            ), f"100 initializations took {total_time:.3f}s, should be under 1s"
            return True

        suite.run_test(
            "Initialization Performance",
            test_initialization_performance,
            "100 BrowserManager initializations complete in under 1 second",
            "Create 100 BrowserManager instances and measure total time",
            "Test performance of BrowserManager initialization",
        )

        # ERROR HANDLING TESTS
        def test_exception_handling():
            """Test graceful exception handling for invalid operations."""
            manager = BrowserManager()
            try:
                # These should handle invalid state gracefully
                manager.is_session_valid()
                manager.get_cookies(["test"])
                result = manager.create_new_tab()
                assert (
                    result is None
                ), "create_new_tab should return None for invalid session"
            except Exception as e:
                assert False, f"Methods should handle invalid state gracefully: {e}"
            return True

        suite.run_test(
            "Exception Handling",
            test_exception_handling,
            "Browser operations handle invalid states gracefully without raising exceptions",
            "Call various browser methods without valid driver and verify no exceptions are raised",
            "Test error handling and graceful degradation for browser operations",
        )

        return suite.finish_suite()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Use centralized path management
    project_root = Path(__file__).resolve().parent.parent
    try:
        sys.path.insert(0, str(project_root))
        from path_manager import standardize_module_imports
        standardize_module_imports()
    except ImportError:
        # Fallback for testing environment
        sys.path.insert(0, str(project_root))

    run_comprehensive_tests()
