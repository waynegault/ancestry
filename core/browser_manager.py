#!/usr/bin/env python3

"""
Browser Manager - Handles all browser/WebDriver operations.

This module extracts browser management functionality from the monolithic
SessionManager class to provide a clean separation of concerns.
"""

# === CORE INFRASTRUCTURE ===
import sys
import os

# Add parent directory to path for standard_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# === THIRD-PARTY IMPORTS ===
from selenium.common.exceptions import (
    InvalidSessionIdException,
    NoSuchWindowException,
    WebDriverException,
)
from selenium.webdriver.remote.webdriver import WebDriver

# === LOCAL IMPORTS ===
from config.config_manager import ConfigManager
from chromedriver import init_webdvr
from selenium_utils import export_cookies
from utils import nav_to_page

# === MODULE CONFIGURATION ===
# Initialize config
config_manager = ConfigManager()
config_schema = config_manager.get_config()

# === TYPE ALIASES ===
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

            # Try to load saved cookies after navigating to base URL
            try:
                from utils import _load_login_cookies
                # Create a minimal session manager-like object for cookie loading
                class CookieLoader:
                    def __init__(self, driver):
                        self.driver = driver

                cookie_loader = CookieLoader(self.driver)
                if _load_login_cookies(cookie_loader):
                    logger.debug("Saved login cookies loaded successfully")
                else:
                    logger.debug("No saved cookies to load or loading failed")
            except Exception as e:
                logger.warning(f"Error loading saved cookies: {e}")

            # Mark as live and set timing
            self.driver_live = True
            self.browser_needed = True
            self.session_start_time = time.time()

            logger.debug("Browser session started successfully")
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
            logger.debug(f"Browser session invalid, will restart: {type(e).__name__}")
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

        logger.debug(f"Starting browser session for action: {action_name}")
        return self.start_browser(action_name)

    def get_cookies(self, cookie_names: list, timeout: int = 60) -> bool:
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


# === Decomposed Helper Functions ===
def _test_browser_manager_initialization():
    manager = BrowserManager()
    assert manager is not None, "BrowserManager should initialize"
    assert manager.driver_live == False, "Should start with driver_live=False"
    assert manager.browser_needed == False, "Should start with browser_needed=False"
    assert manager.driver is None, "Should start with driver=None"
    assert (
        manager.session_start_time is None
    ), "Should start with session_start_time=None"
    return True


def _test_method_availability():
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


def _test_session_validation_no_driver():
    manager = BrowserManager()
    result = manager.is_session_valid()
    assert result == False, "Should return False when no driver exists"
    return True


def _test_ensure_driver_not_needed():
    manager = BrowserManager()
    manager.browser_needed = False
    result = manager.ensure_driver_live("test_action")
    assert result == True, "Should return True when browser not needed"
    return True


def _test_cookie_check_invalid_session():
    manager = BrowserManager()
    result = manager.get_cookies(["test_cookie"])
    assert result == False, "Should return False for invalid session"
    return True


def _test_close_browser_no_driver():
    manager = BrowserManager()
    manager.close_browser()
    assert manager.driver is None, "Driver should remain None"
    assert manager.driver_live == False, "driver_live should be False"
    return True


def _test_state_management():
    manager = BrowserManager()
    manager.browser_needed = True
    assert manager.browser_needed == True, "browser_needed should be modifiable"
    manager.close_browser()
    assert manager.browser_needed == False, "close_browser should reset browser_needed"
    return True


def _test_configuration_access():
    assert config_schema is not None, "config_schema should be available"
    assert logger is not None, "Logger should be initialized"
    return True


def _test_initialization_performance():
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


def _test_exception_handling():
    manager = BrowserManager()
    try:
        manager.is_session_valid()
        manager.get_cookies(["test"])
        result = manager.create_new_tab()
        assert result is None, "create_new_tab should return None for invalid session"
    except Exception as e:
        assert False, f"Methods should handle invalid state gracefully: {e}"
    return True


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for browser_manager.py (decomposed).
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Browser Management & WebDriver Operations", "browser_manager.py"
        )
        suite.start_suite()
        suite.run_test(
            "BrowserManager Initialization",
            _test_browser_manager_initialization,
            "BrowserManager creates successfully with proper initial state (no active driver)",
            "Instantiate BrowserManager and verify all attributes are properly initialized",
            "Test BrowserManager initialization and default state setup",
        )
        suite.run_test(
            "Method Availability",
            _test_method_availability,
            "All essential browser management methods are available and callable",
            "Check that all required methods exist and are callable on BrowserManager instance",
            "Test method availability and callable status for essential browser operations",
        )
        suite.run_test(
            "Session Validation Without Driver",
            _test_session_validation_no_driver,
            "Session validation returns False when no WebDriver is active",
            "Call is_session_valid() on manager with no driver and verify it returns False",
            "Test session validation behavior when no WebDriver instance exists",
        )
        suite.run_test(
            "Ensure Driver When Not Needed",
            _test_ensure_driver_not_needed,
            "ensure_driver_live returns True when browser_needed is False",
            "Set browser_needed=False and call ensure_driver_live to verify it returns True",
            "Test driver management when browser is not required for the action",
        )
        suite.run_test(
            "Cookie Check Invalid Session",
            _test_cookie_check_invalid_session,
            "Cookie retrieval fails gracefully when no valid WebDriver session exists",
            "Call get_cookies() without valid driver session and verify it returns False",
            "Test edge case handling for cookie operations without valid session",
        )
        suite.run_test(
            "Close Browser Without Driver",
            _test_close_browser_no_driver,
            "Browser closure handles case when no driver exists without errors",
            "Call close_browser() when no driver exists and verify no exceptions occur",
            "Test graceful handling of browser closure when no WebDriver is active",
        )
        suite.run_test(
            "Browser State Management",
            _test_state_management,
            "Browser state transitions work correctly (needed→not needed via close_browser)",
            "Set browser_needed=True, then call close_browser() and verify state is reset",
            "Test state management and transitions in browser lifecycle",
        )
        suite.run_test(
            "Configuration Access",
            _test_configuration_access,
            "Required configuration objects are accessible",
            "Verify that configuration objects and logger are properly imported and available",
            "Test configuration and dependency access for browser management",
        )
        suite.run_test(
            "Initialization Performance",
            _test_initialization_performance,
            "100 BrowserManager initializations complete in under 1 second",
            "Create 100 BrowserManager instances and measure total time",
            "Test performance of BrowserManager initialization",
        )
        suite.run_test(
            "Exception Handling",
            _test_exception_handling,
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
        from core_imports import standardize_module_imports

        standardize_module_imports()
    except ImportError:
        # Fallback for testing environment
        sys.path.insert(0, str(project_root))

    run_comprehensive_tests()
