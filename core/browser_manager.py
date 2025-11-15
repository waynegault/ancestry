#!/usr/bin/env python3

"""
Browser Manager - Handles all browser/WebDriver operations.

This module extracts browser management functionality from the monolithic
SessionManager class to provide a clean separation of concerns.
"""

# === CORE INFRASTRUCTURE ===
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for standard_imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import time
from pathlib import Path
from typing import Any, Optional

# === THIRD-PARTY IMPORTS ===
from selenium.common.exceptions import (
    InvalidSessionIdException,
    NoSuchWindowException,
    WebDriverException,
)
from selenium.webdriver.remote.webdriver import WebDriver

from chromedriver import init_webdvr

# === LOCAL IMPORTS ===
from config.config_manager import ConfigManager
from utils import nav_to_page

# === MODULE CONFIGURATION ===
# Initialize config
config_manager = ConfigManager()
config_schema = config_manager.get_config()

# === TYPE ALIASES ===
DriverType = WebDriver | None


class BrowserManager:
    """Manages browser/WebDriver operations and state."""

    def __init__(self) -> None:
        """Initialize the BrowserManager."""
        self.driver: DriverType = None
        self.driver_live: bool = False
        self.browser_needed: bool = False
        self.session_start_time: float | None = None

        logger.debug("BrowserManager initialized")

    def _log_browser_initialization_error(self) -> None:
        """Log detailed browser initialization error message."""
        logger.error("WebDriver initialization failed (init_webdvr returned None).")
        logger.error("=" * 80)
        logger.error("BROWSER INITIALIZATION FAILED")
        logger.error("=" * 80)
        logger.error("Possible causes:")
        logger.error("  1. Chrome is already running - close all Chrome instances")
        logger.error("  2. Chrome profile is corrupted - delete/rename profile")
        logger.error("  3. Chrome/ChromeDriver version mismatch")
        logger.error("  4. Security software blocking Chrome")
        logger.error("")
        logger.error("Run diagnostics: python diagnose_chrome.py")
        logger.error("=" * 80)

    def _verify_browser_window(self) -> bool:
        """Verify browser window is actually open."""
        if not self.driver:
            return False
        try:
            _ = self.driver.current_url
            return True
        except Exception as verify_err:
            logger.error(f"Browser window closed immediately after initialization: {verify_err}")
            logger.error("This indicates a critical Chrome/ChromeDriver issue")
            logger.error("Run diagnostics: python diagnose_chrome.py")
            self.close_browser()
            return False

    def _load_saved_cookies(self) -> None:
        """Try to load saved cookies after navigating to base URL."""
        try:
            from utils import _load_login_cookies
            # Create a minimal session manager-like object for cookie loading
            class CookieLoader:
                def __init__(self, driver: Any) -> None:
                    self.driver = driver

            cookie_loader = CookieLoader(self.driver)
            if _load_login_cookies(cookie_loader):  # type: ignore[arg-type] - CookieLoader has compatible .driver attribute
                logger.debug("Saved login cookies loaded successfully")
            else:
                logger.debug("No saved cookies to load or loading failed")
        except Exception as e:
            logger.warning(f"Error loading saved cookies: {e}")

    def _restore_terminal_focus(self) -> None:
        """Force focus back to the console window on Windows."""
        if sys.platform != "win32":
            return

        try:
            import ctypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            hwnd = kernel32.GetConsoleWindow()
            if not hwnd:
                return

            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE

            user32.SetForegroundWindow(hwnd)
            logger.debug("Terminal focus restored after browser minimization")
        except Exception as exc:
            logger.debug(f"Unable to restore terminal focus: {exc}")

    def _minimize_browser_window(self) -> None:
        """
        Minimize browser window after launch.

        Note: Small delay added to ensure Chrome is fully initialized before minimizing.
        This prevents Chrome 142+ from closing immediately on some Windows configurations.
        """
        if not self.driver:
            logger.warning("Cannot minimize browser: driver not initialized")
            return

        # Small delay to let Chrome stabilize (prevents immediate closure on Chrome 142+)
        import time
        time.sleep(0.5)

        def _safe_resize(driver: Any, width: int, height: int) -> None:
            try:
                current_rect = driver.get_window_rect()
                driver.set_window_rect(
                    x=current_rect.get("x", 0),
                    y=current_rect.get("y", 0),
                    width=width,
                    height=height,
                )
            except Exception:
                driver.set_window_position(0, 0)
                driver.set_window_size(width, height)

        try:
            # Primary method: use WebDriver's minimize_window()
            _safe_resize(self.driver, 1, 1)
            self.driver.minimize_window()
            logger.debug("✅ Browser window minimized successfully")
            self._restore_terminal_focus()
            return
        except Exception as primary_error:
            logger.warning(f"Primary minimize method failed: {primary_error}")

            # Fallback: try setting window position off-screen
            try:
                logger.debug("Attempting fallback: moving window off-screen")
                _safe_resize(self.driver, 1, 1)
                self.driver.set_window_position(-2000, -2000)
                logger.info("⚠️ Browser minimized using fallback method (off-screen positioning)")
                self._restore_terminal_focus()
                return
            except Exception as fallback_error:
                logger.error(
                    f"❌ Failed to minimize browser window (tried 2 methods). "
                    f"Primary: {primary_error}, Fallback: {fallback_error}"
                )
                logger.error("Browser will remain visible - this may be a WebDriver/platform limitation")

    def start_browser(self, action_name: str | None = None) -> bool:
        """
        Start the browser session.

        Args:
            action_name: Optional name of the action for logging

        Returns:
            bool: True if browser started successfully, False otherwise
        """
        try:
            if self.is_session_valid():
                logger.debug("Browser already running and valid")
                return True

            logger.debug(f"🌐 Initializing browser for {action_name or 'action'}...")
            self.driver = init_webdvr()

            if not self.driver:
                self._log_browser_initialization_error()
                return False

            # Verify browser is actually open
            if not self._verify_browser_window():
                return False

            # DON'T navigate during browser startup - let SessionManager handle navigation
            # This prevents circular dependency issues with nav_to_page needing session_manager
            # The browser will navigate to about:blank initially, then SessionManager will
            # handle authentication and navigation to ancestry.co.uk
            logger.debug("✅ Browser initialized (navigation will be handled by SessionManager)")

            # Minimize window after small delay for stability
            self._minimize_browser_window()

            # Mark as live and set timing
            self.driver_live = True
            self.browser_needed = True
            self.session_start_time = time.time()

            logger.debug("✅ Browser ready for session authentication")
            return True

        except Exception as e:
            logger.error(f"Failed to start browser: {e}", exc_info=True)
            self.close_browser()
            return False

    def close_browser(self) -> None:
        """Close the browser and cleanup resources."""
        if self.driver:
            try:
                # Suppress stderr to hide undetected_chromedriver cleanup errors on Windows
                # (OSError: [WinError 6] The handle is invalid)
                import contextlib
                import io
                import sys

                # Store driver reference and clear it immediately to prevent gc issues
                driver_to_close = self.driver
                self.driver = None

                with contextlib.redirect_stderr(io.StringIO()):
                    driver_to_close.quit()
                    # Give Windows time to release handles before gc runs
                    time.sleep(0.1)
                    # Explicitly delete to prevent gc errors
                    del driver_to_close
            except Exception as e:
                logger.warning(f"Error quitting WebDriver: {e}")
                # Ensure driver reference is cleared even on error
                self.driver = None

        # Reset state
        self.driver = None  # Ensure it's cleared
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

    def ensure_driver_live(self, action_name: str | None = None) -> bool:
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

    def get_cookies(self, cookie_names: list, timeout: int = 10) -> bool:
        """
        Check if specified cookies are present in browser session.

        Args:
            cookie_names: List of cookie names to check for
            timeout: Maximum time to wait for cookies (seconds, default 10s)

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

    def create_new_tab(self) -> str | None:
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
            logger.error("Failed to find new tab handle")
            return None

        except Exception as e:
            logger.error(f"Error creating new tab: {e}", exc_info=True)
            return None


# === Decomposed Helper Functions ===
def _test_browser_manager_initialization():
    manager = BrowserManager()
    assert manager is not None, "BrowserManager should initialize"
    assert not manager.driver_live, "Should start with driver_live=False"
    assert not manager.browser_needed, "Should start with browser_needed=False"
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
    assert not result, "Should return False when no driver exists"
    return True


def _test_ensure_driver_not_needed():
    manager = BrowserManager()
    manager.browser_needed = False
    result = manager.ensure_driver_live("test_action")
    assert result, "Should return True when browser not needed"
    return True


def _test_cookie_check_invalid_session():
    manager = BrowserManager()
    result = manager.get_cookies(["test_cookie"])
    assert not result, "Should return False for invalid session"
    return True


def _test_close_browser_no_driver():
    manager = BrowserManager()
    manager.close_browser()
    assert manager.driver is None, "Driver should remain None"
    assert not manager.driver_live, "driver_live should be False"
    return True


def _test_state_management():
    manager = BrowserManager()
    manager.browser_needed = True
    assert manager.browser_needed, "browser_needed should be modifiable"
    manager.close_browser()
    assert not manager.browser_needed, "close_browser should reset browser_needed"
    return True


def _test_configuration_access():
    assert config_schema is not None, "config_schema should be available"
    assert logger is not None, "Logger should be initialized"
    return True


def _test_initialization_performance():
    import time

    start_time = time.time()
    for _ in range(100):
        _ = BrowserManager()
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
        raise AssertionError(f"Methods should handle invalid state gracefully: {e}") from e
    return True


def _test_cookie_timeout_default():
    """Test that default cookie timeout is 10 seconds (not 60)."""
    import inspect

    manager = BrowserManager()
    sig = inspect.signature(manager.get_cookies)
    timeout_param = sig.parameters.get("timeout")
    assert timeout_param is not None, "get_cookies should have timeout parameter"
    assert timeout_param.default == 10, f"Default timeout should be 10s, got {timeout_param.default}s"
    return True


def _test_cookie_timeout_custom():
    """Test that custom cookie timeout can be specified."""
    from unittest.mock import Mock, patch

    manager = BrowserManager()
    manager.driver = Mock()
    manager.driver_live = True

    # Mock get_cookies to return False after timeout
    with patch.object(manager, "is_session_valid", return_value=True):
        # This should timeout quickly with custom timeout
        result = manager.get_cookies(["nonexistent_cookie"], timeout=1)
        assert result is False, "Should return False when cookie not found within timeout"

    return True


def _test_cookie_check_prevents_long_waits():
    """Test that cookie check doesn't wait excessively for missing cookies."""
    import time
    from unittest.mock import Mock, patch

    manager = BrowserManager()
    manager.driver = Mock()
    manager.driver_live = True
    manager.driver.get_cookies.return_value = []  # No cookies

    with patch.object(manager, "is_session_valid", return_value=True):
        start_time = time.time()
        result = manager.get_cookies(["missing_cookie"], timeout=2)
        elapsed_time = time.time() - start_time

        assert result is False, "Should return False when cookie not found"
        assert elapsed_time < 3, f"Should timeout in ~2s, took {elapsed_time:.1f}s"

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
        suite.run_test(
            "Cookie Timeout Default Value",
            _test_cookie_timeout_default,
            "Default cookie timeout is 10 seconds (not 60) to prevent long waits",
            "Inspect get_cookies method signature and verify default timeout parameter is 10",
            "Test that cookie timeout default prevents excessive waits for missing cookies",
        )
        suite.run_test(
            "Cookie Timeout Custom Value",
            _test_cookie_timeout_custom,
            "Custom cookie timeout can be specified when calling get_cookies",
            "Call get_cookies with custom timeout and verify it respects the parameter",
            "Test custom timeout parameter handling in cookie retrieval",
        )
        suite.run_test(
            "Cookie Check Prevents Long Waits",
            _test_cookie_check_prevents_long_waits,
            "Cookie check completes within timeout period for missing cookies",
            "Call get_cookies with 2s timeout for missing cookie and verify it completes in ~2s",
            "Test that cookie check doesn't wait excessively for non-existent cookies",
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
