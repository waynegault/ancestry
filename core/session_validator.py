#!/usr/bin/env python3

"""
Session Validator - Handles session validation and readiness checks.

This module extracts session validation functionality from the monolithic
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

# === STANDARD LIBRARY IMPORTS ===
from datetime import datetime, timezone
from typing import List, Optional, Tuple

# === THIRD-PARTY IMPORTS ===
from selenium.common.exceptions import WebDriverException

# === LOCAL IMPORTS ===
from config.config_manager import ConfigManager

# Initialize config
config_manager = ConfigManager()
config_schema = config_manager.get_config()


class SessionValidator:
    """
    Handles session validation and readiness checks.

    This class manages all session validation functionality including:
    - Login status verification
    - Cookie validation
    - URL checking and handling
    - Readiness checks coordination
    """

    def __init__(self):
        """Initialize the SessionValidator."""
        self.last_js_error_check: datetime = datetime.now(timezone.utc)
        logger.debug("SessionValidator initialized")

    def perform_readiness_checks(
        self,
        browser_manager,
        api_manager,
        action_name: Optional[str] = None,
        max_attempts: int = 3,
    ) -> bool:
        """
        Perform comprehensive readiness checks for the session.

        Args:
            browser_manager: BrowserManager instance
            api_manager: APIManager instance
            action_name: Optional name of the action for logging
            max_attempts: Maximum number of attempts

        Returns:
            bool: True if all checks pass, False otherwise
        """
        logger.debug(
            f"Starting readiness checks for: {action_name or 'Unknown Action'}"
        )
        last_check_error = "Unknown error"

        for attempt in range(1, max_attempts + 1):
            logger.debug(f"Readiness check attempt {attempt} of {max_attempts}")

            try:
                # Check login status and attempt relogin if needed
                login_success, login_error = self._check_login_and_attempt_relogin(
                    browser_manager, attempt
                )
                if not login_success:
                    last_check_error = login_error
                    continue

                # Check and handle current URL
                if not self._check_and_handle_url(browser_manager):
                    logger.error("URL check/handling failed.")
                    last_check_error = "URL check/handling failed"
                    continue

                logger.debug("URL check/handling OK.")

                # Check essential cookies
                cookies_success, cookies_error = self._check_essential_cookies(
                    browser_manager
                )
                if not cookies_success:
                    last_check_error = cookies_error
                    continue

                # Sync cookies to requests session
                sync_success, sync_error = self._sync_cookies_to_requests(
                    browser_manager, api_manager
                )
                if not sync_success:
                    last_check_error = sync_error
                    continue

                # Check CSRF token
                csrf_success, csrf_error = self._check_csrf_token(api_manager)
                if not csrf_success:
                    last_check_error = csrf_error
                    continue

                # All checks passed
                logger.info(f"Readiness checks PASSED on attempt {attempt}.")
                return True

            except WebDriverException as wd_exc:
                logger.error(
                    f"WebDriverException during readiness check attempt {attempt}: {wd_exc}",
                    exc_info=False,
                )
                last_check_error = f"WebDriverException: {wd_exc}"

                if not browser_manager.is_session_valid():
                    logger.error(
                        "Session invalid during readiness check. Aborting checks."
                    )
                    return False

            except Exception as exc:
                logger.error(
                    f"Unexpected exception during readiness check attempt {attempt}: {exc}",
                    exc_info=True,
                )
                last_check_error = f"Exception: {exc}"

            # Wait before next attempt (except on last attempt)
            if attempt < max_attempts:
                import time

                time.sleep(2)

        logger.error(
            f"All {max_attempts} readiness check attempts failed. Last Error: {last_check_error}"
        )
        return False

    def _check_login_and_attempt_relogin(
        self, browser_manager, attempt: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check login status and attempt relogin if needed.

        Args:
            browser_manager: BrowserManager instance
            attempt: Current attempt number

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Import login_status here to avoid circular imports
            from utils import login_status

            logger.debug(f"Checking login status (attempt {attempt})...")
            # Get session manager from browser manager
            session_manager = getattr(browser_manager, "session_manager", None)
            if not session_manager:
                logger.error("No session manager available for login status check")
                return False, "No session manager available"

            login_ok = login_status(
                session_manager, disable_ui_fallback=True
            )  # Use API-only check

            if login_ok is True:
                logger.debug("Login status check: User is logged in.")
                return True, None
            elif login_ok is False:
                logger.warning(
                    "Login status check: User is NOT logged in. Attempting relogin..."
                )

                # Attempt relogin
                relogin_success = self._attempt_relogin(browser_manager)
                if relogin_success:
                    logger.info("Relogin successful.")
                    return True, None
                else:
                    error_msg = "Relogin failed"
                    logger.error(error_msg)
                    return False, error_msg
            else:  # login_ok is None
                error_msg = "Login status check returned None (critical failure)"
                logger.error(error_msg)
                return False, error_msg

        except Exception as e:
            error_msg = f"Exception during login check: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def _attempt_relogin(self, browser_manager) -> bool:
        """
        Attempt to relogin the user.

        Args:
            browser_manager: BrowserManager instance

        Returns:
            bool: True if relogin successful, False otherwise
        """
        try:
            # Import log_in function here to avoid circular imports
            from utils import log_in

            logger.debug("Attempting relogin...")
            # Get session manager from browser manager
            session_manager = getattr(browser_manager, "session_manager", None)
            if not session_manager:
                logger.error("No session manager available for relogin")
                return False

            login_result = log_in(session_manager)  # Pass session manager

            if login_result == "LOGIN_SUCCEEDED":
                logger.info("Relogin successful.")
                return True
            else:
                logger.error(f"Relogin failed: {login_result}")
                return False

        except Exception as e:
            logger.error(f"Exception during relogin attempt: {e}", exc_info=True)
            return False

    def _check_and_handle_url(self, browser_manager) -> bool:
        """
        Check and handle the current URL.

        Args:
            browser_manager: BrowserManager instance

        Returns:
            bool: True if URL handling successful, False otherwise
        """
        if not browser_manager.is_session_valid():
            logger.error("Cannot check URL: Browser session invalid.")
            return False

        try:
            current_url = browser_manager.driver.current_url
            logger.debug(f"Current URL: {current_url}")

            # Check if we're on a valid Ancestry page
            base_url = config_schema.api.base_url or "https://www.ancestry.com"
            if not current_url or not current_url.startswith(base_url):
                logger.warning(f"Not on Ancestry domain. Navigating to base URL...")

                # Import nav_to_page here to avoid circular imports
                from utils import nav_to_page

                nav_success = nav_to_page(
                    browser_manager.driver,
                    base_url,
                    selector="body",
                    session_manager=getattr(browser_manager, "session_manager", None),
                )

                if not nav_success:
                    logger.error("Failed to navigate to base URL.")
                    return False

                logger.debug("Successfully navigated to base URL.")

            return True

        except WebDriverException as e:
            logger.error(f"WebDriverException checking URL: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking URL: {e}", exc_info=True)
            return False

    def _check_essential_cookies(self, browser_manager) -> Tuple[bool, Optional[str]]:
        """
        Check for essential cookies.

        Args:
            browser_manager: BrowserManager instance

        Returns:
            Tuple of (success, error_message)
        """
        essential_cookies = ["OptanonConsent", "trees"]  # Add more as needed

        try:
            if not browser_manager.get_cookies(essential_cookies):
                error_msg = f"Essential cookies not found: {essential_cookies}"
                logger.warning(error_msg)
                return False, error_msg

            logger.debug("Essential cookies check passed.")
            return True, None

        except Exception as e:
            error_msg = f"Exception checking essential cookies: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def _sync_cookies_to_requests(
        self, browser_manager, api_manager
    ) -> Tuple[bool, Optional[str]]:
        """
        Sync cookies from browser to API requests session.

        Args:
            browser_manager: BrowserManager instance
            api_manager: APIManager instance

        Returns:
            Tuple of (success, error_message)
        """
        try:
            sync_success = api_manager.sync_cookies_from_browser(browser_manager)
            if not sync_success:
                error_msg = "Failed to sync cookies to requests session"
                logger.error(error_msg)
                return False, error_msg

            logger.debug("Cookie sync to requests session successful.")
            return True, None

        except Exception as e:
            error_msg = f"Exception syncing cookies: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def _check_csrf_token(self, api_manager) -> Tuple[bool, Optional[str]]:
        """
        Check and retrieve CSRF token if needed.

        Args:
            api_manager: APIManager instance

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if not api_manager.csrf_token:
                logger.debug("CSRF token not available. Attempting to retrieve...")
                csrf_token = api_manager.get_csrf_token()

                if not csrf_token:
                    # CSRF token failure is non-critical for some operations
                    logger.warning("Failed to retrieve CSRF token (non-critical).")
                    return True, None  # Continue anyway

                logger.debug("CSRF token retrieved successfully.")
            else:
                logger.debug("CSRF token already available.")

            return True, None

        except Exception as e:
            error_msg = f"Exception checking CSRF token: {e}"
            logger.error(error_msg, exc_info=True)
            # CSRF token errors are non-critical
            return True, None

    def validate_session_cookies(
        self, browser_manager, required_cookies: List[str]
    ) -> bool:
        """
        Validate that required cookies are present.

        Args:
            browser_manager: BrowserManager instance
            required_cookies: List of required cookie names

        Returns:
            bool: True if all required cookies are present, False otherwise
        """
        try:
            if not browser_manager.is_session_valid():
                logger.error("Cannot validate cookies: Browser session invalid.")
                return False
            return browser_manager.get_cookies(required_cookies)
        except Exception as e:
            logger.error(f"Error validating session cookies: {e}", exc_info=True)
            return False

    def verify_login_status(self, api_manager) -> bool:
        """
        Verify login status using multiple methods.

        Args:
            api_manager: APIManager instance

        Returns:
            bool: True if logged in, False otherwise
        """
        logger.debug("Verifying login status...")

        try:
            # Try API-based verification first
            api_login_status = api_manager.verify_api_login_status()

            if api_login_status is True:
                logger.debug("Login verification successful (API method).")
                return True
            elif api_login_status is False:
                logger.warning("Login verification failed (API method).")
                return False
            else:
                logger.error(
                    "Login verification failed critically (API returned None)."
                )
                return False

        except Exception as e:
            logger.error(
                f"Unexpected error during login verification: {e}", exc_info=True
            )
            return False


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for session_validator.py with real functionality testing.
    Tests initialization, core functionality, edge cases, integration, performance, and error handling.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Session Validation & Readiness Checks", "session_validator.py"
        )
        suite.start_suite()

        # INITIALIZATION TESTS
        def test_session_validator_initialization():
            """Test SessionValidator initialization and component setup."""
            validator = SessionValidator()
            assert validator is not None, "SessionValidator should initialize"
            assert hasattr(
                validator, "last_js_error_check"
            ), "Should have last_js_error_check attribute"
            assert (
                validator.last_js_error_check is not None
            ), "last_js_error_check should be initialized"
            from datetime import datetime

            assert isinstance(
                validator.last_js_error_check, datetime
            ), "last_js_error_check should be datetime"
            return True

        suite.run_test(
            "SessionValidator Initialization",
            test_session_validator_initialization,
            "SessionValidator creates successfully with required attributes for session validation",
            "Instantiate SessionValidator and verify required attributes are properly initialized",
            "Test SessionValidator initialization and attribute setup",
        )

        # CORE FUNCTIONALITY TESTS
        def test_readiness_checks_success():
            """Test successful readiness checks flow with mocked dependencies."""
            from unittest.mock import Mock, patch

            validator = SessionValidator()
            mock_browser = Mock()
            mock_api = Mock()

            # Mock all internal check methods to return success
            with patch.object(
                validator, "_check_login_and_attempt_relogin", return_value=(True, None)
            ), patch.object(
                validator, "_check_and_handle_url", return_value=True
            ), patch.object(
                validator, "_check_essential_cookies", return_value=(True, None)
            ), patch.object(
                validator, "_sync_cookies_to_requests", return_value=(True, None)
            ), patch.object(
                validator, "_check_csrf_token", return_value=(True, None)
            ):

                result = validator.perform_readiness_checks(
                    mock_browser, mock_api, "test_action"
                )
                assert (
                    result is True
                ), "Readiness checks should succeed when all sub-checks pass"
            return True

        suite.run_test(
            "Readiness Checks Success Flow",
            test_readiness_checks_success,
            "All readiness checks pass when mocked dependencies return success",
            "Mock all internal validation methods to return success and verify overall result",
            "Test successful execution path of readiness checks with mocked dependencies",
        )

        def test_login_verification():
            """Test login status verification functionality."""
            from unittest.mock import Mock

            validator = SessionValidator()
            mock_api = Mock()
            mock_api.verify_api_login_status.return_value = True

            result = validator.verify_login_status(mock_api)
            assert (
                result is True
            ), "Login verification should succeed with valid API response"
            mock_api.verify_api_login_status.assert_called_once()
            return True

        suite.run_test(
            "Login Status Verification",
            test_login_verification,
            "Login verification succeeds when API reports user is logged in",
            "Mock API to return successful login status and verify verification result",
            "Test login status verification with mocked API response",
        )

        # EDGE CASES TESTS
        def test_invalid_browser_session():
            """Test handling of invalid browser session."""
            from unittest.mock import Mock

            validator = SessionValidator()
            mock_browser = Mock()
            mock_browser.is_session_valid.return_value = False

            result = validator.validate_session_cookies(mock_browser, ["test_cookie"])
            assert result is False, "Should fail with invalid browser session"
            return True

        suite.run_test(
            "Invalid Browser Session Handling",
            test_invalid_browser_session,
            "Cookie validation fails gracefully when browser session is invalid",
            "Mock browser to return invalid session status and verify validation fails",
            "Test edge case handling for invalid browser sessions",
        )

        def test_login_verification_failure():
            """Test login verification failure cases."""
            from unittest.mock import Mock

            validator = SessionValidator()
            mock_api = Mock()
            mock_api.verify_api_login_status.return_value = False

            result = validator.verify_login_status(mock_api)
            assert result is False, "Should fail when API reports not logged in"
            return True

        suite.run_test(
            "Login Verification Failure",
            test_login_verification_failure,
            "Login verification fails when API reports user is not logged in",
            "Mock API to return failed login status and verify verification fails",
            "Test login verification failure handling",
        )

        # INTEGRATION TESTS
        def test_full_validation_workflow():
            """Test complete validation workflow integration."""
            from unittest.mock import Mock

            validator = SessionValidator()
            mock_browser = Mock()
            mock_api = Mock()

            # Set up successful responses
            mock_browser.is_session_valid.return_value = True
            mock_browser.get_cookies.return_value = True
            mock_api.verify_api_login_status.return_value = True

            # Test cookie validation
            cookie_result = validator.validate_session_cookies(
                mock_browser, ["session_cookie"]
            )
            assert cookie_result is True, "Cookie validation should succeed"

            # Test login verification
            login_result = validator.verify_login_status(mock_api)
            assert login_result is True, "Login verification should succeed"
            return True

        suite.run_test(
            "Full Validation Workflow Integration",
            test_full_validation_workflow,
            "Complete validation workflow succeeds when all components work together",
            "Test both cookie validation and login verification in sequence with mocked success responses",
            "Test integration of cookie validation and login verification workflows",
        )

        # PERFORMANCE TESTS
        def test_initialization_performance():
            """Test SessionValidator initialization performance."""
            import time

            start_time = time.time()
            for _ in range(100):
                validator = SessionValidator()
            end_time = time.time()

            total_time = end_time - start_time
            assert (
                total_time < 1.0
            ), f"100 initializations took {total_time:.3f}s, should be under 1s"
            return True

        suite.run_test(
            "Initialization Performance",
            test_initialization_performance,
            "100 SessionValidator initializations complete in under 1 second",
            "Create 100 SessionValidator instances and measure total time",
            "Test performance of SessionValidator initialization",
        )

        # ERROR HANDLING TESTS
        def test_webdriver_exception_handling():
            """Test handling of WebDriver exceptions."""
            from unittest.mock import Mock, patch
            from selenium.common.exceptions import WebDriverException

            validator = SessionValidator()
            mock_browser = Mock()
            mock_api = Mock()

            # Mock WebDriverException during readiness checks
            with patch.object(
                validator, "_check_login_and_attempt_relogin"
            ) as mock_login:
                mock_login.side_effect = WebDriverException("Browser crashed")
                mock_browser.is_session_valid.return_value = True

                result = validator.perform_readiness_checks(
                    mock_browser, mock_api, max_attempts=1
                )
                assert result is False, "Should fail when WebDriverException occurs"
            return True

        suite.run_test(
            "WebDriver Exception Handling",
            test_webdriver_exception_handling,
            "Readiness checks fail gracefully when WebDriver exceptions occur",
            "Mock WebDriverException during login check and verify graceful failure",
            "Test error handling for WebDriver exceptions during validation",
        )

        def test_general_exception_handling():
            """Test handling of general exceptions."""
            from unittest.mock import Mock

            validator = SessionValidator()
            mock_browser = Mock()
            mock_browser.is_session_valid.side_effect = Exception("Unexpected error")

            result = validator.validate_session_cookies(mock_browser, ["test_cookie"])
            assert result is False, "Should handle unexpected exceptions gracefully"
            return True

        suite.run_test(
            "General Exception Handling",
            test_general_exception_handling,
            "Cookie validation handles unexpected exceptions gracefully",
            "Mock browser to throw unexpected exception and verify graceful failure",
            "Test error handling for general exceptions during validation",
        )

        return suite.finish_suite()


if __name__ == "__main__":
    run_comprehensive_tests()
