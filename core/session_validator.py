#!/usr/bin/env python3

"""
Session Validator - Handles session validation and readiness checks.

This module extracts session validation functionality from the monolithic
SessionManager class to provide a clean separation of concerns.
"""

# === CORE INFRASTRUCTURE ===
import sys

# Add parent directory to path for standard_imports
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
from datetime import datetime, timezone
from typing import Optional

# === THIRD-PARTY IMPORTS ===
from selenium.common.exceptions import WebDriverException

# === LOCAL IMPORTS ===
from config import config_schema

# Use global cached config instance


class SessionValidator:
    """
    Handles session validation and readiness checks.

    This class manages all session validation functionality including:
    - Login status verification
    - Cookie validation
    - URL checking and handling
    - Readiness checks coordination
    """

    def __init__(self) -> None:
        """Initialize the SessionValidator."""
        self.last_js_error_check: datetime = datetime.now(timezone.utc)
        logger.debug("SessionValidator initialized")

    def _perform_all_checks(
        self,
        browser_manager,
        api_manager,
        session_manager,
        action_name: Optional[str],
        skip_csrf: bool,
        attempt: int,
    ) -> tuple[bool, str]:
        """Perform all readiness checks. Returns (success, error_message)."""
        # Check login status and attempt relogin if needed
        login_success, login_error = self._check_login_and_attempt_relogin(
            browser_manager, session_manager, attempt
        )
        if not login_success:
            return False, login_error

        # Check and handle current URL
        if not self._check_and_handle_url(browser_manager):
            logger.error("URL check/handling failed.")
            return False, "URL check/handling failed"

        logger.debug("URL check/handling OK.")

        # Check essential cookies
        cookies_success, cookies_error = self._check_essential_cookies(
            browser_manager, action_name
        )
        if not cookies_success:
            return False, cookies_error

        # Sync cookies to requests session
        sync_success, sync_error = self._sync_cookies_to_requests(
            browser_manager, api_manager
        )
        if not sync_success:
            return False, sync_error

        # Check CSRF token (skip if not needed)
        if not skip_csrf:
            csrf_success, csrf_error = self._check_csrf_token(api_manager)
            if not csrf_success:
                return False, csrf_error
        else:
            logger.debug("Skipping CSRF token check as requested")

        return True, ""

    def _handle_check_exception(
        self, exception: Exception, attempt: int, browser_manager
    ) -> tuple[bool, str]:
        """Handle exceptions during checks. Returns (should_abort, error_message)."""
        if isinstance(exception, WebDriverException):
            logger.error(
                f"WebDriverException during readiness check attempt {attempt}: {exception}",
                exc_info=False,
            )
            error_msg = f"WebDriverException: {exception}"

            if not browser_manager.is_session_valid():
                logger.error(
                    "Session invalid during readiness check. Aborting checks."
                )
                return True, error_msg  # Abort

            return False, error_msg  # Don't abort, retry

        # Other exceptions
        logger.error(
            f"Unexpected exception during readiness check attempt {attempt}: {exception}",
            exc_info=True,
        )
        return False, f"Exception: {exception}"  # Don't abort, retry

    def perform_readiness_checks(
        self,
        browser_manager,
        api_manager,
        session_manager,
        action_name: Optional[str] = None,
        max_attempts: int = 3,
        skip_csrf: bool = False,
    ) -> bool:
        """
        Perform comprehensive readiness checks for the session.

        Args:
            browser_manager: BrowserManager instance
            api_manager: APIManager instance
            session_manager: SessionManager instance for login checks
            action_name: Optional name of the action for logging
            max_attempts: Maximum number of attempts
            skip_csrf: Skip CSRF token validation (for actions that don't need it)

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
                # Perform all checks
                success, error = self._perform_all_checks(
                    browser_manager, api_manager, session_manager,
                    action_name, skip_csrf, attempt
                )

                if success:
                    logger.debug(f"Readiness checks PASSED on attempt {attempt}.")
                    return True

                last_check_error = error

            except Exception as exc:
                should_abort, error_msg = self._handle_check_exception(
                    exc, attempt, browser_manager
                )
                last_check_error = error_msg

                if should_abort:
                    return False

            # Wait before next attempt (except on last attempt)
            if attempt < max_attempts:
                import time
                time.sleep(5)  # Increased from 2 to 5 seconds for better stability

        logger.error(
            f"All {max_attempts} readiness check attempts failed. Last Error: {last_check_error}"
        )
        return False

    def _check_login_and_attempt_relogin(
        self, browser_manager, session_manager, attempt: int
    ) -> tuple[bool, Optional[str]]:
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
            # Use the session manager parameter directly
            if not session_manager:
                logger.error("No session manager provided for login status check")
                return False, "No session manager provided"

            login_ok = login_status(
                session_manager, disable_ui_fallback=True
            )  # Use API-only check

            if login_ok is True:
                logger.debug("Login status check: User is logged in.")
                return True, None
            if login_ok is False:
                logger.warning(
                    "Login status check: User is NOT logged in. Attempting relogin..."
                )

                # Attempt relogin
                relogin_success = self._attempt_relogin(browser_manager, session_manager)
                if relogin_success:
                    logger.info("Relogin successful.")
                    return True, None
                error_msg = "Relogin failed"
                logger.error(error_msg)
                return False, error_msg
            # login_ok is None
            error_msg = "Login status check returned None (critical failure)"
            logger.error(error_msg)
            return False, error_msg

        except Exception as e:
            error_msg = f"Exception during login check: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def _attempt_relogin(self, _browser_manager, session_manager) -> bool:
        """
        Attempt to relogin the user.

        Args:
            browser_manager: BrowserManager instance
            session_manager: SessionManager instance

        Returns:
            bool: True if relogin successful, False otherwise
        """
        try:
            # Import log_in function here to avoid circular imports
            from utils import log_in

            logger.debug("Attempting relogin...")
            # Use the session manager parameter directly
            if not session_manager:
                logger.error("No session manager provided for relogin")
                return False

            login_result = log_in(session_manager)  # Pass session manager

            if login_result == "LOGIN_SUCCEEDED":
                logger.info("Relogin successful.")
                return True
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
                logger.warning("Not on Ancestry domain. Navigating to base URL...")

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

    def _should_skip_cookie_check(self, action_name: Optional[str]) -> tuple[bool, Optional[str]]:
        """Determine if cookie check should be skipped for this action."""
        if not action_name:
            return False, None

        # Map of action patterns to skip reasons
        skip_patterns = {
            "coord": "Action 6 - cookies will be available after navigation",
            "action6b": "Action 6B - cookies will be available after navigation",
            "srch_inbox": "Action 7 - API login verification is sufficient",
            "send_messages": "Action 8 - API login verification is sufficient",
            "process_productive": "Action 9 - API login verification is sufficient",
            "main": "Action 10 - Local file operation",
            "gedcom": "Action 10 - Local file operation",
            "run_action11": "Action 11 - API-based operation",
            "api_report": "Action 11 - API-based operation",
            "refresh": "Browser refresh verification - deferring to later checks",
        }

        action_lower = action_name.lower()
        for pattern, reason in skip_patterns.items():
            if pattern in action_lower:
                logger.debug(f"Skipping essential cookies check for {reason}")
                return True, reason

        return False, None

    def _check_essential_cookies(self, browser_manager, action_name: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """
        Check for essential cookies.

        Args:
            browser_manager: BrowserManager instance
            action_name: Optional action name for context

        Returns:
            Tuple of (success, error_message)
        """
        essential_cookies = ["OptanonConsent", "trees"]

        # Check if we should skip cookie check for this action
        should_skip, skip_reason = self._should_skip_cookie_check(action_name)
        if should_skip:
            return True, skip_reason

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
    ) -> tuple[bool, Optional[str]]:
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

    def _check_csrf_token(self, api_manager) -> tuple[bool, Optional[str]]:
        """
        Check and retrieve CSRF token if needed.
        Uses smart caching to avoid repeated fetches.

        Args:
            api_manager: APIManager instance

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if not api_manager.csrf_token:
                logger.debug("CSRF token not available. Attempting to retrieve (once per session)...")
                csrf_token = api_manager.get_csrf_token()

                if not csrf_token:
                    # CSRF token failure is non-critical for some operations
                    logger.warning("Failed to retrieve CSRF token (non-critical).")
                    return True, None  # Continue anyway

                logger.debug("CSRF token retrieved and cached successfully.")
            # No need to log "already available" - reduces noise

            return True, None

        except Exception as e:
            error_msg = f"Exception checking CSRF token: {e}"
            logger.error(error_msg, exc_info=True)
            # CSRF token errors are non-critical
            return True, None

    def validate_session_cookies(
        self, browser_manager, required_cookies: list[str]
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

    def verify_login_status(self, api_manager, session_manager=None) -> bool:
        """
        Verify login status using multiple methods.

        Args:
            api_manager: APIManager instance
            session_manager: Optional SessionManager for cookie syncing

        Returns:
            bool: True if logged in, False otherwise
        """
        logger.debug("Verifying login status...")

        try:
            # Try API-based verification first (with cookie syncing if session_manager provided)
            api_login_status = api_manager.verify_api_login_status(session_manager)

            if api_login_status is True:
                logger.debug("Login verification successful (API method).")
                return True
            if api_login_status is False:
                logger.warning("Login verification failed (API method).")
                return False
            logger.error(
                "Login verification failed critically (API returned None)."
            )
            return False

        except Exception as e:
            logger.error(
                f"Unexpected error during login verification: {e}", exc_info=True
            )
            return False


# === Decomposed Helper Functions ===
def _test_session_validator_initialization() -> bool:
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


def _test_readiness_checks_success() -> bool:
    from unittest.mock import Mock, patch

    validator = SessionValidator()
    mock_browser = Mock()
    mock_api = Mock()
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


def _test_login_verification() -> bool:
    from unittest.mock import Mock

    validator = SessionValidator()
    mock_api = Mock()
    mock_api.verify_api_login_status.return_value = True
    result = validator.verify_login_status(mock_api)
    assert result is True, "Login verification should succeed with valid API response"
    mock_api.verify_api_login_status.assert_called_once()
    return True


def _test_invalid_browser_session() -> bool:
    from unittest.mock import Mock

    validator = SessionValidator()
    mock_browser = Mock()
    mock_browser.is_session_valid.return_value = False
    result = validator.validate_session_cookies(mock_browser, ["test_cookie"])
    assert result is False, "Should fail with invalid browser session"
    return True


def _test_login_verification_failure() -> bool:
    from unittest.mock import Mock

    validator = SessionValidator()
    mock_api = Mock()
    mock_api.verify_api_login_status.return_value = False
    result = validator.verify_login_status(mock_api)
    assert result is False, "Should fail when API reports not logged in"
    return True


def _test_full_validation_workflow() -> bool:
    from unittest.mock import Mock

    validator = SessionValidator()
    mock_browser = Mock()
    mock_api = Mock()
    mock_browser.is_session_valid.return_value = True
    mock_browser.get_cookies.return_value = True
    mock_api.verify_api_login_status.return_value = True
    cookie_result = validator.validate_session_cookies(mock_browser, ["session_cookie"])
    assert cookie_result is True, "Cookie validation should succeed"
    login_result = validator.verify_login_status(mock_api)
    assert login_result is True, "Login verification should succeed"
    return True


def _test_initialization_performance() -> bool:
    import time

    start_time = time.time()
    for _ in range(100):
        SessionValidator()
    end_time = time.time()
    total_time = end_time - start_time
    assert (
        total_time < 1.0
    ), f"100 initializations took {total_time:.3f}s, should be under 1s"
    return True


def _test_webdriver_exception_handling() -> bool:
    from unittest.mock import Mock, patch

    from selenium.common.exceptions import WebDriverException

    validator = SessionValidator()
    mock_browser = Mock()
    mock_api = Mock()
    mock_session = Mock()
    with patch.object(validator, "_check_login_and_attempt_relogin") as mock_login:
        mock_login.side_effect = WebDriverException("Browser crashed")
        mock_browser.is_session_valid.return_value = True
        result = validator.perform_readiness_checks(
            mock_browser, mock_api, mock_session, max_attempts=1
        )
        assert result is False, "Should fail when WebDriverException occurs"
    return True


def _test_general_exception_handling() -> bool:
    from unittest.mock import Mock

    validator = SessionValidator()
    mock_browser = Mock()
    mock_browser.is_session_valid.side_effect = Exception("Unexpected error")
    result = validator.validate_session_cookies(mock_browser, ["test_cookie"])
    assert result is False, "Should handle unexpected exceptions gracefully"
    return True


def session_validator_module_tests() -> bool:
    """
    Comprehensive test suite for session_validator.py (decomposed).
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Session Validation & Readiness Checks", "session_validator.py"
        )
        suite.start_suite()
        suite.run_test(
            "SessionValidator Initialization",
            _test_session_validator_initialization,
            "SessionValidator creates successfully with required attributes for session validation",
            "Instantiate SessionValidator and verify required attributes are properly initialized",
            "Test SessionValidator initialization and attribute setup",
        )
        suite.run_test(
            "Readiness Checks Success Flow",
            _test_readiness_checks_success,
            "All readiness checks pass when mocked dependencies return success",
            "Mock all internal validation methods to return success and verify overall result",
            "Test successful execution path of readiness checks with mocked dependencies",
        )
        suite.run_test(
            "Login Status Verification",
            _test_login_verification,
            "Login verification succeeds when API reports user is logged in",
            "Mock API to return successful login status and verify verification result",
            "Test login status verification with mocked API response",
        )
        suite.run_test(
            "Invalid Browser Session Handling",
            _test_invalid_browser_session,
            "Cookie validation fails gracefully when browser session is invalid",
            "Mock browser to return invalid session status and verify validation fails",
            "Test edge case handling for invalid browser sessions",
        )
        suite.run_test(
            "Login Verification Failure",
            _test_login_verification_failure,
            "Login verification fails when API reports user is not logged in",
            "Mock API to return failed login status and verify verification fails",
            "Test login verification failure handling",
        )
        suite.run_test(
            "Full Validation Workflow Integration",
            _test_full_validation_workflow,
            "Complete validation workflow succeeds when all components work together",
            "Test both cookie validation and login verification in sequence with mocked success responses",
            "Test integration of cookie validation and login verification workflows",
        )
        suite.run_test(
            "Initialization Performance",
            _test_initialization_performance,
            "100 SessionValidator initializations complete in under 1 second",
            "Create 100 SessionValidator instances and measure total time",
            "Test performance of SessionValidator initialization",
        )
        suite.run_test(
            "WebDriver Exception Handling",
            _test_webdriver_exception_handling,
            "Readiness checks fail gracefully when WebDriver exceptions occur",
            "Mock WebDriverException during login check and verify graceful failure",
            "Test error handling for WebDriver exceptions during validation",
        )
        suite.run_test(
            "General Exception Handling",
            _test_general_exception_handling,
            "Cookie validation handles unexpected exceptions gracefully",
            "Mock browser to throw unexpected exception and verify graceful failure",
            "Test error handling for general exceptions during validation",
        )
        return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(session_validator_module_tests)


if __name__ == "__main__":
    run_comprehensive_tests()
