#!/usr/bin/env python3
from __future__ import annotations

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

import logging

logger = logging.getLogger(__name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

# === THIRD-PARTY IMPORTS ===
from selenium.common.exceptions import WebDriverException

# === LOCAL IMPORTS ===
from config import config_schema

if TYPE_CHECKING:  # Import only for static type checking to avoid circular deps
    from core.session_manager import SessionManager
else:  # pragma: no cover - runtime fallback used only for type hints
    SessionManager = Any

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
        browser_manager: Any,
        api_manager: Any,
        session_manager: Any,
        action_name: Optional[str],
        skip_csrf: bool,
        attempt: int,
    ) -> tuple[bool, str]:
        """Perform all readiness checks. Returns (success, error_message)."""
        ok, error = self._verify_login_and_url(browser_manager, session_manager, attempt)
        if not ok:
            return False, error

        ok, error = self._verify_cookies_and_sync(browser_manager, api_manager, session_manager, action_name)
        if not ok:
            return False, error

        ok, error = self._verify_csrf_if_needed(api_manager, skip_csrf)
        if not ok:
            return False, error

        return True, ""

    def _verify_login_and_url(self, browser_manager: Any, session_manager: Any, attempt: int) -> tuple[bool, str]:
        """Ensure login is valid and URL state is acceptable."""
        login_success, login_error = self._check_login_and_attempt_relogin(browser_manager, session_manager, attempt)
        if not login_success:
            return False, login_error or "Login validation failed"

        if not self._check_and_handle_url(browser_manager):
            logger.error("URL check/handling failed.")
            return False, "URL check/handling failed"

        logger.debug("URL check/handling OK.")
        return True, ""

    def _verify_cookies_and_sync(
        self,
        browser_manager: Any,
        api_manager: Any,
        session_manager: Any,
        action_name: Optional[str],
    ) -> tuple[bool, str]:
        """Validate browser cookies and synchronize them to API sessions."""
        cookies_success, cookies_error = self._check_essential_cookies(browser_manager, action_name)
        if not cookies_success:
            return False, cookies_error or "Essential cookies missing"

        sync_success, sync_error = self._sync_cookies_to_requests(
            browser_manager, api_manager, session_manager=session_manager
        )
        if not sync_success:
            return False, sync_error or "Cookie synchronization failed"

        return True, ""

    def _verify_csrf_if_needed(self, api_manager: Any, skip_csrf: bool) -> tuple[bool, str]:
        """Optionally validate CSRF tokens when required."""
        if skip_csrf:
            logger.debug("Skipping CSRF token check as requested")
            return True, ""

        csrf_success, csrf_error = self._check_csrf_token(api_manager)
        if not csrf_success:
            return False, csrf_error or "CSRF validation failed"

        # Enforce token presence if not skipping
        if not api_manager.csrf_token:
            logger.error("CSRF token required but retrieval failed.")
            return False, "CSRF token retrieval failed"

        return True, ""

    @staticmethod
    def _handle_check_exception(exception: Exception, attempt: int, browser_manager: Any) -> tuple[bool, str]:
        """Handle exceptions during checks. Returns (should_abort, error_message)."""
        if isinstance(exception, WebDriverException):
            logger.error(
                f"WebDriverException during readiness check attempt {attempt}: {exception}",
                exc_info=False,
            )
            error_msg = f"WebDriverException: {exception}"

            if not browser_manager.is_session_valid():
                logger.error("Session invalid during readiness check. Aborting checks.")
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
        browser_manager: Any,
        api_manager: Any,
        session_manager: Any,
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
        logger.debug(f"Starting readiness checks for: {action_name or 'Unknown Action'}")
        last_check_error = "Unknown error"

        for attempt in range(1, max_attempts + 1):
            logger.debug(f"Readiness check attempt {attempt} of {max_attempts}")

            try:
                # Perform all checks
                success, error = self._perform_all_checks(
                    browser_manager, api_manager, session_manager, action_name, skip_csrf, attempt
                )

                if success:
                    logger.debug(f"Readiness checks PASSED on attempt {attempt}.")
                    return True

                last_check_error = error

            except Exception as exc:
                should_abort, error_msg = self._handle_check_exception(exc, attempt, browser_manager)
                last_check_error = error_msg

                if should_abort:
                    return False

            # Wait before next attempt (except on last attempt)
            if attempt < max_attempts:
                import time

                time.sleep(5)  # Increased from 2 to 5 seconds for better stability

        logger.error(f"All {max_attempts} readiness check attempts failed. Last Error: {last_check_error}")
        return False

    def _check_login_and_attempt_relogin(
        self, browser_manager: Any, session_manager: Any, attempt: int
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
            from core.utils import login_status, nav_to_page

            logger.debug(f"Checking login status (attempt {attempt})...")
            # Use the session manager parameter directly
            if not session_manager:
                logger.error("No session manager provided for login status check")
                return False, "No session manager provided"

            driver = getattr(browser_manager, "driver", None)
            base_url = config_schema.api.base_url or "https://www.ancestry.com"

            self._ensure_on_base_url(driver, base_url, session_manager, nav_to_page)

            # CRITICAL FIX: Sync cookies BEFORE checking login status
            # Browser may have valid cookies, but requests session doesn't yet
            self._sync_cookies_for_login(session_manager)

            login_result = login_status(session_manager, disable_ui_fallback=True)  # Use API-only check

            return self._process_login_result(login_result, browser_manager, session_manager)

        except Exception as e:
            error_msg = f"Exception during login check: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    @staticmethod
    def _ensure_on_base_url(
        driver: Any,
        base_url: str,
        session_manager: Any,
        nav_to_page: Callable[..., bool],
    ) -> None:
        """Ensure the browser is on the correct base URL before login checks."""
        if not driver:
            return

        try:
            current_url = driver.current_url
        except Exception:
            current_url = ""

        if current_url and current_url.startswith(base_url):
            return

        logger.debug("Pre-auth navigation to base URL to refresh cookie context...")
        nav_success = nav_to_page(
            driver,
            base_url,
            selector="body",
            session_manager=session_manager,
        )
        if not nav_success:
            logger.warning("Pre-auth navigation failed; continuing with existing session state.")

    @staticmethod
    def _sync_cookies_for_login(session_manager: SessionManager) -> None:
        """Force a cookie sync before login verification when available."""

        try:
            logger.debug("Pre-syncing cookies from browser before login check (forced)...")
            session_manager.sync_cookies_to_requests(force=True)
        except Exception as exc:  # Pragmatic guard to keep login flow resilient
            logger.debug(f"Cookie pre-sync failed (continuing with existing state): {exc}")

    def _process_login_result(
        self, login_result: Any, browser_manager: Any, session_manager: Any
    ) -> tuple[bool, Optional[str]]:
        """Interpret login_status result and perform relogin if necessary."""
        if login_result is True:
            logger.debug("Login status check: User is logged in.")
            return True, None

        if login_result is False:
            logger.warning("Login status check: User is NOT logged in. Attempting relogin...")
            if self._attempt_relogin(browser_manager, session_manager):
                return True, None
            error_msg = "Relogin failed"
            logger.error(error_msg)
            return False, error_msg

        error_msg = "Login status check returned None (critical failure)"
        logger.error(error_msg)
        return False, error_msg

    @staticmethod
    def _attempt_relogin(_browser_manager: Any, session_manager: Any) -> bool:
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
            from core.utils import log_in

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

    @staticmethod
    def _check_and_handle_url(browser_manager: Any) -> bool:
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
                from core.utils import nav_to_page

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

    @staticmethod
    def _should_skip_cookie_check(action_name: Optional[str]) -> tuple[bool, Optional[str]]:
        """Determine if cookie check should be skipped for this action."""
        if not action_name:
            return False, None

        # Map of action patterns to skip reasons
        skip_patterns = {
            "coord": "Action 6 - cookies will be available after navigation",
            "gather_dna": "Action 6 - cookies will be available after navigation",
            "action6b": "Action 6B - cookies will be available after navigation",
            "srch_inbox": "Action 7 - API login verification is sufficient",
            "search_inbox": "Action 7 - API login verification is sufficient",
            "run_daily_review_first_loop_action": "Action 15 - starts with Action 7; API login verification is sufficient",
            "send_messages": "Action 8 - API login verification is sufficient",
            "process_productive": "Action 9 - API login verification is sufficient",
            "main": "Action 10 - Local file operation",
            "gedcom": "Action 10 - Local file operation",
            # Action 10 side-by-side compare (GEDCOM vs API): do not require 'trees' cookie
            "side-by-side": "Action 10 compare - cookie check skipped (no 'trees' required)",
            "side_by_side": "Action 10 compare - cookie check skipped (no 'trees' required)",
            "run_side_by_side_search_wrapper": "Action 10 compare - cookie check skipped (no 'trees' required)",
            "run_merged_search_wrapper": "Action 10 compare - cookie check skipped (no 'trees' required)",
            "run_gedcom_then_api_fallback": "Action 10 compare - cookie check skipped (no 'trees' required)",
            "action10_api_test": "Action 10 API Test - cookie check skipped (no 'trees' required)",
            "run_action11": "API-based operation",
            "api_report": "API-based operation",
            "refresh": "Browser refresh verification - deferring to later checks",
            # Action 13 - shared match scraping uses API calls after navigation
            "fetch_shared_matches": "Action 13 - cookies validated after API call",
            "shared_match": "Action 13 - cookies validated after API call",
        }

        action_lower = action_name.lower()
        for pattern, reason in skip_patterns.items():
            if pattern in action_lower:
                logger.debug(f"Skipping essential cookies check (expected) for action '{action_name}': {reason}")
                return True, reason

        return False, None

    def _check_essential_cookies(
        self,
        browser_manager: Any,
        action_name: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
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
                # Attempt recovery: Refresh page to see if cookies are set
                logger.warning(f"Essential cookies missing: {essential_cookies}. Attempting refresh...")

                if hasattr(browser_manager, "driver") and browser_manager.driver:
                    try:
                        browser_manager.driver.refresh()
                        import time

                        time.sleep(5)  # Wait for reload
                    except Exception as refresh_err:
                        logger.warning(f"Failed to refresh page: {refresh_err}")

                # Check again
                if not browser_manager.get_cookies(essential_cookies):
                    error_msg = f"Essential cookies not found after refresh: {essential_cookies}"
                    logger.warning(error_msg)
                    return False, error_msg

            logger.debug("Essential cookies check passed.")
            return True, None

        except Exception as e:
            error_msg = f"Exception checking essential cookies: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    @staticmethod
    def _sync_cookies_to_requests(
        browser_manager: Any,
        api_manager: Any,
        session_manager: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Sync cookies from browser to API requests session.

        Args:
            browser_manager: BrowserManager instance
            api_manager: APIManager instance
            session_manager: Optional SessionManager instance for recovery support

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # CRITICAL FIX: Pass session_manager to enable automatic recovery on cookie sync failures
            sync_success = api_manager.sync_cookies_from_browser(browser_manager, session_manager=session_manager)
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

    @staticmethod
    def _check_csrf_token(api_manager: Any) -> tuple[bool, Optional[str]]:
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

    @staticmethod
    def validate_session_cookies(browser_manager: Any, required_cookies: list[str]) -> bool:
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

    @staticmethod
    def verify_login_status(
        api_manager: Any,
        session_manager: Optional[Any] = None,
    ) -> bool:
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
            logger.error("Login verification failed critically (API returned None).")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during login verification: {e}", exc_info=True)
            return False


# === Decomposed Helper Functions ===
def _test_session_validator_initialization() -> bool:
    validator = SessionValidator()
    assert validator is not None, "SessionValidator should initialize"
    assert hasattr(validator, "last_js_error_check"), "Should have last_js_error_check attribute"
    assert validator.last_js_error_check is not None, "last_js_error_check should be initialized"
    from datetime import datetime

    assert isinstance(validator.last_js_error_check, datetime), "last_js_error_check should be datetime"
    return True


def _test_readiness_checks_success() -> bool:
    from unittest.mock import Mock, patch

    validator = SessionValidator()
    mock_browser = Mock()
    mock_api = Mock()
    with (
        patch.object(validator, "_check_login_and_attempt_relogin", return_value=(True, None)),
        patch.object(validator, "_check_and_handle_url", return_value=True),
        patch.object(validator, "_check_essential_cookies", return_value=(True, None)),
        patch.object(validator, "_sync_cookies_to_requests", return_value=(True, None)),
        patch.object(validator, "_check_csrf_token", return_value=(True, None)),
    ):
        result = validator.perform_readiness_checks(mock_browser, mock_api, "test_action")
        assert result is True, "Readiness checks should succeed when all sub-checks pass"
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
    assert total_time < 1.0, f"100 initializations took {total_time:.3f}s, should be under 1s"
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
        result = validator.perform_readiness_checks(mock_browser, mock_api, mock_session, max_attempts=1)
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


def _test_should_skip_cookie_check_action6() -> bool:
    """Test that Action 6 (gather_dna_matches) skips cookie check."""
    validator = SessionValidator()

    # Test with 'gather_dna_matches' action name
    should_skip, reason = validator._should_skip_cookie_check("gather_dna_matches")
    assert should_skip is True, "Should skip cookie check for gather_dna_matches"
    assert reason is not None and "Action 6" in reason, f"Reason should mention Action 6, got: {reason}"

    # Test with 'coord' action name (legacy)
    should_skip, reason = validator._should_skip_cookie_check("coord")
    assert should_skip is True, "Should skip cookie check for coord"
    assert reason is not None and "Action 6" in reason, f"Reason should mention Action 6, got: {reason}"

    return True


def _test_should_skip_cookie_check_action7() -> bool:
    """Test that Action 7 (search_inbox) skips cookie check."""
    validator = SessionValidator()

    # Test with 'search_inbox' action name
    should_skip, reason = validator._should_skip_cookie_check("search_inbox")
    assert should_skip is True, "Should skip cookie check for search_inbox"
    assert reason is not None and "Action 7" in reason, f"Reason should mention Action 7, got: {reason}"

    # Test with 'srch_inbox' action name (legacy)
    should_skip, reason = validator._should_skip_cookie_check("srch_inbox")
    assert should_skip is True, "Should skip cookie check for srch_inbox"
    assert reason is not None and "Action 7" in reason, f"Reason should mention Action 7, got: {reason}"

    # Test with Action 15 daily loop action name
    should_skip, reason = validator._should_skip_cookie_check("run_daily_review_first_loop_action")
    assert should_skip is True, "Should skip cookie check for run_daily_review_first_loop_action"
    assert reason is not None and "Action 15" in reason, f"Reason should mention Action 15, got: {reason}"

    return True


def _test_should_skip_cookie_check_action8() -> bool:
    """Test that Action 8 (send_messages) skips cookie check."""
    validator = SessionValidator()

    should_skip, reason = validator._should_skip_cookie_check("send_messages")
    assert should_skip is True, "Should skip cookie check for send_messages"
    assert reason is not None and "Action 8" in reason, f"Reason should mention Action 8, got: {reason}"

    return True


def _test_should_skip_cookie_check_action9() -> bool:
    """Test that Action 9 (process_productive) skips cookie check."""
    validator = SessionValidator()

    should_skip, reason = validator._should_skip_cookie_check("process_productive")
    assert should_skip is True, "Should skip cookie check for process_productive"
    assert reason is not None and "Action 9" in reason, f"Reason should mention Action 9, got: {reason}"

    return True


def _test_should_skip_cookie_check_no_action() -> bool:
    """Test that None action name does not skip cookie check."""
    validator = SessionValidator()

    should_skip, reason = validator._should_skip_cookie_check(None)
    assert should_skip is False, "Should not skip cookie check for None action"
    assert reason is None, f"Reason should be None, got: {reason}"

    return True


def _test_should_skip_cookie_check_unknown_action() -> bool:
    """Test that unknown action names do not skip cookie check."""
    validator = SessionValidator()

    should_skip, reason = validator._should_skip_cookie_check("unknown_action")
    assert should_skip is False, "Should not skip cookie check for unknown action"
    assert reason is None, f"Reason should be None, got: {reason}"

    return True


def _test_should_skip_cookie_check_case_insensitive() -> bool:
    """Test that action name matching is case-insensitive."""
    validator = SessionValidator()

    # Test uppercase
    should_skip, _ = validator._should_skip_cookie_check("GATHER_DNA_MATCHES")
    assert should_skip is True, "Should skip cookie check for uppercase action name"

    # Test mixed case
    should_skip, _ = validator._should_skip_cookie_check("Gather_dna_Matches")
    assert should_skip is True, "Should skip cookie check for mixed case action name"

    # Test with setup suffix (as seen in logs)
    should_skip, _ = validator._should_skip_cookie_check("gather_dna_matches - Setup")
    assert should_skip is True, "Should skip cookie check for action name with suffix"

    return True


def session_validator_module_tests() -> bool:
    """
    Comprehensive test suite for session_validator.py (decomposed).
    """
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Session Validation & Readiness Checks", "session_validator.py")
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
        suite.run_test(
            "Skip Cookie Check - Action 6",
            _test_should_skip_cookie_check_action6,
            "Action 6 (gather_dna_matches) correctly skips cookie check",
            "Test that both 'gather_dna_matches' and 'coord' action names skip cookie check",
            "Test cookie check skip logic for Action 6 with multiple action name patterns",
        )
        suite.run_test(
            "Skip Cookie Check - Action 7",
            _test_should_skip_cookie_check_action7,
            "Action 7 (search_inbox) correctly skips cookie check",
            "Test that both 'search_inbox' and 'srch_inbox' action names skip cookie check",
            "Test cookie check skip logic for Action 7 with multiple action name patterns",
        )
        suite.run_test(
            "Skip Cookie Check - Action 8",
            _test_should_skip_cookie_check_action8,
            "Action 8 (send_messages) correctly skips cookie check",
            "Test that 'send_messages' action name skips cookie check",
            "Test cookie check skip logic for Action 8",
        )
        suite.run_test(
            "Skip Cookie Check - Action 9",
            _test_should_skip_cookie_check_action9,
            "Action 9 (process_productive) correctly skips cookie check",
            "Test that 'process_productive' action name skips cookie check",
            "Test cookie check skip logic for Action 9",
        )
        suite.run_test(
            "Skip Cookie Check - None Action",
            _test_should_skip_cookie_check_no_action,
            "None action name does not skip cookie check",
            "Test that None action name returns False for skip",
            "Test edge case handling for None action name",
        )
        suite.run_test(
            "Skip Cookie Check - Unknown Action",
            _test_should_skip_cookie_check_unknown_action,
            "Unknown action names do not skip cookie check",
            "Test that unknown action names return False for skip",
            "Test edge case handling for unknown action names",
        )
        suite.run_test(
            "Skip Cookie Check - Case Insensitivity",
            _test_should_skip_cookie_check_case_insensitive,
            "Action name matching is case-insensitive and handles suffixes",
            "Test uppercase, mixed case, and action names with suffixes",
            "Test case-insensitive matching and suffix handling in action names",
        )
        return suite.finish_suite()


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(session_validator_module_tests)


if __name__ == "__main__":
    run_comprehensive_tests()
