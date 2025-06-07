"""
Session Validator - Handles session validation and readiness checks.

This module extracts session validation functionality from the monolithic
SessionManager class to provide a clean separation of concerns.
"""

import logging
from typing import Optional, Tuple, List
from datetime import datetime, timezone

from selenium.common.exceptions import WebDriverException

from config import config_instance

logger = logging.getLogger(__name__)


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
            if not current_url or not current_url.startswith(config_instance.BASE_URL):
                logger.warning(f"Not on Ancestry domain. Navigating to base URL...")

                # Import nav_to_page here to avoid circular imports
                from utils import nav_to_page

                nav_success = nav_to_page(
                    browser_manager.driver,
                    config_instance.BASE_URL,
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
        if not browser_manager.is_session_valid():
            logger.error("Cannot validate cookies: Browser session invalid.")
            return False

        try:
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


def run_comprehensive_tests():
    """
    Run comprehensive tests for SessionValidator functionality.
    Tests cover: Initialization, Core Functionality, Edge Cases, Integration, Performance, and Error Handling.
    """
    import sys
    import os
    import time
    from unittest.mock import Mock, MagicMock, patch

    # Suppress logging during tests
    import logging

    logging.getLogger().setLevel(logging.CRITICAL)

    print("=" * 70)
    print("SESSION VALIDATOR - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    def run_test(test_name, test_func):
        nonlocal total_tests, passed_tests
        total_tests += 1
        print(f"\n[TEST {total_tests:02d}] {test_name}")
        print("-" * 50)
        try:
            test_func()
            print("✓ PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            failed_tests.append(f"{test_name}: {str(e)}")

    # =============================================================================
    # INITIALIZATION TESTS
    # =============================================================================
    print("\n" + "=" * 50)
    print("INITIALIZATION TESTS")
    print("=" * 50)

    def test_session_validator_initialization():
        """Test SessionValidator initialization."""
        validator = SessionValidator()
        assert hasattr(
            validator, "last_js_error_check"
        ), "Missing last_js_error_check attribute"
        assert (
            validator.last_js_error_check is not None
        ), "last_js_error_check not initialized"
        print("SessionValidator initialized successfully")

    def test_attributes_setup():
        """Test that all required attributes are properly set up."""
        validator = SessionValidator()
        assert isinstance(
            validator.last_js_error_check, datetime
        ), "last_js_error_check should be datetime"
        print("All attributes properly initialized")

    run_test("SessionValidator Initialization", test_session_validator_initialization)
    run_test("Attributes Setup", test_attributes_setup)

    # =============================================================================
    # CORE FUNCTIONALITY TESTS
    # =============================================================================
    print("\n" + "=" * 50)
    print("CORE FUNCTIONALITY TESTS")
    print("=" * 50)

    def test_readiness_checks_success():
        """Test successful readiness checks flow."""
        validator = SessionValidator()

        # Mock dependencies
        mock_browser = Mock()
        mock_api = Mock()

        # Mock all the internal check methods to return success
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
            print("Readiness checks completed successfully")

    def test_login_verification():
        """Test login status verification."""
        validator = SessionValidator()
        mock_api = Mock()
        mock_api.verify_api_login_status.return_value = True

        result = validator.verify_login_status(mock_api)
        assert result is True, "Login verification should succeed"
        mock_api.verify_api_login_status.assert_called_once()
        print("Login verification working correctly")

    def test_cookie_validation():
        """Test session cookie validation."""
        validator = SessionValidator()
        mock_browser = Mock()
        mock_browser.is_session_valid.return_value = True
        mock_browser.get_cookies.return_value = True

        result = validator.validate_session_cookies(mock_browser, ["test_cookie"])
        assert result is True, "Cookie validation should succeed with valid session"
        print("Cookie validation working correctly")

    run_test("Readiness Checks Success Flow", test_readiness_checks_success)
    run_test("Login Verification", test_login_verification)
    run_test("Cookie Validation", test_cookie_validation)

    # =============================================================================
    # EDGE CASES TESTS
    # =============================================================================
    print("\n" + "=" * 50)
    print("EDGE CASES TESTS")
    print("=" * 50)

    def test_invalid_browser_session():
        """Test handling of invalid browser session."""
        validator = SessionValidator()
        mock_browser = Mock()
        mock_browser.is_session_valid.return_value = False

        result = validator.validate_session_cookies(mock_browser, ["test_cookie"])
        assert result is False, "Should fail with invalid browser session"
        print("Invalid browser session handled correctly")

    def test_missing_required_cookies():
        """Test handling of missing required cookies."""
        validator = SessionValidator()
        mock_browser = Mock()
        mock_browser.is_session_valid.return_value = True
        mock_browser.get_cookies.return_value = False

        result = validator.validate_session_cookies(mock_browser, ["missing_cookie"])
        assert result is False, "Should fail with missing cookies"
        print("Missing cookies handled correctly")

    def test_login_verification_failure():
        """Test login verification failure cases."""
        validator = SessionValidator()
        mock_api = Mock()
        mock_api.verify_api_login_status.return_value = False

        result = validator.verify_login_status(mock_api)
        assert result is False, "Should fail when API reports not logged in"
        print("Login verification failure handled correctly")

    run_test("Invalid Browser Session", test_invalid_browser_session)
    run_test("Missing Required Cookies", test_missing_required_cookies)
    run_test("Login Verification Failure", test_login_verification_failure)

    # =============================================================================
    # INTEGRATION TESTS
    # =============================================================================
    print("\n" + "=" * 50)
    print("INTEGRATION TESTS")
    print("=" * 50)

    def test_readiness_checks_with_retry():
        """Test readiness checks with retry mechanism."""
        validator = SessionValidator()
        mock_browser = Mock()
        mock_api = Mock()

        # First attempt fails, second succeeds
        call_count = 0

        def mock_login_check(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False, "First attempt fail"
            return True, None

        with patch.object(
            validator, "_check_login_and_attempt_relogin", side_effect=mock_login_check
        ), patch.object(
            validator, "_check_and_handle_url", return_value=True
        ), patch.object(
            validator, "_check_essential_cookies", return_value=(True, None)
        ), patch.object(
            validator, "_sync_cookies_to_requests", return_value=(True, None)
        ), patch.object(
            validator, "_check_csrf_token", return_value=(True, None)
        ), patch(
            "time.sleep"
        ):  # Mock sleep to speed up test

            result = validator.perform_readiness_checks(
                mock_browser, mock_api, max_attempts=2
            )
            assert result is True, "Should succeed on retry"
            assert call_count == 2, "Should have attempted twice"
            print("Retry mechanism working correctly")

    def test_full_validation_workflow():
        """Test complete validation workflow integration."""
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

        print("Full validation workflow completed successfully")

    run_test("Readiness Checks with Retry", test_readiness_checks_with_retry)
    run_test("Full Validation Workflow", test_full_validation_workflow)

    # =============================================================================
    # PERFORMANCE TESTS
    # =============================================================================
    print("\n" + "=" * 50)
    print("PERFORMANCE TESTS")
    print("=" * 50)

    def test_initialization_performance():
        """Test SessionValidator initialization performance."""
        start_time = time.time()
        for _ in range(100):
            validator = SessionValidator()
        end_time = time.time()

        total_time = end_time - start_time
        assert (
            total_time < 1.0
        ), f"100 initializations took {total_time:.3f}s, should be under 1s"
        print(f"100 initializations completed in {total_time:.3f}s")

    def test_validation_method_performance():
        """Test validation method call performance."""
        validator = SessionValidator()
        mock_browser = Mock()
        mock_browser.is_session_valid.return_value = True
        mock_browser.get_cookies.return_value = True

        start_time = time.time()
        for _ in range(50):
            validator.validate_session_cookies(mock_browser, ["test_cookie"])
        end_time = time.time()

        total_time = end_time - start_time
        assert (
            total_time < 0.5
        ), f"50 validations took {total_time:.3f}s, should be under 0.5s"
        print(f"50 cookie validations completed in {total_time:.3f}s")

    run_test("Initialization Performance", test_initialization_performance)
    run_test("Validation Method Performance", test_validation_method_performance)

    # =============================================================================
    # ERROR HANDLING TESTS
    # =============================================================================
    print("\n" + "=" * 50)
    print("ERROR HANDLING TESTS")
    print("=" * 50)

    def test_webdriver_exception_handling():
        """Test handling of WebDriver exceptions."""
        validator = SessionValidator()
        mock_browser = Mock()
        mock_api = Mock()

        # Mock WebDriverException during readiness checks
        with patch.object(validator, "_check_login_and_attempt_relogin") as mock_login:
            mock_login.side_effect = WebDriverException("Browser crashed")
            mock_browser.is_session_valid.return_value = True  # Session still valid

            result = validator.perform_readiness_checks(
                mock_browser, mock_api, max_attempts=1
            )
            assert result is False, "Should fail when WebDriverException occurs"
            print("WebDriverException handled correctly")

    def test_general_exception_handling():
        """Test handling of general exceptions."""
        validator = SessionValidator()
        mock_browser = Mock()
        mock_browser.is_session_valid.side_effect = Exception("Unexpected error")

        result = validator.validate_session_cookies(mock_browser, ["test_cookie"])
        assert result is False, "Should handle unexpected exceptions gracefully"
        print("General exceptions handled correctly")

    def test_api_failure_handling():
        """Test handling of API failures."""
        validator = SessionValidator()
        mock_api = Mock()
        mock_api.verify_api_login_status.side_effect = Exception("API error")

        result = validator.verify_login_status(mock_api)
        assert result is False, "Should handle API failures gracefully"
        print("API failures handled correctly")

    run_test("WebDriver Exception Handling", test_webdriver_exception_handling)
    run_test("General Exception Handling", test_general_exception_handling)
    run_test("API Failure Handling", test_api_failure_handling)

    # =============================================================================
    # TEST SUMMARY
    # =============================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print(f"\nFailed Tests:")
        for failure in failed_tests:
            print(f"  ✗ {failure}")

    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    if success_rate >= 90:
        print("🎉 EXCELLENT: Session validation functionality is working well!")
    elif success_rate >= 70:
        print("✅ GOOD: Session validation functionality is mostly working.")
    else:
        print("⚠️  NEEDS ATTENTION: Session validation functionality needs improvement.")

    print("=" * 70)
    return success_rate >= 90


if __name__ == "__main__":
    run_comprehensive_tests()
