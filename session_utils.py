#!/usr/bin/env python3

"""
session_utils.py - Centralized Session Authentication Utility

Provides a unified interface for session authentication that can be used by:
1. main.py - for action execution (optional - main.py already has good session management)
2. Individual action scripts - for their test functions (PRIMARY USE CASE)
3. run_all_tests.py - for test execution

This eliminates code duplication across action test functions and ensures consistent
authentication behavior. Each action script currently has its own _ensure_session_for_tests()
function - this module provides a shared implementation.

Benefits:
- Reduces code duplication (6+ action scripts have nearly identical session setup code)
- Consistent authentication behavior across all tests
- Centralized timeout and error handling
- Session caching for faster test execution
- Easy to update authentication logic in one place
"""

# === STANDARD LIBRARY IMPORTS ===
import threading
from typing import Optional

# === LOCAL IMPORTS ===
from core.session_manager import SessionManager
from logging_config import setup_logging
from utils import log_in, login_status

# === MODULE LOGGER ===
logger = setup_logging(log_level="INFO")

# === GLOBAL SESSION CACHE ===
# Reusable session for tests and main.py execution
_cached_session_manager: Optional[SessionManager] = None
_cached_session_uuid: Optional[str] = None


def create_and_start_session(action_name: str = "Session Setup", timeout: int = 120) -> SessionManager:
    """
    Create and start a new session manager with browser.

    Args:
        action_name: Name of the action for logging
        timeout: Timeout in seconds for session start (default: 120)

    Returns:
        SessionManager: Initialized and started session manager

    Raises:
        AssertionError: If session fails to start within timeout
    """
    logger.info("=" * 80)
    logger.info(f"Creating new session for: {action_name}")
    logger.info("=" * 80)

    logger.info("Step 1: Creating SessionManager...")
    sm = SessionManager()
    logger.info("✅ SessionManager created")

    logger.info("Step 2: Configuring browser requirement...")
    sm.browser_manager.browser_needed = True
    logger.info("✅ Browser marked as needed")

    logger.info("Step 3: Starting session (database + browser)...")

    result: dict[str, bool | str | None] = {"started": False, "error": None}

    def start_session() -> None:
        try:
            result["started"] = sm.start_sess(action_name)
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            result["error"] = str(e)

    # Run session start in a thread with timeout
    thread = threading.Thread(target=start_session, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.error(f"Session start timed out after {timeout} seconds")
        sm.close_sess(keep_db=False)
        raise AssertionError("Session start timed out - website may be down or network issue")

    if result["error"]:
        sm.close_sess(keep_db=False)
        raise AssertionError(f"Session start failed: {result['error']}")

    if not result["started"]:
        sm.close_sess(keep_db=False)
        raise AssertionError("Failed to start session - browser initialization failed")

    logger.info("✅ Session started successfully")
    return sm


def authenticate_session(sm: SessionManager, timeout: int = 60) -> None:
    """
    Authenticate the session by checking login status and logging in if needed.

    Args:
        sm: SessionManager to authenticate
        timeout: Timeout in seconds for authentication (default: 60)

    Raises:
        AssertionError: If authentication fails within timeout
    """
    logger.info("Step 4: Checking login status...")

    result: dict[str, bool | str | None] = {"logged_in": False, "error": None}

    def check_and_login() -> None:
        try:
            status = login_status(sm.driver)
            if status:
                logger.info("✅ Already logged in")
                result["logged_in"] = True
            else:
                logger.info("Not logged in. Attempting login...")
                login_success = log_in(sm.driver)
                if login_success:
                    logger.info("✅ Login successful")
                    result["logged_in"] = True
                else:
                    result["error"] = "Login failed"
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            result["error"] = str(e)

    # Run authentication in a thread with timeout
    thread = threading.Thread(target=check_and_login, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.error(f"Authentication timed out after {timeout} seconds")
        raise AssertionError("Authentication timed out")

    if result["error"]:
        raise AssertionError(f"Authentication failed: {result['error']}")

    if not result["logged_in"]:
        raise AssertionError("Authentication failed - not logged in")


def validate_session_ready(sm: SessionManager, action_name: str = "Session Validation", skip_csrf: bool = True) -> None:
    """
    Validate that session is ready with all identifiers.

    Args:
        sm: SessionManager to validate
        action_name: Name of the action for logging
        skip_csrf: Whether to skip CSRF token check (default: True)

    Raises:
        AssertionError: If session is not ready or UUID is not available
    """
    logger.info("Step 5: Ensuring session is ready...")
    ready = sm.ensure_session_ready(action_name, skip_csrf=skip_csrf)
    if not ready:
        sm.close_sess(keep_db=False)
        raise AssertionError("Session not ready - cookies/identifiers missing")
    logger.info("✅ Session ready")

    logger.info("Step 6: Verifying UUID is available...")
    if not sm.my_uuid:
        sm.close_sess(keep_db=False)
        raise AssertionError("UUID not available - session initialization incomplete")
    logger.info(f"✅ UUID available: {sm.my_uuid}")


def get_authenticated_session(
    action_name: str = "Session Setup",
    reuse_cached: bool = True,
    skip_csrf: bool = True,
    timeout: int = 120
) -> tuple[SessionManager, str]:
    """
    Get an authenticated session manager ready for use.

    This is the main entry point for getting a session. It will:
    1. Return cached session if available and reuse_cached=True
    2. Otherwise create new session, authenticate it, and validate it
    3. Cache the session for future reuse

    Args:
        action_name: Name of the action for logging
        reuse_cached: If True, reuse cached session if available (default: True)
        skip_csrf: Whether to skip CSRF token check (default: True)
        timeout: Timeout in seconds for session operations (default: 120)

    Returns:
        tuple[SessionManager, str]: (session_manager, uuid)

    Raises:
        AssertionError: If session setup fails
    """
    global _cached_session_manager, _cached_session_uuid

    # Return cached session if available and requested
    if reuse_cached and _cached_session_manager and _cached_session_uuid:
        logger.info("=" * 80)
        logger.info("Reusing cached authenticated session")
        logger.info(f"UUID: {_cached_session_uuid}")
        logger.info("=" * 80)
        return _cached_session_manager, _cached_session_uuid

    # Create new session
    sm = create_and_start_session(action_name, timeout)

    # Authenticate the session
    authenticate_session(sm, timeout=60)

    # Validate session is ready
    validate_session_ready(sm, action_name, skip_csrf)

    logger.info("=" * 80)
    logger.info("✅ Valid authenticated session established")
    logger.info(f"UUID: {sm.my_uuid}")
    logger.info("=" * 80)

    # Cache session for reuse
    _cached_session_manager = sm
    _cached_session_uuid = sm.my_uuid

    return sm, sm.my_uuid or ""


def clear_cached_session() -> None:
    """
    Clear the cached session.

    This should be called when you want to force creation of a new session
    on the next call to get_authenticated_session().
    """
    global _cached_session_manager, _cached_session_uuid

    if _cached_session_manager:
        logger.info("Clearing cached session")
        _cached_session_manager = None
        _cached_session_uuid = None


def close_cached_session(keep_db: bool = True) -> None:
    """
    Close and clear the cached session.

    Args:
        keep_db: Whether to keep database connections (default: True)
    """
    global _cached_session_manager, _cached_session_uuid

    if _cached_session_manager:
        logger.info("Closing cached session")
        _cached_session_manager.close_sess(keep_db=keep_db)
        _cached_session_manager = None
        _cached_session_uuid = None


# === COMPATIBILITY WRAPPERS FOR ACTION SCRIPTS ===
# These provide drop-in replacements for the _ensure_session_for_tests() functions
# that exist in action6, action7, action8, action9, action11


def ensure_session_for_tests(
    action_name: str = "Test Session",
    reuse_session: bool = True,
    skip_csrf: bool = True,
    timeout: int = 300
) -> tuple[SessionManager, str]:
    """
    Drop-in replacement for action script _ensure_session_for_tests() functions.

    This is a compatibility wrapper that matches the signature used by action scripts.

    Args:
        action_name: Name of the action for logging
        reuse_session: If True, reuse cached session if available (default: True)
        skip_csrf: Whether to skip CSRF token check (default: True)
        timeout: Timeout in seconds for session operations (default: 300)

    Returns:
        tuple[SessionManager, str]: (session_manager, uuid)

    Example usage in action scripts:
        # Old code (73 lines):
        # def _ensure_session_for_api_tests(reuse_session: bool = True) -> tuple[SessionManager, str]:
        #     ... 73 lines of session setup code ...

        # New code (1 line):
        # from session_utils import ensure_session_for_tests as _ensure_session_for_api_tests
    """
    return get_authenticated_session(action_name, reuse_session, skip_csrf, timeout)


def ensure_session_for_tests_sm_only(
    action_name: str = "Test Session",
    reuse_session: bool = True,
    skip_csrf: bool = True,
    timeout: int = 300
) -> SessionManager:
    """
    Drop-in replacement for action7's _ensure_session_for_tests() which returns only SessionManager.

    Args:
        action_name: Name of the action for logging
        reuse_session: If True, reuse cached session if available (default: True)
        skip_csrf: Whether to skip CSRF token check (default: True)
        timeout: Timeout in seconds for session operations (default: 300)

    Returns:
        SessionManager: Authenticated session manager

    Example usage in action7:
        # Old code (88 lines):
        # def _ensure_session_for_tests(reuse_session: bool = True) -> SessionManager:
        #     ... 88 lines of session setup code ...

        # New code (1 line):
        # from session_utils import ensure_session_for_tests_sm_only as _ensure_session_for_tests
    """
    sm, _ = get_authenticated_session(action_name, reuse_session, skip_csrf, timeout)
    return sm


# ==============================================
# COMPREHENSIVE TEST SUITE
# ==============================================


def _test_create_and_start_session_success() -> bool:
    """Test successful session creation and start."""
    from unittest.mock import MagicMock, patch

    with patch('session_utils.SessionManager') as MockSessionManager:
        # Create mock session manager
        mock_sm = MagicMock()
        mock_sm.start_sess.return_value = True
        mock_sm.browser_manager.browser_needed = False
        MockSessionManager.return_value = mock_sm

        # Test session creation
        sm = create_and_start_session("Test Action", timeout=5)

        # Verify SessionManager was created
        assert MockSessionManager.called, "SessionManager should be instantiated"

        # Verify browser was marked as needed
        assert mock_sm.browser_manager.browser_needed is True, "Browser should be marked as needed"

        # Verify start_sess was called
        assert mock_sm.start_sess.called, "start_sess should be called"

        # Verify returned session manager
        assert sm == mock_sm, "Should return the session manager"

        logger.info("✅ Session creation and start successful")
        return True


def _test_create_and_start_session_timeout() -> bool:
    """Test session creation timeout handling."""
    import time
    from unittest.mock import MagicMock, patch

    with patch('session_utils.SessionManager') as MockSessionManager:
        # Create mock that hangs
        mock_sm = MagicMock()

        def slow_start(*_args: object, **_kwargs: object) -> bool:
            time.sleep(10)  # Longer than timeout
            return True

        mock_sm.start_sess.side_effect = slow_start
        mock_sm.browser_manager.browser_needed = False
        MockSessionManager.return_value = mock_sm

        # Test timeout
        try:
            create_and_start_session("Test Action", timeout=1)
            raise AssertionError("Should have raised AssertionError for timeout")
        except AssertionError as e:
            assert "timed out" in str(e).lower(), f"Error should mention timeout: {e}"
            logger.info("✅ Timeout handling works correctly")
            return True


def _test_create_and_start_session_failure() -> bool:
    """Test session creation failure handling."""
    from unittest.mock import MagicMock, patch

    with patch('session_utils.SessionManager') as MockSessionManager:
        # Create mock that fails
        mock_sm = MagicMock()
        mock_sm.start_sess.side_effect = Exception("Browser initialization failed")
        mock_sm.browser_manager.browser_needed = False
        MockSessionManager.return_value = mock_sm

        # Test failure
        try:
            create_and_start_session("Test Action", timeout=5)
            raise AssertionError("Should have raised AssertionError for failure")
        except AssertionError as e:
            assert "failed" in str(e).lower(), f"Error should mention failure: {e}"
            # Verify cleanup was called
            assert mock_sm.close_sess.called, "close_sess should be called on failure"
            logger.info("✅ Failure handling works correctly")
            return True


def _test_authenticate_session_already_logged_in() -> bool:
    """Test authentication when already logged in."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.driver = MagicMock()

    with patch('session_utils.login_status', return_value=True):
        # Test authentication
        authenticate_session(mock_sm, timeout=5)

        logger.info("✅ Already logged in detection works")
        return True


def _test_authenticate_session_login_required() -> bool:
    """Test authentication when login is required."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.driver = MagicMock()

    with patch('session_utils.login_status', return_value=False), \
         patch('session_utils.log_in', return_value=True):
        # Test authentication
        authenticate_session(mock_sm, timeout=5)

        logger.info("✅ Login process works correctly")
        return True


def _test_authenticate_session_login_failure() -> bool:
    """Test authentication when login fails."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.driver = MagicMock()

    with patch('session_utils.login_status', return_value=False), \
         patch('session_utils.log_in', return_value=False):
        # Test authentication failure
        try:
            authenticate_session(mock_sm, timeout=5)
            raise AssertionError("Should have raised AssertionError for login failure")
        except AssertionError as e:
            assert "failed" in str(e).lower(), f"Error should mention failure: {e}"
            logger.info("✅ Login failure handling works correctly")
            return True


def _test_authenticate_session_timeout() -> bool:
    """Test authentication timeout handling."""
    import time
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_sm.driver = MagicMock()

    def slow_login_check(*_args: object, **_kwargs: object) -> bool:
        time.sleep(10)  # Longer than timeout
        return True

    with patch('session_utils.login_status', side_effect=slow_login_check):
        # Test timeout
        try:
            authenticate_session(mock_sm, timeout=1)
            raise AssertionError("Should have raised AssertionError for timeout")
        except AssertionError as e:
            assert "timed out" in str(e).lower(), f"Error should mention timeout: {e}"
            logger.info("✅ Authentication timeout handling works correctly")
            return True


def _test_validate_session_ready_success() -> bool:
    """Test session validation when session is ready."""
    from unittest.mock import MagicMock

    mock_sm = MagicMock()
    mock_sm.ensure_session_ready.return_value = True
    mock_sm.my_uuid = "test-uuid-12345"

    # Test validation
    validate_session_ready(mock_sm, "Test Action", skip_csrf=True)

    # Verify ensure_session_ready was called
    assert mock_sm.ensure_session_ready.called, "ensure_session_ready should be called"

    logger.info("✅ Session validation works correctly")
    return True


def _test_validate_session_ready_not_ready() -> bool:
    """Test session validation when session is not ready."""
    from unittest.mock import MagicMock

    mock_sm = MagicMock()
    mock_sm.ensure_session_ready.return_value = False

    # Test validation failure
    try:
        validate_session_ready(mock_sm, "Test Action", skip_csrf=True)
        raise AssertionError("Should have raised AssertionError for not ready")
    except AssertionError as e:
        assert "not ready" in str(e).lower(), f"Error should mention not ready: {e}"
        # Verify cleanup was called
        assert mock_sm.close_sess.called, "close_sess should be called on failure"
        logger.info("✅ Not ready handling works correctly")
        return True


def _test_validate_session_ready_no_uuid() -> bool:
    """Test session validation when UUID is missing."""
    from unittest.mock import MagicMock

    mock_sm = MagicMock()
    mock_sm.ensure_session_ready.return_value = True
    mock_sm.my_uuid = None  # Missing UUID

    # Test validation failure
    try:
        validate_session_ready(mock_sm, "Test Action", skip_csrf=True)
        raise AssertionError("Should have raised AssertionError for missing UUID")
    except AssertionError as e:
        assert "uuid" in str(e).lower(), f"Error should mention UUID: {e}"
        # Verify cleanup was called
        assert mock_sm.close_sess.called, "close_sess should be called on failure"
        logger.info("✅ Missing UUID handling works correctly")
        return True


def _test_get_authenticated_session_new() -> bool:
    """Test getting a new authenticated session."""
    from unittest.mock import MagicMock, patch

    # Clear cache first
    clear_cached_session()

    with patch('session_utils.create_and_start_session') as mock_create, \
         patch('session_utils.authenticate_session') as mock_auth, \
         patch('session_utils.validate_session_ready') as mock_validate:

        # Setup mocks
        mock_sm = MagicMock()
        mock_sm.my_uuid = "test-uuid-12345"
        mock_create.return_value = mock_sm

        # Test getting session
        sm, uuid = get_authenticated_session("Test Action", reuse_cached=False)

        # Verify all steps were called
        assert mock_create.called, "create_and_start_session should be called"
        assert mock_auth.called, "authenticate_session should be called"
        assert mock_validate.called, "validate_session_ready should be called"

        # Verify return values
        assert sm == mock_sm, "Should return session manager"
        assert uuid == "test-uuid-12345", "Should return UUID"

        logger.info("✅ New session creation works correctly")
        return True


def _test_get_authenticated_session_cached() -> bool:
    """Test getting a cached authenticated session."""
    from unittest.mock import MagicMock, patch

    # Setup cache
    global _cached_session_manager, _cached_session_uuid
    mock_sm = MagicMock()
    mock_sm.my_uuid = "cached-uuid-12345"
    _cached_session_manager = mock_sm
    _cached_session_uuid = "cached-uuid-12345"

    with patch('session_utils.create_and_start_session') as mock_create:
        # Test getting cached session
        sm, uuid = get_authenticated_session("Test Action", reuse_cached=True)

        # Verify no new session was created
        assert not mock_create.called, "Should not create new session when cached"

        # Verify cached values returned
        assert sm == mock_sm, "Should return cached session manager"
        assert uuid == "cached-uuid-12345", "Should return cached UUID"

        logger.info("✅ Cached session reuse works correctly")

        # Clean up
        clear_cached_session()
        return True


def _test_clear_cached_session() -> bool:
    """Test clearing cached session."""
    from unittest.mock import MagicMock

    global _cached_session_manager, _cached_session_uuid

    # Setup cache
    mock_sm = MagicMock()
    _cached_session_manager = mock_sm
    _cached_session_uuid = "test-uuid"

    # Clear cache
    clear_cached_session()

    # Verify cache is cleared
    assert _cached_session_manager is None, "Cached session manager should be None"
    assert _cached_session_uuid is None, "Cached UUID should be None"

    logger.info("✅ Cache clearing works correctly")
    return True


def _test_close_cached_session() -> bool:
    """Test closing and clearing cached session."""
    from unittest.mock import MagicMock

    global _cached_session_manager, _cached_session_uuid

    # Setup cache
    mock_sm = MagicMock()
    _cached_session_manager = mock_sm
    _cached_session_uuid = "test-uuid"

    # Close cache
    close_cached_session(keep_db=True)

    # Verify close_sess was called
    assert mock_sm.close_sess.called, "close_sess should be called"
    mock_sm.close_sess.assert_called_with(keep_db=True)

    # Verify cache is cleared
    assert _cached_session_manager is None, "Cached session manager should be None"
    assert _cached_session_uuid is None, "Cached UUID should be None"

    logger.info("✅ Cached session closing works correctly")
    return True


def _test_ensure_session_for_tests_wrapper() -> bool:
    """Test ensure_session_for_tests compatibility wrapper."""
    from unittest.mock import patch

    # Clear cache first
    clear_cached_session()

    with patch('session_utils.get_authenticated_session') as mock_get:
        # Setup mock
        from unittest.mock import MagicMock
        mock_sm = MagicMock()
        mock_sm.my_uuid = "wrapper-test-uuid"
        mock_get.return_value = (mock_sm, "wrapper-test-uuid")

        # Test wrapper
        sm, uuid = ensure_session_for_tests("Test Action", reuse_session=True, skip_csrf=True, timeout=300)

        # Verify get_authenticated_session was called with correct args
        assert mock_get.called, "get_authenticated_session should be called"
        mock_get.assert_called_with("Test Action", True, True, 300)

        # Verify return values
        assert sm == mock_sm, "Should return session manager"
        assert uuid == "wrapper-test-uuid", "Should return UUID"

        logger.info("✅ ensure_session_for_tests wrapper works correctly")
        return True


def _test_ensure_session_for_tests_sm_only_wrapper() -> bool:
    """Test ensure_session_for_tests_sm_only compatibility wrapper."""
    from unittest.mock import MagicMock, patch

    # Clear cache first
    clear_cached_session()

    with patch('session_utils.get_authenticated_session') as mock_get:
        # Setup mock
        mock_sm = MagicMock()
        mock_sm.my_uuid = "wrapper-sm-only-uuid"
        mock_get.return_value = (mock_sm, "wrapper-sm-only-uuid")

        # Test wrapper
        sm = ensure_session_for_tests_sm_only("Test Action", reuse_session=False, skip_csrf=False, timeout=120)

        # Verify get_authenticated_session was called with correct args
        assert mock_get.called, "get_authenticated_session should be called"
        mock_get.assert_called_with("Test Action", False, False, 120)

        # Verify return value (only SessionManager, not tuple)
        assert sm == mock_sm, "Should return session manager only"

        logger.info("✅ ensure_session_for_tests_sm_only wrapper works correctly")
        return True


def _test_session_caching_behavior() -> bool:
    """Test that session caching works across multiple calls."""
    from unittest.mock import MagicMock, patch

    # Clear cache first
    clear_cached_session()

    with patch('session_utils.create_and_start_session') as mock_create, \
         patch('session_utils.authenticate_session'), \
         patch('session_utils.validate_session_ready'):

        # Setup mocks
        mock_sm = MagicMock()
        mock_sm.my_uuid = "cache-test-uuid"
        mock_create.return_value = mock_sm

        # First call - should create new session
        sm1, uuid1 = get_authenticated_session("Test 1", reuse_cached=True)
        assert mock_create.call_count == 1, "Should create session on first call"

        # Second call - should reuse cached session
        sm2, uuid2 = get_authenticated_session("Test 2", reuse_cached=True)
        assert mock_create.call_count == 1, "Should NOT create new session on second call"

        # Verify same session returned
        assert sm1 == sm2, "Should return same session manager"
        assert uuid1 == uuid2, "Should return same UUID"

        # Third call with reuse_cached=False - should create new session
        sm3, uuid3 = get_authenticated_session("Test 3", reuse_cached=False)
        assert mock_create.call_count == 2, "Should create new session when reuse_cached=False"
        assert sm3 is not None, "Should return session manager"
        assert uuid3 is not None, "Should return UUID"

        logger.info("✅ Session caching behavior works correctly")

        # Clean up
        clear_cached_session()
        return True


def session_utils_module_tests() -> bool:
    """Comprehensive test suite for session_utils.py"""
    from test_framework import TestSuite

    suite = TestSuite("Session Utils", "session_utils.py")
    suite.start_suite()

    # Category 1: Session Creation Tests
    suite.run_test(
        "Create and start session - success",
        _test_create_and_start_session_success,
        "Session created and started successfully",
        "create_and_start_session()",
        "Tests successful session creation with browser initialization"
    )

    suite.run_test(
        "Create and start session - timeout",
        _test_create_and_start_session_timeout,
        "Timeout handled correctly with cleanup",
        "create_and_start_session()",
        "Tests timeout handling when session start hangs"
    )

    suite.run_test(
        "Create and start session - failure",
        _test_create_and_start_session_failure,
        "Failure handled correctly with cleanup",
        "create_and_start_session()",
        "Tests error handling when session start fails"
    )

    # Category 2: Authentication Tests
    suite.run_test(
        "Authenticate session - already logged in",
        _test_authenticate_session_already_logged_in,
        "Already logged in detected correctly",
        "authenticate_session()",
        "Tests authentication when user is already logged in"
    )

    suite.run_test(
        "Authenticate session - login required",
        _test_authenticate_session_login_required,
        "Login process executed successfully",
        "authenticate_session()",
        "Tests authentication when login is required"
    )

    suite.run_test(
        "Authenticate session - login failure",
        _test_authenticate_session_login_failure,
        "Login failure handled correctly",
        "authenticate_session()",
        "Tests error handling when login fails"
    )

    suite.run_test(
        "Authenticate session - timeout",
        _test_authenticate_session_timeout,
        "Authentication timeout handled correctly",
        "authenticate_session()",
        "Tests timeout handling during authentication"
    )

    # Category 3: Session Validation Tests
    suite.run_test(
        "Validate session ready - success",
        _test_validate_session_ready_success,
        "Session validated successfully",
        "validate_session_ready()",
        "Tests session validation when session is ready"
    )

    suite.run_test(
        "Validate session ready - not ready",
        _test_validate_session_ready_not_ready,
        "Not ready handled correctly with cleanup",
        "validate_session_ready()",
        "Tests error handling when session is not ready"
    )

    suite.run_test(
        "Validate session ready - no UUID",
        _test_validate_session_ready_no_uuid,
        "Missing UUID handled correctly with cleanup",
        "validate_session_ready()",
        "Tests error handling when UUID is missing"
    )

    # Category 4: Main Entry Point Tests
    suite.run_test(
        "Get authenticated session - new",
        _test_get_authenticated_session_new,
        "New session created and authenticated",
        "get_authenticated_session()",
        "Tests creating a new authenticated session"
    )

    suite.run_test(
        "Get authenticated session - cached",
        _test_get_authenticated_session_cached,
        "Cached session reused correctly",
        "get_authenticated_session()",
        "Tests reusing cached authenticated session"
    )

    # Category 5: Cache Management Tests
    suite.run_test(
        "Clear cached session",
        _test_clear_cached_session,
        "Cache cleared successfully",
        "clear_cached_session()",
        "Tests clearing cached session without closing"
    )

    suite.run_test(
        "Close cached session",
        _test_close_cached_session,
        "Cached session closed and cleared",
        "close_cached_session()",
        "Tests closing and clearing cached session"
    )

    suite.run_test(
        "Session caching behavior",
        _test_session_caching_behavior,
        "Caching works across multiple calls",
        "get_authenticated_session()",
        "Tests session caching across multiple calls"
    )

    # Category 6: Compatibility Wrapper Tests
    suite.run_test(
        "ensure_session_for_tests wrapper",
        _test_ensure_session_for_tests_wrapper,
        "Wrapper calls get_authenticated_session correctly",
        "ensure_session_for_tests()",
        "Tests compatibility wrapper for action scripts"
    )

    suite.run_test(
        "ensure_session_for_tests_sm_only wrapper",
        _test_ensure_session_for_tests_sm_only_wrapper,
        "SM-only wrapper returns SessionManager only",
        "ensure_session_for_tests_sm_only()",
        "Tests compatibility wrapper for action7"
    )

    return suite.finish_suite()


# Create standard test runner
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(session_utils_module_tests)

