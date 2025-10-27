#!/usr/bin/env python3

"""
session_utils.py - Global Session Management (SINGLE SOURCE OF TRUTH)

This module provides the ONLY way to access authenticated sessions in the application.

ARCHITECTURE:
1. main.py creates ONE SessionManager at startup
2. main.py registers it as the global session via set_global_session()
3. main.py authenticates it via get_authenticated_session()
4. All actions receive this session as a parameter
5. All tests access this session via ensure_session_for_tests()

RULES:
- NO session creation outside of main.py
- NO alternative paths or fallbacks
- NO backward compatibility
- ONLY the global session is allowed

BENEFITS:
- Single session creation: No duplicate SessionManager instances
- Single authentication: Login happens once at startup
- Consistent state: All code uses the same authenticated session
- Maximum simplicity: Only ONE way to get a session
- DRY principle: Single source of truth
"""

# === STANDARD LIBRARY IMPORTS ===
from typing import Optional

# === LOCAL IMPORTS ===
from core.session_manager import SessionManager
from logging_config import setup_logging

# === MODULE LOGGER ===
logger = setup_logging(log_level="INFO")

# === GLOBAL SESSION CACHE ===
# This is the SINGLE SOURCE OF TRUTH for session management
_cached_session_manager: Optional[SessionManager] = None
_cached_session_uuid: Optional[str] = None

# Print-once guard for authentication banner
_auth_banner_printed: bool = False


# ==============================================
# CORE FUNCTIONS (ONLY THESE ARE NEEDED)
# ==============================================


def set_global_session(session_manager: SessionManager) -> None:
    """
    Register a SessionManager as the global session.

    This should ONLY be called by main.py at startup.

    Args:
        session_manager: The SessionManager instance to register globally
    """
    global _cached_session_manager, _cached_session_uuid
    prev = _cached_session_manager
    _cached_session_manager = session_manager
    # Don't try to get UUID yet - session hasn't been authenticated
    # UUID will be set when get_authenticated_session() is called
    _cached_session_uuid = None
    if prev is session_manager:
        logger.debug("Global session already registered; skipping re-register")
    else:
        logger.info("✅ Global session registered (not yet authenticated)")


def get_global_session() -> Optional[SessionManager]:
    """
    Get the global session manager if it exists.

    Returns:
        Optional[SessionManager]: The cached session manager, or None if not set
    """
    return _cached_session_manager


def _log_session_banner(already_auth: bool, env_uuid: Optional[str], action_name: str) -> None:
    """Log the session banner once per authentication attempt (pre-auth)."""
    if not already_auth:
        logger.info("=" * 80)
        logger.info(f"Using global authenticated session (Action: {action_name})")
        if env_uuid:
            logger.info(f"UUID (from .env): {env_uuid} — will verify during authentication")
        else:
            logger.debug("UUID: Not yet set (will be discovered during authentication)")
        logger.info("=" * 80)
    else:
        logger.debug("Reusing authenticated global session; banner suppressed")


def _ensure_session_ready_or_raise(session_manager: SessionManager, action_name: str, skip_csrf: bool) -> None:
    """Ensure the session is ready; raise AssertionError if not."""
    ready = session_manager.ensure_session_ready(action_name, skip_csrf=skip_csrf)
    if not ready:
        raise AssertionError("Session not ready - cookies/identifiers missing")


def _finalize_first_auth_and_get_uuid(already_auth: bool, session_manager: SessionManager) -> str:
    """On first-time auth, cache UUID and print success banner; return UUID."""
    global _cached_session_uuid, _auth_banner_printed
    if not already_auth:
        if not session_manager.my_uuid:
            raise AssertionError("UUID not available - session initialization incomplete")
        _cached_session_uuid = session_manager.my_uuid
        if not _auth_banner_printed:
            logger.info(f"✅ Global session now authenticated: UUID={_cached_session_uuid}")
            _auth_banner_printed = True
        else:
            logger.debug(f"Global session ready (UUID={_cached_session_uuid})")
    return _cached_session_uuid



def _assert_global_session_exists() -> None:
    """Raise if the global session has not been registered by main.py."""
    global _cached_session_manager
    if not _cached_session_manager:
        raise RuntimeError(
            "No global session available. main.py must call set_global_session() before any actions or tests can run. "
            "If running a script directly, ensure main.py has been run first to register the global session."
        )


def _pre_auth_logging(already_auth: bool, env_uuid: Optional[str], action_name: str) -> None:
    """Centralize pre-auth logging to reduce cyclomatic complexity."""
    global _auth_banner_printed, _cached_session_uuid
    if not already_auth and not _auth_banner_printed:
        _log_session_banner(already_auth, env_uuid, action_name)
    elif already_auth:
        logger.debug(f"Using cached global session (UUID={_cached_session_uuid}) for {action_name}")
    else:
        logger.debug(f"Re-validating global session state for {action_name}")

def get_authenticated_session(
    action_name: str = "Session Setup",
    skip_csrf: bool = True
) -> tuple[SessionManager, str]:
    """
    Get the global authenticated session manager (single source of truth).

    - Requires main.py to have registered the global session via set_global_session().
    - Authenticates the session on first use, then reuses cached credentials.
    """
    global _cached_session_manager, _cached_session_uuid
    global _auth_banner_printed

    # 1) Validate that a global session exists
    _assert_global_session_exists()

    # 2) Determine session state and pre-auth logging
    from config import config_schema
    env_uuid = getattr(getattr(config_schema, 'api', object()), 'my_uuid', None)
    already_auth = bool(_cached_session_uuid and _cached_session_manager.my_uuid)
    _pre_auth_logging(already_auth, env_uuid, action_name)

    # 3) Ensure session is ready (no-op if cached)
    sm = _cached_session_manager
    assert sm is not None
    _ensure_session_ready_or_raise(sm, action_name, skip_csrf)

    # 4) Finalize first-time auth (cache UUID and print banner once)
    _cached_session_uuid = _finalize_first_auth_and_get_uuid(already_auth, sm)

    return _cached_session_manager, _cached_session_uuid


def clear_cached_session() -> None:
    """
    Clear the cached session without closing it.

    This is useful for testing when you want to reset the cache
    without actually closing the session.
    """
    global _cached_session_manager, _cached_session_uuid
    _cached_session_manager = None
    _cached_session_uuid = None
    logger.info("Cached session cleared (not closed)")


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


# ==============================================
# CONVENIENCE WRAPPERS FOR TEST FUNCTIONS
# ==============================================


def ensure_session_for_tests(
    action_name: str = "Test Session",
    skip_csrf: bool = True
) -> tuple[SessionManager, str]:
    """
    Get the global authenticated session for test functions.

    This is a convenience wrapper around get_authenticated_session().

    Args:
        action_name: Name of the action for logging
        skip_csrf: Whether to skip CSRF token check (default: True)

    Returns:
        tuple[SessionManager, str]: (session_manager, uuid)

    Raises:
        RuntimeError: If no global session has been registered

    Example:
        sm, uuid = ensure_session_for_tests("Action 6 Test")
    """
    return get_authenticated_session(action_name, skip_csrf)


def ensure_session_for_tests_sm_only(
    action_name: str = "Test Session",
    skip_csrf: bool = True
) -> SessionManager:
    """
    Get the global authenticated session (SessionManager only) for test functions.

    This returns only the SessionManager, not the UUID.

    Args:
        action_name: Name of the action for logging
        skip_csrf: Whether to skip CSRF token check (default: True)

    Returns:
        SessionManager: Authenticated session manager

    Raises:
        RuntimeError: If no global session has been registered

    Example:
        sm = ensure_session_for_tests_sm_only("Action 7 Test")
    """
    sm, _ = get_authenticated_session(action_name, skip_csrf)
    return sm


# ==============================================
# MODULE TESTS
# ==============================================


def _test_global_session_not_set() -> bool:
    """Test that error is raised when no global session is set."""
    # Clear cache first
    clear_cached_session()

    try:
        get_authenticated_session("Test Action")
        raise AssertionError("Should have raised RuntimeError when no global session")
    except RuntimeError as e:
        assert "No global session available" in str(e)
        logger.info("✅ Correctly raises error when no global session")
        return True


def _test_global_session_set() -> bool:
    """Test that global session is returned when set."""
    from unittest.mock import MagicMock

    # Setup global session
    global _cached_session_manager, _cached_session_uuid
    mock_sm = MagicMock()
    mock_sm.my_uuid = "test-uuid-12345"
    _cached_session_manager = mock_sm
    _cached_session_uuid = "test-uuid-12345"

    # Get global session
    sm, uuid = get_authenticated_session("Test Action")

    # Verify
    assert sm == mock_sm, "Should return global session manager"
    assert uuid == "test-uuid-12345", "Should return global UUID"

    logger.info("✅ Global session returned correctly")

    # Clean up
    clear_cached_session()
    return True


def _test_ensure_session_for_tests_wrapper() -> bool:
    """Test ensure_session_for_tests wrapper."""
    from unittest.mock import MagicMock

    # Setup global session
    global _cached_session_manager, _cached_session_uuid
    mock_sm = MagicMock()
    mock_sm.my_uuid = "wrapper-test-uuid"
    _cached_session_manager = mock_sm
    _cached_session_uuid = "wrapper-test-uuid"

    # Test wrapper
    sm, uuid = ensure_session_for_tests("Test Action")

    # Verify
    assert sm == mock_sm, "Should return session manager"
    assert uuid == "wrapper-test-uuid", "Should return UUID"

    logger.info("✅ ensure_session_for_tests wrapper works")

    # Clean up
    clear_cached_session()
    return True


def session_utils_module_tests() -> bool:
    """Run all module tests."""
    from test_framework import TestSuite

    suite = TestSuite("session_utils.py - Global Session Management", "session_utils.py")

    suite.run_test(
        "No global session error",
        _test_global_session_not_set,
        "Raises error when no global session",
        "get_authenticated_session()",
        "Tests that error is raised when global session not set"
    )

    suite.run_test(
        "Global session returned",
        _test_global_session_set,
        "Returns global session when set",
        "get_authenticated_session()",
        "Tests that global session is returned correctly"
    )

    suite.run_test(
        "ensure_session_for_tests wrapper",
        _test_ensure_session_for_tests_wrapper,
        "Wrapper returns global session",
        "ensure_session_for_tests()",
        "Tests convenience wrapper for test functions"
    )

    return suite.finish_suite()


# Create standard test runner
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(session_utils_module_tests)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SESSION_UTILS.PY - GLOBAL SESSION MANAGEMENT TESTS")
    print("=" * 80 + "\n")

    success = run_comprehensive_tests()

    if success:
        print("\n✅ All session_utils tests passed!")
    else:
        print("\n❌ Some session_utils tests failed!")

    exit(0 if success else 1)
