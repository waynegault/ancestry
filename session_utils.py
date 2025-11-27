#!/usr/bin/env python3

"""
session_utils.py - Global Session Management with Dependency Injection

This module provides session management using the DI container pattern while
maintaining backward compatibility with the previous global state approach.

ARCHITECTURE (DI-based):
1. main.py creates ONE SessionManager at startup
2. main.py registers it via register_session_manager() which uses DI container
3. Consumers access via get_session_manager() which resolves from DI container
4. All action functions receive session_manager as first parameter
5. Helper functions use @requires_session decorator or get_session_manager()

API:
- register_session_manager() - Register SessionManager at startup
- get_session_manager() - Get SessionManager from DI container
- is_session_available() - Check if session is registered
- @requires_session - Decorator for functions needing session

BENEFITS:
- Testability: Easy to inject mocks via DI container
- Single source of truth: One SessionManager instance
- Thread safety: DI container is thread-safe
- Type safety: Protocol-based interface checking
- Explicit dependencies: Clear what functions need sessions
"""

# === STANDARD LIBRARY IMPORTS ===
import contextlib
import functools
import os
import sys
from collections.abc import Callable, Iterator
from typing import Any, Optional, ParamSpec, TypeVar, cast
from unittest import mock

# === LOCAL IMPORTS ===
from core.session_manager import SessionManager
from logging_config import setup_logging

# === MODULE LOGGER ===
_env_log_level = os.getenv("LOG_LEVEL", "INFO")
logger = setup_logging(log_level=_env_log_level)

# === TYPE VARIABLES FOR DECORATORS ===
P = ParamSpec("P")
R = TypeVar("R")


# === GLOBAL SESSION CACHE (LEGACY - KEPT FOR BACKWARD COMPATIBILITY) ===
# New code should use the DI container pattern below
class _State:
    session_manager: Optional[SessionManager] = None
    session_uuid: Optional[str] = None
    auth_banner_printed: bool = False


GLOBAL_SESSION = _State()


# ==============================================
# DI CONTAINER INTEGRATION (PREFERRED APPROACH)
# ==============================================


def _get_di_container() -> Any:
    """Get the DI container (lazy import to avoid circular dependencies)."""
    from core.dependency_injection import get_container

    return get_container()


def register_session_manager(session_manager: SessionManager) -> None:
    """
    Register a SessionManager via the DI container.

    This is the PREFERRED method for registering sessions at startup.
    Also updates the legacy global state for backward compatibility.

    Args:
        session_manager: The SessionManager instance to register
    """
    try:
        container = _get_di_container()
        container.register_instance(SessionManager, session_manager)
        logger.debug("✅ SessionManager registered in DI container")
    except Exception as e:
        logger.warning(f"Could not register in DI container: {e}")

    # Also update legacy global state for backward compatibility
    GLOBAL_SESSION.session_manager = session_manager
    GLOBAL_SESSION.session_uuid = None
    logger.debug("✅ SessionManager registered (DI + legacy)")


def get_session_manager() -> Optional[SessionManager]:
    """
    Get the SessionManager from the DI container.

    This is the PREFERRED method for accessing the session.
    Falls back to legacy global state if DI container is not configured.

    Returns:
        SessionManager if registered, None otherwise
    """
    # Try DI container first
    try:
        container = _get_di_container()
        if container.is_registered(SessionManager):
            return container.resolve(SessionManager)
    except Exception:
        pass  # Fall through to legacy

    # Fallback to legacy global state
    return GLOBAL_SESSION.session_manager


def is_session_available() -> bool:
    """
    Check if a SessionManager is available (either via DI or legacy).

    Returns:
        True if a SessionManager is registered
    """
    return get_session_manager() is not None


# ==============================================
# @requires_session DECORATOR
# ==============================================


class SessionNotAvailableError(RuntimeError):
    """Raised when a function decorated with @requires_session has no session available."""

    def __init__(self, func_name: str, message: str | None = None):
        self.func_name = func_name
        default_msg = (
            f"Function '{func_name}' requires an authenticated session, but none is available. "
            "Ensure main.py has called register_session_manager() before invoking this function."
        )
        super().__init__(message or default_msg)


def requires_session(
    *,
    inject_session: bool = False,
    skip_csrf: bool = True,
    action_name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that ensures a SessionManager is available before function execution.

    Usage:
        @requires_session()
        def my_function():
            sm = get_session_manager()
            # Use sm...

        @requires_session(inject_session=True)
        def my_function(session_manager: SessionManager, other_arg: str):
            # session_manager is automatically injected as first argument
            pass

    Args:
        inject_session: If True, automatically injects SessionManager as first argument
        skip_csrf: Whether to skip CSRF validation when ensuring session ready
        action_name: Name for logging; defaults to function name

    Raises:
        SessionNotAvailableError: If no session is registered
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            func_action_name = action_name or func.__name__

            # Get session manager
            sm = get_session_manager()
            if sm is None:
                raise SessionNotAvailableError(func.__name__)

            # Ensure session is ready
            try:
                sm.ensure_session_ready(func_action_name, skip_csrf=skip_csrf)
            except Exception as e:
                logger.error(f"Session not ready for {func_action_name}: {e}")
                raise SessionNotAvailableError(func.__name__, f"Session exists but is not ready: {e}") from e

            # Inject session if requested
            if inject_session:
                return func(sm, *args, **kwargs)  # type: ignore[arg-type]
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _log_session_banner(already_auth: bool, env_uuid: Optional[str], action_name: str) -> None:
    """Log the session banner once per authentication attempt (pre-auth)."""
    if not already_auth:
        logger.debug(f"Authenticating session for: {action_name}")
        if env_uuid:
            logger.debug(f"UUID (from .env): {env_uuid} — will verify during authentication")
        else:
            logger.debug("UUID: Not yet set (will be discovered during authentication)")
    else:
        logger.debug("Reusing authenticated global session; banner suppressed")


def _ensure_session_ready_or_raise(session_manager: SessionManager, action_name: str, skip_csrf: bool) -> None:
    """Ensure the session is ready; raise AssertionError if not."""
    ready = session_manager.ensure_session_ready(action_name, skip_csrf=skip_csrf)
    if not ready:
        raise AssertionError("Session not ready - cookies/identifiers missing")


def _finalize_first_auth_and_get_uuid(already_auth: bool, session_manager: SessionManager) -> str:
    """On first-time auth, cache UUID and print success banner; return UUID."""
    if not already_auth:
        if not session_manager.my_uuid:
            raise AssertionError("UUID not available - session initialization incomplete")
        GLOBAL_SESSION.session_uuid = session_manager.my_uuid
        if not GLOBAL_SESSION.auth_banner_printed:
            logger.debug(f"✅ Global session now authenticated: UUID={GLOBAL_SESSION.session_uuid}")
            GLOBAL_SESSION.auth_banner_printed = True
        else:
            logger.debug(f"Global session ready (UUID={GLOBAL_SESSION.session_uuid})")
    if GLOBAL_SESSION.session_uuid is None:
        raise AssertionError("UUID not cached after session authentication")
    return GLOBAL_SESSION.session_uuid


def _assert_global_session_exists() -> None:
    """Raise if the global session has not been registered."""
    if not is_session_available():
        raise RuntimeError(
            "No session available. main.py must call register_session_manager() "
            "Ensure main.py has called register_session_manager() before any actions or tests can run. "
            "If running a script directly, ensure main.py has been run first."
        )


def _pre_auth_logging(already_auth: bool, env_uuid: Optional[str], action_name: str) -> None:
    """Centralize pre-auth logging to reduce cyclomatic complexity."""
    if not already_auth and not GLOBAL_SESSION.auth_banner_printed:
        _log_session_banner(already_auth, env_uuid, action_name)
    elif already_auth:
        logger.debug(f"Using cached global session (UUID={GLOBAL_SESSION.session_uuid}) for {action_name}")
    else:
        logger.debug(f"Re-validating global session state for {action_name}")


def get_authenticated_session(action_name: str = "Session Setup", skip_csrf: bool = True) -> tuple[SessionManager, str]:
    """
    Get the global authenticated session manager (single source of truth).

    NOTE: For new code, consider using the @requires_session decorator instead:

        @requires_session(inject_session=True)
        def my_function(session_manager: SessionManager):
            # session_manager is automatically injected
            pass

    - Requires main.py to have registered the session via register_session_manager().
    - Authenticates the session on first use, then reuses cached credentials.
    """
    # 1) Validate that a session exists (DI or legacy)
    _assert_global_session_exists()

    # 2) Determine session state and pre-auth logging
    from config import config_schema

    env_uuid = getattr(getattr(config_schema, 'api', object()), 'my_uuid', None)
    already_auth = bool(GLOBAL_SESSION.session_uuid and getattr(get_session_manager(), "my_uuid", None))
    _pre_auth_logging(already_auth, env_uuid, action_name)

    # 3) Ensure session is ready (no-op if cached)
    sm = get_session_manager()
    assert sm is not None
    _ensure_session_ready_or_raise(sm, action_name, skip_csrf)

    # 4) Finalize first-time auth (cache UUID and print banner once)
    session_uuid = _finalize_first_auth_and_get_uuid(already_auth, sm)
    GLOBAL_SESSION.session_uuid = session_uuid

    return sm, session_uuid


def clear_cached_session() -> None:
    """
    Clear the cached session without closing it.

    This is useful for testing when you want to reset the cache
    without actually closing the session.
    """
    # Clear legacy state
    GLOBAL_SESSION.session_manager = None
    GLOBAL_SESSION.session_uuid = None
    GLOBAL_SESSION.auth_banner_printed = False

    # Clear DI container registration
    try:
        from core.dependency_injection import ServiceRegistry

        ServiceRegistry.clear_container("default")
    except Exception:
        pass

    logger.info("Cached session cleared (not closed)")


def close_cached_session(keep_db: bool = True) -> None:
    """
    Close and clear the cached session.

    Args:
        keep_db: Whether to keep database connections (default: True)
    """
    sm = get_session_manager()
    if sm:
        logger.info("Closing cached session")
        sm.close_sess(keep_db=keep_db)

    # Clear legacy state
    GLOBAL_SESSION.session_manager = None
    GLOBAL_SESSION.session_uuid = None
    GLOBAL_SESSION.auth_banner_printed = False


# ==============================================
# CONVENIENCE WRAPPERS FOR TEST FUNCTIONS
# ==============================================


def ensure_session_for_tests(action_name: str = "Test Session", skip_csrf: bool = True) -> tuple[SessionManager, str]:
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


def ensure_session_for_tests_sm_only(action_name: str = "Test Session", skip_csrf: bool = True) -> SessionManager:
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
        assert "No session available" in str(e)
        logger.info("✅ Correctly raises error when no global session")
        return True


def _test_global_session_set() -> bool:
    """Test that global session is returned when set."""
    from unittest.mock import MagicMock

    # Setup global session via API
    mock_sm = MagicMock()
    mock_sm.my_uuid = "test-uuid-12345"
    mock_sm.ensure_session_ready.return_value = True
    register_session_manager(mock_sm)

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

    # Setup global session via API
    mock_sm = MagicMock()
    mock_sm.my_uuid = "wrapper-test-uuid"
    mock_sm.ensure_session_ready.return_value = True
    register_session_manager(mock_sm)

    # Test wrapper
    sm, uuid = ensure_session_for_tests("Test Action")

    # Verify
    assert sm == mock_sm, "Should return session manager"
    assert uuid == "wrapper-test-uuid", "Should return UUID"

    logger.info("✅ ensure_session_for_tests wrapper works")

    # Clean up
    clear_cached_session()
    return True


@contextlib.contextmanager
def _preserve_global_session_state() -> Iterator[None]:
    prev_manager = GLOBAL_SESSION.session_manager
    prev_uuid = GLOBAL_SESSION.session_uuid
    prev_banner = GLOBAL_SESSION.auth_banner_printed
    try:
        yield
    finally:
        GLOBAL_SESSION.session_manager = prev_manager
        GLOBAL_SESSION.session_uuid = prev_uuid
        GLOBAL_SESSION.auth_banner_printed = prev_banner


def _test_clear_cached_session_resets_state() -> bool:
    with _preserve_global_session_state():
        GLOBAL_SESSION.session_manager = cast(SessionManager, mock.Mock(spec=SessionManager))
        GLOBAL_SESSION.session_uuid = "cached"
        clear_cached_session()
        assert GLOBAL_SESSION.session_manager is None
        assert GLOBAL_SESSION.session_uuid is None
    return True


def _test_close_cached_session_invokes_close() -> bool:
    with _preserve_global_session_state():
        fake_session = mock.Mock(spec=SessionManager)
        # Use register_session_manager to ensure both DI and legacy state are set
        register_session_manager(fake_session)
        GLOBAL_SESSION.session_uuid = "cached"
        close_cached_session(keep_db=False)
        fake_session.close_sess.assert_called_once_with(keep_db=False)
        assert GLOBAL_SESSION.session_manager is None
        assert GLOBAL_SESSION.session_uuid is None
    return True


def _test_register_session_manager_updates_both() -> bool:
    """Test that register_session_manager updates both DI container and legacy state."""
    with _preserve_global_session_state():
        from unittest.mock import MagicMock

        mock_sm = MagicMock(spec=SessionManager)
        mock_sm.my_uuid = "di-test-uuid"

        # Clear state first
        clear_cached_session()

        # Register via new API
        register_session_manager(mock_sm)

        # Verify legacy state is updated
        assert GLOBAL_SESSION.session_manager is mock_sm

        # Verify DI-based accessor works
        retrieved = get_session_manager()
        assert retrieved is mock_sm

        # Verify is_session_available
        assert is_session_available() is True

        logger.info("✅ register_session_manager updates both DI and legacy state")
        clear_cached_session()
    return True


def _test_requires_session_decorator_raises_when_no_session() -> bool:
    """Test that @requires_session raises SessionNotAvailableError when no session."""
    with _preserve_global_session_state():
        clear_cached_session()

        @requires_session()
        def my_function() -> str:
            return "should not reach here"

        try:
            my_function()
            raise AssertionError("Should have raised SessionNotAvailableError")
        except SessionNotAvailableError as e:
            assert "my_function" in str(e)
            logger.info("✅ @requires_session raises error when no session")
    return True


def _test_requires_session_decorator_with_inject() -> bool:
    """Test that @requires_session can inject session_manager as first argument."""
    with _preserve_global_session_state():
        from unittest.mock import MagicMock

        mock_sm = MagicMock(spec=SessionManager)
        mock_sm.my_uuid = "inject-test-uuid"
        mock_sm.ensure_session_ready.return_value = True
        register_session_manager(mock_sm)

        @requires_session(inject_session=True)
        def my_function(session_manager: SessionManager, value: int) -> tuple[SessionManager, int]:
            return session_manager, value

        result_sm, result_val = my_function(42)
        assert result_sm is mock_sm, "Session should be injected"
        assert result_val == 42, "Other args should pass through"

        logger.info("✅ @requires_session(inject_session=True) works")
        clear_cached_session()
    return True


def _test_get_session_manager_fallback_to_legacy() -> bool:
    """Test that get_session_manager falls back to legacy state if DI unavailable."""
    with _preserve_global_session_state():
        from unittest.mock import MagicMock

        # Clear everything first (including DI container)
        clear_cached_session()

        # Set only legacy state (bypass DI by direct assignment AFTER clearing)
        mock_sm = MagicMock(spec=SessionManager)
        GLOBAL_SESSION.session_manager = mock_sm

        # get_session_manager should return it via legacy fallback
        result = get_session_manager()
        assert result is mock_sm, "Should fall back to legacy state"

        logger.info("✅ get_session_manager falls back to legacy state")
        # Don't call clear_cached_session here as _preserve_global_session_state will restore
    return True


def session_utils_module_tests() -> bool:
    """Run all module tests."""
    from test_framework import TestSuite

    suite = TestSuite("session_utils.py - Session Management with DI", "session_utils.py")

    suite.run_test(
        "No global session error",
        _test_global_session_not_set,
        "Raises error when no global session",
        "get_authenticated_session()",
        "Tests that error is raised when global session not set",
    )

    suite.run_test(
        "Global session returned",
        _test_global_session_set,
        "Returns global session when set",
        "get_authenticated_session()",
        "Tests that global session is returned correctly",
    )

    suite.run_test(
        "ensure_session_for_tests wrapper",
        _test_ensure_session_for_tests_wrapper,
        "Wrapper returns global session",
        "ensure_session_for_tests()",
        "Tests convenience wrapper for test functions",
    )

    suite.run_test(
        "clear_cached_session resets state",
        _test_clear_cached_session_resets_state,
        "Ensures cached session references are cleared without closing",
        "clear_cached_session()",
        "Verifies global cache is cleared safely",
    )

    suite.run_test(
        "close_cached_session closes and clears",
        _test_close_cached_session_invokes_close,
        "Ensures close_cached_session invokes close_sess and clears globals",
        "close_cached_session()",
        "Verifies keep_db flag propagates to SessionManager",
    )

    # New DI-based tests
    suite.run_test(
        "register_session_manager updates DI and legacy",
        _test_register_session_manager_updates_both,
        "Registers session in both DI container and legacy state",
        "register_session_manager()",
        "Tests DI container integration with backward compatibility",
    )

    suite.run_test(
        "@requires_session raises without session",
        _test_requires_session_decorator_raises_when_no_session,
        "Decorator raises SessionNotAvailableError when no session registered",
        "@requires_session()",
        "Tests decorator validation behavior",
    )

    suite.run_test(
        "@requires_session with inject_session",
        _test_requires_session_decorator_with_inject,
        "Decorator injects session_manager as first argument",
        "@requires_session(inject_session=True)",
        "Tests decorator injection behavior",
    )

    suite.run_test(
        "get_session_manager fallback to legacy",
        _test_get_session_manager_fallback_to_legacy,
        "Falls back to legacy state if DI container unavailable",
        "get_session_manager()",
        "Tests backward compatibility with legacy global state",
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

    sys.exit(0 if success else 1)
