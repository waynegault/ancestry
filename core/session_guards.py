import os
import sys
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

import logging

from browser.css_selectors import WAIT_FOR_PAGE_SELECTOR
from core.registry_utils import auto_register_module
from core.utils import nav_to_page

logger = logging.getLogger(__name__)
auto_register_module(globals(), __name__)


def ensure_navigation_ready(
    session_manager: Any,
    *,
    action_label: str,
    target_url: str,
    wait_selector: str,
    failure_reason: str,
) -> bool:
    """Shared guard that ensures driver availability and page navigation."""

    driver = session_manager.browser_manager.driver
    if driver is None:
        logger.error(f"Driver not available for {action_label} navigation")
        print("ERROR: Browser session not available. Please rerun login (Action 5).")
        return False

    if not nav_to_page(driver, target_url, wait_selector, session_manager):
        logger.error(f"{action_label} nav FAILED - {failure_reason}")
        print(f"ERROR: {failure_reason} Check network connection.")
        return False

    logger.debug(f"Navigation to {target_url} successful for {action_label}. Waiting briefly before continuing...")
    time.sleep(2)
    return True


def ensure_interactive_session_ready(session_manager: Any, action_label: str) -> bool:
    """Ensure session_manager exists and session_ready flag is true for an action."""
    if not session_manager:
        logger.error(f"Cannot {action_label}: SessionManager is None.")
        return False

    session_ready = getattr(session_manager, "session_ready", None)
    if session_ready is None:
        driver_live = getattr(session_manager, "driver_live", False)
        if driver_live:
            logger.warning("session_ready not set, initializing based on driver_live")
            session_manager.session_ready = True
            session_ready = True
        else:
            logger.warning("session_ready and driver_live not set, initializing to False")
            session_manager.session_ready = False
            session_ready = False

    if not session_ready:
        logger.error(f"Cannot {action_label}: Session not ready.")
        return False

    return True


def require_interactive_session(action_label: str) -> Callable[[Callable[..., Any]], Callable[..., bool]]:
    """Decorator that enforces session readiness before running an action."""

    def decorator(func: Callable[..., Any]) -> Callable[..., bool]:
        @wraps(func)
        def wrapper(session_manager: Any, *args: Any, **kwargs: Any) -> bool:
            if not ensure_interactive_session_ready(session_manager, action_label):
                return False
            result = func(session_manager, *args, **kwargs)
            return bool(result)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Module Tests
# ---------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner


def _test_navigation_requires_driver() -> bool:
    session_manager = MagicMock()
    session_manager.browser_manager.driver = None

    result = ensure_navigation_ready(
        session_manager,
        action_label="Action 7",
        target_url="https://example.com",
        wait_selector=WAIT_FOR_PAGE_SELECTOR,
        failure_reason="Driver missing",
    )

    assert result is False, "Navigation should fail when driver is unavailable"
    return True


def _test_navigation_failure_logs_error() -> bool:
    session_manager = MagicMock()
    session_manager.browser_manager.driver = object()

    module_ref = sys.modules[__name__]
    with patch.object(module_ref, "nav_to_page", return_value=False) as mock_nav:
        result = ensure_navigation_ready(
            session_manager,
            action_label="Action 7",
            target_url="https://example.com",
            wait_selector=WAIT_FOR_PAGE_SELECTOR,
            failure_reason="Navigation failed",
        )

    mock_nav.assert_called_once()
    assert result is False, "Navigation should report failure when nav_to_page returns False"
    return True


def _test_navigation_success_waits() -> bool:
    session_manager = MagicMock()
    session_manager.browser_manager.driver = object()

    module_ref = sys.modules[__name__]
    with (
        patch.object(module_ref, "nav_to_page", return_value=True) as mock_nav,
        patch("core.session_guards.time.sleep") as mock_sleep,
    ):
        result = ensure_navigation_ready(
            session_manager,
            action_label="Action 7",
            target_url="https://example.com",
            wait_selector=WAIT_FOR_PAGE_SELECTOR,
            failure_reason="Should not hit",
        )

    mock_nav.assert_called_once_with(
        session_manager.browser_manager.driver, "https://example.com", WAIT_FOR_PAGE_SELECTOR, session_manager
    )
    mock_sleep.assert_called_once_with(2)
    assert result is True, "Successful navigation should return True"
    return True


def _test_interactive_session_ready_handles_missing_flags() -> bool:
    session_manager = SimpleNamespace(driver_live=True)

    result = ensure_interactive_session_ready(session_manager, "Action 8")

    assert result is True, "driver_live=True should bootstrap session_ready"
    assert session_manager.session_ready is True, "session_ready should be set True"
    return True


def _test_interactive_session_ready_handles_driver_offline() -> bool:
    session_manager = SimpleNamespace(driver_live=False)

    result = ensure_interactive_session_ready(session_manager, "Action 8")

    assert result is False, "driver_live=False should leave session in not-ready state"
    assert session_manager.session_ready is False
    return True


def _test_require_interactive_session_decorator() -> bool:
    calls: list[str] = []

    @require_interactive_session("Action 7")
    def sample_action(sm: Any) -> str:
        calls.append("called")
        assert getattr(sm, "session_ready", False)
        return "ok"

    # Failure path
    assert sample_action(None) is False, "Decorator should block when session manager is missing"
    assert not calls, "Function should not be invoked when guard fails"

    ready_sm = SimpleNamespace(session_ready=True)
    assert sample_action(ready_sm) is True, "Decorator should propagate truthy results"
    assert calls == ["called"], "Function should be invoked once when guard passes"
    return True


def module_tests() -> bool:
    suite = TestSuite("core.session_guards", "core/session_guards.py")

    suite.run_test(
        "Navigation requires driver",
        _test_navigation_requires_driver,
        "Ensures navigation guard fails cleanly when no driver is present.",
    )

    suite.run_test(
        "Navigation failure",
        _test_navigation_failure_logs_error,
        "Ensures navigation guard reports False when nav_to_page fails.",
    )

    suite.run_test(
        "Navigation success",
        _test_navigation_success_waits,
        "Ensures navigation guard waits briefly after successful navigation.",
    )

    suite.run_test(
        "Interactive session bootstrap",
        _test_interactive_session_ready_handles_missing_flags,
        "Ensures driver_live True causes session_ready to bootstrap to True.",
    )

    suite.run_test(
        "Interactive session offline",
        _test_interactive_session_ready_handles_driver_offline,
        "Ensures driver_live False leaves the session not ready.",
    )

    suite.run_test(
        "Require interactive session decorator",
        _test_require_interactive_session_decorator,
        "Ensures decorator blocks calls without a ready session and coerces return values to bool.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


def _should_run_module_tests() -> bool:
    return os.environ.get("RUN_MODULE_TESTS") == "1"


def _print_module_usage() -> int:
    print("core.session_guards only exposes importable helpers; there is no CLI entry point.")
    print("Set RUN_MODULE_TESTS=1 before running this module to execute the embedded tests.")
    return 0


if __name__ == "__main__":
    if _should_run_module_tests():
        success = run_comprehensive_tests()
        raise SystemExit(0 if success else 1)
    raise SystemExit(_print_module_usage())
