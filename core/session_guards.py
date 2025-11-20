from typing import Any, Callable
from functools import wraps
import time
from standard_imports import setup_module
from utils import nav_to_page

logger = setup_module(globals(), __name__)

def ensure_navigation_ready(
    session_manager: Any,
    *,
    action_label: str,
    target_url: str,
    wait_selector: str,
    failure_reason: str,
) -> bool:
    """Shared guard that ensures driver availability and page navigation."""

    driver = session_manager.driver
    if driver is None:
        logger.error(f"Driver not available for {action_label} navigation")
        print("ERROR: Browser session not available. Please rerun login (Action 5).")
        return False

    if not nav_to_page(driver, target_url, wait_selector, session_manager):
        logger.error(f"{action_label} nav FAILED - {failure_reason}")
        print(f"ERROR: {failure_reason} Check network connection.")
        return False

    logger.debug(
        f"Navigation to {target_url} successful for {action_label}. Waiting briefly before continuing..."
    )
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
