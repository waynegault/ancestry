#!/usr/bin/env python3

"""Selenium/WebDriver Utilities for Browser Automation.

Utility functions for browser automation and element interaction using
Selenium WebDriver, separated from general or API-specific utilities.
"""

# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# === CORE INFRASTRUCTURE ===
import logging

from core.error_handling import safe_execute
from core.registry_utils import auto_register_module

logger = logging.getLogger(__name__)
auto_register_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import json
import time
from typing import Any, Optional, Protocol, cast

# === THIRD-PARTY IMPORTS ===
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

# Local imports

# --- Protocols ---


class DriverProtocol(Protocol):
    """Protocol capturing the WebDriver surface to ensure strict typing."""

    def execute_script(self, script: str, *args: object) -> object: ...

    def get_cookies(self) -> list[dict[str, object]]: ...

    def add_cookie(self, cookie: dict[str, object]) -> None: ...

    def get_cookie(self, name: str) -> Optional[dict[str, object]]: ...

    def get_attribute(self, name: str) -> Optional[str]: ...


class WebElementProtocol(Protocol):
    """Protocol for WebElement to ensure strict typing."""

    def get_attribute(self, name: str) -> Optional[str]: ...

    def click(self) -> None: ...

    def clear(self) -> None: ...

    def send_keys(self, *value: object) -> None: ...

    @property
    def text(self) -> str: ...

    def is_displayed(self) -> bool: ...


# --- Selenium Specific Helpers ---


@safe_execute(default_return=False, log_errors=True)
def force_user_agent(driver: Optional[WebDriver], user_agent: str):
    """
    Attempts to force the browser's User-Agent string using Chrome DevTools Protocol.
    Now with unified error handling via safe_execute decorator.
    """
    if not driver:
        logger.warning("Driver is None, cannot set user agent")
        return False

    # Use execute_script to modify the user agent
    driver_proto = cast(DriverProtocol, driver)
    driver_proto.execute_script("navigator.userAgent = arguments[0]", user_agent)
    logger.debug(f"Set user agent to: {user_agent}")
    return True


@safe_execute(default_return="", log_errors=False)
def extract_text(element: Optional[WebElement]) -> str:
    """Extract text from an element safely with unified error handling."""
    if not element:
        return ""

    element_proto = cast(WebElementProtocol, element)
    return element_proto.text or ""


@safe_execute(default_return="", log_errors=False)
def extract_attribute(element: Optional[WebElement], attribute: str) -> str:
    """Extract attribute from an element safely with unified error handling."""
    if not element:
        return ""

    element_proto = cast(WebElementProtocol, element)
    return element_proto.get_attribute(attribute) or ""


@safe_execute(default_return=False, log_errors=False)
def is_elem_there(
    driver: Optional[WebDriver],
    selector: str,
    by: str = By.CSS_SELECTOR,
    *,
    wait: float | int = 0,
) -> bool:
    """Check if element exists with optional wait for presence."""
    if not driver:
        return False

    if wait and wait > 0:
        WebDriverWait(driver, wait).until(expected_conditions.presence_of_element_located((by, selector)))
        return True

    driver.find_element(by, selector)
    return True


@safe_execute(default_return=False, log_errors=False)
def is_browser_open(driver: Optional[WebDriver]) -> bool:
    """Check if browser is still open and responsive with unified error handling."""
    if not driver:
        return False
    # Access current_url - will raise exception if browser is closed
    _ = driver.current_url
    return True


@safe_execute(log_errors=True)
def close_tabs(driver: Optional[WebDriver], keep_first: bool = True) -> None:
    """Close browser tabs with unified error handling."""
    if not driver:
        return

    handles = driver.window_handles
    if keep_first and len(handles) > 1:
        # Close all but first tab
        for handle in handles[1:]:
            driver.switch_to.window(handle)
            driver.close()
        # Switch back to first tab
        driver.switch_to.window(handles[0])
    elif not keep_first:
        # Close all tabs
        for handle in handles:
            driver.switch_to.window(handle)
            driver.close()


@safe_execute(default_return=[], log_errors=False)
def get_driver_cookies(driver: Optional[WebDriver]) -> list[dict[str, Any]]:
    """Get all cookies from driver with unified error handling."""
    if not driver:
        return []

    driver_proto = cast(DriverProtocol, driver)
    return driver_proto.get_cookies()


@safe_execute(default_return=False, log_errors=True)
def export_cookies(driver: Optional[WebDriver], filepath: str) -> bool:
    """Export cookies to file with unified error handling."""
    if not driver:
        return False

    cookies = get_driver_cookies(driver)
    from pathlib import Path

    with Path(filepath).open("w", encoding="utf-8") as f:
        json.dump(cookies, f, indent=2)
    return True


@safe_execute(log_errors=False)
def scroll_to_element(driver: Optional[WebDriver], element: Optional[WebElement]) -> None:
    """Scroll element into view with unified error handling."""
    if not driver or not element:
        return

    driver_proto = cast(DriverProtocol, driver)
    driver_proto.execute_script("arguments[0].scrollIntoView(true);", element)
    time.sleep(0.1)  # Brief pause for scroll completion


@safe_execute(default_return=None, log_errors=False)
def wait_for_element(
    driver: Optional[WebDriver], selector: str, timeout: int = 10, by: str = By.CSS_SELECTOR
) -> Optional[WebElement]:
    """Wait for element to be present with unified error handling."""
    if not driver:
        return None

    wait = WebDriverWait(driver, timeout)
    return wait.until(expected_conditions.presence_of_element_located((by, selector)))


@safe_execute(default_return=False, log_errors=False)
def safe_click(driver: Optional[WebDriver], element: Optional[WebElement]) -> bool:
    """Safely click an element with unified error handling."""
    if not driver or not element:
        return False

    # Scroll to element first
    scroll_to_element(driver, element)
    # Try to click
    element_proto = cast(WebElementProtocol, element)
    element_proto.click()
    return True


@safe_execute(default_return=False, log_errors=False)
def is_element_visible(element: Optional[WebElement]) -> bool:
    """Check if element is visible with unified error handling."""
    if not element:
        return False

    element_proto = cast(WebElementProtocol, element)
    return element_proto.is_displayed()


# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


# Removed smoke test: _test_function_availability - only checked availability in globals()


def _test_safe_click_interactions() -> None:
    """Verify safe_click scrolls elements and respects guard rails."""
    from unittest.mock import MagicMock

    mock_driver = MagicMock()
    mock_element = MagicMock()

    assert safe_click(mock_driver, mock_element) is True
    mock_driver.execute_script.assert_called_once_with("arguments[0].scrollIntoView(true);", mock_element)
    mock_element.click.assert_called_once()
    assert safe_click(None, mock_element) is False
    assert safe_click(mock_driver, None) is False


def _test_force_user_agent_behavior() -> None:
    """Ensure force_user_agent updates navigator.userAgent via execute_script."""
    from unittest.mock import MagicMock

    mock_driver = MagicMock()
    assert force_user_agent(mock_driver, "test-agent") is True
    mock_driver.execute_script.assert_called_once_with("navigator.userAgent = arguments[0]", "test-agent")
    assert force_user_agent(None, "ua") is False


def _test_cookie_export_roundtrip() -> None:
    """Validate export_cookies writes driver cookies to disk."""
    import json as _json
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock

    mock_driver = MagicMock()
    cookie_payload = [{"name": "session", "value": "abc"}]
    mock_driver.get_cookies.return_value = cookie_payload

    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "cookies.json"
        assert export_cookies(mock_driver, str(target)) is True
        on_disk = _json.loads(target.read_text(encoding="utf-8"))
        assert on_disk == cookie_payload


def _test_element_helpers() -> None:
    """Exercise attribute/text extraction and visibility helpers."""
    from unittest.mock import MagicMock

    mock_element = MagicMock()
    mock_element.text = "Hello"
    mock_element.get_attribute.return_value = "https://example"
    mock_element.is_displayed.return_value = True

    assert extract_text(mock_element) == "Hello"
    assert extract_attribute(mock_element, "href") == "https://example"
    assert is_element_visible(mock_element) is True
    assert not extract_attribute(None, "href")
    assert not is_element_visible(None)


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def selenium_utils_module_tests() -> bool:
    """Run comprehensive selenium utilities tests using standardized TestSuite framework."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("Selenium Utilities & Browser Automation", "selenium_utils.py")
    suite.start_suite()

    # Assign behavior-focused module-level test functions
    test_safe_click = _test_safe_click_interactions
    test_force_user_agent = _test_force_user_agent_behavior
    test_cookie_export = _test_cookie_export_roundtrip
    test_element_helpers = _test_element_helpers

    # Define all tests in a data structure to reduce complexity
    tests = [
        # Removed smoke test: Function Availability
        (
            "Safe Click Interactions",
            test_safe_click,
            "safe_click scrolls elements and respects guard clauses",
            "Simulate clicking via MagicMock",
            "safe_click should scroll, click, and guard against missing args",
        ),
        (
            "Force User Agent",
            test_force_user_agent,
            "force_user_agent executes CDP script",
            "Mock driver execute_script calls",
            "Driver execute_script should receive navigator.userAgent override",
        ),
        (
            "Cookie Export",
            test_cookie_export,
            "export_cookies writes JSON payloads",
            "Round-trip cookies to temp file",
            "File contents should match driver.get_cookies output",
        ),
        (
            "Element Helper Functions",
            test_element_helpers,
            "Element helpers extract text/attributes and visibility",
            "Use MagicMock to cover helper routines",
            "Helpers should return data and guard against None inputs",
        ),
    ]

    with suppress_logging():
        # Run all tests from the list
        for test_name, test_func, expected, method, details in tests:
            suite.run_test(test_name, test_func, expected, method, details)

    return suite.finish_suite()


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(selenium_utils_module_tests)


if __name__ == "__main__":
    import sys

    print("🧪 Running Selenium Utils Comprehensive Tests...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
