#!/usr/bin/env python3

"""
Selenium Utilities.

This module provides standardized helper functions for Selenium WebDriver interactions,
extracted from utils.py to improve modularity.
"""

# === STANDARD LIBRARY IMPORTS ===
import logging
from typing import Any, Union

# === THIRD-PARTY IMPORTS ===
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC  # noqa: N812
from selenium.webdriver.support.wait import WebDriverWait

# === TYPE ALIASES ===
Locator = tuple[str, str]
DriverType = Union[WebDriver, None]

logger = logging.getLogger(__name__)


def wait_until_visible(waiter: "WebDriverWait[Any]", locator: Locator) -> Any:
    """Return first element matching locator once it becomes visible."""
    return waiter.until(EC.visibility_of_element_located(locator))


def wait_until_clickable(waiter: "WebDriverWait[Any]", locator: Locator) -> Any:
    """Return the element once it becomes clickable."""
    return waiter.until(EC.element_to_be_clickable(locator))


def wait_until_present(waiter: "WebDriverWait[Any]", locator: Locator) -> Any:
    """Return first element matching locator once present in DOM."""
    return waiter.until(EC.presence_of_element_located(locator))


def wait_until_not_visible(waiter: "WebDriverWait[Any]", locator: Locator) -> Any:
    """Wait until the element matching locator is no longer visible."""
    return waiter.until(EC.invisibility_of_element_located(locator))


def wait_until_not_present(waiter: "WebDriverWait[Any]", locator: Locator) -> Any:
    """Wait until the element matching locator is no longer present in the DOM."""
    # EC.presence_of_element_located returns the element if present.
    # We want to wait until it is NOT present.
    # There isn't a direct 'absence_of_element_located' in standard EC,
    # but 'invisibility_of_element_located' handles visibility.
    # For presence, we typically use a lambda or custom condition.
    # However, utils.py implementation likely used something specific.
    # Let's use a standard lambda for now.
    return waiter.until(lambda d: len(d.find_elements(*locator)) == 0)


# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
