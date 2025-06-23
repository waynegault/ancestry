"""
Selenium/WebDriver utility functions specifically for browser automation
and element interaction, separated from general or API-specific utilities.
"""

# Unified import system - consolidated from multiple sources
from core_imports import (
    standardize_module_imports,
    auto_register_module,
    register_function,
    get_function,
    is_function_available,
    safe_execute,
    get_logger,
)

# Initialize unified system
standardize_module_imports()
auto_register_module(globals(), __name__)

# Standard library imports
import time
import os
import json
import sys
from typing import Optional, Dict

# Third-party imports
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
    InvalidSessionIdException,
    NoSuchWindowException,
)
import undetected_chromedriver as uc

# Local imports
from config import config_schema

# Initialize logger
logger = get_logger(__name__)

# --- Selenium Specific Helpers ---


@safe_execute(default_return=False, log_errors=True)
def force_user_agent(driver: Optional[uc.Chrome], user_agent: str):
    """
    Attempts to force the browser's User-Agent string using Chrome DevTools Protocol.
    Now with unified error handling via safe_execute decorator.
    """
    if not driver:
        logger.warning("Driver is None, cannot set user agent")
        return False

    # Use execute_script to modify the user agent
    driver.execute_script("navigator.userAgent = arguments[0]", user_agent)
    logger.debug(f"Set user agent to: {user_agent}")
    return True


@safe_execute(default_return="", log_errors=False)
def extract_text(element) -> str:
    """Extract text from an element safely with unified error handling."""
    if not element:
        return ""
    return element.text or ""


@safe_execute(default_return="", log_errors=False)
def extract_attribute(element, attribute: str) -> str:
    """Extract attribute from an element safely with unified error handling."""
    if not element:
        return ""
    return element.get_attribute(attribute) or ""


@safe_execute(default_return=False, log_errors=False)
def is_elem_there(driver, selector: str, by: str = By.CSS_SELECTOR) -> bool:
    """Check if element exists with unified error handling."""
    if not driver:
        return False
    driver.find_element(by, selector)
    return True


@safe_execute(default_return=False, log_errors=False)
def is_browser_open(driver) -> bool:
    """Check if browser is still open and responsive with unified error handling."""
    if not driver:
        return False
    # Access current_url - will raise exception if browser is closed
    _ = driver.current_url
    return True


@safe_execute(log_errors=True)
def close_tabs(driver, keep_first: bool = True):
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
def get_driver_cookies(driver) -> list:
    """Get all cookies from driver with unified error handling."""
    if not driver:
        return []
    return driver.get_cookies()


@safe_execute(default_return=False, log_errors=True)
def export_cookies(driver, filepath: str) -> bool:
    """Export cookies to file with unified error handling."""
    if not driver:
        return False

    cookies = get_driver_cookies(driver)
    with open(filepath, "w") as f:
        json.dump(cookies, f, indent=2)
    return True


@safe_execute(log_errors=False)
def scroll_to_element(driver, element):
    """Scroll element into view with unified error handling."""
    if not driver or not element:
        return

    driver.execute_script("arguments[0].scrollIntoView(true);", element)
    time.sleep(0.1)  # Brief pause for scroll completion


@safe_execute(default_return=None, log_errors=False)
def wait_for_element(
    driver, selector: str, timeout: int = 10, by: str = By.CSS_SELECTOR
):
    """Wait for element to be present with unified error handling."""
    if not driver:
        return None

    wait = WebDriverWait(driver, timeout)
    return wait.until(EC.presence_of_element_located((by, selector)))


@safe_execute(default_return=False, log_errors=False)
def safe_click(driver, element):
    """Safely click an element with unified error handling."""
    if not driver or not element:
        return False

    # Scroll to element first
    scroll_to_element(driver, element)
    # Try to click
    element.click()
    return True


@safe_execute(default_return="", log_errors=False)
def get_element_text(element) -> str:
    """Get text from element with unified error handling."""
    if not element:
        return ""
    return element.text or ""


@safe_execute(default_return=False, log_errors=False)
def is_element_visible(element) -> bool:
    """Check if element is visible with unified error handling."""
    if not element:
        return False
    return element.is_displayed()


def run_comprehensive_tests() -> bool:
    """Test all selenium utility functions with unified testing."""
    logger.info("🔧 Testing Selenium Utilities...")

    success = True
    tests_run = 0
    tests_passed = 0

    # Basic function availability tests using unified registry
    required_functions = [
        "force_user_agent",
        "extract_text",
        "extract_attribute",
        "is_elem_there",
        "is_browser_open",
        "close_tabs",
        "get_driver_cookies",
        "export_cookies",
        "scroll_to_element",
        "wait_for_element",
        "safe_click",
        "get_element_text",
        "is_element_visible",
    ]
    for func_name in required_functions:
        tests_run += 1
        if func_name in globals() and callable(globals()[func_name]):
            tests_passed += 1
            logger.info(f"✅ {func_name} is available")
        else:
            logger.error(f"❌ {func_name} is not available")
            success = False

    # Test force_user_agent with mock driver
    tests_run += 1
    try:
        from unittest.mock import MagicMock

        mock_driver = MagicMock()
        result = force_user_agent(mock_driver, "test-agent")
        mock_driver.execute_script.assert_called_once()
        assert result == True
        logger.info("✅ force_user_agent test passed")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ force_user_agent test failed: {e}")
        success = False

    # Test text extraction functions
    tests_run += 1
    try:
        from unittest.mock import MagicMock

        mock_element = MagicMock()
        mock_element.text = "test text"
        result = get_element_text(mock_element)
        assert result == "test text"
        logger.info("✅ get_element_text test passed")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ get_element_text test failed: {e}")
        success = False

    # Test unified error handling performance
    tests_run += 1
    try:
        import time

        start_time = time.time()
        for _ in range(1000):
            # Test safe execution with None input (should handle gracefully)
            extract_text(None)
            extract_attribute(None, "href")
            is_elem_there(None, "selector")
        end_time = time.time()

        duration = end_time - start_time
        assert (
            duration < 0.5
        ), f"1000 safe operations should be fast, took {duration:.3f}s"
        logger.info(f"✅ Performance test passed (1000 ops in {duration:.3f}s)")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        success = False

    status = "PASSED" if success else "FAILED"
    logger.info(f"🎯 Selenium Utilities Tests: {status} ({tests_passed}/{tests_run})")

    return success


# Functions are automatically registered via auto_register_module() at import
# No manual registration needed - this is a key benefit of the unified system


# Register module functions at module load
auto_register_module(globals(), __name__)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)