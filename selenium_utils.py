# selenium_utils.py
"""
Selenium/WebDriver utility functions specifically for browser automation
and element interaction, separated from general or API-specific utilities.
"""

# --- Standard library imports ---
import time
import random
import logging
import os  # Keep os for path manipulation if needed
from typing import Optional, Dict  # Import Optional and Dict for type hints

# --- Third-party imports ---
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
    InvalidSessionIdException,  # Added
    NoSuchWindowException,  # Added
)

# --- Local application imports ---
from config import config_instance, selenium_config
from logging_config import logger

# Note: Removed urllib3 and psutil imports as they weren't used here

# --- Selenium Specific Helpers ---


def force_user_agent(driver: WebDriver, user_agent: str):
    """
    Attempts to force the browser's User-Agent string using Chrome DevTools Protocol.
    """
    if not driver:
        logger.warning("force_user_agent: WebDriver is None.")
        return
    logger.debug(f"Attempting to set User-Agent via CDP to: {user_agent}")
    start_time = time.time()
    try:
        driver.execute_cdp_cmd(
            "Network.setUserAgentOverride", {"userAgent": user_agent}
        )
        duration = time.time() - start_time
        logger.info(f"Successfully set User-Agent via CDP in {duration:.3f} seconds.")
    except Exception as e:
        logger.warning(f"Error setting User-Agent via CDP: {e}", exc_info=True)


# End of force_user_agent


def extract_text(element, selector: str) -> str:
    """
    Safely extracts text content from a Selenium WebElement using a CSS selector.
    """
    if not element:
        logger.warning("extract_text: Parent element is None.")
        return ""
    try:
        target_element = element.find_element(By.CSS_SELECTOR, selector)
        text_content = target_element.text
        return text_content.strip() if text_content else ""
    except NoSuchElementException:
        logger.debug(f"Element '{selector}' not found for text extraction.")
        return ""
    except Exception as e:
        logger.warning(f"Error extracting text '{selector}': {e}", exc_info=False)
        return ""


# End of extract_text


def extract_attribute(element, selector: str, attribute: str) -> str:
    """
    Extracts attribute from a child element. Resolves relative 'href'.
    """
    if not element:
        logger.warning("extract_attribute: Parent element is None.")
        return ""
    try:
        target_element = element.find_element(By.CSS_SELECTOR, selector)
        value = target_element.get_attribute(attribute)
        if attribute == "href" and value:
            # Use urljoin from urllib.parse (needs import in calling module or here)
            from urllib.parse import urljoin  # Local import for utility

            if value.startswith("/"):
                return urljoin(config_instance.BASE_URL, value)
            # Keep absolute and other protocols as is
            elif (
                value.startswith("http://")
                or value.startswith("https://")
                or ":" in value.split("/")[0]
            ):
                return value
            else:  # Assume relative path without leading slash (less common)
                # This might need context (current URL) for perfect resolution,
                # but joining with base is a reasonable default.
                return urljoin(config_instance.BASE_URL, value)
        return value if value else ""
    except NoSuchElementException:
        logger.debug(f"Element '{selector}' not found for attribute '{attribute}'.")
        return ""
    except Exception as e:
        logger.warning(
            f"Error extracting attr '{attribute}' from '{selector}': {e}",
            exc_info=False,
        )
        return ""


# End of extract_attribute


def is_elem_there(
    driver: WebDriver, by: str, value: str, wait: Optional[int] = None
) -> bool:
    """
    Checks if a web element is present in the DOM within a specified timeout.
    """
    if driver is None:
        logger.warning("is_elem_there: WebDriver is None.")
        return False
    effective_wait = wait if wait is not None else selenium_config.ELEMENT_TIMEOUT
    try:
        WebDriverWait(driver, effective_wait).until(
            EC.presence_of_element_located((by, value))
        )
        return True
    except TimeoutException:
        return False
    except Exception as e:
        logger.error(f"Error checking element presence '{value}' ({by}): {e}")
        return False


# End of is_elem_there


def is_browser_open(driver: Optional[WebDriver]) -> bool:
    """
    Checks if the browser window appears open and responsive.
    """
    if driver is None:
        return False
    try:
        _ = driver.window_handles  # Lightweight command
        return True
    except (InvalidSessionIdException, NoSuchWindowException, WebDriverException) as e:
        err_str = str(e).lower()
        if any(
            sub in err_str
            for sub in [
                "invalid session id",
                "target closed",
                "disconnected",
                "no such window",
                "unable to connect",
            ]
        ):
            logger.debug(
                f"Browser appears closed or session invalid: {type(e).__name__}"
            )
            return False
        else:
            logger.warning(
                f"WebDriverException checking browser status (assuming closed): {e}"
            )
            return False
    except Exception as e:
        logger.error(f"Unexpected error checking browser status: {e}", exc_info=True)
        return False


# End of is_browser_open


def close_tabs(driver: WebDriver):
    """
    Closes all browser tabs except the first one. Focuses the first tab.
    """
    if not driver:
        logger.warning("close_tabs: WebDriver instance is None.")
        return
    logger.debug("Closing extra browser tabs...")
    try:
        handles = driver.window_handles
        if len(handles) <= 1:
            logger.debug("No extra tabs to close.")
            return
        original_handle = driver.current_window_handle
        first_handle = handles[0]
        logger.debug(
            f"Original handle: {original_handle}, First handle: {first_handle}"
        )
        closed_count = 0
        for handle in handles[1:]:
            try:
                logger.debug(f"Switching to tab {handle} to close...")
                driver.switch_to.window(handle)
                driver.close()
                logger.debug(f"Closed tab handle: {handle}")
                closed_count += 1
            except NoSuchWindowException:
                logger.warning(f"Tab {handle} already closed.")
            except WebDriverException as e:
                logger.error(f"Error closing tab {handle}: {e}")
        logger.debug(f"Closed {closed_count} extra tabs.")
        remaining_handles = driver.window_handles
        if original_handle in remaining_handles:
            if driver.current_window_handle != original_handle:
                logger.debug(f"Switching back to original tab: {original_handle}")
                driver.switch_to.window(original_handle)
            else:
                logger.debug("Already focused on original tab.")
        elif first_handle in remaining_handles:
            logger.warning(
                f"Original handle {original_handle} missing. Switching to first handle: {first_handle}"
            )
            driver.switch_to.window(first_handle)
        elif remaining_handles:
            logger.error(
                f"Original and first tabs gone. Switching to remaining: {remaining_handles[0]}"
            )
            driver.switch_to.window(remaining_handles[0])
        else:
            logger.error("All browser tabs were closed unexpectedly.")
    except NoSuchWindowException:
        logger.warning("Attempted to close/switch tab that no longer exists.")
    except WebDriverException as e:
        logger.error(f"WebDriverException during close_tabs: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in close_tabs: {e}", exc_info=True)


# End of close_tabs


def get_driver_cookies(driver: WebDriver) -> Dict[str, str]:
    """Retrieves all cookies as a simple dictionary."""
    if not driver:
        logger.warning("Cannot get driver cookies: WebDriver is None.")
        return {}
    try:
        cookies_list = driver.get_cookies()
        return {
            cookie["name"]: cookie["value"]
            for cookie in cookies_list
            if "name" in cookie
        }
    except WebDriverException as e:
        logger.error(f"WebDriverException getting driver cookies: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error getting driver cookies: {e}", exc_info=True)
        return {}


# End of get_driver_cookies


if __name__ == "__main__":
    print("This is the selenium_utils module. Import it into other scripts.")
    logger.info("selenium_utils loaded.")
    # Add basic instantiation tests if needed
# End of selenium_utils.py
