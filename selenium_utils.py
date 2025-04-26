# selenium_utils.py
"""
Selenium/WebDriver utility functions for browser automation and scraping.
"""
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
import time
import random
import logging
from config import config_instance, selenium_config
from logging_config import logger


def force_user_agent(driver: WebDriver, user_agent: str):
    """
    Attempts to force the browser's User-Agent string using Chrome DevTools Protocol.
    """
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


def extract_text(element, selector: str) -> str:
    """
    Safely extracts text content from a Selenium WebElement using a CSS selector.
    """
    try:
        target_element = element.find_element(By.CSS_SELECTOR, selector)
        text_content = target_element.text
        return text_content.strip() if text_content else ""
    except NoSuchElementException:
        logger.debug(
            f"Element with selector '{selector}' not found for text extraction."
        )
        return ""
    except Exception as e:
        logger.warning(
            f"Error extracting text using selector '{selector}': {e}", exc_info=False
        )
        return ""


def extract_attribute(element, selector: str, attribute: str) -> str:
    """
    Extracts the value of a specified attribute from a child web element.
    Handles relative URLs for 'href' attributes, resolving them against BASE_URL.
    """
    try:
        target_element = element.find_element(By.CSS_SELECTOR, selector)
        value = target_element.get_attribute(attribute)
        if attribute == "href" and value:
            if value.startswith("/"):
                return config_instance.BASE_URL.rstrip("/") + value
            elif value.startswith("http://") or value.startswith("https://"):
                return value
            else:
                return value
        return value if value else ""
    except NoSuchElementException:
        logger.debug(
            f"Element with selector '{selector}' not found for attribute '{attribute}' extraction."
        )
        return ""
    except Exception as e:
        logger.warning(
            f"Error extracting attribute '{attribute}' from selector '{selector}': {e}",
            exc_info=False,
        )
        return ""


def is_elem_there(driver: WebDriver, by: str, value: str, wait: int = 2) -> bool:
    """
    Checks if a web element is present in the DOM within a specified timeout.
    """
    if driver is None:
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
        logger.error(f"Error checking element presence for '{value}' ({by}): {e}")
        return False


def get_tot_page(driver: WebDriver) -> int:
    """
    Retrieves the total number of pages from the pagination element on a page.
    """
    element_wait = selenium_config.element_wait(driver)
    PAGINATION_SELECTOR = getattr(
        selenium_config, "PAGINATION_SELECTOR", "pagination-selector"
    )
    logger.debug("Attempting to find pagination element and get total pages...")
    try:
        pagination_element = element_wait.until(
            lambda d: (
                (d.find_elements(By.CSS_SELECTOR, PAGINATION_SELECTOR) or [None])[0]
                if d.find_elements(By.CSS_SELECTOR, PAGINATION_SELECTOR)
                and d.find_elements(By.CSS_SELECTOR, PAGINATION_SELECTOR)[
                    0
                ].get_attribute("total")
                is not None
                else None
            )
        )
        if pagination_element:
            total_str = pagination_element.get_attribute("total")
            try:
                return int(total_str)
            except Exception:
                logger.warning(f"Could not parse total page count: {total_str}")
                return 1
        else:
            logger.debug("Pagination element not found, defaulting to 1 page.")
            return 1
    except TimeoutException:
        logger.debug(
            f"Timeout waiting for pagination element '{PAGINATION_SELECTOR}' or 'total' attribute. Assuming 1 page."
        )
        return 1
    except (NoSuchElementException, IndexError):
        logger.debug(
            f"Pagination element '{PAGINATION_SELECTOR}' not found after wait. Assuming 1 page."
        )
        return 1
    except Exception as e:
        logger.error(f"Unexpected error getting total pages: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    print("selenium_utils.py loaded successfully.")
    print("Available functions:")
    print(" - force_user_agent(driver, user_agent)")
    print(" - extract_text(element, selector)")
    print(" - extract_attribute(element, selector, attribute)")
    print(" - is_elem_there(driver, by, value, wait=2)")
    print(" - get_tot_page(driver)")
    print(
        "\nThis is a utility module for Selenium/WebDriver helpers. No standalone test is run."
    )
