#!/usr/bin/env python3

# utils.py

"""
utils.py - Standalone login script + utilities using config variables.

This script utilizes configuration variables from config.py (loaded via
config_instance and selenium_config) throughout the login functions and
utility functions, enhancing configurability and maintainability.
It includes core session management, API request helpers, browser interaction
utilities, and data structures.
"""

# --- Standard library imports ---
import base64
import contextlib
import inspect
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from urllib.parse import urljoin, urlparse, unquote

# --- Third-party imports ---
import cloudscraper
import requests
import undetected_chromedriver as uc
from requests import Request, Response as RequestsResponse
from requests.adapters import HTTPAdapter
from requests.cookies import RequestsCookieJar
from requests.exceptions import HTTPError, RequestException
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    InvalidSessionIdException,
    NoSuchCookieException,
    NoSuchElementException,
    NoSuchWindowException,
    StaleElementReferenceException,
    TimeoutException,
    UnexpectedAlertPresentException,
    WebDriverException,
)
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from sqlalchemy import create_engine, event, pool as sqlalchemy_pool, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from urllib3.util.retry import Retry

# --- Local application imports ---
from cache import cache as global_cache, cache_result
from chromedriver import init_webdvr
from config import config_instance, selenium_config
from database import Base, ConversationLog, DnaMatch, FamilyTree, MessageType, Person
from logging_config import logger, setup_logging
from my_selectors import *

# --- Constants ---
# Error status strings for _send_message_via_api
SEND_ERROR_INVALID_RECIPIENT = "send_error (invalid_recipient)"
SEND_ERROR_MISSING_OWN_ID = "send_error (missing_own_id)"
SEND_ERROR_INTERNAL_MODE = "send_error (internal_mode_error)"
SEND_ERROR_API_PREP_FAILED = "send_error (api_prep_failed)"
SEND_ERROR_UNEXPECTED_FORMAT = "send_error (unexpected_format)"
SEND_ERROR_VALIDATION_FAILED = "send_error (validation_failed)"
SEND_ERROR_POST_FAILED = "send_error (post_failed)"
SEND_ERROR_UNKNOWN = "send_error (unknown)"
SEND_SUCCESS_DELIVERED = "delivered OK"
SEND_SUCCESS_DRY_RUN = "typed (dry_run)"
# ------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------


def force_user_agent(driver: WebDriver, user_agent: str):
    """
    Attempts to force the browser's User-Agent string using Chrome DevTools Protocol.

    Note: Relies on CDP command which might be less stable across browser versions
          compared to standard WebDriver methods. Use with awareness.

    Args:
        driver: The Selenium WebDriver instance (must support CDP).
        user_agent: The desired User-Agent string.
    """
    # Step 1: Log the attempt
    logger.debug(f"Attempting to set User-Agent via CDP to: {user_agent}")
    start_time = time.time()

    # Step 2: Execute the CDP command
    try:
        logger.debug("Calling execute_cdp_cmd Network.setUserAgentOverride...")
        driver.execute_cdp_cmd(
            "Network.setUserAgentOverride", {"userAgent": user_agent}
        )
        logger.debug("execute_cdp_cmd call returned.")
        duration = time.time() - start_time
        logger.info(f"Successfully set User-Agent via CDP in {duration:.3f} seconds.")
    except Exception as e:
        # Step 3: Log any errors during the process
        logger.warning(f"Error setting User-Agent via CDP: {e}", exc_info=True)


# End of force_user_agent


def parse_cookie(cookie_string: str) -> Dict[str, str]:
    """
    Parses a raw HTTP cookie string into a dictionary of key-value pairs.

    Args:
        cookie_string: The raw cookie string (e.g., "key1=value1; key2=value2").

    Returns:
        A dictionary representing the cookies.
    """
    # Step 1: Initialize cookies dictionary
    cookies: Dict[str, str] = {}
    # Step 2: Split the string by semicolon
    parts = cookie_string.split(";")

    # Step 3: Iterate through each part
    for part in parts:
        part = part.strip()
        # Step 3a: Skip empty parts
        if not part:
            continue
        # Step 3b: Ensure '=' exists and split only at the first '='
        if "=" in part:
            key_value_pair = part.split("=", 1)
            if len(key_value_pair) == 2:
                key, value = key_value_pair
                # Step 3c: Add valid key-value pair to dictionary
                cookies[key] = value
            else:
                logger.debug(f"Skipping invalid cookie part (split error): '{part}'")
        else:
            logger.debug(f"Skipping invalid cookie part (no '='): '{part}'")

    # Step 4: Return the parsed cookies
    return cookies


# End of parse_cookie


def extract_text(element: Any, selector: str) -> str:
    """
    Safely extracts text content from a Selenium WebElement using a CSS selector.

    Args:
        element: The parent Selenium WebElement to search within.
        selector: The CSS selector to locate the target child element.

    Returns:
        The stripped text content of the target element, or an empty string
        if the element is not found or has no text.
    """
    # Step 1: Try finding the element and getting its text
    try:
        target_element = element.find_element(By.CSS_SELECTOR, selector)
        text_content = target_element.text
        # Step 2: Return stripped text or empty string
        return text_content.strip() if text_content else ""
    # Step 3: Handle case where element is not found
    except NoSuchElementException:
        logger.debug(
            f"Element with selector '{selector}' not found for text extraction."
        )
        return ""
    # Step 4: Handle other potential errors during extraction
    except Exception as e:
        logger.warning(
            f"Error extracting text using selector '{selector}': {e}", exc_info=False
        )
        return ""


# End of extract_text


def extract_attribute(element: Any, selector: str, attribute: str) -> str:
    """
    Extracts the value of a specified attribute from a child web element.
    Handles relative URLs for 'href' attributes, resolving them against BASE_URL.

    Args:
        element: The parent Selenium WebElement to search within.
        selector: The CSS selector to locate the target child element.
        attribute: The name of the attribute to extract (e.g., 'href', 'src', 'class').

    Returns:
        The value of the attribute, or an empty string if the element or
        attribute is not found, or if the attribute value is empty.
        Returns a fully resolved URL for relative 'href' attributes.
    """
    # Step 1: Try finding the element and getting the attribute
    try:
        target_element = element.find_element(By.CSS_SELECTOR, selector)
        value = target_element.get_attribute(attribute)

        # Step 2: Handle 'href' attributes specifically for URL resolution
        if attribute == "href" and value:
            if value.startswith("/"):
                # Resolve relative URL against the application's base URL
                return urljoin(config_instance.BASE_URL, value)
            elif value.startswith("http://") or value.startswith("https://"):
                # Return absolute URL directly
                return value
            else:
                # Return other 'href' values (like 'mailto:', 'javascript:') as is
                return value
        # Step 3: Return other attribute values or empty string if None/empty
        return value if value else ""

    # Step 4: Handle case where element is not found
    except NoSuchElementException:
        logger.debug(
            f"Element with selector '{selector}' not found for attribute '{attribute}' extraction."
        )
        return ""
    # Step 5: Handle other potential errors during extraction
    except Exception as e:
        logger.warning(
            f"Error extracting attribute '{attribute}' from selector '{selector}': {e}",
            exc_info=False,
        )
        return ""


# End of extract_attribute


def get_tot_page(driver: WebDriver) -> int:
    """
     Retrieves the total number of pages from the pagination element on a page.

    Args:
        driver: The Selenium WebDriver instance.

    Returns:
        The total number of pages as an integer, or 1 if the pagination element
        or its 'total' attribute cannot be found or parsed.
    """
    # Step 1: Initialize WebDriverWait
    element_wait = selenium_config.element_wait(driver)
    logger.debug("Attempting to find pagination element and get total pages...")

    # Step 2: Wait for the pagination element and its 'total' attribute
    try:
        # Use a lambda function to wait until the element exists AND has the 'total' attribute
        # This avoids potential StaleElementReferenceExceptions
        pagination_element = element_wait.until(
            lambda d: (
                elem[0]  # Return the first element if found
                if (
                    # Find elements matching the selector
                    elem := d.find_elements(By.CSS_SELECTOR, PAGINATION_SELECTOR)
                    # Check if list is not empty and first element has 'total' attribute
                    and elem
                    and elem[0].get_attribute("total") is not None
                )
                else None  # Return None if conditions not met, causing WebDriverWait to retry
            )
        )

        # Step 3: Extract and parse the 'total' attribute if element found
        if pagination_element:
            total_pages_str = pagination_element.get_attribute("total")
            try:
                total_pages = int(total_pages_str)
                logger.debug(f"Found total pages: {total_pages}")
                return total_pages
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Error converting pagination 'total' attribute ('{total_pages_str}') to int: {e}"
                )
                return 1  # Return 1 if conversion fails
        else:
            # This case should technically not be reached if wait succeeds, but included for safety
            logger.debug("Pagination element found by lambda but became None?")
            return 1

    # Step 4: Handle exceptions during the wait or processing
    except TimeoutException:
        logger.debug(
            f"Timeout waiting for pagination element '{PAGINATION_SELECTOR}' or 'total' attribute. Assuming 1 page."
        )
        return 1
    except (NoSuchElementException, IndexError):
        # Although covered by wait, catch explicitly in case element disappears after wait
        logger.debug(
            f"Pagination element '{PAGINATION_SELECTOR}' not found after wait. Assuming 1 page."
        )
        return 1
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error getting total pages: {e}", exc_info=True)
        return 1


# End of get_tot_page


def ordinal_case(text: str) -> str:
    """
    Corrects ordinal suffixes (1st, 2nd, 3rd, 4th) to lowercase within a string,
    often used after applying title casing.

    Example: "1St Cousin" becomes "1st Cousin".

    Args:
        text: The input string.

    Returns:
        The string with ordinal suffixes corrected to lowercase.
    """
    # Step 1: Check for empty input
    if not text:
        return text

    # Step 2: Define a helper function for regex substitution
    def lower_suffix(match: re.Match) -> str:
        """Converts the suffix part (group 2) of the match to lowercase."""
        return match.group(1) + match.group(2).lower()

    # End of lower_suffix

    # Step 3: Use regex to find numbers followed by ordinal suffixes (case-insensitive)
    # and apply the lower_suffix function to each match.
    # Pattern: (\d+) matches one or more digits (captured in group 1)
    #          (st|nd|rd|th) matches the ordinal suffix (case-insensitive, captured in group 2)
    return re.sub(r"(\d+)(st|nd|rd|th)", lower_suffix, text, flags=re.IGNORECASE)


# End of ordinal_case


def is_elem_there(
    driver: WebDriver, by: By, value: str, wait: Optional[int] = None
) -> bool:
    """
    Checks if a web element is present in the DOM within a specified timeout.

    Args:
        driver: The Selenium WebDriver instance.
        by: The Selenium locator strategy (e.g., By.ID, By.CSS_SELECTOR).
        value: The locator value (e.g., 'element-id', '.element-class').
        wait: Optional timeout in seconds. Uses selenium_config.ELEMENT_TIMEOUT
              if None.

    Returns:
        True if the element is present within the timeout, False otherwise.
    """
    # Step 1: Determine the wait timeout
    effective_wait = wait if wait is not None else selenium_config.ELEMENT_TIMEOUT

    # Step 2: Attempt to wait for the element's presence
    try:
        WebDriverWait(driver, effective_wait).until(
            EC.presence_of_element_located((by, value))
        )
        # Step 3: Return True if found
        return True
    # Step 4: Handle timeout exception
    except TimeoutException:
        # Optional: Log timeout (can be verbose)
        # logger.debug(f"Timeout ({effective_wait}s) waiting for element: {by}='{value}'")
        return False
    # Step 5: Handle other potential exceptions during the wait
    except Exception as e:
        logger.error(f"Error checking element presence for '{value}' ({by}): {e}")
        return False


# End of is_elem_there


def format_name(name: Optional[str]) -> str:
    """
    Formats a person's name string to title case, preserving uppercase components
    (like initials or acronyms) and handling None/empty input gracefully.

    Args:
        name: The raw name string.

    Returns:
        The formatted name string, or "Valued Relative" as a fallback default.
    """
    # Step 1: Handle None or empty input
    if not name:
        return "Valued Relative"

    # Step 2: Attempt formatting
    try:
        parts = name.split()
        formatted_parts = []
        # Step 3: Process each part of the name
        for part in parts:
            # Preserve parts that are all uppercase (e.g., "JR", "A", "PhD")
            if part.isupper():
                formatted_parts.append(part)
            # Apply title case to other parts
            else:
                # Use title() method which handles cases like "McHardy" correctly
                formatted_parts.append(part.title())
        # Step 4: Join the parts back together
        return " ".join(formatted_parts)
    # Step 5: Handle potential errors during formatting
    except Exception as e:
        logger.error(f"Error formatting name '{name}': {e}", exc_info=False)
        # Return a safe default on error
        return "Valued Relative"


# End of format_name


@dataclass
class MatchData:
    """
    Represents the collected data for a single DNA match, aggregating information
    from various sources (match list, APIs, tree details).
    """

    # --- Core Identifiers (Usually from initial match list) ---
    guid: str  # Match's test GUID (e.g., 6EAC8EC1-...)
    display_name: str  # Match's display name (e.g., "Frances Mchardy")
    image_url: Optional[str] = None  # URL of the match's profile image

    # --- DNA Match Details (From initial list or details API) ---
    shared_cm: Optional[float] = None  # Shared Centimorgans
    shared_segments: Optional[int] = None  # Number of shared segments
    last_login_str: Optional[str] = (
        None  # Raw string like "Logged in today", "Sep 2023"
    )

    # --- Predicted Relationship (From matchProbabilityData API) ---
    confidence: Optional[float] = None  # Probability score from API
    predicted_relationship: Optional[str] = None  # e.g., "1st–2nd cousin"

    # --- Links (Constructed or from list) ---
    compare_link: Optional[str] = None  # Link to comparison page
    message_link: Optional[str] = None  # Link to message the match

    # --- Tree Related Info ---
    in_my_tree: bool = field(
        default=False
    )  # Set after checking matchesInTree API or list hint

    # --- Details Primarily from Badge/Ladder APIs (if in_my_tree=True) ---
    match_tree_id: Optional[str] = None  # The ID of *their* linked tree (if available)
    cfpid: Optional[str] = (
        None  # The person ID (PID) within *their* tree (if available)
    )
    view_in_tree_link: Optional[str] = (
        None  # Link to view them in *my* tree (constructed)
    )
    facts_link: Optional[str] = None  # Link to facts page in *my* tree (constructed)
    relationship_path: Optional[str] = (
        None  # Raw HTML or processed path from ladder API
    )
    actual_relationship: Optional[str] = (
        None  # e.g., "Mother", "1st cousin 1x removed" (from ladder API)
    )


# End of MatchData class


# ------------------------------
# Decorators
# ------------------------------


def retry(
    MAX_RETRIES: Optional[int] = None,
    BACKOFF_FACTOR: Optional[float] = None,
    MAX_DELAY: Optional[float] = None,
):
    """
    Decorator factory to retry a function with exponential backoff and jitter.
    Uses defaults from config_instance if parameters are not provided.

    Args:
        MAX_RETRIES: Maximum number of retry attempts.
        BACKOFF_FACTOR: Multiplier for the delay between retries.
        MAX_DELAY: Maximum delay in seconds between retries.

    Returns:
        A decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Step 1: Get effective retry parameters from decorator args or config
            attempts = (
                config_instance.MAX_RETRIES if MAX_RETRIES is None else MAX_RETRIES
            )
            backoff = (
                config_instance.BACKOFF_FACTOR
                if BACKOFF_FACTOR is None
                else BACKOFF_FACTOR
            )
            max_d = config_instance.MAX_DELAY if MAX_DELAY is None else MAX_DELAY

            # Step 2: Loop through attempts
            for i in range(attempts):
                try:
                    # Step 3: Execute the decorated function
                    return func(*args, **kwargs)
                except Exception as e:
                    # Step 4: Handle exception and check if max retries reached
                    if i == attempts - 1:
                        logger.error(
                            f"All {attempts} attempts failed for function '{func.__name__}'. Final error: {e}"
                        )
                        raise  # Re-raise the last exception
                    # Step 5: Calculate sleep time with backoff and jitter
                    # Exponential backoff: backoff * (2 ** i)
                    # Cap delay at max_d
                    # Add random jitter (0 to 1 second)
                    sleep_time = min(backoff * (2**i), max_d) + random.uniform(0, 1)
                    logger.warning(
                        f"Attempt {i + 1}/{attempts} for function '{func.__name__}' failed: {e}. Retrying in {sleep_time:.2f}s..."
                    )
                    # Step 6: Wait before retrying
                    time.sleep(sleep_time)

            # This part should ideally not be reached if attempts > 0
            logger.error(
                f"Function '{func.__name__}' failed after all {attempts} retries (exited loop unexpectedly)."
            )
            return None  # Or raise a specific MaxRetriesExceeded error

        # End of wrapper
        return wrapper

    # End of decorator
    return decorator


# End of retry


def retry_api(
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    retry_on_exceptions: Tuple[Type[Exception], ...] = (
        requests.exceptions.RequestException,
    ),
    retry_on_status_codes: Optional[List[int]] = None,
):
    """
    Decorator factory for retrying API calls with exponential backoff, logging,
    and optional retries based on specific HTTP status codes or exceptions.
    Uses defaults from config_instance if arguments are None.

    Args:
        max_retries: Max retry attempts.
        initial_delay: Initial delay before first retry.
        backoff_factor: Multiplier for delay increase.
        retry_on_exceptions: Tuple of Exception types to retry on.
        retry_on_status_codes: List of HTTP status codes to retry on.

    Returns:
        A decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Step 1: Determine effective retry parameters from args or config
            _max_retries = (
                config_instance.MAX_RETRIES if max_retries is None else max_retries
            )
            _initial_delay = (
                config_instance.INITIAL_DELAY
                if initial_delay is None
                else initial_delay
            )
            _backoff_factor = (
                config_instance.BACKOFF_FACTOR
                if backoff_factor is None
                else backoff_factor
            )
            _retry_codes_tuple = (
                config_instance.RETRY_STATUS_CODES
                if retry_on_status_codes is None
                else tuple(retry_on_status_codes)
            )
            _retry_codes_set = set(_retry_codes_tuple)  # Use set for efficient lookup

            # Step 2: Initialize retry state variables
            retries = _max_retries
            delay = _initial_delay
            attempt = 0

            # Step 3: Retry loop
            while retries > 0:
                attempt += 1
                try:
                    # Step 4: Call the decorated API function
                    response = func(*args, **kwargs)

                    # Step 5: Check for status code retry condition
                    should_retry_status = False
                    status_code = None  # Initialize status_code
                    if response is not None and hasattr(response, "status_code"):
                        status_code = response.status_code
                        if status_code in _retry_codes_set:
                            should_retry_status = True

                    if should_retry_status:
                        retries -= 1
                        if retries <= 0:
                            logger.error(
                                f"API Call failed after {_max_retries} retries for '{func.__name__}' (Final Status {status_code}). No more retries."
                            )
                            # Return the last response even if it's a retryable error
                            return response
                        else:
                            # Calculate sleep time with jitter
                            base_sleep = delay * (_backoff_factor ** (attempt - 1))
                            jitter = (
                                random.uniform(-0.1, 0.1) * base_sleep
                            )  # Jitter relative to sleep time
                            sleep_time = min(
                                base_sleep + jitter, config_instance.MAX_DELAY
                            )
                            sleep_time = max(0.1, sleep_time)  # Ensure minimum sleep

                            logger.warning(
                                f"API Call returned status {status_code} (Attempt {attempt}/{_max_retries}) for '{func.__name__}'. Retrying in {sleep_time:.2f}s..."
                            )
                            time.sleep(sleep_time)
                            delay *= _backoff_factor  # Increase delay for next potential retry
                            continue  # Go to the next attempt

                    # Step 6: If status code doesn't trigger retry, return the successful response
                    return response

                # Step 7: Handle exception-based retry condition
                except retry_on_exceptions as e:
                    retries -= 1
                    if retries <= 0:
                        logger.error(
                            f"API Call failed after {_max_retries} retries due to exception for '{func.__name__}'. Final Exception: {type(e).__name__} - {e}",
                            exc_info=False,  # Less verbose for retry failures
                        )
                        raise e  # Re-raise the last exception
                    else:
                        # Calculate sleep time with jitter
                        base_sleep = delay * (_backoff_factor ** (attempt - 1))
                        jitter = random.uniform(-0.1, 0.1) * base_sleep
                        sleep_time = min(base_sleep + jitter, config_instance.MAX_DELAY)
                        sleep_time = max(0.1, sleep_time)

                        logger.warning(
                            f"API Call failed (Attempt {attempt}/{_max_retries}) for '{func.__name__}', retrying in {sleep_time:.2f}s... Exception: {type(e).__name__} - {e}"
                        )
                        time.sleep(sleep_time)
                        delay *= (
                            _backoff_factor  # Increase delay for next potential retry
                        )
                        continue  # Go to the next attempt

                # Step 8: Handle non-retryable exceptions (they propagate out naturally)

            # Should not be reached unless initial retries = 0
            logger.error(f"Exited retry loop unexpectedly for '{func.__name__}'.")
            return None

        # End of wrapper
        return wrapper

    # End of decorator
    return decorator


# End of retry_api


def ensure_browser_open(func: Callable) -> Callable:
    """
    Decorator to ensure the browser controlled by the SessionManager is open
    and the session is valid before executing the decorated function.

    Relies on the SessionManager instance being passed as the first argument
    or as a keyword argument named 'session_manager'.

    Raises:
        TypeError: If SessionManager instance is not found in arguments.
        WebDriverException: If the browser session is not open or invalid.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Step 1: Find the SessionManager instance
        session_manager_instance: Optional[SessionManager] = None
        if args and isinstance(args[0], SessionManager):
            session_manager_instance = args[0]
        elif "session_manager" in kwargs and isinstance(
            kwargs["session_manager"], SessionManager
        ):
            session_manager_instance = kwargs["session_manager"]

        # Step 2: Validate SessionManager presence
        if not session_manager_instance:
            raise TypeError(
                f"Function '{func.__name__}' requires a SessionManager instance "
                "as the first argument or kwarg 'session_manager'."
            )

        # Step 3: Check if the browser session is valid
        if not is_browser_open(session_manager_instance.driver):
            raise WebDriverException(
                f"Browser session invalid/closed when calling function '{func.__name__}'"
            )

        # Step 4: Execute the decorated function
        return func(*args, **kwargs)

    # End of wrapper
    return wrapper


# End of ensure_browser_open


def time_wait(wait_description: str) -> Callable:
    """
    Decorator factory to time Selenium WebDriverWait calls and log duration.

    Args:
        wait_description: A descriptive string for the wait being timed.

    Returns:
        A decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Step 1: Record start time
            start_time = time.time()
            try:
                # Step 2: Execute the decorated wait function
                result = func(*args, **kwargs)
                # Step 3: Calculate and log success duration
                end_time = time.time()
                duration = end_time - start_time
                logger.debug(
                    f"Wait '{wait_description}' completed successfully in {duration:.3f}s."
                )
                return result
            # Step 4: Handle TimeoutException
            except TimeoutException as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.warning(
                    f"Wait '{wait_description}' timed out after {duration:.3f} seconds.",
                    exc_info=False,  # Less verbose log for timeouts
                )
                raise e  # Re-raise the exception
            # Step 5: Handle other exceptions
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(
                    f"Error during wait '{wait_description}' after {duration:.3f} seconds: {e}",
                    exc_info=True,  # Log full traceback for unexpected errors
                )
                raise e  # Re-raise the exception

        # End of wrapper
        return wrapper

    # End of decorator
    return decorator


# End of time_wait


# ------------------------------
# Rate Limiting
# ------------------------------


class DynamicRateLimiter:
    """
    Implements rate limiting using a token bucket algorithm combined with
    dynamic delay adjustment based on API feedback (e.g., 429 errors).

    Manages both the average rate (via tokens) and introduces adaptive delays
    to back off during throttling and speed up when possible.
    """

    def __init__(
        self,
        initial_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        backoff_factor: Optional[float] = None,
        decrease_factor: Optional[float] = None,
        token_capacity: Optional[float] = None,
        token_fill_rate: Optional[float] = None,
        config_instance=config_instance,  # Pass config instance for defaults
    ):
        """
        Initializes the rate limiter. Uses defaults from config_instance if parameters are None.
        """
        # Step 1: Initialize Dynamic Delay Parameters (from args or config)
        self.initial_delay = (
            config_instance.INITIAL_DELAY if initial_delay is None else initial_delay
        )
        self.MAX_DELAY = config_instance.MAX_DELAY if max_delay is None else max_delay
        self.backoff_factor = (
            config_instance.BACKOFF_FACTOR if backoff_factor is None else backoff_factor
        )
        self.decrease_factor = (
            config_instance.DECREASE_FACTOR
            if decrease_factor is None
            else decrease_factor
        )
        self.current_delay = self.initial_delay  # Start with the initial base delay
        self.last_throttled = False  # Flag to track recent throttling

        # Step 2: Initialize Token Bucket Parameters (from args or config)
        self.capacity = float(
            config_instance.TOKEN_BUCKET_CAPACITY
            if token_capacity is None
            else token_capacity
        )
        self.fill_rate = float(
            config_instance.TOKEN_BUCKET_FILL_RATE
            if token_fill_rate is None
            else token_fill_rate
        )
        # Ensure fill_rate is positive to avoid division by zero or infinite waits
        if self.fill_rate <= 0:
            logger.warning(
                f"Token fill rate ({self.fill_rate}) must be positive. Setting to 1.0."
            )
            self.fill_rate = 1.0
        self.tokens = float(self.capacity)  # Start with a full bucket
        self.last_refill_time = time.monotonic()  # Use monotonic clock for accuracy

        # Step 3: Log initialization parameters
        logger.debug(
            f"RateLimiter Init: Capacity={self.capacity:.1f}, FillRate={self.fill_rate:.1f}/s, "
            f"InitialDelay={self.initial_delay:.2f}s, Backoff={self.backoff_factor:.2f}, Decrease={self.decrease_factor:.2f}"
        )

    # End of __init__

    def _refill_tokens(self):
        """Internal method to refills tokens based on elapsed time."""
        # Step 1: Calculate elapsed time since last refill
        now = time.monotonic()
        elapsed = max(0, now - self.last_refill_time)  # Ensure non-negative time

        # Step 2: Calculate tokens to add based on fill rate
        tokens_to_add = elapsed * self.fill_rate

        # Step 3: Update token count, capped at capacity
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)

        # Step 4: Update last refill time
        self.last_refill_time = now
        # Optional: Debug log for refill details
        # logger.debug(f"Refilled tokens. Current: {self.tokens:.2f}/{self.capacity:.1f} (Added: {tokens_to_add:.2f})")

    # End of _refill_tokens

    def wait(self) -> float:
        """
        Waits if necessary based on token availability and the current dynamic delay.
        Consumes one token if available, otherwise waits until a token is generated.
        Applies jitter to the wait time.

        Returns:
            The actual time slept in seconds.
        """
        # Step 1: Refill tokens based on time passed
        self._refill_tokens()
        requested_at = time.monotonic()  # Record time before potential wait
        sleep_duration = 0.0

        # Step 2: Check if a token is available
        if self.tokens >= 1.0:
            # Step 2a: Token available - Consume one token
            self.tokens -= 1.0
            # Apply base dynamic delay + jitter
            jitter_factor = random.uniform(0.8, 1.2)  # Standard +/- 20% jitter
            base_sleep = self.current_delay
            sleep_duration = min(
                base_sleep * jitter_factor, self.MAX_DELAY
            )  # Apply jitter and cap
            sleep_duration = max(0.01, sleep_duration)  # Ensure minimum sleep
            logger.debug(
                f"Token available ({self.tokens:.2f} left). Applying base delay: {sleep_duration:.3f}s (CurrentDelay: {self.current_delay:.2f}s)"
            )
        else:
            # Step 2b: Token not available - Calculate wait time for one token
            wait_needed = (1.0 - self.tokens) / self.fill_rate
            # Add a smaller positive jitter to the token wait time itself
            jitter_amount = random.uniform(0.0, 0.2)  # Add 0-0.2s jitter
            sleep_duration = wait_needed + jitter_amount
            sleep_duration = min(sleep_duration, self.MAX_DELAY)  # Cap total wait
            sleep_duration = max(0.01, sleep_duration)  # Ensure minimum sleep
            logger.debug(
                f"Token bucket empty ({self.tokens:.2f}). Waiting for token: {sleep_duration:.3f}s"
            )

        # Step 3: Perform the calculated sleep
        if sleep_duration > 0:
            time.sleep(sleep_duration)

        # Step 4: Refill tokens *again* after sleeping to account for the wait time
        self._refill_tokens()

        # Step 5: Consume the token *after* waiting if we had to wait
        # This handles the case where tokens were < 1 initially.
        if (
            requested_at == self.last_refill_time
        ):  # Heuristic: if refill time hasn't changed, we waited
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                logger.debug(
                    f"Consumed token after waiting. Tokens left: {self.tokens:.2f}"
                )
            else:
                # This case should be rare if wait calculation is correct.
                logger.warning(
                    f"Waited for token, but still < 1 ({self.tokens:.2f}) after refill. Consuming fraction."
                )
                self.tokens = 0.0  # Consume whatever fraction was refilled

        # Step 6: Return the actual sleep duration
        return sleep_duration

    # End of wait

    def reset_delay(self):
        """Resets the dynamic component of the delay back to its initial value."""
        # Step 1: Check if delay needs resetting
        if self.current_delay != self.initial_delay:
            # Step 2: Reset delay and log
            self.current_delay = self.initial_delay
            logger.info(
                f"Rate limiter base delay reset to initial: {self.initial_delay:.2f}s"
            )
        # Step 3: Reset throttle flag as well
        self.last_throttled = False

    # End of reset_delay

    def decrease_delay(self):
        """
        Gradually decreases the dynamic component of the delay towards the initial
        value if no throttling has occurred recently.
        """
        # Step 1: Check if decrease is applicable (not throttled and above initial delay)
        if not self.last_throttled and self.current_delay > self.initial_delay:
            previous_delay = self.current_delay
            # Step 2: Apply decrease factor, ensuring it doesn't go below initial delay
            self.current_delay = max(
                self.current_delay * self.decrease_factor, self.initial_delay
            )
            # Step 3: Log the decrease if it was significant
            if (
                abs(previous_delay - self.current_delay) > 0.01
            ):  # Log only noticeable changes
                logger.debug(
                    f"Decreased base delay component to {self.current_delay:.2f}s"
                )
        # Step 4: Reset throttle flag after any successful action (decrease or no change)
        self.last_throttled = False

    # End of decrease_delay

    def increase_delay(self):
        """
        Increases the dynamic component of the delay exponentially upon receiving
        throttling feedback (e.g., a 429 response).
        """
        # Step 1: Store previous delay for logging
        previous_delay = self.current_delay
        # Step 2: Apply backoff factor, ensuring it doesn't exceed max delay
        self.current_delay = min(
            self.current_delay * self.backoff_factor, self.MAX_DELAY
        )
        # Step 3: Log the increase (as INFO level)
        logger.info(
            f"Rate limit feedback received. Increased base delay from {previous_delay:.2f}s to {self.current_delay:.2f}s"
        )
        # Step 4: Set the throttle flag
        self.last_throttled = True

    # End of increase_delay

    def is_throttled(self) -> bool:
        """Returns True if the rate limiter increased delay due to recent throttling feedback."""
        return self.last_throttled

    # End of is_throttled


# End of DynamicRateLimiter class


# ------------------------------
# Session Management
# ------------------------------


class SessionManager:
    """
    Manages the Selenium WebDriver session, database connection pool,
    requests.Session, cloudscraper instance, and related state (e.g., user IDs, CSRF).
    Provides methods for starting, verifying, closing, and interacting with these resources.
    """

    def __init__(self):
        """Initializes the SessionManager with configuration and default session variables."""
        # Step 1: Initialize WebDriver and related state flags
        self.driver: Optional[WebDriver] = None
        self.driver_live: bool = (
            False  # Indicates if driver is initialized and on base URL
        )
        self.session_ready: bool = (
            False  # Indicates if driver is live, user logged in, identifiers fetched
        )

        # Step 2: Load configuration values
        self.db_path: str = str(config_instance.DATABASE_FILE.resolve())
        self.selenium_config = selenium_config
        self.ancestry_username: str = config_instance.ANCESTRY_USERNAME
        self.ancestry_password: str = config_instance.ANCESTRY_PASSWORD
        # (Load other selenium_config values like paths, ports, modes, etc.)
        self.debug_port: int = self.selenium_config.DEBUG_PORT
        self.chrome_user_data_dir: Optional[Path] = (
            self.selenium_config.CHROME_USER_DATA_DIR
        )
        self.profile_dir: str = self.selenium_config.PROFILE_DIR
        self.chrome_driver_path: Optional[Path] = (
            self.selenium_config.CHROME_DRIVER_PATH
        )
        self.chrome_browser_path: Optional[Path] = (
            self.selenium_config.CHROME_BROWSER_PATH
        )
        self.chrome_max_retries: int = self.selenium_config.CHROME_MAX_RETRIES
        self.chrome_retry_delay: int = self.selenium_config.CHROME_RETRY_DELAY
        self.headless_mode: bool = self.selenium_config.HEADLESS_MODE

        # Step 3: Initialize Database engine/session factory attributes
        self.engine = None  # SQLAlchemy engine
        self.Session = None  # SQLAlchemy session factory
        self._db_init_attempted = False  # Flag for lazy initialization

        # Step 4: Initialize session state attributes
        self.cache_dir: Path = config_instance.CACHE_DIR
        self.csrf_token: Optional[str] = None
        self.my_profile_id: Optional[str] = None
        self.my_uuid: Optional[str] = None
        self.my_tree_id: Optional[str] = None
        self.tree_owner_name: Optional[str] = None
        self.session_start_time: Optional[float] = None

        # Step 5: Initialize flags for logging identifiers only once
        self._profile_id_logged = False
        self._uuid_logged = False
        self._tree_id_logged = False
        self._owner_logged = False

        # Step 6: Initialize shared requests.Session with retry adapter
        self._requests_session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],  # Codes for retry
            ),
        )
        self._requests_session.mount("http://", adapter)
        self._requests_session.mount("https://", adapter)
        logger.debug("Initialized shared requests.Session with HTTPAdapter.")

        # Step 7: Initialize shared cloudscraper instance (with retry)
        self.scraper: Optional[cloudscraper.CloudScraper] = (
            None  # Initialize as None first
        )
        try:
            self.scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True},
                delay=10,  # Base delay for scraper
            )
            scraper_retry = Retry(
                total=3,
                backoff_factor=0.8,
                status_forcelist=[403, 429, 500, 502, 503, 504],  # Include 403 for CF
            )
            scraper_adapter = HTTPAdapter(max_retries=scraper_retry)
            self.scraper.mount("http://", scraper_adapter)
            self.scraper.mount("https://", scraper_adapter)
            logger.debug(
                "Initialized shared cloudscraper instance with retry strategy."
            )
        except Exception as scraper_init_e:
            logger.error(
                f"Failed to initialize cloudscraper instance: {scraper_init_e}",
                exc_info=True,
            )
            self.scraper = None  # Ensure it's None on failure

        # Step 8: Initialize Dynamic Rate Limiter
        self.dynamic_rate_limiter: DynamicRateLimiter = DynamicRateLimiter()
        self.last_js_error_check: datetime = datetime.now()

        # Step 9: Log SessionManager creation
        logger.debug(f"SessionManager instance created: ID={id(self)}")

    # End of __init__

    def start_sess(self, action_name: Optional[str] = None) -> bool:
        """
        Phase 1 of session start: Initializes the WebDriver instance and navigates
        to the base URL to stabilize the initial state. Sets `driver_live` flag.
        Does NOT perform login checks or identifier fetching.

        Args:
            action_name: Optional description of the action triggering the start.

        Returns:
            True if the driver was initialized and navigated successfully, False otherwise.
        """
        logger.debug(
            f"--- SessionManager Phase 1: Starting Driver ({action_name or 'Unknown Action'}) ---"
        )
        # Step 1: Reset state flags
        self.driver_live = False
        self.session_ready = False
        self.driver = None  # Ensure driver starts as None for this attempt

        # Step 2: Clean up previous session state attributes (if any)
        self.csrf_token = None
        self.my_profile_id = None
        self.my_uuid = None
        self.my_tree_id = None
        self.tree_owner_name = None
        self._reset_logged_flags()

        # Step 3: Initialize DB Engine/Session factory if not already done
        # This is done early to catch DB connection issues before browser start.
        if not self.engine or not self.Session:
            try:
                self._initialize_db_engine_and_session()
            except Exception as db_init_e:
                logger.critical(
                    f"DB Initialization failed during Phase 1 start: {db_init_e}"
                )
                return False  # Critical failure if DB can't initialize

        # Step 4: Ensure requests.Session exists (fallback)
        if not hasattr(self, "_requests_session") or not isinstance(
            self._requests_session, requests.Session
        ):
            self._requests_session = requests.Session()  # Recreate if necessary
            logger.debug(
                "Shared requests.Session initialized (fallback in start_sess)."
            )

        # Step 5: Initialize WebDriver using the helper function (which handles retries)
        logger.debug("Initializing WebDriver instance (using init_webdvr)...")
        try:
            # init_webdvr handles its own internal retries
            self.driver = init_webdvr()
            if not self.driver:
                logger.error(
                    "WebDriver initialization failed (init_webdvr returned None after retries)."
                )
                return False  # Failed to get driver

            logger.debug("WebDriver initialization successful.")

            # Step 6: Navigate to Base URL to stabilize the browser state
            logger.debug(
                f"Navigating to Base URL ({config_instance.BASE_URL}) to stabilize..."
            )
            base_url_nav_ok = nav_to_page(
                self.driver,
                config_instance.BASE_URL,
                selector="body",  # Wait for body element as simple confirmation
                session_manager=self,
            )
            if not base_url_nav_ok:
                logger.error("Failed to navigate to Base URL after WebDriver init.")
                self.close_sess()  # Clean up the partially started driver
                return False

            logger.debug("Initial navigation to Base URL successful.")

            # Step 7: Set flags and record start time on success
            self.driver_live = True
            self.session_start_time = time.time()
            self.last_js_error_check = datetime.now()
            logger.debug("--- SessionManager Phase 1: Driver Start Successful ---")
            return True

        # Step 8: Handle exceptions during driver start or initial navigation
        except WebDriverException as wd_exc:
            logger.error(
                f"WebDriverException during Phase 1 start/base nav: {wd_exc}",
                exc_info=False,
            )
            self.close_sess()  # Clean up
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error during Phase 1 start/base nav: {e}", exc_info=True
            )
            self.close_sess()  # Clean up
            return False

    # End of start_sess

    def ensure_session_ready(self, action_name: Optional[str] = None) -> bool:
        """
        Phase 2 of session start: Ensures the user is logged in and essential
        identifiers (CSRF token, profile ID, UUID, tree ID) are fetched and stored.
        Must be called AFTER start_sess (Phase 1) is successful. Sets `session_ready` flag.

        Args:
            action_name: Optional description of the action requiring a ready session.

        Returns:
            True if the session is verified as logged in and ready, False otherwise.
        """
        logger.debug(
            f"--- SessionManager Phase 2: Ensuring Session Ready ({action_name or 'Unknown Action'}) ---"
        )

        # Step 1: Check prerequisites (driver must be live from Phase 1)
        if not self.driver_live or not self.driver:
            logger.error(
                "Cannot ensure session readiness: Driver not live (Phase 1 required)."
            )
            return False
        # Step 2: Check if already marked as ready
        if self.session_ready:
            logger.debug("Session already marked as ready. Skipping readiness checks.")
            return True

        # Step 3: Reset state specific to readiness checks
        self.csrf_token = None
        self.my_profile_id = None
        self.my_uuid = None
        self.my_tree_id = None
        self.tree_owner_name = None
        self._reset_logged_flags()

        # Step 4: Perform readiness checks with limited retries for this phase
        max_readiness_attempts = 2  # Allow one retry for the entire readiness sequence
        for attempt in range(1, max_readiness_attempts + 1):
            logger.debug(
                f"Session readiness attempt {attempt}/{max_readiness_attempts}..."
            )
            try:
                # Step 4a: Check Login Status (uses API first, then UI fallback)
                logger.debug("Checking login status...")
                login_stat = login_status(self)
                if login_stat is True:
                    logger.debug("Login status check: User is logged in.")
                elif login_stat is False:
                    logger.info(
                        "Login status check: User not logged in. Attempting login..."
                    )
                    login_result = log_in(self)  # Attempt the login process
                    if login_result != "LOGIN_SUCCEEDED":
                        logger.error(
                            f"Login attempt failed ({login_result}). Readiness check failed."
                        )
                        return False  # Fail readiness if login fails
                    logger.info("Login successful.")
                    # Re-verify status after login attempt
                    if not login_status(self):
                        logger.error(
                            "Login status verification failed after successful login report."
                        )
                        return False
                    logger.debug("Login status re-verified successfully after login.")
                else:  # login_stat is None (critical error during check)
                    logger.error(
                        "Login status check failed critically. Readiness check failed."
                    )
                    return False
                logger.debug("Login status confirmed.")

                # Step 4b: Check Current URL (navigate to safe page if needed)
                logger.debug("Checking current URL validity...")
                if not self._check_and_handle_url():
                    logger.error("URL check/handling failed during readiness check.")
                    return False
                logger.debug("URL check/handling completed.")

                # Step 4c: Verify Essential Cookies
                logger.debug("Verifying essential cookies (ANCSESSIONID, SecureATT)...")
                essential_cookies = ["ANCSESSIONID", "SecureATT"]
                if not self.get_cookies(
                    essential_cookies, timeout=15
                ):  # Wait up to 15s
                    logger.error(
                        f"Essential cookies {essential_cookies} not found. Readiness check failed."
                    )
                    return False
                logger.debug("Essential cookies verified.")

                # Step 4d: Synchronize Cookies from WebDriver to requests.Session
                logger.debug("Syncing cookies from WebDriver to requests session...")
                self._sync_cookies()
                logger.debug("Cookies synced to requests session.")

                # Step 4e: Retrieve and Store CSRF Token
                logger.debug("Retrieving CSRF token...")
                self.csrf_token = self.get_csrf()  # Fetches fresh token
                if not self.csrf_token:
                    logger.error(
                        "Failed to retrieve CSRF token. Readiness check failed."
                    )
                    return False
                logger.debug(f"CSRF token retrieved: {self.csrf_token[:10]}...")

                # Step 4f: Retrieve User Identifiers (Profile ID, UUID, Tree ID)
                logger.debug("Retrieving user identifiers...")
                if not self._retrieve_identifiers():
                    # _retrieve_identifiers logs specific errors internally
                    logger.error(
                        "Failed to retrieve one or more essential user identifiers."
                    )
                    return False
                logger.debug("Finished retrieving user identifiers.")

                # Step 4g: Retrieve Tree Owner Name (if tree ID exists)
                logger.debug("Retrieving tree owner name (if applicable)...")
                self._retrieve_tree_owner()  # Logs internally
                logger.debug("Finished retrieving tree owner name.")

                # Step 4h: Mark Session as Ready
                self.session_ready = True
                logger.debug("--- SessionManager Phase 2: Session Ready Successful ---")
                return True  # Success!

            # --- Handle Exceptions during the attempt ---
            except WebDriverException as wd_exc:
                logger.error(
                    f"WebDriverException during readiness check attempt {attempt}: {wd_exc}",
                    exc_info=False,
                )
                # Check if session died, critical if so
                if not self.is_sess_valid():
                    logger.error(
                        "Session became invalid during readiness check. Aborting."
                    )
                    self.driver_live = False  # Mark driver as dead
                    self.session_ready = False
                    self.close_sess()  # Clean up
                    return False
                # If session seems valid, maybe transient error, wait and retry if attempts remain
                if attempt < max_readiness_attempts:
                    logger.info(
                        f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next readiness attempt..."
                    )
                    time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                else:
                    logger.error(
                        "Readiness check failed after final attempt due to WebDriverException."
                    )
                    return False  # Failed after retries

            except Exception as e:
                logger.error(
                    f"Unexpected error during readiness check attempt {attempt}: {e}",
                    exc_info=True,
                )
                if attempt < max_readiness_attempts:
                    logger.info(
                        f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next readiness attempt..."
                    )
                    time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                else:
                    logger.error(
                        "Readiness check failed after final attempt due to unexpected error."
                    )
                    return False

        # --- End of Retry Loop ---
        logger.error(
            f"Session readiness check FAILED after {max_readiness_attempts} attempts."
        )
        return False  # Failed after all retries

    # End of ensure_session_ready

    def _reset_logged_flags(self):
        """Helper method to reset flags that track if identifiers have been logged."""
        self._profile_id_logged = False
        self._uuid_logged = False
        self._tree_id_logged = False
        self._owner_logged = False

    # End of _reset_logged_flags

    def _initialize_db_engine_and_session(self):
        """
        Initializes the SQLAlchemy engine and session factory with pooling.
        Handles re-initialization if called again. Uses config for pool size.
        Applies PRAGMA settings via event listener. Creates tables if needed.
        """
        # Step 1: Check if already initialized for this SessionManager instance
        if self.engine and self.Session:
            logger.debug(
                f"DB Engine/Session already initialized for SM ID={id(self)}. Skipping."
            )
            return

        logger.debug(
            f"SessionManager ID={id(self)} initializing SQLAlchemy Engine/Session..."
        )
        self._db_init_attempted = True  # Mark that we tried

        # Step 2: Dispose existing engine if present (e.g., during restart)
        if self.engine:
            logger.debug(
                f"Disposing existing engine before re-initializing (SM ID={id(self)})."
            )
            try:
                self.engine.dispose()
            except Exception as dispose_e:
                logger.error(f"Error disposing existing engine: {dispose_e}")
            self.engine = None
            self.Session = None

        # Step 3: Create new engine and session factory
        try:
            logger.debug(f"DB Path: {self.db_path}")

            # Step 3a: Determine Pool Size (using env var or config)
            # (Pool size logic remains the same as provided previously)
            pool_size_env_str = os.getenv("DB_POOL_SIZE")
            pool_size_config = getattr(config_instance, "DB_POOL_SIZE", None)
            pool_size_str_to_parse = None
            if pool_size_env_str is not None:
                pool_size_str_to_parse = pool_size_env_str
            elif pool_size_config is not None:
                try:
                    pool_size_str_to_parse = str(int(pool_size_config))
                except (ValueError, TypeError):
                    pool_size_str_to_parse = "50"  # Fallback if config invalid
            else:
                pool_size_str_to_parse = "50"  # Fallback if neither set
            pool_size = 20  # Default if parsing fails
            try:
                parsed_val = int(pool_size_str_to_parse)
                if parsed_val <= 0:
                    logger.warning(
                        f"DB_POOL_SIZE value '{parsed_val}' invalid (<=0). Using default {pool_size}."
                    )
                elif parsed_val == 1:
                    pool_size = 1
                    logger.warning(
                        "DB_POOL_SIZE value '1' detected. Using minimal pool size."
                    )
                else:
                    pool_size = min(parsed_val, 100)  # Cap at 100
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not parse DB_POOL_SIZE '{pool_size_str_to_parse}'. Using default {pool_size}."
                )
            max_overflow = max(5, int(pool_size * 0.2))
            pool_timeout = 30
            pool_class = sqlalchemy_pool.QueuePool
            logger.debug(
                f"DB Pool Config: Size={pool_size}, MaxOverflow={max_overflow}, Timeout={pool_timeout}"
            )

            # Step 3b: Create the Engine
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=False,  # Set to True for verbose SQL logging
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                poolclass=pool_class,
                connect_args={
                    "check_same_thread": False
                },  # Required for SQLite multithreading
            )
            logger.debug(
                f"Created NEW SQLAlchemy engine: ID={id(self.engine)} for SM ID={id(self)}"
            )

            # Step 3c: Set up PRAGMA listener
            @event.listens_for(self.engine, "connect")
            def enable_sqlite_settings(dbapi_connection, connection_record):
                """Listener to set PRAGMA settings upon connection."""
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute(
                        "PRAGMA journal_mode=WAL;"
                    )  # Use Write-Ahead Logging
                    cursor.execute("PRAGMA foreign_keys=ON;")  # Enforce foreign keys
                    # Optional: cursor.execute("PRAGMA synchronous=NORMAL;") # Less strict sync
                    logger.debug("SQLite PRAGMA settings applied (WAL, Foreign Keys).")
                except Exception as pragma_e:
                    logger.error(f"Failed setting PRAGMA: {pragma_e}")
                finally:
                    cursor.close()

            # End of enable_sqlite_settings listener

            # Step 3d: Create the Session Factory
            self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
            logger.debug(f"Created Session factory for Engine ID={id(self.engine)}")

            # Step 3e: Create Tables if they don't exist
            try:
                Base.metadata.create_all(self.engine)
                logger.debug("DB tables checked/created successfully.")
            except Exception as table_create_e:
                logger.error(
                    f"Error creating DB tables: {table_create_e}", exc_info=True
                )
                raise table_create_e  # Re-raise to indicate critical failure

        # Step 4: Handle initialization errors
        except Exception as e:
            logger.critical(f"FAILED to initialize SQLAlchemy: {e}", exc_info=True)
            # Clean up partially created resources
            if self.engine:
                try:
                    self.engine.dispose()
                except Exception:
                    pass
            self.engine = None
            self.Session = None
            raise e  # Re-raise the critical error

    # End of _initialize_db_engine_and_session

    def _check_and_handle_url(self) -> bool:
        """
        Checks if the current browser URL is suitable for general interaction.
        Navigates to the base URL if on a sign-in, logout, MFA, or API path.

        Returns:
            True if the URL is suitable or navigation succeeded, False otherwise.
        """
        # Step 1: Get current URL
        try:
            if self.driver is None:
                logger.error("Driver not initialized. Cannot check URL.")
                return False
            current_url = self.driver.current_url
            logger.debug(f"Current URL for check: {current_url}")
        except WebDriverException as e:
            logger.error(f"Error getting current URL: {e}. Session might be dead.")
            if not self.is_sess_valid():
                logger.warning("Session seems invalid during URL check.")
            return False

        # Step 2: Define unsuitable URL patterns
        base_url_norm = config_instance.BASE_URL.rstrip("/") + "/"
        signin_url_base = urljoin(base_url_norm, "account/signin")
        logout_url_base = urljoin(base_url_norm, "c/logout")
        mfa_url_base = urljoin(base_url_norm, "account/signin/mfa/")
        disallowed_starts = (signin_url_base, logout_url_base, mfa_url_base)
        is_api_path = "/api/" in current_url

        # Step 3: Determine if navigation is needed
        needs_navigation = False
        reason = ""
        if not current_url.startswith(config_instance.BASE_URL.rstrip("/")):
            needs_navigation = True
            reason = "URL does not start with base URL."
        elif any(current_url.startswith(path) for path in disallowed_starts):
            needs_navigation = True
            reason = f"URL starts with disallowed path ({current_url})."
        elif is_api_path:
            needs_navigation = True
            reason = "URL contains '/api/'."

        # Step 4: Perform navigation if needed
        if needs_navigation:
            logger.info(
                f"Current URL unsuitable ({reason}). Navigating to base URL: {base_url_norm}"
            )
            if not nav_to_page(
                self.driver, base_url_norm, selector="body", session_manager=self
            ):
                logger.error("Failed to navigate to base URL during check.")
                return False
            logger.debug("Navigation to base URL successful.")
        else:
            logger.debug("Current URL is suitable, no extra navigation needed.")

        # Step 5: Return success
        return True

    # End of _check_and_handle_url

    def _retrieve_identifiers(self) -> bool:
        """
        Helper method to retrieve and store essential user identifiers (profile ID,
        UUID, Tree ID). Logs identifiers only the first time they are retrieved.

        Returns:
            True if all required identifiers were successfully retrieved, False otherwise.
        """
        all_ok = True  # Assume success initially

        # --- Step 1: Get Profile ID (ucdmid) ---
        if not self.my_profile_id:  # Only fetch if not already set
            logger.debug("Retrieving profile ID (ucdmid)...")
            self.my_profile_id = self.get_my_profileId()  # Calls API helper
            if not self.my_profile_id:
                logger.error("Failed to retrieve profile ID (ucdmid).")
                all_ok = False
            elif not self._profile_id_logged:  # Log only on first successful fetch
                logger.info(f"My profile id: {self.my_profile_id}")
                self._profile_id_logged = True
        elif not self._profile_id_logged:  # Log if already set but not yet logged
            logger.info(f"My profile id: {self.my_profile_id}")
            self._profile_id_logged = True

        # --- Step 2: Get UUID (testId) ---
        if not self.my_uuid:  # Only fetch if not already set
            logger.debug("Retrieving UUID (testId)...")
            self.my_uuid = self.get_my_uuid()  # Calls API helper
            if not self.my_uuid:
                logger.error("Failed to retrieve UUID (testId).")
                all_ok = False
            elif not self._uuid_logged:  # Log only on first successful fetch
                logger.info(f"My uuid: {self.my_uuid}")
                self._uuid_logged = True
        elif not self._uuid_logged:  # Log if already set but not yet logged
            logger.info(f"My uuid: {self.my_uuid}")
            self._uuid_logged = True

        # --- Step 3: Get Tree ID (if TREE_NAME configured) ---
        if (
            config_instance.TREE_NAME and not self.my_tree_id
        ):  # Only fetch if needed and not set
            logger.debug(
                f"Retrieving tree ID for tree name: '{config_instance.TREE_NAME}'..."
            )
            self.my_tree_id = self.get_my_tree_id()  # Calls API helper
            if not self.my_tree_id:
                # Treat as error only if TREE_NAME was specifically configured
                logger.error(
                    f"TREE_NAME '{config_instance.TREE_NAME}' configured, but failed to get corresponding tree ID."
                )
                all_ok = False
            elif not self._tree_id_logged:  # Log only on first successful fetch
                logger.info(f"My tree id: {self.my_tree_id}")
                self._tree_id_logged = True
        elif (
            self.my_tree_id and not self._tree_id_logged
        ):  # Log if already set but not logged
            logger.info(f"My tree id: {self.my_tree_id}")
            self._tree_id_logged = True
        elif not config_instance.TREE_NAME:
            logger.debug("No TREE_NAME configured, skipping tree ID retrieval.")

        # --- Step 4: Return overall success status ---
        return all_ok

    # End of _retrieve_identifiers

    def _retrieve_tree_owner(self):
        """
        Helper method to retrieve the tree owner's display name if a tree ID is known.
        Logs the owner name only the first time it's retrieved.
        """
        # --- Step 1: Check prerequisites (tree ID known, owner name not yet known) ---
        if self.my_tree_id and not self.tree_owner_name:
            logger.debug(
                f"Retrieving tree owner name for tree ID: {self.my_tree_id}..."
            )
            # --- Step 2: Call API helper ---
            self.tree_owner_name = self.get_tree_owner(self.my_tree_id)
            # --- Step 3: Log result (only once) ---
            if self.tree_owner_name and not self._owner_logged:
                logger.info(
                    f"Tree Owner Name: {self.tree_owner_name}.\n"
                )  
                self._owner_logged = True
            elif not self.tree_owner_name:
                logger.warning("Failed to retrieve tree owner name.\n") 
        # --- Step 4: Log if already known but not yet logged ---
        elif self.tree_owner_name and not self._owner_logged:
            logger.info(f"Tree Owner Name: {self.tree_owner_name}\n")  
            self._owner_logged = True
        # --- Step 5: Log if skipping due to no tree ID ---
        elif not self.my_tree_id:
            logger.debug(
                "Skipping tree owner retrieval (no tree ID).\n"
            )  

    # End of _retrieve_tree_owner

    @retry_api()  # Add retry decorator for resilience
    def get_csrf(self) -> Optional[str]:
        """
        Fetches a fresh CSRF token from the Ancestry API using _api_req.

        Note: This should be called when a CSRF token is explicitly needed, typically
              during session readiness checks or before CSRF-protected API calls if
              the cached token might be stale. The SessionManager stores the latest
              fetched token in `self.csrf_token`.

        Returns:
            The CSRF token string if successful, None otherwise.
        """
        # Step 1: Define API endpoint
        csrf_token_url = urljoin(
            config_instance.BASE_URL, "discoveryui-matches/parents/api/csrfToken"
        )
        logger.debug(f"Attempting to fetch fresh CSRF token from: {csrf_token_url}")

        # Step 2: Ensure essential cookies are likely present before API call
        # Although _api_req handles cookies, checking here provides early feedback.
        essential_cookies = ["ANCSESSIONID", "SecureATT"]
        if not self.get_cookies(essential_cookies, timeout=10):
            logger.warning(
                f"Essential cookies {essential_cookies} NOT found before CSRF token API call. API might fail."
            )
            # Continue attempt, _api_req might still succeed if cookies recently added

        # Step 3: Call the API using _api_req helper
        response_data = _api_req(
            url=csrf_token_url,
            driver=self.driver,  # Pass driver for potential UBE header generation
            session_manager=self,
            method="GET",
            use_csrf_token=False,  # Don't send a CSRF token to get one
            api_description="CSRF Token API",
            force_text_response=True,  # Expect plain text response
        )

        # Step 4: Process the response
        if response_data and isinstance(response_data, str):
            # Success: Return the non-empty token string
            csrf_token_val = response_data.strip()
            if csrf_token_val:
                logger.debug(
                    f"CSRF token successfully retrieved (Length: {len(csrf_token_val)})."
                )
                return csrf_token_val
            else:
                logger.error("CSRF token API returned empty string.")
                return None
        elif response_data is None:
            # Failure within _api_req (e.g., retries exhausted)
            logger.warning(
                "Failed to get CSRF token response via _api_req (returned None)."
            )
            return None
        else:
            # Unexpected response format
            logger.error(
                f"Unexpected response type for CSRF token API: {type(response_data)}"
            )
            logger.debug(f"Response data received: {response_data}")
            return None

    # End of get_csrf

    def get_cookies(self, cookie_names: List[str], timeout: int = 30) -> bool:
        """
        Waits until specified cookies are present in the browser session.

        Args:
            cookie_names: A list of cookie names to wait for.
            timeout: Maximum time in seconds to wait.

        Returns:
            True if all specified cookies are found within the timeout, False otherwise.
        """
        # Step 1: Initialization
        start_time = time.time()
        logger.debug(f"Waiting up to {timeout}s for cookies: {cookie_names}...")
        required_lower = {
            name.lower() for name in cookie_names
        }  # Case-insensitive check
        interval = 0.5  # Polling interval
        last_missing_str = ""  # To avoid repetitive logging

        # Step 2: Polling loop
        while time.time() - start_time < timeout:
            try:
                # Step 2a: Check session validity first
                if not self.is_sess_valid():
                    logger.warning("Session became invalid while waiting for cookies.")
                    return False
                if self.driver is None:
                    logger.error("Driver is None. Cannot retrieve cookies.")
                    return False  # Should not happen if is_sess_valid passed

                # Step 2b: Get current cookies
                cookies = self.driver.get_cookies()
                current_cookies_lower = {
                    c["name"].lower() for c in cookies if "name" in c
                }  # Handle potential malformed cookies

                # Step 2c: Check if all required cookies are present
                missing_lower = required_lower - current_cookies_lower
                if not missing_lower:
                    logger.debug(f"All required cookies found: {cookie_names}.")
                    return True  # Success

                # Step 2d: Log missing cookies periodically or on change
                missing_str = ", ".join(sorted(missing_lower))
                if missing_str != last_missing_str:
                    logger.debug(f"Still missing cookies: {missing_str}")
                    last_missing_str = missing_str

                # Step 2e: Wait before next poll
                time.sleep(interval)

            # Step 3: Handle exceptions during polling
            except WebDriverException as e:
                logger.error(f"WebDriverException while retrieving cookies: {e}")
                # Check if session died after the exception
                if not self.is_sess_valid():
                    logger.error(
                        "Session invalid after WebDriverException during cookie retrieval."
                    )
                    return False
                # Otherwise, wait a bit longer and retry polling
                time.sleep(interval * 2)
            except Exception as e:
                logger.error(f"Unexpected error retrieving cookies: {e}", exc_info=True)
                # Decide policy: maybe fail fast or wait and retry
                time.sleep(interval * 2)

        # Step 4: Handle timeout (loop finished without finding all cookies)
        # Recalculate missing cookies one last time for accurate logging
        missing_final = []
        try:
            if self.driver:
                cookies_final = self.driver.get_cookies()
                current_cookies_final_lower = {
                    c["name"].lower() for c in cookies_final if "name" in c
                }
                missing_final = [
                    name
                    for name in cookie_names
                    if name.lower() in (required_lower - current_cookies_final_lower)
                ]
            else:
                missing_final = cookie_names  # Assume all missing if driver gone
        except Exception:  # Catch errors during final check
            missing_final = cookie_names  # Assume all missing on error

        logger.warning(f"Timeout waiting for cookies. Missing: {missing_final}.")
        return False

    # End of get_cookies

    def _sync_cookies(self):
        """
        Synchronizes cookies from the Selenium WebDriver session to the shared
        `requests.Session` instance (`self._requests_session`).
        Handles domain adjustments if necessary (e.g., for SecureATT cookie).
        """
        # Step 1: Check session validity
        if not self.is_sess_valid():
            logger.warning("Cannot sync cookies: WebDriver session invalid.")
            return
        if self.driver is None:
            logger.error("Driver is None. Cannot retrieve cookies for sync.")
            return

        # Step 2: Get cookies from WebDriver
        try:
            driver_cookies = self.driver.get_cookies()
            logger.debug(
                f"Retrieved {len(driver_cookies)} cookies from WebDriver for sync."
            )
        except WebDriverException as e:
            logger.error(f"WebDriverException getting cookies for sync: {e}")
            if not self.is_sess_valid():
                logger.error(
                    "Session invalid after WebDriverException during cookie sync."
                )
            return
        except Exception as e:
            logger.error(
                f"Unexpected error getting cookies for sync: {e}", exc_info=True
            )
            return

        # Step 3: Prepare requests.Session cookie jar
        requests_cookie_jar = self._requests_session.cookies
        requests_cookie_jar.clear()  # Clear existing cookies before syncing

        # Step 4: Iterate and add cookies to requests session
        synced_count = 0
        skipped_count = 0
        for cookie in driver_cookies:
            # Step 4a: Basic validation of cookie structure
            if (
                not isinstance(cookie, dict)
                or "name" not in cookie
                or "value" not in cookie
                or "domain" not in cookie
            ):
                logger.warning(f"Skipping invalid cookie format during sync: {cookie}")
                skipped_count += 1
                continue

            # Step 4b: Prepare cookie attributes for requests library
            try:
                domain_to_set = cookie["domain"]
                # Special handling for SecureATT domain if needed (adjust as necessary)
                # Example: if cookie["name"] == "SecureATT" and domain_to_set == "www.ancestry.co.uk":
                #     domain_to_set = ".ancestry.co.uk" # Use leading dot for subdomain matching

                cookie_attrs = {
                    "name": cookie["name"],
                    "value": cookie["value"],
                    "domain": domain_to_set,
                    "path": cookie.get("path", "/"),  # Default path to '/'
                    "secure": cookie.get("secure", False),
                    "rest": {
                        "httpOnly": cookie.get("httpOnly", False)
                    },  # Pass httpOnly via rest dict
                }
                # Add expiry if present and valid (requests uses 'expires')
                if "expiry" in cookie and cookie["expiry"] is not None:
                    if isinstance(cookie["expiry"], (int, float)):
                        cookie_attrs["expires"] = int(
                            cookie["expiry"]
                        )  # Use integer timestamp
                    else:
                        logger.warning(
                            f"Unexpected expiry format for cookie {cookie['name']}: {cookie['expiry']}"
                        )

                # Step 4c: Set the cookie in the requests session jar
                requests_cookie_jar.set(**cookie_attrs)
                synced_count += 1

            except Exception as set_err:
                logger.warning(
                    f"Failed to set cookie '{cookie.get('name', '??')}' in requests session: {set_err}"
                )
                skipped_count += 1

        # Step 5: Log summary
        if skipped_count > 0:
            logger.warning(
                f"Skipped {skipped_count} cookies during sync due to format/errors."
            )
        logger.debug(f"Successfully synced {synced_count} cookies to requests session.")

    # End of _sync_cookies

    def return_session(self, session: Session):
        """
        Closes the given SQLAlchemy session, returning its underlying
        connection to the engine's connection pool.

        Args:
            session: The SQLAlchemy Session object to close.
        """
        # Step 1: Check if session object is valid
        if session:
            session_id = id(session)  # For logging
            # Step 2: Attempt to close the session
            try:
                # Optional: Log before closing
                # logger.debug(f"Closing DB session {session_id} (returns connection to pool).")
                session.close()
                # Optional: Log after successful close
                # logger.debug(f"DB session {session_id} closed successfully.")
            # Step 3: Handle errors during close
            except Exception as e:
                logger.error(
                    f"Error closing DB session {session_id}: {e}", exc_info=True
                )
        else:
            logger.warning("Attempted to return a None DB session.")

    # End of return_session

    def get_db_conn(self) -> Optional[Session]:
        """
        Gets a new SQLAlchemy session from the session factory.
        Initializes the engine and session factory if they haven't been already.

        Returns:
            A new SQLAlchemy Session instance, or None if initialization or
            session creation fails.
        """
        # Step 1: Log the request for a connection
        engine_id_str = id(self.engine) if self.engine else "None"
        logger.debug(
            f"SessionManager ID={id(self)} get_db_conn called. Current Engine ID: {engine_id_str}"
        )

        # Step 2: Check if DB engine/session factory needs initialization
        # Initialize if never attempted OR if engine/Session became None after a previous attempt
        if not self._db_init_attempted or not self.engine or not self.Session:
            logger.debug(
                f"SessionManager ID={id(self)}: Engine/Session factory not ready. Triggering initialization..."
            )
            try:
                # Step 2a: Attempt initialization
                self._initialize_db_engine_and_session()
                # Step 2b: Check again after initialization attempt
                if not self.Session:
                    # If still no Session factory after init attempt, log error and return None
                    logger.error(
                        f"SessionManager ID={id(self)}: Initialization failed, cannot get DB connection."
                    )
                    return None
            except Exception as init_e:
                # Handle exceptions during the initialization process
                logger.error(
                    f"SessionManager ID={id(self)}: Exception during lazy initialization in get_db_conn: {init_e}"
                )
                return None

        # Step 3: Attempt to get a session from the factory
        try:
            # self.Session should now be a valid sessionmaker factory
            new_session = self.Session()
            logger.debug(
                f"SessionManager ID={id(self)} obtained DB session {id(new_session)} from Engine ID={id(self.engine)}"
            )
            # Step 4: Return the new session
            return new_session
        except Exception as e:
            # Step 5: Handle errors during session creation from the factory
            logger.error(
                f"SessionManager ID={id(self)} Error getting DB session from factory: {e}",
                exc_info=True,
            )
            # If getting a session fails, the engine/pool might be dead.
            # Dispose the engine and reset flags to force re-initialization next time.
            if self.engine:
                try:
                    self.engine.dispose()
                except Exception:
                    pass  # Ignore errors during disposal here
            self.engine = None
            self.Session = None
            self._db_init_attempted = False  # Allow re-init attempt
            return None

    # End of get_db_conn

    @contextlib.contextmanager
    def get_db_conn_context(self) -> Generator[Optional[Session], None, None]:
        """
        Provides a SQLAlchemy session within a context manager.
        Handles session acquisition, commit/rollback, and closing (returning
        the connection to the pool).

        Yields:
            The acquired SQLAlchemy Session, or None if acquisition failed.
        """
        # Step 1: Acquire session using the standard method
        session: Optional[Session] = None
        session_id_for_log = "N/A"
        try:
            session = self.get_db_conn()
            if session:
                session_id_for_log = id(session)
                logger.debug(
                    f"DB Context Manager: Acquired session {session_id_for_log}."
                )
                # Step 2: Yield the session to the 'with' block
                yield session
                # Step 3: Commit if the 'with' block exited without exception
                # Check session is still active before committing
                if session.is_active:
                    logger.debug(
                        f"DB Context Manager: Committing session {session_id_for_log}."
                    )
                    session.commit()
                    logger.debug(
                        f"DB Context Manager: Commit successful for session {session_id_for_log}."
                    )
                else:
                    logger.warning(
                        f"DB Context Manager: Session {session_id_for_log} inactive after yield, skipping commit."
                    )
            else:
                # Step 2b: Yield None if session acquisition failed
                logger.error("DB Context Manager: Failed to obtain DB session.")
                yield None  # Allow 'with' block to proceed but receive None

        # Step 4: Handle exceptions raised within the 'with' block or during commit
        except Exception as e:
            logger.error(
                f"DB Context Manager: Exception occurred ({type(e).__name__}). Rolling back session {session_id_for_log}.",
                exc_info=True,
            )
            if session and session.is_active:
                try:
                    session.rollback()
                    logger.warning(
                        f"DB Context Manager: Rollback successful for session {session_id_for_log}."
                    )
                except Exception as rb_err:
                    logger.error(
                        f"DB Context Manager: Error during rollback for session {session_id_for_log}: {rb_err}"
                    )
            raise e  # Re-raise the original exception after attempting rollback

        # Step 5: Ensure session is returned to pool regardless of success/failure
        finally:
            if session:
                self.return_session(session)
                logger.debug(
                    f"DB Context Manager: Returned session {session_id_for_log} to pool."
                )
            else:
                logger.debug("DB Context Manager: No valid session to return.")

    # End of get_db_conn_context

    def cls_db_conn(self, keep_db: bool = True):
        """
        Disposes the SQLAlchemy engine, effectively closing all pooled connections.
        Does nothing if keep_db is True.

        Args:
            keep_db: If True, the engine is not disposed. If False, it is disposed.
        """
        # Step 1: Check if disposal is requested and if engine exists
        if keep_db:
            engine_id_str = id(self.engine) if self.engine else "None"
            logger.debug(
                f"cls_db_conn called (keep_db=True). Skipping engine disposal for Engine ID: {engine_id_str}"
            )
            return

        if not self.engine:
            logger.debug(
                f"SessionManager ID={id(self)}: No active SQLAlchemy engine to dispose."
            )
            # Ensure related attributes are also None for consistency
            self.Session = None
            self._db_init_attempted = False
            return

        # Step 2: Proceed with disposal
        engine_id = id(self.engine)
        logger.debug(
            f"SessionManager ID={id(self)} cls_db_conn called (keep_db=False). Disposing Engine ID: {engine_id}"
        )
        try:
            # Step 3: Dispose the engine
            self.engine.dispose()
            logger.debug(f"Engine ID={engine_id} disposed successfully.")
        except Exception as e:
            # Step 4: Log errors during disposal
            logger.error(
                f"Error disposing SQLAlchemy engine ID={engine_id}: {e}", exc_info=True
            )
        finally:
            # Step 5: Reset engine/session attributes regardless of disposal success
            self.engine = None
            self.Session = None
            self._db_init_attempted = False  # Reset flag to allow re-initialization

    # End of cls_db_conn

    @retry_api()  # Add retry decorator
    def get_my_profileId(self) -> Optional[str]:
        """
        Retrieves the logged-in user's profile ID (ucdmid) from the Ancestry API.

        Returns:
            The user's profile ID (uppercase string), or None if retrieval fails.
        """
        # Step 1: Define API URL
        url = urljoin(
            config_instance.BASE_URL,
            "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid",
        )
        logger.debug("Attempting to fetch own profile ID (ucdmid)...")

        # Step 2: Call API using helper
        try:
            response_data = _api_req(
                url=url,
                driver=self.driver,  # Pass driver for context
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="Get my profile_id",
            )

            # Step 3: Process response
            if not response_data:
                logger.warning(
                    "Failed to get profile_id response via _api_req (returned None)."
                )
                return None

            if (
                isinstance(response_data, dict)
                and "data" in response_data
                and "ucdmid" in response_data["data"]
            ):
                # Step 3a: Extract, convert to string, and uppercase the ID
                my_profile_id_val = str(response_data["data"]["ucdmid"]).upper()
                logger.debug(f"Successfully retrieved profile_id: {my_profile_id_val}")
                return my_profile_id_val
            else:
                # Step 3b: Log error if expected structure is missing
                logger.error("Could not find 'data.ucdmid' in profile_id API response.")
                logger.debug(f"Full profile_id response data: {response_data}")
                return None

        # Step 4: Handle unexpected errors
        except Exception as e:
            logger.error(f"Unexpected error in get_my_profileId: {e}", exc_info=True)
            return None

    # End of get_my_profileId

    @retry_api()  # Add retry decorator
    def get_my_uuid(self) -> Optional[str]:
        """
        Retrieves the logged-in user's DNA test UUID (testId/sampleId) from the
        header/dna API endpoint.

        Returns:
            The user's UUID (uppercase string), or None if retrieval fails.
        """
        # Step 1: Check session validity
        if not self.is_sess_valid():
            logger.error("get_my_uuid: Session invalid.")
            return None

        # Step 2: Define API URL
        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        logger.debug("Attempting to fetch own UUID (testId) from header/dna API...")

        # Step 3: Call API using helper
        response_data = _api_req(
            url=url,
            driver=self.driver,
            session_manager=self,
            method="GET",
            use_csrf_token=False,
            api_description="Get UUID API",
        )

        # Step 4: Process response
        if response_data:
            if isinstance(response_data, dict) and "testId" in response_data:
                # Step 4a: Extract, convert to string, and uppercase the ID
                my_uuid_val = str(response_data["testId"]).upper()
                logger.debug(f"Successfully retrieved UUID: {my_uuid_val}")
                return my_uuid_val
            else:
                # Step 4b: Log error if 'testId' key is missing
                logger.error("Could not retrieve UUID ('testId' missing in response).")
                logger.debug(f"Full get_my_uuid response data: {response_data}")
                return None
        else:
            # Step 4c: Log error if API call failed
            logger.error(
                "Failed to get header/dna data via _api_req (returned None or error response)."
            )
            return None

    # End of get_my_uuid

    @retry_api()  # Add retry decorator
    def get_my_tree_id(self) -> Optional[str]:
        """
        Retrieves the tree ID corresponding to the TREE_NAME specified in the
        configuration, using the header/trees API endpoint.

        Returns:
            The tree ID string if found, None otherwise or if TREE_NAME is not set.
        """
        # Step 1: Check if TREE_NAME is configured
        tree_name_config = config_instance.TREE_NAME
        if not tree_name_config:
            logger.debug("TREE_NAME not configured, skipping tree ID retrieval.")
            return None

        # Step 2: Check session validity
        if not self.is_sess_valid():
            logger.error("get_my_tree_id: Session invalid.")
            return None

        # Step 3: Define API URL
        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/trees")
        logger.debug(
            f"Attempting to fetch tree ID for TREE_NAME='{tree_name_config}'..."
        )

        # Step 4: Call API using helper
        try:
            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="Header Trees API",
            )

            # Step 5: Process the response
            if (
                response_data
                and isinstance(response_data, dict)
                and "menuitems" in response_data
            ):
                # Step 5a: Iterate through menu items to find matching tree name
                for item in response_data["menuitems"]:
                    if isinstance(item, dict) and item.get("text") == tree_name_config:
                        tree_url = item.get("url")
                        if tree_url and isinstance(tree_url, str):
                            # Step 5b: Extract tree ID from the URL using regex
                            match = re.search(r"/tree/(\d+)", tree_url)
                            if match:
                                my_tree_id_val = match.group(1)
                                logger.debug(
                                    f"Found tree ID '{my_tree_id_val}' for tree '{tree_name_config}'."
                                )
                                return my_tree_id_val
                            else:
                                logger.warning(
                                    f"Found tree '{tree_name_config}', but URL format unexpected: {tree_url}"
                                )
                        else:
                            logger.warning(
                                f"Found tree '{tree_name_config}', but 'url' key missing or invalid."
                            )
                        break  # Stop searching once the named tree is found (even if URL was bad)

                # Step 5c: Log if tree name not found in response
                logger.warning(
                    f"Could not find TREE_NAME '{tree_name_config}' in Header Trees API response."
                )
                return None
            else:
                # Step 5d: Log if response format is unexpected
                logger.warning(
                    "Unexpected response format from Header Trees API (missing 'menuitems'?)."
                )
                logger.debug(f"Full Header Trees response data: {response_data}")
                return None
        # Step 6: Handle exceptions during API call or processing
        except Exception as e:
            logger.error(f"Error fetching/parsing Header Trees API: {e}", exc_info=True)
            return None

    # End of get_my_tree_id

    @retry_api()  # Add retry decorator
    def get_tree_owner(self, tree_id: str) -> Optional[str]:
        """
        Retrieves the display name of the owner of a specified tree ID.

        Args:
            tree_id: The ID of the tree whose owner is to be retrieved.

        Returns:
            The owner's display name string, or None if retrieval fails.
        """
        # Step 1: Validate input tree_id
        if not tree_id:
            logger.warning("Cannot get tree owner: tree_id is missing.")
            return None

        # Step 2: Check session validity
        if not self.is_sess_valid():
            logger.error("get_tree_owner: Session invalid.")
            return None

        # Step 3: Define API URL
        url = urljoin(
            config_instance.BASE_URL,
            f"api/uhome/secure/rest/user/tree-info?tree_id={tree_id}",
        )
        logger.debug(f"Attempting to fetch tree owner name for tree ID: {tree_id}...")

        # Step 4: Call API using helper
        try:
            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="Tree Owner Name API",
            )

            # Step 5: Process the response
            if response_data and isinstance(response_data, dict):
                owner_data = response_data.get("owner")
                if owner_data and isinstance(owner_data, dict):
                    display_name = owner_data.get("displayName")
                    if display_name and isinstance(display_name, str):
                        # Step 5a: Return the display name if found
                        logger.debug(
                            f"Found tree owner '{display_name}' for tree ID {tree_id}."
                        )
                        return display_name
                    else:
                        logger.warning(
                            f"Could not find 'displayName' in owner data for tree {tree_id}."
                        )
                else:
                    logger.warning(
                        f"Could not find 'owner' data in Tree Owner API response for tree {tree_id}."
                    )
                # Log full response if owner/displayName missing
                logger.debug(f"Full Tree Owner API response data: {response_data}")
                return None
            else:
                # Log if API call failed or returned unexpected format
                logger.warning(
                    "Tree Owner API call via _api_req returned unexpected data or None."
                )
                logger.debug(f"Response received: {response_data}")
                return None
        # Step 6: Handle exceptions
        except Exception as e:
            logger.error(
                f"Error fetching/parsing Tree Owner API for tree {tree_id}: {e}",
                exc_info=True,
            )
            return None

    # End of get_tree_owner

    def verify_sess(self) -> bool:
        """
        Verifies the current session status by checking if the user is logged in.
        Prioritizes API check via login_status().

        Returns:
            True if the session is verified as logged in, False otherwise.
        """
        logger.debug("Verifying session status (using login_status)...")
        # Step 1: Call login_status helper (handles API/UI checks)
        login_ok = login_status(self)

        # Step 2: Interpret result
        if login_ok is True:
            logger.debug("Session verification successful (logged in).")
            return True
        elif login_ok is False:
            logger.warning("Session verification failed (user not logged in).")
            # Note: Does not attempt re-login here, caller should handle if needed.
            return False
        else:  # login_ok is None (critical error during check)
            logger.error(
                "Session verification failed critically (login_status returned None)."
            )
            return False

    # End of verify_sess

    def _verify_api_login_status(self) -> Optional[bool]:
        """
        Checks login status specifically via the /header/dna API endpoint.
        This is the preferred method for checking login status quickly.

        Returns:
            True if API response indicates user is logged in.
            False if API response indicates user is NOT logged in (e.g., 401/403).
            None if a critical error occurs during the check (e.g., connection error,
            invalid session before check, unexpected API response format).
        """
        # Step 1: Define API endpoint and description
        api_url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        api_description = "API Login Verification (header/dna)"
        logger.debug(f"Verifying login status via API endpoint: {api_url}...")

        # Step 2: Check driver/session state prerequisites
        if not self.driver or not self.is_sess_valid():
            logger.warning(
                f"{api_description}: Driver/session not valid for API check."
            )
            return None  # Critical if driver isn't ready

        # Step 3: Ensure cookies are synced to requests.Session before API call
        try:
            logger.debug("Syncing cookies before API login check...")
            self._sync_cookies()
        except Exception as sync_e:
            logger.warning(f"Error syncing cookies before API login check: {sync_e}")
            # Proceed cautiously, API call might fail if cookies are missing/stale

        # Step 4: Make the API request using _api_req
        try:
            # Force using requests session which should have synced cookies
            response_data = _api_req(
                url=api_url,
                driver=self.driver,  # Still pass driver for potential UBE header etc.
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description=api_description,
                force_requests=True,
            )

            # Step 5: Process the response from _api_req
            if response_data is None:
                # Indicates total failure in _api_req (e.g., connection/timeout after retries)
                logger.warning(
                    f"{api_description}: _api_req returned None. Returning None."
                )
                return None

            elif isinstance(response_data, requests.Response):
                # _api_req returns the Response object on non-retryable HTTP errors
                status_code = response_data.status_code
                if status_code in [401, 403]:  # Unauthorized or Forbidden
                    logger.debug(
                        f"{api_description}: API check failed with status {status_code}. User NOT logged in."
                    )
                    return False  # Explicitly not logged in
                else:  # Other non-2xx, non-retryable errors
                    logger.warning(
                        f"{api_description}: API check failed with unexpected status {status_code}. Returning None."
                    )
                    return None  # Unexpected error state

            elif isinstance(response_data, dict):
                # Successful 2xx response, check content for expected key ('testId')
                if "testId" in response_data:
                    logger.debug(
                        f"{api_description}: API login check successful ('testId' found)."
                    )
                    return True  # Confirmed logged in
                else:
                    # Received 2xx but content is unexpected - might still be logged in?
                    logger.warning(
                        f"{api_description}: API check succeeded (2xx), but response format unexpected: {response_data}. Assuming logged in cautiously."
                    )
                    return True  # Cautiously assume logged in if 2xx

            else:
                # _api_req returned something else unexpected (e.g., plain text on 2xx?)
                logger.error(
                    f"{api_description}: _api_req returned unexpected type {type(response_data)}. Returning None."
                )
                return None  # Critical error state

        # Step 6: Handle exceptions during the API check process itself
        except Exception as e:
            logger.error(
                f"Unexpected error during {api_description}: {e}", exc_info=True
            )
            return None  # Critical error state

    # End of _verify_api_login_status

    @retry_api()  # Add retry decorator
    def get_header(self) -> bool:
        """
        Retrieves data from the header/dna API endpoint, primarily as a way
        to confirm API accessibility after login or during session checks.

        Returns:
            True if the API call is successful and returns expected data, False otherwise.
        """
        # Step 1: Check session validity
        if not self.is_sess_valid():
            logger.error("get_header: Session invalid.")
            return False

        # Step 2: Define API URL and make the call
        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        logger.debug("Attempting to fetch header/dna API data...")
        response_data = _api_req(
            url,
            self.driver,
            self,  # Pass self (SessionManager instance)
            method="GET",
            use_csrf_token=False,
            api_description="Get UUID API",  # Re-use description (it fetches testId/UUID)
        )

        # Step 3: Process response
        if response_data:
            if isinstance(response_data, dict) and "testId" in response_data:
                logger.debug("Header data retrieved successfully ('testId' found).")
                return True  # Success
            else:
                # Log error if structure is unexpected
                logger.error("Unexpected response structure from header/dna API.")
                logger.debug(f"Response: {response_data}")
                return False
        else:
            # Log error if API call failed
            logger.error(
                "Failed to get header/dna data via _api_req (returned None or error response)."
            )
            return False

    # End of get_header

    def _validate_sess_cookies(self, required_cookies: List[str]) -> bool:
        """
        Checks if all specified cookies exist in the current WebDriver session.

        Args:
            required_cookies: A list of cookie names to check for.

        Returns:
            True if all required cookies are present, False otherwise or if an error occurs.
        """
        # Step 1: Check session validity
        if not self.is_sess_valid():
            logger.warning("Cannot validate cookies: Session invalid.")
            return False
        if self.driver is None:
            logger.error("Driver is None. Cannot validate cookies.")
            return False

        # Step 2: Get current cookies
        try:
            cookies = {
                c["name"]: c["value"] for c in self.driver.get_cookies() if "name" in c
            }
            # Step 3: Check if all required cookies exist
            missing_cookies = [name for name in required_cookies if name not in cookies]
            if not missing_cookies:
                return True  # All found
            else:
                logger.debug(f"Cookie validation failed. Missing: {missing_cookies}")
                return False
        # Step 4: Handle exceptions during cookie retrieval
        except WebDriverException as e:
            logger.error(f"WebDriverException during cookie validation: {e}")
            if not self.is_sess_valid():  # Check if exception invalidated session
                logger.error(
                    "Session invalid after WebDriverException during cookie validation."
                )
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating cookies: {e}", exc_info=True)
            return False

    # End of _validate_sess_cookies

    def is_sess_logged_in(self) -> bool:
        """DEPRECATED: Use login_status() instead."""
        logger.warning("is_sess_logged_in is deprecated. Use login_status() instead.")
        return login_status(self) is True  # Delegate and return boolean

    # End of is_sess_logged_in

    def is_sess_valid(self) -> bool:
        """
        Performs a quick check to see if the WebDriver session is likely still valid
        and the browser is responsive. Handles common exceptions indicating an
        invalid session.

        Returns:
            True if the session appears valid, False otherwise.
        """
        # Step 1: Check if driver object exists
        if not self.driver:
            # Optional: logger.debug("Browser not open (driver is None).")
            return False

        # Step 2: Attempt a lightweight WebDriver command
        try:
            # Accessing window_handles requires communication with the browser driver
            _ = self.driver.window_handles
            # Step 3: Return True if command succeeds
            return True  # If no exception, session is likely valid
        # Step 4: Handle specific exceptions indicating invalid session
        except InvalidSessionIdException:
            logger.debug(
                "Session ID is invalid (browser likely closed or session terminated)."
            )
            return False
        except (
            NoSuchWindowException,
            WebDriverException,
        ) as e:  # Catch other relevant WebDriver errors
            err_str = str(e).lower()
            # Check for common error messages indicating a dead session
            if (
                "disconnected" in err_str
                or "target crashed" in err_str
                or "no such window" in err_str
                or "unable to connect" in err_str
            ):
                logger.warning(f"Session seems invalid due to WebDriverException: {e}")
                return False
            else:
                # Log other WebDriver errors but potentially return False cautiously
                logger.warning(
                    f"Unexpected WebDriverException checking session validity: {e}"
                )
                return False  # Safer to assume invalid on unexpected WebDriver error
        # Step 5: Handle any other unexpected errors
        except Exception as e:
            logger.error(
                f"Unexpected error checking session validity: {e}", exc_info=True
            )
            return False

    # End of is_sess_valid

    def close_sess(self, keep_db: bool = False):
        """
        Closes the Selenium WebDriver session and optionally disposes the
        SQLAlchemy engine (closing the DB connection pool).

        Args:
            keep_db: If True, the database connection pool is kept alive.
                     If False (default), the pool is closed.
        """
        # Step 1: Close WebDriver session if active
        if self.driver:
            logger.debug("Attempting to close WebDriver session...")
            try:
                self.driver.quit()
                logger.debug("WebDriver session quit successfully.")
            except Exception as e:
                logger.error(f"Error closing WebDriver session: {e}", exc_info=True)
            finally:
                self.driver = None  # Ensure driver is None even if quit fails
        else:
            logger.debug("No active WebDriver session to close.")

        # Step 2: Reset state flags
        self.driver_live = False
        self.session_ready = False
        self.csrf_token = None  # Clear CSRF token on session close

        # Step 3: Close DB connection pool (unless asked to keep it)
        if not keep_db:
            logger.debug("Closing database connection pool...")
            self.cls_db_conn(keep_db=False)  # Calls helper to dispose engine
        else:
            logger.debug("Keeping DB connection pool alive (keep_db=True).")

    # End of close_sess

    def restart_sess(self, url: Optional[str] = None) -> bool:
        """
        Restarts the WebDriver session by closing the current one (if any) and
        starting a new one using the two-phase start process (start_sess,
        ensure_session_ready). Optionally navigates to a specified URL after restart.

        Keeps the database connection pool alive during the restart.

        Args:
            url: Optional URL to navigate to after successful restart.

        Returns:
            True if the session restart (both phases) and optional navigation
            are successful, False otherwise.
        """
        # Step 1: Log restart attempt
        logger.warning("Restarting WebDriver session...")

        # Step 2: Close existing session, keeping DB pool alive
        self.close_sess(keep_db=True)

        # Step 3: Perform Phase 1 (Start Driver)
        start_ok = self.start_sess(action_name="Session Restart - Phase 1")
        if not start_ok:
            logger.error("Failed to restart session (Phase 1: Driver Start failed).")
            return False  # Cannot proceed if driver doesn't start

        # Step 4: Perform Phase 2 (Ensure Session Ready)
        ready_ok = self.ensure_session_ready(action_name="Session Restart - Phase 2")
        if not ready_ok:
            logger.error("Failed to restart session (Phase 2: Session Ready failed).")
            # Clean up driver started in Phase 1 if Phase 2 fails
            self.close_sess(keep_db=True)
            return False

        # Step 5: Navigate to URL if provided
        if (
            url and self.driver
        ):  # Check driver exists again after successful ready phase
            logger.info(f"Session restart successful. Re-navigating to: {url}")
            if nav_to_page(self.driver, url, selector="body", session_manager=self):
                logger.info(f"Successfully re-navigated to {url}.")
                return True  # Success (restart + navigation)
            else:
                logger.error(
                    f"Failed to re-navigate to {url} after successful restart."
                )
                return False  # Navigation failed
        elif not url:
            logger.info("Session restart successful (no navigation requested).")
            return True  # Success (restart only)
        else:
            # Should not happen if ready_ok is True, but safeguard
            logger.error(
                "Driver instance missing after successful session restart report."
            )
            return False

    # End of restart_sess

    @ensure_browser_open  # Decorator ensures browser is open before execution
    def make_tab(self) -> Optional[str]:
        """
        Creates a new browser tab and returns its window handle ID.

        Returns:
            The window handle ID (string) of the newly created tab, or None if
            creation or identification fails.
        """
        # Step 1: Get current window handles (decorator ensures driver exists)
        driver = self.driver
        if driver is None:  # Double check although decorator should prevent
            logger.error("Driver is None in make_tab despite decorator.")
            return None
        try:
            tab_list_before = driver.window_handles
            logger.debug(f"Window handles before new tab: {tab_list_before}")
        except WebDriverException as e:
            logger.error(f"Error getting window handles before new tab: {e}")
            return None

        # Step 2: Open a new tab
        try:
            driver.switch_to.new_window("tab")
            logger.debug("Executed new_window('tab') command.")
        except WebDriverException as e:
            logger.error(f"Error executing new_window('tab'): {e}")
            return None

        # Step 3: Wait for the new handle to appear
        try:
            WebDriverWait(driver, selenium_config.NEW_TAB_TIMEOUT).until(
                lambda d: len(d.window_handles) > len(tab_list_before)
            )
            # Step 4: Identify the new handle
            tab_list_after = driver.window_handles
            new_tab_handles = list(set(tab_list_after) - set(tab_list_before))
            if new_tab_handles:
                new_tab_handle = new_tab_handles[0]
                logger.debug(f"New tab handle identified: {new_tab_handle}")
                return new_tab_handle
            else:
                logger.error(
                    "Could not identify new tab handle (set difference empty)."
                )
                logger.debug(f"Handles after: {tab_list_after}")
                return None
        # Step 5: Handle timeout or errors during wait/identification
        except TimeoutException:
            logger.error("Timeout waiting for new tab handle to appear.")
            try:
                logger.debug(f"Window handles during timeout: {driver.window_handles}")
            except Exception:
                pass  # Ignore errors logging handles during error
            return None
        except (IndexError, WebDriverException) as e:
            logger.error(f"Error identifying new tab handle after wait: {e}")
            try:
                logger.debug(f"Window handles during error: {driver.window_handles}")
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred in make_tab: {e}", exc_info=True
            )
            return None

    # End of make_tab

    def check_js_errors(self):
        """Checks for new JavaScript errors in the browser console since the last check."""
        # Step 1: Check session validity
        if not self.is_sess_valid():
            # Optional: logger.debug("Skipping JS error check: Session invalid.")
            return
        if self.driver is None:
            logger.warning("Driver is None. Skipping JS error check.")
            return

        # Step 2: Check if browser logs are supported
        try:
            log_types = self.driver.log_types
            if "browser" not in log_types:
                # Optional: logger.debug("Browser log type not supported.")
                return  # Cannot get logs if type not supported
        except WebDriverException as e:
            logger.warning(f"Could not get log_types: {e}. Skipping JS error check.")
            return

        # Step 3: Retrieve browser logs
        try:
            logs = self.driver.get_log("browser")
        except WebDriverException as e:
            logger.warning(f"WebDriverException getting browser logs: {e}")
            return  # Cannot proceed if logs cannot be retrieved
        except Exception as e:
            logger.error(f"Unexpected error getting browser logs: {e}", exc_info=True)
            return

        # Step 4: Process logs
        new_errors_found = False
        most_recent_error_time_this_check = (
            self.last_js_error_check
        )  # Track latest error in *this* check

        for entry in logs:
            # Step 4a: Filter for SEVERE level errors
            if isinstance(entry, dict) and entry.get("level") == "SEVERE":
                try:
                    # Step 4b: Extract and parse timestamp
                    timestamp_ms = entry.get("timestamp")
                    if timestamp_ms:
                        timestamp_dt = datetime.fromtimestamp(
                            timestamp_ms / 1000.0, tz=timezone.utc
                        )
                        # Step 4c: Check if error is newer than last check time
                        if timestamp_dt > self.last_js_error_check:
                            new_errors_found = True
                            # Step 4d: Log the new error
                            error_message = entry.get("message", "No message")
                            # Try to extract source file/line for better context
                            source_match = re.search(r"(.+?):(\d+)", error_message)
                            source_info = (
                                f" (Source: {source_match.group(1).split('/')[-1]}:{source_match.group(2)})"
                                if source_match
                                else ""
                            )
                            logger.warning(
                                f"JS ERROR DETECTED:{source_info} {error_message}"
                            )
                            # Step 4e: Update latest error time found in this batch
                            if timestamp_dt > most_recent_error_time_this_check:
                                most_recent_error_time_this_check = timestamp_dt
                    else:
                        logger.warning(f"JS Log entry missing timestamp: {entry}")
                except Exception as parse_e:
                    logger.warning(f"Error parsing JS log entry {entry}: {parse_e}")

        # Step 5: Update last check time to the time of the most recent error found *in this specific check*
        # This prevents re-logging the same errors if checks are frequent.
        if new_errors_found:
            self.last_js_error_check = most_recent_error_time_this_check
            logger.debug(
                f"Updated last_js_error_check time to: {self.last_js_error_check}"
            )

    # End of check_js_errors


# End of SessionManager class


# ----------------------------------------------------------------------------
# Stand alone functions
# ----------------------------------------------------------------------------


# Note: Comments added explaining Match List API specific handling rationale.
# Note: Added placeholder comment regarding potential base header refactoring.
def _api_req(
    url: str,
    driver: Optional[WebDriver],
    session_manager: SessionManager,  # Now mandatory
    method: str = "GET",
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    use_csrf_token: bool = True,
    headers: Optional[Dict] = None,
    referer_url: Optional[str] = None,
    api_description: str = "API Call",
    timeout: Optional[int] = None,
    force_requests: bool = False,  # Kept for clarity, but primarily uses requests
    cookie_jar: Optional[RequestsCookieJar] = None,
    allow_redirects: bool = True,
    force_text_response: bool = False,
) -> Optional[Any]:
    """
    Makes an HTTP request using the shared requests.Session from SessionManager.
    Handles header generation (User-Agent, CSRF, UBE, contextual headers),
    cookie synchronization (implicitly via shared session, explicitly before retry loop),
    rate limiting, retries, and response processing. Includes special handling
    for the 'Match List API' call based on observed behavior.

    Args:
        url: The target URL.
        driver: WebDriver instance (used for UBE header and cookie source).
        session_manager: The active SessionManager instance.
        method: HTTP method ('GET', 'POST', etc.).
        data: Form data payload (for POST/PUT).
        json_data: JSON payload (for POST/PUT).
        use_csrf_token: Whether to include the X-CSRF-Token header.
        headers: Additional custom headers to include/override.
        referer_url: Specific Referer header value to use.
        api_description: String describing the API call for logging/context.
        timeout: Request timeout in seconds.
        force_requests: (Maintained for signature, currently ignored)
        cookie_jar: Explicit cookie jar to use (overrides session manager's).
        allow_redirects: Whether to follow redirects automatically.
        force_text_response: If True, always return response.text, even for JSON.

    Returns:
        Parsed JSON dictionary, response text string, requests.Response object on error,
        or None on complete failure after retries.
    """
    # Step 1: Validate prerequisites
    if not session_manager:
        logger.error(f"{api_description}: Aborting - SessionManager instance required.")
        return None
    if not session_manager._requests_session:
        logger.error(
            f"{api_description}: Aborting - SessionManager has no _requests_session."
        )
        return None

    # Check browser state only if driver interaction is expected (e.g., for UBE header)
    browser_needed_for_headers = True  # Assume needed unless proven otherwise
    # Note: UBE/NewRelic/Traceparent headers might be generated even if driver=None,
    # so we primarily check session validity if the *driver* object exists.
    if driver and not session_manager.is_sess_valid():
        logger.error(f"{api_description}: Aborting - Browser session invalid.")
        return None

    # Step 2: Get Retry Configuration
    max_retries = config_instance.MAX_RETRIES
    initial_delay = config_instance.INITIAL_DELAY
    backoff_factor = config_instance.BACKOFF_FACTOR
    max_delay = config_instance.MAX_DELAY
    retry_status_codes = set(config_instance.RETRY_STATUS_CODES)  # Use set

    # Step 3: Prepare Headers
    # Potential Refactoring: Consider a base set of common headers here.
    final_headers: Dict[str, str] = {}
    # Apply contextual headers first
    contextual_headers = config_instance.API_CONTEXTUAL_HEADERS.get(api_description, {})
    final_headers.update({k: v for k, v in contextual_headers.items() if v is not None})
    # Apply explicitly passed headers (override contextual)
    if headers:
        final_headers.update({k: v for k, v in headers.items() if v is not None})

    # Ensure User-Agent
    if "User-Agent" not in final_headers:
        ua = None
        if driver and session_manager.is_sess_valid():
            try:
                ua = driver.execute_script("return navigator.userAgent;")
            except Exception:
                pass  # Ignore errors getting UA from driver
        if not ua:
            ua = random.choice(config_instance.USER_AGENTS)  # Fallback
        final_headers["User-Agent"] = ua

    # Add Referer if provided and not already set
    if referer_url and "Referer" not in final_headers:
        final_headers["Referer"] = referer_url

    # Add CSRF Token if required and not already set
    if use_csrf_token and "X-CSRF-Token" not in final_headers:
        csrf_from_manager = session_manager.csrf_token
        if csrf_from_manager:
            final_headers["X-CSRF-Token"] = csrf_from_manager
        else:
            # Attempt to fetch CSRF if missing - crucial for some calls
            logger.warning(
                f"{api_description}: CSRF token required but missing from SessionManager. Attempting fresh fetch..."
            )
            fresh_csrf = session_manager.get_csrf()
            if fresh_csrf:
                final_headers["X-CSRF-Token"] = fresh_csrf
                session_manager.csrf_token = fresh_csrf  # Update manager's token
                logger.info(
                    f"{api_description}: Successfully fetched missing CSRF token."
                )
            else:
                logger.error(
                    f"{api_description}: CSRF token required but could not be fetched."
                )
                return None  # Fail if CSRF is mandatory and unavailable

    # Add UBE header if driver available
    if (
        driver
        and session_manager.is_sess_valid()
        and "ancestry-context-ube" not in final_headers
    ):
        ube_header = make_ube(driver)
        if ube_header:
            final_headers["ancestry-context-ube"] = ube_header

    # Add Ancestry User ID header if needed and available
    # Check if contextual headers *expect* the userid AND we have it
    if "ancestry-userid" in contextual_headers and session_manager.my_profile_id:
        # Check if the contextual header is set to a placeholder or missing
        if (
            contextual_headers.get("ancestry-userid") is None
            or contextual_headers.get("ancestry-userid") == ""
        ):
            final_headers["ancestry-userid"] = session_manager.my_profile_id.upper()

    # Add New Relic / Trace headers if not present
    if "newrelic" not in final_headers:
        final_headers["newrelic"] = make_newrelic(driver)
    if "traceparent" not in final_headers:
        final_headers["traceparent"] = make_traceparent(driver)
    if "tracestate" not in final_headers:
        final_headers["tracestate"] = make_tracestate(driver)

    # --- Special Handling for "Match List API" ---
    # Rationale: Observed failures when Origin header is present and redirects occur.
    # Forcing allow_redirects=False and removing Origin seems necessary for this specific endpoint currently.
    effective_allow_redirects = allow_redirects
    if api_description == "Match List API":
        # Remove potentially problematic headers
        removed_origin = final_headers.pop("Origin", None)
        # Note: Referer seems okay/needed, keep unless proven problematic.
        if removed_origin:
            logger.debug(
                f"Removed 'Origin' header specifically for '{api_description}'."
            )
        # Ensure cache-control headers are present if not already added contextually
        final_headers.setdefault("Cache-Control", "no-cache")
        final_headers.setdefault("Pragma", "no-cache")
        # Force disable redirects for this specific call
        if effective_allow_redirects:
            logger.debug(f"Forcing allow_redirects=False for '{api_description}'.")
            effective_allow_redirects = False
    # --- End Specific Handling ---

    # Step 4: Prepare Request Details
    request_timeout = timeout if timeout is not None else selenium_config.API_TIMEOUT
    req_session = session_manager._requests_session
    # Use explicit cookie jar if provided, otherwise rely on the synced session jar
    effective_cookies = (
        cookie_jar if cookie_jar is not None else None
    )  # Pass None to use session's default jar
    if cookie_jar is not None:
        logger.debug(f"Using explicitly provided cookie jar for '{api_description}'.")

    logger.debug(
        f"API Req: {method.upper()} {url} (Timeout: {request_timeout}s, AllowRedirects: {effective_allow_redirects})"
    )

    # Step 5: Execute Request with Retry Loop
    retries_left = max_retries
    last_exception = None
    delay = initial_delay

    while retries_left > 0:
        attempt = max_retries - retries_left + 1
        response: Optional[RequestsResponse] = None

        try:
            # --- Cookie Sync before each attempt ---
            # Sync from driver to the shared requests session if no explicit jar is given
            # Note: This ensures requests session has the latest cookies from the browser state
            if cookie_jar is None and driver and session_manager.is_sess_valid():
                # logger.debug(f"{api_description}: Syncing cookies before attempt {attempt}...") # Optional: Verbose log
                try:
                    session_manager._sync_cookies()
                except Exception as sync_err:
                    logger.warning(
                        f"{api_description}: Error syncing cookies (Attempt {attempt}): {sync_err}"
                    )
            # --- End Cookie Sync ---

            # Apply rate limit wait before making the request
            wait_time = session_manager.dynamic_rate_limiter.wait()
            # Optional: Log if wait was significant
            # if wait_time > 0.1: logger.debug(f"Rate limit wait: {wait_time:.2f}s")

            # Log request details (optional, can be verbose)
            if logger.isEnabledFor(logging.DEBUG):
                # Log headers safely (mask sensitive ones)
                log_hdrs = {
                    k: (
                        v[:10] + "..."
                        if k in ["Authorization", "Cookie", "X-CSRF-Token"]
                        and v
                        and len(v) > 15
                        else v
                    )
                    for k, v in final_headers.items()
                }
                logger.debug(f"  Attempt {attempt} Headers: {log_hdrs}")
                if json_data:
                    logger.debug(f"  JSON Payload: {json.dumps(json_data)[:200]}...")
                elif data:
                    logger.debug(f"  Data Payload: {str(data)[:200]}...")

            # Make the request using the requests.Session
            response = req_session.request(
                method=method.upper(),
                url=url,
                headers=final_headers,
                data=data,
                json=json_data,
                timeout=request_timeout,
                verify=True,  # Assume SSL verification needed
                allow_redirects=effective_allow_redirects,
                cookies=effective_cookies,  # Uses session cookies if None
            )
            status = response.status_code
            logger.debug(f"<-- Response Status: {status} {response.reason}")

            # --- Process Response ---
            # Step 5a: Check for retryable status codes
            if status in retry_status_codes:
                retries_left -= 1
                last_exception = HTTPError(
                    f"{status} Server Error: {response.reason}", response=response
                )
                if retries_left <= 0:
                    logger.error(
                        f"{api_description}: Failed after {max_retries} attempts (Final Status {status})."
                    )
                    return response  # Return last error response
                else:
                    # Calculate sleep time
                    sleep_time = min(
                        delay * (backoff_factor ** (attempt - 1)), max_delay
                    ) + random.uniform(0, 0.2)
                    sleep_time = max(0.1, sleep_time)
                    # Increase dynamic delay if throttled (429)
                    if status == 429:
                        session_manager.dynamic_rate_limiter.increase_delay()
                    logger.warning(
                        f"{api_description}: Status {status} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    delay *= backoff_factor
                    continue  # Next attempt

            # Step 5b: Handle redirects when allow_redirects is False
            elif 300 <= status < 400 and not effective_allow_redirects:
                logger.warning(
                    f"{api_description}: Status {status} {response.reason} (Redirects Disabled). Returning response object."
                )
                # Let the caller handle the redirect response
                return response

            # Step 5c: Handle unexpected redirects when allow_redirects is True
            elif 300 <= status < 400 and effective_allow_redirects:
                logger.warning(
                    f"{api_description}: Unexpected final status {status} {response.reason} (Redirects Enabled). Returning response."
                )
                # This shouldn't happen if requests handles redirects, but log and return
                return response

            # Step 5d: Process successful response (2xx)
            elif response.ok:
                # Signal successful request to rate limiter
                session_manager.dynamic_rate_limiter.decrease_delay()
                # Check if plain text response is forced
                if force_text_response:
                    return response.text
                # Otherwise, check content type
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    try:
                        # Handle empty JSON response
                        if not response.content:
                            logger.warning(
                                f"{api_description}: OK ({status}), JSON content-type, but response body EMPTY."
                            )
                            return None  # Treat empty JSON as None
                        return response.json()  # Parse and return JSON
                    except json.JSONDecodeError as json_err:
                        logger.error(
                            f"{api_description}: OK ({status}), but JSON decode FAILED: {json_err}"
                        )
                        logger.debug(
                            f"Response text causing failure: {response.text[:500]}"
                        )
                        return None  # Failed parsing
                # Specific handling for plain text CSRF token
                elif (
                    api_description == "CSRF Token API" and "text/plain" in content_type
                ):
                    csrf_text = response.text.strip()
                    return csrf_text if csrf_text else None
                # Default: return text for other content types
                else:
                    logger.debug(
                        f"{api_description}: OK ({status}), Content-Type '{content_type}'. Returning raw TEXT."
                    )
                    return response.text

            # Step 5e: Handle non-retryable client/server errors (4xx/5xx)
            else:
                # Log authentication errors specifically
                if status in [401, 403]:
                    logger.warning(
                        f"{api_description}: API call failed {status} {response.reason}. Session expired/invalid?"
                    )
                    # Mark session as potentially needing re-check/re-login
                    session_manager.session_ready = False
                else:
                    logger.error(
                        f"{api_description}: Non-retryable error: {status} {response.reason}."
                    )
                # Return the error response object for caller inspection
                return response

        # Step 6: Handle exceptions during the request attempt
        except requests.exceptions.RequestException as e:
            retries_left -= 1
            last_exception = e
            exception_type_name = type(e).__name__
            if retries_left <= 0:
                logger.error(
                    f"{api_description}: {exception_type_name} failed after {max_retries} attempts. Error: {e}",
                    exc_info=False,
                )
                return None  # Failed after all retries
            else:
                # Calculate sleep time
                sleep_time = min(
                    delay * (backoff_factor ** (attempt - 1)), max_delay
                ) + random.uniform(0, 0.2)
                sleep_time = max(0.1, sleep_time)
                logger.warning(
                    f"{api_description}: {exception_type_name} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s... Error: {e}"
                )
                time.sleep(sleep_time)
                delay *= backoff_factor
                continue  # Next attempt
        except Exception as e:
            # Catch any other unexpected errors during the attempt
            logger.critical(
                f"{api_description}: CRITICAL Unexpected error during request attempt {attempt}: {e}",
                exc_info=True,
            )
            return None  # Fail immediately on critical unexpected errors

    # Should only be reached if loop completes without success (e.g., max_retries = 0 initially)
    logger.error(
        f"{api_description}: Exited retry loop unexpectedly. Last Exception: {last_exception}."
    )
    return None


# End of _api_req


def make_ube(driver: Optional[WebDriver]) -> Optional[str]:
    """
    Generates the 'ancestry-context-ube' header value based on current browser state.
    Ensures the 'correlatedSessionId' matches the current 'ANCSESSIONID' cookie.

    Args:
        driver: The active Selenium WebDriver instance.

    Returns:
        The Base64 encoded UBE header string, or None if generation fails.
    """
    # Step 1: Validate driver state
    if not driver:
        logger.debug("Cannot generate UBE header: WebDriver is None.")
        return None
    try:
        # Quick check if driver is responsive
        _ = driver.window_handles
    except WebDriverException as e:
        logger.warning(
            f"Cannot generate UBE header: Session invalid/unresponsive ({type(e).__name__})."
        )
        return None

    # Step 2: Get ANCSESSIONID cookie value
    ancsessionid = None
    try:
        # Try direct cookie retrieval first
        cookie_obj = driver.get_cookie("ANCSESSIONID")
        if cookie_obj and "value" in cookie_obj:
            ancsessionid = cookie_obj["value"]
        # Fallback: Get all cookies if direct retrieval fails
        elif ancsessionid is None:
            cookies_dict = {
                c["name"]: c["value"] for c in driver.get_cookies() if "name" in c
            }
            ancsessionid = cookies_dict.get("ANCSESSIONID")

        if not ancsessionid:
            logger.warning("ANCSESSIONID cookie not found. Cannot generate UBE header.")
            return None
    except (NoSuchCookieException, WebDriverException) as cookie_e:
        logger.warning(f"Error getting ANCSESSIONID cookie for UBE header: {cookie_e}")
        return None

    # Step 3: Define UBE payload components
    # Using static eventId based on observed cURL examples
    event_id = "00000000-0000-0000-0000-000000000000"
    correlated_id = str(
        uuid.uuid4()
    )  # Generate a unique ID for this specific view/event
    # Standard screen names based on observation
    screen_name_standard = "ancestry : uk : en : dna-matches-ui : match-list : 1"
    screen_name_legacy = "ancestry uk : dnamatches-matchlistui : list"
    # Consent string example (adjust if necessary based on actual observed values)
    user_consent = "necessary|preference|performance|analytics1st|analytics3rd|advertising1st|advertising3rd|attribution3rd"

    # Step 4: Construct the UBE data dictionary
    ube_data = {
        "eventId": event_id,
        "correlatedScreenViewedId": correlated_id,
        "correlatedSessionId": ancsessionid,  # Use the retrieved session ID
        "screenNameStandard": screen_name_standard,
        "screenNameLegacy": screen_name_legacy,
        "userConsent": user_consent,
        "vendors": "adobemc",  # Observed value
        "vendorConfigurations": "{}",  # Observed value
    }

    # Step 5: Serialize, encode, and return the header value
    try:
        # Use compact JSON encoding (no spaces)
        json_payload = json.dumps(ube_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        # logger.debug(f"Generated UBE Header: {encoded_payload[:30]}...") # Optional log
        return encoded_payload
    except Exception as e:
        logger.error(f"Error encoding UBE header data: {e}", exc_info=True)
        return None


# End of make_ube


def make_newrelic(driver: Optional[WebDriver]) -> Optional[str]:
    """
    Generates the 'newrelic' header value (used for performance monitoring).
    Does not require the driver instance but kept for consistent signature.

    Args:
        driver: WebDriver instance (ignored).

    Returns:
        The Base64 encoded New Relic header string, or None on error.
    """
    try:
        # Step 1: Define static components and generate dynamic IDs
        trace_id = uuid.uuid4().hex[:16]  # 16-char trace ID
        span_id = uuid.uuid4().hex[:16]  # 16-char span ID
        # Static account/app IDs based on observation
        account_id = "1690570"
        app_id = "1588726612"
        tk = "2611750"  # Observed value

        # Step 2: Construct the data dictionary
        newrelic_data = {
            "v": [0, 1],  # Version info
            "d": {
                "ty": "Browser",  # Type
                "ac": account_id,  # Account ID
                "ap": app_id,  # App ID
                "id": span_id,  # Span ID
                "tr": trace_id,  # Trace ID
                "ti": int(time.time() * 1000),  # Timestamp (milliseconds)
                "tk": tk,
            },
        }

        # Step 3: Serialize, encode, and return
        json_payload = json.dumps(newrelic_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        return encoded_payload
    except Exception as e:
        logger.error(f"Error generating NewRelic header: {e}", exc_info=True)
        return None


# End of make_newrelic


def make_traceparent(driver: Optional[WebDriver]) -> Optional[str]:
    """
    Generates the 'traceparent' header value according to W3C Trace Context spec.
    Does not require the driver instance but kept for consistent signature.

    Args:
        driver: WebDriver instance (ignored).

    Returns:
        The traceparent header string, or None on error.
    """
    try:
        # Step 1: Define components based on W3C spec
        version = "00"  # Current version
        trace_id = uuid.uuid4().hex  # 32-char hex trace ID
        parent_id = uuid.uuid4().hex[:16]  # 16-char hex parent/span ID
        flags = "01"  # Sampled flag (01 = sampled)

        # Step 2: Construct the header string
        traceparent = f"{version}-{trace_id}-{parent_id}-{flags}"
        return traceparent
    except Exception as e:
        logger.error(f"Error generating traceparent header: {e}", exc_info=True)
        return None


# End of make_traceparent


def make_tracestate(driver: Optional[WebDriver]) -> Optional[str]:
    """
    Generates the 'tracestate' header value (W3C Trace Context), potentially
    including vendor-specific information (e.g., New Relic).
    Does not require the driver instance but kept for consistent signature.

    Args:
        driver: WebDriver instance (ignored).

    Returns:
        The tracestate header string, or None on error.
    """
    try:
        # Step 1: Define components (example using New Relic format based on observation)
        tk = "2611750"
        account_id = "1690570"
        app_id = "1588726612"
        span_id = uuid.uuid4().hex[:16]
        timestamp = int(time.time() * 1000)

        # Step 2: Construct the tracestate string (New Relic format example)
        # Format: {vendor_id}@{vendor_specific_data}
        # NR format: {tk}@nr=0-1-{acc_id}-{app_id}-{span_id}----{timestamp}
        tracestate = f"{tk}@nr=0-1-{account_id}-{app_id}-{span_id}----{timestamp}"
        return tracestate
    except Exception as e:
        logger.error(f"Error generating tracestate header: {e}", exc_info=True)
        return None


# End of make_tracestate


def get_driver_cookies(driver: WebDriver) -> Dict[str, str]:
    """
    Retrieves all cookies from the Selenium WebDriver session as a simple dictionary.

    Args:
        driver: The active Selenium WebDriver instance.

    Returns:
        A dictionary mapping cookie names to their values, or an empty dictionary
        if retrieval fails or the driver is invalid.
    """
    # Step 1: Check driver validity
    if not driver:
        logger.warning("Cannot get driver cookies: WebDriver is None.")
        return {}
    # Step 2: Attempt to retrieve cookies
    try:
        cookies_list = driver.get_cookies()
        # Step 3: Convert list of cookie dicts to a simple name:value dict
        cookies_dict = {
            cookie["name"]: cookie["value"]
            for cookie in cookies_list
            if "name" in cookie
        }
        return cookies_dict
    # Step 4: Handle exceptions during retrieval
    except WebDriverException as e:
        logger.error(f"WebDriverException getting driver cookies: {e}")
        # Check session validity again after error
        if not is_browser_open(driver):
            logger.error("Session invalid after WebDriverException getting cookies.")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error getting driver cookies: {e}", exc_info=True)
        return {}


# End of get_driver_cookies


# Note: Constants used for message status defined at the top of the file.
def _send_message_via_api(
    session_manager: SessionManager,
    person: Person,
    message_text: str,
    existing_conv_id: Optional[str],
    log_prefix: str,  # Log prefix for context (e.g., Person ID/Username)
) -> Tuple[Optional[str], Optional[str]]:
    """
    Sends a message using the appropriate Ancestry messaging API endpoint
    (creating a new conversation or adding to an existing one).
    Handles different application modes (dry_run, testing, production) for safety.

    Args:
        session_manager: The active SessionManager instance.
        person: The Person object representing the recipient.
        message_text: The content of the message to send.
        existing_conv_id: The conversation ID if adding to an existing thread,
                          None if creating a new conversation.
        log_prefix: A string prefix for log messages related to this send operation.

    Returns:
        A tuple containing:
        - message_status (str): Status code (e.g., SEND_SUCCESS_DELIVERED,
          SEND_SUCCESS_DRY_RUN, SEND_ERROR_POST_FAILED, etc.)
        - effective_conv_id (Optional[str]): The conversation ID (existing or newly
          created by the API or simulated in dry_run), or None on failure.
    """
    # --- Step 1: Validate Inputs & Get Required IDs ---
    if not session_manager or not session_manager.my_profile_id:
        logger.error(
            f"{log_prefix}: Cannot send message - SessionManager or own profile ID missing."
        )
        return SEND_ERROR_MISSING_OWN_ID, None
    if not person or not person.profile_id:
        logger.error(
            f"{log_prefix}: Cannot send message - Invalid Person object or missing profile ID."
        )
        return SEND_ERROR_INVALID_RECIPIENT, None

    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()
    MY_PROFILE_ID_UPPER = session_manager.my_profile_id.upper()
    recipient_profile_id_upper = person.profile_id.upper()

    # Step 2: Determine Mode and Handle Dry Run
    is_initial = not existing_conv_id
    app_mode = config_instance.APP_MODE

    if app_mode == "dry_run":
        # Simulate success, generate a fake conversation ID if needed
        message_status = SEND_SUCCESS_DRY_RUN
        effective_conv_id = existing_conv_id or f"dryrun_{uuid.uuid4()}"
        logger.debug(f"{log_prefix}: Dry Run - Message send simulation successful.")
        return message_status, effective_conv_id

    # Step 3: Validate Production/Testing Mode (Safety Check)
    if app_mode not in ["production", "testing"]:
        logger.error(f"{log_prefix}: Logic Error - Unexpected APP_MODE '{app_mode}' reached send logic.")
        return SEND_ERROR_INTERNAL_MODE, None

    # --- Proceed with Actual API Send (Production/Testing) ---

    # Step 4: Determine API URL, Payload, Headers, and Description
    send_api_url: str = ""
    payload: Dict[str, Any] = {}
    send_api_desc: str = ""
    api_headers: Dict[str, Any] = {}

    if is_initial:
        # API endpoint for creating a new conversation
        send_api_url = urljoin(
            config_instance.BASE_URL.rstrip("/") + "/",
            "app-api/express/v2/conversations/message",
        )
        send_api_desc = "Create Conversation API"
        # Payload requires content, author, and members list
        payload = {
            "content": message_text,
            "author": MY_PROFILE_ID_LOWER,
            "index": 0, # Observed value, purpose unclear
            "created": 0, # Observed value, purpose unclear
            "conversation_members": [
                {"user_id": recipient_profile_id_upper.lower(), "family_circles": []}, # Recipient
                {"user_id": MY_PROFILE_ID_LOWER}, # Sender (Me)
            ],
        }
    elif existing_conv_id:
        # API endpoint for sending a message to an existing conversation
        send_api_url = urljoin(
            config_instance.BASE_URL.rstrip("/") + "/",
            f"app-api/express/v2/conversations/{existing_conv_id}",
        )
        send_api_desc = "Send Message API (Existing Conv)"
        # Payload only requires content and author
        payload = {"content": message_text, "author": MY_PROFILE_ID_LOWER}
    else:
        # This state should not be reached if logic is correct
        logger.error(f"{log_prefix}: Logic Error - Cannot determine API URL/payload (existing_conv_id issue?).")
        return SEND_ERROR_API_PREP_FAILED, None

    # Step 5: Prepare Headers using contextual settings from config
    ctx_headers = config_instance.API_CONTEXTUAL_HEADERS.get(send_api_desc, {})
    api_headers = ctx_headers.copy() # Start with contextual headers
    # Ensure ancestry-userid header uses the correct uppercase ID
    if "ancestry-userid" in api_headers:
        api_headers["ancestry-userid"] = MY_PROFILE_ID_UPPER

    # Step 6: Make the API call using the _api_req helper
    # force_requests=True uses the shared requests.Session, bypassing direct driver JS execution
    api_response = _api_req(
        url=send_api_url,
        driver=session_manager.driver, # Pass driver for context/headers if needed by _api_req
        session_manager=session_manager,
        method="POST",
        json_data=payload,
        use_csrf_token=False, # Messaging API seems to use different auth/session mechanism
        headers=api_headers,
        api_description=send_api_desc,
        force_requests=True,
    )

    # Step 7: Validate the API response
    message_status = SEND_ERROR_UNKNOWN # Default to unknown error
    new_conversation_id_from_api: Optional[str] = None
    post_ok = False
    api_conv_id: Optional[str] = None
    api_author: Optional[str] = None

    if api_response is not None:
        if isinstance(api_response, dict):
            # Successful API call returned a dictionary
            if is_initial:
                # Validate response for creating a new conversation
                api_conv_id = str(api_response.get("conversation_id", "")) # Ensure string
                msg_details = api_response.get("message", {})
                api_author = str(msg_details.get("author", "")).upper() if isinstance(msg_details, dict) else None
                # Check if conversation ID is present and author matches sender
                if api_conv_id and api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = api_conv_id
                else:
                    logger.error(f"{log_prefix}: API initial response invalid (ConvID: {api_conv_id}, Author: {api_author}).")
            else: # Existing conversation
                # Validate response for sending to existing conversation
                api_author = str(api_response.get("author", "")).upper()
                if api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    # Conversation ID remains the existing one
                    new_conversation_id_from_api = existing_conv_id
                else:
                    logger.error(f"{log_prefix}: API follow-up author validation failed (Author: {api_author}).")

        elif isinstance(api_response, requests.Response):
            # API call failed with an HTTP error response object
            message_status = f"send_error (http_{api_response.status_code})"
            logger.error(f"{log_prefix}: API POST ({send_api_desc}) failed with status {api_response.status_code}.")
        else:
            # API call returned something unexpected (not dict, not Response)
            logger.error(f"{log_prefix}: API call ({send_api_desc}) unexpected success format. Type:{type(api_response)}, Resp:{api_response}")
            message_status = SEND_ERROR_UNEXPECTED_FORMAT

        # Step 8: Determine final status based on validation
        if post_ok:
            message_status = SEND_SUCCESS_DELIVERED
            logger.debug(f"{log_prefix}: Message send to {log_prefix} ACCEPTED by API.")
        elif message_status == SEND_ERROR_UNKNOWN: # If status hasn't been set by specific error
            # This occurs if the response was a dict but validation failed
            message_status = SEND_ERROR_VALIDATION_FAILED
            logger.warning(f"{log_prefix}: API POST validation failed after receiving unexpected success response.")
    else:
        # API call failed completely (e.g., _api_req returned None after retries)
        message_status = SEND_ERROR_POST_FAILED
        logger.error(f"{log_prefix}: API POST ({send_api_desc}) failed (No response/Retries exhausted).")

    # Step 9: Return the final status and potentially the new/existing conversation ID
    return message_status, new_conversation_id_from_api
# End of _send_message_via_api


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_profile_details_for_person(session_manager: SessionManager, profile_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches profile details (first name, contactable status, last login) for a specific Person
    using their Profile ID. Used primarily by Action 7 when creating new Person records.

    Args:
        session_manager: The active SessionManager instance.
        profile_id: The Profile ID (ucdmid) of the person whose details are needed.

    Returns:
        A dictionary containing 'first_name', 'contactable', 'last_logged_in_dt',
        or None if the fetch fails or essential data is missing.
    """
    # Step 1: Validate inputs and session state
    if not profile_id:
        logger.warning("_fetch_profile_details_for_person: Profile ID missing.")
        return None
    if not session_manager or not session_manager.my_profile_id:
        logger.error("_fetch_profile_details_for_person: SessionManager or own profile ID missing.")
        return None
    # Check session validity *before* API call
    if not session_manager.is_sess_valid():
        logger.error(f"_fetch_profile_details_for_person: Session invalid for Profile ID {profile_id}.")
        # Raise exception so retry_api can handle potential session restart if configured
        raise ConnectionError(f"WebDriver session invalid before profile details fetch (Profile: {profile_id})")

    # Step 2: Construct URL and Referer
    api_description = "Profile Details API (Action 7)" # Use specific description for context
    profile_url = urljoin(config_instance.BASE_URL, f"/app-api/express/v1/profiles/details?userId={profile_id.upper()}")
    # Referer for this context is typically the messaging page
    referer_url = urljoin(config_instance.BASE_URL, "/messaging/")
    logger.debug(f"Fetching profile details ({api_description}) for Profile ID {profile_id}...")

    # Step 3: Make API call using _api_req
    try:
        profile_response = _api_req(
            url=profile_url,
            driver=session_manager.driver, # Pass driver for context/headers
            session_manager=session_manager,
            method="GET",
            headers={}, # Contextual headers applied by _api_req based on description
            use_csrf_token=False, # Not typically needed for profile details GET
            api_description=api_description,
            referer_url=referer_url,
        )

        # Step 4: Process the response
        if profile_response and isinstance(profile_response, dict):
            logger.debug(f"Successfully fetched profile details for {profile_id}.")
            result_data: Dict[str, Any] = {}

            # Extract First Name (use 'FirstName' key from response)
            first_name_raw = profile_response.get("FirstName")
            if first_name_raw and isinstance(first_name_raw, str):
                 # Format the name consistently
                 result_data["first_name"] = format_name(first_name_raw)
            else:
                 # Fallback if FirstName missing, try formatting display name if available
                 display_name_raw = profile_response.get("DisplayName")
                 if display_name_raw:
                     formatted_dn = format_name(display_name_raw)
                     # Use the first part of the formatted display name
                     result_data["first_name"] = formatted_dn.split()[0] if formatted_dn != "Valued Relative" else None
                 else:
                      result_data["first_name"] = None # Set explicitly to None if no name found

            # Extract Contactable Status (use 'IsContactable' key)
            contactable_val = profile_response.get("IsContactable")
            result_data["contactable"] = bool(contactable_val) if contactable_val is not None else False # Default False

            # Extract and Parse Last Login Date (use 'LastLoginDate' key)
            last_login_str = profile_response.get("LastLoginDate")
            last_logged_in_dt_aware: Optional[datetime] = None
            if last_login_str:
                try:
                    # Handle standard ISO format, potentially with Z for Zulu/UTC
                    if last_login_str.endswith("Z"):
                        # Replace Z with +00:00 for standard parsing
                        last_logged_in_dt_aware = datetime.fromisoformat(last_login_str.replace("Z", "+00:00"))
                    else:
                        # Assume ISO format, ensure timezone aware (UTC)
                        dt_naive = datetime.fromisoformat(last_login_str)
                        last_logged_in_dt_aware = dt_naive.replace(tzinfo=timezone.utc) if dt_naive.tzinfo is None else dt_naive.astimezone(timezone.utc)
                    result_data["last_logged_in_dt"] = last_logged_in_dt_aware # Store aware datetime
                except (ValueError, TypeError) as date_parse_err:
                     logger.warning(f"Could not parse LastLoginDate '{last_login_str}' for {profile_id}: {date_parse_err}")
                     result_data["last_logged_in_dt"] = None # Set to None on parsing error
            else:
                 result_data["last_logged_in_dt"] = None # Set to None if key missing

            return result_data

        # Handle error responses from _api_req
        elif isinstance(profile_response, requests.Response):
             logger.warning(f"Failed profile details fetch for {profile_id}. Status: {profile_response.status_code}.")
             return None
        # Handle None or other unexpected types from _api_req
        else:
             logger.warning(f"Failed profile details fetch for {profile_id} (Invalid response type: {type(profile_response)}).")
             return None

    # Step 5: Handle exceptions during the process
    except ConnectionError as conn_err: # Specifically catch ConnectionError for retry
        logger.error(f"ConnectionError fetching profile details for {profile_id}: {conn_err}", exc_info=False)
        raise # Re-raise for retry_api decorator
    except requests.exceptions.RequestException as req_e: # Catch other requests errors for retry
        logger.error(f"RequestException fetching profile details for {profile_id}: {req_e}", exc_info=False)
        raise # Re-raise for retry_api decorator
    except Exception as e:
        logger.error(f"Unexpected error fetching profile details for {profile_id}: {e}", exc_info=True)
        return None # Return None for non-retryable unexpected errors
# End of _fetch_profile_details_for_person


# ----------------------------------------------------------------------------
# Login Functions
# ----------------------------------------------------------------------------

# Login Helper 5
@time_wait("Handle 2FA Page") # Apply timing decorator
def handle_twoFA(session_manager: SessionManager) -> bool:
    """
    Handles the Ancestry Two-Factor Authentication (2FA) page interaction.
    Waits for the page, clicks the SMS send button, and waits for user action
    by monitoring the disappearance of 2FA page elements.

    Args:
        session_manager: The active SessionManager instance.

    Returns:
        True if 2FA is successfully handled (login confirmed after), False otherwise.
    """
    # Step 1: Validate driver state
    if session_manager.driver is None:
        logger.error("handle_twoFA: SessionManager driver is None. Cannot proceed.")
        return False
    driver = session_manager.driver # Use local variable for clarity

    # Step 2: Initialize waits
    element_wait = selenium_config.element_wait(driver)
    page_wait = selenium_config.page_wait(driver) # Not strictly used here, but available
    short_wait = selenium_config.short_wait(driver) # For clickable elements

    try:
        logger.debug("Handling Two-Factor Authentication (2FA)...")

        # Step 3: Wait for 2FA page indicator to be present
        try:
            logger.debug(f"Waiting for 2FA page header using selector: '{TWO_STEP_VERIFICATION_HEADER_SELECTOR}'")
            element_wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR))
            )
            logger.debug("2FA page header detected.")
        except TimeoutException:
            # If header not found, check if maybe we logged in successfully anyway
            logger.debug("Did not detect 2FA page header within timeout.")
            if login_status(session_manager):
                logger.info("User appears logged in after checking for 2FA page. Assuming 2FA handled/skipped.")
                return True
            # If not logged in and header not found, it's an unexpected state
            logger.warning("Assuming 2FA not required or page didn't load correctly (header missing).")
            return False

        # Step 4: Wait for and click the SMS 'Send Code' button
        try:
            logger.debug(f"Waiting for 2FA 'Send Code' (SMS) button: '{TWO_FA_SMS_SELECTOR}'")
            # Wait for element to be clickable
            sms_button_clickable = short_wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR)))
            if sms_button_clickable:
                logger.debug("Attempting to click 'Send Code' button using JavaScript...")
                # Use JavaScript click for potential overlay robustness
                driver.execute_script("arguments[0].click();", sms_button_clickable)
                logger.debug("'Send Code' button clicked.")
                # Step 4a: Wait briefly for code input field to appear (visual confirmation)
                try:
                    logger.debug(f"Waiting for 2FA code input field: '{TWO_FA_CODE_INPUT_SELECTOR}'")
                    WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CSS_SELECTOR, TWO_FA_CODE_INPUT_SELECTOR)))
                    logger.debug("Code input field appeared after clicking 'Send Code'.")
                except TimeoutException:
                    logger.warning("Code input field did not appear/become visible after clicking 'Send Code'.")
                except Exception as e_input:
                    logger.error(f"Error waiting for 2FA code input field: {e_input}. Check selector: {TWO_FA_CODE_INPUT_SELECTOR}")
                    # Proceed cautiously, but log the error
            else:
                # Should not happen if wait_until clickable succeeds
                logger.error("'Send Code' button found but not clickable.")
                return False
        except TimeoutException:
            logger.error("Timeout finding or waiting for clickability of the 2FA 'Send Code' button.")
            return False
        except ElementNotInteractableException:
            logger.error("'Send Code' button not interactable (potentially obscured).")
            return False
        except Exception as e:
            logger.error(f"Error clicking 2FA 'Send Code' button: {e}", exc_info=True)
            return False

        # Step 5: Wait for user action (manual code entry)
        # Monitor for the disappearance of 2FA elements as indication of submission
        code_entry_timeout = selenium_config.TWO_FA_CODE_ENTRY_TIMEOUT
        logger.warning(f"Waiting up to {code_entry_timeout}s for user to manually enter 2FA code and submit...")
        start_time = time.time()
        user_action_detected = False
        while time.time() - start_time < code_entry_timeout:
            try:
                # Check if the 2FA header is STILL visible using a very short wait
                WebDriverWait(driver, 0.5).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR))
                )
                # If visible, user hasn't submitted yet, continue polling
                time.sleep(2) # Poll every 2 seconds
            except TimeoutException:
                # Header DISAPPEARED - assume user submitted code
                logger.info("2FA page elements disappeared, assuming user submitted code.")
                user_action_detected = True
                break # Exit the polling loop
            except WebDriverException as e:
                logger.error(f"WebDriver error checking for 2FA header during wait: {e}")
                break # Exit loop, session might be dead
            except Exception as e:
                logger.error(f"Unexpected error checking for 2FA header during wait: {e}")
                break # Exit loop on unexpected error

        # Step 6: Final Verification after loop/action
        if user_action_detected:
            logger.info("Re-checking login status after 2FA page disappearance...")
            time.sleep(1) # Short pause for page transition
            final_status = login_status(session_manager) # Check status again
            if final_status is True:
                logger.info("User completed 2FA successfully (login confirmed after page change).")
                return True
            else:
                logger.error("2FA page disappeared, but final login status check failed or returned False.")
                return False
        else: # Loop timed out
            logger.error(f"Timed out ({code_entry_timeout}s) waiting for user 2FA action (page did not change).")
            return False

    # Step 7: Handle exceptions during the overall 2FA process
    except WebDriverException as e:
        logger.error(f"WebDriverException during 2FA handling: {e}")
        if not is_browser_open(driver): # Check if session died
            logger.error("Session invalid after WebDriverException during 2FA.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during 2FA handling: {e}", exc_info=True)
        return False
# End of handle_twoFA

# Login Helper 4
def enter_creds(driver: WebDriver) -> bool:
    """
    Enters username and password into login fields and attempts to click the Sign In button.
    Includes robustness checks and fallbacks (JS click, Enter key).

    Args:
        driver: The Selenium WebDriver instance positioned on the login page.

    Returns:
        True if credentials were entered and a submit action (click or Enter)
        was successfully initiated, False otherwise.
    """
    # Step 1: Initialize waits and add small initial delay
    element_wait = selenium_config.element_wait(driver)
    short_wait = selenium_config.short_wait(driver)
    time.sleep(random.uniform(0.5, 1.0)) # Human-like pause

    try:
        logger.debug("Entering Credentials and Signing In...")

        # --- Step 2: Enter Username ---
        logger.debug(f"Waiting for username input: '{USERNAME_INPUT_SELECTOR}'...")
        username_input = element_wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR)))
        logger.debug("Username input field found.")
        # Clear field robustly
        try:
            username_input.click(); time.sleep(0.2)
            username_input.clear(); time.sleep(0.2)
        except Exception as e:
            logger.warning(f"Issue clicking/clearing username field ({e}). Attempting JS clear.")
            try: driver.execute_script("arguments[0].value = '';", username_input)
            except Exception as js_e: logger.error(f"Failed JS clear username: {js_e}"); return False
        # Get username from config and send keys
        ancestry_username = config_instance.ANCESTRY_USERNAME
        if not ancestry_username: raise ValueError("ANCESTRY_USERNAME configuration is missing.")
        logger.debug(f"Entering username...") # Don't log username itself
        username_input.send_keys(ancestry_username)
        logger.debug("Username entered.")
        time.sleep(0.3) # Pause

        # --- Step 3: Enter Password ---
        logger.debug(f"Waiting for password input: '{PASSWORD_INPUT_SELECTOR}'...")
        password_input = element_wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR)))
        logger.debug("Password input field found.")
        # Clear field robustly
        try:
            password_input.click(); time.sleep(0.2)
            password_input.clear(); time.sleep(0.2)
        except Exception as e:
            logger.warning(f"Issue clicking/clearing password field ({e}). Attempting JS clear.")
            try: driver.execute_script("arguments[0].value = '';", password_input)
            except Exception as js_e: logger.error(f"Failed JS clear password: {js_e}"); return False
        # Get password from config and send keys
        ancestry_password = config_instance.ANCESTRY_PASSWORD
        if not ancestry_password: raise ValueError("ANCESTRY_PASSWORD configuration is missing.")
        logger.debug("Entering password: ***")
        password_input.send_keys(ancestry_password)
        logger.debug("Password entered.")
        time.sleep(0.5) # Pause

        # --- Step 4: Locate and Click Sign In Button ---
        sign_in_button = None
        try:
            # Wait for button presence first, then clickability
            logger.debug(f"Waiting for sign in button presence: '{SIGN_IN_BUTTON_SELECTOR}'...")
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR)))
            logger.debug("Waiting for sign in button clickability...")
            sign_in_button = short_wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR)))
            logger.debug("Sign in button located and deemed clickable.")
        except TimeoutException:
            # Fallback 1: If button not found/clickable, try sending Enter key
            logger.error("Sign in button not found or not clickable within timeout.")
            logger.warning("Attempting fallback: Sending RETURN key to password field.")
            try:
                password_input.send_keys(Keys.RETURN)
                logger.info("Fallback RETURN key sent to password field.")
                return True # Assume submission initiated
            except Exception as key_e:
                logger.error(f"Failed to send RETURN key: {key_e}")
                return False # Both button click and key press failed
        except Exception as find_e:
            logger.error(f"Unexpected error finding sign in button: {find_e}")
            return False

        # --- Step 5: Attempt Click Actions (if button found) ---
        click_successful = False
        if sign_in_button:
            # Attempt 1: Standard Click
            try:
                logger.debug("Attempting standard click on sign in button...")
                sign_in_button.click()
                logger.debug("Standard click executed.")
                click_successful = True
            except ElementClickInterceptedException:
                logger.warning("Standard click intercepted. Trying JS click...")
            except ElementNotInteractableException as eni_e:
                logger.warning(f"Button not interactable for standard click: {eni_e}. Trying JS click...")
            except Exception as click_e:
                logger.error(f"Error during standard click: {click_e}. Trying JS click...")

            # Attempt 2: JavaScript Click (if standard failed)
            if not click_successful:
                try:
                    logger.debug("Attempting JavaScript click on sign in button...")
                    driver.execute_script("arguments[0].click();", sign_in_button)
                    logger.info("JavaScript click executed.")
                    click_successful = True
                except Exception as js_click_e:
                    logger.error(f"Error during JavaScript click: {js_click_e}")

            # Attempt 3: Send Enter Key (if both clicks failed)
            if not click_successful:
                logger.warning("Both standard and JS clicks failed. Attempting fallback: Sending RETURN key.")
                try:
                    password_input.send_keys(Keys.RETURN)
                    logger.info("Fallback RETURN key sent to password field after failed clicks.")
                    click_successful = True # Assume submission initiated
                except Exception as key_e:
                    logger.error(f"Failed to send RETURN key as final fallback: {key_e}")

        # Step 6: Return overall success of submit action
        return click_successful

    # Step 7: Handle Exceptions during the process
    except TimeoutException as e:
        logger.error(f"Timeout finding username or password input field: {e}")
        return False
    except NoSuchElementException as e: # Should be caught by Wait, but safeguard
        logger.error(f"Username or password input not found (NoSuchElement): {e}")
        return False
    except ValueError as ve: # Catch missing config error
        logger.critical(f"Configuration Error: {ve}")
        return False
    except WebDriverException as e:
        logger.error(f"WebDriver error entering credentials: {e}")
        if not is_browser_open(driver): logger.error("Session invalid during credential entry.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error entering credentials: {e}", exc_info=True)
        return False
# End of enter_creds

# Login Helper 3
@retry(MAX_RETRIES=2, BACKOFF_FACTOR=1, MAX_DELAY=3) # Retry consent handling
def consent(driver: WebDriver) -> bool:
    """
    Handles cookie consent banners/modals.
    Prioritizes removing the overlay via JavaScript, falls back to clicking
    a specific 'Accept' button if JS removal fails.

    Args:
        driver: The Selenium WebDriver instance.

    Returns:
        True if the consent banner was handled successfully (removed or accepted),
        False otherwise.
    """
    # Step 1: Basic validation
    if not driver:
        logger.error("consent: WebDriver instance is None.")
        return False

    # Step 2: Check for the presence of the cookie banner
    logger.debug(f"Checking for cookie consent overlay: '{COOKIE_BANNER_SELECTOR}'")
    overlay_element = None
    try:
        # Use a short wait to detect the banner quickly
        overlay_element = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)))
        logger.debug("Cookie consent overlay DETECTED.")
    except TimeoutException:
        logger.debug("Cookie consent overlay not found. Assuming no consent needed.")
        return True # No banner found, success
    except Exception as e:
        logger.error(f"Error checking for consent banner: {e}")
        return False # Error during detection

    # --- Banner Detected, Attempt Handling ---
    # Step 3: Attempt 1 - Remove overlay using JavaScript
    removed_via_js = False
    if overlay_element:
        try:
            logger.debug("Attempting JS removal of consent overlay...")
            driver.execute_script("arguments[0].remove();", overlay_element)
            # Verify removal by checking if element is gone after short pause
            time.sleep(0.5)
            try:
                # If find_element succeeds here, removal failed
                driver.find_element(By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                logger.warning("Consent overlay still present after JS removal attempt.")
            except NoSuchElementException:
                # If find_element fails (NoSuchElementException), removal was successful
                logger.debug("Cookie consent overlay REMOVED successfully via JS.")
                removed_via_js = True
                return True # Success via JS removal
        except WebDriverException as js_err:
            logger.warning(f"Error removing consent overlay via JS: {js_err}")
        except Exception as e:
            logger.warning(f"Unexpected error during JS removal of consent: {e}")

    # Step 4: Attempt 2 - Click specific 'Accept' button (if JS failed)
    if not removed_via_js:
        logger.debug(f"JS removal failed/skipped. Trying specific accept button: '{consent_ACCEPT_BUTTON_SELECTOR}'")
        try:
            # Wait for the specific accept button to be clickable
            accept_button = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.CSS_SELECTOR, consent_ACCEPT_BUTTON_SELECTOR)))
            if accept_button:
                logger.info("Found specific clickable accept button.")
                # Step 4a: Attempt standard click
                try:
                    accept_button.click()
                    logger.info("Clicked accept button successfully.")
                    time.sleep(1) # Wait for banner to disappear
                    # Verify removal again
                    try:
                        driver.find_element(By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                        logger.warning("Consent overlay still present after clicking accept button.")
                        return False # Click didn't remove it
                    except NoSuchElementException:
                        logger.debug("Consent overlay gone after clicking accept button.")
                        return True # Success via button click
                # Step 4b: Handle click interception, try JS click
                except ElementClickInterceptedException:
                    logger.warning("Click intercepted for accept button, trying JS click...")
                    try:
                        driver.execute_script("arguments[0].click();", accept_button)
                        logger.info("Clicked accept button via JS successfully.")
                        time.sleep(1)
                        # Verify removal
                        try:
                            driver.find_element(By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                            logger.warning("Consent overlay still present after JS clicking accept button.")
                            return False
                        except NoSuchElementException:
                            logger.debug("Consent overlay gone after JS clicking accept button.")
                            return True # Success via JS click
                    except Exception as js_click_err:
                        logger.error(f"Failed JS click for accept button: {js_click_err}")
                # Step 4c: Handle other click errors
                except Exception as click_err:
                    logger.error(f"Error clicking accept button: {click_err}")
        # Step 4d: Handle errors finding/clicking the specific button
        except TimeoutException:
            logger.warning(f"Specific accept button '{consent_ACCEPT_BUTTON_SELECTOR}' not found or not clickable.")
        except Exception as find_err:
            logger.error(f"Error finding/clicking specific accept button: {find_err}")

        # If both JS and button click failed
        logger.warning("Could not remove consent overlay via JS or button click.")
        return False

    # Fallback return, should not be reached if logic is correct
    return False
# End of consent

# Login Main Function 2
def log_in(session_manager: SessionManager) -> str:
    """
    Performs the login process on Ancestry.com.
    Handles navigation, consent, credential entry, and 2FA.

    Args:
        session_manager: The active SessionManager instance.

    Returns:
        A string indicating the login outcome (e.g., "LOGIN_SUCCEEDED",
        "LOGIN_FAILED_BAD_CREDS", "LOGIN_FAILED_2FA_HANDLING", etc.).
    """
    # Step 1: Validate driver state
    driver = session_manager.driver
    if not driver:
        logger.error("Login failed: WebDriver not available in SessionManager.")
        return "LOGIN_ERROR_NO_DRIVER"

    # Step 2: Define target URL and initialize waits
    signin_url = urljoin(config_instance.BASE_URL, "account/signin")

    try:
        # Step 3: Navigate to the sign-in page
        logger.info(f"Navigating to sign-in page: {signin_url}")
        # Wait for username input as confirmation page loaded
        if not nav_to_page(driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager):
            # If nav fails, check if maybe already logged in
            logger.debug("Navigation to sign-in page failed/redirected. Checking login status...")
            current_status = login_status(session_manager)
            if current_status is True:
                logger.info("Detected as already logged in. Login considered successful.")
                return "LOGIN_SUCCEEDED"
            else:
                logger.error("Failed to navigate to login page (and not logged in).")
                return "LOGIN_FAILED_NAVIGATION"
        logger.debug("Successfully navigated to sign-in page.")

        # Step 4: Handle cookie consent banner (if present)
        if not consent(driver):
            # Log warning but continue, login might still work
            logger.warning("Failed to handle consent banner, login might be impacted.")

        # Step 5: Enter login credentials
        if not enter_creds(driver):
            logger.error("Failed during credential entry or submission.")
            # Check for specific "Invalid Credentials" message
            try:
                WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.CSS_SELECTOR, FAILED_LOGIN_SELECTOR)))
                logger.error("Login failed: Specific 'Invalid Credentials' alert detected.")
                return "LOGIN_FAILED_BAD_CREDS"
            except TimeoutException:
                # Check for any generic error alert
                generic_alert_selector = "div.alert[role='alert']" # Common pattern for alerts
                try:
                    alert_element = WebDriverWait(driver, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, generic_alert_selector)))
                    alert_text = alert_element.text if alert_element else 'Unknown error'
                    logger.error(f"Login failed: Generic alert found: '{alert_text}'.")
                    return "LOGIN_FAILED_ERROR_DISPLAYED"
                except TimeoutException:
                    logger.error("Login failed: Credential entry failed, but no specific or generic alert found.")
                    return "LOGIN_FAILED_CREDS_ENTRY"
            except Exception as e:
                logger.warning(f"Error checking for login error message after cred entry failed: {e}")
                return "LOGIN_FAILED_CREDS_ENTRY" # Fallback if error check fails

        # Step 6: Wait for page change after submission
        logger.debug("Credentials submitted. Waiting for potential page change...")
        time.sleep(random.uniform(3.0, 5.0)) # Allow time for redirect/load

        # Step 7: Check for 2-Step Verification page
        two_fa_present = False
        try:
            # Check visibility of the specific 2FA header
            WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)))
            two_fa_present = True
            logger.info("Two-step verification page detected.")
        except TimeoutException:
            logger.debug("Two-step verification page not detected.")
            two_fa_present = False
        except WebDriverException as e:
            # Handle potential errors during the check itself
            logger.error(f"WebDriver error checking for 2FA page: {e}")
            status = login_status(session_manager) # Check status if unsure
            if status is True: return "LOGIN_SUCCEEDED" # Logged in despite error
            if status is False: return "LOGIN_FAILED_UNKNOWN" # Not logged in
            return "LOGIN_FAILED_WEBDRIVER" # Error during check

        # Step 8: Handle 2FA if present
        if two_fa_present:
            if handle_twoFA(session_manager):
                logger.info("Two-step verification handled successfully.")
                # Final verification after 2FA
                if login_status(session_manager):
                    return "LOGIN_SUCCEEDED"
                else:
                    logger.error("Login status check failed AFTER successful 2FA handling report.")
                    return "LOGIN_FAILED_POST_2FA_VERIFY"
            else:
                logger.error("Two-step verification handling failed.")
                return "LOGIN_FAILED_2FA_HANDLING"

        # Step 9: If no 2FA, check login status directly
        else:
            logger.debug("Checking login status directly (no 2FA detected)...")
            login_check_result = login_status(session_manager)
            if login_check_result is True:
                logger.info("Direct login check successful.")
                return "LOGIN_SUCCEEDED"
            elif login_check_result is False:
                # Re-check for error messages if login failed without 2FA
                logger.error("Direct login check failed. Checking for error messages again...")
                try:
                    WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.CSS_SELECTOR, FAILED_LOGIN_SELECTOR)))
                    logger.error("Login failed: Specific 'Invalid Credentials' alert found (post-check).")
                    return "LOGIN_FAILED_BAD_CREDS"
                except TimeoutException:
                    generic_alert_selector = "div.alert[role='alert']"
                    try:
                        alert_element = WebDriverWait(driver, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, generic_alert_selector)))
                        alert_text = alert_element.text if alert_element else 'Unknown error'
                        logger.error(f"Login failed: Generic alert found (post-check): '{alert_text}'.")
                        return "LOGIN_FAILED_ERROR_DISPLAYED"
                    except TimeoutException:
                        # If no error message, check if still on login page
                        try:
                            if driver.current_url.startswith(signin_url):
                                logger.error("Login failed: Still on login page (post-check), no error message found.")
                                return "LOGIN_FAILED_STUCK_ON_LOGIN"
                            else:
                                logger.error("Login failed: Status False, no 2FA, no error msg, not on login page.")
                                return "LOGIN_FAILED_UNKNOWN"
                        except WebDriverException:
                            logger.error("Login failed: Status False, WebDriverException getting URL.")
                            return "LOGIN_FAILED_WEBDRIVER"
                    except Exception as e:
                        logger.error(f"Login failed: Error checking for generic alert (post-check): {e}")
                        return "LOGIN_FAILED_UNKNOWN"
                except Exception as e:
                    logger.error(f"Login failed: Error checking for specific alert (post-check): {e}")
                    return "LOGIN_FAILED_UNKNOWN"
            else: # login_status returned None
                logger.error("Login failed: Critical error during final login status check.")
                return "LOGIN_FAILED_STATUS_CHECK_ERROR"

    # Step 10: Handle outer exceptions
    except TimeoutException as e:
        logger.error(f"Timeout during login process: {e}", exc_info=False)
        return "LOGIN_FAILED_TIMEOUT"
    except WebDriverException as e:
        logger.error(f"WebDriverException during login: {e}", exc_info=False)
        if not is_browser_open(driver): logger.error("Session became invalid during login.")
        return "LOGIN_FAILED_WEBDRIVER"
    except Exception as e:
        logger.error(f"An unexpected error occurred during login: {e}", exc_info=True)
        return "LOGIN_FAILED_UNEXPECTED"
# End of log_in

# Login Status Check Function 1
@retry(MAX_RETRIES=2) # Allow retry for status check
def login_status(session_manager: SessionManager) -> Optional[bool]:
    """
    Checks if the user is currently logged into Ancestry.
    Prioritizes checking via API, falls back to UI element checks if API fails
    or indicates logged out.

    Args:
        session_manager: The active SessionManager instance.

    Returns:
        True: If the user is likely logged in.
        False: If the user is likely logged out.
        None: If a critical error occurred preventing status determination.
    """
    logger.debug("Checking login status (API prioritized)...")

    # --- Basic Prerequisites ---
    # Step 1: Validate SessionManager input
    if not isinstance(session_manager, SessionManager):
        logger.error(f"Invalid argument: Expected SessionManager, got {type(session_manager)}.")
        return None # Cannot proceed without valid SessionManager

    # Step 2: Check basic WebDriver session validity
    if not session_manager.is_sess_valid():
        logger.debug("Login status check: Session invalid or browser closed.")
        return False # Definitely not logged in if session is dead
    driver = session_manager.driver
    if driver is None:
        logger.error("Login status check: Driver is None within SessionManager.")
        return None # Cannot proceed without driver

    # --- API-Based Check (Priority) ---
    # Step 3: Attempt verification using the header/dna API endpoint
    logger.debug("Attempting API login verification (_verify_api_login_status)...")
    api_check_result = session_manager._verify_api_login_status()

    # Step 4: Interpret API result
    if api_check_result is True:
        # API confirmed login - most reliable check
        logger.debug("Login status confirmed TRUE via API check.")
        return True
    elif api_check_result is False:
        # API explicitly indicated NOT logged in (e.g., 401/403)
        logger.debug("API check indicates user NOT logged in. Proceeding to UI check as fallback/confirmation.")
        # Fall through to UI check
    else: # api_check_result is None
        # A critical error occurred during the API check itself (logged within _verify_api_login_status)
        logger.warning("API login check returned None (critical error). Falling back to UI check.")
        # Fall through to UI check

    # --- UI-Based Check (Fallback or Confirmation) ---
    logger.debug("Performing fallback UI login check...")
    try:
        # Step 5: Check for the absence of the main "Sign In" button
        # Absence is a good sign, but not definitive proof.
        login_button_selector = LOG_IN_BUTTON_SELECTOR # e.g., '#secMenuItem-SignIn > span'
        logger.debug(f"UI Check Step 1: Checking ABSENCE of login button: '{login_button_selector}'")
        login_button_present = False
        try:
            # Use a short wait - if it's easily visible, we're logged out.
            WebDriverWait(driver, 2).until(EC.visibility_of_element_located((By.CSS_SELECTOR, login_button_selector)))
            login_button_present = True
            logger.debug("Login button FOUND during UI check.")
        except TimeoutException:
            logger.debug("Login button NOT found during UI check (good indication).")
            login_button_present = False
        except Exception as e:
            # Log error but proceed cautiously, assuming button *might* not be present
            logger.warning(f"Error checking for login button presence: {e}")

        # Step 6: Interpret login button presence
        if login_button_present:
            # If the main login button is clearly visible, user is definitely logged out.
            logger.debug("Login status confirmed FALSE via UI check (login button found).")
            return False
        else:
            # Login button was NOT found. Now check for a logged-in indicator as confirmation.
            # Step 7: Check for the PRESENCE of a logged-in indicator element
            logged_in_selector = CONFIRMED_LOGGED_IN_SELECTOR # e.g., '#navAccount'
            logger.debug(f"UI Check Step 2: Checking PRESENCE of logged-in element: '{logged_in_selector}'")
            # Use helper function `is_elem_there` for this check
            ui_element_present = is_elem_there(driver, By.CSS_SELECTOR, logged_in_selector, wait=3)

            # Step 8: Interpret logged-in indicator presence
            if ui_element_present:
                logger.debug("Login status confirmed TRUE via UI check (login button absent AND logged-in element found).")
                return True
            else:
                # Step 9: Handle Ambiguous UI State
                # API check failed/false, login button absent, logged-in element absent.
                # This could be an intermediate page, an error page, or a UI change.
                current_url_context = "Unknown"
                try: current_url_context = driver.current_url
                except Exception: pass
                logger.warning(f"Login status ambiguous: API failed/false, UI elements inconclusive at URL: {current_url_context}")
                # Defaulting to False in ambiguous state is safer (triggers login if needed)
                return False

    # Step 10: Handle Exceptions during UI checks
    except WebDriverException as e:
        logger.error(f"WebDriverException during UI login_status check: {e}")
        if not is_browser_open(driver): # Use basic check here
            logger.error("Session became invalid during UI login_status check.")
            session_manager.close_sess() # Close if session died
        return None # Return None on critical WebDriver error
    except Exception as e:
        logger.critical(f"CRITICAL Unexpected error during UI login_status check: {e}", exc_info=True)
        return None # Return None on critical unexpected error
# End of login_status


# ------------------------------------------------------------------------------------
# Browser Interaction Functions
# ------------------------------------------------------------------------------------


def is_browser_open(driver: Optional[WebDriver]) -> bool:
    """
    Checks if the browser window associated with the WebDriver instance appears
    to be open and responsive.

    Args:
        driver: The Selenium WebDriver instance (can be None).

    Returns:
        True if the browser seems open and session valid, False otherwise.
    """
    # Step 1: Handle None driver input
    if driver is None:
        return False

    # Step 2: Attempt a lightweight WebDriver command
    try:
        # Accessing window_handles requires communication but is relatively lightweight
        _ = driver.window_handles
        # Step 3: Return True if command succeeds
        return True
    # Step 4: Handle exceptions indicating closed/invalid session
    except (InvalidSessionIdException, NoSuchWindowException, WebDriverException) as e:
        err_str = str(e).lower()
        # Check for common error substrings associated with closed browser/session
        if any(sub in err_str for sub in ["invalid session id", "target closed", "disconnected", "no such window", "unable to connect"]):
            logger.debug(f"Browser appears closed or session invalid: {type(e).__name__}")
            return False
        else:
            # Log other WebDriver errors but return False cautiously
            logger.warning(f"WebDriverException checking browser status (assuming closed): {e}")
            return False
    # Step 5: Handle other unexpected errors
    except Exception as e:
        logger.error(f"Unexpected error checking browser status: {e}", exc_info=True)
        return False
# End of is_browser_open


# Note: This function potentially duplicates state saving in SessionManager if used elsewhere.
# Consider consolidating state saving logic if needed. Currently kept for potential standalone use.
def restore_sess(driver: WebDriver) -> bool:
    """
    Restores session state (cookies, local storage, session storage) from cached
    files specific to the current domain. Performs a hard refresh after restoration.

    Args:
        driver: The active Selenium WebDriver instance.

    Returns:
        True if any state was successfully restored and refreshed, False otherwise.
    """
    # Step 1: Validate driver
    if not driver:
        logger.error("Cannot restore session: WebDriver is None.")
        return False

    # Step 2: Determine domain and cache directory
    cache_dir = config_instance.CACHE_DIR
    domain = ""
    try:
        current_url = driver.current_url
        parsed_url = urlparse(current_url)
        # Extract domain, remove www., ignore port
        domain = parsed_url.netloc.replace("www.", "").split(":")[0]
        if not domain: raise ValueError("Could not extract valid domain from current URL")
    except Exception as e:
        logger.error(f"Could not parse current URL for session restore: {e}. Using fallback.")
        try:
            # Fallback to using BASE_URL domain
            parsed_base = urlparse(config_instance.BASE_URL)
            domain = parsed_base.netloc.replace("www.", "").split(":")[0]
            if not domain: raise ValueError("Could not extract domain from BASE_URL")
            logger.warning(f"Falling back to base URL domain for restore: {domain}")
        except Exception as base_e:
            logger.critical(f"Could not determine domain from base URL either: {base_e}. Cannot restore.")
            return False

    logger.debug(f"Attempting to restore session state from cache for domain: {domain}...")
    cache_dir.mkdir(parents=True, exist_ok=True) # Ensure cache dir exists

    restored_something = False # Flag to track if anything was restored

    # Step 3: Helper function for safe JSON reading
    def safe_json_read(filename: str) -> Optional[Any]:
        """Reads JSON safely from a file in the cache directory."""
        filepath = cache_dir / filename
        if filepath.exists():
            try:
                with filepath.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
                logger.warning(f"Skipping restore from '{filepath.name}' (read/decode error): {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error reading '{filepath.name}': {e}", exc_info=True)
                return None
        return None # File doesn't exist
    # End of safe_json_read

    # Step 4: Restore Cookies
    cookies_file = f"session_cookies_{domain}.json"
    cookies = safe_json_read(cookies_file)
    if cookies and isinstance(cookies, list):
        try:
            logger.debug(f"Restoring {len(cookies)} cookies from cache...")
            driver.delete_all_cookies() # Clear existing browser cookies first
            count = 0
            for cookie in cookies:
                # Basic validation of cookie structure before adding
                if isinstance(cookie, dict) and "name" in cookie and "value" in cookie and "domain" in cookie:
                    try:
                        # Remove unsupported 'expiry' if present (Selenium uses 'expires')
                        if 'expiry' in cookie: del cookie['expiry']
                        # Handle potential 'expires' format if needed (requests uses integer timestamp)
                        if 'expires' in cookie and not isinstance(cookie['expires'], (int, float, type(None))):
                             logger.warning(f"Skipping cookie '{cookie.get('name', '??')}' due to invalid 'expires' format: {cookie['expires']}")
                             continue

                        driver.add_cookie(cookie)
                        count += 1
                    except WebDriverException as add_e:
                        logger.warning(f"Skipping cookie '{cookie.get('name', '??')}' during add: {add_e}")
                else:
                     logger.warning(f"Skipping cookie with invalid format: {cookie}")
            logger.debug(f"Added {count} valid cookies from cache.")
            restored_something = True
        except WebDriverException as e:
            logger.error(f"WebDriver error during cookie restore: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during cookie restore: {e}", exc_info=True)
    elif cookies is not None: # File existed but content was invalid
        logger.warning(f"Cookie cache file '{cookies_file}' has invalid format (expected list).")

    # Step 5: Restore Local Storage
    local_storage_file = f"session_local_storage_{domain}.json"
    local_storage = safe_json_read(local_storage_file)
    if local_storage and isinstance(local_storage, dict):
        try:
            logger.debug(f"Restoring {len(local_storage)} items into localStorage...")
            # JavaScript to clear and repopulate localStorage
            script = """
                 var data = arguments[0]; localStorage.clear();
                 for (var key in data) { if (data.hasOwnProperty(key)) {
                     try { localStorage.setItem(key, data[key]); }
                     catch (e) { console.warn('Failed localStorage setItem:', key, e); }
                 }} return localStorage.length;"""
            restored_count = driver.execute_script(script, local_storage)
            logger.debug(f"localStorage restored ({restored_count} items set).")
            restored_something = True
        except WebDriverException as e:
            logger.error(f"Error restoring localStorage via JS: {e}")
        except Exception as e:
            logger.error(f"Unexpected error restoring localStorage: {e}", exc_info=True)
    elif local_storage is not None:
        logger.warning(f"localStorage cache file '{local_storage_file}' has invalid format (expected dict).")

    # Step 6: Restore Session Storage
    session_storage_file = f"session_session_storage_{domain}.json"
    session_storage = safe_json_read(session_storage_file)
    if session_storage and isinstance(session_storage, dict):
        try:
            logger.debug(f"Restoring {len(session_storage)} items into sessionStorage...")
            # JavaScript to clear and repopulate sessionStorage
            script = """
                 var data = arguments[0]; sessionStorage.clear();
                 for (var key in data) { if (data.hasOwnProperty(key)) {
                     try { sessionStorage.setItem(key, data[key]); }
                     catch (e) { console.warn('Failed sessionStorage setItem:', key, e); }
                 }} return sessionStorage.length;"""
            restored_count = driver.execute_script(script, session_storage)
            logger.debug(f"sessionStorage restored ({restored_count} items set).")
            restored_something = True
        except WebDriverException as e:
            logger.error(f"Error restoring sessionStorage via JS: {e}")
        except Exception as e:
            logger.error(f"Unexpected error restoring sessionStorage: {e}", exc_info=True)
    elif session_storage is not None:
        logger.warning(f"sessionStorage cache file '{session_storage_file}' has invalid format (expected dict).")

    # Step 7: Perform hard refresh if any state was restored
    if restored_something:
        logger.info("Session state restored from cache. Performing hard refresh...")
        try:
            driver.refresh()
            # Wait for page to reload completely after refresh
            WebDriverWait(driver, selenium_config.PAGE_TIMEOUT).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            logger.debug("Hard refresh completed after restore.")
        except TimeoutException:
            logger.warning("Timeout waiting for page load after hard refresh.")
        except WebDriverException as e:
            logger.warning(f"Error during hard refresh: {e}")
        return True # Indicate success even if refresh had minor issues
    else:
        logger.debug("No session state found in cache to restore.")
        return False # Nothing was restored
# End of restore_sess


# Note: This function potentially duplicates state saving in SessionManager if used elsewhere.
# Consider consolidating state saving logic if needed. Currently kept for potential standalone use.
def save_state(driver: WebDriver):
    """
    Saves the current session state (cookies, local storage, session storage)
    to domain-specific JSON files in the cache directory.

    Args:
        driver: The active Selenium WebDriver instance.
    """
    # Step 1: Check if browser session is valid
    if not is_browser_open(driver):
        logger.warning("Browser session invalid/closed. Skipping session state save.")
        return

    # Step 2: Determine domain and cache directory
    cache_dir = config_instance.CACHE_DIR
    domain = ""
    try:
        current_url = driver.current_url
        parsed_url = urlparse(current_url)
        domain = parsed_url.netloc.replace("www.", "").split(":")[0]
        if not domain: raise ValueError("Could not extract domain from current URL")
    except Exception as e:
        logger.error(f"Could not parse current URL for saving state: {e}. Using fallback.")
        try:
            parsed_base = urlparse(config_instance.BASE_URL)
            domain = parsed_base.netloc.replace("www.", "").split(":")[0]
            if not domain: raise ValueError("Could not extract domain from BASE_URL")
            logger.warning(f"Falling back to base URL domain for saving state: {domain}")
        except Exception as base_e:
            logger.critical(f"Could not determine domain from base URL either: {base_e}. Cannot save state.")
            return

    logger.debug(f"Saving session state for domain: {domain}")
    cache_dir.mkdir(parents=True, exist_ok=True) # Ensure cache dir exists

    # Step 3: Helper function for safe JSON writing
    def safe_json_write(data: Any, filename: str) -> bool:
        """Writes data safely to a JSON file in the cache directory."""
        filepath = cache_dir / filename
        try:
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2) # Use indent for readability
            return True
        except (TypeError, IOError, UnicodeEncodeError) as e:
            logger.error(f"Error writing JSON to '{filepath.name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error writing JSON to '{filepath.name}': {e}", exc_info=True)
            return False
    # End of safe_json_write

    # Step 4: Save Cookies
    cookies_file = f"session_cookies_{domain}.json"
    try:
        cookies = driver.get_cookies()
        if cookies is not None: # Check if driver returned cookies
            if safe_json_write(cookies, cookies_file):
                logger.debug(f"Cookies saved ({len(cookies)} items).")
        else:
            logger.warning("Could not retrieve cookies from driver (returned None).")
    except WebDriverException as e:
        logger.error(f"Error getting cookies for saving: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving cookies: {e}", exc_info=True)

    # Step 5: Save Local Storage
    local_storage_file = f"session_local_storage_{domain}.json"
    try:
        # Execute JS to get localStorage content as a dictionary
        local_storage = driver.execute_script(
            "return window.localStorage ? {...window.localStorage} : {};" # Return empty dict if null
        )
        if local_storage: # Check if not empty/null
            if safe_json_write(local_storage, local_storage_file):
                logger.debug(f"localStorage saved ({len(local_storage)} items).")
        else:
            logger.debug("localStorage not available or empty, not saved.")
    except WebDriverException as e:
        logger.error(f"Error getting localStorage via JS: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving localStorage: {e}", exc_info=True)

    # Step 6: Save Session Storage
    session_storage_file = f"session_session_storage_{domain}.json"
    try:
        # Execute JS to get sessionStorage content as a dictionary
        session_storage = driver.execute_script(
            "return window.sessionStorage ? {...window.sessionStorage} : {};" # Return empty dict if null
        )
        if session_storage: # Check if not empty/null
            if safe_json_write(session_storage, session_storage_file):
                logger.debug(f"sessionStorage saved ({len(session_storage)} items).")
        else:
            logger.debug("sessionStorage not available or empty, not saved.")
    except WebDriverException as e:
        logger.error(f"Error getting sessionStorage via JS: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving sessionStorage: {e}", exc_info=True)

    # logger.debug(f"Session state save attempt finished for domain: {domain}.") # Optional summary log
# End of save_state


def close_tabs(driver: WebDriver):
    """
    Closes all browser tabs except the first one. Ensures focus remains on the
    first (original) tab after closing others.

    Args:
        driver: The Selenium WebDriver instance.
    """
    # Step 1: Check if driver exists
    if not driver:
        logger.warning("close_tabs: WebDriver instance is None.")
        return

    logger.debug("Closing extra browser tabs...")
    try:
        # Step 2: Get all window handles
        handles = driver.window_handles
        if len(handles) <= 1:
            logger.debug("No extra tabs to close.")
            return

        # Step 3: Store original handle and identify the first handle
        original_handle = driver.current_window_handle
        first_handle = handles[0]
        logger.debug(f"Original handle: {original_handle}, First handle: {first_handle}")

        # Step 4: Iterate and close all handles except the first one
        closed_count = 0
        for handle in handles[1:]:
            try:
                logger.debug(f"Switching to tab {handle} to close...")
                driver.switch_to.window(handle)
                driver.close()
                logger.debug(f"Closed tab handle: {handle}")
                closed_count += 1
            except NoSuchWindowException:
                logger.warning(f"Tab {handle} already closed or could not be switched to.")
            except WebDriverException as e:
                logger.error(f"Error closing tab {handle}: {e}")

        logger.debug(f"Closed {closed_count} extra tabs.")

        # Step 5: Switch focus back to the primary tab
        # Check if original handle still exists (it should if it was the first tab)
        remaining_handles = driver.window_handles
        if original_handle in remaining_handles:
            if driver.current_window_handle != original_handle:
                 logger.debug(f"Switching back to original tab: {original_handle}")
                 driver.switch_to.window(original_handle)
            else:
                 logger.debug("Already focused on original tab.")
        # Fallback to first handle if original was somehow closed (but shouldn't be)
        elif first_handle in remaining_handles:
             logger.warning(f"Original handle {original_handle} missing. Switching to first handle: {first_handle}")
             driver.switch_to.window(first_handle)
        # Fallback if both original and first are gone (indicates problem)
        elif remaining_handles:
             logger.error(f"Original and first tabs gone. Switching to remaining: {remaining_handles[0]}")
             driver.switch_to.window(remaining_handles[0])
        else:
             logger.error("All browser tabs were closed unexpectedly.")

    # Step 6: Handle exceptions during the process
    except NoSuchWindowException:
        logger.warning("Attempted to close/switch to a tab that no longer exists during cleanup.")
    except WebDriverException as e:
        logger.error(f"WebDriverException during close_tabs: {e}")
        # Check session validity after error
        if not is_browser_open(driver):
            logger.error("Session invalid during close_tabs operation.")
    except Exception as e:
        logger.error(f"Unexpected error in close_tabs: {e}", exc_info=True)
# End of close_tabs


# ------------------------------------------------------------------------------------
# Navigation Functions
# ------------------------------------------------------------------------------------

def nav_to_page(
    driver: WebDriver,
    url: str,
    selector: str = "body", # Default selector to wait for body tag
    session_manager: Optional[SessionManager] = None, # Optional for restart logic
) -> bool:
    """
    Navigates the WebDriver to a specified URL. Includes handling for common
    issues like unexpected redirects to login/MFA pages, page unavailability
    messages, and session invalidity (attempting restart if SessionManager provided).

    Args:
        driver: The Selenium WebDriver instance.
        url: The target URL to navigate to.
        selector: A CSS selector for an element expected on the target page,
                  used to verify successful navigation. Defaults to 'body'.
        session_manager: Optional SessionManager instance to enable session
                         restart attempts if the session becomes invalid.

    Returns:
        True if navigation to the target URL and verification succeeded, False otherwise.
    """
    # Step 1: Validate inputs
    if not driver:
        logger.error("Navigation failed: WebDriver instance is None.")
        return False
    if not url:
        logger.error("Navigation failed: Target URL is required.")
        return False

    # Step 2: Get configuration and prepare URLs
    max_attempts = config_instance.MAX_RETRIES
    page_timeout = selenium_config.PAGE_TIMEOUT
    element_timeout = selenium_config.ELEMENT_TIMEOUT
    target_url_parsed = urlparse(url)
    # Normalize target URL base (scheme + netloc + path, no query/fragment)
    target_url_base = f"{target_url_parsed.scheme}://{target_url_parsed.netloc}{target_url_parsed.path}".rstrip("/")
    # Normalize known redirect URLs
    signin_page_url_base = urljoin(config_instance.BASE_URL, "account/signin").rstrip("/")
    mfa_page_url_base = urljoin(config_instance.BASE_URL, "account/signin/mfa/").rstrip("/")
    # Selectors for unavailability messages
    unavailability_selectors = {
        TEMP_UNAVAILABLE_SELECTOR: ("refresh", 5), # Action: refresh after 5s
        PAGE_NO_LONGER_AVAILABLE_SELECTOR: ("skip", 0), # Action: skip (fail nav)
    }

    # Step 3: Navigation attempt loop
    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Navigation Attempt {attempt}/{max_attempts} to: {url}")
        landed_url = "" # Store URL after potential redirects

        try:
            # Step 3a: Check WebDriver session validity before attempting navigation
            if not is_browser_open(driver):
                logger.error(f"Navigation failed (Attempt {attempt}): Browser session invalid before nav.")
                # Attempt restart only if session manager is provided
                if session_manager:
                    logger.warning("Attempting session restart...")
                    if session_manager.restart_sess():
                        logger.info("Session restarted. Retrying navigation...")
                        # Update driver instance after restart
                        driver = session_manager.driver
                        if not driver: # Check if restart actually provided a driver
                             logger.error("Session restart reported success but driver is still None.")
                             return False
                        continue # Retry the navigation in the next loop iteration
                    else:
                        logger.error("Session restart failed. Cannot navigate.")
                        return False
                else: # No session manager, cannot restart
                    return False

            # Step 3b: Perform the actual navigation
            logger.debug(f"Executing driver.get('{url}')...")
            driver.get(url)
            # Wait for document ready state to be complete or interactive
            WebDriverWait(driver, page_timeout).until(
                lambda d: d.execute_script("return document.readyState") in ["complete", "interactive"]
            )
            # Short pause to allow potential JS redirects to trigger
            time.sleep(random.uniform(0.5, 1.5))

            # Step 3c: Get the URL the browser actually landed on
            try:
                landed_url = driver.current_url
                landed_url_parsed = urlparse(landed_url)
                landed_url_base = f"{landed_url_parsed.scheme}://{landed_url_parsed.netloc}{landed_url_parsed.path}".rstrip("/")
                logger.debug(f"Landed on URL base: {landed_url_base}")
            except WebDriverException as e:
                logger.error(f"Failed to get URL after get() (Attempt {attempt}): {e}. Retrying.")
                continue # Retry navigation attempt

            # --- Step 4: Post-Navigation Verification ---

            # Step 4a: Check for unexpected MFA page redirect
            is_on_mfa_page = False
            try:
                # Quick check for MFA header visibility
                WebDriverWait(driver, 1).until(EC.visibility_of_element_located((By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)))
                is_on_mfa_page = True
            except TimeoutException: pass # Not on MFA page
            except WebDriverException as e: logger.warning(f"WebDriverException checking for MFA header: {e}")

            if is_on_mfa_page:
                logger.warning("Landed on MFA page unexpectedly during navigation.")
                # Fail navigation - MFA requires manual intervention not handled here.
                return False

            # Step 4b: Check for unexpected Login page redirect
            is_on_login_page = False
            # Only check if the landed URL *looks* like the signin page OR if the target wasn't the signin page anyway
            if landed_url_base == signin_page_url_base or target_url_base != signin_page_url_base:
                try:
                    # Quick check for username input visibility
                    WebDriverWait(driver, 1).until(EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR)))
                    is_on_login_page = True
                except TimeoutException: pass # Not on login page
                except WebDriverException as e: logger.warning(f"WebDriverException checking for Login username input: {e}")

            # If landed on login page AND target wasn't the login page -> Attempt re-login
            if is_on_login_page and target_url_base != signin_page_url_base:
                logger.warning("Landed on Login page unexpectedly.")
                if session_manager:
                    logger.info("Attempting re-login...")
                    # Check status first, maybe session is okay despite redirect
                    login_stat = login_status(session_manager)
                    if login_stat is True:
                        logger.info("Login status OK after landing on login page redirect. Retrying navigation.")
                        continue # Retry original navigation
                    else:
                        # Attempt actual login process
                        login_result = log_in(session_manager)
                        if login_result == "LOGIN_SUCCEEDED":
                            logger.info("Re-login successful. Retrying original navigation...")
                            continue # Retry original navigation
                        else:
                            logger.error(f"Re-login attempt failed ({login_result}). Cannot complete navigation.")
                            return False
                else:
                    logger.error("Landed on login page, no SessionManager provided for re-login attempt.")
                    return False

            # Step 4c: Check if landed URL base matches target URL base
            if landed_url_base != target_url_base:
                # Allow specific redirect: from signin page to base URL after successful login
                is_signin_to_base_redirect = (
                    target_url_base == signin_page_url_base and
                    landed_url_base == urlparse(config_instance.BASE_URL).path.rstrip("/")
                )
                if is_signin_to_base_redirect:
                    logger.debug("Redirected from signin page to base URL. Verifying login status...")
                    time.sleep(1) # Allow slight delay for potential state change
                    if session_manager and login_status(session_manager) is True:
                        logger.info("Redirect after signin confirmed as logged in. Navigation OK.")
                        return True # Treat this specific redirect as success if logged in

                # Handle other unexpected redirects
                logger.warning(f"Navigation landed on unexpected URL base: '{landed_url_base}' (Expected: '{target_url_base}')")
                # Check for known unavailability messages
                action, wait_time = _check_for_unavailability(driver, unavailability_selectors)
                if action == "skip": return False # Page no longer available
                if action == "refresh": time.sleep(wait_time); continue # Temporary issue, retry
                # Otherwise, just retry the navigation attempt
                logger.warning("Wrong URL, no specific unavailability message. Retrying.")
                continue

            # --- Step 5: Wait for Target Selector on Correct Page ---
            wait_selector = selector if selector else "body" # Ensure we wait for something
            logger.debug(f"On correct URL base. Waiting up to {element_timeout}s for selector: '{wait_selector}'")
            try:
                WebDriverWait(driver, element_timeout).until(EC.visibility_of_element_located((By.CSS_SELECTOR, wait_selector)))
                logger.debug(f"Navigation successful and element '{wait_selector}' found on:\n{url}")
                return True # SUCCESS! Target URL reached and selector found.

            except TimeoutException:
                # Re-check URL in case of delayed redirect after base URL match
                current_url_on_timeout = "Unknown"
                try: current_url_on_timeout = driver.current_url
                except Exception: pass
                logger.warning(f"Timeout waiting for selector '{wait_selector}' at {current_url_on_timeout} (URL base was correct initially).")
                # Check again for unavailability messages after timeout
                action, wait_time = _check_for_unavailability(driver, unavailability_selectors)
                if action == "skip": return False
                if action == "refresh": time.sleep(wait_time); continue
                # If no specific message, retry navigation
                logger.warning("Timeout on selector, no unavailability message. Retrying navigation.")
                continue

        # --- Step 6: Handle Exceptions During the Attempt ---
        except UnexpectedAlertPresentException as alert_e:
            alert_text = "N/A"
            try: alert_text = alert_e.alert_text
            except: pass
            logger.warning(f"Unexpected alert detected (Attempt {attempt}): {alert_text}")
            try:
                driver.switch_to.alert.accept()
                logger.info("Accepted unexpected alert.")
            except Exception as accept_e:
                logger.error(f"Failed to accept unexpected alert: {accept_e}")
                return False # Fail if alert cannot be handled
            continue # Retry navigation after handling alert

        except WebDriverException as wd_e:
            logger.error(f"WebDriverException during navigation (Attempt {attempt}): {wd_e}", exc_info=False)
            # Check session status and attempt restart if applicable
            if session_manager and not is_browser_open(driver):
                logger.error("WebDriver session invalid after exception. Attempting restart...")
                if session_manager.restart_sess():
                    logger.info("Session restarted. Retrying navigation...")
                    driver = session_manager.driver # Update driver reference
                    if not driver: return False # Ensure driver updated
                    continue # Retry navigation attempt
                else:
                    logger.error("Session restart failed. Cannot complete navigation.")
                    return False
            else:
                # If session seems okay or no restart possible, wait and retry nav
                logger.warning("WebDriverException occurred, session seems valid or no restart possible. Waiting before retry.")
                time.sleep(random.uniform(2, 4))
                continue

        except Exception as e:
            # Catch any other unexpected error during the attempt
            logger.error(f"Unexpected error during navigation (Attempt {attempt}): {e}", exc_info=True)
            time.sleep(random.uniform(2, 4)) # Wait before retrying
            continue

    # --- End of Retry Loop ---
    # Step 7: Log permanent failure after all attempts
    logger.critical(f"Navigation to '{url}' failed permanently after {max_attempts} attempts.")
    try: logger.error(f"Final URL after failure: {driver.current_url}")
    except Exception: logger.error("Could not retrieve final URL after failure.")
    return False
# End of nav_to_page


# Note: Removed _pre_navigation_checks and _check_post_nav_redirects as their
# logic was integrated more directly into the revised nav_to_page function.

def _check_for_unavailability(
    driver: WebDriver, selectors: Dict[str, Tuple[str, int]]
) -> Tuple[Optional[str], int]:
    """
    Checks if known 'page unavailable' messages are present on the current page.

    Args:
        driver: The Selenium WebDriver instance.
        selectors: A dictionary mapping CSS selectors of unavailability messages
                   to a tuple containing the action ('refresh' or 'skip') and
                   a wait time (int) if action is 'refresh'.

    Returns:
        A tuple containing the action string ('refresh' or 'skip') and wait time
        if a message is found, otherwise (None, 0).
    """
    # Step 1: Iterate through the defined unavailability selectors
    for msg_selector, (action, wait_time) in selectors.items():
        # Step 2: Check if the element for the current selector exists (use short wait)
        if is_elem_there(driver, By.CSS_SELECTOR, msg_selector, wait=0.5):
            # Step 3: If found, log the message and return the corresponding action/wait time
            logger.warning(f"Unavailability message found matching selector: '{msg_selector}'. Action: {action}, Wait: {wait_time}s")
            return action, wait_time
    # Step 4: Return None if no unavailability messages were found
    return None, 0
# End of _check_for_unavailability


# ------------------------------------------------------------------------------------
# Main function for standalone testing
# ------------------------------------------------------------------------------------


def main():
    """
    Standalone test function for utils.py.
    Initializes logging, creates a SessionManager, tests session start (Phases 1 & 2),
    identifier retrieval, navigation, API requests, and tab management.
    Ensures proper session cleanup.
    """
    # --- Setup Logging ---
    from logging_config import setup_logging # Local import for standalone test
    from config import config_instance # Local import for standalone test
    db_file_path = config_instance.DATABASE_FILE
    log_filename_only = db_file_path.with_suffix(".log").name
    # Ensure logger used within this main function is the configured one
    global logger # Modify the global logger variable
    logger = setup_logging(log_file=log_filename_only, log_level="DEBUG") # Set DEBUG for testing
    logger.info(f"--- Starting utils.py standalone test run ---")

    # --- Initialization ---
    session_manager: Optional[SessionManager] = None # Initialize
    test_success = True # Assume success initially

    try:
        # Step 1: Create SessionManager instance
        session_manager = SessionManager()

        # Step 2: Test Session Start (Phase 1: Driver Start)
        logger.info("--- Testing SessionManager.start_sess() [Phase 1] ---")
        start_ok = session_manager.start_sess(action_name="Utils Test - Phase 1")
        if not start_ok or not session_manager.driver_live:
            logger.error("SessionManager.start_sess() (Phase 1) FAILED. Aborting.")
            test_success = False
            # Use return for cleaner exit within try block during tests
            return
        else:
            logger.info("SessionManager.start_sess() (Phase 1) PASSED.")
            # Get driver instance for subsequent tests (check it's not None)
            driver_instance = session_manager.driver
            if not driver_instance:
                 logger.error("Driver instance is None after successful Phase 1 report.")
                 test_success = False
                 return

        # Step 3: Test Session Readiness (Phase 2: Login & Identifiers)
        logger.info("--- Testing SessionManager.ensure_session_ready() [Phase 2] ---")
        ready_ok = session_manager.ensure_session_ready(action_name="Utils Test - Phase 2")
        if not ready_ok or not session_manager.session_ready:
            logger.error("SessionManager.ensure_session_ready() (Phase 2) FAILED. Aborting.")
            test_success = False
            return
        else:
            logger.info("SessionManager.ensure_session_ready() (Phase 2) PASSED.")

        # Step 4: Verify Identifiers were retrieved during Phase 2
        logger.info("--- Verifying Identifiers ---")
        errors = []
        if not session_manager.my_profile_id: errors.append("my_profile_id")
        if not session_manager.my_uuid: errors.append("my_uuid")
        if config_instance.TREE_NAME and not session_manager.my_tree_id:
             errors.append(f"my_tree_id (required for TREE_NAME '{config_instance.TREE_NAME}')")
        if not session_manager.csrf_token: errors.append("csrf_token")
        if errors:
            logger.error(f"FAILED to retrieve required identifiers: {', '.join(errors)}")
            test_success = False
        else:
            logger.info("All required identifiers retrieved successfully.")
            logger.debug(f"  Profile ID: {session_manager.my_profile_id}")
            logger.debug(f"  UUID: {session_manager.my_uuid}")
            logger.debug(f"  Tree ID: {session_manager.my_tree_id or 'N/A'}")
            logger.debug(f"  Tree Owner: {session_manager.tree_owner_name or 'N/A'}")
            logger.debug(f"  CSRF Token: {session_manager.csrf_token[:10]}...")

        # Step 5: Test Navigation (to Base URL)
        logger.info("--- Testing Navigation (nav_to_page to BASE_URL) ---")
        # Ensure driver_instance is valid before using
        if not driver_instance:
             logger.error("Cannot test navigation, driver_instance is None.")
             test_success = False
        else:
            nav_ok = nav_to_page(driver=driver_instance, url=config_instance.BASE_URL, selector="body", session_manager=session_manager)
            if nav_ok:
                logger.info("nav_to_page() to BASE_URL PASSED.")
                try:
                    current_url_after_nav = driver_instance.current_url
                    if current_url_after_nav.startswith(config_instance.BASE_URL.rstrip('/')):
                        logger.info(f"Successfully landed on expected base URL: {current_url_after_nav}")
                    else:
                        logger.warning(f"nav_to_page() to base URL landed on slightly different URL: {current_url_after_nav}")
                except Exception as e:
                    logger.warning(f"Could not verify URL after nav_to_page: {e}")
            else:
                logger.error("nav_to_page() to BASE_URL FAILED.")
                test_success = False

        # Step 6: Test API Request Helper (_api_req via CSRF endpoint)
        logger.info("--- Testing API Request (_api_req via CSRF endpoint) ---")
        csrf_url = urljoin(config_instance.BASE_URL, "discoveryui-matches/parents/api/csrfToken")
        csrf_test_response = _api_req(
            url=csrf_url,
            driver=driver_instance, # Pass potentially None driver
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description="CSRF Token API Test",
            force_text_response=True, # Expect plain text
        )
        if csrf_test_response and isinstance(csrf_test_response, str) and len(csrf_test_response) > 10:
            logger.info("CSRF Token API call via _api_req PASSED.")
            logger.debug(f"CSRF Token API test retrieved: {csrf_test_response[:10]}...")
        else:
            logger.error("CSRF Token API call via _api_req FAILED.")
            logger.debug(f"Response received: {csrf_test_response}")
            test_success = False

        # Step 7: Test Browser Tab Management
        logger.info("--- Testing Tab Management (make_tab, close_tabs) ---")
        # Ensure driver_instance is valid before using
        if not driver_instance:
             logger.error("Cannot test tab management, driver_instance is None.")
             test_success = False
        else:
            logger.info("Creating a new tab...")
            new_tab_handle = session_manager.make_tab()
            if new_tab_handle:
                logger.info(f"make_tab() PASSED. New handle: {new_tab_handle}")
                logger.info("Navigating new tab to example.com...")
                try:
                    driver_instance.switch_to.window(new_tab_handle)
                    # Use nav_to_page for robustness
                    if nav_to_page(driver_instance, "https://example.com", selector="body", session_manager=session_manager):
                        logger.info("Navigation in new tab successful.")
                        logger.info("Closing extra tabs...")
                        close_tabs(driver_instance)
                        # Verify only one tab remains
                        handles_after_close = driver_instance.window_handles
                        if len(handles_after_close) == 1:
                            logger.info("close_tabs() PASSED (one tab remaining).")
                            # Ensure focus is back
                            if driver_instance.current_window_handle != handles_after_close[0]:
                                logger.debug("Switching focus back to remaining tab.")
                                driver_instance.switch_to.window(handles_after_close[0])
                        else:
                            logger.error(f"close_tabs() FAILED (expected 1 tab, found {len(handles_after_close)}).")
                            test_success = False
                    else:
                        logger.error("Navigation in new tab FAILED.")
                        test_success = False
                        # Attempt cleanup even if nav failed
                        close_tabs(driver_instance)
                except Exception as tab_e:
                    logger.error(f"Error during tab management test: {tab_e}", exc_info=True)
                    test_success = False
                    # Attempt cleanup on error
                    try: close_tabs(driver_instance)
                    except Exception: pass
            else:
                logger.error("make_tab() FAILED.")
                test_success = False

    # Step 8: Handle outer exceptions
    except Exception as e:
        logger.critical(f"CRITICAL error during utils.py standalone test execution: {e}", exc_info=True)
        test_success = False

    # Step 9: Final cleanup and summary
    finally:
        if session_manager:
            logger.info("Closing session manager in finally block...")
            session_manager.close_sess() # Ensure session is closed

        print("") # Newline for separation
        if test_success:
            logger.info("--- Utils.py standalone test run PASSED ---")
        else:
            logger.error("--- Utils.py standalone test run FAILED ---")
# End of main

if __name__ == "__main__":
    main()
# End of utils.py
