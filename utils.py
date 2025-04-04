#!/usr/bin/env python3

# # utils.py

"""
utils.py - Standalone login script + utilities using config variables.

This script utilizes configuration variables from config.py (loaded via
config_instance and selenium_config) throughout the login functions and
utility functions, enhancing configurability and maintainability.
"""

import time
import random
import os
import logging
import json
import re
import sqlite3
import sys
import inspect
import requests
import uuid
import base64
from functools import wraps, lru_cache
import contextlib
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Generator
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, urljoin
from selenium.webdriver import ChromeOptions
from dataclasses import dataclass, field
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
    UnexpectedAlertPresentException,
    ElementNotInteractableException,
    InvalidSessionIdException,
    NoSuchWindowException,
    ElementClickInterceptedException,
)
from selenium.webdriver.remote.webdriver import WebDriver
from requests import Response, Request
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException, HTTPError
from my_selectors import *
from config import config_instance, selenium_config
from database import ConnectionPool, Base, MessageType
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from logging_config import setup_logging, logger
from chromedriver import init_webdvr


# ------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------

def force_user_agent(driver, user_agent):
    try:
        logger.debug(
            f"force_user_agent: Attempting to set User-Agent to: {user_agent}"
        )  # Added debug log
        start_time = time.time()  # Start timer
        logger.debug(
            "force_user_agent: Calling execute_cdp_cmd..."
        )  # Debug log before CDP command
        driver.execute_cdp_cmd(
            "Network.setUserAgentOverride", {"userAgent": user_agent}
        )
        logger.debug(
            "force_user_agent: execute_cdp_cmd call returned."
        )  # Debug log after CDP command
        duration = time.time() - start_time  # End timer
        logger.info(
            f"force_user_agent: Successfully set User-Agent to: {user_agent} in {duration:.3f} seconds."
        )  # Added duration log
    except Exception as e:
        logger.warning(
            f"force_user_agent: Error while setting user agent: {e}", exc_info=True
        )
# End of force_user_agent

def parse_cookie(cookie_string):
    """Parses a cookie string into a dictionary."""
    cookies = {}
    parts = cookie_string.split(";")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue  # Skip invalid parts without '='
        key_value_pair = part.split("=", 1)  # Split at the first '=' only
        if len(key_value_pair) != 2:
            continue  # Skip invalid parts
        key, value = key_value_pair
        cookies[key] = value
    return cookies
# end parse_cookie

def extract_text(element, selector):
    """
    Safely extracts text froman element using a CSS selector.

    Args:
        element: The Selenium WebElement.
        selector: The CSS selector to locate the target element.

    Returns:
        str: The text content of the element, or an empty string if not found.
    """
    try:
        text = element.find_element(By.CSS_SELECTOR, selector).text
        return text.strip() if text else ""
    except NoSuchElementException:
        return ""
# End of extract_text

def extract_attribute(element, selector, attribute):
    """
    Extracts the value of a specified attribute from a web element.

    Args:
        element: The Selenium WebElement.
        selector: The CSS selector to locate the target element.
        attribute: The name of the attribute to extract.

    Returns:
        str: The value of the attribute, or an empty string if not found.
    """
    try:
        value = element.find_element(By.CSS_SELECTOR, selector).get_attribute(attribute)
        if attribute == "href" and value.startswith("/"):
            return f"https://www.ancestry.co.uk{value}"
        return value
    except:
        return ""
# End of extract_attribute

def get_tot_page(driver):
    """
     Retrieves the total number of pages from the pagination element.

    Args:
        driver: The Selenium WebDriver instance.

    Returns:
        int: The total number of pages, or 1 if an error occurs.
    """
    element_wait = selenium_config.element_wait(driver)
    try:
        # Combined wait for element presence and attribute availability
        pagination_element = element_wait.until(
            lambda driver: (
                driver.find_element(By.CSS_SELECTOR, PAGINATION_SELECTOR)
                if driver.find_element(
                    By.CSS_SELECTOR, PAGINATION_SELECTOR
                ).get_attribute("total")
                is not None
                else None
            )
        )

        if pagination_element is None:
            return (
                1  # Handle the case where the attribute is still None after the timeout
            )

        total_pages = int(pagination_element.get_attribute("total"))
        return total_pages

    except TimeoutException:
        logger.error(f"Timeout waiting for pagination element or 'total' attribute.")
        return 1
    except Exception as e:
        logger.error(f"Error getting total pages: {e}")
        return 1
# End of get_tot_page

def ordinal_case(text: str) -> str:
    """Corrects ordinal suffixes (st, nd, rd, th) to lowercase after title casing."""
    if not text:
        return text

    def lower_suffix(match):
        return match.group(1) + match.group(2).lower()

    return re.sub(r"(\d+)(st|nd|rd|th)", lower_suffix, text, flags=re.IGNORECASE)
# end ordinal case

def is_elem_there(driver, by, value, wait=selenium_config.ELEMENT_TIMEOUT):
    """
    Checks if an element is present within a specified timeout.
    """
    try:
        WebDriverWait(driver, wait).until(EC.presence_of_element_located((by, value)))
        return True
    except TimeoutException:
        # logger.warning(f"Timed out waiting {wait}s for: {value}")
        return False
# End of is_elem_there

def format_name(name: Optional[str]) -> str:
    """
    Formats a name to title case, preserving all-caps components (like initials)
    and handling None or empty input. Returns "Valued Relative" as a fallback.
    """
    if not name:
        return "Valued Relative"  # Return default if name is None or empty
    try:
        parts = name.split()
        formatted_parts = []
        for part in parts:
            # Preserve all-caps (like initials), title case others
            if part.isupper():
                formatted_parts.append(part)
            else:
                formatted_parts.append(part.title())
        return " ".join(formatted_parts)
    except Exception as e:
        # Log error and return a safe default
        logger.error(f"Error formatting name '{name}': {e}", exc_info=False)
        return "Valued Relative"  # Fallback on error
# END OF format_name

@dataclass
class MatchData:
    """Represents the collected data for a single DNA match."""

    # --- Core Identifiers (Usually from initial match list) ---
    guid: str  # Match's test GUID (e.g., 6EAC8EC1-...)
    display_name: str  # Match's display name (e.g., "Frances Mchardy")
    image_url: Optional[str] = None  # URL of the match's profile image

    # --- DNA Match Details (Usually from initial match list) ---
    shared_cm: Optional[float] = None  # Shared Centimorgans
    shared_segments: Optional[int] = None  # Number of shared segments
    last_login_str: Optional[str] = (
        None  # Raw string like "Logged in today", "Sep 2023"
    )

    # --- Predicted Relationship (From matchProbabilityData API) ---
    confidence: Optional[float] = None  # Probability score from API
    predicted_relationship: Optional[str] = None  # e.g., "1st–2nd cousin"

    # --- Links (Constructed) ---
    compare_link: Optional[str] = None  # Link to comparison page
    message_link: Optional[str] = None  # Link to message the match

    # --- Tree Related Info ---
    # Set after checking matchesInTree API
    in_my_tree: bool = field(default=False)

    # Set after fetching badgedetails API for 'in_my_tree' matches
    match_tree_id: Optional[str] = None  # The ID of *their* linked tree
    cfpid: Optional[str] = None  # The person ID (PID) within *their* tree

    # Constructed using my_tree_id and cfpid for 'in_my_tree' matches
    view_in_tree_link: Optional[str] = None  # Link to view them in *my* tree
    facts_link: Optional[str] = None  # Link to facts page in *my* tree

    # Fetched using getladder API for 'in_my_tree' matches
    relationship_path: Optional[str] = None  # Raw HTML or processed path from ladder
    actual_relationship: Optional[str] = None  # e.g., "Mother", "1st cousin 1x removed"
# End Datamatch class


# ------------------------------
# Decorators (Used by all scripts)
# ------------------------------

def retry(MAX_RETRIES=None, BACKOFF_FACTOR=None, MAX_DELAY=None):
    """Decorator to retry a function with exponential backoff and jitter."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = (
                config_instance.MAX_RETRIES if MAX_RETRIES is None else MAX_RETRIES
            )
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == attempts - 1:
                        logger.error(
                            f"All attempts failed for function '{func.__name__}'."
                        )
                        raise
                    sleep_time = min(
                        config_instance.BACKOFF_FACTOR * (2**i) + random.uniform(0, 1),
                        config_instance.MAX_DELAY,
                    )
                    logger.warning(
                        f"Attempt {i + 1} for function '{func.__name__}' failed: {e}. Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
            return None  # Or perhaps raise an exception here

        return wrapper

    return decorator
# end retry

def retry_api(
    max_retries=3,
    initial_delay=5,
    backoff_factor=2,
    retry_on_exceptions=(requests.exceptions.RequestException,),
    retry_on_status_codes=None,
):
    """
    Decorator for retrying API calls with exponential backoff, using logger.
    Optionally retries on specific HTTP status codes OR exceptions.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (int): Initial delay in seconds before the first retry.
        backoff_factor (int): Factor by which delay increases after each retry.
        retry_on_exceptions (tuple): Tuple of exception classes to retry on. Defaults to (requests.exceptions.RequestException,).
        retry_on_status_codes (list or tuple or set): Optional. A list/tuple/set of HTTP status codes to retry on.
                                                      Defaults to None (only retries on RequestExceptions).
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = max_retries
            delay = initial_delay
            attempt = 0
            while retries > 0:
                attempt += 1
                try:
                    response = func(*args, **kwargs)  # Capture the response
                    if (
                        retry_on_status_codes
                        and isinstance(retry_on_status_codes, (list, tuple, set))
                        and response is not None
                        and hasattr(response, "status_code")
                        and response.status_code in retry_on_status_codes
                    ):
                        # Retry on specific status code
                        status_code = response.status_code
                        retries -= 1
                        if retries == 0:
                            logger.error(
                                f"API Call failed after {max_retries} retries for function '{func.__name__}' (Status {status_code}). No more retries.",
                                exc_info=False,
                            )  # No traceback for status code retry
                            return response  # Return the last response, or raise exception if that's preferred
                        else:
                            logger.warning(
                                f"API Call returned retryable status {status_code} (attempt {attempt}) for function '{func.__name__}', retrying in {delay:.2f} seconds..."
                            )
                            time.sleep(delay)
                            delay *= backoff_factor  # Exponential backoff
                            continue  # Go to next retry attempt
                    return response  # Return successful response if not retrying on status code
                except (
                    retry_on_exceptions
                ) as e:  # MODIFIED - Retry on specified exceptions
                    retries -= 1
                    if retries == 0:
                        logger.error(
                            f"API Call failed after {max_retries} retries due to exception for function '{func.__name__}'.",
                            exc_info=True,
                        )  # Keep traceback for RequestExceptions
                        raise  # Re-raise the last exception
                    else:
                        logger.warning(
                            f"API Call failed (attempt {attempt}) for function '{func.__name__}', retrying in {delay:.2f} seconds... Exception: {e}"
                        )  # Log exception type
                    time.sleep(delay)
                    delay *= backoff_factor  # Exponential backoff
            return None  # Or perhaps raise an exception here if all retries fail (for RequestException case only)

        return wrapper

    return decorator
# End of retry_api

def ensure_browser_open(func):
    """Decorator to ensure the browser is open. Relies on SessionManager."""

    @wraps(func)
    def wrapper(session_manager, *args, **kwargs):
        if not is_browser_open(session_manager.driver):
            raise Exception(
                f"Browser is not open when calling function '{func.__name__}'"
            )
        return func(session_manager, *args, **kwargs)

    return wrapper
# End of ensure_browser_open

def time_wait(wait_description):
    """Decorator to time WebDriverWait calls and log duration."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                logger.debug(
                    f"{wait_description} {duration:.3f} s."
                )  # DEBUG level logging
                return result
            except TimeoutException as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.warning(
                    f"Wait '{wait_description}' timed out after {duration:.3f} seconds.",
                    exc_info=True,
                )  # WARNING for timeouts
                raise  # Re-raise TimeoutException
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(
                    f"Error during wait '{wait_description}' after {duration:.3f} seconds: {e}",
                    exc_info=True,
                )  # ERROR level for unexpected errors
                raise  # Re-raise other exceptions

        return wrapper

    return decorator
# End of time_wait


# ------------------------------
# Rate Limiting (Used by all scripts)
# ------------------------------

class DynamicRateLimiter:
    """Implements dynamic rate limiting with exponential backoff and jitter."""

    def __init__(
        self,
        initial_delay=None,
        MAX_DELAY=None,
        backoff_factor=None,
        decrease_factor=None,  # ADDED decrease_factor to init
        config_instance=config_instance,
    ):
        """
        Initializes the rate limiter.
        """
        self.initial_delay = config_instance.INITIAL_DELAY
        self.MAX_DELAY = config_instance.MAX_DELAY
        self.backoff_factor = config_instance.BACKOFF_FACTOR
        self.decrease_factor = config_instance.DECREASE_FACTOR
        self.current_delay = self.initial_delay
        self.last_throttled = False

    # end of __int__

    def wait(self):
        """Wait for a dynamic delay based on rate-limiting conditions."""
        jitter = random.uniform(0.8, 1.2)  # ±20% jitter
        effective_delay = min(self.current_delay * jitter, self.MAX_DELAY)
        time.sleep(effective_delay)
        return effective_delay

    # End of wait

    def reset_delay(self):
        """Resets the delay to the initial value."""
        self.current_delay = self.initial_delay
        logger.info("Reset delay to initial value.")

    # End of reset_delay

    def decrease_delay(self):
        """Gradually reduce the delay when no rate limits are detected."""
        if self.current_delay > self.initial_delay:
            self.current_delay = max(
                self.current_delay * self.decrease_factor, self.initial_delay
            )
            logger.debug(
                f"No rate limit detected. Decreased delay to {self.current_delay:.2f} seconds."
            )
        self.last_throttled = False

    # End of decrease_delay

    def increase_delay(self):
        """Increase the delay exponentially when a rate limit is detected."""
        self.current_delay = min(
            self.current_delay * self.backoff_factor, self.MAX_DELAY
        )
        logger.debug(
            f"Rate limit detected. Increased delay to {self.current_delay:.2f} seconds."
        )
        self.last_throttled = True

    # End of increase_delay

    def is_throttled(self):
        """Returns True if the rate limiter is currently in a throttled state."""
        return self.last_throttled

    # End of is_throttled
# end of class DynamicRateLimiter


# ------------------------------
# Session Management
# ------------------------------

class SessionManager:
    """Manages the Selenium WebDriver session and database connection."""

    def __init__(self):
        """Initializes the SessionManager with configuration and session variables."""
        self.driver = None
        self.db_path = str(config_instance.DATABASE_FILE.resolve())
        self.selenium_config = selenium_config
        self.ancestry_username = config_instance.ANCESTRY_USERNAME
        self.ancestry_password = config_instance.ANCESTRY_PASSWORD
        self.debug_port = self.selenium_config.DEBUG_PORT
        self.chrome_user_data_dir = self.selenium_config.CHROME_USER_DATA_DIR
        self.profile_dir = self.selenium_config.PROFILE_DIR
        self.chrome_driver_path = self.selenium_config.CHROME_DRIVER_PATH
        self.chrome_browser_path = self.selenium_config.CHROME_BROWSER_PATH
        self.chrome_max_retries = self.selenium_config.CHROME_MAX_RETRIES
        self.chrome_retry_delay = self.selenium_config.CHROME_RETRY_DELAY
        self.headless_mode = self.selenium_config.HEADLESS_MODE
        self.session_active = False
        self.cache_dir = config_instance.CACHE_DIR
        self._db_conn_pool = None
        self.engine = None
        self.Session = None
        self.last_js_error_check = datetime.now()
        self.dynamic_rate_limiter = DynamicRateLimiter()
        self.csrf_token = None
        self.api_login_verified = False
        self.my_profile_id = None
        self.my_uuid = None
        self.my_tree_id = None
        self.tree_owner_name = None
        self.session_start_time = None
        # --- MODIFICATION: Configure requests.Session ---
        self._requests_session = requests.Session()
        # Configure adapter with increased pool size
        adapter = HTTPAdapter(
            pool_connections=20,  # Increase connections pool size
            pool_maxsize=50,  # Increase max connections per host
            max_retries=Retry(  # Optional: configure retries directly on adapter
                total=3,  # Example: Max 3 retries total
                backoff_factor=0.5,  # Example: Short delay between retries
                status_forcelist=[500, 502, 503, 504],  # Example retry statuses
            ),
        )
        self._requests_session.mount("http://", adapter)
        self._requests_session.mount("https://", adapter)
        logger.debug(
            "Configured requests.Session with HTTPAdapter (increased pool size)."
        )
        # --- END MODIFICATION ---
        self.cache = self.Cache()  # Instantiate Cache class
    # end of __init__

    def start_sess(self, action_name=None) -> Tuple[bool, Optional[WebDriver]]:
        """
        Starts or reuses a Selenium WebDriver session following a specific sequence:
        Initialize -> WebDriver -> Navigate Base -> Login Check -> URL Check -> Cookies -> CSRF -> Identifiers -> Tree Owner -> Success.
        """
        session_start_success = False
        retry_count = 0
        max_retries = self.selenium_config.CHROME_MAX_RETRIES

        # --- 1. Initialize Session Start ---
        logger.debug(f"1. Session start initiated for action: '{action_name}'...")
        self.driver = None
        self.cache.clear()
        self.csrf_token = None
        self.my_profile_id = None
        self.my_uuid = None
        self.my_tree_id = None
        self.tree_owner_name = None
        self.api_login_verified = False

        # Initialize DB Pool if needed (remains the same)
        if self._db_conn_pool is None:
            try:
                self.engine = create_engine(f"sqlite:///{self.db_path}")

                @event.listens_for(self.engine, "connect")
                def enable_foreign_keys(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    try:
                        cursor.execute("PRAGMA foreign_keys=ON")
                    finally:
                        cursor.close()

                self.Session = sessionmaker(bind=self.engine)
                Base.metadata.create_all(self.engine)
                self._db_conn_pool = ConnectionPool(
                    self.db_path, pool_size=config_instance.DB_POOL_SIZE
                )
                logger.debug("Database connection pool initialized.")
            except SQLAlchemyError as e:
                logger.critical(
                    f"Database initialization CRITICAL error: {e}", exc_info=True
                )
                return False, None

        if not hasattr(self, "_requests_session") or not isinstance(
            self._requests_session, requests.Session
        ):
            self._requests_session = requests.Session()
            logger.debug("requests.Session initialized (fallback).")

        # --- Retry Loop for Session Initialization ---
        while retry_count < max_retries and not session_start_success:
            retry_count += 1
            logger.debug(f"Attempting session start: {retry_count}/{max_retries}\n")

            try:
                # --- 2. Initialize WebDriver ---
                logger.debug("2. Initializing WebDriver instance.")
                self.driver = init_webdvr(attach_attempt=True)
                if not self.driver:
                    logger.error(
                        f"WebDriver initialization failed on attempt {retry_count}.\n"
                    )
                    time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                    continue

                logger.debug("WebDriver initialization successful.")

                # Navigate to Base URL to Stabilize ---
                logger.debug("Navigating to Base URL to stabilize initial state...")
                base_url_nav_ok = nav_to_page(
                    self.driver,
                    config_instance.BASE_URL,
                    selector="body",  # Wait for body
                    session_manager=self,
                )
                if not base_url_nav_ok:
                    logger.error(
                        "Failed to navigate to Base URL after WebDriver init. Aborting attempt.\n"
                    )
                    self.close_sess()
                    # Consider if retry is useful here or just fail
                    time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                    continue  # Retry the whole start_sess attempt
                logger.debug("Initial navigation to Base URL successful.\n")

                # --- 3. Check Login Status & Log In If Needed ---
                logger.debug("3. Checking login status.")
                # login_status now called *after* navigating to base URL
                login_stat = login_status(self)
                if login_stat is True:
                    logger.debug("User is logged in.")
                elif login_stat is False:
                    logger.debug("User not logged in. Attempting login process.")
                    login_result = log_in(self)
                    if login_result != "LOGIN_SUCCEEDED":
                        logger.error(
                            f"Login failed ({login_result}). Aborting session start attempt {retry_count}."
                        )
                        self.close_sess()
                        return False, None  # Fail permanently on login failure
                    logger.debug("Login process successful.")
                    # Re-verify after login attempt
                    if not login_status(self):
                        logger.error(
                            "Login status verification failed even after successful login attempt reported."
                        )
                        self.close_sess()
                        return False, None
                    logger.debug("Login status re-verified successfully after login.")
                else:  # login_stat is None (error during check)
                    logger.error(
                        f"Login status check failed critically. Aborting session start attempt {retry_count}."
                    )
                    self.close_sess()
                    return False, None
                logger.debug("Login status confirmed.\n")

                # --- 4. URL Check & Navigation to Base URL (Refined Logic) ---
                logger.debug("4. Re-checking current URL validity.")
                try:
                    current_url = self.driver.current_url
                    logger.debug(f"Current URL:\n{current_url}")
                    base_url_norm = config_instance.BASE_URL.rstrip("/") + "/"
                    signin_url_base = urljoin(
                        config_instance.BASE_URL, "account/signin"
                    )
                    logout_url_base = urljoin(config_instance.BASE_URL, "c/logout")
                    mfa_url_base = urljoin(
                        config_instance.BASE_URL, "account/signin/mfa/"
                    )

                    disallowed_starts = (signin_url_base, logout_url_base)
                    is_api_path = "/api/" in current_url

                    needs_navigation = False
                    if not current_url.startswith(
                        config_instance.BASE_URL
                    ):  # Check prefix loosely
                        needs_navigation = True
                        logger.debug("Reason: URL does not start with base URL.")
                    elif any(
                        current_url.startswith(path) for path in disallowed_starts
                    ):
                        needs_navigation = True
                        logger.debug(
                            f"Reason: URL starts with disallowed path ({current_url})."
                        )
                    elif is_api_path:
                        needs_navigation = True
                        logger.debug("Reason: URL contains '/api/'.")
                    # No need to re-check MFA here as login status check should have passed

                    if needs_navigation:
                        logger.info(
                            f"Current URL '{current_url}' still unsuitable. Re-navigating to base URL: {base_url_norm}"
                        )
                        if not nav_to_page(
                            self.driver,
                            base_url_norm,
                            selector="body",
                            session_manager=self,
                        ):
                            logger.error(
                                "Failed to re-navigate to base URL. Aborting session start."
                            )
                            if not self.is_sess_valid():
                                self.close_sess()
                            return False, None
                        logger.debug("Re-navigation to base URL successful.")
                    else:
                        logger.debug(
                            "Current URL is suitable, no extra navigation needed."
                        )

                except WebDriverException as e:
                    logger.error(
                        f"Error during URL re-check/navigation: {e}. Aborting session start.",
                        exc_info=True,
                    )
                    self.close_sess()
                    return False, None
                logger.debug("URL re-check completed.\n")

                # --- 5. Verify Essential Cookies ---
                logger.debug(
                    "5. Verifying essential cookies (ANCSESSIONID, SecureATT)."
                )
                essential_cookies = ["ANCSESSIONID", "SecureATT"]
                if not self.get_cookies(essential_cookies, timeout=15):
                    logger.error(
                        f"Essential cookies {essential_cookies} not found. Aborting session start."
                    )
                    self.close_sess()
                    return False, None
                logger.debug("Essential cookies verified.\n")

                # --- 6. Synchronize Cookies to requests.Session ---
                logger.debug("6. Syncing cookies from WebDriver to requests session.")
                self._sync_cookies()
                logger.debug("Cookies synced.\n")

                # --- 7. Retrieve and Store CSRF Token ---
                logger.debug("7. Retrieving CSRF token.")
                self.csrf_token = self.get_csrf()
                if not self.csrf_token:
                    logger.error(
                        "Failed to retrieve CSRF token. Aborting session start."
                    )
                    self.close_sess()
                    return False, None
                logger.debug(f"CSRF token retrieved: {self.csrf_token[:10]}...\n")

                # --- 8. Retrieve User Identifiers ---
                logger.debug("8. Retrieving user identifiers.")
                self.my_profile_id = self.get_my_profileId()
                if not self.my_profile_id:
                    logger.error(
                        "Failed to retrieve profile ID (ucdmid). Aborting session start."
                    )
                    self.close_sess()
                    return False, None
                logger.info(f"My profile id: {self.my_profile_id}")

                self.my_uuid = self.get_my_uuid()
                if not self.my_uuid:
                    logger.error(
                        "Failed to retrieve UUID (testId). Aborting session start."
                    )
                    self.close_sess()
                    return False, None
                logger.info(f"My uuid: {self.my_uuid}")

                self.my_tree_id = self.get_my_tree_id()
                if config_instance.TREE_NAME and not self.my_tree_id:
                    logger.error(
                        f"TREE_NAME '{config_instance.TREE_NAME}' configured, but failed to get corresponding tree ID. Aborting session start."
                    )
                    self.close_sess()
                    return False, None
                elif self.my_tree_id:
                    logger.info(f"My tree id: {self.my_tree_id}")
                else:
                    logger.debug("No TREE_NAME configured or Tree ID not found.")
                logger.debug("Finished retrieving user identifiers.\n")

                # --- 9. Retrieve Tree Owner Name ---
                logger.debug("9. Retrieving tree owner name.")
                if self.my_tree_id:
                    self.tree_owner_name = self.get_tree_owner(self.my_tree_id)
                    if self.tree_owner_name:
                        logger.info(f"Tree Owner Name: {self.tree_owner_name}\n")
                    else:
                        logger.warning("Failed to retrieve tree owner name.\n")
                else:
                    logger.debug("Skipping tree owner retrieval (no tree ID).\n")

                # --- 10. Session Start Successful ---
                session_start_success = True
                self.session_active = True
                self.session_start_time = time.time()
                self.last_js_error_check = datetime.now()
                # Use INFO level for final success message
                logger.debug(
                    f"10. Session started successfully on attempt {retry_count}.\n"
                )
                return True, self.driver

            # --- Handle Exceptions during the attempt (remains the same) ---
            except WebDriverException as wd_exc:
                logger.error(
                    f"WebDriverException during session start attempt {retry_count}: {wd_exc}",
                    exc_info=False,
                )
                logger.debug("Closing potentially broken WebDriver instance for retry.")
                self.close_sess()
                if retry_count < max_retries:
                    logger.info(
                        f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next attempt..."
                    )
                    time.sleep(self.selenium_config.CHROME_RETRY_DELAY)

            except Exception as e:
                logger.error(
                    f"Unexpected error during session start attempt {retry_count}: {e}",
                    exc_info=True,
                )
                self.close_sess()
                if retry_count < max_retries:
                    logger.info(
                        f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next attempt..."
                    )
                    time.sleep(self.selenium_config.CHROME_RETRY_DELAY)

        # --- End of Retry Loop ---
        if not session_start_success:
            logger.critical(
                f"Session start FAILED permanently after {max_retries} attempts."
            )
            self.close_sess()
            return False, None

        # Fallback (should not be reached)
        logger.error(
            "start_sess exited retry loop unexpectedly without success/failure return."
        )
        self.close_sess()
        return False, None
    # end start_sess

    def get_csrf(self) -> Optional[str]:
        """
        Fetches the CSRF token.

        It first checks the cache. If not found, it retrieves it from the API endpoint,
        caches it, and then returns it.
        """
        if self.cache.get("csrfToken"):  # Check cache first
            logger.debug("Using cached CSRF token.")
            return self.cache.get("csrfToken")

        essential_cookies = [
            "ANCSESSIONID",
            "SecureATT",
        ]  # Essential cookies for API call
        if not self.get_cookies(essential_cookies, timeout=30):  # Validate cookies
            logger.warning(
                f"Essential cookies {essential_cookies} NOT found before CSRF token API call."
            )
            return None

        csrf_token_url = urljoin(
            config_instance.BASE_URL, "discoveryui-matches/parents/api/csrfToken"
        )
        headers = {
            "User-Agent": self.driver.execute_script("return navigator.userAgent"),
            "Accept": "application/json",
            "Referer": config_instance.BASE_URL,
            "ancestry-context-ube": make_ube(self.driver),
        }

        if self.driver is None:
            logger.error("WebDriver is not initialized.")
            return None
        response_data = _api_req(
            url=csrf_token_url,
            driver=self.driver,
            session_manager=self,
            method="GET",
            use_csrf_token=False,
            api_description="CSRF Token API",
        )
        if not response_data:
            logger.warning("Failed to get CSRF token - _api_req returned None.")
            return None

        if isinstance(response_data, dict) and "csrfToken" in response_data:
            csrf_token = response_data["csrfToken"]
        elif isinstance(response_data, str):  # Plain text token
            csrf_token = response_data
            logger.info("CSRF token retrieved as plain text string from API endpoint.")
        else:
            logger.error("Unexpected response format for CSRF token API.")
            logger.debug(
                f"Response data type: {type(response_data)}, value: {response_data}"
            )
            return None

        if csrf_token:
            self.cache.set("csrfToken", csrf_token)  # Cache the token
            return csrf_token
        else:
            logger.warning(
                "CSRF token retrieval failed from API endpoint after processing response."
            )
            return None
    # end get_csrf

    def get_cookies(self, cookie_names, timeout=30):
        """Waits until specified cookies are present and returns True, else False."""
        start_time = time.time()
        logger.debug(f"Seeking {cookie_names}.")
        required = {name.lower() for name in cookie_names}
        interval = 0.5
        last_missing = None

        while True:
            try:
                if not self.driver:
                    logger.warning("Driver is None! Cannot retrieve cookies.")
                    return False

                cookies = self.driver.get_cookies()
                current_cookies = {c["name"].lower() for c in cookies}
                found = required.intersection(current_cookies)
                missing = required - found

                if not missing:
                    logger.debug(f"Cookies found.")
                    return True
                if missing != last_missing:
                    original_missing = [
                        name for name in cookie_names if name.lower() in missing
                    ]
                    logger.debug(
                        f"Missing cookies: {original_missing}. Found: {[c['name'] for c in cookies]}"
                    )
                    last_missing = missing
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    original_missing = [
                        name for name in cookie_names if name.lower() in missing
                    ]
                    logger.warning(f"Timeout waiting for cookies: {original_missing}.")
                    return False
                time.sleep(min(interval, timeout - elapsed))
            except Exception as e:
                logger.error(f"Error retrieving cookies: {e}", exc_info=True)
                if time.time() - start_time >= timeout:
                    logger.warning(
                        "Timeout reached after error during cookie retrieval."
                    )
                    return False
                time.sleep(min(interval, timeout - (time.time() - start_time)))
    # end get_cookies

    def _sync_cookies(self):
        """Syncs cookies from WebDriver to requests session, adjusting domain for SecureATT."""
        if not self.driver:
            logger.warning("Cannot sync cookies: WebDriver is None.")
            return

        cookies = self.driver.get_cookies()
        for cookie in cookies:
            if "expiry" in cookie:
                cookie["expires"] = cookie.pop("expiry")
            # Adjust domain for SecureATT cookie
            if (
                cookie["name"] == "SecureATT"
                and cookie["domain"] == "www.ancestry.co.uk"
            ):
                cookie["domain"] = ".ancestry.co.uk"
            self._requests_session.cookies.set(
                cookie["name"],
                cookie["value"],
                domain=cookie["domain"],
                path=cookie.get("path", "/"),
            )
        logger.debug(f"Synced {self._requests_session.cookies.list_domains()}")
    # end _sync_cookies

    class Cache:
        """
        Simple in-memory cache to store API responses to avoid redundant calls.
        Using a class to namespace the cache and avoid potential attribute conflicts in SessionManager.
        """

        def __init__(self):
            self._cache = {}

        def get(self, key):
            return self._cache.get(key)

        def set(self, key, value, timeout=None):
            self._cache[key] = value
            return True  # Indicate successful caching

        def clear(self):
            self._cache = {}
            logger.debug("Cache cleared.")

        #
    # End of Cache class

    def return_session(self, session: Session):
        """Returns a Session to the connection pool if space is available, otherwise closes it."""
        if not session:
            logger.warning("Attempted to return a None session to the pool.")
            return

        session_id = id(session)  # Get ID for logging before potential close

        # Rollback any pending changes before returning/closing
        # This prevents returning a session with an open transaction.
        try:
            # Check if session is still active before attempting rollback
            if session.is_active:
                if session.dirty:
                    logger.warning(
                        f"Session {session_id} returned dirty. Rolling back..."
                    )
                    session.rollback()
                # Close the session after potential rollback, before returning to pool or final close
                # session.close() # Close it here? Or let pool handle it? Let pool handle for now.

            else:
                logger.warning(
                    f"Session {session_id} returned inactive. Cannot rollback/return to pool."
                )
                # Ensure it's fully closed if inactive
                try:
                    session.close()
                except Exception:
                    pass
                return  # Don't try to add an inactive session to the pool

        except Exception as rb_err:
            logger.error(
                f"Error checking/rolling back session {session_id} on return: {rb_err}"
            )
            # Proceed to close/discard the session if rollback fails
            try:
                session.close()
            except Exception:
                pass
            return

        # Check if pool exists and has space (pool might be None if cls_db_conn was called)
        # Also ensure the pool itself and its internal list are valid before checking length/appending
        if (
            self._db_conn_pool
            and hasattr(self._db_conn_pool, "_pool")
            and isinstance(self._db_conn_pool._pool, list)
            and len(self._db_conn_pool._pool) < self._db_conn_pool.pool_size
        ):
            # Check again if active before appending, as operations in finally might change state
            if session.is_active:
                self._db_conn_pool._pool.append(session)
                logger.debug(
                    f"Returned session {session_id} to pool. Pool size now: {len(self._db_conn_pool._pool)}\n"
                )
            else:
                logger.warning(
                    f"Session {session_id} became inactive before returning to pool. Closing.\n"
                )
                try:
                    session.close()
                except Exception:
                    pass
        elif self._db_conn_pool:
            # Pool is full or doesn't exist, close the session instead
            logger.debug(
                f"Pool full or unavailable. Closing returned session {session_id}."
            )
            try:
                if session.is_active:  # Only close if active
                    session.close()
                    logger.debug(f"Session {session_id} closed.")
                else:
                    logger.debug(
                        f"Session {session_id} already inactive, not closing again."
                    )
            except SQLAlchemyError as e:
                logger.error(
                    f"Error closing session {session_id} when returning to full/no pool: {e}"
                )
            except Exception as e:  # Catch other potential errors during close
                logger.error(
                    f"Unexpected error closing session {session_id} when returning: {e}"
                )
        else:
            # Pool doesn't exist anymore
            logger.debug(
                f"DB Pool does not exist. Closing returned session {session_id}."
            )
            try:
                if session.is_active:
                    session.close()
            except Exception:
                pass
    # end return_session

    def get_db_conn(self) -> Optional[Session]:
        """
        Retrieves a database session from the pool. Initializes the pool if necessary.
        Uses the db_path and pool_size from the SessionManager instance.
        """
        if not self.db_path:
            logger.error("Cannot get DB connection: db_path not set in SessionManager.")
            return None

        # Check if the pool needs initialization or re-initialization
        if self._db_conn_pool is None:
            try:
                logger.debug(
                    f"Initializing or re-initializing DB connection pool for: {self.db_path}"
                )
                # --- MODIFICATION: Pass pool_size from config ---
                # Ensure db_path is a string for ConnectionPool constructor
                pool_size = config_instance.DB_POOL_SIZE  # Get from config
                self._db_conn_pool = ConnectionPool(
                    str(self.db_path), pool_size=pool_size
                )
                # --- END MODIFICATION ---
                logger.debug(f"Connection pool initialized with size {pool_size}.")
            except Exception as e:
                logger.critical(
                    f"Failed to initialize DB connection pool: {e}", exc_info=True
                )
                self._db_conn_pool = None  # Ensure it's None if init fails
                return None

        # Pool should be initialized now, attempt to get a connection
        try:
            session = self._db_conn_pool.get_session()
            if session:
                return session
            else:
                # This case should ideally be handled within ConnectionPool.get_session()
                logger.error("Connection pool returned None session.")
                return None
        except Exception as e:
            logger.error(f"Error getting DB session from pool: {e}", exc_info=True)
            return None
    # End of get_db_conn

    @contextlib.contextmanager
    def get_db_conn_context(self) -> Generator[Optional[Session], None, None]:
        """
        Context manager to get a database session from the pool, manage the
        transaction (commit/rollback), and ensure the session is returned.
        Yields the session object or None if retrieval fails.
        """
        session: Optional[Session] = (
            self.get_db_conn()
        )  # Use the existing method to get a session

        if not session:
            logger.error("Failed to obtain DB session within context manager.")
            # Yield None and exit the generator cleanly if no session
            yield None
            return  # Exit generator

        # If session was obtained, proceed with transaction management
        try:
            logger.debug("DB session obtained via context manager. Yielding.")
            yield session  # Provide the session to the 'with' block
            # --- COMMIT: If the 'with' block completes without error ---
            if session.is_active:  # Check if still active
                logger.debug("Committing transaction within context manager.")
                session.commit()
                logger.debug("Transaction committed successfully.")
            else:
                logger.warning(
                    "Session became inactive before commit in context manager."
                )

        except Exception as e:
            # Log exceptions occurring *within* the 'with' block
            logger.error(
                f"Exception within get_db_conn_context block: {e}. Rolling back.",
                exc_info=True,
            )
            # --- ROLLBACK: If an error occurs within the 'with' block ---
            if session.is_active:  # Check if still active before rollback
                try:
                    session.rollback()
                    logger.warning(
                        "Rolled back transaction due to exception in context block."
                    )
                except Exception as rb_err:
                    logger.error(
                        f"Error rolling back session in context manager: {rb_err}"
                    )
            else:
                logger.warning(
                    "Session became inactive before rollback in context manager."
                )
            raise  # Re-raise the exception so the caller knows about it

        finally:
            # This block executes whether the 'with' block succeeded or raised an exception
            if session:
                # Always return the session to the pool
                logger.debug("Returning DB session obtained via context manager.")
                self.return_session(session)  # Use the existing return method
            # else: # Case where session was None initially is handled above
            #     pass
    # End of get_db_conn_context method

    def cls_db_conn(self):
        """Closes all database connections in the pool AND disposes the engine and resets CSRF token."""
        if self._db_conn_pool:
            self._db_conn_pool.clse_all_sess()
            if self.engine:
                self.engine.dispose()
                self._db_conn_pool = None
                self.engine = None
                logger.debug(
                    "Database sessions closed, pool cleared and engine disposed."
                )
        self.csrf_token = None  # Reset CSRF Token on session close/restart - CORRECTED - using self.csrf_token (public)
        logger.debug("CSRF token cache reset.")  # DEBUG log - CSRF cache reset
    # End of cls_db_conn

    def get_my_profileId(self) -> Optional[str]:
        """
        Retrieves the user ID (ucdmid) from the Ancestry API.
        Ensures _api_req is used correctly.
        """
        # Check cache first (optional, profile ID rarely changes)
        # cached_id = self.cache.get('my_profile_id')
        # if cached_id:
        #     logger.debug("Using cached profile ID.")
        #     return cached_id

        url = urljoin(
            config_instance.BASE_URL,
            "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid",
        )
        try:
            # Use _api_req helper
            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,  # GET doesn't need CSRF
                api_description="Get my profile_id",  # Key for contextual headers
            )

            if not response_data:
                logger.warning("Failed to get profile_id response via _api_req.")
                return None

            # Process the response data structure
            if (
                isinstance(response_data, dict)
                and "data" in response_data
                and "ucdmid" in response_data["data"]
            ):
                my_profile_id = response_data["data"]["ucdmid"].upper()
                # self.cache.set('my_profile_id', my_profile_id) # Cache if desired
                return my_profile_id
            else:
                logger.error("Could not find 'data.ucdmid' in profile_id API response.")
                logger.debug(f"Full profile_id response data: {response_data}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in get_my_profileId: {e}", exc_info=True)
            return None
    # end get_my_profileId

    @retry()
    def get_my_uuid(self):
        """Retrieves the test uuid (sampleId) from the header/dna API endpoint"""
        # Check cache first (UUID rarely changes)
        # cached_uuid = self.cache.get('my_uuid')
        # if cached_uuid:
        #     # logger.debug("Using cached UUID.")
        #     return cached_uuid

        if not isinstance(self.driver, WebDriver) or not self.is_sess_valid():
            logger.error("get_my_uuid: Driver is None or session invalid.")
            return None  # Return None instead of False for consistency

        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        # Use _api_req helper
        response_data = _api_req(
            url=url,
            driver=self.driver,
            session_manager=self,
            method="GET",
            use_csrf_token=False,  # GET doesn't need CSRF
            api_description="Get UUID API",  # Key for contextual headers (though defaults might be fine)
        )

        if response_data:
            # Validate the response structure
            if isinstance(response_data, dict) and "testId" in response_data:
                my_uuid_val = response_data["testId"].upper()
                # self.cache.set('my_uuid', my_uuid_val) # Cache if desired
                return my_uuid_val
            else:
                logger.error(
                    "Could not retrieve my_uuid ('testId' missing in response)."
                )
                logger.debug(f"Full get_my_uuid response data: {response_data}")
                return None
        else:
            logger.error("Failed to get header/dna data via _api_req.")
            return None
    # end of get_my_uuid

    def get_my_tree_id(self) -> Optional[str]:
        """
        Retrieves the tree ID based on TREE_NAME from config, using the header/trees API.

        Returns:
            The tree ID (string) if found, otherwise None.
        """
        tree_name_config = config_instance.TREE_NAME
        if not tree_name_config:
            logger.debug(
                "TREE_NAME not configured in .env file, skipping tree ID retrieval."
            )
            return None

        # Check cache first (tree ID rarely changes)
        # cached_tree_id = self.cache.get('my_tree_id')
        # if cached_tree_id:
        #     # logger.debug("Using cached Tree ID.")
        #     return cached_tree_id

        if not isinstance(self.driver, WebDriver) or not self.is_sess_valid():
            logger.error("get_my_tree_id: Driver is None or session invalid.")
            return None

        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/trees")

        try:
            # Use _api_req helper
            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,  # GET doesn't need CSRF
                api_description="Header Trees API",  # Key for contextual headers
            )
            if (
                response_data
                and isinstance(response_data, dict)
                and "menuitems" in response_data
            ):
                for item in response_data["menuitems"]:
                    # Ensure 'text' exists and compare, then extract ID safely
                    if (
                        isinstance(item, dict)
                        and "text" in item
                        and item["text"] == tree_name_config
                    ):
                        tree_url = item.get("url")
                        if tree_url and isinstance(tree_url, str):
                            parts = tree_url.split("/")
                            if len(parts) > 3:
                                my_tree_id_val = parts[3]
                                # self.cache.set('my_tree_id', my_tree_id_val) # Cache if desired
                                return my_tree_id_val
                            else:
                                logger.warning(
                                    f"Found tree '{tree_name_config}', but URL format unexpected: {tree_url}"
                                )
                        else:
                            logger.warning(
                                f"Found tree '{tree_name_config}', but 'url' key missing or invalid."
                            )
                        break  # Stop searching once found (even if URL was bad)

                # If loop finishes without finding/returning
                logger.warning(
                    f"Could not find TREE_NAME '{tree_name_config}' in Header Trees API response menu items."
                )
                return None
            else:
                logger.warning(
                    "Unexpected response format from Header Trees API (missing 'menuitems'?)."
                )
                logger.debug(f"Full Header Trees response data: {response_data}")
                return None
        except Exception as e:
            logger.error(f"Error fetching/parsing Header Trees API: {e}", exc_info=True)
            return None
    # End of get_my_tree_id

    def get_tree_owner(self, tree_id: str) -> Optional[str]:
        """
        Retrieves the tree owner's display name from the Ancestry API using _api_req.
        """
        if not tree_id:
            logger.warning("Cannot get tree owner: tree_id is missing.")
            return None

        if not isinstance(self.driver, WebDriver) or not self.is_sess_valid():
            logger.error("get_tree_owner: Driver is None or session invalid.")
            return None

        url = urljoin(
            config_instance.BASE_URL,
            f"api/uhome/secure/rest/user/tree-info?tree_id={tree_id}",
        )

        try:
            # Use _api_req helper
            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,  # GET doesn't need CSRF
                api_description="Tree Owner Name API",  # Key for contextual headers
            )
            if response_data and isinstance(response_data, dict):
                # Safely access nested data
                owner_data = response_data.get("owner")
                if owner_data and isinstance(owner_data, dict):
                    display_name = owner_data.get("displayName")
                    if display_name and isinstance(display_name, str):
                        return display_name
                    else:
                        logger.warning("Could not find 'displayName' in 'owner' data.")
                else:
                    logger.warning(
                        "Could not find 'owner' data in Tree Owner API response."
                    )
                # Log response if data missing
                logger.debug(f"Full Tree Owner API response data: {response_data}")
                return None
            else:
                logger.warning(
                    "Tree Owner API call via _api_req returned unexpected data or None."
                )
                logger.debug(f"Response received: {response_data}")
                return None
        except Exception as e:
            logger.error(f"Error fetching/parsing Tree Owner API: {e}", exc_info=True)
            return None
    # End of get_tree_owner

    def verify_sess(self):
        """Enhanced session verification with login attempt. REDUCED TIMEOUTS"""
        logger.info("Session Verification Started.")

        try:
            # Navigate to base URL
            logger.debug("Navigating to Base URL...")
            if self.driver:
                self.driver.get(config_instance.BASE_URL)
            else:
                logger.error("Driver is not initialized.")
                return False

            # Wait for page load
            try:
                WebDriverWait(self.driver, 20).until(  # Reduced timeout to 20s
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                logger.debug(
                    "Page Load Successful.",
                    {
                        "current_url": self.driver.current_url,
                        "page_title": self.driver.title,
                        "page_source_length": len(self.driver.page_source),
                    },
                )

            except TimeoutException as e:
                logger.error(
                    "Page Load Timeout.",
                    {
                        "current_url": self.driver.current_url,
                        "page_state": self.driver.execute_script(
                            "return document.readyState"
                        ),
                    },
                    exc_info=True,
                )
                return False

            # Check for session indicators
            if login_status(self):
                logger.info("Session Active.")
                return True
            else:
                logger.warning("Session Expired - Attempting Login.")
                return log_in(self)

        except Exception as e:
            logger.error(
                "Session Verification Failed.",
                {
                    "error": str(e),
                    "current_url": (
                        self.driver.current_url if self.driver else "No driver"
                    ),
                },
                exc_info=True,
            )
            return False
    # End of verify_sess

    def _verify_api_login_status(self) -> bool:
        """
        Checks login status by making a request to a known secure API endpoint using requests.
        Assumes cookies have already been synced to self._requests_session.
        Uses header/dna endpoint. Returns True if API call succeeds (implies login), False otherwise.
        Handles the fact that _api_req returns parsed data on success, None on failure.
        """
        # --- Use cached flag first ---
        if self.api_login_verified:
            logger.debug("API login already verified this session (cached).")
            return True

        logger.debug("Verifying login status via header/dna API endpoint...")
        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        api_description = "API Login Verification (header/dna)"

        try:
            # Make the request using _api_req, forcing requests library
            response_data = _api_req(
                url=url,
                driver=self.driver,  # For UBE
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description=api_description,
                force_requests=True,  # Use requests library path
            )

            # --- CORRECTED LOGIC ---
            # _api_req returns the parsed data on success (2xx status), otherwise None.
            # So, if response_data is not None, the API call was successful.
            if response_data is not None:
                # Optional extra check: Ensure expected data structure for this specific endpoint
                if isinstance(response_data, dict) and "testId" in response_data:
                    logger.debug(
                        f"API login check successful via {api_description} (received valid data)."
                    )
                    self.api_login_verified = True  # Set cache flag on success
                    return True
                else:
                    # Succeeded HTTP-wise, but data format unexpected - still treat as logged in for now?
                    logger.warning(
                        f"API login check via {api_description} succeeded (2xx), but response format unexpected: {response_data}"
                    )
                    # Decide how to handle - safer to assume logged in if HTTP was OK, but flag it.
                    self.api_login_verified = (
                        True  # Cautiously set True, but warning was logged
                    )
                    return True
            else:
                # _api_req returned None, meaning the HTTP call failed (4xx, 5xx, timeout, network error)
                logger.warning(
                    f"API login check failed: {api_description} call returned None (likely HTTP error or timeout)."
                )
                self.api_login_verified = False  # Explicitly set False on failure
                return False
            # --- END CORRECTED LOGIC ---

        except Exception as e:
            # Catch unexpected errors during the _api_req call itself or processing
            logger.error(
                f"Unexpected error during API login status check ({api_description}): {e}",
                exc_info=True,
            )
            self.api_login_verified = False
            return False
    # end _verify_api_login_status

    @retry()
    def get_header(self):
        """Retrieves data from the headerdata API endpoint."""
        if not self.driver:  # ADD THIS CHECK
            logger.error(
                "get_header: Driver is None! Session may not have started correctly."
            )
            return False  # Or raise an exception, depending on desired error handling

        # --- SIMPLIFIED HEADERS - Rely on _api_req ---
        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Priority": "u=1, i",
            "Referer": config_instance.BASE_URL,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        response_data = _api_req(
            url, self.driver, self, method="GET", headers=headers, use_csrf_token=False
        )  # Corrected call - added self (session_manager)
        if response_data:
            # Validate the response structure
            if "testId" in response_data:  # corrected validation
                logger.info("Header data retrieved successfully.")
                return True
            else:
                logger.error("Unexpected response structure from headerdata API.")
                return False
        else:
            logger.error("Failed to get headerdata.")
            return False
    # end get_header

    def _validate_ess_cookies(self, required_cookies):
        """Check if specific cookies exist in the current session."""
        if self.driver:
            cookies = {c["name"]: c["value"] for c in self.driver.get_cookies()}
        else:
            logger.error("Driver is None, cannot get cookies.")
            return False
        return all(cookie in cookies for cookie in required_cookies)
    # End of _validate_ess_cookies

    def _init_hdrs(self):
        """Initializes common API headers. - OLD HEADERS (likely problem)"""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",  # <--- Unnecessary?
            "User-Agent": self.driver.execute_script("return navigator.userAgent"),
        }
    # End of _init_hdrs

    def is_sess_logged_in(self):
        """Checks session validity AND login using UI element verification ONLY."""
        try:
            if not self.driver:
                logger.debug("No driver. Invalid session.")
                return False

            # --- 1. Basic WebDriver check ---
            if not self.is_sess_valid():
                logger.debug("Session is invalid.")
                return False

            # --- 2. UI Element Check for Logged-in Status (FAST)---
            # This is now the *only* check in this function.
            ui_logged_in = login_status(self)
            if ui_logged_in is None:  # Uncertain state
                logger.warning("Inconclusive whether logged in.")
                return False  # Treat uncertain as invalid
            elif not ui_logged_in:
                logger.info("User is NOT logged in.")
                return False  # Not logged in according to UI

            # If we reach here, the UI check passed
            return True

        except Exception as e:
            logger.debug(f" Session check failed: {e}")
            return False
    # End is_sess_logged_in

    def _verify_login(self) -> bool:
        """
        Performs an API-based login verification using a reliable endpoint.
        Caches the verification result for the session duration.
        Returns True if verified, False otherwise.
        CHANGED to use /api/uhome/secure/rest/header/dna endpoint.
        """
        if self.api_login_verified:  # Check cache flag first
            # logger.debug("API login already verified this session.") # Less verbose
            return True

        logger.debug("Performing API-based login verification...")

        # --- CHANGED Endpoint ---
        # Use the header/dna endpoint which is generally reliable after login
        # and used successfully elsewhere (get_my_uuid).
        verification_url = urljoin(
            config_instance.BASE_URL, "api/uhome/secure/rest/header/dna"
        )
        api_description = "API Login Verification (header/dna)"
        # --- End Change ---

        try:
            # Use _api_req, assuming GET request doesn't need CSRF for this endpoint
            response_data = _api_req(
                url=verification_url,
                driver=self.driver,  # Pass driver for header generation if needed
                session_manager=self,
                method="GET",
                use_csrf_token=False,  # Set explicitly to False for this endpoint
                api_description=api_description,
                # Headers are handled by _api_req based on description/defaults
            )

            # Check for a key that indicates success (e.g., 'testId')
            if (
                response_data
                and isinstance(response_data, dict)
                and "testId" in response_data
            ):
                self.api_login_verified = True  # Cache verification success
                logger.debug("API login verification successful.")
                return True
            else:
                logger.warning(
                    f"{api_description} failed: Unexpected response format or missing 'testId'."
                )
                logger.debug(f"Response data received: {response_data}")
                # Don't cache failure, try again next time
                return False

        except Exception as api_e:
            # Catch exceptions raised by _api_req (like HTTPError, Timeout)
            logger.warning(
                f"{api_description} failed due to exception: {api_e}", exc_info=False
            )  # Log less verbosely here
            logger.debug(
                "Traceback for API verification failure:", exc_info=True
            )  # Debug traceback
            return False
    # End of _verify_login

    def is_sess_valid(self):
        """Checks if the current browser session is valid (quick check), handling InvalidSessionIdException."""
        if not self.driver:
            logger.debug("Browser not open.")
            return False
        try:
            # Try a minimal WebDriver command to check session validity
            self.driver.title  # Accessing .title is a fast check.
            return True  # If no exception, browser session is likely valid
        except (
            InvalidSessionIdException
        ):  # Catch specific exception for invalid session
            logger.debug("Session ID is invalid (browser likely closed).")
            return False
        except Exception as e:  # Catch any other exceptions (e.g., WebDriverException)
            logger.warning(f"Error checking session validity: {e}")
            return False
    # End of is_sess_valid

    def close_sess(self):
        """Closes the Selenium WebDriver session."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.error(f"Error closing WebDriver session: {e}", exc_info=True)
            finally:
                self.driver = None
        self.session_active = False
    # End of close_sess

    def restart_sess(self, url=None):
        """Restarts the WebDriver session and optionally navigates to a URL."""
        logger.warning("Restarting WebDriver session...")
        self.close_sess()  # Ensure any existing session is closed
        self.start_sess()
        if url and self.driver:
            try:
                nav_to_page(self.driver, url, selector="main")
                logger.info(f"Successfully re-navigated to {url}.")
            except Exception as e:
                logger.error(f"Failed to re-navigate to {url} after restart: {e}")
    # End of restart_sess

    @ensure_browser_open
    def make_tab(self):
        """Create a new tab and return its handle id"""
        driver = self.driver
        try:
            # 1. Get initial original tab handles
            if driver is None:
                logger.error("WebDriver is not initialized.")
                return None
            tab_list = driver.window_handles
            # logger.debug(f"Initial window handles: {tab_list}")

            # 2. Create new tab using Selenium's new_window method
            driver.switch_to.new_window("tab")
            # 3. Get handles and determine the new tab
            new_tab_list = driver.window_handles
            details_tab = [handle for handle in new_tab_list if handle not in tab_list][
                0
            ]
            # logger.debug(f"New tab handle: {details_tab}")
            return details_tab
        except TimeoutException:
            logger.error("Timeout waiting for new tab to open.")
            if driver is not None:
                logger.debug(f"Window handles during timeout: {driver.window_handles}")
            else:
                logger.debug("Driver is None during timeout.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return None
    # End of make_tab

    def check_js_errors(self):
        """Checks for new JavaScript errors since the last check and logs them."""
        if not self.driver:
            return  # No driver, no errors to check

        try:
            for entry in self.driver.get_log("browser"):
                if entry["level"] == "SEVERE":
                    # Check if the error occurred *after* the last check
                    timestamp = datetime.fromtimestamp(
                        entry["timestamp"] / 1000
                    )  # Convert ms to seconds
                    if timestamp > self.last_js_error_check:
                        logger.warning(f"New JavaScript Error: {entry['message']}")
                        self.last_js_error_check = (
                            timestamp  # Update to time of this error.
                        )
        except Exception as e:
            logger.error(f"Error checking for Javascript errors: {e}", exc_info=True)
    # end of check_js_errors
# end SessionManager


# ----------------------------------------------------------------------------
# Stand alone functions
# ----------------------------------------------------------------------------

def _api_req(
    url: str,
    driver: Optional[WebDriver],
    session_manager: SessionManager,
    method: str = "GET",
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    use_csrf_token: bool = True,
    headers: Optional[Dict] = None,
    referer_url: Optional[str] = None,
    api_description: str = "API Call",
    timeout: Optional[int] = None,
    force_requests: bool = False,  # Added flag to force requests path
) -> Optional[Any]:
    """
    V13.1 REVISED: Makes an HTTP request using Python requests library with retry logic.
    Handles headers, CSRF, cookies, and rate limiting interaction.
    Includes enhanced error logging.
    """
    if not session_manager:
        logger.error(
            f"{api_description}: Aborting - SessionManager instance is required."
        )
        return None

    # --- Retry Configuration ---
    max_retries = config_instance.MAX_RETRIES
    initial_delay = config_instance.BACKOFF_FACTOR
    backoff_factor = config_instance.BACKOFF_FACTOR
    max_delay = config_instance.MAX_DELAY
    retry_status_codes = config_instance.RETRY_STATUS_CODES

    # --- Prepare Headers (Common Logic) ---
    final_headers = {}
    contextual_headers = config_instance.API_CONTEXTUAL_HEADERS.get(api_description, {})
    final_headers.update({k: v for k, v in contextual_headers.items() if v is not None})
    if headers:
        final_headers.update({k: v for k, v in headers.items() if v is not None})
    if "User-Agent" not in final_headers:
        ua = None
        # Attempt to get from driver first for consistency if available
        if driver and session_manager.is_sess_valid():  # Check validity
            try:
                ua = driver.execute_script("return navigator.userAgent;")
            except Exception:
                logger.warning(
                    f"{api_description}: Failed to get User-Agent from driver, using random."
                )
        if not ua:
            ua = random.choice(config_instance.USER_AGENTS)
        final_headers["User-Agent"] = ua
    if referer_url and "Referer" not in final_headers:
        final_headers["Referer"] = referer_url
    if use_csrf_token:
        csrf = session_manager.csrf_token
        if csrf:
            final_headers["X-CSRF-Token"] = csrf
        else:
            logger.warning(f"{api_description}: CSRF token required but not available.")
    # Add UBE if driver available and session valid
    if (
        driver
        and session_manager.is_sess_valid()
        and "ancestry-context-ube" not in final_headers
    ):
        ube_header = make_ube(driver)
        if ube_header:
            final_headers["ancestry-context-ube"] = ube_header
        else:
            logger.warning(f"{api_description}: Failed to generate UBE header.")
    # Add ancestry-userid if needed and available
    if "ancestry-userid" in contextual_headers and session_manager.my_profile_id:
        final_headers["ancestry-userid"] = session_manager.my_profile_id.upper()

    # --- Set Timeout for requests ---
    request_timeout = timeout if timeout is not None else selenium_config.API_TIMEOUT

    # --- Use Python Requests Library ---
    logger.debug(f"Executing API call via Python requests: {method.upper()} \n{url}")
    req_session = session_manager._requests_session
    retries_left = max_retries
    last_exception = None

    while retries_left > 0:
        attempt = max_retries - retries_left + 1
        response = None  # Initialize response to None for error handling scope
        try:
            # --- Sync cookies before each attempt ---
            try:
                # Only sync if driver is available and session seems valid
                if driver and session_manager.is_sess_valid():
                    session_manager._sync_cookies()
                # else: logger.debug("Skipping cookie sync (no driver or invalid session).")
            except Exception as sync_err:
                logger.warning(
                    f"{api_description}: Error syncing cookies before attempt {attempt}: {sync_err}"
                )
                # Continue attempt, but session might fail

            response = req_session.request(
                method=method.upper(),
                url=url,
                headers=final_headers,
                data=data,
                json=json_data,
                timeout=request_timeout,
                verify=True,  # Standard verification
            )
            status = response.status_code
            logger.debug(
                f"{api_description}: Attempt {attempt}/{max_retries} completed. Status: {status}"
            )

            # Check for retryable status codes
            if status in retry_status_codes:
                retries_left -= 1
                last_exception = HTTPError(
                    f"{status} Server Error: {response.reason}", response=response
                )
                if retries_left == 0:
                    # --- Idea 8: Enhanced Logging ---
                    logger.error(
                        f"{api_description}: Failed after {max_retries} attempts (Final Status {status}). Response Snippet: {response.text[:500] if response else 'N/A'}"
                    )
                    # --- End Idea 8 ---
                    if status == 429 and session_manager.dynamic_rate_limiter:
                        session_manager.dynamic_rate_limiter.increase_delay()
                    # Do not raise here, return None to indicate failure to caller
                    return None
                else:
                    base_sleep = initial_delay * (backoff_factor ** (attempt - 1))
                    jitter = random.uniform(-0.5, 0.5) * max(0.1, min(base_sleep, 1.0))
                    sleep_time = min(base_sleep + jitter, max_delay)
                    sleep_time = max(0.1, sleep_time)
                    logger.warning(
                        f"{api_description}: Received status {status} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s..."
                    )
                    if status == 429 and session_manager.dynamic_rate_limiter:
                        session_manager.dynamic_rate_limiter.increase_delay()
                    time.sleep(sleep_time)
                    continue  # Try again

            # Check for success (2xx)
            elif response.ok:
                if session_manager.dynamic_rate_limiter:
                    session_manager.dynamic_rate_limiter.decrease_delay()

                # --- Text parsing for specific API ---
                # Check if this specific call needs plain text
                force_text_parsing = api_description == "Get Ladder API (Batch)"
                if force_text_parsing:
                    return response.text

                # --- Standard JSON / Text Handling ---
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        logger.warning(
                            f"{api_description}: Response OK ({status}), but failed JSON decode."
                        )
                        logger.debug(f"Response text: {response.text[:500]}")
                        return response.text  # Fallback to text
                else:
                    # Handle plain text CSRF token response
                    if (
                        api_description == "CSRF Token API"
                        and "text/plain" in content_type
                    ):
                        return response.text.strip()
                    return response.text  # Return text for other non-JSON types
                # Successful processing exits the loop and function here

            # Handle non-retryable client/server errors (>= 400 and not in retry_status_codes)
            else:
                # --- Idea 8: Enhanced Logging ---
                logger.error(
                    f"{api_description}: Non-retryable error status: {status} {response.reason}. Response Snippet: {response.text[:500]}"
                )
                # --- End Idea 8 ---
                if status in [401, 403]:
                    logger.warning(
                        f"{api_description}: Authentication/Authorization error ({status}) received."
                    )
                    session_manager.api_login_verified = False
                # Do not raise here, return None
                return None

        # Handle Network/Timeout errors specifically
        except requests.exceptions.RequestException as e:
            retries_left -= 1
            last_exception = e
            if retries_left == 0:
                # --- Idea 8: Enhanced Logging ---
                logger.error(
                    f"{api_description}: Network/Timeout error failed after {max_retries} attempts. Final Error: {e}. Status Code (if available): {response.status_code if response else 'N/A'}",
                    exc_info=False,
                )
                # --- End Idea 8 ---
                # Do not raise here, return None
                return None
            else:
                base_sleep = initial_delay * (backoff_factor ** (attempt - 1))
                jitter = random.uniform(-0.5, 0.5) * max(0.1, min(base_sleep, 1.0))
                sleep_time = min(base_sleep + jitter, max_delay)
                sleep_time = max(0.1, sleep_time)
                logger.warning(
                    f"{api_description}: Network/Timeout error (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s... Error: {e}"
                )
                time.sleep(sleep_time)
                continue  # Try again
        except Exception as e:  # Catch other unexpected errors during request attempt
            logger.critical(
                f"{api_description}: CRITICAL Unexpected error during requests attempt {attempt}: {e}",
                exc_info=True,
            )
            # Raise unexpected critical errors
            raise

    # Should only be reached if all retries failed due to retryable status codes or network errors
    logger.error(
        f"{api_description}: Exited retry loop after {max_retries} failed attempts. Returning None."
    )
    return None
# End of _api_req

def make_ube(driver):
    """EXACT copy from test.py"""
    try:
        cookies = {c["name"]: c["value"] for c in driver.get_cookies()}
        ancsessionid = cookies.get("ANCSESSIONID")

        ube_data = {
            "eventId": "00000000-0000-0000-0000-000000000000",
            "correlatedScreenViewedId": str(uuid.uuid4()),
            "correlatedSessionId": ancsessionid,
            "userConsent": "necessary|preference|performance|analytics1st|analytics3rd|advertising1st|advertising3rd|attribution3rd",
            "vendors": "adobemc",
            "vendorConfigurations": "{}",
        }
        return base64.b64encode(json.dumps(ube_data).encode()).decode()
    except Exception as e:
        logger.error(f"UBE Header Error: {str(e)}")
        return None
# End of make_ube

def make_newrelic(driver):  # Standalone function, takes driver
    """Generates the newrelic header value."""
    try:
        newrelic_data = {
            "v": [0, 1],
            "d": {
                "ty": "Browser",
                "ac": "1690570",
                "ap": "1588726612",
                "id": str(uuid.uuid4()),
                "tr": str(uuid.uuid4()),
                "ti": int(time.time() * 1000),
                "tk": "2611750",
            },
        }
        json_payload = json.dumps(newrelic_data, separators=(",", ":"))
        encoded_payload = base64.b64encode(json_payload.encode("utf-8")).decode("utf-8")
        return encoded_payload

    except Exception as e:
        logger.error(f"ERROR GENERATING NEWRELIC HEADER", exc_info=True)
        return None
# End of make_newrelic

def make_traceparent(driver):  # Standalone function, takes driver
    """Generates the traceparent header value."""
    try:
        trace_id = str(uuid.uuid4()).replace("-", "")
        parent_id = str(uuid.uuid4()).replace("-", "")
        traceparent = f"00-{trace_id}-{parent_id}-01"
        return traceparent

    except Exception as e:
        logger.error(f"ERROR GENERATING TRACEPARENT HEADER", exc_info=True)
        return None
# End of make_traceparent

def make_tracestate(driver):  # Standalone function, takes driver
    """Generates the tracestate header value."""
    try:
        tracestate = f"2611750@nr=0-1-1690570-1588726612-{str(uuid.uuid4())}----{int(time.time() * 1000)}"
        return tracestate

    except Exception as e:
        logger.error(f"ERROR GENERATING TRACESTATE HEADER", exc_info=True)
        return None
# End of make_tracestate

def get_driver_cookies(driver):
    """Retrieves cookies from the Selenium driver as a dictionary."""
    cookies_dict = {}
    for cookie in driver.get_cookies():
        cookies_dict[cookie["name"]] = cookie["value"]
    return cookies_dict
# End of get_driver_cookies

def make_cookie_jar(driver_cookies):
    """Creates a RequestsCookieJar from Selenium driver cookies."""
    cookie_jar = RequestsCookieJar()
    for cookie in driver_cookies:
        cookie_jar.set(
            cookie["name"],
            cookie["value"],
            domain=cookie["domain"],
            path=cookie["path"],
            secure=cookie["secure"],
            httpOnly=cookie["httpOnly"],
        )
    return cookie_jar
# End of make_cookie_jar


# ----------------------------------------------------------------------------
# Login
# ----------------------------------------------------------------------------

# Login 5
@time_wait("Handle 2FA Page")
def handle_twoFA(session_manager: SessionManager) -> bool:
    """Handles the two-step verification page, choosing SMS method and waiting for user input."""
    try:
        logger.debug("Handling Two-Factor Authentication (2FA)...")
        driver = session_manager.driver
        if not driver:
            logger.error("2FA handling failed: WebDriver is not available.")
            return False

        # Use wait factory methods from selenium_config
        element_wait = selenium_config.default_wait(driver)  # Default wait for elements
        page_wait = selenium_config.page_load_wait(driver)  # Page load wait
        short_wait = selenium_config.short_wait(driver)  # Short wait for quick checks
        long_wait = selenium_config.long_wait(driver)  # Long wait for user input

        # 1. Wait for 2FA page indicator to be present
        try:
            logger.debug("Waiting for 2FA page header...")
            element_wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                )
            )
            logger.debug("2FA page detected.")
        except TimeoutException:
            logger.warning(
                "Did not detect 2FA page header within timeout. Assuming 2FA not required or page didn't load."
            )
            if login_status(session_manager):
                logger.info(
                    "User appears logged in after checking for 2FA page. Proceeding."
                )
                return True
            return False  # Fail if 2FA page expected but not found

        # 2. Wait for SMS button and click it
        try:
            logger.debug("Waiting for 2FA 'Send Code' (SMS) button...")
            sms_button_present = element_wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR))
            )
            sms_button_clickable = short_wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR))
            )  # Use short wait for clickability

            if sms_button_clickable:
                logger.debug(
                    "Attempting to click 'Send Code' button using JavaScript..."
                )
                driver.execute_script("arguments[0].click();", sms_button_clickable)
                logger.debug("'Send Code' button clicked.")
                try:
                    # Wait for code input field briefly after click
                    short_wait.until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, TWO_FA_CODE_INPUT_SELECTOR)
                        )
                    )
                    logger.debug(
                        "Code input field appeared after clicking 'Send Code'."
                    )
                except TimeoutException:
                    logger.warning(
                        "Code input field did not appear quickly after clicking 'Send Code'."
                    )
            else:
                logger.error("'Send Code' button found but not clickable.")
                return False

        except TimeoutException:
            logger.error("Timeout finding or clicking the 2FA 'Send Code' button.")
            return False
        except Exception as e:
            logger.error(f"Error clicking 2FA 'Send Code' button: {e}", exc_info=True)
            return False

        # 3. Wait for user to enter 2FA code manually (using long_wait)
        code_entry_timeout_value = (
            selenium_config.TWO_FA_CODE_ENTRY_TIMEOUT
        )  # Get value for logging
        logger.warning(
            f"Waiting up to {code_entry_timeout_value}s for user to manually enter 2FA code and submit..."
        )

        start_time = time.time()
        logged_in = False
        while time.time() - start_time < code_entry_timeout_value:
            if login_status(session_manager):
                logged_in = True
                logger.info("User completed 2FA successfully (login confirmed).")
                break
            # Check if 2FA page is still present
            elif not is_elem_there(
                driver, By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR, wait=1
            ):
                logger.warning(
                    "2FA page disappeared, but login not confirmed yet. Waiting briefly..."
                )
                time.sleep(3)
                if login_status(session_manager):
                    logged_in = True
                    logger.info(
                        "User completed 2FA successfully (login confirmed after page change)."
                    )
                    break
                else:
                    logger.error("2FA page disappeared, but login failed.")
                    return False
            time.sleep(2)

        if not logged_in:
            logger.error(
                f"Timed out ({code_entry_timeout_value}s) waiting for user 2FA code entry."
            )
            return False

        return True

    except Exception as e:
        logger.error(f"Unexpected error during 2FA handling: {e}", exc_info=True)
        return False
# End of handle_twoFA

# Login 4
def enter_creds(driver):
    """Enters username and password into the login form."""
    element_wait = selenium_config.element_wait(driver)
    try:

        logger.debug("Entering Credentials and Signing In...")

        logger.debug("Waiting for username input field...")
        username_input = element_wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))
        )
        logger.debug("Username input field found.")
        logger.debug("Clicking username input field...")  # Added log
        username_input.click()
        logger.debug("Clearing username input field...")  # Added log
        username_input.clear()

        ancestry_username = config_instance.ANCESTRY_USERNAME
        logger.debug(f"Entering username: {ancestry_username}")
        if ancestry_username is None:
            raise ValueError("ANCESTRY_USERNAME environment variable is not set")
        username_input.send_keys(ancestry_username)
        logger.debug("Username entered.")

        logger.debug("Waiting for password input field...")
        password_input = element_wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))
        )
        logger.debug("Password input field found.")
        logger.debug("Clicking password input field...")  # Added log
        password_input.click()
        logger.debug("Clearing password input field...")  # Added log
        password_input.clear()
        ancestry_password = config_instance.ANCESTRY_PASSWORD
        logger.debug(f"Entering password: *******")
        if ancestry_password is None:
            raise ValueError("ANCESTRY_PASSWORD environment variable is not set")
        password_input.send_keys(ancestry_password)
        logger.debug("Password entered.")
        logger.debug(" Username and password entered successfully.")

        logger.debug("Waiting for sign in button to be clickable...")
        sign_in_button = element_wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))
        )
        logger.debug("'Sign in' button found.")
        logger.debug("Clicking 'Sign in' button...")
        sign_in_button.click()
        logger.debug("Sign in button clicked...")
        return True

    except TimeoutException:
        logger.error(" Username or password input field not found within timeout.")
        return False
    except NoSuchElementException:
        logger.error("Username or password input field not found.")
        return False
    except Exception as e:
        logger.error(
            f"Unknown error: {e}", exc_info=True
        )  # Keep ERROR logging with exc_info
        return False
# End of enter_creds

# Login 3
@retry()
def consent(driver):
    """Handles cookie consent modal by removing it if present, with enhanced logging."""
    try:
        logger.debug("Checking for cookie consent overlay")
        # Check for overlay
        overlay = driver.find_element(By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
        if overlay:
            logger.debug("Cookie consent overlay DETECTED.")
            logger.debug(
                "Attempting to remove cookie consent overlay using Javascript..."
            )
            driver.execute_script("arguments[0].remove();", overlay)
            logger.debug("Cookie consent overlay REMOVED via Javascript.")
        else:
            logger.debug("Cookie consent not found.")
        logger.debug("Exiting consent function - Success.")  # Log success exit
        return True
    except NoSuchElementException:
        logger.debug(
            "Cookie consent overlay element NOT FOUND.Assuming no consent needed."
        )
        return True
    except Exception as e:
        logger.error(
            f"Error handling cookie consent overlay: {e}", exc_info=True
        )  # More detailed error log
        return False
# End of consent

# Login 2
def log_in(session_manager):
    """Logs in to Ancestry.com, handling 2-step verification if needed."""
    driver = session_manager.driver
    element_wait = selenium_config.element_wait(driver)
    page_wait = selenium_config.page_wait(driver)

    try:
        # 1. Navigate to signin page
        logger.debug("Navigating to signin page...")
        driver.get(urljoin(config_instance.BASE_URL, "account/signin"))

        # 2. Wait for login page to load
        try:
            page_wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR)
                )
            )
        except TimeoutException:
            logger.error("Login page did not load properly.")
            return "LOGIN_FAILED"

        # 3. Handle cookie consent (banner overlay) - if present
        consent(driver)

        # 4. Enter login credentials and submit
        if enter_creds(driver):  # returns True if successful credentials entry
            logger.debug("Login credentials entered successfully.")
            time.sleep(3)

            # 5. Check for 2-step verification - and handle if needed
            if is_elem_there(
                driver, By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR, wait=3
            ):
                logger.info("Two-step verification page detected.")
                if handle_twoFA(session_manager):  # now passing session_manager
                    logger.info("Two-step verification handled successfully.")
                    return "LOGIN_SUCCEEDED"  # Indicate 2FA Success
                else:
                    logger.warning("Two-step verification handling failed.")
                    return (
                        "LOGIN_FAILED_2FA"  # Indicate 2FA Failure - explicit 2FA fail
                    )

            # 6. If no 2-step verification, check for successful login directly
            if login_status(
                session_manager
            ):  # Check logged-in status using UI element - and API
                logger.info("Login successful (no 2FA, direct login).")
                return "LOGIN_SUCCEEDED"  # Indicate Direct Login Success
            else:
                logger.warning(
                    "Login confirmation element not found after login (no 2FA)."
                )
                return "LOGIN_FAILED"  # Indicate Login Failed - no 2FA, but no login confirmation

        else:  # enter_creds returned False (credentials entry failed)
            logger.warning("Failed to enter login credentials.")
            return "LOGIN_CREDS_FAILED"  # Indicate Login Creds Failed

    except TimeoutException:
        logger.error("Timeout during login process.")
        return "LOGIN_TIMEOUT"  # Indicate Login Timeout - general timeout
    except Exception as e:
        logger.error(f"An unexpected error occurred during login: {e}")
        return "LOGIN_ERROR"  # Indicate Login Error - unexpected error
        return "LOGIN_FAILED"
# End of log_in

# Login 1 - REVISED: Prioritize API Check
def login_status(session_manager: SessionManager) -> Optional[bool]:
    """
    REVISED: Checks if the user appears logged in. Prioritizes API verification,
    falls back to UI element check if API fails.
    Returns True if likely logged in, False if likely not, None if critical error occurs.
    """
    logger.debug("Checking login status (API prioritized)...")
    api_check_result: Optional[bool] = None
    ui_check_result: Optional[bool] = None

    try:
        if not isinstance(session_manager, SessionManager):
            logger.error(f"Invalid argument: Expected SessionManager, got {type(session_manager)}.")
            return None

        # --- 1. Basic WebDriver Session Check ---
        if not session_manager.is_sess_valid():
            logger.debug("Login status: Session invalid or browser closed.")
            return False # Session definitely not valid/logged in

        driver = session_manager.driver

        # --- 2. API-Based Login Verification (PRIORITY) ---
        logger.debug("Attempting API login verification...")
        # _verify_api_login_status returns True on success, False on failure/error
        api_check_result = session_manager._verify_api_login_status()

        if api_check_result is True:
            logger.debug("Login status confirmed via API check.")
            # Even if API is OK, quickly check UI isn't showing explicit logout state
            # This is a safety net against weird states.
            try:
                 if is_elem_there(driver, By.CSS_SELECTOR, LOG_IN_BUTTON_SELECTOR, wait=1):
                      logger.warning("API login check passed, but UI shows 'Log In' button. State mismatch?")
                      # In this conflict, trust the API less? Or UI less? Let's trust API but warn.
                      # Return True based on API, but log the discrepancy.
                      return True
                 else:
                      # API ok, UI doesn't show login button -> Confidently logged in
                      return True
            except Exception as ui_safety_check_e:
                 logger.warning(f"Error during UI safety check after successful API check: {ui_safety_check_e}")
                 # Proceed returning True based on API check despite safety check error
                 return True

        elif api_check_result is False:
            logger.debug("API login verification failed. Falling back to UI check.")
            # Proceed to UI check below
        else: # Should not happen if _verify_api_login_status returns True/False
            logger.error("API login verification returned unexpected value. Falling back to UI check.")
            # Proceed to UI check below

        # --- 3. UI-Based Login Verification (FALLBACK) ---
        logger.debug("Performing fallback UI login check...")

        # --- 3a. Handle Consent Overlay ---
        try:
            # Check presence first
            consent_overlay = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR))
            )
            if consent_overlay:
                logger.debug("Consent overlay found during UI fallback check. Attempting removal.")
                driver.execute_script("arguments[0].remove();", consent_overlay)
                logger.debug("Consent overlay removed.")
                time.sleep(0.5) # Brief pause after removal
        except TimeoutException:
            logger.debug("Consent overlay not found during UI fallback check.")
            pass # No overlay found, continue
        except Exception as consent_e:
            logger.warning(f"Error checking/removing consent overlay in UI fallback: {consent_e}")
            # Continue anyway, but log the warning

        # --- 3b. Check for Logged-In UI Element ---
        logged_in_selector = CONFIRMED_LOGGED_IN_SELECTOR
        logger.debug(f"Attempting UI verification using selector:\n'{logged_in_selector}'")
        ui_element_present = is_elem_there(driver, By.CSS_SELECTOR, logged_in_selector, wait=5) # Slightly shorter wait for fallback

        if ui_element_present:
            logger.debug("Login status confirmed via fallback UI check.")
            # If UI shows logged in, but API failed, update API cache flag?
            # Maybe API failed temporarily. Let's update the flag.
            session_manager.api_login_verified = True
            return True
        else:
            logger.debug("Login confirmation UI element not found in fallback check.")
            # --- 3c. Check for Logged-Out UI Element ---
            login_button_selector = LOG_IN_BUTTON_SELECTOR
            login_button_present = is_elem_there(driver, By.CSS_SELECTOR, login_button_selector, wait=1) # Quick check

            if login_button_present:
                 logger.debug("Login status confirmed NOT logged in via fallback UI check ('Log In' button found).")
                 session_manager.api_login_verified = False # Ensure flag matches UI state
                 return False
            else:
                 # --- 3d. Handle Ambiguity ---
                 logger.warning("Login status ambiguous: API failed and neither confirmation nor login UI elements found.")
                 try:
                      current_url = driver.current_url
                      base_url_norm = config_instance.BASE_URL.rstrip('/')
                      # Check if on a page that usually requires login
                      if current_url.startswith(base_url_norm) or "/family-tree" in current_url or "/dna/origins" in current_url:
                           logger.warning("Login status ambiguous via UI, but on a likely post-login page. Assuming TRUE (cautiously).")
                           # Don't update api_login_verified flag here, as API check failed initially
                           return True # Cautiously return True based on URL context
                 except WebDriverException:
                      logger.warning("Could not get current URL during ambiguous login check.")

                 logger.warning("Login status defaulting to FALSE due to ambiguity (API failed, UI unclear).")
                 session_manager.api_login_verified = False # Set flag to false due to ambiguity
                 return False

    # --- Handle Exceptions ---
    except WebDriverException as e:
         logger.error(f"WebDriverException during login_status check: {e}")
         if session_manager and session_manager.driver:
              logger.warning("Closing potentially broken session due to WebDriverException.")
              session_manager.close_sess()
         return None # Return None for critical errors
    except Exception as e:
        logger.critical(f"CRITICAL Unexpected error during login_status check: {e}", exc_info=True)
        return None # Return None for critical errors
# End of login_status


# ------------------------------------------------------------------------------------
# browser
# ------------------------------------------------------------------------------------

def is_browser_open(driver):
    """
    Checks if the browser window is open or closed.

    A simple interaction with a closed browser will raise an exception letting us know it is probably closed
    """
    try:
        time.sleep(0.5)
        # Getting the title is a quick check
        driver.title
        return True  # If no exception, browser is open
    except Exception:
        return False  # Browser is closed or session is invalid
# End of is_browser_open

def restore_sess(driver: WebDriver):
    """Restores session state including cookies, local, and session storage, domain aware."""
    if not driver:
        logger.error("Cannot restore session: WebDriver is None.")
        return False

    cache_dir = config_instance.CACHE_DIR  # Path object
    try:
        current_url = driver.current_url
        domain = urlparse(current_url).netloc.replace("www.", "")
        domain = domain.split(":")[0]  # Remove port
    except Exception as e:
        logger.error(
            f"Could not parse current URL for session restore: {e}. Using fallback."
        )
        try:
            domain = (
                urlparse(config_instance.BASE_URL)
                .netloc.replace("www.", "")
                .split(":")[0]
            )
            logger.warning(
                f"Falling back to base URL domain for session restore: {domain}"
            )
        except Exception:
            logger.error(
                "Could not determine domain from base URL either. Cannot restore."
            )
            return False

    logger.debug(
        f"Attempting to restore session state from cache for domain: {domain}..."
    )
    cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure cache dir exists

    restored_something = False

    def safe_json_read(
        filename: str,
    ) -> Optional[Any]:  # Filename relative to cache_dir
        """Helper function for safe JSON reading."""
        filepath = cache_dir / filename  # Use Path operator
        if filepath.exists():
            try:
                with filepath.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Skipping restore from '{filepath}': {e}")
                return None
            except Exception as e:
                logger.error(
                    f"Unexpected error reading '{filepath}': {e}", exc_info=True
                )
                return None
        else:
            # logger.debug(f"Cache file not found: {filepath}") # Optional: Less verbose
            return None

    # Restore Cookies
    cookies_file = f"session_cookies_{domain}.json"
    cookies = safe_json_read(cookies_file)
    if cookies and isinstance(cookies, list):
        try:
            logger.debug(f"Adding {len(cookies)} cookies from cache...")
            driver.delete_all_cookies()  # Clear first
            count = 0
            for cookie in cookies:
                if (
                    isinstance(cookie, dict)
                    and "name" in cookie
                    and "value" in cookie
                    and "domain" in cookie
                ):
                    try:
                        # Basic domain validation might be useful here if needed
                        driver.add_cookie(cookie)
                        count += 1
                    except WebDriverException as e:
                        logger.warning(
                            f"Skipping cookie '{cookie.get('name', '??')}': {e}"
                        )
            logger.debug(f"Added {count} cookies.")
            restored_something = True
        except WebDriverException as e:
            logger.error(f"WebDriver error during cookie restore: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during cookie restore: {e}", exc_info=True)
    elif cookies is not None:
        logger.warning(
            f"Cookie cache file '{cache_dir / cookies_file}' invalid format."
        )

    # Restore Local Storage
    local_storage_file = f"session_local_storage_{domain}.json"
    local_storage = safe_json_read(local_storage_file)
    if local_storage and isinstance(local_storage, dict):
        try:
            logger.debug(f"Restoring {len(local_storage)} items into localStorage...")
            script = """
                 var data = arguments[0]; localStorage.clear(); /* Clear first */
                 for (var key in data) { if (data.hasOwnProperty(key)) {
                     try { localStorage.setItem(key, data[key]); } catch (e) { console.warn('Failed localStorage setItem:', key, e); }
                 }} return localStorage.length;
             """
            driver.execute_script(script, local_storage)
            logger.debug("localStorage restored.")
            restored_something = True
        except WebDriverException as e:
            logger.error(f"Error restoring localStorage: {e}")
        except Exception as e:
            logger.error(f"Unexpected error restoring localStorage: {e}", exc_info=True)
    elif local_storage is not None:
        logger.warning(
            f"localStorage cache file '{cache_dir / local_storage_file}' invalid format."
        )

    # Restore Session Storage (Similar logic to localStorage)
    session_storage_file = f"session_session_storage_{domain}.json"
    session_storage = safe_json_read(session_storage_file)
    if session_storage and isinstance(session_storage, dict):
        try:
            logger.debug(
                f"Restoring {len(session_storage)} items into sessionStorage..."
            )
            script = """
                 var data = arguments[0]; sessionStorage.clear(); /* Clear first */
                 for (var key in data) { if (data.hasOwnProperty(key)) {
                     try { sessionStorage.setItem(key, data[key]); } catch (e) { console.warn('Failed sessionStorage setItem:', key, e); }
                 }} return sessionStorage.length;
             """
            driver.execute_script(script, session_storage)
            logger.debug("sessionStorage restored.")
            restored_something = True
        except WebDriverException as e:
            logger.error(f"Error restoring sessionStorage: {e}")
        except Exception as e:
            logger.error(
                f"Unexpected error restoring sessionStorage: {e}", exc_info=True
            )
    elif session_storage is not None:
        logger.warning(
            f"sessionStorage cache file '{cache_dir / session_storage_file}' invalid format."
        )

    # Perform hard refresh if anything was restored
    if restored_something:
        logger.info(
            "Session state restore attempt finished. Performing hard refresh..."
        )
        try:
            driver.refresh()
            # Wait for page load completion after refresh
            page_load_wait = selenium_config.page_load_wait(
                driver
            )  # Use factory method
            page_load_wait.until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            logger.debug("Hard refresh completed.")
        except TimeoutException:
            logger.warning("Timeout waiting for page load after hard refresh.")
        except WebDriverException as e:
            logger.warning(f"Error during hard refresh: {e}")
        return True
    else:
        logger.debug("No session state found in cache to restore.")
        return False
# end of restore_sess

def save_state(driver: WebDriver):
    """Saves session state (cookies, local/session storage) to domain-specific files."""
    if not is_browser_open(driver):  # Use refined check
        logger.warning("Browser session invalid/closed. Skipping session state save.")
        return

    cache_dir = config_instance.CACHE_DIR  # Path object
    try:
        current_url = driver.current_url
        domain = urlparse(current_url).netloc.replace("www.", "").split(":")[0]
    except Exception as e:
        logger.error(
            f"Could not parse current URL for saving state: {e}. Using fallback."
        )
        try:
            domain = (
                urlparse(config_instance.BASE_URL)
                .netloc.replace("www.", "")
                .split(":")[0]
            )
            logger.warning(
                f"Falling back to base URL domain for saving state: {domain}"
            )
        except Exception:
            logger.error(
                "Could not determine domain from base URL either. Cannot save state."
            )
            return

    logger.debug(f"Saving session state for domain: {domain}")
    cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure cache dir exists

    def safe_json_write(
        data: Any, filename: str
    ) -> bool:  # Filename relative to cache_dir
        """Helper function for safe JSON writing."""
        filepath = cache_dir / filename  # Use Path operator
        try:
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)  # Use smaller indent
            return True
        except (TypeError, IOError) as e:
            logger.error(f"Error writing JSON to '{filepath}': {e}")
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error writing JSON to '{filepath}': {e}", exc_info=True
            )
            return False

    # Save Cookies
    cookies_file = f"session_cookies_{domain}.json"
    try:
        cookies = driver.get_cookies()
        if cookies is not None:  # Check if cookies were retrieved
            if safe_json_write(cookies, cookies_file):
                logger.debug(f"Cookies saved to: {cache_dir / cookies_file}")
        else:
            logger.warning("Could not retrieve cookies from driver.")
    except WebDriverException as e:
        logger.error(f"Error getting cookies: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving cookies: {e}", exc_info=True)

    # Save Local Storage
    local_storage_file = f"session_local_storage_{domain}.json"
    try:
        # Ensure localStorage exists before trying to spread it
        local_storage = driver.execute_script(
            "return window.localStorage ? {...window.localStorage} : null;"
        )
        if local_storage is not None:
            if safe_json_write(local_storage, local_storage_file):
                logger.debug(f"localStorage saved to: {cache_dir / local_storage_file}")
        else:
            logger.debug("localStorage not available or empty, nothing saved.")
    except WebDriverException as e:
        logger.error(f"Error getting localStorage: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving localStorage: {e}", exc_info=True)

    # Save Session Storage
    session_storage_file = f"session_session_storage_{domain}.json"
    try:
        # Ensure sessionStorage exists
        session_storage = driver.execute_script(
            "return window.sessionStorage ? {...window.sessionStorage} : null;"
        )
        if session_storage is not None:
            if safe_json_write(session_storage, session_storage_file):
                logger.debug(
                    f"sessionStorage saved to: {cache_dir / session_storage_file}"
                )
        else:
            logger.debug("sessionStorage not available or empty, nothing saved.")
    except WebDriverException as e:
        logger.error(f"Error getting sessionStorage: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving sessionStorage: {e}", exc_info=True)

    logger.debug(f"Session state save attempt finished for domain: {domain}.")
# End of save_state

def close_tabs(driver):
    """Closes all but the first tab in the given driver."""
    logger.debug("Closing extra tabs...")
    try:
        while len(driver.window_handles) > 1:
            driver.switch_to.window(driver.window_handles[-1])
            driver.close()
            logger.debug(f"Closed a tab. Remaining handles: {driver.window_handles}")
        driver.switch_to.window(
            driver.window_handles[0]
        )  # Switch back to the first tab
        logger.debug("Switched back to the original tab.")
    except NoSuchWindowException:
        logger.warning("Attempted to close or switch to a tab that no longer exists.")
    except Exception as e:
        logger.error(f"Error in close_tabs: {e}", exc_info=True)
# end close_tabs


# ------------------------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------------------------

def nav_to_page(
    driver: WebDriver,
    url: str = config_instance.BASE_URL,
    selector: str = "main",
    session_manager: Optional[SessionManager] = None,
) -> bool:
    """
    Navigates to a URL, handles temporary unavailability, login redirection,
    and verifies successful navigation by checking URL and waiting for a selector.
    """
    if not driver:
        logger.error("Navigation failed: WebDriver instance is None.")
        return False

    # Use configured retry and timeout values
    max_attempts = config_instance.MAX_RETRIES
    page_timeout = selenium_config.PAGE_TIMEOUT  # For overall page load readiness
    element_timeout = selenium_config.ELEMENT_TIMEOUT  # For specific elements

    target_url_base = url.split("?")[
        0
    ]  # Base URL for comparison, ignoring query params

    # Define selectors/messages indicating page unavailability or errors
    unavailability_selectors = {
        TEMP_UNAVAILABLE_SELECTOR: ("refresh", 5),  # Temp unavailable message
        PAGE_NO_LONGER_AVAILABLE_SELECTOR: ("skip", 0),  # Page gone message
        # Add other known error selectors/messages here
        # e.g., "div.error-page-container": ('refresh', 3),
    }

    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Try {attempt}/{max_attempts} navigating to:\n{url}")
        is_blocked = False  # Flag for temporary blocks like unavailability

        try:
            # --- 1. Pre-Navigation Checks (Login/MFA/Current URL) ---
            try:
                current_url = driver.current_url
            except WebDriverException as e:
                logger.error(
                    f"Failed to get current URL (attempt {attempt}): {e}. Session might be dead."
                )
                # Try to restart if session manager available
                if session_manager:
                    logger.warning(
                        "Attempting session restart due to failure getting current URL."
                    )
                    if session_manager.restart_sess():
                        logger.info(
                            "Session restarted. Retrying navigation from scratch..."
                        )
                        driver = session_manager.driver  # Get the new driver instance
                        if not driver:
                            return False  # Restart failed to provide driver
                        continue  # Restart loop with new driver
                    else:
                        logger.error("Session restart failed. Aborting navigation.")
                        return False
                else:
                    return False  # Cannot proceed without driver

            login_url_base = urljoin(config_instance.BASE_URL, "account/signin")
            mfa_url_base = urljoin(config_instance.BASE_URL, "account/signin/mfa/")

            # Handle MFA page if encountered unexpectedly
            if current_url.startswith(mfa_url_base):
                logger.warning(
                    "Navigation blocked: Currently on MFA page. Requires user interaction."
                )
                # Fail the navigation attempt; MFA needs manual completion.
                return False

            # Handle Login page redirection
            if current_url.startswith(login_url_base):
                logger.warning("Redirected to login page. Attempting re-login...")
                if session_manager:
                    login_result = log_in(session_manager)
                    if login_result == "LOGIN_SUCCEEDED":
                        logger.info(
                            "Re-login successful. Retrying original navigation..."
                        )
                        # After re-login, continue to the next loop iteration to attempt driver.get(url) again
                        continue
                    else:
                        logger.error(
                            f"Re-login failed ({login_result}). Aborting navigation."
                        )
                        return False
                else:
                    # Cannot re-login without session manager
                    logger.error(
                        "Redirected to login, but no SessionManager provided. Cannot re-login."
                    )
                    return False

            # --- 2. Perform Navigation ---
            driver.get(url)

            # --- 3. Post-Navigation Verification ---
            # Wait briefly for potential redirects or initial loading
            time.sleep(random.uniform(0.8, 1.8))  # Slightly longer random pause

            # Verify URL after navigation
            try:
                post_nav_url = driver.current_url
                post_nav_url_base = post_nav_url.split("?")[0]
                logger.debug(f"Try {attempt} got to:\n{post_nav_url}")
            except WebDriverException as e:
                logger.error(
                    f"Failed to get URL after get() (attempt {attempt}): {e}. Retrying."
                )
                continue  # Retry the navigation attempt

            # Check again for redirection to Login/MFA immediately after get()
            if post_nav_url.startswith(mfa_url_base):
                logger.warning(
                    "Redirected to MFA page immediately after navigation attempt."
                )
                return False  # Fail if MFA appears
            if post_nav_url.startswith(login_url_base):
                logger.warning(
                    "Redirected back to login page immediately after navigation attempt."
                )
                continue  # Let next loop attempt handle re-login

            # Check if landed on the target URL (ignoring query params, allowing trailing slash variations)
            if post_nav_url_base.rstrip("/") != target_url_base.rstrip("/"):
                logger.warning(
                    f"Navigation landed on unexpected URL: {post_nav_url} (Expected base: {target_url_base})"
                )
                # Check for error messages on this unexpected page
                for msg_selector, (action, _) in unavailability_selectors.items():
                    if is_elem_there(driver, By.CSS_SELECTOR, msg_selector, wait=1):
                        logger.error(
                            f"Landed on wrong URL ({post_nav_url}) with unavailability message ('{msg_selector}')."
                        )
                        if action == "skip":
                            return False  # Permanent failure
                        is_blocked = True  # Mark as temporarily blocked
                        break  # Exit message check loop
                if is_blocked:
                    logger.warning(
                        "Waiting before retrying due to unavailability message on wrong URL."
                    )
                    time.sleep(
                        unavailability_selectors[msg_selector][1]
                    )  # Wait specified time
                    continue  # Retry navigation attempt
                else:
                    # No specific error message, but wrong URL - treat as failure for this attempt
                    logger.warning(
                        "Unexpected URL and no error message. Retrying navigation."
                    )
                    continue

            # --- 4. Wait for Target Selector ---
            # Determine selector to wait for (use 'body' for APIs or if selector is empty)
            wait_selector = selector if selector and "/api/" not in url else "body"

            logger.debug(
                f"Allow upto {element_timeout}s looking for:\n'{wait_selector}'"
            )
            try:
                # Use visibility_of_element_located for interactive pages
                WebDriverWait(driver, element_timeout).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, wait_selector))
                )
                logger.debug(f"Navigation successful.")
                return True  # SUCCESS! Navigation completed successfully.

            except TimeoutException:
                # Timeout occurred waiting for the target selector
                current_url_on_timeout = driver.current_url
                logger.warning(
                    f"Timeout waiting for selector '{wait_selector}' at {current_url_on_timeout}."
                )

                # Check for unavailability messages specifically *after* the timeout
                for msg_selector, (
                    action,
                    wait_time,
                ) in unavailability_selectors.items():
                    if is_elem_there(driver, By.CSS_SELECTOR, msg_selector, wait=1):
                        is_blocked = True
                        if action == "skip":
                            logger.error(
                                f"Page unavailable ('{msg_selector}' found after timeout). Aborting navigation."
                            )
                            return False  # Permanent failure
                        else:  # refresh
                            logger.warning(
                                f"Page temporarily unavailable ('{msg_selector}' found after timeout). Waiting {wait_time}s before retry."
                            )
                            time.sleep(wait_time)
                            break  # Exit check loop, continue to next navigation attempt

                if is_blocked:
                    continue  # Go to next attempt

                # If no specific message found, log generic timeout and retry
                logger.warning(
                    f"Timeout waiting for selector '{wait_selector}', no specific unavailability message found. Retrying navigation."
                )
                continue  # Go to next attempt

        # --- Handle Exceptions During the Attempt ---
        except UnexpectedAlertPresentException as alert_e:
            logger.warning(
                f"Unexpected alert during navigation attempt {attempt}: {alert_e.alert_text}"
            )
            try:
                driver.switch_to.alert.accept()
                logger.info("Alert accepted.")
            except Exception as accept_e:
                logger.error(f"Failed to accept alert: {accept_e}", exc_info=True)
                return False  # Fail if alert cannot be handled
            continue  # Retry navigation after handling alert

        except WebDriverException as wd_e:
            logger.error(
                f"WebDriverException during navigation attempt {attempt}: {wd_e}",
                exc_info=True,
            )
            # Check if session died and try to restart if possible
            if session_manager and not session_manager.is_sess_valid():
                logger.error("WebDriver session appears invalid. Attempting restart...")
                if session_manager.restart_sess():
                    logger.info(
                        "Session restarted. Retrying navigation from scratch..."
                    )
                    driver = session_manager.driver  # Update driver reference
                    if not driver:
                        return False  # Restart failed
                    continue  # Retry loop with new driver
                else:
                    logger.error("Session restart failed. Aborting navigation.")
                    return False
            else:  # Session seems valid, maybe temporary glitch
                logger.warning(
                    "WebDriverException occurred, session seems valid. Waiting before retry."
                )
                time.sleep(3)
                continue  # Retry the attempt

        except Exception as e:
            # Catch-all for other unexpected errors
            logger.error(
                f"Unexpected error during navigation attempt {attempt}: {e}",
                exc_info=True,
            )
            time.sleep(3)  # Wait before retry
            continue  # Retry the attempt

    # --- End of Retry Loop ---
    logger.critical(
        f"Navigation to {url} failed permanently after {max_attempts} attempts."
    )  # Use CRITICAL
    try:
        final_url = driver.current_url
        logger.error(f"Final URL after failure: {final_url}")
    except Exception:
        logger.error("Could not retrieve final URL after navigation failure.")
    return False  # Return definitive failure
# End of nav_to_page


# ------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------

def main():
    """
    Enhanced main function for standalone database setup and comprehensive testing of utils.py.
    Tests session start, identifier retrieval, API requests, and tab management.
    """
    # Correct placement for standalone script imports
    from logging_config import setup_logging
    from config import (
        config_instance,
    )  # Already imported at top level, but fine here too

    # --- Setup Logging ---
    db_file_path = config_instance.DATABASE_FILE  # Path object
    # Pass only the filename to setup_logging
    log_filename_only = db_file_path.with_suffix(".log").name
    setup_logging(
        log_file=log_filename_only,  # Pass just 'ancestry.log' (or similar)
        log_level="DEBUG",  # Ensure DEBUG level for detailed test logs
    )
    # Log file path is now determined *inside* setup_logging
    logger.info(f"--- Starting utils.py standalone test run ---")
    # logger.info(f"Log file path set by logging_config.") # Path logged within setup_logging

    session_manager = SessionManager()
    test_success = True  # Flag to track overall test success

    try:
        # --- Test Session Start (includes identifier retrieval) ---
        logger.info("--- Testing SessionManager.start_sess() ---")
        start_ok, driver_instance = session_manager.start_sess(action_name="Utils Test")
        if not start_ok or not driver_instance:
            logger.error("SessionManager.start_sess() FAILED. Aborting further tests.")
            test_success = False
            # Use return instead of sys.exit for cleaner testing exit
            return
        else:
            logger.info("SessionManager.start_sess() PASSED.")

            # --- Verify Identifiers Retrieved (already done by start_sess success check) ---
            logger.info("--- Verifying Identifiers (Retrieved during start_sess) ---")
            errors = []
            if not session_manager.my_profile_id:
                errors.append("my_profile_id")
            if not session_manager.my_uuid:
                errors.append("my_uuid")
            # Check tree_id only if TREE_NAME is configured, as it's optional otherwise
            if config_instance.TREE_NAME and not session_manager.my_tree_id:
                errors.append("my_tree_id (required by config)")
            if not session_manager.csrf_token:
                errors.append("csrf_token")

            if errors:
                logger.error(
                    f"FAILED to retrieve required identifiers: {', '.join(errors)}"
                )
                test_success = False
                # Decide if failure to get identifiers should halt further tests
                # return # Optional: Halt if identifiers are critical for subsequent tests
            else:
                logger.info("All required identifiers retrieved successfully.")
                logger.debug(f"Profile ID: {session_manager.my_profile_id}")
                logger.debug(f"UUID: {session_manager.my_uuid}")
                logger.debug(
                    f"Tree ID: {session_manager.my_tree_id or 'N/A'}"
                )  # Show N/A if not retrieved
                logger.debug(f"Tree Owner: {session_manager.tree_owner_name or 'N/A'}")
                logger.debug(f"CSRF Token: {session_manager.csrf_token[:10]}...")

            # --- Test Navigation (Using nav_to_page to Base URL - basic test) ---
            logger.info("--- Testing Navigation (nav_to_page to BASE_URL) ---")
            # Test navigating to the base URL, waiting for the body tag
            nav_ok = nav_to_page(
                driver=driver_instance,
                url=config_instance.BASE_URL,
                selector="body",  # Basic selector for base URL
                session_manager=session_manager,
            )
            if nav_ok:
                logger.info("nav_to_page() to BASE_URL PASSED.")
                try:
                    current_url_after_nav = driver_instance.current_url
                    if current_url_after_nav.startswith(config_instance.BASE_URL):
                        logger.info(
                            f"Successfully landed on expected base URL: {current_url_after_nav}"
                        )
                    else:
                        # This might happen if redirected immediately (e.g., to a dashboard) - not necessarily an error
                        logger.warning(
                            f"nav_to_page() to base URL landed on slightly different URL: {current_url_after_nav}"
                        )
                except Exception as e:
                    logger.warning(f"Could not verify URL after nav_to_page: {e}")
            else:
                logger.error("nav_to_page() to BASE_URL FAILED.")
                test_success = False

            # --- Test API Request Helper (_api_req via CSRF endpoint) ---
            logger.info("--- Testing API Request (_api_req via CSRF endpoint) ---")
            csrf_url = urljoin(
                config_instance.BASE_URL, "discoveryui-matches/parents/api/csrfToken"
            )
            csrf_test_response = _api_req(
                url=csrf_url,
                driver=driver_instance,
                session_manager=session_manager,
                method="GET",
                use_csrf_token=False,
                api_description="CSRF Token API",
            )
            # Check if the response looks like a valid CSRF token response
            if csrf_test_response and (
                (
                    isinstance(csrf_test_response, dict)
                    and "csrfToken" in csrf_test_response
                )
                or (
                    isinstance(csrf_test_response, str) and len(csrf_test_response) > 10
                )
            ):
                logger.info("CSRF Token API call via _api_req PASSED.")
                token_val = (
                    csrf_test_response["csrfToken"]
                    if isinstance(csrf_test_response, dict)
                    else csrf_test_response
                )
                logger.debug(f"CSRF Token API test retrieved: {str(token_val)[:10]}...")
            else:
                logger.error("CSRF Token API call via _api_req FAILED.")
                logger.debug(f"Response received: {csrf_test_response}")
                test_success = False

            # --- Test Browser Tab Management (make_tab, close_tabs) ---
            logger.info("--- Testing Tab Management (make_tab, close_tabs) ---")
            logger.info("Creating a new tab...")
            new_tab_handle = session_manager.make_tab()
            if new_tab_handle:
                logger.info(f"make_tab() PASSED. New handle: {new_tab_handle}")
                logger.info("Navigating new tab to example.com...")
                try:
                    driver_instance.switch_to.window(new_tab_handle)
                    driver_instance.get("https://example.com")
                    # Add a small wait for the page to potentially load
                    WebDriverWait(driver_instance, 5).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                    logger.info("Navigation in new tab successful.")
                    logger.info("Closing extra tabs...")
                    # Ensure close_tabs is called with the correct driver instance
                    close_tabs(driver_instance)
                    if len(driver_instance.window_handles) == 1:
                        logger.info("close_tabs() PASSED (one tab remaining).")
                    else:
                        logger.error(
                            f"close_tabs() FAILED (expected 1 tab, found {len(driver_instance.window_handles)})."
                        )
                        test_success = False
                    # Switch back to original tab just to be safe for any subsequent steps
                    if driver_instance.window_handles:
                        driver_instance.switch_to.window(
                            driver_instance.window_handles[0]
                        )

                except Exception as tab_e:
                    logger.error(
                        f"Error during tab management test: {tab_e}", exc_info=True
                    )
                    test_success = False
            else:
                logger.error("make_tab() FAILED.")
                test_success = False

    except Exception as e:
        # Catch any broad exception during the test setup or execution
        logger.critical(
            f"CRITICAL error during utils.py standalone test execution: {e}",
            exc_info=True,
        )
        test_success = False

    finally:
        # Ensure cleanup happens regardless of test success/failure
        if "session_manager" in locals() and session_manager:
            logger.info("Closing session manager...")
            session_manager.close_sess()  # Closes WebDriver and DB pool

        print("")  # Spacer before final message
        if test_success:
            logger.info("--- Utils.py standalone test run PASSED ---")
        else:
            logger.error("--- Utils.py standalone test run FAILED ---")
# End of main

if __name__ == "__main__":
    main()
