# utils.py

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
from datetime import datetime, timezone # Added timezone back for naive conversion
from typing import Optional, Tuple, List, Dict, Any, Generator
from pathlib import Path
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
from requests import Response, Request # Removed RequestsCookieJar
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy import create_engine, event, text, pool as sqlalchemy_pool
from requests.exceptions import RequestException, HTTPError
from my_selectors import *
from config import config_instance, selenium_config
from database import Base, MessageType
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
        if attribute == "href" and value and value.startswith("/"): # Added check for value existence
            return urljoin(config_instance.BASE_URL, value) # Use urljoin for robustness
        return value if value else "" # Return empty string if attribute is empty/None
    except NoSuchElementException: # Catch only NoSuchElementException
        return ""
    except Exception as e: # Catch other potential errors
        logger.warning(f"Error extracting attribute '{attribute}' from selector '{selector}': {e}")
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
            lambda d: (
                elem if (elem := d.find_elements(By.CSS_SELECTOR, PAGINATION_SELECTOR)) and elem[0].get_attribute("total") is not None else None
            )
        )

        if pagination_element is None:
            logger.debug("Pagination element or 'total' attribute not found/available. Assuming 1 page.")
            return 1

        total_pages = int(pagination_element[0].get_attribute("total"))
        return total_pages

    except TimeoutException:
        logger.debug(f"Timeout waiting for pagination element or 'total' attribute. Assuming 1 page.")
        return 1
    except (NoSuchElementException, IndexError): # Handle cases where element disappears or list is empty
         logger.debug("Pagination element not found after initial wait. Assuming 1 page.")
         return 1
    except (ValueError, TypeError) as e:
         logger.error(f"Error converting total pages attribute to int: {e}")
         return 1
    except Exception as e:
        logger.error(f"Error getting total pages: {e}", exc_info=True) # Log full traceback for unexpected
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

def is_elem_there(driver, by, value, wait=None): # Use None default for wait
    """
    Checks if an element is present within a specified timeout.
    Uses selenium_config.ELEMENT_TIMEOUT if wait is None.
    """
    if wait is None:
        wait = selenium_config.ELEMENT_TIMEOUT # Use configured default

    try:
        WebDriverWait(driver, wait).until(EC.presence_of_element_located((by, value)))
        return True
    except TimeoutException:
        # logger.warning(f"Timed out waiting {wait}s for: {value}") # Optional: Less verbose logging
        return False
    except Exception as e: # Catch other potential errors
         logger.error(f"Error checking element presence for '{value}': {e}")
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
            if part.isupper() and len(part) > 1: # Keep multi-letter caps
                 formatted_parts.append(part)
            elif part.isupper() and len(part) == 1: # Keep single initial caps
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
            backoff = config_instance.BACKOFF_FACTOR if BACKOFF_FACTOR is None else BACKOFF_FACTOR
            max_d = config_instance.MAX_DELAY if MAX_DELAY is None else MAX_DELAY

            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == attempts - 1:
                        logger.error(
                            f"All attempts failed for function '{func.__name__}'."
                        )
                        raise
                    # Correct backoff calculation
                    sleep_time = min(backoff * (2**i), max_d) + random.uniform(0, 1)
                    logger.warning(
                        f"Attempt {i + 1}/{attempts} for function '{func.__name__}' failed: {e}. Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
            # This part should ideally not be reached if attempts > 0,
            # as the loop either returns or raises.
            # Return None or raise a custom exception if preferred after all retries.
            logger.error(f"Function '{func.__name__}' failed after all retries.")
            return None # Or raise MaxRetriesExceededError("...")
        return wrapper
    return decorator
# end retry

def retry_api(
    max_retries=None, # Use None to default to config
    initial_delay=None, # Use None to default to config
    backoff_factor=None, # Use None to default to config
    retry_on_exceptions=(requests.exceptions.RequestException,), # Keep default tuple
    retry_on_status_codes=None, # Use None to default to config
):
    """
    Decorator for retrying API calls with exponential backoff, using logger.
    Optionally retries on specific HTTP status codes OR exceptions.
    Uses defaults from config_instance if arguments are None.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # --- Get defaults from config_instance if args are None ---
            _max_retries = config_instance.MAX_RETRIES if max_retries is None else max_retries
            _initial_delay = config_instance.INITIAL_DELAY if initial_delay is None else initial_delay # Use INITIAL_DELAY from config
            _backoff_factor = config_instance.BACKOFF_FACTOR if backoff_factor is None else backoff_factor
            _retry_codes = config_instance.RETRY_STATUS_CODES if retry_on_status_codes is None else retry_on_status_codes
            # Ensure _retry_codes is a set/tuple for efficient lookup
            if _retry_codes and not isinstance(_retry_codes, (set, tuple)):
                 _retry_codes = set(_retry_codes)
            # --- End Default Handling ---

            retries = _max_retries
            delay = _initial_delay
            attempt = 0

            while retries > 0:
                attempt += 1
                try:
                    response = func(*args, **kwargs)  # Capture the response

                    # Check status code retry condition FIRST
                    should_retry_status = False
                    if (
                        _retry_codes and
                        response is not None and
                        hasattr(response, 'status_code') and # Check before accessing
                        response.status_code in _retry_codes
                    ):
                         should_retry_status = True
                         status_code = response.status_code # Store for logging

                    if should_retry_status:
                        retries -= 1
                        if retries <= 0:
                            logger.error(
                                f"API Call failed after {_max_retries} retries for function '{func.__name__}' (Final Status {status_code}). No more retries.",
                                exc_info=False,
                            )
                            return response # Return last failed response for status code retry
                        else:
                            # Calculate sleep time with jitter
                            base_sleep = delay * (_backoff_factor ** (attempt - 1)) # Apply backoff correctly
                            jitter = random.uniform(-0.1, 0.1) * delay # Smaller jitter?
                            sleep_time = min(base_sleep + jitter, config_instance.MAX_DELAY) # Cap at MAX_DELAY
                            sleep_time = max(0.1, sleep_time) # Ensure minimum sleep

                            logger.warning(
                                f"API Call returned retryable status {status_code} (attempt {attempt}/{_max_retries}) for '{func.__name__}', retrying in {sleep_time:.2f} seconds..."
                            )
                            time.sleep(sleep_time)
                            delay *= _backoff_factor # Increase delay for next potential retry
                            continue # Go to next retry attempt

                    # If status code doesn't trigger retry, return the response
                    return response

                # Handle exception retry condition SECOND
                except retry_on_exceptions as e:
                    retries -= 1
                    if retries <= 0:
                        logger.error(
                            f"API Call failed after {_max_retries} retries due to exception for function '{func.__name__}'. Final Exception: {e}",
                            exc_info=True, # Include traceback for exceptions
                        )
                        raise e # Re-raise the last exception after all retries fail
                    else:
                        # Calculate sleep time with jitter
                        base_sleep = delay * (_backoff_factor ** (attempt - 1)) # Apply backoff correctly
                        jitter = random.uniform(-0.1, 0.1) * delay
                        sleep_time = min(base_sleep + jitter, config_instance.MAX_DELAY) # Cap at MAX_DELAY
                        sleep_time = max(0.1, sleep_time)

                        logger.warning(
                            f"API Call failed (attempt {attempt}/{_max_retries}) for '{func.__name__}', retrying in {sleep_time:.2f} seconds... Exception: {type(e).__name__} - {e}"
                        )
                    time.sleep(sleep_time)
                    delay *= _backoff_factor # Increase delay for next potential retry
                    continue # Go to next retry attempt

                # Non-retryable exceptions will naturally propagate out here

            # Should not be reached unless initial retries = 0, or loop logic error
            logger.error(f"Exited retry loop unexpectedly for '{func.__name__}'.")
            return None

        return wrapper
    return decorator
# End of retry_api

def ensure_browser_open(func):
    """Decorator to ensure the browser is open. Relies on SessionManager."""

    @wraps(func)
    def wrapper(session_manager, *args, **kwargs):
        # Check if session_manager is the first arg and is a SessionManager instance
        if not isinstance(session_manager, SessionManager):
             # Maybe the wrapped function has different signature, check args
             found_sm = None
             if args and isinstance(args[0], SessionManager):
                  found_sm = args[0]
             elif 'session_manager' in kwargs and isinstance(kwargs['session_manager'], SessionManager):
                  found_sm = kwargs['session_manager']

             if not found_sm:
                  raise TypeError(f"Function '{func.__name__}' requires a SessionManager instance as the first argument or kwarg.")
             session_manager = found_sm # Use the found instance

        # Now perform the browser check
        if not is_browser_open(session_manager.driver):
            raise WebDriverException( # Use a more specific exception
                f"Browser is not open or session invalid when calling function '{func.__name__}'"
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
                    f"Wait '{wait_description}' completed in {duration:.3f} s." # Added 'Wait' prefix and 'completed'
                )
                return result
            except TimeoutException as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.warning(
                    f"Wait '{wait_description}' timed out after {duration:.3f} seconds.",
                    exc_info=False, # Less verbose for timeout
                )
                raise e # Re-raise TimeoutException
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(
                    f"Error during wait '{wait_description}' after {duration:.3f} seconds: {e}",
                    exc_info=True,
                )  # ERROR level for unexpected errors
                raise e # Re-raise other exceptions

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
        config_instance=config_instance, # Pass config instance for defaults
    ):
        """
        Initializes the rate limiter using config_instance defaults if not provided.
        """
        self.initial_delay = config_instance.INITIAL_DELAY if initial_delay is None else initial_delay
        self.MAX_DELAY = config_instance.MAX_DELAY if MAX_DELAY is None else MAX_DELAY
        self.backoff_factor = config_instance.BACKOFF_FACTOR if backoff_factor is None else backoff_factor
        self.decrease_factor = config_instance.DECREASE_FACTOR if decrease_factor is None else decrease_factor
        self.current_delay = self.initial_delay
        self.last_throttled = False
        # Log initial values
        # logger.debug(f"RateLimiter init: Initial={self.initial_delay:.2f}, Max={self.MAX_DELAY:.2f}, Backoff={self.backoff_factor:.2f}, Decrease={self.decrease_factor:.2f}")
    # end of __int__

    def wait(self):
        """Wait for a dynamic delay based on rate-limiting conditions."""
        jitter = random.uniform(0.8, 1.2)  # ±20% jitter
        effective_delay = min(self.current_delay * jitter, self.MAX_DELAY)
        # Ensure delay is not negative or extremely small
        effective_delay = max(0.01, effective_delay)
        time.sleep(effective_delay)
        return effective_delay
    # End of wait

    def reset_delay(self):
        """Resets the delay to the initial value."""
        if self.current_delay != self.initial_delay: # Only log if changed
             self.current_delay = self.initial_delay
             logger.info(f"Rate limiter delay reset to initial value: {self.initial_delay:.2f}s")
    # End of reset_delay

    def decrease_delay(self):
        """Gradually reduce the delay when no rate limits are detected."""
        if not self.last_throttled and self.current_delay > self.initial_delay: # Check last_throttled flag
            previous_delay = self.current_delay
            self.current_delay = max(
                self.current_delay * self.decrease_factor, self.initial_delay
            )
            # Log only if delay actually changed significantly
            if abs(previous_delay - self.current_delay) > 0.01:
                logger.debug(
                    f"No rate limit detected. Decreased delay to {self.current_delay:.2f} seconds."
                )
        self.last_throttled = False # Reset flag after decrease attempt
    # End of decrease_delay

    def increase_delay(self):
        """Increase the delay exponentially when a rate limit is detected."""
        if not self.last_throttled: # Only increase if not already throttled in last cycle
            previous_delay = self.current_delay
            self.current_delay = min(
                self.current_delay * self.backoff_factor, self.MAX_DELAY
            )
            logger.info( # Use INFO for increase as it's significant
                f"Rate limit detected. Increased delay from {previous_delay:.2f}s to {self.current_delay:.2f} seconds."
            )
            self.last_throttled = True
        # else: logger.debug("Already throttled, delay remains high.")
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
        self.driver: Optional[WebDriver] = None # Type hint WebDriver
        self.db_path: str = str(config_instance.DATABASE_FILE.resolve())
        self.selenium_config = selenium_config
        self.ancestry_username: str = config_instance.ANCESTRY_USERNAME # Ensure string type
        self.ancestry_password: str = config_instance.ANCESTRY_PASSWORD # Ensure string type
        self.debug_port: int = self.selenium_config.DEBUG_PORT
        self.chrome_user_data_dir: Optional[Path] = self.selenium_config.CHROME_USER_DATA_DIR
        self.profile_dir: str = self.selenium_config.PROFILE_DIR # Ensure string type
        self.chrome_driver_path: Optional[Path] = self.selenium_config.CHROME_DRIVER_PATH
        self.chrome_browser_path: Optional[Path] = self.selenium_config.CHROME_BROWSER_PATH
        self.chrome_max_retries: int = self.selenium_config.CHROME_MAX_RETRIES
        self.chrome_retry_delay: int = self.selenium_config.CHROME_RETRY_DELAY
        self.headless_mode: bool = self.selenium_config.HEADLESS_MODE
        self.session_active: bool = False
        self.cache_dir: Path = config_instance.CACHE_DIR # Ensure Path type
        self.engine = None
        self.Session = None
        self._db_init_attempted = False
        logger.debug(f"SessionManager instance created: ID={id(self)}")
        self.last_js_error_check: datetime = datetime.now()
        self.dynamic_rate_limiter: DynamicRateLimiter = DynamicRateLimiter()
        self.csrf_token: Optional[str] = None
        self.api_login_verified: bool = False
        self.my_profile_id: Optional[str] = None
        self.my_uuid: Optional[str] = None
        self.my_tree_id: Optional[str] = None
        self.tree_owner_name: Optional[str] = None
        self.session_start_time: Optional[float] = None
        self._profile_id_logged = False
        self._uuid_logged = False
        self._tree_id_logged = False
        self._owner_logged = False
        self._requests_session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
            ),
        )
        self._requests_session.mount("http://", adapter)
        self._requests_session.mount("https://", adapter)
        logger.debug("Configured requests.Session with HTTPAdapter.")
        self.cache = self.Cache()
    # end of __init__

    def start_sess(self, action_name: Optional[str] = None) -> Tuple[bool, Optional[WebDriver]]:
        """
        Starts or reuses a Selenium WebDriver session following a specific sequence:
        Initialize -> WebDriver -> Navigate Base -> Login Check -> URL Check -> Cookies -> CSRF -> Identifiers -> Tree Owner -> Success.
        """
        session_start_success = False
        retry_count = 0
        max_retries = self.selenium_config.CHROME_MAX_RETRIES

        # --- 1. Initialize Session Start ---
        logger.debug(f"1. Session start initiated for action: '{action_name or 'Unknown'}'...")
        self.driver = None
        self.cache.clear()
        self.csrf_token = None
        self.my_profile_id = None
        self.my_uuid = None
        self.my_tree_id = None
        self.tree_owner_name = None
        self.api_login_verified = False # Reset verification flag

        # Initialize DB Engine/Session if needed
        if not self.engine or not self.Session:
            self._initialize_db_engine_and_session()
            if not self.engine or not self.Session:
                logger.critical("Failed to initialize DB Engine/Session during session start.")
                return False, None # Fail if DB can't be initialized

        # Ensure requests session exists
        if not hasattr(self, "_requests_session") or not isinstance(self._requests_session, requests.Session):
            self._requests_session = requests.Session()
            logger.debug("requests.Session initialized (fallback).")

        # --- Retry Loop for Session Initialization ---
        while retry_count < max_retries and not session_start_success:
            retry_count += 1
            logger.debug(f"Attempting session start: {retry_count}/{max_retries}\n")

            try:
                # --- 2. Initialize WebDriver ---
                logger.debug("2. Initializing WebDriver instance.")
                self.driver = init_webdvr(attach_attempt=True) # init_webdvr handles its own retries internally now
                if not self.driver:
                    logger.error(f"WebDriver initialization failed after internal retries (attempt {retry_count}).")
                    # No need to sleep/retry here, init_webdvr already did
                    # If init_webdvr fails after its retries, fail start_sess permanently
                    return False, None # Permanent failure if init_webdvr fails

                logger.debug("WebDriver initialization successful.")

                # --- Navigate to Base URL to Stabilize ---
                logger.debug("Navigating to Base URL to stabilize initial state...")
                base_url_nav_ok = nav_to_page(self.driver, config_instance.BASE_URL, selector="body", session_manager=self)
                if not base_url_nav_ok:
                    logger.error("Failed to navigate to Base URL after WebDriver init. Aborting attempt {retry_count}.")
                    self.close_sess() # Clean up driver
                    # Consider if retry is useful here or just fail the whole start_sess
                    if retry_count < max_retries:
                         time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                         continue # Retry the whole start_sess attempt
                    else:
                         return False, None # Fail permanently if base nav fails repeatedly

                logger.debug("Initial navigation to Base URL successful.\n")

                # --- 3. Check Login Status & Log In If Needed ---
                logger.debug("3. Checking login status.")
                login_stat = login_status(self) # Uses API check first, then UI
                if login_stat is True:
                    logger.debug("User is logged in.")
                elif login_stat is False:
                    logger.info("User not logged in. Attempting login process.") # Use INFO
                    login_result = log_in(self)
                    if login_result != "LOGIN_SUCCEEDED":
                        logger.critical(f"Login failed ({login_result}). Aborting session start.") # Use CRITICAL
                        self.close_sess()
                        return False, None # Fail permanently on login failure
                    logger.info("Login process successful.") # Use INFO
                    # Re-verify after login attempt
                    if not login_status(self): # Check again
                        logger.critical("Login status verification failed even after successful login attempt reported.") # Use CRITICAL
                        self.close_sess()
                        return False, None
                    logger.debug("Login status re-verified successfully after login.")
                else: # login_stat is None (critical error during check)
                    logger.critical(f"Login status check failed critically. Aborting session start.") # Use CRITICAL
                    self.close_sess()
                    return False, None
                logger.debug("Login status confirmed.\n")

                # --- 4. URL Check & Navigation to Base URL (Refined Logic) ---
                # This check might be less critical now that login_status is robust, but keep as safety net
                logger.debug("4. Re-checking current URL validity.")
                if not self._check_and_handle_url():
                     logger.error("URL check/handling failed. Aborting session start.")
                     self.close_sess()
                     return False, None
                logger.debug("URL re-check completed.\n")

                # --- 5. Verify Essential Cookies ---
                logger.debug("5. Verifying essential cookies (ANCSESSIONID, SecureATT).")
                essential_cookies = ["ANCSESSIONID", "SecureATT"]
                if not self.get_cookies(essential_cookies, timeout=15):
                    logger.error(f"Essential cookies {essential_cookies} not found. Aborting session start.")
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
                    logger.error("Failed to retrieve CSRF token. Aborting session start.")
                    self.close_sess()
                    return False, None
                logger.debug(f"CSRF token retrieved: {self.csrf_token[:10]}...\n")

                # --- 8. Retrieve User Identifiers ---
                logger.debug("8. Retrieving user identifiers.")
                if not self._retrieve_identifiers():
                     logger.error("Failed to retrieve essential user identifiers. Aborting session start.")
                     self.close_sess()
                     return False, None
                logger.debug("Finished retrieving user identifiers.\n")

                # --- 9. Retrieve Tree Owner Name ---
                logger.debug("9. Retrieving tree owner name.")
                self._retrieve_tree_owner() # Log internally
                logger.debug("Finished retrieving tree owner name.\n")


                # --- 10. Session Start Successful ---
                session_start_success = True
                self.session_active = True
                self.session_start_time = time.time()
                self.last_js_error_check = datetime.now()
                logger.debug(f"10. Session started successfully on attempt {retry_count}.") # Use INFO
                return True, self.driver

            # --- Handle Exceptions during the attempt ---
            except WebDriverException as wd_exc:
                logger.error(f"WebDriverException during session start attempt {retry_count}: {wd_exc}", exc_info=False)
                logger.debug("Closing potentially broken WebDriver instance for retry.")
                self.close_sess() # Clean up driver before retry
                if retry_count < max_retries:
                    logger.info(f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next attempt...")
                    time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                # else loop condition handles exit

            except Exception as e:
                logger.error(f"Unexpected error during session start attempt {retry_count}: {e}", exc_info=True)
                self.close_sess() # Clean up driver
                if retry_count < max_retries:
                    logger.info(f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next attempt...")
                    time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                # else loop condition handles exit


        # --- End of Retry Loop ---
        if not session_start_success:
            logger.critical(f"Session start FAILED permanently after {max_retries} attempts.")
            self.close_sess() # Ensure cleanup on permanent failure
            return False, None

        # Fallback (should not be reached)
        logger.error("start_sess exited retry loop unexpectedly without success/failure return.")
        self.close_sess()
        return False, None
    # end start_sess

    def _initialize_db_engine_and_session(self):
        """Initializes the SQLAlchemy engine and session factory with pooling."""
        # Log which SessionManager instance is doing the init
        logger.debug(f"SessionManager ID={id(self)} attempting DB initialization...")
        self._db_init_attempted = True # Mark that we tried

        # --- If an engine already exists, dispose it first ---
        # This handles potential re-initialization scenarios more cleanly
        if self.engine:
            logger.debug(f"SessionManager ID={id(self)} found existing engine (ID={id(self.engine)}). Disposing before re-initializing.")
            try:
                self.engine.dispose()
            except Exception as dispose_e:
                 logger.error(f"Error disposing existing engine: {dispose_e}")
            self.engine = None
            self.Session = None
        # --- End Dispose ---

        try:
            logger.debug(f"Initializing SQLAlchemy Engine for: {self.db_path}")

            # === POOL SIZE LOGIC V4 (kept from previous attempt) ===
            pool_size_env_str = os.getenv('DB_POOL_SIZE')
            pool_size_config = getattr(config_instance, 'DB_POOL_SIZE', None)
            logger.debug(f"Pool Size Check: Env='{pool_size_env_str}', Config='{pool_size_config}'")

            pool_size_str_to_parse = None
            if pool_size_env_str is not None:
                pool_size_str_to_parse = pool_size_env_str
                logger.debug(f"Pool Size Priority: Using Env value '{pool_size_str_to_parse}' for parsing.")
            elif pool_size_config is not None:
                try:
                     pool_size_str_to_parse = str(int(pool_size_config))
                     logger.debug(f"Pool Size Priority: Using Config value '{pool_size_str_to_parse}' for parsing.")
                except (ValueError, TypeError):
                     logger.warning(f"Pool Size Priority: Config value ('{pool_size_config}') invalid, falling back.")
                     pool_size_str_to_parse = '50'
            else:
                 logger.debug("Pool Size Priority: Neither Env nor Config found, using fallback '50'.")
                 pool_size_str_to_parse = '50'

            pool_size = 20
            try:
                parsed_val = int(pool_size_str_to_parse)
                if parsed_val <= 0:
                    logger.warning(f"DB_POOL_SIZE value '{parsed_val}' invalid (<=0). Using default {pool_size}.")
                elif parsed_val == 1:
                     logger.warning(f"DB_POOL_SIZE value '1' detected. Overriding to minimum default {pool_size} for safety.")
                else:
                    pool_size = min(parsed_val, 100)
                    logger.debug(f"Successfully parsed pool size: {pool_size}")
            except (ValueError, TypeError):
                logger.warning(f"Could not parse DB_POOL_SIZE value '{pool_size_str_to_parse}' as integer. Using default {pool_size}.")

            max_overflow = max(5, int(pool_size * 0.2))
            pool_timeout = 30
            pool_class = sqlalchemy_pool.QueuePool
            # === END POOL SIZE LOGIC ===

            # --- ABSOLUTE FINAL CHECK LOGGING ---
            final_params_log = f"FINAL PARAMS for create_engine: pool_size={pool_size}, max_overflow={max_overflow}, pool_timeout={pool_timeout}"
            logger.debug(f"*** SessionManager ID={id(self)} {final_params_log} ***") # Include SM ID
            # --- END FINAL CHECK ---

            # Create Engine
            self.engine = create_engine(
                f"sqlite:///{self.db_path}", echo=False,
                pool_size=pool_size, max_overflow=max_overflow,
                pool_timeout=pool_timeout, poolclass=pool_class,
                connect_args={"check_same_thread": False}
            )
            # Log the new engine ID
            logger.debug(f"SessionManager ID={id(self)} created NEW engine: ID={id(self.engine)}")

            try:
                actual_pool_size = getattr(self.engine.pool, '_size', 'N/A')
                logger.debug(f"Engine ID={id(self.engine)} pool size reported (internal): {actual_pool_size}")
            except Exception: logger.debug("Could not retrieve detailed engine pool status.")

            # --- PRAGMA listener ---
            @event.listens_for(self.engine, "connect")
            def enable_foreign_keys(dbapi_connection, connection_record):
                 cursor = dbapi_connection.cursor()
                 try:
                      cursor.execute("PRAGMA journal_mode=WAL;")
                      cursor.execute("PRAGMA foreign_keys=ON;")
                 except Exception as pragma_e: logger.error(f"Failed setting PRAGMA: {pragma_e}")
                 finally: cursor.close()

            # --- Session factory ---
            self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
            logger.debug(f"SessionManager ID={id(self)} created Session factory for Engine ID={id(self.engine)}")

            # --- Create tables ---
            try: Base.metadata.create_all(self.engine); logger.debug("DB tables checked/created.")
            except Exception as table_create_e: logger.error(f"Error creating DB tables: {table_create_e}", exc_info=True); raise table_create_e

        except Exception as e:
            logger.critical(f"SessionManager ID={id(self)} FAILED to initialize SQLAlchemy: {e}", exc_info=True)
            # Ensure state is clean on failure
            if self.engine:
                try: self.engine.dispose()
                except Exception: pass
            self.engine = None
            self.Session = None
            raise e # Re-raise
    #  end _initialize_db_engine_and_session

    def _check_and_handle_url(self) -> bool:
        """Checks current URL and navigates to base URL if necessary."""
        try:
            current_url = self.driver.current_url
            logger.debug(f"Current URL for check: {current_url}")
            base_url_norm = config_instance.BASE_URL.rstrip("/") + "/"
            signin_url_base = urljoin(config_instance.BASE_URL, "account/signin")
            logout_url_base = urljoin(config_instance.BASE_URL, "c/logout")
            mfa_url_base = urljoin(config_instance.BASE_URL, "account/signin/mfa/") # Should be handled by login_status

            disallowed_starts = (signin_url_base, logout_url_base, mfa_url_base) # Add MFA here just in case
            is_api_path = "/api/" in current_url

            needs_navigation = False
            if not current_url.startswith(config_instance.BASE_URL.rstrip('/')): # Looser check
                needs_navigation = True
                logger.debug("Reason: URL does not start with base URL.")
            elif any(current_url.startswith(path) for path in disallowed_starts):
                needs_navigation = True
                logger.debug(f"Reason: URL starts with disallowed path ({current_url}).")
            elif is_api_path:
                needs_navigation = True
                logger.debug("Reason: URL contains '/api/'.")

            if needs_navigation:
                logger.info(f"Current URL '{current_url}' unsuitable. Navigating to base URL: {base_url_norm}")
                if not nav_to_page(self.driver, base_url_norm, selector="body", session_manager=self):
                    logger.error("Failed to navigate to base URL during check.")
                    return False
                logger.debug("Navigation to base URL successful.")
            else:
                logger.debug("Current URL is suitable, no extra navigation needed.")
            return True

        except WebDriverException as e:
            logger.error(f"Error during URL check/navigation: {e}. Session might be dead.", exc_info=True)
            # Attempt to check if session is valid, if not, close might be needed
            if not self.is_sess_valid():
                 logger.warning("Session seems invalid during URL check.")
                 # self.close_sess() # Consider closing if invalid
            return False # Indicate failure
        except Exception as e:
             logger.error(f"Unexpected error during URL check: {e}", exc_info=True)
             return False
    # end _check_and_handle_url

    def _retrieve_identifiers(self) -> bool:
        """
        Helper to retrieve profile ID, UUID, and Tree ID.
        Logs identifiers at INFO level only the first time they are retrieved.
        """
        all_ok = True
        # Profile ID
        if not self.my_profile_id: # Only fetch if not already set
            self.my_profile_id = self.get_my_profileId()
            if not self.my_profile_id:
                logger.error("Failed to retrieve profile ID (ucdmid).")
                all_ok = False
            elif not self._profile_id_logged: # Log only first time
                 logger.info(f"My profile id: {self.my_profile_id}")
                 self._profile_id_logged = True
        elif not self._profile_id_logged: # Log if already set but not logged
            logger.info(f"My profile id: {self.my_profile_id}")
            self._profile_id_logged = True

        # UUID
        if not self.my_uuid:
            self.my_uuid = self.get_my_uuid()
            if not self.my_uuid:
                logger.error("Failed to retrieve UUID (testId).")
                all_ok = False
            elif not self._uuid_logged:
                 logger.info(f"My uuid: {self.my_uuid}")
                 self._uuid_logged = True
        elif not self._uuid_logged:
            logger.info(f"My uuid: {self.my_uuid}")
            self._uuid_logged = True

        # Tree ID
        if config_instance.TREE_NAME and not self.my_tree_id: # Only fetch if needed and not set
            self.my_tree_id = self.get_my_tree_id()
            if not self.my_tree_id:
                logger.error(f"TREE_NAME '{config_instance.TREE_NAME}' configured, but failed to get corresponding tree ID.")
                all_ok = False # Treat as error only if TREE_NAME is set
            elif not self._tree_id_logged:
                logger.info(f"My tree id: {self.my_tree_id}")
                self._tree_id_logged = True
        elif self.my_tree_id and not self._tree_id_logged: # Log if already set but not logged
            logger.info(f"My tree id: {self.my_tree_id}")
            self._tree_id_logged = True
        elif not config_instance.TREE_NAME:
             logger.debug("No TREE_NAME configured, skipping tree ID retrieval/logging.")


        return all_ok
    # end _retrieve_identifiers

    def _retrieve_tree_owner(self):
         """
         Helper to retrieve tree owner name.
         Logs owner name at INFO level only the first time it's retrieved.
         """
         if self.my_tree_id and not self.tree_owner_name: # Fetch only if tree ID exists and owner not yet known
              self.tree_owner_name = self.get_tree_owner(self.my_tree_id)
              if self.tree_owner_name and not self._owner_logged:
                   logger.info(f"Tree Owner Name: {self.tree_owner_name}\n")
                   self._owner_logged = True
              elif not self.tree_owner_name:
                   logger.warning("Failed to retrieve tree owner name.\n")
         elif self.tree_owner_name and not self._owner_logged: # Log if already known but not logged
              logger.info(f"Tree Owner Name: {self.tree_owner_name}\n")
              self._owner_logged = True
         elif not self.my_tree_id:
              logger.debug("Skipping tree owner retrieval (no tree ID).\n")
    # end _retrieve_tree_owner

    @retry_api() # Add retry decorator
    def get_csrf(self) -> Optional[str]:
        """
        Fetches the CSRF token using _api_req. Retries on failure.
        """
        # Check cache first - Simple internal cache within SessionManager
        # if self.csrf_token: # Use internal attribute directly
        #     logger.debug("Using cached CSRF token (internal attribute).")
        #     return self.csrf_token
        # No, fetch fresh each time start_sess is called or if needed

        essential_cookies = ["ANCSESSIONID", "SecureATT"]
        if not self.get_cookies(essential_cookies, timeout=10): # Shorter timeout?
            logger.warning(f"Essential cookies {essential_cookies} NOT found before CSRF token API call.")
            return None

        csrf_token_url = urljoin(config_instance.BASE_URL, "discoveryui-matches/parents/api/csrfToken")

        # _api_req handles headers including UBE if driver available
        response_data = _api_req(
            url=csrf_token_url,
            driver=self.driver, # Pass driver for UBE header
            session_manager=self,
            method="GET",
            use_csrf_token=False, # Don't need CSRF to get CSRF
            api_description="CSRF Token API",
        )

        if not response_data:
            logger.warning("Failed to get CSRF token - _api_req returned None.")
            return None

        # Handle potential response formats
        if isinstance(response_data, dict) and "csrfToken" in response_data:
            csrf_token_val = response_data["csrfToken"]
        elif isinstance(response_data, str): # Plain text token
            csrf_token_val = response_data.strip()
            if not csrf_token_val: # Handle empty string response
                 logger.error("CSRF token API returned empty string.")
                 return None
            logger.debug("CSRF token retrieved as plain text string from API endpoint.")
        else:
            logger.error("Unexpected response format for CSRF token API.")
            logger.debug(f"Response data type: {type(response_data)}, value: {response_data}")
            return None

        # Return the valid token (don't cache here, let start_sess handle setting self.csrf_token)
        return csrf_token_val
    # end get_csrf

    def get_cookies(self, cookie_names: List[str], timeout: int = 30) -> bool: # Type hint list
        """Waits until specified cookies are present and returns True, else False."""
        start_time = time.time()
        logger.debug(f"Waiting up to {timeout}s for cookies: {cookie_names}.")
        required_lower = {name.lower() for name in cookie_names}
        interval = 0.5
        last_missing_str = ""

        while time.time() - start_time < timeout:
            try:
                if not self.is_sess_valid(): # Use robust check
                    logger.warning("Session became invalid while waiting for cookies.")
                    return False

                cookies = self.driver.get_cookies()
                current_cookies_lower = {c["name"].lower() for c in cookies}
                missing_lower = required_lower - current_cookies_lower

                if not missing_lower:
                    logger.debug(f"All required cookies found: {cookie_names}.")
                    return True

                # Log missing cookies only if the set changes or periodically
                missing_str = ", ".join(sorted(missing_lower))
                if missing_str != last_missing_str:
                    logger.debug(f"Still missing cookies: {missing_str}")
                    last_missing_str = missing_str

                time.sleep(interval)

            except WebDriverException as e: # Catch WebDriver specific exceptions
                 logger.error(f"WebDriverException while retrieving cookies: {e}")
                 # Check if session died
                 if not self.is_sess_valid():
                      logger.error("Session invalid after WebDriverException during cookie retrieval.")
                      return False
                 # Otherwise, wait and retry
                 time.sleep(interval * 2) # Longer sleep after error
            except Exception as e:
                logger.error(f"Unexpected error retrieving cookies: {e}", exc_info=True)
                # Decide whether to continue or fail on unexpected error
                time.sleep(interval * 2)

        # Loop finished without finding all cookies
        missing_final = [name for name in cookie_names if name.lower() in (required_lower - current_cookies_lower)]
        logger.warning(f"Timeout waiting for cookies. Missing: {missing_final}.")
        return False
    # end get_cookies

    def _sync_cookies(self):
        """Syncs cookies from WebDriver to requests session, adjusting domain for SecureATT."""
        if not self.is_sess_valid(): # Use robust check
            logger.warning("Cannot sync cookies: Session invalid.")
            return

        try:
            cookies = self.driver.get_cookies()
            # Clear existing cookies in requests session before syncing
            self._requests_session.cookies.clear()
            synced_count = 0
            for cookie in cookies:
                # Ensure required fields are present
                if 'name' in cookie and 'value' in cookie and 'domain' in cookie:
                    # Adjust domain for SecureATT cookie if needed
                    domain_to_set = cookie["domain"]
                    if cookie["name"] == "SecureATT" and domain_to_set == "www.ancestry.co.uk":
                        domain_to_set = ".ancestry.co.uk" # Use leading dot for subdomain matching

                    # Prepare cookie attributes for requests session
                    cookie_attrs = {
                        'name': cookie['name'],
                        'value': cookie['value'],
                        'domain': domain_to_set,
                        'path': cookie.get('path', '/'),
                        'secure': cookie.get('secure', False),
                        'rest': {'httpOnly': cookie.get('httpOnly')} # Pass httpOnly via rest dict
                    }
                    # Add expires if present (requests uses 'expires')
                    if 'expiry' in cookie and cookie['expiry'] is not None:
                         cookie_attrs['expires'] = int(cookie['expiry'])

                    self._requests_session.cookies.set(**cookie_attrs)
                    synced_count += 1
                else:
                     logger.warning(f"Skipping invalid cookie format during sync: {cookie}")

            # logger.debug(f"Synced {synced_count} cookies to requests session.") # Less verbose
        except WebDriverException as e:
             logger.error(f"WebDriverException during cookie sync: {e}")
             # Check if session died
             if not self.is_sess_valid():
                  logger.error("Session invalid after WebDriverException during cookie sync.")
        except Exception as e:
            logger.error(f"Unexpected error during cookie sync: {e}", exc_info=True)
    # end _sync_cookies

    class Cache:
        """
        Simple in-memory cache to store API responses to avoid redundant calls.
        Using a class to namespace the cache and avoid potential attribute conflicts in SessionManager.
        """

        def __init__(self):
            self._cache: Dict[str, Any] = {} # Type hint cache dict

        def get(self, key: str) -> Optional[Any]: # Type hint key and return
            return self._cache.get(key)

        def set(self, key: str, value: Any, timeout: Optional[int] = None) -> bool: # Type hint key/value/timeout
            # Timeout is ignored in this simple implementation, but kept for potential future use
            self._cache[key] = value
            # logger.debug(f"Cached value for key: {key}") # Less verbose
            return True # Indicate successful caching

        def clear(self):
            self._cache = {}
            logger.debug("Internal cache cleared.")

        #
    # End of Cache class

    def return_session(self, session: Session):
        """Closes the session, returning the underlying connection to the pool."""
        if session:
            session_id = id(session) # For logging
            try:
                # logger.debug(f"Closing session {session_id} (returns connection to pool).") # Less verbose
                session.close()
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}", exc_info=True)
    # end return_session

    def get_db_conn(self) -> Optional[Session]:
        """Gets a session from the SQLAlchemy session factory. Initializes if needed."""
        # Log which SM instance and engine is involved
        engine_id = id(self.engine) if self.engine else 'None'
        logger.debug(f"SessionManager ID={id(self)} get_db_conn called. Current Engine ID: {engine_id}")

        # --- Check if initialization is needed ---
        # Initialize if never attempted OR if engine/Session became None after previous attempt
        if not self._db_init_attempted or not self.engine or not self.Session:
            logger.debug(f"SessionManager ID={id(self)}: Engine/Session factory not ready. Triggering initialization...")
            try:
                self._initialize_db_engine_and_session()
                # Check again after initialization attempt
                if not self.Session:
                    logger.error(f"SessionManager ID={id(self)}: Initialization failed, cannot get DB connection.")
                    return None
            except Exception as init_e:
                 logger.error(f"SessionManager ID={id(self)}: Exception during lazy initialization in get_db_conn: {init_e}")
                 return None

        # --- Attempt to get session from factory ---
        try:
            # self.Session should now be valid if initialization succeeded
            session = self.Session()
            logger.debug(f"SessionManager ID={id(self)} obtained DB session {id(session)} from Engine ID={id(self.engine)}")
            return session
        except Exception as e:
            logger.error(f"SessionManager ID={id(self)} Error getting DB session from factory: {e}", exc_info=True)
            # If getting session fails, maybe the engine died? Clear it to force re-init next time.
            if self.engine:
                try: self.engine.dispose()
                except Exception: pass
            self.engine = None
            self.Session = None
            self._db_init_attempted = False # Allow re-init attempt
            return None
    # End of get_db_conn

    @contextlib.contextmanager
    def get_db_conn_context(self) -> Generator[Optional[Session], None, None]:
        """Context manager using the new get_db_conn and return_session."""
        session: Optional[Session] = None
        try:
            session = self.get_db_conn()
            if not session:
                logger.error("Failed to obtain DB session within context manager.")
                yield None
                return

            yield session # Provide the session

            # COMMIT if successful exit from 'with' block
            if session.is_active:
                session.commit()
        except Exception as e:
            logger.error(f"Exception within get_db_conn_context block or commit: {e}. Rolling back.", exc_info=True)
            if session and session.is_active:
                try:
                    session.rollback()
                except Exception as rb_err:
                    logger.error(f"Error rolling back session in context manager: {rb_err}")
            raise e # Re-raise
        finally:
            # Ensure session is closed/returned regardless of success/failure
            if session:
                self.return_session(session)
    # End of get_db_conn_context method

    def cls_db_conn(self):
        """Disposes the SQLAlchemy engine, closing all pooled connections."""
        engine_id = id(self.engine) if self.engine else 'None'
        logger.debug(f"SessionManager ID={id(self)} cls_db_conn called. Disposing Engine ID: {engine_id}")
        if self.engine:
            try:
                self.engine.dispose()
                logger.debug(f"Engine ID={engine_id} disposed.")
            except Exception as e:
                logger.error(f"Error disposing SQLAlchemy engine ID={engine_id}: {e}", exc_info=True)
            finally:
                self.engine = None # Set engine to None
                self.Session = None # Set Session factory to None
                self._db_init_attempted = False # Reset flag
        else:
            logger.debug(f"SessionManager ID={id(self)}: No active SQLAlchemy engine to dispose.")
    # End of cls_db_conn

    @retry_api() # Add retry decorator
    def get_my_profileId(self) -> Optional[str]:
        """
        Retrieves the user ID (ucdmid) from the Ancestry API using _api_req. Retries on failure.
        """
        url = urljoin(config_instance.BASE_URL, "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid")
        try:
            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="Get my profile_id",
            )

            if not response_data:
                logger.warning("Failed to get profile_id response via _api_req.")
                return None

            if isinstance(response_data, dict) and "data" in response_data and "ucdmid" in response_data["data"]:
                my_profile_id_val = str(response_data["data"]["ucdmid"]).upper() # Ensure string and uppercase
                return my_profile_id_val
            else:
                logger.error("Could not find 'data.ucdmid' in profile_id API response.")
                logger.debug(f"Full profile_id response data: {response_data}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in get_my_profileId: {e}", exc_info=True)
            return None
    # end get_my_profileId

    @retry_api() # Add retry decorator
    def get_my_uuid(self) -> Optional[str]: # Ensure return type consistency
        """Retrieves the test uuid (sampleId) from the header/dna API endpoint. Retries on failure."""
        if not self.is_sess_valid(): # Check session validity first
            logger.error("get_my_uuid: Session invalid.")
            return None

        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        response_data = _api_req(
            url=url,
            driver=self.driver,
            session_manager=self,
            method="GET",
            use_csrf_token=False,
            api_description="Get UUID API",
        )

        if response_data:
            if isinstance(response_data, dict) and "testId" in response_data:
                my_uuid_val = str(response_data["testId"]).upper() # Ensure string and uppercase
                return my_uuid_val
            else:
                logger.error("Could not retrieve my_uuid ('testId' missing in response).")
                logger.debug(f"Full get_my_uuid response data: {response_data}")
                return None
        else:
            logger.error("Failed to get header/dna data via _api_req.")
            return None
    # end of get_my_uuid

    @retry_api() # Add retry decorator
    def get_my_tree_id(self) -> Optional[str]:
        """
        Retrieves the tree ID based on TREE_NAME from config, using the header/trees API. Retries on failure.
        """
        tree_name_config = config_instance.TREE_NAME
        if not tree_name_config:
            logger.debug("TREE_NAME not configured, skipping tree ID retrieval.")
            return None

        if not self.is_sess_valid(): # Check session validity first
            logger.error("get_my_tree_id: Session invalid.")
            return None

        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/trees")

        try:
            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="Header Trees API",
            )
            if response_data and isinstance(response_data, dict) and "menuitems" in response_data:
                for item in response_data["menuitems"]:
                    if isinstance(item, dict) and item.get("text") == tree_name_config:
                        tree_url = item.get("url")
                        if tree_url and isinstance(tree_url, str):
                            # Robust parsing: find the part after /tree/ and before the next /
                            match = re.search(r"/tree/(\d+)", tree_url)
                            if match:
                                my_tree_id_val = match.group(1)
                                return my_tree_id_val
                            else:
                                logger.warning(f"Found tree '{tree_name_config}', but URL format unexpected: {tree_url}")
                        else:
                            logger.warning(f"Found tree '{tree_name_config}', but 'url' key missing or invalid.")
                        break # Stop searching once found

                logger.warning(f"Could not find TREE_NAME '{tree_name_config}' in Header Trees API response.")
                return None
            else:
                logger.warning("Unexpected response format from Header Trees API (missing 'menuitems'?).")
                logger.debug(f"Full Header Trees response data: {response_data}")
                return None
        except Exception as e:
            logger.error(f"Error fetching/parsing Header Trees API: {e}", exc_info=True)
            return None
    # End of get_my_tree_id

    @retry_api() # Add retry decorator
    def get_tree_owner(self, tree_id: str) -> Optional[str]:
        """
        Retrieves the tree owner's display name using _api_req. Retries on failure.
        """
        if not tree_id:
            logger.warning("Cannot get tree owner: tree_id is missing.")
            return None

        if not self.is_sess_valid(): # Check session validity first
            logger.error("get_tree_owner: Session invalid.")
            return None

        url = urljoin(config_instance.BASE_URL, f"api/uhome/secure/rest/user/tree-info?tree_id={tree_id}")

        try:
            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="Tree Owner Name API",
            )
            if response_data and isinstance(response_data, dict):
                owner_data = response_data.get("owner")
                if owner_data and isinstance(owner_data, dict):
                    display_name = owner_data.get("displayName")
                    if display_name and isinstance(display_name, str):
                        return display_name
                    else:
                        logger.warning("Could not find 'displayName' in 'owner' data.")
                else:
                    logger.warning("Could not find 'owner' data in Tree Owner API response.")
                logger.debug(f"Full Tree Owner API response data: {response_data}")
                return None
            else:
                logger.warning("Tree Owner API call via _api_req returned unexpected data or None.")
                logger.debug(f"Response received: {response_data}")
                return None
        except Exception as e:
            logger.error(f"Error fetching/parsing Tree Owner API: {e}", exc_info=True)
            return None
    # End of get_tree_owner

    def verify_sess(self):
        """Verifies session using login_status (which prioritizes API)."""
        logger.debug("Verifying session status...") # Simplified log
        login_ok = login_status(self) # login_status now handles API/UI checks

        if login_ok is True:
            logger.debug("Session verification successful.")
            return True
        elif login_ok is False:
            logger.warning("Session verification failed (user not logged in).")
            # Optionally attempt re-login here if desired, or let caller handle
            # login_result = log_in(self) ... etc.
            return False
        else: # login_ok is None (critical error)
            logger.error("Session verification failed critically.")
            return False
    # End of verify_sess

    def _verify_api_login_status(self) -> bool:
        """
        Checks login status by making a request to a known secure API endpoint using requests.
        Assumes cookies have already been synced to self._requests_session.
        Uses header/dna endpoint. Returns True if API call succeeds (implies login), False otherwise.
        Handles the fact that _api_req returns parsed data on success, None on failure.
        """
        if self.api_login_verified:
            # logger.debug("API login already verified this session (cached).") # Less verbose
            return True

        logger.debug("Verifying login status via header/dna API endpoint...")
        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        api_description = "API Login Verification (header/dna)"

        try:
            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description=api_description,
                force_requests=True, # Use requests library path
            )

            if response_data is not None:
                if isinstance(response_data, dict) and "testId" in response_data:
                    logger.debug(f"API login check successful via {api_description}.")
                    self.api_login_verified = True # Set cache flag on success
                    return True
                else:
                    logger.warning(f"API login check via {api_description} succeeded (2xx), but response format unexpected: {response_data}")
                    self.api_login_verified = True # Cautiously set True
                    return True
            else:
                logger.warning(f"API login check failed: {api_description} call returned None (likely HTTP error or timeout).")
                self.api_login_verified = False
                return False

        except Exception as e:
            logger.error(f"Unexpected error during API login status check ({api_description}): {e}", exc_info=True) # Log full traceback
            self.api_login_verified = False
            return False
    # end _verify_api_login_status

    @retry_api() # Use retry_api decorator
    def get_header(self) -> bool: # Return bool for success/failure
        """Retrieves data from the header/dna API endpoint. Retries on failure."""
        if not self.is_sess_valid(): # Check session first
            logger.error("get_header: Session invalid.")
            return False

        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        response_data = _api_req(
            url, self.driver, self, method="GET", use_csrf_token=False,
            api_description="Get UUID API" # Use appropriate description
        )
        if response_data:
            if isinstance(response_data, dict) and "testId" in response_data:
                logger.debug("Header data retrieved successfully (testId found).")
                return True
            else:
                logger.error("Unexpected response structure from header/dna API.")
                logger.debug(f"Response: {response_data}")
                return False
        else:
            logger.error("Failed to get header/dna data.")
            return False
    # end get_header

    def _validate_ess_cookies(self, required_cookies: List[str]) -> bool: # Type hint
        """Check if specific cookies exist in the current session."""
        if not self.is_sess_valid(): # Use robust check
             logger.warning("Cannot validate cookies: Session invalid.")
             return False
        try:
            cookies = {c["name"]: c["value"] for c in self.driver.get_cookies()}
            return all(cookie in cookies for cookie in required_cookies)
        except WebDriverException as e:
             logger.error(f"WebDriverException during cookie validation: {e}")
             if not self.is_sess_valid():
                  logger.error("Session invalid after WebDriverException during cookie validation.")
             return False
        except Exception as e:
            logger.error(f"Unexpected error validating cookies: {e}", exc_info=True)
            return False
    # End of _validate_ess_cookies

    def is_sess_logged_in(self) -> bool: # Return bool only
        """DEPRECATED: Use login_status() instead. Checks session validity AND login using UI element verification ONLY."""
        logger.warning("is_sess_logged_in is deprecated. Use login_status() instead.")
        return login_status(self) is True # Delegate to new function, return bool
    # End is_sess_logged_in

    def is_sess_valid(self) -> bool: # Type hint return
        """Checks if the current browser session is valid (quick check), handling InvalidSessionIdException."""
        if not self.driver:
            # logger.debug("Browser not open (driver is None).") # Less verbose
            return False
        try:
            # Try a minimal WebDriver command that requires an active session
            _ = self.driver.window_handles # Accessing handles is lightweight
            return True # If no exception, browser session is likely valid
        except InvalidSessionIdException:
            logger.debug("Session ID is invalid (browser likely closed or session terminated).")
            return False
        except WebDriverException as e: # Catch other WebDriver exceptions
            # Log specific WebDriver exceptions that might indicate a dead session
            if "disconnected" in str(e).lower() or "target crashed" in str(e).lower():
                 logger.warning(f"Session seems invalid due to WebDriverException: {e}")
                 return False
            else:
                 logger.warning(f"Unexpected WebDriverException checking session validity: {e}")
                 # Could still be valid despite other errors, maybe return True cautiously?
                 # For safety, let's return False if any WebDriverException occurs here.
                 return False
        except Exception as e: # Catch any other unexpected exceptions
            logger.error(f"Unexpected error checking session validity: {e}", exc_info=True)
            return False
    # End of is_sess_valid

    def close_sess(self):
        """Closes the Selenium WebDriver session."""
        if self.driver:
            logger.debug("Attempting to close WebDriver session...")
            try:
                self.driver.quit()
                logger.debug("WebDriver session quit successfully.")
            except Exception as e:
                logger.error(f"Error closing WebDriver session: {e}", exc_info=True)
            finally:
                self.driver = None # Ensure driver is None even if quit fails
        else:
             logger.debug("No active WebDriver session to close.")
        self.session_active = False
        self.api_login_verified = False # Reset API verification flag
        self.csrf_token = None # Clear CSRF token
        # Keep DB connection pool alive unless explicitly closed by cls_db_conn
    # End of close_sess

    def restart_sess(self, url: Optional[str] = None) -> bool: # Return bool for success
        """Restarts the WebDriver session and optionally navigates to a URL."""
        logger.warning("Restarting WebDriver session...")
        self.close_sess() # Ensure any existing session is closed
        start_ok, _ = self.start_sess(action_name="Session Restart") # Attempt to start new session
        if not start_ok:
             logger.error("Failed to restart session.")
             return False
        # Session started, now navigate if URL provided
        if url and self.driver:
            logger.info(f"Re-navigating to {url} after restart...")
            if nav_to_page(self.driver, url, selector="body", session_manager=self): # Wait for body
                logger.info(f"Successfully re-navigated to {url}.")
                return True
            else:
                logger.error(f"Failed to re-navigate to {url} after restart.")
                return False # Navigation failed after successful restart
        elif not url:
             return True # Restart succeeded, no navigation needed
        else: # url provided but driver missing after start_sess (shouldn't happen if start_ok is True)
             logger.error("Driver instance missing after successful session restart report.")
             return False
    # End of restart_sess

    @ensure_browser_open
    def make_tab(self) -> Optional[str]: # Type hint return
        """Create a new tab and return its handle id"""
        driver = self.driver # Assume decorator ensures driver exists
        try:
            tab_list_before = driver.window_handles
            # logger.debug(f"Initial window handles: {tab_list_before}")
            driver.switch_to.new_window("tab")
            # Wait briefly for the new handle to appear
            WebDriverWait(driver, selenium_config.NEW_TAB_TIMEOUT).until(
                 lambda d: len(d.window_handles) > len(tab_list_before)
            )
            tab_list_after = driver.window_handles
            new_tab_handle = list(set(tab_list_after) - set(tab_list_before))[0]
            # logger.debug(f"New tab handle: {new_tab_handle}")
            return new_tab_handle
        except TimeoutException:
            logger.error("Timeout waiting for new tab handle to appear.")
            logger.debug(f"Window handles during timeout: {driver.window_handles}")
            return None
        except (IndexError, WebDriverException) as e: # Catch potential errors finding handle
             logger.error(f"Error identifying new tab handle: {e}")
             logger.debug(f"Window handles during error: {driver.window_handles}")
             return None
        except Exception as e:
            logger.error(f"An unexpected error occurred in make_tab: {e}", exc_info=True)
            return None
    # End of make_tab

    def check_js_errors(self):
        """Checks for new JavaScript errors since the last check and logs them."""
        if not self.is_sess_valid(): # Check session validity
            # logger.debug("Skipping JS error check: Session invalid.") # Less verbose
            return

        try:
            # Check if get_log is supported (might not be in headless/certain configs)
            log_types = self.driver.log_types
            if 'browser' not in log_types:
                 # logger.debug("Browser log type not supported by this WebDriver instance.") # Less verbose
                 return

            logs = self.driver.get_log("browser")
            new_errors_found = False
            most_recent_error_time = self.last_js_error_check # Keep track of latest error this check

            for entry in logs:
                # Process only SEVERE level logs
                if entry.get("level") == "SEVERE":
                    try:
                        # Convert ms timestamp to datetime object
                        timestamp_ms = entry.get("timestamp")
                        if timestamp_ms:
                             timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
                             # Check if the error occurred *after* the last check time
                             if timestamp > self.last_js_error_check:
                                  new_errors_found = True
                                  # --- Enhanced Logging ---
                                  error_message = entry.get('message', 'No message')
                                  # Try to extract source and line number if available
                                  source_match = re.search(r'(.+?):(\d+)', error_message)
                                  source_info = f" (Source: {source_match.group(1).split('/')[-1]}:{source_match.group(2)})" if source_match else ""
                                  # Log clearly identifiable JS errors
                                  logger.warning(f"JS ERROR DETECTED:{source_info} {error_message}")
                                  # --- End Enhanced Logging ---
                                  # Update the most recent error time found in this batch
                                  if timestamp > most_recent_error_time:
                                       most_recent_error_time = timestamp
                        else:
                             logger.warning(f"JS Log entry missing timestamp: {entry}")
                    except Exception as parse_e:
                         logger.warning(f"Error parsing JS log entry {entry}: {parse_e}")

            # Update last_js_error_check time to the time of the most recent error found *in this specific check*
            # This prevents re-logging old errors if check_js_errors is called frequently
            if new_errors_found:
                 self.last_js_error_check = most_recent_error_time

        except WebDriverException as e:
            # Handle cases where get_log might fail even if supported initially
            logger.warning(f"WebDriverException checking for Javascript errors: {e}")
        except Exception as e:
            logger.error(f"Unexpected error checking for Javascript errors: {e}", exc_info=True)
    # end of check_js_errors
# end SessionManager


# ----------------------------------------------------------------------------
# Stand alone functions
# ----------------------------------------------------------------------------

def _api_req(
    url: str,
    driver: Optional[WebDriver], # Keep driver optional for UBE header etc.
    session_manager: SessionManager, # Now required
    method: str = "GET",
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    use_csrf_token: bool = True,
    headers: Optional[Dict] = None,
    referer_url: Optional[str] = None,
    api_description: str = "API Call",
    timeout: Optional[int] = None,
    force_requests: bool = False, # Keep flag
) -> Optional[Any]:
    """
    V13.2 REVISED: Makes an HTTP request using Python requests library with retry logic.
    Handles headers, CSRF, cookies, and rate limiting interaction.
    Relies on SessionManager for session state. Ensures session is valid.
    """
    if not session_manager:
        logger.error(f"{api_description}: Aborting - SessionManager instance is required.")
        return None

    # --- Session Validity Check (Crucial!) ---
    # Check if we need an active browser session (e.g., for UBE or if not forcing requests)
    # If forcing requests, we might not strictly need the browser, but need cookies synced.
    # Let's check API verification status - if not verified, browser needed for login/cookie sync.
    browser_needed = not force_requests or not session_manager.api_login_verified
    if browser_needed and not session_manager.is_sess_valid():
         logger.error(f"{api_description}: Aborting - Browser session is invalid or closed.")
         return None

    # --- Retry Configuration ---
    max_retries = config_instance.MAX_RETRIES
    initial_delay = config_instance.INITIAL_DELAY # Use initial delay from config
    backoff_factor = config_instance.BACKOFF_FACTOR
    max_delay = config_instance.MAX_DELAY
    retry_status_codes = config_instance.RETRY_STATUS_CODES

    # --- Prepare Headers ---
    final_headers = {}
    # Start with contextual headers from config
    contextual_headers = config_instance.API_CONTEXTUAL_HEADERS.get(api_description, {})
    final_headers.update({k: v for k, v in contextual_headers.items() if v is not None})
    # Override/add with function-specific headers
    if headers:
        final_headers.update({k: v for k, v in headers.items() if v is not None})

    # Ensure User-Agent
    if "User-Agent" not in final_headers:
        ua = None
        if driver and session_manager.is_sess_valid(): # Check validity
            try: ua = driver.execute_script("return navigator.userAgent;")
            except Exception: pass # Ignore errors getting UA from driver
        if not ua: ua = random.choice(config_instance.USER_AGENTS)
        final_headers["User-Agent"] = ua

    # Ensure Referer if provided and not already set
    if referer_url and "Referer" not in final_headers:
        final_headers["Referer"] = referer_url

    # Add CSRF Token if required
    if use_csrf_token:
        csrf = session_manager.csrf_token # Get from manager attribute
        if csrf: final_headers["X-CSRF-Token"] = csrf
        else:
             # Attempt to fetch CSRF if missing? Or just fail? Fail for now.
             logger.error(f"{api_description}: CSRF token required but not found in SessionManager.")
             return None # Fail if CSRF needed but missing

    # Add UBE Header if driver available
    if driver and session_manager.is_sess_valid() and "ancestry-context-ube" not in final_headers:
        ube_header = make_ube(driver)
        if ube_header: final_headers["ancestry-context-ube"] = ube_header
        else: logger.warning(f"{api_description}: Failed to generate UBE header.")

    # Add ancestry-userid if needed and available
    if "ancestry-userid" in contextual_headers and session_manager.my_profile_id:
        final_headers["ancestry-userid"] = session_manager.my_profile_id.upper()

    # --- Set Timeout ---
    request_timeout = timeout if timeout is not None else selenium_config.API_TIMEOUT

    # --- Use Python Requests Library ---
    logger.debug(f"API Req: {method.upper()} {url}")
    req_session = session_manager._requests_session
    # logger.debug(f"{api_description}: Using requests.Session {id(req_session)}") # Less verbose

    retries_left = max_retries
    last_exception = None
    delay = initial_delay # Initialize delay for retry calculation

    while retries_left > 0:
        attempt = max_retries - retries_left + 1
        response = None

        try:
            # --- Sync cookies BEFORE each request attempt ---
            # Only sync if driver is available and session seems valid
            if driver and session_manager.is_sess_valid():
                try: session_manager._sync_cookies()
                except Exception as sync_err:
                    logger.warning(f"{api_description}: Error syncing cookies (Attempt {attempt}): {sync_err}")
                    # Proceed, but request might fail due to stale cookies

            # --- Make Request ---
            response = req_session.request(
                method=method.upper(), url=url, headers=final_headers,
                data=data, json=json_data, timeout=request_timeout, verify=True,
            )
            status = response.status_code
            # logger.debug(f"{api_description}: Attempt {attempt}/{max_retries}. Status: {status}") # Less verbose

            # --- Handle Response ---
            # 1. Check for Retryable Status Codes
            if status in retry_status_codes:
                retries_left -= 1
                last_exception = HTTPError(f"{status} Server Error: {response.reason}", response=response)
                if retries_left <= 0:
                    logger.error(f"{api_description}: Failed after {max_retries} attempts (Final Status {status}). Resp: {response.text[:200] if response else 'N/A'}")
                    if status == 429 and session_manager.dynamic_rate_limiter: session_manager.dynamic_rate_limiter.increase_delay()
                    return None # Indicate failure
                else:
                    # Calculate delay using session manager's rate limiter logic if possible
                    sleep_time = initial_delay # Start with base delay
                    if session_manager.dynamic_rate_limiter:
                         if status == 429: session_manager.dynamic_rate_limiter.increase_delay()
                         # Use current delay from limiter for backoff calc? Or just fixed backoff? Fixed for now.
                         sleep_time = delay * (backoff_factor ** (attempt - 1)) # Use current delay value
                         jitter = random.uniform(-0.1, 0.1) * delay # Use current delay value
                         sleep_time = min(sleep_time + jitter, max_delay)
                         sleep_time = max(0.1, sleep_time)
                    else: # Fallback if no rate limiter
                         sleep_time = delay * (backoff_factor ** (attempt - 1))
                         sleep_time = min(sleep_time, max_delay)
                         sleep_time = max(0.1, sleep_time)

                    logger.warning(f"{api_description}: Status {status} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    delay *= backoff_factor # Increase delay for next potential retry
                    continue # Try again

            # 2. Check for Success (2xx)
            elif response.ok:
                if session_manager.dynamic_rate_limiter: session_manager.dynamic_rate_limiter.decrease_delay()

                # Handle specific text parsing needs
                force_text_parsing = api_description == "Get Ladder API (Batch)"
                if force_text_parsing: return response.text

                # Standard JSON / Text Handling
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    try: return response.json()
                    except json.JSONDecodeError:
                        logger.warning(f"{api_description}: OK ({status}), but failed JSON decode. Returning text.")
                        logger.debug(f"Response text: {response.text[:500]}")
                        return response.text
                else:
                    # Handle plain text CSRF specifically
                    if api_description == "CSRF Token API" and "text/plain" in content_type:
                        csrf_text = response.text.strip()
                        return csrf_text if csrf_text else None # Return None if empty string
                    return response.text # Return text for other types

            # 3. Handle Non-Retryable Errors (>= 400 and not in retry_codes)
            else:
                logger.error(f"{api_description}: Non-retryable error: {status} {response.reason}. Resp: {response.text[:500]}")
                if status in [401, 403]: # Unauthorized/Forbidden
                    logger.warning(f"{api_description}: Authentication/Authorization error ({status}). Marking API login as unverified.")
                    session_manager.api_login_verified = False
                    # Maybe attempt session restart? For now, just fail.
                return None # Indicate failure

        # --- Handle Network/Timeout Errors ---
        except requests.exceptions.RequestException as e:
            retries_left -= 1
            last_exception = e
            if retries_left <= 0:
                logger.error(f"{api_description}: RequestException failed after {max_retries} attempts. Final Error: {e}", exc_info=False)
                return None # Indicate failure
            else:
                # Calculate delay
                sleep_time = delay * (backoff_factor ** (attempt - 1)) # Use current delay
                sleep_time = min(sleep_time, max_delay)
                sleep_time = max(0.1, sleep_time)
                logger.warning(f"{api_description}: RequestException (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s... Error: {e}")
                time.sleep(sleep_time)
                delay *= backoff_factor # Increase delay for next attempt
                continue # Try again

        # --- Handle Other Unexpected Errors ---
        except Exception as e:
            logger.critical(f"{api_description}: CRITICAL Unexpected error during request attempt {attempt}: {e}", exc_info=True)
            raise e # Re-raise critical unexpected errors

    # Should only be reached if all retries failed
    logger.error(f"{api_description}: Exited retry loop after {max_retries} failed attempts. Last Exception: {last_exception}. Returning None.")
    return None
# End of _api_req

def make_ube(driver: Optional[WebDriver]) -> Optional[str]: # Type hint driver
    """Generates the UBE header value. Requires an active WebDriver session."""
    if not driver:
        logger.debug("Cannot generate UBE header: WebDriver is None.")
        return None
    try:
        # Ensure session is valid before getting cookies
        # Temporarily create a SessionManager to call is_sess_valid. This is not ideal.
        # A better approach would be to pass SessionManager instance if available.
        temp_sm = SessionManager() # Create temporary manager
        temp_sm.driver = driver # Assign driver
        if not temp_sm.is_sess_valid():
             logger.warning("Cannot generate UBE header: Session invalid.")
             return None

        cookies = {c["name"]: c["value"] for c in driver.get_cookies()}
        ancsessionid = cookies.get("ANCSESSIONID")
        if not ancsessionid:
            logger.warning("ANCSESSIONID cookie not found for UBE header.")
            # Optionally try to fetch it again? For now, return None.
            return None

        # Generate unique IDs for the event
        event_id = str(uuid.uuid4())
        correlated_id = str(uuid.uuid4())

        ube_data = {
            "eventId": event_id,
            "correlatedScreenViewedId": correlated_id,
            "correlatedSessionId": ancsessionid,
            "userConsent": "necessary|preference|performance|analytics1st|analytics3rd|advertising1st|advertising3rd|attribution3rd", # Standard consent string
            "vendors": "adobemc", # Standard vendor string
            "vendorConfigurations": "{}", # Empty JSON object as string
        }
        # Use standard base64 encoding
        json_payload = json.dumps(ube_data, separators=(",", ":")).encode('utf-8')
        encoded_payload = base64.b64encode(json_payload).decode('utf-8')
        return encoded_payload

    except WebDriverException as e:
         logger.error(f"WebDriverException generating UBE header: {e}")
         temp_sm = SessionManager() # Create temporary manager again
         temp_sm.driver = driver
         if not temp_sm.is_sess_valid(): # Check again
              logger.error("Session invalid after WebDriverException during UBE generation.")
         return None
    except Exception as e:
        logger.error(f"Unexpected error generating UBE Header: {e}", exc_info=True)
        return None
# End of make_ube

def make_newrelic(driver: Optional[WebDriver]) -> Optional[str]: # Type hint driver
    """Generates the newrelic header value."""
    # This doesn't strictly need the driver, but kept arg for consistency
    try:
        # Generate random hex IDs (more typical format)
        trace_id = uuid.uuid4().hex[:16] # 16 hex chars
        span_id = uuid.uuid4().hex[:16] # 16 hex chars
        account_id = "1690570" # From observed data
        app_id = "1588726612" # From observed data
        tk = "2611750" # From observed data

        newrelic_data = {
            "v": [0, 1], # Version
            "d": {
                "ty": "Browser", # Type
                "ac": account_id, # Account ID
                "ap": app_id, # App ID
                "id": span_id, # Span ID
                "tr": trace_id, # Trace ID
                "ti": int(time.time() * 1000), # Timestamp ms
                "tk": tk, # Trust Key
            },
        }
        json_payload = json.dumps(newrelic_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        return encoded_payload

    except Exception as e:
        logger.error(f"Error generating NewRelic header: {e}", exc_info=True)
        return None
# End of make_newrelic

def make_traceparent(driver: Optional[WebDriver]) -> Optional[str]: # Type hint driver
    """Generates the traceparent header value (W3C Trace Context)."""
    # This doesn't strictly need the driver
    try:
        version = "00"
        trace_id = uuid.uuid4().hex # 32 hex chars
        parent_id = uuid.uuid4().hex[:16] # 16 hex chars for parent/span ID
        flags = "01" # Sampled flag
        traceparent = f"{version}-{trace_id}-{parent_id}-{flags}"
        return traceparent

    except Exception as e:
        logger.error(f"Error generating traceparent header: {e}", exc_info=True)
        return None
# End of make_traceparent

def make_tracestate(driver: Optional[WebDriver]) -> Optional[str]: # Type hint driver
    """Generates the tracestate header value (W3C Trace Context)."""
    # This doesn't strictly need the driver
    try:
        # Based on observed New Relic format within tracestate
        tk = "2611750" # Trust key
        account_id = "1690570"
        app_id = "1588726612"
        span_id = uuid.uuid4().hex[:16] # Matches parent_id in traceparent usually
        timestamp = int(time.time() * 1000)

        # Format: <tk>@nr=<version>-<type>-<account>-<app>-<span_id>-<transaction_id?>-<sampled?>----<timestamp>
        # Transaction ID and sampled flag might be empty/default in basic cases
        tracestate = f"{tk}@nr=0-1-{account_id}-{app_id}-{span_id}----{timestamp}"
        return tracestate

    except Exception as e:
        logger.error(f"Error generating tracestate header: {e}", exc_info=True)
        return None
# End of make_tracestate

def get_driver_cookies(driver: WebDriver) -> Dict[str, str]: # Type hint driver and return
    """Retrieves cookies from the Selenium driver as a simple dictionary."""
    if not driver:
         logger.warning("Cannot get driver cookies: WebDriver is None.")
         return {}
    try:
        cookies_dict = {cookie["name"]: cookie["value"] for cookie in driver.get_cookies()}
        return cookies_dict
    except WebDriverException as e:
         logger.error(f"WebDriverException getting driver cookies: {e}")
         # Check session validity after error
         temp_sm = SessionManager() # Create temporary manager
         temp_sm.driver = driver
         if not temp_sm.is_sess_valid():
              logger.error("Session invalid after WebDriverException getting cookies.")
         return {}
    except Exception as e:
        logger.error(f"Unexpected error getting driver cookies: {e}", exc_info=True)
        return {}
# End of get_driver_cookies


# ----------------------------------------------------------------------------
# Login
# ----------------------------------------------------------------------------

# Login 5
@time_wait("Handle 2FA Page")
def handle_twoFA(session_manager: SessionManager) -> bool:
    """Handles the two-step verification page, choosing SMS method and waiting for user input."""
    driver = session_manager.driver
    if not driver:
        logger.error("2FA handling failed: WebDriver is not available.")
        return False

    # Use wait factory methods from selenium_config
    element_wait = selenium_config.element_wait(driver) # Default wait for elements
    page_wait = selenium_config.page_wait(driver) # Page load wait
    short_wait = selenium_config.short_wait(driver) # Short wait for quick checks
    # long_wait = selenium_config.long_wait(driver) # Not used directly here, timeout value used instead

    try:
        logger.debug("Handling Two-Factor Authentication (2FA)...")

        # 1. Wait for 2FA page indicator to be present
        try:
            logger.debug("Waiting for 2FA page header...")
            element_wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)))
            logger.debug("2FA page detected.")
        except TimeoutException:
            logger.debug("Did not detect 2FA page header within timeout.")
            # Check if already logged in despite missing header (e.g., race condition)
            if login_status(session_manager):
                logger.info("User appears logged in after checking for 2FA page. Proceeding.")
                return True
            logger.warning("Assuming 2FA not required or page didn't load correctly.")
            return False # Fail if 2FA page expected but not found and not logged in

        # 2. Wait for SMS button and click it
        try:
            logger.debug("Waiting for 2FA 'Send Code' (SMS) button...")
            # Wait for presence first
            sms_button_present = element_wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR)))
            # Then wait briefly for clickability
            sms_button_clickable = short_wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR)))

            if sms_button_clickable:
                logger.debug("Attempting to click 'Send Code' button using JavaScript...")
                driver.execute_script("arguments[0].click();", sms_button_clickable)
                logger.debug("'Send Code' button clicked.")
                # Optional: Wait briefly for code input field to appear after click
                try:
                    short_wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, TWO_FA_CODE_INPUT_SELECTOR)))
                    logger.debug("Code input field appeared after clicking 'Send Code'.")
                except TimeoutException:
                    logger.warning("Code input field did not appear quickly after clicking 'Send Code'.")
            else:
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

        # 3. Wait for user to enter 2FA code manually
        code_entry_timeout_value = selenium_config.TWO_FA_CODE_ENTRY_TIMEOUT
        logger.warning(f"Waiting up to {code_entry_timeout_value}s for user to manually enter 2FA code and submit...")

        start_time = time.time()
        logged_in = False
        while time.time() - start_time < code_entry_timeout_value:
            current_status = login_status(session_manager)
            if current_status is True:
                logged_in = True
                logger.info("User completed 2FA successfully (login confirmed).")
                break
            elif current_status is None: # Critical error during status check
                 logger.error("Critical error checking login status during 2FA wait.")
                 return False

            # Check if 2FA page is still present
            try:
                 # Use short wait to check if header still exists
                 WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)))
                 # Still on 2FA page, continue waiting
            except TimeoutException:
                 # 2FA page disappeared
                 logger.warning("2FA page disappeared, re-checking login status immediately...")
                 if login_status(session_manager): # Check immediately after page change
                      logged_in = True
                      logger.info("User completed 2FA successfully (login confirmed after page change).")
                      break
                 else:
                      logger.error("2FA page disappeared, but login still not confirmed.")
                      return False
            except WebDriverException as e:
                 logger.error(f"WebDriver error checking for 2FA header: {e}")
                 # Potentially session died, fail
                 return False

            time.sleep(2) # Poll every 2 seconds

        if not logged_in:
            logger.error(f"Timed out ({code_entry_timeout_value}s) waiting for user 2FA code entry.")
            return False

        return True

    except WebDriverException as e:
         logger.error(f"WebDriverException during 2FA handling: {e}")
         if not session_manager.is_sess_valid(): # Check if session died
              logger.error("Session invalid after WebDriverException during 2FA.")
         return False
    except Exception as e:
        logger.error(f"Unexpected error during 2FA handling: {e}", exc_info=True)
        return False
# End of handle_twoFA

# Login 4
def enter_creds(driver: WebDriver) -> bool: # Type hint driver and return
    """Enters username and password into the login form."""
    element_wait = selenium_config.element_wait(driver)
    try:
        logger.debug("Entering Credentials and Signing In...")

        # --- Username ---
        logger.debug("Waiting for username input field...")
        username_input = element_wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))) # Use visibility
        logger.debug("Username input field found.")
        username_input.click() # Click first
        username_input.clear() # Then clear

        ancestry_username = config_instance.ANCESTRY_USERNAME
        if not ancestry_username: # Check if empty or None
            raise ValueError("ANCESTRY_USERNAME configuration is missing or empty.")
        logger.debug(f"Entering username: {ancestry_username}")
        username_input.send_keys(ancestry_username)
        logger.debug("Username entered.")

        # --- Password ---
        logger.debug("Waiting for password input field...")
        password_input = element_wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))) # Use visibility
        logger.debug("Password input field found.")
        password_input.click()
        password_input.clear()

        ancestry_password = config_instance.ANCESTRY_PASSWORD
        if not ancestry_password:
             raise ValueError("ANCESTRY_PASSWORD configuration is missing or empty.")
        logger.debug("Entering password: ***") # Avoid logging password
        password_input.send_keys(ancestry_password)
        logger.debug("Password entered.")

        # --- Sign In Button ---
        logger.debug("Waiting for sign in button to be clickable...")
        sign_in_button = element_wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR)))
        logger.debug("Clicking 'Sign in' button...")
        try:
             sign_in_button.click()
             logger.debug("Sign in button clicked.")
             return True
        except ElementClickInterceptedException:
             logger.warning("Sign in button click intercepted, trying JS click...")
             driver.execute_script("arguments[0].click();", sign_in_button)
             logger.debug("Sign in button clicked via JS.")
             return True
        except Exception as click_e:
             logger.error(f"Failed to click sign in button: {click_e}", exc_info=True)
             return False

    except TimeoutException:
        logger.error("Username or password input field not found or visible within timeout.")
        return False
    except NoSuchElementException: # Should be caught by Wait, but safety check
        logger.error("Username or password input field not found (NoSuchElement).")
        return False
    except ValueError as ve: # Catch missing config error
         logger.critical(f"Configuration Error: {ve}") # Use CRITICAL
         return False
    except WebDriverException as e: # Catch general WebDriver errors
         logger.error(f"WebDriver error entering credentials: {e}")
         temp_sm = SessionManager() # Create temporary manager
         temp_sm.driver = driver
         if not temp_sm.is_sess_valid(): # Check session validity
              logger.error("Session invalid during credential entry.")
         return False
    except Exception as e:
        logger.error(f"Unexpected error entering credentials: {e}", exc_info=True)
        return False
# End of enter_creds

# Login 3
@retry(MAX_RETRIES=2, BACKOFF_FACTOR=1, MAX_DELAY=3) # Add retry for consent handling
def consent(driver: WebDriver) -> bool: # Type hint driver and return
    """Handles cookie consent modal by removing it if present, with enhanced logging and retry."""
    try:
        logger.debug("Checking for cookie consent overlay")
        # Use a short wait to find the element
        overlay = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR))
        )
        # If found, attempt removal
        logger.debug("Cookie consent overlay DETECTED.")
        logger.debug("Attempting to remove cookie consent overlay using Javascript...")
        driver.execute_script("arguments[0].remove();", overlay)
        # Optional: Verify removal - check if element is gone after short pause
        time.sleep(0.5)
        try:
            driver.find_element(By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
            logger.warning("Consent overlay still present after JS removal attempt.")
            # Could try clicking an accept button here as fallback if JS fails
            # Example: driver.find_element(By.CSS_SELECTOR, ACCEPT_BUTTON_SELECTOR).click()
            return False # Indicate potential failure if still present
        except NoSuchElementException:
             logger.debug("Cookie consent overlay REMOVED successfully via Javascript.")
             return True # Successfully removed

    except TimeoutException: # Element not found within the short wait
        logger.debug("Cookie consent overlay not found. Assuming no consent needed.")
        return True
    except NoSuchElementException: # Should be caught by Wait, but safety check
        logger.debug("Cookie consent overlay not found (NoSuchElement).")
        return True
    except WebDriverException as e:
         logger.error(f"WebDriver error handling cookie consent: {e}")
         temp_sm = SessionManager() # Create temporary manager
         temp_sm.driver = driver
         if not temp_sm.is_sess_valid():
              logger.error("Session invalid during consent handling.")
         return False # Fail on WebDriver errors during consent
    except Exception as e:
        logger.error(f"Unexpected error handling cookie consent overlay: {e}", exc_info=True)
        return False # Fail on other errors
# End of consent

# Login 2
def log_in(session_manager: SessionManager) -> str: # Return specific status strings
    """Logs in to Ancestry.com, handling 2-step verification if needed. Returns status string."""
    driver = session_manager.driver
    if not driver:
         logger.error("Login failed: WebDriver not available.")
         return "LOGIN_ERROR_NO_DRIVER"

    page_wait = selenium_config.page_wait(driver)

    try:
        # 1. Navigate to signin page
        signin_url = urljoin(config_instance.BASE_URL, "account/signin")
        logger.debug(f"Navigating to signin page: {signin_url}")
        # Use nav_to_page for robustness
        if not nav_to_page(driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager):
            logger.error("Failed to navigate to or load the login page.")
            return "LOGIN_FAILED_NAVIGATION"

        # 2. Handle cookie consent (banner overlay) - if present
        if not consent(driver):
             logger.warning("Failed to handle consent banner, login might be impacted.")
             # Continue cautiously

        # 3. Enter login credentials and submit
        if not enter_creds(driver):
             logger.error("Failed to enter login credentials.")
             return "LOGIN_FAILED_CREDS_ENTRY" # More specific error

        logger.debug("Credentials submitted. Waiting for potential page change...")
        # Wait briefly for redirection or 2FA page to appear
        time.sleep(random.uniform(2.0, 4.0)) # Random pause

        # 4. Check for 2-step verification AFTER submitting credentials
        # Use a slightly longer wait here as the page might take time to load
        two_fa_present = is_elem_there(driver, By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR, wait=5)

        if two_fa_present:
            logger.info("Two-step verification page detected.")
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

        # 5. If no 2-step verification, check for successful login directly
        else:
             logger.debug("Two-step verification page not detected. Checking login status directly.")
             login_check_result = login_status(session_manager)
             if login_check_result is True:
                 logger.info("Login successful (no 2FA required).")
                 return "LOGIN_SUCCEEDED"
             elif login_check_result is False:
                 # Check for specific login error messages on the page
                 # Use a generic selector for alerts, as FAILED_LOGIN_SELECTOR might be too specific
                 login_error_alert_selector = "div.alert[role='alert']" # More generic alert selector
                 try:
                      if is_elem_there(driver, By.CSS_SELECTOR, login_error_alert_selector, wait=1):
                           error_msg_element = driver.find_element(By.CSS_SELECTOR, login_error_alert_selector)
                           error_msg = error_msg_element.text if error_msg_element else "Unknown error message"
                           logger.error(f"Login failed. Error message found: '{error_msg}'")
                           # Check if it looks like invalid credentials
                           if "invalid credentials" in error_msg.lower() or "username or password invalid" in error_msg.lower():
                                return "LOGIN_FAILED_BAD_CREDS" # Specific error for bad credentials
                           else:
                                return "LOGIN_FAILED_ERROR_DISPLAYED" # Other error displayed
                 except Exception:
                      pass # Ignore errors checking for specific message
                 logger.error("Login failed: Login status check returned False (no 2FA, no specific error msg found).")
                 return "LOGIN_FAILED_UNKNOWN" # Generic failure
             else: # login_status returned None (critical error)
                 logger.error("Login failed: Critical error during final login status check.")
                 return "LOGIN_FAILED_STATUS_CHECK_ERROR"

    except TimeoutException as e:
        logger.error(f"Timeout during login process: {e}")
        return "LOGIN_FAILED_TIMEOUT"
    except WebDriverException as e:
         logger.error(f"WebDriverException during login: {e}")
         if not session_manager.is_sess_valid():
              logger.error("Session became invalid during login attempt.")
         return "LOGIN_FAILED_WEBDRIVER"
    except Exception as e:
        logger.error(f"An unexpected error occurred during login: {e}", exc_info=True)
        return "LOGIN_FAILED_UNEXPECTED"
# End of log_in

# Login 1 - REVISED: Prioritize API Check
@retry(MAX_RETRIES=2)
def login_status(session_manager: SessionManager) -> Optional[bool]:
    """
    REVISED: Checks if the user appears logged in. Prioritizes API verification,
    falls back to UI element check if API fails.
    Returns True if likely logged in, False if likely not, None if critical error occurs.
    """
    # logger.debug("Checking login status (API prioritized)...") # Less verbose
    api_check_result: Optional[bool] = None
    ui_check_result: Optional[bool] = None

    try:
        if not isinstance(session_manager, SessionManager):
            logger.error(f"Invalid argument: Expected SessionManager, got {type(session_manager)}.")
            return None

        # --- 1. Basic WebDriver Session Check ---
        if not session_manager.is_sess_valid():
            # logger.debug("Login status: Session invalid or browser closed.") # Less verbose
            return False # Session definitely not valid/logged in

        driver = session_manager.driver # Assume driver exists if session is valid

        # --- 2. API-Based Login Verification (PRIORITY) ---
        # logger.debug("Attempting API login verification...") # Less verbose
        api_check_result = session_manager._verify_api_login_status() # Handles its own logging

        if api_check_result is True:
            # logger.debug("Login status confirmed via API check.") # Less verbose
            # Optional safety check: Quickly ensure UI isn't showing explicit logout state
            try:
                 if is_elem_there(driver, By.CSS_SELECTOR, LOG_IN_BUTTON_SELECTOR, wait=1):
                      logger.warning("API login check passed, but UI shows 'Log In' button. State mismatch?")
                      # Trust API but warn.
                 # else: Pass # API ok, UI doesn't show login button -> Good state
                 return True # Return True based on API success
            except Exception as ui_safety_check_e:
                 logger.warning(f"Error during UI safety check after successful API check: {ui_safety_check_e}")
                 return True # Proceed based on API check

        elif api_check_result is False:
            logger.debug("API login verification failed. Falling back to UI check.")
            # Proceed to UI check below
        # else: # Should not happen if _verify_api_login_status returns True/False
        #    logger.error("API login verification returned unexpected value. Falling back to UI check.")
        # Proceed to UI check below

        # --- 3. UI-Based Login Verification (FALLBACK) ---
        logger.debug("Performing fallback UI login check...")

        # --- 3a. Handle Consent Overlay ---
        # Use short wait, fail silently if error occurs during consent check
        try: consent(driver)
        except Exception: pass

        # --- 3b. Check for Logged-In UI Element ---
        logged_in_selector = CONFIRMED_LOGGED_IN_SELECTOR
        # logger.debug(f"Attempting UI verification using selector: '{logged_in_selector}'") # Less verbose
        ui_element_present = is_elem_there(driver, By.CSS_SELECTOR, logged_in_selector, wait=5) # Use configured wait

        if ui_element_present:
            logger.debug("Login status confirmed via fallback UI check.")
            session_manager.api_login_verified = True # Update flag based on UI success
            return True
        else:
            # logger.debug("Login confirmation UI element not found in fallback check.") # Less verbose
            # --- 3c. Check for Logged-Out UI Element ---
            login_button_selector = LOG_IN_BUTTON_SELECTOR
            login_button_present = is_elem_there(driver, By.CSS_SELECTOR, login_button_selector, wait=1)

            if login_button_present:
                 # logger.debug("Login status confirmed NOT logged in via fallback UI check ('Log In' button found).") # Less verbose
                 session_manager.api_login_verified = False
                 return False
            else:
                 # --- 3d. Handle Ambiguity ---
                 logger.warning("Login status ambiguous: API failed and neither confirmation nor login UI elements found.")
                 # Try getting current URL as context clue
                 current_url_context = "Unknown"
                 try: current_url_context = driver.current_url
                 except Exception: pass
                 logger.debug(f"Ambiguous login state at URL: {current_url_context}")
                 # Defaulting to False in ambiguous state after API failure seems safest
                 session_manager.api_login_verified = False
                 return False

    # --- Handle Exceptions ---
    except WebDriverException as e:
         logger.error(f"WebDriverException during login_status check: {e}")
         if session_manager and not session_manager.is_sess_valid():
              logger.error("Session became invalid during login_status check.")
              session_manager.close_sess() # Ensure cleanup if session died
         return None # Return None for critical errors
    except Exception as e:
        logger.critical(f"CRITICAL Unexpected error during login_status check: {e}", exc_info=True)
        return None # Return None for critical errors
# End of login_status


# ------------------------------------------------------------------------------------
# browser
# ------------------------------------------------------------------------------------

def is_browser_open(driver: Optional[WebDriver]) -> bool: # Type hint driver
    """
    Checks if the browser window associated with the driver instance is open.
    """
    if driver is None:
        return False
    try:
        # Accessing window_handles is a lightweight way to check if the browser session is active
        _ = driver.window_handles
        return True # If no exception, browser is responsive
    except WebDriverException as e:
         # Specific exceptions indicating closed browser/session
         if "invalid session id" in str(e).lower() or \
            "target closed" in str(e).lower() or \
            "disconnected" in str(e).lower():
            # logger.debug(f"Browser appears closed or session invalid: {e}") # Less verbose
            return False
         else:
              # Other WebDriver errors might occur even if browser is open
              logger.warning(f"WebDriverException checking browser status (but might still be open): {e}")
              # Cautiously return True unless it's a known "closed" error? Or False? Let's be cautious.
              return False
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error checking browser status: {e}", exc_info=True)
        return False
# End of is_browser_open

def restore_sess(driver: WebDriver) -> bool: # Type hint driver & return
    """Restores session state including cookies, local, and session storage, domain aware."""
    if not driver:
        logger.error("Cannot restore session: WebDriver is None.")
        return False

    cache_dir = config_instance.CACHE_DIR
    domain = "" # Initialize domain

    try:
        current_url = driver.current_url
        parsed_url = urlparse(current_url)
        domain = parsed_url.netloc.replace("www.", "").split(":")[0]
        if not domain: # Handle case where domain parsing fails
             raise ValueError("Could not extract valid domain from current URL")
    except Exception as e:
        logger.error(f"Could not parse current URL ({driver.current_url if driver else 'N/A'}) for session restore: {e}. Using fallback.")
        try:
            parsed_base = urlparse(config_instance.BASE_URL)
            domain = parsed_base.netloc.replace("www.", "").split(":")[0]
            if not domain: raise ValueError("Could not extract domain from BASE_URL")
            logger.warning(f"Falling back to base URL domain for session restore: {domain}")
        except Exception as base_e:
            logger.critical(f"Could not determine domain from base URL either: {base_e}. Cannot restore.")
            return False

    logger.debug(f"Attempting to restore session state from cache for domain: {domain}...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    restored_something = False

    def safe_json_read(filename: str) -> Optional[Any]:
        """Helper function for safe JSON reading."""
        filepath = cache_dir / filename
        if filepath.exists():
            try:
                with filepath.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Skipping restore from '{filepath.name}': {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error reading '{filepath.name}': {e}", exc_info=True)
                return None
        return None

    # Restore Cookies
    cookies_file = f"session_cookies_{domain}.json"
    cookies = safe_json_read(cookies_file)
    if cookies and isinstance(cookies, list):
        try:
            # logger.debug(f"Adding {len(cookies)} cookies from cache...") # Less verbose
            driver.delete_all_cookies()
            count = 0
            for cookie in cookies:
                # Basic validation of cookie structure
                if isinstance(cookie, dict) and "name" in cookie and "value" in cookie and "domain" in cookie:
                    try: driver.add_cookie(cookie) ; count += 1
                    except WebDriverException as e: logger.warning(f"Skipping cookie '{cookie.get('name', '??')}': {e}")
            logger.debug(f"Added {count} cookies.")
            restored_something = True
        except WebDriverException as e: logger.error(f"WebDriver error during cookie restore: {e}")
        except Exception as e: logger.error(f"Unexpected error during cookie restore: {e}", exc_info=True)
    elif cookies is not None: logger.warning(f"Cookie cache file '{cookies_file}' invalid format.")

    # Restore Local Storage
    local_storage_file = f"session_local_storage_{domain}.json"
    local_storage = safe_json_read(local_storage_file)
    if local_storage and isinstance(local_storage, dict):
        try:
            # logger.debug(f"Restoring {len(local_storage)} items into localStorage...") # Less verbose
            script = """
                 var data = arguments[0]; localStorage.clear();
                 for (var key in data) { if (data.hasOwnProperty(key)) {
                     try { localStorage.setItem(key, data[key]); } catch (e) { console.warn('Failed localStorage setItem:', key, e); }
                 }} return localStorage.length;"""
            driver.execute_script(script, local_storage)
            logger.debug("localStorage restored.")
            restored_something = True
        except WebDriverException as e: logger.error(f"Error restoring localStorage: {e}")
        except Exception as e: logger.error(f"Unexpected error restoring localStorage: {e}", exc_info=True)
    elif local_storage is not None: logger.warning(f"localStorage cache file '{local_storage_file}' invalid format.")

    # Restore Session Storage
    session_storage_file = f"session_session_storage_{domain}.json"
    session_storage = safe_json_read(session_storage_file)
    if session_storage and isinstance(session_storage, dict):
        try:
            # logger.debug(f"Restoring {len(session_storage)} items into sessionStorage...") # Less verbose
            script = """
                 var data = arguments[0]; sessionStorage.clear();
                 for (var key in data) { if (data.hasOwnProperty(key)) {
                     try { sessionStorage.setItem(key, data[key]); } catch (e) { console.warn('Failed sessionStorage setItem:', key, e); }
                 }} return sessionStorage.length;"""
            driver.execute_script(script, session_storage)
            logger.debug("sessionStorage restored.")
            restored_something = True
        except WebDriverException as e: logger.error(f"Error restoring sessionStorage: {e}")
        except Exception as e: logger.error(f"Unexpected error restoring sessionStorage: {e}", exc_info=True)
    elif session_storage is not None: logger.warning(f"sessionStorage cache file '{session_storage_file}' invalid format.")

    # Perform hard refresh if anything was restored
    if restored_something:
        logger.info("Session state restored from cache. Performing hard refresh...")
        try:
            driver.refresh()
            WebDriverWait(driver, selenium_config.PAGE_TIMEOUT).until( # Wait for load
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            logger.debug("Hard refresh completed.")
        except TimeoutException: logger.warning("Timeout waiting for page load after hard refresh.")
        except WebDriverException as e: logger.warning(f"Error during hard refresh: {e}")
        return True
    else:
        logger.debug("No session state found in cache to restore.")
        return False
# end of restore_sess

def save_state(driver: WebDriver):
    """Saves session state (cookies, local/session storage) to domain-specific files."""
    if not is_browser_open(driver):
        logger.warning("Browser session invalid/closed. Skipping session state save.")
        return

    cache_dir = config_instance.CACHE_DIR
    domain = "" # Initialize

    try:
        current_url = driver.current_url
        parsed_url = urlparse(current_url)
        domain = parsed_url.netloc.replace("www.", "").split(":")[0]
        if not domain: raise ValueError("Could not extract domain from current URL")
    except Exception as e:
        logger.error(f"Could not parse current URL ({driver.current_url if driver else 'N/A'}) for saving state: {e}. Using fallback.")
        try:
            parsed_base = urlparse(config_instance.BASE_URL)
            domain = parsed_base.netloc.replace("www.", "").split(":")[0]
            if not domain: raise ValueError("Could not extract domain from BASE_URL")
            logger.warning(f"Falling back to base URL domain for saving state: {domain}")
        except Exception as base_e:
            logger.critical(f"Could not determine domain from base URL either: {base_e}. Cannot save state.")
            return

    logger.debug(f"Saving session state for domain: {domain}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def safe_json_write(data: Any, filename: str) -> bool:
        """Helper function for safe JSON writing."""
        filepath = cache_dir / filename
        try:
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except (TypeError, IOError) as e: logger.error(f"Error writing JSON to '{filepath.name}': {e}") ; return False
        except Exception as e: logger.error(f"Unexpected error writing JSON to '{filepath.name}': {e}", exc_info=True) ; return False

    # Save Cookies
    cookies_file = f"session_cookies_{domain}.json"
    try:
        cookies = driver.get_cookies()
        if cookies is not None:
            if safe_json_write(cookies, cookies_file): logger.debug(f"Cookies saved.") # Less verbose
        else: logger.warning("Could not retrieve cookies from driver.")
    except WebDriverException as e: logger.error(f"Error getting cookies: {e}")
    except Exception as e: logger.error(f"Unexpected error saving cookies: {e}", exc_info=True)

    # Save Local Storage
    local_storage_file = f"session_local_storage_{domain}.json"
    try:
        local_storage = driver.execute_script("return window.localStorage ? {...window.localStorage} : {};") # Return empty dict if null
        if local_storage: # Check if not empty
            if safe_json_write(local_storage, local_storage_file): logger.debug(f"localStorage saved.")
        else: logger.debug("localStorage not available or empty.")
    except WebDriverException as e: logger.error(f"Error getting localStorage: {e}")
    except Exception as e: logger.error(f"Unexpected error saving localStorage: {e}", exc_info=True)

    # Save Session Storage
    session_storage_file = f"session_session_storage_{domain}.json"
    try:
        session_storage = driver.execute_script("return window.sessionStorage ? {...window.sessionStorage} : {};") # Return empty dict if null
        if session_storage:
            if safe_json_write(session_storage, session_storage_file): logger.debug(f"sessionStorage saved.")
        else: logger.debug("sessionStorage not available or empty.")
    except WebDriverException as e: logger.error(f"Error getting sessionStorage: {e}")
    except Exception as e: logger.error(f"Unexpected error saving sessionStorage: {e}", exc_info=True)

    # logger.debug(f"Session state save attempt finished for domain: {domain}.") # Less verbose
# End of save_state

def close_tabs(driver: WebDriver): # Type hint driver
    """Closes all but the first tab in the given driver."""
    if not driver: return
    logger.debug("Closing extra tabs...")
    try:
        handles = driver.window_handles
        if len(handles) <= 1:
             logger.debug("No extra tabs to close.")
             return

        original_handle = driver.current_window_handle # Store current handle
        first_handle = handles[0]

        # Close all tabs except the first one
        for handle in handles[1:]:
            try:
                driver.switch_to.window(handle)
                driver.close()
                logger.debug(f"Closed tab handle: {handle}")
            except NoSuchWindowException:
                 logger.warning(f"Tab {handle} already closed.")
            except WebDriverException as e:
                 logger.error(f"Error closing tab {handle}: {e}")

        # Switch back to the first tab (or original if it was the first)
        # Check if original handle still exists (it should if it was the first)
        remaining_handles = driver.window_handles
        if original_handle in remaining_handles:
            driver.switch_to.window(original_handle)
            logger.debug(f"Switched back to original tab: {original_handle}")
        elif first_handle in remaining_handles: # Fallback to first handle if original was closed
             driver.switch_to.window(first_handle)
             logger.debug(f"Switched back to first tab: {first_handle}")
        elif remaining_handles: # Switch to whatever is left if both original and first are gone
             driver.switch_to.window(remaining_handles[0])
             logger.warning(f"Original and first tab closed, switched to remaining: {remaining_handles[0]}")
        else:
             logger.error("All tabs were closed unexpectedly.")
             # This indicates a problem, maybe the driver session died.

    except NoSuchWindowException:
        logger.warning("Attempted to close or switch to a tab that no longer exists during cleanup.")
    except WebDriverException as e:
         logger.error(f"WebDriverException in close_tabs: {e}")
         temp_sm = SessionManager() # Create temporary manager
         temp_sm.driver = driver
         if not temp_sm.is_sess_valid():
              logger.error("Session invalid during close_tabs.")
    except Exception as e:
        logger.error(f"Unexpected error in close_tabs: {e}", exc_info=True)
# end close_tabs


# ------------------------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------------------------

def nav_to_page(
    driver: WebDriver,
    url: str, # url defaults removed, must be provided
    selector: str = "body", # Default selector changed to body
    session_manager: Optional[SessionManager] = None,
) -> bool:
    """
    Navigates to a URL, handles temporary unavailability, login redirection,
    and verifies successful navigation by checking URL and waiting for a selector.
    Requires SessionManager for potential relogin.
    """
    if not driver:
        logger.error("Navigation failed: WebDriver instance is None.")
        return False
    if not url:
         logger.error("Navigation failed: Target URL is required.")
         return False

    # Use configured retry and timeout values
    max_attempts = config_instance.MAX_RETRIES
    element_timeout = selenium_config.ELEMENT_TIMEOUT

    # Normalize target URL base for comparison
    target_url_parsed = urlparse(url)
    target_url_base = f"{target_url_parsed.scheme}://{target_url_parsed.netloc}{target_url_parsed.path}".rstrip('/')

    # Define selectors/messages indicating page unavailability or errors
    unavailability_selectors = {
        TEMP_UNAVAILABLE_SELECTOR: ("refresh", 5),
        PAGE_NO_LONGER_AVAILABLE_SELECTOR: ("skip", 0),
        # Add other known error patterns if needed
    }

    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Nav Attempt {attempt}/{max_attempts} to: {url}")
        is_blocked = False

        try:
            # --- 1. Pre-Navigation Checks (Login/MFA/Current URL) ---
            if not _pre_navigation_checks(driver, session_manager):
                 # _pre_navigation_checks logs errors, attempt restart or fail
                 if session_manager:
                      logger.warning("Pre-navigation check failed. Attempting session restart...")
                      if session_manager.restart_sess():
                           logger.info("Session restarted. Retrying navigation from scratch...")
                           driver = session_manager.driver # Get new driver instance
                           if not driver: return False # Restart failed badly
                           continue # Restart loop with new driver
                      else:
                           logger.error("Session restart failed. Aborting navigation.")
                           return False
                 else: # No session manager to restart
                      return False # Fail if pre-checks fail without restart capability

            # --- 2. Perform Navigation ---
            driver.get(url)

            # --- 3. Post-Navigation Verification ---
            # Wait briefly for initial loading/redirects
            time.sleep(random.uniform(0.5, 1.5))

            # Verify URL after navigation attempt
            post_nav_url = "" # Initialize
            try:
                post_nav_url = driver.current_url
                post_nav_url_parsed = urlparse(post_nav_url)
                post_nav_url_base = f"{post_nav_url_parsed.scheme}://{post_nav_url_parsed.netloc}{post_nav_url_parsed.path}".rstrip('/')
                # logger.debug(f"Nav Attempt {attempt} landed on: {post_nav_url}") # Less verbose
            except WebDriverException as e:
                logger.error(f"Failed to get URL after get() (Attempt {attempt}): {e}. Retrying.")
                continue # Retry the whole navigation attempt

            # Re-check for login/mfa immediately
            if _check_post_nav_redirects(post_nav_url):
                 continue # Redirected, loop will retry pre-checks

            # Check if landed on the target URL (handle variations)
            if post_nav_url_base != target_url_base:
                 logger.warning(f"Navigation landed on unexpected URL base: '{post_nav_url_base}' (Expected: '{target_url_base}')")
                 # Check for error messages on the unexpected page
                 action, wait_time = _check_for_unavailability(driver, unavailability_selectors)
                 if action == "skip": return False # Permanent failure
                 elif action == "refresh":
                      logger.warning(f"Waiting {wait_time}s due to unavailability message on wrong URL.")
                      time.sleep(wait_time)
                      continue # Retry navigation attempt
                 else: # No specific message, but wrong URL
                      logger.warning("Wrong URL and no specific error message. Retrying navigation.")
                      continue

            # --- 4. Wait for Target Selector ---
            wait_selector = selector if selector else "body" # Default to body
            # logger.debug(f"Waiting up to {element_timeout}s for selector: '{wait_selector}'") # Less verbose
            try:
                WebDriverWait(driver, element_timeout).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, wait_selector))
                )
                logger.debug(f"Navigation to '{url}' successful (Selector '{wait_selector}' found).")
                return True # SUCCESS!

            except TimeoutException:
                # Timeout waiting for selector
                current_url_on_timeout = "N/A"
                try: current_url_on_timeout = driver.current_url
                except Exception: pass
                logger.warning(f"Timeout waiting for selector '{wait_selector}' at {current_url_on_timeout}.")

                # Check for unavailability messages AFTER timeout
                action, wait_time = _check_for_unavailability(driver, unavailability_selectors)
                if action == "skip": return False # Permanent failure
                elif action == "refresh":
                    logger.warning(f"Page unavailable after timeout. Waiting {wait_time}s before retry.")
                    time.sleep(wait_time)
                    continue # Retry navigation attempt
                else: # Generic timeout, no specific message
                    logger.warning(f"Timeout waiting for selector '{wait_selector}', no unavailability message found. Retrying navigation.")
                    continue # Retry navigation attempt

        # --- Handle Exceptions During the Attempt ---
        except UnexpectedAlertPresentException as alert_e:
            logger.warning(f"Unexpected alert (Attempt {attempt}): {alert_e.alert_text}")
            try: driver.switch_to.alert.accept() ; logger.info("Alert accepted.")
            except Exception as accept_e: logger.error(f"Failed to accept alert: {accept_e}") ; return False
            continue # Retry navigation

        except WebDriverException as wd_e:
            logger.error(f"WebDriverException (Attempt {attempt}): {wd_e}", exc_info=True) # Log full traceback for WebDriverException
            if session_manager and not session_manager.is_sess_valid():
                logger.error("WebDriver session appears invalid. Attempting restart...")
                if session_manager.restart_sess():
                    logger.info("Session restarted. Retrying navigation...")
                    driver = session_manager.driver # Update driver reference
                    if not driver: return False
                    continue
                else: logger.error("Session restart failed.") ; return False
            else: logger.warning("WebDriverException, session seems valid. Waiting before retry.") ; time.sleep(3) ; continue

        except Exception as e:
            logger.error(f"Unexpected error (Attempt {attempt}): {e}", exc_info=True)
            time.sleep(3)
            continue # Retry the attempt

    # --- End of Retry Loop ---
    logger.critical(f"Navigation to '{url}' failed permanently after {max_attempts} attempts.")
    try: logger.error(f"Final URL after failure: {driver.current_url}")
    except Exception: logger.error("Could not retrieve final URL after failure.")
    return False

# --- Helper functions for nav_to_page ---

def _pre_navigation_checks(driver: WebDriver, session_manager: Optional[SessionManager]) -> bool:
     """Performs pre-navigation checks for login status and MFA."""
     try:
          current_url = driver.current_url
     except WebDriverException as e:
          logger.error(f"Failed to get current URL in pre-check: {e}.")
          return False # Cannot proceed without knowing current state

     login_url_base = urljoin(config_instance.BASE_URL, "account/signin")
     mfa_url_base = urljoin(config_instance.BASE_URL, "account/signin/mfa/")

     if current_url.startswith(mfa_url_base):
          logger.warning("Pre-check failed: Currently on MFA page.")
          return False

     if current_url.startswith(login_url_base):
          logger.warning("Pre-check: On login page. Attempting re-login...")
          if session_manager:
                login_result = log_in(session_manager)
                if login_result == "LOGIN_SUCCEEDED":
                     logger.info("Re-login successful during pre-check.")
                     # Need to verify URL *after* re-login again
                     try:
                          time.sleep(1) # Pause after login
                          post_login_url = driver.current_url
                          if post_login_url.startswith(login_url_base) or post_login_url.startswith(mfa_url_base):
                               logger.error("Still on login/MFA page after re-login attempt.")
                               return False
                          return True # Re-login appears successful
                     except WebDriverException as post_login_e:
                          logger.error(f"Error getting URL after re-login: {post_login_e}")
                          return False
                else:
                     logger.error(f"Re-login failed ({login_result}).")
                     return False
          else:
                logger.error("On login page, but no SessionManager provided for re-login.")
                return False
     return True # Passed pre-checks
# End ofnav_to_page

def _check_post_nav_redirects(post_nav_url: str) -> bool:
     """Checks if the URL after navigation is a login or MFA page."""
     login_url_base = urljoin(config_instance.BASE_URL, "account/signin")
     mfa_url_base = urljoin(config_instance.BASE_URL, "account/signin/mfa/")
     if post_nav_url.startswith(mfa_url_base):
          logger.warning("Redirected to MFA page immediately after navigation.")
          return True # Indicates redirect occurred
     if post_nav_url.startswith(login_url_base):
          logger.warning("Redirected back to login page immediately after navigation.")
          return True # Indicates redirect occurred
     return False
# end _check_post_nav_redirects

def _check_for_unavailability(driver: WebDriver, selectors: Dict[str, Tuple[str, int]]) -> Tuple[Optional[str], int]:
     """Checks for unavailability messages on the current page."""
     for msg_selector, (action, wait_time) in selectors.items():
          if is_elem_there(driver, By.CSS_SELECTOR, msg_selector, wait=0.5): # Quick check
                logger.warning(f"Unavailability message found: '{msg_selector}' Action: {action}")
                return action, wait_time
     return None, 0
# End of_check_for_unavailability


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
    log_filename_only = db_file_path.with_suffix(".log").name
    setup_logging(log_file=log_filename_only, log_level="DEBUG")
    logger.info(f"--- Starting utils.py standalone test run ---")

    session_manager = SessionManager()
    test_success = True

    try:
        # --- Test Session Start (includes identifier retrieval) ---
        logger.info("--- Testing SessionManager.start_sess() ---")
        start_ok, driver_instance = session_manager.start_sess(action_name="Utils Test")
        if not start_ok or not driver_instance:
            logger.error("SessionManager.start_sess() FAILED. Aborting further tests.")
            test_success = False
            return # Use return for cleaner exit in tests
        else:
            logger.info("SessionManager.start_sess() PASSED.")

            # --- Verify Identifiers ---
            logger.info("--- Verifying Identifiers (Retrieved during start_sess) ---")
            errors = []
            if not session_manager.my_profile_id: errors.append("my_profile_id")
            if not session_manager.my_uuid: errors.append("my_uuid")
            if config_instance.TREE_NAME and not session_manager.my_tree_id: errors.append("my_tree_id (required by config)")
            if not session_manager.csrf_token: errors.append("csrf_token")
            if errors:
                logger.error(f"FAILED to retrieve required identifiers: {', '.join(errors)}")
                test_success = False
            else:
                logger.info("All required identifiers retrieved successfully.")
                logger.debug(f" Profile ID: {session_manager.my_profile_id}")
                logger.debug(f" UUID: {session_manager.my_uuid}")
                logger.debug(f" Tree ID: {session_manager.my_tree_id or 'N/A'}")
                logger.debug(f" Tree Owner: {session_manager.tree_owner_name or 'N/A'}")
                logger.debug(f" CSRF Token: {session_manager.csrf_token[:10]}...")

            # --- Test Navigation ---
            logger.info("--- Testing Navigation (nav_to_page to BASE_URL) ---")
            nav_ok = nav_to_page(driver=driver_instance, url=config_instance.BASE_URL, selector="body", session_manager=session_manager)
            if nav_ok:
                logger.info("nav_to_page() to BASE_URL PASSED.")
                try:
                    current_url_after_nav = driver_instance.current_url
                    if current_url_after_nav.startswith(config_instance.BASE_URL.rstrip('/')):
                        logger.info(f"Successfully landed on expected base URL: {current_url_after_nav}")
                    else:
                        logger.warning(f"nav_to_page() to base URL landed on slightly different URL: {current_url_after_nav}")
                except Exception as e: logger.warning(f"Could not verify URL after nav_to_page: {e}")
            else:
                logger.error("nav_to_page() to BASE_URL FAILED.")
                test_success = False

            # --- Test API Request Helper (_api_req via CSRF endpoint) ---
            logger.info("--- Testing API Request (_api_req via CSRF endpoint) ---")
            csrf_url = urljoin(config_instance.BASE_URL, "discoveryui-matches/parents/api/csrfToken")
            csrf_test_response = _api_req(url=csrf_url, driver=driver_instance, session_manager=session_manager, method="GET", use_csrf_token=False, api_description="CSRF Token API")
            if csrf_test_response and ((isinstance(csrf_test_response, dict) and "csrfToken" in csrf_test_response) or (isinstance(csrf_test_response, str) and len(csrf_test_response) > 10)):
                logger.info("CSRF Token API call via _api_req PASSED.")
                token_val = csrf_test_response['csrfToken'] if isinstance(csrf_test_response, dict) else csrf_test_response
                logger.debug(f"CSRF Token API test retrieved: {str(token_val)[:10]}...")
            else:
                logger.error("CSRF Token API call via _api_req FAILED.")
                logger.debug(f"Response received: {csrf_test_response}")
                test_success = False

            # --- Test Browser Tab Management ---
            logger.info("--- Testing Tab Management (make_tab, close_tabs) ---")
            logger.info("Creating a new tab...")
            new_tab_handle = session_manager.make_tab()
            if new_tab_handle:
                logger.info(f"make_tab() PASSED. New handle: {new_tab_handle}")
                logger.info("Navigating new tab to example.com...")
                try:
                    driver_instance.switch_to.window(new_tab_handle)
                    # Use nav_to_page for navigation robustness
                    if nav_to_page(driver_instance, "https://example.com", selector="body", session_manager=session_manager):
                        logger.info("Navigation in new tab successful.")
                        logger.info("Closing extra tabs...")
                        close_tabs(driver_instance) # Pass driver explicitly
                        handles_after_close = driver_instance.window_handles
                        if len(handles_after_close) == 1:
                            logger.info("close_tabs() PASSED (one tab remaining).")
                            # Ensure focus is back on the remaining tab
                            if driver_instance.current_window_handle != handles_after_close[0]:
                                 driver_instance.switch_to.window(handles_after_close[0])
                                 logger.debug("Switched focus back to remaining tab.")
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

    except Exception as e:
        logger.critical(f"CRITICAL error during utils.py standalone test execution: {e}", exc_info=True)
        test_success = False

    finally:
        if "session_manager" in locals() and session_manager:
            logger.info("Closing session manager...")
            session_manager.close_sess()

        print("")
        if test_success:
            logger.info("--- Utils.py standalone test run PASSED ---")
        else:
            logger.error("--- Utils.py standalone test run FAILED ---")
# End of main

if __name__ == "__main__":
    main()


# End of utils.py