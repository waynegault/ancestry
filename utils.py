#!/usr/bin/env python3

# # utils.py

"""
utils.py - Standalone login script + utilities using config variables.

This script utilizes configuration variables from config.py (loaded via
config_instance and selenium_config) throughout the login functions and
utility functions, enhancing configurability and maintainability.
"""

# --- Standard library imports ---
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
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any, Generator, Callable
from pathlib import Path

# --- Third-party imports ---
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from urllib.parse import urlparse, urljoin, unquote
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
    NoSuchCookieException,
)
from requests.cookies import RequestsCookieJar
from selenium.webdriver.remote.webdriver import WebDriver
from requests import Response as RequestsResponse, Request
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy import create_engine, event, text, pool as sqlalchemy_pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from requests.exceptions import RequestException, HTTPError
import cloudscraper 

# --- Local application imports ---
from my_selectors import *
from config import config_instance, selenium_config
from database import Base, MessageType
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
        if (
            attribute == "href" and value and value.startswith("/")
        ):  # Added check for value existence
            return urljoin(
                config_instance.BASE_URL, value
            )  # Use urljoin for robustness
        return value if value else ""  # Return empty string if attribute is empty/None
    except NoSuchElementException:  # Catch only NoSuchElementException
        return ""
    except Exception as e:  # Catch other potential errors
        logger.warning(
            f"Error extracting attribute '{attribute}' from selector '{selector}': {e}"
        )
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
                elem
                if (elem := d.find_elements(By.CSS_SELECTOR, PAGINATION_SELECTOR))
                and elem[0].get_attribute("total") is not None
                else None
            )
        )

        if pagination_element is None:
            logger.debug(
                "Pagination element or 'total' attribute not found/available. Assuming 1 page."
            )
            return 1

        total_pages = int(pagination_element[0].get_attribute("total"))
        return total_pages

    except TimeoutException:
        logger.debug(
            f"Timeout waiting for pagination element or 'total' attribute. Assuming 1 page."
        )
        return 1
    except (
        NoSuchElementException,
        IndexError,
    ):  # Handle cases where element disappears or list is empty
        logger.debug(
            "Pagination element not found after initial wait. Assuming 1 page."
        )
        return 1
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting total pages attribute to int: {e}")
        return 1
    except Exception as e:
        logger.error(
            f"Error getting total pages: {e}", exc_info=True
        )  # Log full traceback for unexpected
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


def is_elem_there(driver, by, value, wait=None):  # Use None default for wait
    """
    Checks if an element is present within a specified timeout.
    Uses selenium_config.ELEMENT_TIMEOUT if wait is None.
    """
    if wait is None:
        wait = selenium_config.ELEMENT_TIMEOUT  # Use configured default

    try:
        WebDriverWait(driver, wait).until(EC.presence_of_element_located((by, value)))
        return True
    except TimeoutException:
        # logger.warning(f"Timed out waiting {wait}s for: {value}") # Optional: Less verbose logging
        return False
    except Exception as e:  # Catch other potential errors
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
            if part.isupper() and len(part) > 1:  # Keep multi-letter caps
                formatted_parts.append(part)
            elif part.isupper() and len(part) == 1:  # Keep single initial caps
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
            backoff = (
                config_instance.BACKOFF_FACTOR
                if BACKOFF_FACTOR is None
                else BACKOFF_FACTOR
            )
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
            return None  # Or raise MaxRetriesExceededError("...")

        return wrapper

    return decorator
# end retry


def retry_api(
    max_retries=None,  # Use None to default to config
    initial_delay=None,  # Use None to default to config
    backoff_factor=None,  # Use None to default to config
    retry_on_exceptions=(requests.exceptions.RequestException,),  # Keep default tuple
    retry_on_status_codes=None,  # Use None to default to config
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
            _max_retries = (
                config_instance.MAX_RETRIES if max_retries is None else max_retries
            )
            _initial_delay = (
                config_instance.INITIAL_DELAY
                if initial_delay is None
                else initial_delay
            )  # Use INITIAL_DELAY from config
            _backoff_factor = (
                config_instance.BACKOFF_FACTOR
                if backoff_factor is None
                else backoff_factor
            )
            _retry_codes = (
                config_instance.RETRY_STATUS_CODES
                if retry_on_status_codes is None
                else retry_on_status_codes
            )
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
                        _retry_codes
                        and response is not None
                        and hasattr(response, "status_code")  # Check before accessing
                        and response.status_code in _retry_codes
                    ):
                        should_retry_status = True
                        status_code = response.status_code  # Store for logging

                    if should_retry_status:
                        retries -= 1
                        if retries <= 0:
                            logger.error(
                                f"API Call failed after {_max_retries} retries for function '{func.__name__}' (Final Status {status_code}). No more retries.",
                                exc_info=False,
                            )
                            return response  # Return last failed response for status code retry
                        else:
                            # Calculate sleep time with jitter
                            base_sleep = delay * (
                                _backoff_factor ** (attempt - 1)
                            )  # Apply backoff correctly
                            jitter = (
                                random.uniform(-0.1, 0.1) * delay
                            )  # Smaller jitter?
                            sleep_time = min(
                                base_sleep + jitter, config_instance.MAX_DELAY
                            )  # Cap at MAX_DELAY
                            sleep_time = max(0.1, sleep_time)  # Ensure minimum sleep

                            logger.warning(
                                f"API Call returned retryable status {status_code} (attempt {attempt}/{_max_retries}) for '{func.__name__}', retrying in {sleep_time:.2f} seconds..."
                            )
                            time.sleep(sleep_time)
                            delay *= _backoff_factor  # Increase delay for next potential retry
                            continue  # Go to next retry attempt

                    # If status code doesn't trigger retry, return the response
                    return response

                # Handle exception retry condition SECOND
                except retry_on_exceptions as e:
                    retries -= 1
                    if retries <= 0:
                        logger.error(
                            f"API Call failed after {_max_retries} retries due to exception for function '{func.__name__}'. Final Exception: {e}",
                            exc_info=True,  # Include traceback for exceptions
                        )
                        raise e  # Re-raise the last exception after all retries fail
                    else:
                        # Calculate sleep time with jitter
                        base_sleep = delay * (
                            _backoff_factor ** (attempt - 1)
                        )  # Apply backoff correctly
                        jitter = random.uniform(-0.1, 0.1) * delay
                        sleep_time = min(
                            base_sleep + jitter, config_instance.MAX_DELAY
                        )  # Cap at MAX_DELAY
                        sleep_time = max(0.1, sleep_time)

                        logger.warning(
                            f"API Call failed (attempt {attempt}/{_max_retries}) for '{func.__name__}', retrying in {sleep_time:.2f} seconds... Exception: {type(e).__name__} - {e}"
                        )
                    time.sleep(sleep_time)
                    delay *= _backoff_factor  # Increase delay for next potential retry
                    continue  # Go to next retry attempt

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
            elif "session_manager" in kwargs and isinstance(
                kwargs["session_manager"], SessionManager
            ):
                found_sm = kwargs["session_manager"]

            if not found_sm:
                raise TypeError(
                    f"Function '{func.__name__}' requires a SessionManager instance as the first argument or kwarg."
                )
            session_manager = found_sm  # Use the found instance

        # Now perform the browser check
        if not is_browser_open(session_manager.driver):
            raise WebDriverException(  # Use a more specific exception
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
                    f"Wait '{wait_description}' completed in {duration:.3f} s."  # Added 'Wait' prefix and 'completed'
                )
                return result
            except TimeoutException as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.warning(
                    f"Wait '{wait_description}' timed out after {duration:.3f} seconds.",
                    exc_info=False,  # Less verbose for timeout
                )
                raise e  # Re-raise TimeoutException
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(
                    f"Error during wait '{wait_description}' after {duration:.3f} seconds: {e}",
                    exc_info=True,
                )  # ERROR level for unexpected errors
                raise e  # Re-raise other exceptions

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
        config_instance=config_instance,  # Pass config instance for defaults
    ):
        """
        Initializes the rate limiter using config_instance defaults if not provided.
        """
        self.initial_delay = (
            config_instance.INITIAL_DELAY if initial_delay is None else initial_delay
        )
        self.MAX_DELAY = config_instance.MAX_DELAY if MAX_DELAY is None else MAX_DELAY
        self.backoff_factor = (
            config_instance.BACKOFF_FACTOR if backoff_factor is None else backoff_factor
        )
        self.decrease_factor = (
            config_instance.DECREASE_FACTOR
            if decrease_factor is None
            else decrease_factor
        )
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
        if self.current_delay != self.initial_delay:  # Only log if changed
            self.current_delay = self.initial_delay
            logger.info(
                f"Rate limiter delay reset to initial value: {self.initial_delay:.2f}s"
            )
    # End of reset_delay

    def decrease_delay(self):
        """Gradually reduce the delay when no rate limits are detected."""
        if (
            not self.last_throttled and self.current_delay > self.initial_delay
        ):  # Check last_throttled flag
            previous_delay = self.current_delay
            self.current_delay = max(
                self.current_delay * self.decrease_factor, self.initial_delay
            )
            # Log only if delay actually changed significantly
            if abs(previous_delay - self.current_delay) > 0.01:
                logger.debug(
                    f"No rate limit detected. Decreased delay to {self.current_delay:.2f} seconds."
                )
        self.last_throttled = False  # Reset flag after decrease attempt
    # End of decrease_delay

    def increase_delay(self):
        """Increase the delay exponentially when a rate limit is detected."""
        if (
            not self.last_throttled
        ):  # Only increase if not already throttled in last cycle
            previous_delay = self.current_delay
            self.current_delay = min(
                self.current_delay * self.backoff_factor, self.MAX_DELAY
            )
            logger.info(  # Use INFO for increase as it's significant
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
        self.driver: Optional[WebDriver] = None  # Type hint WebDriver
        self.db_path: str = str(config_instance.DATABASE_FILE.resolve())
        self.selenium_config = selenium_config
        self.ancestry_username: str = (
            config_instance.ANCESTRY_USERNAME
        )  # Ensure string type
        self.ancestry_password: str = (
            config_instance.ANCESTRY_PASSWORD
        )  # Ensure string type
        self.debug_port: int = self.selenium_config.DEBUG_PORT
        self.chrome_user_data_dir: Optional[Path] = (
            self.selenium_config.CHROME_USER_DATA_DIR
        )
        self.profile_dir: str = self.selenium_config.PROFILE_DIR  # Ensure string type
        self.chrome_driver_path: Optional[Path] = (
            self.selenium_config.CHROME_DRIVER_PATH
        )
        self.chrome_browser_path: Optional[Path] = (
            self.selenium_config.CHROME_BROWSER_PATH
        )
        self.chrome_max_retries: int = self.selenium_config.CHROME_MAX_RETRIES
        self.chrome_retry_delay: int = self.selenium_config.CHROME_RETRY_DELAY
        self.headless_mode: bool = self.selenium_config.HEADLESS_MODE
        # --- Refactoring V3 State Flags ---
        self.driver_live: bool = False
        self.session_ready: bool = False
        # --- Initialize DB/Session attributes early ---
        self.engine = None
        self.Session = None
        self._db_init_attempted = False
        # --- Initialize other attributes ---
        self.cache_dir: Path = (
            config_instance.CACHE_DIR
        )  # Keep for now if save/restore uses it
        self.csrf_token: Optional[str] = None
        self.my_profile_id: Optional[str] = None
        self.my_uuid: Optional[str] = None
        self.my_tree_id: Optional[str] = None
        self.tree_owner_name: Optional[str] = None
        self.session_start_time: Optional[float] = None
        self._profile_id_logged = False
        self._uuid_logged = False
        self._tree_id_logged = False
        self._owner_logged = False

        logger.debug(f"SessionManager instance created: ID={id(self)}")

        # --- Initialize requests.Session FIRST ---
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

        # --- Initialize scraper attribute to None BEFORE trying to set it up ---
        self.scraper = None

        # --- Initialize Rate Limiter ---
        self.dynamic_rate_limiter: DynamicRateLimiter = DynamicRateLimiter()
        self.last_js_error_check: datetime = datetime.now()

        # --- Cloudscraper Initialization (MOVED HERE) ---
        try:
            self.scraper = cloudscraper.create_scraper(
                browser={  # Mimic browser headers
                    "browser": "chrome",
                    "platform": "windows",
                    "desktop": True,
                },
                delay=10,  # Add a base delay
            )
            # Configure retry strategy for the scraper's session
            scraper_retry = Retry(
                total=3,
                backoff_factor=0.8,  # Slightly gentler backoff for scraper
                status_forcelist=[
                    429,
                    500,
                    502,
                    503,
                    504,
                    403,
                ],  # Add 403 for CF challenges
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
            self.scraper = None  # Ensure scraper is None if init fails
        # --- END Cloudscraper Initialization ---
    # end of __init__

    def start_sess(self, action_name: Optional[str] = None) -> bool:
        """
        REVISED V3.1: Phase 1 of session start. Initializes driver and navigates to base URL.
        Sets self.driver_live = True on success.
        Does NOT perform login checks or identifier fetching.
        """
        logger.debug(f"--- SessionManager Phase 1: Starting Driver ({action_name}) ---")
        self.driver_live = False  # Reset flag at the start of attempt
        self.session_ready = False # Also reset session ready flag
        self.driver = None         # Ensure driver starts as None

        # Clean up old session state if any existed
        # self.cache.clear() # Removed - use global cache clear if needed externally
        self.csrf_token = None
        self.my_profile_id = None
        self.my_uuid = None
        self.my_tree_id = None
        self.tree_owner_name = None
        self._reset_logged_flags() # Helper to reset logging flags

        # Initialize DB Engine/Session if needed
        if not self.engine or not self.Session:
            try:
                self._initialize_db_engine_and_session()
            except Exception as db_init_e:
                logger.critical(f"DB Initialization failed during start_sess: {db_init_e}")
                return False # Fail Phase 1 if DB can't init

        # Ensure requests session exists
        if not hasattr(self, "_requests_session") or not isinstance(
            self._requests_session, requests.Session
        ):
            self._requests_session = requests.Session()
            logger.debug("requests.Session initialized (fallback).")

        # --- Retry Loop specifically for Driver Initialization ---
        # init_webdvr now handles its own retries internally
        logger.debug("Initializing WebDriver instance (using init_webdvr)...")
        try:
            self.driver = init_webdvr() # No attach_attempt flag needed here
            if not self.driver:
                logger.error("WebDriver initialization failed (init_webdvr returned None).")
                return False # Failed to get driver after retries

            logger.debug("WebDriver initialization successful.")

            # --- Navigate to Base URL to Stabilize ---
            logger.debug("Navigating to Base URL to stabilize initial state...")
            base_url_nav_ok = nav_to_page(
                self.driver,
                config_instance.BASE_URL,
                selector="body", # Wait for body to exist
                session_manager=self,
            )
            if not base_url_nav_ok:
                logger.error("Failed to navigate to Base URL after WebDriver init.")
                self.close_sess() # Clean up driver
                return False

            logger.debug("Initial navigation to Base URL successful.")
            self.driver_live = True # Set flag indicating driver is up and on base URL
            self.session_start_time = time.time()
            self.last_js_error_check = datetime.now()
            logger.debug("--- SessionManager Phase 1: Driver Start Successful ---")
            return True

        except WebDriverException as wd_exc:
            logger.error(f"WebDriverException during driver start/base nav: {wd_exc}", exc_info=False)
            self.close_sess()
            return False
        except Exception as e:
            logger.error(f"Unexpected error during driver start/base nav: {e}", exc_info=True)
            self.close_sess()
            return False
    # end start_sess (Phase 1)

    def ensure_session_ready(self, action_name: Optional[str] = None) -> bool:
        """
        REVISED V3.1: Phase 2 of session start. Ensures user is logged in and identifiers fetched.
        Must be called AFTER start_sess (Phase 1) is successful.
        Sets self.session_ready = True on success.
        """
        logger.debug(f"--- SessionManager Phase 2: Ensuring Session Ready ({action_name}) ---")
        if not self.driver_live or not self.driver:
            logger.error("Cannot ensure session readiness: Driver not live (Phase 1 not run/failed).")
            return False
        if self.session_ready:
            logger.debug("Session already marked as ready. Skipping readiness checks.")
            return True

        # Reset state flags specific to Phase 2
        self.csrf_token = None
        self.my_profile_id = None
        self.my_uuid = None
        self.my_tree_id = None
        self.tree_owner_name = None
        self._reset_logged_flags()

        max_readiness_attempts = 2 # Limit retries for the readiness phase itself
        for attempt in range(1, max_readiness_attempts + 1):
            logger.debug(f"Session readiness attempt {attempt}/{max_readiness_attempts}...")
            try:
                # --- 1. Check Login Status ---
                logger.debug("Checking login status...")
                login_stat = login_status(self) # Uses API check first, then UI

                if login_stat is True:
                    logger.debug("User is logged in.")
                elif login_stat is False:
                    logger.info("User not logged in. Attempting login process...")
                    login_result = log_in(self)
                    if login_result != "LOGIN_SUCCEEDED":
                        logger.error(f"Login failed ({login_result}). Readiness check failed.")
                        # Don't close session here, let caller decide
                        return False
                    logger.info("Login successful.")
                    # Re-verify after login attempt
                    if not login_status(self): # Check again
                        logger.error("Login status verification failed even after successful login attempt reported.")
                        return False
                    logger.debug("Login status re-verified successfully after login.")
                else: # login_stat is None (critical error during check)
                    logger.error(f"Login status check failed critically. Readiness check failed.")
                    return False
                logger.debug("Login status confirmed.")

                # --- 2. URL Check & Correction (Optional but good practice) ---
                logger.debug("Checking current URL validity...")
                if not self._check_and_handle_url():
                    logger.error("URL check/handling failed during readiness check.")
                    return False
                logger.debug("URL check/handling completed.")

                # --- 3. Verify Essential Cookies ---
                logger.debug("Verifying essential cookies (ANCSESSIONID, SecureATT)...")
                essential_cookies = ["ANCSESSIONID", "SecureATT"]
                if not self.get_cookies(essential_cookies, timeout=15):
                    logger.error(f"Essential cookies {essential_cookies} not found. Readiness check failed.")
                    return False
                logger.debug("Essential cookies verified.")

                # --- 4. Synchronize Cookies to requests.Session ---
                logger.debug("Syncing cookies from WebDriver to requests session...")
                self._sync_cookies()
                logger.debug("Cookies synced.")

                # --- 5. Retrieve and Store CSRF Token ---
                logger.debug("Retrieving CSRF token...")
                self.csrf_token = self.get_csrf()
                if not self.csrf_token:
                    logger.error("Failed to retrieve CSRF token. Readiness check failed.")
                    return False
                logger.debug(f"CSRF token retrieved: {self.csrf_token[:10]}...")

                # --- 6. Retrieve User Identifiers ---
                logger.debug("Retrieving user identifiers...")
                if not self._retrieve_identifiers():
                    logger.error("Failed to retrieve essential user identifiers. Readiness check failed.")
                    return False
                logger.debug("Finished retrieving user identifiers.")

                # --- 7. Retrieve Tree Owner Name (if applicable) ---
                logger.debug("Retrieving tree owner name (if applicable)...")
                self._retrieve_tree_owner() # Log internally
                logger.debug("Finished retrieving tree owner name.")

                # --- 8. Mark Session as Ready ---
                self.session_ready = True
                logger.debug("--- SessionManager Phase 2: Session Ready Successful ---")
                return True # Success!

            # --- Handle Exceptions during the attempt ---
            except WebDriverException as wd_exc:
                logger.error(f"WebDriverException during readiness check attempt {attempt}: {wd_exc}", exc_info=False)
                # If session died, we can't continue readiness checks
                if not self.is_sess_valid():
                    logger.error("Session became invalid during readiness check. Aborting.")
                    self.driver_live = False # Mark driver as not live
                    self.session_ready = False
                    self.close_sess() # Clean up
                    return False
                # If session still seems valid, maybe it was a transient error
                if attempt < max_readiness_attempts:
                    logger.info(f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next readiness attempt...")
                    time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                else:
                    logger.error("Readiness check failed after final attempt due to WebDriverException.")
                    return False

            except Exception as e:
                logger.error(f"Unexpected error during readiness check attempt {attempt}: {e}", exc_info=True)
                if attempt < max_readiness_attempts:
                    logger.info(f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next readiness attempt...")
                    time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                else:
                    logger.error("Readiness check failed after final attempt due to unexpected error.")
                    return False

        # --- End of Retry Loop ---
        logger.error(f"Session readiness check FAILED after {max_readiness_attempts} attempts.")
        return False
    # end ensure_session_ready (Phase 2)

    def _reset_logged_flags(self):
        """Helper to reset the logging flags for identifiers."""
        self._profile_id_logged = False
        self._uuid_logged = False
        self._tree_id_logged = False
        self._owner_logged = False
    # end _reset_logged_flags

    def _initialize_db_engine_and_session(self):
        """Initializes the SQLAlchemy engine and session factory with pooling."""
        # Check if already initialized
        if self.engine and self.Session:
            logger.debug(
                f"DB Engine/Session already initialized for SM ID={id(self)}. Skipping."
            )
            return

        logger.debug(f"SessionManager ID={id(self)} attempting DB initialization...")
        self._db_init_attempted = True

        # Dispose existing engine if present (handles re-init scenarios)
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

        try:
            logger.debug(f"Initializing SQLAlchemy Engine for: {self.db_path}")

            # === POOL SIZE LOGIC V4 (copied from previous correct version) ===
            pool_size_env_str = os.getenv("DB_POOL_SIZE")
            pool_size_config = getattr(config_instance, "DB_POOL_SIZE", None)
            logger.debug(
                f"Pool Size Check: Env='{pool_size_env_str}', Config='{pool_size_config}'"
            )

            pool_size_str_to_parse = None
            if pool_size_env_str is not None:
                pool_size_str_to_parse = pool_size_env_str
                logger.debug(
                    f"Pool Size Priority: Using Env value '{pool_size_str_to_parse}' for parsing."
                )
            elif pool_size_config is not None:
                try:
                    pool_size_str_to_parse = str(int(pool_size_config))
                    logger.debug(
                        f"Pool Size Priority: Using Config value '{pool_size_str_to_parse}' for parsing."
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        f"Pool Size Priority: Config value ('{pool_size_config}') invalid, falling back."
                    )
                    pool_size_str_to_parse = "50"
            else:
                logger.debug(
                    "Pool Size Priority: Neither Env nor Config found, using fallback '50'."
                )
                pool_size_str_to_parse = "50"

            pool_size = 20  # Default if parsing fails below
            try:
                parsed_val = int(pool_size_str_to_parse)
                if parsed_val <= 0:
                    logger.warning(
                        f"DB_POOL_SIZE value '{parsed_val}' invalid (<=0). Using default {pool_size}."
                    )
                elif parsed_val == 1:
                    pool_size = 1  # Allow 1 explicitly, though maybe warn
                    logger.warning(
                        f"DB_POOL_SIZE value '1' detected. Using minimal pool size."
                    )
                else:
                    pool_size = min(parsed_val, 100)  # Cap at 100
                    logger.debug(f"Successfully parsed pool size: {pool_size}")
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not parse DB_POOL_SIZE value '{pool_size_str_to_parse}' as integer. Using default {pool_size}."
                )

            max_overflow = max(5, int(pool_size * 0.2))
            pool_timeout = 30
            pool_class = sqlalchemy_pool.QueuePool
            # === END POOL SIZE LOGIC ===

            final_params_log = f"FINAL PARAMS for create_engine: pool_size={pool_size}, max_overflow={max_overflow}, pool_timeout={pool_timeout}"
            logger.debug(f"*** SessionManager ID={id(self)} {final_params_log} ***")

            self.engine = create_engine(  # Assign to self.engine
                f"sqlite:///{self.db_path}",
                echo=False,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                poolclass=pool_class,
                connect_args={"check_same_thread": False},
            )
            logger.debug(
                f"SessionManager ID={id(self)} created NEW engine: ID={id(self.engine)}"
            )

            # Log pool size
            try:
                actual_pool_size = getattr(self.engine.pool, "_size", "N/A")
                logger.debug(
                    f"Engine ID={id(self.engine)} pool size reported (internal): {actual_pool_size}"
                )
            except Exception:
                logger.debug("Could not retrieve detailed engine pool status.")

            # --- PRAGMA listener ---
            @event.listens_for(self.engine, "connect")
            def enable_foreign_keys(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA journal_mode=WAL;")
                    cursor.execute("PRAGMA foreign_keys=ON;")
                except Exception as pragma_e:
                    logger.error(f"Failed setting PRAGMA: {pragma_e}")
                finally:
                    cursor.close()

            self.Session = sessionmaker(
                bind=self.engine, expire_on_commit=False
            )  # Assign to self.Session
            logger.debug(
                f"SessionManager ID={id(self)} created Session factory for Engine ID={id(self.engine)}"
            )

            # Create tables
            try:
                Base.metadata.create_all(self.engine)
                logger.debug("DB tables checked/created.")
            except Exception as table_create_e:
                logger.error(
                    f"Error creating DB tables: {table_create_e}", exc_info=True
                )
                raise table_create_e

        except Exception as e:
            logger.critical(
                f"SessionManager ID={id(self)} FAILED to initialize SQLAlchemy: {e}",
                exc_info=True,
            )
            if self.engine:
                try:
                    self.engine.dispose()
                except Exception:
                    pass
            self.engine = None  # Ensure engine is None on failure
            self.Session = None  # Ensure Session is None on failure
            raise e
    # end _initialize_db_engine_and_session

    def _check_and_handle_url(self) -> bool:
        """Checks current URL and navigates to base URL if necessary."""
        try:
            if self.driver is None:
                logger.error("Driver is not initialized. Cannot retrieve current URL.")
                return False
            current_url = self.driver.current_url
            logger.debug(f"Current URL for check: {current_url}")
            base_url_norm = config_instance.BASE_URL.rstrip("/") + "/"
            signin_url_base = urljoin(config_instance.BASE_URL, "account/signin")
            logout_url_base = urljoin(config_instance.BASE_URL, "c/logout")
            mfa_url_base = urljoin(
                config_instance.BASE_URL, "account/signin/mfa/"
            )  # Should be handled by login_status

            disallowed_starts = (
                signin_url_base,
                logout_url_base,
                mfa_url_base,
            )  # Add MFA here just in case
            is_api_path = "/api/" in current_url

            needs_navigation = False
            if not current_url.startswith(
                config_instance.BASE_URL.rstrip("/")
            ):  # Looser check
                needs_navigation = True
                logger.debug("Reason: URL does not start with base URL.")
            elif any(current_url.startswith(path) for path in disallowed_starts):
                needs_navigation = True
                logger.debug(
                    f"Reason: URL starts with disallowed path ({current_url})."
                )
            elif is_api_path:
                needs_navigation = True
                logger.debug("Reason: URL contains '/api/'.")

            if needs_navigation:
                logger.info(
                    f"Current URL '{current_url}' unsuitable. Navigating to base URL: {base_url_norm}"
                )
                if not nav_to_page(
                    self.driver, base_url_norm, selector="body", session_manager=self
                ):
                    logger.error("Failed to navigate to base URL during check.")
                    return False
                logger.debug("Navigation to base URL successful.")
            else:
                logger.debug("Current URL is suitable, no extra navigation needed.")
            return True

        except WebDriverException as e:
            logger.error(
                f"Error during URL check/navigation: {e}. Session might be dead.",
                exc_info=True,
            )
            # Attempt to check if session is valid, if not, close might be needed
            if not self.is_sess_valid():
                logger.warning("Session seems invalid during URL check.")
                # self.close_sess() # Consider closing if invalid
            return False  # Indicate failure
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
        if not self.my_profile_id:  # Only fetch if not already set
            self.my_profile_id = self.get_my_profileId()
            if not self.my_profile_id:
                logger.error("Failed to retrieve profile ID (ucdmid).")
                all_ok = False
            elif not self._profile_id_logged:  # Log only first time
                logger.info(f"My profile id: {self.my_profile_id}")
                self._profile_id_logged = True
        elif not self._profile_id_logged:  # Log if already set but not logged
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
        if (
            config_instance.TREE_NAME and not self.my_tree_id
        ):  # Only fetch if needed and not set
            self.my_tree_id = self.get_my_tree_id()
            if not self.my_tree_id:
                logger.error(
                    f"TREE_NAME '{config_instance.TREE_NAME}' configured, but failed to get corresponding tree ID."
                )
                all_ok = False  # Treat as error only if TREE_NAME is set
            elif not self._tree_id_logged:
                logger.info(f"My tree id: {self.my_tree_id}")
                self._tree_id_logged = True
        elif (
            self.my_tree_id and not self._tree_id_logged
        ):  # Log if already set but not logged
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
        if (
            self.my_tree_id and not self.tree_owner_name
        ):  # Fetch only if tree ID exists and owner not yet known
            self.tree_owner_name = self.get_tree_owner(self.my_tree_id)
            if self.tree_owner_name and not self._owner_logged:
                logger.info(f"Tree Owner Name: {self.tree_owner_name}.\n")
                self._owner_logged = True
            elif not self.tree_owner_name:
                logger.warning("Failed to retrieve tree owner name.\n")
        elif (
            self.tree_owner_name and not self._owner_logged
        ):  # Log if already known but not logged
            logger.info(f"Tree Owner Name: {self.tree_owner_name}\n")
            self._owner_logged = True
        elif not self.my_tree_id:
            logger.debug("Skipping tree owner retrieval (no tree ID).\n")
    # end _retrieve_tree_owner

    @retry_api()  # Add retry decorator
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
        if not self.get_cookies(essential_cookies, timeout=10):  # Shorter timeout?
            logger.warning(
                f"Essential cookies {essential_cookies} NOT found before CSRF token API call."
            )
            return None

        csrf_token_url = urljoin(
            config_instance.BASE_URL, "discoveryui-matches/parents/api/csrfToken"
        )

        # _api_req handles headers including UBE if driver available
        response_data = _api_req(
            url=csrf_token_url,
            driver=self.driver,  # Pass driver for UBE header
            session_manager=self,
            method="GET",
            use_csrf_token=False,  # Don't need CSRF to get CSRF
            api_description="CSRF Token API",
        )

        if not response_data:
            logger.warning("Failed to get CSRF token - _api_req returned None.")
            return None

        # Handle potential response formats
        if isinstance(response_data, dict) and "csrfToken" in response_data:
            csrf_token_val = response_data["csrfToken"]
        elif isinstance(response_data, str):  # Plain text token
            csrf_token_val = response_data.strip()
            if not csrf_token_val:  # Handle empty string response
                logger.error("CSRF token API returned empty string.")
                return None
            logger.debug("CSRF token retrieved as plain text string from API endpoint.")
        else:
            logger.error("Unexpected response format for CSRF token API.")
            logger.debug(
                f"Response data type: {type(response_data)}, value: {response_data}"
            )
            return None

        # Return the valid token (don't cache here, let start_sess handle setting self.csrf_token)
        return csrf_token_val
    # end get_csrf

    def get_cookies(
        self, cookie_names: List[str], timeout: int = 30
    ) -> bool:  # Type hint list
        """Waits until specified cookies are present and returns True, else False."""
        start_time = time.time()
        logger.debug(f"Waiting up to {timeout}s for cookies: {cookie_names}.")
        required_lower = {name.lower() for name in cookie_names}
        interval = 0.5
        last_missing_str = ""

        while time.time() - start_time < timeout:
            try:
                if not self.is_sess_valid():  # Use robust check
                    logger.warning("Session became invalid while waiting for cookies.")
                    return False

                if self.driver is None:
                    logger.error("Driver is not initialized. Cannot retrieve cookies.")
                    return False
                # Simplified check as is_sess_valid already checked driver existence
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

            except WebDriverException as e:  # Catch WebDriver specific exceptions
                logger.error(f"WebDriverException while retrieving cookies: {e}")
                # Check if session died
                if not self.is_sess_valid():
                    logger.error(
                        "Session invalid after WebDriverException during cookie retrieval."
                    )
                    return False
                # Otherwise, wait and retry
                time.sleep(interval * 2)  # Longer sleep after error
            except Exception as e:
                logger.error(f"Unexpected error retrieving cookies: {e}", exc_info=True)
                # Decide whether to continue or fail on unexpected error
                time.sleep(interval * 2)

        # Loop finished without finding all cookies
        # Recalculate missing cookies one last time
        missing_final = []
        try:
            if self.driver:
                cookies_final = self.driver.get_cookies()
                current_cookies_final_lower = {c["name"].lower() for c in cookies_final}
                missing_final = [name for name in cookie_names if name.lower() in (required_lower - current_cookies_final_lower)]
            else:
                missing_final = cookie_names # Assume all are missing if driver gone
        except Exception: # Catch errors during final check
            missing_final = cookie_names # Assume all missing on error

        logger.warning(f"Timeout waiting for cookies. Missing: {missing_final}.")
        return False
    # end get_cookies

    def _sync_cookies(self):
        """Syncs cookies from WebDriver to requests session, adjusting domain for SecureATT."""
        if not self.is_sess_valid():  # Use robust check
            logger.warning("Cannot sync cookies: Session invalid.")
            return

        try:
            if self.driver is None:
                logger.error("Driver is None. Cannot retrieve cookies for sync.")
                return
            cookies = self.driver.get_cookies()
            # Clear existing cookies in requests session before syncing
            self._requests_session.cookies.clear()
            synced_count = 0
            for cookie in cookies:
                # Ensure required fields are present
                if "name" in cookie and "value" in cookie and "domain" in cookie:
                    # Adjust domain for SecureATT cookie if needed
                    domain_to_set = cookie["domain"]
                    if (
                        cookie["name"] == "SecureATT"
                        and domain_to_set == "www.ancestry.co.uk"
                    ):
                        domain_to_set = (
                            ".ancestry.co.uk"  # Use leading dot for subdomain matching
                        )

                    # Prepare cookie attributes for requests session
                    cookie_attrs = {
                        "name": cookie["name"],
                        "value": cookie["value"],
                        "domain": domain_to_set,
                        "path": cookie.get("path", "/"),
                        "secure": cookie.get("secure", False),
                        "rest": {
                            "httpOnly": cookie.get("httpOnly")
                        },  # Pass httpOnly via rest dict
                    }
                    # Add expires if present (requests uses 'expires')
                    if "expiry" in cookie and cookie["expiry"] is not None:
                        # Check if expiry is already an int timestamp (preferred by requests)
                        if isinstance(cookie["expiry"], (int, float)):
                            cookie_attrs["expires"] = int(cookie["expiry"])
                        # Handle other potential formats (less common for expiry) if needed
                        # else: logger.warning(f"Unexpected expiry format for cookie {cookie['name']}: {cookie['expiry']}")

                    self._requests_session.cookies.set(**cookie_attrs)
                    synced_count += 1
                else:
                    logger.warning(
                        f"Skipping invalid cookie format during sync: {cookie}"
                    )

            logger.debug(f"Synced {synced_count} cookies to requests session.") # Less verbose
        except WebDriverException as e:
            logger.error(f"WebDriverException during cookie sync: {e}")
            # Check if session died
            if not self.is_sess_valid():
                logger.error(
                    "Session invalid after WebDriverException during cookie sync."
                )
        except Exception as e:
            logger.error(f"Unexpected error during cookie sync: {e}", exc_info=True)
    # end _sync_cookies

    # --- Removed inner Cache class ---

    def return_session(self, session: Session):
        """Closes the session, returning the underlying connection to the pool."""
        if session:
            session_id = id(session)  # For logging
            try:
                # logger.debug(f"Closing session {session_id} (returns connection to pool).") # Less verbose
                session.close()
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}", exc_info=True)
    # end return_session

    def get_db_conn(self) -> Optional[Session]:
        """Gets a session from the SQLAlchemy session factory. Initializes if needed."""
        # Log which SM instance and engine is involved
        engine_id = id(self.engine) if self.engine else "None"
        logger.debug(
            f"SessionManager ID={id(self)} get_db_conn called. Current Engine ID: {engine_id}"
        )

        # --- Check if initialization is needed ---
        # Initialize if never attempted OR if engine/Session became None after previous attempt
        if not self._db_init_attempted or not self.engine or not self.Session:
            logger.debug(
                f"SessionManager ID={id(self)}: Engine/Session factory not ready. Triggering initialization..."
            )
            try:
                self._initialize_db_engine_and_session()
                # Check again after initialization attempt
                if not self.Session:
                    logger.error(
                        f"SessionManager ID={id(self)}: Initialization failed, cannot get DB connection."
                    )
                    return None
            except Exception as init_e:
                logger.error(
                    f"SessionManager ID={id(self)}: Exception during lazy initialization in get_db_conn: {init_e}"
                )
                return None

        # --- Attempt to get session from factory ---
        try:
            # self.Session should now be valid if initialization succeeded
            session = self.Session()
            logger.debug(
                f"SessionManager ID={id(self)} obtained DB session {id(session)} from Engine ID={id(self.engine)}"
            )
            return session
        except Exception as e:
            logger.error(
                f"SessionManager ID={id(self)} Error getting DB session from factory: {e}",
                exc_info=True,
            )
            # If getting session fails, maybe the engine died? Clear it to force re-init next time.
            if self.engine:
                try:
                    self.engine.dispose()
                except Exception:
                    pass
            self.engine = None
            self.Session = None
            self._db_init_attempted = False  # Allow re-init attempt
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

            yield session  # Provide the session

            # COMMIT if successful exit from 'with' block
            if session.is_active:
                session.commit()
        except Exception as e:
            logger.error(
                f"Exception within get_db_conn_context block or commit: {e}. Rolling back.",
                exc_info=True,
            )
            if session and session.is_active:
                try:
                    session.rollback()
                except Exception as rb_err:
                    logger.error(
                        f"Error rolling back session in context manager: {rb_err}"
                    )
            raise e  # Re-raise
        finally:
            # Ensure session is closed/returned regardless of success/failure
            if session:
                self.return_session(session)
    # End of get_db_conn_context method

    def cls_db_conn(self, keep_db: bool = True):
        """
        Disposes the SQLAlchemy engine, closing all pooled connections.
        Optionally keeps the engine alive if keep_db is True.
        """
        # Check self.engine existence BEFORE accessing it for logging/disposal
        if not self.engine:
            logger.debug(
                f"SessionManager ID={id(self)}: No active SQLAlchemy engine to dispose."
            )
            # Ensure other related attributes are also None
            self.Session = None
            self._db_init_attempted = False
            return  # Nothing more to do

        engine_id = id(self.engine)  # Safe to get ID now
        if keep_db:
            logger.debug(
                f"SessionManager ID={id(self)} cls_db_conn called (keep_db=True). Skipping engine disposal for Engine ID: {engine_id}"
            )
            return  # Do nothing if keeping DB

        logger.debug(
            f"SessionManager ID={id(self)} cls_db_conn called (keep_db=False). Disposing Engine ID: {engine_id}"
        )
        try:
            self.engine.dispose()
            logger.debug(f"Engine ID={engine_id} disposed.")
        except Exception as e:
            logger.error(
                f"Error disposing SQLAlchemy engine ID={engine_id}: {e}",
                exc_info=True,
            )
        finally:
            # Always set engine/Session to None after disposal attempt
            self.engine = None
            self.Session = None
            self._db_init_attempted = False  # Reset flag
    # End of cls_db_conn

    @retry_api()  # Add retry decorator
    def get_my_profileId(self) -> Optional[str]:
        """
        Retrieves the user ID (ucdmid) from the Ancestry API using _api_req. Retries on failure.
        """
        url = urljoin(
            config_instance.BASE_URL,
            "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid",
        )
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

            if (
                isinstance(response_data, dict)
                and "data" in response_data
                and "ucdmid" in response_data["data"]
            ):
                my_profile_id_val = str(
                    response_data["data"]["ucdmid"]
                ).upper()  # Ensure string and uppercase
                return my_profile_id_val
            else:
                logger.error("Could not find 'data.ucdmid' in profile_id API response.")
                logger.debug(f"Full profile_id response data: {response_data}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in get_my_profileId: {e}", exc_info=True)
            return None
    # end get_my_profileId

    @retry_api()  # Add retry decorator
    def get_my_uuid(self) -> Optional[str]:  # Ensure return type consistency
        """Retrieves the test uuid (sampleId) from the header/dna API endpoint. Retries on failure."""
        if not self.is_sess_valid():  # Check session validity first
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
                my_uuid_val = str(
                    response_data["testId"]
                ).upper()  # Ensure string and uppercase
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

    @retry_api()  # Add retry decorator
    def get_my_tree_id(self) -> Optional[str]:
        """
        Retrieves the tree ID based on TREE_NAME from config, using the header/trees API. Retries on failure.
        """
        tree_name_config = config_instance.TREE_NAME
        if not tree_name_config:
            logger.debug("TREE_NAME not configured, skipping tree ID retrieval.")
            return None

        if not self.is_sess_valid():  # Check session validity first
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
            if (
                response_data
                and isinstance(response_data, dict)
                and "menuitems" in response_data
            ):
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
                                logger.warning(
                                    f"Found tree '{tree_name_config}', but URL format unexpected: {tree_url}"
                                )
                        else:
                            logger.warning(
                                f"Found tree '{tree_name_config}', but 'url' key missing or invalid."
                            )
                        break  # Stop searching once found

                logger.warning(
                    f"Could not find TREE_NAME '{tree_name_config}' in Header Trees API response."
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

    @retry_api()  # Add retry decorator
    def get_tree_owner(self, tree_id: str) -> Optional[str]:
        """
        Retrieves the tree owner's display name using _api_req. Retries on failure.
        """
        if not tree_id:
            logger.warning("Cannot get tree owner: tree_id is missing.")
            return None

        if not self.is_sess_valid():  # Check session validity first
            logger.error("get_tree_owner: Session invalid.")
            return None

        url = urljoin(
            config_instance.BASE_URL,
            f"api/uhome/secure/rest/user/tree-info?tree_id={tree_id}",
        )

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
                    logger.warning(
                        "Could not find 'owner' data in Tree Owner API response."
                    )
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
        """Verifies session using login_status (which prioritizes API)."""
        logger.debug("Verifying session status...")  # Simplified log
        login_ok = login_status(self)  # login_status now handles API/UI checks

        if login_ok is True:
            logger.debug("Session verification successful.")
            return True
        elif login_ok is False:
            logger.warning("Session verification failed (user not logged in).")
            # Optionally attempt re-login here if desired, or let caller handle
            # login_result = log_in(self) ... etc.
            return False
        else:  # login_ok is None (critical error)
            logger.error("Session verification failed critically.")
            return False
    # End of verify_sess

    def _verify_api_login_status(self) -> Optional[bool]:
        """
        REVISED V4: Checks login status via /header/dna API.
        Returns True if API confirms login, False if API indicates not logged in,
        None if a critical error occurs during the check.
        """
        # Removed internal caching flag (self.api_login_verified)

        logger.debug("Verifying login status via header/dna API endpoint...")
        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        api_description = "API Login Verification (header/dna)"

        # Check driver state before proceeding
        if not self.driver or not self.is_sess_valid():
            logger.warning(f"{api_description}: Driver/session not valid for API check.")
            return None # Critical error if driver isn't ready

        # Ensure cookies are synced *before* this specific API call
        self._sync_cookies()

        try:
            # Pass self (SessionManager instance) correctly
            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,  # Pass the instance itself
                method="GET",
                use_csrf_token=False,
                api_description=api_description,
                force_requests=True, # Use requests session which should now have synced cookies
            )

            # Handle cases based on _api_req return value
            if response_data is None:
                # _api_req returns None on total failure (retries exhausted)
                logger.warning(f"{api_description}: _api_req returned None (likely connection/timeout after retries). Returning None.")
                return None # Critical error state

            elif isinstance(response_data, requests.Response):
                # _api_req returns the Response object on non-2xx status codes that aren't retryable
                status_code = response_data.status_code
                if status_code in [401, 403]:
                    logger.debug(f"{api_description}: API check failed with status {status_code}. User not logged in.")
                    return False # Explicitly not logged in
                else:
                    logger.warning(f"{api_description}: API check failed with unexpected status {status_code}. Returning None.")
                    return None # Unexpected error state

            elif isinstance(response_data, dict):
                # Successful 2xx response, check content
                if "testId" in response_data:
                    logger.debug(f"API login check successful via {api_description}.")
                    return True
                else:
                    # Succeeded (2xx) but unexpected format - Treat as error? Or logged in?
                    # Let's treat unexpected format cautiously as potentially logged in but warn.
                    logger.warning(
                        f"API login check via {api_description} succeeded (2xx), but response format unexpected: {response_data}"
                    )
                    return True # Cautiously assume logged in if 2xx

            else:
                # _api_req returned something else unexpected (e.g., string from non-JSON 2xx)
                logger.error(f"{api_description}: _api_req returned unexpected type {type(response_data)}. Returning None.")
                return None # Critical error state

        except Exception as e:
            # Catch unexpected errors during the verification process itself
            logger.error(
                f"Unexpected error during API login status check ({api_description}): {e}",
                exc_info=True,
            )
            return None # Critical error state
    # end _verify_api_login_status

    @retry_api()  # Use retry_api decorator
    def get_header(self) -> bool:  # Return bool for success/failure
        """Retrieves data from the header/dna API endpoint. Retries on failure."""
        if not self.is_sess_valid():  # Check session first
            logger.error("get_header: Session invalid.")
            return False

        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        response_data = _api_req(
            url,
            self.driver,
            self,
            method="GET",
            use_csrf_token=False,
            api_description="Get UUID API",  # Use appropriate description
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

    def _validate_sess_cookies(self, required_cookies: List[str]) -> bool:  # Type hint
        """Check if specific cookies exist in the current session."""
        if not self.is_sess_valid():  # Use robust check
            logger.warning("Cannot validate cookies: Session invalid.")
            return False
        try:
            if self.driver is not None:
                cookies = {c["name"]: c["value"] for c in self.driver.get_cookies()}
            else:
                logger.error("Driver is not initialized. Cannot retrieve cookies.")
                cookies = {}
            return all(cookie in cookies for cookie in required_cookies)
        except WebDriverException as e:
            logger.error(f"WebDriverException during cookie validation: {e}")
            if not self.is_sess_valid():
                logger.error(
                    "Session invalid after WebDriverException during cookie validation."
                )
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating cookies: {e}", exc_info=True)
            return False
    # End of _validate_sess_cookies

    def is_sess_logged_in(self) -> bool:  # Return bool only
        """DEPRECATED: Use login_status() instead. Checks session validity AND login using UI element verification ONLY."""
        logger.warning("is_sess_logged_in is deprecated. Use login_status() instead.")
        return login_status(self) is True  # Delegate to new function, return bool
    # End is_sess_logged_in

    def is_sess_valid(self) -> bool:  # Type hint return
        """Checks if the current browser session is valid (quick check), handling InvalidSessionIdException."""
        if not self.driver:
            # logger.debug("Browser not open (driver is None).") # Less verbose
            return False
        try:
            # Try a minimal WebDriver command that requires an active session
            _ = self.driver.window_handles  # Accessing handles is lightweight
            return True  # If no exception, browser session is likely valid
        except InvalidSessionIdException:
            logger.debug(
                "Session ID is invalid (browser likely closed or session terminated)."
            )
            return False
        except WebDriverException as e:  # Catch other WebDriver exceptions
            # Log specific WebDriver exceptions that might indicate a dead session
            if "disconnected" in str(e).lower() or "target crashed" in str(e).lower():
                logger.warning(f"Session seems invalid due to WebDriverException: {e}")
                return False
            else:
                logger.warning(
                    f"Unexpected WebDriverException checking session validity: {e}"
                )
                # Could still be valid despite other errors, maybe return True cautiously?
                # For safety, let's return False if any WebDriverException occurs here.
                return False
        except Exception as e:  # Catch any other unexpected exceptions
            logger.error(
                f"Unexpected error checking session validity: {e}", exc_info=True
            )
            return False
    # End of is_sess_valid

    def close_sess(self, keep_db: bool = False): # Added keep_db flag
        """Closes the Selenium WebDriver session and optionally the DB pool."""
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

        # Reset state flags
        self.driver_live = False
        self.session_ready = False
        self.csrf_token = None # Clear CSRF token

        # Close DB connection pool unless asked to keep it
        if not keep_db:
            self.cls_db_conn(keep_db=False) # Call helper to dispose engine
        else:
            logger.debug("Keeping DB connection pool alive (keep_db=True).")
    # End of close_sess

    def restart_sess(
        self, url: Optional[str] = None
    ) -> bool:  # Return bool for success
        """Restarts the WebDriver session and optionally navigates to a URL."""
        logger.warning("Restarting WebDriver session...")
        self.close_sess(keep_db=True) # Ensure any existing session is closed, but keep DB pool

        # --- V3 Refactoring: Use two-phase start ---
        # Phase 1: Start Driver
        start_ok = self.start_sess(action_name="Session Restart - Phase 1")
        if not start_ok:
            logger.error("Failed to restart session (Phase 1: Driver Start failed).")
            return False

        # Phase 2: Ensure Session Ready (Login, Identifiers)
        ready_ok = self.ensure_session_ready(action_name="Session Restart - Phase 2")
        if not ready_ok:
            logger.error("Failed to restart session (Phase 2: Session Ready failed).")
            # Clean up driver started in Phase 1
            self.close_sess(keep_db=True) # Keep DB pool if restart fails here
            return False
        # --- End V3 Refactoring ---

        # Session started and ready, now navigate if URL provided
        if url and self.driver:
            logger.info(f"Re-navigating to {url} after restart...")
            if nav_to_page(
                self.driver, url, selector="body", session_manager=self
            ):  # Wait for body
                logger.info(f"Successfully re-navigated to {url}.")
                return True
            else:
                logger.error(f"Failed to re-navigate to {url} after restart.")
                return False  # Navigation failed after successful restart
        elif not url:
            return True  # Restart succeeded, no navigation needed
        else:  # url provided but driver missing after start_sess (shouldn't happen if start_ok is True)
            logger.error(
                "Driver instance missing after successful session restart report."
            )
            return False
    # End of restart_sess

    @ensure_browser_open
    def make_tab(self) -> Optional[str]:  # Type hint return
        """Create a new tab and return its handle id"""
        driver = self.driver  # Assume decorator ensures driver exists
        try:
            if driver is None:
                logger.error(
                    "Driver is not initialized. Cannot retrieve window handles."
                )
                return None
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
            if driver is not None:
                logger.debug(f"Window handles during timeout: {driver.window_handles}")
            else:
                logger.error(
                    "Driver is None while attempting to log window handles during timeout."
                )
            return None
        except (
            IndexError,
            WebDriverException,
        ) as e:  # Catch potential errors finding handle
            logger.error(f"Error identifying new tab handle: {e}")
            if driver is not None:
                logger.debug(f"Window handles during error: {driver.window_handles}")
            else:
                logger.error(
                    "Driver is None while attempting to log window handles during error."
                )
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred in make_tab: {e}", exc_info=True
            )
            return None
    # End of make_tab

    def check_js_errors(self):
        """Checks for new JavaScript errors since the last check and logs them."""
        if not self.is_sess_valid():  # Check session validity
            # logger.debug("Skipping JS error check: Session invalid.") # Less verbose
            return

        try:
            # Check if get_log is supported (might not be in headless/certain configs)
            if self.driver is not None:
                log_types = self.driver.log_types
            else:
                logger.warning("Driver is not initialized. Skipping log type check.")
                return
            if "browser" not in log_types:
                # logger.debug("Browser log type not supported by this WebDriver instance.") # Less verbose
                return

            logs = self.driver.get_log("browser")
            new_errors_found = False
            most_recent_error_time = (
                self.last_js_error_check
            )  # Keep track of latest error this check

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
                                error_message = entry.get("message", "No message")
                                # Try to extract source and line number if available
                                source_match = re.search(r"(.+?):(\d+)", error_message)
                                source_info = (
                                    f" (Source: {source_match.group(1).split('/')[-1]}:{source_match.group(2)})"
                                    if source_match
                                    else ""
                                )
                                # Log clearly identifiable JS errors
                                logger.warning(
                                    f"JS ERROR DETECTED:{source_info} {error_message}"
                                )
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
            logger.error(
                f"Unexpected error checking for Javascript errors: {e}", exc_info=True
            )
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
    force_requests: bool = False,
    cookie_jar: Optional[RequestsCookieJar] = None,
    allow_redirects: bool = True,
    force_text_response: bool = False,  # <<< ADDED force_text_response flag
) -> Optional[Any]:
    """
    V14.3 CORRECTED: Makes HTTP request. Allows passing cookie jar, allow_redirects, force_text_response.
    - Added `force_text_response` parameter (default False).
    - Checks force_text_response before Content-Type for parsing.
    """
    # [Initial checks and header assembly remain the same as previous version]
    if not session_manager:
        logger.error(
            f"{api_description}: Aborting - SessionManager instance is required."
        )
        return None
    browser_needed = not force_requests # If forcing requests, assume browser isn't strictly needed for THIS call
    if browser_needed and not session_manager.is_sess_valid():
        logger.error(
            f"{api_description}: Aborting - Browser session is invalid or closed."
        )
        return None
    max_retries = config_instance.MAX_RETRIES
    initial_delay = config_instance.INITIAL_DELAY
    backoff_factor = config_instance.BACKOFF_FACTOR
    max_delay = config_instance.MAX_DELAY
    retry_status_codes = config_instance.RETRY_STATUS_CODES
    final_headers = {}
    contextual_headers = config_instance.API_CONTEXTUAL_HEADERS.get(api_description, {})
    final_headers.update({k: v for k, v in contextual_headers.items() if v is not None})
    if headers:
        final_headers.update({k: v for k, v in headers.items() if v is not None})
    if "User-Agent" not in final_headers:
        ua = None
        if driver and session_manager.is_sess_valid():
            try:
                ua = driver.execute_script("return navigator.userAgent;")
            except Exception:
                pass
        if not ua:
            ua = random.choice(config_instance.USER_AGENTS)
        final_headers["User-Agent"] = ua
    if referer_url and "Referer" not in final_headers:
        final_headers["Referer"] = referer_url
    if use_csrf_token and "X-CSRF-Token" not in final_headers:
        csrf_from_manager = session_manager.csrf_token
        if csrf_from_manager:
            final_headers["X-CSRF-Token"] = csrf_from_manager
        else:
            logger.error(f"{api_description}: CSRF token required but not found.")
            return None
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
    if "ancestry-userid" in contextual_headers and session_manager.my_profile_id:
        # Check if my_profile_id exists before trying to upper()
        if session_manager.my_profile_id:
            final_headers["ancestry-userid"] = session_manager.my_profile_id.upper()
        else:
            logger.warning(f"{api_description}: Expected ancestry-userid header, but session_manager.my_profile_id is None.")

    if "newrelic" not in final_headers:
        final_headers["newrelic"] = make_newrelic(driver)
    if "traceparent" not in final_headers:
        final_headers["traceparent"] = make_traceparent(driver)
    if "tracestate" not in final_headers:
        final_headers["tracestate"] = make_tracestate(driver)
    if api_description == "Match List API":
        if "Cache-Control" not in final_headers:
            final_headers["Cache-Control"] = "no-cache"
        if "Pragma" not in final_headers:
            final_headers["Pragma"] = "no-cache"

    request_timeout = timeout if timeout is not None else selenium_config.API_TIMEOUT
    logger.debug(
        f"API Req: {method.upper()} {url} (AllowRedirects={allow_redirects}, ForceText={force_text_response})"
    )
    req_session = session_manager._requests_session
    effective_cookies = cookie_jar if cookie_jar is not None else req_session.cookies
    if cookie_jar is not None:
        logger.debug(f"Using explicitly provided cookie jar for '{api_description}'.")

    retries_left = max_retries
    last_exception = None
    delay = initial_delay

    while retries_left > 0:
        attempt = max_retries - retries_left + 1
        response: Optional[RequestsResponse] = None
        try:
            if cookie_jar is None and driver and session_manager.is_sess_valid():
                try:
                    session_manager._sync_cookies()
                except Exception as sync_err:
                    logger.warning(
                        f"{api_description}: Error syncing cookies (Attempt {attempt}): {sync_err}"
                    )

            # [Detailed Logging remains the same]
            if logger.isEnabledFor(logging.DEBUG):
                # (Log headers, cookies, payload as before)
                pass

            # Make Request
            response = req_session.request(
                method=method.upper(),
                url=url,
                headers=final_headers,
                data=data,
                json=json_data,
                timeout=request_timeout,
                verify=True,
                allow_redirects=allow_redirects,
                cookies=effective_cookies,
            )
            status = response.status_code

            # Log Response
            logger.debug(f"<-- Response Status: {status} {response.reason}")
            # [Detailed response logging remains the same]
            if logger.isEnabledFor(logging.DEBUG):
                # (Log headers, history, body preview as before)
                pass

            # --- Response Handling ---
            if status in retry_status_codes:
                # [Retry logic remains the same]
                retries_left -= 1
                last_exception = HTTPError(
                    f"{status} Server Error: {response.reason}", response=response
                )
                if retries_left <= 0:
                    logger.error(
                        f"{api_description}: Failed after {max_retries} attempts (Final Status {status})."
                    )
                    return response
                else:
                    sleep_time = min(
                        delay * (backoff_factor ** (attempt - 1)), max_delay
                    )
                    sleep_time = max(0.1, sleep_time)
                    if session_manager.dynamic_rate_limiter and status == 429:
                        session_manager.dynamic_rate_limiter.increase_delay()
                    logger.warning(
                        f"{api_description}: Status {status} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    delay *= backoff_factor
                    continue

            elif 300 <= status < 400 and not allow_redirects:
                # [Handling disabled redirects remains the same]
                logger.warning(
                    f"{api_description}: Received redirect status {status} {response.reason} (Redirects were disabled). Returning response object."
                )
                return response

            elif 300 <= status < 400 and allow_redirects:
                # [Handling unexpected final redirects remains the same]
                logger.warning(
                    f"{api_description}: Received unexpected final redirect status {status} {response.reason} (Redirects were enabled)."
                )
                return response

            elif response.ok:  # Status 2xx
                if session_manager.dynamic_rate_limiter:
                    session_manager.dynamic_rate_limiter.decrease_delay()

                # <<< CHECK force_text_response FIRST >>>
                if force_text_response:
                    logger.debug(
                        f"{api_description}: OK ({status}). Returning TEXT response as forced."
                    )
                    return response.text
                # <<< END CHECK >>>

                # --- If not forced, check Content-Type ---
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    try:
                        # --- Add check for empty response body before JSON decode ---
                        if not response.content:
                            logger.warning(
                                f"{api_description}: OK ({status}), Content-Type is JSON, but response body is EMPTY. Returning None."
                            )
                            return None
                        # --- End check for empty body ---
                        return response.json()
                    except json.JSONDecodeError as json_err:
                        logger.error(
                            f"{api_description}: OK ({status}), Content-Type is JSON, but JSON decode FAILED: {json_err}"
                        )
                        logger.debug(
                            f"Response text causing decode error: {response.text[:500]}"
                        )
                        return None  # Indicate failure if JSON expected but invalid
                else:
                    # Specific handling for CSRF text/plain remains useful
                    if (
                        api_description == "CSRF Token API"
                        and "text/plain" in content_type
                    ):
                        logger.debug(
                            f"{api_description}: OK ({status}), Content-Type '{content_type}'. Returning TEXT as expected."
                        )
                        csrf_text = response.text.strip()
                        return csrf_text if csrf_text else None
                    else:  # General fallback for other non-JSON types
                        logger.warning(
                            f"{api_description}: Request OK ({status}), but received unexpected Content-Type '{content_type}'. Returning TEXT."
                        )
                        return response.text

            else:  # Non-retryable error >= 400
                # [Error handling remains the same]
                if status in [401, 403]:
                    logger.warning(
                        f"{api_description}: API call failed with status {status} {response.reason}. Likely not logged in or session expired."
                    )
                    # session_manager.api_login_verified = False # Removed this flag
                    session_manager.session_ready = False # Use session_ready flag instead
                else:
                    logger.error(
                        f"{api_description}: Non-retryable error: {status} {response.reason}."
                    )
                return response

        # --- Exception Handling ---
        except requests.exceptions.RequestException as e:
            # [Exception retry logic remains the same]
            retries_left -= 1
            last_exception = e
            if retries_left <= 0:
                logger.error(
                    f"{api_description}: RequestException failed after {max_retries} attempts. Final Error: {e}",
                    exc_info=False,
                )
                return None
            else:
                sleep_time = min(delay * (backoff_factor ** (attempt - 1)), max_delay)
                sleep_time = max(0.1, sleep_time)
                logger.warning(
                    f"{api_description}: RequestException (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s... Error: {e}"
                )
                time.sleep(sleep_time)
                delay *= backoff_factor
                continue
        except Exception as e:
            logger.critical(
                f"{api_description}: CRITICAL Unexpected error during request attempt {attempt}: {e}",
                exc_info=True,
            )
            return None

    logger.error(
        f"{api_description}: Exited retry loop after {max_retries} failed attempts. Last Exception: {last_exception}. Returning None."
    )
    return None
# End of _api_req


def make_ube(driver: Optional[WebDriver]) -> Optional[str]:
    """
    REVISED V14.1: Generates UBE header. Ensures correlatedSessionId matches ANCSESSIONID cookie.
    Uses static null eventId from cURL example.
    Removed internal SessionManager creation, uses driver directly.
    """
    if not driver:
        logger.debug("Cannot generate UBE header: WebDriver is None.")
        return None
    try:
        # --- Use helper function or direct check for session validity ---
        # if not is_browser_open(driver): # Check using the global helper
        # Or inline check:
        try:
            _ = driver.window_handles  # Simple check if driver is responsive
        except WebDriverException as e:
            logger.warning(
                f"Cannot generate UBE header: Session invalid/unresponsive ({type(e).__name__})."
            )
            return None
        # --- End session validity check ---

        ancsessionid = None
        try:
            cookie_obj = driver.get_cookie("ANCSESSIONID")
            if cookie_obj and "value" in cookie_obj:
                ancsessionid = cookie_obj["value"]
            else:
                cookies = {c["name"]: c["value"] for c in driver.get_cookies()}
                ancsessionid = cookies.get("ANCSESSIONID")

            if not ancsessionid:
                logger.warning("ANCSESSIONID cookie not found for UBE header.")
                return None
        except NoSuchCookieException:
            logger.warning(
                "ANCSESSIONID cookie not found via get_cookie for UBE header."
            )
            return None
        except WebDriverException as cookie_e:
            logger.warning(
                f"WebDriver error getting ANCSESSIONID cookie for UBE: {cookie_e}"
            )
            return None

        event_id = "00000000-0000-0000-0000-000000000000"
        correlated_id = str(uuid.uuid4())
        screen_name_standard = "ancestry : uk : en : dna-matches-ui : match-list : 1"
        screen_name_legacy = "ancestry uk : dnamatches-matchlistui : list"

        ube_data = {
            "eventId": event_id,
            "correlatedScreenViewedId": correlated_id,
            "correlatedSessionId": ancsessionid,
            "screenNameStandard": screen_name_standard,
            "screenNameLegacy": screen_name_legacy,
            "userConsent": "necessary|preference|performance|analytics1st|analytics3rd|advertising1st|advertising3rd|attribution3rd",
            "vendors": "adobemc",
            "vendorConfigurations": "{}",
        }

        # logger.debug(f"UBE JSON Payload before encoding: {json.dumps(ube_data)}") # Less verbose

        json_payload = json.dumps(ube_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        return encoded_payload

    except WebDriverException as e:
        logger.error(f"WebDriverException generating UBE header: {e}")
        # Check validity again after error
        if not is_browser_open(driver):
            logger.error(
                "Session invalid after WebDriverException during UBE generation."
            )
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating UBE Header: {e}", exc_info=True)
        return None
# End of make_ube


def make_newrelic(driver: Optional[WebDriver]) -> Optional[str]: # driver arg kept for consistency, but not used
    """Generates the newrelic header value. No driver needed."""
    try:
        trace_id = uuid.uuid4().hex[:16]
        span_id = uuid.uuid4().hex[:16]
        account_id = "1690570"
        app_id = "1588726612"
        tk = "2611750"

        newrelic_data = {
            "v": [0, 1],
            "d": {
                "ty": "Browser", "ac": account_id, "ap": app_id,
                "id": span_id, "tr": trace_id,
                "ti": int(time.time() * 1000), "tk": tk,
            },
        }
        json_payload = json.dumps(newrelic_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        return encoded_payload
    except Exception as e:
        logger.error(f"Error generating NewRelic header: {e}", exc_info=True)
        return None
# End of make_newrelic


def make_traceparent(
    driver: Optional[WebDriver],
) -> Optional[str]:  # driver arg kept for consistency, but not used
    """Generates the traceparent header value (W3C Trace Context). No driver needed."""
    try:
        version = "00"
        trace_id = uuid.uuid4().hex
        parent_id = uuid.uuid4().hex[:16]
        flags = "01"  # Sampled flag
        traceparent = f"{version}-{trace_id}-{parent_id}-{flags}"
        return traceparent
    except Exception as e:
        logger.error(f"Error generating traceparent header: {e}", exc_info=True)
        return None
# End of make_traceparent


def make_tracestate(driver: Optional[WebDriver]) -> Optional[str]: # driver arg kept for consistency, but not used
    """Generates the tracestate header value (W3C Trace Context). No driver needed."""
    try:
        tk = "2611750"
        account_id = "1690570"
        app_id = "1588726612"
        span_id = uuid.uuid4().hex[:16]
        timestamp = int(time.time() * 1000)
        tracestate = f"{tk}@nr=0-1-{account_id}-{app_id}-{span_id}----{timestamp}"
        return tracestate
    except Exception as e:
        logger.error(f"Error generating tracestate header: {e}", exc_info=True)
        return None
# End of make_tracestate

def get_driver_cookies(
    driver: WebDriver,
) -> Dict[str, str]:
    """
    Retrieves cookies from the Selenium driver as a simple dictionary.
    Removed internal SessionManager creation.
    """
    if not driver:
        logger.warning("Cannot get driver cookies: WebDriver is None.")
        return {}
    try:
        cookies_dict = {
            cookie["name"]: cookie["value"] for cookie in driver.get_cookies()
        }
        return cookies_dict
    except WebDriverException as e:
        logger.error(f"WebDriverException getting driver cookies: {e}")
        # Check session validity directly after error
        if not is_browser_open(driver): # Use helper function
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
    """
    REFINED: Handles the two-step verification page. Waits for user input by checking
    for the disappearance of the 2FA page elements, reducing log noise from repeated API checks.
    """
    if session_manager.driver is None:
        logger.error("SessionManager driver is None. Cannot proceed.")
        return False
    else:
        # Simplified: driver existence already confirmed by `is None` check
        driver = session_manager.driver

    element_wait = selenium_config.element_wait(driver)
    page_wait = selenium_config.page_wait(driver)
    short_wait = selenium_config.short_wait(driver)

    try:
        logger.debug("Handling Two-Factor Authentication (2FA)...")

        # 1. Wait for 2FA page indicator to be present
        try:
            logger.debug(
                "Waiting for 2FA page header using selector: '{}'".format(
                    TWO_STEP_VERIFICATION_HEADER_SELECTOR
                )
            )
            element_wait.until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                )
            )
            logger.debug("2FA page detected.")
        except TimeoutException:
            logger.debug("Did not detect 2FA page header within timeout.")
            if login_status(session_manager):  # Check final status if header not found
                logger.info(
                    "User appears logged in after checking for 2FA page. Proceeding."
                )
                return True
            logger.warning("Assuming 2FA not required or page didn't load correctly.")
            return False  # Fail if 2FA page expected but not found and not logged in

        # 2. Wait for SMS button and click it
        try:
            logger.debug(
                "Waiting for 2FA 'Send Code' (SMS) button using selector: '{}'".format(
                    TWO_FA_SMS_SELECTOR
                )
            )
            sms_button_clickable = short_wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR))
            )

            if sms_button_clickable:
                logger.debug(
                    "Attempting to click 'Send Code' button using JavaScript..."
                )
                driver.execute_script("arguments[0].click();", sms_button_clickable)
                logger.debug("'Send Code' button clicked.")
                # Wait for code input field to appear
                try:
                    logger.debug(
                        "Waiting for 2FA code input field using selector: '{}'".format(
                            TWO_FA_CODE_INPUT_SELECTOR
                        )
                    )
                    # Wait for visibility, as it might be present but hidden initially
                    WebDriverWait(driver, 5).until(
                        EC.visibility_of_element_located(
                            (By.CSS_SELECTOR, TWO_FA_CODE_INPUT_SELECTOR)
                        )
                    )
                    logger.debug(
                        "Code input field appeared after clicking 'Send Code'."
                    )
                except TimeoutException:
                    logger.warning(
                        "Code input field did not appear or become visible after clicking 'Send Code'."
                    )
                    # Might still work if user enters code quickly, but log warning.
                except Exception as e_input:
                    logger.error(
                        f"Error waiting for 2FA code input field: {e_input}. Check selector: {TWO_FA_CODE_INPUT_SELECTOR}"
                    )
                    return False
            else:
                logger.error("'Send Code' button found but not clickable.")
                return False

        except TimeoutException:
            logger.error(
                "Timeout finding or waiting for clickability of the 2FA 'Send Code' button."
            )
            return False
        except ElementNotInteractableException:
            logger.error("'Send Code' button not interactable (potentially obscured).")
            return False
        except Exception as e:
            logger.error(f"Error clicking 2FA 'Send Code' button: {e}", exc_info=True)
            return False

        # 3. Wait for user action by checking for disappearance of 2FA elements
        code_entry_timeout_value = selenium_config.TWO_FA_CODE_ENTRY_TIMEOUT
        logger.warning(
            f"Waiting up to {code_entry_timeout_value}s for user to manually enter 2FA code and submit..."
        )

        start_time = time.time()
        user_action_detected = False
        while time.time() - start_time < code_entry_timeout_value:
            # --- Check if 2FA page elements are STILL present ---
            try:
                # Use a very short wait to check if header is still visible
                WebDriverWait(driver, 0.5).until(
                    EC.visibility_of_element_located(
                        (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                    )
                )
                # Still on 2FA page, continue waiting
                time.sleep(2)  # Poll every 2 seconds
            except TimeoutException:
                # 2FA header (or other key element) DISAPPEARED! User likely submitted code.
                logger.info(
                    "2FA page elements disappeared, assuming user submitted code."
                )
                user_action_detected = True
                break  # Exit the loop
            except WebDriverException as e:
                logger.error(
                    f"WebDriver error checking for 2FA header during wait: {e}"
                )
                # Session might be dead, exit loop and check status below
                break
            except Exception as e:
                logger.error(
                    f"Unexpected error checking for 2FA header during wait: {e}"
                )
                break  # Exit loop on unexpected error

        # --- 4. Final Verification after loop ---
        if user_action_detected:
            logger.info("Re-checking login status after 2FA page disappearance...")
            time.sleep(1)  # Short pause for page transition to settle
            final_status = login_status(session_manager)
            if final_status is True:
                logger.info(
                    "User completed 2FA successfully (login confirmed after page change)."
                )
                return True
            else:
                logger.error(
                    "2FA page disappeared, but final login status check failed or returned False."
                )
                return False
        else:  # Loop timed out
            logger.error(
                f"Timed out ({code_entry_timeout_value}s) waiting for user 2FA action (2FA page elements did not disappear)."
            )
            # Optional: Check status one last time even on timeout
            # final_status = login_status(session_manager)
            # if final_status is True:
            #     logger.warning("Timed out waiting, but final login status check PASSED unexpectedly.")
            #     return True
            return False

    except WebDriverException as e:
        logger.error(f"WebDriverException during 2FA handling: {e}")
        if session_manager and not is_browser_open(driver):  # Check if session died
            logger.error("Session invalid after WebDriverException during 2FA.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during 2FA handling: {e}", exc_info=True)
        return False
# End of handle_twoFA

# Login 4
def enter_creds(driver: WebDriver) -> bool:  # Type hint driver and return
    """REFINED: Enters username/password, attempts to click Sign In button robustly."""
    element_wait = selenium_config.element_wait(driver)
    short_wait = selenium_config.short_wait(
        driver
    )  # Use shorter wait for button perhaps

    # Add a small initial delay before interacting with the form
    time.sleep(random.uniform(0.5, 1.0))

    try:
        logger.debug("Entering Credentials and Signing In...")

        # --- Username ---
        logger.debug("Waiting for username input field...")
        username_input = element_wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))
        )
        logger.debug("Username input field found.")
        try:
            username_input.click()
            time.sleep(0.2)  # Short pause after click
            username_input.clear()
            time.sleep(0.2)  # Short pause after clear
        except Exception as e:
            logger.warning(
                f"Issue clicking/clearing username field: {e}. Attempting JS clear."
            )
            try:
                driver.execute_script("arguments[0].value = '';", username_input)
            except Exception as js_e:
                logger.error(f"Failed to clear username field via JS: {js_e}")
                return False  # Fail if cannot clear

        ancestry_username = config_instance.ANCESTRY_USERNAME
        if not ancestry_username:
            raise ValueError("ANCESTRY_USERNAME configuration is missing or empty.")
        logger.debug(f"Entering username: {ancestry_username}")
        username_input.send_keys(ancestry_username)
        logger.debug("Username entered.")
        time.sleep(0.3)  # Pause after entering username

        # --- Password ---
        logger.debug("Waiting for password input field...")
        password_input = element_wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))
        )
        logger.debug("Password input field found.")
        try:
            password_input.click()
            time.sleep(0.2)
            password_input.clear()
            time.sleep(0.2)
        except Exception as e:
            logger.warning(
                f"Issue clicking/clearing password field: {e}. Attempting JS clear."
            )
            try:
                driver.execute_script("arguments[0].value = '';", password_input)
            except Exception as js_e:
                logger.error(f"Failed to clear password field via JS: {js_e}")
                return False

        ancestry_password = config_instance.ANCESTRY_PASSWORD
        if not ancestry_password:
            raise ValueError("ANCESTRY_PASSWORD configuration is missing or empty.")
        logger.debug("Entering password: ***")
        password_input.send_keys(ancestry_password)
        logger.debug("Password entered.")
        time.sleep(0.5)  # Pause after entering password, before finding button

        # --- Sign In Button ---
        sign_in_button = None
        try:
            # Wait for button to be present first
            logger.debug("Waiting for sign in button presence...")
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR)
                )
            )
            # Then specifically wait for clickability
            logger.debug("Waiting for sign in button to be clickable...")
            sign_in_button = short_wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))
            )
            logger.debug("Sign in button located and deemed clickable.")

        except TimeoutException:
            logger.error("Sign in button not found or not clickable within timeout.")
            # As a fallback, try submitting via Enter key on password field
            logger.warning("Attempting fallback: Sending RETURN key to password field.")
            try:
                password_input.send_keys(Keys.RETURN)
                logger.info("Fallback RETURN key sent to password field.")
                return True  # Assume submission initiated
            except Exception as key_e:
                logger.error(f"Failed to send RETURN key: {key_e}")
                return False  # Both button click and key press failed
        except Exception as find_e:
            logger.error(f"Unexpected error finding sign in button: {find_e}")
            return False

        # --- Attempt Click Actions ---
        click_successful = False
        if sign_in_button:
            # Attempt 1: Standard Click
            try:
                sign_in_button.click()
                logger.debug("Standard click executed on sign in button.")
                click_successful = True  # Mark as successful
            except ElementClickInterceptedException:
                logger.warning("Standard click intercepted for sign in button.")
                # Proceed to JS click attempt
            except ElementNotInteractableException as eni_e:
                logger.warning(
                    f"Sign in button not interactable for standard click: {eni_e}"
                )
                # Proceed to JS click attempt
            except Exception as click_e:
                logger.error(
                    f"Error during standard click on sign in button: {click_e}"
                )
                # Proceed to JS click attempt (unless it was a fatal error?)

            # Attempt 2: JavaScript Click (if standard click failed or was intercepted/uninteractable)
            if not click_successful:
                try:
                    logger.warning("Attempting JavaScript click on sign in button...")
                    driver.execute_script("arguments[0].click();", sign_in_button)
                    logger.info("JavaScript click executed on sign in button.")
                    click_successful = True  # Mark successful
                except Exception as js_click_e:
                    logger.error(
                        f"Error during JavaScript click on sign in button: {js_click_e}"
                    )

            # Attempt 3: Send Enter Key (If both clicks failed)
            if not click_successful:
                logger.warning(
                    "Both standard and JS clicks failed or were problematic. Attempting fallback: Sending RETURN key to password field."
                )
                try:
                    password_input.send_keys(Keys.RETURN)
                    logger.info("Fallback RETURN key sent to password field.")
                    click_successful = True  # Assume submission initiated
                except Exception as key_e:
                    logger.error(
                        f"Failed to send RETURN key as final fallback: {key_e}"
                    )

        return click_successful  # Return True only if some click/submit action was successfully executed

    except TimeoutException as e:  # Catch timeout finding user/pass fields
        logger.error(f"Timeout finding username or password input field: {e}")
        return False
    except NoSuchElementException as e:  # Should be caught by Wait, but safety check
        logger.error(f"Username or password input field not found (NoSuchElement): {e}")
        return False
    except ValueError as ve:  # Catch missing config error
        logger.critical(f"Configuration Error: {ve}")
        return False
    except WebDriverException as e:  # Catch general WebDriver errors
        logger.error(f"WebDriver error entering credentials: {e}")
        if not is_browser_open(driver):  # Check session validity
            logger.error("Session invalid during credential entry.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error entering credentials: {e}", exc_info=True)
        return False
# End of enter_creds

# Login 3
@retry(MAX_RETRIES=2, BACKOFF_FACTOR=1, MAX_DELAY=3)  # Add retry for consent handling
def consent(driver: WebDriver) -> bool:  # Type hint driver and return
    """Handles cookie consent modal by removing it or clicking the specific accept button."""
    if not driver:
        return False  # Added safety check

    try:
        logger.debug(
            "Checking for cookie consent overlay using selector: '{}'".format(
                COOKIE_BANNER_SELECTOR
            )
        )
        overlay_element = None
        try:
            # Use a short wait to find the banner element
            overlay_element = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                )
            )
            logger.debug("Cookie consent overlay DETECTED.")
        except TimeoutException:
            logger.debug(
                "Cookie consent overlay not found. Assuming no consent needed."
            )
            return True

        # Attempt 1: Remove overlay with JavaScript
        removed_via_js = False
        if overlay_element:
            try:
                logger.debug(
                    "Attempting to remove cookie consent overlay using Javascript..."
                )
                driver.execute_script("arguments[0].remove();", overlay_element)
                # Verify removal - check if element is gone after short pause
                time.sleep(0.5)
                try:
                    driver.find_element(By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                    logger.warning(
                        "Consent overlay still present after JS removal attempt."
                    )
                except NoSuchElementException:
                    logger.debug(
                        "Cookie consent overlay REMOVED successfully via Javascript."
                    )
                    removed_via_js = True
                    return True  # Successfully removed
            except WebDriverException as js_err:
                logger.warning(f"Error removing consent overlay via JS: {js_err}")
            except Exception as e:
                logger.warning(f"Unexpected error during JS removal of consent: {e}")

        # Attempt 2: Click the specific "Accept" button (if JS removal failed)
        if not removed_via_js:
            logger.debug(
                "Attempting to find and click the specific 'Accept' button using selector: '{}'".format(
                    consent_ACCEPT_BUTTON_SELECTOR
                )
            )
            try:
                # Use the specific selector provided
                accept_button = WebDriverWait(
                    driver, 2
                ).until(  # Short wait for specific button
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, consent_ACCEPT_BUTTON_SELECTOR)
                    )
                )
                if accept_button:
                    logger.info("Found specific clickable accept button.")
                    try:
                        accept_button.click()
                        logger.info("Clicked accept button successfully.")
                        # Optional: Wait a moment for banner to disappear after click
                        time.sleep(1)
                        # Verify removal again
                        try:
                            driver.find_element(By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                            logger.warning(
                                "Consent overlay still present after clicking accept button."
                            )
                            return (
                                False  # Explicitly fail if banner persists after click
                            )
                        except NoSuchElementException:
                            logger.debug(
                                "Consent overlay gone after clicking accept button."
                            )
                        return True  # Success
                    except ElementClickInterceptedException:
                        logger.warning(
                            "Click intercepted for accept button, trying JS click..."
                        )
                        try:
                            driver.execute_script(
                                "arguments[0].click();", accept_button
                            )
                            logger.info("Clicked accept button via JS successfully.")
                            time.sleep(1)
                            # Verify removal again
                            try:
                                driver.find_element(
                                    By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR
                                )
                                logger.warning(
                                    "Consent overlay still present after JS clicking accept button."
                                )
                                return False
                            except NoSuchElementException:
                                logger.debug(
                                    "Consent overlay gone after JS clicking accept button."
                                )
                            return True  # Assume success
                        except Exception as js_click_err:
                            logger.error(
                                f"Failed JS click for accept button: {js_click_err}"
                            )
                    except Exception as click_err:
                        logger.error(f"Error clicking accept button: {click_err}")
            except TimeoutException:
                logger.warning(
                    "Specific accept button '{}' not found or not clickable.".format(
                        consent_ACCEPT_BUTTON_SELECTOR
                    )
                )
                # Fallback: Maybe try generic selectors again? Or just fail? Let's fail for now.
            except Exception as find_err:
                logger.error(
                    f"Error finding/clicking specific accept button: {find_err}"
                )

            logger.warning(
                "Could not remove consent overlay via JS or clicking the specific accept button."
            )
            return False  # Failed to handle consent

    except WebDriverException as e:
        logger.error(f"WebDriver error handling cookie consent: {e}")
        try:
            if not is_browser_open(driver):
                logger.error("Session invalid during consent handling.")
        except Exception:
            pass
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error handling cookie consent overlay: {e}", exc_info=True
        )
        return False

    # Ensure a return value for all code paths
    return False
# End of consent

# Login 2
def log_in(session_manager: "SessionManager") -> str:  # Return specific status strings
    """REVISED: Logs in to Ancestry.com, uses specific selectors and revised nav_to_page."""
    # Ensure SessionManager type hint works if defined later
    from utils import (
        SessionManager,
        nav_to_page,
        consent,
        enter_creds,
        is_elem_there,
        handle_twoFA,
        login_status,
        urljoin,
        config_instance,
        selenium_config,
        USERNAME_INPUT_SELECTOR,
        TWO_STEP_VERIFICATION_HEADER_SELECTOR,
        FAILED_LOGIN_SELECTOR,
        is_browser_open,
    )

    driver = session_manager.driver
    if not driver:
        logger.error("Login failed: WebDriver not available.")
        return "LOGIN_ERROR_NO_DRIVER"

    page_wait = selenium_config.page_wait(driver)
    signin_url = urljoin(config_instance.BASE_URL, "account/signin")

    try:
        # 1. Navigate to signin page (using revised nav_to_page)
        logger.info(f"Navigating to signin page: {signin_url}")
        # Wait for the username input as confirmation the page loaded.
        if not nav_to_page(
            driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager
        ):
            # --- Check if failure was due to being *already* logged in ---
            logger.debug("Navigation to signin page failed. Checking current login status as potential cause...")
            current_status = login_status(session_manager)
            if current_status is True:
                 logger.info("Navigation to signin page failed, but now detected as logged in. Login considered successful.")
                 return "LOGIN_SUCCEEDED"
            # --- End check ---
            logger.error("Failed to navigate to or load the login page properly (and not already logged in).")
            return "LOGIN_FAILED_NAVIGATION"
        logger.debug("Successfully navigated to signin page.")

        # 2. Handle cookie consent (banner overlay) - if present
        if not consent(driver):
            logger.warning("Failed to handle consent banner, login might be impacted.")

        # 3. Enter login credentials and submit using specific selectors
        if not enter_creds(driver):
            logger.error("Failed to enter login credentials.")
            # Check for the specific invalid credentials alert
            try:
                # Use presence check for the specific error div
                WebDriverWait(driver, 1).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, FAILED_LOGIN_SELECTOR)
                    )
                )
                logger.error(
                    "Login failed: Specific 'Invalid Credentials' alert found."
                )
                return "LOGIN_FAILED_BAD_CREDS"
            except TimeoutException:
                # Specific alert not found, check for generic ones
                generic_alert_selector = "div.alert[role='alert']"
                try:
                    alert_element = WebDriverWait(driver, 0.5).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, generic_alert_selector)
                        )
                    )
                    logger.error(
                        f"Login failed: Generic alert found: '{alert_element.text if alert_element else 'Unknown'}'."
                    )
                    return "LOGIN_FAILED_ERROR_DISPLAYED"
                except TimeoutException:
                    logger.error(
                        "Login failed: Credential entry failed, but no specific or generic alert found."
                    )
                    return (
                        "LOGIN_FAILED_CREDS_ENTRY"  # Return entry failure if no alert
                    )
            except Exception as e:
                logger.warning(
                    f"Error checking for login error message after cred entry failed: {e}"
                )
                return "LOGIN_FAILED_CREDS_ENTRY"

        logger.debug("Credentials submitted. Waiting for potential page change...")
        time.sleep(random.uniform(3.0, 5.0))

        # --- Refresh current URL after submit ---
        try:
            current_url_after_submit = driver.current_url
            logger.debug(f"URL after credential submission: {current_url_after_submit}")
        except WebDriverException as e:
            logger.warning(f"Could not get URL immediately after login submit: {e}")
            current_url_after_submit = ""

        # 4. Check for 2-step verification using specific selector
        two_fa_present = False
        try:
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                )
            )
            two_fa_present = True
        except TimeoutException:
            two_fa_present = False
        except WebDriverException as e:
            logger.error(f"WebDriver error checking for 2FA page: {e}")
            status = login_status(session_manager)
            if status is True:
                return "LOGIN_SUCCEEDED"
            if status is False:
                return "LOGIN_FAILED_UNKNOWN"
            return "LOGIN_FAILED_WEBDRIVER"

        if two_fa_present:
            logger.info("Two-step verification page detected.")
            if handle_twoFA(
                session_manager
            ):  # handle_twoFA uses specific selectors now
                logger.info("Two-step verification handled successfully.")
                if login_status(session_manager):
                    return "LOGIN_SUCCEEDED"
                else:
                    logger.error(
                        "Login status check failed AFTER successful 2FA handling report."
                    )
                    return "LOGIN_FAILED_POST_2FA_VERIFY"
            else:
                logger.error("Two-step verification handling failed.")
                return "LOGIN_FAILED_2FA_HANDLING"

        # 5. If no 2-step verification, check for successful login directly
        else:
            logger.debug(
                "Two-step verification page not detected. Checking login status directly."
            )
            login_check_result = login_status(
                session_manager
            )  # Uses specific selectors internally now
            if login_check_result is True:
                logger.debug("No 2FA required.")
                return "LOGIN_SUCCEEDED"
            elif login_check_result is False:
                # Check again for specific error messages
                try:
                    WebDriverWait(driver, 1).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, FAILED_LOGIN_SELECTOR)
                        )
                    )
                    logger.error(
                        "Login failed: Specific 'Invalid Credentials' alert found (post-check)."
                    )
                    return "LOGIN_FAILED_BAD_CREDS"
                except TimeoutException:
                    # Specific alert not found, check generic
                    generic_alert_selector = "div.alert[role='alert']"
                    try:
                        alert_element = WebDriverWait(driver, 0.5).until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, generic_alert_selector)
                            )
                        )
                        logger.error(
                            f"Login failed: Generic alert found (post-check): '{alert_element.text if alert_element else 'Unknown'}'."
                        )
                        return "LOGIN_FAILED_ERROR_DISPLAYED"
                    except TimeoutException:
                        # No error alert found, check if still on login page
                        try:
                            if driver.current_url.startswith(signin_url):
                                logger.error(
                                    "Login failed: Still on login page (post-check), no specific error message found."
                                )
                                return "LOGIN_FAILED_STUCK_ON_LOGIN"
                            else:
                                logger.error(
                                    "Login failed: Login status is False, no 2FA, no specific error msg found, not on login page."
                                )
                                return "LOGIN_FAILED_UNKNOWN"
                        except WebDriverException:
                            logger.error(
                                "Login failed: Login status False, WebDriverException getting URL."
                            )
                            return "LOGIN_FAILED_WEBDRIVER"
                    except Exception as e:
                        logger.error(
                            f"Login failed: Error checking for generic alert (post-check): {e}"
                        )
                        return "LOGIN_FAILED_UNKNOWN"
                except Exception as e:
                    logger.error(
                        f"Login failed: Error checking for specific alert (post-check): {e}"
                    )
                    return "LOGIN_FAILED_UNKNOWN"
            else:  # login_status returned None (critical error)
                logger.error(
                    "Login failed: Critical error during final login status check."
                )
                return "LOGIN_FAILED_STATUS_CHECK_ERROR"

    except TimeoutException as e:
        logger.error(f"Timeout during login process: {e}", exc_info=False)
        return "LOGIN_FAILED_TIMEOUT"
    except WebDriverException as e:
        logger.error(f"WebDriverException during login: {e}", exc_info=False)
        if session_manager and not is_browser_open(driver):
            logger.error("Session became invalid during login attempt.")
        return "LOGIN_FAILED_WEBDRIVER"
    except Exception as e:
        logger.error(f"An unexpected error occurred during login: {e}", exc_info=True)
        return "LOGIN_FAILED_UNEXPECTED"
# End of log_in

# Login 1
@retry(MAX_RETRIES=2)
def login_status(session_manager: SessionManager) -> Optional[bool]:
    """
    REVISED V4: Checks login status using API (/header/dna) and robust UI selectors as fallback.
    Returns True if likely logged in, False if likely not, None if critical error.
    Ensures cookie sync before API check.
    """
    logger.debug("Checking login status (API prioritized)...")
    api_check_result: Optional[bool] = None
    ui_check_result: Optional[bool] = None

    try:
        if not isinstance(session_manager, SessionManager):
            logger.error(
                f"Invalid argument: Expected SessionManager, got {type(session_manager)}."
            )
            return None

        # --- 1. Basic WebDriver Session Check ---
        if (
            not session_manager.is_sess_valid()
        ):  # Uses the more robust check from SessionManager
            logger.debug("Login status: Session invalid or browser closed.")
            return False

        driver = session_manager.driver
        if driver is None:
             logger.error("Login status check: Driver is None.")
             return None # Cannot proceed without driver

        # --- 2. API-Based Login Verification (PRIORITY) ---
        logger.debug("Attempting API login verification...")
        # --- Ensure cookies are synced BEFORE API check ---
        try:
            logger.debug("Syncing cookies before API login check...")
            session_manager._sync_cookies()
        except Exception as sync_e:
             logger.warning(f"Error syncing cookies before API login check: {sync_e}")
             # Proceed with API check cautiously, might fail if cookies were crucial
        # --- End cookie sync ---
        api_check_result = session_manager._verify_api_login_status() # Now uses /header/dna

        if api_check_result is True:
            logger.debug("Login status confirmed via API check.")
            return True # API confirmed login, no need for UI check

        elif api_check_result is False:
            logger.debug("API login verification indicates user NOT logged in. Falling back to UI check.")
        else: # API check returned None (critical error occurred, logged in _verify_api_login_status)
            logger.warning("API login check returned None (critical error). Falling back to UI check.")

        # --- 3. UI-Based Login Verification (FALLBACK) ---
        logger.debug("Performing fallback UI login check...")

        # --- 3a. Handle Consent Overlay ---
        # Removed consent check here - it should be handled by navigation/login process if needed.
        # Trying to handle consent here might interfere with checking the actual page state.

        # --- 3b. Check for Logged-Out UI Element (Absence is good) ---
        login_button_selector = LOG_IN_BUTTON_SELECTOR # e.g., '#secMenuItem-SignIn > span'
        logger.debug(f"Attempting UI verification: Checking absence of login button: '{login_button_selector}'")
        # Use a short wait (e.g., 1-2 seconds) to check if the login button *is* present
        login_button_present = False
        try:
            WebDriverWait(driver, 2).until(
                 EC.visibility_of_element_located((By.CSS_SELECTOR, login_button_selector))
            )
            login_button_present = True
            logger.debug("Login button FOUND during UI check.")
        except TimeoutException:
            # Login button NOT found within timeout - good sign!
            logger.debug("Login button NOT found during UI check (good indication).")
            login_button_present = False
        except Exception as e:
            logger.warning(f"Error checking for login button presence: {e}")
            # Proceed cautiously, assume maybe not present if error occurred

        if login_button_present:
            # If login button IS reliably found, user is definitely logged out.
            logger.debug("Login status confirmed NOT logged in via UI check (login button found).")
            return False
        else:
            # Login button NOT found. Now check for logged-in indicator as confirmation.
            # --- 3c. Check for Logged-In UI Element ---
            logged_in_selector = CONFIRMED_LOGGED_IN_SELECTOR # e.g., '#navAccount'
            logger.debug(f"Login button absent. Checking presence of logged-in element: '{logged_in_selector}'")
            ui_element_present = is_elem_there(
                driver, By.CSS_SELECTOR, logged_in_selector, wait=3 # Short wait okay here
            )

            if ui_element_present:
                logger.debug(
                    "Login status confirmed via fallback UI check (login button absent AND logged-in element found)."
                )
                return True
            else:
                # --- 3d. Handle Ambiguity ---
                logger.warning(
                    "Login status ambiguous: API failed/false, login button absent, AND logged-in element absent."
                )
                current_url_context = "Unknown"
                try:
                    current_url_context = driver.current_url
                except Exception:
                    pass
                logger.debug(f"Ambiguous login state at URL: {current_url_context}")
                # Defaulting to False in ambiguous state is safer to trigger login if needed
                return False

    # --- Handle Exceptions ---
    except WebDriverException as e:
        logger.error(f"WebDriverException during login_status check: {e}")
        if session_manager and not is_browser_open(driver):  # Use basic check here
            logger.error("Session became invalid during login_status check.")
            session_manager.close_sess()
        return None
    except Exception as e:
        logger.critical(
            f"CRITICAL Unexpected error during login_status check: {e}", exc_info=True
        )
        return None
# End of login_status


# ------------------------------------------------------------------------------------
# browser
# ------------------------------------------------------------------------------------

def is_browser_open(driver: Optional[WebDriver]) -> bool:  # Type hint driver
    """
    Checks if the browser window associated with the driver instance is open.
    (Copied from SessionManager for standalone use in helpers)
    """
    if driver is None:
        return False
    try:
        _ = driver.window_handles
        return True
    except WebDriverException as e:
        if (
            "invalid session id" in str(e).lower()
            or "target closed" in str(e).lower()
            or "disconnected" in str(e).lower()
            or "no such window" in str(e).lower() # Added NoSuchWindowException check
        ):
            logger.debug(f"Browser appears closed or session invalid: {type(e).__name__}")
            return False
        else:
            logger.warning(
                f"WebDriverException checking browser status (but might still be open): {e}"
            )
            return False
    except Exception as e:
        logger.error(f"Unexpected error checking browser status: {e}", exc_info=True)
        return False
# End of is_browser_open


def restore_sess(driver: WebDriver) -> bool:  # Type hint driver & return
    """Restores session state including cookies, local, and session storage, domain aware."""
    if not driver:
        logger.error("Cannot restore session: WebDriver is None.")
        return False

    cache_dir = config_instance.CACHE_DIR
    domain = ""  # Initialize domain

    try:
        current_url = driver.current_url
        parsed_url = urlparse(current_url)
        domain = parsed_url.netloc.replace("www.", "").split(":")[0]
        if not domain:  # Handle case where domain parsing fails
            raise ValueError("Could not extract valid domain from current URL")
    except Exception as e:
        logger.error(
            f"Could not parse current URL ({driver.current_url if driver else 'N/A'}) for session restore: {e}. Using fallback."
        )
        try:
            parsed_base = urlparse(config_instance.BASE_URL)
            domain = parsed_base.netloc.replace("www.", "").split(":")[0]
            if not domain:
                raise ValueError("Could not extract domain from BASE_URL")
            logger.warning(
                f"Falling back to base URL domain for session restore: {domain}"
            )
        except Exception as base_e:
            logger.critical(
                f"Could not determine domain from base URL either: {base_e}. Cannot restore."
            )
            return False

    logger.debug(
        f"Attempting to restore session state from cache for domain: {domain}..."
    )
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
                logger.error(
                    f"Unexpected error reading '{filepath.name}': {e}", exc_info=True
                )
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
                if (
                    isinstance(cookie, dict)
                    and "name" in cookie
                    and "value" in cookie
                    and "domain" in cookie
                ):
                    try:
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
        logger.warning(f"Cookie cache file '{cookies_file}' invalid format.")

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
        except WebDriverException as e:
            logger.error(f"Error restoring localStorage: {e}")
        except Exception as e:
            logger.error(f"Unexpected error restoring localStorage: {e}", exc_info=True)
    elif local_storage is not None:
        logger.warning(
            f"localStorage cache file '{local_storage_file}' invalid format."
        )

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
        except WebDriverException as e:
            logger.error(f"Error restoring sessionStorage: {e}")
        except Exception as e:
            logger.error(
                f"Unexpected error restoring sessionStorage: {e}", exc_info=True
            )
    elif session_storage is not None:
        logger.warning(
            f"sessionStorage cache file '{session_storage_file}' invalid format."
        )

    # Perform hard refresh if anything was restored
    if restored_something:
        logger.info("Session state restored from cache. Performing hard refresh...")
        try:
            driver.refresh()
            WebDriverWait(driver, selenium_config.PAGE_TIMEOUT).until(  # Wait for load
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
    if not is_browser_open(driver):
        logger.warning("Browser session invalid/closed. Skipping session state save.")
        return

    cache_dir = config_instance.CACHE_DIR
    domain = ""  # Initialize

    try:
        current_url = driver.current_url
        parsed_url = urlparse(current_url)
        domain = parsed_url.netloc.replace("www.", "").split(":")[0]
        if not domain:
            raise ValueError("Could not extract domain from current URL")
    except Exception as e:
        logger.error(
            f"Could not parse current URL ({driver.current_url if driver else 'N/A'}) for saving state: {e}. Using fallback."
        )
        try:
            parsed_base = urlparse(config_instance.BASE_URL)
            domain = parsed_base.netloc.replace("www.", "").split(":")[0]
            if not domain:
                raise ValueError("Could not extract domain from BASE_URL")
            logger.warning(
                f"Falling back to base URL domain for saving state: {domain}"
            )
        except Exception as base_e:
            logger.critical(
                f"Could not determine domain from base URL either: {base_e}. Cannot save state."
            )
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
        except (TypeError, IOError) as e:
            logger.error(f"Error writing JSON to '{filepath.name}': {e}")
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error writing JSON to '{filepath.name}': {e}",
                exc_info=True,
            )
            return False

    # Save Cookies
    cookies_file = f"session_cookies_{domain}.json"
    try:
        cookies = driver.get_cookies()
        if cookies is not None:
            if safe_json_write(cookies, cookies_file):
                logger.debug(f"Cookies saved.")  # Less verbose
        else:
            logger.warning("Could not retrieve cookies from driver.")
    except WebDriverException as e:
        logger.error(f"Error getting cookies: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving cookies: {e}", exc_info=True)

    # Save Local Storage
    local_storage_file = f"session_local_storage_{domain}.json"
    try:
        local_storage = driver.execute_script(
            "return window.localStorage ? {...window.localStorage} : {};"
        )  # Return empty dict if null
        if local_storage:  # Check if not empty
            if safe_json_write(local_storage, local_storage_file):
                logger.debug(f"localStorage saved.")
        else:
            logger.debug("localStorage not available or empty.")
    except WebDriverException as e:
        logger.error(f"Error getting localStorage: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving localStorage: {e}", exc_info=True)

    # Save Session Storage
    session_storage_file = f"session_session_storage_{domain}.json"
    try:
        session_storage = driver.execute_script(
            "return window.sessionStorage ? {...window.sessionStorage} : {};"
        )  # Return empty dict if null
        if session_storage:
            if safe_json_write(session_storage, session_storage_file):
                logger.debug(f"sessionStorage saved.")
        else:
            logger.debug("sessionStorage not available or empty.")
    except WebDriverException as e:
        logger.error(f"Error getting sessionStorage: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving sessionStorage: {e}", exc_info=True)

    # logger.debug(f"Session state save attempt finished for domain: {domain}.") # Less verbose
# End of save_state


def close_tabs(driver: WebDriver):  # Type hint driver
    """Closes all but the first tab in the given driver."""
    if not driver:
        return
    logger.debug("Closing extra tabs...")
    try:
        handles = driver.window_handles
        if len(handles) <= 1:
            logger.debug("No extra tabs to close.")
            return

        original_handle = driver.current_window_handle  # Store current handle
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
        elif (
            first_handle in remaining_handles
        ):  # Fallback to first handle if original was closed
            driver.switch_to.window(first_handle)
            logger.debug(f"Switched back to first tab: {first_handle}")
        elif (
            remaining_handles
        ):  # Switch to whatever is left if both original and first are gone
            driver.switch_to.window(remaining_handles[0])
            logger.warning(
                f"Original and first tab closed, switched to remaining: {remaining_handles[0]}"
            )
        else:
            logger.error("All tabs were closed unexpectedly.")
            # This indicates a problem, maybe the driver session died.

    except NoSuchWindowException:
        logger.warning(
            "Attempted to close or switch to a tab that no longer exists during cleanup."
        )
    except WebDriverException as e:
        logger.error(f"WebDriverException in close_tabs: {e}")
        temp_sm = SessionManager()  # Create temporary manager
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
    url: str,  # url defaults removed, must be provided
    selector: str = "body",  # Default selector changed to body
    session_manager: Optional["SessionManager"] = None,  # Use forward reference
) -> bool:
    """
    REVISED: Navigates, handles redirects (using specific selectors for login/MFA pages),
    verifies successful navigation.
    """
    if not driver:
        logger.error("Navigation failed: WebDriver instance is None.")
        return False
    if not url:
        logger.error("Navigation failed: Target URL is required.")
        return False

    from utils import (
        SessionManager,
        is_browser_open,
        log_in,
        login_status,
        urljoin,
        config_instance,
        selenium_config,
        USERNAME_INPUT_SELECTOR,
        TWO_STEP_VERIFICATION_HEADER_SELECTOR,
        TEMP_UNAVAILABLE_SELECTOR,
        PAGE_NO_LONGER_AVAILABLE_SELECTOR,
        _check_for_unavailability,
    )  # Add necessary imports

    max_attempts = config_instance.MAX_RETRIES
    page_timeout = selenium_config.PAGE_TIMEOUT
    element_timeout = selenium_config.ELEMENT_TIMEOUT

    target_url_parsed = urlparse(url)
    target_url_base = f"{target_url_parsed.scheme}://{target_url_parsed.netloc}{target_url_parsed.path}".rstrip(
        "/"
    )
    signin_page_url_base = urljoin(config_instance.BASE_URL, "account/signin").rstrip(
        "/"
    )  # Normalize
    mfa_page_url_base = urljoin(config_instance.BASE_URL, "account/signin/mfa/").rstrip(
        "/"
    )  # Normalize

    unavailability_selectors = {
        TEMP_UNAVAILABLE_SELECTOR: ("refresh", 5),
        PAGE_NO_LONGER_AVAILABLE_SELECTOR: ("skip", 0),
    }

    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Nav Attempt {attempt}/{max_attempts} to: {url}")
        landed_url = ""

        try:
            # --- 1. Check WebDriver session validity ---
            if not is_browser_open(driver):
                logger.error(
                    f"Navigation failed (Attempt {attempt}): Browser session is not valid."
                )
                if session_manager:
                    logger.warning(
                        "Session invalid before navigation. Attempting restart..."
                    )
                    if session_manager.restart_sess():
                        logger.info("Session restarted. Retrying navigation...")
                        driver = session_manager.driver
                        if not driver:
                            return False
                        continue
                    else:
                        logger.error("Session restart failed.")
                        return False
                else:
                    return False

            # --- 2. Perform Navigation ---
            logger.debug(f"Executing driver.get('{url}')...")
            driver.get(url)
            WebDriverWait(driver, page_timeout).until(
                lambda d: d.execute_script("return document.readyState")
                in ["complete", "interactive"]
            )
            time.sleep(random.uniform(0.5, 1.5))  # Wait for JS redirects

            # --- 3. Post-Navigation Verification ---
            try:
                landed_url = driver.current_url
                landed_url_parsed = urlparse(landed_url)
                landed_url_base = f"{landed_url_parsed.scheme}://{landed_url_parsed.netloc}{landed_url_parsed.path}".rstrip(
                    "/"
                )
                logger.debug(
                    f"Nav Attempt {attempt} landed on URL base: {landed_url_base}"
                )
            except WebDriverException as e:
                logger.error(
                    f"Failed to get URL after get() (Attempt {attempt}): {e}. Retrying."
                )
                continue

            # --- 3a. Check for Unexpected Login/MFA Redirects ---
            # Check if landed on MFA page by checking for its specific header
            is_on_mfa_page = False
            try:
                WebDriverWait(driver, 1).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                    )
                )
                is_on_mfa_page = True
            except TimeoutException:
                is_on_mfa_page = False
            except WebDriverException as e:
                logger.warning(
                    f"WebDriverException checking for MFA header: {e}"
                )  # Continue cautiously

            if is_on_mfa_page:
                logger.warning("Landed on MFA page unexpectedly during navigation.")
                # Fail navigation as the target page wasn't reached and MFA needs intervention.
                return False

            # Check if landed on Login page by checking for username input
            is_on_login_page = False
            # Only check if the landed URL base looks like the signin page OR if target wasn't signin page
            if (
                landed_url_base == signin_page_url_base
                or target_url_base != signin_page_url_base
            ):
                try:
                    WebDriverWait(driver, 1).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR)
                        )
                    )
                    is_on_login_page = True
                except TimeoutException:
                    is_on_login_page = False
                except WebDriverException as e:
                    logger.warning(
                        f"WebDriverException checking for Login username input: {e}"
                    )  # Continue cautiously

            # If landed on login page AND target wasn't login page
            if is_on_login_page and target_url_base != signin_page_url_base:
                logger.warning("Landed on Login page unexpectedly.")
                if session_manager:
                    logger.info("Attempting re-login...")
                    login_stat = login_status(session_manager)
                    if login_stat is True:
                        logger.info(
                            "Login status confirmed OK after landing on login page. Retrying navigation."
                        )
                        continue
                    else:
                        login_result = log_in(session_manager)
                        if login_result == "LOGIN_SUCCEEDED":
                            logger.info(
                                "Re-login successful. Retrying original navigation..."
                            )
                            continue
                        else:
                            logger.error(f"Re-login failed ({login_result}).")
                            return False
                else:
                    logger.error(
                        "Landed on login page, no SessionManager for re-login."
                    )
                    return False

            # --- 3b. Check if landed on the target URL base ---
            if landed_url_base != target_url_base:
                # Special handling for expected redirects AFTER login attempt
                # If target was signin page, and we landed on base URL, check login status
                if target_url_base == signin_page_url_base and landed_url_base == urlparse(config_instance.BASE_URL).path.rstrip('/'):
                     logger.debug("Redirected from signin page to base URL. Checking login status...")
                     time.sleep(1) # Allow slight delay
                     if session_manager and login_status(session_manager) is True:
                          logger.info("Redirected from signin, but now logged in. Navigation OK.")
                          return True # Treat as success if logged in

                logger.warning(
                    f"Navigation landed on unexpected URL base: '{landed_url_base}' (Expected: '{target_url_base}')"
                )
                action, wait_time = _check_for_unavailability(
                    driver, unavailability_selectors
                )
                if action == "skip":
                    return False
                elif action == "refresh":
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning("Wrong URL, no specific message. Retrying.")
                    continue

            # --- 4. Wait for Target Selector ---
            wait_selector = selector if selector else "body"
            logger.debug(
                f"On correct URL base. Waiting up to {element_timeout}s for selector: '{wait_selector}'"
            )
            try:
                WebDriverWait(driver, element_timeout).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, wait_selector))
                )
                logger.debug(f"Navigation successful:\n{url}")
                return True  # SUCCESS!

            except TimeoutException:
                current_url_on_timeout = landed_url  # Use already fetched URL
                logger.warning(
                    f"Timeout waiting for selector '{wait_selector}' at {current_url_on_timeout} (URL base was correct)."
                )
                action, wait_time = _check_for_unavailability(
                    driver, unavailability_selectors
                )
                if action == "skip":
                    return False
                elif action == "refresh":
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning("Timeout, no unavailability message. Retrying.")
                    continue

        # --- Handle Exceptions During the Attempt ---
        except UnexpectedAlertPresentException as alert_e:
            logger.warning(
                f"Unexpected alert (Attempt {attempt}): {alert_e.alert_text}"
            )
            try:
                driver.switch_to.alert.accept()
                logger.info("Alert accepted.")
            except Exception as accept_e:
                logger.error(f"Failed to accept alert: {accept_e}")
                return False
            continue

        except WebDriverException as wd_e:
            logger.error(
                f"WebDriverException during navigation (Attempt {attempt}): {wd_e}",
                exc_info=False,
            )
            if session_manager and not is_browser_open(driver):
                logger.error(
                    "WebDriver session invalid after exception. Attempting restart..."
                )
                if session_manager.restart_sess():
                    logger.info("Session restarted. Retrying navigation...")
                    if not session_manager.driver:
                        return False
                    driver = session_manager.driver
                    continue
                else:
                    logger.error("Session restart failed.")
                    return False
            else:
                logger.warning(
                    "WebDriverException occurred, session seems valid/no restart. Waiting before retry."
                )
                time.sleep(random.uniform(2, 4))
                continue

        except Exception as e:
            logger.error(
                f"Unexpected error during navigation (Attempt {attempt}): {e}",
                exc_info=True,
            )
            time.sleep(random.uniform(2, 4))
            continue

    # --- End of Retry Loop ---
    logger.critical(
        f"Navigation to '{url}' failed permanently after {max_attempts} attempts."
    )
    try:
        logger.error(f"Final URL after failure: {driver.current_url}")
    except Exception:
        logger.error("Could not retrieve final URL after failure.")
    return False
# End of nav_to_page


def _pre_navigation_checks(
    driver: WebDriver, session_manager: Optional[SessionManager]
) -> bool:
    """Performs pre-navigation checks for login status and MFA."""
    try:
        current_url = driver.current_url
    except WebDriverException as e:
        logger.error(f"Failed to get current URL in pre-check: {e}.")
        return False  # Cannot proceed without knowing current state

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
                    time.sleep(1)  # Pause after login
                    post_login_url = driver.current_url
                    if post_login_url.startswith(
                        login_url_base
                    ) or post_login_url.startswith(mfa_url_base):
                        logger.error("Still on login/MFA page after re-login attempt.")
                        return False
                    return True  # Re-login appears successful
                except WebDriverException as post_login_e:
                    logger.error(f"Error getting URL after re-login: {post_login_e}")
                    return False
            else:
                logger.error(f"Re-login failed ({login_result}).")
                return False
        else:
            logger.error("On login page, but no SessionManager provided for re-login.")
            return False
    return True  # Passed pre-checks
# End ofnav_to_page


def _check_post_nav_redirects(post_nav_url: str) -> bool:
    """Checks if the URL after navigation is a login or MFA page."""
    login_url_base = urljoin(config_instance.BASE_URL, "account/signin")
    mfa_url_base = urljoin(config_instance.BASE_URL, "account/signin/mfa/")
    if post_nav_url.startswith(mfa_url_base):
        logger.warning("Redirected to MFA page immediately after navigation.")
        return True  # Indicates redirect occurred
    if post_nav_url.startswith(login_url_base):
        logger.warning("Redirected back to login page immediately after navigation.")
        return True  # Indicates redirect occurred
    return False
# end _check_post_nav_redirects


def _check_for_unavailability(
    driver: WebDriver, selectors: Dict[str, Tuple[str, int]]
) -> Tuple[Optional[str], int]:
    """Checks for unavailability messages on the current page."""
    for msg_selector, (action, wait_time) in selectors.items():
        if is_elem_there(
            driver, By.CSS_SELECTOR, msg_selector, wait=0.5
        ):  # Quick check
            logger.warning(
                f"Unavailability message found: '{msg_selector}' Action: {action}"
            )
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
        # V3 Change: Call start_sess (Phase 1) first
        start_ok = session_manager.start_sess(action_name="Utils Test - Phase 1")
        if not start_ok or not session_manager.driver_live:
            logger.error("SessionManager.start_sess() (Phase 1) FAILED. Aborting further tests.")
            test_success = False
            return # Use return for cleaner exit in tests
        else:
            logger.info("SessionManager.start_sess() (Phase 1) PASSED.")
            driver_instance = session_manager.driver # Get driver instance

            # V3 Change: Call ensure_session_ready (Phase 2)
            logger.info("--- Testing SessionManager.ensure_session_ready() ---")
            ready_ok = session_manager.ensure_session_ready(action_name="Utils Test - Phase 2")
            if not ready_ok or not session_manager.session_ready:
                logger.error("SessionManager.ensure_session_ready() (Phase 2) FAILED. Aborting further tests.")
                test_success = False
                return
            else:
                 logger.info("SessionManager.ensure_session_ready() (Phase 2) PASSED.")

            # --- Verify Identifiers ---
            logger.info("--- Verifying Identifiers (Retrieved during ensure_session_ready) ---")
            errors = []
            if not session_manager.my_profile_id:
                errors.append("my_profile_id")
            if not session_manager.my_uuid:
                errors.append("my_uuid")
            if config_instance.TREE_NAME and not session_manager.my_tree_id:
                errors.append("my_tree_id (required by config)")
            if not session_manager.csrf_token:
                errors.append("csrf_token")
            if errors:
                logger.error(
                    f"FAILED to retrieve required identifiers: {', '.join(errors)}"
                )
                test_success = False
            else:
                logger.info("All required identifiers retrieved successfully.")
                logger.debug(f" Profile ID: {session_manager.my_profile_id}")
                logger.debug(f" UUID: {session_manager.my_uuid}")
                logger.debug(f" Tree ID: {session_manager.my_tree_id or 'N/A'}")
                logger.debug(f" Tree Owner: {session_manager.tree_owner_name or 'N/A'}")
                if session_manager.csrf_token:
                    logger.debug(f" CSRF Token: {session_manager.csrf_token[:10]}...")
                else:
                    logger.debug(" CSRF Token: None")

            # --- Test Navigation ---
            logger.info("--- Testing Navigation (nav_to_page to BASE_URL) ---")
            nav_ok = nav_to_page(
                driver=driver_instance,
                url=config_instance.BASE_URL,
                selector="body",
                session_manager=session_manager,
            )
            if nav_ok:
                logger.info("nav_to_page() to BASE_URL PASSED.")
                try:
                    current_url_after_nav = driver_instance.current_url
                    if current_url_after_nav.startswith(
                        config_instance.BASE_URL.rstrip("/")
                    ):
                        logger.info(
                            f"Successfully landed on expected base URL: {current_url_after_nav}"
                        )
                    else:
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
                    if nav_to_page(
                        driver_instance,
                        "https://example.com",
                        selector="body",
                        session_manager=session_manager,
                    ):
                        logger.info("Navigation in new tab successful.")
                        logger.info("Closing extra tabs...")
                        close_tabs(driver_instance)  # Pass driver explicitly
                        handles_after_close = driver_instance.window_handles
                        if len(handles_after_close) == 1:
                            logger.info("close_tabs() PASSED (one tab remaining).")
                            # Ensure focus is back on the remaining tab
                            if (
                                driver_instance.current_window_handle
                                != handles_after_close[0]
                            ):
                                driver_instance.switch_to.window(handles_after_close[0])
                                logger.debug("Switched focus back to remaining tab.")
                        else:
                            logger.error(
                                f"close_tabs() FAILED (expected 1 tab, found {len(handles_after_close)})."
                            )
                            test_success = False
                    else:
                        logger.error("Navigation in new tab FAILED.")
                        test_success = False
                        # Attempt cleanup even if nav failed
                        close_tabs(driver_instance)

                except Exception as tab_e:
                    logger.error(
                        f"Error during tab management test: {tab_e}", exc_info=True
                    )
                    test_success = False
                    # Attempt cleanup on error
                    try:
                        close_tabs(driver_instance)
                    except Exception:
                        pass
            else:
                logger.error("make_tab() FAILED.")
                test_success = False

    except Exception as e:
        logger.critical(
            f"CRITICAL error during utils.py standalone test execution: {e}",
            exc_info=True,
        )
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
