#!/usr/bin/env python3

# utils.py

"""
utils.py - Core Session Management, API Requests, General Utilities

Manages Selenium/Requests sessions, handles core API interaction (_api_req),
provides general utilities (decorators, formatting, rate limiting),
and includes login/session verification logic closely tied to SessionManager.
"""

# --- Ensure core utility functions are always importable ---
import re
import logging
from typing import Optional

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
# NOTE: ordinal_case and format_name are defined above all other imports to ensure they are always available for import.

# --- Third-party and local imports ---
# These are placed in a try/except block to ensure that import errors do not prevent
# the core utility functions (format_name, ordinal_case) from being available for import.
try:
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
    from urllib.parse import urlparse, urlunparse  # Keep this

    # --- Local application imports ---
    from cache import cache as global_cache, cache_result
    from chromedriver import init_webdvr
    from config import config_instance, selenium_config
    from database import Base, ConversationLog, DnaMatch, FamilyTree, MessageType, Person
    from logging_config import logger, setup_logging
    from my_selectors import *
    # Import specific selenium utils needed internally
    from selenium_utils import is_browser_open, is_elem_there, close_tabs  # Added imports
except ImportError as e:
    logging.warning(f"Optional dependency import failed in utils.py: {e}. Some features may not be available.")

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
# Helper functions (General Utilities)
# ------------------------------------------------------------------------------------


def parse_cookie(cookie_string: str) -> Dict[str, str]:
    """
    Parses a raw HTTP cookie string into a dictionary of key-value pairs.
    """
    cookies: Dict[str, str] = {}
    parts = cookie_string.split(";")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key_value_pair = part.split("=", 1)
            if len(key_value_pair) == 2:
                key, value = key_value_pair
                cookies[key] = value
            else:
                logger.debug(f"Skipping invalid cookie part (split error): '{part}'")
        else:
            logger.debug(f"Skipping invalid cookie part (no '='): '{part}'")
    return cookies


# End of parse_cookie


def ordinal_case(text: str) -> str:
    """
    Corrects ordinal suffixes (1st, 2nd, 3rd, 4th) to lowercase within a string,
    often used after applying title casing. Handles relationship terms simply.
    """
    if not text:
        return text
    if isinstance(text, str) and not text.isdigit():  # Handle relationship terms
        words = text.title().split()
        lc_words = {"Of", "The", "A", "An", "In", "On", "At", "For", "To", "With"}
        for i, word in enumerate(words):
            if i > 0 and word in lc_words:
                words[i] = word.lower()
        return " ".join(words)
    # Handle numbers
    try:
        num = int(text)
        if 11 <= (num % 100) <= 13:
            suffix = "th"
        else:
            last_digit = num % 10
            if last_digit == 1:
                suffix = "st"
            elif last_digit == 2:
                suffix = "nd"
            elif last_digit == 3:
                suffix = "rd"
            else:
                suffix = "th"
        return str(num) + suffix
    except (ValueError, TypeError):
        return str(text)


# End of ordinal_case


def format_name(name: Optional[str]) -> str:
    """
    Formats a person's name string to title case, preserving uppercase components
    (like initials or acronyms) and handling None/empty input gracefully.
    Also removes GEDCOM-style slashes around surnames.
    """
    if not name or not isinstance(name, str):
        return "Valued Relative"
    try:
        # Remove GEDCOM slashes first
        cleaned_name = name.strip()
        cleaned_name = re.sub(r"\s*/([^/]+)/\s*$", r" \1", cleaned_name).strip()
        cleaned_name = re.sub(r"^/", "", cleaned_name).strip()
        cleaned_name = re.sub(r"/$", "", cleaned_name).strip()
        # Replace multiple spaces
        cleaned_name = re.sub(r"\s+", " ", cleaned_name)
        # Title case preserving uppercase parts
        parts = cleaned_name.split()
        formatted_parts = []
        for part in parts:
            if part.isupper():
                formatted_parts.append(part)
            else:
                formatted_parts.append(part.title())
        final_name = " ".join(formatted_parts)
        return final_name if final_name else "Valued Relative"
    except Exception as e:
        logger.error(f"Error formatting name '{name}': {e}", exc_info=False)
        return "Valued Relative"


# End of format_name

# ------------------------------
# Decorators (Remain in utils.py)
# ------------------------------


def retry(
    MAX_RETRIES: Optional[int] = None,
    BACKOFF_FACTOR: Optional[float] = None,
    MAX_DELAY: Optional[float] = None,
):
    """Decorator factory to retry a function with exponential backoff and jitter."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = MAX_RETRIES or (getattr(config_instance, "MAX_RETRIES", 3)) or 3
            backoff = (
                BACKOFF_FACTOR or (getattr(config_instance, "BACKOFF_FACTOR", 1)) or 1
            )
            max_delay = MAX_DELAY or (getattr(config_instance, "MAX_DELAY", 10)) or 10
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == attempts - 1:
                        raise
                    sleep_time = min(backoff * (2**i), max_delay) + random.uniform(
                        0, 0.5
                    )  # Added jitter
                    logger.warning(
                        f"Retry {i+1}/{attempts} for {func.__name__} after exception: {e}. Sleeping {sleep_time:.2f}s."
                    )
                    time.sleep(sleep_time)
            # Should not be reached if attempts > 0
            logger.error(
                f"Function '{func.__name__}' failed after all {attempts} retries (exited loop unexpectedly)."
            )
            return None  # Or raise custom error

        return wrapper

    return decorator


# End of retry


def retry_api(
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    retry_on_exceptions: Tuple[Type[Exception], ...] = (
        requests.exceptions.RequestException,
        ConnectionError,  # Added ConnectionError
    ),
    retry_on_status_codes: Optional[List[int]] = None,
):
    """Decorator factory for retrying API calls with exponential backoff, logging, etc."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _max_retries = (
                max_retries if max_retries is not None else config_instance.MAX_RETRIES
            )
            _initial_delay = (
                initial_delay
                if initial_delay is not None
                else config_instance.INITIAL_DELAY
            )
            _backoff_factor = (
                backoff_factor
                if backoff_factor is not None
                else config_instance.BACKOFF_FACTOR
            )
            _retry_codes_set = set(
                retry_on_status_codes
                if retry_on_status_codes is not None
                else config_instance.RETRY_STATUS_CODES
            )

            retries = _max_retries
            delay = _initial_delay
            attempt = 0

            while retries > 0:
                attempt += 1
                try:
                    response = func(*args, **kwargs)
                    status_code = getattr(response, "status_code", None)

                    if status_code is not None and status_code in _retry_codes_set:
                        should_retry_status = True
                    else:
                        should_retry_status = False

                    if should_retry_status:
                        retries -= 1
                        if retries <= 0:
                            logger.error(
                                f"API Call failed after {_max_retries} retries for '{func.__name__}' (Final Status {status_code})."
                            )
                            return response  # Return last response on final failure
                        sleep_time = min(
                            delay * (_backoff_factor ** (attempt - 1)),
                            config_instance.MAX_DELAY,
                        ) + random.uniform(0, 0.2)
                        sleep_time = max(0.1, sleep_time)
                        logger.warning(
                            f"API Call status {status_code} (Attempt {attempt}/{_max_retries}) for '{func.__name__}'. Retrying in {sleep_time:.2f}s..."
                        )
                        time.sleep(sleep_time)
                        delay *= _backoff_factor
                        continue
                    return response  # Success or non-retryable error code

                except retry_on_exceptions as e:
                    retries -= 1
                    if retries <= 0:
                        logger.error(
                            f"API Call failed after {_max_retries} retries for '{func.__name__}'. Final Exception: {type(e).__name__} - {e}",
                            exc_info=False,
                        )
                        raise e  # Re-raise the last exception
                    sleep_time = min(
                        delay * (_backoff_factor ** (attempt - 1)),
                        config_instance.MAX_DELAY,
                    ) + random.uniform(0, 0.2)
                    sleep_time = max(0.1, sleep_time)
                    logger.warning(
                        f"API Call exception (Attempt {attempt}/{_max_retries}) for '{func.__name__}', retrying in {sleep_time:.2f}s... Exception: {type(e).__name__} - {e}"
                    )
                    time.sleep(sleep_time)
                    delay *= _backoff_factor
                    continue
            logger.error(f"Exited retry loop unexpectedly for '{func.__name__}'.")
            return None  # Should not be reached normally

        return wrapper

    return decorator


# End of retry_api


def ensure_browser_open(func: Callable) -> Callable:
    """Decorator to ensure browser session is valid before executing."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        session_manager_instance: Optional[SessionManager] = None
        if args and isinstance(args[0], SessionManager):
            session_manager_instance = args[0]
        elif "session_manager" in kwargs and isinstance(
            kwargs["session_manager"], SessionManager
        ):
            session_manager_instance = kwargs["session_manager"]
        if not session_manager_instance:
            raise TypeError(
                f"Function '{func.__name__}' requires a SessionManager instance."
            )
        if not is_browser_open(session_manager_instance.driver):
            raise WebDriverException(
                f"Browser session invalid/closed when calling function '{func.__name__}'"
            )
        return func(*args, **kwargs)

    return wrapper


# End of ensure_browser_open


def time_wait(wait_description: str) -> Callable:
    """Decorator factory to time Selenium WebDriverWait calls."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(
                    f"Wait '{wait_description}' completed successfully in {duration:.3f}s."
                )
                return result
            except TimeoutException as e:
                duration = time.time() - start_time
                logger.warning(
                    f"Wait '{wait_description}' timed out after {duration:.3f} seconds.",
                    exc_info=False,
                )
                raise e
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Error during wait '{wait_description}' after {duration:.3f} seconds: {e}",
                    exc_info=True,
                )
                raise e

        return wrapper

    return decorator


# End of time_wait


# ------------------------------
# Rate Limiting (Remains in utils.py)
# ------------------------------
class DynamicRateLimiter:
    # ... (Implementation remains unchanged) ...
    def __init__(
        self,
        initial_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        backoff_factor: Optional[float] = None,
        decrease_factor: Optional[float] = None,
        token_capacity: Optional[float] = None,
        token_fill_rate: Optional[float] = None,
        config_instance=config_instance,
    ):
        self.initial_delay = (
            initial_delay
            if initial_delay is not None
            else config_instance.INITIAL_DELAY
        )
        self.MAX_DELAY = (
            max_delay if max_delay is not None else config_instance.MAX_DELAY
        )
        self.backoff_factor = (
            backoff_factor
            if backoff_factor is not None
            else config_instance.BACKOFF_FACTOR
        )
        self.decrease_factor = (
            decrease_factor
            if decrease_factor is not None
            else config_instance.DECREASE_FACTOR
        )
        self.current_delay = self.initial_delay
        self.last_throttled = False
        self.capacity = float(
            token_capacity
            if token_capacity is not None
            else config_instance.TOKEN_BUCKET_CAPACITY
        )
        self.fill_rate = float(
            token_fill_rate
            if token_fill_rate is not None
            else config_instance.TOKEN_BUCKET_FILL_RATE
        )
        if self.fill_rate <= 0:
            logger.warning(
                f"Token fill rate ({self.fill_rate}) must be positive. Setting to 1.0."
            )
            self.fill_rate = 1.0
        self.tokens = float(self.capacity)
        self.last_refill_time = time.monotonic()
        logger.debug(
            f"RateLimiter Init: Capacity={self.capacity:.1f}, FillRate={self.fill_rate:.1f}/s, InitialDelay={self.initial_delay:.2f}s, Backoff={self.backoff_factor:.2f}, Decrease={self.decrease_factor:.2f}"
        )

    # End of __init__
    def _refill_tokens(self):
        now = time.monotonic()
        elapsed = max(0, now - self.last_refill_time)
        tokens_to_add = elapsed * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = now

    # End of _refill_tokens
    def wait(self) -> float:
        self._refill_tokens()
        requested_at = time.monotonic()
        sleep_duration = 0.0
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            jitter_factor = random.uniform(0.8, 1.2)
            base_sleep = self.current_delay
            sleep_duration = min(base_sleep * jitter_factor, self.MAX_DELAY)
            sleep_duration = max(0.01, sleep_duration)
            logger.debug(
                f"Token available ({self.tokens:.2f} left). Applying base delay: {sleep_duration:.3f}s (CurrentDelay: {self.current_delay:.2f}s)"
            )
        else:
            wait_needed = (1.0 - self.tokens) / self.fill_rate
            jitter_amount = random.uniform(0.0, 0.2)
            sleep_duration = wait_needed + jitter_amount
            sleep_duration = min(sleep_duration, self.MAX_DELAY)
            sleep_duration = max(0.01, sleep_duration)
            logger.debug(
                f"Token bucket empty ({self.tokens:.2f}). Waiting for token: {sleep_duration:.3f}s"
            )
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        self._refill_tokens()
        if requested_at == self.last_refill_time:
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                logger.debug(
                    f"Consumed token after waiting. Tokens left: {self.tokens:.2f}"
                )
            else:
                logger.warning(
                    f"Waited for token, but still < 1 ({self.tokens:.2f}) after refill. Consuming fraction."
                )
                self.tokens = 0.0
        return sleep_duration

    # End of wait
    def reset_delay(self):
        if self.current_delay != self.initial_delay:
            self.current_delay = self.initial_delay
            logger.info(
                f"Rate limiter base delay reset to initial: {self.initial_delay:.2f}s"
            )
        self.last_throttled = False

    # End of reset_delay
    def decrease_delay(self):
        if not self.last_throttled and self.current_delay > self.initial_delay:
            previous_delay = self.current_delay
            self.current_delay = max(
                self.current_delay * self.decrease_factor, self.initial_delay
            )
            if abs(previous_delay - self.current_delay) > 0.01:
                logger.debug(
                    f"Decreased base delay component to {self.current_delay:.2f}s"
                )
        self.last_throttled = False

    # End of decrease_delay
    def increase_delay(self):
        previous_delay = self.current_delay
        self.current_delay = min(
            self.current_delay * self.backoff_factor, self.MAX_DELAY
        )
        logger.info(
            f"Rate limit feedback received. Increased base delay from {previous_delay:.2f}s to {self.current_delay:.2f}s"
        )
        self.last_throttled = True

    # End of increase_delay
    def is_throttled(self) -> bool:
        return self.last_throttled

    # End of is_throttled


# End of DynamicRateLimiter class


# ------------------------------
# Session Management (Remains in utils.py)
# ------------------------------
class SessionManager:
    # ... (Implementation remains unchanged, ensure it imports 'is_browser_open' from selenium_utils) ...
    def __init__(self):
        self.driver: Optional[WebDriver] = None
        self.driver_live: bool = False
        self.session_ready: bool = False
        self.db_path: str = str(config_instance.DATABASE_FILE.resolve())
        self.selenium_config = selenium_config
        self.ancestry_username: str = config_instance.ANCESTRY_USERNAME
        self.ancestry_password: str = config_instance.ANCESTRY_PASSWORD
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
        self.engine = None
        self.Session = None
        self._db_init_attempted = False
        self.cache_dir: Path = config_instance.CACHE_DIR
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
        self._requests_session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=Retry(
                total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
            ),
        )
        self._requests_session.mount("http://", adapter)
        self._requests_session.mount("https://", adapter)
        logger.debug("Initialized shared requests.Session with HTTPAdapter.")
        self.scraper: Optional[cloudscraper.CloudScraper] = None
        try:
            self.scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True},
                delay=10,
            )
            scraper_retry = Retry(
                total=3,
                backoff_factor=0.8,
                status_forcelist=[403, 429, 500, 502, 503, 504],
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
            self.scraper = None
        self.dynamic_rate_limiter: DynamicRateLimiter = DynamicRateLimiter()
        self.last_js_error_check: datetime = datetime.now()
        logger.debug(f"SessionManager instance created: ID={id(self)}\n")

    # End of __init__
    def start_sess(self, action_name: Optional[str] = None) -> bool:
        logger.debug(
            f"--- SessionManager Phase 1: Starting Driver ({action_name or 'Unknown Action'}) ---\n"
        )
        self.driver_live = False
        self.session_ready = False
        self.driver = None
        self.csrf_token = None
        self.my_profile_id = None
        self.my_uuid = None
        self.my_tree_id = None
        self.tree_owner_name = None
        self._reset_logged_flags()
        if not self.engine or not self.Session:
            try:
                self._initialize_db_engine_and_session()
            except Exception as db_init_e:
                logger.critical(
                    f"DB Initialization failed during Phase 1 start: {db_init_e}"
                )
                return False
        if not hasattr(self, "_requests_session") or not isinstance(
            self._requests_session, requests.Session
        ):
            self._requests_session = requests.Session()
            logger.debug(
                "Shared requests.Session initialized (fallback in start_sess)."
            )
        logger.debug("Initializing WebDriver instance (using init_webdvr)...")
        try:
            self.driver = init_webdvr()
            if not self.driver:
                logger.error(
                    "WebDriver initialization failed (init_webdvr returned None after retries)."
                )
                return False
            logger.debug("WebDriver initialization successful.")
            logger.debug(
                f"Navigating to Base URL ({config_instance.BASE_URL}) to stabilize..."
            )
            base_url_nav_ok = nav_to_page(
                self.driver,
                config_instance.BASE_URL,
                selector="body",
                session_manager=self,
            )
            if not base_url_nav_ok:
                logger.error("Failed to navigate to Base URL after WebDriver init.")
                self.close_sess()
                return False
            logger.debug("Initial navigation to Base URL successful.")
            self.driver_live = True
            self.session_start_time = time.time()
            self.last_js_error_check = datetime.now()
            logger.debug("--- SessionManager Phase 1: Driver Start Successful ---")
            return True
        except WebDriverException as wd_exc:
            logger.error(
                f"WebDriverException during Phase 1 start/base nav: {wd_exc}",
                exc_info=False,
            )
            self.close_sess()
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error during Phase 1 start/base nav: {e}", exc_info=True
            )
            self.close_sess()
            return False

    # End of start_sess
    def ensure_driver_live(
        self, action_name: Optional[str] = "Ensure Driver Live"
    ) -> bool:
        if self.driver_live:
            logger.debug(f"Driver already live (Action: {action_name}).")
            return True
        else:
            logger.debug(
                f"Driver not live, attempting start (Action: {action_name})..."
            )
            return self.start_sess(action_name=action_name)

    # End of ensure_driver_live

    def ensure_session_ready(self, action_name: Optional[str] = None) -> bool:
        """
        Ensures the session is ready for actions by checking driver state, performing readiness checks, and always fetching identifiers and tree owner name.
        """
        logger.debug("TRACE: Entered ensure_session_ready")

        if not self.ensure_driver_live(action_name=f"{action_name} - Ensure Driver"):
            logger.error(f"Cannot ensure session ready for '{action_name}': Driver start failed.")
            logger.debug("TRACE: ensure_session_ready - ensure_driver_live failed")

            return False

        ready = False # Default to False
        logger.debug("TRACE: About to call _perform_readiness_checks")

        try:
            logger.debug("TRACE: Calling _perform_readiness_checks")

            ready = self._perform_readiness_checks(action_name=f"{action_name} - Readiness Checks")
            logger.debug(f"TRACE: _perform_readiness_checks returned: {ready}")

        except Exception as e:
            logger.critical(f"Exception in _perform_readiness_checks: {e}", exc_info=True)

            return False # Or potentially raise, depending on desired behavior
        except BaseException as be:
            # Catching BaseException is generally discouraged unless cleaning up resources
            logger.critical(f"BaseException in _perform_readiness_checks: {type(be).__name__}: {be}", exc_info=True)

            raise # Re-raise BaseException as it might be KeyboardInterrupt etc.

        # --- Always retrieve identifiers and owner if driver is live, regardless of 'ready' status ---
        # This ensures these values are populated if the session is usable at all,
        # even if some readiness checks failed but didn't kill the driver.
        logger.debug("[Patch] Attempting to retrieve identifiers and tree owner after readiness checks...")
        self._retrieve_identifiers()
        self._retrieve_tree_owner()

        # Optionally, log the results after fetch
        logger.debug(f"[Patch] Identifiers after fetch: profile_id={self.my_profile_id}, uuid={self.my_uuid}, tree_id={self.my_tree_id}")
        logger.debug(f"[Patch] Tree owner name after fetch: {self.tree_owner_name}")

        # *** FIX: Update the instance state based on the readiness check outcome ***
        self.session_ready = ready
        logger.debug(f"TRACE: Set self.session_ready to: {self.session_ready}")

        logger.debug(f"TRACE: Exiting ensure_session_ready (returning {ready})")
        # print(f"TRACE: Exiting ensure_session_ready (returning {ready})")
        return ready
    # End of ensure_session_ready

    def _perform_readiness_checks(self, action_name: Optional[str] = None) -> bool:
        """
        Perform a sequence of checks to ensure the session is ready.

        :param action_name: An optional name for the action that initiated the checks.
        :return: True if all checks passed, False otherwise.
        """
        max_attempts = 2
        attempt = 0
        while attempt < max_attempts:
            try:
                logger.debug(f"Performing readiness checks (action={action_name})...")

                # Initial checks
                if not self.driver_live or not self.driver:
                    logger.error("Cannot perform readiness checks: Driver not live.")
                    return False

                # Check cookie persistence setup
                cookies_backup_path = os.path.join(
                    str(self.chrome_user_data_dir), "ancestry_cookies.json"
                )
                if not os.path.exists(self.chrome_user_data_dir) or not os.listdir(
                    self.chrome_user_data_dir
                ):
                    logger.warning(
                        f"[Persistence] Chrome user data directory '{self.chrome_user_data_dir}' does not exist or is empty!\n"
                        "If you want to avoid 2FA, never delete or reset this directory after a successful login.\n"
                        "If this is your first run, complete 2FA manually and keep the browser/profile for future runs."
                    )
                else:
                    logger.info(
                        f"[Persistence] Chrome user data directory is present: {self.chrome_user_data_dir}"
                    )

                # Check login status
                login_status_result = login_status(self)
                if login_status_result is False:
                    logger.info("Not logged in. Attempting remedial actions...")
                    # Try cookie import first if backup exists
                    if os.path.exists(cookies_backup_path):
                        logger.info(f"Cookie backup found. Attempting import...")
                        try:
                            # Go to base URL before cookie import for stability
                            self.driver.get(config_instance.BASE_URL)
                            import_cookies(self.driver, cookies_backup_path)
                            self.driver.refresh()
                            time.sleep(1)  # Allow refresh to settle
                            logger.info(
                                "Imported cookies from backup and refreshed page."
                            )
                            # Re-check status after import
                            login_status_after_import = login_status(self)
                            if login_status_after_import is True:
                                logger.info(
                                    "Login restored via cookie import. Continuing checks..."
                                )
                                login_status_result = True
                            else:
                                logger.warning(
                                    "Cookie import attempted, but login status still False."
                                )
                        except Exception as import_err:
                            logger.warning(f"Cookie import failed: {import_err}")

                    # If still not logged in, attempt automated login
                    if login_status_result is False:
                        logger.info("Attempting login via automation...")
                        login_result = log_in(self)
                        if login_result != "LOGIN_SUCCEEDED":
                            logger.error(
                                f"Login attempt failed ({login_result}). Readiness check failed on attempt {attempt}."
                            )
                            if attempt < max_attempts:
                                continue
                            else:
                                return False
                        # Login succeeded, export cookies and verify status again
                        logger.info("Login successful via automation.")
                        try:
                            logger.debug(
                                f"Exporting cookies after successful login..."
                            )
                            export_cookies(self.driver, cookies_backup_path)
                            logger.info(
                                f"Cookies exported to {cookies_backup_path}"
                            )
                        except Exception as export_err:
                            logger.warning(
                                f"Failed to export cookies after login: {export_err}"
                            )

                        # Final verification after automated login
                        login_status_result = login_status(self)
                        if login_status_result is not True:
                            logger.error(
                                "Login status verification failed even after successful login report."
                            )
                            if attempt < max_attempts:
                                continue
                            else:
                                return False
                        logger.debug(
                            "Login status re-verified successfully after automation."
                        )

                # Check URL and handle if logged in
                if login_status_result is True:
                    logger.debug("Checking/Handling current URL...")
                    if not self._check_and_handle_url():
                        logger.error("URL check/handling failed.")
                        return False
                    logger.debug("URL check/handling OK.")

                # Verify essential cookies
                logger.debug("Verifying essential cookies...")
                essential_cookies = ["ANCSESSIONID", "SecureATT"]
                if not self.get_cookies(
                    essential_cookies, timeout=5
                ):  # Shorter timeout here
                    logger.error(
                        f"Essential cookies {essential_cookies} not found after checks."
                    )
                    return False
                logger.debug("Essential cookies OK.")

                # Sync cookies to requests session
                logger.debug("Syncing cookies to requests session...")
                self._sync_cookies()  # Assuming this logs errors internally if needed
                logger.debug("Cookies synced.")

                # Ensure CSRF token
                logger.debug("Ensuring CSRF token...")
                if not self.csrf_token or len(self.csrf_token) < 20:
                    self.csrf_token = (
                        self.get_csrf()
                    )  # Fetch if missing/invalid
                if not self.csrf_token:
                    logger.error("Failed to retrieve/verify CSRF token.")
                    return False
                logger.debug("CSRF token OK.")

                # If all checks passed for this attempt
                logger.info(f"Readiness checks PASSED on attempt {attempt+1}.")
                return True

            except WebDriverException as wd_exc:
                logger.error(
                    f"WebDriverException during readiness check attempt {attempt+1}: {wd_exc}",
                    exc_info=False,  # Keep concise for loop
                )
                if not self.is_sess_valid():
                    logger.error(
                        "Session invalid during readiness check. Aborting."
                    )
                    self.driver_live = False  # Update state
                    self.session_ready = False
                    # Don't call close_sess here, let the caller handle full cleanup
                    return False
                # If session still valid but error occurred, log and wait before retry
                if attempt >= max_attempts:
                    logger.error(
                        "Readiness checks failed after final attempt (WebDriverException)."
                    )
                    return False
                logger.info(
                    f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next readiness attempt..."
                )
                time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                attempt += 1
                continue  # Go to next attempt

            except Exception as e:
                logger.error(
                    f"Unexpected error during readiness check attempt {attempt}: {e}",
                    exc_info=True,  # Show full trace for unexpected errors
                )
                if attempt >= max_attempts:
                    logger.error(
                        "Readiness checks failed after final attempt (Exception)."
                    )
                    return False
                logger.info(
                    f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next readiness attempt..."
                )
                time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
                attempt += 1
                continue  # Go to next attempt

        # Loop finished without returning True
        logger.error(
            f"All {max_attempts} readiness check attempts failed."
        )
        return False
    # End of _perform_readiness_checks

    def _initialize_db_engine_and_session(self):
        if self.engine and self.Session:
            logger.debug(
                f"DB Engine/Session already initialized for SM ID={id(self)}. Skipping."
            )
            return
        logger.debug(
            f"SessionManager ID={id(self)} initializing SQLAlchemy Engine/Session..."
        )
        self._db_init_attempted = True
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
            logger.debug(f"DB Path: {self.db_path}")
            pool_size_env_str = os.getenv("DB_POOL_SIZE")
            pool_size_config = getattr(config_instance, "DB_POOL_SIZE", None)
            pool_size_str_to_parse = None
            if pool_size_env_str is not None:
                pool_size_str_to_parse = pool_size_env_str
            elif pool_size_config is not None:
                try:
                    pool_size_str_to_parse = str(int(pool_size_config))
                except (ValueError, TypeError):
                    pool_size_str_to_parse = "50"
            else:
                pool_size_str_to_parse = "50"
            pool_size = 20
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
                    pool_size = min(parsed_val, 100)
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
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=False,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                poolclass=pool_class,
                connect_args={"check_same_thread": False},
            )
            logger.debug(
                f"Created NEW SQLAlchemy engine: ID={id(self.engine)} for SM ID={id(self)}"
            )

            @event.listens_for(self.engine, "connect")
            def enable_sqlite_settings(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA journal_mode=WAL;")
                    cursor.execute("PRAGMA foreign_keys=ON;")
                    logger.debug("SQLite PRAGMA settings applied (WAL, Foreign Keys).")
                except Exception as pragma_e:
                    logger.error(f"Failed setting PRAGMA: {pragma_e}")
                finally:
                    cursor.close()

            self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
            logger.debug(f"Created Session factory for Engine ID={id(self.engine)}")
            try:
                Base.metadata.create_all(self.engine)
                logger.debug("DB tables checked/created successfully.")
            except Exception as table_create_e:
                logger.error(
                    f"Error creating DB tables: {table_create_e}", exc_info=True
                )
                raise table_create_e
        except Exception as e:
            logger.critical(f"FAILED to initialize SQLAlchemy: {e}", exc_info=True)
            if self.engine:
                try:
                    self.engine.dispose()
                except Exception:
                    pass
            self.engine = None
            self.Session = None
            self._db_init_attempted = False
            raise e

    # End of _initialize_db_engine_and_session
    def _check_and_handle_url(self) -> bool:
        try:
            current_url = self.driver.current_url
            logger.debug(f"Current URL for check: {current_url}")
        except WebDriverException as e:
            logger.error(f"Error getting current URL: {e}. Session might be dead.")
            if not self.is_sess_valid():
                logger.warning("Session seems invalid during URL check.")
            return False
        except AttributeError:
            logger.error("Driver attribute is None. Cannot check URL.")
            return False
        base_url_norm = config_instance.BASE_URL.rstrip("/") + "/"
        signin_url_base = urljoin(base_url_norm, "account/signin").rstrip("/")
        logout_url_base = urljoin(base_url_norm, "c/logout").rstrip("/")
        mfa_url_base = urljoin(base_url_norm, "account/signin/mfa/").rstrip("/")
        disallowed_starts = (signin_url_base, logout_url_base, mfa_url_base)
        is_api_path = "/api/" in current_url
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
        return True

    # End of _check_and_handle_url
    def _retrieve_identifiers(self) -> bool:
        logger.debug("TRACE: Entered _retrieve_identifiers")

        all_ok = True
        if not self.my_profile_id:
            logger.debug("Retrieving profile ID (ucdmid)...")
            self.my_profile_id = self.get_my_profileId()
            if not self.my_profile_id:
                logger.error("Failed to retrieve profile ID (ucdmid).")
                all_ok = False
            elif not self._profile_id_logged:
                logger.info(f"My profile id: {self.my_profile_id}")
                self._profile_id_logged = True
        elif not self._profile_id_logged:
            logger.info(f"My profile id: {self.my_profile_id}")
            self._profile_id_logged = True
        if not self.my_uuid:
            logger.debug("Retrieving UUID (testId)...")
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
        if config_instance.TREE_NAME and not self.my_tree_id:
            logger.debug(
                f"Retrieving tree ID for tree name: '{config_instance.TREE_NAME}'..."
            )
            self.my_tree_id = self.get_my_tree_id()
            if not self.my_tree_id:
                logger.error(
                    f"TREE_NAME '{config_instance.TREE_NAME}' configured, but failed to get corresponding tree ID."
                )
                all_ok = False
            elif not self._tree_id_logged:
                logger.info(f"My tree id: {self.my_tree_id}")
                self._tree_id_logged = True
        elif self.my_tree_id and not self._tree_id_logged:
            logger.info(f"My tree id: {self.my_tree_id}")
            self._tree_id_logged = True
        elif not config_instance.TREE_NAME:
            logger.debug("No TREE_NAME configured, skipping tree ID retrieval.")
        logger.debug("TRACE: Exiting _retrieve_identifiers")

        return all_ok

    # End of _retrieve_identifiers
    def _retrieve_tree_owner(self) -> bool:
        logger.debug("TRACE: Entered _retrieve_tree_owner")

        if self.tree_owner_name:
            if not self._owner_logged:
                logger.info(f"Tree owner name: {self.tree_owner_name}")
                self._owner_logged = True
            logger.debug("TRACE: Exiting _retrieve_tree_owner (already set)")

            return True
        logger.debug("Retrieving tree owner name...")
        if not self.my_tree_id:
            logger.error("Cannot retrieve tree owner name: my_tree_id is not set.")
            logger.debug("TRACE: Exiting _retrieve_tree_owner (no tree id)")

            return False
        self.tree_owner_name = self.get_tree_owner(self.my_tree_id)
        if not self.tree_owner_name:
            logger.error("Failed to retrieve tree owner name.")
            logger.debug("TRACE: Exiting _retrieve_tree_owner (failed)")

            return False
        logger.info(f"Tree owner name: {self.tree_owner_name}")
        self._owner_logged = True
        logger.debug("TRACE: Exiting _retrieve_tree_owner (success)")

        return True

    # End of _retrieve_tree_owner
    @retry_api()
    def get_csrf(self) -> Optional[str]:
        csrf_token_url = urljoin(
            config_instance.BASE_URL, "discoveryui-matches/parents/api/csrfToken"
        )
        logger.debug(f"Attempting to fetch fresh CSRF token from: {csrf_token_url}")
        essential_cookies = ["ANCSESSIONID", "SecureATT"]
        if not self.get_cookies(essential_cookies, timeout=10):
            logger.warning(
                f"Essential cookies {essential_cookies} NOT found before CSRF token API call. API might fail."
            )
        response_data = _api_req(
            url=csrf_token_url,
            driver=self.driver,
            session_manager=self,
            method="GET",
            use_csrf_token=False,
            api_description="CSRF Token API",
            force_text_response=True,
        )
        if response_data and isinstance(response_data, str):
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
            logger.warning(
                "Failed to get CSRF token response via _api_req (returned None)."
            )
            return None
        else:
            logger.error(
                f"Unexpected response type for CSRF token API: {type(response_data)}"
            )
            logger.debug(f"Response data received: {response_data}")
            return None

    # End of get_csrf
    def get_cookies(self, cookie_names: List[str], timeout: int = 30) -> bool:
        start_time = time.time()
        logger.debug(f"Waiting up to {timeout}s for cookies: {cookie_names}...")
        required_lower = {name.lower() for name in cookie_names}
        interval = 0.5
        last_missing_str = ""
        while time.time() - start_time < timeout:
            try:
                if not self.is_sess_valid():
                    logger.warning("Session became invalid while waiting for cookies.")
                    return False
                cookies = self.driver.get_cookies()
                current_cookies_lower = {
                    c["name"].lower() for c in cookies if "name" in c
                }
                missing_lower = required_lower - current_cookies_lower
                if not missing_lower:
                    logger.debug(f"All required cookies found: {cookie_names}.")
                    return True
                missing_str = ", ".join(sorted(missing_lower))
                if missing_str != last_missing_str:
                    logger.debug(f"Still missing cookies: {missing_str}")
                    last_missing_str = missing_str
                time.sleep(interval)
            except WebDriverException as e:
                logger.error(f"WebDriverException while retrieving cookies: {e}")
                if not self.is_sess_valid():
                    logger.error(
                        "Session invalid after WebDriverException during cookie retrieval."
                    )
                    return False
                time.sleep(interval * 2)
            except Exception as e:
                logger.error(f"Unexpected error retrieving cookies: {e}", exc_info=True)
                time.sleep(interval * 2)
        missing_final = []
        try:
            if self.driver and self.is_sess_valid():
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
                missing_final = cookie_names
        except Exception:
            missing_final = cookie_names
        logger.warning(f"Timeout waiting for cookies. Missing: {missing_final}.")
        return False

    # End of get_cookies
    def _sync_cookies(self):
        if not self.is_sess_valid():
            logger.warning("Cannot sync cookies: WebDriver session invalid.")
            return
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
        if not hasattr(self, "_requests_session") or not isinstance(
            self._requests_session, requests.Session
        ):
            logger.error("Cannot sync cookies: requests.Session not initialized.")
            return
        requests_cookie_jar = self._requests_session.cookies
        requests_cookie_jar.clear()
        synced_count = 0
        skipped_count = 0
        for cookie in driver_cookies:
            if (
                not isinstance(cookie, dict)
                or "name" not in cookie
                or "value" not in cookie
                or "domain" not in cookie
            ):
                logger.warning(f"Skipping invalid cookie format during sync: {cookie}")
                skipped_count += 1
                continue
            try:
                domain_to_set = cookie["domain"]
                cookie_attrs = {
                    "name": cookie["name"],
                    "value": cookie["value"],
                    "domain": domain_to_set,
                    "path": cookie.get("path", "/"),
                    "secure": cookie.get("secure", False),
                    "rest": {"httpOnly": cookie.get("httpOnly", False)},
                }
                if "expiry" in cookie and cookie["expiry"] is not None:
                    if isinstance(cookie["expiry"], (int, float)):
                        cookie_attrs["expires"] = int(cookie["expiry"])
                    else:
                        logger.warning(
                            f"Unexpected expiry format for cookie {cookie['name']}: {cookie['expiry']}"
                        )
                requests_cookie_jar.set(**cookie_attrs)
                synced_count += 1
            except Exception as set_err:
                logger.warning(
                    f"Failed to set cookie '{cookie.get('name', '??')}' in requests session: {set_err}"
                )
                skipped_count += 1
        if skipped_count > 0:
            logger.warning(
                f"Skipped {skipped_count} cookies during sync due to format/errors."
            )
        logger.debug(f"Successfully synced {synced_count} cookies to requests session.")

    # End of _sync_cookies
    def return_session(self, session: Session):
        if session:
            session_id = id(session)
            try:
                session.close()
            except Exception as e:
                logger.error(
                    f"Error closing DB session {session_id}: {e}", exc_info=True
                )
        else:
            logger.warning("Attempted to return a None DB session.")

    # End of return_session
    def _reset_logged_flags(self):
        self._profile_id_logged = False
        self._uuid_logged = False
        self._tree_id_logged = False
        self._owner_logged = False

    def get_db_conn(self) -> Optional[Session]:
        engine_id_str = id(self.engine) if self.engine else "None"
        logger.debug(
            f"SessionManager ID={id(self)} get_db_conn called. Current Engine ID: {engine_id_str}"
        )
        if not self._db_init_attempted or not self.engine or not self.Session:
            logger.debug(
                f"SessionManager ID={id(self)}: Engine/Session factory not ready. Triggering initialization..."
            )
            try:
                self._initialize_db_engine_and_session()
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
        try:
            new_session = self.Session()
            logger.debug(
                f"SessionManager ID={id(self)} obtained DB session {id(new_session)} from Engine ID={id(self.engine)}"
            )
            return new_session
        except Exception as e:
            logger.error(
                f"SessionManager ID={id(self)} Error getting DB session from factory: {e}",
                exc_info=True,
            )
            if self.engine:
                try:
                    self.engine.dispose()
                except Exception:
                    pass
            self.engine = None
            self.Session = None
            self._db_init_attempted = False
            return None

    # End of get_db_conn
    @contextlib.contextmanager
    def get_db_conn_context(self) -> Generator[Optional[Session], None, None]:
        session: Optional[Session] = None
        session_id_for_log = "N/A"
        try:
            session = self.get_db_conn()
            if session:
                session_id_for_log = id(session)
                logger.debug(
                    f"DB Context Manager: Acquired session {session_id_for_log}."
                )
                yield session
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
                logger.error("DB Context Manager: Failed to obtain DB session.")
                yield None
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
            raise e
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
            self.Session = None
            self._db_init_attempted = False
            return
        engine_id = id(self.engine)
        logger.debug(
            f"SessionManager ID={id(self)} cls_db_conn called (keep_db=False). Disposing Engine ID: {engine_id}"
        )
        try:
            self.engine.dispose()
            logger.debug(f"Engine ID={engine_id} disposed successfully.")
        except Exception as e:
            logger.error(
                f"Error disposing SQLAlchemy engine ID={engine_id}: {e}", exc_info=True
            )
        finally:
            self.engine = None
            self.Session = None
            self._db_init_attempted = False

    # End of cls_db_conn
    @retry_api()
    def get_my_profileId(self) -> Optional[str]:
        url = urljoin(
            config_instance.BASE_URL,
            "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid",
        )
        logger.debug("Attempting to fetch own profile ID (ucdmid)...")
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
                logger.warning(
                    "Failed to get profile_id response via _api_req (returned None)."
                )
                return None
            if (
                isinstance(response_data, dict)
                and "data" in response_data
                and "ucdmid" in response_data["data"]
            ):
                my_profile_id_val = str(response_data["data"]["ucdmid"]).upper()
                logger.debug(f"Successfully retrieved profile_id: {my_profile_id_val}")
                return my_profile_id_val
            else:
                logger.error("Could not find 'data.ucdmid' in profile_id API response.")
                logger.debug(f"Full profile_id response data: {response_data}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error in get_my_profileId: {e}", exc_info=True)
            return None

    # End of get_my_profileId
    @retry_api()
    def get_my_uuid(self) -> Optional[str]:
        if not self.is_sess_valid():
            logger.error("get_my_uuid: Session invalid.")
            return None
        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        logger.debug("Attempting to fetch own UUID (testId) from header/dna API...")
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
                my_uuid_val = str(response_data["testId"]).upper()
                logger.debug(f"Successfully retrieved UUID: {my_uuid_val}")
                return my_uuid_val
            else:
                logger.error("Could not retrieve UUID ('testId' missing in response).")
                logger.debug(f"Full get_my_uuid response data: {response_data}")
                return None
        else:
            logger.error(
                "Failed to get header/dna data via _api_req (returned None or error response)."
            )
            return None

    # End of get_my_uuid
    @retry_api()
    def get_my_tree_id(self) -> Optional[str]:
        tree_name_config = config_instance.TREE_NAME
        if not tree_name_config:
            logger.debug("TREE_NAME not configured, skipping tree ID retrieval.")
            return None
        if not self.is_sess_valid():
            logger.error("get_my_tree_id: Session invalid.")
            return None
        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/trees")
        logger.debug(
            f"Attempting to fetch tree ID for TREE_NAME='{tree_name_config}'..."
        )
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
                        break
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
    @retry_api()
    def get_tree_owner(self, tree_id: str) -> Optional[str]:
        if not tree_id:
            logger.warning("Cannot get tree owner: tree_id is missing.")
            return None
        if not self.is_sess_valid():
            logger.error("get_tree_owner: Session invalid.")
            return None
        url = urljoin(
            config_instance.BASE_URL,
            f"api/uhome/secure/rest/user/tree-info?tree_id={tree_id}",
        )
        logger.debug(f"Attempting to fetch tree owner name for tree ID: {tree_id}...")
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
                logger.debug(f"Full Tree Owner API response data: {response_data}")
                return None
            else:
                logger.warning(
                    "Tree Owner API call via _api_req returned unexpected data or None."
                )
                logger.debug(f"Response received: {response_data}")
                return None
        except Exception as e:
            logger.error(
                f"Error fetching/parsing Tree Owner API for tree {tree_id}: {e}",
                exc_info=True,
            )
            return None

    # End of get_tree_owner
    def verify_sess(self) -> bool:
        logger.debug("Verifying session status (using login_status)...")
        login_ok = login_status(self)
        if login_ok is True:
            logger.debug("Session verification successful (logged in).")
            return True
        elif login_ok is False:
            logger.warning("Session verification failed (user not logged in).")
            return False
        else:
            logger.error(
                "Session verification failed critically (login_status returned None)."
            )
            return False

    # End of verify_sess
    def _verify_api_login_status(self) -> Optional[bool]:
        api_url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        api_description = "API Login Verification (header/dna)"
        logger.debug(f"Verifying login status via API endpoint: {api_url}...")
        if not self.driver or not self.is_sess_valid():
            logger.warning(
                f"{api_description}: Driver/session not valid for API check."
            )
            return None
        try:
            logger.debug("Syncing cookies before API login check...")
            self._sync_cookies()
        except Exception as sync_e:
            logger.warning(f"Error syncing cookies before API login check: {sync_e}")
        try:
            response_data = _api_req(
                url=api_url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description=api_description,
            )
            if response_data is None:
                logger.warning(
                    f"{api_description}: _api_req returned None. Returning None."
                )
                return None
            elif isinstance(response_data, requests.Response):
                status_code = response_data.status_code
                if status_code in [401, 403]:
                    logger.debug(
                        f"{api_description}: API check failed with status {status_code}. User NOT logged in."
                    )
                    return False
                else:
                    logger.warning(
                        f"{api_description}: API check failed with unexpected status {status_code}. Returning None."
                    )
                    return None
            elif isinstance(response_data, dict):
                if "testId" in response_data:
                    logger.debug(
                        f"{api_description}: API login check successful ('testId' found)."
                    )
                    return True
                else:
                    logger.warning(
                        f"{api_description}: API check succeeded (2xx), but response format unexpected: {response_data}. Assuming logged in cautiously."
                    )
                    return True
            else:
                logger.error(
                    f"{api_description}: _api_req returned unexpected type {type(response_data)}. Returning None."
                )
                return None
        except Exception as e:
            logger.error(
                f"Unexpected error during {api_description}: {e}", exc_info=True
            )
            return None

    # End of _verify_api_login_status
    @retry_api()
    def get_header(self) -> bool:
        if not self.is_sess_valid():
            logger.error("get_header: Session invalid.")
            return False
        url = urljoin(config_instance.BASE_URL, "api/uhome/secure/rest/header/dna")
        logger.debug("Attempting to fetch header/dna API data...")
        response_data = _api_req(
            url,
            self.driver,
            self,
            method="GET",
            use_csrf_token=False,
            api_description="Get UUID API",
        )
        if response_data:
            if isinstance(response_data, dict) and "testId" in response_data:
                logger.debug("Header data retrieved successfully ('testId' found).")
                return True
            else:
                logger.error("Unexpected response structure from header/dna API.")
                logger.debug(f"Response: {response_data}")
                return False
        else:
            logger.error(
                "Failed to get header/dna data via _api_req (returned None or error response)."
            )
            return False

    # End of get_header
    def _validate_sess_cookies(self, required_cookies: List[str]) -> bool:
        if not self.is_sess_valid():
            logger.warning("Cannot validate cookies: Session invalid.")
            return False
        try:
            cookies = {
                c["name"]: c["value"] for c in self.driver.get_cookies() if "name" in c
            }
            missing_cookies = [name for name in required_cookies if name not in cookies]
            if not missing_cookies:
                return True
            else:
                logger.debug(f"Cookie validation failed. Missing: {missing_cookies}")
                return False
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
    def is_sess_logged_in(self) -> bool:
        logger.warning("is_sess_logged_in is deprecated. Use login_status() instead.")
        return login_status(self) is True

    # End of is_sess_logged_in
    def is_sess_valid(self) -> bool:
        if not self.driver:
            return False
        try:
            _ = self.driver.window_handles
            return True
        except InvalidSessionIdException:
            logger.debug(
                "Session ID is invalid (browser likely closed or session terminated)."
            )
            return False
        except (NoSuchWindowException, WebDriverException) as e:
            err_str = str(e).lower()
            if any(
                sub in err_str
                for sub in [
                    "disconnected",
                    "target crashed",
                    "no such window",
                    "unable to connect",
                ]
            ):
                logger.warning(f"Session seems invalid due to WebDriverException: {e}")
                return False
            else:
                logger.warning(
                    f"Unexpected WebDriverException checking session validity: {e}"
                )
                return False
        except Exception as e:
            logger.error(
                f"Unexpected error checking session validity: {e}", exc_info=True
            )
            return False

    # End of is_sess_valid
    def close_sess(self, keep_db: bool = False):
        if self.driver:
            logger.debug("Attempting to close WebDriver session...")
            try:
                self.driver.quit()
                logger.debug("WebDriver session quit successfully.")
            except Exception as e:
                logger.error(f"Error closing WebDriver session: {e}", exc_info=True)
            finally:
                self.driver = None
        else:
            logger.debug("No active WebDriver session to close.")
        self.driver_live = False
        self.session_ready = False
        self.csrf_token = None
        if not keep_db:
            logger.debug("Closing database connection pool...")
            self.cls_db_conn(keep_db=False)
        else:
            logger.debug("Keeping DB connection pool alive (keep_db=True).")

    # End of close_sess
    def restart_sess(self, url: Optional[str] = None) -> bool:
        logger.warning("Restarting WebDriver session...")
        self.close_sess(keep_db=True)
        start_ok = self.start_sess(action_name="Session Restart - Phase 1")
        if not start_ok:
            logger.error("Failed to restart session (Phase 1: Driver Start failed).")
            return False
        ready_ok = self.ensure_session_ready(action_name="Session Restart - Phase 2")
        if not ready_ok:
            logger.error("Failed to restart session (Phase 2: Session Ready failed).")
            self.close_sess(keep_db=True)
            return False
        if url and self.driver:
            logger.info(f"Session restart successful. Re-navigating to: {url}")
            if nav_to_page(self.driver, url, selector="body", session_manager=self):
                logger.info(f"Successfully re-navigated to {url}.")
                return True
            else:
                logger.error(
                    f"Failed to re-navigate to {url} after successful restart."
                )
                return False
        elif not url:
            logger.info("Session restart successful (no navigation requested).")
            return True
        else:
            logger.error(
                "Driver instance missing after successful session restart report."
            )
            return False

    # End of restart_sess
    @ensure_browser_open
    def make_tab(self) -> Optional[str]:
        driver = self.driver
        if driver is None:
            logger.error("Driver is None in make_tab despite decorator.")
            return None
        try:
            tab_list_before = driver.window_handles
            logger.debug(f"Window handles before new tab: {tab_list_before}")
        except WebDriverException as e:
            logger.error(f"Error getting window handles before new tab: {e}")
            return None
        try:
            driver.switch_to.new_window("tab")
            logger.debug("Executed new_window('tab') command.")
        except WebDriverException as e:
            logger.error(f"Error executing new_window('tab'): {e}")
            return None
        try:
            WebDriverWait(driver, selenium_config.NEW_TAB_TIMEOUT).until(
                lambda d: len(d.window_handles) > len(tab_list_before)
            )
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
        except TimeoutException:
            logger.error("Timeout waiting for new tab handle to appear.")
            try:
                logger.debug(f"Window handles during timeout: {driver.window_handles}")
            except Exception:
                pass
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
        if not self.is_sess_valid():
            return
        if self.driver is None:
            logger.warning("Driver is None. Skipping JS error check.")
            return
        try:
            log_types = self.driver.log_types
            if "browser" not in log_types:
                return
        except WebDriverException as e:
            logger.warning(f"Could not get log_types: {e}. Skipping JS error check.")
            return
        try:
            logs = self.driver.get_log("browser")
        except WebDriverException as e:
            logger.warning(f"WebDriverException getting browser logs: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error getting browser logs: {e}", exc_info=True)
            return
        new_errors_found = False
        most_recent_error_time_this_check = self.last_js_error_check
        for entry in logs:
            if isinstance(entry, dict) and entry.get("level") == "SEVERE":
                try:
                    timestamp_ms = entry.get("timestamp")
                    if timestamp_ms:
                        timestamp_dt = datetime.fromtimestamp(
                            timestamp_ms / 1000.0, tz=timezone.utc
                        )
                        if timestamp_dt > self.last_js_error_check:
                            new_errors_found = True
                            error_message = entry.get("message", "No message")
                            source_match = re.search(r"(.+?):(\d+)", error_message)
                            source_info = (
                                f" (Source: {source_match.group(1).split('/')[-1]}:{source_match.group(2)})"
                                if source_match
                                else ""
                            )
                            logger.warning(
                                f"JS ERROR DETECTED:{source_info} {error_message}"
                            )
                            if timestamp_dt > most_recent_error_time_this_check:
                                most_recent_error_time_this_check = timestamp_dt
                    else:
                        logger.warning(f"JS Log entry missing timestamp: {entry}")
                except Exception as parse_e:
                    logger.warning(f"Error parsing JS log entry {entry}: {parse_e}")
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


def _api_req(
    url: str,
    driver: Optional["WebDriver"],
    session_manager: "SessionManager",  # Forward reference okay here
    method: str = "GET",
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    use_csrf_token: bool = True,
    headers: Optional[Dict] = None,
    referer_url: Optional[str] = None,
    api_description: str = "API Call",
    timeout: Optional[int] = None,
    cookie_jar: Optional["RequestsCookieJar"] = None,
    allow_redirects: bool = True,
    force_text_response: bool = False,
    add_default_origin: bool = True,
) -> Optional[Any]:
    """
    V1.9 REVISED: Makes HTTP request using shared requests.Session.
    Corrected calls to internal helper functions (removed 'utils.' prefix).
    Includes dynamic header generation, cookie sync, rate limiting, retries,
    response processing, debug logging, conditional 'ancestry-userid'.

    Args:
        # ... (Args remain the same) ...

    Returns:
        # ... (Returns remain the same) ...
    """
    # Step 1: Validate prerequisites
    # Use global config_instance and selenium_config assumed to be imported in utils.py
    global config_instance, selenium_config
    if not session_manager or not session_manager._requests_session:
        logger.error(
            f"{api_description}: Aborting - SessionManager or requests_session missing."
        )
        return None

    # Step 2: Get Retry Configuration
    max_retries = config_instance.MAX_RETRIES
    initial_delay = config_instance.INITIAL_DELAY
    backoff_factor = config_instance.BACKOFF_FACTOR
    max_delay = config_instance.MAX_DELAY
    retry_status_codes = set(config_instance.RETRY_STATUS_CODES)

    # --- Step 3: Prepare Static Parts of Headers ---
    base_headers: Dict[str, str] = {}
    base_headers["User-Agent"] = random.choice(config_instance.USER_AGENTS)
    base_headers["Accept"] = "application/json, text/plain, */*"
    http_method = method.upper()
    if add_default_origin and http_method not in ["GET", "HEAD", "OPTIONS"]:
        try:
            parsed_base_url = urlparse(config_instance.BASE_URL)
            origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"
            base_headers["Origin"] = origin_header_value
        except Exception as parse_err:
            logger.warning(f"Could not parse BASE_URL for Origin header: {parse_err}")
    base_headers["Referer"] = referer_url or config_instance.BASE_URL
    base_headers.setdefault("Cache-Control", "no-cache")
    base_headers.setdefault("Pragma", "no-cache")
    contextual_headers = config_instance.API_CONTEXTUAL_HEADERS.get(api_description, {})
    for key, value in contextual_headers.items():
        if value is not None and key not in base_headers:
            base_headers[key] = value
    if headers:
        filtered_overrides = {k: v for k, v in headers.items() if v is not None}
        base_headers.update(filtered_overrides)
        if filtered_overrides:
            logger.debug(
                f"Applied {len(filtered_overrides)} explicit header overrides for {api_description}."
            )
    if api_description == "Match List API":
        base_headers.pop("Origin", None)
    # --- End Static Header Prep ---

    # Step 4: Prepare Request Details
    request_timeout = timeout if timeout is not None else selenium_config.API_TIMEOUT
    req_session = session_manager._requests_session
    effective_allow_redirects = allow_redirects
    if api_description == "Match List API" and effective_allow_redirects:
        logger.debug(f"Forcing allow_redirects=False for '{api_description}'.")
        effective_allow_redirects = False
    logger.debug(
        f"API Req: {http_method} {url} (Timeout: {request_timeout}s, AllowRedirects: {effective_allow_redirects})"
    )

    # Step 5: Execute Request with Retry Loop
    retries_left = max_retries
    last_exception = None
    delay = initial_delay

    while retries_left > 0:
        attempt = max_retries - retries_left + 1
        response: Optional[requests.Response] = None
        final_headers = base_headers.copy()

        current_driver = session_manager.driver
        driver_is_valid = current_driver and session_manager.is_sess_valid()
        if not driver_is_valid and (
            api_description not in ["CSRF Token API", "Get my profile_id"]
        ):
            logger.warning(
                f"{api_description}: Browser session invalid or driver is None (Attempt {attempt}). Dynamic headers might be incomplete."
            )

        try:
            # --- Sync Cookies ---
            if driver_is_valid:
                try:
                    session_manager._sync_cookies()
                    logger.debug(
                        f"{api_description}: Cookies synced (Attempt {attempt})."
                    )
                except Exception as sync_err:
                    logger.warning(
                        f"{api_description}: Error syncing cookies (Attempt {attempt}): {sync_err}"
                    )
            else:
                logger.debug(
                    f"{api_description}: Skipping cookie sync - driver invalid (Attempt {attempt})"
                )

            # --- Generate Dynamic Headers ---
            ua = None
            if driver_is_valid:
                try:
                    ua = current_driver.execute_script("return navigator.userAgent;")
                    final_headers["User-Agent"] = ua
                except Exception:
                    pass
            # --- CORRECTED CALLS (remove 'utils.') ---
            final_headers["newrelic"] = make_newrelic(current_driver) or ""
            final_headers["traceparent"] = make_traceparent(current_driver) or ""
            final_headers["tracestate"] = make_tracestate(current_driver) or ""
            final_headers = {k: v for k, v in final_headers.items() if v}
            if driver_is_valid:
                ube_header = make_ube(current_driver)
                if ube_header:
                    final_headers["ancestry-context-ube"] = ube_header
            # --- END CORRECTIONS ---

            # --- Conditionally Add ancestry-userid ---
            exclude_userid_for = [
                "Ancestry Facts JSON Endpoint",
                "Ancestry Person Picker",
            ]
            if (
                session_manager.my_profile_id
                and api_description not in exclude_userid_for
            ):
                final_headers["ancestry-userid"] = session_manager.my_profile_id.upper()
            elif api_description in exclude_userid_for:
                if session_manager.my_profile_id:
                    logger.debug(
                        f"Omitting 'ancestry-userid' header for '{api_description}'."
                    )
            # --- End Conditional UserID ---

            if use_csrf_token:  # CSRF Logic
                csrf_token = session_manager.csrf_token
                if csrf_token:
                    raw_token_val = csrf_token
                    if isinstance(csrf_token, str) and csrf_token.strip().startswith(
                        "{"
                    ):
                        try:
                            token_obj = json.loads(csrf_token)
                            raw_token_val = token_obj.get("csrfToken", csrf_token)
                        except Exception:
                            pass
                    final_headers["X-CSRF-Token"] = raw_token_val
                    logger.debug(
                        f"{api_description}: Added X-CSRF-Token header (Attempt {attempt})."
                    )
                else:
                    logger.warning(
                        f"{api_description}: CSRF token requested but not found (Attempt {attempt})."
                    )
            # --- End Dynamic Headers ---

            # Apply rate limit wait
            wait_time = session_manager.dynamic_rate_limiter.wait()
            if wait_time > 0.1:
                logger.debug(
                    f"{api_description}: Rate limit wait: {wait_time:.2f}s (Attempt {attempt})"
                )

            # --- Debug Logging ---
            log_level_for_debug = logging.DEBUG
            if logger.isEnabledFor(log_level_for_debug):
                log_hdrs_debug = {}
                sensitive_keys_debug = {
                    "authorization",
                    "cookie",
                    "x-csrf-token",
                    "ancestry-context-ube",
                    "newrelic",
                    "traceparent",
                    "tracestate",
                }
                for k, v in final_headers.items():
                    k_lower = k.lower()
                    if k_lower in sensitive_keys_debug and v and len(v) > 20:
                        log_hdrs_debug[k] = v[:10] + "..." + v[-5:]
                    elif v:
                        log_hdrs_debug[k] = v
                logger.debug(
                    f"[_api_req Attempt {attempt} for '{api_description}'] Sending Headers: {log_hdrs_debug}"
                )
                try:
                    cookie_jar_to_log = req_session.cookies
                    cookies_dict_log = cookie_jar_to_log.get_dict()
                    logger.debug(
                        f"[_api_req Attempt {attempt} for '{api_description}'] Sending Cookies: {list(cookies_dict_log.keys())}"
                    )
                except Exception as cookie_log_err:
                    logger.warning(
                        f"[_api_req DEBUG] Could not log cookies: {cookie_log_err}"
                    )
            # --- End Debug Logging ---

            # --- Make the request ---
            logger.debug(
                f"[_api_req Attempt {attempt} for '{api_description}'] >>> Calling requests.request..."
            )
            response = req_session.request(
                method=http_method,
                url=url,
                headers=final_headers,
                data=data,
                json=json_data,
                timeout=request_timeout,
                verify=True,
                allow_redirects=effective_allow_redirects,
                cookies=req_session.cookies,
            )
            logger.debug(
                f"[_api_req Attempt {attempt} for '{api_description}'] <<< requests.request returned."
            )
            # --- Request finished ---

            status = response.status_code
            logger.debug(f"<-- Response Status: {status} {response.reason}")

            # --- Process Response ---
            if status in retry_status_codes:
                retries_left -= 1
                last_exception = HTTPError(
                    f"{status} Error", response=response
                )  # Use requests.exceptions.HTTPError
                if retries_left <= 0:
                    logger.error(
                        f"{api_description}: Failed after {max_retries} attempts (Final Status {status})."
                    )
                    return response
                else:
                    sleep_time = min(
                        delay * (backoff_factor ** (attempt - 1)), max_delay
                    ) + random.uniform(0, 0.2)
                    sleep_time = max(0.1, sleep_time)
                    if status == 429:
                        session_manager.dynamic_rate_limiter.increase_delay()
                    logger.warning(
                        f"{api_description}: Status {status} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    delay *= backoff_factor
                    continue
            elif 300 <= status < 400 and not effective_allow_redirects:
                logger.warning(
                    f"{api_description}: Status {status} {response.reason} (Redirects Disabled)."
                )
                return response
            elif 300 <= status < 400 and effective_allow_redirects:
                logger.warning(
                    f"{api_description}: Unexpected final status {status} {response.reason} (Redirects Enabled)."
                )
                return response
            elif response.ok:
                session_manager.dynamic_rate_limiter.decrease_delay()
                if force_text_response:
                    return response.text
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    try:
                        return response.json() if response.content else None
                    except json.JSONDecodeError as json_err:
                        logger.error(
                            f"{api_description}: OK ({status}), but JSON decode FAILED: {json_err}\nContent: {response.text[:500]}"
                        )
                        return None
                elif (
                    api_description == "CSRF Token API" and "text/plain" in content_type
                ):
                    csrf_text = response.text.strip()
                    return csrf_text if csrf_text else None
                else:
                    logger.debug(
                        f"{api_description}: OK ({status}), Content-Type '{content_type}'. Returning raw TEXT."
                    )
                    return response.text
            else:
                if status in [401, 403]:
                    logger.warning(
                        f"{api_description}: API call failed {status} {response.reason}. Session expired/invalid?"
                    )
                    session_manager.session_ready = False
                else:
                    logger.error(
                        f"{api_description}: Non-retryable error: {status} {response.reason}."
                    )
                try:
                    logger.debug(f"Error Response Body: {response.text[:500]}")
                except Exception:
                    pass
                return response

        # --- Handle exceptions during the request attempt ---
        except RequestException as e:  # Use requests.exceptions.RequestException
            logger.debug(
                f"[_api_req Attempt {attempt} for '{api_description}'] <<< requests.request raised {type(e).__name__}"
            )
            retries_left -= 1
            last_exception = e
            exception_type_name = type(e).__name__
            if retries_left <= 0:
                logger.error(
                    f"{api_description}: {exception_type_name} failed after {max_retries} attempts. Error: {e}",
                    exc_info=False,
                )
                return None
            else:
                sleep_time = min(
                    delay * (backoff_factor ** (attempt - 1)), max_delay
                ) + random.uniform(0, 0.2)
                sleep_time = max(0.1, sleep_time)
                logger.warning(
                    f"{api_description}: {exception_type_name} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s... Error: {e}"
                )
                time.sleep(sleep_time)
                delay *= backoff_factor
                continue
        except Exception as e:
            logger.debug(
                f"[_api_req Attempt {attempt} for '{api_description}'] <<< requests.request raised UNEXPECTED {type(e).__name__}"
            )
            logger.critical(
                f"{api_description}: CRITICAL Unexpected error during request attempt {attempt}: {e}",
                exc_info=True,
            )
            return None

    # Should only be reached if loop completes without success
    logger.error(
        f"{api_description}: Exited retry loop unexpectedly. Last Exception: {last_exception}."
    )
    return None


# End of _api_req

def make_ube(driver: Optional[WebDriver]) -> Optional[str]:
    """
    Generates the 'ancestry-context-ube' header value based on current browser state.
    Ensures the 'correlatedSessionId' matches the current 'ANCSESSIONID' cookie.
    """
    if not driver:
        return None
    try:
        _ = driver.window_handles  # Quick check if driver is responsive
    except WebDriverException as e:
        logger.warning(
            f"Cannot generate UBE header: Session invalid/unresponsive ({type(e).__name__})."
        )
        return None
    ancsessionid = None
    try:
        cookie_obj = driver.get_cookie("ANCSESSIONID")
        if cookie_obj and "value" in cookie_obj:
            ancsessionid = cookie_obj["value"]
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
    event_id = "00000000-0000-0000-0000-000000000000"
    correlated_id = str(uuid.uuid4())
    screen_name_standard = "ancestry : uk : en : dna-matches-ui : match-list : 1"
    screen_name_legacy = "ancestry uk : dnamatches-matchlistui : list"
    user_consent = "necessary|preference|performance|analytics1st|analytics3rd|advertising1st|advertising3rd|attribution3rd"
    ube_data = {
        "eventId": event_id,
        "correlatedScreenViewedId": correlated_id,
        "correlatedSessionId": ancsessionid,
        "screenNameStandard": screen_name_standard,
        "screenNameLegacy": screen_name_legacy,
        "userConsent": user_consent,
        "vendors": "adobemc",
        "vendorConfigurations": "{}",
    }
    try:
        json_payload = json.dumps(ube_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        return encoded_payload
    except Exception as e:
        logger.error(f"Error encoding UBE header data: {e}", exc_info=True)
        return None


# End of make_ube


def make_newrelic(driver: Optional[WebDriver]) -> Optional[str]:
    """Generates the 'newrelic' header value."""
    try:
        trace_id = uuid.uuid4().hex[:16]
        span_id = uuid.uuid4().hex[:16]
        account_id = "1690570"
        app_id = "1588726612"
        tk = "2611750"
        newrelic_data = {
            "v": [0, 1],
            "d": {
                "ty": "Browser",
                "ac": account_id,
                "ap": app_id,
                "id": span_id,
                "tr": trace_id,
                "ti": int(time.time() * 1000),
                "tk": tk,
            },
        }
        json_payload = json.dumps(newrelic_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        return encoded_payload
    except Exception as e:
        logger.error(f"Error generating NewRelic header: {e}", exc_info=True)
        return None


# End of make_newrelic


def make_traceparent(driver: Optional[WebDriver]) -> Optional[str]:
    """Generates the 'traceparent' header value."""
    try:
        version = "00"
        trace_id = uuid.uuid4().hex
        parent_id = uuid.uuid4().hex[:16]
        flags = "01"
        traceparent = f"{version}-{trace_id}-{parent_id}-{flags}"
        return traceparent
    except Exception as e:
        logger.error(f"Error generating traceparent header: {e}", exc_info=True)
        return None


# End of make_traceparent


def make_tracestate(driver: Optional[WebDriver]) -> Optional[str]:
    """Generates the 'tracestate' header value."""
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


def _send_message_via_api(
    session_manager: SessionManager,
    person: Person,
    message_text: str,
    existing_conv_id: Optional[str],
    log_prefix: str,  # Log prefix for context (e.g., Person ID/Username)
) -> Tuple[Optional[str], Optional[str]]:
    """Sends a message using the appropriate Ancestry messaging API endpoint."""
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
    is_initial = not existing_conv_id
    app_mode = config_instance.APP_MODE
    if app_mode == "dry_run":
        message_status = SEND_SUCCESS_DRY_RUN
        effective_conv_id = existing_conv_id or f"dryrun_{uuid.uuid4()}"
        logger.debug(f"{log_prefix}: Dry Run - Message send simulation successful.")
        return message_status, effective_conv_id
    if app_mode not in ["production", "testing"]:
        logger.error(
            f"{log_prefix}: Logic Error - Unexpected APP_MODE '{app_mode}' reached send logic."
        )
        return SEND_ERROR_INTERNAL_MODE, None
    send_api_url: str = ""
    payload: Dict[str, Any] = {}
    send_api_desc: str = ""
    api_headers: Dict[str, Any] = {}
    if is_initial:
        send_api_url = urljoin(
            config_instance.BASE_URL.rstrip("/") + "/",
            "app-api/express/v2/conversations/message",
        )
        send_api_desc = "Create Conversation API"
        payload = {
            "content": message_text,
            "author": MY_PROFILE_ID_LOWER,
            "index": 0,
            "created": 0,
            "conversation_members": [
                {"user_id": recipient_profile_id_upper.lower(), "family_circles": []},
                {"user_id": MY_PROFILE_ID_LOWER},
            ],
        }
    elif existing_conv_id:
        send_api_url = urljoin(
            config_instance.BASE_URL.rstrip("/") + "/",
            f"app-api/express/v2/conversations/{existing_conv_id}",
        )
        send_api_desc = "Send Message API (Existing Conv)"
        payload = {"content": message_text, "author": MY_PROFILE_ID_LOWER}
    else:
        logger.error(
            f"{log_prefix}: Logic Error - Cannot determine API URL/payload (existing_conv_id issue?)."
        )
        return SEND_ERROR_API_PREP_FAILED, None
    ctx_headers = config_instance.API_CONTEXTUAL_HEADERS.get(send_api_desc, {})
    api_headers = ctx_headers.copy()
    api_response = _api_req(
        url=send_api_url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="POST",
        json_data=payload,
        use_csrf_token=False,
        headers=api_headers,
        api_description=send_api_desc,
    )
    message_status = SEND_ERROR_UNKNOWN
    new_conversation_id_from_api: Optional[str] = None
    post_ok = False
    api_conv_id: Optional[str] = None
    api_author: Optional[str] = None
    if api_response is not None:
        if isinstance(api_response, dict):
            if is_initial:
                api_conv_id = str(api_response.get("conversation_id", ""))
                msg_details = api_response.get("message", {})
                api_author = (
                    str(msg_details.get("author", "")).upper()
                    if isinstance(msg_details, dict)
                    else None
                )
                if api_conv_id and api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = api_conv_id
                else:
                    logger.error(
                        f"{log_prefix}: API initial response invalid (ConvID: {api_conv_id}, Author: {api_author})."
                    )
            else:
                api_author = str(api_response.get("author", "")).upper()
                if api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = existing_conv_id
                else:
                    logger.error(
                        f"{log_prefix}: API follow-up author validation failed (Author: {api_author})."
                    )
        elif isinstance(api_response, requests.Response):
            message_status = f"send_error (http_{api_response.status_code})"
            logger.error(
                f"{log_prefix}: API POST ({send_api_desc}) failed with status {api_response.status_code}."
            )
        else:
            logger.error(
                f"{log_prefix}: API call ({send_api_desc}) unexpected success format. Type:{type(api_response)}, Resp:{api_response}"
            )
            message_status = SEND_ERROR_UNEXPECTED_FORMAT
        if post_ok:
            message_status = SEND_SUCCESS_DELIVERED
            logger.debug(f"{log_prefix}: Message send to {log_prefix} ACCEPTED by API.")
        elif message_status == SEND_ERROR_UNKNOWN:
            message_status = SEND_ERROR_VALIDATION_FAILED
            logger.warning(
                f"{log_prefix}: API POST validation failed after receiving unexpected success response."
            )
    else:
        message_status = SEND_ERROR_POST_FAILED
        logger.error(
            f"{log_prefix}: API POST ({send_api_desc}) failed (No response/Retries exhausted)."
        )
    return message_status, new_conversation_id_from_api


# End of _send_message_via_api


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_profile_details_for_person(
    session_manager: SessionManager, profile_id: str
) -> Optional[Dict[str, Any]]:
    """Fetches profile details for a specific Person using their Profile ID."""
    if not profile_id:
        logger.warning("_fetch_profile_details_for_person: Profile ID missing.")
        return None
    if not session_manager or not session_manager.my_profile_id:
        logger.error(
            "_fetch_profile_details_for_person: SessionManager or own profile ID missing."
        )
        return None
    if not session_manager.is_sess_valid():
        logger.error(
            f"_fetch_profile_details_for_person: Session invalid for Profile ID {profile_id}."
        )
        raise ConnectionError(
            f"WebDriver session invalid before profile details fetch (Profile: {profile_id})"
        )
    api_description = "Profile Details API (Action 7)"
    profile_url = urljoin(
        config_instance.BASE_URL,
        f"/app-api/express/v1/profiles/details?userId={profile_id.upper()}",
    )
    referer_url = urljoin(config_instance.BASE_URL, "/messaging/")
    logger.debug(
        f"Fetching profile details ({api_description}) for Profile ID {profile_id}..."
    )
    try:
        profile_response = _api_req(
            url=profile_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers={},
            use_csrf_token=False,
            api_description=api_description,
            referer_url=referer_url,
        )
        if profile_response and isinstance(profile_response, dict):
            logger.debug(f"Successfully fetched profile details for {profile_id}.")
            result_data: Dict[str, Any] = {}
            first_name_raw = profile_response.get("FirstName")
            if first_name_raw and isinstance(first_name_raw, str):
                result_data["first_name"] = format_name(first_name_raw)
            else:
                display_name_raw = profile_response.get("DisplayName")
                if display_name_raw:
                    formatted_dn = format_name(display_name_raw)
                    result_data["first_name"] = (
                        formatted_dn.split()[0]
                        if formatted_dn != "Valued Relative"
                        else None
                    )
                else:
                    result_data["first_name"] = None
            contactable_val = profile_response.get("IsContactable")
            result_data["contactable"] = (
                bool(contactable_val) if contactable_val is not None else False
            )
            last_login_str = profile_response.get("LastLoginDate")
            last_logged_in_dt_aware: Optional[datetime] = None
            if last_login_str:
                try:
                    if last_login_str.endswith("Z"):
                        last_logged_in_dt_aware = datetime.fromisoformat(
                            last_login_str.replace("Z", "+00:00")
                        )
                    else:
                        dt_naive = datetime.fromisoformat(last_login_str)
                        last_logged_in_dt_aware = (
                            dt_naive.replace(tzinfo=timezone.utc)
                            if dt_naive.tzinfo is None
                            else dt_naive.astimezone(timezone.utc)
                        )
                    result_data["last_logged_in_dt"] = last_logged_in_dt_aware
                except (ValueError, TypeError) as date_parse_err:
                    logger.warning(
                        f"Could not parse LastLoginDate '{last_login_str}' for {profile_id}: {date_parse_err}"
                    )
                    result_data["last_logged_in_dt"] = None
            else:
                result_data["last_logged_in_dt"] = None
            return result_data
        elif isinstance(profile_response, requests.Response):
            logger.warning(
                f"Failed profile details fetch for {profile_id}. Status: {profile_response.status_code}."
            )
            return None
        else:
            logger.warning(
                f"Failed profile details fetch for {profile_id} (Invalid response type: {type(profile_response)})."
            )
            return None
    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching profile details for {profile_id}: {conn_err}",
            exc_info=False,
        )
        raise
    except requests.exceptions.RequestException as req_e:
        logger.error(
            f"RequestException fetching profile details for {profile_id}: {req_e}",
            exc_info=False,
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error fetching profile details for {profile_id}: {e}",
            exc_info=True,
        )
        return None


# End of _fetch_profile_details_for_person


# ----------------------------------------------------------------------------
# Login Functions (Remain in utils.py)
# ----------------------------------------------------------------------------
# Define the CSS selector for the Two-Step Verification header
TWO_STEP_VERIFICATION_HEADER_SELECTOR = "h1.two-step-verification-header"  # Replace with the actual selector

# Login Helper 5
@time_wait("Handle 2FA Page")
def handle_twoFA(session_manager: SessionManager) -> bool:
    # ... (Implementation remains unchanged) ...
    if session_manager.driver is None:
        logger.error("handle_twoFA: SessionManager driver is None. Cannot proceed.")
        return False
    driver = session_manager.driver
    element_wait = selenium_config.element_wait(driver)
    page_wait = selenium_config.page_wait(driver)
    short_wait = selenium_config.short_wait(driver)
    try:
        logger.debug("Handling Two-Factor Authentication (2FA)...")
        try:
            logger.debug(
                f"Waiting for 2FA page header using selector: '{TWO_STEP_VERIFICATION_HEADER_SELECTOR}'"
            )
            element_wait.until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                )
            )
            logger.debug("2FA page header detected.")
        except TimeoutException:
            logger.debug("Did not detect 2FA page header within timeout.")
            if login_status(session_manager):
                logger.info(
                    "User appears logged in after checking for 2FA page. Assuming 2FA handled/skipped."
                )
                return True
            logger.warning(
                "Assuming 2FA not required or page didn't load correctly (header missing)."
            )
            return False
        try:
            logger.debug(
                f"Waiting for 2FA 'Send Code' (SMS) button: '{TWO_FA_SMS_SELECTOR}'"
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
                try:
                    logger.debug(
                        f"Waiting for 2FA code input field: '{TWO_FA_CODE_INPUT_SELECTOR}'"
                    )
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
                        "Code input field did not appear/become visible after clicking 'Send Code'."
                    )
                except Exception as e_input:
                    logger.error(
                        f"Error waiting for 2FA code input field: {e_input}. Check selector: {TWO_FA_CODE_INPUT_SELECTOR}"
                    )
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
        code_entry_timeout = selenium_config.TWO_FA_CODE_ENTRY_TIMEOUT
        logger.warning(
            f"Waiting up to {code_entry_timeout}s for user to manually enter 2FA code and submit..."
        )
        start_time = time.time()
        user_action_detected = False
        while time.time() - start_time < code_entry_timeout:
            try:
                WebDriverWait(driver, 0.5).until(
                    EC.visibility_of_element_located(
                        (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                    )
                )
                time.sleep(2)
            except TimeoutException:
                logger.info(
                    "2FA page elements disappeared, assuming user submitted code."
                )
                user_action_detected = True
                break
            except WebDriverException as e:
                logger.error(
                    f"WebDriver error checking for 2FA header during wait: {e}"
                )
                break
            except Exception as e:
                logger.error(
                    f"Unexpected error checking for 2FA header during wait: {e}"
                )
                break
        if user_action_detected:
            logger.info("Re-checking login status after 2FA page disappearance...")
            time.sleep(1)
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
        else:
            logger.error(
                f"Timed out ({code_entry_timeout}s) waiting for user 2FA action (page did not change)."
            )
            return False
    except WebDriverException as e:
        logger.error(f"WebDriverException during 2FA handling: {e}")
        if not is_browser_open(driver):
            logger.error("Session invalid after WebDriverException during 2FA.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during 2FA handling: {e}", exc_info=True)
        return False
    return False  # Ensure a return value for all code paths


# End of handle_twoFA


# Login Helper 4
def enter_creds(driver: WebDriver) -> bool:
    # ... (Implementation remains unchanged) ...
    element_wait = selenium_config.element_wait(driver)
    short_wait = selenium_config.short_wait(driver)
    time.sleep(random.uniform(0.5, 1.0))
    try:
        logger.debug("Entering Credentials and Signing In...")
        logger.debug(f"Waiting for username input: '{USERNAME_INPUT_SELECTOR}'...")
        username_input = element_wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))
        )
        logger.debug("Username input field found.")
        try:
            username_input.click()
            time.sleep(0.2)
            username_input.clear()
            time.sleep(0.2)
        except Exception as e:
            logger.warning(
                f"Issue clicking/clearing username field ({e}). Attempting JS clear."
            )
            try:
                driver.execute_script("arguments[0].value = '';", username_input)
            except Exception as js_e:
                logger.error(f"Failed JS clear username: {js_e}")
                return False
        ancestry_username = config_instance.ANCESTRY_USERNAME
        if not ancestry_username:
            raise ValueError("ANCESTRY_USERNAME configuration is missing.")
        logger.debug(f"Entering username...")
        username_input.send_keys(ancestry_username)
        logger.debug("Username entered.")
        time.sleep(0.3)
        logger.debug(f"Waiting for password input: '{PASSWORD_INPUT_SELECTOR}'...")
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
                f"Issue clicking/clearing password field ({e}). Attempting JS clear."
            )
            try:
                driver.execute_script("arguments[0].value = '';", password_input)
            except Exception as js_e:
                logger.error(f"Failed JS clear password: {js_e}")
                return False
        ancestry_password = config_instance.ANCESTRY_PASSWORD
        if not ancestry_password:
            raise ValueError("ANCESTRY_PASSWORD configuration is missing.")
        logger.debug("Entering password: ***")
        password_input.send_keys(ancestry_password)
        logger.debug("Password entered.")
        time.sleep(0.5)
        sign_in_button = None
        try:
            logger.debug(
                f"Waiting for sign in button presence: '{SIGN_IN_BUTTON_SELECTOR}'..."
            )
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR)
                )
            )
            logger.debug("Waiting for sign in button clickability...")
            sign_in_button = short_wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))
            )
            logger.debug("Sign in button located and deemed clickable.")
        except TimeoutException:
            logger.error("Sign in button not found or not clickable within timeout.")
            logger.warning("Attempting fallback: Sending RETURN key to password field.")
            try:
                password_input.send_keys(Keys.RETURN)
                logger.info("Fallback RETURN key sent to password field.")
                return True
            except Exception as key_e:
                logger.error(f"Failed to send RETURN key: {key_e}")
                return False
        except Exception as find_e:
            logger.error(f"Unexpected error finding sign in button: {find_e}")
            return False
        click_successful = False
        if sign_in_button:
            try:
                logger.debug("Attempting standard click on sign in button...")
                sign_in_button.click()
                logger.debug("Standard click executed.")
                click_successful = True
            except ElementClickInterceptedException:
                logger.warning("Standard click intercepted. Trying JS click...")
            except ElementNotInteractableException as eni_e:
                logger.warning(
                    f"Button not interactable for standard click: {eni_e}. Trying JS click..."
                )
            except Exception as click_e:
                logger.error(
                    f"Error during standard click: {click_e}. Trying JS click..."
                )
            if not click_successful:
                try:
                    logger.debug("Attempting JavaScript click on sign in button...")
                    driver.execute_script("arguments[0].click();", sign_in_button)
                    logger.info("JavaScript click executed.")
                    click_successful = True
                except Exception as js_click_e:
                    logger.error(f"Error during JavaScript click: {js_click_e}")
            if not click_successful:
                logger.warning(
                    "Both standard and JS clicks failed. Attempting fallback: Sending RETURN key."
                )
                try:
                    password_input.send_keys(Keys.RETURN)
                    logger.info(
                        "Fallback RETURN key sent to password field after failed clicks."
                    )
                    click_successful = True
                except Exception as key_e:
                    logger.error(
                        f"Failed to send RETURN key as final fallback: {key_e}"
                    )
        return click_successful
    except TimeoutException as e:
        logger.error(f"Timeout finding username or password input field: {e}")
        return False
    except NoSuchElementException as e:
        logger.error(f"Username or password input not found (NoSuchElement): {e}")
        return False
    except ValueError as ve:
        logger.critical(f"Configuration Error: {ve}")
        return False
    except WebDriverException as e:
        logger.error(f"WebDriver error entering credentials: {e}")
        if not is_browser_open(driver):
            logger.error("Session invalid during credential entry.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error entering credentials: {e}", exc_info=True)
        return False


# End of enter_creds


# Login Helper 3
@retry(MAX_RETRIES=2, BACKOFF_FACTOR=1, MAX_DELAY=3)
def consent(driver: WebDriver) -> bool:
    # ... (Implementation remains unchanged) ...
    if not driver:
        logger.error("consent: WebDriver instance is None.")
        return False
    logger.debug(f"Checking for cookie consent overlay: '{COOKIE_BANNER_SELECTOR}'")
    overlay_element = None
    try:
        overlay_element = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR))
        )
        logger.debug("Cookie consent overlay DETECTED.")
    except TimeoutException:
        logger.debug("Cookie consent overlay not found. Assuming no consent needed.")
        return True
    except Exception as e:
        logger.error(f"Error checking for consent banner: {e}")
        return False
    removed_via_js = False
    if overlay_element:
        try:
            logger.debug("Attempting JS removal of consent overlay...")
            driver.execute_script("arguments[0].remove();", overlay_element)
            time.sleep(0.5)
            try:
                driver.find_element(By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                logger.warning(
                    "Consent overlay still present after JS removal attempt."
                )
            except NoSuchElementException:
                logger.debug("Cookie consent overlay REMOVED successfully via JS.")
                removed_via_js = True
                return True
        except WebDriverException as js_err:
            logger.warning(f"Error removing consent overlay via JS: {js_err}")
        except Exception as e:
            logger.warning(f"Unexpected error during JS removal of consent: {e}")
    if not removed_via_js:
        logger.debug(
            f"JS removal failed/skipped. Trying specific accept button: '{consent_ACCEPT_BUTTON_SELECTOR}'"
        )
        try:
            accept_button = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, consent_ACCEPT_BUTTON_SELECTOR)
                )
            )
            if accept_button:
                logger.info("Found specific clickable accept button.")
                try:
                    accept_button.click()
                    logger.info("Clicked accept button successfully.")
                    time.sleep(1)
                    try:
                        driver.find_element(By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                        logger.warning(
                            "Consent overlay still present after clicking accept button."
                        )
                        return False
                    except NoSuchElementException:
                        logger.debug(
                            "Consent overlay gone after clicking accept button."
                        )
                        return True
                except ElementClickInterceptedException:
                    logger.warning(
                        "Click intercepted for accept button, trying JS click..."
                    )
                    try:
                        driver.execute_script("arguments[0].click();", accept_button)
                        logger.info("Clicked accept button via JS successfully.")
                        time.sleep(1)
                        try:
                            driver.find_element(By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                            logger.warning(
                                "Consent overlay still present after JS clicking accept button."
                            )
                            return False
                        except NoSuchElementException:
                            logger.debug(
                                "Consent overlay gone after JS clicking accept button."
                            )
                            return True
                    except Exception as js_click_err:
                        logger.error(
                            f"Failed JS click for accept button: {js_click_err}"
                        )
                except Exception as click_err:
                    logger.error(f"Error clicking accept button: {click_err}")
        except TimeoutException:
            logger.warning(
                f"Specific accept button '{consent_ACCEPT_BUTTON_SELECTOR}' not found or not clickable."
            )
        except Exception as find_err:
            logger.error(f"Error finding/clicking specific accept button: {find_err}")
        logger.warning("Could not remove consent overlay via JS or button click.")
        return False
    return False


# End of consent


# Login Main Function 2
def log_in(session_manager: SessionManager) -> str:
    # ... (Implementation remains unchanged) ...
    driver = session_manager.driver
    if not driver:
        logger.error("Login failed: WebDriver not available in SessionManager.")
        return "LOGIN_ERROR_NO_DRIVER"
    signin_url = urljoin(config_instance.BASE_URL, "account/signin")
    try:
        logger.info(f"Navigating to sign-in page: {signin_url}")
        if not nav_to_page(
            driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager
        ):
            logger.debug(
                "Navigation to sign-in page failed/redirected. Checking login status..."
            )
            current_status = login_status(session_manager)
            if current_status is True:
                logger.info(
                    "Detected as already logged in. Login considered successful."
                )
                return "LOGIN_SUCCEEDED"
            else:
                logger.error("Failed to navigate to login page (and not logged in).")
                return "LOGIN_FAILED_NAVIGATION"
        logger.debug("Successfully navigated to sign-in page.")
        if not consent(driver):
            logger.warning("Failed to handle consent banner, login might be impacted.")
        if not enter_creds(driver):
            logger.error("Failed during credential entry or submission.")
            try:
                WebDriverWait(driver, 1).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, FAILED_LOGIN_SELECTOR)
                    )
                )
                logger.error(
                    "Login failed: Specific 'Invalid Credentials' alert detected."
                )
                return "LOGIN_FAILED_BAD_CREDS"
            except TimeoutException:
                generic_alert_selector = "div.alert[role='alert']"
                try:
                    alert_element = WebDriverWait(driver, 0.5).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, generic_alert_selector)
                        )
                    )
                    alert_text = (
                        alert_element.text if alert_element else "Unknown error"
                    )
                    logger.error(f"Login failed: Generic alert found: '{alert_text}'.")
                    return "LOGIN_FAILED_ERROR_DISPLAYED"
                except TimeoutException:
                    logger.error(
                        "Login failed: Credential entry failed, but no specific or generic alert found."
                    )
                    return "LOGIN_FAILED_CREDS_ENTRY"
            except Exception as e:
                logger.warning(
                    f"Error checking for login error message after cred entry failed: {e}"
                )
                return "LOGIN_FAILED_CREDS_ENTRY"
        logger.debug("Credentials submitted. Waiting for potential page change...")
        time.sleep(random.uniform(3.0, 5.0))
        two_fa_present = False
        try:
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                )
            )
            two_fa_present = True
            logger.info("Two-step verification page detected.")
        except TimeoutException:
            logger.debug("Two-step verification page not detected.")
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
            if handle_twoFA(session_manager):
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
        else:
            logger.debug("Checking login status directly (no 2FA detected)...")
            login_check_result = login_status(session_manager)
            if login_check_result is True:
                logger.info("Direct login check successful.")
                return "LOGIN_SUCCEEDED"
            elif login_check_result is False:
                logger.error(
                    "Direct login check failed. Checking for error messages again..."
                )
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
                    generic_alert_selector = "div.alert[role='alert']"
                    try:
                        alert_element = WebDriverWait(driver, 0.5).until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, generic_alert_selector)
                            )
                        )
                        alert_text = (
                            alert_element.text if alert_element else "Unknown error"
                        )
                        logger.error(
                            f"Login failed: Generic alert found (post-check): '{alert_text}'."
                        )
                        return "LOGIN_FAILED_ERROR_DISPLAYED"
                    except TimeoutException:
                        try:
                            if driver.current_url.startswith(signin_url):
                                logger.error(
                                    "Login failed: Still on login page (post-check), no error message found."
                                )
                                return "LOGIN_FAILED_STUCK_ON_LOGIN"
                            else:
                                logger.error(
                                    "Login failed: Status False, no 2FA, no error msg, not on login page."
                                )
                                return "LOGIN_FAILED_UNKNOWN"
                        except WebDriverException:
                            logger.error(
                                "Login failed: Status False, WebDriverException getting URL."
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
            else:
                logger.error(
                    "Login failed: Critical error during final login status check."
                )
                return "LOGIN_FAILED_STATUS_CHECK_ERROR"
    except TimeoutException as e:
        logger.error(f"Timeout during login process: {e}", exc_info=False)
        return "LOGIN_FAILED_TIMEOUT"
    except WebDriverException as e:
        logger.error(f"WebDriverException during login: {e}", exc_info=False)
        if not is_browser_open(driver):
            logger.error("Session became invalid during login.")
        return "LOGIN_FAILED_WEBDRIVER"
    except Exception as e:
        logger.error(f"An unexpected error occurred during login: {e}", exc_info=True)
        return "LOGIN_FAILED_UNEXPECTED"


# End of log_in


# Login Status Check Function 1
@retry(MAX_RETRIES=2)
def login_status(session_manager: SessionManager) -> Optional[bool]:
    # ... (Implementation remains unchanged) ...
    logger.debug("Checking login status (API prioritized)...")
    if not isinstance(session_manager, SessionManager):
        logger.error(
            f"Invalid argument: Expected SessionManager, got {type(session_manager)}."
        )
        return None
    if not session_manager.is_sess_valid():
        logger.debug("Login status check: Session invalid or browser closed.")
        return False
    driver = session_manager.driver
    if driver is None:
        logger.error("Login status check: Driver is None within SessionManager.")
        return None
    logger.debug("Attempting API login verification (_verify_api_login_status)...")
    api_check_result = session_manager._verify_api_login_status()
    if api_check_result is True:
        logger.debug("Login status confirmed TRUE via API check.")
        return True
    elif api_check_result is False:
        logger.debug(
            "API check indicates user NOT logged in. Proceeding to UI check as fallback/confirmation."
        )
    else:
        logger.warning(
            "API login check returned None (critical error). Falling back to UI check."
        )
    logger.debug("Performing fallback UI login check...")
    try:
        login_button_selector = LOG_IN_BUTTON_SELECTOR
        logger.debug(
            f"UI Check Step 1: Checking ABSENCE of login button: '{login_button_selector}'"
        )
        login_button_present = False
        try:
            WebDriverWait(driver, 2).until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, login_button_selector)
                )
            )
            login_button_present = True
            logger.debug("Login button FOUND during UI check.")
        except TimeoutException:
            logger.debug("Login button NOT found during UI check (good indication).")
            login_button_present = False
        except Exception as e:
            logger.warning(f"Error checking for login button presence: {e}")
        if login_button_present:
            logger.debug(
                "Login status confirmed FALSE via UI check (login button found)."
            )
            return False
        else:
            logged_in_selector = CONFIRMED_LOGGED_IN_SELECTOR
            logger.debug(
                f"UI Check Step 2: Checking PRESENCE of logged-in element: '{logged_in_selector}'"
            )
            ui_element_present = is_elem_there(
                driver, By.CSS_SELECTOR, logged_in_selector, wait=3
            )
            if ui_element_present:
                logger.debug(
                    "Login status confirmed TRUE via UI check (login button absent AND logged-in element found)."
                )
                return True
            else:
                current_url_context = "Unknown"
                try:
                    current_url_context = driver.current_url
                except Exception:
                    pass
                logger.warning(
                    f"Login status ambiguous: API failed/false, UI elements inconclusive at URL: {current_url_context}"
                )
                return False
    except WebDriverException as e:
        logger.error(f"WebDriverException during UI login_status check: {e}")
        if not is_browser_open(driver):
            logger.error("Session became invalid during UI login_status check.")
            session_manager.close_sess()
        return None
    except Exception as e:
        logger.critical(
            f"CRITICAL Unexpected error during UI login_status check: {e}",
            exc_info=True,
        )
        return None


# End of login_status


# ------------------------------------------------------------------------------------
# Navigation Functions (Remains in utils.py)
# ------------------------------------------------------------------------------------
def nav_to_page(
    driver: WebDriver,
    url: str,
    selector: str = "body",
    session_manager: Optional[SessionManager] = None,
) -> bool:
    # ... (Implementation remains unchanged, uses is_browser_open, _check_for_unavailability, login_status, log_in) ...
    if not driver:
        logger.error("Navigation failed: WebDriver instance is None.")
        return False
    if not url:
        logger.error("Navigation failed: Target URL is required.")
        return False
    max_attempts = config_instance.MAX_RETRIES
    page_timeout = selenium_config.PAGE_TIMEOUT
    element_timeout = selenium_config.ELEMENT_TIMEOUT
    target_url_parsed = urlparse(url)
    target_url_base = f"{target_url_parsed.scheme}://{target_url_parsed.netloc}{target_url_parsed.path}".rstrip(
        "/"
    )
    signin_page_url_base = urljoin(config_instance.BASE_URL, "account/signin").rstrip(
        "/"
    )
    mfa_page_url_base = urljoin(config_instance.BASE_URL, "account/signin/mfa/").rstrip(
        "/"
    )
    unavailability_selectors = {
        TEMP_UNAVAILABLE_SELECTOR: ("refresh", 5),
        PAGE_NO_LONGER_AVAILABLE_SELECTOR: ("skip", 0),
    }
    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Navigation Attempt {attempt}/{max_attempts} to: {url}")
        landed_url = ""
        try:
            if not is_browser_open(driver):
                logger.error(
                    f"Navigation failed (Attempt {attempt}): Browser session invalid before nav."
                )
                if session_manager:
                    logger.warning("Attempting session restart...")
                    if session_manager.restart_sess():
                        logger.info("Session restarted. Retrying navigation...")
                        driver = session_manager.driver
                        if not driver:
                            logger.error(
                                "Session restart reported success but driver is still None."
                            )
                            return False
                        continue
                    else:
                        logger.error("Session restart failed. Cannot navigate.")
                        return False
                else:
                    return False
            logger.debug(f"Executing driver.get('{url}')...")
            driver.get(url)
            WebDriverWait(driver, page_timeout).until(
                lambda d: d.execute_script("return document.readyState")
                in ["complete", "interactive"]
            )
            time.sleep(random.uniform(0.5, 1.5))
            try:
                landed_url = driver.current_url
                landed_url_parsed = urlparse(landed_url)
                landed_url_base = f"{landed_url_parsed.scheme}://{landed_url_parsed.netloc}{landed_url_parsed.path}".rstrip(
                    "/"
                )
                logger.debug(f"Landed on URL base: {landed_url_base}")
            except WebDriverException as e:
                logger.error(
                    f"Failed to get URL after get() (Attempt {attempt}): {e}. Retrying."
                )
                continue
            is_on_mfa_page = False
            try:
                WebDriverWait(driver, 1).until(
                    EC.visibility_of_element_located(
                        (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                    )
                )
                is_on_mfa_page = True
            except TimeoutException:
                pass
            except WebDriverException as e:
                logger.warning(f"WebDriverException checking for MFA header: {e}")
            if is_on_mfa_page:
                logger.warning("Landed on MFA page unexpectedly during navigation.")
                return False
            is_on_login_page = False
            if (
                landed_url_base == signin_page_url_base
                or target_url_base != signin_page_url_base
            ):
                try:
                    WebDriverWait(driver, 1).until(
                        EC.visibility_of_element_located(
                            (By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR)
                        )
                    )
                    is_on_login_page = True
                except TimeoutException:
                    pass
                except WebDriverException as e:
                    logger.warning(
                        f"WebDriverException checking for Login username input: {e}"
                    )
            if is_on_login_page and target_url_base != signin_page_url_base:
                logger.warning("Landed on Login page unexpectedly.")
                if session_manager:
                    logger.info("Attempting re-login...")
                    login_stat = login_status(session_manager)
                    if login_stat is True:
                        logger.info(
                            "Login status OK after landing on login page redirect. Retrying navigation."
                        )
                        continue
                    login_result = log_in(session_manager)
                    logger.debug(f"[Readiness] log_in() returned: {login_result}")
                    if login_result == "LOGIN_SUCCEEDED":
                        logger.info("[Persistence] If you just completed 2FA, DO NOT CLOSE the browser or delete the user data directory! Let the session settle for a minute, then rerun automation to avoid future 2FA prompts.")
                        logger.info("Re-login successful. Retrying original navigation...")
                        continue
                    else:
                        logger.error(f"Re-login attempt failed ({login_result}). Cannot complete navigation.")
                        return False
                else:
                    logger.error(
                        "Landed on login page, no SessionManager provided for re-login attempt."
                    )
                    return False
            if landed_url_base != target_url_base:
                is_signin_to_base_redirect = (
                    target_url_base == signin_page_url_base
                    and landed_url_base
                    == urlparse(config_instance.BASE_URL).path.rstrip("/")
                )
                if is_signin_to_base_redirect:
                    logger.debug(
                        "Redirected from signin page to base URL. Verifying login status..."
                    )
                    time.sleep(1)
                    if session_manager and login_status(session_manager) is True:
                        logger.info(
                            "Redirect after signin confirmed as logged in. Navigation OK."
                        )
                        return True
                logger.warning(
                    f"Navigation landed on unexpected URL base: '{landed_url_base}' (Expected: '{target_url_base}')"
                )
                action, wait_time = _check_for_unavailability(
                    driver, unavailability_selectors
                )
                if action == "skip":
                    return False
                if action == "refresh":
                    time.sleep(wait_time)
                    continue
                logger.warning(
                    "Wrong URL, no specific unavailability message. Retrying."
                )
                continue
            wait_selector = selector if selector else "body"
            logger.debug(
                f"On correct URL base. Waiting up to {element_timeout}s for selector: '{wait_selector}'"
            )
            try:
                WebDriverWait(driver, element_timeout).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, wait_selector))
                )
                logger.debug(
                    f"Navigation successful and element '{wait_selector}' found on:\n{url}"
                )
                return True
            except TimeoutException:
                current_url_on_timeout = "Unknown"
                try:
                    current_url_on_timeout = driver.current_url
                except Exception:
                    pass
                logger.warning(
                    f"Timeout waiting for selector '{wait_selector}' at {current_url_on_timeout} (URL base was correct initially)."
                )
                action, wait_time = _check_for_unavailability(
                    driver, unavailability_selectors
                )
                if action == "skip":
                    return False
                if action == "refresh":
                    time.sleep(wait_time)
                    continue
                logger.warning(
                    "Timeout on selector, no unavailability message. Retrying navigation."
                )
                continue
        except UnexpectedAlertPresentException as alert_e:
            alert_text = "N/A"
            try:
                alert_text = alert_e.alert_text
            except:
                pass
            logger.warning(
                f"Unexpected alert detected (Attempt {attempt}): {alert_text}"
            )
            try:
                driver.switch_to.alert.accept()
                logger.info("Accepted unexpected alert.")
            except Exception as accept_e:
                logger.error(f"Failed to accept unexpected alert: {accept_e}")
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
                    driver = session_manager.driver
                    if not driver:
                        return False
                    continue
                else:
                    logger.error("Session restart failed. Cannot complete navigation.")
                    return False
            else:
                logger.warning(
                    "WebDriverException occurred, session seems valid or no restart possible. Waiting before retry."
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
    logger.critical(
        f"Navigation to '{url}' failed permanently after {max_attempts} attempts."
    )
    try:
        logger.error(f"Final URL after failure: {driver.current_url}")
    except Exception:
        logger.error("Could not retrieve final URL after failure.")
    return False


# End of nav_to_page


def _check_for_unavailability(
    driver: WebDriver, selectors: Dict[str, Tuple[str, int]]
) -> Tuple[Optional[str], int]:
    """Checks if known 'page unavailable' messages are present."""
    # Step 1: Iterate through selectors
    for msg_selector, (action, wait_time) in selectors.items():
        # Step 2: Use selenium_utils helper 'is_elem_there'
        if is_elem_there(driver, By.CSS_SELECTOR, msg_selector, wait=0.5):
            logger.warning(
                f"Unavailability message found matching selector: '{msg_selector}'. Action: {action}, Wait: {wait_time}s"
            )
            return action, wait_time
    # Step 3: Return default if none found
    return None, 0


def main():
    """
    Standalone test function for utils.py.
    Runs a sequence of readiness and identifier checks on SessionManager,
    prints/logs all results, and ensures all errors are visible.
    V2: Simplified readiness checks, improved logging.
    """
    # --- Setup Logging ---
    from logging_config import setup_logging
    from config import config_instance
    import traceback

    db_file_path = config_instance.DATABASE_FILE
    log_filename_only = db_file_path.with_suffix(".log").name
    global logger
    logger = setup_logging(log_file=log_filename_only, log_level="DEBUG")

    session_manager: Optional[SessionManager] = None
    driver_instance = None
    test_success = True  # Assume success initially

    try:
        # --- PHASE 1: Driver/DB Setup ---
        logger.info("=== PHASE 1: Testing SessionManager.start_sess() ===\n")
        session_manager = SessionManager()
        start_ok = session_manager.start_sess(action_name="Utils Test - Phase 1")
        if not start_ok or not session_manager.driver_live:
            logger.error("SessionManager.start_sess() (Phase 1) FAILED. Aborting.")
            test_success = False
            return  # Cannot proceed without driver
        driver_instance = session_manager.driver
        if not driver_instance:
            logger.error(
                "Driver instance is None after successful Phase 1 report. Aborting."
            )
            test_success = False
            return  # Cannot proceed without driver
        logger.info("SessionManager.start_sess() (Phase 1) PASSED.\n")

        # --- PHASE 2: Session Readiness ---
        logger.info("=== PHASE 2: Testing SessionManager.ensure_session_ready() ===\n")
        ready_ok = session_manager.ensure_session_ready(
            action_name="Utils Test - Phase 2"
        )

        if not ready_ok:
            logger.error(
                "SessionManager.ensure_session_ready() (Phase 2) FAILED. Running diagnostics...\n"
            )
            test_success = False

            # --- PHASE 2c/2d Combined: Readiness Diagnostics (only run on failure) ---
            logger.info("=== PHASE 2 Diag: Running Readiness Sub-Checks ===\n")
            readiness_results = []
            diag_errors = []

            # 1. Login status
            login_stat_result = login_status(session_manager)
            login_stat_str = (
                "PASSED"
                if login_stat_result is True
                else f"FAILED ({login_stat_result})"
            )
            readiness_results.append(("Login status", login_stat_str))
            if login_stat_result is not True:
                diag_errors.append("Login status check failed")

            # 2. Essential cookies
            essential_cookies = ["ANCSESSIONID", "SecureATT"]
            cookies_ok = session_manager.get_cookies(essential_cookies, timeout=5)
            cookies_str = "PASSED" if cookies_ok else "FAILED"
            readiness_results.append(("Essential cookies", cookies_str))
            if not cookies_ok:
                diag_errors.append("Essential cookies missing")

            # 3. CSRF token (Fetch ONCE for diagnostics if needed)
            if not session_manager.csrf_token or len(session_manager.csrf_token) < 20:
                logger.debug("Diag: Fetching CSRF token as it was missing/invalid...")
                session_manager.csrf_token = session_manager.get_csrf()  # Store it

            csrf_valid = (
                session_manager.csrf_token and len(session_manager.csrf_token) >= 20
            )
            csrf_str = (
                "PASSED"
                if csrf_valid
                else f"FAILED (Value: {str(session_manager.csrf_token)[:10]}...)"
            )
            readiness_results.append(("CSRF token", csrf_str))
            if not csrf_valid:
                diag_errors.append("CSRF token invalid")

            # 4. Identifiers (Check instance attributes populated by ensure_session_ready)
            profile_id_ok = bool(session_manager.my_profile_id)
            uuid_ok = bool(session_manager.my_uuid)
            tree_id_ok = (
                bool(session_manager.my_tree_id) if config_instance.TREE_NAME else True
            )  # Only fail if TREE_NAME set and ID missing
            ids_str = (
                "PASSED" if (profile_id_ok and uuid_ok and tree_id_ok) else "FAILED"
            )
            readiness_results.append(("Identifiers", ids_str))
            if not profile_id_ok:
                diag_errors.append("my_profile_id missing")
            if not uuid_ok:
                diag_errors.append("my_uuid missing")
            if config_instance.TREE_NAME and not tree_id_ok:
                diag_errors.append("my_tree_id missing (for configured tree)")

            # 5. Tree owner name
            owner_ok = (
                bool(session_manager.tree_owner_name)
                if config_instance.TREE_NAME
                else True
            )  # Only check if tree expected
            owner_str = "PASSED" if owner_ok else "FAILED"
            readiness_results.append(("Tree owner name", owner_str))
            if config_instance.TREE_NAME and not owner_ok:
                diag_errors.append("tree_owner_name missing")

            # Log summary table
            table_header = (
                f"\n{'Readiness Check':<25} | {'Result':<20}\n{'-'*25}+{'-'*21}"
            )
            table_rows = [
                f"{name:<25} | {result:<20}" for name, result in readiness_results
            ]
            table = table_header + "\n" + "\n".join(table_rows)
            logger.info(f"Readiness Diagnostics Table:{table}\n")
            if diag_errors:
                logger.error(f"Diagnostic Errors Summary: {'; '.join(diag_errors)}\n")

        else:
            # Phase 2 Passed - Log identifiers briefly
            logger.info("SessionManager.ensure_session_ready() (Phase 2) PASSED.\n")
            logger.info("--- Key Identifiers Retrieved ---")
            logger.info(f"  Profile ID: {session_manager.my_profile_id}")
            logger.info(f"  UUID: {session_manager.my_uuid}")
            logger.info(f"  Tree ID: {session_manager.my_tree_id or 'N/A'}")
            logger.info(f"  Tree Owner: {session_manager.tree_owner_name or 'N/A'}")
            # Log stored CSRF token
            csrf_display = (
                f"{str(session_manager.csrf_token)[:10]}..."
                if session_manager.csrf_token
                else "None"
            )
            logger.info(f"  CSRF Token: {csrf_display}\n")

        # --- PHASE 3: Navigation Test (Only if driver exists) ---
        logger.info("=== PHASE 3: Testing Navigation (nav_to_page to BASE_URL) ===\n")
        if not driver_instance:
            logger.error(
                "Cannot test navigation, driver_instance is None. SKIPPING PHASE 3."
            )
            test_success = False  # Mark as failure if driver disappeared
        else:
            nav_ok = nav_to_page(
                driver=driver_instance,
                url=config_instance.BASE_URL,
                selector="body",  # Basic check
                session_manager=session_manager,
            )
            if nav_ok:
                logger.info("nav_to_page() to BASE_URL PASSED.")
                try:
                    current_url_after_nav = driver_instance.current_url
                    # Looser check for base URL start
                    if current_url_after_nav.startswith(
                        config_instance.BASE_URL.rstrip("/")
                    ):
                        logger.info(
                            f"Successfully landed on expected base URL: {current_url_after_nav}\n"
                        )
                    else:
                        logger.warning(
                            f"nav_to_page() to base URL landed on different URL: {current_url_after_nav}. Marking as navigation issue.\n"
                        )
                        # Treat unexpected URL as a failure even if element found
                        test_success = False
                        nav_ok = False  # Override for subsequent checks
                except Exception as e:
                    logger.warning(
                        f"Could not verify URL after nav_to_page: {e}. Proceeding cautiously.\n"
                    )
            else:
                logger.error("nav_to_page() to BASE_URL FAILED.\n")
                test_success = False

        # --- PHASE 4: API Request Test (Only if session seems ready) ---
        logger.info(
            "=== PHASE 4: Testing API Request (_api_req via CSRF endpoint) ===\n"
        )
        # Use session_ready flag which should be reliable now
        if not session_manager or not session_manager.session_ready:
            logger.warning(
                "Cannot test API Request, session not ready. SKIPPING PHASE 4."
            )
            # Don't necessarily mark as failure if Phase 2 already failed
        else:
            csrf_url = urljoin(
                config_instance.BASE_URL, "discoveryui-matches/parents/api/csrfToken"
            )
            # Use the CSRF token already fetched if available, otherwise _api_req will handle it
            csrf_test_response = _api_req(
                url=csrf_url,
                driver=driver_instance,  # Pass driver instance
                session_manager=session_manager,
                method="GET",
                use_csrf_token=False,  # CSRF endpoint doesn't need the token sent to it
                api_description="CSRF Token API Test",
                force_text_response=True,  # Get the raw token string
            )

            if (
                csrf_test_response
                and isinstance(csrf_test_response, str)
                and len(csrf_test_response) > 10  # Basic validity check
            ):
                logger.info("CSRF Token API call via _api_req PASSED.")
                logger.debug(
                    f"CSRF Token API test retrieved: {csrf_test_response[:40]}...\n"
                )
                # Optionally verify it matches the stored one if needed
                # if session_manager.csrf_token and csrf_test_response != session_manager.csrf_token:
                #    logger.warning("Fresh CSRF token from API test differs from stored token.")
            else:
                logger.error("CSRF Token API call via _api_req FAILED.")
                logger.debug(f"Response received: {csrf_test_response}\n")
                test_success = False

        # --- PHASE 5: Tab Management Test (Only if driver exists) ---
        logger.info("=== PHASE 5: Testing Tab Management (make_tab, close_tabs) ===\n")
        if (
            not driver_instance or not session_manager.is_sess_valid()
        ):  # Re-check validity
            logger.warning(
                "Cannot test Tab Management, driver invalid or None. SKIPPING PHASE 5."
            )
            if not test_success:  # If already failing, don't override
                pass
            elif driver_instance:  # Driver existed but became invalid
                test_success = False
        else:
            logger.info("Creating a new tab...")
            initial_handles_list = driver_instance.window_handles  # Get initial handles
            initial_handle = initial_handles_list[0]  # Assume first is the one to keep
            new_tab_handle = session_manager.make_tab()

            if new_tab_handle:
                logger.info(f"make_tab() PASSED. New handle: {new_tab_handle}")
                logger.info("Navigating new tab to example.com...")
                try:
                    driver_instance.switch_to.window(new_tab_handle)
                    nav_new_tab_ok = nav_to_page(
                        driver_instance,
                        "https://example.com",
                        selector="body",
                        session_manager=session_manager,
                    )
                    if nav_new_tab_ok:
                        logger.info("Navigation in new tab successful.")
                        logger.info("Closing extra tabs...")
                        # Ensure close_tabs is imported or called correctly
                        try:
                            from selenium_utils import close_tabs

                            # Call close_tabs without the unexpected keyword argument.
                            close_tabs(driver_instance)

                            handles_after_close = driver_instance.window_handles

                            if len(handles_after_close) == 1:
                                logger.info(
                                    "close_tabs() PASSED (correctly left one tab open)."
                                )
                                if handles_after_close[0] == initial_handle:
                                    logger.debug(
                                        f"Remaining tab handle '{handles_after_close[0]}' matches initial handle."
                                    )
                                else:
                                    logger.warning(
                                        f"Remaining tab handle '{handles_after_close[0]}' does NOT match initial handle '{initial_handle}'. Check close_tabs logic."
                                    )
                                if (
                                    driver_instance.current_window_handle
                                    != handles_after_close[0]
                                ):
                                    logger.debug(
                                        "Switching focus back to the single remaining tab."
                                    )
                                    driver_instance.switch_to.window(
                                        handles_after_close[0]
                                    )
                            else:
                                logger.error(
                                    f"close_tabs() FAILED (expected 1 tab, found {len(handles_after_close)}: {handles_after_close})."
                                )
                                test_success = False

                        except ImportError:
                            logger.error(
                                "Could not import 'close_tabs' from 'selenium_utils'. Tab test incomplete."
                            )
                            test_success = False
                        except (
                            TypeError
                        ) as te:  # Catch the specific error if it persists
                            logger.error(
                                f"TypeError calling close_tabs: {te}. Check selenium_utils.py definition.",
                                exc_info=True,
                            )
                            test_success = False
                        except Exception as close_tab_err:
                            logger.error(
                                f"Error during close_tabs execution: {close_tab_err}",
                                exc_info=True,
                            )
                            test_success = False
                    else:
                        logger.error("Navigation in new tab FAILED.")
                        test_success = False
                        try:
                            from selenium_utils import close_tabs

                            close_tabs(driver_instance)
                        except Exception:
                            logger.warning(
                                "Failed to cleanup tabs after navigation failure."
                            )
                except Exception as tab_e:
                    logger.error(
                        f"Error during tab management test steps: {tab_e}",
                        exc_info=True,
                    )
                    test_success = False
                    try:
                        from selenium_utils import close_tabs

                        current_handles_cleanup = driver_instance.window_handles
                        if len(current_handles_cleanup) > 1:
                            logger.warning("Attempting tab cleanup after error...")
                            close_tabs(driver_instance)
                    except Exception as cleanup_err:
                        logger.error(
                            f"Error during tab cleanup after error: {cleanup_err}"
                        )
            else:
                logger.error("make_tab() FAILED.")
                test_success = False

    except Exception as e:
        logger.critical(
            f"CRITICAL error during utils.py standalone test execution: {e}",
            exc_info=True,
        )
        test_success = False  # Mark as failed on critical error
    finally:
        if session_manager:
            logger.info("Closing session manager in finally block...")
            session_manager.close_sess()
        else:
            logger.info("No SessionManager instance to close.")

        print("")  # Add a newline for separation
        if test_success:
            logger.info("--- Utils.py standalone test run PASSED ---")
        else:
            logger.error("--- Utils.py standalone test run FAILED ---")
        # Optional: Exit with non-zero code on failure for CI/CD
        # if not test_success:
        #     sys.exit(1)


# End of main

if __name__ == "__main__":
    main()
# End of utils.py
