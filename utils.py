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
    TypeAlias,
    Union,
)
from urllib.parse import urljoin, urlparse, unquote, urlunparse


# --- Type Aliases ---
ApiResponseType: TypeAlias = Union[Dict[str, Any], List[Any], str, bytes, None]
DriverType: TypeAlias = Optional["WebDriver"]  # Forward reference
SessionManagerType: TypeAlias = Optional["SessionManager"]  # Forward reference
RequestsResponseTypeOptional: TypeAlias = Optional["RequestsResponse"]

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

# API Path Constants
API_PATH_CSRF_TOKEN = "discoveryui-matches/parents/api/csrfToken"
API_PATH_PROFILE_ID = "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid"
API_PATH_UUID = "api/uhome/secure/rest/header/dna"
API_PATH_HEADER_TREES = "api/uhome/secure/rest/header/trees"
API_PATH_TREE_OWNER_INFO = "api/uhome/secure/rest/user/tree-info"
API_PATH_PROFILE_DETAILS = "/app-api/express/v1/profiles/details"
API_PATH_SEND_MESSAGE_NEW = "app-api/express/v2/conversations/message"
API_PATH_SEND_MESSAGE_EXISTING = "app-api/express/v2/conversations/{conv_id}"

# Key Dictionary Keys
KEY_UCDMID = "ucdmid"
KEY_TEST_ID = "testId"
KEY_DATA = "data"
KEY_MENUITEMS = "menuitems"
KEY_URL = "url"
KEY_TEXT = "text"
KEY_OWNER = "owner"
KEY_DISPLAY_NAME = "displayName"
KEY_CONVERSATION_ID = "conversation_id"
KEY_MESSAGE = "message"
KEY_AUTHOR = "author"
KEY_FIRST_NAME = "FirstName"
KEY_LAST_LOGIN_DATE = "LastLoginDate"
KEY_IS_CONTACTABLE = "IsContactable"


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
    from requests.exceptions import HTTPError, RequestException, JSONDecodeError
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
    from database import (
        Base,
        ConversationLog,
        DnaMatch,
        FamilyTree,
        MessageType,
        Person,
    )
    from logging_config import logger, setup_logging
    from my_selectors import *

    # Import specific selenium utils needed internally
    from selenium_utils import (
        is_browser_open,
        is_elem_there,
        close_tabs,
    )  # Added imports
except ImportError as import_err:
    logging.warning(
        f"Optional dependency import failed in utils.py: {import_err}. Some features may not be available."
    )
    # Define dummy types if imports fail, to prevent NameErrors later
    if "WebDriver" not in locals():
        WebDriver = type(None)
    if "RequestsResponse" not in locals():
        RequestsResponse = type(None)
    if "SessionManager" not in locals():
        SessionManager = type(None)  # Forward ref handled later
    if "Person" not in locals():
        Person = type(None)
    if "Session" not in locals():
        Session = type(None)
    if "RequestsCookieJar" not in locals():
        RequestsCookieJar = type(None)


# ------------------------------------------------------------------------------------
# Helper functions (General Utilities)
# ------------------------------------------------------------------------------------


def parse_cookie(cookie_string: str) -> Dict[str, str]:
    """
    Parses a raw HTTP cookie string into a dictionary of key-value pairs.
    Handles empty keys and values.
    """
    cookies: Dict[str, str] = {}
    if not isinstance(cookie_string, str):
        logger.warning("parse_cookie received non-string input, returning empty dict.")
        return cookies

    parts = cookie_string.split(";")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)  # Split only on the first equals
            key = key.strip()
            value = value.strip()
            # Allow empty keys/values if that's the format
            cookies[key] = value
        else:
            # Handle parts without equals (e.g., flags like 'Secure') - ignore them for this simple dict
            logger.debug(f"Skipping cookie part without '=': '{part}'")
    return cookies


# End of parse_cookie


def ordinal_case(text: Union[str, int]) -> str:
    """
    Corrects ordinal suffixes (1st, 2nd, 3rd, 4th) to lowercase within a string,
    often used after applying title casing. Handles relationship terms simply.
    Accepts string or integer input for numbers.
    """
    if not text and text != 0:  # Handle empty string, None, etc. but allow 0
        return str(text) if text is not None else ""

    # Try converting to int first
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
        # If not an integer, assume it's a string to be title-cased
        if isinstance(text, str):
            words = text.title().split()
            lc_words = {"Of", "The", "A", "An", "In", "On", "At", "For", "To", "With"}
            for i, word in enumerate(words):
                # Keep particles lowercase unless they start the string
                if i > 0 and word in lc_words:
                    words[i] = word.lower()
            return " ".join(words)
        else:
            # If not int or str, just convert to string
            return str(text)


# End of ordinal_case


def format_name(name: Optional[str]) -> str:
    """
    Formats a person's name string to title case, preserving uppercase components
    (like initials or acronyms) and handling None/empty input gracefully.
    Also removes GEDCOM-style slashes around surnames anywhere in the string.
    Handles common name particles and prefixes like Mc/Mac/O'.
    """
    if not name or not isinstance(name, str):
        return "Valued Relative"

    # Handle purely numeric or symbolic names early
    if name.isdigit() or re.fullmatch(r"[^a-zA-Z]+", name):
        logger.debug(
            f"Formatting name: Input '{name}' appears non-alphabetic, returning as is."
        )
        return name.strip()  # Return stripped original

    try:
        # 1. Pre-processing: Clean spaces and GEDCOM slashes
        cleaned_name = name.strip()
        cleaned_name = re.sub(r"\s*/([^/]+)/\s*", r" \1 ", cleaned_name)  # Space-padded
        cleaned_name = re.sub(
            r"^[/\s]+|[/\s]+$", "", cleaned_name
        )  # Trim leading/trailing slashes/spaces
        cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()  # Collapse spaces

        # 2. Define particles and exceptions
        lowercase_particles = {
            "van",
            "von",
            "der",
            "den",
            "de",
            "di",
            "da",
            "do",
            "la",
            "le",
            "el",
        }
        # Handle multi-word particles by checking subsequent words
        # Example: "van der" should stay lowercase if not at start

        uppercase_exceptions = {
            "II",
            "III",
            "IV",
            "SR",
            "JR",
        }  # Keep these fully uppercase

        # 3. Process each part of the name
        parts = cleaned_name.split()
        formatted_parts = []
        i = 0
        while i < len(parts):
            part = parts[i]
            part_lower = part.lower()

            # Check for multi-word particles first
            is_multi_word_particle = False
            if i > 0 and part_lower in lowercase_particles:
                # Check if the *next* word is also a particle (e.g., "van", "der")
                if i + 1 < len(parts) and parts[i + 1].lower() in lowercase_particles:
                    # Assume it's a multi-word particle, keep both lowercase
                    formatted_parts.append(part_lower)
                    formatted_parts.append(parts[i + 1].lower())
                    i += 2  # Skip the next word as well
                    is_multi_word_particle = True
                    continue  # Move to the word after the multi-word particle
                # Check if it's the last word - might still be a single particle
                elif i == len(parts) - 1:
                    formatted_parts.append(part_lower)
                    i += 1
                    is_multi_word_particle = True  # Treat as handled
                    continue

            if is_multi_word_particle:
                continue  # Already handled above

            # Handle single-word particles (check after multi-word)
            if i > 0 and part_lower in lowercase_particles:
                formatted_parts.append(part_lower)
                i += 1
                continue

            # Handle fully uppercase exceptions
            if part.upper() in uppercase_exceptions:
                formatted_parts.append(part.upper())
                i += 1
                continue

            # Handle hyphenated parts (e.g., Smith-Jones, van-der-Beek)
            if "-" in part:
                hyphenated = []
                hp_parts = part.split("-")
                for hp_idx, hp in enumerate(hp_parts):
                    # Lowercase particle within hyphenated name? (e.g., Marie-van-something)
                    if hp_idx > 0 and hp.lower() in lowercase_particles:
                        hyphenated.append(hp.lower())
                    elif hp:  # Avoid empty strings from multiple hyphens
                        hyphenated.append(
                            hp.capitalize()
                        )  # Basic capitalize for hyphen parts
                formatted_parts.append(
                    "-".join(filter(None, hyphenated))
                )  # Filter empty strings
                i += 1
                continue

            # Handle apostrophes (e.g., O'Malley, d'Artagnan)
            if (
                "'" in part and len(part) > 1 and not part.endswith("'")
            ):  # Avoid possessives like "Smith's"
                apostrophe_parts = part.split("'")
                if (
                    len(apostrophe_parts) == 2
                    and apostrophe_parts[0]
                    and apostrophe_parts[1]
                ):
                    # Capitalize first part, capitalize letter after apostrophe
                    formatted_part = (
                        apostrophe_parts[0].capitalize()
                        + "'"
                        + apostrophe_parts[1].capitalize()
                    )
                    formatted_parts.append(formatted_part)
                else:  # Fallback for complex cases like O'Malley's
                    formatted_parts.append(part.capitalize())
                i += 1
                continue

            # Handle prefixes (Mc/Mac)
            if part_lower.startswith("mc") and len(part) > 2:
                formatted_parts.append("Mc" + part[2:].capitalize())
                i += 1
                continue
            if part_lower.startswith("mac") and len(part) > 3:
                if part_lower == "mac":  # Just the prefix itself
                    formatted_parts.append("Mac")
                else:
                    formatted_parts.append("Mac" + part[3:].capitalize())
                i += 1
                continue

            # Handle initials (J., P.) - should be uppercase
            if len(part) == 2 and part.endswith(".") and part[0].isalpha():
                formatted_parts.append(part[0].upper() + ".")
                i += 1
                continue
            if len(part) == 1 and part.isalpha():  # Single letter initial without dot
                formatted_parts.append(part.upper())
                i += 1
                continue

            # Default: Capitalize the word
            formatted_parts.append(part.capitalize())
            i += 1

        # 4. Join and final cleanup
        final_name = " ".join(formatted_parts)
        final_name = re.sub(r"\s+", " ", final_name).strip()  # Final space check

        return final_name if final_name else "Valued Relative"

    except Exception as e:
        logger.error(f"Error formatting name '{name}': {e}", exc_info=False)
        # Fallback to basic title case on error during complex logic
        try:
            # Use basic title() as a last resort fallback
            return name.title() if isinstance(name, str) else "Valued Relative"
        except AttributeError:
            return "Valued Relative"  # If input wasn't even a string originally


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
            # Use config_instance if available, otherwise use decorator args or fallback defaults
            cfg = config_instance if config_instance else None
            attempts = (
                MAX_RETRIES
                if MAX_RETRIES is not None
                else (getattr(cfg, "MAX_RETRIES", 3) if cfg else 3)
            )
            backoff = (
                BACKOFF_FACTOR
                if BACKOFF_FACTOR is not None
                else (getattr(cfg, "BACKOFF_FACTOR", 1.0) if cfg else 1.0)
            )
            max_delay = (
                MAX_DELAY
                if MAX_DELAY is not None
                else (getattr(cfg, "MAX_DELAY", 10.0) if cfg else 10.0)
            )

            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == attempts - 1:
                        logger.error(
                            f"Function '{func.__name__}' failed after {attempts} retries. Final Exception: {e}",
                            exc_info=False,
                        )
                        raise  # Re-raise the last exception
                    # Calculate sleep time with exponential backoff and jitter
                    sleep_time = min(backoff * (2**i), max_delay) + random.uniform(
                        0, 0.5
                    )
                    logger.warning(
                        f"Retry {i+1}/{attempts} for {func.__name__} after exception: {type(e).__name__}. Sleeping {sleep_time:.2f}s."
                    )
                    time.sleep(sleep_time)
            # Should not be reached if attempts > 0
            logger.error(
                f"Function '{func.__name__}' failed after all {attempts} retries (exited loop unexpectedly)."
            )
            # Raise a generic error or return None based on expected behavior
            raise RuntimeError(f"Function {func.__name__} failed after all retries.")

        return wrapper

    return decorator


# End of retry


def retry_api(
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    retry_on_exceptions: Tuple[Type[Exception], ...] = (
        requests.exceptions.RequestException,  # Base class for requests exceptions
        ConnectionError,
        TimeoutError,  # Added TimeoutError
    ),
    retry_on_status_codes: Optional[List[int]] = None,
):
    """Decorator factory for retrying API calls with exponential backoff, logging, etc."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Use config_instance if available for defaults
            cfg = config_instance if config_instance else None

            _max_retries = (
                max_retries
                if max_retries is not None
                else (getattr(cfg, "MAX_RETRIES", 3) if cfg else 3)
            )
            _initial_delay = (
                initial_delay
                if initial_delay is not None
                else (getattr(cfg, "INITIAL_DELAY", 0.5) if cfg else 0.5)
            )
            _backoff_factor = (
                backoff_factor
                if backoff_factor is not None
                else (getattr(cfg, "BACKOFF_FACTOR", 1.5) if cfg else 1.5)
            )
            _retry_codes_set = set(
                retry_on_status_codes
                if retry_on_status_codes is not None
                else (
                    getattr(cfg, "RETRY_STATUS_CODES", {429, 500, 502, 503, 504})
                    if cfg
                    else {429, 500, 502, 503, 504}
                )
            )
            _max_delay = getattr(cfg, "MAX_DELAY", 60.0) if cfg else 60.0

            retries = _max_retries
            delay = _initial_delay
            attempt = 0

            last_exception: Optional[Exception] = None
            last_response: RequestsResponseTypeOptional = None

            while retries > 0:
                attempt += 1
                try:
                    response = func(*args, **kwargs)
                    last_response = response  # Store last response regardless of status

                    # Check if response is a requests.Response object to get status_code
                    status_code: Optional[int] = None
                    if isinstance(response, requests.Response):
                        status_code = response.status_code

                    should_retry_status = False
                    if status_code is not None and status_code in _retry_codes_set:
                        should_retry_status = True
                        last_exception = requests.exceptions.HTTPError(
                            f"{status_code} Error", response=response
                        )

                    if should_retry_status:
                        retries -= 1
                        if retries <= 0:
                            logger.error(
                                f"API Call failed after {_max_retries} retries for '{func.__name__}' (Final Status {status_code})."
                            )
                            return response  # Return last response on final failure
                        sleep_time = min(
                            delay * (_backoff_factor ** (attempt - 1)), _max_delay
                        ) + random.uniform(0, 0.2)
                        sleep_time = max(0.1, sleep_time)  # Ensure minimum sleep
                        logger.warning(
                            f"API Call status {status_code} (Attempt {attempt}/{_max_retries}) for '{func.__name__}'. Retrying in {sleep_time:.2f}s..."
                        )
                        time.sleep(sleep_time)
                        delay *= _backoff_factor
                        continue
                    else:
                        # Success or non-retryable error code
                        return response

                except retry_on_exceptions as e:
                    last_exception = e
                    retries -= 1
                    if retries <= 0:
                        logger.error(
                            f"API Call failed after {_max_retries} retries for '{func.__name__}'. Final Exception: {type(e).__name__} - {e}",
                            exc_info=False,
                        )
                        raise e  # Re-raise the last exception
                    sleep_time = min(
                        delay * (_backoff_factor ** (attempt - 1)), _max_delay
                    ) + random.uniform(0, 0.2)
                    sleep_time = max(0.1, sleep_time)
                    logger.warning(
                        f"API Call exception '{type(e).__name__}' (Attempt {attempt}/{_max_retries}) for '{func.__name__}', retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    delay *= _backoff_factor
                    continue
                except (
                    Exception
                ) as e:  # Catch unexpected errors during the function call itself
                    logger.error(
                        f"Unexpected error during API call attempt {attempt} for '{func.__name__}': {e}",
                        exc_info=True,
                    )
                    raise e  # Re-raise unexpected errors immediately

            # Loop finished without returning (shouldn't happen if max_retries > 0)
            logger.error(
                f"Exited retry loop unexpectedly for '{func.__name__}'. Last status: {getattr(last_response, 'status_code', 'N/A')}, Last exception: {last_exception}"
            )
            if last_exception:
                raise last_exception  # Raise the last known exception
            else:
                # Return the last response if available, otherwise raise runtime error
                return (
                    last_response
                    if last_response is not None
                    else RuntimeError(
                        f"{func.__name__} failed after all retries without specific exception."
                    )
                )

        return wrapper

    return decorator


# End of retry_api


def ensure_browser_open(func: Callable) -> Callable:
    """Decorator to ensure browser session is valid before executing."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        session_manager_instance: SessionManagerType = None
        driver_instance: DriverType = None

        # Find SessionManager and WebDriver instance from args or kwargs
        if args:
            if isinstance(args[0], SessionManager):
                session_manager_instance = args[0]
                driver_instance = session_manager_instance.driver
            elif isinstance(args[0], WebDriver):  # If driver is passed directly
                driver_instance = args[0]
                # Try to find SessionManager elsewhere if needed (less common)
                if len(args) > 1 and isinstance(args[1], SessionManager):
                    session_manager_instance = args[1]
                elif "session_manager" in kwargs and isinstance(
                    kwargs["session_manager"], SessionManager
                ):
                    session_manager_instance = kwargs["session_manager"]

        if (
            not driver_instance
            and "driver" in kwargs
            and isinstance(kwargs["driver"], WebDriver)
        ):
            driver_instance = kwargs["driver"]
            # Look for SessionManager again if driver was in kwargs
            if "session_manager" in kwargs and isinstance(
                kwargs["session_manager"], SessionManager
            ):
                session_manager_instance = kwargs["session_manager"]
            elif args and isinstance(args[0], SessionManager):
                session_manager_instance = args[0]

        if not driver_instance:
            # Fallback: maybe session_manager is passed only in kwargs
            if "session_manager" in kwargs and isinstance(
                kwargs["session_manager"], SessionManager
            ):
                session_manager_instance = kwargs["session_manager"]
                driver_instance = session_manager_instance.driver

        # Validation
        if not driver_instance:
            # Try to infer from 'self' if it's a method of a class with a driver attribute
            if (
                args
                and hasattr(args[0], "driver")
                and isinstance(getattr(args[0], "driver", None), WebDriver)
            ):
                driver_instance = args[0].driver
            else:
                raise TypeError(
                    f"Function '{func.__name__}' decorated with @ensure_browser_open requires a WebDriver instance passed as first arg, 'driver' kwarg, or via SessionManager."
                )

        # Check browser status
        # Use the imported is_browser_open utility function
        if not is_browser_open(driver_instance):
            raise WebDriverException(
                f"Browser session invalid/closed when calling function '{func.__name__}'"
            )

        # Execute the original function
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
    """
    Implements a token bucket rate limiter with dynamic delay adjustments based on feedback.
    """

    def __init__(
        self,
        initial_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        backoff_factor: Optional[float] = None,
        decrease_factor: Optional[float] = None,
        token_capacity: Optional[float] = None,
        token_fill_rate: Optional[float] = None,
    ):
        cfg = config_instance  # Use alias for brevity

        self.initial_delay = (
            initial_delay
            if initial_delay is not None
            else (getattr(cfg, "INITIAL_DELAY", 0.5) if cfg else 0.5)
        )
        self.MAX_DELAY = (
            max_delay
            if max_delay is not None
            else (getattr(cfg, "MAX_DELAY", 60.0) if cfg else 60.0)
        )
        self.backoff_factor = (
            backoff_factor
            if backoff_factor is not None
            else (getattr(cfg, "BACKOFF_FACTOR", 1.8) if cfg else 1.8)
        )
        self.decrease_factor = (
            decrease_factor
            if decrease_factor is not None
            else (getattr(cfg, "DECREASE_FACTOR", 0.98) if cfg else 0.98)
        )

        self.current_delay = self.initial_delay
        self.last_throttled = False

        self.capacity = float(
            token_capacity
            if token_capacity is not None
            else (getattr(cfg, "TOKEN_BUCKET_CAPACITY", 10.0) if cfg else 10.0)
        )
        self.fill_rate = float(
            token_fill_rate
            if token_fill_rate is not None
            else (getattr(cfg, "TOKEN_BUCKET_FILL_RATE", 2.0) if cfg else 2.0)
        )

        if self.fill_rate <= 0:
            logger.warning(
                f"Token fill rate ({self.fill_rate}) must be positive. Setting to 1.0."
            )
            self.fill_rate = 1.0

        self.tokens = float(self.capacity)  # Start with a full bucket
        self.last_refill_time = time.monotonic()

        logger.debug(
            f"RateLimiter Init: Capacity={self.capacity:.1f}, FillRate={self.fill_rate:.1f}/s, InitialDelay={self.initial_delay:.2f}s, MaxDelay={self.MAX_DELAY:.1f}s, Backoff={self.backoff_factor:.2f}, Decrease={self.decrease_factor:.2f}"
        )

    # End of __init__

    def _refill_tokens(self):
        """Refills tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = max(0, now - self.last_refill_time)
        tokens_to_add = elapsed * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = now

    # End of _refill_tokens

    def wait(self) -> float:
        """Waits if necessary based on token availability and current delay settings."""
        self._refill_tokens()
        requested_at = time.monotonic()  # Track time before potential sleep
        sleep_duration = 0.0

        # Option 1: Token is available, apply base delay + jitter
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            jitter_factor = random.uniform(0.8, 1.2)
            base_sleep = self.current_delay
            sleep_duration = min(base_sleep * jitter_factor, self.MAX_DELAY)
            sleep_duration = max(0.01, sleep_duration)  # Ensure minimal sleep
            logger.debug(
                f"Token available ({self.tokens:.2f} left). Applying base delay: {sleep_duration:.3f}s (CurrentDelay: {self.current_delay:.2f}s)"
            )
        # Option 2: Token bucket is empty, wait for a token to generate
        else:
            wait_needed = (1.0 - self.tokens) / self.fill_rate
            jitter_amount = random.uniform(
                0.0, 0.2
            )  # Smaller jitter when waiting for token
            sleep_duration = wait_needed + jitter_amount
            sleep_duration = min(sleep_duration, self.MAX_DELAY)  # Cap wait time
            sleep_duration = max(0.01, sleep_duration)  # Ensure minimal sleep
            logger.debug(
                f"Token bucket empty ({self.tokens:.2f}). Waiting for token: {sleep_duration:.3f}s"
            )

        # Perform the sleep if needed
        if sleep_duration > 0:
            time.sleep(sleep_duration)

        # Refill again after sleep to account for time passed during sleep,
        # and potentially consume token if refill happened exactly at request time
        self._refill_tokens()
        # If no time passed between the initial request and the post-sleep refill
        # (unlikely but possible if sleep was very short or zero),
        # and we had enough tokens after refill, consume one.
        # This prevents double consumption if refill already happened during sleep.
        if requested_at == self.last_refill_time:
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                logger.debug(
                    f"Consumed token immediately after waiting (post-refill). Tokens left: {self.tokens:.2f}"
                )
            else:
                # This case should be rare: waited but still no token after refill.
                logger.warning(
                    f"Waited for token, but still < 1 ({self.tokens:.2f}) after refill. Consuming fraction."
                )
                self.tokens = 0.0  # Drain remaining fraction

        return sleep_duration

    # End of wait

    def reset_delay(self):
        """Resets the dynamic delay component to the initial value."""
        if self.current_delay != self.initial_delay:
            logger.info(
                f"Rate limiter base delay reset from {self.current_delay:.2f}s to initial: {self.initial_delay:.2f}s"
            )
            self.current_delay = self.initial_delay
        self.last_throttled = False

    # End of reset_delay

    def decrease_delay(self):
        """Decreases the dynamic delay component towards the initial value."""
        if not self.last_throttled and self.current_delay > self.initial_delay:
            previous_delay = self.current_delay
            self.current_delay = max(
                self.current_delay * self.decrease_factor, self.initial_delay
            )
            # Log only if the change is significant
            if abs(previous_delay - self.current_delay) > 0.01:
                logger.debug(
                    f"Decreased base delay component to {self.current_delay:.2f}s"
                )
        self.last_throttled = False

    # End of decrease_delay

    def increase_delay(self):
        """Increases the dynamic delay component due to throttling feedback."""
        previous_delay = self.current_delay
        self.current_delay = min(
            self.current_delay * self.backoff_factor, self.MAX_DELAY
        )
        if (
            abs(previous_delay - self.current_delay) > 0.01
        ):  # Log only if changed significantly
            logger.info(
                f"Rate limit feedback received. Increased base delay from {previous_delay:.2f}s to {self.current_delay:.2f}s"
            )
        else:
            logger.debug(
                f"Rate limit feedback received, but delay already at max ({self.MAX_DELAY:.2f}s) or increase too small."
            )
        self.last_throttled = True

    # End of increase_delay

    def is_throttled(self) -> bool:
        """Returns whether the last feedback indicated throttling."""
        return self.last_throttled

    # End of is_throttled


# End of DynamicRateLimiter class


# ------------------------------
# Session Management (Remains in utils.py)
# ------------------------------
class SessionManager:
    """
    Manages WebDriver and requests sessions, database connections,
    and essential user identifiers (CSRF token, profile ID, UUID, tree ID).
    Includes methods for session startup, validation, readiness checks, and cleanup.
    """

    def __init__(self):
        """Initializes SessionManager, loading config and setting initial state."""
        self.driver: DriverType = None
        self.driver_live: bool = False
        self.session_ready: bool = False

        # Load config safely
        if not config_instance or not selenium_config:
            logger.critical(
                "Configuration instances (config_instance, selenium_config) not loaded. SessionManager cannot initialize."
            )
            # Raise an error to prevent instantiation with missing config
            raise RuntimeError(
                "Configuration not loaded. Cannot create SessionManager."
            )

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

        self.engine = None  # SQLAlchemy engine
        self.Session: Optional[sessionmaker] = None  # SQLAlchemy session factory
        self._db_init_attempted: bool = False

        self.cache_dir: Path = config_instance.CACHE_DIR

        # Session identifiers
        self.csrf_token: Optional[str] = None
        self.my_profile_id: Optional[str] = None
        self.my_uuid: Optional[str] = None
        self.my_tree_id: Optional[str] = None
        self.tree_owner_name: Optional[str] = None

        self.session_start_time: Optional[float] = None
        self._profile_id_logged: bool = False
        self._uuid_logged: bool = False
        self._tree_id_logged: bool = False
        self._owner_logged: bool = False

        # Shared requests session setup
        self._requests_session = requests.Session()
        retry_strategy = Retry(
            total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(
            pool_connections=20, pool_maxsize=50, max_retries=retry_strategy
        )
        self._requests_session.mount("http://", adapter)
        self._requests_session.mount("https://", adapter)
        logger.debug("Initialized shared requests.Session with HTTPAdapter.")

        # Shared Cloudscraper instance
        self.scraper: Optional[cloudscraper.CloudScraper] = None
        try:
            self.scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True},
                delay=10,  # Initial delay for Cloudflare checks
            )
            scraper_retry = Retry(
                total=3,
                backoff_factor=0.8,
                status_forcelist=[403, 429, 500, 502, 503, 504],
                allowed_methods=[
                    "HEAD",
                    "GET",
                    "OPTIONS",
                    "POST",
                ],  # Allow retries on POST for relevant APIs
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

        # Rate Limiter
        self.dynamic_rate_limiter: DynamicRateLimiter = DynamicRateLimiter()

        # JS Error Tracking
        self.last_js_error_check: datetime = datetime.now(
            timezone.utc
        )  # Use timezone-aware datetime

        logger.debug(f"SessionManager instance created: ID={id(self)}\n")

    # End of __init__

    def start_sess(self, action_name: Optional[str] = None) -> bool:
        """
        Starts Phase 1: Initializes the WebDriver instance and navigates to the base URL.
        Resets session identifiers. Initializes DB connection if needed.

        :param action_name: Optional name for logging context.
        :return: True if driver started and initial navigation succeeded, False otherwise.
        """
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

        # Initialize DB engine/session factory if not already done
        if not self.engine or not self.Session:
            try:
                self._initialize_db_engine_and_session()
            except Exception as db_init_e:
                logger.critical(
                    f"DB Initialization failed during Phase 1 start: {db_init_e}"
                )
                return False

        # Ensure requests session exists
        if not hasattr(self, "_requests_session") or not isinstance(
            self._requests_session, requests.Session
        ):
            logger.warning(
                "Re-initializing missing shared requests.Session in start_sess."
            )
            self._requests_session = requests.Session()
            # Consider re-adding adapter here if needed

        logger.debug("Initializing WebDriver instance (using init_webdvr)...")
        try:
            self.driver = (
                init_webdvr()
            )  # Assumes init_webdvr handles retries internally
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
                selector="body",  # Wait for basic body tag
                session_manager=self,  # Pass self for context
            )
            if not base_url_nav_ok:
                logger.error("Failed to navigate to Base URL after WebDriver init.")
                self.close_sess()  # Clean up partially started session
                return False
            logger.debug("Initial navigation to Base URL successful.")

            # Mark driver as live and record start time
            self.driver_live = True
            self.session_start_time = time.time()
            self.last_js_error_check = datetime.now(timezone.utc)

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
        """Ensures the WebDriver is initialized and live. Starts it if necessary."""
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
        Ensures the session is fully ready for actions by:
        1. Ensuring the driver is live (Phase 1).
        2. Performing readiness checks (login, cookies, CSRF) (Phase 2).
        3. Retrieving essential identifiers (profile ID, UUID, tree ID, owner name).

        Updates self.session_ready state.

        :param action_name: Optional name for logging context.
        :return: True if the session is fully ready, False otherwise.
        """
        logger.debug(
            f"TRACE: Entered ensure_session_ready (Action: {action_name or 'Default'})"
        )

        # Step 1: Ensure driver is live
        if not self.ensure_driver_live(action_name=f"{action_name} - Ensure Driver"):
            logger.error(
                f"Cannot ensure session ready for '{action_name}': Driver start failed."
            )
            self.session_ready = False  # Ensure state is correct
            logger.debug("TRACE: ensure_session_ready - ensure_driver_live failed")
            return False

        # Step 2: Perform readiness checks
        ready_checks_ok = False
        try:
            logger.debug("TRACE: Calling _perform_readiness_checks")
            ready_checks_ok = self._perform_readiness_checks(
                action_name=f"{action_name} - Readiness Checks"
            )
            logger.debug(
                f"TRACE: _perform_readiness_checks returned: {ready_checks_ok}"
            )
        except Exception as e:
            logger.critical(
                f"Exception in _perform_readiness_checks: {e}", exc_info=True
            )
            self.session_ready = False  # Mark not ready on exception
            return False
        except BaseException as be:
            logger.critical(
                f"BaseException in _perform_readiness_checks: {type(be).__name__}: {be}",
                exc_info=True,
            )
            self.session_ready = False  # Mark not ready
            raise  # Re-raise critical exceptions like KeyboardInterrupt

        # Step 3: Always attempt to retrieve identifiers if driver is live
        # This populates them even if readiness checks had minor (non-fatal) issues.
        identifiers_ok = self._retrieve_identifiers()
        owner_ok = (
            self._retrieve_tree_owner() if config_instance.TREE_NAME else True
        )  # Only retrieve owner if tree is configured

        # Log results post-fetch attempt
        logger.debug(
            f"Identifiers after fetch: profile_id={self.my_profile_id}, uuid={self.my_uuid}, tree_id={self.my_tree_id}"
        )
        logger.debug(f"Tree owner name after fetch: {self.tree_owner_name}")
        if not identifiers_ok:
            logger.warning("One or more essential identifiers could not be retrieved.")
        if config_instance.TREE_NAME and not owner_ok:
            logger.warning("Tree owner name could not be retrieved (Tree configured).")

        # Step 4: Determine final readiness state
        # Session is ready ONLY IF readiness checks passed AND essential IDs were retrieved
        # (owner name retrieval is optional unless tree is configured)
        self.session_ready = ready_checks_ok and identifiers_ok and owner_ok

        logger.debug(
            f"TRACE: Set self.session_ready to: {self.session_ready} (ChecksOK: {ready_checks_ok}, IDsOK: {identifiers_ok}, OwnerOK: {owner_ok})"
        )
        logger.debug(
            f"TRACE: Exiting ensure_session_ready (returning {self.session_ready})"
        )
        return self.session_ready

    # End of ensure_session_ready

    def _perform_readiness_checks(self, action_name: Optional[str] = None) -> bool:
        """
        Perform a sequence of checks (login, cookies, CSRF) to ensure the session is ready.
        Attempts retries and remedial actions (cookie import, automated login).

        :param action_name: An optional name for the action that initiated the checks.
        :return: True if all essential checks passed, False otherwise.
        """
        max_attempts = 2
        attempt = 0
        last_check_error: Optional[str] = None

        while attempt < max_attempts:
            attempt += 1
            logger.debug(
                f"Performing readiness checks (Attempt {attempt}/{max_attempts}, Action: {action_name})..."
            )
            last_check_error = None  # Reset error for this attempt

            try:
                # Initial checks
                if not self.driver_live or not self.driver:
                    logger.error("Cannot perform readiness checks: Driver not live.")
                    last_check_error = "Driver not live"
                    # If driver isn't live, retrying readiness checks won't help
                    return False

                # --- Check 1: Login Status & Remediation ---
                login_status_result = login_status(self)
                if login_status_result is False:
                    logger.info("Not logged in. Attempting remedial actions...")
                    cookies_backup_path = self._get_cookie_backup_path()

                    # Try cookie import first if backup exists
                    if cookies_backup_path and os.path.exists(cookies_backup_path):
                        logger.info("Cookie backup found. Attempting import...")
                        try:
                            self.driver.get(
                                config_instance.BASE_URL
                            )  # Go to base URL for stability
                            import_cookies(self.driver, str(cookies_backup_path))
                            self.driver.refresh()
                            time.sleep(1)  # Allow refresh to settle
                            logger.info(
                                "Imported cookies from backup and refreshed page."
                            )
                            # Re-check status after import
                            if login_status(self) is True:
                                logger.info("Login restored via cookie import.")
                                login_status_result = True  # Update status
                            else:
                                logger.warning(
                                    "Cookie import attempted, but login status still False."
                                )
                        except (
                            WebDriverException,
                            OSError,
                            ValueError,
                            TypeError,
                        ) as import_err:
                            logger.warning(f"Cookie import failed: {import_err}")

                    # If still not logged in, attempt automated login
                    if login_status_result is False:
                        logger.info("Attempting login via automation...")
                        login_result_str = log_in(self)
                        if login_result_str != "LOGIN_SUCCEEDED":
                            logger.error(
                                f"Login attempt failed ({login_result_str}). Readiness check failed on attempt {attempt}."
                            )
                            last_check_error = (
                                f"Login attempt failed: {login_result_str}"
                            )
                            continue  # Go to next attempt if retries left

                        # Login succeeded, export cookies and verify status again
                        logger.info("Login successful via automation.")
                        if cookies_backup_path:
                            try:
                                export_cookies(self.driver, str(cookies_backup_path))
                                logger.info(
                                    f"Cookies exported to {cookies_backup_path}"
                                )
                            except (WebDriverException, OSError, IOError) as export_err:
                                logger.warning(
                                    f"Failed to export cookies after login: {export_err}"
                                )

                        # Final verification after automated login
                        login_status_result = login_status(self)
                        if login_status_result is not True:
                            logger.error(
                                "Login status verification failed even after successful login report."
                            )
                            last_check_error = (
                                "Login status False after reported success"
                            )
                            continue  # Go to next attempt
                        logger.debug(
                            "Login status re-verified successfully after automation."
                        )

                elif login_status_result is None:
                    logger.error(
                        "Critical error checking login status (returned None)."
                    )
                    last_check_error = "login_status returned None"
                    # This is likely unrecoverable, don't retry
                    return False

                # --- Check 2: URL Handling (if logged in) ---
                if login_status_result is True:
                    logger.debug("Checking/Handling current URL...")
                    if not self._check_and_handle_url():
                        logger.error("URL check/handling failed.")
                        last_check_error = "URL check/handling failed"
                        # This might be recoverable on retry if it was a temporary page
                        continue
                    logger.debug("URL check/handling OK.")

                # --- Check 3: Essential Cookies ---
                logger.debug("Verifying essential cookies...")
                essential_cookies = ["ANCSESSIONID", "SecureATT"]
                if not self.get_cookies(essential_cookies, timeout=5):
                    logger.error(f"Essential cookies {essential_cookies} not found.")
                    last_check_error = f"Essential cookies missing: {essential_cookies}"
                    # Cookie issues might resolve after login/refresh, so retry
                    continue
                logger.debug("Essential cookies OK.")

                # --- Check 4: Sync Cookies to Requests ---
                logger.debug("Syncing cookies to requests session...")
                try:
                    self._sync_cookies()
                    logger.debug("Cookies synced.")
                except Exception as sync_e:  # Catch broad errors during sync
                    logger.error(f"Cookie sync failed: {sync_e}", exc_info=True)
                    last_check_error = f"Cookie sync failed: {sync_e}"
                    continue  # May resolve on retry

                # --- Check 5: Ensure CSRF Token ---
                logger.debug("Ensuring CSRF token...")
                if not self.csrf_token or len(self.csrf_token) < 20:
                    self.csrf_token = self.get_csrf()  # Fetch if missing/invalid
                if (
                    not self.csrf_token or len(self.csrf_token) < 20
                ):  # Check again after fetch attempt
                    logger.error("Failed to retrieve/verify valid CSRF token.")
                    last_check_error = "CSRF token missing/invalid"
                    continue  # Retry might help if it was a transient API issue
                logger.debug("CSRF token OK.")

                # --- All Checks Passed for this attempt ---
                logger.info(f"Readiness checks PASSED on attempt {attempt}.")
                return True

            except WebDriverException as wd_exc:
                logger.error(
                    f"WebDriverException during readiness check attempt {attempt}: {wd_exc}",
                    exc_info=False,
                )
                last_check_error = f"WebDriverException: {wd_exc}"
                if not self.is_sess_valid():
                    logger.error(
                        "Session invalid during readiness check. Aborting checks."
                    )
                    self.driver_live = False
                    self.session_ready = False
                    return False  # Don't retry if session is dead
                # If session still valid but error occurred, wait before retry
                if attempt >= max_attempts:
                    logger.error(
                        "Readiness checks failed after final attempt (WebDriverException)."
                    )
                    return False
            except Exception as e:
                logger.error(
                    f"Unexpected error during readiness check attempt {attempt}: {e}",
                    exc_info=True,
                )
                last_check_error = f"Unexpected Error: {e}"
                if attempt >= max_attempts:
                    logger.error(
                        "Readiness checks failed after final attempt (Exception)."
                    )
                    return False

            # If loop continues (check failed but retries remain)
            logger.info(
                f"Waiting {self.selenium_config.CHROME_RETRY_DELAY}s before next readiness attempt (Last Error: {last_check_error})..."
            )
            time.sleep(self.selenium_config.CHROME_RETRY_DELAY)
            # Continue to next attempt in while loop

        # Loop finished without returning True
        logger.error(
            f"All {max_attempts} readiness check attempts failed. Last Error: {last_check_error}"
        )
        return False

    # End of _perform_readiness_checks

    def _get_cookie_backup_path(self) -> Optional[Path]:
        """Constructs the path for the cookie backup file."""
        if not self.chrome_user_data_dir:
            logger.warning(
                "Cannot get cookie backup path: Chrome user data directory not set."
            )
            return None
        # Use a consistent filename within the user data directory
        return self.chrome_user_data_dir / "ancestry_cookies.json"

    # End of _get_cookie_backup_path

    def _initialize_db_engine_and_session(self):
        """Initializes the SQLAlchemy engine and session factory if not already done."""
        if self.engine and self.Session:
            logger.debug(
                f"DB Engine/Session already initialized for SM ID={id(self)}. Skipping."
            )
            return

        logger.debug(
            f"SessionManager ID={id(self)} initializing SQLAlchemy Engine/Session..."
        )
        self._db_init_attempted = True  # Mark that we tried

        # Dispose existing engine if somehow present without Session factory
        if self.engine:
            logger.warning(
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

            # Determine pool size safely from config or default
            pool_size = getattr(config_instance, "DB_POOL_SIZE", 10)
            if not isinstance(pool_size, int) or pool_size <= 0:
                logger.warning(f"Invalid DB_POOL_SIZE '{pool_size}'. Using default 10.")
                pool_size = 10
            pool_size = min(pool_size, 100)  # Cap pool size

            max_overflow = max(5, int(pool_size * 0.2))  # At least 5 overflow
            pool_timeout = 30  # Standard timeout
            pool_class = sqlalchemy_pool.QueuePool  # Default pool class

            logger.debug(
                f"DB Pool Config: Size={pool_size}, MaxOverflow={max_overflow}, Timeout={pool_timeout}"
            )

            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=False,  # Disable SQL query logging by default
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                poolclass=pool_class,
                connect_args={
                    "check_same_thread": False
                },  # Required for SQLite multi-threading
            )
            logger.debug(
                f"Created NEW SQLAlchemy engine: ID={id(self.engine)} for SM ID={id(self)}"
            )

            # Add PRAGMA settings listener
            @event.listens_for(self.engine, "connect")
            def enable_sqlite_settings(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA journal_mode=WAL;")
                    cursor.execute("PRAGMA foreign_keys=ON;")
                    cursor.execute(
                        "PRAGMA synchronous=NORMAL;"
                    )  # Balance safety and speed
                    logger.debug(
                        "SQLite PRAGMA settings applied (WAL, Foreign Keys, Sync Normal)."
                    )
                except sqlite3.Error as pragma_e:  # Catch specific sqlite errors
                    logger.error(f"Failed setting PRAGMA: {pragma_e}")
                finally:
                    cursor.close()

            # Create session factory
            self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
            logger.debug(f"Created Session factory for Engine ID={id(self.engine)}")

            # Ensure tables are created
            try:
                Base.metadata.create_all(self.engine)
                logger.debug("DB tables checked/created successfully.")
            except SQLAlchemyError as table_create_e:
                logger.error(
                    f"Error creating DB tables: {table_create_e}", exc_info=True
                )
                raise  # Re-raise to signal failure

        except SQLAlchemyError as sql_e:  # Catch specific SQLAlchemy errors
            logger.critical(f"FAILED to initialize SQLAlchemy: {sql_e}", exc_info=True)
            if self.engine:
                self.engine.dispose()
            self.engine = None
            self.Session = None
            self._db_init_attempted = False  # Reset attempt flag on failure
            raise sql_e  # Re-raise
        except Exception as e:  # Catch other unexpected errors
            logger.critical(
                f"UNEXPECTED error initializing SQLAlchemy: {e}", exc_info=True
            )
            if self.engine:
                self.engine.dispose()
            self.engine = None
            self.Session = None
            self._db_init_attempted = False
            raise e

    # End of _initialize_db_engine_and_session

    def _check_and_handle_url(self) -> bool:
        """Checks if the current browser URL is suitable. Navigates to base URL if not."""
        if not self.driver:
            logger.error("Driver attribute is None. Cannot check URL.")
            return False
        try:
            current_url = self.driver.current_url
            logger.debug(f"Current URL for check: {current_url}")
        except WebDriverException as e:
            logger.error(f"Error getting current URL: {e}. Session might be dead.")
            if not self.is_sess_valid():
                logger.warning("Session seems invalid during URL check.")
            return False
        except (
            AttributeError
        ):  # Should not happen if self.driver check passed, but good practice
            logger.error("Driver attribute became None unexpectedly. Cannot check URL.")
            return False

        # Normalize URLs for comparison (scheme, netloc, path, ignoring query/fragment)
        base_url_parsed = urlparse(config_instance.BASE_URL)
        base_url_norm = urlunparse(
            (
                base_url_parsed.scheme,
                base_url_parsed.netloc,
                base_url_parsed.path.rstrip("/"),
                "",
                "",
                "",
            )
        ).rstrip("/")

        signin_url_base = urljoin(config_instance.BASE_URL, "account/signin").rstrip(
            "/"
        )
        logout_url_base = urljoin(config_instance.BASE_URL, "c/logout").rstrip("/")
        mfa_url_base = urljoin(config_instance.BASE_URL, "account/signin/mfa/").rstrip(
            "/"
        )
        # Normalize disallowed URLs as well
        disallowed_starts = (
            urlunparse(urlparse(signin_url_base)[:3] + ("", "", "")).rstrip("/"),
            urlunparse(urlparse(logout_url_base)[:3] + ("", "", "")).rstrip("/"),
            urlunparse(urlparse(mfa_url_base)[:3] + ("", "", "")).rstrip("/"),
        )

        current_url_parsed = urlparse(current_url)
        current_url_norm = urlunparse(
            (
                current_url_parsed.scheme,
                current_url_parsed.netloc,
                current_url_parsed.path.rstrip("/"),
                "",
                "",
                "",
            )
        ).rstrip("/")

        is_api_path = "/api/" in current_url  # Check raw URL for API path segment

        needs_navigation = False
        reason = ""

        # Check 1: Is it outside the base domain?
        if current_url_parsed.netloc != base_url_parsed.netloc:
            needs_navigation = True
            reason = f"URL domain ({current_url_parsed.netloc}) differs from base ({base_url_parsed.netloc})."
        # Check 2: Is it one of the disallowed pages (login, logout, mfa)?
        elif any(current_url_norm == path for path in disallowed_starts):
            needs_navigation = True
            reason = f"URL matches disallowed path ({current_url_norm})."
        # Check 3: Does it look like an API call page?
        elif is_api_path:
            needs_navigation = True
            reason = "URL contains '/api/'."

        if needs_navigation:
            logger.info(
                f"Current URL unsuitable ({reason}). Navigating to base URL: {config_instance.BASE_URL}"
            )
            if not nav_to_page(
                self.driver,
                config_instance.BASE_URL,
                selector="body",
                session_manager=self,
            ):
                logger.error("Failed to navigate to base URL during check.")
                return False
            logger.debug("Navigation to base URL successful.")
        else:
            logger.debug("Current URL is suitable, no extra navigation needed.")

        return True

    # End of _check_and_handle_url

    def _retrieve_identifiers(self) -> bool:
        """Fetches and stores profile ID, UUID, and Tree ID if configured."""
        logger.debug("TRACE: Entered _retrieve_identifiers")
        if not self.is_sess_valid():
            logger.error("_retrieve_identifiers: Session is invalid.")
            return False

        all_ok = True

        # Profile ID (ucdmid)
        if not self.my_profile_id:
            logger.debug("Retrieving profile ID (ucdmid)...")
            self.my_profile_id = (
                self.get_my_profileId()
            )  # Assumes get_my_profileId handles retries
            if not self.my_profile_id:
                logger.error("Failed to retrieve profile ID (ucdmid).")
                all_ok = False
            elif not self._profile_id_logged:
                logger.info(f"My profile id: {self.my_profile_id}")
                self._profile_id_logged = True
        elif not self._profile_id_logged:  # Log if already set but not logged
            logger.info(f"My profile id: {self.my_profile_id}")
            self._profile_id_logged = True

        # UUID (testId)
        if not self.my_uuid:
            logger.debug("Retrieving UUID (testId)...")
            self.my_uuid = self.get_my_uuid()  # Assumes get_my_uuid handles retries
            if not self.my_uuid:
                logger.error("Failed to retrieve UUID (testId).")
                all_ok = False
            elif not self._uuid_logged:
                logger.info(f"My uuid: {self.my_uuid}")
                self._uuid_logged = True
        elif not self._uuid_logged:  # Log if already set but not logged
            logger.info(f"My uuid: {self.my_uuid}")
            self._uuid_logged = True

        # Tree ID (Optional based on config)
        if config_instance.TREE_NAME and not self.my_tree_id:
            logger.debug(
                f"Retrieving tree ID for tree name: '{config_instance.TREE_NAME}'..."
            )
            self.my_tree_id = (
                self.get_my_tree_id()
            )  # Assumes get_my_tree_id handles retries
            if not self.my_tree_id:
                logger.error(
                    f"TREE_NAME '{config_instance.TREE_NAME}' configured, but failed to get corresponding tree ID."
                )
                all_ok = False  # Fail if tree configured but ID not found
            elif not self._tree_id_logged:
                logger.info(f"My tree id: {self.my_tree_id}")
                self._tree_id_logged = True
        elif (
            self.my_tree_id and not self._tree_id_logged
        ):  # Log if already set but not logged
            logger.info(f"My tree id: {self.my_tree_id}")
            self._tree_id_logged = True
        elif not config_instance.TREE_NAME:
            logger.debug("No TREE_NAME configured, skipping tree ID retrieval.")

        logger.debug(
            f"TRACE: Exiting _retrieve_identifiers (Overall success: {all_ok})"
        )
        return all_ok

    # End of _retrieve_identifiers

    def _retrieve_tree_owner(self) -> bool:
        """Fetches and stores the tree owner name if Tree ID is available."""
        logger.debug("TRACE: Entered _retrieve_tree_owner")
        if not self.is_sess_valid():
            logger.error("_retrieve_tree_owner: Session is invalid.")
            return False

        # Only proceed if Tree ID is set (either from config or previous retrieval)
        if not self.my_tree_id:
            logger.error("Cannot retrieve tree owner name: my_tree_id is not set.")
            logger.debug("TRACE: Exiting _retrieve_tree_owner (no tree id)")
            return False  # Cannot proceed without tree ID

        # If already retrieved and logged, just return True
        if self.tree_owner_name and self._owner_logged:
            logger.debug("TRACE: Exiting _retrieve_tree_owner (already set and logged)")
            return True

        logger.debug("Retrieving tree owner name...")
        self.tree_owner_name = self.get_tree_owner(
            self.my_tree_id
        )  # Assumes get_tree_owner handles retries

        if not self.tree_owner_name:
            logger.error("Failed to retrieve tree owner name.")
            logger.debug("TRACE: Exiting _retrieve_tree_owner (failed)")
            return False

        # Log only once
        if not self._owner_logged:
            logger.info(f"Tree owner name: {self.tree_owner_name}")
            self._owner_logged = True

        logger.debug("TRACE: Exiting _retrieve_tree_owner (success)")
        return True

    # End of _retrieve_tree_owner

    @retry_api()  # Use decorator for retries
    def get_csrf(self) -> Optional[str]:
        """Fetches a fresh CSRF token from the API."""
        if not self.is_sess_valid():
            logger.error("get_csrf: Session invalid.")
            return None

        csrf_token_url = urljoin(config_instance.BASE_URL, API_PATH_CSRF_TOKEN)
        logger.debug(f"Attempting to fetch fresh CSRF token from: {csrf_token_url}")

        # Check essential cookies before API call
        essential_cookies = ["ANCSESSIONID", "SecureATT"]
        if not self.get_cookies(essential_cookies, timeout=10):  # Short timeout ok here
            logger.warning(
                f"Essential cookies {essential_cookies} NOT found before CSRF token API call. API might fail."
            )

        # Make the API request using the shared helper
        response_data = _api_req(
            url=csrf_token_url,
            driver=self.driver,
            session_manager=self,
            method="GET",
            use_csrf_token=False,  # Don't send a token to get a token
            api_description="CSRF Token API",
            force_text_response=True,  # Expect plain text
        )

        # Process the response
        if response_data and isinstance(response_data, str):
            csrf_token_val = response_data.strip()
            if csrf_token_val and len(csrf_token_val) > 20:  # Basic validity check
                logger.debug(
                    f"CSRF token successfully retrieved (Length: {len(csrf_token_val)})."
                )
                self.csrf_token = csrf_token_val  # Store the retrieved token
                return csrf_token_val
            else:
                logger.error(
                    f"CSRF token API returned empty or invalid string: '{csrf_token_val}'"
                )
                return None
        elif response_data is None:
            logger.warning(
                "Failed to get CSRF token response via _api_req (returned None/error)."
            )
            return None
        else:
            # Handle unexpected response types (e.g., requests.Response object on non-2xx)
            status = getattr(response_data, "status_code", "N/A")
            logger.error(
                f"Unexpected response type or status ({status}) for CSRF token API: {type(response_data)}"
            )
            logger.debug(f"Response data received: {response_data}")
            return None

    # End of get_csrf

    def get_cookies(self, cookie_names: List[str], timeout: int = 30) -> bool:
        """
        Waits for specified WebDriver cookies to be present.

        :param cookie_names: List of cookie names to wait for.
        :param timeout: Maximum time in seconds to wait.
        :return: True if all cookies were found within the timeout, False otherwise.
        """
        if not self.driver:
            return False  # Cannot get cookies without driver

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

                # Get current cookies safely
                cookies = self.driver.get_cookies()
                current_cookies_lower = {
                    c["name"].lower()
                    for c in cookies
                    if isinstance(c, dict) and "name" in c
                }

                missing_lower = required_lower - current_cookies_lower
                if not missing_lower:
                    logger.debug(f"All required cookies found: {cookie_names}.")
                    return True

                # Log missing cookies only if the set changes
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
                time.sleep(interval * 2)  # Longer sleep after error
            except Exception as e:  # Catch unexpected errors
                logger.error(f"Unexpected error retrieving cookies: {e}", exc_info=True)
                time.sleep(interval * 2)

        # After timeout, perform one final check
        missing_final = []
        try:
            if self.is_sess_valid():
                cookies_final = self.driver.get_cookies()
                current_cookies_final_lower = {
                    c["name"].lower()
                    for c in cookies_final
                    if isinstance(c, dict) and "name" in c
                }
                missing_final = [
                    name
                    for name in cookie_names
                    if name.lower() not in current_cookies_final_lower
                ]
            else:
                missing_final = cookie_names  # Assume all missing if session died
        except Exception:  # Catch errors during final check
            missing_final = cookie_names  # Assume all missing on error

        if missing_final:
            logger.warning(f"Timeout waiting for cookies. Missing: {missing_final}.")
            return False
        else:
            # Should ideally have returned True in the loop if found
            logger.debug("Cookies found in final check after loop (unexpected).")
            return True

    # End of get_cookies

    def _sync_cookies(self):
        """Syncs cookies from WebDriver to the shared requests.Session."""
        if not self.is_sess_valid():
            logger.warning("Cannot sync cookies: WebDriver session invalid.")
            return
        if not self.driver:
            logger.error("Cannot sync cookies: WebDriver instance is None.")
            return
        if not hasattr(self, "_requests_session") or not self._requests_session:
            logger.error("Cannot sync cookies: requests.Session not initialized.")
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
        except Exception as e:  # Catch unexpected errors
            logger.error(
                f"Unexpected error getting cookies for sync: {e}", exc_info=True
            )
            return

        # Sync to requests session
        requests_cookie_jar = self._requests_session.cookies
        requests_cookie_jar.clear()  # Clear existing requests cookies first
        synced_count = 0
        skipped_count = 0

        for cookie in driver_cookies:
            # Validate cookie format
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
                # Prepare attributes for requests.cookies.set
                cookie_attrs = {
                    "name": cookie["name"],
                    "value": cookie["value"],
                    "domain": cookie["domain"],
                    "path": cookie.get("path", "/"),
                    "secure": cookie.get("secure", False),
                    # requests uses 'rest' dict for additional flags like HttpOnly
                    "rest": {"httpOnly": cookie.get("httpOnly", False)},
                }
                # Handle expiry (Selenium uses 'expiry', requests uses 'expires')
                # Ensure expiry is an integer timestamp (seconds since epoch)
                if "expiry" in cookie and cookie["expiry"] is not None:
                    try:
                        # Selenium expiry might be float or int
                        cookie_attrs["expires"] = int(cookie["expiry"])
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Skipping invalid expiry format for cookie {cookie['name']}: {cookie['expiry']}"
                        )
                        # Optionally skip setting the cookie entirely if expiry is critical and invalid
                        # skipped_count += 1
                        # continue

                # Set the cookie in the requests jar
                requests_cookie_jar.set(**cookie_attrs)
                synced_count += 1

            except Exception as set_err:  # Catch errors during requests_cookie_jar.set
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
        """Closes a SQLAlchemy session, returning it to the pool."""
        if session:
            session_id = id(session)
            try:
                session.close()
                logger.debug(f"DB session {session_id} closed and returned to pool.")
            except Exception as e:  # Catch potential errors during close
                logger.error(
                    f"Error closing DB session {session_id}: {e}", exc_info=True
                )
        else:
            logger.warning("Attempted to return a None DB session.")

    # End of return_session

    def _reset_logged_flags(self):
        """Resets flags used to prevent redundant logging of identifiers."""
        self._profile_id_logged = False
        self._uuid_logged = False
        self._tree_id_logged = False
        self._owner_logged = False

    # End of _reset_logged_flags

    def get_db_conn(self) -> Optional[Session]:
        """Obtains a new SQLAlchemy session from the session factory."""
        engine_id_str = id(self.engine) if self.engine else "None"
        logger.debug(
            f"SessionManager ID={id(self)} get_db_conn called. Current Engine ID: {engine_id_str}"
        )

        # Initialize DB if needed (lazy initialization)
        if not self._db_init_attempted or not self.engine or not self.Session:
            logger.debug(
                f"SessionManager ID={id(self)}: Engine/Session factory not ready. Triggering initialization..."
            )
            try:
                self._initialize_db_engine_and_session()
                if not self.Session:  # Check if factory was created successfully
                    logger.error(
                        f"SessionManager ID={id(self)}: Initialization failed, cannot get DB connection."
                    )
                    return None
            except Exception as init_e:
                logger.error(
                    f"SessionManager ID={id(self)}: Exception during lazy initialization in get_db_conn: {init_e}"
                )
                return None

        # Get session from factory
        try:
            new_session: Session = self.Session()  # Type hint the session object
            logger.debug(
                f"SessionManager ID={id(self)} obtained DB session {id(new_session)} from Engine ID={id(self.engine)}"
            )
            return new_session
        except Exception as e:  # Catch errors getting session from factory
            logger.error(
                f"SessionManager ID={id(self)} Error getting DB session from factory: {e}",
                exc_info=True,
            )
            # Attempt to dispose engine if getting session fails, might indicate pool issues
            if self.engine:
                try:
                    self.engine.dispose()
                except Exception:
                    pass
            self.engine = None
            self.Session = None
            self._db_init_attempted = False  # Reset flag
            return None

    # End of get_db_conn

    @contextlib.contextmanager
    def get_db_conn_context(self) -> Generator[Optional[Session], None, None]:
        """Context manager for obtaining and managing a DB session."""
        session: Optional[Session] = None
        session_id_for_log = "N/A"
        try:
            session = self.get_db_conn()
            if session:
                session_id_for_log = id(session)
                logger.debug(
                    f"DB Context Manager: Acquired session {session_id_for_log}."
                )
                yield session  # Provide the session to the `with` block
                # Commit only if the session is still active after the block
                if session.is_active:
                    try:
                        session.commit()
                        logger.debug(
                            f"DB Context Manager: Commit successful for session {session_id_for_log}."
                        )
                    except SQLAlchemyError as commit_err:
                        logger.error(
                            f"DB Context Manager: Commit failed for session {session_id_for_log}: {commit_err}. Rolling back."
                        )
                        session.rollback()  # Rollback on commit error
                        raise  # Re-raise the commit error
                else:
                    logger.warning(
                        f"DB Context Manager: Session {session_id_for_log} inactive after yield, skipping commit."
                    )
            else:
                # Failed to get session from get_db_conn()
                logger.error("DB Context Manager: Failed to obtain DB session.")
                yield None  # Yield None if acquisition failed

        except SQLAlchemyError as sql_e:  # Catch DB-specific errors
            logger.error(
                f"DB Context Manager: SQLAlchemyError ({type(sql_e).__name__}). Rolling back session {session_id_for_log}.",
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
            raise sql_e  # Re-raise the original SQLAlchemy error
        except Exception as e:  # Catch other errors within the `with` block
            logger.error(
                f"DB Context Manager: Unexpected Exception ({type(e).__name__}). Rolling back session {session_id_for_log}.",
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
            raise e  # Re-raise other exceptions

        finally:
            # Ensure session is always returned to the pool
            if session:
                self.return_session(session)
            else:
                logger.debug("DB Context Manager: No valid session to return.")

    # End of get_db_conn_context

    def cls_db_conn(self, keep_db: bool = True):
        """Closes the database connection pool (disposes the engine)."""
        if keep_db:
            engine_id_str = id(self.engine) if self.engine else "None"
            logger.debug(
                f"cls_db_conn called (keep_db=True). Skipping engine disposal for Engine ID: {engine_id_str}"
            )
            return

        # Proceed with disposal if keep_db is False
        if not self.engine:
            logger.debug(
                f"SessionManager ID={id(self)}: No active SQLAlchemy engine to dispose."
            )
            self.Session = None  # Also clear session factory
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
            # Ensure state is fully reset after disposal attempt
            self.engine = None
            self.Session = None
            self._db_init_attempted = False

    # End of cls_db_conn

    @retry_api()
    def get_my_profileId(self) -> Optional[str]:
        """Fetches the user's own profile ID (ucdmid) via API."""
        if not self.is_sess_valid():
            logger.error("get_my_profileId: Session invalid.")
            return None

        url = urljoin(config_instance.BASE_URL, API_PATH_PROFILE_ID)
        logger.debug("Attempting to fetch own profile ID (ucdmid)...")

        try:
            response_data: ApiResponseType = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="Get my profile_id",
            )

            if not response_data:
                logger.warning(
                    "Failed to get profile_id response via _api_req (returned None/error)."
                )
                return None

            # Expecting a dictionary like {"data": {"ucdmid": "..."}}
            if isinstance(response_data, dict) and KEY_DATA in response_data:
                data_dict = response_data[KEY_DATA]
                if isinstance(data_dict, dict) and KEY_UCDMID in data_dict:
                    my_profile_id_val = str(data_dict[KEY_UCDMID]).upper()
                    logger.debug(
                        f"Successfully retrieved profile_id: {my_profile_id_val}"
                    )
                    self.my_profile_id = my_profile_id_val  # Store it
                    return my_profile_id_val
                else:
                    logger.error(
                        f"Could not find '{KEY_UCDMID}' in '{KEY_DATA}' dict of profile_id API response."
                    )
                    logger.debug(f"Data dict received: {data_dict}")
                    return None
            else:
                # Handle case where response is not dict or missing 'data'
                status = (
                    getattr(response_data, "status_code", "N/A")
                    if isinstance(response_data, requests.Response)
                    else "N/A"
                )
                logger.error(
                    f"Unexpected response format (Type: {type(response_data)}, Status: {status}) for profile_id API."
                )
                logger.debug(f"Full profile_id response data: {response_data}")
                return None

        except Exception as e:  # Catch unexpected errors during processing
            logger.error(f"Unexpected error in get_my_profileId: {e}", exc_info=True)
            return None

    # End of get_my_profileId

    @retry_api()
    def get_my_uuid(self) -> Optional[str]:
        """Fetches the user's own UUID (testId) via API."""
        if not self.is_sess_valid():
            logger.error("get_my_uuid: Session invalid.")
            return None

        url = urljoin(config_instance.BASE_URL, API_PATH_UUID)
        logger.debug("Attempting to fetch own UUID (testId) from header/dna API...")

        response_data: ApiResponseType = _api_req(
            url=url,
            driver=self.driver,
            session_manager=self,
            method="GET",
            use_csrf_token=False,
            api_description="Get UUID API",
        )

        if response_data and isinstance(response_data, dict):
            if KEY_TEST_ID in response_data:
                my_uuid_val = str(response_data[KEY_TEST_ID]).upper()
                logger.debug(f"Successfully retrieved UUID: {my_uuid_val}")
                self.my_uuid = my_uuid_val  # Store it
                return my_uuid_val
            else:
                logger.error(
                    f"Could not retrieve UUID ('{KEY_TEST_ID}' missing in response)."
                )
                logger.debug(f"Full get_my_uuid response data: {response_data}")
                return None
        elif response_data is None:
            logger.error(
                "Failed to get header/dna data via _api_req (returned None/error)."
            )
            return None
        else:
            # Handle unexpected response types (e.g., requests.Response object on non-2xx)
            status = (
                getattr(response_data, "status_code", "N/A")
                if isinstance(response_data, requests.Response)
                else "N/A"
            )
            logger.error(
                f"Unexpected response format (Type: {type(response_data)}, Status: {status}) for UUID API."
            )
            logger.debug(f"Full get_my_uuid response data: {response_data}")
            return None

    # End of get_my_uuid

    @retry_api()
    def get_my_tree_id(self) -> Optional[str]:
        """Fetches the Tree ID corresponding to the configured TREE_NAME via API."""
        tree_name_config = config_instance.TREE_NAME
        if not tree_name_config:
            logger.debug("TREE_NAME not configured, skipping tree ID retrieval.")
            return None
        if not self.is_sess_valid():
            logger.error("get_my_tree_id: Session invalid.")
            return None

        url = urljoin(config_instance.BASE_URL, API_PATH_HEADER_TREES)
        logger.debug(
            f"Attempting to fetch tree ID for TREE_NAME='{tree_name_config}'..."
        )

        try:
            response_data: ApiResponseType = _api_req(
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
                and KEY_MENUITEMS in response_data
                and isinstance(response_data[KEY_MENUITEMS], list)
            ):
                for item in response_data[KEY_MENUITEMS]:
                    if (
                        isinstance(item, dict)
                        and item.get(KEY_TEXT) == tree_name_config
                    ):
                        tree_url = item.get(KEY_URL)
                        if tree_url and isinstance(tree_url, str):
                            # Extract tree ID using regex
                            match = re.search(r"/tree/(\d+)", tree_url)
                            if match:
                                my_tree_id_val = match.group(1)
                                logger.debug(
                                    f"Found tree ID '{my_tree_id_val}' for tree '{tree_name_config}'."
                                )
                                self.my_tree_id = my_tree_id_val  # Store it
                                return my_tree_id_val
                            else:
                                logger.warning(
                                    f"Found tree '{tree_name_config}', but URL format unexpected: {tree_url}"
                                )
                        else:
                            logger.warning(
                                f"Found tree '{tree_name_config}', but '{KEY_URL}' key missing or invalid."
                            )
                        # Found the matching tree name, stop searching
                        break
                # If loop finishes without finding the tree
                logger.warning(
                    f"Could not find TREE_NAME '{tree_name_config}' in Header Trees API response."
                )
                return None
            else:
                # Handle unexpected response format
                status = (
                    getattr(response_data, "status_code", "N/A")
                    if isinstance(response_data, requests.Response)
                    else "N/A"
                )
                logger.warning(
                    f"Unexpected response format from Header Trees API (Type: {type(response_data)}, Status: {status})."
                )
                logger.debug(f"Full Header Trees response data: {response_data}")
                return None

        except Exception as e:  # Catch unexpected errors during processing
            logger.error(f"Error fetching/parsing Header Trees API: {e}", exc_info=True)
            return None

    # End of get_my_tree_id

    @retry_api()
    def get_tree_owner(self, tree_id: str) -> Optional[str]:
        """Fetches the display name of the owner for a given Tree ID via API."""
        if not tree_id:
            logger.warning("Cannot get tree owner: tree_id is missing.")
            return None
        if not isinstance(tree_id, str):
            logger.warning(
                f"Invalid tree_id type provided: {type(tree_id)}. Expected string."
            )
            return None
        if not self.is_sess_valid():
            logger.error("get_tree_owner: Session invalid.")
            return None

        url = urljoin(
            config_instance.BASE_URL, f"{API_PATH_TREE_OWNER_INFO}?tree_id={tree_id}"
        )
        logger.debug(f"Attempting to fetch tree owner name for tree ID: {tree_id}...")

        try:
            response_data: ApiResponseType = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="Tree Owner Name API",
            )

            if response_data and isinstance(response_data, dict):
                owner_data = response_data.get(KEY_OWNER)
                if owner_data and isinstance(owner_data, dict):
                    display_name = owner_data.get(KEY_DISPLAY_NAME)
                    if display_name and isinstance(display_name, str):
                        logger.debug(
                            f"Found tree owner '{display_name}' for tree ID {tree_id}."
                        )
                        self.tree_owner_name = display_name  # Store it
                        return display_name
                    else:
                        logger.warning(
                            f"Could not find '{KEY_DISPLAY_NAME}' in owner data for tree {tree_id}."
                        )
                else:
                    logger.warning(
                        f"Could not find '{KEY_OWNER}' data in Tree Owner API response for tree {tree_id}."
                    )

                # Log full response if owner/displayName missing
                logger.debug(f"Full Tree Owner API response data: {response_data}")
                return None  # Return None if data is missing
            else:
                # Handle unexpected response format
                status = (
                    getattr(response_data, "status_code", "N/A")
                    if isinstance(response_data, requests.Response)
                    else "N/A"
                )
                logger.warning(
                    f"Tree Owner API call returned unexpected data (Type: {type(response_data)}, Status: {status}) or None."
                )
                logger.debug(f"Response received: {response_data}")
                return None

        except Exception as e:  # Catch unexpected errors during processing
            logger.error(
                f"Error fetching/parsing Tree Owner API for tree {tree_id}: {e}",
                exc_info=True,
            )
            return None

    # End of get_tree_owner

    def verify_sess(self) -> bool:
        """Verifies session status by checking login status."""
        logger.debug("Verifying session status (using login_status)...")
        try:
            login_ok = login_status(self)
            if login_ok is True:
                logger.debug("Session verification successful (logged in).")
                return True
            elif login_ok is False:
                logger.warning("Session verification failed (user not logged in).")
                return False
            else:  # login_status returned None (critical error)
                logger.error(
                    "Session verification failed critically (login_status returned None)."
                )
                return False
        except Exception as e:
            logger.error(
                f"Unexpected error during session verification: {e}", exc_info=True
            )
            return False

    # End of verify_sess

    def _verify_api_login_status(self) -> Optional[bool]:
        """
        Verifies login status using a simple authenticated API endpoint.

        Returns:
            True: If the API call succeeds (implies logged in).
            False: If the API call fails with 401/403 (implies not logged in).
            None: If the check fails for other reasons (network error, unexpected status, etc.).
        """
        api_url = urljoin(config_instance.BASE_URL, API_PATH_UUID)  # Use UUID endpoint
        api_description = "API Login Verification (header/dna)"
        logger.debug(f"Verifying login status via API endpoint: {api_url}...")

        if not self.driver or not self.is_sess_valid():
            logger.warning(
                f"{api_description}: Driver/session not valid for API check."
            )
            return None  # Cannot perform check

        # Sync cookies before the check
        try:
            logger.debug("Syncing cookies before API login check...")
            self._sync_cookies()
        except Exception as sync_e:
            logger.warning(f"Error syncing cookies before API login check: {sync_e}")
            # Proceed anyway, but it might fail

        try:
            # Make the API request - use a shorter timeout for a simple check
            response_data: ApiResponseType = _api_req(
                url=api_url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description=api_description,
                timeout=15,  # Shorter timeout for login check
            )

            # Analyze the response
            if response_data is None:
                # _api_req already logged the retry failures/errors
                logger.warning(
                    f"{api_description}: _api_req returned None. Assuming login check failed."
                )
                return None
            elif isinstance(response_data, requests.Response):
                # Got a Response object, indicating _api_req failed before parsing JSON
                # Typically means a non-2xx status code that wasn't retryable
                status_code = response_data.status_code
                if status_code in [401, 403]:
                    logger.debug(
                        f"{api_description}: API check failed with status {status_code}. User NOT logged in."
                    )
                    return False
                else:
                    # Unexpected non-retryable error
                    logger.warning(
                        f"{api_description}: API check failed with unexpected status {status_code}. Returning None."
                    )
                    return None
            elif isinstance(response_data, dict):
                # Got a dictionary, implies 2xx status
                if KEY_TEST_ID in response_data:
                    logger.debug(
                        f"{api_description}: API login check successful ('{KEY_TEST_ID}' found)."
                    )
                    return True
                else:
                    # 2xx status but unexpected content
                    logger.warning(
                        f"{api_description}: API check succeeded (2xx), but response format unexpected (missing '{KEY_TEST_ID}'). Assuming logged in cautiously."
                    )
                    logger.debug(f"API Response Content: {response_data}")
                    return True  # Treat as logged in if API returned 2xx
            else:
                # Unexpected type returned by _api_req
                logger.error(
                    f"{api_description}: _api_req returned unexpected type {type(response_data)}. Returning None."
                )
                logger.debug(f"API Response Data: {response_data}")
                return None

        except (
            RequestException
        ) as req_e:  # Catch request exceptions explicitly if retry_api is bypassed/fails
            logger.error(
                f"RequestException during {api_description}: {req_e}", exc_info=False
            )
            return None
        except Exception as e:  # Catch other unexpected errors
            logger.error(
                f"Unexpected error during {api_description}: {e}", exc_info=True
            )
            return None

    # End of _verify_api_login_status

    @retry_api()
    def get_header(self) -> bool:
        """DEPRECATED? Fetches header/dna data. Use specific getters instead."""
        logger.warning(
            "get_header() is likely deprecated. Use get_my_uuid() etc. instead."
        )
        if not self.is_sess_valid():
            logger.error("get_header: Session invalid.")
            return False

        url = urljoin(config_instance.BASE_URL, API_PATH_UUID)
        logger.debug("Attempting to fetch header/dna API data...")

        response_data: ApiResponseType = _api_req(
            url,
            self.driver,
            self,
            method="GET",
            use_csrf_token=False,
            api_description="Get UUID API (via get_header)",
        )

        if (
            response_data
            and isinstance(response_data, dict)
            and KEY_TEST_ID in response_data
        ):
            logger.debug("Header data retrieved successfully ('testId' found).")
            return True
        else:
            status = (
                getattr(response_data, "status_code", "N/A")
                if isinstance(response_data, requests.Response)
                else "N/A"
            )
            logger.error(
                f"Failed to get header/dna data or unexpected structure (Type: {type(response_data)}, Status: {status})."
            )
            logger.debug(f"Response: {response_data}")
            return False

    # End of get_header

    def _validate_sess_cookies(self, required_cookies: List[str]) -> bool:
        """Validates the presence of required cookies in the current WebDriver session."""
        if not self.is_sess_valid():
            logger.warning("Cannot validate cookies: Session invalid.")
            return False
        if not self.driver:
            logger.error("Cannot validate cookies: WebDriver instance is None.")
            return False

        try:
            # Get current cookies as a dictionary {name: value}
            cookies = {
                c["name"]: c["value"]
                for c in self.driver.get_cookies()
                if isinstance(c, dict) and "name" in c
            }
            missing_cookies = [name for name in required_cookies if name not in cookies]

            if not missing_cookies:
                logger.debug(f"Cookie validation successful for: {required_cookies}")
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
        except Exception as e:  # Catch unexpected errors
            logger.error(f"Unexpected error validating cookies: {e}", exc_info=True)
            return False

    # End of _validate_sess_cookies

    def is_sess_logged_in(self) -> bool:
        """DEPRECATED: Use login_status() or verify_sess() instead."""
        logger.warning(
            "is_sess_logged_in is deprecated. Use login_status() or verify_sess() instead."
        )
        return self.verify_sess()  # Delegate to verify_sess

    # End of is_sess_logged_in

    def is_sess_valid(self) -> bool:
        """Checks if the WebDriver session is still active and responsive."""
        if not self.driver:
            return False
        try:
            # Accessing window_handles is a lightweight way to check session validity
            _ = self.driver.window_handles
            # Check if the title is accessible (sometimes handles exist but page crashed)
            _ = self.driver.title
            return True
        except InvalidSessionIdException:
            logger.debug(
                "Session ID is invalid (browser likely closed or session terminated)."
            )
            return False
        except (NoSuchWindowException, WebDriverException) as e:
            # Check for common connection/crash related messages
            err_str = str(e).lower()
            if any(
                sub in err_str
                for sub in [
                    "disconnected",
                    "target crashed",
                    "no such window",
                    "unable to connect",
                    "invalid session id",
                ]
            ):
                logger.warning(
                    f"Session seems invalid due to WebDriverException: {type(e).__name__}"
                )
                return False
            else:
                # Log other WebDriverExceptions but maybe cautiously return True? Or False?
                # Let's be cautious: if a WebDriverException occurs here, assume invalid.
                logger.warning(
                    f"Unexpected WebDriverException checking session validity, assuming invalid: {e}"
                )
                return False
        except Exception as e:  # Catch other unexpected errors
            logger.error(
                f"Unexpected error checking session validity: {e}", exc_info=True
            )
            return False

    # End of is_sess_valid

    def close_sess(self, keep_db: bool = False):
        """Closes the WebDriver session and optionally the DB connection pool."""
        if self.driver:
            logger.debug("Attempting to close WebDriver session...")
            try:
                self.driver.quit()
                logger.debug("WebDriver session quit successfully.")
            except WebDriverException as e:  # More specific exception
                logger.error(f"Error closing WebDriver session: {e}", exc_info=False)
            except Exception as e:  # Catch other potential errors
                logger.error(
                    f"Unexpected error closing WebDriver session: {e}", exc_info=True
                )
            finally:
                self.driver = None  # Ensure driver is set to None
        else:
            logger.debug("No active WebDriver session to close.")

        # Reset state flags
        self.driver_live = False
        self.session_ready = False
        self.csrf_token = None  # Clear CSRF token on close

        # Handle DB connection
        if not keep_db:
            logger.debug("Closing database connection pool...")
            self.cls_db_conn(keep_db=False)
        else:
            logger.debug("Keeping DB connection pool alive (keep_db=True).")

    # End of close_sess

    def restart_sess(self, url: Optional[str] = None) -> bool:
        """Restarts the session (closes driver, starts new one, ensures readiness)."""
        logger.warning("Restarting WebDriver session...")
        self.close_sess(keep_db=True)  # Keep DB connection during restart

        # Phase 1: Start new driver
        start_ok = self.start_sess(action_name="Session Restart - Phase 1")
        if not start_ok:
            logger.error("Failed to restart session (Phase 1: Driver Start failed).")
            return False

        # Phase 2: Ensure new session is ready
        ready_ok = self.ensure_session_ready(action_name="Session Restart - Phase 2")
        if not ready_ok:
            logger.error("Failed to restart session (Phase 2: Session Ready failed).")
            self.close_sess(keep_db=True)  # Clean up the failed restart attempt
            return False

        # Optional: Navigate to a specific URL after restart
        if url and self.driver:
            logger.info(f"Session restart successful. Re-navigating to: {url}")
            if nav_to_page(self.driver, url, selector="body", session_manager=self):
                logger.info(f"Successfully re-navigated to {url}.")
                return True
            else:
                logger.error(
                    f"Failed to re-navigate to {url} after successful restart."
                )
                return False  # Navigation failed after restart
        elif not url:
            logger.info("Session restart successful (no navigation requested).")
            return True
        else:  # Should not happen if ready_ok is True
            logger.error(
                "Driver instance missing after successful session restart report."
            )
            return False

    # End of restart_sess

    @ensure_browser_open
    def make_tab(self) -> Optional[str]:
        """Opens a new browser tab and returns its window handle."""
        driver = self.driver  # Driver existence ensured by decorator
        if driver is None:  # Should not happen due to decorator, but satisfy linters
            logger.error("Driver is None in make_tab despite decorator.")
            return None

        try:
            tab_list_before = driver.window_handles
            logger.debug(f"Window handles before new tab: {tab_list_before}")

            # Execute JS to open a new tab
            driver.switch_to.new_window("tab")
            logger.debug("Executed new_window('tab') command.")

            # Wait for the new handle to appear
            WebDriverWait(driver, selenium_config.NEW_TAB_TIMEOUT).until(
                lambda d: len(d.window_handles) > len(tab_list_before)
            )

            tab_list_after = driver.window_handles
            # Find the new handle using set difference
            new_tab_handles = list(set(tab_list_after) - set(tab_list_before))

            if new_tab_handles:
                new_tab_handle = new_tab_handles[0]
                logger.debug(f"New tab handle identified: {new_tab_handle}")
                return new_tab_handle
            else:
                # This case should ideally not happen if the wait succeeded
                logger.error(
                    "Could not identify new tab handle (set difference empty after wait)."
                )
                logger.debug(
                    f"Handles before: {tab_list_before}, Handles after: {tab_list_after}"
                )
                return None

        except TimeoutException:
            logger.error("Timeout waiting for new tab handle to appear.")
            try:
                logger.debug(f"Window handles during timeout: {driver.window_handles}")
            except Exception:
                pass
            return None
        except WebDriverException as e:  # Catch specific Selenium errors
            logger.error(f"WebDriverException identifying new tab handle: {e}")
            try:
                logger.debug(f"Window handles during error: {driver.window_handles}")
            except Exception:
                pass
            return None
        except Exception as e:  # Catch unexpected errors
            logger.error(
                f"An unexpected error occurred in make_tab: {e}", exc_info=True
            )
            return None

    # End of make_tab

    def check_js_errors(self):
        """Checks the browser console log for severe JavaScript errors since the last check."""
        if not self.is_sess_valid() or not self.driver:
            logger.debug("Skipping JS error check: Session invalid or driver None.")
            return

        try:
            # Check if the 'browser' log type is available
            log_types = self.driver.log_types
            if "browser" not in log_types:
                logger.debug("Browser log type not available, skipping JS error check.")
                return
        except WebDriverException as e:
            # Handle cases where getting log_types fails (e.g., driver disconnected)
            logger.warning(f"Could not get log_types: {e}. Skipping JS error check.")
            return
        except AttributeError:  # If self.driver became None unexpectedly
            logger.warning(
                "Driver became None before getting log_types. Skipping JS error check."
            )
            return

        try:
            # Retrieve browser logs
            logs = self.driver.get_log("browser")
        except WebDriverException as e:
            logger.warning(f"WebDriverException getting browser logs: {e}")
            return
        except Exception as e:  # Catch other unexpected errors
            logger.error(f"Unexpected error getting browser logs: {e}", exc_info=True)
            return

        new_errors_found = False
        most_recent_error_time_this_check = (
            self.last_js_error_check
        )  # Initialize with last check time

        for entry in logs:
            # Ensure entry is a dictionary and has expected keys
            if not isinstance(entry, dict):
                continue
            level = entry.get("level")
            timestamp_ms = entry.get("timestamp")  # Milliseconds since epoch

            if level == "SEVERE" and timestamp_ms:
                try:
                    # Convert ms timestamp to timezone-aware datetime (UTC)
                    timestamp_dt = datetime.fromtimestamp(
                        timestamp_ms / 1000.0, tz=timezone.utc
                    )

                    # Check if this error is newer than the last check
                    if timestamp_dt > self.last_js_error_check:
                        new_errors_found = True
                        error_message = entry.get("message", "No message")

                        # Try to extract source file/line number
                        source_match = re.search(r"(.+?):(\d+)", error_message)
                        source_info = ""
                        if source_match:
                            # Get filename only from potential path
                            filename = (
                                source_match.group(1).split("/")[-1].split("\\")[-1]
                            )
                            line_num = source_match.group(2)
                            source_info = f" (Source: {filename}:{line_num})"

                        logger.warning(
                            f"JS ERROR DETECTED:{source_info} {error_message}"
                        )

                        # Update the time of the most recent error found *in this batch*
                        if timestamp_dt > most_recent_error_time_this_check:
                            most_recent_error_time_this_check = timestamp_dt
                except (TypeError, ValueError) as parse_e:
                    logger.warning(
                        f"Error parsing JS log entry timestamp {timestamp_ms}: {parse_e}"
                    )
                except Exception as entry_proc_e:  # Catch other errors processing entry
                    logger.warning(
                        f"Error processing JS log entry {entry}: {entry_proc_e}"
                    )

        # Update the last check time to the timestamp of the latest error found in this run
        # This prevents re-logging old errors if the check runs again quickly
        if new_errors_found:
            self.last_js_error_check = most_recent_error_time_this_check
            logger.debug(
                f"Updated last_js_error_check time to: {self.last_js_error_check.isoformat()}"
            )

    # End of check_js_errors


# End of SessionManager class


# ----------------------------------------------------------------------------
# Stand alone functions
# ----------------------------------------------------------------------------


def _prepare_api_headers(
    session_manager: SessionManager,
    driver: DriverType,
    api_description: str,
    base_headers: Dict[str, str],
    use_csrf_token: bool,
    add_default_origin: bool,
) -> Dict[str, str]:
    """Prepares the headers for an API request, including dynamic values."""
    final_headers = base_headers.copy()
    cfg = config_instance  # Alias

    # --- Add User-Agent (try driver first, then random default) ---
    ua_set = False
    if driver:
        try:
            ua = driver.execute_script("return navigator.userAgent;")
            if ua and isinstance(ua, str):
                final_headers["User-Agent"] = ua
                ua_set = True
        except WebDriverException:  # Handle driver error getting UA
            logger.debug(
                f"[{api_description}] WebDriver error getting User-Agent, using default."
            )
        except Exception as e:
            logger.warning(
                f"[{api_description}] Unexpected error getting User-Agent: {e}, using default."
            )
    if not ua_set:
        final_headers["User-Agent"] = random.choice(
            cfg.USER_AGENTS if cfg else ["Mozilla/5.0"]
        )

    # --- Add Origin if applicable ---
    http_method = base_headers.get(
        "_method", "GET"
    ).upper()  # Assume GET if not passed internally
    if add_default_origin and http_method not in ["GET", "HEAD", "OPTIONS"]:
        try:
            if cfg and cfg.BASE_URL:
                parsed_base_url = urlparse(cfg.BASE_URL)
                origin_header_value = (
                    f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"
                )
                final_headers["Origin"] = origin_header_value
            else:
                logger.warning(
                    f"[{api_description}] Cannot set default Origin header: Config or BASE_URL missing."
                )
        except Exception as parse_err:
            logger.warning(
                f"[{api_description}] Could not parse BASE_URL for Origin header: {parse_err}"
            )

    # --- Add Dynamic Headers (NewRelic, Traceparent, etc.) ---
    # These functions handle None driver gracefully
    final_headers["newrelic"] = make_newrelic(driver) or ""
    final_headers["traceparent"] = make_traceparent(driver) or ""
    final_headers["tracestate"] = make_tracestate(driver) or ""

    # --- Add UBE Header ---
    if driver:  # Only try if driver exists
        ube_header = make_ube(driver)
        if ube_header:
            final_headers["ancestry-context-ube"] = ube_header
        else:
            logger.debug(f"[{api_description}] Could not generate UBE header.")

    # --- Add CSRF Token if needed ---
    if use_csrf_token:
        csrf_token = session_manager.csrf_token
        if csrf_token:
            # Handle potential JSON string format vs raw token
            raw_token_val = csrf_token
            if isinstance(csrf_token, str) and csrf_token.strip().startswith("{"):
                try:
                    token_obj = json.loads(csrf_token)
                    raw_token_val = token_obj.get(
                        "csrfToken", csrf_token
                    )  # Default to original if key missing
                except json.JSONDecodeError:
                    logger.warning(
                        f"[{api_description}] CSRF token looks like JSON but failed to parse, using raw value."
                    )
            final_headers["X-CSRF-Token"] = raw_token_val
            logger.debug(f"[{api_description}] Added X-CSRF-Token header.")
        else:
            logger.warning(
                f"[{api_description}] CSRF token requested but not found in SessionManager."
            )

    # --- Conditionally Add ancestry-userid ---
    exclude_userid_for = [
        "Ancestry Facts JSON Endpoint",  # Add API descriptions that should NOT have the header
        "Ancestry Person Picker",
        "CSRF Token API",  # Don't need user ID to get CSRF token
        # Add others as needed
    ]
    if session_manager.my_profile_id and api_description not in exclude_userid_for:
        final_headers["ancestry-userid"] = session_manager.my_profile_id.upper()
    elif api_description in exclude_userid_for and session_manager.my_profile_id:
        logger.debug(
            f"[{api_description}] Omitting 'ancestry-userid' header as configured."
        )

    # --- Final Cleanup: Remove empty headers ---
    final_headers = {k: v for k, v in final_headers.items() if v}

    return final_headers


# End of _prepare_api_headers


def _api_req(
    url: str,
    driver: DriverType,  # Optional WebDriver instance
    session_manager: SessionManager,  # Required SessionManager instance
    method: str = "GET",
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    use_csrf_token: bool = True,
    headers: Optional[Dict[str, str]] = None,  # Explicit header overrides
    referer_url: Optional[str] = None,
    api_description: str = "API Call",
    timeout: Optional[int] = None,
    cookie_jar: Optional[RequestsCookieJar] = None,  # Allow passing specific cookie jar
    allow_redirects: bool = True,
    force_text_response: bool = False,
    add_default_origin: bool = True,
) -> Union[ApiResponseType, RequestsResponseTypeOptional]:  # Return type hint
    """
    Makes an HTTP request using the shared requests.Session from SessionManager.
    Handles dynamic header generation, cookie synchronization, rate limiting,
    retries, and basic response processing.

    Args:
        url: The target URL for the API request.
        driver: The current WebDriver instance (optional, used for dynamic headers/cookies).
        session_manager: The active SessionManager instance.
        method: HTTP method (GET, POST, etc.). Defaults to GET.
        data: Dictionary for form-encoded data (POST/PUT).
        json_data: Dictionary for JSON payload (POST/PUT).
        use_csrf_token: Whether to include the X-CSRF-Token header. Defaults to True.
        headers: Optional dictionary of headers to explicitly override defaults.
        referer_url: Optional Referer header value. Defaults to BASE_URL.
        api_description: A descriptive name for the API call (for logging).
        timeout: Optional request timeout in seconds. Defaults to selenium_config.API_TIMEOUT.
        cookie_jar: Optional specific requests.cookies.RequestsCookieJar to use. Defaults to session_manager's jar.
        allow_redirects: Whether to allow requests to follow redirects. Defaults to True.
        force_text_response: If True, always return the response.text. Defaults to False.
        add_default_origin: If True, adds Origin header for non-GET/HEAD requests. Defaults to True.


    Returns:
        - Parsed JSON (dict/list) if response Content-Type is application/json and parsing succeeds.
        - Raw text (str) if force_text_response is True or if Content-Type is not JSON.
        - None if the request fails after all retries due to network issues or retryable errors.
        - requests.Response object if the request results in a non-retryable HTTP error status
          (e.g., 400, 401, 404) or if allow_redirects is False and a redirect occurs.
    """
    # --- Step 1: Validate prerequisites ---
    if not session_manager or not session_manager._requests_session:
        logger.error(
            f"{api_description}: Aborting - SessionManager or internal requests_session missing."
        )
        return None
    if not config_instance or not selenium_config:
        logger.error(f"{api_description}: Aborting - Config instances not loaded.")
        return None

    cfg = config_instance  # Alias
    sel_cfg = selenium_config  # Alias

    # --- Step 2: Get Retry Configuration ---
    # These could potentially be passed via kwargs in the future if needed per-call
    max_retries = cfg.MAX_RETRIES
    initial_delay = cfg.INITIAL_DELAY
    backoff_factor = cfg.BACKOFF_FACTOR
    max_delay = cfg.MAX_DELAY
    retry_status_codes = set(cfg.RETRY_STATUS_CODES)

    # --- Step 3: Prepare Base Headers ---
    base_headers: Dict[str, str] = {
        # User-Agent will be added dynamically in _prepare_api_headers
        "Accept": "application/json, text/plain, */*",
        # Origin will be added dynamically if needed
        "Referer": referer_url or cfg.BASE_URL,
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "_method": method.upper(),  # Pass method for origin logic
    }
    # Apply context-specific headers from config
    contextual_headers = cfg.API_CONTEXTUAL_HEADERS.get(api_description, {})
    base_headers.update({k: v for k, v in contextual_headers.items() if v is not None})
    # Apply explicit overrides from the 'headers' argument
    if headers:
        filtered_overrides = {k: v for k, v in headers.items() if v is not None}
        base_headers.update(filtered_overrides)
        if filtered_overrides:
            logger.debug(
                f"[{api_description}] Applied {len(filtered_overrides)} explicit header overrides."
            )

    # --- Step 4: Prepare Request Details ---
    request_timeout = timeout if timeout is not None else sel_cfg.API_TIMEOUT
    req_session = session_manager._requests_session
    effective_cookie_jar = cookie_jar if cookie_jar is not None else req_session.cookies
    http_method = method.upper()

    # Special handling for specific API calls (e.g., redirects)
    effective_allow_redirects = allow_redirects
    if api_description == "Match List API" and effective_allow_redirects:
        logger.debug(f"Forcing allow_redirects=False for '{api_description}'.")
        effective_allow_redirects = False

    logger.debug(
        f"API Req: {http_method} {url} (Timeout: {request_timeout}s, AllowRedirects: {effective_allow_redirects})"
    )

    # --- Step 5: Execute Request with Retry Loop ---
    retries_left = max_retries
    last_exception: Optional[Exception] = None
    response: RequestsResponseTypeOptional = (
        None  # Ensure response is defined outside loop
    )
    current_delay = initial_delay  # Delay for retry backoff

    while retries_left > 0:
        attempt = max_retries - retries_left + 1
        final_headers: Dict[str, str] = {}  # Reset headers each attempt

        # Check driver validity (only impacts dynamic header generation)
        driver_is_valid = driver and session_manager.is_sess_valid()
        if not driver_is_valid and attempt == 1:  # Log only once
            # Check if this API *requires* driver-dependent headers (UBE, etc.)
            # For now, just warn generally.
            apis_needing_driver_headers = {
                "Match List API",
                "Match Probability API (Cloudscraper)",
            }  # Example
            if api_description in apis_needing_driver_headers:
                logger.warning(
                    f"{api_description}: Browser session invalid (Attempt {attempt}). Dynamic headers (UBE, etc.) might be missing or stale."
                )
            else:
                logger.debug(
                    f"{api_description}: Browser session invalid or driver is None (Attempt {attempt}). Dynamic headers might be incomplete."
                )

        try:
            # --- Sync Cookies (if driver is valid) ---
            if driver_is_valid:
                try:
                    session_manager._sync_cookies()  # Sync WebDriver cookies to requests session
                    logger.debug(
                        f"[{api_description}] Cookies synced (Attempt {attempt})."
                    )
                except Exception as sync_err:  # Catch errors during sync
                    logger.warning(
                        f"[{api_description}] Error syncing cookies (Attempt {attempt}): {sync_err}"
                    )
            else:
                logger.debug(
                    f"[{api_description}] Skipping cookie sync - driver invalid (Attempt {attempt})"
                )

            # --- Generate Headers ---
            final_headers = _prepare_api_headers(
                session_manager,
                driver,
                api_description,
                base_headers,
                use_csrf_token,
                add_default_origin,
            )

            # --- Apply Rate Limit Wait ---
            wait_time = session_manager.dynamic_rate_limiter.wait()
            if wait_time > 0.1:  # Log only significant waits
                logger.debug(
                    f"[{api_description}] Rate limit wait: {wait_time:.2f}s (Attempt {attempt})"
                )

            # --- Log Request Details (Optional Debugging) ---
            if logger.isEnabledFor(logging.DEBUG):
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
                    if k.lower() in sensitive_keys_debug and v and len(v) > 20:
                        log_hdrs_debug[k] = v[:10] + "..." + v[-5:]
                    else:
                        log_hdrs_debug[k] = v
                logger.debug(
                    f"[_api_req Attempt {attempt} '{api_description}'] Sending Headers: {log_hdrs_debug}"
                )
                # try: # Logging cookies can be verbose
                #      cookies_dict_log = effective_cookie_jar.get_dict()
                #      logger.debug(f"[_api_req Attempt {attempt} '{api_description}'] Sending Cookies: {list(cookies_dict_log.keys())}")
                # except Exception as cookie_log_err: logger.warning(f"[_api_req DEBUG] Could not log cookies: {cookie_log_err}")

            # --- Make the request ---
            logger.debug(
                f"[_api_req Attempt {attempt} '{api_description}'] >>> Calling requests.request..."
            )
            response = req_session.request(
                method=http_method,
                url=url,
                headers=final_headers,
                data=data,
                json=json_data,
                timeout=request_timeout,
                verify=True,  # Standard SSL verification
                allow_redirects=effective_allow_redirects,
                cookies=effective_cookie_jar,  # Use the determined cookie jar
            )
            logger.debug(
                f"[_api_req Attempt {attempt} '{api_description}'] <<< requests.request returned."
            )

            # --- Process Response ---
            status = response.status_code
            logger.debug(f"<-- Response Status: {status} {response.reason}")

            # Check for retryable status codes
            if status in retry_status_codes:
                retries_left -= 1
                last_exception = HTTPError(
                    f"{status} Error", response=response
                )  # Use requests exception
                if retries_left <= 0:
                    logger.error(
                        f"{api_description}: Failed after {max_retries} attempts (Final Status {status})."
                    )
                    return response  # Return the last failed response object
                else:
                    sleep_time = min(
                        current_delay * (backoff_factor ** (attempt - 1)), max_delay
                    ) + random.uniform(0, 0.2)
                    sleep_time = max(0.1, sleep_time)  # Min sleep
                    # Increase delay if rate limited (429)
                    if status == 429:
                        session_manager.dynamic_rate_limiter.increase_delay()
                    logger.warning(
                        f"{api_description}: Status {status} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    current_delay *= backoff_factor  # Increase delay for next retry
                    continue  # Go to next iteration of the while loop

            # Handle redirects if allow_redirects is False
            elif 300 <= status < 400 and not effective_allow_redirects:
                logger.warning(
                    f"{api_description}: Status {status} {response.reason} (Redirects Disabled). Returning Response object."
                )
                return response  # Return the response object indicating redirection

            # Handle unexpected redirects if allow_redirects is True (should have been followed)
            elif 300 <= status < 400 and effective_allow_redirects:
                # This shouldn't normally happen with requests default behavior
                logger.warning(
                    f"{api_description}: Unexpected final status {status} {response.reason} (Redirects Enabled). Returning Response object."
                )
                return response

            # Handle successful responses (2xx)
            elif response.ok:
                session_manager.dynamic_rate_limiter.decrease_delay()  # Success, decrease future delay

                # Return raw text if requested
                if force_text_response:
                    return response.text

                # Try to parse JSON if applicable
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    try:
                        # Handle empty response body for JSON
                        return response.json() if response.content else None
                    except JSONDecodeError as json_err:
                        logger.error(
                            f"{api_description}: OK ({status}), but JSON decode FAILED: {json_err}\nContent: {response.text[:500]}..."
                        )
                        # Return None or the raw text on decode failure? Let's return None.
                        return None
                # Handle plain text explicitly for CSRF token
                elif (
                    api_description == "CSRF Token API" and "text/plain" in content_type
                ):
                    csrf_text = response.text.strip()
                    return (
                        csrf_text if csrf_text else None
                    )  # Return None if empty string

                # Otherwise, return raw text
                else:
                    logger.debug(
                        f"{api_description}: OK ({status}), Content-Type '{content_type}'. Returning raw TEXT."
                    )
                    return response.text

            # Handle non-retryable error status codes (4xx, 5xx not in retry list)
            else:
                if status in [401, 403]:
                    logger.warning(
                        f"{api_description}: API call failed {status} {response.reason}. Session expired/invalid?"
                    )
                    session_manager.session_ready = False  # Mark session as not ready
                else:
                    logger.error(
                        f"{api_description}: Non-retryable error: {status} {response.reason}."
                    )
                try:
                    logger.debug(f"Error Response Body: {response.text[:500]}")
                except Exception:
                    pass
                return response  # Return the Response object for the caller to handle

        # --- Handle exceptions during the request attempt ---
        except RequestException as e:  # Catch requests-specific exceptions
            logger.debug(
                f"[_api_req Attempt {attempt} '{api_description}'] <<< requests.request raised {type(e).__name__}"
            )
            retries_left -= 1
            last_exception = e
            if retries_left <= 0:
                logger.error(
                    f"{api_description}: Request failed after {max_retries} attempts. Final Error: {e}",
                    exc_info=False,
                )
                return None  # Return None after all retries fail for network errors
            else:
                sleep_time = min(
                    current_delay * (backoff_factor ** (attempt - 1)), max_delay
                ) + random.uniform(0, 0.2)
                sleep_time = max(0.1, sleep_time)
                logger.warning(
                    f"{api_description}: {type(e).__name__} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s... Error: {e}"
                )
                time.sleep(sleep_time)
                current_delay *= backoff_factor
                continue  # Go to next attempt
        except Exception as e:  # Catch other unexpected errors during the process
            logger.critical(
                f"{api_description}: CRITICAL Unexpected error during request attempt {attempt}: {e}",
                exc_info=True,
            )
            # Do not retry on unexpected errors
            return None  # Or re-raise? Returning None seems safer for now.

    # --- Should only be reached if loop completes without success (e.g., retries exhausted) ---
    logger.error(
        f"{api_description}: Exited retry loop unexpectedly. Last Status: {getattr(response, 'status_code', 'N/A')}, Last Exception: {last_exception}."
    )
    # Return the last response if it exists (likely a failed one), otherwise None
    return response if response is not None else None


# End of _api_req


def make_ube(driver: DriverType) -> Optional[str]:
    """
    Generates the 'ancestry-context-ube' header value based on current browser state.
    Ensures the 'correlatedSessionId' matches the current 'ANCSESSIONID' cookie.
    """
    if not driver:
        return None

    try:
        # Quick check if driver is responsive - accessing title is usually safe
        _ = driver.title
    except WebDriverException as e:
        logger.warning(
            f"Cannot generate UBE header: Session invalid/unresponsive ({type(e).__name__})."
        )
        return None

    ancsessionid: Optional[str] = None
    try:
        # Try direct get_cookie first
        cookie_obj = driver.get_cookie("ANCSESSIONID")
        if cookie_obj and isinstance(cookie_obj, dict) and "value" in cookie_obj:
            ancsessionid = cookie_obj["value"]
        # Fallback: get all cookies if direct fails (might happen during page transitions)
        elif ancsessionid is None:
            cookies_dict = {
                c["name"]: c["value"]
                for c in driver.get_cookies()
                if isinstance(c, dict) and "name" in c
            }
            ancsessionid = cookies_dict.get("ANCSESSIONID")

        if not ancsessionid:
            logger.warning("ANCSESSIONID cookie not found. Cannot generate UBE header.")
            return None
    except (NoSuchCookieException, WebDriverException) as cookie_e:
        logger.warning(f"Error getting ANCSESSIONID cookie for UBE header: {cookie_e}")
        return None
    except Exception as e:  # Catch unexpected errors
        logger.error(
            f"Unexpected error getting ANCSESSIONID for UBE: {e}", exc_info=True
        )
        return None

    # Construct UBE data payload
    event_id = "00000000-0000-0000-0000-000000000000"  # Typically zeroed
    correlated_id = str(uuid.uuid4())  # Unique per request/view
    # These screen names might need updating if the UI changes significantly
    screen_name_standard = "ancestry : uk : en : dna-matches-ui : match-list : 1"
    screen_name_legacy = "ancestry uk : dnamatches-matchlistui : list"
    # Consent string format might change
    user_consent = "necessary|preference|performance|analytics1st|analytics3rd|advertising1st|advertising3rd|attribution3rd"

    ube_data = {
        "eventId": event_id,
        "correlatedScreenViewedId": correlated_id,
        "correlatedSessionId": ancsessionid,  # Link to current ANCSESSIONID
        "screenNameStandard": screen_name_standard,
        "screenNameLegacy": screen_name_legacy,
        "userConsent": user_consent,
        "vendors": "adobemc",  # Common vendor string
        "vendorConfigurations": "{}",  # Typically empty JSON object
    }

    # Encode the payload
    try:
        json_payload = json.dumps(ube_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        return encoded_payload
    except (
        json.JSONDecodeError,
        TypeError,
        base64.binascii.Error,
    ) as encode_e:  # More specific errors
        logger.error(f"Error encoding UBE header data: {encode_e}", exc_info=True)
        return None
    except Exception as e:  # Catch other unexpected errors
        logger.error(f"Unexpected error encoding UBE header: {e}", exc_info=True)
        return None


# End of make_ube


def make_newrelic(driver: DriverType) -> Optional[str]:
    """Generates the 'newrelic' header value."""
    # This function doesn't strictly need the driver, but keeps signature consistent
    try:
        # Generate IDs
        trace_id = uuid.uuid4().hex[:16]  # 16 hex chars
        span_id = uuid.uuid4().hex[:16]  # 16 hex chars

        # Common Ancestry NewRelic IDs (might change over time)
        account_id = "1690570"
        app_id = "1588726612"
        license_key_part = "2611750"  # Often referred to as 'tk' or part of license key

        newrelic_data = {
            "v": [0, 1],  # Version info
            "d": {
                "ty": "Browser",  # Type
                "ac": account_id,  # Account ID
                "ap": app_id,  # Application ID
                "id": span_id,  # Span ID for this interaction
                "tr": trace_id,  # Trace ID linking parts of a request
                "ti": int(time.time() * 1000),  # Timestamp in milliseconds
                "tk": license_key_part,  # Token/license key part
            },
        }
        # Encode payload
        json_payload = json.dumps(newrelic_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        return encoded_payload
    except (json.JSONDecodeError, TypeError, base64.binascii.Error) as encode_e:
        logger.error(f"Error generating NewRelic header: {encode_e}", exc_info=True)
        return None
    except Exception as e:  # Catch other unexpected errors
        logger.error(f"Unexpected error generating NewRelic header: {e}", exc_info=True)
        return None


# End of make_newrelic


def make_traceparent(driver: DriverType) -> Optional[str]:
    """Generates the W3C 'traceparent' header value."""
    # This function doesn't strictly need the driver
    try:
        version = "00"  # Current version
        trace_id = uuid.uuid4().hex  # 32 hex chars
        parent_id = uuid.uuid4().hex[
            :16
        ]  # 16 hex chars (span ID of the parent/requester)
        flags = "01"  # Sampled flag (01 means sampled)

        traceparent = f"{version}-{trace_id}-{parent_id}-{flags}"
        return traceparent
    except Exception as e:
        logger.error(f"Error generating traceparent header: {e}", exc_info=True)
        return None


# End of make_traceparent


def make_tracestate(driver: DriverType) -> Optional[str]:
    """Generates the W3C 'tracestate' header value, including NewRelic info."""
    # This function doesn't strictly need the driver
    try:
        # NewRelic specific part of tracestate
        # Use same IDs as in make_newrelic for consistency
        tk = "2611750"
        account_id = "1690570"
        app_id = "1588726612"
        span_id = uuid.uuid4().hex[:16]  # Generate a span ID for this state
        timestamp = int(time.time() * 1000)

        # Format follows NewRelic's convention: tk@nr=...
        # 0-1 = Version info? Priority?
        # ---- = Placeholder for parent info (usually empty for browser-initiated)
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
) -> Tuple[str, Optional[str]]:  # Return status string and optional conversation ID
    """
    Sends a message using the appropriate Ancestry messaging API endpoint.
    Handles both initial messages and replies to existing conversations.

    Returns:
        Tuple containing a status string (e.g., SEND_SUCCESS_DELIVERED, SEND_ERROR_*)
        and the conversation ID (new or existing, or None on failure).
    """
    # --- Input Validation ---
    if not session_manager or not session_manager.my_profile_id:
        logger.error(
            f"{log_prefix}: Cannot send message - SessionManager or own profile ID missing."
        )
        return SEND_ERROR_MISSING_OWN_ID, None
    # Ensure person object is valid and has profile_id
    if not isinstance(person, Person) or not person.profile_id:
        logger.error(
            f"{log_prefix}: Cannot send message - Invalid Person object or missing profile ID."
        )
        return SEND_ERROR_INVALID_RECIPIENT, None
    if not isinstance(message_text, str) or not message_text.strip():
        logger.error(
            f"{log_prefix}: Cannot send message - Message text is empty or invalid."
        )
        return SEND_ERROR_API_PREP_FAILED, None  # Reusing status, indicates bad input

    # --- Mode Check ---
    app_mode = config_instance.APP_MODE if config_instance else "unknown"
    if app_mode == "dry_run":
        message_status = SEND_SUCCESS_DRY_RUN
        # Generate a fake conv ID for dry run consistency if needed
        effective_conv_id = existing_conv_id or f"dryrun_{uuid.uuid4()}"
        logger.info(
            f"{log_prefix}: Dry Run - Simulated message send to {person.username or person.profile_id}."
        )
        return message_status, effective_conv_id
    elif app_mode not in ["production", "testing"]:
        logger.error(
            f"{log_prefix}: Logic Error - Unexpected APP_MODE '{app_mode}' reached send logic."
        )
        return SEND_ERROR_INTERNAL_MODE, None

    # --- Prepare API Call ---
    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()
    MY_PROFILE_ID_UPPER = session_manager.my_profile_id.upper()
    recipient_profile_id_upper = person.profile_id.upper()

    is_initial = not existing_conv_id
    send_api_url: str = ""
    payload: Dict[str, Any] = {}
    send_api_desc: str = ""
    api_headers: Dict[str, Any] = {}

    try:
        if is_initial:
            # Create new conversation
            send_api_url = urljoin(
                config_instance.BASE_URL.rstrip("/") + "/", API_PATH_SEND_MESSAGE_NEW
            )
            send_api_desc = "Create Conversation API"
            payload = {
                "content": message_text,
                "author": MY_PROFILE_ID_LOWER,  # API seems to expect lowercase here
                "index": 0,  # Standard value for new message
                "created": 0,  # API seems to ignore this, server sets timestamp
                "conversation_members": [
                    {
                        "user_id": recipient_profile_id_upper.lower(),
                        "family_circles": [],
                    },
                    {"user_id": MY_PROFILE_ID_LOWER},
                ],
            }
        elif existing_conv_id:
            # Send message to existing conversation
            formatted_path = API_PATH_SEND_MESSAGE_EXISTING.format(
                conv_id=existing_conv_id
            )
            send_api_url = urljoin(
                config_instance.BASE_URL.rstrip("/") + "/", formatted_path
            )
            send_api_desc = "Send Message API (Existing Conv)"
            payload = {
                "content": message_text,
                "author": MY_PROFILE_ID_LOWER,  # API expects lowercase author
            }
        else:
            # Should not happen if logic is correct
            logger.error(
                f"{log_prefix}: Logic Error - Cannot determine API URL/payload (existing_conv_id issue?)."
            )
            return SEND_ERROR_API_PREP_FAILED, None

        # Get contextual headers
        ctx_headers = config_instance.API_CONTEXTUAL_HEADERS.get(send_api_desc, {})
        api_headers = ctx_headers.copy()

    except Exception as prep_err:  # Catch errors during URL formatting/payload creation
        logger.error(
            f"{log_prefix}: Error preparing API request data: {prep_err}", exc_info=True
        )
        return SEND_ERROR_API_PREP_FAILED, None

    # --- Make API Call ---
    api_response = _api_req(
        url=send_api_url,
        driver=session_manager.driver,  # Pass driver for potential header needs
        session_manager=session_manager,
        method="POST",
        json_data=payload,
        use_csrf_token=False,  # Messaging API usually doesn't require CSRF
        headers=api_headers,
        api_description=send_api_desc,
    )

    # --- Process API Response ---
    message_status = SEND_ERROR_UNKNOWN  # Default status
    new_conversation_id_from_api: Optional[str] = None
    post_ok = False

    if api_response is None:
        # _api_req failed after retries (network/server error)
        message_status = SEND_ERROR_POST_FAILED
        logger.error(
            f"{log_prefix}: API POST ({send_api_desc}) failed (No response/Retries exhausted)."
        )
    elif isinstance(api_response, requests.Response):
        # Got a Response object, indicates non-retryable HTTP error
        message_status = f"send_error (http_{api_response.status_code})"
        logger.error(
            f"{log_prefix}: API POST ({send_api_desc}) failed with status {api_response.status_code}."
        )
        try:
            logger.debug(f"Error response body: {api_response.text[:500]}")
        except Exception:
            pass
    elif isinstance(api_response, dict):
        # Got a dictionary, indicates 2xx success from _api_req
        try:
            if is_initial:
                # Expecting {"conversation_id": "...", "message": {"author": "..."}}
                api_conv_id = str(api_response.get(KEY_CONVERSATION_ID, ""))
                msg_details = api_response.get(KEY_MESSAGE, {})
                api_author = (
                    str(msg_details.get(KEY_AUTHOR, "")).upper()
                    if isinstance(msg_details, dict)
                    else None
                )

                if api_conv_id and api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = api_conv_id
                else:
                    logger.error(
                        f"{log_prefix}: API initial response format invalid (ConvID: '{api_conv_id}', Author: '{api_author}', Expected Author: '{MY_PROFILE_ID_UPPER}')."
                    )
                    logger.debug(f"API Response: {api_response}")
                    message_status = SEND_ERROR_VALIDATION_FAILED
            else:
                # Expecting {"author": "..."} for reply
                api_author = str(api_response.get(KEY_AUTHOR, "")).upper()
                if api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = (
                        existing_conv_id  # Use the ID we sent to
                    )
                else:
                    logger.error(
                        f"{log_prefix}: API follow-up author validation failed (Author: '{api_author}', Expected Author: '{MY_PROFILE_ID_UPPER}')."
                    )
                    logger.debug(f"API Response: {api_response}")
                    message_status = SEND_ERROR_VALIDATION_FAILED

            if post_ok:
                message_status = SEND_SUCCESS_DELIVERED
                logger.info(
                    f"{log_prefix}: Message send to {person.username or person.profile_id} successful (ConvID: {new_conversation_id_from_api})."
                )

        except Exception as parse_err:  # Catch errors parsing the success response
            logger.error(
                f"{log_prefix}: Error parsing successful API response ({send_api_desc}): {parse_err}",
                exc_info=True,
            )
            logger.debug(f"API Response received: {api_response}")
            message_status = SEND_ERROR_UNEXPECTED_FORMAT

    else:
        # _api_req returned something unexpected (e.g., string when JSON expected)
        logger.error(
            f"{log_prefix}: API call ({send_api_desc}) unexpected success format. Type:{type(api_response)}, Resp:{api_response}"
        )
        message_status = SEND_ERROR_UNEXPECTED_FORMAT

    # Final check on status if post_ok was never set True
    if not post_ok and message_status == SEND_ERROR_UNKNOWN:
        message_status = SEND_ERROR_VALIDATION_FAILED  # Set a more specific error if validation failed silently
        logger.warning(
            f"{log_prefix}: Message send attempt concluded with status: {message_status}"
        )

    return message_status, new_conversation_id_from_api


# End of _send_message_via_api


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_profile_details_for_person(
    session_manager: SessionManager, profile_id: str
) -> Optional[Dict[str, Any]]:
    """
    Fetches profile details (FirstName, LastLoginDate, IsContactable) for a specific Person using their Profile ID.

    Returns:
        Dictionary with keys 'first_name', 'last_logged_in_dt', 'contactable',
        or None if fetching fails or profile_id is invalid.
    """
    if not profile_id or not isinstance(profile_id, str):
        logger.warning(
            "_fetch_profile_details_for_person: Profile ID missing or invalid."
        )
        return None
    if not session_manager or not session_manager.my_profile_id:
        logger.error(
            "_fetch_profile_details_for_person: SessionManager or own profile ID missing."
        )
        return None  # Cannot proceed

    # Check session validity before making API call
    if not session_manager.is_sess_valid():
        logger.error(
            f"_fetch_profile_details_for_person: Session invalid for Profile ID {profile_id}."
        )
        # Don't raise ConnectionError here, let retry_api handle potential transient issues if called externally.
        # If called internally, the caller should handle the invalid session state.
        return None  # Indicate failure due to invalid session

    api_description = "Profile Details API (Single)"  # More specific description
    profile_url = urljoin(
        config_instance.BASE_URL,
        f"{API_PATH_PROFILE_DETAILS}?userId={profile_id.upper()}",
    )
    referer_url = urljoin(
        config_instance.BASE_URL, "/messaging/"
    )  # Common referer for messaging context

    logger.debug(
        f"Fetching profile details ({api_description}) for Profile ID {profile_id}..."
    )

    try:
        profile_response: ApiResponseType = _api_req(
            url=profile_url,
            driver=session_manager.driver,  # Pass driver if needed for headers
            session_manager=session_manager,
            method="GET",
            headers={},  # No specific overrides needed usually
            use_csrf_token=False,  # Generally not needed for profile details GET
            api_description=api_description,
            referer_url=referer_url,
        )

        if profile_response and isinstance(profile_response, dict):
            logger.debug(f"Successfully fetched profile details for {profile_id}.")
            result_data: Dict[str, Any] = {
                "first_name": None,
                "last_logged_in_dt": None,
                "contactable": False,
            }

            # --- Extract FirstName ---
            # Use constant for key
            first_name_raw = profile_response.get(KEY_FIRST_NAME)
            if first_name_raw and isinstance(first_name_raw, str):
                result_data["first_name"] = format_name(first_name_raw)
            else:
                # Fallback to DisplayName if FirstName is missing
                display_name_raw = profile_response.get(KEY_DISPLAY_NAME)
                if display_name_raw and isinstance(display_name_raw, str):
                    formatted_dn = format_name(display_name_raw)
                    # Try to get first word as first name
                    result_data["first_name"] = (
                        formatted_dn.split()[0]
                        if formatted_dn != "Valued Relative"
                        else None
                    )
                else:
                    # Log if both are missing
                    logger.warning(
                        f"Could not extract FirstName or DisplayName for profile {profile_id}"
                    )

            # --- Extract IsContactable ---
            contactable_val = profile_response.get(KEY_IS_CONTACTABLE)
            result_data["contactable"] = (
                bool(contactable_val) if contactable_val is not None else False
            )

            # --- Extract LastLoginDate ---
            last_login_str = profile_response.get(KEY_LAST_LOGIN_DATE)
            if last_login_str and isinstance(last_login_str, str):
                try:
                    # Handle ISO 8601 format, assuming UTC if 'Z' present or no offset specified
                    if last_login_str.endswith("Z"):
                        dt_aware = datetime.fromisoformat(
                            last_login_str.replace("Z", "+00:00")
                        )
                    elif (
                        "+" in last_login_str or "-" in last_login_str[10:]
                    ):  # Check for explicit offset
                        dt_aware = datetime.fromisoformat(last_login_str)
                    else:  # Assume UTC if no offset info
                        dt_naive = datetime.fromisoformat(last_login_str)
                        dt_aware = dt_naive.replace(tzinfo=timezone.utc)

                    # Ensure it's UTC
                    result_data["last_logged_in_dt"] = dt_aware.astimezone(timezone.utc)

                except (ValueError, TypeError) as date_parse_err:
                    logger.warning(
                        f"Could not parse LastLoginDate '{last_login_str}' for {profile_id}: {date_parse_err}"
                    )
                    # Keep default None
            else:
                logger.debug(
                    f"LastLoginDate missing or invalid for profile {profile_id}"
                )

            return result_data

        # Handle non-dict responses (e.g., Response object on error, None on retry failure)
        elif isinstance(profile_response, requests.Response):
            logger.warning(
                f"Failed profile details fetch for {profile_id}. Status: {profile_response.status_code}."
            )
            return None
        elif profile_response is None:
            logger.warning(
                f"Failed profile details fetch for {profile_id} (_api_req returned None)."
            )
            return None
        else:
            logger.warning(
                f"Failed profile details fetch for {profile_id} (Invalid response type: {type(profile_response)})."
            )
            return None

    except (
        RequestException
    ) as req_e:  # Catch connection/request errors if decorator fails/bypassed
        logger.error(
            f"RequestException fetching profile details for {profile_id}: {req_e}",
            exc_info=False,
        )
        return None  # Indicate failure
    except Exception as e:  # Catch other unexpected errors
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
TWO_STEP_VERIFICATION_HEADER_SELECTOR = (
    "h1.two-step-verification-header"  # Example selector
)


# Login Helper 5
@time_wait("Handle 2FA Page")
def handle_twoFA(session_manager: SessionManager) -> bool:
    """Handles the Two-Factor Authentication page interaction."""
    if session_manager.driver is None:
        logger.error("handle_twoFA: SessionManager driver is None. Cannot proceed.")
        return False
    driver = session_manager.driver
    element_wait = selenium_config.element_wait(driver)
    page_wait = selenium_config.page_wait(driver)
    short_wait = selenium_config.short_wait(driver)

    try:
        logger.debug("Handling Two-Factor Authentication (2FA)...")

        # --- Wait for 2FA page header ---
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
            # Check if maybe we logged in successfully anyway (e.g., remembered device)
            if login_status(session_manager) is True:
                logger.info(
                    "User appears logged in after checking for 2FA page. Assuming 2FA handled/skipped."
                )
                return True
            logger.warning(
                "Assuming 2FA not required or page didn't load correctly (header missing)."
            )
            return False  # Cannot proceed if header missing and not logged in
        except WebDriverException as e:
            logger.error(f"WebDriverException waiting for 2FA header: {e}")
            return False

        # --- Click 'Send Code' via SMS (example) ---
        try:
            logger.debug(
                f"Waiting for 2FA 'Send Code' (SMS) button: '{TWO_FA_SMS_SELECTOR}'"
            )
            # Wait for clickable, short timeout
            sms_button_clickable = short_wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR))
            )

            if sms_button_clickable:
                logger.debug(
                    "Attempting to click 'Send Code' button using JavaScript..."
                )
                try:
                    driver.execute_script("arguments[0].click();", sms_button_clickable)
                    logger.debug("'Send Code' button clicked.")
                except WebDriverException as click_err:
                    logger.error(
                        f"Error clicking 'Send Code' button via JS: {click_err}"
                    )
                    return False

                # Optional: Wait briefly for code input field to appear after click
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
                except WebDriverException as e_input:
                    logger.error(
                        f"Error waiting for 2FA code input field: {e_input}. Check selector: {TWO_FA_CODE_INPUT_SELECTOR}"
                    )
            else:
                # Should not happen if wait succeeded, but good practice
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
        except WebDriverException as e:
            logger.error(
                f"WebDriverException clicking 2FA 'Send Code' button: {e}",
                exc_info=False,
            )
            return False
        except Exception as e:  # Catch other unexpected errors
            logger.error(
                f"Unexpected error clicking 2FA 'Send Code' button: {e}", exc_info=True
            )
            return False

        # --- Wait for User Action ---
        code_entry_timeout = selenium_config.TWO_FA_CODE_ENTRY_TIMEOUT
        logger.warning(
            f"Waiting up to {code_entry_timeout}s for user to manually enter 2FA code and submit..."
        )
        start_time = time.time()
        user_action_detected = False
        while time.time() - start_time < code_entry_timeout:
            try:
                # Check if the 2FA header is *still* visible. If not, user likely submitted.
                WebDriverWait(driver, 0.5).until_not(
                    EC.visibility_of_element_located(
                        (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                    )
                )
                logger.info(
                    "2FA page elements disappeared, assuming user submitted code."
                )
                user_action_detected = True
                break  # Exit loop
            except TimeoutException:
                # Header still visible, continue waiting
                time.sleep(2)  # Wait before checking again
            except NoSuchElementException:  # Header gone immediately
                logger.info("2FA header element no longer present.")
                user_action_detected = True
                break
            except WebDriverException as e:
                # Handle potential errors during the check
                logger.error(
                    f"WebDriver error checking for 2FA header during wait: {e}"
                )
                break  # Exit loop on error
            except Exception as e:  # Catch other unexpected errors
                logger.error(
                    f"Unexpected error checking for 2FA header during wait: {e}"
                )
                break

        # --- Verify Outcome ---
        if user_action_detected:
            logger.info("Re-checking login status after potential 2FA submission...")
            time.sleep(1)  # Allow page to potentially redirect/settle
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
            # Loop timed out
            logger.error(
                f"Timed out ({code_entry_timeout}s) waiting for user 2FA action (page did not change)."
            )
            return False

    except WebDriverException as e:  # Catch WebDriver errors during the overall process
        logger.error(f"WebDriverException during 2FA handling: {e}")
        if not is_browser_open(driver):
            logger.error("Session invalid after WebDriverException during 2FA.")
        return False
    except Exception as e:  # Catch other unexpected errors
        logger.error(f"Unexpected error during 2FA handling: {e}", exc_info=True)
        return False
    # Fallback return, should ideally be covered by logic above
    return False


# End of handle_twoFA


# Login Helper 4
def enter_creds(driver: WebDriver) -> bool:
    """Enters username and password into login form fields and clicks sign in."""
    element_wait = selenium_config.element_wait(driver)
    short_wait = selenium_config.short_wait(driver)
    time.sleep(random.uniform(0.5, 1.0))  # Small random delay before interaction

    try:
        logger.debug("Entering Credentials and Signing In...")

        # --- Enter Username ---
        logger.debug(f"Waiting for username input: '{USERNAME_INPUT_SELECTOR}'...")
        username_input = element_wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))
        )
        logger.debug("Username input field found.")
        try:
            # Clear field robustly before sending keys
            username_input.click()
            time.sleep(0.1)
            username_input.clear()
            time.sleep(0.1)
            # Try JS clear as fallback
            driver.execute_script("arguments[0].value = '';", username_input)
            time.sleep(0.1)
        except (ElementNotInteractableException, StaleElementReferenceException) as e:
            logger.warning(
                f"Issue clicking/clearing username field ({e}). Proceeding cautiously."
            )
        except WebDriverException as e:
            logger.error(
                f"WebDriverException clicking/clearing username: {e}. Aborting."
            )
            return False

        ancestry_username = config_instance.ANCESTRY_USERNAME
        if not ancestry_username:
            raise ValueError("ANCESTRY_USERNAME configuration is missing.")
        logger.debug("Entering username...")
        username_input.send_keys(ancestry_username)
        logger.debug("Username entered.")
        time.sleep(random.uniform(0.2, 0.4))

        # --- Enter Password ---
        logger.debug(f"Waiting for password input: '{PASSWORD_INPUT_SELECTOR}'...")
        password_input = element_wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))
        )
        logger.debug("Password input field found.")
        try:
            # Clear field robustly
            password_input.click()
            time.sleep(0.1)
            password_input.clear()
            time.sleep(0.1)
            driver.execute_script("arguments[0].value = '';", password_input)
            time.sleep(0.1)
        except (ElementNotInteractableException, StaleElementReferenceException) as e:
            logger.warning(
                f"Issue clicking/clearing password field ({e}). Proceeding cautiously."
            )
        except WebDriverException as e:
            logger.error(
                f"WebDriverException clicking/clearing password: {e}. Aborting."
            )
            return False

        ancestry_password = config_instance.ANCESTRY_PASSWORD
        if not ancestry_password:
            raise ValueError("ANCESTRY_PASSWORD configuration is missing.")
        logger.debug("Entering password: ***")
        password_input.send_keys(ancestry_password)
        logger.debug("Password entered.")
        time.sleep(random.uniform(0.3, 0.6))

        # --- Click Sign In Button ---
        sign_in_button = None
        try:
            logger.debug(
                f"Waiting for sign in button presence: '{SIGN_IN_BUTTON_SELECTOR}'..."
            )
            # Wait for presence first, then clickability
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
                return True  # Assume submission if key sent without error
            except (WebDriverException, ElementNotInteractableException) as key_e:
                logger.error(f"Failed to send RETURN key: {key_e}")
                return False
        except WebDriverException as find_e:  # Catch other find errors
            logger.error(f"Unexpected WebDriver error finding sign in button: {find_e}")
            return False

        # Attempt to click the button found
        click_successful = False
        if sign_in_button:
            # Try standard click first
            try:
                logger.debug("Attempting standard click on sign in button...")
                sign_in_button.click()
                logger.debug("Standard click executed.")
                click_successful = True
            except (
                ElementClickInterceptedException,
                ElementNotInteractableException,
                StaleElementReferenceException,
            ) as click_intercept_err:
                logger.warning(
                    f"Standard click failed ({type(click_intercept_err).__name__}). Trying JS click..."
                )
            except (
                WebDriverException
            ) as click_err:  # Catch other webdriver errors during click
                logger.error(
                    f"WebDriver error during standard click: {click_err}. Trying JS click..."
                )

            # Try JavaScript click if standard failed
            if not click_successful:
                try:
                    logger.debug("Attempting JavaScript click on sign in button...")
                    driver.execute_script("arguments[0].click();", sign_in_button)
                    logger.info("JavaScript click executed.")
                    click_successful = True
                except WebDriverException as js_click_e:
                    logger.error(f"Error during JavaScript click: {js_click_e}")

            # Fallback: Send RETURN key if both clicks failed
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
                except (WebDriverException, ElementNotInteractableException) as key_e:
                    logger.error(
                        f"Failed to send RETURN key as final fallback: {key_e}"
                    )

        return click_successful  # Return True if any click/key method seemed to work

    # Catch errors during the process
    except (TimeoutException, NoSuchElementException) as e:
        logger.error(
            f"Timeout or Element not found finding username/password field: {e}"
        )
        return False
    except ValueError as ve:  # Config error (missing username/password)
        logger.critical(f"Configuration Error: {ve}")
        return False
    except WebDriverException as e:
        logger.error(f"WebDriver error entering credentials: {e}")
        if not is_browser_open(driver):
            logger.error("Session invalid during credential entry.")
        return False
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"Unexpected error entering credentials: {e}", exc_info=True)
        return False


# End of enter_creds


# Login Helper 3
@retry(MAX_RETRIES=2, BACKOFF_FACTOR=1, MAX_DELAY=3)  # Add retry for consent handling
def consent(driver: WebDriver) -> bool:
    """Handles the cookie consent banner if present."""
    if not driver:
        logger.error("consent: WebDriver instance is None.")
        return False

    logger.debug(f"Checking for cookie consent overlay: '{COOKIE_BANNER_SELECTOR}'")
    overlay_element = None
    try:
        # Use a short wait to find the banner
        overlay_element = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR))
        )
        logger.debug("Cookie consent overlay DETECTED.")
    except TimeoutException:
        logger.debug("Cookie consent overlay not found. Assuming no consent needed.")
        return True  # No banner, proceed
    except WebDriverException as e:  # Catch errors finding element
        logger.error(f"Error checking for consent banner: {e}")
        return False  # Indicate failure if check fails

    # If overlay detected, try to handle it
    removed_via_js = False
    if overlay_element:
        # Attempt 1: Try removing the element directly with JS
        try:
            logger.debug("Attempting JS removal of consent overlay...")
            driver.execute_script("arguments[0].remove();", overlay_element)
            time.sleep(0.5)  # Allow DOM to update
            # Verify removal
            try:
                WebDriverWait(driver, 1).until_not(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                    )
                )
                logger.debug("Cookie consent overlay REMOVED successfully via JS.")
                removed_via_js = True
                return True  # Success
            except TimeoutException:
                logger.warning(
                    "Consent overlay still present after JS removal attempt."
                )
            except WebDriverException as verify_err:
                logger.warning(
                    f"Error verifying overlay removal after JS: {verify_err}"
                )

        except WebDriverException as js_err:
            logger.warning(
                f"Error removing consent overlay via JS: {js_err}. Trying button click..."
            )
        except Exception as e:  # Catch other unexpected errors during JS removal
            logger.warning(
                f"Unexpected error during JS removal of consent: {e}. Trying button click..."
            )

    # Attempt 2: Try clicking the specific accept button if JS removal failed/skipped
    if not removed_via_js:
        logger.debug(
            f"JS removal failed/skipped. Trying specific accept button: '{consent_ACCEPT_BUTTON_SELECTOR}'"
        )
        try:
            # Wait for the button to be clickable
            accept_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, consent_ACCEPT_BUTTON_SELECTOR)
                )
            )
            logger.info("Found specific clickable accept button.")

            # Try standard click first
            try:
                accept_button.click()
                logger.info("Clicked accept button successfully.")
                # Verify removal
                WebDriverWait(driver, 2).until_not(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                    )
                )
                logger.debug("Consent overlay gone after clicking accept button.")
                return True  # Success
            except ElementClickInterceptedException:
                logger.warning(
                    "Click intercepted for accept button, trying JS click..."
                )
                # Fall through to JS click attempt
            except (
                TimeoutException,
                NoSuchElementException,
            ):  # If overlay gone after click
                logger.debug(
                    "Consent overlay likely gone after standard click (verification timed out/not found)."
                )
                return True
            except WebDriverException as click_err:
                logger.error(
                    f"Error during standard click on accept button: {click_err}. Trying JS click..."
                )

            # Try JS click as fallback
            try:
                logger.debug("Attempting JS click on accept button...")
                driver.execute_script("arguments[0].click();", accept_button)
                logger.info("Clicked accept button via JS successfully.")
                # Verify removal
                WebDriverWait(driver, 2).until_not(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)
                    )
                )
                logger.debug("Consent overlay gone after JS clicking accept button.")
                return True  # Success
            except (
                TimeoutException,
                NoSuchElementException,
            ):  # If overlay gone after JS click
                logger.debug(
                    "Consent overlay likely gone after JS click (verification timed out/not found)."
                )
                return True
            except WebDriverException as js_click_err:
                logger.error(f"Failed JS click for accept button: {js_click_err}")

        except TimeoutException:
            logger.warning(
                f"Specific accept button '{consent_ACCEPT_BUTTON_SELECTOR}' not found or not clickable."
            )
        except WebDriverException as find_err:  # Catch errors finding/interacting
            logger.error(f"Error finding/clicking specific accept button: {find_err}")
        except Exception as e:  # Catch other unexpected errors
            logger.error(
                f"Unexpected error handling consent button: {e}", exc_info=True
            )

    # If both JS removal and button click failed
    logger.error("Could not remove consent overlay via JS or button click.")
    return False


# End of consent


# Login Main Function 2
def log_in(session_manager: SessionManager) -> str:
    """
    Automates the login process: navigates to signin, handles consent,
    enters credentials, handles 2FA, and verifies final login status.

    Returns:
        A status string indicating success ("LOGIN_SUCCEEDED") or failure type.
    """
    driver = session_manager.driver
    if not driver:
        logger.error("Login failed: WebDriver not available in SessionManager.")
        return "LOGIN_ERROR_NO_DRIVER"

    signin_url = urljoin(config_instance.BASE_URL, "account/signin")

    try:
        # --- Step 1: Navigate to Sign-in Page ---
        logger.info(f"Navigating to sign-in page: {signin_url}")
        # Wait for username input as indication of page load
        if not nav_to_page(
            driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager
        ):
            # Navigation failed or redirected. Check if already logged in.
            logger.debug(
                "Navigation to sign-in page failed/redirected. Checking login status..."
            )
            current_status = login_status(session_manager)
            if current_status is True:
                logger.info(
                    "Detected as already logged in during navigation attempt. Login considered successful."
                )
                return "LOGIN_SUCCEEDED"
            else:
                logger.error("Failed to navigate to login page (and not logged in).")
                return "LOGIN_FAILED_NAVIGATION"
        logger.debug("Successfully navigated to sign-in page.")

        # --- Step 2: Handle Consent Banner ---
        if not consent(driver):
            logger.warning("Failed to handle consent banner, login might be impacted.")
            # Continue anyway, maybe it wasn't essential

        # --- Step 3: Enter Credentials ---
        if not enter_creds(driver):
            logger.error("Failed during credential entry or submission.")
            # Check for specific error messages on the page
            try:
                # Check for specific 'invalid credentials' message
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
                # Check for any generic alert box
                generic_alert_selector = "div.alert[role='alert']"  # Example
                try:
                    alert_element = WebDriverWait(driver, 0.5).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, generic_alert_selector)
                        )
                    )
                    alert_text = (
                        alert_element.text
                        if alert_element and alert_element.text
                        else "Unknown error"
                    )
                    logger.error(f"Login failed: Generic alert found: '{alert_text}'.")
                    return "LOGIN_FAILED_ERROR_DISPLAYED"
                except TimeoutException:
                    logger.error(
                        "Login failed: Credential entry failed, but no specific or generic alert found."
                    )
                    return "LOGIN_FAILED_CREDS_ENTRY"  # Credential entry itself failed
                except (
                    WebDriverException
                ) as alert_err:  # Handle errors checking for alerts
                    logger.warning(
                        f"Error checking for generic login error message: {alert_err}"
                    )
                    return "LOGIN_FAILED_CREDS_ENTRY"  # Assume cred entry failed
            except (
                WebDriverException
            ) as alert_err:  # Handle errors checking for specific alert
                logger.warning(
                    f"Error checking for specific login error message: {alert_err}"
                )
                return "LOGIN_FAILED_CREDS_ENTRY"

        # --- Step 4: Wait and Check for 2FA ---
        logger.debug("Credentials submitted. Waiting for potential page change...")
        time.sleep(random.uniform(3.0, 5.0))  # Allow time for redirect or 2FA page load

        two_fa_present = False
        try:
            # Check if the 2FA header is now visible
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
            # If error checking, verify login status as fallback
            status = login_status(session_manager)
            if status is True:
                return "LOGIN_SUCCEEDED"
            elif status is False:
                return "LOGIN_FAILED_UNKNOWN"  # Error + not logged in
            else:
                return "LOGIN_FAILED_STATUS_CHECK_ERROR"  # Critical status check error

        # --- Step 5: Handle 2FA or Verify Login ---
        if two_fa_present:
            if handle_twoFA(session_manager):
                logger.info("Two-step verification handled successfully.")
                # Re-verify login status after 2FA
                if login_status(session_manager) is True:
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
            # No 2FA detected, check login status directly
            logger.debug("Checking login status directly (no 2FA detected)...")
            login_check_result = login_status(session_manager)
            if login_check_result is True:
                logger.info("Direct login check successful.")
                return "LOGIN_SUCCEEDED"
            elif login_check_result is False:
                # Verify why it failed if no 2FA was shown
                logger.error(
                    "Direct login check failed. Checking for error messages again..."
                )
                try:
                    # Check specific error again
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
                    # Check generic error again
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
                        # Still on login page?
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
                            return "LOGIN_FAILED_WEBDRIVER"  # Session likely dead
                    except (
                        WebDriverException
                    ) as alert_err:  # Error checking generic alert
                        logger.error(
                            f"Login failed: Error checking for generic alert (post-check): {alert_err}"
                        )
                        return "LOGIN_FAILED_UNKNOWN"
                except WebDriverException as alert_err:  # Error checking specific alert
                    logger.error(
                        f"Login failed: Error checking for specific alert (post-check): {alert_err}"
                    )
                    return "LOGIN_FAILED_UNKNOWN"
            else:  # login_status returned None
                logger.error(
                    "Login failed: Critical error during final login status check."
                )
                return "LOGIN_FAILED_STATUS_CHECK_ERROR"

    # --- Catch errors during the overall login process ---
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
@retry(MAX_RETRIES=2)  # Add retry in case of transient UI issues during check
def login_status(session_manager: SessionManager) -> Optional[bool]:
    """
    Checks if the user is currently logged in. Prioritizes API check, falls back to UI check.

    Returns:
        True if logged in, False if not logged in, None if the check fails critically.
    """
    logger.debug("Checking login status (API prioritized)...")
    if not isinstance(session_manager, SessionManager):
        logger.error(
            f"Invalid argument: Expected SessionManager, got {type(session_manager)}."
        )
        return None  # Critical argument error
    if not session_manager.is_sess_valid():
        logger.debug("Login status check: Session invalid or browser closed.")
        return False  # Cannot be logged in if session is invalid
    driver = session_manager.driver
    if driver is None:
        logger.error("Login status check: Driver is None within SessionManager.")
        return None  # Critical state error

    # --- Attempt 1: API Check ---
    logger.debug("Attempting API login verification (_verify_api_login_status)...")
    api_check_result = session_manager._verify_api_login_status()

    if api_check_result is True:
        logger.debug("Login status confirmed TRUE via API check.")
        return True
    elif api_check_result is False:
        logger.debug(
            "API check indicates user NOT logged in. Proceeding to UI check as confirmation."
        )
        # Fall through to UI check
    else:  # api_check_result is None
        logger.warning(
            "API login check returned None (error during check). Falling back to UI check."
        )
        # Fall through to UI check

    # --- Attempt 2: UI Check (Fallback) ---
    logger.debug("Performing fallback UI login check...")
    try:
        # Check 1: Absence of login button (a strong indicator when logged in)
        login_button_selector = LOG_IN_BUTTON_SELECTOR  # Assumes this selector exists
        logger.debug(
            f"UI Check Step 1: Checking ABSENCE of login button: '{login_button_selector}'"
        )
        login_button_present = False
        try:
            # Use a short wait to check for visibility
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
        except WebDriverException as e:  # Handle errors during check
            logger.warning(f"Error checking for login button presence: {e}")
            # If we can't check for login button, rely on logged-in element check

        # If login button is present, definitely not logged in
        if login_button_present:
            logger.debug(
                "Login status confirmed FALSE via UI check (login button found)."
            )
            return False

        # Check 2: Presence of a known logged-in element (if login button absent)
        logged_in_selector = (
            CONFIRMED_LOGGED_IN_SELECTOR  # Assumes this selector exists
        )
        logger.debug(
            f"UI Check Step 2: Checking PRESENCE of logged-in element: '{logged_in_selector}'"
        )
        # Use helper function is_elem_there for robust check
        ui_element_present = is_elem_there(
            driver, By.CSS_SELECTOR, logged_in_selector, wait=3
        )

        if ui_element_present:
            logger.debug(
                "Login status confirmed TRUE via UI check (login button absent AND logged-in element found)."
            )
            return True
        else:
            # Login button absent, logged-in element also absent. Ambiguous state.
            current_url_context = "Unknown"
            try:
                current_url_context = driver.current_url
            except Exception:
                pass
            logger.warning(
                f"Login status ambiguous: API failed/false, UI elements inconclusive at URL: {current_url_context}"
            )
            # Default to False if UI check is ambiguous after API check indicated False or failed
            return False

    except WebDriverException as e:
        logger.error(f"WebDriverException during UI login_status check: {e}")
        if not is_browser_open(driver):
            logger.error("Session became invalid during UI login_status check.")
            session_manager.close_sess()  # Close the dead session
        return None  # Return None on critical WebDriver error during check
    except Exception as e:  # Catch other unexpected errors
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
    selector: str = "body",  # CSS selector to wait for as indication of page load success
    session_manager: SessionManagerType = None,  # Pass SessionManager for context/restart
) -> bool:
    """
    Navigates the WebDriver to a given URL, waits for a specific element,
    and handles common issues like redirects, unavailability messages, and session restarts.

    Args:
        driver: The WebDriver instance.
        url: The target URL to navigate to.
        selector: A CSS selector for an element expected on the target page.
                  Defaults to 'body'. Used to verify successful navigation.
        session_manager: The active SessionManager instance (optional, needed for restart).

    Returns:
        True if navigation succeeded and the selector was found, False otherwise.
    """
    if not driver:
        logger.error("Navigation failed: WebDriver instance is None.")
        return False
    if not url or not isinstance(url, str):
        logger.error(f"Navigation failed: Target URL '{url}' is invalid.")
        return False

    max_attempts = getattr(config_instance, "MAX_RETRIES", 3) if config_instance else 3
    page_timeout = (
        getattr(selenium_config, "PAGE_TIMEOUT", 40) if selenium_config else 40
    )
    element_timeout = (
        getattr(selenium_config, "ELEMENT_TIMEOUT", 20) if selenium_config else 20
    )

    # Normalize target URL base (scheme, netloc, path) for comparison
    try:
        target_url_parsed = urlparse(url)
        target_url_base = urlunparse(
            (
                target_url_parsed.scheme,
                target_url_parsed.netloc,
                target_url_parsed.path.rstrip("/"),
                "",
                "",
                "",
            )
        ).rstrip("/")
    except ValueError as url_parse_err:
        logger.error(f"Failed to parse target URL '{url}': {url_parse_err}")
        return False

    # Define common problematic URLs/selectors
    signin_page_url_base = (
        urljoin(config_instance.BASE_URL, "account/signin").rstrip("/")
        if config_instance
        else ""
    )
    mfa_page_url_base = (
        urljoin(config_instance.BASE_URL, "account/signin/mfa/").rstrip("/")
        if config_instance
        else ""
    )
    # Selectors for known 'unavailable' pages
    unavailability_selectors = {
        TEMP_UNAVAILABLE_SELECTOR: ("refresh", 5),  # Selector : (action, wait_seconds)
        PAGE_NO_LONGER_AVAILABLE_SELECTOR: ("skip", 0),
        # Add other known error page selectors here
    }

    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Navigation Attempt {attempt}/{max_attempts} to: {url}")
        landed_url = ""
        landed_url_base = ""

        try:
            # --- Pre-Navigation Checks ---
            if not is_browser_open(driver):
                logger.error(
                    f"Navigation failed (Attempt {attempt}): Browser session invalid before nav."
                )
                if session_manager:
                    logger.warning("Attempting session restart...")
                    if session_manager.restart_sess():
                        logger.info("Session restarted. Retrying navigation...")
                        driver = session_manager.driver  # Get the new driver instance
                        if not driver:  # Check if restart actually provided a driver
                            logger.error(
                                "Session restart reported success but driver is still None."
                            )
                            return False
                        continue  # Retry navigation with new driver
                    else:
                        logger.error("Session restart failed. Cannot navigate.")
                        return False  # Unrecoverable
                else:
                    logger.error(
                        "Session invalid and no SessionManager provided for restart."
                    )
                    return False  # Unrecoverable

            # --- Navigation Execution ---
            logger.debug(f"Executing driver.get('{url}')...")
            driver.get(url)

            # Wait for document ready state (basic page load signal)
            WebDriverWait(driver, page_timeout).until(
                lambda d: d.execute_script("return document.readyState")
                in ["complete", "interactive"]
            )
            # Small pause allowing JS/redirects to potentially trigger
            time.sleep(random.uniform(0.5, 1.5))

            # --- Post-Navigation Checks ---
            try:
                landed_url = driver.current_url
                landed_url_parsed = urlparse(landed_url)
                landed_url_base = urlunparse(
                    (
                        landed_url_parsed.scheme,
                        landed_url_parsed.netloc,
                        landed_url_parsed.path.rstrip("/"),
                        "",
                        "",
                        "",
                    )
                ).rstrip("/")
                logger.debug(f"Landed on URL base: {landed_url_base}")
            except WebDriverException as e:
                logger.error(
                    f"Failed to get current URL after get() (Attempt {attempt}): {e}. Retrying."
                )
                continue  # Retry the navigation attempt

            # Check for MFA page
            is_on_mfa_page = False
            try:
                # Use short wait, presence is enough
                WebDriverWait(driver, 1).until(
                    EC.visibility_of_element_located(
                        (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)
                    )
                )
                is_on_mfa_page = True
            except (TimeoutException, NoSuchElementException):
                pass  # Expected if not on MFA page
            except WebDriverException as e:
                logger.warning(f"WebDriverException checking for MFA header: {e}")

            if is_on_mfa_page:
                logger.error(
                    "Landed on MFA page unexpectedly during navigation. Navigation failed."
                )
                # Should not attempt re-login here, indicates a prior login state issue
                return False  # Fail navigation

            # Check for Login page (only if *not* intentionally navigating there)
            is_on_login_page = False
            if (
                target_url_base != signin_page_url_base
            ):  # Don't check if login is the target
                try:
                    # Check if username input exists (strong indicator of login page)
                    WebDriverWait(driver, 1).until(
                        EC.visibility_of_element_located(
                            (By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR)
                        )
                    )
                    is_on_login_page = True
                except (TimeoutException, NoSuchElementException):
                    pass  # Expected if not on login page
                except WebDriverException as e:
                    logger.warning(
                        f"WebDriverException checking for Login username input: {e}"
                    )

            if is_on_login_page:
                logger.warning(
                    "Landed on Login page unexpectedly. Attempting re-login..."
                )
                if session_manager:
                    login_stat = login_status(
                        session_manager
                    )  # Check if maybe login just happened
                    if login_stat is True:
                        logger.info(
                            "Login status OK after landing on login page redirect. Retrying original navigation."
                        )
                        continue  # Retry original nav_to_page call
                    else:
                        # Attempt automated login
                        login_result_str = log_in(session_manager)
                        if login_result_str == "LOGIN_SUCCEEDED":
                            logger.info(
                                "Re-login successful. Retrying original navigation..."
                            )
                            continue  # Retry original nav_to_page call
                        else:
                            logger.error(
                                f"Re-login attempt failed ({login_result_str}). Cannot complete navigation."
                            )
                            return False  # Fail navigation if re-login fails
                else:
                    logger.error(
                        "Landed on login page, no SessionManager provided for re-login attempt."
                    )
                    return False  # Fail navigation

            # Check if landed on an unexpected URL (and not login/mfa)
            # Allow for slight variations (e.g., trailing slash) via base comparison
            if landed_url_base != target_url_base:
                # Check if it's a known redirect (e.g., signin page redirecting to base URL after successful login)
                is_signin_to_base_redirect = (
                    target_url_base == signin_page_url_base
                    and landed_url_base
                    == urlparse(config_instance.BASE_URL).path.rstrip("/")
                )
                if is_signin_to_base_redirect:
                    logger.debug(
                        "Redirected from signin page to base URL. Verifying login status..."
                    )
                    time.sleep(1)  # Allow settling
                    if session_manager and login_status(session_manager) is True:
                        logger.info(
                            "Redirect after signin confirmed as logged in. Considering original navigation target 'signin' successful."
                        )
                        return True  # Treat as success if login was the goal and we are now logged in

                # If not the known redirect, check for unavailability messages
                logger.warning(
                    f"Navigation landed on unexpected URL base: '{landed_url_base}' (Expected: '{target_url_base}')"
                )
                action, wait_time = _check_for_unavailability(
                    driver, unavailability_selectors
                )
                if action == "skip":
                    logger.error("Page no longer available message found. Skipping.")
                    return False  # Fail navigation
                elif action == "refresh":
                    logger.info(
                        f"Temporary unavailability message found. Waiting {wait_time}s and retrying..."
                    )
                    time.sleep(wait_time)
                    continue  # Retry navigation attempt
                else:
                    # Wrong URL, no specific message, likely a redirect issue
                    logger.warning(
                        "Wrong URL, no specific unavailability message found. Retrying navigation."
                    )
                    continue  # Retry navigation attempt

            # --- Final Check: Element on Page ---
            # If we reached here, we are on the correct URL base (or handled redirects)
            wait_selector = (
                selector if selector else "body"
            )  # Default to body if no selector provided
            logger.debug(
                f"On correct URL base. Waiting up to {element_timeout}s for selector: '{wait_selector}'"
            )
            try:
                WebDriverWait(driver, element_timeout).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, wait_selector))
                )
                logger.debug(
                    f"Navigation successful and element '{wait_selector}' found on: {url}"
                )
                return True  # Success!

            except TimeoutException:
                # Correct URL, but target element didn't appear
                current_url_on_timeout = "Unknown"
                try:
                    current_url_on_timeout = driver.current_url
                except Exception:
                    pass
                logger.warning(
                    f"Timeout waiting for selector '{wait_selector}' at {current_url_on_timeout} (URL base was correct)."
                )

                # Check again for unavailability messages that might have appeared late
                action, wait_time = _check_for_unavailability(
                    driver, unavailability_selectors
                )
                if action == "skip":
                    return False
                elif action == "refresh":
                    time.sleep(wait_time)
                    continue  # Retry navigation

                logger.warning(
                    "Timeout on selector, no unavailability message. Retrying navigation."
                )
                continue  # Retry navigation attempt

            except (
                WebDriverException
            ) as el_wait_err:  # Catch errors during element wait
                logger.error(
                    f"WebDriverException waiting for selector '{wait_selector}': {el_wait_err}"
                )
                continue  # Retry navigation

        # --- Handle Exceptions During Navigation Attempt ---
        except UnexpectedAlertPresentException as alert_e:
            alert_text = "N/A"
            try:
                alert_text = alert_e.alert_text
            except AttributeError:
                pass  # alert_text might not be available
            logger.warning(
                f"Unexpected alert detected (Attempt {attempt}): {alert_text}"
            )
            try:
                driver.switch_to.alert.accept()
                logger.info("Accepted unexpected alert.")
            except Exception as accept_e:
                logger.error(f"Failed to accept unexpected alert: {accept_e}")
                return False  # Fail if alert cannot be handled
            continue  # Retry navigation after handling alert

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
                    driver = session_manager.driver  # Get new driver
                    if not driver:
                        return False  # Fail if restart didn't provide driver
                    continue  # Retry navigation
                else:
                    logger.error("Session restart failed. Cannot complete navigation.")
                    return False  # Unrecoverable
            else:
                logger.warning(
                    "WebDriverException occurred, session seems valid or no restart possible. Waiting before retry."
                )
                time.sleep(random.uniform(2, 4))
                continue  # Retry navigation attempt

        except Exception as e:  # Catch other unexpected errors
            logger.error(
                f"Unexpected error during navigation (Attempt {attempt}): {e}",
                exc_info=True,
            )
            time.sleep(random.uniform(2, 4))  # Wait before retry
            continue  # Retry navigation attempt

    # --- Failed After All Attempts ---
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
    """Checks if known 'page unavailable' messages are present using provided selectors."""
    # Check if driver is usable
    if not is_browser_open(driver):
        logger.warning("Cannot check for unavailability: driver session invalid.")
        return None, 0

    for msg_selector, (action, wait_time) in selectors.items():
        # Use selenium_utils helper 'is_elem_there' with a very short wait
        if is_elem_there(driver, By.CSS_SELECTOR, msg_selector, wait=0.5):
            logger.warning(
                f"Unavailability message found matching selector: '{msg_selector}'. Action: {action}, Wait: {wait_time}s"
            )
            return action, wait_time  # Return action (refresh/skip) and wait time

    # Return default (no action, zero wait) if no matching selectors found
    return None, 0


# End of _check_for_unavailability


def main():
    """
    Standalone test suite for utils.py.
    Runs a sequence of tests covering core utilities, session management,
    API helpers, and other functions within this module.
    Provides a clear PASS/FAIL summary for each test and an overall result.
    """
    # --- Standard library imports needed for main ---
    import sys
    import traceback
    from textwrap import dedent

    # --- Local imports needed for main ---
    # Import necessary functions/classes specifically for testing within main
    from logging_config import setup_logging
    from config import config_instance, selenium_config

    # Import WebDriver explicitly if needed, handle potential import errors
    try:
        from selenium.webdriver.remote.webdriver import WebDriver
        from selenium.common.exceptions import WebDriverException
    except ImportError:
        WebDriver = type(None)  # Define as NoneType if selenium not installed
        WebDriverException = Exception  # Use base Exception as fallback

    # Re-assign the global logger for the main function's scope
    # Use INFO level to make test output cleaner
    global logger
    # Ensure config_instance is valid before proceeding
    if not config_instance or not config_instance.DATABASE_FILE:
        print(
            "ERROR: config_instance not loaded correctly. Cannot proceed with utils.py main test."
        )
        logging.basicConfig()  # Ensure basic logging works
        logging.critical(
            "config_instance not loaded correctly. Cannot proceed with utils.py main test."
        )
        sys.exit(1)

    # Setup logging for the test run
    try:
        db_file_path = config_instance.DATABASE_FILE
        log_filename_only = db_file_path.with_suffix(".log").name
        logger = setup_logging(log_file=log_filename_only, log_level="INFO")
        logger.info("--- Starting utils.py Standalone Test Suite ---")
    except Exception as log_setup_err:
        print(f"CRITICAL: Failed to set up logging: {log_setup_err}")
        logging.basicConfig()
        logging.critical(f"Failed to set up logging: {log_setup_err}", exc_info=True)
        sys.exit(1)

    # --- Test Runner Helper ---
    test_results = []

    def _run_test(
        test_name: str, test_func: Callable, *args, **kwargs
    ) -> Tuple[str, str, str]:
        """Runs a single test, logs result, and returns status."""
        logger.info(f"[ RUNNING ] {test_name}")
        status = "FAIL"  # Default to FAIL
        message = ""
        expect_none = kwargs.pop("expected_none", False)  # Pop internal flag

        try:
            result = test_func(*args, **kwargs)  # Pass cleaned kwargs

            # Determine PASS/FAIL based on result and expectations
            assertion_passed = False
            if isinstance(result, bool):
                assertion_passed = result  # Lambda returned True/False directly
                if not assertion_passed:
                    message = "Assertion in test function failed (returned False)"
            elif expect_none and result is None:
                assertion_passed = True  # Expected None, got None
            elif result is None and not expect_none:
                # Implicit None usually means success if no exception, unless failure expected
                assertion_passed = True  # Assume implicit None is PASS
            elif expect_none and result is not None:
                assertion_passed = False  # Expected None, got something else
                message = f"Expected None, but got {type(result)}"
            elif (
                result is not None
            ):  # Any other non-None result is treated as PASS if no exception
                assertion_passed = True

            status = "PASS" if assertion_passed else "FAIL"

        except Exception as e:
            status = "FAIL"
            message = f"{type(e).__name__}: {str(e)}"
            # Reduce noise: only log traceback for critical errors maybe
            # logger.error(f"Exception details for {test_name}: {message}", exc_info=True)
            logger.error(f"Exception details for {test_name}: {message}")

        result_log_level = logging.INFO if status == "PASS" else logging.ERROR
        log_message = f"[ {status:<6} ] {test_name}{f': {message}' if message and status == 'FAIL' else ''}"
        logger.log(result_log_level, log_message)
        return (test_name, status, message)

    # End of _run_test

    # --- Test Execution ---
    session_manager: SessionManagerType = None
    driver_instance: DriverType = None
    overall_status = "PASS"  # Assume PASS initially

    try:
        # === Section 1: Basic Utility Functions ===
        logger.info("\n--- Section 1: Basic Utility Functions ---")

        # 1.1 parse_cookie
        test_results.append(
            _run_test(
                "parse_cookie (valid)",
                lambda: parse_cookie("key1=value1; key2=value2 ; key3=val3=")
                == {"key1": "value1", "key2": "value2", "key3": "val3="},
            )
        )
        test_results.append(
            _run_test(
                "parse_cookie (empty/invalid)",
                lambda: parse_cookie(
                    " ; keyonly ; =valueonly; malformed=part=again ; valid=true "
                )
                == {"": "valueonly", "malformed": "part=again", "valid": "true"},
            )
        )
        test_results.append(
            _run_test(
                "parse_cookie (empty value)",
                lambda: parse_cookie("key=; next=val") == {"key": "", "next": "val"},
            )
        )
        test_results.append(
            _run_test(
                "parse_cookie (extra spacing)",
                lambda: parse_cookie(" key = value ; next = val ")
                == {"key": "value", "next": "val"},
            )
        )

        # 1.2 ordinal_case
        test_results.append(
            _run_test(
                "ordinal_case (numbers)",
                lambda: ordinal_case("1") == "1st"
                and ordinal_case("22") == "22nd"
                and ordinal_case("13") == "13th"
                and ordinal_case("104") == "104th",
            )
        )
        test_results.append(
            _run_test(
                "ordinal_case (string title)",
                lambda: ordinal_case("first cousin once removed")
                == "First Cousin Once Removed",
            )
        )
        test_results.append(
            _run_test(
                "ordinal_case (string specific lc)",
                lambda: ordinal_case("mother of the bride") == "Mother of the Bride",
            )
        )
        test_results.append(
            _run_test("ordinal_case (integer input)", lambda: ordinal_case(3) == "3rd")
        )

        # 1.3 format_name
        test_results.append(
            _run_test(
                "format_name (simple)",
                lambda: format_name("john smith") == "John Smith",
            )
        )
        test_results.append(
            _run_test(
                "format_name (GEDCOM simple)", lambda: format_name("/Smith/") == "Smith"
            )
        )
        test_results.append(
            _run_test(
                "format_name (GEDCOM start)",
                lambda: format_name("/Smith/ John") == "Smith John",
            )
        )
        test_results.append(
            _run_test(
                "format_name (GEDCOM end)",
                lambda: format_name("John /Smith/") == "John Smith",
            )
        )
        test_results.append(
            _run_test(
                "format_name (GEDCOM middle)",
                lambda: format_name("John /Smith/ Jr") == "John Smith JR",
            )
        )
        test_results.append(
            _run_test(
                "format_name (GEDCOM surrounding spaces)",
                lambda: format_name("  John   /Smith/   Jr  ") == "John Smith JR",
            )
        )
        test_results.append(
            _run_test(
                "format_name (with initials)",
                lambda: format_name("J. P. Morgan") == "J. P. Morgan",
            )
        )
        test_results.append(
            _run_test(
                "format_name (None input)",
                lambda: format_name(None) == "Valued Relative",
            )
        )
        test_results.append(
            _run_test(
                "format_name (Uppercase preserved/Particles)",
                lambda: format_name("McDONALD van der BEEK III")
                == "McDonald van der Beek III",
            )
        )
        test_results.append(
            _run_test(
                "format_name (Hyphenated)",
                lambda: format_name("jean-luc picard") == "Jean-Luc Picard",
            )
        )
        test_results.append(
            _run_test(
                "format_name (Apostrophe)",
                lambda: format_name("o'malley") == "O'Malley",
            )
        )
        test_results.append(
            _run_test(
                "format_name (Multiple spaces)",
                lambda: format_name("Jane  Elizabeth   Doe") == "Jane Elizabeth Doe",
            )
        )
        test_results.append(
            _run_test(
                "format_name (Numeric input)", lambda: format_name("12345") == "12345"
            )
        )
        test_results.append(
            _run_test(
                "format_name (Symbol input)", lambda: format_name("!@#$%^") == "!@#$%^"
            )
        )

        # === Section 2: Session Manager Lifecycle & Readiness ===
        logger.info("\n--- Section 2: Session Manager Lifecycle & Readiness ---")

        # 2.1 Instantiate SessionManager
        logger.info("[ RUNNING ] SessionManager Instantiation")
        try:
            session_manager = SessionManager()
            logger.info("[ PASS    ] SessionManager Instantiation")
            test_results.append(("SessionManager Instantiation", "PASS", ""))
        except Exception as sm_init_err:
            logger.error(f"[ FAIL    ] SessionManager Instantiation: {sm_init_err}")
            test_results.append(
                (
                    "SessionManager Instantiation",
                    "FAIL",
                    f"{type(sm_init_err).__name__}: {sm_init_err}",
                )
            )
            session_manager = None
            logger.error(
                "SessionManager instantiation failed. Skipping session-dependent tests."
            )

        # 2.2 SessionManager.start_sess
        if session_manager:
            start_sess_name, start_sess_status, start_sess_msg = _run_test(
                "SessionManager.start_sess()",
                session_manager.start_sess,
                action_name="Utils Test - Start Sess",
            )
            test_results.append((start_sess_name, start_sess_status, start_sess_msg))
            if start_sess_status == "PASS":
                driver_instance = session_manager.driver
            else:
                driver_instance = None
                logger.error("start_sess failed. Skipping tests requiring live driver.")
        else:
            test_results.append(
                (
                    "SessionManager.start_sess()",
                    "SKIPPED",
                    "SessionManager instantiation failed",
                )
            )

        # 2.3 SessionManager.ensure_session_ready
        if session_manager and driver_instance and session_manager.driver_live:
            ensure_ready_name, ensure_ready_status, ensure_ready_msg = _run_test(
                "SessionManager.ensure_session_ready()",
                session_manager.ensure_session_ready,
                action_name="Utils Test - Ensure Ready",
            )
            test_results.append(
                (ensure_ready_name, ensure_ready_status, ensure_ready_msg)
            )

            if ensure_ready_status == "FAIL":
                logger.warning("ensure_session_ready() FAILED. Running diagnostics...")
                diag_results = []
                login_stat_result = (
                    login_status(session_manager)
                    if session_manager.is_sess_valid()
                    else None
                )
                diag_results.append(
                    (
                        "Login status",
                        (
                            "PASSED"
                            if login_stat_result is True
                            else f"FAILED ({login_stat_result})"
                        ),
                    )
                )
                essential_cookies = ["ANCSESSIONID", "SecureATT"]
                cookies_ok = (
                    session_manager.get_cookies(essential_cookies, timeout=5)
                    if session_manager.is_sess_valid()
                    else False
                )
                diag_results.append(
                    ("Essential cookies", "PASSED" if cookies_ok else "FAILED")
                )
                if (
                    not session_manager.csrf_token
                    or len(session_manager.csrf_token) < 20
                ) and session_manager.is_sess_valid():
                    logger.info("Diag: Fetching CSRF token...")
                    try:
                        session_manager.csrf_token = session_manager.get_csrf()
                    except Exception as csrf_diag_err:
                        logger.error(f"Diag: Error fetching CSRF: {csrf_diag_err}")
                csrf_valid = bool(
                    session_manager.csrf_token and len(session_manager.csrf_token) >= 20
                )
                diag_results.append(
                    ("CSRF token", "PASSED" if csrf_valid else "FAILED")
                )
                ids_ok = bool(session_manager.my_profile_id and session_manager.my_uuid)
                tree_id_needed = bool(config_instance.TREE_NAME)
                tree_id_ok = (
                    bool(session_manager.my_tree_id) if tree_id_needed else True
                )
                ids_ok = ids_ok and tree_id_ok
                diag_results.append(("Identifiers", "PASSED" if ids_ok else "FAILED"))
                logger.info("--- Readiness Diagnostics ---")
                for name, result in diag_results:
                    logger.info(f"  - {name:<20}: {result}")
                logger.info("--- End Diagnostics ---")
        else:
            skip_reason = "Prerequisites failed (SM init or start_sess)"
            if not session_manager:
                skip_reason = "SessionManager instantiation failed"
            elif not driver_instance or not session_manager.driver_live:
                skip_reason = "start_sess failed"
            test_results.append(
                ("SessionManager.ensure_session_ready()", "SKIPPED", skip_reason)
            )

        # === Section 3: Session-Dependent Utilities ===
        logger.info("\n--- Section 3: Session-Dependent Utilities ---")

        session_ready_for_section_3 = bool(
            session_manager and session_manager.session_ready and driver_instance
        )
        skip_reason_s3 = "Session not ready" if not session_ready_for_section_3 else ""

        # 3.1 Header Generation (make_*)
        if session_ready_for_section_3:
            test_results.append(
                _run_test("make_ube()", lambda: bool(make_ube(driver_instance)))
            )
            test_results.append(
                _run_test(
                    "make_newrelic()", lambda: bool(make_newrelic(driver_instance))
                )
            )
            test_results.append(
                _run_test(
                    "make_traceparent()",
                    lambda: bool(make_traceparent(driver_instance)),
                )
            )
            test_results.append(
                _run_test(
                    "make_tracestate()", lambda: bool(make_tracestate(driver_instance))
                )
            )
        else:
            test_results.extend(
                [
                    ("make_ube()", "SKIPPED", skip_reason_s3),
                    ("make_newrelic()", "SKIPPED", skip_reason_s3),
                    ("make_traceparent()", "SKIPPED", skip_reason_s3),
                    ("make_tracestate()", "SKIPPED", skip_reason_s3),
                ]
            )

        # 3.2 Navigation (nav_to_page)
        if session_ready_for_section_3:
            nav_name, nav_status, nav_msg = _run_test(
                "nav_to_page() (to BASE_URL)",
                nav_to_page,
                driver=driver_instance,
                url=config_instance.BASE_URL,
                selector="body",
                session_manager=session_manager,
            )
            test_results.append((nav_name, nav_status, nav_msg))
            if nav_status == "PASS":
                try:
                    current_url = driver_instance.current_url
                    if not current_url.startswith(config_instance.BASE_URL.rstrip("/")):
                        logger.warning(
                            f"Navigation test PASSED element check, but landed on unexpected URL: {current_url}"
                        )
                except Exception as e:
                    logger.warning(f"Could not verify URL after nav_to_page test: {e}")
        else:
            test_results.append(
                ("nav_to_page() (to BASE_URL)", "SKIPPED", skip_reason_s3)
            )

        # 3.3 API Request (_api_req via CSRF fetch)
        if session_ready_for_section_3:
            csrf_url = urljoin(config_instance.BASE_URL, API_PATH_CSRF_TOKEN)

            def _test_csrf_api_req():
                response = _api_req(
                    url=csrf_url,
                    driver=driver_instance,
                    session_manager=session_manager,
                    method="GET",
                    use_csrf_token=False,
                    api_description="CSRF Token API Test",
                    force_text_response=True,
                )
                return isinstance(response, str) and len(response) > 20

            api_test_name, api_test_status, api_test_msg = _run_test(
                "_api_req() (fetch CSRF token)", _test_csrf_api_req
            )
            test_results.append((api_test_name, api_test_status, api_test_msg))
        else:
            test_results.append(
                ("_api_req() (fetch CSRF token)", "SKIPPED", skip_reason_s3)
            )

        # 3.4 _send_message_via_api (Dry Run / Input Validation)
        if session_manager:
            try:
                from database import Person  # Ensure Person is available

                if not session_manager.my_profile_id and session_manager.session_ready:
                    logger.warning("Fetching my_profile_id for _send_message test...")
                    session_manager.my_profile_id = session_manager.get_my_profileId()

                if not session_manager.my_profile_id:
                    logger.warning(
                        "my_profile_id missing, skipping _send_message tests."
                    )
                    test_results.extend(
                        [
                            (
                                "_send_message_via_api (dry_run)",
                                "SKIPPED",
                                "my_profile_id missing",
                            ),
                            (
                                "_send_message_via_api (invalid recipient)",
                                "SKIPPED",
                                "my_profile_id missing",
                            ),
                        ]
                    )
                else:
                    dummy_person_ok = Person(
                        profile_id="DUMMY-PROFILE-ID-OK", username="dummy_ok"
                    )
                    dummy_person_bad = Person(profile_id=None, username="dummy_bad")
                    original_app_mode = config_instance.APP_MODE
                    config_instance.APP_MODE = "dry_run"
                    test_results.append(
                        _run_test(
                            "_send_message_via_api (dry_run)",
                            lambda: _send_message_via_api(
                                session_manager,
                                dummy_person_ok,
                                "Test message",
                                None,
                                "DryRunTest",
                            )[0]
                            == SEND_SUCCESS_DRY_RUN,
                        )
                    )
                    test_results.append(
                        _run_test(
                            "_send_message_via_api (invalid recipient)",
                            lambda: _send_message_via_api(
                                session_manager,
                                dummy_person_bad,
                                "Test",
                                None,
                                "InvalidTest",
                            )[0]
                            == SEND_ERROR_INVALID_RECIPIENT,
                        )
                    )
                    config_instance.APP_MODE = original_app_mode
            except ImportError:
                test_results.append(
                    (
                        "_send_message_via_api Tests",
                        "SKIPPED",
                        "Could not import Person class",
                    )
                )
            except Exception as send_test_e:
                test_results.append(
                    (
                        "_send_message_via_api Tests",
                        "FAIL",
                        f"Unexpected error in test setup: {send_test_e}",
                    )
                )
        else:
            test_results.append(
                (
                    "_send_message_via_api Tests",
                    "SKIPPED",
                    "SessionManager instantiation failed",
                )
            )

        # 3.5 _fetch_profile_details_for_person (Input Validation)
        if session_manager:
            if not session_manager.my_profile_id and session_manager.session_ready:
                logger.warning(
                    "Fetching my_profile_id for _fetch_profile_details test..."
                )
                session_manager.my_profile_id = session_manager.get_my_profileId()

            if not session_manager.my_profile_id:
                logger.warning(
                    "my_profile_id missing, skipping _fetch_profile_details test."
                )
                test_results.append(
                    (
                        "_fetch_profile_details_for_person (invalid input)",
                        "SKIPPED",
                        "my_profile_id missing",
                    )
                )
            else:
                test_results.append(
                    _run_test(
                        "_fetch_profile_details_for_person (invalid input)",
                        lambda: _fetch_profile_details_for_person(session_manager, "")
                        is None,
                        expected_none=True,
                    )
                )
        else:
            test_results.append(
                (
                    "_fetch_profile_details_for_person (invalid input)",
                    "SKIPPED",
                    "SessionManager instantiation failed",
                )
            )

        # === Section 4: Tab Management ===
        logger.info("\n--- Section 4: Tab Management ---")
        if session_ready_for_section_3:
            initial_handles = []
            try:
                initial_handles = driver_instance.window_handles
                make_tab_name, make_tab_status, make_tab_msg = _run_test(
                    "SessionManager.make_tab()", session_manager.make_tab
                )
                test_results.append((make_tab_name, make_tab_status, make_tab_msg))

                if make_tab_status == "PASS":
                    try:
                        from selenium_utils import close_tabs

                        handles_after_make = driver_instance.window_handles
                        if len(handles_after_make) > len(initial_handles):
                            logger.info(
                                "make_tab appears successful (handle count increased)."
                            )
                            close_tab_name, close_tab_status, close_tab_msg = _run_test(
                                "close_tabs()", close_tabs, driver=driver_instance
                            )
                            test_results.append(
                                (close_tab_name, close_tab_status, close_tab_msg)
                            )

                            if close_tab_status == "PASS":
                                handles_after_close = driver_instance.window_handles
                                if len(handles_after_close) != 1:
                                    logger.error(
                                        f"close_tabs test failed verification: Expected 1 handle, found {len(handles_after_close)}"
                                    )
                                    for i, res in enumerate(test_results):
                                        if res[0] == "close_tabs()":
                                            test_results[i] = (
                                                res[0],
                                                "FAIL",
                                                f"Post-test verification failed: Expected 1 handle, found {len(handles_after_close)}",
                                            )
                                        break
                                else:
                                    logger.info(
                                        "close_tabs() post-test verification PASSED (1 tab remaining)."
                                    )

                        else:
                            logger.error(
                                f"make_tab test failed verification: Handle count did not increase ({len(initial_handles)} -> {len(handles_after_make)})"
                            )
                            for i, res in enumerate(test_results):
                                if res[0] == "SessionManager.make_tab()":
                                    if res[1] == "PASS":
                                        test_results[i] = (
                                            res[0],
                                            "FAIL",
                                            "Verification failed: Handle count did not increase",
                                        )
                                    break
                            test_results.append(
                                (
                                    "close_tabs()",
                                    "SKIPPED",
                                    "make_tab verification failed",
                                )
                            )
                    except ImportError:
                        test_results.append(
                            (
                                "close_tabs()",
                                "SKIPPED",
                                "Could not import from selenium_utils",
                            )
                        )
                    except Exception as tab_close_e:
                        test_results.append(
                            (
                                "close_tabs()",
                                "FAIL",
                                f"Exception during close_tabs test: {tab_close_e}",
                            )
                        )
                else:
                    test_results.append(("close_tabs()", "SKIPPED", "make_tab failed"))
            except WebDriverException as e:
                test_results.append(
                    (
                        "SessionManager.make_tab()",
                        "FAIL",
                        f"WebDriverException during test: {e}",
                    )
                )
                test_results.append(
                    (
                        "close_tabs()",
                        "SKIPPED",
                        "make_tab failed due to WebDriverException",
                    )
                )
            except Exception as e:
                test_results.append(
                    (
                        "SessionManager.make_tab()",
                        "FAIL",
                        f"Unexpected exception during test: {e}",
                    )
                )
                test_results.append(
                    (
                        "close_tabs()",
                        "SKIPPED",
                        "make_tab failed due to unexpected exception",
                    )
                )
        else:
            test_results.append(
                ("SessionManager.make_tab()", "SKIPPED", skip_reason_s3)
            )
            test_results.append(("close_tabs()", "SKIPPED", skip_reason_s3))

    except Exception as e:
        logger.critical(
            f"--- CRITICAL ERROR during test execution: {e} ---", exc_info=True
        )
        overall_status = "FAIL"
        test_results.append(("Test Suite Execution", "FAIL", f"Critical error: {e}"))

    finally:
        # === Cleanup ===
        if session_manager and session_manager.driver_live:
            logger.info("Closing session manager in finally block...")
            session_manager.close_sess(keep_db=True)
        elif session_manager:
            logger.info(
                "Session manager exists but driver not live, attempting minimal cleanup..."
            )
            session_manager.cls_db_conn(keep_db=True)
        else:
            logger.info("No SessionManager instance to close.")

        # === Summary Report ===
        logger.info("\n--- Test Summary ---")
        name_width = (
            max(len(name) for name, _, _ in test_results) if test_results else 45
        )
        name_width = max(name_width, 45)
        status_width = 8
        header = (
            f"{'Test Name':<{name_width}} | {'Status':<{status_width}} | {'Message'}"
        )
        logger.info(header)
        logger.info("-" * (name_width + status_width + 12))

        final_fail_count = 0
        final_skip_count = 0
        for name, status, message in test_results:
            if status == "FAIL":
                final_fail_count += 1
                overall_status = "FAIL"
                logger.error(
                    f"{name:<{name_width}} | {status:<{status_width}} | {message}"
                )
            elif status == "SKIPPED":
                final_skip_count += 1
                logger.warning(
                    f"{name:<{name_width}} | {status:<{status_width}} | {message}"
                )
            else:  # PASS
                logger.info(f"{name:<{name_width}} | {status:<{status_width}} |")

        logger.info("-" * (len(header)))

        # === Overall Conclusion ===
        total_tests = len(test_results)
        passed_tests = total_tests - final_fail_count - final_skip_count

        summary_line = f"Result: {overall_status} ({passed_tests} passed, {final_fail_count} failed, {final_skip_count} skipped out of {total_tests} tests)"
        if overall_status == "PASS":
            logger.info(summary_line)
            logger.info("--- Utils.py standalone test run PASSED ---")
        else:
            logger.error(summary_line)
            logger.error("--- Utils.py standalone test run FAILED ---")

        # Optional: Exit with non-zero code on failure for CI/CD
        # if overall_status == "FAIL":
        #      sys.exit(1)


# End of main


if __name__ == "__main__":
    main()
# End of utils.py
