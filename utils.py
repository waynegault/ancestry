#!/usr/bin/env python3

"""
utils.py - Core Session Management, API Requests, General Utilities

Manages Selenium/Requests sessions, handles core API interaction (_api_req),
provides general utilities (decorators, formatting, rate limiting),
and includes login/session verification logic closely tied to SessionManager.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
    register_function,
    get_function,
    is_function_available,
    auto_register_module,  # Needed for testing
)

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === SESSION MANAGER IMPORT ===
# Import SessionManager from core module - use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.session_manager import SessionManager
else:
    # Runtime import to avoid circular dependency issues
    SessionManager = None

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import json
import logging
import re
import sys
import time
import threading
from typing import (
    Optional,
    Dict,
    Any,
    Union,
    List,
    Tuple,
    Callable,
    Type,
    Generator,
    IO,
)  # Consolidated typing imports
import asyncio  # For async/await patterns
import base64  # For make_ube
import binascii  # For make_ube
import contextlib  # Added import for contextlib
import random  # For make_newrelic, retry_api, DynamicRateLimiter
import sqlite3  # For SessionManager._initialize_db_engine_and_session (pragma exception)
import time
import uuid  # For make_ube, make_traceparent, make_tracestate
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

# === THIRD-PARTY IMPORTS ===
try:
    import aiohttp  # For async HTTP requests
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available - async API functions will be disabled")

import cloudscraper
import requests
from requests import Response as RequestsResponse
from selenium.webdriver.remote.webdriver import WebDriver

# === LOCAL IMPORTS ===
# (Note: Some imports done locally to avoid circular dependencies)

# === TYPE ALIASES ===
# Define type aliases
RequestsResponseTypeOptional = Optional[RequestsResponse]
ApiResponseType = Union[Dict[str, Any], List[Any], str, bytes, None, RequestsResponse]
DriverType = Optional[WebDriver]
SessionManagerType = Optional[
    "SessionManager"
]  # Use string literal for forward reference

# === MODULE CONSTANTS ===
# Key constants remain here or moved to api_utils as appropriate
API_PATH_CSRF_TOKEN = "discoveryui-matches/parents/api/csrfToken"
API_PATH_PROFILE_ID = "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid"
API_PATH_UUID = "api/uhome/secure/rest/header/dna"

KEY_UCDMID = "ucdmid"
KEY_TEST_ID = "testId"
KEY_DATA = "data"

# --- Third-party and local imports ---
# Keep the warning for optional dependencies, but don't define dummies.
# If essential ones fail, other parts of the code will raise errors.
try:
    from requests import Response as RequestsResponse
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
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.wait import WebDriverWait
    from sqlalchemy import create_engine, event, pool as sqlalchemy_pool, inspect
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import Session, sessionmaker
    from urllib3.util.retry import Retry

    # --- Local application imports ---
    # Assume these are essential or handled elsewhere if missing
    from chromedriver import init_webdvr
    from config import config_manager, config_schema
    from core_imports import get_logger

    # Initialize logger with standardized pattern
    logger = get_logger(__name__)

    from database import Base  # Import Base for table creation
    from my_selectors import *

    from selenium_utils import (
        is_browser_open,
        is_elem_there,
        export_cookies,
    )

    # Do NOT import api_utils here at the top level

except ImportError as import_err:
    # Log failure for other imports but don't define dummies
    logging.critical(
        f"Essential dependency import failed in utils.py: {import_err}. Script cannot continue.",
        exc_info=True,
    )
    # Re-raise the error to stop execution
    raise import_err

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
    MagicMock,
)

# ------------------------------------------------------------------------------------
# Helper functions (General Utilities)
# ------------------------------------------------------------------------------------

def parse_cookie(cookie_string: str) -> Dict[str, str]:
    """
    Parse a raw HTTP cookie string into a dictionary of key-value pairs.

    Handles various cookie formats including empty keys, empty values, and
    malformed entries. Provides robust parsing for web scraping and session
    management scenarios.

    Args:
        cookie_string: Raw HTTP cookie string with semicolon-separated pairs.

    Returns:
        Dict[str, str]: Dictionary mapping cookie names to values.

    Examples:
        >>> parse_cookie("key1=value1; key2=value2")
        {'key1': 'value1', 'key2': 'value2'}
        >>> parse_cookie("key=; next=val")
        {'key': '', 'next': 'val'}
        >>> parse_cookie(" ; keyonly ; =valueonly")
        {'': 'valueonly'}
        >>> parse_cookie("malformed=part=again")
        {'malformed': 'part=again'}
    """
    cookies: Dict[str, str] = {}
    if not isinstance(cookie_string, str):
        logger.warning("parse_cookie received non-string input, returning empty dict.")
        return cookies
    # End of if

    parts = cookie_string.split(";")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # End of if
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            cookies[key] = value
        else:
            logger.debug(f"Skipping cookie part without '=': '{part}'")
        # End of if/else
    # End of for
    return cookies

# End of parse_cookie

def ordinal_case(text: Union[str, int]) -> str:
    """
    Convert text to title case with proper ordinal suffix handling.

    Converts text to title case and ensures ordinal suffixes (1st, 2nd, 3rd, 4th)
    are properly formatted. Handles both string and integer inputs for numbers.
    Commonly used for relationship terms and genealogical descriptions.

    Args:
        text: The text to format, can be string or integer.

    Returns:
        str: Formatted text with proper title case and ordinal suffixes.

    Examples:
        >>> ordinal_case("first cousin once removed")
        'First Cousin Once Removed'
        >>> ordinal_case(1)
        '1st'
        >>> ordinal_case(22)
        '22nd'
        >>> ordinal_case(13)
        '13th'
        >>> ordinal_case("mother of the bride")
        'Mother of the Bride'
    """
    if not text and text != 0:
        return str(text) if text is not None else ""
    # End of if

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
            # End of if/elif/else
        # End of if/else
        return str(num) + suffix
    except (ValueError, TypeError):
        if isinstance(text, str):
            words = text.title().split()
            lc_words = {"Of", "The", "A", "An", "In", "On", "At", "For", "To", "With"}
            for i, word in enumerate(words):
                if i > 0 and word in lc_words:
                    words[i] = word.lower()
                # End of if
            # End of for
            return " ".join(words)
        else:
            return str(text)
        # End of if/else
    # End of try/except

# End of ordinal_case

def format_name(name: Optional[str]) -> str:
    """
    Format a person's name string with proper capitalization and GEDCOM cleanup.

    Converts names to title case while preserving uppercase components like initials,
    removes GEDCOM-style slashes around surnames, and handles common name particles
    and prefixes correctly. Provides graceful handling of None/empty input.

    Args:
        name: The name string to format, or None.

    Returns:
        str: Formatted name string, or "Valued Relative" if input is None/empty.

    Examples:
        >>> format_name("john smith")
        'John Smith'
        >>> format_name("/Smith/ John")
        'Smith John'
        >>> format_name("McDONALD")
        'McDonald'
        >>> format_name("o'malley")
        "O'Malley"
        >>> format_name(None)
        'Valued Relative'
        >>> format_name("J. P. Morgan")
        'J. P. Morgan'
    """
    if not name or not isinstance(name, str):
        return "Valued Relative"
    # End of if

    if name.isdigit() or re.fullmatch(r"[^a-zA-Z]+", name):
        logger.debug(
            f"Formatting name: Input '{name}' appears non-alphabetic, returning as is."
        )
        stripped_name = name.strip()
        return stripped_name if stripped_name else "Valued Relative"
    # End of if

    try:
        cleaned_name = name.strip()
        # Handle GEDCOM slashes more robustly
        cleaned_name = re.sub(r"\s*/([^/]+)/\s*", r" \1 ", cleaned_name)  # Middle
        cleaned_name = re.sub(r"^/([^/]+)/\s*", r"\1 ", cleaned_name)  # Start
        cleaned_name = re.sub(r"\s*/([^/]+)/$", r" \1", cleaned_name)  # End
        cleaned_name = re.sub(r"^/([^/]+)/$", r"\1", cleaned_name)  # Only

        cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()

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
        uppercase_exceptions = {"II", "III", "IV", "SR", "JR"}

        parts = cleaned_name.split()
        formatted_parts = []
        i = 0
        while i < len(parts):
            part = parts[i]
            part_lower = part.lower()

            if i > 0 and part_lower in lowercase_particles:
                formatted_parts.append(part_lower)
                i += 1
                continue
            # End of if

            if part.upper() in uppercase_exceptions:
                formatted_parts.append(part.upper())
                i += 1
                continue
            # End of if

            # *** NEW/MODIFIED LOGIC FOR QUOTED NICKNAMES AND APOSTROPHES ***
            if part.startswith("'") and part.endswith("'") and len(part) > 2:
                # Handles parts like 'Betty' or 'bo'
                # Capitalize the content within the quotes
                inner_content = part[1:-1]
                formatted_parts.append("'" + inner_content.capitalize() + "'")
                i += 1
                continue
            # End of if

            if "-" in part:
                hyphenated_elements = []
                sub_parts = part.split("-")
                for idx, sub_part in enumerate(sub_parts):
                    if idx > 0 and sub_part.lower() in lowercase_particles:
                        hyphenated_elements.append(sub_part.lower())
                    elif sub_part:  # Ensure sub_part is not empty
                        hyphenated_elements.append(sub_part.capitalize())
                    # End of if/elif
                # End of for
                formatted_parts.append("-".join(filter(None, hyphenated_elements)))
                i += 1
                continue
            # End of if

            # Handle names like O'Malley, D'Angelo
            if (
                "'" in part
                and len(part) > 1
                and not (part.startswith("'") or part.endswith("'"))
            ):
                # This condition targets internal apostrophes like in O'Malley
                # It avoids single-quoted parts like 'Betty' which are handled above.
                name_pieces = part.split("'")
                # Capitalize the first letter of each piece around the apostrophe
                formatted_apostrophe_part = "'".join(
                    p.capitalize() for p in name_pieces
                )
                formatted_parts.append(formatted_apostrophe_part)
                i += 1
                continue
            # End of if

            if part_lower.startswith("mc") and len(part) > 2:
                formatted_parts.append("Mc" + part[2:].capitalize())
                i += 1
                continue
            # End of if
            if part_lower.startswith("mac") and len(part) > 3:
                if part_lower == "mac":  # Handle "Mac" itself
                    formatted_parts.append("Mac")
                else:
                    formatted_parts.append("Mac" + part[3:].capitalize())
                # End of if/else
                i += 1
                continue
            # End of if

            if (
                len(part) == 2 and part.endswith(".") and part[0].isalpha()
            ):  # Initials like J.
                formatted_parts.append(part[0].upper() + ".")
                i += 1
                continue
            # End of if
            if (
                len(part) == 1 and part.isalpha()
            ):  # Single letter initials without period
                formatted_parts.append(part.upper())
                i += 1
                continue
            # End of if

            # Default: capitalize the part
            formatted_parts.append(part.capitalize())
            i += 1
        # End of while

        final_name = " ".join(formatted_parts)
        final_name = re.sub(
            r"\s+", " ", final_name
        ).strip()  # Consolidate multiple spaces
        return final_name if final_name else "Valued Relative"
    except Exception as e:
        logger.error(f"Error formatting name '{name}': {e}", exc_info=False)
        try:
            return name.title() if isinstance(name, str) else "Valued Relative"
        except AttributeError:
            return "Valued Relative"
        # End of try/except
    # End of try/except

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

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cfg = config_schema  # Use new config system
            attempts = (
                MAX_RETRIES
                if MAX_RETRIES is not None
                else getattr(cfg, "MAX_RETRIES", 3)
            )
            backoff = (
                BACKOFF_FACTOR
                if BACKOFF_FACTOR is not None
                else getattr(cfg, "BACKOFF_FACTOR", 1.0)
            )
            max_delay = (
                MAX_DELAY if MAX_DELAY is not None else getattr(cfg, "MAX_DELAY", 10.0)
            )
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == attempts - 1:
                        logger.error(
                            f"Function '{func.__name__}' failed after {attempts} retries. Final Exception: {e}",
                            exc_info=False,  # Keep simple log for retry failure
                        )
                        raise  # Re-raise the final exception
                    # End of if
                    sleep_time = min(backoff * (2**i), max_delay) + random.uniform(
                        0, 0.5
                    )
                    logger.warning(
                        f"Retry {i+1}/{attempts} for {func.__name__} after exception: {type(e).__name__}. Sleeping {sleep_time:.2f}s."
                    )
                    time.sleep(sleep_time)
                # End of try/except
            # End of for
            # This part should ideally not be reached if raise is used above
            logger.error(
                f"Function '{func.__name__}' failed after all {attempts} retries (exited loop unexpectedly)."
            )
            raise RuntimeError(f"Function {func.__name__} failed after all retries.")

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
        requests.exceptions.RequestException,  # type: ignore # Assume imported
        ConnectionError,
        TimeoutError,
    ),
    retry_on_status_codes: Optional[List[int]] = None,
):
    """Decorator factory for retrying API calls with exponential backoff, logging, etc."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cfg = config_schema  # Use new config system
            _max_retries = (
                max_retries
                if max_retries is not None
                else getattr(cfg, "MAX_RETRIES", 3)
            )
            _initial_delay = (
                initial_delay
                if initial_delay is not None
                else getattr(cfg, "INITIAL_DELAY", 0.5)
            )
            _backoff_factor = (
                backoff_factor
                if backoff_factor is not None
                else getattr(cfg, "BACKOFF_FACTOR", 1.5)
            )
            _retry_codes_set = set(
                retry_on_status_codes
                if retry_on_status_codes is not None
                else getattr(cfg, "RETRY_STATUS_CODES", [429, 500, 502, 503, 504])
            )
            _max_delay = getattr(cfg, "MAX_DELAY", 60.0)
            retries = _max_retries
            delay = _initial_delay
            attempt = 0
            last_exception: Optional[Exception] = None
            last_response: RequestsResponseTypeOptional = None
            while retries > 0:
                attempt += 1
                try:
                    response = func(*args, **kwargs)
                    last_response = response
                    status_code: Optional[int] = None
                    if isinstance(response, requests.Response):  # type: ignore # Assume imported
                        status_code = response.status_code
                    # End of if
                    should_retry_status = False
                    if status_code is not None and status_code in _retry_codes_set:
                        should_retry_status = True
                        last_exception = requests.exceptions.HTTPError(  # type: ignore
                            f"{status_code} Error", response=response
                        )
                    # End of if
                    if should_retry_status:
                        retries -= 1
                        if retries <= 0:
                            logger.error(
                                f"API Call failed after {_max_retries} retries for '{func.__name__}' (Final Status {status_code})."
                            )
                            # Return the response object on final failure
                            return response
                        # End of if
                        sleep_time = min(
                            delay * (_backoff_factor ** (attempt - 1)), _max_delay
                        ) + random.uniform(0, 0.2)
                        sleep_time = max(0.1, sleep_time)
                        logger.warning(
                            f"API Call status {status_code} (Attempt {attempt}/{_max_retries}) for '{func.__name__}'. Retrying in {sleep_time:.2f}s..."
                        )
                        time.sleep(sleep_time)
                        delay *= _backoff_factor
                        continue
                    else:
                        # Success or non-retryable error, return the response
                        return response
                    # End of if/else
                except retry_on_exceptions as e:
                    last_exception = e
                    retries -= 1
                    if retries <= 0:
                        logger.error(
                            f"API Call failed after {_max_retries} retries for '{func.__name__}'. Final Exception: {type(e).__name__} - {e}",
                            exc_info=False,
                        )
                        # Raise the exception on final retry failure
                        raise e
                    # End of if
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
                except Exception as e:
                    # Non-retryable exception occurred
                    logger.error(
                        f"Unexpected error during API call attempt {attempt} for '{func.__name__}': {e}",
                        exc_info=True,
                    )
                    raise e  # Re-raise immediately
                # End of try/except
            # End of while
            # Should only be reached if the loop completes unexpectedly (e.g., condition error)
            # Or if the last attempt resulted in a retryable error but retries hit 0
            logger.error(
                f"Exited retry loop for '{func.__name__}'. Last status: {getattr(last_response, 'status_code', 'N/A')}, Last exception: {last_exception}"
            )
            if last_exception:
                raise last_exception  # Re-raise the last exception if one occurred
            else:
                # This case implies a retryable status on the last attempt
                return (
                    last_response
                    if last_response is not None
                    else RuntimeError(f"{func.__name__} failed after all retries.")
                )
            # End of if/else

        # End of wrapper
        return wrapper

    # End of decorator
    return decorator

# End of retry_api

def ensure_browser_open(func: Callable) -> Callable:
    """Decorator to ensure browser session is valid before executing."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        session_manager_instance: SessionManagerType = None
        driver_instance: DriverType = None

        # Logic to find WebDriver instance (simplified)
        if args:
            if isinstance(args[0], SessionManager):  # type: ignore # Assume SessionManager available
                session_manager_instance = args[0]
                driver_instance = session_manager_instance.driver
            elif isinstance(args[0], WebDriver):  # type: ignore # Assume WebDriver available
                driver_instance = args[0]
            # End of if/elif
        # End of if
        if not driver_instance and "driver" in kwargs:
            if isinstance(kwargs["driver"], WebDriver):  # type: ignore
                driver_instance = kwargs["driver"]
            # End of if
        # End of if
        if not driver_instance and "session_manager" in kwargs:
            if isinstance(kwargs["session_manager"], SessionManager):  # type: ignore
                session_manager_instance = kwargs["session_manager"]
                driver_instance = session_manager_instance.driver
            # End of if
        # End of if

        # Final check and raise error if no driver found
        if not driver_instance:
            raise TypeError(
                f"Function '{func.__name__}' decorated with @ensure_browser_open requires a WebDriver instance."
            )
        # End of if

        # Check if browser is open using utility function
        if not is_browser_open(driver_instance):
            raise WebDriverException(  # type: ignore
                f"Browser session invalid/closed when calling function '{func.__name__}'"
            )
        # End of if
        return func(*args, **kwargs)

    # End of wrapper
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
                return result
            except TimeoutException as e:  # type: ignore # Assume imported
                duration = time.time() - start_time
                logger.warning(
                    f"Wait '{wait_description}' timed out after {duration:.3f} seconds.",
                    exc_info=False,  # Don't need full trace for timeout
                )
                raise e  # Re-raise TimeoutException
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Error during wait '{wait_description}' after {duration:.3f} seconds: {e}",
                    exc_info=True,  # Log full trace for other errors
                )
                raise e  # Re-raise other exceptions
            # End of try/except

        # End of wrapper
        return wrapper

    # End of decorator
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
        cfg = config_schema  # Use new config system
        api = getattr(cfg, "api", None)
        # Use APIConfig-backed values to ensure .env is respected
        self.initial_delay = (
            initial_delay
            if initial_delay is not None
            else (getattr(api, "initial_delay", 2.0) if api else 2.0)
        )
        self.MAX_DELAY = (
            max_delay if max_delay is not None else (getattr(api, "max_delay", 60.0) if api else 60.0)
        )
        self.backoff_factor = (
            backoff_factor
            if backoff_factor is not None
            else (getattr(api, "retry_backoff_factor", 4.0) if api else 4.0)
        )
        self.decrease_factor = (
            decrease_factor
            if decrease_factor is not None
            else 0.98
        )
        self.current_delay = self.initial_delay
        self.last_throttled = False
        # Token Bucket parameters (capacity=burst_limit, fill_rate=requests_per_second)
        self.capacity = float(
            token_capacity
            if token_capacity is not None
            else (getattr(api, "burst_limit", 3.0) if api else 3.0)
        )
        self.fill_rate = float(
            token_fill_rate
            if token_fill_rate is not None
            else (getattr(api, "requests_per_second", 0.5) if api else 0.5)
        )
        self._lock = threading.Lock()
        if self.fill_rate <= 0:
            logger.warning(
                f"Token fill rate ({self.fill_rate}) must be positive. Setting to 1.0."
            )
            self.fill_rate = 1.0
        # End of if
        self.tokens = float(self.capacity)
        self.last_refill_time = time.monotonic()
        logger.debug(
            f"RateLimiter Init: Capacity={self.capacity:.1f}, FillRate={self.fill_rate:.1f}/s, InitialDelay={self.initial_delay:.2f}s, MaxDelay={self.MAX_DELAY:.1f}s, Backoff={self.backoff_factor:.2f}, Decrease={self.decrease_factor:.2f}"
        )

    # End of __init__

    def _refill_tokens(self) -> None:
        now = time.monotonic()
        elapsed = max(0, now - self.last_refill_time)
        tokens_to_add = elapsed * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = now

    # End of _refill_tokens

    def wait(self) -> float:
        # Serialize token accounting to ensure correctness under concurrency
        with self._lock:
            self._refill_tokens()
            sleep_duration = 0.0

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                # Apply base delay even if token is available
                jitter_factor = random.uniform(0.8, 1.2)
                base_sleep = self.current_delay
                sleep_duration = min(base_sleep * jitter_factor, self.MAX_DELAY)
                sleep_duration = max(0.01, sleep_duration)  # Ensure minimum sleep
                logger.debug(
                    f"Token available ({self.tokens:.2f} left). Applying base delay: {sleep_duration:.3f}s (CurrentDelay: {self.current_delay:.2f}s)"
                )
            else:
                # Token bucket empty, wait for a token to generate
                wait_needed = (1.0 - self.tokens) / self.fill_rate
                jitter_amount = random.uniform(0.0, 0.2)  # Small extra jitter
                sleep_duration = wait_needed + jitter_amount
                sleep_duration = min(sleep_duration, self.MAX_DELAY)  # Cap wait time
                sleep_duration = max(0.01, sleep_duration)  # Ensure minimum sleep
                logger.debug(
                    f"Token bucket empty ({self.tokens:.2f}). Waiting for token: {sleep_duration:.3f}s"
                )
        # End of with

        # Perform the sleep outside the lock
        if sleep_duration > 0:
            time.sleep(sleep_duration)

        # After sleeping, do a quick refill under lock to update state
        with self._lock:
            self._refill_tokens()

        return sleep_duration

    # End of wait

    def reset_delay(self) -> None:
        if self.current_delay != self.initial_delay:
            logger.info(
                f"Rate limiter base delay reset from {self.current_delay:.2f}s to initial: {self.initial_delay:.2f}s"
            )
            self.current_delay = self.initial_delay
        # End of if
        self.last_throttled = False

    # End of reset_delay

    def decrease_delay(self) -> None:
        if not self.last_throttled and self.current_delay > self.initial_delay:
            previous_delay = self.current_delay
            self.current_delay = max(
                self.current_delay * self.decrease_factor, self.initial_delay
            )
            if (
                abs(previous_delay - self.current_delay) > 0.01
            ):  # Log only significant changes
                logger.debug(
                    f"Decreased base delay component to {self.current_delay:.2f}s"
                )
            # End of if
        # End of if
        self.last_throttled = False  # Reset flag after successful operation

    # End of decrease_delay

    def increase_delay(self) -> None:
        previous_delay = self.current_delay
        self.current_delay = min(
            self.current_delay * self.backoff_factor, self.MAX_DELAY
        )
        if (
            abs(previous_delay - self.current_delay) > 0.01
        ):  # Log only significant changes
            logger.info(
                f"Rate limit feedback received. Increased base delay from {previous_delay:.2f}s to {self.current_delay:.2f}s"
            )
        else:
            logger.debug(
                f"Rate limit feedback received, but delay already at max ({self.MAX_DELAY:.2f}s) or increase too small."
            )
        # End of if/else
        self.last_throttled = True

    # End of increase_delay

    def is_throttled(self) -> bool:
        return self.last_throttled

    # End of is_throttled

# End of DynamicRateLimiter class

# ------------------------------
# Session Management (MOVED TO core.session_manager)
# ------------------------------
# SessionManager class has been moved to core.session_manager.SessionManager
# Import it from there: from core.session_manager import SessionManager

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Stand alone functions
# ----------------------------------------------------------------------------

def _prepare_base_headers(
    method: str,
    api_description: str,
    referer_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Prepares the base headers for an API request.

    Args:
        method: HTTP method (GET, POST, etc.)
        api_description: Description of the API being called
        referer_url: Optional referer URL
        headers: Optional additional headers

    Returns:
        Dictionary of base headers
    """
    cfg = config_schema  # Use new config system

    # Create base headers
    base_headers: Dict[str, str] = {
        "Accept": "application/json, text/plain, */*",
        "Referer": referer_url or cfg.api.base_url,
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "_method": method.upper(),  # Internal key for _prepare_api_headers
    }

    # Apply contextual headers from config
    contextual_headers = getattr(cfg.api, "contextual_headers", {}).get(
        api_description, {}
    )
    if isinstance(contextual_headers, dict):
        base_headers.update(
            {k: v for k, v in contextual_headers.items() if v is not None}
        )
    else:
        logger.warning(
            f"[{api_description}] Expected dict for contextual headers, got {type(contextual_headers)}"
        )

    # Apply explicit overrides
    if headers:
        filtered_overrides = {k: v for k, v in headers.items() if v is not None}
        base_headers.update(filtered_overrides)
        if filtered_overrides:
            logger.debug(
                f"[{api_description}] Applied {len(filtered_overrides)} explicit header overrides."
            )
        # End of if
    # End of if

    return base_headers

# End of _prepare_base_headers

def _prepare_api_headers(
    session_manager: SessionManager,  # Assume available
    driver: DriverType,
    api_description: str,
    base_headers: Dict[str, str],
    use_csrf_token: bool,
    add_default_origin: bool,
) -> Dict[str, str]:
    """Generates the final headers for an API request."""
    final_headers = base_headers.copy()
    cfg = config_schema  # Use new config system
    ua_set = False

    # Get User-Agent from browser if possible (skip session validation to prevent recursion)
    if driver and session_manager.driver:  # Simple driver check without session validation
        try:
            ua = driver.execute_script("return navigator.userAgent;")
            if ua and isinstance(ua, str):
                final_headers["User-Agent"] = ua
                ua_set = True
            # End of if
        except WebDriverException:  # type: ignore
            logger.debug(
                f"[{api_description}] WebDriver error getting User-Agent, using default."
            )
        except Exception as e:
            logger.debug(
                f"[{api_description}] Error getting User-Agent: {e}, using default."
            )
        # End of try/except
    # End of if

    # Fallback User-Agent if driver failed or wasn't available/valid
    if not ua_set:
        final_headers["User-Agent"] = random.choice(cfg.api.user_agents)
        logger.debug(
            f"[{api_description}] Using default User-Agent: {final_headers['User-Agent']}"
        )
    # End of if

    # Add Origin header for relevant methods
    http_method = base_headers.get("_method", "GET").upper()
    if add_default_origin and http_method not in ["GET", "HEAD", "OPTIONS"]:
        try:
            parsed_base_url = urlparse(cfg.api.base_url)
            origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"
            final_headers["Origin"] = origin_header_value
        except Exception as parse_err:
            logger.warning(
                f"[{api_description}] Could not parse BASE_URL for Origin header: {parse_err}"
            )
        # End of try/except
    # End of if

    # Skip dynamic header generation to prevent recursion during API requests
    # These headers are not essential for basic API functionality
    logger.debug(f"[{api_description}] Skipping dynamic header generation to prevent recursion.")

    # Add CSRF token if requested and available
    if use_csrf_token:
        csrf_token = session_manager.csrf_token
        if csrf_token:
            raw_token_val = csrf_token
            # Handle potential JSON structure in token (legacy?)
            if isinstance(csrf_token, str) and csrf_token.strip().startswith("{"):
                try:
                    token_obj = json.loads(csrf_token)
                    raw_token_val = token_obj.get("csrfToken", csrf_token)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[{api_description}] CSRF token looks like JSON but failed to parse, using raw value."
                    )
                # End of try/except
            # End of if
            final_headers["X-CSRF-Token"] = str(raw_token_val)  # Ensure string
            logger.debug(f"[{api_description}] Added X-CSRF-Token header.")
        else:
            logger.warning(
                f"[{api_description}] CSRF token requested but not found in SessionManager."
            )
        # End of if/else
    # End of if

    # Add User ID header (conditionally)
    exclude_userid_for = {  # Use set for faster lookup
        "Ancestry Facts JSON Endpoint",
        "Ancestry Person Picker",
        "CSRF Token API",
    }
    if session_manager.my_profile_id and api_description not in exclude_userid_for:
        final_headers["ancestry-userid"] = session_manager.my_profile_id.upper()
    elif api_description in exclude_userid_for and session_manager.my_profile_id:
        logger.debug(
            f"[{api_description}] Omitting 'ancestry-userid' header as configured."
        )
    # End of if/elif

    # Remove any headers with None values (e.g., if dynamic generation failed)
    final_headers = {k: v for k, v in final_headers.items() if v is not None}

    # Remove internal _method key
    final_headers.pop("_method", None)

    return final_headers

# End of _prepare_api_headers

def _sync_cookies_for_request(
    session_manager: SessionManager,
    driver: DriverType,
    api_description: str,
    attempt: int = 1,
) -> bool:
    """
    Synchronizes cookies from the WebDriver to the requests session.
    Uses smart once-per-session syncing to avoid repetitive operations.

    Args:
        session_manager: The session manager instance
        driver: The WebDriver instance
        api_description: Description of the API being called
        attempt: The current attempt number

    Returns:
        True if cookies were synced successfully, False otherwise
    """
    # Check driver validity for dynamic headers/cookies (avoid is_sess_valid() to prevent recursion)
    driver_is_valid = driver and session_manager.driver
    if not driver_is_valid:
        if attempt == 1:  # Only log on first attempt
            logger.warning(
                f"[{api_description}] Browser session invalid or driver None (Attempt {attempt}). Dynamic headers might be incomplete/stale."
            )
        return False

    # Use smart cookie syncing that only syncs once per session
    try:
        # Check if cookies are already synced for this session
        if hasattr(session_manager, '_session_cookies_synced') and session_manager._session_cookies_synced:
            # Cookies already synced, no need to sync again
            return True

        # Use the session manager's smart sync method
        session_manager._sync_cookies_to_requests()
        return True

    except Exception as e:
        logger.error(f"[{api_description}] Exception during cookie sync (Attempt {attempt}): {e}", exc_info=True)
        return False

# End of _sync_cookies_for_request

def _apply_rate_limiting(
    session_manager: SessionManager,
    api_description: str,
    attempt: int = 1,
) -> float:
    """
    Applies rate limiting wait time.

    Args:
        session_manager: The session manager instance
        api_description: Description of the API being called
        attempt: The current attempt number

    Returns:
        The wait time applied
    """
    wait_time = session_manager.dynamic_rate_limiter.wait() if session_manager.dynamic_rate_limiter else 0
    if wait_time > 0.1:  # Log only significant waits
        logger.debug(
            f"[{api_description}] Rate limit wait: {wait_time:.2f}s (Attempt {attempt})"
        )
    # End of if
    return wait_time

# End of _apply_rate_limiting

def _log_request_details(
    api_description: str,
    attempt: int,
    http_method: str,
    url: str,
    headers: Dict[str, str],
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
) -> None:
    """
    Logs the details of the API request.

    Args:
        api_description: Description of the API being called
        attempt: The current attempt number
        http_method: The HTTP method being used
        url: The URL being requested
        headers: The headers being sent
        data: Optional form data
        json_data: Optional JSON data
    """
    # Mask sensitive headers for logging
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
    for k, v_val in headers.items():
        # Ensure value is string for processing
        str_v = str(v_val)
        if k.lower() in sensitive_keys_debug and str_v and len(str_v) > 20:
            log_hdrs_debug[k] = str_v[:10] + "..." + str_v[-5:]
        else:
            log_hdrs_debug[k] = str_v
        # End of if/else
    # End of for

    # Minimal request logging to reduce noise
    if attempt > 1:  # Only log retries
        logger.debug(f"🌐 {api_description}: {http_method} {url} (attempt {attempt})")

# End of _log_request_details

def _prepare_api_request(
    session_manager: SessionManager,
    driver: DriverType,
    url: str,
    method: str,
    api_description: str,
    attempt: int,
    headers: Optional[Dict[str, str]] = None,
    referer_url: Optional[str] = None,
    use_csrf_token: bool = True,
    add_default_origin: bool = True,
    timeout: Optional[int] = None,
    cookie_jar: Optional[RequestsCookieJar] = None,  # type: ignore
    allow_redirects: bool = True,
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    json: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Prepares all aspects of an API request including headers, cookies, and rate limiting.

    Args:
        session_manager: The session manager instance
        driver: The WebDriver instance
        url: The URL to request
        method: The HTTP method to use
        api_description: Description of the API being called
        attempt: The current attempt number
        headers: Optional additional headers
        referer_url: Optional referer URL
        use_csrf_token: Whether to include the CSRF token
        add_default_origin: Whether to add the default origin header
        timeout: Optional request timeout
        cookie_jar: Optional cookie jar to use
        allow_redirects: Whether to follow redirects
        data: Optional form data
        json_data: Optional JSON data

    Returns:
        Dictionary containing all prepared request parameters
    """
    sel_cfg = config_schema.selenium  # Use new config system

    # Prepare base headers
    base_headers = _prepare_base_headers(
        method=method,
        api_description=api_description,
        referer_url=referer_url,
        headers=headers,
    )

    # Prepare request details
    request_timeout = timeout if timeout is not None else sel_cfg.api_timeout
    req_session = session_manager._requests_session
    effective_cookie_jar = cookie_jar if cookie_jar is not None else req_session.cookies
    http_method = method.upper()

    # Handle specific API quirks (e.g., allow_redirects)
    effective_allow_redirects = allow_redirects
    # Note: Match List API should allow redirects (as it did in working version from 2 months ago)

    logger.debug(
        f"[{api_description}] Preparing Request: Method={http_method}, URL={url}, Timeout={request_timeout}s, AllowRedirects={effective_allow_redirects}"
    )

    # Sync cookies
    _sync_cookies_for_request(
        session_manager=session_manager,
        driver=driver,
        api_description=api_description,
        attempt=attempt,
    )

    # Generate final headers
    final_headers = _prepare_api_headers(
        session_manager=session_manager,
        driver=driver,
        api_description=api_description,
        base_headers=base_headers,
        use_csrf_token=use_csrf_token,
        add_default_origin=add_default_origin,
    )

    # Apply rate limiting
    _apply_rate_limiting(
        session_manager=session_manager,
        api_description=api_description,
        attempt=attempt,
    )

    # Use json parameter if provided, otherwise use json_data
    effective_json_data = json if json is not None else json_data

    # Log request details
    _log_request_details(
        api_description=api_description,
        attempt=attempt,
        http_method=http_method,
        url=url,
        headers=final_headers,
        data=data,
        json_data=effective_json_data,
    )

    # Return all prepared request parameters
    return {
        "method": http_method,
        "url": url,
        "headers": final_headers,
        "data": data,
        "json": effective_json_data,  # Use 'json' for requests.request, not 'json_data'
        "timeout": request_timeout,
        "verify": True,  # Standard verification
        "allow_redirects": effective_allow_redirects,
        "cookies": effective_cookie_jar,
    }

# End of _prepare_api_request

def _execute_api_request(
    session_manager: SessionManager,
    api_description: str,
    request_params: Dict[str, Any],
    attempt: int = 1,
) -> RequestsResponseTypeOptional:
    """
    Executes an API request using the prepared request parameters.

    Args:
        session_manager: The session manager instance
        api_description: Description of the API being called
        request_params: Dictionary containing all request parameters
        attempt: The current attempt number

    Returns:
        The response object from the request, or None if an error occurred
    """
    req_session = session_manager._requests_session

    try:
        # Execute the request
        response = req_session.request(**request_params)

        # Log consolidated response details
        status = response.status_code
        reason = response.reason
        logger.debug(
            f"🌐 {api_description} request completed: {status} {reason}"
        )
        return response

    except RequestException as e:  # type: ignore
        logger.warning(
            f"[_api_req Attempt {attempt} '{api_description}'] RequestException: {type(e).__name__} - {e}"
        )
        return None
    except Exception as e:
        logger.critical(
            f"{api_description}: CRITICAL Unexpected error during request attempt {attempt}: {e}",
            exc_info=True,
        )
        return None

# End of _execute_api_request

def _process_api_response(
    response: RequestsResponseTypeOptional,
    api_description: str,
    force_text_response: bool = False,
) -> ApiResponseType:
    """
    Processes the response from an API request.

    Args:
        response: The response object from the request
        api_description: Description of the API being called
        force_text_response: Whether to force the response to be returned as text

    Returns:
        The processed response data (JSON, text, or None)
    """
    if response is None:
        return None

    # Get status code and reason
    status = response.status_code
    reason = response.reason

    # Handle successful responses (2xx)
    if response.ok:

        # Force text response if requested
        if force_text_response:
            logger.debug(f"{api_description}: Force text response requested.")
            return response.text

        # Process based on content type
        content_type = response.headers.get("content-type", "").lower()
        logger.debug(f"{api_description}: Content-Type: '{content_type}'")

        if "application/json" in content_type:
            try:
                # Handle empty response body for JSON
                json_result = response.json() if response.content else None
                logger.debug(f"{api_description}: Successfully parsed JSON response.")
                return json_result
            except JSONDecodeError as json_err:  # type: ignore
                logger.error(
                    f"{api_description}: OK ({status}), but JSON decode FAILED: {json_err}"
                )
                try:
                    logger.debug(
                        f"   << Response Text (JSON Error): {response.text[:500]}..."
                    )
                except Exception:
                    pass
                # End of try/except
                # Return None because caller expected JSON but didn't get it
                return None
            # End of try/except
        elif api_description == "CSRF Token API" and "text/plain" in content_type:
            csrf_text = response.text.strip()
            logger.debug(
                f"{api_description}: Received text/plain as expected for CSRF."
            )
            return csrf_text if csrf_text else None
        else:
            logger.debug(
                f"{api_description}: OK ({status}), Content-Type '{content_type}'. Returning raw TEXT."
            )
            return response.text
        # End of if/elif/else content_type
    # End of if response.ok

    # For non-successful responses, return the response object itself
    # This allows the caller to handle specific error cases
    return response

# End of _process_api_response

def _api_req(
    url: str,
    driver: DriverType,
    session_manager: SessionManager,  # type: ignore
    method: str = "GET",
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    json: Optional[Dict] = None,
    use_csrf_token: bool = True,
    headers: Optional[Dict[str, str]] = None,
    referer_url: Optional[str] = None,
    api_description: str = "API Call",
    timeout: Optional[int] = None,
    cookie_jar: Optional[RequestsCookieJar] = None,  # type: ignore
    allow_redirects: bool = True,
    force_text_response: bool = False,
    add_default_origin: bool = True,
    redirect_count: int = 0,
) -> Union[ApiResponseType, RequestsResponseTypeOptional]:
    """
    Makes an HTTP request using the shared requests.Session from SessionManager.
    Handles dynamic header generation, cookie synchronization, rate limiting,
    retries, and basic response processing. Includes enhanced logging.

    Returns: Parsed JSON (dict/list), raw text (str), None on retryable failure,
             or Response object on non-retryable error/redirect disabled.
    """
    # --- Diagnostic Log: Function Entry ---
    logger.debug(f"[_api_req ENTRY] api_description: '{api_description}', url: {url}")

    # --- Step 1: Validate prerequisites ---
    if not session_manager or not session_manager._requests_session:
        logger.error(
            f"{api_description}: Aborting - SessionManager or internal requests_session missing."
        )
        return None
    # End of if
    if not config_schema:
        logger.error(f"{api_description}: Aborting - Config schema not loaded.")
        return None
    # End of if

    cfg = config_schema.api
    sel_cfg = config_schema.selenium

    # --- Step 2: Get Retry Configuration ---
    max_retries = cfg.max_retries
    initial_delay = cfg.initial_delay
    backoff_factor = cfg.retry_backoff_factor
    max_delay = cfg.max_delay
    retry_status_codes = set(cfg.retry_status_codes)

    # --- Diagnostic Log: Retry Params ---
    logger.debug(
        f"[_api_req PRE-LOOP] api_description: '{api_description}', max_retries: {max_retries}, "
        f"initial_delay: {initial_delay}, backoff_factor: {backoff_factor}"
    )

    # --- Step 3: Execute Request with Retry Loop ---
    retries_left = max_retries
    last_exception: Optional[Exception] = None
    response: RequestsResponseTypeOptional = None
    current_delay = initial_delay

    while retries_left > 0:
        attempt = max_retries - retries_left + 1
        # --- Diagnostic Log: Loop Entry ---
        logger.debug(
            f"[_api_req LOOP ENTRY] api_description: '{api_description}', attempt: {attempt}/{max_retries}, retries_left: {retries_left}"
        )

        try:
            # --- Step 3.1: Prepare the request ---
            request_params = _prepare_api_request(
                session_manager=session_manager,
                driver=driver,
                url=url,
                method=method,
                api_description=api_description,
                attempt=attempt,
                headers=headers,
                referer_url=referer_url,
                use_csrf_token=use_csrf_token,
                add_default_origin=add_default_origin,
                timeout=timeout,
                cookie_jar=cookie_jar,
                allow_redirects=allow_redirects,
                data=data,
                json_data=json_data,
                json=json,  # Pass the json parameter if provided
            )

            # --- Step 3.2: Execute the request ---
            response = _execute_api_request(
                session_manager=session_manager,
                api_description=api_description,
                request_params=request_params,
                attempt=attempt,
            )

            # If request failed with an exception, response will be None
            if response is None:
                retries_left -= 1
                if retries_left <= 0:
                    logger.error(
                        f"{api_description}: Request failed after {max_retries} attempts."
                    )
                    return None
                else:
                    sleep_time = min(
                        current_delay * (backoff_factor ** (attempt - 1)), max_delay
                    ) + random.uniform(0, 0.2)
                    sleep_time = max(0.1, sleep_time)
                    logger.warning(
                        f"{api_description}: Request error (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    current_delay *= backoff_factor
                    continue  # Go to next iteration of the while loop
                # End of if/else retries_left
            # End of if response is None

            # --- Step 3.3: Check for retryable status codes ---
            status = response.status_code
            reason = response.reason

            if status in retry_status_codes:
                retries_left -= 1
                logger.warning(
                    f"[_api_req Attempt {attempt} '{api_description}'] Received retryable status: {status} {reason}"
                )
                last_exception = HTTPError(f"{status} Error", response=response)  # type: ignore
                if retries_left <= 0:
                    logger.error(
                        f"{api_description}: Failed after {max_retries} attempts (Final Status {status}). Returning Response object."
                    )
                    try:
                        logger.debug(
                            f"   << Final Response Text (Retry Fail): {response.text[:500]}..."
                        )
                    except Exception:
                        pass
                    # End of try/except
                    return response
                else:
                    sleep_time = min(
                        current_delay * (backoff_factor ** (attempt - 1)), max_delay
                    ) + random.uniform(0, 0.2)
                    sleep_time = max(0.1, sleep_time)
                    if status == 429:  # Too Many Requests
                        if session_manager.dynamic_rate_limiter:
                            session_manager.dynamic_rate_limiter.increase_delay()
                    # End of if
                    logger.warning(
                        f"{api_description}: Status {status} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s..."
                    )
                    try:
                        logger.debug(
                            f"   << Response Text (Retry): {response.text[:500]}..."
                        )
                    except Exception:
                        pass
                    # End of try/except
                    time.sleep(sleep_time)
                    current_delay *= backoff_factor
                    continue  # Go to next iteration of the while loop
                # End of if/else retries_left
            # End of if status in retry_status_codes

            # --- Step 3.4: Handle redirects ---
            # Handle redirects if allow_redirects is False
            if 300 <= status < 400 and not request_params["allow_redirects"]:
                logger.warning(
                    f"{api_description}: Status {status} {reason} (Redirects Disabled). Returning Response object."
                )
                logger.debug(
                    f"   << Redirect Location: {response.headers.get('Location')}"
                )
                return response
            # End of if

            # Generalized manual redirect handling for all 3xx codes
            if 300 <= status < 400 and request_params["allow_redirects"]:
                location = response.headers.get('Location')
                redirect_count = getattr(request_params, 'redirect_count', 0)
                if location and redirect_count < 5:
                    logger.warning(
                        f"{api_description}: Received {status} {reason}. Following redirect to {location}. (Redirect {redirect_count+1}/5)"
                    )
                    # Switch to GET for 301, 302, 303 if original method is POST/PUT
                    redirect_method = method
                    if status in [301, 302, 303] and method.upper() in ["POST", "PUT"]:
                        redirect_method = "GET"
                        logger.debug(f"Switching method to GET for redirect status {status}.")
                    # Preserve headers/cookies
                    new_headers = dict(headers) if headers else None
                    # Add diagnostics for redirect chain/history
                    logger.info(f"Redirect chain: {api_description} -> {location}")
                    # Pass redirect_count to prevent infinite loop
                    return _api_req(
                        url=location,
                        driver=driver,
                        session_manager=session_manager,
                        method=redirect_method,
                        data=None if redirect_method == "GET" else data,
                        json_data=None if redirect_method == "GET" else json_data,
                        json=None if redirect_method == "GET" else json,
                        use_csrf_token=use_csrf_token,
                        headers=new_headers,
                        referer_url=referer_url,
                        api_description=api_description + f" (redirected {redirect_count+1})",
                        timeout=timeout,
                        cookie_jar=cookie_jar,
                        allow_redirects=False,  # Prevent further recursion
                        force_text_response=force_text_response,
                        add_default_origin=add_default_origin,
                        # Custom param to track redirect count
                        redirect_count=redirect_count+1,
                    )
                else:
                    logger.warning(
                        f"{api_description}: Unexpected final status {status} {reason} (Redirects Enabled). Returning Response object."
                    )
                    logger.debug(
                        f"   << Redirect Location: {location}"
                    )
                    return response
            # End of if

            # --- Step 3.5: Handle non-retryable error status codes ---
            if not response.ok:
                # For login verification API, use debug level for 401/403 errors
                if (
                    api_description == "API Login Verification (header/dna)"
                    and status in [401, 403]
                ):
                    logger.debug(
                        f"[_api_req Attempt {attempt} '{api_description}'] Received expected status: {status} {reason}"
                    )
                else:
                    logger.warning(
                        f"[_api_req Attempt {attempt} '{api_description}'] Received NON-retryable error status: {status} {reason}"
                    )
                if status in [401, 403]:
                    # For login verification API, don't log a warning as this is expected when not logged in
                    if api_description == "API Login Verification (header/dna)":
                        logger.debug(
                            f"{api_description}: API call returned {status} {reason}. User not logged in."
                        )
                    else:
                        logger.warning(
                            f"{api_description}: API call failed {status} {reason}. Session expired/invalid?"
                        )
                    session_manager.session_ready = False  # Mark session as not ready
                else:
                    logger.error(
                        f"{api_description}: Non-retryable error: {status} {reason}."
                    )
                # Log error response only for non-auth errors and if response has useful content
                if status not in [401, 403, 429] and hasattr(response, 'text'):
                    try:
                        error_text = response.text[:200]
                        if error_text.strip() and not error_text.startswith('<!DOCTYPE'):  # Skip HTML error pages
                            logger.debug(f"   << Error Response: {error_text}...")
                    except Exception:
                        pass
                return response  # Return the Response object for the caller to handle
            # End of if not response.ok

            # --- Step 3.6: Process successful response ---
            if response.ok:
                if session_manager.dynamic_rate_limiter:
                    session_manager.dynamic_rate_limiter.decrease_delay()  # Success, decrease future delay

                # Process the response
                return _process_api_response(
                    response=response,
                    api_description=api_description,
                    force_text_response=force_text_response,
                )
            # End of if response.ok

        # --- Handle exceptions during the request attempt ---
        except RequestException as e:  # type: ignore
            logger.warning(
                f"[_api_req Attempt {attempt} '{api_description}'] RequestException: {type(e).__name__} - {e}"
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
            # End of if/else retries_left
        except Exception as e:
            logger.critical(
                f"{api_description}: CRITICAL Unexpected error during request attempt {attempt}: {e}",
                exc_info=True,
            )
            return None  # Return None on unexpected errors within the loop
        # End of try/except block for request attempt

        # If we get here, the request was successful and processed
        break  # Exit the retry loop
    # End of while retries_left > 0

    # --- Diagnostic Log: Function Exit ---
    logger.debug(
        f"[_api_req EXIT] api_description: '{api_description}', attempts: {max_retries - retries_left + 1}/{max_retries}"
    )

    # --- Should only be reached if loop completes without success (e.g., retries exhausted) ---
    if response is None:
        logger.error(
            f"{api_description}: Exited retry loop. Last Exception: {last_exception}."
        )
        return None
    # End of if

    # Return the last response (this should be a non-retryable error response)
    logger.debug(
        f"[_api_req '{api_description}'] Returning last Response object (Status: {response.status_code})."
    )
    return response

# End of _api_req

def make_ube(driver: DriverType) -> Optional[str]:
    if not driver:
        return None
    # End of if
    try:
        _ = driver.title  # Quick check for session validity
    except WebDriverException as e:  # type: ignore
        logger.warning(
            f"Cannot generate UBE header: Session invalid/unresponsive ({type(e).__name__})."
        )
        return None
    # End of try/except

    ancsessionid: Optional[str] = None
    try:
        # Try fetching specific cookie first
        cookie_obj = driver.get_cookie("ANCSESSIONID")
        if cookie_obj and isinstance(cookie_obj, dict) and "value" in cookie_obj:
            ancsessionid = cookie_obj["value"]
        elif ancsessionid is None:  # Fallback to getting all cookies if specific fails
            cookies_dict = {
                c["name"]: c["value"]
                for c in driver.get_cookies()
                if isinstance(c, dict) and "name" in c
            }
            ancsessionid = cookies_dict.get("ANCSESSIONID")
        # End of if/elif
        if not ancsessionid:
            logger.warning("ANCSESSIONID cookie not found. Cannot generate UBE header.")
            return None
        # End of if
    except (NoSuchCookieException, WebDriverException) as cookie_e:  # type: ignore
        logger.warning(f"Error getting ANCSESSIONID cookie for UBE header: {cookie_e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error getting ANCSESSIONID for UBE: {e}", exc_info=True
        )
        return None
    # End of try/except

    # Construct UBE data payload
    event_id = (
        "00000000-0000-0000-0000-000000000000"  # Typically zero GUID for this header
    )
    correlated_id = str(uuid.uuid4())  # Unique ID for this interaction
    screen_name_standard = (
        "ancestry : uk : en : dna-matches-ui : match-list : 1"  # Example
    )
    screen_name_legacy = "ancestry uk : dnamatches-matchlistui : list"  # Example
    user_consent = "necessary|preference|performance|analytics1st|analytics3rd|advertising1st|advertising3rd|attribution3rd"  # Example consent string
    ube_data = {
        "eventId": event_id,
        "correlatedScreenViewedId": correlated_id,
        "correlatedSessionId": ancsessionid,
        "screenNameStandard": screen_name_standard,
        "screenNameLegacy": screen_name_legacy,
        "userConsent": user_consent,
        "vendors": "adobemc",  # Example
        "vendorConfigurations": "{}",  # Usually empty JSON string
    }

    # Encode the payload
    try:
        json_payload = json.dumps(ube_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        return encoded_payload
    except (json.JSONDecodeError, TypeError, binascii.Error) as encode_e:
        logger.error(f"Error encoding UBE header data: {encode_e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error encoding UBE header: {e}", exc_info=True)
        return None
    # End of try/except

# End of make_ube

def make_newrelic(driver: DriverType) -> Optional[str]:
    # This function generates a plausible NewRelic header structure.
    # Exact values might vary, but the format is generally consistent.
    # The driver argument is kept for potential future use but isn't strictly needed now.
    try:
        trace_id = uuid.uuid4().hex[:16]  # Shorter trace ID part
        span_id = uuid.uuid4().hex[:16]  # Span ID
        # These IDs seem static or tied to Ancestry's NewRelic account/app
        account_id = "1690570"
        app_id = "1588726612"
        license_key_part = "2611750"  # Obfuscated/partial license key part

        newrelic_data = {
            "v": [0, 1],  # Version info
            "d": {
                "ty": "Browser",  # Type
                "ac": account_id,
                "ap": app_id,
                "id": span_id,
                "tr": trace_id,
                "ti": int(time.time() * 1000),  # Timestamp in ms
                "tk": license_key_part,
            },
        }
        json_payload = json.dumps(newrelic_data, separators=(",", ":")).encode("utf-8")
        encoded_payload = base64.b64encode(json_payload).decode("utf-8")
        return encoded_payload
    except (json.JSONDecodeError, TypeError, binascii.Error) as encode_e:
        logger.error(f"Error generating NewRelic header: {encode_e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating NewRelic header: {e}", exc_info=True)
        return None
    # End of try/except

# End of make_newrelic

def make_traceparent(driver: DriverType) -> Optional[str]:
    # Generates a W3C Trace Context traceparent header.
    # Driver argument kept for consistency, not currently used.
    try:
        version = "00"  # Standard version
        trace_id = uuid.uuid4().hex  # Full 32-char trace ID
        parent_id = uuid.uuid4().hex[:16]  # 16-char parent/span ID
        flags = "01"  # Sampled flag (usually 01)
        traceparent = f"{version}-{trace_id}-{parent_id}-{flags}"
        return traceparent
    except Exception as e:
        logger.error(f"Error generating traceparent header: {e}", exc_info=True)
        return None
    # End of try/except

# End of make_traceparent

def make_tracestate(driver: DriverType) -> Optional[str]:
    # Generates a tracestate header, often including NewRelic state.
    # Driver argument kept for consistency, not currently used.
    try:
        # NewRelic specific part of tracestate
        tk = "2611750"  # Corresponds to license key part in newrelic header
        account_id = "1690570"
        app_id = "1588726612"
        span_id = uuid.uuid4().hex[:16]  # Another span ID
        timestamp = int(time.time() * 1000)
        # Format follows NewRelic's tracestate structure
        tracestate = f"{tk}@nr=0-1-{account_id}-{app_id}-{span_id}----{timestamp}"
        # Other vendors could potentially be added, comma-separated
        return tracestate
    except Exception as e:
        logger.error(f"Error generating tracestate header: {e}", exc_info=True)
        return None
    # End of try/except

# End of make_tracestate

# ----------------------------------------------------------------------------
# Login Functions (Remain in utils.py)
# ----------------------------------------------------------------------------
TWO_STEP_VERIFICATION_HEADER_SELECTOR = "h1.two-step-verification-header"

@time_wait("Handle 2FA Page")
def handle_twoFA(session_manager: SessionManager) -> bool:  # type: ignore
    if session_manager.driver is None:
        logger.error("handle_twoFA: SessionManager driver is None. Cannot proceed.")
        return False
    # End of if
    driver = session_manager.driver
    element_wait = WebDriverWait(driver, config_schema.selenium.explicit_wait)
    page_wait = WebDriverWait(driver, config_schema.selenium.page_load_timeout)
    short_wait = WebDriverWait(driver, config_schema.selenium.implicit_wait)
    try:
        print(
            "Two-factor authentication required. Please check your email or phone for a verification code."
        )
        logger.debug("Handling Two-Factor Authentication (2FA)...")
        try:
            logger.debug(
                f"Waiting for 2FA page header using selector: '{TWO_STEP_VERIFICATION_HEADER_SELECTOR}'"
            )
            element_wait.until(
                EC.visibility_of_element_located(  # type: ignore
                    (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)  # type: ignore
                )
            )
            logger.debug("2FA page header detected.")
        except TimeoutException:  # type: ignore
            logger.debug("Did not detect 2FA page header within timeout.")
            if (
                login_status(session_manager, disable_ui_fallback=True) is True
            ):  # API check only for speed
                logger.info(
                    "User appears logged in after checking for 2FA page. Assuming 2FA handled/skipped."
                )
                return True
            # End of if
            logger.warning(
                "Assuming 2FA not required or page didn't load correctly (header missing)."
            )
            return False  # Return False if 2FA page wasn't detected and not logged in
        except WebDriverException as e:  # type: ignore
            logger.error(f"WebDriverException waiting for 2FA header: {e}")
            return False
        # End of try/except

        # Try clicking SMS button
        try:
            logger.debug(f"Waiting for 2FA 'Send Code' (SMS) button: '{TWO_FA_SMS_SELECTOR}'")  # type: ignore
            sms_button_clickable = short_wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR))  # type: ignore
            )
            if sms_button_clickable:
                logger.debug(
                    "Attempting to click 'Send Code' button using JavaScript..."
                )
                try:
                    driver.execute_script("arguments[0].click();", sms_button_clickable)
                    logger.debug("'Send Code' button clicked.")
                    # Wait briefly for code input field to potentially appear
                    try:
                        logger.debug(f"Waiting for 2FA code input field: '{TWO_FA_CODE_INPUT_SELECTOR}'")  # type: ignore
                        WebDriverWait(driver, 5).until(  # type: ignore
                            EC.visibility_of_element_located(  # type: ignore
                                (By.CSS_SELECTOR, TWO_FA_CODE_INPUT_SELECTOR)  # type: ignore
                            )
                        )
                        logger.debug(
                            "Code input field appeared after clicking 'Send Code'."
                        )
                    except TimeoutException:  # type: ignore
                        logger.warning(
                            "Code input field did not appear/become visible after clicking 'Send Code'."
                        )
                    except WebDriverException as e_input:  # type: ignore
                        logger.error(f"Error waiting for 2FA code input field: {e_input}. Check selector: {TWO_FA_CODE_INPUT_SELECTOR}")  # type: ignore
                    # End of inner try/except
                except WebDriverException as click_err:  # type: ignore
                    logger.error(
                        f"Error clicking 'Send Code' button via JS: {click_err}"
                    )
                    # Don't return False yet, proceed to wait for manual entry
                # End of try/except
            else:
                # This case should be rare if element_to_be_clickable succeeded
                logger.error(
                    "'Send Code' button found but reported as not clickable by Selenium."
                )
                # Proceed to wait for manual entry anyway
            # End of if/else sms_button_clickable
        except TimeoutException:  # type: ignore
            logger.error(
                "Timeout finding or waiting for clickability of the 2FA 'Send Code' button."
            )
            # Proceed to wait for manual entry
        except ElementNotInteractableException:  # type: ignore
            logger.error("'Send Code' button not interactable (potentially obscured).")
            # Proceed to wait for manual entry
        except WebDriverException as e:  # type: ignore
            logger.error(
                f"WebDriverException interacting with 2FA 'Send Code' button: {e}",
                exc_info=False,
            )
            # Proceed to wait for manual entry
        except Exception as e:
            logger.error(
                f"Unexpected error clicking 2FA 'Send Code' button: {e}", exc_info=True
            )
            # Proceed to wait for manual entry
        # End of try/except block for clicking SMS button

        # Wait for user action (manual code entry and submission)
        code_entry_timeout = config_schema.selenium.two_fa_code_entry_timeout
        logger.warning(
            f"Waiting up to {code_entry_timeout}s for user to manually enter 2FA code and submit..."
        )
        start_time = time.time()
        user_action_detected = False
        while time.time() - start_time < code_entry_timeout:
            try:
                # Check if the 2FA header is GONE (indicates page change/submission)
                WebDriverWait(driver, 0.5).until_not(  # type: ignore
                    EC.visibility_of_element_located(  # type: ignore
                        (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)  # type: ignore
                    )
                )
                logger.info(
                    "2FA page elements disappeared, assuming user submitted code."
                )
                user_action_detected = True
                break  # Exit wait loop
            except TimeoutException:  # type: ignore
                # Header still present, continue waiting
                time.sleep(2)  # Check every 2 seconds
            except NoSuchElementException:  # type: ignore
                # Header element gone, assume submitted
                logger.info("2FA header element no longer present.")
                user_action_detected = True
                break  # Exit wait loop
            except WebDriverException as e:  # type: ignore
                # Handle potential errors during the check
                logger.error(
                    f"WebDriver error checking for 2FA header during wait: {e}"
                )
                # If session dies here, login_status check later will fail
                break  # Exit wait loop on error
            except Exception as e:
                logger.error(
                    f"Unexpected error checking for 2FA header during wait: {e}"
                )
                break  # Exit wait loop on error
            # End of try/except
        # End of while loop

        # Final check after waiting
        if user_action_detected:
            logger.info("Re-checking login status after potential 2FA submission...")
            time.sleep(1)  # Allow page to settle
            final_status = login_status(
                session_manager, disable_ui_fallback=False
            )  # Use UI fallback for reliability
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
            # End of if/else
        else:
            logger.error(
                f"Timed out ({code_entry_timeout}s) waiting for user 2FA action (page did not change)."
            )
            return False
        # End of if/else user_action_detected

    except WebDriverException as e:  # type: ignore
        logger.error(f"WebDriverException during 2FA handling: {e}")
        if not is_browser_open(driver):
            logger.error("Session invalid after WebDriverException during 2FA.")
        # End of if
        return False
    except Exception as e:
        logger.error(f"Unexpected error during 2FA handling: {e}", exc_info=True)
        return False
    # End of try/except for overall 2FA handling

    # Should not be reachable unless an early return was missed
    # return False

# End of handle_twoFA

def enter_creds(driver: WebDriver) -> bool:  # type: ignore
    element_wait = WebDriverWait(driver, config_schema.selenium.explicit_wait)
    short_wait = WebDriverWait(driver, 5)  # Short wait for quick checks
    time.sleep(random.uniform(0.5, 1.0))  # Small random wait
    try:
        logger.debug("Entering Credentials and Signing In...")

        # --- Username ---
        logger.debug(f"Waiting for username input: '{USERNAME_INPUT_SELECTOR}'...")  # type: ignore
        username_input = element_wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))  # type: ignore
        )
        logger.debug("Username input field found.")
        try:
            # Attempt to clear field robustly
            username_input.click()
            time.sleep(0.1)
            username_input.clear()
            time.sleep(0.1)
            # JS clear as fallback/additional measure
            driver.execute_script("arguments[0].value = '';", username_input)
            time.sleep(0.1)
        except (ElementNotInteractableException, StaleElementReferenceException) as e:  # type: ignore
            logger.warning(
                f"Issue clicking/clearing username field ({type(e).__name__}). Proceeding cautiously."
            )
        except WebDriverException as e:  # type: ignore
            logger.error(
                f"WebDriverException clicking/clearing username: {e}. Aborting."
            )
            return False
        # End of try/except

        # Check config value exists
        ancestry_username = config_schema.api.username
        if not ancestry_username:
            raise ValueError("ANCESTRY_USERNAME configuration is missing.")
        # End of if
        logger.debug("Entering username...")
        username_input.send_keys(ancestry_username)
        logger.debug("Username entered.")
        time.sleep(random.uniform(0.2, 0.4))

        # --- Password ---
        logger.debug(f"Waiting for password input: '{PASSWORD_INPUT_SELECTOR}'...")  # type: ignore
        password_input = element_wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))  # type: ignore
        )
        logger.debug("Password input field found.")
        try:
            # Attempt to clear field robustly
            password_input.click()
            time.sleep(0.1)
            password_input.clear()
            time.sleep(0.1)
            driver.execute_script("arguments[0].value = '';", password_input)
            time.sleep(0.1)
        except (ElementNotInteractableException, StaleElementReferenceException) as e:  # type: ignore
            logger.warning(
                f"Issue clicking/clearing password field ({type(e).__name__}). Proceeding cautiously."
            )
        except WebDriverException as e:  # type: ignore
            logger.error(
                f"WebDriverException clicking/clearing password: {e}. Aborting."
            )
            return False
        # End of try/except

        # Check config value exists
        ancestry_password = config_schema.api.password
        if not ancestry_password:
            raise ValueError("ANCESTRY_PASSWORD configuration is missing.")
        # End of if
        logger.debug("Entering password: ***")
        password_input.send_keys(ancestry_password)
        logger.debug("Password entered.")
        time.sleep(random.uniform(0.3, 0.6))

        # --- Sign In Button ---
        sign_in_button = None
        try:
            logger.debug(f"Waiting for sign in button presence: '{SIGN_IN_BUTTON_SELECTOR}'...")  # type: ignore
            WebDriverWait(driver, 5).until(  # type: ignore
                EC.presence_of_element_located((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))  # type: ignore
            )
            logger.debug("Waiting for sign in button clickability...")
            sign_in_button = short_wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))  # type: ignore
            )
            logger.debug("Sign in button located and deemed clickable.")
        except TimeoutException:  # type: ignore
            logger.error("Sign in button not found or not clickable within timeout.")
            logger.warning("Attempting fallback: Sending RETURN key to password field.")
            try:
                password_input.send_keys(Keys.RETURN)
                logger.info("Fallback RETURN key sent to password field.")
                return True  # Assume submission worked
            except (WebDriverException, ElementNotInteractableException) as key_e:  # type: ignore
                logger.error(f"Failed to send RETURN key: {key_e}")
                return False  # Fallback also failed
            # End of try/except
        except WebDriverException as find_e:  # type: ignore
            logger.error(f"Unexpected WebDriver error finding sign in button: {find_e}")
            return False
        # End of try/except

        # Click button using multiple methods if needed
        click_successful = False
        if sign_in_button:
            # Attempt 1: Standard click
            try:
                logger.debug("Attempting standard click on sign in button...")
                sign_in_button.click()
                logger.debug("Standard click executed.")
                click_successful = True
            except (ElementClickInterceptedException, ElementNotInteractableException, StaleElementReferenceException) as click_intercept_err:  # type: ignore
                logger.warning(
                    f"Standard click failed ({type(click_intercept_err).__name__}). Trying JS click..."
                )
            except WebDriverException as click_err:  # type: ignore
                logger.error(
                    f"WebDriver error during standard click: {click_err}. Trying JS click..."
                )
            # End of try/except standard click

            # Attempt 2: JavaScript click (if standard failed)
            if not click_successful:
                try:
                    logger.debug("Attempting JavaScript click on sign in button...")
                    driver.execute_script("arguments[0].click();", sign_in_button)
                    logger.info("JavaScript click executed.")
                    click_successful = True
                except WebDriverException as js_click_e:  # type: ignore
                    logger.error(f"Error during JavaScript click: {js_click_e}")
                # End of try/except JS click
            # End of if not click_successful

            # Attempt 3: Send RETURN key (if clicks failed)
            if not click_successful:
                logger.warning(
                    "Both standard and JS clicks failed. Attempting fallback: Sending RETURN key."
                )
                try:
                    # Send to password field as it likely still has focus
                    password_input.send_keys(Keys.RETURN)
                    logger.info(
                        "Fallback RETURN key sent to password field after failed clicks."
                    )
                    click_successful = True
                except (WebDriverException, ElementNotInteractableException) as key_e:  # type: ignore
                    logger.error(
                        f"Failed to send RETURN key as final fallback: {key_e}"
                    )
                # End of try/except RETURN key
            # End of if not click_successful (after JS)
        # End of if sign_in_button

        return click_successful  # Return True if any click/submit method seemed to work

    except (TimeoutException, NoSuchElementException) as e:  # type: ignore
        logger.error(
            f"Timeout or Element not found finding username/password field: {e}"
        )
        return False
    except ValueError as ve:  # Catch missing config
        logger.critical(f"Configuration Error: {ve}")
        # Re-raise config error as it's critical
        raise ve
    except WebDriverException as e:  # type: ignore
        logger.error(f"WebDriver error entering credentials: {e}")
        if not is_browser_open(driver):
            logger.error("Session invalid during credential entry.")
        # End of if
        return False
    except Exception as e:
        logger.error(f"Unexpected error entering credentials: {e}", exc_info=True)
        return False
    # End of try/except

# End of enter_creds

@retry(MAX_RETRIES=2, BACKOFF_FACTOR=1, MAX_DELAY=3)  # Add retry for consent handling
def consent(driver: WebDriver) -> bool:  # type: ignore
    """Handles the cookie consent banner if present."""
    if not driver:
        logger.error("consent: WebDriver instance is None.")
        return False
    # End of if

    logger.debug(f"Checking for cookie consent overlay: '{COOKIE_BANNER_SELECTOR}'")  # type: ignore
    overlay_element = None
    try:
        # Use a short wait to find the banner
        overlay_element = WebDriverWait(driver, 3).until(  # type: ignore
            EC.presence_of_element_located((By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR))  # type: ignore
        )
        logger.debug("Cookie consent overlay DETECTED.")
    except TimeoutException:  # type: ignore
        logger.debug("Cookie consent overlay not found. Assuming no consent needed.")
        return True  # No banner, proceed
    except WebDriverException as e:  # Catch errors finding element # type: ignore
        logger.error(f"Error checking for consent banner: {e}")
        return False  # Indicate failure if check fails
    # End of try/except

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
                WebDriverWait(driver, 1).until_not(  # type: ignore
                    EC.presence_of_element_located(  # type: ignore
                        (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)  # type: ignore
                    )
                )
                logger.debug("Cookie consent overlay REMOVED successfully via JS.")
                removed_via_js = True
                return True  # Success
            except TimeoutException:  # type: ignore
                logger.warning(
                    "Consent overlay still present after JS removal attempt."
                )
            except WebDriverException as verify_err:  # type: ignore
                logger.warning(
                    f"Error verifying overlay removal after JS: {verify_err}"
                )
            # End of try/except verification
        except WebDriverException as js_err:  # type: ignore
            logger.warning(
                f"Error removing consent overlay via JS: {js_err}. Trying button click..."
            )
        except Exception as e:  # Catch other unexpected errors during JS removal
            logger.warning(
                f"Unexpected error during JS removal of consent: {e}. Trying button click..."
            )
        # End of try/except JS removal
    # End of if overlay_element

    # Attempt 2: Try clicking the specific accept button if JS removal failed/skipped
    if not removed_via_js:
        logger.debug(
            f"JS removal failed/skipped. Trying specific accept button: '{CONSENT_ACCEPT_BUTTON_SELECTOR}'"
        )
        try:
            # Wait for the button to be clickable
            accept_button = WebDriverWait(driver, 3).until(  # type: ignore
                EC.element_to_be_clickable(  # type: ignore
                    (By.CSS_SELECTOR, CONSENT_ACCEPT_BUTTON_SELECTOR)  # type: ignore
                )
            )
            logger.info("Found specific clickable accept button.")

            # Try standard click first
            try:
                accept_button.click()
                logger.info("Clicked accept button successfully.")
                # Verify removal
                WebDriverWait(driver, 2).until_not(  # type: ignore
                    EC.presence_of_element_located(  # type: ignore
                        (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)  # type: ignore
                    )
                )
                logger.debug("Consent overlay gone after clicking accept button.")
                return True  # Success
            except ElementClickInterceptedException:  # type: ignore
                logger.warning(
                    "Click intercepted for accept button, trying JS click..."
                )
                # Fall through to JS click attempt
            except (
                TimeoutException,
                NoSuchElementException,
            ):  # If overlay gone after click # type: ignore
                logger.debug(
                    "Consent overlay likely gone after standard click (verification timed out/not found)."
                )
                return True
            except WebDriverException as click_err:  # type: ignore
                logger.error(
                    f"Error during standard click on accept button: {click_err}. Trying JS click..."
                )
            # End of try/except standard click

            # Try JS click as fallback
            try:
                logger.debug("Attempting JS click on accept button...")
                driver.execute_script("arguments[0].click();", accept_button)
                logger.info("Clicked accept button via JS successfully.")
                # Verify removal
                WebDriverWait(driver, 2).until_not(  # type: ignore
                    EC.presence_of_element_located(  # type: ignore
                        (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR)  # type: ignore
                    )
                )
                logger.debug("Consent overlay gone after JS clicking accept button.")
                return True  # Success
            except (
                TimeoutException,
                NoSuchElementException,
            ):  # If overlay gone after JS click # type: ignore
                logger.debug(
                    "Consent overlay likely gone after JS click (verification timed out/not found)."
                )
                return True
            except WebDriverException as js_click_err:  # type: ignore
                logger.error(f"Failed JS click for accept button: {js_click_err}")
            # End of try/except JS click

        except TimeoutException:  # type: ignore
            logger.warning(
                f"Specific accept button '{CONSENT_ACCEPT_BUTTON_SELECTOR}' not found or not clickable."
            )
        except (
            WebDriverException
        ) as find_err:  # Catch errors finding/interacting # type: ignore
            logger.error(f"Error finding/clicking specific accept button: {find_err}")
        except Exception as e:  # Catch other unexpected errors
            logger.error(
                f"Unexpected error handling consent button: {e}", exc_info=True
            )
        # End of try/except button click block
    # End of if not removed_via_js

    # If both JS removal and button click failed
    logger.error("Could not remove consent overlay via JS or button click.")
    return False

# End of consent

# Login Main Function
def _check_initial_login_status(session_manager: SessionManager) -> Optional[str]:
    """
    Check if user is already logged in before attempting login process.

    Performs a quick API-based login status check to avoid unnecessary
    navigation and credential entry if the user is already authenticated.

    Args:
        session_manager: The SessionManager instance containing authentication state.

    Returns:
        Optional[str]: "LOGIN_SUCCEEDED" if already logged in, None otherwise.

    Example:
        >>> status = _check_initial_login_status(session_manager)
        >>> if status == "LOGIN_SUCCEEDED":
        ...     print("User already authenticated")
    """
    initial_status = login_status(
        session_manager, disable_ui_fallback=True
    )  # API check only for speed
    if initial_status is True:
        print("Already logged in. No need to sign in again.")
        return "LOGIN_SUCCEEDED"
    return None


def _navigate_to_signin_page(driver: WebDriver, session_manager: SessionManager) -> str:  # type: ignore
    """
    Navigate to the Ancestry sign-in page and handle redirects or errors.

    Attempts to navigate to the sign-in page and waits for the username input
    field to appear. Handles cases where navigation fails or the user is
    already logged in and gets redirected.

    Args:
        driver: The WebDriver instance for browser automation.
        session_manager: The SessionManager instance for login status checks.

    Returns:
        str: "NAVIGATION_SUCCESS" if successful, "LOGIN_SUCCEEDED" if already
             logged in, or "LOGIN_FAILED_NAVIGATION" if navigation fails.

    Example:
        >>> result = _navigate_to_signin_page(driver, session_manager)
        >>> if result == "NAVIGATION_SUCCESS":
        ...     print("Ready to enter credentials")
    """
    signin_url = urljoin(config_schema.api.base_url, "account/signin")

    # Wait for username input as indication of page load
    if not nav_to_page(
        driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager  # type: ignore
    ):
        # Navigation failed or redirected. Check if already logged in.
        logger.debug(
            "Navigation to sign-in page failed/redirected. Checking login status..."
        )
        current_status = login_status(
            session_manager, disable_ui_fallback=True
        )  # API check only for speed
        if current_status is True:
            logger.info(
                "Detected as already logged in during navigation attempt. Login considered successful."
            )
            return "LOGIN_SUCCEEDED"
        else:
            logger.error("Failed to navigate to login page (and not logged in).")
            return "LOGIN_FAILED_NAVIGATION"

    logger.debug("Successfully navigated to sign-in page.")
    return "NAVIGATION_SUCCESS"


def _handle_credential_entry(driver: WebDriver) -> str:  # type: ignore
    """
    Handle credential entry and check for authentication errors.

    Attempts to enter user credentials and checks for various error conditions
    including invalid credentials alerts and generic error messages. Provides
    detailed error classification for troubleshooting.

    Args:
        driver: The WebDriver instance for browser automation.

    Returns:
        str: "CREDENTIALS_SUCCESS" if successful, or specific error codes like
             "LOGIN_FAILED_BAD_CREDS", "LOGIN_FAILED_ERROR_DISPLAYED", or
             "LOGIN_FAILED_CREDS_ENTRY" for different failure scenarios.

    Example:
        >>> result = _handle_credential_entry(driver)
        >>> if result == "CREDENTIALS_SUCCESS":
        ...     print("Credentials entered successfully")
        >>> elif result == "LOGIN_FAILED_BAD_CREDS":
        ...     print("Invalid username or password")
    """
    if not enter_creds(driver):
        logger.error("Failed during credential entry or submission.")
        # Check for specific error messages on the page
        try:
            # Check for specific 'invalid credentials' message
            WebDriverWait(driver, 1).until(  # type: ignore
                EC.presence_of_element_located(  # type: ignore
                    (By.CSS_SELECTOR, FAILED_LOGIN_SELECTOR)  # type: ignore
                )
            )
            logger.error(
                "Login failed: Specific 'Invalid Credentials' alert detected."
            )
            return "LOGIN_FAILED_BAD_CREDS"
        except TimeoutException:  # type: ignore
            # Check for any generic alert box
            generic_alert_selector = "div.alert[role='alert']"  # Example
            try:
                alert_element = WebDriverWait(driver, 0.5).until(  # type: ignore
                    EC.presence_of_element_located(  # type: ignore
                        (By.CSS_SELECTOR, generic_alert_selector)  # type: ignore
                    )
                )
                alert_text = (
                    alert_element.text
                    if alert_element and alert_element.text
                    else "Unknown error"
                )
                logger.error(f"Login failed: Generic alert found: '{alert_text}'.")
                return "LOGIN_FAILED_ERROR_DISPLAYED"
            except TimeoutException:  # type: ignore
                logger.error(
                    "Login failed: Credential entry failed, but no specific or generic alert found."
                )
                return "LOGIN_FAILED_CREDS_ENTRY"  # Credential entry itself failed
            except (
                WebDriverException
            ) as alert_err:  # Handle errors checking for alerts # type: ignore
                logger.warning(
                    f"Error checking for generic login error message: {alert_err}"
                )
                return "LOGIN_FAILED_CREDS_ENTRY"  # Assume cred entry failed
        except (
            WebDriverException
        ) as alert_err:  # Handle errors checking for specific alert # type: ignore
            logger.warning(
                f"Error checking for specific login error message: {alert_err}"
            )
            return "LOGIN_FAILED_CREDS_ENTRY"

    return "CREDENTIALS_SUCCESS"


def _check_for_2fa(driver: WebDriver, session_manager: SessionManager) -> Union[bool, str]:  # type: ignore
    """Check if 2FA is present and handle WebDriver errors."""
    logger.debug("Credentials submitted. Waiting for potential page change...")
    time.sleep(random.uniform(3.0, 5.0))  # Allow time for redirect or 2FA page load

    try:
        # Check if the 2FA header is now visible
        WebDriverWait(driver, 5).until(  # type: ignore
            EC.visibility_of_element_located(  # type: ignore
                (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)  # type: ignore
            )
        )
        logger.info("Two-step verification page detected.")
        return True
    except TimeoutException:  # type: ignore
        logger.debug("Two-step verification page not detected.")
        return False
    except WebDriverException as e:  # type: ignore
        logger.error(f"WebDriver error checking for 2FA page: {e}")
        # If error checking, verify login status as fallback
        status = login_status(
            session_manager, disable_ui_fallback=True
        )  # API check only for speed
        if status is True:
            return "LOGIN_SUCCEEDED"
        elif status is False:
            return "LOGIN_FAILED_UNKNOWN"  # Error + not logged in
        else:
            return "LOGIN_FAILED_STATUS_CHECK_ERROR"  # Critical status check error


def _handle_2fa_verification(session_manager: SessionManager) -> str:
    """Handle 2FA verification process."""
    if handle_twoFA(session_manager):
        logger.info("Two-step verification handled successfully.")
        # Re-verify login status after 2FA
        if login_status(session_manager) is True:
            print("\n✓ Two-factor authentication completed successfully!")
            return "LOGIN_SUCCEEDED"
        else:
            logger.error(
                "Login status check failed AFTER successful 2FA handling report."
            )
            return "LOGIN_FAILED_POST_2FA_VERIFY"
    else:
        logger.error("Two-step verification handling failed.")
        return "LOGIN_FAILED_2FA_HANDLING"


def _handle_no_2fa_verification(driver: WebDriver, session_manager: SessionManager) -> str:  # type: ignore
    """Handle login verification when no 2FA is detected."""
    signin_url = urljoin(config_schema.api.base_url, "account/signin")

    # No 2FA detected, check login status directly
    logger.debug("Checking login status directly (no 2FA detected)...")
    login_check_result = login_status(
        session_manager, disable_ui_fallback=False
    )  # Use UI fallback for reliability

    if login_check_result is True:
        print("\n✓ Login successful!")
        return "LOGIN_SUCCEEDED"
    elif login_check_result is False:
        # Verify why it failed if no 2FA was shown
        print("\n✗ Login failed. Please check your credentials.")
        logger.error(
            "Direct login check failed. Checking for error messages again..."
        )
        try:
            # Check specific error again
            WebDriverWait(driver, 1).until(  # type: ignore
                EC.presence_of_element_located(  # type: ignore
                    (By.CSS_SELECTOR, FAILED_LOGIN_SELECTOR)  # type: ignore
                )
            )
            logger.error(
                "Login failed: Specific 'Invalid Credentials' alert found (post-check)."
            )
            return "LOGIN_FAILED_BAD_CREDS"
        except TimeoutException:  # type: ignore
            # Check generic error again
            generic_alert_selector = "div.alert[role='alert']"
            try:
                alert_element = WebDriverWait(driver, 0.5).until(  # type: ignore
                    EC.presence_of_element_located(  # type: ignore
                        (By.CSS_SELECTOR, generic_alert_selector)  # type: ignore
                    )
                )
                alert_text = (
                    alert_element.text if alert_element else "Unknown error"
                )
                logger.error(
                    f"Login failed: Generic alert found (post-check): '{alert_text}'."
                )
                return "LOGIN_FAILED_ERROR_DISPLAYED"
            except TimeoutException:  # type: ignore
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
                except WebDriverException:  # type: ignore
                    logger.error(
                        "Login failed: Status False, WebDriverException getting URL."
                    )
                    return "LOGIN_FAILED_WEBDRIVER"  # Session likely dead
            except (
                WebDriverException
            ) as alert_err:  # Error checking generic alert # type: ignore
                logger.error(
                    f"Login failed: Error checking for generic alert (post-check): {alert_err}"
                )
                return "LOGIN_FAILED_UNKNOWN"
        except (
            WebDriverException
        ) as alert_err:  # Error checking specific alert # type: ignore
            logger.error(
                f"Login failed: Error checking for specific alert (post-check): {alert_err}"
            )
            return "LOGIN_FAILED_UNKNOWN"
    else:  # login_status returned None
        logger.error(
            "Login failed: Critical error during final login status check."
        )
        return "LOGIN_FAILED_STATUS_CHECK_ERROR"


def _handle_2fa_and_verification(driver: WebDriver, session_manager: SessionManager) -> str:  # type: ignore
    """Handle 2FA detection and verification process."""
    two_fa_check = _check_for_2fa(driver, session_manager)

    # If 2FA check returned a login result, return it
    if isinstance(two_fa_check, str):
        return two_fa_check

    # Handle 2FA or direct verification based on detection
    if two_fa_check:  # 2FA present
        return _handle_2fa_verification(session_manager)
    else:  # No 2FA detected
        return _handle_no_2fa_verification(driver, session_manager)


def log_in(session_manager: SessionManager) -> str:  # type: ignore
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

    # First check if already logged in before attempting navigation
    initial_check = _check_initial_login_status(session_manager)
    if initial_check:
        return initial_check

    try:
        # --- Step 1: Navigate to Sign-in Page ---
        nav_result = _navigate_to_signin_page(driver, session_manager)
        if nav_result != "NAVIGATION_SUCCESS":
            return nav_result

        # --- Step 2: Handle Consent Banner ---
        if not consent(driver):
            logger.warning("Failed to handle consent banner, login might be impacted.")
            # Continue anyway, maybe it wasn't essential

        # --- Step 3: Enter Credentials ---
        cred_result = _handle_credential_entry(driver)
        if cred_result != "CREDENTIALS_SUCCESS":
            return cred_result

        # --- Step 4: Handle 2FA and Final Verification ---
        return _handle_2fa_and_verification(driver, session_manager)

    # --- Catch errors during the overall login process ---
    except TimeoutException as e:  # type: ignore
        logger.error(f"Timeout during login process: {e}", exc_info=False)
        return "LOGIN_FAILED_TIMEOUT"
    except WebDriverException as e:  # type: ignore
        logger.error(f"WebDriverException during login: {e}", exc_info=False)
        if not is_browser_open(driver):
            logger.error("Session became invalid during login.")
        # End of if
        return "LOGIN_FAILED_WEBDRIVER"
    except Exception as e:
        logger.error(f"An unexpected error occurred during login: {e}", exc_info=True)
        return "LOGIN_FAILED_UNEXPECTED"
    # End of try/except login process

# End of log_in

# Login Status Check Function
@retry(MAX_RETRIES=2)  # Add retry in case of transient API issues during check
def login_status(session_manager: SessionManager, disable_ui_fallback: bool = False) -> Optional[bool]:  # type: ignore
    """
    Checks if the user is currently logged in. Prioritizes API check, with optional UI fallback.

    Args:
        session_manager: The session manager instance
        disable_ui_fallback: If True, only use API check and never fall back to UI check

    Returns:
        True if logged in, False if not logged in, None if the check fails critically.
    """
    # --- Validate arguments and session state ---
    # Check if session_manager has the expected attributes instead of isinstance check
    if not hasattr(session_manager, 'is_sess_valid') or not hasattr(session_manager, 'driver'):
        logger.error(
            f"Invalid argument: Expected SessionManager-like object, got {type(session_manager)}."
        )
        return None  # Critical argument error
    # End of if

    if not session_manager.is_sess_valid():
        logger.debug("Session is invalid, user cannot be logged in.")
        return False  # Cannot be logged in if session is invalid
    # End of if

    driver = session_manager.driver
    if driver is None:
        logger.error("Login status check: Driver is None within SessionManager.")
        return None  # Critical state error
    # End of if

    # --- Primary Check: API Verification ---
    logger.debug("Performing primary API-based login status check...")
    try:
        # Sync cookies before API check to ensure latest state
        session_manager._sync_cookies()

        # Perform API check
        api_check_result = session_manager.api_manager.verify_api_login_status()

        # If API check is definitive, return its result
        if api_check_result is True:
            logger.debug("API login check confirmed user is logged in.")
            return True
        elif api_check_result is False:
            logger.debug("API login check confirmed user is NOT logged in.")
            return False
        # End of if/elif

        logger.warning("API login check returned ambiguous result (None).")
    except Exception as e:
        logger.error(f"Exception during API login check: {e}", exc_info=True)
        api_check_result = None  # Ensure we continue to UI fallback
    # End of try/except

    # If API check is ambiguous (None) and UI fallback is disabled, return None
    if api_check_result is None and disable_ui_fallback:
        logger.warning(
            "API login check was ambiguous and UI fallback is disabled. Status unknown."
        )
        return None
    # End of if

    # --- Secondary Check: UI Verification (Fallback) ---
    logger.debug("Performing fallback UI-based login status check...")
    try:
        # Check 1: Presence of a known logged-in element (most reliable indicator)
        logged_in_selector = CONFIRMED_LOGGED_IN_SELECTOR  # type: ignore # Assumes defined in my_selectors
        logger.debug(f"Checking for logged-in indicator: '{logged_in_selector}'")

        # Use helper function is_elem_there for robust check
        ui_element_present = is_elem_there(driver, By.CSS_SELECTOR, logged_in_selector, wait=3)  # type: ignore

        if ui_element_present:
            logger.debug("UI check: Logged-in indicator found. User is logged in.")
            return True
        # End of if

        # Check 2: Presence of login button (if present, definitely not logged in)
        login_button_selector = LOG_IN_BUTTON_SELECTOR  # type: ignore # Assumes defined in my_selectors
        logger.debug(f"Checking for login button: '{login_button_selector}'")

        # Use helper function is_elem_there for robust check
        login_button_present = is_elem_there(driver, By.CSS_SELECTOR, login_button_selector, wait=3)  # type: ignore

        if login_button_present:
            logger.debug("UI check: Login button found. User is NOT logged in.")
            return False
        # End of if

        # Check 3: Navigate to base URL and check again if both checks were inconclusive
        if not ui_element_present and not login_button_present:
            logger.debug(
                "UI check inconclusive. Navigating to base URL for clearer check..."
            )
            try:
                current_url = driver.current_url
                base_url = config_schema.api.base_url

                # Only navigate if not already on base URL
                if not current_url.startswith(base_url):
                    driver.get(base_url)
                    time.sleep(2)  # Allow page to load

                    # Check again after navigation
                    ui_element_present = is_elem_there(driver, By.CSS_SELECTOR, logged_in_selector, wait=3)  # type: ignore
                    if ui_element_present:
                        logger.debug(
                            "UI check after navigation: Logged-in indicator found. User is logged in."
                        )
                        return True
                    # End of if

                    login_button_present = is_elem_there(driver, By.CSS_SELECTOR, login_button_selector, wait=3)  # type: ignore
                    if login_button_present:
                        logger.debug(
                            "UI check after navigation: Login button found. User is NOT logged in."
                        )
                        return False
                    # End of if
                # End of if
            except Exception as nav_e:
                logger.warning(
                    f"Error during navigation for secondary UI check: {nav_e}"
                )
                # Continue to default return
            # End of try/except
        # End of if

        # Default to False in ambiguous cases for security reasons
        logger.debug(
            "UI check still inconclusive after all checks. Defaulting to NOT logged in for security."
        )
        return False

    except WebDriverException as e:  # type: ignore
        logger.error(f"WebDriverException during UI login_status check: {e}")
        if not is_browser_open(driver):
            logger.warning("Browser appears to be closed. Closing session.")
            session_manager.close_sess()  # Close the dead session
        # End of if
        return None  # Return None on critical WebDriver error during check
    except Exception as e:  # Catch other unexpected errors
        logger.error(
            f"Unexpected error during UI login_status check: {e}", exc_info=True
        )
        return None
    # End of try/except UI check

# End of login_status

# ------------------------------------------------------------------------------------
# Navigation Functions (Remains in utils.py)
# ------------------------------------------------------------------------------------
def nav_to_page(
    driver: WebDriver,  # type: ignore
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
    # End of if
    if not url or not isinstance(url, str):
        logger.error(f"Navigation failed: Target URL '{url}' is invalid.")
        return False
    # End of if

    max_attempts = config_schema.api.max_retries
    page_timeout = config_schema.selenium.page_load_timeout
    element_timeout = config_schema.selenium.explicit_wait

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
    # End of try/except

    # Define common problematic URLs/selectors
    signin_page_url_base = urljoin(config_schema.api.base_url, "account/signin").rstrip("/")
    mfa_page_url_base = urljoin(config_schema.api.base_url, "account/signin/mfa/").rstrip("/")
    # Selectors for known 'unavailable' pages
    unavailability_selectors = {
        TEMP_UNAVAILABLE_SELECTOR: ("refresh", 5),  # type: ignore # Selector : (action, wait_seconds)
        PAGE_NO_LONGER_AVAILABLE_SELECTOR: ("skip", 0),  # type: ignore
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
                        # Get the new driver instance with type assertion
                        driver_instance = session_manager.driver
                        if driver_instance is not None:
                            driver = driver_instance  # Only assign if not None
                        if not driver:  # Check if restart actually provided a driver
                            logger.error(
                                "Session restart reported success but driver is still None."
                            )
                            return False
                        # End of if
                        continue  # Retry navigation with new driver
                    else:
                        logger.error("Session restart failed. Cannot navigate.")
                        return False  # Unrecoverable
                    # End of if/else restart
                else:
                    logger.error(
                        "Session invalid and no SessionManager provided for restart."
                    )
                    return False  # Unrecoverable
                # End of if/else session_manager
            # End of if not is_browser_open

            # --- Navigation Execution ---
            logger.debug(f"Executing driver.get('{url}')...")
            driver.get(url)

            # Wait for document ready state (basic page load signal)
            WebDriverWait(driver, page_timeout).until(  # type: ignore
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
            except WebDriverException as e:  # type: ignore
                logger.error(
                    f"Failed to get current URL after get() (Attempt {attempt}): {e}. Retrying."
                )
                continue  # Retry the navigation attempt
            # End of try/except

            # Check for MFA page
            is_on_mfa_page = False
            try:
                # Use short wait, presence is enough
                WebDriverWait(driver, 1).until(  # type: ignore
                    EC.visibility_of_element_located(  # type: ignore
                        (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)  # type: ignore
                    )
                )
                is_on_mfa_page = True
            except (TimeoutException, NoSuchElementException):  # type: ignore
                pass  # Expected if not on MFA page
            except WebDriverException as e:  # type: ignore
                logger.warning(f"WebDriverException checking for MFA header: {e}")
            # End of try/except

            if is_on_mfa_page:
                logger.error(
                    "Landed on MFA page unexpectedly during navigation. Navigation failed."
                )
                # Should not attempt re-login here, indicates a prior login state issue
                return False  # Fail navigation
            # End of if

            # Check for Login page (only if *not* intentionally navigating there)
            is_on_login_page = False
            if (
                target_url_base != signin_page_url_base
            ):  # Don't check if login is the target
                try:
                    # Check if username input exists (strong indicator of login page)
                    WebDriverWait(driver, 1).until(  # type: ignore
                        EC.visibility_of_element_located(  # type: ignore
                            (By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR)  # type: ignore
                        )
                    )
                    is_on_login_page = True
                except (TimeoutException, NoSuchElementException):  # type: ignore
                    pass  # Expected if not on login page
                except WebDriverException as e:  # type: ignore
                    logger.warning(
                        f"WebDriverException checking for Login username input: {e}"
                    )
                # End of try/except
            # End of if target_url_base != signin_page_url_base

            if is_on_login_page:
                logger.warning(
                    "Landed on Login page unexpectedly. Checking login status first..."
                )
                if session_manager:
                    # First check if we're already logged in (API might say yes even if UI shows login page)
                    login_stat = login_status(
                        session_manager, disable_ui_fallback=True
                    )  # API check only for speed
                    if login_stat is True:
                        logger.info(
                            "Login status OK after landing on login page redirect. Retrying original navigation."
                        )
                        continue  # Retry original nav_to_page call
                    else:
                        # Attempt automated login
                        logger.info(
                            "Not logged in according to API. Attempting re-login..."
                        )
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
                        # End of if/else login_result_str
                    # End of if/else login_stat
                else:
                    logger.error(
                        "Landed on login page, no SessionManager provided for re-login attempt."
                    )
                    return False  # Fail navigation
                # End of if/else session_manager
            # End of if is_on_login_page

            # Check if landed on an unexpected URL (and not login/mfa)
            # Allow for slight variations (e.g., trailing slash) via base comparison
            if landed_url_base != target_url_base:
                # Check if it's a known redirect (e.g., signin page redirecting to base URL after successful login)
                is_signin_to_base_redirect = (
                    target_url_base == signin_page_url_base
                    and landed_url_base
                    == urlparse(config_schema.api.base_url).path.rstrip("/")
                )
                if is_signin_to_base_redirect:
                    logger.debug(
                        "Redirected from signin page to base URL. Verifying login status..."
                    )
                    time.sleep(1)  # Allow settling
                    if (
                        session_manager
                        and login_status(session_manager, disable_ui_fallback=True)
                        is True
                    ):  # API check only for speed
                        logger.info(
                            "Redirect after signin confirmed as logged in. Considering original navigation target 'signin' successful."
                        )
                        return True  # Treat as success if login was the goal and we are now logged in
                    # End of if
                # End of if is_signin_to_base_redirect

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
                # End of if/elif/else action
            # End of if landed_url_base != target_url_base

            # --- Final Check: Element on Page ---
            # If we reached here, we are on the correct URL base (or handled redirects)
            wait_selector = (
                selector if selector else "body"
            )  # Default to body if no selector provided
            logger.debug(
                f"On correct URL base. Waiting up to {element_timeout}s for selector: '{wait_selector}'"
            )
            try:
                WebDriverWait(driver, element_timeout).until(  # type: ignore
                    EC.visibility_of_element_located((By.CSS_SELECTOR, wait_selector))  # type: ignore
                )
                logger.debug(
                    f"Navigation successful and element '{wait_selector}' found on: {url}"
                )
                return True  # Success!

            except TimeoutException:  # type: ignore
                # Correct URL, but target element didn't appear
                current_url_on_timeout = "Unknown"
                try:
                    current_url_on_timeout = driver.current_url
                except Exception:
                    pass
                # End of try/except
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
                # End of if/elif

                logger.warning(
                    "Timeout on selector, no unavailability message. Retrying navigation."
                )
                continue  # Retry navigation attempt

            except (
                WebDriverException
            ) as el_wait_err:  # Catch errors during element wait # type: ignore
                logger.error(
                    f"WebDriverException waiting for selector '{wait_selector}': {el_wait_err}"
                )
                continue  # Retry navigation
            # End of try/except for final check

        # --- Handle Exceptions During Navigation Attempt ---
        except UnexpectedAlertPresentException as alert_e:  # type: ignore
            alert_text = "N/A"
            try:
                alert_text = alert_e.alert_text  # type: ignore
            except AttributeError:
                pass
            # End of try/except
            logger.warning(
                f"Unexpected alert detected (Attempt {attempt}): {alert_text}"
            )
            try:
                driver.switch_to.alert.accept()
                logger.info("Accepted unexpected alert.")
            except Exception as accept_e:
                logger.error(f"Failed to accept unexpected alert: {accept_e}")
                return False  # Fail if alert cannot be handled
            # End of try/except
            continue  # Retry navigation after handling alert

        except WebDriverException as wd_e:  # type: ignore
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
                    # Get the new driver instance with type assertion
                    driver_instance = session_manager.driver
                    if driver_instance is not None:
                        driver = driver_instance  # Only assign if not None
                    if not driver:
                        return False  # Fail if restart didn't provide driver
                    # End of if
                    continue  # Retry navigation
                else:
                    logger.error("Session restart failed. Cannot complete navigation.")
                    return False  # Unrecoverable
                # End of if/else restart
            else:
                logger.warning(
                    "WebDriverException occurred, session seems valid or no restart possible. Waiting before retry."
                )
                time.sleep(random.uniform(2, 4))
                continue  # Retry navigation attempt
            # End of if/else session_manager
        except Exception as e:  # Catch other unexpected errors
            logger.error(
                f"Unexpected error during navigation (Attempt {attempt}): {e}",
                exc_info=True,
            )
            time.sleep(random.uniform(2, 4))  # Wait before retry
            continue  # Retry navigation attempt
        # End of try/except block for navigation attempt

    # --- Failed After All Attempts ---
    logger.critical(
        f"Navigation to '{url}' failed permanently after {max_attempts} attempts."
    )
    try:
        logger.error(f"Final URL after failure: {driver.current_url}")
    except Exception:
        logger.error("Could not retrieve final URL after failure.")
    # End of try/except
    return False

# End of nav_to_page

def _check_for_unavailability(
    driver: WebDriver, selectors: Dict[str, Tuple[str, int]]  # type: ignore
) -> Tuple[Optional[str], int]:
    """Checks if known 'page unavailable' messages are present using provided selectors."""
    # Check if driver is usable
    if not is_browser_open(driver):
        logger.warning("Cannot check for unavailability: driver session invalid.")
        return None, 0
    # End of if

    for msg_selector, (action, wait_time) in selectors.items():
        # Use selenium_utils helper 'is_elem_there' with a very short wait
        # Assume is_elem_there is imported
        if is_elem_there(driver, By.CSS_SELECTOR, msg_selector, wait=1):  # type: ignore
            logger.warning(
                f"Unavailability message found matching selector: '{msg_selector}'. Action: {action}, Wait: {wait_time}s"
            )
            return action, wait_time  # Return action (refresh/skip) and wait time
        # End of if
    # End of for

    # Return default (no action, zero wait) if no matching selectors found
    return None, 0

# End of _check_for_unavailability

def main() -> None:
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
    # Imports are assumed successful due to strict checks at top level
    from logging_config import setup_logging
    from config.config_manager import ConfigManager

    pass  # main function placeholder, test logic removed

def test_parse_cookie():
    """Test cookie parsing with various cookie string formats"""
    try:
        test_cases = [
            (
                "session_id=abc123; path=/; domain=.example.com",
                {"session_id": "abc123", "path": "/", "domain": ".example.com"},
                "Standard cookie format",
            ),
            ("", {}, "Empty cookie string"),
            ("single=value", {"single": "value"}, "Single cookie"),
            (
                "a=1; b=2; c=3",
                {"a": "1", "b": "2", "c": "3"},
                "Multiple cookies",
            ),
            (
                "invalid_part; valid=test",
                {"valid": "test"},
                "Mixed valid/invalid parts",
            ),
        ]

        for cookie_str, expected, description in test_cases:
            result = parse_cookie(cookie_str)
            if result != expected:
                return False
        return True
    except Exception:
        return False


def test_ordinal_case():
    """Test ordinal number formatting with various input types"""
    try:
        test_cases = [
            (1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"),
            (11, "11th"), (12, "12th"), (13, "13th"),
            (21, "21st"), (22, "22nd"), (23, "23rd"),
            (101, "101st"), ("Great Uncle", "Great Uncle"),
        ]

        for input_val, expected in test_cases:
            result = ordinal_case(input_val)
            if result != expected:
                return False
        return True
    except Exception:
        return False


def test_format_name():
    """Test name formatting with various input types and edge cases"""
    try:
        test_cases = [
            ("john doe", "John Doe"),
            (None, "Valued Relative"),
            ("", "Valued Relative"),
            ("john /doe/", "John Doe"),
            ("o'malley", "O'Malley"),
            ("mcdonald", "McDonald"),
            ("macleod", "MacLeod"),
            ("'Betty'", "Betty"),
            ("mary-jane smith-jones", "Mary-Jane Smith-Jones"),
            ("j. r. r. tolkien", "J. R. R. Tolkien"),
        ]

        for input_val, expected in test_cases:
            result = format_name(input_val)
            if result != expected:
                return False
        return True
    except Exception:
        return False

def test_decorators():
    """Test decorator availability and functionality"""
    try:
        # Test retry decorator availability
        assert callable(retry), "retry decorator should be callable"
        assert callable(retry_api), "retry_api decorator should be callable"
        assert callable(ensure_browser_open), "ensure_browser_open decorator should be callable"
        assert callable(time_wait), "time_wait decorator should be callable"

        # Basic decorator functionality test
        @retry(MAX_RETRIES=1, BACKOFF_FACTOR=0.001)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success", "Retry decorator should work"
        return True
    except Exception:
        return False


def test_rate_limiter():
    """Test DynamicRateLimiter instantiation and basic functionality"""
    try:
        limiter = DynamicRateLimiter(initial_delay=0.001, max_delay=0.01)
        assert limiter is not None, "Rate limiter should instantiate"
        assert hasattr(limiter, "wait"), "Rate limiter should have wait method"
        assert hasattr(limiter, "adjust_delay"), "Rate limiter should have adjust_delay method"
        assert hasattr(limiter, "get_stats"), "Rate limiter should have get_stats method"

        # Test basic wait functionality
        import time
        start_time = time.time()
        limiter.wait()
        elapsed = time.time() - start_time
        assert elapsed < 1.0, "Wait should complete quickly in test"
        return True
    except Exception:
        return False


def test_session_manager():
    """Test SessionManager class availability and basic attributes"""
    try:
        # Import SessionManager directly to avoid circular import issues
        from core.session_manager import SessionManager

        sm = SessionManager()
        assert sm is not None, "SessionManager should instantiate"
        assert hasattr(sm, "driver_live"), "SessionManager should have driver_live attribute"
        assert hasattr(sm, "session_ready"), "SessionManager should have session_ready attribute"
        assert hasattr(sm, "browser_manager"), "SessionManager should have browser_manager attribute"
        assert hasattr(sm, "db_manager"), "SessionManager should have db_manager attribute"

        # Test initial state
        assert not sm.driver_live, "Driver should not be live initially"
        assert not sm.session_ready, "Session should not be ready initially"
        return True
    except Exception:
        return False


def test_api_request_function():
    """Test _api_req function availability"""
    try:
        assert callable(_api_req), "_api_req function should be callable"

        # Test function signature (should not raise errors)
        import inspect as inspect_module
        sig = inspect_module.signature(_api_req)
        assert len(sig.parameters) >= 2, "_api_req should accept multiple parameters"
        return True
    except Exception:
        return False


def test_login_status_function():
    """Test login_status function availability"""
    try:
        assert callable(login_status), "login_status function should be callable"

        # Test function signature
        import inspect as inspect_module
        sig = inspect_module.signature(login_status)
        assert "session_manager" in sig.parameters, "login_status should accept session_manager parameter"
        return True
    except Exception:
        return False


def test_module_registration():
    """Test module registration functions"""
    try:
        assert callable(auto_register_module), "auto_register_module should be available"
        assert callable(register_function), "register_function should be available"
        assert callable(get_function), "get_function should be available"

        # Test that core classes are available
        assert "format_name" in globals(), "format_name should be in globals"
        assert "DynamicRateLimiter" in globals(), "DynamicRateLimiter should be in globals"
        assert "SessionManager" in globals(), "SessionManager should be in globals"
        return True
    except Exception:
        return False


def test_performance_validation():
    """Test performance of key operations"""
    try:
        import time
        start_time = time.time()

        # Format name performance
        for i in range(100):
            format_name(f"test name {i}")

        # Ordinal case performance
        for i in range(1, 101):
            ordinal_case(i)

        elapsed = time.time() - start_time
        assert elapsed < 1.0, f"Performance test should complete quickly, took {elapsed:.3f}s"
        return True
    except Exception:
        return False


# Removed duplicate function definition - the real one is in the test section

# ==============================================
# Module Registration
# ==============================================

# Module setup already handled by setup_module() call at top of file

# ==============================================
# Standalone Test Block
# ==============================================
def utils_module_tests() -> bool:
    """
    Comprehensive test suite for utils.py with real functionality testing.
    Tests core utility functions, decorators, rate limiting, session management, and performance.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite(
        "Core Utilities & Session Management", "utils.py"
    )
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Cookie parsing functionality",
            test_parse_cookie,
            "5 cookie formats tested: standard, empty, single, multiple, mixed valid/invalid parts.",
            "Test cookie parsing with various cookie string formats.",
            "Test parse_cookie with: 'session_id=abc123; path=/', '', 'single=value', 'a=1; b=2; c=3', 'invalid_part; valid=test'.",
        )

        suite.run_test(
            "Ordinal number formatting",
            test_ordinal_case,
            "12 ordinal tests: 1st, 2nd, 3rd, 4th, 11th-13th (special), 21st-23rd, 101st, text input.",
            "Test ordinal number formatting with various input types.",
            "Test ordinal_case with: 1→'1st', 2→'2nd', 3→'3rd', 11→'11th', 21→'21st', 'Great Uncle'→'Great Uncle'.",
        )

        suite.run_test(
            "Name formatting functionality",
            test_format_name,
            "11 name formats: basic, None→'Valued Relative', GEDCOM /slashes/, O'Malley, McDonald, MacLeod, 'Betty', hyphenated, initials.",
            "Test name formatting with various input types and edge cases.",
            "Test format_name with: 'john doe'→'John Doe', None→'Valued Relative', 'john /doe/'→'John Doe', 'o'malley'→'O'Malley'.",
        )

        suite.run_test(
            "Decorator availability and functionality",
            test_decorators,
            "Test availability and basic functionality of retry, API, and timing decorators",
            "Decorators provide robust function enhancement capabilities",
            "All utility decorators are available and function correctly",
        )

        suite.run_test(
            "Dynamic rate limiting",
            test_rate_limiter,
            "Test DynamicRateLimiter instantiation and basic rate limiting functionality",
            "Rate limiting manages API request timing and prevents throttling",
            "Dynamic rate limiter provides effective request flow control",
        )

        suite.run_test(
            "Session management",
            test_session_manager,
            "Test SessionManager class instantiation and basic session management features",
            "Session management provides browser automation and session handling",
            "SessionManager class provides complete session lifecycle management",
        )

        suite.run_test(
            "API request functionality",
            test_api_request_function,
            "Test _api_req function availability and signature validation",
            "API request function provides core HTTP request capabilities",
            "Core API request functionality is available and properly configured",
        )

        suite.run_test(
            "Login status checking",
            test_login_status_function,
            "Test login_status function availability and parameter validation",
            "Login status checking provides authentication state verification",
            "Login status functionality is available for session validation",
        )

        suite.run_test(
            "Module registration system",
            test_module_registration,
            "Test module registration functions and verify core functions are registered",
            "Module registration provides optimized function access",
            "All core utility functions are properly registered and accessible",
        )

        suite.run_test(
            "Performance validation",
            test_performance_validation,
            "Test performance of name formatting and ordinal operations with datasets",
            "Performance validation ensures efficient utility function execution",
            "Utility functions complete processing within reasonable time limits",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive utils tests using standardized TestSuite format."""
    return utils_module_tests()


if __name__ == "__main__":
    pass  # Standalone test logic removed


# === CONTEXT MANAGERS FOR RESOURCE MANAGEMENT ===

@contextlib.contextmanager
def safe_file_operation(file_path: Union[str, Path], mode: str = 'r', encoding: str = 'utf-8') -> Generator[IO[Any], None, None]:
    """
    Context manager for safe file operations with automatic cleanup.

    Provides robust file handling with automatic resource cleanup, error handling,
    and encoding management. Ensures files are properly closed even if exceptions occur.

    Args:
        file_path: Path to the file to open.
        mode: File open mode (e.g., 'r', 'w', 'a').
        encoding: File encoding (default: 'utf-8').

    Yields:
        TextIO: File object for reading/writing operations.

    Example:
        >>> with safe_file_operation('data.txt', 'w') as f:
        ...     f.write('Hello, World!')
        >>> with safe_file_operation('data.txt', 'r') as f:
        ...     content = f.read()
    """
    file_path = Path(file_path)
    file_handle = None

    try:
        file_handle = open(file_path, mode, encoding=encoding)
        yield file_handle
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Cannot access file: {file_path}") from e
    except PermissionError as e:
        logger.error(f"Permission denied accessing file: {file_path}")
        raise PermissionError(f"Access denied to file: {file_path}") from e
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading file {file_path}: {e}")
        raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end,
                               f"Encoding error in file {file_path}: {e.reason}") from e
    except Exception as e:
        logger.error(f"Unexpected error with file {file_path}: {e}")
        raise RuntimeError(f"Unexpected error accessing file {file_path}") from e
    finally:
        if file_handle and not file_handle.closed:
            try:
                file_handle.close()
                logger.debug(f"File closed successfully: {file_path}")
            except Exception as close_error:
                logger.warning(f"Error closing file {file_path}: {close_error}")


@contextlib.contextmanager
def api_session_context(session_manager: Optional['SessionManager'] = None) -> Generator[requests.Session, None, None]:
    """
    Context manager for API session management with automatic cleanup.

    Provides a managed requests session with proper cookie handling, timeout
    configuration, and automatic cleanup. Integrates with SessionManager for
    authentication state management.

    Args:
        session_manager: Optional SessionManager for authentication integration.

    Yields:
        requests.Session: Configured session for API requests.

    Example:
        >>> with api_session_context(session_manager) as session:
        ...     response = session.get('https://api.ancestry.com/endpoint')
    """
    session = None

    try:
        if session_manager and hasattr(session_manager, '_requests_session') and session_manager._requests_session:
            # Use existing session from SessionManager
            session = session_manager._requests_session
            logger.debug("Using existing session from SessionManager")
        else:
            # Create new session
            session = requests.Session()
            # Note: requests.Session doesn't have a timeout attribute
            # Timeout should be passed to individual request methods
            logger.debug("Created new API session")

        # Configure session headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
        })

        yield session

    except Exception as e:
        logger.error(f"Error in API session context: {e}")
        raise
    finally:
        # Only close session if we created it (not from SessionManager)
        if session and (not session_manager or not hasattr(session_manager, '_requests_session') or session != session_manager._requests_session):
            try:
                session.close()
                logger.debug("API session closed successfully")
            except Exception as close_error:
                logger.warning(f"Error closing API session: {close_error}")


# === ASYNC API OPERATIONS (Phase 7.4.1) ===

@contextlib.asynccontextmanager
async def async_api_session_context(session_manager: Optional['SessionManager'] = None):
    """
    Async context manager for HTTP session management with aiohttp.

    Provides a managed aiohttp session with proper cookie handling, timeout
    configuration, and automatic cleanup. Integrates with SessionManager for
    authentication state management.

    Args:
        session_manager: Optional SessionManager for authentication integration.

    Yields:
        aiohttp.ClientSession: Configured async session for API requests.

    Example:
        >>> async with async_api_session_context(session_manager) as session:
        ...     async with session.get('https://api.ancestry.com/endpoint') as response:
        ...         data = await response.json()
    """
    if not AIOHTTP_AVAILABLE or aiohttp is None:
        raise ImportError("aiohttp is required for async API operations. Install with: pip install aiohttp")
    timeout = aiohttp.ClientTimeout(total=30)  # Default timeout
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)  # Connection pooling

    # Configure headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Extract cookies from session manager if available
    cookies = None
    if session_manager and hasattr(session_manager, '_requests_session') and session_manager._requests_session:
        try:
            # Convert requests cookies to aiohttp format
            cookies = {}
            for cookie in session_manager._requests_session.cookies:
                cookies[cookie.name] = cookie.value
            logger.debug(f"Transferred {len(cookies)} cookies from SessionManager")
        except Exception as e:
            logger.warning(f"Failed to transfer cookies from SessionManager: {e}")

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers=headers,
        cookies=cookies
    ) as session:
        logger.debug("Created async API session")
        try:
            yield session
        finally:
            logger.debug("Async API session closed")


async def async_api_request(
    url: str,
    method: str = "GET",
    session_manager: Optional['SessionManager'] = None,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    timeout: Optional[int] = None,
    api_description: str = "Async API Call",
    max_retries: int = 3,
    backoff_factor: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Make an async HTTP request with retry logic and error handling.

    Args:
        url: The URL to request
        method: HTTP method (GET, POST, etc.)
        session_manager: Optional SessionManager for authentication
        headers: Optional additional headers
        data: Optional form data
        json_data: Optional JSON data
        timeout: Optional timeout in seconds
        api_description: Description for logging
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor for retries

    Returns:
        Optional[Dict]: Response data as dictionary, or None on failure

    Example:
        >>> data = await async_api_request(
        ...     "https://api.ancestry.com/suggest",
        ...     method="POST",
        ...     json_data={"query": "John Smith"},
        ...     session_manager=session_manager
        ... )
    """
    if not AIOHTTP_AVAILABLE:
        logger.warning(f"[{api_description}] aiohttp not available - falling back to synchronous request")
        # Could implement fallback to requests here if needed
        return None
    async with async_api_session_context(session_manager) as session:
        # Merge headers
        request_headers = {}
        if headers:
            request_headers.update(headers)

        # Add CSRF token if available
        if session_manager and hasattr(session_manager, 'csrf_token') and session_manager.csrf_token:
            request_headers['X-CSRF-TOKEN'] = session_manager.csrf_token

        # Set timeout
        request_timeout = aiohttp.ClientTimeout(total=timeout or 30) if aiohttp else None

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"[{api_description}] Async attempt {attempt}/{max_retries}: {method} {url}")

                async with session.request(
                    method=method,
                    url=url,
                    headers=request_headers,
                    data=data,
                    json=json_data,
                    timeout=request_timeout
                ) as response:
                    logger.debug(f"[{api_description}] Response: {response.status} {response.reason}")

                    if response.status == 200:
                        try:
                            result = await response.json()
                            logger.info(f"[{api_description}] Successful async request (attempt {attempt})")
                            return result
                        except (aiohttp.ContentTypeError if aiohttp else Exception):
                            # Try to get text response
                            text_result = await response.text()
                            logger.debug(f"[{api_description}] Non-JSON response: {text_result[:200]}")
                            return {"text": text_result}
                    elif response.status == 429:  # Rate limit
                        logger.warning(f"[{api_description}] Rate limited (429), attempt {attempt}/{max_retries}")
                        if attempt < max_retries:
                            await asyncio.sleep(backoff_factor * (2 ** attempt))
                            continue
                        else:
                            logger.error(f"[{api_description}] Rate limit exceeded after {max_retries} attempts")
                            return None
                    else:
                        logger.warning(f"[{api_description}] HTTP {response.status}: {response.reason}")
                        if attempt < max_retries:
                            await asyncio.sleep(backoff_factor * attempt)
                            continue
                        else:
                            return None

            except asyncio.TimeoutError:
                logger.warning(f"[{api_description}] Timeout on attempt {attempt}/{max_retries}")
                if attempt < max_retries:
                    await asyncio.sleep(backoff_factor * attempt)
                    continue
                else:
                    logger.error(f"[{api_description}] Timeout after {max_retries} attempts")
                    return None
            except Exception as e:
                logger.error(f"[{api_description}] Error on attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(backoff_factor * attempt)
                    continue
                else:
                    logger.error(f"[{api_description}] Failed after {max_retries} attempts")
                    return None

    return None


async def async_batch_api_requests(
    requests: List[Dict[str, Any]],
    session_manager: Optional['SessionManager'] = None,
    max_concurrent: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Optional[Dict[str, Any]]]:
    """
    Execute multiple API requests concurrently with controlled concurrency.

    Args:
        requests: List of request dictionaries with 'url', 'method', etc.
        session_manager: Optional SessionManager for authentication
        max_concurrent: Maximum number of concurrent requests
        progress_callback: Optional callback for progress updates

    Returns:
        List of response dictionaries in the same order as input requests

    Example:
        >>> requests = [
        ...     {"url": "https://api.ancestry.com/person/1", "api_description": "Person 1"},
        ...     {"url": "https://api.ancestry.com/person/2", "api_description": "Person 2"},
        ... ]
        >>> results = await async_batch_api_requests(requests, session_manager)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_request(request_data: Dict[str, Any], index: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        async with semaphore:
            result = await async_api_request(session_manager=session_manager, **request_data)
            if progress_callback:
                progress_callback(index + 1, len(requests))
            return index, result

    # Create tasks for all requests
    tasks = [
        bounded_request(request_data, i)
        for i, request_data in enumerate(requests)
    ]

    # Execute all tasks concurrently
    logger.info(f"Starting {len(requests)} async API requests with max_concurrent={max_concurrent}")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Sort results by original index and extract values
    sorted_results: List[Optional[Dict[str, Any]]] = [None] * len(requests)
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Async batch request failed: {result}")
            continue
        if isinstance(result, tuple) and len(result) == 2:
            index, data = result
            sorted_results[index] = data

    successful_count = sum(1 for r in sorted_results if r is not None)
    logger.info(f"Completed async batch: {successful_count}/{len(requests)} successful")

    return sorted_results


# === ASYNC FILE I/O OPERATIONS (Phase 7.4.2) ===

try:
    import aiofiles  # For async file operations
    AIOFILES_AVAILABLE = True
except ImportError:
    aiofiles = None  # type: ignore
    AIOFILES_AVAILABLE = False
    logger.warning("aiofiles not available - async file operations will use thread pool fallback")

@contextlib.asynccontextmanager
async def async_file_context(
    file_path: Union[str, Path],
    mode: str = "r",
    encoding: str = "utf-8",
    **kwargs
):
    """
    Async context manager for file operations.

    Uses aiofiles if available, otherwise falls back to thread pool execution
    of standard file operations for async compatibility.

    Args:
        file_path: Path to the file
        mode: File open mode
        encoding: File encoding
        **kwargs: Additional arguments for file opening

    Yields:
        File handle for async operations

    Example:
        >>> async with async_file_context("data.json", "r") as f:
        ...     content = await f.read()
    """
    file_path = Path(file_path)

    if AIOFILES_AVAILABLE and aiofiles is not None:
        # Use aiofiles for true async file I/O
        async with aiofiles.open(file_path, mode=mode, encoding=encoding, **kwargs) as f:  # type: ignore
            logger.debug(f"Async file opened with aiofiles: {file_path} (mode: {mode})")
            yield f
    else:
        # Fallback to thread pool execution
        loop = asyncio.get_event_loop()

        def _open_file():
            return open(file_path, mode=mode, encoding=encoding, **kwargs)

        file_handle = await loop.run_in_executor(None, _open_file)

        try:
            logger.debug(f"Async file opened with thread pool: {file_path} (mode: {mode})")

            # Create async wrapper for file operations
            class AsyncFileWrapper:
                def __init__(self, file_handle, loop):
                    self._file = file_handle
                    self._loop = loop

                async def read(self, size=-1):
                    return await self._loop.run_in_executor(None, self._file.read, size)

                async def write(self, data):
                    return await self._loop.run_in_executor(None, self._file.write, data)

                async def readline(self):
                    return await self._loop.run_in_executor(None, self._file.readline)

                async def readlines(self):
                    return await self._loop.run_in_executor(None, self._file.readlines)

                def __getattr__(self, name):
                    # Delegate other attributes to the underlying file
                    return getattr(self._file, name)

            yield AsyncFileWrapper(file_handle, loop)

        finally:
            await loop.run_in_executor(None, file_handle.close)


async def async_read_json_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Read and parse a JSON file asynchronously.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON data as dictionary, or None on failure

    Example:
        >>> data = await async_read_json_file("config.json")
        >>> if data:
        ...     print(f"Loaded config with {len(data)} keys")
    """
    try:
        async with async_file_context(file_path, "r") as f:
            content = await f.read()
            return json.loads(content)
    except FileNotFoundError:
        logger.warning(f"JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return None


async def async_write_json_file(
    file_path: Union[str, Path],
    data: Dict[str, Any],
    indent: int = 2,
    ensure_ascii: bool = False
) -> bool:
    """
    Write data to a JSON file asynchronously.

    Args:
        file_path: Path to the JSON file
        data: Data to write
        indent: JSON indentation
        ensure_ascii: Whether to ensure ASCII encoding

    Returns:
        True if successful, False otherwise

    Example:
        >>> success = await async_write_json_file(
        ...     "output.json", {"key": "value"}
        ... )
    """
    try:
        # Create parent directory if it doesn't exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize JSON
        json_content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

        async with async_file_context(file_path, "w") as f:
            await f.write(json_content)

        logger.debug(f"Successfully wrote JSON file: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        return False


async def async_read_text_file(file_path: Union[str, Path]) -> Optional[str]:
    """
    Read a text file asynchronously.

    Args:
        file_path: Path to the text file

    Returns:
        File content as string, or None on failure
    """
    try:
        async with async_file_context(file_path, "r") as f:
            return await f.read()
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return None


async def async_write_text_file(file_path: Union[str, Path], content: str) -> bool:
    """
    Write content to a text file asynchronously.

    Args:
        file_path: Path to the text file
        content: Content to write

    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with async_file_context(file_path, "w") as f:
            await f.write(content)

        logger.debug(f"Successfully wrote text file: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error writing text file {file_path}: {e}")
        return False


async def async_batch_file_operations(
    operations: List[Dict[str, Any]],
    max_concurrent: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[bool]:
    """
    Execute multiple file operations concurrently.

    Args:
        operations: List of operation dictionaries with 'type', 'path', and operation-specific params
        max_concurrent: Maximum concurrent operations
        progress_callback: Optional progress callback

    Returns:
        List of success/failure results

    Example:
        >>> operations = [
        ...     {"type": "read_json", "path": "file1.json"},
        ...     {"type": "write_json", "path": "file2.json", "data": {"key": "value"}},
        ... ]
        >>> results = await async_batch_file_operations(operations)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_operation(operation: Dict[str, Any], index: int) -> Tuple[int, bool]:
        async with semaphore:
            try:
                op_type = operation["type"]
                path = operation["path"]

                if op_type == "read_json":
                    result = await async_read_json_file(path)
                    success = result is not None
                elif op_type == "write_json":
                    success = await async_write_json_file(path, operation["data"])
                elif op_type == "read_text":
                    result = await async_read_text_file(path)
                    success = result is not None
                elif op_type == "write_text":
                    success = await async_write_text_file(path, operation["content"])
                else:
                    logger.error(f"Unknown operation type: {op_type}")
                    success = False

                if progress_callback:
                    progress_callback(index + 1, len(operations))

                return index, success

            except Exception as e:
                logger.error(f"File operation {index} failed: {e}")
                return index, False

    # Create tasks for all operations
    tasks = [
        bounded_operation(operation, i)
        for i, operation in enumerate(operations)
    ]

    # Execute all tasks concurrently
    logger.info(f"Starting {len(operations)} async file operations with max_concurrent={max_concurrent}")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Sort results by original index and extract success values
    sorted_results = [False] * len(operations)
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Async file operation failed: {result}")
            continue
        if isinstance(result, tuple) and len(result) == 2:
            index, success = result
            sorted_results[index] = success

    successful_count = sum(sorted_results)
    logger.info(f"Completed async file operations: {successful_count}/{len(operations)} successful")

    return sorted_results

# End of utils.py
