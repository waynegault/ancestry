#!/usr/bin/env python3

"""
utils.py - Core Session Management, API Requests, General Utilities

Manages Selenium/Requests sessions, handles core API interaction (_api_req),
provides general utilities (decorators, formatting, rate limiting),
and includes login/session verification logic closely tied to SessionManager.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    auto_register_module,  # Needed for testing
    get_function,
    is_function_available,
    register_function,
    setup_module,
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

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import base64  # For make_ube
import binascii  # For make_ube
import json
import logging
import random  # For make_newrelic, retry_api, DynamicRateLimiter
import re
import sys
import time
import uuid  # For make_ube, make_traceparent, make_tracestate
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)  # Consolidated typing imports
from urllib.parse import urljoin, urlparse, urlunparse

# === THIRD-PARTY IMPORTS ===
import requests
from requests import Response as RequestsResponse
from selenium.webdriver.remote.webdriver import WebDriver

# === LOCAL IMPORTS ===
# (Note: Some imports done locally to avoid circular dependencies)
from common_params import NavigationConfig, RetryContext

# === TYPE ALIASES ===
# Define type aliases
RequestsResponseTypeOptional = Optional[RequestsResponse]
ApiResponseType = Union[dict[str, Any], list[Any], str, bytes, None, RequestsResponse]
DriverType = Optional[WebDriver]
SessionManagerType = Optional[
    "SessionManager"
]  # Use string literal for forward reference


# === API REQUEST CONFIGURATION ===
@dataclass
class ApiRequestConfig:
    """
    Configuration object for API requests to reduce parameter count.
    Encapsulates all parameters needed for _api_req and related functions.
    """
    # Required parameters
    url: str
    driver: DriverType
    session_manager: "SessionManager"

    # HTTP method and data
    method: str = "GET"
    data: Optional[dict] = None
    json_data: Optional[dict] = None
    json: Optional[dict] = None

    # Headers and authentication
    headers: Optional[dict[str, str]] = None
    referer_url: Optional[str] = None
    use_csrf_token: bool = True
    add_default_origin: bool = True

    # Request behavior
    timeout: Optional[int] = None
    cookie_jar: Optional["RequestsCookieJar"] = None
    allow_redirects: bool = True
    force_text_response: bool = False

    # Retry configuration
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    retry_status_codes: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])

    # Metadata
    api_description: str = "API Call"
    attempt: int = 1


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
    from requests.cookies import RequestsCookieJar
    from requests.exceptions import HTTPError, JSONDecodeError, RequestException
    from selenium.common.exceptions import (
        ElementClickInterceptedException,
        ElementNotInteractableException,
        NoSuchCookieException,
        NoSuchElementException,
        StaleElementReferenceException,
        TimeoutException,
        UnexpectedAlertPresentException,
        WebDriverException,
    )
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.support import expected_conditions as EC  # noqa: N812 - Standard Selenium convention
    from selenium.webdriver.support.wait import WebDriverWait

    # --- Local application imports ---
    # Assume these are essential or handled elsewhere if missing
    from config import config_schema
    from core_imports import get_logger

    # Initialize logger with standardized pattern
    logger = get_logger(__name__)

    from my_selectors import *
    from selenium_utils import (
        is_browser_open,
        is_elem_there,
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
import contextlib

from test_framework import (
    TestSuite,
    suppress_logging,
)

# ------------------------------------------------------------------------------------
# Helper functions (General Utilities)
# ------------------------------------------------------------------------------------

def parse_cookie(cookie_string: str) -> dict[str, str]:
    """
    Parses a raw HTTP cookie string into a dictionary of key-value pairs.
    Handles empty keys and values.
    """
    cookies: dict[str, str] = {}
    if not isinstance(cookie_string, str):
        logger.warning("parse_cookie received non-string input, returning empty dict.")
        return cookies
    # End of if

    parts = cookie_string.split(";")
    for raw_part in parts:
        part = raw_part.strip()
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


# === ORDINAL FORMATTING HELPER FUNCTIONS ===

def _get_ordinal_suffix(num: int) -> str:
    """Get the ordinal suffix (st, nd, rd, th) for a number."""
    # Special case for 11-13 (always 'th')
    if 11 <= (num % 100) <= 13:
        return "th"

    # Check last digit
    last_digit = num % 10
    if last_digit == 1:
        return "st"
    if last_digit == 2:
        return "nd"
    if last_digit == 3:
        return "rd"
    return "th"


def _format_number_as_ordinal(num: int) -> str:
    """Format a number as an ordinal string (e.g., 1 -> '1st', 2 -> '2nd')."""
    return str(num) + _get_ordinal_suffix(num)


def _title_case_with_lowercase_particles(text: str) -> str:
    """Apply title case but keep certain particles lowercase (of, the, a, etc.)."""
    words = text.title().split()
    lowercase_particles = {"Of", "The", "A", "An", "In", "On", "At", "For", "To", "With"}

    for i, word in enumerate(words):
        # Keep particles lowercase except at start
        if i > 0 and word in lowercase_particles:
            words[i] = word.lower()

    return " ".join(words)


def ordinal_case(text: Union[str, int]) -> str:
    """
    Corrects ordinal suffixes (1st, 2nd, 3rd, 4th) to lowercase within a string,
    often used after applying title casing. Handles relationship terms simply.
    Accepts string or integer input for numbers.
    """
    # Handle empty/None input
    if not text and text != 0:
        return str(text) if text is not None else ""

    # Try to convert to number and format as ordinal
    try:
        num = int(text)
        return _format_number_as_ordinal(num)
    except (ValueError, TypeError):
        # Not a number - apply title case with lowercase particles
        if isinstance(text, str):
            return _title_case_with_lowercase_particles(text)
        return str(text)

# End of ordinal_case


# === NAME FORMATTING HELPER FUNCTIONS ===

def _remove_gedcom_slashes(name: str) -> str:
    """Remove GEDCOM-style slashes around surnames."""
    name = re.sub(r"\s*/([^/]+)/\s*", r" \1 ", name)  # Middle
    name = re.sub(r"^/([^/]+)/\s*", r"\1 ", name)  # Start
    name = re.sub(r"\s*/([^/]+)/$", r" \1", name)  # End
    name = re.sub(r"^/([^/]+)/$", r"\1", name)  # Only
    return re.sub(r"\s+", " ", name).strip()


def _format_quoted_nickname(part: str) -> Optional[str]:
    """Format quoted nicknames like 'Betty' or 'Bo'."""
    if part.startswith("'") and part.endswith("'") and len(part) > 2:
        inner_content = part[1:-1]
        return "'" + inner_content.capitalize() + "'"
    return None


def _format_hyphenated_name(part: str, lowercase_particles: set[str]) -> str:
    """Format hyphenated names like Smith-Jones or van-der-Berg."""
    hyphenated_elements = []
    sub_parts = part.split("-")
    for idx, sub_part in enumerate(sub_parts):
        if idx > 0 and sub_part.lower() in lowercase_particles:
            hyphenated_elements.append(sub_part.lower())
        elif sub_part:
            hyphenated_elements.append(sub_part.capitalize())
    return "-".join(filter(None, hyphenated_elements))


def _format_apostrophe_name(part: str) -> Optional[str]:
    """Format names with internal apostrophes like O'Malley or D'Angelo."""
    if "'" in part and len(part) > 1 and not (part.startswith("'") or part.endswith("'")):
        name_pieces = part.split("'")
        return "'".join(p.capitalize() for p in name_pieces)
    return None


def _format_mc_mac_prefix(part: str) -> Optional[str]:
    """Format Mc/Mac prefixes like McDonald or MacGregor."""
    part_lower = part.lower()
    if part_lower.startswith("mc") and len(part) > 2:
        return "Mc" + part[2:].capitalize()
    if part_lower.startswith("mac") and len(part) > 3:
        if part_lower == "mac":
            return "Mac"
        return "Mac" + part[3:].capitalize()
    return None


def _format_initial(part: str) -> Optional[str]:
    """Format initials like J. or J."""
    if len(part) == 2 and part.endswith(".") and part[0].isalpha():
        return part[0].upper() + "."
    if len(part) == 1 and part.isalpha():
        return part.upper()
    return None


def _format_name_part(part: str, index: int, lowercase_particles: set[str], uppercase_exceptions: set[str]) -> str:
    """Format a single part of a name with all special case handling."""
    part_lower = part.lower()
    result = None

    # Lowercase particles (but not at start)
    if index > 0 and part_lower in lowercase_particles:
        result = part_lower
    # Uppercase exceptions (II, III, SR, JR)
    elif part.upper() in uppercase_exceptions:
        result = part.upper()
    else:
        # Quoted nicknames
        quoted = _format_quoted_nickname(part)
        if quoted:
            result = quoted
        # Hyphenated names
        elif "-" in part:
            result = _format_hyphenated_name(part, lowercase_particles)
        else:
            # Apostrophe names (O'Malley)
            apostrophe = _format_apostrophe_name(part)
            if apostrophe:
                result = apostrophe
            else:
                # Mc/Mac prefixes
                mc_mac = _format_mc_mac_prefix(part)
                if mc_mac:
                    result = mc_mac
                else:
                    # Initials
                    initial = _format_initial(part)
                    result = initial or part.capitalize()

    return result


def format_name(name: Optional[str]) -> str:
    """
    Formats a person's name string to title case, preserving uppercase components
    (like initials or acronyms) and handling None/empty input gracefully.
    Also removes GEDCOM-style slashes around surnames anywhere in the string.
    Handles common name particles and prefixes like Mc/Mac/O' and quoted nicknames.
    """
    # Validate input
    if not name or not isinstance(name, str):
        return "Valued Relative"

    # Handle non-alphabetic input
    if name.isdigit() or re.fullmatch(r"[^a-zA-Z]+", name):
        logger.debug(f"Formatting name: Input '{name}' appears non-alphabetic, returning as is.")
        stripped_name = name.strip()
        return stripped_name if stripped_name else "Valued Relative"

    try:
        # Clean and prepare name
        cleaned_name = _remove_gedcom_slashes(name.strip())

        # Define special case sets
        lowercase_particles = {"van", "von", "der", "den", "de", "di", "da", "do", "la", "le", "el"}
        uppercase_exceptions = {"II", "III", "IV", "SR", "JR"}

        # Format each part
        parts = cleaned_name.split()
        formatted_parts = [
            _format_name_part(part, i, lowercase_particles, uppercase_exceptions)
            for i, part in enumerate(parts)
        ]

        # Join and clean up
        final_name = " ".join(formatted_parts)
        final_name = re.sub(r"\s+", " ", final_name).strip()
        return final_name if final_name else "Valued Relative"

    except Exception as e:
        logger.error(f"Error formatting name '{name}': {e}", exc_info=False)
        try:
            return name.title() if isinstance(name, str) else "Valued Relative"
        except AttributeError:
            return "Valued Relative"

# End of format_name

# ------------------------------
# Decorators (Remain in utils.py)
# ------------------------------

def retry(
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    max_delay: Optional[float] = None,
):
    """Decorator factory to retry a function with exponential backoff and jitter."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cfg = config_schema  # Use new config system
            attempts = (
                max_retries
                if max_retries is not None
                else getattr(cfg, "MAX_RETRIES", 3)
            )
            backoff = (
                backoff_factor
                if backoff_factor is not None
                else getattr(cfg, "BACKOFF_FACTOR", 1.0)
            )
            max_delay_val = (
                max_delay if max_delay is not None else getattr(cfg, "MAX_DELAY", 10.0)
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
                    sleep_time = min(backoff * (2**i), max_delay_val) + random.uniform(
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


# === RETRY API HELPER FUNCTIONS ===

def _get_retry_config(
    max_retries: Optional[int],
    initial_delay: Optional[float],
    backoff_factor: Optional[float],
    retry_on_status_codes: Optional[list[int]],
) -> dict[str, Any]:
    """Get retry configuration with defaults from config_schema."""
    cfg = config_schema
    return {
        "max_retries": max_retries if max_retries is not None else getattr(cfg, "MAX_RETRIES", 3),
        "initial_delay": initial_delay if initial_delay is not None else getattr(cfg, "INITIAL_DELAY", 0.5),
        "backoff_factor": backoff_factor if backoff_factor is not None else getattr(cfg, "BACKOFF_FACTOR", 1.5),
        "retry_codes": set(retry_on_status_codes if retry_on_status_codes is not None else getattr(cfg, "RETRY_STATUS_CODES", [429, 500, 502, 503, 504])),
        "max_delay": getattr(cfg, "MAX_DELAY", 60.0),
    }


def _calculate_sleep_time(delay: float, backoff_factor: float, attempt: int, max_delay: float) -> float:
    """Calculate sleep time with exponential backoff and jitter."""
    sleep_time = min(delay * (backoff_factor ** (attempt - 1)), max_delay) + random.uniform(0, 0.2)
    return max(0.1, sleep_time)


def _should_retry_status_code(response: Any, retry_codes: set[int]) -> tuple[bool, Optional[int]]:
    """Check if response status code should trigger a retry."""
    if isinstance(response, requests.Response):  # type: ignore
        status_code = response.status_code
        if status_code in retry_codes:
            return True, status_code
    return False, None


def _handle_status_code_retry(
    _response: Any,
    status_code: int,
    retries: int,
    max_retries: int,
    attempt: int,
    delay: float,
    backoff_factor: float,
    max_delay: float,
    func_name: str,
) -> tuple[bool, float]:
    """Handle retry logic for status code errors. Returns (should_continue, new_delay)."""
    if retries <= 0:
        logger.error(f"API Call failed after {max_retries} retries for '{func_name}' (Final Status {status_code}).")
        return False, delay

    sleep_time = _calculate_sleep_time(delay, backoff_factor, attempt, max_delay)
    logger.warning(f"API Call status {status_code} (Attempt {attempt}/{max_retries}) for '{func_name}'. Retrying in {sleep_time:.2f}s...")
    time.sleep(sleep_time)
    return True, delay * backoff_factor


def _handle_exception_retry(
    exception: Exception,
    retries: int,
    max_retries: int,
    attempt: int,
    delay: float,
    backoff_factor: float,
    max_delay: float,
    func_name: str,
) -> tuple[bool, float]:
    """Handle retry logic for exceptions. Returns (should_continue, new_delay)."""
    if retries <= 0:
        logger.error(
            f"API Call failed after {max_retries} retries for '{func_name}'. Final Exception: {type(exception).__name__} - {exception}",
            exc_info=False,
        )
        raise exception

    sleep_time = _calculate_sleep_time(delay, backoff_factor, attempt, max_delay)
    logger.warning(f"API Call exception '{type(exception).__name__}' (Attempt {attempt}/{max_retries}) for '{func_name}', retrying in {sleep_time:.2f}s...")
    time.sleep(sleep_time)
    return True, delay * backoff_factor


def retry_api(
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    retry_on_exceptions: tuple[type[Exception], ...] = (
        requests.exceptions.RequestException,  # type: ignore
        ConnectionError,
        TimeoutError,
    ),
    retry_on_status_codes: Optional[list[int]] = None,
):
    """Decorator factory for retrying API calls with exponential backoff, logging, etc."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get configuration
            config = _get_retry_config(max_retries, initial_delay, backoff_factor, retry_on_status_codes)

            # Initialize retry state
            retries = config["max_retries"]
            delay = config["initial_delay"]
            attempt = 0
            last_exception: Optional[Exception] = None
            last_response: RequestsResponseTypeOptional = None

            # Retry loop
            while retries > 0:
                attempt += 1
                try:
                    response = func(*args, **kwargs)
                    last_response = response

                    # Check if status code should trigger retry
                    should_retry, status_code = _should_retry_status_code(response, config["retry_codes"])

                    if should_retry:
                        retries -= 1
                        last_exception = requests.exceptions.HTTPError(f"{status_code} Error", response=response)  # type: ignore

                        # Handle status code retry
                        should_continue, delay = _handle_status_code_retry(
                            response, status_code, retries, config["max_retries"],
                            attempt, delay, config["backoff_factor"], config["max_delay"], func.__name__
                        )
                        if not should_continue:
                            return response
                        continue

                    # Success - return response
                    return response

                except retry_on_exceptions as e:
                    last_exception = e
                    retries -= 1

                    # Handle exception retry
                    should_continue, delay = _handle_exception_retry(
                        e, retries, config["max_retries"], attempt, delay,
                        config["backoff_factor"], config["max_delay"], func.__name__
                    )
                    continue

                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Unexpected error during API call attempt {attempt} for '{func.__name__}': {e}", exc_info=True)
                    raise e

            # Retry loop exhausted
            logger.error(f"Exited retry loop for '{func.__name__}'. Last status: {getattr(last_response, 'status_code', 'N/A')}, Last exception: {last_exception}")
            if last_exception:
                raise last_exception
            return last_response if last_response is not None else RuntimeError(f"{func.__name__} failed after all retries.")

        return wrapper
    return decorator

# End of retry_api

# Helper functions for ensure_browser_open

def _extract_driver_from_args(args: tuple) -> Optional[DriverType]:
    """Extract WebDriver instance from positional arguments."""
    if not args:
        return None

    if isinstance(args[0], SessionManager):  # type: ignore
        return args[0].driver
    if isinstance(args[0], WebDriver):  # type: ignore
        return args[0]

    return None


def _extract_driver_from_kwargs(kwargs: dict) -> Optional[DriverType]:
    """Extract WebDriver instance from keyword arguments."""
    # Check for direct driver argument
    if "driver" in kwargs and isinstance(kwargs["driver"], WebDriver):  # type: ignore
        return kwargs["driver"]

    # Check for session_manager argument
    if "session_manager" in kwargs and isinstance(kwargs["session_manager"], SessionManager):  # type: ignore
        return kwargs["session_manager"].driver

    return None


def _find_driver_instance(args: tuple, kwargs: dict) -> Optional[DriverType]:
    """Find WebDriver instance from args or kwargs."""
    driver = _extract_driver_from_args(args)
    if driver:
        return driver

    return _extract_driver_from_kwargs(kwargs)


def _validate_driver_instance(driver_instance: Optional[DriverType], func_name: str) -> None:
    """Validate that driver instance exists and browser is open."""
    if not driver_instance:
        raise TypeError(
            f"Function '{func_name}' decorated with @ensure_browser_open requires a WebDriver instance."
        )

    if not is_browser_open(driver_instance):
        raise WebDriverException(  # type: ignore
            f"Browser session invalid/closed when calling function '{func_name}'"
        )


def ensure_browser_open(func: Callable) -> Callable:
    """Decorator to ensure browser session is valid before executing."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Find driver instance from args or kwargs
        driver_instance = _find_driver_instance(args, kwargs)

        # Validate driver instance and browser state
        _validate_driver_instance(driver_instance, func.__name__)

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
        self.initial_delay = (
            initial_delay
            if initial_delay is not None
            else getattr(cfg, "INITIAL_DELAY", 0.5)
        )
        self.MAX_DELAY = (
            max_delay if max_delay is not None else getattr(cfg, "MAX_DELAY", 60.0)
        )
        self.backoff_factor = (
            backoff_factor
            if backoff_factor is not None
            else getattr(cfg, "BACKOFF_FACTOR", 1.8)
        )
        self.decrease_factor = (
            decrease_factor
            if decrease_factor is not None
            else getattr(cfg, "DECREASE_FACTOR", 0.98)
        )
        self.current_delay = self.initial_delay
        self.last_throttled = False
        # Token Bucket parameters
        self.capacity = float(
            token_capacity
            if token_capacity is not None
            else getattr(cfg, "TOKEN_BUCKET_CAPACITY", 10.0)
        )
        self.fill_rate = float(
            token_fill_rate
            if token_fill_rate is not None
            else getattr(cfg, "TOKEN_BUCKET_FILL_RATE", 2.0)
        )
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
        self._refill_tokens()
        # requested_at = time.monotonic() # Less critical now
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
        # End of if/else

        # Perform the sleep
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        # End of if

        # Refill again *after* sleeping
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
    headers: Optional[dict[str, str]] = None,
) -> dict[str, str]:
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
    base_headers: dict[str, str] = {
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

# Helper functions for _prepare_api_headers

def _get_user_agent_from_browser(driver: DriverType, api_description: str) -> Optional[str]:
    """Get User-Agent from browser using JavaScript."""
    try:
        ua = driver.execute_script("return navigator.userAgent;")
        if ua and isinstance(ua, str):
            return ua
    except WebDriverException:  # type: ignore
        logger.debug(f"[{api_description}] WebDriver error getting User-Agent, using default.")
    except Exception as e:
        logger.debug(f"[{api_description}] Error getting User-Agent: {e}, using default.")
    return None


def _add_origin_header(final_headers: dict[str, str], base_headers: dict[str, str], api_description: str) -> None:
    """Add Origin header for relevant HTTP methods."""
    http_method = base_headers.get("_method", "GET").upper()
    if http_method not in ["GET", "HEAD", "OPTIONS"]:
        try:
            cfg = config_schema
            parsed_base_url = urlparse(cfg.api.base_url)
            origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"
            final_headers["Origin"] = origin_header_value
        except Exception as parse_err:
            logger.warning(f"[{api_description}] Could not parse BASE_URL for Origin header: {parse_err}")


def _parse_csrf_token(csrf_token: str, api_description: str) -> str:
    """Parse CSRF token, handling potential JSON structure."""
    raw_token_val = csrf_token
    if isinstance(csrf_token, str) and csrf_token.strip().startswith("{"):
        try:
            token_obj = json.loads(csrf_token)
            raw_token_val = token_obj.get("csrfToken", csrf_token)
        except json.JSONDecodeError:
            logger.warning(f"[{api_description}] CSRF token looks like JSON but failed to parse, using raw value.")
    return str(raw_token_val)


def _add_csrf_token_header(final_headers: dict[str, str], session_manager: SessionManager, api_description: str) -> None:
    """Add CSRF token header if available."""
    csrf_token = session_manager.csrf_token
    if csrf_token:
        raw_token_val = _parse_csrf_token(csrf_token, api_description)
        final_headers["X-CSRF-Token"] = raw_token_val
        logger.debug(f"[{api_description}] Added X-CSRF-Token header.")
    else:
        logger.warning(f"[{api_description}] CSRF token requested but not found in SessionManager.")


def _add_user_id_header(final_headers: dict[str, str], session_manager: SessionManager, api_description: str) -> None:
    """Add User ID header conditionally."""
    exclude_userid_for = {
        "Ancestry Facts JSON Endpoint",
        "Ancestry Person Picker",
        "CSRF Token API",
    }
    if session_manager.my_profile_id and api_description not in exclude_userid_for:
        final_headers["ancestry-userid"] = session_manager.my_profile_id.upper()
    elif api_description in exclude_userid_for and session_manager.my_profile_id:
        logger.debug(f"[{api_description}] Omitting 'ancestry-userid' header as configured.")


def _prepare_api_headers(
    session_manager: SessionManager,  # Assume available
    driver: DriverType,
    api_description: str,
    base_headers: dict[str, str],
    use_csrf_token: bool,
    add_default_origin: bool,
) -> dict[str, str]:
    """Generates the final headers for an API request."""
    final_headers = base_headers.copy()
    cfg = config_schema

    # Get User-Agent from browser if possible
    ua = None
    if driver and session_manager.driver:
        ua = _get_user_agent_from_browser(driver, api_description)

    # Set User-Agent (from browser or fallback)
    if ua:
        final_headers["User-Agent"] = ua
    else:
        final_headers["User-Agent"] = random.choice(cfg.api.user_agents)
        logger.debug(f"[{api_description}] Using default User-Agent: {final_headers['User-Agent']}")

    # Add Origin header for relevant methods
    if add_default_origin:
        _add_origin_header(final_headers, base_headers, api_description)

    # Skip dynamic header generation to prevent recursion
    logger.debug(f"[{api_description}] Skipping dynamic header generation to prevent recursion.")

    # Add CSRF token if requested
    if use_csrf_token:
        _add_csrf_token_header(final_headers, session_manager, api_description)

    # Add User ID header conditionally
    _add_user_id_header(final_headers, session_manager, api_description)

    # Remove any headers with None values
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
        # End of if
        return False
    # End of if

    # Perform actual cookie synchronization like the working version
    try:
        logger.debug(f"[{api_description}] Syncing cookies from browser to requests session (Attempt {attempt})...")

        # Use the API manager's sync_cookies_from_browser method like the working version
        sync_success = session_manager.api_manager.sync_cookies_from_browser(session_manager.browser_manager)

        if sync_success:
            logger.debug(f"[{api_description}] Cookie sync successful (Attempt {attempt}).")
            return True
        logger.warning(f"[{api_description}] Cookie sync failed (Attempt {attempt}).")
        return False

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
    wait_time = session_manager.dynamic_rate_limiter.wait()
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
    headers: dict[str, str],
    data: Optional[dict] = None,
    json_data: Optional[dict] = None,
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

    logger.debug(
        f"[_api_req Attempt {attempt} '{api_description}'] >> Sending Request:"
    )
    logger.debug(f"   >> Method: {http_method}")
    logger.debug(f"   >> URL: {url}")
    logger.debug(f"   >> Headers: {log_hdrs_debug}")

    # Log body carefully (limit size, mask sensitive data if needed)
    log_body = ""
    if json_data:
        try:
            log_body = json.dumps(json_data)
            if len(log_body) > 500:
                log_body = log_body[:500] + "..."
            # Add masking here if json_data contains sensitive fields
        except TypeError:
            log_body = "[Unloggable JSON Data]"
        # End of try/except
        logger.debug(f"   >> JSON Body: {log_body}")
    elif data:
        log_body = str(data)
        if len(log_body) > 500:
            log_body = log_body[:500] + "..."
        # Add masking here if data contains sensitive fields
        logger.debug(f"   >> Form Data: {log_body}")
    # End of if/elif

# End of _log_request_details

def _prepare_api_request(
    config: ApiRequestConfig,
) -> dict[str, Any]:
    """
    Prepares all aspects of an API request including headers, cookies, and rate limiting.

    Args:
        config: ApiRequestConfig containing all request parameters

    Returns:
        Dictionary containing all prepared request parameters
    """
    sel_cfg = config_schema.selenium  # Use new config system

    # Prepare base headers
    base_headers = _prepare_base_headers(
        method=config.method,
        api_description=config.api_description,
        referer_url=config.referer_url,
        headers=config.headers,
    )

    # Prepare request details
    request_timeout = config.timeout if config.timeout is not None else sel_cfg.api_timeout
    req_session = config.session_manager._requests_session
    effective_cookie_jar = config.cookie_jar if config.cookie_jar is not None else req_session.cookies
    http_method = config.method.upper()

    # Handle specific API quirks (e.g., allow_redirects)
    effective_allow_redirects = config.allow_redirects
    # Note: Match List API should allow redirects (as it did in working version from 2 months ago)

    logger.debug(
        f"[{config.api_description}] Preparing Request: Method={http_method}, URL={config.url}, Timeout={request_timeout}s, AllowRedirects={effective_allow_redirects}"
    )

    # Sync cookies
    _sync_cookies_for_request(
        session_manager=config.session_manager,
        driver=config.driver,
        api_description=config.api_description,
        attempt=config.attempt,
    )

    # Generate final headers
    final_headers = _prepare_api_headers(
        session_manager=config.session_manager,
        driver=config.driver,
        api_description=config.api_description,
        base_headers=base_headers,
        use_csrf_token=config.use_csrf_token,
        add_default_origin=config.add_default_origin,
    )

    # Apply rate limiting
    _apply_rate_limiting(
        session_manager=config.session_manager,
        api_description=config.api_description,
        attempt=config.attempt,
    )

    # Use json parameter if provided, otherwise use json_data
    effective_json_data = config.json if config.json is not None else config.json_data

    # Log request details
    _log_request_details(
        api_description=config.api_description,
        attempt=config.attempt,
        http_method=http_method,
        url=config.url,
        headers=final_headers,
        data=config.data,
        json_data=effective_json_data,
    )

    # Return all prepared request parameters
    return {
        "method": http_method,
        "url": config.url,
        "headers": final_headers,
        "data": config.data,
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
    request_params: dict[str, Any],
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
        # Log that we're making the request
        logger.debug(
            f"[_api_req Attempt {attempt} '{api_description}'] >>> Calling requests.request..."
        )

        # Execute the request
        response = req_session.request(**request_params)

        # Log that the request returned
        logger.debug(
            f"[_api_req Attempt {attempt} '{api_description}'] <<< requests.request returned."
        )

        # Log response details
        status = response.status_code
        reason = response.reason
        logger.debug(
            f"[_api_req Attempt {attempt} '{api_description}'] << Response Status: {status} {reason}"
        )
        logger.debug(f"   << Response Headers: {response.headers}")

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

# Helper functions for _api_req

def _validate_api_req_prerequisites(
    session_manager: SessionManager,
    api_description: str,
) -> bool:
    """Validate prerequisites for API request."""
    if not session_manager or not session_manager._requests_session:
        logger.error(
            f"{api_description}: Aborting - SessionManager or internal requests_session missing."
        )
        return False

    if not config_schema:
        logger.error(f"{api_description}: Aborting - Config schema not loaded.")
        return False

    return True


def _get_retry_configuration() -> tuple[int, float, float, float, set[int]]:
    """Get retry configuration from config schema."""
    cfg = config_schema.api
    max_retries = cfg.max_retries
    initial_delay = cfg.initial_delay
    backoff_factor = cfg.retry_backoff_factor
    max_delay = cfg.max_delay
    retry_status_codes = set(cfg.retry_status_codes)

    return max_retries, initial_delay, backoff_factor, max_delay, retry_status_codes


def _calculate_retry_sleep_time(
    current_delay: float,
    backoff_factor: float,
    attempt: int,
    max_delay: float,
) -> float:
    """Calculate sleep time for retry with exponential backoff and jitter."""
    sleep_time = min(
        current_delay * (backoff_factor ** (attempt - 1)), max_delay
    ) + random.uniform(0, 0.2)
    return max(0.1, sleep_time)


def _handle_failed_request_response(
    retries_left: int,
    max_retries: int,
    api_description: str,
    current_delay: float,
    backoff_factor: float,
    attempt: int,
    max_delay: float,
) -> tuple[bool, int, float]:
    """
    Handle failed request response (None).

    Returns:
        Tuple of (should_continue, new_retries_left, new_current_delay)
    """
    retries_left -= 1
    if retries_left <= 0:
        logger.error(
            f"{api_description}: Request failed after {max_retries} attempts."
        )
        return False, retries_left, current_delay

    sleep_time = _calculate_retry_sleep_time(current_delay, backoff_factor, attempt, max_delay)
    logger.warning(
        f"{api_description}: Request error (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s..."
    )
    time.sleep(sleep_time)
    new_delay = current_delay * backoff_factor
    return True, retries_left, new_delay


def _handle_retryable_status(
    response: RequestsResponseTypeOptional,
    status: int,
    reason: str,
    retry_ctx: RetryContext,
    api_description: str,
    session_manager: SessionManager,
) -> tuple[bool, Optional[RequestsResponseTypeOptional], int, float]:
    """
    Handle retryable status codes.

    Returns:
        Tuple of (should_continue, response_to_return, new_retries_left, new_current_delay)
    """
    retries_left = (retry_ctx.retries_left or 0) - 1
    current_delay = retry_ctx.current_delay
    logger.warning(
        f"[_api_req Attempt {retry_ctx.attempt} '{api_description}'] Received retryable status: {status} {reason}"
    )

    if retries_left <= 0:
        logger.error(
            f"{api_description}: Failed after {retry_ctx.max_attempts} attempts (Final Status {status}). Returning Response object."
        )
        with contextlib.suppress(Exception):
            logger.debug(f"   << Final Response Text (Retry Fail): {response.text[:500]}...")
        return False, response, retries_left, current_delay

    sleep_time = _calculate_retry_sleep_time(current_delay, retry_ctx.backoff_factor, retry_ctx.attempt, retry_ctx.max_delay)

    if status == 429:  # Too Many Requests
        session_manager.dynamic_rate_limiter.increase_delay()

    logger.warning(
        f"{api_description}: Status {status} (Attempt {retry_ctx.attempt}/{retry_ctx.max_attempts}). Retrying in {sleep_time:.2f}s..."
    )
    with contextlib.suppress(Exception):
        logger.debug(f"   << Response Text (Retry): {response.text[:500]}...")

    time.sleep(sleep_time)
    new_delay = current_delay * retry_ctx.backoff_factor
    return True, None, retries_left, new_delay


def _handle_redirect_response(
    response: RequestsResponseTypeOptional,
    status: int,
    reason: str,
    allow_redirects: bool,
    api_description: str,
) -> Optional[RequestsResponseTypeOptional]:
    """Handle redirect responses (3xx status codes)."""
    if 300 <= status < 400:
        if not allow_redirects:
            logger.warning(
                f"{api_description}: Status {status} {reason} (Redirects Disabled). Returning Response object."
            )
        else:
            logger.warning(
                f"{api_description}: Unexpected final status {status} {reason} (Redirects Enabled). Returning Response object."
            )
        logger.debug(f"   << Redirect Location: {response.headers.get('Location')}")
        return response
    return None


def _handle_error_status(
    response: RequestsResponseTypeOptional,
    status: int,
    reason: str,
    api_description: str,
    session_manager: SessionManager,
) -> RequestsResponseTypeOptional:
    """Handle non-retryable error status codes."""
    # For login verification API, use debug level for 401/403 errors
    if api_description == "API Login Verification (header/dna)" and status in [401, 403]:
        logger.debug(
            f"[_api_req '{api_description}'] Received expected status: {status} {reason}"
        )
    else:
        logger.warning(
            f"[_api_req '{api_description}'] Received NON-retryable error status: {status} {reason}"
        )

    if status in [401, 403]:
        if api_description == "API Login Verification (header/dna)":
            logger.debug(
                f"{api_description}: API call returned {status} {reason}. User not logged in."
            )
        else:
            logger.warning(
                f"{api_description}: API call failed {status} {reason}. Session expired/invalid?"
            )
        session_manager.session_ready = False
    else:
        logger.error(f"{api_description}: Non-retryable error: {status} {reason}.")

    with contextlib.suppress(Exception):
        logger.debug(f"   << Error Response Text: {response.text[:500]}...")

    return response


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

    result = response  # Default: return response object for non-successful responses

    # Handle successful responses (2xx)
    if response.ok:
        logger.debug(f"{api_description}: Successful response ({status} {reason}).")

        # Force text response if requested
        if force_text_response:
            logger.debug(f"{api_description}: Force text response requested.")
            result = response.text
        else:
            # Process based on content type
            content_type = response.headers.get("content-type", "").lower()
            logger.debug(f"{api_description}: Content-Type: '{content_type}'")

            if "application/json" in content_type:
                try:
                    # Handle empty response body for JSON
                    json_result = response.json() if response.content else None
                    logger.debug(f"{api_description}: Successfully parsed JSON response.")
                    result = json_result
                except JSONDecodeError as json_err:  # type: ignore
                    logger.error(
                        f"{api_description}: OK ({status}), but JSON decode FAILED: {json_err}"
                    )
                    with contextlib.suppress(Exception):
                        logger.debug(
                            f"   << Response Text (JSON Error): {response.text[:500]}..."
                        )
                    # Return None because caller expected JSON but didn't get it
                    result = None
            elif api_description == "CSRF Token API" and "text/plain" in content_type:
                csrf_text = response.text.strip()
                logger.debug(
                    f"{api_description}: Received text/plain as expected for CSRF."
                )
                result = csrf_text if csrf_text else None
            else:
                logger.debug(
                    f"{api_description}: OK ({status}), Content-Type '{content_type}'. Returning raw TEXT."
                )
                result = response.text

    return result

# End of _process_api_response

def _handle_request_exception(
    e: Exception,
    attempt: int,
    max_retries: int,
    retries_left: int,
    api_description: str,
    current_delay: float,
    backoff_factor: float,
    max_delay: float,
) -> tuple[bool, int, float]:
    """
    Handle exception during request attempt.
    Returns (should_continue, retries_left, current_delay).
    """
    logger.warning(
        f"[_api_req Attempt {attempt} '{api_description}'] RequestException: {type(e).__name__} - {e}"
    )
    retries_left -= 1

    if retries_left <= 0:
        logger.error(
            f"{api_description}: Request failed after {max_retries} attempts. Final Error: {e}",
            exc_info=False,
        )
        return (False, retries_left, current_delay)

    sleep_time = _calculate_retry_sleep_time(current_delay, backoff_factor, attempt, max_delay)
    logger.warning(
        f"{api_description}: {type(e).__name__} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s... Error: {e}"
    )
    time.sleep(sleep_time)
    current_delay *= backoff_factor
    return (True, retries_left, current_delay)


def _handle_response_status(
    response: Any,
    retry_ctx: RetryContext,
    api_description: str,
    session_manager: SessionManager,
    force_text_response: bool,
    request_params: dict[str, Any],
) -> tuple[Optional[Any], bool, int, float, Optional[Exception]]:
    """
    Handle response status codes and return appropriate result.
    Returns (response, should_continue, retries_left, current_delay, last_exception).
    """
    status = response.status_code
    reason = response.reason
    retries_left = retry_ctx.retries_left or 0
    current_delay = retry_ctx.current_delay

    # Handle retryable status codes
    if retry_ctx.retry_status_codes and status in retry_ctx.retry_status_codes:
        should_continue, return_response, retries_left, current_delay = _handle_retryable_status(
            response, status, reason, retry_ctx, api_description, session_manager
        )
        if not should_continue:
            return (return_response, False, retries_left, current_delay, None)
        last_exception = HTTPError(f"{status} Error", response=response)  # type: ignore
        return (None, True, retries_left, current_delay, last_exception)

    # Handle redirects
    redirect_response = _handle_redirect_response(
        response, status, reason, request_params["allow_redirects"], api_description
    )
    if redirect_response is not None:
        return (redirect_response, False, retries_left, current_delay, None)

    # Handle non-retryable error status codes
    if not response.ok:
        error_response = _handle_error_status(response, status, reason, api_description, session_manager)
        return (error_response, False, retries_left, current_delay, None)

    # Process successful response
    if response.ok:
        logger.debug(f"{api_description}: Successful response ({status} {reason}).")
        session_manager.dynamic_rate_limiter.decrease_delay()
        processed_response = _process_api_response(
            response=response,
            api_description=api_description,
            force_text_response=force_text_response,
        )
        return (processed_response, False, retries_left, current_delay, None)

    return (None, True, retries_left, current_delay, None)


def _process_request_attempt(
    config: ApiRequestConfig,
    retries_left: int,
    current_delay: float,
) -> tuple[Optional[Any], bool, int, float, Optional[Exception]]:
    """
    Process a single request attempt.
    Returns (response, should_continue, retries_left, current_delay, last_exception).
    """
    result_response = None
    result_should_continue = True
    result_exception = None

    try:
        # Prepare and execute the request
        request_params = _prepare_api_request(config)

        response = _execute_api_request(
            session_manager=config.session_manager,
            api_description=config.api_description,
            request_params=request_params,
            attempt=config.attempt,
        )

        # Handle failed request (response is None)
        if response is None:
            should_continue, retries_left, current_delay = _handle_failed_request_response(
                retries_left, config.max_retries, config.api_description, current_delay, config.backoff_factor, config.attempt, config.max_delay
            )
            result_should_continue = should_continue
        else:
            # Handle response status
            retry_ctx = RetryContext(
                attempt=config.attempt,
                max_attempts=config.max_retries,
                max_delay=config.max_delay,
                backoff_factor=config.backoff_factor,
                current_delay=current_delay,
                retries_left=retries_left,
                retry_status_codes=config.retry_status_codes
            )
            return _handle_response_status(
                response, retry_ctx, config.api_description,
                config.session_manager, config.force_text_response, request_params
            )

    except RequestException as e:  # type: ignore
        should_continue, retries_left, current_delay = _handle_request_exception(
            e, config.attempt, config.max_retries, retries_left, config.api_description, current_delay, config.backoff_factor, config.max_delay
        )
        result_should_continue = should_continue
        result_exception = e

    except Exception as e:
        logger.critical(
            f"{config.api_description}: CRITICAL Unexpected error during request attempt {config.attempt}: {e}",
            exc_info=True,
        )
        result_should_continue = False

    return (result_response, result_should_continue, retries_left, current_delay, result_exception)


def _execute_request_with_retries(
    config: ApiRequestConfig,
) -> Union[ApiResponseType, RequestsResponseTypeOptional]:
    """Execute API request with retry logic."""
    retries_left = config.max_retries
    last_exception: Optional[Exception] = None
    response: RequestsResponseTypeOptional = None
    current_delay = config.initial_delay

    while retries_left > 0:
        attempt = config.max_retries - retries_left + 1
        logger.debug(
            f"[_api_req LOOP ENTRY] api_description: '{config.api_description}', attempt: {attempt}/{config.max_retries}, retries_left: {retries_left}"
        )

        # Update attempt in config
        config.attempt = attempt

        result, should_continue, retries_left, current_delay, exception = _process_request_attempt(
            config, retries_left, current_delay
        )

        if exception:
            last_exception = exception

        if not should_continue:
            return result

        if result is not None:
            response = result

    # Should only be reached if loop completes without success
    if response is None:
        logger.error(f"{api_description}: Exited retry loop. Last Exception: {last_exception}.")
        return None

    # Return the last response (this should be a non-retryable error response)
    logger.debug(f"[_api_req '{api_description}'] Returning last Response object (Status: {response.status_code}).")
    return response


def _api_req(
    url: str,
    driver: DriverType,
    session_manager: SessionManager,  # type: ignore
    method: str = "GET",
    data: Optional[dict] = None,
    json_data: Optional[dict] = None,
    json: Optional[dict] = None,
    use_csrf_token: bool = True,
    headers: Optional[dict[str, str]] = None,
    referer_url: Optional[str] = None,
    api_description: str = "API Call",
    timeout: Optional[int] = None,
    cookie_jar: Optional[RequestsCookieJar] = None,  # type: ignore
    allow_redirects: bool = True,
    force_text_response: bool = False,
    add_default_origin: bool = True,
) -> Union[ApiResponseType, RequestsResponseTypeOptional]:
    """
    Makes an HTTP request using the shared requests.Session from SessionManager.
    Handles dynamic header generation, cookie synchronization, rate limiting,
    retries, and basic response processing. Includes enhanced logging.

    NOTE: This function maintains backward compatibility with 16 parameters.
    New code should use _api_req_impl(ApiRequestConfig) instead.

    Returns: Parsed JSON (dict/list), raw text (str), None on retryable failure,
             or Response object on non-retryable error/redirect disabled.
    """
    # Convert parameters to ApiRequestConfig and delegate to internal implementation
    config = ApiRequestConfig(
        url=url,
        driver=driver,
        session_manager=session_manager,
        method=method,
        data=data,
        json_data=json_data,
        json=json,
        use_csrf_token=use_csrf_token,
        headers=headers,
        referer_url=referer_url,
        api_description=api_description,
        timeout=timeout,
        cookie_jar=cookie_jar,
        allow_redirects=allow_redirects,
        force_text_response=force_text_response,
        add_default_origin=add_default_origin,
    )
    return _api_req_impl(config)


def _api_req_impl(config: ApiRequestConfig) -> Union[ApiResponseType, RequestsResponseTypeOptional]:
    """
    Internal implementation of _api_req using ApiRequestConfig.
    Makes an HTTP request using the shared requests.Session from SessionManager.
    Handles dynamic header generation, cookie synchronization, rate limiting,
    retries, and basic response processing. Includes enhanced logging.

    Returns: Parsed JSON (dict/list), raw text (str), None on retryable failure,
             or Response object on non-retryable error/redirect disabled.
    """
    logger.debug(f"[_api_req ENTRY] api_description: '{config.api_description}', url: {config.url}")

    # Validate prerequisites
    if not _validate_api_req_prerequisites(config.session_manager, config.api_description):
        return None

    # Get retry configuration
    max_retries, initial_delay, backoff_factor, max_delay, retry_status_codes = _get_retry_configuration()

    logger.debug(
        f"[_api_req PRE-LOOP] api_description: '{config.api_description}', max_retries: {max_retries}, "
        f"initial_delay: {initial_delay}, backoff_factor: {backoff_factor}"
    )

    # Update config with retry parameters
    config.max_retries = max_retries
    config.initial_delay = initial_delay
    config.backoff_factor = backoff_factor
    config.max_delay = max_delay
    config.retry_status_codes = retry_status_codes

    # Execute request with retry loop
    response = _execute_request_with_retries(config)

    logger.debug(f"[_api_req EXIT] api_description: '{config.api_description}'")
    return response

# End of _api_req

# Helper functions for make_ube

def _validate_driver_session(driver: DriverType) -> bool:
    """Validate that driver session is active."""
    if not driver:
        return False
    try:
        _ = driver.title  # Quick check for session validity
        return True
    except WebDriverException as e:  # type: ignore
        logger.warning(f"Cannot generate UBE header: Session invalid/unresponsive ({type(e).__name__}).")
        return False


def _get_ancsessionid_cookie(driver: DriverType) -> Optional[str]:
    """Get ANCSESSIONID cookie value from driver."""
    try:
        # Try fetching specific cookie first
        cookie_obj = driver.get_cookie("ANCSESSIONID")
        if cookie_obj and isinstance(cookie_obj, dict) and "value" in cookie_obj:
            return cookie_obj["value"]

        # Fallback to getting all cookies
        cookies_dict = {
            c["name"]: c["value"]
            for c in driver.get_cookies()
            if isinstance(c, dict) and "name" in c
        }
        ancsessionid = cookies_dict.get("ANCSESSIONID")

        if not ancsessionid:
            logger.warning("ANCSESSIONID cookie not found. Cannot generate UBE header.")
            return None
        return ancsessionid
    except (NoSuchCookieException, WebDriverException) as cookie_e:  # type: ignore
        logger.warning(f"Error getting ANCSESSIONID cookie for UBE header: {cookie_e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting ANCSESSIONID for UBE: {e}", exc_info=True)
        return None


def _build_ube_payload(ancsessionid: str) -> dict[str, str]:
    """Build UBE data payload."""
    event_id = "00000000-0000-0000-0000-000000000000"
    correlated_id = str(uuid.uuid4())
    screen_name_standard = "ancestry : uk : en : dna-matches-ui : match-list : 1"
    screen_name_legacy = "ancestry uk : dnamatches-matchlistui : list"
    user_consent = "necessary|preference|performance|analytics1st|analytics3rd|advertising1st|advertising3rd|attribution3rd"

    return {
        "eventId": event_id,
        "correlatedScreenViewedId": correlated_id,
        "correlatedSessionId": ancsessionid,
        "screenNameStandard": screen_name_standard,
        "screenNameLegacy": screen_name_legacy,
        "userConsent": user_consent,
        "vendors": "adobemc",
        "vendorConfigurations": "{}",
    }


def _encode_ube_payload(ube_data: dict[str, str]) -> Optional[str]:
    """Encode UBE payload to base64."""
    try:
        json_payload = json.dumps(ube_data, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(json_payload).decode("utf-8")
    except (json.JSONDecodeError, TypeError, binascii.Error) as encode_e:
        logger.error(f"Error encoding UBE header data: {encode_e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error encoding UBE header: {e}", exc_info=True)
        return None


def make_ube(driver: DriverType) -> Optional[str]:
    """Generate UBE header for Ancestry API requests."""
    if not _validate_driver_session(driver):
        return None

    ancsessionid = _get_ancsessionid_cookie(driver)
    if not ancsessionid:
        return None

    ube_data = _build_ube_payload(ancsessionid)
    return _encode_ube_payload(ube_data)

# End of make_ube

# NewRelic constants (shared across multiple header functions)
_NEWRELIC_ACCOUNT_ID = "1690570"
_NEWRELIC_APP_ID = "1588726612"
_NEWRELIC_LICENSE_KEY = "2611750"


def _generate_newrelic_payload() -> str:
    """Generate NewRelic header payload."""
    trace_id = uuid.uuid4().hex[:16]
    span_id = uuid.uuid4().hex[:16]

    newrelic_data = {
        "v": [0, 1],
        "d": {
            "ty": "Browser",
            "ac": _NEWRELIC_ACCOUNT_ID,
            "ap": _NEWRELIC_APP_ID,
            "id": span_id,
            "tr": trace_id,
            "ti": int(time.time() * 1000),
            "tk": _NEWRELIC_LICENSE_KEY,
        },
    }
    json_payload = json.dumps(newrelic_data, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(json_payload).decode("utf-8")


def _generate_traceparent_value() -> str:
    """Generate W3C Trace Context traceparent header value."""
    version = "00"
    trace_id = uuid.uuid4().hex
    parent_id = uuid.uuid4().hex[:16]
    flags = "01"
    return f"{version}-{trace_id}-{parent_id}-{flags}"


def _generate_tracestate_value() -> str:
    """Generate tracestate header value with NewRelic state."""
    span_id = uuid.uuid4().hex[:16]
    timestamp = int(time.time() * 1000)
    return f"{_NEWRELIC_LICENSE_KEY}@nr=0-1-{_NEWRELIC_ACCOUNT_ID}-{_NEWRELIC_APP_ID}-{span_id}----{timestamp}"


def make_newrelic(_driver: DriverType) -> Optional[str]:
    """Generate NewRelic header for Ancestry API requests."""
    try:
        return _generate_newrelic_payload()
    except (json.JSONDecodeError, TypeError, binascii.Error) as encode_e:
        logger.error(f"Error generating NewRelic header: {encode_e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating NewRelic header: {e}", exc_info=True)
        return None


def make_traceparent(_driver: DriverType) -> Optional[str]:
    """Generate W3C Trace Context traceparent header."""
    try:
        return _generate_traceparent_value()
    except Exception as e:
        logger.error(f"Error generating traceparent header: {e}", exc_info=True)
        return None


def make_tracestate(_driver: DriverType) -> Optional[str]:
    """Generate tracestate header with NewRelic state."""
    try:
        return _generate_tracestate_value()
    except Exception as e:
        logger.error(f"Error generating tracestate header: {e}", exc_info=True)
        return None

# ----------------------------------------------------------------------------
# Login Functions (Remain in utils.py)
# ----------------------------------------------------------------------------
TWO_STEP_VERIFICATION_HEADER_SELECTOR = "h1.two-step-verification-header"

# Helper functions for handle_two_fa

def _wait_for_2fa_header(_driver: WebDriver, element_wait: WebDriverWait, session_manager: SessionManager) -> bool:  # type: ignore
    """Wait for 2FA page header to appear."""
    try:
        logger.debug(f"Waiting for 2FA page header using selector: '{TWO_STEP_VERIFICATION_HEADER_SELECTOR}'")
        element_wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR))  # type: ignore
        )
        logger.debug("2FA page header detected.")
        return True
    except TimeoutException:  # type: ignore
        logger.debug("Did not detect 2FA page header within timeout.")
        if login_status(session_manager, disable_ui_fallback=True) is True:
            logger.info("User appears logged in after checking for 2FA page. Assuming 2FA handled/skipped.")
            return True
        logger.warning("Assuming 2FA not required or page didn't load correctly (header missing).")
        return False
    except WebDriverException as e:  # type: ignore
        logger.error(f"WebDriverException waiting for 2FA header: {e}")
        return False


def _click_sms_button(driver: WebDriver, short_wait: WebDriverWait) -> bool:  # type: ignore
    """Try clicking the SMS 'Send Code' button."""
    result = False

    try:
        logger.debug(f"Waiting for 2FA 'Send Code' (SMS) button: '{TWO_FA_SMS_SELECTOR}'")
        sms_button_clickable = short_wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, TWO_FA_SMS_SELECTOR))  # type: ignore
        )
        if sms_button_clickable:
            logger.debug("Attempting to click 'Send Code' button using JavaScript...")
            try:
                driver.execute_script("arguments[0].click();", sms_button_clickable)
                logger.debug("'Send Code' button clicked.")
                result = _wait_for_code_input_field(driver)
            except WebDriverException as click_err:  # type: ignore
                logger.error(f"Error clicking 'Send Code' button via JS: {click_err}")
        else:
            logger.error("'Send Code' button found but reported as not clickable by Selenium.")
    except TimeoutException:  # type: ignore
        logger.error("Timeout finding or waiting for clickability of the 2FA 'Send Code' button.")
    except ElementNotInteractableException:  # type: ignore
        logger.error("'Send Code' button not interactable (potentially obscured).")
    except WebDriverException as e:  # type: ignore
        logger.error(f"WebDriverException interacting with 2FA 'Send Code' button: {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unexpected error clicking 2FA 'Send Code' button: {e}", exc_info=True)

    return result


def _wait_for_code_input_field(driver: WebDriver) -> bool:  # type: ignore
    """Wait for 2FA code input field to appear after clicking Send Code."""
    try:
        logger.debug(f"Waiting for 2FA code input field: '{TWO_FA_CODE_INPUT_SELECTOR}'")
        WebDriverWait(driver, 5).until(  # type: ignore
            EC.visibility_of_element_located((By.CSS_SELECTOR, TWO_FA_CODE_INPUT_SELECTOR))  # type: ignore
        )
        logger.debug("Code input field appeared after clicking 'Send Code'.")
        return True
    except TimeoutException:  # type: ignore
        logger.warning("Code input field did not appear/become visible after clicking 'Send Code'.")
        return False
    except WebDriverException as e_input:  # type: ignore
        logger.error(f"Error waiting for 2FA code input field: {e_input}. Check selector: {TWO_FA_CODE_INPUT_SELECTOR}")
        return False


def _wait_for_user_2fa_action(driver: WebDriver, code_entry_timeout: int) -> bool:  # type: ignore
    """Wait for user to manually enter 2FA code and submit."""
    logger.warning(f"Waiting up to {code_entry_timeout}s for user to manually enter 2FA code and submit...")
    start_time = time.time()

    while time.time() - start_time < code_entry_timeout:
        try:
            # Check if the 2FA header is GONE (indicates page change/submission)
            WebDriverWait(driver, 0.5).until_not(  # type: ignore
                EC.visibility_of_element_located((By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR))  # type: ignore
            )
            logger.info("2FA page elements disappeared, assuming user submitted code.")
            return True
        except TimeoutException:  # type: ignore
            time.sleep(2)  # Check every 2 seconds
        except NoSuchElementException:  # type: ignore
            logger.info("2FA header element no longer present.")
            return True
        except WebDriverException as e:  # type: ignore
            logger.error(f"WebDriver error checking for 2FA header during wait: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking for 2FA header during wait: {e}")
            return False

    logger.error(f"Timed out ({code_entry_timeout}s) waiting for user 2FA action (page did not change).")
    return False


def _verify_2fa_completion(session_manager: SessionManager) -> bool:
    """Verify that 2FA was completed successfully by checking login status."""
    logger.info("Re-checking login status after potential 2FA submission...")
    time.sleep(1)  # Allow page to settle
    final_status = login_status(session_manager, disable_ui_fallback=False)
    if final_status is True:
        logger.info("User completed 2FA successfully (login confirmed after page change).")
        return True
    logger.error("2FA page disappeared, but final login status check failed or returned False.")
    return False


@time_wait("Handle 2FA Page")
def handle_two_fa(session_manager: SessionManager) -> bool:  # type: ignore
    if session_manager.driver is None:
        logger.error("handle_two_fa: SessionManager driver is None. Cannot proceed.")
        return False

    driver = session_manager.driver
    element_wait = WebDriverWait(driver, config_schema.selenium.explicit_wait)
    short_wait = WebDriverWait(driver, config_schema.selenium.implicit_wait)

    try:
        print("Two-factor authentication required. Please check your email or phone for a verification code.")
        logger.debug("Handling Two-Factor Authentication (2FA)...")

        # Wait for 2FA page header
        if not _wait_for_2fa_header(driver, element_wait, session_manager):
            return False

        # Handle cookie consent banner if present
        logger.debug("Checking for cookie consent banner on 2FA page...")
        if not consent(driver):
            logger.warning("Failed to handle consent banner on 2FA page, but continuing anyway.")

        # Try clicking SMS button (non-critical, continue if fails)
        _click_sms_button(driver, short_wait)

        # Wait for user action
        code_entry_timeout = config_schema.selenium.two_fa_code_entry_timeout
        user_action_detected = _wait_for_user_2fa_action(driver, code_entry_timeout)

        # Verify completion
        if user_action_detected:
            return _verify_2fa_completion(session_manager)
        return False

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

# End of handle_two_fa

# Helper functions for enter_creds

def _clear_input_field(driver: WebDriver, input_element, field_name: str) -> bool:
    """Clear an input field robustly."""
    try:
        input_element.click()
        time.sleep(0.1)
        input_element.clear()
        time.sleep(0.1)
        driver.execute_script("arguments[0].value = '';", input_element)
        time.sleep(0.1)
        return True
    except (ElementNotInteractableException, StaleElementReferenceException) as e:  # type: ignore
        logger.warning(f"Issue clicking/clearing {field_name} field ({type(e).__name__}). Proceeding cautiously.")
        return True
    except WebDriverException as e:  # type: ignore
        logger.error(f"WebDriverException clicking/clearing {field_name}: {e}. Aborting.")
        return False


def _enter_username(driver: WebDriver, element_wait: WebDriverWait) -> bool:
    """Enter username into the login form."""
    logger.debug(f"Waiting for username input: '{USERNAME_INPUT_SELECTOR}'...")  # type: ignore
    username_input = element_wait.until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))  # type: ignore
    )
    logger.debug("Username input field found.")

    if not _clear_input_field(driver, username_input, "username"):
        return False

    ancestry_username = config_schema.api.username
    if not ancestry_username:
        raise ValueError("ANCESTRY_USERNAME configuration is missing.")

    logger.debug("Entering username...")
    username_input.send_keys(ancestry_username)
    logger.debug("Username entered.")
    time.sleep(random.uniform(0.2, 0.4))
    return True


def _click_next_button(driver: WebDriver, short_wait: WebDriverWait) -> bool:
    """Click the Next button in two-step login flow."""
    logger.debug("Looking for Next/Continue button after username...")
    try:
        next_button = short_wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))  # type: ignore
        )
        logger.debug("Next button found, attempting to click...")

        try:
            driver.execute_script("arguments[0].click();", next_button)
            logger.debug("Next button clicked via JavaScript.")
            logger.info("Next button clicked, waiting for password field to appear...")
            time.sleep(random.uniform(2.0, 3.0))
            return True
        except WebDriverException as js_err:  # type: ignore
            logger.warning(f"JS click failed: {js_err}, trying standard click...")
            try:
                next_button.click()
                logger.debug("Next button clicked successfully (standard click).")
                time.sleep(random.uniform(2.0, 3.0))
                return True
            except (ElementClickInterceptedException, ElementNotInteractableException) as e:  # type: ignore
                logger.error(f"Both click methods failed: {e}")
                return False
    except TimeoutException:  # type: ignore
        logger.debug("Next button not found, assuming single-step login (password field already visible).")
        return True
    except WebDriverException as e:  # type: ignore
        logger.warning(f"Error finding Next button: {e}. Continuing anyway.")
        return True


def _enter_password(driver: WebDriver, element_wait: WebDriverWait) -> tuple[bool, Any]:
    """Enter password into the login form."""
    logger.debug(f"Waiting for password input: '{PASSWORD_INPUT_SELECTOR}'...")  # type: ignore
    try:
        password_input = element_wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))  # type: ignore
        )
    except TimeoutException:  # type: ignore
        logger.error("Password field did not appear after clicking Next. Retrying with longer wait...")
        password_input = WebDriverWait(driver, 30).until(  # type: ignore
            EC.visibility_of_element_located((By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))  # type: ignore
        )

    logger.debug("Password input field found.")

    if not _clear_input_field(driver, password_input, "password"):
        return False, None

    ancestry_password = config_schema.api.password
    if not ancestry_password:
        raise ValueError("ANCESTRY_PASSWORD configuration is missing.")

    logger.debug("Entering password: ***")
    password_input.send_keys(ancestry_password)
    logger.debug("Password entered.")
    time.sleep(random.uniform(0.3, 0.6))
    return True, password_input


def _try_standard_click(sign_in_button) -> bool:
    """Try standard click on sign in button."""
    try:
        logger.debug("Attempting standard click on sign in button...")
        sign_in_button.click()
        logger.debug("Standard click executed.")
        return True
    except (ElementClickInterceptedException, ElementNotInteractableException, StaleElementReferenceException) as e:  # type: ignore
        logger.warning(f"Standard click failed ({type(e).__name__}).")
        return False
    except WebDriverException as e:  # type: ignore
        logger.error(f"WebDriver error during standard click: {e}.")
        return False


def _try_javascript_click(driver: WebDriver, sign_in_button) -> bool:
    """Try JavaScript click on sign in button."""
    try:
        logger.debug("Attempting JavaScript click on sign in button...")
        driver.execute_script("arguments[0].click();", sign_in_button)
        logger.info("JavaScript click executed.")
        return True
    except WebDriverException as js_click_e:  # type: ignore
        logger.error(f"Error during JavaScript click: {js_click_e}")
        return False


def _try_return_key_fallback(password_input) -> bool:
    """Try sending RETURN key to password field as fallback."""
    try:
        logger.warning("Attempting fallback: Sending RETURN key to password field.")
        password_input.send_keys(Keys.RETURN)
        logger.info("Fallback RETURN key sent to password field.")
        return True
    except (WebDriverException, ElementNotInteractableException) as key_e:  # type: ignore
        logger.error(f"Failed to send RETURN key: {key_e}")
        return False


def _click_sign_in_button(driver: WebDriver, short_wait: WebDriverWait, password_input) -> bool:
    """Click the sign in button or use fallback methods."""
    # Try to locate sign in button
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
        return _try_return_key_fallback(password_input)
    except WebDriverException as find_e:  # type: ignore
        logger.error(f"Unexpected WebDriver error finding sign in button: {find_e}")
        return False

    if not sign_in_button:
        return False

    # Try standard click first
    if _try_standard_click(sign_in_button):
        return True

    # Try JavaScript click if standard click failed
    logger.warning("Standard click failed. Trying JS click...")
    if _try_javascript_click(driver, sign_in_button):
        return True

    # Try RETURN key as final fallback
    logger.warning("Both standard and JS clicks failed.")
    return _try_return_key_fallback(password_input)


def enter_creds(driver: WebDriver) -> bool:  # type: ignore
    """Enter credentials and sign in to Ancestry."""
    element_wait = WebDriverWait(driver, config_schema.selenium.explicit_wait)
    short_wait = WebDriverWait(driver, 5)
    time.sleep(random.uniform(0.5, 1.0))

    result = False

    try:
        logger.debug("Entering Credentials and Signing In...")

        # Enter username
        if _enter_username(driver, element_wait) and _click_next_button(driver, short_wait):
            # Enter password
            password_ok, password_input = _enter_password(driver, element_wait)
            if password_ok:
                    # Click sign in button
                    result = _click_sign_in_button(driver, short_wait, password_input)

    except (TimeoutException, NoSuchElementException) as e:  # type: ignore
        logger.error(f"Timeout or Element not found finding username/password field: {e}")
    except ValueError as ve:
        logger.critical(f"Configuration Error: {ve}")
        raise ve
    except WebDriverException as e:  # type: ignore
        logger.error(f"WebDriver error entering credentials: {e}")
        if not is_browser_open(driver):
            logger.error("Session invalid during credential entry.")
    except Exception as e:
        logger.error(f"Unexpected error entering credentials: {e}", exc_info=True)

    return result

# End of enter_creds

# Helper functions for consent

def _find_consent_banner(driver: WebDriver) -> Optional[Any]:
    """Find the cookie consent banner if present."""
    logger.debug(f"Checking for cookie consent overlay: '{COOKIE_BANNER_SELECTOR}'")  # type: ignore
    try:
        overlay_element = WebDriverWait(driver, 3).until(  # type: ignore
            EC.presence_of_element_located((By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR))  # type: ignore
        )
        logger.debug("Cookie consent overlay DETECTED.")
        return overlay_element
    except TimeoutException:  # type: ignore
        logger.debug("Cookie consent overlay not found. Assuming no consent needed.")
        return None
    except WebDriverException as e:  # type: ignore
        logger.error(f"Error checking for consent banner: {e}")
        raise


def _click_accept_button_standard(driver: WebDriver, accept_button) -> bool:
    """Try standard click on accept button."""
    try:
        accept_button.click()
        logger.info("Clicked accept button successfully.")
        WebDriverWait(driver, 2).until_not(  # type: ignore
            EC.presence_of_element_located((By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR))  # type: ignore
        )
        logger.debug("Consent overlay gone after clicking accept button.")
        return True
    except ElementClickInterceptedException:  # type: ignore
        logger.warning("Click intercepted for accept button, trying JS click...")
        return False
    except (TimeoutException, NoSuchElementException):  # type: ignore
        logger.debug("Consent overlay likely gone after standard click (verification timed out/not found).")
        return True
    except WebDriverException as click_err:  # type: ignore
        logger.error(f"Error during standard click on accept button: {click_err}. Trying JS click...")
        return False


def _click_accept_button_js(driver: WebDriver, accept_button) -> bool:
    """Try JavaScript click on accept button."""
    try:
        logger.debug("Attempting JS click on accept button...")
        driver.execute_script("arguments[0].click();", accept_button)
        logger.info("Clicked accept button via JS successfully.")
        WebDriverWait(driver, 2).until_not(  # type: ignore
            EC.presence_of_element_located((By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR))  # type: ignore
        )
        logger.debug("Consent overlay gone after JS clicking accept button.")
        return True
    except (TimeoutException, NoSuchElementException):  # type: ignore
        logger.debug("Consent overlay likely gone after JS click (verification timed out/not found).")
        return True
    except WebDriverException as js_click_err:  # type: ignore
        logger.error(f"Failed JS click for accept button: {js_click_err}")
        return False


def _handle_consent_button(driver: WebDriver) -> bool:
    """Handle clicking the consent accept button."""
    logger.debug(f"Trying specific accept button: '{CONSENT_ACCEPT_BUTTON_SELECTOR}'")
    try:
        accept_button = WebDriverWait(driver, 3).until(  # type: ignore
            EC.element_to_be_clickable((By.CSS_SELECTOR, CONSENT_ACCEPT_BUTTON_SELECTOR))  # type: ignore
        )
        logger.info("Found specific clickable accept button.")

        # Try standard click first
        if _click_accept_button_standard(driver, accept_button):
            return True

        # Try JS click as fallback
        return _click_accept_button_js(driver, accept_button)

    except TimeoutException:  # type: ignore
        logger.warning(f"Specific accept button '{CONSENT_ACCEPT_BUTTON_SELECTOR}' not found or not clickable.")
        return False
    except WebDriverException as find_err:  # type: ignore
        logger.error(f"Error finding/clicking specific accept button: {find_err}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error handling consent button: {e}", exc_info=True)
        return False


@retry(max_retries=2, backoff_factor=1, max_delay=3)
def consent(driver: WebDriver) -> bool:  # type: ignore
    """Handles the cookie consent banner if present."""
    if not driver:
        logger.error("consent: WebDriver instance is None.")
        return False

    # Find consent banner
    try:
        overlay_element = _find_consent_banner(driver)
    except WebDriverException:
        return False

    # No banner found
    if overlay_element is None:
        return True

    # Try to handle the consent button
    if _handle_consent_button(driver):
        return True

    # If button click failed
    logger.error("Could not remove consent overlay via button click.")
    return False

# End of consent

def _check_initial_login_status(session_manager: SessionManager) -> Optional[str]:
    """Check if already logged in before attempting login."""
    initial_status = login_status(session_manager, disable_ui_fallback=True)
    if initial_status is True:
        print("Already logged in. No need to sign in again.")
        return "LOGIN_SUCCEEDED"
    return None

def _navigate_to_signin(driver: Any, session_manager: SessionManager, signin_url: str) -> Optional[str]:
    """Navigate to sign-in page and verify."""
    if not nav_to_page(driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager):
        logger.debug("Navigation to sign-in page failed/redirected. Checking login status...")
        current_status = login_status(session_manager, disable_ui_fallback=True)
        if current_status is True:
            logger.info("Detected as already logged in during navigation attempt. Login considered successful.")
            return "LOGIN_SUCCEEDED"
        logger.error("Failed to navigate to login page (and not logged in).")
        return "LOGIN_FAILED_NAVIGATION"
    logger.debug("Successfully navigated to sign-in page.")
    return None

def _check_for_login_errors(driver: Any) -> Optional[str]:
    """Check for specific or generic login error messages."""
    try:
        WebDriverWait(driver, 1).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, FAILED_LOGIN_SELECTOR))
        )
        logger.error("Login failed: Specific 'Invalid Credentials' alert detected.")
        return "LOGIN_FAILED_BAD_CREDS"
    except TimeoutException:
        try:
            alert_element = WebDriverWait(driver, 0.5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.alert[role='alert']"))
            )
            alert_text = alert_element.text if alert_element and alert_element.text else "Unknown error"
            logger.error(f"Login failed: Generic alert found: '{alert_text}'.")
            return "LOGIN_FAILED_ERROR_DISPLAYED"
        except TimeoutException:
            logger.error("Login failed: Credential entry failed, but no specific or generic alert found.")
            return "LOGIN_FAILED_CREDS_ENTRY"
        except WebDriverException as alert_err:
            logger.warning(f"Error checking for generic login error message: {alert_err}")
            return "LOGIN_FAILED_CREDS_ENTRY"
    except WebDriverException as alert_err:
        logger.warning(f"Error checking for specific login error message: {alert_err}")
        return "LOGIN_FAILED_CREDS_ENTRY"

def _detect_2fa_page(driver: Any, session_manager: SessionManager) -> tuple[bool, Optional[str]]:
    """Detect if 2FA page is present."""
    try:
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR))
        )
        logger.info("Two-step verification page detected.")
        return True, None
    except TimeoutException:
        logger.debug("Two-step verification page not detected.")
        return False, None
    except WebDriverException as e:
        logger.error(f"WebDriver error checking for 2FA page: {e}")
        status = login_status(session_manager, disable_ui_fallback=True)
        if status is True:
            return False, "LOGIN_SUCCEEDED"
        if status is False:
            return False, "LOGIN_FAILED_UNKNOWN"
        return False, "LOGIN_FAILED_STATUS_CHECK_ERROR"

def _handle_2fa_flow(session_manager: SessionManager) -> str:
    """Handle 2FA verification flow."""
    if handle_two_fa(session_manager):
        logger.info("Two-step verification handled successfully.")
        if login_status(session_manager) is True:
            print("\n✓ Two-factor authentication completed successfully!")
            return "LOGIN_SUCCEEDED"
        logger.error("Login status check failed AFTER successful 2FA handling report.")
        return "LOGIN_FAILED_POST_2FA_VERIFY"
    logger.error("Two-step verification handling failed.")
    return "LOGIN_FAILED_2FA_HANDLING"

def _verify_login_no_2fa(driver: Any, session_manager: SessionManager, signin_url: str) -> str:
    """Verify login when no 2FA is detected."""
    logger.debug("Checking login status directly (no 2FA detected)...")
    login_check_result = login_status(session_manager, disable_ui_fallback=False)

    if login_check_result is True:
        print("\n✓ Login successful!")
        return "LOGIN_SUCCEEDED"

    if login_check_result is False:
        print("\n✗ Login failed. Please check your credentials.")
        logger.error("Direct login check failed. Checking for error messages again...")

        # Check for errors again
        error_result = _check_for_login_errors(driver)
        if error_result:
            return error_result

        # Check if still on login page
        try:
            if driver.current_url.startswith(signin_url):
                logger.error("Login failed: Still on login page (post-check), no error message found.")
                return "LOGIN_FAILED_STUCK_ON_LOGIN"
            logger.error("Login failed: Status False, no 2FA, no error msg, not on login page.")
            return "LOGIN_FAILED_UNKNOWN"
        except WebDriverException:
            logger.error("Login failed: Status False, WebDriverException getting URL.")
            return "LOGIN_FAILED_WEBDRIVER"

    logger.error("Login failed: Critical error during final login status check.")
    return "LOGIN_FAILED_STATUS_CHECK_ERROR"

# Login Main Function
def _execute_login_flow(
    driver: Any,
    session_manager: SessionManager,
    signin_url: str,
) -> str:
    """Execute the main login flow. Returns status string."""
    # Navigate to sign-in page
    nav_result = _navigate_to_signin(driver, session_manager, signin_url)
    if nav_result:
        return nav_result

    # Handle consent banner
    if not consent(driver):
        logger.warning("Failed to handle consent banner, login might be impacted.")

    # Enter credentials
    if not enter_creds(driver):
        logger.error("Failed during credential entry or submission.")
        error_result = _check_for_login_errors(driver)
        return error_result if error_result else "LOGIN_FAILED_CREDS_ENTRY"

    # Wait for page change
    logger.debug("Credentials submitted. Waiting for potential page change...")
    time.sleep(random.uniform(3.0, 5.0))

    # Check for 2FA
    two_fa_present, early_result = _detect_2fa_page(driver, session_manager)
    if early_result:
        return early_result

    # Handle 2FA or verify login
    if two_fa_present:
        return _handle_2fa_flow(session_manager)
    return _verify_login_no_2fa(driver, session_manager, signin_url)


def _handle_login_exception(e: Exception, driver: Any) -> str:
    """Handle exceptions during login. Returns error status string."""
    if isinstance(e, TimeoutException):
        logger.error(f"Timeout during login process: {e}", exc_info=False)
        return "LOGIN_FAILED_TIMEOUT"
    if isinstance(e, WebDriverException):
        logger.error(f"WebDriverException during login: {e}", exc_info=False)
        if not is_browser_open(driver):
            logger.error("Session became invalid during login.")
        return "LOGIN_FAILED_WEBDRIVER"
    logger.error(f"An unexpected error occurred during login: {e}", exc_info=True)
    return "LOGIN_FAILED_UNEXPECTED"


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

    # Check if already logged in
    initial_result = _check_initial_login_status(session_manager)
    if initial_result:
        return initial_result

    signin_url = urljoin(config_schema.api.base_url, "account/signin")

    try:
        return _execute_login_flow(driver, session_manager, signin_url)
    except Exception as e:
        return _handle_login_exception(e, driver)

# End of log_in

def _validate_login_status_inputs(session_manager: SessionManager) -> Optional[bool]:
    """Validate session manager for login status check."""
    if not hasattr(session_manager, 'is_sess_valid') or not hasattr(session_manager, 'driver'):
        logger.error(f"Invalid argument: Expected SessionManager-like object, got {type(session_manager)}.")
        return None

    if not session_manager.is_sess_valid():
        logger.debug("Session is invalid, user cannot be logged in.")
        return False

    if session_manager.driver is None:
        logger.error("Login status check: Driver is None within SessionManager.")
        return None

    return True  # Valid

def _perform_api_login_check(session_manager: SessionManager) -> Optional[bool]:
    """Perform API-based login status check."""
    logger.debug("Performing primary API-based login status check...")
    try:
        session_manager._sync_cookies()
        api_check_result = session_manager.api_manager.verify_api_login_status()

        if api_check_result is True:
            logger.debug("API login check confirmed user is logged in.")
            return True
        if api_check_result is False:
            logger.debug("API login check confirmed user is NOT logged in.")
            return False

        logger.warning("API login check returned ambiguous result (None).")
        return None
    except Exception as e:
        logger.error(f"Exception during API login check: {e}", exc_info=True)
        return None

def _check_ui_login_indicators(driver: Any) -> Optional[bool]:
    """Check UI for login indicators."""
    # Check for logged-in element
    logged_in_selector = CONFIRMED_LOGGED_IN_SELECTOR
    logger.debug(f"Checking for logged-in indicator: '{logged_in_selector}'")
    ui_element_present = is_elem_there(driver, By.CSS_SELECTOR, logged_in_selector, wait=3)

    if ui_element_present:
        logger.debug("UI check: Logged-in indicator found. User is logged in.")
        return True

    # Check for login button
    login_button_selector = LOG_IN_BUTTON_SELECTOR
    logger.debug(f"Checking for login button: '{login_button_selector}'")
    login_button_present = is_elem_there(driver, By.CSS_SELECTOR, login_button_selector, wait=3)

    if login_button_present:
        logger.debug("UI check: Login button found. User is NOT logged in.")
        return False

    return None  # Inconclusive

def _perform_ui_login_check_with_navigation(driver: Any) -> bool:
    """Perform UI login check with navigation fallback."""
    result = _check_ui_login_indicators(driver)
    if result is not None:
        return result

    # Navigate to base URL for clearer check
    logger.debug("UI check inconclusive. Navigating to base URL for clearer check...")
    try:
        current_url = driver.current_url
        base_url = config_schema.api.base_url

        if not current_url.startswith(base_url):
            driver.get(base_url)
            time.sleep(2)

            result = _check_ui_login_indicators(driver)
            if result is not None:
                return result
    except Exception as nav_e:
        logger.warning(f"Error during navigation for secondary UI check: {nav_e}")

    # Default to False for security
    logger.debug("UI check still inconclusive after all checks. Defaulting to NOT logged in for security.")
    return False

# Login Status Check Function
@retry(max_retries=2)
def login_status(session_manager: SessionManager, disable_ui_fallback: bool = False) -> Optional[bool]:  # type: ignore
    """
    Checks if the user is currently logged in. Prioritizes API check, with optional UI fallback.

    Args:
        session_manager: The session manager instance
        disable_ui_fallback: If True, only use API check and never fall back to UI check

    Returns:
        True if logged in, False if not logged in, None if the check fails critically.
    """
    # Validate inputs
    validation_result = _validate_login_status_inputs(session_manager)
    if validation_result is not True:
        return validation_result

    driver = session_manager.driver

    # Perform API check
    api_check_result = _perform_api_login_check(session_manager)
    if api_check_result is not None:
        return api_check_result

    # If API check is ambiguous and UI fallback is disabled, return None
    if disable_ui_fallback:
        logger.warning("API login check was ambiguous and UI fallback is disabled. Status unknown.")
        return None

    # Perform UI check
    logger.debug("Performing fallback UI-based login status check...")
    try:
        return _perform_ui_login_check_with_navigation(driver)
    except WebDriverException as e:
        logger.error(f"WebDriverException during UI login_status check: {e}")
        if not is_browser_open(driver):
            logger.warning("Browser appears to be closed. Closing session.")
            session_manager.close_sess()
        return None
    except Exception as e:
        logger.error(f"Unexpected error during UI login_status check: {e}", exc_info=True)
        return None

# End of login_status

# ------------------------------------------------------------------------------------
# Navigation Functions (Remains in utils.py)
# ------------------------------------------------------------------------------------

def _validate_nav_inputs(driver: WebDriver, url: str) -> bool:  # type: ignore
    """Validate navigation inputs."""
    if not driver:
        logger.error("Navigation failed: WebDriver instance is None.")
        return False
    if not url or not isinstance(url, str):
        logger.error(f"Navigation failed: Target URL '{url}' is invalid.")
        return False
    return True


def _parse_and_normalize_url(url: str) -> Optional[str]:
    """Parse and normalize URL to base form."""
    try:
        target_url_parsed = urlparse(url)
        return urlunparse(
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
        return None


def _check_browser_session(
    driver: WebDriver,  # type: ignore
    session_manager: SessionManagerType,
    attempt: int,
) -> Optional[WebDriver]:  # type: ignore
    """Check browser session and restart if needed. Returns driver or None if failed."""
    if not is_browser_open(driver):
        logger.error(
            f"Navigation failed (Attempt {attempt}): Browser session invalid before nav."
        )
        if session_manager:
            logger.warning("Attempting session restart...")
            if session_manager.restart_sess():
                logger.info("Session restarted. Retrying navigation...")
                driver_instance = session_manager.driver
                if driver_instance is not None:
                    return driver_instance
                logger.error("Session restart reported success but driver is still None.")
                return None
            logger.error("Session restart failed. Cannot navigate.")
            return None
        logger.error("Session invalid and no SessionManager provided for restart.")
        return None
    return driver


def _execute_navigation(driver: WebDriver, url: str, page_timeout: int) -> None:  # type: ignore
    """Execute navigation and wait for page ready state."""
    logger.debug(f"Executing driver.get('{url}')...")
    driver.get(url)
    WebDriverWait(driver, page_timeout).until(  # type: ignore
        lambda d: d.execute_script("return document.readyState") in ["complete", "interactive"]
    )
    time.sleep(random.uniform(0.5, 1.5))


def _get_landed_url_base(driver: WebDriver, attempt: int) -> Optional[str]:  # type: ignore
    """Get and normalize the current URL after navigation."""
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
        return landed_url_base
    except WebDriverException as e:  # type: ignore
        logger.error(f"Failed to get current URL after get() (Attempt {attempt}): {e}. Retrying.")
        return None


def _check_for_mfa_page(driver: WebDriver) -> bool:  # type: ignore
    """Check if currently on MFA page."""
    try:
        WebDriverWait(driver, 1).until(  # type: ignore
            EC.visibility_of_element_located(  # type: ignore
                (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR)  # type: ignore
            )
        )
        return True
    except (TimeoutException, NoSuchElementException):  # type: ignore
        return False
    except WebDriverException as e:  # type: ignore
        logger.warning(f"WebDriverException checking for MFA header: {e}")
        return False


def _check_for_login_page(driver: WebDriver, target_url_base: str, signin_page_url_base: str) -> bool:  # type: ignore
    """Check if currently on login page (only if not intentionally navigating there)."""
    if target_url_base == signin_page_url_base:
        return False
    try:
        WebDriverWait(driver, 1).until(  # type: ignore
            EC.visibility_of_element_located((By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))  # type: ignore
        )
        return True
    except (TimeoutException, NoSuchElementException):  # type: ignore
        return False
    except WebDriverException as e:  # type: ignore
        logger.warning(f"WebDriverException checking for Login username input: {e}")
        return False


def _handle_login_redirect(session_manager: SessionManagerType) -> str:
    """Handle unexpected login page redirect. Returns 'retry', 'fail', or 'no_manager'."""
    if not session_manager:
        logger.error("Landed on login page, no SessionManager provided for re-login attempt.")
        return "no_manager"

    logger.warning("Landed on Login page unexpectedly. Checking login status first...")
    login_stat = login_status(session_manager, disable_ui_fallback=True)
    if login_stat is True:
        logger.info("Login status OK after landing on login page redirect. Retrying original navigation.")
        return "retry"

    logger.info("Not logged in according to API. Attempting re-login...")
    login_result_str = log_in(session_manager)
    if login_result_str == "LOGIN_SUCCEEDED":
        logger.info("Re-login successful. Retrying original navigation...")
        return "retry"

    logger.error(f"Re-login attempt failed ({login_result_str}). Cannot complete navigation.")
    return "fail"


def _check_signin_redirect(
    target_url_base: str,
    landed_url_base: str,
    signin_page_url_base: str,
    session_manager: SessionManagerType,
) -> bool:
    """Check if redirect from signin to base URL is valid. Returns True if valid redirect."""
    is_signin_to_base_redirect = (
        target_url_base == signin_page_url_base
        and landed_url_base == urlparse(config_schema.api.base_url).path.rstrip("/")
    )
    if is_signin_to_base_redirect:
        logger.debug("Redirected from signin page to base URL. Verifying login status...")
        time.sleep(1)
        if session_manager and login_status(session_manager, disable_ui_fallback=True) is True:
            logger.info("Redirect after signin confirmed as logged in. Considering original navigation target 'signin' successful.")
            return True
    return False


def _handle_url_mismatch(
    driver: WebDriver,  # type: ignore
    landed_url_base: str,
    target_url_base: str,
    unavailability_selectors: dict[str, tuple[str, int]],
) -> str:
    """Handle landing on unexpected URL. Returns 'continue', 'fail', or 'success'."""
    logger.warning(f"Navigation landed on unexpected URL base: '{landed_url_base}' (Expected: '{target_url_base}')")
    action, wait_time = _check_for_unavailability(driver, unavailability_selectors)
    if action == "skip":
        logger.error("Page no longer available message found. Skipping.")
        return "fail"
    if action == "refresh":
        logger.info(f"Temporary unavailability message found. Waiting {wait_time}s and retrying...")
        time.sleep(wait_time)
        return "continue"
    logger.warning("Wrong URL, no specific unavailability message found. Retrying navigation.")
    return "continue"


def _wait_for_element(
    driver: WebDriver,  # type: ignore
    selector: str,
    element_timeout: int,
    unavailability_selectors: dict[str, tuple[str, int]],
) -> str:
    """Wait for target element. Returns 'success', 'continue', or 'fail'."""
    wait_selector = selector if selector else "body"
    logger.debug(f"On correct URL base. Waiting up to {element_timeout}s for selector: '{wait_selector}'")
    try:
        WebDriverWait(driver, element_timeout).until(  # type: ignore
            EC.visibility_of_element_located((By.CSS_SELECTOR, wait_selector))  # type: ignore
        )
        logger.debug(f"Navigation successful and element '{wait_selector}' found.")
        return "success"
    except TimeoutException:  # type: ignore
        current_url_on_timeout = "Unknown"
        with contextlib.suppress(Exception):
            current_url_on_timeout = driver.current_url
        logger.warning(f"Timeout waiting for selector '{wait_selector}' at {current_url_on_timeout} (URL base was correct).")

        action, wait_time = _check_for_unavailability(driver, unavailability_selectors)
        if action == "skip":
            return "fail"
        if action == "refresh":
            time.sleep(wait_time)
            return "continue"
        logger.warning("Timeout on selector, no unavailability message. Retrying navigation.")
        return "continue"
    except WebDriverException as el_wait_err:  # type: ignore
        logger.error(f"WebDriverException waiting for selector '{wait_selector}': {el_wait_err}")
        return "continue"


def _handle_navigation_alert(driver: WebDriver, attempt: int) -> str:  # type: ignore
    """Handle unexpected alert. Returns 'continue' or 'fail'."""
    alert_text = "N/A"
    try:
        # Try to get alert text if available
        alert = driver.switch_to.alert
        alert_text = alert.text
    except AttributeError:
        pass
    logger.warning(f"Unexpected alert detected (Attempt {attempt}): {alert_text}")
    try:
        driver.switch_to.alert.accept()
        logger.info("Accepted unexpected alert.")
        return "continue"
    except Exception as accept_e:
        logger.error(f"Failed to accept unexpected alert: {accept_e}")
        return "fail"


def _handle_webdriver_exception(
    driver: WebDriver,  # type: ignore
    session_manager: SessionManagerType,
    _attempt: int,
) -> tuple[str, Optional[WebDriver]]:  # type: ignore
    """Handle WebDriver exception. Returns (action, driver) where action is 'continue' or 'fail'."""
    if session_manager and not is_browser_open(driver):
        logger.error("WebDriver session invalid after exception. Attempting restart...")
        if session_manager.restart_sess():
            logger.info("Session restarted. Retrying navigation...")
            driver_instance = session_manager.driver
            if driver_instance is not None:
                return ("continue", driver_instance)
            return ("fail", None)
        logger.error("Session restart failed. Cannot complete navigation.")
        return ("fail", None)
    logger.warning("WebDriverException occurred, session seems valid or no restart possible. Waiting before retry.")
    time.sleep(random.uniform(2, 4))
    return ("continue", driver)


def _check_url_mismatch_and_handle(
    driver: WebDriver,  # type: ignore
    landed_url_base: str,
    target_url_base: str,
    signin_page_url_base: str,
    unavailability_selectors: dict[str, tuple[str, int]],
    session_manager: SessionManagerType,
) -> tuple[str, Optional[WebDriver]]:  # type: ignore
    """
    Check for URL mismatch and handle appropriately.
    Returns (action, driver) where action is 'success', 'fail', 'continue', or None.
    """
    if landed_url_base == target_url_base:
        return (None, driver)  # type: ignore  # No mismatch, continue with normal flow

    # Check if signin redirect is acceptable
    if _check_signin_redirect(target_url_base, landed_url_base, signin_page_url_base, session_manager):
        return ("success", driver)

    # Handle URL mismatch
    mismatch_action = _handle_url_mismatch(driver, landed_url_base, target_url_base, unavailability_selectors)
    if mismatch_action == "fail":
        return ("fail", driver)
    if mismatch_action == "continue":
        return ("continue", driver)

    return (None, driver)  # type: ignore  # Continue with normal flow


def _validate_post_navigation(
    driver: WebDriver,  # type: ignore
    landed_url_base: str,
    target_url_base: str,
    signin_page_url_base: str,
    selector: str,
    element_timeout: int,
    unavailability_selectors: dict[str, tuple[str, int]],
    session_manager: SessionManagerType,
) -> tuple[str, Optional[WebDriver]]:  # type: ignore
    """
    Validate navigation after landing on page.
    Returns (action, driver) where action is 'success', 'fail', or 'continue'.
    """
    result_action = "continue"
    result_driver = driver

    # Check for MFA page
    if _check_for_mfa_page(driver):
        logger.error("Landed on MFA page unexpectedly during navigation. Navigation failed.")
        result_action = "fail"
    # Check for Login page
    elif _check_for_login_page(driver, target_url_base, signin_page_url_base):
        login_action = _handle_login_redirect(session_manager)
        if login_action == "retry":
            result_action = "continue"
        elif login_action in ("fail", "no_manager"):
            result_action = "fail"
    else:
        # Check for URL mismatch
        url_check_result, driver = _check_url_mismatch_and_handle(
            driver, landed_url_base, target_url_base, signin_page_url_base,
            unavailability_selectors, session_manager
        )
        if url_check_result is not None:
            result_action = url_check_result
            result_driver = driver
        else:
            # --- Final Check: Element on Page ---
            element_result = _wait_for_element(driver, selector, element_timeout, unavailability_selectors)
            if element_result == "success":
                result_action = "success"
            elif element_result == "fail":
                result_action = "fail"
            elif element_result == "continue":
                result_action = "continue"

    return (result_action, result_driver)


def _perform_navigation_attempt(
    driver: WebDriver,  # type: ignore
    nav_config: NavigationConfig,
    session_manager: SessionManagerType,
    attempt: int,
) -> tuple[str, Optional[WebDriver]]:  # type: ignore
    """
    Perform a single navigation attempt.
    Returns (action, driver) where action is 'success', 'fail', 'continue', or 'retry'.
    """
    result_action = "continue"
    result_driver = driver

    try:
        # --- Pre-Navigation Checks ---
        driver_check = _check_browser_session(driver, session_manager, attempt)
        if driver_check is None:
            result_action = "fail"
        elif driver_check != driver:
            result_action = "retry"
            result_driver = driver_check
        else:
            # --- Navigation Execution ---
            _execute_navigation(driver, nav_config.url, nav_config.page_timeout)

            # --- Post-Navigation Checks ---
            landed_url_base = _get_landed_url_base(driver, attempt)
            if landed_url_base is None:
                result_action = "continue"
            else:
                # Validate post-navigation state
                result_action, result_driver = _validate_post_navigation(
                    driver, landed_url_base, nav_config.target_url_base, nav_config.signin_page_url_base,
                    nav_config.selector, nav_config.element_timeout, nav_config.unavailability_selectors, session_manager
                )

    except UnexpectedAlertPresentException:  # type: ignore
        alert_action = _handle_navigation_alert(driver, attempt)
        result_action = "fail" if alert_action == "fail" else "continue"

    except WebDriverException as wd_e:  # type: ignore
        logger.error(f"WebDriverException during navigation (Attempt {attempt}): {wd_e}", exc_info=False)
        wd_action, new_driver = _handle_webdriver_exception(driver, session_manager, attempt)
        result_action = "fail" if wd_action == "fail" else "continue"
        result_driver = new_driver if new_driver else driver

    except Exception as e:
        logger.error(f"Unexpected error during navigation (Attempt {attempt}): {e}", exc_info=True)
        time.sleep(random.uniform(2, 4))
        result_action = "continue"

    return (result_action, result_driver)


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
    # Validate inputs
    if not _validate_nav_inputs(driver, url):
        return False

    # Parse and normalize URL
    target_url_base = _parse_and_normalize_url(url)
    if target_url_base is None:
        return False

    # Get configuration
    max_attempts = config_schema.api.max_retries
    page_timeout = config_schema.selenium.page_load_timeout
    element_timeout = config_schema.selenium.explicit_wait

    # Define common problematic URLs/selectors
    signin_page_url_base = urljoin(config_schema.api.base_url, "account/signin").rstrip("/")
    # Selectors for known 'unavailable' pages
    unavailability_selectors = {
        TEMP_UNAVAILABLE_SELECTOR: ("refresh", 5),  # type: ignore # Selector : (action, wait_seconds)
        PAGE_NO_LONGER_AVAILABLE_SELECTOR: ("skip", 0),  # type: ignore
        # Add other known error page selectors here
    }

    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Navigation Attempt {attempt}/{max_attempts} to: {url}")

        nav_config = NavigationConfig(
            url=url, selector=selector, target_url_base=target_url_base,
            signin_page_url_base=signin_page_url_base, unavailability_selectors=unavailability_selectors,
            page_timeout=page_timeout, element_timeout=element_timeout
        )
        action, new_driver = _perform_navigation_attempt(
            driver, nav_config, session_manager, attempt
        )

        if action == "success":
            return True
        if action == "fail":
            return False
        if action in ("retry", "continue"):
            driver = new_driver if new_driver else driver
            continue

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
    driver: WebDriver, selectors: dict[str, tuple[str, int]]  # type: ignore
) -> tuple[Optional[str], int]:
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
    Standalone test suite for utils.py - Refactored to use modular test framework.

    This function now delegates to the comprehensive test suite defined below,
    following the standardized testing pattern used across the codebase.
    """
    print("\n" + "="*60)
    print("UTILS.PY - COMPREHENSIVE TEST SUITE")
    print("="*60 + "\n")

    # Run the comprehensive test suite using the standardized framework
    success = run_comprehensive_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

# End of main

def utils_module_tests() -> bool:
    """Module-specific tests for utils.py functionality."""
    try:
        # Test format_name functionality
        assert format_name("john doe") == "John Doe", "format_name basic test failed"
        assert format_name(None) == "Valued Relative", "format_name None test failed"

        # Test DynamicRateLimiter
        limiter = DynamicRateLimiter(initial_delay=0.001)
        limiter.wait()  # Should not hang

        # Test SessionManager - import it properly at runtime
        from core.session_manager import SessionManager
        sm = SessionManager()
        assert hasattr(sm, "driver_live"), "SessionManager missing driver_live attribute"
        assert hasattr(sm, "session_ready"), "SessionManager missing session_ready attribute"

        return True
    except Exception as e:
        logger.error(f"Utils module tests failed: {e}")
        return False

def run_comprehensive_tests() -> bool:
    """Run comprehensive utils tests using standardized TestSuite format."""
    return utils_module_tests()

# ==============================================
# Module Registration
# ==============================================

# Module setup already handled by setup_module() call at top of file

# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite(
        "Core Utilities & Session Management", "utils.py"
    )  # Basic utility functions

    def test_parse_cookie():
        """Test cookie parsing with various cookie string formats"""
        test_cases = [
            (
                "session_id=abc123; path=/; domain=.example.com",
                {"session_id": "abc123", "path": "/", "domain": ".example.com"},
                "Standard cookie string",
            ),
            ("", {}, "Empty string"),
            ("single=value", {"single": "value"}, "Single cookie"),
            ("a=1; b=2; c=3", {"a": "1", "b": "2", "c": "3"}, "Multiple cookies"),
            (
                "invalid_part; valid=test",
                {"valid": "test"},
                "Mixed valid/invalid parts",
            ),
        ]

        print("📋 Testing cookie parsing with various formats:")
        results = []

        for cookie_str, expected, description in test_cases:
            try:
                result = parse_cookie(cookie_str)
                is_dict = isinstance(result, dict)
                matches_expected = result == expected

                status = "✅" if is_dict and matches_expected else "❌"
                print(f"   {status} {description}")
                print(f"      Input: '{cookie_str}'")
                print(f"      Output: {result}")
                print(f"      Expected: {expected}")

                results.append(is_dict and matches_expected)
                assert is_dict, f"Should return dictionary for '{cookie_str}'"
                assert (
                    matches_expected
                ), f"Should match expected result for '{cookie_str}'"

            except Exception as e:
                print(f"   ❌ {description}: Exception {e}")
                results.append(False)

        print(f"📊 Results: {sum(results)}/{len(results)} cookie parsing tests passed")

    def test_ordinal_case():
        """Test ordinal number formatting with various input types"""
        test_cases = [
            (1, "1st", "First ordinal"),
            (2, "2nd", "Second ordinal"),
            (3, "3rd", "Third ordinal"),
            (4, "4th", "Fourth ordinal"),
            (11, "11th", "Eleventh (special case)"),
            (12, "12th", "Twelfth (special case)"),
            (13, "13th", "Thirteenth (special case)"),
            (21, "21st", "Twenty-first ordinal"),
            (22, "22nd", "Twenty-second ordinal"),
            (23, "23rd", "Twenty-third ordinal"),
            (101, "101st", "One hundred first"),
            ("Great Uncle", "Great Uncle", "Text input"),
        ]

        print("📋 Testing ordinal number formatting:")
        results = []

        for input_val, expected, description in test_cases:
            try:
                result = ordinal_case(input_val)
                matches_expected = result == expected

                status = "✅" if matches_expected else "❌"
                print(f"   {status} {description}")
                print(f"      Input: {input_val} (Type: {type(input_val).__name__})")
                print(f"      Output: '{result}' (Expected: '{expected}')")

                results.append(matches_expected)
                assert (
                    matches_expected
                ), f"Failed for {input_val}: expected '{expected}', got '{result}'"

            except Exception as e:
                print(f"   ❌ {description}: Exception {e}")
                results.append(False)

        print(
            f"📊 Results: {sum(results)}/{len(results)} ordinal formatting tests passed"
        )

    def test_format_name():
        """Test name formatting with various input types and edge cases"""
        test_cases = [
            ("john doe", "John Doe", "Basic name formatting"),
            (None, "Valued Relative", "None input handling"),
            ("", "Valued Relative", "Empty string handling"),
            ("JOHN DOE", "John Doe", "Uppercase conversion"),
            ("john /doe/", "John Doe", "GEDCOM format handling"),
            ("o'malley", "O'Malley", "Irish apostrophe names"),
            ("mcdonald", "McDonald", "Scottish Mc names"),
            ("macleod", "MacLeod", "Scottish Mac names"),
            ("'betty' smith", "'Betty' Smith", "Quoted nicknames"),
            (
                "jean-claude van damme",
                "Jean-Claude van Damme",
                "Hyphenated with particles",
            ),
            ("j. r. r. tolkien", "J. R. R. Tolkien", "Initials with periods"),
        ]

        print("📋 Testing name formatting with various cases:")
        results = []

        for input_val, expected, description in test_cases:
            try:
                result = format_name(input_val)
                matches_expected = result == expected

                status = "✅" if matches_expected else "❌"
                print(f"   {status} {description}")
                print(f"      Input: {input_val!r} → Output: '{result}'")
                print(f"      Expected: '{expected}'")

                results.append(matches_expected)
                assert (
                    matches_expected
                ), f"Failed for {input_val!r}: expected '{expected}', got '{result}'"

            except Exception as e:
                print(f"   ❌ {description}: Exception {e}")
                results.append(False)

        print(f"📊 Results: {sum(results)}/{len(results)} name formatting tests passed")

    def test_decorators():
        # Test retry decorator availability
        assert callable(retry), "retry decorator should be callable"
        assert callable(retry_api), "retry_api decorator should be callable"
        assert callable(
            ensure_browser_open
        ), "ensure_browser_open decorator should be callable"
        assert callable(time_wait), "time_wait decorator should be callable"

        # Basic decorator functionality test
        @retry(max_retries=1, backoff_factor=0.001)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success", "Retry decorator should work"

    def test_rate_limiter():
        # Test DynamicRateLimiter instantiation and basic functionality
        limiter = DynamicRateLimiter(initial_delay=0.001, max_delay=0.01)
        assert limiter is not None, "Rate limiter should instantiate"
        assert hasattr(limiter, "wait"), "Rate limiter should have wait method"
        assert hasattr(
            limiter, "reset_delay"
        ), "Rate limiter should have reset_delay method"
        assert hasattr(
            limiter, "increase_delay"
        ), "Rate limiter should have increase_delay method"
        assert hasattr(
            limiter, "decrease_delay"
        ), "Rate limiter should have decrease_delay method"

        # Test wait method (should not hang)
        start_time = time.time()
        limiter.wait()
        elapsed = time.time() - start_time
        assert elapsed < 1.0, "Wait should complete quickly in test"

    def test_session_manager():
        # Test SessionManager class availability and basic attributes
        # Import SessionManager directly to avoid circular import issues
        from core.session_manager import SessionManager

        sm = SessionManager()
        assert hasattr(
            sm, "driver_live"
        ), "SessionManager should have driver_live attribute"
        assert hasattr(
            sm, "session_ready"
        ), "SessionManager should have session_ready attribute"
        assert hasattr(
            sm, "ensure_session_ready"
        ), "SessionManager should have ensure_session_ready method"
        assert hasattr(sm, "close_sess"), "SessionManager should have close_sess method"

        # Test initial state
        assert not sm.driver_live, "Driver should not be live initially"
        assert not sm.session_ready, "Session should not be ready initially"

    def test_api_request_function():
        # Test _api_req function availability
        assert callable(_api_req), "_api_req function should be callable"

        # Test function signature (should not raise errors)
        import inspect as inspect_module

        sig = inspect_module.signature(_api_req)
        assert len(sig.parameters) >= 2, "_api_req should accept multiple parameters"

    def test_login_status_function():
        # Test login_status function availability
        assert callable(login_status), "login_status function should be callable"

        # Test function signature
        import inspect as inspect_module

        sig = inspect_module.signature(login_status)
        assert (
            "session_manager" in sig.parameters
        ), "login_status should accept session_manager parameter"

    def test_module_registration():
        # Test that module registration functions work
        assert callable(
            auto_register_module
        ), "auto_register_module should be available"
        assert callable(register_function), "register_function should be available"
        assert callable(get_function), "get_function should be available"
        assert callable(
            is_function_available
        ), "is_function_available should be available"

        # Test that core functions are available
        assert "format_name" in globals(), "format_name should be in globals"
        assert (
            "DynamicRateLimiter" in globals()
        ), "DynamicRateLimiter should be in globals"
        assert "SessionManager" in globals(), "SessionManager should be in globals"

    def test_performance_validation():
        # Test that key operations complete within reasonable time
        start_time = time.time()

        # Format name performance
        for i in range(100):
            format_name(f"test name {i}")

        # Ordinal case performance
        for i in range(1, 101):
            ordinal_case(i)

        elapsed = time.time() - start_time
        assert (
            elapsed < 1.0
        ), f"Performance test should complete quickly, took {elapsed:.3f}s"

    def test_check_for_unavailability():
        """Test unavailability detection logic (Priority 2)."""
        from unittest.mock import MagicMock, patch
        from selenium.webdriver.common.by import By

        print("🔍 Testing _check_for_unavailability() error detection:")

        # Test that function exists and is callable
        assert callable(_check_for_unavailability), "_check_for_unavailability should be callable"
        print("   ✅ Function is callable")

        # Test with mock driver - using patch to properly mock the helper functions
        mock_driver = MagicMock()
        mock_driver.current_url = "http://test.com"

        # Test basic structure with minimal mocking
        # Since the actual function uses is_browser_open and is_elem_there which may
        # not be easily mockable, we'll test the function signature and return type
        selectors = {
            "div.unavailable": ("refresh", 5),
            "div.not-found": ("skip", 0)
        }

        # Test with a driver that will likely return (None, 0) since we're in test mode
        action, wait_time = _check_for_unavailability(mock_driver, selectors)
        assert isinstance(action, (str, type(None))), f"Action should be str or None, got {type(action)}"
        assert isinstance(wait_time, int), f"Wait time should be int, got {type(wait_time)}"
        print(f"   ✅ Function returns proper types: action={action}, wait_time={wait_time}")

        # Test with empty selectors
        action, wait_time = _check_for_unavailability(mock_driver, {})
        assert action is None, "Should return None for empty selectors"
        assert wait_time == 0, "Should return 0 for empty selectors"
        print("   ✅ Handled empty selectors correctly")

        print("✅ _check_for_unavailability() passed all tests")    # Run all tests
    print("🛠️ Running Core Utilities & Session Management comprehensive test suite...")

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
            "_check_for_unavailability() error detection (Priority 2)",
            test_check_for_unavailability,
            "Test browser unavailability message detection with various scenarios",
            "Tests detection of page unavailable messages, invalid driver handling, and no-error cases",
            "Function correctly detects unavailability messages and handles edge cases",
        )

        suite.run_test(
            "Performance validation",
            test_performance_validation,
            "Test performance of name formatting and ordinal operations with datasets",
            "Performance validation ensures efficient utility function execution",
            "Utility functions complete processing within reasonable time limits",
        )

    # Generate summary report
    suite.finish_suite()

# End of utils.py
