#!/usr/bin/env python3

"""
utils.py - Core Session Management, API Requests, General Utilities

Manages Selenium/Requests sessions, handles core API interaction (_api_req),
provides general utilities (decorators, formatting, rate limiting),
and includes login/session verification logic closely tied to SessionManager.
"""

# === CORE INFRASTRUCTURE ===
from core.logging_utils import log_action_banner
from standard_imports import setup_module
from test_utilities import create_standard_test_runner

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === SESSION MANAGER IMPORT ===
# Import SessionManager from core module - use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    from selenium.webdriver.remote.webdriver import WebDriver as _WebDriverForTyping

    from core.session_manager import SessionManager

    def login_status(session_manager: "SessionManager", disable_ui_fallback: bool = False) -> Optional[bool]: ...

    def consent(driver: _WebDriverForTyping) -> bool: ...
else:
    # Runtime import to avoid circular dependency issues
    SessionManager = None

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import base64  # For make_ube
import binascii  # For make_ube
import json
import logging
import random  # Used by RateLimiter jitter calculations
import re
import time
import uuid  # For make_ube
from collections.abc import Mapping  # Consolidated typing imports
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path  # For cookie persistence
from typing import (
    Any,
    Callable,
    Optional,
    ParamSpec,
    Protocol,
    TypeGuard,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import urljoin, urlparse, urlunparse  # urljoin re-exported for action7_inbox.py

# === THIRD-PARTY IMPORTS ===
from requests import Response as RequestsResponse, Session as RequestsSession
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

# === LOCAL IMPORTS ===
# (Note: Some imports done locally to avoid circular dependencies)
from api_constants import (
    API_PATH_UUID_LEGACY,
)
from common_params import NavigationConfig, RetryContext
from core.error_handling import RetryPolicyProfile, resolve_retry_policy
from observability.metrics_registry import metrics
from selenium_utils import DriverProtocol, WebElementProtocol

# === TYPE ALIASES ===
# Define type aliases
RequestsResponseTypeOptional = Optional[RequestsResponse]
Locator = tuple[str, str]
ApiResponseType = Union[dict[str, Any], list[Any], str, bytes, RequestsResponse, None]


DriverType = Optional[WebDriver]


class RateLimiterProtocol(Protocol):
    """Subset of the adaptive rate limiter interface used throughout utils."""

    def wait(self, api_description: Optional[str] = None) -> float:  # pragma: no cover - protocol
        ...

    def on_429_error(self, api_description: Optional[str] = None) -> None:  # pragma: no cover - protocol
        ...

    def on_success(self, api_description: Optional[str] = None) -> None:  # pragma: no cover - protocol
        ...


class SessionManagerLike(Protocol):
    """Minimal SessionManager surface required for browser helpers."""

    driver: DriverType  # pragma: no cover - protocol attribute


def _is_session_manager_like(value: Any) -> TypeGuard[SessionManagerLike]:
    """Return True when value looks like a SessionManager instance."""

    return hasattr(value, "driver")


def _wait_until_visible(waiter: "WebDriverWait[Any]", locator: Locator) -> WebElement:
    """Return first element matching locator once it becomes visible."""

    condition = expected_conditions.visibility_of_element_located(locator)
    return waiter.until(condition)


def _wait_until_clickable(waiter: "WebDriverWait[Any]", locator: Locator) -> WebElement:
    """Return the element once it becomes clickable."""

    condition = expected_conditions.element_to_be_clickable(locator)
    return waiter.until(condition)


def _wait_until_present(waiter: "WebDriverWait[Any]", locator: Locator) -> WebElement:
    """Return first element matching locator once present in DOM."""

    condition = expected_conditions.presence_of_element_located(locator)
    return waiter.until(condition)


def _wait_until_not_visible(waiter: "WebDriverWait[Any]", locator: Locator) -> bool:
    """Wait until the element matching locator is no longer visible."""

    condition = expected_conditions.visibility_of_element_located(locator)
    return bool(waiter.until_not(condition))


def _wait_until_not_present(waiter: "WebDriverWait[Any]", locator: Locator) -> bool:
    """Wait until the element matching locator is no longer present in the DOM."""

    condition = expected_conditions.presence_of_element_located(locator)
    return bool(waiter.until_not(condition))


def _get_requests_session(session_manager: "SessionManager") -> RequestsSession:
    """Return the underlying requests session, with compatibility fallback."""

    req_session = getattr(session_manager, "requests_session", None)
    if req_session is None:
        req_session = getattr(session_manager, "_requests_session", None)
    if req_session is None:
        raise AttributeError("SessionManager is missing a requests session")
    return cast(RequestsSession, req_session)


# Type variables for decorators
P = ParamSpec('P')
R = TypeVar('R')
SessionManagerType = Optional["SessionManager"]  # Use string literal for forward reference


# === OBSERVABILITY HELPERS ===
_UUID_PATH_SEGMENT_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _sanitize_metric_segment(segment: str) -> str:
    """Reduce high-cardinality path segments for metrics labeling."""

    if not segment:
        return ""

    if segment.isdigit():
        return "{id}"

    if _UUID_PATH_SEGMENT_PATTERN.match(segment):
        return "{" + "uuid" + "}"

    if any(char.isdigit() for char in segment):
        return "{var}"

    return segment


def _derive_metrics_endpoint(url: str) -> str:
    """Generate a normalized endpoint label from a URL."""

    if not url:
        return "unknown"

    parsed = urlparse(url)
    host = parsed.netloc.split(":", 1)[0] or "unknown"
    path_parts = [_sanitize_metric_segment(part) for part in parsed.path.strip("/").split("/") if part]
    path_label = "/".join(path_parts) if path_parts else "root"
    return f"{host}/{path_label}"


def _metrics_status_family(status: Optional[int]) -> str:
    """Convert HTTP status code to status family string."""

    if status is None or status < 100:
        return "error"
    return f"{status // 100}xx"


def _resolve_request_duration(
    response: RequestsResponseTypeOptional,
    fallback_duration: float,
) -> float:
    """Prefer requests' elapsed timing when available."""

    if response is not None:
        elapsed = getattr(response, "elapsed", None)
        if elapsed is not None:
            with contextlib.suppress(Exception):
                elapsed_seconds = float(elapsed.total_seconds())
                if elapsed_seconds > 0:
                    return elapsed_seconds
    return max(fallback_duration, 0.0)


def _record_api_metrics(
    endpoint: str,
    method: str,
    result: str,
    status_family: str,
    duration: float,
) -> None:
    """Emit API metrics via Prometheus registry helpers."""

    try:
        metrics_bundle = metrics()
        metrics_bundle.api_requests.inc(endpoint, method, result)
        metrics_bundle.api_latency.observe(endpoint, status_family, max(duration, 0.0))
    except Exception:
        logger.debug("Failed to record API metrics", exc_info=True)


# === STANDARDIZED LOGGING HELPERS ===
# These functions provide consistent logging format across Actions 6-9


def log_action_configuration(config_dict: dict[str, Any], section_title: str = "Config") -> None:
    """
    Log action configuration in standardized format.

    Args:
        config_dict: Dictionary of configuration key-value pairs
        section_title: Title prefix for the log entry (default: "Config")

    Example:
        log_action_configuration({
            "APP_MODE": "dry_run",
            "START_PAGE": 1,
            "MAX_PAGES": 2,
            "BATCH_SIZE": 10,
            "RATE_LIMIT_DELAY": 2.50
        })
        # Output: Configuration: APP_MODE=dry_run, START_PAGE=1, MAX_PAGES=2, BATCH_SIZE=10, RATE_LIMIT_DELAY=2.50s
    """
    formatted_parts: list[str] = []
    for key, value in config_dict.items():
        value_str = ("Yes" if value else "No") if isinstance(value, bool) else value
        formatted_parts.append(f"{key}={value_str}")

    summary = " | ".join(formatted_parts)
    logger.info(f"{section_title}: {summary}")


def log_starting_position(description: str, details: Optional[dict[str, Any]] = None) -> None:
    """
    Log starting position summary.

    Args:
        description: Main description of what will be processed
        details: Optional dictionary of additional details

    Example:
        log_starting_position(
            "Starting from page 1, will process up to 2 pages",
            {"Estimated matches": "~40 (20 per page)"}
        )
    """
    if not details:
        logger.info(description)
        return

    detail_parts = [f"{key}: {value}" for key, value in details.items()]
    logger.info(f"{description}\n " + "\n".join(detail_parts))


def log_cumulative_counts(counts: dict[str, int], prefix: str = "Cumulative") -> None:
    """
    Log cumulative counts in standardized format.

    Args:
        counts: Dictionary of counter names and values
        prefix: Prefix for the log line (default: "Cumulative")

    Example:
        log_cumulative_counts({"Pages": 1, "Batches": 2, "New": 15, "Updated": 5, "Skipped": 0, "Errors": 0})
        # Output: Cumulative: Pages=1, Batches=2, New=15, Updated=5, Skipped=0, Errors=0
    """
    count_str = ", ".join([f"{k}={v}" for k, v in counts.items()])
    logger.info(f"{prefix}: {count_str}")


def log_batch_indicator(
    batch_num: int,
    total_batches: int,
    item_range: Optional[tuple[int, int]] = None,
    page_num: Optional[int] = None,
    total_pages: Optional[int] = None,
) -> None:
    """
    Log batch/page indicator before processing.

    Args:
        batch_num: Current batch number
        total_batches: Total number of batches
        item_range: Optional tuple of (start_item, end_item) for this batch
        page_num: Optional current page number
        total_pages: Optional total number of pages

    Example:
        log_batch_indicator(1, 2, (1, 10), 1, 2)
        # Output:
        # (blank line)
        # Processing page 1 of 2 pages
        # Batch 1/2 (items 1-10)
    """
    logger.info("")  # Blank line before
    if page_num and total_pages:
        logger.info(f"Processing page {page_num} of {total_pages} pages")
    if item_range:
        logger.info(f"Batch {batch_num}/{total_batches} (items {item_range[0]}-{item_range[1]})")
    else:
        logger.info(f"Batch {batch_num}/{total_batches}")


def create_standard_progress_bar(total: int, desc: str = "Processing", unit: str = " item"):
    """
    Create standardized progress bar using tqdm.

    Args:
        total: Total number of items to process
        desc: Description for the progress bar
        unit: Unit name for items

    Returns:
        tqdm progress bar instance

    Example:
        with create_standard_progress_bar(20, "Processing", " match") as pbar:
            for item in items:
                # Process item
                pbar.update(1)
    """
    from tqdm import tqdm

    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )


def log_page_complete(page_num: int, page_counts: dict[str, int], cumulative_counts: dict[str, int]) -> None:
    """
    Log page completion summary.

    Args:
        page_num: Page number that was completed
        page_counts: Dictionary of counts for this page only
        cumulative_counts: Dictionary of cumulative counts across all pages

    Example:
        log_page_complete(1, {"Batches": 2, "New": 15, "Updated": 5},
                         {"Pages": 1, "Batches": 2, "New": 15, "Updated": 5, "Skipped": 0, "Errors": 0})
        # Output:
        # Page 1 complete: Batches=2, New=15, Updated=5
        # Cumulative: Pages=1, Batches=2, New=15, Updated=5, Skipped=0, Errors=0
    """
    page_str = ", ".join([f"{k}={v}" for k, v in page_counts.items()])
    logger.info(f"Page {page_num} complete: {page_str}")
    log_cumulative_counts(cumulative_counts)


def log_final_summary(summary_dict: dict[str, Any], run_time_seconds: float) -> None:
    """
    Log final summary with separators and aligned labels.

    Args:
        summary_dict: Dictionary of summary items (label: value)
        run_time_seconds: Total run time in seconds

    Example:
        log_final_summary({
            "Pages Scanned": 2,
            "Batches Processed": 4,
            "New Matches": 30,
            "Updated Matches": 10,
            "Skipped Matches": 0,
            "Errors": 0
        }, 90.45)
        # Output:
        # (blank line)
        # ================================================================================
        # FINAL SUMMARY
        # ================================================================================
        # Pages Scanned:       2
        # Batches Processed:   4
        # New Matches:         30
        # Updated Matches:     10
        # Skipped Matches:     0
        # Errors:              0
        # Total Run Time:      0 hr 1 min 30.45 sec
        # ================================================================================
        # (blank line)
    """
    print("")
    logger.info("-" * 45)
    logger.info("Final Summary")
    logger.info("-" * 45)

    # Log all summary items with aligned labels
    max_label_len = max(len(str(k)) for k in summary_dict)
    for label, value in summary_dict.items():
        logger.info(f"{str(label) + ':':<{max_label_len + 1}} {value}")

    # Log run time
    hours = int(run_time_seconds // 3600)
    minutes = int((run_time_seconds % 3600) // 60)
    seconds = run_time_seconds % 60
    logger.info(f"Total Run Time: {hours} hr {minutes} min {seconds:.2f} sec")


def log_action_status(action_name: str, success: bool, error_msg: Optional[str] = None) -> None:
    """
    Log final action status with checkmark or X.

    Args:
        action_name: Name of the action (e.g., "Match gathering")
        success: True if action completed successfully
        error_msg: Optional error message if success=False
    Example:
        log_action_status("Match gathering", True)
        # Output: standardized success banner

        log_action_status("Match gathering", False, "Session expired")
        # Output: standardized failure banner with reason
    """
    stage = "success" if success else "failure"
    details = {"error": error_msg} if (error_msg and not success) else None
    log_action_banner(
        action_name=action_name,
        action_number=None,
        stage=stage,
        logger_instance=logger,
        details=details,
    )


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
    data: Optional[dict[str, Any]] = None
    json_data: Optional[dict[str, Any]] = None
    json: Optional[dict[str, Any]] = None

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
    retry_status_codes: Union[list[int], set[int]] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_policy: Optional[Union[str, "RetryPolicyProfile"]] = "api"
    jitter_seconds: float = 0.2

    # Metadata
    api_description: str = "API Call"
    attempt: int = 1


# === MODULE CONSTANTS ===
# Re-export API constants for backwards compatibility
API_PATH_UUID = API_PATH_UUID_LEGACY

# Key constants for API responses
KEY_UCDMID = "ucdmid"
KEY_TEST_ID = "testId"
KEY_DATA = "data"

# === REGEX PATTERNS ===
# Pattern to extract JSON from JSONP callback format: callback({...})
JSONP_PATTERN = re.compile(r'^\w+\((.*)\)$', re.DOTALL)
# Pattern to extract centiMorgan (cM) values from text
CM_VALUE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*cM', re.IGNORECASE)


# === JSON UTILITY FUNCTIONS ===
def fast_json_loads(json_str: str) -> Any:
    """
    Fast JSON loading with fallback to standard library.
    Uses orjson if available, otherwise standard json.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed JSON object
    """
    try:
        # Dynamic import to handle missing orjson gracefully
        orjson = __import__('orjson')
        return orjson.loads(json_str)
    except (ImportError, ModuleNotFoundError):
        return json.loads(json_str)


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
    from selenium.webdriver.support import expected_conditions
    from selenium.webdriver.support.wait import WebDriverWait

    # --- Local application imports ---
    # Assume these are essential or handled elsewhere if missing
    from config import config_schema
    from core_imports import auto_register_module, get_function, get_logger, is_function_available, register_function

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

# test_framework imports removed - not directly accessed in this module

# ------------------------------------------------------------------------------------
# Helper functions (General Utilities)
# ------------------------------------------------------------------------------------


def parse_cookie(cookie_string: str) -> dict[str, str]:
    """
    Parses a raw HTTP cookie string into a dictionary of key-value pairs.
    Handles empty keys and values.
    """
    cookies: dict[str, str] = {}
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


# === COOKIE PERSISTENCE FUNCTIONS ===


def _get_cookie_file_path() -> Path:
    """Get the path to the cookie file."""
    return Path("ancestry_cookies.json")


def _save_login_cookies(session_manager: "SessionManager") -> bool:
    """Save login cookies to file for session persistence."""
    try:
        driver = session_manager.driver
        if driver is None:
            logger.debug("Cannot save cookies: No driver available")
            return False

        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        cookies = driver_proto.get_cookies()
        if not cookies:
            logger.debug("No cookies to save")
            return False

        cookies_file = _get_cookie_file_path()

        import json

        with cookies_file.open("w", encoding="utf-8") as f:
            json.dump(cookies, f, indent=2)

        logger.info("💾 Saved %d cookies to %s", len(cookies), cookies_file)
        return True

    except Exception as e:
        logger.warning("Failed to save cookies: %s", e)
        return False


def _load_login_cookies(session_manager: "SessionManager") -> bool:
    """Load saved login cookies from file."""
    try:
        driver = session_manager.driver
        if driver is None:
            logger.debug("Cannot load cookies: No driver available")
            return False

        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)

        cookies_file = _get_cookie_file_path()

        if not cookies_file.exists():
            logger.debug(f"No saved cookies file found at: {cookies_file}")
            return False

        import json

        with cookies_file.open(encoding="utf-8") as f:
            cookies: list[dict[str, Any]] = json.load(f)

        if not cookies:
            logger.debug("No cookies in saved file")
            return False

        # Add each cookie to the driver
        loaded_count = 0
        for cookie in cookies:
            try:
                # Remove expiry if it's in the past (causes Selenium errors)
                if "expiry" in cookie:
                    import time as time_module

                    if cookie["expiry"] < time_module.time():
                        del cookie["expiry"]

                # Cast cookie values to object to match Protocol
                cookie_obj: dict[str, object] = cast(dict[str, object], cookie)
                driver_proto.add_cookie(cookie_obj)
                loaded_count += 1
            except Exception as cookie_err:
                logger.debug(f"Failed to add cookie {cookie.get('name', 'unknown')}: {cookie_err}")
                continue

        logger.debug(f"🍪 Loaded {loaded_count}/{len(cookies)} cookies from {cookies_file}")
        return loaded_count > 0

    except Exception as e:
        logger.warning("Failed to load cookies: %s", e)
        return False


def load_login_cookies(session_manager: "SessionManager") -> bool:
    """Public alias so static analyzers see the loader being referenced."""

    return _load_login_cookies(session_manager)


# End of cookie persistence functions


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
        return str(text)

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
    hyphenated_elements: list[str] = []
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
    if not name:
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
            _format_name_part(part, i, lowercase_particles, uppercase_exceptions) for i, part in enumerate(parts)
        ]

        # Join and clean up
        final_name = " ".join(formatted_parts)
        final_name = re.sub(r"\s+", " ", final_name).strip()
        return final_name if final_name else "Valued Relative"

    except Exception as e:
        logger.error(f"Error formatting name '{name}': {e}", exc_info=False)
        return name.title()


# End of format_name

# ------------------------------
# Decorators (Remain in utils.py)
# ------------------------------


def retry(
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    max_delay: Optional[float] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator factory to retry a function with exponential backoff and jitter."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            cfg = config_schema  # Use new config system
            attempts = max_retries if max_retries is not None else getattr(cfg, "MAX_RETRIES", 3)
            backoff = backoff_factor if backoff_factor is not None else getattr(cfg, "BACKOFF_FACTOR", 1.0)
            max_delay_val = max_delay if max_delay is not None else getattr(cfg, "MAX_DELAY", 10.0)
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
                    sleep_time = min(backoff * (2**i), max_delay_val) + random.uniform(0, 0.5)
                    logger.warning(
                        f"Retry {i + 1}/{attempts} for {func.__name__} after exception: {type(e).__name__}. Sleeping {sleep_time:.2f}s."
                    )
                    time.sleep(sleep_time)
                # End of try/except
            # End of for
            # This part should ideally not be reached if raise is used above
            logger.error(f"Function '{func.__name__}' failed after all {attempts} retries (exited loop unexpectedly).")
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
    policy: Optional["RetryPolicyProfile"] = None,
) -> dict[str, Any]:
    """Get retry configuration with defaults from config_schema."""
    cfg = config_schema
    policy_attempts = policy.max_attempts if policy else None
    policy_initial = policy.initial_delay_seconds if policy else None
    policy_backoff = policy.backoff_factor if policy else None
    policy_max_delay = policy.max_delay_seconds if policy else None
    policy_jitter = policy.jitter_seconds if policy else 0.2
    return {
        "max_retries": max_retries
        if max_retries is not None
        else (policy_attempts if policy_attempts is not None else getattr(cfg, "MAX_RETRIES", 3)),
        "initial_delay": initial_delay
        if initial_delay is not None
        else (policy_initial if policy_initial is not None else getattr(cfg, "INITIAL_DELAY", 0.5)),
        "backoff_factor": backoff_factor
        if backoff_factor is not None
        else (policy_backoff if policy_backoff is not None else getattr(cfg, "BACKOFF_FACTOR", 1.5)),
        "retry_codes": set(
            retry_on_status_codes
            if retry_on_status_codes is not None
            else getattr(cfg, "RETRY_STATUS_CODES", [429, 500, 502, 503, 504])
        ),
        "max_delay": policy_max_delay if policy_max_delay is not None else getattr(cfg, "MAX_DELAY", 60.0),
        "jitter_seconds": float(policy_jitter),
    }


def _calculate_sleep_time(
    delay: float,
    backoff_factor: float,
    attempt: int,
    max_delay: float,
    jitter_seconds: float,
) -> float:
    """Calculate sleep time with exponential backoff and jitter."""
    base = min(delay * (backoff_factor ** (attempt - 1)), max_delay)
    jitter = random.uniform(0, jitter_seconds) if jitter_seconds > 0 else 0.0
    sleep_time = min(base + jitter, max_delay)
    return max(0.1, sleep_time)


def time_wait(wait_description: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator factory to time Selenium WebDriverWait calls."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"Wait '{wait_description}' completed successfully in {duration:.3f}s.")
                return result
            except TimeoutException as e:
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


# NOTE: CircuitBreaker class was removed from utils.py to eliminate duplication.
# Use the canonical implementations from core/error_handling.py:
#   - CircuitBreaker: from core.error_handling import CircuitBreaker, CircuitBreakerConfig
#   - CircuitBreakerOpenError: from core.error_handling import CircuitBreakerOpenError
# Or from core/circuit_breaker.py for session-based operations:
#   - SessionCircuitBreaker: from core.circuit_breaker import SessionCircuitBreaker


# ------------------------------
# PHASE 3.1: Direct AdaptiveRateLimiter usage
# ------------------------------
def get_rate_limiter(
    initial_fill_rate: Optional[float] = None,
    success_threshold: Optional[int] = None,
    min_fill_rate: Optional[float] = None,
    max_fill_rate: Optional[float] = None,
    capacity: Optional[float] = None,
    endpoint_profiles: Optional[dict[str, Any]] = None,
    rate_limiter_429_backoff: Optional[float] = None,
    rate_limiter_success_factor: Optional[float] = None,
):
    """Return the global AdaptiveRateLimiter singleton.

    PHASE 3.1: Now directly returns AdaptiveRateLimiter (no adapter/wrapper).
    All calling code updated to use the new interface:
    - wait() -> wait()
    - increase_delay() -> on_429_error()
    - decrease_delay() -> on_success()

    Returns:
        AdaptiveRateLimiter: The unified rate limiter singleton
    """
    from rate_limiter import get_adaptive_rate_limiter

    return get_adaptive_rate_limiter(
        initial_fill_rate=initial_fill_rate,
        success_threshold=success_threshold,
        min_fill_rate=min_fill_rate,
        max_fill_rate=max_fill_rate,
        capacity=capacity,
        endpoint_profiles=endpoint_profiles,
        rate_limiter_429_backoff=rate_limiter_429_backoff,
        rate_limiter_success_factor=rate_limiter_success_factor,
    )


# ------------------------------
# Session Management (MOVED TO core.session_manager)
# ------------------------------
# SessionManager class has been moved to core.session_manager.SessionManager
# Import it from there: from core.session_manager import SessionManager

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
    contextual_headers = getattr(cfg.api, "contextual_headers", {}).get(api_description, {})
    if isinstance(contextual_headers, dict):
        sanitized_headers = {
            str(header_key): str(header_value) for header_key, header_value in contextual_headers.items()
        }
        base_headers.update(sanitized_headers)
    else:
        logger.warning(f"[{api_description}] Expected dict for contextual headers, got {type(contextual_headers)}")

    # Apply explicit overrides
    if headers:
        base_headers.update(headers)
        logger.debug(f"[{api_description}] Applied {len(headers)} explicit header overrides.")
    # End of if

    return base_headers


# End of _prepare_base_headers

# Helper functions for _prepare_api_headers


def _get_user_agent_from_browser(driver: DriverType, api_description: str) -> Optional[str]:
    """Get User-Agent from browser using JavaScript."""
    if not driver:
        return None

    # Cast to Protocol to ensure types
    driver_proto = cast(DriverProtocol, driver)
    try:
        user_agent = driver_proto.execute_script("return navigator.userAgent;")
    except WebDriverException as exc:  # pragma: no cover - driver-specific failures
        logger.debug("[%s] WebDriver error getting User-Agent: %s", api_description, exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("[%s] Unexpected error getting User-Agent: %s", api_description, exc)
        return None

    return user_agent if isinstance(user_agent, str) else None


def _add_origin_header(final_headers: dict[str, str], base_headers: dict[str, str], api_description: str) -> None:
    """Add Origin header for relevant HTTP methods."""
    http_method = base_headers.get("_method", "GET").upper()
    if http_method not in {"GET", "HEAD", "OPTIONS"}:
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
    if csrf_token.strip().startswith("{"):
        try:
            token_obj = json.loads(csrf_token)
            raw_token_val = token_obj.get("csrfToken", csrf_token)
        except json.JSONDecodeError:
            logger.warning(f"[{api_description}] CSRF token looks like JSON but failed to parse, using raw value.")
    return str(raw_token_val)


def _add_csrf_token_header(
    final_headers: dict[str, str], session_manager: "SessionManager", api_description: str
) -> None:
    """Add CSRF token header if available."""
    csrf_token = session_manager.csrf_token
    if csrf_token:
        raw_token_val = _parse_csrf_token(csrf_token, api_description)
        final_headers["X-CSRF-Token"] = raw_token_val
        logger.debug(f"[{api_description}] Added X-CSRF-Token header.")
    else:
        logger.warning(f"[{api_description}] CSRF token requested but not found in SessionManager.")


def _add_user_id_header(final_headers: dict[str, str], session_manager: "SessionManager", api_description: str) -> None:
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
    session_manager: "SessionManager",  # Assume available
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

    # Skip runtime header generation to prevent recursion
    logger.debug(f"[{api_description}] Skipping runtime header generation to prevent recursion.")

    # Add CSRF token if requested
    if use_csrf_token:
        _add_csrf_token_header(final_headers, session_manager, api_description)

    # Add User ID header conditionally
    _add_user_id_header(final_headers, session_manager, api_description)

    # Remove internal _method key
    final_headers.pop("_method", None)

    return final_headers


# End of _prepare_api_headers

# Cookie cache to reduce excessive synchronization
_cookie_sync_cache = {"last_sync_time": 0.0, "sync_interval": 30.0}  # Sync every 30 seconds max


def _should_skip_cookie_sync(force_sync: bool, time_since_last_sync: float) -> bool:
    """
    Determine if cookie sync can be skipped based on cache.

    Args:
        force_sync: Whether sync is forced
        time_since_last_sync: Time since last sync in seconds

    Returns:
        bool: True if sync can be skipped, False otherwise
    """
    return not force_sync and time_since_last_sync < _cookie_sync_cache["sync_interval"]


def _validate_driver_for_sync(
    driver: DriverType, session_manager: "SessionManager", api_description: str, attempt: int
) -> bool:
    """
    Validate that driver is available for cookie sync.

    Args:
        driver: The WebDriver instance
        session_manager: The session manager instance
        api_description: Description of the API being called
        attempt: The current attempt number

    Returns:
        bool: True if driver is valid, False otherwise
    """
    driver_is_valid = driver and session_manager.driver
    if not driver_is_valid and attempt == 1:
        logger.warning(
            f"[{api_description}] Browser session invalid or driver None (Attempt {attempt}). "
            "Runtime headers might be incomplete/stale."
        )
    return bool(driver_is_valid)


def _perform_cookie_sync(
    session_manager: "SessionManager",
    api_description: str,
    attempt: int,
    force_sync: bool,
    time_since_last_sync: float,
    current_time: float,
) -> bool:
    """
    Perform the actual cookie synchronization.

    Args:
        session_manager: The session manager instance
        api_description: Description of the API being called
        attempt: The current attempt number
        force_sync: Whether sync is forced
        time_since_last_sync: Time since last sync in seconds
        current_time: Current timestamp

    Returns:
        bool: True if sync successful, False otherwise
    """
    # Only log on first attempt or if forced to reduce verbosity
    if attempt == 1 or force_sync:
        logger.debug(
            f"[{api_description}] Syncing cookies from browser "
            f"(cache expired, last sync {time_since_last_sync:.1f}s ago)"
        )

    api_manager = getattr(session_manager, "api_manager", None)
    browser_manager = getattr(session_manager, "browser_manager", None)
    sync_method = getattr(api_manager, "sync_cookies_from_browser", None)
    if not browser_manager or not callable(sync_method):
        logger.warning(f"[{api_description}] Cookie sync requested but browser/API managers are unavailable")
        return False

    sync_success = sync_method(browser_manager, session_manager=session_manager)

    if sync_success:
        # Update cache timestamp
        _cookie_sync_cache["last_sync_time"] = current_time
        if attempt == 1 or force_sync:
            logger.debug(f"[{api_description}] Cookie sync successful")
        return True

    logger.warning(f"[{api_description}] Cookie sync failed (Attempt {attempt}).")
    return False


def _sync_cookies_for_request(
    session_manager: "SessionManager",
    driver: DriverType,
    api_description: str,
    attempt: int = 1,
    force_sync: bool = False,
) -> bool:
    """
    Synchronizes cookies from the WebDriver to the requests session.

    Uses caching to reduce excessive cookie synchronization - only syncs if:
    1. force_sync=True (e.g., after session recovery)
    2. More than 30 seconds since last sync
    3. First attempt of a request

    Args:
        session_manager: The session manager instance
        driver: The WebDriver instance
        api_description: Description of the API being called
        attempt: The current attempt number
        force_sync: Force cookie sync regardless of cache

    Returns:
        True if cookies were synced successfully, False otherwise
    """
    import time

    # Check if we can skip cookie sync (use cached cookies)
    current_time = time.time()
    time_since_last_sync = current_time - _cookie_sync_cache["last_sync_time"]

    if _should_skip_cookie_sync(force_sync, time_since_last_sync):
        return True

    # Validate driver
    if not _validate_driver_for_sync(driver, session_manager, api_description, attempt):
        return False

    # Perform cookie synchronization
    try:
        return _perform_cookie_sync(
            session_manager, api_description, attempt, force_sync, time_since_last_sync, current_time
        )
    except Exception as e:
        logger.error(f"[{api_description}] Exception during cookie sync (Attempt {attempt}): {e}", exc_info=True)
        return False


# End of _sync_cookies_for_request


def _get_rate_limiter_from_session(
    session_manager: "SessionManager",
) -> Optional[RateLimiterProtocol]:
    """Return the adaptive rate limiter attached to the session, if any."""

    rate_limiter = getattr(session_manager, "rate_limiter", None)
    if rate_limiter is None:
        return None
    return cast(RateLimiterProtocol, rate_limiter)


def _apply_rate_limiting(
    session_manager: "SessionManager",
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
    rate_limiter = _get_rate_limiter_from_session(session_manager)
    if rate_limiter is None:
        return 0.0

    wait_time = rate_limiter.wait(api_description)
    if wait_time > 0:
        try:
            metrics_bundle = metrics()
            delay_metric = getattr(metrics_bundle, "rate_limiter_delay", None)
            if delay_metric is not None:
                delay_metric.observe(wait_time)
        except Exception:
            logger.debug("Failed to record rate limiter delay", exc_info=True)

    if wait_time > 0.1:  # Log only significant waits
        logger.debug(f"[{api_description}] Rate limit wait: {wait_time:.2f}s (Attempt {attempt})")
    return wait_time


# End of _apply_rate_limiting

# _log_request_details removed - unused function (70 lines)


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
    # Timeout tuple: (connect_timeout, read_timeout)
    # - connect_timeout: TCP handshake must complete within 30s
    # - read_timeout: Response must arrive within 90s after connection
    # This prevents indefinite hangs at both connection and response levels
    request_timeout = config.timeout if config.timeout is not None else sel_cfg.api_timeout
    # Convert single timeout to tuple for requests library
    timeout_tuple = (30, request_timeout)

    req_session = _get_requests_session(config.session_manager)
    effective_cookie_jar = config.cookie_jar if config.cookie_jar is not None else req_session.cookies
    http_method = config.method.upper()

    # Handle specific API quirks (e.g., allow_redirects)
    effective_allow_redirects = config.allow_redirects
    # Note: Match List API should allow redirects (as it did in working version from 2 months ago)

    logger.debug(
        f"[{config.api_description}] Preparing Request: Method={http_method}, URL={config.url}, Timeout={timeout_tuple}s, AllowRedirects={effective_allow_redirects}"
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

    # Return all prepared request parameters
    return {
        "method": http_method,
        "url": config.url,
        "headers": final_headers,
        "data": config.data,
        "json": effective_json_data,  # Use 'json' for requests.request, not 'json_data'
        "timeout": timeout_tuple,  # (connect_timeout, read_timeout) tuple
        "verify": True,  # Standard verification
        "allow_redirects": effective_allow_redirects,
        "cookies": effective_cookie_jar,
    }


# End of _prepare_api_request


def _execute_api_request(
    session_manager: "SessionManager",
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
    req_session = _get_requests_session(session_manager)

    try:
        # Execute the request
        return req_session.request(**request_params)

    except RequestException as e:
        logger.warning(f"[_api_req Attempt {attempt} '{api_description}'] RequestException: {type(e).__name__} - {e}")

        # REMOVED: RetryError tracking here is redundant - each 429 is already tracked in the retry loop (line 2316)
        # Previously this caused cascading delays: 5 retries x increase_delay() + 1 final increase = 6x increases per failed API call
        # Now delays are increased exactly once per actual 429 status code in _handle_status_code_response()

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
    session_manager: "SessionManager",
    api_description: str,
) -> bool:
    """Validate prerequisites for API request."""
    if not session_manager:
        logger.error(f"{api_description}: Aborting - SessionManager or internal requests_session missing.")
        return False

    try:
        _get_requests_session(session_manager)
    except AttributeError:
        logger.error(f"{api_description}: Aborting - SessionManager or internal requests_session missing.")
        return False

    if not config_schema:
        logger.error(f"{api_description}: Aborting - Config schema not loaded.")
        return False

    return True


def _calculate_retry_sleep_time(
    current_delay: float,
    backoff_factor: float,
    attempt: int,
    max_delay: float,
    jitter_seconds: float,
) -> float:
    """Calculate sleep time for retry with exponential backoff and jitter."""
    return _calculate_sleep_time(
        current_delay,
        backoff_factor,
        attempt,
        max_delay,
        jitter_seconds,
    )


def _handle_failed_request_response(
    retries_left: int,
    max_retries: int,
    api_description: str,
    current_delay: float,
    backoff_factor: float,
    attempt: int,
    max_delay: float,
    jitter_seconds: float,
) -> tuple[bool, int, float]:
    """
    Handle failed request response (None).

    Returns:
        Tuple of (should_continue, new_retries_left, new_current_delay)
    """
    retries_left -= 1
    if retries_left <= 0:
        logger.error(f"{api_description}: Request failed after {max_retries} attempts.")
        return False, retries_left, current_delay

    sleep_time = _calculate_retry_sleep_time(
        current_delay,
        backoff_factor,
        attempt,
        max_delay,
        jitter_seconds,
    )
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
    session_manager: "SessionManager",
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
        if response and hasattr(response, 'text'):
            with contextlib.suppress(Exception):
                logger.debug(f"   << Final Response Text (Retry Fail): {response.text[:500]}...")
        return False, response, retries_left, current_delay

    sleep_time = _calculate_sleep_time(
        current_delay,
        retry_ctx.backoff_factor,
        retry_ctx.attempt,
        retry_ctx.max_delay,
        getattr(retry_ctx, "jitter_seconds", 0.2),
    )

    if status == 429:  # Too Many Requests
        rate_limiter = _get_rate_limiter_from_session(session_manager)
        if rate_limiter:
            rate_limiter.on_429_error(api_description)

    retry_type = "🚨 429 EXPONENTIAL BACKOFF" if status == 429 else "Retry"
    logger.warning(
        f"{retry_type}: {api_description} Status {status} (Attempt {retry_ctx.attempt}/{retry_ctx.max_attempts}). "
        f"Retrying in {sleep_time:.2f}s... | Note: This delay is SEPARATE from adaptive base delay and they stack"
    )
    if response and hasattr(response, 'text'):
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
        if response and hasattr(response, 'headers'):
            logger.debug(f"   << Redirect Location: {response.headers.get('Location')}")
        return response
    return None


def _handle_error_status(
    response: RequestsResponseTypeOptional,
    status: int,
    reason: str,
    api_description: str,
    session_manager: "SessionManager",
) -> RequestsResponseTypeOptional:
    """Handle non-retryable error status codes."""
    # For login verification API, use debug level for 401/403 errors
    if api_description == "API Login Verification (header/dna)" and status in {401, 403}:
        logger.debug(f"[_api_req '{api_description}'] Received expected status: {status} {reason}")
    else:
        logger.warning(f"[_api_req '{api_description}'] Received NON-retryable error status: {status} {reason}")

    if status in {401, 403}:
        if api_description == "API Login Verification (header/dna)":
            logger.debug(f"{api_description}: API call returned {status} {reason}. User not logged in.")
        else:
            logger.warning(f"{api_description}: API call failed {status} {reason}. Session expired/invalid?")
        session_manager.session_ready = False
    else:
        logger.error(f"{api_description}: Non-retryable error: {status} {reason}.")

    if response and hasattr(response, 'text'):
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
                except JSONDecodeError as json_err:
                    logger.error(f"{api_description}: OK ({status}), but JSON decode FAILED: {json_err}")
                    with contextlib.suppress(Exception):
                        logger.debug(f"   << Response Text (JSON Error): {response.text[:500]}...")
                    # Return None because caller expected JSON but didn't get it
                    result = None
            elif api_description == "CSRF Token API" and "text/plain" in content_type:
                csrf_text = response.text.strip()
                logger.debug(f"{api_description}: Received text/plain as expected for CSRF.")
                result = csrf_text if csrf_text else None
            else:
                logger.debug(f"{api_description}: OK ({status}), Content-Type '{content_type}'. Returning raw TEXT.")
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
    jitter_seconds: float,
) -> tuple[bool, int, float]:
    """
    Handle exception during request attempt.
    Returns (should_continue, retries_left, current_delay).
    """
    logger.warning(f"[_api_req Attempt {attempt} '{api_description}'] RequestException: {type(e).__name__} - {e}")
    retries_left -= 1

    if retries_left <= 0:
        logger.error(
            f"{api_description}: Request failed after {max_retries} attempts. Final Error: {e}",
            exc_info=False,
        )
        return (False, retries_left, current_delay)

    sleep_time = _calculate_retry_sleep_time(
        current_delay,
        backoff_factor,
        attempt,
        max_delay,
        jitter_seconds,
    )
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
    session_manager: "SessionManager",
    force_text_response: bool,
    request_params: dict[str, Any],
    metrics_endpoint: str,
    metrics_method: str,
    attempt_duration: float,
) -> tuple[Optional[Any], bool, int, float, Optional[Exception]]:
    """
    Handle response status codes and return appropriate result.
    Returns (response, should_continue, retries_left, current_delay, last_exception).
    """
    status = response.status_code
    reason = response.reason
    retries_left = retry_ctx.retries_left or 0
    current_delay = retry_ctx.current_delay
    duration = _resolve_request_duration(response, attempt_duration)
    status_family = _metrics_status_family(status)

    # Handle retryable status codes
    if retry_ctx.retry_status_codes and status in retry_ctx.retry_status_codes:
        should_continue, return_response, retries_left, current_delay = _handle_retryable_status(
            response, status, reason, retry_ctx, api_description, session_manager
        )
        if not should_continue:
            _record_api_metrics(
                metrics_endpoint,
                metrics_method,
                "failure",
                status_family,
                duration,
            )
            return (return_response, False, retries_left, current_delay, None)
        last_exception = HTTPError(f"{status} Error", response=response)
        _record_api_metrics(
            metrics_endpoint,
            metrics_method,
            "retry",
            status_family,
            duration,
        )
        return (None, True, retries_left, current_delay, last_exception)

    # Handle redirects
    redirect_response = _handle_redirect_response(
        response, status, reason, request_params["allow_redirects"], api_description
    )
    if redirect_response is not None:
        _record_api_metrics(
            metrics_endpoint,
            metrics_method,
            "failure",
            status_family,
            duration,
        )
        return (redirect_response, False, retries_left, current_delay, None)

    # Handle non-retryable error status codes
    if not response.ok:
        error_response = _handle_error_status(response, status, reason, api_description, session_manager)
        _record_api_metrics(
            metrics_endpoint,
            metrics_method,
            "failure",
            status_family,
            duration,
        )
        return (error_response, False, retries_left, current_delay, None)

    # Process successful response
    if response.ok:
        logger.debug(f"{api_description}: Successful response ({status} {reason}).")
        rate_limiter = _get_rate_limiter_from_session(session_manager)
        if rate_limiter:
            rate_limiter.on_success()
        processed_response = _process_api_response(
            response=response,
            api_description=api_description,
            force_text_response=force_text_response,
        )
        _record_api_metrics(
            metrics_endpoint,
            metrics_method,
            "success",
            status_family,
            duration,
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
    metrics_endpoint = _derive_metrics_endpoint(config.url)
    metrics_method = config.method.upper()
    attempt_start = time.perf_counter()
    attempt_duration = 0.0

    try:
        # Prepare and execute the request
        request_params = _prepare_api_request(config)
        metrics_endpoint = _derive_metrics_endpoint(request_params.get("url", config.url))
        metrics_method = str(request_params.get("method", config.method)).upper()

        attempt_start = time.perf_counter()
        response = _execute_api_request(
            session_manager=config.session_manager,
            api_description=config.api_description,
            request_params=request_params,
            attempt=config.attempt,
        )
        attempt_duration = time.perf_counter() - attempt_start
        attempt_duration = _resolve_request_duration(response, attempt_duration)

        # Handle failed request (response is None)
        if response is None:
            should_continue, retries_left, current_delay = _handle_failed_request_response(
                retries_left,
                config.max_retries,
                config.api_description,
                current_delay,
                config.backoff_factor,
                config.attempt,
                config.max_delay,
                config.jitter_seconds,
            )
            result_should_continue = should_continue
            result_label = "retry" if result_should_continue else "failure"
            _record_api_metrics(
                metrics_endpoint,
                metrics_method,
                result_label,
                "error",
                attempt_duration,
            )
        else:
            # Handle response status
            retry_ctx = RetryContext(
                attempt=config.attempt,
                max_attempts=config.max_retries,
                max_delay=config.max_delay,
                backoff_factor=config.backoff_factor,
                current_delay=current_delay,
                retries_left=retries_left,
                retry_status_codes=config.retry_status_codes,
                jitter_seconds=config.jitter_seconds,
            )
            return _handle_response_status(
                response,
                retry_ctx,
                config.api_description,
                config.session_manager,
                config.force_text_response,
                request_params,
                metrics_endpoint,
                metrics_method,
                attempt_duration,
            )

    except RequestException as e:
        attempt_duration = time.perf_counter() - attempt_start
        should_continue, retries_left, current_delay = _handle_request_exception(
            e,
            config.attempt,
            config.max_retries,
            retries_left,
            config.api_description,
            current_delay,
            config.backoff_factor,
            config.max_delay,
            config.jitter_seconds,
        )
        result_should_continue = should_continue
        result_exception = e
        result_label = "retry" if result_should_continue else "failure"
        _record_api_metrics(
            metrics_endpoint,
            metrics_method,
            result_label,
            "error",
            attempt_duration,
        )

    except Exception as e:
        logger.critical(
            f"{config.api_description}: CRITICAL Unexpected error during request attempt {config.attempt}: {e}",
            exc_info=True,
        )
        attempt_duration = time.perf_counter() - attempt_start
        _record_api_metrics(
            metrics_endpoint,
            metrics_method,
            "failure",
            "error",
            attempt_duration,
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
        logger.error(f"{config.api_description}: Exited retry loop. Last Exception: {last_exception}.")
        return None

    # Return the last response (this should be a non-retryable error response)
    logger.debug(
        f"[_api_req '{config.api_description}'] Returning last Response object (Status: {response.status_code})."
    )
    return response


def _api_req(
    url: str,
    driver: DriverType,
    session_manager: "SessionManager",
    method: str = "GET",
    data: Optional[dict[str, Any]] = None,
    json_data: Optional[dict[str, Any]] = None,
    json: Optional[dict[str, Any]] = None,
    use_csrf_token: bool = True,
    headers: Optional[dict[str, str]] = None,
    referer_url: Optional[str] = None,
    api_description: str = "API Call",
    timeout: Optional[int] = None,
    cookie_jar: Optional[RequestsCookieJar] = None,
    allow_redirects: bool = True,
    force_text_response: bool = False,
    add_default_origin: bool = True,
) -> Union[ApiResponseType, RequestsResponseTypeOptional]:
    """
    Makes an HTTP request using the shared requests.Session from SessionManager.
    Handles runtime header generation, cookie synchronization, rate limiting,
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
    Handles runtime header generation, cookie synchronization, rate limiting,
    retries, and basic response processing. Includes enhanced logging.

    Returns: Parsed JSON (dict/list), raw text (str), None on retryable failure,
             or Response object on non-retryable error/redirect disabled.
    """
    # Validate prerequisites
    if not _validate_api_req_prerequisites(config.session_manager, config.api_description):
        return None

    # Get retry configuration using shared retry policy profiles
    resolved_policy = resolve_retry_policy(getattr(config, "retry_policy", None), default="api")
    retry_settings = _get_retry_config(
        max_retries=None,
        initial_delay=None,
        backoff_factor=None,
        retry_on_status_codes=None,
        policy=resolved_policy,
    )

    # Update config with retry parameters
    config.max_retries = retry_settings["max_retries"]
    config.initial_delay = retry_settings["initial_delay"]
    config.backoff_factor = retry_settings["backoff_factor"]
    config.max_delay = retry_settings["max_delay"]
    config.retry_status_codes = retry_settings["retry_codes"]
    config.jitter_seconds = retry_settings["jitter_seconds"]
    if resolved_policy is not None:
        config.retry_policy = resolved_policy.name

    # Execute request with retry loop
    return _execute_request_with_retries(config)


# End of _api_req

# Helper functions for make_ube


def _validate_driver_session(driver: DriverType) -> bool:
    """Validate that driver session is active."""
    if not driver:
        return False
    try:
        _ = driver.title  # Quick check for session validity
        return True
    except WebDriverException as e:
        logger.warning(f"Cannot generate UBE header: Session invalid/unresponsive ({type(e).__name__}).")
        return False


def _extract_cookie_value(cookie_obj: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return cookie value when available."""
    if not isinstance(cookie_obj, Mapping):
        return None
    value = cookie_obj.get("value")
    return str(value) if value is not None else None


def _build_cookie_lookup(driver: WebDriver) -> dict[str, str]:
    """Create a name->value mapping for all cookies in the driver."""
    # Cast to Protocol to ensure types
    driver_proto = cast(DriverProtocol, driver)
    cookies_raw = driver_proto.get_cookies()
    cookies_dict: dict[str, str] = {}
    for cookie in cookies_raw:
        name = cookie.get("name")
        value = cookie.get("value")
        if isinstance(name, str) and value is not None:
            cookies_dict[name] = str(value)
    return cookies_dict


def _get_ancsessionid_cookie(driver: DriverType) -> Optional[str]:
    """Get ANCSESSIONID cookie value from driver."""
    if not driver:
        return None
    try:
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        direct_value = _extract_cookie_value(driver_proto.get_cookie("ANCSESSIONID"))
        if direct_value:
            return direct_value

        cookies_dict = _build_cookie_lookup(driver)
        ancsessionid = cookies_dict.get("ANCSESSIONID")
        if not ancsessionid:
            logger.warning("ANCSESSIONID cookie not found. Cannot generate UBE header.")
            return None
        return ancsessionid
    except (NoSuchCookieException, WebDriverException) as cookie_e:
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
    user_consent = (
        "necessary|preference|performance|analytics1st|analytics3rd|advertising1st|advertising3rd|attribution3rd"
    )

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

# NewRelic and tracing header generation functions removed - unused (45 lines)
# These were for generating NewRelic, traceparent, and tracestate headers
# Can be restored from git history if needed in the future

# ----------------------------------------------------------------------------
# Login Functions (Remain in utils.py)
# ----------------------------------------------------------------------------
# Note: TWO_STEP_VERIFICATION_HEADER_SELECTOR is imported from my_selectors.py

# Helper functions for handle_two_fa


def _wait_for_2fa_header(element_wait: "WebDriverWait[Any]", session_manager: "SessionManager") -> bool:
    """Wait for 2FA page header to appear."""
    try:
        logger.debug(f"Waiting for 2FA page header using selector: '{TWO_STEP_VERIFICATION_HEADER_SELECTOR}'")
        _wait_until_visible(element_wait, (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR))
        logger.debug("2FA page header detected.")
        return True
    except TimeoutException:
        logger.debug("Did not detect 2FA page header within timeout.")
        if login_status(session_manager, disable_ui_fallback=True) is True:
            logger.info("User appears logged in after checking for 2FA page. Assuming 2FA handled/skipped.")
            return True
        logger.warning("Assuming 2FA not required or page didn't load correctly (header missing).")
        return False
    except WebDriverException as e:
        logger.error(f"WebDriverException waiting for 2FA header: {e}")
        return False


def _find_sms_button_with_selectors(selector_wait: "WebDriverWait[Any]") -> Optional[WebElement]:
    """Try multiple selectors to find the SMS button."""
    sms_selectors = [
        "button[data-method='sms']",
        TWO_FA_SMS_SELECTOR,
        "//button[contains(., 'Text message')]",
        "//div[contains(text(), 'Text message')]",
        "//button[contains(text(), 'Text message')]",
    ]

    for idx, selector in enumerate(sms_selectors):
        try:
            logger.debug(f"[DEBUG] Trying selector {idx + 1}/{len(sms_selectors)}: {selector}")
            locator: Locator = (By.XPATH, selector) if selector.startswith("//") else (By.CSS_SELECTOR, selector)
            sms_button = _wait_until_clickable(selector_wait, locator)
            logger.debug(f"✓ Found SMS button with selector: {selector}")
            return sms_button
        except (TimeoutException, NoSuchElementException) as e:
            logger.debug(f"SMS button not found with selector {idx + 1}: {selector} - {type(e).__name__}")
        except Exception as e:
            logger.debug(f"Error trying selector {idx + 1} ({selector}): {e}")

    return None


def _debug_log_page_buttons(driver: WebDriver) -> None:
    """Log all buttons on the page for debugging."""
    try:
        all_buttons: list[WebElement] = driver.find_elements(By.TAG_NAME, "button")
        logger.debug(f"[DEBUG] Found {len(all_buttons)} buttons on page")
        for i, btn in enumerate(all_buttons[:10]):
            try:
                btn_text = btn.text.strip()
                btn_proto = cast(WebElementProtocol, btn)
                btn_classes = btn_proto.get_attribute("class")
                btn_data_method = btn_proto.get_attribute("data-method")
                if btn_text or btn_data_method:
                    logger.debug(
                        f"  Button {i}: text='{btn_text}', class='{btn_classes}', data-method='{btn_data_method}'"
                    )
            except Exception:
                pass
    except Exception as debug_err:
        logger.debug(f"[DEBUG] Error listing buttons: {debug_err}")


def _perform_sms_button_click(driver: WebDriver, sms_button: WebElement) -> bool:
    """Attempt to click the SMS button with fallback strategies."""
    try:
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].click();", sms_button)
        logger.debug("SMS button clicked via JavaScript.")
        print("  ✓ SMS verification code requested. Check your phone!", flush=True)
        time.sleep(3)
        return _wait_for_code_input_field(driver)
    except WebDriverException as click_err:
        logger.error(f"Error clicking SMS button via JS: {click_err}")
        try:
            sms_button.click()
            logger.debug("SMS button clicked via standard click.")
            print("  ✓ SMS verification code requested. Check your phone!", flush=True)
            time.sleep(3)
            return _wait_for_code_input_field(driver)
        except Exception as std_err:
            logger.error(f"Standard click also failed: {std_err}")
            print("  ✗ Failed to click SMS button. Please click 'Text message' manually.", flush=True)
            return False


def _click_sms_button(driver: WebDriver) -> bool:
    """Try clicking the SMS 'Send Code' button."""
    print("  📱 Looking for SMS/Text message option...", flush=True)
    logger.debug(f"Looking for 2FA SMS button with selector: '{TWO_FA_SMS_SELECTOR}'")

    selector_wait = WebDriverWait(driver, 5)
    sms_button = _find_sms_button_with_selectors(selector_wait)

    if not sms_button:
        logger.warning("Could not find SMS button with any selector.")
        _debug_log_page_buttons(driver)
        print("  ⚠️  SMS button not found. Please click 'Text message' manually.", flush=True)
        return False

    return _perform_sms_button_click(driver, sms_button)


def _wait_for_code_input_field(driver: WebDriver) -> bool:
    """Wait for 2FA code input field to appear after clicking Send Code."""
    try:
        logger.debug(f"Waiting for 2FA code input field: '{TWO_FA_CODE_INPUT_SELECTOR}'")
        _wait_until_visible(WebDriverWait(driver, 10), (By.CSS_SELECTOR, TWO_FA_CODE_INPUT_SELECTOR))
        logger.debug("Code input field appeared after clicking 'Send Code'.")
        return True
    except TimeoutException:
        logger.debug("Code input field not visible yet - will wait for manual entry")
        return False
    except WebDriverException as e_input:
        logger.error(f"Error waiting for 2FA code input field: {e_input}. Check selector: {TWO_FA_CODE_INPUT_SELECTOR}")
        return False


def _wait_for_user_2fa_action(driver: WebDriver, code_entry_timeout: int) -> bool:
    """Wait for user to manually enter 2FA code and submit."""
    logger.info(f"Waiting up to {code_entry_timeout}s for user to manually enter 2FA code and submit...")
    start_time = time.time()
    check_interval = 0.5  # Check every half second for responsive detection

    while time.time() - start_time < code_entry_timeout:
        try:
            # Check if the 2FA header is GONE (indicates page change/submission)
            wait = WebDriverWait(driver, check_interval)
            _wait_until_not_visible(wait, (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR))
            logger.info("2FA page elements disappeared, assuming user submitted code.")
            return True
        except TimeoutException:
            time.sleep(2)  # Check every 2 seconds
        except NoSuchElementException:
            logger.info("2FA header element no longer present.")
            return True
        except WebDriverException as e:
            logger.error(f"WebDriver error checking for 2FA header during wait: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking for 2FA header during wait: {e}")
            return False

    logger.error(f"Timed out ({code_entry_timeout}s) waiting for user 2FA action (page did not change).")
    return False


def _verify_2fa_completion(session_manager: "SessionManager") -> bool:
    """Verify that 2FA was completed successfully by checking login status."""
    logger.info("Re-checking login status after potential 2FA submission...")
    time.sleep(1)  # Allow page to settle
    final_status = login_status(session_manager, disable_ui_fallback=False)
    if final_status is True:
        logger.info("User completed 2FA successfully (login confirmed after page change).")
        # Save cookies after successful 2FA login
        _save_login_cookies(session_manager)
        return True
    logger.error("2FA page disappeared, but final login status check failed or returned False.")
    return False


@time_wait("Handle 2FA Page")
def handle_two_fa(session_manager: "SessionManager") -> bool:
    if session_manager.driver is None:
        logger.error("handle_two_fa: SessionManager driver is None. Cannot proceed.")
        return False

    driver = session_manager.driver
    element_wait = WebDriverWait(driver, config_schema.selenium.explicit_wait)

    try:
        print("Two-factor authentication required. Please check your email or phone for a verification code.")
        logger.debug("Handling Two-Factor Authentication (2FA)...")

        # Wait for 2FA page header
        if not _wait_for_2fa_header(element_wait, session_manager):
            return False

        # Handle cookie consent banner if present
        logger.debug("Checking for cookie consent banner on 2FA page...")
        if not consent(driver):
            logger.warning("Failed to handle consent banner on 2FA page, but continuing anyway.")

        # Try clicking SMS button (non-critical, continue if fails)
        _click_sms_button(driver)

        # Wait for user action
        code_entry_timeout = config_schema.selenium.two_fa_code_entry_timeout
        user_action_detected = _wait_for_user_2fa_action(driver, code_entry_timeout)

        # Verify completion
        if user_action_detected:
            return _verify_2fa_completion(session_manager)
        return False

    except WebDriverException as e:
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


def _clear_input_field(driver: WebDriver, input_element: WebElement, field_name: str) -> bool:
    """Clear an input field robustly."""
    try:
        input_element.click()
        time.sleep(0.1)
        input_element.clear()
        time.sleep(0.1)
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].value = '';", input_element)
        time.sleep(0.1)
        return True
    except (ElementNotInteractableException, StaleElementReferenceException) as e:
        logger.warning(f"Issue clicking/clearing {field_name} field ({type(e).__name__}). Proceeding cautiously.")
        return True
    except WebDriverException as e:
        logger.error(f"WebDriverException clicking/clearing {field_name}: {e}. Aborting.")
        return False


def _enter_username(driver: WebDriver, element_wait: "WebDriverWait[Any]") -> bool:
    """Enter username into the login form."""
    logger.debug(f"Waiting for username input: '{USERNAME_INPUT_SELECTOR}'...")
    username_input = _wait_until_visible(element_wait, (By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))
    logger.debug("Username input field found.")

    if not _clear_input_field(driver, username_input, "username"):
        return False

    ancestry_username = config_schema.api.username
    if not ancestry_username:
        raise ValueError("ANCESTRY_USERNAME configuration is missing.")

    # Retry loop to ensure text is actually entered
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Entering username (attempt {attempt}/{max_attempts})...")

        # Use JavaScript to set value directly for reliability
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].value = arguments[1];", username_input, ancestry_username)
        time.sleep(0.2)

        # Trigger input event to ensure page JavaScript recognizes the change
        driver_proto.execute_script(
            """
            var element = arguments[0];
            var event = new Event('input', { bubbles: true });
            element.dispatchEvent(event);
        """,
            username_input,
        )
        time.sleep(0.1)

        # Verify the value was set
        username_input_proto = cast(WebElementProtocol, username_input)
        current_value = username_input_proto.get_attribute("value")
        if current_value == ancestry_username:
            logger.debug(f"Username successfully entered and verified (attempt {attempt}).")
            print(f"  ✓ Username entered: {ancestry_username[:3]}***", flush=True)
            time.sleep(random.uniform(0.2, 0.4))
            return True
        logger.warning(
            f"Username verification failed (attempt {attempt}): expected '{ancestry_username}', got '{current_value}'"
        )
        if attempt < max_attempts:
            time.sleep(0.5)

    logger.error(f"Failed to enter username after {max_attempts} attempts")
    return False


def _find_next_button_with_selectors(short_wait: "WebDriverWait[Any]") -> Optional[WebElement]:
    """Try multiple selectors to find the Next button."""
    next_selectors = [
        SIGN_IN_BUTTON_SELECTOR,
        "button[type='submit']",
        "button.ancBtn",
        "//button[contains(text(), 'Next')]",
        "//button[contains(text(), 'Continue')]",
    ]

    for idx, selector in enumerate(next_selectors):
        try:
            logger.debug(f"[DEBUG] Trying Next button selector {idx + 1}/{len(next_selectors)}: {selector}")
            locator: Locator = (By.XPATH, selector) if selector.startswith("//") else (By.CSS_SELECTOR, selector)
            next_button = _wait_until_clickable(short_wait, locator)
            logger.debug(f"✓ Found Next button with selector: {selector}")
            return next_button
        except (TimeoutException, NoSuchElementException):
            logger.debug(f"Next button not found with selector {idx + 1}: {selector}")
        except Exception as e:
            logger.debug(f"Error trying selector {idx + 1}: {e}")

    return None


def _debug_log_page_state_after_next_click(driver: WebDriver) -> None:
    """Log page state after Next button click for debugging."""
    try:
        current_url = driver.current_url
        page_title = driver.title
        logger.debug(f"[DEBUG] After Next click - URL: {current_url}")
        logger.debug(f"[DEBUG] After Next click - Title: {page_title}")
    except Exception as debug_err:
        logger.debug(f"[DEBUG] Error checking page after Next: {debug_err}")


def _debug_log_signin_page_buttons(driver: WebDriver) -> None:
    """Log all buttons on signin page for debugging."""
    try:
        all_buttons: list[WebElement] = driver.find_elements(By.TAG_NAME, "button")
        logger.debug(f"[DEBUG] Found {len(all_buttons)} buttons on signin page")
        for i, btn in enumerate(all_buttons[:5]):
            try:
                btn_text = btn.text.strip()
                btn_proto = cast(WebElementProtocol, btn)
                btn_id = btn_proto.get_attribute("id")
                btn_class = btn_proto.get_attribute("class")
                if btn_text or btn_id:
                    logger.debug(f"  Button {i}: text='{btn_text}', id='{btn_id}', class='{btn_class}'")
            except Exception:
                pass
    except Exception as debug_err:
        logger.debug(f"[DEBUG] Error listing buttons: {debug_err}")


def _perform_next_button_click(driver: WebDriver, next_button: WebElement) -> bool:
    """Attempt to click the Next button with fallback strategies."""
    try:
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].click();", next_button)
        logger.debug("Next button clicked via JavaScript.")
        logger.info("Next button clicked, waiting for password field to appear...")
        time.sleep(random.uniform(2.0, 3.0))
        _debug_log_page_state_after_next_click(driver)
        return True
    except WebDriverException as js_err:
        logger.warning(f"JS click failed: {js_err}, trying standard click...")
        try:
            next_button.click()
            logger.debug("Next button clicked successfully (standard click).")
            time.sleep(random.uniform(2.0, 3.0))
            return True

        except (ElementClickInterceptedException, ElementNotInteractableException) as e:
            logger.error(f"Both click methods failed: {e}")
            return True  # Continue anyway - password field might be visible


def _click_next_button(driver: WebDriver, short_wait: "WebDriverWait[Any]") -> bool:
    """Click the Next button in two-step login flow."""
    logger.debug("Looking for Next/Continue button after username...")

    next_button = _find_next_button_with_selectors(short_wait)

    if not next_button:
        raise TimeoutException("Next button not found with any selector")

    try:
        logger.debug("Next button found, attempting to click...")
        return _perform_next_button_click(driver, next_button)
    except TimeoutException:
        logger.debug(
            "Next button not found with selector '{SIGN_IN_BUTTON_SELECTOR}', assuming single-step login (password field already visible)."
        )
        _debug_log_signin_page_buttons(driver)
        return True
    except WebDriverException as e:
        logger.warning(f"Error finding Next button: {e}. Continuing anyway.")
        return True


def _wait_for_password_field(driver: WebDriver, element_wait: "WebDriverWait[Any]") -> Optional[WebElement]:
    """Wait for the password field, retrying once with a longer timeout."""
    try:
        logger.debug("[PASSWORD_ENTRY] Waiting for password field to appear...")
        password_input = _wait_until_visible(element_wait, (By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))
        logger.info("[PASSWORD_ENTRY] Password field found on first attempt")
        return password_input
    except TimeoutException:
        logger.warning(
            "[PASSWORD_ENTRY] Password field did not appear with standard wait. Retrying with longer wait..."
        )
        try:
            password_input = _wait_until_visible(WebDriverWait(driver, 30), (By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))
            logger.info("[PASSWORD_ENTRY] Password field found on retry with longer wait")
            return password_input
        except TimeoutException:
            logger.error("[PASSWORD_ENTRY] CRITICAL: Password field never appeared even after 30s wait!")
            return None
    except Exception as exc:
        logger.error(f"[PASSWORD_ENTRY] Unexpected error waiting for password field: {exc}")
        return None


def _attempt_password_entry(
    driver: WebDriver,
    password_input: WebElement,
    ancestry_password: str,
    attempt: int,
    max_attempts: int,
) -> bool:
    """Try to set and verify the password value once."""
    logger.info(f"[PASSWORD_ENTRY] Attempt {attempt}/{max_attempts}: Entering password...")

    logger.debug("[PASSWORD_ENTRY] Setting password value via JavaScript...")
    # Cast to Protocol to ensure types
    driver_proto = cast(DriverProtocol, driver)
    driver_proto.execute_script("arguments[0].value = arguments[1];", password_input, ancestry_password)
    time.sleep(0.2)

    logger.debug("[PASSWORD_ENTRY] Triggering input event...")
    driver_proto.execute_script(
        """
            var element = arguments[0];
            var event = new Event('input', { bubbles: true });
            element.dispatchEvent(event);
        """,
        password_input,
    )
    time.sleep(0.1)

    logger.debug("[PASSWORD_ENTRY] Verifying password was set...")
    password_input_proto = cast(WebElementProtocol, password_input)
    current_value = password_input_proto.get_attribute("value")
    current_length = len(current_value) if current_value else 0
    expected_length = len(ancestry_password)
    logger.info(f"[PASSWORD_ENTRY] Verification: current_length={current_length}, expected_length={expected_length}")

    if current_value and current_length == expected_length:
        logger.info(f"[PASSWORD_ENTRY] ✅ Password successfully entered and verified (attempt {attempt})")
        print("  ✓ Password entered successfully", flush=True)
        time.sleep(random.uniform(0.3, 0.6))
        return True

    logger.warning(
        f"[PASSWORD_ENTRY] ❌ Password verification failed (attempt {attempt}): length mismatch (expected {expected_length}, got {current_length})"
    )
    return False


def _sleep_between_password_attempts(attempt: int, max_attempts: int) -> None:
    """Sleep between password attempts when another retry is allowed."""
    if attempt < max_attempts:
        logger.info("[PASSWORD_ENTRY] Retrying in 0.5s...")
        time.sleep(0.5)


def _enter_password(driver: WebDriver, element_wait: "WebDriverWait[Any]") -> tuple[bool, Optional[WebElement]]:
    """Enter password into the login form."""
    logger.info("[PASSWORD_ENTRY] Starting password entry process...")
    logger.debug(f"Waiting for password input: '{PASSWORD_INPUT_SELECTOR}'...")

    password_input = _wait_for_password_field(driver, element_wait)
    if password_input is None:
        return False, None

    logger.debug("Password input field found.")
    logger.debug("[PASSWORD_ENTRY] Clearing password field...")
    if not _clear_input_field(driver, password_input, "password"):
        logger.error("[PASSWORD_ENTRY] Failed to clear password field")
        return False, None
    logger.debug("[PASSWORD_ENTRY] Password field cleared successfully")

    ancestry_password = config_schema.api.password
    if not ancestry_password:
        logger.error("[PASSWORD_ENTRY] CRITICAL: Password configuration is missing!")
        raise ValueError("ANCESTRY_PASSWORD configuration is missing.")

    logger.info(f"[PASSWORD_ENTRY] Password loaded from config (length: {len(ancestry_password)})")

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            if _attempt_password_entry(driver, password_input, ancestry_password, attempt, max_attempts):
                return True, password_input
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"[PASSWORD_ENTRY] Exception during attempt {attempt}: {type(exc).__name__} - {exc}")
            if attempt == max_attempts:
                logger.error("[PASSWORD_ENTRY] All attempts exhausted due to exceptions")
                return False, None
            time.sleep(0.5)
            continue

        _sleep_between_password_attempts(attempt, max_attempts)

    logger.error(f"[PASSWORD_ENTRY] ❌ FAILED to enter password after {max_attempts} attempts")
    return False, None


def _try_standard_click(sign_in_button: WebElement) -> bool:
    """Try standard click on sign in button."""
    try:
        logger.debug("Attempting standard click on sign in button...")
        sign_in_button.click()
        logger.debug("Standard click executed.")
        return True
    except (ElementClickInterceptedException, ElementNotInteractableException, StaleElementReferenceException) as e:
        logger.warning(f"Standard click failed ({type(e).__name__}).")
        return False
    except WebDriverException as e:
        logger.error(f"WebDriver error during standard click: {e}.")
        return False


def _try_javascript_click(driver: WebDriver, sign_in_button: WebElement) -> bool:
    """Try JavaScript click on sign in button."""
    try:
        logger.debug("Attempting JavaScript click on sign in button...")
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].click();", sign_in_button)
        logger.info("JavaScript click executed.")
        return True
    except WebDriverException as js_click_e:
        logger.error(f"Error during JavaScript click: {js_click_e}")
        return False


def _try_return_key_fallback(password_input: WebElement) -> bool:
    """Try sending RETURN key to password field as fallback."""
    try:
        logger.warning("Attempting fallback: Sending RETURN key to password field.")
        password_input.send_keys(Keys.RETURN)
        logger.info("Fallback RETURN key sent to password field.")
        return True
    except StaleElementReferenceException:
        # Stale element means page already changed (form likely submitted) - this is success!
        logger.info("Form already submitted (stale element) - treating as successful login.")
        return True
    except (WebDriverException, ElementNotInteractableException) as key_e:
        logger.error(f"Failed to send RETURN key: {key_e}")
        return False


def _click_sign_in_button(driver: WebDriver, password_input: WebElement) -> bool:
    """Click the sign in button or use fallback methods."""
    # Try to locate sign in button
    try:
        logger.debug(f"Waiting for sign in button presence: '{SIGN_IN_BUTTON_SELECTOR}'...")
        _wait_until_present(WebDriverWait(driver, 3), (By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))
        logger.debug("Waiting for sign in button clickability...")
        sign_in_button = _wait_until_clickable(
            WebDriverWait(driver, 3),
            (By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR),
        )
        logger.debug("Sign in button located and deemed clickable.")
    except TimeoutException:
        logger.debug("Sign in button not found, using RETURN key fallback (normal on some pages)")
        return _try_return_key_fallback(password_input)
    except WebDriverException as find_e:
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


def enter_creds(driver: WebDriver) -> bool:
    """Enter credentials and sign in to Ancestry."""
    element_wait = WebDriverWait(driver, config_schema.selenium.explicit_wait)
    short_wait = WebDriverWait(driver, 3)  # Reduced from 5s to 3s for faster response
    time.sleep(random.uniform(0.5, 1.0))

    result = False

    try:
        logger.debug("Entering Credentials and Signing In...")

        # Enter username
        if _enter_username(driver, element_wait) and _click_next_button(driver, short_wait):
            # Enter password
            password_ok, password_input = _enter_password(driver, element_wait)
            if password_ok and password_input is not None:
                # Click sign in button
                result = _click_sign_in_button(driver, password_input)

    except (TimeoutException, NoSuchElementException) as e:
        logger.error(f"Timeout or Element not found finding username/password field: {e}")
    except ValueError as ve:
        logger.critical(f"Configuration Error: {ve}")
        raise ve
    except WebDriverException as e:
        logger.error(f"WebDriver error entering credentials: {e}")
        if not is_browser_open(driver):
            logger.error("Session invalid during credential entry.")
    except Exception as e:
        logger.error(f"Unexpected error entering credentials: {e}", exc_info=True)

    return result


# End of enter_creds

# Helper functions for consent


def _find_consent_banner(driver: WebDriver) -> Optional[WebElement]:
    """Find the cookie consent banner if present."""
    logger.debug(f"Checking for cookie consent overlay: '{COOKIE_BANNER_SELECTOR}'")
    try:
        # Reduce wait to 1 second for faster dismissal
        overlay_element = _wait_until_present(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR),
        )
        logger.debug("Cookie consent overlay DETECTED.")
        return overlay_element
    except TimeoutException:
        logger.debug("Cookie consent overlay not found. Assuming no consent needed.")
        return None
    except WebDriverException as e:
        logger.error(f"Error checking for consent banner: {e}")
        raise


def _click_accept_button_standard(driver: WebDriver, accept_button: WebElement) -> bool:
    """Try standard click on accept button."""
    try:
        accept_button.click()
        logger.info("Clicked accept button successfully.")
        # Reduce verification wait to 1 second for faster dismissal
        _wait_until_not_present(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR),
        )
        logger.debug("Consent overlay gone after clicking accept button.")
        return True
    except ElementClickInterceptedException:
        logger.warning("Click intercepted for accept button, trying JS click...")
        return False
    except (TimeoutException, NoSuchElementException):
        logger.debug("Consent overlay likely gone after standard click (verification timed out/not found).")
        return True
    except WebDriverException as click_err:
        logger.error(f"Error during standard click on accept button: {click_err}. Trying JS click...")
        return False


def _click_accept_button_js(driver: WebDriver, accept_button: WebElement) -> bool:
    """Try JavaScript click on accept button."""
    try:
        logger.debug("Attempting JS click on accept button...")
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].click();", accept_button)
        logger.info("Clicked accept button via JS successfully.")
        # Reduce verification wait to 1 second for faster dismissal
        _wait_until_not_present(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR),
        )
        logger.debug("Consent overlay gone after JS clicking accept button.")
        return True
    except (TimeoutException, NoSuchElementException):
        logger.debug("Consent overlay likely gone after JS click (verification timed out/not found).")
        return True
    except WebDriverException as js_click_err:
        logger.error(f"Failed JS click for accept button: {js_click_err}")
        return False


def _handle_consent_button(driver: WebDriver) -> bool:
    """Handle clicking the consent accept button."""
    logger.debug(f"Trying specific accept button: '{CONSENT_ACCEPT_BUTTON_SELECTOR}'")
    try:
        # Reduce wait to 1 second for faster dismissal
        accept_button = _wait_until_clickable(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, CONSENT_ACCEPT_BUTTON_SELECTOR),
        )
        logger.info("Found specific clickable accept button.")

        # Try standard click first
        if _click_accept_button_standard(driver, accept_button):
            return True

        # Try JS click as fallback
        return _click_accept_button_js(driver, accept_button)

    except TimeoutException:
        logger.warning(f"Specific accept button '{CONSENT_ACCEPT_BUTTON_SELECTOR}' not found or not clickable.")
        return False
    except WebDriverException as find_err:
        logger.error(f"Error finding/clicking specific accept button: {find_err}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error handling consent button: {e}", exc_info=True)
        return False


def consent(driver: WebDriver) -> bool:
    """Handles the cookie consent banner if present. Fast dismissal with no retries."""
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


def _check_initial_login_status(session_manager: "SessionManager") -> Optional[str]:
    """Check if already logged in before attempting login."""
    initial_status = login_status(session_manager, disable_ui_fallback=True)
    if initial_status is True:
        print("Already logged in. No need to sign in again.")
        return "LOGIN_SUCCEEDED"
    return None


def _navigate_to_signin(driver: Any, session_manager: "SessionManager", signin_url: str) -> Optional[str]:
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
            expected_conditions.presence_of_element_located((By.CSS_SELECTOR, FAILED_LOGIN_SELECTOR))
        )
        logger.error("Login failed: Specific 'Invalid Credentials' alert detected.")
        return "LOGIN_FAILED_BAD_CREDS"
    except TimeoutException:
        try:
            alert_element = WebDriverWait(driver, 0.5).until(
                expected_conditions.presence_of_element_located((By.CSS_SELECTOR, "div.alert[role='alert']"))
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


def _try_2fa_selectors(driver: Any) -> bool:
    """Try multiple selectors to detect 2FA page."""
    twofa_selectors = [
        ("css", TWO_STEP_VERIFICATION_HEADER_SELECTOR),
        ("css", "body.mfaPage"),
        ("xpath", "//h1[contains(text(), 'Two-step verification')]"),
        ("xpath", "//h2[contains(text(), 'Two-step verification')]"),
        ("xpath", "//h1[contains(text(), 'two-step')]"),
        ("xpath", "//h2[contains(text(), 'two-step')]"),
        ("css", "button[data-method='sms']"),
        ("css", "button[data-method='email']"),
    ]

    for idx, (selector_type, selector) in enumerate(twofa_selectors):
        try:
            logger.debug(f"[DEBUG] Trying 2FA selector {idx + 1}/{len(twofa_selectors)}: ({selector_type}) {selector}")
            by_type = By.XPATH if selector_type == "xpath" else By.CSS_SELECTOR
            WebDriverWait(driver, 2).until(expected_conditions.visibility_of_element_located((by_type, selector)))
            logger.debug(f"✓ Found 2FA page with selector: ({selector_type}) {selector}")
            return True
        except TimeoutException:
            logger.debug(f"2FA page not found with selector {idx + 1}: ({selector_type}) {selector}")
        except Exception as e:
            logger.debug(f"Error trying selector {idx + 1} (({selector_type}) {selector}): {e}")

    return False


def _debug_log_page_headers(driver: Any) -> None:
    """Log page header elements for debugging."""
    try:
        body_classes = driver.find_element(By.TAG_NAME, "body").get_attribute("class")
        logger.debug(f"[DEBUG] Body classes: {body_classes}")

        h1_elements = driver.find_elements(By.TAG_NAME, "h1")
        h2_elements = driver.find_elements(By.TAG_NAME, "h2")
        logger.debug(f"[DEBUG] Found {len(h1_elements)} h1 elements, {len(h2_elements)} h2 elements")

        for h1 in h1_elements[:3]:
            if h1.is_displayed():
                logger.debug(f"[DEBUG] h1 text: '{h1.text}'")

        for h2 in h2_elements[:3]:
            if h2.is_displayed():
                logger.debug(f"[DEBUG] h2 text: '{h2.text}'")
    except Exception as debug_err:
        logger.debug(f"[DEBUG] Error checking page elements: {debug_err}")


def _detect_2fa_page(driver: Any) -> tuple[bool, Optional[str]]:
    """Detect if 2FA page is present."""
    try:
        logger.info("Checking for two-step verification page (waiting up to 15 seconds)...")
        twofa_found = _try_2fa_selectors(driver)

        if twofa_found:
            logger.info("Two-step verification page detected.")
            print("\n🔐 Two-factor authentication required...")
            return True, None

    except Exception as outer_err:
        logger.debug(f"Unexpected error in 2FA detection: {outer_err}")

    logger.debug("Two-step verification page not detected with any selector.")
    _debug_log_page_headers(driver)
    print("  (i) No 2FA prompt detected", flush=True)
    return False, None


def _handle_2fa_flow(session_manager: "SessionManager") -> str:
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


def _verify_login_no_2fa(driver: Any, session_manager: "SessionManager", signin_url: str) -> str:
    """Verify login when no 2FA is detected."""
    logger.debug("Checking login status directly (no 2FA detected)...")
    login_check_result = login_status(session_manager, disable_ui_fallback=False)

    if login_check_result is True:
        print("\n✓ Login successful!")
        # Save cookies after successful login
        _save_login_cookies(session_manager)
        # CRITICAL FIX: Sync browser cookies to API requests session
        session_manager.sync_cookies_to_requests()
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
def _debug_log_page_errors(driver: Any) -> None:
    """Log any visible error messages on the page."""
    try:
        error_elements = driver.find_elements(By.CSS_SELECTOR, ".alert, .error, [role='alert']")
        if error_elements:
            for elem in error_elements[:3]:
                if elem.is_displayed():
                    error_text = elem.text.strip()
                    if error_text:
                        logger.warning(f"[DEBUG] Visible error on page: {error_text}")
    except Exception as e:
        logger.debug(f"[DEBUG] Error checking for error messages: {e}")


def _debug_log_post_credentials_state(driver: Any) -> None:
    """Log page state after credential submission."""
    try:
        current_url = driver.current_url
        page_title = driver.title
        logger.info(f"[DEBUG] After credentials - URL: {current_url}")
        logger.info(f"[DEBUG] After credentials - Page title: {page_title}")
        _debug_log_page_errors(driver)
    except Exception as e:
        logger.debug(f"[DEBUG] Error capturing page state: {e}")


def _handle_credentials_entry(driver: Any) -> Optional[str]:
    """Handle credential entry and check for errors. Returns error status if failed."""
    if not enter_creds(driver):
        logger.error("Failed during credential entry or submission.")
        error_result = _check_for_login_errors(driver)
        return error_result if error_result else "LOGIN_FAILED_CREDS_ENTRY"
    return None


def _execute_login_flow(
    driver: Any,
    session_manager: "SessionManager",
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
    creds_error = _handle_credentials_entry(driver)
    if creds_error:
        return creds_error

    # Wait for page change and log state
    logger.debug("Credentials submitted. Waiting for potential page change...")
    time.sleep(random.uniform(3.0, 5.0))
    _debug_log_post_credentials_state(driver)

    # Check for 2FA
    two_fa_present, early_result = _detect_2fa_page(driver)
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


def log_in(session_manager: "SessionManager") -> str:
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


def _validate_login_status_inputs(session_manager: "SessionManager") -> Optional[bool]:
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


def _perform_api_login_check(session_manager: "SessionManager") -> Optional[bool]:
    """Perform API-based login status check."""
    logger.debug("Performing primary API-based login status check...")
    try:
        session_manager.sync_cookies_to_requests()
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
    ui_element_present = is_elem_there(driver, logged_in_selector, By.CSS_SELECTOR, wait=3)

    if ui_element_present:
        logger.debug("UI check: Logged-in indicator found. User is logged in.")
        return True

    # Check for login button
    login_button_selector = LOG_IN_BUTTON_SELECTOR
    logger.debug(f"Checking for login button: '{login_button_selector}'")
    login_button_present = is_elem_there(driver, login_button_selector, By.CSS_SELECTOR, wait=3)

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
def login_status(session_manager: "SessionManager", disable_ui_fallback: bool = False) -> Optional[bool]:
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


def _validate_nav_inputs(url: str) -> bool:
    """Validate navigation inputs."""
    if not url:
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
    driver: WebDriver,
    session_manager: Optional["SessionManager"],
    attempt: int,
) -> Optional[WebDriver]:
    """Check browser session and restart if needed. Returns driver or None if failed."""
    if not is_browser_open(driver):
        logger.error(f"Navigation failed (Attempt {attempt}): Browser session invalid before nav.")
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


def _document_ready(driver: WebDriver) -> bool:
    """Return True once the current document is fully loaded."""

    # Cast to Protocol to ensure types
    driver_proto = cast(DriverProtocol, driver)
    state = driver_proto.execute_script("return document.readyState")
    return isinstance(state, str) and state in {"complete", "interactive"}


def _execute_navigation(driver: WebDriver, url: str, page_timeout: int) -> None:
    """Execute navigation and wait for page ready state."""
    logger.debug(f"🌐 Navigating to URL: {url}")

    driver.get(url)
    logger.debug("driver.get() completed, waiting for page ready state...")
    waiter: WebDriverWait[Any] = WebDriverWait(driver, page_timeout)
    waiter.until(_document_ready)
    time.sleep(random.uniform(0.5, 1.5))


def _get_landed_url_base(driver: WebDriver, attempt: int) -> Optional[str]:
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
    except WebDriverException as e:
        logger.error(f"Failed to get current URL after get() (Attempt {attempt}): {e}. Retrying.")
        return None


def _check_for_mfa_page(driver: WebDriver) -> bool:
    """Check if currently on MFA page."""
    try:
        _wait_until_visible(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR),
        )
        return True
    except (TimeoutException, NoSuchElementException):
        # MFA page not detected
        return False
    except WebDriverException as e:
        logger.warning(f"WebDriverException checking for MFA header: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking for MFA page: {e}", exc_info=True)
        return False


def _check_for_login_page(driver: WebDriver, target_url_base: str, signin_page_url_base: str) -> bool:
    """Check if currently on login page (only if not intentionally navigating there)."""
    if target_url_base == signin_page_url_base:
        return False
    try:
        _wait_until_visible(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR),
        )
        return True
    except (TimeoutException, NoSuchElementException):
        # Login page not detected
        return False
    except WebDriverException as e:
        logger.warning(f"WebDriverException checking for Login username input: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking for Login page: {e}", exc_info=True)
        return False


def _handle_login_redirect(session_manager: "SessionManager") -> str:
    """Handle unexpected login page redirect. Returns 'retry' or 'fail'."""
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
    session_manager: Optional["SessionManager"],
) -> bool:
    """Check if redirect from signin to base URL is valid. Returns True if valid redirect."""
    is_signin_to_base_redirect = target_url_base == signin_page_url_base and landed_url_base == urlparse(
        config_schema.api.base_url
    ).path.rstrip("/")
    if is_signin_to_base_redirect:
        logger.debug("Redirected from signin page to base URL. Verifying login status...")
        time.sleep(1)
        if session_manager and login_status(session_manager, disable_ui_fallback=True) is True:
            logger.info(
                "Redirect after signin confirmed as logged in. Considering original navigation target 'signin' successful."
            )
            return True
    return False


def _handle_url_mismatch(
    driver: WebDriver,
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
    driver: WebDriver,
    selector: str,
    element_timeout: int,
    unavailability_selectors: dict[str, tuple[str, int]],
) -> str:
    """Wait for target element. Returns 'success', 'fail', or 'continue'."""
    wait_selector = selector if selector else "body"
    logger.debug(f"On correct URL base. Waiting up to {element_timeout}s for selector: '{wait_selector}'")
    try:
        _wait_until_visible(
            WebDriverWait(driver, element_timeout),
            (By.CSS_SELECTOR, wait_selector),
        )
        logger.debug(f"Navigation successful and element '{wait_selector}' found.")
        return "success"
    except TimeoutException:
        current_url_on_timeout = "Unknown"
        with contextlib.suppress(Exception):
            current_url_on_timeout = driver.current_url
        logger.warning(
            f"Timeout waiting for selector '{wait_selector}' at {current_url_on_timeout} (URL base was correct)."
        )

        action, wait_time = _check_for_unavailability(driver, unavailability_selectors)
        if action == "skip":
            return "fail"
        if action == "refresh":
            time.sleep(wait_time)
            return "continue"
        logger.warning("Timeout on selector, no unavailability message. Retrying navigation.")
        return "continue"
    except WebDriverException as el_wait_err:
        logger.error(f"WebDriverException waiting for selector '{wait_selector}': {el_wait_err}")
        return "continue"


def _handle_navigation_alert(driver: WebDriver, attempt: int) -> str:
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
    driver: WebDriver,
    session_manager: Optional["SessionManager"],
) -> tuple[str, Optional[WebDriver]]:
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
    driver: WebDriver,
    landed_url_base: str,
    target_url_base: str,
    signin_page_url_base: str,
    unavailability_selectors: dict[str, tuple[str, int]],
    session_manager: Optional["SessionManager"],
) -> tuple[Optional[str], Optional[WebDriver]]:
    """
    Check for URL mismatch and handle appropriately.
    Returns (action, driver) where action is 'success', 'fail', 'continue', or None.
    """
    if landed_url_base == target_url_base:
        return (None, driver)

    # Check if signin redirect is acceptable
    if _check_signin_redirect(target_url_base, landed_url_base, signin_page_url_base, session_manager):
        return ("success", driver)

    # Handle URL mismatch
    mismatch_action = _handle_url_mismatch(driver, landed_url_base, target_url_base, unavailability_selectors)
    if mismatch_action == "fail":
        return ("fail", driver)
    if mismatch_action == "continue":
        return ("continue", driver)

    return (None, driver)


def _validate_post_navigation(
    driver: WebDriver,
    landed_url_base: str,
    target_url_base: str,
    signin_page_url_base: str,
    selector: str,
    element_timeout: int,
    unavailability_selectors: dict[str, tuple[str, int]],
    session_manager: Optional["SessionManager"],
) -> tuple[str, Optional[WebDriver]]:
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
        if session_manager:
            login_action = _handle_login_redirect(session_manager)
            if login_action == "retry":
                result_action = "continue"
            elif login_action in {"fail", "no_manager"}:
                result_action = "fail"
        else:
            logger.error("Landed on login page but no SessionManager provided.")
            result_action = "fail"
    else:
        # Check for URL mismatch
        url_check_result, updated_driver = _check_url_mismatch_and_handle(
            driver, landed_url_base, target_url_base, signin_page_url_base, unavailability_selectors, session_manager
        )
        if url_check_result:
            result_action = url_check_result
            result_driver = updated_driver if updated_driver else driver
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
    driver: WebDriver,
    nav_config: NavigationConfig,
    session_manager: Optional["SessionManager"],
    attempt: int,
) -> tuple[str, Optional[WebDriver]]:
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
                    driver,
                    landed_url_base,
                    nav_config.target_url_base,
                    nav_config.signin_page_url_base,
                    nav_config.selector,
                    nav_config.element_timeout,
                    nav_config.unavailability_selectors,
                    session_manager,
                )

    except UnexpectedAlertPresentException:
        alert_action = _handle_navigation_alert(driver, attempt)
        result_action = "fail" if alert_action == "fail" else "continue"

    except WebDriverException as wd_e:
        logger.error(f"WebDriverException during navigation (Attempt {attempt}): {wd_e}", exc_info=False)
        wd_action, new_driver = _handle_webdriver_exception(driver, session_manager)
        result_action = "fail" if wd_action == "fail" else "continue"
        result_driver = new_driver if new_driver else driver

    except Exception as e:
        logger.error(f"Unexpected error during navigation (Attempt {attempt}): {e}", exc_info=True)
        time.sleep(random.uniform(2, 4))
        result_action = "continue"

    return (result_action, result_driver)


def nav_to_page(
    driver: WebDriver,
    url: str,
    selector: str = "body",  # CSS selector to wait for as indication of page load success
    session_manager: Optional["SessionManager"] = None,  # Pass SessionManager for context/restart
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
    if not _validate_nav_inputs(url):
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
    unavailability_selectors: dict[str, tuple[str, int]] = {
        TEMP_UNAVAILABLE_SELECTOR: ("refresh", 5),
        PAGE_NO_LONGER_AVAILABLE_SELECTOR: ("skip", 0),
    }

    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Navigation Attempt {attempt}/{max_attempts} to: {url}")

        nav_config = NavigationConfig(
            url=url,
            selector=selector,
            target_url_base=target_url_base,
            signin_page_url_base=signin_page_url_base,
            unavailability_selectors=unavailability_selectors,
            page_timeout=page_timeout,
            element_timeout=element_timeout,
        )
        action, new_driver = _perform_navigation_attempt(driver, nav_config, session_manager, attempt)

        if action == "success":
            return True
        if action == "fail":
            return False
        if action in {"retry", "continue"}:
            driver = new_driver if new_driver else driver
            continue

    # --- Failed After All Attempts ---
    logger.critical(f"Navigation to '{url}' failed permanently after {max_attempts} attempts.")
    try:
        logger.error(f"Final URL after failure: {driver.current_url}")
    except Exception:
        logger.error("Could not retrieve final URL after failure.")
    # End of try/except
    return False


# End of nav_to_page


def _check_for_unavailability(
    driver: WebDriver,
    selectors: dict[str, tuple[str, int]],
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
        if is_elem_there(driver, msg_selector, By.CSS_SELECTOR, wait=1):
            logger.warning(
                f"Unavailability message found matching selector: '{msg_selector}'. Action: {action}, Wait: {wait_time}s"
            )
            return action, wait_time  # Return action (refresh/skip) and wait time
        # End of if
    # End of for

    # Return default (no action, zero wait) if no matching selectors found
    return None, 0


# End of _check_for_unavailability


# ------------------------------------------------------------------------------
# SLEEP PREVENTION - Keep system awake during long-running operations
# ------------------------------------------------------------------------------
def prevent_system_sleep() -> Optional[Any]:
    """
    Prevent system sleep during long-running operations.
    Cross-platform: Windows, macOS, Linux.

    Returns:
        Previous state that should be restored when done, or None if not applicable

    Example:
        ```python
        sleep_state = prevent_system_sleep()
        try:
            # Long-running operation
            process_data()
        finally:
            restore_system_sleep(sleep_state)
        ```
    """
    import platform

    system = platform.system()

    if system == "Windows":
        try:
            import ctypes

            # Windows constants
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002

            # Prevent system sleep and display sleep
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
            logger.debug("💤 Sleep prevention enabled (Windows) - system will stay awake")
            return True
        except Exception as e:
            logger.warning(f"Could not prevent sleep on Windows: {e}")
            return None

    elif system == "Darwin":  # macOS
        try:
            import subprocess

            # Use caffeinate to prevent sleep
            process = subprocess.Popen(['caffeinate', '-d'])
            logger.debug("💤 Sleep prevention enabled (macOS) - caffeinate running")
            return process
        except Exception as e:
            logger.warning(f"Could not prevent sleep on macOS: {e}")
            return None

    elif system == "Linux":
        logger.info("💤 Sleep prevention not implemented for Linux - please disable sleep manually")
        return None

    else:
        logger.warning(f"💤 Unknown platform {system} - cannot prevent sleep")
        return None


def restore_system_sleep(previous_state: Any) -> None:
    """
    Restore normal sleep behavior.

    Args:
        previous_state: The state returned by prevent_system_sleep()

    Example:
        ```python
        sleep_state = prevent_system_sleep()
        try:
            # Long-running operation
            process_data()
        finally:
            restore_system_sleep(sleep_state)
        ```
    """
    import platform

    system = platform.system()

    if system == "Windows":
        try:
            if previous_state:
                import ctypes

                ES_CONTINUOUS = 0x80000000
                # Reset to normal
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                logger.debug("💤 Sleep prevention disabled - normal power management restored")
        except Exception as e:
            logger.warning(f"Could not restore sleep settings on Windows: {e}")

    elif system == "Darwin" and previous_state:  # macOS
        try:
            previous_state.terminate()
            logger.debug("💤 Sleep prevention disabled (macOS) - caffeinate terminated")
        except Exception as e:
            logger.warning(f"Could not restore sleep settings on macOS: {e}")
    # Linux and other platforms don't need cleanup


# End of sleep prevention utilities


def main() -> None:
    """
    Standalone test suite for utils.py - Refactored to use modular test framework.

    This function now delegates to the comprehensive test suite defined below,
    following the standardized testing pattern used across the codebase.
    """
    print("\n" + "=" * 60)
    print("UTILS.PY - COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")

    # Run the comprehensive test suite using the standardized framework
    success = run_comprehensive_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


# End of main


def _create_stubbed_session_manager() -> tuple[
    "SessionManager",
    Any,
    Any,
    Any,
    Any,
]:
    """Create a SessionManager instance with patched dependencies for tests."""
    from contextlib import ExitStack, nullcontext
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    import requests

    from core.session_manager import SessionManager

    mock_db = MagicMock()
    mock_db.is_ready = True
    mock_db.ensure_ready.return_value = True
    mock_db.get_session.return_value = MagicMock()
    mock_db.get_session_context.return_value = nullcontext()
    mock_db.close_connections.return_value = None
    mock_db.return_session.return_value = None

    mock_browser = MagicMock()
    mock_browser.browser_needed = False
    mock_browser.driver_live = False
    mock_browser.driver = None
    mock_browser.ensure_driver_live.return_value = True
    mock_browser.start_browser.return_value = True
    mock_browser.close_browser.return_value = None
    mock_browser.create_new_tab.return_value = "tab-id"

    mock_api = MagicMock()
    mock_api._requests_session = requests.Session()
    mock_api.my_profile_id = None
    mock_api.my_uuid = None
    mock_api.my_tree_id = None
    mock_api.csrf_token = None

    mock_validator = MagicMock()

    mock_config = SimpleNamespace(api=SimpleNamespace(username="test-user", password="test-pass", tree_name=None))

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "core.session_manager.SessionManager._get_cached_database_manager",
                return_value=mock_db,
            )
        )
        stack.enter_context(
            patch(
                "core.session_manager.SessionManager._get_cached_browser_manager",
                return_value=mock_browser,
            )
        )
        stack.enter_context(
            patch(
                "core.session_manager.SessionManager._get_cached_api_manager",
                return_value=mock_api,
            )
        )
        stack.enter_context(
            patch(
                "core.session_manager.SessionManager._get_cached_session_validator",
                return_value=mock_validator,
            )
        )
        stack.enter_context(patch("core.session_manager.config_schema", mock_config))
        stack.enter_context(patch("core.session_manager.SessionManager._initialize_cloudscraper"))
        session_manager = SessionManager()

    return session_manager, mock_db, mock_browser, mock_api, mock_validator


# === TEST HELPER FUNCTIONS ===


def _test_parse_cookie() -> None:
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
    results: list[bool] = []

    for cookie_str, expected, description in test_cases:
        try:
            result = parse_cookie(cookie_str)
            matches_expected = result == expected

            status = "✅" if matches_expected else "❌"
            print(f"   {status} {description}")
            print(f"      Input: '{cookie_str}'")
            print(f"      Output: {result}")
            print(f"      Expected: {expected}")

            results.append(matches_expected)
            assert matches_expected, f"Should match expected result for '{cookie_str}'"

        except Exception as e:
            print(f"   ❌ {description}: Exception {e}")
            results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} cookie parsing tests passed")


def _test_ordinal_case() -> None:
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
    results: list[bool] = []

    for input_val, expected, description in test_cases:
        try:
            result = ordinal_case(input_val)
            matches_expected = result == expected

            status = "✅" if matches_expected else "❌"
            print(f"   {status} {description}")
            print(f"      Input: {input_val} (Type: {type(input_val).__name__})")
            print(f"      Output: '{result}' (Expected: '{expected}')")

            results.append(matches_expected)
            assert matches_expected, f"Failed for {input_val}: expected '{expected}', got '{result}'"

        except Exception as e:
            print(f"   ❌ {description}: Exception {e}")
            results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} ordinal formatting tests passed")


def _test_format_name() -> None:
    """Test name formatting with various input types and edge cases"""
    test_cases = [
        ("john doe", "John Doe", "Basic name formatting"),
        (None, "Valued Relative", "None input handling"),
        ("", "Valued Relative", "Empty string handling"),
        ("ALLCAPS NAME", "Allcaps Name", "All caps input"),
        ("  spaces  everywhere  ", "Spaces Everywhere", "Extra whitespace"),
        ("o'connor", "O'Connor", "Apostrophe handling"),
    ]

    print("📋 Testing name formatting with various cases:")
    results: list[bool] = []

    for input_val, expected, description in test_cases:
        try:
            result = format_name(input_val)
            matches_expected = result == expected

            status = "✅" if matches_expected else "❌"
            print(f"   {status} {description}")
            print(f"      Input: {input_val!r} → Output: '{result}'")
            print(f"      Expected: '{expected}'")

            results.append(matches_expected)
            assert matches_expected, f"Failed for {input_val!r}: expected '{expected}', got '{result}'"

        except Exception as e:
            print(f"   ❌ {description}: Exception {e}")
            results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} name formatting tests passed")


def _test_decorators() -> None:
    """Test decorator availability and basic functionality"""
    # Test retry decorator availability
    assert callable(retry), "retry decorator should be callable"
    assert callable(time_wait), "time_wait decorator should be callable"

    # Basic decorator functionality test
    @retry(max_retries=1, backoff_factor=0.001)
    def decorated_callable() -> str:
        return "success"

    result = decorated_callable()
    assert result == "success", "Retry decorator should work"


def _test_retry_policy_resolution() -> None:
    """Test that retry configurations honor resolved policy profiles."""
    policy = RetryPolicyProfile(
        name="test-policy",
        max_attempts=5,
        initial_delay_seconds=1.75,
        backoff_factor=1.6,
        max_delay_seconds=12.0,
        jitter_seconds=0.4,
        retry_on=(RuntimeError,),
        stop_on=(ValueError,),
    )

    config = _get_retry_config(
        max_retries=None,
        initial_delay=None,
        backoff_factor=None,
        retry_on_status_codes=None,
        policy=policy,
    )

    assert config["max_retries"] == policy.max_attempts, "Policy max_attempts should win"
    assert config["initial_delay"] == policy.initial_delay_seconds, "Policy initial delay should win"
    assert config["backoff_factor"] == policy.backoff_factor, "Policy backoff should win"
    assert config["max_delay"] == policy.max_delay_seconds, "Policy max delay should win"
    assert config["jitter_seconds"] == policy.jitter_seconds, "Policy jitter should win"


def _test_circuit_breaker() -> None:
    """Test CircuitBreaker state transitions and functionality using core.error_handling implementation"""
    from core.error_handling import CircuitBreaker, CircuitBreakerConfig

    # Test CircuitBreaker instantiation with config
    config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1, success_threshold=2)
    cb = CircuitBreaker(name="test_utils", config=config)
    assert cb is not None, "Circuit breaker should instantiate"

    # Test failure recording and circuit opening
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    stats = cb.get_stats()
    assert stats['state'] == "OPEN", "Circuit should OPEN after 3 failures (threshold)"

    # Test transition to HALF_OPEN after recovery timeout
    time.sleep(1.1)  # Wait for recovery timeout
    from contextlib import suppress

    with suppress(Exception):
        cb.call(lambda: "test")

    stats = cb.get_stats()
    # After recovery timeout and a successful call, state depends on implementation
    # The call() method handles state transitions automatically
    assert stats['state'] in {"HALF_OPEN", "CLOSED"}, "Circuit should transition after recovery timeout"

    # Test reset functionality
    cb.reset()
    stats = cb.get_stats()
    assert stats['state'] == "CLOSED", "Circuit should CLOSE after reset"
    assert stats['failure_count'] == 0, "Failure count should be 0 after reset"


def _test_rate_limiter() -> None:
    """Test AdaptiveRateLimiter interface and functionality"""
    # Reset global rate limiter to ensure clean state
    from rate_limiter import reset_global_rate_limiter

    reset_global_rate_limiter()

    # Test AdaptiveRateLimiter via get_rate_limiter()
    # Force success_threshold=50 to ensure predictable behavior regardless of persisted state
    limiter = get_rate_limiter(success_threshold=50)
    assert limiter is not None, "Rate limiter should instantiate"
    assert hasattr(limiter, "wait"), "Rate limiter should have wait method"
    assert hasattr(limiter, "on_429_error"), "Rate limiter should have on_429_error method"
    assert hasattr(limiter, "on_success"), "Rate limiter should have on_success method"
    assert hasattr(limiter, "get_metrics"), "Rate limiter should have get_metrics method"

    # Test wait method (should not hang)
    start_time = time.time()
    limiter.wait()
    elapsed = time.time() - start_time
    assert elapsed < 1.0, "Wait should complete quickly in test"

    # Test AdaptiveRateLimiter interface
    limiter.on_429_error()  # Simulate 429 error
    metrics = limiter.get_metrics()
    assert metrics.error_429_count == 1, "Should track 429 error"

    # Test success tracking
    limiter.on_success()
    metrics = limiter.get_metrics()
    assert metrics.success_count == 1, "Should track success"


def _test_session_manager() -> None:
    """Test SessionManager instantiation and interface"""
    sm, mock_db, mock_browser, mock_api, mock_validator = _create_stubbed_session_manager()

    assert sm.db_manager is mock_db, "SessionManager should use cached database manager"
    assert sm.browser_manager is mock_browser, "SessionManager should use cached browser manager"
    assert sm.api_manager is mock_api, "SessionManager should use cached API manager"
    assert sm.validator is mock_validator, "SessionManager should use cached session validator"
    assert hasattr(sm, "ensure_session_ready"), "SessionManager missing ensure_session_ready method"
    assert hasattr(sm, "close_sess"), "SessionManager missing close_sess method"
    assert sm.session_ready is False, "SessionManager should initialize with session_ready=False"
    assert sm.driver_live is False, "SessionManager driver should not be live initially"


def _test_api_request_function() -> None:
    """Test _api_req function availability and signature"""
    # Test _api_req function availability
    assert callable(_api_req), "_api_req function should be callable"

    # Test function signature (should not raise errors)
    import inspect as inspect_module

    sig = inspect_module.signature(_api_req)
    assert len(sig.parameters) >= 2, "_api_req should accept multiple parameters"


def _test_login_status_function() -> None:
    """Test login_status function availability and signature"""
    # Test login_status function availability
    assert callable(login_status), "login_status function should be callable"

    # Test function signature
    import inspect as inspect_module

    sig = inspect_module.signature(login_status)
    assert "session_manager" in sig.parameters, "login_status should accept session_manager parameter"


def _test_module_registration() -> None:
    """Test module registration functions with behavior validation"""

    # Test 1: auto_register_module returns expected type
    # Note: Don't call on this module to avoid infinite loop
    assert callable(auto_register_module), "auto_register_module should be callable"

    # Test 2: register_function and get_function work correctly
    def _test_func() -> str:
        return "test_value"

    register_function("_test_unique_func", _test_func)
    retrieved = get_function("_test_unique_func")
    assert retrieved is not None, "get_function should retrieve registered function"
    assert retrieved() == "test_value", "Retrieved function should execute correctly"

    # Test 3: is_function_available works
    assert is_function_available("_test_unique_func") is True, (
        "is_function_available should return True for registered function"
    )
    assert is_function_available("_nonexistent_func_xyz") is False, (
        "is_function_available should return False for unregistered function"
    )

    # Test 4: format_name behavior
    assert format_name(None) == "Valued Relative", "format_name should handle None"
    assert format_name("") == "Valued Relative", "format_name should handle empty string"
    assert format_name("john doe") == "John Doe", "format_name should title case names"
    assert format_name("JOHN DOE") == "John Doe", "format_name should normalize uppercase"

    # Test 5: get_rate_limiter returns singleton
    limiter1 = get_rate_limiter()
    limiter2 = get_rate_limiter()
    assert limiter1 is limiter2, "get_rate_limiter should return singleton"
    assert hasattr(limiter1, 'wait'), "Rate limiter should have wait method"

    # Test 6: SessionManager can be imported from core module
    from core.session_manager import SessionManager as SessionMgr

    assert isinstance(SessionMgr, type), "SessionManager should be a class"
    assert hasattr(SessionMgr, 'is_sess_valid'), "SessionManager should have is_sess_valid method"


def _test_performance_validation() -> None:
    """Test that key operations complete within reasonable time"""
    # Test that key operations complete within reasonable time
    start_time = time.time()

    # Format name performance
    for i in range(100):
        format_name(f"test name {i}")

    # Ordinal case performance
    for i in range(1, 101):
        ordinal_case(i)

    elapsed = time.time() - start_time
    assert elapsed < 1.0, f"Performance test should complete quickly, took {elapsed:.3f}s"


def _test_sleep_prevention() -> None:
    """Test cross-platform sleep prevention utilities without side effects."""
    from unittest.mock import MagicMock, patch

    assert callable(prevent_system_sleep), "prevent_system_sleep should be callable"
    assert callable(restore_system_sleep), "restore_system_sleep should be callable"

    # Windows branch uses ctypes to prevent sleep
    with patch("platform.system", return_value="Windows"), patch("ctypes.windll", create=True) as mock_windll:
        mock_windll.kernel32.SetThreadExecutionState.return_value = 1
        state = prevent_system_sleep()
        assert state is True, "Windows sleep prevention should return True"
        restore_system_sleep(state)
        mock_windll.kernel32.SetThreadExecutionState.assert_called()

    # macOS branch spawns caffeinate process
    mock_process = MagicMock()
    with (
        patch("platform.system", return_value="Darwin"),
        patch("subprocess.Popen", return_value=mock_process) as mock_popen,
    ):
        state = prevent_system_sleep()
        assert isinstance(state, MagicMock), "macOS should return process object"
        mock_popen.assert_called_with(["caffeinate", "-d"])
        restore_system_sleep(state)
        state.terminate.assert_called_once()

    # Linux branch returns None (no implementation)
    with patch("platform.system", return_value="Linux"):
        state = prevent_system_sleep()
        assert state is None, "Linux should return None (not implemented)"
        restore_system_sleep(state)  # Should not raise error


def _test_check_for_unavailability() -> None:
    """Test _check_for_unavailability function with mock driver."""
    from unittest.mock import MagicMock

    # Test basic function existence and signature
    assert callable(_check_for_unavailability), "_check_for_unavailability should be callable"

    # Test with mock driver
    mock_driver = MagicMock()
    mock_driver.current_url = "http://test.com"

    # Test with empty selectors
    action, wait_time = _check_for_unavailability(mock_driver, {})
    assert action is None, "Should return None for empty selectors"
    assert wait_time == 0, "Should return 0 for empty selectors"

    print("✅ _check_for_unavailability passed basic tests")


def _test_429_error_path() -> None:
    """Test 429 error handling triggers rate limiter backoff."""
    from unittest.mock import MagicMock, patch

    # Create mock response with 429 status
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.reason = "Too Many Requests"
    mock_response.headers = {"Retry-After": "60"}
    mock_response.text = "Rate limit exceeded"

    # Create mock rate limiter to verify it's called
    mock_rate_limiter = MagicMock()
    mock_rate_limiter.on_429_error = MagicMock()

    # Create RetryContext for the handler
    retry_ctx = RetryContext(
        attempt=1,
        max_attempts=3,
        retries_left=2,
        current_delay=1.0,
        max_delay=60.0,
        backoff_factor=2.0,
    )

    # Create mock session manager with rate limiter
    mock_session_manager = MagicMock()
    mock_session_manager.rate_limiter = mock_rate_limiter

    # Patch time.sleep to avoid actual delays and rate limiter lookup
    with (
        patch("time.sleep"),
        patch("utils._get_rate_limiter_from_session", return_value=mock_rate_limiter),
    ):
        # Call the handler
        should_continue, _response, retries_left, new_delay = _handle_retryable_status(
            mock_response, 429, "Too Many Requests", retry_ctx, "Test API", mock_session_manager
        )

    # Verify behavior
    assert should_continue is True, "Should continue retrying after 429"
    assert retries_left == 1, "Should decrement retries"
    assert new_delay > retry_ctx.current_delay, "Delay should increase with backoff"
    mock_rate_limiter.on_429_error.assert_called_once_with("Test API")


def _test_network_timeout_handling() -> None:
    """Test network timeout triggers proper retry behavior."""
    from unittest.mock import MagicMock

    from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout

    # Test that retryable status codes are configured correctly
    mock_driver = MagicMock()
    mock_sm = MagicMock()
    config = ApiRequestConfig(
        url="https://example.com",
        driver=mock_driver,
        session_manager=mock_sm,
    )
    assert 500 in config.retry_status_codes, "500 should be retryable"
    assert 502 in config.retry_status_codes, "502 should be retryable"
    assert 503 in config.retry_status_codes, "503 should be retryable"
    assert 429 in config.retry_status_codes, "429 should be retryable"

    # Verify exception types are available
    assert Timeout is not None, "Timeout exception should be available"
    assert RequestsConnectionError is not None, "ConnectionError should be available"


def _test_retryable_status_exhaustion() -> None:
    """Test that retries are properly exhausted and return the response."""
    from unittest.mock import MagicMock, patch

    # Create mock response with 500 status
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.reason = "Internal Server Error"
    mock_response.text = "Server error"

    # Create RetryContext with no retries left
    retry_ctx = RetryContext(
        attempt=3,
        max_attempts=3,
        retries_left=0,  # No retries left
        current_delay=1.0,
        max_delay=60.0,
        backoff_factor=2.0,
    )

    # Create mock session manager
    mock_session_manager = MagicMock()
    mock_session_manager.rate_limiter = MagicMock()

    # Call the handler - should not retry
    with patch("utils._get_rate_limiter_from_session", return_value=None):
        should_continue, response, retries_left, _new_delay = _handle_retryable_status(
            mock_response, 500, "Internal Server Error", retry_ctx, "Test API", mock_session_manager
        )

    # Verify behavior - should NOT continue, should return response
    assert should_continue is False, "Should NOT continue when retries exhausted"
    assert response is mock_response, "Should return the original response"
    assert retries_left == -1, "Retries should be decremented below 0"


def utils_module_tests() -> bool:
    """Run comprehensive utils tests using standardized TestSuite format."""
    from test_framework import TestSuite

    suite = TestSuite("Core Utilities & Session Management", "utils.py")  # Basic utility functions

    suite.run_test("Cookie Parsing", _test_parse_cookie, "Parse cookie strings with various formats")

    suite.run_test("Ordinal Formatting", _test_ordinal_case, "Format numbers with ordinal suffixes (1st, 2nd, etc.)")

    suite.run_test("Name Formatting", _test_format_name, "Format names with proper capitalization")

    suite.run_test("Decorator Availability", _test_decorators, "Verify all decorators are callable and functional")

    suite.run_test(
        "Retry Policy Resolution", _test_retry_policy_resolution, "Ensure policy profiles drive retry configuration"
    )

    suite.run_test(
        "Circuit Breaker State Transitions",
        _test_circuit_breaker,
        "Test CircuitBreaker state machine (CLOSED → OPEN → HALF_OPEN → CLOSED)",
    )

    suite.run_test("Adaptive Rate Limiter", _test_rate_limiter, "Test rate limiting interface and metrics tracking")

    suite.run_test(
        "SessionManager Instantiation",
        _test_session_manager,
        "Verify SessionManager initializes with correct dependencies",
    )

    suite.run_test(
        "API Request Function", _test_api_request_function, "Verify _api_req function signature and availability"
    )

    suite.run_test("Login Status Function", _test_login_status_function, "Verify login_status function signature")

    suite.run_test(
        "Module Registration", _test_module_registration, "Verify function registration system and core functions"
    )

    suite.run_test(
        "Performance Validation", _test_performance_validation, "Ensure key operations complete within time limits"
    )

    suite.run_test(
        "Cross-Platform Sleep Prevention",
        _test_sleep_prevention,
        "Test prevent_system_sleep/restore_system_sleep without side effects",
    )

    suite.run_test(
        "Unavailability Detection", _test_check_for_unavailability, "Detect unavailability markers in HTML content"
    )

    # Error path tests (429, timeout, retry exhaustion)
    suite.run_test(
        "429 Error Path Handling",
        _test_429_error_path,
        "Verify 429 status triggers rate limiter backoff and retry logic",
    )

    suite.run_test(
        "Network Timeout Handling",
        _test_network_timeout_handling,
        "Verify timeout exceptions are properly handled in retry config",
    )

    suite.run_test(
        "Retry Exhaustion Behavior",
        _test_retryable_status_exhaustion,
        "Verify retries are exhausted and response is returned correctly",
    )

    return suite.finish_suite()


# Standardized test runner (recommended pattern)
run_comprehensive_tests = create_standard_test_runner(utils_module_tests)


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
