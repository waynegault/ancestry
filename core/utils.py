#!/usr/bin/env python3

"""
utils.py - Core Session Management, API Requests, General Utilities

Manages Selenium/Requests sessions, handles core API interaction (_api_req),
provides general utilities (decorators, formatting, rate limiting),
and includes login/session verification logic closely tied to SessionManager.
"""

# === SESSION MANAGER IMPORT ===
# Import SessionManager from core module - use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from requests.cookies import RequestsCookieJar

    from core.session_manager import SessionManager
else:
    # Runtime import to avoid circular dependency issues
    SessionManager = None
# === STANDARD LIBRARY IMPORTS ===
import base64  # For make_ube
import binascii  # For make_ube
import contextlib
import json
import logging
import random  # Used by RateLimiter jitter calculations
import re
import time
import uuid  # For make_ube
from collections.abc import (
    Callable,
    Mapping,  # Consolidated typing imports
)
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path  # For cookie persistence
from typing import (
    Any,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import urljoin, urlparse, urlunparse  # urljoin re-exported for action7_inbox.py

# === THIRD-PARTY IMPORTS ===
from requests import Response as RequestsResponse
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

from browser.selenium_utils import DriverProtocol, WebElementProtocol
from core.common_params import NavigationConfig, RetryContext
from core.error_handling import RetryPolicyProfile

# === LOCAL IMPORTS ===
# (Note: Some imports done locally to avoid circular dependencies)
from core.logging_utils import (
    log_action_configuration as _log_action_configuration_impl,
    log_action_status as _log_action_status_impl,
    log_batch_indicator as _log_batch_indicator_impl,
    log_cumulative_counts as _log_cumulative_counts_impl,
    log_final_summary as _log_final_summary_impl,
    log_page_complete as _log_page_complete_impl,
    log_starting_position as _log_starting_position_impl,
)
from core.progress_indicators import create_progress_indicator
from core.protocols import RateLimiterProtocol

# === REFACTORED IMPORTS ===
from core.selenium_utils import (
    wait_until_clickable as _wait_until_clickable_impl,
    wait_until_not_present as _wait_until_not_present_impl,
    wait_until_not_visible as _wait_until_not_visible_impl,
    wait_until_present as _wait_until_present_impl,
    wait_until_visible as _wait_until_visible_impl,
)
from observability.metrics_registry import metrics
from observability.utils import (
    derive_metrics_endpoint as _derive_metrics_endpoint_impl,
    metrics_status_family as _metrics_status_family_impl,
    record_api_metrics as _record_api_metrics_impl,
    resolve_request_duration as _resolve_request_duration_impl,
    sanitize_metric_segment as _sanitize_metric_segment_impl,
)
from testing.test_utilities import create_standard_test_runner

# === MODULE SETUP ===
logger = logging.getLogger(__name__)

# === TYPE ALIASES ===
# Define type aliases
RequestsResponseTypeOptional = RequestsResponse | None
Locator = tuple[str, str]
ApiResponseType = dict[str, Any] | list[Any] | str | bytes | RequestsResponse | None


DriverType = WebDriver | None


# === SELENIUM HELPERS (Delegated) ===
_wait_until_visible = _wait_until_visible_impl
_wait_until_clickable = _wait_until_clickable_impl
_wait_until_present = _wait_until_present_impl
_wait_until_not_visible = _wait_until_not_visible_impl
_wait_until_not_present = _wait_until_not_present_impl


# Type variables for decorators
P = ParamSpec('P')
R = TypeVar('R')
SessionManagerType = Optional["SessionManager"]  # Use string literal for forward reference


_sanitize_metric_segment = _sanitize_metric_segment_impl
_derive_metrics_endpoint = _derive_metrics_endpoint_impl
_metrics_status_family = _metrics_status_family_impl
_resolve_request_duration = _resolve_request_duration_impl
_record_api_metrics = _record_api_metrics_impl


log_action_configuration = _log_action_configuration_impl
log_starting_position = _log_starting_position_impl
log_cumulative_counts = _log_cumulative_counts_impl
log_batch_indicator = _log_batch_indicator_impl


def create_standard_progress_bar(total: int, desc: str = "Processing", unit: str = " item") -> Any:
    """
    Create standardized progress bar using core.progress_indicators.

    Args:
        total: Total number of items to process
        desc: Description for the progress bar
        unit: Unit name for items

    Returns:
        tqdm-compatible progress bar instance
    """
    # Create progress indicator but access the underlying tqdm bar for compatibility
    # with existing code that expects a tqdm object
    indicator = create_progress_indicator(desc, total=total, unit=unit)
    indicator.start()
    return indicator.progress_bar


# === STANDARDIZED LOGGING HELPERS (Delegated) ===
log_action_configuration = _log_action_configuration_impl
log_starting_position = _log_starting_position_impl
log_cumulative_counts = _log_cumulative_counts_impl
log_batch_indicator = _log_batch_indicator_impl
log_page_complete = _log_page_complete_impl
log_final_summary = _log_final_summary_impl
log_action_status = _log_action_status_impl

# === API REQUEST CONFIGURATION ===


@dataclass
class ApiRequestConfig:
    """Configuration for API requests."""

    url: str
    # HTTP method and data
    method: str = "GET"
    data: dict[str, Any] | None = None
    json_data: dict[str, Any] | None = None
    json: dict[str, Any] | None = None

    # Headers and authentication
    headers: dict[str, str] | None = None
    referer_url: str | None = None
    use_csrf_token: bool = True
    add_default_origin: bool = True

    # Request behavior
    timeout: int | None = None
    cookie_jar: Optional["RequestsCookieJar"] = None
    allow_redirects: bool = True
    force_text_response: bool = False

    # Retry configuration
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    retry_status_codes: list[int] | set[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_policy: Union[str, "RetryPolicyProfile"] | None = "api"
    jitter_seconds: float = 0.2

    # Metadata
    api_description: str = "API Call"
    attempt: int = 1
    session_manager: Optional["SessionManager"] = None


# === MODULE CONSTANTS ===
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
from requests import Response as RequestsResponse
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

# ------------------------------------------------------------------------------------
# Helper functions (General Utilities)
# ------------------------------------------------------------------------------------
# Cookie parsing/persistence/sync/UBE functions (extracted to browser/cookie_utils)
from browser.cookie_utils import (
    build_cookie_lookup,
    build_cookie_lookup as _build_cookie_lookup,
    build_ube_payload,
    build_ube_payload as _build_ube_payload,
    cookie_sync_cache,
    cookie_sync_cache as _cookie_sync_cache,
    encode_ube_payload,
    encode_ube_payload as _encode_ube_payload,
    extract_cookie_value,
    extract_cookie_value as _extract_cookie_value,
    get_ancsessionid_cookie,
    get_ancsessionid_cookie as _get_ancsessionid_cookie,
    get_cookie_file_path,
    get_cookie_file_path as _get_cookie_file_path,
    load_login_cookies,
    load_login_cookies_impl,
    load_login_cookies_impl as _load_login_cookies,
    make_ube,
    parse_cookie,
    perform_cookie_sync,
    perform_cookie_sync as _perform_cookie_sync,
    save_login_cookies,
    save_login_cookies as _save_login_cookies,
    should_skip_cookie_sync,
    should_skip_cookie_sync as _should_skip_cookie_sync,
    sync_cookies_for_request,
    sync_cookies_for_request as _sync_cookies_for_request,
    validate_driver_for_sync,
    validate_driver_for_sync as _validate_driver_for_sync,
    validate_driver_session,
    validate_driver_session as _validate_driver_session,
)

# from core_imports import auto_register_module, get_function, get_logger, is_function_available, register_function
# Initialize logger with standardized pattern
# logger = get_logger(__name__)
from browser.css_selectors import *
from browser.selenium_utils import (
    is_browser_open,
    is_elem_there,
)

# --- Local application imports ---
from config import config_schema

# String formatting utilities (extracted to core/string_utils)
from core.string_utils import (
    _format_apostrophe_name,
    _format_hyphenated_name,
    _format_initial,
    _format_mc_mac_prefix,
    _format_name_part,
    _format_number_as_ordinal,
    _format_quoted_nickname,
    _get_ordinal_suffix,
    _remove_gedcom_slashes,
    _title_case_with_lowercase_particles,
    format_name,
    ordinal_case,
)

# ------------------------------
# Decorators (Remain in utils.py)
# ------------------------------


def retry(
    max_retries: int | None = None,
    backoff_factor: float | None = None,
    max_delay: float | None = None,
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


# ------------------------------
# PHASE 3.1: Direct AdaptiveRateLimiter usage
# ------------------------------
def get_rate_limiter(
    initial_fill_rate: float | None = None,
    success_threshold: int | None = None,
    min_fill_rate: float | None = None,
    max_fill_rate: float | None = None,
    capacity: float | None = None,
    endpoint_profiles: dict[str, Any] | None = None,
    rate_limiter_429_backoff: float | None = None,
    rate_limiter_success_factor: float | None = None,
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
    from core.rate_limiter import get_adaptive_rate_limiter

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
    referer_url: str | None = None,
    headers: dict[str, str] | None = None,
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


def _get_user_agent_from_browser(driver: DriverType, api_description: str) -> str | None:
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
    csrf_token = None
    if hasattr(session_manager, "api_manager"):
        csrf_token = session_manager.api_manager.csrf_token
    elif hasattr(session_manager, "csrf_token"):
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

    my_profile_id = None
    if hasattr(session_manager, "api_manager"):
        my_profile_id = session_manager.api_manager.my_profile_id
    elif hasattr(session_manager, "my_profile_id"):
        my_profile_id = session_manager.my_profile_id

    if my_profile_id and api_description not in exclude_userid_for:
        final_headers["ancestry-userid"] = my_profile_id.upper()
    elif api_description in exclude_userid_for and my_profile_id:
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
    if driver and session_manager.browser_manager.driver:
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

# Cookie sync functions re-exported from browser.cookie_utils (see import block above)


def _get_rate_limiter_from_session(
    session_manager: Optional["SessionManager"],
) -> RateLimiterProtocol | None:
    """Return the adaptive rate limiter attached to the session, if any."""
    if session_manager is None:
        return None
    rate_limiter = getattr(session_manager, "rate_limiter", None)
    if rate_limiter is None:
        return None
    return cast(RateLimiterProtocol, rate_limiter)


def _apply_rate_limiting(
    session_manager: Optional["SessionManager"],
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


def _calculate_retry_sleep_time(
    current_delay: float,
    max_delay: float,
    jitter_seconds: float = 0.2,
) -> float:
    """Calculate sleep time for retry with jitter."""
    sleep_time = current_delay
    if jitter_seconds > 0:
        sleep_time += random.uniform(0, jitter_seconds)
    return min(sleep_time, max_delay)


def _prepare_api_request(config: ApiRequestConfig) -> dict[str, Any]:
    """Prepare arguments for requests.Session.request."""
    request_params: dict[str, Any] = {
        "method": config.method,
        "url": config.url,
        "headers": config.headers,
        "timeout": config.timeout,
        "allow_redirects": config.allow_redirects,
    }

    if config.json_data is not None:
        request_params["json"] = config.json_data
    elif config.json is not None:
        request_params["json"] = config.json
    elif config.data is not None:
        request_params["data"] = config.data

    if config.cookie_jar is not None:
        request_params["cookies"] = config.cookie_jar

    return request_params


def _execute_api_request(
    session_manager: Optional["SessionManager"],
    request_params: dict[str, Any],
    api_description: str,
) -> RequestsResponseTypeOptional:
    """Execute the API request using the session manager."""
    if (
        session_manager
        and hasattr(session_manager, "api_manager")
        and hasattr(session_manager.api_manager, "_requests_session")
    ):
        try:
            return session_manager.api_manager._requests_session.request(**request_params)
        except Exception as e:
            logger.error(f"{api_description}: Request failed via APIManager: {e}")
            return None

    # Fallback
    import requests

    try:
        return requests.request(**request_params)
    except Exception as e:
        logger.error(f"{api_description}: Request failed via requests: {e}")
        return None


def _handle_failed_request_response(
    response: RequestsResponseTypeOptional,
    retry_ctx: RetryContext,
    api_description: str,
    session_manager: Optional["SessionManager"],
) -> tuple[bool, int, float]:
    """
    Handle failed request response (status code check).

    Returns:
        Tuple of (should_continue, retries_left, current_delay)
    """
    status = response.status_code if response else 0
    reason = response.reason if response else "Unknown Error"

    # Check if retryable
    is_retryable = False
    if retry_ctx.retry_status_codes:
        is_retryable = status in retry_ctx.retry_status_codes

    if is_retryable:
        should_continue, _, retries_left, current_delay = _handle_retryable_status(
            response, status, reason, retry_ctx, api_description, session_manager
        )
        return should_continue, retries_left, current_delay

    # Non-retryable error
    _handle_error_status(response, status, reason, api_description, session_manager)
    return False, (retry_ctx.retries_left or 0) - 1, retry_ctx.current_delay


def _handle_429_error(
    response: RequestsResponseTypeOptional,
    api_description: str,
    session_manager: Optional["SessionManager"],
) -> None:
    """Handle 429 Too Many Requests error."""
    rate_limiter = _get_rate_limiter_from_session(session_manager)
    if rate_limiter:
        retry_after: float | None = None
        if response and hasattr(response, 'headers'):
            retry_header = response.headers.get("Retry-After")
            if retry_header:
                with contextlib.suppress(ValueError, TypeError):
                    retry_after = float(retry_header)

        rate_limiter.on_429_error(api_description, retry_after=retry_after)


def _handle_retryable_status(
    response: RequestsResponseTypeOptional,
    status: int,
    reason: str,
    retry_ctx: RetryContext,
    api_description: str,
    session_manager: Optional["SessionManager"],
) -> tuple[bool, RequestsResponseTypeOptional | None, int, float]:
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
        _handle_429_error(response, api_description, session_manager)

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
) -> RequestsResponseTypeOptional | None:
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
    session_manager: Optional["SessionManager"],
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
        if session_manager:
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
    session_manager: Optional["SessionManager"],
    force_text_response: bool,
    request_params: dict[str, Any],
    metrics_endpoint: str,
    metrics_method: str,
    attempt_duration: float,
) -> tuple[Any | None, bool, int, float, Exception | None]:
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
            rate_limiter.on_success(api_description)
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


def _api_req(
    session_manager: Optional["SessionManager"],
    url: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    referer_url: str | None = None,
    timeout: int | None = None,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    retry_status_codes: list[int] | set[int] | None = None,
    retry_policy: str | RetryPolicyProfile | None = "api",
    api_description: str = "API Call",
    force_text_response: bool = False,
    use_csrf_token: bool = True,
    add_default_origin: bool = True,
) -> ApiResponseType:
    """
    Execute an API request with retries, rate limiting, and error handling.
    """
    # Prepare headers if session_manager is available
    final_headers = headers or {}
    if session_manager:
        driver = session_manager.browser_manager.driver
        final_headers = _prepare_api_headers(
            session_manager,
            driver,
            api_description,
            final_headers,
            use_csrf_token,
            add_default_origin,
        )

    config = ApiRequestConfig(
        url=url,
        method=method,
        data=data,
        json_data=json_data,
        headers=final_headers,
        referer_url=referer_url,
        timeout=timeout,
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        max_delay=max_delay,
        retry_status_codes=retry_status_codes or [429, 500, 502, 503, 504],
        retry_policy=retry_policy,
        api_description=api_description,
        force_text_response=force_text_response,
        session_manager=session_manager,
        use_csrf_token=use_csrf_token,
        add_default_origin=add_default_origin,
    )
    return _api_req_impl(config)


def _api_req_impl(config: ApiRequestConfig) -> ApiResponseType:
    """Implementation of API request logic using ApiRequestConfig."""
    retries_left = config.max_retries
    current_delay = config.initial_delay

    while True:
        response, should_continue, retries_left, current_delay, _ = _process_request_attempt(
            config, retries_left, current_delay
        )

        if response is not None:
            return response

        if not should_continue:
            return None

        config.attempt += 1


def _process_request_attempt(
    config: ApiRequestConfig,
    retries_left: int,
    current_delay: float,
) -> tuple[Any | None, bool, int, float, Exception | None]:
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

        # Apply rate limiting
        _apply_rate_limiting(config.session_manager, config.api_description, config.attempt)

        response = _execute_api_request(
            session_manager=config.session_manager,
            request_params=request_params,
            api_description=config.api_description,
        )
        attempt_duration = time.perf_counter() - attempt_start
        attempt_duration = _resolve_request_duration(response, attempt_duration)

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

        # Handle failed request (response is None)
        if response is None:
            if config.session_manager:
                should_continue, retries_left, current_delay = _handle_failed_request_response(
                    response,
                    retry_ctx,
                    config.api_description,
                    config.session_manager,
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
                return (None, result_should_continue, retries_left, current_delay, None)
            # Fallback if no session manager
            return (None, False, 0, current_delay, None)
        # Handle response status
        if config.session_manager:
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
        # Fallback if no session manager
        if response.ok:
            return (response, False, retries_left, current_delay, None)
        return (response, False, 0, current_delay, None)

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


# End of _api_req

# UBE helper functions re-exported from browser.cookie_utils (see import block above)

# ----------------------------------------------------------------------------
# Login/authentication & navigation functions (extracted to browser/login, browser/navigation)
# Lazy re-exported to avoid circular imports (browser.login imports core.utils)
# ----------------------------------------------------------------------------
_LAZY_RE_EXPORTS: dict[str, str] = {
    # browser.login
    "_attempt_password_entry": "browser.login",
    "_check_for_login_errors": "browser.login",
    "_check_initial_login_status": "browser.login",
    "_check_ui_login_indicators": "browser.login",
    "_clear_input_field": "browser.login",
    "_click_accept_button_js": "browser.login",
    "_click_accept_button_standard": "browser.login",
    "_click_next_button": "browser.login",
    "_click_sign_in_button": "browser.login",
    "_click_sms_button": "browser.login",
    "_debug_log_page_buttons": "browser.login",
    "_debug_log_page_errors": "browser.login",
    "_debug_log_page_headers": "browser.login",
    "_debug_log_page_state_after_next_click": "browser.login",
    "_debug_log_post_credentials_state": "browser.login",
    "_debug_log_signin_page_buttons": "browser.login",
    "_detect_2fa_page": "browser.login",
    "_enter_password": "browser.login",
    "_enter_username": "browser.login",
    "_execute_login_flow": "browser.login",
    "_find_consent_banner": "browser.login",
    "_find_next_button_with_selectors": "browser.login",
    "_find_sms_button_with_selectors": "browser.login",
    "_handle_2fa_flow": "browser.login",
    "_handle_consent_button": "browser.login",
    "_handle_credentials_entry": "browser.login",
    "_handle_login_exception": "browser.login",
    "_navigate_to_signin": "browser.login",
    "_perform_api_login_check": "browser.login",
    "_perform_next_button_click": "browser.login",
    "_perform_sms_button_click": "browser.login",
    "_perform_ui_login_check_with_navigation": "browser.login",
    "_sleep_between_password_attempts": "browser.login",
    "_try_2fa_selectors": "browser.login",
    "_try_javascript_click": "browser.login",
    "_try_return_key_fallback": "browser.login",
    "_try_standard_click": "browser.login",
    "_validate_login_status_inputs": "browser.login",
    "_verify_2fa_completion": "browser.login",
    "_verify_login_no_2fa": "browser.login",
    "_wait_for_2fa_header": "browser.login",
    "_wait_for_code_input_field": "browser.login",
    "_wait_for_password_field": "browser.login",
    "_wait_for_user_2fa_action": "browser.login",
    "consent": "browser.login",
    "enter_creds": "browser.login",
    "handle_two_fa": "browser.login",
    "log_in": "browser.login",
    "login_status": "browser.login",
    # browser.navigation
    "_check_browser_session": "browser.navigation",
    "_check_for_login_page": "browser.navigation",
    "_check_for_mfa_page": "browser.navigation",
    "_check_for_unavailability": "browser.navigation",
    "_check_signin_redirect": "browser.navigation",
    "_check_url_mismatch_and_handle": "browser.navigation",
    "_document_ready": "browser.navigation",
    "_dump_navigation_debug_artifacts": "browser.navigation",
    "_execute_navigation": "browser.navigation",
    "_get_landed_url_base": "browser.navigation",
    "_handle_login_redirect": "browser.navigation",
    "_handle_navigation_alert": "browser.navigation",
    "_handle_url_mismatch": "browser.navigation",
    "_handle_webdriver_exception": "browser.navigation",
    "_parse_and_normalize_url": "browser.navigation",
    "_perform_navigation_attempt": "browser.navigation",
    "_validate_nav_inputs": "browser.navigation",
    "_validate_post_navigation": "browser.navigation",
    "_wait_for_element": "browser.navigation",
    "nav_to_page": "browser.navigation",
}


def __getattr__(name: str) -> Any:
    """Lazy re-export for functions moved to browser.login and browser.navigation."""
    module_path = _LAZY_RE_EXPORTS.get(name)
    if module_path:
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, name)
        globals()[name] = val  # Cache for subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ------------------------------------------------------------------------------
# SLEEP PREVENTION - Keep system awake during long-running operations
# ------------------------------------------------------------------------------
def prevent_system_sleep() -> Any | None:
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
    from core.rate_limiter import reset_global_rate_limiter

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
    endpoint = "test-api"
    start_time = time.time()
    limiter.wait(endpoint)
    elapsed = time.time() - start_time
    assert elapsed < 1.0, "Wait should complete quickly in test"

    # Test AdaptiveRateLimiter interface - per-endpoint 429 handling
    limiter.on_429_error(endpoint)
    metrics = limiter.get_metrics()
    assert metrics.error_429_count == 1, "Should track 429 error"

    # Test success tracking (per-endpoint)
    limiter.on_success(endpoint)
    state = limiter.get_endpoint_state(endpoint)
    assert state is not None, "Endpoint state should exist"
    assert state.success_count == 1, "Should track success per endpoint"


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


def utils_module_tests() -> bool:
    """Run comprehensive utils tests using standardized TestSuite format."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Core Utilities & Session Management", "utils.py")  # Basic utility functions

    suite.run_test("Decorator Availability", _test_decorators, "Verify all decorators are callable and functional")

    suite.run_test(
        "Circuit Breaker State Transitions",
        _test_circuit_breaker,
        "Test CircuitBreaker state machine (CLOSED → OPEN → HALF_OPEN → CLOSED)",
    )

    suite.run_test("Adaptive Rate Limiter", _test_rate_limiter, "Test rate limiting interface and metrics tracking")

    suite.run_test(
        "Performance Validation", _test_performance_validation, "Ensure key operations complete within time limits"
    )

    suite.run_test(
        "Cross-Platform Sleep Prevention",
        _test_sleep_prevention,
        "Test prevent_system_sleep/restore_system_sleep without side effects",
    )

    return suite.finish_suite()


# Standardized test runner (recommended pattern)
run_comprehensive_tests = create_standard_test_runner(utils_module_tests)


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
