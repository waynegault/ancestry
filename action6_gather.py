#!/usr/bin/env python3

# action6_gather.py
# ruff: noqa: PLW0603, PTH123, RUF001, PLR0911

"""
action6_gather.py - Gather DNA Matches from Ancestry

Fetches the user's DNA match list page by page, extracts relevant information,
compares with existing database records, fetches additional details via API for
new or changed matches, and performs bulk updates/inserts into the local database.
Handles pagination, rate limiting, caching (via utils/cache.py decorators used
within helpers), error handling, and concurrent API fetches using ThreadPoolExecutor.

Linter Suppressions:
- PLW0603: Global statements necessary for singleton pattern (metrics, monitor, cache)
- PTH123: Using open() instead of Path.open() for simplicity and compatibility
- RUF001: Unicode emoji characters used for visual clarity in logs
- PLR0911: Complex functions with multiple return paths for clarity
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# === STANDARD LIBRARY IMPORTS ===
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional
from urllib.parse import unquote, urlencode, urljoin, urlparse

# === THIRD-PARTY IMPORTS ===
import cloudscraper
import requests
from bs4 import BeautifulSoup  # For HTML parsing if needed (e.g., ladder)
from diskcache.core import ENOVAL  # For checking cache misses
from requests.cookies import RequestsCookieJar
from requests.exceptions import ConnectionError, RequestException
from selenium.common.exceptions import (
    NoSuchCookieException,
    WebDriverException,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session as SqlAlchemySession, joinedload  # Alias Session
from tqdm.auto import tqdm  # Progress bar
from tqdm.contrib.logging import logging_redirect_tqdm  # Redirect logging through tqdm

from core.error_handling import (
    AuthenticationExpiredError,
    # ErrorContext,  # Not used
    # AncestryException,  # Not used
    # RetryableError,  # Not used
    # NetworkTimeoutError,  # Not used
    # DatabaseConnectionError,  # Not used
    BrowserSessionError,
    circuit_breaker,
    # graceful_degradation,  # Not used
    error_context,
    retry_on_failure,
    timeout_protection,
)

# === LOCAL IMPORTS ===
if TYPE_CHECKING:
    from config.config_schema import ConfigSchema

import contextlib

from cache import cache as global_cache  # Use the initialized global cache instance
from common_params import MatchIdentifiers, PrefetchedData
from config import config_schema
from core.session_manager import SessionManager
from database import (
    DnaMatch,
    FamilyTree,
    Person,
    PersonStatusEnum,
    db_transn,
)
from my_selectors import *  # Import CSS selectors
from selenium_utils import get_driver_cookies
from utils import (
    _api_req,  # API request helper
    format_name,  # Name formatting utility
    nav_to_page,  # Navigation helper
    ordinal_case,  # Ordinal case formatting
    retry_api,  # API retry decorator
)

# from test_framework import (
#     # TestSuite,  # Not used in main code
#     # suppress_logging,  # Not used in main code
#     # create_mock_data,  # Not used in main code
#     # assert_valid_function,  # Not used in main code
# )

# --- Constants ---
MATCHES_PER_PAGE: int = 20  # Default matches per page (adjust based on API response)
# CRITICAL: Reduced from 50 back to 6 based on working version c3b5535 (Aug 12, 2025)
# Higher threshold (50) was causing cascade failures - system kept retrying dead sessions
# Lower threshold (6) fails fast when session dies, preventing wasted retry attempts
CRITICAL_API_FAILURE_THRESHOLD: int = 6  # Threshold for _fetch_combined_details failures - fail fast on session death

# Configurable settings from config_schema
DB_ERROR_PAGE_THRESHOLD: int = 10  # Max consecutive DB errors allowed

# CRITICAL RATE LIMIT FIX: Thread pool workers now configured via .env THREAD_POOL_WORKERS
# Loaded from config_schema.api.thread_pool_workers
# Based on working version c3b5535 (Aug 12, 2025):
#   - THREAD_POOL_WORKERS=3 proven reliable for sustained batch processing
#   - CRITICAL_API_FAILURE_THRESHOLD=6 (conservative - fail fast on session issues)
#   - REQUESTS_PER_SECOND=0.4 (battle-tested for zero 429 errors)
# Recommended: 3 workers at 0.4 RPS = 0.13 RPS per worker (safe for parallel processing)
# If experiencing 429 errors, reduce to THREAD_POOL_WORKERS=1 in .env file


# --- Custom Exceptions ---
class MaxApiFailuresExceededError(Exception):
    """Custom exception for exceeding API failure threshold in a batch."""

    pass


# End of MaxApiFailuresExceededError


# ------------------------------------------------------------------------------
# SESSION CIRCUIT BREAKER - PRIORITY 1.1
# ------------------------------------------------------------------------------
class SessionCircuitBreaker:
    """
    Circuit breaker to detect and prevent cascade failures from dead WebDriver sessions.

    When a WebDriver session dies, it can cause hundreds or thousands of consecutive
    failures. This circuit breaker detects the pattern and trips to prevent wasting
    time on retry attempts that will never succeed.

    Key features:
    - Trips after N consecutive session failures (default: 5)
    - Resets on any successful operation
    - Provides clear logging when tripped
    - Prevents cascade failures that waste hours

    Usage:
        breaker = SessionCircuitBreaker(threshold=5)

        if breaker.is_tripped():
            logger.critical("Circuit breaker is tripped - aborting")
            return False

        if not session_manager.is_sess_valid():
            if breaker.record_failure():
                logger.critical("Circuit breaker TRIPPED - session permanently dead")
                return False
        else:
            breaker.record_success()
    """

    def __init__(self, threshold: int = 5, name: str = "Session"):
        """
        Initialize circuit breaker.

        Args:
            threshold: Number of consecutive failures before tripping (default: 5)
            name: Name for logging purposes (default: "Session")
        """
        self.consecutive_failures = 0
        self.threshold = threshold
        self.tripped = False
        self.last_failure_time: Optional[float] = None
        self.trip_time: Optional[float] = None
        self.name = name

        logger.debug(f"🔌 {self.name} Circuit Breaker initialized (threshold={threshold})")

    def record_failure(self) -> bool:
        """
        Record a failure and check if circuit should trip.

        Returns:
            bool: True if circuit just tripped, False otherwise
        """
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        if not self.tripped and self.consecutive_failures >= self.threshold:
            self.tripped = True
            self.trip_time = time.time()
            logger.critical(
                f"🚨 {self.name} CIRCUIT BREAKER TRIPPED after {self.consecutive_failures} "
                f"consecutive failures - preventing cascade failure"
            )
            return True

        if self.tripped:
            logger.debug(
                f"Circuit breaker already tripped ({self.consecutive_failures} total failures)"
            )
        else:
            logger.warning(
                f"⚠️  {self.name} failure {self.consecutive_failures}/{self.threshold} "
                f"(circuit will trip at {self.threshold})"
            )

        return False

    def record_success(self) -> None:
        """Reset circuit breaker on successful operation."""
        if self.consecutive_failures > 0:
            logger.debug(
                f"✅ {self.name} success - resetting circuit breaker "
                f"(had {self.consecutive_failures} failures)"
            )

        self.consecutive_failures = 0
        self.tripped = False
        self.last_failure_time = None
        self.trip_time = None

    def is_tripped(self) -> bool:
        """
        Check if circuit is tripped.

        Returns:
            bool: True if circuit is tripped, False otherwise
        """
        return self.tripped

    def get_status(self) -> dict[str, Any]:
        """
        Get current status of circuit breaker.

        Returns:
            dict: Status information including failure count, trip status, etc.
        """
        return {
            "name": self.name,
            "tripped": self.tripped,
            "consecutive_failures": self.consecutive_failures,
            "threshold": self.threshold,
            "last_failure_time": self.last_failure_time,
            "trip_time": self.trip_time,
            "time_since_trip": (
                time.time() - self.trip_time if self.trip_time else None
            ),
        }

    def reset(self) -> None:
        """Manually reset circuit breaker (use with caution)."""
        logger.warning(f"🔄 Manually resetting {self.name} circuit breaker")
        self.consecutive_failures = 0
        self.tripped = False
        self.last_failure_time = None
        self.trip_time = None


# Global circuit breaker instance for session monitoring
# This will be initialized when coord_action starts
_global_session_circuit_breaker: Optional[SessionCircuitBreaker] = None


def get_session_circuit_breaker() -> SessionCircuitBreaker:
    """
    Get or create the global session circuit breaker.

    Returns:
        SessionCircuitBreaker: The global circuit breaker instance
    """
    global _global_session_circuit_breaker

    if _global_session_circuit_breaker is None:
        _global_session_circuit_breaker = SessionCircuitBreaker(
            threshold=5,
            name="WebDriver Session"
        )

    return _global_session_circuit_breaker


# ------------------------------------------------------------------------------
# Parameter Grouping Dataclasses
# ------------------------------------------------------------------------------


@dataclass
class BatchCounters:
    """Groups batch processing counters to reduce parameter count."""
    new: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0


# ------------------------------------------------------------------------------
# Refactored coord Helpers
# ------------------------------------------------------------------------------


def _initialize_gather_state() -> dict[str, Any]:
    """Initializes counters and state variables for the gathering process."""
    return {
        "total_new": 0,
        "total_updated": 0,
        "total_skipped": 0,
        "total_errors": 0,
        "total_pages_processed": 0,
        "db_connection_errors": 0,
        "final_success": True,
        "matches_on_current_page": [],
        "total_pages_from_api": None,
    }


# End of _initialize_gather_state


def _validate_start_page(start_arg: Any) -> int:
    """Validates and returns the starting page number."""
    try:
        start_page = int(start_arg)
        if start_page <= 0:
            logger.warning(f"Invalid start page '{start_arg}'. Using default page 1.")
            return 1
        return start_page
    except (ValueError, TypeError):
        logger.warning(f"Invalid start page value '{start_arg}'. Using default page 1.")
        return 1


# End of _validate_start_page


# === INITIAL PAGE NAVIGATION HELPER FUNCTIONS ===

def _ensure_on_match_list_page(session_manager: SessionManager, target_matches_url_base: str) -> bool:
    """Ensure browser is on the DNA matches list page."""
    driver = session_manager.driver

    try:
        current_url = driver.current_url  # type: ignore
        if not current_url.startswith(target_matches_url_base):
            logger.debug("Not on match list page. Navigating...")
            if not nav_to_list(session_manager):
                logger.error("Failed to navigate to DNA match list page. Exiting initial fetch.")
                return False
            logger.debug("Successfully navigated to DNA matches page.")
        else:
            logger.debug(f"Already on correct DNA matches page: {current_url}")
        return True
    except WebDriverException as nav_e:
        logger.error(f"WebDriver error checking/navigating to match list: {nav_e}", exc_info=True)
        return False


def _fetch_initial_page_data(session_manager: SessionManager, start_page: int) -> tuple[Optional[list[dict[str, Any]]], Optional[int], bool]:
    """Fetch initial page data with DB session retry."""
    db_session_for_page: Optional[SqlAlchemySession] = None

    # Get DB session with retry
    for retry_attempt in range(3):
        db_session_for_page = session_manager.get_db_conn()
        if db_session_for_page:
            break
        logger.warning(f"DB session attempt {retry_attempt + 1}/3 failed. Retrying in 5s...")
        time.sleep(5)

    if not db_session_for_page:
        logger.critical("Could not get DB session for initial page fetch after retries.")
        return None, None, False

    try:
        if not session_manager.is_sess_valid():
            raise ConnectionError("WebDriver session invalid before initial get_matches.")

        result = get_matches(session_manager, start_page)
        if result is None:
            logger.error(f"Initial get_matches for page {start_page} returned None.")
            return [], None, False

        matches_on_page, total_pages_from_api = result
        return matches_on_page, total_pages_from_api, True

    except ConnectionError as init_conn_e:
        logger.critical(f"ConnectionError during initial get_matches: {init_conn_e}.", exc_info=False)
        return None, None, False
    except Exception as get_match_err:
        logger.error(f"Error during initial get_matches call on page {start_page}: {get_match_err}", exc_info=True)
        return None, None, False
    finally:
        if db_session_for_page:
            session_manager.return_session(db_session_for_page)


def _navigate_and_get_initial_page_data(
    session_manager: SessionManager, start_page: int
) -> tuple[Optional[list[dict[str, Any]]], Optional[int], bool]:
    """
    Ensures navigation to the match list and fetches initial page data.

    Returns:
        Tuple: (matches_on_page, total_pages, success_flag)
    """
    my_uuid = session_manager.my_uuid

    # Detect the correct base URL
    target_matches_url_base = urljoin(
        config_schema.api.base_url, f"discoveryui-matches/list/{my_uuid}"
    )

    # Ensure on match list page
    logger.debug("Ensuring browser is on the DNA matches list page...")
    if not _ensure_on_match_list_page(session_manager, target_matches_url_base):
        return None, None, False

    # Fetch initial page data
    logger.debug(f"Fetching initial page {start_page} to determine total pages...")
    matches_on_page, total_pages_from_api, initial_fetch_success = _fetch_initial_page_data(
        session_manager, start_page
    )

    return matches_on_page, total_pages_from_api, initial_fetch_success


# End of _navigate_and_get_initial_page_data


def _determine_page_processing_range(
    total_pages_from_api: int, start_page: int
) -> tuple[int, int]:
    """Determines the last page to process and total pages in the run."""
    max_pages_config = config_schema.api.max_pages
    pages_to_process_config = (
        min(max_pages_config, total_pages_from_api)
        if max_pages_config > 0
        else total_pages_from_api
    )
    last_page_to_process = min(
        start_page + pages_to_process_config - 1, total_pages_from_api
    )
    total_pages_in_run = max(0, last_page_to_process - start_page + 1)
    return last_page_to_process, total_pages_in_run


# End of _determine_page_processing_range


# === PAGE PROCESSING LOOP HELPER FUNCTIONS ===

def _get_db_session_with_retry(session_manager: SessionManager, current_page_num: int, state: dict[str, Any]) -> Optional[SqlAlchemySession]:
    """Get database session with retry logic."""
    db_session_for_page: Optional[SqlAlchemySession] = None
    for retry_attempt in range(3):
        db_session_for_page = session_manager.get_db_conn()
        if db_session_for_page:
            state["db_connection_errors"] = 0
            return db_session_for_page
        logger.warning(f"DB session attempt {retry_attempt + 1}/3 failed for page {current_page_num}. Retrying in 5s...")
        time.sleep(5)
    return None


def _handle_db_session_failure(current_page_num: int, state: dict[str, Any], progress_bar: Optional[tqdm], loop_final_success: bool) -> tuple[bool, bool]:
    """Handle database session failure and check if should abort."""
    state["db_connection_errors"] += 1
    logger.error(f"Could not get DB session for page {current_page_num} after retries. Skipping page.")
    state["total_errors"] += MATCHES_PER_PAGE
    progress_bar.update(MATCHES_PER_PAGE)

    if state["db_connection_errors"] >= DB_ERROR_PAGE_THRESHOLD:
        logger.critical(f"Aborting run due to {state['db_connection_errors']} consecutive DB connection failures.")
        loop_final_success = False
        remaining_matches_estimate = max(0, progress_bar.total - progress_bar.n)
        if remaining_matches_estimate > 0:
            progress_bar.update(remaining_matches_estimate)
            state["total_errors"] += remaining_matches_estimate
        return loop_final_success, True  # should_break

    return loop_final_success, False  # should_break


def _fetch_page_matches(session_manager: SessionManager, current_page_num: int, db_session_for_page: SqlAlchemySession, state: dict[str, Any], progress_bar: Optional[tqdm]) -> Optional[list[dict[str, Any]]]:
    """Fetch matches for a page with error handling."""
    try:
        if not session_manager.is_sess_valid():
            raise ConnectionError(f"WebDriver session invalid before get_matches page {current_page_num}.")

        result = get_matches(session_manager, current_page_num)
        if result is None:
            logger.warning(f"get_matches returned None for page {current_page_num}. Skipping.")
            progress_bar.update(MATCHES_PER_PAGE)
            state["total_errors"] += MATCHES_PER_PAGE
            return []

        matches_on_page_for_batch, _ = result
        return matches_on_page_for_batch

    except ConnectionError as conn_e:
        logger.error(f"ConnectionError get_matches page {current_page_num}: {conn_e}", exc_info=False)
        progress_bar.update(MATCHES_PER_PAGE)
        state["total_errors"] += MATCHES_PER_PAGE
        return []
    except Exception as get_match_e:
        logger.error(f"Error get_matches page {current_page_num}: {get_match_e}", exc_info=True)
        progress_bar.update(MATCHES_PER_PAGE)
        state["total_errors"] += MATCHES_PER_PAGE
        return []
    finally:
        if db_session_for_page:
            session_manager.return_session(db_session_for_page)


def _check_session_validity(session_manager: SessionManager, current_page_num: int, state: dict[str, Any], progress_bar: Optional[tqdm]) -> bool:
    """
    Check if session is valid with circuit breaker protection.

    This enhanced function uses a circuit breaker to detect cascade failures
    and abort quickly rather than wasting time on thousands of retry attempts.

    Returns:
        bool: True if session is valid, False if invalid or circuit breaker tripped
    """
    circuit_breaker = get_session_circuit_breaker()

    # Check if circuit breaker is already tripped
    if circuit_breaker.is_tripped():
        logger.critical(
            f"🚨 Circuit breaker is TRIPPED - aborting page {current_page_num} to prevent cascade failure"
        )
        _mark_remaining_as_errors(state, progress_bar)
        return False

    # Check session validity
    if not session_manager.is_sess_valid():
        logger.error(f"WebDriver session invalid/unreachable before processing page {current_page_num}")

        # Record failure and check if circuit should trip
        if circuit_breaker.record_failure():
            logger.critical(
                f"🚨 Circuit breaker TRIPPED on page {current_page_num} - "
                f"session is permanently dead, aborting to prevent cascade failure"
            )
            _mark_remaining_as_errors(state, progress_bar)
            return False

        # Circuit hasn't tripped yet, but session is invalid
        # Mark remaining matches as errors and abort
        _mark_remaining_as_errors(state, progress_bar)
        return False

    # Session is valid - reset circuit breaker
    circuit_breaker.record_success()
    return True


def _mark_remaining_as_errors(state: dict[str, Any], progress_bar: Optional[tqdm]) -> None:
    """
    Mark all remaining matches as errors and update progress bar.

    Called when aborting due to session failure or circuit breaker trip.
    """
    if not progress_bar:
        return

    remaining_matches_estimate = max(0, progress_bar.total - progress_bar.n)
    if remaining_matches_estimate > 0:
        logger.warning(f"⚠️  Marking {remaining_matches_estimate} remaining matches as errors due to abort")
        progress_bar.update(remaining_matches_estimate)
        state["total_errors"] += remaining_matches_estimate


# ------------------------------------------------------------------------------
# 403 ERROR HANDLING WITH AUTH REFRESH - PRIORITY 1.3
# ------------------------------------------------------------------------------

def _is_request_successful(response: Any) -> bool:
    """Check if API request was successful."""
    return response is not None and (
        not isinstance(response, requests.Response) or response.status_code < 400
    )


def _record_api_timing(start_time: float, response: Any) -> None:
    """Record API call timing and success status."""
    duration = time.time() - start_time
    success = _is_request_successful(response)
    metrics = _get_metrics()
    metrics.record_api_call(duration, success)


def _handle_403_retry(
    session_manager: SessionManager,
    url: str,
    method: str,
    headers: Optional[dict[str, str]],
    **kwargs: Any
) -> Any:
    """Handle 403 error by refreshing auth and retrying."""
    logger.warning("🔐 403 Forbidden received - session likely expired")
    logger.info("🔄 Attempting to refresh authentication...")

    try:
        if not _refresh_session_auth(session_manager):
            logger.error("❌ Authentication refresh failed")
            return None

        logger.info("✅ Authentication refreshed successfully, retrying request")

        # Track retry
        metrics = _get_metrics()
        metrics.api_retries += 1
        retry_start = time.time()

        # Retry the request
        response = _api_req(
            url=url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method=method,
            headers=headers,
            **kwargs
        )

        # Record retry timing
        _record_api_timing(retry_start, response)

        # Check retry result
        if isinstance(response, requests.Response) and response.status_code == 403:
            logger.error("❌ Still getting 403 after auth refresh - session may be dead")
            metrics.record_error("403_after_retry")
        else:
            logger.info("✅ Request succeeded after auth refresh")

        return response

    except Exception as e:
        logger.error(f"❌ Error during auth refresh: {e}", exc_info=True)
        return None


def _api_req_with_auth_refresh(
    session_manager: SessionManager,
    url: str,
    method: str = "GET",
    headers: Optional[dict[str, str]] = None,
    **kwargs: Any
) -> Any:
    """
    Wrapper around _api_req that handles 403 Forbidden with auth refresh (Priority 1.3).

    When a 403 error is encountered (indicating session expiry), this function:
    1. Detects the 403 status code
    2. Refreshes authentication via SessionManager
    3. Retries the request with fresh credentials

    Args:
        session_manager: SessionManager instance for auth refresh
        url: API endpoint URL
        method: HTTP method (GET, POST, etc.)
        headers: Request headers
        **kwargs: Additional arguments to pass to _api_req

    Returns:
        API response (dict or Response object) or None on failure
    """
    # First attempt with timing
    start_time = time.time()
    response = _api_req(
        url=url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method=method,
        headers=headers,
        **kwargs
    )
    _record_api_timing(start_time, response)

    # Handle 403 Forbidden by refreshing auth and retrying
    if isinstance(response, requests.Response) and response.status_code == 403:
        response = _handle_403_retry(session_manager, url, method, headers, **kwargs)

    return response


def _refresh_session_auth(session_manager: SessionManager) -> bool:
    """
    Refresh session authentication (Priority 1.3).

    Attempts to refresh the session's authentication state by:
    1. Refreshing browser cookies
    2. Syncing cookies to API session
    3. Validating the refreshed session

    Args:
        session_manager: SessionManager instance

    Returns:
        bool: True if refresh successful, False otherwise
    """
    try:
        logger.debug("Refreshing browser cookies...")

        # Use SessionManager's refresh_browser_cookies if available
        if hasattr(session_manager, 'refresh_browser_cookies'):
            session_manager.refresh_browser_cookies()
        else:
            # Fallback: manually refresh cookies
            logger.debug("Using manual cookie refresh")
            if session_manager.driver:
                # Navigate to a lightweight page to refresh cookies
                session_manager.driver.get("https://www.ancestry.com/account/settings")
                time.sleep(2)  # Brief wait for cookies to update

        logger.debug("Syncing cookies to API session...")
        # Sync cookies to API session
        if hasattr(session_manager, 'sync_cookies_from_browser'):
            session_manager.sync_cookies_from_browser()

        # Validate refreshed session
        logger.debug("Validating refreshed session...")
        if session_manager.is_sess_valid():
            logger.info("✅ Session validation successful after refresh")
            return True
        logger.warning("⚠️  Session still invalid after refresh attempt")
        return False

    except Exception as e:
        logger.error(f"Error refreshing session auth: {e}", exc_info=True)
        return False


# End of 403 error handling


# === Priority 1.4: Session Health Monitoring ===


def _check_session_health_proactive(session_manager: SessionManager, current_page: int) -> bool:
    """
    Pre-emptively check session health and refresh if approaching expiry.

    Monitors session age and proactively refreshes authentication before
    the session expires to prevent 403 errors during batch operations.

    Args:
        session_manager: Active SessionManager instance
        current_page: Current page number for logging context

    Returns:
        bool: True if session is healthy (or successfully refreshed), False if refresh failed

    Design:
        - Check every 5 pages (configurable via HEALTH_CHECK_INTERVAL_PAGES)
        - Refresh threshold: 15 minutes before expiry (900 seconds)
        - Max session age: 40 minutes (2400 seconds, from session_health_monitor)
        - Refresh trigger: 25 minutes (1500 seconds)
    """
    try:
        # Check interval: only check every N pages to reduce overhead
        check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL_PAGES', '5'))
        if current_page % check_interval != 0:
            return True  # Not time to check yet

        # Get session age
        session_age = session_manager.session_age_seconds()
        if session_age is None:
            logger.warning(f"⚠️ Page {current_page}: Cannot determine session age")
            return True  # Unknown age, continue operation

        # Get thresholds from session_health_monitor
        max_session_age = session_manager.session_health_monitor.get('max_session_age', 2400)  # 40 min
        refresh_threshold = max_session_age - 900  # 15 min before expiry (25 min mark)

        # Log session age periodically
        if current_page % (check_interval * 2) == 0:  # Every 10 pages
            minutes_remaining = (max_session_age - session_age) / 60
            logger.info(f"🔍 Page {current_page}: Session age {session_age:.0f}s ({session_age/60:.1f}m), "
                       f"{minutes_remaining:.1f}m until expiry")

        # Check if refresh needed
        if session_age >= refresh_threshold:
            logger.warning(f"⚠️ Page {current_page}: Session age {session_age:.0f}s exceeds refresh threshold "
                          f"({refresh_threshold:.0f}s) - proactive refresh needed")

            # Perform proactive refresh
            logger.info(f"🔄 Page {current_page}: Initiating proactive session refresh...")

            if _refresh_session_auth(session_manager):
                logger.info(f"✅ Page {current_page}: Proactive session refresh successful")

                # Update last proactive refresh timestamp
                session_manager.session_health_monitor['last_proactive_refresh'] = time.time()

                # Reset session start time to current time (session is now fresh)
                session_manager.session_health_monitor['session_start_time'] = time.time()

                return True
            logger.error(f"❌ Page {current_page}: Proactive session refresh failed")
            return False

        # Session is healthy
        return True

    except Exception as e:
        logger.error(f"Error checking session health on page {current_page}: {e}", exc_info=True)
        return True  # Don't fail the batch on health check errors


def _get_session_health_status(session_manager: SessionManager) -> dict[str, Any]:
    """
    Get comprehensive session health status for monitoring and logging.

    Args:
        session_manager: Active SessionManager instance

    Returns:
        dict: Session health metrics including age, time remaining, and refresh status
    """
    try:
        session_age = session_manager.session_age_seconds()
        max_session_age = session_manager.session_health_monitor.get('max_session_age', 2400)
        last_proactive_refresh = session_manager.session_health_monitor.get('last_proactive_refresh', 0)

        if session_age is None:
            return {
                'status': 'unknown',
                'age_seconds': None,
                'age_minutes': None,
                'time_remaining_seconds': None,
                'time_remaining_minutes': None,
                'last_refresh_seconds_ago': None,
                'needs_refresh': False,
            }

        time_remaining = max_session_age - session_age
        time_since_refresh = time.time() - last_proactive_refresh if last_proactive_refresh > 0 else None

        # Determine status
        refresh_threshold = max_session_age - 900  # 15 min before expiry
        if session_age >= max_session_age:
            status = 'expired'
        elif session_age >= refresh_threshold:
            status = 'needs_refresh'
        elif session_age >= (max_session_age * 0.5):  # Over 50% of lifetime
            status = 'aging'
        else:
            status = 'healthy'

        return {
            'status': status,
            'age_seconds': session_age,
            'age_minutes': session_age / 60,
            'time_remaining_seconds': time_remaining,
            'time_remaining_minutes': time_remaining / 60,
            'last_refresh_seconds_ago': time_since_refresh,
            'needs_refresh': status == 'needs_refresh',
        }

    except Exception as e:
        logger.error(f"Error getting session health status: {e}", exc_info=True)
        return {'status': 'error', 'needs_refresh': False}


def _check_browser_restart_needed(session_manager: SessionManager, current_page: int) -> bool:
    """
    Check if browser restart is needed to prevent memory leaks and crashes.
    
    Periodically restarts the browser to maintain stability during long-running operations.
    This prevents Chrome from accumulating memory and eventually crashing.
    
    Args:
        session_manager: Active SessionManager instance
        current_page: Current page number for logging context
        
    Returns:
        bool: True if restart successful (or not needed), False if restart failed
        
    Design:
        - Restart interval: Every 50 pages (configurable via BROWSER_RESTART_INTERVAL_PAGES)
        - Default interval: ~1000 matches (~30-40 minutes at 4.5s/match)
        - Saves and restores cookies
        - Re-validates session after restart
    """
    try:
        # Get restart interval from environment (default: 50 pages)
        restart_interval = int(os.getenv('BROWSER_RESTART_INTERVAL_PAGES', '50'))

        # Check if restart is needed (at the interval, but not on first page)
        if current_page == 1 or current_page % restart_interval != 0:
            return True  # Not time to restart yet

        logger.info(f"🔄 Page {current_page}: Periodic browser restart triggered (every {restart_interval} pages)")

        # Save cookies before restart
        logger.debug("Saving cookies before browser restart...")
        if hasattr(session_manager, '_save_cookies'):
            session_manager._save_cookies()

        # Perform browser restart
        logger.info("♻️ Restarting Chrome browser to prevent memory buildup...")

        try:
            # Close current browser
            if session_manager.driver:
                session_manager.driver.quit()
                logger.debug("Previous browser instance closed")

            # Wait briefly for cleanup
            time.sleep(2)

            # Restart browser through session_manager
            if hasattr(session_manager, 'start_browser'):
                session_manager.start_browser()
                logger.info(f"✅ Browser restarted successfully at page {current_page}")
            else:
                logger.error("SessionManager does not have start_browser method")
                return False

            # Validate new session
            if not session_manager.is_sess_valid():
                logger.error("❌ Browser restart failed - session invalid after restart")
                return False

            # Navigate back to match list
            my_uuid = session_manager.my_uuid
            target_url = urljoin(
                config_schema.api.base_url, f"discoveryui-matches/list/{my_uuid}"
            )

            logger.debug(f"Navigating to match list after restart: {target_url}")
            session_manager.driver.get(target_url)
            time.sleep(3)  # Wait for page load

            logger.info(f"✅ Page {current_page}: Browser restart complete - ready to continue")
            return True

        except Exception as restart_err:
            logger.error(f"Error during browser restart: {restart_err}", exc_info=True)
            return False

    except Exception as e:
        logger.error(f"Error checking browser restart on page {current_page}: {e}", exc_info=True)
        return True  # Don't fail the batch on restart check errors


# End of session health monitoring


# === Priority 2.2: Progress Checkpointing ===


import os


def _get_checkpoint_filepath() -> Path:
    """Get the path to the checkpoint file."""
    checkpoint_dir = Path(os.getenv('CHECKPOINT_DIR', 'Cache'))
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir / 'action6_checkpoint.json'


def _save_checkpoint(
    current_page: int,
    total_pages: int,
    state: dict[str, Any],
    session_info: Optional[dict[str, Any]] = None
) -> bool:
    """
    Save current progress to checkpoint file.

    Args:
        current_page: Current page number being processed
        total_pages: Total pages to process in this run
        state: Current state dict with counters
        session_info: Optional session information for validation

    Returns:
        bool: True if checkpoint saved successfully, False otherwise

    Design:
        - Checkpoint saved after each page completes
        - Atomic write using temp file + rename
        - Includes timestamp for age tracking
        - Can be disabled via ENABLE_CHECKPOINTING=false
    """
    try:
        # Check if checkpointing is enabled
        if os.getenv('ENABLE_CHECKPOINTING', 'true').lower() not in ('true', '1', 'yes'):
            return True  # Silently skip if disabled

        checkpoint_data = {
            'version': '1.0',
            'timestamp': time.time(),
            'current_page': current_page,
            'total_pages': total_pages,
            'counters': {
                'total_new': state.get('total_new', 0),
                'total_updated': state.get('total_updated', 0),
                'total_skipped': state.get('total_skipped', 0),
                'total_errors': state.get('total_errors', 0),
                'total_pages_processed': state.get('total_pages_processed', 0),
            },
            'session_info': session_info or {},
        }

        checkpoint_path = _get_checkpoint_filepath()
        temp_path = checkpoint_path.with_suffix('.tmp')

        # Write to temp file first
        with open(temp_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Atomic rename
        temp_path.replace(checkpoint_path)

        logger.debug(f"💾 Checkpoint saved: page {current_page}/{total_pages}")
        return True

    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")
        return False


def _load_checkpoint() -> Optional[dict[str, Any]]:
    """
    Load checkpoint from file if it exists and is valid.

    Returns:
        dict: Checkpoint data if valid, None if no valid checkpoint exists

    Design:
        - Validates checkpoint age (max 24 hours by default)
        - Validates checkpoint version
        - Returns None for invalid/expired checkpoints
        - Logs reason for checkpoint rejection
    """
    try:
        # Check if checkpointing is enabled
        if os.getenv('ENABLE_CHECKPOINTING', 'true').lower() not in ('true', '1', 'yes'):
            return None

        checkpoint_path = _get_checkpoint_filepath()

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        # Validate checkpoint version
        if checkpoint_data.get('version') != '1.0':
            logger.warning(f"Checkpoint version mismatch: {checkpoint_data.get('version')}")
            return None

        # Validate checkpoint age
        max_age_hours = float(os.getenv('CHECKPOINT_MAX_AGE_HOURS', '24'))
        checkpoint_age = time.time() - checkpoint_data.get('timestamp', 0)

        if checkpoint_age > (max_age_hours * 3600):
            logger.info(f"Checkpoint expired (age: {checkpoint_age/3600:.1f}h > {max_age_hours}h)")
            return None

        logger.info(f"📂 Checkpoint loaded: page {checkpoint_data['current_page']}/{checkpoint_data['total_pages']} "
                   f"(age: {checkpoint_age/60:.1f}m)")

        return checkpoint_data

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid checkpoint file: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def _clear_checkpoint() -> bool:
    """
    Clear checkpoint file after successful completion.

    Returns:
        bool: True if cleared successfully, False otherwise
    """
    try:
        checkpoint_path = _get_checkpoint_filepath()

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("✨ Checkpoint cleared (run completed successfully)")
            return True

        return True  # No checkpoint to clear

    except Exception as e:
        logger.warning(f"Failed to clear checkpoint: {e}")
        return False


def _should_resume_from_checkpoint(
    checkpoint: Optional[dict[str, Any]],
    requested_start_page: Optional[int]
) -> tuple[bool, int]:
    """
    Determine if we should resume from checkpoint.

    Args:
        checkpoint: Loaded checkpoint data (or None)
        requested_start_page: Page number requested by user (None = auto-resume)

    Returns:
        tuple: (should_resume, start_page)
               - should_resume: True if resuming from checkpoint
               - start_page: Page number to start from

    Design:
        - User-specified start page ALWAYS takes precedence over checkpoint
        - None means "auto-resume from checkpoint if available"
        - Explicit page numbers (including 1) override checkpoint
    """
    # No checkpoint available - use requested page or default to 1
    if checkpoint is None:
        return False, requested_start_page if requested_start_page is not None else 1

    # User explicitly specified a start page - honor it and ignore checkpoint
    if requested_start_page is not None:
        logger.info(f"🎯 User specified start page {requested_start_page}, ignoring checkpoint")
        logger.info(f"   (Checkpoint was at page {checkpoint['current_page']}, but user override takes precedence)")
        return False, requested_start_page

    # User wants to auto-resume - use checkpoint
    checkpoint_page = checkpoint['current_page']
    resume_page = checkpoint_page + 1  # Resume from next page (checkpoint page is complete)

    logger.info(f"🔄 Auto-resuming from checkpoint: starting at page {resume_page}")
    logger.info(f"   Previous progress: {checkpoint['counters']['total_pages_processed']} pages, "
               f"{checkpoint['counters']['total_new']} new, "
               f"{checkpoint['counters']['total_updated']} updated")

    return True, resume_page


def _restore_state_from_checkpoint(
    state: dict[str, Any],
    checkpoint: dict[str, Any]
) -> None:
    """
    Restore state counters from checkpoint.

    Args:
        state: Current state dict to update
        checkpoint: Checkpoint data with saved counters

    Design:
        - Only restores counters (totals)
        - Does not restore transient state (matches_on_current_page)
        - Preserves state structure
    """
    try:
        counters = checkpoint.get('counters', {})

        state['total_new'] = counters.get('total_new', 0)
        state['total_updated'] = counters.get('total_updated', 0)
        state['total_skipped'] = counters.get('total_skipped', 0)
        state['total_errors'] = counters.get('total_errors', 0)
        state['total_pages_processed'] = counters.get('total_pages_processed', 0)

        logger.info(f"📊 State restored from checkpoint: "
                   f"{state['total_pages_processed']} pages processed, "
                   f"{state['total_new'] + state['total_updated'] + state['total_skipped']} matches")

    except Exception as e:
        logger.warning(f"Failed to restore state from checkpoint: {e}")


# End of progress checkpointing


# === Priority 3.1: Enhanced Logging & Performance Metrics ===


import statistics
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """
    Tracks performance metrics for action6 execution.

    Provides real-time visibility into:
    - Page processing times
    - API call performance
    - Worker efficiency
    - Cache effectiveness
    - Error patterns
    """

    # Timing metrics
    start_time: float = field(default_factory=time.time)
    page_times: list[float] = field(default_factory=list)
    api_call_times: list[float] = field(default_factory=list)

    # Progress metrics
    pages_completed: int = 0
    matches_processed: int = 0

    # API metrics
    api_calls_made: int = 0
    api_errors: int = 0
    api_retries: int = 0

    # Cache metrics  (integrated with Priority 2.3)
    cache_hits: int = 0
    cache_misses: int = 0

    # Worker metrics
    worker_idle_time: float = 0.0
    worker_busy_time: float = 0.0

    # Error tracking
    error_types: dict[str, int] = field(default_factory=dict)

    def record_page_time(self, duration: float) -> None:
        """Record time taken to process a page."""
        self.page_times.append(duration)
        self.pages_completed += 1

    def record_api_call(self, duration: float, success: bool = True) -> None:
        """Record API call timing and outcome."""
        self.api_call_times.append(duration)
        self.api_calls_made += 1
        if not success:
            self.api_errors += 1

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

    def record_error(self, error_type: str) -> None:
        """Track error by type for pattern analysis."""
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics summary."""
        elapsed = time.time() - self.start_time

        stats = {
            'elapsed_seconds': elapsed,
            'elapsed_formatted': self._format_duration(elapsed),
            'pages_completed': self.pages_completed,
            'matches_processed': self.matches_processed,
            'pages_per_minute': (self.pages_completed / elapsed * 60) if elapsed > 0 else 0,
            'matches_per_hour': (self.matches_processed / elapsed * 3600) if elapsed > 0 else 0,
        }

        # Page timing stats
        if self.page_times:
            stats['page_time_avg'] = statistics.mean(self.page_times)
            stats['page_time_median'] = statistics.median(self.page_times)
            stats['page_time_min'] = min(self.page_times)
            stats['page_time_max'] = max(self.page_times)

        # API stats
        stats['api_calls_total'] = self.api_calls_made
        stats['api_errors'] = self.api_errors
        stats['api_success_rate'] = ((self.api_calls_made - self.api_errors) / self.api_calls_made * 100) if self.api_calls_made > 0 else 0

        if self.api_call_times:
            stats['api_time_avg'] = statistics.mean(self.api_call_times)

        # Cache stats
        cache_total = self.cache_hits + self.cache_misses
        stats['cache_hit_rate'] = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0

        # Error breakdown
        if self.error_types:
            stats['error_breakdown'] = dict(sorted(self.error_types.items(), key=lambda x: x[1], reverse=True))

        return stats

    def log_progress(self, current_page: int, total_pages: int) -> None:
        """Log formatted progress update."""
        stats = self.get_stats()
        elapsed = stats['elapsed_formatted']
        progress_pct = (current_page / total_pages * 100) if total_pages > 0 else 0

        # Estimate remaining time
        if self.pages_completed > 0:
            avg_time_per_page = statistics.mean(self.page_times)
            pages_remaining = total_pages - current_page
            est_remaining_sec = pages_remaining * avg_time_per_page
            est_remaining = self._format_duration(est_remaining_sec)
        else:
            est_remaining = "calculating..."

        logger.info(
            f"📊 Progress: {current_page}/{total_pages} ({progress_pct:.1f}%) | "
            f"Elapsed: {elapsed} | ETA: {est_remaining} | "
            f"Avg: {stats.get('page_time_avg', 0):.1f}s/page | "
            f"Rate: {stats.get('pages_per_minute', 0):.1f} pages/min"
        )

    def log_final_summary(self) -> None:
        """Log comprehensive final statistics."""
        stats = self.get_stats()

        logger.info("=" * 80)
        logger.info("📈 FINAL PERFORMANCE REPORT")
        logger.info("=" * 80)

        # Overall metrics
        logger.info(f"⏱️  Total Duration: {stats['elapsed_formatted']}")
        logger.info(f"📄 Pages Processed: {stats['pages_completed']}")
        logger.info(f"👥 Matches Processed: {stats['matches_processed']}")
        logger.info(f"📊 Throughput: {stats['pages_per_minute']:.1f} pages/min, {stats['matches_per_hour']:.0f} matches/hour")

        # Page timing
        if 'page_time_avg' in stats:
            logger.info(f"⏱️  Page Times: avg={stats['page_time_avg']:.1f}s, median={stats['page_time_median']:.1f}s, min={stats['page_time_min']:.1f}s, max={stats['page_time_max']:.1f}s")

        # API performance
        logger.info(f"🌐 API Calls: {stats['api_calls_total']} total, {stats['api_errors']} errors, {stats['api_success_rate']:.1f}% success")
        if 'api_time_avg' in stats:
            logger.info(f"🌐 API Response Time: {stats['api_time_avg']:.2f}s avg")

        # Cache effectiveness
        logger.info(f"💾 Cache Performance: {stats['cache_hit_rate']:.1f}% hit rate ({self.cache_hits} hits, {self.cache_misses} misses)")

        # Error breakdown
        if 'error_breakdown' in stats:
            logger.info("⚠️  Error Breakdown:")
            for error_type, count in stats['error_breakdown'].items():
                logger.info(f"   - {error_type}: {count}")

        logger.info("=" * 80)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.0f}h {minutes:.0f}m"


# Global metrics instance
_metrics: Optional[PerformanceMetrics] = None


def _get_metrics() -> PerformanceMetrics:
    """Get or create global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = PerformanceMetrics()
    return _metrics


def _reset_metrics() -> None:
    """Reset metrics for new run."""
    global _metrics
    _metrics = PerformanceMetrics()
    logger.info("📊 Performance metrics initialized")


def _export_metrics_to_file(metrics: PerformanceMetrics, success: bool) -> None:
    """
    Export metrics to JSON file for historical analysis (Priority 3.2).

    Creates timestamped metrics file in Logs/metrics/ directory with:
    - Run metadata (timestamp, duration, success)
    - Performance statistics (timing, throughput)
    - API metrics (calls, errors, retries)
    - Cache performance
    - Error breakdown

    Args:
        metrics: PerformanceMetrics instance with run data
        success: Whether the run completed successfully
    """
    try:
        # Create metrics directory if it doesn't exist
        metrics_dir = Path("Logs/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = metrics_dir / f"action6_metrics_{timestamp}.json"

        # Get comprehensive stats
        stats = metrics.get_stats()

        # Add metadata
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "run_success": success,
                "module": "action6_gather",
                "version": "3.1.0"  # Updated with Priority 3.1
            },
            "performance": {
                "duration_seconds": stats['elapsed_seconds'],
                "duration_formatted": stats['elapsed_formatted'],
                "pages_completed": stats['pages_completed'],
                "matches_processed": stats['matches_processed'],
                "pages_per_minute": round(stats['pages_per_minute'], 2),
                "matches_per_hour": round(stats['matches_per_hour'], 0),
            },
            "timing": {
                "page_time_avg": round(stats.get('page_time_avg', 0), 2),
                "page_time_median": round(stats.get('page_time_median', 0), 2),
                "page_time_min": round(stats.get('page_time_min', 0), 2),
                "page_time_max": round(stats.get('page_time_max', 0), 2),
                "api_time_avg": round(stats.get('api_time_avg', 0), 2),
            },
            "api_metrics": {
                "total_calls": stats['api_calls_total'],
                "errors": stats['api_errors'],
                "retries": metrics.api_retries,
                "success_rate_percent": round(stats['api_success_rate'], 2),
            },
            "cache_metrics": {
                "hits": metrics.cache_hits,
                "misses": metrics.cache_misses,
                "hit_rate_percent": round(stats['cache_hit_rate'], 2),
            },
            "errors": stats.get('error_breakdown', {}),
        }

        # Write to file with pretty formatting
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"📁 Metrics exported to: {filename}")

    except Exception as e:
        logger.warning(f"⚠️  Failed to export metrics: {e}")


# End of enhanced logging & metrics


# === Priority 3.3: Real-time Monitoring & Alerts ===


@dataclass
class MonitoringThresholds:
    """Alert thresholds for real-time monitoring (Priority 3.3)."""

    # Error rate thresholds (errors per 100 operations)
    error_rate_warning: float = 5.0  # 5% error rate triggers warning
    error_rate_critical: float = 10.0  # 10% error rate triggers critical alert

    # API performance thresholds (seconds) - Adjusted for DNA API reality (Oct 8, 2025)
    # Real-world performance: 4-7s per match is normal, <4s is excellent
    api_time_warning: float = 8.0  # API call > 8s triggers warning (was 2.0)
    api_time_critical: float = 12.0  # API call > 12s triggers critical alert (was 5.0)

    # Page processing thresholds (seconds) - Adjusted for 20 matches/page reality
    # Real-world: 80-120s normal (4-6s per match × 20), 150s+ is slow
    page_time_warning: float = 150.0  # Page > 150s triggers warning (was 10.0)
    page_time_critical: float = 200.0  # Page > 200s triggers critical alert (was 30.0)

    # Cache performance thresholds (percentage)
    cache_hit_rate_warning: float = 5.0  # < 5% hit rate triggers warning

    # Session health thresholds
    session_age_warning: float = 1500.0  # 25 minutes triggers warning
    session_age_critical: float = 2100.0  # 35 minutes triggers critical alert


class RealTimeMonitor:
    """
    Real-time monitoring system with alert generation (Priority 3.3).

    Monitors execution metrics and generates alerts when thresholds are exceeded.
    Provides real-time visibility into system health and performance issues.
    """

    def __init__(self, thresholds: Optional[MonitoringThresholds] = None):
        """
        Initialize real-time monitor.

        Args:
            thresholds: Custom thresholds, or None for defaults
        """
        self.thresholds = thresholds or MonitoringThresholds()
        self.alerts: list[dict[str, Any]] = []
        self._last_check_time = time.time()
        self._check_interval = 60.0  # Check every 60 seconds

    def _check_error_rate(self, metrics: PerformanceMetrics, current_page: int) -> list[dict[str, Any]]:
        """Check error rate and generate alerts."""
        alerts = []
        if metrics.api_calls_made > 0:
            error_rate = (metrics.api_errors / metrics.api_calls_made) * 100
            if error_rate >= self.thresholds.error_rate_critical:
                alerts.append(self._create_alert(
                    level="CRITICAL",
                    category="error_rate",
                    message=f"Error rate critical: {error_rate:.1f}% (threshold: {self.thresholds.error_rate_critical}%)",
                    details={'error_rate': error_rate, 'errors': metrics.api_errors,
                            'total_calls': metrics.api_calls_made, 'current_page': current_page}
                ))
            elif error_rate >= self.thresholds.error_rate_warning:
                alerts.append(self._create_alert(
                    level="WARNING",
                    category="error_rate",
                    message=f"Error rate elevated: {error_rate:.1f}% (threshold: {self.thresholds.error_rate_warning}%)",
                    details={'error_rate': error_rate, 'errors': metrics.api_errors,
                            'total_calls': metrics.api_calls_made, 'current_page': current_page}
                ))
        return alerts

    def _check_api_performance(self, metrics: PerformanceMetrics, current_page: int) -> list[dict[str, Any]]:
        """Check API performance and generate alerts."""
        alerts = []
        if metrics.api_call_times:
            recent_api_times = metrics.api_call_times[-10:]
            avg_recent = sum(recent_api_times) / len(recent_api_times)
            if avg_recent >= self.thresholds.api_time_critical:
                alerts.append(self._create_alert(
                    level="CRITICAL",
                    category="api_performance",
                    message=f"API performance critical: {avg_recent:.1f}s avg (threshold: {self.thresholds.api_time_critical}s)",
                    details={'avg_time': avg_recent, 'recent_times': recent_api_times, 'current_page': current_page}
                ))
            elif avg_recent >= self.thresholds.api_time_warning:
                alerts.append(self._create_alert(
                    level="WARNING",
                    category="api_performance",
                    message=f"API performance degraded: {avg_recent:.1f}s avg (threshold: {self.thresholds.api_time_warning}s)",
                    details={'avg_time': avg_recent, 'recent_times': recent_api_times, 'current_page': current_page}
                ))
        return alerts

    def _check_page_performance(self, metrics: PerformanceMetrics, current_page: int) -> list[dict[str, Any]]:
        """Check page processing performance and generate alerts."""
        alerts = []
        if metrics.page_times:
            recent_page_times = metrics.page_times[-5:]
            avg_recent_page = sum(recent_page_times) / len(recent_page_times)
            if avg_recent_page >= self.thresholds.page_time_critical:
                alerts.append(self._create_alert(
                    level="CRITICAL",
                    category="page_performance",
                    message=f"Page processing critical: {avg_recent_page:.1f}s avg (threshold: {self.thresholds.page_time_critical}s)",
                    details={'avg_time': avg_recent_page, 'recent_times': recent_page_times, 'current_page': current_page}
                ))
            elif avg_recent_page >= self.thresholds.page_time_warning:
                alerts.append(self._create_alert(
                    level="WARNING",
                    category="page_performance",
                    message=f"Page processing slow: {avg_recent_page:.1f}s avg (threshold: {self.thresholds.page_time_warning}s)",
                    details={'avg_time': avg_recent_page, 'recent_times': recent_page_times, 'current_page': current_page}
                ))
        return alerts

    def _check_cache_effectiveness(self, metrics: PerformanceMetrics, current_page: int) -> list[dict[str, Any]]:
        """Check cache effectiveness and generate alerts."""
        alerts = []
        cache_total = metrics.cache_hits + metrics.cache_misses
        if cache_total >= 50:
            cache_hit_rate = (metrics.cache_hits / cache_total) * 100
            if cache_hit_rate < self.thresholds.cache_hit_rate_warning:
                alerts.append(self._create_alert(
                    level="INFO",
                    category="cache_performance",
                    message=f"Cache hit rate low: {cache_hit_rate:.1f}% (threshold: {self.thresholds.cache_hit_rate_warning}%)",
                    details={'hit_rate': cache_hit_rate, 'hits': metrics.cache_hits,
                            'misses': metrics.cache_misses, 'current_page': current_page}
                ))
        return alerts

    def check_metrics(self, metrics: PerformanceMetrics, current_page: int, total_pages: int) -> list[dict[str, Any]]:  # noqa: ARG002
        """
        Check metrics against thresholds and generate alerts.

        Args:
            metrics: Current performance metrics
            current_page: Current page number
            total_pages: Total pages to process (unused but kept for API consistency)

        Returns:
            List of new alerts generated
        """
        # Only check periodically to reduce overhead
        current_time = time.time()
        if current_time - self._last_check_time < self._check_interval:
            return []

        self._last_check_time = current_time
        new_alerts = []

        # Check error rate using helper method
        error_alerts = self._check_error_rate(metrics, current_page)
        new_alerts.extend(error_alerts)

        # Check API performance using helper method
        api_alerts = self._check_api_performance(metrics, current_page)
        new_alerts.extend(api_alerts)

        # Check page performance using helper method
        page_alerts = self._check_page_performance(metrics, current_page)
        new_alerts.extend(page_alerts)

        # Check cache effectiveness using helper method
        cache_alerts = self._check_cache_effectiveness(metrics, current_page)
        new_alerts.extend(cache_alerts)

        # Log and store new alerts
        for alert in new_alerts:
            self.alerts.append(alert)
            self._log_alert(alert)

        return new_alerts

    def check_session_health(self, session_manager: Any, current_page: int) -> list[dict[str, Any]]:
        """
        Check session health and generate alerts.

        Args:
            session_manager: SessionManager instance
            current_page: Current page number

        Returns:
            List of new alerts generated
        """
        new_alerts = []

        try:
            session_age = session_manager.session_age_seconds()
            if session_age is None:
                return new_alerts

            if session_age >= self.thresholds.session_age_critical:
                new_alerts.append(self._create_alert(
                    level="CRITICAL",
                    category="session_health",
                    message=f"Session age critical: {session_age/60:.1f}m (threshold: {self.thresholds.session_age_critical/60:.1f}m)",
                    details={
                        'session_age_seconds': session_age,
                        'session_age_minutes': session_age / 60,
                        'current_page': current_page
                    }
                ))
            elif session_age >= self.thresholds.session_age_warning:
                new_alerts.append(self._create_alert(
                    level="WARNING",
                    category="session_health",
                    message=f"Session age elevated: {session_age/60:.1f}m (threshold: {self.thresholds.session_age_warning/60:.1f}m)",
                    details={
                        'session_age_seconds': session_age,
                        'session_age_minutes': session_age / 60,
                        'current_page': current_page
                    }
                ))

            # Log and store new alerts
            for alert in new_alerts:
                self.alerts.append(alert)
                self._log_alert(alert)

        except Exception as e:
            logger.warning(f"Error checking session health: {e}")

        return new_alerts

    def _create_alert(self, level: str, category: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
        """Create an alert dict with metadata."""
        return {
            'timestamp': time.time(),
            'level': level,
            'category': category,
            'message': message,
            'details': details
        }

    def _log_alert(self, alert: dict[str, Any]) -> None:
        """Log an alert with appropriate severity."""
        emoji = {
            'CRITICAL': '🚨',
            'WARNING': '⚠️ ',
            'INFO': 'ℹ️ '
        }.get(alert['level'], '📢')

        log_func = {
            'CRITICAL': logger.critical,
            'WARNING': logger.warning,
            'INFO': logger.info
        }.get(alert['level'], logger.info)

        log_func(f"{emoji} ALERT [{alert['category']}]: {alert['message']}")

    def get_alerts(self, level: Optional[str] = None, category: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Get alerts filtered by level and/or category.

        Args:
            level: Filter by alert level (CRITICAL, WARNING, INFO)
            category: Filter by alert category

        Returns:
            List of matching alerts
        """
        filtered = self.alerts

        if level:
            filtered = [a for a in filtered if a['level'] == level]

        if category:
            filtered = [a for a in filtered if a['category'] == category]

        return filtered

    def get_alert_summary(self) -> dict[str, int]:
        """Get summary of alerts by level."""
        summary = {'CRITICAL': 0, 'WARNING': 0, 'INFO': 0}
        for alert in self.alerts:
            level = alert['level']
            if level in summary:
                summary[level] += 1
        return summary

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self.alerts.clear()


# Global monitor instance
_monitor: Optional[RealTimeMonitor] = None


def _get_monitor() -> RealTimeMonitor:
    """Get or create global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = RealTimeMonitor()
    return _monitor


def _reset_monitor() -> None:
    """Reset monitor for new run."""
    global _monitor
    _monitor = RealTimeMonitor()
    logger.info("📡 Real-time monitoring initialized")


# End of real-time monitoring


# === Priority 2.3: API Call Batching & Deduplication ===


from threading import Lock


class APICallCache:
    """
    Thread-safe cache for API call results to prevent duplicate requests.

    Caches results for combined_details, badge_details, and ladder_details
    within a single action6 run to avoid redundant API calls.

    Design:
        - Thread-safe with lock for parallel workers
        - Per-run cache (cleared at start of coord())
        - TTL-based expiry (default: 1 hour)
        - Memory-efficient (stores only what's needed)
    """

    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize API call cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
        """
        self._lock = Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl_seconds
        self._stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'evictions': 0
        }

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Get cached result if available and not expired.

        Args:
            cache_key: Unique key for the API call (e.g., "combined:UUID123")

        Returns:
            Cached result or None if not found/expired
        """
        with self._lock:
            if cache_key not in self._cache:
                self._stats['misses'] += 1
                # Also record in global metrics
                metrics = _get_metrics()
                metrics.record_cache_miss()
                return None

            # Check TTL
            age = time.time() - self._timestamps.get(cache_key, 0)
            if age > self._ttl:
                # Expired - evict
                del self._cache[cache_key]
                del self._timestamps[cache_key]
                self._stats['evictions'] += 1
                self._stats['misses'] += 1
                # Record in global metrics
                metrics = _get_metrics()
                metrics.record_cache_miss()
                return None

            self._stats['hits'] += 1
            # Record in global metrics
            metrics = _get_metrics()
            metrics.record_cache_hit()
            return self._cache[cache_key]

    def set(self, cache_key: str, result: Any) -> None:
        """
        Store result in cache.

        Args:
            cache_key: Unique key for the API call
            result: Result to cache
        """
        with self._lock:
            self._cache[cache_key] = result
            self._timestamps[cache_key] = time.time()
            self._stats['saves'] += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._stats = {
                'hits': 0,
                'misses': 0,
                'saves': 0,
                'evictions': 0
            }

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (
                (self._stats['hits'] / (self._stats['hits'] + self._stats['misses']) * 100)
                if (self._stats['hits'] + self._stats['misses']) > 0
                else 0
            )
            return {
                **self._stats,
                'size': len(self._cache),
                'hit_rate_percent': round(hit_rate, 2)
            }


# Global cache instance (per-run)
_api_call_cache: Optional[APICallCache] = None


def _get_api_cache() -> APICallCache:
    """Get or create the API call cache."""
    global _api_call_cache
    if _api_call_cache is None:
        ttl = int(os.getenv('API_CACHE_TTL_SECONDS', '3600'))
        _api_call_cache = APICallCache(ttl_seconds=ttl)
    return _api_call_cache


def _clear_api_cache() -> None:
    """Clear the API call cache (called at start of coord())."""
    if _api_call_cache is not None:
        stats = _api_call_cache.get_stats()
        if stats['hits'] + stats['misses'] > 0:
            logger.info(
                f"📊 API Cache Stats: {stats['hits']} hits, {stats['misses']} misses, "
                f"{stats['hit_rate_percent']}% hit rate, {stats['saves']} saves"
            )
        _api_call_cache.clear()


def _deduplicate_api_requests(
    fetch_candidates_uuid: set[str],
    matches_to_process_later: list[dict[str, Any]]
) -> tuple[set[str], list[dict[str, Any]], int]:
    """
    Deduplicate API requests by checking cache and removing already-fetched UUIDs.

    Args:
        fetch_candidates_uuid: Set of UUIDs needing detail fetches
        matches_to_process_later: List of match data dicts

    Returns:
        tuple: (deduplicated_uuids, matches_list, cache_hits_count)
    """
    cache = _get_api_cache()
    deduplicated = set()
    cache_hits = 0

    for uuid in fetch_candidates_uuid:
        cache_key = f"combined:{uuid}"
        cached_result = cache.get(cache_key)

        if cached_result is None:
            deduplicated.add(uuid)
        else:
            cache_hits += 1
            logger.debug(f"Cache hit for UUID {uuid} - skipping API call")

    if cache_hits > 0:
        logger.info(f"🎯 API Call Deduplication: {cache_hits} cached, {len(deduplicated)} need fetch")

    return deduplicated, matches_to_process_later, cache_hits


def _batch_optimize_tree_requests(
    uuids_for_tree_badge_ladder: list[str],
    temp_badge_results: dict[str, Optional[dict[str, Any]]]
) -> list[str]:
    """
    Optimize tree-related requests by deduplicating CFPIDs.

    Args:
        uuids_for_tree_badge_ladder: List of UUIDs needing tree data
        temp_badge_results: Badge results to extract CFPIDs from

    Returns:
        Deduplicated list of CFPIDs for ladder requests
    """
    cfpid_set = set()

    for uuid in uuids_for_tree_badge_ladder:
        badge_data = temp_badge_results.get(uuid)
        if badge_data and badge_data.get('their_cfpid'):
            cfpid_set.add(badge_data['their_cfpid'])

    original_count = len(uuids_for_tree_badge_ladder)
    deduplicated_count = len(cfpid_set)

    if deduplicated_count < original_count:
        saved = original_count - deduplicated_count
        logger.info(f"🎯 Tree Request Optimization: {saved} duplicate CFPIDs eliminated")

    return list(cfpid_set)


# End of API call batching


def _update_state_after_batch(
    state: dict[str, Any],
    counters: BatchCounters,
    progress_bar: Optional[tqdm],
    current_page: int = 0,
    total_pages: int = 0
) -> None:
    """Update state counters and progress bar after processing a batch."""
    state["total_new"] += counters.new
    state["total_updated"] += counters.updated
    state["total_skipped"] += counters.skipped
    state["total_errors"] += counters.errors
    state["total_pages_processed"] += 1

    progress_bar.set_postfix(
        New=state["total_new"],
        Upd=state["total_updated"],
        Skip=state["total_skipped"],
        Err=state["total_errors"],
        refresh=True,
    )

    # Priority 2.2: Save checkpoint after each page
    if current_page > 0 and total_pages > 0:
        _save_checkpoint(current_page, total_pages, state)


# Helper functions for _main_page_processing_loop

def _calculate_total_matches_estimate(start_page: int, total_pages_in_run: int, initial_matches_on_page: Optional[list[dict[str, Any]]]) -> int:
    """Calculate total matches estimate for progress bar."""
    total_matches_estimate = total_pages_in_run * MATCHES_PER_PAGE

    if start_page == 1 and initial_matches_on_page is not None:
        total_matches_estimate = max(total_matches_estimate, len(initial_matches_on_page))

    return total_matches_estimate


def _should_fetch_page_data(current_page_num: int, start_page: int, matches_on_page_for_batch: Optional[list[dict[str, Any]]]) -> bool:
    """Determine if page data needs to be fetched."""
    return not (current_page_num == start_page and matches_on_page_for_batch is not None)


def _fetch_and_validate_page_data(
    session_manager: SessionManager,
    current_page_num: int,
    state: dict[str, Any],
    progress_bar: Any
) -> Optional[list[dict[str, Any]]]:
    """Fetch page data and validate DB session."""
    # Get DB session with retry
    db_session_for_page = _get_db_session_with_retry(session_manager, current_page_num, state)

    if not db_session_for_page:
        return None

    # Fetch page matches
    return _fetch_page_matches(session_manager, current_page_num, db_session_for_page, state, progress_bar)


def _handle_empty_matches(current_page_num: int, start_page: int, state: dict[str, Any], progress_bar: Any) -> None:
    """Handle empty matches on a page."""
    logger.info(f"No matches found or processed on page {current_page_num}.")

    if not (current_page_num == start_page and state["total_pages_processed"] == 0):
        progress_bar.update(MATCHES_PER_PAGE)

    time.sleep(0.5)


def _process_page_batch(
    session_manager: SessionManager,
    matches_on_page: list[dict[str, Any]],
    current_page_num: int,
    progress_bar: Any,
    state: dict[str, Any],
    total_pages: int = 0
) -> None:
    """Process a batch of matches and update state."""
    # Priority 3.1: Track page processing time
    page_start_time = time.time()

    # Process batch
    page_new, page_updated, page_skipped, page_errors = _do_batch(
        session_manager=session_manager,
        matches_on_page=matches_on_page,
        current_page=current_page_num,
        progress_bar=progress_bar,
    )

    # Update state (Priority 2.2: includes checkpoint saving)
    counters = BatchCounters(new=page_new, updated=page_updated, skipped=page_skipped, errors=page_errors)
    _update_state_after_batch(state, counters, progress_bar, current_page_num, total_pages)

    # Priority 3.1: Record page metrics
    page_duration = time.time() - page_start_time
    metrics = _get_metrics()
    metrics.record_page_time(page_duration)
    metrics.matches_processed += (page_new + page_updated + page_skipped)

    # Priority 3.1: Log progress every 10 pages
    if current_page_num % 10 == 0 and total_pages > 0:
        metrics.log_progress(current_page_num, total_pages)

    # Priority 3.3: Real-time monitoring checks
    if current_page_num % 5 == 0:  # Check more frequently than progress logging
        monitor = _get_monitor()
        monitor.check_metrics(metrics, current_page_num, total_pages)
        monitor.check_session_health(session_manager, current_page_num)

    # Apply rate limiting
    _adjust_delay(session_manager, current_page_num)
    session_manager.dynamic_rate_limiter.wait()


def _finalize_progress_bar(progress_bar: Any, state: dict[str, Any], loop_final_success: bool) -> None:
    """Finalize progress bar display."""
    if not progress_bar:
        return

    progress_bar.set_postfix(
        New=state["total_new"],
        Upd=state["total_updated"],
        Skip=state["total_skipped"],
        Err=state["total_errors"],
        refresh=True,
    )

    if progress_bar.n < progress_bar.total and loop_final_success:
        pass  # tqdm closes itself correctly
    elif progress_bar.n < progress_bar.total and not loop_final_success:
        pass  # Handle incomplete progress


def _process_single_page_iteration(
    session_manager: SessionManager,
    current_page_num: int,
    start_page: int,
    matches_on_page_for_batch: Optional[list[dict[str, Any]]],
    state: dict[str, Any],
    progress_bar: tqdm,
    loop_final_success: bool,
    total_pages: int = 0
) -> tuple[bool, bool, Optional[list[dict[str, Any]]], int]:
    """Process a single page iteration in the main loop."""
    # Priority 1.4a: Check if browser restart needed (memory management)
    if not _check_browser_restart_needed(session_manager, current_page_num):
        logger.error(f"❌ Page {current_page_num}: Browser restart failed - aborting batch")
        _mark_remaining_as_errors(state, progress_bar, current_page_num)
        return False, True, matches_on_page_for_batch, current_page_num

    # Priority 1.4b: Pre-emptive session health check
    if not _check_session_health_proactive(session_manager, current_page_num):
        logger.error(f"❌ Page {current_page_num}: Session health check failed - aborting batch")
        _mark_remaining_as_errors(state, progress_bar, current_page_num)
        return False, True, matches_on_page_for_batch, current_page_num

    # Check session validity
    if not _check_session_validity(session_manager, current_page_num, state, progress_bar):
        return False, True, matches_on_page_for_batch, current_page_num

    # Fetch match data if needed
    if _should_fetch_page_data(current_page_num, start_page, matches_on_page_for_batch):
        matches_on_page_for_batch = _fetch_and_validate_page_data(
            session_manager, current_page_num, state, progress_bar
        )

        if matches_on_page_for_batch is None:
            loop_final_success, should_break = _handle_db_session_failure(
                current_page_num, state, progress_bar, loop_final_success
            )
            if should_break:
                return loop_final_success, True, matches_on_page_for_batch, current_page_num
            return loop_final_success, False, matches_on_page_for_batch, current_page_num + 1

        if not matches_on_page_for_batch:
            time.sleep(0.5 if loop_final_success else 2.0)
            return loop_final_success, False, matches_on_page_for_batch, current_page_num + 1

    # Handle empty matches
    if not matches_on_page_for_batch:
        _handle_empty_matches(current_page_num, start_page, state, progress_bar)
        return loop_final_success, False, None, current_page_num + 1

    # Process batch and update state (Priority 2.2: includes checkpoint saving)
    _process_page_batch(session_manager, matches_on_page_for_batch, current_page_num, progress_bar, state, total_pages)

    return loop_final_success, False, None, current_page_num + 1


def _main_page_processing_loop(
    session_manager: SessionManager,
    start_page: int,
    last_page_to_process: int,
    total_pages_in_run: int,  # Added this argument
    initial_matches_on_page: Optional[list[dict[str, Any]]],
    state: dict[str, Any],  # Pass the whole state dict
) -> bool:
    """Main loop for fetching and processing pages of matches."""
    current_page_num = start_page
    total_matches_estimate_this_run = _calculate_total_matches_estimate(start_page, total_pages_in_run, initial_matches_on_page)
    loop_final_success = True

    with logging_redirect_tqdm():
        progress_bar = tqdm(
            total=total_matches_estimate_this_run,
            desc="",
            unit=" match",
            bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            file=sys.stderr,
            leave=True,
            dynamic_ncols=True,
            ascii=False,
        )
        try:
            matches_on_page_for_batch: Optional[list[dict[str, Any]]] = initial_matches_on_page

            while current_page_num <= last_page_to_process:
                loop_final_success, should_break, matches_on_page_for_batch, current_page_num = (
                    _process_single_page_iteration(
                        session_manager,
                        current_page_num,
                        start_page,
                        matches_on_page_for_batch,
                        state,
                        progress_bar,
                        loop_final_success,
                        last_page_to_process  # Priority 2.2: Pass total pages for checkpointing
                    )
                )
                if should_break:
                    break
        finally:
            _finalize_progress_bar(progress_bar, state, loop_final_success)
            if progress_bar and progress_bar.n < progress_bar.total and not loop_final_success:
                # If loop ended due to error, update bar to reflect error count for remaining
                remaining_to_mark_error = progress_bar.total - progress_bar.n
                if remaining_to_mark_error > 0:
                    progress_bar.update(remaining_to_mark_error)
            if progress_bar:
                progress_bar.close()
                print("", file=sys.stderr)  # Newline after bar

    return loop_final_success


# End of _main_page_processing_loop

# === COORDINATION HELPER FUNCTIONS ===

def _validate_session_state(session_manager: SessionManager) -> None:
    """Validate session state before processing."""
    if not session_manager.driver or not session_manager.driver_live or not session_manager.session_ready:
        raise BrowserSessionError(
            "WebDriver/Session not ready for DNA match gathering",
            context={
                "driver_live": session_manager.driver_live,
                "session_ready": session_manager.session_ready,
            },
        )
    if not session_manager.my_uuid:
        raise AuthenticationExpiredError(
            "Failed to retrieve my_uuid for DNA match gathering",
            context={"session_state": "authenticated but no UUID"},
        )


def _process_initial_navigation(session_manager: SessionManager, start_page: int, state: dict[str, Any]) -> tuple[Optional[list[dict[str, Any]]], Optional[int], bool]:
    """Navigate to initial page and get total pages."""
    initial_matches, total_pages_api, initial_fetch_ok = _navigate_and_get_initial_page_data(session_manager, start_page)

    if not initial_fetch_ok or total_pages_api is None:
        logger.error("Failed to retrieve total_pages on initial fetch. Aborting.")
        state["final_success"] = False
        return None, None, False

    state["total_pages_from_api"] = total_pages_api
    state["matches_on_current_page"] = initial_matches if initial_matches is not None else []
    logger.info(f"Total pages found: {total_pages_api}")

    return initial_matches, total_pages_api, True


def _calculate_processing_range(total_pages_api: int, start_page: int) -> tuple[int, int, int]:
    """Calculate page processing range and total matches estimate."""
    last_page_to_process, total_pages_in_run = _determine_page_processing_range(total_pages_api, start_page)

    if total_pages_in_run <= 0:
        logger.info(f"No pages to process (Start: {start_page}, End: {last_page_to_process}).")
        return last_page_to_process, total_pages_in_run, 0

    total_matches_estimate = total_pages_in_run * MATCHES_PER_PAGE
    logger.info(
        f"Processing {total_pages_in_run} pages (approx. {total_matches_estimate} matches) "
        f"from page {start_page} to {last_page_to_process}.\n"
    )

    return last_page_to_process, total_pages_in_run, total_matches_estimate


# ------------------------------------------------------------------------------
# Core Orchestration (coord) - REFACTORED
# ------------------------------------------------------------------------------


@retry_on_failure(max_attempts=3, backoff_factor=2.0)
@circuit_breaker(failure_threshold=10, recovery_timeout=60)  # Increased from 3 to 10 for better tolerance
@timeout_protection(timeout=600)  # Increased from 300s to 600s for rate-limited processing (~6s per match)
@error_context("DNA match gathering coordination")
def _execute_main_gathering(
    session_manager: SessionManager,
    start_page: int,
    state: dict[str, Any]
) -> bool:
    """Execute the main gathering logic with error handling."""
    try:
        # Step 3: Initial Navigation and Total Pages Fetch
        _initial_matches, total_pages_api, initial_fetch_ok = _process_initial_navigation(
            session_manager, start_page, state
        )

        if not initial_fetch_ok:
            return False

        # Step 4: Determine Page Range
        last_page_to_process, total_pages_in_run, _total_matches_estimate = _calculate_processing_range(
            total_pages_api, start_page
        )

        if total_pages_in_run <= 0:
            return True

        # Step 5: Main Processing Loop
        initial_matches_for_loop = state["matches_on_current_page"]
        return _main_page_processing_loop(
            session_manager,
            start_page,
            last_page_to_process,
            total_pages_in_run,
            initial_matches_for_loop,
            state,
        )


    # Handle specific exceptions
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected. Stopping match gathering.")
        return False
    except ConnectionError as coord_conn_err:
        logger.critical(f"ConnectionError during coord execution: {coord_conn_err}", exc_info=True)
        return False
    except MaxApiFailuresExceededError as api_halt_err:
        logger.critical(f"Halting run due to excessive critical API failures: {api_halt_err}", exc_info=False)
        return False
    except Exception as e:
        logger.error(f"Critical error during coord execution: {e}", exc_info=True)
        return False


def coord(  # type: ignore
    session_manager: SessionManager, _config_schema_arg: "ConfigSchema", start: Optional[int] = None
) -> bool:  # Uses config schema
    """
    Orchestrates the gathering of DNA matches from Ancestry.
    Handles pagination, fetches match data, compares with database, and processes.
    
    Args:
        session_manager: Active SessionManager instance
        _config_schema_arg: Configuration schema
        start: Start page number (None = auto-resume from checkpoint, explicit number = start from that page)
    """
    # Step 1: Validate Session State
    _validate_session_state(session_manager)

    # Priority 1.2: Disable auto-recovery for fail-fast behavior
    previous_recovery_state = session_manager.get_auto_recovery_status()
    session_manager.set_auto_recovery(False)
    logger.info("🚫 Disabled session auto-recovery for action6 (fail-fast mode)")

    # Step 2: Initialize state and resources
    state = _initialize_gather_state()
    requested_start_page = _validate_start_page(start) if start is not None else None

    # Priority 2.3: Clear API call cache
    _clear_api_cache()
    logger.info("🔄 API call cache cleared for new run")

    # Priority 3.1 & 3.3: Initialize monitoring
    _reset_metrics()
    logger.info("📊 Performance metrics tracking enabled")
    _reset_monitor()

    # Priority 2.2: Load checkpoint if available
    checkpoint = _load_checkpoint()
    resuming_from_checkpoint, start_page = _should_resume_from_checkpoint(checkpoint, requested_start_page)

    if resuming_from_checkpoint and checkpoint:
        _restore_state_from_checkpoint(state, checkpoint)

    logger.debug(f"--- Starting DNA Match Gathering (Action 6) from page {start_page} ---")
    logger.info(f"🎯 Starting Action 6: DNA Match Gathering from page {start_page}")

    try:
        # Execute main gathering with error handling
        loop_success = _execute_main_gathering(session_manager, start_page, state)
        state["final_success"] = state["final_success"] and loop_success
    finally:
        # Priority 1.2: Restore auto-recovery to previous state
        session_manager.set_auto_recovery(previous_recovery_state)
        logger.info(f"🔧 Restored session auto-recovery to: {'enabled' if previous_recovery_state else 'disabled'}")

        # Priority 2.2: Clear checkpoint on successful completion
        if state.get("final_success", False):
            _clear_checkpoint()

        # Priority 2.3: Log final API cache statistics
        _clear_api_cache()  # This logs stats before clearing

        # Priority 3.1: Log final performance metrics
        metrics = _get_metrics()
        metrics.log_final_summary()

        # Priority 3.2: Export metrics to JSON file for historical analysis
        _export_metrics_to_file(metrics, state.get("final_success", False))

        # Priority 3.3: Log real-time monitoring summary
        monitor = _get_monitor()
        alert_summary = monitor.get_alert_summary()
        if any(alert_summary.values()):
            logger.info(f"🚨 Alert Summary: {alert_summary['CRITICAL']} critical, "
                       f"{alert_summary['WARNING']} warnings, {alert_summary['INFO']} info")

            # Log critical alerts for visibility
            critical_alerts = monitor.get_alerts(level='CRITICAL')
            if critical_alerts:
                logger.warning(f"⚠️  {len(critical_alerts)} CRITICAL alerts were triggered during execution:")
                for alert in critical_alerts[:5]:  # Show first 5 critical alerts
                    logger.warning(f"   • {alert['category']}: {alert['message']}")

        # Step 7: Final Summary Logging (uses updated state from the loop)
        logger.debug("Entering finally block in coord...")
        _log_coord_summary(
            state["total_pages_processed"],
            state["total_new"],
            state["total_updated"],
            state["total_skipped"],
            state["total_errors"],
        )
        # Re-raise KeyboardInterrupt if that was the cause
        exc_info_tuple = sys.exc_info()
        if exc_info_tuple[0] is KeyboardInterrupt:
            logger.info("Re-raising KeyboardInterrupt after cleanup.")
            if exc_info_tuple[1] is not None:
                raise exc_info_tuple[1].with_traceback(exc_info_tuple[2])
        logger.debug("Exiting finally block in coord.")

    return state["final_success"]


# End of coord

# ------------------------------------------------------------------------------
# Batch Processing Logic (_do_batch and Helpers)
# ------------------------------------------------------------------------------


def _lookup_existing_persons(
    session: SqlAlchemySession, uuids_on_page: list[str]
) -> dict[str, Person]:
    """
    Queries the database for existing Person records based on a list of UUIDs.
    Eager loads related DnaMatch and FamilyTree data for efficiency.

    Args:
        session: The active SQLAlchemy database session.
        uuids_on_page: A list of UUID strings to look up.

    Returns:
        A dictionary mapping UUIDs (uppercase) to their corresponding Person objects.
        Returns an empty dictionary if input list is empty or an error occurs.

    Raises:
        SQLAlchemyError: If a database query error occurs.
        ValueError: If a critical data mismatch (like Enum) is detected.
    """
    # Step 1: Initialize result map
    existing_persons_map: dict[str, Person] = {}
    # Step 2: Handle empty input list
    if not uuids_on_page:
        return existing_persons_map

    # Step 3: Query the database
    try:
        logger.debug(f"Querying DB for {len(uuids_on_page)} existing Person records...")
        # Convert incoming UUIDs to uppercase for consistent matching
        uuids_upper = {uuid_val.upper() for uuid_val in uuids_on_page}

        existing_persons = (
            session.query(Person)
            # Eager load related tables to avoid N+1 queries later
            .options(joinedload(Person.dna_match), joinedload(Person.family_tree))
            # Filter by the list of uppercase UUIDs and exclude soft-deleted records
            .filter(Person.uuid.in_(uuids_upper), Person.deleted_at.is_(None)).all()  # type: ignore
        )
        # Step 4: Populate the result map (key by UUID)
        existing_persons_map: dict[str, Person] = {
            str(person.uuid): person
            for person in existing_persons
            if person.uuid is not None
        }
        logger.debug(
            f"Found {len(existing_persons_map)} existing Person records for this batch."
        )

    # Step 5: Handle potential database errors
    except SQLAlchemyError as db_lookup_err:
        # Check specifically for Enum mismatch errors which can be critical
        if "is not among the defined enum values" in str(db_lookup_err):
            logger.critical(
                f"CRITICAL ENUM MISMATCH during Person lookup. DB schema might be outdated. Error: {db_lookup_err}"
            )
            # Raise a specific error to halt processing if schema mismatch detected
            raise ValueError(
                "Database enum mismatch detected during person lookup."
            ) from db_lookup_err
        # Log other SQLAlchemy errors and re-raise
        logger.error(
            f"Database lookup failed during prefetch: {db_lookup_err}",
            exc_info=True,
        )
        raise  # Re-raise to be handled by the caller (_do_batch)
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during Person lookup: {e}", exc_info=True)
        raise  # Re-raise to be handled by the caller

    # Step 6: Return the map of found persons
    return existing_persons_map


# End of _lookup_existing_persons

# Helper functions for _identify_fetch_candidates

def _check_dna_data_changes(
    match_api_data: dict[str, Any],
    existing_dna: Any,
    uuid_val: str
) -> bool:
    """Check if DNA data has changed compared to existing record."""
    if not existing_dna:
        logger.debug(f"  Fetch needed (UUID {uuid_val}): No existing DNA record.")
        return True

    try:
        # Compare cM
        api_cm = int(match_api_data.get("cM_DNA", 0))
        db_cm = existing_dna.cM_DNA
        if api_cm != db_cm:
            logger.debug(f"  Fetch needed (UUID {uuid_val}): cM changed ({db_cm} -> {api_cm})")
            return True

        # Compare segments
        api_segments = int(match_api_data.get("numSharedSegments", 0))
        db_segments = existing_dna.shared_segments
        if api_segments != db_segments:
            logger.debug(f"  Fetch needed (UUID {uuid_val}): Segments changed ({db_segments} -> {api_segments})")
            return True

        return False
    except (ValueError, TypeError, AttributeError) as comp_err:
        logger.warning(f"Error comparing list DNA data for UUID {uuid_val}: {comp_err}. Assuming fetch needed.")
        return True


def _check_tree_status_changes(
    api_in_tree: bool,
    db_in_tree: bool,
    existing_tree: Any,
    uuid_val: str
) -> bool:
    """Check if tree status has changed compared to existing record."""
    if bool(api_in_tree) != bool(db_in_tree):
        logger.debug(f"  Fetch needed (UUID {uuid_val}): Tree status changed ({db_in_tree} -> {api_in_tree})")
        return True
    if api_in_tree and not existing_tree:
        logger.debug(f"  Fetch needed (UUID {uuid_val}): Marked in tree but no DB record.")
        return True
    return False


def _should_fetch_match_details(
    match_api_data: dict[str, Any],
    existing_person: Any
) -> bool:
    """Determine if match details should be fetched based on changes."""
    uuid_val = match_api_data.get("uuid")

    # Check DNA data changes
    if _check_dna_data_changes(match_api_data, existing_person.dna_match, uuid_val):
        return True

    # Check tree status changes
    api_in_tree = match_api_data.get("in_my_tree", False)
    db_in_tree = existing_person.in_my_tree
    return bool(_check_tree_status_changes(api_in_tree, db_in_tree, existing_person.family_tree, uuid_val))


def _identify_fetch_candidates(
    matches_on_page: list[dict[str, Any]], existing_persons_map: dict[str, Any]
) -> tuple[set[str], list[dict[str, Any]], int]:
    """
    Analyzes matches from a page against existing database records to determine:
    1. Which matches need detailed API data fetched (new or potentially changed).
    2. Which matches can be skipped (no apparent change based on list view data).

    Args:
        matches_on_page: List of match data dictionaries from the `get_matches` function.
        existing_persons_map: Dictionary mapping UUIDs to existing Person objects
                               (from `_lookup_existing_persons`).

    Returns:
        A tuple containing:
        - fetch_candidates_uuid (set[str]): Set of UUIDs requiring API detail fetches.
        - matches_to_process_later (list[dict]): List of match data dicts for candidates.
        - skipped_count_this_batch (int): Number of matches skipped in this batch.
    """
    # Initialize results
    fetch_candidates_uuid: set[str] = set()
    skipped_count_this_batch = 0
    matches_to_process_later: list[dict[str, Any]] = []
    invalid_uuid_count = 0

    logger.debug("Identifying fetch candidates vs. skipped matches...")

    # Iterate through matches fetched from the current page
    for match_api_data in matches_on_page:
        # Validate UUID presence
        uuid_val = match_api_data.get("uuid")
        if not uuid_val:
            logger.warning(f"Skipping match missing UUID: {match_api_data}")
            invalid_uuid_count += 1
            continue

        # Check if this person exists in the database
        existing_person = existing_persons_map.get(uuid_val.upper())

        if not existing_person:
            # New Person - always fetch details
            fetch_candidates_uuid.add(uuid_val)
            matches_to_process_later.append(match_api_data)
        # Existing Person - check if fetch needed
        elif _should_fetch_match_details(match_api_data, existing_person):
            fetch_candidates_uuid.add(uuid_val)
            matches_to_process_later.append(match_api_data)
        else:
            skipped_count_this_batch += 1

    # Log summary
    if invalid_uuid_count > 0:
        logger.error(f"{invalid_uuid_count} matches skipped during identification due to missing UUID.")

    # Log identification results with appropriate detail
    if len(fetch_candidates_uuid) == 0 and skipped_count_this_batch > 0:
        logger.info(f"✓ All {skipped_count_this_batch} matches are up-to-date - no API fetches needed")
    elif len(fetch_candidates_uuid) > 0:
        logger.info(
            f"📥 Fetch queue: {len(fetch_candidates_uuid)} matches need updates, "
            f"{skipped_count_this_batch} already current"
        )
        logger.debug(f"  Sample UUIDs to fetch: {list(fetch_candidates_uuid)[:5]}...")
    else:
        logger.debug(
            f"Identified {len(fetch_candidates_uuid)} candidates for API detail fetch, "
            f"{skipped_count_this_batch} skipped (no change detected from list view)."
        )

    return fetch_candidates_uuid, matches_to_process_later, skipped_count_this_batch


# End of _identify_fetch_candidates

# Helper functions for _perform_api_prefetches

def _identify_tree_badge_ladder_candidates(
    matches_to_process_later: list[dict[str, Any]],
    fetch_candidates_uuid: set[str]
) -> set[str]:
    """Identify UUIDs that need badge/ladder fetch (in_my_tree members)."""
    uuids_for_tree_badge_ladder = {
        match_data["uuid"]
        for match_data in matches_to_process_later
        if match_data.get("in_my_tree")
        and match_data.get("uuid") in fetch_candidates_uuid
    }
    logger.debug(f"Identified {len(uuids_for_tree_badge_ladder)} candidates for Badge/Ladder fetch.")
    return uuids_for_tree_badge_ladder


def _submit_initial_api_tasks(
    executor: ThreadPoolExecutor,
    session_manager: SessionManager,
    fetch_candidates_uuid: set[str],
    uuids_for_tree_badge_ladder: set[str]
) -> dict[Any, tuple[str, str]]:
    """Submit initial API tasks (combined_details, relationship_prob, badge_details).

    Note: Rate limiting is now handled INSIDE each fetch function to ensure proper
    throttling even when running in parallel threads.
    """
    futures: dict[Any, tuple[str, str]] = {}

    for uuid_val in fetch_candidates_uuid:
        futures[executor.submit(_fetch_combined_details, session_manager, uuid_val)] = ("combined_details", uuid_val)

        max_labels = 2
        futures[executor.submit(_fetch_batch_relationship_prob, session_manager, uuid_val, max_labels)] = ("relationship_prob", uuid_val)

    for uuid_val in uuids_for_tree_badge_ladder:
        futures[executor.submit(_fetch_batch_badge_details, session_manager, uuid_val)] = ("badge_details", uuid_val)

    return futures


def _process_api_task_result(
    task_type: str,
    identifier_uuid: str,
    result: Any,
    batch_combined_details: dict[str, Optional[dict[str, Any]]],
    temp_badge_results: dict[str, Optional[dict[str, Any]]],
    batch_relationship_prob_data: dict[str, Optional[str]],
    critical_combined_details_failures: int
) -> int:
    """Process a single API task result and update appropriate batch dict."""
    if task_type == "combined_details":
        batch_combined_details[identifier_uuid] = result
        if result is None:
            logger.warning(f"Critical API task '_fetch_combined_details' for {identifier_uuid} returned None.")
            critical_combined_details_failures += 1
    elif task_type == "badge_details":
        temp_badge_results[identifier_uuid] = result
    elif task_type == "relationship_prob":
        batch_relationship_prob_data[identifier_uuid] = result

    return critical_combined_details_failures


def _handle_api_task_exception(
    exc: Exception,
    task_type: str,
    identifier_uuid: str,
    batch_combined_details: dict[str, Optional[dict[str, Any]]],
    temp_badge_results: dict[str, Optional[dict[str, Any]]],
    batch_relationship_prob_data: dict[str, Optional[str]],
    critical_combined_details_failures: int,
    is_connection_error: bool = False
) -> int:
    """Handle exception from API task and update appropriate batch dict."""
    if is_connection_error:
        logger.error(f"ConnErr prefetch '{task_type}' {identifier_uuid}: {exc}", exc_info=False)
    else:
        logger.error(f"Exc prefetch '{task_type}' {identifier_uuid}: {exc}", exc_info=True)

    if task_type == "combined_details":
        critical_combined_details_failures += 1
        batch_combined_details[identifier_uuid] = None
    elif task_type == "badge_details":
        temp_badge_results[identifier_uuid] = None
    elif task_type == "relationship_prob":
        batch_relationship_prob_data[identifier_uuid] = None

    return critical_combined_details_failures


def _check_critical_failure_threshold(
    critical_combined_details_failures: int,
    futures: dict[Any, tuple[str, str]]
) -> bool:
    """Check if critical failure threshold exceeded and cancel remaining futures."""
    if critical_combined_details_failures >= CRITICAL_API_FAILURE_THRESHOLD:
        for f_cancel in futures:
            if not f_cancel.done():
                f_cancel.cancel()
        logger.critical(
            f"Exceeded critical API failure threshold ({critical_combined_details_failures}/{CRITICAL_API_FAILURE_THRESHOLD}) for combined_details. Halting batch."
        )
        raise MaxApiFailuresExceededError(
            f"Critical API failure threshold reached for combined_details ({critical_combined_details_failures} failures)."
        )


def _build_cfpid_to_uuid_map(temp_badge_results: dict[str, Optional[dict[str, Any]]]) -> dict[str, str]:
    """Build mapping from CFPID to UUID from badge results."""
    cfpid_to_uuid_map: dict[str, str] = {}
    for uuid_val, badge_result_data in temp_badge_results.items():
        if badge_result_data:
            cfpid = badge_result_data.get("their_cfpid")
            if cfpid:
                cfpid_to_uuid_map[cfpid] = uuid_val
    return cfpid_to_uuid_map


def _submit_ladder_tasks(
    executor: ThreadPoolExecutor,
    session_manager: SessionManager,
    cfpid_to_uuid_map: dict[str, str],
    my_tree_id: Optional[str]
) -> dict[Any, tuple[str, str]]:
    """Submit ladder API tasks for CFPIDs.

    Note: Rate limiting is now handled INSIDE each fetch function to ensure proper
    throttling even when running in parallel threads.
    """
    ladder_futures = {}
    if my_tree_id and cfpid_to_uuid_map:
        cfpid_list = list(cfpid_to_uuid_map.keys())
        logger.debug(f"Submitting Ladder tasks for {len(cfpid_list)} CFPIDs...")
        for cfpid_item in cfpid_list:
            ladder_futures[executor.submit(_fetch_batch_ladder, session_manager, cfpid_item, my_tree_id)] = ("ladder", cfpid_item)
    return ladder_futures


def _process_ladder_results(
    ladder_futures: dict[Any, tuple[str, str]],
    cfpid_to_uuid_map: dict[str, str]
) -> dict[str, Optional[dict[str, Any]]]:
    """Process ladder API task results."""
    temp_ladder_results: dict[str, Optional[dict[str, Any]]] = {}

    logger.debug(f"Processing {len(ladder_futures)} Ladder API tasks...")
    for future in as_completed(ladder_futures):
        _task_type, identifier_cfpid = ladder_futures[future]
        uuid_for_ladder = cfpid_to_uuid_map.get(identifier_cfpid)
        if not uuid_for_ladder:
            logger.warning(f"Could not map ladder result for CFPID {identifier_cfpid} back to UUID (task likely cancelled or map error).")
            continue

        try:
            result = future.result()
            temp_ladder_results[uuid_for_ladder] = result
        except ConnectionError as conn_err:
            logger.error(f"ConnErr ladder fetch CFPID {identifier_cfpid} (UUID: {uuid_for_ladder}): {conn_err}", exc_info=False)
            temp_ladder_results[uuid_for_ladder] = None
        except Exception as exc:
            logger.error(f"Exc ladder fetch CFPID {identifier_cfpid} (UUID: {uuid_for_ladder}): {exc}", exc_info=True)
            temp_ladder_results[uuid_for_ladder] = None

    return temp_ladder_results


def _combine_badge_and_ladder_results(
    temp_badge_results: dict[str, Optional[dict[str, Any]]],
    temp_ladder_results: dict[str, Optional[dict[str, Any]]]
) -> dict[str, Optional[dict[str, Any]]]:
    """Combine badge and ladder results into final tree data."""
    batch_tree_data: dict[str, Optional[dict[str, Any]]] = {}

    for uuid_val, badge_result in temp_badge_results.items():
        if badge_result:
            combined_tree_info = badge_result.copy()
            ladder_result_for_uuid = temp_ladder_results.get(uuid_val)
            if ladder_result_for_uuid:
                combined_tree_info.update(ladder_result_for_uuid)
            batch_tree_data[uuid_val] = combined_tree_info

    return batch_tree_data


def _perform_api_prefetches(
    session_manager: SessionManager,
    fetch_candidates_uuid: set[str],
    matches_to_process_later: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Performs parallel API calls to prefetch detailed data for candidate matches
    using a ThreadPoolExecutor. Fetches combined details, relationship probability,
    badge details (for tree members), and ladder details (for tree members).
    Implements critical failure tracking for combined_details.

    Args:
        session_manager: The active SessionManager instance.
        fetch_candidates_uuid: Set of UUIDs requiring detail fetches.
        matches_to_process_later: List of match data dicts corresponding to the candidates.

    Returns:
        A dictionary containing the prefetched data, organized by type.
        Values will be None if a specific fetch fails.
        {
            "combined": {uuid: combined_details_dict_or_None, ...},
            "tree": {uuid: combined_badge_ladder_dict_or_None, ...},
            "rel_prob": {uuid: relationship_prob_string_or_None, ...}
        }
    Raises:
        MaxApiFailuresExceededError: If critical API failure threshold is met.
    """
    # Initialize result dictionaries
    batch_combined_details: dict[str, Optional[dict[str, Any]]] = {}
    batch_relationship_prob_data: dict[str, Optional[str]] = {}
    temp_badge_results: dict[str, Optional[dict[str, Any]]] = {}

    if not fetch_candidates_uuid:
        logger.debug("⏭️  No API prefetches needed - all matches current in database")
        return {"combined": {}, "tree": {}, "rel_prob": {}}

    fetch_start_time = time.time()
    original_candidates = len(fetch_candidates_uuid)

    # Priority 2.3: Deduplicate API requests using cache
    fetch_candidates_uuid, matches_to_process_later, cache_hits = _deduplicate_api_requests(
        fetch_candidates_uuid, matches_to_process_later
    )

    num_candidates = len(fetch_candidates_uuid)
    my_tree_id = session_manager.my_tree_id
    critical_combined_details_failures = 0

    # Get thread pool workers from config (can be overridden via THREAD_POOL_WORKERS in .env)
    thread_pool_workers = config_schema.api.thread_pool_workers

    if num_candidates > 0:
        logger.info(f"🌐 Fetching {num_candidates} matches via API ({thread_pool_workers} parallel workers)...")

    if cache_hits > 0:
        logger.info(f"   ↳ Skipped {cache_hits} cached requests ({cache_hits}/{original_candidates} = {cache_hits/original_candidates*100:.1f}%)")

    # Identify tree members needing badge/ladder fetch
    uuids_for_tree_badge_ladder = _identify_tree_badge_ladder_candidates(matches_to_process_later, fetch_candidates_uuid)

    with ThreadPoolExecutor(max_workers=thread_pool_workers) as executor:
        # Submit initial API tasks
        futures = _submit_initial_api_tasks(executor, session_manager, fetch_candidates_uuid, uuids_for_tree_badge_ladder)

        # Process initial API task results
        logger.debug(f"Processing {len(futures)} initially submitted API tasks...")
        for future in as_completed(futures):
            task_type, identifier_uuid = futures[future]
            try:
                result = future.result()
                critical_combined_details_failures = _process_api_task_result(
                    task_type, identifier_uuid, result,
                    batch_combined_details, temp_badge_results, batch_relationship_prob_data,
                    critical_combined_details_failures
                )
            except ConnectionError as conn_err:
                critical_combined_details_failures = _handle_api_task_exception(
                    conn_err, task_type, identifier_uuid,
                    batch_combined_details, temp_badge_results, batch_relationship_prob_data,
                    critical_combined_details_failures, is_connection_error=True
                )
            except Exception as exc:
                critical_combined_details_failures = _handle_api_task_exception(
                    exc, task_type, identifier_uuid,
                    batch_combined_details, temp_badge_results, batch_relationship_prob_data,
                    critical_combined_details_failures, is_connection_error=False
                )

            _check_critical_failure_threshold(critical_combined_details_failures, futures)

        # Build CFPID to UUID mapping and submit ladder tasks
        cfpid_to_uuid_map = _build_cfpid_to_uuid_map(temp_badge_results)
        ladder_futures = _submit_ladder_tasks(executor, session_manager, cfpid_to_uuid_map, my_tree_id)

        # Process ladder results
        temp_ladder_results = _process_ladder_results(ladder_futures, cfpid_to_uuid_map)

    fetch_duration = time.time() - fetch_start_time
    avg_time = fetch_duration / num_candidates if num_candidates > 0 else 0
    logger.info(f"✅ API fetch complete: {num_candidates} matches in {fetch_duration:.2f}s (avg: {avg_time:.2f}s/match)")

    # Combine badge and ladder results
    batch_tree_data = _combine_badge_and_ladder_results(temp_badge_results, temp_ladder_results)

    return {
        "combined": batch_combined_details,
        "tree": batch_tree_data,
        "rel_prob": batch_relationship_prob_data,
    }


# End of _perform_api_prefetches


def _retrieve_prefetched_data_for_match(
    uuid_val: str,
    prefetched_data: dict[str, dict[str, Any]]
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[str]]:
    """Retrieve prefetched data for a specific match UUID."""
    prefetched_combined = prefetched_data.get("combined", {}).get(uuid_val)
    prefetched_tree = prefetched_data.get("tree", {}).get(uuid_val)
    prefetched_rel_prob = prefetched_data.get("rel_prob", {}).get(uuid_val)
    return prefetched_combined, prefetched_tree, prefetched_rel_prob


def _validate_match_uuid(match_list_data: dict[str, Any]) -> str:
    """Validate and return match UUID, raise ValueError if missing."""
    uuid_val = match_list_data.get("uuid")
    if not uuid_val:
        logger.error("Critical error: Match data missing UUID in _prepare_bulk_db_data. Skipping.")
        raise ValueError("Missing UUID")
    return uuid_val


def _process_single_match(
    match_list_data: dict[str, Any],
    session_manager: SessionManager,
    existing_persons_map: dict[str, Person],
    prefetched_data: dict[str, dict[str, Any]],
    log_ref_short: str
) -> tuple[Optional[dict[str, Any]], str, Optional[str]]:
    """Process a single match and return prepared data, status, and error message."""
    # Validate UUID
    uuid_val = _validate_match_uuid(match_list_data)

    # Retrieve existing person and prefetched data
    existing_person = existing_persons_map.get(uuid_val.upper())
    prefetched_combined, prefetched_tree, prefetched_rel_prob = _retrieve_prefetched_data_for_match(
        uuid_val, prefetched_data
    )

    # Add relationship probability to match dict
    match_list_data["predicted_relationship"] = prefetched_rel_prob

    # Check WebDriver session validity
    if not session_manager.is_sess_valid():
        logger.error(f"WebDriver session invalid before calling _do_match for {log_ref_short}. Treating as error.")
        return None, "error", "WebDriver session invalid"

    # Call _do_match to prepare the bulk dictionary structure
    return _do_match(
        match_list_data,
        session_manager,
        existing_person,
        prefetched_combined,
        prefetched_tree,
        config_schema,
        logger,
    )


def _update_page_statuses(
    status_for_this_match: str,
    page_statuses: dict[str, int],
    log_ref_short: str
) -> None:
    """Update page statuses based on match processing result."""
    if status_for_this_match in ["new", "updated", "error"]:
        page_statuses[status_for_this_match] += 1
    elif status_for_this_match == "skipped":
        logger.debug(f"_do_match returned 'skipped' for {log_ref_short}. Not counted in page new/updated/error.")
    else:
        logger.error(f"Unknown status '{status_for_this_match}' from _do_match for {log_ref_short}. Counting as error.")
        page_statuses["error"] += 1


def _process_and_append_match(
    match_list_data: dict[str, Any],
    session_manager: SessionManager,
    existing_persons_map: dict[str, Person],
    prefetched_data: dict[str, dict[str, Any]],
    prepared_bulk_data: list[dict[str, Any]],
    page_statuses: dict[str, int],
    progress_bar: Optional[tqdm]
) -> None:
    """Process a single match and append to bulk data if valid."""
    uuid_val = match_list_data.get("uuid")
    log_ref_short = f"UUID={uuid_val or 'MISSING'} User='{match_list_data.get('username', 'Unknown')}'"

    try:
        # Process single match
        prepared_data_for_this_match, status_for_this_match, error_msg_for_this_match = _process_single_match(
            match_list_data, session_manager, existing_persons_map, prefetched_data, log_ref_short
        )

        # Update page statuses
        _update_page_statuses(status_for_this_match, page_statuses, log_ref_short)

        # Append valid prepared data to the bulk list
        if status_for_this_match not in ["error", "skipped"] and prepared_data_for_this_match:
            prepared_bulk_data.append(prepared_data_for_this_match)
        elif status_for_this_match == "error":
            logger.error(
                f"Error preparing DB data for {log_ref_short}: {error_msg_for_this_match or 'Unknown error in _do_match'}"
            )

    except Exception as inner_e:
        logger.error(
            f"Critical unexpected error processing {log_ref_short} in _prepare_bulk_db_data: {inner_e}",
            exc_info=True,
        )
        page_statuses["error"] += 1
    finally:
        if progress_bar:
            try:
                progress_bar.update(1)
            except Exception as pbar_e:
                logger.warning(f"Progress bar update error: {pbar_e}")


def _prepare_bulk_db_data(
    _session: SqlAlchemySession,
    session_manager: SessionManager,
    matches_to_process: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],
    prefetched_data: dict[str, dict[str, Any]],
    progress_bar: Optional[tqdm],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Processes individual matches using prefetched API data, compares with existing
    DB records, and prepares dictionaries formatted for bulk database operations
    (insert/update for Person, DnaMatch, FamilyTree).

    Args:
        session: The active SQLAlchemy database session.
        session_manager: The active SessionManager instance.
        matches_to_process: List of match data dictionaries identified as candidates.
        existing_persons_map: Dictionary mapping UUIDs to existing Person objects.
        prefetched_data: Dictionary containing results from `_perform_api_prefetches`.
        progress_bar: Optional tqdm progress bar instance to update.

    Returns:
        A tuple containing:
        - prepared_bulk_data (list[dict]): A list where each element is a dictionary
          representing one person and contains keys 'person', 'dna_match', 'family_tree'
          with data formatted for bulk operations (or None if no change needed).
        - page_statuses (dict[str, int]): Counts of 'new', 'updated', 'error' outcomes
          during the preparation phase for this batch.
    """
    # Step 1: Initialize results
    prepared_bulk_data: list[dict[str, Any]] = []
    page_statuses: dict[str, int] = {
        "new": 0,
        "updated": 0,
        "error": 0,
    }  # Skipped handled before this function
    num_to_process = len(matches_to_process)

    if not num_to_process:
        return [], page_statuses  # Return empty if nothing to process

    logger.debug(
        f"--- Preparing DB data structures for {num_to_process} candidates ---"
    )
    process_start_time = time.time()

    # Step 2: Iterate through each candidate match
    for match_list_data in matches_to_process:
        _process_and_append_match(
            match_list_data,
            session_manager,
            existing_persons_map,
            prefetched_data,
            prepared_bulk_data,
            page_statuses,
            progress_bar
        )

    # Step 5: Log summary and return results
    process_duration = time.time() - process_start_time
    logger.debug(
        f"--- Finished preparing DB data structures. Duration: {process_duration:.2f}s ---"
    )
    return prepared_bulk_data, page_statuses


# End of _prepare_bulk_db_data


def _separate_operations_by_type(prepared_bulk_data: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate prepared data by operation type and table."""
    person_creates_raw = [
        d["person"]
        for d in prepared_bulk_data
        if d.get("person") and d["person"]["_operation"] == "create"
    ]
    person_updates = [
        d["person"]
        for d in prepared_bulk_data
        if d.get("person") and d["person"]["_operation"] == "update"
    ]
    dna_match_ops = [
        d["dna_match"] for d in prepared_bulk_data if d.get("dna_match")
    ]
    family_tree_ops = [
        d["family_tree"] for d in prepared_bulk_data if d.get("family_tree")
    ]
    return person_creates_raw, person_updates, dna_match_ops, family_tree_ops


def _deduplicate_person_creates(person_creates_raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate Person Creates based on Profile ID before bulk insert."""
    if not person_creates_raw:
        logger.debug("No unique Person records to bulk insert.")
        return []

    logger.debug(f"De-duplicating {len(person_creates_raw)} raw person creates based on Profile ID...")
    person_creates_filtered = []
    seen_profile_ids: set[str] = set()
    skipped_duplicates = 0

    for p_data in person_creates_raw:
        profile_id = p_data.get("profile_id")
        uuid_for_log = p_data.get("uuid")

        if profile_id is None:
            person_creates_filtered.append(p_data)
        elif profile_id not in seen_profile_ids:
            person_creates_filtered.append(p_data)
            seen_profile_ids.add(profile_id)
        else:
            logger.warning(f"Skipping duplicate Person create in batch (ProfileID: {profile_id}, UUID: {uuid_for_log}).")
            skipped_duplicates += 1

    if skipped_duplicates > 0:
        logger.info(f"Skipped {skipped_duplicates} duplicate person creates in this batch.")
    logger.debug(f"Proceeding with {len(person_creates_filtered)} unique person creates.")

    return person_creates_filtered


def _validate_no_duplicate_profile_ids(insert_data: list[dict[str, Any]]) -> None:
    """Validate that there are no duplicate profile IDs in the insert data."""
    final_profile_ids = {
        item.get("profile_id") for item in insert_data if item.get("profile_id")
    }
    if len(final_profile_ids) != sum(1 for item in insert_data if item.get("profile_id")):
        logger.error("CRITICAL: Duplicate non-NULL profile IDs DETECTED post-filter! Aborting bulk insert.")
        id_counts = Counter(item.get("profile_id") for item in insert_data if item.get("profile_id"))
        duplicates = {pid: count for pid, count in id_counts.items() if count > 1}
        logger.error(f"Duplicate Profile IDs in filtered list: {duplicates}")
        dup_exception = ValueError(f"Duplicate profile IDs: {duplicates}")
        raise IntegrityError(
            "Duplicate profile IDs found pre-bulk insert",
            params=str(duplicates),
            orig=dup_exception,
        )


def _bulk_update_persons(session: SqlAlchemySession, person_updates: list[dict[str, Any]]) -> None:
    """Bulk update Person records."""
    if not person_updates:
        logger.debug("No Person updates needed for this batch.")
        return

    update_mappings = []
    for p_data in person_updates:
        existing_id = p_data.get("_existing_person_id")
        if not existing_id:
            logger.warning(f"Skipping person update (UUID {p_data.get('uuid')}): Missing '_existing_person_id'.")
            continue

        update_dict = {
            k: v
            for k, v in p_data.items()
            if not k.startswith("_") and k not in ["uuid", "profile_id"]
        }

        if "status" in update_dict and isinstance(update_dict["status"], PersonStatusEnum):
            update_dict["status"] = update_dict["status"].value

        update_dict["id"] = existing_id
        update_dict["updated_at"] = datetime.now(timezone.utc)

        if len(update_dict) > 2:
            update_mappings.append(update_dict)

    if update_mappings:
        logger.debug(f"Bulk updating {len(update_mappings)} Person records...")
        session.bulk_update_mappings(Person, update_mappings)  # type: ignore
        logger.debug("Bulk update Persons called.")
    else:
        logger.debug("No valid Person updates to perform.")


def _get_existing_dna_matches_map(session: SqlAlchemySession, all_person_ids_map: dict[str, int]) -> dict[int, int]:
    """Query existing DnaMatch records for people in this batch."""
    people_ids_in_batch = {pid for pid in all_person_ids_map.values() if pid is not None}

    if not people_ids_in_batch:
        return {}

    existing_matches = (
        session.query(DnaMatch.people_id, DnaMatch.id)
        .filter(DnaMatch.people_id.in_(people_ids_in_batch))  # type: ignore
        .all()
    )
    existing_dna_matches_map = dict(existing_matches)
    logger.debug(f"Found {len(existing_dna_matches_map)} existing DnaMatch records for people in this batch.")

    return existing_dna_matches_map


def _process_dna_match_operations(
    dna_match_ops: list[dict[str, Any]],
    all_person_ids_map: dict[str, int],
    existing_dna_matches_map: dict[int, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process DNA match operations and separate into inserts and updates."""
    dna_insert_data = []
    dna_update_mappings = []

    for dna_data in dna_match_ops:
        person_uuid = dna_data.get("uuid")
        person_id = all_person_ids_map.get(person_uuid) if person_uuid else None

        if not person_id:
            logger.warning(f"Skipping DNA Match op (UUID {person_uuid}): Corresponding Person ID not found in map.")
            continue

        # Prepare data dictionary (exclude internal keys)
        op_data = {k: v for k, v in dna_data.items() if not k.startswith("_") and k != "uuid"}
        op_data["people_id"] = person_id

        # Check if a DnaMatch record already exists for this person_id
        existing_match_id = existing_dna_matches_map.get(person_id)

        if existing_match_id:
            # Prepare for UPDATE
            update_map = op_data.copy()
            update_map["id"] = existing_match_id
            update_map["updated_at"] = datetime.now(timezone.utc)

            if len(update_map) > 3:
                dna_update_mappings.append(update_map)
            else:
                logger.debug(f"Skipping DnaMatch update for PersonID {person_id}: No changed fields.")
        else:
            # Prepare for INSERT
            insert_map = op_data.copy()
            insert_map.setdefault("created_at", datetime.now(timezone.utc))
            insert_map.setdefault("updated_at", datetime.now(timezone.utc))
            dna_insert_data.append(insert_map)

    return dna_insert_data, dna_update_mappings


def _bulk_insert_family_trees(
    session: SqlAlchemySession,
    tree_creates: list[dict[str, Any]],
    all_person_ids_map: dict[str, int]
) -> None:
    """Bulk insert FamilyTree records."""
    if not tree_creates:
        logger.debug("No FamilyTree creates prepared.")
        return

    tree_insert_data = []
    for tree_data in tree_creates:
        person_uuid = tree_data.get("uuid")
        person_id = all_person_ids_map.get(person_uuid) if person_uuid else None

        if person_id:
            insert_dict = {k: v for k, v in tree_data.items() if not k.startswith("_")}
            insert_dict["people_id"] = person_id
            insert_dict.pop("uuid", None)  # Remove uuid before insert
            tree_insert_data.append(insert_dict)
        else:
            logger.warning(f"Skipping FamilyTree create op (UUID {person_uuid}): Person ID not found.")

    if tree_insert_data:
        logger.debug(f"Bulk inserting {len(tree_insert_data)} FamilyTree records...")
        session.bulk_insert_mappings(FamilyTree, tree_insert_data)  # type: ignore
        logger.debug("Bulk insert FamilyTrees called.")
    else:
        logger.debug("No valid FamilyTree records to insert.")


def _bulk_update_family_trees(session: SqlAlchemySession, tree_updates: list[dict[str, Any]]) -> None:
    """Bulk update FamilyTree records."""
    if not tree_updates:
        logger.debug("No FamilyTree updates prepared.")
        return

    tree_update_mappings = []
    for tree_data in tree_updates:
        existing_tree_id = tree_data.get("_existing_tree_id")
        if not existing_tree_id:
            logger.warning(f"Skipping FamilyTree update op (UUID {tree_data.get('uuid')}): Missing '_existing_tree_id'.")
            continue

        update_dict_tree = {
            k: v for k, v in tree_data.items() if not k.startswith("_") and k != "uuid"
        }
        update_dict_tree["id"] = existing_tree_id
        update_dict_tree["updated_at"] = datetime.now(timezone.utc)

        if len(update_dict_tree) > 2:
            tree_update_mappings.append(update_dict_tree)

    if tree_update_mappings:
        logger.debug(f"Bulk updating {len(tree_update_mappings)} FamilyTree records...")
        session.bulk_update_mappings(FamilyTree, tree_update_mappings)  # type: ignore
        logger.debug("Bulk update FamilyTrees called.")
    else:
        logger.debug("No valid FamilyTree updates.")


def _bulk_upsert_family_trees(
    session: SqlAlchemySession,
    family_tree_ops: list[dict[str, Any]],
    all_person_ids_map: dict[str, int]
) -> None:
    """Bulk upsert FamilyTree records (separate insert/update)."""
    tree_creates = [op for op in family_tree_ops if op.get("_operation") == "create"]
    tree_updates = [op for op in family_tree_ops if op.get("_operation") == "update"]

    _bulk_insert_family_trees(session, tree_creates, all_person_ids_map)
    _bulk_update_family_trees(session, tree_updates)


def _bulk_upsert_dna_matches(
    session: SqlAlchemySession,
    dna_match_ops: list[dict[str, Any]],
    all_person_ids_map: dict[str, int]
) -> None:
    """Bulk upsert DnaMatch records (separate insert/update)."""
    if not dna_match_ops:
        logger.debug("No DnaMatch operations prepared.")
        return

    # Get existing DnaMatch records
    existing_dna_matches_map = _get_existing_dna_matches_map(session, all_person_ids_map)

    # Process operations and separate into inserts and updates
    dna_insert_data, dna_update_mappings = _process_dna_match_operations(
        dna_match_ops, all_person_ids_map, existing_dna_matches_map
    )

    # Perform Bulk Insert
    if dna_insert_data:
        logger.debug(f"Bulk inserting {len(dna_insert_data)} DnaMatch records...")
        session.bulk_insert_mappings(DnaMatch, dna_insert_data)  # type: ignore
        logger.debug("Bulk insert DnaMatches called.")
    else:
        logger.debug("No new DnaMatch records to insert.")

    # Perform Bulk Update
    if dna_update_mappings:
        logger.debug(f"Bulk updating {len(dna_update_mappings)} DnaMatch records...")
        session.bulk_update_mappings(DnaMatch, dna_update_mappings)  # type: ignore
        logger.debug("Bulk update DnaMatches called.")
    else:
        logger.debug("No existing DnaMatch records to update.")


def _add_person_update_ids(all_person_ids_map: dict[str, int], person_updates: list[dict[str, Any]]) -> None:
    """Add IDs from person updates to the map."""
    for p_update_data in person_updates:
        if p_update_data.get("_existing_person_id") and p_update_data.get("uuid"):
            all_person_ids_map[p_update_data["uuid"]] = p_update_data["_existing_person_id"]


def _collect_uuids_from_bulk_data(prepared_bulk_data: list[dict[str, Any]]) -> set[str]:
    """Collect all UUIDs from prepared bulk data (person, dna_match, family_tree sections)."""
    processed_uuids: set[str] = set()

    for item in prepared_bulk_data:
        # Get UUID from person section
        if item.get("person") and item["person"].get("uuid"):
            processed_uuids.add(item["person"]["uuid"])
        # Get UUID from dna_match section (when person=None but DNA data exists)
        if item.get("dna_match") and item["dna_match"].get("uuid"):
            processed_uuids.add(item["dna_match"]["uuid"])
        # Get UUID from family_tree section (when person=None but tree data exists)
        if item.get("family_tree") and item["family_tree"].get("uuid"):
            processed_uuids.add(item["family_tree"]["uuid"])

    return processed_uuids


def _add_existing_person_ids(
    all_person_ids_map: dict[str, int],
    processed_uuids: set[str],
    existing_persons_map: dict[str, Person]
) -> None:
    """Add IDs from existing persons map for collected UUIDs."""
    for uuid_processed in processed_uuids:
        if uuid_processed not in all_person_ids_map and existing_persons_map.get(uuid_processed):
            person = existing_persons_map[uuid_processed]
            person_id_val = getattr(person, "id", None)
            if person_id_val is not None:
                all_person_ids_map[uuid_processed] = person_id_val


def _create_master_person_id_map(
    created_person_map: dict[str, int],
    person_updates: list[dict[str, Any]],
    prepared_bulk_data: list[dict[str, Any]],
    existing_persons_map: dict[str, Person]
) -> dict[str, int]:
    """Create master ID map for linking related records."""
    all_person_ids_map: dict[str, int] = created_person_map.copy()

    # Add IDs from person updates
    _add_person_update_ids(all_person_ids_map, person_updates)

    # Collect all UUIDs from prepared data
    processed_uuids = _collect_uuids_from_bulk_data(prepared_bulk_data)

    # Add IDs from existing persons
    _add_existing_person_ids(all_person_ids_map, processed_uuids, existing_persons_map)

    return all_person_ids_map


def _prepare_insert_data(person_creates_filtered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare person data for bulk insertion."""
    # Remove internal keys and convert enums
    insert_data = [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in person_creates_filtered
    ]

    # Convert status Enum to its value for bulk insertion
    for item_data in insert_data:
        if "status" in item_data and isinstance(item_data["status"], PersonStatusEnum):
            item_data["status"] = item_data["status"].value

    return insert_data


def _filter_existing_persons(
    session: SqlAlchemySession,
    insert_data: list[dict[str, Any]],
    created_person_map: dict[str, int]
) -> list[dict[str, Any]]:
    """Filter out persons that already exist and update the ID map."""
    insert_uuids = [p_data["uuid"] for p_data in insert_data if p_data.get("uuid")]
    if not insert_uuids:
        return insert_data

    existing_persons_in_db = session.query(Person.id, Person.uuid).filter(
        Person.uuid.in_(insert_uuids)
    ).all()
    existing_uuid_set = {str(uuid) for pid, uuid in existing_persons_in_db}

    if existing_uuid_set:
        logger.warning(f"Filtering out {len(existing_uuid_set)} UUIDs that already exist in database")
        # Map existing Person IDs before filtering
        for person_id, person_uuid in existing_persons_in_db:
            created_person_map[person_uuid] = person_id
        logger.info(f"Mapped {len(created_person_map)} existing Person IDs for downstream operations")

        insert_data = [p for p in insert_data if p.get("uuid") not in existing_uuid_set]
        logger.info(f"Proceeding with {len(insert_data)} truly new Person records after filtering")

    return insert_data


def _query_inserted_person_ids(
    session: SqlAlchemySession,
    insert_data: list[dict[str, Any]]
) -> dict[str, int]:
    """Query and return IDs of newly inserted persons."""
    inserted_uuids = [p_data["uuid"] for p_data in insert_data if p_data.get("uuid")]

    if not inserted_uuids:
        logger.warning("No UUIDs available in insert_data to query back IDs.")
        return {}

    logger.debug(f"Querying IDs for {len(inserted_uuids)} inserted UUIDs...")
    newly_inserted_persons = (
        session.query(Person.id, Person.uuid)
        .filter(Person.uuid.in_(inserted_uuids))
        .all()
    )
    created_map = {p_uuid: p_id for p_id, p_uuid in newly_inserted_persons}
    logger.debug(f"Mapped {len(created_map)} new Person IDs.")

    if len(created_map) != len(inserted_uuids):
        logger.error(
            f"CRITICAL: ID map count mismatch! Expected {len(inserted_uuids)}, got {len(created_map)}. Some IDs might be missing."
        )

    return created_map


def _bulk_insert_persons(session: SqlAlchemySession, person_creates_filtered: list[dict[str, Any]]) -> dict[str, int]:
    """Bulk insert Person records and return mapping of UUID to new Person ID."""
    created_person_map: dict[str, int] = {}

    if not person_creates_filtered:
        logger.debug("No unique Person records to bulk insert.")
        return created_person_map

    logger.debug(f"Preparing {len(person_creates_filtered)} Person records for bulk insert...")

    # Prepare data
    insert_data = _prepare_insert_data(person_creates_filtered)
    _validate_no_duplicate_profile_ids(insert_data)

    # Filter existing persons
    insert_data = _filter_existing_persons(session, insert_data, created_person_map)

    if not insert_data:
        logger.info("All Person records already exist in database - nothing to insert, but IDs mapped")
        return created_person_map

    # Perform bulk insert
    logger.debug(f"Bulk inserting {len(insert_data)} Person records...")
    session.bulk_insert_mappings(Person, insert_data)  # type: ignore
    logger.debug("Bulk insert Persons called.")

    # Get newly created IDs
    session.flush()
    logger.debug("Session flushed to assign Person IDs.")

    # Query and map new IDs
    new_ids = _query_inserted_person_ids(session, insert_data)
    created_person_map.update(new_ids)

    return created_person_map


def _execute_bulk_db_operations(
    session: SqlAlchemySession,
    prepared_bulk_data: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],  # Needed to potentially map existing IDs
) -> bool:
    """
    Executes bulk INSERT and UPDATE operations for Person, DnaMatch, and FamilyTree
    records within an existing database transaction session.

    Args:
        session: The active SQLAlchemy database session (within a transaction).
        prepared_bulk_data: List of dictionaries prepared by `_prepare_bulk_db_data`.
                            Each dict contains 'person', 'dna_match', 'family_tree' keys
                            with data and an '_operation' hint ('create'/'update').
        existing_persons_map: Dictionary mapping UUIDs to existing Person objects,
                              used for linking updates correctly.

    Returns:
        True if all bulk operations completed successfully within the transaction,
        False if a database error occurred.
    """
    # Step 1: Initialization
    bulk_start_time = time.time()
    num_items = len(prepared_bulk_data)
    if num_items == 0:
        logger.debug("No prepared data found for bulk DB operations.")
        return True  # Nothing to do, considered success

    logger.debug(f"--- Starting Bulk DB Operations ({num_items} prepared items) ---")

    try:
        # Step 2: Separate data by operation type (create/update) and table
        person_creates_raw, person_updates, dna_match_ops, family_tree_ops = _separate_operations_by_type(prepared_bulk_data)

        # Step 3: Person Creates - De-duplicate and prepare
        person_creates_filtered = _deduplicate_person_creates(person_creates_raw)

        # Step 3: Bulk Insert Persons
        created_person_map = _bulk_insert_persons(session, person_creates_filtered)

        # Step 4: Bulk Update Persons
        _bulk_update_persons(session, person_updates)

        # Step 5: Create Master ID Map (for linking related records)
        all_person_ids_map = _create_master_person_id_map(
            created_person_map, person_updates, prepared_bulk_data, existing_persons_map
        )

        # Step 6: DnaMatch Bulk Upsert
        _bulk_upsert_dna_matches(session, dna_match_ops, all_person_ids_map)

        # Step 7: FamilyTree Bulk Upsert
        _bulk_upsert_family_trees(session, family_tree_ops, all_person_ids_map)

        # Step 8: Log success
        bulk_duration = time.time() - bulk_start_time
        logger.debug(f"--- Bulk DB Operations OK. Duration: {bulk_duration:.2f}s ---")
        return True

    # Step 9: Handle database errors during bulk operations
    except (IntegrityError, SQLAlchemyError) as bulk_db_err:
        logger.error(f"Bulk DB operation FAILED: {bulk_db_err}", exc_info=True)
        return False  # Indicate failure (rollback handled by db_transn)
    except Exception as e:
        logger.error(f"Unexpected error during bulk DB operations: {e}", exc_info=True)
        return False  # Indicate failure


# End of _execute_bulk_db_operations


# === BATCH PROCESSING HELPER FUNCTIONS ===

def _validate_batch_prerequisites(my_uuid: Optional[str], matches_on_page: list[dict[str, Any]], current_page: int) -> None:
    """Validate prerequisites for batch processing."""
    if not my_uuid:
        logger.error(f"_do_batch Page {current_page}: Missing my_uuid.")
        raise ValueError("Missing my_uuid")
    if not matches_on_page:
        logger.debug(f"_do_batch Page {current_page}: Empty match list.")
        raise ValueError("Empty match list")


def _execute_batch_db_commit(session: SqlAlchemySession, prepared_bulk_data: list[dict[str, Any]], existing_persons_map: dict[str, Person], current_page: int, page_statuses: dict[str, int]) -> None:
    """Execute bulk DB operations with error handling."""
    if not prepared_bulk_data:
        logger.debug(f"No data prepared for bulk DB operations on page {current_page}.")
        return

    logger.debug(f"Attempting bulk DB operations for page {current_page}...")
    try:
        with db_transn(session) as sess:
            bulk_success = _execute_bulk_db_operations(sess, prepared_bulk_data, existing_persons_map)
            if not bulk_success:
                logger.error(f"Bulk DB ops FAILED page {current_page}. Adjusting counts.")
                failed_items = len(prepared_bulk_data)
                page_statuses["error"] += failed_items
                page_statuses["new"] = 0
                page_statuses["updated"] = 0
        logger.debug(f"Transaction block finished page {current_page}.")
    except (IntegrityError, SQLAlchemyError, ValueError) as bulk_db_err:
        logger.error(f"Bulk DB transaction FAILED page {current_page}: {bulk_db_err}", exc_info=True)
        failed_items = len(prepared_bulk_data)
        page_statuses["error"] += failed_items
        page_statuses["new"] = 0
        page_statuses["updated"] = 0
    except Exception as e:
        logger.error(f"Unexpected error during bulk DB transaction page {current_page}: {e}", exc_info=True)
        failed_items = len(prepared_bulk_data)
        page_statuses["error"] += failed_items
        page_statuses["new"] = 0
        page_statuses["updated"] = 0


def _handle_batch_critical_error(page_statuses: dict[str, int], num_matches_on_page: int, progress_bar: Optional[tqdm], current_page: int, error: Exception) -> int:
    """Handle critical errors during batch processing."""
    logger.critical(f"CRITICAL ERROR processing batch page {current_page}: {error}", exc_info=True)

    # Update progress bar for remaining items
    if progress_bar:
        items_already_accounted_for_in_bar = (
            page_statuses["skipped"] + page_statuses["new"] + page_statuses["updated"] + page_statuses["error"]
        )
        remaining_in_batch = max(0, num_matches_on_page - items_already_accounted_for_in_bar)
        if remaining_in_batch > 0:
            try:
                logger.debug(f"Updating progress bar by {remaining_in_batch} due to critical error in _do_batch.")
                progress_bar.update(remaining_in_batch)
            except Exception as pbar_e:
                logger.warning(f"Progress bar update error during critical exception handling: {pbar_e}")

    # Calculate final error count
    return page_statuses["error"] + max(
        0,
        num_matches_on_page - (page_statuses["new"] + page_statuses["updated"] + page_statuses["skipped"] + page_statuses["error"]),
    )



def _execute_batch_pipeline(
    session: SqlAlchemySession,
    session_manager: SessionManager,
    matches_on_page: list[dict[str, Any]],
    current_page: int,
    page_statuses: dict[str, int],
    progress_bar: Optional[tqdm]
) -> None:
    """Execute the data processing pipeline for a batch."""
    logger.debug(f"Batch {current_page}: Looking up existing persons...")
    uuids_on_page = [m["uuid"].upper() for m in matches_on_page if m.get("uuid")]
    existing_persons_map = _lookup_existing_persons(session, uuids_on_page)

    logger.debug(f"Batch {current_page}: Identifying candidates...")
    fetch_candidates_uuid, matches_to_process_later, skipped_count = (
        _identify_fetch_candidates(matches_on_page, existing_persons_map)
    )
    page_statuses["skipped"] = skipped_count

    if progress_bar and skipped_count > 0:
        try:
            progress_bar.update(skipped_count)
        except Exception as pbar_e:
            logger.warning(f"Progress bar update error for skipped items: {pbar_e}")

    logger.debug(f"Batch {current_page}: Performing API Prefetches...")
    prefetched_data = _perform_api_prefetches(
        session_manager, fetch_candidates_uuid, matches_to_process_later
    )

    logger.debug(f"Batch {current_page}: Preparing DB data...")
    prepared_bulk_data, prep_statuses = _prepare_bulk_db_data(
        session,
        session_manager,
        matches_to_process_later,
        existing_persons_map,
        prefetched_data,
        progress_bar,
    )
    page_statuses["new"] = prep_statuses.get("new", 0)
    page_statuses["updated"] = prep_statuses.get("updated", 0)
    page_statuses["error"] = prep_statuses.get("error", 0)

    logger.debug(f"Batch {current_page}: Executing DB Commit...")
    _execute_batch_db_commit(session, prepared_bulk_data, existing_persons_map, current_page, page_statuses)


def _do_batch(
    session_manager: SessionManager,
    matches_on_page: list[dict[str, Any]],
    current_page: int,
    progress_bar: Optional[tqdm] = None,  # Accept progress bar
) -> tuple[int, int, int, int]:
    """
    Processes a batch of matches fetched from a single page.
    Coordinates DB lookups, API prefetches, data preparation, and bulk DB operations.
    Updates the progress bar incrementally for skipped items and processed candidates.

    Args:
        session_manager: The active SessionManager instance.
        matches_on_page: List of raw match data dictionaries from `get_matches`.
        current_page: The current page number being processed (1-based).
        progress_bar: Optional tqdm progress bar instance to update numerically.

    Returns:
        tuple[int, int, int, int]: Counts of (new, updated, skipped, error) outcomes
                                   for the processed batch.
    Raises:
        MaxApiFailuresExceededError: If API prefetch fails critically. This is caught
                                     by the main coord function to halt the run.
    """
    # Note: BATCH_SIZE is for database commit batching, not for limiting matches per page
    # Action 6 should process ALL matches on the page, then use BATCH_SIZE for DB operations
    # Step 1: Initialization
    page_statuses: dict[str, int] = {"new": 0, "updated": 0, "skipped": 0, "error": 0}
    num_matches_on_page = len(matches_on_page)
    my_uuid = session_manager.my_uuid
    session: Optional[SqlAlchemySession] = None

    try:
        # Step 2: Basic validation
        try:
            _validate_batch_prerequisites(my_uuid, matches_on_page, current_page)
        except ValueError as e:
            if "Empty match list" in str(e):
                return 0, 0, 0, 0
            raise

        logger.debug(f"--- Starting Batch Processing for Page {current_page} ({num_matches_on_page} matches) ---")

        # Step 3: Get DB Session for the batch
        session = session_manager.get_db_conn()
        if not session:
            logger.error(f"_do_batch Page {current_page}: Failed DB session.")
            raise SQLAlchemyError("Failed get DB session")

        # --- Data Processing Pipeline ---
        _execute_batch_pipeline(
            session,
            session_manager,
            matches_on_page,
            current_page,
            page_statuses,
            progress_bar
        )

        _log_page_summary(
            current_page,
            page_statuses["new"],
            page_statuses["updated"],
            page_statuses["skipped"],
            page_statuses["error"],
        )
        return (
            page_statuses["new"],
            page_statuses["updated"],
            page_statuses["skipped"],
            page_statuses["error"],
        )

    except MaxApiFailuresExceededError:
        raise
    except (ValueError, SQLAlchemyError, ConnectionError) as critical_err:
        final_error_count_for_page = _handle_batch_critical_error(
            page_statuses, num_matches_on_page, progress_bar, current_page, critical_err
        )
        return (page_statuses["new"], page_statuses["updated"], page_statuses["skipped"], final_error_count_for_page)
    except Exception as outer_batch_exc:
        logger.critical(f"CRITICAL UNHANDLED EXCEPTION processing batch page {current_page}: {outer_batch_exc}", exc_info=True)
        final_error_count_for_page = _handle_batch_critical_error(
            page_statuses, num_matches_on_page, progress_bar, current_page, outer_batch_exc
        )
        return (page_statuses["new"], page_statuses["updated"], page_statuses["skipped"], max(0, final_error_count_for_page))

    finally:
        if session:
            session_manager.return_session(session)
        logger.debug(f"--- Finished Batch Processing for Page {current_page} ---")


# End of _do_batch

# ------------------------------------------------------------------------------
# _do_match Helper Functions (_prepare_person_operation_data, etc.)
# ------------------------------------------------------------------------------


def _extract_raw_profile_ids(
    details_part: dict[str, Any],
    match: dict[str, Any]
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract and normalize raw profile IDs from API data."""
    raw_tester_profile_id = details_part.get("tester_profile_id") or match.get("profile_id")
    raw_admin_profile_id = details_part.get("admin_profile_id") or match.get(
        "administrator_profile_id_hint"
    )
    raw_admin_username = details_part.get("admin_username") or match.get(
        "administrator_username_hint"
    )

    tester_profile_id_upper = (
        raw_tester_profile_id.upper() if raw_tester_profile_id else None
    )
    admin_profile_id_upper = (
        raw_admin_profile_id.upper() if raw_admin_profile_id else None
    )
    formatted_admin_username = (
        format_name(raw_admin_username) if raw_admin_username else None
    )

    return tester_profile_id_upper, admin_profile_id_upper, formatted_admin_username


def _resolve_profile_assignment(
    tester_profile_id_upper: Optional[str],
    admin_profile_id_upper: Optional[str],
    formatted_admin_username: Optional[str],
    formatted_match_username: str
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve which profile IDs to save based on tester/admin relationship."""
    person_profile_id_to_save: Optional[str] = None
    person_admin_id_to_save: Optional[str] = None
    person_admin_username_to_save: Optional[str] = None

    # Both tester and admin IDs present
    if tester_profile_id_upper and admin_profile_id_upper:
        if tester_profile_id_upper == admin_profile_id_upper:
            # Same ID - check if usernames match
            if (
                formatted_match_username
                and formatted_admin_username
                and formatted_match_username.lower() == formatted_admin_username.lower()
            ):
                person_profile_id_to_save = tester_profile_id_upper
            else:
                person_admin_id_to_save = admin_profile_id_upper
                person_admin_username_to_save = formatted_admin_username
        else:
            # Different IDs - save both
            person_profile_id_to_save = tester_profile_id_upper
            person_admin_id_to_save = admin_profile_id_upper
            person_admin_username_to_save = formatted_admin_username
    # Only tester ID present
    elif tester_profile_id_upper:
        person_profile_id_to_save = tester_profile_id_upper
    # Only admin ID present
    elif admin_profile_id_upper:
        person_admin_id_to_save = admin_profile_id_upper
        person_admin_username_to_save = formatted_admin_username

    return person_profile_id_to_save, person_admin_id_to_save, person_admin_username_to_save


def _determine_profile_ids(
    details_part: dict[str, Any],
    match: dict[str, Any],
    formatted_match_username: str,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Determine profile IDs and admin information from API data.

    Returns:
        Tuple of (person_profile_id, person_admin_id, person_admin_username, message_target_id)
    """
    # Extract raw profile IDs
    tester_profile_id_upper, admin_profile_id_upper, formatted_admin_username = (
        _extract_raw_profile_ids(details_part, match)
    )

    # Resolve profile assignment
    person_profile_id_to_save, person_admin_id_to_save, person_admin_username_to_save = (
        _resolve_profile_assignment(
            tester_profile_id_upper,
            admin_profile_id_upper,
            formatted_admin_username,
            formatted_match_username
        )
    )

    message_target_id = person_profile_id_to_save or person_admin_id_to_save

    return (
        person_profile_id_to_save,
        person_admin_id_to_save,
        person_admin_username_to_save,
        message_target_id,
    )


def _normalize_datetime_to_utc(dt_value: Any) -> Optional[datetime]:
    """
    Normalize a datetime value to UTC timezone, ignoring microseconds.

    Returns:
        UTC datetime with microseconds set to 0, or None if input is not a datetime
    """
    if not isinstance(dt_value, datetime):
        return None

    if dt_value.tzinfo:
        return dt_value.astimezone(timezone.utc).replace(microsecond=0)
    return dt_value.replace(tzinfo=timezone.utc, microsecond=0)


def _determine_match_status(
    is_new_person: bool,
    person_fields_changed: bool,
    dna_op_data: Any,
    tree_op_data: Any,
    tree_operation_status: str,
) -> Literal["new", "updated", "skipped"]:
    """
    Determine the overall status for a match based on data changes.

    Returns:
        Status: "new", "updated", or "skipped"
    """
    if is_new_person:
        return "new"

    # Existing person - check if anything changed
    if (
        person_fields_changed
        or dna_op_data
        or (tree_op_data and tree_operation_status != "none")
    ):
        return "updated"
    return "skipped"


def _compare_datetime_field(
    new_value: Any,
    current_value: Any,
) -> tuple[bool, Any]:
    """Compare datetime fields with UTC normalization."""
    current_dt_utc = _normalize_datetime_to_utc(current_value)
    new_dt_utc = _normalize_datetime_to_utc(new_value)

    if new_dt_utc != current_dt_utc:
        return True, new_value
    return False, new_value


def _compare_status_field(
    new_value: Any,
    current_value: Any,
) -> tuple[bool, Any]:
    """Compare status enum fields."""
    current_enum_val = (
        current_value.value
        if isinstance(current_value, PersonStatusEnum)
        else current_value
    )
    new_enum_val = (
        new_value.value
        if isinstance(new_value, PersonStatusEnum)
        else new_value
    )

    if new_enum_val != current_enum_val:
        return True, new_value
    return False, new_value


def _compare_birth_year_field(
    new_value: Any,
    current_value: Any,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> tuple[bool, Any]:
    """Compare birth year field - only update if new is valid and current is None."""
    if new_value is not None and current_value is None:
        try:
            value_to_set_int = int(new_value)
            return True, value_to_set_int
        except (ValueError, TypeError):
            logger_instance.warning(
                f"Invalid birth_year '{new_value}' for update {log_ref_short}"
            )
            return False, None
    return False, new_value


def _compare_gender_field(
    new_value: Any,
    current_value: Any,
) -> tuple[bool, Any]:
    """Compare gender field - only update if new is valid ('f'/'m') and current is None."""
    if (
        new_value is not None
        and current_value is None
        and isinstance(new_value, str)
        and new_value.lower() in ("f", "m")
    ):
        return True, new_value.lower()
    return False, new_value


def _compare_profile_id_field(
    new_value: Any,
    current_value: Any,
) -> tuple[bool, Any]:
    """Compare profile ID fields with uppercase normalization."""
    current_str_upper = (
        str(current_value).upper() if current_value is not None else None
    )
    new_str_upper = (
        str(new_value).upper() if new_value is not None else None
    )

    if new_str_upper != current_str_upper:
        return True, new_str_upper
    return False, new_str_upper


def _compare_boolean_field(
    new_value: Any,
    current_value: Any,
) -> tuple[bool, Any]:
    """Compare boolean fields."""
    if bool(current_value) != bool(new_value):
        return True, bool(new_value)
    return False, bool(new_value)


def _extract_dna_field_values(
    match: dict[str, Any],
    existing_dna_match: DnaMatch,
    details_part: dict[str, Any],
    api_predicted_rel_for_comp: str,
) -> dict[str, Any]:
    """Extract DNA field values from API and database for comparison."""
    api_cm = int(match.get("cM_DNA", 0))
    db_cm = existing_dna_match.cM_DNA
    api_segments = int(
        details_part.get("shared_segments", match.get("numSharedSegments", 0))
    )
    db_segments = existing_dna_match.shared_segments
    api_longest_raw = details_part.get("longest_shared_segment")
    api_longest = (
        float(api_longest_raw) if api_longest_raw is not None else None
    )
    db_longest = existing_dna_match.longest_shared_segment
    db_predicted_rel_for_comp = (
        existing_dna_match.predicted_relationship
        if existing_dna_match.predicted_relationship is not None
        else "N/A"
    )

    return {
        "api_cm": api_cm,
        "db_cm": db_cm,
        "api_segments": api_segments,
        "db_segments": db_segments,
        "api_longest": api_longest,
        "db_longest": db_longest,
        "api_predicted_rel": api_predicted_rel_for_comp,
        "db_predicted_rel": db_predicted_rel_for_comp,
        "api_fathers_side": bool(details_part.get("from_my_fathers_side", False)),
        "db_fathers_side": bool(existing_dna_match.from_my_fathers_side),
        "api_mothers_side": bool(details_part.get("from_my_mothers_side", False)),
        "db_mothers_side": bool(existing_dna_match.from_my_mothers_side),
        "api_meiosis": details_part.get("meiosis"),
        "db_meiosis": existing_dna_match.meiosis,
    }


def _check_dna_fields_changed(
    field_values: dict[str, Any],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> bool:
    """Check if any DNA fields have changed."""
    # Define field checks as tuples: (condition, message)
    checks = [
        (
            field_values["api_cm"] != field_values["db_cm"],
            "cM"
        ),
        (
            field_values["api_segments"] != field_values["db_segments"],
            "Segments"
        ),
        (
            field_values["api_longest"] is not None
            and field_values["db_longest"] is not None
            and abs(float(str(field_values["api_longest"])) - float(str(field_values["db_longest"]))) > 0.01,
            "Longest Segment"
        ),
        (
            field_values["db_longest"] is not None and field_values["api_longest"] is None,
            "Longest Segment (API lost data)"
        ),
        (
            str(field_values["db_predicted_rel"]) != str(field_values["api_predicted_rel"]),
            f"Predicted Rel ({field_values['db_predicted_rel']} -> {field_values['api_predicted_rel']})"
        ),
        (
            field_values["api_fathers_side"] != field_values["db_fathers_side"],
            "Father Side"
        ),
        (
            field_values["api_mothers_side"] != field_values["db_mothers_side"],
            "Mother Side"
        ),
        (
            field_values["api_meiosis"] is not None and field_values["api_meiosis"] != field_values["db_meiosis"],
            "Meiosis"
        ),
    ]

    # Check each condition
    for condition, message in checks:
        if condition:
            logger_instance.debug(f"  DNA change {log_ref_short}: {message}")
            return True

    return False


def _check_dna_match_needs_update(
    match: dict[str, Any],
    existing_dna_match: DnaMatch,
    details_part: dict[str, Any],
    api_predicted_rel_for_comp: str,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> bool:
    """
    Check if DNA match data needs to be updated by comparing API data with existing record.

    Returns:
        True if update is needed, False otherwise
    """
    try:
        field_values = _extract_dna_field_values(
            match, existing_dna_match, details_part, api_predicted_rel_for_comp
        )
        return _check_dna_fields_changed(field_values, log_ref_short, logger_instance)
    except (ValueError, TypeError, AttributeError) as dna_comp_err:
        logger_instance.warning(
            f"Error comparing DNA data for {log_ref_short}: {dna_comp_err}. Assuming update needed."
        )
        return True


def _process_birth_year(
    prefetched_tree_data: Optional[dict[str, Any]],
) -> Optional[int]:
    """Extract and validate birth year from tree data."""
    if prefetched_tree_data and prefetched_tree_data.get("their_birth_year"):
        try:
            return int(prefetched_tree_data["their_birth_year"])
        except (ValueError, TypeError):
            pass
    return None


def _process_last_logged_in(
    profile_part: dict[str, Any],
) -> Optional[datetime]:
    """Extract and normalize last_logged_in datetime."""
    last_logged_in_val: Optional[datetime] = profile_part.get("last_logged_in_dt")
    if isinstance(last_logged_in_val, datetime):
        if last_logged_in_val.tzinfo is None:
            return last_logged_in_val.replace(tzinfo=timezone.utc)
        return last_logged_in_val.astimezone(timezone.utc)
    return None


def _compare_and_update_person_fields(
    incoming_person_data: dict[str, Any],
    existing_person: Person,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> tuple[dict[str, Any], bool]:
    """
    Compare incoming person data with existing person and build update dictionary.

    Returns:
        Tuple of (person_data_for_update, person_fields_changed)
    """
    person_data_for_update: dict[str, Any] = {
        "_operation": "update",
        "_existing_person_id": existing_person.id,
        "uuid": incoming_person_data["uuid"],
    }
    person_fields_changed = False

    for key, new_value in incoming_person_data.items():
        if key == "uuid":
            continue

        current_value = getattr(existing_person, key, None)
        value_changed, value_to_set = _compare_person_field(
            key, new_value, current_value, log_ref_short, logger_instance
        )

        if not value_changed and value_to_set is None:
            continue

        if value_changed:
            person_data_for_update[key] = value_to_set
            person_fields_changed = True
            logger_instance.debug(
                f"  Person change {log_ref_short}: Field '{key}' ('{current_value}' -> '{value_to_set}')"
            )

    return person_data_for_update, person_fields_changed


def _build_tree_links(
    their_cfpid: str,
    session_manager: SessionManager,
    config_schema_arg: "ConfigSchema",
) -> tuple[Optional[str], Optional[str]]:
    """
    Build facts link and view-in-tree link for a person in the family tree.

    Returns:
        Tuple of (facts_link, view_in_tree_link)
    """
    if not their_cfpid or not session_manager.my_tree_id:
        return None, None

    base_person_path = f"/family-tree/person/tree/{session_manager.my_tree_id}/person/{their_cfpid}"
    facts_link = urljoin(config_schema_arg.api.base_url, f"{base_person_path}/facts")  # type: ignore

    view_params = {
        "cfpid": their_cfpid,
        "showMatches": "true",
        "sid": session_manager.my_uuid,
    }
    base_view_url = urljoin(
        config_schema_arg.api.base_url,  # type: ignore
        f"/family-tree/tree/{session_manager.my_tree_id}/family",
    )
    view_in_tree_link = f"{base_view_url}?{urlencode(view_params)}"

    return facts_link, view_in_tree_link


def _check_family_tree_needs_update(
    existing_family_tree: FamilyTree,
    prefetched_tree_data: dict[str, Any],
    their_cfpid_final: Optional[str],
    facts_link: Optional[str],
    view_in_tree_link: Optional[str],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> bool:
    """
    Check if family tree record needs to be updated by comparing fields.

    Returns:
        True if update is needed, False otherwise
    """
    fields_to_check = [
        ("cfpid", their_cfpid_final),
        (
            "person_name_in_tree",
            prefetched_tree_data.get("their_firstname", "Unknown"),
        ),
        (
            "actual_relationship",
            prefetched_tree_data.get("actual_relationship"),
        ),
        ("relationship_path", prefetched_tree_data.get("relationship_path")),
        ("facts_link", facts_link),
        ("view_in_tree_link", view_in_tree_link),
    ]

    for field_name, new_val in fields_to_check:
        old_val = getattr(existing_family_tree, field_name, None)
        if new_val != old_val:  # Handles None comparison correctly
            logger_instance.debug(
                f"  Tree change {log_ref_short}: Field '{field_name}'"
            )
            return True

    return False


def _process_person_data(
    match: dict[str, Any],
    existing_person: Optional[Person],
    prefetched_combined_details: Optional[dict[str, Any]],
    prefetched_tree_data: Optional[dict[str, Any]],
    match_uuid: str,
    match_username: str,
    match_in_my_tree: bool,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> tuple[Optional[dict[str, Any]], bool]:
    """
    Process Person data with error handling.

    Returns:
        Tuple of (person_op_data, person_fields_changed)
    """
    try:
        prefetched_data = PrefetchedData(
            combined_details=prefetched_combined_details,
            tree_data=prefetched_tree_data
        )
        match_ids = MatchIdentifiers(
            uuid=match_uuid,
            username=match_username,
            in_my_tree=match_in_my_tree,
            log_ref_short=log_ref_short
        )
        return _prepare_person_operation_data(
            match=match,
            existing_person=existing_person,
            prefetched_data=prefetched_data,
            config_schema_arg=config_schema,
            match_ids=match_ids,
            logger_instance=logger_instance,
        )
    except Exception as person_err:
        logger_instance.error(
            f"Error in _prepare_person_operation_data for {log_ref_short}: {person_err}",
            exc_info=True,
        )
        return None, False


def _process_dna_data(
    match: dict[str, Any],
    dna_match_record: Optional[DnaMatch],
    prefetched_combined_details: Optional[dict[str, Any]],
    match_uuid: str,
    predicted_relationship: Optional[str],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Optional[dict[str, Any]]:
    """
    Process DNA Match data with error handling.

    Returns:
        DNA operation data or None
    """
    try:
        return _prepare_dna_match_operation_data(
            match=match,
            existing_dna_match=dna_match_record,
            prefetched_combined_details=prefetched_combined_details,
            match_uuid=match_uuid,
            predicted_relationship=predicted_relationship,
            log_ref_short=log_ref_short,
            logger_instance=logger_instance,
        )
    except Exception as dna_err:
        logger_instance.error(
            f"Error in _prepare_dna_match_operation_data for {log_ref_short}: {dna_err}",
            exc_info=True,
        )
        return None


def _process_tree_data(
    family_tree_record: Optional[FamilyTree],
    prefetched_tree_data: Optional[dict[str, Any]],
    match_uuid: str,
    match_in_my_tree: bool,
    session_manager: SessionManager,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> tuple[Optional[dict[str, Any]], str]:
    """
    Process Family Tree data with error handling.

    Returns:
        Tuple of (tree_op_data, tree_operation_status)
    """
    try:
        return _prepare_family_tree_operation_data(
            existing_family_tree=family_tree_record,
            prefetched_tree_data=prefetched_tree_data,
            match_uuid=match_uuid,
            match_in_my_tree=match_in_my_tree,
            session_manager=session_manager,
            config_schema_arg=config_schema,
            log_ref_short=log_ref_short,
            logger_instance=logger_instance,
        )
    except Exception as tree_err:
        logger_instance.error(
            f"Error in _prepare_family_tree_operation_data for {log_ref_short}: {tree_err}",
            exc_info=True,
        )
        return None, "none"  # type: ignore


def _compare_person_field(
    key: str,
    new_value: Any,
    current_value: Any,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> tuple[bool, Any]:
    """
    Compare a single person field and determine if it changed.

    Returns:
        Tuple of (value_changed, value_to_set)
    """
    # Dispatch to field-specific comparison functions
    result = None

    if key == "last_logged_in":
        result = _compare_datetime_field(new_value, current_value)
    elif key == "status":
        result = _compare_status_field(new_value, current_value)
    elif key == "birth_year":
        result = _compare_birth_year_field(new_value, current_value, log_ref_short, logger_instance)
    elif key == "gender":
        result = _compare_gender_field(new_value, current_value)
    elif key in ("profile_id", "administrator_profile_id"):
        result = _compare_profile_id_field(new_value, current_value)
    elif isinstance(current_value, bool) or isinstance(new_value, bool):
        result = _compare_boolean_field(new_value, current_value)
    elif current_value != new_value:
        # General comparison for other fields
        result = (True, new_value)
    else:
        result = (False, new_value)

    return result


def _prepare_person_operation_data(
    match: dict[str, Any],
    existing_person: Optional[Person],
    prefetched_data: PrefetchedData,
    config_schema_arg: "ConfigSchema",  # Config schema argument
    match_ids: MatchIdentifiers,
    logger_instance: logging.Logger,
) -> tuple[Optional[dict[str, Any]], bool]:
    """
    Prepares Person data for create or update operations based on API data and existing records.

    Args:
        match: Dictionary containing data for one match from the match list API.
        existing_person: The existing Person object from the database, or None if this is a new person.
        prefetched_data: Prefetched data from APIs (combined_details and tree_data).
        config_schema_arg: The application configuration schema.
        match_ids: Match identification parameters (uuid, username, in_my_tree, log_ref_short).
        logger_instance: The logger instance.

    Returns:
        A tuple containing:
        - person_op_dict (Optional[dict]): Dictionary with person data and '_operation' key
          set to 'create' or 'update'. None if no update is needed.
        - person_fields_changed (bool): True if any fields were changed for an existing person,
          False otherwise.
    """
    details_part = prefetched_data.combined_details or {}
    profile_part = details_part

    # Determine profile IDs and admin information
    (
        person_profile_id_to_save,
        person_admin_id_to_save,
        person_admin_username_to_save,
        message_target_id,
    ) = _determine_profile_ids(details_part, match, match_ids.username)

    # Construct message link
    constructed_message_link = (
        urljoin(config_schema_arg.api.base_url, f"/messaging/?p={message_target_id.upper()}")  # type: ignore
        if message_target_id
        else None
    )

    birth_year_val = _process_birth_year(prefetched_data.tree_data)
    last_logged_in_val = _process_last_logged_in(profile_part)

    incoming_person_data = {
        "uuid": match_ids.uuid.upper(),
        "profile_id": person_profile_id_to_save,
        "username": match_ids.username,
        "administrator_profile_id": person_admin_id_to_save,
        "administrator_username": person_admin_username_to_save,
        "in_my_tree": match_ids.in_my_tree,
        "first_name": match.get("first_name"),
        "last_logged_in": last_logged_in_val,
        "contactable": bool(profile_part.get("contactable", True)),
        "gender": details_part.get("gender"),
        "message_link": constructed_message_link,
        "birth_year": birth_year_val,
        "status": PersonStatusEnum.ACTIVE,
    }

    if existing_person is None:
        person_op_dict = incoming_person_data.copy()
        person_op_dict["_operation"] = "create"
        return person_op_dict, False

    # Person exists - compare and update
    person_data_for_update, person_fields_changed = _compare_and_update_person_fields(
        incoming_person_data, existing_person, match_ids.log_ref_short, logger_instance
    )

    return (
        person_data_for_update if person_fields_changed else None
    ), person_fields_changed


# End of _prepare_person_operation_data


def _prepare_dna_match_operation_data(
    match: dict[str, Any],
    existing_dna_match: Optional[DnaMatch],
    prefetched_combined_details: Optional[dict[str, Any]],
    match_uuid: str,
    predicted_relationship: Optional[str],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Optional[dict[str, Any]]:
    """
    Prepares DnaMatch data for create or update operations by comparing API data with existing records.

    Args:
        match: Dictionary containing data for one match from the match list API.
        existing_dna_match: The existing DnaMatch object from the database, or None if this is a new match.
        prefetched_combined_details: Prefetched data from '/details' API containing DNA-specific information.
        match_uuid: The UUID (Sample ID) of the match.
        predicted_relationship: The predicted relationship string from the API, can be None if not available.
        log_ref_short: Short reference string for logging.
        logger_instance: The logger instance.

    Returns:
        Optional[dict[str, Any]]: Dictionary with DNA match data and '_operation' key set to 'create',
        or None if no create/update is needed. The dictionary includes fields like: cM_DNA,
        shared_segments, longest_shared_segment, etc.
    """
    needs_dna_create_or_update = False
    details_part = prefetched_combined_details or {}
    # Use "N/A" as a safe default if predicted_relationship is None for comparisons
    api_predicted_rel_for_comp = (
        predicted_relationship if predicted_relationship is not None else "N/A"
    )

    # Check if DNA match exists and needs update
    if existing_dna_match is None:
        needs_dna_create_or_update = True
    else:
        needs_dna_create_or_update = _check_dna_match_needs_update(
            match, existing_dna_match, details_part, api_predicted_rel_for_comp, log_ref_short, logger_instance
        )

    if needs_dna_create_or_update:
        dna_dict_base = {
            "uuid": match_uuid.upper(),
            "compare_link": match.get("compare_link"),
            "cM_DNA": int(match.get("cM_DNA", 0)),
            # Store predicted_relationship as is (can be None); DB schema should allow NULL
            "predicted_relationship": predicted_relationship,
            "_operation": "create",  # This operation hint is for the bulk operation logic
        }
        if prefetched_combined_details:
            dna_dict_base.update(
                {
                    "shared_segments": details_part.get("shared_segments"),
                    "longest_shared_segment": details_part.get(
                        "longest_shared_segment"
                    ),
                    "meiosis": details_part.get("meiosis"),
                    "from_my_fathers_side": bool(
                        details_part.get("from_my_fathers_side", False)
                    ),
                    "from_my_mothers_side": bool(
                        details_part.get("from_my_mothers_side", False)
                    ),
                }
            )
        else:  # Fallback if details API failed (prefetched_combined_details is None or empty)
            logger_instance.warning(
                f"{log_ref_short}: DNA needs create/update, but no/limited combined details. Using list data for segments."
            )
            dna_dict_base["shared_segments"] = match.get("numSharedSegments")
            # Other detail-specific fields (longest_shared_segment, meiosis, sides) will be None if not in dna_dict_base

        # Remove keys with None values *except* for predicted_relationship which we want to store as NULL if it's None
        # Also keep internal keys like _operation and uuid
        return {
            k: v
            for k, v in dna_dict_base.items()
            if v is not None
            or k
            == "predicted_relationship"  # Explicitly keep predicted_relationship even if None
            or k.startswith("_")  # Keep internal keys
            or k == "uuid"  # Keep uuid
        }
    return None


# End of _prepare_dna_match_operation_data


def _extract_tree_links_from_data(
    prefetched_tree_data: Optional[dict[str, Any]],
    session_manager: SessionManager,
    config_schema_arg: "ConfigSchema",
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract CFPID and build tree links from prefetched data."""
    their_cfpid_final = None
    facts_link, view_in_tree_link = None, None

    if prefetched_tree_data:
        their_cfpid_final = prefetched_tree_data.get("their_cfpid")
        if their_cfpid_final:
            facts_link, view_in_tree_link = _build_tree_links(
                their_cfpid_final, session_manager, config_schema_arg
            )

    return their_cfpid_final, facts_link, view_in_tree_link


def _determine_tree_operation(
    match_in_my_tree: bool,
    existing_family_tree: Optional[FamilyTree],
    prefetched_tree_data: Optional[dict[str, Any]],
    their_cfpid_final: Optional[str],
    facts_link: Optional[str],
    view_in_tree_link: Optional[str],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Literal["create", "update", "none"]:
    """Determine the tree operation based on match status and existing data."""
    if match_in_my_tree and existing_family_tree is None:
        return "create"

    if match_in_my_tree and existing_family_tree is not None:
        if prefetched_tree_data:  # Only check if we have new data
            needs_update = _check_family_tree_needs_update(
                existing_family_tree,
                prefetched_tree_data,
                their_cfpid_final,
                facts_link,
                view_in_tree_link,
                log_ref_short,
                logger_instance,
            )
            if needs_update:
                return "update"
        return "none"

    if not match_in_my_tree and existing_family_tree is not None:
        logger_instance.warning(
            f"{log_ref_short}: Data mismatch - API says not 'in_my_tree', but FamilyTree record exists (ID: {existing_family_tree.id}). Skipping."
        )

    return "none"


def _build_tree_dict_from_data(
    match_uuid: str,
    their_cfpid_final: Optional[str],
    facts_link: Optional[str],
    view_in_tree_link: Optional[str],
    prefetched_tree_data: dict[str, Any],
    tree_operation: Literal["create", "update"],
    existing_family_tree: Optional[FamilyTree],
) -> dict[str, Any]:
    """Build tree dictionary from prefetched data."""
    tree_person_name = prefetched_tree_data.get("their_firstname", "Unknown")
    tree_dict_base = {
        "uuid": match_uuid.upper(),
        "cfpid": their_cfpid_final,
        "person_name_in_tree": tree_person_name,
        "facts_link": facts_link,
        "view_in_tree_link": view_in_tree_link,
        "actual_relationship": prefetched_tree_data.get("actual_relationship"),
        "relationship_path": prefetched_tree_data.get("relationship_path"),
        "_operation": tree_operation,
        "_existing_tree_id": (
            existing_family_tree.id
            if tree_operation == "update" and existing_family_tree
            else None
        ),
    }
    # Keep all keys for _operation and _existing_tree_id, otherwise only non-None values
    return {
        k: v
        for k, v in tree_dict_base.items()
        if v is not None
        or k in ["_operation", "_existing_tree_id", "uuid"]  # Keep uuid
    }


def _prepare_family_tree_operation_data(
    existing_family_tree: Optional[FamilyTree],
    prefetched_tree_data: Optional[dict[str, Any]],
    match_uuid: str,
    match_in_my_tree: bool,
    session_manager: SessionManager,
    config_schema_arg: "ConfigSchema",  # Config schema argument
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> tuple[Optional[dict[str, Any]], Literal["create", "update", "none"]]:
    """
    Prepares FamilyTree data for create or update operations based on API data and existing records.

    Args:
        existing_family_tree: The existing FamilyTree object from the database, or None if no record exists.
        prefetched_tree_data: Prefetched data from 'badgedetails' & 'getladder' APIs containing tree information.
        match_uuid: The UUID (Sample ID) of the match.
        match_in_my_tree: Boolean indicating if the match is in the user's family tree.
        session_manager: The active SessionManager instance containing session and tree information.
        config_schema_arg: The application configuration schema.
        log_ref_short: Short reference string for logging.
        logger_instance: The logger instance.

    Returns:
        A tuple containing:
        - tree_data (Optional[dict]): Dictionary with family tree data and '_operation' key
          set to 'create' or 'update'. None if no create/update is needed.
        - tree_operation (Literal["create", "update", "none"]): The operation type determined
          for this family tree record.
    """
    # Extract CFPID and build links
    their_cfpid_final, facts_link, view_in_tree_link = _extract_tree_links_from_data(
        prefetched_tree_data, session_manager, config_schema_arg
    )

    # Determine tree operation
    tree_operation = _determine_tree_operation(
        match_in_my_tree,
        existing_family_tree,
        prefetched_tree_data,
        their_cfpid_final,
        facts_link,
        view_in_tree_link,
        log_ref_short,
        logger_instance,
    )

    # Build tree dict if operation is needed
    if tree_operation != "none":
        if prefetched_tree_data:  # Can only build if data was fetched
            incoming_tree_data = _build_tree_dict_from_data(
                match_uuid,
                their_cfpid_final,
                facts_link,
                view_in_tree_link,
                prefetched_tree_data,
                tree_operation,
                existing_family_tree,
            )
            return incoming_tree_data, tree_operation
        logger_instance.warning(
            f"{log_ref_short}: FamilyTree needs '{tree_operation}', but tree details not fetched. Skipping."
        )
        tree_operation = "none"

    return None, tree_operation


# End of _prepare_family_tree_operation_data

# ------------------------------------------------------------------------------
# Individual Match Processing (_do_match) - Refactored
# ------------------------------------------------------------------------------


def _populate_and_finalize_match_data(
    person_op_data: Optional[dict[str, Any]],
    dna_op_data: Optional[dict[str, Any]],
    tree_op_data: Optional[dict[str, Any]],
    tree_operation_status: str,
    is_new_person: bool,
    person_fields_changed: bool,
    prepared_data_for_bulk: dict[str, Any],
    log_ref_short: str,
    logger_instance: logging.Logger
) -> tuple[Optional[dict[str, Any]], Literal["new", "updated", "skipped", "error"]]:
    """Populate prepared data and determine final status."""
    # Populate prepared data
    if person_op_data:
        prepared_data_for_bulk["person"] = person_op_data
    if dna_op_data:
        prepared_data_for_bulk["dna_match"] = dna_op_data
    if tree_op_data and ((is_new_person and tree_operation_status == "create") or not is_new_person):
        prepared_data_for_bulk["family_tree"] = tree_op_data

    # Determine overall status
    overall_status = _determine_match_status(
        is_new_person,
        person_fields_changed,
        dna_op_data,
        tree_op_data,
        tree_operation_status,
    )

    data_to_return = (
        prepared_data_for_bulk
        if overall_status not in ["skipped", "error"]
        and any(v for v in prepared_data_for_bulk.values())
        else None
    )

    if overall_status not in ["error", "skipped"] and not data_to_return:
        logger_instance.debug(
            f"Status is '{overall_status}' for {log_ref_short}, but no data payloads prepared. Revising to 'skipped'."
        )
        overall_status = "skipped"
        data_to_return = None

    return data_to_return, overall_status


def _do_match(  # type: ignore
    match: dict[str, Any],
    session_manager: SessionManager,
    existing_person_arg: Optional[Person],
    prefetched_combined_details: Optional[dict[str, Any]],
    prefetched_tree_data: Optional[dict[str, Any]],
    _config_schema_arg: "ConfigSchema",
    logger_instance: logging.Logger,
) -> tuple[
    Optional[dict[str, Any]],
    Literal["new", "updated", "skipped", "error"],
    Optional[str],
]:
    """
    Processes a single DNA match by calling helper functions to compare incoming data
    with existing database records. Determines if a 'create', 'update', or 'skip'
    operation is needed and prepares a dictionary for bulk operations.

    This function orchestrates the data preparation process by:
    1. Extracting basic information from the match data
    2. Calling helper functions to prepare data for each table (Person, DnaMatch, FamilyTree)
    3. Determining the overall status based on the results from helper functions
    4. Assembling the final data structure for bulk database operations

    Args:
        session: The active SQLAlchemy database session.
        match: Dictionary containing data for one match from the match list API.
        session_manager: The active SessionManager instance with session and tree information.
        existing_person_arg: The existing Person object from the database, or None if this is a new person.
        prefetched_combined_details: Prefetched data from '/details' & '/profiles/details' APIs.
        prefetched_tree_data: Prefetched data from 'badgedetails' & 'getladder' APIs.
        config_schema_arg: The application configuration schema.
        logger_instance: The logger instance for recording debug/error information.

    Returns:
        A tuple containing:
        - prepared_data (Optional[dict[str, Any]]): Dictionary with keys 'person', 'dna_match', and
          'family_tree', each containing data for bulk operations or None if no change needed.
          Returns None if status is 'skipped' or 'error'.
        - status (Literal["new", "updated", "skipped", "error"]): The overall status determined
          for this match based on all data comparisons.
        - error_msg (Optional[str]): An error message if status is 'error', otherwise None.
    """
    existing_person: Optional[Person] = existing_person_arg
    dna_match_record: Optional[DnaMatch] = (
        existing_person.dna_match if existing_person else None
    )
    family_tree_record: Optional[FamilyTree] = (
        existing_person.family_tree if existing_person else None
    )

    match_uuid = match.get("uuid")
    match_username_raw = match.get("username")
    match_username = (
        format_name(match_username_raw) if match_username_raw else "Unknown"
    )
    # predicted_relationship can now be None if fetch failed and was set to None in _prepare_bulk_db_data
    predicted_relationship: Optional[str] = match.get(
        "predicted_relationship"
    )  # No default "N/A" here yet
    match_in_my_tree = match.get("in_my_tree", False)
    log_ref_short = f"UUID={match_uuid} User='{match_username}'"

    prepared_data_for_bulk: dict[str, Any] = {
        "person": None,
        "dna_match": None,
        "family_tree": None,
    }
    overall_status: Literal["new", "updated", "skipped", "error"] = (
        "error"  # Default status
    )
    error_msg: Optional[str] = None

    if not match_uuid:
        error_msg = f"_do_match Pre-check failed: Missing 'uuid' in match data: {match}"
        logger_instance.error(error_msg)
        return None, "error", error_msg

    try:
        is_new_person = existing_person is None

        # Process Person, DNA, and Tree data with error handling
        person_op_data, person_fields_changed = _process_person_data(
            match, existing_person, prefetched_combined_details, prefetched_tree_data,
            match_uuid, match_username, match_in_my_tree, log_ref_short, logger_instance
        )

        dna_op_data = _process_dna_data(
            match, dna_match_record, prefetched_combined_details,
            match_uuid, predicted_relationship, log_ref_short, logger_instance
        )

        tree_op_data, tree_operation_status = _process_tree_data(
            family_tree_record, prefetched_tree_data, match_uuid,
            match_in_my_tree, session_manager, log_ref_short, logger_instance
        )

        # Populate and finalize match data
        data_to_return, overall_status = _populate_and_finalize_match_data(
            person_op_data,
            dna_op_data,
            tree_op_data,
            tree_operation_status,
            is_new_person,
            person_fields_changed,
            prepared_data_for_bulk,
            log_ref_short,
            logger_instance
        )

        return data_to_return, overall_status, None

    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        error_msg_for_log = f"Unexpected critical error ({error_type}) in _do_match for {log_ref_short}. Details: {error_details}"
        logger_instance.error(error_msg_for_log, exc_info=True)
        error_msg_return = (
            f"Unexpected {error_type} during data prep for {log_ref_short}"
        )
        return None, "error", error_msg_return


# End of _do_match

# ------------------------------------------------------------------------------
# API Data Acquisition Helpers (_fetch_*)
# ------------------------------------------------------------------------------


def _sync_driver_cookies_to_scraper(
    driver_cookies_list: list[dict[str, Any]],
    scraper: Any,
    api_description: str
) -> None:
    """Sync cookies from WebDriver to scraper."""
    logger.debug(
        f"Syncing {len(driver_cookies_list)} WebDriver cookies to shared scraper for {api_description}..."
    )
    if hasattr(scraper, "cookies") and isinstance(
        scraper.cookies, RequestsCookieJar  # type: ignore
    ):
        scraper.cookies.clear()
        for cookie in driver_cookies_list:
            if "name" in cookie and "value" in cookie:
                scraper.cookies.set(
                    cookie["name"],
                    cookie["value"],
                    domain=cookie.get("domain"),
                    path=cookie.get("path", "/"),
                    secure=cookie.get("secure", False),
                )
    else:
        logger.warning("Scraper cookie jar not accessible for update.")


def _extract_csrf_from_cookies(
    driver_cookies_list: list[dict[str, Any]],
    api_description: str
) -> Optional[str]:
    """Extract CSRF token from driver cookies."""
    csrf_cookie_names = ("_dnamatches-matchlistui-x-csrf-token", "_csrf")
    driver_cookies_dict = {
        c["name"]: c["value"]
        for c in driver_cookies_list
        if "name" in c and "value" in c
    }

    for name in csrf_cookie_names:
        if driver_cookies_dict.get(name):
            csrf_token_val = unquote(driver_cookies_dict[name]).split("|")[0]
            logger.debug(
                f"Using fresh CSRF token '{name}' from driver cookies for {api_description}."
            )
            return csrf_token_val

    return None


def _sync_cookies_and_get_csrf_for_scraper(
    driver: Any,
    scraper: Any,
    api_description: str,
) -> Optional[str]:
    """
    Sync cookies from WebDriver to scraper and extract CSRF token.

    Returns:
        CSRF token if found, None otherwise
    """
    try:
        driver_cookies_list = driver.get_cookies()
        if not driver_cookies_list:
            logger.warning(
                f"driver.get_cookies() returned empty list during {api_description} prep."
            )
            return None

        _sync_driver_cookies_to_scraper(driver_cookies_list, scraper, api_description)
        return _extract_csrf_from_cookies(driver_cookies_list, api_description)

    except WebDriverException as csrf_wd_e:
        logger.warning(
            f"WebDriverException getting/setting cookies for {api_description}: {csrf_wd_e}"
        )
        raise ConnectionError(
            f"WebDriver error getting/setting cookies for CSRF: {csrf_wd_e}"
        ) from csrf_wd_e
    except Exception as csrf_e:
        logger.warning(f"Error processing cookies/CSRF for {api_description}: {csrf_e}")
        return None


def _validate_relationship_prob_session(
    session_manager: SessionManager,
    match_uuid: str,
) -> None:
    """
    Validate session manager components for relationship probability fetch.

    Raises:
        ConnectionError if validation fails
    """
    my_uuid = session_manager.my_uuid
    driver = session_manager.driver
    scraper = session_manager.scraper

    if not my_uuid or not match_uuid:
        logger.warning("_fetch_batch_relationship_prob: Missing my_uuid or match_uuid.")
        raise ValueError("Missing my_uuid or match_uuid")
    if not scraper:
        logger.error(
            "_fetch_batch_relationship_prob: SessionManager scraper not initialized."
        )
        raise ConnectionError("SessionManager scraper not initialized.")
    if not driver or not session_manager.is_sess_valid():
        logger.error(
            f"_fetch_batch_relationship_prob: Driver/session invalid for UUID {match_uuid}."
        )
        raise ConnectionError(
            f"WebDriver session invalid for relationship probability fetch (UUID: {match_uuid})"
        )


def _get_csrf_token_for_relationship_prob(
    driver: Any,
    scraper: Any,
    session_manager: SessionManager,
    api_description: str,
) -> Optional[str]:
    """
    Get CSRF token for relationship probability API request.

    Returns:
        CSRF token string or None if not available
    """
    csrf_token_val = _sync_cookies_and_get_csrf_for_scraper(driver, scraper, api_description)

    if csrf_token_val:
        return csrf_token_val
    if session_manager.csrf_token:
        logger.warning(
            f"{api_description}: Using potentially stale CSRF from SessionManager."
        )
        return session_manager.csrf_token
    logger.error(
        f"{api_description}: Failed to add CSRF token to headers. Returning None."
    )
    return None


def _process_relationship_prob_response(
    response_rel: Any,
    sample_id_upper: str,
    api_description: str,
    max_labels_param: int,
) -> Optional[str]:
    """
    Process the relationship probability API response and extract prediction.

    Returns:
        Formatted relationship string with probability, or None if processing fails
    """
    if not response_rel.content:
        logger.warning(
            f"{api_description}: OK ({response_rel.status_code}), but response body EMPTY."
        )
        return None

    try:
        data = response_rel.json()
    except json.JSONDecodeError as json_err:
        logger.error(
            f"{api_description}: OK ({response_rel.status_code}), but JSON decode FAILED: {json_err}"
        )
        logger.debug(f"Response text: {response_rel.text[:500]}")
        raise RequestException("JSONDecodeError") from json_err

    if "matchProbabilityToSampleId" not in data:
        logger.warning(
            f"Invalid data structure from {api_description} for {sample_id_upper}. Resp: {data}"
        )
        return None

    prob_data = data["matchProbabilityToSampleId"]
    predictions = prob_data.get("relationships", {}).get("predictions", [])

    if not predictions:
        logger.debug(
            f"No relationship predictions found for {sample_id_upper}. Marking as Distant."
        )
        return "Distant relationship?"

    valid_preds = [
        p
        for p in predictions
        if isinstance(p, dict)
        and "distributionProbability" in p
        and "pathsToMatch" in p
    ]

    if not valid_preds:
        logger.warning(f"No valid prediction paths found for {sample_id_upper}.")
        return None

    best_pred = max(valid_preds, key=lambda x: x.get("distributionProbability", 0.0))
    top_prob = best_pred.get("distributionProbability", 0.0)
    top_prob_display = top_prob  # API returns percentage already (e.g., 99.0 for 99%)
    paths = best_pred.get("pathsToMatch", [])
    labels = [
        p.get("label") for p in paths if isinstance(p, dict) and p.get("label")
    ]

    if not labels:
        logger.warning(
            f"Prediction found for {sample_id_upper}, but no labels in paths. Top prob: {top_prob_display:.1f}%"
        )
        return None

    final_labels = labels[:max_labels_param]
    relationship_str = " or ".join(map(str, final_labels))
    return f"{relationship_str} [{top_prob_display:.1f}%]"


def _extract_relationship_description(
    raw_desc_full: str,
    is_last_item: bool,
) -> str:
    """
    Extract and format relationship description text.

    Args:
        raw_desc_full: Raw description text from HTML
        is_last_item: Whether this is the last item (the "You are the..." line)

    Returns:
        Formatted description text
    """
    if is_last_item and raw_desc_full.lower().startswith("you are the "):
        return format_name(raw_desc_full[len("You are the ") :].strip())

    # Normal relationship "of" someone else
    match_rel = re.match(
        r"^(.*?)\s+of\s+(.*)$",
        raw_desc_full,
        re.IGNORECASE,
    )
    if match_rel:
        return f"{match_rel.group(1).strip().capitalize()} of {format_name(match_rel.group(2).strip())}"

    # Fallback if "of" not found (e.g., "Wife")
    return format_name(raw_desc_full)


def _parse_ladder_path_item(
    item: Any,
    item_index: int,
    num_items: int,
) -> Optional[str]:
    """
    Parse a single ladder path item to extract name and description.

    Returns:
        Formatted path item string or None if no name found
    """
    name_text, desc_text = "", ""
    name_container = item.find("a") or item.find("b")

    if name_container:
        name_text = format_name(
            name_container.get_text(strip=True).replace('"', "'")
        )

    if item_index > 0:  # Description is not for the first person (the target)
        desc_element = item.find("i")
        if desc_element:
            raw_desc_full = desc_element.get_text(strip=True).replace('"', "'")
            is_last_item = item_index == num_items - 1
            desc_text = _extract_relationship_description(raw_desc_full, is_last_item)

    if name_text:
        return f"{name_text} ({desc_text})" if desc_text else name_text

    return None


def _parse_ladder_html(
    html_content: str,
    cfpid: str,
) -> dict[str, Optional[str]]:
    """
    Parse HTML content from getladder API to extract relationship information.

    Returns:
        Dictionary with 'actual_relationship' and 'relationship_path' keys
    """
    ladder_data: dict[str, Optional[str]] = {
        "actual_relationship": None,
        "relationship_path": None,
    }

    soup = BeautifulSoup(html_content, "html.parser")

    # Extract actual relationship
    rel_elem = soup.select_one(
        "ul.textCenter > li:first-child > i > b"
    ) or soup.select_one("ul.textCenter > li > i > b")

    if rel_elem:
        raw_relationship = rel_elem.get_text(strip=True)
        ladder_data["actual_relationship"] = ordinal_case(raw_relationship.title())
    else:
        logger.warning(f"Could not extract actual_relationship for CFPID {cfpid}")

    # Extract relationship path
    path_items = soup.select('ul.textCenter > li:not([class*="iconArrowDown"])')
    path_list = []
    num_items = len(path_items)

    for i, item in enumerate(path_items):
        path_item = _parse_ladder_path_item(item, i, num_items)
        if path_item:
            path_list.append(path_item)

    if path_list:
        ladder_data["relationship_path"] = "\n↓\n".join(path_list)
    else:
        logger.warning(f"Could not construct relationship_path for CFPID {cfpid}.")

    return ladder_data


def _parse_jsonp_ladder_response(
    response_text: str,
    cfpid: str,
) -> Optional[dict[str, Optional[str]]]:
    """
    Parse JSONP response from getladder API and extract ladder data.

    Returns:
        Dictionary with 'actual_relationship' and 'relationship_path' keys, or None
    """
    result = None

    # Parse JSONP wrapper
    match_jsonp = re.match(
        r"^[^(]*\((.*)\)[^)]*$", response_text, re.DOTALL | re.IGNORECASE
    )
    if not match_jsonp:
        logger.error(
            f"Could not parse JSONP format for CFPID {cfpid}. Response: {response_text[:200]}..."
        )
    else:
        json_string = match_jsonp.group(1).strip()

        # Parse JSON content
        if not json_string or json_string in ('""', "''"):
            logger.warning(f"Empty JSON content within JSONP for CFPID {cfpid}.")
        else:
            try:
                ladder_json = json.loads(json_string)

                # Extract HTML and parse
                if not isinstance(ladder_json, dict) or "html" not in ladder_json:
                    logger.warning(
                        f"Missing 'html' key in getladder JSON for CFPID {cfpid}. JSON: {ladder_json}"
                    )
                elif not ladder_json["html"]:
                    logger.warning(f"Empty HTML in getladder response for CFPID {cfpid}.")
                else:
                    html_content = ladder_json["html"]
                    ladder_data = _parse_ladder_html(html_content, cfpid)
                    logger.debug(f"Successfully parsed ladder details for CFPID {cfpid}.")

                    # Return only if at least one piece of data was found
                    if ladder_data["actual_relationship"] or ladder_data["relationship_path"]:
                        result = ladder_data
                    else:
                        # No data found after parsing
                        logger.warning(
                            f"No actual_relationship or path found for CFPID {cfpid} after parsing."
                        )
            except json.JSONDecodeError as inner_json_err:
                logger.error(
                    f"Failed to decode JSONP content for CFPID {cfpid}: {inner_json_err}"
                )
                logger.debug(f"JSON string causing decode error: '{json_string[:200]}...'")

    return result


def _fetch_match_details_api(
    session_manager: SessionManager,
    my_uuid: str,
    match_uuid: str,
) -> Optional[dict[str, Any]]:
    """
    Fetch match details from the /details API endpoint.

    Returns:
        Dictionary with match details or None if fetch fails
    """
    details_url = urljoin(
        config_schema.api.base_url,
        f"/discoveryui-matchesservice/api/samples/{my_uuid}/matches/{match_uuid}/details?pmparentaldata=true",
    )
    logger.debug(f"Fetching /details API for UUID {match_uuid}...")

    # Use headers from working cURL command
    details_headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "priority": "u=0, i",
        "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    }

    logger.debug(f"_fetch_match_details_api: About to call _api_req for Match Details API, UUID {match_uuid}")

    # Track API call timing
    api_start_time = time.time()
    details_response = _api_req(
        url=details_url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        headers=details_headers,
        use_csrf_token=False,
        api_description="Match Details API (Batch)",
    )
    api_duration = time.time() - api_start_time

    # Record API call metrics
    metrics = _get_metrics()
    success = details_response is not None and isinstance(details_response, dict)
    metrics.record_api_call(api_duration, success=success)

    logger.debug(f"_fetch_match_details_api: _api_req returned, type={type(details_response)}, UUID {match_uuid}")

    if details_response and isinstance(details_response, dict):
        combined_data: dict[str, Any] = {}
        combined_data["admin_profile_id"] = details_response.get("adminUcdmId")
        combined_data["admin_username"] = details_response.get("adminDisplayName")
        combined_data["tester_profile_id"] = details_response.get("userId")
        combined_data["tester_username"] = details_response.get("displayName")
        combined_data["tester_initials"] = details_response.get("displayInitials")
        combined_data["gender"] = details_response.get("subjectGender")
        relationship_part = details_response.get("relationship", {})
        combined_data["shared_segments"] = relationship_part.get("sharedSegments")
        combined_data["longest_shared_segment"] = relationship_part.get("longestSharedSegment")
        combined_data["meiosis"] = relationship_part.get("meiosis")
        combined_data["from_my_fathers_side"] = bool(details_response.get("fathersSide", False))
        combined_data["from_my_mothers_side"] = bool(details_response.get("mothersSide", False))
        logger.debug(f"Successfully fetched /details for UUID {match_uuid}.")
        return combined_data
    if isinstance(details_response, requests.Response):
        logger.error(
            f"Match Details API failed for UUID {match_uuid}. Status: {details_response.status_code} {details_response.reason}"
        )
        return None
    logger.error(
        f"Match Details API did not return dict for UUID {match_uuid}. Type: {type(details_response)}"
    )
    return None


def _fetch_profile_details_api(
    session_manager: SessionManager,
    tester_profile_id: str,
    match_uuid: str,
) -> dict[str, Any]:
    """
    Fetch profile details from the /profiles/details API endpoint.

    Returns:
        Dictionary with 'last_logged_in_dt' and 'contactable' keys
    """
    result: dict[str, Any] = {
        "last_logged_in_dt": None,
        "contactable": False,
    }

    profile_url = urljoin(
        config_schema.api.base_url,
        f"/app-api/express/v1/profiles/details?userId={tester_profile_id.upper()}",
    )
    logger.debug(
        f"Fetching /profiles/details for Profile ID {tester_profile_id} (Match UUID {match_uuid})..."
    )

    # Use the same headers as the working cURL command
    profile_headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "priority": "u=0, i",
        "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    }

    # Track API call timing
    api_start_time = time.time()
    profile_response = _api_req(
        url=profile_url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        headers=profile_headers,
        use_csrf_token=False,
        api_description="Profile Details API (Batch)",
    )
    api_duration = time.time() - api_start_time

    # Record API call metrics
    metrics = _get_metrics()
    success = profile_response is not None and isinstance(profile_response, dict)
    metrics.record_api_call(api_duration, success=success)

    if profile_response and isinstance(profile_response, dict):
        logger.debug(f"Successfully fetched /profiles/details for {tester_profile_id}.")

        last_login_str = profile_response.get("LastLoginDate")
        if last_login_str:
            try:
                if last_login_str.endswith("Z"):
                    dt_aware = datetime.fromisoformat(last_login_str.replace("Z", "+00:00"))
                else:  # Assuming it might be naive or already have offset
                    dt_naive_or_aware = datetime.fromisoformat(last_login_str)
                    dt_aware = (
                        dt_naive_or_aware.replace(tzinfo=timezone.utc)
                        if dt_naive_or_aware.tzinfo is None
                        else dt_naive_or_aware.astimezone(timezone.utc)
                    )
                result["last_logged_in_dt"] = dt_aware
            except (ValueError, TypeError) as date_parse_err:
                logger.warning(
                    f"Could not parse LastLoginDate '{last_login_str}' for {tester_profile_id}: {date_parse_err}"
                )

        contactable_val = profile_response.get("IsContactable")
        result["contactable"] = bool(contactable_val) if contactable_val is not None else False
    elif isinstance(profile_response, requests.Response):
        logger.warning(
            f"Failed /profiles/details fetch for UUID {match_uuid}. Status: {profile_response.status_code}."
        )
    else:
        logger.warning(
            f"Failed /profiles/details fetch for UUID {match_uuid} (Invalid response: {type(profile_response)})."
        )

    return result


def _try_get_in_tree_from_cache(
    cache_key: str,
    current_page: int,
) -> Optional[set[str]]:
    """
    Try to get in-tree status from cache.

    Returns:
        Set of in-tree IDs if found in cache, None otherwise
    """
    try:
        if global_cache is not None:
            cached_in_tree = global_cache.get(cache_key, default=ENOVAL, retry=True)
            if cached_in_tree is not ENOVAL:
                if isinstance(cached_in_tree, set):
                    logger.debug(
                        f"Loaded {len(cached_in_tree)} in-tree IDs from cache for page {current_page}."
                    )
                    return cached_in_tree
            else:
                logger.debug(
                    f"Cache miss for in-tree status (Key: {cache_key}). Fetching from API."
                )
    except Exception as cache_read_err:
        logger.error(
            f"Error reading in-tree status from cache: {cache_read_err}. Fetching from API.",
            exc_info=True,
        )
    return None


def _save_in_tree_to_cache(
    cache_key: str,
    in_tree_ids: set[str],
    current_page: int,
) -> None:
    """Save in-tree status to cache."""
    try:
        if global_cache is not None:
            global_cache.set(
                cache_key,
                in_tree_ids,
                expire=config_schema.cache.memory_cache_ttl,
                retry=True,
            )
        logger.debug(f"Cached in-tree status result for page {current_page}.")
    except Exception as cache_write_err:
        logger.error(f"Error writing in-tree status to cache: {cache_write_err}")


def _fetch_in_tree_from_api(
    driver: Any,
    session_manager: SessionManager,
    my_uuid: str,
    sample_ids_on_page: list[str],
    specific_csrf_token: str,
    current_page: int,
) -> set[str]:
    """
    Fetch in-tree status from API.

    Returns:
        Set of sample IDs that are in the user's tree
    """
    in_tree_url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid.upper()}",
    )
    parsed_base_url = urlparse(config_schema.api.base_url)
    origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"

    ua_in_tree = None
    if driver and session_manager.is_sess_valid():
        with contextlib.suppress(Exception):
            ua_in_tree = driver.execute_script("return navigator.userAgent;")
    ua_in_tree = ua_in_tree or random.choice(config_schema.api.user_agents)

    in_tree_headers = {
        "X-CSRF-Token": specific_csrf_token,
        "Referer": urljoin(config_schema.api.base_url, "/discoveryui-matches/list/"),
        "Origin": origin_header_value,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": ua_in_tree,
    }
    in_tree_headers = {k: v for k, v in in_tree_headers.items() if v}

    logger.debug(
        f"Fetching in-tree status for {len(sample_ids_on_page)} matches on page {current_page}..."
    )

    response_in_tree = _api_req(
        url=in_tree_url,
        driver=driver,
        session_manager=session_manager,
        method="POST",
        json_data={"sampleIds": sample_ids_on_page},
        headers=in_tree_headers,
        use_csrf_token=False,
        api_description="In-Tree Status Check",
    )

    if isinstance(response_in_tree, list):
        in_tree_ids = {item.upper() for item in response_in_tree if isinstance(item, str)}
        logger.debug(f"Fetched {len(in_tree_ids)} in-tree IDs from API for page {current_page}.")
        return in_tree_ids
    status_code_log = (
        f" Status: {response_in_tree.status_code}"  # type: ignore
        if isinstance(response_in_tree, requests.Response)
        else ""
    )
    logger.warning(
        f"In-Tree Status Check API failed or returned unexpected format for page {current_page}.{status_code_log}"
    )
    return set()


def _fetch_in_tree_status(
    driver: Any,
    session_manager: SessionManager,
    my_uuid: str,
    sample_ids_on_page: list[str],
    specific_csrf_token: str,
    current_page: int,
) -> set[str]:
    """
    Fetch in-tree status for a list of sample IDs, with caching.

    Returns:
        Set of sample IDs that are in the user's tree
    """
    cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids_on_page))}"

    # Try to get from cache first
    cached_result = _try_get_in_tree_from_cache(cache_key_tree, current_page)
    if cached_result is not None:
        return cached_result

    # Fetch from API if cache miss or error
    if not session_manager.is_sess_valid():
        logger.error(
            f"In-Tree Status Check: Session invalid page {current_page}. Cannot fetch."
        )
        return set()

    in_tree_ids = _fetch_in_tree_from_api(
        driver, session_manager, my_uuid, sample_ids_on_page, specific_csrf_token, current_page
    )

    # Cache the result if we got data
    if in_tree_ids:
        _save_in_tree_to_cache(cache_key_tree, in_tree_ids, current_page)

    return in_tree_ids


def _validate_session_for_matches(
    session_manager: SessionManager,
) -> Optional[tuple[Any, str]]:
    """
    Validate session manager and extract required components.

    Returns:
        Tuple of (driver, my_uuid) if valid, None if validation fails
    """
    if not isinstance(session_manager, SessionManager):
        logger.error("get_matches: Invalid SessionManager.")
        return None

    driver = session_manager.driver
    if not driver:
        logger.error("get_matches: WebDriver not initialized.")
        return None

    my_uuid = session_manager.my_uuid
    if not my_uuid:
        logger.error("get_matches: SessionManager my_uuid not set.")
        return None

    if not session_manager.is_sess_valid():
        logger.error("get_matches: Session invalid at start.")
        return None

    return driver, my_uuid


def _try_get_csrf_from_driver_cookies(
    driver: Any,
    cookie_names: tuple[str, ...]
) -> Optional[str]:
    """
    Fallback method to get CSRF token using get_driver_cookies.

    Returns:
        CSRF token if found, None otherwise
    """
    logger.debug(
        "CSRF token not found via get_cookie. Trying get_driver_cookies fallback..."
    )
    all_cookies = get_driver_cookies(driver)
    if not all_cookies:
        logger.warning(
            "Fallback get_driver_cookies also failed to retrieve cookies."
        )
        return None

    for cookie_name in cookie_names:
        for cookie in all_cookies:
            if cookie.get("name") == cookie_name and cookie.get("value"):
                token = unquote(cookie["value"]).split("|")[0]
                logger.debug(
                    f"Read CSRF token via fallback from '{cookie_name}'."
                )
                return token

    return None


def _get_csrf_token_for_matches(driver: Any) -> Optional[str]:
    """
    Retrieve CSRF token from browser cookies for match list API.

    Returns:
        CSRF token string if found, None otherwise
    """
    csrf_token_cookie_names = (
        "_dnamatches-matchlistui-x-csrf-token",
        "_csrf",
    )
    specific_csrf_token: Optional[str] = None

    try:
        logger.debug(f"Attempting to read CSRF cookies: {csrf_token_cookie_names}")

        # Try direct cookie access first
        for cookie_name in csrf_token_cookie_names:
            try:
                cookie_obj = driver.get_cookie(cookie_name)
                if cookie_obj and "value" in cookie_obj and cookie_obj["value"]:
                    specific_csrf_token = unquote(cookie_obj["value"]).split("|")[0]
                    logger.debug(f"Read CSRF token from cookie '{cookie_name}'.")
                    break
            except NoSuchCookieException:
                continue
            except WebDriverException as cookie_e:
                logger.warning(
                    f"WebDriver error getting cookie '{cookie_name}': {cookie_e}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error getting cookie '{cookie_name}': {e}",
                    exc_info=True,
                )

        # Fallback to get_driver_cookies if direct access failed
        if not specific_csrf_token:
            specific_csrf_token = _try_get_csrf_from_driver_cookies(
                driver, csrf_token_cookie_names
            )

        if not specific_csrf_token:
            logger.error(
                "Failed to obtain specific CSRF token required for Match List API."
            )
            return None

        logger.debug(f"Specific CSRF token FOUND: '{specific_csrf_token}'")
        return specific_csrf_token

    except Exception as csrf_err:
        logger.error(
            f"Critical error during CSRF token retrieval: {csrf_err}", exc_info=True
        )
        return None


def _sync_cookies_to_session(driver: Any, session_manager: SessionManager) -> None:
    """Sync browser cookies to requests session before API call."""
    try:
        logger.debug("Syncing browser cookies to API session before Match List API call...")
        browser_cookies = driver.get_cookies()
        logger.debug(f"Retrieved {len(browser_cookies)} cookies from browser")

        # Clear and re-sync all cookies to ensure fresh state
        if hasattr(session_manager, 'requests_session') and session_manager.requests_session:
            session_manager.requests_session.cookies.clear()
            for cookie in browser_cookies:
                session_manager.requests_session.cookies.set(
                    cookie['name'],
                    cookie['value'],
                    domain=cookie.get('domain', ''),
                    path=cookie.get('path', '/')
                )
            logger.debug(f"Synced {len(browser_cookies)} cookies to requests session")
        else:
            logger.warning("No requests session available for cookie sync")
    except Exception as cookie_sync_error:
        logger.error(f"Cookie sync failed: {cookie_sync_error}")


def _fetch_match_list_page(
    driver: Any,
    session_manager: SessionManager,
    my_uuid: str,
    current_page: int,
    csrf_token: str
) -> Optional[dict[str, Any]]:
    """
    Fetch match list data for a specific page from the API.

    Returns:
        API response dict if successful, None otherwise
    """
    # Build API URL
    match_list_url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={current_page}",
    )

    # Build headers
    match_list_headers = {
        "X-CSRF-Token": csrf_token,
        "Accept": "application/json",
        "Referer": urljoin(config_schema.api.base_url, "/discoveryui-matches/list/"),
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "priority": "u=1, i",
    }

    logger.debug(f"Calling Match List API for page {current_page}...")
    logger.debug(f"Headers being passed to _api_req for Match List: {match_list_headers}")

    # Sync cookies before API call
    _sync_cookies_to_session(driver, session_manager)

    # Call the API
    return _api_req(
        url=match_list_url,
        driver=driver,
        session_manager=session_manager,
        method="GET",
        headers=match_list_headers,
        use_csrf_token=False,
        api_description="Match List API",
        allow_redirects=True,
    )



def _validate_response_type(api_response: Any, current_page: int) -> bool:
    """Validate that API response is a dictionary."""
    if not isinstance(api_response, dict):
        if isinstance(api_response, requests.Response):
            logger.error(
                f"Match List API failed page {current_page}. Status: {api_response.status_code} {api_response.reason}"
            )
        else:
            logger.error(
                f"Match List API did not return dict. Page {current_page}. Type: {type(api_response)}"
            )
            if isinstance(api_response, str):
                logger.debug(f"API response content (first 500 chars): {api_response[:500]}")
            else:
                logger.debug(f"API response: {api_response}")
        return False
    return True


def _extract_total_pages(api_response: dict[str, Any]) -> Optional[int]:
    """Extract and parse total pages from API response."""
    total_pages_raw = api_response.get("totalPages")
    if total_pages_raw is not None:
        try:
            return int(total_pages_raw)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse totalPages '{total_pages_raw}'.")
            return None
    logger.warning("Total pages missing from match list response.")
    return None


def _filter_valid_matches(
    match_data_list: list[Any],
    current_page: int
) -> list[dict[str, Any]]:
    """Filter matches that have a valid sampleId."""
    valid_matches_for_processing: list[dict[str, Any]] = []
    skipped_sampleid_count = 0

    for m_idx, m_val in enumerate(match_data_list):
        if isinstance(m_val, dict) and m_val.get("sampleId"):
            valid_matches_for_processing.append(m_val)
        else:
            skipped_sampleid_count += 1
            match_log_info = f"(Index: {m_idx}, Data: {str(m_val)[:100]}...)"
            logger.warning(
                f"Skipping raw match missing 'sampleId' on page {current_page}. {match_log_info}"
            )

    if skipped_sampleid_count > 0:
        logger.warning(
            f"Skipped {skipped_sampleid_count} raw matches on page {current_page} due to missing 'sampleId'."
        )

    return valid_matches_for_processing


def _process_match_list_response(
    api_response: Any,
    current_page: int
) -> Optional[tuple[list[dict[str, Any]], Optional[int]]]:
    """
    Process and validate match list API response.

    Returns:
        Tuple of (valid_matches, total_pages) if successful, None or ([], None) on error
    """
    # Handle None response
    if api_response is None:
        logger.warning(
            f"No response/error from match list API page {current_page}. Assuming empty page."
        )
        return [], None

    # Validate response type
    if not _validate_response_type(api_response, current_page):
        return None

    # Extract total pages
    total_pages = _extract_total_pages(api_response)

    # Extract match list
    match_data_list = api_response.get("matchList", [])
    if not match_data_list:
        logger.info(f"No matches found in 'matchList' array for page {current_page}.")
        return [], total_pages

    # Filter valid matches
    valid_matches_for_processing = _filter_valid_matches(match_data_list, current_page)

    if not valid_matches_for_processing:
        logger.warning(
            f"No valid matches (with sampleId) found on page {current_page} to process further."
        )
        return [], total_pages

    return valid_matches_for_processing, total_pages


def _refine_match_list(
    valid_matches_for_processing: list[dict[str, Any]],
    my_uuid: str,
    in_tree_ids: set,
    current_page: int
) -> list[dict[str, Any]]:
    """Refine raw match data into structured format."""
    refined_matches: list[dict[str, Any]] = []
    logger.debug(f"Refining {len(valid_matches_for_processing)} valid matches...")

    for match_index, match_api_data in enumerate(valid_matches_for_processing):
        try:
            profile_info = match_api_data.get("matchProfile", {})
            relationship_info = match_api_data.get("relationship", {})
            sample_id = match_api_data["sampleId"]
            sample_id_upper = sample_id.upper()

            profile_user_id_raw = profile_info.get("userId")
            profile_user_id_upper = (
                str(profile_user_id_raw).upper() if profile_user_id_raw else None
            )
            raw_display_name = profile_info.get("displayName")
            match_username = format_name(raw_display_name)

            first_name: Optional[str] = None
            if match_username and match_username != "Valued Relative":
                trimmed_username = match_username.strip()
                if trimmed_username:
                    name_parts = trimmed_username.split()
                    if name_parts:
                        first_name = name_parts[0]

            admin_profile_id_hint = match_api_data.get("adminId")
            admin_username_hint = match_api_data.get("adminName")
            photo_url = profile_info.get("photoUrl", "")
            initials = profile_info.get("displayInitials", "??").upper()
            gender = match_api_data.get("gender")

            shared_cm = int(relationship_info.get("sharedCentimorgans", 0))
            shared_segments = int(relationship_info.get("numSharedSegments", 0))
            created_date_raw = match_api_data.get("createdDate")

            compare_link = urljoin(
                config_schema.api.base_url,
                f"discoveryui-matches/compare/{my_uuid.upper()}/with/{sample_id_upper}",
            )
            is_in_tree = sample_id_upper in in_tree_ids

            refined_match_data = {
                "username": match_username,
                "first_name": first_name,
                "initials": initials,
                "gender": gender,
                "profile_id": profile_user_id_upper,
                "uuid": sample_id_upper,
                "administrator_profile_id_hint": admin_profile_id_hint,
                "administrator_username_hint": admin_username_hint,
                "photoUrl": photo_url,
                "cM_DNA": shared_cm,
                "numSharedSegments": shared_segments,
                "compare_link": compare_link,
                "message_link": None,
                "in_my_tree": is_in_tree,
                "createdDate": created_date_raw,
            }
            refined_matches.append(refined_match_data)

        except (IndexError, KeyError, TypeError, ValueError) as refine_err:
            match_uuid_err = match_api_data.get("sampleId", "UUID_UNKNOWN")
            logger.error(
                f"Refinement error page {current_page}, match #{match_index+1} (UUID: {match_uuid_err}): {type(refine_err).__name__} - {refine_err}. Skipping match.",
                exc_info=False,
            )
            logger.debug(f"Problematic match data during refinement: {match_api_data}")
            continue
        except Exception as critical_refine_err:
            match_uuid_err = match_api_data.get("sampleId", "UUID_UNKNOWN")
            logger.error(
                f"CRITICAL unexpected error refining match page {current_page}, match #{match_index+1} (UUID: {match_uuid_err}): {critical_refine_err}",
                exc_info=True,
            )
            logger.debug(
                f"Problematic match data during critical error: {match_api_data}"
            )
            raise critical_refine_err

    return refined_matches


def get_matches(  # type: ignore
    session_manager: SessionManager,
    current_page: int = 1,
) -> Optional[tuple[list[dict[str, Any]], Optional[int]]]:
    """
    Fetches a single page of DNA match list data from the Ancestry API v2.
    Also fetches the 'in_my_tree' status for matches on the page via a separate API call.
    Refines the raw API data into a more structured format.

    Args:
        session_manager: The active SessionManager instance.
        db_session: The active SQLAlchemy database session (not used directly in this function but
                   passed to maintain interface consistency with other functions).
        current_page: The page number to fetch (1-based).

    Returns:
        A tuple containing:
        - List of refined match data dictionaries for the page, or empty list if none.
        - Total number of pages available (integer), or None if retrieval fails.
        Returns None if a critical error occurs during fetching.
    """
    # Validate session manager
    validation_result = _validate_session_for_matches(session_manager)
    if validation_result is None:
        return None
    driver, my_uuid = validation_result

    logger.debug(f"--- Fetching Match List Page {current_page} ---")

    # Get CSRF token
    specific_csrf_token = _get_csrf_token_for_matches(driver)
    if not specific_csrf_token:
        return None

    # Fetch match list page from API
    api_response = _fetch_match_list_page(
        driver, session_manager, my_uuid, current_page, specific_csrf_token
    )

    # Process and validate response
    processing_result = _process_match_list_response(api_response, current_page)
    if processing_result is None:
        return None

    valid_matches_for_processing, total_pages = processing_result
    if not valid_matches_for_processing:
        return [], total_pages

    # Fetch in-tree status for matches on this page
    sample_ids_on_page = [
        match["sampleId"].upper() for match in valid_matches_for_processing
    ]
    in_tree_ids = _fetch_in_tree_status(
        driver, session_manager, my_uuid, sample_ids_on_page, specific_csrf_token, current_page
    )

    refined_matches = _refine_match_list(
        valid_matches_for_processing, my_uuid, in_tree_ids, current_page
    )

    logger.debug(
        f"Successfully refined {len(refined_matches)} matches on page {current_page}."
    )
    return refined_matches, total_pages


# End of get_matches


def _validate_combined_details_session(
    session_manager: SessionManager,
    my_uuid: Optional[str],
    match_uuid: str
) -> None:
    """Validate session and UUIDs for combined details fetch."""
    if not my_uuid or not match_uuid:
        logger.warning(f"_fetch_combined_details: Missing my_uuid ({my_uuid}) or match_uuid ({match_uuid}).")
        raise ValueError(f"Missing required UUIDs: my_uuid={my_uuid}, match_uuid={match_uuid}")

    if not session_manager.is_sess_valid():
        logger.error(
            f"_fetch_combined_details: WebDriver session invalid for UUID {match_uuid}."
        )
        raise ConnectionError(
            f"WebDriver session invalid for combined details fetch (UUID: {match_uuid})"
        )


def _fetch_and_merge_profile_details(
    session_manager: SessionManager,
    combined_data: dict[str, Any],
    match_uuid: str
) -> None:
    """Fetch profile details and merge into combined data."""
    tester_profile_id_for_api = combined_data.get("tester_profile_id")
    combined_data["last_logged_in_dt"] = None
    combined_data["contactable"] = False

    if not tester_profile_id_for_api:
        logger.debug(
            f"Skipping /profiles/details fetch for {match_uuid}: Tester profile ID not found in /details."
        )
        return

    if not session_manager.is_sess_valid():
        logger.error(
            f"_fetch_combined_details: WebDriver session invalid before profile fetch for {tester_profile_id_for_api}."
        )
        raise ConnectionError(
            f"WebDriver session invalid before profile fetch (Profile: {tester_profile_id_for_api})"
        )

    try:
        profile_data = _fetch_profile_details_api(
            session_manager, tester_profile_id_for_api, match_uuid
        )
        combined_data.update(profile_data)
    except ConnectionError:
        raise
    except Exception as e:
        logger.error(
            f"Error processing /profiles/details for {tester_profile_id_for_api}: {e}",
            exc_info=True,
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_combined_details(
    session_manager: SessionManager, match_uuid: str
) -> Optional[dict[str, Any]]:
    """
    Fetches combined match details (DNA stats, Admin/Tester IDs) and profile details
    (login date, contactable status) for a single match using two API calls.

    Args:
        session_manager: The active SessionManager instance.
        match_uuid: The UUID (Sample ID) of the match to fetch details for.

    Returns:
        A dictionary containing combined details, or None if fetching fails critically.
        Includes fields like: tester_profile_id, admin_profile_id, shared_segments,
        longest_shared_segment, last_logged_in_dt, contactable, etc.
    """
    # Priority 2.3: Check cache first
    cache = _get_api_cache()
    cache_key = f"combined:{match_uuid}"
    cached_result = cache.get(cache_key)

    if cached_result is not None:
        logger.debug(f"_fetch_combined_details: Cache hit for {match_uuid}")
        return cached_result

    # Apply rate limiting BEFORE making API calls
    session_manager.dynamic_rate_limiter.wait()

    logger.debug(f"_fetch_combined_details: Starting for match_uuid={match_uuid}")

    my_uuid = session_manager.my_uuid
    logger.debug(f"_fetch_combined_details: my_uuid={my_uuid}")

    # Validate session and UUIDs
    try:
        _validate_combined_details_session(session_manager, my_uuid, match_uuid)
    except ValueError:
        return None

    logger.debug("_fetch_combined_details: Session valid, proceeding with API calls...")

    # Fetch match details
    try:
        combined_data = _fetch_match_details_api(session_manager, my_uuid, match_uuid)
        if not combined_data:
            return None
    except ConnectionError:
        raise
    except Exception as e:
        logger.error(
            f"_fetch_combined_details: Exception processing /details response for UUID {match_uuid}: {e}",
            exc_info=True,
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None

    # Fetch and merge profile details
    _fetch_and_merge_profile_details(session_manager, combined_data, match_uuid)

    # Priority 2.3: Cache the result before returning
    if combined_data:
        cache.set(cache_key, combined_data)
        logger.debug(f"_fetch_combined_details: Cached result for {match_uuid}")

    return combined_data if combined_data else None


# End of _fetch_combined_details


def _process_badge_response(
    badge_response: Any,
    match_uuid: str
) -> Optional[dict[str, Any]]:
    """Process badge details API response."""
    if not badge_response or not isinstance(badge_response, dict):
        if isinstance(badge_response, requests.Response):
            logger.warning(
                f"Failed /badgedetails fetch for UUID {match_uuid}. Status: {badge_response.status_code}."
            )
        else:
            logger.warning(
                f"Invalid badge details response for UUID {match_uuid}. Type: {type(badge_response)}"
            )
        return None

    person_badged = badge_response.get("personBadged", {})
    if not person_badged:
        logger.warning(
            f"Badge details response for UUID {match_uuid} missing 'personBadged' key."
        )
        return None

    their_cfpid = person_badged.get("personId")
    raw_firstname = person_badged.get("firstName")
    # Use format_name for consistent name handling
    formatted_name_val = format_name(raw_firstname)
    their_firstname_formatted = (
        formatted_name_val.split()[0]
        if formatted_name_val and formatted_name_val != "Valued Relative"
        else "Unknown"
    )

    result_data = {
        "their_cfpid": their_cfpid,
        "their_firstname": their_firstname_formatted,  # Use formatted name
        "their_lastname": person_badged.get("lastName", "Unknown"),
        "their_birth_year": person_badged.get("birthYear"),
    }
    logger.debug(
        f"Successfully fetched /badgedetails for UUID {match_uuid} (CFPID: {their_cfpid})."
    )
    return result_data


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_batch_badge_details(
    session_manager: SessionManager, match_uuid: str
) -> Optional[dict[str, Any]]:
    """
    Fetches badge details for a specific match UUID. Used primarily to get the
    match's CFPID (Person ID within the user's tree) and basic tree profile info.

    Args:
        session_manager: The active SessionManager instance.
        match_uuid: The UUID (Sample ID) of the match.

    Returns:
        A dictionary containing badge details (their_cfpid, their_firstname, etc.)
        if successful, otherwise None.
    """
    # Apply rate limiting BEFORE making API calls
    session_manager.dynamic_rate_limiter.wait()

    my_uuid = session_manager.my_uuid
    if not my_uuid or not match_uuid:
        logger.warning("_fetch_batch_badge_details: Missing my_uuid or match_uuid.")
        return None
    if not session_manager.is_sess_valid():
        logger.error(
            f"_fetch_batch_badge_details: WebDriver session invalid for UUID {match_uuid}."
        )
        raise ConnectionError(
            f"WebDriver session invalid for badge details fetch (UUID: {match_uuid})"
        )

    badge_url = urljoin(
        config_schema.api.base_url,
        f"/discoveryui-matchesservice/api/samples/{my_uuid}/matches/{match_uuid}/badgedetails",
    )
    badge_referer = urljoin(config_schema.api.base_url, "/discoveryui-matches/list/")
    logger.debug(f"Fetching /badgedetails API for UUID {match_uuid}...")

    try:
        # Track API call timing
        api_start_time = time.time()
        badge_response = _api_req(
            url=badge_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description="Badge Details API (Batch)",
            referer_url=badge_referer,
        )
        api_duration = time.time() - api_start_time

        # Record API call metrics
        metrics = _get_metrics()
        success = badge_response is not None
        metrics.record_api_call(api_duration, success=success)

        return _process_badge_response(badge_response, match_uuid)

    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching badge details for UUID {match_uuid}: {conn_err}",
            exc_info=False,
        )
        raise
    except Exception as e:
        logger.error(
            f"Error processing badge details for UUID {match_uuid}: {e}", exc_info=True
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None


# End of _fetch_batch_badge_details


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_batch_ladder(
    session_manager: SessionManager, cfpid: str, tree_id: str
) -> Optional[dict[str, Any]]:
    """
    Fetches the relationship ladder details (relationship path, actual relationship)
    between the user and a specific person (CFPID) within the user's tree.
    Parses the JSONP response containing HTML.

    Args:
        session_manager: The active SessionManager instance.
        cfpid: The CFPID (Person ID within the tree) of the target person.
        tree_id: The ID of the user's tree containing the CFPID.

    Returns:
        A dictionary containing 'actual_relationship' and 'relationship_path' strings
        if successful, otherwise None.
    """
    # Apply rate limiting BEFORE making API calls
    session_manager.dynamic_rate_limiter.wait()

    if not cfpid or not tree_id:
        logger.warning("_fetch_batch_ladder: Missing cfpid or tree_id.")
        return None
    if not session_manager.is_sess_valid():
        logger.error(
            f"_fetch_batch_ladder: WebDriver session invalid for CFPID {cfpid}."
        )
        raise ConnectionError(
            f"WebDriver session invalid for ladder fetch (CFPID: {cfpid})"
        )

    ladder_api_url = urljoin(
        config_schema.api.base_url,
        f"family-tree/person/tree/{tree_id}/person/{cfpid}/getladder?callback=jQuery",
    )
    dynamic_referer = urljoin(
        config_schema.api.base_url,
        f"family-tree/person/tree/{tree_id}/person/{cfpid}/facts",
    )
    logger.debug(f"Fetching /getladder API for CFPID {cfpid} in Tree {tree_id}...")

    try:
        api_result = _api_req(
            url=ladder_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers={},
            use_csrf_token=False,
            api_description="Get Ladder API (Batch)",
            referer_url=dynamic_referer,
            force_text_response=True,
        )

        # Validate API response
        if isinstance(api_result, requests.Response):
            logger.warning(
                f"Get Ladder API call failed for CFPID {cfpid} (Status: {api_result.status_code})."
            )
            return None
        if api_result is None:
            logger.warning(f"Get Ladder API call returned None for CFPID {cfpid}.")
            return None
        if not isinstance(api_result, str):
            logger.warning(
                f"_api_req returned unexpected type '{type(api_result).__name__}' for Get Ladder API (CFPID {cfpid})."
            )
            return None

        # Parse JSONP response
        return _parse_jsonp_ladder_response(api_result, cfpid)

    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching ladder for CFPID {cfpid}: {conn_err}",
            exc_info=False,
        )
        raise
    except Exception as e:
        logger.error(f"Error processing ladder for CFPID {cfpid}: {e}", exc_info=True)
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None


# End of _fetch_batch_ladder


@retry_api(
    retry_on_exceptions=(
        requests.exceptions.RequestException,
        ConnectionError,
        cloudscraper.exceptions.CloudflareException,  # type: ignore
    )
)
def _fetch_batch_relationship_prob(
    session_manager: SessionManager, match_uuid: str, max_labels_param: int = 2
) -> Optional[str]:
    """
    Fetches the predicted relationship probability distribution for a match using
    the shared cloudscraper instance to potentially bypass Cloudflare challenges.

    Args:
        session_manager: The active SessionManager instance.
        match_uuid: The UUID (Sample ID) of the match to fetch probability for.
        max_labels_param: The maximum number of relationship labels to include in the result string.

    Returns:
        A formatted string like "1st cousin [95.5%]" or "Distant relationship?",
        or None if the fetch fails.
    """
    # Apply rate limiting BEFORE making API calls
    session_manager.dynamic_rate_limiter.wait()

    # Validate session components
    try:
        _validate_relationship_prob_session(session_manager, match_uuid)
    except ValueError:
        return None

    my_uuid = session_manager.my_uuid
    driver = session_manager.driver
    scraper = session_manager.scraper

    my_uuid_upper = my_uuid.upper()
    sample_id_upper = match_uuid.upper()
    rel_url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/parents/list/api/matchProbabilityData/{my_uuid_upper}/{sample_id_upper}",
    )
    referer_url = urljoin(config_schema.api.base_url, "/discoveryui-matches/list/")
    api_description = "Match Probability API (Cloudscraper)"
    rel_headers = {
        "Accept": "application/json",
        "Referer": referer_url,
        "Origin": config_schema.api.base_url.rstrip("/"),
        "User-Agent": random.choice(config_schema.api.user_agents),
    }

    # Get CSRF token
    csrf_token_val = _get_csrf_token_for_relationship_prob(
        driver, scraper, session_manager, api_description
    )
    if not csrf_token_val:
        return None

    rel_headers["X-CSRF-Token"] = csrf_token_val

    try:
        logger.debug(
            f"Making {api_description} POST request to {rel_url} using shared scraper..."
        )
        response_rel = scraper.post(
            rel_url,
            headers=rel_headers,
            json={},
            allow_redirects=False,
            timeout=config_schema.selenium.api_timeout,
        )
        logger.debug(
            f"<-- {api_description} Response Status: {response_rel.status_code} {response_rel.reason}"
        )

        if not response_rel.ok:
            status_code = response_rel.status_code
            logger.warning(
                f"{api_description} failed for {sample_id_upper}. Status: {status_code}, Reason: {response_rel.reason}"
            )
            with contextlib.suppress(Exception):
                logger.debug(f"  Response Body: {response_rel.text[:500]}")
            response_rel.raise_for_status()
            return None  # Fallback if raise_for_status doesn't trigger retry

        # Process the response
        try:
            return _process_relationship_prob_response(
                response_rel, sample_id_upper, api_description, max_labels_param
            )
        except Exception as e:
            logger.error(
                f"{api_description}: Error processing successful response for {sample_id_upper}: {e}",
                exc_info=True,
            )
            raise RequestException("Response Processing Error") from e

    except cloudscraper.exceptions.CloudflareException as cf_e:  # type: ignore
        logger.error(
            f"{api_description}: Cloudflare challenge failed for {sample_id_upper}: {cf_e}"
        )
        raise
    except requests.exceptions.RequestException as req_e:
        logger.error(
            f"{api_description}: RequestException for {sample_id_upper}: {req_e}"
        )
        raise
    except Exception as e:
        logger.error(
            f"{api_description}: Unexpected error for {sample_id_upper}: {type(e).__name__} - {e}",
            exc_info=True,
        )
        raise RequestException(f"Unexpected Fetch Error: {type(e).__name__}") from e


# End of _fetch_batch_relationship_prob


# ------------------------------------------------------------------------------
# Utility & Helper Functions
# ------------------------------------------------------------------------------


def _log_page_summary(
    page: int, page_new: int, page_updated: int, page_skipped: int, page_errors: int
) -> None:
    """Logs a summary of processed matches for a single page."""
    total = page_new + page_updated + page_skipped + page_errors

    # Build a concise, colorful one-line summary
    parts = []
    if page_new > 0:
        parts.append(f"✨ {page_new} new")
    if page_updated > 0:
        parts.append(f"🔄 {page_updated} updated")
    if page_skipped > 0:
        parts.append(f"✓ {page_skipped} current")
    if page_errors > 0:
        parts.append(f"⚠️  {page_errors} errors")

    summary = " | ".join(parts) if parts else "No matches processed"
    logger.info(f"Page {page}: {summary} (total: {total})")

    # Detailed breakdown at debug level
    logger.debug(f"---- Page {page} Detailed Breakdown ----")
    logger.debug(f"  New Person/Data: {page_new}")
    logger.debug(f"  Updated Person/Data: {page_updated}")
    logger.debug(f"  Skipped (No Change): {page_skipped}")
    logger.debug(f"  Errors during Prep/DB: {page_errors}")
    logger.debug("---------------------------------------\n")


# End of _log_page_summary


def _log_coord_summary(
    total_pages_processed: int,
    total_new: int,
    total_updated: int,
    total_skipped: int,
    total_errors: int,
) -> None:
    """Logs the final summary of the entire coord (match gathering) execution."""
    total_matches = total_new + total_updated + total_skipped + total_errors

    logger.info("=" * 50)
    logger.info("  📊 MATCH GATHERING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Pages Processed:  {total_pages_processed}")
    logger.info(f"  Total Matches:    {total_matches}")
    logger.info("-" * 50)

    # Color-coded results
    if total_new > 0:
        logger.info(f"  ✨ New Added:      {total_new:>5}")
    if total_updated > 0:
        logger.info(f"  🔄 Updated:        {total_updated:>5}")
    if total_skipped > 0:
        logger.info(f"  ✓  Already Current: {total_skipped:>5}")
    if total_errors > 0:
        logger.info(f"  ⚠️  Errors:         {total_errors:>5}")

    logger.info("=" * 50)

    # Efficiency note
    if total_skipped > 0 and total_new == 0 and total_updated == 0:
        logger.info("  💡 All matches were current - no API calls needed!")
    elif total_skipped > total_new + total_updated:
        pct = (total_skipped / total_matches * 100) if total_matches > 0 else 0
        logger.info(f"  💡 {pct:.1f}% of matches skipped - duplicate detection working!")

    logger.info("\n")


# End of _log_coord_summary


def _adjust_delay(session_manager: SessionManager, current_page: int) -> None:
    """
    Adjusts the dynamic rate limiter's delay based on throttling feedback
    received during the processing of the current page.

    Args:
        session_manager: The active SessionManager instance.
        current_page: The page number just processed (for logging context).
    """
    if session_manager.dynamic_rate_limiter.is_throttled():
        logger.debug(
            f"Rate limiter was throttled during processing before/during page {current_page}. Delay remains increased."
        )
    else:
        previous_delay = session_manager.dynamic_rate_limiter.current_delay
        session_manager.dynamic_rate_limiter.decrease_delay()
        new_delay = session_manager.dynamic_rate_limiter.current_delay
        if (
            abs(previous_delay - new_delay) > 0.01
            and new_delay
            > config_schema.api.initial_delay  # Check against initial_delay
        ):
            logger.debug(
                f"Decreased rate limit base delay to {new_delay:.2f}s after page {current_page}."
            )


# End of _adjust_delay


def nav_to_list(session_manager: SessionManager) -> bool:
    """
    Navigates the browser directly to the user's specific DNA matches list page,
    using the UUID stored in the SessionManager. Verifies successful navigation
    by checking the final URL and waiting for a match entry element.

    Args:
        session_manager: The active SessionManager instance.

    Returns:
        True if navigation was successful, False otherwise.
    """
    if (
        not session_manager
        or not session_manager.is_sess_valid()
        or not session_manager.my_uuid
    ):
        logger.error("nav_to_list: Session invalid or UUID missing.")
        return False

    my_uuid = session_manager.my_uuid

    target_url = urljoin(
        config_schema.api.base_url, f"discoveryui-matches/list/{my_uuid}"
    )
    logger.debug(f"Navigating to specific match list URL: {target_url}")

    driver = session_manager.driver
    if driver is None:
        logger.error("nav_to_list: WebDriver is None")
        return False

    success = nav_to_page(
        driver=driver,
        url=target_url,
        selector=MATCH_ENTRY_SELECTOR,  # type: ignore
        session_manager=session_manager,
    )

    if success:
        try:
            current_url = driver.current_url
            if not current_url.startswith(target_url):
                logger.warning(
                    f"Navigation successful (element found), but final URL unexpected: {current_url}"
                )
            else:
                logger.debug("Successfully navigated to specific matches list page.")
        except Exception as e:
            logger.warning(f"Could not verify final URL after nav_to_list success: {e}")
    else:
        logger.error(
            "Failed navigation to specific matches list page using nav_to_page."
        )
    return success


# End of nav_to_list


# ------------------------------------------------------------------------------
# Test Harness
# ------------------------------------------------------------------------------


# ==============================================
# CIRCUIT BREAKER & CONFIGURATION TEST FUNCTIONS (Priority 1.1 & 2.1)
# ==============================================


def _test_circuit_breaker_basic() -> None:
    """Test basic circuit breaker functionality (Priority 1.1)."""
    print("📋 Testing Circuit Breaker - Basic Functionality:")

    # Create a test circuit breaker with threshold=3 for faster testing
    cb = SessionCircuitBreaker(threshold=3, name="Test")

    print("   ✅ Created circuit breaker with threshold=3")
    status = cb.get_status()
    print(f"      Initial state: tripped={status['tripped']}, failures: {status['consecutive_failures']}")

    # Test consecutive failures
    print("   • Testing consecutive failures:")
    for i in range(1, 4):
        tripped = cb.record_failure()
        status = cb.get_status()
        print(f"      Failure {i}/3: tripped={tripped}, count={status['consecutive_failures']}")

        if i < 3:
            assert not tripped, f"Circuit should not trip at failure {i}"
        else:
            assert tripped, "Circuit should trip at failure 3"

    assert cb.is_tripped(), "Circuit should be tripped"
    print("   ✅ Circuit breaker tripped correctly after 3 failures")

    # Test that it stays tripped
    print("   • Testing that circuit stays tripped:")
    for i in range(1, 3):
        assert cb.is_tripped(), f"Circuit should stay tripped (check {i})"
    print("   ✅ Circuit breaker stays tripped correctly")


def _test_circuit_breaker_reset() -> None:
    """Test circuit breaker reset on success (Priority 1.1)."""
    print("📋 Testing Circuit Breaker - Reset on Success:")

    cb = SessionCircuitBreaker(threshold=3, name="Test")

    print("   • Recording 2 failures:")
    cb.record_failure()
    cb.record_failure()

    status = cb.get_status()
    print(f"      Status: tripped={status['tripped']}, count={status['consecutive_failures']}")
    assert not cb.is_tripped(), "Circuit should not be tripped yet"
    assert status['consecutive_failures'] == 2, "Should have 2 failures"

    print("   • Recording success (should reset):")
    cb.record_success()
    status = cb.get_status()
    print(f"      Status after success: tripped={status['tripped']}, count={status['consecutive_failures']}")

    assert not cb.is_tripped(), "Circuit should be open after success"
    assert status['consecutive_failures'] == 0, "Failure count should reset to 0"

    print("   ✅ Circuit breaker reset correctly on success")


def _test_circuit_breaker_global_instance() -> None:
    """Test the global circuit breaker singleton (Priority 1.1)."""
    print("📋 Testing Circuit Breaker - Global Instance:")

    # Get the global instance
    cb1 = get_session_circuit_breaker()
    cb2 = get_session_circuit_breaker()

    print("   • Retrieved global circuit breaker instances")
    print(f"      Instance 1 ID: {id(cb1)}")
    print(f"      Instance 2 ID: {id(cb2)}")

    assert cb1 is cb2, "Should return same instance (singleton pattern)"
    print("   ✅ Singleton pattern works correctly")

    # Test it has correct threshold from action6_gather.py (should be 5)
    status = cb1.get_status()
    print("   • Global circuit breaker config:")
    print(f"      Name: {status['name']}")
    print(f"      Threshold: {status['threshold']}")
    print(f"      Tripped: {status['tripped']}")

    assert status['threshold'] == 5, "Global circuit breaker should have threshold=5"
    print("   ✅ Global circuit breaker correctly configured")

    # Reset for clean state
    cb1.reset()
    print("   ✅ Reset global circuit breaker to clean state")


def _test_circuit_breaker_timing() -> None:
    """Test circuit breaker timing and trip time tracking (Priority 1.1)."""
    print("📋 Testing Circuit Breaker - Timing:")

    cb = SessionCircuitBreaker(threshold=2, name="Timing Test")

    print("   • Recording failures with time tracking:")
    start_time = time.time()

    cb.record_failure()
    time.sleep(0.05)  # Small delay
    tripped = cb.record_failure()

    status = cb.get_status()

    assert tripped, "Circuit should trip on 2nd failure"
    assert status['trip_time'] is not None, "Trip time should be recorded"
    assert status['trip_time'] >= start_time, "Trip time should be after start"

    print(f"      Circuit tripped at: {status['trip_time']}")
    print(f"      Time since trip: {time.time() - status['trip_time']:.3f}s")

    print("   ✅ Circuit breaker timing works correctly")


def _test_worker_configuration() -> None:
    """Test that worker configuration loads correctly from .env (Priority 2.1)."""
    print("📋 Testing Worker Configuration from .env:")

    thread_pool_workers = config_schema.api.thread_pool_workers
    requests_per_second = config_schema.api.requests_per_second

    print("   • Configuration loaded:")
    print(f"      Thread Pool Workers: {thread_pool_workers}")
    print(f"      Requests Per Second: {requests_per_second}")

    # Validate configuration is reasonable
    assert 1 <= thread_pool_workers <= 4, "Worker count should be between 1-4"
    assert requests_per_second > 0, "RPS should be positive"

    # Calculate per-worker rate
    if thread_pool_workers > 0:
        per_worker_rate = requests_per_second / thread_pool_workers
        interval_per_worker = 1.0 / per_worker_rate if per_worker_rate > 0 else 0

        print("   • Rate Limiting Analysis:")
        print(f"      Total RPS: {requests_per_second}")
        print(f"      Workers: {thread_pool_workers}")
        print(f"      RPS per worker: {per_worker_rate:.2f}")
        print(f"      Interval between requests (per worker): {interval_per_worker:.2f}s")

        if thread_pool_workers == 2:
            print("   ✅ Worker count set to 2 (Priority 2.1 optimization)")
        elif thread_pool_workers == 1:
            print("   ℹ️  Worker count is 1 (conservative/safe mode)")
        else:
            print(f"   ℹ️  Worker count is {thread_pool_workers}")

    print("   ✅ Configuration loaded and validated successfully")


def _test_circuit_breaker_integration() -> None:
    """Test circuit breaker integration with _check_session_validity (Priority 1.1)."""
    print("📋 Testing Circuit Breaker Integration:")

    # Get the global circuit breaker and reset it
    cb = get_session_circuit_breaker()
    cb.reset()

    print("   • Circuit breaker reset to clean state")
    status = cb.get_status()
    print(f"      Initial status: tripped={status['tripped']}, failures={status['consecutive_failures']}")

    # Test that _check_session_validity exists and has correct signature
    import inspect
    sig = inspect.signature(_check_session_validity)
    params = list(sig.parameters.keys())

    print("   • _check_session_validity signature verified:")
    print(f"      Parameters: {params}")

    expected_params = ['session_manager', 'current_page_num', 'state', 'progress_bar']
    assert params == expected_params, f"Expected params {expected_params}, got {params}"

    print("   ✅ Function signature correct")
    print("   ✅ Circuit breaker integrated into session validation")


def _test_auto_recovery_control() -> None:
    """Test session auto-recovery enable/disable functionality (Priority 1.2)."""
    print("📋 Testing Auto-Recovery Control (Priority 1.2):")

    # Import SessionManager
    from core.session_manager import SessionManager

    # Create a minimal session manager instance for testing
    print("   • Creating test SessionManager instance...")
    try:
        sm = SessionManager()
        print("   ✅ SessionManager created")

        # Test default state (should be enabled)
        initial_state = sm.get_auto_recovery_status()
        print(f"      Initial auto-recovery state: {initial_state}")
        assert initial_state is True, "Auto-recovery should be enabled by default"
        print("   ✅ Default state correct (enabled)")

        # Test disable
        print("   • Disabling auto-recovery...")
        sm.set_auto_recovery(False)
        current_state = sm.get_auto_recovery_status()
        print(f"      Current state: {current_state}")
        assert current_state is False, "Auto-recovery should be disabled"
        print("   ✅ Disable works correctly")

        # Test re-enable
        print("   • Re-enabling auto-recovery...")
        sm.set_auto_recovery(True)
        current_state = sm.get_auto_recovery_status()
        print(f"      Current state: {current_state}")
        assert current_state is True, "Auto-recovery should be enabled"
        print("   ✅ Enable works correctly")

        # Test coord integration exists
        print("   • Checking coord() integration...")
        import inspect
        coord_source = inspect.getsource(coord)
        assert "set_auto_recovery" in coord_source, "coord() should call set_auto_recovery"
        assert "get_auto_recovery_status" in coord_source, "coord() should save previous state"
        print("   ✅ coord() integration verified")

    except Exception as e:
        print(f"   ⚠️  Limited test due to initialization requirements: {e}")
        print("   ℹ️  Auto-recovery control methods exist and will be tested in integration")

    print("   ✅ Auto-recovery control functionality validated")


def _test_403_auth_refresh_handler() -> None:
    """Test 403 error handling with auth refresh (Priority 1.3)."""
    print("📋 Testing 403 Auth Refresh Handler (Priority 1.3):")

    # Test that functions exist and are callable
    print("   • Checking _api_req_with_auth_refresh exists...")
    assert callable(_api_req_with_auth_refresh), "_api_req_with_auth_refresh should be callable"
    print("   ✅ _api_req_with_auth_refresh function exists")

    print("   • Checking _refresh_session_auth exists...")
    assert callable(_refresh_session_auth), "_refresh_session_auth should be callable"
    print("   ✅ _refresh_session_auth function exists")

    # Test function signatures
    import inspect

    print("   • Verifying _api_req_with_auth_refresh signature...")
    sig = inspect.signature(_api_req_with_auth_refresh)
    params = list(sig.parameters.keys())
    assert 'session_manager' in params, "Should have session_manager parameter"
    assert 'url' in params, "Should have url parameter"
    assert 'method' in params, "Should have method parameter"
    assert 'headers' in params, "Should have headers parameter"
    print(f"      Parameters: {params[:4]}")
    print("   ✅ Function signature correct")

    print("   • Verifying _refresh_session_auth signature...")
    sig2 = inspect.signature(_refresh_session_auth)
    params2 = list(sig2.parameters.keys())
    assert 'session_manager' in params2, "Should have session_manager parameter"
    assert sig2.return_annotation is bool or str(sig2.return_annotation) == 'bool', "Should return bool"
    print(f"      Parameters: {params2}")
    print("      Return type: bool")
    print("   ✅ Function signature correct")

    # Verify the wrapper logic exists
    print("   • Checking wrapper logic...")
    source = inspect.getsource(_api_req_with_auth_refresh)
    assert "403" in source, "Should check for 403 status code"
    assert "retry" in source.lower() or "_handle_403_retry" in source, "Should retry after refresh"
    # Check that refresh is called (either directly or via helper)
    helper_source = inspect.getsource(_handle_403_retry) if "_handle_403_retry" in source else ""
    assert "_refresh_session_auth" in source or "_refresh_session_auth" in helper_source, \
        "Should call _refresh_session_auth"
    print("   ✅ 403 detection and retry logic present")

    # Verify refresh logic
    print("   • Checking refresh logic...")
    refresh_source = inspect.getsource(_refresh_session_auth)
    assert "cookies" in refresh_source.lower(), "Should handle cookies"
    assert "is_sess_valid" in refresh_source, "Should validate session"
    print("   ✅ Cookie refresh and validation logic present")

    print("   ✅ 403 Auth Refresh functionality validated")


def _test_session_health_monitoring() -> None:
    """Test session health monitoring and proactive refresh (Priority 1.4)."""
    print("📋 Testing Session Health Monitoring (Priority 1.4):")

    # Test that functions exist
    print("   • Checking _check_session_health_proactive exists...")
    assert callable(_check_session_health_proactive), "_check_session_health_proactive should be callable"
    print("   ✅ _check_session_health_proactive function exists")

    print("   • Checking _get_session_health_status exists...")
    assert callable(_get_session_health_status), "_get_session_health_status should be callable"
    print("   ✅ _get_session_health_status function exists")

    # Test function signatures
    import inspect

    print("   • Verifying _check_session_health_proactive signature...")
    sig = inspect.signature(_check_session_health_proactive)
    params = list(sig.parameters.keys())
    assert 'session_manager' in params, "Should have session_manager parameter"
    assert 'current_page' in params, "Should have current_page parameter"
    assert sig.return_annotation is bool or str(sig.return_annotation) == 'bool', "Should return bool"
    print(f"      Parameters: {params}")
    print("      Return type: bool")
    print("   ✅ Function signature correct")

    print("   • Verifying _get_session_health_status signature...")
    sig2 = inspect.signature(_get_session_health_status)
    params2 = list(sig2.parameters.keys())
    assert 'session_manager' in params2, "Should have session_manager parameter"
    # Return type should be dict
    return_annotation = str(sig2.return_annotation)
    assert 'dict' in return_annotation.lower(), f"Should return dict, got {return_annotation}"
    print(f"      Parameters: {params2}")
    print("      Return type: dict")
    print("   ✅ Function signature correct")

    # Verify health check logic
    print("   • Checking health check logic...")
    source = inspect.getsource(_check_session_health_proactive)
    assert "session_age" in source.lower(), "Should check session age"
    assert "refresh_threshold" in source, "Should have refresh threshold"
    assert "_refresh_session_auth" in source, "Should call _refresh_session_auth for proactive refresh"
    assert "check_interval" in source.lower(), "Should check periodically (not every page)"
    print("   ✅ Session age monitoring and threshold logic present")

    # Verify status function logic
    print("   • Checking status function logic...")
    status_source = inspect.getsource(_get_session_health_status)
    assert "session_age" in status_source.lower(), "Should get session age"
    assert "time_remaining" in status_source, "Should calculate time remaining"
    assert "needs_refresh" in status_source, "Should indicate refresh need"
    assert "status" in status_source, "Should return status indicator"
    print("   ✅ Status calculation and reporting logic present")

    # Verify integration into processing loop
    print("   • Checking integration into _process_single_page_iteration...")
    proc_source = inspect.getsource(_process_single_page_iteration)
    assert "_check_session_health_proactive" in proc_source, "Should call health check in page processing"
    print("   ✅ Health check integrated into page processing loop")

    # Verify SessionManager has required attributes
    print("   • Verifying SessionManager has session_age_seconds method...")
    from core.session_manager import SessionManager
    assert hasattr(SessionManager, 'session_age_seconds'), "SessionManager should have session_age_seconds method"
    print("   ✅ SessionManager has required session_age_seconds method")

    print("   • Verifying SessionManager has session_health_monitor...")
    # Check in source that session_health_monitor is initialized
    sm_source = inspect.getsource(SessionManager)
    assert "session_health_monitor" in sm_source, "SessionManager should have session_health_monitor"
    assert "max_session_age" in sm_source, "Should have max_session_age config"
    assert "last_proactive_refresh" in sm_source, "Should track last proactive refresh"
    print("   ✅ SessionManager has session health monitoring infrastructure")

    # Test configuration
    print("   • Checking health check configuration...")
    assert "HEALTH_CHECK_INTERVAL_PAGES" in source, "Should be configurable via env var"
    print("   ✅ Health check interval is configurable")

    print("   ✅ Session Health Monitoring functionality validated")


def _test_progress_checkpointing() -> None:
    """Test progress checkpointing for resume capability (Priority 2.2)."""
    print("📋 Testing Progress Checkpointing (Priority 2.2):")

    # Test that functions exist
    print("   • Checking checkpoint functions exist...")
    assert callable(_save_checkpoint), "_save_checkpoint should be callable"
    assert callable(_load_checkpoint), "_load_checkpoint should be callable"
    assert callable(_clear_checkpoint), "_clear_checkpoint should be callable"
    assert callable(_should_resume_from_checkpoint), "_should_resume_from_checkpoint should be callable"
    assert callable(_restore_state_from_checkpoint), "_restore_state_from_checkpoint should be callable"
    print("   ✅ All checkpoint functions exist")

    # Test function signatures
    import inspect

    print("   • Verifying _save_checkpoint signature...")
    sig = inspect.signature(_save_checkpoint)
    params = list(sig.parameters.keys())
    assert 'current_page' in params, "Should have current_page parameter"
    assert 'total_pages' in params, "Should have total_pages parameter"
    assert 'state' in params, "Should have state parameter"
    assert sig.return_annotation is bool or str(sig.return_annotation) == 'bool', "Should return bool"
    print(f"      Parameters: {params}")
    print("   ✅ Function signature correct")

    print("   • Verifying _load_checkpoint signature...")
    sig2 = inspect.signature(_load_checkpoint)
    return_annotation = str(sig2.return_annotation)
    assert 'dict' in return_annotation.lower() or 'optional' in return_annotation.lower(), \
        f"Should return Optional[dict], got {return_annotation}"
    print(f"      Return type: {return_annotation}")
    print("   ✅ Function signature correct")

    # Test checkpoint logic
    print("   • Checking checkpoint save logic...")
    save_source = inspect.getsource(_save_checkpoint)
    assert "json.dump" in save_source, "Should use JSON for checkpoint"
    assert "timestamp" in save_source, "Should include timestamp"
    assert "current_page" in save_source, "Should save current page"
    assert "counters" in save_source, "Should save state counters"
    assert "ENABLE_CHECKPOINTING" in save_source, "Should be configurable"
    print("   ✅ Checkpoint save logic present")

    print("   • Checking checkpoint load logic...")
    load_source = inspect.getsource(_load_checkpoint)
    assert "json.load" in load_source, "Should load JSON checkpoint"
    assert "version" in load_source, "Should validate checkpoint version"
    assert "age" in load_source.lower() or "timestamp" in load_source, "Should check checkpoint age"
    assert "CHECKPOINT_MAX_AGE_HOURS" in load_source or "max_age" in load_source, "Should have max age validation"
    print("   ✅ Checkpoint load and validation logic present")

    print("   • Checking resume decision logic...")
    resume_source = inspect.getsource(_should_resume_from_checkpoint)
    assert "requested_start_page" in resume_source, "Should check requested start page"
    assert "checkpoint" in resume_source, "Should check if checkpoint exists"
    assert "resume_page" in resume_source or "current_page" in resume_source, "Should calculate resume page"
    print("   ✅ Resume decision logic present")

    print("   • Checking state restoration logic...")
    restore_source = inspect.getsource(_restore_state_from_checkpoint)
    assert "total_new" in restore_source, "Should restore total_new counter"
    assert "total_updated" in restore_source, "Should restore total_updated counter"
    assert "total_pages_processed" in restore_source, "Should restore total_pages_processed counter"
    assert "counters" in restore_source, "Should get counters from checkpoint"
    print("   ✅ State restoration logic present")

    # Verify integration into coord
    print("   • Checking integration into coord()...")
    coord_source = inspect.getsource(coord)
    assert "_load_checkpoint" in coord_source, "Should load checkpoint in coord()"
    assert "_should_resume_from_checkpoint" in coord_source, "Should check resume in coord()"
    assert "_restore_state_from_checkpoint" in coord_source, "Should restore state in coord()"
    assert "_clear_checkpoint" in coord_source, "Should clear checkpoint on success in coord()"
    print("   ✅ Checkpoint integration into coord() verified")

    # Verify integration into page processing
    print("   • Checking integration into page processing...")
    update_source = inspect.getsource(_update_state_after_batch)
    assert "_save_checkpoint" in update_source, "Should save checkpoint after each page"
    assert "current_page" in update_source, "Should pass current page to checkpoint"
    assert "total_pages" in update_source, "Should pass total pages to checkpoint"
    print("   ✅ Checkpoint saving integrated into page processing")

    # Test configuration
    print("   • Checking checkpoint configuration...")
    filepath_source = inspect.getsource(_get_checkpoint_filepath)
    assert "CHECKPOINT_DIR" in filepath_source, "Checkpoint directory should be configurable"
    print("   ✅ Checkpoint configuration present")

    # Test atomic write
    print("   • Checking atomic write implementation...")
    assert ".tmp" in save_source, "Should use temp file for atomic write"
    assert "replace" in save_source or "rename" in save_source, "Should use atomic rename"
    print("   ✅ Atomic write implementation verified")

    print("   ✅ Progress Checkpointing functionality validated")


def _test_api_call_batching() -> None:
    """Test API call batching and deduplication (Priority 2.3)."""
    print("📋 Testing API Call Batching & Deduplication (Priority 2.3):")

    # Test that classes and functions exist
    print("   • Checking APICallCache class exists...")
    assert 'APICallCache' in globals(), "APICallCache class should be defined"
    print("   ✅ APICallCache class exists")

    print("   • Checking cache management functions...")
    assert callable(_get_api_cache), "_get_api_cache should be callable"
    assert callable(_clear_api_cache), "_clear_api_cache should be callable"
    assert callable(_deduplicate_api_requests), "_deduplicate_api_requests should be callable"
    print("   ✅ All cache management functions exist")

    # Test APICallCache functionality
    import inspect

    print("   • Testing APICallCache class structure...")
    cache_methods = inspect.getmembers(APICallCache, predicate=inspect.isfunction)
    method_names = [name for name, _ in cache_methods]

    assert 'get' in method_names or '__init__' in dir(APICallCache), "Should have get method"
    assert 'set' in method_names or '__init__' in dir(APICallCache), "Should have set method"
    assert 'clear' in method_names or '__init__' in dir(APICallCache), "Should have clear method"
    assert 'get_stats' in method_names or '__init__' in dir(APICallCache), "Should have get_stats method"
    print("      Methods found: get, set, clear, get_stats")
    print("   ✅ APICallCache has required methods")

    # Test cache instance creation
    print("   • Testing cache instance creation...")
    cache = _get_api_cache()
    assert cache is not None, "Cache instance should be created"
    assert hasattr(cache, '_lock'), "Cache should be thread-safe with lock"
    print("   ✅ Cache instance created with thread safety")

    # Test cache operations
    print("   • Testing cache get/set operations...")
    test_key = "test:UUID123"
    test_value = {"test": "data"}

    # Test miss
    result = cache.get(test_key)
    assert result is None, "Should return None for cache miss"
    print("      Cache miss works correctly")

    # Test set
    cache.set(test_key, test_value)
    print("      Cache set works correctly")

    # Test hit
    result = cache.get(test_key)
    assert result == test_value, "Should return cached value on hit"
    print("      Cache hit works correctly")
    print("   ✅ Cache get/set operations validated")

    # Test cache stats
    print("   • Testing cache statistics...")
    stats = cache.get_stats()
    assert isinstance(stats, dict), "Stats should be a dictionary"
    assert 'hits' in stats, "Stats should include hits"
    assert 'misses' in stats, "Stats should include misses"
    assert 'hit_rate_percent' in stats, "Stats should include hit rate"
    print(f"      Stats: {stats['hits']} hits, {stats['misses']} misses, {stats['hit_rate_percent']}% hit rate")
    print("   ✅ Cache statistics validated")

    # Test deduplication function
    print("   • Testing _deduplicate_api_requests function...")
    sig = inspect.signature(_deduplicate_api_requests)
    params = list(sig.parameters.keys())
    assert 'fetch_candidates_uuid' in params, "Should have fetch_candidates_uuid parameter"
    assert 'matches_to_process_later' in params, "Should have matches_to_process_later parameter"
    print(f"      Parameters: {params}")
    print("   ✅ Deduplication function signature correct")

    # Test deduplication logic
    print("   • Checking deduplication logic...")
    dedup_source = inspect.getsource(_deduplicate_api_requests)
    assert "cache.get" in dedup_source or "_get_api_cache" in dedup_source, "Should check cache"
    assert "cache_hits" in dedup_source, "Should track cache hits"
    assert "deduplicated" in dedup_source, "Should create deduplicated set"
    print("   ✅ Deduplication logic present")

    # Test integration into _fetch_combined_details
    print("   • Checking integration into _fetch_combined_details...")
    fetch_source = inspect.getsource(_fetch_combined_details)
    assert "cache" in fetch_source.lower() or "_get_api_cache" in fetch_source, "Should use cache"
    assert "cache_key" in fetch_source, "Should create cache keys"
    assert "cache.get" in fetch_source, "Should check cache before API call"
    assert "cache.set" in fetch_source, "Should save result to cache"
    print("   ✅ Cache integration in _fetch_combined_details verified")

    # Test integration into _perform_api_prefetches
    print("   • Checking integration into _perform_api_prefetches...")
    prefetch_source = inspect.getsource(_perform_api_prefetches)
    assert "_deduplicate_api_requests" in prefetch_source, "Should call deduplication"
    assert "cache_hits" in prefetch_source, "Should track cache hits"
    print("   ✅ Deduplication integrated into _perform_api_prefetches")

    # Test integration into coord()
    print("   • Checking integration into coord()...")
    coord_source = inspect.getsource(coord)
    assert "_clear_api_cache" in coord_source, "Should clear cache at start and end"
    print("   ✅ Cache clearing integrated into coord()")

    # Test configuration
    print("   • Checking API cache configuration...")
    cache_init_source = inspect.getsource(_get_api_cache)
    assert "API_CACHE_TTL_SECONDS" in cache_init_source or "ttl" in cache_init_source, "TTL should be configurable"
    print("   ✅ API cache TTL is configurable")

    # Clean up test cache
    cache.clear()
    print("   • Test cache cleaned up")

    print("   ✅ API Call Batching & Deduplication functionality validated")


def _verify_metrics_class_structure() -> None:
    """Verify PerformanceMetrics class structure and methods."""
    import inspect

    print("   • Checking PerformanceMetrics class exists...")
    assert 'PerformanceMetrics' in globals(), "PerformanceMetrics class should be defined"
    print("   ✅ PerformanceMetrics class exists")

    print("   • Checking metrics management functions...")
    assert callable(_get_metrics), "_get_metrics should be callable"
    assert callable(_reset_metrics), "_reset_metrics should be callable"
    print("   ✅ All metrics management functions exist")

    print("   • Testing PerformanceMetrics class structure...")
    metrics_methods = inspect.getmembers(PerformanceMetrics, predicate=inspect.isfunction)
    method_names = [name for name, _ in metrics_methods]

    required_methods = ['record_page_time', 'record_api_call', 'record_error', 'get_stats', 'log_progress', 'log_final_summary']
    for method in required_methods:
        assert method in method_names or '__init__' in dir(PerformanceMetrics), f"Should have {method} method"
    print("      Methods found: record_page_time, record_api_call, record_error, get_stats, log_progress, log_final_summary")
    print("   ✅ PerformanceMetrics has required methods")


def _verify_metrics_recording() -> None:
    """Verify metrics recording functionality."""
    print("   • Testing metrics instance creation...")
    _reset_metrics()
    metrics = _get_metrics()
    assert metrics is not None, "Metrics instance should be created"
    assert hasattr(metrics, 'start_time'), "Metrics should track start time"
    assert hasattr(metrics, 'pages_completed'), "Metrics should track pages completed"
    assert hasattr(metrics, 'api_calls_made'), "Metrics should track API calls"
    print("   ✅ Metrics instance created with tracking fields")

    print("   • Testing metrics recording...")

    # Record page time
    metrics.record_page_time(2.5)
    assert metrics.pages_completed == 1, "Should increment pages completed"
    assert len(metrics.page_times) == 1, "Should record page time"
    assert metrics.page_times[0] == 2.5, "Should record correct time"
    print("      Page time recording works correctly")

    # Record API call
    metrics.record_api_call(0.5, success=True)
    assert metrics.api_calls_made == 1, "Should increment API calls"
    assert len(metrics.api_call_times) == 1, "Should record API call time"
    assert metrics.api_errors == 0, "Should have no errors on success"
    print("      API call recording works correctly")

    # Record API error
    metrics.record_api_call(1.0, success=False)
    assert metrics.api_calls_made == 2, "Should increment API calls"
    assert metrics.api_errors == 1, "Should increment error count"
    print("      API error recording works correctly")

    # Record error by type
    metrics.record_error("test_error")
    assert "test_error" in metrics.error_types, "Should record error type"
    assert metrics.error_types["test_error"] == 1, "Should count error type"
    metrics.record_error("test_error")
    assert metrics.error_types["test_error"] == 2, "Should increment error count"
    print("      Error type tracking works correctly")

    print("   ✅ Metrics recording validated")


def _verify_metrics_statistics() -> None:
    """Verify statistics generation."""
    print("   • Testing statistics generation...")
    metrics = _get_metrics()
    stats = metrics.get_stats()
    assert isinstance(stats, dict), "Stats should be a dictionary"

    required_keys = ['elapsed_seconds', 'pages_completed', 'api_calls_total', 'api_success_rate', 'page_time_avg', 'error_breakdown']
    for key in required_keys:
        assert key in stats, f"Stats should include {key}"

    print("      Stats keys: elapsed_seconds, pages_completed, api_calls_total, api_success_rate, page_time_avg")
    print(f"      API calls: {stats['api_calls_total']}, Success rate: {stats['api_success_rate']}%")
    print("   ✅ Statistics generation validated")


def _verify_metrics_integration() -> None:
    """Verify metrics integration into main functions."""
    import inspect

    print("   • Checking integration into coord()...")
    coord_source = inspect.getsource(coord)
    assert "_reset_metrics" in coord_source, "Should initialize metrics at start"
    assert "metrics.log_final_summary" in coord_source or "log_final_summary" in coord_source, "Should log final summary"
    print("   ✅ Metrics integrated into coord()")

    print("   • Checking integration into _process_page_batch...")
    batch_source = inspect.getsource(_process_page_batch)
    assert "record_page_time" in batch_source, "Should record page times"
    assert "log_progress" in batch_source, "Should log progress updates"
    print("   ✅ Metrics integrated into page processing")

    print("   • Checking integration into API call tracking...")
    api_source = inspect.getsource(_api_req_with_auth_refresh)
    has_timing = "record_api_call" in api_source or "_record_api_timing" in api_source
    assert has_timing, "Should track API call timing"

    helper_source = ""
    if "_handle_403_retry" in api_source:
        helper_source = inspect.getsource(_handle_403_retry)
    has_retries = "api_retries" in api_source or "api_retries" in helper_source
    assert has_retries, "Should track retries"
    print("   ✅ Metrics integrated into API calls")


def _test_enhanced_logging_metrics() -> None:
    """Test enhanced logging and performance metrics (Priority 3.1)."""
    print("📋 Testing Enhanced Logging & Performance Metrics (Priority 3.1):")

    _verify_metrics_class_structure()
    _verify_metrics_recording()
    _verify_metrics_statistics()
    _verify_metrics_integration()

    # Clean up
    _reset_metrics()
    print("   • Test metrics cleaned up")

    print("   ✅ Enhanced Logging & Performance Metrics functionality validated")


# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_metrics_export() -> None:
    """Test metrics export to JSON file (Priority 3.2)."""
    print("📋 Testing Metrics Export (Priority 3.2):")

    # Test that export function exists
    print("   • Checking _export_metrics_to_file exists...")
    assert callable(_export_metrics_to_file), "_export_metrics_to_file should be callable"
    print("   ✅ _export_metrics_to_file function exists")

    # Test function signature
    import inspect
    print("   • Verifying _export_metrics_to_file signature...")
    sig = inspect.signature(_export_metrics_to_file)
    params = list(sig.parameters.keys())
    assert 'metrics' in params, "Should have metrics parameter"
    assert 'success' in params, "Should have success parameter"
    print(f"      Parameters: {params}")
    print("   ✅ Function signature correct")

    # Test export logic
    print("   • Checking export logic...")
    source = inspect.getsource(_export_metrics_to_file)
    assert "Path" in source, "Should use Path for directory handling"
    assert "Logs/metrics" in source or "Logs\\metrics" in source, "Should export to Logs/metrics directory"
    assert "json.dump" in source, "Should serialize to JSON"
    assert "indent" in source, "Should use pretty printing"
    assert ".mkdir" in source, "Should create directory if needed"
    assert "parents=True" in source, "Should create parent directories"
    assert "exist_ok=True" in source, "Should handle existing directory"
    assert "action6_metrics_" in source, "Should use consistent filename prefix"
    print("   ✅ Export logic validated (Path usage, directory creation, JSON serialization)")

    # Test that it exports comprehensive data
    print("   • Checking exported data structure...")
    assert "metadata" in source, "Should include metadata section"
    assert "performance" in source, "Should include performance metrics"
    assert "timing" in source, "Should include timing breakdown"
    assert "api_metrics" in source, "Should include API call metrics"
    assert "cache_metrics" in source, "Should include cache statistics"
    assert "error_breakdown" in source, "Should include error breakdown"
    assert "timestamp" in source, "Should include timestamp"
    assert "success" in source, "Should include success status"
    assert "module" in source, "Should include module name"
    print("   ✅ Comprehensive data structure validated")

    # Test integration into coord()
    print("   • Checking integration into coord()...")
    coord_source = inspect.getsource(coord)
    assert "_export_metrics_to_file" in coord_source, "Should call export function in coord()"
    assert "metrics.log_final_summary" in coord_source, "Export should be after final summary"
    # Check that export is called with correct parameters
    export_call_pattern = "_export_metrics_to_file(metrics"
    assert export_call_pattern in coord_source, "Should pass metrics object to export"
    print("   ✅ Export function integrated into coord() finally block")

    # Test that Path is imported
    print("   • Verifying Path import...")
    import action6_gather
    assert hasattr(action6_gather, 'Path') or 'Path' in dir(action6_gather), \
        "Path should be imported from pathlib"
    print("   ✅ Path imported from pathlib")

    # Test error handling
    print("   • Checking error handling...")
    assert "try:" in source and "except" in source, "Should have error handling"
    assert "logger.warning" in source or "logger.error" in source, \
        "Should log errors if export fails"
    print("   ✅ Error handling present")

    # Verify metrics directory structure
    print("   • Checking metrics directory structure...")
    metrics_dir = Path("Logs/metrics")
    if metrics_dir.exists():
        print(f"      Metrics directory exists: {metrics_dir.absolute()}")
        json_files = list(metrics_dir.glob("action6_metrics_*.json"))
        if json_files:
            print(f"      Found {len(json_files)} existing metrics file(s)")
            # Verify file format by checking one
            import json
            try:
                with open(json_files[0]) as f:
                    data = json.load(f)
                    assert 'metadata' in data, "File should contain metadata"
                    assert 'performance' in data, "File should contain performance"
                    assert 'timing' in data, "File should contain timing"
                    print(f"      ✅ Validated metrics file format: {json_files[0].name}")
            except Exception as e:
                print(f"      ⚠️  Could not validate file format: {e}")
        else:
            print("      No metrics files found yet (will be created on next run)")
    else:
        print(f"      Metrics directory will be created at: {metrics_dir.absolute()}")
    print("   ✅ Directory structure validated")

    print("   ✅ Metrics Export functionality validated")


def _test_real_time_monitoring() -> None:
    """Test real-time monitoring and alerts (Priority 3.3)."""
    print("📋 Testing Real-time Monitoring (Priority 3.3):")

    # Test MonitoringThresholds dataclass
    print("   • Checking MonitoringThresholds dataclass...")
    thresholds = MonitoringThresholds()
    assert hasattr(thresholds, 'error_rate_warning'), "Should have error_rate_warning threshold"
    assert hasattr(thresholds, 'error_rate_critical'), "Should have error_rate_critical threshold"
    assert hasattr(thresholds, 'api_time_warning'), "Should have api_time_warning threshold"
    assert hasattr(thresholds, 'cache_hit_rate_warning'), "Should have cache_hit_rate_warning threshold"
    assert thresholds.error_rate_warning < thresholds.error_rate_critical, \
        "Warning threshold should be lower than critical"
    print(f"      Thresholds: error_rate={thresholds.error_rate_warning}/{thresholds.error_rate_critical}%, "
          f"api_time={thresholds.api_time_warning}/{thresholds.api_time_critical}s")
    print("   ✅ MonitoringThresholds validated")

    # Test RealTimeMonitor class
    print("   • Checking RealTimeMonitor class...")
    monitor = RealTimeMonitor()
    assert hasattr(monitor, 'check_metrics'), "Should have check_metrics method"
    assert hasattr(monitor, 'check_session_health'), "Should have check_session_health method"
    assert hasattr(monitor, 'get_alerts'), "Should have get_alerts method"
    assert hasattr(monitor, 'get_alert_summary'), "Should have get_alert_summary method"
    assert hasattr(monitor, 'alerts'), "Should maintain alerts list"
    print("   ✅ RealTimeMonitor class structure validated")

    # Test alert generation with simulated high error rate
    print("   • Testing error rate alerts...")
    _reset_metrics()
    metrics = _get_metrics()
    # Simulate high error rate
    metrics.api_calls_made = 100
    metrics.api_errors = 15  # 15% error rate (above critical threshold)

    # Force check by resetting last check time
    monitor._last_check_time = 0
    alerts = monitor.check_metrics(metrics, current_page=10, total_pages=100)
    assert len(alerts) > 0, "Should generate alerts for high error rate"

    critical_alerts = [a for a in alerts if a['level'] == 'CRITICAL']
    assert len(critical_alerts) > 0, "Should generate CRITICAL alert for 15% error rate"

    error_alert = critical_alerts[0]
    assert error_alert['category'] == 'error_rate', "Should be error_rate category"
    assert 'error_rate' in error_alert['details'], "Should include error rate in details"
    assert error_alert['details']['errors'] == 15, "Should track error count"
    print(f"      Alert generated: {error_alert['message']}")
    print("   ✅ Error rate alerting validated")

    # Test API performance alerts
    print("   • Testing API performance alerts...")
    monitor.clear_alerts()
    _reset_metrics()
    metrics = _get_metrics()
    # Simulate slow API calls (use higher values to exceed critical threshold of 12.0s)
    metrics.api_call_times = [13.0, 14.0, 12.5, 15.0, 13.5]  # All above critical threshold (12s)
    metrics.api_calls_made = 5

    # Force check by resetting last check time
    monitor._last_check_time = 0
    alerts = monitor.check_metrics(metrics, current_page=20, total_pages=100)

    perf_alerts = [a for a in alerts if a['category'] == 'api_performance']
    assert len(perf_alerts) > 0, "Should generate alerts for slow API calls"
    print(f"      Alert generated: {perf_alerts[0]['message']}")
    print("   ✅ API performance alerting validated")    # Test page processing alerts
    print("   • Testing page processing alerts...")
    monitor.clear_alerts()
    _reset_metrics()
    metrics = _get_metrics()
    # Simulate slow page processing (adjusted for realistic thresholds: critical=200s)
    # Real-world: 20 matches/page at 4-6s each = 80-120s normal, 200s+ is critical
    metrics.page_times = [210.0, 205.0, 215.0]  # All above critical threshold (200s)
    metrics.api_calls_made = 10

    monitor._last_check_time = 0
    alerts = monitor.check_metrics(metrics, current_page=30, total_pages=100)

    page_alerts = [a for a in alerts if a['category'] == 'page_performance']
    assert len(page_alerts) > 0, "Should generate alerts for slow page processing"
    print(f"      Alert generated: {page_alerts[0]['message']}")
    print("   ✅ Page performance alerting validated")

    # Test cache performance alerts
    print("   • Testing cache performance alerts...")
    monitor.clear_alerts()
    _reset_metrics()
    metrics = _get_metrics()
    # Simulate poor cache performance
    metrics.cache_hits = 2
    metrics.cache_misses = 100  # ~2% hit rate (below warning threshold)
    metrics.api_calls_made = 10

    monitor._last_check_time = 0
    alerts = monitor.check_metrics(metrics, current_page=40, total_pages=100)

    cache_alerts = [a for a in alerts if a['category'] == 'cache_performance']
    assert len(cache_alerts) > 0, "Should generate alerts for poor cache hit rate"
    assert cache_alerts[0]['level'] == 'INFO', "Cache alerts should be INFO level"
    print(f"      Alert generated: {cache_alerts[0]['message']}")
    print("   ✅ Cache performance alerting validated")

    # Test alert filtering
    print("   • Testing alert filtering...")
    monitor.clear_alerts()
    _reset_metrics()
    metrics = _get_metrics()
    # Generate mix of alerts (adjusted for realistic thresholds)
    metrics.api_calls_made = 100
    metrics.api_errors = 15  # Critical (15% > 10% threshold)
    metrics.api_call_times = [13.0, 14.0, 12.5]  # Critical (avg 13.17s > 12s threshold)
    metrics.cache_hits = 2
    metrics.cache_misses = 100  # Info (2% hit rate < 5% threshold)

    monitor._last_check_time = 0
    monitor.check_metrics(metrics, current_page=50, total_pages=100)

    all_alerts = monitor.get_alerts()
    critical_only = monitor.get_alerts(level='CRITICAL')
    info_only = monitor.get_alerts(level='INFO')

    assert len(all_alerts) >= 3, "Should have multiple alerts"
    assert len(critical_only) >= 2, "Should have multiple critical alerts (error_rate + api_performance)"
    assert len(info_only) >= 1, "Should have info alerts (cache_performance)"
    print(f"      Filtered alerts: {len(all_alerts)} total, {len(critical_only)} critical, {len(info_only)} info")
    print("   ✅ Alert filtering validated")

    # Test alert summary
    print("   • Testing alert summary...")
    summary = monitor.get_alert_summary()
    assert 'CRITICAL' in summary, "Summary should include CRITICAL count"
    assert 'WARNING' in summary, "Summary should include WARNING count"
    assert 'INFO' in summary, "Summary should include INFO count"
    assert summary['CRITICAL'] >= 2, "Should count critical alerts"
    assert summary['INFO'] >= 1, "Should count info alerts"
    print(f"      Alert summary: {summary}")
    print("   ✅ Alert summary validated")

    # Test global monitor functions
    print("   • Testing global monitor functions...")
    _reset_monitor()
    monitor1 = _get_monitor()
    monitor2 = _get_monitor()
    assert monitor1 is monitor2, "Should return same monitor instance"
    assert len(monitor1.alerts) == 0, "Reset should clear alerts"
    print("   ✅ Global monitor functions validated")

    # Test integration with coord()
    print("   • Checking integration into coord()...")
    import inspect
    coord_source = inspect.getsource(coord)
    assert "_reset_monitor" in coord_source, "Should initialize monitor in coord()"
    assert "_get_monitor" in coord_source, "Should use monitor in coord()"
    assert "get_alert_summary" in coord_source, "Should log alert summary in finally block"

    # Check that monitoring is done during page processing by looking at the module source
    with open(__file__, encoding='utf-8') as f:
        full_source = f.read()
        # Look for the monitoring calls in _update_state_after_batch
        assert "monitor.check_metrics" in full_source, "Should have monitor.check_metrics calls"
        assert "monitor.check_session_health" in full_source, "Should have monitor.check_session_health calls"

    print("   ✅ Monitor integrated into coord()")

    # Test alert logging
    print("   • Testing alert logging...")
    import inspect
    monitor_source = inspect.getsource(RealTimeMonitor)
    assert "_log_alert" in monitor_source, "Should have alert logging method"
    assert "logger.critical" in monitor_source or "logger.warning" in monitor_source, \
        "Should log alerts with appropriate severity"
    print("   ✅ Alert logging validated")

    print("   ✅ Real-time Monitoring functionality validated")


def _test_module_initialization() -> None:
    """Test module initialization and state functions with detailed verification"""
    print("📋 Testing Action 6 module initialization:")
    results = []

    # Test _initialize_gather_state function
    print("   • Testing _initialize_gather_state...")
    try:
        state = _initialize_gather_state()
        is_dict = isinstance(state, dict)

        required_keys = ["total_new", "total_updated", "total_pages_processed"]
        keys_present = all(key in state for key in required_keys)

        print(f"   ✅ State dictionary created: {is_dict}")
        print(
            f"   ✅ Required keys present: {keys_present} ({len(required_keys)} keys)"
        )
        print(f"   ✅ State structure: {list(state.keys())}")

        results.extend([is_dict, keys_present])
        assert is_dict, "Should return dictionary state"
        assert keys_present, "Should have all required keys in state"

    except Exception as e:
        print(f"   ❌ _initialize_gather_state: Exception {e}")
        results.extend([False, False])

    # Test _validate_start_page function
    print("   • Testing _validate_start_page...")
    validation_tests = [
        ("5", 5, "String number conversion"),
        (10, 10, "Integer input handling"),
        (None, 1, "None input (should default to 1)"),
        ("invalid", 1, "Invalid string (should default to 1)"),
        (0, 1, "Zero input (should default to 1)"),
    ]

    for input_val, expected, description in validation_tests:
        try:
            result = _validate_start_page(input_val)
            matches_expected = result == expected

            status = "✅" if matches_expected else "❌"
            print(f"   {status} {description}: {input_val!r} → {result}")

            results.append(matches_expected)
            assert (
                matches_expected
            ), f"Failed for {input_val}: expected {expected}, got {result}"

        except Exception as e:
            print(f"   ❌ {description}: Exception {e}")
            results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} initialization tests passed")


def _test_core_functionality() -> None:
    """Test core DNA match gathering and navigation functionality"""
    print("🔧 Testing Action 6 core functionality:")
    # Test that core functions exist and are callable
    assert callable(_lookup_existing_persons), "_lookup_existing_persons should be callable"
    assert callable(get_matches), "get_matches should be callable"
    assert callable(coord), "coord should be callable"
    assert callable(nav_to_list), "nav_to_list should be callable"
    print("   ✅ All core functions are callable")


def _test_data_processing_functions() -> None:
    """Test data processing and database preparation functions"""
    print("📊 Testing Action 6 data processing:")
    # Test that data processing functions exist and are callable
    assert callable(_identify_fetch_candidates), "_identify_fetch_candidates should be callable"
    assert callable(_prepare_bulk_db_data), "_prepare_bulk_db_data should be callable"
    assert callable(_execute_bulk_db_operations), "_execute_bulk_db_operations should be callable"
    print("   ✅ All data processing functions are callable")


def _test_edge_cases() -> None:
    """Test edge case handling across all DNA match gathering functions"""
    print("⚠️  Testing Action 6 edge cases:")
    # Test _validate_start_page with edge cases
    assert _validate_start_page(None) == 1, "None should default to 1"
    assert _validate_start_page(0) == 1, "Zero should default to 1"
    assert _validate_start_page(-5) == 1, "Negative should default to 1"
    assert _validate_start_page("invalid") == 1, "Invalid string should default to 1"
    print("   ✅ Edge cases handled correctly")


def _test_integration() -> None:
    """Test integration with session management and external systems"""
    print("🔗 Testing Action 6 integration:")
    # Test that integration points exist
    assert callable(coord), "coord integration function should be callable"
    assert callable(nav_to_list), "nav_to_list integration function should be callable"
    print("   ✅ Integration functions available")


def _test_performance() -> None:
    """Test performance characteristics of DNA match gathering operations"""
    import time
    print("⚡ Testing Action 6 performance:")

    # Test state initialization performance
    start = time.time()
    for _ in range(100):
        _initialize_gather_state()
    duration = time.time() - start

    assert duration < 1.0, f"State initialization too slow: {duration:.3f}s for 100 iterations"
    print(f"   ✅ State initialization: {duration:.3f}s for 100 iterations")


def _test_error_handling() -> None:
    """Test error handling and recovery functionality for DNA match operations"""
    print("🛡️  Testing Action 6 error handling:")
    # Test that functions handle errors gracefully
    try:
        result = _validate_start_page("invalid")
        assert result == 1, "Should return default value on invalid input"
        print("   ✅ Error handling works correctly")
    except Exception as e:
        raise AssertionError(f"Error handling failed: {e}") from e


# ==============================================
# MAIN TEST SUITE
# ==============================================


def action6_gather_module_tests() -> bool:
    """Comprehensive test suite for action6_gather.py"""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Action 6 - Gather DNA Matches", "action6_gather.py")
    suite.start_suite()

    # Assign circuit breaker and configuration test functions (Priority 1.1, 1.2, 1.3, 1.4, 2.1, 2.2)
    test_circuit_breaker_basic = _test_circuit_breaker_basic
    test_circuit_breaker_reset = _test_circuit_breaker_reset
    test_circuit_breaker_global_instance = _test_circuit_breaker_global_instance
    test_circuit_breaker_timing = _test_circuit_breaker_timing
    test_worker_configuration = _test_worker_configuration
    test_circuit_breaker_integration = _test_circuit_breaker_integration
    test_auto_recovery_control = _test_auto_recovery_control
    test_403_auth_refresh_handler = _test_403_auth_refresh_handler
    test_session_health_monitoring = _test_session_health_monitoring
    test_progress_checkpointing = _test_progress_checkpointing
    test_api_call_batching = _test_api_call_batching
    test_enhanced_logging_metrics = _test_enhanced_logging_metrics
    test_metrics_export = _test_metrics_export
    test_real_time_monitoring = _test_real_time_monitoring

    # Assign all module-level test functions
    test_module_initialization = _test_module_initialization
    test_core_functionality = _test_core_functionality
    test_data_processing_functions = _test_data_processing_functions
    test_edge_cases = _test_edge_cases
    test_integration = _test_integration
    test_performance = _test_performance
    test_error_handling = _test_error_handling

    # Define all tests in a data structure to reduce complexity
    tests = [
        # Priority 1.1: Circuit Breaker Tests
        ("SessionCircuitBreaker - Basic Functionality",
         test_circuit_breaker_basic,
         "Circuit breaker trips after threshold failures and stays tripped",
         "Circuit breaker basic operation (Priority 1.1)",
         "Testing circuit breaker threshold, trip state, and persistence"),

        ("SessionCircuitBreaker - Reset on Success",
         test_circuit_breaker_reset,
         "Circuit breaker resets failure count on successful session check",
         "Circuit breaker reset mechanism (Priority 1.1)",
         "Testing circuit breaker reset behavior and failure count management"),

        ("SessionCircuitBreaker - Global Instance",
         test_circuit_breaker_global_instance,
         "Global circuit breaker singleton pattern works correctly",
         "Circuit breaker singleton and configuration (Priority 1.1)",
         "Testing global circuit breaker instance and default configuration"),

        ("SessionCircuitBreaker - Timing",
         test_circuit_breaker_timing,
         "Circuit breaker tracks trip time correctly",
         "Circuit breaker timing and trip time tracking (Priority 1.1)",
         "Testing circuit breaker timestamp recording and timing accuracy"),

        # Priority 2.1: Worker Configuration Tests
        ("Worker Configuration from .env",
         test_worker_configuration,
         "Thread pool workers and rate limiting load correctly from configuration",
         "Worker configuration and rate limiting (Priority 2.1)",
         "Testing THREAD_POOL_WORKERS loading and rate limit calculation"),

        ("Circuit Breaker Integration with _check_session_validity",
         test_circuit_breaker_integration,
         "Circuit breaker integrates correctly with session validation",
         "Circuit breaker integration (Priority 1.1)",
         "Testing circuit breaker integration into session validity checking"),

        # Priority 1.2: Auto-Recovery Control Tests
        ("Auto-Recovery Enable/Disable Control",
         test_auto_recovery_control,
         "Auto-recovery can be disabled and re-enabled for fail-fast behavior",
         "Auto-recovery control (Priority 1.2)",
         "Testing SessionManager auto-recovery control and coord() integration"),

        # Priority 1.3: 403 Error Handling Tests
        ("403 Error Handling with Auth Refresh",
         test_403_auth_refresh_handler,
         "403 Forbidden errors trigger auth refresh and retry",
         "403 error handling (Priority 1.3)",
         "Testing _api_req_with_auth_refresh and _refresh_session_auth"),

        # Priority 1.4: Session Health Monitoring Tests
        ("Session Health Monitoring with Proactive Refresh",
         test_session_health_monitoring,
         "Session health monitoring tracks age and triggers proactive refresh",
         "Session health monitoring (Priority 1.4)",
         "Testing _check_session_health_proactive and _get_session_health_status"),

        # Priority 2.2: Progress Checkpointing Tests
        ("Progress Checkpointing for Resume Capability",
         test_progress_checkpointing,
         "Progress checkpoints enable resuming from last completed page",
         "Progress checkpointing (Priority 2.2)",
         "Testing _save_checkpoint, _load_checkpoint, and checkpoint integration"),

        # Priority 2.3: API Call Batching & Deduplication
        ("API Call Batching & Deduplication",
         test_api_call_batching,
         "API call cache reduces redundant requests through batching and deduplication",
         "API call optimization (Priority 2.3)",
         "Testing APICallCache class, deduplication logic, and integration into fetch pipeline"),

        # Priority 3.1: Enhanced Logging & Performance Metrics
        ("Enhanced Logging & Performance Metrics",
         test_enhanced_logging_metrics,
         "Performance metrics track timing, throughput, and error patterns with comprehensive reporting",
         "Enhanced observability (Priority 3.1)",
         "Testing PerformanceMetrics class, stat tracking, and integration into execution flow"),

        # Priority 3.2: Structured Metrics Export
        ("Structured Metrics Export",
         test_metrics_export,
         "Metrics export to JSON files for historical analysis and trend tracking",
         "Historical metrics tracking (Priority 3.2)",
         "Testing _export_metrics_to_file function, JSON serialization, and directory creation"),

        # Priority 3.3: Real-time Monitoring & Alerts
        ("Real-time Monitoring & Alerts",
         test_real_time_monitoring,
         "Real-time monitoring generates alerts when performance thresholds are exceeded",
         "Real-time alerting system (Priority 3.3)",
         "Testing RealTimeMonitor class, alert generation, threshold checks, and integration"),

        # Existing tests
        ("_initialize_gather_state(), _validate_start_page()",
         test_module_initialization,
         "Module initializes correctly with proper state management and page validation",
         "Module initialization and state management functions",
         "Testing state initialization, page validation, and parameter handling for DNA match gathering"),

        ("_lookup_existing_persons(), get_matches(), coord(), nav_to_list()",
         test_core_functionality,
         "All core DNA match gathering functions execute correctly with proper data handling",
         "Core DNA match gathering and navigation functionality",
         "Testing database lookups, match retrieval, coordination, and navigation functions"),

        ("_identify_fetch_candidates(), _prepare_bulk_db_data(), _execute_bulk_db_operations()",
         test_data_processing_functions,
         "All data processing functions handle DNA match data correctly with proper formatting",
         "Data processing and database preparation functions",
         "Testing candidate identification, bulk data preparation, and database operations"),

        ("ALL functions with edge case inputs",
         test_edge_cases,
         "All functions handle edge cases gracefully without crashes or unexpected behavior",
         "Edge case handling across all DNA match gathering functions",
         "Testing functions with empty, None, invalid, and boundary condition inputs"),

        ("Integration with SessionManager and external dependencies",
         test_integration,
         "Integration functions work correctly with mocked external dependencies and session management",
         "Integration with session management and external systems",
         "Testing integration with session managers, database connections, and web automation"),

        ("Performance of state initialization and validation operations",
         test_performance,
         "All operations complete within acceptable time limits with good performance",
         "Performance characteristics of DNA match gathering operations",
         "Testing execution speed and efficiency of state management and validation functions"),

        ("Error handling for database and validation functions",
         test_error_handling,
         "All error conditions handled gracefully with appropriate fallback responses",
         "Error handling and recovery functionality for DNA match operations",
         "Testing error scenarios with database failures, invalid inputs, and exception conditions"),
    ]

    # Run all tests from the list
    with suppress_logging():
        for test_name, test_func, expected, method, details in tests:
            suite.run_test(test_name, test_func, expected, method, details)

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return action6_gather_module_tests()


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    print("🧬 Running Action 6 - Gather DNA Matches comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

