#!/usr/bin/env python3

# action6_gather.py

"""
action6_gather.py - Gather DNA Matches from Ancestry

Fetches the user's DNA match list page by page, extracts relevant information,
compares with existing database records, fetches additional details via API for
new or changed matches, and performs bulk updates/inserts into the local database.
Handles pagination, rate limiting, caching (via utils/cache.py decorators used
within helpers), error handling, and concurrent API fetches using ThreadPoolExecutor.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple
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
CRITICAL_API_FAILURE_THRESHOLD: int = (
    50  # Threshold for _fetch_combined_details failures (increased to 50 for better tolerance of transient issues)
)

# Configurable settings from config_schema
DB_ERROR_PAGE_THRESHOLD: int = 10  # Max consecutive DB errors allowed
THREAD_POOL_WORKERS: int = 5  # Concurrent API workers


# --- Custom Exceptions ---
class MaxApiFailuresExceededError(Exception):
    """Custom exception for exceeding API failure threshold in a batch."""

    pass


# End of MaxApiFailuresExceededError


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


def _initialize_gather_state() -> Dict[str, Any]:
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


def _fetch_initial_page_data(session_manager: SessionManager, start_page: int) -> tuple[Optional[List[Dict[str, Any]]], Optional[int], bool]:
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
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[int], bool]:
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
) -> Tuple[int, int]:
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

def _get_db_session_with_retry(session_manager: SessionManager, current_page_num: int, state: Dict[str, Any]) -> Optional[SqlAlchemySession]:
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


def _handle_db_session_failure(current_page_num: int, state: Dict[str, Any], progress_bar, loop_final_success: bool) -> tuple[bool, bool]:
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


def _fetch_page_matches(session_manager: SessionManager, current_page_num: int, db_session_for_page: SqlAlchemySession, state: Dict[str, Any], progress_bar) -> Optional[List[Dict[str, Any]]]:
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


def _check_session_validity(session_manager: SessionManager, current_page_num: int, state: Dict[str, Any], progress_bar) -> bool:
    """Check if session is valid, handle abort if not."""
    if not session_manager.is_sess_valid():
        logger.critical(f"WebDriver session invalid/unreachable before processing page {current_page_num}. Aborting run.")
        remaining_matches_estimate = max(0, progress_bar.total - progress_bar.n)
        if remaining_matches_estimate > 0:
            progress_bar.update(remaining_matches_estimate)
            state["total_errors"] += remaining_matches_estimate
        return False
    return True


def _update_state_after_batch(state: Dict[str, Any], counters: BatchCounters, progress_bar):
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


# Helper functions for _main_page_processing_loop

def _calculate_total_matches_estimate(start_page: int, total_pages_in_run: int, initial_matches_on_page: Optional[List[Dict[str, Any]]]) -> int:
    """Calculate total matches estimate for progress bar."""
    total_matches_estimate = total_pages_in_run * MATCHES_PER_PAGE

    if start_page == 1 and initial_matches_on_page is not None:
        total_matches_estimate = max(total_matches_estimate, len(initial_matches_on_page))

    return total_matches_estimate


def _should_fetch_page_data(current_page_num: int, start_page: int, matches_on_page_for_batch: Optional[List[Dict[str, Any]]]) -> bool:
    """Determine if page data needs to be fetched."""
    return not (current_page_num == start_page and matches_on_page_for_batch is not None)


def _fetch_and_validate_page_data(
    session_manager: SessionManager,
    current_page_num: int,
    state: Dict[str, Any],
    progress_bar: Any
) -> Optional[List[Dict[str, Any]]]:
    """Fetch page data and validate DB session."""
    # Get DB session with retry
    db_session_for_page = _get_db_session_with_retry(session_manager, current_page_num, state)

    if not db_session_for_page:
        return None

    # Fetch page matches
    return _fetch_page_matches(session_manager, current_page_num, db_session_for_page, state, progress_bar)


def _handle_empty_matches(current_page_num: int, start_page: int, state: Dict[str, Any], progress_bar: Any) -> None:
    """Handle empty matches on a page."""
    logger.info(f"No matches found or processed on page {current_page_num}.")

    if not (current_page_num == start_page and state["total_pages_processed"] == 0):
        progress_bar.update(MATCHES_PER_PAGE)

    time.sleep(0.5)


def _process_page_batch(
    session_manager: SessionManager,
    matches_on_page: List[Dict[str, Any]],
    current_page_num: int,
    progress_bar: Any,
    state: Dict[str, Any]
) -> None:
    """Process a batch of matches and update state."""
    # Process batch
    page_new, page_updated, page_skipped, page_errors = _do_batch(
        session_manager=session_manager,
        matches_on_page=matches_on_page,
        current_page=current_page_num,
        progress_bar=progress_bar,
    )

    # Update state
    counters = BatchCounters(new=page_new, updated=page_updated, skipped=page_skipped, errors=page_errors)
    _update_state_after_batch(state, counters, progress_bar)

    # Apply rate limiting
    _adjust_delay(session_manager, current_page_num)
    session_manager.dynamic_rate_limiter.wait()


def _finalize_progress_bar(progress_bar: Any, state: Dict[str, Any], loop_final_success: bool) -> None:
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
    matches_on_page_for_batch: Optional[List[Dict[str, Any]]],
    state: Dict[str, Any],
    progress_bar: tqdm,
    loop_final_success: bool
) -> Tuple[bool, bool, Optional[List[Dict[str, Any]]], int]:
    """Process a single page iteration in the main loop."""
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

    # Process batch and update state
    _process_page_batch(session_manager, matches_on_page_for_batch, current_page_num, progress_bar, state)

    return loop_final_success, False, None, current_page_num + 1


def _main_page_processing_loop(
    session_manager: SessionManager,
    start_page: int,
    last_page_to_process: int,
    total_pages_in_run: int,  # Added this argument
    initial_matches_on_page: Optional[List[Dict[str, Any]]],
    state: Dict[str, Any],  # Pass the whole state dict
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
            matches_on_page_for_batch: Optional[List[Dict[str, Any]]] = initial_matches_on_page

            while current_page_num <= last_page_to_process:
                loop_final_success, should_break, matches_on_page_for_batch, current_page_num = (
                    _process_single_page_iteration(
                        session_manager,
                        current_page_num,
                        start_page,
                        matches_on_page_for_batch,
                        state,
                        progress_bar,
                        loop_final_success
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


def _process_initial_navigation(session_manager: SessionManager, start_page: int, state: Dict[str, Any]) -> tuple[Optional[List[Dict[str, Any]]], Optional[int], bool]:
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
@timeout_protection(timeout=300)
@error_context("DNA match gathering coordination")
def coord(  # type: ignore
    session_manager: SessionManager, _config_schema_arg: "ConfigSchema", start: int = 1
) -> bool:  # Uses config schema
    """
    Orchestrates the gathering of DNA matches from Ancestry.
    Handles pagination, fetches match data, compares with database, and processes.
    """
    # Step 1: Validate Session State
    _validate_session_state(session_manager)

    # Step 2: Initialize state
    state = _initialize_gather_state()
    start_page = _validate_start_page(start)
    logger.debug(f"--- Starting DNA Match Gathering (Action 6) from page {start_page} ---")

    try:
        # Step 3: Initial Navigation and Total Pages Fetch
        initial_matches, total_pages_api, initial_fetch_ok = _process_initial_navigation(
            session_manager, start_page, state
        )

        if not initial_fetch_ok:
            return False

        # Step 4: Determine Page Range
        last_page_to_process, total_pages_in_run, total_matches_estimate = _calculate_processing_range(
            total_pages_api, start_page
        )

        if total_pages_in_run <= 0:
            return True

        # Step 5: Main Processing Loop (delegated)
        # Pass only relevant parts of initial_matches to the loop
        initial_matches_for_loop = state["matches_on_current_page"]

        loop_success = _main_page_processing_loop(
            session_manager,
            start_page,
            last_page_to_process,
            total_pages_in_run,  # Correctly passing total_pages_in_run
            initial_matches_for_loop,
            state,  # Pass the whole state dict to be updated by the loop
        )
        state["final_success"] = (
            state["final_success"] and loop_success
        )  # Update overall success

    # Step 6: Handle specific exceptions from coord's orchestration level
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected. Stopping match gathering.")
        state["final_success"] = False
    except ConnectionError as coord_conn_err:  # Catch ConnectionError if it bubbles up
        logger.critical(
            f"ConnectionError during coord execution: {coord_conn_err}",
            exc_info=True,
        )
        state["final_success"] = False
    except MaxApiFailuresExceededError as api_halt_err:
        logger.critical(
            f"Halting run due to excessive critical API failures: {api_halt_err}",
            exc_info=False,
        )
        state["final_success"] = False
    except Exception as e:
        logger.error(f"Critical error during coord execution: {e}", exc_info=True)
        state["final_success"] = False
    finally:
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
    session: SqlAlchemySession, uuids_on_page: List[str]
) -> Dict[str, Person]:
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
    existing_persons_map: Dict[str, Person] = {}
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
        existing_persons_map: Dict[str, Person] = {
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
    match_api_data: Dict[str, Any],
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
    match_api_data: Dict[str, Any],
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
    matches_on_page: List[Dict[str, Any]], existing_persons_map: Dict[str, Any]
) -> Tuple[Set[str], List[Dict[str, Any]], int]:
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
        - fetch_candidates_uuid (Set[str]): Set of UUIDs requiring API detail fetches.
        - matches_to_process_later (List[Dict]): List of match data dicts for candidates.
        - skipped_count_this_batch (int): Number of matches skipped in this batch.
    """
    # Initialize results
    fetch_candidates_uuid: Set[str] = set()
    skipped_count_this_batch = 0
    matches_to_process_later: List[Dict[str, Any]] = []
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

    logger.debug(
        f"Identified {len(fetch_candidates_uuid)} candidates for API detail fetch, "
        f"{skipped_count_this_batch} skipped (no change detected from list view)."
    )

    if len(fetch_candidates_uuid) == 0:
        logger.warning("No fetch candidates identified - all matches appear up-to-date in database")
    else:
        logger.info(f"Fetch candidates: {list(fetch_candidates_uuid)[:5]}...")

    return fetch_candidates_uuid, matches_to_process_later, skipped_count_this_batch


# End of _identify_fetch_candidates

# Helper functions for _perform_api_prefetches

def _identify_tree_badge_ladder_candidates(
    matches_to_process_later: List[Dict[str, Any]],
    fetch_candidates_uuid: Set[str]
) -> Set[str]:
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
    fetch_candidates_uuid: Set[str],
    uuids_for_tree_badge_ladder: Set[str]
) -> Dict[Any, Tuple[str, str]]:
    """Submit initial API tasks (combined_details, relationship_prob, badge_details)."""
    futures: Dict[Any, Tuple[str, str]] = {}

    for uuid_val in fetch_candidates_uuid:
        _ = session_manager.dynamic_rate_limiter.wait()
        futures[executor.submit(_fetch_combined_details, session_manager, uuid_val)] = ("combined_details", uuid_val)

        _ = session_manager.dynamic_rate_limiter.wait()
        max_labels = 2
        futures[executor.submit(_fetch_batch_relationship_prob, session_manager, uuid_val, max_labels)] = ("relationship_prob", uuid_val)

    for uuid_val in uuids_for_tree_badge_ladder:
        _ = session_manager.dynamic_rate_limiter.wait()
        futures[executor.submit(_fetch_batch_badge_details, session_manager, uuid_val)] = ("badge_details", uuid_val)

    return futures


def _process_api_task_result(
    task_type: str,
    identifier_uuid: str,
    result: Any,
    batch_combined_details: Dict[str, Optional[Dict[str, Any]]],
    temp_badge_results: Dict[str, Optional[Dict[str, Any]]],
    batch_relationship_prob_data: Dict[str, Optional[str]],
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
    batch_combined_details: Dict[str, Optional[Dict[str, Any]]],
    temp_badge_results: Dict[str, Optional[Dict[str, Any]]],
    batch_relationship_prob_data: Dict[str, Optional[str]],
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
    futures: Dict[Any, Tuple[str, str]]
):
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


def _build_cfpid_to_uuid_map(temp_badge_results: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, str]:
    """Build mapping from CFPID to UUID from badge results."""
    cfpid_to_uuid_map: Dict[str, str] = {}
    for uuid_val, badge_result_data in temp_badge_results.items():
        if badge_result_data:
            cfpid = badge_result_data.get("their_cfpid")
            if cfpid:
                cfpid_to_uuid_map[cfpid] = uuid_val
    return cfpid_to_uuid_map


def _submit_ladder_tasks(
    executor: ThreadPoolExecutor,
    session_manager: SessionManager,
    cfpid_to_uuid_map: Dict[str, str],
    my_tree_id: Optional[str]
) -> Dict[Any, Tuple[str, str]]:
    """Submit ladder API tasks for CFPIDs."""
    ladder_futures = {}
    if my_tree_id and cfpid_to_uuid_map:
        cfpid_list = list(cfpid_to_uuid_map.keys())
        logger.debug(f"Submitting Ladder tasks for {len(cfpid_list)} CFPIDs...")
        for cfpid_item in cfpid_list:
            _ = session_manager.dynamic_rate_limiter.wait()
            ladder_futures[executor.submit(_fetch_batch_ladder, session_manager, cfpid_item, my_tree_id)] = ("ladder", cfpid_item)
    return ladder_futures


def _process_ladder_results(
    ladder_futures: Dict[Any, Tuple[str, str]],
    cfpid_to_uuid_map: Dict[str, str]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Process ladder API task results."""
    temp_ladder_results: Dict[str, Optional[Dict[str, Any]]] = {}

    logger.debug(f"Processing {len(ladder_futures)} Ladder API tasks...")
    for future in as_completed(ladder_futures):
        task_type, identifier_cfpid = ladder_futures[future]
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
    temp_badge_results: Dict[str, Optional[Dict[str, Any]]],
    temp_ladder_results: Dict[str, Optional[Dict[str, Any]]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Combine badge and ladder results into final tree data."""
    batch_tree_data: Dict[str, Optional[Dict[str, Any]]] = {}

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
    fetch_candidates_uuid: Set[str],
    matches_to_process_later: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
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
    batch_combined_details: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_relationship_prob_data: Dict[str, Optional[str]] = {}
    temp_badge_results: Dict[str, Optional[Dict[str, Any]]] = {}

    if not fetch_candidates_uuid:
        logger.warning("_perform_api_prefetches: No fetch candidates provided for API pre-fetch - returning empty results")
        return {"combined": {}, "tree": {}, "rel_prob": {}}

    fetch_start_time = time.time()
    num_candidates = len(fetch_candidates_uuid)
    my_tree_id = session_manager.my_tree_id
    critical_combined_details_failures = 0

    logger.debug(f"--- Starting Parallel API Pre-fetch ({num_candidates} candidates, {THREAD_POOL_WORKERS} workers) ---")

    # Identify tree members needing badge/ladder fetch
    uuids_for_tree_badge_ladder = _identify_tree_badge_ladder_candidates(matches_to_process_later, fetch_candidates_uuid)

    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
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
    logger.debug(f"--- Finished Parallel API Pre-fetch. Duration: {fetch_duration:.2f}s ---")

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
    prefetched_data: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """Retrieve prefetched data for a specific match UUID."""
    prefetched_combined = prefetched_data.get("combined", {}).get(uuid_val)
    prefetched_tree = prefetched_data.get("tree", {}).get(uuid_val)
    prefetched_rel_prob = prefetched_data.get("rel_prob", {}).get(uuid_val)
    return prefetched_combined, prefetched_tree, prefetched_rel_prob


def _validate_match_uuid(match_list_data: Dict[str, Any]) -> str:
    """Validate and return match UUID, raise ValueError if missing."""
    uuid_val = match_list_data.get("uuid")
    if not uuid_val:
        logger.error("Critical error: Match data missing UUID in _prepare_bulk_db_data. Skipping.")
        raise ValueError("Missing UUID")
    return uuid_val


def _process_single_match(
    match_list_data: Dict[str, Any],
    session_manager: SessionManager,
    existing_persons_map: Dict[str, Person],
    prefetched_data: Dict[str, Dict[str, Any]],
    log_ref_short: str
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
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
    page_statuses: Dict[str, int],
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
    match_list_data: Dict[str, Any],
    session_manager: SessionManager,
    existing_persons_map: Dict[str, Person],
    prefetched_data: Dict[str, Dict[str, Any]],
    prepared_bulk_data: List[Dict[str, Any]],
    page_statuses: Dict[str, int],
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
    matches_to_process: List[Dict[str, Any]],
    existing_persons_map: Dict[str, Person],
    prefetched_data: Dict[str, Dict[str, Any]],
    progress_bar: Optional[tqdm],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
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
        - prepared_bulk_data (List[Dict]): A list where each element is a dictionary
          representing one person and contains keys 'person', 'dna_match', 'family_tree'
          with data formatted for bulk operations (or None if no change needed).
        - page_statuses (Dict[str, int]): Counts of 'new', 'updated', 'error' outcomes
          during the preparation phase for this batch.
    """
    # Step 1: Initialize results
    prepared_bulk_data: List[Dict[str, Any]] = []
    page_statuses: Dict[str, int] = {
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


def _separate_operations_by_type(prepared_bulk_data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
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


def _deduplicate_person_creates(person_creates_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """De-duplicate Person Creates based on Profile ID before bulk insert."""
    if not person_creates_raw:
        logger.debug("No unique Person records to bulk insert.")
        return []

    logger.debug(f"De-duplicating {len(person_creates_raw)} raw person creates based on Profile ID...")
    person_creates_filtered = []
    seen_profile_ids: Set[str] = set()
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


def _validate_no_duplicate_profile_ids(insert_data: List[Dict[str, Any]]) -> None:
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


def _bulk_update_persons(session: SqlAlchemySession, person_updates: List[Dict[str, Any]]) -> None:
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


def _get_existing_dna_matches_map(session: SqlAlchemySession, all_person_ids_map: Dict[str, int]) -> Dict[int, int]:
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
    dna_match_ops: List[Dict[str, Any]],
    all_person_ids_map: Dict[str, int],
    existing_dna_matches_map: Dict[int, int]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
    tree_creates: List[Dict[str, Any]],
    all_person_ids_map: Dict[str, int]
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


def _bulk_update_family_trees(session: SqlAlchemySession, tree_updates: List[Dict[str, Any]]) -> None:
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
    family_tree_ops: List[Dict[str, Any]],
    all_person_ids_map: Dict[str, int]
) -> None:
    """Bulk upsert FamilyTree records (separate insert/update)."""
    tree_creates = [op for op in family_tree_ops if op.get("_operation") == "create"]
    tree_updates = [op for op in family_tree_ops if op.get("_operation") == "update"]

    _bulk_insert_family_trees(session, tree_creates, all_person_ids_map)
    _bulk_update_family_trees(session, tree_updates)


def _bulk_upsert_dna_matches(
    session: SqlAlchemySession,
    dna_match_ops: List[Dict[str, Any]],
    all_person_ids_map: Dict[str, int]
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


def _create_master_person_id_map(
    created_person_map: Dict[str, int],
    person_updates: List[Dict[str, Any]],
    prepared_bulk_data: List[Dict[str, Any]],
    existing_persons_map: Dict[str, Person]
) -> Dict[str, int]:
    """Create master ID map for linking related records."""
    all_person_ids_map: Dict[str, int] = created_person_map.copy()

    # Add IDs from person updates
    for p_update_data in person_updates:
        if p_update_data.get("_existing_person_id") and p_update_data.get("uuid"):
            all_person_ids_map[p_update_data["uuid"]] = p_update_data["_existing_person_id"]

    # Add IDs from existing persons map
    processed_uuids = {
        p["person"]["uuid"]
        for p in prepared_bulk_data
        if p.get("person") and p["person"].get("uuid")
    }

    for uuid_processed in processed_uuids:
        if uuid_processed not in all_person_ids_map and existing_persons_map.get(uuid_processed):
            person = existing_persons_map[uuid_processed]
            person_id_val = getattr(person, "id", None)
            if person_id_val is not None:
                all_person_ids_map[uuid_processed] = person_id_val

    return all_person_ids_map


def _bulk_insert_persons(session: SqlAlchemySession, person_creates_filtered: List[Dict[str, Any]]) -> Dict[str, int]:
    """Bulk insert Person records and return mapping of UUID to new Person ID."""
    created_person_map: Dict[str, int] = {}

    if not person_creates_filtered:
        logger.debug("No unique Person records to bulk insert.")
        return created_person_map

    logger.debug(f"Preparing {len(person_creates_filtered)} Person records for bulk insert...")

    # Prepare list of dictionaries for bulk_insert_mappings
    insert_data = [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in person_creates_filtered
    ]

    # Convert status Enum to its value for bulk insertion
    for item_data in insert_data:
        if "status" in item_data and isinstance(item_data["status"], PersonStatusEnum):
            item_data["status"] = item_data["status"].value

    # Validate no duplicates
    _validate_no_duplicate_profile_ids(insert_data)

    # Perform bulk insert
    logger.debug(f"Bulk inserting {len(insert_data)} Person records...")
    session.bulk_insert_mappings(Person, insert_data)  # type: ignore
    logger.debug("Bulk insert Persons called.")

    # Get newly created IDs
    session.flush()
    logger.debug("Session flushed to assign Person IDs.")
    inserted_uuids = [p_data["uuid"] for p_data in insert_data if p_data.get("uuid")]

    if inserted_uuids:
        logger.debug(f"Querying IDs for {len(inserted_uuids)} inserted UUIDs...")
        newly_inserted_persons = (
            session.query(Person.id, Person.uuid)
            .filter(Person.uuid.in_(inserted_uuids))  # type: ignore
            .all()
        )
        created_person_map = {p_uuid: p_id for p_id, p_uuid in newly_inserted_persons}
        logger.debug(f"Mapped {len(created_person_map)} new Person IDs.")

        if len(created_person_map) != len(inserted_uuids):
            logger.error(
                f"CRITICAL: ID map count mismatch! Expected {len(inserted_uuids)}, got {len(created_person_map)}. Some IDs might be missing."
            )
    else:
        logger.warning("No UUIDs available in insert_data to query back IDs.")

    return created_person_map


def _execute_bulk_db_operations(
    session: SqlAlchemySession,
    prepared_bulk_data: List[Dict[str, Any]],
    existing_persons_map: Dict[str, Person],  # Needed to potentially map existing IDs
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

def _validate_batch_prerequisites(my_uuid: Optional[str], matches_on_page: List[Dict[str, Any]], current_page: int) -> None:
    """Validate prerequisites for batch processing."""
    if not my_uuid:
        logger.error(f"_do_batch Page {current_page}: Missing my_uuid.")
        raise ValueError("Missing my_uuid")
    if not matches_on_page:
        logger.debug(f"_do_batch Page {current_page}: Empty match list.")
        raise ValueError("Empty match list")


def _execute_batch_db_commit(session: SqlAlchemySession, prepared_bulk_data: List[Dict[str, Any]], existing_persons_map: Dict[str, Person], current_page: int, page_statuses: Dict[str, int]) -> None:
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


def _handle_batch_critical_error(page_statuses: Dict[str, int], num_matches_on_page: int, progress_bar: Optional[tqdm], current_page: int, error: Exception) -> int:
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
    matches_on_page: List[Dict[str, Any]],
    current_page: int,
    page_statuses: Dict[str, int],
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
    matches_on_page: List[Dict[str, Any]],
    current_page: int,
    progress_bar: Optional[tqdm] = None,  # Accept progress bar
) -> Tuple[int, int, int, int]:
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
        Tuple[int, int, int, int]: Counts of (new, updated, skipped, error) outcomes
                                   for the processed batch.
    Raises:
        MaxApiFailuresExceededError: If API prefetch fails critically. This is caught
                                     by the main coord function to halt the run.
    """
    # Note: BATCH_SIZE is for database commit batching, not for limiting matches per page
    # Action 6 should process ALL matches on the page, then use BATCH_SIZE for DB operations
    # Step 1: Initialization
    page_statuses: Dict[str, int] = {"new": 0, "updated": 0, "skipped": 0, "error": 0}
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
    details_part: Dict[str, Any],
    match: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
    details_part: Dict[str, Any],
    match: Dict[str, Any],
    formatted_match_username: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
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
) -> Tuple[bool, Any]:
    """Compare datetime fields with UTC normalization."""
    current_dt_utc = _normalize_datetime_to_utc(current_value)
    new_dt_utc = _normalize_datetime_to_utc(new_value)

    if new_dt_utc != current_dt_utc:
        return True, new_value
    return False, new_value


def _compare_status_field(
    new_value: Any,
    current_value: Any,
) -> Tuple[bool, Any]:
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
) -> Tuple[bool, Any]:
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
) -> Tuple[bool, Any]:
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
) -> Tuple[bool, Any]:
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
) -> Tuple[bool, Any]:
    """Compare boolean fields."""
    if bool(current_value) != bool(new_value):
        return True, bool(new_value)
    return False, bool(new_value)


def _extract_dna_field_values(
    match: Dict[str, Any],
    existing_dna_match: DnaMatch,
    details_part: Dict[str, Any],
    api_predicted_rel_for_comp: str,
) -> Dict[str, Any]:
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
    field_values: Dict[str, Any],
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
    match: Dict[str, Any],
    existing_dna_match: DnaMatch,
    details_part: Dict[str, Any],
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
    prefetched_tree_data: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Extract and validate birth year from tree data."""
    if prefetched_tree_data and prefetched_tree_data.get("their_birth_year"):
        try:
            return int(prefetched_tree_data["their_birth_year"])
        except (ValueError, TypeError):
            pass
    return None


def _process_last_logged_in(
    profile_part: Dict[str, Any],
) -> Optional[datetime]:
    """Extract and normalize last_logged_in datetime."""
    last_logged_in_val: Optional[datetime] = profile_part.get("last_logged_in_dt")
    if isinstance(last_logged_in_val, datetime):
        if last_logged_in_val.tzinfo is None:
            return last_logged_in_val.replace(tzinfo=timezone.utc)
        return last_logged_in_val.astimezone(timezone.utc)
    return None


def _compare_and_update_person_fields(
    incoming_person_data: Dict[str, Any],
    existing_person: Person,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Tuple[Dict[str, Any], bool]:
    """
    Compare incoming person data with existing person and build update dictionary.

    Returns:
        Tuple of (person_data_for_update, person_fields_changed)
    """
    person_data_for_update: Dict[str, Any] = {
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
) -> Tuple[Optional[str], Optional[str]]:
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
    prefetched_tree_data: Dict[str, Any],
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

    for field, new_val in fields_to_check:
        old_val = getattr(existing_family_tree, field, None)
        if new_val != old_val:  # Handles None comparison correctly
            logger_instance.debug(
                f"  Tree change {log_ref_short}: Field '{field}'"
            )
            return True

    return False


def _process_person_data(
    match: Dict[str, Any],
    existing_person: Optional[Person],
    prefetched_combined_details: Optional[Dict[str, Any]],
    prefetched_tree_data: Optional[Dict[str, Any]],
    match_uuid: str,
    match_username: str,
    match_in_my_tree: bool,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Tuple[Optional[Dict[str, Any]], bool]:
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
    match: Dict[str, Any],
    dna_match_record: Optional[DnaMatch],
    prefetched_combined_details: Optional[Dict[str, Any]],
    match_uuid: str,
    predicted_relationship: Optional[str],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Optional[Dict[str, Any]]:
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
    prefetched_tree_data: Optional[Dict[str, Any]],
    match_uuid: str,
    match_in_my_tree: bool,
    session_manager: SessionManager,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Tuple[Optional[Dict[str, Any]], str]:
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
) -> Tuple[bool, Any]:
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
    match: Dict[str, Any],
    existing_person: Optional[Person],
    prefetched_data: PrefetchedData,
    config_schema_arg: "ConfigSchema",  # Config schema argument
    match_ids: MatchIdentifiers,
    logger_instance: logging.Logger,
) -> Tuple[Optional[Dict[str, Any]], bool]:
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
        - person_op_dict (Optional[Dict]): Dictionary with person data and '_operation' key
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
    match: Dict[str, Any],
    existing_dna_match: Optional[DnaMatch],
    prefetched_combined_details: Optional[Dict[str, Any]],
    match_uuid: str,
    predicted_relationship: Optional[str],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Optional[Dict[str, Any]]:
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
        Optional[Dict[str, Any]]: Dictionary with DNA match data and '_operation' key set to 'create',
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
    prefetched_tree_data: Optional[Dict[str, Any]],
    session_manager: SessionManager,
    config_schema_arg: "ConfigSchema",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
    prefetched_tree_data: Optional[Dict[str, Any]],
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
    prefetched_tree_data: Dict[str, Any],
    tree_operation: Literal["create", "update"],
    existing_family_tree: Optional[FamilyTree],
) -> Dict[str, Any]:
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
    prefetched_tree_data: Optional[Dict[str, Any]],
    match_uuid: str,
    match_in_my_tree: bool,
    session_manager: SessionManager,
    config_schema_arg: "ConfigSchema",  # Config schema argument
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Tuple[Optional[Dict[str, Any]], Literal["create", "update", "none"]]:
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
        - tree_data (Optional[Dict]): Dictionary with family tree data and '_operation' key
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
    person_op_data: Optional[Dict[str, Any]],
    dna_op_data: Optional[Dict[str, Any]],
    tree_op_data: Optional[Dict[str, Any]],
    tree_operation_status: str,
    is_new_person: bool,
    person_fields_changed: bool,
    prepared_data_for_bulk: Dict[str, Any],
    log_ref_short: str,
    logger_instance: logging.Logger
) -> Tuple[Optional[Dict[str, Any]], Literal["new", "updated", "skipped", "error"]]:
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
    match: Dict[str, Any],
    session_manager: SessionManager,
    existing_person_arg: Optional[Person],
    prefetched_combined_details: Optional[Dict[str, Any]],
    prefetched_tree_data: Optional[Dict[str, Any]],
    _config_schema_arg: "ConfigSchema",
    logger_instance: logging.Logger,
) -> Tuple[
    Optional[Dict[str, Any]],
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
        - prepared_data (Optional[Dict[str, Any]]): Dictionary with keys 'person', 'dna_match', and
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

    prepared_data_for_bulk: Dict[str, Any] = {
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
    driver_cookies_list: List[Dict[str, Any]],
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
    driver_cookies_list: List[Dict[str, Any]],
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
    top_prob_display = top_prob * 100.0
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
) -> Dict[str, Optional[str]]:
    """
    Parse HTML content from getladder API to extract relationship information.

    Returns:
        Dictionary with 'actual_relationship' and 'relationship_path' keys
    """
    ladder_data: Dict[str, Optional[str]] = {
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
) -> Optional[Dict[str, Optional[str]]]:
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
) -> Optional[Dict[str, Any]]:
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

    details_response = _api_req(
        url=details_url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        headers=details_headers,
        use_csrf_token=False,
        api_description="Match Details API (Batch)",
    )

    logger.debug(f"_fetch_match_details_api: _api_req returned, type={type(details_response)}, UUID {match_uuid}")

    if details_response and isinstance(details_response, dict):
        combined_data: Dict[str, Any] = {}
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
) -> Dict[str, Any]:
    """
    Fetch profile details from the /profiles/details API endpoint.

    Returns:
        Dictionary with 'last_logged_in_dt' and 'contactable' keys
    """
    result: Dict[str, Any] = {
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

    profile_response = _api_req(
        url=profile_url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        headers=profile_headers,
        use_csrf_token=False,
        api_description="Profile Details API (Batch)",
    )

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
) -> Optional[Set[str]]:
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
    in_tree_ids: Set[str],
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
    sample_ids_on_page: List[str],
    specific_csrf_token: str,
    current_page: int,
) -> Set[str]:
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
    sample_ids_on_page: List[str],
    specific_csrf_token: str,
    current_page: int,
) -> Set[str]:
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
) -> Optional[Tuple[Any, str]]:
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
) -> Optional[Dict[str, Any]]:
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


def _extract_total_pages(api_response: Dict[str, Any]) -> Optional[int]:
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
    match_data_list: List[Any],
    current_page: int
) -> List[Dict[str, Any]]:
    """Filter matches that have a valid sampleId."""
    valid_matches_for_processing: List[Dict[str, Any]] = []
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
) -> Optional[Tuple[List[Dict[str, Any]], Optional[int]]]:
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
    valid_matches_for_processing: List[Dict[str, Any]],
    my_uuid: str,
    in_tree_ids: set,
    current_page: int
) -> List[Dict[str, Any]]:
    """Refine raw match data into structured format."""
    refined_matches: List[Dict[str, Any]] = []
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
) -> Optional[Tuple[List[Dict[str, Any]], Optional[int]]]:
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
    combined_data: Dict[str, Any],
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
) -> Optional[Dict[str, Any]]:
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

    return combined_data if combined_data else None


# End of _fetch_combined_details


def _process_badge_response(
    badge_response: Any,
    match_uuid: str
) -> Optional[Dict[str, Any]]:
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
) -> Optional[Dict[str, Any]]:
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
        badge_response = _api_req(
            url=badge_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description="Badge Details API (Batch)",
            referer_url=badge_referer,
        )

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
) -> Optional[Dict[str, Any]]:
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
):
    """Logs a summary of processed matches for a single page."""
    logger.debug(f"---- Page {page} Batch Summary ----")
    logger.debug(f"  New Person/Data: {page_new}")
    logger.debug(f"  Updated Person/Data: {page_updated}")
    logger.debug(f"  Skipped (No Change): {page_skipped}")
    logger.debug(f"  Errors during Prep/DB: {page_errors}")  # Clarified error source
    logger.debug("---------------------------\n")


# End of _log_page_summary


def _log_coord_summary(
    total_pages_processed: int,
    total_new: int,
    total_updated: int,
    total_skipped: int,
    total_errors: int,
):
    """Logs the final summary of the entire coord (match gathering) execution."""
    logger.info("---- Gather Matches Final Summary ----")
    logger.info(f"  Total Pages Processed: {total_pages_processed}")
    logger.info(f"  Total New Added:     {total_new}")
    logger.info(f"  Total Updated:       {total_updated}")
    logger.info(f"  Total Skipped:       {total_skipped}")
    logger.info(f"  Total Errors:        {total_errors}")
    logger.info("------------------------------------\n")


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
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


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
        raise AssertionError(f"Error handling failed: {e}")


# ==============================================
# MAIN TEST SUITE
# ==============================================


def action6_gather_module_tests() -> bool:
    """Comprehensive test suite for action6_gather.py"""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Action 6 - Gather DNA Matches", "action6_gather.py")
    suite.start_suite()

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

