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

# --- Standard library imports ---
import contextlib # For db_transn context manager if used directly (unlikely now)
import json
import logging
import math
import random
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

# --- Third-party imports ---
import cloudscraper # For specific API calls if needed, though _api_req preferred
import requests
from bs4 import BeautifulSoup, Tag # For HTML parsing if needed (e.g., ladder)
from cachetools import Cache # If additional in-memory caching needed
from diskcache.core import ENOVAL # For checking cache misses
from requests.adapters import HTTPAdapter
from requests.cookies import RequestsCookieJar
from requests.exceptions import ConnectionError, HTTPError, RequestException
from selenium.common.exceptions import (
    NoSuchCookieException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session as SqlAlchemySession, joinedload # Alias Session
from tqdm.auto import tqdm # Progress bar
from tqdm.contrib.logging import logging_redirect_tqdm # Redirect logging through tqdm

# --- Local application imports ---
from cache import cache as global_cache # Use the initialized global cache instance
from cache import cache_result # Decorator for caching function results
from config import config_instance, selenium_config # Configuration singletons
from database import ( # Database models and utilities
    DnaMatch,
    FamilyTree,
    Person,
    PersonStatusEnum, # Enums
    ConversationLog, # Import even if not directly used here, for relationships
    MessageType, # Import even if not directly used here
    db_transn, # Transaction context manager
)
from logging_config import logger # Use configured logger
from my_selectors import * # Import CSS selectors
from utils import ( # Core utilities
    DynamicRateLimiter,
    MatchData, # Data class (though not directly used here anymore)
    SessionManager,
    _api_req, # API request helper
    format_name, # Name formatting utility
    get_driver_cookies, # Cookie utility
    make_newrelic, # Header generation utilities
    make_traceparent,
    make_tracestate,
    make_ube,
    nav_to_page, # Navigation utility
    ordinal_case, # String formatting utility
    retry, # Decorators
    retry_api,
    time_wait,
    urljoin, # URL utility
)


# --- Constants ---
MATCHES_PER_PAGE: int = 20 # Default matches per page (adjust based on API response)

# Configurable settings from config_instance
DB_ERROR_PAGE_THRESHOLD: int = config_instance._get_int_env("DB_ERROR_PAGE_THRESHOLD", 10) # Max consecutive DB errors allowed
THREAD_POOL_WORKERS: int = config_instance._get_int_env("GATHER_THREAD_POOL_WORKERS", 5) # Concurrent API workers

# ------------------------------------------------------------------------------
# Core Orchestration (coord)
# ------------------------------------------------------------------------------

def coord(session_manager: SessionManager, config_instance, start: int = 1) -> bool:
    """
    Orchestrates the gathering of DNA matches from Ancestry.
    Handles pagination, fetches match data (list view and details via API),
    compares with the local database, and triggers batch processing for updates/inserts.

    Args:
        session_manager: The active SessionManager instance.
        config_instance: The application configuration instance.
        start: The page number to start gathering from (default is 1).

    Returns:
        True if the process completed successfully (or partially without critical errors),
        False if a critical error occurred preventing further processing.
    """
    # Step 1: Validate Session State
    driver = session_manager.driver
    if not driver or not session_manager.driver_live or not session_manager.session_ready:
        logger.error("coord: WebDriver/Session not ready. Exiting gather action.")
        return False
    my_uuid = session_manager.my_uuid
    if not my_uuid:
        logger.error("coord: Failed to retrieve my_uuid from session_manager. Exiting.")
        return False

    # Step 2: Initialize counters and state variables
    total_new, total_updated, total_skipped, total_errors = 0, 0, 0, 0
    total_pages_processed = 0
    progress_bar: Optional[tqdm] = None
    final_success = True # Assume success until a critical error occurs
    target_matches_url_base = urljoin(config_instance.BASE_URL, f"discoveryui-matches/list/{my_uuid}")
    total_pages: Optional[int] = None # Will be determined by first API call
    last_page_to_process: Optional[int] = None # Determined by config and total_pages
    matches_on_page: List[Dict[str, Any]] = [] # Stores data for the current page
    db_connection_errors = 0 # Counter for consecutive DB session failures

    # Step 3: Validate and set start page
    try:
        start_page = int(start)
        if start_page <= 0:
            logger.warning(f"Invalid start page '{start}'. Using default page 1.")
            start_page = 1
    except (ValueError, TypeError):
        logger.warning(f"Invalid start page value '{start}'. Using default page 1.")
        start_page = 1

    logger.debug(f"--- Starting DNA Match Gathering (Action 6) from page {start_page} ---")

    try: # Main execution block
        # Step 4: Ensure browser is on the correct DNA match list page
        logger.debug("Ensuring browser is on the DNA matches list page...")
        try:
            current_url = driver.current_url
            # Check if current URL starts with the expected base for the match list
            if not current_url.startswith(target_matches_url_base):
                logger.debug("Not on match list page. Navigating...")
                if not nav_to_list(session_manager):
                    logger.error("Failed to navigate to DNA match list page. Exiting coord.")
                    return False
                logger.debug("Successfully navigated to DNA matches page.")
            else:
                logger.debug(f"Already on correct DNA matches page: {current_url}")
        except WebDriverException as nav_e:
            logger.error(f"WebDriver error checking/navigating to match list: {nav_e}", exc_info=True)
            return False # Cannot proceed without navigation

        # Step 5: Initial Fetch to Determine Total Pages
        logger.debug(f"Fetching initial page {start_page} to determine total pages...")
        # --- Get DB Session with Retry ---
        db_session_for_page: Optional[SqlAlchemySession] = None
        for retry_attempt in range(3):
            db_session_for_page = session_manager.get_db_conn()
            if db_session_for_page: break
            logger.warning(f"DB session attempt {retry_attempt + 1}/3 failed. Retrying in 5s...")
            time.sleep(5)
        if not db_session_for_page:
            logger.critical("Could not get DB session for initial page fetch after retries. Aborting.")
            return False
        # --- End DB Session Retry ---

        fetched_total_pages: Optional[int] = None
        try:
            # Fetch match data for the starting page
            if not session_manager.is_sess_valid(): # Re-check session before API call
                raise ConnectionError("WebDriver session invalid before initial get_matches.")
            result = get_matches(session_manager, db_session_for_page, start_page)
            if result is None:
                matches_on_page, fetched_total_pages = [], None # Handle None result
                logger.error(f"Initial get_matches for page {start_page} returned None.")
            else:
                matches_on_page, fetched_total_pages = result
            db_connection_errors = 0 # Reset counter on successful fetch
        except ConnectionError as init_conn_e:
            logger.critical(f"ConnectionError during initial get_matches: {init_conn_e}. Aborting.", exc_info=False)
            final_success = False
        except Exception as get_match_err:
            logger.error(f"Error during initial get_matches call on page {start_page}: {get_match_err}", exc_info=True)
            final_success = False # Mark as failed but continue cleanup
        finally:
            if db_session_for_page:
                session_manager.return_session(db_session_for_page) # Ensure session is returned

        # Abort if initial fetch failed critically or couldn't get total pages
        if not final_success or fetched_total_pages is None:
            logger.error("Failed to retrieve total_pages on initial fetch. Aborting.")
            return False
        total_pages = fetched_total_pages
        logger.info(f"Total pages found: {total_pages}")

        # Step 6: Determine Page Range to Process based on MAX_PAGES config
        max_pages_config = config_instance.MAX_PAGES
        # Calculate the number of pages to process from config (0 means all)
        pages_to_process_config = (min(max_pages_config, total_pages) if max_pages_config > 0 else total_pages)
        # Calculate the last page number in the range for this run
        last_page_to_process = min(start_page + pages_to_process_config - 1, total_pages)
        # Calculate total pages actually being processed in this run
        total_pages_in_run = max(0, last_page_to_process - start_page + 1)

        if total_pages_in_run <= 0:
            logger.info(f"No pages to process (Start: {start_page}, End: {last_page_to_process}).")
            return True # Nothing to do, considered successful

        # Estimate total matches for progress bar using CORRECTED constant
        total_matches_estimate = total_pages_in_run * MATCHES_PER_PAGE
        logger.info(f"Processing {total_pages_in_run} pages (approx. {total_matches_estimate} matches) from page {start_page} to {last_page_to_process}.\n")

        # Step 7: Main Processing Loop (Page by Page)
        with logging_redirect_tqdm(): # Redirect logging through tqdm for clean output
            # *** CORRECTED bar_format ***
            progress_bar = tqdm(
                total=total_matches_estimate,
                desc="Gathering Matches",
                unit=" match",
                ncols=100,
                # Removed time elements: [{elapsed}<{remaining}, {rate_fmt}]
                bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
                file=sys.stderr, # Ensure bar goes to stderr
                leave=True, # Keep final bar
            )

            current_page_num = start_page
            while True:
                # Step 7a: Check if processing limit reached
                if current_page_num > last_page_to_process:
                    logger.debug(f"Current page {current_page_num} exceeds processing limit {last_page_to_process}. Stopping.")
                    break

                # Step 7b: Check WebDriver session validity before processing page
                if not session_manager.is_sess_valid():
                    logger.critical(f"WebDriver session invalid/unreachable before processing page {current_page_num}. Aborting run.")
                    final_success = False
                    # Update progress bar for remaining estimated items as errors
                    remaining_matches_estimate = max(0, progress_bar.total - progress_bar.n)
                    if remaining_matches_estimate > 0:
                        logger.info(f"Marking {remaining_matches_estimate} remaining estimated matches as errors.")
                        progress_bar.update(remaining_matches_estimate)
                        total_errors += remaining_matches_estimate
                    break # Exit loop on critical session failure

                # Step 7c: Fetch match data for the current page (unless already fetched)
                # Skip fetch only if it's the start page AND we have data from initial fetch
                if not (current_page_num == start_page and matches_on_page):
                    db_session_for_page = None
                    # --- Get DB Session with Retry ---
                    for retry_attempt in range(3):
                        db_session_for_page = session_manager.get_db_conn()
                        if db_session_for_page:
                            db_connection_errors = 0 # Reset counter on success
                            break
                        logger.warning(f"DB session attempt {retry_attempt + 1}/3 failed for page {current_page_num}. Retrying in 5s...")
                        time.sleep(5)
                    if not db_session_for_page:
                        db_connection_errors += 1
                        logger.error(f"Could not get DB session for page {current_page_num} after retries. Skipping page.")
                        total_errors += MATCHES_PER_PAGE # Assume full page error
                        # *** Ensure progress bar update on page skip ***
                        if progress_bar: progress_bar.update(MATCHES_PER_PAGE)
                        if db_connection_errors >= DB_ERROR_PAGE_THRESHOLD:
                            logger.critical(f"Aborting run due to {db_connection_errors} consecutive DB connection failures.")
                            final_success = False
                            # Mark remaining as errors
                            remaining_matches_estimate = max(0, progress_bar.total - progress_bar.n)
                            if remaining_matches_estimate > 0: progress_bar.update(remaining_matches_estimate); total_errors += remaining_matches_estimate
                            break # Exit loop
                        current_page_num += 1
                        continue # Skip to next page
                    # --- End DB Session Retry ---

                    try:
                        if not session_manager.is_sess_valid(): # Re-check before API call
                            raise ConnectionError(f"WebDriver session invalid before get_matches page {current_page_num}.")
                        result = get_matches(session_manager, db_session_for_page, current_page_num)
                        if result is None:
                            matches_on_page = [] # Treat None result as empty
                            logger.warning(f"get_matches returned None for page {current_page_num}. Skipping.")
                            # *** Ensure progress bar update on page skip ***
                            if progress_bar: progress_bar.update(MATCHES_PER_PAGE); total_errors += MATCHES_PER_PAGE
                        else:
                            matches_on_page, _ = result # Ignore total_pages from subsequent calls
                    except ConnectionError as conn_e:
                        logger.error(f"ConnectionError get_matches page {current_page_num}: {conn_e}", exc_info=False)
                        # *** Ensure progress bar update on page skip ***
                        if progress_bar: progress_bar.update(MATCHES_PER_PAGE); total_errors += MATCHES_PER_PAGE
                        matches_on_page = [] # Clear data on error
                        time.sleep(5) # Pause after connection error
                        current_page_num += 1; continue # Skip this page
                    except Exception as get_match_e:
                        logger.error(f"Error get_matches page {current_page_num}: {get_match_e}", exc_info=True)
                        # *** Ensure progress bar update on page skip ***
                        if progress_bar: progress_bar.update(MATCHES_PER_PAGE); total_errors += MATCHES_PER_PAGE
                        matches_on_page = [] # Clear data on error
                        time.sleep(2) # Shorter pause for other errors
                        current_page_num += 1; continue # Skip this page
                    finally:
                        if db_session_for_page:
                            session_manager.return_session(db_session_for_page)

                # Step 7d: Process the fetched matches if any found
                if not matches_on_page:
                    # Handle empty page (could be end of list or API glitch)
                    logger.info(f"No matches found or processed on page {current_page_num}.")
                    # Update progress bar estimate if it wasn't the initial empty fetch
                    if progress_bar and not (current_page_num == start_page and total_pages_processed == 0):
                        # *** Ensure progress bar update on empty page ***
                        progress_bar.update(MATCHES_PER_PAGE) # Advance by expected amount
                    matches_on_page = [] # Ensure list is empty for next iteration
                    current_page_num += 1
                    time.sleep(0.5) # Small pause
                    continue # Go to next page

                # Step 7e: Call the batch processing function
                # Note: _do_batch updates the progress bar internally for each item processed
                page_new, page_updated, page_skipped, page_errors = _do_batch(
                    session_manager=session_manager,
                    matches_on_page=matches_on_page,
                    current_page=current_page_num,
                    progress_bar=progress_bar, # Pass progress bar instance
                )

                # Step 7f: Update overall counters
                total_new += page_new
                total_updated += page_updated
                total_skipped += page_skipped
                total_errors += page_errors
                total_pages_processed += 1

                # Step 7g: Update progress bar postfix with cumulative stats
                progress_bar.set_postfix(New=total_new, Upd=total_updated, Skip=total_skipped, Err=total_errors, refresh=True)

                # Step 7h: Adjust rate limiter delay based on success/throttling
                _adjust_delay(session_manager, current_page_num)
                # Apply inter-page delay using rate limiter
                inter_page_delay = session_manager.dynamic_rate_limiter.wait()
                # Optional: logger.debug(f"Rate limit inter-page delay: {inter_page_delay:.2f}s")

                # --- CRITICAL: Clear matches_on_page for the next iteration ---
                matches_on_page = []
                # --- End Clear ---

                # Step 7i: Move to the next page
                current_page_num += 1

            # --- End of Page Loop ---

            # *** Ensure progress bar reaches 100% if stopped early but successfully ***
            if progress_bar and progress_bar.n < progress_bar.total and final_success:
                logger.debug(f"Loop finished early or estimate high. Closing progress bar at {progress_bar.n}/{progress_bar.total}")
                # Option 1: Force update to total (might look jerky)
                # progress_bar.update(progress_bar.total - progress_bar.n)
                # Option 2: Just close it - it will show the final count vs total estimate
                # Closing without forcing update is generally preferred unless 100% is essential display
                pass # Bar will close in finally block

    # Step 8: Handle specific exceptions (e.g., KeyboardInterrupt)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected. Stopping match gathering.")
        final_success = False # Mark as incomplete
    except ConnectionError as coord_conn_err: # Catch connection errors potentially missed
        logger.critical(f"ConnectionError during coord execution: {coord_conn_err}", exc_info=True)
        final_success = False
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Critical error during coord execution: {e}", exc_info=True)
        final_success = False

    # Step 9: Final cleanup and summary logging
    finally:
        logger.debug("Entering finally block in coord...")
        if progress_bar:
            # Ensure final stats are reflected in the bar before closing
            progress_bar.set_postfix(New=total_new, Upd=total_updated, Skip=total_skipped, Err=total_errors, refresh=True)
            # Ensure bar closes cleanly even if loop exited early
            if progress_bar.n < progress_bar.total:
                progress_bar.total = progress_bar.n # Adjust total to current count before closing
            progress_bar.close()
            print("", file=sys.stderr) # Newline after final bar closes

        # Log the summary of the gathering process
        _log_coord_summary(total_pages_processed, total_new, total_updated, total_skipped, total_errors)

        # Re-raise KeyboardInterrupt if that was the cause of exit
        exc_info = sys.exc_info()
        if exc_info[0] is KeyboardInterrupt:
            logger.info("Re-raising KeyboardInterrupt after cleanup.")
            if exc_info[1] is not None:
                raise exc_info[1].with_traceback(exc_info[2]) # Preserve traceback

        logger.debug("Exiting finally block in coord.")

    # Step 10: Return overall success status
    return final_success
# End of coord

# ------------------------------------------------------------------------------
# Batch Processing Logic (_do_batch and Helpers)
# ------------------------------------------------------------------------------

def _lookup_existing_persons(session: SqlAlchemySession, uuids_on_page: List[str]) -> Dict[str, Person]:
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
            .options(
                joinedload(Person.dna_match),
                joinedload(Person.family_tree)
            )
            # Filter by the list of uppercase UUIDs
            .filter(Person.uuid.in_(uuids_upper))
            .all()
        )
        # Step 4: Populate the result map (key by UUID)
        existing_persons_map = {person.uuid: person for person in existing_persons if person.uuid}
        logger.debug(f"Found {len(existing_persons_map)} existing Person records for this batch.")

    # Step 5: Handle potential database errors
    except SQLAlchemyError as db_lookup_err:
        # Check specifically for Enum mismatch errors which can be critical
        if "is not among the defined enum values" in str(db_lookup_err):
            logger.critical(f"CRITICAL ENUM MISMATCH during Person lookup. DB schema might be outdated. Error: {db_lookup_err}")
            # Raise a specific error to halt processing if schema mismatch detected
            raise ValueError("Database enum mismatch detected during person lookup.") from db_lookup_err
        else:
            # Log other SQLAlchemy errors and re-raise
            logger.error(f"Database lookup failed during prefetch: {db_lookup_err}", exc_info=True)
            raise # Re-raise to be handled by the caller (_do_batch)
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during Person lookup: {e}", exc_info=True)
        raise # Re-raise to be handled by the caller

    # Step 6: Return the map of found persons
    return existing_persons_map
# End of _lookup_existing_persons


def _identify_fetch_candidates(
    matches_on_page: List[Dict[str, Any]],
    existing_persons_map: Dict[str, Person]
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
    # Step 1: Initialize results
    fetch_candidates_uuid: Set[str] = set()
    skipped_count_this_batch = 0
    matches_to_process_later: List[Dict[str, Any]] = []
    invalid_uuid_count = 0

    logger.debug("Identifying fetch candidates vs. skipped matches...")

    # Step 2: Iterate through matches fetched from the current page
    for match_api_data in matches_on_page:
        # Step 2a: Validate UUID presence
        uuid_val = match_api_data.get("uuid")
        if not uuid_val:
            logger.warning(f"Skipping match missing UUID: {match_api_data}")
            invalid_uuid_count += 1
            continue

        # Step 2b: Check if this person exists in the database
        existing_person = existing_persons_map.get(uuid_val.upper()) # Use uppercase UUID

        if not existing_person:
            # --- Case 1: New Person ---
            # Always fetch details for new people.
            fetch_candidates_uuid.add(uuid_val)
            matches_to_process_later.append(match_api_data)
        else:
            # --- Case 2: Existing Person ---
            # Determine if details fetch is needed based on potential changes.
            needs_fetch = False
            existing_dna = existing_person.dna_match
            existing_tree = existing_person.family_tree
            db_in_tree = existing_person.in_my_tree
            api_in_tree = match_api_data.get("in_my_tree", False) # From get_matches

            # Step 2c: Check for changes in core DNA list data
            if existing_dna:
                try:
                    # Compare cM (integer conversion for safety)
                    api_cm = int(match_api_data.get("cM_DNA", 0))
                    db_cm = existing_dna.cM_DNA
                    if api_cm != db_cm: needs_fetch = True; logger.debug(f"  Fetch needed (UUID {uuid_val}): cM changed ({db_cm} -> {api_cm})")

                    # Compare segments (integer conversion)
                    api_segments = int(match_api_data.get("numSharedSegments", 0))
                    db_segments = existing_dna.shared_segments
                    # NOTE: Use >= comparison for segments as list view might be lower than detail view sometimes? Or stick to != ? Sticking to != for now.
                    if api_segments != db_segments: needs_fetch = True; logger.debug(f"  Fetch needed (UUID {uuid_val}): Segments changed ({db_segments} -> {api_segments})")

                except (ValueError, TypeError, AttributeError) as comp_err:
                     logger.warning(f"Error comparing list DNA data for UUID {uuid_val}: {comp_err}. Assuming fetch needed.")
                     needs_fetch = True
            else:
                # If DNA record doesn't exist, fetch details.
                needs_fetch = True
                logger.debug(f"  Fetch needed (UUID {uuid_val}): No existing DNA record.")

            # Step 2d: Check for changes in tree status or missing tree record
            if bool(api_in_tree) != bool(db_in_tree):
                # If tree linkage status changed, fetch details.
                needs_fetch = True
                logger.debug(f"  Fetch needed (UUID {uuid_val}): Tree status changed ({db_in_tree} -> {api_in_tree})")
            elif api_in_tree and not existing_tree:
                # If marked in tree but no DB record exists, fetch details.
                needs_fetch = True
                logger.debug(f"  Fetch needed (UUID {uuid_val}): Marked in tree but no DB record.")

            # Step 2e: Add to fetch list or increment skipped count
            if needs_fetch:
                fetch_candidates_uuid.add(uuid_val)
                matches_to_process_later.append(match_api_data)
            else:
                skipped_count_this_batch += 1

    # Step 3: Log summary of identification
    if invalid_uuid_count > 0:
        logger.error(f"{invalid_uuid_count} matches skipped during identification due to missing UUID.")
    logger.debug(f"Identified {len(fetch_candidates_uuid)} candidates for API detail fetch, {skipped_count_this_batch} skipped (no change detected from list view).")

    # Step 4: Return results
    return fetch_candidates_uuid, matches_to_process_later, skipped_count_this_batch
# End of _identify_fetch_candidates


def _perform_api_prefetches(
    session_manager: SessionManager,
    fetch_candidates_uuid: Set[str],
    matches_to_process_later: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Performs parallel API calls to prefetch detailed data for candidate matches
    using a ThreadPoolExecutor. Fetches combined details, relationship probability,
    badge details (for tree members), and ladder details (for tree members).

    Args:
        session_manager: The active SessionManager instance.
        fetch_candidates_uuid: Set of UUIDs requiring detail fetches.
        matches_to_process_later: List of match data dicts corresponding to the candidates.

    Returns:
        A dictionary containing the prefetched data, organized by type:
        {
            "combined": {uuid: combined_details_dict_or_None, ...},
            "tree": {uuid: combined_badge_ladder_dict_or_None, ...},
            "rel_prob": {uuid: relationship_prob_string_or_None, ...}
        }
    """
    # Step 1: Initialize result dictionaries and check for candidates
    batch_combined_details: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_badge_data: Dict[str, Optional[Dict[str, Any]]] = {} # Temporary store for badge results
    batch_ladder_data: Dict[str, Optional[Dict[str, Any]]] = {} # Temporary store for ladder results
    batch_relationship_prob_data: Dict[str, Optional[str]] = {}
    batch_tree_data: Dict[str, Dict[str, Any]] = {} # Final combined tree/ladder data

    if not fetch_candidates_uuid:
        logger.debug("No fetch candidates provided for API pre-fetch.")
        return {"combined": {}, "tree": {}, "rel_prob": {}} # Return empty structure

    # Step 2: Initialize ThreadPoolExecutor and futures tracking
    futures: Dict[Any, Tuple[str, str]] = {} # Map future object to (task_type, identifier)
    fetch_start_time = time.time()
    num_candidates = len(fetch_candidates_uuid)
    my_tree_id = session_manager.my_tree_id # Get tree ID needed for ladder calls

    logger.debug(f"--- Starting Parallel API Pre-fetch ({num_candidates} candidates, {THREAD_POOL_WORKERS} workers) ---")

    # Step 3: Identify UUIDs needing Badge/Ladder details (those marked in_my_tree)
    uuids_for_tree_badge_ladder = {
        match_data["uuid"]
        for match_data in matches_to_process_later
        if match_data.get("in_my_tree") and match_data.get("uuid") in fetch_candidates_uuid
    }
    logger.debug(f"Identified {len(uuids_for_tree_badge_ladder)} candidates for Badge/Ladder fetch.")

    # Step 4: Submit API tasks to the ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
        # --- Submit Combined Details & Relationship Probability tasks for ALL candidates ---
        for uuid_val in fetch_candidates_uuid:
            # Apply rate limit wait *before* submitting each task
            _ = session_manager.dynamic_rate_limiter.wait()
            # Submit combined details fetch task
            futures[executor.submit(_fetch_combined_details, session_manager, uuid_val)] = ("combined_details", uuid_val)

            _ = session_manager.dynamic_rate_limiter.wait()
            # Submit relationship probability fetch task (assuming default max labels)
            max_labels = 2 # Could be made configurable if needed
            futures[executor.submit(_fetch_batch_relationship_prob, session_manager, uuid_val, max_labels)] = ("relationship_prob", uuid_val)

        # --- Submit Badge Details tasks ONLY for tree members ---
        for uuid_val in uuids_for_tree_badge_ladder:
            _ = session_manager.dynamic_rate_limiter.wait()
            # Submit badge details fetch task
            futures[executor.submit(_fetch_batch_badge_details, session_manager, uuid_val)] = ("badge_details", uuid_val)

        # Step 5: Process completed futures as they finish (Combined, RelProb, Badge)
        temp_badge_results: Dict[str, Optional[Dict[str, Any]]] = {} # Store badge results keyed by UUID
        logger.debug(f"Processing {len(futures)} initially submitted API tasks...")
        for future in as_completed(futures):
            task_type, identifier_uuid = futures[future] # Identifier is UUID here
            try:
                result = future.result() # Get result (can raise exceptions)
                if result is not None:
                    if task_type == "combined_details": batch_combined_details[identifier_uuid] = result
                    elif task_type == "badge_details": temp_badge_results[identifier_uuid] = result
                    elif task_type == "relationship_prob": batch_relationship_prob_data[identifier_uuid] = result
                # else: logger.debug(f"Prefetch task '{task_type}' for {identifier_uuid} returned None.")
            except ConnectionError as conn_err:
                logger.error(f"ConnErr prefetch '{task_type}' {identifier_uuid}: {conn_err}", exc_info=False)
                if task_type == "relationship_prob": batch_relationship_prob_data[identifier_uuid] = "N/A (Conn Error)"
            except Exception as exc:
                logger.error(f"Exc prefetch '{task_type}' {identifier_uuid}: {exc}", exc_info=False)
                if task_type == "relationship_prob": batch_relationship_prob_data[identifier_uuid] = "N/A (Fetch Error)"

        # Step 6: Submit Ladder Details tasks based on successful Badge results
        cfpid_to_uuid_map: Dict[str, str] = {} # Map CFPID back to UUID for storing results
        ladder_futures = {}
        if my_tree_id and temp_badge_results:
            cfpid_list_for_ladder: List[str] = []
            for uuid_val, badge_result in temp_badge_results.items():
                cfpid = badge_result.get("their_cfpid") if badge_result else None
                if cfpid: cfpid_list_for_ladder.append(cfpid); cfpid_to_uuid_map[cfpid] = uuid_val

            if cfpid_list_for_ladder:
                logger.debug(f"Submitting Ladder tasks for {len(cfpid_list_for_ladder)} CFPIDs...")
                for cfpid in cfpid_list_for_ladder:
                    _ = session_manager.dynamic_rate_limiter.wait()
                    ladder_futures[executor.submit(_fetch_batch_ladder, session_manager, cfpid, my_tree_id)] = ("ladder", cfpid)

        # Step 7: Process completed Ladder futures
        logger.debug(f"Processing {len(ladder_futures)} Ladder API tasks...")
        for future in as_completed(ladder_futures):
            task_type, identifier_cfpid = ladder_futures[future] # Identifier is CFPID here
            try:
                result = future.result()
                if result is not None:
                    uuid_for_ladder = cfpid_to_uuid_map.get(identifier_cfpid)
                    if uuid_for_ladder: batch_ladder_data[uuid_for_ladder] = result
                    else: logger.warning(f"Could not map ladder result for CFPID {identifier_cfpid} back to UUID.")
            except ConnectionError as conn_err:
                logger.error(f"ConnErr ladder fetch CFPID {identifier_cfpid}: {conn_err}", exc_info=False)
            except Exception as exc:
                logger.error(f"Exc ladder fetch CFPID {identifier_cfpid}: {exc}", exc_info=False)

    # --- End ThreadPoolExecutor block ---

    fetch_duration = time.time() - fetch_start_time
    logger.debug(f"--- Finished Parallel API Pre-fetch. Duration: {fetch_duration:.2f}s ---")

    # Step 8: Combine Badge and Ladder results into final Tree data dictionary (keyed by UUID)
    for uuid_val, badge_result in temp_badge_results.items():
        if badge_result: # Check if badge fetch was successful
            combined_tree_info = badge_result.copy() # Start with badge data
            ladder_result_for_uuid = batch_ladder_data.get(uuid_val)
            if ladder_result_for_uuid: combined_tree_info.update(ladder_result_for_uuid) # Add ladder data
            batch_tree_data[uuid_val] = combined_tree_info

    # Step 9: Return the aggregated prefetched data
    return {
        "combined": batch_combined_details,
        "tree": batch_tree_data,
        "rel_prob": batch_relationship_prob_data,
    }
# End of _perform_api_prefetches


def _prepare_bulk_db_data(
    session: SqlAlchemySession,
    session_manager: SessionManager,
    matches_to_process: List[Dict[str, Any]],
    existing_persons_map: Dict[str, Person],
    prefetched_data: Dict[str, Dict[str, Any]],
    progress_bar: Optional[tqdm]
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
    page_statuses: Dict[str, int] = {"new": 0, "updated": 0, "error": 0} # Skipped handled before this function
    num_to_process = len(matches_to_process)

    if not num_to_process:
        return [], page_statuses # Return empty if nothing to process

    logger.debug(f"--- Preparing DB data structures for {num_to_process} candidates ---")
    process_start_time = time.time()

    # Step 2: Iterate through each candidate match
    for match_list_data in matches_to_process:
        # Initialize state for this match
        uuid_val = match_list_data.get("uuid")
        log_ref_short = f"UUID={uuid_val or 'MISSING'} User='{match_list_data.get('username', 'Unknown')}'"
        prepared_data_for_this_match: Optional[Dict[str, Any]] = None
        status_for_this_match: Literal["new", "updated", "skipped", "error"] = "error" # Default to error
        error_msg_for_this_match: Optional[str] = None

        try:
            # Step 2a: Basic validation
            if not uuid_val:
                logger.error(f"Critical error: Match data missing UUID in _prepare_bulk_db_data. Skipping.")
                status_for_this_match = "error"; error_msg_for_this_match = "Missing UUID"
                raise ValueError("Missing UUID") # Stop processing this item

            # Step 2b: Retrieve existing person and prefetched data
            existing_person = existing_persons_map.get(uuid_val.upper())
            prefetched_combined = prefetched_data.get("combined", {}).get(uuid_val)
            prefetched_tree = prefetched_data.get("tree", {}).get(uuid_val)
            prefetched_rel_prob = prefetched_data.get("rel_prob", {}).get(uuid_val)

            # Step 2c: Add relationship probability to match dict *before* calling _do_match
            match_list_data["predicted_relationship"] = prefetched_rel_prob or "N/A (Fetch Failed)"

            # Step 2d: Check WebDriver session validity before calling _do_match
            if not session_manager.is_sess_valid():
                logger.error(f"WebDriver session invalid before calling _do_match for {log_ref_short}. Treating as error.")
                status_for_this_match = "error"
                error_msg_for_this_match = "WebDriver session invalid"
                # Need to raise an exception or handle this state appropriately to stop/skip
                # For now, let it proceed but the status is error.
            else:
                # Step 2e: Call _do_match to compare data and prepare the bulk dictionary structure
                (prepared_data_for_this_match, status_for_this_match, error_msg_for_this_match) = _do_match(
                    session=session,
                    match=match_list_data, # Pass the match data from the list
                    session_manager=session_manager,
                    existing_person_arg=existing_person,
                    prefetched_combined_details=prefetched_combined,
                    prefetched_tree_data=prefetched_tree,
                )

            # Step 2f: Tally status based on _do_match result
            if status_for_this_match in ["new", "updated", "error"]:
                page_statuses[status_for_this_match] += 1
            elif status_for_this_match == "skipped":
                 logger.warning(f"Unexpected 'skipped' status from _do_match for {log_ref_short}. Logging but not counting.")
            else: # Handle unknown status string
                 logger.error(f"Unknown status '{status_for_this_match}' from _do_match for {log_ref_short}. Counting as error.")
                 page_statuses["error"] += 1

            # Step 2g: Append valid prepared data to the bulk list
            if status_for_this_match != "error" and prepared_data_for_this_match:
                prepared_bulk_data.append(prepared_data_for_this_match)
            elif status_for_this_match == "error":
                logger.error(f"Error preparing DB data for {log_ref_short}: {error_msg_for_this_match or 'Unknown error in _do_match'}")

        # Step 3: Handle unexpected exceptions during single match processing
        except Exception as inner_e:
            logger.error(f"Critical unexpected error processing {log_ref_short} in _prepare_bulk_db_data: {inner_e}", exc_info=True)
            page_statuses["error"] += 1 # Count as error for this item
        finally:
            # Step 4: Update progress bar after processing each item (regardless of outcome)
            # ***** Ensure update happens even on error/skip within try block *****
            if progress_bar:
                try: progress_bar.update(1)
                except Exception as pbar_e: logger.warning(f"Progress bar update error: {pbar_e}")

    # Step 5: Log summary and return results
    process_duration = time.time() - process_start_time
    logger.debug(f"--- Finished preparing DB data structures. Duration: {process_duration:.2f}s ---")
    return prepared_bulk_data, page_statuses
# End of _prepare_bulk_db_data


def _execute_bulk_db_operations(
    session: SqlAlchemySession,
    prepared_bulk_data: List[Dict[str, Any]],
    existing_persons_map: Dict[str, Person] # Needed to potentially map existing IDs
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
        return True # Nothing to do, considered success

    logger.debug(f"--- Starting Bulk DB Operations ({num_items} prepared items) ---")

    try:
        # Step 2: Separate data by operation type (create/update) and table
        # Person Operations
        person_creates_raw = [d["person"] for d in prepared_bulk_data if d.get("person") and d["person"]["_operation"] == "create"]
        person_updates = [d["person"] for d in prepared_bulk_data if d.get("person") and d["person"]["_operation"] == "update"]
        # DnaMatch/FamilyTree Operations (Assume create/update logic handled in _do_match prep)
        dna_match_ops = [d["dna_match"] for d in prepared_bulk_data if d.get("dna_match")]
        family_tree_ops = [d["family_tree"] for d in prepared_bulk_data if d.get("family_tree")]

        created_person_map: Dict[str, int] = {} # Maps UUID -> new Person ID

        # --- Step 3: Person Creates ---
        # De-duplicate Person Creates based on Profile ID before bulk insert
        person_creates_filtered = []
        seen_profile_ids: Set[str] = set() # Track non-null profile IDs seen in this batch
        skipped_duplicates = 0
        if person_creates_raw:
            logger.debug(f"De-duplicating {len(person_creates_raw)} raw person creates based on Profile ID...")
            for p_data in person_creates_raw:
                profile_id = p_data.get("profile_id") # Already uppercase from prep if exists
                uuid_for_log = p_data.get('uuid') # For logging skipped items
                if profile_id is None:
                    person_creates_filtered.append(p_data) # Allow creates with null profile ID
                elif profile_id not in seen_profile_ids:
                    person_creates_filtered.append(p_data)
                    seen_profile_ids.add(profile_id)
                else:
                    logger.warning(f"Skipping duplicate Person create in batch (ProfileID: {profile_id}, UUID: {uuid_for_log}).")
                    skipped_duplicates += 1
            if skipped_duplicates > 0: logger.info(f"Skipped {skipped_duplicates} duplicate person creates in this batch.")
            logger.debug(f"Proceeding with {len(person_creates_filtered)} unique person creates.")

        # Bulk Insert Persons (if any unique creates remain)
        if person_creates_filtered:
            logger.debug(f"Preparing {len(person_creates_filtered)} Person records for bulk insert...")
            # Prepare list of dictionaries for bulk_insert_mappings
            insert_data = [{k: v for k, v in p.items() if not k.startswith("_")} for p in person_creates_filtered]
            # Convert status Enum to its value for bulk insertion
            for item_data in insert_data:
                if "status" in item_data and isinstance(item_data["status"], PersonStatusEnum):
                     item_data["status"] = item_data["status"].value
            # Final check for duplicates *within the filtered list* (shouldn't happen if de-dup logic is right)
            final_profile_ids = {item.get("profile_id") for item in insert_data if item.get("profile_id")}
            if len(final_profile_ids) != sum(1 for item in insert_data if item.get("profile_id")):
                 logger.error("CRITICAL: Duplicate non-NULL profile IDs DETECTED post-filter! Aborting bulk insert.")
                 id_counts = Counter(item.get("profile_id") for item in insert_data if item.get("profile_id"))
                 duplicates = {pid: count for pid, count in id_counts.items() if count > 1}
                 logger.error(f"Duplicate Profile IDs in filtered list: {duplicates}")
                 raise IntegrityError("Duplicate profile IDs found pre-bulk insert", params=duplicates, orig=None)

            # Perform bulk insert
            logger.debug(f"Bulk inserting {len(insert_data)} Person records...")
            session.bulk_insert_mappings(Person, insert_data)
            logger.debug("Bulk insert Persons called.")

            # --- Get newly created IDs ---
            session.flush(); logger.debug("Session flushed to assign Person IDs.")
            inserted_uuids = [p_data["uuid"] for p_data in insert_data if p_data.get("uuid")]
            if inserted_uuids:
                logger.debug(f"Querying IDs for {len(inserted_uuids)} inserted UUIDs...")
                newly_inserted_persons = session.query(Person.id, Person.uuid).filter(Person.uuid.in_(inserted_uuids)).all()
                created_person_map = {p_uuid: p_id for p_id, p_uuid in newly_inserted_persons}
                logger.debug(f"Mapped {len(created_person_map)} new Person IDs.")
                if len(created_person_map) != len(inserted_uuids):
                     logger.error(f"CRITICAL: ID map count mismatch! Expected {len(inserted_uuids)}, got {len(created_person_map)}. Some IDs might be missing.")
            else: logger.warning("No UUIDs available in insert_data to query back IDs.")
        else: logger.debug("No unique Person records to bulk insert.")

        # --- Step 4: Person Updates ---
        if person_updates:
            update_mappings = []
            for p_data in person_updates:
                existing_id = p_data.get("_existing_person_id")
                if not existing_id: logger.warning(f"Skipping person update (UUID {p_data.get('uuid')}): Missing '_existing_person_id'."); continue
                update_dict = {k: v for k, v in p_data.items() if not k.startswith("_") and k not in ["uuid", "profile_id"]}
                if "status" in update_dict and isinstance(update_dict["status"], PersonStatusEnum):
                    update_dict["status"] = update_dict["status"].value
                update_dict["id"] = existing_id
                update_dict["updated_at"] = datetime.now(timezone.utc)
                if len(update_dict) > 2: update_mappings.append(update_dict)

            if update_mappings:
                logger.debug(f"Bulk updating {len(update_mappings)} Person records..."); session.bulk_update_mappings(Person, update_mappings); logger.debug("Bulk update Persons called.")
            else: logger.debug("No valid Person updates to perform.")
        else: logger.debug("No Person updates needed for this batch.")

        # --- Step 5: Create Master ID Map (for linking related records) ---
        all_person_ids_map: Dict[str, int] = created_person_map.copy()
        for p_update_data in person_updates:
            if p_update_data.get("_existing_person_id") and p_update_data.get("uuid"):
                all_person_ids_map[p_update_data["uuid"]] = p_update_data["_existing_person_id"]
        processed_uuids = {p["person"]["uuid"] for p in prepared_bulk_data if p.get("person") and p["person"].get("uuid")}
        for uuid_processed in processed_uuids:
            if uuid_processed not in all_person_ids_map and existing_persons_map.get(uuid_processed):
                all_person_ids_map[uuid_processed] = existing_persons_map[uuid_processed].id

        # --- Step 6: DnaMatch Bulk Upsert (Simplified: Insert Only Approach) ---
        if dna_match_ops:
            dna_insert_data = []
            for dna_data in dna_match_ops: # Assumes these are only for inserts now
                person_uuid = dna_data.get("uuid")
                person_id = all_person_ids_map.get(person_uuid) if person_uuid else None
                if person_id:
                    insert_dict = {k: v for k, v in dna_data.items() if not k.startswith("_")}
                    insert_dict["people_id"] = person_id
                    dna_insert_data.append(insert_dict)
                else: logger.warning(f"Skipping DNA Match op (UUID {person_uuid}): Corresponding Person ID not found.")
            if dna_insert_data:
                logger.debug(f"Bulk inserting {len(dna_insert_data)} DnaMatch records..."); session.bulk_insert_mappings(DnaMatch, dna_insert_data); logger.debug("Bulk insert DnaMatches called.")
            else: logger.debug("No valid DnaMatch records to insert.")
        else: logger.debug("No DnaMatch operations.")

        # --- Step 7: FamilyTree Bulk Upsert ---
        tree_creates = [op for op in family_tree_ops if op.get("_operation") == "create"]
        tree_updates = [op for op in family_tree_ops if op.get("_operation") == "update"]

        if tree_creates:
            tree_insert_data = []
            for tree_data in tree_creates:
                person_uuid = tree_data.get("uuid")
                person_id = all_person_ids_map.get(person_uuid) if person_uuid else None
                if person_id:
                    insert_dict = {k: v for k, v in tree_data.items() if not k.startswith("_")}
                    insert_dict["people_id"] = person_id
                    tree_insert_data.append(insert_dict)
                else: logger.warning(f"Skipping FamilyTree create op (UUID {person_uuid}): Person ID not found.")
            if tree_insert_data:
                logger.debug(f"Bulk inserting {len(tree_insert_data)} FamilyTree records..."); session.bulk_insert_mappings(FamilyTree, tree_insert_data); logger.debug("Bulk insert FamilyTrees called.")
            else: logger.debug("No valid FamilyTree records to insert.")
        else: logger.debug("No FamilyTree creates prepared.")

        if tree_updates:
            tree_update_mappings = []
            for tree_data in tree_updates:
                 existing_tree_id = tree_data.get("_existing_tree_id")
                 if not existing_tree_id: logger.warning(f"Skipping FamilyTree update op (UUID {tree_data.get('uuid')}): Missing '_existing_tree_id'."); continue
                 update_dict_tree = {k: v for k, v in tree_data.items() if not k.startswith("_") and k != "uuid"}
                 update_dict_tree["id"] = existing_tree_id
                 update_dict_tree["updated_at"] = datetime.now(timezone.utc)
                 if len(update_dict_tree) > 2: tree_update_mappings.append(update_dict_tree)
            if tree_update_mappings:
                 logger.debug(f"Bulk updating {len(tree_update_mappings)} FamilyTree records..."); session.bulk_update_mappings(FamilyTree, tree_update_mappings); logger.debug("Bulk update FamilyTrees called.")
            else: logger.debug("No valid FamilyTree updates.")
        else: logger.debug("No FamilyTree updates prepared.")

        # Step 8: Log success
        bulk_duration = time.time() - bulk_start_time
        logger.debug(f"--- Bulk DB Operations OK. Duration: {bulk_duration:.2f}s ---")
        return True

    # Step 9: Handle database errors during bulk operations
    except (IntegrityError, SQLAlchemyError) as bulk_db_err:
        logger.error(f"Bulk DB operation FAILED: {bulk_db_err}", exc_info=True)
        return False # Indicate failure (rollback handled by db_transn)
    except Exception as e:
        logger.error(f"Unexpected error during bulk DB operations: {e}", exc_info=True)
        return False # Indicate failure
# End of _execute_bulk_db_operations


def _do_batch(
    session_manager: SessionManager,
    matches_on_page: List[Dict[str, Any]],
    current_page: int,
    progress_bar: Optional[tqdm] = None
) -> Tuple[int, int, int, int]:
    """
    Processes a batch of matches fetched from a single page.
    Coordinates DB lookups, API prefetches, data preparation, and bulk DB operations.
    Updates the progress bar incrementally for both skipped and processed items.

    Args:
        session_manager: The active SessionManager instance.
        matches_on_page: List of raw match data dictionaries from `get_matches`.
        current_page: The current page number being processed (for logging).
        progress_bar: Optional tqdm progress bar instance to update.

    Returns:
        Tuple[int, int, int, int]: Counts of (new, updated, skipped, error) outcomes
                                   for the processed batch.
    """
    # Step 1: Initialization
    page_statuses: Dict[str, int] = {"new": 0, "updated": 0, "skipped": 0, "error": 0}
    num_matches_on_page = len(matches_on_page)
    my_uuid = session_manager.my_uuid
    session: Optional[SqlAlchemySession] = None # Initialize session variable

    try:
        # Step 2: Basic validation
        if not my_uuid:
            logger.error(f"_do_batch Page {current_page}: Missing my_uuid.")
            raise ValueError("Missing my_uuid")
        if not matches_on_page:
            logger.debug(f"_do_batch Page {current_page}: Empty match list.")
            return 0, 0, 0, 0

        logger.debug(f"--- Starting Batch Processing for Page {current_page} ({num_matches_on_page} matches) ---")

        # Step 3: Get DB Session for the batch
        session = session_manager.get_db_conn()
        if not session:
            logger.error(f"_do_batch Page {current_page}: Failed DB session.")
            raise SQLAlchemyError("Failed get DB session")

        # --- Data Processing Pipeline ---
        # Step 4: Lookup Existing Persons
        uuids_on_page = [m["uuid"].upper() for m in matches_on_page if m.get("uuid")]
        existing_persons_map = _lookup_existing_persons(session, uuids_on_page)

        # Step 5: Identify Fetch Candidates vs. Skipped Matches
        fetch_candidates_uuid, matches_to_process_later, skipped_count = _identify_fetch_candidates(matches_on_page, existing_persons_map)
        # Record the count of skipped items for the final summary
        page_statuses["skipped"] = skipped_count

        # --- Step 5b: Incrementally update progress bar for SKIPPED items ---
        # Iterate through the original list and update progress for items NOT needing fetch.
        if progress_bar:
            skipped_updated_in_bar = 0
            for match_data in matches_on_page:
                uuid_val = match_data.get("uuid")
                # Check if this match's UUID is NOT in the set of candidates to fetch
                if uuid_val and uuid_val not in fetch_candidates_uuid:
                    try:
                        progress_bar.update(1)
                        skipped_updated_in_bar += 1
                    except Exception as pbar_e:
                        logger.warning(f"Progress bar update error for skipped item: {pbar_e}")
            # Optional: Log if the count updated doesn't match expected skipped_count
            if skipped_updated_in_bar != skipped_count:
                 logger.warning(f"Progress bar updated {skipped_updated_in_bar} times for skipped items, but expected {skipped_count}.")
        # --- End incremental skip update ---

        # Step 6: Perform Parallel API Pre-fetches (only for candidates)
        prefetched_data = _perform_api_prefetches(session_manager, fetch_candidates_uuid, matches_to_process_later)

        # Step 7: Process Matches and Prepare Data for Bulk DB Operations
        # Note: _prepare_bulk_db_data updates the progress bar internally for each CANDIDATE item processed
        prepared_bulk_data, prep_statuses = _prepare_bulk_db_data(session, session_manager, matches_to_process_later, existing_persons_map, prefetched_data, progress_bar)
        page_statuses["new"] = prep_statuses.get("new", 0)
        page_statuses["updated"] = prep_statuses.get("updated", 0)
        page_statuses["error"] = prep_statuses.get("error", 0) # Errors during preparation

        # Step 8: Execute Bulk DB Operations within a Transaction
        if prepared_bulk_data:
            logger.debug(f"Attempting bulk DB operations for page {current_page}...")
            try:
                with db_transn(session) as sess:
                    bulk_success = _execute_bulk_db_operations(sess, prepared_bulk_data, existing_persons_map)
                    if not bulk_success:
                        # If bulk fails, adjust page statuses to reflect errors
                        logger.error(f"Bulk DB ops FAILED page {current_page}. Adjusting counts.")
                        failed_items = len(prepared_bulk_data)
                        # Add these DB errors to any preparation errors
                        page_statuses["error"] += failed_items
                        # Set new/updated to 0 as the commit failed
                        page_statuses["new"] = 0
                        page_statuses["updated"] = 0
                logger.debug(f"Transaction block finished page {current_page}.")
            except (IntegrityError, SQLAlchemyError, ValueError) as bulk_db_err:
                # Catch errors during transaction commit/operation
                logger.error(f"Bulk DB transaction FAILED page {current_page}: {bulk_db_err}", exc_info=True)
                failed_items = len(prepared_bulk_data)
                page_statuses["error"] += failed_items
                page_statuses["new"] = 0
                page_statuses["updated"] = 0
            except Exception as e:
                # Catch unexpected errors during transaction
                logger.error(f"Unexpected error during bulk DB transaction page {current_page}: {e}", exc_info=True)
                failed_items = len(prepared_bulk_data)
                page_statuses["error"] += failed_items
                page_statuses["new"] = 0
                page_statuses["updated"] = 0
        else:
            logger.debug(f"No data prepared for bulk page {current_page}.")

        # Step 9: Log page summary and return final counts
        _log_page_summary(current_page, page_statuses["new"], page_statuses["updated"], page_statuses["skipped"], page_statuses["error"])
        return page_statuses["new"], page_statuses["updated"], page_statuses["skipped"], page_statuses["error"]

    # Step 10: Handle critical exceptions during batch processing
    except (ValueError, SQLAlchemyError, ConnectionError) as critical_err: # Catch specific critical errors
        logger.critical(f"CRITICAL ERROR processing batch page {current_page}: {critical_err}", exc_info=True)
        if progress_bar:
            # Calculate remaining items to update progress bar as errors
            # Need to account for items potentially already updated (skipped or processed)
            processed_count_before_error = sum(page_statuses.values()) # Includes skips updated earlier
            # The items processed within _prepare_bulk_db_data might be partial if error happened there
            # It's complex to know exactly how many *were* updated.
            # Safest is to update by the remaining difference based on total.
            remaining_in_batch = max(0, num_matches_on_page - progress_bar.n)
            if remaining_in_batch > 0:
                try:
                    progress_bar.update(remaining_in_batch)
                except Exception as pbar_e:
                    logger.warning(f"Progress bar update error during critical exception handling: {pbar_e}")
        # Return all remaining items as errors
        return page_statuses["new"], page_statuses["updated"], page_statuses["skipped"], page_statuses["error"] + max(0, num_matches_on_page - sum(page_statuses.values()))
    except Exception as outer_batch_exc:
        logger.critical(f"CRITICAL UNHANDLED EXCEPTION processing batch page {current_page}: {outer_batch_exc}", exc_info=True)
        if progress_bar:
            # Similar logic as above for updating progress bar on error
            remaining_in_batch = max(0, num_matches_on_page - progress_bar.n)
            if remaining_in_batch > 0:
                try:
                    progress_bar.update(remaining_in_batch)
                except Exception:
                    pass # Ignore errors updating progress bar during exception handling
        # Calculate final errors accurately based on processed counts
        final_error_count = num_matches_on_page - (page_statuses["new"] + page_statuses["updated"] + page_statuses["skipped"])
        return page_statuses["new"], page_statuses["updated"], page_statuses["skipped"], max(0, final_error_count)
    # Step 11: Ensure DB session is returned
    finally:
        if session:
            session_manager.return_session(session)
        logger.debug(f"--- Finished Batch Processing for Page {current_page} ---")
# End of _do_batch

# ------------------------------------------------------------------------------
# Individual Match Processing (_do_match)
# ------------------------------------------------------------------------------

def _do_match(
    session: SqlAlchemySession,
    match: Dict[str, Any], # Raw match data from get_matches (with added predicted_relationship)
    session_manager: SessionManager,
    existing_person_arg: Optional[Person], # Prefetched existing Person object or None
    prefetched_combined_details: Optional[Dict[str, Any]], # Prefetched combined API data
    prefetched_tree_data: Optional[Dict[str, Any]], # Prefetched badge+ladder API data
) -> Tuple[Optional[Dict[str, Any]], Literal["new", "updated", "skipped", "error"], Optional[str]]:
    """
    Processes a single DNA match by comparing incoming data (from list + prefetched APIs)
    with existing database records (passed via existing_person_arg). Determines if
    a 'create', 'update', or 'skip' operation is needed for Person, DnaMatch, and
    FamilyTree records, and prepares a dictionary structure suitable for bulk operations.

    Args:
        session: The active SQLAlchemy database session.
        match: Dictionary containing data for one match from the match list API,
               potentially augmented with 'predicted_relationship'.
        session_manager: The active SessionManager instance.
        existing_person_arg: The existing Person object from the database (with
                               eager-loaded relationships), or None if the person is new.
        prefetched_combined_details: Dictionary of prefetched data from the
                                     '/details' and '/profiles/details' APIs, or None.
        prefetched_tree_data: Dictionary of prefetched data from the 'badgedetails'
                              and 'getladder' APIs, or None.

    Returns:
        A tuple containing:
        - prepared_data (Optional[Dict]): A dictionary structured for bulk operations:
          {'person': person_dict, 'dna_match': dna_dict, 'family_tree': tree_dict}.
          Values are None if no operation is needed for that table. Returns None
          if the overall status is 'skipped' or 'error'.
        - status (Literal): 'new', 'updated', 'skipped', or 'error'.
        - error_msg (Optional[str]): An error message if status is 'error'.
    """
    # Step 1: Initialization and Basic Validation
    existing_person: Optional[Person] = existing_person_arg
    # Access related records directly from the eager-loaded existing_person object
    dna_match_record: Optional[DnaMatch] = existing_person.dna_match if existing_person else None
    family_tree_record: Optional[FamilyTree] = existing_person.family_tree if existing_person else None

    match_uuid = match.get("uuid")
    match_username_raw = match.get("username")
    # Format username immediately
    match_username = format_name(match_username_raw) if match_username_raw else "Unknown"

    # Get potentially pre-populated predicted relationship
    predicted_relationship = match.get("predicted_relationship", "N/A")
    match_in_my_tree = match.get("in_my_tree", False) # From get_matches in-tree check
    log_ref_short = f"UUID={match_uuid} User='{match_username}'" # Shorter ref for logs

    # Structure to hold data prepared for bulk operations
    prepared_data_for_bulk: Dict[str, Any] = {"person": None, "dna_match": None, "family_tree": None}
    person_update_needed: bool = False # Flag specific to Person table updates
    overall_status: Literal["new", "updated", "skipped", "error"] = "error" # Default status
    error_msg: Optional[str] = None

    if not match_uuid:
        error_msg = f"_do_match Pre-check failed: Missing 'uuid' in match data: {match}"
        logger.error(error_msg)
        return None, "error", error_msg

    try:
        # Step 2: Determine if this is a new person or an existing one
        is_new_person = existing_person is None

        # Step 3: Prepare Incoming Data from APIs / Match List
        # --- Extract data primarily from prefetched combined details ---
        details_part = prefetched_combined_details or {} # Use empty dict if fetch failed
        # Profile info also comes from combined details (admin, tester IDs/names)
        profile_part = details_part

        # Extract Tester and Admin Profile IDs/Usernames (prefer details, fallback list hints)
        raw_tester_profile_id = details_part.get("tester_profile_id") or match.get("profile_id") # Match list 'profile_id' is tester
        raw_admin_profile_id = details_part.get("admin_profile_id") or match.get("administrator_profile_id_hint")
        raw_admin_username = details_part.get("admin_username") or match.get("administrator_username_hint")
        formatted_admin_username = format_name(raw_admin_username) if raw_admin_username else None
        # Ensure IDs are uppercase for storage/comparison
        tester_profile_id_upper = raw_tester_profile_id.upper() if raw_tester_profile_id else None
        admin_profile_id_upper = raw_admin_profile_id.upper() if raw_admin_profile_id else None

        # Determine which IDs/username to save based on relationship (Tester vs Admin)
        person_profile_id_to_save: Optional[str] = None
        person_admin_id_to_save: Optional[str] = None
        person_admin_username_to_save: Optional[str] = None

        # Logic based on observed scenarios (A, B, C, D from previous notes)
        if tester_profile_id_upper and admin_profile_id_upper:
            if tester_profile_id_upper == admin_profile_id_upper:
                # Case C/D: Tester IS the Admin
                # If match display name matches admin name, store as profile_id (Case D)
                if match_username and formatted_admin_username and match_username.lower() == formatted_admin_username.lower():
                    person_profile_id_to_save = tester_profile_id_upper
                    person_admin_id_to_save = None; person_admin_username_to_save = None
                else: # If names don't match, assume admin manages profile (Case C)
                    person_profile_id_to_save = None # Store primary ID as Admin
                    person_admin_id_to_save = admin_profile_id_upper
                    person_admin_username_to_save = formatted_admin_username
            else:
                # Case B: Tester and Admin are different people
                person_profile_id_to_save = tester_profile_id_upper
                person_admin_id_to_save = admin_profile_id_upper
                person_admin_username_to_save = formatted_admin_username
        elif tester_profile_id_upper: # Only tester found
            # Case A: Tester manages own kit (or admin info missing)
            person_profile_id_to_save = tester_profile_id_upper
            person_admin_id_to_save = None; person_admin_username_to_save = None
        elif admin_profile_id_upper: # Only admin found
            # Case C variation: Only admin info available
            person_profile_id_to_save = None
            person_admin_id_to_save = admin_profile_id_upper
            person_admin_username_to_save = formatted_admin_username
        # else: Neither ID found (handled by None defaults)

        # Construct message link based on the primary contactable ID
        message_target_id = person_profile_id_to_save or person_admin_id_to_save
        constructed_message_link = urljoin(config_instance.BASE_URL, f"/messaging/?p={message_target_id.upper()}") if message_target_id else None

        # Extract other Person fields from prefetched data
        birth_year_val: Optional[int] = None
        if prefetched_tree_data and prefetched_tree_data.get("their_birth_year"):
            try: birth_year_val = int(prefetched_tree_data["their_birth_year"])
            except (ValueError, TypeError): pass # Ignore invalid year

        # Extract and ensure last_logged_in is timezone-aware UTC datetime
        last_logged_in_val: Optional[datetime] = profile_part.get("last_logged_in_dt") # Already processed in _fetch_combined
        if isinstance(last_logged_in_val, datetime) and last_logged_in_val.tzinfo is None:
             last_logged_in_val = last_logged_in_val.replace(tzinfo=timezone.utc)
        elif isinstance(last_logged_in_val, datetime):
             last_logged_in_val = last_logged_in_val.astimezone(timezone.utc)

        # Create dictionary representing incoming Person data for comparison/creation
        incoming_person_data = {
            "uuid": match_uuid.upper(), # Ensure uppercase
            "profile_id": person_profile_id_to_save,
            "username": match_username, # Use formatted name
            "administrator_profile_id": person_admin_id_to_save,
            "administrator_username": person_admin_username_to_save,
            "in_my_tree": match_in_my_tree, # From get_matches
            "first_name": match.get("first_name"), # From get_matches (refined)
            "last_logged_in": last_logged_in_val, # From combined details fetch
            "contactable": bool(profile_part.get("contactable", True)), # Default True if unknown
            "gender": details_part.get("gender"), # From combined details fetch
            "message_link": constructed_message_link,
            "birth_year": birth_year_val, # From tree details fetch
            "status": PersonStatusEnum.ACTIVE, # Default for comparison/new
        }

        # Step 4: Prepare Incoming DNA Data (if needed)
        incoming_dna_data: Optional[Dict[str, Any]] = None
        needs_dna_create_or_update = False
        if dna_match_record is None:
            # Always create if no existing record
            needs_dna_create_or_update = True
        else:
            # Compare existing DB record with incoming data (prefer details API, fallback list)
            try:
                api_cm = int(match.get("cM_DNA", 0)) # Use list cM
                db_cm = dna_match_record.cM_DNA
                # Use details API segments if available, else list segments
                api_segments = int(details_part.get("shared_segments", match.get("numSharedSegments", 0)))
                db_segments = dna_match_record.shared_segments
                # Longest segment only available in details API
                api_longest_raw = details_part.get("longest_shared_segment")
                api_longest = float(api_longest_raw) if api_longest_raw is not None else None
                db_longest = dna_match_record.longest_shared_segment
                api_predicted_rel = predicted_relationship # Already fetched/set

                # Check for differences that warrant an update
                if api_cm != db_cm: needs_dna_create_or_update = True; logger.debug(f"  DNA change {log_ref_short}: cM")
                elif api_segments != db_segments: needs_dna_create_or_update = True; logger.debug(f"  DNA change {log_ref_short}: Segments")
                # Compare floats carefully for longest segment
                elif api_longest is not None and abs(api_longest - (db_longest or -1.0)) > 0.01: needs_dna_create_or_update = True; logger.debug(f"  DNA change {log_ref_short}: Longest Segment")
                elif db_longest is not None and api_longest is None: needs_dna_create_or_update = True # Handle case where API lost data?
                elif dna_match_record.predicted_relationship != api_predicted_rel: needs_dna_create_or_update = True; logger.debug(f"  DNA change {log_ref_short}: Predicted Rel")
                # Add checks for meiosis, father/mother side flags if needed
                elif bool(details_part.get("from_my_fathers_side", False)) != bool(dna_match_record.from_my_fathers_side): needs_dna_create_or_update = True; logger.debug(f"  DNA change {log_ref_short}: Father Side")
                elif bool(details_part.get("from_my_mothers_side", False)) != bool(dna_match_record.from_my_mothers_side): needs_dna_create_or_update = True; logger.debug(f"  DNA change {log_ref_short}: Mother Side")
                # Meiosis comparison
                api_meiosis = details_part.get("meiosis")
                if api_meiosis is not None and api_meiosis != dna_match_record.meiosis: needs_dna_create_or_update = True; logger.debug(f"  DNA change {log_ref_short}: Meiosis")

            except (ValueError, TypeError, AttributeError) as dna_comp_err:
                 logger.warning(f"Error comparing DNA data for {log_ref_short}: {dna_comp_err}. Assuming update needed.")
                 needs_dna_create_or_update = True

        # Build the DNA data dictionary if create/update needed
        if needs_dna_create_or_update:
            # Base dictionary for DNA data
            dna_dict_base = {
                "uuid": match_uuid.upper(),
                "compare_link": match.get("compare_link"), # From match list
                "cM_DNA": int(match.get("cM_DNA", 0)), # Use list cM
                "predicted_relationship": predicted_relationship, # Use potentially updated value
                "_operation": "create", # Flag for bulk handler (treat updates as replace for simplicity now)
            }
            # Add details from prefetched data if available
            if prefetched_combined_details:
                 dna_dict_base.update({
                    "shared_segments": details_part.get("shared_segments"),
                    "longest_shared_segment": details_part.get("longest_shared_segment"),
                    "meiosis": details_part.get("meiosis"),
                    "from_my_fathers_side": bool(details_part.get("from_my_fathers_side", False)),
                    "from_my_mothers_side": bool(details_part.get("from_my_mothers_side", False)),
                 })
            else: # Fallback if details API failed
                 logger.warning(f"{log_ref_short}: DNA needs create/update, but no combined details fetched. Using limited list data.")
                 dna_dict_base["shared_segments"] = match.get("numSharedSegments") # Use list segments

            # Remove keys with None values before adding to bulk data
            incoming_dna_data = {k: v for k, v in dna_dict_base.items() if v is not None}


        # Step 5: Prepare Incoming Tree Data (if needed)
        incoming_tree_data: Optional[Dict[str, Any]] = None
        should_have_tree = match_in_my_tree # Boolean indicating if they *should* have a tree link
        tree_operation: Literal["create", "update", "none"] = "none" # Default to no operation
        view_in_tree_link, facts_link = None, None
        their_cfpid_final = None

        # Construct links only if tree data was successfully prefetched
        if prefetched_tree_data:
            their_cfpid_final = prefetched_tree_data.get("their_cfpid")
            # Requires own tree ID to construct links
            if their_cfpid_final and session_manager.my_tree_id:
                base_person_path = f"/family-tree/person/tree/{session_manager.my_tree_id}/person/{their_cfpid_final}"
                facts_link = urljoin(config_instance.BASE_URL, f"{base_person_path}/facts")
                # Construct view link with necessary parameters
                view_params = {"cfpid": their_cfpid_final, "showMatches": "true", "sid": session_manager.my_uuid}
                base_view_url = urljoin(config_instance.BASE_URL, f"/family-tree/tree/{session_manager.my_tree_id}/family")
                view_in_tree_link = f"{base_view_url}?{urlencode(view_params)}"

        # Determine if tree operation (create/update) is needed
        if should_have_tree and family_tree_record is None:
            # Should be in tree, but no DB record exists -> Create
            tree_operation = "create"
        elif should_have_tree and family_tree_record is not None:
            # Should be in tree, and DB record exists -> Check for Updates
            if prefetched_tree_data: # Only check if we have new data to compare
                fields_to_check = [
                    ("cfpid", their_cfpid_final),
                    ("person_name_in_tree", prefetched_tree_data.get("their_firstname", "Unknown")), # Use 'their_firstname' from badge data
                    ("actual_relationship", prefetched_tree_data.get("actual_relationship")), # From ladder data
                    ("relationship_path", prefetched_tree_data.get("relationship_path")), # From ladder data
                    ("facts_link", facts_link), # Constructed link
                    ("view_in_tree_link", view_in_tree_link), # Constructed link
                ]
                for field, new_val in fields_to_check:
                    old_val = getattr(family_tree_record, field, None)
                    # Check if new value is different from old, handling None correctly
                    if (new_val != old_val):
                         tree_operation = "update"
                         logger.debug(f"  Tree change {log_ref_short}: Field '{field}'")
                         break # Found a change, no need to check further
            # else: No prefetched data, cannot determine if update needed, leave as 'none'
        elif not should_have_tree and family_tree_record is not None:
            # Should NOT be in tree, but DB record exists -> Anomaly
            logger.warning(f"{log_ref_short}: Data mismatch - API says not 'in_my_tree', but FamilyTree record exists (ID: {family_tree_record.id}). Deletion not implemented, skipping tree op.")
            tree_operation = "none" # Do nothing for now

        # Build the tree data dictionary if create/update is needed
        if tree_operation != "none":
            if prefetched_tree_data: # Can only build if data was fetched
                tree_person_name = prefetched_tree_data.get("their_firstname", "Unknown") # Use name from badge
                tree_dict_base = {
                    "uuid": match_uuid.upper(), # Include UUID for linking during bulk ops
                    "cfpid": their_cfpid_final,
                    "person_name_in_tree": tree_person_name,
                    "facts_link": facts_link,
                    "view_in_tree_link": view_in_tree_link,
                    "actual_relationship": prefetched_tree_data.get("actual_relationship"),
                    "relationship_path": prefetched_tree_data.get("relationship_path"),
                    "_operation": tree_operation, # Flag for bulk handler
                    "_existing_tree_id": (family_tree_record.id if tree_operation == "update" else None),
                }
                # Remove keys with None values before adding
                incoming_tree_data = {k: v for k, v in tree_dict_base.items() if v is not None or k in ['_operation', '_existing_tree_id']}
            else:
                # Cannot perform operation if data wasn't fetched
                logger.warning(f"{log_ref_short}: FamilyTree needs '{tree_operation}', but required tree details were not fetched. Skipping tree operation.")
                tree_operation = "none" # Reset operation type

        # Step 6: Final Assembly - Combine prepared data based on overall status (New vs Existing Person)
        if is_new_person:
            # --- New Person ---
            overall_status = "new"
            # Prepare person data with 'create' operation flag
            person_data_for_bulk = incoming_person_data.copy()
            person_data_for_bulk["_operation"] = "create"
            prepared_data_for_bulk["person"] = person_data_for_bulk
            # Include DNA data if prepared
            if incoming_dna_data: prepared_data_for_bulk["dna_match"] = incoming_dna_data
            # Include Tree data ONLY if operation is 'create' and data exists
            if incoming_tree_data and incoming_tree_data["_operation"] == "create":
                 prepared_data_for_bulk["family_tree"] = incoming_tree_data

        else: # --- Existing Person ---
            # Check if Person record itself needs update (compare incoming_person_data with existing_person)
            person_data_for_update: Dict[str, Any] = {
                "_operation": "update",
                "_existing_person_id": existing_person.id, # Include existing ID for bulk update mapping
                "uuid": match_uuid.upper(), # Include UUID for reference
            }
            person_update_needed = False # Reset flag for this person
            # Compare relevant Person fields
            for key, new_value in incoming_person_data.items():
                 if key == "uuid": continue # Skip UUID comparison for update dict
                 current_value = getattr(existing_person, key, None)
                 value_changed = False
                 value_to_set = new_value # Default to new value

                 # Apply specific comparison logic similar to create_or_update_person
                 if key == "last_logged_in":
                     current_dt_utc = (current_value.astimezone(timezone.utc).replace(microsecond=0) if isinstance(current_value, datetime) and current_value.tzinfo else (current_value.replace(tzinfo=timezone.utc, microsecond=0) if isinstance(current_value, datetime) else None))
                     new_dt_utc = (new_value.astimezone(timezone.utc).replace(microsecond=0) if isinstance(new_value, datetime) and new_value.tzinfo else (new_value.replace(tzinfo=timezone.utc, microsecond=0) if isinstance(new_value, datetime) else None))
                     if new_dt_utc != current_dt_utc: value_changed = True; value_to_set = new_value
                 elif key == "status":
                      if isinstance(new_value, PersonStatusEnum) and new_value != current_value: value_changed = True; value_to_set = new_value
                 elif key == "birth_year" and new_value is not None and current_value is None:
                      try: value_to_set = int(new_value); value_changed = True
                      except (ValueError, TypeError): logger.warning(f"Invalid birth_year '{new_value}' for update {log_ref_short}"); continue
                 elif key == "gender" and new_value is not None and current_value is None and isinstance(new_value, str) and new_value.lower() in ('f','m'):
                      value_to_set = new_value.lower(); value_changed = True
                 elif key in ("profile_id", "administrator_profile_id") and value_to_set is not None: # Ensure uppercase for ID comparison
                      if str(current_value).upper() != str(value_to_set).upper(): value_changed = True; value_to_set = str(value_to_set).upper()
                 elif isinstance(current_value, bool): # Handle booleans
                      if bool(current_value) != bool(value_to_set): value_changed = True; value_to_set = bool(value_to_set)
                 elif current_value != value_to_set: # General comparison
                      value_changed = True

                 # Add field to update dict if changed
                 if value_changed:
                      person_data_for_update[key] = value_to_set
                      person_update_needed = True

            # Assemble final bulk data for existing person
            if person_update_needed:
                prepared_data_for_bulk["person"] = person_data_for_update
            if incoming_dna_data: # Add DNA data if needed (treated as replace/create)
                prepared_data_for_bulk["dna_match"] = incoming_dna_data
            if incoming_tree_data: # Add Tree data if needed (create or update)
                prepared_data_for_bulk["family_tree"] = incoming_tree_data

            # Determine overall status for existing person
            if person_update_needed or incoming_dna_data or (incoming_tree_data and tree_operation != "none"):
                overall_status = "updated"
            else:
                overall_status = "skipped" # No changes needed anywhere

        # Step 7: Return prepared data only if an operation is needed
        data_to_return = prepared_data_for_bulk if overall_status not in ["skipped", "error"] else None
        if not any(v for v in prepared_data_for_bulk.values()) and overall_status not in ["error", "skipped"]:
             # If status is new/updated but no data dicts were actually prepared (shouldn't happen)
             logger.warning(f"Status is '{overall_status}' for {log_ref_short}, but no data prepared for bulk. Returning skipped.")
             overall_status = "skipped"
             data_to_return = None

        return data_to_return, overall_status, None # Return None for error message on success

    # Step 8: Handle unexpected exceptions during processing
    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        error_msg_for_log = f"Unexpected critical error ({error_type}) in _do_match for {log_ref_short}. Details: {error_details}"
        logger.error(error_msg_for_log, exc_info=True)
        # Return concise error message for the caller
        error_msg_return = f"Unexpected {error_type} during data prep for {log_ref_short}"
        return None, "error", error_msg_return
# End of _do_match


# ------------------------------------------------------------------------------
# API Data Acquisition Helpers (_fetch_*)
# ------------------------------------------------------------------------------

def get_matches(
    session_manager: SessionManager,
    db_session: SqlAlchemySession, # Pass DB session if needed (e.g., for lookups within get_matches)
    current_page: int = 1,
) -> Optional[Tuple[List[Dict[str, Any]], Optional[int]]]:
    """
    Fetches a single page of DNA match list data from the Ancestry API v2.
    Also fetches the 'in_my_tree' status for matches on the page via a separate API call.
    Refines the raw API data into a more structured format.

    Args:
        session_manager: The active SessionManager instance.
        db_session: The active SQLAlchemy database session (passed but unused currently).
        current_page: The page number to fetch (1-based).

    Returns:
        A tuple containing:
        - List of refined match data dictionaries for the page, or empty list if none.
        - Total number of pages available (integer), or None if retrieval fails.
        Returns None if a critical error occurs during fetching.
    """
    # Step 1: Validate Session Manager state
    if not isinstance(session_manager, SessionManager): logger.error("get_matches: Invalid SessionManager."); return None
    driver = session_manager.driver
    if not driver: logger.error("get_matches: WebDriver not initialized."); return None
    my_uuid = session_manager.my_uuid
    if not my_uuid: logger.error("get_matches: SessionManager my_uuid not set."); return None
    if not session_manager.is_sess_valid(): logger.error("get_matches: Session invalid at start."); return None

    logger.debug(f"--- Fetching Match List Page {current_page} ---")

    # Step 2: Get Specific CSRF Token from Browser Cookie (Required by Match List API)
    # This API seems to require a CSRF token set in a specific cookie, potentially
    # different from the main CSRF token used elsewhere.
    specific_csrf_token: Optional[str] = None
    csrf_token_cookie_names = ("_dnamatches-matchlistui-x-csrf-token", "_csrf") # Primary and fallback names
    try:
        # Optional: Wait briefly for match list element to ensure page JS might have set cookie
        # try: WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, MATCH_ENTRY_SELECTOR)))
        # except TimeoutException: logger.warning("Match list element not found before CSRF cookie read.")

        logger.debug(f"Attempting to read CSRF cookies: {csrf_token_cookie_names}")
        # Try getting cookies directly first
        for cookie_name in csrf_token_cookie_names:
            try:
                cookie_obj = driver.get_cookie(cookie_name)
                if cookie_obj and "value" in cookie_obj and cookie_obj["value"]:
                    # Extract token (often needs unquoting and splitting)
                    specific_csrf_token = unquote(cookie_obj["value"]).split("|")[0]
                    logger.debug(f"Read CSRF token from cookie '{cookie_name}'.")
                    break # Stop searching once found
            except NoSuchCookieException: continue # Try next name if not found
            except WebDriverException as cookie_e: logger.warning(f"WebDriver error getting cookie '{cookie_name}': {cookie_e}")
            except Exception as e: logger.error(f"Unexpected error getting cookie '{cookie_name}': {e}", exc_info=True)

        # Fallback: Use get_driver_cookies if direct method failed
        if not specific_csrf_token:
            logger.debug("CSRF token not found via get_cookie. Trying get_driver_cookies fallback...")
            all_cookies = get_driver_cookies(driver) # Fetches all cookies as dict
            if all_cookies:
                for cookie_name in csrf_token_cookie_names:
                    if cookie_name in all_cookies and all_cookies[cookie_name]:
                        specific_csrf_token = unquote(all_cookies[cookie_name]).split("|")[0]
                        logger.debug(f"Read CSRF token via fallback from '{cookie_name}'.")
                        break
            else: logger.warning("Fallback get_driver_cookies also failed to retrieve cookies.")

        if not specific_csrf_token:
            logger.error("Failed to obtain specific CSRF token required for Match List API.")
            return None # Cannot proceed without token

    except Exception as csrf_err:
         logger.error(f"Critical error during CSRF token retrieval: {csrf_err}", exc_info=True)
         return None

    # Step 3: Call Match List API
    # Construct URL for the specific page
    match_list_url = urljoin(config_instance.BASE_URL, f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={current_page}")
    # Prepare headers specific to this API call
    match_list_headers = {
        "x-csrf-token": specific_csrf_token, # Pass the specific token found
        # Other headers are added by _api_req using contextual settings for "Match List API"
    }
    logger.debug(f"Calling Match List API for page {current_page}...")
    api_response = _api_req(
        url=match_list_url,
        driver=driver,
        session_manager=session_manager,
        method="GET",
        headers=match_list_headers, # Pass specific headers needed
        use_csrf_token=False, # Already included specific token in headers
        api_description="Match List API", # Triggers special handling in _api_req
    )

    # Step 4: Process Match List API Response
    total_pages: Optional[int] = None
    match_data_list: List[Dict] = []
    if api_response is None:
        logger.warning(f"No response/error from match list API page {current_page}. Assuming empty page.")
        return [], None # Return empty list, unknown total pages
    if not isinstance(api_response, dict):
        logger.error(f"Match List API did not return dict. Page {current_page}. Type: {type(api_response)}")
        return None # Indicate critical failure
    # Extract total pages
    total_pages_raw = api_response.get("totalPages")
    if total_pages_raw is not None:
        try: total_pages = int(total_pages_raw)
        except (ValueError, TypeError): logger.warning(f"Could not parse totalPages '{total_pages_raw}'.")
    else: logger.warning("Total pages missing from match list response.")
    # Extract match list
    match_data_list = api_response.get("matchList", [])
    if not match_data_list: logger.info(f"No matches found in 'matchList' array for page {current_page}.")

    # Step 5: Filter Matches missing 'sampleId' (should be UUID)
    valid_matches_for_processing: List[Dict[str, Any]] = []
    skipped_sampleid_count = 0
    for m in match_data_list:
        # Step 5a: Check if 'sampleId' key exists and has a value
        if isinstance(m, dict) and m.get("sampleId"):
            valid_matches_for_processing.append(m)
        else:
            # Step 5b: Log and count matches skipped due to missing sampleId
            skipped_sampleid_count += 1
            # Include context in log message
            match_log_info = f"(Index: {match_data_list.index(m)}, Data: {str(m)[:100]}...)"
            logger.warning(f"Skipping raw match missing 'sampleId' on page {current_page}. {match_log_info}")
    # Step 5c: Log summary if any matches were skipped
    if skipped_sampleid_count > 0:
        logger.warning(f"Skipped {skipped_sampleid_count} raw matches on page {current_page} due to missing 'sampleId'.")
    # Step 5d: Check if any valid matches remain for further processing
    if not valid_matches_for_processing:
        logger.warning(f"No valid matches (with sampleId) found on page {current_page} to process further.")
        return [], total_pages # Return empty list, potentially valid total_pages
    # --- End Filter Matches ---

    # Step 6: Fetch In-Tree Status for Valid Matches
    sample_ids_on_page = [match["sampleId"].upper() for match in valid_matches_for_processing]
    in_tree_ids: Set[str] = set() # Stores UUIDs of matches found in user's tree
    # Use a hash of the sample IDs as part of the cache key for stability
    cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids_on_page))}"

    # Step 6a: Check cache first
    try:
        # Use ENOVAL to differentiate cache miss from stored None
        cached_in_tree = global_cache.get(cache_key_tree, default=ENOVAL, retry=True)
        if cached_in_tree is not ENOVAL:
            if isinstance(cached_in_tree, set):
                in_tree_ids = cached_in_tree
                logger.debug(f"Loaded {len(in_tree_ids)} in-tree IDs from cache for page {current_page}.")
            else:
                # Handle case where cached value is not the expected type
                logger.warning(f"Cache hit for {cache_key_tree}, but type is {type(cached_in_tree)}, not set. Refetching.")
                # Proceed to fetch from API
        else:
            logger.debug(f"Cache miss for in-tree status (Key: {cache_key_tree}). Fetching from API.")
            # Proceed to fetch from API
    except Exception as cache_read_err:
         logger.error(f"Error reading in-tree status from cache: {cache_read_err}. Fetching from API.", exc_info=True)
         # Proceed to fetch from API

    # Step 6b: Fetch from API if not cached or cache error
    if not in_tree_ids: # Fetch only if cache miss or invalid cache data
        if not session_manager.is_sess_valid():
            logger.error(f"In-Tree Status Check: Session invalid page {current_page}. Cannot fetch.")
            # Depending on requirements, could return None here or proceed without in_tree info
        else:
            # Define URL and headers for the in-tree check API
            in_tree_url = urljoin(config_instance.BASE_URL, f"discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid.upper()}")
            # This API requires CSRF token, Content-Type, Origin etc.
            in_tree_headers = {
                "X-CSRF-Token": specific_csrf_token, # Use the same CSRF token as match list
                # Other necessary headers are added contextually by _api_req
            }
            logger.debug(f"Fetching in-tree status for {len(sample_ids_on_page)} matches on page {current_page}...")
            response_in_tree = _api_req(
                url=in_tree_url,
                driver=driver,
                session_manager=session_manager,
                method="POST",
                json_data={"sampleIds": sample_ids_on_page}, # Send list of sample IDs
                headers=in_tree_headers, # Pass specific headers
                use_csrf_token=False, # Already included in headers dict
                api_description="In-Tree Status Check",
            )
            # Process the response
            if isinstance(response_in_tree, list):
                # API returns a list of sample IDs that ARE in the tree
                in_tree_ids = {item.upper() for item in response_in_tree if isinstance(item, str)}
                logger.debug(f"Fetched {len(in_tree_ids)} in-tree IDs from API for page {current_page}.")
                # Cache the result
                try:
                    global_cache.set(cache_key_tree, in_tree_ids, expire=config_instance.CACHE_TIMEOUT, retry=True)
                    logger.debug(f"Cached in-tree status result for page {current_page}.")
                except Exception as cache_write_err:
                    logger.error(f"Error writing in-tree status to cache: {cache_write_err}")
            else:
                logger.warning(f"In-Tree Status Check API failed or returned unexpected format for page {current_page}. Response: {response_in_tree}")
                # Proceed without in_tree info for this page if API fails

    # Step 7: Refine Match Data into Standardized Format
    refined_matches: List[Dict[str, Any]] = []
    logger.debug(f"Refining {len(valid_matches_for_processing)} valid matches...")
    for match_index, match_api_data in enumerate(valid_matches_for_processing):
        try:
            # Step 7a: Extract primary components from raw API data
            profile_info = match_api_data.get("matchProfile", {})
            relationship_info = match_api_data.get("relationship", {})
            sample_id = match_api_data["sampleId"] # Already validated to exist
            sample_id_upper = sample_id.upper()

            # Step 7b: Extract profile details
            profile_user_id_raw = profile_info.get("userId")
            profile_user_id_upper = str(profile_user_id_raw).upper() if profile_user_id_raw else None
            raw_display_name = profile_info.get("displayName")
            # Format the display name using helper
            match_username = format_name(raw_display_name)

            # Step 7c: Refined first name extraction
            first_name: Optional[str] = None
            if match_username and match_username != "Valued Relative":
                trimmed_username = match_username.strip()
                if trimmed_username:
                    name_parts = trimmed_username.split()
                    if name_parts: first_name = name_parts[0] # Take the first part

            # Step 7d: Extract admin hints and other profile fields
            admin_profile_id_hint = match_api_data.get("adminId") # Hint from list API
            admin_username_hint = match_api_data.get("adminName") # Hint from list API
            photo_url = profile_info.get("photoUrl", "")
            initials = profile_info.get("displayInitials", "??").upper() # Use ?? if missing
            gender = match_api_data.get("gender") # Sometimes available in list

            # Step 7e: Extract relationship details
            shared_cm = int(relationship_info.get("sharedCentimorgans", 0))
            shared_segments = int(relationship_info.get("numSharedSegments", 0))
            created_date_raw = match_api_data.get("createdDate") # Match added date

            # Step 7f: Construct compare link
            compare_link = urljoin(config_instance.BASE_URL, f"discoveryui-matches/compare/{my_uuid.upper()}/with/{sample_id_upper}")

            # Step 7g: Determine in_my_tree status from prefetched set
            is_in_tree = sample_id_upper in in_tree_ids

            # Step 7h: Assemble the refined dictionary
            refined_match_data = {
                "username": match_username, # Formatted display name
                "first_name": first_name, # Extracted first name
                "initials": initials,
                "gender": gender,
                "profile_id": profile_user_id_upper, # Tester's Profile ID
                "uuid": sample_id_upper, # Sample ID / Test ID
                "administrator_profile_id_hint": admin_profile_id_hint, # Hint only
                "administrator_username_hint": admin_username_hint, # Hint only
                "photoUrl": photo_url,
                "cM_DNA": shared_cm,
                "numSharedSegments": shared_segments, # From list view
                "compare_link": compare_link,
                "message_link": None, # To be constructed later if needed
                "in_my_tree": is_in_tree,
                "createdDate": created_date_raw, # Raw match date
                # Note: predicted_relationship added later in _do_batch using prefetched data
            }
            refined_matches.append(refined_match_data)

        # Step 8: Handle errors during refinement of a single match
        except (IndexError, KeyError, TypeError, ValueError) as refine_err:
            # Log potentially recoverable errors and skip this match
            match_uuid_err = match_api_data.get('sampleId', 'UUID_UNKNOWN')
            logger.error(f"Refinement error page {current_page}, match #{match_index+1} (UUID: {match_uuid_err}): {type(refine_err).__name__} - {refine_err}. Skipping match.", exc_info=False)
            logger.debug(f"Problematic match data during refinement: {match_api_data}")
            continue # Continue to the next match
        except Exception as critical_refine_err:
            # Log and re-raise critical unexpected errors
            match_uuid_err = match_api_data.get('sampleId', 'UUID_UNKNOWN')
            logger.error(f"CRITICAL unexpected error refining match page {current_page}, match #{match_index+1} (UUID: {match_uuid_err}): {critical_refine_err}", exc_info=True)
            logger.debug(f"Problematic match data during critical error: {match_api_data}")
            raise critical_refine_err # Re-raise to be caught by outer handler

    # Step 9: Log successful refinement count and return results
    logger.debug(f"Successfully refined {len(refined_matches)} matches on page {current_page}.")
    return refined_matches, total_pages

# End of get_matches


# Note: Combined details fetch remains the same as previously provided.
@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_combined_details(session_manager: SessionManager, match_uuid: str) -> Optional[Dict[str, Any]]:
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
    # Step 1: Validate inputs and session state
    my_uuid = session_manager.my_uuid
    if not my_uuid or not match_uuid:
        logger.warning("_fetch_combined_details: Missing my_uuid or match_uuid.")
        return None
    if not session_manager.is_sess_valid():
        logger.error(f"_fetch_combined_details: WebDriver session invalid for UUID {match_uuid}.")
        # Raise specific error type that retry_api decorator can catch
        raise ConnectionError(f"WebDriver session invalid for combined details fetch (UUID: {match_uuid})")

    # Step 2: Initialize result dictionary
    combined_data: Dict[str, Any] = {}

    # Step 3: Fetch Match Details (/details endpoint)
    details_url = urljoin(config_instance.BASE_URL, f"/discoveryui-matchesservice/api/samples/{my_uuid}/matches/{match_uuid}/details?pmparentaldata=true")
    # Referer should mimic navigating to the compare page
    details_referer = urljoin(config_instance.BASE_URL, f"/discoveryui-matches/compare/{my_uuid}/with/{match_uuid}")
    logger.debug(f"Fetching /details API for UUID {match_uuid}...")
    try:
        details_response = _api_req(
            url=details_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False, # Typically not needed for GET details
            api_description="Match Details API (Batch)",
            referer_url=details_referer,
        )
        # Process successful /details response
        if details_response and isinstance(details_response, dict):
            # Extract relevant fields into combined_data
            combined_data["admin_profile_id"] = details_response.get("adminUcdmId")
            combined_data["admin_username"] = details_response.get("adminDisplayName")
            combined_data["tester_profile_id"] = details_response.get("userId")
            combined_data["tester_username"] = details_response.get("displayName") # Redundant? Keep for now
            combined_data["tester_initials"] = details_response.get("displayInitials")
            combined_data["gender"] = details_response.get("subjectGender")
            # Extract relationship sub-dict
            relationship_part = details_response.get("relationship", {})
            combined_data["shared_segments"] = relationship_part.get("sharedSegments")
            combined_data["longest_shared_segment"] = relationship_part.get("longestSharedSegment")
            combined_data["meiosis"] = relationship_part.get("meiosis")
            # Extract parental flags
            combined_data["from_my_fathers_side"] = bool(details_response.get("fathersSide", False))
            combined_data["from_my_mothers_side"] = bool(details_response.get("mothersSide", False))
            logger.debug(f"Successfully fetched /details for UUID {match_uuid}.")
        elif isinstance(details_response, requests.Response): # Handle HTTP errors returned by _api_req
             logger.warning(f"Failed /details fetch for UUID {match_uuid}. Status: {details_response.status_code}. Skipping profile fetch.")
             return None # Fail combined fetch if details part fails
        else: # Handle None or unexpected type from _api_req
             logger.warning(f"Failed /details fetch for UUID {match_uuid} (Invalid response: {type(details_response)}). Skipping profile fetch.")
             return None # Fail combined fetch

    except ConnectionError as conn_err:
        # Handle connection errors specifically if needed, or let retry_api handle
        logger.error(f"ConnectionError fetching /details for UUID {match_uuid}: {conn_err}", exc_info=False)
        raise # Re-raise for retry_api
    except Exception as e:
        logger.error(f"Error processing /details response for UUID {match_uuid}: {e}", exc_info=True)
        if isinstance(e, requests.exceptions.RequestException): raise # Re-raise request exceptions for retry
        return None # Return None for other unexpected processing errors

    # Step 4: Fetch Profile Details (/profiles/details endpoint)
    tester_profile_id_for_api = combined_data.get("tester_profile_id") # Get ID from details response
    my_profile_id_header = session_manager.my_profile_id # Own ID needed for header

    # Initialize profile fields in combined_data to ensure they exist
    combined_data["last_logged_in_dt"] = None
    combined_data["contactable"] = False # Default to False if fetch fails

    # Check prerequisites for profile fetch
    if not tester_profile_id_for_api:
        logger.debug(f"Skipping /profiles/details fetch for {match_uuid}: Tester profile ID not found in /details.")
    elif not my_profile_id_header:
        logger.warning(f"Skipping /profiles/details fetch for {match_uuid}: Own profile ID missing for header.")
    elif not session_manager.is_sess_valid(): # Re-check session before next API call
        logger.error(f"_fetch_combined_details: WebDriver session invalid before profile fetch for {tester_profile_id_for_api}.")
        raise ConnectionError(f"WebDriver session invalid before profile fetch (Profile: {tester_profile_id_for_api})")
    else:
        # Construct profile URL and make API call
        profile_url = urljoin(config_instance.BASE_URL, f"/app-api/express/v1/profiles/details?userId={tester_profile_id_for_api.upper()}")
        # Headers added by _api_req based on description "Profile Details API (Batch)"
        logger.debug(f"Fetching /profiles/details for Profile ID {tester_profile_id_for_api} (Match UUID {match_uuid})...")
        try:
            profile_response = _api_req(
                url=profile_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                headers={}, # Contextual headers applied by _api_req
                use_csrf_token=False, # Not typically needed
                api_description="Profile Details API (Batch)",
                referer_url=details_referer, # Reuse referer from details call
            )
            # Process successful profile response
            if profile_response and isinstance(profile_response, dict):
                logger.debug(f"Successfully fetched /profiles/details for {tester_profile_id_for_api}.")
                # Extract last login date and parse into timezone-aware datetime
                last_login_str = profile_response.get("LastLoginDate")
                if last_login_str:
                    try:
                        if last_login_str.endswith("Z"): # Handle ISO format with ZULU timezone
                            dt_aware = datetime.fromisoformat(last_login_str.replace("Z", "+00:00"))
                        else: # Assume ISO format, make timezone aware (UTC)
                            dt_naive = datetime.fromisoformat(last_login_str)
                            dt_aware = dt_naive.replace(tzinfo=timezone.utc) if dt_naive.tzinfo is None else dt_naive.astimezone(timezone.utc)
                        combined_data["last_logged_in_dt"] = dt_aware # Store aware datetime object
                    except (ValueError, TypeError) as date_parse_err:
                         logger.warning(f"Could not parse LastLoginDate '{last_login_str}' for {tester_profile_id_for_api}: {date_parse_err}")
                # Extract contactable status (default to False if missing/None)
                contactable_val = profile_response.get("IsContactable")
                combined_data["contactable"] = bool(contactable_val) if contactable_val is not None else False
            elif isinstance(profile_response, requests.Response): # Handle HTTP errors
                 logger.warning(f"Failed /profiles/details fetch for {tester_profile_id_for_api}. Status: {profile_response.status_code}.")
            else: # Handle None or unexpected type
                 logger.warning(f"Failed /profiles/details fetch for {tester_profile_id_for_api} (Invalid response: {type(profile_response)}).")

        except ConnectionError as conn_err:
             # Handle connection errors during profile fetch
             logger.error(f"ConnectionError fetching /profiles/details for {tester_profile_id_for_api}: {conn_err}", exc_info=False)
             raise # Re-raise for retry_api
        except Exception as e:
             # Handle other errors during profile fetch
             logger.error(f"Error processing /profiles/details for {tester_profile_id_for_api}: {e}", exc_info=True)
             if isinstance(e, requests.exceptions.RequestException): raise # Re-raise request exceptions for retry

    # Step 5: Return the combined data dictionary
    # Return collected data even if profile fetch part failed, as details part might be useful
    return combined_data if combined_data else None # Return None only if no data collected at all
# End of _fetch_combined_details


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_batch_badge_details(session_manager: SessionManager, match_uuid: str) -> Optional[Dict[str, Any]]:
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
    # Step 1: Validate inputs and session state
    my_uuid = session_manager.my_uuid
    if not my_uuid or not match_uuid:
        logger.warning("_fetch_batch_badge_details: Missing my_uuid or match_uuid.")
        return None
    if not session_manager.is_sess_valid():
        logger.error(f"_fetch_batch_badge_details: WebDriver session invalid for UUID {match_uuid}.")
        raise ConnectionError(f"WebDriver session invalid for badge details fetch (UUID: {match_uuid})")

    # Step 2: Construct URL and Referer
    badge_url = urljoin(config_instance.BASE_URL, f"/discoveryui-matchesservice/api/samples/{my_uuid}/matches/{match_uuid}/badgedetails")
    # Referer typically the match list page for this API
    badge_referer = urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/")
    logger.debug(f"Fetching /badgedetails API for UUID {match_uuid}...")

    # Step 3: Make API call
    try:
        badge_response = _api_req(
            url=badge_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False, # Typically not needed
            api_description="Badge Details API (Batch)",
            referer_url=badge_referer,
        )

        # Step 4: Process response
        if badge_response and isinstance(badge_response, dict):
            person_badged = badge_response.get("personBadged", {}) # Extract relevant sub-dict
            if not person_badged:
                 logger.warning(f"Badge details response for UUID {match_uuid} missing 'personBadged' key.")
                 return None

            # Extract required details
            their_cfpid = person_badged.get("personId") # This is the key CFPID
            raw_firstname = person_badged.get("firstName")
            # Format name robustly - get first name part after formatting
            formatted_name_obj = format_name(raw_firstname)
            their_firstname_formatted = formatted_name_obj.split()[0] if formatted_name_obj != "Valued Relative" else "Unknown"

            # Prepare result dictionary
            result_data = {
                "their_cfpid": their_cfpid,
                "their_firstname": their_firstname_formatted, # Use formatted first name part
                "their_lastname": person_badged.get("lastName", "Unknown"), # Include last name if available
                "their_birth_year": person_badged.get("birthYear"), # Include birth year if available
            }
            logger.debug(f"Successfully fetched /badgedetails for UUID {match_uuid} (CFPID: {their_cfpid}).")
            return result_data
        elif isinstance(badge_response, requests.Response): # Handle HTTP errors
             logger.warning(f"Failed /badgedetails fetch for UUID {match_uuid}. Status: {badge_response.status_code}.")
             return None
        else: # Handle None or unexpected type
             logger.warning(f"Invalid badge details response for UUID {match_uuid}. Type: {type(badge_response)}")
             return None

    # Step 5: Handle exceptions
    except ConnectionError as conn_err:
        logger.error(f"ConnectionError fetching badge details for UUID {match_uuid}: {conn_err}", exc_info=False)
        raise # Re-raise for retry_api
    except Exception as e:
        logger.error(f"Error processing badge details for UUID {match_uuid}: {e}", exc_info=True)
        if isinstance(e, requests.exceptions.RequestException): raise # Re-raise for retry_api
        return None # Return None for other processing errors
# End of _fetch_batch_badge_details


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_batch_ladder(session_manager: SessionManager, cfpid: str, tree_id: str) -> Optional[Dict[str, Any]]:
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
    # Step 1: Validate inputs and session state
    if not cfpid or not tree_id:
        logger.warning("_fetch_batch_ladder: Missing cfpid or tree_id.")
        return None
    if not session_manager.is_sess_valid():
        logger.error(f"_fetch_batch_ladder: WebDriver session invalid for CFPID {cfpid}.")
        raise ConnectionError(f"WebDriver session invalid for ladder fetch (CFPID: {cfpid})")

    # Step 2: Construct URL and Referer
    # URL for the getladder API endpoint
    ladder_api_url = urljoin(config_instance.BASE_URL, f"family-tree/person/tree/{tree_id}/person/{cfpid}/getladder?callback=jQuery") # Assumes jQuery callback format
    # Referer should be the 'facts' page for the person being queried
    dynamic_referer = urljoin(config_instance.BASE_URL, f"family-tree/person/tree/{tree_id}/person/{cfpid}/facts")
    logger.debug(f"Fetching /getladder API for CFPID {cfpid} in Tree {tree_id}...")

    # Step 3: Make API call (expecting JSONP, force text response)
    ladder_data: Dict[str, Optional[str]] = {"actual_relationship": None, "relationship_path": None}
    # Headers added contextually by _api_req for "Get Ladder API"
    try:
        api_result = _api_req(
            url=ladder_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers={}, # Use contextual headers
            use_csrf_token=False, # Not typically needed
            api_description="Get Ladder API (Batch)",
            referer_url=dynamic_referer,
            force_text_response=True, # Crucial for parsing JSONP
        )

        # Step 4: Process response text
        if isinstance(api_result, requests.Response): # Handle HTTP errors from _api_req
            logger.warning(f"Get Ladder API call failed for CFPID {cfpid} (Status: {api_result.status_code}).")
            return None
        elif api_result is None: # Handle complete failure from _api_req
            logger.warning(f"Get Ladder API call returned None for CFPID {cfpid}.")
            return None
        elif not isinstance(api_result, str): # Ensure we got text back
            logger.warning(f"_api_req returned unexpected type '{type(api_result).__name__}' for Get Ladder API (CFPID {cfpid}).")
            return None

        response_text = api_result
        # Step 4a: Parse JSONP format (extract content within parentheses)
        match_jsonp = re.match(r"^[^(]*\((.*)\)[^)]*$", response_text, re.DOTALL | re.IGNORECASE)
        if not match_jsonp:
            logger.error(f"Could not parse JSONP format for CFPID {cfpid}. Response: {response_text[:200]}...")
            return None

        json_string = match_jsonp.group(1).strip()
        # Step 4b: Parse the extracted JSON string
        try:
            if not json_string or json_string in ('""', "''"):
                 logger.warning(f"Empty JSON content within JSONP for CFPID {cfpid}.")
                 return None # No HTML to parse
            ladder_json = json.loads(json_string)

            # Step 4c: Extract HTML content and parse with BeautifulSoup
            if isinstance(ladder_json, dict) and "html" in ladder_json:
                html_content = ladder_json["html"]
                if html_content:
                    soup = BeautifulSoup(html_content, "html.parser")

                    # --- Extract Actual Relationship ---
                    # Try primary selector first, then fallback
                    rel_elem = soup.select_one("ul.textCenter > li:first-child > i > b") or \
                               soup.select_one("ul.textCenter > li > i > b") # Fallback
                    if rel_elem:
                        raw_relationship = rel_elem.get_text(strip=True)
                        # Apply title casing and fix ordinal suffixes
                        ladder_data["actual_relationship"] = ordinal_case(raw_relationship.title())
                    else: logger.warning(f"Could not extract actual_relationship for CFPID {cfpid}")

                    # --- Extract Relationship Path ---
                    # Select list items that are *not* the downward arrow dividers
                    path_items = soup.select('ul.textCenter > li:not([class*="iconArrowDown"])')
                    path_list = []
                    num_items = len(path_items)
                    for i, item in enumerate(path_items):
                        name_text, desc_text = "", ""
                        # Find name (within link <a> or bold <b> tags)
                        name_container = item.find("a") or item.find("b")
                        if name_container: name_text = format_name(name_container.get_text(strip=True).replace('"', "'"))
                        # Find description (within italic <i> tags, skip first item 'Me')
                        if i > 0:
                            desc_element = item.find("i")
                            if desc_element:
                                raw_desc_full = desc_element.get_text(strip=True).replace('"', "'")
                                # Handle "You are the..." case specifically for the last item
                                if i == num_items - 1 and raw_desc_full.lower().startswith("you are the "):
                                    desc_text = format_name(raw_desc_full[len("You are the "):].strip())
                                else: # Try parsing "Relation of Person" format
                                    match_rel = re.match(r"^(.*?)\s+of\s+(.*)$", raw_desc_full, re.IGNORECASE)
                                    if match_rel: desc_text = f"{match_rel.group(1).strip().capitalize()} of {format_name(match_rel.group(2).strip())}"
                                    else: desc_text = format_name(raw_desc_full) # Fallback

                        # Add formatted entry to path list if name exists
                        if name_text: path_list.append(f"{name_text} ({desc_text})" if desc_text else name_text)

                    # Join path items with arrows
                    if path_list: ladder_data["relationship_path"] = "\n↓\n".join(path_list)
                    else: logger.warning(f"Could not construct relationship_path for CFPID {cfpid}.")

                    logger.debug(f"Successfully parsed ladder details for CFPID {cfpid}.")
                    return ladder_data # Return extracted data
                else: # HTML content was empty
                    logger.warning(f"Empty HTML in getladder response for CFPID {cfpid}.")
                    return None
            else: # JSON structure missing 'html' key
                logger.warning(f"Missing 'html' key in getladder JSON for CFPID {cfpid}. JSON: {ladder_json}")
                return None
        except json.JSONDecodeError as inner_json_err:
            logger.error(f"Failed to decode JSONP content for CFPID {cfpid}: {inner_json_err}")
            logger.debug(f"JSON string causing decode error: '{json_string[:200]}...'")
            return None

    # Step 5: Handle exceptions
    except ConnectionError as conn_err:
        logger.error(f"ConnectionError fetching ladder for CFPID {cfpid}: {conn_err}", exc_info=False)
        raise # Re-raise for retry_api
    except Exception as e:
        logger.error(f"Error processing ladder for CFPID {cfpid}: {e}", exc_info=True)
        if isinstance(e, requests.exceptions.RequestException): raise # Re-raise for retry_api
        return None # Return None for other unexpected errors
# End of _fetch_batch_ladder


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError, cloudscraper.exceptions.CloudflareException))
def _fetch_batch_relationship_prob(session_manager: SessionManager, match_uuid: str, max_labels_param: int = 2) -> Optional[str]:
    """
    Fetches the predicted relationship probability distribution for a match using
    the shared cloudscraper instance to potentially bypass Cloudflare challenges.

    Args:
        session_manager: The active SessionManager instance.
        match_uuid: The UUID (Sample ID) of the target match.
        max_labels_param: The maximum number of relationship labels to include
                          in the result string (e.g., 2 for "1st or 2nd Cousin").

    Returns:
        A formatted string like "1st cousin [95.5%]" or "Distant relationship?",
        or None if the fetch fails. Returns "N/A (Error...)" on specific failures.
    """
    # Step 1: Validate inputs and session state
    my_uuid = session_manager.my_uuid
    driver = session_manager.driver # Needed for cookie sync
    scraper = session_manager.scraper # Use shared scraper instance

    if not my_uuid or not match_uuid:
        logger.warning("_fetch_batch_relationship_prob: Missing my_uuid or match_uuid.")
        return "N/A (Error - Missing IDs)"
    if not scraper:
        logger.error("_fetch_batch_relationship_prob: SessionManager scraper not initialized.")
        raise ConnectionError("SessionManager scraper not initialized.") # Raise for retry
    # Check driver state before attempting cookie sync
    if not driver or not session_manager.is_sess_valid():
        logger.error(f"_fetch_batch_relationship_prob: Driver/session invalid for UUID {match_uuid}.")
        raise ConnectionError(f"WebDriver session invalid for relationship probability fetch (UUID: {match_uuid})")

    # Step 2: Prepare URL and Headers
    my_uuid_upper = my_uuid.upper()
    sample_id_upper = match_uuid.upper()
    rel_url = urljoin(config_instance.BASE_URL, f"discoveryui-matches/parents/list/api/matchProbabilityData/{my_uuid_upper}/{sample_id_upper}")
    referer_url = urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/")
    api_description = "Match Probability API (Cloudscraper)"
    # Basic headers, CSRF/other common headers handled below or by scraper/config
    rel_headers = {
        "Accept": "application/json",
        "Referer": referer_url,
        "Origin": config_instance.BASE_URL.rstrip("/"),
        # Add User-Agent as scraper might not always mimic browser perfectly
        "User-Agent": random.choice(config_instance.USER_AGENTS),
    }

    # Step 3: Synchronize Cookies and Get CSRF Token
    # Sync latest cookies from WebDriver to the shared cloudscraper session
    # Get CSRF token directly from WebDriver cookies for this specific API call
    csrf_token_val: Optional[str] = None
    csrf_cookie_names = ("_dnamatches-matchlistui-x-csrf-token", "_csrf")
    try:
        driver_cookies_list = driver.get_cookies()
        if driver_cookies_list:
            logger.debug(f"Syncing {len(driver_cookies_list)} WebDriver cookies to shared scraper for {api_description}...")
            # Clear scraper's current cookies and add fresh ones
            if hasattr(scraper, 'cookies') and isinstance(scraper.cookies, RequestsCookieJar):
                scraper.cookies.clear()
                for cookie in driver_cookies_list:
                    if "name" in cookie and "value" in cookie:
                        scraper.cookies.set(
                            cookie["name"], cookie["value"],
                            domain=cookie.get("domain"), path=cookie.get("path", "/"),
                            secure=cookie.get("secure", False)
                        )
            else: logger.warning("Scraper cookie jar not accessible for update.")

            # Extract CSRF token from the fetched driver cookies
            driver_cookies_dict = {c["name"]: c["value"] for c in driver_cookies_list if "name" in c and "value" in c}
            for name in csrf_cookie_names:
                if name in driver_cookies_dict and driver_cookies_dict[name]:
                    csrf_token_val = unquote(driver_cookies_dict[name]).split("|")[0]
                    rel_headers["X-CSRF-Token"] = csrf_token_val # Add token to headers
                    logger.debug(f"Using fresh CSRF token '{name}' from driver cookies for {api_description}.")
                    break
        else:
            logger.warning("driver.get_cookies() returned empty list during {api_description} prep.")
    except WebDriverException as csrf_wd_e:
        logger.warning(f"WebDriverException getting/setting cookies for {api_description}: {csrf_wd_e}")
        raise ConnectionError(f"WebDriver error getting/setting cookies for CSRF: {csrf_wd_e}") from csrf_wd_e
    except Exception as csrf_e:
        logger.warning(f"Error processing cookies/CSRF for {api_description}: {csrf_e}")

    # Fallback CSRF handling (if not found in fresh cookies)
    if "X-CSRF-Token" not in rel_headers:
        if session_manager.csrf_token:
             logger.warning(f"{api_description}: Using potentially stale CSRF from SessionManager.")
             rel_headers["X-CSRF-Token"] = session_manager.csrf_token
        else:
             logger.error(f"{api_description}: Failed to add CSRF token to headers (fresh and fallback failed).")
             return "N/A (Error - Missing CSRF)"

    # Step 4: Make the API call using the shared cloudscraper instance
    try:
        logger.debug(f"Making {api_description} POST request to {rel_url} using shared scraper...")
        response_rel = scraper.post(
            rel_url,
            headers=rel_headers,
            json={}, # API expects empty JSON payload
            allow_redirects=False, # Don't follow redirects
            timeout=selenium_config.API_TIMEOUT,
        )
        logger.debug(f"<-- {api_description} Response Status: {response_rel.status_code} {response_rel.reason}")

        # Step 5: Process the response
        if not response_rel.ok:
            # Handle HTTP errors
            status_code = response_rel.status_code
            logger.warning(f"{api_description} failed for {sample_id_upper}. Status: {status_code}, Reason: {response_rel.reason}")
            try: logger.debug(f"  Response Body: {response_rel.text[:500]}") # Log error body
            except Exception: pass
            response_rel.raise_for_status() # Raise HTTPError for retry_api decorator
            return f"N/A (HTTP Error {status_code})" # Fallback return if raise_for_status doesn't trigger retry

        # Process successful (2xx) response
        try:
            if not response_rel.content:
                 logger.warning(f"{api_description}: OK ({response_rel.status_code}), but response body EMPTY.")
                 return "N/A (Empty Response)"
            data = response_rel.json() # Parse JSON

            # Validate response structure
            if "matchProbabilityToSampleId" not in data:
                logger.warning(f"Invalid data structure from {api_description} for {sample_id_upper}. Resp: {data}")
                return "N/A (Invalid Data Structure)"

            # Extract predictions
            prob_data = data["matchProbabilityToSampleId"]
            predictions = prob_data.get("relationships", {}).get("predictions", [])
            if not predictions:
                logger.debug(f"No relationship predictions found for {sample_id_upper}. Marking as Distant.")
                return "Distant relationship?" # Default if no predictions

            # Filter valid predictions and find the one with highest probability
            valid_preds = [p for p in predictions if isinstance(p, dict) and "distributionProbability" in p and "pathsToMatch" in p]
            if not valid_preds:
                logger.warning(f"No valid prediction paths found for {sample_id_upper}.")
                return "N/A (No Valid Paths)"

            best_pred = max(valid_preds, key=lambda x: x.get("distributionProbability", 0.0))
            top_prob = best_pred.get("distributionProbability", 0.0) * 100.0 # Convert to percentage
            paths = best_pred.get("pathsToMatch", [])
            labels = [p.get("label") for p in paths if isinstance(p, dict) and p.get("label")]

            if not labels:
                logger.warning(f"Prediction found for {sample_id_upper}, but no labels in paths.")
                return f"N/A (No Labels) [{top_prob:.1f}%]"

            # Format result string with top labels and probability
            final_labels = labels[:max_labels_param] # Apply label limit
            relationship_str = " or ".join(map(str, final_labels))
            return f"{relationship_str} [{top_prob:.1f}%]"

        except json.JSONDecodeError as json_err:
            logger.error(f"{api_description}: OK ({response_rel.status_code}), but JSON decode FAILED: {json_err}")
            logger.debug(f"Response text: {response_rel.text[:500]}")
            raise RequestException("JSONDecodeError") from json_err # Trigger retry
        except Exception as e:
            logger.error(f"{api_description}: Error processing successful response for {sample_id_upper}: {e}", exc_info=True)
            raise RequestException("Response Processing Error") from e # Trigger retry

    # Step 6: Handle exceptions caught by retry_api decorator or others
    except cloudscraper.exceptions.CloudflareException as cf_e:
        logger.error(f"{api_description}: Cloudflare challenge failed for {sample_id_upper}: {cf_e}")
        raise # Let retry_api handle
    except requests.exceptions.RequestException as req_e:
        logger.error(f"{api_description}: RequestException for {sample_id_upper}: {req_e}")
        raise # Let retry_api handle
    except Exception as e:
        logger.error(f"{api_description}: Unexpected error for {sample_id_upper}: {type(e).__name__} - {e}", exc_info=True)
        raise RequestException(f"Unexpected Fetch Error: {type(e).__name__}") from e # Trigger retry if possible
# End of _fetch_batch_relationship_prob


# ------------------------------------------------------------------------------
# Utility & Helper Functions
# ------------------------------------------------------------------------------

def _log_page_summary(page: int, page_new: int, page_updated: int, page_skipped: int, page_errors: int):
    """Logs a summary of processed matches for a single page."""
    # Step 1: Log header for the page summary
    logger.debug(f"---- Page {page} Batch Summary ----")
    # Step 2: Log counts for each outcome category
    logger.debug(f"  New Person/Data: {page_new}")
    logger.debug(f"  Updated Person/Data: {page_updated}")
    logger.debug(f"  Skipped (No Change): {page_skipped}")
    logger.debug(f"  Errors during Prep: {page_errors}")
    # Step 3: Log footer
    logger.debug("---------------------------\n") # Add newline for readability
# End of _log_page_summary

def _log_coord_summary(total_pages_processed: int, total_new: int, total_updated: int, total_skipped: int, total_errors: int):
    """Logs the final summary of the entire coord (match gathering) execution."""
    # Step 1: Log header for the final summary
    logger.info("---- Gather Matches Final Summary ----")
    # Step 2: Log cumulative counts across all processed pages
    logger.info(f"  Total Pages Processed: {total_pages_processed}")
    logger.info(f"  Total New Added:     {total_new}")
    logger.info(f"  Total Updated:       {total_updated}")
    logger.info(f"  Total Skipped:       {total_skipped}")
    logger.info(f"  Total Errors:        {total_errors}")
    # Step 3: Log footer
    logger.info("------------------------------------\n") # Add newline for readability
# End of _log_coord_summary

def _adjust_delay(session_manager: SessionManager, current_page: int):
    """
    Adjusts the dynamic rate limiter's delay based on throttling feedback
    received during the processing of the current page.

    Args:
        session_manager: The active SessionManager instance.
        current_page: The page number just processed (for logging context).
    """
    # Step 1: Check if the rate limiter was recently throttled
    if session_manager.dynamic_rate_limiter.is_throttled():
        # If throttled, the delay was already increased. Log this fact.
        logger.debug(f"Rate limiter was throttled during processing before/during page {current_page}. Delay remains increased.")
        # Note: The `increase_delay` method sets the `last_throttled` flag.
    else:
        # Step 2: If not throttled, attempt to decrease the delay gradually
        # Store previous delay for logging comparison
        previous_delay = session_manager.dynamic_rate_limiter.current_delay
        session_manager.dynamic_rate_limiter.decrease_delay() # Call decrease method
        new_delay = session_manager.dynamic_rate_limiter.current_delay
        # Step 3: Log the decrease only if it was significant and above initial floor
        if (abs(previous_delay - new_delay) > 0.01 and # Noticeable change
            new_delay > config_instance.INITIAL_DELAY): # Still above base minimum
             logger.debug(f"Decreased rate limit base delay to {new_delay:.2f}s after page {current_page}.")
        # The `decrease_delay` method also resets the `last_throttled` flag.
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
    # Step 1: Validate session state and UUID presence
    if not session_manager or not session_manager.is_sess_valid() or not session_manager.my_uuid:
        logger.error("nav_to_list: Session invalid or UUID missing.")
        return False

    # Step 2: Construct the target URL
    my_uuid = session_manager.my_uuid
    target_url = urljoin(config_instance.BASE_URL, f"discoveryui-matches/list/{my_uuid}")
    logger.debug(f"Navigating to specific match list URL: {target_url}")

    # Step 3: Call the generic navigation function
    # Wait for a specific element known to be on the match list page
    success = nav_to_page(
        driver=session_manager.driver,
        url=target_url,
        selector=MATCH_ENTRY_SELECTOR, # Wait for first match entry card
        session_manager=session_manager,
    )

    # Step 4: Log outcome and perform basic URL check
    if success:
        try:
            current_url = session_manager.driver.current_url
            # Check if we actually landed on the expected URL (or close enough)
            if not current_url.startswith(target_url):
                logger.warning(f"Navigation successful (element found), but final URL unexpected: {current_url}")
            else:
                logger.debug("Successfully navigated to specific matches list page.")
        except Exception as e:
            logger.warning(f"Could not verify final URL after nav_to_list success: {e}")
    else:
        logger.error("Failed navigation to specific matches list page using nav_to_page.")

    # Step 5: Return success status
    return success
# End of nav_to_list

# --- End of action6_gather.py ---
