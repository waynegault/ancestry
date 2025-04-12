#!/usr/bin/env python3

# action6_gather.py
# V14.34: Removed faulty cachetools fallback initialization.

# Standard library imports (alphabetical)
import contextlib  # Needed for db_transn context manager
import json
import logging
import math
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
import traceback

# Third-party imports (alphabetical by package)
import cloudscraper
import requests
from bs4 import BeautifulSoup, Tag

from cachetools import Cache
from diskcache.core import ENOVAL 
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
from sqlalchemy.orm import Session, Session as SqlAlchemySession, joinedload
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Local application imports (alphabetical by module)
from cache import cache as global_cache 
from cache import cache_result

from config import config_instance, selenium_config
from database import (
    DnaMatch,
    FamilyTree,
    Person,
    PersonStatusEnum,
    db_transn,
)
from my_selectors import *
from utils import (
    DynamicRateLimiter,
    SessionManager,
    _api_req,
    format_name,
    get_driver_cookies,
    make_newrelic,
    make_traceparent,
    make_tracestate,
    make_ube,
    nav_to_page,
    ordinal_case,
    retry,
    retry_api,
    time_wait,
    urljoin,
)



#################################################################################
# 1. Setup & Verification
#################################################################################

# Initialize logging
logger = logging.getLogger("logger")

# estimated matches per page for progress bar
MATCHES_PER_PAGE = 20

# Configurable settings (moved from hardcoded values)
DB_ERROR_PAGE_THRESHOLD = config_instance._get_int_env(
    "DB_ERROR_PAGE_THRESHOLD", 10
)  # Allow 10 page errors by default
THREAD_POOL_WORKERS = config_instance._get_int_env(
    "GATHER_THREAD_POOL_WORKERS", 3
)  # Default to 3 workers


#################################################################################
# 2. Core Orchestration
#################################################################################


def coord(session_manager: SessionManager, config_instance, start: int = 1) -> bool:
    """
    V14.34: Gathers DNA matches, processing page-by-page.
    - Uses updated session state checks (driver_live, session_ready).
    - Adds retry for getting DB session.
    - Uses configurable DB error threshold.
    - Fixes cache object issue.
    """
    driver = session_manager.driver
    # --- Use updated session state checks ---
    if (
        not driver
        or not session_manager.driver_live
        or not session_manager.session_ready
    ):
        logger.error(
            "WebDriver not initialized, driver not live, or session not ready. Exiting coord."
        )
        return False
    # --- End state check update ---

    total_new, total_updated, total_skipped, total_errors = 0, 0, 0, 0
    total_pages_processed = 0
    progress_bar = None
    final_success = True
    my_uuid = session_manager.my_uuid
    if not my_uuid:
        logger.error("Failed to retrieve my_uuid from session_manager. Exiting coord.")
        return False
    target_matches_url_base = urljoin(
        config_instance.BASE_URL, f"discoveryui-matches/list/{my_uuid}"
    )
    total_pages: Optional[int] = None
    last_page: Optional[int] = None
    total_pages_to_process_in_run = 0
    matches_on_page: List[Dict[str, Any]] = []
    total_matches_estimate = 0
    db_connection_errors = 0  # Counter for consecutive DB connection errors

    try:
        if not isinstance(start, int) or start <= 0:
            logger.warning(
                f"Invalid start parameter '{start}'. Using default start page 1."
            )
            start_page = 1
        else:
            start_page = start
    except Exception:
        logger.exception(  # Log traceback for unexpected errors
            f"Error processing start parameter '{start}'. Using default start page 1."
        )
        start_page = 1

    try:
        logger.debug("11. Ensure we are on the DNA matches page...")
        try:
            current_url = driver.current_url
            if not current_url.startswith(target_matches_url_base):
                logger.debug("Navigating to DNA matches page.")
                if not nav_to_list(session_manager):
                    logger.error(
                        "Failed to navigate to DNA match list page. Exiting coord."
                    )
                    return False
                else:
                    logger.debug("Successfully navigated to DNA matches page.\n")
            else:
                logger.debug(f"Already on correct DNA matches page: {current_url}.\n")
        except WebDriverException as nav_e:
            logger.error(f"WebDriver error checking/navigating: {nav_e}", exc_info=True)
            return False

        logger.debug(f"Fetching initial page {start_page} to determine total pages...")
        # --- Retry getting initial DB connection ---
        db_session_for_page = None
        for retry_attempt in range(3):  # Retry up to 3 times for DB session
            db_session_for_page = session_manager.get_db_conn()
            if db_session_for_page:
                break
            logger.warning(
                f"Could not get DB session for initial page fetch (attempt {retry_attempt + 1}/3). Retrying in 5s..."
            )
            time.sleep(5)
        # --- End Retry ---

        fetched_total_pages = None
        if not db_session_for_page:
            logger.error(
                f"Could not get DB session for initial page fetch after retries. Aborting."
            )
            return False

        try:
            if not session_manager.is_sess_valid():
                logger.critical(
                    f"WebDriver session invalid before initial get_matches. Aborting run."
                )
                return False
            # NOTE: Pass the actual global_cache object if get_matches needs it,
            # but it shouldn't need it directly if cache_result is used inside get_matches.
            # Pass db_session instead.
            result = get_matches(session_manager, db_session_for_page, start_page)
            if result is None:
                matches_on_page, fetched_total_pages = [], None
            else:
                matches_on_page, fetched_total_pages = result
            db_connection_errors = 0  # Reset counter on success
        except ConnectionError as init_conn_e:
            logger.critical(
                f"ConnectionError during initial get_matches: {init_conn_e}. Aborting.",
                exc_info=False,
            )
            final_success = False
        except Exception as get_match_err:
            logger.error(
                f"Error during initial get_matches call on page {start_page}: {get_match_err}",
                exc_info=True,
            )
            final_success = False
        finally:
            if db_session_for_page:
                session_manager.return_session(db_session_for_page)

        if not final_success:  # If initial fetch failed critically
            return False

        if fetched_total_pages is None:
            logger.error(
                "Failed to retrieve total_pages on initial fetch (get_matches returned None). Aborting."
            )
            return False
        total_pages = fetched_total_pages
        logger.info(f"Total pages found: {total_pages}\n")

        max_pages_config = config_instance.MAX_PAGES
        pages_to_process_config = (
            min(max_pages_config, total_pages) if max_pages_config != 0 else total_pages
        )
        last_page = min(start_page + pages_to_process_config - 1, total_pages)
        total_pages_to_process_in_run = max(0, last_page - start_page + 1)

        if total_pages_to_process_in_run <= 0:
            logger.info("No pages to process based on start/end page calculation.")
            return True
        total_matches_estimate = total_pages_to_process_in_run * MATCHES_PER_PAGE
        logger.info(
            f"Processing {total_pages_to_process_in_run} pages (approx. {total_matches_estimate} matches) from {start_page} to {last_page}.\n"
        )

        with logging_redirect_tqdm():
            progress_bar = tqdm(
                total=total_matches_estimate,
                desc="Gathering Matches",
                unit=" match",
                ncols=100,
                bar_format="{percentage:<3.0f}%|{bar}|",
                file=sys.stderr,
                leave=True,
            )
            logger.debug("Processing matches page by page inside redirect context...")
            current_page_num = start_page
            while True:
                if current_page_num > last_page:
                    logger.debug(
                        f"Current page {current_page_num} exceeds processing limit {last_page}. Stopping."
                    )
                    break

                if not session_manager.is_sess_valid():
                    logger.critical(
                        f"WebDriver session invalid/unreachable before processing page {current_page_num}. Likely due to resource exhaustion (e.g., WinError 10055). Aborting run."
                    )
                    final_success = False
                    remaining_pages = max(0, last_page - current_page_num + 1)
                    total_errors += remaining_pages * MATCHES_PER_PAGE
                    if progress_bar:
                        remaining_matches_estimate = max(
                            0, progress_bar.total - progress_bar.n
                        )
                        if remaining_matches_estimate > 0:
                            logger.info(
                                f"Updating progress bar by {remaining_matches_estimate} for aborted run."
                            )
                            progress_bar.update(remaining_matches_estimate)
                    break

                if not (current_page_num == start_page and matches_on_page):
                    db_session_for_page = None
                    # --- Retry getting DB connection ---
                    for retry_attempt in range(3):
                        db_session_for_page = session_manager.get_db_conn()
                        if db_session_for_page:
                            db_connection_errors = 0  # Reset error count on success
                            break
                        logger.warning(
                            f"Could not get DB session for page {current_page_num} (attempt {retry_attempt + 1}/3). Retrying in 5s..."
                        )
                        time.sleep(5)
                    # --- End Retry ---

                    if not db_session_for_page:
                        db_connection_errors += 1
                        logger.error(
                            f"Could not get DB session for page {current_page_num} after retries. Skipping page."
                        )
                        total_errors += MATCHES_PER_PAGE
                        if progress_bar:  # Increment error count in progress bar
                            progress_bar.update(
                                MATCHES_PER_PAGE
                            )  # Assume full page error
                        # Check configurable threshold
                        if db_connection_errors >= DB_ERROR_PAGE_THRESHOLD:
                            logger.critical(
                                f"Aborting run due to persistent DB connection failures ({db_connection_errors} consecutive failed pages)."
                            )
                            final_success = False
                            break
                        current_page_num += 1
                        continue

                    try:
                        if not session_manager.is_sess_valid():
                            logger.critical(
                                f"WebDriver session became invalid just before get_matches for page {current_page_num}. Aborting run."
                            )
                            final_success = False
                            remaining_pages = max(0, last_page - current_page_num + 1)
                            total_errors += remaining_pages * MATCHES_PER_PAGE
                            if progress_bar:
                                remaining_matches_estimate = max(
                                    0, progress_bar.total - progress_bar.n
                                )
                                if remaining_matches_estimate > 0:
                                    logger.info(
                                        f"Updating progress bar by {remaining_matches_estimate} for aborted run."
                                    )
                                    progress_bar.update(remaining_matches_estimate)
                            break
                        matches_on_page, _ = get_matches(
                            session_manager, db_session_for_page, current_page_num
                        )
                    except ConnectionError as conn_e:
                        logger.error(
                            f"ConnectionError getting matches for page {current_page_num}: {conn_e}",
                            exc_info=False,
                        )
                        total_errors += MATCHES_PER_PAGE
                        if progress_bar:
                            progress_bar.update(MATCHES_PER_PAGE)
                        time.sleep(5)
                        matches_on_page = []  # Clear page data on error
                        current_page_num += 1
                        continue  # Skip processing this page
                    except Exception as get_match_e:
                        logger.error(
                            f"Error getting matches for page {current_page_num}: {get_match_e}",
                            exc_info=True,
                        )
                        total_errors += MATCHES_PER_PAGE
                        if progress_bar:
                            progress_bar.update(MATCHES_PER_PAGE)
                        time.sleep(5)
                        matches_on_page = []  # Clear page data on error
                        current_page_num += 1
                        continue  # Skip processing this page
                    finally:
                        if db_session_for_page:
                            session_manager.return_session(db_session_for_page)

                if matches_on_page is None:
                    logger.warning(
                        f"get_matches returned None for page {current_page_num}. Skipping page processing."
                    )
                    total_errors += MATCHES_PER_PAGE
                    if progress_bar:
                        progress_bar.update(MATCHES_PER_PAGE)
                    time.sleep(2)
                    matches_on_page = []  # Clear page data
                    current_page_num += 1
                    continue
                elif not matches_on_page:
                    logger.info(f"No matches found on page {current_page_num}.")
                    # Only update progress bar if it's not the very first attempt where matches_on_page might be empty initially
                    if progress_bar and not (
                        current_page_num == start_page and total_pages_processed == 0
                    ):
                        # Estimate update based on expected matches per page
                        progress_bar.update(MATCHES_PER_PAGE)
                    time.sleep(1)
                    matches_on_page = []  # Clear page data
                    current_page_num += 1
                    continue

                # --- Call the refactored batch processing function ---
                page_new, page_updated, page_skipped, page_errors = _do_batch(
                    session_manager=session_manager,
                    matches_on_page=matches_on_page,
                    current_page=current_page_num,
                    progress_bar=progress_bar,
                )
                # --- End call ---

                total_new += page_new
                total_updated += page_updated
                total_skipped += page_skipped
                total_errors += page_errors
                total_pages_processed += 1

                if progress_bar:
                    # Update postfix AFTER processing the batch
                    progress_bar.set_postfix(
                        New=total_new,
                        Upd=total_updated,
                        Skip=total_skipped,
                        Err=total_errors,
                        refresh=True,
                    )
                _adjust_delay(session_manager, current_page_num)
                inter_page_delay = (
                    session_manager.dynamic_rate_limiter.wait()
                )  # Use wait() for delay
                logger.debug(f"Rate limit inter-page delay: {inter_page_delay:.2f}s")

                # --- IMPORTANT: Clear matches_on_page for the next iteration ---
                matches_on_page = []
                current_page_num += 1
                # --- End Clear ---

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt...")
        final_success = False
    except ConnectionError as coord_conn_err:
        logger.critical(
            f"ConnectionError during coord execution: {coord_conn_err}", exc_info=True
        )
        final_success = False
    except Exception as e:
        logger.error(f"Critical error during coord execution: {e}", exc_info=True)
        final_success = False
    finally:
        logger.debug("Entering finally block in coord...")
        if progress_bar:
            # Ensure final stats are shown
            progress_bar.set_postfix(
                New=total_new,
                Upd=total_updated,
                Skip=total_skipped,
                Err=total_errors,
                refresh=True,
            )
            progress_bar.close()
            print("", file=sys.stderr)  # Newline after final bar

        _log_coord_summary(
            total_pages_processed, total_new, total_updated, total_skipped, total_errors
        )

        exc_info = sys.exc_info()
        if exc_info[0] is KeyboardInterrupt:
            logger.info("Re-raising KeyboardInterrupt after cleanup.")
            if exc_info[1] is not None:
                raise exc_info[1].with_traceback(exc_info[2])

        logger.debug("Exiting finally block in coord.")

    return final_success


# end of coord


#################################################################################
# 3. Batch Processing Logic (_do_batch and Helpers)
#################################################################################


def _lookup_existing_persons(
    session: SqlAlchemySession, uuids_on_page: List[str]
) -> Dict[str, Person]:
    """Queries the database for existing Person records based on UUIDs."""
    existing_persons_map: Dict[str, Person] = {}
    if not uuids_on_page:
        return existing_persons_map

    try:
        existing_persons = (
            session.query(Person)
            .options(joinedload(Person.dna_match), joinedload(Person.family_tree))
            .filter(Person.uuid.in_(uuids_on_page))
            .all()
        )
        existing_persons_map = {person.uuid: person for person in existing_persons}
        logger.debug(
            f"Found {len(existing_persons_map)} existing Person records for this batch."
        )
    except SQLAlchemyError as db_lookup_err:
        if "is not among the defined enum values" in str(db_lookup_err):
            logger.critical(
                f"CRITICAL ENUM MISMATCH during DB lookup for existing Persons. Error: {db_lookup_err}"
            )
            # Re-raise a specific error or handle appropriately based on application needs
            raise ValueError(
                "Database enum mismatch detected during person lookup."
            ) from db_lookup_err
        else:
            logger.error(f"Initial DB lookup failed: {db_lookup_err}", exc_info=True)
            # Depending on severity, might re-raise or return empty map
            raise  # Re-raise to be caught by _do_batch's outer handler
    except Exception as e:
        logger.error(f"Unexpected DB lookup error: {e}", exc_info=True)
        raise  # Re-raise to be caught by _do_batch's outer handler
    return existing_persons_map


# end _lookup_existing_persons


def _identify_fetch_candidates(
    matches_on_page: List[Dict[str, Any]], existing_persons_map: Dict[str, Person]
) -> Tuple[Set[str], List[Dict[str, Any]], int]:
    """Identifies which matches require API fetching versus skipping."""
    fetch_candidates_uuid: Set[str] = set()
    skipped_count_this_batch = 0
    matches_to_process_later: List[Dict[str, Any]] = []
    invalid_uuid_count = 0

    logger.debug("Identifying fetch candidates/skipped...")
    for match in matches_on_page:
        uuid_val = match.get("uuid")
        if not uuid_val:
            logger.warning(f"Skip match missing UUID: {match}")
            invalid_uuid_count += 1  # Track separately from skipped
            continue

        existing_person = existing_persons_map.get(uuid_val)

        if not existing_person:
            fetch_candidates_uuid.add(uuid_val)
            matches_to_process_later.append(match)
        else:
            needs_fetch = False
            existing_dna = existing_person.dna_match
            existing_tree = existing_person.family_tree
            db_in_tree = existing_person.in_my_tree
            api_in_tree = match.get("in_my_tree", False)

            if existing_dna:
                api_cm = match.get("cM_DNA")
                db_cm = existing_dna.cM_DNA
                api_segments = match.get(
                    "numSharedSegments"
                )  # Initial segments from match list
                db_segments = existing_dna.shared_segments
                # Check if API values differ from DB
                if api_cm is not None and db_cm is not None and int(api_cm) != db_cm:
                    needs_fetch = True
                if (
                    api_segments is not None
                    and db_segments is not None
                    and int(api_segments) != db_segments
                ):
                    needs_fetch = (
                        True  # Fetch details if segments differ from list view
                    )
                # Add check for predicted relationship change if available in list view?
                # Assuming 'predicted_relationship' isn't reliably in the list view for this check.
            else:  # Need fetch if no existing DNA record
                needs_fetch = True

            # Check Tree differences requiring details fetch
            if bool(api_in_tree) != bool(db_in_tree):
                needs_fetch = True
            elif (
                api_in_tree and not existing_tree
            ):  # Need fetch if in tree but no record
                needs_fetch = True

            if needs_fetch:
                fetch_candidates_uuid.add(uuid_val)
                matches_to_process_later.append(match)
            else:
                skipped_count_this_batch += 1

    if invalid_uuid_count > 0:
        logger.error(f"{invalid_uuid_count} matches skipped due to missing UUID.")

    logger.debug(
        f"Identified {len(fetch_candidates_uuid)} fetch candidates, {skipped_count_this_batch} skipped (no change)."
    )
    return fetch_candidates_uuid, matches_to_process_later, skipped_count_this_batch


# end _identify_fetch_candidates


def _perform_api_prefetches(
    session_manager: SessionManager,
    fetch_candidates_uuid: Set[str],
    matches_to_process_later: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Performs parallel API prefetches for candidates using ThreadPoolExecutor."""
    batch_combined_details: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_badge_data: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_ladder_data: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_relationship_prob_data: Dict[str, Optional[str]] = {}
    batch_tree_data: Dict[str, Dict[str, Any]] = {}
    futures = {}
    fetch_start_time = time.time()
    num_candidates = len(fetch_candidates_uuid)
    my_tree_id = session_manager.my_tree_id

    if not fetch_candidates_uuid:
        logger.debug("No fetch candidates for API pre-fetch.")
        return {"combined": {}, "tree": {}, "rel_prob": {}}

    logger.debug(f"--- Starting Pre-fetch ({num_candidates} candidates) ---")
    uuids_for_tree_badge = {
        uuid
        for uuid in fetch_candidates_uuid
        if any(
            m["uuid"] == uuid and m.get("in_my_tree") for m in matches_to_process_later
        )
    }

    # Adjust max_workers based on config
    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
        # Submit tasks for combined details and relationship probability
        for uuid_val in fetch_candidates_uuid:
            _ = session_manager.dynamic_rate_limiter.wait()  # Apply delay before submit
            futures[
                executor.submit(_fetch_combined_details, session_manager, uuid_val)
            ] = ("combined_details", uuid_val)

            _ = session_manager.dynamic_rate_limiter.wait()
            # Assuming max_labels_to_show is constant or retrieved elsewhere. Using default 2.
            futures[
                executor.submit(
                    _fetch_batch_relationship_prob, session_manager, uuid_val, 2
                )
            ] = ("relationship_prob", uuid_val)

        # Submit tasks for badge details (only for those marked in_my_tree)
        for uuid_val in uuids_for_tree_badge:
            _ = session_manager.dynamic_rate_limiter.wait()
            futures[
                executor.submit(_fetch_batch_badge_details, session_manager, uuid_val)
            ] = ("badge_details", uuid_val)

        # Process results as they complete
        temp_badge_results = {}
        for future in as_completed(futures):
            task_type, identifier = futures[future]
            try:
                result = future.result()
                if result is not None:
                    if task_type == "combined_details":
                        batch_combined_details[identifier] = result
                    elif task_type == "badge_details":
                        temp_badge_results[identifier] = (
                            result  # Store badge results temporarily
                        )
                    elif task_type == "relationship_prob":
                        batch_relationship_prob_data[identifier] = result
            except ConnectionError as conn_err:
                logger.error(
                    f"ConnErr pre-fetch '{task_type}' for {identifier}: {conn_err}",
                    exc_info=False,
                )
                if task_type == "relationship_prob":
                    batch_relationship_prob_data[identifier] = "N/A (Conn Error)"
            except Exception as exc:
                logger.error(
                    f"Exc pre-fetch '{task_type}' for {identifier}: {exc}",
                    exc_info=False,
                )
                if task_type == "relationship_prob":
                    batch_relationship_prob_data[identifier] = "N/A (Fetch Error)"

        # Submit tasks for ladder details based on badge results
        cfpid_to_uuid_map = {}
        ladder_futures = {}
        if my_tree_id and temp_badge_results:
            cfpid_list = []
            for uuid_val, badge_result in temp_badge_results.items():
                cfpid = badge_result.get("their_cfpid") if badge_result else None
                if cfpid:
                    cfpid_list.append(cfpid)
                    cfpid_to_uuid_map[cfpid] = uuid_val  # Map cfpid back to uuid
            if cfpid_list:
                logger.debug(
                    f"Submitting ladder pre-fetch for {len(cfpid_list)} CFPIDs..."
                )
                for cfpid in cfpid_list:
                    _ = session_manager.dynamic_rate_limiter.wait()
                    # Ensure my_tree_id is passed correctly
                    ladder_futures[
                        executor.submit(
                            _fetch_batch_ladder, session_manager, cfpid, my_tree_id
                        )
                    ] = ("ladder", cfpid)

        # Process ladder results
        for future in as_completed(ladder_futures):
            task_type, cfpid = ladder_futures[future]
            try:
                result = future.result()
                if result is not None:
                    uuid_val = cfpid_to_uuid_map.get(cfpid)  # Get UUID using the map
                    if uuid_val:
                        batch_ladder_data[uuid_val] = (
                            result  # Store ladder data keyed by UUID
                        )
                    else:
                        logger.warning(
                            f"Could not map ladder result for CFPID {cfpid} back to UUID."
                        )
            except ConnectionError as conn_err:
                logger.error(
                    f"ConnErr ladder fetch CFPID {cfpid}: {conn_err}", exc_info=False
                )
            except Exception as exc:
                logger.error(f"Exc ladder fetch CFPID {cfpid}: {exc}", exc_info=False)
    # End ThreadPoolExecutor block

    fetch_duration = time.time() - fetch_start_time
    logger.debug(f"--- Finished Pre-fetch. Duration: {fetch_duration:.2f}s ---")

    # Combine badge and ladder data into batch_tree_data, keyed by UUID
    for uuid_val, badge_result in temp_badge_results.items():
        if badge_result:  # Ensure badge_result is not None
            combined_tree_info = badge_result.copy()
            ladder_result_for_uuid = batch_ladder_data.get(uuid_val)
            if ladder_result_for_uuid:
                combined_tree_info.update(ladder_result_for_uuid)
            batch_tree_data[uuid_val] = combined_tree_info

    # Return combined prefetched data
    return {
        "combined": batch_combined_details,
        "tree": batch_tree_data,
        "rel_prob": batch_relationship_prob_data,
    }


# end _perform_api_prefetches


def _prepare_bulk_db_data(
    session: SqlAlchemySession,
    session_manager: SessionManager,
    matches_to_process: List[Dict[str, Any]],
    existing_persons_map: Dict[str, Person],
    prefetched_data: Dict[str, Dict[str, Any]],
    progress_bar: Optional[tqdm],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Processes matches using prefetched data and prepares bulk DB dictionaries."""
    prepared_bulk_data: List[Dict[str, Any]] = []
    page_statuses: Dict[str, int] = {
        "new": 0,
        "updated": 0,
        "error": 0,
    }  # Skipped handled outside
    num_to_process = len(matches_to_process)

    if not num_to_process:
        return [], page_statuses

    logger.debug(f"--- Preparing DB data for {num_to_process} candidates ---")
    process_start_time = time.time()

    for match in matches_to_process:
        uuid_val = match.get("uuid")
        _case_name = match.get("username", f"Unknown UUID {uuid_val}")
        prepared_data_for_this_match: Optional[Dict[str, Any]] = None
        status_for_this_match: Literal["new", "updated", "skipped", "error"] = "error"
        error_msg_for_this_match: Optional[str] = None

        try:
            existing_person = existing_persons_map.get(uuid_val)
            # Extract prefetched data safely
            prefetched_combined = prefetched_data.get("combined", {}).get(uuid_val)
            prefetched_tree = prefetched_data.get("tree", {}).get(uuid_val)
            prefetched_rel_prob = prefetched_data.get("rel_prob", {}).get(uuid_val)
            # Add relationship probability to the match dict *before* calling _do_match
            match["predicted_relationship"] = (
                prefetched_rel_prob or "N/A (Fetch Failed)"
            )

            if not session_manager.is_sess_valid():
                logger.error(f"WD session invalid before _do_match: {_case_name}")
                status_for_this_match = "error"
                error_msg_for_this_match = "WebDriver session invalid"
            else:
                # Call _do_match to get the prepared data dict and status
                (
                    prepared_data_for_this_match,
                    status_for_this_match,
                    error_msg_for_this_match,
                ) = _do_match(
                    session=session,
                    match=match,
                    session_manager=session_manager,
                    existing_person_arg=existing_person,
                    prefetched_combined_details=prefetched_combined,
                    prefetched_tree_data=prefetched_tree,
                )

            # --- Tally Status ---
            # Handle 'skipped' status returned by _do_match (though logic should prevent it here)
            if status_for_this_match in ["new", "updated", "error"]:
                page_statuses[status_for_this_match] += 1
            elif status_for_this_match == "skipped":
                logger.warning(
                    f"Unexpected 'skipped' status from _do_match for {_case_name}. Treating as processed."
                )
                # Don't increment error, but log it. Progress bar updated below.
            else:
                logger.error(
                    f"Unknown status '{status_for_this_match}' from _do_match for {_case_name}."
                )
                page_statuses["error"] += 1

            # --- Append Valid Data ---
            if status_for_this_match != "error" and prepared_data_for_this_match:
                prepared_bulk_data.append(prepared_data_for_this_match)
            elif status_for_this_match == "error":
                logger.error(
                    f"Error preparing DB data for {_case_name}: {error_msg_for_this_match}"
                )

        except Exception as inner_e:
            logger.error(
                f"Critical error processing {_case_name}: {inner_e}", exc_info=True
            )
            page_statuses["error"] += 1
        finally:
            if progress_bar:
                try:
                    progress_bar.update(1)
                except Exception as pbar_e:
                    logger.warning(f"Progress bar update error: {pbar_e}")

    process_duration = time.time() - process_start_time
    logger.debug(
        f"--- Finished preparing DB data. Duration: {process_duration:.2f}s ---"
    )
    return prepared_bulk_data, page_statuses


# end _prepare_bulk_db_data


def _execute_bulk_db_operations(
    session: SqlAlchemySession,
    prepared_bulk_data: List[Dict[str, Any]],
    existing_persons_map: Dict[str, Person],
) -> bool:
    """Executes bulk insert/update operations within an existing transaction."""
    bulk_start_time = time.time()
    num_items = len(prepared_bulk_data)
    logger.debug(f"--- Starting Bulk DB Ops ({num_items} items) ---")

    try:
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
        dna_match_creates = [
            d["dna_match"] for d in prepared_bulk_data if d.get("dna_match")
        ]  # Assume only creates needed for DNA/Tree based on _do_match logic
        family_tree_creates = [
            d["family_tree"]
            for d in prepared_bulk_data
            if d.get("family_tree") and d["family_tree"]["_operation"] == "create"
        ]
        family_tree_updates = [
            d["family_tree"]
            for d in prepared_bulk_data
            if d.get("family_tree") and d["family_tree"]["_operation"] == "update"
        ]
        created_person_map: Dict[str, int] = {}  # Maps UUID -> new Person ID

        # --- De-duplicate Person Creates ---
        person_creates_filtered = []
        seen_profile_ids = set()
        skipped_duplicates = 0
        if person_creates_raw:
            logger.debug(
                f"De-duplicating {len(person_creates_raw)} raw person creates..."
            )
            for p_data in person_creates_raw:
                profile_id = p_data.get("profile_id")
                if profile_id is None:
                    person_creates_filtered.append(p_data)
                elif profile_id not in seen_profile_ids:
                    person_creates_filtered.append(p_data)
                    seen_profile_ids.add(profile_id)
                else:
                    logger.warning(
                        f"Skip duplicate Person create ProfileID: {profile_id} (UUID: {p_data.get('uuid')})"
                    )
                    skipped_duplicates += 1
            if skipped_duplicates > 0:
                logger.warning(
                    f"Skipped {skipped_duplicates} duplicate person creates."
                )
            logger.debug(
                f"Proceeding with {len(person_creates_filtered)} unique person creates."
            )

        # --- Bulk Insert Persons ---
        if person_creates_filtered:
            logger.debug(
                f"Preparing {len(person_creates_filtered)} Person records for bulk insert..."
            )
            insert_data = [
                {k: v for k, v in p.items() if not k.startswith("_")}
                for p in person_creates_filtered
            ]
            default_status_enum = PersonStatusEnum.ACTIVE

            for item_data in insert_data:
                status_val = item_data.get("status")
                if isinstance(status_val, PersonStatusEnum):
                    item_data["status"] = (
                        status_val.value
                    )  # Convert Enum to value for bulk
                elif status_val is None:
                    item_data["status"] = default_status_enum.value
                else:
                    try:
                        item_data["status"] = PersonStatusEnum(
                            str(status_val).upper()
                        ).value
                    except ValueError:
                        logger.warning(
                            f"Invalid status value '{status_val}' during insert prep, using default."
                        )
                        item_data["status"] = default_status_enum.value
            # ...(duplicate profile ID check)...
            profile_ids_in_insert_data = [
                item.get("profile_id") for item in insert_data
            ]
            non_null_profile_ids = [
                pid for pid in profile_ids_in_insert_data if pid is not None
            ]
            if len(non_null_profile_ids) != len(set(non_null_profile_ids)):
                logger.error(
                    "CRITICAL: Duplicate non-NULL profile IDs DETECTED pre-bulk insert!"
                )
                id_counts = Counter(non_null_profile_ids)
                duplicates = {
                    pid: count for pid, count in id_counts.items() if count > 1
                }
                logger.error(f"Duplicate Profile IDs: {duplicates}")
                raise IntegrityError(
                    "Duplicate profile IDs found pre-insert",
                    params=duplicates,
                    orig=None,
                )  # Stop before inserting
            else:
                logger.debug(
                    "Verified uniqueness of non-NULL profile IDs pre-bulk insert."
                )

            logger.debug(f"Bulk inserting {len(insert_data)} Person records...")
            session.bulk_insert_mappings(Person, insert_data)
            logger.debug("Bulk insert Persons called.")
            # --- Get newly created IDs ---
            session.flush()  # Flush to get IDs assigned
            logger.debug("Session flushed for Person IDs.")
            inserted_uuids = [
                p_data["uuid"] for p_data in insert_data if p_data.get("uuid")
            ]
            if inserted_uuids:
                logger.debug(
                    f"Querying IDs for {len(inserted_uuids)} inserted UUIDs..."
                )
                newly_inserted_persons = (
                    session.query(Person.id, Person.uuid)
                    .filter(Person.uuid.in_(inserted_uuids))
                    .all()
                )
                created_person_map = {
                    p_uuid: p_id for p_id, p_uuid in newly_inserted_persons
                }
                logger.debug(f"Mapped {len(created_person_map)} new Person IDs.")
                if len(created_person_map) != len(inserted_uuids):
                    logger.error(
                        f"CRITICAL: ID map mismatch! Expected {len(inserted_uuids)}, got {len(created_person_map)}."
                    )
                    missing_uuids = set(inserted_uuids) - set(created_person_map.keys())
                    logger.error(f"Missing UUIDs: {missing_uuids}")
                    # This is critical, maybe raise error? For now, log error and continue cautiously.
            else:
                logger.warning("No UUIDs in insert_data for ID query.")
        else:
            logger.debug("No unique Person records to bulk insert.")

        # --- Bulk Update Persons ---
        if person_updates:
            update_mappings = []
            for p_data in person_updates:
                existing_id = p_data.get("_existing_person_id")
                if not existing_id:
                    logger.warning(
                        f"Skip person update UUID {p_data.get('uuid')}: Missing existing ID."
                    )
                    continue
                update_dict = {
                    k: v
                    for k, v in p_data.items()
                    if not k.startswith("_") and k not in ["uuid", "profile_id"]
                }
                if "status" in update_dict:
                    status_val = update_dict["status"]
                    if isinstance(status_val, PersonStatusEnum):
                        update_dict["status"] = (
                            status_val.value
                        )  # Convert Enum to value
                    elif status_val is not None:
                        try:
                            update_dict["status"] = PersonStatusEnum(
                                str(status_val).upper()
                            ).value
                        except ValueError:
                            logger.warning(
                                f"Invalid status value '{status_val}' during update prep, skipping status update for ID {existing_id}."
                            )
                            del update_dict["status"]
                    else:  # Handle explicit None status?
                        update_dict["status"] = (
                            None  # Or should default? Assuming None is allowed.
                        )
                if update_dict:  # Only add if there are fields to update
                    update_dict["id"] = existing_id
                    update_dict["updated_at"] = datetime.now(
                        timezone.utc
                    )  # Add update timestamp
                    update_mappings.append(update_dict)
            if update_mappings:
                logger.info(f"Bulk updating {len(update_mappings)} Person records...")
                session.bulk_update_mappings(Person, update_mappings)
                logger.debug("Bulk updated persons.")
            else:
                logger.debug("No Person updates needed.")

        # --- Create Master ID Map ---
        # Combines newly created IDs and existing IDs for linking related records
        all_person_ids_map = created_person_map.copy()
        # Add IDs from updated persons
        for p_update_data in person_updates:
            if p_update_data.get("_existing_person_id") and p_update_data.get("uuid"):
                all_person_ids_map[p_update_data["uuid"]] = p_update_data[
                    "_existing_person_id"
                ]
        # Add IDs for persons who were processed but neither created nor updated (e.g., only DNA/Tree changed)
        for uuid_processed in {
            p["person"]["uuid"] for p in prepared_bulk_data if p.get("person")
        }:
            if uuid_processed not in all_person_ids_map and existing_persons_map.get(
                uuid_processed
            ):
                all_person_ids_map[uuid_processed] = existing_persons_map[
                    uuid_processed
                ].id

        # --- Bulk Insert DnaMatch ---
        if dna_match_creates:
            dna_insert_data = []
            for dna_data in dna_match_creates:
                person_uuid = dna_data.get("uuid")
                person_id = all_person_ids_map.get(person_uuid)
                if person_id:
                    insert_dict = {
                        k: v for k, v in dna_data.items() if not k.startswith("_")
                    }
                    insert_dict["people_id"] = person_id
                    dna_insert_data.append(insert_dict)
                else:
                    logger.warning(
                        f"Skip DNA create UUID {person_uuid}: Person ID not found in master map."
                    )
            if dna_insert_data:
                logger.debug(
                    f"Bulk inserting {len(dna_insert_data)} DnaMatch records..."
                )
                session.bulk_insert_mappings(DnaMatch, dna_insert_data)
                logger.debug("Bulk inserted DnaMatches.")
            else:
                logger.debug("No valid DnaMatch records to insert.")

        # --- Bulk Insert/Update FamilyTree ---
        if family_tree_creates:
            tree_insert_data = []
            for tree_data in family_tree_creates:
                person_uuid = tree_data.get("uuid")
                person_id = all_person_ids_map.get(person_uuid)
                if person_id:
                    insert_dict = {
                        k: v for k, v in tree_data.items() if not k.startswith("_")
                    }
                    insert_dict["people_id"] = person_id
                    tree_insert_data.append(insert_dict)
                else:
                    logger.warning(
                        f"Skip FT create UUID {person_uuid}: Person ID not found in master map."
                    )
            if tree_insert_data:
                logger.debug(
                    f"Bulk inserting {len(tree_insert_data)} FamilyTree records..."
                )
                session.bulk_insert_mappings(FamilyTree, tree_insert_data)
                logger.debug("Bulk inserted FamilyTrees.")
            else:
                logger.debug("No valid FamilyTree records to insert.")

        if family_tree_updates:
            tree_update_mappings = []
            for tree_data in family_tree_updates:
                existing_tree_id = tree_data.get("_existing_tree_id")
                if not existing_tree_id:
                    logger.warning(
                        f"Skip FT update UUID {tree_data.get('uuid')}: Missing existing tree ID."
                    )
                    continue
                update_dict_tree = {
                    k: v
                    for k, v in tree_data.items()
                    if not k.startswith("_") and k != "uuid"
                }
                if update_dict_tree:  # Only add if there are fields to update
                    update_dict_tree["id"] = existing_tree_id
                    update_dict_tree["updated_at"] = datetime.now(timezone.utc)
                    # Ensure people_id is present if not explicitly updated
                    person_id_tree = all_person_ids_map.get(tree_data.get("uuid"))
                    if person_id_tree and "people_id" not in update_dict_tree:
                        update_dict_tree["people_id"] = (
                            person_id_tree  # Shouldn't be needed if ID doesn't change, but safe
                        )
                    tree_update_mappings.append(update_dict_tree)
            if tree_update_mappings:
                logger.info(
                    f"Bulk updating {len(tree_update_mappings)} FamilyTree records..."
                )
                session.bulk_update_mappings(FamilyTree, tree_update_mappings)
                logger.debug("Bulk updated FamilyTrees.")
            else:
                logger.debug("No FT updates needed.")

        bulk_duration = time.time() - bulk_start_time
        logger.debug(f"--- Bulk DB Ops OK. Duration: {bulk_duration:.2f}s ---")
        return True  # Indicate success

    except (IntegrityError, SQLAlchemyError) as bulk_db_err:
        logger.error(f"Bulk DB FAILED: {bulk_db_err}", exc_info=True)
        # No need to rollback here, db_transn context manager handles it
        return False  # Indicate failure
    except Exception as e:
        logger.error(f"Unexpected error during bulk DB operations: {e}", exc_info=True)
        return False  # Indicate failure


# end _execute_bulk_db_operations


def _do_batch(
    session_manager: SessionManager,
    matches_on_page: List[Dict[str, Any]],
    current_page: int,
    progress_bar: Optional[tqdm] = None,
) -> Tuple[int, int, int, int]:
    """
    V14.34: Processes batch using helpers. Handles cache error.
    """
    # Initialization
    page_new, page_updated, page_skipped, page_errors = 0, 0, 0, 0
    num_matches_on_page = len(matches_on_page)
    my_uuid = session_manager.my_uuid
    # Page status tracking dictionary
    page_statuses: Dict[str, int] = {"new": 0, "updated": 0, "skipped": 0, "error": 0}

    session = None  # Initialize session variable

    try:
        # Basic checks
        if not my_uuid:
            logger.error(f"_do_batch Page {current_page}: Missing my_uuid.")
            if progress_bar:
                progress_bar.update(num_matches_on_page)
            return 0, 0, 0, num_matches_on_page

        logger.debug(
            f"--- Starting Refactored Batch for Page {current_page} ({num_matches_on_page} matches) ---"
        )

        # DB Session (acquired once for the batch)
        session = session_manager.get_db_conn()
        if not session:
            logger.error(f"_do_batch Page {current_page}: Failed get DB session.")
            if progress_bar:
                progress_bar.update(num_matches_on_page)
            return 0, 0, 0, num_matches_on_page

        # Step 1: Lookup Existing Persons
        uuids_on_page = [m["uuid"] for m in matches_on_page if m.get("uuid")]
        existing_persons_map = _lookup_existing_persons(session, uuids_on_page)

        # Step 2: Identify Fetch Candidates and Skipped Matches
        fetch_candidates_uuid, matches_to_process_later, skipped_count = (
            _identify_fetch_candidates(matches_on_page, existing_persons_map)
        )
        if progress_bar and skipped_count > 0:
            progress_bar.update(skipped_count)
        page_statuses["skipped"] = skipped_count

        # Step 3: Targeted Pre-fetching (API Calls)
        prefetched_data = _perform_api_prefetches(
            session_manager, fetch_candidates_uuid, matches_to_process_later
        )

        # Step 4: Process Matches and Prepare Bulk Data
        prepared_bulk_data, prep_statuses = _prepare_bulk_db_data(
            session,
            session_manager,
            matches_to_process_later,
            existing_persons_map,
            prefetched_data,
            progress_bar,
        )
        # Update page statuses based on preparation results
        page_statuses["new"] = prep_statuses.get("new", 0)
        page_statuses["updated"] = prep_statuses.get("updated", 0)
        page_statuses["error"] = prep_statuses.get(
            "error", 0
        )  # Add errors from prep phase

        # Step 5: Execute Bulk DB Operations (within a transaction)
        if prepared_bulk_data:
            try:
                with db_transn(session) as sess:  # Use the existing session
                    logger.debug(
                        f"Entered transaction block bulk page {current_page} (Session: {id(sess)})."
                    )
                    bulk_success = _execute_bulk_db_operations(
                        sess, prepared_bulk_data, existing_persons_map
                    )
                    if not bulk_success:
                        # If bulk ops failed, mark all prepared items as errors
                        logger.error(
                            f"Bulk DB operations failed for page {current_page}. Adjusting counts."
                        )
                        failed_items = len(prepared_bulk_data)
                        page_statuses["error"] += failed_items
                        page_statuses["new"] = 0  # Reset counts as they didn't commit
                        page_statuses["updated"] = 0
                    logger.debug(
                        f"Exiting transaction block scope for page {current_page} (Commit/Rollback follows)."
                    )
                # Post-transaction check (optional)
                logger.debug(
                    f"Transaction block finished for page {current_page}. Checking session state..."
                )
                # ...(post-transaction query check kept from previous version)...
                if session:
                    logger.debug(
                        f"  Session {id(session)} active status after transaction block: {session.is_active}"
                    )
                    try:
                        committed_uuids = [
                            p["person"]["uuid"]
                            for p in prepared_bulk_data
                            if p.get("person") and p["person"].get("uuid")
                        ]
                        if committed_uuids:
                            count = (
                                session.query(func.count(Person.id))
                                .filter(Person.uuid.in_(committed_uuids))
                                .scalar()
                            )
                            logger.debug(
                                f"  Post-transaction query check: Found {count}/{len(committed_uuids)} people for this batch via direct query."
                            )
                        else:
                            logger.debug(
                                "  Post-transaction query check: No new person UUIDs found in prepared data for query."
                            )
                    except KeyError as ke:
                        logger.error(
                            f"  Post-transaction query check failed: KeyError - {ke}. Prepared data structure might be wrong."
                        )
                        logger.debug(
                            f"  Problematic prepared data structure (first item): {prepared_bulk_data[0] if prepared_bulk_data else 'N/A'}"
                        )
                    except Exception as query_err:
                        logger.warning(
                            f"  Post-transaction query check failed: {query_err}",
                            exc_info=True,
                        )
                else:
                    logger.warning("  Session object is None after transaction block?!")

            except (
                IntegrityError,
                SQLAlchemyError,
                ValueError,
            ) as bulk_db_err:  # Catch specific errors from bulk helper or transaction
                # db_transn handles rollback
                logger.error(
                    f"Bulk DB transaction FAILED page {current_page}: {bulk_db_err}",
                    exc_info=True,
                )
                failed_items = len(prepared_bulk_data)
                page_statuses["error"] += failed_items
                page_statuses["new"] = 0
                page_statuses["updated"] = 0
                logger.warning(
                    f"Page {current_page} counts adjusted due to DB transaction failure: {page_statuses}"
                )
            except Exception as e:  # Catch other unexpected errors during transaction
                logger.error(
                    f"Unexpected error during bulk DB transaction page {current_page}: {e}",
                    exc_info=True,
                )
                failed_items = len(prepared_bulk_data)
                page_statuses["error"] += failed_items
                page_statuses["new"] = 0
                page_statuses["updated"] = 0
                logger.warning(
                    f"Page {current_page} counts adjusted due to unexpected transaction error: {page_statuses}"
                )
        else:
            logger.debug(
                f"No data prepared for bulk operations on page {current_page}."
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

    except ValueError as ve:  # Catch specific critical errors like enum mismatch
        logger.critical(
            f"CRITICAL VALUE ERROR in _do_batch page {current_page}: {ve}",
            exc_info=True,
        )
        if progress_bar:
            # Update progress bar for all items on the page as errors
            processed_count = sum(page_statuses.values())
            remaining_in_batch = num_matches_on_page - processed_count
            if remaining_in_batch > 0:
                progress_bar.update(remaining_in_batch)
        return 0, 0, 0, num_matches_on_page  # Return all as errors
    except Exception as outer_batch_exc:
        logger.critical(
            f"CRITICAL UNHANDLED EXCEPTION in _do_batch for page {current_page}: {outer_batch_exc}",
            exc_info=True,
        )
        if progress_bar:
            processed_count = sum(page_statuses.values())
            remaining_in_batch = num_matches_on_page - processed_count
            if remaining_in_batch > 0:
                logger.debug(
                    f"Updating progress bar by {remaining_in_batch} due to outer exception."
                )
                try:
                    progress_bar.update(remaining_in_batch)
                except Exception as pbar_e:
                    logger.warning(
                        f"Bar update error during exception handling: {pbar_e}"
                    )
                # Adjust error count directly
                page_statuses["error"] = (
                    page_statuses.get("error", 0) + remaining_in_batch
                )

        logger.error(
            f"Returning error tuple from _do_batch due to unhandled exception."
        )
        # Ensure final counts don't exceed total matches on page
        final_new = page_statuses.get("new", 0)
        final_updated = page_statuses.get("updated", 0)
        final_skipped = page_statuses.get("skipped", 0)
        # Calculate final error count to make sure total matches page total
        final_error = num_matches_on_page - (final_new + final_updated + final_skipped)
        return (
            final_new,
            final_updated,
            final_skipped,
            max(0, final_error),
        )  # Ensure error count >= 0
    finally:
        # Ensure session is returned even if errors occurred
        if session:
            session_manager.return_session(session)
        logger.debug(f"--- Finished Batch Processing for Page {current_page} ---")


# End of _do_batch


#################################################################################
# 4. Individual Match Processing (_do_match)
#################################################################################


def _do_match(
    session: Session,
    match: Dict[str, Any],
    session_manager: SessionManager,
    existing_person_arg: Optional[Person],
    prefetched_combined_details: Optional[Dict[str, Any]],
    prefetched_tree_data: Optional[Dict[str, Any]],
) -> Tuple[
    Optional[Dict[str, Any]],
    Literal["new", "updated", "skipped", "error"],
    Optional[str],
]:
    """
    V14.34: Prepares data for a single match for bulk operations. No changes needed.
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
    match_username = format_name(match_username_raw)
    predicted_relationship = match.get(
        "predicted_relationship", "N/A"
    )  # Now fetched/added in _prepare_bulk_db_data
    match_in_my_tree = match.get("in_my_tree", False)
    log_ref = f"UUID={match_uuid or 'N/A'} User='{match_username or 'Unknown'}'"
    log_ref_short = f"UUID={match_uuid} User='{match_username}'"
    prepared_data_for_bulk: Dict[str, Any] = {
        "person": None,
        "dna_match": None,
        "family_tree": None,
    }
    person_update_needed: bool = False
    overall_status: Literal["new", "updated", "skipped", "error"] = "error"
    error_msg: Optional[str] = None

    if not match_uuid:
        error_msg = f"Pre-check failed: Missing 'uuid' in match data: {match}"
        logger.error(error_msg)
        return None, "error", error_msg

    try:
        is_new_person = existing_person is None

        # Step 2: Prepare Incoming Data & Determine Profile/Admin IDs
        details_part = prefetched_combined_details or {}
        profile_part = (
            prefetched_combined_details or {}
        )  # Use same dict for profile info from combined fetch
        # --- Get IDs from prefetched data OR fallback to match list hints ---
        raw_tester_profile_id = details_part.get("tester_profile_id") or match.get(
            "profile_id"
        )
        raw_admin_profile_id = details_part.get("admin_profile_id") or match.get(
            "administrator_profile_id_hint"
        )
        raw_admin_username = details_part.get("admin_username") or match.get(
            "administrator_username_hint"
        )
        formatted_admin_username = format_name(raw_admin_username)
        tester_profile_id_upper = (
            raw_tester_profile_id.upper() if raw_tester_profile_id else None
        )
        admin_profile_id_upper = (
            raw_admin_profile_id.upper() if raw_admin_profile_id else None
        )
        # --- End ID Extraction ---
        person_profile_id_to_save = None
        person_admin_id_to_save = None
        person_admin_username_to_save = None

        # Determine Person IDs based on Scenarios (Unchanged)
        if tester_profile_id_upper and admin_profile_id_upper:
            if tester_profile_id_upper == admin_profile_id_upper:
                if (
                    match_username
                    and formatted_admin_username
                    and match_username.lower() == formatted_admin_username.lower()
                ):  # D
                    person_profile_id_to_save = tester_profile_id_upper
                    person_admin_id_to_save = None
                    person_admin_username_to_save = None
                else:  # C
                    person_profile_id_to_save = None
                    person_admin_id_to_save = admin_profile_id_upper
                    person_admin_username_to_save = formatted_admin_username
            else:  # B
                person_profile_id_to_save = tester_profile_id_upper
                person_admin_id_to_save = admin_profile_id_upper
                person_admin_username_to_save = formatted_admin_username
        elif tester_profile_id_upper and not admin_profile_id_upper:  # A
            person_profile_id_to_save = tester_profile_id_upper
            person_admin_id_to_save = None
            person_admin_username_to_save = None
        elif not tester_profile_id_upper and admin_profile_id_upper:  # C variation
            person_profile_id_to_save = None
            person_admin_id_to_save = admin_profile_id_upper
            person_admin_username_to_save = formatted_admin_username
        else:  # Neither found
            logger.warning(f"{log_ref}: Neither tester nor admin profile ID found.")
            person_profile_id_to_save = None
            person_admin_id_to_save = None
            person_admin_username_to_save = None

        # Construct message link
        message_target_id = person_profile_id_to_save or person_admin_id_to_save
        constructed_message_link = None
        if message_target_id:
            constructed_message_link = urljoin(
                config_instance.BASE_URL, f"/messaging/?p={message_target_id.upper()}"
            )

        # Extract other person fields
        birth_year_val = None
        if prefetched_tree_data and prefetched_tree_data.get("their_birth_year"):
            try:
                birth_year_val = int(prefetched_tree_data["their_birth_year"])
            except (ValueError, TypeError):
                pass
        # --- Ensure last_logged_in is datetime object ---
        last_logged_in_val = profile_part.get("last_logged_in_dt")
        if last_logged_in_val and not isinstance(last_logged_in_val, datetime):
            # Attempt conversion if it's a string (adjust format as needed)
            if isinstance(last_logged_in_val, str):
                try:
                    if last_logged_in_val.endswith("Z"):
                        last_logged_in_val = datetime.fromisoformat(
                            last_logged_in_val.replace("Z", "+00:00")
                        )
                    else:
                        dt_naive = datetime.fromisoformat(last_logged_in_val)
                        last_logged_in_val = (
                            dt_naive.replace(tzinfo=timezone.utc)
                            if dt_naive.tzinfo is None
                            else dt_naive
                        )
                except ValueError:
                    logger.warning(
                        f"Could not parse last_logged_in string: {last_logged_in_val}"
                    )
                    last_logged_in_val = None
            else:
                last_logged_in_val = None  # Ignore if not string or datetime
        # Ensure timezone awareness
        if isinstance(last_logged_in_val, datetime):
            last_logged_in_val = (
                last_logged_in_val.astimezone(timezone.utc)
                if last_logged_in_val.tzinfo
                else last_logged_in_val.replace(tzinfo=timezone.utc)
            )
        # --- End last_logged_in handling ---

        # --- Incoming Person Data Dict ---
        incoming_person_data = {
            "uuid": match_uuid.upper(),
            "profile_id": person_profile_id_to_save,
            "username": match_username,
            "administrator_profile_id": person_admin_id_to_save,
            "administrator_username": person_admin_username_to_save,
            "in_my_tree": match_in_my_tree,
            "first_name": match.get("first_name"),  # Get from match list initially
            "last_logged_in": last_logged_in_val,  # Use processed datetime
            "contactable": profile_part.get(
                "contactable", False
            ),  # From combined details
            "gender": details_part.get("gender"),  # From combined details
            "message_link": constructed_message_link,
            "birth_year": birth_year_val,  # From tree details
            "status": PersonStatusEnum.ACTIVE,  # Default status
        }

        # Prepare incoming DNA data
        incoming_dna_data = None
        needs_dna_create_or_update = False
        if dna_match_record is None:
            needs_dna_create_or_update = True
        else:  # Check if fetched details differ from existing record
            # Use prefetched details if available
            api_cm = match.get("cM_DNA")  # Get cM from match list data
            db_cm = dna_match_record.cM_DNA
            api_segments = (
                prefetched_combined_details.get("shared_segments")
                if prefetched_combined_details
                else match.get("numSharedSegments")
            )  # Prefer detail, fallback list
            db_segments = dna_match_record.shared_segments
            api_longest = (
                prefetched_combined_details.get("longest_shared_segment")
                if prefetched_combined_details
                else None
            )  # Only in details
            db_longest = dna_match_record.longest_shared_segment

            # Compare relevant fields
            if (
                (api_cm is not None and db_cm is not None and int(api_cm) != db_cm)
                or (
                    api_segments is not None
                    and db_segments is not None
                    and int(api_segments) != db_segments
                )
                or (
                    api_longest is not None
                    and db_longest is not None
                    and float(api_longest) != db_longest
                )
                or (dna_match_record.predicted_relationship != predicted_relationship)
            ):  # Compare predicted relationship
                needs_dna_create_or_update = True

        if needs_dna_create_or_update:
            # Build dict using prefetched details primarily, fall back to match list data
            dna_dict_base = {
                "uuid": match_uuid.upper(),
                "compare_link": match.get("compare_link"),
                "_operation": "create",
            }
            if prefetched_combined_details:
                dna_dict_base.update(
                    {
                        "cM_DNA": match.get("cM_DNA"),  # Still use list cM
                        "predicted_relationship": predicted_relationship,
                        "shared_segments": prefetched_combined_details.get(
                            "shared_segments"
                        ),
                        "longest_shared_segment": prefetched_combined_details.get(
                            "longest_shared_segment"
                        ),
                        "meiosis": prefetched_combined_details.get("meiosis"),
                        "from_my_fathers_side": prefetched_combined_details.get(
                            "from_my_fathers_side", False
                        ),
                        "from_my_mothers_side": prefetched_combined_details.get(
                            "from_my_mothers_side", False
                        ),
                    }
                )
            else:  # Fallback if details fetch failed
                logger.warning(
                    f"{log_ref}: DNA needs create/update, but no combined details fetched. Using limited data."
                )
                dna_dict_base.update(
                    {
                        "cM_DNA": match.get("cM_DNA"),
                        "predicted_relationship": predicted_relationship,
                        "shared_segments": match.get(
                            "numSharedSegments"
                        ),  # Use list segments
                    }
                )
            incoming_dna_data = dna_dict_base

        # Prepare incoming Tree data
        incoming_tree_data = None
        should_have_tree = match_in_my_tree
        tree_operation: Literal["create", "update", "none"] = "none"
        view_in_tree_link, facts_link = None, None
        their_cfpid_final = None

        if prefetched_tree_data:  # Check if we have *any* tree data first
            their_cfpid_final = prefetched_tree_data.get("their_cfpid")
            if their_cfpid_final and session_manager.my_tree_id:
                base_person_path = f"/family-tree/person/tree/{session_manager.my_tree_id}/person/{their_cfpid_final}"
                facts_link = urljoin(
                    config_instance.BASE_URL, f"{base_person_path}/facts"
                )
                view_params = {
                    "cfpid": their_cfpid_final,
                    "showMatches": "true",
                    "sid": session_manager.my_uuid,
                }
                base_view_url = urljoin(
                    config_instance.BASE_URL,
                    f"/family-tree/tree/{session_manager.my_tree_id}/family",
                )
                view_in_tree_link = f"{base_view_url}?{urlencode(view_params)}"

        # Determine if tree operation is needed
        if should_have_tree and family_tree_record is None:
            tree_operation = "create"
        elif should_have_tree and family_tree_record is not None:
            if (
                prefetched_tree_data
            ):  # Only check for updates if we have data to compare
                # Use already calculated links/data
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
                    (
                        "relationship_path",
                        prefetched_tree_data.get("relationship_path"),
                    ),
                    ("facts_link", facts_link),
                    ("view_in_tree_link", view_in_tree_link),
                ]
                for field, new_val in fields_to_check:
                    old_val = getattr(family_tree_record, field, None)
                    if (new_val is not None and new_val != old_val) or (
                        new_val is None and old_val is not None
                    ):
                        tree_operation = "update"
                        break
        elif not should_have_tree and family_tree_record is not None:
            logger.warning(
                f"{log_ref}: Data mismatch: Not 'in_my_tree', but FT record exists (ID: {family_tree_record.id}). Deletion not implemented."
            )
            # Consider setting status to ARCHIVED or similar if this happens?
            tree_operation = "none"

        # Build tree data dictionary if create/update needed
        if tree_operation != "none":
            if prefetched_tree_data:  # Only build if we have data
                tree_person_name = prefetched_tree_data.get(
                    "their_firstname", "Unknown"
                )
                incoming_tree_data = {
                    "uuid": match_uuid.upper(),
                    "cfpid": their_cfpid_final,
                    "person_name_in_tree": tree_person_name,
                    "facts_link": facts_link,
                    "view_in_tree_link": view_in_tree_link,
                    "actual_relationship": prefetched_tree_data.get(
                        "actual_relationship"
                    ),
                    "relationship_path": prefetched_tree_data.get("relationship_path"),
                    "_operation": tree_operation,
                    "_existing_tree_id": (
                        family_tree_record.id
                        if family_tree_record and tree_operation == "update"
                        else None
                    ),
                }
            else:  # Tree needed, but no data fetched
                logger.warning(
                    f"{log_ref}: FT needs {tree_operation}, but no tree details fetched. Cannot create/update FT record."
                )
                # Ensure tree_operation is reset if no data can be prepared
                tree_operation = "none"

        # Step 3: Compare and Build Bulk Data Dictionary
        if is_new_person:
            person_data_for_bulk = incoming_person_data.copy()
            person_data_for_bulk["_operation"] = "create"
            prepared_data_for_bulk["person"] = person_data_for_bulk
            if incoming_dna_data:
                prepared_data_for_bulk["dna_match"] = incoming_dna_data
            # Only add tree data if operation is 'create' and data exists
            if incoming_tree_data and incoming_tree_data["_operation"] == "create":
                prepared_data_for_bulk["family_tree"] = incoming_tree_data
            overall_status = "new"
        else:  # Existing Person
            person_data_for_update = {
                "_operation": "update",
                "_existing_person_id": existing_person.id,
                "uuid": match_uuid.upper(),
            }
            person_update_needed = False
            fields_to_compare_person = [
                ("username", "username"),
                ("profile_id", "profile_id"),
                ("administrator_profile_id", "administrator_profile_id"),
                ("administrator_username", "administrator_username"),
                ("in_my_tree", "in_my_tree"),
                ("first_name", "first_name"),
                ("contactable", "contactable"),
                ("gender", "gender"),
                ("message_link", "message_link"),
                ("birth_year", "birth_year"),
                ("status", "status"),  # Include status comparison
            ]
            # Compare standard fields
            for db_field, incoming_key in fields_to_compare_person:
                new_val = incoming_person_data.get(incoming_key)
                old_val = getattr(existing_person, db_field, None)

                # Type normalization for comparison
                if incoming_key in ["contactable", "in_my_tree"]:
                    new_val = bool(new_val)
                if db_field == "status" and isinstance(new_val, PersonStatusEnum):
                    new_val = new_val.value  # Compare Enum values
                if isinstance(old_val, PersonStatusEnum):
                    old_val = old_val.value  # Compare Enum values

                if (new_val is not None and new_val != old_val) or (
                    new_val is None and old_val is not None
                ):
                    # Special handling for fields we only want to populate if currently NULL
                    # Example: Don't overwrite existing birth_year/gender if new value is None,
                    # but DO update if existing is None and new has a value.
                    if (
                        db_field in ["gender", "birth_year"]
                        and new_val is None
                        and old_val is not None
                    ):
                        pass  # Don't overwrite existing value with None
                    else:
                        # Prepare value for update (handle Enum for status)
                        value_for_update = (
                            PersonStatusEnum(new_val)
                            if db_field == "status" and new_val is not None
                            else new_val
                        )
                        person_data_for_update[db_field] = value_for_update
                        person_update_needed = True

            # Compare last_logged_in (DateTime)
            new_dt = incoming_person_data.get("last_logged_in")
            old_dt = existing_person.last_logged_in
            # Ensure both are aware UTC datetimes for comparison
            new_utc = None
            if isinstance(new_dt, datetime):
                new_utc = (
                    new_dt.astimezone(timezone.utc).replace(microsecond=0)
                    if new_dt.tzinfo
                    else new_dt.replace(tzinfo=timezone.utc, microsecond=0)
                )
            old_utc = None
            if isinstance(old_dt, datetime):
                old_utc = (
                    old_dt.astimezone(timezone.utc).replace(microsecond=0)
                    if old_dt.tzinfo
                    else old_dt.replace(tzinfo=timezone.utc, microsecond=0)
                )

            if new_utc != old_utc:  # Handles None comparison correctly
                person_data_for_update["last_logged_in"] = (
                    new_dt  # Store original aware datetime object
                )
                person_update_needed = True

            # --- Final Assembly for Existing Person ---
            if person_update_needed:
                prepared_data_for_bulk["person"] = person_data_for_update
            if incoming_dna_data:
                prepared_data_for_bulk["dna_match"] = (
                    incoming_dna_data  # Assume DNA is always create/replace if needed
                )
            if incoming_tree_data:
                prepared_data_for_bulk["family_tree"] = (
                    incoming_tree_data  # Handles create or update
                )

            # Determine overall status based on what needs changing
            if (
                person_update_needed
                or incoming_dna_data
                or (incoming_tree_data and tree_operation != "none")
            ):
                overall_status = "updated"
            else:
                overall_status = "skipped"

        # Only return data if something actually needs to be created or updated
        data_to_return = prepared_data_for_bulk if overall_status != "skipped" else None
        return data_to_return, overall_status, None  # No error message if successful

    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        error_msg_for_log = f"Unexpected critical error ({error_type}) in _do_match for {log_ref}. Details: {error_details}"
        logger.error(error_msg_for_log, exc_info=True)
        error_msg_return = (
            f"Unexpected {error_type} during data prep for {log_ref_short}"
        )
        return None, "error", error_msg_return


# End of _do_match


#################################################################################
# 5. API Data Acquisition Helpers
#################################################################################


def get_matches(
    session_manager: SessionManager,
    db_session: SqlAlchemySession,
    current_page: int = 1,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[int]]:
    """
    V14.38: Fetches match list data. Fixes NameError in outer exception block, adds traceback import.
    Initializes total_pages. Makes refinement loop more robust.
    """
    # Initialize variables
    total_pages: Optional[int] = None  # <<< Initialize total_pages
    if not isinstance(session_manager, SessionManager):
        logger.error("Invalid SessionManager passed to get_matches.")
        return None, None
    driver = session_manager.driver
    if not driver:
        logger.error("WebDriver not initialized in get_matches.")
        return None, None
    if not session_manager.my_uuid:
        logger.error("SessionManager my_uuid not initialized in get_matches.")
        return None, None
    if not session_manager.is_sess_valid():
        logger.error("get_matches: Session invalid at start.")
        return None, None

    my_uuid = session_manager.my_uuid
    specific_csrf_token = None
    found_token_name = None
    csrf_token_cookie_name = "_dnamatches-matchlistui-x-csrf-token"
    fallback_csrf_cookie_name = "_csrf"
    in_tree_ids: Set[str] = set()

    try:
        # --- CSRF Token Retrieval ---
        logger.debug(
            f"Waiting for match list element '{MATCH_ENTRY_SELECTOR}' before reading CSRF cookies..."
        )
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, MATCH_ENTRY_SELECTOR))
            )
            logger.debug("Match list element found.")
            time.sleep(0.5)
        except TimeoutException:
            logger.warning(
                f"Timeout waiting for match list element '{MATCH_ENTRY_SELECTOR}'. Cookie read might fail."
            )
        except Exception as wait_e:
            logger.warning(f"Error waiting for match list element: {wait_e}.")

        logger.debug(f"Attempting to read CSRF cookies...")
        for cookie_name in [csrf_token_cookie_name, fallback_csrf_cookie_name]:
            try:
                cookie_obj = driver.get_cookie(cookie_name)
                if (
                    cookie_obj
                    and isinstance(cookie_obj, dict)
                    and "value" in cookie_obj
                    and cookie_obj["value"]
                ):
                    specific_csrf_token = unquote(cookie_obj["value"]).split("|")[0]
                    found_token_name = cookie_name
                    logger.debug(f"Read CSRF token from cookie '{found_token_name}'.")
                    break
            except NoSuchCookieException:
                continue
            except WebDriverException as cookie_e:
                logger.warning(
                    f"WebDriver error getting cookie '{cookie_name}': {cookie_e}"
                )
                raise ConnectionError(
                    f"WebDriver error getting CSRF cookie: {cookie_e}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error getting cookie '{cookie_name}': {e}",
                    exc_info=True,
                )
                continue

        if not specific_csrf_token:
            logger.debug(f"CSRF token not found via get_cookie. Trying fallback...")
            all_cookies = get_driver_cookies(driver)
            if all_cookies:
                for cookie_name in [csrf_token_cookie_name, fallback_csrf_cookie_name]:
                    if cookie_name in all_cookies and all_cookies[cookie_name]:
                        specific_csrf_token = unquote(all_cookies[cookie_name]).split(
                            "|"
                        )[0]
                        found_token_name = cookie_name
                        logger.debug(
                            f"Read CSRF token via fallback ('{found_token_name}')."
                        )
                        break
            else:
                logger.warning("Fallback get_driver_cookies also failed.")

        if not specific_csrf_token:
            logger.error(
                "Failed to obtain a valid CSRF token. Cannot call Match List API."
            )
            return None, None
        # --- End CSRF ---

        # --- API Call for Match List ---
        match_list_url = urljoin(
            config_instance.BASE_URL,
            f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={current_page}",
        )
        chrome_version = "125"  # Example, keep dynamic if needed
        user_agent = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version}.0.0.0 Safari/537.36"
        sec_ch_ua = f'"Google Chrome";v="{chrome_version}", "Not-A.Brand";v="8", "Chromium";v="{chrome_version}"'
        match_list_headers = {
            "User-Agent": user_agent,
            "accept": "application/json",
            "Referer": urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
            "x-csrf-token": specific_csrf_token,
            "sec-ch-ua": sec_ch_ua,
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            # Origin removed by _api_req for this description
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "dnt": "1",
        }
        api_response = _api_req(
            url=match_list_url,
            driver=driver,
            session_manager=session_manager,
            method="GET",
            headers=match_list_headers,
            use_csrf_token=False,
            api_description="Match List API",  # Triggers special handling in _api_req
        )
        # --- End API Call ---

        # --- Process API Response ---
        if api_response is None:
            logger.warning(
                f"No response/error from match list API page {current_page}."
            )
            return None, None
        if not isinstance(api_response, dict):
            logger.error(
                f"Match List API did not return dict. Page {current_page}. Type: {type(api_response)}"
            )
            return None, None

        total_pages_raw = api_response.get("totalPages")
        if total_pages_raw is not None:
            try:
                total_pages = int(total_pages_raw)
            except (ValueError, TypeError):
                logger.warning(f"Could not parse totalPages '{total_pages_raw}'.")
        else:
            logger.warning("totalPages missing from response.")

        match_data_list = api_response.get("matchList", [])
        if not match_data_list:
            logger.info(f"No matches found in 'matchList' page {current_page}.")
            return [], total_pages
        # --- End Process API Response ---

        # --- Filter Matches (SampleID Check) ---
        valid_matches_for_processing: List[Dict[str, Any]] = []
        skipped_sampleid_count = 0
        for m in match_data_list:
            if isinstance(m, dict) and m.get("sampleId"):
                valid_matches_for_processing.append(m)
            else:
                skipped_sampleid_count += 1
                logger.warning(
                    f"Skipping raw match missing 'sampleId' page {current_page}."
                )
        if skipped_sampleid_count > 0:
            logger.warning(
                f"Skipped {skipped_sampleid_count} matches page {current_page} (missing 'sampleId')."
            )
        if not valid_matches_for_processing:
            logger.warning(f"No valid matches page {current_page}.")
            return [], total_pages  # Return empty list and potentially None total_pages
        # --- End Filter Matches ---

        # --- Fetch In-Tree Status ---
        sample_ids_on_page = [
            match["sampleId"].upper() for match in valid_matches_for_processing
        ]
        cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids_on_page))}"

        # Use get with default=ENOVAL to distinguish miss from None value
        cached_in_tree = global_cache.get(cache_key_tree, default=ENOVAL, retry=True)

        if cached_in_tree is not ENOVAL and isinstance(cached_in_tree, set):
            in_tree_ids = cached_in_tree
            logger.debug(f"Loaded {len(in_tree_ids)} in-tree IDs from cache.")
        else:
            if cached_in_tree is not ENOVAL:
                logger.warning(
                    f"Cache hit for {cache_key_tree}, but type is {type(cached_in_tree)}, not set. Refetching."
                )

            if not session_manager.is_sess_valid():
                logger.error(f"In-Tree Check: Session invalid page {current_page}.")
            else:
                in_tree_url = urljoin(
                    config_instance.BASE_URL,
                    f"discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid.upper()}",
                )
                logger.debug(f"Fetching in-tree status page {current_page}...")
                in_tree_headers = {
                    "X-CSRF-Token": specific_csrf_token,
                    "Content-Type": "application/json",
                    "User-Agent": user_agent,
                    "Referer": urljoin(
                        config_instance.BASE_URL, "/discoveryui-matches/list/"
                    ),
                    "Origin": config_instance.BASE_URL.rstrip("/"),
                    "Accept": "application/json",
                }
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
                    in_tree_ids = {
                        item.upper()
                        for item in response_in_tree
                        if isinstance(item, str)
                    }
                    try:
                        global_cache.set(
                            cache_key_tree,
                            in_tree_ids,
                            expire=config_instance.CACHE_TIMEOUT,
                        )
                        logger.debug(
                            f"Fetched/cached {len(in_tree_ids)} in-tree IDs page {current_page}."
                        )
                    except Exception as cache_write_err:
                        logger.error(
                            f"Error writing to cache for key {cache_key_tree}: {cache_write_err}"
                        )
                else:
                    logger.warning(
                        f"In-Tree Status Check API failed/unexpected page {current_page}. Resp: {response_in_tree}"
                    )
        # --- End Fetch In-Tree ---

        # --- Refine Match Data ---
        refined_matches: List[Dict[str, Any]] = []
        for match_index, match in enumerate(
            valid_matches_for_processing
        ):  # Use enumerate for better logging
            try:
                profile = match.get("matchProfile", {})
                relationship = match.get("relationship", {})
                # Sample ID check removed as list comprehension ensures it exists
                sample_id_upper = match["sampleId"].upper()

                profile_user_id = profile.get("userId")
                profile_user_id_upper = (
                    str(profile_user_id).upper() if profile_user_id else None
                )
                raw_display_name = profile.get("displayName")
                match_username = format_name(raw_display_name)

                # --- Refined first_name extraction (V14.36 logic) ---
                first_name = None
                if match_username and match_username != "Valued Relative":
                    trimmed_username = match_username.strip()
                    if trimmed_username:
                        name_parts = trimmed_username.split()
                        if name_parts:
                            first_name = name_parts[0]
                # --- End first_name extraction ---

                admin_profile_id_hint = match.get("adminId")
                admin_username_hint = match.get("adminName")
                compare_link = urljoin(
                    config_instance.BASE_URL,
                    f"discoveryui-matches/compare/{my_uuid.upper()}/with/{sample_id_upper}",
                )

                refined_match_data = {
                    "username": match_username,
                    "first_name": first_name,
                    "initials": profile.get("displayInitials", "??").upper(),
                    "gender": match.get("gender"),
                    "profile_id": profile_user_id_upper,
                    "uuid": sample_id_upper,
                    "administrator_profile_id_hint": admin_profile_id_hint,
                    "administrator_username_hint": admin_username_hint,
                    "photoUrl": profile.get("photoUrl", ""),
                    "cM_DNA": int(relationship.get("sharedCentimorgans", 0)),
                    "numSharedSegments": int(relationship.get("numSharedSegments", 0)),
                    "compare_link": compare_link,
                    "message_link": None,
                    "in_my_tree": sample_id_upper in in_tree_ids,
                    "createdDate": match.get("createdDate"),
                }
                refined_matches.append(refined_match_data)

            # --- Inner Exception Handling (Catching errors during refinement) ---
            except (IndexError, KeyError, TypeError, ValueError) as refine_err:
                # Catch specific, potentially recoverable errors during refinement
                logger.error(
                    f"Refinement error on page {current_page} for match {match_index+1} (UUID: {match.get('uuid', 'N/A')}): {type(refine_err).__name__} - {refine_err}. Skipping this match.",
                    exc_info=False,  # Keep log cleaner for skippable errors
                )
                logger.debug(f"Problematic match data: {match}")
                # Continue to the next match instead of raising
                continue
            except Exception as critical_refine_err:
                # Catch unexpected errors during refinement and raise them
                logger.error(
                    f"CRITICAL unexpected error refining match page {current_page}, match {match_index+1} (UUID: {match.get('uuid', 'N/A')}): {critical_refine_err}",
                    exc_info=True,  # Log full traceback for critical errors
                )
                logger.debug(f"Problematic match data: {match}")
                raise critical_refine_err  # Re-raise to be caught by the outer handler
            # --- End Inner Exception Handling ---

        logger.debug(
            f"Successfully refined {len(refined_matches)} matches on page {current_page}."
        )
        return refined_matches, total_pages  # Return refined list and total_pages

    # --- Outer Exception Handling ---
    except (
        ConnectionError,
        RequestException,
        NoSuchCookieException,
        WebDriverException,
    ) as known_err:
        logger.error(
            f"Known error get_matches page {current_page}: {type(known_err).__name__} - {known_err}",
            exc_info=False,
        )
        if isinstance(known_err, ConnectionError):
            raise known_err
        return None, None
    except Exception as e:
        # --- MODIFICATION: Robust outer exception logging ---
        exc_type, exc_value, tb = (
            sys.exc_info()
        )  # Use different name than imported module
        # Log basic info without traceback first
        logger.critical(
            f"CRITICAL outer error get_matches page {current_page}: {exc_type.__name__} - {exc_value}"
        )
        # Optionally log traceback separately if needed, more robustly
        try:
            tb_lines = traceback.format_exception(exc_type, exc_value, tb)
            logger.debug("Traceback:\n" + "".join(tb_lines))
        except Exception as log_tb_err:
            logger.error(
                f"Could not format traceback during critical error logging: {log_tb_err}"
            )
        # --- END MODIFICATION ---
        return None, None  # <<< Return None, None in outer exception block


# End of get_matches


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_combined_details(
    session_manager: SessionManager, match_uuid: str
) -> Optional[Dict[str, Any]]:
    """
    V14.34: Fetches combined match/profile details. No changes needed.
    """
    if not session_manager.my_uuid or not match_uuid:
        logger.warning("Missing my_uuid or match_uuid for combined details fetch.")
        return None
    if not session_manager.is_sess_valid():
        logger.error(
            f"Combined details fetch: WebDriver session invalid for UUID {match_uuid}."
        )
        raise ConnectionError(
            f"WebDriver session invalid for combined details fetch (UUID: {match_uuid})"
        )
    details_data = {}
    profile_data = {}
    combined_data = {}
    details_url = urljoin(
        config_instance.BASE_URL,
        f"/discoveryui-matchesservice/api/samples/{session_manager.my_uuid}/matches/{match_uuid}/details?pmparentaldata=true",
    )
    details_referer = urljoin(
        config_instance.BASE_URL,
        f"/discoveryui-matches/compare/{session_manager.my_uuid}/with/{match_uuid}",
    )
    try:
        # --- Fetch Match Details Part ---
        details_response = _api_req(
            url=details_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description="Match Details API (Batch)",
            referer_url=details_referer,
        )
        if details_response and isinstance(details_response, dict):
            details_data = details_response
            # Extract relevant fields
            combined_data["admin_profile_id"] = details_data.get("adminUcdmId")
            combined_data["admin_username"] = details_data.get("adminDisplayName")
            combined_data["tester_profile_id"] = details_data.get("userId")
            combined_data["tester_username"] = details_data.get("displayName")
            combined_data["tester_initials"] = details_data.get("displayInitials")
            combined_data["gender"] = details_data.get("subjectGender")
            relationship_part = details_data.get("relationship", {})
            combined_data["shared_segments"] = relationship_part.get("sharedSegments")
            combined_data["longest_shared_segment"] = relationship_part.get(
                "longestSharedSegment"
            )
            combined_data["meiosis"] = relationship_part.get("meiosis")
            combined_data["from_my_fathers_side"] = details_data.get(
                "fathersSide", False
            )
            combined_data["from_my_mothers_side"] = details_data.get(
                "mothersSide", False
            )
        else:
            logger.warning(
                f"Failed to get valid /details response for UUID {match_uuid}."
            )
            # Don't return yet, try profile details if possible

    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching /details for UUID {match_uuid}: {conn_err}",
            exc_info=False,
        )
        raise  # Re-raise to be handled by retry_api
    except Exception as e:
        logger.error(
            f"Error fetching /details for UUID {match_uuid}: {e}", exc_info=True
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise  # Re-raise specific request exceptions
        # For other exceptions, maybe return None or partial data? Returning None for now.
        return None

    # --- Fetch Profile Details Part ---
    tester_profile_id_for_api = combined_data.get("tester_profile_id")
    my_profile_id_header = session_manager.my_profile_id

    # Initialize profile-related fields in combined_data
    combined_data["last_logged_in_dt"] = None
    combined_data["contactable"] = False

    if not tester_profile_id_for_api:
        logger.debug(
            f"Skipping /profiles/details fetch for {match_uuid}: tester_profile_id not found in /details response."
        )
    elif not my_profile_id_header:
        logger.warning(
            f"Skipping /profiles/details fetch for {match_uuid}: Own profile ID missing for header."
        )
    elif (
        not session_manager.is_sess_valid()
    ):  # Check session again before next API call
        logger.error(
            f"Combined details fetch: WebDriver session invalid before fetching profile for {tester_profile_id_for_api}."
        )
        raise ConnectionError(  # Raise error to trigger retry if session died
            f"WebDriver session invalid before profile fetch (Profile: {tester_profile_id_for_api})"
        )
    else:
        profile_url = urljoin(
            config_instance.BASE_URL,
            f"/app-api/express/v1/profiles/details?userId={tester_profile_id_for_api.upper()}",
        )
        # Headers specific to profile API
        profile_headers = {
            # Headers populated by _api_req based on contextual config
        }
        try:
            profile_response = _api_req(
                url=profile_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                headers=profile_headers,  # Pass empty dict, _api_req adds contextual ones
                use_csrf_token=False,
                api_description="Profile Details API (Batch)",
                referer_url=details_referer,  # Reuse details referer
            )
            if profile_response and isinstance(profile_response, dict):
                profile_data = profile_response
                last_login_str = profile_data.get("LastLoginDate")
                contactable_val = profile_data.get("IsContactable", False)
                last_login_dt = None
                if last_login_str:
                    try:
                        if last_login_str.endswith("Z"):
                            last_login_dt = datetime.fromisoformat(
                                last_login_str.replace("Z", "+00:00")
                            )
                        else:  # Assume ISO format, make timezone aware (UTC)
                            dt_naive = datetime.fromisoformat(last_login_str)
                            last_login_dt = (
                                dt_naive.replace(tzinfo=timezone.utc)
                                if dt_naive.tzinfo is None
                                else dt_naive.astimezone(
                                    timezone.utc
                                )  # Convert if already aware
                            )
                        # Store the aware datetime object
                        combined_data["last_logged_in_dt"] = last_login_dt
                    except (ValueError, TypeError) as date_parse_err:
                        logger.warning(
                            f"Could not parse LastLoginDate '{last_login_str}' for {tester_profile_id_for_api}: {date_parse_err}"
                        )
                combined_data["contactable"] = bool(contactable_val)
            else:
                logger.warning(
                    f"Failed to get valid /profiles/details response for {tester_profile_id_for_api}. Type: {type(profile_response)}"
                )

        except NameError as ne:  # Catch potential timezone issues if library missing
            logger.critical(
                f"NameError in _fetch_combined_details (likely timezone): {ne}",
                exc_info=True,
            )
        except ConnectionError as conn_err:
            logger.error(
                f"ConnectionError fetching /profiles/details for {tester_profile_id_for_api}: {conn_err}",
                exc_info=False,
            )
            raise  # Re-raise to be handled by retry_api
        except Exception as e:
            logger.error(
                f"Error fetching /profiles/details for {tester_profile_id_for_api}: {e}",
                exc_info=True,
            )
            if isinstance(e, requests.exceptions.RequestException):
                raise  # Re-raise specific request exceptions

    # Return the combined data, even if profile fetch failed partially
    return (
        combined_data if combined_data else None
    )  # Return None only if initial details fetch failed entirely


# end _fetch_combined_details


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_batch_badge_details(
    session_manager: SessionManager, match_uuid: str
) -> Optional[Dict[str, Any]]:
    """
    V14.34: Fetches badge details for a match UUID. No changes needed.
    """
    if not session_manager.my_uuid or not match_uuid:
        logger.warning("Missing my_uuid or match_uuid for badge details fetch.")
        return None
    if not session_manager.is_sess_valid():
        logger.error(
            f"Badge details fetch: WebDriver session invalid for UUID {match_uuid}."
        )
        raise ConnectionError(
            f"WebDriver session invalid for badge details fetch (UUID: {match_uuid})"
        )
    badge_url = urljoin(
        config_instance.BASE_URL,
        f"/discoveryui-matchesservice/api/samples/{session_manager.my_uuid}/matches/{match_uuid}/badgedetails",
    )
    badge_referer = urljoin(
        config_instance.BASE_URL, "/discoveryui-matches/list/"
    )  # Referer for badge API
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
        if badge_response and isinstance(badge_response, dict):
            person_badged = badge_response.get("personBadged", {})
            raw_firstname = person_badged.get("firstName")
            # Format name robustly using helper
            their_firstname_formatted = (
                format_name(raw_firstname).split()[0]
                if raw_firstname and format_name(raw_firstname) != "Valued Relative"
                else "Unknown"
            )

            return {
                "their_cfpid": person_badged.get("personId"),
                "their_firstname": their_firstname_formatted,  # Use formatted first name part
                "their_lastname": person_badged.get("lastName", "Unknown"),
                "their_birth_year": person_badged.get("birthYear"),
            }
        else:
            logger.warning(
                f"Invalid badge details response for UUID {match_uuid}. Type: {type(badge_response)}"
            )
            return None
    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching badge details for UUID {match_uuid}: {conn_err}",
            exc_info=False,
        )
        raise
    except Exception as e:
        logger.error(
            f"Error fetching badge details for UUID {match_uuid}: {e}", exc_info=True
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None


# end _fetch_batch_badge_details


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_batch_ladder(
    session_manager: SessionManager, cfpid: str, tree_id: str
) -> Optional[Dict[str, Any]]:
    """
    V14.34: Fetches relationship ladder details. No changes needed.
    """
    if not cfpid or not tree_id:
        logger.warning("Missing cfpid or tree_id for ladder fetch.")
        return None
    if not session_manager.is_sess_valid():
        logger.error(f"Ladder fetch: WebDriver session invalid for CFPID {cfpid}.")
        raise ConnectionError(
            f"WebDriver session invalid for ladder fetch (CFPID: {cfpid})"
        )
    ladder_api_url = urljoin(
        config_instance.BASE_URL,
        f"family-tree/person/tree/{tree_id}/person/{cfpid}/getladder?callback=jQuery",
    )
    # Dynamic referer based on the facts page for the person
    dynamic_referer = urljoin(
        config_instance.BASE_URL,
        f"family-tree/person/tree/{tree_id}/person/{cfpid}/facts",
    )
    ladder_data = {}
    ladder_headers = {
        # Headers populated by _api_req based on contextual config
    }
    try:
        # Expecting JSONP, so force text response
        api_result = _api_req(
            url=ladder_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers=ladder_headers,
            use_csrf_token=False,
            api_description="Get Ladder API (Batch)",
            referer_url=dynamic_referer,
            force_text_response=True,
        )

        # Handle potential failure returns from _api_req
        if isinstance(api_result, requests.Response):  # Indicates non-2xx error
            logger.warning(
                f"Get Ladder API call failed for cfpid {cfpid} (Status: {api_result.status_code}). Returning None."
            )
            return None
        elif api_result is None:
            logger.warning(
                f"Get Ladder API call returned None (likely connection error after retries) for cfpid {cfpid}. Returning None."
            )
            return None
        elif not isinstance(
            api_result, str
        ):  # Should be string due to force_text_response=True
            logger.warning(
                f"_api_req returned unexpected type '{type(api_result).__name__}' for Get Ladder API for cfpid {cfpid}. Returning None."
            )
            return None

        response_text = api_result
        # Parse JSONP: Remove potential leading/trailing junk, extract JSON inside parentheses
        match_jsonp = re.match(
            r"^[^(]*\((.*)\)[^)]*$", response_text, re.DOTALL | re.IGNORECASE
        )
        if match_jsonp:
            json_string = match_jsonp.group(1).strip()
            try:
                if not json_string or json_string == '""' or json_string == "''":
                    logger.warning(
                        f"Empty JSON content within JSONP for cfpid {cfpid}."
                    )
                    return None  # No HTML to parse
                ladder_json = json.loads(json_string)

                if isinstance(ladder_json, dict) and "html" in ladder_json:
                    html_content = ladder_json["html"]
                    if html_content:
                        soup = BeautifulSoup(html_content, "html.parser")
                        actual_relationship_text = None
                        relationship_path_text = None

                        # Extract Actual Relationship (more robust selector)
                        rel_elem = soup.select_one(
                            "ul.textCenter > li:first-child > i > b"  # Original selector
                        )
                        if not rel_elem:  # Fallback selector if first doesn't work
                            rel_elem = soup.select_one(
                                "ul.textCenter > li > i > b"
                            )  # More generic within the list

                        if rel_elem:
                            raw_relationship = rel_elem.get_text(strip=True)
                            actual_relationship_text = ordinal_case(
                                raw_relationship.title()
                            )
                        else:
                            logger.warning(
                                f"Could not extract actual_relationship for cfpid: {cfpid}"
                            )

                        # Extract Relationship Path (simplified logic)
                        path_items = soup.select(
                            'ul.textCenter > li:not([class*="iconArrowDown"])'  # Select list items that are not arrows
                        )
                        path_list = []
                        num_items = len(path_items)

                        for i, item in enumerate(path_items):
                            name_text, desc_text = "", ""
                            # Find name (usually in <b> or <a><b>)
                            name_container = item.find("a") or item.find("b")
                            if name_container:
                                name_text = format_name(
                                    name_container.get_text(strip=True).replace(
                                        '"', "'"
                                    )
                                )

                            # Find description (usually in <i>, skip first item)
                            if i > 0:
                                desc_element = item.find("i")
                                if desc_element:
                                    raw_desc_full = desc_element.get_text(strip=True)
                                    cleaned_desc_full = raw_desc_full.replace('"', "'")
                                    # Handle "You are the..." case for the last item
                                    if (
                                        i == num_items - 1
                                        and cleaned_desc_full.lower().startswith(
                                            "you are the "
                                        )
                                    ):
                                        desc_text = format_name(
                                            cleaned_desc_full[
                                                len("You are the ") :
                                            ].strip()
                                        )
                                    else:
                                        # Try to parse "Relation of Person" format
                                        match_rel = re.match(
                                            r"^(.*?)\s+of\s+(.*)$",
                                            cleaned_desc_full,
                                            re.IGNORECASE,
                                        )
                                        if match_rel:
                                            desc_text = f"{match_rel.group(1).strip().capitalize()} of {format_name(match_rel.group(2).strip())}"
                                        else:  # Fallback to formatted full description
                                            desc_text = format_name(cleaned_desc_full)

                            if name_text:
                                path_list.append(
                                    f"{name_text} ({desc_text})"
                                    if desc_text
                                    else name_text
                                )

                        if path_list:
                            relationship_path_text = "\n↓\n".join(path_list)
                        else:
                            logger.warning(
                                f"Could not construct relationship_path for cfpid {cfpid}."
                            )

                        # Store extracted data
                        ladder_data["actual_relationship"] = actual_relationship_text
                        ladder_data["relationship_path"] = relationship_path_text
                        return ladder_data
                    else:
                        logger.warning(
                            f"Empty HTML in getladder response for cfpid {cfpid}."
                        )
                        return None  # No data if HTML is empty
                else:
                    logger.warning(
                        f"Missing 'html' key in getladder JSON for cfpid {cfpid}. JSON: {ladder_json}"
                    )
            except json.JSONDecodeError as inner_json_err:
                logger.error(
                    f"Failed to decode JSONP content for cfpid {cfpid}: {inner_json_err}"
                )
                logger.debug(f"JSON string causing decode error: '{json_string[:200]}'")
                return None
        else:  # JSONP regex didn't match
            logger.error(
                f"Could not parse JSONP format for cfpid {cfpid}. Response text: {response_text[:200]}"
            )
        return None  # Return None if JSONP parsing fails

    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching ladder for CFPID {cfpid}: {conn_err}",
            exc_info=False,
        )
        raise  # Re-raise to be handled by retry_api
    except Exception as e:
        logger.error(
            f"Error fetching/parsing ladder for CFPID {cfpid}: {e}", exc_info=True
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise  # Re-raise specific request exceptions
        return None  # Return None for other unexpected errors


# end _fetch_batch_ladder


# --- Refactored: _fetch_batch_relationship_prob uses shared cloudscraper instance ---
@retry_api(
    retry_on_exceptions=(
        requests.exceptions.RequestException,
        ConnectionError,
        cloudscraper.exceptions.CloudflareException,
    )
)
def _fetch_batch_relationship_prob(
    session_manager: SessionManager, match_uuid: str, max_labels_param: int
) -> Optional[str]:
    """
    V14.34: Fetches relationship probability using shared cloudscraper. No changes needed.
    """
    driver = session_manager.driver  # Get driver for cookie sync
    if not session_manager.my_uuid or not match_uuid:
        logger.warning(
            "Missing my_uuid or match_uuid for relationship probability fetch."
        )
        return "N/A (Error - Missing IDs)"

    scraper = session_manager.scraper  # Get shared scraper
    if not scraper:
        logger.error(
            "SessionManager scraper not initialized. Cannot fetch relationship probability."
        )
        raise ConnectionError("SessionManager scraper not initialized.")

    # Check driver validity before trying to get cookies
    if not driver or not session_manager.is_sess_valid():
        logger.error(
            f"Relationship prob fetch: Driver/session invalid for UUID {match_uuid}."
        )
        raise ConnectionError(
            f"WebDriver session invalid for relationship probability fetch (UUID: {match_uuid})"
        )

    my_uuid_upper = session_manager.my_uuid.upper()
    sample_id_upper = match_uuid.upper()
    rel_url = urljoin(
        config_instance.BASE_URL,
        f"discoveryui-matches/parents/list/api/matchProbabilityData/{my_uuid_upper}/{sample_id_upper}",
    )
    referer_url = urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/")

    # --- Prepare Headers (moved outside try block) ---
    # Base headers, CSRF handled below
    rel_headers = {
        "Accept": "application/json",
        # Other headers added by _api_req from config if needed, or set manually here
        "Referer": referer_url,
        "Origin": config_instance.BASE_URL.rstrip("/"),
        # Add other necessary headers like User-Agent, sec-ch-ua etc. if _api_req doesn't handle them for scraper
        "User-Agent": random.choice(config_instance.USER_AGENTS),  # Example
    }

    # --- Cookie and CSRF Handling ---
    csrf_token_val = None
    csrf_cookie_names = ("_dnamatches-matchlistui-x-csrf-token", "_csrf")
    try:
        driver_cookies_list = driver.get_cookies()
        if driver_cookies_list:
            # Sync cookies to shared scraper
            logger.debug(
                f"Updating shared scraper cookies from WebDriver ({len(driver_cookies_list)})..."
            )
            if hasattr(scraper, "cookies") and isinstance(
                scraper.cookies, RequestsCookieJar
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
            # Find CSRF token from driver cookies
            driver_cookies_dict = {
                c["name"]: c["value"]
                for c in driver_cookies_list
                if "name" in c and "value" in c
            }
            for name in csrf_cookie_names:
                if name in driver_cookies_dict and driver_cookies_dict[name]:
                    csrf_token_val = unquote(driver_cookies_dict[name]).split("|")[0]
                    rel_headers["X-CSRF-Token"] = csrf_token_val
                    logger.debug(
                        f"Using fresh CSRF token '{name}' from driver cookies."
                    )
                    break
        else:
            logger.warning("driver.get_cookies() returned None/empty.")
    except WebDriverException as csrf_wd_e:
        logger.warning(
            f"WebDriverException getting/setting cookies for CSRF: {csrf_wd_e}"
        )
        raise ConnectionError(
            f"WebDriver error getting/setting cookies for CSRF: {csrf_wd_e}"
        )
    except Exception as csrf_e:
        logger.warning(f"Could not get/set CSRF token from driver cookies: {csrf_e}")

    if "X-CSRF-Token" not in rel_headers:
        if session_manager.csrf_token:
            logger.warning("Using potentially stale CSRF from SessionManager.")
            rel_headers["X-CSRF-Token"] = session_manager.csrf_token
        else:
            logger.error(
                "Failed to add CSRF token to headers (fresh and fallback failed)."
            )
            return "N/A (Error - Missing CSRF)"

    api_description = "Match Probability API (Cloudscraper)"
    try:
        # --- Use Shared Scraper ---
        logger.debug(
            f"Making {api_description} POST request to {rel_url} using shared scraper"
        )
        response_rel = scraper.post(
            rel_url,
            headers=rel_headers,
            json={},  # Empty JSON payload expected by API
            allow_redirects=False,
            timeout=selenium_config.API_TIMEOUT,
        )
        # --- End Use Shared Scraper ---

        logger.debug(
            f"<-- {api_description} Response Status: {response_rel.status_code} {response_rel.reason}"
        )

        if not response_rel.ok:
            status_code = response_rel.status_code
            logger.warning(
                f"{api_description} failed for {sample_id_upper}. Status: {status_code}, Reason: {response_rel.reason}"
            )
            try:
                logger.debug(f"  Response Body: {response_rel.text[:500]}")
            except Exception:
                pass
            # Raise appropriate exception for retry_api
            response_rel.raise_for_status()  # Raises HTTPError for bad statuses
            # If raise_for_status doesn't raise (e.g., 404?), return specific error string
            return "N/A (API Error/Redirect)"

        # Process successful response
        try:
            if not response_rel.content:
                logger.warning(
                    f"{api_description}: OK ({response_rel.status_code}), but response body EMPTY."
                )
                return "N/A (Empty Response)"
            data = response_rel.json()
            if "matchProbabilityToSampleId" not in data:
                logger.warning(
                    f"Invalid data structure from {api_description} for {sample_id_upper}. Resp: {data}"
                )
                return "N/A (Invalid Data Structure)"

            prob_data = data["matchProbabilityToSampleId"]
            predictions = prob_data.get("relationships", {}).get("predictions", [])
            if not predictions:
                logger.debug(
                    f"No relationship predictions found for {sample_id_upper}. Marking as Distant."
                )
                return "Distant relationship?"

            # Filter and find best prediction
            valid_preds = [
                p
                for p in predictions
                if isinstance(p, dict)
                and "distributionProbability" in p
                and "pathsToMatch" in p
            ]
            if not valid_preds:
                logger.warning(
                    f"No valid prediction paths found for {sample_id_upper}."
                )
                return "N/A (No Valid Paths)"

            best_pred = max(
                valid_preds, key=lambda x: x.get("distributionProbability", 0.0)
            )
            top_prob = best_pred.get("distributionProbability", 0.0)
            paths = best_pred.get("pathsToMatch", [])
            labels = [
                p.get("label") for p in paths if isinstance(p, dict) and p.get("label")
            ]

            if not labels:
                logger.warning(
                    f"Prediction found for {sample_id_upper}, but no labels in paths."
                )
                return f"N/A (No Labels) [{top_prob:.1f}%]"

            final_labels = labels[:max_labels_param]  # Use parameter
            relationship_str = " or ".join(map(str, final_labels))
            return f"{relationship_str} [{top_prob:.1f}%]"

        except json.JSONDecodeError as json_err:
            logger.error(
                f"{api_description}: OK ({response_rel.status_code}), but JSON decode FAILED: {json_err}"
            )
            logger.debug(f"Response text: {response_rel.text[:500]}")
            raise RequestException("JSONDecodeError") from json_err  # Trigger retry
        except Exception as e:
            logger.error(
                f"{api_description}: Error processing successful response for {sample_id_upper}: {e}",
                exc_info=True,
            )
            raise RequestException("Response Processing Error") from e  # Trigger retry

    # Catch specific exceptions handled by retry_api or raise others
    except cloudscraper.exceptions.CloudflareException as cf_e:
        logger.error(
            f"{api_description}: Cloudflare challenge failed for {sample_id_upper}: {cf_e}"
        )
        raise  # Let retry_api handle
    except requests.exceptions.RequestException as req_e:
        logger.error(
            f"{api_description}: RequestException for {sample_id_upper}: {req_e}"
        )
        raise  # Let retry_api handle
    except Exception as e:
        logger.error(
            f"{api_description}: Unexpected error for {sample_id_upper}: {type(e).__name__} - {e}",
            exc_info=True,
        )
        # Raise a generic exception that retry_api might handle
        raise RequestException(f"Unexpected Fetch Error: {type(e).__name__}") from e


# end _fetch_batch_relationship_prob


#################################################################################
# 6. Utility & Helper Functions
#################################################################################


def _log_page_summary(page, page_new, page_updated, page_skipped, page_errors):
    """Logs a summary of processed matches for a single page."""
    logger.debug(f"---- Page {page} Summary ----")
    logger.debug(f"  New matches:     {page_new}")
    logger.debug(f"  Updated matches: {page_updated}")
    logger.debug(f"  Skipped matches: {page_skipped}")
    logger.debug(f"  Error matches:   {page_errors}")
    logger.debug("-----------------------\n")


# end of _log_page_summary


def _log_coord_summary(
    total_pages_processed, total_new, total_updated, total_skipped, total_errors
):
    """Logs the final summary of the coord's execution."""
    logger.info("---- Gather Matches Final Summary ----")
    logger.info(f"  Total Pages Processed: {total_pages_processed}")
    logger.info(f"  Total New Matches:     {total_new}")
    logger.info(f"  Total Updated Matches: {total_updated}")
    logger.info(f"  Total Skipped Matches: {total_skipped}")
    logger.info(f"  Total Errors:          {total_errors}")
    logger.info("------------------------------------\n")


# end of _log_coord_summary


def _adjust_delay(session_manager, page):
    """Adjusts the dynamic rate limiter delay after processing a page."""
    if session_manager.dynamic_rate_limiter.is_throttled():
        logger.debug(f"Rate limiter throttled during page {page}.")
    else:
        # Only decrease if not throttled
        previous_delay = session_manager.dynamic_rate_limiter.current_delay
        session_manager.dynamic_rate_limiter.decrease_delay()
        new_delay = session_manager.dynamic_rate_limiter.current_delay
        # Log decrease only if significant and above initial delay
        if (
            abs(previous_delay - new_delay) > 0.01
            and new_delay > config_instance.INITIAL_DELAY
        ):
            logger.debug(f"Decreased rate limit delay to {new_delay:.2f}s.")


# End of _adjust_delay


def nav_to_list(session_manager) -> bool:
    """Navigates directly to the user's specific DNA matches list page using their UUID."""
    if not session_manager.is_sess_valid() or not session_manager.my_uuid:
        logger.error("Session invalid or UUID missing for nav_to_list.")
        return False

    matches_url_with_uuid = urljoin(
        config_instance.BASE_URL, f"discoveryui-matches/list/{session_manager.my_uuid}"
    )
    logger.debug(f"Navigating to specific match list: {matches_url_with_uuid}")

    success = nav_to_page(
        session_manager.driver,
        matches_url_with_uuid,
        selector=MATCH_ENTRY_SELECTOR,  # Wait for a match entry to appear
        session_manager=session_manager,
    )

    if success:
        try:
            current_url = session_manager.driver.current_url
            if not current_url.startswith(matches_url_with_uuid):
                logger.warning(
                    f"Navigation OK, but final URL unexpected: {current_url}"
                )
            else:
                logger.debug("Successfully landed on specific matches list page.")
        except Exception as e:
            logger.warning(f"Could not verify final URL after nav_to_list: {e}")
    else:
        logger.error("Failed nav to specific matches list page.")

    return success


# end nav_to_list

# end of action6_gather.py
