#!/usr/bin/env python3

# action6_gather.py

# Standard library imports (alphabetical)
import json
import logging
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
import contextlib  # Needed for db_transn context manager

# Third-party imports (alphabetical by package)
import cloudscraper
import requests
from bs4 import BeautifulSoup, Tag

from requests.adapters import HTTPAdapter
from requests.cookies import RequestsCookieJar
from requests.exceptions import HTTPError, RequestException, ConnectionError
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
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, Session as SqlAlchemySession, joinedload
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Local application imports (alphabetical by module)
from cache import cache_result
from cache import cache as global_cache
from database import Person, DnaMatch, FamilyTree, db_transn, PersonStatusEnum

if global_cache is None:
    from cachetools import Cache
    global_cache = Cache(maxsize=1000)  # Initialize with a default cache if not set
from config import config_instance, selenium_config
from database import Person, DnaMatch, FamilyTree, db_transn, PersonStatusEnum
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

#################################################################################
# 2. Core Orchestration
#################################################################################


def coord(session_manager: SessionManager, config_instance, start: int = 1) -> bool:
    """
    V14.23: Gathers DNA matches, processing page-by-page.
    - Uses updated session state checks (driver_live, session_ready).
    """
    driver = session_manager.driver
    # --- Use updated session state checks ---
    if not driver or not session_manager.driver_live or not session_manager.session_ready:
        logger.error("WebDriver not initialized, driver not live, or session not ready. Exiting coord.")
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

    try:
        if not isinstance(start, int) or start <= 0:
            logger.warning(
                f"Invalid start parameter '{start}'. Using default start page 1."
            )
            start_page = 1
        else:
            start_page = start
    except Exception:
        logger.error(
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
        db_session_for_page = session_manager.get_db_conn()
        fetched_total_pages = None
        if not db_session_for_page:
            logger.error(f"Could not get DB session for initial page fetch. Aborting.")
            return False
        try:
            if not session_manager.is_sess_valid():
                logger.critical(
                    f"WebDriver session invalid before initial get_matches. Aborting run."
                )
                return False
            matches_on_page, fetched_total_pages = get_matches(
                session_manager, db_session_for_page, start_page
            )
        except ConnectionError as init_conn_e:
            logger.critical(
                f"ConnectionError during initial get_matches: {init_conn_e}. Aborting.",
                exc_info=False,
            )
            session_manager.return_session(db_session_for_page)
            return False
        except Exception as get_match_err:
            logger.error(
                f"Error during initial get_matches call on page {start_page}: {get_match_err}",
                exc_info=True,
            )
            session_manager.return_session(db_session_for_page)
            return False
        finally:
            if db_session_for_page:
                session_manager.return_session(db_session_for_page)

        if fetched_total_pages is None:
            logger.error(
                "Failed to retrieve total_pages on initial fetch (get_matches returned None). Aborting."
            )
            return False
        total_pages = fetched_total_pages
        logger.info(f"Total pages found: {total_pages}")

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
                bar_format="{percentage:3.0f}%|{bar}|",
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
                    db_session_for_page = session_manager.get_db_conn()
                    if not db_session_for_page:
                        logger.error(
                            f"Could not get DB session for page {current_page_num}. Skipping page."
                        )
                        total_errors += MATCHES_PER_PAGE
                        if progress_bar: # Increment error count in progress bar
                            progress_bar.update(MATCHES_PER_PAGE) # Assume full page error
                        # Allow more DB errors before aborting? Maybe configurable?
                        if total_errors > (10 * MATCHES_PER_PAGE): # e.g., allow 10 page errors
                            logger.critical(
                                "Aborting run due to persistent DB connection failures."
                            )
                            final_success = False
                            break
                        time.sleep(5) # Longer sleep on DB error
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
                        if progress_bar: progress_bar.update(MATCHES_PER_PAGE)
                        time.sleep(5)
                        current_page_num += 1
                        matches_on_page = []
                        continue
                    except Exception as get_match_e:
                        logger.error(
                            f"Error getting matches for page {current_page_num}: {get_match_e}",
                            exc_info=True,
                        )
                        total_errors += MATCHES_PER_PAGE
                        if progress_bar: progress_bar.update(MATCHES_PER_PAGE)
                        time.sleep(5)
                        current_page_num += 1
                        matches_on_page = []
                        continue
                    finally:
                        session_manager.return_session(db_session_for_page)

                if matches_on_page is None:
                    logger.warning(
                        f"get_matches returned None for page {current_page_num}. Skipping page processing."
                    )
                    total_errors += MATCHES_PER_PAGE
                    if progress_bar: progress_bar.update(MATCHES_PER_PAGE)
                    time.sleep(2)
                    current_page_num += 1
                    matches_on_page = []
                    continue
                elif not matches_on_page:
                    logger.info(f"No matches found on page {current_page_num}.")
                    # Only update progress bar if it's not the very first attempt where matches_on_page might be empty initially
                    if progress_bar and not (current_page_num == start_page and total_pages_processed == 0):
                        # Estimate update based on expected matches per page
                        progress_bar.update(MATCHES_PER_PAGE)
                    time.sleep(1)
                    current_page_num += 1
                    continue

                page_new, page_updated, page_skipped, page_errors = _do_batch(
                    session_manager=session_manager,
                    matches_on_page=matches_on_page,
                    current_page=current_page_num,
                    last_page_in_run=(last_page if last_page is not None else "?"),
                    max_labels_to_show=2,
                    progress_bar=progress_bar, # Pass the bar instance
                )
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
                inter_page_delay = session_manager.dynamic_rate_limiter.wait() # Use wait() for delay
                logger.debug(f"Rate limit inter-page delay: {inter_page_delay:.2f}s")
                current_page_num += 1
                matches_on_page = [] # Clear for next iteration

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
            print(file=sys.stderr) # Newline after final bar
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


def _do_batch(
    session_manager: SessionManager,
    matches_on_page: List[Dict[str, Any]],
    current_page: int,
    last_page_in_run: Union[int, str],
    max_labels_to_show: int,
    progress_bar: Optional[tqdm] = None,
) -> Tuple[int, int, int, int]:
    """
    V14.29 FIX: Corrects SyntaxError in ladder pre-fetch exception handling.
    Processes batch using optimized pre-fetch & BULK operations for Person/DNA/Tree.
    """
    # ...(Initialization, DB Lookup, Candidate Identification - unchanged)...
    page_new, page_updated, page_skipped, page_errors = 0, 0, 0, 0
    num_matches_on_page = len(matches_on_page)
    my_uuid = session_manager.my_uuid
    my_tree_id = session_manager.my_tree_id

    if not my_uuid:
        logger.error(f"_do_batch Page {current_page}: Missing my_uuid.")
        return 0, 0, 0, num_matches_on_page
    logger.debug(
        f"--- Starting Optimized Batch for Page {current_page} ({num_matches_on_page} matches) ---"
    )
    uuids_on_page = [m["uuid"] for m in matches_on_page if m.get("uuid")]
    existing_persons_map: Dict[str, Person] = {}
    session = session_manager.get_db_conn()
    if not session:
        logger.error(f"_do_batch Page {current_page}: Failed to get DB session.")
        progress_bar.update(num_matches_on_page) if progress_bar else None
        return 0, 0, 0, num_matches_on_page
    try:
        if uuids_on_page:
            existing_persons = (
                session.query(Person)
                .options(joinedload(Person.dna_match), joinedload(Person.family_tree))
                .filter(Person.uuid.in_(uuids_on_page))
                .all()
            )
            existing_persons_map = {person.uuid: person for person in existing_persons}
            logger.debug(
                f"Found {len(existing_persons_map)} existing Person records for page {current_page}."
            )
    except SQLAlchemyError as db_lookup_err:
        logger.error(
            f"Initial DB lookup failed page {current_page}: {db_lookup_err}",
            exc_info=True,
        )
        session_manager.return_session(session)
        progress_bar.update(num_matches_on_page) if progress_bar else None
        return 0, 0, 0, num_matches_on_page
    except Exception as e:
        logger.error(
            f"Unexpected error during initial DB lookup page {current_page}: {e}",
            exc_info=True,
        )
        session_manager.return_session(session)
        progress_bar.update(num_matches_on_page) if progress_bar else None
        return 0, 0, 0, num_matches_on_page
    finally:
        pass
    fetch_candidates_uuid: Set[str] = set()
    skipped_count_this_batch = 0
    matches_to_process_later: List[Dict[str, Any]] = []
    logger.debug("Identifying fetch candidates and skipped matches...")
    for match in matches_on_page:
        uuid_val = match.get("uuid")
        if not uuid_val:
            logger.warning(f"Skipping match due to missing UUID: {match}")
            page_errors += 1
            progress_bar.update(1) if progress_bar else None
            continue
        existing_person = existing_persons_map.get(uuid_val)
        if not existing_person:
            fetch_candidates_uuid.add(uuid_val)
            matches_to_process_later.append(match)
        else:
            needs_fetch = False
            existing_dna = existing_person.dna_match
            existing_tree = existing_person.family_tree
            if existing_dna:
                api_cm = match.get("cM_DNA")
                db_cm = existing_dna.cM_DNA
                api_segments = match.get("numSharedSegments")
                db_segments = existing_dna.shared_segments
                if api_cm is not None and db_cm is not None and int(api_cm) != db_cm:
                    needs_fetch = True
                if (
                    api_segments is not None
                    and db_segments is not None
                    and int(api_segments) != db_segments
                ):
                    needs_fetch = True
            else:
                needs_fetch = True
            api_in_tree = match.get("in_my_tree", False)
            db_in_tree = existing_person.in_my_tree
            if bool(api_in_tree) != bool(db_in_tree):
                needs_fetch = True
            elif api_in_tree and not existing_tree:
                needs_fetch = True
            if needs_fetch:
                fetch_candidates_uuid.add(uuid_val)
                matches_to_process_later.append(match)
            else:
                skipped_count_this_batch += 1
    if progress_bar and skipped_count_this_batch > 0:
        progress_bar.update(skipped_count_this_batch)
        page_skipped += skipped_count_this_batch
    logger.debug(
        f"Identified {len(fetch_candidates_uuid)} fetch candidates and {skipped_count_this_batch} skipped matches."
    )

    # --- 3. Targeted Pre-fetching ---
    batch_combined_details: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_badge_data: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_ladder_data: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_relationship_prob_data: Dict[str, Optional[str]] = {}
    futures = {}
    fetch_start_time = time.time()
    if fetch_candidates_uuid:
        logger.debug(
            f"--- Starting Targeted Pre-fetch for Page {current_page} ({len(fetch_candidates_uuid)} candidates) ---"
        )
        uuids_for_tree_badge = {
            uuid
            for uuid in fetch_candidates_uuid
            if any(
                m["uuid"] == uuid and m.get("in_my_tree")
                for m in matches_to_process_later
            )
        }
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit combined, relationship prob, and initial badge details
            for uuid_val in fetch_candidates_uuid:
                delay = session_manager.dynamic_rate_limiter.wait()
                futures[
                    executor.submit(_fetch_combined_details, session_manager, uuid_val)
                ] = ("combined_details", uuid_val)
                delay = session_manager.dynamic_rate_limiter.wait()
                futures[
                    executor.submit(
                        _fetch_batch_relationship_prob,
                        session_manager,
                        uuid_val,
                        max_labels_to_show,
                    )
                ] = ("relationship_prob", uuid_val)
            for uuid_val in uuids_for_tree_badge:
                delay = session_manager.dynamic_rate_limiter.wait()
                futures[
                    executor.submit(
                        _fetch_batch_badge_details, session_manager, uuid_val
                    )
                ] = ("badge_details", uuid_val)

            # Process initial results
            temp_badge_results = {}
            for future in as_completed(futures):
                task_type, identifier = futures[future]
                try:  # Outer try for general future processing
                    result = future.result()
                    if result is not None:
                        if task_type == "combined_details":
                            batch_combined_details[identifier] = result
                        elif task_type == "badge_details":
                            temp_badge_results[identifier] = result
                        elif task_type == "relationship_prob":
                            batch_relationship_prob_data[identifier] = result
                except ConnectionError as conn_err:
                    logger.error(
                        f"ConnErr pre-fetch '{task_type}' for {identifier}: {conn_err}",
                        exc_info=False,
                    )
                    batch_relationship_prob_data[identifier] = "N/A (Conn Error)"
                except Exception as exc:
                    logger.error(
                        f"Exc pre-fetch '{task_type}' for {identifier}: {exc}",
                        exc_info=False,
                    )
                    batch_relationship_prob_data[identifier] = "N/A (Fetch Error)"

            # --- Ladder Pre-fetch (depends on badge results) ---
            cfpid_to_uuid_map = {}
            ladder_futures = {}
            if my_tree_id and temp_badge_results:
                cfpid_list = []
                for uuid_val, badge_result in temp_badge_results.items():
                    cfpid = badge_result.get("their_cfpid")
                    if cfpid:
                        cfpid_list.append(cfpid)
                        cfpid_to_uuid_map[cfpid] = uuid_val
                if cfpid_list:
                    logger.debug(
                        f"Submitting ladder pre-fetch for {len(cfpid_list)} CFPIDs..."
                    )
                    for cfpid in cfpid_list:
                        delay = session_manager.dynamic_rate_limiter.wait()
                        ladder_futures[
                            executor.submit(
                                _fetch_batch_ladder, session_manager, cfpid, my_tree_id
                            )
                        ] = ("ladder", cfpid)

            # Process ladder results separately
            for future in as_completed(ladder_futures):
                task_type, cfpid = ladder_futures[future]
                # --- MODIFICATION: Corrected try/except block for ladder results ---
                try:  # Inner try specifically for ladder future results
                    result = future.result()
                    if result is not None:
                        batch_ladder_data[cfpid] = result
                    # else: logger.debug(f"Pre-fetch task 'ladder' for CFPID {cfpid} returned None.") # Verbose
                except ConnectionError as conn_err:
                    logger.error(
                        f"ConnErr pre-fetch 'ladder' for CFPID {cfpid}: {conn_err}",
                        exc_info=False,
                    )
                    # Optionally mark related entry as error? For now, just log.
                except Exception as exc:
                    logger.error(
                        f"Exc pre-fetch 'ladder' for CFPID {cfpid}: {exc}",
                        exc_info=False,
                    )
                    # Optionally mark related entry as error? For now, just log.
                # --- END MODIFICATION ---
        # --- End Ladder Fetch ---

        fetch_duration = time.time() - fetch_start_time
        logger.debug(
            f"--- Finished Targeted Pre-fetch for Page {current_page}. Duration: {fetch_duration:.2f}s ---"
        )
        batch_tree_data: Dict[str, Dict[str, Any]] = {}
        for uuid_val, badge_result in temp_badge_results.items():
            combined_tree_info = badge_result.copy()
            cfpid = badge_result.get("their_cfpid")
            if cfpid and cfpid in batch_ladder_data:
                combined_tree_info.update(batch_ladder_data[cfpid])
            batch_tree_data[uuid_val] = combined_tree_info
    else:
        logger.debug("No fetch candidates identified for this batch.")

    # --- 4. Process & Prepare Bulk Data ---
    # ...(Unchanged from V14.28)...
    prepared_bulk_data: List[Dict[str, Any]] = []
    page_statuses: Dict[str, int] = {
        "new": 0,
        "updated": 0,
        "skipped": skipped_count_this_batch,
        "error": page_errors,
    }
    process_start_time = time.time()
    logger.debug(f"--- Processing {len(matches_to_process_later)} candidates ---")
    for match in matches_to_process_later:
        uuid_val = match.get("uuid")
        _case_name = match.get("username", f"Unknown Match UUID {uuid_val}")
        try:
            existing_person = existing_persons_map.get(uuid_val)
            prefetched_combined = batch_combined_details.get(uuid_val)
            prefetched_tree = batch_tree_data.get(uuid_val)
            prefetched_rel_prob = batch_relationship_prob_data.get(uuid_val)
            match["predicted_relationship"] = (
                prefetched_rel_prob or "N/A (Fetch Failed)"
            )
            if not session_manager.is_sess_valid():
                logger.error(f"WD session invalid before _do_match for {_case_name}.")
                page_statuses["error"] += 1
                continue
            prepared_data, status, error_msg = _do_match(
                session=session,
                match=match,
                session_manager=session_manager,
                existing_person_arg=existing_person,
                prefetched_combined_details=prefetched_combined,
                prefetched_tree_data=prefetched_tree,
            )
            person_data_from_do_match = (
                prepared_data.get("person") if prepared_data else None
            )
            profile_id_from_do_match = (
                person_data_from_do_match.get("profile_id")
                if person_data_from_do_match
                else "N/A"
            )
            # logger.debug(f"  _do_match result for {_case_name} (UUID:{uuid_val}): Status='{status}', Prepared ProfileID='{profile_id_from_do_match}'") # Verbose
            page_statuses[status] += 1
            if status != "error" and prepared_data:
                prepared_bulk_data.append(prepared_data)
            elif status == "error":
                logger.error(f"Error preparing DB data for {_case_name}: {error_msg}")
        except Exception as inner_e:
            logger.error(
                f"Critical error processing candidate {_case_name} page {current_page}: {inner_e}",
                exc_info=True,
            )
            page_statuses["error"] += 1
        finally:
            if progress_bar:
                try:
                    progress_bar.update(1)
                except Exception as pbar_e:
                    logger.warning(f"Error updating progress bar: {pbar_e}")

    # --- 5. Bulk DB Operations ---
    # ...(Unchanged from V14.28 - includes de-duplication, detailed logging, post-flush ID query)...
    if prepared_bulk_data:
        logger.debug(
            f"--- Starting Bulk DB Operations for Page {current_page} ({len(prepared_bulk_data)} items) ---"
        )
        bulk_start_time = time.time()
        try:
            with db_transn(session):
                logger.debug(
                    f"Entered transaction block for bulk operations page {current_page}."
                )
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
                ]
                family_tree_creates = [
                    d["family_tree"]
                    for d in prepared_bulk_data
                    if d.get("family_tree")
                    and d["family_tree"]["_operation"] == "create"
                ]
                family_tree_updates = [
                    d["family_tree"]
                    for d in prepared_bulk_data
                    if d.get("family_tree")
                    and d["family_tree"]["_operation"] == "update"
                ]
                created_person_map: Dict[str, int] = {}

                person_creates_filtered = []
                seen_profile_ids = set()
                skipped_duplicates = 0
                if person_creates_raw:
                    logger.debug(
                        f"De-duplicating {len(person_creates_raw)} raw person create entries based on profile_id..."
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
                                f"Skipping duplicate Person create entry for Profile ID: {profile_id} (UUID: {p_data.get('uuid')})"
                            )
                            skipped_duplicates += 1
                    if skipped_duplicates > 0:
                        logger.warning(
                            f"Skipped {skipped_duplicates} duplicate person create entries."
                        )
                    logger.debug(
                        f"Proceeding with {len(person_creates_filtered)} unique person create entries."
                    )

                if person_creates_filtered:
                    logger.debug(
                        f"Bulk inserting {len(person_creates_filtered)} new Person records..."
                    )
                    insert_data = [
                        {k: v for k, v in p.items() if not k.startswith("_")}
                        for p in person_creates_filtered
                    ]
                    default_status = PersonStatusEnum.ACTIVE
                    for item_data in insert_data:
                        if "status" not in item_data or item_data["status"] is None:
                            item_data["status"] = default_status
                    profile_ids_in_insert_data = [
                        item.get("profile_id") for item in insert_data
                    ]
                    logger.debug(
                        f"Profile IDs being sent to bulk_insert_mappings: {profile_ids_in_insert_data}"
                    )
                    non_null_profile_ids = [
                        pid for pid in profile_ids_in_insert_data if pid is not None
                    ]
                    if len(non_null_profile_ids) != len(set(non_null_profile_ids)):
                        logger.error(
                            "CRITICAL: Duplicate non-NULL profile IDs detected in insert_data JUST BEFORE bulk insert!"
                        )
                        from collections import Counter

                        id_counts = Counter(non_null_profile_ids)
                        duplicates = {
                            pid: count for pid, count in id_counts.items() if count > 1
                        }
                        logger.error(f"Duplicate Profile IDs found: {duplicates}")
                    else:
                        logger.debug(
                            "Verified uniqueness of non-NULL profile IDs in insert_data."
                        )
                    session.bulk_insert_mappings(Person, insert_data)
                    logger.debug("Flushing session to obtain new Person IDs...")
                    session.flush()
                    logger.debug("Session flushed.")
                    inserted_uuids = [
                        p_data["uuid"] for p_data in insert_data if p_data.get("uuid")
                    ]
                    if inserted_uuids:
                        logger.debug(
                            f"Querying for IDs of {len(inserted_uuids)} inserted persons using UUIDs..."
                        )
                        newly_inserted_persons = (
                            session.query(Person.id, Person.uuid)
                            .filter(Person.uuid.in_(inserted_uuids))
                            .all()
                        )
                        created_person_map = {
                            p_uuid: p_id for p_id, p_uuid in newly_inserted_persons
                        }
                        logger.debug(
                            f"Mapped {len(created_person_map)} new Person IDs using UUIDs."
                        )
                        if len(created_person_map) != len(inserted_uuids):
                            logger.error(
                                f"CRITICAL: ID mapping mismatch! Expected {len(inserted_uuids)}, found {len(created_person_map)}."
                            )
                            missing_uuids = set(inserted_uuids) - set(
                                created_person_map.keys()
                            )
                            logger.error(f"Missing UUIDs: {missing_uuids}")
                    else:
                        logger.warning(
                            "No UUIDs found in insert_data to query for IDs."
                        )
                else:
                    logger.debug("No unique Person records to bulk insert.")

                if person_updates:
                    update_mappings = []
                    for p_data in person_updates:
                        existing_id = p_data.get("_existing_person_id")
                        if not existing_id:
                            logger.warning(
                                f"Skipping person update for UUID {p_data.get('uuid')}: Missing existing ID."
                            )
                            continue
                        update_dict = {
                            k: v
                            for k, v in p_data.items()
                            if not k.startswith("_") and k not in ["uuid", "profile_id"]
                        }
                        if update_dict:
                            update_dict["id"] = existing_id
                            update_dict["updated_at"] = datetime.now(timezone.utc)
                            update_mappings.append(update_dict)
                    if update_mappings:
                        logger.debug(
                            f"Bulk updating {len(update_mappings)} existing Person records..."
                        )
                        session.bulk_update_mappings(Person, update_mappings)
                        logger.debug("Bulk updated persons.")
                    else:
                        logger.debug("No Person records needed bulk updating.")

                all_person_ids_map = created_person_map.copy()
                for p_data in person_updates:
                    if p_data.get("_existing_person_id") and p_data.get("uuid"):
                        all_person_ids_map[p_data["uuid"]] = p_data[
                            "_existing_person_id"
                        ]
                for uuid_processed in {
                    p["person"]["uuid"] for p in prepared_bulk_data if p.get("person")
                }:
                    if (
                        uuid_processed not in all_person_ids_map
                        and existing_persons_map.get(uuid_processed)
                    ):
                        all_person_ids_map[uuid_processed] = existing_persons_map[
                            uuid_processed
                        ].id

                if dna_match_creates:
                    dna_insert_data = []
                    for dna_data in dna_match_creates:
                        person_uuid = dna_data.get("uuid")
                        person_id = all_person_ids_map.get(person_uuid)
                        if person_id:
                            insert_dict = {
                                k: v
                                for k, v in dna_data.items()
                                if not k.startswith("_")
                            }
                            insert_dict["people_id"] = person_id
                            dna_insert_data.append(insert_dict)
                        else:
                            logger.warning(
                                f"Skipping DNA Match create for UUID {person_uuid}: Corresponding Person ID not found in final map."
                            )
                    if dna_insert_data:
                        logger.debug(
                            f"Bulk inserting {len(dna_insert_data)} DnaMatch records..."
                        )
                        session.bulk_insert_mappings(DnaMatch, dna_insert_data)
                        logger.debug("Bulk inserted DnaMatches.")
                    else:
                        logger.debug("No valid DnaMatch records to bulk insert.")

                if family_tree_creates:
                    tree_insert_data = []
                    for tree_data in family_tree_creates:
                        person_uuid = tree_data.get("uuid")
                        person_id = all_person_ids_map.get(person_uuid)
                        if person_id:
                            insert_dict = {
                                k: v
                                for k, v in tree_data.items()
                                if not k.startswith("_")
                            }
                            insert_dict["people_id"] = person_id
                            tree_insert_data.append(insert_dict)
                        else:
                            logger.warning(
                                f"Skipping FamilyTree create for UUID {person_uuid}: Corresponding Person ID not found."
                            )
                    if tree_insert_data:
                        logger.debug(
                            f"Bulk inserting {len(tree_insert_data)} FamilyTree records..."
                        )
                        session.bulk_insert_mappings(FamilyTree, tree_insert_data)
                        logger.debug("Bulk inserted FamilyTrees.")
                if family_tree_updates:
                    tree_update_mappings = []
                    for tree_data in family_tree_updates:
                        existing_tree_id = tree_data.get("_existing_tree_id")
                        if not existing_tree_id:
                            logger.warning(
                                f"Skipping FT update for UUID {tree_data.get('uuid')}: Missing existing ID."
                            )
                            continue
                        update_dict_tree = {
                            k: v
                            for k, v in tree_data.items()
                            if not k.startswith("_") and k != "uuid"
                        }
                        if update_dict_tree:
                            update_dict_tree["id"] = existing_tree_id
                            update_dict_tree["updated_at"] = datetime.now(timezone.utc)
                            person_id_tree = all_person_ids_map.get(
                                tree_data.get("uuid")
                            )
                            if person_id_tree and "people_id" not in update_dict_tree:
                                update_dict_tree["people_id"] = person_id_tree
                            tree_update_mappings.append(update_dict_tree)
                    if tree_update_mappings:
                        logger.debug(
                            f"Bulk updating {len(tree_update_mappings)} FamilyTree records..."
                        )
                        session.bulk_update_mappings(FamilyTree, tree_update_mappings)
                        logger.debug("Bulk updated FamilyTrees.")
                    else:
                        logger.debug("No FamilyTree records needed bulk updating.")

                logger.debug(
                    f"Exiting transaction block for bulk ops page {current_page} (Commit follows)."
                )

            bulk_duration = time.time() - bulk_start_time
            logger.debug(
                f"Bulk operations for page {current_page} completed. Duration: {bulk_duration:.2f}s."
            )

        except IntegrityError as bulk_integrity_err:
            logger.error(
                f"Bulk DB op FAILED page {current_page} (IntegrityError): {bulk_integrity_err}",
                exc_info=True,
            )
            failed_items = len(prepared_bulk_data)
            page_statuses["error"] += failed_items
            page_statuses["new"] = 0
            page_statuses["updated"] = 0
            logger.warning(f"Page {current_page} counts adjusted: {page_statuses}")
        except SQLAlchemyError as bulk_db_err:
            logger.error(
                f"Bulk DB op FAILED page {current_page} (SQLAlchemyError): {bulk_db_err}",
                exc_info=True,
            )
            failed_items = len(prepared_bulk_data)
            page_statuses["error"] += failed_items
            page_statuses["new"] = 0
            page_statuses["updated"] = 0
            logger.warning(f"Page {current_page} counts adjusted: {page_statuses}")
        except Exception as bulk_e_unexp:
            logger.critical(
                f"Unexpected Bulk DB Error page {current_page}: {bulk_e_unexp}",
                exc_info=True,
            )
            failed_items = len(prepared_bulk_data)
            page_statuses["error"] += failed_items
            page_statuses["new"] = 0
            page_statuses["updated"] = 0
            logger.warning(f"Page {current_page} counts adjusted: {page_statuses}")
    else:
        logger.debug(f"No data prepared for bulk DB operations on page {current_page}.")

    # Final return uses aggregated statuses
    return (
        page_statuses["new"],
        page_statuses["updated"],
        page_statuses["skipped"],
        page_statuses["error"],
    )
# end of _do_batch


def _do_match(
    session: Session,
    match: Dict[str, Any],
    session_manager: SessionManager,
    existing_person_arg: Optional[Person],  # Argument is kept
    prefetched_combined_details: Optional[Dict[str, Any]],
    prefetched_tree_data: Optional[Dict[str, Any]],
) -> Tuple[
    Optional[Dict[str, Any]],
    Literal["new", "updated", "skipped", "error"],
    Optional[str],
]:
    """
    V14.30 REVISED: Removes redundant fallback DB lookup. Relies solely on
    existing_person_arg provided by _do_batch.
    Processes match data, uses pre-fetched existing_person,
    and returns prepared data dictionary for bulk operations.
    """
    # --- Use existing_person_arg if provided ---
    existing_person: Optional[Person] = existing_person_arg
    # --- End modification ---

    dna_match_record: Optional[DnaMatch] = (
        existing_person.dna_match if existing_person else None
    )
    family_tree_record: Optional[FamilyTree] = (
        existing_person.family_tree if existing_person else None
    )
    match_uuid = match.get("uuid")
    match_username_raw = match.get("username")
    match_username = format_name(match_username_raw)
    predicted_relationship = match.get("predicted_relationship", "N/A")
    match_in_my_tree = match.get("in_my_tree", False)
    log_ref = f"UUID={match_uuid or 'N/A'} User='{match_username or 'Unknown'}'"
    log_ref_short = f"UUID={match_uuid} User='{match_username}'"
    prepared_data_for_bulk: Dict[str, Any] = {
        "person": None,
        "dna_match": None,
        "family_tree": None,
    }
    person_update_needed: bool = False
    tree_update_needed: bool = False
    overall_status: Literal["new", "updated", "skipped", "error"] = "error"

    if not match_uuid:
        error_msg = f"Pre-check failed: Missing 'uuid' in match data: {match}"
        logger.error(error_msg)
        return None, "error", error_msg

    try:
        # --- REMOVED Fallback DB Lookup ---
        # The 'if not existing_person_arg:' block that called get_person_by_uuid is removed.
        # --- END REMOVAL ---

        is_new_person = existing_person is None

        # Step 2: Prepare Incoming Data & Determine Profile/Admin IDs based on 4 scenarios
        # ...(logic for determining IDs, message link, birth year - unchanged)...
        details_part = prefetched_combined_details or {}
        profile_part = prefetched_combined_details or {}  # Use same source for now
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
        person_profile_id_to_save = None
        person_admin_id_to_save = None
        person_admin_username_to_save = None
        if tester_profile_id_upper and admin_profile_id_upper:
            if tester_profile_id_upper == admin_profile_id_upper:
                if (
                    match_username
                    and formatted_admin_username
                    and match_username.lower() == formatted_admin_username.lower()
                ):
                    # Scenario D: Admin's own test
                    person_profile_id_to_save = tester_profile_id_upper
                    person_admin_id_to_save = None
                    person_admin_username_to_save = None
                else:
                    # Scenario C: Managed Non-Member
                    person_profile_id_to_save = None
                    person_admin_id_to_save = admin_profile_id_upper
                    person_admin_username_to_save = formatted_admin_username
            else:  # Scenario B: Managed Member
                person_profile_id_to_save = tester_profile_id_upper
                person_admin_id_to_save = admin_profile_id_upper
                person_admin_username_to_save = formatted_admin_username
        elif (
            tester_profile_id_upper and not admin_profile_id_upper
        ):  # Scenario A: Self-Managed Member
            person_profile_id_to_save = tester_profile_id_upper
            person_admin_id_to_save = None
            person_admin_username_to_save = None
        elif (
            not tester_profile_id_upper and admin_profile_id_upper
        ):  # Likely Scenario C variation
            person_profile_id_to_save = None
            person_admin_id_to_save = admin_profile_id_upper
            person_admin_username_to_save = formatted_admin_username
        else:
            logger.warning(f"{log_ref}: Neither tester nor admin profile ID found.")
            person_profile_id_to_save = None
            person_admin_id_to_save = None
            person_admin_username_to_save = None
        message_target_id = person_admin_id_to_save or person_profile_id_to_save
        constructed_message_link = None
        if message_target_id and session_manager.my_uuid:
            target_upper = message_target_id
            my_uuid_upper = session_manager.my_uuid.upper()
            match_uuid_upper = match_uuid.upper()
            constructed_message_link = urljoin(
                config_instance.BASE_URL,
                f"/messaging/?p={target_upper}&testguid1={my_uuid_upper}&testguid2={match_uuid_upper}",
            )
        birth_year_val = None
        if prefetched_tree_data and prefetched_tree_data.get("their_birth_year"):
            try:
                birth_year_val = int(prefetched_tree_data["their_birth_year"])
            except (ValueError, TypeError):
                pass

        incoming_person_data = {
            "uuid": match_uuid.upper(),
            "profile_id": person_profile_id_to_save,
            "username": match_username,
            "administrator_profile_id": person_admin_id_to_save,
            "administrator_username": person_admin_username_to_save,
            "in_my_tree": match_in_my_tree,
            "first_name": match.get("first_name"),
            "last_logged_in": profile_part.get("last_logged_in_dt"),
            "contactable": profile_part.get("contactable", False),
            "gender": details_part.get("gender"),
            "message_link": constructed_message_link,
            "birth_year": birth_year_val,
        }

        # ...(Prepare incoming DNA/Tree data - unchanged)...
        incoming_dna_data = None
        needs_dna_create_or_update = False
        if dna_match_record is None:
            needs_dna_create_or_update = True
        elif prefetched_combined_details:
            api_cm = match.get("cM_DNA")
            db_cm = dna_match_record.cM_DNA
            if api_cm is not None and db_cm is not None and int(api_cm) != db_cm:
                needs_dna_create_or_update = True
        if needs_dna_create_or_update and prefetched_combined_details is not None:
            incoming_dna_data = {
                "uuid": match_uuid.upper(),
                "compare_link": match.get("compare_link"),
                "cM_DNA": match.get("cM_DNA"),
                "predicted_relationship": predicted_relationship,
                "shared_segments": prefetched_combined_details.get("shared_segments"),
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
                "_operation": "create",
            }
        elif needs_dna_create_or_update and prefetched_combined_details is None:
            logger.warning(
                f"{log_ref}: DNA Match needs create/update, but no details fetched."
            )

        incoming_tree_data = None
        should_have_tree = match_in_my_tree
        tree_operation: Literal["create", "update", "none"] = "none"
        if should_have_tree and family_tree_record is None:
            tree_operation = "create"
        elif should_have_tree and family_tree_record is not None:
            if prefetched_tree_data:
                fields_to_check = [
                    "cfpid",
                    "person_name_in_tree",
                    "facts_link",
                    "view_in_tree_link",
                    "actual_relationship",
                    "relationship_path",
                ]
                for field in fields_to_check:
                    new_val = prefetched_tree_data.get(field)
                    old_val = getattr(family_tree_record, field, None)
                    if new_val != old_val:
                        tree_operation = "update"
                        break
            # else: logger.debug(f"{log_ref}: Tree record exists, assuming no update needed.") # Verbose
        elif not should_have_tree and family_tree_record is not None:
            logger.warning(
                f"{log_ref}: Data mismatch: Not 'in_my_tree', but FT record exists (ID: {family_tree_record.id}). Not deleting."
            )
            tree_operation = "none"
        if tree_operation != "none" and prefetched_tree_data:
            view_in_tree_link, facts_link = None, None
            their_cfpid_final = prefetched_tree_data.get("their_cfpid")
            if their_cfpid_final and session_manager.my_tree_id:
                base_tree_url = urljoin(
                    config_instance.BASE_URL,
                    f"/family-tree/person/tree/{session_manager.my_tree_id}/person/{their_cfpid_final}",
                )
                view_in_tree_link = urljoin(base_tree_url, "family")
                facts_link = urljoin(base_tree_url, "facts")
            tree_person_name = prefetched_tree_data.get("their_firstname", "Unknown")
            incoming_tree_data = {
                "uuid": match_uuid.upper(),
                "cfpid": their_cfpid_final,
                "person_name_in_tree": tree_person_name,
                "facts_link": facts_link,
                "view_in_tree_link": view_in_tree_link,
                "actual_relationship": prefetched_tree_data.get("actual_relationship"),
                "relationship_path": prefetched_tree_data.get("relationship_path"),
                "_operation": tree_operation,
                "_existing_tree_id": (
                    family_tree_record.id
                    if family_tree_record and tree_operation == "update"
                    else None
                ),
            }
        elif tree_operation != "none" and not prefetched_tree_data:
            logger.warning(
                f"{log_ref}: FamilyTree needs {tree_operation}, but no tree details fetched."
            )

        # Step 3: Compare and Build Bulk Data Dictionary
        # ...(Logic for comparing fields and building prepared_data_for_bulk unchanged)...
        if is_new_person:
            # logger.debug(f"{log_ref}: Preparing data for NEW Person.") # Verbose
            person_data_for_bulk = incoming_person_data.copy()
            person_data_for_bulk["_operation"] = "create"
            prepared_data_for_bulk["person"] = person_data_for_bulk
            if incoming_dna_data:
                prepared_data_for_bulk["dna_match"] = (
                    incoming_dna_data  # logger.debug(f"{log_ref}: Prep NEW DnaMatch.") # Verbose
                )
            if incoming_tree_data and incoming_tree_data["_operation"] == "create":
                prepared_data_for_bulk["family_tree"] = (
                    incoming_tree_data  # logger.debug(f"{log_ref}: Prep NEW FamilyTree.") # Verbose
                )
            overall_status = "new"
        else:  # Existing Person
            person_data_for_update = {
                "_operation": "update",
                "_existing_person_id": existing_person.id,
                "uuid": match_uuid.upper(),
            }
            person_update_needed = False
            new_dt = incoming_person_data.get("last_logged_in")
            old_dt = existing_person.last_logged_in
            new_naive_ts = None
            old_naive_ts = None
            if isinstance(new_dt, datetime):
                new_naive_ts = new_dt.astimezone(timezone.utc).replace(
                    tzinfo=None, microsecond=0
                )
            if isinstance(old_dt, datetime):
                old_naive_ts = (
                    old_dt.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)
                    if old_dt.tzinfo
                    else old_dt.replace(microsecond=0)
                )
            if new_naive_ts != old_naive_ts:
                person_data_for_update["last_logged_in"] = new_dt
                person_update_needed = True
            if bool(existing_person.contactable) != bool(
                incoming_person_data.get("contactable", False)
            ):
                person_data_for_update["contactable"] = bool(
                    incoming_person_data.get("contactable", False)
                )
                person_update_needed = True
            new_birth_year = incoming_person_data.get("birth_year")
            if new_birth_year is not None and existing_person.birth_year is None:
                try:
                    birth_year_int = int(new_birth_year)
                    person_data_for_update["birth_year"] = birth_year_int
                    person_update_needed = True
                except (ValueError, TypeError):
                    pass
            if bool(existing_person.in_my_tree) != bool(
                incoming_person_data.get("in_my_tree", False)
            ):
                person_data_for_update["in_my_tree"] = bool(
                    incoming_person_data.get("in_my_tree", False)
                )
                person_update_needed = True
            new_gender = incoming_person_data.get("gender")
            if (
                new_gender is not None
                and existing_person.gender is None
                and isinstance(new_gender, str)
                and new_gender.lower() in ("f", "m")
            ):
                person_data_for_update["gender"] = new_gender.lower()
                person_update_needed = True
            new_admin_id = incoming_person_data.get("administrator_profile_id")
            new_admin_user = incoming_person_data.get("administrator_username")
            if existing_person.administrator_profile_id != new_admin_id:
                person_data_for_update["administrator_profile_id"] = new_admin_id
                person_update_needed = True
            if existing_person.administrator_username != new_admin_user:
                person_data_for_update["administrator_username"] = new_admin_user
                person_update_needed = True
            new_message_link = incoming_person_data.get("message_link")
            if existing_person.message_link != new_message_link and new_message_link:
                person_data_for_update["message_link"] = new_message_link
                person_update_needed = True
            new_username = incoming_person_data.get("username")
            if existing_person.username != new_username and new_username:
                person_data_for_update["username"] = new_username
                person_update_needed = True

            if person_update_needed:
                prepared_data_for_bulk["person"] = (
                    person_data_for_update  # logger.debug(f"{log_ref}: Person data prepared for bulk update.") # Verbose
                )
            # else: logger.debug(f"{log_ref}: No changes detected for Person.") # Verbose
            if incoming_dna_data:
                prepared_data_for_bulk["dna_match"] = (
                    incoming_dna_data  # logger.debug(f"{log_ref}: Prep NEW/UPDATED DnaMatch.") # Verbose
                )
            if incoming_tree_data:
                prepared_data_for_bulk["family_tree"] = (
                    incoming_tree_data  # logger.debug(f"{log_ref}: Prep {tree_operation} FamilyTree.") # Verbose
                )
            if (
                person_update_needed
                or incoming_dna_data
                or (incoming_tree_data and tree_operation != "none")
            ):
                overall_status = "updated"
            else:
                overall_status = "skipped"

        # logger.debug(f"Final overall status determination for {log_ref_short}: {overall_status}") # Verbose
        data_to_return = prepared_data_for_bulk if overall_status != "skipped" else None
        return data_to_return, overall_status, None

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
# 3. API Data Acquisition
#################################################################################


def get_matches(
    session_manager: SessionManager,
    db_session: SqlAlchemySession,
    current_page: int = 1,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[int]]: # Return Optional List
    """
    V14.24 FIX 9: Fetches matches. Returns (None, None) on critical processing errors.
    """
    total_pages: Optional[int] = None
    if not isinstance(session_manager, SessionManager): logger.error("Invalid SessionManager"); return None, None # Return None tuple
    driver = session_manager.driver
    if not driver: logger.error("WebDriver not initialized"); return None, None # Return None tuple
    if not session_manager.my_uuid: logger.error("SessionManager my_uuid not initialized"); return None, None # Return None tuple
    if not session_manager.is_sess_valid(): logger.error("get_matches: Session invalid at start."); return None, None # Return None tuple

    my_uuid = session_manager.my_uuid
    csrf_token_cookie_name = "_dnamatches-matchlistui-x-csrf-token"
    fallback_csrf_cookie_name = "_csrf"
    specific_csrf_token = None
    found_token_name = None

    try:
        logger.debug(f"Waiting for match list element '{MATCH_ENTRY_SELECTOR}' before reading CSRF cookies...")
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, MATCH_ENTRY_SELECTOR))
            )
            logger.debug("Match list element found.")
            time.sleep(0.5)
        except TimeoutException:
            logger.warning(f"Timeout waiting for match list element '{MATCH_ENTRY_SELECTOR}'. Cookie read might fail.")
        except Exception as wait_e:
            logger.warning(f"Error waiting for match list element: {wait_e}. Proceeding cautiously.")

        logger.debug(f"Attempting to read CSRF cookies...")
        for cookie_name in [csrf_token_cookie_name, fallback_csrf_cookie_name]:
            try:
                cookie_obj = driver.get_cookie(cookie_name)
                if cookie_obj and isinstance(cookie_obj, dict) and "value" in cookie_obj and cookie_obj["value"]:
                    specific_csrf_token = unquote(cookie_obj["value"]).split("|")[0]; found_token_name = cookie_name
                    logger.debug(f"Read CSRF token from cookie '{found_token_name}'.")
                    break
            except NoSuchCookieException: continue
            except WebDriverException as cookie_e: logger.warning(f"WebDriver error getting cookie '{cookie_name}': {cookie_e}"); raise ConnectionError(f"WebDriver error getting CSRF cookie: {cookie_e}")
            except Exception as e: logger.error(f"Unexpected error getting cookie '{cookie_name}': {e}", exc_info=True); continue

        if not specific_csrf_token:
            logger.debug(f"CSRF token not found via get_cookie. Trying fallback...")
            all_cookies = get_driver_cookies(driver)
            if all_cookies:
                for cookie_name in [csrf_token_cookie_name, fallback_csrf_cookie_name]:
                    if cookie_name in all_cookies and all_cookies[cookie_name]:
                        specific_csrf_token = unquote(all_cookies[cookie_name]).split("|")[0]; found_token_name = cookie_name
                        logger.debug(f"Read CSRF token via fallback ('{found_token_name}').")
                        break
            else: logger.warning("Fallback get_driver_cookies failed.")

        if not specific_csrf_token:
            logger.error("Failed to obtain a valid CSRF token from cookies. Cannot call Match List API.")
            return None, None # Return None tuple

        match_list_url = urljoin(config_instance.BASE_URL, f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={current_page}")
        logger.debug(f"Fetching match list page {current_page} using requests...")
        chrome_version = "125"; user_agent = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version}.0.0.0 Safari/537.36"
        sec_ch_ua = f'"Google Chrome";v="{chrome_version}", "Not-A.Brand";v="8", "Chromium";v="{chrome_version}"'
        match_list_headers = {
            "User-Agent": user_agent, "accept": "application/json",
            "Referer": urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
            "x-csrf-token": specific_csrf_token, "sec-ch-ua": sec_ch_ua, "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"', "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Origin": config_instance.BASE_URL.rstrip("/"), "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors", "sec-fetch-site": "same-origin", "dnt": "1", "Content-Type": "application/json",
        }
        api_response = _api_req(url=match_list_url, driver=driver, session_manager=session_manager, method="GET",
                                headers=match_list_headers, use_csrf_token=False, api_description="Match List API", allow_redirects=False)

        if api_response is None: logger.warning(f"No response/error from match list API page {current_page}."); return None, None # Return None tuple
        if not isinstance(api_response, dict):
            logger.error(f"Match List API did not return dict (type {type(api_response)}). Page {current_page}.")
            if isinstance(api_response, (str, bytes)): logger.debug(f"Content preview: {api_response[:500]}")
            return None, None # Return None tuple

        total_pages_raw = api_response.get("totalPages")
        total_pages = None
        if total_pages_raw is not None:
            try: total_pages = int(total_pages_raw)
            except (ValueError, TypeError): logger.warning(f"Could not parse totalPages '{total_pages_raw}'.")
        else: logger.warning("totalPages missing from Match List API response.")

        match_data_list = api_response.get("matchList", [])
        if not match_data_list: logger.info(f"No matches found in 'matchList' page {current_page}."); return [], total_pages # Return empty list ok

        logger.debug(f"Got {len(match_data_list)} raw matches from API page {current_page}.")

        valid_matches_for_processing: List[Dict[str, Any]] = []
        skipped_sampleid_count = 0
        for m in match_data_list:
            if isinstance(m, dict) and m.get("sampleId"): valid_matches_for_processing.append(m)
            else: skipped_sampleid_count += 1; logger.warning(f"Skipping raw match missing 'sampleId' page {current_page}.")
        if skipped_sampleid_count > 0: logger.warning(f"Skipped {skipped_sampleid_count} matches page {current_page} (missing 'sampleId').")
        if not valid_matches_for_processing: logger.warning(f"No matches with valid 'sampleId' page {current_page}."); return [], total_pages # Return empty list ok

        sample_ids_on_page = [match["sampleId"].upper() for match in valid_matches_for_processing]
        in_tree_ids: Set[str] = set()
        cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids_on_page))}"
        cached_in_tree = global_cache.get(cache_key_tree, default=None)
        if cached_in_tree is not None and isinstance(cached_in_tree, set):
            in_tree_ids = cached_in_tree; logger.debug(f"Loaded {len(in_tree_ids)} in-tree IDs from cache.")
        else:
            if not session_manager.is_sess_valid(): logger.error(f"In-Tree Check: Session invalid page {current_page}.")
            else:
                in_tree_url = urljoin(config_instance.BASE_URL, f"discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid.upper()}")
                logger.debug(f"Fetching in-tree status page {current_page}...")
                in_tree_headers = {"X-CSRF-Token": specific_csrf_token, "Content-Type": "application/json", "User-Agent": user_agent, "Referer": urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"), "Origin": config_instance.BASE_URL.rstrip("/"), "Accept": "application/json"}
                response_in_tree = _api_req(url=in_tree_url, driver=driver, session_manager=session_manager, method="POST", json_data={"sampleIds": sample_ids_on_page}, headers=in_tree_headers, use_csrf_token=False, api_description="In-Tree Status Check")
                if isinstance(response_in_tree, list):
                    in_tree_ids = {item.upper() for item in response_in_tree if isinstance(item, str)}
                    global_cache.set(cache_key_tree, in_tree_ids, expire=config_instance.CACHE_TIMEOUT)
                    logger.debug(f"Fetched/cached {len(in_tree_ids)} in-tree IDs page {current_page}.")
                else: logger.warning(f"In-Tree Status Check API failed/unexpected format page {current_page}. Resp: {response_in_tree}")

        refined_matches: List[Dict[str, Any]] = []
        for match in valid_matches_for_processing: # IndexErrors likely occurred within this loop
            # Wrap the refinement logic in a try/except to catch errors per match
            try:
                profile = match.get("matchProfile", {})
                relationship = match.get("relationship", {})
                # Check if essential keys exist before accessing directly
                if "sampleId" not in match:
                     logger.warning(f"Skipping match refinement due to missing sampleId: {match}")
                     continue # Skip this match

                sample_id_upper = match["sampleId"].upper()
                profile_user_id = profile.get("userId")
                profile_user_id_upper = (str(profile_user_id).upper() if profile_user_id else None)
                raw_display_name = profile.get("displayName")
                match_username = format_name(raw_display_name)
                first_name = (match_username.split()[0] if match_username != "Valued Relative" else None)
                admin_profile_id_hint = match.get("adminId")
                admin_username_hint = match.get("adminName")
                compare_link = urljoin(config_instance.BASE_URL, f"discoveryui-matches/compare/{my_uuid.upper()}/with/{sample_id_upper}")

                refined_match_data = {
                    "username": match_username, "first_name": first_name, "initials": profile.get("displayInitials", "??").upper(),
                    "gender": match.get("gender"), "profile_id": profile_user_id_upper, "uuid": sample_id_upper,
                    "administrator_profile_id_hint": admin_profile_id_hint, "administrator_username_hint": admin_username_hint,
                    "photoUrl": profile.get("photoUrl", ""), "cM_DNA": int(relationship.get("sharedCentimorgans", 0)),
                    "numSharedSegments": int(relationship.get("numSharedSegments", 0)), "compare_link": compare_link,
                    "message_link": None, "in_my_tree": sample_id_upper in in_tree_ids, "createdDate": match.get("createdDate"),
                }
                refined_matches.append(refined_match_data)
            except IndexError as ie: # Catch specific IndexError during refinement
                 logger.error(f"IndexError refining match data on page {current_page}: {ie}. Match data: {match}", exc_info=True)
                 # Optionally: continue to next match instead of failing the whole page?
                 # For now, let the outer exception handler catch this.
                 raise # Re-raise to be caught by the main try/except
            except Exception as refine_e: # Catch other potential errors during refinement
                 logger.error(f"Error refining match data on page {current_page}: {refine_e}. Match data: {match}", exc_info=True)
                 raise # Re-raise to be caught by the main try/except

        logger.debug(f"Processed page {current_page}: Raw={len(match_data_list)}, Refined={len(refined_matches)}")
        return refined_matches, total_pages

    except ConnectionError as e:
        logger.error(f"Network/Connection error page {current_page}: {e}", exc_info=False)
        raise e # Re-raise for coord to handle recovery
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error page {current_page}: {e}", exc_info=True)
        return None, None # Indicate failure
    except NoSuchCookieException as e:
        logger.critical(f"Critical error: Could not find required CSRF cookie. {e}", exc_info=True)
        return None, None # Indicate failure
    except WebDriverException as e:
        logger.error(f"WebDriver error during get_matches page {current_page}: {e}", exc_info=True)
        # Check if session died
        if session_manager and not session_manager.is_sess_valid():
             logger.error("Session became invalid during get_matches WebDriverException.")
             # Don't raise ConnectionError here, let the None return signal failure
        return None, None # Indicate failure
    except Exception as e:
        # *** MODIFIED HERE ***
        logger.critical(f"Critical error processing match data for page {current_page}: {e}", exc_info=True)
        return None, None # Return None tuple to signal error to coord
        # *** END MODIFICATION ***
# end get_matches


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_combined_details(
    session_manager: SessionManager, match_uuid: str
) -> Optional[Dict[str, Any]]:
    """V14.21"""
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
    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching /details for UUID {match_uuid}: {conn_err}",
            exc_info=False,
        )
        raise
    except Exception as e:
        logger.error(
            f"Error fetching /details for UUID {match_uuid}: {e}", exc_info=True
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise

    tester_profile_id_for_api = combined_data.get("tester_profile_id")
    my_profile_id_header = session_manager.my_profile_id
    if not tester_profile_id_for_api:
        logger.debug(
            f"Skipping /profiles/details fetch for {match_uuid}: tester_profile_id not found in /details response."
        )
        combined_data["last_logged_in_dt"] = None
        combined_data["contactable"] = False
    elif not my_profile_id_header:
        logger.warning(
            f"Skipping /profiles/details fetch for {match_uuid}: Own profile ID missing for header."
        )
        combined_data["last_logged_in_dt"] = None
        combined_data["contactable"] = False
    elif not session_manager.is_sess_valid():
        logger.error(
            f"Combined details fetch: WebDriver session invalid before fetching profile for {tester_profile_id_for_api}."
        )
        combined_data["last_logged_in_dt"] = None
        combined_data["contactable"] = False
        raise ConnectionError(
            f"WebDriver session invalid before profile fetch (Profile: {tester_profile_id_for_api})"
        )
    else:
        profile_url = urljoin(
            config_instance.BASE_URL,
            f"/app-api/express/v1/profiles/details?userId={tester_profile_id_for_api.upper()}",
        )
        profile_headers = {
            "accept": "application/json",
            "ancestry-clientpath": "express-fe",
            "ancestry-userid": my_profile_id_header.upper(),
            "cache-control": "no-cache",
            "pragma": "no-cache",
        }
        try:
            profile_response = _api_req(
                url=profile_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                headers=profile_headers,
                use_csrf_token=False,
                api_description="Profile Details API (Batch)",
                referer_url=details_referer,
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
                        else:
                            dt_naive = datetime.fromisoformat(last_login_str)
                            last_login_dt = (
                                dt_naive.replace(tzinfo=timezone.utc)
                                if dt_naive.tzinfo is None
                                else dt_naive
                            )
                        combined_data["last_logged_in_dt"] = last_login_dt.astimezone(
                            timezone.utc
                        )
                    except (ValueError, TypeError) as date_parse_err:
                        logger.warning(
                            f"Could not parse LastLoginDate '{last_login_str}' for {tester_profile_id_for_api}: {date_parse_err}"
                        )
                        combined_data["last_logged_in_dt"] = None
                else:
                    combined_data["last_logged_in_dt"] = None
                combined_data["contactable"] = bool(contactable_val)
            else:
                logger.warning(
                    f"Failed to get valid /profiles/details response for {tester_profile_id_for_api}. Type: {type(profile_response)}"
                )
                combined_data["last_logged_in_dt"] = None
                combined_data["contactable"] = False
        except NameError as ne:
            logger.critical(
                f"NameError in _fetch_combined_details (likely timezone): {ne}",
                exc_info=True,
            )
            combined_data["last_logged_in_dt"] = None
            combined_data["contactable"] = False
        except ConnectionError as conn_err:
            logger.error(
                f"ConnectionError fetching /profiles/details for {tester_profile_id_for_api}: {conn_err}",
                exc_info=False,
            )
            combined_data["last_logged_in_dt"] = None
            combined_data["contactable"] = False
            raise
        except Exception as e:
            logger.error(
                f"Error fetching /profiles/details for {tester_profile_id_for_api}: {e}",
                exc_info=True,
            )
            combined_data["last_logged_in_dt"] = None
            combined_data["contactable"] = False
            if isinstance(e, requests.exceptions.RequestException):
                raise
    return combined_data if match_uuid else None
# end _fetch_combined_details


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_batch_badge_details(
    session_manager: SessionManager, match_uuid: str
) -> Optional[Dict[str, Any]]:
    """V14.21"""
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
    try:
        badge_response = _api_req(
            url=badge_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description="Badge Details API (Batch)",
            referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
        )
        if badge_response and isinstance(badge_response, dict):
            person_badged = badge_response.get("personBadged", {})
            raw_firstname = person_badged.get("firstName")
            their_firstname_formatted = (
                format_name(raw_firstname).split()[0] if raw_firstname else "Unknown"
            )
            return {
                "their_cfpid": person_badged.get("personId"),
                "their_firstname": their_firstname_formatted,
                "their_lastname": person_badged.get("lastName", "Unknown"),
                "their_birth_year": person_badged.get("birthYear"),
            }
        else:
            logger.warning(f"Invalid badge details response for UUID {match_uuid}.")
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
    """V14.21"""
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
    dynamic_referer = urljoin(
        config_instance.BASE_URL,
        f"family-tree/person/tree/{tree_id}/person/{cfpid}/facts",
    )
    ladder_data = {}
    ladder_headers = {
        "Accept": "*/*",
        "Origin": config_instance.BASE_URL.rstrip("/"),
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
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
        if isinstance(api_result, requests.Response):
            logger.warning(
                f"Get Ladder API call failed for cfpid {cfpid} (handled by _api_req). Returning None."
            )
            return None
        elif api_result is None:
            logger.warning(
                f"Get Ladder API call returned None for cfpid {cfpid}. Returning None."
            )
            return None
        elif not isinstance(api_result, str):
            logger.warning(
                f"_api_req returned unexpected type '{type(api_result).__name__}' for Get Ladder API for cfpid {cfpid}. Returning None."
            )
            return None
        response_text = api_result
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
                    return None
                ladder_json = json.loads(json_string)
                if isinstance(ladder_json, dict) and "html" in ladder_json:
                    html_content = ladder_json["html"]
                    if html_content:
                        soup = BeautifulSoup(html_content, "html.parser")
                        actual_relationship_text = None
                        relationship_path_text = None
                        rel_elem = soup.select_one(
                            "ul.textCenter > li:first-child > i > b"
                        )
                        if rel_elem:
                            raw_relationship = rel_elem.get_text(strip=True)
                            actual_relationship_text = ordinal_case(
                                raw_relationship.title()
                            )
                        else:
                            logger.warning(
                                f"Could not extract actual_relationship for cfpid: {cfpid}"
                            )
                        path_items = soup.select(
                            'ul.textCenter > li:not([class*="iconArrowDown"])'
                        )
                        path_list = []
                        num_items = len(path_items)
                        for i, item in enumerate(path_items):
                            name_text, desc_text, raw_name_extracted = "", "", ""
                            name_link = item.find("a")
                            name_bold = item.find("b") if not name_link else None
                            if isinstance(name_link, Tag):
                                nested_b = (
                                    name_link.find("b")
                                    if isinstance(name_link, Tag)
                                    else None
                                )
                                raw_name_extracted = (
                                    nested_b.get_text(strip=True)
                                    if isinstance(nested_b, Tag)
                                    else name_link.get_text(strip=True)
                                )
                            elif name_bold:
                                raw_name_extracted = name_bold.get_text(strip=True)
                            else:
                                parts = item.get_text(separator="\n", strip=True).split(
                                    "\n"
                                )
                                raw_name_extracted = parts[0] if parts else ""
                            if raw_name_extracted:
                                name_text = format_name(
                                    " ".join(
                                        raw_name_extracted.replace('"', "'").split()
                                    )
                                )
                            if i > 0:
                                desc_element = item.find("i")
                                if desc_element:
                                    raw_desc_full = desc_element.get_text(strip=True)
                                    cleaned_desc_full = raw_desc_full.replace('"', "'")
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
                                        match_rel = re.match(
                                            r"^(.*?)\s+of\s+(.*)$",
                                            cleaned_desc_full,
                                            re.IGNORECASE,
                                        )
                                        if match_rel:
                                            desc_text = f"{match_rel.group(1).strip().capitalize()} of {format_name(match_rel.group(2).strip())}"
                                        else:
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
                        ladder_data["actual_relationship"] = actual_relationship_text
                        ladder_data["relationship_path"] = relationship_path_text
                        return ladder_data
                    else:
                        logger.warning(
                            f"Empty HTML in getladder response for cfpid {cfpid}."
                        )
                else:
                    logger.warning(
                        f"Missing 'html' key in getladder JSON for cfpid {cfpid}."
                    )
            except json.JSONDecodeError as inner_json_err:
                logger.error(
                    f"Failed to decode JSONP content for cfpid {cfpid}: {inner_json_err}"
                )
                logger.debug(f"JSON string causing decode error: '{json_string[:200]}'")
                return None
        else:
            logger.error(f"Could not parse JSONP format for cfpid {cfpid}.")
        return None
    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching ladder for CFPID {cfpid}: {conn_err}",
            exc_info=False,
        )
        raise
    except Exception as e:
        logger.error(
            f"Error fetching/parsing ladder for CFPID {cfpid}: {e}", exc_info=True
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None
# end _fetch_batch_ladder


def _fetch_batch_relationship_prob(
    session_manager: SessionManager, match_uuid: str, max_labels_param: int
) -> Optional[str]:
    """
    V14.21 (Refactored): Uses a shared cloudscraper instance from SessionManager.
    """
    driver = session_manager.driver
    if not session_manager.my_uuid or not match_uuid:
        logger.warning(
            "Missing my_uuid or match_uuid for relationship probability fetch."
        )
        return "N/A (Error - Missing IDs)"

    # --- Refactoring Step 1: Get shared scraper instance ---
    # Assume SessionManager initializes and holds the scraper instance
    # e.g., self.scraper = cloudscraper.create_scraper(...) in SessionManager.__init__
    scraper = session_manager.scraper
    if not scraper:
        logger.error(
            "SessionManager does not have a valid scraper instance. Cannot fetch relationship probability."
        )
        # Raise an exception or return an error string, depending on desired handling
        # Raising ConnectionError aligns with other checks indicating session issues
        raise ConnectionError("SessionManager scraper not initialized.")
    # --- End Refactoring Step 1 ---

    if not driver or not session_manager.is_sess_valid():
        logger.error(
            f"Relationship prob fetch: Driver not available or session invalid for UUID {match_uuid}."
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
    chrome_version = "125"  # Consider making this dynamic or configurable if needed
    user_agent = (
        f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{chrome_version}.0.0.0 Safari/537.36"
    )
    sec_ch_ua = (
        f'"Google Chrome";v="{chrome_version}", '
        f'"Not-A.Brand";v="8", "Chromium";v="{chrome_version}"'
    )
    rel_headers = {
        "Accept": "application/json",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
        "Origin": config_instance.BASE_URL.rstrip("/"),
        "Pragma": "no-cache",
        "Priority": "u=1, i",
        "Referer": referer_url,
        "sec-ch-ua": sec_ch_ua,
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "User-Agent": user_agent,
        "X-Requested-With": "XMLHttpRequest",
        # CSRF token handled below
    }

    csrf_token_val = None
    csrf_cookie_names = ("_dnamatches-matchlistui-x-csrf-token", "_csrf")
    try:
        driver_cookies_list = driver.get_cookies()
        if driver_cookies_list:
            driver_cookies_dict = {
                c["name"]: c["value"]
                for c in driver_cookies_list
                if "name" in c and "value" in c
            }
            # --- Refactoring Step 2: Update shared scraper's cookies ---
            # It's good practice to keep the scraper's cookies in sync
            # This replaces the previous logic of just fetching cookies for the local scraper
            logger.debug(
                f"Updating shared scraper cookie jar with {len(driver_cookies_list)} cookies from WebDriver..."
            )
            # Clear potentially stale cookies before adding fresh ones
            # Note: Depending on cloudscraper/requests version, direct access might differ.
            # This assumes a standard RequestsCookieJar interface.
            if hasattr(scraper, "cookies") and isinstance(
                scraper.cookies, RequestsCookieJar
            ):
                scraper.cookies.clear()  # Clear existing cookies in the shared scraper

            for cookie in driver_cookies_list:
                if "name" in cookie and "value" in cookie:
                    # Set cookies in the shared scraper instance
                    scraper.cookies.set(
                        cookie["name"],
                        cookie["value"],
                        domain=cookie.get("domain"),
                        path=cookie.get("path", "/"),
                        secure=cookie.get("secure", False),
                        # RequestsCookieJar doesn't directly handle httpOnly
                        # expires=cookie.get("expiry") # Handle expiry if needed
                    )
            # --- End Refactoring Step 2 ---

            # Now, find the CSRF token from the dictionary we already built
            for name in csrf_cookie_names:
                if name in driver_cookies_dict and driver_cookies_dict[name]:
                    csrf_token_val = unquote(driver_cookies_dict[name]).split("|")[0]
                    rel_headers["X-CSRF-Token"] = csrf_token_val
                    logger.debug(
                        f"Using fresh CSRF token '{name}' from driver cookies for rel prob."
                    )
                    break
        else:
            logger.warning("driver.get_cookies() returned None or empty list.")

    except WebDriverException as csrf_wd_e:
        logger.warning(
            f"WebDriverException getting/setting cookies for CSRF token: {csrf_wd_e}"
        )
        # Propagate as ConnectionError as it prevents the network call
        raise ConnectionError(
            f"WebDriver error getting/setting cookies for CSRF: {csrf_wd_e}"
        )
    except Exception as csrf_e:
        logger.warning(f"Could not get/set CSRF token from driver cookies: {csrf_e}")

    # Fallback CSRF token logic remains the same
    if "X-CSRF-Token" not in rel_headers:
        if session_manager.csrf_token:
            logger.warning(
                "Using potentially stale CSRF token from SessionManager as fallback."
            )
            rel_headers["X-CSRF-Token"] = session_manager.csrf_token
        else:
            logger.error(
                "Failed to add CSRF token to headers for cloudscraper request (fresh and fallback failed)."
            )
            return "N/A (Error - Missing CSRF)"

    api_description = "Match Probability API (Cloudscraper)"
    try:
        # --- Refactoring Step 3: Use the shared scraper ---
        # Remove: scraper = cloudscraper.create_scraper(delay=5)
        # Remove: Cookie setting logic here (moved above)

        logger.debug(
            f"Making {api_description} POST request to {rel_url} using shared scraper"
        )
        # Use the scraper instance from the session_manager
        response_rel = scraper.post(
            rel_url,
            headers=rel_headers,
            json={},  # Ensure payload is correct if needed
            allow_redirects=False,
            timeout=selenium_config.API_TIMEOUT,
        )
        # --- End Refactoring Step 3 ---

        logger.debug(
            f"<-- {api_description} Response Status: {response_rel.status_code} {response_rel.reason}"
        )

        # Response handling logic remains the same
        if not response_rel.ok:
            status_code = response_rel.status_code
            logger.warning(
                f"{api_description} failed for {sample_id_upper}. Status: {status_code}, Reason: {response_rel.reason}"
            )
            try:
                logger.debug(f"  Response Body: {response_rel.text[:500]}")
            except Exception:
                pass
            if 300 <= status_code < 400:
                logger.warning(
                    f"  -> Redirect detected (Loc: {response_rel.headers.get('Location')}). Check headers/cookies."
                )
            # Only raise for critical errors that retry won't fix or indicate session issues
            if status_code in [
                401,
                403,
                500,
                502,
                503,
                504,
            ]:  # 403 might be CF challenge or auth
                # Let retry_api handle RequestExceptions, raise others like auth errors
                if status_code in [401]:  # Unauthorized usually means session issue
                    raise RequestException(
                        f"Auth Error ({status_code})", response=response_rel
                    )
                # CloudflareException might be raised by scraper.post itself
                # raise_for_status will convert others to HTTPError
                try:
                    response_rel.raise_for_status()
                except HTTPError as http_err:
                    # Check if it's a cloudflare specific error based on content maybe?
                    # For now, re-raise to let retry_api handle potentially transient ones
                    raise http_err

            return "N/A (API Error/Redirect)"  # For non-critical errors after logging

        try:
            if not response_rel.content:
                logger.warning(
                    f"{api_description}: OK ({response_rel.status_code}), but response body is EMPTY. Returning None."
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
            final_labels = labels[:max_labels_param]
            relationship_str = " or ".join(map(str, final_labels))
            return f"{relationship_str} [{top_prob:.1f}%]"
        except json.JSONDecodeError as json_err:
            logger.error(
                f"{api_description}: OK ({response_rel.status_code}), but JSON decode FAILED: {json_err}"
            )
            logger.debug(
                f"Response text causing decode error: {response_rel.text[:500]}"
            )
            # Let retry_api handle this if it's transient
            raise RequestException("JSONDecodeError") from json_err
        except Exception as e:
            logger.error(
                f"{api_description}: Error processing successful response for {sample_id_upper}: {e}",
                exc_info=True,
            )
            # Re-raise general exceptions to allow retry potentially
            raise RequestException("Response Processing Error") from e

    # Catch specific exceptions that retry_api handles, raise others
    except cloudscraper.exceptions.CloudflareException as cf_e:
        logger.error(
            f"{api_description}: Cloudflare challenge failed for {sample_id_upper}: {cf_e}"
        )
        raise cf_e  # Let retry_api handle this
    except requests.exceptions.RequestException as req_e:
        # This includes ConnectionError, HTTPError, Timeout etc.
        logger.error(
            f"{api_description}: RequestException for {sample_id_upper}: {req_e}"
        )
        raise req_e  # Let retry_api handle this
    except Exception as e:
        # Catch-all for unexpected errors during the request phase
        logger.error(
            f"{api_description}: Unexpected error for {sample_id_upper}: {type(e).__name__} - {e}",
            exc_info=True,
        )
        # Raise a generic RequestException so retry_api might catch it,
        # or return an error string if retrying is unlikely to help.
        # Given the previous errors, let's make it retryable.
        raise RequestException(f"Unexpected Fetch Error: {type(e).__name__}") from e
# end _fetch_batch_relationship_prob


#################################################################################
# 5. 'Create or Update' Database Operations (Uses imported functions)
#################################################################################


class PersonProcessingError(Exception):
    """Custom exception for errors during Person creation/update."""

    pass
# end of PersonProcessingError class


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
    logger.info("------------------------------------")
# end of _log_coord_summary


def _adjust_delay(session_manager, page):
    """Adjusts the dynamic rate limiter delay after processing a page."""
    if session_manager.dynamic_rate_limiter.is_throttled():
        logger.debug(f"Rate limiter was throttled during processing of page {page}.")
    else:
        session_manager.dynamic_rate_limiter.decrease_delay()
        if (
            session_manager.dynamic_rate_limiter.current_delay
            > config_instance.INITIAL_DELAY
        ):
            logger.debug(
                f"Decreased delay for next page to {session_manager.dynamic_rate_limiter.current_delay:.2f} seconds."
            )
# End of _adjust_delay


def nav_to_list(session_manager) -> bool:
    """Navigates directly to the user's specific DNA matches list page using their UUID."""
    if not session_manager.is_sess_valid() or not session_manager.my_uuid:
        logger.error(
            "Session invalid or user UUID missing. Cannot navigate to matches list."
        )
        return False
    matches_url_with_uuid = urljoin(
        config_instance.BASE_URL, f"discoveryui-matches/list/{session_manager.my_uuid}"
    )
    success = nav_to_page(
        session_manager.driver,
        matches_url_with_uuid,
        selector=MATCH_ENTRY_SELECTOR,
        session_manager=session_manager,
    )
    if success:
        try:
            current_url = session_manager.driver.current_url
            if not current_url.startswith(matches_url_with_uuid):
                logger.warning(
                    f"Navigation reported success, but final URL is unexpected: {current_url}"
                )
        except Exception as e:
            logger.warning(f"Could not verify final URL after nav_to_list: {e}")
    else:
        logger.error("Failed to navigate to specific matches list page.")
    return success
# end nav_to_list

# end of action6_gather.py
