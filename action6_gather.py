#!/usr/bin/env python3

# action6_gather.py

# Standard library imports (alphabetical)
import json
import logging
import math
import random
import re
import time
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, unquote

# Third-party imports (alphabetical by package)
import requests
import cloudscraper
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup, Tag
from requests.exceptions import RequestException, HTTPError
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
    NoSuchCookieException,
)
from selenium.webdriver.common.by import By
from requests.cookies import RequestsCookieJar
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sqlalchemy import func, Column
from sqlalchemy.orm import joinedload, Session as SqlAlchemySession, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Local application imports (alphabetical by module)
from cache import cache_result
from config import config_instance, selenium_config

from database import (
    DnaMatch,
    FamilyTree,
    Person,
    db_transn,
    create_or_update_person,
    create_dna_match,
    create_family_tree,
    get_person_by_uuid,
    get_person_by_profile_id,
    find_existing_person,
)
from my_selectors import *
from utils import (
    DynamicRateLimiter,
    SessionManager,
    _api_req,
    ordinal_case,
    make_newrelic,
    make_traceparent,
    make_tracestate,
    make_ube,
    nav_to_page,
    retry_api,
    retry,
    time_wait,
    get_driver_cookies,
    format_name,
    urljoin
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
    V14.15 REVISED: Gathers DNA matches, processing page-by-page.
    - Progress bar initialized inside logging_redirect_tqdm.
    - Uses stderr for progress bar.
    - Uses improved bar format with description and postfix.
    - Updates postfix with cumulative stats after each page batch.
    - Relies on _do_batch for per-match updates.
    - Removed bulk updates on page-level errors in this loop.
    """
    driver = session_manager.driver
    if not driver or not session_manager.session_active:
        logger.error("WebDriver not initialized or session not active. Exiting coord.")
        return False
    total_new, total_updated, total_skipped, total_errors = 0, 0, 0, 0
    total_pages_processed = 0
    progress_bar = None # Define here for finally block scope
    final_success = True
    my_uuid = session_manager.my_uuid
    if not my_uuid:
        logger.error("Failed to retrieve my_uuid from session_manager. Exiting coord.")
        return False
    target_matches_url_base = urljoin(config_instance.BASE_URL, f"discoveryui-matches/list/{my_uuid}")
    total_pages: Optional[int] = None
    last_page: Optional[int] = None
    total_pages_to_process_in_run = 0
    matches_on_page : List[Dict[str, Any]] = []
    total_matches_estimate = 0

    try:
        if not isinstance(start, int) or start <= 0:
            logger.warning(f"Invalid start parameter '{start}'. Using default start page 1.")
            start_page = 1
        else: start_page = start
    except Exception:
         logger.error(f"Error processing start parameter '{start}'. Using default start page 1.")
         start_page = 1

    try:
        logger.debug("11. Ensure we are on the DNA matches page...")
        try: # Navigation
            current_url = driver.current_url
            if not current_url.startswith(target_matches_url_base):
                logger.debug("Navigating to DNA matches page.")
                if not nav_to_list(session_manager):
                    logger.error("Failed to navigate to DNA match list page. Exiting coord.")
                    return False
                else: logger.debug("Successfully navigated to DNA matches page.\n")
            else: logger.debug(f"Already on correct DNA matches page: {current_url}.\n")
        except WebDriverException as nav_e:
            logger.error(f"WebDriver error checking/navigating: {nav_e}", exc_info=True); return False

        logger.debug(f"Fetching initial page {start_page} to determine total pages...")
        db_session_for_page = session_manager.get_db_conn()
        fetched_total_pages = None
        if not db_session_for_page: logger.error(f"Could not get DB session for initial page fetch. Aborting."); return False
        try: matches_on_page, fetched_total_pages = get_matches(session_manager, db_session_for_page, start_page)
        except Exception as get_match_err: logger.error(f"Error during initial get_matches call on page {start_page}: {get_match_err}", exc_info=True); session_manager.return_session(db_session_for_page); return False
        finally:
            if db_session_for_page: session_manager.return_session(db_session_for_page)

        if fetched_total_pages is None: logger.error("Failed to retrieve total_pages on initial fetch. Aborting."); return False
        total_pages = fetched_total_pages
        logger.info(f"Total pages found: {total_pages}\n")

        max_pages_config = config_instance.MAX_PAGES
        pages_to_process_config = (min(max_pages_config, total_pages) if max_pages_config != 0 else total_pages)
        last_page = min(start_page + pages_to_process_config - 1, total_pages)
        total_pages_to_process_in_run = max(0, last_page - start_page + 1)

        if total_pages_to_process_in_run <= 0: logger.info("No pages to process based on start/end page calculation."); return True
        total_matches_estimate = total_pages_to_process_in_run * MATCHES_PER_PAGE
        logger.info(f"Processing {total_pages_to_process_in_run} pages (approx. {total_matches_estimate} matches) from {start_page} to {last_page}.\n")


        with logging_redirect_tqdm():
            progress_bar = tqdm(
                total=total_matches_estimate,
                desc="Gathering Matches", # Added description
                unit=" match",
                ncols=100, # Adjusted width
                # Improved bar format
                bar_format="{percentage:3.0f}%|{bar}|",
                file=sys.stderr, # Use stderr
                leave=True # Keep bar after finish
            )

            logger.debug("Processing matches page by page inside redirect context...")
            current_page_num = start_page
            while True:
                if current_page_num > last_page: logger.debug(f"Current page {current_page_num} exceeds processing limit {last_page}. Stopping."); break

                if not (current_page_num == start_page and matches_on_page):
                    db_session_for_page = session_manager.get_db_conn()
                    if not db_session_for_page:
                        logger.error(f"Could not get DB session for page {current_page_num}. Skipping page.")
                        # Error counter reflects approximate missed matches for summary
                        total_errors += MATCHES_PER_PAGE
                        # --- REMOVED bulk progress bar update on DB error ---
                        if total_errors > (3 * MATCHES_PER_PAGE): logger.critical("Aborting run due to persistent DB connection failures."); final_success = False; break
                        time.sleep(2); current_page_num += 1
                        continue # Skip to next page iteration
                    try: matches_on_page, _ = get_matches(session_manager, db_session_for_page, current_page_num)
                    finally: session_manager.return_session(db_session_for_page)

                if matches_on_page is None or not matches_on_page:
                    logger.warning(f"No matches/error fetching for page {current_page_num}. Skipping page processing.")
                    # --- REMOVED bulk progress bar update on empty page ---
                    time.sleep(2); current_page_num += 1; matches_on_page = []; continue

                page_new, page_updated, page_skipped, page_errors = _do_batch(
                    session_manager=session_manager,
                    matches_on_page=matches_on_page,
                    current_page=current_page_num,
                    last_page_in_run=(last_page if last_page is not None else "?"),
                    max_labels_to_show=2,
                    progress_bar=progress_bar # Pass the bar object
                )

                # Update cumulative totals
                total_new += page_new; total_updated += page_updated; total_skipped += page_skipped; total_errors += page_errors
                total_pages_processed += 1

                # --- Update postfix with cumulative totals ---
                if progress_bar:
                    progress_bar.set_postfix(
                        New=total_new, Upd=total_updated, Skip=total_skipped, Err=total_errors, refresh=True # Use cumulative totals, refresh needed here
                    )

                _adjust_delay(session_manager, current_page_num)
                current_page_num += 1
                matches_on_page = [] # Clear for next iteration

    except KeyboardInterrupt: logger.warning("Keyboard interrupt..."); final_success = False; _log_coord_summary(total_pages_processed, total_new, total_updated, total_skipped, total_errors); raise
    except Exception as e: logger.error(f"Critical error during coord execution: {e}", exc_info=True); final_success = False
    finally:
        logger.debug("Entering finally block in coord...")
        if progress_bar:
            progress_bar.close()
            print(file=sys.stderr) # Use stderr for newline
        _log_coord_summary(total_pages_processed, total_new, total_updated, total_skipped, total_errors)
        logger.debug("Exiting finally block in coord.")

    return final_success
# end of coord

def _do_batch(
    session_manager: SessionManager,
    matches_on_page: List[Dict[str, Any]],
    current_page: int,
    last_page_in_run: Union[int, str],
    max_labels_to_show: int,
    progress_bar: Optional[tqdm] = None, # Use tqdm hint from auto import
) -> Tuple[int, int, int, int]:
    """
    V14.15 (Revised): Processes a batch of matches using pre-fetching.
    - Updates progress bar per match in a finally block for robustness.
    - Removed tiny sleep after update.
    - Progress bar update handled within finally block of each match loop iteration.
    - Removed bulk update on initial DB error.
    - Kept update in outer finally for remaining items if critical error occurs mid-loop.
    """
    page_new, page_updated, page_skipped, page_errors = 0, 0, 0, 0
    num_matches = len(matches_on_page)
    my_uuid = session_manager.my_uuid
    my_tree_id = session_manager.my_tree_id
    if not my_uuid:
        logger.error(f"_do_batch Page {current_page}: Missing my_uuid. Cannot process.")
        # --- REMOVED bulk progress bar update on UUID error ---
        return 0, 0, 0, num_matches # Return errors equal to expected matches

    # --- Pre-fetch Details Concurrently ---
    logger.debug(f"--- Starting Batch Pre-fetch for Page {current_page} ({num_matches} matches) ---")
    uuids_on_page = [m["uuid"] for m in matches_on_page if m.get("uuid")]
    uuids_for_tree_badge = [m["uuid"] for m in matches_on_page if m.get("uuid") and m.get("in_my_tree")]
    uuids_for_combined_details = uuids_on_page
    uuids_for_relationships = uuids_on_page

    batch_combined_details: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_badge_data: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_ladder_data: Dict[str, Optional[Dict[str, Any]]] = {}
    batch_relationship_prob_data: Dict[str, Optional[str]] = {}

    futures = {}
    fetch_start_time = time.time()

    # Use ThreadPoolExecutor for concurrent fetching
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks for combined details
        for uuid_val in uuids_for_combined_details:
            delay = session_manager.dynamic_rate_limiter.wait() # Apply rate limiting before submitting
            future = executor.submit(_fetch_combined_details, session_manager, uuid_val)
            futures[future] = ("combined_details", uuid_val)

        # Submit tasks for badge details (only if in tree)
        for uuid_val in uuids_for_tree_badge:
            delay = session_manager.dynamic_rate_limiter.wait()
            future = executor.submit(_fetch_batch_badge_details, session_manager, uuid_val)
            futures[future] = ("badge_details", uuid_val)

        # Submit tasks for relationship probability
        for uuid_val in uuids_for_relationships:
            delay = session_manager.dynamic_rate_limiter.wait()
            future = executor.submit(_fetch_batch_relationship_prob, session_manager, uuid_val, max_labels_to_show)
            futures[future] = ("relationship_prob", uuid_val)

        # Process results as they complete (for combined, badge, relationship)
        temp_badge_results = {} # Store badge results temporarily for ladder fetch
        for future in as_completed(futures):
             task_type, identifier = futures[future]
             try:
                  result = future.result() # Get result (might be None on failure)
                  if result is not None:
                       if task_type == "combined_details": batch_combined_details[identifier] = result
                       elif task_type == "badge_details": temp_badge_results[identifier] = result
                       elif task_type == "relationship_prob": batch_relationship_prob_data[identifier] = result
                  else:
                       logger.debug(f"Pre-fetch task '{task_type}' for {identifier} returned None.")
             except Exception as exc:
                  logger.error(f"Exception in pre-fetch task '{task_type}' for {identifier}: {exc}", exc_info=False)

        # --- Ladder Fetch (dependent on badge results) ---
        cfpid_to_uuid_map = {}
        ladder_futures = {}
        if my_tree_id:
             # Collect CFPIDs from successful badge fetches
             cfpid_list = []
             for uuid_val, badge_result in temp_badge_results.items():
                  cfpid = badge_result.get("their_cfpid")
                  if cfpid:
                       cfpid_list.append(cfpid)
                       cfpid_to_uuid_map[cfpid] = uuid_val # Map CFPID back to UUID

             # Submit ladder tasks if CFPIDs were found
             if cfpid_list:
                  logger.debug(f"Submitting ladder pre-fetch for {len(cfpid_list)} CFPIDs...")
                  for cfpid in cfpid_list:
                       delay = session_manager.dynamic_rate_limiter.wait()
                       future = executor.submit(_fetch_batch_ladder, session_manager, cfpid, my_tree_id)
                       ladder_futures[future] = ("ladder", cfpid) # Use CFPID as identifier here
             else:
                  logger.debug("No valid CFPIDs found from badge details for ladder pre-fetch.")
        else:
             logger.debug("My tree ID not available, skipping ladder pre-fetch.")

        # Process ladder results
        for future in as_completed(ladder_futures):
             task_type, cfpid = ladder_futures[future] # Identifier is CFPID
             try:
                  result = future.result()
                  if result is not None:
                       batch_ladder_data[cfpid] = result # Store by CFPID
                  else:
                       logger.debug(f"Pre-fetch task 'ladder' for CFPID {cfpid} returned None.")
             except Exception as exc:
                  logger.error(f"Exception in pre-fetch task 'ladder' for CFPID {cfpid}: {exc}", exc_info=False)

    fetch_duration = time.time() - fetch_start_time
    logger.debug(f"--- Finished Batch Pre-fetch for Page {current_page}. Duration: {fetch_duration:.2f}s ---")
    # Log counts of fetched items
    logger.debug(f" Fetched: Combined={len(batch_combined_details)}, Badge={len(temp_badge_results)}, RelProb={len(batch_relationship_prob_data)}, Ladder={len(batch_ladder_data)}")

    # --- Combine Tree Data (Badge + Ladder) ---
    batch_tree_data: Dict[str, Dict[str, Any]] = {}
    for uuid_val, badge_result in temp_badge_results.items():
        combined_tree_info = badge_result.copy() # Start with badge data
        cfpid = badge_result.get("their_cfpid")
        # Look up corresponding ladder data using CFPID
        if cfpid and cfpid in batch_ladder_data:
            combined_tree_info.update(batch_ladder_data[cfpid]) # Add ladder data
        batch_tree_data[uuid_val] = combined_tree_info # Store combined tree data by UUID

    # --- Process Matches Using Prefetched Data ---
    prepared_bulk_data: List[Dict[str, Any]] = []
    page_statuses: Dict[str, int] = {"new": 0, "updated": 0, "skipped": 0, "error": 0}
    session = session_manager.get_db_conn()
    processed_count_in_loop = 0 # Track how many items actually entered the loop processing stage

    if not session:
        logger.error(f"_do_batch Page {current_page}: Failed to get DB session.")
        # --- REMOVED bulk progress bar update on DB error ---
        return 0, 0, 0, num_matches # Return errors equal to expected matches

    try:
        # --- REVISED LOOP with finally for progress bar update ---
        for match_index, match in enumerate(matches_on_page):
            status_for_item: Literal["new", "updated", "skipped", "error", "unknown"] = "unknown"
            try:
                processed_count_in_loop += 1 # Increment counter as we start processing an item
                _case_name = match.get("username", f"Unknown Match {match_index+1}")
                logger.debug(f"#### Page {current_page} - Prep Match {match_index + 1}/{num_matches}: {_case_name} ####")

                uuid_val = match.get("uuid")
                if not uuid_val:
                    logger.warning(f"Skipping prep for match {match_index+1} on page {current_page}: Missing UUID.")
                    page_statuses["error"] += 1
                    status_for_item = "error"
                    continue # Skip to finally block for this item

                # Retrieve prefetched data for this UUID
                prefetched_combined = batch_combined_details.get(uuid_val)
                prefetched_tree = batch_tree_data.get(uuid_val) # Already combined badge+ladder
                prefetched_rel_prob = batch_relationship_prob_data.get(uuid_val)

                # Assign predicted relationship to match dict (used by _do_match)
                match["predicted_relationship"] = (prefetched_rel_prob or "N/A (Fetch Failed)")

                # Process the match using prefetched data
                prepared_data, status, error_msg = _do_match(
                    session=session,
                    match=match,
                    session_manager=session_manager,
                    prefetched_combined_details=prefetched_combined,
                    prefetched_tree_data=prefetched_tree
                )
                status_for_item = status # Update status for use in finally (though not strictly needed now)
                logger.debug(f"  -> Match {_case_name}: _do_match status '{status}'. Tallying.")
                page_statuses[status] += 1
                if status != "error" and prepared_data:
                    prepared_bulk_data.append(prepared_data)
                elif status == "error":
                    logger.error(f"  -> Error preparing DB data for {_case_name}: {error_msg}")

                logger.debug(f"Finished {_case_name} data preparation.\n")

            except Exception as inner_e:
                logger.error(f"Critical error preparing data for match {_case_name} on page {current_page}: {inner_e}", exc_info=True)
                if status_for_item != "error": # Avoid double counting if _do_match already flagged error
                     page_statuses["error"] += 1
                status_for_item = "error"
                logger.debug(f"  -> Tallying 'error' due to exception.")
            finally:
                # --- Update bar in finally to ensure it always happens ---
                if progress_bar:
                    try:
                        progress_bar.update(1)
                        # --- REMOVED time.sleep(0.01) ---
                    except Exception as pbar_e:
                        logger.warning(f"Error updating progress bar for match: {pbar_e}")
                # --- End Update ---
        # --- End of loop for matches_on_page ---

        # --- Bulk Database Operations (outside the match loop) ---
        if prepared_bulk_data:
            logger.debug(f"--- Starting Bulk DB Operations for Page {current_page} ({len(prepared_bulk_data)} items) ---")
            # ... (bulk operations remain the same as before) ...
            bulk_start_time = time.time()
            try:
                # --- Extract data for bulk operations ---
                person_creates = [
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
                    d["dna_match"]
                    for d in prepared_bulk_data
                    if d.get("dna_match") # Assumes always create if present
                ]
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

                created_person_map: Dict[str, int] = {}

                # --- 1. Bulk Insert Persons ---
                if person_creates:
                    logger.debug(f"Bulk inserting {len(person_creates)} new Person records...")
                    insert_data = [{k: v for k, v in p.items() if not k.startswith("_")} for p in person_creates]
                    result = session.bulk_insert_mappings(Person, insert_data, return_defaults=True)
                    session.flush()
                    for p_data in insert_data:
                        if p_data.get("id") and p_data.get("uuid"): created_person_map[p_data["uuid"]] = p_data["id"]
                        else: logger.error(f"Person ID or UUID missing after bulk insert/flush for: {p_data.get('username')}")
                    logger.debug(f"Bulk inserted {len(created_person_map)} persons.")

                # --- 2. Bulk Update Persons ---
                if person_updates:
                    update_mappings = []
                    for p_data in person_updates:
                        existing_id = p_data.get("_existing_person_id")
                        if not existing_id: logger.warning(f"Skipping person update for UUID {p_data.get('uuid')}: Missing existing ID."); continue
                        update_dict = { k: v for k, v in p_data.items() if not k.startswith("_") and k != "uuid" }
                        if update_dict:
                            update_dict["id"] = existing_id; update_dict["updated_at"] = datetime.now()
                            update_mappings.append(update_dict)
                    if update_mappings:
                        logger.debug(f"Bulk updating {len(update_mappings)} existing Person records...")
                        session.bulk_update_mappings(Person, update_mappings)
                        logger.debug("Bulk updated persons.")
                    else: logger.debug("No Person records needed bulk updating this batch.")

                # --- 3. Prepare map of ALL relevant Person IDs ---
                all_person_ids_map = created_person_map.copy()
                for p_update_data in person_updates:
                    if p_update_data.get("_existing_person_id") and p_update_data.get("uuid"): all_person_ids_map[p_update_data["uuid"]] = p_update_data["_existing_person_id"]

                # --- 4. Bulk Insert DNA Matches ---
                if dna_match_creates:
                    dna_insert_data = []
                    for dna_data in dna_match_creates:
                        person_uuid = dna_data.get("uuid"); person_id = all_person_ids_map.get(person_uuid)
                        if person_id:
                            insert_dict = {k: v for k, v in dna_data.items() if not k.startswith("_")}
                            insert_dict["people_id"] = person_id; dna_insert_data.append(insert_dict)
                        else: logger.warning(f"Skipping DNA Match create for UUID {person_uuid}: Corresponding Person ID not found.")
                    if dna_insert_data:
                        logger.debug(f"Bulk inserting {len(dna_insert_data)} DnaMatch records...")
                        session.bulk_insert_mappings(DnaMatch, dna_insert_data); logger.debug("Bulk inserted DnaMatches.")

                # --- 5. Bulk Insert Family Trees ---
                if family_tree_creates:
                    tree_insert_data = []
                    for tree_data in family_tree_creates:
                         person_uuid = tree_data.get("uuid"); person_id = all_person_ids_map.get(person_uuid)
                         if person_id:
                              insert_dict = {k: v for k, v in tree_data.items() if not k.startswith('_')}
                              insert_dict["people_id"] = person_id; tree_insert_data.append(insert_dict)
                         else: logger.warning(f"Skipping FamilyTree create for UUID {person_uuid}: Corresponding Person ID not found.")
                    if tree_insert_data:
                        logger.debug(f"Bulk inserting {len(tree_insert_data)} FamilyTree records...")
                        session.bulk_insert_mappings(FamilyTree, tree_insert_data); logger.debug("Bulk inserted FamilyTrees.")

                # --- 6. Bulk Update Family Trees ---
                if family_tree_updates:
                    tree_update_mappings = []
                    for tree_data in family_tree_updates:
                        existing_id = tree_data.get("_existing_tree_id")
                        if not existing_id: logger.warning(f"Skipping FamilyTree update for UUID {tree_data.get('uuid')}: Missing existing ID."); continue
                        update_dict = {k: v for k, v in tree_data.items() if not k.startswith('_') and k != 'uuid'}
                        if update_dict:
                             update_dict["id"] = existing_id; update_dict["updated_at"] = datetime.now()
                             person_id = all_person_ids_map.get(tree_data.get("uuid"))
                             if person_id and "people_id" not in update_dict: update_dict["people_id"] = person_id
                             tree_update_mappings.append(update_dict)
                    if tree_update_mappings:
                        logger.debug(f"Bulk updating {len(tree_update_mappings)} FamilyTree records...")
                        session.bulk_update_mappings(FamilyTree, tree_update_mappings); logger.debug("Bulk updated FamilyTrees.")
                    else: logger.debug("No FamilyTree records needed bulk updating this batch.")

                # --- Final Commit ---
                logger.debug(f"Attempting final commit for page {current_page}...")
                session.commit()
                bulk_duration = time.time() - bulk_start_time
                logger.debug(f"Commit successful for page {current_page}. Bulk operations duration: {bulk_duration:.2f}s.")
            except (IntegrityError, SQLAlchemyError) as bulk_err:
                 logger.error(f"Bulk DB operation FAILED for page {current_page}: {bulk_err}", exc_info=True);
                 if session and session.is_active: session.rollback()
                 page_statuses["error"] += len(prepared_bulk_data); page_statuses["new"] = 0; page_statuses["updated"] = 0 # Mark all prepared items as errors
                 logger.warning(f"Page {current_page} counts adjusted due to bulk error: {page_statuses}")
            except Exception as bulk_e_unexp:
                 logger.critical(f"Unexpected Bulk DB Error for page {current_page}: {bulk_e_unexp}", exc_info=True);
                 if session and session.is_active: session.rollback()
                 page_statuses["error"] += len(prepared_bulk_data); page_statuses["new"] = 0; page_statuses["updated"] = 0 # Mark all prepared items as errors
                 logger.warning(f"Page {current_page} counts adjusted due to unexpected bulk error: {page_statuses}")
        else:
            logger.debug(f"No data prepared for bulk DB operations on page {current_page}.")

        # --- Log Page Summary ---
        _log_page_summary(current_page, page_statuses["new"], page_statuses["updated"], page_statuses["skipped"], page_statuses["error"])

    except Exception as outer_e:
        logger.error(f"Critical error during page {current_page} processing loop: {outer_e}", exc_info=True)
        if session and session.is_active:
            try: session.rollback(); logger.debug(f"Rolled back session.")
            except Exception as rb_err: logger.error(f"Failed rollback: {rb_err}")
        # Estimate errors for matches not processed in the loop due to outer error
        # Calculate remaining based on how many entered the processing loop
        remaining_count = num_matches - processed_count_in_loop
        page_statuses["error"] += remaining_count # Add remaining unprocessed as errors
        if progress_bar and remaining_count > 0:
            logger.warning(f"Updating progress bar by {remaining_count} for items skipped due to outer error.")
            try:
                progress_bar.update(remaining_count) # Update bar for unprocessed items
            except Exception as pbar_e: logger.warning(f"Error updating progress bar during outer error: {pbar_e}")

    finally:
        if session: # Ensure session is returned
            session_manager.return_session(session)

    return page_statuses["new"], page_statuses["updated"], page_statuses["skipped"], page_statuses["error"]
# end of _do_batch

def _do_match(
    session: Session,
    match: Dict[str, Any],
    session_manager: SessionManager,
    prefetched_combined_details: Optional[Dict[str, Any]],
    prefetched_tree_data: Optional[Dict[str, Any]],
) -> Tuple[
    Optional[Dict[str, Any]], Literal["new", "updated", "skipped", "error"], Optional[str]
]:
    """
    V14.5 REVISED: Processes match data, compares with existing DB record (if any),
    and returns prepared data dictionary containing ONLY the fields needing creation or update.
    - Determines 'new', 'updated', or 'skipped' status based on actual changes.
    - Uses format_name consistently.
    - Only updates username/message_link if the existing value is None/empty.
    """
    existing_person: Optional[Person] = None
    dna_match_record: Optional[DnaMatch] = None
    family_tree_record: Optional[FamilyTree] = None # Correct variable for existing tree

    match_uuid = match.get("uuid")
    match_username = match.get("username") # Already formatted from get_matches
    predicted_relationship = match.get("predicted_relationship", "N/A")
    match_in_my_tree = match.get("in_my_tree", False)
    log_ref = f"UUID={match_uuid or 'N/A'} User='{match_username or 'Unknown'}'"
    log_ref_short = f"UUID={match_uuid} User='{match_username}'"

    prepared_data_for_bulk: Dict[str, Any] = {"person": None, "dna_match": None, "family_tree": None}
    person_update_needed: bool = False
    tree_update_needed: bool = False
    overall_status: Literal["new", "updated", "skipped", "error"] = "error"

    if not match_uuid:
        error_msg = f"Pre-check failed: Missing 'uuid' in match data: {match}"
        logger.error(error_msg); return None, "error", error_msg

    try:
        # Step 1: DB Lookup (Same as V14.4)
        logger.debug(f"{log_ref}: Performing initial DB lookup by UUID...")
        try:
            existing_person = (
                session.query(Person)
                .options(joinedload(Person.dna_match), joinedload(Person.family_tree))
                .filter(Person.uuid == match_uuid.upper())
                .first()
            )
            if existing_person:
                logger.debug(f"{log_ref}: Found existing Person ID {existing_person.id}.")
                dna_match_record = existing_person.dna_match
                family_tree_record = existing_person.family_tree # Assign here
            else: logger.debug(f"{log_ref}: No existing person found by UUID.")
        except SQLAlchemyError as db_lookup_err:
            logger.error(f"Initial DB lookup failed for {log_ref_short}: {db_lookup_err}", exc_info=True)
            return None, "error", f"Initial DB lookup failed for {log_ref_short}"
        except Exception as lookup_err:
            logger.error(f"Unexpected error during initial DB lookup for {log_ref_short}: {lookup_err}", exc_info=True)
            return None, "error", f"Unexpected DB lookup error for {log_ref_short}"

        is_new_person = existing_person is None

        # Step 2: Prepare Incoming Data (Same as V14.4 - uses format_name for names)
        details_part = prefetched_combined_details or {}; profile_part = prefetched_combined_details or {}
        tester_profile_id = details_part.get("tester_profile_id") or match.get("profile_id")
        admin_profile_id = details_part.get("admin_profile_id") or match.get("administrator_profile_id_hint")
        raw_admin_username = details_part.get("admin_username") or match.get("administrator_username_hint")
        admin_username = format_name(raw_admin_username) # Format consistently
        person_profile_id_to_save, person_admin_id_to_save, person_admin_username_to_save = (None, None, None)
        if admin_profile_id and (not tester_profile_id or admin_profile_id.upper() != tester_profile_id.upper()):
             person_profile_id_to_save = tester_profile_id
             person_admin_id_to_save = admin_profile_id
             person_admin_username_to_save = admin_username
        elif admin_profile_id and tester_profile_id and admin_profile_id.upper() == tester_profile_id.upper():
             if admin_username and match_username and match_username.lower() != admin_username.lower():
                  person_profile_id_to_save = None
                  person_admin_id_to_save = admin_profile_id
                  person_admin_username_to_save = admin_username
             else:
                  person_profile_id_to_save = tester_profile_id
        elif tester_profile_id and not admin_profile_id:
             person_profile_id_to_save = tester_profile_id
        else:
             person_admin_id_to_save = admin_profile_id
             person_admin_username_to_save = admin_username
        message_target_id = person_admin_id_to_save or person_profile_id_to_save; constructed_message_link = None
        if message_target_id and session_manager.my_uuid:
             target_upper = message_target_id.upper(); my_uuid_upper = session_manager.my_uuid.upper(); match_uuid_upper = match_uuid.upper()
             constructed_message_link = urljoin(config_instance.BASE_URL, f"/messaging/?p={target_upper}&testguid1={my_uuid_upper}&testguid2={match_uuid_upper}")
        birth_year_val = None
        if prefetched_tree_data and prefetched_tree_data.get("their_birth_year"):
            try: birth_year_val = int(prefetched_tree_data["their_birth_year"])
            except (ValueError, TypeError): pass
        incoming_person_data = {
            "uuid": match_uuid.upper(), "profile_id": (person_profile_id_to_save.upper() if person_profile_id_to_save else None),
            "username": match_username, # Already formatted
            "administrator_profile_id": (person_admin_id_to_save.upper() if person_admin_id_to_save else None),
            "administrator_username": person_admin_username_to_save, # Already formatted
            "in_my_tree": match_in_my_tree,
            "first_name": match.get("first_name"), # Already formatted from get_matches
            "last_logged_in": profile_part.get("last_logged_in_dt"), "contactable": profile_part.get("contactable", False),
            "gender": details_part.get("gender"), "message_link": constructed_message_link, "birth_year": birth_year_val,
        }
        incoming_dna_data = None
        if dna_match_record is None and prefetched_combined_details is not None:
            incoming_dna_data = {
                "uuid": match_uuid.upper(), "compare_link": match.get("compare_link"), "cM_DNA": match.get("cM_DNA"),
                "predicted_relationship": predicted_relationship, "shared_segments": prefetched_combined_details.get("shared_segments"),
                "longest_shared_segment": prefetched_combined_details.get("longest_shared_segment"), "meiosis": prefetched_combined_details.get("meiosis"),
                "from_my_fathers_side": prefetched_combined_details.get("from_my_fathers_side", False), "from_my_mothers_side": prefetched_combined_details.get("from_my_mothers_side", False),
                "_operation": "create"
            }
        elif dna_match_record is None and prefetched_combined_details is None: logger.warning(f"{log_ref}: DNA Match should be created, but no details were fetched.")
        incoming_tree_data = None
        should_have_tree = match_in_my_tree and prefetched_tree_data is not None
        if should_have_tree:
            view_in_tree_link, facts_link = None, None
            their_cfpid_final = prefetched_tree_data.get("their_cfpid")
            if their_cfpid_final and session_manager.my_tree_id:
                base_tree_url = urljoin(config_instance.BASE_URL, f"/family-tree/person/tree/{session_manager.my_tree_id}/person/{their_cfpid_final}")
                view_in_tree_link = urljoin(base_tree_url, "family"); facts_link = urljoin(base_tree_url, "facts")
            tree_person_name = prefetched_tree_data.get("their_firstname", "Unknown") # Already formatted from _fetch_batch_badge_details
            incoming_tree_data = {
                "uuid": match_uuid.upper(), "cfpid": their_cfpid_final,
                "person_name_in_tree": tree_person_name, # Use formatted name
                "facts_link": facts_link, "view_in_tree_link": view_in_tree_link, "actual_relationship": prefetched_tree_data.get("actual_relationship"),
                "relationship_path": prefetched_tree_data.get("relationship_path"), "_operation": "create" if family_tree_record is None else "update",
                "_existing_tree_id": family_tree_record.id if family_tree_record else None
            }
        elif not match_in_my_tree and family_tree_record is not None: logger.debug(f"{log_ref}: Should not have tree record, but one exists. No tree data prepared.")


        # Step 3: Compare and Build Bulk Data Dictionary
        if is_new_person:
            # --- NEW PERSON --- (Same logic as V14.4)
            logger.debug(f"{log_ref}: Preparing data for NEW Person.")
            person_data_for_bulk = incoming_person_data.copy(); person_data_for_bulk["_operation"] = "create"
            prepared_data_for_bulk["person"] = person_data_for_bulk
            if incoming_dna_data: prepared_data_for_bulk["dna_match"] = incoming_dna_data; logger.debug(f"{log_ref}: Preparing data for NEW DnaMatch.")
            if incoming_tree_data: prepared_data_for_bulk["family_tree"] = incoming_tree_data; logger.debug(f"{log_ref}: Preparing data for NEW FamilyTree.")
            overall_status = "new"
        else:
            # --- EXISTING PERSON --- Compare fields ---
            person_data_for_update = {"_operation": "update", "_existing_person_id": existing_person.id, "uuid": match_uuid.upper()}
            person_update_needed = False

            # --- Comparisons with revised logic for username/message_link ---
            # last_logged_in (same logic)
            new_dt = incoming_person_data.get("last_logged_in"); old_dt = existing_person.last_logged_in; new_naive_ts = None; old_naive_ts = None
            if isinstance(new_dt, datetime): new_naive_ts = new_dt.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)
            if isinstance(old_dt, datetime): old_naive_ts = old_dt.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0) if old_dt.tzinfo else old_dt.replace(microsecond=0)
            if new_naive_ts != old_naive_ts: logger.debug(f"  -> Change detected for last_logged_in: {old_naive_ts} -> {new_naive_ts}"); person_data_for_update["last_logged_in"] = new_dt; person_update_needed = True

            # contactable (same logic)
            if bool(existing_person.contactable) != bool(incoming_person_data.get("contactable", False)): logger.debug(f"  -> Change detected for contactable"); person_data_for_update["contactable"] = bool(incoming_person_data.get("contactable", False)); person_update_needed = True

            # birth_year (same logic - only update if currently None)
            new_birth_year = incoming_person_data.get("birth_year")
            if new_birth_year is not None and existing_person.birth_year is None:
                 try: birth_year_int = int(new_birth_year); logger.debug(f"  -> Change detected for birth_year (adding)"); person_data_for_update["birth_year"] = birth_year_int; person_update_needed = True
                 except (ValueError, TypeError): logger.warning(f"  Skipping birth_year update for {log_ref}: New value '{new_birth_year}' not valid int.")

            # in_my_tree (same logic - update if different)
            if bool(existing_person.in_my_tree) != bool(incoming_person_data.get("in_my_tree", False)): logger.debug(f"  -> Change detected for in_my_tree"); person_data_for_update["in_my_tree"] = bool(incoming_person_data.get("in_my_tree", False)); person_update_needed = True

            # gender (same logic - only update if currently None)
            new_gender = incoming_person_data.get("gender")
            if new_gender is not None and existing_person.gender is None and isinstance(new_gender, str) and new_gender.lower() in ('f','m'): logger.debug(f"  -> Change detected for gender (adding)"); person_data_for_update["gender"] = new_gender.lower(); person_update_needed = True

            # admin info (same logic - update if different)
            new_admin_id = incoming_person_data.get("administrator_profile_id"); new_admin_user = incoming_person_data.get("administrator_username") # Already formatted
            if existing_person.administrator_profile_id != new_admin_id: logger.debug(f"  -> Change detected for administrator_profile_id"); person_data_for_update["administrator_profile_id"] = new_admin_id; person_update_needed = True
            if existing_person.administrator_username != new_admin_user: logger.debug(f"  -> Change detected for administrator_username"); person_data_for_update["administrator_username"] = new_admin_user; person_update_needed = True

            # --- REVISED: message link (only update if existing is empty/None) ---
            new_message_link = incoming_person_data.get("message_link")
            if not existing_person.message_link and new_message_link:
                 logger.debug(f"  -> Adding missing message_link for {log_ref}")
                 person_data_for_update["message_link"] = new_message_link
                 person_update_needed = True
            elif existing_person.message_link and new_message_link and existing_person.message_link != new_message_link:
                 logger.debug(f"  -> Skipping message_link update for {log_ref} (existing value present)")
            # --- END REVISED ---

            # --- REVISED: username (only update if existing is empty/None) ---
            new_username = incoming_person_data.get("username") # Already formatted
            if not existing_person.username and new_username:
                 logger.debug(f"  -> Adding missing username for {log_ref}")
                 person_data_for_update["username"] = new_username
                 person_update_needed = True
            elif existing_person.username and new_username and existing_person.username != new_username:
                 logger.debug(f"  -> Skipping username update for {log_ref} (existing value present and different: '{existing_person.username}' vs '{new_username}')")
            # --- END REVISED ---

            # --- Prepare Person data only if changes detected ---
            if person_update_needed: prepared_data_for_bulk["person"] = person_data_for_update; logger.debug(f"{log_ref}: Person data prepared for bulk update (changes found).")
            else: logger.debug(f"{log_ref}: No changes detected for Person.")

            # --- Prepare DNA data (only if creating) ---
            if incoming_dna_data: prepared_data_for_bulk["dna_match"] = incoming_dna_data; logger.debug(f"{log_ref}: Preparing data for NEW DnaMatch.")

            # --- Prepare Tree data (check for updates - same logic as V14.4) ---
            if incoming_tree_data and incoming_tree_data["_operation"] == "create":
                 prepared_data_for_bulk["family_tree"] = incoming_tree_data; logger.debug(f"{log_ref}: Preparing data for NEW FamilyTree.")
            elif incoming_tree_data and incoming_tree_data["_operation"] == "update":
                 tree_data_for_update = {"_operation": "update", "_existing_tree_id": family_tree_record.id, "uuid": match_uuid.upper()}
                 tree_update_needed = False
                 fields_to_check = ["cfpid", "person_name_in_tree", "facts_link", "view_in_tree_link", "actual_relationship", "relationship_path"]
                 for field in fields_to_check:
                      new_value = incoming_tree_data.get(field)
                      old_value = getattr(family_tree_record, field, None) if family_tree_record else None
                      if new_value != old_value:
                           logger.debug(f"  -> Tree Change detected for {field}: '{old_value}' -> '{new_value}'")
                           tree_data_for_update[field] = new_value; tree_update_needed = True
                 if tree_update_needed: prepared_data_for_bulk["family_tree"] = tree_data_for_update; logger.debug(f"{log_ref}: FamilyTree data prepared for bulk update (changes found).")
                 else: logger.debug(f"{log_ref}: No changes detected for FamilyTree.")

            # --- Determine overall status ---
            if person_update_needed or incoming_dna_data or tree_update_needed or (incoming_tree_data and incoming_tree_data["_operation"] == "create"):
                 overall_status = "updated"
            else: overall_status = "skipped"

        # Step 4: Final Return
        logger.debug(f"Final overall status determination for {log_ref_short}: {overall_status}")
        data_to_return = prepared_data_for_bulk if overall_status != "skipped" else None
        return data_to_return, overall_status, None

    except Exception as e:
        error_msg = f"Unexpected critical error in _do_match data preparation for {log_ref}: {e}."
        logger.error(error_msg, exc_info=True)
        # Ensure we return the correct tuple format on error
        return None, "error", error_msg # Return None for data, status 'error', and the message
# End of _do_match (V14.5)

#################################################################################
# 3. API Data Acquisition
#################################################################################

def get_matches(
    session_manager: SessionManager, db_session: SqlAlchemySession, current_page: int = 1
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    V14.16 REVISED: Fetches and processes match list data for a SINGLE page.
    - Uses specific headers similar to test2.py.
    - Sets allow_redirects=False for the Match List API call.
    - Retrieves CSRF token robustly.
    - Uses specific User-Agent and Sec-CH-UA headers.
    """
    total_pages: Optional[int] = None

    if not isinstance(session_manager, SessionManager):
        logger.error("Invalid SessionManager passed to get_matches.")
        return [], None

    driver = session_manager.driver
    if not driver or not session_manager.is_sess_valid():
        logger.error("WebDriver session is not valid in get_matches.")
        return [], None

    if not session_manager.my_uuid:
        logger.error("SessionManager my_uuid is not initialized in get_matches.")
        return [], None

    my_uuid = session_manager.my_uuid
    csrf_token_cookie_name = "_dnamatches-matchlistui-x-csrf-token"
    fallback_csrf_cookie_name = "_csrf" # Added fallback

    try:
        # --- Get CSRF Token from Cookie ---
        logger.debug(f"Attempting to read CSRF cookie '{csrf_token_cookie_name}' or fallback '{fallback_csrf_cookie_name}' from browser...")
        specific_csrf_token = None
        found_token_name = None
        for cookie_name in [csrf_token_cookie_name, fallback_csrf_cookie_name]:
            try:
                cookie_obj = driver.get_cookie(cookie_name)
                if cookie_obj and isinstance(cookie_obj, dict) and 'value' in cookie_obj:
                    # Decode and split value if needed (similar to test2.py)
                    raw_value = cookie_obj['value']
                    if raw_value:
                         specific_csrf_token = unquote(raw_value).split('|')[0]
                         found_token_name = cookie_name
                         logger.info(f"Successfully read CSRF token from cookie '{found_token_name}': {specific_csrf_token[:10]}...")
                         break # Found a token, stop checking
                    else:
                         logger.debug(f"Cookie '{cookie_name}' found but value is empty.")
                # else: logger.debug(f"Cookie '{cookie_name}' not found or invalid format via get_cookie.")

            except NoSuchCookieException:
                 logger.debug(f"CSRF cookie '{cookie_name}' not found via get_cookie.")
                 continue # Try next cookie name
            except WebDriverException as cookie_e:
                 logger.warning(f"WebDriver error getting cookie '{cookie_name}': {cookie_e}")
                 continue # Try next cookie name
            except Exception as e:
                 logger.error(f"Unexpected error getting cookie '{cookie_name}': {e}", exc_info=True)
                 continue # Try next cookie name

        # Fallback using get_driver_cookies if specific attempts failed
        if not specific_csrf_token:
            logger.debug(f"CSRF token not found via get_cookie. Trying fallback with get_driver_cookies...")
            all_cookies = get_driver_cookies(driver)
            for cookie_name in [csrf_token_cookie_name, fallback_csrf_cookie_name]:
                 if cookie_name in all_cookies:
                      raw_value = all_cookies[cookie_name]
                      if raw_value:
                           specific_csrf_token = unquote(raw_value).split('|')[0]
                           found_token_name = cookie_name
                           logger.info(f"Successfully read CSRF token via fallback get_driver_cookies ('{found_token_name}'): {specific_csrf_token[:10]}...")
                           break
                      else:
                           logger.debug(f"Cookie '{cookie_name}' found via fallback but value is empty.")

        if not specific_csrf_token:
             logger.error(f"Failed to obtain a valid CSRF token from cookies. Cannot call Match List API.")
             return [], None
        # --- End CSRF Token Reading ---

        # --- Fetch Match List Data ---
        match_list_url = urljoin(
            config_instance.BASE_URL,
            f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={current_page}",
        )
        logger.debug(f"Fetching match list for page {current_page} using requests...")

        # --- Define headers based on test2.py ---
        # Use a consistent Chrome version (e.g., 125) or make dynamic if needed
        chrome_version = "125" # Or retrieve dynamically if preferred
        user_agent = (
            f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            f"(KHTML, like Gecko) Chrome/{chrome_version}.0.0.0 Safari/537.36"
        )
        sec_ch_ua = (
            f'"Google Chrome";v="{chrome_version}", '
            '"Not-A.Brand";v="8", "Chromium";v="{chrome_version}"'
        )

        match_list_headers = {
            "User-Agent": user_agent,
            "accept": "application/json",
            "Referer": urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"), # Referer from match list page
            "x-csrf-token": specific_csrf_token, # Use the retrieved token
            "sec-ch-ua": sec_ch_ua,
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8", # Common language header
            # Add other potentially important headers observed in successful requests
            "Origin": config_instance.BASE_URL.rstrip('/'),
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            # "priority": "u=1, i", # This might be less critical
            "dnt": "1", # Do Not Track often included
            # Content-Type might not be needed for GET, but doesn't hurt
            "Content-Type": "application/json",
        }
        # UBE header is usually added by _api_req if driver is available

        logger.debug(f"Headers prepared for Match List API call: {match_list_headers}")

        api_response = _api_req(
            url=match_list_url,
            driver=driver,
            session_manager=session_manager,
            method="GET",
            headers=match_list_headers,
            use_csrf_token=False, # Token is manually included in headers
            api_description="Match List API",
            allow_redirects=False, # *** CRITICAL CHANGE: Do not allow redirects ***
        )

        # --- Handle API Response ---
        if api_response is None:
            # _api_req logs errors internally, including status codes if response object exists
            logger.warning(f"No response or error from match list API for page {current_page}.")
            return [], None

        if not isinstance(api_response, dict):
            logger.error(
                f"Match List API call did not return a dictionary (received {type(api_response)}). Aborting page {current_page}."
            )
            # Log content if it's not a dict
            if isinstance(api_response, (str, bytes)):
                try:
                    content_preview = api_response.decode() if isinstance(api_response, bytes) else api_response
                    logger.debug(f"Received non-dict content (first 500 chars): {content_preview[:500]}")
                except Exception as decode_err:
                     logger.debug(f"Received non-dict content (could not decode): {type(api_response)}")
            else:
                 logger.debug(f"Received non-dict/non-str data: {api_response}")
            return [], None

        # --- Process the Dictionary Response (should be 200 OK now) ---
        total_pages_raw = api_response.get("totalPages"); total_pages = None
        if total_pages_raw is not None:
            try: total_pages = int(total_pages_raw)
            except (ValueError, TypeError): logger.warning(f"Could not parse totalPages '{total_pages_raw}' to int.")
        else: logger.warning("totalPages key missing from Match List API response.")

        match_data_list = api_response.get("matchList", [])
        if not match_data_list:
            logger.info(f"No matches found in 'matchList' for page {current_page}.")
            # Still return total_pages if available
            return [], total_pages

        logger.debug(f"Got {len(match_data_list)} raw matches from API on page {current_page}.")

        valid_matches_for_processing: List[Dict[str, Any]] = []
        skipped_sampleid_count = 0
        for m in match_data_list:
            if isinstance(m, dict) and m.get("sampleId"): valid_matches_for_processing.append(m)
            else: skipped_sampleid_count += 1; logger.warning(f"Skipping raw match due to missing 'sampleId' on page {current_page}. Data: {m}")
        if skipped_sampleid_count > 0: logger.warning(f"Skipped {skipped_sampleid_count} matches on page {current_page} (missing 'sampleId').")
        if not valid_matches_for_processing: logger.warning(f"No matches with valid 'sampleId' found on page {current_page}."); return [], total_pages

        sample_ids_on_page = [match["sampleId"].upper() for match in valid_matches_for_processing]

        # --- In-Tree Status Check (remains the same, uses POST, default allow_redirects=True) ---
        in_tree_ids: Set[str] = set()
        cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids_on_page))}"
        cached_in_tree = session_manager.cache.get(cache_key_tree)
        if cached_in_tree is not None and isinstance(cached_in_tree, set):
            in_tree_ids = cached_in_tree
            logger.debug(f"Loaded {len(in_tree_ids)} in-tree IDs from cache for page {current_page}.")
        else:
            in_tree_url = urljoin(config_instance.BASE_URL, f"discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid.upper()}")
            logger.debug(f"Fetching in-tree status for page {current_page}...")
            # Re-use the retrieved CSRF token for this POST request
            in_tree_headers = {
                 "X-CSRF-Token": specific_csrf_token, # Use same token
                 "Content-Type": "application/json",
                 # Add other potentially relevant headers if needed, like User-Agent
                 "User-Agent": user_agent,
                 "Referer": urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
                 "Origin": config_instance.BASE_URL.rstrip('/'),
                 "Accept": "application/json", # Explicitly accept JSON
            }
            # UBE will be added by _api_req
            response_in_tree = _api_req(
                url=in_tree_url, driver=driver, session_manager=session_manager,
                method="POST", json_data={"sampleIds": sample_ids_on_page},
                headers=in_tree_headers,
                use_csrf_token=False, # Manually included token
                api_description="In-Tree Status Check",
                # allow_redirects=True (default) is fine here for POST
            )
            if isinstance(response_in_tree, list):
                in_tree_ids = {item.upper() for item in response_in_tree if isinstance(item, str)}
                session_manager.cache.set(cache_key_tree, in_tree_ids, timeout=config_instance.CACHE_TIMEOUT)
                logger.debug(f"Fetched/cached {len(in_tree_ids)} in-tree IDs for page {current_page}.")
            else: logger.warning(f"In-Tree Status Check API failed or returned unexpected format for page {current_page}. Response: {response_in_tree}")

        # --- Compile Final Refined Match Data (remains the same) ---
        refined_matches: List[Dict[str, Any]] = []
        skipped_profile_id_count = 0
        for match in valid_matches_for_processing:
             profile = match.get("matchProfile", {}); relationship = match.get("relationship", {})
             sample_id_upper = match["sampleId"].upper(); profile_user_id = profile.get("userId")
             raw_display_name = profile.get("displayName"); match_username = format_name(raw_display_name)
             first_name = match_username.split()[0] if match_username != "Valued Relative" else None
             admin_profile_id_hint = match.get("adminId"); admin_username_hint = match.get("adminName")
             if not profile_user_id: logger.debug(f"Match '{match_username}' (UUID: {sample_id_upper}) missing tester 'profile_id'."); skipped_profile_id_count += 1; profile_user_id_upper = None
             else: profile_user_id_upper = str(profile_user_id).upper()
             compare_link = urljoin(config_instance.BASE_URL, f"discoveryui-matches/compare/{my_uuid.upper()}/with/{sample_id_upper}")
             refined_match_data = { "username": match_username, "first_name": first_name, "initials": profile.get("displayInitials", "??").upper(), "gender": match.get("gender"), "profile_id": profile_user_id_upper, "uuid": sample_id_upper, "administrator_profile_id_hint": admin_profile_id_hint, "administrator_username_hint": admin_username_hint, "photoUrl": profile.get("photoUrl", ""), "cM_DNA": int(relationship.get("sharedCentimorgans", 0)), "numSharedSegments": int(relationship.get("numSharedSegments", 0)), "compare_link": compare_link, "message_link": None, "in_my_tree": sample_id_upper in in_tree_ids, "createdDate": match.get("createdDate"), }
             refined_matches.append(refined_match_data)

        if skipped_profile_id_count > 0: logger.warning(f"Skipped {skipped_profile_id_count} matches on page {current_page} due to missing tester 'profile_id'.")
        logger.debug(f"Processed page {current_page}: Raw={len(match_data_list)}, Refined={len(refined_matches)}")
        return refined_matches, total_pages

    # --- Exception Handling (remains same) ---
    except requests.exceptions.RequestException as e: logger.error(f"Network/Request error processing page {current_page}: {e}", exc_info=True); return [], None
    except NoSuchCookieException as e: logger.critical(f"Critical error in get_matches: Could not find required CSRF cookie. {e}", exc_info=True); return [], None
    except Exception as e: logger.critical(f"Critical error processing match data for page {current_page}: {e}", exc_info=True); return [], None
# end get_matches

@retry_api()
def _fetch_combined_details(
    session_manager: SessionManager, match_uuid: str
) -> Optional[Dict[str, Any]]:
    """
    V13 FIXED: Fetches data from /details and /profiles/details for a single match UUID.
    - Corrected timezone import error.
    - Returns a combined dictionary or None on failure.
    """
    if not session_manager.my_uuid or not match_uuid:
        logger.warning("Missing my_uuid or match_uuid for combined details fetch.")
        return None

    details_data = {}
    profile_data = {}
    combined_data = {}

    # 1. Fetch /details
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
            driver=session_manager.driver,  # Still needed for UBE header generation
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description="Match Details API (Batch)",
            referer_url=details_referer,
        )
        if details_response and isinstance(details_response, dict):
            details_data = details_response
            # Add essential details to combined_data immediately
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
            # Continue to profile fetch if possible, but mark details as potentially incomplete
    except Exception as e:
        logger.error(
            f"Error fetching /details for UUID {match_uuid}: {e}", exc_info=True
        )
        # Propagate RequestException for retry decorator
        if isinstance(e, requests.exceptions.RequestException):
            raise

    # 2. Fetch /profiles/details (using tester_profile_id from /details)
    tester_profile_id_for_api = combined_data.get("tester_profile_id")
    my_profile_id_header = session_manager.my_profile_id

    if not tester_profile_id_for_api:
        logger.debug(
            f"Skipping /profiles/details fetch for {match_uuid}: tester_profile_id not found."
        )
        combined_data["last_logged_in_dt"] = None
        combined_data["contactable"] = False
    elif not my_profile_id_header:
        logger.warning(
            f"Skipping /profiles/details fetch for {match_uuid}: Own profile ID missing for header."
        )
        combined_data["last_logged_in_dt"] = None
        combined_data["contactable"] = False
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
                driver=session_manager.driver,  # For UBE
                session_manager=session_manager,
                method="GET",
                headers=profile_headers,
                use_csrf_token=False,
                api_description="Profile Details API (Batch)",
                referer_url=details_referer,  # Use same referer
            )
            if profile_response and isinstance(profile_response, dict):
                profile_data = profile_response
                last_login_str = profile_data.get("LastLoginDate")
                contactable_val = profile_data.get("IsContactable", False)
                last_login_dt = None
                if last_login_str:
                    try:
                        # --- V13 FIX: Ensure timezone is imported and used ---
                        if last_login_str.endswith("Z"):
                            # Parse ISO format string correctly, aware of Z for UTC
                            last_login_dt = datetime.fromisoformat(
                                last_login_str.replace("Z", "+00:00")
                            )
                        else:
                            # Assume naive string needs UTC assigned if no timezone info
                            dt_naive = datetime.fromisoformat(last_login_str)
                            if dt_naive.tzinfo is None:
                                last_login_dt = dt_naive.replace(tzinfo=timezone.utc)
                            else:
                                last_login_dt = dt_naive  # Already timezone-aware

                        # Store aware datetime directly, ensuring it's UTC
                        combined_data["last_logged_in_dt"] = last_login_dt.astimezone(
                            timezone.utc
                        )
                        # --- END FIX ---
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
        except (
            NameError
        ) as ne:  # Catch if timezone import was missing (belt-and-braces)
            logger.critical(
                f"NameError in _fetch_combined_details (likely timezone): {ne}",
                exc_info=True,
            )
            combined_data["last_logged_in_dt"] = None
            combined_data["contactable"] = False
            # Do not raise, return what we have so far
        except Exception as e:
            logger.error(
                f"Error fetching /profiles/details for {tester_profile_id_for_api}: {e}",
                exc_info=True,
            )
            combined_data["last_logged_in_dt"] = None
            combined_data["contactable"] = False
            if isinstance(e, requests.exceptions.RequestException):
                raise

    # Return combined data only if we got at least the essential tester_profile_id
    return combined_data if "tester_profile_id" in combined_data else None
# end _fetch_combined_details

@retry_api()
def _fetch_batch_badge_details(
    session_manager: SessionManager, match_uuid: str
) -> Optional[Dict[str, Any]]:
    """
    Fetches /badgedetails for a single match UUID.
    Uses format_name for consistency.
    """
    if not session_manager.my_uuid or not match_uuid:
        logger.warning("Missing my_uuid or match_uuid for badge details fetch.")
        return None

    badge_url = urljoin(
        config_instance.BASE_URL,
        f"/discoveryui-matchesservice/api/samples/{session_manager.my_uuid}/matches/{match_uuid}/badgedetails",
    )
    try:
        badge_response = _api_req(
            url=badge_url,
            driver=session_manager.driver,  # For UBE
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,  # Usually not needed for GET
            api_description="Badge Details API (Batch)",
            referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
        )

        if badge_response and isinstance(badge_response, dict):
            person_badged = badge_response.get("personBadged", {})
            # --- Use format_name ---
            raw_firstname = person_badged.get("firstName")
            their_firstname_formatted = format_name(raw_firstname).split()[0] if raw_firstname else "Unknown"
            # --- End format_name ---

            return {  # Return only the needed fields
                "their_cfpid": person_badged.get("personId"),
                "their_firstname": their_firstname_formatted, # Use formatted name
                "their_lastname": person_badged.get("lastName", "Unknown"), # Keep as is for now, format_name focuses on first/full names
                "their_birth_year": person_badged.get("birthYear"),
            }
        else:
            logger.warning(f"Invalid badge details response for UUID {match_uuid}.")
            return None
    except Exception as e:
        logger.error(
            f"Error fetching badge details for UUID {match_uuid}: {e}", exc_info=True
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None
# end _fetch_batch_badge_details

@retry_api()
def _fetch_batch_ladder(
    session_manager: SessionManager, cfpid: str, tree_id: str
) -> Optional[Dict[str, Any]]:
    """
    V14.2 REVISED: Fetches and parses /getladder for a single CFPID.
    - Forces text response parsing in _api_req.
    """
    if not cfpid or not tree_id:
        logger.warning("Missing cfpid or tree_id for ladder fetch.")
        return None

    ladder_api_url = urljoin(
        config_instance.BASE_URL,
        f"family-tree/person/tree/{tree_id}/person/{cfpid}/getladder?callback=jQuery",
    )
    dynamic_referer = urljoin(
        config_instance.BASE_URL,
        f"family-tree/person/tree/{tree_id}/person/{cfpid}/facts",
    )
    ladder_data = {}

    try:
        # --- Request text response explicitly ---
        response_text = _api_req(
            url=ladder_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers={"Accept": "*/*"}, # Keep broad accept header
            use_csrf_token=False,
            api_description="Get Ladder API (Batch)",
            referer_url=dynamic_referer,
            force_text_response=True, # <<< FORCE TEXT RESPONSE >>>
        )
        # --- End request text response explicitly ---

        if response_text and isinstance(response_text, str):
            # --- JSONP Parsing Logic (remains the same) ---
            match_jsonp = re.match(
                r"^[^(]*\((.*)\)[^)]*$", response_text, re.DOTALL | re.IGNORECASE
            )
            if match_jsonp:
                json_string = match_jsonp.group(1).strip()
                try:
                    # --- Check for empty JSON string specifically ---
                    if not json_string or json_string == '""' or json_string == "''":
                        logger.warning(f"Empty JSON content within JSONP for cfpid {cfpid}.")
                        return None # Treat empty content as failure
                    # --- End check for empty JSON string ---

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
                                if name_link:
                                    nested_b = name_link.find("b")
                                    raw_name_extracted = (
                                        nested_b.get_text(strip=True)
                                        if nested_b
                                        else name_link.get_text(strip=True)
                                    )
                                elif name_bold:
                                    raw_name_extracted = name_bold.get_text(strip=True)
                                else:
                                    parts = item.get_text(
                                        separator="\n", strip=True
                                    ).split("\n")
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
                                        raw_desc_full = desc_element.get_text(
                                            strip=True
                                        )
                                        cleaned_desc_full = raw_desc_full.replace(
                                            '"', "'"
                                        )
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
                                                desc_text = format_name(
                                                    cleaned_desc_full
                                                )
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

                            ladder_data["actual_relationship"] = (
                                actual_relationship_text
                            )
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
                # --- Catch JSONDecodeError specifically for the inner parsing ---
                except json.JSONDecodeError as inner_json_err:
                    logger.error(f"Failed to decode JSONP content for cfpid {cfpid}: {inner_json_err}")
                    logger.debug(f"JSON string causing decode error: '{json_string[:200]}'") # Log the problematic string
                    return None # Failure
            else:
                logger.error(f"Could not parse JSONP format for cfpid {cfpid}.")
        # --- Handle case where _api_req returned None or non-string ---
        elif response_text is None:
             logger.warning(f"_api_req returned None for Get Ladder API for cfpid {cfpid}.")
        else:
             logger.warning(f"_api_req returned unexpected type '{type(response_text).__name__}' for Get Ladder API for cfpid {cfpid}.")

        return None  # Return None if parsing failed or response invalid

    except Exception as e:
        logger.error(
            f"Error fetching/parsing ladder for CFPID {cfpid}: {e}", exc_info=True
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None
# end _fetch_batch_ladder

@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, HTTPError, cloudscraper.exceptions.CloudflareException)) # Add CloudflareException
def _fetch_batch_relationship_prob(
    session_manager: SessionManager, match_uuid: str, max_labels_param: int
) -> Optional[str]:
    """
    V14.5 REVISED: Fetches /matchProbabilityData using cloudscraper.
    - Uses cloudscraper to handle potential Cloudflare/JS challenges.
    - Constructs headers based on cURL example.
    - Copies cookies from WebDriver to scraper session.
    """
    driver = session_manager.driver

    if not session_manager.my_uuid or not match_uuid:
        logger.warning("Missing my_uuid or match_uuid for relationship probability fetch.")
        return "N/A (Error - Missing IDs)"
    # Removed redundant CSRF check here, rely on getting it fresh below

    if not driver or not session_manager.is_sess_valid():
         logger.error("Driver not available or session invalid for fetching relationship probability.")
         return "N/A (Error - Driver Invalid)"

    my_uuid_upper = session_manager.my_uuid.upper()
    sample_id_upper = match_uuid.upper()
    rel_url = urljoin(
        config_instance.BASE_URL,
        f"discoveryui-matches/parents/list/api/matchProbabilityData/{my_uuid_upper}/{sample_id_upper}",
    )
    referer_url = urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/")

    # --- Prepare Headers based on cURL & Script ---
    chrome_version = "125" # Consistent version
    user_agent = (
        f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{chrome_version}.0.0.0 Safari/537.36"
    )
    sec_ch_ua = (
        f'"Google Chrome";v="{chrome_version}", '
        '"Not-A.Brand";v="8", "Chromium";v="{chrome_version}"'
    )
    rel_headers = {
        "Accept": "application/json", # Match cURL
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache", # Add from cURL
        "Content-Type": "application/json",
        # CSRF Token added below after fetching fresh
        "Origin": config_instance.BASE_URL.rstrip('/'),
        "Pragma": "no-cache", # Add from cURL
        "Priority": "u=1, i", # Add from cURL (optional)
        "Referer": referer_url,
        "sec-ch-ua": sec_ch_ua,
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "User-Agent": user_agent,
    }

    # --- Get CSRF Token from WebDriver cookies just before call ---
    csrf_token_val = None
    csrf_cookie_names = ("_dnamatches-matchlistui-x-csrf-token", "_csrf")
    try:
        # Use a temporary dict to avoid modifying SessionManager state if get_cookies fails
        driver_cookies_list = driver.get_cookies()
        if driver_cookies_list:
             driver_cookies_dict = {c['name']: c['value'] for c in driver_cookies_list if 'name' in c and 'value' in c}
             for name in csrf_cookie_names:
                  if name in driver_cookies_dict and driver_cookies_dict[name]:
                       csrf_token_val = unquote(driver_cookies_dict[name]).split('|')[0]
                       rel_headers["X-CSRF-Token"] = csrf_token_val # Add specific token to header
                       logger.debug(f"Using fresh CSRF token '{name}' from driver cookies for rel prob.")
                       break
        else:
             logger.warning("driver.get_cookies() returned None or empty list.")

    except WebDriverException as csrf_wd_e:
         logger.warning(f"WebDriverException getting cookies for CSRF token: {csrf_wd_e}")
    except Exception as csrf_e:
        logger.warning(f"Could not get fresh CSRF token from driver cookies: {csrf_e}")

    if "X-CSRF-Token" not in rel_headers:
         # Fallback: Try using the SessionManager's token if fresh failed
         if session_manager.csrf_token:
              logger.warning("Using potentially stale CSRF token from SessionManager as fallback.")
              rel_headers["X-CSRF-Token"] = session_manager.csrf_token
         else:
              logger.error("Failed to add CSRF token to headers for cloudscraper request (fresh and fallback failed).")
              return "N/A (Error - Missing CSRF)"

    api_description = "Match Probability API (Cloudscraper)"

    try:
        # --- Create scraper instance ---
        scraper = cloudscraper.create_scraper(delay=5)

        # --- Set cookies from WebDriver ---
        try:
            driver_cookies = driver.get_cookies() # Get cookies again for scraper
            if not driver_cookies:
                logger.warning(f"{api_description}: WebDriver returned no cookies to set in scraper.")
            else:
                logger.debug(f"{api_description}: Populating scraper cookie jar with {len(driver_cookies)} cookies from WebDriver...")
                for cookie in driver_cookies:
                    if 'name' in cookie and 'value' in cookie and 'domain' in cookie:
                        # Use scraper's internal cookie handling
                        scraper.cookies.set(
                            cookie['name'],
                            cookie['value'],
                            domain=cookie.get('domain'),
                            path=cookie.get('path', '/')
                        )
        except WebDriverException as cookie_err:
            logger.error(f"{api_description}: WebDriverException getting cookies for scraper: {cookie_err}")
        except Exception as e:
            logger.error(f"{api_description}: Unexpected error setting cookies for scraper: {e}", exc_info=True)

        # --- Make the POST request using cloudscraper ---
        logger.debug(f"Making {api_description} POST request to {rel_url}")
        response_rel = scraper.post(
            rel_url,
            headers=rel_headers,
            json={}, # Send empty JSON body
            allow_redirects=False,
            timeout=selenium_config.API_TIMEOUT
        )
        # --- End request ---

        logger.debug(f"<-- {api_description} Response Status: {response_rel.status_code} {response_rel.reason}")

        # --- Handle Response ---
        if not response_rel.ok:
             status_code = response_rel.status_code
             logger.warning(
                 f"{api_description} failed for {sample_id_upper}. Status: {status_code}, Reason: {response_rel.reason}"
             )
             try: logger.debug(f"  Response Body: {response_rel.text[:500]}")
             except Exception: pass
             if 300 <= status_code < 400:
                  logger.warning(f"  -> Redirect detected (Loc: {response_rel.headers.get('Location')}). Check headers/cookies.")
             return "N/A (API Error/Redirect)"

        # --- Process successful (2xx) response ---
        try:
            if not response_rel.content:
                logger.warning(f"{api_description}: OK ({response_rel.status_code}), but response body is EMPTY. Returning None.")
                return "N/A (Empty Response)"

            data = response_rel.json()

            # --- Relationship parsing logic (remains the same) ---
            if "matchProbabilityToSampleId" not in data:
                logger.warning(f"Invalid data structure from {api_description} for {sample_id_upper}. Resp: {data}")
                return "N/A (Invalid Data Structure)"
            prob_data = data["matchProbabilityToSampleId"]
            predictions = prob_data.get("relationships", {}).get("predictions", [])
            if not predictions:
                logger.debug(f"No relationship predictions found for {sample_id_upper}. Marking as Distant.")
                return "Distant relationship?"
            valid_preds = [p for p in predictions if isinstance(p, dict) and "distributionProbability" in p and "pathsToMatch" in p]
            if not valid_preds:
                logger.warning(f"No valid prediction paths found for {sample_id_upper}.")
                return "N/A (No Valid Paths)"
            best_pred = max(valid_preds, key=lambda x: x.get("distributionProbability", 0.0))
            top_prob = best_pred.get("distributionProbability", 0.0)
            paths = best_pred.get("pathsToMatch", [])
            labels = [p.get("label") for p in paths if isinstance(p, dict) and p.get("label")]
            if not labels:
                logger.warning(f"Prediction found for {sample_id_upper}, but no labels in paths.")
                return f"N/A (No Labels) [{top_prob:.1f}%]"
            final_labels = labels[:max_labels_param]
            relationship_str = " or ".join(map(str, final_labels))
            return f"{relationship_str} [{top_prob:.1f}%]"
            # --- End relationship parsing ---

        except json.JSONDecodeError as json_err:
            logger.error(f"{api_description}: OK ({response_rel.status_code}), but JSON decode FAILED: {json_err}")
            logger.debug(f"Response text causing decode error: {response_rel.text[:500]}")
            return "N/A (JSON Decode Error)"
        except Exception as e:
            logger.error(f"{api_description}: Error processing successful response for {sample_id_upper}: {e}", exc_info=True)
            return "N/A (Processing Error)"

    # --- Handle Exceptions during scraper request ---
    except cloudscraper.exceptions.CloudflareException as cf_e:
         logger.error(f"{api_description}: Cloudflare challenge failed for {sample_id_upper}: {cf_e}")
         raise cf_e # Re-raise for retry_api decorator
    except requests.exceptions.RequestException as req_e:
        logger.error(f"{api_description}: RequestException for {sample_id_upper}: {req_e}")
        raise req_e # Re-raise for retry_api decorator
    except Exception as e:
        logger.error(f"{api_description}: Unexpected error for {sample_id_upper}: {type(e).__name__} - {e}", exc_info=True)
        return "N/A (Fetch Error)"
# end _fetch_batch_relationship_prob


#################################################################################
# 5. 'Create or Update' Database Operations
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
    logger.info("---- Gather Matches Final Summary ----") # More specific title
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
    # Ensure session is valid and UUID is available
    if not session_manager.is_sess_valid() or not session_manager.my_uuid:
        logger.error(
            "Session invalid or user UUID missing. Cannot navigate to matches list."
        )
        return False

    # --- CHANGE: Construct the URL with the user's UUID ---
    matches_url_with_uuid = urljoin(
        config_instance.BASE_URL, f"discoveryui-matches/list/{session_manager.my_uuid}"
    )
    # --- END CHANGE ---

    # Call nav_to_page with the CORRECT target URL
    success = nav_to_page(
        session_manager.driver,
        matches_url_with_uuid,  # Pass the UUID-specific URL
        selector=MATCH_ENTRY_SELECTOR,  # Wait for a match entry
        session_manager=session_manager,
    )

    if success:
        # Add extra verification if needed
        try:
            current_url = session_manager.driver.current_url
            if not current_url.startswith(matches_url_with_uuid):
                logger.warning(
                    f"Navigation reported success, but final URL is unexpected: {current_url}"
                )
                # return False # Optionally treat unexpected URL as failure
        except Exception as e:
            logger.warning(f"Could not verify final URL after nav_to_list: {e}")
    else:
        logger.error("Failed to navigate to specific matches list page.")

    return success
# end nav_to_list

# end of action6_gather.py
