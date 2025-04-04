#!/usr/bin/env python3

# action6_gather.py

# Standard library imports (alphabetical)
import json
import logging
import math
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

# Third-party imports (alphabetical by package)
import requests
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup, Tag
from requests.exceptions import RequestException
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
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
from my_selectors import MATCH_ENTRY_SELECTOR
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
)

#################################################################################
# 1. Setup & Verification
#################################################################################

# Initialize logging
logger = logging.getLogger("logger")


#################################################################################
# 2. Core Orchestration
#################################################################################

def coord(session_manager: SessionManager, config_instance, start: int = 1) -> bool:
    """
    V13.6 REVISED: Gathers DNA matches, processing page-by-page.
    - Added logging after accumulation within the loop.
    """
    # Ensure session and driver are valid
    driver = session_manager.driver
    if not driver or not session_manager.session_active:
        logger.error("WebDriver not initialized or session not active. Exiting coord.")
        return False

    # Initialise counts
    total_new, total_updated, total_skipped, total_errors = 0, 0, 0, 0
    total_pages_processed = 0
    my_uuid = session_manager.my_uuid

    if not my_uuid:
        logger.error("Failed to retrieve my_uuid from session_manager. Exiting coord.")
        return False

    target_matches_url_base = urljoin(
        config_instance.BASE_URL, f"discoveryui-matches/list/{my_uuid}"
    )
    final_success = True
    total_pages: Optional[int] = None
    last_page: Optional[int] = None

    try:
        # 11. Ensure we are on the DNA Match Page
        # ... (Navigation check remains the same) ...
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
                    current_url_after_nav = driver.current_url
                    if not current_url_after_nav.startswith(target_matches_url_base):
                        logger.error(
                            f"nav_to_list ended on unexpected URL: {current_url_after_nav}. Exiting coord."
                        )
                        return False
                    logger.debug("Successfully navigated to DNA matches page.\n")
            else:
                logger.debug(f"Already on correct DNA matches page: {current_url}.\n")
        except WebDriverException as nav_e:
            logger.error(f"WebDriver error checking/navigating: {nav_e}", exc_info=True)
            return False

        # 13. Determine initial page range guess
        logger.debug("13. Determining initial page range guess...")
        max_pages_config = config_instance.MAX_PAGES
        start_page = max(1, start)
        temp_limit_pages = 100000 if max_pages_config == 0 else max_pages_config
        last_page_guess = (
            (start_page + temp_limit_pages - 1)
            if max_pages_config != 0
            else float("inf")
        )
        logger.debug(
            f"Initial plan: Start at page {start_page}, process up to {temp_limit_pages} pages (limit guess: {last_page_guess}).\n"
        )

        # Process page by page (While loop)
        logger.debug("Processing matches page by page...")
        current_page_num = start_page
        while True:
            # Get total pages if unknown
            if total_pages is None:
                logger.debug(
                    f"Fetching matches for page {current_page_num} (will also get total pages)..."
                )
                db_session_for_page = session_manager.get_db_conn()
                if not db_session_for_page:
                    logger.error(
                        f"Could not get DB session for page {current_page_num}. Skipping page."
                    )
                    total_errors += 50
                    time.sleep(2)
                    current_page_num += 1
                    if total_errors > 150:
                        logger.critical(
                            "Aborting run due to persistent DB connection failures."
                        )
                        final_success = False
                        break
                    continue
                try:
                    matches_on_page, fetched_total_pages = get_matches(
                        session_manager, db_session_for_page, current_page_num
                    )
                finally:
                    session_manager.return_session(db_session_for_page)

                if fetched_total_pages is None:
                    logger.warning(
                        f"Failed to retrieve total_pages from get_matches on page {current_page_num}. Retrying once..."
                    )
                    time.sleep(5)
                    db_session_for_retry = session_manager.get_db_conn()
                    if not db_session_for_retry:
                        logger.error(
                            f"Could not get DB session for page {current_page_num} retry. Aborting."
                        )
                        final_success = False
                        break
                    try:
                        matches_on_page, fetched_total_pages = get_matches(
                            session_manager, db_session_for_retry, current_page_num
                        )
                    finally:
                        session_manager.return_session(db_session_for_retry)
                    if fetched_total_pages is None:
                        logger.error(
                            f"Failed to retrieve total_pages after retry. Aborting."
                        )
                        final_success = False
                        break

                total_pages = fetched_total_pages
                logger.info(f"Total pages: {total_pages}\n")
                pages_to_process_config = (
                    min(max_pages_config, total_pages)
                    if max_pages_config != 0
                    else total_pages
                )
                last_page = min(start_page + pages_to_process_config - 1, total_pages)
                total_pages_to_process_in_run = last_page - start_page + 1

                if current_page_num > last_page:
                    logger.info(
                        f"Current page {current_page_num} exceeds actual last page {last_page}. Stopping."
                    )
                    break
                if start_page > last_page:
                    logger.warning(
                        f"Start page ({start_page}) > Last page ({last_page}). No pages."
                    )
                    break
                elif start_page == last_page:
                    logger.debug(
                        f"Processing page {start_page} only (Total: {total_pages}).\n"
                    )
                else:
                    logger.info(
                        f"Processing {total_pages_to_process_in_run} pages from {start_page} to {last_page} (Total pages: {total_pages}).\n"
                    )
            else:
                # Total pages known, check limit
                if last_page is not None and current_page_num > last_page:
                    logger.debug(
                        f"Current page {current_page_num} exceeds processing limit {last_page}. Stopping."
                    )
                    break
                page_log_header = f"====== Processing Page {current_page_num}/{last_page} (Overall: {total_pages}) ======"
                logger.debug(page_log_header)
                logger.debug(f"Fetching matches from page {current_page_num}...")
                db_session_for_page = session_manager.get_db_conn()
                if not db_session_for_page:
                    logger.error(
                        f"Could not get DB session for page {current_page_num}. Skipping page."
                    )
                    total_errors += 50
                    time.sleep(2)
                    current_page_num += 1
                    if total_errors > 150:
                        logger.critical(
                            "Aborting run due to persistent DB connection failures."
                        )
                        final_success = False
                        break
                    continue
                try:
                    matches_on_page, _ = get_matches(
                        session_manager, db_session_for_page, current_page_num
                    )
                finally:
                    session_manager.return_session(db_session_for_page)

            # Check matches_on_page
            if matches_on_page is None or not matches_on_page:
                if total_pages is not None and current_page_num > total_pages:
                    logger.info(
                        f"Reached page {current_page_num} which is beyond total pages ({total_pages}). Stopping."
                    )
                    break
                else:
                    logger.warning(
                        f"No matches/error fetching for page {current_page_num}. Skipping."
                    )
                    time.sleep(2)
                    current_page_num += 1
                    continue

            num_matches_on_page = len(matches_on_page)
            logger.debug(
                f"Found {num_matches_on_page} matches on page {current_page_num}.\n"
            )

            # 15. Process the batch
            page_new, page_updated, page_skipped, page_errors = _do_batch(
                session_manager=session_manager,
                matches_on_page=matches_on_page,
                current_page=current_page_num,
                last_page_in_run=(last_page if last_page is not None else "?"),
                max_labels_to_show=2,
            )

            # Accumulate totals
            total_new += page_new
            total_updated += page_updated
            total_skipped += page_skipped
            total_errors += page_errors
            total_pages_processed += 1


            logger.debug(f"--- After page {current_page_num} accumulation ---")
            logger.debug(
                f"  Page results: New={page_new}, Upd={page_updated}, Skip={page_skipped}, Err={page_errors}"
            )
            logger.debug(
                f"  Running Totals: New={total_new}, Upd={total_updated}, Skip={total_skipped}, Err={total_errors}"
            )
            logger.debug(f"  Total Pages Processed: {total_pages_processed}\n")


            _adjust_delay(session_manager, current_page_num)
            current_page_num += 1

        # --- END PAGE-BY-PAGE PROCESSING ---

        _log_coord_summary(
            total_pages_processed, total_new, total_updated, total_skipped, total_errors
        )

    # ... (Exception handling remains the same) ...
    except KeyboardInterrupt:
        logger.warning(
            "Keyboard interrupt during coord. Attempting graceful shutdown..."
        )
        final_success = False
        _log_coord_summary(
            total_pages_processed, total_new, total_updated, total_skipped, total_errors
        )
        raise
    except Exception as e:
        logger.error(f"Critical error during coord execution: {e}", exc_info=True)
        final_success = False

    return final_success
# end of coord

def _do_batch(
    session_manager,
    matches_on_page,
    current_page,
    last_page_in_run,
    max_labels_to_show: int,
):
    """
    V14.2 REVISED: Processes a batch of matches for a single page.
    - Pre-fetches API data.
    - Calls revised _do_match V14.1 to get prepared data with only changed fields.
    - Performs bulk database operations using simplified update logic.
    - Corrected SyntaxError from previous version.
    """
    # --- Initialization & API Pre-fetching (Same as V14.1) ---
    page_new, page_updated, page_skipped, page_errors = 0, 0, 0, 0
    num_matches = len(matches_on_page)
    my_uuid = session_manager.my_uuid
    my_tree_id = session_manager.my_tree_id
    if not my_uuid: return 0, 0, 0, num_matches
    logger.debug(f"--- Starting Batch Pre-fetch for Page {current_page} ({num_matches} matches) ---")
    # ... (Keep the entire API pre-fetching block from V14.1) ...
    uuids_on_page = [m["uuid"] for m in matches_on_page if m.get("uuid")]
    uuids_for_tree_badge = [m["uuid"] for m in matches_on_page if m.get("uuid") and m.get("in_my_tree")]
    uuids_for_combined_details = uuids_on_page
    uuids_for_relationships = uuids_on_page
    batch_combined_details = {}; batch_badge_data = {}; batch_ladder_data = {}; batch_relationship_prob_data = {}
    futures = {}; fetch_start_time = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        # ... (Executor submits same as V14.1) ...
        for uuid_val in uuids_for_combined_details: delay = session_manager.dynamic_rate_limiter.wait(); future = executor.submit(_fetch_combined_details, session_manager, uuid_val); futures[future] = ("combined_details", uuid_val)
        for uuid_val in uuids_for_tree_badge: delay = session_manager.dynamic_rate_limiter.wait(); future = executor.submit(_fetch_batch_badge_details, session_manager, uuid_val); futures[future] = ("badge_details", uuid_val)
        for uuid_val in uuids_for_relationships: delay = session_manager.dynamic_rate_limiter.wait(); future = executor.submit(_fetch_batch_relationship_prob, session_manager, uuid_val, max_labels_to_show); futures[future] = ("relationship_prob", uuid_val)
        temp_badge_results = {}
        for future in as_completed(futures):
             task_type, identifier = futures[future]
             try:
                  result = future.result()
                  if result is not None:
                       if task_type == "combined_details": batch_combined_details[identifier] = result
                       elif task_type == "badge_details": temp_badge_results[identifier] = result
                       elif task_type == "relationship_prob": batch_relationship_prob_data[identifier] = result
             except Exception as exc: logger.error(f"Exception in pre-fetch task '{task_type}' for {identifier}: {exc}", exc_info=False)
        cfpid_to_uuid_map = {}; ladder_futures = {}
        if my_tree_id:
             for uuid_val, badge_result in temp_badge_results.items():
                  cfpid = badge_result.get("their_cfpid")
                  if cfpid: cfpid_to_uuid_map[cfpid] = uuid_val; delay = session_manager.dynamic_rate_limiter.wait(); future = executor.submit(_fetch_batch_ladder, session_manager, cfpid, my_tree_id); ladder_futures[future] = ("ladder", cfpid)
        else: logger.debug("My tree ID not available, skipping ladder pre-fetch.")
        for future in as_completed(ladder_futures):
             task_type, cfpid = ladder_futures[future]
             try:
                  result = future.result()
                  if result is not None: batch_ladder_data[cfpid] = result
             except Exception as exc: logger.error(f"Exception in pre-fetch task 'ladder' for CFPID {cfpid}: {exc}", exc_info=False)
    fetch_duration = time.time() - fetch_start_time
    logger.debug(f"--- Finished Batch Pre-fetch for Page {current_page}. Duration: {fetch_duration:.2f}s ---")
    # ... (Logging of pre-fetch results same as V14.1) ...
    logger.debug(f"  Combined Details: {len(batch_combined_details)}/{len(uuids_for_combined_details)}")
    logger.debug(f"  Badge Details: {len(temp_badge_results)}/{len(uuids_for_tree_badge)}")
    logger.debug(f"  Relationship Probs: {len(batch_relationship_prob_data)}/{len(uuids_for_relationships)}")
    logger.debug(f"  Ladders: {len(batch_ladder_data)}/{len(cfpid_to_uuid_map)}")
    batch_tree_data = {}
    for uuid_val, badge_result in temp_badge_results.items():
        combined_tree_info = badge_result.copy()
        cfpid = badge_result.get("their_cfpid")
        if cfpid and cfpid in batch_ladder_data: combined_tree_info.update(batch_ladder_data[cfpid])
        batch_tree_data[uuid_val] = combined_tree_info
    # --- END API Pre-fetching block ---

    # --- Process Matches and Collect Data ---
    prepared_bulk_data: List[Dict[str, Any]] = []
    page_statuses: Dict[str, int] = {"new": 0, "updated": 0, "skipped": 0, "error": 0}
    session = session_manager.get_db_conn()
    if not session: return 0, 0, 0, num_matches
    try:
        for match_index, match in enumerate(matches_on_page):
            _case_name = match.get("username", f"Unknown Match {match_index+1}")
            logger.debug(f"#### Page {current_page} - Prep Match {match_index + 1}/{num_matches}: {_case_name} ####")

            # --- CORRECTED SYNTAX ---
            uuid_val = match.get("uuid")
            if not uuid_val:
                logger.warning(f"Skipping prep for match {match_index+1} on page {current_page}: Missing UUID.")
                page_statuses["error"] += 1
                continue
            # --- END CORRECTION ---

            prefetched_combined = batch_combined_details.get(uuid_val)
            prefetched_tree = batch_tree_data.get(uuid_val)
            prefetched_rel_prob = batch_relationship_prob_data.get(uuid_val)
            match["predicted_relationship"] = (prefetched_rel_prob or "N/A (Fetch Failed)")
            try:
                # Call revised _do_match V14.1 to get prepared data (only changed fields)
                prepared_data, status, error_msg = _do_match(
                    session=session, match=match, session_manager=session_manager,
                    prefetched_combined_details=prefetched_combined, prefetched_tree_data=prefetched_tree,
                )
                logger.debug(f"Match {_case_name}: _do_match returned status '{status}'")
                page_statuses[status] += 1
                logger.debug(f"  -> Tallying '{status}'. Count: {page_statuses[status]}")
                if status != "error" and prepared_data:
                    prepared_bulk_data.append(prepared_data) # Add dict with prepared person/dna/tree data
                elif status == "error": logger.error(f"  -> Error preparing data for {_case_name}: {error_msg}")
                logger.debug(f"Finished {_case_name} data preparation.\n")
            except Exception as inner_e:
                logger.error(f"Critical error preparing data for match {_case_name} on page {current_page}: {inner_e}", exc_info=True)
                page_statuses["error"] += 1; logger.debug(f"  -> Tallying 'error' due to exception. Count: {page_statuses['error']}")

        # --- Perform Bulk Database Operations (Same as V14.1, includes simplified updates) ---
        if prepared_bulk_data:
            logger.debug(f"--- Starting Bulk DB Operations for Page {current_page} ({len(prepared_bulk_data)} items) ---")
            bulk_start_time = time.time()
            try:
                # Separate data by type and operation
                person_creates = [d["person"] for d in prepared_bulk_data if d.get("person") and d["person"]["_operation"] == "create"]
                person_updates = [d["person"] for d in prepared_bulk_data if d.get("person") and d["person"]["_operation"] == "update"]
                dna_match_creates = [d["dna_match"] for d in prepared_bulk_data if d.get("dna_match")] # Always create
                family_tree_creates = [d["family_tree"] for d in prepared_bulk_data if d.get("family_tree") and d["family_tree"]["_operation"] == "create"]
                family_tree_updates = [d["family_tree"] for d in prepared_bulk_data if d.get("family_tree") and d["family_tree"]["_operation"] == "update"]

                # --- Bulk Person Create ---
                created_person_map = {}
                if person_creates:
                    # ... (same as V14.1) ...
                    logger.debug(f"Bulk inserting {len(person_creates)} new Person records...")
                    insert_data = [{k: v for k, v in p.items() if not k.startswith('_')} for p in person_creates]
                    session.bulk_insert_mappings(Person, insert_data, return_defaults=True)
                    session.flush()
                    for p_data in insert_data:
                         if p_data.get('id') and p_data.get('uuid'): created_person_map[p_data['uuid']] = p_data['id']
                    logger.debug(f"Bulk inserted {len(created_person_map)} persons.")


                # --- Simplified Bulk Person Update ---
                if person_updates:
                    # ... (same simplified logic as V14.1) ...
                    update_mappings = []
                    for p_data in person_updates:
                         existing_id = p_data.get("_existing_person_id")
                         if not existing_id: logger.warning(f"Skipping person update for UUID {p_data.get('uuid')}: Missing existing ID."); continue
                         update_dict = {k: v for k, v in p_data.items() if not k.startswith('_') and k != 'uuid'}
                         if update_dict:
                              update_dict["id"] = existing_id; update_dict["updated_at"] = datetime.now()
                              update_mappings.append(update_dict)
                         else: logger.debug(f"Skipping person update mapping for ID {existing_id} (no changed fields provided by _do_match).")
                    if update_mappings:
                         logger.debug(f"Bulk updating {len(update_mappings)} existing Person records...")
                         session.bulk_update_mappings(Person, update_mappings); logger.debug("Bulk updated persons.")
                    else: logger.debug("No Person records needed bulk updating this batch.")


                # --- Fetch ALL relevant Person IDs ---
                all_person_ids_map = created_person_map.copy()
                for p_update_data in person_updates:
                     if p_update_data.get("_existing_person_id") and p_update_data.get("uuid"):
                          all_person_ids_map[p_update_data["uuid"]] = p_update_data["_existing_person_id"]


                # --- Bulk DNA Match Create ---
                if dna_match_creates:
                    # ... (same as V14.1) ...
                    dna_insert_data = []
                    for dna_data in dna_match_creates:
                        person_uuid = dna_data.get("uuid"); person_id = all_person_ids_map.get(person_uuid)
                        if person_id:
                             insert_dict = {k: v for k, v in dna_data.items() if not k.startswith('_')}
                             insert_dict["people_id"] = person_id; dna_insert_data.append(insert_dict)
                        else: logger.warning(f"Skipping DNA Match create for UUID {person_uuid}: Corresponding Person ID not found.")
                    if dna_insert_data:
                         logger.debug(f"Bulk inserting {len(dna_insert_data)} DnaMatch records...")
                         session.bulk_insert_mappings(DnaMatch, dna_insert_data); logger.debug("Bulk inserted DnaMatches.")


                # --- Bulk Family Tree Create ---
                if family_tree_creates:
                    # ... (same as V14.1) ...
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


                # --- Simplified Bulk Family Tree Update ---
                if family_tree_updates:
                    # ... (same simplified logic as V14.1) ...
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
                        else: logger.debug(f"Skipping tree update mapping for ID {existing_id} (no changed fields provided by _do_match).")
                    if tree_update_mappings:
                        logger.debug(f"Bulk updating {len(tree_update_mappings)} FamilyTree records...")
                        session.bulk_update_mappings(FamilyTree, tree_update_mappings); logger.debug("Bulk updated FamilyTrees.")
                    else: logger.debug("No FamilyTree records needed bulk updating this batch.")


                # --- Final Commit for the Page ---
                logger.debug(f"Attempting final commit for page {current_page}...")
                session.commit()
                bulk_duration = time.time() - bulk_start_time
                logger.debug(f"Commit successful for page {current_page}. Bulk operations duration: {bulk_duration:.2f}s.")

            # --- Bulk Error Handling ---
            except (IntegrityError, SQLAlchemyError) as bulk_err:
                 logger.error(f"Bulk DB operation FAILED for page {current_page}: {bulk_err}", exc_info=True)
                 if session.is_active: session.rollback()
                 page_statuses["error"] += len(prepared_bulk_data); page_statuses["new"] = 0; page_statuses["updated"] = 0
            except Exception as bulk_e_unexp:
                 logger.critical(f"Unexpected Bulk DB Error for page {current_page}: {bulk_e_unexp}", exc_info=True)
                 if session.is_active: session.rollback()
                 page_statuses["error"] += len(prepared_bulk_data); page_statuses["new"] = 0; page_statuses["updated"] = 0
        else:
            logger.debug(f"No data prepared for bulk DB operations on page {current_page}.")

        # --- Log Page Summary ---
        logger.debug(f"---- Page {current_page}/{last_page_in_run} Summary ----")
        logger.debug(f"  New:     {page_statuses['new']}")
        logger.debug(f"  Updated: {page_statuses['updated']}")
        logger.debug(f"  Skipped: {page_statuses['skipped']}")
        logger.debug(f"  Errors:  {page_statuses['error']}")
        logger.debug("-----------------------\n")

    # --- Outer Error Handling & Finally Block ---
    except Exception as outer_e:
        logger.error(f"Critical error during page {current_page} processing loop: {outer_e}", exc_info=True)
        if session.is_active:
            try: session.rollback(); logger.debug(f"Rolled back session.")
            except Exception as rb_err: logger.error(f"Failed rollback: {rb_err}")
        processed_count = sum(page_statuses.values()); remaining_count = num_matches - processed_count
        page_statuses["error"] += remaining_count
    finally:
        session_manager.return_session(session)

    return page_statuses["new"], page_statuses["updated"], page_statuses["skipped"], page_statuses["error"]
# end of _do_batch (V14.2)

@retry()
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
    V14.3 REVISED: Processes match data, compares with existing DB record (if any),
    and returns prepared data dictionary containing ONLY the fields needing creation or update.
    - Determines 'new', 'updated', or 'skipped' status based on actual changes.
    - Corrected second NameError for existing tree record ID during update prep.
    """
    existing_person: Optional[Person] = None
    dna_match_record: Optional[DnaMatch] = None
    family_tree_record: Optional[FamilyTree] = None # Correct variable for existing tree

    match_uuid = match.get("uuid")
    match_username = match.get("username")
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
        # Step 1: DB Lookup (Loads related objects including family_tree_record)
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
                session.expire(existing_person, ["dna_match", "family_tree"])
            else: logger.debug(f"{log_ref}: No existing person found by UUID.")
        except SQLAlchemyError as db_lookup_err:
            logger.error(f"Initial DB lookup failed for {log_ref_short}: {db_lookup_err}", exc_info=True)
            return None, "error", f"Initial DB lookup failed for {log_ref_short}"
        except Exception as lookup_err:
            logger.error(f"Unexpected error during initial DB lookup for {log_ref_short}: {lookup_err}", exc_info=True)
            return None, "error", f"Unexpected DB lookup error for {log_ref_short}"

        is_new_person = existing_person is None

        # Step 2: Prepare Incoming Data (Same as V14.2)
        # --- Prepare Person Data ---
        details_part = prefetched_combined_details or {}; profile_part = prefetched_combined_details or {}
        tester_profile_id = details_part.get("tester_profile_id") or match.get("profile_id")
        admin_profile_id = details_part.get("admin_profile_id") or match.get("administrator_profile_id_hint")
        admin_username = details_part.get("admin_username") or match.get("administrator_username_hint")
        person_profile_id_to_save, person_admin_id_to_save, person_admin_username_to_save = (None, None, None)
        if admin_profile_id and (not tester_profile_id or admin_profile_id.upper() != tester_profile_id.upper()): person_profile_id_to_save, person_admin_id_to_save, person_admin_username_to_save = (tester_profile_id, admin_profile_id, admin_username)
        elif admin_profile_id and tester_profile_id and admin_profile_id.upper() == tester_profile_id.upper():
            if admin_username and match_username and match_username.lower() != admin_username.lower(): person_profile_id_to_save, person_admin_id_to_save, person_admin_username_to_save = (None, admin_profile_id, admin_username)
            else: person_profile_id_to_save = tester_profile_id
        elif tester_profile_id and not admin_profile_id: person_profile_id_to_save = tester_profile_id
        else: person_admin_id_to_save, person_admin_username_to_save = (admin_profile_id, admin_username)
        message_target_id = person_admin_id_to_save or person_profile_id_to_save; constructed_message_link = None
        if message_target_id and session_manager.my_uuid: constructed_message_link = urljoin(config_instance.BASE_URL, f"/messaging/?p={message_target_id.upper()}&testguid1={session_manager.my_uuid.upper()}&testguid2={match_uuid.upper()}")
        birth_year_val = None
        if prefetched_tree_data and prefetched_tree_data.get("their_birth_year"):
            try: birth_year_val = int(prefetched_tree_data["their_birth_year"])
            except (ValueError, TypeError): pass
        incoming_person_data = {
            "uuid": match_uuid.upper(), "profile_id": (person_profile_id_to_save.upper() if person_profile_id_to_save else None),
            "username": match_username, "administrator_profile_id": (person_admin_id_to_save.upper() if person_admin_id_to_save else None),
            "administrator_username": person_admin_username_to_save, "in_my_tree": match_in_my_tree, "first_name": match.get("first_name"),
            "last_logged_in": profile_part.get("last_logged_in_dt"), "contactable": profile_part.get("contactable", False),
            "gender": details_part.get("gender"), "message_link": constructed_message_link, "birth_year": birth_year_val,
        }
        # --- Prepare DNA Data ---
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
        # --- Prepare Tree Data ---
        incoming_tree_data = None
        should_have_tree = match_in_my_tree and prefetched_tree_data is not None
        if should_have_tree:
            view_in_tree_link, facts_link = None, None
            their_cfpid_final = prefetched_tree_data.get("their_cfpid")
            if their_cfpid_final and session_manager.my_tree_id:
                base_tree_url = urljoin(config_instance.BASE_URL, f"/family-tree/person/tree/{session_manager.my_tree_id}/person/{their_cfpid_final}")
                view_in_tree_link = urljoin(base_tree_url, "family"); facts_link = urljoin(base_tree_url, "facts")
            incoming_tree_data = {
                "uuid": match_uuid.upper(), "cfpid": their_cfpid_final, "person_name_in_tree": prefetched_tree_data.get("their_firstname", "Unknown"),
                "facts_link": facts_link, "view_in_tree_link": view_in_tree_link, "actual_relationship": prefetched_tree_data.get("actual_relationship"),
                "relationship_path": prefetched_tree_data.get("relationship_path"), "_operation": "create" if family_tree_record is None else "update",
                "_existing_tree_id": family_tree_record.id if family_tree_record else None
            }
        elif not match_in_my_tree and family_tree_record is not None: logger.debug(f"{log_ref}: Should not have tree record, but one exists. No tree data prepared.")

        # Step 3: Compare and Build Bulk Data Dictionary
        if is_new_person:
            # --- NEW PERSON --- (Same as V14.2)
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
            # --- Comparisons (Same as V14.2) ---
            new_dt = incoming_person_data.get("last_logged_in"); old_dt = existing_person.last_logged_in; new_naive_ts = None; old_naive_ts = None
            if isinstance(new_dt, datetime): new_naive_ts = new_dt.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)
            if isinstance(old_dt, datetime): old_naive_ts = old_dt.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0) if old_dt.tzinfo else old_dt.replace(microsecond=0)
            if new_naive_ts != old_naive_ts: logger.debug(f"  -> Change detected for last_logged_in: {old_naive_ts} -> {new_naive_ts}"); person_data_for_update["last_logged_in"] = new_dt; person_update_needed = True
            if bool(existing_person.contactable) != bool(incoming_person_data.get("contactable", False)): logger.debug(f"  -> Change detected for contactable"); person_data_for_update["contactable"] = bool(incoming_person_data.get("contactable", False)); person_update_needed = True
            if incoming_person_data.get("birth_year") is not None and existing_person.birth_year is None:
                 try: birth_year_int = int(incoming_person_data["birth_year"]); logger.debug(f"  -> Change detected for birth_year (adding)"); person_data_for_update["birth_year"] = birth_year_int; person_update_needed = True
                 except (ValueError, TypeError): pass
            if bool(existing_person.in_my_tree) != bool(incoming_person_data.get("in_my_tree", False)): logger.debug(f"  -> Change detected for in_my_tree"); person_data_for_update["in_my_tree"] = bool(incoming_person_data.get("in_my_tree", False)); person_update_needed = True
            new_gender = incoming_person_data.get("gender")
            if new_gender is not None and existing_person.gender is None and isinstance(new_gender, str) and new_gender.lower() in ('f','m'): logger.debug(f"  -> Change detected for gender (adding)"); person_data_for_update["gender"] = new_gender.lower(); person_update_needed = True
            new_admin_id = incoming_person_data.get("administrator_profile_id"); new_admin_user = incoming_person_data.get("administrator_username")
            if existing_person.administrator_profile_id != new_admin_id: logger.debug(f"  -> Change detected for administrator_profile_id"); person_data_for_update["administrator_profile_id"] = new_admin_id; person_update_needed = True
            if existing_person.administrator_username != new_admin_user: logger.debug(f"  -> Change detected for administrator_username"); person_data_for_update["administrator_username"] = new_admin_user; person_update_needed = True
            if existing_person.message_link != incoming_person_data.get("message_link"): logger.debug(f"  -> Change detected for message_link"); person_data_for_update["message_link"] = incoming_person_data.get("message_link"); person_update_needed = True
            if existing_person.username != incoming_person_data.get("username"): logger.debug(f"  -> Change detected for username"); person_data_for_update["username"] = incoming_person_data.get("username"); person_update_needed = True

            # --- Prepare Person data only if changes detected ---
            if person_update_needed: prepared_data_for_bulk["person"] = person_data_for_update; logger.debug(f"{log_ref}: Person data prepared for bulk update (changes found).")
            else: logger.debug(f"{log_ref}: No changes detected for Person.")

            # --- Prepare DNA data (only if creating) ---
            if incoming_dna_data: prepared_data_for_bulk["dna_match"] = incoming_dna_data; logger.debug(f"{log_ref}: Preparing data for NEW DnaMatch.")

            # --- Prepare Tree data (check for updates) ---
            if incoming_tree_data and incoming_tree_data["_operation"] == "create":
                 prepared_data_for_bulk["family_tree"] = incoming_tree_data; logger.debug(f"{log_ref}: Preparing data for NEW FamilyTree.")
            elif incoming_tree_data and incoming_tree_data["_operation"] == "update":
                 tree_data_for_update = {
                      "_operation": "update",
                      # --- CORRECTED VARIABLE NAME ---
                      "_existing_tree_id": family_tree_record.id,
                      # --- END CORRECTION ---
                      "uuid": match_uuid.upper()
                 }
                 tree_update_needed = False
                 fields_to_check = ["cfpid", "person_name_in_tree", "facts_link", "view_in_tree_link", "actual_relationship", "relationship_path"]
                 for field in fields_to_check:
                      new_value = incoming_tree_data.get(field)
                      # --- Ensure family_tree_record is not None before accessing attributes ---
                      old_value = getattr(family_tree_record, field, None) if family_tree_record else None
                      # --- End Check ---
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
# End of _do_match (V14.3)

#################################################################################
# 3. API Data Acquisition
#################################################################################

def get_matches(
    session_manager: SessionManager, db_session: Session, current_page: int = 1
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    V13 REVISED: Fetches and processes match list data for a SINGLE page.
    - Returns (match_list, total_pages).
    - Removes predicted relationship fetching (moved to _do_batch).
    - Still fetches in-tree status.
    - Uses requests library via _api_req (Idea 5 applied).
    """
    total_pages: Optional[int] = None  # Initialize total_pages

    if not isinstance(session_manager, SessionManager):
        logger.error("Invalid SessionManager passed to get_matches.")
        return [], None
    if not session_manager.driver or not session_manager.is_sess_valid():
        logger.error("WebDriver session is not valid in get_matches.")
        return [], None
    if not session_manager.my_uuid:
        logger.error("SessionManager my_uuid is not initialized in get_matches.")
        return [], None

    my_uuid = session_manager.my_uuid

    try:
        # --- 1. Fetch Match List Data ---
        match_list_url = urljoin(
            config_instance.BASE_URL,
            f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={current_page}",
        )
        logger.debug(f"Fetching match list for page {current_page} using requests...")
        # Idea 5: _api_req defaults to requests, no force_requests needed
        api_response = _api_req(
            url=match_list_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=True,  # Required for this endpoint
            api_description="Match List API",
            referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
        )

        if api_response is None:
            logger.warning(f"No response from match list API for page {current_page}.")
            return [], None
        if not isinstance(api_response, dict):
            logger.warning(
                f"Unexpected response type from match list API page {current_page}. Type: {type(api_response)}."
            )
            logger.debug(f"Response data: {api_response}")
            return [], None

        # --- Extract total_pages (Idea 1) ---
        total_pages_raw = api_response.get("totalPages")
        if total_pages_raw is not None:
            try:
                total_pages = int(total_pages_raw)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not parse totalPages '{total_pages_raw}' to int."
                )
                total_pages = None  # Mark as unknown if parsing fails
        else:
            logger.warning("totalPages key missing from Match List API response.")
            total_pages = None

        match_data_list = api_response.get("matchList", [])
        if not match_data_list:
            logger.info(f"No matches found in 'matchList' for page {current_page}.")
            return (
                [],
                total_pages,
            )  # Return empty list but potentially known total_pages
        logger.debug(
            f"Got {len(match_data_list)} raw matches from API on page {current_page}."
        )

        valid_matches_for_processing: List[Dict[str, Any]] = []
        skipped_sampleid_count = 0
        for m in match_data_list:
            if isinstance(m, dict) and m.get("sampleId"):
                valid_matches_for_processing.append(m)
            else:
                skipped_sampleid_count += 1
                logger.warning(
                    f"Skipping raw match due to missing 'sampleId' on page {current_page}. Data: {m}"
                )
        if skipped_sampleid_count > 0:
            logger.warning(
                f"Skipped {skipped_sampleid_count} matches on page {current_page} (missing 'sampleId')."
            )
        if not valid_matches_for_processing:
            logger.warning(
                f"No matches with valid 'sampleId' found on page {current_page}."
            )
            return [], total_pages

        sample_ids_on_page = [
            match["sampleId"].upper() for match in valid_matches_for_processing
        ]

        # --- In-Tree Status Check (Remains here, relatively lightweight) ---
        in_tree_ids: Set[str] = set()
        cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids_on_page))}"
        cached_in_tree = session_manager.cache.get(cache_key_tree)
        if cached_in_tree is not None and isinstance(cached_in_tree, set):
            in_tree_ids = cached_in_tree
            logger.debug(
                f"Loaded {len(in_tree_ids)} in-tree IDs from cache for page {current_page}."
            )
        else:
            in_tree_url = urljoin(
                config_instance.BASE_URL,
                f"discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid.upper()}",
            )
            logger.debug(f"Fetching in-tree status for page {current_page}...")
            response_in_tree = _api_req(
                url=in_tree_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="POST",
                json_data={"sampleIds": sample_ids_on_page},
                use_csrf_token=True,
                api_description="In-Tree Status Check",
                referer_url=urljoin(
                    config_instance.BASE_URL, "/discoveryui-matches/list/"
                ),
            )
            if isinstance(response_in_tree, list):
                in_tree_ids = {
                    item.upper() for item in response_in_tree if isinstance(item, str)
                }
                session_manager.cache.set(
                    cache_key_tree, in_tree_ids, timeout=config_instance.CACHE_TIMEOUT
                )
                logger.debug(
                    f"Fetched/cached {len(in_tree_ids)} in-tree IDs for page {current_page}."
                )
            else:
                logger.warning(
                    f"In-Tree Status Check API failed or returned unexpected format for page {current_page}."
                )

        # --- REMOVED: Conditional Predicted Relationship Processing (Moved to _do_batch) ---

        # --- Compile Final Refined Match Data ---
        refined_matches: List[Dict[str, Any]] = []
        skipped_profile_id_count = 0

        for match in valid_matches_for_processing:
            profile = match.get("matchProfile", {})
            relationship = match.get("relationship", {})
            sample_id_upper = match["sampleId"].upper()
            profile_user_id = profile.get("userId")
            match_username = profile.get("displayName", "Unknown").title()
            admin_profile_id_hint = match.get("adminId")
            admin_username_hint = match.get("adminName")

            if not profile_user_id:
                logger.debug(
                    f"Match '{match_username}' (UUID: {sample_id_upper}) missing tester 'profile_id'."
                )
                skipped_profile_id_count += 1
                profile_user_id_upper = None
            else:
                profile_user_id_upper = str(profile_user_id).upper()

            compare_link = urljoin(
                config_instance.BASE_URL,
                f"discoveryui-matches/compare/{my_uuid.upper()}/with/{sample_id_upper}",
            )
            first_name = (
                match_username.split()[0] if match_username != "Unknown" else None
            )

            # Predicted relationship will be added later in _do_batch
            refined_match_data = {
                "username": match_username,
                "first_name": first_name,
                "initials": profile.get("displayInitials", "??").upper(),
                "gender": match.get("gender"),
                "profile_id": profile_user_id_upper,  # Store hint
                "uuid": sample_id_upper,
                "administrator_profile_id_hint": admin_profile_id_hint,
                "administrator_username_hint": admin_username_hint,
                "photoUrl": profile.get("photoUrl", ""),
                "cM_DNA": int(relationship.get("sharedCentimorgans", 0)),
                "numSharedSegments": int(relationship.get("numSharedSegments", 0)),
                "compare_link": compare_link,
                "message_link": None,  # Set later in _do_match
                # "predicted_relationship": "N/A", # Placeholder - Set in _do_batch
                "in_my_tree": sample_id_upper in in_tree_ids,
                "createdDate": match.get("createdDate"),
            }
            refined_matches.append(refined_match_data)

        logger.debug(
            f"Processed page {current_page}: Raw={len(match_data_list)}, Refined={len(refined_matches)}"
        )
        # Removed logging of individual matches here for brevity

        # --- Return refined matches AND total_pages ---
        return refined_matches, total_pages

    except requests.exceptions.RequestException as e:
        logger.error(
            f"Network/Request error processing page {current_page}: {e}", exc_info=True
        )
        return [], None  # Return None for total_pages on error
    except Exception as e:
        logger.critical(
            f"Critical error processing match data for page {current_page}: {e}",
            exc_info=True,
        )
        return [], None  # Return None for total_pages on error
# end get_matches

@retry_api(max_retries=3, initial_delay=1, backoff_factor=2)  # Use decorator
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

@retry_api(max_retries=3, initial_delay=1, backoff_factor=2)
def _fetch_batch_badge_details(
    session_manager: SessionManager, match_uuid: str
) -> Optional[Dict[str, Any]]:
    """Fetches /badgedetails for a single match UUID."""
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
            full_firstname = person_badged.get("firstName", "Unknown")
            words = full_firstname.strip().split()
            their_firstname = words[0] if words else "Unknown"

            return {  # Return only the needed fields
                "their_cfpid": person_badged.get("personId"),
                "their_firstname": their_firstname,
                "their_lastname": person_badged.get("lastName", "Unknown"),
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

@retry_api(
    max_retries=3, initial_delay=1.5, backoff_factor=2
)  # Slightly longer delay maybe
def _fetch_batch_ladder(
    session_manager: SessionManager, cfpid: str, tree_id: str
) -> Optional[Dict[str, Any]]:
    """Fetches and parses /getladder for a single CFPID."""
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
        # This API returns JSONP (text), force text parsing in _api_req
        response_text = _api_req(
            url=ladder_api_url,
            driver=session_manager.driver,  # For UBE
            session_manager=session_manager,
            method="GET",
            headers={"Accept": "*/*"},
            use_csrf_token=False,
            api_description="Get Ladder API (Batch)",  # Key to ensure text parsing if needed
            referer_url=dynamic_referer,
        )

        if response_text and isinstance(response_text, str):
            match_jsonp = re.match(
                r"^[^(]*\((.*)\)[^)]*$", response_text, re.DOTALL | re.IGNORECASE
            )
            if match_jsonp:
                json_string = match_jsonp.group(1).strip()
                try:
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
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSONP content for cfpid {cfpid}.")
            else:
                logger.error(f"Could not parse JSONP format for cfpid {cfpid}.")
        else:
            logger.warning(
                f"No/invalid response text from Get Ladder API for cfpid {cfpid}."
            )

        return None  # Return None if parsing failed

    except Exception as e:
        logger.error(
            f"Error fetching/parsing ladder for CFPID {cfpid}: {e}", exc_info=True
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None
# end _fetch_batch_ladder

@retry_api(max_retries=3, initial_delay=1, backoff_factor=2)
def _fetch_batch_relationship_prob(
    session_manager: SessionManager, match_uuid: str, max_labels_param: int
) -> Optional[str]:
    """
    V13.3 FIXED: Fetches and processes /matchProbabilityData for a single match UUID.
    - Uses the passed max_labels_param parameter correctly.
    - Returns relationship string.
    """
    # Removed global MAX_LABELS_TO_SHOW access - use the parameter

    if not session_manager.my_uuid or not match_uuid:
        logger.warning(
            "Missing my_uuid or match_uuid for relationship probability fetch."
        )
        return "N/A (Error - Missing IDs)"

    my_uuid_upper = session_manager.my_uuid.upper()
    sample_id_upper = match_uuid.upper()
    rel_url = urljoin(
        config_instance.BASE_URL,
        f"discoveryui-matches/parents/list/api/matchProbabilityData/{my_uuid_upper}/{sample_id_upper}",
    )
    try:
        response_rel = _api_req(
            url=rel_url,
            driver=session_manager.driver,  # For UBE
            session_manager=session_manager,
            method="POST",
            json_data={},
            use_csrf_token=True,
            api_description="Match Probability API (Batch)",
            referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
        )
        if (
            not response_rel
            or not isinstance(response_rel, dict)
            or "matchProbabilityToSampleId" not in response_rel
        ):
            logger.warning(
                f"Invalid data format from Match Probability API for {sample_id_upper}. Resp: {response_rel}"
            )
            return "N/A (Invalid Data)"

        prob_data = response_rel["matchProbabilityToSampleId"]
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

        # --- V13.3 FIX: Use the passed parameter ---
        final_labels = labels[:max_labels_param]
        # --- END FIX ---
        relationship_str = " or ".join(map(str, final_labels))
        return f"{relationship_str} [{top_prob:.1f}%]"

    except Exception as e:
        # Log less verbosely for common errors, but include type
        log_level = (
            logging.ERROR
            if not isinstance(e, (requests.exceptions.RequestException, HTTPError))
            else logging.WARNING
        )
        logger.log(
            log_level,
            f"Error fetching relationship probability for {sample_id_upper}: {type(e).__name__} - {e}",
            exc_info=False,
        )
        # Debug log full traceback if needed
        # logger.debug(f"Traceback for relationship prob error:", exc_info=True)

        # Re-raise only retryable exceptions for the decorator
        if isinstance(e, (requests.exceptions.RequestException, HTTPError)):
            raise
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
):  # Renamed arg
    """Logs the final summary of the coord's execution."""
    logger.info("---- Final Summary ----")
    # --- MODIFIED Label ---
    logger.info(f"  Total Pages Processed: {total_pages_processed}")
    # --- END MODIFICATION ---
    logger.info(f"  Total New:       {total_new}")
    logger.info(f"  Total Updated:   {total_updated}")
    logger.info(f"  Total Skipped:   {total_skipped}")
    logger.info(f"  Total Errors:    {total_errors}")
    logger.info("-----------------------\n")
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
