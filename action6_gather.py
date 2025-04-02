#!/usr/bin/env python3

# action6_gather.py

# Standard library imports (alphabetical)
import json
import logging
import math
import random
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

# Third-party imports (alphabetical by package)
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup, Tag
from requests.exceptions import RequestException
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException
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
    find_existing_person
)
from my_selectors import (
    MATCH_ENTRY_SELECTOR)
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
    format_name   
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
    Gathers DNA matches and saves data using the API. Processes page-by-page.
    Includes final DB count check removed.
    """
    # Ensure session and driver are valid
    driver = session_manager.driver
    if not driver or not session_manager.session_active:
        logger.error("WebDriver not initialized or session not active. Exiting coord.")
        return False

    # Initialise counts
    total_new, total_updated, total_skipped, total_errors = 0, 0, 0, 0
    total_pages_processed = 0  # Track pages actually processed
    my_uuid = session_manager.my_uuid

    if not my_uuid:
        logger.error("Failed to retrieve my_uuid from session_manager. Exiting coord.")
        return False

    # Define the target URL base using the user's UUID
    target_matches_url_base = urljoin(
        config_instance.BASE_URL, f"discoveryui-matches/list/{my_uuid}"
    )
    final_success = True  # Track overall success, default to True

    try:
        # 11. Ensure we are on the DNA Match Page using nav_to_list() if needed
        logger.debug("11. Ensure we are on the DNA matches page...")
        try:
            current_url = driver.current_url
            if not current_url.startswith(target_matches_url_base):
                logger.debug("Navigating to DNA matches page.")
                if not nav_to_list(session_manager):
                    logger.error(
                        "Failed to navigate to DNA match list page via nav_to_list(). Exiting coord."
                    )
                    return False
                else:
                    current_url_after_nav = driver.current_url  # Recheck URL after nav
                    if not current_url_after_nav.startswith(target_matches_url_base):
                        logger.error(
                            f"nav_to_list reported success but ended on unexpected URL:\n {current_url_after_nav}. Exiting coord."
                        )
                        return False
                    logger.debug("Successfully navigated to DNA matches page.\n")
            else:
                logger.debug(
                    f"Already on correct DNA matches page:\n ({current_url}).\n"
                )
        except WebDriverException as nav_e:
            logger.error(
                f"WebDriver error checking/navigating to matches page: {nav_e}",
                exc_info=True,
            )
            return False  # Exit if we can't even check the URL

        # 12. Get page count from api
        logger.debug("12. Getting page count...")
        total_pages = get_page_count(session_manager, my_uuid)
        if total_pages is None:
            # Attempt retry if page count fetch fails initially
            logger.warning("Failed to retrieve page count from API. Retrying once...")
            time.sleep(5) # Wait before retrying
            total_pages = get_page_count(session_manager, my_uuid)
            if total_pages is None:
                 logger.error("Failed to retrieve page count after retry. Exiting coord.")
                 return False
            else:
                 logger.info(f"Successfully retrieved page count on retry: {total_pages}\n")
        else:
            logger.info(f"Page count: {total_pages}\n")


        # 13. Determine page range to process
        logger.debug("13. Determining page range to collect...")
        max_pages_config = config_instance.MAX_PAGES
        pages_to_process_config = (
            min(max_pages_config, total_pages) if max_pages_config != 0 else total_pages
        )
        start_page = min(start, total_pages)  # Ensure start page isn't > total_pages
        start_page = max(1, start_page)  # Ensure start_page is at least 1
        last_page = min(start_page + pages_to_process_config - 1, total_pages)
        total_pages_to_process_in_run = last_page - start_page + 1 # Calculate actual number of pages in this run

        if start_page > last_page:
            logger.warning(
                f"Calculated start_page ({start_page}) is greater than last_page ({last_page}). No pages to process."
            )
            return True  # Nothing to process is not an error

        # Logging the determined range
        if start_page == last_page == 1:
            logger.debug("Processing first page only.\n")
        elif start_page == last_page != 1:
            logger.debug(f"Processing page {start_page} only.\n")
        else:
            logger.debug(
                f"Processing {total_pages_to_process_in_run} pages from {start_page} to {last_page} (Total pages: {total_pages}).\n"
            )

        # --- MODIFIED: Process page by page ---
        logger.debug("Processing matches page by page...")

        for current_page_num in range(start_page, last_page + 1):
            logger.info(
                f"====== Processing Page {current_page_num}/{last_page} (Overall pages: {total_pages}) ======"
            )

            # 14. Fetch matches for the current page
            logger.debug(f"Fetching matches from page {current_page_num}...")
            matches_on_page = get_matches(session_manager, current_page_num)

            if matches_on_page is None or not matches_on_page:
                # Handle failure to get matches for a page
                logger.warning(
                    f"No matches found or error fetching matches for page {current_page_num}. Skipping page."
                )
                # Optionally increase total_errors if skipping a page is considered an error
                # total_errors += 1 # Decide if this constitutes an error
                time.sleep(2) # Small pause if a page fetch fails
                continue # Move to the next page

            num_matches_on_page = len(matches_on_page)
            logger.info(f"Found {num_matches_on_page} matches on page {current_page_num}.\n")

            # 15. Process the batch of matches for the current page
            # Pass current page number and total pages for logging context
            page_new, page_updated, page_skipped, page_errors = _do_batch(
                session_manager, matches_on_page, current_page_num, last_page
            )

            # Accumulate totals
            total_new += page_new
            total_updated += page_updated
            total_skipped += page_skipped
            total_errors += page_errors
            total_pages_processed += 1  # Increment count of pages actually processed

            # Optional: Adjust delay between pages if needed
            _adjust_delay(session_manager, current_page_num)

        # --- END PAGE-BY-PAGE PROCESSING ---

        # Log final summary
        _log_coord_summary(
            total_pages_processed, total_new, total_updated, total_skipped, total_errors
        )

        # --- REMOVED: Final DB count verification block ---

    except KeyboardInterrupt:
         logger.warning("Keyboard interrupt detected during coord execution. Attempting graceful shutdown...")
         final_success = False # Mark as failure due to interruption
         # Perform minimal cleanup if possible (closing session is handled by main finally block)
         _log_coord_summary(total_pages_processed, total_new, total_updated, total_skipped, total_errors)
         # Re-raise to allow main loop to catch it
         raise

    except Exception as e:
        logger.error(f"Critical error during coord execution: {e}", exc_info=True)
        final_success = False  # Indicate failure on major exception

    return final_success
# end of coord

def _do_batch(session_manager, matches_on_page, current_page, total_pages_to_process):
    """Processes a batch of matches for a single page, handling relationships and details."""

    # Initialise batch (page) counts
    page_new, page_updated, page_skipped, page_errors = 0, 0, 0, 0
    num_matches = len(matches_on_page)

    # Use a single session for the entire batch (page) for efficiency
    session = session_manager.get_db_conn()
    if not session:
        logger.error(
            f"Failed to get database session for page {current_page} processing."
        )
        # Count all matches on the page as errors if session fails
        return 0, 0, 0, num_matches

    try:
        for match_index, match in enumerate(matches_on_page):
            _case_name = match.get(
                "username", f"Unknown Match (Index {match_index} on Page {current_page})"
            )
            # --- MODIFIED Log Format ---
            logger.debug(f"#### Page {current_page} - Match {match_index + 1}/{num_matches}: {_case_name} ####")
            # --- END MODIFICATION ---

            try:
                # Process the single match using the shared session
                result, _ = _do_match(session, match, session_manager=session_manager)
                logger.debug(f"Finished {_case_name} ({result}).\n")

                # Tally results count
                if result == "new":
                    page_new += 1
                elif result == "updated":
                    page_updated += 1
                elif result == "skipped":
                    page_skipped += 1
                elif result == "error":
                    page_errors += 1

            except Exception as inner_e:  # Catch errors within the loop for a single match
                logger.error(
                    f"Critical error processing match {_case_name} on page {current_page}: {inner_e}",
                    exc_info=True,
                )
                page_errors += 1

        # --- Page Summary Logging ---
        # --- MODIFIED Log Format ---
        logger.debug(f"---- Page {current_page}/{total_pages_to_process} Summary ----")
        logger.debug(f"  New:     {page_new}")
        logger.debug(f"  Updated: {page_updated}")
        logger.debug(f"  Skipped: {page_skipped}")
        logger.debug(f"  Errors:  {page_errors}")
        logger.debug("-----------------------\n")
        # --- END MODIFICATION ---

    except Exception as outer_e:  # Catch errors affecting the whole page loop
        logger.error(
            f"Critical error during page {current_page} processing loop: {outer_e}",
            exc_info=True,
        )
        if session.is_active:
            try:
                 session.rollback()
                 logger.debug(f"Rolled back session due to error on page {current_page} loop.")
            except Exception as rb_err:
                 logger.error(f"Failed to rollback session after error on page {current_page}: {rb_err}")
        # Count remaining matches on the page as errors
        remaining_count = num_matches - (page_new + page_updated + page_skipped + page_errors)
        page_errors += remaining_count

    finally:
        # Always return the session to the pool
        session_manager.return_session(session)

    return page_new, page_updated, page_skipped, page_errors
# end of _do_batch


@retry()
def _do_match(
    session: Session, match: Dict[str, Any], session_manager: SessionManager
) -> Tuple[Literal["skipped", "updated", "new", "error"], Optional[str]]:
    """
    Processes a single match. Fetches details, determines profile/admin IDs, calls
    create_or_update_person (V8 - which now handles need_update logic),
    and conditionally creates DNA/Tree records based on its return values.
    Performs post-commit integrity checks.
    """
    person_record: Optional[Person] = None
    person_id_for_verification: Optional[int] = None

    # --- Get initial data for logging and processing ---
    match_uuid = match.get("uuid")
    match_username = match.get("username")
    initial_match_profile_id_hint = match.get("profile_id") # Might be None
    log_ref = (
        f"UUID={match_uuid}" if match_uuid else f"InitialProfileID={initial_match_profile_id_hint}" if initial_match_profile_id_hint else "Unknown Match"
    )
    if not match_username:
        match_username = "Unknown"
        logger.warning(f"{log_ref.split(' User=')[0]}: Missing 'username' in input match data. Using 'Unknown'.")

    log_ref += f" User='{match_username}'"

    if not match_uuid:
        error_msg = f"Pre-check failed: Missing 'uuid' in match data: {match}"
        logger.error(error_msg)
        return "error", error_msg

    # --- Initialize status variables ---
    overall_status: Literal["new", "updated", "skipped", "error"] = "error" # Default status
    # --- REMOVED person_status initialization here, determined by create_or_update_person V8 ---
    dna_status: Literal["created", "skipped", "error"] = "skipped" # Default, set by create_dna_match
    tree_status: Literal["created", "skipped", "error"] = "skipped" # Default, set by create_family_tree
    # --- Flags now determined by create_or_update_person ---
    create_dna_needed: bool = False
    fetch_tree_data: bool = False

    # --- Initialize data variables ---
    tree_data_to_process: Optional[Dict[str, Any]] = None
    dna_data_to_save: Optional[Dict[str, Any]] = None # Initialize
    family_tree_args: Optional[Dict[str, Any]] = None # Initialize
    details_fetched: Optional[Dict[str, Any]] = None # Initialize

    try:
        # --- 1. Fetch Detailed Match Details (Mandatory) ---
        log_ref_short = f"UUID={match_uuid} User='{match_username}'" # Shorter ref for some logs
        delay = session_manager.dynamic_rate_limiter.wait()
        logger.debug(f"Waited {delay:.2f}s before fetching details for {log_ref_short}")
        logger.debug(f"Fetching /details and profile APIs for {log_ref_short} to determine correct profile/admin IDs.")
        details_fetched = _get_match_details_and_admin(session_manager, match_uuid)
        if not details_fetched:
             logger.error(f"CRITICAL: Failed to fetch /details and/or profile API for {log_ref_short}. Aborting match processing.")
             return "error", f"Details API fetch failed for {log_ref_short}"

        # --- 2. Prepare data for Person Create/Update (using V8 logic in create_or_update_person) ---
        # Determine management status and IDs to save (Logic remains same as V7)
        tester_profile_id = details_fetched.get("tester_profile_id")
        admin_profile_id = details_fetched.get("admin_profile_id")
        admin_username = details_fetched.get("admin_username")

        person_profile_id_to_save = None
        person_admin_id_to_save = None
        person_admin_username_to_save = None

        if admin_profile_id and (not tester_profile_id or admin_profile_id.upper() != tester_profile_id.upper()):
            person_profile_id_to_save = tester_profile_id
            person_admin_id_to_save = admin_profile_id
            person_admin_username_to_save = admin_username
        elif admin_profile_id and tester_profile_id and admin_profile_id.upper() == tester_profile_id.upper():
            if admin_username and match_username.lower() != admin_username.lower():
                 person_profile_id_to_save = None # Managed kit despite matching IDs
                 person_admin_id_to_save = admin_profile_id
                 person_admin_username_to_save = admin_username
            else: # Self-managed
                 person_profile_id_to_save = tester_profile_id
                 person_admin_id_to_save = None
                 person_admin_username_to_save = None
        elif tester_profile_id and not admin_profile_id: # Self-managed
            person_profile_id_to_save = tester_profile_id
            person_admin_id_to_save = None
            person_admin_username_to_save = None
        else: # Fallback
            person_profile_id_to_save = None
            person_admin_id_to_save = admin_profile_id
            person_admin_username_to_save = admin_username

        # Construct message link (Logic remains same as V7)
        message_target_id = admin_profile_id or tester_profile_id
        constructed_message_link = None
        if message_target_id and session_manager.my_uuid:
             constructed_message_link = urljoin(config_instance.BASE_URL, f"/messaging/?p={message_target_id.upper()}&testguid1={session_manager.my_uuid.upper()}&testguid2={match_uuid.upper()}")

        # Consolidate data for save
        # NOTE: create_or_update_person_v8 will handle filtering the updates.
        # We DO NOT fetch birth_year here anymore, it's passed only if fetched during tree processing later.
        person_data_to_save = {
            "uuid": match_uuid.upper(),
            "username": match_username,
            "profile_id": person_profile_id_to_save.upper() if person_profile_id_to_save else None,
            "administrator_profile_id": person_admin_id_to_save.upper() if person_admin_id_to_save else None,
            "administrator_username": person_admin_username_to_save,
            "in_my_tree": match.get("in_my_tree", False), # Pass the initial flag from match list
            "first_name": match.get("first_name"),
            "last_logged_in": details_fetched.get("last_logged_in_dt"), # Aware UTC datetime or None
            "contactable": details_fetched.get("contactable", False),
            # "birth_year": REMOVED - Only updated if tree data is fetched later
            "gender": details_fetched.get("gender"),
            "message_link": constructed_message_link,
        }
        logger.debug(f"Data prepared for Person save/update V8: {person_data_to_save}")

        # --- 3. Create or Update Person record (V8) & Get Related Data Needs ---
        person_status: Literal["created", "updated", "skipped", "error"] # Declare type here
        try:
            # Call the revised function V8 - unpack all 4 return values
            (
                person_record,
                person_status,
                create_dna_needed, # Flag determined by V8 function
                fetch_tree_data, # Flag determined by V8 function
            ) = create_or_update_person(session, person_data_to_save)

            if person_record is None or person_status == "error":
                 logger.error(f"Person create/update V8 returned error for {log_ref_short}.")
                 if session.is_active: session.rollback()
                 return "error", f"Person create/update V8 failed for {log_ref_short}"
            person_id_for_verification = person_record.id
            logger.debug(f"Person processed (Status: {person_status}). DNA Needed: {create_dna_needed}, Tree Fetch Needed: {fetch_tree_data}")

        except Exception as p_err:
             logger.error(f"Unexpected error during person create/update V8 for {log_ref_short}: {p_err}", exc_info=True)
             if session.is_active: session.rollback()
             return "error", f"Unexpected person processing error V8 for {log_ref_short}"

        if person_record is None or person_id_for_verification is None:
             error_msg = f"Person object or ID invalid after create/update V8 for {log_ref_short}. Aborting."
             logger.error(error_msg)
             if session.is_active: session.rollback()
             return "error", error_msg

        # --- 4. Fetch Tree Data (if needed, determined by create_or_update_person) ---
        if fetch_tree_data:
            delay = session_manager.dynamic_rate_limiter.wait()
            logger.debug(f"Waited {delay:.2f}s before fetching tree badge details for {log_ref_short}")
            logger.debug(f"Fetching family tree data for {log_ref_short} (needed: {fetch_tree_data}).")
            tree_api_data = _get_tree(session, match, session_manager) # match has uuid/username needed
            if tree_api_data:
                tree_data_to_process = {
                    "person_name_in_tree": tree_api_data.get("their_firstname", "Unknown"),
                    "their_cfpid": tree_api_data.get("their_cfpid"),
                    "their_birth_year": tree_api_data.get("their_birth_year"), # Get birth year here
                }
                their_cfpid = tree_data_to_process.get("their_cfpid")
                if their_cfpid and session_manager.my_tree_id:
                     delay = session_manager.dynamic_rate_limiter.wait()
                     logger.debug(f"Waited {delay:.2f}s before fetching relationship ladder for {log_ref_short}")
                     relationship_data = _get_relShip(session_manager, session_manager.my_tree_id, their_cfpid)
                     base_tree_url = urljoin(config_instance.BASE_URL, f"/family-tree/person/tree/{session_manager.my_tree_id}/person/{their_cfpid}")
                     tree_data_to_process["view_in_tree_link"] = urljoin(base_tree_url, "family")
                     tree_data_to_process["facts_link"] = urljoin(base_tree_url, "facts")
                     if relationship_data:
                         tree_data_to_process["actual_relationship"] = relationship_data.get("actual_relationship")
                         tree_data_to_process["relationship_path"] = relationship_data.get("relationship_path")
                     else: logger.warning(f"Failed to get relationship details for CFPID: {their_cfpid}.")

                     # --- Potentially update Person birth_year if fetched and needed ---
                     fetched_birth_year = tree_data_to_process.get("their_birth_year")
                     current_person_birth_year = getattr(person_record, "birth_year", None) # Get current from DB object
                     if fetched_birth_year is not None and current_person_birth_year is None:
                          try:
                              birth_year_int = int(fetched_birth_year)
                              logger.debug(f"  Updating Person birth_year from Tree data: '{current_person_birth_year}' -> '{birth_year_int}'")
                              setattr(person_record, "birth_year", birth_year_int)
                              # If person status was 'skipped', promote it to 'updated' because we changed birth year
                              if person_status == "skipped":
                                   person_status = "updated"
                                   logger.debug("  Person status promoted to 'updated' due to birth year addition.")
                          except (ValueError, TypeError):
                              logger.warning(f"  Skipping birth_year update from Tree: New value '{fetched_birth_year}' is not valid integer.")
                     # --- End birth year update ---

                elif not their_cfpid: logger.warning(f"CFPID missing for {log_ref_short}. Cannot fetch relationship or build tree links.")
                elif not session_manager.my_tree_id: logger.warning("my_tree_id missing. Cannot construct tree links or fetch relationship.")
            else:
                logger.warning(f"Failed to fetch tree data (_get_tree returned None) for {log_ref_short}. Resetting fetch_tree_data.")
                fetch_tree_data = False # Reset flag if fetch failed

        # --- 5. Create DNA Match Record (if needed, determined by create_or_update_person) ---
        dna_status = "skipped"
        if create_dna_needed:
            logger.debug(f"Proceeding with DNA Match creation for {log_ref_short} (needed: {create_dna_needed}).")
            pred_rel = match.get("predicted_relationship", "N/A")
            dna_data_to_save = {
                "people_id": person_record.id,
                "compare_link": match.get("compare_link"),
                "cM_DNA": match.get("cM_DNA"),
                "predicted_relationship": pred_rel,
                "uuid": match_uuid.upper(), # For logging in create_dna_match
                "shared_segments": details_fetched.get("shared_segments"),
                "longest_shared_segment": details_fetched.get("longest_shared_segment"),
                "meiosis": details_fetched.get("meiosis"),
                "from_my_fathers_side": details_fetched.get("from_my_fathers_side", False),
                "from_my_mothers_side": details_fetched.get("from_my_mothers_side", False),
            }
            dna_status = create_dna_match(session, dna_data_to_save)
            if dna_status == "error": logger.error(f"Failed to create DNA match for {log_ref_short}.")
            elif dna_status == "skipped": logger.warning(f"DNA match creation needed but skipped for {log_ref_short}.")
        else:
            logger.debug(f"Skipping DNA Match creation for {log_ref_short} (create_dna_needed=False).")

        # --- 6. Create Family Tree Record (if needed AND tree data was successfully fetched) ---
        tree_status = "skipped"
        # Condition: Tree fetch was attempted (fetch_tree_data was True) AND succeeded AND resulted in data
        if fetch_tree_data and tree_data_to_process:
             logger.debug(f"Processing fetched family tree data for {log_ref_short} (Person ID: {person_record.id}).")
             # Prepare args using the fetched tree_data_to_process
             family_tree_args = {
                 "people_id": person_record.id,
                 "cfpid": tree_data_to_process.get("their_cfpid"),
                 "person_name_in_tree": tree_data_to_process.get("person_name_in_tree"),
                 "facts_link": tree_data_to_process.get("facts_link"),
                 "view_in_tree_link": tree_data_to_process.get("view_in_tree_link"),
                 "actual_relationship": tree_data_to_process.get("actual_relationship"),
                 "relationship_path": tree_data_to_process.get("relationship_path")
             }
             logger.debug(f"Arguments prepared for create_family_tree: {family_tree_args}")
             tree_status = create_family_tree(session, family_tree_args)
             if tree_status == "error": logger.error(f"Failed to process family tree for {log_ref_short}.")
             elif tree_status == "skipped": logger.warning(f"Family tree processing needed but skipped for {log_ref_short}.")
        elif fetch_tree_data and not tree_data_to_process:
             logger.warning(f"Skipping Family Tree creation for {log_ref_short} as initial fetch failed or yielded no data.")
        else: # fetch_tree_data was False initially
             logger.debug(f"Skipping Family Tree fetch/creation for {log_ref_short} (fetch_tree_data=False).")

        # --- 7. Determine Overall Status and Commit/Rollback ---
        # Use the potentially updated person_status (from birth year update)
        any_errors = (person_status == "error" or dna_status == "error" or tree_status == "error")
        if any_errors:
            overall_status = "error"
            error_msg = f"Processing error for {log_ref_short} (P:{person_status}, D:{dna_status}, T:{tree_status}). Rolling back."
            logger.warning(error_msg)
            if session.is_active: session.rollback()
            return overall_status, error_msg

        # Check if session has pending changes (new, dirty, deleted objects)
        needs_commit = session.new or session.dirty or session.deleted
        if not needs_commit:
            logger.debug(f"Session is clean for {log_ref_short}. Final status determination based on function returns.")
            # If person was created/updated, session should be dirty unless flush/commit happened internally
            if person_status == "created":
                 logger.warning(f"Person status was 'created' but session clean for {log_ref_short}? Setting overall to 'new'.")
                 overall_status = "new"
                 needs_commit = True # Assume commit is needed if status is created
            elif person_status == "updated":
                 logger.warning(f"Person status was 'updated' but session clean for {log_ref_short}? Setting overall to 'updated'.")
                 overall_status = "updated"
                 needs_commit = True # Assume commit is needed if status is updated
            else: # person_status is 'skipped'
                 overall_status = "skipped"
                 logger.debug(f"No DB changes detected and person status is '{person_status}'. Overall status: '{overall_status}' for {log_ref_short}.")
                 return overall_status, None # No commit needed, return skipped

        # Proceed with commit if needed
        if needs_commit:
            try:
                commit_log_ref = f"{log_ref_short} (Person ID: {person_id_for_verification})"
                logger.debug(f"Attempting commit for {commit_log_ref} due to detected session changes or explicit status.")
                session.commit()
                logger.debug(f"Commit successful for {commit_log_ref}.")

                # Determine final status *after* successful commit based on primary function returns
                if person_status == "created":
                    overall_status = "new"
                elif person_status == "updated" or dna_status == "created" or tree_status == "created":
                    # If person was updated OR if DNA/Tree was newly created, overall is 'updated'
                    overall_status = "updated"
                else: # person_status == 'skipped' and dna/tree were also skipped
                    overall_status = "skipped"
                    logger.debug(f"Commit occurred for {commit_log_ref} but final status determination is 'skipped' (P:'{person_status}', D:'{dna_status}', T:'{tree_status}').")

            except (IntegrityError, SQLAlchemyError) as commit_e:
                error_msg = f"Database Commit FAILED for {log_ref_short}"
                logger.error(f"{error_msg}: {commit_e}", exc_info=True)
                if session.is_active: session.rollback()
                dna_data_log = dna_data_to_save if dna_data_to_save else 'N/A'
                tree_args_log = family_tree_args if family_tree_args else tree_data_to_process
                logger.error(f"Data state before failed commit: Person={person_data_to_save}, DNA={dna_data_log}, Tree={tree_args_log}")
                return "error", error_msg
            except Exception as E:
                error_msg = f"Unexpected commit error for {log_ref_short}"
                logger.error(f"{error_msg}: {E}", exc_info=True)
                if session.is_active: session.rollback()
                return "error", error_msg

        # --- 8. Post-commit Verification (Integrity Checks) ---
        # Logic remains the same as V7, using person_status and other statuses
        if person_id_for_verification and (person_status == "created" or person_status == "updated"):
            try:
                logger.debug(f"Performing post-commit verification for {log_ref_short} (Person ID: {person_id_for_verification}).")
                session.expire(person_record) # Expire first to force reload
                session.refresh(person_record, ['dna_match', 'family_tree']) # Refresh relationships too
                verify_person = person_record # Use the refreshed object

                if not verify_person:
                    logger.error(f"CRITICAL Post-commit verification FAILED for Person ID {person_id_for_verification}! Record MISSING after refresh.")
                    overall_status = "error"
                else:
                    logger.debug(f"Post-commit Person verification OK for {log_ref_short}.")

                    # Check DNA integrity if DNA was expected to be created
                    if dna_status == "created" and dna_data_to_save:
                         verify_dna = verify_person.dna_match
                         if not verify_dna:
                              logger.error(f"Post-commit DNA FAILED: No DNA record found for {log_ref_short} (expected created).")
                         elif verify_dna.cM_DNA != dna_data_to_save.get("cM_DNA"):
                              logger.error(f"Post-commit DNA FAILED: cM mismatch! DB={verify_dna.cM_DNA}, Expected={dna_data_to_save.get('cM_DNA')} for {log_ref_short}.")
                         else: logger.debug(f"Post-commit DNA verification OK for {log_ref_short}.")

                    # Check Tree integrity if Tree was expected to be created OR if person is marked 'in_my_tree'
                    if verify_person.in_my_tree:
                         verify_tree = verify_person.family_tree
                         if not verify_tree:
                              logger.warning(f"Post-commit Tree INTEGRITY WARNING: Person {log_ref_short} has in_my_tree=True, but no FamilyTree record found.")
                         elif tree_status == "created" and family_tree_args: # If we attempted creation
                              if verify_tree.cfpid != family_tree_args.get("cfpid"):
                                   logger.error(f"Post-commit Tree FAILED: CFPID mismatch! DB='{verify_tree.cfpid}', Expected='{family_tree_args.get('cfpid')}' for {log_ref_short}.")
                              else: logger.debug(f"Post-commit Tree verification OK for {log_ref_short} (CFPID='{verify_tree.cfpid}').")
                         else: # Tree exists, but wasn't created this run
                              logger.debug(f"Post-commit Tree verification OK (existing record found) for {log_ref_short} (CFPID='{verify_tree.cfpid if verify_tree else 'N/A'}').")

            except Exception as verify_e:
                logger.error(f"Exception during post-commit verification for {log_ref_short}: {verify_e}", exc_info=True)

        # --- 9. Return final determined status ---
        logger.debug(f"Final overall status for {log_ref_short}: {overall_status}")
        return overall_status, None

    # --- 10. Handle Broad Exceptions ---
    # Logic remains the same as V7
    except (requests.exceptions.RequestException, WebDriverException) as net_e:
        error_msg = f"Network/WebDriver error processing match {log_ref}: {net_e}"
        logger.warning(error_msg, exc_info=False) # Less verbose logging for network errors
        logger.debug("Traceback for Network/WebDriver error:", exc_info=True)
        if session and session.is_active:
            try: session.rollback(); logger.debug(f"Rolled back session due to Network/WebDriver error for {log_ref}.")
            except Exception as rb_err: logger.error(f"Error rolling back session after Network/WebDriver error for {log_ref}: {rb_err}")
        return "error", error_msg

    except Exception as e:
        error_msg = f"Unexpected critical error in _do_match for {log_ref}: {e}."
        logger.error(error_msg, exc_info=True)
        if session and session.is_active:
            try: session.rollback(); logger.debug(f"Rolled back session due to unexpected error for {log_ref}.")
            except Exception as rb_err: logger.error(f"Error rolling back session after unexpected error for {log_ref}: {rb_err}")
        return "error", f"Unexpected error processing {log_ref}: {e}"
# End of _do_match 

def _log_page_summary(page, page_new, page_updated, page_skipped, page_errors):
    """Logs a summary of processed matches for a single page."""
    logger.debug(f"---- Page {page} Summary ----")
    logger.debug(f"  New matches:     {page_new}")
    logger.debug(f"  Updated matches: {page_updated}")
    logger.debug(f"  Skipped matches: {page_skipped}")
    logger.debug(f"  Error matches:   {page_errors}")
    logger.debug("-----------------------\n")
# end of _log_page_summary

def _log_coord_summary(total_pages_processed, total_new, total_updated, total_skipped, total_errors): # Renamed arg
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
        if session_manager.dynamic_rate_limiter.current_delay > config_instance.INITIAL_DELAY:
            logger.debug(f"Decreased delay for next page to {session_manager.dynamic_rate_limiter.current_delay:.2f} seconds.")
# End of _adjust_delay

#################################################################################
# 3. API Data Acquisition
#################################################################################

def get_page_count(session_manager, my_uuid: str) -> Optional[int]:
    """
    Retrieves the total number of DNA match pages from the Ancestry API.
    Forces use of requests library and ENSURES CSRF TOKEN IS USED.
    """
    match_list_url = f"{urljoin(config_instance.BASE_URL,f'discoveryui-matches/parents/list/api/matchList/{my_uuid}')}?currentPage=1"

    # Headers are now managed within _api_req based on description
    try:
        api_response = _api_req(
            url=match_list_url,
            driver=session_manager.driver, # Still needed for potential UBE header
            session_manager=session_manager,
            method="GET",
            # --- MODIFICATION: Enable CSRF Token ---
            use_csrf_token=True, # This endpoint seems to require CSRF even for GET
            # --- END MODIFICATION ---
            api_description="Match List API",
            referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
            force_requests=True # *** Force Python requests library ***
        )
        if api_response and isinstance(api_response, dict) and "totalPages" in api_response:
            total_pages = api_response["totalPages"]
            return int(total_pages)
        else:
            # Log specific reason if possible (e.g., _api_req returning None implies HTTP error)
            if api_response is None:
                 logger.warning("Failed to retrieve page count: Match List API call failed (returned None).")
            # --- MODIFICATION: Handle non-dict response more gracefully ---
            elif isinstance(api_response, str):
                 logger.warning(f"Unexpected string response from Match List API: '{api_response[:100]}...'")
            # --- END MODIFICATION ---
            else:
                 logger.warning(f"Unexpected response format from Match List API. Type: {type(api_response)}")
                 logger.debug(f"Full response data: {api_response}")
            return None

    except Exception as e:
        logger.error(f"Error fetching total pages from API: {e}", exc_info=True)
        return None
# end get_page_count

def get_matches(
    session_manager: SessionManager, current_page: int = 1
) -> List[Dict[str, Any]]:
    """
    Fetches and processes match list data for a SINGLE page. Includes administrator
    hints, attempts to parse last login date, and removes initial message_link construction.
    Forces use of requests library for the main match list API call and ENSURES CSRF TOKEN IS USED.
    Adds debug logging for refined matches.
    """
    MAX_LABELS_TO_SHOW = 2

    if not isinstance(session_manager, SessionManager):
        logger.error("Invalid SessionManager passed to get_matches.")
        return []
    if not session_manager.driver or not session_manager.is_sess_valid():
        logger.error("WebDriver session is not valid in get_matches.")
        return []
    if not session_manager.my_uuid:
        logger.error("SessionManager my_uuid is not initialized in get_matches.")
        return []

    my_uuid = session_manager.my_uuid
    # --- Initialize predicted_relationships dictionary ---
    predicted_relationships: Dict[str, str] = {} # Ensure it's defined early

    try:
        # --- 1. Fetch Match List Data ---
        match_list_url = urljoin(
            config_instance.BASE_URL,
            f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={current_page}",
        )
        logger.debug(f"Fetching match list for page {current_page} using requests...")
        api_response = _api_req(
            url=match_list_url,
            driver=session_manager.driver, # Still needed for potential UBE header
            session_manager=session_manager,
            method="GET",
            use_csrf_token=True, # This endpoint seems to require CSRF even for GET
            api_description="Match List API",
            referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
            force_requests=True # *** Force Python requests library ***
        )

        if api_response is None:
            logger.warning(
                f"No response received from match list API for page {current_page} (using requests)."
            )
            return []
        if isinstance(api_response, str):
            logger.warning(f"Received string instead of dict from match list API for page {current_page} (using requests): '{api_response[:100]}...'")
            return []
        if not isinstance(api_response, dict):
            logger.warning(
                f"Unexpected data type received from match list API for page {current_page} (using requests). Type: {type(api_response)}."
            )
            logger.debug(f"Response data: {api_response}")
            return []

        match_data_list = api_response.get("matchList", [])
        if not match_data_list:
            logger.info(f"No matches found in 'matchList' for page {current_page}.")
            return []

        logger.debug(f"Got {len(match_data_list)} raw matches from API on page {current_page}.")

        # --- Filter raw matches for essential 'sampleId' ---
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
                f"Skipped {skipped_sampleid_count} raw matches on page {current_page} due to missing 'sampleId'."
            )
        if not valid_matches_for_processing:
            logger.warning(
                f"No matches with valid 'sampleId' found on page {current_page}."
            )
            return []

        sample_ids = [match["sampleId"].upper() for match in valid_matches_for_processing]

        # --- In-Tree and Relationship processing ---
        in_tree_ids: Set[str] = set() # Ensure type hint
        # predicted_relationships dict is initialized earlier

        # 2. In-Tree Status Check (logic remains the same)
        cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids))}"
        cached_in_tree = session_manager.cache.get(cache_key_tree)
        if cached_in_tree is not None and isinstance(cached_in_tree, set): # Check type from cache
            in_tree_ids = cached_in_tree
            logger.debug(f"Loaded {len(in_tree_ids)} in-tree IDs from cache.")
        else:
            in_tree_url = urljoin(
                config_instance.BASE_URL,
                f"discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid.upper()}",
            )
            logger.debug("Fetching in-tree status...")
            # This API call likely needs CSRF and uses POST, so _api_req will use requests by default
            response_in_tree = _api_req(
                url=in_tree_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="POST",
                json_data={"sampleIds": sample_ids},
                use_csrf_token=True,
                api_description="In-Tree Status Check",
                referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
            )
            if isinstance(response_in_tree, list):
                in_tree_ids = {
                    item.upper() for item in response_in_tree if isinstance(item, str)
                }
                # Use config_instance.CACHE_TIMEOUT for consistency
                session_manager.cache.set(
                    cache_key_tree, in_tree_ids, timeout=config_instance.CACHE_TIMEOUT
                )
                logger.debug(f"Fetched/cached {len(in_tree_ids)} in-tree IDs.")
            elif response_in_tree is None:
                logger.warning("In-Tree Status Check API failed (_api_req returned None).")
            else:
                logger.error(
                    f"Unexpected in-tree status response format: {type(response_in_tree)}"
                )

        # 3. Parallel Predicted Relationship Processing (logic remains the same)
        logger.debug("Getting predicted relationships...")

        @retry()
        def process_sample(sample_id: str) -> tuple[str, str]:
            """Fetches relationship probability."""
            # Ensure my_uuid is available inside the nested function
            if not session_manager.my_uuid:
                logger.error("Cannot process relationship sample: my_uuid missing.")
                return sample_id.upper(), "N/A (Error - Missing Own UUID)"
            try:
                my_uuid_upper = session_manager.my_uuid.upper()
                sample_id_upper = sample_id.upper()
                rel_url = urljoin(
                    config_instance.BASE_URL,
                    f"discoveryui-matches/parents/list/api/matchProbabilityData/{my_uuid_upper}/{sample_id_upper}",
                )
                # This API call needs CSRF and uses POST, so _api_req will use requests by default
                response_rel = _api_req(
                    url=rel_url,
                    driver=session_manager.driver,
                    session_manager=session_manager,
                    method="POST",
                    json_data={}, # Empty JSON body
                    use_csrf_token=True,
                    api_description="Match Probability API",
                    referer_url=urljoin(
                        config_instance.BASE_URL, "/discoveryui-matches/list/"
                    ),
                )
                if response_rel is None:
                    # Raise specific exception to trigger retry
                    raise requests.exceptions.RequestException(f"Match Probability API req failed for {sample_id}")
                if not isinstance(response_rel, dict) or "matchProbabilityToSampleId" not in response_rel:
                    logger.warning(f"Invalid data format from Match Probability API for {sample_id_upper}. Resp: {response_rel}")
                    return sample_id_upper, "N/A (Invalid Data)"

                prob_data = response_rel["matchProbabilityToSampleId"]
                predictions = prob_data.get("relationships", {}).get("predictions", [])
                if not predictions:
                    logger.debug(f"No relationship predictions found for {sample_id_upper}. Marking as Distant.")
                    return sample_id_upper, "Distant relationship?" # Or "No predictions"

                valid_preds = [
                    p
                    for p in predictions
                    if isinstance(p, dict) and "distributionProbability" in p and "pathsToMatch" in p
                ]
                if not valid_preds:
                    logger.warning(f"No valid prediction paths found for {sample_id_upper}.")
                    return sample_id_upper, "N/A (No Valid Paths)"

                # Find the prediction with the highest probability
                best_pred = max(valid_preds, key=lambda x: x.get("distributionProbability", 0.0))
                top_prob = best_pred.get("distributionProbability", 0.0) * 100 # Convert to percentage
                paths = best_pred.get("pathsToMatch", [])
                labels = [p.get("label") for p in paths if isinstance(p, dict) and p.get("label")]

                if not labels:
                    logger.warning(f"Prediction found for {sample_id_upper}, but no labels in paths.")
                    return sample_id_upper, f"N/A (No Labels) [{top_prob:.1f}%]"

                # Limit labels shown and join them
                final_labels = labels[:MAX_LABELS_TO_SHOW]
                relationship_str = " or ".join(map(str, final_labels))
                return sample_id_upper, f"{relationship_str} [{top_prob:.1f}%]"

            except requests.exceptions.RequestException as req_e:
                logger.warning(f"RequestException processing relationship for {sample_id}: {req_e}. Retrying...")
                raise req_e # Re-raise to trigger retry decorator
            except Exception as e:
                logger.error(
                    f"Unexpected error processing relationship for {sample_id}: {e}",
                    exc_info=False, # Keep False for less noise, True for deep debug
                )
                logger.debug(f"Traceback:", exc_info=True)
                return sample_id.upper(), "N/A (Error)"

        # Execute relationship processing in parallel
        predicted_relationships = {} # Re-initialize just before the parallel execution
        skipped_rel_count = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_sid = {
                executor.submit(process_sample, sid): sid.upper() for sid in sample_ids
            }
            for future in as_completed(future_to_sid):
                sid_upper = future_to_sid[future]
                try:
                    _, rel_str = future.result()
                    predicted_relationships[sid_upper] = rel_str
                except Exception as exc:
                    logger.error(
                        f"Relationship future for {sid_upper} failed after retries: {exc}"
                    )
                    predicted_relationships[sid_upper] = "N/A (Future Error)"
                    skipped_rel_count += 1
        logger.debug(
            f"Got {len(predicted_relationships)} predicted relationships ({skipped_rel_count} errors)."
        )


        # --- 4. Compile Final Refined Match Data ---
        refined_matches: List[Dict[str, Any]] = [] # Ensure type hint
        skipped_profile_id_count = 0
        skipped_last_login_count = 0 # Track missing last login

        for match in valid_matches_for_processing:
            profile = match.get("matchProfile", {})
            relationship = match.get("relationship", {})
            sample_id_upper = match["sampleId"].upper()

            profile_user_id = profile.get("userId")
            match_username = profile.get("displayName", "Unknown").title()

            admin_profile_id_hint = match.get("adminId")
            admin_username_hint = match.get("adminName")

            # --- Extract and parse lastLoginDate (from initial match list API) ---
            last_login_str = match.get("lastLoginDate")
            last_login_dt = None
            if last_login_str:
                try:
                    # Handle 'Z' for UTC timezone explicitly
                    if last_login_str.endswith('Z'):
                         last_login_dt = datetime.fromisoformat(last_login_str.replace('Z', '+00:00'))
                    else:
                         # Assume ISO format without timezone or with offset already included
                         last_login_dt = datetime.fromisoformat(last_login_str)
                except (ValueError, TypeError) as date_parse_e:
                    logger.warning(
                        f"Could not parse lastLoginDate string '{last_login_str}' for {match_username}. Error: {date_parse_e}"
                    )
                    skipped_last_login_count += 1
                except Exception as date_parse_e:
                    logger.error(
                        f"Unexpected error parsing lastLoginDate '{last_login_str}' for {match_username}: {date_parse_e}",
                        exc_info=False,
                    )
                    skipped_last_login_count += 1
            else:
                # logger.debug(f"lastLoginDate missing for {match_username}.") # Optional debug log
                skipped_last_login_count += 1
            # --- END MODIFICATION ---

            if not profile_user_id:
                logger.debug(
                    f"Match '{match_username}' (UUID: {sample_id_upper}) missing tester 'profile_id' in initial list. Will attempt fetch via details API."
                )
                skipped_profile_id_count += 1
                profile_user_id_upper = None
            else:
                profile_user_id_upper = str(profile_user_id).upper()

            compare_link = urljoin(
                config_instance.BASE_URL,
                f"discoveryui-matches/compare/{my_uuid.upper()}/with/{sample_id_upper}",
            )
            # message_link is constructed later in _do_match

            # Derive first name
            first_name = match_username.split()[0] if match_username != "Unknown" else None

            refined_match_data = {
                "username": match_username,
                "first_name": first_name,
                "initials": profile.get("displayInitials", "??").upper(),
                "gender": match.get("gender"), # Gender hint (may be overwritten later)
                "profile_id": profile_user_id_upper, # Might be None initially (THIS IS A HINT ONLY)
                "uuid": sample_id_upper,
                "administrator_profile_id_hint": admin_profile_id_hint,
                "administrator_username_hint": admin_username_hint,
                "photoUrl": profile.get("photoUrl", ""),
                "cM_DNA": int(relationship.get("sharedCentimorgans", 0)),
                "numSharedSegments": int(relationship.get("numSharedSegments", 0)),
                "compare_link": compare_link,
                "message_link": None, # Set later
                # --- Ensure lookup uses uppercase UUID ---
                "predicted_relationship": predicted_relationships.get(
                    sample_id_upper, "N/A"
                ),
                # --- End ensure ---
                "in_my_tree": sample_id_upper in in_tree_ids,
                "createdDate": match.get("createdDate"),
                # Store the potentially parsed datetime object (or None)
                "last_logged_in_dt": last_login_dt,
            }
            refined_matches.append(refined_match_data)

        total_raw = len(match_data_list)
        total_valid_uuid = len(valid_matches_for_processing)
        final_count = len(refined_matches)
        logger.debug(
            f"Processed page {current_page}: Raw={total_raw}, ValidUUID={total_valid_uuid}, Refined={final_count} (MissingTesterProfileID={skipped_profile_id_count}, MissingLastLogin={skipped_last_login_count})"
        )

        # --- Added Debugging ---
        logger.debug(f"Refined matches being returned from get_matches (Page {current_page}):")
        for i, rm in enumerate(refined_matches):
             logger.debug(f"  Match {i+1}: User='{rm.get('username')}', PredRel='{rm.get('predicted_relationship')}'")
        # --- End Added Debugging ---

        return refined_matches

    except requests.exceptions.RequestException as e:
        logger.error(
            f"Network/Request error processing page {current_page}: {e}", exc_info=True
        )
        return []
    except Exception as e:
        logger.critical(
            f"Critical error processing match data for page {current_page}: {e}",
            exc_info=True,
        )
        return []
# end get_matches

def _get_tree(session, match: dict, session_manager) -> Optional[dict]:  # Point 8
    """
    Fetches details for a DNA match from Ancestry's API, including their first name.
    Adds detailed logging of the API response and parsed data.
    """
    my_uuid = session_manager.my_uuid
    their_uuid = match.get("uuid")
    username = match.get("username", "Unknown")  # Get username for logging
    if not their_uuid or not my_uuid:
        logger.warning(
            f"_get_tree: Missing their_uuid or my_uuid in match data for {username}: {match}"
        )
        return None

    badge_url = urljoin(
        config_instance.BASE_URL,
        f"/discoveryui-matchesservice/api/samples/{my_uuid}/matches/{their_uuid}/badgedetails",
    )

    headers = config_instance.API_CONTEXTUAL_HEADERS.get(
        "Badge Details API", {"Accept": "application/json"}
    )

    try:
        api_response = _api_req(
            url=badge_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers=headers,
            use_csrf_token=False,
            api_description="Badge Details API",
            referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
        )

        # --- Log the raw API response ---
        logger.debug(
            f"_get_tree raw API response for {username} (UUID: {their_uuid}):\n{api_response}"
        )
        # ---

        if api_response and isinstance(api_response, dict):
            person_badged = api_response.get("personBadged", {})
            full_firstname = person_badged.get("firstName", "Unknown")
            words = full_firstname.strip().split()
            their_firstname = words[0] if words else "Unknown"

            tree_data = {
                "their_cfpid": person_badged.get("personId"),
                "their_firstname": their_firstname,
                "their_lastname": person_badged.get("lastName", "Unknown"),
                "their_birth_year": person_badged.get("birthYear"),
            }

            # --- Log the parsed tree_data before returning ---
            logger.debug(f"_get_tree parsed data for {username}: {tree_data}")
            # ---

            if not tree_data.get("their_cfpid"):
                logger.warning(
                    f"_get_tree: Missing 'personId' (CFPID) in parsed badge details for {username}."
                )
                # Still return partial data, let caller handle missing cfpid
                return tree_data

            return tree_data  # Return the full parsed data if cfpid exists
        else:
            logger.warning(
                f"_get_tree returned non-dict or empty response for {username}. Type: {type(api_response)}"
            )
            return None

    except Exception as e:
        logger.error(
            f"Failed to fetch/parse tree details for {username}: {str(e)}", exc_info=True
        )
        return None
# End of _get_tree


@retry()
def _get_relShip(
    session_manager: SessionManager, tree_id: str, cfpid: str
) -> Optional[Dict[str, Any]]:
    """
    Fetches relationship path and actual relationship using the getladder API via
    the execute_script-based _api_req. Handles JSONP response format and HTML parsing.
    Corrects casing in relationship path description names.
    """
    # 1. Initialisation checks
    if not session_manager:
        logger.error("SessionManager not provided to _get_relShip.")
        return None
    if not session_manager.driver or not session_manager.is_sess_valid():  # Check driver validity
        logger.error("WebDriver session invalid in _get_relShip.")
        return None

    if not tree_id or not cfpid:
        logger.warning(
            f"Missing tree_id ({tree_id}) or cfpid ({cfpid}) for relationship API call."
        )
        return None
    logger.debug(
        f"Fetching relationship details for tree {tree_id}, cfpid {cfpid}."
    )  # Added tree_id for context

    # 2. Construct API URL and dynamic Referer
    api_url = urljoin(
        config_instance.BASE_URL,
        f"family-tree/person/tree/{tree_id}/person/{cfpid}/getladder?callback=jQuery",  # Added param
    )
    dynamic_referer = urljoin(
        config_instance.BASE_URL, f"family-tree/person/tree/{tree_id}/person/{cfpid}/facts"
    )

    try:
        # 3. Make API call using _api_req
        response_text = _api_req(
            url=api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers={"Accept": "*/*"},
            use_csrf_token=False,
            api_description="Get Ladder API",
            referer_url=dynamic_referer,
        )

        # 4. Process the response
        if response_text is None:
            logger.error(
                f"Failed to get relationship data for cfpid {cfpid}: _api_req returned None."
            )
            return None
        if not isinstance(response_text, str):
            logger.error(
                f"Unexpected response type from _api_req for cfpid {cfpid}: {type(response_text)}. Expected string."
            )
            logger.debug(f"Response data: {response_text}")
            return None

        # 5. Parse JSONP format
        match = re.match(r"^[^(]*\((.*)\)[^)]*$", response_text, re.DOTALL | re.IGNORECASE)
        if not match:
            logger.error(
                f"Could not parse JSONP response format for cfpid {cfpid}. Regex failed."
            )
            logger.debug(f"Full Response Text Snippet: {response_text[:500]}...")
            return None
        json_string = match.group(1).strip()

        # 6. Decode JSON and Extract HTML
        try:
            ladder_data = json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from getladder for cfpid {cfpid}: {e}.")
            logger.debug(f"Extracted JSON String Snippet: {json_string[:500]}...")
            return None

        if not isinstance(ladder_data, dict) or "html" not in ladder_data:
            logger.warning(
                f"Unexpected structure in getladder JSON (missing 'html') for cfpid {cfpid}."
            )
            logger.debug(f"Ladder Data Received: {ladder_data}")
            return None

        html_content = ladder_data["html"]
        if not html_content:
            logger.warning(
                f"HTML content in getladder response is empty for cfpid {cfpid}."
            )
            return {"actual_relationship": None, "relationship_path": None}

        # 7. Parse HTML for Relationship and Path
        soup = BeautifulSoup(html_content, "html.parser")
        actual_relationship = None
        relationship_path = None

        # Extract Actual Relationship
        relationship_element = soup.select_one('ul.textCenter > li:first-child > i > b')
        if relationship_element:
            raw_relationship = relationship_element.get_text(strip=True)
            # Apply title casing and ordinal correction
            actual_relationship = ordinal_case(raw_relationship.title())
        else:
            logger.warning(
                f"Could not extract actual_relationship from HTML for cfpid: {cfpid}"
            )

        # Extract Relationship Path
        path_items = soup.select('ul.textCenter > li:not([class*="iconArrowDown"])')
        relationship_path_list = []
        num_items = len(path_items)

        for i, item in enumerate(path_items):
            name_text = ""  # Will hold the formatted name
            desc_text = ""  # Will hold the formatted description

            # --- Extract Name (apply format_name here) ---
            raw_name_extracted = ""
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
                potential_name_parts = item.get_text(separator="\n", strip=True).split(
                    "\n"
                )
                if potential_name_parts:
                    raw_name_extracted = potential_name_parts[0]

            # Clean and format the extracted name
            if raw_name_extracted:
                cleaned_name = " ".join(raw_name_extracted.replace('"', "'").split())
                name_text = format_name(cleaned_name)  # Format the main name
            else:
                logger.warning(
                    f"Could not extract raw name for path item {i+1}, cfpid {cfpid}."
                )
            # --- End Extract Name ---

            # --- Extract and Format Description ---
            if i > 0:  # Skip first item 'You'
                desc_element = item.find("i")
                if desc_element:
                    raw_desc_full = desc_element.get_text(strip=True)
                    cleaned_desc_full = raw_desc_full.replace('"', "'")

                    # Check for "You are the ..." prefix for the last item
                    if i == num_items - 1 and cleaned_desc_full.lower().startswith(
                        "you are the "
                    ):
                        relationship_part = cleaned_desc_full[len("You are the ") :].strip()
                        # Apply format_name specifically to this relationship part
                        desc_text = format_name(relationship_part)
                    else:
                        # Attempt to parse names within the description like "Relation of Name Name"
                        match = re.match(
                            r"^(.*?)\s+of\s+(.*)$", cleaned_desc_full, re.IGNORECASE
                        )
                        if match:
                            relation_part = match.group(1).strip()
                            name_part = match.group(2).strip()
                            # Format the name part and reconstruct the description
                            formatted_name_part = format_name(name_part)
                            # Capitalize the relation part (e.g., "Son", "Daughter")
                            desc_text = f"{relation_part.capitalize()} of {formatted_name_part}"
                        else:
                            # If regex doesn't match, format the whole description string
                            desc_text = format_name(cleaned_desc_full)
                else:
                    logger.warning(
                        f"Could not find description (<i> tag) for path item {i+1} for cfpid {cfpid}."
                    )
            # --- End Extract Description ---

            # Build list item string (name_text is already formatted)
            if name_text:
                list_item = f"{name_text} ({desc_text})" if desc_text else name_text
                relationship_path_list.append(list_item)
            # else: name extraction warning already logged above

        # --- Join path items ---
        if relationship_path_list:
            relationship_path = "\n↓\n".join(relationship_path_list)
        else:
            logger.warning(
                f"Could not construct relationship_path for cfpid {cfpid}."
            )

        # 8. Return results
        result_data = {
            "actual_relationship": actual_relationship,
            "relationship_path": relationship_path,
        }
        # Log slightly differently for debug clarity
        log_rel = actual_relationship if actual_relationship else "N/A"
        log_path_len = len(relationship_path) if relationship_path else 0
        logger.debug(
            f"Got relationship details for cfpid {cfpid}: Actual='{log_rel}', Path estimated length: {log_path_len}"
        )
        return result_data

    except Exception as e:
        logger.error(
            f"Unexpected error processing relationship details for cfpid {cfpid}: {e}",
            exc_info=True,
        )
        raise  # Re-raise exception for retry decorator
# End of _get_relShip

#################################################################################
# 4. Match Data Processing & Database Integration
#################################################################################

def need_update(
    session: SqlAlchemySession, match: Dict[str, Any]
) -> Tuple[Optional[Person], bool, bool]:
    """
    REVISED: Checks if related data needs updating based *only* on UUID lookup.
    Determines if DNA record needs creation or Tree data needs fetching.
    Does NOT determine if Person record itself needs update (handled later).
    Expires object state before comparisons.

    Args:
        session: The SQLAlchemy session.
        match: Dictionary containing initial match data including 'uuid', 'username', 'in_my_tree'.

    Returns:
        Tuple: (existing_person_hint, create_dna_needed, fetch_tree_data)
               existing_person_hint is the Person object found by UUID (or None).
    """
    create_dna_needed = False
    fetch_tree_data = False
    existing_person_hint: Optional[Person] = None

    match_uuid = match.get("uuid")
    match_username = match.get("username", f"Unknown Match UUID {match_uuid or 'N/A'}")
    match_in_my_tree = match.get("in_my_tree", False)

    log_ref = f"UUID='{match_uuid or 'N/A'}' / User='{match_username}'"

    if not match_uuid:
        logger.error(f"Cannot process need_update for {match_username}: 'uuid' is missing.")
        return None, False, False # Cannot proceed without UUID

    try:
        # --- Lookup primarily by UUID ---
        # Eager load relationships needed for checks
        existing_person_hint = (
            session.query(Person)
            .options(joinedload(Person.dna_match), joinedload(Person.family_tree)) # Eager load
            .filter(Person.uuid == match_uuid.upper())
            .first()
        )

        # --- Expire state after fetch, before comparison (still useful) ---
        if existing_person_hint:
            try:
                logger.debug(
                    f"{log_ref}: Expiring state for existing Person ID {existing_person_hint.id} before checks."
                )
                session.expire(existing_person_hint, ['dna_match', 'family_tree']) # Expire specific relationships too if needed
            except Exception as expire_e:
                logger.warning(
                    f"Could not expire session state for Person ID {existing_person_hint.id}: {expire_e}"
                )
        # --- End Expire ---

        # --- Determine need based on existing record ---
        if existing_person_hint:
            # Person found by UUID
            logger.debug(f"{log_ref}: Found existing Person ID {existing_person_hint.id} by UUID.")

            # Check DnaMatch Need: Needs creation if Person exists but DnaMatch doesn't
            # Accessing existing_person_hint.dna_match will trigger load if expired/not loaded
            existing_dna_record = existing_person_hint.dna_match
            if existing_dna_record is None:
                logger.debug(f"{log_ref}: Existing Person found, but no DnaMatch record. Needs DNA creation.")
                create_dna_needed = True
            else:
                logger.debug(f"{log_ref}: Existing DnaMatch record found. No DNA creation needed.")
                create_dna_needed = False

            # Check FamilyTree Need: Needs fetch if status changes to 'in_my_tree'
            # Accessing existing_person_hint.in_my_tree reloads if needed
            db_in_my_tree = existing_person_hint.in_my_tree
            if match_in_my_tree and not db_in_my_tree:
                logger.debug(
                    f"{log_ref}: Status changed to 'in_my_tree'. Needs tree data fetch."
                )
                fetch_tree_data = True
            elif db_in_my_tree and not match_in_my_tree:
                 logger.warning(f"{log_ref}: Status changed FROM 'in_my_tree' to False. Skipping tree fetch, existing tree data might become stale.")
                 fetch_tree_data = False
            else:
                logger.debug(f"{log_ref}: 'in_my_tree' status unchanged ({match_in_my_tree}). No tree data fetch needed based on status change.")
                fetch_tree_data = False

        # --- Person Not Found by UUID ---
        else:
            logger.debug(f"{log_ref}: No existing person found by UUID.")
            existing_person_hint = None # Ensure it's None
            # If person doesn't exist, DNA match also doesn't exist
            create_dna_needed = True
            # If person doesn't exist, fetch tree data only if the flag is set
            fetch_tree_data = match_in_my_tree

    except SQLAlchemyError as e:
         logger.error(f"Database error during need_update check for {log_ref}: {e}", exc_info=True)
         return None, False, False # Return defaults on error
    except Exception as e:
         logger.error(f"Unexpected error during need_update check for {log_ref}: {e}", exc_info=True)
         return None, False, False

    # --- Final Decision Logging ---
    actions_needed = []
    if create_dna_needed: actions_needed.append("Create DNA Match")
    if fetch_tree_data: actions_needed.append("Fetch/Process Tree Data")

    if not actions_needed: final_decision = "No DNA/Tree actions needed."
    else: final_decision = " AND ".join(actions_needed)
    logger.debug(f"{log_ref}: Need Update Check Complete. Actions needed: {final_decision}")

    return existing_person_hint, create_dna_needed, fetch_tree_data
# End of need_update

#################################################################################
# 5. 'Create or Update' Database Operations
#################################################################################

class PersonProcessingError(Exception):
    """Custom exception for errors during Person creation/update."""
    pass
# end of PersonProcessingError class

def _do_person(session: Session, match: Dict[str, Any]) -> Tuple[Optional[Person], Literal["new", "updated", "skipped", "error"]]:
    """Creates or updates a Person record based primarily on profile_id.
       Flushes the session after adding a new person to ensure ID is available.

    Args:
        session: The SQLAlchemy Session.
        match: Dictionary containing match data including 'profile_id', 'username',
               'uuid', 'message_link', 'in_my_tree'.

    Returns:
        A tuple containing the Person object (or None on error) and the status:
        "new", "updated", "skipped", "error".
    """
    profile_id = match.get("profile_id")
    if profile_id:
        profile_id = profile_id.upper()  # Ensure consistent case for lookup
    username = match.get("username")
    uuid_val = match.get("uuid")
    if uuid_val:
        uuid_val = uuid_val.upper()
    else:
        # If profile_id exists, log with that, otherwise use generic message
        log_id = f"profile_id {profile_id}" if profile_id else "unknown profile"
        logger.error(f"_do_person: UUID is missing in match data for {log_id}. Skipping.")
        return None, "error"

    if not profile_id:
        logger.error(f"_do_person: Missing profile_id for match UUID {uuid_val}. Skipping.")
        return None, "error"
    if not username:
        logger.warning(
            f"_do_person: Missing username for profile_id {profile_id}. Will use 'Unknown'."
        )
        username = "Unknown"  # Use a default if missing

    try:
        # --- MODIFIED LOOKUP: Primarily by profile_id ---
        person = (
            session.query(Person).filter(Person.profile_id == profile_id).first()
        )

        if person:
            # Person exists, check if update is needed
            updated = False
            log_identifier = (
                f"{person.username} (ID: {person.id}, Profile: {profile_id})"
            )  # More detailed log identifier

            # Update username if different (case-insensitive comparison)
            if person.username.lower() != username.lower():
                logger.debug(
                    f"Updating username for {log_identifier} from '{person.username}' to '{username}'."
                )
                person.username = username
                updated = True

            # Update UUID if different OR if it was previously None/empty
            # Check both None and empty string for robustness
            if person.uuid != uuid_val and (person.uuid or uuid_val):  # Avoid logging if both are None/empty
                logger.debug(
                    f"Updating UUID for {log_identifier} from '{person.uuid}' to '{uuid_val}'."
                )
                person.uuid = uuid_val
                updated = True

            # Update message_link if different
            match_message_link = match.get("message_link")
            if person.message_link != match_message_link:
                logger.debug(
                    f"Updating message_link for {log_identifier} from '{person.message_link}' to '{match_message_link}'."
                )
                person.message_link = match_message_link
                updated = True

            # Update in_my_tree if different
            match_in_my_tree = bool(match.get("in_my_tree", False))
            if person.in_my_tree is not match_in_my_tree:  # Correct comparison
                logger.debug(
                    f"Updating in_my_tree status to {match_in_my_tree} for {log_identifier}."
                )
                person.in_my_tree = match_in_my_tree
                updated = True

            # Update status if necessary (e.g., ensure it's active)
            # if person.status != "active":
            #     logger.debug(f"Updating status to 'active' for {log_identifier}.")
            #     person.status = "active"
            #     updated = True

            if updated:
                # No flush needed here for updates, commit happens in _do_match
                logger.debug(
                    f"Updated existing 'people' record for {log_identifier}."
                )
                return person, "updated"
            else:
                logger.debug(
                    f"Existing 'people' record for {log_identifier} requires no update."
                )
                return person, "skipped"  # Return skipped if no updates were made

        else:
            # Person does not exist, create new record
            logger.debug(
                f"Creating new 'people' record for profile_id {profile_id} (Username: {username})."
            )
            new_person = Person(
                uuid=uuid_val,
                profile_id=profile_id,
                username=username,
                message_link=match.get("message_link"),
                in_my_tree=bool(match.get("in_my_tree", False)),  # Ensure boolean
                status="active",
            )
            session.add(new_person)
            session.flush()  # <<< FLUSH TO GET ID

            # --- CRITICAL CHECK: Ensure ID is assigned after flush ---
            if new_person.id is None:
                logger.error(
                    f"Person ID not assigned after flush for {username} (profile_id: {profile_id})! Rolling back this attempt."
                )
                raise PersonProcessingError(
                    f"Person ID not assigned after flush for {username}"
                )

            logger.debug(
                f"Created record for {username} (Profile: {profile_id}), assigned ID: {new_person.id}"
            )
            return new_person, "new"

    except SQLAlchemyError as e:
        log_username = (
            username if "username" in locals() and username else f"profile_id {profile_id}"
        )
        logger.error(
            f"Database error in _do_person for {log_username}: {e}", exc_info=True
        )
        # Rollback will be handled by _do_match
        return None, "error"
    except PersonProcessingError as e:  # Catch specific error from ID check
        log_username = (
            username if "username" in locals() and username else f"profile_id {profile_id}"
        )
        logger.error(f"Person processing error in _do_person for {log_username}: {e}")
        return None, "error"
    except Exception as e:  # Catch any other unexpected error
        log_username = (
            username if "username" in locals() and username else f"profile_id {profile_id}"
        )
        logger.critical(
            f"Unexpected error in _do_person for {log_username}: {e}", exc_info=True
        )
        return None, "error"
# End of _do_person

def _do_DNA(
    session: Session, person: Person, match: Dict[str, Any]
) -> Literal["created", "updated", "skipped", "error"]:  # Point 7
    """
    Creates a new DNA Match record associated with a Person.
    Handles new fields. Does NOT update existing records per requirement 3.
    """
    if not person or person.id is None:
        logger.error("_do_DNA called with invalid Person object.")
        return "error"

    person_id = person.id
    log_ref = f"PersonID={person_id}, KitUUID={match.get('uuid', 'N/A')}"

    # Validate essential match data
    required_keys = ("compare_link", "cM_DNA", "predicted_relationship")
    if not all(key in match and match[key] is not None for key in required_keys):  # Check for None as well
        logger.error(
            f"_do_DNA: Missing required non-null DNA data for {log_ref}. Match data: {match}"
        )
        return "error"

    try:
        cm_dna_val = int(match["cM_DNA"])
        if cm_dna_val < 0:
            raise ValueError("cM_DNA cannot be negative")
    except (TypeError, ValueError, KeyError):
        logger.error(
            f"_do_DNA: Invalid cM_DNA value '{match.get('cM_DNA')}' for {log_ref}."
        )
        return "error"
    # Validate optional numeric fields
    def validate_optional_numeric(key, value, allow_float=False):
        if value is None:
            return None  # None is valid
        try:
            # Handle potential strings like 'N/A' or empty strings before conversion
            if isinstance(value, str) and not value.replace('.', '', 1).isdigit():
                logger.warning(
                    f"_do_DNA: Non-numeric value '{value}' for {key} in {log_ref}. Setting to None."
                )
                return None
            return float(value) if allow_float else int(value)
        except (TypeError, ValueError):
            logger.warning(
                f"_do_DNA: Invalid {key} '{value}' for {log_ref}. Setting to None."
            )
            return None

    shared_segments_val = validate_optional_numeric(
        "shared_segments", match.get("shared_segments")
    )
    longest_segment_val = validate_optional_numeric(
        "longest_shared_segment", match.get("longest_shared_segment"), allow_float=True
    )
    meiosis_val = validate_optional_numeric("meiosis", match.get("meiosis"))

    try:
        # Requirement 3: Only CREATE, never update. Check if exists first.
        dna_match = session.query(DnaMatch).filter_by(people_id=person_id).first()

        if dna_match:
            logger.debug(
                f"Existing 'dna_match' record found for {log_ref}. Skipping update per requirement."
            )
            return "skipped"
        else:
            # Create new DNA match record including new fields
            logger.debug(f"Creating new 'dna_match' record for {log_ref}")
            new_dna_match = DnaMatch(
                people_id=person_id,
                compare_link=match["compare_link"],
                cM_DNA=cm_dna_val,
                predicted_relationship=match["predicted_relationship"],
                # --- MODIFIED: Add new fields on creation ---
                shared_segments=shared_segments_val,
                longest_shared_segment=longest_segment_val,
                meiosis=meiosis_val,
                from_my_fathers_side=bool(match.get("from_my_fathers_side")),  # Ensure boolean
                from_my_mothers_side=bool(match.get("from_my_mothers_side")),  # Ensure boolean
                # --- END MODIFICATION ---
            )
            session.add(new_dna_match)
            # Flush might be needed if ID used immediately, but commit in _do_match handles persistence
            # session.flush()
            logger.debug(f"Staged new 'dna_match' record for creation: {log_ref}")
            return "created"  # Commit happens in _do_match

    except IntegrityError as ie:  # Should be rare now due to explicit check
        session.rollback()
        logger.error(
            f"IntegrityError in _do_DNA for {log_ref}: {ie}. Likely concurrent creation.",
            exc_info=True,
        )
        # Check again just in case
        existing = session.query(DnaMatch).filter_by(people_id=person_id).first()
        if existing:
            logger.warning("Found existing DnaMatch after IntegrityError, treating as skipped.")
            return "skipped"
        return "error"
    except SQLAlchemyError as e:
        logger.error(
            f"Database error in _do_DNA for {log_ref}: {e}", exc_info=True
        )
        return "error"  # Rollback handled by _do_match
    except Exception as e:  # Catch any other unexpected error
        logger.error(
            f"Unexpected error in _do_DNA for {log_ref}: {e}", exc_info=True
        )
        return "error"  # Rollback handled by _do_match
# End of _do_DNA

def _do_tree(session: Session, person: Person, tree_data: Optional[Dict[str, Any]]) -> Literal["created", "updated", "skipped", "error"]:
    """
    Creates or updates a FamilyTree record, using person_name_in_tree column.
    Skips updates for existing records per requirement 3. Adds pre-add logging including cfpid.
    """
    if not person or person.id is None:
        logger.error("_do_tree called with invalid Person object.")
        return "error"

    if not tree_data:
        logger.debug(
            f"_do_tree: No tree data provided for {person.username} (people_id: {person.id}). Skipping DB operation."
        )
        return "skipped"

    person_name_from_tree = tree_data.get("person_name_in_tree")
    facts_link_val = tree_data.get("facts_link")
    view_link_val = tree_data.get("view_in_tree_link")
    actual_rel_val = tree_data.get("actual_relationship")
    rel_path_val = tree_data.get("relationship_path")
    their_cfpid = tree_data.get("their_cfpid") # Get cfpid from the provided data

    try:
        family_tree = session.query(FamilyTree).filter_by(people_id=person.id).first()

        if family_tree:
            # ... (existing tree logic - NO CHANGE) ...
            return "skipped"
        else:
            if not tree_data:
                # ... (no tree data handling - NO CHANGE) ...
                return "skipped"

            # --- ADDED: Log data just before creating object - includes CFPID now
            logger.debug(
                f"Preparing to create FamilyTree for Person ID {person.id}: "
                f"CFPID='{their_cfpid}', Name='{person_name_from_tree}', Rel='{actual_rel_val}', " # Added CFPID logging
                f"FactsLink='{facts_link_val is not None}', "
                f"ViewLink='{view_link_val is not None}', "
                f"Path='{rel_path_val is not None}'"
            )
            # --- End Log ---

            new_family_tree = FamilyTree(
                people_id=person.id,
                cfpid=their_cfpid, # Assign the extracted cfpid
                person_name_in_tree=person_name_from_tree,
                facts_link=facts_link_val,
                view_in_tree_link=view_link_val,
                actual_relationship=actual_rel_val,
                relationship_path=rel_path_val,
            )
            # ... (rest of FamilyTree object creation and saving - NO CHANGE) ...
            session.add(new_family_tree)
            logger.debug(f"Staged new record in 'family_tree' for {person.username}.")
            return "created"  # Commit happens in _do_match

    except IntegrityError as ie:
        # ... (IntegrityError handling - NO CHANGE) ...
        return "error"
    except SQLAlchemyError as e:
        # ... (SQLAlchemyError handling - NO CHANGE) ...
        return "error"
    except Exception as e:
        # ... (unexpected error handling - NO CHANGE) ...
        return "error"
# End of _do_tree

#################################################################################
# 6. Utility & Helper Functions
#################################################################################

def nav_to_list(session_manager) -> bool:
    """Navigates directly to the user's specific DNA matches list page using their UUID."""
    # Ensure session is valid and UUID is available
    if not session_manager.is_sess_valid() or not session_manager.my_uuid:
        logger.error("Session invalid or user UUID missing. Cannot navigate to matches list.")
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


@retry()
def _get_match_details_and_admin(
    session_manager: SessionManager, match_uuid: str
) -> Optional[Dict[str, Any]]:
    """
    Fetches detailed match information, including administrator details, gender,
    and enhanced relationship info, from the /details API endpoint using _api_req.
    Also fetches profile details from /app-api/express/v1/profiles/details for last login and contactable status.
    Includes ancestry-userid header for profile details API. Uses correct tester_profile_id for profile details API.
    Parses lastLoginDate string into datetime.
    Corrected: Parses profile details API response directly (removes "profile" key check).
    """
    if not session_manager.my_uuid:
        logger.error("Cannot get match details: Own UUID (my_uuid) is missing.")
        return None
    if not match_uuid:
        logger.error("Cannot get match details: Target match_uuid is missing.")
        return None
    if not session_manager.my_profile_id:
        logger.error("Cannot get profile details: Own Profile ID (my_profile_id) is missing for header.")
        return None

    details_url = urljoin(
        config_instance.BASE_URL,
        f"/discoveryui-matchesservice/api/samples/{session_manager.my_uuid}/matches/{match_uuid}/details?pmparentaldata=true",
    )
    referer = urljoin(
        config_instance.BASE_URL,
        f"/discoveryui-matches/compare/{session_manager.my_uuid}/with/{match_uuid}",
    )
    profile_details_referer = referer

    logger.debug(f"Fetching match details for UUID: {match_uuid}")
    details = {}

    # --- Fetch Basic Match Details First ---
    try:
        response_data = _api_req(
            url=details_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description="Match Details API",
            referer_url=referer,
        )

        if not response_data or not isinstance(response_data, dict):
            logger.warning(
                f"Failed to get valid details response for UUID {match_uuid} from {details_url}. Response: {response_data}"
            )
            raise requests.exceptions.RequestException(
                f"Match Details API returned invalid data for {match_uuid}"
            )

        # Populate details from the first API response
        details["admin_profile_id"] = response_data.get("adminUcdmId")
        details["admin_username"] = response_data.get("adminDisplayName")
        details["tester_profile_id"] = response_data.get("userId")
        details["tester_username"] = response_data.get("displayName")
        details["tester_initials"] = response_data.get("displayInitials")
        details["gender"] = response_data.get("subjectGender")

        relationship_data = response_data.get("relationship", {})
        details["shared_segments"] = relationship_data.get("sharedSegments")
        details["longest_shared_segment"] = relationship_data.get("longestSharedSegment")
        details["meiosis"] = relationship_data.get("meiosis")
        details["from_my_fathers_side"] = response_data.get("fathersSide", False)
        details["from_my_mothers_side"] = response_data.get("mothersSide", False)

    except Exception as e:
        logger.error(
            f"Error fetching basic match details for UUID {match_uuid} from {details_url}: {e}",
            exc_info=True,
        )
        if isinstance(e, (requests.exceptions.RequestException, WebDriverException)):
            raise e
        return None

    # --- Fetch Profile Details (Last Login, Contactable) ---
    tester_profile_id_for_api = details.get("tester_profile_id")

    if not tester_profile_id_for_api:
        logger.warning(f"Skipping profile details fetch for {match_uuid}: tester_profile_id not found in previous API response.")
        details["last_logged_in_dt"] = None
        details["contactable"] = False
    else:
        profile_details_url_final = urljoin(
            config_instance.BASE_URL,
            f"/app-api/express/v1/profiles/details?userId={tester_profile_id_for_api.upper()}"
        )
        profile_api_headers = {
            "accept": "application/json",
            "ancestry-clientpath": "express-fe",
            "ancestry-userid": session_manager.my_profile_id.upper(),
            "cache-control": "no-cache",
            "pragma": "no-cache",
        }
        logger.debug(f"Fetching profile details from: {profile_details_url_final}")
        try:
            profile_response_data = _api_req(
                url=profile_details_url_final,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                headers=profile_api_headers,
                use_csrf_token=False,
                api_description="Profile Details API",
                referer_url=profile_details_referer,
            )

            # *** CORRECTED PARSING LOGIC ***
            # Check if response is a dictionary (removed check for "profile" key)
            if profile_response_data and isinstance(profile_response_data, dict):
                # Directly access keys from the response dictionary
                last_login_str = profile_response_data.get("LastLoginDate")
                contactable_val = profile_response_data.get("IsContactable", False) # Prefer IsContactable

                last_login_dt = None
                if last_login_str:
                    try:
                        if last_login_str.endswith('Z'):
                             last_login_dt = datetime.fromisoformat(last_login_str.replace('Z', '+00:00'))
                        else:
                             last_login_dt = datetime.fromisoformat(last_login_str)
                        logger.debug(f"Parsed lastLoginDate for {tester_profile_id_for_api}: {last_login_dt}")
                    except (ValueError, TypeError) as date_parse_e:
                        logger.warning(f"Could not parse lastLoginDate string '{last_login_str}' from profile details API for {tester_profile_id_for_api}: {date_parse_e}")
                else:
                     logger.debug(f"LastLoginDate missing in profile details response for {tester_profile_id_for_api}")

                details["last_logged_in_dt"] = last_login_dt
                details["contactable"] = bool(contactable_val)
                logger.debug(f"Fetched profile details for {tester_profile_id_for_api}: last_login={last_login_dt}, contactable={details['contactable']}")
            # *** END CORRECTED PARSING ***
            else:
                # Log if the API call succeeded (200) but didn't return a dictionary
                logger.warning(f"Profile Details API for {tester_profile_id_for_api} returned successfully but response was not a dictionary or was empty. Response: {profile_response_data}")
                details["last_logged_in_dt"] = None
                details["contactable"] = False

        except Exception as e:
            logger.error(
                f"Error during profile details fetch for UUID {match_uuid} (ProfileID {tester_profile_id_for_api}) from {profile_details_url_final}: {e}",
                exc_info=True,
            )
            details["last_logged_in_dt"] = None
            details["contactable"] = False

    # Basic validation
    if not details.get("admin_profile_id") and details.get("admin_username") != details.get("tester_username"):
        logger.debug(
            f"Admin Profile ID missing/same as tester, but admin name differs for {match_uuid}."
        )

    logger.debug(f"Finished details fetch process for {match_uuid}.")
    return details
# end of _get_match_details_and_admin

# end of action6_gather.py