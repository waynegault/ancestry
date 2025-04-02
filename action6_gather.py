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
            # Pass DB session for Idea 3 (conditional relationship fetch)
            db_session_for_page = session_manager.get_db_conn()
            if not db_session_for_page:
                 logger.error(f"Could not get DB session for page {current_page_num} match pre-check. Skipping page.")
                 total_errors += 50 # Assume 50 errors if can't get DB session for checks
                 time.sleep(2)
                 continue

            try:
                matches_on_page = get_matches(session_manager, db_session_for_page, current_page_num)
            finally:
                 session_manager.return_session(db_session_for_page) # Ensure session is returned

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
    V12 REVISED: Processes a match integrating optimizations.
    1. Single DB Lookup (Person + DNA + Tree) by UUID.
    2. Determine needs (DNA create, Tree fetch, Person update) based on lookup.
    3. Conditionally fetch APIs based *only* on determined needs.
    4. Prepare data dicts.
    5. Call simplified DB write functions.
    6. Commit.
    """
    existing_person: Optional[Person] = None
    dna_match_record: Optional[DnaMatch] = None
    family_tree_record: Optional[FamilyTree] = None

    # --- Get initial data ---
    match_uuid = match.get("uuid")
    match_username = match.get("username")
    match_in_my_tree = match.get("in_my_tree", False) # Get flag early from initial match list
    log_ref = f"UUID={match_uuid or 'N/A'} User='{match_username or 'Unknown'}'"
    log_ref_short = f"UUID={match_uuid} User='{match_username}'"

    if not match_uuid:
        error_msg = f"Pre-check failed: Missing 'uuid' in match data: {match}"
        logger.error(error_msg)
        return "error", error_msg

    # --- Initialize status/flags/data ---
    overall_status: Literal["new", "updated", "skipped", "error"] = "error"
    dna_status: Literal["created", "skipped", "error"] = "skipped" # Default to skipped
    tree_status: Literal["created", "skipped", "error"] = "skipped" # Default to skipped
    person_status: Literal["created", "updated", "skipped", "error"] = "skipped" # Default to skipped

    create_dna_needed: bool = False
    fetch_tree_data: bool = False
    person_update_needed: bool = False # Flag if existing Person needs update
    is_new_person: bool = False # Flag if person needs creation

    # Data buckets
    full_person_data_to_save: Optional[Dict[str, Any]] = None
    dna_data_to_save: Optional[Dict[str, Any]] = None
    tree_data_to_save: Optional[Dict[str, Any]] = None

    # API Results
    details_fetched: Optional[Dict[str, Any]] = None
    tree_api_data_result: Optional[Dict[str, Any]] = None
    relationship_data_result: Optional[Dict[str, Any]] = None

    try:
        # --- Step 1: Single DB Lookup (Read-Only) ---
        logger.debug(f"{log_ref}: Performing initial DB lookup by UUID...")
        try:
            # Eager load everything needed for need determination
            existing_person = (
                session.query(Person)
                .options(joinedload(Person.dna_match), joinedload(Person.family_tree))
                .filter(Person.uuid == match_uuid.upper())
                .first()
            )

            if existing_person:
                 logger.debug(f"{log_ref}: Found existing Person ID {existing_person.id}. Evaluating needs.")
                 dna_match_record = existing_person.dna_match
                 family_tree_record = existing_person.family_tree
                 # Expire state *after* fetching initial related data for checks, before potential updates
                 session.expire(existing_person, ['dna_match', 'family_tree'])
            else:
                 logger.debug(f"{log_ref}: No existing person found by UUID.")
                 is_new_person = True

        except SQLAlchemyError as db_lookup_err:
             logger.error(f"Initial DB lookup failed for {log_ref_short}: {db_lookup_err}", exc_info=True)
             # Don't rollback here, it was read-only
             return "error", f"Initial DB lookup failed for {log_ref_short}"
        except Exception as lookup_err:
             logger.error(f"Unexpected error during initial DB lookup for {log_ref_short}: {lookup_err}", exc_info=True)
             return "error", f"Unexpected DB lookup error for {log_ref_short}"

        # --- Step 2: Determine Needs based on Lookup ---
        if is_new_person:
            create_dna_needed = True
            fetch_tree_data = match_in_my_tree # Fetch only if flag is initially True
            person_update_needed = False # It's a creation, not an update
        else: # Person exists
            # DNA Need
            create_dna_needed = (dna_match_record is None)

            # Tree Need (based on flag change or missing record when flag is true)
            db_in_my_tree = existing_person.in_my_tree # Access potentially expired attribute to reload
            # Condition 1: Flag changes False -> True
            if match_in_my_tree and not db_in_my_tree:
                 fetch_tree_data = True
            # Condition 2: Flag is True, but tree record missing
            elif match_in_my_tree and family_tree_record is None:
                 fetch_tree_data = True
            else: # Covers: Both False, Both True+Record Exists, True->False change
                 fetch_tree_data = False

            # Person Update Need (Check if specific fields might need updating)
            # We will fetch details API if DNA is needed, or if last_login/contactable *could* change
            # For simplicity now, let's assume we *might* need an update if the person exists.
            # We'll refine this based on which APIs are actually fetched.
            person_update_needed = True # Assume potential update needed initially if person exists

        logger.debug(f"{log_ref}: Needs Determined: NewPerson={is_new_person}, UpdatePerson={person_update_needed}, CreateDNA={create_dna_needed}, FetchTree={fetch_tree_data}")

        # --- Step 3: Conditionally Fetch API Data ---
        # Determine which APIs *must* be called
        fetch_details_api_needed = create_dna_needed or person_update_needed # Fetch details if creating DNA or potentially updating person
        fetch_tree_api_needed = fetch_tree_data # Tree API needed only if flag indicates

        logger.debug(f"{log_ref}: API Fetch Needed: Details={fetch_details_api_needed}, Tree={fetch_tree_api_needed}")

        # Use ThreadPoolExecutor only if multiple API calls are needed concurrently
        if fetch_details_api_needed and fetch_tree_api_needed:
            logger.debug(f"{log_ref}: Fetching Details and Tree data in parallel.")
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                # Submit Details Fetch Task
                delay_details = session_manager.dynamic_rate_limiter.wait()
                logger.debug(f"Waited {delay_details:.2f}s before submitting details fetch task for {log_ref_short}")
                futures[executor.submit(_get_match_details_and_admin, session_manager, match_uuid)] = "details"
                # Submit Tree Fetch Task
                delay_tree = session_manager.dynamic_rate_limiter.wait()
                logger.debug(f"Waited {delay_tree:.2f}s before submitting tree fetch task for {log_ref_short}")
                futures[executor.submit(_fetch_full_tree_details, session_manager, match)] = "tree" # Use new combined func

                for future in as_completed(futures):
                    task_type = futures[future]
                    try:
                        if task_type == "details":
                            details_fetched = future.result()
                            if not details_fetched: logger.error(f"CRITICAL: Parallel details fetch failed for {log_ref_short}.")
                        elif task_type == "tree":
                            # _fetch_full_tree_details returns a single dict or None
                            tree_fetch_result = future.result()
                            if tree_fetch_result:
                                 tree_api_data_result = tree_fetch_result # Assign combined result
                                 relationship_data_result = tree_fetch_result # Also assign for compatibility
                                 logger.debug(f"Tree fetch task completed successfully for {log_ref_short}.")
                            else:
                                 logger.warning(f"Tree fetch task completed but returned no data for {log_ref_short}.")
                                 fetch_tree_data = False # Reset flag if fetch failed
                    except Exception as exc:
                        logger.error(f"Exception in parallel task '{task_type}' for {log_ref_short}: {exc}", exc_info=True)
                        if task_type == "details": details_fetched = None # Mark as failed
                        elif task_type == "tree": fetch_tree_data = False # Reset flag

        elif fetch_details_api_needed: # Only Details needed
            logger.debug(f"{log_ref}: Fetching only Details data.")
            delay_details = session_manager.dynamic_rate_limiter.wait()
            logger.debug(f"Waited {delay_details:.2f}s before details fetch for {log_ref_short}")
            try:
                details_fetched = _get_match_details_and_admin(session_manager, match_uuid)
                if not details_fetched: logger.error(f"CRITICAL: Details fetch failed for {log_ref_short}.")
            except Exception as exc:
                 logger.error(f"Exception fetching details for {log_ref_short}: {exc}", exc_info=True)
                 details_fetched = None

        elif fetch_tree_api_needed: # Only Tree needed
            logger.debug(f"{log_ref}: Fetching only Tree data.")
            delay_tree = session_manager.dynamic_rate_limiter.wait()
            logger.debug(f"Waited {delay_tree:.2f}s before tree fetch for {log_ref_short}")
            try:
                tree_fetch_result = _fetch_full_tree_details(session_manager, match)
                if tree_fetch_result:
                     tree_api_data_result = tree_fetch_result
                     relationship_data_result = tree_fetch_result
                     logger.debug(f"Tree fetch task completed successfully for {log_ref_short}.")
                else:
                     logger.warning(f"Tree fetch task completed but returned no data for {log_ref_short}.")
                     fetch_tree_data = False # Reset flag
            except Exception as exc:
                 logger.error(f"Exception fetching tree data for {log_ref_short}: {exc}", exc_info=True)
                 fetch_tree_data = False # Reset flag

        # --- Step 4: Prepare Final Person Data ---
        # Only prepare if we are creating OR updating
        if is_new_person or person_update_needed:
            # Determine profile/admin IDs and message link (prioritize fetched data)
            tester_profile_id = details_fetched.get("tester_profile_id") if details_fetched else match.get("profile_id") # Fallback to initial hint
            admin_profile_id = details_fetched.get("admin_profile_id") if details_fetched else match.get("administrator_profile_id_hint")
            admin_username = details_fetched.get("admin_username") if details_fetched else match.get("administrator_username_hint")

            person_profile_id_to_save = None
            person_admin_id_to_save = None
            person_admin_username_to_save = None

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
                    person_admin_id_to_save = None
                    person_admin_username_to_save = None
            elif tester_profile_id and not admin_profile_id:
                person_profile_id_to_save = tester_profile_id
                person_admin_id_to_save = None
                person_admin_username_to_save = None
            else: # Neither tester nor admin ID reliably determined or only admin ID exists
                person_profile_id_to_save = None
                person_admin_id_to_save = admin_profile_id # Save admin if it exists
                person_admin_username_to_save = admin_username

            message_target_id = person_admin_id_to_save or person_profile_id_to_save
            constructed_message_link = None
            if message_target_id and session_manager.my_uuid:
                constructed_message_link = urljoin(config_instance.BASE_URL, f"/messaging/?p={message_target_id.upper()}&testguid1={session_manager.my_uuid.upper()}&testguid2={match_uuid.upper()}")

            # Assemble the complete data dictionary, using fetched data where available
            full_person_data_to_save = {
                "uuid": match_uuid.upper(),
                "username": match_username,
                "profile_id": person_profile_id_to_save.upper() if person_profile_id_to_save else None,
                "administrator_profile_id": person_admin_id_to_save.upper() if person_admin_id_to_save else None,
                "administrator_username": person_admin_username_to_save,
                "in_my_tree": match_in_my_tree, # Use the flag from initial match list
                "first_name": match.get("first_name"),
                "last_logged_in": details_fetched.get("last_logged_in_dt") if details_fetched else None,
                "contactable": details_fetched.get("contactable", False) if details_fetched else False,
                "gender": details_fetched.get("gender") if details_fetched else None,
                "message_link": constructed_message_link,
                "birth_year": tree_api_data_result.get("their_birth_year") if tree_api_data_result else None,
            }
            logger.debug(f"{log_ref}: Final Person data prepared: {full_person_data_to_save}")
        else:
             logger.debug(f"{log_ref}: No person creation or update needed based on initial checks.")


        # --- Step 5: Perform DB Writes (Person, DNA, Tree) ---
        person_record_for_relations = existing_person # Use existing if found

        # 5a. Create or Update Person
        if is_new_person or person_update_needed:
            if not full_person_data_to_save:
                 logger.error(f"{log_ref}: Logic error - Person create/update needed but data not prepared.")
                 return "error", f"Person data preparation failed for {log_ref_short}"

            try:
                # Call V9 simplified function
                person_record_for_relations, person_status = create_or_update_person(
                    session, full_person_data_to_save, existing_person=existing_person
                )

                if person_record_for_relations is None or person_status == "error":
                    logger.error(f"Person create/update DB write failed for {log_ref_short}.")
                    if session.is_active: session.rollback() # Rollback on direct failure
                    return "error", f"Person DB write failed for {log_ref_short}"
                logger.debug(f"{log_ref}: Person DB write status: {person_status}")

            except Exception as p_final_err:
                logger.error(f"Unexpected error during person DB write for {log_ref_short}: {p_final_err}", exc_info=True)
                if session.is_active: session.rollback()
                return "error", f"Unexpected person DB write error for {log_ref_short}"
        else:
            # Person existed and no update fields triggered a change
            person_status = "skipped"
            logger.debug(f"{log_ref}: Person DB write skipped (no changes detected).")

        # Ensure we have a person ID for relations if needed
        if person_record_for_relations is None and (create_dna_needed or fetch_tree_data):
             logger.error(f"{log_ref}: Cannot create relations - Person record is missing after create/update/skip step.")
             # If status wasn't error before, make it error now. Rollback likely needed.
             if session.is_active and person_status != "error": session.rollback()
             return "error", f"Person record missing for relations for {log_ref_short}"

        # 5b. Create DNA Record (Conditional)
        if create_dna_needed:
            if not details_fetched:
                 logger.error(f"{log_ref}: Cannot create DNA Match - required details were not fetched or fetch failed.")
                 # Mark overall as error, rollback likely needed
                 if session.is_active: session.rollback()
                 return "error", f"Missing details for DNA creation for {log_ref_short}"

            logger.debug(f"{log_ref}: Proceeding with DNA Match creation.")
            pred_rel = match.get("predicted_relationship", "N/A") # Use relationship from initial list
            dna_data_to_save = {
                "people_id": person_record_for_relations.id,
                "compare_link": match.get("compare_link"), # From initial list
                "cM_DNA": match.get("cM_DNA"), # From initial list
                "predicted_relationship": pred_rel,
                "uuid": match_uuid.upper(), # From initial list
                # --- Fields from details API ---
                "shared_segments": details_fetched.get("shared_segments"),
                "longest_shared_segment": details_fetched.get("longest_shared_segment"),
                "meiosis": details_fetched.get("meiosis"),
                "from_my_fathers_side": details_fetched.get("from_my_fathers_side", False),
                "from_my_mothers_side": details_fetched.get("from_my_mothers_side", False),
            }
            dna_status = create_dna_match(session, dna_data_to_save)
            if dna_status == "error":
                 logger.error(f"Failed to create DNA match for {log_ref_short}. Rolling back.")
                 if session.is_active: session.rollback()
                 return "error", f"DNA match creation failed for {log_ref_short}"
            elif dna_status == "skipped":
                 # This case indicates a race condition or logic error if create_dna_needed was True
                 logger.warning(f"DNA match creation needed but skipped (DB state inconsistent?) for {log_ref_short}.")
            else: # created
                 logger.debug(f"{log_ref}: DNA Match created successfully (staged).")
        else:
            logger.debug(f"{log_ref}: Skipping DNA Match creation (not needed).")


        # 5c. Create Tree Record (Conditional)
        if fetch_tree_data:
            if not tree_api_data_result:
                 logger.warning(f"{log_ref}: Skipping Family Tree creation as tree API fetch failed or returned no data.")
                 # Don't treat as critical error, just skip tree part
            else:
                 logger.debug(f"{log_ref}: Proceeding with Family Tree creation/update.")
                 view_in_tree_link = None
                 facts_link = None
                 their_cfpid_final = tree_api_data_result.get("their_cfpid")
                 if their_cfpid_final and session_manager.my_tree_id:
                     base_tree_url = urljoin(config_instance.BASE_URL, f"/family-tree/person/tree/{session_manager.my_tree_id}/person/{their_cfpid_final}")
                     view_in_tree_link = urljoin(base_tree_url, "family")
                     facts_link = urljoin(base_tree_url, "facts")

                 tree_data_to_save = {
                     "people_id": person_record_for_relations.id,
                     "cfpid": their_cfpid_final,
                     "person_name_in_tree": tree_api_data_result.get("their_firstname", "Unknown"), # Use name from tree API
                     "facts_link": facts_link,
                     "view_in_tree_link": view_in_tree_link,
                     "actual_relationship": relationship_data_result.get("actual_relationship") if relationship_data_result else None,
                     "relationship_path": relationship_data_result.get("relationship_path") if relationship_data_result else None,
                 }
                 logger.debug(f"{log_ref}: Arguments prepared for create_family_tree: {tree_data_to_save}")
                 tree_status = create_family_tree(session, tree_data_to_save)
                 if tree_status == "error":
                      logger.error(f"Failed to create family tree for {log_ref_short}. Rolling back.")
                      if session.is_active: session.rollback()
                      return "error", f"Family tree creation failed for {log_ref_short}"
                 elif tree_status == "skipped":
                      logger.warning(f"Family tree creation needed but skipped (DB state inconsistent?) for {log_ref_short}.")
                 else: # created
                      logger.debug(f"{log_ref}: Family Tree created successfully (staged).")
        else: # fetch_tree_data was False initially
             logger.debug(f"{log_ref}: Skipping Family Tree creation (not needed).")

        # --- Step 6: Commit/Rollback ---
        # Determine final overall status based on individual component statuses
        if person_status == "error" or dna_status == "error" or tree_status == "error":
             overall_status = "error"
             # Rollback should have happened within the specific error handling block
             error_msg = f"Processing error for {log_ref_short} (P:{person_status}, D:{dna_status}, T:{tree_status}). Transaction rolled back."
             logger.warning(error_msg)
             return overall_status, error_msg # Return error
        elif person_status == "created" or dna_status == "created" or tree_status == "created":
             overall_status = "new" if person_status == "created" else "updated"
             commit_needed = True
        elif person_status == "updated": # Person updated, but no new DNA/Tree
             overall_status = "updated"
             commit_needed = True
        else: # All components were skipped
             overall_status = "skipped"
             commit_needed = False

        # Commit if any changes were staged
        if commit_needed:
            try:
                commit_log_ref = f"{log_ref_short} (Person ID: {person_record_for_relations.id if person_record_for_relations else 'N/A'})"
                logger.debug(f"Attempting commit for {commit_log_ref} (Overall Status: {overall_status}).")
                session.commit()
                logger.debug(f"Commit successful for {commit_log_ref}.")
            except (IntegrityError, SQLAlchemyError) as commit_e:
                 error_msg = f"Database Commit FAILED for {log_ref_short}"
                 logger.error(f"{error_msg}: {commit_e}", exc_info=True)
                 if session.is_active: session.rollback()
                 return "error", error_msg
            except Exception as E:
                 error_msg = f"Unexpected commit error for {log_ref_short}"
                 logger.error(f"{error_msg}: {E}", exc_info=True)
                 if session.is_active: session.rollback()
                 return "error", error_msg
        else:
             logger.debug(f"{log_ref}: No commit needed (overall status: skipped).")

        # --- Step 7: Return final status ---
        logger.debug(f"Final overall status for {log_ref_short}: {overall_status}")
        return overall_status, None

    # --- Exception Handling (Outer Catch) ---
    except Exception as e:
        error_msg = f"Unexpected critical error in _do_match for {log_ref}: {e}."
        logger.error(error_msg, exc_info=True)
        if session and session.is_active:
            try: session.rollback(); logger.debug(f"Rolled back session for {log_ref} due to critical error.")
            except Exception as rb_err: logger.error(f"Error rolling back session for {log_ref} after critical error: {rb_err}")
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
    session_manager: SessionManager, db_session: Session, current_page: int = 1
) -> List[Dict[str, Any]]:
    """
    REVISED: Fetches and processes match list data for a SINGLE page.
    - Includes administrator hints.
    - Fetches predicted relationships CONDITIONALLY based on DB check.
    - Parses last login date from profile details API only (removed initial attempt).
    - Forces use of requests library for the main match list API call.
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
    predicted_relationships: Dict[str, str] = {} # Initialize empty

    try:
        # --- 1. Fetch Match List Data (No change here) ---
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
            logger.warning(f"No response received from match list API for page {current_page} (using requests).")
            return []
        if isinstance(api_response, str):
            logger.warning(f"Received string instead of dict from match list API for page {current_page} (using requests): '{api_response[:100]}...'")
            return []
        if not isinstance(api_response, dict):
            logger.warning(f"Unexpected data type received from match list API for page {current_page} (using requests). Type: {type(api_response)}.")
            logger.debug(f"Response data: {api_response}")
            return []

        match_data_list = api_response.get("matchList", [])
        if not match_data_list:
            logger.info(f"No matches found in 'matchList' for page {current_page}.")
            return []
        logger.debug(f"Got {len(match_data_list)} raw matches from API on page {current_page}.")

        # Filter raw matches for essential 'sampleId'
        valid_matches_for_processing: List[Dict[str, Any]] = []
        skipped_sampleid_count = 0
        for m in match_data_list:
            if isinstance(m, dict) and m.get("sampleId"):
                valid_matches_for_processing.append(m)
            else:
                skipped_sampleid_count += 1
                logger.warning(f"Skipping raw match due to missing 'sampleId' on page {current_page}. Data: {m}")
        if skipped_sampleid_count > 0:
            logger.warning(f"Skipped {skipped_sampleid_count} raw matches on page {current_page} due to missing 'sampleId'.")
        if not valid_matches_for_processing:
            logger.warning(f"No matches with valid 'sampleId' found on page {current_page}.")
            return []

        sample_ids_on_page = [match["sampleId"].upper() for match in valid_matches_for_processing]
        sample_ids_set = set(sample_ids_on_page) # Set for efficient lookup

        # --- In-Tree Status Check (No change here) ---
        in_tree_ids: Set[str] = set()
        cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids_on_page))}" # Use page-specific key
        cached_in_tree = session_manager.cache.get(cache_key_tree)
        if cached_in_tree is not None and isinstance(cached_in_tree, set):
            in_tree_ids = cached_in_tree
            logger.debug(f"Loaded {len(in_tree_ids)} in-tree IDs from cache for page {current_page}.")
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
                referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
            )
            if isinstance(response_in_tree, list):
                in_tree_ids = {item.upper() for item in response_in_tree if isinstance(item, str)}
                session_manager.cache.set(cache_key_tree, in_tree_ids, timeout=config_instance.CACHE_TIMEOUT)
                logger.debug(f"Fetched/cached {len(in_tree_ids)} in-tree IDs for page {current_page}.")
            elif response_in_tree is None:
                logger.warning(f"In-Tree Status Check API failed for page {current_page}.")
            else:
                logger.error(f"Unexpected in-tree status response format for page {current_page}: {type(response_in_tree)}")


        # --- MODIFIED: Conditional Predicted Relationship Processing ---
        logger.debug(f"Checking DB for existing DNA records for {len(sample_ids_on_page)} matches on page {current_page}...")
        try:
            # Batch query to find existing Person IDs linked to DNA matches for this page's UUIDs
            existing_dna_people_ids = set(
                db_session.query(DnaMatch.people_id)
                .join(Person, DnaMatch.people_id == Person.id)
                .filter(Person.uuid.in_(sample_ids_on_page))
                .distinct()
                .all()
            )
             # Flatten the list of tuples
            existing_dna_people_ids_flat = {pid[0] for pid in existing_dna_people_ids}


            # Determine which UUIDs *might* need relationship fetching (those without existing DNA record)
            # Note: This still requires fetching Person records to link UUID to people_id, slightly complex.
            # Alternative: Simpler check - just fetch Person.uuid for existing DnaMatch records?
            # Let's try the simpler alternative first:

            existing_dna_match_uuids = set(
                 row[0] for row in db_session.query(Person.uuid)
                 .join(DnaMatch, Person.id == DnaMatch.people_id)
                 .filter(Person.uuid.in_(sample_ids_on_page))
                 .distinct()
                 .all()
            )

            sample_ids_needing_relationship_fetch = sample_ids_set - existing_dna_match_uuids
            logger.debug(f"Found {len(existing_dna_match_uuids)} existing DNA matches in DB for this page.")
            logger.debug(f"Will fetch predicted relationships for {len(sample_ids_needing_relationship_fetch)} matches.")

        except SQLAlchemyError as db_err:
            logger.error(f"Database error checking existing DNA matches for page {current_page}: {db_err}", exc_info=True)
            # If DB check fails, cautiously fetch for all matches on the page
            sample_ids_needing_relationship_fetch = sample_ids_set
            logger.warning("Fetching relationships for all matches due to DB check error.")
        except Exception as e:
             logger.error(f"Unexpected error checking existing DNA matches: {e}", exc_info=True)
             sample_ids_needing_relationship_fetch = sample_ids_set
             logger.warning("Fetching relationships for all matches due to unexpected error.")

        # Fetch relationships only for those needing it
        if sample_ids_needing_relationship_fetch:
            logger.debug("Getting predicted relationships for needed matches...")
            @retry()
            def process_sample(sample_id: str) -> tuple[str, str]:
                """Fetches relationship probability (inner function, unchanged)."""
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
                    response_rel = _api_req(
                        url=rel_url,
                        driver=session_manager.driver,
                        session_manager=session_manager,
                        method="POST",
                        json_data={},
                        use_csrf_token=True,
                        api_description="Match Probability API",
                        referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
                    )
                    if response_rel is None:
                        raise requests.exceptions.RequestException(f"Match Probability API req failed for {sample_id}")
                    if not isinstance(response_rel, dict) or "matchProbabilityToSampleId" not in response_rel:
                        logger.warning(f"Invalid data format from Match Probability API for {sample_id_upper}. Resp: {response_rel}")
                        return sample_id_upper, "N/A (Invalid Data)"
                    prob_data = response_rel["matchProbabilityToSampleId"]
                    predictions = prob_data.get("relationships", {}).get("predictions", [])
                    if not predictions:
                        logger.debug(f"No relationship predictions found for {sample_id_upper}. Marking as Distant.")
                        return sample_id_upper, "Distant relationship?"
                    valid_preds = [p for p in predictions if isinstance(p, dict) and "distributionProbability" in p and "pathsToMatch" in p]
                    if not valid_preds:
                        logger.warning(f"No valid prediction paths found for {sample_id_upper}.")
                        return sample_id_upper, "N/A (No Valid Paths)"
                    best_pred = max(valid_preds, key=lambda x: x.get("distributionProbability", 0.0))
                    top_prob = best_pred.get("distributionProbability", 0.0)
                    paths = best_pred.get("pathsToMatch", [])
                    labels = [p.get("label") for p in paths if isinstance(p, dict) and p.get("label")]
                    if not labels:
                        logger.warning(f"Prediction found for {sample_id_upper}, but no labels in paths.")
                        return sample_id_upper, f"N/A (No Labels) [{top_prob:.1f}%]"
                    final_labels = labels[:MAX_LABELS_TO_SHOW]
                    relationship_str = " or ".join(map(str, final_labels))
                    return sample_id_upper, f"{relationship_str} [{top_prob:.1f}%]"
                except requests.exceptions.RequestException as req_e:
                    logger.warning(f"RequestException processing relationship for {sample_id}: {req_e}. Retrying...")
                    raise req_e
                except Exception as e:
                    logger.error(f"Unexpected error processing relationship for {sample_id}: {e}", exc_info=False)
                    logger.debug(f"Traceback:", exc_info=True)
                    return sample_id.upper(), "N/A (Error)"
            # End inner function process_sample

            skipped_rel_count = 0
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_sid = {
                    executor.submit(process_sample, sid): sid.upper() for sid in sample_ids_needing_relationship_fetch
                }
                for future in as_completed(future_to_sid):
                    sid_upper = future_to_sid[future]
                    try:
                        _, rel_str = future.result()
                        predicted_relationships[sid_upper] = rel_str
                    except Exception as exc:
                        logger.error(f"Relationship future for {sid_upper} failed after retries: {exc}")
                        predicted_relationships[sid_upper] = "N/A (Future Error)"
                        skipped_rel_count += 1
            logger.debug(f"Fetched {len(predicted_relationships)} needed predicted relationships ({skipped_rel_count} errors).")
        else:
             logger.debug("No new predicted relationships need fetching for this page (all have existing DNA records).")
        # --- END MODIFIED Relationship Fetch ---


        # --- Compile Final Refined Match Data ---
        refined_matches: List[Dict[str, Any]] = []
        skipped_profile_id_count = 0
        # Removed skipped_last_login_count as it's now fetched later

        for match in valid_matches_for_processing:
            profile = match.get("matchProfile", {})
            relationship = match.get("relationship", {})
            sample_id_upper = match["sampleId"].upper()

            profile_user_id = profile.get("userId")
            match_username = profile.get("displayName", "Unknown").title()

            admin_profile_id_hint = match.get("adminId")
            admin_username_hint = match.get("adminName")

            # --- REMOVED lastLoginDate parsing here ---
            # last_login_dt is now fetched in _get_match_details_and_admin

            if not profile_user_id:
                logger.debug(f"Match '{match_username}' (UUID: {sample_id_upper}) missing tester 'profile_id' in initial list. Will attempt fetch via details API.")
                skipped_profile_id_count += 1
                profile_user_id_upper = None
            else:
                profile_user_id_upper = str(profile_user_id).upper()

            compare_link = urljoin(
                config_instance.BASE_URL,
                f"discoveryui-matches/compare/{my_uuid.upper()}/with/{sample_id_upper}",
            )
            first_name = match_username.split()[0] if match_username != "Unknown" else None

            # Get predicted relationship (either newly fetched or default 'N/A' if skipped)
            pred_rel_value = predicted_relationships.get(sample_id_upper, "N/A (DB Exists)")

            refined_match_data = {
                "username": match_username,
                "first_name": first_name,
                "initials": profile.get("displayInitials", "??").upper(),
                "gender": match.get("gender"), # Initial hint
                "profile_id": profile_user_id_upper, # Initial hint
                "uuid": sample_id_upper,
                "administrator_profile_id_hint": admin_profile_id_hint,
                "administrator_username_hint": admin_username_hint,
                "photoUrl": profile.get("photoUrl", ""),
                "cM_DNA": int(relationship.get("sharedCentimorgans", 0)),
                "numSharedSegments": int(relationship.get("numSharedSegments", 0)), # Keep original name if needed elsewhere
                "compare_link": compare_link,
                "message_link": None, # Set later
                "predicted_relationship": pred_rel_value, # Use fetched or default
                "in_my_tree": sample_id_upper in in_tree_ids,
                "createdDate": match.get("createdDate"),
                # REMOVED: "last_logged_in_dt": last_login_dt,
            }
            refined_matches.append(refined_match_data)

        total_raw = len(match_data_list)
        total_valid_uuid = len(valid_matches_for_processing)
        final_count = len(refined_matches)
        logger.debug(
            f"Processed page {current_page}: Raw={total_raw}, ValidUUID={total_valid_uuid}, Refined={final_count} (MissingTesterProfileID={skipped_profile_id_count})"
        )

        logger.debug(f"Refined matches being returned from get_matches (Page {current_page}):")
        for i, rm in enumerate(refined_matches):
             logger.debug(f"  Match {i+1}: User='{rm.get('username')}', PredRel='{rm.get('predicted_relationship')}'")

        return refined_matches

    except requests.exceptions.RequestException as e:
        logger.error(f"Network/Request error processing page {current_page}: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.critical(f"Critical error processing match data for page {current_page}: {e}", exc_info=True)
        return []
# end get_matches


def _fetch_full_tree_details(
    session_manager: SessionManager, match_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    V1 REVISED: Fetches tree badge details and then relationship data sequentially.
    Combines logic of _get_tree and _get_relShip. Handles rate limiting.

    Args:
        session_manager: The active SessionManager instance.
        match_data: Dictionary containing match data ('uuid', 'username').

    Returns:
        A dictionary containing combined tree and relationship data, or None if failed.
        Keys include: 'their_cfpid', 'their_firstname', 'their_lastname',
                      'their_birth_year', 'actual_relationship', 'relationship_path'.
    """
    tree_api_data = None
    relationship_data = None
    log_ref_short = f"UUID={match_data.get('uuid')} User='{match_data.get('username', 'Unknown')}' (Full Tree Fetch)"
    my_uuid = session_manager.my_uuid
    their_uuid = match_data.get("uuid")
    username = match_data.get("username", "Unknown")

    if not their_uuid or not my_uuid:
        logger.warning(f"_fetch_full_tree_details: Missing their_uuid or my_uuid for {username}: {match_data}")
        return None

    try:
        # --- 1. Fetch Tree Badge Details ---
        delay_badge = session_manager.dynamic_rate_limiter.wait()
        logger.debug(f"Waited {delay_badge:.2f}s before fetching tree badge details for {log_ref_short}")

        badge_url = urljoin(
            config_instance.BASE_URL,
            f"/discoveryui-matchesservice/api/samples/{my_uuid}/matches/{their_uuid}/badgedetails",
        )
        badge_api_response = _api_req(
            url=badge_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description="Badge Details API",
            referer_url=urljoin(config_instance.BASE_URL, "/discoveryui-matches/list/"),
        )

        logger.debug(f"Badge Details raw API response for {log_ref_short}:\n{badge_api_response}")

        if badge_api_response and isinstance(badge_api_response, dict):
            person_badged = badge_api_response.get("personBadged", {})
            full_firstname = person_badged.get("firstName", "Unknown")
            words = full_firstname.strip().split()
            their_firstname = words[0] if words else "Unknown"

            tree_api_data = {
                "their_cfpid": person_badged.get("personId"),
                "their_firstname": their_firstname,
                "their_lastname": person_badged.get("lastName", "Unknown"),
                "their_birth_year": person_badged.get("birthYear"),
            }
            logger.debug(f"Parsed badge data for {log_ref_short}: {tree_api_data}")
        else:
            logger.warning(f"Badge Details API returned non-dict or empty response for {log_ref_short}. Type: {type(badge_api_response)}")
            return None # Cannot proceed without badge details

        # --- 2. Fetch Relationship Ladder Details (if badge details successful) ---
        their_cfpid = tree_api_data.get("their_cfpid")
        my_tree_id = session_manager.my_tree_id

        if their_cfpid and my_tree_id:
            delay_ladder = session_manager.dynamic_rate_limiter.wait()
            logger.debug(f"Waited {delay_ladder:.2f}s before fetching relationship ladder for {log_ref_short}")
            logger.debug(f"Fetching relationship details for tree {my_tree_id}, cfpid {their_cfpid}.")

            # Construct API URL and dynamic Referer
            ladder_api_url = urljoin(
                config_instance.BASE_URL,
                f"family-tree/person/tree/{my_tree_id}/person/{their_cfpid}/getladder?callback=jQuery",
            )
            dynamic_referer = urljoin(
                config_instance.BASE_URL, f"family-tree/person/tree/{my_tree_id}/person/{their_cfpid}/facts"
            )

            # Make API call using _api_req
            response_text = _api_req(
                url=ladder_api_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                headers={"Accept": "*/*"},
                use_csrf_token=False,
                api_description="Get Ladder API",
                referer_url=dynamic_referer,
            )

            # Process the JSONP response
            if response_text and isinstance(response_text, str):
                match_jsonp = re.match(r"^[^(]*\((.*)\)[^)]*$", response_text, re.DOTALL | re.IGNORECASE)
                if match_jsonp:
                    json_string = match_jsonp.group(1).strip()
                    try:
                        ladder_data = json.loads(json_string)
                        if isinstance(ladder_data, dict) and "html" in ladder_data:
                            html_content = ladder_data["html"]
                            if html_content:
                                soup = BeautifulSoup(html_content, "html.parser")
                                actual_relationship_text = None
                                relationship_path_text = None

                                # Extract Actual Relationship
                                rel_elem = soup.select_one('ul.textCenter > li:first-child > i > b')
                                if rel_elem:
                                    raw_relationship = rel_elem.get_text(strip=True)
                                    actual_relationship_text = ordinal_case(raw_relationship.title())
                                else:
                                    logger.warning(f"Could not extract actual_relationship from HTML for cfpid: {their_cfpid}")

                                # Extract Relationship Path
                                path_items = soup.select('ul.textCenter > li:not([class*="iconArrowDown"])')
                                path_list = []
                                num_items = len(path_items)
                                for i, item in enumerate(path_items):
                                    name_text = ""
                                    desc_text = ""
                                    raw_name_extracted = ""
                                    name_link = item.find("a")
                                    name_bold = item.find("b") if not name_link else None
                                    if name_link:
                                        nested_b = name_link.find("b")
                                        raw_name_extracted = (nested_b.get_text(strip=True) if nested_b else name_link.get_text(strip=True))
                                    elif name_bold: raw_name_extracted = name_bold.get_text(strip=True)
                                    else:
                                        parts = item.get_text(separator="\n", strip=True).split("\n")
                                        if parts: raw_name_extracted = parts[0]
                                    if raw_name_extracted:
                                        cleaned_name = " ".join(raw_name_extracted.replace('"', "'").split())
                                        name_text = format_name(cleaned_name)
                                    if i > 0:
                                        desc_element = item.find("i")
                                        if desc_element:
                                            raw_desc_full = desc_element.get_text(strip=True)
                                            cleaned_desc_full = raw_desc_full.replace('"', "'")
                                            if i == num_items - 1 and cleaned_desc_full.lower().startswith("you are the "):
                                                relationship_part = cleaned_desc_full[len("You are the ") :].strip()
                                                desc_text = format_name(relationship_part)
                                            else:
                                                match_rel = re.match(r"^(.*?)\s+of\s+(.*)$", cleaned_desc_full, re.IGNORECASE)
                                                if match_rel:
                                                    relation_part = match_rel.group(1).strip()
                                                    name_part = match_rel.group(2).strip()
                                                    formatted_name_part = format_name(name_part)
                                                    desc_text = f"{relation_part.capitalize()} of {formatted_name_part}"
                                                else:
                                                    desc_text = format_name(cleaned_desc_full)
                                    if name_text:
                                        list_item = f"{name_text} ({desc_text})" if desc_text else name_text
                                        path_list.append(list_item)
                                if path_list:
                                    relationship_path_text = "\n↓\n".join(path_list)
                                else:
                                    logger.warning(f"Could not construct relationship_path for cfpid {their_cfpid}.")

                                # Store results
                                relationship_data = {
                                    "actual_relationship": actual_relationship_text,
                                    "relationship_path": relationship_path_text,
                                }
                                log_rel = actual_relationship_text if actual_relationship_text else "N/A"
                                log_path_len = len(relationship_path_text) if relationship_path_text else 0
                                logger.debug(f"Got relationship details for cfpid {their_cfpid}: Actual='{log_rel}', Path estimated length: {log_path_len}")

                            else:
                                logger.warning(f"HTML content in getladder response is empty for cfpid {their_cfpid}.")
                        else:
                            logger.warning(f"Unexpected structure in getladder JSON (missing 'html') for cfpid {their_cfpid}.")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON from getladder for cfpid {their_cfpid}: {e}.")
                else:
                    logger.error(f"Could not parse JSONP response format for cfpid {their_cfpid}. Regex failed.")
            elif response_text is None:
                 logger.error(f"Failed to get relationship data for cfpid {their_cfpid}: _api_req returned None.")
            else:
                 logger.error(f"Unexpected response type from _api_req for cfpid {their_cfpid}: {type(response_text)}. Expected string.")

        elif not their_cfpid:
             logger.warning(f"CFPID missing from badge details result for {log_ref_short}. Cannot fetch relationship.")
        elif not my_tree_id:
             logger.warning(f"my_tree_id missing. Cannot fetch relationship for {log_ref_short}.")

        # --- Combine results ---
        if tree_api_data:
            if relationship_data:
                 tree_api_data.update(relationship_data) # Add relationship keys to tree_api_data dict
            return tree_api_data # Return combined dict
        else:
            return None # Return None if tree_api_data failed

    except Exception as e:
        logger.error(f"Error in _fetch_full_tree_details for {log_ref_short}: {e}", exc_info=True)
        return None
# End _fetch_full_tree_details


#################################################################################
# 4. Match Data Processing & Database Integration
#################################################################################





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

def _fetch_tree_and_relationship_data(
    session_manager: SessionManager, match_data: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Fetches tree badge details (_get_tree) and then relationship data (_get_relShip) sequentially.
    Applies rate limiter waits appropriately. Designed to be run in a separate thread.
    Returns a tuple: (tree_api_data, relationship_data)
    """
    tree_api_data = None
    relationship_data = None
    log_ref_short = f"UUID={match_data.get('uuid')} User='{match_data.get('username', 'Unknown')}' (Tree Fetch Thread)"

    try:
        # --- Wait and Fetch Tree Details ---
        # Apply wait before the first tree-related API call in this sequence
        delay_badge = session_manager.dynamic_rate_limiter.wait()
        logger.debug(f"Waited {delay_badge:.2f}s before fetching tree badge details for {log_ref_short}")
        # Note: _get_tree now needs a dummy session argument if we pass match_data instead of session
        # Let's keep passing session for now, assuming thread-safety of reading operations
        # or modify _get_tree later if needed. For now, assume session is passed correctly.
        # Correction: Pass session_manager instead of session to _get_tree as it's used for API calls.
        tree_api_data = _get_tree(None, match_data, session_manager) # Pass None for session, match_data has info

        if tree_api_data:
            their_cfpid = tree_api_data.get("their_cfpid")
            if their_cfpid and session_manager.my_tree_id:
                # --- Wait and Fetch Relationship Details ---
                # Apply wait before the second tree-related API call
                delay_ladder = session_manager.dynamic_rate_limiter.wait()
                logger.debug(f"Waited {delay_ladder:.2f}s before fetching relationship ladder for {log_ref_short}")
                relationship_data = _get_relShip(session_manager, session_manager.my_tree_id, their_cfpid)
            elif not their_cfpid:
                 logger.warning(f"CFPID missing from _get_tree result for {log_ref_short}. Cannot fetch relationship.")
            elif not session_manager.my_tree_id:
                 logger.warning(f"my_tree_id missing. Cannot fetch relationship for {log_ref_short}.")
        else:
            logger.warning(f"_get_tree returned None for {log_ref_short}.")

    except Exception as e:
        logger.error(f"Error in parallel tree fetch thread for {log_ref_short}: {e}", exc_info=True)
        # Return None for both if an error occurs

    return tree_api_data, relationship_data
# End _fetch_tree_and_relationship_data

# end of action6_gather.py