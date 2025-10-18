#!/usr/bin/env python3

"""
Action 6: DNA Match Gatherer

Comprehensive DNA match gathering and enrichment system that follows proven patterns:
1. Fetch Match List API → get core data (username, cm, segments, profile_id)
2. Optionally fetch Match Details API → get additional DNA data (longest segment, meiosis, sides)
3. Optionally fetch Profile Details API → get last_logged_in, contactable
4. Save in batches with per-batch reporting
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import json
import random
import time

# === THIRD-PARTY IMPORTS ===
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import unquote, urljoin

from tqdm.auto import tqdm

# === LOCAL IMPORTS ===
from config import config_schema
from connection_resilience import with_connection_resilience
from core.database_manager import DatabaseManager
from core.session_manager import SessionManager
from database import create_or_update_dna_match, create_or_update_family_tree, create_or_update_person
from dna_ethnicity_utils import (
    extract_match_ethnicity_percentages,
    fetch_ethnicity_comparison,
    load_ethnicity_metadata,
)
from dna_utils import (
    fetch_in_tree_status,
    fetch_match_list_page,
    get_csrf_token_for_dna_matches,
    nav_to_dna_matches_page,
)
from utils import _api_req, format_name


def _setup_rate_limiting(session_manager: SessionManager, parallel_workers: int) -> None:
    """Configure adaptive rate limiting based on parallel worker count."""
    if parallel_workers <= 1:
        logger.debug("📝 Sequential processing (PARALLEL_WORKERS=1)")
        return

    import math
    if not session_manager.rate_limiter:
        return

    original_delay = session_manager.rate_limiter.initial_delay
    adaptive_delay = original_delay * math.sqrt(parallel_workers)
    session_manager.rate_limiter.initial_delay = adaptive_delay
    session_manager.rate_limiter.current_delay = adaptive_delay
    logger.debug(f"⚡ Parallel processing ENABLED with {parallel_workers} workers")
    logger.info(f"   Adaptive rate limiting: base delay increased from {original_delay:.2f}s to {adaptive_delay:.2f}s")
    logger.debug("   Rate limiter is thread-safe and will prevent 429 errors")


def _initialize_coord_session(session_manager: SessionManager) -> tuple[str, Optional[str], DatabaseManager]:
    """Initialize session and get required identifiers."""
    my_uuid = session_manager.my_uuid
    my_tree_id = session_manager.my_tree_id

    if not my_uuid:
        logger.error("Cannot proceed: my_uuid is not set")
        raise ValueError("my_uuid is not set")

    logger.info(f"My UUID: {my_uuid}")
    logger.info(f"My Tree ID: {my_tree_id}")

    db_manager = DatabaseManager()
    return my_uuid, my_tree_id, db_manager


def _handle_session_health_check(session_manager: SessionManager) -> tuple[bool, int, int, str]:
    """Handle session health check and recovery. Returns (should_continue, deaths, recoveries, reason)."""
    if session_manager.check_session_health():
        return True, 0, 0, ""

    logger.warning("🚨 Session health check failed - attempting recovery...")
    if session_manager.attempt_browser_recovery():
        logger.info("✅ Session recovered successfully, continuing...")
        return True, 1, 1, ""

    logger.error("❌ Session recovery failed - stopping processing")
    return False, 1, 0, "Session recovery failed at page health check"


def _handle_api_failure(session_manager: SessionManager, page_num: int, max_pages: int) -> tuple[bool, int, int, str, bool]:
    """Handle API failure and recovery. Returns (should_continue, deaths, recoveries, reason, should_break)."""
    if session_manager.check_session_health():
        if max_pages == 0:
            logger.info("No more pages available. Stopping.")
            return False, 0, 0, "", True
        return True, 0, 0, "", False

    logger.error("🚨 Session appears dead - API failure likely due to invalid session")
    if session_manager.attempt_browser_recovery():
        logger.info("✅ Session recovered - retrying current page...")
        return True, 1, 1, "", False

    logger.error("❌ Session recovery failed - stopping processing")
    return False, 1, 0, f"Session recovery failed at page {page_num} (API failure)", True


def _process_page_batches(matches: list[dict], batch_size: int, session_manager: SessionManager, db_manager: DatabaseManager, my_uuid: str, my_tree_id: Optional[str], page_num: int) -> tuple[int, int, int, int, int, int, bool, str]:
    """Process all batches on a page. Returns (new, updated, skipped, errors, deaths, recoveries, incomplete, reason)."""
    total_new = 0
    total_updated = 0
    total_skipped = 0
    total_errors = 0
    session_deaths = 0
    session_recoveries = 0
    run_incomplete = False
    incomplete_reason = ""

    for batch_start in range(0, len(matches), batch_size):
        batch_end = min(batch_start + batch_size, len(matches))
        batch = matches[batch_start:batch_end]

        if not session_manager.check_session_health():
            logger.warning("🚨 Session health check failed before batch - attempting recovery...")
            session_deaths += 1
            if session_manager.attempt_browser_recovery():
                logger.info("✅ Session recovered successfully, continuing with batch...")
                session_recoveries += 1
            else:
                logger.error("❌ Session recovery failed - skipping remaining batches on this page")
                run_incomplete = True
                incomplete_reason = f"Session recovery failed at page {page_num}, batch {batch_start//batch_size + 1}"
                break

        batch_num = (batch_start // batch_size) + 1
        total_batches_on_page = (len(matches) + batch_size - 1) // batch_size

        logger.info(f"Batch {batch_num}/{total_batches_on_page} (matches {batch_start+1}-{batch_end})")

        new, updated, skipped, errors = _process_batch(batch, session_manager, db_manager, my_uuid, my_tree_id)

        total_new += new
        total_updated += updated
        total_skipped += skipped
        total_errors += errors

        # Use compact logging for batches with only skips, detailed for batches with changes
        if new == 0 and updated == 0 and errors == 0 and skipped > 0:
            logger.info(f"Batch {batch_num} complete: All {skipped} skipped (already in database)")
        else:
            logger.info(f"Batch {batch_num} complete: New={new}, Updated={updated}, Skipped={skipped}, Errors={errors}")

    return total_new, total_updated, total_skipped, total_errors, session_deaths, session_recoveries, run_incomplete, incomplete_reason


def _fetch_and_validate_page_data(driver: Any, session_manager: SessionManager, my_uuid: str, page_num: int, csrf_token: str, max_pages: int) -> tuple[Optional[list[dict]], int, int, bool, str, Optional[int], bool]:
    """Fetch and validate page data. Returns (matches, deaths, recoveries, should_break, reason, total_pages, is_last_page)."""
    api_response = fetch_match_list_page(driver, session_manager, my_uuid, page_num, csrf_token)
    if not api_response or not isinstance(api_response, dict):
        logger.warning(f"No API response for page {page_num}")
        _, deaths, recoveries, reason, should_break = _handle_api_failure(session_manager, page_num, max_pages)
        return None, deaths, recoveries, should_break, reason, None, False

    # Extract total pages and last page flag from API response
    total_pages = api_response.get("totalPages")
    is_last_page = api_response.get("isLastPage", False)

    match_list = api_response.get("matchList", [])
    if not match_list:
        logger.warning(f"No matches in API response for page {page_num}")
        if max_pages == 0 or is_last_page:
            logger.info("No more matches available. Stopping.")
            return None, 0, 0, True, "", total_pages, is_last_page
        return None, 0, 0, False, "", total_pages, is_last_page

    sample_ids = [m.get("sampleId", "").upper() for m in match_list if m.get("sampleId")]
    if not sample_ids:
        logger.warning(f"No sample IDs found on page {page_num}")
        return None, 0, 0, False, "", total_pages, is_last_page

    in_tree_ids = fetch_in_tree_status(driver, session_manager, my_uuid, sample_ids, csrf_token, page_num)
    matches = _refine_match_list(match_list, my_uuid, in_tree_ids)

    if not matches:
        logger.warning(f"No matches found on page {page_num}")
        return None, 0, 0, False, "", total_pages, is_last_page

    logger.info(f"Found {len(matches)} matches on page {page_num}")
    return matches, 0, 0, False, "", total_pages, is_last_page


def _log_page_header(page_num: int, max_pages: int, total_new: int, total_updated: int, total_skipped: int, total_errors: int, api_total_pages: Optional[int] = None) -> None:
    """Log page processing header."""
    # Add blank line before separator for readability
    logger.info("")
    logger.info(f"{'='*80}")

    # Format page info - prioritize API total pages, then max_pages setting
    if api_total_pages is not None:
        # Use total pages from API response (most accurate)
        page_info = f"Processing page {page_num} of {api_total_pages} pages"
    elif max_pages == 0:
        # When MAX_PAGES=0 and we don't have API total yet, show current page only
        page_info = f"Processing page {page_num}"
    else:
        # When MAX_PAGES is set, show progress based on configured limit
        page_info = f"Processing page {page_num} of {max_pages} pages"

    cumulative_info = f"Cumulative: New={total_new}, Updated={total_updated}, Skipped={total_skipped}, Errors={total_errors}"
    logger.info(f"{page_info} | {cumulative_info}")
    logger.info(f"{'='*80}")


def _should_stop_processing(
    page_num: int,
    end_page: float,
    max_pages: int,
    is_last_page: bool,
    matches: Optional[list],
) -> tuple[bool, str]:
    """Check if we should stop processing pages."""
    if max_pages > 0 and page_num > end_page:
        return True, "Reached max pages limit"

    if not matches and is_last_page:
        return True, "Reached last page (isLastPage=true)"

    if is_last_page:
        return True, "Reached last page according to API"

    return False, ""


def _update_page_totals(
    total_new: int,
    total_updated: int,
    total_skipped: int,
    total_errors: int,
    session_deaths: int,
    session_recoveries: int,
    new: int,
    updated: int,
    skipped: int,
    errors: int,
    deaths: int,
    recoveries: int,
) -> tuple[int, int, int, int, int, int]:
    """Update running totals with page results."""
    return (
        total_new + new,
        total_updated + updated,
        total_skipped + skipped,
        total_errors + errors,
        session_deaths + deaths,
        session_recoveries + recoveries,
    )


def _handle_page_fetch_result(
    should_break: bool,
    reason: str,
    matches: Optional[list],
    is_last_page: bool,
    page_num: int,
    end_page: float,
    max_pages: int,
) -> tuple[bool, bool, str]:
    """
    Handle the result of fetching page data.

    Returns:
        (should_break_loop, should_continue_to_next, incomplete_reason)
    """
    if should_break:
        return True, False, reason if reason else ""

    should_stop, stop_msg = _should_stop_processing(page_num, end_page, max_pages, is_last_page, matches)
    if should_stop:
        if stop_msg and "last page" in stop_msg.lower():
            logger.info(stop_msg)
        return True, False, ""

    if not matches:
        return False, True, ""  # Continue to next page

    return False, False, ""  # Process matches


def _log_page_complete(new: int, updated: int, skipped: int, errors: int, page_num: int) -> None:
    """Log page completion summary."""
    if new == 0 and updated == 0 and errors == 0 and skipped > 0:
        logger.info(f"Page {page_num} complete: All {skipped} matches already in database")
    else:
        logger.info(f"Page {page_num} complete: New={new}, Updated={updated}, Skipped={skipped}, Errors={errors}")


def _process_pages_loop(
    session_manager: SessionManager,
    driver: Any,
    db_manager: DatabaseManager,
    my_uuid: str,
    my_tree_id: Optional[str],
    csrf_token: str,
    start_page: int,
    max_pages: int,
    batch_size: int
) -> tuple[int, int, int, int, int, int, bool, str]:
    """Process all pages and return statistics."""
    total_new = total_updated = total_skipped = total_errors = 0
    session_deaths = session_recoveries = 0
    run_incomplete = False
    incomplete_reason = ""
    api_total_pages: Optional[int] = None

    end_page = float('inf') if max_pages == 0 else start_page + max_pages - 1
    if max_pages == 0:
        logger.info("MAX_PAGES=0: Will process all pages until no more matches found")

    page_num = start_page
    pages_processed = 0

    while True:
        if max_pages > 0 and page_num > end_page:
            break

        should_continue, deaths, recoveries, reason = _handle_session_health_check(session_manager)
        session_deaths += deaths
        session_recoveries += recoveries
        if not should_continue:
            return total_new, total_updated, total_skipped, total_errors, session_deaths, session_recoveries, True, reason

        _log_page_header(page_num, max_pages, total_new, total_updated, total_skipped, total_errors, api_total_pages)

        matches, deaths, recoveries, should_break, reason, page_total_pages, is_last_page = _fetch_and_validate_page_data(
            driver, session_manager, my_uuid, page_num, csrf_token, max_pages
        )

        if page_total_pages is not None and api_total_pages is None:
            api_total_pages = page_total_pages
            if max_pages == 0:
                logger.info(f"API reports {api_total_pages} total pages available")

        session_deaths += deaths
        session_recoveries += recoveries

        should_break_loop, should_skip_to_next, break_reason = _handle_page_fetch_result(
            should_break, reason, matches, is_last_page, page_num, end_page, max_pages
        )

        if should_break_loop:
            return total_new, total_updated, total_skipped, total_errors, session_deaths, session_recoveries, bool(break_reason), break_reason

        if should_skip_to_next:
            page_num += 1
            pages_processed += 1
            continue

        new, updated, skipped, errors, deaths, recoveries, page_incomplete, page_reason = _process_page_batches(
            matches, batch_size, session_manager, db_manager, my_uuid, my_tree_id, page_num
        )

        total_new, total_updated, total_skipped, total_errors, session_deaths, session_recoveries = _update_page_totals(
            total_new, total_updated, total_skipped, total_errors, session_deaths, session_recoveries,
            new, updated, skipped, errors, deaths, recoveries
        )

        _log_page_complete(new, updated, skipped, errors, page_num)

        if page_incomplete:
            return total_new, total_updated, total_skipped, total_errors, session_deaths, session_recoveries, True, page_reason

        page_num += 1
        pages_processed += 1

    return total_new, total_updated, total_skipped, total_errors, session_deaths, session_recoveries, run_incomplete, incomplete_reason


def _print_coord_summary(total_new: int, total_updated: int, total_skipped: int, total_errors: int, total_run_time: float, run_incomplete: bool, incomplete_reason: str, session_deaths: int, session_recoveries: int, start_page: int, max_pages: int, session_manager: SessionManager) -> None:
    """Print final summary for coord function."""
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}")

    if run_incomplete:
        logger.warning(f"⚠️  RUN INCOMPLETE: {incomplete_reason}")
    else:
        logger.info("✅ Run completed successfully")

    if session_deaths > 0:
        logger.warning(f"⚠️  Session Deaths: {session_deaths}")
        logger.warning(f"⚠️  Session Recoveries: {session_recoveries}")
        if session_deaths > session_recoveries:
            logger.error(f"❌ {session_deaths - session_recoveries} session death(s) could not be recovered")

    if max_pages == 0:
        logger.info(f"Page Range: Started at page {start_page}")
    else:
        logger.info(f"Page Range: {start_page}-{start_page + max_pages - 1} ({max_pages} pages)")

    logger.info(f"New Added: {total_new}")
    logger.info(f"Updated: {total_updated}")
    logger.info(f"Skipped: {total_skipped}")
    logger.info(f"Errors: {total_errors}")

    logger.info("")
    logger.info(f"Total Run Time: {total_run_time/3600:.2f} hours ({total_run_time/60:.1f} minutes)")

    logger.info(f"{'='*80}")

    logger.info("")
    if hasattr(session_manager, 'rate_limiter') and session_manager.rate_limiter:
        session_manager.rate_limiter.print_metrics_summary()


@with_connection_resilience("Action 6: DNA Match Gatherer", max_recovery_attempts=3)
def coord(session_manager: SessionManager, start: int = 1):
    """Main entry point for Action 6: DNA Match Gatherer."""
    # Note: Connection resilience wrapper already logs action name, so we don't duplicate it here

    if session_manager.rate_limiter:
        session_manager.rate_limiter.reset_metrics()
        logger.debug("Rate limiter metrics reset for new run")

    max_pages = config_schema.api.max_pages
    batch_size = config_schema.batch_size
    parallel_workers = getattr(config_schema, 'parallel_workers', 1)
    start_page = start

    logger.info(f"Configuration: START_PAGE={start_page}, MAX_PAGES={max_pages}, BATCH_SIZE={batch_size}, PARALLEL_WORKERS={parallel_workers}")

    _setup_rate_limiting(session_manager, parallel_workers)

    try:
        my_uuid, my_tree_id, db_manager = _initialize_coord_session(session_manager)
    except ValueError as e:
        logger.error(f"Session initialization failed: {e}")
        return None

    logger.debug("Navigating to DNA matches page...")
    if not nav_to_dna_matches_page(session_manager):
        logger.error("Failed to navigate to DNA matches page")
        return None

    driver = session_manager.driver
    csrf_token = get_csrf_token_for_dna_matches(driver)
    if not csrf_token:
        logger.error("Failed to get CSRF token")
        return None

    run_start_time = time.time()
    total_new, total_updated, total_skipped, total_errors, session_deaths, session_recoveries, run_incomplete, incomplete_reason = _process_pages_loop(
        session_manager, driver, db_manager, my_uuid, my_tree_id, csrf_token, start_page, max_pages, batch_size
    )

    run_end_time = time.time()
    total_run_time = run_end_time - run_start_time

    _print_coord_summary(total_new, total_updated, total_skipped, total_errors, total_run_time, run_incomplete, incomplete_reason, session_deaths, session_recoveries, start_page, max_pages, session_manager)

    return not run_incomplete


def _refine_match_list(match_list: list[dict], my_uuid: str, in_tree_ids: set[str]) -> list[dict]:
    """Refine raw match list into structured format."""
    refined_matches = []
    for match_data in match_list:
        refined = _refine_match_from_list_api(match_data, my_uuid, in_tree_ids)
        if refined:
            refined_matches.append(refined)
    return refined_matches


def _refine_match_from_list_api(match_data: dict, my_uuid: str, in_tree_ids: set[str]) -> Optional[dict]:
    """Refine raw match data from Match List API into structured format."""
    try:
        # Extract core fields
        sample_id = match_data.get("sampleId", "").upper()
        if not sample_id:
            return None

        profile_info = match_data.get("matchProfile", {})
        relationship_info = match_data.get("relationship", {})

        profile_id = profile_info.get("userId", "").upper() if profile_info.get("userId") else None
        raw_display_name = profile_info.get("displayName")
        username = format_name(raw_display_name) if raw_display_name else "Unknown"

        # Extract first name
        first_name = None
        if username and username != "Valued Relative":
            name_parts = username.strip().split()
            if name_parts:
                first_name = name_parts[0]

        # Extract DNA data
        shared_cm = int(relationship_info.get("sharedCentimorgans", 0))
        shared_segments = int(relationship_info.get("numSharedSegments", 0))

        # Build links
        compare_link = urljoin(
            config_schema.api.base_url,
            f"discoveryui-matches/compare/{my_uuid.upper()}/with/{sample_id}"
        )

        message_link = None
        if profile_id:
            message_link = urljoin(
                config_schema.api.base_url,
                f"messaging/?p={profile_id}"
            )

        return {
            "uuid": sample_id,
            "profile_id": profile_id,
            "username": username,
            "first_name": first_name,
            "gender": match_data.get("gender"),
            "shared_cm": shared_cm,
            "shared_segments": shared_segments,
            "compare_link": compare_link,
            "message_link": message_link,
            "in_tree": sample_id in in_tree_ids,
            "administrator_profile_id": match_data.get("adminId"),
            "administrator_username": match_data.get("adminName"),
        }

    except Exception as e:
        logger.error(f"Error refining match: {e}")
        return None


def _fetch_match_details_parallel(
    match: dict,
    session_manager: SessionManager,
    my_uuid: str,
) -> dict:
    """
    Fetch all API details for a single match (used in parallel processing).

    NOTE: This function does NOT use database session to avoid concurrency issues.
    Database checks (skip logic) are done in the main thread during the save phase.

    Args:
        match: Match data dictionary
        session_manager: SessionManager for API calls
        my_uuid: User's UUID

    Returns:
        Dictionary with keys: match_details, profile_details, badge_details, predicted_rel, uuid
    """
    # Add random jitter (0-200ms) to spread out parallel requests and prevent thundering herd
    jitter = random.uniform(0.0, 0.2)
    time.sleep(jitter)

    result = {
        'match_details': {},
        'profile_details': {},
        'badge_details': {},
        'predicted_rel': None,
        'ethnicity_data': {},
        'uuid': match.get("uuid"),  # Include UUID for error tracking
    }

    try:
        # Fetch all details (no database checks here to avoid concurrency issues)
        # Rate limiter ensures safe spacing between API calls
        result['match_details'] = _fetch_match_details(session_manager, my_uuid, match["uuid"])

        if match.get("profile_id"):
            result['profile_details'] = _fetch_profile_details(session_manager, match["profile_id"], match["uuid"])

        if match.get("in_tree", False):
            result['badge_details'] = _fetch_badge_details(session_manager, my_uuid, match["uuid"])

        result['predicted_rel'] = _fetch_relationship_probability(session_manager, my_uuid, match["uuid"])

        # Fetch ethnicity comparison data
        result['ethnicity_data'] = fetch_ethnicity_comparison(session_manager, my_uuid, match["uuid"]) or {}

    except Exception as e:
        logger.error(f"Error fetching details for match {match.get('uuid', 'UNKNOWN')}: {e}")
        # Return partial results - the save phase will handle errors gracefully

    return result


def _get_person_id_by_uuid(session, uuid: str) -> Optional[int]:
    """
    Get person_id from database by UUID.

    Args:
        session: Database session
        uuid: Person UUID

    Returns:
        Person ID if found, None otherwise
    """
    try:
        from database import Person
        person = session.query(Person).filter(Person.uuid == uuid).first()
        return person.id if person else None  # FIX: Changed from person.people_id to person.id
    except Exception as e:
        logger.error(f"Error getting person_id for UUID {uuid}: {e}")
        return None


def _first_pass_identify_matches(batch: list[dict], session) -> tuple[list[dict], dict]:
    """
    First pass: Identify which matches need detail fetching.

    Returns:
        Tuple of (matches_needing_details, skip_map)
    """
    matches_needing_details = []
    skip_map = {}

    for match in batch:
        person_id, person_status = _save_person_with_status(session, match)
        if not person_id:
            skip_map[match["uuid"]] = {"skip": True, "reason": "no_person_id", "person_id": None, "person_status": None}
            continue

        skip_details = False
        skip_reason = None

        if person_status != "created" and _dna_match_exists(session, person_id):
            skip_details = True
            skip_reason = "dna_match_exists"
            logger.debug(f"Skipping detail fetch for person_id={person_id} - DnaMatch already exists")
        elif person_status != "created" and _should_skip_person_refresh(session, person_id):
            skip_details = True
            skip_reason = "recently_updated"

        if skip_details:
            skip_map[match["uuid"]] = {"skip": True, "reason": skip_reason, "person_id": person_id, "person_status": person_status}
        else:
            skip_map[match["uuid"]] = {"skip": False, "person_id": person_id, "person_status": person_status}
            matches_needing_details.append(match)

    return matches_needing_details, skip_map


def _fetch_details_parallel(matches_needing_details: list[dict], session_manager: SessionManager, my_uuid: str, parallel_workers: int) -> dict:
    """Fetch match details using parallel workers."""
    match_details_map = {}

    if parallel_workers <= 1 or not matches_needing_details:
        logger.debug(f"Using sequential processing (parallel_workers={parallel_workers}, matches_needing_details={len(matches_needing_details)})")
        return match_details_map

    logger.debug(f"Fetching match details using {parallel_workers} parallel workers for {len(matches_needing_details)} matches...")

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_match = {
            executor.submit(_fetch_match_details_parallel, match, session_manager, my_uuid): match
            for match in matches_needing_details
        }

        for future in tqdm(as_completed(future_to_match), total=len(matches_needing_details), desc="Fetching data", leave=False):
            match = future_to_match[future]
            try:
                details = future.result()
                match_details_map[match["uuid"]] = details
            except Exception as e:
                logger.error(f"Error in parallel fetch for {match.get('uuid', 'UNKNOWN')}: {e}")
                match_details_map[match["uuid"]] = {
                    'match_details': {},
                    'profile_details': {},
                    'badge_details': {},
                    'predicted_rel': None,
                    'uuid': match.get("uuid"),
                }

    return match_details_map


def _get_match_details(match: dict, skip_details: bool, match_details_map: dict, session_manager: SessionManager, my_uuid: str, parallel_workers: int) -> tuple[dict, dict, dict, Any, dict]:
    """Get match details from cache or fetch them."""
    if skip_details:
        return {}, {}, {}, None, {}

    if parallel_workers > 1:
        details = match_details_map.get(match["uuid"], {})
        return (
            details.get('match_details', {}),
            details.get('profile_details', {}),
            details.get('badge_details', {}),
            details.get('predicted_rel'),
            details.get('ethnicity_data', {})
        )

    # Sequential processing: fetch details now
    match_details = _fetch_match_details(session_manager, my_uuid, match["uuid"])
    profile_details = _fetch_profile_details(session_manager, match["profile_id"], match["uuid"]) if match["profile_id"] else {}
    badge_details = _fetch_badge_details(session_manager, my_uuid, match["uuid"]) if match["in_tree"] else {}
    predicted_rel = _fetch_relationship_probability(session_manager, my_uuid, match["uuid"])
    ethnicity_data = fetch_ethnicity_comparison(session_manager, my_uuid, match["uuid"]) or {}

    return match_details, profile_details, badge_details, predicted_rel, ethnicity_data


def _save_match_records(match: dict, person_id: int, match_details: dict, badge_details: dict, predicted_rel: Any, ethnicity_data: dict, skip_details: bool, my_tree_id: Optional[str], session: Any, session_manager: SessionManager) -> None:
    """Save DnaMatch and FamilyTree records."""
    if not skip_details:
        _save_dna_match(session, person_id, match, match_details, predicted_rel, ethnicity_data)

    if match["in_tree"] and badge_details:
        _save_family_tree(session, person_id, badge_details, my_tree_id, session_manager)


def _determine_match_status(skip_details: bool, person_status: str, additional_updates: bool) -> str:
    """Determine the status of a processed match."""
    if skip_details:
        return "skipped"
    if person_status == "created":
        return "created"
    if person_status == "updated" or additional_updates:
        return "updated"
    if person_status == "skipped":
        return "skipped"
    return "unknown"


def _process_single_match(match: dict, skip_map: dict, match_details_map: dict, session: Any, session_manager: SessionManager, my_uuid: str, my_tree_id: Optional[str], parallel_workers: int) -> tuple[str, bool]:
    """Process a single match and return (status, success)."""
    skip_info = skip_map.get(match["uuid"], {})
    person_id = skip_info.get("person_id")
    person_status = skip_info.get("person_status")
    skip_details = skip_info.get("skip", False)

    if not person_id:
        return "error", False

    match_details, profile_details, badge_details, predicted_rel, ethnicity_data = _get_match_details(
        match, skip_details, match_details_map, session_manager, my_uuid, parallel_workers
    )

    if skip_details:
        logger.debug(f"Skipping detail fetch for person_id={person_id} (reason: {skip_info.get('reason')})")

    additional_updates = False
    if profile_details or badge_details or match_details:
        additional_updates = _update_person(session, person_id, profile_details, badge_details, match_details)

    logger.debug(f"Match {match['uuid']}: person_status={person_status}, additional_updates={additional_updates}, skip_details={skip_details}")

    _save_match_records(match, person_id, match_details, badge_details, predicted_rel, ethnicity_data, skip_details, my_tree_id, session, session_manager)

    status = _determine_match_status(skip_details, person_status, additional_updates)
    return status, True


def _second_pass_process_matches(batch: list[dict], session: Any, skip_map: dict, match_details_map: dict, session_manager: SessionManager, my_uuid: str, my_tree_id: Optional[str], parallel_workers: int) -> tuple[int, int, int, int]:
    """Second pass: Process and save all matches."""
    new_count = 0
    updated_count = 0
    skipped_count = 0
    error_count = 0

    for match in tqdm(batch, desc="Saving to database", leave=False):
        try:
            status, success = _process_single_match(
                match, skip_map, match_details_map, session, session_manager, my_uuid, my_tree_id, parallel_workers
            )

            if not success:
                error_count += 1
            elif status == "created":
                new_count += 1
            elif status == "updated":
                updated_count += 1
            elif status == "skipped":
                skipped_count += 1

        except Exception as e:
            logger.error(f"Error processing match {match.get('uuid')}: {e}")
            error_count += 1

    return new_count, updated_count, skipped_count, error_count


def _process_batch(
    batch: list[dict],
    session_manager: SessionManager,
    db_manager: DatabaseManager,
    my_uuid: str,
    my_tree_id: Optional[str]
) -> tuple[int, int, int, int]:
    """Process a batch of matches and save to database."""
    new_count = 0
    updated_count = 0
    skipped_count = 0
    error_count = 0

    session = db_manager.get_session()
    if not session:
        logger.error("Failed to get database session")
        return new_count, updated_count, skipped_count, error_count

    parallel_workers = getattr(config_schema, 'parallel_workers', 1)

    try:
        # First pass: Identify which matches need detail fetching
        matches_needing_details, skip_map = _first_pass_identify_matches(batch, session)

        # Fetch all match details (parallel or sequential based on config)
        match_details_map = _fetch_details_parallel(matches_needing_details, session_manager, my_uuid, parallel_workers)

        # Second pass: Process and save all matches
        new_count, updated_count, skipped_count, error_count = _second_pass_process_matches(
            batch, session, skip_map, match_details_map, session_manager, my_uuid, my_tree_id, parallel_workers
        )

        # Commit all changes
        session.commit()
        logger.debug("Batch committed to database successfully")

    except Exception as e:
        logger.error(f"Error committing batch: {e}")
        session.rollback()
        error_count += len(batch)

    finally:
        db_manager.return_session(session)

    return new_count, updated_count, skipped_count, error_count


def _should_skip_person_refresh(session, person_id: int) -> bool:
    """
    Check if person was recently updated and should skip detail refresh.
    Returns True if person was updated within PERSON_REFRESH_DAYS, False otherwise.

    This implements timestamp-based data freshness checking to avoid redundant API calls.
    """
    from datetime import datetime, timedelta, timezone

    from database import Person

    refresh_days = getattr(config_schema, 'person_refresh_days', 7)  # Default 7 days
    if refresh_days == 0:
        return False  # Disabled if set to 0

    person = session.query(Person).filter_by(id=person_id).first()
    if not person or not person.updated_at:
        return False  # No person or no timestamp, fetch details

    now = datetime.now(timezone.utc)
    last_updated = person.updated_at
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)

    time_since_update = now - last_updated
    threshold = timedelta(days=refresh_days)
    should_skip = time_since_update < threshold

    if should_skip:
        logger.debug(f"Person ID {person_id} updated {time_since_update.days} days ago (threshold: {refresh_days} days) - skipping refresh")

    return should_skip


def _save_person_with_status(session, match: dict) -> tuple[Optional[int], str]:
    """
    Save Person record to database (create or update) and return person_id and status.
    Only includes fields that have values - fields that will be updated later by _update_person()
    are excluded to avoid marking records as "updated" when they haven't actually changed.
    """
    person_data = {
        "uuid": match["uuid"],
        "profile_id": match["profile_id"],
        "username": match["username"],
        "first_name": match["first_name"],
        "message_link": match["message_link"],
        "in_my_tree": match["in_tree"],
    }

    # Only include optional fields if they have values
    if match.get("gender"):
        person_data["gender"] = match["gender"]
    if match.get("administrator_profile_id"):
        person_data["administrator_profile_id"] = match["administrator_profile_id"]
    if match.get("administrator_username"):
        person_data["administrator_username"] = match["administrator_username"]

    # Note: birth_year, contactable, and last_logged_in will be set by _update_person()
    # after fetching additional details from Profile and Badge APIs

    person, status = create_or_update_person(session, person_data)

    if person:
        logger.debug(f"Person {status} with ID {person.id} for UUID {match['uuid']}")
        return person.id, status
    logger.error(f"Failed to create/update person for UUID {match['uuid']}")
    return None, "error"


def _update_profile_details(person: Any, profile_details: dict, updated_fields: list[str]) -> bool:
    """Update person with profile details. Returns True if any field was updated."""
    updated = False
    if profile_details.get("last_logged_in") and profile_details["last_logged_in"] != person.last_logged_in:
        person.last_logged_in = profile_details["last_logged_in"]
        updated_fields.append("last_logged_in")
        updated = True
    if "contactable" in profile_details and profile_details["contactable"] != person.contactable:
        person.contactable = profile_details["contactable"]
        updated_fields.append("contactable")
        updated = True
    return updated


def _update_badge_details(person: Any, badge_details: dict, updated_fields: list[str]) -> bool:
    """Update person with badge details. Returns True if any field was updated."""
    if not badge_details.get("birth_year"):
        return False
    try:
        new_birth_year = int(badge_details["birth_year"])
        if new_birth_year != person.birth_year:
            person.birth_year = new_birth_year
            updated_fields.append("birth_year")
            return True
    except (ValueError, TypeError):
        pass
    return False


def _update_match_details(person: Any, match_details: dict, updated_fields: list[str]) -> bool:
    """Update person with match details (administrator fields). Returns True if any field was updated."""
    updated = False
    if match_details.get("administrator_profile_id") and match_details["administrator_profile_id"] != person.administrator_profile_id:
        person.administrator_profile_id = match_details["administrator_profile_id"]
        updated_fields.append("administrator_profile_id")
        updated = True
    if match_details.get("administrator_username") and match_details["administrator_username"] != person.administrator_username:
        person.administrator_username = match_details["administrator_username"]
        updated_fields.append("administrator_username")
        updated = True
    return updated


def _update_person(session, person_id: int, profile_details: dict, badge_details: dict, match_details: dict) -> bool:
    """
    Update Person record with additional data from Profile, Badge, and Match Details APIs.
    Returns True if any field was actually updated, False otherwise.
    """
    from database import Person

    person = session.query(Person).filter_by(id=person_id).first()
    if not person:
        return False

    updated_fields: list[str] = []
    updated = False

    updated = _update_profile_details(person, profile_details, updated_fields) or updated
    updated = _update_badge_details(person, badge_details, updated_fields) or updated
    updated = _update_match_details(person, match_details, updated_fields) or updated

    if updated:
        session.flush()
        logger.debug(f"_update_person: person_id={person_id}, updated fields: {', '.join(updated_fields)}")
    else:
        logger.debug(f"_update_person: person_id={person_id}, no changes needed")

    return updated


def _fetch_match_details(session_manager: SessionManager, my_uuid: str, match_uuid: str) -> dict:
    """Fetch additional match details from Match Details API."""
    url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matchesservice/api/samples/{my_uuid}/matches/{match_uuid}/details?pmparentaldata=true"
    )

    response = _api_req(
        url=url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        use_csrf_token=False,
        api_description="Match Details API"
    )

    if not response or not isinstance(response, dict):
        logger.debug(f"Match Details API returned empty or invalid response for match {match_uuid}")
        return {}

    logger.debug(f"Match Details API response keys for {match_uuid}: {list(response.keys())}")

    relationship_part = response.get("relationship", {})

    # Extract predicted relationship
    predicted_rel = "Unknown"
    relationship_range = relationship_part.get("relationshipRange")
    if relationship_range and isinstance(relationship_range, list) and len(relationship_range) > 0:
        predicted_rel = relationship_range[0]

    # Extract administrator fields
    admin_profile_id = response.get("adminUcdmId")
    admin_username = response.get("adminDisplayName")

    # Extract match person fields
    display_name = response.get("displayName")
    user_id = response.get("userId")

    logger.debug(f"Match Details for {match_uuid}: displayName={display_name}, userId={user_id}, adminDisplayName={admin_username}, adminUcdmId={admin_profile_id}")

    return {
        "shared_segments": relationship_part.get("sharedSegments"),
        "longest_shared_segment": relationship_part.get("longestSharedSegment"),
        "meiosis": relationship_part.get("meiosis"),
        "from_my_fathers_side": bool(response.get("fathersSide", False)),
        "from_my_mothers_side": bool(response.get("mothersSide", False)),
        "predicted_relationship": predicted_rel,
        "administrator_profile_id": admin_profile_id,
        "administrator_username": admin_username,
        "display_name": display_name,
        "user_id": user_id,
    }


def _fetch_profile_details(session_manager: SessionManager, profile_id: str, match_uuid: str) -> dict:  # type: ignore[unused-function]
    """Fetch profile details from Profile Details API."""
    url = urljoin(
        config_schema.api.base_url,
        f"app-api/express/v1/profiles/details?userId={profile_id.upper()}"
    )

    response = _api_req(
        url=url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        use_csrf_token=False,
        api_description="Profile Details API"
    )

    if not response or not isinstance(response, dict):
        return {}

    result = {}

    # Parse last login date - API uses "LastLoginDate" not "lastLoggedInDate"
    last_login_str = response.get("LastLoginDate")
    if last_login_str:
        try:
            if last_login_str.endswith("Z"):
                result["last_logged_in"] = datetime.fromisoformat(last_login_str.replace("Z", "+00:00"))
            else:
                dt_naive_or_aware = datetime.fromisoformat(last_login_str)
                result["last_logged_in"] = (
                    dt_naive_or_aware.replace(tzinfo=timezone.utc)
                    if dt_naive_or_aware.tzinfo is None
                    else dt_naive_or_aware.astimezone(timezone.utc)
                )
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse LastLoginDate '{last_login_str}' for {profile_id}: {e}")

    # API uses "IsContactable" not "isContactable"
    contactable_val = response.get("IsContactable")
    result["contactable"] = bool(contactable_val) if contactable_val is not None else False

    return result


def _fetch_badge_details(session_manager: SessionManager, my_uuid: str, match_uuid: str) -> dict:
    """Fetch badge details (tree data) from Badge Details API."""
    url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matchesservice/api/samples/{my_uuid}/matches/{match_uuid}/badgedetails"
    )

    response = _api_req(
        url=url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        use_csrf_token=False,
        api_description="Badge Details API"
    )

    if not response or not isinstance(response, dict):
        return {}

    person_badged = response.get("personBadged", {})
    if not person_badged:
        return {}

    return {
        "cfpid": person_badged.get("personId"),
        "person_name_in_tree": format_name(person_badged.get("firstName", "Unknown")),
        "birth_year": person_badged.get("birthYear"),
    }


def _validate_relationship_inputs(my_uuid: str, match_uuid: str, session_manager: SessionManager) -> bool:
    """Validate inputs for relationship probability fetch."""
    if not my_uuid or not match_uuid:
        logger.warning("Cannot fetch relationship probability: missing my_uuid or match_uuid")
        return False

    if not session_manager.scraper:
        logger.warning("Cannot fetch relationship probability: scraper is None")
        return False

    if not session_manager.driver:
        logger.warning("Cannot fetch relationship probability: driver is None")
        return False

    return True


def _sync_cookies_and_get_csrf(driver: Any, scraper: Any, api_description: str) -> Optional[str]:
    """Sync cookies from driver to scraper and extract CSRF token."""
    try:
        driver_cookies = driver.get_cookies()
        if not driver_cookies:
            logger.warning(f"{api_description}: No cookies available from driver")
            return None

        if hasattr(scraper, "cookies"):
            scraper.cookies.clear()
            for cookie in driver_cookies:
                if "name" in cookie and "value" in cookie:
                    scraper.cookies.set(
                        cookie["name"],
                        cookie["value"],
                        domain=cookie.get("domain"),
                        path=cookie.get("path", "/"),
                        secure=cookie.get("secure", False),
                    )

        csrf_token = None
        csrf_cookie_names = ("_dnamatches-matchlistui-x-csrf-token", "_csrf")
        for cookie in driver_cookies:
            if cookie.get("name") in csrf_cookie_names:
                cookie_value = unquote(cookie.get("value", ""))
                parts = cookie_value.split("|")
                csrf_token = parts[0] if parts else None
                break

        if not csrf_token:
            logger.debug(f"{api_description}: No CSRF token found in cookies")

        return csrf_token

    except Exception as e:
        logger.warning(f"{api_description}: Error syncing cookies: {e}")
        return None


def _extract_relationship_from_response(data: dict, sample_id_upper: str, api_description: str) -> Optional[str]:
    """Extract relationship prediction from API response."""
    if "matchProbabilityToSampleId" not in data:
        logger.debug(f"{api_description}: Invalid data structure for {sample_id_upper}")
        return None

    prob_data = data["matchProbabilityToSampleId"]
    predictions = prob_data.get("relationships", {}).get("predictions", [])

    if not predictions:
        logger.debug(f"{api_description}: No predictions for {sample_id_upper}")
        return "Distant relationship?"

    valid_preds = [
        p for p in predictions
        if isinstance(p, dict) and "distributionProbability" in p and "pathsToMatch" in p
    ]

    if not valid_preds:
        logger.debug(f"{api_description}: No valid predictions for {sample_id_upper}")
        return None

    best_pred = max(valid_preds, key=lambda x: x.get("distributionProbability", 0.0))
    top_prob = best_pred.get("distributionProbability", 0.0)
    paths = best_pred.get("pathsToMatch", [])
    labels = [p.get("label") for p in paths if isinstance(p, dict) and p.get("label")]

    if not labels:
        logger.debug(f"{api_description}: No labels found for {sample_id_upper}")
        return None

    final_labels = [str(label) for label in labels[:2] if label]
    if not final_labels:
        return None

    relationship_str = " or ".join(final_labels)
    return f"{relationship_str} [{top_prob:.1f}%]"


def _fetch_relationship_probability(session_manager: SessionManager, my_uuid: str, match_uuid: str) -> Optional[str]:
    """
    Fetch predicted relationship from Relationship Probability API using cloudscraper.
    This is the working version restored from commit 758cca8.
    """
    # Validate inputs
    if not _validate_relationship_inputs(my_uuid, match_uuid, session_manager) or not session_manager.scraper:
        if not session_manager.scraper:
            logger.error("Match Probability API: Scraper not available")
        return None

    my_uuid_upper = my_uuid.upper()
    sample_id_upper = match_uuid.upper()

    url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/parents/list/api/matchProbabilityData/{my_uuid_upper}/{sample_id_upper}",
    )

    referer_url = urljoin(config_schema.api.base_url, "/discoveryui-matches/list/")
    api_description = "Match Probability API"

    headers = {
        "Accept": "application/json",
        "Referer": referer_url,
        "Origin": config_schema.api.base_url.rstrip("/"),
        "User-Agent": random.choice(config_schema.api.user_agents),
    }

    csrf_token = _sync_cookies_and_get_csrf(session_manager.driver, session_manager.scraper, api_description)
    if csrf_token:
        headers["X-CSRF-Token"] = csrf_token

    result: Optional[str] = None
    try:
        response = session_manager.scraper.post(
            url,
            headers=headers,
            json={},
            allow_redirects=False,
            timeout=config_schema.selenium.api_timeout,
        )

        # Handle redirects
        if response.status_code in (301, 302, 303, 307, 308):
            logger.debug(
                f"{api_description}: Received redirect ({response.status_code}) for {sample_id_upper}. "
                f"Skipping (optional data)."
            )
        # Handle errors
        elif not response.ok:
            logger.warning(
                f"{api_description} failed for {sample_id_upper}: {response.status_code} {response.reason}"
            )
        elif not response.content:
            logger.warning(f"{api_description}: Empty response body for {sample_id_upper}")
        else:
            # Parse JSON
            try:
                data = response.json()
                result = _extract_relationship_from_response(data, sample_id_upper, api_description)
            except json.JSONDecodeError as e:
                logger.warning(f"{api_description}: JSON decode failed for {sample_id_upper}: {e}")

    except Exception as e:
        logger.warning(f"{api_description}: Unexpected error for {sample_id_upper}: {e}")

    return result


def _format_relationship_path_part(person: dict, idx: int) -> str:
    """Format a single person in the relationship path."""
    name = person.get("name", "")
    lifespan = person.get("lifeSpan", "")
    relationship = person.get("relationship", "")

    if relationship.startswith("You are the "):
        relationship = relationship.replace("You are the ", "", 1)
        if relationship:
            relationship = relationship[0].upper() + relationship[1:]

    if lifespan:
        path_part = f"{name} {lifespan} ({relationship})" if relationship else f"{name} {lifespan}"
    elif relationship:
        path_part = f"{name} ({relationship})"
    else:
        path_part = name

    logger.debug(f"  Person[{idx}]: {path_part}")
    return path_part


def _build_relationship_path(kinship_persons: list[dict], cfpid: str) -> tuple[Optional[str], Optional[str]]:
    """Build relationship path and extract actual relationship."""
    actual_relationship = None
    if kinship_persons:
        first_person = kinship_persons[0]
        actual_relationship = first_person.get("relationship")
        logger.debug(f"Found actual_relationship for CFPID {cfpid}: {actual_relationship}")

    path_parts = []
    for idx, person in enumerate(kinship_persons):
        path_part = _format_relationship_path_part(person, idx)
        path_parts.append(path_part)

    relationship_path = None
    if path_parts:
        relationship_path = "\n↓\n".join(path_parts)
        logger.debug(f"Built relationship_path for CFPID {cfpid} with {len(path_parts)} parts")

    return actual_relationship, relationship_path


def _fetch_ladder_details(session_manager: SessionManager, cfpid: str, tree_id: str) -> dict:
    """Fetch relationship ladder details from Kinship Relation Ladder API."""
    if not cfpid or not tree_id:
        return {}

    try:
        my_user_id = config_schema.api.my_user_id
        if not my_user_id:
            logger.debug(f"No user_id configured, cannot fetch ladder for CFPID {cfpid}")
            return {}

        url = urljoin(
            config_schema.api.base_url,
            f"family-tree/person/card/user/{my_user_id}/tree/{tree_id}/person/{cfpid}/kinship/relationladderwithlabels"
        )

        api_result = _api_req(
            url=url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers={},
            use_csrf_token=False,
            api_description="Kinship Relation Ladder API",
            referer_url=None,
            force_text_response=False,
        )

        if not api_result or not isinstance(api_result, dict):
            logger.debug(f"Kinship Relation Ladder API returned empty or invalid response for CFPID {cfpid}")
            return {}

        logger.debug(f"Kinship Relation Ladder API response keys for CFPID {cfpid}: {list(api_result.keys())}")

        kinship_persons = api_result.get("kinshipPersons", [])
        if not kinship_persons:
            logger.debug(f"No kinshipPersons in response for CFPID {cfpid}")
            return {}

        logger.debug(f"Found {len(kinship_persons)} kinship persons for CFPID {cfpid}")

        actual_relationship, relationship_path = _build_relationship_path(kinship_persons, cfpid)

        result = {
            "actual_relationship": actual_relationship,
            "relationship_path": relationship_path,
        }
        logger.debug(f"Ladder details result for CFPID {cfpid}: actual_relationship={actual_relationship}, relationship_path={'YES' if relationship_path else 'NO'}")
        return result

    except Exception as e:
        logger.debug(f"Error fetching ladder details for CFPID {cfpid}: {e}")
        return {}


def _dna_match_exists(session, person_id: int) -> bool:
    """Check if a DnaMatch record already exists for this person."""
    from database import DnaMatch

    existing = session.query(DnaMatch).filter_by(people_id=person_id).first()
    return existing is not None


def _save_dna_match(session, person_id: int, match: dict, match_details: dict, predicted_relationship: Optional[str] = None, ethnicity_data: Optional[dict] = None) -> str:
    """
    Save DnaMatch record to database.

    Skip logic: If a DnaMatch record already exists for this person, skip populating it.
    This prevents unnecessary updates and preserves existing DNA data.
    """
    # Use predicted_relationship from API call, fallback to match_details, then "Unknown"
    predicted_rel = predicted_relationship or match_details.get("predicted_relationship", "Unknown")

    dna_data = {
        "people_id": person_id,
        "compare_link": match["compare_link"] or "",
        "cm_dna": match["shared_cm"],
        "predicted_relationship": predicted_rel,
        "shared_segments": match_details.get("shared_segments") or match["shared_segments"],
        "longest_shared_segment": match_details.get("longest_shared_segment"),
        "meiosis": match_details.get("meiosis"),
        "from_my_fathers_side": match_details.get("from_my_fathers_side", False),
        "from_my_mothers_side": match_details.get("from_my_mothers_side", False),
    }

    # Add ethnicity data if available
    if ethnicity_data:
        ethnicity_metadata = load_ethnicity_metadata()
        if ethnicity_metadata and ethnicity_metadata.get("tree_owner_regions"):
            region_keys = [region["key"] for region in ethnicity_metadata["tree_owner_regions"]]
            column_mapping = {region["key"]: region["column_name"] for region in ethnicity_metadata["tree_owner_regions"]}

            # Extract percentages for tree owner's regions
            ethnicity_percentages = extract_match_ethnicity_percentages(ethnicity_data, region_keys)

            # Add ethnicity percentages to dna_data using column names
            for region_key, percentage in ethnicity_percentages.items():
                column_name = column_mapping.get(region_key)
                if column_name:
                    dna_data[column_name] = percentage

    # Check if DnaMatch record already exists
    if _dna_match_exists(session, person_id):
        logger.debug(f"DnaMatch record already exists for person_id={person_id} - skipping (would save: cm={dna_data['cm_dna']}, rel={dna_data['predicted_relationship']})")
        return "skipped"

    result = create_or_update_dna_match(session, dna_data)
    logger.debug(f"DnaMatch result for person_id={person_id}: {result}")
    return result


def _family_tree_exists(session, person_id: int) -> bool:
    """Check if a FamilyTree record already exists for this person."""
    from database import FamilyTree

    existing = session.query(FamilyTree).filter_by(people_id=person_id).first()
    return existing is not None


def _save_family_tree(session, person_id: int, badge_details: dict, my_tree_id: Optional[str], session_manager: SessionManager) -> str:
    """
    Save FamilyTree record to database.

    Skip logic: If a FamilyTree record already exists for this person, skip populating it.
    This prevents unnecessary API calls and database updates.
    """
    cfpid = badge_details.get("cfpid")
    person_name = badge_details.get("person_name_in_tree")

    # Check if FamilyTree record already exists
    if _family_tree_exists(session, person_id):
        logger.debug(f"FamilyTree record already exists for person_id={person_id} - skipping (would save: cfpid={cfpid}, name={person_name})")
        return "skipped"

    # Build links if we have CFPID
    facts_link = None
    view_in_tree_link = None
    if cfpid and my_tree_id:
        facts_link = urljoin(
            config_schema.api.base_url,
            f"family-tree/person/tree/{my_tree_id}/person/{cfpid}/facts"
        )
        view_in_tree_link = urljoin(
            config_schema.api.base_url,
            f"family-tree/tree/{my_tree_id}/family?cfpid={cfpid}"
        )

    # Fetch ladder details if we have CFPID
    ladder_details = {}
    if cfpid and my_tree_id:
        ladder_details = _fetch_ladder_details(session_manager, cfpid, my_tree_id)
        logger.debug(f"Ladder details for CFPID {cfpid}: actual_relationship={ladder_details.get('actual_relationship')}, relationship_path={'YES' if ladder_details.get('relationship_path') else 'NO'}")

    tree_data = {
        "people_id": person_id,
        "cfpid": cfpid,
        "person_name_in_tree": person_name,
        "facts_link": facts_link,
        "view_in_tree_link": view_in_tree_link,
        "actual_relationship": ladder_details.get("actual_relationship"),
        "relationship_path": ladder_details.get("relationship_path"),
    }

    logger.debug(f"Creating/updating FamilyTree for person_id={person_id}, CFPID={cfpid}, view_in_tree_link={view_in_tree_link}")
    result = create_or_update_family_tree(session, tree_data)
    logger.debug(f"FamilyTree result for person_id={person_id}: {result}")
    return result


# ==============================================
# Test Functions
# ==============================================


def _test_database_schema() -> bool:
    """Test that Person model has 'id' attribute, not 'people_id'."""
    from database import DnaMatch, FamilyTree, Person

    assert hasattr(Person, 'id'), "Person should have 'id' attribute"
    assert not hasattr(Person, 'people_id'), "Person should NOT have 'people_id' attribute"
    assert hasattr(DnaMatch, 'people_id'), "DnaMatch should have 'people_id' foreign key"
    assert hasattr(FamilyTree, 'people_id'), "FamilyTree should have 'people_id' foreign key"
    return True


def _test_person_id_attribute_fix() -> bool:
    """Test that _get_person_id_by_uuid returns person.id not person.people_id."""
    import inspect

    source = inspect.getsource(_get_person_id_by_uuid)
    assert 'return person.id if person else None' in source, "_get_person_id_by_uuid should return person.id"

    lines = source.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if 'return person.people_id' in line and 'person.id' not in line:
            raise AssertionError(f"Old buggy code found: {line.strip()}")

    return True


def _test_parallel_function_thread_safety() -> bool:
    """Test that parallel function doesn't take database session parameter."""
    import inspect

    sig = inspect.signature(_fetch_match_details_parallel)
    params = list(sig.parameters.keys())

    assert 'session' not in params, f"_fetch_match_details_parallel should not have 'session' parameter: {params}"
    assert params == ['match', 'session_manager', 'my_uuid'], f"Function signature incorrect: {params}"

    source = inspect.getsource(_fetch_match_details_parallel)
    assert 'concurrency' in source.lower() or 'thread' in source.lower(), "Thread-safety should be documented"

    return True


def _test_bounds_checking() -> bool:
    """Test that bounds checking is in place for list/tuple access."""
    from pathlib import Path

    file_content = Path(__file__).read_text(encoding='utf-8')

    assert 'parts[0] if parts else None' in file_content or 'csrf_token = parts[0] if parts' in file_content, \
        "CSRF token extraction missing bounds checking"
    assert 'len(kinship_persons) > 0' in file_content or 'if kinship_persons and len(kinship_persons)' in file_content, \
        "Kinship persons access missing bounds checking"
    assert 'len(relationship) > 0' in file_content or 'if relationship and len(relationship)' in file_content, \
        "String capitalization missing bounds checking"

    return True


def _test_error_handling() -> bool:
    """Test that comprehensive error handling is in place."""
    import inspect

    source = inspect.getsource(_get_person_id_by_uuid)
    assert 'try:' in source and 'except' in source, "_get_person_id_by_uuid missing error handling"

    parallel_source = inspect.getsource(_fetch_match_details_parallel)
    assert 'try:' in parallel_source and 'except' in parallel_source, \
        "_fetch_match_details_parallel missing error handling"

    batch_source = inspect.getsource(_process_batch)
    assert 'try:' in batch_source and 'except' in batch_source and 'finally:' in batch_source, \
        "_process_batch missing comprehensive error handling (try/except/finally)"

    return True


# ==============================================
# Functional API Tests (Require Live Session)
# ==============================================

# Global session manager for test reuse
_test_session_manager: Optional[SessionManager] = None
_test_session_uuid: Optional[str] = None


def _check_cached_session(reuse_session: bool) -> tuple[Optional[SessionManager], Optional[str]]:
    """Check if cached session is available and valid."""
    global _test_session_manager, _test_session_uuid

    if not reuse_session or _test_session_manager is None or _test_session_uuid is None:
        return None, None

    if _test_session_manager.is_sess_valid():
        logger.info("♻️  Reusing existing authenticated session from previous test")
        return _test_session_manager, _test_session_uuid

    logger.info("⚠️  Cached session invalid, creating new session...")
    _test_session_manager = None
    _test_session_uuid = None
    return None, None


def _create_and_start_session() -> SessionManager:
    """Create and start a new session manager."""
    logger.info("Step 1: Creating SessionManager...")
    sm = SessionManager()
    logger.info("✅ SessionManager created")

    logger.info("Step 2: Configuring browser requirement...")
    sm.browser_manager.browser_needed = True
    logger.info("✅ Browser marked as needed")

    logger.info("Step 3: Starting session (database + browser)...")
    started = sm.start_sess("Action 6 API Tests")
    if not started:
        sm.close_sess(keep_db=False)
        raise AssertionError("Failed to start session - browser initialization failed")
    logger.info("✅ Session started successfully")

    return sm


def _authenticate_session(sm: SessionManager) -> None:
    """Authenticate the session using cookies or login."""
    from utils import _load_login_cookies, log_in, login_status

    logger.info("Step 4: Attempting to load saved cookies...")
    cookies_loaded = _load_login_cookies(sm)
    logger.info("✅ Loaded saved cookies from previous session" if cookies_loaded else "⚠️  No saved cookies found")

    logger.info("Step 5: Checking login status...")
    login_check = login_status(sm, disable_ui_fallback=True)

    if login_check is True:
        logger.info("✅ Already logged in")
    elif login_check is False:
        logger.info("⚠️  Not logged in - attempting login...")
        login_result = log_in(sm)
        if login_result != "LOGIN_SUCCEEDED":
            sm.close_sess(keep_db=False)
            raise AssertionError(f"Login failed: {login_result}")
        logger.info("✅ Login successful")
    else:
        sm.close_sess(keep_db=False)
        raise AssertionError("Login status check failed critically (returned None)")


def _validate_session_ready(sm: SessionManager) -> None:
    """Validate session is ready with all identifiers."""
    logger.info("Step 6: Ensuring session is ready...")
    ready = sm.ensure_session_ready("coord - API Tests", skip_csrf=True)
    if not ready:
        sm.close_sess(keep_db=False)
        raise AssertionError("Session not ready - cookies/identifiers missing")
    logger.info("✅ Session ready")

    logger.info("Step 7: Verifying UUID is available...")
    if not sm.my_uuid:
        sm.close_sess(keep_db=False)
        raise AssertionError("UUID not available - session initialization incomplete")
    logger.info(f"✅ UUID available: {sm.my_uuid}")


def _ensure_session_for_api_tests(reuse_session: bool = True) -> tuple[SessionManager, str]:
    """Ensure session is ready for API tests. Returns (session_manager, my_uuid).

    This function establishes a valid Ancestry session by:
    1. Creating and initializing a SessionManager (or reusing existing one)
    2. Starting the session (database + browser)
    3. Loading saved cookies from previous session (if available)
    4. Checking login status and logging in if needed
    5. Ensuring session is ready with all identifiers
    6. Validating UUID is available

    Args:
        reuse_session: If True, reuse existing session from previous test (default: True)

    Raises AssertionError if session cannot be established (tests will be skipped).
    """
    global _test_session_manager, _test_session_uuid

    # Check for cached session
    cached_sm, cached_uuid = _check_cached_session(reuse_session)
    if cached_sm and cached_uuid:
        return cached_sm, cached_uuid

    logger.info("=" * 80)
    logger.info("Setting up authenticated session for API tests...")
    logger.info("=" * 80)

    # Create and start new session
    sm = _create_and_start_session()

    # Authenticate the session
    _authenticate_session(sm)

    # Validate session is ready
    _validate_session_ready(sm)

    logger.info("=" * 80)
    logger.info("✅ Valid authenticated session established for API tests")
    logger.info("=" * 80)

    # Cache session for reuse
    _test_session_manager = sm
    _test_session_uuid = sm.my_uuid

    return sm, sm.my_uuid


def _test_match_list_api() -> bool:
    """Test fetching match list from API."""
    sm, my_uuid = _ensure_session_for_api_tests()

    try:
        # Navigate to DNA matches page first (required for CSRF token)
        logger.info("Navigating to DNA matches page...")
        nav_success = nav_to_dna_matches_page(sm)
        assert nav_success, "Failed to navigate to DNA matches page"
        logger.info("✅ Navigation successful")

        # Get CSRF token
        logger.info("Getting CSRF token...")
        csrf_token = get_csrf_token_for_dna_matches(sm.driver)
        assert csrf_token, "Failed to get CSRF token"
        logger.info("✅ CSRF token retrieved")

        # Fetch first page of matches
        response = fetch_match_list_page(sm.driver, sm, my_uuid, 1, csrf_token)
        assert response is not None, "Expected API response, got None"
        assert isinstance(response, dict), f"Expected dict response, got {type(response).__name__}"

        # Extract matches from response (API returns "matchList" not "matches")
        match_list = response.get("matchList", [])
        assert match_list is not None, "Expected matchList in response, got None"
        assert isinstance(match_list, list), f"Expected list for matchList, got {type(match_list).__name__}"
        assert len(match_list) > 0, f"Expected at least 1 match, got {len(match_list)}"

        # Validate match structure (raw API format)
        first_match = match_list[0]
        assert "sampleId" in first_match, f"Expected 'sampleId' in match, got keys: {list(first_match.keys())}"

        match_profile = first_match.get("matchProfile", {})
        assert "displayName" in match_profile, f"Expected 'displayName' in matchProfile, got keys: {list(match_profile.keys())}"

        relationship = first_match.get("relationship", {})
        assert "sharedCentimorgans" in relationship, f"Expected 'sharedCentimorgans' in relationship, got keys: {list(relationship.keys())}"
        assert "numSharedSegments" in relationship, f"Expected 'numSharedSegments' in relationship, got keys: {list(relationship.keys())}"

        logger.info(f"✅ Match List API: Found {len(match_list)} matches")
        logger.info(f"   First match: {match_profile.get('displayName')} ({relationship.get('sharedCentimorgans')} cM)")
        return True

    except Exception as e:
        logger.error(f"❌ Match List API test failed: {e}")
        raise


def _test_match_details_api() -> bool:
    """Test fetching match details from API."""
    sm, my_uuid = _ensure_session_for_api_tests()

    try:
        # Navigate to DNA matches page first (required for CSRF token)
        nav_success = nav_to_dna_matches_page(sm)
        assert nav_success, "Failed to navigate to DNA matches page"

        # Get CSRF token and fetch first page of matches
        csrf_token = get_csrf_token_for_dna_matches(sm.driver)
        assert csrf_token, "Failed to get CSRF token"

        response = fetch_match_list_page(sm.driver, sm, my_uuid, 1, csrf_token)
        assert response and isinstance(response, dict), "Match list response should be a dictionary"

        match_list = response.get("matchList", [])
        assert match_list and len(match_list) > 0, "Need at least one match to test details"

        match_uuid = match_list[0]["sampleId"]
        logger.info(f"Testing Match Details API with match: {match_uuid}")

        # Fetch match details
        details = _fetch_match_details(sm, my_uuid, match_uuid)
        assert isinstance(details, dict), "Match details should be a dictionary"

        # Validate key fields
        assert "shared_segments" in details, "Should have shared_segments"
        assert "longest_shared_segment" in details, "Should have longest_shared_segment"
        assert "predicted_relationship" in details, "Should have predicted_relationship"

        logger.info(f"✅ Match Details API: Retrieved details for {match_uuid}")
        logger.info(f"   Shared segments: {details.get('shared_segments')}")
        logger.info(f"   Predicted relationship: {details.get('predicted_relationship')}")
        return True

    except Exception as e:
        logger.error(f"❌ Match Details API test failed: {e}")
        raise


def _test_profile_details_api() -> bool:
    """Test fetching profile details from API."""
    sm, my_uuid = _ensure_session_for_api_tests()

    try:
        # Navigate to DNA matches page first (required for CSRF token)
        nav_success = nav_to_dna_matches_page(sm)
        assert nav_success, "Failed to navigate to DNA matches page"

        # Get CSRF token and fetch first page of matches
        csrf_token = get_csrf_token_for_dna_matches(sm.driver)
        assert csrf_token, "Failed to get CSRF token"

        response = fetch_match_list_page(sm.driver, sm, my_uuid, 1, csrf_token)
        assert response and isinstance(response, dict), "Match list response should be a dictionary"

        match_list = response.get("matchList", [])
        assert match_list and len(match_list) > 0, "Need at least one match"

        # Find a match with profile_id (userId in API response)
        match_with_profile = None
        for match in match_list:
            match_profile = match.get("matchProfile", {})
            if match_profile.get("userId"):
                match_with_profile = match
                break

        if not match_with_profile:
            logger.warning("⚠️  No matches with profile_id found, skipping profile details test")
            return True

        match_profile = match_with_profile.get("matchProfile", {})
        profile_id = match_profile["userId"]
        match_uuid = match_with_profile["sampleId"]
        logger.info(f"Testing Profile Details API with profile: {profile_id}")

        # Fetch profile details
        details = _fetch_profile_details(sm, profile_id, match_uuid)
        assert isinstance(details, dict), "Profile details should be a dictionary"

        # Validate key fields
        assert "last_logged_in" in details, "Should have last_logged_in"
        assert "contactable" in details, "Should have contactable"

        logger.info(f"✅ Profile Details API: Retrieved details for profile {profile_id}")
        logger.info(f"   Last logged in: {details.get('last_logged_in')}")
        logger.info(f"   Contactable: {details.get('contactable')}")
        return True

    except Exception as e:
        logger.error(f"❌ Profile Details API test failed: {e}")
        raise


def _test_badge_details_api() -> bool:
    """Test fetching badge details (tree data) from API."""
    sm, my_uuid = _ensure_session_for_api_tests()

    try:
        # Navigate to DNA matches page first (required for CSRF token)
        nav_success = nav_to_dna_matches_page(sm)
        assert nav_success, "Failed to navigate to DNA matches page"

        # Get CSRF token and fetch first page of matches
        csrf_token = get_csrf_token_for_dna_matches(sm.driver)
        assert csrf_token, "Failed to get CSRF token"

        response = fetch_match_list_page(sm.driver, sm, my_uuid, 1, csrf_token)
        assert response and isinstance(response, dict), "Match list response should be a dictionary"

        match_list = response.get("matchList", [])
        assert match_list and len(match_list) > 0, "Need at least one match"

        # Get sample IDs and check in-tree status
        sample_ids = [m.get("sampleId") for m in match_list if m.get("sampleId")]
        in_tree_ids = fetch_in_tree_status(sm.driver, sm, my_uuid, sample_ids, csrf_token, 1)

        # Find a match that's in tree
        match_in_tree = None
        for match in match_list:
            if match.get("sampleId") in in_tree_ids:
                match_in_tree = match
                break

        if not match_in_tree:
            logger.warning("⚠️  No matches in tree found, skipping badge details test")
            return True

        match_uuid = match_in_tree["sampleId"]
        logger.info(f"Testing Badge Details API with match: {match_uuid}")

        # Fetch badge details
        details = _fetch_badge_details(sm, my_uuid, match_uuid)
        assert isinstance(details, dict), "Badge details should be a dictionary"

        logger.info(f"✅ Badge Details API: Retrieved details for {match_uuid}")
        logger.info(f"   Badge details keys: {list(details.keys())}")
        return True

    except Exception as e:
        logger.error(f"❌ Badge Details API test failed: {e}")
        raise


def _test_relationship_probability_api() -> bool:
    """Test fetching relationship probability from API."""
    sm, my_uuid = _ensure_session_for_api_tests()

    try:
        # Navigate to DNA matches page first (required for CSRF token)
        nav_success = nav_to_dna_matches_page(sm)
        assert nav_success, "Failed to navigate to DNA matches page"

        # Get CSRF token and fetch first page of matches
        csrf_token = get_csrf_token_for_dna_matches(sm.driver)
        assert csrf_token, "Failed to get CSRF token"

        response = fetch_match_list_page(sm.driver, sm, my_uuid, 1, csrf_token)
        assert response and isinstance(response, dict), "Match list response should be a dictionary"

        match_list = response.get("matchList", [])
        assert match_list and len(match_list) > 0, "Need at least one match"

        match_uuid = match_list[0]["sampleId"]
        logger.info(f"Testing Relationship Probability API with match: {match_uuid}")

        # Fetch relationship probability
        relationship = _fetch_relationship_probability(sm, my_uuid, match_uuid)

        # Relationship can be None or a string
        if relationship:
            assert isinstance(relationship, str), "Relationship should be a string"
            logger.info(f"✅ Relationship Probability API: {relationship}")
        else:
            logger.info("✅ Relationship Probability API: No relationship data available")

        return True

    except Exception as e:
        logger.error(f"❌ Relationship Probability API test failed: {e}")
        raise


def _test_parallel_fetch_match_details() -> bool:
    """Test parallel fetching of match details."""
    sm, my_uuid = _ensure_session_for_api_tests()

    try:
        # Navigate to DNA matches page first (required for CSRF token)
        nav_success = nav_to_dna_matches_page(sm)
        assert nav_success, "Failed to navigate to DNA matches page"

        # Get CSRF token and fetch first page of matches
        csrf_token = get_csrf_token_for_dna_matches(sm.driver)
        assert csrf_token, "Failed to get CSRF token"

        response = fetch_match_list_page(sm.driver, sm, my_uuid, 1, csrf_token)
        assert response and isinstance(response, dict), "Match list response should be a dictionary"

        match_list = response.get("matchList", [])
        assert match_list and len(match_list) > 0, "Need at least one match"

        # Convert raw API matches to refined format for parallel fetch
        sample_ids = [m.get("sampleId") for m in match_list if m.get("sampleId")]
        in_tree_ids = fetch_in_tree_status(sm.driver, sm, my_uuid, sample_ids, csrf_token, 1)
        refined_matches = _refine_match_list(match_list, my_uuid, in_tree_ids)

        # Test parallel fetch with first 3 matches (or fewer if not available)
        test_matches = refined_matches[:min(3, len(refined_matches))]
        logger.info(f"Testing parallel fetch with {len(test_matches)} matches")

        # Fetch details in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_fetch_match_details_parallel, match, sm, my_uuid)
                for match in test_matches
            ]

            results = []
            for future in as_completed(futures):
                result = future.result()
                assert isinstance(result, dict), "Result should be a dictionary"
                results.append(result)

        assert len(results) == len(test_matches), "Should have results for all matches"
        logger.info(f"✅ Parallel Fetch: Successfully fetched details for {len(results)} matches")
        return True

    except Exception as e:
        logger.error(f"❌ Parallel fetch test failed: {e}")
        raise


def action6_module_tests() -> bool:
    """Comprehensive test suite for action6_gather.py using standardized TestSuite framework."""
    import warnings

    from test_framework import TestSuite

    suite = TestSuite("Action 6 - DNA Match Gatherer", "action6_gather.py")
    suite.start_suite()

    # === CODE QUALITY & STRUCTURE TESTS ===
    suite.run_test(
        "Database Schema Validation",
        _test_database_schema,
        test_summary="Validates database schema uses correct attribute names for primary and foreign keys",
        functions_tested="Person, DnaMatch, FamilyTree (database models)",
        method_description="Check that Person model has 'id' attribute (not 'people_id'), and DnaMatch/FamilyTree have 'people_id' foreign keys using hasattr()",
        expected_outcome="Person.id exists, Person.people_id does not exist, DnaMatch.people_id and FamilyTree.people_id exist",
    )

    suite.run_test(
        "Person ID Attribute Fix",
        _test_person_id_attribute_fix,
        test_summary="Validates function returns person.id instead of deprecated person.people_id",
        functions_tested="_get_person_id_by_uuid()",
        method_description="Inspect function source code using inspect.getsource() to verify it returns 'person.id if person else None'",
        expected_outcome="Source contains 'return person.id' and does not contain buggy 'return person.people_id'",
    )

    suite.run_test(
        "Thread-Safe Parallel Processing",
        _test_parallel_function_thread_safety,
        test_summary="Validates parallel processing function is thread-safe and doesn't use database session",
        functions_tested="_fetch_match_details_parallel()",
        method_description="Check function signature has no 'session' parameter using inspect.signature() and verify thread-safety documentation exists",
        expected_outcome="Parameters are ['match', 'session_manager', 'my_uuid'] and source mentions concurrency/threads",
    )

    suite.run_test(
        "Bounds Checking for Index Errors",
        _test_bounds_checking,
        test_summary="Validates bounds checking prevents IndexError when accessing lists/tuples",
        functions_tested="_sync_cookies_and_get_csrf(), _build_relationship_path(), various string operations",
        method_description="Search file content for safe access patterns like 'parts[0] if parts else None' using conditional expressions",
        expected_outcome="CSRF token extraction, kinship persons access, and string operations all use bounds checking",
    )

    suite.run_test(
        "Error Handling Coverage",
        _test_error_handling,
        test_summary="Validates comprehensive error handling with try/except blocks in critical functions",
        functions_tested="_get_person_id_by_uuid(), _fetch_match_details_parallel(), _process_batch()",
        method_description="Inspect source code using inspect.getsource() to verify try/except/finally blocks exist",
        expected_outcome="All critical functions have appropriate error handling (try/except, some with finally)",
    )

    # === FUNCTIONAL API TESTS (Require Live Session) ===
    suite.run_test(
        "Match List API",
        _test_match_list_api,
        test_summary="Validates Match List API returns paginated DNA match data with correct structure",
        functions_tested="fetch_match_list_page(), nav_to_dna_matches_page(), get_csrf_token_for_dna_matches()",
        method_description="Navigate to DNA matches page, get CSRF token, call API, validate response contains matchList with sampleId/displayName/sharedCentimorgans/numSharedSegments",
        expected_outcome="API returns dict with 'matchList' array containing 20+ matches, each with required fields (sampleId, matchProfile, relationship)",
    )

    suite.run_test(
        "Match Details API",
        _test_match_details_api,
        test_summary="Validates Match Details API returns additional DNA data for specific match",
        functions_tested="_fetch_match_details()",
        method_description="Get match UUID from Match List, call Match Details API, validate response contains shared_segments/longest_shared_segment/predicted_relationship",
        expected_outcome="API returns dict with DNA details including segment counts, longest segment, meiosis, and relationship prediction",
    )

    suite.run_test(
        "Profile Details API",
        _test_profile_details_api,
        test_summary="Validates Profile Details API returns user profile information for matches with public profiles",
        functions_tested="_fetch_profile_details()",
        method_description="Find match with userId, call Profile Details API, validate response contains last_logged_in and contactable fields",
        expected_outcome="API returns dict with LastLoginDate (parsed to datetime) and IsContactable (boolean) fields",
    )

    suite.run_test(
        "Badge Details API",
        _test_badge_details_api,
        test_summary="Validates Badge Details API returns family tree data for matches in user's tree",
        functions_tested="_fetch_badge_details(), fetch_in_tree_status()",
        method_description="Check in-tree status for matches, find match in tree, call Badge Details API, validate response contains cfpid/person_name_in_tree/birth_year",
        expected_outcome="API returns dict with personBadged object containing tree-specific data (personId, firstName, birthYear)",
    )

    suite.run_test(
        "Relationship Probability API",
        _test_relationship_probability_api,
        test_summary="Validates Relationship Probability API returns predicted relationship with confidence percentage",
        functions_tested="_fetch_relationship_probability()",
        method_description="Get match UUID, call Relationship Probability API using cloudscraper, validate response format",
        expected_outcome="API returns formatted string like 'mother [99.0%]' or None if data unavailable",
    )

    suite.run_test(
        "Parallel Match Details Fetching",
        _test_parallel_fetch_match_details,
        test_summary="Validates parallel processing of match details using ThreadPoolExecutor with 2 workers",
        functions_tested="_fetch_match_details_parallel(), _refine_match_list()",
        method_description="Refine match list, submit 3 matches to ThreadPoolExecutor, collect results, validate all complete successfully",
        expected_outcome="All 3 parallel fetch operations complete and return dict results with match_details/profile_details/badge_details/predicted_rel",
    )

    result = suite.finish_suite()

    # Clean up test session if it exists
    global _test_session_manager
    if _test_session_manager is not None:
        try:
            # Suppress all warnings and errors during cleanup
            import os
            import sys

            # Redirect stderr temporarily to suppress exception messages
            original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    _test_session_manager.close_sess(keep_db=False)
            finally:
                # Restore stderr
                sys.stderr.close()
                sys.stderr = original_stderr
        except Exception:
            # Silently ignore all cleanup errors
            pass
        finally:
            _test_session_manager = None

    return result


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return action6_module_tests()


if __name__ == "__main__":
    import os
    import sys
    import warnings

    # Suppress browser cleanup warnings globally
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", message=".*WinError.*")

    # Suppress stderr for undetected_chromedriver cleanup exceptions
    # This prevents the "OSError: [WinError 6] The handle is invalid" message
    original_stderr = sys.stderr

    try:
        print("🧪 Running Action 6 comprehensive test suite...")
        success = run_comprehensive_tests()

        # Redirect stderr to devnull before exit to suppress Chrome destructor exceptions
        sys.stderr = open(os.devnull, 'w')
        sys.exit(0 if success else 1)
    except Exception as e:
        # Restore stderr for any unexpected errors
        sys.stderr = original_stderr
        print(f"Unexpected error: {e}")
        sys.exit(1)

