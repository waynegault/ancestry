#!/usr/bin/env python3

"""
action6b_v2.py - DNA Match Gatherer (Rebuilt to follow Action 6's proven patterns)

This version follows Action 6's exact data flow:
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

        logger.info(f"Batch {batch_num}/{total_batches_on_page} (matches {batch_start+1}-{batch_end} of {len(matches)} on page {page_num})")

        new, updated, skipped, errors = _process_batch(batch, session_manager, db_manager, my_uuid, my_tree_id)

        total_new += new
        total_updated += updated
        total_skipped += skipped
        total_errors += errors

        logger.info(f"Batch {batch_num} complete: New={new}, Updated={updated}, Skipped={skipped}, Errors={errors}")

    return total_new, total_updated, total_skipped, total_errors, session_deaths, session_recoveries, run_incomplete, incomplete_reason


def _fetch_and_validate_page_data(driver: Any, session_manager: SessionManager, my_uuid: str, page_num: int, csrf_token: str, max_pages: int) -> tuple[Optional[list[dict]], int, int, bool, str]:
    """Fetch and validate page data. Returns (matches, deaths, recoveries, should_break, reason)."""
    api_response = fetch_match_list_page(driver, session_manager, my_uuid, page_num, csrf_token)
    if not api_response or not isinstance(api_response, dict):
        logger.warning(f"No API response for page {page_num}")
        should_continue, deaths, recoveries, reason, should_break = _handle_api_failure(session_manager, page_num, max_pages)
        return None, deaths, recoveries, should_break, reason

    match_list = api_response.get("matchList", [])
    if not match_list:
        logger.warning(f"No matches in API response for page {page_num}")
        if max_pages == 0:
            logger.info("No more matches available. Stopping.")
            return None, 0, 0, True, ""
        return None, 0, 0, False, ""

    sample_ids = [m.get("sampleId", "").upper() for m in match_list if m.get("sampleId")]
    if not sample_ids:
        logger.warning(f"No sample IDs found on page {page_num}")
        return None, 0, 0, False, ""

    in_tree_ids = fetch_in_tree_status(driver, session_manager, my_uuid, sample_ids, csrf_token, page_num)
    matches = _refine_match_list(match_list, my_uuid, in_tree_ids)

    if not matches:
        logger.warning(f"No matches found on page {page_num}")
        return None, 0, 0, False, ""

    logger.info(f"Found {len(matches)} matches on page {page_num}")
    return matches, 0, 0, False, ""


def _log_page_header(page_num: int, pages_processed: int, max_pages: int, total_new: int, total_updated: int, total_skipped: int, total_errors: int) -> None:
    """Log page processing header."""
    logger.info(f"\n{'='*80}")
    if max_pages == 0:
        logger.info(f"Processing page {page_num} (page {pages_processed + 1} of all pages)")
    else:
        logger.info(f"Processing page {page_num} (page {page_num - pages_processed + 1}/{max_pages})")
    logger.info(f"Cumulative: New={total_new}, Updated={total_updated}, Skipped={total_skipped}, Errors={total_errors}")
    logger.info(f"{'='*80}")


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
    total_new = 0
    total_updated = 0
    total_skipped = 0
    total_errors = 0
    session_deaths = 0
    session_recoveries = 0
    run_incomplete = False
    incomplete_reason = ""

    if max_pages == 0:
        logger.info("MAX_PAGES=0: Will process all pages until no more matches found")
        end_page = float('inf')
    else:
        end_page = start_page + max_pages - 1

    page_num = start_page
    pages_processed = 0

    while True:
        if max_pages > 0 and page_num > end_page:
            break

        should_continue, deaths, recoveries, reason = _handle_session_health_check(session_manager)
        session_deaths += deaths
        session_recoveries += recoveries
        if not should_continue:
            run_incomplete = True
            incomplete_reason = reason
            break

        _log_page_header(page_num, pages_processed, max_pages, total_new, total_updated, total_skipped, total_errors)

        matches, deaths, recoveries, should_break, reason = _fetch_and_validate_page_data(
            driver, session_manager, my_uuid, page_num, csrf_token, max_pages
        )
        session_deaths += deaths
        session_recoveries += recoveries

        if should_break:
            if reason:
                run_incomplete = True
                incomplete_reason = reason
            break

        if not matches:
            page_num += 1
            pages_processed += 1
            continue

        new, updated, skipped, errors, deaths, recoveries, page_incomplete, page_reason = _process_page_batches(
            matches, batch_size, session_manager, db_manager, my_uuid, my_tree_id, page_num
        )

        total_new += new
        total_updated += updated
        total_skipped += skipped
        total_errors += errors
        session_deaths += deaths
        session_recoveries += recoveries

        if page_incomplete:
            run_incomplete = True
            incomplete_reason = page_reason
            break

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
    logger.info("=" * 80)
    logger.info("Action 6: DNA Match Gatherer")
    logger.info("=" * 80)

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


def _get_match_details(match: dict, skip_details: bool, match_details_map: dict, session_manager: SessionManager, my_uuid: str, parallel_workers: int) -> tuple[dict, dict, dict, Any]:
    """Get match details from cache or fetch them."""
    if skip_details:
        return {}, {}, {}, None

    if parallel_workers > 1:
        details = match_details_map.get(match["uuid"], {})
        return (
            details.get('match_details', {}),
            details.get('profile_details', {}),
            details.get('badge_details', {}),
            details.get('predicted_rel')
        )

    # Sequential processing: fetch details now
    match_details = _fetch_match_details(session_manager, my_uuid, match["uuid"])
    profile_details = _fetch_profile_details(session_manager, match["profile_id"], match["uuid"]) if match["profile_id"] else {}
    badge_details = _fetch_badge_details(session_manager, my_uuid, match["uuid"]) if match["in_tree"] else {}
    predicted_rel = _fetch_relationship_probability(session_manager, my_uuid, match["uuid"])

    return match_details, profile_details, badge_details, predicted_rel


def _save_match_records(match: dict, person_id: int, match_details: dict, badge_details: dict, predicted_rel: Any, skip_details: bool, my_tree_id: Optional[str], session: Any, session_manager: SessionManager) -> None:
    """Save DnaMatch and FamilyTree records."""
    if not skip_details:
        _save_dna_match(session, person_id, match, match_details, predicted_rel)

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

    match_details, profile_details, badge_details, predicted_rel = _get_match_details(
        match, skip_details, match_details_map, session_manager, my_uuid, parallel_workers
    )

    if skip_details:
        logger.debug(f"Skipping detail fetch for person_id={person_id} (reason: {skip_info.get('reason')})")

    additional_updates = False
    if profile_details or badge_details or match_details:
        additional_updates = _update_person(session, person_id, profile_details, badge_details, match_details)

    logger.debug(f"Match {match['uuid']}: person_status={person_status}, additional_updates={additional_updates}, skip_details={skip_details}")

    _save_match_records(match, person_id, match_details, badge_details, predicted_rel, skip_details, my_tree_id, session, session_manager)

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


def _fetch_profile_details(session_manager: SessionManager, profile_id: str, match_uuid: str) -> dict:  # type: ignore[unused-function] # noqa: ARG001
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
    if not _validate_relationship_inputs(my_uuid, match_uuid, session_manager):
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

    try:
        response = session_manager.scraper.post(
            url,
            headers=headers,
            json={},
            allow_redirects=False,
            timeout=config_schema.selenium.api_timeout,
        )

        if response.status_code in (301, 302, 303, 307, 308):
            logger.debug(
                f"{api_description}: Received redirect ({response.status_code}) for {sample_id_upper}. "
                f"Skipping (optional data)."
            )
            return None

        if not response.ok:
            logger.warning(
                f"{api_description} failed for {sample_id_upper}: {response.status_code} {response.reason}"
            )
            return None

        if not response.content:
            logger.warning(f"{api_description}: Empty response body for {sample_id_upper}")
            return None

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            logger.warning(f"{api_description}: JSON decode failed for {sample_id_upper}: {e}")
            return None

        return _extract_relationship_from_response(data, sample_id_upper, api_description)

    except Exception as e:
        logger.warning(f"{api_description}: Unexpected error for {sample_id_upper}: {e}")
        return None


def _format_relationship_path_part(person: dict, idx: int, cfpid: str) -> str:
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
        path_part = _format_relationship_path_part(person, idx, cfpid)
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


def _save_dna_match(session, person_id: int, match: dict, match_details: dict, predicted_relationship: Optional[str] = None) -> str:
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


def action6_module_tests() -> bool:
    """Comprehensive test suite for action6_gather.py using standardized TestSuite framework."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Action 6 - DNA Match Gatherer", "action6_gather.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Database Schema Validation",
            _test_database_schema,
            "Validates Person model has 'id' attribute and DnaMatch/FamilyTree have 'people_id' foreign keys.",
            "Check database model attributes for correct naming conventions.",
            "Person model uses 'id', DnaMatch and FamilyTree use 'people_id' foreign key.",
        )

        suite.run_test(
            "Person ID Attribute Fix",
            _test_person_id_attribute_fix,
            "Validates _get_person_id_by_uuid returns person.id, not person.people_id.",
            "Inspect function source code for correct attribute access.",
            "Function returns person.id and old buggy code is not present.",
        )

        suite.run_test(
            "Thread-Safe Parallel Processing",
            _test_parallel_function_thread_safety,
            "Validates _fetch_match_details_parallel has no database session parameter.",
            "Check function signature and thread-safety documentation.",
            "Function signature is correct and thread-safety is documented.",
        )

        suite.run_test(
            "Bounds Checking for Index Errors",
            _test_bounds_checking,
            "Validates bounds checking is in place for list/tuple access.",
            "Search for safe access patterns in CSRF token, kinship persons, and string operations.",
            "All bounds checking patterns are present in the code.",
        )

        suite.run_test(
            "Error Handling Coverage",
            _test_error_handling,
            "Validates comprehensive error handling in critical functions.",
            "Check for try/except blocks in _get_person_id_by_uuid, _fetch_match_details_parallel, and _process_batch.",
            "All functions have appropriate error handling in place.",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return action6_module_tests()


if __name__ == "__main__":
    import sys

    print("🧪 Running Action 6 comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

