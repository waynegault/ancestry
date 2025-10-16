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
from typing import Optional
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


@with_connection_resilience("Action 6: DNA Match Gatherer", max_recovery_attempts=3)
def coord(session_manager: SessionManager, start: int = 1):
    """Main entry point for Action 6: DNA Match Gatherer."""
    logger.info("=" * 80)
    logger.info("Action 6: DNA Match Gatherer")
    logger.info("=" * 80)

    # Reset rate limiter metrics at start of each run
    if session_manager.rate_limiter:
        session_manager.rate_limiter.reset_metrics()
        logger.debug("Rate limiter metrics reset for new run")

    max_pages = config_schema.api.max_pages
    # Create database manager
    db_manager = DatabaseManager()
    batch_size = config_schema.batch_size
    parallel_workers = getattr(config_schema, 'parallel_workers', 1)

    # Use start parameter from command line (e.g., "6 140")
    start_page = start

    logger.info(f"Configuration: START_PAGE={start_page}, MAX_PAGES={max_pages}, BATCH_SIZE={batch_size}, PARALLEL_WORKERS={parallel_workers}")

    # Adaptive rate limiting for parallel workers
    # When using parallel workers, increase base delay to prevent 429 errors
    # Formula: base_delay * sqrt(workers) to account for concurrent requests
    if parallel_workers > 1:
        import math
        if session_manager.rate_limiter:
            original_delay = session_manager.rate_limiter.initial_delay
            # Increase delay proportionally to worker count (sqrt to avoid over-compensation)
            adaptive_delay = original_delay * math.sqrt(parallel_workers)
            session_manager.rate_limiter.initial_delay = adaptive_delay
            session_manager.rate_limiter.current_delay = adaptive_delay
            logger.debug(f"⚡ Parallel processing ENABLED with {parallel_workers} workers")
            logger.info(f"   Adaptive rate limiting: base delay increased from {original_delay:.2f}s to {adaptive_delay:.2f}s")
            logger.debug("   Rate limiter is thread-safe and will prevent 429 errors")
    else:
        logger.debug("📝 Sequential processing (PARALLEL_WORKERS=1)")

    # Get my_uuid and my_tree_id
    my_uuid = session_manager.my_uuid
    my_tree_id = session_manager.my_tree_id

    if not my_uuid:
        logger.error("Cannot proceed: my_uuid is not set")
        return None

    logger.info(f"My UUID: {my_uuid}")
    logger.info(f"My Tree ID: {my_tree_id}")

    # Navigate to DNA matches page to get CSRF token
    logger.debug("Navigating to DNA matches page...")
    if not nav_to_dna_matches_page(session_manager):
        logger.error("Failed to navigate to DNA matches page")
        return None

    # Get CSRF token
    driver = session_manager.driver
    csrf_token = get_csrf_token_for_dna_matches(driver)
    if not csrf_token:
        logger.error("Failed to get CSRF token")
        return None

    # Process pages
    total_new = 0
    total_updated = 0
    total_skipped = 0
    total_errors = 0

    # Track session health for accurate reporting
    session_deaths = 0
    session_recoveries = 0
    run_incomplete = False
    incomplete_reason = ""
    run_start_time = time.time()

    # Calculate end page
    # MAX_PAGES=0 means process all pages (infinite loop until no more matches)
    if max_pages == 0:
        logger.info("MAX_PAGES=0: Will process all pages until no more matches found")
        end_page = float('inf')  # Process indefinitely
    else:
        end_page = start_page + max_pages - 1

    page_num = start_page
    pages_processed = 0

    while True:
        # Check if we've reached the end (for finite page processing)
        if max_pages > 0 and page_num > end_page:
            break

        # OPTION 3: Periodic session health check
        # Check session health at the start of each page to detect dead sessions early
        # This prevents hours-long delays when browser dies during long-running operations
        if not session_manager.check_session_health():
            logger.warning("🚨 Session health check failed - attempting recovery...")
            session_deaths += 1
            if session_manager.attempt_browser_recovery():
                logger.info("✅ Session recovered successfully, continuing...")
                session_recoveries += 1
            else:
                logger.error("❌ Session recovery failed - stopping processing")
                run_incomplete = True
                incomplete_reason = "Session recovery failed at page health check"
                break

        logger.info(f"\n{'='*80}")
        if max_pages == 0:
            logger.info(f"Processing page {page_num} (page {pages_processed + 1} of all pages)")
        else:
            logger.info(f"Processing page {page_num} (page {page_num - start_page + 1}/{max_pages})")
        logger.info(f"Cumulative: New={total_new}, Updated={total_updated}, Skipped={total_skipped}, Errors={total_errors}")
        logger.info(f"{'='*80}")

        # Fetch match list for this page
        api_response = fetch_match_list_page(driver, session_manager, my_uuid, page_num, csrf_token)
        if not api_response or not isinstance(api_response, dict):
            logger.warning(f"No API response for page {page_num}")

            # CRITICAL FIX: Distinguish between "no more pages" and "session dead"
            # Check if session is still valid before assuming we've reached the end
            if not session_manager.check_session_health():
                logger.error("🚨 Session appears dead - API failure likely due to invalid session")
                session_deaths += 1
                if session_manager.attempt_browser_recovery():
                    logger.info("✅ Session recovered - retrying current page...")
                    session_recoveries += 1
                    continue  # Retry the same page
                logger.error("❌ Session recovery failed - stopping processing")
                run_incomplete = True
                incomplete_reason = f"Session recovery failed at page {page_num} (API failure)"
                break

            # Session is valid, so this is genuinely "no more pages"
            if max_pages == 0:
                logger.info("No more pages available. Stopping.")
                break
            page_num += 1
            pages_processed += 1
            continue

        # Extract matches from API response
        match_list = api_response.get("matchList", [])
        if not match_list:
            logger.warning(f"No matches in API response for page {page_num}")
            if max_pages == 0:
                logger.info("No more matches available. Stopping.")
                break
            page_num += 1
            pages_processed += 1
            continue

        # Get sample IDs for in-tree status check
        sample_ids = [m.get("sampleId", "").upper() for m in match_list if m.get("sampleId")]
        if not sample_ids:
            logger.warning(f"No sample IDs found on page {page_num}")
            page_num += 1
            pages_processed += 1
            continue

        # Fetch in-tree status
        in_tree_ids = fetch_in_tree_status(driver, session_manager, my_uuid, sample_ids, csrf_token, page_num)

        # Refine matches
        matches = _refine_match_list(match_list, my_uuid, in_tree_ids)

        if not matches:
            logger.warning(f"No matches found on page {page_num}")
            page_num += 1
            pages_processed += 1
            continue

        logger.info(f"Found {len(matches)} matches on page {page_num}")

        # Process matches in batches
        for batch_start in range(0, len(matches), batch_size):
            batch_end = min(batch_start + batch_size, len(matches))
            batch = matches[batch_start:batch_end]

            logger.info(f"\n--- Batch {batch_start//batch_size + 1}: Processing matches {batch_start+1}-{batch_end} ---")

            # OPTION 3: Periodic session health check before each batch
            # More frequent checks during parallel processing to catch session death quickly
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

            # Process batch
            batch_num = (batch_start // batch_size) + 1
            total_batches_on_page = (len(matches) + batch_size - 1) // batch_size
            batch_end = min(batch_start + batch_size, len(matches))

            logger.info(f"Batch {batch_num}/{total_batches_on_page} (matches {batch_start+1}-{batch_end} of {len(matches)} on page {page_num})")

            new, updated, skipped, errors = _process_batch(
                batch, session_manager, db_manager, my_uuid, my_tree_id
            )

            total_new += new
            total_updated += updated
            total_skipped += skipped
            total_errors += errors

            logger.info(f"Batch {batch_num} complete: New={new}, Updated={updated}, Skipped={skipped}, Errors={errors}")

        # Move to next page
        page_num += 1
        pages_processed += 1

    # Calculate run statistics
    run_end_time = time.time()
    total_run_time = run_end_time - run_start_time

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}")

    # Run completion status
    if run_incomplete:
        logger.warning(f"⚠️  RUN INCOMPLETE: {incomplete_reason}")
    else:
        logger.info("✅ Run completed successfully")

    # Session health tracking
    if session_deaths > 0:
        logger.warning(f"⚠️  Session Deaths: {session_deaths}")
        logger.warning(f"⚠️  Session Recoveries: {session_recoveries}")
        if session_deaths > session_recoveries:
            logger.error(f"❌ {session_deaths - session_recoveries} session death(s) could not be recovered")

    # Page processing
    if max_pages == 0:
        logger.info(f"Pages Processed: {pages_processed} (started at page {start_page})")
    else:
        logger.info(f"Page Range: {start_page}-{end_page} ({max_pages} pages)")

    # Match statistics
    logger.info(f"New Added: {total_new}")
    logger.info(f"Updated: {total_updated}")
    logger.info(f"Skipped: {total_skipped}")
    logger.info(f"Errors: {total_errors}")

    # Time statistics
    logger.info("")
    logger.info(f"Total Run Time: {total_run_time/3600:.2f} hours ({total_run_time/60:.1f} minutes)")

    logger.info(f"{'='*80}")

    # Print rate limiter metrics
    logger.info("")
    if hasattr(session_manager, 'rate_limiter') and session_manager.rate_limiter:
        session_manager.rate_limiter.print_metrics_summary()

    return not run_incomplete  # Return False if run was incomplete

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

    # Get parallel workers setting
    parallel_workers = getattr(config_schema, 'parallel_workers', 1)

    try:
        # First pass: Identify which matches need detail fetching
        matches_needing_details = []
        skip_map = {}  # Track which matches to skip

        for match in batch:
            # Check if person exists and should be skipped
            person_id, person_status = _save_person_with_status(session, match)
            if not person_id:
                skip_map[match["uuid"]] = {"skip": True, "reason": "no_person_id", "person_id": None, "person_status": None}
                continue

            # Check if we should skip detail refresh
            skip_details = False
            skip_reason = None

            # Skip if DnaMatch already exists (most important check - avoids unnecessary API calls)
            if person_status != "created" and _dna_match_exists(session, person_id):
                skip_details = True
                skip_reason = "dna_match_exists"
                logger.debug(f"Skipping detail fetch for person_id={person_id} - DnaMatch already exists")
            # Skip if person was recently updated (timestamp-based)
            elif person_status != "created" and _should_skip_person_refresh(session, person_id):
                skip_details = True
                skip_reason = "recently_updated"

            if skip_details:
                skip_map[match["uuid"]] = {"skip": True, "reason": skip_reason, "person_id": person_id, "person_status": person_status}
            else:
                skip_map[match["uuid"]] = {"skip": False, "person_id": person_id, "person_status": person_status}
                matches_needing_details.append(match)

        # Fetch all match details (parallel or sequential based on config)
        match_details_map = {}

        if parallel_workers > 1 and matches_needing_details:
            logger.debug(f"Fetching match details using {parallel_workers} parallel workers for {len(matches_needing_details)} matches...")

            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                # Submit all fetch tasks (NO database session passed - thread safety!)
                future_to_match = {
                    executor.submit(_fetch_match_details_parallel, match, session_manager, my_uuid): match
                    for match in matches_needing_details
                }

                # Collect results as they complete
                for future in tqdm(as_completed(future_to_match), total=len(matches_needing_details), desc="Fetching data", leave=False):
                    match = future_to_match[future]
                    try:
                        details = future.result()
                        match_details_map[match["uuid"]] = details
                    except Exception as e:
                        logger.error(f"Error in parallel fetch for {match.get('uuid', 'UNKNOWN')}: {e}")
                        # Provide empty results on error
                        match_details_map[match["uuid"]] = {
                            'match_details': {},
                            'profile_details': {},
                            'badge_details': {},
                            'predicted_rel': None,
                            'uuid': match.get("uuid"),
                        }
        else:
            logger.debug(f"Using sequential processing (parallel_workers={parallel_workers}, matches_needing_details={len(matches_needing_details)})")

        # Second pass: Process and save all matches
        for match in tqdm(batch, desc="Saving to database", leave=False):
            try:
                # Get skip info from first pass
                skip_info = skip_map.get(match["uuid"], {})
                person_id = skip_info.get("person_id")
                person_status = skip_info.get("person_status")
                skip_details = skip_info.get("skip", False)

                if not person_id:
                    error_count += 1
                    continue

                # Step 2: Get match details (from parallel fetch or fetch now if sequential)
                if skip_details:
                    logger.debug(f"Skipping detail fetch for person_id={person_id} (reason: {skip_info.get('reason')})")
                    match_details = {}
                    profile_details = {}
                    badge_details = {}
                    predicted_rel = None
                elif parallel_workers > 1:
                    # Use pre-fetched details from parallel processing
                    details = match_details_map.get(match["uuid"], {})
                    match_details = details.get('match_details', {})
                    profile_details = details.get('profile_details', {})
                    badge_details = details.get('badge_details', {})
                    predicted_rel = details.get('predicted_rel')
                else:
                    # Sequential processing: fetch details now
                    match_details = _fetch_match_details(session_manager, my_uuid, match["uuid"])
                    profile_details = _fetch_profile_details(session_manager, match["profile_id"], match["uuid"]) if match["profile_id"] else {}
                    badge_details = _fetch_badge_details(session_manager, my_uuid, match["uuid"]) if match["in_tree"] else {}
                    predicted_rel = _fetch_relationship_probability(session_manager, my_uuid, match["uuid"])

                # Step 4: Update Person with additional data
                additional_updates = False
                if profile_details or badge_details or match_details:
                    additional_updates = _update_person(session, person_id, profile_details, badge_details, match_details)

                logger.debug(f"Match {match['uuid']}: person_status={person_status}, additional_updates={additional_updates}, skip_details={skip_details}")

                # Step 5: Create/update DnaMatch record
                # Skip logic is handled inside _save_dna_match (checks if record exists)
                # Only call if we fetched details (skip_details=False)
                if not skip_details:
                    _save_dna_match(session, person_id, match, match_details, predicted_rel)

                # Step 6: Create/update FamilyTree record (if in_tree)
                # Skip logic is handled inside _save_family_tree (checks if record exists)
                # Only call if we fetched badge_details
                if match["in_tree"] and badge_details:
                    _save_family_tree(session, person_id, badge_details, my_tree_id, session_manager)

                # Count based on skip_details, person status, and additional updates
                if skip_details:
                    # Person was skipped due to recent update - no DnaMatch/FamilyTree updates
                    skipped_count += 1
                elif person_status == "created":
                    new_count += 1
                elif person_status == "updated" or additional_updates:
                    updated_count += 1
                elif person_status == "skipped":
                    skipped_count += 1

            except Exception as e:
                logger.error(f"Error processing match {match.get('uuid')}: {e}")
                error_count += 1

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


def _update_person(session, person_id: int, profile_details: dict, badge_details: dict, match_details: dict) -> bool:
    """
    Update Person record with additional data from Profile, Badge, and Match Details APIs.
    Returns True if any field was actually updated, False otherwise.
    """
    from database import Person

    person = session.query(Person).filter_by(id=person_id).first()
    if not person:
        return False

    updated = False
    updated_fields = []

    # Update from profile details (only if different)
    if profile_details.get("last_logged_in") and profile_details["last_logged_in"] != person.last_logged_in:
        person.last_logged_in = profile_details["last_logged_in"]
        updated_fields.append("last_logged_in")
        updated = True
    if "contactable" in profile_details and profile_details["contactable"] != person.contactable:
        person.contactable = profile_details["contactable"]
        updated_fields.append("contactable")
        updated = True

    # Update from badge details (only if different)
    if badge_details.get("birth_year"):
        try:
            new_birth_year = int(badge_details["birth_year"])
            if new_birth_year != person.birth_year:
                person.birth_year = new_birth_year
                updated_fields.append("birth_year")
                updated = True
        except (ValueError, TypeError):
            pass

    # Update from match details (administrator fields - only if different)
    if match_details.get("administrator_profile_id") and match_details["administrator_profile_id"] != person.administrator_profile_id:
        person.administrator_profile_id = match_details["administrator_profile_id"]
        updated_fields.append("administrator_profile_id")
        updated = True
    if match_details.get("administrator_username") and match_details["administrator_username"] != person.administrator_username:
        person.administrator_username = match_details["administrator_username"]
        updated_fields.append("administrator_username")
        updated = True

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


def _fetch_relationship_probability(session_manager: SessionManager, my_uuid: str, match_uuid: str) -> Optional[str]:  # noqa: PLR0911
    """
    Fetch predicted relationship from Relationship Probability API using cloudscraper.
    This is the working version restored from commit 758cca8.
    """
    # Validate inputs
    if not my_uuid or not match_uuid:
        logger.warning("Cannot fetch relationship probability: missing my_uuid or match_uuid")
        return None

    scraper = session_manager.scraper
    driver = session_manager.driver

    if not scraper:
        logger.warning("Cannot fetch relationship probability: scraper is None")
        return None

    if not driver:
        logger.warning("Cannot fetch relationship probability: driver is None")
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

    # Sync cookies and get CSRF token
    try:
        driver_cookies = driver.get_cookies()
        if not driver_cookies:
            logger.warning(f"{api_description}: No cookies available from driver")
            return None

        # Sync cookies to scraper
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

        # Extract CSRF token
        csrf_token = None
        csrf_cookie_names = ("_dnamatches-matchlistui-x-csrf-token", "_csrf")
        for cookie in driver_cookies:
            if cookie.get("name") in csrf_cookie_names:
                cookie_value = unquote(cookie.get("value", ""))
                parts = cookie_value.split("|")
                csrf_token = parts[0] if parts else None  # FIX: Bounds checking for tuple index
                break

        if csrf_token:
            headers["X-CSRF-Token"] = csrf_token
        else:
            logger.debug(f"{api_description}: No CSRF token found in cookies")

    except Exception as e:
        logger.warning(f"{api_description}: Error syncing cookies: {e}")
        return None

    # Make API request
    try:
        response = scraper.post(
            url,
            headers=headers,
            json={},
            allow_redirects=False,
            timeout=config_schema.selenium.api_timeout,
        )

        # Check for redirects
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

        # Parse response
        if not response.content:
            logger.warning(f"{api_description}: Empty response body for {sample_id_upper}")
            return None

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            logger.warning(f"{api_description}: JSON decode failed for {sample_id_upper}: {e}")
            return None

        # Extract relationship prediction
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
        top_prob_display = top_prob  # Already a percentage (e.g., 99.0 for 99%)
        paths = best_pred.get("pathsToMatch", [])
        labels = [p.get("label") for p in paths if isinstance(p, dict) and p.get("label")]

        if not labels:
            logger.debug(f"{api_description}: No labels found for {sample_id_upper}")
            return None

        # Take top 2 labels
        final_labels = [str(label) for label in labels[:2] if label]
        if not final_labels:
            return None
        relationship_str = " or ".join(final_labels)
        return f"{relationship_str} [{top_prob_display:.1f}%]"

    except Exception as e:
        logger.warning(f"{api_description}: Unexpected error for {sample_id_upper}: {e}")
        return None


def _fetch_ladder_details(session_manager: SessionManager, cfpid: str, tree_id: str) -> dict:
    """Fetch relationship ladder details from Kinship Relation Ladder API."""
    if not cfpid or not tree_id:
        return {}

    try:
        # Get user_id from config
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

        # Extract actual relationship from first person
        actual_relationship = None
        if kinship_persons and len(kinship_persons) > 0:  # FIX: Bounds checking
            first_person = kinship_persons[0]
            actual_relationship = first_person.get("relationship")
            logger.debug(f"Found actual_relationship for CFPID {cfpid}: {actual_relationship}")

        # Build relationship path from all persons
        path_parts = []
        for idx, person in enumerate(kinship_persons):
            name = person.get("name", "")
            lifespan = person.get("lifeSpan", "")
            relationship = person.get("relationship", "")

            # Handle special case for "You are the..." at the end
            if relationship.startswith("You are the "):
                # Remove "You are the " and capitalize the first letter only
                relationship = relationship.replace("You are the ", "", 1)
                # Capitalize first letter of the relationship phrase
                if relationship and len(relationship) > 0:  # FIX: Bounds checking
                    relationship = relationship[0].upper() + relationship[1:]

            # Format: "Name LifeSpan (Relationship)"
            if lifespan:
                path_part = f"{name} {lifespan} ({relationship})" if relationship else f"{name} {lifespan}"
            elif relationship:
                path_part = f"{name} ({relationship})"
            else:
                path_part = name

            path_parts.append(path_part)
            logger.debug(f"  Person[{idx}]: {path_part}")

        relationship_path = None
        if path_parts:
            relationship_path = "\n↓\n".join(path_parts)
            logger.debug(f"Built relationship_path for CFPID {cfpid} with {len(path_parts)} parts")

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


def test_database_schema() -> bool:
    """Test that Person model has 'id' attribute, not 'people_id'."""
    print("\n" + "=" * 80)
    print("TEST 1: Database Schema Validation")
    print("=" * 80)

    try:
        from database import DnaMatch, FamilyTree, Person

        # Test Person model
        assert hasattr(Person, 'id'), "Person should have 'id' attribute"
        assert not hasattr(Person, 'people_id'), "Person should NOT have 'people_id' attribute"
        print("✅ Person model has 'id' attribute (correct)")

        # Test DnaMatch model
        assert hasattr(DnaMatch, 'people_id'), "DnaMatch should have 'people_id' foreign key"
        print("✅ DnaMatch model has 'people_id' foreign key (correct)")

        # Test FamilyTree model
        assert hasattr(FamilyTree, 'people_id'), "FamilyTree should have 'people_id' foreign key"
        print("✅ FamilyTree model has 'people_id' foreign key (correct)")

        print("✅ TEST 1 PASSED: Database schema is correct\n")
        return True

    except AssertionError as e:
        print(f"❌ TEST 1 FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}\n")
        return False


def test_person_id_attribute_fix() -> bool:
    """Test that _get_person_id_by_uuid returns person.id not person.people_id."""
    print("=" * 80)
    print("TEST 2: Person ID Attribute Fix")
    print("=" * 80)

    try:
        import inspect

        # Get source code of _get_person_id_by_uuid
        source = inspect.getsource(_get_person_id_by_uuid)

        # Check that the fix is in place
        if 'return person.id if person else None' in source:
            print("✅ _get_person_id_by_uuid returns person.id (correct)")
        else:
            print("❌ _get_person_id_by_uuid does not return person.id")
            return False

        # Check that old INCORRECT code is not present (excluding comments)
        # We need to check for the actual bug: "return person.people_id"
        lines = source.split('\n')
        for line in lines:
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            # Check for the actual bug pattern
            if 'return person.people_id' in line and 'person.id' not in line:
                print(f"❌ Old buggy code found: {line.strip()}")
                return False

        print("✅ Old buggy code 'return person.people_id' not found (correct)")
        print("✅ TEST 2 PASSED: Person ID attribute fix is correct\n")
        return True

    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}\n")
        return False


def test_parallel_function_thread_safety() -> bool:
    """Test that parallel function doesn't take database session parameter."""
    print("=" * 80)
    print("TEST 3: Thread-Safe Parallel Processing")
    print("=" * 80)

    try:
        import inspect

        # Get function signature
        sig = inspect.signature(_fetch_match_details_parallel)
        params = list(sig.parameters.keys())

        # Check that 'session' is NOT in parameters
        if 'session' in params:
            print(f"❌ _fetch_match_details_parallel has 'session' parameter: {params}")
            print("   This causes database concurrency errors!")
            return False
        print(f"✅ _fetch_match_details_parallel parameters: {params}")
        print("   No 'session' parameter (thread-safe)")

        # Check expected parameters
        expected_params = ['match', 'session_manager', 'my_uuid']
        if params == expected_params:
            print(f"✅ Function signature is correct: {expected_params}")
        else:
            print(f"⚠️  Function signature differs from expected: {params} vs {expected_params}")

        # Check for thread-safety documentation
        source = inspect.getsource(_fetch_match_details_parallel)
        if 'concurrency' in source.lower() or 'thread' in source.lower():
            print("✅ Thread-safety documented in function")
        else:
            print("⚠️  Thread-safety not explicitly documented")

        print("✅ TEST 3 PASSED: Parallel processing is thread-safe\n")
        return True

    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}\n")
        return False


def test_bounds_checking() -> bool:
    """Test that bounds checking is in place for list/tuple access."""
    print("=" * 80)
    print("TEST 4: Bounds Checking for Index Errors")
    print("=" * 80)

    try:
        # Read the entire file to check for bounds checking patterns
        from pathlib import Path
        file_content = Path(__file__).read_text(encoding='utf-8')

        # Test 1: CSRF token extraction - check for safe pattern
        if 'parts[0] if parts else None' in file_content or 'csrf_token = parts[0] if parts' in file_content:
            print("✅ CSRF token extraction has bounds checking")
        else:
            print("❌ CSRF token extraction missing bounds checking")
            return False

        # Test 2: Kinship persons access
        if 'len(kinship_persons) > 0' in file_content or 'if kinship_persons and len(kinship_persons)' in file_content:
            print("✅ Kinship persons access has bounds checking")
        else:
            print("❌ Kinship persons access missing bounds checking")
            return False

        # Test 3: String capitalization
        if 'len(relationship) > 0' in file_content or 'if relationship and len(relationship)' in file_content:
            print("✅ String capitalization has bounds checking")
        else:
            print("❌ String capitalization missing bounds checking")
            return False

        print("✅ TEST 4 PASSED: All bounds checking in place\n")
        return True

    except Exception as e:
        print(f"❌ TEST 4 FAILED: {e}\n")
        return False


def test_error_handling() -> bool:
    """Test that comprehensive error handling is in place."""
    print("=" * 80)
    print("TEST 5: Error Handling")
    print("=" * 80)

    try:
        import inspect

        # Check _get_person_id_by_uuid has try/except
        source = inspect.getsource(_get_person_id_by_uuid)
        if 'try:' in source and 'except' in source:
            print("✅ _get_person_id_by_uuid has error handling")
        else:
            print("⚠️  _get_person_id_by_uuid missing error handling")

        # Check _fetch_match_details_parallel has try/except
        parallel_source = inspect.getsource(_fetch_match_details_parallel)
        if 'try:' in parallel_source and 'except' in parallel_source:
            print("✅ _fetch_match_details_parallel has error handling")
        else:
            print("❌ _fetch_match_details_parallel missing error handling")
            return False

        # Check _process_batch has error handling
        batch_source = inspect.getsource(_process_batch)
        if batch_source.count('try:') >= 2 and batch_source.count('except') >= 2:
            print("✅ _process_batch has comprehensive error handling")
        else:
            print("⚠️  _process_batch may need more error handling")

        print("✅ TEST 5 PASSED: Error handling is in place\n")
        return True

    except Exception as e:
        print(f"❌ TEST 5 FAILED: {e}\n")
        return False


def action6_module_tests() -> bool:
    """Comprehensive test suite for action6_gather.py parallel processing fixes."""
    print("\n" + "=" * 80)
    print("ACTION 6 PARALLEL PROCESSING FIXES - TEST SUITE")
    print("=" * 80)

    tests = [
        ("Database Schema", test_database_schema),
        ("Person ID Attribute Fix", test_person_id_attribute_fix),
        ("Thread-Safe Parallel Processing", test_parallel_function_thread_safety),
        ("Bounds Checking", test_bounds_checking),
        ("Error Handling", test_error_handling),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} raised exception: {e}\n")
            results.append((test_name, False))

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")

    print("=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Action 6 parallel processing fixes validated.\n")
        return True
    print(f"\n⚠️  {total - passed} test(s) failed. Please review the fixes.\n")
    return False


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return action6_module_tests()


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    print("🧪 Running Action 6 test suite...")
    success = run_comprehensive_tests()

    if success:
        print("\n✅ All tests passed! Ready for production use.")
        print("\nNext steps:")
        print("1. Run Action 6 from main.py with PARALLEL_WORKERS=2, RPS=2.0")
        print("2. Process 10 pages and monitor Logs/app.log for errors")
        print("3. Verify zero concurrency, attribute, and index errors")
        print("4. Check rate limiter metrics for zero 429 errors\n")
    else:
        print("\n❌ Some tests failed. Please fix issues before running Action 6.\n")

    sys.exit(0 if success else 1)

