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
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, unquote

# === THIRD-PARTY IMPORTS ===
from tqdm.auto import tqdm

# === LOCAL IMPORTS ===
from config import config_schema
from core.database_manager import DatabaseManager
from core.session_manager import SessionManager
from database import create_or_update_person, create_or_update_dna_match, create_or_update_family_tree
from dna_utils import (
    fetch_in_tree_status,
    fetch_match_list_page,
    get_csrf_token_for_dna_matches,
    nav_to_dna_matches_page,
)
from utils import _api_req, format_name


def coord(session_manager: SessionManager, start: int = 1):
    """Main entry point for Action 6: DNA Match Gatherer (start parameter ignored, always starts from page 1)."""
    logger.info("=" * 80)
    logger.info("Action 6: DNA Match Gatherer")
    logger.info("=" * 80)

    max_pages = config_schema.api.max_pages
    # Create database manager
    db_manager = DatabaseManager()
    batch_size = config_schema.batch_size

    logger.info(f"Configuration: MAX_PAGES={max_pages}, BATCH_SIZE={batch_size}")
    
    # Get my_uuid and my_tree_id
    my_uuid = session_manager.my_uuid
    my_tree_id = session_manager.my_tree_id
    
    if not my_uuid:
        logger.error("Cannot proceed: my_uuid is not set")
        return
    
    logger.info(f"My UUID: {my_uuid}")
    logger.info(f"My Tree ID: {my_tree_id}")

    # Navigate to DNA matches page to get CSRF token
    logger.info("Navigating to DNA matches page...")
    if not nav_to_dna_matches_page(session_manager):
        logger.error("Failed to navigate to DNA matches page")
        return

    # Get CSRF token
    driver = session_manager.driver
    csrf_token = get_csrf_token_for_dna_matches(driver)
    if not csrf_token:
        logger.error("Failed to get CSRF token")
        return

    # Process pages
    total_new = 0
    total_updated = 0
    total_skipped = 0
    total_errors = 0

    for page_num in range(1, max_pages + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing page {page_num}/{max_pages}...")
        logger.info(f"{'='*80}")

        # Fetch match list for this page
        api_response = fetch_match_list_page(driver, session_manager, my_uuid, page_num, csrf_token)
        if not api_response or not isinstance(api_response, dict):
            logger.warning(f"No API response for page {page_num}")
            continue

        # Extract matches from API response
        match_list = api_response.get("matchList", [])
        if not match_list:
            logger.warning(f"No matches in API response for page {page_num}")
            continue

        # Get sample IDs for in-tree status check
        sample_ids = [m.get("sampleId", "").upper() for m in match_list if m.get("sampleId")]
        if not sample_ids:
            logger.warning(f"No sample IDs found on page {page_num}")
            continue

        # Fetch in-tree status
        in_tree_ids = fetch_in_tree_status(driver, session_manager, my_uuid, sample_ids, csrf_token, page_num)
        logger.info(f"Found {len(in_tree_ids)} matches in tree on page {page_num}")

        # Refine matches
        matches = _refine_match_list(match_list, my_uuid, in_tree_ids)
        
        if not matches:
            logger.warning(f"No matches found on page {page_num}")
            continue
        
        logger.info(f"Found {len(matches)} matches on page {page_num}")
        
        # Process matches in batches
        for batch_start in range(0, len(matches), batch_size):
            batch_end = min(batch_start + batch_size, len(matches))
            batch = matches[batch_start:batch_end]
            
            logger.info(f"\n--- Batch {batch_start//batch_size + 1}: Processing matches {batch_start+1}-{batch_end} ---")
            
            # Process batch
            new, updated, skipped, errors = _process_batch(
                batch, session_manager, db_manager, my_uuid, my_tree_id
            )
            
            total_new += new
            total_updated += updated
            total_skipped += skipped
            total_errors += errors
            
            logger.info(f"Batch complete: New={new}, Updated={updated}, Skipped={skipped}, Errors={errors}")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Pages Processed: {max_pages}")
    logger.info(f"New Added: {total_new}")
    logger.info(f"Updated: {total_updated}")
    logger.info(f"Skipped: {total_skipped}")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"{'='*80}")

    return True

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

    try:
        for match in tqdm(batch, desc="Processing matches", leave=False):
            try:
                # Step 1: Create/update Person record (initial data)
                person_id, person_status = _save_person_with_status(session, match)
                if not person_id:
                    error_count += 1
                    continue

                # Step 2: Check if person was recently updated (skip fetching details if so)
                # Always fetch details for newly created people
                skip_details = False
                if person_status != "created":
                    skip_details = _should_skip_person_refresh(session, person_id)

                # Step 3: Fetch additional details (only if needed)
                match_details = {}
                profile_details = {}
                badge_details = {}
                predicted_rel = None

                if not skip_details:
                    match_details = _fetch_match_details(session_manager, my_uuid, match["uuid"])
                    profile_details = _fetch_profile_details(session_manager, match["profile_id"], match["uuid"]) if match["profile_id"] else {}
                    badge_details = _fetch_badge_details(session_manager, my_uuid, match["uuid"]) if match["in_tree"] else {}
                    predicted_rel = _fetch_relationship_probability(session_manager, my_uuid, match["uuid"])
                else:
                    logger.debug(f"Skipping detail fetch for person_id={person_id} (recently updated)")

                # Step 4: Update Person with additional data
                additional_updates = False
                if profile_details or badge_details or match_details:
                    additional_updates = _update_person(session, person_id, profile_details, badge_details, match_details)

                logger.debug(f"Match {match['uuid']}: person_status={person_status}, additional_updates={additional_updates}, skip_details={skip_details}")

                # Step 5: Create/update DnaMatch record
                _save_dna_match(session, person_id, match, match_details, predicted_rel)

                # Step 6: Create/update FamilyTree record (if in_tree)
                if match["in_tree"] and badge_details:
                    _save_family_tree(session, person_id, badge_details, my_tree_id, session_manager)

                # Count based on combined person status and additional updates
                if person_status == "created":
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
        logger.info(f"Batch committed successfully")

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
    """
    from database import Person
    from datetime import datetime, timezone, timedelta

    # Get refresh threshold from config
    refresh_days = config_schema.person_refresh_days

    # If refresh_days is 0, always fetch details
    if refresh_days == 0:
        return False

    # Get person's last update time
    person = session.query(Person).filter_by(id=person_id).first()
    if not person or not person.updated_at:
        return False

    # Calculate time since last update
    now = datetime.now(timezone.utc)
    last_updated = person.updated_at

    # Ensure last_updated is timezone-aware
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)

    time_since_update = now - last_updated
    threshold = timedelta(days=refresh_days)

    # Skip if updated within threshold
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
    else:
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

    # Update from profile details (only if different)
    if profile_details.get("last_logged_in") and profile_details["last_logged_in"] != person.last_logged_in:
        person.last_logged_in = profile_details["last_logged_in"]
        logger.debug(f"_update_person: Updated last_logged_in for person_id={person_id}")
        updated = True
    if "contactable" in profile_details and profile_details["contactable"] != person.contactable:
        person.contactable = profile_details["contactable"]
        logger.debug(f"_update_person: Updated contactable for person_id={person_id}")
        updated = True

    # Update from badge details (only if different)
    if badge_details.get("birth_year"):
        try:
            new_birth_year = int(badge_details["birth_year"])
            if new_birth_year != person.birth_year:
                person.birth_year = new_birth_year
                logger.debug(f"_update_person: Updated birth_year for person_id={person_id}")
                updated = True
        except (ValueError, TypeError):
            pass

    # Update from match details (administrator fields - only if different)
    if match_details.get("administrator_profile_id") and match_details["administrator_profile_id"] != person.administrator_profile_id:
        person.administrator_profile_id = match_details["administrator_profile_id"]
        logger.debug(f"_update_person: Updated administrator_profile_id for person_id={person_id}: {match_details['administrator_profile_id']}")
        updated = True
    if match_details.get("administrator_username") and match_details["administrator_username"] != person.administrator_username:
        person.administrator_username = match_details["administrator_username"]
        logger.debug(f"_update_person: Updated administrator_username for person_id={person_id}: {match_details['administrator_username']}")
        updated = True

    if updated:
        session.flush()

    logger.debug(f"_update_person: person_id={person_id}, updated={updated}")
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


def _fetch_profile_details(session_manager: SessionManager, profile_id: str, match_uuid: str) -> dict:
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


def _fetch_relationship_probability(session_manager: SessionManager, my_uuid: str, match_uuid: str) -> Optional[str]:
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
                csrf_token = unquote(cookie.get("value", "")).split("|")[0]
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
        if kinship_persons:
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
                if relationship:
                    relationship = relationship[0].upper() + relationship[1:]

            # Format: "Name LifeSpan (Relationship)"
            if lifespan:
                if relationship:
                    path_part = f"{name} {lifespan} ({relationship})"
                else:
                    path_part = f"{name} {lifespan}"
            else:
                if relationship:
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


def _save_dna_match(session, person_id: int, match: dict, match_details: dict, predicted_relationship: Optional[str] = None) -> str:
    """Save DnaMatch record to database."""
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

    result = create_or_update_dna_match(session, dna_data)
    logger.debug(f"DnaMatch result for person_id={person_id}: {result}")
    return result


def _save_family_tree(session, person_id: int, badge_details: dict, my_tree_id: Optional[str], session_manager: SessionManager) -> str:
    """Save FamilyTree record to database."""
    cfpid = badge_details.get("cfpid")
    person_name = badge_details.get("person_name_in_tree")

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

