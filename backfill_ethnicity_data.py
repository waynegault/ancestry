#!/usr/bin/env python3

"""
backfill_ethnicity_data.py - Backfill ethnicity data for existing DNA matches

This temporary script fetches ethnicity comparison data for all existing DNA matches
and populates the ethnicity columns in the dna_match table.

After Action 6 is updated to collect ethnicity data automatically, this script
can be removed.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import ctypes
import os
import time
from typing import Any

# === THIRD-PARTY IMPORTS ===
from sqlalchemy import text
from tqdm.auto import tqdm

# === LOCAL IMPORTS ===
from core.database_manager import DatabaseManager
from core.session_manager import SessionManager
from dna_ethnicity_utils import (
    extract_match_ethnicity_percentages,
    fetch_ethnicity_comparison,
)
from setup_ethnicity_tracking import load_ethnicity_metadata

# Windows sleep prevention constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def prevent_sleep() -> None:
    """Prevent Windows from going to sleep during long-running operations."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        logger.info("✅ Sleep prevention enabled")
    except Exception as e:
        logger.warning(f"Failed to prevent sleep: {e}")


def allow_sleep() -> None:
    """Allow Windows to sleep again after operations complete."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        logger.info("Sleep prevention disabled")
    except Exception as e:
        logger.warning(f"Failed to restore sleep settings: {e}")


def get_global_rate_limiter_stats() -> dict[str, Any]:
    """
    Get statistics from the global rate limiter in utils.py.

    Returns:
        Dict with current_delay and other stats if available
    """
    try:
        from utils import get_rate_limiter
        limiter = get_rate_limiter()
        return {
            "current_delay": limiter.current_delay,
            "initial_delay": limiter.initial_delay,
            "max_delay": limiter.max_delay,
        }
    except Exception:
        return {"current_delay": 0.0, "initial_delay": 0.0, "max_delay": 0.0}


def get_tree_owner_test_guid() -> str:
    """Get tree owner's DNA test GUID from environment."""
    my_uuid = os.getenv("MY_UUID")
    if not my_uuid:
        raise ValueError("MY_UUID not found in .env file")
    return my_uuid


def get_all_dna_matches_with_test_guids(
    db_manager: DatabaseManager,
    ethnicity_columns: list[str]
) -> list[dict[str, Any]]:
    """
    Fetch all DNA matches that have test GUIDs (UUIDs) and no ethnicity data yet.

    Args:
        db_manager: DatabaseManager instance
        ethnicity_columns: List of ethnicity column names to check

    Returns:
        List of dicts with people_id and uuid (only matches without ethnicity data)
    """
    with db_manager.get_session_context() as session:
        if not session:
            logger.error("Failed to get database session")
            return []

        # Build WHERE clause to skip matches that already have ethnicity data
        # Skip if ANY ethnicity column has a non-zero value
        ethnicity_checks = " AND ".join([f"(dm.{col} IS NULL OR dm.{col} = 0)" for col in ethnicity_columns])

        # Query for all people with DNA matches and UUIDs, but no ethnicity data
        query = text(f"""
            SELECT p.id as people_id, p.uuid
            FROM people p
            INNER JOIN dna_match dm ON p.id = dm.people_id
            WHERE p.uuid IS NOT NULL AND p.uuid != ''
            AND ({ethnicity_checks})
        """)

        result = session.execute(query).fetchall()

        matches = [
            {"people_id": row[0], "uuid": row[1]}
            for row in result
        ]

        logger.info(f"Found {len(matches)} DNA matches with test GUIDs (without ethnicity data)")
        return matches


def update_match_ethnicity(
    db_manager: DatabaseManager,
    people_id: int,
    ethnicity_data: dict[str, int],
    column_mapping: dict[str, str]
) -> bool:
    """
    Update ethnicity columns for a DNA match.

    Args:
        db_manager: DatabaseManager instance
        people_id: Person ID to update
        ethnicity_data: Dict mapping region keys to percentages
        column_mapping: Dict mapping region keys to column names

    Returns:
        True if successful, False otherwise
    """
    if not ethnicity_data:
        return False

    with db_manager.get_session_context() as session:
        if not session:
            logger.error("Failed to get database session")
            return False

        # Build UPDATE statement dynamically
        set_clauses = []
        for region_key, percentage in ethnicity_data.items():
            column_name = column_mapping.get(region_key)
            if column_name:
                set_clauses.append(f"{column_name} = {percentage}")

        if not set_clauses:
            return False

        update_sql = text(
            f"UPDATE dna_match SET {', '.join(set_clauses)} WHERE people_id = {people_id}"
        )

        session.execute(update_sql)
        session.commit()

        return True


def backfill_ethnicity_data(max_matches: int | None = None, batch_size: int = 50, batch_delay: float = 5.0) -> None:  # noqa: PLR0911
    """
    Main backfill function to populate ethnicity data for existing matches.

    Args:
        max_matches: Maximum number of matches to process (None = all matches)
        batch_size: Number of matches to process before taking a break
        batch_delay: Seconds to wait between batches (to avoid rate limits)
    """
    logger.info("=" * 80)
    logger.info("DNA ETHNICITY DATA BACKFILL")
    logger.info("=" * 80)

    if max_matches:
        logger.info(f"⚠️  Limited to processing {max_matches} matches")
    logger.info(f"Batch size: {batch_size} matches")
    logger.info(f"Batch delay: {batch_delay}s")
    logger.info("")

    # Load ethnicity metadata
    logger.info("Loading ethnicity metadata...")
    metadata = load_ethnicity_metadata()

    if not metadata or not metadata.get("tree_owner_regions"):
        logger.error("No ethnicity metadata found. Please run setup_ethnicity_tracking.py first.")
        return

    tree_owner_regions = metadata["tree_owner_regions"]
    logger.info(f"Loaded {len(tree_owner_regions)} tree owner regions")

    # Build mappings
    region_keys = [region["key"] for region in tree_owner_regions]
    column_mapping = {region["key"]: region["column_name"] for region in tree_owner_regions}
    ethnicity_columns = [region["column_name"] for region in tree_owner_regions]

    # Prevent Windows from sleeping during long operation
    prevent_sleep()

    # Get tree owner's test GUID
    try:
        tree_owner_guid = get_tree_owner_test_guid()
        logger.info(f"Tree owner DNA test GUID: {tree_owner_guid}")
    except ValueError as e:
        logger.error(f"Failed to get tree owner test GUID: {e}")
        return

    # Initialize and authenticate session manager
    logger.info("Initializing session manager...")
    sm = SessionManager()

    logger.info("Initializing database manager...")
    db_manager = DatabaseManager()

    try:
        # Mark browser as needed
        sm.browser_manager.browser_needed = True

        # Start session with browser
        logger.info("Starting session...")
        if not sm.start_sess("Ethnicity Data Backfill"):
            logger.error("Failed to start session")
            return

        # Authenticate session
        logger.info("Authenticating session...")
        from utils import _load_login_cookies, log_in, login_status

        # Try to load saved cookies
        cookies_loaded = _load_login_cookies(sm)
        if cookies_loaded:
            logger.info("✅ Loaded saved cookies from previous session")
        else:
            logger.info("⚠️  No saved cookies found")

        # Check login status
        login_check = login_status(sm, disable_ui_fallback=True)

        if login_check is True:
            logger.info("✅ Already logged in")
        elif login_check is False:
            logger.info("⚠️  Not logged in - attempting login...")
            login_result = log_in(sm)
            if login_result != "LOGIN_SUCCEEDED":
                logger.error(f"Login failed: {login_result}")
                return
            logger.info("✅ Login successful")
        else:
            logger.error("Login status check failed critically")
            return

        # Ensure session is ready with all identifiers
        # Use "coord" as action name to skip cookie checks (cookies come after navigation)
        logger.info("Ensuring session is ready...")
        ready = sm.ensure_session_ready("coord", skip_csrf=True)
        if not ready:
            logger.error("Session not ready - cookies/identifiers missing")
            return

        if not sm.my_uuid:
            logger.error("UUID not available - session initialization incomplete")
            return

        logger.info(f"✅ Session ready with UUID: {sm.my_uuid}")

        # Navigate to DNA matches page
        logger.info("Navigating to DNA matches page...")
        from dna_utils import nav_to_dna_matches_page
        if not nav_to_dna_matches_page(sm):
            logger.error("Failed to navigate to DNA matches page")
            return

        logger.info("✅ Navigation successful")

        # Get all DNA matches with test GUIDs (skip those with existing ethnicity data)
        logger.info("Fetching DNA matches from database (skipping those with existing ethnicity data)...")
        matches = get_all_dna_matches_with_test_guids(db_manager, ethnicity_columns)

        if not matches:
            logger.warning("No DNA matches found with test GUIDs (or all already have ethnicity data)")
            return

        # Limit matches if requested
        if max_matches and len(matches) > max_matches:
            logger.info(f"Limiting to first {max_matches} of {len(matches)} matches")
            matches = matches[:max_matches]

        logger.info(f"Processing {len(matches)} DNA matches...")
        logger.info("🚀 Using global rate limiter from utils.py (automatically adjusts based on API responses)")
        logger.info("⏱️  Adding 1.5s delay between calls to avoid overwhelming ethnicity API")

        # Process each match
        updated_count = 0
        skipped_count = 0
        error_count = 0
        last_call_time = 0.0

        for idx, match in enumerate(tqdm(matches, desc="Backfilling ethnicity data"), start=1):
            people_id = match["people_id"]
            match_uuid = match["uuid"]

            try:
                # Add minimum delay between calls (ethnicity API is more rate-limited than others)
                elapsed = time.time() - last_call_time
                if elapsed < 1.5:
                    time.sleep(1.5 - elapsed)

                # Fetch ethnicity comparison (uses global rate limiter from utils.py)
                last_call_time = time.time()
                comparison_data = fetch_ethnicity_comparison(sm, tree_owner_guid, match_uuid)

                if not comparison_data:
                    logger.debug(f"No ethnicity comparison data for match {match_uuid}")
                    skipped_count += 1
                    continue

                # Extract percentages
                ethnicity_percentages = extract_match_ethnicity_percentages(
                    comparison_data, region_keys
                )

                # Update database
                if update_match_ethnicity(db_manager, people_id, ethnicity_percentages, column_mapping):
                    updated_count += 1
                else:
                    skipped_count += 1

                # Periodic status update with rate limiter stats
                if idx % batch_size == 0 and idx < len(matches):
                    stats = get_global_rate_limiter_stats()
                    logger.info(
                        f"📊 Progress: {idx}/{len(matches)} | "
                        f"Current delay: {stats['current_delay']:.2f}s | "
                        f"Updated: {updated_count} | Skipped: {skipped_count}"
                    )
                    # Optional batch delay (can be set to 0 now that we have dynamic limiting)
                    if batch_delay > 0:
                        time.sleep(batch_delay)

            except Exception as e:
                logger.error(f"Error processing match {match_uuid}: {e}")
                error_count += 1
                continue

        # Final statistics
        stats = get_global_rate_limiter_stats()
        logger.info("\n" + "=" * 80)
        logger.info("BACKFILL COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total matches processed: {len(matches)}")
        logger.info(f"Successfully updated: {updated_count}")
        logger.info(f"Skipped (no data): {skipped_count}")
        logger.info(f"Errors: {error_count}")
        logger.info("")
        logger.info("Global Rate Limiter Statistics:")
        logger.info(f"  Final delay: {stats['current_delay']:.2f}s")
        logger.info(f"  Initial delay: {stats['initial_delay']:.2f}s")
        logger.info(f"  Max delay: {stats['max_delay']:.2f}s")
        logger.info("=" * 80)

    finally:
        # Re-enable sleep
        allow_sleep()
        sm.close_sess()


if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    # Ethnicity API has stricter rate limits - use conservative defaults
    max_matches = None
    batch_size = 50  # Report progress every 50 matches
    batch_delay = 0.0  # No batch delay needed - 1.5s per-call delay handles it

    if len(sys.argv) > 1:
        try:
            max_matches = int(sys.argv[1])
            print(f"Processing maximum {max_matches} matches")
        except ValueError:
            print(f"Invalid max_matches argument: {sys.argv[1]}")
            print("Usage: python backfill_ethnicity_data.py [max_matches] [batch_size] [batch_delay]")
            print("Example: python backfill_ethnicity_data.py 100 25 10.0")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            print(f"Invalid batch_size argument: {sys.argv[2]}")
            sys.exit(1)

    if len(sys.argv) > 3:
        try:
            batch_delay = float(sys.argv[3])
        except ValueError:
            print(f"Invalid batch_delay argument: {sys.argv[3]}")
            sys.exit(1)

    backfill_ethnicity_data(max_matches=max_matches, batch_size=batch_size, batch_delay=batch_delay)

