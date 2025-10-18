#!/usr/bin/env python3

"""
migrate_ethnicity_columns.py - Migrate ethnicity columns from region IDs to region names

This script:
1. Reads the current ethnicity_regions.json metadata
2. Creates new columns with region names (e.g., ethnicity_north_east_scotland)
3. Copies data from old columns (e.g., ethnicity_08302) to new columns
4. Drops old columns
5. Updates ethnicity_regions.json with new column names
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import json
import os
from pathlib import Path

# === THIRD-PARTY IMPORTS ===
from sqlalchemy import text

# === LOCAL IMPORTS ===
from core.database_manager import DatabaseManager
from core.session_manager import SessionManager
from dna_ethnicity_utils import (
    fetch_ethnicity_region_names,
    fetch_tree_owner_ethnicity_regions,
    sanitize_column_name,
)
from setup_ethnicity_tracking import (
    column_exists,
    load_ethnicity_metadata,
)


def get_tree_owner_test_guid() -> str:
    """Get tree owner's DNA test GUID from environment."""
    my_uuid = os.getenv("MY_UUID")
    if not my_uuid:
        raise ValueError("MY_UUID not found in .env file")
    return my_uuid


def migrate_ethnicity_columns() -> None:  # noqa: PLR0911
    """
    Migrate ethnicity columns from region IDs to region names.
    """
    logger.info("=" * 80)
    logger.info("ETHNICITY COLUMN MIGRATION")
    logger.info("=" * 80)

    # Load current metadata
    logger.info("Loading current ethnicity metadata...")
    metadata = load_ethnicity_metadata()

    if not metadata or not metadata.get("tree_owner_regions"):
        logger.error("No ethnicity metadata found. Please run setup_ethnicity_tracking.py first.")
        return

    current_regions = metadata["tree_owner_regions"]
    logger.info(f"Found {len(current_regions)} regions in metadata")

    # Get tree owner's test GUID
    try:
        tree_owner_guid = get_tree_owner_test_guid()
        logger.info(f"Tree owner DNA test GUID: {tree_owner_guid}")
    except ValueError as e:
        logger.error(f"Failed to get tree owner test GUID: {e}")
        return

    # Initialize session manager to fetch fresh ethnicity data
    logger.info("Initializing session manager...")
    sm = SessionManager()

    try:
        # Mark browser as needed
        sm.browser_manager.browser_needed = True

        # Start session with browser
        logger.info("Starting session...")
        if not sm.start_sess("Ethnicity Column Migration"):
            logger.error("Failed to start session")
            return

        # Authenticate session
        logger.info("Authenticating...")
        from utils import _load_login_cookies, log_in, login_status

        # Try to load saved cookies
        cookies_loaded = _load_login_cookies(sm)
        if cookies_loaded:
            logger.info("✅ Loaded saved cookies")
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
        # Use "coord" as action name to skip cookie checks
        logger.info("Ensuring session is ready...")
        ready = sm.ensure_session_ready("coord", skip_csrf=True)
        if not ready:
            logger.error("Session not ready - cookies/identifiers missing")
            return

        if not sm.my_uuid:
            logger.error("UUID not available - session initialization incomplete")
            return

        logger.info(f"✅ Session ready with UUID: {sm.my_uuid}")

        # Navigate to DNA matches page (required for ethnicity API to work)
        logger.info("Navigating to DNA matches page...")
        from dna_utils import nav_to_dna_matches_page
        if not nav_to_dna_matches_page(sm):
            logger.error("Failed to navigate to DNA matches page")
            return

        logger.info("✅ Navigation successful")

        # Fetch fresh ethnicity data with region names
        logger.info("Fetching tree owner's ethnicity regions with names...")
        ethnicity_data = fetch_tree_owner_ethnicity_regions(sm, tree_owner_guid)

        if not ethnicity_data or "regions" not in ethnicity_data:
            logger.error("Failed to fetch tree owner's ethnicity regions")
            return

        regions = ethnicity_data["regions"]
        logger.info(f"Fetched {len(regions)} regions")

        # Fetch region names from public API
        region_keys = [region["key"] for region in regions]
        logger.info(f"Fetching region names for {len(region_keys)} regions from public API...")
        region_names = fetch_ethnicity_region_names(sm, region_keys)

        if not region_names:
            logger.error("Failed to fetch region names from public API")
            return

        logger.info(f"Fetched {len(region_names)} region name mappings")

        # Create mapping of old column names to new column names
        migration_map = []
        for region in regions:
            region_key = region["key"]
            # Get name from API, fallback to region key if not found
            region_name: str = region_names.get(region_key, region_key)  # type: ignore
            old_column = f"ethnicity_{region_key.lower()}"
            new_column = sanitize_column_name(region_name)

            migration_map.append({
                "key": region_key,
                "name": region_name,
                "percentage": region["percentage"],
                "old_column": old_column,
                "new_column": new_column,
            })

            logger.info(f"  {region_name:40s} {region['percentage']:3d}%")
            logger.info(f"    Migration: {old_column} → {new_column}")

        # Initialize database manager
        logger.info("\nInitializing database manager...")
        db_manager = DatabaseManager()

        # Perform migration for each region
        logger.info("\nMigrating columns...")
        logger.info("-" * 80)

        for mapping in migration_map:
            old_col = mapping["old_column"]
            new_col = mapping["new_column"]
            region_name = mapping["name"]

            # Check if old column exists
            if not column_exists(db_manager, "dna_match", old_col):
                logger.warning(f"Old column '{old_col}' does not exist - skipping")
                continue

            # Check if new column already exists
            if column_exists(db_manager, "dna_match", new_col):
                logger.info(f"New column '{new_col}' already exists - skipping creation")
            else:
                # Create new column
                logger.info(f"Creating new column '{new_col}' for {region_name}...")
                with db_manager.get_session_context() as session:
                    if not session:
                        logger.error("Failed to get database session")
                        continue

                    sql = text(f"ALTER TABLE dna_match ADD COLUMN {new_col} INTEGER DEFAULT 0")
                    session.execute(sql)
                    session.commit()
                logger.info(f"  ✅ Created column '{new_col}'")

            # Copy data from old column to new column
            logger.info(f"Copying data from '{old_col}' to '{new_col}'...")
            with db_manager.get_session_context() as session:
                if not session:
                    logger.error("Failed to get database session")
                    continue

                sql = text(f"UPDATE dna_match SET {new_col} = {old_col}")
                session.execute(sql)
                session.commit()

                # Count how many rows were updated
                count_sql = text(f"SELECT COUNT(*) FROM dna_match WHERE {new_col} IS NOT NULL")
                count_result = session.execute(count_sql).fetchone()
                if count_result:
                    logger.info(f"  ✅ Copied {count_result[0]} rows")
                else:
                    logger.info(f"  ✅ Copied data from {old_col} to {new_col}")

            # Drop old column (SQLite requires recreating the table, so we'll skip this for now)
            # Instead, we'll just set old column values to NULL to mark them as migrated
            logger.info(f"Clearing old column '{old_col}'...")
            with db_manager.get_session_context() as session:
                if not session:
                    logger.error("Failed to get database session")
                    continue

                sql = text(f"UPDATE dna_match SET {old_col} = NULL")
                session.execute(sql)
                session.commit()
                logger.info(f"  ✅ Cleared old column '{old_col}'")

        # Update metadata file with new column names
        logger.info("\nUpdating ethnicity_regions.json with new column names...")
        new_metadata = {
            "tree_owner_regions": [
                {
                    "key": mapping["key"],
                    "name": mapping["name"],
                    "percentage": mapping["percentage"],
                    "column_name": mapping["new_column"],
                }
                for mapping in migration_map
            ]
        }

        Path("ethnicity_regions.json").write_text(
            json.dumps(new_metadata, indent=2), encoding="utf-8"
        )

        logger.info("  ✅ Updated ethnicity_regions.json")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("MIGRATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Migrated {len(migration_map)} ethnicity columns:")
        for mapping in migration_map:
            logger.info(f"  {mapping['name']:40s} → {mapping['new_column']}")
        logger.info("=" * 80)
        logger.info("\nOld columns have been cleared (set to NULL).")
        logger.info("You can manually drop them later if needed using SQLite tools.")
        logger.info("=" * 80)

    finally:
        sm.close_sess()


if __name__ == "__main__":
    migrate_ethnicity_columns()

