#!/usr/bin/env python3

"""
setup_ethnicity_tracking.py - One-time setup for DNA ethnicity tracking

This script:
1. Fetches the tree owner's DNA ethnicity regions
2. Creates dynamic columns in the dna_match table for each region
3. Stores region metadata for future reference

This should be run once to set up the infrastructure. After that, Action 6
will automatically populate ethnicity data for new matches.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import json
import os
from typing import Any

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


ETHNICITY_METADATA_FILE = "ethnicity_regions.json"


def get_tree_owner_test_guid() -> str:
    """Get tree owner's DNA test GUID from environment."""
    my_uuid = os.getenv("MY_UUID")
    if not my_uuid:
        raise ValueError("MY_UUID not found in .env file")
    return my_uuid


def save_ethnicity_metadata(
    regions: list[dict[str, Any]],
    region_names: dict[str, str]
) -> None:
    """
    Save ethnicity region metadata to JSON file for future reference.
    
    Args:
        regions: List of region dicts from tree owner's ethnicity data
        region_names: Dict mapping region keys to names
    """
    metadata = {
        "tree_owner_regions": [
            {
                "key": region["key"],
                "name": region_names.get(region["key"], f"Unknown Region {region['key']}"),
                "percentage": region["percentage"],
                "column_name": sanitize_column_name(
                    region_names.get(region["key"], f"unknown_region_{region['key']}")
                )
            }
            for region in regions
        ]
    }
    
    with open(ETHNICITY_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved ethnicity metadata to {ETHNICITY_METADATA_FILE}")


def load_ethnicity_metadata() -> dict[str, Any]:
    """
    Load ethnicity region metadata from JSON file.
    
    Returns:
        Dict containing tree_owner_regions list
    """
    if not os.path.exists(ETHNICITY_METADATA_FILE):
        logger.warning(f"Ethnicity metadata file {ETHNICITY_METADATA_FILE} not found")
        return {"tree_owner_regions": []}
    
    with open(ETHNICITY_METADATA_FILE, 'r') as f:
        return json.load(f)


def column_exists(db_manager: DatabaseManager, table_name: str, column_name: str) -> bool:
    """
    Check if a column exists in a table.

    Args:
        db_manager: DatabaseManager instance
        table_name: Name of the table
        column_name: Name of the column

    Returns:
        True if column exists, False otherwise
    """
    with db_manager.get_session_context() as session:
        if not session:
            logger.error("Failed to get database session")
            return False

        result = session.execute(
            text(f"PRAGMA table_info({table_name})")
        ).fetchall()

        existing_columns = [row[1] for row in result]
        return column_name in existing_columns


def add_ethnicity_column(
    db_manager: DatabaseManager,
    column_name: str,
    region_name: str
) -> bool:
    """
    Add an ethnicity column to the dna_match table.

    Args:
        db_manager: DatabaseManager instance
        column_name: Sanitized column name
        region_name: Human-readable region name

    Returns:
        True if column was added, False if it already existed
    """
    if column_exists(db_manager, "dna_match", column_name):
        logger.info(f"Column '{column_name}' already exists in dna_match table")
        return False

    with db_manager.get_session_context() as session:
        if not session:
            logger.error("Failed to get database session")
            return False

        # Add column with INTEGER type (percentage 0-100)
        # Note: SQLite doesn't support comments in ALTER TABLE, so we log the purpose instead
        logger.debug(f"Adding column {column_name} for ethnicity percentage of {region_name}")
        sql = text(f"ALTER TABLE dna_match ADD COLUMN {column_name} INTEGER DEFAULT 0")
        session.execute(sql)
        session.commit()

    logger.info(f"Added column '{column_name}' to dna_match table for region '{region_name}'")
    return True


def setup_ethnicity_tracking() -> None:
    """
    Main setup function to initialize ethnicity tracking.
    
    This function:
    1. Fetches tree owner's ethnicity regions
    2. Fetches region name mappings
    3. Creates database columns for each region
    4. Saves metadata for future reference
    """
    logger.info("=" * 80)
    logger.info("DNA ETHNICITY TRACKING SETUP")
    logger.info("=" * 80)
    
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

    try:
        # Mark browser as needed
        sm.browser_manager.browser_needed = True

        # Start session with browser
        logger.info("Starting session...")
        if not sm.start_sess("Ethnicity Tracking Setup"):
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

        # Fetch tree owner's ethnicity regions
        logger.info("Fetching tree owner's ethnicity regions...")
        ethnicity_data = fetch_tree_owner_ethnicity_regions(sm, tree_owner_guid)

        if not ethnicity_data or "regions" not in ethnicity_data:
            logger.error("Failed to fetch tree owner's ethnicity regions")
            return
        
        regions = ethnicity_data["regions"]
        logger.info(f"Found {len(regions)} ethnicity regions for tree owner")

        # Fetch region names from public API
        region_keys = [region["key"] for region in regions]
        logger.info(f"Fetching ethnicity region names for {len(region_keys)} regions from public API...")
        region_names = fetch_ethnicity_region_names(sm, region_keys)

        if not region_names:
            logger.error("Failed to fetch region names from public API")
            return

        logger.info(f"Fetched {len(region_names)} region name mappings from public API")

        # Display tree owner's regions
        logger.info("\nTree Owner's DNA Ethnicity Regions:")
        logger.info("-" * 80)
        for region in sorted(regions, key=lambda r: r["percentage"], reverse=True):
            region_key = region["key"]
            # Try to get name from region_names dict, then from region itself, then use key
            region_name = region_names.get(region_key) or region.get("name") or region_key
            percentage = region["percentage"]
            logger.info(f"  {region_name:40s} {percentage:3d}%")
        logger.info("-" * 80)
        
        # Save metadata
        logger.info("\nSaving ethnicity metadata...")
        save_ethnicity_metadata(regions, region_names)
        
        # Initialize database manager
        logger.info("\nInitializing database manager...")
        db_manager = DatabaseManager()
        
        # Add columns to dna_match table
        logger.info("\nAdding ethnicity columns to dna_match table...")
        columns_added = 0
        columns_existed = 0
        
        for region in regions:
            region_key = region["key"]
            region_name = region_names.get(region_key, f"Unknown Region {region_key}")
            column_name = sanitize_column_name(region_name)
            
            if add_ethnicity_column(db_manager, column_name, region_name):
                columns_added += 1
            else:
                columns_existed += 1
        
        logger.info(f"\nColumns added: {columns_added}")
        logger.info(f"Columns already existed: {columns_existed}")
        
        logger.info("\n" + "=" * 80)
        logger.info("ETHNICITY TRACKING SETUP COMPLETE")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("1. Run the ethnicity backfill script to populate data for existing matches")
        logger.info("2. Action 6 will automatically collect ethnicity data for new matches")

    finally:
        sm.close_sess()


if __name__ == "__main__":
    setup_ethnicity_tracking()

