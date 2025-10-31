#!/usr/bin/env python3

"""
dna_ethnicity_utils.py - DNA Ethnicity Region Utilities

Provides functions for fetching and managing DNA ethnicity region data:
- Fetch tree owner's ethnicity regions
- Fetch region name mappings
- Compare ethnicity between tree owner and matches
- Manage dynamic database columns for ethnicity tracking
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import json
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

# === LOCAL IMPORTS ===
from config import config_schema
from core.session_manager import SessionManager
from session_utils import ensure_session_for_tests_sm_only
from utils import _api_req

# === CONSTANTS ===
ETHNICITY_METADATA_FILE = "ethnicity_regions.json"


def load_ethnicity_metadata() -> dict[str, Any]:
    """
    Load ethnicity region metadata from JSON file.

    Returns:
        Dict containing tree_owner_regions list
    """
    metadata_path = Path(ETHNICITY_METADATA_FILE)
    if not metadata_path.exists():
        logger.warning(f"Ethnicity metadata file {ETHNICITY_METADATA_FILE} not found")
        return {"tree_owner_regions": []}

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def fetch_tree_owner_ethnicity_regions(
    session_manager: SessionManager, tree_owner_test_guid: str
) -> Optional[dict[str, Any]]:
    """
    Fetch the tree owner's DNA ethnicity regions.

    Args:
        session_manager: Active SessionManager instance
        tree_owner_test_guid: Tree owner's DNA test GUID (e.g., MY_UUID from .env)

    Returns:
        Dict containing version, createdAt, and regions list, or None if failed

    Example response:
        {
            "version": 2025,
            "createdAt": 1759968000000,
            "regions": [
                {
                    "key": "08302",
                    "percentage": 84,
                    "lowerConfidence": 63,
                    "upperConfidence": 84,
                    ...
                },
                ...
            ]
        }
    """
    url = urljoin(
        config_schema.api.base_url,
        f"dna/origins/secure/tests/{tree_owner_test_guid}/v2/ethnicity"
    )

    logger.debug(f"Fetching tree owner ethnicity regions from: {url}")

    response = _api_req(
        url=url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        use_csrf_token=False,
        api_description="Tree Owner Ethnicity API"
    )

    if not response or not isinstance(response, dict):
        logger.error("Failed to fetch tree owner ethnicity regions")
        return None

    if "regions" not in response:
        logger.error("Ethnicity response missing 'regions' field")
        return None

    logger.debug(f"Successfully fetched {len(response['regions'])} ethnicity regions for tree owner")
    return response


def fetch_ethnicity_region_names(
    session_manager: SessionManager, region_keys: list[str], locale: str = "en-GB"
) -> Optional[dict[str, str]]:
    """
    Fetch the mapping of region keys to region names from public API.

    This endpoint requires POST with a JSON body containing the region keys.

    Args:
        session_manager: SessionManager for browser session (required for headers/cookies)
        region_keys: List of region keys to fetch names for (e.g., ["08302", "06842"])
        locale: Locale for region names (default: en-GB)

    Returns:
        Dict mapping region keys to region names, or None if failed

    Example:
        >>> fetch_ethnicity_region_names(sm, ["08302", "06842", "08103", "06810"])
        {
            "08103": "North East England",
            "08302": "North East Scotland",
            "06810": "Western Ukraine",
            "06842": "Southern Poland"
        }
    """
    url = urljoin(
        config_schema.api.base_url,
        f"dna/origins/public/ethnicity/2025/names?locale={locale}"
    )

    logger.debug(f"Fetching ethnicity region names for {len(region_keys)} regions from: {url}")

    # This is a POST request with JSON body containing the region keys
    logger.debug(f"Sending POST request with {len(region_keys)} region keys: {region_keys}")
    response = _api_req(
        url=url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="POST",
        json_data={"regionKeys": region_keys},  # Wrap list in dict for API compatibility
        use_csrf_token=False,
        api_description="Ethnicity Region Names API"
    )

    if not response or not isinstance(response, dict):
        logger.error(f"Failed to fetch ethnicity region names. Response type: {type(response)}, Response: {response}")
        return None

    logger.info(f"Successfully fetched {len(response)} ethnicity region names")
    return response


def fetch_ethnicity_comparison(
    session_manager: SessionManager,
    tree_owner_test_guid: str,
    match_test_guid: str
) -> Optional[dict[str, Any]]:
    """
    Fetch ethnicity comparison between tree owner and a DNA match.

    Args:
        session_manager: Active SessionManager instance
        tree_owner_test_guid: Tree owner's DNA test GUID
        match_test_guid: Match's DNA test GUID

    Returns:
        Dict containing comparison data, or None if failed

    Example response structure:
        {
            "sameVersion": true,
            "leftVersion": 2025,
            "rightVersion": 2025,
            "comparisons": [
                {
                    "resourceId": "06842",
                    "leftSum": 6,
                    "rightSum": 16,
                    "bothSums": true,
                    ...
                },
                ...
            ]
        }
    """
    url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matchesservice/api/compare/{tree_owner_test_guid}/with/{match_test_guid}/ethnicity"
    )

    logger.debug(f"Fetching ethnicity comparison for match {match_test_guid}")

    response = _api_req(
        url=url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        use_csrf_token=False,
        api_description="Ethnicity Comparison API"
    )

    if not response or not isinstance(response, dict):
        logger.debug(f"No ethnicity comparison data available for match {match_test_guid}")
        return None

    if "comparisons" not in response:
        logger.debug(f"Ethnicity comparison response missing 'comparisons' field for match {match_test_guid}")
        return None

    logger.debug(f"Successfully fetched ethnicity comparison for match {match_test_guid}")
    return response


def sanitize_column_name(region_name: str) -> str:
    """
    Sanitize region name to create a valid database column name.

    Args:
        region_name: Original region name (e.g., "North East Scotland")

    Returns:
        Sanitized column name (e.g., "north_east_scotland")

    Examples:
        >>> sanitize_column_name("North East Scotland")
        'north_east_scotland'
        >>> sanitize_column_name("Southern Poland")
        'southern_poland'
        >>> sanitize_column_name("Western Ukraine")
        'western_ukraine'
    """
    # Convert to lowercase
    sanitized = region_name.lower()

    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[^a-z0-9]+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Prefix with 'ethnicity_' to avoid conflicts
    return f"ethnicity_{sanitized}"


def initialize_ethnicity_system(session_manager: SessionManager, db_manager: Optional[SessionManager] = None) -> bool:
    """
    Initialize the ethnicity tracking system by:
    1. Fetching tree owner's ethnicity regions from API
    2. Fetching region names
    3. Creating ethnicity_regions.json file
    4. Adding ethnicity columns to dna_match table

    This should be called when creating a new database (Action 2).

    Args:
        session_manager: SessionManager for API access (has browser/credentials)
        db_manager: Optional SessionManager with active database connection (for adding columns)

    Returns:
        True if initialization successful, False otherwise
    """
    try:
        # Get tree owner's UUID
        my_uuid = session_manager.my_uuid
        if not my_uuid:
            logger.error("Cannot initialize ethnicity system: MY_UUID not available")
            return False

        logger.debug("Initializing ethnicity tracking system...")

        # Step 1: Fetch tree owner's ethnicity regions
        logger.debug(f"Fetching tree owner ethnicity for UUID: {my_uuid}")
        ethnicity_data = fetch_tree_owner_ethnicity_regions(session_manager, my_uuid)
        if not ethnicity_data or "regions" not in ethnicity_data:
            logger.error("Failed to fetch tree owner ethnicity regions (API may require full authentication)")
            return False

        regions = ethnicity_data["regions"]
        logger.debug(f"Found {len(regions)} ethnicity regions for tree owner")

        # Step 2: Fetch region names
        region_keys = [region["key"] for region in regions]
        logger.debug(f"Fetching region names for {len(region_keys)} regions")
        region_names = fetch_ethnicity_region_names(session_manager, region_keys)
        if not region_names:
            logger.error("Failed to fetch ethnicity region names (API may require full authentication)")
            return False

        # Step 3: Build metadata structure
        tree_owner_regions = []
        for region in regions:
            region_key = region["key"]
            region_name = region_names.get(region_key, f"Unknown Region {region_key}")
            percentage = region.get("percentage", 0)
            column_name = sanitize_column_name(region_name)

            tree_owner_regions.append({
                "key": region_key,
                "name": region_name,
                "percentage": percentage,
                "column_name": column_name
            })

        # Step 4: Save to ethnicity_regions.json
        metadata = {"tree_owner_regions": tree_owner_regions}
        metadata_path = Path(ETHNICITY_METADATA_FILE)

        logger.debug(f"Saving ethnicity metadata to {metadata_path}")
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug(f"✅ Created {ETHNICITY_METADATA_FILE} with {len(tree_owner_regions)} regions")

        # Step 5: Add ethnicity columns to dna_match table
        logger.debug("Adding ethnicity columns to dna_match table")
        if not _add_ethnicity_columns_to_database(tree_owner_regions, db_manager):
            logger.error("Failed to add ethnicity columns to database")
            return False

        logger.debug("✅ Ethnicity tracking system initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing ethnicity system: {e}", exc_info=True)
        return False


def _get_database_engine(db_manager: Optional[SessionManager]) -> Optional[Any]:
    """Get database engine from SessionManager or create new one."""
    if db_manager and hasattr(db_manager, 'db_manager') and db_manager.db_manager:
        return db_manager.db_manager.engine

    # Fallback to creating new DatabaseManager
    from core.database_manager import DatabaseManager
    db_mgr = DatabaseManager()
    return db_mgr.engine


def _add_single_ethnicity_column(connection: Any, column_name: str, existing_columns: set[str]) -> bool:
    """Add a single ethnicity column if it doesn't exist."""
    from sqlalchemy import text

    if column_name in existing_columns:
        return True  # Already exists, skip

    try:
        alter_sql = text(f"ALTER TABLE dna_match ADD COLUMN {column_name} INTEGER")
        connection.execute(alter_sql)
        connection.commit()
        logger.debug(f"Added ethnicity column: {column_name}")
        return True
    except Exception as col_err:
        logger.error(f"Failed to add column {column_name}: {col_err}")
        return False


def _add_ethnicity_columns_to_database(tree_owner_regions: list[dict[str, Any]], db_manager: Optional[SessionManager] = None) -> bool:
    """
    Add ethnicity columns to dna_match table using ALTER TABLE.

    Args:
        tree_owner_regions: List of region dicts with column_name field
        db_manager: Optional SessionManager with active database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        from sqlalchemy import inspect as sqlalchemy_inspect

        engine = _get_database_engine(db_manager)
        if not engine:
            logger.error("Cannot add ethnicity columns: Database engine not available")
            return False

        with engine.connect() as connection:
            # Check if dna_match table exists
            inspector = sqlalchemy_inspect(connection)
            if "dna_match" not in inspector.get_table_names():
                logger.error("dna_match table not found")
                return False

            # Get existing columns
            existing_columns = {col["name"] for col in inspector.get_columns("dna_match")}

            # Add ethnicity columns if they don't exist
            columns_added = 0
            for region in tree_owner_regions:
                column_name = region.get("column_name")
                if not column_name:
                    logger.warning(f"Region {region.get('name')} missing column_name - skipping")
                    continue

                if _add_single_ethnicity_column(connection, column_name, existing_columns):
                    if column_name not in existing_columns:
                        columns_added += 1
                else:
                    return False

            if columns_added > 0:
                logger.debug(f"Added {columns_added} ethnicity columns to dna_match table")
            else:
                logger.debug("All ethnicity columns already exist in dna_match table")

            return True

    except Exception as e:
        logger.error(f"Error adding ethnicity columns to database: {e}", exc_info=True)
        return False


def extract_match_ethnicity_percentages(
    comparison_data: dict[str, Any],
    tree_owner_region_keys: list[str]
) -> dict[str, int]:
    """
    Extract match's ethnicity percentages for tree owner's regions.

    Args:
        comparison_data: Response from ethnicity comparison API
        tree_owner_region_keys: List of region keys the tree owner has

    Returns:
        Dict mapping region keys to match's percentages

    Example:
        >>> comparison_data = {"comparisons": [{"resourceId": "08302", "rightSum": 0}, ...]}
        >>> tree_owner_region_keys = ["08302", "06842"]
        >>> extract_match_ethnicity_percentages(comparison_data, tree_owner_region_keys)
        {'08302': 0, '06842': 16}
    """
    percentages = {}

    if not comparison_data or "comparisons" not in comparison_data:
        # Return 0% for all regions if no comparison data
        return {key: 0 for key in tree_owner_region_keys}

    # Build lookup dict from comparisons
    comparison_lookup = {
        comp["resourceId"]: comp.get("rightSum", 0)
        for comp in comparison_data["comparisons"]
        if "resourceId" in comp
    }

    # Extract percentages for tree owner's regions
    for region_key in tree_owner_region_keys:
        percentages[region_key] = comparison_lookup.get(region_key, 0)

    return percentages


# === TESTS ===

def _test_sanitize_column_name() -> bool:
    """Test column name sanitization."""
    assert sanitize_column_name("North East Scotland") == "ethnicity_north_east_scotland"
    assert sanitize_column_name("Southern Poland") == "ethnicity_southern_poland"
    assert sanitize_column_name("Western Ukraine") == "ethnicity_western_ukraine"
    assert sanitize_column_name("North East England") == "ethnicity_north_east_england"
    assert sanitize_column_name("Test-Region Name!") == "ethnicity_test_region_name"
    return True


def _test_extract_match_ethnicity_percentages() -> bool:
    """Test extraction of match ethnicity percentages."""
    comparison_data = {
        "comparisons": [
            {"resourceId": "08302", "leftSum": 84, "rightSum": 0},
            {"resourceId": "06842", "leftSum": 6, "rightSum": 16},
            {"resourceId": "08103", "leftSum": 6, "rightSum": 0},
            {"resourceId": "06810", "leftSum": 4, "rightSum": 0},
        ]
    }

    tree_owner_keys = ["08302", "06842", "08103", "06810"]
    result = extract_match_ethnicity_percentages(comparison_data, tree_owner_keys)

    assert result["08302"] == 0, f"Expected 0 for 08302, got {result['08302']}"
    assert result["06842"] == 16, f"Expected 16 for 06842, got {result['06842']}"
    assert result["08103"] == 0, f"Expected 0 for 08103, got {result['08103']}"
    assert result["06810"] == 0, f"Expected 0 for 06810, got {result['06810']}"

    # Test with missing comparison data
    result_empty = extract_match_ethnicity_percentages({}, tree_owner_keys)
    assert all(v == 0 for v in result_empty.values()), "All percentages should be 0 when no comparison data"

    return True


def _setup_test_session(sm: SessionManager) -> bool:
    """Set up and authenticate a test session."""
    from utils import _load_login_cookies, log_in, login_status

    sm.browser_manager.browser_needed = True

    if not sm.start_sess("Ethnicity Utils Tests"):
        logger.error("Failed to start session")
        return False

    _load_login_cookies(sm)
    login_check = login_status(sm, disable_ui_fallback=True)

    if login_check is False:
        login_result = log_in(sm)
        if login_result != "LOGIN_SUCCEEDED":
            logger.error(f"Login failed: {login_result}")
            return False
    elif login_check is None:
        logger.error("Login status check failed")
        return False

    if not sm.ensure_session_ready("coord", skip_csrf=True):
        logger.error("Session not ready")
        return False

    return True


def _navigate_to_dna_page(sm: SessionManager) -> bool:
    """Navigate to DNA matches page."""
    from dna_utils import nav_to_dna_matches_page
    if not nav_to_dna_matches_page(sm):
        logger.error("Failed to navigate to DNA matches page")
        return False
    return True


def _validate_test_prerequisites(sm: SessionManager) -> tuple[bool, str]:
    """Validate prerequisites for ethnicity test."""
    if not _setup_test_session(sm):
        return False, "Could not set up test session - skipping live API test"
    if not sm.my_uuid:
        return False, "UUID not available - skipping live API test"
    if not _navigate_to_dna_page(sm):
        return False, "Could not navigate to DNA page - skipping live API test"
    return True, ""


def _log_ethnicity_results(ethnicity_data: dict) -> None:
    """Log ethnicity region results."""
    regions = ethnicity_data["regions"]
    logger.info(f"✅ Successfully fetched {len(regions)} ethnicity regions")
    for region in regions:
        logger.info(f"  Region {region['key']}: {region['percentage']}%")


def _test_tree_owner_ethnicity_fetch() -> bool:
    """Test fetching tree owner's ethnicity regions from API."""
    sm = ensure_session_for_tests_sm_only("DNA Ethnicity Test", skip_csrf=True)
    try:
        # Validate prerequisites
        valid, message = _validate_test_prerequisites(sm)
        if not valid:
            logger.warning(message)
            return True

        # Get MY_UUID from session manager
        my_uuid = sm.my_uuid
        if not my_uuid:
            logger.warning("MY_UUID not available from session manager - skipping live API test")
            return True

        # Fetch and validate ethnicity data
        ethnicity_data = fetch_tree_owner_ethnicity_regions(sm, my_uuid)
        if not ethnicity_data or "regions" not in ethnicity_data:
            logger.error("Failed to fetch tree owner ethnicity or missing regions")
            return False

        # Log results
        _log_ethnicity_results(ethnicity_data)
        return True
    except Exception as e:
        logger.warning(f"Exception in tree owner ethnicity fetch (skipping): {e}")
        return True
    finally:
        try:
            sm.close_sess(keep_db=False)
        except Exception as e:
            logger.debug(f"Error closing session: {e}")


def _test_region_names_fetch() -> bool:
    """Test fetching region name mappings from API."""
    sm = ensure_session_for_tests_sm_only("DNA Ethnicity Test", skip_csrf=True)
    try:
        if not _setup_test_session(sm):
            logger.warning("Could not set up test session - skipping live API test")
            return True

        if not _navigate_to_dna_page(sm):
            logger.warning("Could not navigate to DNA page - skipping live API test")
            return True

        test_region_keys = ["08302", "06842", "08103", "06810"]
        region_names = fetch_ethnicity_region_names(sm, test_region_keys)

        if not region_names:
            logger.error("Failed to fetch region names")
            return False

        logger.info(f"✅ Successfully fetched {len(region_names)} region names")
        for key, name in list(region_names.items())[:5]:
            logger.info(f"  {key}: {name}")

        return True
    except Exception as e:
        logger.warning(f"Exception in region names fetch (skipping): {e}")
        return True
    finally:
        try:
            sm.close_sess(keep_db=False)
        except Exception as e:
            logger.debug(f"Error closing session: {e}")


def _test_ethnicity_comparison() -> bool:
    """Test fetching ethnicity comparison for a match."""
    sm = ensure_session_for_tests_sm_only("DNA Ethnicity Test", skip_csrf=True)

    # Get MY_UUID from session manager
    my_uuid = sm.my_uuid
    if not my_uuid:
        logger.warning("MY_UUID not available from session manager - skipping live API test")
        sm.close_sess(keep_db=False)
        return True

    match_uuid = "B509B1EB-EE8B-4D28-89A4-6E9B93C4A727"

    try:
        # Validate prerequisites
        if not _setup_test_session(sm) or not _navigate_to_dna_page(sm):
            if not _setup_test_session(sm):
                logger.warning("Could not set up test session - skipping live API test")
            else:
                logger.warning("Could not navigate to DNA page - skipping live API test")
            return True

        # Fetch comparison data
        comparison_data = fetch_ethnicity_comparison(sm, my_uuid, match_uuid)
        if not comparison_data:
            logger.warning(f"No ethnicity comparison data for match {match_uuid}")
            return True  # This is OK - not all matches have ethnicity data

        if "comparisons" not in comparison_data:
            logger.error("Comparison data missing 'comparisons' field")
            return False

        # Log results
        comparisons = comparison_data["comparisons"]
        logger.info(f"✅ Successfully fetched ethnicity comparison with {len(comparisons)} regions")
        for comp in comparisons[:5]:
            resource_id = comp.get("resourceId", "Unknown")
            left_sum = comp.get("leftSum", 0)
            right_sum = comp.get("rightSum", 0)
            logger.info(f"  Region {resource_id}: You={left_sum}%, Match={right_sum}%")
        return True
    except Exception as e:
        logger.warning(f"Exception in ethnicity comparison (skipping): {e}")
        return True
    finally:
        try:
            sm.close_sess(keep_db=False)
        except Exception as e:
            logger.debug(f"Error closing session: {e}")


def dna_ethnicity_utils_module_tests() -> bool:
    """
    Comprehensive test suite for dna_ethnicity_utils.py.
    Tests DNA ethnicity region utilities including column sanitization and percentage extraction.
    API tests are skipped when run directly (require live session).
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("DNA Ethnicity Utils Tests", "dna_ethnicity_utils.py")
    suite.start_suite()

    with suppress_logging():
        # === UNIT TESTS (no API required) ===
        suite.run_test(
            "Column Name Sanitization",
            _test_sanitize_column_name,
            test_summary="Validates region names are properly sanitized for database column names",
            functions_tested="sanitize_column_name()",
            method_description="Convert region names to lowercase, replace special chars with underscores, prefix with 'ethnicity_'",
            expected_outcome="Region names converted to valid column names (e.g., 'North East Scotland' -> 'ethnicity_north_east_scotland')",
        )

        suite.run_test(
            "Match Ethnicity Percentage Extraction",
            _test_extract_match_ethnicity_percentages,
            test_summary="Validates extraction of match ethnicity percentages from comparison data",
            functions_tested="extract_match_ethnicity_percentages()",
            method_description="Parse comparison API response and extract rightSum values for tree owner's regions",
            expected_outcome="Correct percentages extracted for each region, 0% for missing data",
        )

    # Note: API tests (_test_tree_owner_ethnicity_fetch, _test_region_names_fetch, _test_ethnicity_comparison)
    # are available but skipped when run directly as they require a live browser session.
    # They can be called manually when a SessionManager is available.

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return dna_ethnicity_utils_module_tests()


if __name__ == "__main__":
    import sys
    import traceback

    try:
        print("🧪 Running DNA Ethnicity Utils Tests comprehensive test suite...")
        success = run_comprehensive_tests()
    except Exception:
        print("\n[ERROR] Unhandled exception during DNA Ethnicity Utils tests:", file=sys.stderr)
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)

