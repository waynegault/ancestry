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
import re
from typing import Any, Optional
from urllib.parse import urljoin

# === LOCAL IMPORTS ===
from config import config_schema
from core.session_manager import SessionManager
from utils import _api_req


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
    
    logger.info(f"Successfully fetched {len(response['regions'])} ethnicity regions for tree owner")
    return response


def fetch_ethnicity_region_names(
    session_manager: SessionManager, locale: str = "en-GB"
) -> Optional[dict[str, str]]:
    """
    Fetch the mapping of region keys to region names.
    
    Args:
        session_manager: Active SessionManager instance
        locale: Locale for region names (default: en-GB)
    
    Returns:
        Dict mapping region keys to region names, or None if failed
        
    Example response:
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
    
    logger.debug(f"Fetching ethnicity region names from: {url}")
    
    response = _api_req(
        url=url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        use_csrf_token=False,
        api_description="Ethnicity Region Names API"
    )
    
    if not response or not isinstance(response, dict):
        logger.error("Failed to fetch ethnicity region names")
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


def _test_tree_owner_ethnicity_fetch() -> bool:
    """Test fetching tree owner's ethnicity regions from API."""
    import os

    my_uuid = os.getenv("MY_UUID")
    if not my_uuid:
        logger.error("MY_UUID not found in .env")
        return False

    sm = SessionManager()

    try:
        # Mark browser as needed
        sm.browser_manager.browser_needed = True

        # Start session
        if not sm.start_sess("Ethnicity Utils Tests"):
            logger.error("Failed to start session")
            return False

        # Authenticate session
        from utils import _load_login_cookies, log_in, login_status
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

        # Ensure session ready (use "coord" to skip cookie checks)
        if not sm.ensure_session_ready("coord", skip_csrf=True):
            logger.error("Session not ready")
            return False

        if not sm.my_uuid:
            logger.error("UUID not available")
            return False

        # Navigate to DNA matches page
        from dna_utils import nav_to_dna_matches_page
        if not nav_to_dna_matches_page(sm):
            logger.error("Failed to navigate to DNA matches page")
            return False

        # Fetch tree owner's ethnicity
        ethnicity_data = fetch_tree_owner_ethnicity_regions(sm, my_uuid)

        if not ethnicity_data:
            logger.error("Failed to fetch tree owner ethnicity")
            return False

        if "regions" not in ethnicity_data:
            logger.error("Ethnicity data missing 'regions' field")
            return False

        regions = ethnicity_data["regions"]
        logger.info(f"✅ Successfully fetched {len(regions)} ethnicity regions")

        # Display regions
        for region in regions:
            logger.info(f"  Region {region['key']}: {region['percentage']}%")

        return True

    finally:
        sm.close_sess()


def _test_region_names_fetch() -> bool:
    """Test fetching region name mappings from API."""
    sm = SessionManager()

    try:
        # Mark browser as needed
        sm.browser_manager.browser_needed = True

        # Start session
        if not sm.start_sess("Ethnicity Utils Tests"):
            logger.error("Failed to start session")
            return False

        # Authenticate session
        from utils import _load_login_cookies, log_in, login_status
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

        # Ensure session ready (use "coord" to skip cookie checks)
        if not sm.ensure_session_ready("coord", skip_csrf=True):
            logger.error("Session not ready")
            return False

        # Navigate to DNA matches page
        from dna_utils import nav_to_dna_matches_page
        if not nav_to_dna_matches_page(sm):
            logger.error("Failed to navigate to DNA matches page")
            return False

        # Fetch region names
        region_names = fetch_ethnicity_region_names(sm)

        if not region_names:
            logger.error("Failed to fetch region names")
            return False

        logger.info(f"✅ Successfully fetched {len(region_names)} region names")

        # Display some examples
        for key, name in list(region_names.items())[:5]:
            logger.info(f"  {key}: {name}")

        return True

    finally:
        sm.close_sess()


def _test_ethnicity_comparison() -> bool:
    """Test fetching ethnicity comparison for a match."""
    import os

    my_uuid = os.getenv("MY_UUID")
    if not my_uuid:
        logger.error("MY_UUID not found in .env")
        return False

    # Use a known match UUID (Brent Husson from example)
    match_uuid = "B509B1EB-EE8B-4D28-89A4-6E9B93C4A727"

    sm = SessionManager()

    try:
        # Mark browser as needed
        sm.browser_manager.browser_needed = True

        # Start session
        if not sm.start_sess("Ethnicity Utils Tests"):
            logger.error("Failed to start session")
            return False

        # Authenticate session
        from utils import _load_login_cookies, log_in, login_status
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

        # Ensure session ready (use "coord" to skip cookie checks)
        if not sm.ensure_session_ready("coord", skip_csrf=True):
            logger.error("Session not ready")
            return False

        # Navigate to DNA matches page
        from dna_utils import nav_to_dna_matches_page
        if not nav_to_dna_matches_page(sm):
            logger.error("Failed to navigate to DNA matches page")
            return False

        # Fetch comparison
        comparison_data = fetch_ethnicity_comparison(sm, my_uuid, match_uuid)

        if not comparison_data:
            logger.warning(f"No ethnicity comparison data for match {match_uuid}")
            return True  # This is OK - not all matches have ethnicity data

        if "comparisons" not in comparison_data:
            logger.error("Comparison data missing 'comparisons' field")
            return False

        comparisons = comparison_data["comparisons"]
        logger.info(f"✅ Successfully fetched ethnicity comparison with {len(comparisons)} regions")

        # Display some comparisons
        for comp in comparisons[:5]:
            resource_id = comp.get("resourceId", "Unknown")
            left_sum = comp.get("leftSum", 0)
            right_sum = comp.get("rightSum", 0)
            logger.info(f"  Region {resource_id}: You={left_sum}%, Match={right_sum}%")

        return True

    finally:
        sm.close_sess()


if __name__ == "__main__":
    from test_framework import TestSuite

    suite = TestSuite("DNA Ethnicity Utils Tests", "dna_ethnicity_utils.py")
    suite.start_suite()

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

    # === API TESTS (require live session) ===
    suite.run_test(
        "Tree Owner Ethnicity Fetch",
        _test_tree_owner_ethnicity_fetch,
        test_summary="Validates fetching tree owner's ethnicity regions from Ancestry API",
        functions_tested="fetch_tree_owner_ethnicity_regions()",
        method_description="Call ethnicity API and parse response with regions and percentages",
        expected_outcome="Successfully fetch and parse ethnicity regions for tree owner",
    )

    suite.run_test(
        "Region Names Fetch",
        _test_region_names_fetch,
        test_summary="Validates fetching region name mappings from Ancestry API",
        functions_tested="fetch_ethnicity_region_names()",
        method_description="Call region names API and parse response with key-to-name mappings",
        expected_outcome="Successfully fetch region name mappings",
    )

    suite.run_test(
        "Ethnicity Comparison",
        _test_ethnicity_comparison,
        test_summary="Validates fetching ethnicity comparison between tree owner and match",
        functions_tested="fetch_ethnicity_comparison()",
        method_description="Call comparison API and parse response with ethnicity overlap data",
        expected_outcome="Successfully fetch comparison data or gracefully handle missing data",
    )

    suite.finish_suite()

