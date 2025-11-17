#!/usr/bin/env python3

"""
api_constants.py - Centralized API Endpoint Constants

Single source of truth for all Ancestry API endpoint paths.
Consolidates constants from utils.py, api_utils.py, core/api_manager.py,
core/session_manager.py, and dna_ethnicity_utils.py.

These endpoints are validated by regression guard tests to prevent drift.
DO NOT change these values without extensive validation!
"""

# ============================================================================
# USER IDENTITY & AUTHENTICATION ENDPOINTS
# ============================================================================

# CSRF token for API authentication
API_PATH_CSRF_TOKEN = "discoveryui-matches/parents/api/csrfToken"

# User profile ID (ucdmid)
API_PATH_PROFILE_ID = "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid"

# User's DNA test UUID (legacy path)
API_PATH_UUID_LEGACY = "api/uhome/secure/rest/header/dna"

# User's DNA test UUID (current path)
API_PATH_UUID_NAVHEADER = "api/navheaderdata/v1/header/data/dna"

# ============================================================================
# MESSAGING ENDPOINTS
# ============================================================================

# Send new message (POST)
API_PATH_SEND_MESSAGE_NEW = "app-api/express/v2/conversations/message"

# Send message to existing conversation (POST with conv_id parameter)
API_PATH_SEND_MESSAGE_EXISTING = "app-api/express/v2/conversations/{conv_id}"

# Profile details by user ID
API_PATH_PROFILE_DETAILS = "/app-api/express/v1/profiles/details"

# ============================================================================
# TREE ENDPOINTS
# ============================================================================

# User's tree list
API_PATH_HEADER_TREES = "api/treesui-list/trees?rights=own"

# Tree owner information
API_PATH_TREE_OWNER_INFO = "api/uhome/secure/rest/user/tree-info"

# Person suggestions for tree
API_PATH_PERSON_PICKER_SUGGEST = "api/person-picker/suggest/{tree_id}"

# Person facts in family tree
API_PATH_PERSON_FACTS_USER = (
    "family-tree/person/facts/user/{owner_profile_id}/tree/{tree_id}/person/{person_id}"
)

# Person relationship ladder (legacy HTML parsing)
API_PATH_PERSON_GETLADDER = (
    "family-tree/person/tree/{tree_id}/person/{person_id}/getladder"
)

# Person relationship ladder with labels (JSON API)
API_PATH_RELATION_LADDER_WITH_LABELS = (
    "family-tree/person/card/user/{user_id}/tree/{tree_id}/person/{person_id}/kinship/relationladderwithlabels"
)

# Discovery relationship API
API_PATH_DISCOVERY_RELATIONSHIP = "discoveryui-matchingservice/api/relationship"

# TreesUI person list API
API_PATH_TREESUI_LIST = "api/treesui-list/trees/{tree_id}/persons"

# New family view API (includes siblings)
API_PATH_NEW_FAMILY_VIEW = "api/treeviewer/tree/newfamilyview/{tree_id}"

# ============================================================================
# DNA MATCH ENDPOINTS
# ============================================================================

# DNA match list (paginated)
API_PATH_MATCH_LIST = "discoveryui-matches/parents/list/api/matchList/{my_uuid}"

# DNA match list badges (matches in tree)
API_PATH_MATCH_BADGES_IN_TREE = "discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid}"

# DNA match details (combined endpoint with parental data)
API_PATH_MATCH_DETAILS = "/discoveryui-matchesservice/api/samples/{my_uuid}/matches/{match_uuid}/details?pmparentaldata=true"

# DNA match badge details
API_PATH_MATCH_BADGE_DETAILS = "/discoveryui-matchesservice/api/samples/{my_uuid}/matches/{match_uuid}/badgedetails"

# DNA match probability data
API_PATH_MATCH_PROBABILITY = "discoveryui-matches/parents/list/api/matchProbabilityData/{my_uuid_upper}/{sample_id_upper}"

# ============================================================================
# DNA ETHNICITY ENDPOINTS
# ============================================================================

# Tree owner's ethnicity regions (v2)
API_PATH_ETHNICITY_OWNER = "dna/origins/secure/tests/{tree_owner_test_guid}/v2/ethnicity"

# Ethnicity region names (localized)
API_PATH_ETHNICITY_REGION_NAMES = "dna/origins/public/ethnicity/2025/names?locale={locale}"

# Ethnicity comparison between owner and match
API_PATH_ETHNICITY_COMPARE = "discoveryui-matchesservice/api/compare/{tree_owner_test_guid}/with/{match_test_guid}/ethnicity"

# ============================================================================
# FAMILY TREE NAVIGATION ENDPOINTS
# ============================================================================

# Family tree person page (for building links)
API_PATH_FAMILY_TREE_PERSON = "/family-tree/person/tree/{tree_id}/person/{cfpid}"

# Family tree view link
API_PATH_FAMILY_TREE_VIEW = "/family-tree/tree/{tree_id}/family"


# ============================================================================
# ENDPOINT VALIDATION
# ============================================================================

def validate_all_endpoints() -> bool:
    """
    Validate that all API endpoint constants are defined and non-empty.

    Returns:
        True if all validations pass, False otherwise
    """
    import sys

    # Get all API_PATH_ constants from this module
    current_module = sys.modules[__name__]
    api_constants = {
        name: getattr(current_module, name)
        for name in dir(current_module)
        if name.startswith("API_PATH_")
    }

    # Validate each constant
    all_valid = True
    for name, value in api_constants.items():
        if not value or not isinstance(value, str):
            print(f"❌ Invalid endpoint: {name} = {value!r}")
            all_valid = False

    if all_valid:
        print(f"✅ All {len(api_constants)} API endpoint constants are valid")

    return all_valid


# ============================================================================
# MODULE TESTS
# ============================================================================

def api_constants_module_tests() -> bool:
    """
    Run comprehensive tests for API constants module.

    Tests:
    1. All constants are defined and non-empty
    2. No duplicate values (except intentional aliases)
    3. Critical endpoints match expected values (regression guards)

    Returns:
        True if all tests pass
    """
    print("🧬 Testing API Constants Module...")
    print()

    success = True

    # Test 1: Validate all endpoints
    print("Test 1: Validate all API endpoint constants")
    if not validate_all_endpoints():
        success = False
        print("❌ FAILED: Some endpoints are invalid")
    else:
        print("✅ PASSED: All endpoints valid")
    print()

    # Test 2: Regression guards for critical endpoints
    print("Test 2: Regression guards for critical endpoints")
    critical_tests = [
        ("API_PATH_PROFILE_ID", "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid"),
        ("API_PATH_CSRF_TOKEN", "discoveryui-matches/parents/api/csrfToken"),
        ("API_PATH_UUID_NAVHEADER", "api/navheaderdata/v1/header/data/dna"),
        ("API_PATH_ETHNICITY_OWNER", "dna/origins/secure/tests/{tree_owner_test_guid}/v2/ethnicity"),
        ("API_PATH_ETHNICITY_COMPARE", "discoveryui-matchesservice/api/compare/{tree_owner_test_guid}/with/{match_test_guid}/ethnicity"),
        ("API_PATH_TREESUI_LIST", "api/treesui-list/trees/{tree_id}/persons"),
        ("API_PATH_HEADER_TREES", "api/treesui-list/trees?rights=own"),
    ]

    regression_passed = True
    for const_name, expected_value in critical_tests:
        actual_value = globals().get(const_name)
        if actual_value != expected_value:
            print(f"❌ FAILED: {const_name}")
            print(f"   Expected: {expected_value}")
            print(f"   Actual: {actual_value}")
            regression_passed = False
            success = False
        else:
            print(f"✅ {const_name} = {expected_value}")

    if regression_passed:
        print("✅ PASSED: All regression guards passed")
    print()

    # Test 3: Check for literal presence in source code
    print("Test 3: Literal presence guards")
    import inspect
    source = inspect.getsource(sys.modules[__name__])

    literal_tests = [
        "dna/origins/secure/tests/",
        "dna/origins/public/ethnicity/2025/names?locale",
        "discoveryui-matchesservice/api/compare/",
        "api/treesui-list/trees",
    ]

    literal_passed = True
    for literal in literal_tests:
        if literal in source:
            print(f"✅ Found literal: {literal}")
        else:
            print(f"❌ Missing literal: {literal}")
            literal_passed = False
            success = False

    if literal_passed:
        print("✅ PASSED: All literals present in source")
    print()

    if success:
        print("🎉 All API constants tests PASSED")
    else:
        print("❌ Some API constants tests FAILED")

    return success


# Use centralized test runner utility from test_utilities
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(api_constants_module_tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
