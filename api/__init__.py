"""API Integration Package.

Provides unified API management including:
- api_constants: Centralized API endpoint constants
- api_utils: API request orchestration and authentication
- api_search_core: Core API search functionality
- api_search_utils: API search utilities and helpers
- tree_update: Tree modification service for Ancestry.com
"""

from api.api_constants import (
    API_PATH_CSRF_TOKEN,
    API_PATH_DISCOVERY_RELATIONSHIP,
    API_PATH_EDIT_RELATIONSHIPS,
    API_PATH_ETHNICITY_COMPARE,
    API_PATH_ETHNICITY_OWNER,
    API_PATH_ETHNICITY_REGION_NAMES,
    API_PATH_FAMILY_TREE_PERSON,
    API_PATH_FAMILY_TREE_VIEW,
    API_PATH_HEADER_TREES,
    API_PATH_MATCH_BADGE_DETAILS,
    API_PATH_MATCH_BADGES_IN_TREE,
    API_PATH_MATCH_DETAILS,
    API_PATH_MATCH_LIST,
    API_PATH_MATCH_PROBABILITY,
    API_PATH_NEW_FAMILY_VIEW,
    API_PATH_PERSON_FACTS_USER,
    API_PATH_PERSON_GETLADDER,
    API_PATH_PERSON_PICKER_SUGGEST,
    API_PATH_PROFILE_DETAILS,
    API_PATH_PROFILE_ID,
    API_PATH_RELATION_LADDER_WITH_LABELS,
    API_PATH_SEND_MESSAGE_EXISTING,
    API_PATH_SEND_MESSAGE_NEW,
    API_PATH_SHARED_MATCHES,
    API_PATH_TREE_OWNER_INFO,
    API_PATH_TREESUI_LIST,
    API_PATH_UUID_LEGACY,
    API_PATH_UUID_NAVHEADER,
    validate_all_endpoints,
)
from api.tree_update import (
    TreeOperationType,
    TreeUpdateRequest,
    TreeUpdateResponse,
    TreeUpdateResult,
    TreeUpdateService,
    apply_approved_facts_batch,
)

__all__ = [
    # API Constants
    "API_PATH_CSRF_TOKEN",
    "API_PATH_DISCOVERY_RELATIONSHIP",
    "API_PATH_EDIT_RELATIONSHIPS",
    "API_PATH_ETHNICITY_COMPARE",
    "API_PATH_ETHNICITY_OWNER",
    "API_PATH_ETHNICITY_REGION_NAMES",
    "API_PATH_FAMILY_TREE_PERSON",
    "API_PATH_FAMILY_TREE_VIEW",
    "API_PATH_HEADER_TREES",
    "API_PATH_MATCH_BADGES_IN_TREE",
    "API_PATH_MATCH_BADGE_DETAILS",
    "API_PATH_MATCH_DETAILS",
    "API_PATH_MATCH_LIST",
    "API_PATH_MATCH_PROBABILITY",
    "API_PATH_NEW_FAMILY_VIEW",
    "API_PATH_PERSON_FACTS_USER",
    "API_PATH_PERSON_GETLADDER",
    "API_PATH_PERSON_PICKER_SUGGEST",
    "API_PATH_PROFILE_DETAILS",
    "API_PATH_PROFILE_ID",
    "API_PATH_RELATION_LADDER_WITH_LABELS",
    "API_PATH_SEND_MESSAGE_EXISTING",
    "API_PATH_SEND_MESSAGE_NEW",
    "API_PATH_SHARED_MATCHES",
    "API_PATH_TREE_OWNER_INFO",
    "API_PATH_TREESUI_LIST",
    "API_PATH_UUID_LEGACY",
    "API_PATH_UUID_NAVHEADER",
    "validate_all_endpoints",
    # Tree Update exports
    "TreeOperationType",
    "TreeUpdateRequest",
    "TreeUpdateResponse",
    "TreeUpdateResult",
    "TreeUpdateService",
    "apply_approved_facts_batch",
]

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that every symbol in __all__ is importable and not None."
    import importlib

    pkg = importlib.import_module(__name__)
    missing: list[str] = []
    for name in __all__:
        try:
            val = getattr(pkg, name)
            if val is None:
                missing.append(f"{name} is None")
        except AttributeError:
            missing.append(f"{name} not found")
    if missing:
        print(f"  FAIL  {__name__}: {', '.join(missing)}")
        return False
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
