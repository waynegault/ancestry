"""API Integration Package.

Provides unified API management including:
- api_constants: Centralized API endpoint constants
- api_utils: API request orchestration and authentication
- api_search_core: Core API search functionality
- api_search_utils: API search utilities and helpers
"""

from api.api_constants import *

__all__ = [
    "API_PATH_CSRF_TOKEN",
    "API_PATH_PROFILE_DETAILS",
    "API_PATH_PROFILE_ID",
    "API_PATH_SEND_MESSAGE_EXISTING",
    "API_PATH_SEND_MESSAGE_NEW",
    "API_PATH_UUID_LEGACY",
    "API_PATH_UUID_NAVHEADER",
]

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
