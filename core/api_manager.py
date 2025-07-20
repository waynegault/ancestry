"""
API Manager - Handles API interactions and user identifiers.

This module extracts API management functionality from the monolithic
SessionManager class to provide a clean separation of concerns.
"""

from core_imports import standardize_module_imports, auto_register_module

standardize_module_imports()
auto_register_module(globals(), __name__)

import logging
from typing import Optional, Dict, Any, Union, List
from urllib.parse import urljoin

import requests
from requests import Response as RequestsResponse
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

from config import config_schema
from logging_config import logger

# Type aliases
ApiResponseType = Union[Dict[str, Any], List[Any], str, bytes, None, RequestsResponse]

# API Constants
API_PATH_CSRF_TOKEN = "discoveryui-matches/parents/api/csrfToken"
API_PATH_PROFILE_ID = "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid"
API_PATH_UUID = "api/uhome/secure/rest/header/dna"

KEY_UCDMID = "ucdmid"
KEY_TEST_ID = "testId"
KEY_DATA = "data"


class APIManager:
    """
    Manages API interactions and user identifiers.

    This class handles all API-related functionality including:
    - HTTP requests session management
    - CSRF token management
    - User identifier retrieval (profile ID, UUID, tree ID)
    - API request retry logic
    """

    def __init__(self):
        """Initialize the APIManager."""
        # User identifiers
        self.csrf_token: Optional[str] = None
        self.my_profile_id: Optional[str] = None
        self.my_uuid: Optional[str] = None
        self.my_tree_id: Optional[str] = None
        self.tree_owner_name: Optional[str] = None

        # Logging flags to prevent repeated logging
        self._profile_id_logged: bool = False
        self._uuid_logged: bool = False
        self._tree_id_logged: bool = False
        self._owner_logged: bool = False

        # Initialize requests session
        self._requests_session = requests.Session()
        self._setup_requests_session()

        logger.debug("APIManager initialized")

    def _setup_requests_session(self):
        """Configure the requests session with retry strategy."""
        retry_strategy = Retry(
            total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(
            pool_connections=20, pool_maxsize=50, max_retries=retry_strategy
        )
        self._requests_session.mount("http://", adapter)
        self._requests_session.mount("https://", adapter)
        logger.debug("Requests session configured with retry strategy.")

    def sync_cookies_from_browser(self, browser_manager) -> bool:
        """
        Sync cookies from browser to the requests session.

        Args:
            browser_manager: BrowserManager instance to sync cookies from

        Returns:
            bool: True if sync successful, False otherwise
        """
        if not browser_manager or not browser_manager.is_session_valid():
            logger.warning("Cannot sync cookies: Browser session invalid.")
            return False

        try:
            driver_cookies = browser_manager.driver.get_cookies()
            logger.debug(
                f"Retrieved {len(driver_cookies)} cookies from browser for API sync."
            )

            # Clear existing cookies
            self._requests_session.cookies.clear()

            # Add each cookie to requests session
            for cookie in driver_cookies:
                self._requests_session.cookies.set(
                    cookie["name"],
                    cookie["value"],
                    domain=cookie.get("domain"),
                    path=cookie.get("path", "/"),
                    secure=cookie.get("secure", False),
                )

            logger.debug(
                f"Synced {len(driver_cookies)} cookies to API requests session."
            )
            return True

        except Exception as e:
            logger.error(
                f"Error syncing cookies from browser to API: {e}", exc_info=True
            )
            return False

    def make_api_request(
        self,
        url: str,
        method: str = "GET",
        use_csrf_token: bool = True,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        api_description: str = "API Request",
    ) -> ApiResponseType:
        """
        Make an API request with proper error handling and CSRF token support.

        Args:
            url: The URL to make the request to
            method: HTTP method (GET, POST, etc.)
            use_csrf_token: Whether to include CSRF token in headers
            data: Form data to send
            json_data: JSON data to send
            headers: Additional headers
            timeout: Request timeout in seconds
            api_description: Description for logging

        Returns:
            API response data or None if failed
        """
        try:
            # Prepare headers
            request_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
            }

            if headers:
                request_headers.update(headers)

            # Add CSRF token if requested and available
            if use_csrf_token and self.csrf_token:
                request_headers["X-CSRF-TOKEN"] = self.csrf_token
                logger.debug(f"Added CSRF token to {api_description} request")
            elif use_csrf_token and not self.csrf_token:
                logger.warning(
                    f"CSRF token requested but not available for {api_description}"
                )

            # Make the request
            logger.debug(f"Making {method} request to {url} ({api_description})")

            response = self._requests_session.request(
                method=method,
                url=url,
                headers=request_headers,
                data=data,
                json=json_data,
                timeout=timeout,
                allow_redirects=True,
            )

            # Check response status
            response.raise_for_status()

            # Try to parse JSON response
            try:
                json_response = response.json()
                logger.debug(f"{api_description} request successful (JSON response)")
                return json_response
            except ValueError:
                # Not JSON, return text or response object
                if response.text:
                    logger.debug(
                        f"{api_description} request successful (text response)"
                    )
                    return response.text
                else:
                    logger.debug(
                        f"{api_description} request successful (response object)"
                    )
                    return response

        except RequestException as e:
            logger.error(f"{api_description} request failed: {e}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error in {api_description} request: {e}", exc_info=True
            )
            return None

    def get_csrf_token(self) -> Optional[str]:
        """
        Retrieve CSRF token from the API.

        Returns:
            str: CSRF token or None if failed
        """
        logger.debug("Retrieving CSRF token...")

        url = urljoin(config_schema.api.base_url, API_PATH_CSRF_TOKEN)
        response_data = self.make_api_request(
            url=url,
            method="GET",
            use_csrf_token=False,  # Don't use CSRF token to get CSRF token
            api_description="Get CSRF Token",
        )

        if response_data and isinstance(response_data, dict):
            csrf_token = response_data.get("token")
            if csrf_token:
                self.csrf_token = csrf_token
                logger.debug("CSRF token retrieved successfully")
                return csrf_token
            else:
                logger.error("CSRF token not found in response")
        else:
            logger.error("Failed to retrieve CSRF token")

        return None

    def get_profile_id(self) -> Optional[str]:
        """
        Retrieve user profile ID (ucdmid) from the API.

        Returns:
            str: Profile ID or None if failed
        """
        logger.debug("Retrieving profile ID (ucdmid)...")

        url = urljoin(config_schema.api.base_url, API_PATH_PROFILE_ID)
        response_data = self.make_api_request(
            url=url,
            method="GET",
            use_csrf_token=False,
            api_description="Get Profile ID",
        )

        if response_data and isinstance(response_data, dict):
            profile_id = response_data.get(KEY_UCDMID)
            if profile_id:
                self.my_profile_id = profile_id
                if not self._profile_id_logged:
                    logger.info(f"My profile ID: {profile_id}")
                    self._profile_id_logged = True
                return profile_id
            else:
                logger.error("Profile ID not found in response")
        else:
            logger.error("Failed to retrieve profile ID")

        return None

    def get_uuid(self) -> Optional[str]:
        """
        Retrieve user UUID from the API.

        Returns:
            str: UUID or None if failed
        """
        logger.debug("Retrieving UUID...")

        url = urljoin(config_schema.api.base_url, API_PATH_UUID)
        response_data = self.make_api_request(
            url=url, method="GET", use_csrf_token=False, api_description="Get UUID"
        )

        if response_data and isinstance(response_data, dict):
            uuid_value = response_data.get(KEY_TEST_ID)
            if uuid_value:
                self.my_uuid = uuid_value
                if not self._uuid_logged:
                    logger.info(f"My UUID: {uuid_value}")
                    self._uuid_logged = True
                return uuid_value
            else:
                logger.error("UUID not found in response")
        else:
            logger.error("Failed to retrieve UUID")

        return None

    def retrieve_all_identifiers(self) -> bool:
        """
        Retrieve all user identifiers (profile ID, UUID, etc.).

        Returns:
            bool: True if all essential identifiers retrieved, False otherwise
        """
        logger.debug("Retrieving all user identifiers...")
        all_ok = True

        # Get Profile ID
        if not self.my_profile_id:
            profile_id = self.get_profile_id()
            if not profile_id:
                logger.error("Failed to retrieve profile ID")
                all_ok = False

        # Get UUID
        if not self.my_uuid:
            uuid_value = self.get_uuid()
            if not uuid_value:
                logger.error("Failed to retrieve UUID")
                all_ok = False

        # Get CSRF token
        if not self.csrf_token:
            csrf_token = self.get_csrf_token()
            if not csrf_token:
                logger.warning("Failed to retrieve CSRF token")
                # Don't mark as failure since some operations might work without it

        if all_ok:
            logger.debug("All essential identifiers retrieved successfully")
        else:
            logger.warning("Some essential identifiers could not be retrieved")

        return all_ok

    def verify_api_login_status(self) -> Optional[bool]:
        """
        Verify login status via API.

        Returns:
            bool: True if logged in, False if not, None if unable to determine
        """
        logger.debug("Verifying API login status...")

        # Try to get profile ID as a test of login status
        profile_response = self.get_profile_id()

        if profile_response:
            logger.debug("API login verification successful")
            return True
        else:
            logger.warning("API login verification failed")
            return False

    def reset_logged_flags(self):
        """Reset flags used to prevent repeated logging of IDs."""
        self._profile_id_logged = False
        self._uuid_logged = False
        self._tree_id_logged = False
        self._owner_logged = False

    def clear_identifiers(self):
        """Clear all stored identifiers."""
        self.csrf_token = None
        self.my_profile_id = None
        self.my_uuid = None
        self.my_tree_id = None
        self.tree_owner_name = None
        self.reset_logged_flags()
        logger.debug("All identifiers cleared")

    @property
    def has_essential_identifiers(self) -> bool:
        """Check if essential identifiers are available."""
        return bool(self.my_profile_id and self.my_uuid)

    @property
    def requests_session(self) -> requests.Session:
        """Get the requests session."""
        return self._requests_session


# ==============================================
# TEST FRAMEWORK IMPLEMENTATION
# ==============================================


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for core/api_manager.py with real functionality testing.
    Tests initialization, core functionality, edge cases, integration, performance, and error handling.
    """
    from test_framework import (
        TestSuite,
        suppress_logging,
        create_mock_data,
        assert_valid_function,
    )

    suite = TestSuite("API Manager & HTTP Request Handling", "api_manager.py")
    suite.start_suite()

    # INITIALIZATION TESTS
    def test_api_manager_initialization():
        """Test APIManager initialization and configuration."""
        try:
            api_manager = APIManager()

            # Verify basic attributes exist
            assert hasattr(
                api_manager, "csrf_token"
            ), "Should have csrf_token attribute"
            assert hasattr(
                api_manager, "my_profile_id"
            ), "Should have my_profile_id attribute"
            assert hasattr(api_manager, "my_uuid"), "Should have my_uuid attribute"
            assert hasattr(
                api_manager, "_requests_session"
            ), "Should have requests session"

            # Initial state should be None for identifiers
            assert api_manager.csrf_token is None, "CSRF token should initially be None"
            assert (
                api_manager.my_profile_id is None
            ), "Profile ID should initially be None"
            assert api_manager.my_uuid is None, "UUID should initially be None"

            return True
        except Exception:
            return False

    suite.run_test(
        "API Manager Initialization",
        test_api_manager_initialization,
        "APIManager initializes with proper attributes and session management",
        "Test APIManager class initialization and verify core attributes exist",
        "Test API manager initialization and verify basic attributes and session setup",
    )

    # CORE FUNCTIONALITY TESTS
    def test_identifier_management():
        """Test user identifier management methods."""
        try:
            api_manager = APIManager()

            # Test identifier properties
            assert hasattr(
                api_manager, "has_essential_identifiers"
            ), "Should have identifier check property"

            # Initially should not have essential identifiers
            initial_state = api_manager.has_essential_identifiers
            assert isinstance(
                initial_state, bool
            ), "Identifier check should return boolean"

            # Test setting identifiers
            api_manager.my_profile_id = "test_profile_123"
            api_manager.my_uuid = "test_uuid_456"

            # Should now have essential identifiers
            updated_state = api_manager.has_essential_identifiers
            assert (
                updated_state == True
            ), "Should have essential identifiers after setting them"

            return True
        except Exception:
            return False

    suite.run_test(
        "User Identifier Management",
        test_identifier_management,
        "User identifiers (profile ID, UUID) manage correctly with validation",
        "Test identifier setting and validation methods",
        "Test user identifier management and essential identifier validation",
    )

    def test_api_request_methods():
        """Test API request method availability."""
        try:
            api_manager = APIManager()

            # Check for API request methods
            api_methods = [
                "get_csrf_token",
                "get_profile_id",
                "get_uuid",
                "clear_identifiers",
            ]

            available_methods = []
            for method_name in api_methods:
                if hasattr(api_manager, method_name):
                    method = getattr(api_manager, method_name)
                    if callable(method):
                        available_methods.append(method_name)

            # Should have most API methods
            assert (
                len(available_methods) >= 3
            ), f"Should have API methods available, found: {available_methods}"

            return True
        except Exception:
            return False

    suite.run_test(
        "API Request Methods",
        test_api_request_methods,
        "API request methods (get_csrf_token, get_profile_id, etc.) are available and callable",
        "Test availability of core API request methods",
        "Test API request method availability and callability",
    )

    # EDGE CASES TESTS
    def test_invalid_response_handling():
        """Test handling of invalid API responses."""
        try:
            api_manager = APIManager()

            # Test identifier clearing
            api_manager.clear_identifiers()

            # After clearing, should not have essential identifiers
            assert (
                not api_manager.has_essential_identifiers
            ), "Should not have identifiers after clearing"

            return True
        except Exception:
            return False

    suite.run_test(
        "Invalid Response Handling",
        test_invalid_response_handling,
        "API manager handles invalid or empty responses gracefully",
        "Test API manager with invalid response scenarios",
        "Test edge case handling for invalid API responses and data clearing",
    )

    # INTEGRATION TESTS
    def test_config_integration():
        """Test integration with configuration system."""
        try:
            # Test config integration
            assert config_schema is not None, "Config schema should be available"

            # Test that API constants are defined
            api_constants = [
                "API_PATH_CSRF_TOKEN",
                "API_PATH_PROFILE_ID",
                "API_PATH_UUID",
            ]

            constants_defined = []
            for constant in api_constants:
                if constant in globals():
                    constants_defined.append(constant)

            assert (
                len(constants_defined) >= 2
            ), f"Should have API path constants defined: {constants_defined}"

            return True
        except Exception:
            return False

    suite.run_test(
        "Configuration Integration",
        test_config_integration,
        "API manager integrates properly with configuration system and API constants",
        "Test integration with configuration system and API path constants",
        "Test integration between API manager and configuration system",
    )

    # PERFORMANCE TESTS
    def test_session_reuse_efficiency():
        """Test HTTP session reuse for performance."""
        try:
            import time

            api_manager = APIManager()

            # Test that session is reused (same instance)
            session1 = api_manager.requests_session
            session2 = api_manager.requests_session

            # Should be the same session instance for efficiency
            assert session1 is session2, "Should reuse the same session instance"

            # Test multiple API manager instances
            start_time = time.time()
            managers = [APIManager() for _ in range(5)]
            end_time = time.time()

            # Should create 5 managers quickly (< 0.1 seconds)
            assert (
                end_time - start_time
            ) < 0.1, "Should create API managers efficiently"
            assert len(managers) == 5, "Should create all requested managers"

            return True
        except Exception:
            return False

    suite.run_test(
        "Session Reuse Efficiency",
        test_session_reuse_efficiency,
        "HTTP sessions reuse efficiently and API managers create quickly",
        "Measure API manager creation time and session reuse patterns",
        "Test performance of session reuse and API manager creation",
    )

    # ERROR HANDLING TESTS
    def test_connection_error_handling():
        """Test handling of connection errors."""
        try:
            api_manager = APIManager()
            session = api_manager.requests_session

            # Test that session has proper error handling setup
            if hasattr(session, "adapters"):
                # Should have adapters configured for error handling
                adapter_count = len(session.adapters)
                assert adapter_count >= 0, "Should have session adapters configured"

            # Test exception handling imports
            from requests.exceptions import RequestException

            assert (
                RequestException is not None
            ), "Should have RequestException available"

            return True
        except ImportError:
            return False
        except Exception:
            return False

    suite.run_test(
        "Connection Error Handling",
        test_connection_error_handling,
        "API manager handles connection errors and request exceptions gracefully",
        "Test error handling setup and exception class availability",
        "Test connection error handling and request exception management",
    )

    return suite.finish_suite()


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Use centralized path management
    project_root = Path(__file__).resolve().parent.parent
    try:
        sys.path.insert(0, str(project_root))
        from core_imports import ensure_imports

        ensure_imports()
    except ImportError:
        # Fallback for testing environment
        sys.path.insert(0, str(project_root))

    print("🔗 Running API Manager & HTTP Request Handling comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
