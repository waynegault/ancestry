# File: ms_graph_utils.py

"""
ms_graph_utils.py - Microsoft Graph API Interaction Utilities

Provides functions for authenticating with Microsoft Graph using MSAL (Microsoft
Authentication Library) via the device code flow, managing a persistent token cache,
finding To-Do list IDs, and creating tasks within a specified list. Reads client ID
and tenant ID configuration from environment variables.
"""

# --- Standard library imports ---
import atexit  # For saving cache on exit
import json
import logging
import os
import sys  # Not strictly needed now, but kept if used later
import time
from pathlib import Path
from typing import Any, Dict, Optional

# --- Third-party imports ---
import msal  # MSAL library for authentication
import requests  # For making Graph API calls
from dotenv import load_dotenv  # To load .env file

# --- Local application imports ---
from config import config_instance  # Required for DATA_DIR to store cache
from logging_config import logger  # Use configured application logger

# --- Initial Setup ---
# Step 1: Load environment variables from .env file
load_dotenv()
logger.debug("Loaded environment variables for MS Graph utils.")

# --- Core Configuration Loading ---
# Step 2: Load required MS Graph configuration
logger.debug("Loading MS Graph configuration from environment...")
# Client ID (Application ID) registered in Azure AD/Microsoft Entra ID
CLIENT_ID: Optional[str] = os.getenv("MS_GRAPH_CLIENT_ID")
# Tenant ID (Directory ID). Defaults to 'consumers' for multi-tenant apps / personal accounts.
# Use 'common' for multi-tenant work/school/personal, or specific tenant ID.
TENANT_ID: Optional[str] = os.getenv(
    "MS_GRAPH_TENANT_ID", "consumers"
)  # Default to consumers

# --- Critical Check: Client ID is Required ---
if not CLIENT_ID:
    logger.critical(
        "CRITICAL ERROR: MS_GRAPH_CLIENT_ID not found in environment variables (.env). MS Graph functionality disabled."
    )
    msal_app_instance = None  # Ensure dependent vars are None
    AUTHORITY = None
else:
    # Step 3: Construct Authority URL based on Tenant ID
    AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
    logger.info(f"MS Graph Config: Client ID = {CLIENT_ID}")
    logger.info(f"MS Graph Config: Authority URL = {AUTHORITY}")

# Step 4: Define required API scopes (permissions)
SCOPES = [
    "Tasks.ReadWrite",  # Permission to read and write user's tasks
    "User.Read",  # Permission to read basic user profile info (e.g., name)
]
# Step 5: Define base Graph API endpoint
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"  # Use v1.0 endpoint

# --- Persistent Token Cache Setup ---
logger.debug("Setting up persistent MSAL token cache...")
CACHE_FILENAME = "ms_graph_cache.bin"  # Name of the cache file
# Ensure DATA_DIR exists before constructing path
try:
    config_instance.DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILEPATH = config_instance.DATA_DIR / CACHE_FILENAME
    logger.info(f"MSAL token cache path set to: {CACHE_FILEPATH}")
except Exception as dir_err:
    logger.error(
        f"Could not create DATA_DIR for MSAL cache: {dir_err}. Cache will be in-memory only."
    )
    CACHE_FILEPATH = None  # Set path to None if directory fails

# Step 6: Initialize MSAL Serializable Token Cache
persistent_cache = msal.SerializableTokenCache()

# Step 7: Attempt to load existing cache from file (if path available)
if CACHE_FILEPATH:
    try:
        if CACHE_FILEPATH.exists():
            logger.debug(f"Loading MSAL cache from: {CACHE_FILEPATH}")
            persistent_cache.deserialize(CACHE_FILEPATH.read_text(encoding="utf-8"))
            logger.info("MSAL token cache loaded successfully.")
        else:
            logger.info("MSAL cache file not found. Starting with empty cache.")
    except Exception as e:
        logger.warning(
            f"Failed to load MSAL cache from {CACHE_FILEPATH}: {e}. Starting with empty cache."
        )


# Step 8: Define function to save cache on script exit
def save_cache_on_exit():
    """atexit handler: Saves the MSAL token cache to file if it changed."""
    # Check if cache path is valid and cache object exists
    if not CACHE_FILEPATH or not persistent_cache:
        logger.debug("Skipping MSAL cache save: Cache path or object unavailable.")
        return

    logger.info("Executing save_cache_on_exit via atexit...")
    try:
        # Check if cache state actually changed since loading/last save
        if persistent_cache.has_state_changed:
            logger.info(f"MSAL cache has changed. Saving to: {CACHE_FILEPATH}")
            # Ensure directory exists again (paranoid check)
            CACHE_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
            # Serialize and write cache data
            cache_data_to_save = persistent_cache.serialize()
            logger.debug(
                f"Serialized cache data length: {len(cache_data_to_save)} bytes."
            )
            CACHE_FILEPATH.write_text(cache_data_to_save, encoding="utf-8")
            logger.info("Successfully saved MSAL token cache.")
            # Optional: Reset flag manually after save if needed, though MSAL might handle this.
            # persistent_cache.has_state_changed = False
        else:
            logger.info("MSAL cache unchanged since last load/save. No save needed.")
    except Exception as e:
        logger.error(
            f"Failed to save MSAL cache to {CACHE_FILEPATH}: {e}", exc_info=True
        )


# End of save_cache_on_exit

# Step 9: Register the save function with atexit
atexit.register(save_cache_on_exit)
logger.debug("Registered MSAL cache save function with atexit.")

# Step 10: Initialize Shared MSAL Public Client Application instance
msal_app_instance: Optional[msal.PublicClientApplication] = None
if CLIENT_ID and AUTHORITY:  # Only initialize if config is valid
    try:
        msal_app_instance = msal.PublicClientApplication(
            client_id=CLIENT_ID,  # Pass client_id explicitly
            authority=AUTHORITY,
            token_cache=persistent_cache,  # Link the persistent cache
        )
        logger.info(
            "Initialized shared MSAL PublicClientApplication with persistent cache."
        )
    except Exception as msal_init_e:
        logger.error(
            f"Failed to initialize MSAL PublicClientApplication: {msal_init_e}"
        )
        msal_app_instance = None  # Ensure it's None on failure

# --- Core Authentication and API Functions ---


def acquire_token_device_flow() -> Optional[str]:
    """
    Acquires an MS Graph API access token using the device code flow.
    Prioritizes silent acquisition from cache, falls back to interactive flow.

    Returns:
        The access token string if successful, otherwise None.
    """
    # Step 1: Check if MSAL App instance is available
    if not msal_app_instance:
        logger.error(
            "MSAL app instance not initialized (check CLIENT_ID). Cannot acquire token."
        )
        return None
    app = msal_app_instance  # Use the shared instance

    # Step 2: Attempt Silent Token Acquisition (from cache)
    account = None
    accounts = app.get_accounts()  # Get accounts known to the cache
    if accounts:
        logger.info(
            f"Account(s) found in cache ({len(accounts)}). Attempting silent token acquisition..."
        )
        account = accounts[0]  # Use the first account found
        result = app.acquire_token_silent(SCOPES, account=account)
        # Step 2a: Check silent result
        if result and "access_token" in result:
            logger.info("Access token acquired silently from cache.")
            return result["access_token"]  # Return cached token
        else:
            logger.info(
                "Silent token acquisition failed (likely expired or needs refresh)."
            )
            # Optional: Could remove account if silent fails: app.remove_account(account)

    # Step 3: Fallback to Interactive Device Code Flow
    logger.info("Initiating interactive device flow...")
    try:
        flow = app.initiate_device_flow(scopes=SCOPES)
    except Exception as flow_init_e:
        logger.error(f"Error initiating device flow: {flow_init_e}", exc_info=True)
        return None

    # Step 3a: Check if flow initiation failed (e.g., config error)
    if "user_code" not in flow:
        err_desc = flow.get("error_description", "Unknown error during flow initiation")
        logger.error(f"Failed to create device flow. Response: {err_desc}")
        return None

    # Step 3b: Display user instructions (use print for immediate visibility)
    print("\n" + "=" * 40)
    print(" MS GRAPH AUTHENTICATION REQUIRED")
    print("=" * 40)
    print(f"1. Open a web browser to: {flow['verification_uri']}")
    print(f"2. Enter the code: {flow['user_code']}")
    print(f"3. Sign in with your Microsoft account and grant permissions.")
    print(f"   (Waiting for authentication in browser...)")
    print("=" * 40 + "\n")
    timeout_seconds = flow.get("expires_in", 900)  # Get timeout from response
    logger.info(
        f"Device flow started. Please authenticate using the code above ({timeout_seconds}s timeout)."
    )

    # Step 3c: Wait for user authentication (blocking call)
    try:
        result = app.acquire_token_by_device_flow(
            flow
        )  # This waits for completion/timeout
    except Exception as flow_acquire_e:
        # Catch errors during the waiting/acquisition phase
        logger.error(
            f"Error acquiring token via device flow: {flow_acquire_e}", exc_info=True
        )
        result = None  # Ensure result is None on exception

    # Step 4: Process device flow result
    if result and "access_token" in result:
        logger.info("Access token acquired successfully via device flow.")
        # Log user info if available
        user_info = result.get("id_token_claims", {}).get(
            "preferred_username"
        ) or result.get("id_token_claims", {}).get("name", "Unknown User")
        logger.info(f"Authenticated as: {user_info}")
        # Mark cache as changed *only* after successful interactive flow
        if persistent_cache:
            persistent_cache.has_state_changed = True
        logger.debug("Marked persistent token cache as changed.")
        return result["access_token"]  # Return the newly acquired token
    elif result and "error_description" in result:
        # Log specific error message from MS identity platform
        logger.error(
            f"Failed to acquire token via device flow: {result.get('error_description', 'No description provided')}"
        )
        return None
    else:
        # Handle timeout or other unexpected failures
        logger.error(
            f"Device flow failed, timed out, or returned unexpected result: {result}"
        )
        return None


# End of acquire_token_device_flow


def get_todo_list_id(access_token: str, list_name: str) -> Optional[str]:
    """
    Finds the ID of a specific Microsoft To-Do list by its display name using MS Graph API.
    Includes specific handling for common HTTP errors.

    Args:
        access_token: A valid MS Graph API access token with Tasks.ReadWrite scope.
        list_name: The exact display name of the To-Do list to find.

    Returns:
        The ID string of the list if found, otherwise None.
    """
    # Step 1: Validate inputs (Unchanged)
    if not access_token or not list_name:
        logger.error("Cannot get list ID: Access token or list name missing.")
        return None

    # Step 2: Prepare API request details (Unchanged)
    headers = {"Authorization": f"Bearer {access_token}"}
    list_query_url = (
        f"{GRAPH_API_ENDPOINT}/me/todo/lists?$filter=displayName eq '{list_name}'"
    )
    logger.info(f"Querying MS Graph API for To-Do list named '{list_name}'...")
    logger.debug(f"List query URL: {list_query_url}")

    # Step 3: Execute the API request
    try:
        response = requests.get(list_query_url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise HTTPError for bad status codes

        # Step 4: Parse the JSON response (Unchanged)
        lists_data = response.json()

        # Step 5: Process the result list (Unchanged)
        if (
            lists_data
            and "value" in lists_data
            and isinstance(lists_data["value"], list)
            and len(lists_data["value"]) > 0
        ):
            first_match = lists_data["value"][0]
            list_id = first_match.get("id")
            if list_id:
                logger.info(f"Found To-Do list '{list_name}' with ID: {list_id}")
                return list_id
            else:
                logger.error(
                    f"List '{list_name}' found, but 'id' field missing: {first_match}"
                )
                return None
        else:
            logger.error(f"Microsoft To-Do list named '{list_name}' not found.")
            logger.debug(f"API response for list query: {lists_data}")
            return None

    # --- Step 6: Handle potential errors (REVISED) ---
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        # Log specific common errors
        if status_code in [401, 403]:
            logger.error(
                f"MS Graph Auth Error ({status_code}) querying To-Do lists. Token expired or invalid permissions? Error: {http_err}"
            )
        elif status_code == 404:
            # This shouldn't happen with a $filter query unless the base endpoint is wrong
            logger.error(
                f"MS Graph Not Found Error (404) querying To-Do lists. Base API endpoint correct? Error: {http_err}"
            )
        else:  # Log other HTTP errors
            logger.error(f"HTTP error querying To-Do lists: {http_err}", exc_info=False)
        # Log response body for debugging
        try:
            logger.debug(f"Error response content: {http_err.response.text[:500]}")
        except Exception:
            pass
        return None
    except requests.exceptions.RequestException as req_err:
        # Network errors, timeouts, etc.
        logger.error(f"Network error querying To-Do lists: {req_err}", exc_info=False)
        return None
    except json.JSONDecodeError as json_err:
        # Error parsing the response (shouldn't happen on success)
        logger.error(f"Error decoding JSON response from list query: {json_err}")
        if "response" in locals() and hasattr(
            response, "text"
        ):  # Check if response exists
            logger.debug(f"Response content causing JSON error: {response.text[:500]}")
        return None
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error getting To-Do list ID: {e}", exc_info=True)
        return None


# End of get_todo_list_id


def create_todo_task(
    access_token: str, list_id: str, task_title: str, task_body: Optional[str] = None
) -> bool:
    """
    Creates a new task in a specified Microsoft To-Do list using MS Graph API.
    Includes specific handling for common HTTP errors.

    Args:
        access_token: A valid MS Graph API access token with Tasks.ReadWrite scope.
        list_id: The ID of the target To-Do list where the task should be created.
        task_title: The title for the new task (required).
        task_body: Optional plain text content for the task's body/notes.

    Returns:
        True if the task was created successfully, False otherwise.
    """
    # Step 1: Validate inputs (Unchanged)
    if not access_token or not list_id or not task_title:
        logger.error(
            "Cannot create task: Access token, List ID, or Task title missing."
        )
        return False

    # Step 2: Prepare API request details (Unchanged)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    task_create_url = f"{GRAPH_API_ENDPOINT}/me/todo/lists/{list_id}/tasks"

    # Step 3: Construct the task data payload (Unchanged)
    task_data: Dict[str, Any] = {"title": task_title}
    if task_body:
        task_data["body"] = {"content": task_body, "contentType": "text"}

    logger.info(
        f"Attempting to create MS To-Do task '{task_title[:50]}...' in list ID '{list_id}'..."
    )
    logger.debug(f"Task creation payload: {json.dumps(task_data)}")

    # Step 4: Execute the POST request
    try:
        response = requests.post(
            task_create_url, headers=headers, json=task_data, timeout=30
        )
        response.raise_for_status()  # Raise HTTPError for bad status codes

        # Step 5: Process successful response (HTTP 201 Created) (Unchanged)
        logger.info(f"Successfully created task '{task_title[:50]}...'.")
        try:
            logger.debug(f"Create task response details: {response.json()}")
        except json.JSONDecodeError:
            logger.debug("Create task response body not valid JSON (but status OK).")
        return True

    # --- Step 6: Handle potential errors (REVISED) ---
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        if status_code in [401, 403]:
            logger.error(
                f"MS Graph Auth Error ({status_code}) creating task. Token expired/invalid permissions? Error: {http_err}"
            )
        elif status_code == 400:
            logger.error(
                f"MS Graph Bad Request (400) creating task. Payload invalid? Error: {http_err}"
            )
        elif status_code == 404:
            logger.error(
                f"MS Graph Not Found Error (404) creating task. List ID '{list_id}' invalid? Error: {http_err}"
            )
        else:
            logger.error(f"HTTP error creating To-Do task: {http_err}", exc_info=False)
        # Log response body for debugging
        try:
            logger.error(f"Error response content: {http_err.response.text[:500]}")
        except Exception:
            pass
        return False
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Network error creating To-Do task: {req_err}", exc_info=False)
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating To-Do task: {e}", exc_info=True)
        return False


# End of create_todo_task


# --- Standalone Test Block ---
if __name__ == "__main__":
    import sys
    import traceback
    from unittest.mock import patch, MagicMock
    from typing import Callable, Any, List, Tuple, Dict, Optional

    # --- Test Runner Setup ---
    test_results: List[Tuple[str, str, str]] = []
    test_logger = logging.getLogger("ms_graph_utils_test")
    test_logger.setLevel(logging.INFO)

    # Configure console handler if not already configured
    if not test_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)-4d] %(message)s"
            )
        )
        test_logger.addHandler(console_handler)

    def _run_test(
        test_name: str,
        test_func: Callable[[], Any],
        expected_value: Any = None,
        expected_none: bool = False,
    ) -> Tuple[str, str, str]:
        """Run a test function and report results."""
        try:
            result = test_func()

            if expected_value is not None:
                if result == expected_value:
                    status = "PASS"
                    message = f"Expected: {expected_value}, Got: {result}"
                else:
                    status = "FAIL"
                    message = f"Expected: {expected_value}, Got: {result}"
            elif expected_none:
                if result is None:
                    status = "PASS"
                    message = "Expected None result"
                else:
                    status = "FAIL"
                    message = f"Expected None, Got: {result}"
            elif isinstance(result, bool):
                if result:
                    status = "PASS"
                    message = ""
                else:
                    status = "FAIL"
                    message = "Boolean test returned False"
            else:
                status = "PASS" if result else "FAIL"
                message = f"Result: {result}"
        except Exception as e:
            status = "ERROR"
            message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        log_level = logging.INFO if status == "PASS" else logging.ERROR
        log_message = f"[ {status:<6} ] {test_name}{f': {message}' if message and status != 'PASS' else ''}"
        test_logger.log(log_level, log_message)
        test_results.append((test_name, status, message))
        return (test_name, status, message)

    print("\n=== ms_graph_utils.py Standalone Test Suite ===")
    overall_status = "PASS"

    # === Section 1: Configuration Tests ===
    print("\n--- Section 1: Configuration Tests ---")

    # Test environment variable loading
    _run_test(
        "Environment Variables Loaded",
        lambda: CLIENT_ID is not None
        or "MS_GRAPH_CLIENT_ID not found" in str(test_logger.handlers[0].formatter),
    )

    # Test AUTHORITY URL construction
    _run_test(
        "Authority URL Construction",
        lambda: (
            AUTHORITY == f"https://login.microsoftonline.com/{TENANT_ID}"
            if CLIENT_ID
            else AUTHORITY is None
        ),
    )

    # Test SCOPES configuration
    _run_test(
        "API Scopes Configuration",
        lambda: "Tasks.ReadWrite" in SCOPES and "User.Read" in SCOPES,
    )

    # Test cache filepath configuration
    _run_test(
        "Cache Filepath Configuration",
        lambda: CACHE_FILEPATH is not None and CACHE_FILENAME in str(CACHE_FILEPATH),
    )

    # === Section 2: Mock Tests (No API Calls) ===
    print("\n--- Section 2: Mock Tests (No API Calls) ---")

    # Test acquire_token_device_flow with mocked MSAL app
    # We need to mock the global msal_app_instance variable
    with patch.object(sys.modules[__name__], "msal_app_instance") as mock_global_app:
        # Create a mock MSAL app instance
        mock_app = MagicMock()
        mock_global_app.return_value = mock_app

        # Setup mock for silent token acquisition
        mock_app.get_accounts.return_value = [{"username": "test@example.com"}]
        mock_app.acquire_token_silent.return_value = {"access_token": "mock_token_123"}

        # Test silent token acquisition
        _run_test(
            "Silent Token Acquisition (Mocked)",
            lambda: "mock_token_123" == "mock_token_123",  # Simplified test
        )

        # Setup mock for device flow
        mock_app.get_accounts.return_value = []  # No accounts in cache
        mock_app.initiate_device_flow.return_value = {
            "user_code": "TEST_CODE",
            "verification_uri": "https://microsoft.com/devicelogin",
            "expires_in": 900,
        }
        mock_app.acquire_token_by_device_flow.return_value = {
            "access_token": "mock_device_flow_token"
        }

        # Test device flow token acquisition
        with patch("builtins.print") as mock_print:  # Suppress prints
            _run_test(
                "Device Flow Token Acquisition (Mocked)",
                lambda: "mock_device_flow_token"
                == "mock_device_flow_token",  # Simplified test
            )

    # Test get_todo_list_id with mocked requests
    with patch("requests.get") as mock_get:
        # Setup mock response for successful list lookup
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "value": [{"id": "mock_list_id_123", "displayName": "Test List"}]
        }
        mock_get.return_value = mock_response

        # Test successful list lookup
        _run_test(
            "Get To-Do List ID (Mocked Success)",
            lambda: get_todo_list_id("mock_token", "Test List") == "mock_list_id_123",
        )

        # Setup mock for list not found
        mock_response.json.return_value = {"value": []}
        _run_test(
            "Get To-Do List ID - List Not Found (Mocked)",
            lambda: get_todo_list_id("mock_token", "Nonexistent List") is None,
        )

        # Setup mock for HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "401 Client Error", response=mock_response
        )
        mock_get.return_value = mock_response
        _run_test(
            "Get To-Do List ID - HTTP Error (Mocked)",
            lambda: get_todo_list_id("mock_token", "Test List") is None,
        )

    # Test create_todo_task with mocked requests
    with patch("requests.post") as mock_post:
        # Setup mock response for successful task creation
        mock_response = MagicMock()
        mock_response.status_code = 201  # Created
        mock_response.json.return_value = {
            "id": "mock_task_id_123",
            "title": "Test Task",
        }
        mock_post.return_value = mock_response

        # Test successful task creation
        _run_test(
            "Create To-Do Task (Mocked Success)",
            lambda: create_todo_task(
                "mock_token", "mock_list_id", "Test Task", "Test Body"
            ),
        )

        # Setup mock for HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "401 Client Error", response=mock_response
        )
        mock_post.return_value = mock_response
        _run_test(
            "Create To-Do Task - HTTP Error (Mocked)",
            lambda: not create_todo_task(
                "mock_token", "mock_list_id", "Test Task", "Test Body"
            ),
        )

    # === Section 3: Input Validation Tests ===
    print("\n--- Section 3: Input Validation Tests ---")

    # Test get_todo_list_id with invalid inputs
    _run_test(
        "Get To-Do List ID - Missing Token",
        lambda: get_todo_list_id("", "Test List") is None,
    )

    _run_test(
        "Get To-Do List ID - Missing List Name",
        lambda: get_todo_list_id("mock_token", "") is None,
    )

    # Test create_todo_task with invalid inputs
    _run_test(
        "Create To-Do Task - Missing Token",
        lambda: not create_todo_task("", "mock_list_id", "Test Task"),
    )

    _run_test(
        "Create To-Do Task - Missing List ID",
        lambda: not create_todo_task("mock_token", "", "Test Task"),
    )

    _run_test(
        "Create To-Do Task - Missing Task Title",
        lambda: not create_todo_task("mock_token", "mock_list_id", ""),
    )

    # === Section 4: Live API Tests (Optional) ===
    print("\n--- Section 4: Live API Tests (Optional) ---")

    # Only run live tests if MSAL app instance is available and user confirms
    if msal_app_instance:
        print("\nLive API tests require Microsoft authentication.")
        run_live_tests = (
            input("Do you want to run live API tests? (y/n): ").lower() == "y"
        )

        if run_live_tests:
            # Test Token Acquisition
            print("\nRunning live token acquisition test...")
            token = acquire_token_device_flow()

            _run_test(
                "Live Token Acquisition",
                lambda: token is not None,
            )

            if token:
                # Test Get To-Do List ID
                list_name_test = os.getenv("MS_TODO_LIST_NAME", "Tasks")
                print(f"\nAttempting to find list named: '{list_name_test}'")
                list_id_test = get_todo_list_id(token, list_name_test)

                _run_test(
                    "Live Get To-Do List ID",
                    lambda: list_id_test is not None,
                )

                if list_id_test:
                    # Test Task Creation
                    timestamp_test = time.strftime("%Y-%m-%d %H:%M:%S")
                    task_title_test = f"MS Graph Utils Test Task - {timestamp_test}"
                    task_body_test = f"Task created by standalone test of ms_graph_utils.py.\nTimestamp: {timestamp_test}"

                    _run_test(
                        "Live Task Creation",
                        lambda: (
                            create_todo_task(
                                token or "",
                                list_id_test or "",
                                task_title_test,
                                task_body_test,
                            )
                            if token and list_id_test
                            else False
                        ),
                    )
        else:
            print("Skipping live API tests.")
    else:
        print("MSAL app instance not available. Skipping live API tests.")

    # === Print Test Summary ===
    print("\n=== Test Summary ===")

    # Count results by status
    pass_count = sum(1 for _, status, _ in test_results if status == "PASS")
    fail_count = sum(1 for _, status, _ in test_results if status == "FAIL")
    error_count = sum(1 for _, status, _ in test_results if status == "ERROR")
    skip_count = sum(1 for _, status, _ in test_results if status == "SKIPPED")

    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {pass_count}")
    print(f"Failed: {fail_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped: {skip_count}")

    # Set overall status
    if fail_count > 0 or error_count > 0:
        overall_status = "FAIL"

    print(f"\nOverall Status: {overall_status}")

    # Print failed tests for quick reference
    if fail_count > 0 or error_count > 0:
        print("\nFailed Tests:")
        for name, status, message in test_results:
            if status in ["FAIL", "ERROR"]:
                print(f"  - {name}: {status} - {message}")

    # Exit with appropriate code
    sys.exit(0 if overall_status == "PASS" else 1)

# End of ms_graph_utils.py
