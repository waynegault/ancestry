# File: ms_graph_utils.py
# V1.3: Reorganized structure, enhanced comments, early CLIENT_ID check.

import os
import json
import logging
import time
import sys
from typing import Optional, Dict, Any
import msal
import requests
from dotenv import load_dotenv
import atexit
from config import config_instance  # Required for DATA_DIR

# --- Initial Setup ---
load_dotenv()
logger = logging.getLogger(__name__)  # Use module-specific logger for utils

# --- Core Configuration Loading ---
logger.debug("Loading MS Graph configuration from environment...")
# Load Client ID (Required) - Standardized to MS_GRAPH_CLIENT_ID
CLIENT_ID: Optional[str] = os.getenv("MS_GRAPH_CLIENT_ID")
# Load Tenant ID (Optional - defaults to 'consumers' for multi-tenant/personal accounts)
TENANT_ID: Optional[str] = os.getenv("MS_GRAPH_TENANT_ID")

# --- Early Exit if Client ID is Missing ---
if not CLIENT_ID:
    logger.critical(
        "CRITICAL ERROR: MS_GRAPH_CLIENT_ID not found in environment variables (.env). MS Graph functionality disabled."
    )
    # Set dependent variables to None to prevent errors later if script continues
    msal_app_instance = None
    AUTHORITY = None
else:
    # --- Derive Dependent Configuration ---
    # Construct Authority URL based on Tenant ID or default to 'consumers'
    AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID or 'consumers'}"
    logger.info(f"MS Graph Using Client ID: {CLIENT_ID}")
    logger.info(f"MS Graph Using Authority URL: {AUTHORITY}")

# Define required API scopes and the Graph API endpoint
SCOPES = [
    "Tasks.ReadWrite",
    "User.Read",
]  # Permissions needed for To-Do tasks and reading user profile
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

# --- Persistent Token Cache Setup ---
logger.debug("Setting up persistent MSAL token cache...")
CACHE_FILENAME = "ms_graph_cache.bin"  # File to store cached tokens
CACHE_FILEPATH = config_instance.DATA_DIR / CACHE_FILENAME  # Full path within DATA_DIR

# Initialize the serializable cache object
persistent_cache = msal.SerializableTokenCache()

# Attempt to load existing cache from file on script startup
try:
    if CACHE_FILEPATH.exists():
        logger.debug(f"Loading MSAL cache from: {CACHE_FILEPATH}")
        persistent_cache.deserialize(CACHE_FILEPATH.read_text(encoding="utf-8"))
        logger.info("MSAL token cache loaded successfully.")
    else:
        logger.debug("MSAL cache file not found. Starting with empty cache.")
except Exception as e:
    logger.warning(
        f"Failed to load MSAL cache from {CACHE_FILEPATH}: {e}. Starting with empty cache."
    )


def save_cache_on_exit():
    """atexit handler: Saves the MSAL token cache to a file if it has changed."""
    logger.info("Executing save_cache_on_exit via atexit...")
    try:
        state_changed = persistent_cache.has_state_changed
        logger.debug(f"Cache 'has_state_changed' flag is: {state_changed}")
        if state_changed:
            logger.info(f"MSAL cache has changed. Attempting save to: {CACHE_FILEPATH}")
            # Ensure the Data directory exists
            CACHE_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
            cache_data_to_save = persistent_cache.serialize()
            logger.debug(
                f"Serialized cache data length: {len(cache_data_to_save)} bytes."
            )
            if len(cache_data_to_save) < 10:
                logger.warning(
                    "Serialized cache data seems very small. Potential issue?"
                )

            CACHE_FILEPATH.write_text(cache_data_to_save, encoding="utf-8")
            logger.info(f"Successfully wrote cache data to {CACHE_FILEPATH}.")
            # Reset flag after successful save prevents re-saving if script exits abnormally later
            persistent_cache.has_state_changed = False
        else:
            logger.info("MSAL cache unchanged. No save needed.")
    except Exception as e:
        # Log error but allow script exit to continue
        logger.error(
            f"Failed to save MSAL cache to {CACHE_FILEPATH}: {e}", exc_info=True
        )

# End save_cache_on_exit

# Register the save function to run when the script exits cleanly
atexit.register(save_cache_on_exit)
logger.debug("Registered MSAL cache save function with atexit.")

# --- Shared MSAL App Instance Initialization ---
# This instance will be used by acquire_token_device_flow
msal_app_instance: Optional[msal.PublicClientApplication] = None
if CLIENT_ID and AUTHORITY:  # Check if config loaded okay
    try:
        msal_app_instance = msal.PublicClientApplication(
            CLIENT_ID,
            authority=AUTHORITY,
            token_cache=persistent_cache,  # Link the persistent cache
        )
        logger.debug(
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
    It prioritizes silent acquisition using the persistent cache, falling back
    to the interactive device flow if needed.

    Returns:
        The access token string if successful, otherwise None.
    """
    # Step 1: Check if MSAL App was initialized successfully
    if not msal_app_instance:
        logger.error(
            "MSAL app not initialized (check MS_GRAPH_CLIENT_ID). Cannot acquire token."
        )
        return None

    app = msal_app_instance  # Use the shared instance

    # Step 2: Attempt Silent Token Acquisition from Cache
    account = None
    accounts = app.get_accounts()  # Get accounts known to the cache
    if accounts:
        logger.info(
            f"Account(s) found in cache ({len(accounts)}). Attempting silent acquisition for the first account."
        )
        account = accounts[0]  # Use the first cached account (common case)
        result = app.acquire_token_silent(SCOPES, account=account)
        # Check if silent acquisition was successful
        if result and "access_token" in result:
            logger.info("Access token acquired silently from cache.")
            return result["access_token"]
        else:
            logger.info(
                "Silent token acquisition failed (token expired or needs refresh)."
            )
            # Optional: Could attempt to remove the account if silent fails consistently
            # app.remove_account(account)
            # logger.info("Removed potentially stale account from cache.")

    # Step 3: Fallback to Interactive Device Code Flow
    logger.info("Attempting interactive device flow...")
    flow = app.initiate_device_flow(scopes=SCOPES)

    # Check if flow initiation failed
    if "user_code" not in flow:
        err_desc = flow.get("error_description", "Unknown error")
        logger.error(f"Failed to create device flow. Response: {err_desc}")
        return None

    # Step 3a: Display instructions for the user
    # Using print for direct user visibility during interactive phase
    print(
        f"\n--- MS GRAPH AUTH REQUIRED ---\n"
        f"To sign in, use a web browser to open the page:\n{flow['verification_uri']}\n"
        f"and enter the code: {flow['user_code']}\n"
        f"Waiting for you to authenticate in the browser...\n"
        f"------------------------------"
    )
    timeout_seconds = flow.get("expires_in", 900)  # Default 15 minutes
    logger.info(
        f"Device flow initiated. Waiting for user authentication ({timeout_seconds}s timeout)..."
    )

    # Step 3b: Wait for user to authenticate in browser
    # This call blocks until the user completes the flow, it times out, or an error occurs.
    result = app.acquire_token_by_device_flow(flow)

    # Step 4: Process the result of the device flow attempt
    if result and "access_token" in result:
        logger.info("Access token acquired successfully via device flow.")
        # Log authenticated user info if available in the token claims
        if "id_token_claims" in result:
            user_info = result["id_token_claims"].get("preferred_username") or result[
                "id_token_claims"
            ].get("name")
            if user_info:
                logger.info(f"Authenticated as: {user_info}")
        # Mark the cache as changed so it gets saved on exit
        persistent_cache.has_state_changed = True
        logger.debug("Marked persistent cache as changed.")
        return result["access_token"]
    elif result and "error_description" in result:
        # Log specific error from Microsoft identity platform
        logger.error(
            f"Failed to acquire token via device flow: {result.get('error_description')}"
        )
        return None
    else:
        # Handle timeout or other unexpected result structures
        logger.error(
            f"Device flow failed, timed out, or returned unexpected result: {result}"
        )
        return None
# end acquire_token_device_flow


def get_todo_list_id(access_token: str, list_name: str) -> Optional[str]:
    """
    Finds the ID of a specific Microsoft To-Do list by its display name.

    Args:
        access_token: A valid MS Graph API access token.
        list_name: The exact display name of the To-Do list to find.

    Returns:
        The ID string of the list if found, otherwise None.
    """
    # Step 1: Validate input
    if not access_token:
        logger.error("Cannot get list ID: Access token is missing.")
        return None
    if not list_name:
        logger.error("Cannot get list ID: Target list name is missing.")
        return None

    # Step 2: Prepare the request
    headers = {"Authorization": f"Bearer {access_token}"}
    # Use OData filter to find the list by name efficiently
    list_query_url = (
        f"{GRAPH_API_ENDPOINT}/me/todo/lists?$filter=displayName eq '{list_name}'"
    )
    logger.info(f"Querying Graph API for To-Do list named '{list_name}'...")
    logger.debug(f"List query URL: {list_query_url}")

    # Step 3: Execute the API request and handle potential errors
    try:
        response = requests.get(list_query_url, headers=headers, timeout=30)
        # Check for HTTP errors (4xx, 5xx)
        response.raise_for_status()

        # Step 4: Parse the JSON response
        lists_data = response.json()

        # Step 5: Process the results
        # The response contains a 'value' list of matching lists
        if lists_data and "value" in lists_data and len(lists_data["value"]) > 0:
            # Assuming list names are unique enough for the purpose, take the first match
            first_match = lists_data["value"][0]
            list_id = first_match.get("id")
            if list_id:
                logger.info(f"Found list '{list_name}' with ID: {list_id}")
                return list_id
            else:
                # This shouldn't happen if the API response is valid
                logger.error(
                    f"List '{list_name}' found, but 'id' field is missing in the response item: {first_match}"
                )
                return None
        else:
            # List not found or empty response
            logger.error(
                f"Microsoft To-Do list named '{list_name}' not found for the authenticated user."
            )
            logger.debug(f"Response from list query: {lists_data}")
            return None

    # Handle specific exceptions during the process
    except requests.exceptions.RequestException as e:
        logger.error(f"Network or HTTP error querying To-Do lists: {e}", exc_info=True)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from list query: {e}")
        # Log the raw response text if decoding fails
        logger.debug(f"Response content that failed decoding: {response.text}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error getting list ID: {e}", exc_info=True)
        return None
# end get_todo_list_id


def create_todo_task(
    access_token: str, list_id: str, task_title: str, task_body: Optional[str] = None
) -> bool:
    """
    Creates a new task in a specified Microsoft To-Do list.

    Args:
        access_token: A valid MS Graph API access token.
        list_id: The ID of the target To-Do list.
        task_title: The title for the new task.
        task_body: Optional plain text content for the task body.

    Returns:
        True if the task was created successfully, False otherwise.
    """
    # Step 1: Validate inputs
    if not access_token:
        logger.error("Cannot create task: Access token is missing.")
        return False
    if not list_id:
        logger.error("Cannot create task: Target List ID is missing.")
        return False
    if not task_title:
        logger.error("Cannot create task: Task title is missing.")
        return False

    # Step 2: Prepare the request
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",  # Specify content type for POST
    }
    # Construct the URL for creating tasks within the specific list
    task_create_url = f"{GRAPH_API_ENDPOINT}/me/todo/lists/{list_id}/tasks"

    # Step 3: Construct the task data payload (JSON body)
    task_data: Dict[str, Any] = {"title": task_title}
    # Add body content if provided
    if task_body:
        task_data["body"] = {
            "content": task_body,
            "contentType": "text",  # Assuming plain text body content
        }

    logger.info(f"Attempting to create task '{task_title}' in list ID '{list_id}'...")
    logger.debug(f"Task creation payload: {json.dumps(task_data)}")

    # Step 4: Execute the API request and handle potential errors
    try:
        response = requests.post(
            task_create_url, headers=headers, json=task_data, timeout=30
        )
        # Check for HTTP errors (e.g., 400 Bad Request, 401 Unauthorized, 404 List Not Found)
        response.raise_for_status()

        # Step 5: Process successful response
        logger.info(f"Successfully created task '{task_title}'.")
        # Log the created task details (optional, contains ID, etc.)
        try:
            logger.debug(f"Create task response details: {response.json()}")
        except json.JSONDecodeError:
            logger.debug(
                "Create task response body was not valid JSON (but status was OK)."
            )
        return True  # Indicate success

    # Handle specific exceptions
    except requests.exceptions.RequestException as e:
        logger.error(f"Network or HTTP error creating To-Do task: {e}", exc_info=True)
        # Log the response body from the error if available
        if hasattr(e, "response") and e.response is not None:
            try:
                logger.error(f"Error response content: {e.response.text}")
            except Exception:
                pass  # Ignore errors logging the error response itself
        return False  # Indicate failure
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error creating task: {e}", exc_info=True)
        return False  # Indicate failure
# end create_todo_task


# --- Standalone Test Block ---
if __name__ == "__main__":
    # Setup basic logging for testing this module directly
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)-4d] %(message)s",
    )
    logger.info("--- Testing ms_graph_utils.py with Persistent Cache ---")

    # Check if MSAL app instance was successfully created earlier
    if not msal_app_instance:
        logger.critical(
            "MSAL App could not be initialized. Check MS_GRAPH_CLIENT_ID in environment variables."
        )
    else:
        # Test 1: Acquire Token (will use cache or prompt for device flow)
        logger.info("--- Test 1: Acquiring Access Token ---")
        token = acquire_token_device_flow()

        if token:
            logger.info("Test 1 PASSED: Authentication successful.")

            # Test 2: Get To-Do List ID (only if token acquired)
            logger.info("--- Test 2: Getting To-Do List ID ---")
            list_name_test = os.getenv(
                "MS_TODO_LIST_NAME", "Tasks"
            )  # Get target list name
            list_id_test = get_todo_list_id(token, list_name_test)

            if list_id_test:
                logger.info(
                    f"Test 2 PASSED: Successfully retrieved list ID for '{list_name_test}'."
                )

                # Test 3: Create a Task (only if list ID acquired)
                logger.info("--- Test 3: Creating a Test Task ---")
                timestamp_test = time.strftime("%Y-%m-%d %H:%M:%S")
                task_title_test = f"MS Graph Utils Test Task - {timestamp_test}"
                task_body_test = f"Task created by standalone test of ms_graph_utils.py.\nTimestamp: {timestamp_test}"
                create_success = create_todo_task(
                    token, list_id_test, task_title_test, task_body_test
                )

                if create_success:
                    logger.info("Test 3 PASSED: Task creation successful.")
                else:
                    logger.error("Test 3 FAILED: Task creation failed.")

            else:
                logger.error(
                    f"Test 2 FAILED: Failed to retrieve list ID for '{list_name_test}'. Skipping Test 3."
                )

        else:
            logger.error(
                "Test 1 FAILED: Authentication failed. Skipping further tests."
            )

    # --- Final message regardless of success/failure ---
    logger.info("--- MS Graph Utils Test Finished ---")

# End of ms_graph_utils.py
