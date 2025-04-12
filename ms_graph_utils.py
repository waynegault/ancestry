# ms_graph_utils.py
# V1.1: Added persistent token cache using msal.SerializableTokenCache

import os
import json
import logging
import time
from typing import Optional, Dict, Any
import msal
import requests
from dotenv import load_dotenv
import atexit  # For saving cache on exit
from config import config_instance  # To get DATA_DIR

load_dotenv()
logger = logging.getLogger(__name__)

# --- Configuration ---
CLIENT_ID: Optional[str] = os.getenv("CLIENT_ID") or os.getenv("MS_GRAPH_CLIENT_ID")
TENANT_ID: Optional[str] = os.getenv("MS_GRAPH_TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID or 'consumers'}"
SCOPES = ["Tasks.ReadWrite", "User.Read"]
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

# --- Persistent Token Cache Setup ---
CACHE_FILENAME = "ms_graph_cache.bin"
CACHE_FILEPATH = config_instance.DATA_DIR / CACHE_FILENAME
persistent_cache = msal.SerializableTokenCache()

try:
    if CACHE_FILEPATH.exists():
        logger.debug(f"Loading MSAL cache from: {CACHE_FILEPATH}")
        persistent_cache.deserialize(CACHE_FILEPATH.read_text(encoding="utf-8"))
    else:
        logger.debug("MSAL cache file not found. Starting with empty cache.")
except Exception as e:
    logger.warning(
        f"Failed to load MSAL cache from {CACHE_FILEPATH}: {e}. Starting with empty cache."
    )


def save_cache_on_exit():
    """Saves the MSAL token cache if it has changed."""
    try:
        if persistent_cache.has_state_changed:
            logger.debug(f"MSAL cache has changed. Saving to: {CACHE_FILEPATH}")
            CACHE_FILEPATH.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure directory exists
            CACHE_FILEPATH.write_text(persistent_cache.serialize(), encoding="utf-8")
            persistent_cache.has_state_changed = False  # Reset flag after saving
        else:
            logger.debug("MSAL cache unchanged. No save needed.")
    except Exception as e:
        logger.error(
            f"Failed to save MSAL cache to {CACHE_FILEPATH}: {e}", exc_info=True
        )


atexit.register(save_cache_on_exit)
# --- End Cache Setup ---


# --- Shared MSAL App Instance ---
msal_app_instance: Optional[msal.PublicClientApplication] = None
if CLIENT_ID:
    msal_app_instance = msal.PublicClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        token_cache=persistent_cache,  # Use the persistent cache
    )
    logger.debug(
        "Initialized shared MSAL PublicClientApplication with persistent cache."
    )
else:
    logger.error("Cannot initialize shared MSAL app: CLIENT_ID is missing.")
# --- End Shared Instance ---


def acquire_token_device_flow() -> Optional[str]:  # Removed 'app' parameter
    """
    V1.1: Acquires an access token using the device code flow.
    Uses the shared, persistent MSAL app instance.
    """
    if not msal_app_instance:
        logger.error(
            "MSAL app not initialized (missing Client ID?). Cannot acquire token."
        )
        return None

    app = msal_app_instance  # Use the shared instance

    account = None
    accounts = app.get_accounts()
    if accounts:
        logger.info("Account(s) found in cache, attempting silent acquisition.")
        account = accounts[0]  # Use the first cached account
        result = app.acquire_token_silent(SCOPES, account=account)
        if result and "access_token" in result:
            logger.info("Access token acquired silently from cache.")
            return result["access_token"]
        else:
            logger.info("Silent token acquisition failed, attempting device flow.")
            # If silent fails, remove potentially bad account from cache?
            # app.remove_account(account) # Optional: Be careful with this

    # Fallback to device flow
    logger.debug("Initiating device flow...")
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        logger.error(
            f"Failed to create device flow. Response: {flow.get('error_description')}"
        )
        return None

    # Display instructions for the user
    print(
        f"\n--- MS GRAPH AUTH REQUIRED ---\n"
        f"To sign in, use a web browser to open the page:\n{flow['verification_uri']}\n"
        f"and enter the code: {flow['user_code']}\n"
        f"Waiting for you to authenticate in the browser...\n"
        f"------------------------------"
    )
    logger.info(
        f"Device flow initiated. Waiting for user authentication ({flow.get('expires_in', 900)}s timeout)..."
    )

    # Poll for token, respecting the flow's timeout
    result = app.acquire_token_by_device_flow(
        flow  # This blocks until timeout or success/error
    )

    # Process result
    if result and "access_token" in result:
        logger.info("Access token acquired successfully via device flow.")
        if "id_token_claims" in result:
            user_info = result["id_token_claims"].get("preferred_username") or result[
                "id_token_claims"
            ].get("name")
            if user_info:
                logger.info(f"Authenticated as: {user_info}")
        persistent_cache.has_state_changed = (
            True  # Mark cache as changed after successful device flow
        )
        return result["access_token"]
    elif result and "error_description" in result:
        logger.error(
            f"Failed to acquire token via device flow: {result.get('error_description')}"
        )
        return None
    else:
        logger.error(f"Device flow failed or timed out. Result: {result}")
        return None


# end acquire_token_device_flow


def get_todo_list_id(access_token: str, list_name: str) -> Optional[str]:
    """Finds the ID of a specific To-Do list by its display name."""
    if not access_token:
        logger.error("Cannot get list ID: Access token is missing.")
        return None

    headers = {"Authorization": f"Bearer {access_token}"}
    list_query_url = (
        f"{GRAPH_API_ENDPOINT}/me/todo/lists?$filter=displayName eq '{list_name}'"
    )
    logger.info(f"Querying for To-Do list named '{list_name}'...")

    try:
        response = requests.get(list_query_url, headers=headers, timeout=30)
        response.raise_for_status()

        lists_data = response.json()
        if lists_data and "value" in lists_data and len(lists_data["value"]) > 0:
            list_id = lists_data["value"][0].get("id")
            if list_id:
                logger.info(f"Found list '{list_name}' with ID: {list_id}")
                return list_id
            else:
                logger.error(f"List '{list_name}' found, but 'id' field is missing.")
                return None
        else:
            logger.error(f"To-Do list named '{list_name}' not found.")
            logger.debug(f"Response from list query: {lists_data}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying To-Do lists: {e}", exc_info=True)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from list query: {e}")
        logger.debug(f"Response text: {response.text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting list ID: {e}", exc_info=True)
        return None


# end get_todo_list_id


def create_todo_task(
    access_token: str, list_id: str, task_title: str, task_body: Optional[str] = None
) -> bool:
    """Creates a new task in the specified To-Do list."""
    if not access_token or not list_id:
        logger.error("Cannot create task: Access token or List ID is missing.")
        return False

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    task_create_url = f"{GRAPH_API_ENDPOINT}/me/todo/lists/{list_id}/tasks"

    task_data: Dict[str, Any] = {"title": task_title}
    if task_body:
        task_data["body"] = {"content": task_body, "contentType": "text"}

    logger.info(f"Attempting to create task '{task_title}' in list {list_id}...")
    logger.debug(f"Task payload: {json.dumps(task_data)}")

    try:
        response = requests.post(
            task_create_url, headers=headers, json=task_data, timeout=30
        )
        response.raise_for_status()

        logger.info(f"Successfully created task '{task_title}'.")
        logger.debug(f"Create task response: {response.json()}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating To-Do task: {e}", exc_info=True)
        if hasattr(e, "response") and e.response is not None:
            try:
                logger.error(f"Error response content: {e.response.text}")
            except Exception:
                pass
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating task: {e}", exc_info=True)
        return False


# end create_todo_task


# --- Standalone Test ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)-4d] %(message)s",
    )
    logger.info("--- Testing ms_graph_utils with Persistent Cache ---")
    if not msal_app_instance:  # Check if app instance was created
        logger.critical("MSAL App could not be initialized. Check CLIENT_ID.")
    else:
        # Acquire token (will use cache or prompt for device flow)
        token = acquire_token_device_flow()
        if token:
            logger.info(
                "Authentication successful (using persistent cache or device flow)."
            )
            # Example: Get list ID (optional test)
            # list_name_test = os.getenv("MS_TODO_LIST_NAME", "Tasks")
            # list_id_test = get_todo_list_id(token, list_name_test)
            # if list_id_test:
            #     logger.info(f"Successfully retrieved list ID for '{list_name_test}'.")
            # else:
            #     logger.error(f"Failed to retrieve list ID for '{list_name_test}'.")
        else:
            logger.error("Authentication failed.")
    logger.info("--- MS Graph Utils Test Finished ---")

# End of ms_graph_utils.py
