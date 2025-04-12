# test_ms_graph_todo.py

import os
import json
import logging
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import msal
import requests
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)-4d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logging.getLogger("msal").setLevel(logging.INFO)  # Reduce MSAL verbosity slightly
logging.getLogger("urllib3").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Config from .env ---
# Read CLIENT_ID, fallback to MS_GRAPH_CLIENT_ID if CLIENT_ID not set
CLIENT_ID: Optional[str] = os.getenv("CLIENT_ID") or os.getenv("MS_GRAPH_CLIENT_ID")
TENANT_ID: Optional[str] = os.getenv("MS_GRAPH_TENANT_ID")
# Use 'common' authority if TENANT_ID is missing, common works for most scenarios including personal accounts
AUTHORITY = "https://login.microsoftonline.com/consumers"
SCOPES = ["Tasks.ReadWrite", "User.Read"]  # Permissions needed
TARGET_LIST_NAME: str = os.getenv(
    "MS_TODO_LIST_NAME", "Tasks"
)  # Default to "Tasks" list if not set

# --- Microsoft Graph API Configuration ---
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

# --- Check Configuration ---
if not CLIENT_ID:
    logger.critical(
        "CRITICAL ERROR: CLIENT_ID (or MS_GRAPH_CLIENT_ID) not found in .env file. Exiting."
    )
    sys.exit(1)
logger.info(f"Using Client ID: {CLIENT_ID}")
logger.info(f"Using Authority URL: {AUTHORITY}")
logger.info(f"Target To-Do List Name: '{TARGET_LIST_NAME}'")


def acquire_token_device_flow(app: msal.PublicClientApplication) -> Optional[str]:
    """Acquires an access token using the device code flow."""
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

    # Fallback to device flow
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        logger.error(
            f"Failed to create device flow. Response: {flow.get('error_description')}"
        )
        return None

    print(
        f"To sign in, use a web browser to open the page {flow['verification_uri']} "
        f"and enter the code {flow['user_code']} to authenticate."
    )
    logger.info(
        f"Device flow initiated. Waiting for user authentication ({flow.get('expires_in', 900)}s timeout)..."
    )

    # Poll for token, respecting the flow's timeout
    result = app.acquire_token_by_device_flow(
        flow
    )  # This blocks until timeout or success/error

    if result and "access_token" in result:
        logger.info("Access token acquired successfully via device flow.")
        # Optional: Log user info
        if "id_token_claims" in result:
            user_info = result["id_token_claims"].get("preferred_username") or result[
                "id_token_claims"
            ].get("name")
            if user_info:
                logger.info(f"Authenticated as: {user_info}")
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
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        lists_data = response.json()
        if lists_data and "value" in lists_data and len(lists_data["value"]) > 0:
            # Assume first match is the correct one if names are unique
            list_id = lists_data["value"][0].get("id")
            if list_id:
                logger.info(f"Found list '{list_name}' with ID: {list_id}")
                return list_id
            else:
                logger.error(
                    f"List '{list_name}' found, but 'id' field is missing in response."
                )
                return None
        else:
            logger.error(f"To-Do list named '{list_name}' not found.")
            logger.debug(f"Response from list query: {lists_data}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying To-Do lists: {e}", exc_info=True)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from list query: {e}")
        logger.debug(f"Response content: {response.text}")
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
        task_data["body"] = {
            "content": task_body,
            "contentType": "text",  # or "html" if body contains HTML
        }

    logger.info(f"Attempting to create task '{task_title}' in list {list_id}...")
    logger.debug(f"Task payload: {json.dumps(task_data)}")

    try:
        response = requests.post(
            task_create_url, headers=headers, json=task_data, timeout=30
        )
        response.raise_for_status()  # Raise HTTPError for bad responses

        logger.info(f"Successfully created task '{task_title}'.")
        logger.debug(f"Create task response: {response.json()}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating To-Do task: {e}", exc_info=True)
        # Log response body if possible
        if hasattr(e, "response") and e.response is not None:
            try:
                logger.error(f"Error response content: {e.response.text}")
            except Exception:
                pass  # Ignore errors logging the error response itself
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating task: {e}", exc_info=True)
        return False


# end create_todo_task


def main():
    """Main function to test MS Graph API authentication and task creation."""
    logger.info("--- Starting Microsoft Graph To-Do Test Script ---")

    # 1. Initialize MSAL Public Client Application
    # Cache location can be customized, default is in-memory
    # For persistence across runs, specify a file path:
    # cache = msal.SerializableTokenCache()
    # if os.path.exists("my_msal_cache.bin"):
    #     cache.deserialize(open("my_msal_cache.bin", "r").read())
    # atexit.register(lambda: open("my_msal_cache.bin", "w").write(cache.serialize()) if cache.has_state_changed else None)
    # app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY, token_cache=cache)
    # Using default in-memory cache for simplicity in this test:
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

    # 2. Acquire Access Token
    access_token = acquire_token_device_flow(app)
    if not access_token:
        logger.critical("Failed to acquire access token. Exiting.")
        return

    # 3. Get To-Do List ID
    list_id = get_todo_list_id(access_token, TARGET_LIST_NAME)
    if not list_id:
        logger.critical(
            f"Failed to find To-Do list ID for '{TARGET_LIST_NAME}'. Exiting."
        )
        return

    # 4. Create a Test Task
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    test_task_title = f"Test Task from Script - {timestamp}"
    test_task_body = f"This task was created automatically by the test_ms_graph_todo.py script.\nClient ID: {CLIENT_ID}"
    success = create_todo_task(access_token, list_id, test_task_title, test_task_body)

    if success:
        logger.info("--- Microsoft Graph To-Do Test Script Finished Successfully ---")
    else:
        logger.error("--- Microsoft Graph To-Do Test Script Finished with Errors ---")


# end main

if __name__ == "__main__":
    main()
