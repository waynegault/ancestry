# ms_graph_utils.py


import os
import json
import logging
import time
from typing import Optional, Dict, Any
import msal
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)  # Use module-specific logger

# Load config specifically needed for this module
CLIENT_ID: Optional[str] = os.getenv("CLIENT_ID") or os.getenv("MS_GRAPH_CLIENT_ID")
TENANT_ID: Optional[str] = os.getenv("MS_GRAPH_TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID or 'consumers'}"  # Use consumers based on testing
SCOPES = ["Tasks.ReadWrite", "User.Read"]
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"



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



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)-4d] %(message)s",
    )
    logger.info("Testing ms_graph_utils...")
    if not CLIENT_ID:
        logger.error("CLIENT_ID not set.")
    else:
        app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
        token = acquire_token_device_flow(app)
        if token:
            logger.info("Authentication successful.")
            # Add more tests if needed
        else:
            logger.error("Authentication failed.")
