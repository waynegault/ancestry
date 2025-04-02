# File: action8_messaging.py

# Standard library imports
import inspect
import json
import logging
import math
import os
import random
import re
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone # Added timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from urllib.parse import urljoin, urlparse, urlencode

# Third-party imports
import requests # Keep requests if used by _api_req indirectly
from sqlalchemy import (Boolean, Column, DateTime, Enum as SQLEnum, Integer,
                        String, desc, func)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DbSession, joinedload # Use DbSession alias

# Local application imports
from config import config_instance, selenium_config # Keep selenium_config for API timeouts maybe?
from database import (InboxStatus, Person, RoleType, db_transn, DnaMatch, FamilyTree,
                      MessageHistory, MessageType)
from utils import (_api_req, DynamicRateLimiter, SessionManager, retry,
                 time_wait, make_ube, format_name)
from cache import cache_result
# --- Remove UI selectors, keep WAIT_FOR_PAGE_SELECTOR if used by any remaining nav ---
# from my_selectors import ( ... )

# Initialize logging
logger = logging.getLogger("logger")

# --- Constants ---
# Log app mode
logger.info(f"APP_MODE is: {config_instance.APP_MODE}")

# Define message intervals based on app mode
MESSAGE_INTERVALS = {
    "testing": timedelta(minutes=1),
    "production": timedelta(weeks=8),
    "dry_run": timedelta(seconds=10), # Adjusted dry_run delay
}
MIN_MESSAGE_INTERVAL = MESSAGE_INTERVALS.get(config_instance.APP_MODE, timedelta(weeks=6)) # Default to production
logger.info(f"Using minimum message interval: {MIN_MESSAGE_INTERVAL}")

# Message types (keys should match messages.json)
MESSAGE_TYPES = {
    "In_Tree-Initial": "In_Tree-Initial",
    "In_Tree-Follow_Up": "In_Tree-Follow_Up",
    "In_Tree-Final_Reminder": "In_Tree-Final_Reminder",
    "Out_Tree-Initial": "Out_Tree-Initial",
    "Out_Tree-Follow_Up": "Out_Tree-Follow_Up",
    "Out_Tree-Final_Reminder": "Out_Tree-Final_Reminder",
    "In_Tree-Initial_for_was_Out_Tree": "In_Tree-Initial_for_was_Out_Tree",
}

# Confirmation constants (from test script)
CONFIRMATION_DELAY = 5
CONFIRMATION_MESSAGE_FETCH_LIMIT = 25 # How many recent messages to fetch for confirmation

# --- Load Message Templates ---
@cache_result("message_templates")
def load_message_templates() -> Dict[str, str]:  # Explicitly annotate the return type
    """Loads message templates from messages.json."""
    messages_path = Path("messages.json")
    if not messages_path.exists():
        logger.critical(f"CRITICAL: messages.json not found at {messages_path.resolve()}")
        return {}
    try:
        with messages_path.open("r", encoding="utf-8") as f:
            templates = json.load(f)
            if not isinstance(templates, dict) or not all(isinstance(v, str) for v in templates.values()):
                logger.critical(f"CRITICAL: messages.json does not contain a dictionary of strings.")
                return {}
            logger.info("Message templates loaded successfully.")
            return templates
    except json.JSONDecodeError as e:
        logger.critical(f"CRITICAL: Error decoding messages.json: {e}")
        return {}
    except Exception as e:
        logger.critical(f"CRITICAL: Error loading messages.json: {e}", exc_info=True)
        return {}
# End of load_message_templates

MESSAGE_TEMPLATES: Dict[str, str] = cast(Dict[str, str], load_message_templates())  # Explicitly cast the return type
if not MESSAGE_TEMPLATES:
    logger.error("Message templates failed to load. Messaging functionality will be limited.")

# --- Helper Functions ---

# format_name function moved to utils.py, imported from there.

def _find_conversation_id_for_person(session: DbSession, person_id: int) -> Optional[str]:
    """Looks up the conversation ID for a given person ID in the database."""
    logger.debug(f"Looking up InboxStatus for people_id: {person_id}")
    try:
        inbox_entry = session.query(InboxStatus).filter(InboxStatus.people_id == person_id).first()
        if not inbox_entry:
            logger.debug(f"InboxStatus entry not found for people_id: {person_id}. Cannot get conversation ID.") # Changed to DEBUG
            return None

        if inbox_entry.conversation_id is None:
            logger.debug(f"InboxStatus found for people_id {person_id}, but conversation_id is NULL.") # Changed to DEBUG
            return None

        logger.debug(f"Found conversation_id: {inbox_entry.conversation_id} for people_id {person_id}")
        return inbox_entry.conversation_id
    except SQLAlchemyError as e:
        logger.error(f"Database error during conversation ID lookup for people_id {person_id}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during conversation ID lookup for people_id {person_id}: {e}", exc_info=True)
        return None
# end _find_conversation_id_for_person


def determine_next_message_type(
    last_message_details: Optional[Tuple[str, datetime, str]],
    is_in_family_tree: bool,
    last_received_timestamp: Optional[datetime],
) -> Optional[str]:
    """
    Determines the next message type based on history and inbox status.
    (Logic remains largely the same as before, but added logging)
    """
    logger.debug(f"Determining next msg type. Last Sent: {last_message_details}, In Tree: {is_in_family_tree}, Last Received: {last_received_timestamp}")

    # --- Condition 1: No message ever sent ---
    if not last_message_details:
        if last_received_timestamp:
            logger.debug("No previous message sent, but a message has been received. Skipping.")
            return None
        next_type = "In_Tree-Initial" if is_in_family_tree else "Out_Tree-Initial"
        logger.debug(f"Next type determined: {next_type} (No prior message sent, none received).")
        return next_type

    # --- Extract details of the last sent message ---
    last_message_type, last_sent_at, last_message_status = last_message_details

    # Ensure last_sent_at is offset-aware (assuming UTC from DB) or naive matching local time
    if isinstance(last_sent_at, datetime):
        if last_sent_at.tzinfo is None:
             # If DB time is naive, assume it's local time for comparison with datetime.now()
             pass # Or convert both to UTC if DB time is UTC: last_sent_at = last_sent_at.replace(tzinfo=timezone.utc)
        # else: It's offset-aware, proceed
    else:
        logger.warning(f"Invalid last_sent_at type ({type(last_sent_at)}), cannot reliably determine next step.")
        return None

    # --- Condition 2: Check for reply ---
    if last_received_timestamp and isinstance(last_received_timestamp, datetime):
        # Ensure comparison is valid (naive vs aware) - Assuming last_received is also naive or same timezone
        if last_received_timestamp.tzinfo is None and last_sent_at.tzinfo is not None:
             # Example: make last_sent_at naive if it was aware (adjust based on actual timezones)
             # last_sent_at_compare = last_sent_at.astimezone(None) # Requires appropriate local timezone setup
             logger.warning("Timezone mismatch between last_received and last_sent_at. Skipping reply check.") # Fallback
        elif last_received_timestamp.tzinfo is not None and last_sent_at.tzinfo is None:
             logger.warning("Timezone mismatch between last_received and last_sent_at. Skipping reply check.") # Fallback
        elif last_received_timestamp > last_sent_at:
            logger.debug(f"Reply received ({last_received_timestamp}) after last sent message ({last_sent_at}). No follow-up needed.")
            return None

    # --- Condition 3: Check message interval ---
    # Use MIN_MESSAGE_INTERVAL defined globally
    # Ensure comparison uses compatible datetime objects (naive vs aware)
    now = datetime.now() # Naive local time
    if last_sent_at.tzinfo is not None:
        # Example: Convert now() to UTC if last_sent_at is UTC
        # now = datetime.now(timezone.utc)
        logger.warning("Comparing naive datetime.now() with offset-aware last_sent_at. Interval check might be inaccurate.") # Fallback

    time_since_last_message = now - last_sent_at
    if time_since_last_message < MIN_MESSAGE_INTERVAL:
        logger.debug(f"Skipping follow-up: Too soon since last message ({time_since_last_message} < {MIN_MESSAGE_INTERVAL}).")
        return None

    # --- Condition 4: Determine next step based on last type and current tree status ---
    next_type: Optional[str] = None # Default to None
    if is_in_family_tree:
        if last_message_type.startswith("Out_Tree"):
            next_type = "In_Tree-Initial_for_was_Out_Tree"
        elif last_message_type == "In_Tree-Initial" or last_message_type == "In_Tree-Initial_for_was_Out_Tree":
            next_type = "In_Tree-Follow_Up"
        elif last_message_type == "In_Tree-Follow_Up":
            next_type = "In_Tree-Final_Reminder"
        # else: stays None (end of In_Tree sequence)
    else: # Match is Out_Tree
        if last_message_type.startswith("In_Tree"):
            logger.warning(f"Match was In_Tree ({last_message_type}) but is now Out_Tree. Skipping.")
            next_type = None
        elif last_message_type == "Out_Tree-Initial":
            next_type = "Out_Tree-Follow_Up"
        elif last_message_type == "Out_Tree-Follow_Up":
            next_type = "Out_Tree-Final_Reminder"
        # else: stays None (end of Out_Tree sequence)

    if next_type:
        logger.debug(f"Next type determined: {next_type} (Following sequence).")
    else:
        logger.debug("Next type determined: None (End of sequence or skip condition).")

    return next_type
# End of determine_next_message_type

def _fetch_conversation_messages(session_manager: SessionManager, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches the most recent messages for a given conversation ID via API GET.
    Uses the confirmed endpoint /conversations/{id}/messages.
    Returns a list of message objects or None on failure.
    Forces requests library path for reliability.
    """
    # --- Prerequisite Checks ---
    if not conversation_id: logger.error("Cannot fetch messages: conversation_id is missing."); return None
    if not session_manager: logger.error("Cannot fetch messages: SessionManager not available."); return None # Check session_manager itself
    if not session_manager.my_profile_id: logger.error("Cannot fetch messages: my_profile_id missing."); return None
    # Driver might not be strictly needed if forcing requests, but keep check for now for UBE header
    if not session_manager.driver: logger.warning("Cannot fetch messages: driver not available (UBE header might be missing).");

    logger.info(f"Attempting to fetch recent messages for conversation_id: {conversation_id} for confirmation.")

    # --- Prepare API Call ---
    try:
        base_url_with_slash = config_instance.BASE_URL.rstrip('/') + '/'
        base_api_path = f"app-api/express/v2/conversations/{conversation_id}/messages"
        query_params = {"limit": CONFIRMATION_MESSAGE_FETCH_LIMIT}
        api_url = urljoin(base_url_with_slash, f"{base_api_path}?{urlencode(query_params)}")
    except Exception as url_e:
        logger.error(f"Failed to construct fetch messages URL: {url_e}", exc_info=True)
        return None

    api_description = "Fetch Conversation Messages API (action8 confirmation)"

    # Headers construction moved inside _api_req based on api_description

    logger.info(f"Sending GET to: {api_url} via requests library")

    # --- Execute API Call ---
    response_data: Optional[Any] = None
    try:
        # Force requests path for reliability
        response_data = _api_req(
            url=api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description=api_description,
            force_requests=True # Use requests library
        )
    except Exception as api_call_e:
        logger.error(f"Exception during _api_req (Fetch Messages - requests): {api_call_e}", exc_info=True)
        return None

    # --- Process Response ---
    if response_data is None:
        logger.error("API call to fetch messages failed (returned None).")
        return None

    messages_list: Optional[List[Dict[str, Any]]] = None
    if isinstance(response_data, dict) and "messages" in response_data and isinstance(response_data["messages"], list):
        messages_list = response_data["messages"]
        total_count = response_data.get("paging", {}).get("total_count", "N/A")
        logger.info(f"Successfully fetched {len(messages_list)} messages (out of total {total_count}) for confirmation.")
    else:
        logger.error(f"Failed to parse messages from API response. Expected dict with 'messages' list, but got type: {type(response_data)}.")
        logger.debug(f"Full Fetch Response Data: {response_data}")
        return None

    # Final structure check
    if messages_list:
        # Optional: Add more validation if needed
        # Example: Check if timestamps are valid ISO strings
        for msg in messages_list:
            if 'created_at' in msg and isinstance(msg['created_at'], str):
                try:
                    # Attempt parsing to validate format
                    datetime.fromisoformat(msg['created_at'].replace('Z', '+00:00'))
                except ValueError:
                    logger.warning(f"Invalid created_at format found in fetched message: {msg.get('created_at')}")
                    # Handle invalid format if needed (e.g., skip message, return None)
            # Add other validation as needed
        return messages_list
    else:
        return None # Error already logged
# end _fetch_conversation_messages

def send_messages_to_matches(session_manager: SessionManager) -> bool:
    """Sends messages to DNA matches based on criteria, using API calls."""
    logger.info("Starting message sending process (API Based - Dual Endpoint)...")

    # --- Prerequisites ---
    if not session_manager: logger.error("SessionManager instance is required."); return False
    # Check API login status instead of full session active, as we primarily use requests
    if not session_manager._verify_api_login_status():
         logger.error("API login verification failed. Cannot send messages.")
         return False
    # Driver is still needed for UBE header generation initially
    if not session_manager.driver: logger.warning("Driver not available, UBE header will be missing from API requests.")

    if not session_manager.my_profile_id: logger.error("Cannot send messages: Own profile ID missing."); return False
    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()
    MY_PROFILE_ID_UPPER = session_manager.my_profile_id.upper()
    if not MESSAGE_TEMPLATES: logger.error("Cannot send messages: Message templates failed to load."); return False

    overall_success = True
    processed_count = 0
    sent_count = 0
    skipped_count = 0
    error_count = 0
    potential_matches = []
    max_to_send = config_instance.MAX_INBOX

    try:
        # Use context manager for DB session
        with session_manager.get_db_conn_context() as db_session:
            if not db_session:
                logger.error("Failed to get database connection. Cannot proceed.")
                return False

            # --- Query for candidates ---
            potential_matches_query = (
                db_session.query(
                    DnaMatch.people_id, DnaMatch.predicted_relationship,
                    Person.username, Person.profile_id, Person.in_my_tree, Person.first_name,
                    FamilyTree.actual_relationship, FamilyTree.person_name_in_tree, FamilyTree.relationship_path
                )
                .join(Person, DnaMatch.people_id == Person.id)
                .outerjoin(FamilyTree, Person.id == FamilyTree.people_id)
                .filter(Person.profile_id.isnot(None), Person.profile_id != "", Person.profile_id != "UNKNOWN") # More strict filter
                .filter(Person.status == "active") # Only message active matches
                .order_by(Person.id) # Ensure consistent processing order
            )
            potential_matches = potential_matches_query.all()

            logger.info(f"Found {len(potential_matches)} potential DNA matches to check.")
            if not potential_matches: return True # Nothing to process

            total_family_tree_rows = db_session.query(func.count(FamilyTree.id)).scalar() or 0
            logger.info(f"Total rows found in family_tree table: {total_family_tree_rows}")

            # --- Iterate through matches ---
            for match_data in potential_matches:
                # Check max send limit inside the loop
                if max_to_send != 0 and sent_count >= max_to_send:
                    logger.info(f"Reached MAX_INBOX limit ({max_to_send}). Stopping message sending.")
                    break
                processed_count += 1

                # Unpack data safely
                try:
                    ( person_id, predicted_relationship, username, profile_id, is_in_family_tree, first_name,
                        actual_relationship, person_name_in_tree, relationship_path ) = match_data
                    # Validate essential unpacked data
                    if not person_id or not profile_id or not username:
                         raise ValueError("Missing essential person data (id, profile_id, username)")
                except (ValueError, TypeError) as unpack_e:
                    logger.error(f"Error unpacking or validating match data: {unpack_e}. Row: {match_data}", exc_info=True)
                    error_count += 1; overall_success = False; continue # Skip this row

                recipient_profile_id_upper = profile_id.upper()
                log_prefix = f"{username} (ID: {person_id}, Profile: {recipient_profile_id_upper})"
                logger.debug(f"\n--- Processing {log_prefix} ---")

                # Find existing conversation ID from DB
                conversation_id = _find_conversation_id_for_person(db_session, person_id)

                # Get last sent message and last received message details
                last_message_sent_details: Optional[Tuple[str, datetime, str]] = None
                last_received_message_timestamp: Optional[datetime] = None
                they_sent_last: bool = False
                try:
                    # --- Get Last Sent Message ---
                    last_sent_query = (
                        db_session.query(MessageHistory.sent_at, MessageType.type_name, MessageHistory.status)
                        .join(MessageType)
                        .filter(MessageHistory.people_id == person_id)
                        .order_by(desc(MessageHistory.sent_at))
                        .first()
                    )
                    if last_sent_query:
                        last_message_sent_details = (last_sent_query.type_name, last_sent_query.sent_at, last_sent_query.status)
                        logger.debug(f"Last sent details: Type={last_sent_query.type_name}, At={last_sent_query.sent_at}, Status={last_sent_query.status}")
                    else:
                        logger.debug("No previous message sent found in history.")

                    # --- Get Inbox Status (Last Received) ---
                    inbox_status = db_session.query(InboxStatus).filter(InboxStatus.people_id == person_id).first()
                    if inbox_status:
                         logger.debug(f"InboxStatus found: Role={inbox_status.my_role}, LastMsgAt={inbox_status.last_message_timestamp}, ConvID={inbox_status.conversation_id}")
                         # They sent last only if their role was author (meaning our role is recipient)
                         if inbox_status.my_role == RoleType.RECIPIENT and inbox_status.last_message_timestamp:
                              last_received_message_timestamp = inbox_status.last_message_timestamp
                              they_sent_last = True
                              logger.debug(f"They sent last confirmed. Last received at: {last_received_message_timestamp}")
                         # Update conversation_id if found in InboxStatus and not found earlier
                         if not conversation_id and inbox_status.conversation_id:
                              conversation_id = inbox_status.conversation_id
                              logger.debug(f"Updated conversation_id from InboxStatus: {conversation_id}")
                    else:
                         logger.debug("No InboxStatus found.")

                except SQLAlchemyError as db_e:
                    logger.error(f"Database error querying history/inbox for {log_prefix}: {db_e}", exc_info=True)
                    error_count += 1; overall_success = False; continue # Skip this person on DB error

                # Skip if they sent the last message and we haven't replied yet
                if they_sent_last and last_received_message_timestamp: # Check timestamp exists
                     # Make datetimes timezone-aware (assuming UTC if naive) for safe comparison
                     if last_message_sent_details:
                         lm_type, lm_sent_at, lm_status = last_message_sent_details
                         # Make aware assuming UTC
                         if lm_sent_at.tzinfo is None: lm_sent_at = lm_sent_at.replace(tzinfo=timezone.utc)
                         if last_received_message_timestamp.tzinfo is None: last_received_message_timestamp = last_received_message_timestamp.replace(tzinfo=timezone.utc)

                         if last_received_message_timestamp > lm_sent_at:
                              logger.info(f"Skipping {log_prefix}: They sent the last message ({last_received_message_timestamp}), awaiting reply or interval.")
                              skipped_count += 1; continue
                         else:
                              logger.debug(f"They sent last ({last_received_message_timestamp}), but our last sent ({lm_sent_at}) was later. Proceeding.")
                     else:
                          # They sent a message, but we never sent one before.
                           logger.info(f"Skipping {log_prefix}: They sent the first message ({last_received_message_timestamp}).")
                           skipped_count += 1; continue


                # Determine the next message type based on rules
                next_message_type_key = determine_next_message_type(last_message_sent_details, bool(is_in_family_tree), last_received_message_timestamp)
                if not next_message_type_key:
                    logger.info(f"Skipping {log_prefix}: No appropriate next message type determined (rule/timing).")
                    skipped_count += 1; continue

                # Get the message template
                message_template = MESSAGE_TEMPLATES.get(next_message_type_key)
                if not message_template:
                    logger.error(f"Missing message template for key '{next_message_type_key}' for {log_prefix}.")
                    error_count += 1; overall_success = False; continue # Skip if template missing

                # Select the best name to use
                name_to_use = "Valued Relative"; source_of_name="Fallback"
                if is_in_family_tree and person_name_in_tree:
                    name_to_use = person_name_in_tree; source_of_name="FamilyTree"
                elif first_name:
                    name_to_use = first_name; source_of_name="Person.first_name"
                elif username and username != "Unknown": # Use username if first_name unavailable
                    name_to_use = username; source_of_name="Person.username"
                # Use the imported format_name function from utils
                formatted_name = format_name(name_to_use)
                logger.debug(f"Using name '{formatted_name}' (Source: {source_of_name}) for formatting.")

                # Prepare data for formatting the message template
                format_data = {
                    "name": formatted_name,
                    "predicted_relationship": predicted_relationship or "N/A",
                    "actual_relationship": actual_relationship or "N/A",
                    "relationship_path": relationship_path or "N/A",
                    "total_rows": total_family_tree_rows
                }

                # Check for missing keys before formatting
                try:
                    # Find all {key} patterns in the template
                    required_keys_in_template = set(re.findall(r'\{(\w+)\}', message_template))
                except Exception as regex_e:
                    logger.error(f"Regex error checking template keys for '{next_message_type_key}': {regex_e}")
                    required_keys_in_template = set() # Assume none needed on error

                # Check if all required keys exist and have non-None values in format_data
                missing_keys = [key for key in required_keys_in_template if key not in format_data or format_data[key] is None]
                # Allow 'N/A' as a valid value if a key is optional but present
                missing_keys = [key for key in missing_keys if format_data.get(key) != "N/A"]

                if missing_keys:
                    logger.warning(f"Skipping {log_prefix}: Missing/None required keys {missing_keys} in format data for template '{next_message_type_key}'. Available: {list(format_data.keys())}")
                    error_count += 1; overall_success = False; continue

                # Format the message
                try:
                    message_text = message_template.format(**format_data)
                except KeyError as e:
                     logger.error(f"KeyError formatting message for {log_prefix} (Template: '{next_message_type_key}'): Missing key {e}. Available: {list(format_data.keys())}", exc_info=False)
                     error_count += 1; overall_success = False; continue
                except Exception as e:
                    logger.error(f"Error formatting message for {log_prefix} using template '{next_message_type_key}': {e}", exc_info=True)
                    error_count += 1; overall_success = False; continue

                logger.debug(f"Formatted message for {log_prefix} ({next_message_type_key}): '{message_text[:100]}...'")

                # --- Send Message via API ---
                message_status: str = "skipped (logic_error)"
                db_update_needed: bool = False
                send_or_type_success: bool = False
                new_conversation_id_from_api: Optional[str] = None
                is_initial_message: bool = not conversation_id
                send_api_url: str = ""
                payload: Dict[str, Any] = {}
                send_api_description: str = ""

                # Determine API endpoint and payload based on whether it's an initial message
                if is_initial_message:
                    send_api_url = urljoin(config_instance.BASE_URL.rstrip('/')+'/', "app-api/express/v2/conversations/message")
                    send_api_description = "Create Conversation API"
                    payload = {
                        "content": message_text,
                        "author": MY_PROFILE_ID_LOWER, # Author must be lowercase here
                        "index": 0, # Required for creation
                        "created": 0, # Required for creation
                        "conversation_members": [
                            {"user_id": recipient_profile_id_upper.lower(), "family_circles": [] }, # Recipient lowercase
                            {"user_id": MY_PROFILE_ID_LOWER} # Self lowercase
                        ]
                    }
                elif conversation_id: # Existing conversation
                    send_api_url = urljoin(config_instance.BASE_URL.rstrip('/')+'/', f"app-api/express/v2/conversations/{conversation_id}")
                    send_api_description = "Send Message API (Existing Conv)"
                    payload = {
                        "content": message_text,
                        "author": MY_PROFILE_ID_LOWER # Author lowercase
                    }
                else:
                    # This condition should theoretically not be reached due to conversation_id checks
                    logger.error(f"Cannot determine send API URL/payload for {log_prefix}. is_initial={is_initial_message}, conv_id={conversation_id}")
                    error_count += 1; overall_success = False; continue

                # Headers are now mostly handled by _api_req using API_CONTEXTUAL_HEADERS
                send_api_headers = {} # Pass empty dict, _api_req will populate

                # --- Handle Different Modes (Dry Run, Production, Testing) ---
                if config_instance.APP_MODE == "dry_run":
                    log_action = "create conversation and send" if is_initial_message else "send follow-up"
                    logger.info(f"[DRY RUN] Would {log_action} '{next_message_type_key}' to {log_prefix} with name '{formatted_name}'. Message: '{message_text[:100]}...'")
                    message_status = "typed (dry_run)"
                    db_update_needed = True
                    send_or_type_success = True
                    # Simulate conversation ID creation for dry run consistency
                    if is_initial_message:
                        new_conversation_id_from_api = f"dryrun_{uuid.uuid4()}"

                elif config_instance.APP_MODE in ["production", "testing"]:
                    log_action = "create conversation and send" if is_initial_message else "send follow-up"
                    logger.info(f"Attempting to {log_action} '{next_message_type_key}' to {log_prefix} via API...")
                    send_response_data: Optional[Any] = None
                    try:
                        # Execute API POST request using helper, force requests, disable CSRF
                        send_response_data = _api_req(
                            url=send_api_url,
                            driver=session_manager.driver, # Still needed for UBE header
                            session_manager=session_manager,
                            method="POST",
                            json_data=payload,
                            use_csrf_token=False, # Explicitly disable CSRF based on cURL success
                            headers=send_api_headers, # Pass empty dict, headers derived from api_description
                            api_description=send_api_description,
                            force_requests=True # Force requests library path
                        )
                    except Exception as api_e:
                        logger.error(f"Exception during _api_req ({send_api_description}) for {log_prefix}: {api_e}", exc_info=True)
                        message_status = "send_error (api_exception)"
                        error_count += 1; overall_success = False; continue # Skip to next person

                    # Process API response
                    if send_response_data is not None:
                        # --- Check for successful initial message (status 201 expected) ---
                        if is_initial_message and isinstance(send_response_data, dict) and 'conversation_id' in send_response_data:
                            # Initial message successful: Extract conversation ID
                            new_conv_id = str(send_response_data.get('conversation_id'))
                            msg_details = send_response_data.get('message', {})
                            resp_author = str(msg_details.get('author', '')).upper() if isinstance(msg_details, dict) else ''
                            # Ensure our profile ID is uppercase for comparison
                            if new_conv_id and resp_author == MY_PROFILE_ID_UPPER:
                                conversation_id = new_conv_id # Update conversation_id for subsequent steps
                                new_conversation_id_from_api = new_conv_id # Store for DB update
                                message_status = "sent (api_created)" # Initial status
                                send_or_type_success = True; db_update_needed = True
                                logger.info(f"Successfully created conversation ({conversation_id}) and sent initial '{next_message_type_key}' to {log_prefix}.")
                            else:
                                logger.error(f"API created conversation for {log_prefix}, but response invalid. ConvID: '{new_conv_id}', Author: '{resp_author}'. Expected author: '{MY_PROFILE_ID_UPPER}'")
                                message_status="send_error (create_bad_response)"
                                error_count+=1; send_or_type_success=False; overall_success=False
                        # --- Check for successful follow-up message (status 200 or 201 expected, check content) ---
                        # Follow-up returns the message object itself, not the conversation object
                        elif not is_initial_message and isinstance(send_response_data, dict) and 'index' in send_response_data and 'author' in send_response_data:
                            # Follow-up message successful: Check author
                            resp_author = str(send_response_data.get('author', '')).upper()
                            resp_content = send_response_data.get('content')
                            # Ensure our profile ID is uppercase for comparison
                            if resp_author == MY_PROFILE_ID_UPPER and resp_content == message_text:
                                logger.info(f"Successfully sent follow-up '{next_message_type_key}' to {log_prefix} (ConvID: {conversation_id}).")
                                message_status="sent (api_ok)" # Initial status
                                send_or_type_success=True; db_update_needed=True
                            else:
                                logger.error(f"API sent follow-up for {log_prefix}, but response validation failed. Author: '{resp_author}' (Exp: '{MY_PROFILE_ID_UPPER}'), Content Match: {resp_content == message_text}")
                                message_status="send_error (followup_bad_response)"
                                error_count+=1; send_or_type_success=False; overall_success=False
                        else:
                            # Unexpected response format
                            logger.error(f"API call ({send_api_description}) for {log_prefix} returned unexpected response format. Type: {type(send_response_data)}")
                            logger.debug(f"Full response data: {send_response_data}")
                            message_status="send_error (bad_response)"
                            error_count+=1; send_or_type_success=False; overall_success=False

                        # --- Confirmation Step (If send appeared successful) ---
                        if send_or_type_success and conversation_id:
                            logger.debug(f"Pausing {CONFIRMATION_DELAY}s before confirmation fetch for {log_prefix}...")
                            time.sleep(CONFIRMATION_DELAY)
                            fetched_messages = _fetch_conversation_messages(session_manager, conversation_id)
                            if fetched_messages is not None:
                                confirmation_window = timedelta(minutes=2) # Widen window slightly
                                now_utc = datetime.now(timezone.utc) # Use timezone-aware now
                                confirmed = False
                                # --- ADDED: Log details for comparison ---
                                logger.debug(f"CONFIRM CHECK for {log_prefix}:")
                                logger.debug(f"  - Expected Author: {MY_PROFILE_ID_UPPER}")
                                expected_content_snippet = f"{message_text[:50]}...{message_text[-50:]}" if len(message_text) > 100 else message_text
                                logger.debug(f"  - Expected Content Snippet: {expected_content_snippet}")
                                logger.debug(f"  - Confirmation Window: {confirmation_window}")
                                logger.debug(f"  - Current UTC Time: {now_utc}")
                                # --- END ADDED ---
                                for i, msg in enumerate(fetched_messages): # Add index for logging
                                    msg_author = str(msg.get('author', '')).upper()
                                    msg_content = msg.get('content')
                                    msg_created_str = msg.get('created_at')
                                    msg_index = msg.get('index', 'N/A')

                                    # --- ADDED: Detailed log for each fetched message ---
                                    fetched_content_snippet = f"{str(msg_content)[:50]}...{str(msg_content)[-50:]}" if msg_content and len(str(msg_content)) > 100 else str(msg_content)
                                    logger.debug(f"  Checking Fetched Msg [{i}] (Index: {msg_index}):")
                                    logger.debug(f"    Author: {msg_author} (Match: {msg_author == MY_PROFILE_ID_UPPER})")
                                    logger.debug(f"    Content Snippet: {fetched_content_snippet} (Match: {msg_content == message_text})")
                                    logger.debug(f"    Created Str: {msg_created_str}")
                                    # --- END ADDED ---

                                    # Check Author and Content first (most likely mismatches)
                                    author_match = (msg_author == MY_PROFILE_ID_UPPER)
                                    content_match = (msg_content == message_text)

                                    if author_match and content_match:
                                         if msg_created_str and isinstance(msg_created_str, str):
                                              try:
                                                   # Parse ISO timestamp, assume UTC if Z present, make aware
                                                   msg_created_dt = datetime.fromisoformat(msg_created_str.replace('Z', '+00:00'))
                                                   # Ensure it's offset-aware for comparison
                                                   if msg_created_dt.tzinfo is None:
                                                        msg_created_dt = msg_created_dt.replace(tzinfo=timezone.utc) # Assume UTC if naive

                                                   time_diff = now_utc - msg_created_dt
                                                   time_match = (time_diff >= timedelta(seconds=0) and time_diff < confirmation_window) # Ensure not in future

                                                   # --- ADDED: Log timestamp comparison ---
                                                   logger.debug(f"    Parsed Timestamp: {msg_created_dt}")
                                                   logger.debug(f"    Time Difference: {time_diff}")
                                                   logger.debug(f"    Within Window: {time_match}")
                                                   # --- END ADDED ---

                                                   if time_match:
                                                        confirmed = True
                                                        logger.debug(f"    --> CONFIRMED MATCH at index {msg_index}")
                                                        break # Stop checking once confirmed
                                              except ValueError:
                                                   logger.warning(f"    Could not parse timestamp '{msg_created_str}' during confirmation.")
                                         else:
                                              logger.warning(f"    Missing or invalid 'created_at' field in fetched message.")
                                    # else: logger.debug(f"    Author or Content mismatch. Skipping time check.") # Optional: Add if needed

                                # End loop
                                if confirmed:
                                    message_status = "sent_confirmed" # Update status
                                    logger.info(f"Message send to {log_prefix} confirmed by fetching recent messages.")
                                else:
                                    message_status = "send_error (not_confirmed)"
                                    logger.warning(f"Message send confirmation FAILED for {log_prefix}: Sent message not found in recent history within {confirmation_window}.")
                                    overall_success = False # Mark overall as failed if confirmation fails
                            else:
                                # Fetch failed, cannot confirm
                                message_status = "send_error (confirm_fetch_failed)"
                                logger.warning(f"Message send confirmation check FAILED for {log_prefix}: Could not fetch recent messages.")
                                overall_success = False # Mark overall as failed if confirmation fails
                    else:
                        # API call returned None
                        logger.error(f"API POST ({send_api_description}) for {log_prefix} failed (returned None or error).")
                        message_status = "send_error (post_failed)"
                        error_count += 1; send_or_type_success = False; overall_success = False

                else: # Unknown APP_MODE
                    logger.warning(f"Skipping message send for {log_prefix} due to unsupported APP_MODE: {config_instance.APP_MODE}")
                    skipped_count += 1
                    send_or_type_success = False


                # --- Update Database ---
                if db_update_needed:
                    try:
                        # Get the MessageType object from the database
                        msg_type_obj = db_session.query(MessageType).filter(MessageType.type_name == next_message_type_key).first()
                        if not msg_type_obj:
                            # This should ideally not happen if seeding is correct
                            raise ValueError(f"MessageType '{next_message_type_key}' not found in database for {log_prefix}")

                        current_time = datetime.now() # Use consistent timestamp for updates

                        # Create new MessageHistory record
                        new_history = MessageHistory(
                            people_id=person_id,
                            message_type_id=msg_type_obj.id,
                            message_text=message_text[:4999], # Truncate if necessary
                            status=message_status, # Use the (potentially updated) status
                            sent_at=current_time
                        )
                        db_session.add(new_history)

                        # Update or Create InboxStatus record
                        inbox_status_to_update = db_session.query(InboxStatus).filter(InboxStatus.people_id == person_id).first()

                        # Determine the conversation ID to use (newly created or existing)
                        effective_conv_id = new_conversation_id_from_api or conversation_id

                        if inbox_status_to_update:
                             # Update existing InboxStatus
                             inbox_status_to_update.my_role = RoleType.AUTHOR # We just sent a message
                             inbox_status_to_update.last_message = message_text[:97] + '...' if len(message_text) > 100 else message_text
                             inbox_status_to_update.last_message_timestamp = current_time
                             # Update conversation ID if it changed (e.g., first message in dry run/prod)
                             if effective_conv_id and inbox_status_to_update.conversation_id != effective_conv_id:
                                 inbox_status_to_update.conversation_id = effective_conv_id
                                 logger.debug(f"Updated InboxStatus Conv ID to {effective_conv_id} for {log_prefix}")
                             logger.debug(f"Updated InboxStatus for {log_prefix}: Role=AUTHOR")
                        elif effective_conv_id: # Only create if we have a valid conversation ID
                             # Create new InboxStatus
                             new_inbox_status = InboxStatus(
                                  people_id=person_id,
                                  conversation_id=effective_conv_id,
                                  my_role=RoleType.AUTHOR,
                                  last_message=message_text[:97] + '...' if len(message_text) > 100 else message_text,
                                  last_message_timestamp=current_time
                             )
                             db_session.add(new_inbox_status)
                             logger.debug(f"Created new InboxStatus for {log_prefix}: Role=AUTHOR, ConvID={effective_conv_id}")
                        else:
                             # Log warning if we couldn't update or create InboxStatus (e.g., no conv ID after failed API call)
                             logger.warning(f"Could not update/create InboxStatus for {log_prefix}: No effective conversation ID available (Status: {message_status}).")

                        # Increment sent count ONLY if message status indicates confirmed success or dry run typing
                        if message_status == "sent_confirmed" or message_status == "typed (dry_run)":
                             sent_count += 1
                        # Flush changes for this person before the loop delay
                        db_session.flush()
                        logger.info(f"Database changes staged for {log_prefix} (Status: {message_status}).")

                    except ValueError as val_err: # Catch specific error from MessageType check
                        logger.error(f"Database update error for {log_prefix}: {val_err}", exc_info=True)
                        if db_session.is_active: db_session.rollback() # Rollback this specific attempt
                        error_count += 1; overall_success = False; continue
                    except SQLAlchemyError as db_e: # Catch general DB errors
                        logger.error(f"Database error staging updates for {log_prefix}: {db_e}", exc_info=True)
                        if db_session.is_active: db_session.rollback() # Rollback this specific attempt
                        error_count += 1; overall_success = False; continue
                    except Exception as db_e_unexp: # Catch unexpected errors
                         logger.critical(f"Unexpected error staging DB updates for {log_prefix}: {db_e_unexp}", exc_info=True)
                         if db_session.is_active: db_session.rollback()
                         error_count += 1; overall_success = False; continue


                # --- Delay Between Messages ---
                # Get base delay from config based on mode
                delay_td = MESSAGE_INTERVALS.get(config_instance.APP_MODE, timedelta(seconds=5)) # Default 5s
                delay_seconds = delay_td.total_seconds()

                # Add randomness unless in dry run
                if config_instance.APP_MODE != "dry_run":
                    delay_seconds = random.uniform(delay_seconds * 0.8, delay_seconds * 1.2)

                logger.debug(f"Waiting {delay_seconds:.2f}s before next message...")
                time.sleep(delay_seconds)

            # --- End of loop ---
            # Commit happens automatically outside the loop via db_transn context manager if no exceptions occurred

    except Exception as outer_e:
        logger.critical(f"Critical error during message processing loop: {outer_e}", exc_info=True)
        overall_success = False
        # Rollback is handled by the context manager

    # --- Final Summary ---
    logger.info("--- Message Sending Summary ----")
    potential_matches_count = len(potential_matches)
    logger.info(f"  Potential Matches Found: {potential_matches_count}")
    logger.info(f"  Processed:             {processed_count}")
    logger.info(f"  Sent/Typed (DB Update):{sent_count}")
    logger.info(f"  Skipped (Policy/Rule): {skipped_count}")
    logger.info(f"  Errors (API/DB/etc.):  {error_count}")
    logger.info(f"  Overall Success:       {overall_success}")
    logger.info("---------------------------------")

    return overall_success
# End of send_messages_to_matches (API Based)


# --- Main function for standalone testing (Modified for API) ---
def main():
    """Main function for standalone testing of Action 8 (API version)."""
    # Use local import for standalone execution
    from logging_config import setup_logging # Assuming logging_config is correct

    # --- Setup Logging ---
    try:
        db_file_path = config_instance.DATABASE_FILE
        log_filename_only = db_file_path.with_suffix(".log").name
        # Ensure logger is configured only once
        global logger
        if 'logger' not in globals() or not isinstance(logger, logging.Logger) or not logger.hasHandlers():
            logger = setup_logging(log_file=log_filename_only, log_level="DEBUG") # Assign return value
        else:
            if logger and isinstance(logger, logging.Logger):
                logger.setLevel(logging.DEBUG) # Ensure level is DEBUG for testing
            else:
                # Fallback if logger exists but is not a Logger object
                logger = logging.getLogger("logger")
                logger.setLevel(logging.DEBUG)
        logger.info(f"--- Starting Action 8 (API Messaging) Standalone Test ---")
        logger.info(f"APP_MODE: {config_instance.APP_MODE}")

    except Exception as log_setup_e:
        import sys; import logging as pylogging # Fallback imports
        print(f"CRITICAL: Error during logging setup: {log_setup_e}", file=sys.stderr)
        pylogging.basicConfig(level=pylogging.DEBUG);
        logger = pylogging.getLogger("Action8Fallback")
        logger.info(f"--- Starting Action 8 (API Messaging) Standalone Test (Logging Fallback) ---")
        logger.error(f"Initial logging setup failed: {log_setup_e}", exc_info=True)

    # --- Create a SessionManager ---
    session_manager = SessionManager()
    action_success = False

    try:
        # Start the browser session (needed for UBE header, login handled)
        logger.info("Attempting to start session (needed for headers)...")
        start_ok, _ = session_manager.start_sess(action_name="Action 8 API Test")
        if start_ok: # Driver presence implicitly checked by start_ok=True
            logger.info("Session started successfully.")

            # Call the main action function (now uses API)
            action_success = send_messages_to_matches(session_manager)

            if action_success:
                logger.info("send_messages_to_matches (API version) completed successfully.")
            else:
                logger.error("send_messages_to_matches (API version) reported errors or failed.")
        else:
             logger.critical("Failed to start session. Cannot run messaging action.")
             action_success = False

    except Exception as e:
        logger.critical(f"Critical error in Action 8 standalone main: {e}", exc_info=True)
        action_success = False
    finally:
        logger.info("Closing session manager...")
        if session_manager:
            session_manager.close_sess() # Clean up session
        logger.info(f"--- Action 8 Standalone Test Finished (Success: {action_success}) ---")

# --- Run main() if executed directly ---
if __name__ == "__main__":
    main()