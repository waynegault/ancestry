#!/usr/bin/env python3

# File: action8_messaging.py

#####################################################
# Imports
#####################################################

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
# >>> Removed _fetch_conversation_messages from utils import as it's no longer used here <<<
from utils import (_api_req, DynamicRateLimiter, SessionManager, retry,
                 time_wait, make_ube, format_name)
from cache import cache_result

#####################################################
# Initialise
#####################################################

# Initialize logging
logger = logging.getLogger("logger")

# app mode
logger.info(f"APP_MODE is: {config_instance.APP_MODE}")

# Define message intervals based on app mode
MESSAGE_INTERVALS = {
    "testing": timedelta(seconds=5),
    "production": timedelta(weeks=8),
    "dry_run": timedelta(seconds=5) }
MIN_MESSAGE_INTERVAL = MESSAGE_INTERVALS.get(config_instance.APP_MODE, timedelta(weeks=6)) # Default to production
logger.info(f"Using minimum message interval: {MIN_MESSAGE_INTERVAL}")

#####################################################
#Templates
#####################################################

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

#####################################################
# Which message?
#####################################################

def determine_next_message_type(
    last_message_details: Optional[Tuple[str, datetime, str]],
    is_in_family_tree: bool,
    last_received_timestamp: Optional[datetime],
) -> Optional[str]:
    """
    Determines the next message type based on history and inbox status.
    (Logic remains largely the same as before, but added logging)
    """
    # --- Idea 3: Added Logging ---
    logger.debug(f"Determining next msg type:")
    logger.debug(f"  - Last Sent Details (Type, SentAt, Status): {last_message_details}")
    logger.debug(f"  - Is Currently In Tree: {is_in_family_tree}")
    logger.debug(f"  - Last Received Timestamp: {last_received_timestamp}")
    # --- End Idea 3 ---

    # --- Condition 1: No message ever sent by script ---
    if not last_message_details:
        if last_received_timestamp:
            logger.debug("Result: Skip (No prior script message, but a message has been received).")
            return None
        next_type = "In_Tree-Initial" if is_in_family_tree else "Out_Tree-Initial"
        logger.debug(f"Result: {next_type} (No prior script message, none received).")
        return next_type

    # --- Extract details of the last sent message ---
    last_message_type, last_sent_at, last_message_status = last_message_details

    # --- Timezone Handling/Validation ---
    # Ensure last_sent_at is offset-aware UTC for comparison
    if isinstance(last_sent_at, datetime):
        if last_sent_at.tzinfo is None:
             logger.warning(f"Assuming naive last_sent_at ({last_sent_at}) from DB is UTC.")
             last_sent_at = last_sent_at.replace(tzinfo=timezone.utc)
        elif last_sent_at.tzinfo != timezone.utc:
             logger.warning(f"Converting last_sent_at from {last_sent_at.tzinfo} to UTC.")
             last_sent_at = last_sent_at.astimezone(timezone.utc)
        # else: Already aware UTC
    else:
        logger.error(f"Invalid last_sent_at type ({type(last_sent_at)}), cannot reliably determine next step.")
        return None

    # --- Condition 2: Check for reply ---
    if last_received_timestamp and isinstance(last_received_timestamp, datetime):
        # Ensure last_received is offset-aware UTC for comparison
        if last_received_timestamp.tzinfo is None:
             logger.warning(f"Assuming naive last_received_timestamp ({last_received_timestamp}) from DB is UTC.")
             last_received_timestamp_utc = last_received_timestamp.replace(tzinfo=timezone.utc)
        elif last_received_timestamp.tzinfo != timezone.utc:
             logger.warning(f"Converting last_received_timestamp from {last_received_timestamp.tzinfo} to UTC.")
             last_received_timestamp_utc = last_received_timestamp.astimezone(timezone.utc)
        else: # Already aware UTC
             last_received_timestamp_utc = last_received_timestamp

        # Compare aware UTC timestamps
        if last_received_timestamp_utc > last_sent_at:
            logger.debug(f"Result: Skip (Reply received at {last_received_timestamp_utc} after last script message sent at {last_sent_at}).")
            return None

    # --- Condition 3: Check message interval ---
    now_utc = datetime.now(timezone.utc)
    time_since_last_message = now_utc - last_sent_at

    if time_since_last_message < MIN_MESSAGE_INTERVAL:
        logger.debug(f"Result: Skip (Interval not met: {time_since_last_message} < {MIN_MESSAGE_INTERVAL}).")
        return None

    # --- Condition 4: Determine next step based on last type and current tree status ---
    next_type: Optional[str] = None # Default to None
    if is_in_family_tree:
        if last_message_type.startswith("Out_Tree"):
            next_type = "In_Tree-Initial_for_was_Out_Tree"
            logger.debug("Context: Match was Out_Tree, now In_Tree.")
        elif last_message_type == "In_Tree-Initial" or last_message_type == "In_Tree-Initial_for_was_Out_Tree":
            next_type = "In_Tree-Follow_Up"
            logger.debug(f"Context: Following up on {last_message_type}.")
        elif last_message_type == "In_Tree-Follow_Up":
            next_type = "In_Tree-Final_Reminder"
            logger.debug("Context: Sending final In_Tree reminder.")
        # else: stays None (end of In_Tree sequence)
    else: # Match is Out_Tree
        if last_message_type.startswith("In_Tree"):
            logger.warning(f"Context: Match was In_Tree ({last_message_type}) but is now Out_Tree. Skipping.")
            next_type = None
        elif last_message_type == "Out_Tree-Initial":
            next_type = "Out_Tree-Follow_Up"
            logger.debug("Context: Following up on Out_Tree-Initial.")
        elif last_message_type == "Out_Tree-Follow_Up":
            next_type = "Out_Tree-Final_Reminder"
            logger.debug("Context: Sending final Out_Tree reminder.")
        # else: stays None (end of Out_Tree sequence)

    if next_type:
        logger.debug(f"Result: {next_type} (Following sequence).")
    else:
        logger.debug("Result: None (End of sequence or skip condition met).")

    return next_type
# End of determine_next_message_type

#####################################################
# Send messages
#####################################################

def send_messages_to_matches(session_manager: SessionManager) -> bool:
    """Sends messages to DNA matches based on criteria, using API calls."""
    logger.info("Starting message sending process (API Based - Dual Endpoint)...")

    # --- Prerequisites ---
    if not session_manager: logger.error("SessionManager instance is required."); return False
    if not session_manager._verify_api_login_status():
         logger.error("API login verification failed. Cannot send messages.")
         return False
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
        with session_manager.get_db_conn_context() as db_session:
            if not db_session:
                logger.error("Failed to get database connection. Cannot proceed.")
                return False

            potential_matches_query = (
                db_session.query(
                    DnaMatch.people_id, DnaMatch.predicted_relationship,
                    Person.username, Person.profile_id, Person.in_my_tree, Person.first_name,
                    FamilyTree.actual_relationship, FamilyTree.person_name_in_tree, FamilyTree.relationship_path
                )
                .join(Person, DnaMatch.people_id == Person.id)
                .outerjoin(FamilyTree, Person.id == FamilyTree.people_id)
                .filter(Person.profile_id.isnot(None), Person.profile_id != "", Person.profile_id != "UNKNOWN")
                .filter(Person.status == "active")
                .filter(Person.contactable == True)
                .order_by(Person.id)
            )
            potential_matches = potential_matches_query.all()

            logger.info(f"Found {len(potential_matches)} potential CONTACTABLE DNA matches to check.")
            if not potential_matches: return True

            total_family_tree_rows = db_session.query(func.count(FamilyTree.id)).scalar() or 0
            logger.info(f"Total rows found in family_tree table: {total_family_tree_rows}")

            for match_data in potential_matches:
                # --- Initialize variables at the start of the loop iteration ---
                person_id: Optional[int] = None
                predicted_relationship: Optional[str] = None
                username: Optional[str] = None
                profile_id: Optional[str] = None
                is_in_family_tree: bool = False
                first_name: Optional[str] = None
                actual_relationship: Optional[str] = None
                person_name_in_tree: Optional[str] = None
                relationship_path: Optional[str] = None
                last_message_sent_details: Optional[Tuple[str, datetime, str]] = None
                last_received_timestamp: Optional[datetime] = None
                conversation_id: Optional[str] = None
                they_sent_last: bool = False
                inbox_role: Optional[RoleType] = None
                # --- End Initialization ---

                if max_to_send != 0 and sent_count >= max_to_send:
                    logger.info(f"Reached MAX_INBOX limit ({max_to_send}). Stopping message sending.")
                    break
                processed_count += 1

                try:
                    # Unpack data
                    ( person_id, predicted_relationship, username, profile_id, is_in_family_tree, first_name,
                        actual_relationship, person_name_in_tree, relationship_path ) = match_data
                    if not person_id or not profile_id or not username:
                         raise ValueError("Missing essential person data (id, profile_id, username)")
                except (ValueError, TypeError) as unpack_e:
                    logger.error(f"Error unpacking or validating match data: {unpack_e}. Row: {match_data}", exc_info=True)
                    error_count += 1; overall_success = False; continue

                # Assertions to satisfy type checkers after unpacking check
                assert person_id is not None
                assert profile_id is not None
                assert username is not None

                recipient_profile_id_upper = profile_id.upper()
                log_prefix = f"{username} (ID: {person_id}, Profile: {recipient_profile_id_upper})"
                logger.debug(f"\n--- Processing {log_prefix} ---")

                # --- Process this match within a single try block after unpacking ---
                try:
                    # --- DB Queries ---
                    last_sent_query = (
                        db_session.query(MessageHistory.sent_at, MessageType.type_name, MessageHistory.status)
                        .join(MessageType)
                        .filter(MessageHistory.people_id == person_id)
                        .order_by(desc(MessageHistory.sent_at))
                        .first()
                    )
                    if last_sent_query:
                        sent_at_db = last_sent_query.sent_at
                        if sent_at_db.tzinfo is None: sent_at_db = sent_at_db.replace(tzinfo=timezone.utc)
                        elif sent_at_db.tzinfo != timezone.utc: sent_at_db = sent_at_db.astimezone(timezone.utc)
                        last_message_sent_details = (last_sent_query.type_name, sent_at_db, last_sent_query.status)
                        logger.debug(f"Last sent details: Type={last_sent_query.type_name}, At={sent_at_db}, Status={last_sent_query.status}")
                    else:
                        logger.debug("No previous message sent found in history.")

                    inbox_status = db_session.query(InboxStatus).filter(InboxStatus.people_id == person_id).first()
                    if inbox_status:
                         logger.debug(f"InboxStatus found: Role={inbox_status.my_role}, LastMsgAt={inbox_status.last_message_timestamp}, ConvID={inbox_status.conversation_id}")
                         conversation_id = inbox_status.conversation_id
                         inbox_role = inbox_status.my_role
                         if inbox_status.last_message_timestamp and inbox_status.last_message_timestamp > datetime.now():
                              logger.warning(f"InboxStatus for {log_prefix} has a timestamp in the future: {inbox_status.last_message_timestamp}. Using it anyway.")
                         if inbox_status.my_role == RoleType.RECIPIENT and inbox_status.last_message_timestamp:
                              received_at_db = inbox_status.last_message_timestamp
                              if received_at_db.tzinfo is None: received_at_db = received_at_db.replace(tzinfo=timezone.utc)
                              elif received_at_db.tzinfo != timezone.utc: received_at_db = received_at_db.astimezone(timezone.utc)
                              last_received_timestamp = received_at_db # Assigned here
                              they_sent_last = True
                              logger.debug(f"They sent last confirmed. Last received at: {last_received_timestamp}")
                    else:
                         logger.debug("No InboxStatus found.")

                    # DB Consistency Check
                    if not last_message_sent_details and inbox_role == RoleType.AUTHOR:
                         logger.warning(f"DB Consistency Check: MessageHistory is empty for {log_prefix}, but InboxStatus.my_role is AUTHOR.")

                    # --- Reply Check ---
                    if they_sent_last and last_received_timestamp:
                         should_skip_reply = False
                         if last_message_sent_details:
                             lm_type, lm_sent_at_utc, lm_status = last_message_sent_details
                             if last_received_timestamp > lm_sent_at_utc:
                                  should_skip_reply = True
                             # else: our last sent was later/equal, don't skip
                         else:
                               should_skip_reply = True # They sent first

                         if should_skip_reply:
                              logger.info(f"Skipping {log_prefix}: Reply Check indicates no action needed.")
                              skipped_count += 1
                              continue # Go to the next iteration of the main FOR loop

                    # --- Determine Next Message ---
                    logger.debug(f"Calling determine_next_message_type with last_received_timestamp: {last_received_timestamp} (Type: {type(last_received_timestamp)})")
                    next_message_type_key = determine_next_message_type(last_message_sent_details, bool(is_in_family_tree), last_received_timestamp)
                    if not next_message_type_key:
                        logger.info(f"Skipping {log_prefix}: No appropriate next message type determined (rule/timing).")
                        skipped_count += 1
                        continue # Go to the next iteration of the main FOR loop

                    # --- Format Message ---
                    message_template = MESSAGE_TEMPLATES.get(next_message_type_key)
                    if not message_template:
                        logger.error(f"Missing message template for key '{next_message_type_key}' for {log_prefix}.")
                        error_count += 1; overall_success = False; continue

                    name_to_use = "Valued Relative"; source_of_name="Fallback"
                    if is_in_family_tree and person_name_in_tree:
                        name_to_use = person_name_in_tree; source_of_name="FamilyTree"
                    elif first_name:
                        name_to_use = first_name; source_of_name="Person.first_name"
                    elif username and username != "Unknown":
                        name_to_use = username; source_of_name="Person.username"
                    formatted_name = format_name(name_to_use)
                    logger.debug(f"Using name '{formatted_name}' (Source: {source_of_name}) for formatting.")

                    format_data = {
                        "name": formatted_name,
                        "predicted_relationship": predicted_relationship or "N/A",
                        "actual_relationship": actual_relationship or "N/A",
                        "relationship_path": relationship_path or "N/A",
                        "total_rows": total_family_tree_rows
                    }

                    try:
                        required_keys_in_template = set(re.findall(r'\{(\w+)\}', message_template))
                    except Exception as regex_e:
                        logger.error(f"Regex error checking template keys for '{next_message_type_key}': {regex_e}")
                        required_keys_in_template = set()

                    missing_keys = [key for key in required_keys_in_template if key not in format_data or format_data[key] is None]
                    missing_keys = [key for key in missing_keys if format_data.get(key) != "N/A"]

                    if missing_keys:
                        logger.warning(f"Skipping {log_prefix}: Missing/None required keys {missing_keys} in format data for template '{next_message_type_key}'. Available: {list(format_data.keys())}")
                        error_count += 1; overall_success = False; continue

                    try:
                        message_text = message_template.format(**format_data)
                    except KeyError as e:
                         logger.error(f"KeyError formatting message for {log_prefix} (Template: '{next_message_type_key}'): Missing key {e}. Available: {list(format_data.keys())}", exc_info=False)
                         error_count += 1; overall_success = False; continue
                    except Exception as e:
                        logger.error(f"Error formatting message for {log_prefix} using template '{next_message_type_key}': {e}", exc_info=True)
                        error_count += 1; overall_success = False; continue

                    logger.debug(f"Formatted message for {log_prefix} ({next_message_type_key}): '{message_text[:100]}...'")

                    # --- API Call / Dry Run ---
                    message_status: str = "skipped (logic_error)"
                    new_conversation_id_from_api_local: Optional[str] = None # Use local var inside try
                    is_initial_message: bool = not conversation_id
                    logger.debug(f"Determining API action: conversation_id='{conversation_id}', is_initial_message={is_initial_message}")
                    send_api_url: str = ""
                    payload: Dict[str, Any] = {}
                    send_api_description: str = ""

                    if is_initial_message:
                        send_api_url = urljoin(config_instance.BASE_URL.rstrip('/')+'/', "app-api/express/v2/conversations/message")
                        send_api_description = "Create Conversation API"
                        payload = {
                            "content": message_text, "author": MY_PROFILE_ID_LOWER, "index": 0, "created": 0,
                            "conversation_members": [
                                {"user_id": recipient_profile_id_upper.lower(), "family_circles": [] },
                                {"user_id": MY_PROFILE_ID_LOWER}
                            ]
                        }
                    elif conversation_id:
                        send_api_url = urljoin(config_instance.BASE_URL.rstrip('/')+'/', f"app-api/express/v2/conversations/{conversation_id}")
                        send_api_description = "Send Message API (Existing Conv)"
                        payload = {"content": message_text, "author": MY_PROFILE_ID_LOWER}
                    else:
                        # This path should technically not be reachable if is_initial_message logic is correct
                        logger.error(f"Logic Error: Cannot determine send API URL/payload for {log_prefix}. is_initial={is_initial_message}, conv_id={conversation_id}")
                        error_count += 1; overall_success = False; continue

                    send_api_headers = {}

                    if config_instance.APP_MODE == "dry_run":
                        log_action = "create conversation and send" if is_initial_message else "send message to existing conversation"
                        logger.info(f"[DRY RUN] Would {log_action} '{next_message_type_key}' to {log_prefix} with name '{formatted_name}'. Message: '{message_text[:100]}...'")
                        message_status = "typed (dry_run)"
                        if is_initial_message:
                            new_conversation_id_from_api_local = f"dryrun_{uuid.uuid4()}"

                    elif config_instance.APP_MODE in ["production", "testing"]:
                        log_action = "create conversation and send" if is_initial_message else "send follow-up"
                        logger.info(f"Attempting to {log_action} '{next_message_type_key}' to {log_prefix} via API...")
                        send_response_data: Optional[Any] = None
                        try:
                            send_response_data = _api_req(
                                url=send_api_url, driver=session_manager.driver, session_manager=session_manager,
                                method="POST", json_data=payload, use_csrf_token=False, headers=send_api_headers,
                                api_description=send_api_description, force_requests=True
                            )
                        except Exception as api_e:
                            logger.error(f"Exception during _api_req ({send_api_description}) for {log_prefix}: {api_e}", exc_info=True)
                            message_status = "send_error (api_exception)"
                            error_count += 1; overall_success = False; continue

                        if send_response_data is not None:
                            post_status_ok = False
                            response_author_match = False
                            if is_initial_message and isinstance(send_response_data, dict) and 'conversation_id' in send_response_data:
                                 new_conv_id = str(send_response_data.get('conversation_id'))
                                 msg_details = send_response_data.get('message', {})
                                 resp_author = str(msg_details.get('author', '')).upper() if isinstance(msg_details, dict) else ''
                                 if new_conv_id and resp_author == MY_PROFILE_ID_UPPER:
                                     post_status_ok = True
                                     response_author_match = True
                                     conversation_id = new_conv_id # Update conversation_id for this iteration
                                     new_conversation_id_from_api_local = new_conv_id # Store the newly created ID
                                     logger.info(f"API POST successful (initial): ConvID {conversation_id} created, author matched.")
                                 else:
                                     logger.error(f"API created conversation, but response invalid. ConvID: '{new_conv_id}', Author: '{resp_author}' (Exp: '{MY_PROFILE_ID_UPPER}')")
                            elif not is_initial_message and isinstance(send_response_data, dict) and 'index' in send_response_data and 'author' in send_response_data:
                                 resp_author = str(send_response_data.get('author', '')).upper()
                                 if resp_author == MY_PROFILE_ID_UPPER:
                                     post_status_ok = True
                                     response_author_match = True
                                     logger.info(f"API POST successful (follow-up): Author matched.")
                                     resp_content = send_response_data.get('content')
                                     if resp_content != message_text:
                                          logger.warning(f"API response content mismatch (ignoring). Sent: '{message_text[:50]}...', Rcvd: '{str(resp_content)[:50]}...'")
                                 else:
                                     logger.error(f"API sent follow-up, but author validation failed. Author: '{resp_author}' (Exp: '{MY_PROFILE_ID_UPPER}')")
                            else:
                                logger.error(f"API call ({send_api_description}) for {log_prefix} returned unexpected response format after POST. Type: {type(send_response_data)}")
                                logger.debug(f"Full response data: {send_response_data}")

                            if post_status_ok and response_author_match:
                                message_status = "delivered OK"
                                logger.info(f"Message send to {log_prefix} ACCEPTED by API.")
                            else:
                                message_status = "send_error (post_validation_failed)"
                                error_count += 1
                                overall_success = False
                                logger.warning(f"API POST validation failed for {log_prefix}. Message status set to: {message_status}")

                        else:
                            logger.error(f"API POST ({send_api_description}) for {log_prefix} failed (returned None or error).")
                            message_status = "send_error (post_failed)"
                            error_count += 1; overall_success = False

                    else: # Unknown APP_MODE
                        logger.warning(f"Skipping message send for {log_prefix} due to unsupported APP_MODE: {config_instance.APP_MODE}")
                        skipped_count += 1

                    # --- Database Update Logic ---
                    if message_status == "delivered OK" or message_status == "typed (dry_run)":
                        try:
                            msg_type_obj = db_session.query(MessageType).filter(MessageType.type_name == next_message_type_key).first()
                            if not msg_type_obj:
                                raise ValueError(f"MessageType '{next_message_type_key}' not found in database for {log_prefix}")

                            current_time_utc = datetime.now(timezone.utc)
                            current_time_utc_truncated = current_time_utc.replace(second=0, microsecond=0)
                            logger.debug(f"Using truncated timestamp for DB: {current_time_utc_truncated}")

                            truncated_message_history = message_text[:config_instance.MESSAGE_TRUNCATION_LENGTH]
                            new_history = MessageHistory(
                                people_id=person_id,
                                message_type_id=msg_type_obj.id,
                                message_text=truncated_message_history,
                                status=message_status,
                                sent_at=current_time_utc_truncated
                            )
                            db_session.add(new_history)
                            logger.debug(f"Added MessageHistory record for {log_prefix} (Status: {message_status}, Length: {len(truncated_message_history)}).")

                            if message_status == "delivered OK":
                                inbox_status_to_update = db_session.query(InboxStatus).filter(InboxStatus.people_id == person_id).first()
                                effective_conv_id_for_db = new_conversation_id_from_api_local or conversation_id # Use local var

                                if not effective_conv_id_for_db:
                                    logger.error(f"DB Update Error for {log_prefix}: Cannot update InboxStatus, effective conversation ID is missing even though status is {message_status}.")
                                    raise ValueError("Cannot update InboxStatus without a valid conversation ID")

                                truncated_message_inbox = message_text[:config_instance.MESSAGE_TRUNCATION_LENGTH]

                                if inbox_status_to_update:
                                    inbox_status_to_update.my_role = RoleType.AUTHOR
                                    inbox_status_to_update.last_message = truncated_message_inbox
                                    inbox_status_to_update.last_message_timestamp = current_time_utc_truncated
                                    if inbox_status_to_update.conversation_id != effective_conv_id_for_db:
                                        inbox_status_to_update.conversation_id = effective_conv_id_for_db
                                        logger.debug(f"Updated InboxStatus Conv ID to {effective_conv_id_for_db} for {log_prefix}")
                                    logger.debug(f"Updated InboxStatus for {log_prefix}: Role=AUTHOR, Length: {len(truncated_message_inbox)}")
                                else:
                                    new_inbox_status = InboxStatus(
                                        people_id=person_id,
                                        conversation_id=effective_conv_id_for_db,
                                        my_role=RoleType.AUTHOR,
                                        last_message=truncated_message_inbox,
                                        last_message_timestamp=current_time_utc_truncated
                                    )
                                    db_session.add(new_inbox_status)
                                    logger.debug(f"Created new InboxStatus for {log_prefix}: Role=AUTHOR, ConvID={effective_conv_id_for_db}, Length: {len(truncated_message_inbox)}")

                            if message_status == "delivered OK":
                                 sent_count += 1

                            db_session.flush()
                            logger.info(f"Database changes staged for {log_prefix} (Status: {message_status}).")

                        except ValueError as val_err:
                            logger.error(f"Database update error for {log_prefix}: {val_err}", exc_info=True)
                            if db_session.is_active: db_session.rollback() # Rollback should be safe here
                            error_count += 1; overall_success = False; continue
                        except SQLAlchemyError as db_e:
                            logger.error(f"Database error staging updates for {log_prefix}: {db_e}", exc_info=True)
                            if db_session.is_active: db_session.rollback()
                            error_count += 1; overall_success = False; continue
                        except Exception as db_e_unexp:
                             logger.critical(f"Unexpected error staging DB updates for {log_prefix}: {db_e_unexp}", exc_info=True)
                             if db_session.is_active: db_session.rollback()
                             error_count += 1; overall_success = False; continue

                    # --- Delay Between Messages ---
                    delay_td = MESSAGE_INTERVALS.get(config_instance.APP_MODE, timedelta(seconds=5))
                    delay_seconds = delay_td.total_seconds()
                    if config_instance.APP_MODE != "dry_run":
                        delay_seconds = random.uniform(delay_seconds * 0.8, delay_seconds * 1.2)
                    logger.debug(f"Waiting {delay_seconds:.2f}s before next message...")
                    time.sleep(delay_seconds)

                # --- End of the inner try block for a single match's processing ---
                except Exception as inner_loop_e:
                    # Catch any unexpected error during the processing of a single match
                    # This includes errors *after* the DB query block
                    logger.error(f"Unexpected error processing match {log_prefix}: {inner_loop_e}", exc_info=True)
                    error_count += 1
                    overall_success = False
                    # Continue to the next match
                    continue

            # --- End of main FOR loop ---

    except Exception as outer_e:
        logger.critical(f"Critical error during message processing loop setup or context manager: {outer_e}", exc_info=True)
        overall_success = False # Ensure overall success is False on outer exception

    # --- Final Summary ---
    logger.info("--- Message Sending Summary ----")
    potential_matches_count = len(potential_matches) if potential_matches else 0
    logger.info(f"  Potential Matches Found: {potential_matches_count}")
    logger.info(f"  Processed:             {processed_count}")
    logger.info(f"  Sent (API Accepted):   {sent_count}")
    logger.info(f"  Skipped (Policy/Rule): {skipped_count}")
    logger.info(f"  Errors (API/DB/etc.):  {error_count}")
    logger.info(f"  Overall Success:       {overall_success}")
    logger.info("---------------------------------")

    return overall_success
# End of send_messages_to_matches 

#####################################################
# Stand alone testing
#####################################################

def main():
    """Main function for standalone testing of Action 8 (API version)."""
    # Use local import for standalone execution
    from logging_config import setup_logging # Assuming logging_config is correct

    # --- Setup Logging ---
    try:
        # Ensure config_instance is available
        from config import config_instance
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
            logger.info("Session started successfully. Proceeding to call send_messages_to_matches...") # Added detail

            # Call the main action function (now uses API)
            action_success = send_messages_to_matches(session_manager) # <<< CALL HAPPENS HERE <<<

            # Log result immediately after call returns
            if action_success:
                logger.info("send_messages_to_matches (API version) completed successfully.")
            else:
                # Use ERROR level if the function explicitly returns False
                logger.error("send_messages_to_matches (API version) reported errors or failed (returned False).")
        else:
             logger.critical("Failed to start session. Cannot run messaging action.")
             action_success = False # Ensure action_success is False if session start fails

    except Exception as e:
        # Log critical errors occurring outside the action function but within main try block
        logger.critical(f"Critical error in Action 8 standalone main execution: {e}", exc_info=True)
        action_success = False # Ensure action_success is False on exception
    finally:
        logger.info("Closing session manager...")
        if session_manager:
            session_manager.close_sess() # Clean up session
        logger.info(f"--- Action 8 Standalone Test Finished (Overall Success: {action_success}) ---") # Log final status
# end main

# --- Run main() if executed directly ---
if __name__ == "__main__":
    main()