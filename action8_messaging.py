# REVISED action8_messaging.py - FIXED NameError

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

# REMOVED timezone import as we'll use naive local time now
from datetime import datetime, timedelta  # , timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from urllib.parse import urljoin, urlparse, urlencode

# Third-party imports
import requests  # Keep requests if used by _api_req indirectly

# --- V14.1 FIX: Explicitly import 'select' from sqlalchemy.sql ---
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Integer,
    String,
    desc,
    func,
    Row,
    Subquery,
    over,
)  # Removed select from here
from sqlalchemy.sql import select  # Added explicit import here

# --- END FIX ---
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (
    Session as DbSession,
    joinedload,
    aliased,
)  # Use DbSession alias, added aliased

# Local application imports
from config import (
    config_instance,
    selenium_config,
)  # Keep selenium_config for API timeouts maybe?
from database import (
    InboxStatus,
    Person,
    RoleType,
    db_transn,
    DnaMatch,
    FamilyTree,
    MessageHistory,
    MessageType,
)
from utils import (
    _api_req,
    DynamicRateLimiter,
    SessionManager,
    retry,
    time_wait,
    make_ube,
    format_name,
)
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
    "dry_run": timedelta(seconds=5),
}
MIN_MESSAGE_INTERVAL = MESSAGE_INTERVALS.get(
    config_instance.APP_MODE, timedelta(weeks=8)
)  # Use production as default fallback
logger.info(f"Using minimum message interval: {MIN_MESSAGE_INTERVAL}")

#####################################################
# Templates
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
        logger.critical(
            f"CRITICAL: messages.json not found at {messages_path.resolve()}"
        )
        return {}
    try:
        with messages_path.open("r", encoding="utf-8") as f:
            templates = json.load(f)
            if not isinstance(templates, dict) or not all(
                isinstance(v, str) for v in templates.values()
            ):
                logger.critical(
                    f"CRITICAL: messages.json does not contain a dictionary of strings."
                )
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

MESSAGE_TEMPLATES: Dict[str, str] = cast(
    Dict[str, str], load_message_templates()
)  # Explicitly cast the return type
if not MESSAGE_TEMPLATES:
    logger.error(
        "Message templates failed to load. Messaging functionality will be limited."
    )

#####################################################
# Which message?
#####################################################


def determine_next_message_type(
    last_message_details: Optional[Tuple[str, datetime, str]], is_in_family_tree: bool
) -> Optional[str]:
    """
    V2 REVISED: Determines the next message type based ONLY on history and current tree status.
    Assumes caller has already checked for replies and message interval.
    Adds specific logging for why a message is skipped.
    """
    logger.debug(f"Currently In Tree: {is_in_family_tree}")

    if not last_message_details:
        next_type = "In_Tree-Initial" if is_in_family_tree else "Out_Tree-Initial"
        logger.debug(f"Result: {next_type} (Reason: No prior script message).")
        return next_type

    last_message_type, last_sent_at, last_message_status = last_message_details
    next_type: Optional[str] = None
    skip_reason: str = "End of sequence or other condition met"  # Default skip reason

    if is_in_family_tree:
        if last_message_type.startswith("Out_Tree"):
            next_type = "In_Tree-Initial_for_was_Out_Tree"
            logger.debug("Match was Out_Tree, now In_Tree.")
        elif (
            last_message_type == "In_Tree-Initial"
            or last_message_type == "In_Tree-Initial_for_was_Out_Tree"
        ):
            next_type = "In_Tree-Follow_Up"
            logger.debug(f"Following up on {last_message_type}.")
        elif last_message_type == "In_Tree-Follow_Up":
            next_type = "In_Tree-Final_Reminder"
            logger.debug("Sending final In_Tree reminder.")
        elif last_message_type == "In_Tree-Final_Reminder":
            skip_reason = f"End of In_Tree sequence (last was {last_message_type})"
            logger.debug(f"{skip_reason}.")
        else:
            skip_reason = (
                f"Unexpected previous In_Tree message type: {last_message_type}"
            )
            logger.warning(f"{skip_reason}.")
    else:  # Match is Out_Tree
        if last_message_type.startswith("In_Tree"):
            skip_reason = f"Match was In_Tree ({last_message_type}) but is now Out_Tree"
            logger.warning(f"{skip_reason}. Skipping.")
        elif last_message_type == "Out_Tree-Initial":
            next_type = "Out_Tree-Follow_Up"
            logger.debug("Following up on Out_Tree-Initial.")
        elif last_message_type == "Out_Tree-Follow_Up":
            next_type = "Out_Tree-Final_Reminder"
            logger.debug("Sending final Out_Tree reminder.")
        elif last_message_type == "Out_Tree-Final_Reminder":
            skip_reason = f"End of Out_Tree sequence (last was {last_message_type})"
            logger.debug(f"{skip_reason}.")
        else:
            skip_reason = (
                f"Unexpected previous Out_Tree message type: {last_message_type}"
            )
            logger.warning(f"{skip_reason}.")

    if next_type:
        logger.debug(f"Sending {next_type} (Reason: Following sequence).")
    else:
        logger.debug(f"Skipping: {skip_reason}.")

    return next_type
# End of determine_next_message_type

#####################################################
# Send messages
#####################################################


def send_messages_to_matches(session_manager: SessionManager) -> bool:
    """
    V14.1 REVISED: Sends messages to DNA matches based on criteria, using API calls.
    - Optimizes performance by pre-fetching MessageHistory and InboxStatus data.
    - Stores naive LOCAL time.
    - Fixes NameError for 'select'.
    - Batches DB commits.
    """

    # --- Prerequisites (same as before) ---
    if not session_manager: logger.error("SessionManager instance is required."); return False
    if not session_manager._verify_api_login_status(): logger.error("API login verification failed."); return False
    if not session_manager.driver: logger.warning("Driver not available, UBE header missing.")
    if not session_manager.my_profile_id: logger.error("Own profile ID missing."); return False
    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()
    MY_PROFILE_ID_UPPER = session_manager.my_profile_id.upper()
    if not MESSAGE_TEMPLATES: logger.error("Message templates failed to load."); return False

    overall_success = True
    processed_count = 0
    sent_count = 0
    skipped_count = 0
    error_count = 0
    potential_matches = []
    max_to_send = config_instance.MAX_INBOX
    db_commit_batch_size = 10 # Commit after every N successful sends
    db_objects_to_commit: List[Any] = [] # Accumulate objects for batch commit

    try:
        # Use context manager for the entire operation to ensure commit/rollback
        with session_manager.get_db_conn_context() as db_session:
            if not db_session:
                logger.error("Failed to get DB connection.")
                return False

            # --- Pre-fetching (Steps 1-4, same as V14.1) ---
            # 1. Fetch Potential Matches
            potential_matches_query = (
                db_session.query(
                    DnaMatch.people_id, DnaMatch.predicted_relationship, Person.username,
                    Person.profile_id, Person.in_my_tree, Person.first_name,
                    FamilyTree.actual_relationship, FamilyTree.person_name_in_tree, FamilyTree.relationship_path
                )
                .join(Person, DnaMatch.people_id == Person.id)
                .outerjoin(FamilyTree, Person.id == FamilyTree.people_id)
                .filter(Person.profile_id.isnot(None), Person.profile_id != "", Person.profile_id != "UNKNOWN")
                .filter(Person.status == "active").filter(Person.contactable == True)
                .order_by(Person.id)
            )
            potential_matches = potential_matches_query.all()
            logger.debug(f"Found {len(potential_matches)} potential CONTACTABLE DNA matches to check.")
            if not potential_matches: return True
            potential_person_ids = [match.people_id for match in potential_matches if match.people_id]
            if not potential_person_ids: logger.info("No valid people_ids found."); return True

            # 2. Pre-fetch Message History
            logger.debug(f"Pre-fetching latest MessageHistory for {len(potential_person_ids)} people...")
            latest_history_data: Dict[int, Tuple[str, datetime, str]] = {}
            try:
                row_number_subquery: Subquery = (
                    select(
                        MessageHistory.people_id, MessageHistory.sent_at, MessageType.type_name, MessageHistory.status,
                        func.row_number().over(partition_by=MessageHistory.people_id, order_by=MessageHistory.sent_at.desc()).label("rn")
                    ).join(MessageType, MessageHistory.message_type_id == MessageType.id)
                     .where(MessageHistory.people_id.in_(potential_person_ids)).subquery()
                )
                aliased_subquery = aliased(row_number_subquery, name="latest_msg")
                latest_history_query = select(
                    aliased_subquery.c.people_id, aliased_subquery.c.type_name,
                    aliased_subquery.c.sent_at, aliased_subquery.c.status
                ).where(aliased_subquery.c.rn == 1)
                latest_history_results = db_session.execute(latest_history_query).fetchall()
                for row in latest_history_results:
                    person_id, type_name, sent_at_db, status = row
                    if sent_at_db:
                        sent_at_local = sent_at_db.replace(tzinfo=None) if sent_at_db.tzinfo else sent_at_db
                        latest_history_data[person_id] = (type_name, sent_at_local, status)
                    else: logger.warning(f"Found MessageHistory for PersonID {person_id}, but sent_at is NULL.")
                logger.debug(f"Fetched latest MessageHistory data for {len(latest_history_data)} people.")
            except SQLAlchemyError as db_err: logger.error(f"DB error pre-fetching MessageHistory: {db_err}", exc_info=True); return False
            except Exception as e: logger.error(f"Error processing pre-fetched MessageHistory: {e}", exc_info=True); return False

            # 3. Pre-fetch Inbox Status
            logger.debug(f"Pre-fetching InboxStatus for {len(potential_person_ids)} people...")
            inbox_data: Dict[int, InboxStatus] = {}
            try:
                inbox_results = db_session.query(InboxStatus).filter(InboxStatus.people_id.in_(potential_person_ids)).all()
                inbox_data = {status.people_id: status for status in inbox_results}
                logger.debug(f"Fetched InboxStatus data for {len(inbox_data)} people.")
            except SQLAlchemyError as db_err: logger.error(f"DB error pre-fetching InboxStatus: {db_err}", exc_info=True); return False
            except Exception as e: logger.error(f"Error processing pre-fetched InboxStatus: {e}", exc_info=True); return False

            # 4. Fetch Total Family Tree Rows
            total_family_tree_rows = db_session.query(func.count(FamilyTree.id)).scalar() or 0
            logger.debug(f"Total rows found in family_tree table: {total_family_tree_rows}")
            # --- END Pre-fetching ---


            # --- 5. Main Loop ---
            logger.debug(f"Processing {len(potential_matches)} matches...\n")
            for match_data in potential_matches:
                # --- Initialize Loop Variables (same as before) ---
                person_id: Optional[int] = None; username: Optional[str] = None; profile_id: Optional[str] = None
                is_in_family_tree: bool = False; first_name: Optional[str] = None
                person_name_in_tree: Optional[str] = None; original_predicted_relationship: Optional[str] = None
                original_actual_relationship: Optional[str] = None; original_relationship_path: Optional[str] = None
                last_message_sent_details: Optional[Tuple[str, datetime, str]] = None
                last_sent_at_local: Optional[datetime] = None; last_received_at_local: Optional[datetime] = None
                conversation_id: Optional[str] = None; they_sent_last: bool = False
                inbox_role: Optional[RoleType] = None
                # Removed new_history_record = None

                if max_to_send != 0 and sent_count >= max_to_send:
                    logger.debug(f"Reached MAX_INBOX limit ({max_to_send}). Stopping.")
                    break
                processed_count += 1

                # --- Unpack Match Data (same as before) ---
                try:
                    (person_id, original_predicted_relationship, username, profile_id, is_in_family_tree, first_name,
                     original_actual_relationship, person_name_in_tree, original_relationship_path) = match_data
                    if not person_id or not profile_id or not username: raise ValueError("Missing essential person data")
                except (ValueError, TypeError) as unpack_e:
                    logger.error(f"Error unpacking data: {unpack_e}. Row: {match_data}", exc_info=True)
                    error_count += 1; overall_success = False; continue
                assert person_id is not None; assert profile_id is not None; assert username is not None
                recipient_profile_id_upper = profile_id.upper(); log_prefix = f"{username} #{person_id}"
                logger.debug(f"#### Processing {log_prefix} ####")

                # --- Process this match ---
                try:
                    now_local = datetime.now()

                    # --- Use Pre-fetched Data (Step 6, same as V14.1) ---
                    last_message_sent_details = latest_history_data.get(person_id)
                    if last_message_sent_details: last_sent_at_local = last_message_sent_details[1]
                    else: last_sent_at_local = None
                    inbox_status = inbox_data.get(person_id)
                    if inbox_status:
                        conversation_id = inbox_status.conversation_id; inbox_role = inbox_status.my_role
                        received_at_db = inbox_status.last_message_timestamp
                        if received_at_db:
                            last_received_at_local = received_at_db.replace(tzinfo=None) if received_at_db.tzinfo else received_at_db
                            if inbox_status.my_role == RoleType.RECIPIENT: they_sent_last = True
                        elif inbox_status.my_role == RoleType.RECIPIENT:
                            logger.warning(f"DB Consistency Issue: Inbox role RECIPIENT for {log_prefix}, but timestamp is NULL.")
                            they_sent_last = False
                        else: last_received_at_local = None
                    else: last_received_at_local = None; they_sent_last = False
                    if not last_message_sent_details and inbox_role == RoleType.AUTHOR: logger.debug(f"DB Consistency Check: MessageHistory empty for {username}, my_role is AUTHOR.")

                    # --- Reply & Interval Checks (Steps 7 & 8, same as V14.1) ---
                    if they_sent_last and last_received_at_local:
                         if last_sent_at_local:
                              if last_received_at_local > last_sent_at_local: logger.debug(f"Skipping {log_prefix}: Reply received."); skipped_count += 1; continue
                         else: logger.debug(f"Skipping {log_prefix}: Reply received, but no prior script msg.\n"); skipped_count += 1; continue
                    if last_sent_at_local:
                         time_since = now_local - last_sent_at_local
                         if time_since < MIN_MESSAGE_INTERVAL: logger.debug(f"Skipping {log_prefix}: Interval not met ({time_since} < {MIN_MESSAGE_INTERVAL})."); skipped_count += 1; continue

                    # --- Determine & Format Message (Steps 9 & 10, same as V14.1) ---
                    next_message_type_key = determine_next_message_type(last_message_sent_details, bool(is_in_family_tree))
                    if not next_message_type_key: skipped_count += 1; continue
                    message_template = MESSAGE_TEMPLATES.get(next_message_type_key)
                    if not message_template: logger.error(f"Missing template key '{next_message_type_key}'."); error_count += 1; overall_success = False; continue
                    name_to_use = "Valued Relative"; source = "Fallback"
                    if is_in_family_tree and person_name_in_tree: name_to_use = person_name_in_tree; source = "Tree"
                    elif first_name: name_to_use = first_name; source = "Person.first"
                    elif username and username != "Unknown":
                        clean_user = username.replace("(managed by ", "").replace(")", "").strip()
                        if clean_user and clean_user != "Unknown User": name_to_use = clean_user; source = "Person.user"
                        else: source = "Person.user (unknown)"
                    formatted_name = format_name(name_to_use)
                    format_data = {"name": formatted_name, "predicted_relationship": original_predicted_relationship or "N/A",
                                   "actual_relationship": original_actual_relationship or "N/A", "relationship_path": original_relationship_path or "N/A",
                                   "total_rows": total_family_tree_rows}
                    # ... (Formatting validation logic, same as V14.1) ...
                    try: req_keys = set(re.findall(r"\{(\w+)\}", message_template))
                    except Exception as regex_e: logger.error(f"Regex err:{regex_e}"); req_keys = set(format_data.keys())
                    keys_using_fallback = []; critical_keys = {"name"}; critical_keys_failed = []
                    original_data_map = {"predicted_relationship": original_predicted_relationship, "actual_relationship": original_actual_relationship, "relationship_path": original_relationship_path, "name": name_to_use}
                    for key in req_keys:
                         original_value = original_data_map.get(key); value_in_final_dict = format_data.get(key)
                         if value_in_final_dict == "N/A" and not original_value: keys_using_fallback.append(key)
                         if key in critical_keys and (value_in_final_dict == "N/A" and not original_value): critical_keys_failed.append(f"{key} (original data missing/empty)")
                         if key == "name" and value_in_final_dict == "Valued Relative":
                              if not name_to_use or name_to_use == "Valued Relative": critical_keys_failed.append(f"{key} (original name missing/invalid)")
                    if critical_keys_failed: logger.error(f"CRITICAL formatting keys missing/invalid original data for {log_prefix}: {critical_keys_failed}. Skipping."); error_count += 1; overall_success = False; continue
                    elif keys_using_fallback: logger.warning(f"Formatting {log_prefix}: Original data missing/empty for keys: {keys_using_fallback}. Message will use 'N/A'.")
                    try: message_text = message_template.format(**format_data)
                    except KeyError as e: logger.error(f"KeyError fmt msg for {log_prefix}: Missing key {e}.", exc_info=False); error_count += 1; overall_success = False; continue
                    except Exception as e: logger.error(f"Error fmt msg for {log_prefix}: {e}", exc_info=True); error_count += 1; overall_success = False; continue


                    # --- API Call Prep (Step 11, same as V14.1) ---
                    message_status: str = "skipped (logic_error)"
                    new_conversation_id_from_api_local: Optional[str] = None
                    is_initial_message: bool = not conversation_id; send_api_url: str = ""; payload: Dict[str, Any] = {}
                    send_api_desc: str = ""; api_headers = {}
                    if is_initial_message:
                        send_api_url = urljoin(config_instance.BASE_URL.rstrip("/") + "/", "app-api/express/v2/conversations/message")
                        send_api_desc = "Create Conversation API"
                        payload = {"content": message_text, "author": MY_PROFILE_ID_LOWER, "index": 0, "created": 0,
                                   "conversation_members": [{"user_id": recipient_profile_id_upper.lower(), "family_circles": []}, {"user_id": MY_PROFILE_ID_LOWER}]}
                    elif conversation_id:
                        send_api_url = urljoin(config_instance.BASE_URL.rstrip("/") + "/", f"app-api/express/v2/conversations/{conversation_id}")
                        send_api_desc = "Send Message API (Existing Conv)"
                        payload = {"content": message_text, "author": MY_PROFILE_ID_LOWER}
                    else: logger.error(f"Logic Error: Cannot determine API URL/payload for {log_prefix}."); error_count += 1; overall_success = False; continue
                    ctx_headers = config_instance.API_CONTEXTUAL_HEADERS.get(send_api_desc, {})
                    if "ancestry-userid" in ctx_headers: api_headers["ancestry-userid"] = MY_PROFILE_ID_UPPER

                    # --- Execute API Call (Step 12, same as V14.1) ---
                    if config_instance.APP_MODE == "dry_run":
                        action = "create" if is_initial_message else "send to"; logger.debug(f"[DRY RUN] Would {action} '{next_message_type_key}' to {username}.")
                        message_status = "typed (dry_run)"
                        if is_initial_message: new_conversation_id_from_api_local = f"dryrun_{uuid.uuid4()}"
                        else: new_conversation_id_from_api_local = conversation_id
                    elif config_instance.APP_MODE in ["production", "testing"]:
                        action = "create" if is_initial_message else "send follow-up"; logger.info(f"Attempting to {action} '{next_message_type_key}' to {log_prefix} via API...")
                        api_response: Optional[Any] = None
                        try:
                            api_response = _api_req(url=send_api_url, driver=session_manager.driver, session_manager=session_manager,
                                                    method="POST", json_data=payload, use_csrf_token=False, headers=api_headers,
                                                    api_description=send_api_desc, force_requests=True)
                        except Exception as api_e:
                            logger.error(f"Exception during _api_req ({send_api_desc}) for {log_prefix}: {api_e}", exc_info=True)
                            message_status = "send_error (api_exception)"; error_count += 1; overall_success = False; continue
                        # --- Validate API response (same as V14.1) ---
                        post_ok = False; author_match = False; api_conv_id: Optional[str] = None; api_author: Optional[str] = None
                        if api_response is not None:
                            if is_initial_message:
                                if isinstance(api_response, dict) and "conversation_id" in api_response:
                                    api_conv_id = str(api_response.get("conversation_id"))
                                    msg_details = api_response.get("message", {}); api_author = str(msg_details.get("author", "")).upper() if isinstance(msg_details, dict) else None
                                    if api_conv_id and api_author == MY_PROFILE_ID_UPPER:
                                        post_ok = True; author_match = True; new_conversation_id_from_api_local = api_conv_id
                                    else: logger.error(f"API initial response invalid. ConvID:'{api_conv_id}', Author:'{api_author}'")
                                else: logger.error(f"API call ({send_api_desc}) unexpected format (initial). Type:{type(api_response)}"); logger.debug(f"Resp:{api_response}")
                            else: # Follow-up
                                if isinstance(api_response, dict) and "author" in api_response:
                                    api_author = str(api_response.get("author", "")).upper()
                                    if api_author == MY_PROFILE_ID_UPPER:
                                        post_ok = True; author_match = True; new_conversation_id_from_api_local = conversation_id
                                        if api_response.get("content") != message_text: logger.warning(f"API response content mismatch.")
                                    else: logger.error(f"API follow-up author validation failed. Author:'{api_author}'")
                                else: logger.error(f"API call ({send_api_desc}) unexpected format (follow-up). Type:{type(api_response)}"); logger.debug(f"Resp:{api_response}")
                            if post_ok and author_match: message_status = "delivered OK"; logger.info(f"Message send to {log_prefix} ACCEPTED.")
                            else: message_status = "send_error (validation_failed)"; error_count += 1; overall_success = False; logger.warning(f"API POST validation failed for {log_prefix}.")
                        else: logger.error(f"API POST ({send_api_desc}) for {log_prefix} failed."); message_status = "send_error (post_failed)"; error_count += 1; overall_success = False
                    else: logger.warning(f"Skipping send for {log_prefix} due to unsupported APP_MODE: {config_instance.APP_MODE}"); skipped_count += 1; continue


                    # --- MODIFIED: Database Update Preparation ---
                    if message_status in ("delivered OK", "typed (dry_run)"):
                        try:
                            msg_type_obj = db_session.query(MessageType).filter(MessageType.type_name == next_message_type_key).first()
                            if not msg_type_obj: raise ValueError(f"MessageType '{next_message_type_key}' not found in DB.")

                            current_time_local = now_local
                            trunc_hist = message_text[:config_instance.MESSAGE_TRUNCATION_LENGTH]

                            # Create MessageHistory object
                            new_history = MessageHistory(
                                people_id=person_id, message_type_id=msg_type_obj.id,
                                message_text=trunc_hist, status=message_status, sent_at=current_time_local
                            )
                            db_objects_to_commit.append(new_history) # Add to batch list
                            logger.debug("Prepared MessageHistory for batch commit.")

                            # Prepare InboxStatus update/creation
                            inbox_status_upd = inbox_data.get(person_id) # Get from pre-fetched data
                            effective_conv_id = new_conversation_id_from_api_local
                            if not effective_conv_id:
                                logger.error(f"DB Prep Error for {log_prefix}: Conv ID missing!"); error_count += 1; overall_success = False; continue
                            trunc_inbox = message_text[:config_instance.MESSAGE_TRUNCATION_LENGTH]

                            if inbox_status_upd: # Update existing
                                updated_inbox = False
                                if inbox_status_upd.my_role != RoleType.AUTHOR: inbox_status_upd.my_role = RoleType.AUTHOR; updated_inbox = True
                                if inbox_status_upd.last_message != trunc_inbox: inbox_status_upd.last_message = trunc_inbox; updated_inbox = True
                                db_ts_naive_inbox = inbox_status_upd.last_message_timestamp.replace(tzinfo=None) if inbox_status_upd.last_message_timestamp and inbox_status_upd.last_message_timestamp.tzinfo else inbox_status_upd.last_message_timestamp
                                if db_ts_naive_inbox != current_time_local: inbox_status_upd.last_message_timestamp = current_time_local; updated_inbox = True
                                if inbox_status_upd.conversation_id != effective_conv_id: inbox_status_upd.conversation_id = effective_conv_id; updated_inbox = True; logger.debug(f"Updated Inbox Conv ID for {log_prefix}")
                                if updated_inbox:
                                    inbox_status_upd.last_updated = now_local
                                    db_objects_to_commit.append(inbox_status_upd) # Add updated object to batch list
                                    logger.debug("Prepared updated InboxStatus for batch commit.")
                                else: logger.debug(f"InboxStatus for {log_prefix} up-to-date.")
                            else: # Create new InboxStatus
                                new_inbox = InboxStatus(
                                    people_id=person_id, conversation_id=effective_conv_id, my_role=RoleType.AUTHOR,
                                    last_message=trunc_inbox, last_message_timestamp=current_time_local
                                )
                                db_objects_to_commit.append(new_inbox) # Add new object to batch list
                                logger.debug("Prepared new InboxStatus for batch commit.")
                                inbox_data[person_id] = new_inbox # Add to local dict for consistency within run

                            logger.debug(f"DB objects prepared: {message_status}.")
                            if message_status == "delivered OK":
                                sent_count += 1

                            # --- Commit periodically ---
                            if len(db_objects_to_commit) >= db_commit_batch_size:
                                logger.debug(f"Reached batch size ({db_commit_batch_size}). Committing staged DB objects...")
                                try:
                                    db_session.add_all(db_objects_to_commit)
                                    db_session.flush() # Flush to check constraints early
                                    # Commit happens via context manager at the end, or here if we want mid-run commits
                                    # db_session.commit() # Uncomment for mid-run commits
                                    logger.debug(f"Committed batch of {len(db_objects_to_commit)} DB objects.")
                                    db_objects_to_commit = [] # Clear the batch
                                except Exception as commit_err:
                                     logger.error(f"Error committing DB batch: {commit_err}", exc_info=True)
                                     # db_session.rollback() # Rollback handled by context manager
                                     error_count += len(db_objects_to_commit) # Mark affected items as errors
                                     db_objects_to_commit = [] # Clear batch after error
                                     overall_success = False
                                     # Potentially stop processing or just log and continue? Continue for now.

                        except (ValueError, SQLAlchemyError) as db_prep_err:
                             logger.error(f"DB prep error for {log_prefix}: {db_prep_err}", exc_info=True)
                             error_count += 1; overall_success = False; continue # Skip this message
                        except Exception as db_prep_unexp:
                             logger.critical(f"Unexpected DB prep error for {log_prefix}: {db_prep_unexp}", exc_info=True)
                             error_count += 1; overall_success = False; continue # Skip this message

                except Exception as inner_loop_e:
                    # --- Inner loop error handling (same as V14.1) ---
                    if isinstance(inner_loop_e, TypeError) and "can't compare offset-naive and offset-aware datetimes" in str(inner_loop_e):
                        logger.critical(f"CRITICAL DATETIME COMPARISON ERROR for {log_prefix}: {inner_loop_e}", exc_info=True)
                    elif isinstance(inner_loop_e, TypeError) and ("replace() got an unexpected keyword argument 'tzinfo'" in str(inner_loop_e) or "astimezone() cannot be applied to a naive datetime" in str(inner_loop_e)):
                        logger.critical(f"CRITICAL DATETIME CONVERSION ERROR for {log_prefix}: {inner_loop_e}\n", exc_info=True)
                    else: logger.error(f"Unexpected error processing {log_prefix}: {inner_loop_e}\n", exc_info=True)
                    error_count += 1; overall_success = False; continue

            # --- Commit any remaining objects after the loop ---
            if db_objects_to_commit:
                logger.debug(f"Committing final batch of {len(db_objects_to_commit)} DB objects...")
                try:
                     db_session.add_all(db_objects_to_commit)
                     db_session.flush() # Flush to check constraints early
                     # Commit happens via context manager
                     logger.debug("Final DB object batch staged for commit.")
                     db_objects_to_commit = [] # Clear the batch
                except Exception as commit_err:
                     logger.error(f"Error committing final DB batch: {commit_err}", exc_info=True)
                     # Rollback handled by context manager
                     error_count += len(db_objects_to_commit)
                     overall_success = False

            logger.debug("Finished processing all potential matches.")
            # Final commit happens automatically via context manager `get_db_conn_context`

    except Exception as outer_e:
        logger.critical(f"Critical error during message processing loop: {outer_e}", exc_info=True)
        overall_success = False
        # Rollback happens automatically via context manager

    # --- Final Summary (same as V14.1) ---
    logger.info("--- Message Sending Summary ----")
    potential_count = len(potential_matches) if potential_matches else 0
    logger.info(f"  Potential Matches Found: {potential_count}")
    logger.info(f"  Processed:             {processed_count}")
    logger.info(f"  Sent (API Accepted):   {sent_count}")
    logger.info(f"  Skipped (Policy/Rule): {skipped_count}")
    logger.info(f"  Errors (API/DB/etc.):  {error_count}")
    logger.info(f"  Overall Success:       {overall_success}")
    logger.info("---------------------------------\n")

    return overall_success
# End of send_messages_to_matches (V14.1 + Batch Commit)


#####################################################
# Stand alone testing
#####################################################


def main():
    """Main function for standalone testing of Action 8 (API version)."""
    from logging_config import setup_logging

    # --- Setup Logging ---
    try:
        from config import config_instance

        db_file_path = config_instance.DATABASE_FILE
        log_filename_only = db_file_path.with_suffix(".log").name
        global logger
        if (
            "logger" not in globals()
            or not isinstance(logger, logging.Logger)
            or not logger.hasHandlers()
        ):
            logger = setup_logging(log_file=log_filename_only, log_level="DEBUG")
        else:
            if logger and isinstance(logger, logging.Logger):
                logger.setLevel(logging.DEBUG)
            else:
                logger = logging.getLogger("logger")
                logger.setLevel(logging.DEBUG)
        logger.info(f"--- Starting Action 8 Standalone Test ---")
        logger.info(f"APP_MODE: {config_instance.APP_MODE}")
    except Exception as log_setup_e:
        import sys
        import logging as pylogging

        print(f"CRITICAL: Error during logging setup: {log_setup_e}", file=sys.stderr)
        pylogging.basicConfig(level=pylogging.DEBUG)
        logger = pylogging.getLogger("Action8Fallback")
        logger.info(f"--- Starting Action 8 Standalone Test (Fallback Logging) ---")
        logger.error(f"Initial logging setup failed: {log_setup_e}", exc_info=True)

    session_manager = SessionManager()
    action_success = False

    try:
        logger.info("Attempting to start session...")
        start_ok, _ = session_manager.start_sess(action_name="Action 8 Test")
        if start_ok:
            logger.info("Session started. Proceeding to send_messages_to_matches...")
            action_success = send_messages_to_matches(session_manager)
            if action_success:
                logger.info("send_messages_to_matches completed successfully.")
            else:
                logger.error("send_messages_to_matches reported errors/failed.")
        else:
            logger.critical("Failed to start session. Cannot run messaging.")
            action_success = False
    except Exception as e:
        logger.critical(
            f"Critical error in Action 8 standalone main: {e}", exc_info=True
        )
        action_success = False
    finally:
        logger.info("Closing session manager...")
        if session_manager:
            session_manager.close_sess()
        logger.info(
            f"--- Action 8 Standalone Test Finished (Overall Success: {action_success}) ---"
        )
# end main

if __name__ == "__main__":
    main()

# <<< END OF action8_messaging.py >>>
