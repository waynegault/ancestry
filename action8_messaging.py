# File: action8_messaging.py
# V1.16: Uses 2-row ConversationLog model, updates status to archive.

#!/usr/bin/env python3

# Standard library imports
import enum
import inspect
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast, Set
from urllib.parse import urlencode, urljoin, urlparse

# Third-party imports
import requests

# Removed tqdm imports
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Integer,
    String,
    Subquery,
    desc,
    func,
    over,
    select as sql_select,
    update,  # Added update
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DbSession, aliased, joinedload

# Local application imports
from cache import cache_result
from config import config_instance, selenium_config
from database import (
    DnaMatch,
    FamilyTree,
    ConversationLog,  # Use new log model
    MessageType,
    Person,
    db_transn,
    PersonStatusEnum,  # Use new enum
)
from utils import (
    DynamicRateLimiter,
    SessionManager,
    _api_req,
    format_name,
    make_ube,
    retry,
    time_wait,
    login_status,
)


# Define RoleType locally if needed
class RoleType(enum.Enum):
    AUTHOR = "AUTHOR"
    RECIPIENT = "RECIPIENT"


# End of RoleType definition

#####################################################
# Initialise
#####################################################
logger = logging.getLogger("logger")
logger.info(f"APP_MODE is: {config_instance.APP_MODE}")
MESSAGE_INTERVALS = {
    "testing": timedelta(seconds=5),
    "production": timedelta(weeks=8),
    "dry_run": timedelta(seconds=5),
}
MIN_MESSAGE_INTERVAL = MESSAGE_INTERVALS.get(
    config_instance.APP_MODE, timedelta(weeks=8)
)
logger.info(f"Using minimum message interval: {MIN_MESSAGE_INTERVAL}")

#####################################################
# Templates
#####################################################
MESSAGE_TYPES = {
    "In_Tree-Initial": "In_Tree-Initial",
    "In_Tree-Follow_Up": "In_Tree-Follow_Up",
    "In_Tree-Final_Reminder": "In_Tree-Final_Reminder",
    "Out_Tree-Initial": "Out_Tree-Initial",
    "Out_Tree-Follow_Up": "Out_Tree-Follow_Up",
    "Out_Tree-Final_Reminder": "Out_Tree-Final_Reminder",
    "In_Tree-Initial_for_was_Out_Tree": "In_Tree-Initial_for_was_Out_Tree",
    "User_Requested_Desist": "User_Requested_Desist",
    "Respond_to_No_Thanks": "Respond_to_No_Thanks",
    "Productive_Reply_Acknowledgement": "Productive_Reply_Acknowledgement",
}


@cache_result("message_templates")
def load_message_templates() -> Dict[str, str]:
    """Loads message templates from messages.json."""
    messages_path = Path(__file__).resolve().parent / "messages.json"
    if not messages_path.exists():
        logger.critical(f"CRITICAL: messages.json not found")
        return {}
    try:
        with messages_path.open("r", encoding="utf-8") as f:
            templates = json.load(f)
        if not isinstance(templates, dict) or not all(
            isinstance(v, str) for v in templates.values()
        ):
            logger.critical(f"CRITICAL: messages.json invalid format.")
            return {}
        return templates
    except Exception as e:
        logger.critical(f"CRITICAL: Error loading messages.json: {e}", exc_info=True)
        return {}


# End of load_message_templates

MESSAGE_TEMPLATES: Dict[str, str] = cast(Dict[str, str], load_message_templates())
if not MESSAGE_TEMPLATES:
    logger.error("Message templates failed to load.")


#####################################################
# Which message?
#####################################################
def determine_next_message_type(
    last_script_message_type: Optional[str], is_in_family_tree: bool
) -> Optional[str]:
    """V3 REVISED: Determines next msg type based on last SCRIPT message type name."""
    # ... (content unchanged from V1.15) ...
    next_type: Optional[str] = None
    skip_reason: str = "End of sequence"
    if not last_script_message_type:
        next_type = "In_Tree-Initial" if is_in_family_tree else "Out_Tree-Initial"
        skip_reason = "No prior script message"
    elif is_in_family_tree:
        if last_script_message_type.startswith("Out_Tree"):
            next_type = "In_Tree-Initial_for_was_Out_Tree"
            skip_reason = "Match now In Tree"
        elif last_script_message_type in [
            "In_Tree-Initial",
            "In_Tree-Initial_for_was_Out_Tree",
        ]:
            next_type = "In_Tree-Follow_Up"
            skip_reason = "Following In Tree Initial"
        elif last_script_message_type == "In_Tree-Follow_Up":
            next_type = "In_Tree-Final_Reminder"
            skip_reason = "Following In Tree Follow Up"
        else:
            skip_reason = f"End of In Tree sequence (last={last_script_message_type})"
    else:  # Not In Tree
        if last_script_message_type.startswith("In_Tree"):
            skip_reason = (
                f"Match was In Tree ({last_script_message_type}), now Out_Tree"
            )
        elif last_script_message_type == "Out_Tree-Initial":
            next_type = "Out_Tree-Follow_Up"
            skip_reason = "Following Out Tree Initial"
        elif last_script_message_type == "Out_Tree-Follow_Up":
            next_type = "Out_Tree-Final_Reminder"
            skip_reason = "Following Out Tree Follow Up"
        else:
            skip_reason = f"End of Out Tree sequence (last={last_script_message_type})"
    return next_type


# End of determine_next_message_type


#####################################################
# Send messages
#####################################################
def send_messages_to_matches(session_manager: SessionManager) -> bool:
    """
    V1.16 REVISED: Uses 2-row ConversationLog, updates status to archive after ACK.
    Commits in batches.
    """
    # --- Prerequisites ---
    if not session_manager or login_status(session_manager) is not True:
        return False
    if not session_manager.my_profile_id:
        return False
    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()
    MY_PROFILE_ID_UPPER = session_manager.my_profile_id.upper()
    if not MESSAGE_TEMPLATES:
        return False

    # --- Initialization ---
    overall_success = True
    processed_count = 0
    sent_count = 0
    skipped_count = 0
    error_count = 0
    max_to_send = config_instance.MAX_INBOX
    db_commit_batch_size = 10
    # Accumulators for batch DB operations
    conv_log_upserts: List[Dict[str, Any]] = []
    person_updates: Dict[int, Dict[str, Any]] = {}  # Store updates keyed by person_id
    total_potential = 0
    stop_reason = None

    # --- Fetch Message Type IDs ---
    message_type_ids: Dict[str, int] = {}
    ack_type_id: Optional[int] = None  # Store ACK type ID separately
    try:
        with session_manager.get_db_conn_context() as session_types:
            if not session_types:
                return False
            results = session_types.query(MessageType.id, MessageType.type_name).all()
            message_type_ids = {name: id for id, name in results}
            ack_type_id = message_type_ids.get("User_Requested_Desist")
            if not ack_type_id:
                logger.error("User_Requested_Desist type ID missing.")
                return False
            # logger.debug(f"Fetched {len(message_type_ids)} message type IDs.") # Verbose
    except Exception as e:
        logger.error(f"Error fetching MessageType IDs: {e}")
        return False

    # --- Main Processing ---
    session = None
    try:
        session = session_manager.get_db_conn()
        if not session:
            logger.error("Failed DB session.")
            return False

        # --- Pre-fetching Potential Matches (Joining with ConversationLog 'OUT' row) ---
        logger.debug("--- Pre-fetching Potential Matches (Action 8 / 2-Row Log) ---")

        # Alias for the ConversationLog join
        LatestOutLog = aliased(ConversationLog)

        potential_matches_query = (
            session.query(
                Person,  # Select the whole Person object
                DnaMatch.predicted_relationship,
                FamilyTree.actual_relationship,
                FamilyTree.person_name_in_tree,
                FamilyTree.relationship_path,
                LatestOutLog.latest_timestamp.label(
                    "last_script_message_ts"
                ),  # Timestamp from OUT row
                MessageType.type_name.label(
                    "last_script_message_type"
                ),  # Type name from OUT row
                LatestOutLog.message_type_id.label(
                    "last_script_message_type_id"
                ),  # ID from OUT row
            )
            # Join Person to OUT row in ConversationLog
            .outerjoin(
                LatestOutLog,
                (Person.id == LatestOutLog.people_id)
                & (LatestOutLog.direction == "OUT"),
            )
            # Join the OUT row to MessageType to get the type name
            .outerjoin(MessageType, LatestOutLog.message_type_id == MessageType.id)
            # Regular joins for DNA/Tree info
            .outerjoin(DnaMatch, Person.id == DnaMatch.people_id)
            .outerjoin(FamilyTree, Person.id == FamilyTree.people_id)
            # Filters
            .filter(
                Person.profile_id.isnot(None),
                Person.profile_id != "",
                Person.profile_id != "UNKNOWN",
            )
            .filter(Person.contactable == True)
            .filter(
                Person.status != PersonStatusEnum.ARCHIVE
            )  # Filter out archived people
            .order_by(Person.id)
        )
        potential_matches_results = potential_matches_query.all()
        total_potential = len(potential_matches_results)
        logger.debug(
            f"Found {total_potential} potential contactable, non-archived matches."
        )
        if not potential_matches_results:
            logger.info("No matches to message.")
            return True

        total_family_tree_rows = session.query(func.count(FamilyTree.id)).scalar() or 0

        # --- Process Matches ---
        logger.info(f"Processing {total_potential} matches...")
        for row_data in potential_matches_results:
            item_skipped = False
            item_error = False
            message_sent_this_iteration = False
            person_obj: Optional[Person] = None
            profile_id: Optional[str] = None
            person_status: Optional[PersonStatusEnum] = None
            last_script_message_ts: Optional[datetime] = None
            last_script_message_type: Optional[str] = None
            last_script_message_type_id: Optional[int] = None
            # Other fields needed for formatting
            predicted_relationship = None
            actual_relationship = None
            person_name_in_tree = None
            relationship_path = None
            is_in_family_tree = False
            first_name = None

            if max_to_send != 0 and sent_count >= max_to_send:
                stop_reason = f"Send Limit ({max_to_send})"
                break
            processed_count += 1

            try:  # Unpack data
                person_obj = row_data[0]
                predicted_relationship = row_data[1]
                actual_relationship = row_data[2]
                person_name_in_tree = row_data[3]
                relationship_path = row_data[4]
                last_script_message_ts = row_data[5]
                last_script_message_type = row_data[6]
                last_script_message_type_id = row_data[7]

                if not person_obj:
                    raise ValueError("Person object missing from query result")
                person_id = person_obj.id
                username = person_obj.username
                profile_id = person_obj.profile_id
                person_status = person_obj.status
                is_in_family_tree = bool(person_obj.in_my_tree)
                first_name = person_obj.first_name
                if not all([person_id, profile_id, username, person_status]):
                    raise ValueError("Essential person data missing from Person object")
                recipient_profile_id_upper = profile_id.upper()
                log_prefix = f"{username} #{person_id}"

                # --- Main Logic Checks ---
                now_utc = datetime.now(timezone.utc)
                next_message_type_key: Optional[str] = None

                # 1. Check Status
                if person_status == PersonStatusEnum.DESIST:
                    # Check if ACK was the *last* message sent
                    if last_script_message_type_id == ack_type_id:
                        logger.debug(
                            f"Skipping {log_prefix}: Status 'desist', ACK was last OUT msg. Needs status update to archive."
                        )
                        # Mark for status update to ARCHIVE, but don't send anything now
                        if people_id not in person_updates:
                            person_updates[people_id] = {}
                        person_updates[people_id]["status"] = PersonStatusEnum.ARCHIVE
                        skipped_count += 1
                        continue
                    else:  # Desist status, but last sent wasn't ACK -> Send ACK
                        logger.info(
                            f"{log_prefix}: Status 'desist', sending Acknowledgment."
                        )
                        next_message_type_key = "User_Requested_Desist"
                elif (
                    person_status != PersonStatusEnum.ACTIVE
                ):  # Skip if not active (and already handled desist above)
                    logger.debug(
                        f"Skipping {log_prefix}: Status is '{person_status.value}'."
                    )
                    skipped_count += 1
                    continue
                else:  # Status is active
                    # 2. Interval Check
                    if last_script_message_ts:
                        last_sent_aware = (
                            last_script_message_ts.astimezone(timezone.utc)
                            if last_script_message_ts.tzinfo
                            else last_script_message_ts.replace(tzinfo=timezone.utc)
                        )
                        if (now_utc - last_sent_aware) < MIN_MESSAGE_INTERVAL:
                            skipped_count += 1
                            continue
                    # 3. Determine Next Message Type
                    next_message_type_key = determine_next_message_type(
                        last_script_message_type, is_in_family_tree
                    )
                    if not next_message_type_key:
                        skipped_count += 1
                        continue

                if not next_message_type_key:
                    logger.error(f"Logic Error: No msg type for {log_prefix}")
                    error_count += 1
                    continue

                # --- Format Message ---
                message_template = MESSAGE_TEMPLATES.get(next_message_type_key)
                if not message_template:
                    logger.error(f"Missing template '{next_message_type_key}'.")
                    error_count += 1
                    continue
                name_to_use = "Valued Relative"
                if is_in_family_tree and person_name_in_tree:
                    name_to_use = person_name_in_tree
                elif first_name:
                    name_to_use = first_name
                elif username and username != "Unknown":
                    clean_user = (
                        username.replace("(managed by ", "").replace(")", "").strip()
                    )
                    if clean_user and clean_user != "Unknown User":
                        name_to_use = clean_user
                formatted_name = format_name(name_to_use)
                format_data = {
                    "name": formatted_name,
                    "predicted_relationship": predicted_relationship or "N/A",
                    "actual_relationship": actual_relationship or "N/A",
                    "relationship_path": relationship_path or "N/A",
                    "total_rows": total_family_tree_rows,
                }
                try:
                    message_text = message_template.format(**format_data)
                except Exception as e:
                    logger.error(f"Error formatting msg for {log_prefix}: {e}")
                    error_count += 1
                    continue

                # --- API Call Prep & Execution ---
                message_status: str = "error (prep)"
                conversation_id_to_log: Optional[str] = None
                # Find existing conversation ID from the last OUT log entry if possible
                last_out_log = (
                    session.query(ConversationLog.conversation_id)
                    .filter(
                        ConversationLog.people_id == person_id,
                        ConversationLog.direction == "OUT",
                    )
                    .order_by(ConversationLog.latest_timestamp.desc())
                    .first()
                )
                if last_out_log and last_out_log[0]:
                    conversation_id_to_log = last_out_log[0]
                elif next_message_type_key == "User_Requested_Desist":
                    logger.warning(f"Cannot send ACK {log_prefix}: No prior conv ID.")
                    error_count += 1
                    continue

                is_initial_message = conversation_id_to_log is None
                send_api_url: str = ""
                payload: Dict[str, Any] = {}
                send_api_desc: str = ""
                api_headers = {}
                if is_initial_message:
                    send_api_url = urljoin(
                        config_instance.BASE_URL,
                        "app-api/express/v2/conversations/message",
                    )
                    send_api_desc = "Create Conversation API"
                    payload = {
                        "content": message_text,
                        "author": MY_PROFILE_ID_LOWER,
                        "index": 0,
                        "created": 0,
                        "conversation_members": [
                            {"user_id": recipient_profile_id_upper.lower()},
                            {"user_id": MY_PROFILE_ID_LOWER},
                        ],
                    }
                elif conversation_id_to_log:
                    send_api_url = urljoin(
                        config_instance.BASE_URL,
                        f"app-api/express/v2/conversations/{conversation_id_to_log}",
                    )
                    send_api_desc = "Send Message API (Existing Conv)"
                    payload = {
                        "content": message_text,
                        "author": MY_PROFILE_ID_LOWER,
                    }
                else:
                    logger.error(f"Logic Error API URL {log_prefix}.")
                    error_count += 1
                    continue

                ctx_headers = config_instance.API_CONTEXTUAL_HEADERS.get(
                    send_api_desc, {}
                )
                api_headers = ctx_headers.copy()
                if "ancestry-userid" in api_headers and session_manager.my_profile_id:
                    api_headers["ancestry-userid"] = MY_PROFILE_ID_UPPER

                if config_instance.APP_MODE == "dry_run":
                    logger.info(
                        f"Dry Run: Would send '{next_message_type_key}' to {log_prefix}"
                    )
                    message_status = "typed (dry_run)"
                    sent_count += 1
                    message_sent_this_iteration = True
                    if is_initial_message:
                        conversation_id_to_log = f"dryrun_{uuid.uuid4()}"
                elif config_instance.APP_MODE in ["production", "testing"]:
                    action = "create" if is_initial_message else "reply"
                    logger.info(
                        f"Sending ({action}) '{next_message_type_key}' to {log_prefix}..."
                    )
                    api_response: Optional[Any] = None
                    try:
                        if not session_manager.is_sess_valid():
                            raise WebDriverException(
                                f"Session invalid sending {log_prefix}"
                            )
                        api_response = _api_req(
                            url=send_api_url,
                            driver=session_manager.driver,
                            session_manager=session_manager,
                            method="POST",
                            json_data=payload,
                            use_csrf_token=False,
                            headers=api_headers,
                            api_description=send_api_desc,
                            force_requests=True,
                        )
                    except Exception as api_e:
                        logger.error(f"Exception API send {log_prefix}: {api_e}.")
                        message_status = "send_error (api_exception)"
                        error_count += 1
                        continue

                    post_ok = False
                    api_conv_id: Optional[str] = None
                    api_author: Optional[str] = None
                    if api_response is not None:
                        if is_initial_message:
                            if (
                                isinstance(api_response, dict)
                                and "conversation_id" in api_response
                            ):
                                api_conv_id = str(api_response.get("conversation_id"))
                                msg_details = api_response.get("message", {})
                                api_author = (
                                    str(msg_details.get("author", "")).upper()
                                    if isinstance(msg_details, dict)
                                    else None
                                )
                                if api_conv_id and api_author == MY_PROFILE_ID_UPPER:
                                    post_ok = True
                                    conversation_id_to_log = api_conv_id
                            else:
                                logger.error(
                                    f"API initial response invalid {log_prefix}. Resp:{api_response}"
                                )
                        else:  # Follow-up/ACK
                            if (
                                isinstance(api_response, dict)
                                and "author" in api_response
                            ):
                                api_author = str(api_response.get("author", "")).upper()
                                if api_author == MY_PROFILE_ID_UPPER:
                                    post_ok = True
                            else:
                                logger.error(
                                    f"API reply response invalid {log_prefix}. Resp:{api_response}"
                                )
                        if post_ok:
                            message_status = "delivered OK"
                            sent_count += 1
                            message_sent_this_iteration = True
                        else:
                            message_status = "send_error (validation)"
                            error_count += 1
                            logger.warning(f"API POST validation failed {log_prefix}.")
                            continue
                    else:
                        logger.error(f"API POST failed (No response) {log_prefix}.")
                        message_status = "send_error (post_failed)"
                        error_count += 1
                        continue
                else:
                    logger.warning(
                        f"Skipping send (APP_MODE: {config_instance.APP_MODE})"
                    )
                    skipped_count += 1
                    continue

                # --- UPSERT Sent Message to ConversationLog ---
                if message_sent_this_iteration and conversation_id_to_log:
                    try:
                        msg_type_id = message_type_ids.get(next_message_type_key)
                        if not msg_type_id:
                            raise ValueError(
                                f"MessageType '{next_message_type_key}' missing."
                            )
                        current_time_utc = datetime.now(timezone.utc)
                        # Prepare data for UPSERT
                        log_upsert_data = {
                            "conversation_id": conversation_id_to_log,
                            "direction": "OUT",
                            "people_id": person_id,
                            "latest_message_content": message_text[
                                : config_instance.MESSAGE_TRUNCATION_LENGTH
                            ],  # Truncate
                            "latest_timestamp": current_time_utc,
                            "message_type_id": msg_type_id,
                            "script_message_status": message_status,
                            "ai_sentiment": None,  # Always null for OUT
                            "updated_at": current_time_utc,  # Set update time
                        }
                        conv_log_upserts.append(log_upsert_data)  # Add to batch list

                        # If ACK sent, mark Person for status update to ARCHIVE
                        if next_message_type_key == "User_Requested_Desist":
                            logger.info(
                                f"Marking Person {person_id} status for update to 'archive'."
                            )
                            if person_id not in person_updates:
                                person_updates[person_id] = {}
                            person_updates[person_id][
                                "status"
                            ] = PersonStatusEnum.ARCHIVE  # Use Enum

                    except Exception as db_log_err:
                        logger.error(
                            f"Error preparing ConvLog UPSERT {log_prefix}: {db_log_err}"
                        )
                        error_count += 1
                # --- End UPSERT Prep ---

            except ValueError as ve:
                logger.error(f"Data Error item {processed_count}: {ve}")
                error_count += 1
            except Exception as loop_e:
                logger.error(
                    f"Unexpected Error item {processed_count}: {loop_e}", exc_info=True
                )
                error_count += 1
            finally:
                # Commit periodically
                if len(conv_log_upserts) + len(person_updates) >= db_commit_batch_size:
                    # logger.debug(f"Committing batch: {len(conv_log_upserts)} logs, {len(person_updates)} persons") # Verbose
                    updates_done = self._commit_batch_data_upsert(
                        session, conv_log_upserts, person_updates
                    )
                    if updates_done < len(person_updates):
                        error_count += (
                            len(person_updates) - updates_done
                        )  # Count person updates that failed
                    if len(conv_log_upserts) > 0:
                        error_count += len(
                            conv_log_upserts
                        )  # Assume logs failed if commit failed
                    conv_log_upserts.clear()
                    person_updates.clear()

        # --- End Main Loop ---
        if stop_reason:
            logger.info(f"Message sending stopped: {stop_reason}")

    except Exception as outer_e:
        logger.critical(
            f"Critical error during message sending: {outer_e}", exc_info=True
        )
        overall_success = False
    finally:
        # --- Final Commit ---
        if session and (conv_log_upserts or person_updates):
            logger.info(
                f"Committing final batch: {len(conv_log_upserts)} logs, {len(person_updates)} persons..."
            )
            final_updates_done = self._commit_batch_data_upsert(
                session, conv_log_upserts, person_updates, is_final_attempt=True
            )
            if final_updates_done < len(person_updates):
                error_count += len(person_updates) - final_updates_done
            if len(conv_log_upserts) > 0 and final_updates_done == 0:
                error_count += len(conv_log_upserts)  # If commit failed entirely
            # Don't clear here as we are exiting

        # --- Final Summary ---
        logger.info("---- Message Sending Summary (2-Row Log) ----")
        logger.info(f"  Potential Matches Evaluated: {total_potential}")
        logger.info(f"  Processed:                   {processed_count}")
        logger.info(f"  Sent/DryRun:                 {sent_count}")
        logger.info(f"  Skipped (Policy/Rule):       {skipped_count}")
        logger.info(f"  Errors (API/DB/etc.):        {error_count}")
        logger.info(f"  Overall Success:             {overall_success}")
        logger.info("-------------------------------------------")

        if session:
            session_manager.return_session(session)

    return overall_success


# End of send_messages_to_matches


# --- UPSERT Helper for 2-Row Model ---
def _commit_batch_data_upsert(
    session: DbSession,
    log_upserts: List[Dict],
    person_updates: Dict[int, Dict],
    is_final_attempt: bool = False,
) -> int:
    """Helper function to UPSERT ConversationLog entries and update Person statuses using session.merge()."""
    updated_person_count = 0
    if not log_upserts and not person_updates:
        return updated_person_count
    log_prefix = "[Final Save] " if is_final_attempt else "[Batch Save] "
    logger.debug(
        f"{log_prefix} UPSERT/Update: {len(log_upserts)} logs, {len(person_updates)} persons."
    )

    try:
        with db_transn(session):  # Use transaction context manager
            # ConversationLog UPSERTs using merge
            if log_upserts:
                # logger.debug(f"{log_prefix}Merging {len(log_upserts)} ConversationLog state entries...") # Verbose
                for data in log_upserts:
                    # Create instance to merge. SQLAlchemy handles insert/update based on PK.
                    log_obj = ConversationLog(**data)
                    session.merge(log_obj)  # Merge handles UPSERT
                # logger.debug(f"{log_prefix}Merged {len(log_upserts)} log entries.") # Verbose

            # Person status updates using bulk_update_mappings
            if person_updates:
                pids_to_update = list(person_updates.keys())
                # logger.info(f"{log_prefix}Updating {len(pids_to_update)} persons statuses...") # Verbose
                update_values = [
                    {
                        "id": pid,
                        "status": data["status"],
                        "updated_at": datetime.now(timezone.utc),
                    }
                    for pid, data in person_updates.items()
                ]
                if update_values:
                    session.bulk_update_mappings(Person, update_values)
                    updated_person_count = len(update_values)
                    logger.info(
                        f"{log_prefix}Updated {updated_person_count} persons statuses."
                    )

        # logger.debug(f"{log_prefix}Commit successful.") # Verbose
        return updated_person_count
    except Exception as commit_err:
        logger.error(f"{log_prefix}Commit FAILED: {commit_err}", exc_info=True)
        return 0


# End of _commit_batch_data_upsert


#####################################################
# Stand alone testing
#####################################################
def main():
    """Main function for standalone testing of Action 8."""
    from logging_config import setup_logging

    log_filename_only = "action8_test.log"
    try:
        from config import config_instance

        db_file_path = config_instance.DATABASE_FILE
        log_filename_only = db_file_path.with_suffix(".log").name
    except Exception as config_log_err:
        import sys

        print(
            f"Warning: Could not get log filename from config: {config_log_err}",
            file=sys.stderr,
        )
    global logger
    try:
        logger = setup_logging(log_file=log_filename_only, log_level="DEBUG")
    except Exception as log_setup_e:
        import sys, logging as pylogging

        print(f"CRITICAL: Logging setup error: {log_setup_e}", file=sys.stderr)
        pylogging.basicConfig(level=pylogging.DEBUG)
        logger = pylogging.getLogger("Action8Fallback")
        logger.error(f"Initial logging setup failed: {log_setup_e}", exc_info=True)

    logger.info(f"--- Starting Action 8 Standalone Test (2-Row Log) ---")
    logger.info(f"APP_MODE: {config_instance.APP_MODE}")
    session_manager = SessionManager()
    action_success = False
    try:
        logger.info("Attempting to start session...")
        start_ok = session_manager.start_sess(
            action_name="Action 8 Test"
        )  # Corrected: No index
        if start_ok:
            ready_ok = session_manager.ensure_session_ready(
                action_name="Action 8 Test Ready"
            )
            if ready_ok:
                logger.info("Session ready. Proceeding...")
                action_success = send_messages_to_matches(session_manager)
                if action_success:
                    logger.info("send_messages_to_matches OK.")
                else:
                    logger.error("send_messages_to_matches FAILED.")
            else:
                logger.critical("Failed session readiness.")
                action_success = False
        else:
            logger.critical("Failed to start session.")
            action_success = False
    except Exception as e:
        logger.critical(f"Critical error in Action 8 main: {e}", exc_info=True)
        action_success = False
    finally:
        logger.info("Closing session manager...")
        if session_manager:
            session_manager.close_sess()
        logger.info(
            f"--- Action 8 Standalone Test Finished (Success: {action_success}) ---"
        )


# end main

if __name__ == "__main__":
    main()
# <<< END OF action8_messaging.py >>>
