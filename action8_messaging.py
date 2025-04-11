# File: action8_messaging.py
# V14.58: Refactored send_messages_to_matches into smaller helpers. Added progress bar. Fixed 'and_' import. Added detailed comments.

#!/usr/bin/env python3

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
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from urllib.parse import urlencode, urljoin, urlparse

# Third-party imports
import requests
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
    and_,  
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.orm import (
    Session as DbSession,
    aliased,
    joinedload,
)
from sqlalchemy.sql import select
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Local application imports
from cache import cache_result
from config import config_instance, selenium_config
from database import (
    DnaMatch,
    FamilyTree,
    MessageType,
    Person,
    RoleType,
    db_transn,
    ConversationLog,
    MessageDirectionEnum,
    PersonStatusEnum,
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


#####################################################
# Initialise & Templates
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
)
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
    "User_Requested_Desist": "User_Requested_Desist",
}


@cache_result("message_templates")
def load_message_templates() -> Dict[str, str]:
    """Loads message templates from messages.json, validating required keys."""
    # Step 1: Define path to messages.json
    messages_path = Path("messages.json")

    # Step 2: Check if file exists
    if not messages_path.exists():
        logger.critical(
            f"CRITICAL: messages.json not found at {messages_path.resolve()}"
        )
        return {}

    # Step 3: Read and parse the JSON file
    try:
        with messages_path.open("r", encoding="utf-8") as f:
            templates = json.load(f)

        # Step 4: Validate JSON structure (must be dict of strings)
        if not isinstance(templates, dict) or not all(
            isinstance(v, str) for v in templates.values()
        ):
            logger.critical(
                f"CRITICAL: messages.json does not contain a dictionary of strings."
            )
            return {}

        # Step 5: Validate required template keys are present
        required_keys = set(MESSAGE_TYPES.keys())
        missing_keys = required_keys - set(templates.keys())
        if missing_keys:
            logger.critical(
                f"CRITICAL: messages.json missing required template keys: {', '.join(missing_keys)}"
            )
            return {}  # Fail if critical templates missing

        # Step 6: Log success and return templates
        logger.info("Message templates loaded successfully.")
        return templates
    except json.JSONDecodeError as e:
        logger.critical(f"CRITICAL: Error decoding messages.json: {e}")
        return {}
    except Exception as e:
        logger.critical(f"CRITICAL: Error loading messages.json: {e}", exc_info=True)
        return {}
# End of load_message_templates

MESSAGE_TEMPLATES: Dict[str, str] = cast(Dict[str, str], load_message_templates())
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
    Determines the next standard message type based on history and tree status.
    """
    # Step 1: Log current tree status
    logger.debug(f"Determining next msg: Currently In Tree: {is_in_family_tree}")

    # Step 2: Handle initial message case (no previous script message)
    if not last_message_details:
        next_type = "In_Tree-Initial" if is_in_family_tree else "Out_Tree-Initial"
        logger.debug(f"Result: {next_type} (Reason: No prior script message).")
        return next_type

    # Step 3: Unpack details of the last script message sent
    last_message_type, last_sent_at, last_message_status = last_message_details
    next_type: Optional[str] = None
    skip_reason: str = "End of sequence or other condition met"

    # Step 4: Determine next message based on current tree status and last message type
    if is_in_family_tree:
        # --- Logic for IN_TREE matches ---
        if last_message_type.startswith("Out_Tree"):
            next_type = (
                "In_Tree-Initial_for_was_Out_Tree"  # Handle switch from Out to In
            )
            logger.debug("Match was Out_Tree, now In_Tree.")
        elif (
            last_message_type == "In_Tree-Initial"
            or last_message_type == "In_Tree-Initial_for_was_Out_Tree"
        ):
            next_type = "In_Tree-Follow_Up"  # Follow up initial In_Tree message
            logger.debug(f"Following up on {last_message_type}.")
        elif last_message_type == "In_Tree-Follow_Up":
            next_type = "In_Tree-Final_Reminder"  # Send final reminder
            logger.debug("Sending final In_Tree reminder.")
        elif last_message_type == "In_Tree-Final_Reminder":
            skip_reason = f"End of In_Tree sequence (last was {last_message_type})"  # End of sequence
            logger.debug(f"{skip_reason}.")
        else:
            # Handle other cases (like ACK already sent or unknown type)
            if last_message_type == "User_Requested_Desist":
                skip_reason = f"Skipping standard message: Last sent was Desist ACK."
                logger.debug(f"{skip_reason}.")
            elif last_message_type == "Unknown":
                skip_reason = "Previous message type was Unknown."
                logger.warning(f"{skip_reason} for In_Tree match.")
            else:
                skip_reason = f"Unexpected previous In_Tree type: {last_message_type}"
                logger.warning(f"{skip_reason}.")
    else:
        # --- Logic for OUT_OF_TREE matches ---
        if last_message_type.startswith("In_Tree"):
            if last_message_type == "In_Tree-Initial_for_was_Out_Tree":
                # Handle switch Out -> In -> Out: Restart Out sequence
                logger.warning(f"Match was Out->In->Out. Restarting Out sequence.")
                next_type = "Out_Tree-Initial"
            else:
                # Skip if they were In_Tree but are now Out_Tree
                skip_reason = (
                    f"Match was In_Tree ({last_message_type}) but is now Out_Tree"
                )
                logger.warning(f"{skip_reason}. Skipping standard message.")
        elif last_message_type == "Out_Tree-Initial":
            next_type = "Out_Tree-Follow_Up"  # Follow up initial Out_Tree message
            logger.debug("Following up on Out_Tree-Initial.")
        elif last_message_type == "Out_Tree-Follow_Up":
            next_type = "Out_Tree-Final_Reminder"  # Send final reminder
            logger.debug("Sending final Out_Tree reminder.")
        elif last_message_type == "Out_Tree-Final_Reminder":
            skip_reason = f"End of Out_Tree sequence (last was {last_message_type})"  # End of sequence
            logger.debug(f"{skip_reason}.")
        else:
            # Handle other cases (like ACK already sent or unknown type)
            if last_message_type == "User_Requested_Desist":
                skip_reason = f"Skipping standard message: Last sent was Desist ACK."
                logger.debug(f"{skip_reason}.")
            elif last_message_type == "Unknown":
                skip_reason = "Previous message type was Unknown."
                logger.warning(f"{skip_reason} for Out_Tree match.")
            else:
                skip_reason = f"Unexpected previous Out_Tree type: {last_message_type}"
                logger.warning(f"{skip_reason}.")

    # Step 5: Log the final decision and return the next message type key or None
    if next_type:
        logger.debug(
            f"Next standard message: {next_type} (Reason: Following sequence)."
        )
    else:
        logger.debug(f"No next standard message: {skip_reason}.")
    return next_type
# End of determine_next_message_type

#####################################################
# Helper Functions (New & Refactored)
#####################################################


def _commit_messaging_batch(
    session: DbSession,
    logs_to_add: List[ConversationLog],
    person_updates: Dict[int, PersonStatusEnum],
    batch_num: int,
) -> bool:
    """Commits a batch of new ConversationLog entries and Person status updates."""
    # Step 1: Check if there's anything to commit
    if not logs_to_add and not person_updates:
        logger.debug(f"Batch {batch_num}: Nothing to commit.")
        return True

    # Step 2: Log preparation details
    logger.info(
        f"Attempting batch commit (Batch {batch_num}): {len(logs_to_add)} logs, {len(person_updates)} persons..."
    )

    try:
        # Step 3: Enter transaction context
        with db_transn(session):
            # Step 3a: Add new ConversationLog entries if any
            if logs_to_add:
                logger.debug(f" Adding {len(logs_to_add)} ConversationLog entries...")
                session.add_all(logs_to_add)

            # Step 3b: Prepare and execute Person status updates if any
            if person_updates:
                update_mappings = [
                    {
                        "id": pid,
                        "status": status_enum,  # Enum object is expected by bulk_update_mappings
                        "updated_at": datetime.now(timezone.utc),
                    }
                    for pid, status_enum in person_updates.items()
                ]
                logger.debug(f" Updating {len(update_mappings)} Person statuses...")
                session.bulk_update_mappings(Person, update_mappings)

            # Step 3c: Log session state just before commit attempt (optional but helpful)
            logger.debug("Session state BEFORE commit attempt:")
            logger.debug(f"  Dirty objects: {len(session.dirty)}")
            logger.debug(f"  New objects: {len(session.new)}")
            logger.debug(f"  Deleted objects: {len(session.deleted)}")

        # Step 4: Log successful commit (db_transn handles actual commit)
        logger.info(f"Batch commit successful (Batch {batch_num}).")
        return True

    except IntegrityError as ie:
        # Step 5a: Handle integrity errors (e.g., unique constraint)
        logger.error(
            f"DB UNIQUE constraint error during messaging batch commit (Batch {batch_num}): {ie}",
            exc_info=False,
        )
        return False
    except Exception as e:
        # Step 5b: Handle other commit errors
        logger.error(
            f"Error committing messaging batch (Batch {batch_num}): {e}", exc_info=True
        )
        return False
# End of _commit_messaging_batch


def _prefetch_messaging_data(
    db_session: DbSession,
) -> Tuple[
    Optional[Dict[str, int]],
    Optional[List[Person]],
    Optional[Dict[int, ConversationLog]],
    Optional[Dict[int, ConversationLog]],
]:
    """Fetches all necessary data from the database before processing messages."""
    message_type_map: Optional[Dict[str, int]] = None
    candidate_persons: Optional[List[Person]] = None
    latest_in_log_map: Optional[Dict[int, ConversationLog]] = None
    latest_out_log_map: Optional[Dict[int, ConversationLog]] = None

    try:
        # Step 1: Fetch MessageType IDs and create a mapping
        logger.debug("Prefetching MessageType IDs...")
        message_types = db_session.query(MessageType.id, MessageType.type_name).all()
        message_type_map = {name: mt_id for mt_id, name in message_types}
        if not message_type_map or len(message_type_map) < len(MESSAGE_TYPES):
            logger.critical(
                f"Failed to fetch all required MessageType IDs. Found: {list(message_type_map.keys())}"
            )
            return None, None, None, None
        if "User_Requested_Desist" not in message_type_map:
            logger.critical(
                "CRITICAL: 'User_Requested_Desist' MessageType ID not found."
            )
            return None, None, None, None
        logger.debug(f"Fetched {len(message_type_map)} MessageType IDs.")

        # Step 2: Fetch Candidate Persons (Status ACTIVE or DESIST) with eager loading
        logger.debug("Prefetching candidate persons (Status ACTIVE or DESIST)...")
        candidate_persons = (
            db_session.query(Person)
            .options(
                joinedload(Person.dna_match),  # Eager load DNA match data
                joinedload(Person.family_tree),  # Eager load Family Tree data
            )
            .filter(
                Person.profile_id.isnot(None),  # Must have profile ID
                Person.profile_id != "",  # Must not be empty
                Person.profile_id != "UNKNOWN",  # Must not be unknown
                Person.contactable == True,  # Must be contactable
                Person.status.in_(
                    [PersonStatusEnum.ACTIVE, PersonStatusEnum.DESIST]
                ),  # Filter by status
            )
            .order_by(Person.id)  # Ensure consistent processing order
            .all()
        )
        logger.debug(f"Fetched {len(candidate_persons)} potential candidates.")
        if not candidate_persons:
            return message_type_map, [], {}, {}  # Return empty if no candidates

        # Step 3: Fetch Latest ConversationLog entries (IN and OUT) for candidates
        candidate_person_ids = [p.id for p in candidate_persons if p.id]
        if not candidate_person_ids:
            return (
                message_type_map,
                candidate_persons,
                {},
                {},
            )  # Should not happen if persons found

        logger.debug(
            f"Prefetching latest IN/OUT ConversationLogs for {len(candidate_person_ids)} people..."
        )
        # Step 3a: Subquery to find the latest timestamp for each person/direction pair
        latest_ts_subq = (
            db_session.query(
                ConversationLog.people_id,
                ConversationLog.direction,
                func.max(ConversationLog.latest_timestamp).label("max_ts"),
            )
            .filter(ConversationLog.people_id.in_(candidate_person_ids))
            .group_by(ConversationLog.people_id, ConversationLog.direction)
            .subquery("latest_ts")
        )
        # Step 3b: Join back to get the full log entry for those latest timestamps
        latest_logs_query = (
            db_session.query(ConversationLog)
            .join(
                latest_ts_subq,
                and_(  # Use and_() for multiple join conditions
                    ConversationLog.people_id == latest_ts_subq.c.people_id,
                    ConversationLog.direction == latest_ts_subq.c.direction,
                    ConversationLog.latest_timestamp == latest_ts_subq.c.max_ts,
                ),
            )
            .options(
                joinedload(ConversationLog.message_type)
            )  # Eager load message type name
        )
        latest_logs = latest_logs_query.all()

        # Step 3c: Populate maps with the latest logs
        latest_in_log_map = {}
        latest_out_log_map = {}
        for log in latest_logs:
            if log.direction == MessageDirectionEnum.IN:
                latest_in_log_map[log.people_id] = log
            elif log.direction == MessageDirectionEnum.OUT:
                latest_out_log_map[log.people_id] = log
        logger.debug(f"Fetched latest IN logs for {len(latest_in_log_map)} people.")
        logger.debug(f"Fetched latest OUT logs for {len(latest_out_log_map)} people.")

        # Step 4: Return all fetched data
        return (
            message_type_map,
            candidate_persons,
            latest_in_log_map,
            latest_out_log_map,
        )

    except SQLAlchemyError as db_err:
        logger.error(f"DB error during pre-fetching: {db_err}", exc_info=True)
        return None, None, None, None
    except Exception as e:
        logger.error(f"Unexpected error during pre-fetching: {e}", exc_info=True)
        return None, None, None, None
# End of _prefetch_messaging_data


def _send_message_via_api(
    session_manager: SessionManager,
    person: Person,
    message_text: str,
    existing_conv_id: Optional[str],
    log_prefix: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Sends a message using the appropriate API endpoint (create or existing)."""
    # Step 1: Get required IDs
    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()
    MY_PROFILE_ID_UPPER = session_manager.my_profile_id.upper()
    recipient_profile_id_upper = person.profile_id.upper()
    is_initial = not existing_conv_id

    # Step 2: Determine API URL, payload, and description based on whether it's an initial message
    send_api_url: str = ""
    payload: Dict[str, Any] = {}
    send_api_desc: str = ""
    api_headers = {}

    if is_initial:
        send_api_url = urljoin(
            config_instance.BASE_URL.rstrip("/") + "/",
            "app-api/express/v2/conversations/message",
        )
        send_api_desc = "Create Conversation API"
        payload = {
            "content": message_text,
            "author": MY_PROFILE_ID_LOWER,
            "index": 0,
            "created": 0,
            "conversation_members": [
                {"user_id": recipient_profile_id_upper.lower(), "family_circles": []},
                {"user_id": MY_PROFILE_ID_LOWER},
            ],
        }
    elif existing_conv_id:
        send_api_url = urljoin(
            config_instance.BASE_URL.rstrip("/") + "/",
            f"app-api/express/v2/conversations/{existing_conv_id}",
        )
        send_api_desc = "Send Message API (Existing Conv)"
        payload = {"content": message_text, "author": MY_PROFILE_ID_LOWER}
    else:  # Should not happen if logic is correct
        logger.error(f"Logic Error: Cannot determine API URL/payload for {log_prefix}.")
        return "send_error (api_prep_failed)", None

    # Step 3: Prepare Headers
    ctx_headers = config_instance.API_CONTEXTUAL_HEADERS.get(send_api_desc, {})
    api_headers = ctx_headers.copy()
    if "ancestry-userid" in api_headers and MY_PROFILE_ID_UPPER:
        api_headers["ancestry-userid"] = MY_PROFILE_ID_UPPER

    # Step 4: Make the API call using _api_req helper
    api_response = _api_req(
        url=send_api_url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="POST",
        json_data=payload,
        use_csrf_token=False,
        headers=api_headers,
        api_description=send_api_desc,
        force_requests=True,  # Use requests for messaging API
    )

    # Step 5: Validate the API response
    message_status = "send_error (unknown)"  # Default status
    new_conversation_id_from_api = None
    post_ok = False
    api_conv_id = None
    api_author = None

    if api_response is not None:
        if is_initial:  # Validation for creating a new conversation
            if isinstance(api_response, dict) and "conversation_id" in api_response:
                api_conv_id = str(api_response.get("conversation_id"))
                msg_details = api_response.get("message", {})
                api_author = (
                    str(msg_details.get("author", "")).upper()
                    if isinstance(msg_details, dict)
                    else None
                )
                # Check if conversation ID is present and author matches our ID
                if api_conv_id and api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = api_conv_id
                else:
                    logger.error(
                        f"API initial response invalid (values) for {log_prefix}. ConvID: {api_conv_id}, Author: {api_author}"
                    )
            else:
                logger.error(
                    f"API call ({send_api_desc}) unexpected format (initial) for {log_prefix}. Resp:{api_response}"
                )
        else:  # Validation for sending to an existing conversation
            if isinstance(api_response, dict) and "author" in api_response:
                api_author = str(api_response.get("author", "")).upper()
                # Check if author matches our ID
                if api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = existing_conv_id  # Reuse existing ID
                else:
                    logger.error(
                        f"API follow-up author validation failed for {log_prefix}."
                    )
            else:
                logger.error(
                    f"API call ({send_api_desc}) unexpected format (follow-up) for {log_prefix}. Resp:{api_response}"
                )

        # Step 6: Determine final status based on validation
        if post_ok:
            message_status = "delivered OK"
            logger.debug(f"Message send to {log_prefix} ACCEPTED by API.")
        else:
            message_status = "send_error (validation_failed)"
            logger.warning(f"API POST validation failed for {log_prefix}.")
    else:  # Handle case where _api_req returned None (e.g., connection error after retries)
        message_status = "send_error (post_failed)"
        logger.error(
            f"API POST ({send_api_desc}) for {log_prefix} failed (No response/Retries exhausted)."
        )

    # Step 7: Return the final status and potentially the new conversation ID
    return message_status, new_conversation_id_from_api
# End of _send_message_via_api


def _process_single_person(
    db_session: DbSession,
    session_manager: SessionManager,
    person: Person,
    latest_in_log: Optional[ConversationLog],
    latest_out_log: Optional[ConversationLog],
    message_type_map: Dict[str, int],
) -> Tuple[Optional[ConversationLog], Optional[Tuple[int, PersonStatusEnum]], str]:
    """
    Processes a single person: applies rules, determines/sends message, prepares DB updates.
    Returns: Tuple(new_log_entry_or_None, person_update_tuple_or_None, status_string)
    status_string indicates outcome: 'sent', 'acked', 'skipped', 'error'
    """
    log_prefix = f"{person.username} #{person.id} (Status: {person.status.name})"
    message_to_send_key: Optional[str] = None
    send_reason = "Unknown"
    status_string = "error"  # Default return status

    try:
        # --- Step 1: Rule Checks ---
        # Skip if ARCHIVED
        if person.status == PersonStatusEnum.ARCHIVE:
            logger.debug(f"Skipping {log_prefix}: Status is ARCHIVE.")
            raise StopIteration("skipped")

        # Skip if user replied since last script message
        if (
            latest_in_log
            and latest_out_log
            and latest_in_log.latest_timestamp > latest_out_log.latest_timestamp
        ):
            logger.debug(
                f"Skipping {log_prefix}: User replied since last script message."
            )
            raise StopIteration("skipped")
        # Skip if user initiated the conversation
        elif latest_in_log and not latest_out_log:
            logger.debug(f"Skipping {log_prefix}: User sent first message.")
            raise StopIteration("skipped")

        # Skip if minimum interval hasn't passed since last OUT message
        now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        if latest_out_log and latest_out_log.latest_timestamp:
            last_out_ts_naive = latest_out_log.latest_timestamp.astimezone(
                timezone.utc
            ).replace(tzinfo=None)
            if (now_utc_naive - last_out_ts_naive) < MIN_MESSAGE_INTERVAL:
                logger.debug(f"Skipping {log_prefix}: Within minimum interval.")
                raise StopIteration("skipped")

        # --- Step 2: Determine Message Key based on Person Status ---
        if person.status == PersonStatusEnum.DESIST:
            # --- Step 2a: Handle DESIST status (send ACK if not already sent) ---
            desist_ack_type_id = message_type_map["User_Requested_Desist"]
            ack_already_sent = (
                latest_out_log and latest_out_log.message_type_id == desist_ack_type_id
            )
            if ack_already_sent:
                logger.debug(f"Skipping {log_prefix}: DESIST already acknowledged.")
                raise StopIteration("skipped")
            else:
                message_to_send_key = "User_Requested_Desist"
                send_reason = "DESIST Acknowledgment"
        elif person.status == PersonStatusEnum.ACTIVE:
            # --- Step 2b: Handle ACTIVE status (determine next standard message) ---
            last_script_message_details: Optional[Tuple[str, datetime, str]] = None
            if (
                latest_out_log and latest_out_log.message_type
            ):  # Check if related message_type was loaded
                last_out_ts_naive = latest_out_log.latest_timestamp.astimezone(
                    timezone.utc
                ).replace(tzinfo=None)
                last_script_message_details = (
                    latest_out_log.message_type.type_name,  # Access loaded type name
                    last_out_ts_naive,
                    latest_out_log.script_message_status or "Unknown",
                )
            message_to_send_key = determine_next_message_type(
                last_script_message_details, bool(person.in_my_tree)
            )
            if not message_to_send_key:
                # Reason logged in determine_next_message_type
                raise StopIteration("skipped")
            send_reason = "Standard Sequence"
        else:
            # Should not happen due to pre-filtering, but handle defensively
            logger.error(
                f"Unexpected status '{person.status.name}' for {log_prefix}. Skipping."
            )
            raise StopIteration("skipped")

        # --- Step 3: Format Message ---
        # Ensure a message key was determined
        if not message_to_send_key:
            logger.error(
                f"Logic Error: No message key determined for {log_prefix}. Skipping."
            )
            raise StopIteration("error")

        message_template = MESSAGE_TEMPLATES.get(message_to_send_key)
        if not message_template:
            logger.error(
                f"Missing template for key '{message_to_send_key}'. Skipping {log_prefix}."
            )
            raise StopIteration("error")

        # Gather data for formatting (use pre-loaded relationships)
        dna_match = person.dna_match
        family_tree = person.family_tree
        name_to_use = (
            family_tree.person_name_in_tree
            if family_tree and family_tree.person_name_in_tree
            else (
                person.first_name
                if person.first_name
                else (
                    person.username
                    if person.username != "Unknown User"
                    else "Valued Relative"
                )
            )
        )
        formatted_name = format_name(name_to_use)
        total_rows = (
            db_session.query(func.count(FamilyTree.id)).scalar() or 0
        )  # Query total tree rows
        format_data = {
            "name": formatted_name,
            "predicted_relationship": (
                dna_match.predicted_relationship if dna_match else "N/A"
            ),
            "actual_relationship": (
                family_tree.actual_relationship if family_tree else "N/A"
            ),
            "relationship_path": (
                family_tree.relationship_path if family_tree else "N/A"
            ),
            "total_rows": total_rows,
        }
        try:
            message_text = message_template.format(**format_data)
        except KeyError as e:
            logger.error(
                f"Missing key '{e}' in format_data for template '{message_to_send_key}', log_prefix {log_prefix}."
            )
            raise StopIteration("error")
        except Exception as e:
            logger.error(f"Formatting error for {log_prefix}: {e}")
            raise StopIteration("error")

        # --- Step 4: Send Message (or perform Dry Run) ---
        # Determine existing conversation ID from logs
        existing_conversation_id = (
            latest_out_log.conversation_id
            if latest_out_log
            else latest_in_log.conversation_id if latest_in_log else None
        )
        message_status, effective_conv_id = "error", None  # Initialize

        if config_instance.APP_MODE == "dry_run":
            logger.info(
                f"Dry Run: Would send '{message_to_send_key}' ({send_reason}) to {log_prefix}"
            )
            message_status = "typed (dry_run)"
            effective_conv_id = (
                existing_conversation_id or f"dryrun_{uuid.uuid4()}"
            )  # Generate dummy ID if needed
        elif config_instance.APP_MODE in ["production", "testing"]:
            # Call helper function to send message via API
            message_status, effective_conv_id = _send_message_via_api(
                session_manager,
                person,
                message_text,
                existing_conversation_id,
                log_prefix,
            )
        else:
            logger.warning(
                f"Skipping send for {log_prefix} due to APP_MODE: {config_instance.APP_MODE}"
            )
            raise StopIteration("skipped")  # Skipped by App Mode

        # --- Step 5: Prepare DB Updates based on outcome ---
        new_log_entry = None
        person_update = None
        # Proceed only if send/dry run was considered successful
        if message_status in ("delivered OK", "typed (dry_run)"):
            message_type_id_to_log = message_type_map.get(message_to_send_key)
            # Validate necessary IDs
            if not message_type_id_to_log or not effective_conv_id:
                logger.error(
                    f"Critical error preparing DB update for {log_prefix} (missing type ID or Conv ID)."
                )
                raise StopIteration("error")

            # Create new ConversationLog object
            current_time_for_db = datetime.now(timezone.utc)
            trunc_msg_content = message_text[
                : config_instance.MESSAGE_TRUNCATION_LENGTH
            ]
            new_log_entry = ConversationLog(
                conversation_id=effective_conv_id,
                direction=MessageDirectionEnum.OUT,
                people_id=person.id,
                latest_message_content=trunc_msg_content,
                latest_timestamp=current_time_for_db,
                ai_sentiment=None,  # OUT logs don't have AI sentiment
                message_type_id=message_type_id_to_log,
                script_message_status=message_status,
                updated_at=current_time_for_db,
            )

            # Check if a DESIST ACK was successfully sent (in live mode)
            if (
                message_to_send_key == "User_Requested_Desist"
                and message_status == "delivered OK"
            ):
                # Prepare person status update to ARCHIVE
                logger.info(
                    f"Staging Person status update to ARCHIVE for {log_prefix} (ACK sent)."
                )
                person_update = (person.id, PersonStatusEnum.ARCHIVE)
                status_string = "acked"  # Set return status
            else:
                status_string = (
                    "sent"  # Set return status for standard message/dry run ACK
                )
        else:  # Handle send failure
            status_string = "error"

        # Return prepared data and outcome status
        return new_log_entry, person_update, status_string

    except StopIteration as si:
        # Cleanly handle intentional skipping
        status_val = (
            si.value if si.value else "skipped"
        )  # Default to 'skipped' if no value
        logger.debug(f"StopIteration caught for {log_prefix}, status: {status_val}")
        return None, None, status_val  # Return the status indicating why it stopped
    except Exception as e:
        # Catch any unexpected errors during processing
        logger.error(f"Unexpected error processing {log_prefix}: {e}\n", exc_info=True)
        return None, None, "error"  # Return error status
# End of _process_single_person

#####################################################
# Main Function: send_messages_to_matches
#####################################################


def send_messages_to_matches(session_manager: SessionManager) -> bool:
    """
    V14.59: Sends messages based on ConversationLog state and Person status.
    - Uses helper functions for pre-fetching and processing.
    - Includes a progress bar.
    - Added diagnostic logging.
    """
    # --- Step 1: Prerequisites Checks ---
    if not session_manager:
        logger.error("SM required.")
        return False
    if login_status(session_manager) is not True:
        logger.error("Login failed.")
        return False
    if not session_manager.my_profile_id:
        logger.error("Own profile ID missing.")
        return False
    if not MESSAGE_TEMPLATES:
        logger.error("Message templates failed.")
        return False
    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()

    # --- Step 2: Initialization ---
    sent_count, acked_count, skipped_count, error_count = 0, 0, 0, 0
    db_logs_to_add: List[ConversationLog] = []
    person_updates: Dict[int, PersonStatusEnum] = {}
    progress_bar = None
    total_candidates = 0
    critical_db_error_occurred = False
    batch_num = 0
    db_commit_batch_size = config_instance.BATCH_SIZE
    max_to_send = config_instance.MAX_INBOX
    overall_success = True  # Assume success initially

    try:
        # --- Step 3: Get DB Session ---
        # Use context manager for session handling
        with session_manager.get_db_conn_context() as db_session:
            # *** DIAGNOSTIC LOGGING ***
            logger.debug("--- DIAG: Entered DB Session context manager ---")
            if not db_session:
                logger.error("Failed get DB conn inside context manager.")
                # Manually return False as yield won't happen
                # Need to handle this case better, maybe raise? For now, log and return.
                return False

            # --- Step 4: Pre-fetch Data ---
            logger.info("--- Starting Pre-fetching for Action 8 (Messaging) ---")
            (
                message_type_map,
                candidate_persons,
                latest_in_log_map,
                latest_out_log_map,
            ) = _prefetch_messaging_data(db_session)

            # *** DIAGNOSTIC LOGGING - Inspect Prefetch Results ***
            logger.debug("--- DIAG: Prefetch Results ---")
            logger.debug(
                f"  Type(message_type_map): {type(message_type_map)}, Len/Keys: {len(message_type_map.keys()) if isinstance(message_type_map, dict) else 'N/A'}"
            )
            logger.debug(
                f"  Type(candidate_persons): {type(candidate_persons)}, Len: {len(candidate_persons) if isinstance(candidate_persons, list) else 'N/A'}"
            )
            logger.debug(
                f"  Type(latest_in_log_map): {type(latest_in_log_map)}, Len/Keys: {len(latest_in_log_map.keys()) if isinstance(latest_in_log_map, dict) else 'N/A'}"
            )
            logger.debug(
                f"  Type(latest_out_log_map): {type(latest_out_log_map)}, Len/Keys: {len(latest_out_log_map.keys()) if isinstance(latest_out_log_map, dict) else 'N/A'}"
            )
            # *** END DIAGNOSTIC LOGGING ***

            # Validate pre-fetched data
            if (
                message_type_map is None
                or candidate_persons is None
                or latest_in_log_map is None
                or latest_out_log_map is None
            ):
                logger.error("Failed during pre-fetching data.")
                # No need to return False here, already inside context manager, let it exit
                raise Exception(
                    "Prefetching failed"
                )  # Raise exception to trigger rollback/finally
            if not candidate_persons:
                logger.info("No candidates found meeting criteria. Finishing.")
                # No need to return True here, let it exit context manager cleanly
                # Set overall_success = True if this is considered success
                overall_success = True  # No candidates is not an error
                # Need to skip the loop below
                total_candidates = 0  # Ensure loop doesn't run
            else:
                total_candidates = len(candidate_persons)
            logger.info("--- Pre-fetching Finished ---")

            # *** DIAGNOSTIC LOGGING - Before tqdm loop ***
            logger.debug(
                f"--- DIAG: About to enter tqdm loop (Total Candidates: {total_candidates}) ---"
            )
            # *** END DIAGNOSTIC LOGGING ***

            # --- Step 5: Main Processing Loop with Progress Bar ---
            if total_candidates > 0:  # Only run loop if candidates exist
                with logging_redirect_tqdm():
                    # Initialize progress bar
                    progress_bar = tqdm(
                        total=total_candidates,
                        desc="Sending Messages",
                        unit=" match",
                        ncols=100,
                        bar_format="{percentage:3.0f}%|{bar}| Sent: {postfix[Sent]} | Ack: {postfix[Ack]} | Skip: {postfix[Skip]} | Err: {postfix[Err]}",
                        leave=True,
                        postfix={"Sent": 0, "Ack": 0, "Skip": 0, "Err": 0},
                    )

                    for person in candidate_persons:
                        # Step 5a: Check for critical DB error flag
                        if critical_db_error_occurred:
                            logger.warning(
                                "Aborting loop due to previous critical DB error."
                            )
                            break

                        # Step 5b: Check Send Limit
                        if (
                            max_to_send != 0
                            and (sent_count + acked_count) >= max_to_send
                        ):
                            logger.info(
                                f"Reached MAX_INBOX limit ({max_to_send}). Stopping loop."
                            )
                            break

                        # Step 5c: Process the individual person using helper function
                        new_log, person_update, status = _process_single_person(
                            db_session,
                            session_manager,
                            person,
                            latest_in_log_map.get(person.id),
                            latest_out_log_map.get(person.id),
                            message_type_map,
                        )

                        # Step 5d: Tally results and stage DB updates
                        if status == "sent":
                            sent_count += 1
                            if new_log:
                                db_logs_to_add.append(new_log)
                        elif status == "acked":
                            acked_count += 1
                            if new_log:
                                db_logs_to_add.append(new_log)
                            if person_update:
                                person_updates[person_update[0]] = person_update[1]
                        elif status == "skipped":
                            skipped_count += 1
                        else:  # 'error'
                            error_count += 1
                            overall_success = (
                                False  # Mark overall as failed if any error occurs
                            )

                        # Step 5e: Update progress bar display
                        progress_bar.set_postfix(
                            Sent=sent_count,
                            Ack=acked_count,
                            Skip=skipped_count,
                            Err=error_count,
                            refresh=False,
                        )
                        progress_bar.update(1)

                        # Step 5f: Commit DB changes periodically
                        if (
                            len(db_logs_to_add) + len(person_updates)
                            >= db_commit_batch_size
                        ):
                            batch_num += 1
                            commit_ok = _commit_messaging_batch(
                                db_session, db_logs_to_add, person_updates, batch_num
                            )
                            if commit_ok:
                                # Clear lists only on successful commit
                                db_logs_to_add.clear()
                                person_updates.clear()
                            else:
                                # Handle critical commit failure
                                logger.critical(
                                    f"CRITICAL: Batch commit {batch_num} failed. Aborting processing."
                                )
                                critical_db_error_occurred = True
                                overall_success = False
                                break  # Exit inner loop immediately

                    # --- End Main Person Loop ---
                    if progress_bar:
                        progress_bar.close()  # Close progress bar after loop

                # --- Step 6: Final Commit ---
                # Check critical flag before final commit as well
                if not critical_db_error_occurred and (
                    db_logs_to_add or person_updates
                ):
                    batch_num += 1
                    final_commit_ok = _commit_messaging_batch(
                        db_session, db_logs_to_add, person_updates, batch_num
                    )
                    if not final_commit_ok:
                        logger.error("Final batch commit failed.")
                        overall_success = False  # Mark overall run as failed
            else:
                logger.info("Skipping processing loop as there are no candidates.")

    except Exception as outer_e:
        # --- Step 7: Handle Outer Exceptions ---
        logger.critical(
            f"CRITICAL: Unhandled error during message processing: {outer_e}",
            exc_info=True,
        )
        if progress_bar and not progress_bar.disable:
            progress_bar.close()  # Ensure bar closes on error
        overall_success = False  # Outer exception means failure
    finally:
        # --- Step 8: Final Summary Logging ---
        # Calculate processed count accurately based on loop iterations or total candidates if aborted early
        processed_count = sent_count + acked_count + skipped_count + error_count
        if progress_bar and processed_count < total_candidates:  # If loop exited early
            processed_count = progress_bar.n  # Use actual iterations completed

        if critical_db_error_occurred:  # Adjust counts if aborted by DB error
            remaining_unprocessed = total_candidates - processed_count
            if remaining_unprocessed > 0:
                logger.warning(
                    f"Adding {remaining_unprocessed} unprocessed items to error count due to critical DB abort."
                )
                error_count += remaining_unprocessed
                processed_count += (
                    remaining_unprocessed  # Ensure processed matches total candidates
                )

        logger.info("--- Message Sending Summary ----")
        logger.info(f"  Potential Candidates (Active/Desist): {total_candidates}")
        logger.info(f"  Processed in Detail:       {processed_count}")
        logger.info(f"  Standard Messages Sent/DryRun: {sent_count}")
        logger.info(f"  Desist ACKs Sent/DryRun:   {acked_count}")
        logger.info(f"  Skipped (Policy/Rule/Status): {skipped_count}")
        logger.info(f"  Errors (API/DB/etc.):    {error_count}")
        logger.info(f"  Overall Success:           {overall_success}")
        logger.info("---------------------------------\n")

    return overall_success
# End of send_messages_to_matches


# --- main() function for standalone testing (unchanged) ---
def main():
    """Main function for standalone testing of Action 8 (API version)."""
    from logging_config import setup_logging  # Keep local import

    # --- Setup Logging ---
    try:
        from config import config_instance  # Keep local import

        db_file_path = config_instance.DATABASE_FILE
        log_filename_only = db_file_path.with_suffix(".log").name
        global logger  # Ensure logger is treated as global
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
        import logging as pylogging  # Fallback imports

        print(f"CRITICAL: Error during logging setup: {log_setup_e}", file=sys.stderr)
        pylogging.basicConfig(level=pylogging.DEBUG)
        logger = pylogging.getLogger("Action8Fallback")
        logger.info(f"--- Starting Action 8 Standalone Test (Fallback Logging) ---")
        logger.error(f"Initial logging setup failed: {log_setup_e}", exc_info=True)

    session_manager = SessionManager()
    action_success = False  # Initialize action_success

    try:
        logger.info("Attempting to start session...")
        # Phase 1 call returns a single boolean
        start_ok = session_manager.start_sess(action_name="Action 8 Test - Phase 1")
        if start_ok:
            logger.info("Phase 1 OK. Ensuring session ready (Phase 2)...")
            # Phase 2 call returns a single boolean
            ready_ok = session_manager.ensure_session_ready(
                action_name="Action 8 Test - Phase 2"
            )
            if ready_ok:
                logger.info("Phase 2 OK. Proceeding to send_messages_to_matches...")
                # *** ADDED DEBUG LOGGING BEFORE CALL ***
                logger.debug("<<< ABOUT TO CALL send_messages_to_matches >>>")
                # Action 8 call returns a single boolean
                action_success = send_messages_to_matches(session_manager)
                # *** ADDED DEBUG LOGGING AFTER CALL ***
                logger.debug(
                    f"<<< RETURNED FROM send_messages_to_matches with result: {action_success} >>>"
                )

                if action_success:  # Simple boolean check
                    logger.info("send_messages_to_matches completed successfully.")
                else:  # Simple boolean check
                    logger.error("send_messages_to_matches reported errors/failed.")
            else:
                logger.critical("Failed Phase 2 (Session Ready). Cannot run messaging.")
                action_success = False  # Assign boolean
        else:
            logger.critical("Failed Phase 1 (Driver Start). Cannot run messaging.")
            action_success = False  # Assign boolean
    except Exception as e:
        # The error is caught here
        logger.critical(
            f"Critical error in Action 8 standalone main: {e}", exc_info=True
        )
        action_success = False  # Assign boolean
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
