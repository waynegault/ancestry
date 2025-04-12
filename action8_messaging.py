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
    _send_message_via_api
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
    V14.71: Determines the next standard message type based on history and tree status.
    - Skips if the last message sent was User_Requested_Desist.
    """
    # Step 1: Log inputs
    logger.debug(f"Determining next msg:")
    logger.debug(f"  In Tree: {is_in_family_tree}")
    logger.debug(f"  Last Msg Details: {last_message_details}")  # Log received details

    # Step 2: Handle initial message case (no previous script message)
    if not last_message_details:
        next_type = "In_Tree-Initial" if is_in_family_tree else "Out_Tree-Initial"
        logger.debug(f"  Result: {next_type} (Reason: No prior script message).")
        return next_type

    # Step 3: Unpack details of the last script message sent
    last_message_type, last_sent_at, last_message_status = last_message_details
    next_type: Optional[str] = None
    skip_reason: str = "End of sequence or other condition met"
    logger.debug(
        f"  Last Msg Type: {last_message_type}, Sent: {last_sent_at}, Status: {last_message_status}"
    )

    # --- ADDED: Step 3.5: Check if last message was Desist ACK ---
    if last_message_type == "User_Requested_Desist":
        skip_reason = "Skipping standard message: Last sent was Desist ACK."
        logger.debug(f"  Skip Reason: {skip_reason}.")
        return None  # Do not proceed with standard sequence after sending ACK
    # --- END ADDED Check ---

    # Step 4: Determine next message based on current tree status and last message type
    if is_in_family_tree:
        # --- Logic for IN_TREE matches ---
        if last_message_type.startswith("Out_Tree"):
            next_type = (
                "In_Tree-Initial_for_was_Out_Tree"  # Handle switch from Out to In
            )
            logger.debug("  Reason: Match was Out_Tree, now In_Tree.")
        elif (
            last_message_type == "In_Tree-Initial"
            or last_message_type == "In_Tree-Initial_for_was_Out_Tree"
        ):
            next_type = "In_Tree-Follow_Up"  # Follow up initial In_Tree message
            logger.debug(f"  Reason: Following up on {last_message_type}.")
        elif last_message_type == "In_Tree-Follow_Up":
            next_type = "In_Tree-Final_Reminder"  # Send final reminder
            logger.debug("  Reason: Sending final In_Tree reminder.")
        elif last_message_type == "In_Tree-Final_Reminder":
            skip_reason = f"End of In_Tree sequence (last was {last_message_type})"  # End of sequence
            logger.debug(f"  Skip Reason: {skip_reason}.")
        else:
            # Handle other cases (like Unknown type - Desist ACK handled above)
            if last_message_type == "Unknown":
                skip_reason = "Previous message type was Unknown."
                logger.warning(f"  Skip Reason: {skip_reason} for In_Tree match.")
            else:
                skip_reason = f"Unexpected previous In_Tree type: {last_message_type}"
                logger.warning(f"  Skip Reason: {skip_reason}.")
    else:
        # --- Logic for OUT_OF_TREE matches ---
        if last_message_type.startswith("In_Tree"):
            if last_message_type == "In_Tree-Initial_for_was_Out_Tree":
                # Handle switch Out -> In -> Out: Restart Out sequence
                logger.warning(
                    f"  Reason: Match was Out->In->Out. Restarting Out sequence."
                )
                next_type = "Out_Tree-Initial"
            else:
                # Skip if they were In_Tree but are now Out_Tree
                skip_reason = (
                    f"Match was In_Tree ({last_message_type}) but is now Out_Tree"
                )
                logger.warning(
                    f"  Skip Reason: {skip_reason}. Skipping standard message."
                )
        elif last_message_type == "Out_Tree-Initial":
            next_type = "Out_Tree-Follow_Up"  # Follow up initial Out_Tree message
            logger.debug("  Reason: Following up on Out_Tree-Initial.")
        elif last_message_type == "Out_Tree-Follow_Up":
            next_type = "Out_Tree-Final_Reminder"  # Send final reminder
            logger.debug("  Reason: Sending final Out_Tree reminder.")
        elif last_message_type == "Out_Tree-Final_Reminder":
            skip_reason = f"End of Out_Tree sequence (last was {last_message_type})"  # End of sequence
            logger.debug(f"  Skip Reason: {skip_reason}.")
        else:
            # Handle other cases (like Unknown type - Desist ACK handled above)
            if last_message_type == "Unknown":
                skip_reason = "Previous message type was Unknown."
                logger.warning(f"  Skip Reason: {skip_reason} for Out_Tree match.")
            else:
                skip_reason = f"Unexpected previous Out_Tree type: {last_message_type}"
                logger.warning(f"  Skip Reason: {skip_reason}.")

    # Step 5: Log the final decision and return the next message type key or None
    if next_type:
        logger.debug(f"  Final Decision: Send '{next_type}'.")
    else:
        logger.debug(f"  Final Decision: Skip ({skip_reason}).")
    return next_type
# End of determine_next_message_type

#####################################################
# Helper Functions
#####################################################


def _commit_messaging_batch(
    session: DbSession,
    logs_to_add: List[ConversationLog],
    person_updates: Dict[int, PersonStatusEnum],
    batch_num: int,
) -> bool:
    """
    V14.65: Commits logs using add_all and bulk_update_mappings.
    (Reverted to V14.61 logic)
    """
    if not logs_to_add and not person_updates:
        return True

    logger.debug(
        f"Attempting batch commit (Batch {batch_num}): {len(logs_to_add)} logs, {len(person_updates)} persons..."
    )

    try:
        with db_transn(session):
            if logs_to_add:
                logger.debug(
                    f" Adding {len(logs_to_add)} new ConversationLog entries..."
                )
                session.add_all(logs_to_add)  # Use add_all for new entries

            if person_updates:
                update_mappings = [
                    {
                        "id": pid,
                        "status": status_enum,
                        "updated_at": datetime.now(timezone.utc),
                    }
                    for pid, status_enum in person_updates.items()
                ]
                logger.debug(f" Updating {len(update_mappings)} Person statuses...")
                session.bulk_update_mappings(Person, update_mappings)

            logger.debug("Session state BEFORE commit attempt:")
            logger.debug(f"  Dirty objects: {len(session.dirty)}")
            logger.debug(f"  New objects: {len(session.new)}")
            logger.debug(f"  Deleted objects: {len(session.deleted)}")

        logger.debug(f"Batch commit successful (Batch {batch_num}).")
        return True

    except IntegrityError as ie:
        logger.error(
            f"DB UNIQUE constraint error during messaging batch commit (Batch {batch_num}): {ie}",
            exc_info=False,
        )
        return False
    except Exception as e:
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
    """
    V14.71: Fetches all necessary data from the database before processing messages.
    - Eager loads Person.status.
    - Eager loads ConversationLog.message_type.
    - Filters for ACTIVE or DESIST status only.
    """
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
                # No need to load ConversationLog here, fetched separately below
            )
            .filter(
                Person.profile_id.isnot(None),  # Must have profile ID
                Person.profile_id != "",  # Must not be empty
                Person.profile_id != "UNKNOWN",  # Must not be unknown
                Person.contactable == True,  # Must be contactable
                Person.status.in_(
                    [PersonStatusEnum.ACTIVE, PersonStatusEnum.DESIST]
                ),  # Filter by status: ACTIVE or DESIST
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
            # --- Ensure timestamp is timezone-aware ---
            if log.latest_timestamp and log.latest_timestamp.tzinfo is None:
                log.latest_timestamp = log.latest_timestamp.replace(tzinfo=timezone.utc)
            # --- End timestamp check ---

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


def _process_single_person(
    db_session: DbSession,
    session_manager: SessionManager,
    person: Person,
    latest_in_log: Optional[ConversationLog],
    latest_out_log: Optional[ConversationLog],
    message_type_map: Dict[str, int],
) -> Tuple[Optional[ConversationLog], Optional[Tuple[int, PersonStatusEnum]], str]:
    """
    V14.71: Processes a single person for messaging.
    - Checks Person status (ACTIVE, DESIST). Skips ARCHIVE.
    - Handles sending Desist ACK if status is DESIST and ACK not sent.
    - Determines standard next message if status is ACTIVE.
    - Prepares log entry (new or update) and potential Person status update to ARCHIVE.
    - Returns new log entry (if created), person update tuple, and status string.
    """
    log_prefix = f"{person.username} #{person.id} (Status: {person.status.name})"
    message_to_send_key: Optional[str] = None
    send_reason = "Unknown"
    status_string = "error"  # Default return status
    new_log_entry = (
        None  # Initialize log entry as None (will be set if new one created)
    )
    person_update: Optional[Tuple[int, PersonStatusEnum]] = (
        None  # Initialize person update
    )

    try:
        # --- Step 0: Log Input Log State ---
        logger.debug(f"--- Processing Person: {log_prefix} ---")
        # ...(logging unchanged)...
        if latest_in_log:
            logger.debug(
                f"  Latest IN Log: Present (TS: {latest_in_log.latest_timestamp}, Sentiment: {latest_in_log.ai_sentiment})"
            )
        else:
            logger.debug(f"  Latest IN Log: None")
        if latest_out_log:
            msg_type_name = (
                getattr(latest_out_log.message_type, "type_name", "Unknown")
                if latest_out_log.message_type
                else "None"
            )
            logger.debug(
                f"  Latest OUT Log: Present (TS: {latest_out_log.latest_timestamp}, Type: {msg_type_name}, Status: {latest_out_log.script_message_status})"
            )
        else:
            logger.debug(f"  Latest OUT Log: None")

        # --- Step 1: Rule Checks based on Status and History ---
        # Skip ARCHIVED people immediately
        if person.status == PersonStatusEnum.ARCHIVE:
            logger.debug(f"Skipping {log_prefix}: Status is ARCHIVE.")
            raise StopIteration("skipped")
        # Skip BLOCKED/DEAD if they were somehow fetched (shouldn't be with current filter)
        if person.status in (PersonStatusEnum.BLOCKED, PersonStatusEnum.DEAD):
            logger.debug(f"Skipping {log_prefix}: Status is {person.status.name}.")
            raise StopIteration("skipped")

        # --- Step 2: Determine Message Key based on Person Status ---
        if person.status == PersonStatusEnum.DESIST:
            # --- Handle DESIST status: Send ACK if not already sent ---
            logger.debug(f"{log_prefix}: Status is DESIST. Checking if ACK needed.")
            desist_ack_type_id = message_type_map.get("User_Requested_Desist")
            if not desist_ack_type_id:  # Should not happen if prefetch worked
                logger.critical(
                    "CRITICAL: User_Requested_Desist MessageType ID missing from map."
                )
                raise StopIteration("error")

            # Check if the last OUT message was the Desist ACK
            ack_already_sent = False
            if latest_out_log and latest_out_log.message_type_id == desist_ack_type_id:
                # Check status as well? Maybe only count if 'delivered OK' or 'typed (dry_run)'?
                # For now, just check if the type matches.
                ack_already_sent = True

            if ack_already_sent:
                logger.debug(f"Skipping {log_prefix}: Desist ACK already sent.")
                raise StopIteration("skipped")
            else:
                logger.debug(f"Action required for {log_prefix}: Send Desist ACK.")
                message_to_send_key = "User_Requested_Desist"
                send_reason = "DESIST Acknowledgment"

        elif person.status == PersonStatusEnum.ACTIVE:
            # --- Handle ACTIVE status: Determine next standard message ---
            logger.debug(
                f"{log_prefix}: Status is ACTIVE. Determining next standard message."
            )
            # Check standard rules (reply received, time interval)
            reply_received_after_last_out = False
            if latest_in_log and latest_out_log:
                if latest_in_log.latest_timestamp > latest_out_log.latest_timestamp:
                    reply_received_after_last_out = True
            elif (
                latest_in_log and not latest_out_log
            ):  # Received reply but never sent anything
                reply_received_after_last_out = True

            if reply_received_after_last_out:
                logger.debug(
                    f"Skipping {log_prefix}: Reply received after last script message."
                )
                raise StopIteration("skipped")

            # Check time interval since last OUT message
            if latest_out_log and latest_out_log.latest_timestamp:
                # Ensure comparison is timezone-aware
                now_utc = datetime.now(timezone.utc)
                last_out_ts_utc = (
                    latest_out_log.latest_timestamp
                )  # Assumed already aware from prefetch
                time_since_last = now_utc - last_out_ts_utc
                if time_since_last < MIN_MESSAGE_INTERVAL:
                    logger.debug(
                        f"Skipping {log_prefix}: Interval not met ({time_since_last} < {MIN_MESSAGE_INTERVAL})."
                    )
                    raise StopIteration("skipped")
                else:
                    logger.debug(
                        f"Interval met ({time_since_last} >= {MIN_MESSAGE_INTERVAL})."
                    )
            else:
                logger.debug(
                    f"{log_prefix}: No previous OUT message, interval check skipped."
                )

            # Determine next message type based on history
            last_script_message_details: Optional[Tuple[str, datetime, str]] = None
            if latest_out_log and latest_out_log.message_type:
                last_type_name = getattr(
                    latest_out_log.message_type, "type_name", "Unknown"
                )
                last_status = latest_out_log.script_message_status or "Unknown"
                # Ensure timestamp is datetime for determine_next_message_type
                last_ts = latest_out_log.latest_timestamp  # Already timezone-aware
                last_script_message_details = (last_type_name, last_ts, last_status)

            message_to_send_key = determine_next_message_type(
                last_script_message_details, bool(person.in_my_tree)
            )
            if not message_to_send_key:
                logger.debug(
                    f"Skipping {log_prefix}: No appropriate next standard message."
                )
                raise StopIteration("skipped")
            send_reason = "Standard Sequence"
        else:
            # Should not happen if prefetch filter works
            logger.error(
                f"Unexpected person status {person.status.name} for {log_prefix}. Skipping."
            )
            raise StopIteration("skipped")

        # --- Step 3: Format Message ---
        if not message_to_send_key:  # Should have been caught by now, but safety check
            logger.error(f"Logic Error: message_to_send_key is None for {log_prefix}.")
            raise StopIteration("error")

        message_template = MESSAGE_TEMPLATES.get(message_to_send_key)
        if not message_template:
            logger.error(
                f"Template missing for key '{message_to_send_key}' for {log_prefix}."
            )
            raise StopIteration("error")

        # ...(Formatting unchanged)...
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
        # Fetch total rows count - consider caching this if it's slow
        total_rows = 0
        try:
            total_rows = db_session.query(func.count(FamilyTree.id)).scalar() or 0
        except Exception as count_e:
            logger.warning(
                f"Could not get FamilyTree count for message formatting: {count_e}"
            )

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
        except KeyError as ke:
            logger.error(
                f"Missing key {ke} in format_data for template '{message_to_send_key}' for {log_prefix}"
            )
            raise StopIteration("error")
        except Exception as e:
            logger.error(f"Formatting error for {log_prefix}: {e}")
            raise StopIteration("error")

        # --- Step 4: Send Message (or perform Dry Run) ---
        existing_conversation_id = (
            latest_out_log.conversation_id
            if latest_out_log
            else latest_in_log.conversation_id if latest_in_log else None
        )
        message_status, effective_conv_id = "error", None

        if config_instance.APP_MODE == "dry_run":
            message_status = "typed (dry_run)"
            effective_conv_id = existing_conversation_id or f"dryrun_{uuid.uuid4()}"
            logger.debug(
                f"Dry Run: Would send '{message_to_send_key}' ({send_reason}) to {log_prefix}"
            )
        elif config_instance.APP_MODE in ["production", "testing"]:
            logger.debug(
                f"Attempting to send '{message_to_send_key}' ({send_reason}) to {log_prefix}"
            )
            message_status, effective_conv_id = _send_message_via_api(
                session_manager,
                person,
                message_text,
                existing_conversation_id,
                log_prefix,
            )
        else:  # Should not happen
            logger.error(f"Invalid APP_MODE '{config_instance.APP_MODE}' for sending.")
            raise StopIteration("skipped")

        # --- Step 5: Prepare DB Updates ---
        if message_status in ("delivered OK", "typed (dry_run)"):
            message_type_id_to_log = message_type_map.get(message_to_send_key)
            # Validate IDs
            if not message_type_id_to_log:
                logger.error(
                    f"CRITICAL: Could not find MessageType ID for key '{message_to_send_key}' for {log_prefix}."
                )
                raise StopIteration("error")
            if not effective_conv_id:
                logger.error(
                    f"CRITICAL: effective_conv_id is missing after successful send/dry-run for {log_prefix}."
                )
                raise StopIteration("error")
            current_time_for_db = datetime.now(timezone.utc)
            trunc_msg_content = message_text[
                : config_instance.MESSAGE_TRUNCATION_LENGTH
            ]
            # --- Logic to Update Existing or Create New OUT Log ---
            if latest_out_log and latest_out_log.conversation_id == effective_conv_id:
                # UPDATE existing OUT log
                logger.debug(
                    f"Preparing update for existing OUT log ConvID={effective_conv_id}"
                )
                latest_out_log.latest_message_content = trunc_msg_content
                latest_out_log.latest_timestamp = current_time_for_db
                latest_out_log.message_type_id = message_type_id_to_log
                latest_out_log.script_message_status = message_status
                latest_out_log.updated_at = current_time_for_db
                # Add the modified existing object to the session - let commit handler manage this
                # db_session.add(latest_out_log) # REMOVED - Rely on session tracking changes
                new_log_entry = None  # Indicate no *new* log object created
            else:
                # CREATE new OUT log
                logger.debug(f"Preparing new OUT log for ConvID={effective_conv_id}")
                new_log_entry = ConversationLog(
                    conversation_id=effective_conv_id,
                    direction=MessageDirectionEnum.OUT,
                    people_id=person.id,
                    latest_message_content=trunc_msg_content,
                    latest_timestamp=current_time_for_db,
                    ai_sentiment=None,  # AI sentiment not applicable for OUT logs
                    message_type_id=message_type_id_to_log,
                    script_message_status=message_status,
                    # created_at is handled by default
                    updated_at=current_time_for_db,  # Set updated_at explicitly
                )
            # --- Set person status update to ARCHIVE if Desist ACK sent ---
            if (
                message_to_send_key == "User_Requested_Desist"
                # Only update status if message was actually delivered (or dry run)
                and message_status in ("delivered OK", "typed (dry_run)")
            ):
                logger.debug(
                    f"Staging Person status update to ARCHIVE for {log_prefix} (ACK sent/dry-run)."
                )
                person_update = (person.id, PersonStatusEnum.ARCHIVE)
                status_string = "acked"  # Specific status for ACK
            else:
                status_string = "sent"  # Standard sent/dry-run status
        else:  # Handle send failure
            logger.warning(
                f"Message send failed for {log_prefix} with status '{message_status}'."
            )
            status_string = "error"
            # Do not prepare log or person update on send error

        # Return prepared data and outcome status
        return new_log_entry, person_update, status_string

    except StopIteration as si:
        status_val = si.value if si.value else "skipped"
        # logger.debug(f"StopIteration caught for {log_prefix}, status: {status_val}") # Less verbose
        return None, None, status_val
    except Exception as e:
        logger.error(f"Unexpected error processing {log_prefix}: {e}\n", exc_info=True)
        return None, None, "error"
# End of _process_single_person


#####################################################
# Main Functions: send_messages_to_matches
#####################################################


def send_messages_to_matches(session_manager: SessionManager) -> bool:
    """
    V14.71: Sends messages based on ConversationLog state and Person status.
    - Fetches required data using _prefetch_messaging_data.
    - Iterates through candidates, calling _process_single_person.
    - Collects log entries and person updates for batch commit.
    - Handles Person status updates (ACTIVE -> ARCHIVE after ACK).
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

    # --- Step 2: Initialization ---
    sent_count, acked_count, skipped_count, error_count = 0, 0, 0, 0
    db_logs_to_add: List[ConversationLog] = []
    person_updates: Dict[int, PersonStatusEnum] = (
        {}
    )  # Maps person_id -> new Status Enum
    progress_bar = None
    total_candidates = 0
    critical_db_error_occurred = False
    batch_num = 0
    db_commit_batch_size = config_instance.BATCH_SIZE
    max_to_send = config_instance.MAX_INBOX  # Max standard + ACK messages combined
    overall_success = True
    pbar_total = 0
    processed_in_loop = 0  # Track actual loop iterations

    try:
        # --- Step 3: Get DB Session & Pre-fetch Data ---
        with session_manager.get_db_conn_context() as db_session:
            logger.debug("--- DIAG: Entered DB Session context manager ---")
            if not db_session:
                raise Exception("DB Session Error")

            logger.debug("--- Starting Pre-fetching for Action 8 (Messaging) ---")
            (
                message_type_map,
                candidate_persons,
                latest_in_log_map,
                latest_out_log_map,
            ) = _prefetch_messaging_data(db_session)

            if (  # Check if any prefetch failed
                message_type_map is None
                or candidate_persons is None
                or latest_in_log_map is None
                or latest_out_log_map is None
            ):
                logger.error("Prefetching essential data failed. Aborting.")
                return False  # Indicate failure

            total_candidates = len(candidate_persons)
            if total_candidates == 0:
                logger.info(
                    "No candidates found meeting criteria (ACTIVE/DESIST, Contactable). Finishing."
                )
                overall_success = True
            else:
                pbar_total = total_candidates
                logger.debug(
                    f"PBar Total initially set to {pbar_total} (total candidates)."
                )

            logger.debug("--- Pre-fetching Finished ---")
            logger.debug(
                f"--- DIAG: About to enter tqdm loop (Total Candidates: {total_candidates}, Initial PBar Total: {pbar_total}) ---"
            )

            # --- Step 4: Main Processing Loop with Progress Bar ---
            if pbar_total > 0:
                logger.info("Progress...\n")
                with logging_redirect_tqdm():
                    logger.debug("--- DIAG: Entered logging_redirect_tqdm context ---")
                    try:
                        progress_bar = tqdm(
                            total=pbar_total,
                            desc="Processing Candidates",
                            unit=" candidate",
                            ncols=100,
                            bar_format="{percentage:3.0f}%|{bar}|",  # Simplified format
                            leave=True,
                        )
                        logger.debug(
                            f"--- DIAG: tqdm progress bar initialized (Total={pbar_total}) ---"
                        )
                    except Exception as tqdm_e:
                        logger.error(f"Failed to initialize progress bar: {tqdm_e}")
                        progress_bar = None  # Continue without bar

                    for person in candidate_persons:
                        processed_in_loop += 1  # Increment loop counter at the start

                        # Check for critical DB error flag
                        if critical_db_error_occurred:
                            logger.warning(
                                "Aborting loop due to previous critical DB error."
                            )
                            break

                        # Check Send Limit BEFORE processing
                        # Limit applies to successfully sent standard messages AND ACKs
                        if (
                            max_to_send > 0
                            and (sent_count + acked_count) >= max_to_send
                        ):
                            logger.info(
                                f"Reached MAX_INBOX limit ({max_to_send}) for sent/acked messages. Stopping loop."
                            )
                            break

                        # Process the individual person
                        new_log, person_update, status = _process_single_person(
                            db_session,  # Pass session for potential OUT log updates
                            session_manager,
                            person,
                            latest_in_log_map.get(person.id),
                            latest_out_log_map.get(person.id),
                            message_type_map,
                        )

                        # Tally results based on status string
                        if status == "sent":
                            sent_count += 1
                        elif status == "acked":
                            acked_count += 1
                        elif status == "skipped":
                            skipped_count += 1
                        else:  # status == "error"
                            error_count += 1
                            overall_success = False  # Mark run as failed if any error

                        # Collect data for batch commit
                        if new_log:  # Only add if a *new* log object was created
                            db_logs_to_add.append(new_log)
                        if person_update:  # Collect person status updates
                            # person_update is (person_id, PersonStatusEnum.ARCHIVE)
                            person_updates[person_update[0]] = person_update[1]

                        # Update progress bar AFTER processing
                        if progress_bar:
                            progress_bar.update(1)

                        # Commit periodically or if data threshold reached
                        if (
                            len(db_logs_to_add) + len(person_updates)
                            >= db_commit_batch_size
                        ):
                            batch_num += 1
                            commit_ok = _commit_messaging_batch(
                                db_session, db_logs_to_add, person_updates, batch_num
                            )
                            if commit_ok:
                                db_logs_to_add.clear()
                                person_updates.clear()
                            else:
                                logger.critical(
                                    f"CRITICAL: Batch commit {batch_num} failed. Aborting processing."
                                )
                                critical_db_error_occurred = True
                                overall_success = False
                                break  # Exit inner loop on critical commit failure
                    # --- End Main Person Loop ---
            else:
                logger.info("Skipping processing loop as there are no candidates.")

            # --- Step 5: Final Commit ---
            if not critical_db_error_occurred and (db_logs_to_add or person_updates):
                batch_num += 1
                logger.info(f"Performing final commit (Batch {batch_num})...")
                final_commit_ok = _commit_messaging_batch(
                    db_session, db_logs_to_add, person_updates, batch_num
                )
                if not final_commit_ok:
                    logger.error("Final batch commit failed.")
                    overall_success = False

    except Exception as outer_e:
        # --- Step 6: Handle Outer Exceptions ---
        logger.critical(
            f"CRITICAL: Unhandled error during message processing: {outer_e}",
            exc_info=True,
        )
        overall_success = False
    finally:
        # --- Step 7: Finalize Progress Bar & Log Summary ---
        if progress_bar:
            # Adjust total to match actual iterations if loop exited early
            if not progress_bar.disable and progress_bar.n < progress_bar.total:
                logger.debug(
                    f"Loop finished early at {progress_bar.n}/{progress_bar.total}. Adjusting bar total."
                )
                progress_bar.total = progress_bar.n  # Set total to current count
                progress_bar.refresh()  # Force refresh

            progress_bar.close()
            print("", file=sys.stderr)  # Add blank line after progress bar

        # Calculate final error count, accounting for early exit due to DB error
        error_count_final = error_count
        if critical_db_error_occurred and total_candidates > processed_in_loop:
            unaccounted = total_candidates - processed_in_loop
            logger.warning(
                f"Adding {unaccounted} unaccounted candidates (due to DB error exit) to error count for summary."
            )
            error_count_final += unaccounted

        logger.info("--- Message Sending Summary ----")
        logger.info(f"  Potential Candidates:        {total_candidates}")
        logger.info(f"  Processed (Iterated):      {processed_in_loop}")
        logger.info(f"  Standard Messages Sent/DryRun: {sent_count}")
        logger.info(f"  Desist ACKs Sent/DryRun:   {acked_count}")
        logger.info(f"  Skipped (Policy/Rule/Status): {skipped_count}")
        logger.info(f"  Errors (API/DB/Unaccounted): {error_count_final}")
        logger.info(f"  Overall Success:           {overall_success}")
        logger.info("---------------------------------\n")
    return overall_success
# End of send_messages_to_matches

#####################################################
# STand alone testing
#####################################################

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
