# File: action8_messaging.py

"""
action8_messaging.py - Send Standard Templated Messages

Handles the logic for sending predefined messages (Initial, Follow-up, Reminder,
Desist Acknowledgement) to DNA matches based on their status, tree linkage,
communication history, and configured time intervals. Uses templates from
messages.json and respects application mode (dry_run, testing, production)
for recipient filtering. Updates the database (`conversation_log`, `people`)
after sending/simulating messages.
"""

# --- Standard library imports ---
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
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, cast
from urllib.parse import urlencode, urljoin, urlparse

# --- Third-party imports ---
import requests
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Index,
    Integer,
    String,
    Subquery,
    and_,
    desc,
    func,
    inspect as sa_inspect,
    over,
    text,
    update,
    select as sql_select,
)  # Keep and_ import

# Corrected SQLAlchemy ORM imports
from sqlalchemy.orm import (
    Session,  # Use Session directly
    aliased,
    joinedload,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from tqdm.auto import tqdm  # Progress bar
from tqdm.contrib.logging import logging_redirect_tqdm  # Logging integration

# --- Local application imports ---
from cache import cache_result  # Caching utility
from config import config_instance, selenium_config  # Configuration singletons
from database import (  # Database models and utilities
    ConversationLog,
    DnaMatch,
    FamilyTree,
    MessageDirectionEnum,
    MessageType,
    Person,
    PersonStatusEnum,
    RoleType,
    db_transn,
    commit_bulk_data,
)
from logging_config import logger  # Use configured logger
from utils import (  # Core utilities
    DynamicRateLimiter,  # Rate limiter (accessed via SessionManager)
    SessionManager,
    _api_req,  # API request helper (unused directly here, via _send_message)
    format_name,  # Name formatting
    login_status,  # Login check utility
    retry,  # Decorators (unused here)
    retry_api,  # Decorators (unused here)
    time_wait,  # Decorators (unused here)
)
from api_utils import (  # API utilities
    call_send_message_api,  # Real API function for sending messages
    SEND_SUCCESS_DELIVERED,  # Status constants
    SEND_SUCCESS_DRY_RUN,
)


# --- Initialization & Template Loading ---
logger.info(f"Action 8 Initializing: APP_MODE is {config_instance.APP_MODE}")

# Define message intervals based on app mode (controls time between follow-ups)
MESSAGE_INTERVALS = {
    "testing": timedelta(seconds=10),  # Short interval for testing
    "production": timedelta(weeks=8),  # Standard interval for production
    "dry_run": timedelta(seconds=10),  # Short interval for dry runs
}
MIN_MESSAGE_INTERVAL: timedelta = MESSAGE_INTERVALS.get(
    config_instance.APP_MODE, timedelta(weeks=8)
)
logger.info(f"Action 8 Using minimum message interval: {MIN_MESSAGE_INTERVAL}")

# Define standard message type keys (must match messages.json)
MESSAGE_TYPES_ACTION8: Dict[str, str] = {
    "In_Tree-Initial": "In_Tree-Initial",
    "In_Tree-Follow_Up": "In_Tree-Follow_Up",
    "In_Tree-Final_Reminder": "In_Tree-Final_Reminder",
    "Out_Tree-Initial": "Out_Tree-Initial",
    "Out_Tree-Follow_Up": "Out_Tree-Follow_Up",
    "Out_Tree-Final_Reminder": "Out_Tree-Final_Reminder",
    "In_Tree-Initial_for_was_Out_Tree": "In_Tree-Initial_for_was_Out_Tree",
    # Note: Productive Reply ACK is handled by Action 9
    "User_Requested_Desist": "User_Requested_Desist",  # Handled here if Person status is DESIST
}


@cache_result("message_templates")  # Cache the loaded templates
def load_message_templates() -> Dict[str, str]:
    """
    Loads message templates from the 'messages.json' file.
    Validates that all required template keys for Action 8 are present.

    Returns:
        A dictionary mapping template keys (type names) to template strings.
        Returns an empty dictionary if loading or validation fails.
    """
    # Step 1: Define path to messages.json relative to this file's parent
    try:
        script_dir = Path(__file__).resolve().parent
        messages_path = script_dir / "messages.json"
        logger.debug(f"Attempting to load message templates from: {messages_path}")
    except Exception as path_e:
        logger.critical(f"CRITICAL: Could not determine script directory: {path_e}")
        return {}

    # Step 2: Check if file exists
    if not messages_path.exists():
        logger.critical(f"CRITICAL: messages.json not found at {messages_path}")
        return {}

    # Step 3: Read and parse the JSON file
    try:
        with messages_path.open("r", encoding="utf-8") as f:
            templates = json.load(f)

        # Step 4: Validate structure (must be dict of strings)
        if not isinstance(templates, dict) or not all(
            isinstance(v, str) for v in templates.values()
        ):
            logger.critical(
                "CRITICAL: messages.json content is not a valid dictionary of strings."
            )
            return {}

        # Step 5: Validate that all required keys for Action 8 exist
        # Note: We check against MESSAGE_TYPES_ACTION8 specifically
        required_keys = set(MESSAGE_TYPES_ACTION8.keys())
        # Add Productive ACK key as well, as it might be loaded here even if used in Action 9
        required_keys.add("Productive_Reply_Acknowledgement")
        missing_keys = required_keys - set(templates.keys())
        if missing_keys:
            logger.critical(
                f"CRITICAL: messages.json is missing required template keys: {', '.join(missing_keys)}"
            )
            return {}

        # Step 6: Log success and return templates
        logger.info(
            f"Message templates loaded and validated successfully ({len(templates)} total templates)."
        )
        return templates
    except json.JSONDecodeError as e:
        logger.critical(f"CRITICAL: Error decoding messages.json: {e}")
        return {}
    except Exception as e:
        logger.critical(
            f"CRITICAL: Unexpected error loading messages.json: {e}", exc_info=True
        )
        return {}


# End of load_message_templates

# Load templates into a global variable for easy access
MESSAGE_TEMPLATES: Dict[str, str] = load_message_templates()
# Critical check: exit if essential templates failed to load
# Check against Action 8 specific keys + the Productive ACK key
required_check_keys = set(MESSAGE_TYPES_ACTION8.keys())
required_check_keys.add("Productive_Reply_Acknowledgement")
if not MESSAGE_TEMPLATES or not all(
    key in MESSAGE_TEMPLATES for key in required_check_keys
):
    logger.critical(
        "Essential message templates failed to load. Cannot proceed reliably."
    )
    # Optionally: sys.exit(1) here if running standalone or want hard failure
    # For now, allow script to potentially continue but log critical error.


# ------------------------------------------------------------------------------
# Message Type Determination Logic
# ------------------------------------------------------------------------------


def determine_next_message_type(
    last_message_details: Optional[
        Tuple[str, datetime, str]
    ],  # (type_name, sent_at_utc, status)
    is_in_family_tree: bool,
) -> Optional[str]:
    """
    Determines the next standard message type key (from MESSAGE_TYPES_ACTION8)
    to send based on the last *script-sent* message and the match's current
    tree status.

    Args:
        last_message_details: A tuple containing details of the last OUT message
                              sent by the script (type name, timestamp, status),
                              or None if no script message has been sent yet.
                              Timestamp MUST be timezone-aware UTC.
        is_in_family_tree: Boolean indicating if the match is currently linked
                           in the user's family tree.

    Returns:
        The string key for the next message template to use (e.g., "In_Tree-Follow_Up"),
        or None if no standard message should be sent according to the sequence rules.
    """
    # Step 1: Log inputs for debugging
    logger.debug(f"Determining next message type:")
    logger.debug(f"  Is In Tree: {is_in_family_tree}")
    logger.debug(f"  Last Script Msg Details: {last_message_details}")

    # Step 2: Handle initial message case (no previous script message logged)
    if not last_message_details:
        next_type = (
            MESSAGE_TYPES_ACTION8["In_Tree-Initial"]
            if is_in_family_tree
            else MESSAGE_TYPES_ACTION8["Out_Tree-Initial"]
        )
        logger.debug(
            f"  Decision: Send '{next_type}' (Reason: No prior script message)."
        )
        return next_type

    # Step 3: Unpack details of the last script message
    last_message_type, last_sent_at_utc, last_message_status = last_message_details
    next_type: Optional[str] = None  # Initialize next type
    skip_reason: str = "End of sequence or other condition met"  # Default skip reason
    logger.debug(
        f"  Last Sent Type: '{last_message_type}', SentAt: {last_sent_at_utc}, Status: '{last_message_status}'"
    )

    # --- Add Check: If last message was Desist ACK, stop standard sequence ---
    if last_message_type == MESSAGE_TYPES_ACTION8["User_Requested_Desist"]:
        skip_reason = "Last script message was Desist ACK."
        logger.debug(f"  Decision: Skip ({skip_reason})")
        return None  # Do not send further standard messages after ACK

    # Step 4: Determine next message based on tree status and last message type
    if is_in_family_tree:
        # --- Logic for IN_TREE matches ---
        if last_message_type.startswith("Out_Tree"):
            # Switched from Out-of-Tree to In-Tree: Send specific initial message
            next_type = MESSAGE_TYPES_ACTION8["In_Tree-Initial_for_was_Out_Tree"]
            logger.debug("  Reason: Match was Out_Tree, now In_Tree.")
        elif last_message_type in [
            MESSAGE_TYPES_ACTION8["In_Tree-Initial"],
            MESSAGE_TYPES_ACTION8["In_Tree-Initial_for_was_Out_Tree"],
        ]:
            # Sent initial In-Tree message -> Send Follow-Up
            next_type = MESSAGE_TYPES_ACTION8["In_Tree-Follow_Up"]
            logger.debug(f"  Reason: Following up on '{last_message_type}'.")
        elif last_message_type == MESSAGE_TYPES_ACTION8["In_Tree-Follow_Up"]:
            # Sent Follow-Up -> Send Final Reminder
            next_type = MESSAGE_TYPES_ACTION8["In_Tree-Final_Reminder"]
            logger.debug("  Reason: Sending final In_Tree reminder.")
        elif last_message_type == MESSAGE_TYPES_ACTION8["In_Tree-Final_Reminder"]:
            # Sent Final Reminder -> End of sequence
            skip_reason = f"End of In_Tree sequence (last was {last_message_type})"
            logger.debug(f"  Skip Reason: {skip_reason}.")
        else:  # Handle unexpected previous type for In_Tree match
            skip_reason = f"Unexpected previous In_Tree type: '{last_message_type}'"
            logger.warning(f"  Decision: Skip ({skip_reason})")

    else:  # Not in family tree
        # --- Logic for OUT_OF_TREE matches ---
        if last_message_type.startswith("In_Tree"):
            # Match was previously In_Tree but is now Out_Tree
            if (
                last_message_type
                == MESSAGE_TYPES_ACTION8["In_Tree-Initial_for_was_Out_Tree"]
            ):
                # Specific case: Was Out -> In -> Out again. Restart Out sequence?
                logger.warning(
                    "  Reason: Match was Out->In->Out. Restarting Out_Tree sequence."
                )
                next_type = MESSAGE_TYPES_ACTION8["Out_Tree-Initial"]
            else:
                # General case: Was In_Tree, now Out_Tree. Usually stop messaging.
                skip_reason = (
                    f"Match was In_Tree ('{last_message_type}') but is now Out_Tree."
                )
                logger.warning(f"  Decision: Skip ({skip_reason})")
        elif last_message_type == MESSAGE_TYPES_ACTION8["Out_Tree-Initial"]:
            # Sent initial Out_Tree -> Send Follow-Up
            next_type = MESSAGE_TYPES_ACTION8["Out_Tree-Follow_Up"]
            logger.debug("  Reason: Following up on Out_Tree-Initial.")
        elif last_message_type == MESSAGE_TYPES_ACTION8["Out_Tree-Follow_Up"]:
            # Sent Follow-Up -> Send Final Reminder
            next_type = MESSAGE_TYPES_ACTION8["Out_Tree-Final_Reminder"]
            logger.debug("  Reason: Sending final Out_Tree reminder.")
        elif last_message_type == MESSAGE_TYPES_ACTION8["Out_Tree-Final_Reminder"]:
            # Sent Final Reminder -> End of sequence
            skip_reason = f"End of Out_Tree sequence (last was {last_message_type})"
            logger.debug(f"  Skip Reason: {skip_reason}.")
        else:  # Handle unexpected previous type for Out_Tree match
            skip_reason = f"Unexpected previous Out_Tree type: '{last_message_type}'"
            logger.warning(f"  Decision: Skip ({skip_reason})")

    # Step 5: Log final decision and return key or None
    if next_type:
        logger.debug(f"  Final Decision: Send '{next_type}'.")
    else:
        logger.debug(f"  Final Decision: Skip ({skip_reason}).")
    return next_type


# End of determine_next_message_type

# ------------------------------------------------------------------------------
# Database and Processing Helpers
# ------------------------------------------------------------------------------


def _commit_messaging_batch(
    session: Session,
    logs_to_add: List[ConversationLog],  # List of ConversationLog OBJECTS
    person_updates: Dict[int, PersonStatusEnum],  # Dict of {person_id: new_status_enum}
    batch_num: int,
) -> bool:
    """
    Commits a batch of ConversationLog entries (OUT direction) and Person status updates
    to the database. Uses bulk insert for new logs and individual updates for existing ones.

    Args:
        session: The active SQLAlchemy database session.
        logs_to_add: List of fully populated ConversationLog objects to add/update.
        person_updates: Dictionary mapping Person IDs to their new PersonStatusEnum.
        batch_num: The current batch number (for logging).

    Returns:
        True if the commit was successful, False otherwise.
    """
    # Step 1: Check if there's anything to commit
    if not logs_to_add and not person_updates:
        logger.debug(f"Batch Commit (Msg Batch {batch_num}): No data to commit.")
        return True

    logger.debug(
        f"Attempting batch commit (Msg Batch {batch_num}): {len(logs_to_add)} logs, {len(person_updates)} persons..."
    )

    # Step 2: Perform DB operations within a transaction context
    try:
        with db_transn(session) as sess:  # Use the provided session in the transaction
            log_inserts_data = []
            log_updates_to_process = (
                []
            )  # List to hold tuples: (existing_log_obj, new_log_obj)

            # --- Step 2a: Prepare ConversationLog data: Separate Inserts and Updates ---
            if logs_to_add:
                logger.debug(
                    f" Preparing {len(logs_to_add)} ConversationLog entries for upsert..."
                )
                # Extract unique keys from the input OBJECTS
                log_keys_to_check = set()
                valid_log_objects = []  # Store objects that have valid keys
                for log_obj in logs_to_add:
                    if log_obj.conversation_id and log_obj.direction:
                        log_keys_to_check.add(
                            (log_obj.conversation_id, log_obj.direction)
                        )
                        valid_log_objects.append(log_obj)
                    else:
                        logger.error(
                            f"Invalid log object data (Msg Batch {batch_num}): Missing key info. Skipping log object."
                        )

                # Query for existing logs matching the keys in this batch
                existing_logs_map: Dict[
                    Tuple[str, MessageDirectionEnum], ConversationLog
                ] = {}
                if log_keys_to_check:
                    existing_logs = (
                        sess.query(ConversationLog)
                        .filter(
                            tuple_(
                                ConversationLog.conversation_id,
                                ConversationLog.direction,
                            ).in_([(cid, denum) for cid, denum in log_keys_to_check])
                        )
                        .all()
                    )
                    existing_logs_map = {
                        (log.conversation_id, log.direction): log
                        for log in existing_logs
                        if log.direction
                    }
                    logger.debug(
                        f" Prefetched {len(existing_logs_map)} existing ConversationLog entries for batch."
                    )

                # Process each valid log object
                for log_object in valid_log_objects:
                    log_key = (log_object.conversation_id, log_object.direction)
                    existing_log = existing_logs_map.get(log_key)

                    if existing_log:
                        # Prepare for individual update by pairing existing and new objects
                        log_updates_to_process.append((existing_log, log_object))
                    else:
                        # Prepare for bulk insert by converting object to dict
                        try:
                            insert_map = {
                                c.key: getattr(log_object, c.key)
                                for c in sa_inspect(log_object).mapper.column_attrs
                                # Include None values as they might be valid states (e.g., ai_sentiment for OUT)
                            }
                            # Ensure Enums are handled if needed (SQLAlchemy mapping usually handles this)
                            if isinstance(
                                insert_map.get("direction"), MessageDirectionEnum
                            ):
                                insert_map["direction"] = insert_map["direction"].value
                            # Ensure timestamp is added if missing (should be set by caller)
                            if (
                                "latest_timestamp" not in insert_map
                                or insert_map["latest_timestamp"] is None
                            ):
                                logger.warning(
                                    f"Timestamp missing for new log ConvID {log_object.conversation_id}. Setting to now."
                                )
                                insert_map["latest_timestamp"] = datetime.now(
                                    timezone.utc
                                )
                            elif isinstance(insert_map["latest_timestamp"], datetime):
                                # Ensure TZ aware UTC
                                ts_val = insert_map["latest_timestamp"]
                                insert_map["latest_timestamp"] = (
                                    ts_val.astimezone(timezone.utc)
                                    if ts_val.tzinfo
                                    else ts_val.replace(tzinfo=timezone.utc)
                                )

                            log_inserts_data.append(insert_map)
                        except Exception as prep_err:
                            logger.error(
                                f"Error preparing new log object for bulk insert (Msg Batch {batch_num}, ConvID: {log_object.conversation_id}): {prep_err}",
                                exc_info=True,
                            )

                # --- Execute Bulk Insert ---
                if log_inserts_data:
                    logger.debug(
                        f" Attempting bulk insert for {len(log_inserts_data)} ConversationLog entries..."
                    )
                    try:
                        sess.bulk_insert_mappings(ConversationLog, log_inserts_data)
                        logger.debug(
                            f" Bulk insert mappings called for {len(log_inserts_data)} logs."
                        )
                    except IntegrityError as ie:
                        logger.warning(
                            f"IntegrityError during bulk insert (likely duplicate ConvID/Direction): {ie}. Some logs might not have been inserted."
                        )
                        # Need robust handling if this occurs - maybe skip or attempt update?
                    except Exception as bulk_err:
                        logger.error(
                            f"Error during ConversationLog bulk insert (Msg Batch {batch_num}): {bulk_err}",
                            exc_info=True,
                        )
                        raise  # Rollback transaction

                # --- Perform Individual Updates ---
                updated_individually_count = 0
                if log_updates_to_process:
                    logger.debug(
                        f" Processing {len(log_updates_to_process)} individual ConversationLog updates..."
                    )
                    for existing_log, new_data_obj in log_updates_to_process:
                        try:
                            has_changes = False
                            # Compare relevant fields from the new object against the existing one
                            fields_to_compare = [
                                "latest_message_content",
                                "latest_timestamp",
                                "message_type_id",
                                "script_message_status",
                                "ai_sentiment",
                            ]
                            for field in fields_to_compare:
                                new_value = getattr(new_data_obj, field, None)
                                old_value = getattr(existing_log, field, None)
                                # Handle timestamp comparison carefully (aware)
                                if field == "latest_timestamp":
                                    old_ts_aware = (
                                        old_value.astimezone(timezone.utc)
                                        if isinstance(old_value, datetime)
                                        and old_value.tzinfo
                                        else (
                                            old_value.replace(tzinfo=timezone.utc)
                                            if isinstance(old_value, datetime)
                                            else None
                                        )
                                    )
                                    new_ts_aware = (
                                        new_value.astimezone(timezone.utc)
                                        if isinstance(new_value, datetime)
                                        and new_value.tzinfo
                                        else (
                                            new_value.replace(tzinfo=timezone.utc)
                                            if isinstance(new_value, datetime)
                                            else None
                                        )
                                    )
                                    if new_ts_aware != old_ts_aware:
                                        setattr(existing_log, field, new_ts_aware)
                                        has_changes = True
                                elif new_value != old_value:
                                    setattr(existing_log, field, new_value)
                                    has_changes = True
                            # Update timestamp if any changes occurred
                            if has_changes:
                                existing_log.updated_at = datetime.now(timezone.utc)
                                updated_individually_count += 1
                        except Exception as update_err:
                            logger.error(
                                f"Error updating individual log ConvID {existing_log.conversation_id}/{existing_log.direction}: {update_err}",
                                exc_info=True,
                            )
                    logger.debug(
                        f" Finished {updated_individually_count} individual log updates."
                    )

            # --- Step 2b: Person Status Updates (Bulk Update - remains the same) ---
            if person_updates:
                update_mappings = []
                logger.debug(
                    f" Preparing {len(person_updates)} Person status updates..."
                )
                for pid, status_enum in person_updates.items():
                    if not isinstance(status_enum, PersonStatusEnum):
                        logger.warning(
                            f"Invalid status type '{type(status_enum)}' for Person ID {pid}. Skipping update."
                        )
                        continue
                    update_mappings.append(
                        {
                            "id": pid,
                            "status": status_enum,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                if update_mappings:
                    logger.debug(
                        f" Updating {len(update_mappings)} Person statuses via bulk..."
                    )
                    sess.bulk_update_mappings(Person, update_mappings)

            logger.debug(
                f" Exiting transaction block (Msg Batch {batch_num}, commit follows)."
            )
        # --- Transaction automatically commits here if no exceptions ---
        logger.debug(f"Batch commit successful (Msg Batch {batch_num}).")
        return True

    # Step 3/4: Handle exceptions during commit
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
    db_session: Session,  # Use Session type hint
) -> Tuple[
    Optional[Dict[str, int]],
    Optional[List[Person]],
    Optional[Dict[int, ConversationLog]],
    Optional[Dict[int, ConversationLog]],
]:
    """
    Fetches data needed for the messaging process in bulk to minimize DB queries.
    - Fetches MessageType name-to-ID mapping.
    - Fetches candidate Person records (ACTIVE or DESIST status, contactable=True).
    - Fetches the latest IN and OUT ConversationLog for each candidate.

    Args:
        db_session: The active SQLAlchemy database session.

    Returns:
        A tuple containing:
        - message_type_map (Dict[str, int]): Map of type_name to MessageType ID.
        - candidate_persons (List[Person]): List of Person objects meeting criteria.
        - latest_in_log_map (Dict[int, ConversationLog]): Map of people_id to latest IN log.
        - latest_out_log_map (Dict[int, ConversationLog]): Map of people_id to latest OUT log.
        Returns (None, None, None, None) if essential data fetching fails.
    """
    # Step 1: Initialize results
    message_type_map: Optional[Dict[str, int]] = None
    candidate_persons: Optional[List[Person]] = None
    latest_in_log_map: Dict[int, ConversationLog] = {}  # Use dict for direct lookup
    latest_out_log_map: Dict[int, ConversationLog] = {}  # Use dict for direct lookup
    logger.debug("--- Starting Pre-fetching for Action 8 (Messaging) ---")

    try:
        # Step 2: Fetch MessageType map
        logger.debug("Prefetching MessageType name-to-ID map...")
        message_types = db_session.query(MessageType.id, MessageType.type_name).all()
        message_type_map = {name: mt_id for mt_id, name in message_types}
        # Validate essential types exist (check against keys needed for this action)
        required_keys = set(MESSAGE_TYPES_ACTION8.keys())
        if not all(key in message_type_map for key in required_keys):
            missing = required_keys - set(message_type_map.keys())
            logger.critical(
                f"CRITICAL: Failed to fetch required MessageType IDs. Missing: {missing}"
            )
            return None, None, None, None
        logger.debug(f"Fetched {len(message_type_map)} MessageType IDs.")

        # Step 3: Fetch Candidate Persons
        logger.debug(
            "Prefetching candidate persons (Status ACTIVE or DESIST, Contactable=True)..."
        )
        candidate_persons = (
            db_session.query(Person)
            .options(
                joinedload(Person.dna_match),  # Eager load needed data for formatting
                joinedload(Person.family_tree),  # Eager load needed data for formatting
            )
            .filter(
                Person.profile_id.isnot(None),  # Ensure profile ID exists
                Person.profile_id != "UNKNOWN",
                Person.contactable == True,  # Only contactable people
                Person.status.in_(
                    [PersonStatusEnum.ACTIVE, PersonStatusEnum.DESIST]
                ),  # Eligible statuses
            )
            .order_by(Person.id)  # Consistent order
            .all()
        )
        logger.debug(f"Fetched {len(candidate_persons)} potential candidates.")
        if not candidate_persons:
            return message_type_map, [], {}, {}  # Return empty results if no candidates

        # Step 4: Fetch Latest Conversation Logs for candidates
        candidate_person_ids: List[int] = [
            p.id for p in candidate_persons if p.id is not None
        ]
        if not candidate_person_ids:  # Should have IDs if persons were fetched
            logger.warning("No valid Person IDs found from candidate query.")
            return message_type_map, candidate_persons, {}, {}

        logger.debug(
            f"Prefetching latest IN/OUT logs for {len(candidate_person_ids)} candidates..."
        )
        # Subquery to find max timestamp per person per direction
        latest_ts_subq = (
            db_session.query(
                ConversationLog.people_id,
                ConversationLog.direction,
                func.max(ConversationLog.latest_timestamp).label("max_ts"),
            )
            .filter(ConversationLog.people_id.in_(candidate_person_ids))
            .group_by(ConversationLog.people_id, ConversationLog.direction)
            .subquery("latest_ts_subq")  # Alias the subquery
        )
        # Join back to get the full log entry matching the max timestamp
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
        latest_logs: List[ConversationLog] = latest_logs_query.all()

        # Populate maps (ensure timestamps are aware UTC)
        min_aware_dt = datetime.min.replace(tzinfo=timezone.utc)  # For safety
        for log in latest_logs:
            # --- Ensure timestamp is timezone-aware ---
            log_ts = log.latest_timestamp
            if log_ts and log_ts.tzinfo is None:
                log.latest_timestamp = log_ts.replace(tzinfo=timezone.utc)
            elif log_ts:  # Already aware, ensure UTC
                log.latest_timestamp = log_ts.astimezone(timezone.utc)
            else:  # Handle potential None timestamp from DB?
                log.latest_timestamp = min_aware_dt  # Assign min datetime if None

            # Add to appropriate map based on direction
            if log.direction == MessageDirectionEnum.IN and log.people_id is not None:
                latest_in_log_map[log.people_id] = log
            elif (
                log.direction == MessageDirectionEnum.OUT and log.people_id is not None
            ):
                latest_out_log_map[log.people_id] = log

        logger.debug(f"Prefetched latest IN logs for {len(latest_in_log_map)} people.")
        logger.debug(
            f"Prefetched latest OUT logs for {len(latest_out_log_map)} people."
        )
        logger.debug("--- Pre-fetching Finished ---")

        # Step 5: Return all prefetched data
        return (
            message_type_map,
            candidate_persons,
            latest_in_log_map,
            latest_out_log_map,
        )

    # Step 6: Handle errors during prefetching
    except SQLAlchemyError as db_err:
        logger.error(f"DB error during messaging pre-fetching: {db_err}", exc_info=True)
        return None, None, None, None
    except Exception as e:
        logger.error(
            f"Unexpected error during messaging pre-fetching: {e}", exc_info=True
        )
        return None, None, None, None


# End of _prefetch_messaging_data


def _process_single_person(
    db_session: Session,  # Use Session type hint
    session_manager: SessionManager,
    person: Person,  # Prefetched Person object
    latest_in_log: Optional[ConversationLog],  # Prefetched latest IN log or None
    latest_out_log: Optional[ConversationLog],  # Prefetched latest OUT log or None
    message_type_map: Dict[str, int],  # Prefetched map
) -> Tuple[Optional[ConversationLog], Optional[Tuple[int, PersonStatusEnum]], str]:
    """
    Processes a single person to determine if a message should be sent,
    formats the message, sends/simulates it, and prepares database updates.

    Args:
        db_session: The active SQLAlchemy database session.
        session_manager: The active SessionManager instance.
        person: The Person object to process (with eager-loaded relationships).
        latest_in_log: The latest prefetched IN ConversationLog for this person.
        latest_out_log: The latest prefetched OUT ConversationLog for this person.
        message_type_map: Dictionary mapping message type names to their DB IDs.

    Returns:
        A tuple containing:
        - new_log_entry (Optional[ConversationLog]): The prepared OUT log object if a message was sent/simulated, else None.
        - person_update (Optional[Tuple[int, PersonStatusEnum]]): Tuple of (person_id, new_status) if status needs update, else None.
        - status_string (str): "sent", "acked", "skipped", or "error".
    """
    # --- Step 0: Initialization and Logging ---
    log_prefix = f"{person.username} #{person.id} (Status: {person.status.name if person.status else 'Unknown'})"
    message_to_send_key: Optional[str] = None  # Key from MESSAGE_TEMPLATES
    send_reason = "Unknown"  # Reason for sending/skipping
    status_string: Literal["sent", "acked", "skipped", "error"] = (
        "error"  # Default outcome
    )
    new_log_entry: Optional[ConversationLog] = None  # Prepared log object
    person_update: Optional[Tuple[int, PersonStatusEnum]] = None  # Staged status update
    now_utc = datetime.now(timezone.utc)  # Consistent timestamp for checks
    min_aware_dt = datetime.min.replace(tzinfo=timezone.utc)  # For comparisons

    logger.debug(f"--- Processing Person: {log_prefix} ---")
    # Optional: Log latest log details for debugging
    # if latest_in_log: logger.debug(f"  Latest IN: {latest_in_log.latest_timestamp} ({latest_in_log.ai_sentiment})") else: logger.debug("  Latest IN: None")
    # if latest_out_log: logger.debug(f"  Latest OUT: {latest_out_log.latest_timestamp} ({getattr(latest_out_log.message_type, 'type_name', 'N/A')}, {latest_out_log.script_message_status})") else: logger.debug("  Latest OUT: None")

    try:  # Main processing block for this person
        # --- Step 1: Check Person Status for Eligibility ---
        if person.status in (
            PersonStatusEnum.ARCHIVE,
            PersonStatusEnum.BLOCKED,
            PersonStatusEnum.DEAD,
        ):
            logger.debug(f"Skipping {log_prefix}: Status is '{person.status.name}'.")
            raise StopIteration("skipped (status)")  # Use StopIteration to exit cleanly

        # --- Step 2: Determine Action based on Status (DESIST vs ACTIVE) ---
        if person.status == PersonStatusEnum.DESIST:
            # Handle DESIST status: Send ACK if not already sent
            logger.debug(
                f"{log_prefix}: Status is DESIST. Checking if Desist ACK needed."
            )
            desist_ack_type_id = message_type_map.get("User_Requested_Desist")
            if not desist_ack_type_id:  # Should have been checked during prefetch
                logger.critical("CRITICAL: User_Requested_Desist ID missing map.")
                raise StopIteration("error (config)")
            # Check if the latest OUT message *was* the Desist ACK
            ack_already_sent = bool(
                latest_out_log and latest_out_log.message_type_id == desist_ack_type_id
            )
            if ack_already_sent:
                logger.debug(
                    f"Skipping {log_prefix}: Desist ACK already sent (Last OUT Type ID: {latest_out_log.message_type_id if latest_out_log else 'N/A'})."
                )
                # If ACK sent but status still DESIST, could change to ARCHIVE here or Action 9
                raise StopIteration("skipped (ack_sent)")
            else:
                # ACK needs to be sent
                message_to_send_key = "User_Requested_Desist"
                send_reason = "DESIST Acknowledgment"
                logger.debug(f"Action needed for {log_prefix}: Send Desist ACK.")

        elif person.status == PersonStatusEnum.ACTIVE:
            # Handle ACTIVE status: Check rules for sending standard messages
            logger.debug(f"{log_prefix}: Status is ACTIVE. Checking messaging rules...")

            # Rule 1: Check if reply received since last script message
            last_out_ts_utc = (
                latest_out_log.latest_timestamp
                if latest_out_log and latest_out_log.latest_timestamp
                else min_aware_dt
            )
            last_in_ts_utc = (
                latest_in_log.latest_timestamp
                if latest_in_log and latest_in_log.latest_timestamp
                else min_aware_dt
            )
            if last_in_ts_utc > last_out_ts_utc:
                logger.debug(
                    f"Skipping {log_prefix}: Reply received ({last_in_ts_utc}) after last script msg ({last_out_ts_utc})."
                )
                raise StopIteration("skipped (reply)")

            # Rule 2: Check time interval since last script message
            if latest_out_log and latest_out_log.latest_timestamp:
                time_since_last = now_utc - latest_out_log.latest_timestamp
                if time_since_last < MIN_MESSAGE_INTERVAL:
                    logger.debug(
                        f"Skipping {log_prefix}: Interval not met ({time_since_last} < {MIN_MESSAGE_INTERVAL})."
                    )
                    raise StopIteration("skipped (interval)")
                # else: logger.debug(f"Interval met for {log_prefix}.")
            # else: logger.debug(f"No previous OUT message for {log_prefix}, interval check skipped.")

            # Rule 3: Determine next message type in sequence
            last_script_message_details: Optional[Tuple[str, datetime, str]] = None
            if (
                latest_out_log and latest_out_log.latest_timestamp
            ):  # Check timestamp exists
                # Prepare details tuple for determine_next_message_type
                last_type_name = (
                    getattr(latest_out_log.message_type, "type_name", "Unknown")
                    if latest_out_log.message_type
                    else "Unknown"
                )
                last_status = latest_out_log.script_message_status or "Unknown"
                last_ts_utc = latest_out_log.latest_timestamp  # Already aware UTC
                last_script_message_details = (last_type_name, last_ts_utc, last_status)

            message_to_send_key = determine_next_message_type(
                last_script_message_details, bool(person.in_my_tree)
            )
            if not message_to_send_key:
                # No appropriate next message in the standard sequence
                logger.debug(
                    f"Skipping {log_prefix}: No appropriate next standard message found."
                )
                raise StopIteration("skipped (sequence)")
            send_reason = "Standard Sequence"
            logger.debug(
                f"Action needed for {log_prefix}: Send '{message_to_send_key}'."
            )

        else:  # Should not happen if prefetch filters correctly
            logger.error(
                f"Unexpected status '{getattr(person.status, 'name', 'UNKNOWN')}' encountered for {log_prefix}. Skipping."
            )
            raise StopIteration("error (unexpected_status)")

        # --- Step 3: Format the Selected Message ---
        if not message_to_send_key or message_to_send_key not in MESSAGE_TEMPLATES:
            logger.error(
                f"Logic Error: Invalid/missing message key '{message_to_send_key}' for {log_prefix}."
            )
            raise StopIteration("error (template_key)")
        message_template = MESSAGE_TEMPLATES[message_to_send_key]

        # Prepare data for template formatting
        dna_match = person.dna_match  # Eager loaded
        family_tree = person.family_tree  # Eager loaded
        # Determine best name to use (Tree Name > First Name > Username)
        name_to_use = (
            family_tree.person_name_in_tree
            if family_tree and family_tree.person_name_in_tree
            else (
                person.first_name
                if person.first_name
                else (
                    person.username
                    if person.username not in ["Unknown", "Unknown User"]
                    else "Valued Relative"
                )
            )
        )  # Fallback name
        formatted_name = format_name(name_to_use)

        # Get total rows count (optional, consider caching if slow)
        total_rows_in_tree = 0
        try:
            total_rows_in_tree = (
                db_session.query(func.count(FamilyTree.id)).scalar() or 0
            )
        except Exception as count_e:
            logger.warning(f"Could not get FamilyTree count for formatting: {count_e}")

        format_data = {
            "name": formatted_name,
            "predicted_relationship": (
                getattr(dna_match, "predicted_relationship", "N/A")
                if dna_match
                else "N/A"
            ),
            "actual_relationship": (
                getattr(family_tree, "actual_relationship", "N/A")
                if family_tree
                else "N/A"
            ),
            "relationship_path": (
                getattr(family_tree, "relationship_path", "N/A")
                if family_tree
                else "N/A"
            ),
            "total_rows": total_rows_in_tree,
        }
        try:
            message_text = message_template.format(**format_data)
        except KeyError as ke:
            logger.error(
                f"Template formatting error (Missing key {ke}) for '{message_to_send_key}' {log_prefix}"
            )
            raise StopIteration("error (template_format)")
        except Exception as e:
            logger.error(
                f"Unexpected template formatting error for {log_prefix}: {e}",
                exc_info=True,
            )
            raise StopIteration("error (template_format)")

        # --- Step 4: Apply Mode/Recipient Filtering ---
        app_mode = config_instance.APP_MODE
        testing_profile_id_config = config_instance.TESTING_PROFILE_ID
        # Use profile_id for filtering (should exist for contactable ACTIVE/DESIST persons)
        current_profile_id = (
            person.profile_id or "UNKNOWN"
        )  # Already uppercase if exists
        send_message_flag = True  # Default to sending
        skip_log_reason = ""

        if app_mode == "testing":
            if not testing_profile_id_config:  # Critical config error
                logger.error(
                    f"Testing mode active, but TESTING_PROFILE_ID not configured. Skipping {log_prefix}."
                )
                send_message_flag = False
                skip_log_reason = "skipped (config_error)"
            elif (
                current_profile_id != testing_profile_id_config
            ):  # Not the designated test recipient
                send_message_flag = False
                skip_log_reason = (
                    f"skipped (testing_mode_filter: not {testing_profile_id_config})"
                )
                logger.debug(
                    f"Testing Mode: Skipping send to {log_prefix} ({skip_log_reason})."
                )
        elif (
            app_mode == "production"
            and testing_profile_id_config
            and current_profile_id == testing_profile_id_config
        ):
            # In production, skip sending to the designated test profile
            send_message_flag = False
            skip_log_reason = (
                f"skipped (production_mode_filter: is {testing_profile_id_config})"
            )
            logger.info(
                f"Production Mode: Skipping send to test profile {log_prefix} ({skip_log_reason})."
            )
        # `dry_run` mode is handled internally by _send_message_via_api

        # --- Step 5: Send/Simulate Message ---
        if send_message_flag:
            logger.debug(
                f"Processing {log_prefix}: Sending/Simulating '{message_to_send_key}' ({send_reason})..."
            )
            # Determine existing conversation ID (prefer OUT log, fallback IN log)
            existing_conversation_id = (
                latest_out_log.conversation_id
                if latest_out_log and latest_out_log.conversation_id
                else (
                    latest_in_log.conversation_id
                    if latest_in_log and latest_in_log.conversation_id
                    else None
                )
            )
            # Call the real API send function
            log_prefix_for_api = f"Action8: {person.username} #{person.id}"
            message_status, effective_conv_id = call_send_message_api(
                session_manager,
                person,
                message_text,
                existing_conversation_id,
                log_prefix_for_api,
            )
        else:
            # If filtered out, use the skip reason as the status for logging
            message_status = skip_log_reason
            # Try to get a conv ID for logging consistency, or generate placeholder
            effective_conv_id = (
                latest_out_log.conversation_id
                if latest_out_log and latest_out_log.conversation_id
                else (
                    latest_in_log.conversation_id
                    if latest_in_log and latest_in_log.conversation_id
                    else f"skipped_{uuid.uuid4()}"
                )
            )

        # --- Step 6: Prepare Database Updates based on outcome ---
        if message_status in (
            "delivered OK",
            "typed (dry_run)",
        ) or message_status.startswith("skipped ("):
            # Prepare new OUT log entry if message sent, simulated, or intentionally skipped by filter
            message_type_id_to_log = message_type_map.get(message_to_send_key)
            if (
                not message_type_id_to_log
            ):  # Should not happen if templates loaded correctly
                logger.error(
                    f"CRITICAL: MessageType ID missing for key '{message_to_send_key}' for {log_prefix}."
                )
                raise StopIteration("error (db_config)")
            if (
                not effective_conv_id
            ):  # Should be set by _send_message_via_api or placeholder
                logger.error(
                    f"CRITICAL: effective_conv_id missing after successful send/simulation/skip for {log_prefix}."
                )
                raise StopIteration("error (internal)")

            # Log content: Prepend skip reason if skipped, otherwise use message text
            log_content = (
                f"[{message_status.upper()}] {message_text}"
                if not send_message_flag
                else message_text
            )[
                : config_instance.MESSAGE_TRUNCATION_LENGTH
            ]  # Truncate
            current_time_for_db = datetime.now(timezone.utc)
            logger.debug(
                f"Preparing new OUT log entry for ConvID {effective_conv_id}, PersonID {person.id}"
            )
            # Create the ConversationLog OBJECT directly
            new_log_entry = ConversationLog(
                conversation_id=effective_conv_id,
                direction=MessageDirectionEnum.OUT,
                people_id=person.id,
                latest_message_content=log_content,
                latest_timestamp=current_time_for_db,
                ai_sentiment=None,  # Not applicable for OUT messages
                message_type_id=message_type_id_to_log,
                script_message_status=message_status,  # Record actual outcome/skip reason
                # updated_at handled by default/onupdate in model
            )

            # Determine overall status and potential person status update
            if message_to_send_key == "User_Requested_Desist":
                # If Desist ACK sent/simulated, stage person update to ARCHIVE
                logger.debug(
                    f"Staging Person status update to ARCHIVE for {log_prefix} (ACK sent/simulated)."
                )
                person_update = (person.id, PersonStatusEnum.ARCHIVE)
                status_string = "acked"  # Specific status for ACK
            elif send_message_flag:
                # Standard message sent/simulated successfully
                status_string = "sent"
            else:
                # Standard message skipped by filter
                status_string = "skipped"  # Use 'skipped' status string
        else:
            # Handle actual send failure reported by _send_message_via_api
            logger.warning(
                f"Message send failed for {log_prefix} with status '{message_status}'. No DB changes staged."
            )
            status_string = "error"  # Indicate send error

        # Step 7: Return prepared updates and status
        return new_log_entry, person_update, status_string

    # --- Step 8: Handle clean exits via StopIteration ---
    except StopIteration as si:
        status_val = (
            str(si.value) if si.value else "skipped"
        )  # Get status string from exception value
        # logger.debug(f"{log_prefix}: Processing stopped cleanly with status '{status_val}'.")
        return None, None, status_val  # Return None for updates, status string
    # --- Step 9: Handle unexpected errors ---
    except Exception as e:
        logger.error(
            f"Unexpected critical error processing {log_prefix}: {e}", exc_info=True
        )
        return None, None, "error"  # Return None, None, 'error'


# End of _process_single_person


# ------------------------------------------------------------------------------
# Main Action Function
# ------------------------------------------------------------------------------


def send_messages_to_matches(session_manager: SessionManager) -> bool:
    """
    Main function for Action 8.
    Fetches eligible candidates, determines the appropriate message to send (if any)
    based on rules and history, sends/simulates the message, and updates the database.
    Uses the unified commit_bulk_data function.

    Args:
        session_manager: The active SessionManager instance.

    Returns:
        True if the process completed without critical database errors, False otherwise.
        Note: Individual message send failures are logged but do not cause the
              entire action to return False unless they lead to a DB commit failure.
    """
    # --- Step 1: Initialization ---
    logger.debug("--- Starting Action 8: Send Standard Messages ---")
    # Validate prerequisites
    if not session_manager or not session_manager.my_profile_id:
        logger.error("Action 8: SM/Profile ID missing.")
        return False
    if not MESSAGE_TEMPLATES:
        logger.error("Action 8: Message templates not loaded.")
        return False
    if (
        login_status(session_manager, disable_ui_fallback=True) is not True
    ):  # API check only for speed
        logger.error("Action 8: Not logged in.")
        return False

    # Counters for summary
    sent_count, acked_count, skipped_count, error_count = 0, 0, 0, 0
    processed_in_loop = 0
    # Lists for batch DB operations
    db_logs_to_add_dicts: List[Dict[str, Any]] = []  # Store prepared Log DICTIONARIES
    person_updates: Dict[int, PersonStatusEnum] = (
        {}
    )  # Store {person_id: new_status_enum}
    # Configuration
    total_candidates = 0
    critical_db_error_occurred = False  # Track if a commit fails critically
    batch_num = 0
    db_commit_batch_size = max(
        1, config_instance.BATCH_SIZE
    )  # Ensure positive batch size
    # Limit number of messages *successfully sent* (sent + acked) in one run (0 = unlimited)
    max_messages_to_send_this_run = config_instance.MAX_INBOX  # Reuse MAX_INBOX setting
    overall_success = True  # Track overall process success

    # --- Step 2: Get DB Session and Pre-fetch Data ---
    db_session: Optional[Session] = None  # Use Session type hint
    try:
        db_session = session_manager.get_db_conn()
        if not db_session:
            # Log critical error if session cannot be obtained
            logger.critical("Action 8: Failed to get DB Session. Aborting.")
            # Ensure cleanup if needed, though SessionManager handles pool
            return False  # Abort if DB session fails

        # Prefetch all data needed for processing loop
        (message_type_map, candidate_persons, latest_in_log_map, latest_out_log_map) = (
            _prefetch_messaging_data(db_session)
        )
        # Validate prefetched data
        if (
            message_type_map is None
            or candidate_persons is None
            or latest_in_log_map is None
            or latest_out_log_map is None
        ):
            logger.error("Action 8: Prefetching essential data failed. Aborting.")
            # Ensure session is returned even on prefetch failure
            if db_session:
                session_manager.return_session(db_session)
            return False

        total_candidates = len(candidate_persons)
        if total_candidates == 0:
            logger.info(
                "Action 8: No candidates found meeting messaging criteria. Finishing.\n"
            )
            # No candidates is considered a successful run
        else:
            logger.info(f"Action 8: Found {total_candidates} candidates to process.")
            # Log limit if applicable
            if max_messages_to_send_this_run > 0:
                logger.info(
                    f"Action 8: Will send/ack a maximum of {max_messages_to_send_this_run} messages this run.\n"
                )

        # --- Step 3: Main Processing Loop ---
        if total_candidates > 0:
            # Setup progress bar
            tqdm_args = {
                "total": total_candidates,
                "desc": "",  # Add space after desc for alignment
                "unit": " person",
                "dynamic_ncols": True,
                "leave": True,
                "bar_format": "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            }
            logger.debug("Processing candidates...")
            with logging_redirect_tqdm(), tqdm(
                **tqdm_args, postfix={"Sent": 0, "ACK": 0, "Skip": 0, "Err": 0}
            ) as progress_bar:
                for person in candidate_persons:
                    processed_in_loop += 1
                    if critical_db_error_occurred:
                        # Update bar for remaining skipped items due to critical error
                        remaining_to_skip = total_candidates - processed_in_loop + 1
                        skipped_count += remaining_to_skip
                        if progress_bar:
                            progress_bar.update(remaining_to_skip)
                            progress_bar.set_postfix(
                                Sent=sent_count,
                                ACK=acked_count,
                                Skip=skipped_count,
                                Err=error_count,
                                refresh=True,
                            )
                        break  # Stop if previous batch commit failed

                    # --- Check Max Send Limit ---
                    current_sent_total = sent_count + acked_count
                    if (
                        max_messages_to_send_this_run > 0
                        and current_sent_total >= max_messages_to_send_this_run
                    ):
                        # Only log the limit message once
                        if not hasattr(progress_bar, "limit_logged"):
                            logger.info(
                                f"Message sending limit ({max_messages_to_send_this_run}) reached. Skipping remaining."
                            )
                            setattr(
                                progress_bar, "limit_logged", True
                            )  # Mark as logged
                        # Increment skipped count for this specific skipped item
                        skipped_count += 1
                        # Update postfix and bar, then continue to next person
                        if progress_bar:
                            progress_bar.set_postfix(
                                Sent=sent_count,
                                ACK=acked_count,
                                Skip=skipped_count,
                                Err=error_count,
                                refresh=False,
                            )
                            progress_bar.update(1)
                        continue  # Skip processing this person

                    # --- Process Single Person ---
                    # _process_single_person still returns a ConversationLog object or None
                    new_log_object, person_update_tuple, status = (
                        _process_single_person(
                            db_session,
                            session_manager,
                            person,
                            latest_in_log_map.get(person.id),
                            latest_out_log_map.get(person.id),
                            message_type_map,
                        )
                    )

                    # --- Tally Results & Collect DB Updates ---
                    log_dict_to_add: Optional[Dict[str, Any]] = None
                    if new_log_object:
                        try:
                            # Convert the SQLAlchemy object attributes to a dictionary
                            log_dict_to_add = {
                                c.key: getattr(new_log_object, c.key)
                                for c in sa_inspect(new_log_object).mapper.column_attrs
                                if hasattr(new_log_object, c.key)  # Ensure attr exists
                            }
                            # Ensure required keys for commit_bulk_data are present and correct type
                            if not all(
                                k in log_dict_to_add
                                for k in [
                                    "conversation_id",
                                    "direction",
                                    "people_id",
                                    "latest_timestamp",
                                ]
                            ):
                                raise ValueError(
                                    "Missing required keys in log object conversion"
                                )
                            if not isinstance(
                                log_dict_to_add["latest_timestamp"], datetime
                            ):
                                raise ValueError(
                                    "Invalid timestamp type in log object conversion"
                                )
                            # Pass Enum directly for direction, commit func handles it
                            log_dict_to_add["direction"] = new_log_object.direction
                            # Normalize timestamp just in case
                            ts_val = log_dict_to_add["latest_timestamp"]
                            log_dict_to_add["latest_timestamp"] = (
                                ts_val.astimezone(timezone.utc)
                                if ts_val.tzinfo
                                else ts_val.replace(tzinfo=timezone.utc)
                            )

                        except Exception as conversion_err:
                            logger.error(
                                f"Failed to convert ConversationLog object to dict for {person.id}: {conversion_err}",
                                exc_info=True,
                            )
                            log_dict_to_add = None  # Prevent adding malformed data
                            status = "error"  # Treat as error if conversion fails

                    # Update counters and collect data based on status
                    if status == "sent":
                        sent_count += 1
                        if log_dict_to_add:
                            db_logs_to_add_dicts.append(log_dict_to_add)
                    elif status == "acked":
                        acked_count += 1
                        if log_dict_to_add:
                            db_logs_to_add_dicts.append(log_dict_to_add)
                        if person_update_tuple:
                            person_updates[person_update_tuple[0]] = (
                                person_update_tuple[1]
                            )
                    elif status.startswith("skipped"):
                        skipped_count += 1
                        # If skipped due to filter/rules, still add the log entry if one was prepared
                        # This logs the skip reason in the script_message_status field.
                        if log_dict_to_add:
                            db_logs_to_add_dicts.append(log_dict_to_add)
                    else:  # status == "error" or unexpected
                        error_count += 1
                        overall_success = False

                    # Update progress bar postfix data AND advance bar
                    if progress_bar:
                        progress_bar.set_postfix(
                            Sent=sent_count,
                            ACK=acked_count,
                            Skip=skipped_count,
                            Err=error_count,
                            refresh=False,
                        )
                        progress_bar.update(1)

                    # --- Commit Batch Periodically ---
                    if (
                        len(db_logs_to_add_dicts) + len(person_updates)
                    ) >= db_commit_batch_size:
                        batch_num += 1
                        logger.debug(
                            f"Commit threshold reached ({len(db_logs_to_add_dicts)} logs). Committing Action 8 Batch {batch_num}..."
                        )
                        # --- CALL NEW FUNCTION ---
                        try:
                            logs_committed_count, persons_updated_count = (
                                commit_bulk_data(
                                    session=db_session,
                                    log_upserts=db_logs_to_add_dicts,  # Pass list of dicts
                                    person_updates=person_updates,
                                    context=f"Action 8 Batch {batch_num}",
                                )
                            )
                            # Commit successful (no exception raised)
                            db_logs_to_add_dicts.clear()
                            person_updates.clear()
                            logger.debug(
                                f"Action 8 Batch {batch_num} commit finished (Logs Processed: {logs_committed_count}, Persons Updated: {persons_updated_count})."
                            )
                        except Exception as commit_e:
                            # commit_bulk_data should handle internal errors and logging,
                            # but catch here to set critical flag and stop loop.
                            logger.critical(
                                f"CRITICAL: Messaging batch commit {batch_num} FAILED: {commit_e}",
                                exc_info=True,
                            )
                            critical_db_error_occurred = True
                            overall_success = False
                            break  # Stop processing loop

                # --- End Main Person Loop ---
        # --- End Conditional Processing Block (if total_candidates > 0) ---

        # --- Step 4: Final Commit ---
        if not critical_db_error_occurred and (db_logs_to_add_dicts or person_updates):
            batch_num += 1
            logger.debug(
                f"Performing final commit for remaining items (Batch {batch_num})..."
            )
            try:
                # --- CALL NEW FUNCTION ---
                final_logs_saved, final_persons_updated = commit_bulk_data(
                    session=db_session,
                    log_upserts=db_logs_to_add_dicts,
                    person_updates=person_updates,
                    context="Action 8 Final Save",
                )
                # Commit successful
                db_logs_to_add_dicts.clear()
                person_updates.clear()
                logger.debug(
                    f"Action 8 Final commit executed (Logs Processed: {final_logs_saved}, Persons Updated: {final_persons_updated})."
                )
            except Exception as final_commit_e:
                logger.error(
                    f"Final Action 8 batch commit FAILED: {final_commit_e}",
                    exc_info=True,
                )
                overall_success = False

    # --- Step 5: Handle Outer Exceptions ---
    except Exception as outer_e:
        logger.critical(
            f"CRITICAL: Unhandled error during Action 8 execution: {outer_e}",
            exc_info=True,
        )
        overall_success = False
    # --- Step 6: Final Cleanup and Summary ---
    finally:
        if db_session:
            session_manager.return_session(db_session)  # Ensure session is returned

        # Log Summary
        # Adjust final skipped count if loop was stopped early by critical error
        if critical_db_error_occurred and total_candidates > processed_in_loop:
            unprocessed_count = total_candidates - processed_in_loop
            logger.warning(
                f"Adding {unprocessed_count} unprocessed candidates to skipped count due to DB commit failure."
            )
            skipped_count += unprocessed_count

        print(" ")  # Spacer
        logger.info("--- Action 8: Message Sending Summary ---")
        logger.info(f"  Candidates Considered:              {total_candidates}")
        logger.info(f"  Candidates Processed in Loop:       {processed_in_loop}")
        logger.info(f"  Template Messages Sent/Simulated:   {sent_count}")
        logger.info(f"  Desist ACKs Sent/Simulated:         {acked_count}")
        logger.info(f"  Skipped (Rules/Filter/Limit/Error): {skipped_count}")
        logger.info(f"  Errors during processing/sending:   {error_count}")
        logger.info(f"  Overall Action Success:             {overall_success}")
        logger.info("-----------------------------------------\n")

    # Step 7: Return overall success status
    return overall_success


# End of send_messages_to_matches


# --- Standalone Testing ---
def main():
    """Main function for standalone testing of Action 8 messaging."""
    # Step 1: Setup Logging
    from logging_config import setup_logging  # Local import

    global logger  # Ensure global logger is used/modified
    try:
        from config import config_instance  # Local import

        db_file_path = config_instance.DATABASE_FILE
        log_filename_only = db_file_path.with_suffix(".log").name
        logger = setup_logging(
            log_file=log_filename_only, log_level="DEBUG"
        )  # Use DEBUG for testing
        logger.info(f"--- Starting Action 8 Standalone Test ---")
        logger.info(f"APP_MODE: {config_instance.APP_MODE}")
    except Exception as log_setup_e:
        # Fallback logging
        print(f"CRITICAL: Error during logging setup: {log_setup_e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger("Action8Fallback")
        logger.info(f"--- Starting Action 8 Standalone Test (Fallback Logging) ---")

    # Step 2: Initialize Session Manager
    session_manager: Optional[SessionManager] = None
    action_success = False  # Default to failure

    # Step 3: Execute Action within try/finally for cleanup
    try:
        session_manager = SessionManager()
        logger.info("Attempting session start (Phase 1 & 2)...")
        # Perform both phases of session start
        if session_manager.start_sess(action_name="Action 8 Test - Phase 1"):
            if session_manager.ensure_session_ready(
                action_name="Action 8 Test - Phase 2"
            ):
                logger.info("Session ready. Proceeding to send_messages_to_matches...")
                # Call the main action function
                action_success = send_messages_to_matches(session_manager)
                logger.info(
                    f"send_messages_to_matches completed. Result: {action_success}"
                )
            else:
                logger.critical("Failed Phase 2 (Session Ready). Cannot run messaging.")
        else:
            logger.critical("Failed Phase 1 (Driver Start). Cannot run messaging.")
    except Exception as e:
        logger.critical(
            f"Critical error in Action 8 standalone main: {e}", exc_info=True
        )
        action_success = False  # Ensure failure on exception
    finally:
        # Step 4: Cleanup Session Manager
        logger.info("Closing session manager in finally block...")
        if session_manager:
            session_manager.close_sess()
        logger.info(
            f"--- Action 8 Standalone Test Finished (Overall Success: {action_success}) ---"
        )


# End of main

if __name__ == "__main__":
    main()
# End of action8_messaging.py
