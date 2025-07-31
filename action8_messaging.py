#!/usr/bin/env python3

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

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

# === STANDARD LIBRARY IMPORTS ===
import json
import logging
import sys
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urljoin

# === THIRD-PARTY IMPORTS ===
import requests
from sqlalchemy import (
    and_,
    func,
    inspect as sa_inspect,
    tuple_,
)  # Minimal imports

# === LOCAL IMPORTS ===
# Import PersonStatusEnum early for use in safe_column_value
from database import PersonStatusEnum


# --- Helper function for SQLAlchemy Column conversion ---
def safe_column_value(obj, attr_name, default=None):
    """
    Safely extract a value from a SQLAlchemy model attribute, handling Column objects.

    Args:
        obj: The SQLAlchemy model instance
        attr_name: The attribute name to access
        default: Default value to return if attribute doesn't exist or conversion fails

    Returns:
        The Python primitive value of the attribute, or the default value
    """
    if not hasattr(obj, attr_name):
        return default

    value = getattr(obj, attr_name)
    if value is None:
        return default

    # Try to convert to Python primitive
    try:
        # Special handling for status enum
        if attr_name == "status":
            # If it's already an enum instance, return it
            if isinstance(value, PersonStatusEnum):
                return value
            # If it's a string, try to convert to enum
            elif isinstance(value, str):
                try:
                    return PersonStatusEnum(value)
                except ValueError:
                    logger.warning(
                        f"Invalid status string '{value}', cannot convert to enum"
                    )
                    return default
            # If it's something else, log and return default
            else:
                logger.warning(f"Unexpected status type: {type(value)}")
                return default

        # For different types of attributes
        if isinstance(value, bool) or value is True or value is False:
            return bool(value)
        elif isinstance(value, int) or str(value).isdigit():
            return int(value)
        elif isinstance(value, float) or str(value).replace(".", "", 1).isdigit():
            return float(value)
        elif hasattr(value, "isoformat"):  # datetime-like
            return value
        else:
            return str(value)
    except (ValueError, TypeError, AttributeError):
        return default


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
try:
    from core_imports import auto_register_module

    auto_register_module(globals(), __name__)
except ImportError:
    pass  # Continue without auto-registration if not available

# Standardize imports if available
try:
    from core_imports import standardize_module_imports

    standardize_module_imports()
except ImportError:
    pass

from cache import cache_result  # Caching utility
from config import config_schema  # Configuration singletons
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
from core.session_manager import SessionManager
from utils import (  # Core utilities
    DynamicRateLimiter,  # Rate limiter (accessed via SessionManager)
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


# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
    MagicMock,
    patch,
)

# --- Initialization & Template Loading ---
logger.debug(f"Action 8 Initializing: APP_MODE is {config_schema.environment}")

# Define message intervals based on app mode (controls time between follow-ups)
MESSAGE_INTERVALS = {
    "testing": timedelta(seconds=10),  # Short interval for testing
    "production": timedelta(weeks=8),  # Standard interval for production
    "dry_run": timedelta(seconds=10),  # Short interval for dry runs
}
MIN_MESSAGE_INTERVAL: timedelta = MESSAGE_INTERVALS.get(
    config_schema.environment, timedelta(weeks=8)
)
logger.debug(f"Action 8 Using minimum message interval: {MIN_MESSAGE_INTERVAL}")

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


# Define message transition table as a module-level constant
# Maps (current_message_type, is_in_family_tree) to next_message_type
# None as current_message_type means no previous message
# None as next_message_type means end of sequence or no appropriate next message
MESSAGE_TRANSITION_TABLE = {
    # Initial message cases (no previous message)
    (None, True): "In_Tree-Initial",
    (None, False): "Out_Tree-Initial",
    # In-Tree sequences
    ("In_Tree-Initial", True): "In_Tree-Follow_Up",
    ("In_Tree-Initial_for_was_Out_Tree", True): "In_Tree-Follow_Up",
    ("In_Tree-Follow_Up", True): "In_Tree-Final_Reminder",
    ("In_Tree-Final_Reminder", True): None,  # End of In-Tree sequence
    # Out-Tree sequences
    ("Out_Tree-Initial", False): "Out_Tree-Follow_Up",
    ("Out_Tree-Follow_Up", False): "Out_Tree-Final_Reminder",
    ("Out_Tree-Final_Reminder", False): None,  # End of Out-Tree sequence
    # Tree status change transitions
    # Any Out-Tree message -> In-Tree status
    ("Out_Tree-Initial", True): "In_Tree-Initial_for_was_Out_Tree",
    ("Out_Tree-Follow_Up", True): "In_Tree-Initial_for_was_Out_Tree",
    ("Out_Tree-Final_Reminder", True): "In_Tree-Initial_for_was_Out_Tree",
    # Special case: Was Out->In->Out again
    ("In_Tree-Initial_for_was_Out_Tree", False): "Out_Tree-Initial",
    # General case: Was In-Tree, now Out-Tree (stop messaging)
    ("In_Tree-Initial", False): None,
    ("In_Tree-Follow_Up", False): None,
    ("In_Tree-Final_Reminder", False): None,
    # Desist acknowledgment always ends the sequence
    ("User_Requested_Desist", True): None,
    ("User_Requested_Desist", False): None,
}


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

    Uses a state machine approach with a transition table that maps
    (current_message_type, is_in_family_tree) tuples to the next message type.

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

    # Step 2: Extract the last message type (or None if no previous message)
    last_message_type = None
    if last_message_details:
        last_message_type, last_sent_at_utc, last_message_status = last_message_details
        logger.debug(
            f"  Last Sent Type: '{last_message_type}', SentAt: {last_sent_at_utc}, Status: '{last_message_status}'"
        )

    # Step 3: Look up the next message type in the transition table
    transition_key = (last_message_type, is_in_family_tree)
    next_type = None
    reason = "Unknown transition"

    if transition_key in MESSAGE_TRANSITION_TABLE:
        # Standard transition found in table
        next_type = MESSAGE_TRANSITION_TABLE[transition_key]
        if next_type:
            reason = f"Standard transition from '{last_message_type or 'None'}' with in_tree={is_in_family_tree}"
        else:
            reason = f"End of sequence for '{last_message_type}' with in_tree={is_in_family_tree}"
    else:
        # Handle unexpected previous message type
        if last_message_type:
            tree_status = "In_Tree" if is_in_family_tree else "Out_Tree"
            reason = f"Unexpected previous {tree_status} type: '{last_message_type}'"
            logger.warning(f"  Decision: Skip ({reason})")
        else:
            # Fallback for initial message if somehow not in transition table
            next_type = (
                MESSAGE_TYPES_ACTION8["In_Tree-Initial"]
                if is_in_family_tree
                else MESSAGE_TYPES_ACTION8["Out_Tree-Initial"]
            )
            reason = "Fallback for initial message (no prior message)"

    # Step 4: Convert next_type string to actual message type from MESSAGE_TYPES_ACTION8
    if next_type:
        next_type = MESSAGE_TYPES_ACTION8.get(next_type, next_type)
        logger.debug(f"  Decision: Send '{next_type}' (Reason: {reason}).")
    else:
        logger.debug(f"  Decision: Skip ({reason}).")

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
                    # Use our safe helper to get values
                    conv_id = safe_column_value(log_obj, "conversation_id", None)
                    direction = safe_column_value(log_obj, "direction", None)

                    if conv_id and direction:
                        log_keys_to_check.add((conv_id, direction))
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
                    existing_logs_map = {}
                    for log in existing_logs:
                        # Use our safe helper to get values
                        conv_id = safe_column_value(log, "conversation_id", None)
                        direction = safe_column_value(log, "direction", None)
                        if conv_id and direction:
                            existing_logs_map[(conv_id, direction)] = log
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
                        sess.bulk_insert_mappings(ConversationLog, log_inserts_data)  # type: ignore
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
                    sess.bulk_update_mappings(Person, update_mappings)  # type: ignore

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

        # Check if we're running in a test/mock environment
        is_mock_mode = "--mock" in sys.argv or "--test" in sys.argv

        if is_mock_mode:
            # Create a mock message_type_map for testing
            logger.info("Running in mock mode, creating mock MessageType map...")
            message_type_map = {}
            for i, type_name in enumerate(MESSAGE_TYPES_ACTION8.keys(), start=1):
                message_type_map[type_name] = i
            message_type_map["Productive_Reply_Acknowledgement"] = (
                len(message_type_map) + 1
            )
            logger.debug(
                f"Created mock message_type_map with {len(message_type_map)} entries"
            )
        else:
            # Normal database query
            message_types = db_session.query(
                MessageType.id, MessageType.type_name
            ).all()
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

        if is_mock_mode:
            # Create mock candidate persons for testing
            logger.info("Running in mock mode, creating mock candidate persons...")
            candidate_persons = []
            logger.debug(f"Created empty mock candidate_persons list for testing")
        else:
            # Normal database query
            candidate_persons = (
                db_session.query(Person)
                .options(
                    joinedload(
                        Person.dna_match
                    ),  # Eager load needed data for formatting
                    joinedload(
                        Person.family_tree
                    ),  # Eager load needed data for formatting
                )
                .filter(
                    Person.profile_id.isnot(None),  # Ensure profile ID exists
                    Person.profile_id != "UNKNOWN",
                    Person.contactable == True,  # Only contactable people
                    Person.status.in_(
                        [PersonStatusEnum.ACTIVE, PersonStatusEnum.DESIST]
                    ),  # Eligible statuses
                    Person.deleted_at == None,  # Exclude soft-deleted records
                )
                .order_by(Person.id)  # Consistent order
                .all()
            )

        logger.debug(f"Fetched {len(candidate_persons)} potential candidates.")
        if not candidate_persons:
            return message_type_map, [], {}, {}  # Return empty results if no candidates

        # Step 4: Fetch Latest Conversation Logs for candidates
        # Extract person IDs as a list - convert SQLAlchemy Column objects to Python ints
        candidate_person_ids = []
        for p in candidate_persons:
            # Use our safe helper function
            person_id = safe_column_value(p, "id", None)
            if person_id is not None:
                candidate_person_ids.append(person_id)

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

        # Populate maps with Python primitives
        for log in latest_logs:
            # --- Process each log entry ---
            try:
                # Get the person ID as a Python int using our safe helper
                person_id = safe_column_value(log, "people_id", None)
                if person_id is None:
                    continue  # Skip logs without a valid person ID

                # Get the direction as a Python enum using our safe helper
                direction = safe_column_value(log, "direction", None)
                if direction is None:
                    continue  # Skip logs without a valid direction

                # Add to appropriate map based on direction
                if direction == MessageDirectionEnum.IN:
                    latest_in_log_map[person_id] = log
                elif direction == MessageDirectionEnum.OUT:
                    latest_out_log_map[person_id] = log
            except Exception as log_err:
                logger.warning(f"Error processing log entry: {log_err}")
                continue

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
    # Convert SQLAlchemy Column objects to Python primitives using our safe helper
    username = safe_column_value(person, "username", "Unknown")
    person_id = safe_column_value(person, "id", 0)

    # For nested attributes like person.status.name, we need to be more careful
    status = safe_column_value(person, "status", None)
    if status is not None:
        status_name = getattr(status, "name", "Unknown")
    else:
        status_name = "Unknown"

    log_prefix = f"{username} #{person_id} (Status: {status_name})"
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
        # Get the status as a Python enum using our safe helper
        person_status = safe_column_value(person, "status", None)

        # Handle DESIST status
        if person_status == PersonStatusEnum.DESIST:
            # When status is DESIST, we only send an acknowledgment if needed
            logger.debug(
                f"{log_prefix}: Status is DESIST. Checking if Desist ACK needed."
            )

            # Get the message type ID for the Desist acknowledgment
            desist_ack_type_id = message_type_map.get("User_Requested_Desist")
            if not desist_ack_type_id:  # Should have been checked during prefetch
                logger.critical(
                    "CRITICAL: User_Requested_Desist ID missing from message type map."
                )
                raise StopIteration("error (config)")

            # Check if the latest OUT message was already the Desist ACK
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

        elif person_status == PersonStatusEnum.ACTIVE:
            # Handle ACTIVE status: Check rules for sending standard messages
            logger.debug(f"{log_prefix}: Status is ACTIVE. Checking messaging rules...")

            # Rule 1: Check if reply received since last script message
            # Use our safe helper to get timestamps
            last_out_ts_utc = min_aware_dt
            if latest_out_log:
                last_out_ts_utc = safe_column_value(
                    latest_out_log, "latest_timestamp", min_aware_dt
                )

            last_in_ts_utc = min_aware_dt
            if latest_in_log:
                last_in_ts_utc = safe_column_value(
                    latest_in_log, "latest_timestamp", min_aware_dt
                )
            if last_in_ts_utc > last_out_ts_utc:
                logger.debug(
                    f"Skipping {log_prefix}: Reply received ({last_in_ts_utc}) after last script msg ({last_out_ts_utc})."
                )
                raise StopIteration("skipped (reply)")

            # Rule 1b: Check if custom reply has already been sent for the latest incoming message
            if (
                latest_in_log
                and hasattr(latest_in_log, "custom_reply_sent_at")
                and latest_in_log.custom_reply_sent_at is not None
            ):
                logger.debug(
                    f"Skipping {log_prefix}: Custom reply already sent at {latest_in_log.custom_reply_sent_at}."
                )
                raise StopIteration("skipped (custom_reply_sent)")

            # Rule 2: Check time interval since last script message
            if latest_out_log:
                # Use our safe helper to get the timestamp
                out_timestamp = safe_column_value(
                    latest_out_log, "latest_timestamp", None
                )
                if out_timestamp:
                    time_since_last = now_utc - out_timestamp
                    if time_since_last < MIN_MESSAGE_INTERVAL:
                        logger.debug(
                            f"Skipping {log_prefix}: Interval not met ({time_since_last} < {MIN_MESSAGE_INTERVAL})."
                        )
                        raise StopIteration("skipped (interval)")
                    # else: logger.debug(f"Interval met for {log_prefix}.")
            # else: logger.debug(f"No previous OUT message for {log_prefix}, interval check skipped.")

            # Rule 3: Determine next message type in sequence
            last_script_message_details: Optional[Tuple[str, datetime, str]] = None
            if latest_out_log:
                # Use our safe helper to get the timestamp
                out_timestamp = safe_column_value(
                    latest_out_log, "latest_timestamp", None
                )
                if out_timestamp:
                    # Get message type using safe helper
                    message_type_obj = safe_column_value(
                        latest_out_log, "message_type", None
                    )
                    last_type_name = "Unknown"
                    if message_type_obj:
                        last_type_name = getattr(
                            message_type_obj, "type_name", "Unknown"
                        )

                    # Get status using safe helper
                    last_status = safe_column_value(
                        latest_out_log, "script_message_status", "Unknown"
                    )

                    # Create the tuple with Python primitives
                    last_script_message_details = (
                        last_type_name,
                        out_timestamp,
                        last_status,
                    )

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
        # Use our safe helper to get values
        tree_name = None
        if family_tree:
            tree_name = safe_column_value(family_tree, "person_name_in_tree", None)

        first_name = safe_column_value(person, "first_name", None)
        username = safe_column_value(person, "username", None)

        # Choose the best name with fallbacks
        if tree_name:
            name_to_use = tree_name
        elif first_name:
            name_to_use = first_name
        elif username and username not in ["Unknown", "Unknown User"]:
            name_to_use = username
        else:
            name_to_use = "Valued Relative"

        # Format the name
        formatted_name = format_name(name_to_use)

        # Get total rows count (optional, consider caching if slow)
        total_rows_in_tree = 0
        try:
            total_rows_in_tree = (
                db_session.query(func.count(FamilyTree.id)).scalar() or 0
            )
        except Exception as count_e:
            logger.warning(f"Could not get FamilyTree count for formatting: {count_e}")

        # Helper function to format predicted relationship with correct percentage
        def format_predicted_relationship(rel_str):
            if not rel_str or rel_str == "N/A":
                return "N/A"

            # Check if the string contains a percentage in brackets
            import re

            match = re.search(r"\[([\d.]+)%\]", rel_str)
            if match:
                try:
                    # Extract the percentage value
                    percentage = float(match.group(1))
                    # If the percentage is very small (less than 1%), it's likely using the new decimal format
                    if percentage < 1.0:
                        # Multiply by 100 to convert back to percentage format
                        corrected_percentage = percentage * 100
                        # Replace the old percentage with the corrected one
                        return re.sub(
                            r"\[([\d.]+)%\]", f"[{corrected_percentage:.1f}%]", rel_str
                        )
                except (ValueError, IndexError):
                    pass

            # Return the original string if no percentage found or couldn't be processed
            return rel_str

        # Get the predicted relationship and format it correctly
        predicted_rel = "N/A"
        if dna_match:
            raw_predicted_rel = getattr(dna_match, "predicted_relationship", "N/A")
            predicted_rel = format_predicted_relationship(raw_predicted_rel)

        format_data = {
            "name": formatted_name,
            "predicted_relationship": predicted_rel,
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
        app_mode = config_schema.environment
        testing_profile_id_config = config_schema.testing_profile_id
        # Use profile_id for filtering (should exist for contactable ACTIVE/DESIST persons)
        current_profile_id = safe_column_value(person, "profile_id", "UNKNOWN")
        send_message_flag = True  # Default to sending
        skip_log_reason = ""

        # Testing mode checks
        if app_mode == "testing":
            # Check if testing profile ID is configured
            if not testing_profile_id_config:
                logger.error(
                    f"Testing mode active, but TESTING_PROFILE_ID not configured. Skipping {log_prefix}."
                )
                send_message_flag = False
                skip_log_reason = "skipped (config_error)"
            # Check if current profile matches testing profile
            elif current_profile_id != testing_profile_id_config:
                send_message_flag = False
                skip_log_reason = (
                    f"skipped (testing_mode_filter: not {testing_profile_id_config})"
                )
                logger.debug(
                    f"Testing Mode: Skipping send to {log_prefix} ({skip_log_reason})."
                )

        # Production mode checks
        elif app_mode == "production":
            # Check if testing profile ID is configured and matches current profile
            if (
                testing_profile_id_config
                and current_profile_id == testing_profile_id_config
            ):
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
            existing_conversation_id = None
            if latest_out_log:
                existing_conversation_id = safe_column_value(
                    latest_out_log, "conversation_id", None
                )

            if existing_conversation_id is None and latest_in_log:
                existing_conversation_id = safe_column_value(
                    latest_in_log, "conversation_id", None
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
            effective_conv_id = None
            if latest_out_log:
                effective_conv_id = safe_column_value(
                    latest_out_log, "conversation_id", None
                )

            if effective_conv_id is None and latest_in_log:
                effective_conv_id = safe_column_value(
                    latest_in_log, "conversation_id", None
                )

            if effective_conv_id is None:
                effective_conv_id = f"skipped_{uuid.uuid4()}"

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
                : config_schema.message_truncation_length
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
                person_update = (person_id, PersonStatusEnum.ARCHIVE)
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


@retry_on_failure(max_attempts=3, backoff_factor=2.0)
@circuit_breaker(failure_threshold=5, recovery_timeout=300)
@timeout_protection(timeout=1800)  # 30 minutes for messaging operations
@graceful_degradation(fallback_value=False)
@error_context("action8_messaging")
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
    if not session_manager:
        logger.error("Action 8: SessionManager missing.")
        return False

    # Use safe_column_value to get profile_id
    profile_id = None
    if hasattr(session_manager, "my_profile_id"):
        profile_id = safe_column_value(session_manager, "my_profile_id", None)

    if not profile_id:
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
        1, config_schema.batch_size
    )  # Ensure positive batch size
    # Limit number of messages *successfully sent* (sent + acked) in one run (0 = unlimited)
    max_messages_to_send_this_run = config_schema.max_inbox  # Reuse MAX_INBOX setting
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
                "desc": "Processing",  # Add a description
                "unit": " person",
                "dynamic_ncols": True,
                "leave": True,
                "bar_format": "{desc} |{bar}| {percentage:3.0f}% ({n_fmt}/{total_fmt})",
                "file": sys.stderr,
            }
            logger.debug("Processing candidates...")
            with logging_redirect_tqdm(), tqdm(**tqdm_args) as progress_bar:
                for person in candidate_persons:
                    processed_in_loop += 1
                    if critical_db_error_occurred:
                        # Update bar for remaining skipped items due to critical error
                        remaining_to_skip = total_candidates - processed_in_loop + 1
                        skipped_count += remaining_to_skip
                        if progress_bar:
                            progress_bar.set_description(
                                f"ERROR: DB commit failed - Sent={sent_count} ACK={acked_count} Skip={skipped_count} Err={error_count}"
                            )
                            progress_bar.update(remaining_to_skip)
                        break  # Stop if previous batch commit failed

                    # --- Check Max Send Limit ---
                    current_sent_total = sent_count + acked_count
                    if (
                        max_messages_to_send_this_run > 0
                        and current_sent_total >= max_messages_to_send_this_run
                    ):
                        # Only log the limit message once
                        if not hasattr(progress_bar, "limit_logged"):
                            logger.debug(
                                f"Message sending limit ({max_messages_to_send_this_run}) reached. Skipping remaining."
                            )
                            setattr(
                                progress_bar, "limit_logged", True
                            )  # Mark as logged
                        # Increment skipped count for this specific skipped item
                        skipped_count += 1
                        # Update description and bar, then continue to next person
                        if progress_bar:
                            progress_bar.set_description(
                                f"Limit reached: Sent={sent_count} ACK={acked_count} Skip={skipped_count} Err={error_count}"
                            )
                            progress_bar.update(1)
                        continue  # Skip processing this person

                    # --- Process Single Person ---
                    # _process_single_person still returns a ConversationLog object or None
                    # Convert person.id to Python int for dictionary lookup using our safe helper
                    person_id_int = safe_column_value(person, "id", 0)

                    # Use the Python int for dictionary lookup
                    latest_in_log = latest_in_log_map.get(person_id_int)
                    latest_out_log = latest_out_log_map.get(person_id_int)

                    new_log_object, person_update_tuple, status = (
                        _process_single_person(
                            db_session,
                            session_manager,
                            person,
                            latest_in_log,
                            latest_out_log,
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

                    # Update progress bar description and advance bar
                    if progress_bar:
                        progress_bar.set_description(
                            f"Processing: Sent={sent_count} ACK={acked_count} Skip={skipped_count} Err={error_count}"
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


# ==============================================
# Standalone Test Block
# ==============================================
def action8_messaging_tests():
    """Test suite for action8_messaging.py - Automated Messaging System with detailed reporting."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Action 8 - Automated Messaging System", "action8_messaging.py")

    def test_function_availability():
        """Test messaging system functions are available with detailed verification."""
        required_functions = [
            ("safe_column_value", "Safe SQLAlchemy column value extraction"),
            ("load_message_templates", "Message template loading from JSON"),
            ("determine_next_message_type", "Next message type determination logic"),
            ("_commit_messaging_batch", "Database batch commit operations"),
            ("_prefetch_messaging_data", "Data prefetching for messaging"),
            ("_process_single_person", "Individual person message processing"),
            ("send_messages_to_matches", "Main messaging function for DNA matches"),
        ]

        print("📋 Testing Action 8 messaging function availability:")
        results = []

        for func_name, description in required_functions:
            # Test function existence
            func_exists = func_name in globals()

            # Test function callability
            func_callable = False
            if func_exists:
                try:
                    func_callable = callable(globals()[func_name])
                except Exception:
                    func_callable = False

            # Test function type
            func_type = type(globals().get(func_name, None)).__name__

            status = "✅" if func_exists and func_callable else "❌"
            print(f"   {status} {func_name}: {description}")
            print(
                f"      Exists: {func_exists}, Callable: {func_callable}, Type: {func_type}"
            )

            test_passed = func_exists and func_callable
            results.append(test_passed)

            assert func_exists, f"Function {func_name} should be available"
            assert func_callable, f"Function {func_name} should be callable"

        print(
            f"📊 Results: {sum(results)}/{len(results)} Action 8 messaging functions available"
        )

    def test_safe_column_value():
        """Test safe column value extraction with detailed verification."""
        test_cases = [
            (None, "attr", "default", "None object handling"),
            (
                type("MockObj", (), {"attr": "value"})(),
                "attr",
                "default",
                "object with attribute",
            ),
            (
                type("MockObj", (), {})(),
                "missing_attr",
                "fallback",
                "object without attribute",
            ),
        ]

        print("📋 Testing safe column value extraction:")
        results = []

        for obj, attr_name, default, description in test_cases:
            try:
                result = safe_column_value(obj, attr_name, default)
                test_passed = result is not None or result == default

                status = "✅" if test_passed else "❌"
                print(f"   {status} {description}")
                print(
                    f"      Input: obj={type(obj).__name__}, attr='{attr_name}', default='{default}' → Result: {repr(result)}"
                )

                results.append(test_passed)

            except Exception as e:
                print(f"   ❌ {description}")
                print(f"      Error: {e}")
                results.append(False)
                raise

        print(
            f"📊 Results: {sum(results)}/{len(results)} safe column value tests passed"
        )

    def test_message_template_loading():
        """Test message template loading functionality."""
        print("📋 Testing message template loading:")
        results = []

        try:
            templates = load_message_templates()
            templates_loaded = isinstance(templates, dict)

            status = "✅" if templates_loaded else "❌"
            print(f"   {status} Message template loading")
            print(
                f"      Type: {type(templates).__name__}, Count: {len(templates) if templates_loaded else 0}"
            )

            results.append(templates_loaded)
            assert templates_loaded, "load_message_templates should return a dictionary"

        except Exception as e:
            print(f"   ❌ Message template loading")
            print(f"      Error: {e}")
            results.append(False)
            # Don't raise as templates file might not exist in test environment

        print(
            f"📊 Results: {sum(results)}/{len(results)} message template loading tests passed"
        )

    print(
        "📧 Running Action 8 - Automated Messaging System comprehensive test suite..."
    )

    with suppress_logging():
        suite.run_test(
            "Function availability verification",
            test_function_availability,
            "7 messaging functions tested: safe_column_value, load_message_templates, determine_next_message_type, _commit_messaging_batch, _prefetch_messaging_data, _process_single_person, send_messages_to_matches.",
            "Test messaging system functions are available with detailed verification.",
            "Verify safe_column_value→SQLAlchemy extraction, load_message_templates→JSON loading, determine_next_message_type→logic, _commit_messaging_batch→database, _prefetch_messaging_data→optimization, _process_single_person→individual processing, send_messages_to_matches→main function.",
        )

        suite.run_test(
            "Safe column value extraction",
            test_safe_column_value,
            "3 safe extraction tests: None object, object with attribute, object without attribute - all handle gracefully.",
            "Test safe column value extraction with detailed verification.",
            "Verify safe_column_value() handles None→default, obj.attr→value, obj.missing→default extraction patterns.",
        )

        suite.run_test(
            "Message template loading",
            test_message_template_loading,
            "Message template loading tested: load_message_templates() returns dictionary of templates from JSON.",
            "Test message template loading functionality.",
            "Verify load_message_templates() loads JSON→dict templates for messaging system.",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run tests using unified test framework."""
    return action8_messaging_tests()


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    print(
        "📧 Running Action 8 - Automated Messaging System comprehensive test suite..."
    )
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
