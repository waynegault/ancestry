# File: action8_messaging.py
# V14.14: Initialize tqdm inside logging_redirect_tqdm context.

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
from tqdm.auto import tqdm
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
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (
    Session as DbSession,
    aliased,
    joinedload,
)
from sqlalchemy.sql import select
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
)
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
def load_message_templates() -> Dict[str, str]:
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
    skip_reason: str = "End of sequence or other condition met"

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
    else:
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
    V14.55 REVISED: Handles IntegrityError during dry runs gracefully without aborting.
    Refined final error count logic. Consolidated previous fixes.
    Logs outgoing messages to ConversationLog. Filters by ACTIVE status.
    Skips if last message is IN. Checks interval based on last OUT log.
    """
    # --- Prerequisites ---
    if not session_manager:
        logger.error("SessionManager instance is required.")
        return False
    if login_status(session_manager) is not True:
        logger.error("Login status check failed.")
        return False
    if not session_manager.driver:
        logger.warning("Driver not available, UBE header missing.")
    if not session_manager.my_profile_id:
        logger.error("Own profile ID missing.")
        return False
    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()
    MY_PROFILE_ID_UPPER = session_manager.my_profile_id.upper()
    if not MESSAGE_TEMPLATES:
        logger.error("Message templates failed to load.")
        return False

    overall_success = True
    processed_count = 0
    sent_count = 0
    skipped_count = 0
    error_count = 0
    potential_matches = []
    max_to_send = config_instance.MAX_INBOX
    db_commit_batch_size = config_instance.BATCH_SIZE
    db_objects_to_commit: List[ConversationLog] = []
    progress_bar = None
    total_potential = 0
    critical_db_error_occurred = False  # Flag for critical commit errors

    try:
        with session_manager.get_db_conn_context() as db_session:
            if not db_session:
                logger.error("Failed to get DB connection.")
                return False

            # --- Pre-fetching ---
            logger.debug("--- Starting Pre-fetching for Action 8 (Messaging) ---")
            logger.debug("Constructing potential_matches_query...")
            try:  # DB State Check
                person_count = db_session.query(func.count(Person.id)).scalar()
                active_person_count = (
                    db_session.query(func.count(Person.id))
                    .filter(Person.status == PersonStatusEnum.ACTIVE)
                    .scalar()
                )
                contactable_person_count = (
                    db_session.query(func.count(Person.id))
                    .filter(Person.contactable == True)
                    .scalar()
                )
                dna_match_count = db_session.query(func.count(DnaMatch.id)).scalar()
                logger.debug(
                    f"DB State Check: Total Persons={person_count}, Active={active_person_count}, Contactable={contactable_person_count}, DnaMatches={dna_match_count}"
                )
            except Exception as count_e:
                logger.error(f"Error performing DB state counts: {count_e}")

            # Corrected Initial Query
            potential_matches_query = (
                db_session.query(
                    Person.id.label("people_id"),
                    DnaMatch.predicted_relationship,
                    Person.username,
                    Person.profile_id,
                    Person.in_my_tree,
                    Person.first_name,
                    FamilyTree.actual_relationship,
                    FamilyTree.person_name_in_tree,
                    FamilyTree.relationship_path,
                    Person.status.label("person_status"),
                )
                .select_from(Person)
                .join(DnaMatch, Person.id == DnaMatch.people_id)
                .outerjoin(FamilyTree, Person.id == FamilyTree.people_id)
                .filter(
                    Person.profile_id.isnot(None),
                    Person.profile_id != "",
                    Person.profile_id != "UNKNOWN",
                    Person.contactable == True,
                    Person.status == PersonStatusEnum.ACTIVE,
                )
                .order_by(Person.id)
            )
            # Raw SQL Logging
            try:
                from sqlalchemy.dialects import sqlite

                raw_sql = str(
                    potential_matches_query.statement.compile(
                        dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}
                    )
                )
                logger.debug(f"Raw SQL for potential_matches_query:\n{raw_sql}")
            except Exception as sql_e:
                logger.warning(f"Could not compile raw SQL for logging: {sql_e}")

            potential_matches = potential_matches_query.all()
            total_potential = len(potential_matches)
            # Logging After Query
            logger.debug(
                f"Query executed. Found {total_potential} potential ACTIVE & contactable DNA matches."
            )
            if total_potential > 0:
                logger.debug(
                    f"First potential match data (sample): {potential_matches[0]}"
                )

            if not potential_matches:
                logger.info(
                    "No potential matches meet ACTIVE/Contactable criteria. Finishing."
                )
                return True
            potential_person_ids = [
                match.people_id for match in potential_matches if match.people_id
            ]
            if not potential_person_ids:
                logger.info("No valid person IDs. Finishing.")
                return True

            # Fetch latest OUTGOING script message details
            logger.debug(
                f"Pre-fetching latest OUTGOING ConversationLog for {len(potential_person_ids)} people..."
            )
            latest_outgoing_log_data: Dict[int, Tuple[str, datetime, str]] = {}
            try:
                latest_out_subq = (
                    db_session.query(
                        ConversationLog.people_id,
                        func.max(ConversationLog.latest_timestamp).label("max_ts"),
                    )
                    .filter(
                        ConversationLog.people_id.in_(potential_person_ids),
                        ConversationLog.direction == MessageDirectionEnum.OUT,
                    )
                    .group_by(ConversationLog.people_id)
                    .subquery("latest_out_ts")
                )
                latest_outgoing_logs = (
                    db_session.query(
                        ConversationLog.people_id,
                        MessageType.type_name,
                        ConversationLog.latest_timestamp,
                        ConversationLog.script_message_status,
                    )
                    .join(
                        latest_out_subq,
                        (ConversationLog.people_id == latest_out_subq.c.people_id)
                        & (
                            ConversationLog.latest_timestamp == latest_out_subq.c.max_ts
                        ),
                    )
                    .outerjoin(
                        MessageType, ConversationLog.message_type_id == MessageType.id
                    )
                    .filter(ConversationLog.direction == MessageDirectionEnum.OUT)
                    .all()
                )
                for pid, type_n, sent_db, stat in latest_outgoing_logs:
                    if sent_db:
                        sent_naive = (
                            sent_db.astimezone(timezone.utc).replace(tzinfo=None)
                            if sent_db.tzinfo
                            else sent_db.replace(tzinfo=None)
                        )
                        latest_outgoing_log_data[pid] = (
                            type_n or "Unknown",
                            sent_naive,
                            stat or "Unknown",
                        )
                logger.debug(
                    f"Fetched latest OUT Log data for {len(latest_outgoing_log_data)} people."
                )
            except SQLAlchemyError as db_err:
                logger.error(f"DB error pre-fetching OUT Logs: {db_err}", exc_info=True)
                return False
            except Exception as e:
                logger.error(
                    f"Error processing pre-fetched OUT Logs: {e}", exc_info=True
                )
                return False

            # Fetch latest INCOMING message details
            logger.debug(
                f"Pre-fetching latest INCOMING ConversationLog timestamp for {len(potential_person_ids)} people..."
            )
            latest_incoming_ts_data: Dict[int, datetime] = {}
            try:
                latest_in_subq = (
                    db_session.query(
                        ConversationLog.people_id,
                        func.max(ConversationLog.latest_timestamp).label("max_ts"),
                    )
                    .filter(
                        ConversationLog.people_id.in_(potential_person_ids),
                        ConversationLog.direction == MessageDirectionEnum.IN,
                    )
                    .group_by(ConversationLog.people_id)
                    .subquery("latest_in_ts")
                )
                latest_incoming_results = db_session.query(
                    latest_in_subq.c.people_id, latest_in_subq.c.max_ts
                ).all()
                for pid, ts_db in latest_incoming_results:
                    if ts_db:
                        ts_naive = (
                            ts_db.astimezone(timezone.utc).replace(tzinfo=None)
                            if ts_db.tzinfo
                            else ts_db.replace(tzinfo=None)
                        )
                        latest_incoming_ts_data[pid] = ts_naive
                logger.debug(
                    f"Fetched latest IN Log timestamp for {len(latest_incoming_ts_data)} people."
                )
            except SQLAlchemyError as db_err:
                logger.error(f"DB error pre-fetching IN Logs: {db_err}", exc_info=True)
                return False
            except Exception as e:
                logger.error(
                    f"Error processing pre-fetched IN Logs: {e}", exc_info=True
                )
                return False

            # Fetch existing Conversation IDs
            conversation_ids: Dict[int, str] = {}
            try:
                # Note: .distinct(column) is dialect-specific (PostgreSQL).
                # For SQLite, a common pattern is to use a subquery or group_by.
                # However, fetching all and processing in Python is often simplest if dataset isn't huge.
                # Using previous simpler query for broader compatibility:
                conv_id_results = (
                    db_session.query(
                        ConversationLog.people_id, ConversationLog.conversation_id
                    )
                    .filter(ConversationLog.people_id.in_(potential_person_ids))
                    .filter(
                        ConversationLog.conversation_id.isnot(None)
                    )  # Ensure not null
                    .group_by(
                        ConversationLog.people_id
                    )  # Get one per person (last one based on internal order)
                    .all()
                )
                conversation_ids = {pid: cid for pid, cid in conv_id_results if cid}
                logger.debug(
                    f"Fetched existing conversation IDs for {len(conversation_ids)} people."
                )
            except SQLAlchemyError as db_err:
                logger.error(f"DB error pre-fetching Conv IDs: {db_err}", exc_info=True)
                return False
            except Exception as e:
                logger.error(
                    f"Error processing pre-fetched Conv IDs: {e}", exc_info=True
                )
                return False

            logger.debug("--- Pre-fetching Finished ---")
            logger.info(f"Processing {total_potential} potential matches...\n")

            # --- Main Loop ---
            with logging_redirect_tqdm():
                progress_bar = tqdm(
                    total=total_potential,
                    desc="Sending Messages",
                    unit=" match",
                    ncols=100,
                    bar_format="{percentage:3.0f}% |{bar}|",
                    leave=True,
                )
                stop_reason = None
                for match_data in potential_matches:
                    if critical_db_error_occurred:
                        logger.warning(
                            "Aborting loop due to previous critical DB error during commit."
                        )
                        remaining_items = total_potential - processed_count
                        if progress_bar and remaining_items > 0:
                            error_count += remaining_items
                            progress_bar.update(remaining_items)
                        break

                    # Reset item variables
                    item_skipped = False
                    item_error = False
                    person_id = None
                    username = None
                    profile_id = None
                    is_in_family_tree = False
                    first_name = None
                    person_name_in_tree = None
                    original_predicted_relationship = None
                    original_actual_relationship = None
                    original_relationship_path = None
                    person_status = None
                    last_script_message_details = None
                    last_reply_timestamp = None
                    existing_conversation_id = None

                    # Check Limits
                    if max_to_send != 0 and sent_count >= max_to_send:
                        stop_reason = f"Send Limit ({max_to_send})"
                        logger.info(
                            f"Reached MAX_INBOX limit ({max_to_send}). Stopping."
                        )
                        break
                    processed_count += 1

                    # Unpack Data
                    try:
                        (
                            person_id,
                            original_predicted_relationship,
                            username,
                            profile_id,
                            is_in_family_tree,
                            first_name,
                            original_actual_relationship,
                            person_name_in_tree,
                            original_relationship_path,
                            person_status,
                        ) = match_data
                        if (
                            not person_id
                            or not profile_id
                            or not username
                            or not person_status
                        ):
                            raise ValueError("Missing essential person data")
                        assert person_id is not None
                        assert profile_id is not None
                        assert username is not None
                        assert person_status is not None
                    except (ValueError, TypeError) as unpack_e:
                        logger.error(
                            f"Error unpacking data: {unpack_e}. Row: {match_data}",
                            exc_info=True,
                        )
                        error_count += 1
                        overall_success = False
                        item_error = True
                        if progress_bar:
                            progress_bar.update(1)
                        continue

                    recipient_profile_id_upper = profile_id.upper()
                    log_prefix = f"{username} #{person_id}"

                    # Main Processing Logic for this item
                    try:
                        now_local_naive_utc = datetime.now(timezone.utc).replace(
                            tzinfo=None
                        )
                        last_script_message_details = latest_outgoing_log_data.get(
                            person_id
                        )
                        last_reply_timestamp = latest_incoming_ts_data.get(person_id)
                        existing_conversation_id = conversation_ids.get(person_id)

                        # --- Rule Checks ---
                        if person_status != PersonStatusEnum.ACTIVE:
                            logger.warning(
                                f"Skipping {log_prefix}: Status is {person_status.name} (Should be filtered)."
                            )
                            skipped_count += 1
                            item_skipped = True
                            raise StopIteration
                        user_replied_since_last_script = (
                            last_reply_timestamp
                            and last_script_message_details
                            and last_reply_timestamp > last_script_message_details[1]
                        ) or (last_reply_timestamp and not last_script_message_details)
                        if user_replied_since_last_script:
                            logger.debug(
                                f"Skipping {log_prefix}: User sent the last message (reply pending)."
                            )
                            skipped_count += 1
                            item_skipped = True
                            raise StopIteration
                        if (
                            last_script_message_details
                            and (now_local_naive_utc - last_script_message_details[1])
                            < MIN_MESSAGE_INTERVAL
                        ):
                            logger.debug(
                                f"Skipping {log_prefix}: Within minimum message interval."
                            )
                            skipped_count += 1
                            item_skipped = True
                            raise StopIteration

                        # --- Determine Next Message ---
                        next_message_type_key = determine_next_message_type(
                            last_script_message_details, bool(is_in_family_tree)
                        )
                        if not next_message_type_key:
                            logger.debug(
                                f"Skipping {log_prefix}: No next message type."
                            )
                            skipped_count += 1
                            item_skipped = True
                            raise StopIteration

                        # --- Format Message (Corrected V14.48) ---
                        message_template = MESSAGE_TEMPLATES.get(next_message_type_key)
                        if not message_template:
                            logger.error(
                                f"Missing template key '{next_message_type_key}'."
                            )
                            error_count += 1
                            overall_success = False
                            item_error = True
                            raise StopIteration
                        name_to_use = None
                        source = None
                        if is_in_family_tree and person_name_in_tree:
                            name_to_use = person_name_in_tree
                            source = "Tree"
                        elif first_name:
                            name_to_use = first_name
                            source = "Person.first"
                        elif username and username != "Unknown":
                            cleaned_username = (
                                username.replace("(managed by ", "")
                                .replace(")", "")
                                .strip()
                            )
                            if cleaned_username and cleaned_username != "Unknown User":
                                name_to_use = cleaned_username
                                source = "Person.user"
                            else:
                                source = "Person.user (unknown)"
                        else:
                            source = "Username unavailable"
                        if name_to_use is None:
                            name_to_use = "Valued Relative"
                        if source is None:
                            source = "Fallback default"
                        formatted_name = format_name(name_to_use)
                        logger.debug(
                            f"Using name '{formatted_name}' (Source: {source}) for {log_prefix}"
                        )
                        total_family_tree_rows = (
                            db_session.query(func.count(FamilyTree.id)).scalar() or 0
                        )
                        format_data = {
                            "name": formatted_name,
                            "predicted_relationship": original_predicted_relationship
                            or "N/A",
                            "actual_relationship": original_actual_relationship
                            or "N/A",
                            "relationship_path": original_relationship_path or "N/A",
                            "total_rows": total_family_tree_rows,
                        }
                        try:
                            message_text = message_template.format(**format_data)
                        except Exception as e:
                            logger.error(f"Error fmt msg for {log_prefix}: {e}.")
                            error_count += 1
                            overall_success = False
                            item_error = True
                            raise StopIteration
                        logger.debug(
                            f"Formatted message for {log_prefix}: '{message_text[:100]}...'"
                        )

                        # --- API Call Prep ---
                        message_status: str = "skipped (logic_error)"
                        new_conversation_id_from_api_local: Optional[str] = None
                        is_initial_message: bool = not existing_conversation_id
                        send_api_url: str = ""
                        payload: Dict[str, Any] = {}
                        send_api_desc: str = ""
                        api_headers = {}
                        if is_initial_message:
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
                                    {
                                        "user_id": recipient_profile_id_upper.lower(),
                                        "family_circles": [],
                                    },
                                    {"user_id": MY_PROFILE_ID_LOWER},
                                ],
                            }
                        elif existing_conversation_id:
                            send_api_url = urljoin(
                                config_instance.BASE_URL.rstrip("/") + "/",
                                f"app-api/express/v2/conversations/{existing_conversation_id}",
                            )
                            send_api_desc = "Send Message API (Existing Conv)"
                            payload = {
                                "content": message_text,
                                "author": MY_PROFILE_ID_LOWER,
                            }
                        else:
                            logger.error(
                                f"Logic Error: Cannot determine API URL/payload for {log_prefix}."
                            )
                            error_count += 1
                            overall_success = False
                            item_error = True
                            raise StopIteration
                        ctx_headers = config_instance.API_CONTEXTUAL_HEADERS.get(
                            send_api_desc, {}
                        )
                        api_headers = ctx_headers.copy()
                        if (
                            "ancestry-userid" in api_headers
                            and session_manager.my_profile_id
                        ):
                            api_headers["ancestry-userid"] = MY_PROFILE_ID_UPPER

                        # --- Execute API Call / Dry Run ---
                        if config_instance.APP_MODE == "dry_run":
                            logger.debug(
                                f"Dry Run: Would send '{next_message_type_key}' to {log_prefix}"
                            )
                            message_status = "typed (dry_run)"
                            if is_initial_message:
                                new_conversation_id_from_api_local = (
                                    f"dryrun_{uuid.uuid4()}"
                                )
                            else:
                                new_conversation_id_from_api_local = (
                                    existing_conversation_id
                                )
                        elif config_instance.APP_MODE in ["production", "testing"]:
                            action = (
                                "create" if is_initial_message else "send follow-up"
                            )
                            logger.info(
                                f"Sending ({action}) '{next_message_type_key}' to {log_prefix}..."
                            )
                            api_response: Optional[Any] = None
                            try:
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
                                logger.error(
                                    f"Exception during _api_req for {log_prefix}: {api_e}."
                                )
                                message_status = "send_error (api_exception)"
                                error_count += 1
                                overall_success = False
                                item_error = True
                                raise StopIteration

                            # --- CORRECTED API Response Validation Logic ---
                            post_ok = False
                            api_conv_id = None
                            api_author = None
                            if api_response is not None:
                                if is_initial_message:
                                    # Check format first (is dict and has key)
                                    if (
                                        isinstance(api_response, dict)
                                        and "conversation_id" in api_response
                                    ):
                                        # Now check values
                                        api_conv_id = str(
                                            api_response.get("conversation_id")
                                        )
                                        msg_details = api_response.get("message", {})
                                        api_author = (
                                            str(msg_details.get("author", "")).upper()
                                            if isinstance(msg_details, dict)
                                            else None
                                        )

                                        if (
                                            api_conv_id
                                            and api_author == MY_PROFILE_ID_UPPER
                                        ):
                                            # Values are valid
                                            post_ok = True
                                            new_conversation_id_from_api_local = (
                                                api_conv_id
                                            )
                                        else:
                                            # Values invalid (but format was okay)
                                            logger.error(
                                                f"API initial response invalid (values) for {log_prefix}. ConvID: {api_conv_id}, Author: {api_author}"
                                            )
                                    else:
                                        # Response format was unexpected
                                        logger.error(
                                            f"API call ({send_api_desc}) unexpected format (initial) for {log_prefix}. Resp:{api_response}"
                                        )
                                else:  # Follow-up message validation
                                    if (
                                        isinstance(api_response, dict)
                                        and "author" in api_response
                                    ):
                                        api_author = str(
                                            api_response.get("author", "")
                                        ).upper()
                                        if api_author == MY_PROFILE_ID_UPPER:
                                            post_ok = True
                                            new_conversation_id_from_api_local = (
                                                existing_conversation_id
                                            )
                                        else:
                                            logger.error(
                                                f"API follow-up author validation failed for {log_prefix}."
                                            )
                                    else:
                                        logger.error(
                                            f"API call ({send_api_desc}) unexpected format (follow-up) for {log_prefix}. Resp:{api_response}"
                                        )

                                # --- Final status check based on post_ok ---
                                if post_ok:
                                    message_status = "delivered OK"
                                    logger.debug(
                                        f"Message send to {log_prefix} ACCEPTED."
                                    )
                                else:
                                    message_status = "send_error (validation_failed)"
                                    error_count += 1
                                    overall_success = False
                                    item_error = True
                                    logger.warning(
                                        f"API POST validation failed for {log_prefix}."
                                    )
                                    raise StopIteration
                            else:
                                logger.error(
                                    f"API POST ({send_api_desc}) for {log_prefix} failed (No response)."
                                )
                                message_status = "send_error (post_failed)"
                                error_count += 1
                                overall_success = False
                                item_error = True
                                raise StopIteration
                        else:
                            logger.warning(
                                f"Skipping send for {log_prefix} due to APP_MODE: {config_instance.APP_MODE}"
                            )
                            skipped_count += 1
                            item_skipped = True
                            raise StopIteration

                        # --- DB Update Prep ---
                        if message_status in ("delivered OK", "typed (dry_run)"):
                            if (
                                message_status == "delivered OK"
                                or config_instance.APP_MODE == "dry_run"
                            ):
                                sent_count += 1
                            try:  # DB Prep Try (Corrected V14.46)
                                msg_type_obj = None
                                try:
                                    msg_type_obj = (
                                        db_session.query(MessageType)
                                        .filter(
                                            MessageType.type_name
                                            == next_message_type_key
                                        )
                                        .one_or_none()
                                    )
                                except Exception as mt_err:
                                    logger.error(
                                        f"Error querying MessageType ID for '{next_message_type_key}': {mt_err}",
                                        exc_info=True,
                                    )
                                    error_count += 1
                                    overall_success = False
                                    item_error = True
                                    raise StopIteration(
                                        "Failed MessageType Query"
                                    ) from mt_err
                                if not msg_type_obj:
                                    logger.error(
                                        f"CRITICAL: MessageType '{next_message_type_key}' not found!"
                                    )
                                    error_count += 1
                                    overall_success = False
                                    item_error = True
                                    raise StopIteration(
                                        f"MessageType '{next_message_type_key}' missing"
                                    )

                                current_time_for_db = datetime.now(timezone.utc)
                                trunc_msg_content = message_text[
                                    : config_instance.MESSAGE_TRUNCATION_LENGTH
                                ]
                                effective_conv_id = new_conversation_id_from_api_local
                                if not effective_conv_id:
                                    logger.error(
                                        f"DB Prep Error for {log_prefix}: Effective Conv ID missing!"
                                    )
                                    error_count += 1
                                    overall_success = False
                                    item_error = True
                                    raise StopIteration("Missing Conv ID for DB Prep")

                                new_log_out = ConversationLog(
                                    conversation_id=effective_conv_id,
                                    direction=MessageDirectionEnum.OUT,
                                    people_id=person_id,
                                    latest_message_content=trunc_msg_content,
                                    latest_timestamp=current_time_for_db,
                                    ai_sentiment=None,
                                    message_type_id=msg_type_obj.id,
                                    script_message_status=message_status,
                                    updated_at=current_time_for_db,
                                )
                                db_objects_to_commit.append(new_log_out)
                                logger.debug(
                                    f"Staged OUT ConversationLog for {log_prefix} (Type: {next_message_type_key}, Status: {message_status}, ConvID: {effective_conv_id})"
                                )

                                # --- Commit periodically (Corrected V14.55 Error Handling) ---
                                if len(db_objects_to_commit) >= db_commit_batch_size:
                                    # --- De-duplication Logic ---
                                    unique_logs = []
                                    seen_combinations = set()

                                    for log_entry in db_objects_to_commit:
                                        # Use conversation_id and direction as the unique key
                                        combination = (
                                            log_entry.conversation_id,
                                            log_entry.direction,
                                        )
                                        if combination not in seen_combinations:
                                            unique_logs.append(log_entry)
                                            seen_combinations.add(combination)
                                        else:
                                            logger.warning(
                                                f"  Duplicate log entry skipped in batch commit: {combination}"
                                            )  # Reduced verbosity to warning

                                    db_objects_to_commit_unique = (
                                        unique_logs  # Use de-duplicated list for commit
                                    )

                                    # --- Add detailed logging BEFORE commit ---
                                    logger.debug(
                                        f"Preparing to commit batch of {len(db_objects_to_commit_unique)} unique ConversationLog entries (original batch size: {len(db_objects_to_commit)})."
                                    )
                                    if logger.isEnabledFor(
                                        logging.DEBUG
                                    ):  # Only if debug logging is enabled
                                        for (
                                            log_entry
                                        ) in (
                                            db_objects_to_commit_unique
                                        ):  # Iterate over the unique list
                                            log_repr = {
                                                "conversation_id": log_entry.conversation_id,
                                                "direction": log_entry.direction.value,
                                                "people_id": log_entry.people_id,
                                                "message_type_id": log_entry.message_type_id,
                                                "script_message_status": log_entry.script_message_status,
                                            }
                                            logger.debug(
                                                f"  Log Entry for Commit: {log_repr}"
                                            )
                                    # --- End detailed logging ---

                                    # --- Attempt Commit ---
                                    logger.debug(
                                        f"Reached batch size ({db_commit_batch_size}). Committing ConvLogs..."
                                    )
                                    try:
                                        with db_transn(db_session):
                                            db_session.add_all(
                                                [
                                                    obj
                                                    for obj in db_objects_to_commit_unique
                                                    if isinstance(obj, ConversationLog)
                                                ]
                                            )
                                        logger.debug(
                                            f"Committed batch of {len(db_objects_to_commit_unique)} ConvLogs."
                                        )
                                        db_objects_to_commit = (
                                            []
                                        )  # Clear original list on SUCCESS
                                    except IntegrityError as ie:
                                        if config_instance.APP_MODE == "dry_run":
                                            logger.warning(
                                                f"Dry Run: UNIQUE constraint error during batch commit (likely duplicate dryrun_ ID): {ie}. Skipping commit for this batch, continuing run."
                                            )
                                            db_objects_to_commit = (
                                                []
                                            )  # Clear original list after rollback
                                            # Don't set critical error flag or abort loop for dry run integrity error
                                        else:  # Live mode - treat as critical
                                            logger.error(
                                                f"DB UNIQUE constraint error during batch commit: {ie}",
                                                exc_info=False,
                                            )
                                            error_count += len(
                                                db_objects_to_commit_unique
                                            )
                                            overall_success = False
                                            db_objects_to_commit = []
                                            critical_db_error_occurred = True
                                            logger.critical(
                                                "Aborting Action 8 run due to DB batch commit failure (IntegrityError)."
                                            )
                                            raise StopIteration(
                                                "DB Batch Commit Failed - IntegrityError"
                                            ) from ie
                                    except Exception as commit_err:
                                        logger.error(
                                            f"Error committing ConvLog batch: {commit_err}",
                                            exc_info=True,
                                        )
                                        error_count += len(db_objects_to_commit_unique)
                                        overall_success = False
                                        db_objects_to_commit = []
                                        critical_db_error_occurred = True
                                        logger.critical(
                                            "Aborting Action 8 run due to DB batch commit failure."
                                        )
                                        raise StopIteration(
                                            "DB Batch Commit Failed"
                                        ) from commit_err

                            except StopIteration:
                                raise  # Re-raise StopIteration
                            except (
                                ValueError,
                                SQLAlchemyError,
                                Exception,
                            ) as db_prep_err:
                                logger.error(
                                    f"DB prep error for {log_prefix}: {db_prep_err}",
                                    exc_info=True,
                                )
                                error_count += 1
                                overall_success = False
                                item_error = True
                                raise StopIteration from db_prep_err

                    except StopIteration:
                        pass
                    except Exception as inner_loop_e:
                        logger.error(
                            f"Unexpected error processing {log_prefix}: {inner_loop_e}\n",
                            exc_info=True,
                        )
                        error_count += 1
                        overall_success = False
                        item_error = True

                    finally:  # Update progress bar
                        if progress_bar:
                            progress_bar.set_postfix(
                                Sent=sent_count,
                                Skip=skipped_count,
                                Err=error_count,
                                refresh=False,
                            )
                            progress_bar.update(1)
                # --- End Main Loop for one item ---

            # --- End Main Loop (for match_data...) ---

            # Commit remaining objects after loop (Corrected V14.55 Error Handling)
            if db_objects_to_commit and not critical_db_error_occurred:
                # --- De-duplication Logic for Final Batch ---
                unique_final_logs = []
                seen_final_combinations = set()
                for final_log_entry in db_objects_to_commit:
                    final_combination = (
                        final_log_entry.conversation_id,
                        final_log_entry.direction,
                    )
                    if final_combination not in seen_final_combinations:
                        unique_final_logs.append(final_log_entry)
                        seen_final_combinations.add(final_combination)
                    else:
                        logger.warning(
                            f"  Duplicate log entry skipped in final commit: {final_combination}"
                        )
                db_objects_to_commit_final_unique = unique_final_logs
                # --- End De-duplication ---

                logger.debug(
                    f"Committing final batch of {len(db_objects_to_commit_final_unique)} unique DB objects (original remaining: {len(db_objects_to_commit)})..."
                )
                try:
                    with db_transn(db_session):
                        db_session.add_all(
                            [
                                obj
                                for obj in db_objects_to_commit_final_unique
                                if isinstance(obj, ConversationLog)
                            ]
                        )
                    logger.debug("Final DB object batch committed.")
                    db_objects_to_commit = []
                except IntegrityError as final_ie:
                    if config_instance.APP_MODE == "dry_run":
                        logger.warning(
                            f"Dry Run: UNIQUE constraint error during final batch commit: {final_ie}. Data not saved."
                        )
                        db_objects_to_commit = []
                    else:  # Live mode
                        logger.error(
                            f"DB UNIQUE constraint error during final batch commit: {final_ie}",
                            exc_info=False,
                        )
                        error_count += len(db_objects_to_commit_final_unique)
                        overall_success = False
                        db_objects_to_commit = []
                except Exception as final_commit_err:
                    logger.error(
                        f"Error committing final DB batch: {final_commit_err}",
                        exc_info=True,
                    )
                    error_count += len(db_objects_to_commit_final_unique)
                    overall_success = False
                    db_objects_to_commit = []

            logger.debug("Finished processing all potential matches.\n")

    # --- Outer Exception Handling & Final Summary (Corrected V14.55) ---
    except Exception as outer_e:
        logger.critical(
            f"CRITICAL: Critical error during message processing loop: {outer_e}",
            exc_info=True,
        )
        overall_success = False
    finally:
        if progress_bar:
            progress_bar.close()
            print("", file=sys.stderr)
        # Adjust final error count only if abort was due to a *non-dry run* critical DB error
        if critical_db_error_occurred:
            remaining_unprocessed = total_potential - processed_count
            if remaining_unprocessed > 0:
                logger.warning(
                    f"Adding {remaining_unprocessed} unprocessed items to error count due to critical DB abort."
                )
                error_count += remaining_unprocessed
        logger.info("--- Message Sending Summary ----")
        logger.info(f"  Potential Matches Checked: {total_potential}")
        logger.info(f"  Processed in Detail:     {processed_count}")
        logger.info(f"  Sent/DryRun:           {sent_count}")
        logger.info(f"  Skipped (Policy/Rule): {skipped_count}")
        logger.info(f"  Errors (API/DB/etc.):  {error_count}")
        logger.info(f"  Overall Success:       {overall_success}")
        logger.info("---------------------------------\n")

    return overall_success


# End of send_messages_to_matches


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
