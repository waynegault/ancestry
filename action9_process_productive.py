# File: action9_process_productive.py
# V0.1: Initial implementation for processing PRODUCTIVE messages.

#!/usr/bin/env python3

#####################################################
# Imports
#####################################################

# Standard library imports
import logging
import time
import json
from typing import Any, Dict, List, Optional, Tuple, cast
from datetime import datetime, timezone

# Third-party imports
import msal  # For MS Graph auth if needed directly (ms_graph_utils handles it mostly)
from sqlalchemy.orm import Session as DbSession, joinedload
from sqlalchemy.exc import IntegrityError
from sqlalchemy import desc, and_, func
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# --- Local application imports ---
from config import config_instance  # Configuration singleton
from database import (  # Database models and utilities
    ConversationLog,
    DnaMatch,  # Import for relationships if needed, though not directly used here
    FamilyTree,  # Needed for formatting messages
    MessageDirectionEnum,
    MessageType,
    Person,
    PersonStatusEnum,
    RoleType,  # Currently unused enum
    db_transn,  # Transaction context manager
)
from logging_config import logger  # Use configured logger
import ms_graph_utils  # Utility functions for MS Graph API interaction
from utils import SessionManager, _send_message_via_api, format_name  # Core utilities
from ai_interface import (  # AI interaction functions
    classify_message_intent,  # Although not used here, keep import? Maybe remove later.
    extract_and_suggest_tasks,
)
from cache import cache_result  # Caching utility (used for templates)


# --- Constants ---
PRODUCTIVE_SENTIMENT = "PRODUCTIVE"  # Sentiment string set by Action 7
ACKNOWLEDGEMENT_MESSAGE_TYPE = (
    "Productive_Reply_Acknowledgement"  # Key in messages.json
)
ACKNOWLEDGEMENT_SUBJECT = (
    "Re: Our DNA Connection - Thank You!"  # Optional: Default subject if needed
)


# --- Helper Functions ---


def _get_message_context(
    db_session: DbSession,
    person_id: int,
    limit: int = config_instance.AI_CONTEXT_MESSAGES_COUNT,
) -> List[ConversationLog]:
    """
    Fetches the last 'limit' ConversationLog entries for a given person_id,
    ordered by timestamp ascending (oldest first).

    Args:
        db_session: The active SQLAlchemy database session.
        person_id: The ID of the person whose message context is needed.
        limit: The maximum number of messages to retrieve.

    Returns:
        A list of ConversationLog objects, sorted oldest to newest, or an empty list on error.
    """
    # Step 1: Query database for recent logs
    try:
        context_logs = (
            db_session.query(ConversationLog)
            .filter(ConversationLog.people_id == person_id)
            .order_by(ConversationLog.latest_timestamp.desc())  # Fetch newest first
            .limit(limit)  # Limit the number fetched
            .all()
        )
        # Step 2: Sort the fetched logs by timestamp ascending (oldest first) for AI context
        return sorted(
            context_logs,
            key=lambda log: (
                log.latest_timestamp
                if log.latest_timestamp
                else datetime.min.replace(tzinfo=timezone.utc)
            ),
        )
    except SQLAlchemyError as e:
        # Step 3: Log database errors
        logger.error(
            f"DB error fetching message context for Person ID {person_id}: {e}",
            exc_info=True,
        )
        return []
    except Exception as e:
        # Step 4: Log unexpected errors
        logger.error(
            f"Unexpected error fetching message context for Person ID {person_id}: {e}",
            exc_info=True,
        )
        return []

# End of _get_message_context


def _format_context_for_ai_extraction(
    context_logs: List[ConversationLog], my_pid_lower: str
) -> str:
    """
    Formats a list of ConversationLog objects (sorted oldest to newest) into a
    single string suitable for the AI extraction/task suggestion prompt.
    Labels messages as "SCRIPT:" or "USER:" and truncates long messages.

    Args:
        context_logs: List of ConversationLog objects, sorted oldest to newest.
        my_pid_lower: The script user's profile ID (lowercase) for labeling.

    Returns:
        A formatted string representing the conversation history.
    """
    # Step 1: Initialize list for formatted lines
    context_lines = []
    # Step 2: Get truncation limit from config
    max_words = config_instance.AI_CONTEXT_MESSAGE_MAX_WORDS

    # Step 3: Iterate through sorted logs (oldest first)
    for log in context_logs:
        # Step 3a: Determine label based on direction
        # Note: Assumes IN logs have author != my_pid_lower, OUT logs have author == my_pid_lower
        # This might need adjustment if author field isn't reliably populated or needed.
        # Using direction is simpler and more reliable here.
        author_label = (
            "USER: " if log.direction == MessageDirectionEnum.IN else "SCRIPT: "
        )

        # Step 3b: Get message content and handle potential None
        content = log.latest_message_content or ""

        # Step 3c: Truncate content by word count if necessary
        words = content.split()
        if len(words) > max_words:
            truncated_content = " ".join(words[:max_words]) + "..."
        else:
            truncated_content = content

        # Step 3d: Append formatted line to the list
        context_lines.append(f"{author_label}{truncated_content}")

    # Step 4: Join lines into a single string separated by newlines
    return "\n".join(context_lines)
# End of _format_context_for_ai_extraction


def _search_ancestry_tree(session_manager: SessionManager, names: List[str]):
    """
    Placeholder function for searching the user's connected Ancestry tree
    (potentially via GEDCOM or API if implemented) for names extracted by the AI.

    Args:
        session_manager: The SessionManager instance (provides context if needed).
        names: A list of names extracted from the conversation.
    """
    # Step 1: Check if there are names to search for
    if names:
        # Step 2: Log placeholder message (replace with actual implementation)
        # TODO: Implement actual tree search logic (GEDCOM parsing or API calls)
        logger.info(
            f"(Placeholder) Action 9 Tree Search: Would search Ancestry tree for names: {', '.join(names)}"
        )
    else:
        # Step 3: Log if no names provided
        logger.debug(
            "(Placeholder) Action 9 Tree Search: No names extracted to search."
        )
    # Step 4: Return value (currently None, could return search results)
    return None  # Placeholder return

# End of _search_ancestry_tree


@cache_result("action9_message_templates", ignore_args=True)  # Cache templates globally
def _load_templates_for_action9() -> Dict[str, str]:
    """
    Loads message templates from messages.json using the shared loader function.
    Specifically validates the presence of the 'Productive_Reply_Acknowledgement' template.

    Returns:
        A dictionary of templates, or an empty dictionary if loading fails or the
        required acknowledgement template is missing.
    """
    # Step 1: Call the shared template loader function (defined in action8_messaging)
    from action8_messaging import (
        load_message_templates,
    )  # Local import to avoid circular dependency at module level

    all_templates = load_message_templates()

    # Step 2: Validate the required template for this action
    if ACKNOWLEDGEMENT_MESSAGE_TYPE not in all_templates:
        logger.critical(
            f"CRITICAL ERROR: Required template key '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' not found in messages.json! Action 9 cannot send acknowledgements."
        )
        return {}  # Return empty dict to signal failure

    # Step 3: Return loaded templates if validation passes
    return all_templates
# End of _load_templates_for_action9


def _commit_action9_batch(
    db_session: DbSession,
    logs_to_add: List[Dict[str, Any]],  # Data dictionaries for logs
    person_updates: Dict[int, PersonStatusEnum],  # Person ID -> Status Enum
    batch_num: int,
) -> bool:
    """
    Commits a batch of database changes specific to Action 9.
    Handles UPSERT logic for ConversationLog entries (OUT direction for ACKs)
    and bulk updates for Person statuses (typically to ARCHIVE).

    Args:
        db_session: The active SQLAlchemy database session.
        logs_to_add: List of data dictionaries for ConversationLog entries.
        person_updates: Dictionary mapping Person IDs to their new status Enum.
        batch_num: The current batch number (for logging).

    Returns:
        True if the commit was successful, False otherwise.
    """
    # Step 1: Check if there's data to commit
    if not logs_to_add and not person_updates:
        logger.debug(f"Action 9 Batch Commit (Batch {batch_num}): No data to commit.")
        return True

    logger.debug(
        f"Attempting Action 9 batch commit (Batch {batch_num}): {len(logs_to_add)} logs, {len(person_updates)} person updates..."
    )

    # Step 2: Perform DB operations within a transaction
    try:
        with db_transn(db_session) as sess:  # Use transaction context manager
            processed_logs_count = 0

            # --- Step 2a: ConversationLog Upsert ---
            # Process OUT logs (Acknowledgements) sent by this action
            if logs_to_add:
                logger.debug(
                    f" Upserting {len(logs_to_add)} OUT ConversationLog entries (Acks)..."
                )
                for log_data in logs_to_add:
                    try:
                        # Extract keys and validate direction (should always be OUT)
                        conv_id = log_data.get("conversation_id")
                        direction_str = log_data.get("direction")
                        people_id = log_data.get("people_id")
                        if direction_str != MessageDirectionEnum.OUT.value:
                            logger.error(
                                f"Logic Error: Action 9 trying to log non-OUT message? ConvID {conv_id}, Dir: {direction_str}. Skipping log."
                            )
                            continue
                        if not conv_id or not people_id:
                            logger.error(
                                f"Missing conv_id/people_id in Action 9 log data: {log_data}. Skipping log."
                            )
                            continue
                        direction_enum = MessageDirectionEnum.OUT

                        # Find existing OUT log for this conversation
                        existing_log = (
                            sess.query(ConversationLog)
                            .filter_by(
                                conversation_id=conv_id, direction=direction_enum
                            )
                            .with_for_update()
                            .first()
                        )  # Lock row

                        # Prepare update values (ensure timestamp is aware)
                        ts_val = log_data.get("latest_timestamp")
                        aware_timestamp = (
                            ts_val.astimezone(timezone.utc)
                            if isinstance(ts_val, datetime) and ts_val.tzinfo
                            else (
                                ts_val.replace(tzinfo=timezone.utc)
                                if isinstance(ts_val, datetime)
                                else None
                            )
                        )
                        if not aware_timestamp:
                            logger.error(
                                f"Invalid timestamp for ACK log ConvID {conv_id}. Skipping log."
                            )
                            continue

                        update_values = {
                            k: v
                            for k, v in log_data.items()
                            if k
                            not in [
                                "conversation_id",
                                "direction",
                                "created_at",
                                "updated_at",
                            ]
                            and (
                                v is not None
                                or k
                                in [
                                    "ai_sentiment",
                                    "message_type_id",
                                    "script_message_status",
                                ]
                            )
                        }  # Allow specific nulls
                        update_values["latest_timestamp"] = aware_timestamp
                        update_values["updated_at"] = datetime.now(timezone.utc)

                        if existing_log:
                            # Update existing OUT log (e.g., if resending ACK?)
                            # logger.debug(f"  Updating existing OUT log for {conv_id} (PersonID: {people_id})")
                            for key, value in update_values.items():
                                setattr(existing_log, key, value)
                        else:
                            # Insert new OUT log
                            # logger.debug(f"  Inserting new OUT log for {conv_id} (PersonID: {people_id})")
                            new_log_obj = ConversationLog(
                                conversation_id=conv_id,
                                direction=direction_enum,
                                **update_values,
                            )
                            sess.add(new_log_obj)
                        processed_logs_count += 1
                    except Exception as inner_log_exc:
                        logger.error(
                            f" Error processing single ACK log item (ConvID: {log_data.get('conversation_id')}, PersonID: {log_data.get('people_id')}): {inner_log_exc}",
                            exc_info=True,
                        )
                logger.debug(
                    f" Finished processing {processed_logs_count} ACK log entries for upsert."
                )
            # --- End Log Upsert ---

            # --- Step 2b: Person Status Updates (to ARCHIVE) ---
            if person_updates:
                update_mappings = []
                logger.debug(
                    f" Preparing {len(person_updates)} Person status updates (to ARCHIVE)..."
                )
                for pid, status_enum in person_updates.items():
                    if status_enum != PersonStatusEnum.ARCHIVE:
                        logger.warning(
                            f"Action 9 attempting non-ARCHIVE update for Person {pid} to {status_enum.name}? Expected ARCHIVE. Applying ARCHIVE."
                        )
                        status_to_set = PersonStatusEnum.ARCHIVE  # Force ARCHIVE
                    else:
                        status_to_set = status_enum
                    # logger.debug(f"  Prep update Person ID {pid}: status -> {status_to_set.name}")
                    update_mappings.append(
                        {
                            "id": pid,
                            "status": status_to_set,  # Pass Enum for bulk update
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                # Perform bulk update
                if update_mappings:
                    logger.debug(
                        f" Updating {len(update_mappings)} Person statuses via bulk..."
                    )
                    sess.bulk_update_mappings(Person, update_mappings)
            # --- End Person Update ---

        # --- Commit happens automatically via db_transn context manager ---
        logger.debug(f"Action 9 Batch commit successful (Batch {batch_num}).")
        return True

    # Step 3: Handle DB errors during commit
    except IntegrityError as ie:
        logger.error(
            f"DB UNIQUE constraint error during Action 9 batch commit (Batch {batch_num}): {ie}",
            exc_info=False,
        )
        return False
    except Exception as e:
        logger.error(
            f"Error committing Action 9 batch (Batch {batch_num}): {e}", exc_info=True
        )
        return False
# End of _commit_action9_batch

# ------------------------------------------------------------------------------
# Main Function: process_productive_messages
# ------------------------------------------------------------------------------


def process_productive_messages(session_manager: SessionManager) -> bool:
    """
    Main function for Action 9. Finds persons with recent 'PRODUCTIVE' messages,
    extracts info/tasks via AI, creates MS To-Do tasks, sends acknowledgements,
    and updates database status.

    Args:
        session_manager: The active SessionManager instance.

    Returns:
        True if the process completed without critical database errors, False otherwise.
    """
    # --- Step 1: Initialization ---
    logger.info("--- Starting Action 9: Process Productive Messages ---")
    if not session_manager or not session_manager.my_profile_id:
        logger.error("Action 9: SM/Profile ID missing.")
        return False
    my_pid_lower = session_manager.my_profile_id.lower()
    overall_success = True
    # Counters for summary
    processed_count = 0
    tasks_created_count = 0
    acks_sent_count = 0
    archived_count = 0  # Persons staged for ARCHIVE
    error_count = 0
    skipped_count = 0
    total_candidates = 0
    # MS Graph state
    ms_graph_token: Optional[str] = None
    ms_list_id: Optional[str] = None
    ms_list_name = config_instance.MS_TODO_LIST_NAME
    ms_auth_attempted = False  # Track if we tried auth this run
    # Batching state
    batch_num = 0
    critical_db_error_occurred = False
    logs_to_add: List[Dict[str, Any]] = []  # Store ACK log data for batch commit
    person_updates: Dict[int, PersonStatusEnum] = (
        {}
    )  # Store {pid: ARCHIVE} for batch commit
    batch_size = max(1, config_instance.BATCH_SIZE)  # DB commit batch size
    commit_threshold = batch_size
    # Processing limit
    limit = config_instance.MAX_PRODUCTIVE_TO_PROCESS  # 0 = unlimited

    # Step 2: Load and Validate Templates
    message_templates = _load_templates_for_action9()
    if not message_templates:  # Loader logs critical error if required template missing
        logger.error("Action 9: Required message templates failed to load. Aborting.")
        return False
    ack_template = message_templates[ACKNOWLEDGEMENT_MESSAGE_TYPE]

    # Step 3: Get DB Session and Required MessageType ID
    db_session: Optional[DbSession] = None
    try:
        db_session = session_manager.get_db_conn()
        if not db_session:
            logger.error("Action 9: Failed get DB session.")
            return False

        # Get the DB ID for the Acknowledgement message type
        ack_msg_type_obj = (
            db_session.query(MessageType.id)
            .filter(MessageType.type_name == ACKNOWLEDGEMENT_MESSAGE_TYPE)
            .scalar()
        )
        if not ack_msg_type_obj:
            logger.error(
                f"Action 9: MessageType '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' not found in DB. Aborting."
            )
            return False
        ack_msg_type_id = ack_msg_type_obj  # Store the ID

        # --- Step 4: Query Candidate Persons ---
        logger.debug(
            "Querying for candidate Persons (Status ACTIVE, Sentiment PRODUCTIVE)..."
        )
        # Subquery to get the timestamp of the latest IN message for each person
        latest_in_log_subq = (
            db_session.query(
                ConversationLog.people_id,
                func.max(ConversationLog.latest_timestamp).label("max_in_ts"),
            )
            .filter(ConversationLog.direction == MessageDirectionEnum.IN)
            .group_by(ConversationLog.people_id)
            .subquery("latest_in_sub")
        )
        # Subquery to get the timestamp of the latest OUT message (ACK) for each person
        latest_ack_out_log_subq = (
            db_session.query(
                ConversationLog.people_id,
                func.max(ConversationLog.latest_timestamp).label("max_ack_out_ts"),
            )
            .filter(
                ConversationLog.direction == MessageDirectionEnum.OUT,
                ConversationLog.message_type_id
                == ack_msg_type_id,  # Filter specifically for ACK type
            )
            .group_by(ConversationLog.people_id)
            .subquery("latest_ack_out_sub")
        )

        # Main query to find candidates
        candidates_query = (
            db_session.query(Person)
            .options(
                joinedload(Person.family_tree)
            )  # Eager load tree for formatting ACK
            # Join to get the latest IN log's timestamp and sentiment
            .join(latest_in_log_subq, Person.id == latest_in_log_subq.c.people_id)
            .join(
                ConversationLog,
                and_(
                    Person.id == ConversationLog.people_id,
                    ConversationLog.direction == MessageDirectionEnum.IN,
                    ConversationLog.latest_timestamp == latest_in_log_subq.c.max_in_ts,
                    ConversationLog.ai_sentiment
                    == PRODUCTIVE_SENTIMENT,  # Filter by sentiment
                ),
            )
            # Left join to find the latest ACK sent (if any)
            .outerjoin(
                latest_ack_out_log_subq,
                Person.id == latest_ack_out_log_subq.c.people_id,
            )
            .filter(
                Person.status == PersonStatusEnum.ACTIVE,  # Must be ACTIVE status
                # Filter out persons where an ACK has already been sent *after* the productive message
                (
                    latest_ack_out_log_subq.c.max_ack_out_ts == None
                )  # No ACK ever sent OR
                | (
                    latest_ack_out_log_subq.c.max_ack_out_ts
                    < latest_in_log_subq.c.max_in_ts
                ),  # Last ACK older than productive msg
            )
            .order_by(Person.id)  # Consistent processing order
        )

        # Apply limit if configured
        if limit > 0:
            candidates_query = candidates_query.limit(limit)
            logger.debug(
                f"Action 9 Processing limited to {limit} productive candidates..."
            )

        candidates: List[Person] = candidates_query.all()
        total_candidates = len(candidates)

        if not candidates:
            logger.info(
                "Action 9: No eligible ACTIVE persons found with unprocessed PRODUCTIVE messages."
            )
            return True  # Nothing to do, considered success

        logger.info(
            f"Action 9: Found {total_candidates} candidates with productive messages to process."
        )

        # --- Step 5: Processing Loop ---
        tqdm_args = {
            "total": total_candidates,
            "desc": "Processing Productive",
            "unit": " person",
            "ncols": 100,
            "leave": True,
            "bar_format": "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [Tasks:{postfix[t]}, ACKs:{postfix[a]}, Skip:{postfix[s]}, Err:{postfix[e]}]",
        }
        logger.info("Processing candidates...")
        with logging_redirect_tqdm(), tqdm(
            **tqdm_args, postfix={"t": 0, "a": 0, "s": 0, "e": 0}
        ) as progress_bar:
            for person in candidates:
                processed_count += 1
                log_prefix = f"Productive: {person.username} #{person.id}"
                person_success = True  # Flag for this person's processing outcome

                # Check if loop should stop due to critical DB error
                if critical_db_error_occurred:
                    logger.warning(
                        f"Skipping remaining candidates ({total_candidates - processed_count + 1}) due to previous DB commit error."
                    )
                    error_count += (
                        total_candidates - processed_count + 1
                    )  # Count remaining as errors
                    progress_bar.update(
                        total_candidates - processed_count + 1
                    )  # Update bar fully
                    break  # Exit loop

                try:
                    # --- Step 5a: Rate Limit ---
                    wait_time = session_manager.dynamic_rate_limiter.wait()
                    # Optional log: if wait_time > 0.1: logger.debug(f"{log_prefix}: Rate limit wait: {wait_time:.2f}s")

                    # --- Step 5b: Get Message Context ---
                    # We need context for the AI extraction call
                    logger.debug(f"{log_prefix}: Getting message context...")
                    context_logs = _get_message_context(db_session, person.id)
                    if not context_logs:
                        logger.warning(
                            f"Skipping {log_prefix}: Failed to retrieve message context."
                        )
                        skipped_count += 1
                        person_success = False
                        continue  # Skip this person

                    # --- Step 5c: Call AI for Extraction & Task Suggestion ---
                    formatted_context = _format_context_for_ai_extraction(
                        context_logs, my_pid_lower
                    )
                    logger.debug(
                        f"{log_prefix}: Calling AI for extraction/task suggestion..."
                    )
                    if (
                        not session_manager.is_sess_valid()
                    ):  # Check session before AI call
                        logger.error(
                            f"Session invalid before AI extraction call for {log_prefix}. Skipping person."
                        )
                        error_count += 1
                        person_success = False
                        continue
                    ai_response = extract_and_suggest_tasks(
                        formatted_context, session_manager
                    )

                    # --- Step 5d: Process AI Response ---
                    # Initialize with default empty structures
                    extracted_data: Dict[str, List[str]] = {
                        "mentioned_names": [],
                        "mentioned_locations": [],
                        "mentioned_dates": [],
                        "potential_relationships": [],
                        "key_facts": [],
                    }
                    suggested_tasks: List[str] = []
                    summary_for_ack = "your message"  # Default summary for ACK

                    if ai_response:  # Check if AI call returned something
                        # --- Validate AI JSON Response Structure ---
                        if isinstance(ai_response, dict):
                            extracted_data_raw = ai_response.get(
                                "extracted_data"
                            )  # Safely get top-level key
                            suggested_tasks_raw = ai_response.get(
                                "suggested_tasks"
                            )  # Safely get top-level key

                            # Validate 'extracted_data' structure and content
                            if isinstance(extracted_data_raw, dict):
                                # Safely get nested lists, default to empty list if key missing or not a list
                                extracted_data["mentioned_names"] = (
                                    extracted_data_raw.get("mentioned_names", [])
                                    if isinstance(
                                        extracted_data_raw.get("mentioned_names"), list
                                    )
                                    else []
                                )
                                extracted_data["mentioned_locations"] = (
                                    extracted_data_raw.get("mentioned_locations", [])
                                    if isinstance(
                                        extracted_data_raw.get("mentioned_locations"),
                                        list,
                                    )
                                    else []
                                )
                                extracted_data["mentioned_dates"] = (
                                    extracted_data_raw.get("mentioned_dates", [])
                                    if isinstance(
                                        extracted_data_raw.get("mentioned_dates"), list
                                    )
                                    else []
                                )
                                extracted_data["potential_relationships"] = (
                                    extracted_data_raw.get(
                                        "potential_relationships", []
                                    )
                                    if isinstance(
                                        extracted_data_raw.get(
                                            "potential_relationships"
                                        ),
                                        list,
                                    )
                                    else []
                                )
                                extracted_data["key_facts"] = (
                                    extracted_data_raw.get("key_facts", [])
                                    if isinstance(
                                        extracted_data_raw.get("key_facts"), list
                                    )
                                    else []
                                )
                                logger.debug(
                                    f"{log_prefix}: Successfully parsed 'extracted_data'."
                                )
                            else:
                                logger.warning(
                                    f"{log_prefix}: AI response missing 'extracted_data' dictionary or has incorrect type: {type(extracted_data_raw)}. Using defaults."
                                )
                                # extracted_data already defaults to empty lists

                            # Validate 'suggested_tasks' structure and content
                            if isinstance(suggested_tasks_raw, list):
                                # Filter to ensure all items in the list are strings
                                suggested_tasks = [
                                    str(item)
                                    for item in suggested_tasks_raw
                                    if isinstance(item, str)
                                ]
                                if len(suggested_tasks) != len(suggested_tasks_raw):
                                    logger.warning(
                                        f"{log_prefix}: Some items in 'suggested_tasks' were not strings and were filtered out."
                                    )
                                logger.debug(
                                    f"{log_prefix}: Successfully parsed 'suggested_tasks' ({len(suggested_tasks)} valid tasks found)."
                                )
                            else:
                                logger.warning(
                                    f"{log_prefix}: AI response missing 'suggested_tasks' list or has incorrect type: {type(suggested_tasks_raw)}. Using defaults."
                                )
                                # suggested_tasks already defaults to empty list

                            # --- Generate Summary for ACK ---
                            # (This logic now safely uses the validated extracted_data dictionary)
                            summary_parts = []
                            if extracted_data.get("mentioned_names"):
                                summary_parts.append(
                                    f"names ({', '.join(extracted_data['mentioned_names'])})"
                                )
                            if extracted_data.get("mentioned_locations"):
                                summary_parts.append(
                                    f"locations ({', '.join(extracted_data['mentioned_locations'])})"
                                )
                            if extracted_data.get("mentioned_dates"):
                                summary_parts.append(
                                    f"dates ({', '.join(extracted_data['mentioned_dates'])})"
                                )
                            if extracted_data.get("key_facts"):
                                summary_parts.append(
                                    f"key facts ('{'; '.join(extracted_data['key_facts'])}')"
                                )
                            if summary_parts:
                                summary_for_ack = "information regarding " + ", ".join(
                                    summary_parts
                                )
                            else:
                                logger.debug(
                                    f"{log_prefix}: No specific details extracted by AI to include in ACK summary."
                                )
                                # summary_for_ack remains "your message"

                        else:  # AI response was not None, but JSON structure invalid at the top level
                            logger.warning(
                                f"{log_prefix}: AI extraction response was None or not a dictionary. Cannot process."
                            )
                            # Proceed without extracted data/tasks, using defaults

                    else:  # AI call failed completely (extract_and_suggest_tasks returned None)
                        logger.warning(
                            f"{log_prefix}: AI extraction/task suggestion call failed. Proceeding without AI data."
                        )
                        # Proceed without extracted data/tasks, using defaults

                    # --- Step 5e: Optional Tree Search ---
                    # (This logic now uses the validated extracted_data dictionary)
                    # TODO: Implement tree search based on extracted_data.get('mentioned_names', [])
                    tree_search_results = _search_ancestry_tree(
                        session_manager, extracted_data.get("mentioned_names", [])
                    )
                    # Use tree_search_results to refine tasks or ACK summary

                    # --- Step 5f: MS Graph Task Creation ---
                    # (This logic now uses the validated suggested_tasks list)
                    if suggested_tasks:
                        # Ensure MS Graph Auth Token exists (attempt flow if first time)
                        if not ms_graph_token and not ms_auth_attempted:
                            logger.info(
                                "Attempting MS Graph authentication (device flow)..."
                            )
                            ms_graph_token = ms_graph_utils.acquire_token_device_flow()
                            ms_auth_attempted = True  # Mark that we tried
                            if not ms_graph_token:
                                logger.error("MS Graph authentication failed.")

                        # Ensure MS List ID exists
                        if ms_graph_token and not ms_list_id:
                            logger.info(
                                f"Looking up MS To-Do List ID for '{ms_list_name}'..."
                            )
                            ms_list_id = ms_graph_utils.get_todo_list_id(
                                ms_graph_token, ms_list_name
                            )
                            if not ms_list_id:
                                logger.error(
                                    f"Failed find/get MS List ID for '{ms_list_name}'. Tasks cannot be created."
                                )

                        # Create tasks if token and list ID are available (respect app mode)
                        if ms_graph_token and ms_list_id:
                            app_mode_for_tasks = config_instance.APP_MODE
                            if app_mode_for_tasks == "dry_run":
                                logger.info(
                                    f"{log_prefix}: DRY RUN - Skipping MS To-Do task creation for {len(suggested_tasks)} suggested tasks."
                                )
                            else:  # Create tasks in 'testing' and 'production' modes
                                logger.info(
                                    f"{log_prefix}: Creating {len(suggested_tasks)} MS To-Do tasks (Mode: {app_mode_for_tasks})..."
                                )
                                for task_index, task_desc in enumerate(
                                    suggested_tasks
                                ):  # Use validated list
                                    # Format task title and body
                                    task_title = f"Ancestry Follow-up: {person.username or 'Unknown'} (#{person.id})"
                                    # Add more context to body (e.g., link, summary)
                                    task_body = f"AI Suggested Task ({task_index+1}/{len(suggested_tasks)}): {task_desc}\n\nMatch: {person.username} (#{person.id})\nProfile: {person.profile_id or 'N/A'}\nConvID: {context_logs[-1].conversation_id if context_logs else 'N/A'}"  # Example context
                                    # Call utility to create the task
                                    task_ok = ms_graph_utils.create_todo_task(
                                        ms_graph_token,
                                        ms_list_id,
                                        task_title,
                                        task_body,
                                    )
                                    if task_ok:
                                        tasks_created_count += 1
                                    else:
                                        logger.warning(
                                            f"{log_prefix}: Failed to create MS task: '{task_desc[:100]}...'"
                                        )
                        elif (
                            suggested_tasks
                        ):  # Log if tasks suggested but cannot be created (uses validated list)
                            logger.warning(
                                f"{log_prefix}: Skipping MS task creation ({len(suggested_tasks)} tasks) - MS Auth/List ID unavailable."
                            )

                    # --- Step 5g: Format Acknowledgement Message ---
                    # (This logic now uses the validated 'summary_for_ack')
                    try:
                        # Use best available name for greeting
                        name_to_use_ack = format_name(
                            person.first_name or person.username
                        )
                        message_text = ack_template.format(
                            name=name_to_use_ack, summary=summary_for_ack
                        )  # Use potentially updated summary
                    except KeyError as ke:
                        logger.error(
                            f"{log_prefix}: ACK template formatting error (Key {ke}). Using generic fallback."
                        )
                        message_text = f"Dear {format_name(person.username)},\n\nThank you for your message and the information!\n\nWayne"
                    except Exception as fmt_e:
                        logger.error(
                            f"{log_prefix}: Unexpected ACK formatting error: {fmt_e}. Using generic fallback."
                        )
                        message_text = f"Dear {format_name(person.username)},\n\nThank you!\n\nWayne"

                    # --- Step 5h: Apply Mode/Recipient Filtering ---
                    app_mode = config_instance.APP_MODE
                    testing_profile_id_config = config_instance.TESTING_PROFILE_ID
                    current_profile_id = (
                        person.profile_id or "UNKNOWN"
                    )  # Already uppercase if exists
                    send_ack_flag = True
                    skip_log_reason_ack = ""

                    if app_mode == "testing":
                        if not testing_profile_id_config:
                            logger.error(
                                f"Testing mode, but TESTING_PROFILE_ID not set. Skipping ACK for {log_prefix}."
                            )
                            send_ack_flag = False
                            skip_log_reason_ack = "skipped (config_error)"
                        elif current_profile_id != testing_profile_id_config:
                            send_ack_flag = False
                            skip_log_reason_ack = f"skipped (testing_mode_filter: not {testing_profile_id_config})"
                            logger.info(
                                f"Testing Mode: Skipping ACK send to {log_prefix} ({skip_log_reason_ack})."
                            )
                    elif (
                        app_mode == "production"
                        and testing_profile_id_config
                        and current_profile_id == testing_profile_id_config
                    ):
                        send_ack_flag = False
                        skip_log_reason_ack = f"skipped (production_mode_filter: is {testing_profile_id_config})"
                        logger.info(
                            f"Production Mode: Skipping ACK send to test profile {log_prefix} ({skip_log_reason_ack})."
                        )
                    # dry_run handled by _send_message_via_api

                    # --- Step 5i: Send/Simulate Acknowledgement Message ---
                    if send_ack_flag:
                        logger.info(
                            f"{log_prefix}: Sending/Simulating '{ACKNOWLEDGEMENT_MESSAGE_TYPE}'..."
                        )
                        # Need conversation ID (use latest context message's ID)
                        conv_id_for_send = (
                            context_logs[-1].conversation_id if context_logs else None
                        )
                        if not conv_id_for_send:
                            logger.error(
                                f"{log_prefix}: Cannot find conversation ID to send ACK. Skipping send."
                            )
                            error_count += 1
                            person_success = False
                            continue  # Skip person if no conv ID
                        # Call API send helper
                        send_status, _ = _send_message_via_api(
                            session_manager,
                            person,
                            message_text,
                            conv_id_for_send,
                            log_prefix,
                        )
                    else:  # Message filtered out
                        send_status = skip_log_reason_ack  # Use skip reason as status
                        conv_id_for_send = (
                            context_logs[-1].conversation_id
                            if context_logs
                            else f"skipped_{uuid.uuid4()}"
                        )  # Use placeholder if needed

                    # --- Step 5j: Stage Database Updates ---
                    if send_status in (
                        "delivered OK",
                        "typed (dry_run)",
                    ) or send_status.startswith("skipped ("):
                        if send_ack_flag:
                            acks_sent_count += 1  # Count only actual/simulated sends
                        logger.info(
                            f"{log_prefix}: Staging DB updates for ACK (Status: {send_status})."
                        )
                        # Prepare OUT log data dictionary
                        log_data = {
                            "conversation_id": conv_id_for_send,
                            "direction": MessageDirectionEnum.OUT.value,  # Use value for dict
                            "people_id": person.id,
                            "latest_message_content": (
                                f"[{send_status.upper()}] {message_text}"
                                if not send_ack_flag
                                else message_text
                            )[: config_instance.MESSAGE_TRUNCATION_LENGTH],
                            "latest_timestamp": datetime.now(timezone.utc),
                            "message_type_id": ack_msg_type_id,
                            "script_message_status": send_status,
                            "ai_sentiment": None,
                        }
                        logs_to_add.append(log_data)
                        # Stage Person status update to ARCHIVE
                        person_updates[person.id] = PersonStatusEnum.ARCHIVE
                        archived_count += 1
                        logger.debug(f"{log_prefix}: Person status staged for ARCHIVE.")
                    else:  # Handle actual send failure from API
                        logger.error(
                            f"{log_prefix}: Failed to send ACK (Status: {send_status}). No DB changes staged for this person."
                        )
                        error_count += 1
                        person_success = False

                    # --- Step 5k: Trigger Batch Commit ---
                    if (len(logs_to_add) + len(person_updates)) >= commit_threshold:
                        batch_num += 1
                        logger.info(
                            f"Commit threshold reached ({len(logs_to_add)} logs). Committing Action 9 Batch {batch_num}..."
                        )
                        commit_ok = _commit_action9_batch(
                            db_session, logs_to_add, person_updates, batch_num
                        )
                        if commit_ok:
                            logs_to_add.clear()
                            person_updates.clear()
                        else:
                            logger.critical(
                                f"CRITICAL: Action 9 Batch commit {batch_num} FAILED."
                            )
                            critical_db_error_occurred = True
                            overall_success = False
                            break  # Stop outer loop

                # --- Step 6: Handle errors during individual person processing ---
                except StopIteration as si:  # Catch clean skips/stops
                    status_val = str(si.value) if si.value else "skipped"
                    logger.debug(
                        f"{log_prefix}: Processing stopped cleanly for this person. Status: '{status_val}'."
                    )
                    if status_val.startswith("skipped"):
                        skipped_count += 1
                    elif status_val.startswith("error"):
                        error_count += 1
                        person_success = False
                except Exception as person_proc_err:
                    logger.error(
                        f"CRITICAL error processing {log_prefix}: {person_proc_err}",
                        exc_info=True,
                    )
                    error_count += 1
                    person_success = False

                # --- Step 7: Update overall success and progress bar ---
                finally:
                    if not person_success:
                        overall_success = False  # Mark failure if any person failed
                    # Update progress bar postfix data AND advance bar
                    progress_bar.set_postfix(
                        t=tasks_created_count,
                        a=acks_sent_count,
                        s=skipped_count,
                        e=error_count,
                        refresh=False,
                    )
                    progress_bar.update(1)
            # --- End Main Person Processing Loop ---

        # --- Step 8: Final Commit for any remaining data ---
        if not critical_db_error_occurred and (logs_to_add or person_updates):
            batch_num += 1
            logger.info(
                f"Committing final Action 9 batch (Batch {batch_num}) with {len(logs_to_add)} logs, {len(person_updates)} updates..."
            )
            final_commit_ok = _commit_action9_batch(
                db_session, logs_to_add, person_updates, batch_num
            )
            if not final_commit_ok:
                logger.error("Final Action 9 batch commit FAILED.")
                overall_success = False
            else:
                logs_to_add.clear()
                person_updates.clear()

    # --- Step 9: Handle Outer Exceptions ---
    except Exception as outer_e:
        logger.critical(
            f"CRITICAL: Unhandled exception in process_productive_messages: {outer_e}",
            exc_info=True,
        )
        overall_success = False
    # --- Step 10: Final Cleanup and Summary ---
    finally:
        if db_session:
            session_manager.return_session(db_session)  # Ensure session is returned

        # Log Summary
        print(" ")  # Spacer
        logger.info("------ Action 9: Process Productive Summary -------")
        # Adjust counts if stopped early
        final_processed = processed_count
        final_errors = error_count
        if critical_db_error_occurred and total_candidates > processed_count:
            unprocessed = total_candidates - processed_count
            logger.warning(
                f"Adding {unprocessed} unprocessed candidates to error count due to DB failure."
            )
            final_errors += unprocessed

        logger.info(f"  Candidates Queried:         {total_candidates}")
        logger.info(f"  Candidates Processed:       {final_processed}")
        logger.info(f"  Skipped (Rules/Filter):     {skipped_count}")
        logger.info(f"  MS To-Do Tasks Created:     {tasks_created_count}")
        logger.info(f"  Acks Sent/Simulated:        {acks_sent_count}")
        logger.info(f"  Persons Archived (Staged):  {archived_count}")
        logger.info(f"  Errors during processing:   {final_errors}")
        logger.info(f"  Overall Success:            {overall_success}")
        logger.info("--------------------------------------------------\n")

    # Step 11: Return overall success status
    return overall_success


# End of process_productive_messages
# end process_productive_messages

# --- End of action9_process_productive.py ---
