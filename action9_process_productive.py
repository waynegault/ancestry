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

# Local application imports
from config import config_instance
from database import (
    Person,
    ConversationLog,
    MessageType,
    PersonStatusEnum,
    MessageDirectionEnum,
    db_transn,
)
from utils import SessionManager, _send_message_via_api, format_name
from ai_interface import (
    classify_message_intent,
)  # Keep for now, might need new func later
import ms_graph_utils  # Import the whole module
from cache import cache_result  # Use cache for message templates

#####################################################
# Initialize Logging & Constants
#####################################################

logger = logging.getLogger("logger")
PRODUCTIVE_SENTIMENT = "PRODUCTIVE"
ACKNOWLEDGEMENT_MESSAGE_TYPE = "Productive_Reply_Acknowledgement"
ACKNOWLEDGEMENT_SUBJECT = "Re: Our DNA Connection - Thank You!"  # Default subject if needed separate from template


#####################################################
# Placeholder / Helper Functions
#####################################################

def _get_message_context(
    db_session: DbSession,
    person_id: int,
    limit: int = config_instance.AI_CONTEXT_MESSAGES_COUNT,
) -> List[ConversationLog]:
    """Fetches the last 'limit' messages for a person, ordered by timestamp."""
    try:
        # Fetch both IN and OUT logs, order by timestamp, take the last 'limit'
        context_logs = (
            db_session.query(ConversationLog)
            .filter(ConversationLog.people_id == person_id)
            .order_by(ConversationLog.latest_timestamp.desc())
            .limit(limit)
            .all()
        )
        # Reverse to get oldest first for AI processing
        return sorted(context_logs, key=lambda log: log.latest_timestamp)
    except Exception as e:
        logger.error(
            f"Error fetching message context for Person ID {person_id}: {e}",
            exc_info=True,
        )
        return []
# End of _get_message_context


def _format_context_for_ai_extraction(
    context_logs: List[ConversationLog], my_pid_lower: str
) -> str:
    """Formats message history for the AI extraction prompt."""
    context_lines = []
    max_words = config_instance.AI_CONTEXT_MESSAGE_MAX_WORDS
    for log in context_logs:
        # Determine author label
        author_label = "USER: "  # Assume USER unless it's an OUT log
        if log.direction == MessageDirectionEnum.OUT:
            author_label = "SCRIPT: "

        # Get content and truncate if necessary
        content = log.latest_message_content or ""
        words = content.split()
        if len(words) > max_words:
            truncated_content = " ".join(words[:max_words]) + "..."
        else:
            truncated_content = content

        context_lines.append(f"{author_label}{truncated_content}")

    return "\n".join(context_lines)
# End of _format_context_for_ai_extraction


def _call_ai_for_extraction(
    context_history: str, session_manager: SessionManager
) -> Optional[Dict[str, Any]]:
    """
    Placeholder for calling the AI model to extract information and suggest tasks.
    TODO: Replace with actual call to a potentially new function in ai_interface.py
          or modify classify_message_intent if suitable.
    """
    logger.debug("--- SIMULATING AI EXTRACTION & TASK SUGGESTION ---")
    # Simulate waiting for AI
    time.sleep(0.5)

    # --- Simulated Response Structure ---
    # In a real implementation, this would come from the AI model after parsing
    simulated_response = {
        "extracted_data": {
            "mentioned_names": ["John Smith", "Mary Anne Jones"],
            "mentioned_locations": ["Glasgow", "County Cork"],
            "mentioned_dates": ["abt 1880", "1912"],
            "potential_relationships": ["Grandfather", "Possible sibling match"],
            "key_facts": ["Immigrated via Liverpool", "Worked in coal mines"],
        },
        "suggested_tasks": [
            "Check 1881 Scotland Census for John Smith in Glasgow.",
            "Search immigration records for Mary Anne Jones arriving Liverpool around 1910-1915.",
            "Compare shared matches between Wayne and this match.",
            "Review County Cork birth records for Jones around 1880.",
        ],
    }
    # Simulate potential failure
    # return None
    # Simulate empty extraction
    # simulated_response["extracted_data"] = {}
    # simulated_response["suggested_tasks"] = []

    logger.debug(
        f"--- SIMULATED AI RESPONSE: {json.dumps(simulated_response, indent=2)} ---"
    )
    return simulated_response
# End of _call_ai_for_extraction


# Optional placeholder - not implemented in this phase
def _search_ancestry_tree(session_manager: SessionManager, names: List[str]):
    """Placeholder for searching the user's tree for extracted names."""
    if names:
        logger.info(
            f"(Placeholder) Would search Ancestry tree for names: {', '.join(names)}"
        )
    else:
        logger.debug("(Placeholder) No names extracted to search in tree.")
    # TODO: Implement using Ancestry API if needed.
# End of _search_ancestry_tree


@cache_result("action9_message_templates", ignore_args=True)
def _load_templates_for_action9() -> Dict[str, str]:
    """Loads templates, specifically ensuring the ACK template exists."""
    # Use helper from action8, slightly modified for clarity
    from action8_messaging import load_message_templates  # Local import

    all_templates = load_message_templates()
    if ACKNOWLEDGEMENT_MESSAGE_TYPE not in all_templates:
        logger.critical(
            f"CRITICAL: Template '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' missing from messages.json!"
        )
        return {}  # Return empty if critical template missing
    # Return only needed templates or all? Return all for now.
    return all_templates
# End of _load_templates_for_action9


def _commit_action9_batch(
    db_session: DbSession,
    logs_to_add: List[Dict[str, Any]],
    person_updates: Dict[int, PersonStatusEnum],
    batch_num: int,
) -> bool:
    """
    V0.2: Implements Upsert logic for logs.
    Commits a batch of ConversationLog additions/updates and Person status updates.
    Uses individual add/update for logs and bulk_update_mappings for person status.
    """
    if not logs_to_add and not person_updates:
        logger.debug(f"Batch Commit (Batch {batch_num}): No data to commit.")
        return True  # Nothing to do

    logger.debug(
        f"Attempting batch commit (Batch {batch_num}): {len(logs_to_add)} logs, {len(person_updates)} persons..."
    )

    try:
        with db_transn(db_session) as sess:
            # --- Upsert Logic for ConversationLog ---
            processed_logs_count = 0
            if logs_to_add:
                logger.debug(
                    f" Upserting {len(logs_to_add)} ConversationLog entries..."
                )
                for log_data in logs_to_add:
                    try:
                        conv_id = log_data.get("conversation_id")
                        direction_str = log_data.get(
                            "direction"
                        )  # Expecting string 'IN' or 'OUT' if created from enum .value, or enum object

                        # --- Convert direction string/enum to Enum object ---
                        direction_enum = None
                        if isinstance(direction_str, MessageDirectionEnum):
                            direction_enum = direction_str
                        elif isinstance(direction_str, str):
                            try:
                                direction_enum = MessageDirectionEnum(
                                    direction_str.upper()
                                )
                            except ValueError:
                                logger.error(
                                    f"Invalid direction string '{direction_str}' in log data for ConvID {conv_id}. Skipping."
                                )
                                continue
                        else:
                            logger.error(
                                f"Invalid direction type ({type(direction_str)}) in log data for ConvID {conv_id}. Skipping."
                            )
                            continue
                        # --- End Direction Handling ---

                        if not conv_id:
                            logger.error(
                                f"Missing conversation_id in log data. Skipping: {log_data}"
                            )
                            continue

                        # Query for existing record
                        existing_log = (
                            sess.query(ConversationLog)
                            .filter_by(
                                conversation_id=conv_id, direction=direction_enum
                            )
                            .first()
                        )

                        # Prepare update data dictionary (excluding PKs)
                        # Ensure timestamp is timezone-aware datetime
                        ts_val = log_data.get("latest_timestamp")
                        aware_timestamp = None
                        if isinstance(ts_val, datetime):
                            aware_timestamp = (
                                ts_val.astimezone(timezone.utc)
                                if ts_val.tzinfo
                                else ts_val.replace(tzinfo=timezone.utc)
                            )
                        else:
                            logger.error(
                                f"Invalid timestamp type ({type(ts_val)}) for ConvID {conv_id}/{direction_enum}. Skipping."
                            )
                            continue  # Need a valid timestamp

                        update_values = {
                            k: v
                            for k, v in log_data.items()
                            if k
                            not in [
                                "conversation_id",
                                "direction",
                                "created_at",
                                "updated_at",
                            ]  # Exclude PKs and auto-timestamps
                        }
                        update_values["latest_timestamp"] = (
                            aware_timestamp  # Use the aware timestamp
                        )
                        update_values["updated_at"] = datetime.now(
                            timezone.utc
                        )  # Always update 'updated_at'

                        if existing_log:
                            # Update existing log
                            logger.debug(
                                f"  Updating existing log for {conv_id}/{direction_enum.name}"
                            )
                            for key, value in update_values.items():
                                setattr(existing_log, key, value)
                            # No need to sess.add() for updates if object fetched from session
                        else:
                            # Insert new log
                            logger.debug(
                                f"  Inserting new log for {conv_id}/{direction_enum.name}"
                            )
                            new_log_data = update_values.copy()
                            new_log_data["conversation_id"] = conv_id
                            new_log_data["direction"] = direction_enum
                            # Ensure people_id is present
                            if (
                                "people_id" not in new_log_data
                                or not new_log_data["people_id"]
                            ):
                                logger.error(
                                    f"Missing 'people_id' for new log {conv_id}/{direction_enum.name}. Skipping."
                                )
                                continue
                            new_log_obj = ConversationLog(**new_log_data)
                            sess.add(new_log_obj)  # Add the new object

                        processed_logs_count += 1

                    except Exception as inner_log_exc:
                        logger.error(
                            f" Error processing single log item (ConvID: {log_data.get('conversation_id')}, Dir: {log_data.get('direction')}): {inner_log_exc}",
                            exc_info=True,
                        )
                        # Decide if one failure should fail the batch? For now, continue processing others.

                logger.debug(
                    f" Finished processing {processed_logs_count} log entries for upsert."
                )
            # --- End Upsert Logic ---

            # Prepare Person status updates (remains the same)
            if person_updates:
                update_mappings = [
                    {
                        "id": pid,
                        "status": status_enum,
                        "updated_at": datetime.now(timezone.utc),
                    }
                    for pid, status_enum in person_updates.items()
                ]
                if update_mappings:
                    logger.debug(f" Updating {len(update_mappings)} Person statuses...")
                    sess.bulk_update_mappings(Person, update_mappings)

            # Log session state before commit attempt (optional)
            # logger.debug(f"  Session state before commit (Batch {batch_num}): Dirty={len(sess.dirty)}, New={len(sess.new)}")

        logger.debug(f"Batch commit successful (Batch {batch_num}).")
        return True

    except IntegrityError as ie:  # Now this except block will work
        logger.error(
            f"DB UNIQUE constraint error during Action 9 batch commit (Batch {batch_num}): {ie}",
            exc_info=False,  # Less verbose for constraint errors
        )
        return False
    except Exception as e:
        logger.error(
            f"Error committing Action 9 batch (Batch {batch_num}): {e}", exc_info=True
        )
        return False
# End of _commit_action9_batch

#####################################################
# Main Function: process_productive_messages
#####################################################

def process_productive_messages(session_manager: SessionManager) -> bool:
    """
    V0.3: Implements batch database commits.
    Processes messages marked as PRODUCTIVE by Action 7.
    Extracts info, creates tasks (honoring dry_run), sends ack, archives person.
    Applies MAX_PRODUCTIVE_TO_PROCESS limit if set.
    """
    if not session_manager or not session_manager.my_profile_id:
        logger.error("Action 9 requires SessionManager with profile ID.")
        return False
    if not session_manager.driver_live:
        logger.error("Action 9 requires a live WebDriver session.")
        return False
    logger.info("--- Starting Action 9: Process Productive Messages ---")
    my_pid_lower = session_manager.my_profile_id.lower()
    overall_success = True
    processed_count = 0
    tasks_created_count = 0
    acks_sent_count = 0
    archived_count = 0
    error_count = 0
    skipped_count = 0
    total_candidates = 0
    ms_graph_token: Optional[str] = None
    ms_list_id: Optional[str] = None
    ms_list_name = config_instance.MS_TODO_LIST_NAME
    ms_auth_attempted = False
    batch_num = 0
    critical_db_error_occurred = False  # Flag to stop processing if commit fails
    # --- Batching Setup ---
    logs_to_add: List[Dict[str, Any]] = []
    person_updates: Dict[int, PersonStatusEnum] = {}
    batch_size = config_instance.BATCH_SIZE
    # Ensure batch size is positive for triggering logic
    commit_threshold = batch_size if batch_size > 0 else float("inf")
    logger.debug(f"Action 9 Batch Commit Threshold: {commit_threshold}")
    # --- End Batching Setup ---
    limit = config_instance.MAX_PRODUCTIVE_TO_PROCESS
    message_templates = _load_templates_for_action9()
    if not message_templates or ACKNOWLEDGEMENT_MESSAGE_TYPE not in message_templates:
        logger.error("Failed to load required message templates. Aborting.")
        return False
    ack_template = message_templates[ACKNOWLEDGEMENT_MESSAGE_TYPE]
    try:
        # Get DB session *outside* the loop for batching
        db_session = session_manager.get_db_conn()
        if not db_session:
            logger.error("Failed to get DB session for Action 9.")
            return False
        # Get MessageType ID for acknowledgement (once)
        ack_msg_type_obj = (
            db_session.query(MessageType)
            .filter(MessageType.type_name == ACKNOWLEDGEMENT_MESSAGE_TYPE)
            .first()
        )
        if not ack_msg_type_obj:
            logger.error(
                f"MessageType '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' not found in database. Seed DB?"
            )
            session_manager.return_session(db_session)  # Return session before exiting
            return False
        ack_msg_type_id = ack_msg_type_obj.id
        # --- Query Candidates ---
        logger.debug("Querying for candidate Persons (Status ACTIVE)...")
        # <<< --- Candidate Query Logic remains unchanged --- >>>
        latest_in_log_subq = (
            db_session.query(
                ConversationLog.people_id,
                func.max(ConversationLog.latest_timestamp).label("max_in_ts"),
            )
            .filter(ConversationLog.direction == MessageDirectionEnum.IN)
            .group_by(ConversationLog.people_id)
            .subquery("latest_in_sub")
        )
        latest_out_log_subq = (
            db_session.query(
                ConversationLog.people_id,
                func.max(ConversationLog.latest_timestamp).label("max_out_ts"),
            )
            .filter(ConversationLog.direction == MessageDirectionEnum.OUT)
            .group_by(ConversationLog.people_id)
            .subquery("latest_out_sub")
        )
        candidates_query = (
            db_session.query(Person)
            .options(
                joinedload(Person.conversation_log_entries).options(
                    joinedload(ConversationLog.message_type)
                )
            )
            .outerjoin(latest_in_log_subq, Person.id == latest_in_log_subq.c.people_id)
            .outerjoin(
                latest_out_log_subq, Person.id == latest_out_log_subq.c.people_id
            )
            .join(
                ConversationLog,
                and_(
                    Person.id == ConversationLog.people_id,
                    ConversationLog.direction == MessageDirectionEnum.IN,
                    ConversationLog.latest_timestamp == latest_in_log_subq.c.max_in_ts,
                ),
            )
            .filter(Person.status == PersonStatusEnum.ACTIVE)
            .filter(ConversationLog.ai_sentiment == PRODUCTIVE_SENTIMENT)
            .order_by(Person.id)  # Consistent order
        )
        if limit > 0:
            candidates_query = candidates_query.limit(limit)
            logger.info(
                f"Processing max {limit} productive candidates due to MAX_PRODUCTIVE_TO_PROCESS setting."
            )
        candidates = candidates_query.all()
        total_candidates = len(candidates)
        if not candidates:
            logger.info(
                "No ACTIVE persons found with latest IN message marked as PRODUCTIVE."
            )
            session_manager.return_session(db_session)  # Return session
            return True
        logger.info(
            f"Found {total_candidates} candidates with productive messages to process."
        )
        # --- Processing Loop ---
        tqdm_args = {
            "total": total_candidates,
            "desc": "Processing Productive",
            "unit": " person",
            "ncols": 100,
            "leave": True,
            "bar_format": "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
        }
        with logging_redirect_tqdm(), tqdm(**tqdm_args) as progress_bar:
            for person in candidates:
                processed_count += 1
                log_prefix = f"{person.username} #{person.id}"
                person_success = True

                # --- Check if critical DB error occurred in previous batch ---
                if critical_db_error_occurred:
                    logger.warning(
                        f"Skipping remaining candidates due to previous DB commit error."
                    )
                    error_count += (
                        total_candidates - processed_count + 1
                    )  # Mark remaining as error
                    break  # Stop processing loop
                try:
                    # --- Per-person logic remains largely the same until DB update ---
                    wait_time = session_manager.dynamic_rate_limiter.wait()
                    # ... (find latest logs, skip checks, AI call, task creation remain the same) ...
                    # <<< --- Logic finding latest_in_log, latest_out_log --- >>>
                    latest_in_log: Optional[ConversationLog] = None
                    latest_out_log: Optional[ConversationLog] = None
                    latest_in_ts = datetime.min.replace(tzinfo=timezone.utc)
                    latest_out_ts = datetime.min.replace(tzinfo=timezone.utc)
                    for log in person.conversation_log_entries:
                        log_ts = log.latest_timestamp
                        if log_ts and log_ts.tzinfo is None:
                            log_ts = log_ts.replace(tzinfo=timezone.utc)
                        if log.direction == MessageDirectionEnum.IN:
                            if log_ts and log_ts > latest_in_ts:
                                latest_in_ts, latest_in_log = log_ts, log
                        elif log.direction == MessageDirectionEnum.OUT:
                            if log_ts and log_ts > latest_out_ts:
                                latest_out_ts, latest_out_log = log_ts, log
                    # <<< --- Sanity checks / Skip conditions --- >>>
                    if (
                        not latest_in_log
                        or latest_in_log.ai_sentiment != PRODUCTIVE_SENTIMENT
                    ):
                        logger.warning(
                            f"Skipping {log_prefix}: Latest IN log not found or not PRODUCTIVE."
                        )
                        skipped_count += 1
                        continue
                    ack_already_sent = False
                    if (
                        latest_out_log
                        and latest_out_log.message_type_id == ack_msg_type_id
                        and latest_out_log.latest_timestamp
                        > latest_in_log.latest_timestamp
                    ):
                        ack_already_sent = True
                    if ack_already_sent:
                        logger.debug(
                            f"Skipping {log_prefix}: Acknowledgement already sent."
                        )
                        if (
                            person.status == PersonStatusEnum.ACTIVE
                        ):  # Still archive if missed before
                            logger.warning(
                                f"{log_prefix}: ACK sent but status ACTIVE. Staging archival."
                            )
                            person_updates[person.id] = PersonStatusEnum.ARCHIVE
                            archived_count += 1  # Count staging
                        skipped_count += 1
                        continue
                    # <<< --- Get context, call AI --- >>>
                    context_logs = _get_message_context(db_session, person.id)
                    if not context_logs:
                        logger.warning(f"Skipping {log_prefix}: Failed context.")
                        skipped_count += 1
                        continue
                    formatted_context = _format_context_for_ai_extraction(
                        context_logs, my_pid_lower
                    )
                    ai_response = _call_ai_for_extraction(
                        formatted_context, session_manager
                    )
                    extracted_data, suggested_tasks, summary_for_ack = (
                        {},
                        [],
                        "your message",
                    )
                    if ai_response:
                        extracted_data = ai_response.get("extracted_data", {})
                        suggested_tasks = ai_response.get("suggested_tasks", [])
                        # ... (build summary_for_ack) ...
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
                        logger.warning(f"{log_prefix}: AI extraction failed.")
                    # <<< --- MS Graph Tasks (respecting dry run) --- >>>
                    if suggested_tasks:
                        if not ms_graph_token and not ms_auth_attempted:
                            logger.info("Attempting MS Graph auth...")
                            try:
                                ms_graph_app = msal.PublicClientApplication(
                                    config_instance.MS_GRAPH_CLIENT_ID,
                                    authority=ms_graph_utils.AUTHORITY,
                                )
                                ms_graph_token = (
                                    ms_graph_utils.acquire_token_device_flow(
                                        ms_graph_app
                                    )
                                )
                                ms_auth_attempted = True
                                if not ms_graph_token:
                                    logger.error("MS Graph auth failed.")
                            except Exception as auth_err:
                                logger.error(f"MS Graph auth error: {auth_err}")
                                ms_auth_attempted = True
                        if ms_graph_token and not ms_list_id:
                            logger.info(f"Looking up MS List ID '{ms_list_name}'...")
                            ms_list_id = ms_graph_utils.get_todo_list_id(
                                ms_graph_token, ms_list_name
                            )
                            if not ms_list_id:
                                logger.error(f"Failed find MS List '{ms_list_name}'.")
                        if ms_graph_token and ms_list_id:
                            if config_instance.APP_MODE == "dry_run":
                                logger.info(
                                    f"{log_prefix}: DRY RUN - Skipping MS tasks."
                                )
                            else:
                                logger.debug(
                                    f"{log_prefix}: Creating {len(suggested_tasks)} MS tasks..."
                                )
                                for task_desc in suggested_tasks:
                                    task_title = f"Ancestry Follow-up: {person.username} (#{person.id})"
                                    task_body = f"AI Suggested Task based on message from {person.username} (#{person.id}):\n\n{task_desc}\n\nRelated Profile ID: {person.profile_id}\nConversation ID: {latest_in_log.conversation_id}"
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
                                            f"{log_prefix}: Failed create task: '{task_desc}'"
                                        )
                        elif suggested_tasks:
                            logger.warning(
                                f"{log_prefix}: Skipping task creation (MS Auth/List ID missing)."
                            )
                    # <<< --- Format ACK message --- >>>
                    try:
                        name_to_use = format_name(person.first_name or person.username)
                        message_text = ack_template.format(
                            name=name_to_use, summary=summary_for_ack
                        )
                    except Exception as fmt_e:
                        logger.error(
                            f"{log_prefix}: ACK formatting error: {fmt_e}. Using generic."
                        )
                        message_text = f"Dear {name_to_use},\n\nThank you!\n\nWayne"

                    # <<< --- Send ACK message --- >>>
                    logger.info(f"Processing {log_prefix}: Sending acknowledgement...")
                    conv_id_to_use = latest_in_log.conversation_id
                    send_status, _ = _send_message_via_api(
                        session_manager,
                        person,
                        message_text,
                        conv_id_to_use,
                        log_prefix,
                    )
                    # --- Prepare Database Updates (NO COMMIT HERE) ---
                    if send_status in ("delivered OK", "typed (dry_run)"):
                        logger.info(
                            f"{log_prefix}: Staging DB updates after ACK ({send_status})."
                        )
                        acks_sent_count += 1  # Count successful sends/dry-runs

                        # Prepare Log data
                        log_data = {
                            "conversation_id": conv_id_to_use,
                            "direction": MessageDirectionEnum.OUT,  # Store Enum directly ok? Check helper. Let's store string.
                            "people_id": person.id,
                            "latest_message_content": message_text[
                                : config_instance.MESSAGE_TRUNCATION_LENGTH
                            ],
                            "latest_timestamp": datetime.now(timezone.utc),
                            "message_type_id": ack_msg_type_id,
                            "script_message_status": send_status,
                            "updated_at": datetime.now(timezone.utc),
                            # ai_sentiment is None for OUT
                        }
                        logs_to_add.append(log_data)

                        # Prepare Person update
                        person_updates[person.id] = PersonStatusEnum.ARCHIVE
                        archived_count += 1  # Count successful staging
                    else:
                        logger.error(
                            f"{log_prefix}: Failed send ACK (Status: {send_status}). No DB changes staged."
                        )
                        error_count += 1
                        person_success = False  # Mark person as errored

                    # --- Trigger Batch Commit if Threshold Reached ---
                    current_batch_size = len(logs_to_add) + len(person_updates)
                    if current_batch_size >= commit_threshold:
                        batch_num += 1
                        logger.info(
                            f"Commit threshold {commit_threshold} reached. Committing Batch {batch_num}..."
                        )
                        commit_ok = _commit_action9_batch(
                            db_session, logs_to_add, person_updates, batch_num
                        )
                        if commit_ok:
                            logs_to_add.clear()
                            person_updates.clear()
                        else:
                            logger.critical(
                                f"CRITICAL: Batch commit {batch_num} FAILED. Stopping further processing."
                            )
                            critical_db_error_occurred = True
                            overall_success = False
                            # Don't reset counters, let summary reflect staged counts before failure
                            break  # Exit the main processing loop

                except Exception as person_proc_err:
                    logger.error(
                        f"CRITICAL error processing {log_prefix}: {person_proc_err}",
                        exc_info=True,
                    )
                    error_count += 1
                    person_success = False

                finally:
                    if not person_success:
                        overall_success = False
                    progress_bar.update(1)
            # --- End Loop ---
        # --- Final Commit after Loop ---
        if not critical_db_error_occurred and (logs_to_add or person_updates):
            batch_num += 1
            logger.info(f"Committing final batch (Batch {batch_num})...")
            final_commit_ok = _commit_action9_batch(
                db_session, logs_to_add, person_updates, batch_num
            )
            if not final_commit_ok:
                logger.error("Final batch commit FAILED.")
                overall_success = False  # Mark overall failure if final commit fails
            else:
                logs_to_add.clear()  # Clear on final success too
                person_updates.clear()

    except Exception as outer_e:
        logger.critical(
            f"Unhandled exception in process_productive_messages: {outer_e}",
            exc_info=True,
        )
        overall_success = False
    finally:
        # --- Final Summary ---
        # Return the session if it was obtained successfully
        if "db_session" in locals() and db_session is not None:
            session_manager.return_session(db_session)
        print(" ")  # Newline after progress bar
        logger.info("--- Action 9 Summary ----")
        # Adjust summary based on potential early exit due to DB error
        final_processed = processed_count
        final_errors = error_count
        if critical_db_error_occurred:
            logger.warning(
                f"Summary reflects state *before* processing stopped due to DB error."
            )
            # Optionally adjust error count: final_errors += (total_candidates - processed_count)

        logger.info(f"  Candidates Queried (w/ limit): {total_candidates}")
        logger.info(f"  Candidates Processed:       {final_processed}")
        logger.info(f"  Skipped (Conditions Met):   {skipped_count}")
        logger.info(f"  MS To-Do Tasks Created:     {tasks_created_count}")
        logger.info(f"  Acks Sent/Staged:         {acks_sent_count}")  # Renamed Staged
        logger.info(f"  Persons Archived/Staged:    {archived_count}")  # Renamed Staged
        logger.info(f"  Errors during processing:   {final_errors}")
        logger.info(f"  Overall Success:            {overall_success}")
        logger.info("--------------------------\n")
    return overall_success
# End of process_productive_messages


# end of  action9_process_productive.py
