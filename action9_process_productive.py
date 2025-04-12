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
    extract_and_suggest_tasks,
) 
import ms_graph_utils 
from cache import cache_result 

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
        context_logs = (
            db_session.query(ConversationLog)
            .filter(ConversationLog.people_id == person_id)
            .order_by(ConversationLog.latest_timestamp.desc())
            .limit(limit)
            .all()
        )
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
        author_label = "USER: "
        if log.direction == MessageDirectionEnum.OUT:
            author_label = "SCRIPT: "
        content = log.latest_message_content or ""
        words = content.split()
        if len(words) > max_words:
            truncated_content = " ".join(words[:max_words]) + "..."
        else:
            truncated_content = content
        context_lines.append(f"{author_label}{truncated_content}")
    return "\n".join(context_lines)
# End of _format_context_for_ai_extraction


def _search_ancestry_tree(session_manager: SessionManager, names: List[str]):
    """Placeholder for searching the user's tree for extracted names."""
    if names:
        logger.info(
            f"(Placeholder) Would search Ancestry tree for names: {', '.join(names)}"
        )
    else:
        logger.debug("(Placeholder) No names extracted to search in tree.")
# End of _search_ancestry_tree


@cache_result("action9_message_templates", ignore_args=True)
def _load_templates_for_action9() -> Dict[str, str]:
    """Loads templates, specifically ensuring the ACK template exists."""
    from action8_messaging import load_message_templates

    all_templates = load_message_templates()
    if ACKNOWLEDGEMENT_MESSAGE_TYPE not in all_templates:
        logger.critical(f"CRITICAL: Template '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' missing!")
        return {}
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
        return True

    logger.debug(
        f"Attempting batch commit (Batch {batch_num}): {len(logs_to_add)} logs, {len(person_updates)} persons..."
    )

    try:
        with db_transn(db_session) as sess:
            processed_logs_count = 0
            if logs_to_add:
                logger.debug(
                    f" Upserting {len(logs_to_add)} ConversationLog entries..."
                )
                for log_data in logs_to_add:
                    try:
                        conv_id = log_data.get("conversation_id")
                        direction_str = log_data.get("direction")
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
                                    f"Invalid direction str '{direction_str}' ConvID {conv_id}. Skip."
                                )
                                continue
                        else:
                            logger.error(
                                f"Invalid direction type ({type(direction_str)}) ConvID {conv_id}. Skip."
                            )
                            continue
                        if not conv_id:
                            logger.error(f"Missing conv_id. Skip: {log_data}")
                            continue

                        existing_log = (
                            sess.query(ConversationLog)
                            .filter_by(
                                conversation_id=conv_id, direction=direction_enum
                            )
                            .first()
                        )
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
                                f"Invalid timestamp type ({type(ts_val)}) ConvID {conv_id}/{direction_enum}. Skip."
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
                        }
                        update_values["latest_timestamp"] = aware_timestamp
                        update_values["updated_at"] = datetime.now(timezone.utc)

                        if existing_log:
                            logger.debug(
                                f"  Updating existing log for {conv_id}/{direction_enum.name}"
                            )
                            for key, value in update_values.items():
                                setattr(existing_log, key, value)
                        else:
                            logger.debug(
                                f"  Inserting new log for {conv_id}/{direction_enum.name}"
                            )
                            new_log_data = update_values.copy()
                            new_log_data["conversation_id"] = conv_id
                            new_log_data["direction"] = direction_enum
                            if (
                                "people_id" not in new_log_data
                                or not new_log_data["people_id"]
                            ):
                                logger.error(
                                    f"Missing 'people_id' new log {conv_id}/{direction_enum.name}. Skip."
                                )
                                continue
                            new_log_obj = ConversationLog(**new_log_data)
                            sess.add(new_log_obj)
                        processed_logs_count += 1
                    except Exception as inner_log_exc:
                        logger.error(
                            f" Error processing single log item (ConvID: {log_data.get('conversation_id')}, Dir: {log_data.get('direction')}): {inner_log_exc}",
                            exc_info=True,
                        )
                logger.debug(
                    f" Finished processing {processed_logs_count} log entries for upsert."
                )

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

        logger.debug(f"Batch commit successful (Batch {batch_num}).")
        return True
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

#####################################################
# Main Function: process_productive_messages
#####################################################


def process_productive_messages(session_manager: SessionManager) -> bool:
    """
    V0.7: Uses shared MSAL instance with persistent cache.
    Processes messages marked as PRODUCTIVE by Action 7.
    Extracts info, creates tasks (honoring dry_run), sends ack, archives person.
    Applies MAX_PRODUCTIVE_TO_PROCESS limit if set.
    Implements batch database commits.
    """
    # --- Basic validation of SessionManager and its state ---
    if not session_manager or not session_manager.my_profile_id:
        logger.error("Action 9 requires SessionManager with profile ID.")
        return False
    if not session_manager.driver_live:
        logger.error("Action 9 requires a live WebDriver session.")
        return False

    logger.info("--- Starting Action 9: Process Productive Messages ---")
    my_pid_lower = session_manager.my_profile_id.lower()

    # --- Initialize counters and state variables ---
    overall_success = True
    processed_count = 0
    tasks_created_count = 0
    acks_sent_count = 0
    archived_count = 0
    error_count = 0
    skipped_count = 0
    total_candidates = 0
    # MS Graph state (token/list_id are cached per run)
    ms_graph_token: Optional[str] = None
    ms_list_id: Optional[str] = None
    ms_list_name = config_instance.MS_TODO_LIST_NAME
    ms_auth_attempted = False  # Track if we've tried auth this run
    batch_num = 0
    critical_db_error_occurred = False

    # --- Setup for batch database commits ---
    logs_to_add: List[Dict[str, Any]] = []
    person_updates: Dict[int, PersonStatusEnum] = {}
    batch_size = config_instance.BATCH_SIZE
    # Ensure batch size is positive for triggering logic
    commit_threshold = batch_size if batch_size > 0 else float("inf")
    logger.debug(f"Action 9 Batch Commit Threshold: {commit_threshold}")

    # --- Get processing limit from config ---
    limit = config_instance.MAX_PRODUCTIVE_TO_PROCESS

    # --- Load and validate necessary message templates ---
    message_templates = _load_templates_for_action9()
    if not message_templates or ACKNOWLEDGEMENT_MESSAGE_TYPE not in message_templates:
        logger.error("Failed to load required message templates. Aborting.")
        return False
    ack_template = message_templates[ACKNOWLEDGEMENT_MESSAGE_TYPE]

    db_session = None  # Initialize db_session outside try/finally
    try:
        # --- Get a database session ---
        db_session = session_manager.get_db_conn()
        if not db_session:
            logger.error("Failed to get DB session for Action 9.")
            return False  # Cannot proceed without DB session

        # --- Get the MessageType ID for acknowledgements (required for logging) ---
        ack_msg_type_obj = (
            db_session.query(MessageType)
            .filter(MessageType.type_name == ACKNOWLEDGEMENT_MESSAGE_TYPE)
            .first()
        )
        if not ack_msg_type_obj:
            logger.error(
                f"MessageType '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' not found in database. Seed DB?"
            )
            # Ensure session is returned if we exit early
            if db_session:
                session_manager.return_session(db_session)
            return False
        ack_msg_type_id = ack_msg_type_obj.id

        # --- Query Database for Candidate Persons ---
        logger.debug("Querying for candidate Persons (Status ACTIVE)...")
        # Subquery to efficiently find the latest IN message timestamp per person
        latest_in_log_subq = (
            db_session.query(
                ConversationLog.people_id,
                func.max(ConversationLog.latest_timestamp).label("max_in_ts"),
            )
            .filter(ConversationLog.direction == MessageDirectionEnum.IN)
            .group_by(ConversationLog.people_id)
            .subquery("latest_in_sub")
        )
        # Subquery to efficiently find the latest OUT message timestamp per person
        latest_out_log_subq = (
            db_session.query(
                ConversationLog.people_id,
                func.max(ConversationLog.latest_timestamp).label("max_out_ts"),
            )
            .filter(ConversationLog.direction == MessageDirectionEnum.OUT)
            .group_by(ConversationLog.people_id)
            .subquery("latest_out_sub")
        )
        # Main query to find ACTIVE persons whose latest IN message is PRODUCTIVE
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
            .order_by(Person.id)
        )
        # Apply the processing limit if configured
        if limit > 0:
            candidates_query = candidates_query.limit(limit)
            logger.info(
                f"Processing max {limit} productive candidates due to MAX_PRODUCTIVE_TO_PROCESS setting."
            )

        # Execute the query
        candidates = candidates_query.all()
        total_candidates = len(candidates)  # Get count after limit is applied

        if not candidates:
            logger.info(
                "No ACTIVE persons found with latest IN message marked as PRODUCTIVE."
            )
            # Ensure session is returned if we exit early
            if db_session:
                session_manager.return_session(db_session)
            return True  # No work to do is considered success

        logger.info(
            f"Found {total_candidates} candidates with productive messages to process."
        )

        # --- Main Processing Loop ---
        tqdm_args = {
            "total": total_candidates,  # Use actual count after limit
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
                person_success = True  # Assume success for this person initially

                # Check if a critical DB error stopped processing in a previous iteration
                if critical_db_error_occurred:
                    logger.warning(
                        f"Skipping remaining candidates due to previous DB commit error."
                    )
                    error_count += total_candidates - processed_count + 1
                    break  # Stop processing loop

                try:
                    # Apply rate limiting before processing each person
                    wait_time = session_manager.dynamic_rate_limiter.wait()
                    if wait_time > 0.1:
                        logger.debug(f"{log_prefix}: Rate limit wait: {wait_time:.2f}s")

                    # Find the specific latest IN and OUT logs from the loaded collection
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

                    # --- Sanity Checks & Skip Conditions ---
                    if (
                        not latest_in_log
                        or latest_in_log.ai_sentiment != PRODUCTIVE_SENTIMENT
                    ):
                        logger.warning(
                            f"Skipping {log_prefix}: Latest IN log not found/productive."
                        )
                        skipped_count += 1
                        continue

                    # Check if an ACK was already sent *after* this productive message
                    ack_already_sent = False
                    if (
                        latest_out_log
                        and latest_out_log.message_type_id == ack_msg_type_id
                    ):
                        if (
                            latest_out_log.latest_timestamp
                            > latest_in_log.latest_timestamp
                        ):
                            ack_already_sent = True
                    if ack_already_sent:
                        logger.debug(f"Skipping {log_prefix}: ACK already sent.")
                        if person.status == PersonStatusEnum.ACTIVE:
                            logger.warning(
                                f"{log_prefix}: ACK sent but status ACTIVE. Staging archive."
                            )
                            person_updates[person.id] = PersonStatusEnum.ARCHIVE
                            archived_count += 1
                        skipped_count += 1
                        continue

                    # --- Get Message Context for AI ---
                    logger.debug(f"Processing {log_prefix}: Getting message context...")
                    context_logs = _get_message_context(db_session, person.id)
                    if not context_logs:
                        logger.warning(f"Skipping {log_prefix}: Failed context.")
                        skipped_count += 1
                        continue
                    formatted_context = _format_context_for_ai_extraction(
                        context_logs, my_pid_lower
                    )

                    # --- Call AI for Extraction & Tasks ---
                    logger.debug(
                        f"Processing {log_prefix}: Calling AI for extraction..."
                    )
                    ai_response = extract_and_suggest_tasks(
                        formatted_context, session_manager
                    )

                    # Process AI response
                    extracted_data, suggested_tasks, summary_for_ack = (
                        {},
                        [],
                        "your message",
                    )
                    if ai_response:
                        extracted_data = ai_response.get("extracted_data", {})
                        suggested_tasks = ai_response.get("suggested_tasks", [])
                        if not isinstance(extracted_data, dict):
                            extracted_data = {}
                        if not isinstance(suggested_tasks, list):
                            suggested_tasks = []
                        logger.info(
                            f"{log_prefix}: AI suggested {len(suggested_tasks)} tasks."
                        )
                        # Build summary
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
                        logger.warning(
                            f"{log_prefix}: AI extraction call failed or returned invalid data."
                        )

                    # --- MS Graph Task Creation (Honoring Dry Run) ---
                    if suggested_tasks:
                        # Authenticate Graph API if needed (uses shared persistent cache)
                        if not ms_graph_token and not ms_auth_attempted:
                            logger.info(
                                "Attempting MS Graph authentication (Device Flow)..."
                            )
                            try:
                                # Call function which uses shared MSAL instance
                                ms_graph_token = (
                                    ms_graph_utils.acquire_token_device_flow()
                                )
                                ms_auth_attempted = True  # Mark that we tried
                                if not ms_graph_token:
                                    logger.error(
                                        "MS Graph authentication failed. Cannot create To-Do tasks."
                                    )
                            except Exception as auth_err:
                                logger.error(
                                    f"Error during MS Graph auth: {auth_err}",
                                    exc_info=True,
                                )
                                ms_auth_attempted = True

                        # Get To-Do List ID if needed (cache within run)
                        if ms_graph_token and not ms_list_id:
                            logger.info(
                                f"Looking up MS To-Do List ID for '{ms_list_name}'..."
                            )
                            ms_list_id = ms_graph_utils.get_todo_list_id(
                                ms_graph_token, ms_list_name
                            )
                            if not ms_list_id:
                                logger.error(f"Failed find MS List '{ms_list_name}'.")

                        # Create tasks if possible AND NOT IN DRY RUN
                        if ms_graph_token and ms_list_id:
                            if config_instance.APP_MODE == "dry_run":
                                logger.info(
                                    f"{log_prefix}: DRY RUN - Skipping MS To-Do task creation ({len(suggested_tasks)} tasks)."
                                )
                            else:
                                logger.debug(
                                    f"{log_prefix}: Creating {len(suggested_tasks)} MS To-Do tasks..."
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
                                f"{log_prefix}: Skipping task creation (MS Graph Auth/List ID missing or failed)."
                            )

                    # --- (Optional) Ancestry Tree Search Placeholder ---
                    # _search_ancestry_tree(session_manager, extracted_data.get("mentioned_names", []))

                    # --- Format Acknowledgement Message ---
                    logger.debug(
                        f"Processing {log_prefix}: Preparing acknowledgement message..."
                    )
                    try:
                        name_to_use = format_name(person.first_name or person.username)
                        message_text = ack_template.format(
                            name=name_to_use, summary=summary_for_ack
                        )
                    except Exception as fmt_e:
                        logger.error(
                            f"{log_prefix}: ACK formatting error: {fmt_e}. Using generic."
                        )
                        message_text = f"Dear {name_to_use},\n\nThank you for your message!\n\nWarmest regards,\n\nWayne"

                    # --- Send Acknowledgement via API ---
                    logger.info(f"Processing {log_prefix}: Sending acknowledgement...")
                    conv_id_to_use = latest_in_log.conversation_id
                    send_status, _ = _send_message_via_api(
                        session_manager,
                        person,
                        message_text,
                        conv_id_to_use,
                        log_prefix,
                    )

                    # --- Stage Database Updates (Conditionally) ---
                    if send_status in ("delivered OK", "typed (dry_run)"):
                        logger.info(
                            f"{log_prefix}: Staging DB updates after ACK ({send_status})."
                        )
                        acks_sent_count += 1

                        # Prepare Log data dict for batching
                        log_data = {
                            "conversation_id": conv_id_to_use,
                            "direction": MessageDirectionEnum.OUT.value,  # Store string value
                            "people_id": person.id,
                            "latest_message_content": message_text[
                                : config_instance.MESSAGE_TRUNCATION_LENGTH
                            ],
                            "latest_timestamp": datetime.now(timezone.utc),
                            "message_type_id": ack_msg_type_id,
                            "script_message_status": send_status,
                            "updated_at": datetime.now(timezone.utc),
                            "ai_sentiment": None,
                        }
                        logs_to_add.append(log_data)

                        # Stage Person status update
                        person_updates[person.id] = PersonStatusEnum.ARCHIVE
                        archived_count += 1
                    else:
                        logger.error(
                            f"{log_prefix}: Failed send ACK (Status: {send_status}). No DB changes staged."
                        )
                        error_count += 1
                        person_success = False

                    # --- Trigger Batch Commit if Threshold Reached ---
                    # Check size of logs_to_add, as person_updates might contain duplicates if staging archive fails then succeeds
                    current_staged_log_count = len(logs_to_add)
                    if current_staged_log_count >= commit_threshold:
                        batch_num += 1
                        logger.info(
                            f"Commit threshold {commit_threshold} reached ({current_staged_log_count} logs). Committing Batch {batch_num}..."
                        )
                        commit_ok = _commit_action9_batch(
                            db_session, logs_to_add, person_updates, batch_num
                        )
                        if commit_ok:
                            logs_to_add.clear()
                            person_updates.clear()  # Clear both on success
                        else:
                            logger.critical(
                                f"CRITICAL: Batch commit {batch_num} FAILED. Stopping further processing."
                            )
                            critical_db_error_occurred = True
                            overall_success = False
                            break  # Exit the main processing loop

                # --- Catch errors during individual person processing ---
                except Exception as person_proc_err:
                    logger.error(
                        f"CRITICAL error processing {log_prefix}: {person_proc_err}",
                        exc_info=True,
                    )
                    error_count += 1
                    person_success = False

                # --- Update overall success flag and progress bar ---
                finally:
                    if not person_success:
                        overall_success = False
                    progress_bar.update(1)
            # --- End Main Person Processing Loop ---

        # --- Final Commit (after loop finishes or breaks) ---
        if not critical_db_error_occurred and (logs_to_add or person_updates):
            batch_num += 1
            logger.info(f"Committing final batch (Batch {batch_num})...")
            final_commit_ok = _commit_action9_batch(
                db_session, logs_to_add, person_updates, batch_num
            )
            if not final_commit_ok:
                logger.error("Final batch commit FAILED.")
                overall_success = False
            else:
                logs_to_add.clear()
                person_updates.clear()

    # --- Catch errors outside the main loop ---
    except Exception as outer_e:
        logger.critical(
            f"Unhandled exception in process_productive_messages: {outer_e}",
            exc_info=True,
        )
        overall_success = False
    # --- Final Cleanup and Summary ---
    finally:
        # Ensure the DB session is returned to the pool
        if db_session:
            session_manager.return_session(db_session)

        # Log final summary statistics
        print(" ")  # Newline after progress bar
        logger.info("--- Action 9 Summary ----")
        final_processed = processed_count
        final_errors = error_count
        if critical_db_error_occurred:
            logger.warning(
                f"Summary reflects state *before* processing stopped due to DB error."
            )
            final_errors += total_candidates - processed_count

        logger.info(f"  Candidates Queried (w/ limit): {total_candidates}")
        logger.info(f"  Candidates Processed:       {final_processed}")
        logger.info(f"  Skipped (Conditions Met):   {skipped_count}")
        logger.info(f"  MS To-Do Tasks Created:     {tasks_created_count}")
        logger.info(f"  Acks Sent/Staged:         {acks_sent_count}")
        logger.info(f"  Persons Archived/Staged:    {archived_count}")
        logger.info(f"  Errors during processing:   {final_errors}")
        logger.info(f"  Overall Success:            {overall_success}")
        logger.info("--------------------------\n")

    return overall_success
# End of process_productive_messages


# end of  action9_process_productive.py
 