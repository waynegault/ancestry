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
from database import (
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


def _search_gedcom_for_names(names: List[str]):
    """Placeholder: Searches the configured GEDCOM file for names."""
    # TODO: Implement GEDCOM parsing and search logic.
    gedcom_path = config_instance.GEDCOM_FILE_PATH
    if gedcom_path and gedcom_path.exists():
        logger.info(
            f"(Placeholder) Would search GEDCOM {gedcom_path.name} for: {names}"
        )
        # Add actual search logic here
        pass
    elif gedcom_path:
        logger.warning(f"GEDCOM search configured but file not found: {gedcom_path}")
    else:
        logger.warning("GEDCOM search called but GEDCOM_FILE_PATH not configured.")
    return None  # Placeholder return


# End of _search_gedcom_for_names


def _search_api_for_names(session_manager: SessionManager, names: List[str]):
    """Placeholder: Searches Ancestry (potentially via API) for names."""
    # TODO: Implement Ancestry API search logic (likely undocumented).
    logger.info(f"(Placeholder) Would search Ancestry API for names: {names}")
    # Add actual API search logic here
    return None  # Placeholder return


# End of _search_api_for_names


def _search_ancestry_tree(session_manager: SessionManager, names: List[str]):
    """
    Placeholder dispatcher for searching the user's tree (GEDCOM or API)
    for names extracted by the AI, based on configuration.

    Args:
        session_manager: The SessionManager instance.
        names: A list of names extracted from the conversation.

    Returns:
        Currently None. Intended to return search results eventually.
    """
    # Step 1: Check if there are names to search for
    if not names:
        logger.debug("Action 9 Tree Search: No names extracted to search.")
        return None

    search_method = config_instance.TREE_SEARCH_METHOD
    logger.info(f"Action 9 Tree Search: Method configured as '{search_method}'.")

    # Step 2: Dispatch based on configured method
    if search_method == "GEDCOM":
        return _search_gedcom_for_names(names)
    elif search_method == "API":
        return _search_api_for_names(session_manager, names)
    elif search_method == "NONE":
        logger.info("Action 9 Tree Search: Method set to NONE. Skipping search.")
        return None
    else:  # Should be caught by config loading, but safety check
        logger.error(
            f"Action 9 Tree Search: Invalid TREE_SEARCH_METHOD '{search_method}' encountered."
        )
        return None


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


# ------------------------------------------------------------------------------
# Main Function: process_productive_messages
# ------------------------------------------------------------------------------


def process_productive_messages(session_manager: SessionManager) -> bool:
    """
    Main function for Action 9. Finds persons with recent 'PRODUCTIVE' messages,
    extracts info/tasks via AI, creates MS To-Do tasks, sends acknowledgements,
    and updates database status using the unified commit_bulk_data function.
    Includes improved summary generation for ACKs and
    robust parsing of AI JSON response.

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
    processed_count = 0
    tasks_created_count = 0
    acks_sent_count = 0
    archived_count = 0  # Track how many are staged for archive
    error_count = 0
    skipped_count = 0
    total_candidates = 0
    ms_graph_token: Optional[str] = None
    ms_list_id: Optional[str] = None
    ms_list_name = config_instance.MS_TODO_LIST_NAME
    ms_auth_attempted = False  # Track if we tried auth this run
    batch_num = 0
    critical_db_error_occurred = False
    logs_to_add_dicts: List[Dict[str, Any]] = []  # Store log data as dicts
    person_updates: Dict[int, PersonStatusEnum] = {}  # Store {person_id: status_enum}
    batch_size = max(1, config_instance.BATCH_SIZE)
    commit_threshold = batch_size
    # Limit number of productive messages processed in one run (0 = unlimited)
    limit = config_instance.MAX_PRODUCTIVE_TO_PROCESS

    # Step 2: Load and Validate Templates
    message_templates = _load_templates_for_action9()
    if not message_templates:
        logger.error("Action 9: Required message templates failed to load. Aborting.")
        return False
    ack_template = message_templates[ACKNOWLEDGEMENT_MESSAGE_TYPE]

    # Step 3: Get DB Session and Required MessageType ID
    db_session: Optional[DbSession] = None
    try:
        db_session = session_manager.get_db_conn()
        if not db_session:
            logger.critical("Action 9: Failed get DB session. Aborting.")
            return False  # Critical failure if DB unavailable

        ack_msg_type_obj = (
            db_session.query(MessageType.id)
            .filter(MessageType.type_name == ACKNOWLEDGEMENT_MESSAGE_TYPE)
            .scalar()
        )
        if not ack_msg_type_obj:
            logger.critical(
                f"Action 9: MessageType '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' not found in DB. Aborting."
            )
            if db_session:
                session_manager.return_session(db_session)  # Release session
            return False  # Cannot proceed without the ACK type ID
        ack_msg_type_id = ack_msg_type_obj

        # --- Step 4: Query Candidate Persons ---
        logger.debug(
            "Querying for candidate Persons (Status ACTIVE, Sentiment PRODUCTIVE)..."
        )
        # Subquery to find the timestamp of the latest IN message for each person
        latest_in_log_subq = (
            db_session.query(
                ConversationLog.people_id,
                func.max(ConversationLog.latest_timestamp).label("max_in_ts"),
            )
            .filter(ConversationLog.direction == MessageDirectionEnum.IN)
            .group_by(ConversationLog.people_id)
            .subquery("latest_in_sub")
        )
        # Subquery to find the timestamp of the latest OUT *Acknowledgement* message for each person
        latest_ack_out_log_subq = (
            db_session.query(
                ConversationLog.people_id,
                func.max(ConversationLog.latest_timestamp).label("max_ack_out_ts"),
            )
            .filter(
                ConversationLog.direction == MessageDirectionEnum.OUT,
                ConversationLog.message_type_id
                == ack_msg_type_id,  # Check specific ACK type
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
            .join(latest_in_log_subq, Person.id == latest_in_log_subq.c.people_id)
            .join(  # Join to the specific IN log entry that is PRODUCTIVE and latest
                ConversationLog,
                and_(
                    Person.id == ConversationLog.people_id,
                    ConversationLog.direction == MessageDirectionEnum.IN,
                    ConversationLog.latest_timestamp == latest_in_log_subq.c.max_in_ts,
                    ConversationLog.ai_sentiment == PRODUCTIVE_SENTIMENT,
                ),
            )
            # Left join to find the latest ACK timestamp (if any)
            .outerjoin(
                latest_ack_out_log_subq,
                Person.id == latest_ack_out_log_subq.c.people_id,
            )
            # Filter conditions:
            .filter(
                # Must be currently ACTIVE
                Person.status == PersonStatusEnum.ACTIVE,
                # EITHER no ACK has ever been sent OR the latest PRODUCTIVE IN message
                # is NEWER than the latest ACK sent for this person.
                (latest_ack_out_log_subq.c.max_ack_out_ts == None)  # No ACK sent
                | (  # Or, latest IN is newer than latest ACK
                    latest_ack_out_log_subq.c.max_ack_out_ts
                    < latest_in_log_subq.c.max_in_ts
                ),
            )
            .order_by(Person.id)  # Consistent processing order
        )

        # Apply limit if configured
        if limit > 0:
            candidates_query = candidates_query.limit(limit)
            logger.debug(
                f"Action 9 Processing limited to {limit} productive candidates..."
            )

        # Fetch candidates
        candidates: List[Person] = candidates_query.all()
        total_candidates = len(candidates)

        if not candidates:
            logger.info(
                "Action 9: No eligible ACTIVE persons found with unprocessed PRODUCTIVE messages."
            )
            if db_session:
                session_manager.return_session(db_session)  # Release session
            return True  # Successful run, nothing to do
        logger.info(
            f"Action 9: Found {total_candidates} candidates with productive messages to process."
        )

        # --- Step 5: Processing Loop ---
        # Setup progress bar
        tqdm_args = {
            "total": total_candidates,
            "desc": "Processing Productive ",  # Add space after desc for alignment
            "unit": " person",
            "ncols": 100,
            "leave": True,
            # Use a format that includes postfix for dynamic stats
            "bar_format": "{desc}|{bar}| {n_fmt}/{total_fmt} {postfix}",
        }
        logger.info("Processing candidates...")
        with logging_redirect_tqdm(), tqdm(
            **tqdm_args,
            postfix={"t": 0, "a": 0, "s": 0, "e": 0},  # Tasks, Acks, Skip, Error
        ) as progress_bar:
            for person in candidates:
                processed_count += 1
                log_prefix = f"Productive: {person.username} #{person.id}"
                person_success = True  # Assume success for this person initially
                if critical_db_error_occurred:
                    # Update bar for remaining skipped items due to critical error
                    remaining_to_skip = total_candidates - processed_count + 1
                    skipped_count += remaining_to_skip
                    if progress_bar:
                        progress_bar.update(remaining_to_skip)
                        progress_bar.set_postfix(
                            t=tasks_created_count,
                            a=acks_sent_count,
                            s=skipped_count,
                            e=error_count,
                            refresh=True,
                        )
                    logger.warning(
                        f"Skipping remaining {remaining_to_skip} candidates due to previous DB commit error."
                    )
                    break  # Stop processing loop

                try:
                    # --- Step 5a: Rate Limit ---
                    wait_time = session_manager.dynamic_rate_limiter.wait()
                    # Optional log: if wait_time > 0.1: logger.debug(f"{log_prefix}: Rate limit wait: {wait_time:.2f}s")

                    # --- Step 5b: Get Message Context ---
                    logger.debug(f"{log_prefix}: Getting message context...")
                    context_logs = _get_message_context(db_session, person.id)
                    if not context_logs:
                        logger.warning(
                            f"Skipping {log_prefix}: Failed to retrieve message context."
                        )
                        skipped_count += 1
                        person_success = False
                        # Update postfix here before continuing
                        if progress_bar:
                            progress_bar.set_postfix(
                                t=tasks_created_count,
                                a=acks_sent_count,
                                s=skipped_count,
                                e=error_count,
                                refresh=False,
                            )  # No immediate refresh needed
                        continue  # Skip to next person

                    # --- Step 5c: Call AI for Extraction & Task Suggestion ---
                    formatted_context = _format_context_for_ai_extraction(
                        context_logs, my_pid_lower
                    )
                    logger.debug(
                        f"{log_prefix}: Calling AI for extraction/task suggestion..."
                    )
                    if not session_manager.is_sess_valid():
                        logger.error(
                            f"Session invalid before AI extraction call for {log_prefix}. Skipping person."
                        )
                        error_count += 1
                        person_success = False
                        if progress_bar:
                            progress_bar.set_postfix(
                                t=tasks_created_count,
                                a=acks_sent_count,
                                s=skipped_count,
                                e=error_count,
                                refresh=False,
                            )
                        continue
                    ai_response = extract_and_suggest_tasks(
                        formatted_context, session_manager
                    )

                    # --- Step 5d: Process AI Response (with Robust Parsing) ---
                    extracted_data: Dict[str, List[str]] = {
                        "mentioned_names": [],
                        "mentioned_locations": [],
                        "mentioned_dates": [],
                        "potential_relationships": [],
                        "key_facts": [],
                    }
                    suggested_tasks: List[str] = []
                    summary_for_ack = "your message"  # Default summary

                    # --- Robust AI JSON Parsing ---
                    if ai_response and isinstance(ai_response, dict):
                        logger.debug(
                            f"{log_prefix}: AI response received. Parsing structure..."
                        )
                        extracted_data_raw = ai_response.get("extracted_data")
                        suggested_tasks_raw = ai_response.get("suggested_tasks")

                        # Parse extracted_data substructure carefully
                        if isinstance(extracted_data_raw, dict):
                            expected_keys = [
                                "mentioned_names",
                                "mentioned_locations",
                                "mentioned_dates",
                                "potential_relationships",
                                "key_facts",
                            ]
                            for key in expected_keys:
                                value = extracted_data_raw.get(key)
                                if value is not None and isinstance(value, list):
                                    # Filter for strings within the list
                                    extracted_data[key] = [
                                        str(item)
                                        for item in value
                                        if isinstance(item, (str, int, float))
                                    ]
                                    if len(extracted_data[key]) != len(value):
                                        logger.warning(
                                            f"{log_prefix}: Filtered non-string items from 'extracted_data.{key}'."
                                        )
                                else:
                                    logger.warning(
                                        f"{log_prefix}: AI JSON 'extracted_data' key '{key}' missing or not a list. Using empty list."
                                    )
                                    extracted_data[key] = (
                                        []
                                    )  # Ensure key exists with empty list
                        else:
                            logger.warning(
                                f"{log_prefix}: AI response 'extracted_data' missing or not a dict. Using default empty data."
                            )

                        # Parse suggested_tasks carefully
                        if suggested_tasks_raw is not None and isinstance(
                            suggested_tasks_raw, list
                        ):
                            suggested_tasks = [
                                str(item)
                                for item in suggested_tasks_raw
                                if isinstance(item, str)
                            ]
                            if len(suggested_tasks) != len(suggested_tasks_raw):
                                logger.warning(
                                    f"{log_prefix}: Filtered non-string items from 'suggested_tasks'."
                                )
                            logger.debug(
                                f"{log_prefix}: Parsed 'suggested_tasks' ({len(suggested_tasks)} valid tasks)."
                            )
                        else:
                            logger.warning(
                                f"{log_prefix}: AI response 'suggested_tasks' missing or not a list. Using empty list."
                            )
                            suggested_tasks = []

                        # --- Generate Summary for ACK ---
                        summary_parts = []

                        def format_list_for_summary(
                            items: List[str], max_items: int = 3
                        ) -> str:
                            if not items:
                                return ""
                            display_items = items[:max_items]
                            more_count = len(items) - max_items
                            suffix = f" and {more_count} more" if more_count > 0 else ""
                            quoted_items = [f"'{item}'" for item in display_items]
                            return ", ".join(quoted_items) + suffix

                        names_str = format_list_for_summary(
                            extracted_data.get("mentioned_names", [])
                        )
                        locs_str = format_list_for_summary(
                            extracted_data.get("mentioned_locations", [])
                        )
                        dates_str = format_list_for_summary(
                            extracted_data.get("mentioned_dates", [])
                        )
                        facts_str = format_list_for_summary(
                            extracted_data.get("key_facts", []), max_items=2
                        )
                        if names_str:
                            summary_parts.append(f"the names {names_str}")
                        if locs_str:
                            summary_parts.append(f"locations like {locs_str}")
                        if dates_str:
                            summary_parts.append(f"dates including {dates_str}")
                        if facts_str:
                            summary_parts.append(f"details such as {facts_str}")

                        if summary_parts:
                            summary_for_ack = (
                                "the details about "
                                + ", ".join(summary_parts[:-1])
                                + (
                                    f" and {summary_parts[-1]}"
                                    if len(summary_parts) > 1
                                    else summary_parts[0]
                                )
                            )
                        else:
                            summary_for_ack = "the information you provided"  # Fallback if nothing specific extracted
                        logger.debug(
                            f"{log_prefix}: Generated ACK summary: '{summary_for_ack}'"
                        )
                    else:
                        logger.warning(
                            f"{log_prefix}: AI extraction response was None or invalid. Using default ACK summary."
                        )
                        summary_for_ack = (
                            "the information you provided"  # Use default on AI failure
                        )

                    # --- Step 5e: Optional Tree Search ---
                    tree_search_results = _search_ancestry_tree(
                        session_manager, extracted_data.get("mentioned_names", [])
                    )
                    # TODO: Process tree_search_results if needed

                    # --- Step 5f: MS Graph Task Creation ---
                    if suggested_tasks:
                        # MS Graph Auth Check (only try once per run)
                        if not ms_graph_token and not ms_auth_attempted:
                            logger.info(
                                "Attempting MS Graph authentication (device flow)..."
                            )
                            ms_graph_token = ms_graph_utils.acquire_token_device_flow()
                            ms_auth_attempted = True
                            if not ms_graph_token:
                                logger.error("MS Graph authentication failed.")

                        # MS Graph List ID Check (only try once per run if auth succeeded)
                        if ms_graph_token and not ms_list_id:
                            logger.info(
                                f"Looking up MS To-Do List ID for '{ms_list_name}'..."
                            )
                            ms_list_id = ms_graph_utils.get_todo_list_id(
                                ms_graph_token, ms_list_name
                            )
                            if not ms_list_id:
                                logger.error(
                                    f"Failed find/get MS List ID for '{ms_list_name}'."
                                )

                        # Create Tasks if possible
                        if ms_graph_token and ms_list_id:
                            app_mode_for_tasks = config_instance.APP_MODE
                            if app_mode_for_tasks == "dry_run":
                                logger.info(
                                    f"{log_prefix}: DRY RUN - Skipping MS To-Do task creation for {len(suggested_tasks)} suggested tasks."
                                )
                            else:
                                logger.info(
                                    f"{log_prefix}: Creating {len(suggested_tasks)} MS To-Do tasks (Mode: {app_mode_for_tasks})..."
                                )
                                for task_index, task_desc in enumerate(suggested_tasks):
                                    task_title = f"Ancestry Follow-up: {person.username or 'Unknown'} (#{person.id})"
                                    # Include conv ID in body for reference
                                    last_conv_id = (
                                        context_logs[-1].conversation_id
                                        if context_logs
                                        else "N/A"
                                    )
                                    task_body = f"AI Suggested Task ({task_index+1}/{len(suggested_tasks)}): {task_desc}\n\nMatch: {person.username or 'Unknown'} (#{person.id})\nProfile: {person.profile_id or 'N/A'}\nConvID: {last_conv_id}"
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
                        ):  # Log skip reason if prerequisites not met
                            logger.warning(
                                f"{log_prefix}: Skipping MS task creation ({len(suggested_tasks)} tasks) - MS Auth/List ID unavailable."
                            )

                    # --- Step 5g: Format Acknowledgement Message ---
                    try:
                        # Use first name if available, else username
                        name_to_use_ack = format_name(
                            person.first_name or person.username
                        )
                        message_text = ack_template.format(
                            name=name_to_use_ack, summary=summary_for_ack
                        )
                    except KeyError as ke:
                        logger.error(
                            f"{log_prefix}: ACK template formatting error (Key {ke}). Using generic fallback."
                        )
                        message_text = f"Dear {format_name(person.username)},\n\nThank you for your message and the information!\n\nWayne"  # Simple fallback
                    except Exception as fmt_e:
                        logger.error(
                            f"{log_prefix}: Unexpected ACK formatting error: {fmt_e}. Using generic fallback."
                        )
                        message_text = f"Dear {format_name(person.username)},\n\nThank you!\n\nWayne"  # Simpler fallback

                    # --- Step 5h: Apply Mode/Recipient Filtering ---
                    app_mode = config_instance.APP_MODE
                    testing_profile_id_config = config_instance.TESTING_PROFILE_ID
                    current_profile_id = person.profile_id or "UNKNOWN"
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
                    conv_id_for_send = (
                        context_logs[-1].conversation_id if context_logs else None
                    )  # Get conv ID from last log entry
                    if not conv_id_for_send:
                        logger.error(
                            f"{log_prefix}: Cannot find conversation ID to send ACK. Skipping send."
                        )
                        error_count += 1
                        person_success = False
                        # Update postfix before continuing
                        if progress_bar:
                            progress_bar.set_postfix(
                                t=tasks_created_count,
                                a=acks_sent_count,
                                s=skipped_count,
                                e=error_count,
                                refresh=False,
                            )
                        continue  # Skip to next person

                    if send_ack_flag:
                        logger.info(
                            f"{log_prefix}: Sending/Simulating '{ACKNOWLEDGEMENT_MESSAGE_TYPE}'..."
                        )
                        send_status, effective_conv_id = _send_message_via_api(
                            session_manager,
                            person,
                            message_text,
                            conv_id_for_send,  # Pass existing conv ID
                            log_prefix,
                        )
                    else:  # Skipped due to filter
                        send_status = skip_log_reason_ack
                        effective_conv_id = (
                            conv_id_for_send  # Use existing conv ID for logging
                        )

                    # --- Step 5j: Stage Database Updates ---
                    # Ensure effective_conv_id is not None before staging log
                    if not effective_conv_id:
                        logger.error(
                            f"{log_prefix}: effective_conv_id is None after send/skip. Cannot stage log entry."
                        )
                        if send_status not in (
                            "delivered OK",
                            "typed (dry_run)",
                        ) and not send_status.startswith("skipped ("):
                            error_count += (
                                1  # Count error only if send wasn't successful/skipped
                            )
                        person_success = False  # Mark person as failed
                    # Proceed only if effective_conv_id exists
                    elif send_status in (
                        "delivered OK",
                        "typed (dry_run)",
                    ) or send_status.startswith("skipped ("):
                        if send_ack_flag:
                            acks_sent_count += 1
                        logger.info(
                            f"{log_prefix}: Staging DB updates for ACK (Status: {send_status})."
                        )
                        # Prepare the dictionary for the log entry
                        log_data = {
                            "conversation_id": effective_conv_id,  # Use the confirmed/existing ID
                            "direction": MessageDirectionEnum.OUT,  # Pass Enum
                            "people_id": person.id,
                            "latest_message_content": (
                                f"[{send_status.upper()}] {message_text}"
                                if not send_ack_flag
                                else message_text  # Prepend skip reason if skipped
                            )[
                                : config_instance.MESSAGE_TRUNCATION_LENGTH
                            ],  # Truncate
                            "latest_timestamp": datetime.now(
                                timezone.utc
                            ),  # Use current UTC time
                            "message_type_id": ack_msg_type_id,
                            "script_message_status": send_status,  # Log outcome/skip reason
                            "ai_sentiment": None,  # N/A for OUT
                        }
                        logs_to_add_dicts.append(log_data)
                        # Stage person update to ARCHIVE
                        person_updates[person.id] = PersonStatusEnum.ARCHIVE
                        archived_count += 1
                        logger.debug(f"{log_prefix}: Person status staged for ARCHIVE.")
                    else:  # Send failed with specific error status
                        logger.error(
                            f"{log_prefix}: Failed to send ACK (Status: {send_status}). No DB changes staged for this person."
                        )
                        error_count += 1
                        person_success = False

                    # --- Step 5k: Trigger Batch Commit ---
                    if (
                        len(logs_to_add_dicts) + len(person_updates)
                    ) >= commit_threshold:
                        batch_num += 1
                        logger.info(
                            f"Commit threshold reached ({len(logs_to_add_dicts)} logs). Committing Action 9 Batch {batch_num}..."
                        )
                        try:
                            # --- CALL UNIFIED FUNCTION ---
                            logs_committed_count, persons_updated_count = (
                                commit_bulk_data(
                                    session=db_session,
                                    log_upserts=logs_to_add_dicts,
                                    person_updates=person_updates,
                                    context=f"Action 9 Batch {batch_num}",
                                )
                            )
                            # Commit successful (no exception)
                            logs_to_add_dicts.clear()
                            person_updates.clear()
                            logger.debug(
                                f"Action 9 Batch {batch_num} commit finished (Logs Processed: {logs_committed_count}, Persons Updated: {persons_updated_count})."
                            )
                        except Exception as commit_e:
                            logger.critical(
                                f"CRITICAL: Action 9 Batch commit {batch_num} FAILED: {commit_e}",
                                exc_info=True,
                            )
                            critical_db_error_occurred = True
                            overall_success = False
                            break  # Stop processing loop

                # --- Step 6: Handle errors during individual person processing ---
                except StopIteration as si:  # Catch clean exits if _process used it
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
                        overall_success = False
                    # Update postfix and bar
                    if progress_bar:
                        progress_bar.set_postfix(
                            t=tasks_created_count,
                            a=acks_sent_count,
                            s=skipped_count,
                            e=error_count,
                            refresh=False,
                        )
                        # Ensure update happens even if loop continues due to error
                        # progress_bar.update(1) # Update call already moved earlier
            # --- End Main Person Processing Loop ---

        # --- Step 8: Final Commit for any remaining data ---
        if not critical_db_error_occurred and (logs_to_add_dicts or person_updates):
            batch_num += 1
            logger.info(
                f"Committing final Action 9 batch (Batch {batch_num}) with {len(logs_to_add_dicts)} logs, {len(person_updates)} updates..."
            )
            try:
                # --- CALL UNIFIED FUNCTION ---
                final_logs_saved, final_persons_updated = commit_bulk_data(
                    session=db_session,
                    log_upserts=logs_to_add_dicts,
                    person_updates=person_updates,
                    context="Action 9 Final Save",
                )
                # Commit successful
                logs_to_add_dicts.clear()
                person_updates.clear()
                logger.debug(
                    f"Action 9 Final commit executed (Logs Processed: {final_logs_saved}, Persons Updated: {final_persons_updated})."
                )
            except Exception as final_commit_e:
                logger.error(
                    f"Final Action 9 batch commit FAILED: {final_commit_e}",
                    exc_info=True,
                )
                overall_success = False

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

        print(" ")  # Spacer before summary
        logger.info("------ Action 9: Process Productive Summary -------")
        final_processed = processed_count
        final_errors = error_count
        # Adjust counts if stopped early by critical DB error
        if critical_db_error_occurred and total_candidates > processed_count:
            unprocessed = total_candidates - processed_count
            logger.warning(
                f"Adding {unprocessed} unprocessed candidates to error count due to DB failure."
            )
            final_errors += unprocessed

        logger.info(f"  Candidates Queried:         {total_candidates}")
        logger.info(f"  Candidates Processed:       {final_processed}")
        logger.info(f"  Skipped (Context/Filter):   {skipped_count}")
        logger.info(f"  MS To-Do Tasks Created:     {tasks_created_count}")
        logger.info(f"  Acks Sent/Simulated:        {acks_sent_count}")
        logger.info(f"  Persons Archived (Staged):  {archived_count}")
        logger.info(f"  Errors during processing:   {final_errors}")
        logger.info(f"  Overall Success:            {overall_success}")
        logger.info("--------------------------------------------------\n")

    # Step 11: Return overall success status
    return overall_success


# End of process_productive_messages


# --- End of action9_process_productive.py ---
