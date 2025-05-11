#!/usr/bin/env python3

# auto_responder.py

"""
Automated response system for Ancestry messages.

This module integrates the capabilities of actions 7, 8, 9, 10, and 11 to:
1. Retrieve new messages from the inbox
2. Process messages to identify mentioned people
3. Search for mentioned people in GEDCOM and/or Ancestry API
4. Generate and send personalized responses with family details and relationships
"""

# --- Standard library imports ---
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# --- Third-party imports ---
from sqlalchemy import and_, func
from sqlalchemy.orm import Session as DbSession, joinedload
from sqlalchemy.exc import SQLAlchemyError
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# --- Local application imports ---
from config import config_instance
from database import (
    ConversationLog,
    MessageDirectionEnum,
    MessageType,
    Person,
    PersonStatusEnum,
    db_transn,
)
from logging_config import logger
from utils import SessionManager
from ai_interface import extract_and_suggest_tasks
from api_utils import call_send_message_api

# Import action-specific modules
from action9_process_productive import (
    _process_ai_response,
    _search_ancestry_tree as action9_search_ancestry_tree,
    _format_context_for_ai_extraction,
    _get_message_context,
)

# No need to import GEDCOM or API utilities directly
# We'll use action9's search function which already handles both

# --- Constants ---
PRODUCTIVE_SENTIMENT = "PRODUCTIVE"  # Sentiment string set by Action 7
CUSTOM_RESPONSE_MESSAGE_TYPE = "Automated_Genealogy_Response"  # Key in messages.json
DEFAULT_SEARCH_METHOD = "BOTH"  # "GEDCOM", "API", or "BOTH"
MAX_PEOPLE_TO_INCLUDE = 3  # Maximum number of people to include in a response
EXCLUSION_KEYWORDS = [
    "do not respond",
    "no reply",
    "unsubscribe",
]  # Keywords that indicate no response should be sent

# --- Helper Functions ---


def _search_for_people(
    session_manager: SessionManager, names: List[str]
) -> Dict[str, Any]:
    """
    Search for people in the ancestry tree.

    This is a simplified implementation that delegates to action9's _search_ancestry_tree function.
    In a future enhancement, this could be expanded to use both GEDCOM and API methods directly.

    Args:
        session_manager: The active SessionManager instance
        names: List of names to search for

    Returns:
        Dictionary with search results and relationship paths
    """
    # For now, we'll use action9's search function which already has the logic
    # to search for people and find their relationships
    try:
        logger.debug(f"Searching for people: {names}")

        # Initialize results
        combined_results = {"results": [], "relationship_paths": {}}

        # Search for each name
        for name in names:
            # We'll pass the name directly to the search function as a list

            # Call action9's search function
            search_results = action9_search_ancestry_tree(
                session_manager, [name]  # Pass as a list of names
            )

            # Merge results
            if search_results and "results" in search_results:
                combined_results["results"].extend(search_results.get("results", []))
            if search_results and "relationship_paths" in search_results:
                combined_results["relationship_paths"].update(
                    search_results.get("relationship_paths", {})
                )

        return combined_results
    except Exception as e:
        logger.error(f"Error searching for people: {e}", exc_info=True)
        return {"results": [], "relationship_paths": {}}


def should_exclude_message(message_content: Any) -> bool:
    """
    Check if a message should be excluded from automated responses.

    Args:
        message_content: The content of the message to check (str or SQLAlchemy Column)

    Returns:
        True if the message should be excluded, False otherwise
    """
    # Convert to string if it's not already
    content_str = str(message_content) if message_content is not None else ""

    if not content_str:
        return True

    # Check for exclusion keywords
    message_lower = content_str.lower()
    for keyword in EXCLUSION_KEYWORDS:
        if keyword.lower() in message_lower:
            logger.debug(f"Message contains exclusion keyword '{keyword}'. Skipping.")
            return True

    return False


def get_unprocessed_messages(db_session: DbSession) -> List[Dict[str, Any]]:
    """
    Retrieve unprocessed messages that require a response.

    Args:
        db_session: The active database session

    Returns:
        List of message dictionaries with conversation and person details
    """
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

    # Find the custom response message type ID
    custom_response_type = (
        db_session.query(MessageType.id)
        .filter(MessageType.type_name == CUSTOM_RESPONSE_MESSAGE_TYPE)
        .scalar()
    )

    # If custom response type doesn't exist, log warning but continue
    if not custom_response_type:
        logger.warning(f"MessageType '{CUSTOM_RESPONSE_MESSAGE_TYPE}' not found in DB.")

    # Subquery to find the timestamp of the latest custom response for each person
    latest_custom_response_subq = None
    if custom_response_type:
        latest_custom_response_subq = (
            db_session.query(
                ConversationLog.people_id,
                func.max(ConversationLog.latest_timestamp).label("max_custom_ts"),
            )
            .filter(
                ConversationLog.direction == MessageDirectionEnum.OUT,
                ConversationLog.message_type_id == custom_response_type,
            )
            .group_by(ConversationLog.people_id)
            .subquery("latest_custom_sub")
        )

    # Main query to find candidates
    candidates_query = (
        db_session.query(Person)
        .options(joinedload(Person.family_tree))
        .join(latest_in_log_subq, Person.id == latest_in_log_subq.c.people_id)
        .join(  # Join to the specific IN log entry that is the latest
            ConversationLog,
            and_(
                Person.id == ConversationLog.people_id,
                ConversationLog.direction == MessageDirectionEnum.IN,
                ConversationLog.latest_timestamp == latest_in_log_subq.c.max_in_ts,
            ),
        )
    )

    # Add filter for custom response if type exists
    if custom_response_type is not None and latest_custom_response_subq is not None:
        candidates_query = candidates_query.outerjoin(
            latest_custom_response_subq,
            Person.id == latest_custom_response_subq.c.people_id,
        ).filter(
            # EITHER no custom response has ever been sent OR the latest IN message
            # is NEWER than the latest custom response sent for this person.
            (
                latest_custom_response_subq.c.max_custom_ts == None
            )  # No custom response sent
            | (  # Or, latest IN is newer than latest custom response
                latest_custom_response_subq.c.max_custom_ts
                < latest_in_log_subq.c.max_in_ts
            ),
        )

    # Filter for ACTIVE status
    candidates_query = candidates_query.filter(
        # Must be currently ACTIVE
        Person.status == PersonStatusEnum.ACTIVE,
        # Not deleted
        Person.deleted_at == None,
    )

    # Order by timestamp (newest first)
    candidates_query = candidates_query.order_by(latest_in_log_subq.c.max_in_ts.desc())

    # Fetch candidates
    candidates = candidates_query.all()

    # Process candidates into a list of dictionaries
    result = []
    for person in candidates:
        # Get the latest IN message for this person
        latest_in_message = (
            db_session.query(ConversationLog)
            .filter(
                ConversationLog.people_id == person.id,
                ConversationLog.direction == MessageDirectionEnum.IN,
            )
            .order_by(ConversationLog.latest_timestamp.desc())
            .first()
        )

        if latest_in_message:
            # Check if message should be excluded
            if should_exclude_message(latest_in_message.latest_message_content):
                continue

            # Add to result list
            result.append(
                {
                    "person": person,
                    "conversation_id": latest_in_message.conversation_id,
                    "message_content": latest_in_message.latest_message_content,
                    "timestamp": latest_in_message.latest_timestamp,
                    "sentiment": latest_in_message.ai_sentiment,
                }
            )

    return result


# --- Main Function ---


def process_and_respond_to_messages(session_manager: SessionManager) -> bool:
    """
    Main function to process new messages and generate automated responses.

    1. Retrieves new messages from inbox
    2. Analyzes messages to extract mentioned people
    3. Searches for people in GEDCOM and/or API
    4. Generates and sends personalized responses
    5. Updates database with response information

    Args:
        session_manager: The active SessionManager instance

    Returns:
        True if processing completed successfully, False otherwise
    """
    # --- Step 1: Initialization ---
    logger.info("--- Starting Auto-Responder: Process and Respond to Messages ---")
    if not session_manager or not session_manager.my_profile_id:
        logger.error("Auto-Responder: SessionManager or profile ID missing.")
        return False

    my_pid_lower = session_manager.my_profile_id.lower()
    overall_success = True
    processed_count = 0
    responses_sent_count = 0
    archived_count = 0
    error_count = 0
    skipped_count = 0
    total_candidates = 0

    # --- Step 2: Get DB Session and Required MessageType ID ---
    db_session: Optional[DbSession] = None
    try:
        db_session = session_manager.get_db_conn()
        if not db_session:
            logger.critical("Auto-Responder: Failed to get DB session. Aborting.")
            return False

        # Get custom response message type ID
        custom_response_type = (
            db_session.query(MessageType.id)
            .filter(MessageType.type_name == CUSTOM_RESPONSE_MESSAGE_TYPE)
            .scalar()
        )

        if not custom_response_type:
            logger.critical(
                f"Auto-Responder: MessageType '{CUSTOM_RESPONSE_MESSAGE_TYPE}' not found in DB. Aborting."
            )
            if db_session:
                session_manager.return_session(db_session)
            return False

        # --- Step 3: Get Unprocessed Messages ---
        logger.debug("Retrieving unprocessed messages that require a response...")
        unprocessed_messages = get_unprocessed_messages(db_session)
        total_candidates = len(unprocessed_messages)

        if not unprocessed_messages:
            logger.info(
                "Auto-Responder: No unprocessed messages found that require a response."
            )
            if db_session:
                session_manager.return_session(db_session)
            return True

        logger.info(f"Auto-Responder: Found {total_candidates} messages to process.")

        # --- Step 4: Process Messages ---
        # Setup progress bar
        tqdm_args = {
            "total": total_candidates,
            "desc": "Processing",
            "unit": " message",
            "dynamic_ncols": True,
            "leave": True,
            "bar_format": "{desc} |{bar}| {percentage:3.0f}% ({n_fmt}/{total_fmt})",
            "file": sys.stderr,
        }

        with logging_redirect_tqdm(), tqdm(**tqdm_args) as progress_bar:
            for message_data in unprocessed_messages:
                processed_count += 1
                person = message_data["person"]
                conversation_id = message_data["conversation_id"]
                # Note: message_content and sentiment are available in message_data if needed later

                log_prefix = f"Auto-Responder: {person.username} #{person.id}"

                try:
                    # Update progress bar
                    if progress_bar:
                        progress_bar.set_description(f"Processing {person.username}")
                        progress_bar.update(1)

                    # --- Step 4a: Get Message Context ---
                    logger.debug(f"{log_prefix}: Getting message context...")
                    context_logs = _get_message_context(db_session, person.id)
                    if not context_logs:
                        logger.warning(
                            f"Skipping {log_prefix}: Failed to retrieve message context."
                        )
                        skipped_count += 1
                        continue

                    # --- Step 4b: Format Context for AI ---
                    formatted_context = _format_context_for_ai_extraction(
                        context_logs, my_pid_lower
                    )

                    # --- Step 4c: Call AI for Entity Extraction ---
                    logger.debug(f"{log_prefix}: Calling AI for entity extraction...")
                    if not session_manager.is_sess_valid():
                        logger.error(
                            f"Session invalid before AI extraction call for {log_prefix}. Skipping message."
                        )
                        error_count += 1
                        continue

                    ai_response = extract_and_suggest_tasks(
                        formatted_context, session_manager
                    )

                    # --- Step 4d: Process AI Response ---
                    processed_response = _process_ai_response(ai_response, log_prefix)
                    extracted_data = processed_response["extracted_data"]
                    # Note: suggested_tasks are available in processed_response if needed later

                    # Log the results
                    entity_counts = {k: len(v) for k, v in extracted_data.items()}
                    logger.debug(
                        f"{log_prefix}: Extracted entities: {json.dumps(entity_counts)}"
                    )

                    # --- Step 4e: Check if People are Mentioned ---
                    mentioned_names = extracted_data.get("mentioned_names", [])
                    if not mentioned_names:
                        logger.debug(
                            f"{log_prefix}: No names mentioned in message. Skipping person lookup."
                        )
                        skipped_count += 1
                        continue

                    # --- Step 4f: Search for Mentioned People ---
                    logger.debug(
                        f"{log_prefix}: Searching for mentioned people: {mentioned_names}"
                    )
                    # Use our search function to find people mentioned in the message
                    tree_search_results = _search_for_people(
                        session_manager, mentioned_names
                    )

                    # Check if any matches were found
                    matches = tree_search_results.get("results", [])
                    relationship_paths = tree_search_results.get(
                        "relationship_paths", {}
                    )

                    if not matches:
                        logger.debug(
                            f"{log_prefix}: No matches found for mentioned names. Skipping."
                        )
                        skipped_count += 1
                        continue

                    # --- Step 4g: Generate Response ---
                    # For now, we'll use a simple template-based response
                    # In the future, this will be replaced with an AI-generated response

                    # Format match details
                    match_details = []
                    for match in matches[:MAX_PEOPLE_TO_INCLUDE]:
                        match_id = match.get("id")
                        name = f"{match.get('first_name', '')} {match.get('surname', '')}".strip()
                        birth_year = match.get("birth_year", "Unknown")
                        birth_place = match.get("birth_place", "Unknown")
                        death_year = match.get("death_year", "Unknown")
                        death_place = match.get("death_place", "Unknown")

                        # Get relationship path if available
                        relationship_path = relationship_paths.get(match_id, "Unknown")

                        # Format details
                        detail = f"{name} (b. {birth_year}, {birth_place}"
                        if death_year and death_year != "Unknown":
                            detail += f", d. {death_year}, {death_place}"
                        detail += f")\nRelationship: {relationship_path}"

                        match_details.append(detail)

                    # Create response message
                    if len(match_details) == 1:
                        # Single person response
                        response_message = f"Thank you for your message about {matches[0].get('first_name', '')} {matches[0].get('surname', '')}. "
                        response_message += f"I've found this person in my family tree:\n\n{match_details[0]}\n\n"
                        response_message += "Would you like to know more about this person or any other relatives?"
                    else:
                        # Multiple people response
                        response_message = f"Thank you for your message mentioning several people. I've found the following in my family tree:\n\n"
                        response_message += "\n\n".join(match_details)
                        response_message += "\n\nWould you like more information about any of these individuals or their relationships?"

                    # --- Step 4h: Send Response ---
                    logger.debug(f"{log_prefix}: Sending response message...")
                    message_status, effective_conv_id = call_send_message_api(
                        session_manager,
                        person,
                        response_message,
                        conversation_id,
                        f"Auto-Responder: {person.username} #{person.id}",
                    )

                    # --- Step 4i: Update Database ---
                    if message_status in ("delivered OK", "typed (dry_run)"):
                        # Create new conversation log entry
                        new_log_entry = ConversationLog(
                            conversation_id=effective_conv_id,
                            direction=MessageDirectionEnum.OUT,
                            people_id=person.id,
                            latest_message_content=response_message[
                                : config_instance.MESSAGE_TRUNCATION_LENGTH
                            ],
                            latest_timestamp=datetime.now(timezone.utc),
                            ai_sentiment=None,  # Not applicable for OUT messages
                            message_type_id=custom_response_type,
                            script_message_status=message_status,
                        )

                        # Update person status to ARCHIVE
                        person.status = PersonStatusEnum.ARCHIVE
                        person.updated_at = datetime.now(timezone.utc)

                        # Commit changes to database
                        try:
                            with db_transn(db_session) as sess:
                                sess.add(new_log_entry)
                                sess.add(person)

                            responses_sent_count += 1
                            archived_count += 1
                            logger.info(
                                f"{log_prefix}: Response sent and person archived successfully."
                            )
                        except SQLAlchemyError as e:
                            logger.error(
                                f"{log_prefix}: Database error updating conversation log: {e}"
                            )
                            error_count += 1
                    else:
                        logger.error(
                            f"{log_prefix}: Failed to send response. Status: {message_status}"
                        )
                        error_count += 1

                except Exception as e:
                    logger.error(
                        f"{log_prefix}: Error processing message: {e}", exc_info=True
                    )
                    error_count += 1

        # --- Step 5: Log Summary ---
        logger.info("\n--- Auto-Responder Summary ---")
        logger.info(f"Total messages processed: {processed_count}")
        logger.info(f"Responses sent: {responses_sent_count}")
        logger.info(f"People archived: {archived_count}")
        logger.info(f"Messages skipped: {skipped_count}")
        logger.info(f"Errors encountered: {error_count}")

        # Return session to pool
        if db_session:
            session_manager.return_session(db_session)

        return overall_success

    except Exception as e:
        logger.critical(f"Auto-Responder: Critical error: {e}", exc_info=True)
        # Return session to pool
        if db_session:
            session_manager.return_session(db_session)
        return False
