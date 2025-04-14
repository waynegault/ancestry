#!/usr/bin/env python3

# action7_inbox.py

"""
action7_inbox.py - Process Ancestry Inbox Messages

Fetches conversations from the Ancestry inbox API, compares them with the local
database (`conversation_log`), identifies new incoming messages, classifies their
intent using AI (via `ai_interface.py`), and updates the database accordingly.
Uses batch processing, pagination (cursors), rate limiting, and handles potential
session invalidity.
"""

# --- Standard library imports ---
import enum
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
from urllib.parse import urljoin

# --- Third-party imports ---
import requests
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Index,  # Added Index
    Integer,
    String,
    Subquery,
    desc,
    func,
    inspect as sa_inspect,  # Added inspect
    over,
    text,
    update,
    select as sql_select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert 
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session as DbSession, aliased, joinedload
from selenium.common.exceptions import WebDriverException  
from tqdm.auto import tqdm 
from tqdm.contrib.logging import logging_redirect_tqdm 

# --- Local application imports ---
from ai_interface import classify_message_intent 
from config import config_instance 
from database import (  
    ConversationLog,
    MessageDirectionEnum,
    MessageType,  
    Person,
    PersonStatusEnum,
    db_transn,
    commit_bulk_data,
)
from logging_config import logger  
from utils import (  
    DynamicRateLimiter,
    SessionManager,
    _api_req,  
    format_name, 
    retry,
    retry_api,
    time_wait,
    urljoin,
    _fetch_profile_details_for_person
)


# --- Main Class ---
class InboxProcessor:
    """
    Handles the process of fetching, analyzing, and logging Ancestry inbox conversations.
    Uses API calls, AI sentiment analysis, and database interactions.
    """

    def __init__(self, session_manager: SessionManager):
        """Initializes the InboxProcessor."""
        # Step 1: Store SessionManager and Rate Limiter
        self.session_manager = session_manager
        self.dynamic_rate_limiter = (
            session_manager.dynamic_rate_limiter
        )  # Use manager's limiter

        # Step 2: Load Configuration Settings
        self.max_inbox_limit = (
            config_instance.MAX_INBOX
        )  # Max conversations to process (0=unlimited)
        # Determine batch size, ensuring it doesn't exceed the overall limit if one is set
        default_batch = min(
            config_instance.BATCH_SIZE, 50
        )  # Default batch size, capped at 50
        self.api_batch_size = (
            min(default_batch, self.max_inbox_limit)
            if self.max_inbox_limit > 0
            else default_batch
        )
        # AI Context settings
        self.ai_context_msg_count = config_instance.AI_CONTEXT_MESSAGES_COUNT
        self.ai_context_max_words = (
            config_instance.AI_CONTEXT_MESSAGE_MAX_WORDS
        )  # Correct assignment

        # Correct the attribute name in the logging statement
        logger.debug(
            f"InboxProcessor initialized: MaxInbox={self.max_inbox_limit}, BatchSize={self.api_batch_size}, AIContext={self.ai_context_msg_count} msgs / {self.ai_context_max_words} words."
        )
    # End of __init__

    # --- Private Helper Methods ---

    @retry_api()  # Apply retry decorator for resilience
    def _get_all_conversations_api(
        self, session_manager: SessionManager, limit: int, cursor: Optional[str] = None
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Fetches a single batch of conversation overview data from the Ancestry API.

        Args:
            session_manager: The active SessionManager instance.
            limit: The maximum number of conversations to fetch in this batch.
            cursor: The pagination cursor from the previous API response (if any).

        Returns:
            A tuple containing:
            - List of processed conversation info dictionaries, or None on failure.
            - The 'forward_cursor' string for the next page, or None if no more pages.

        Raises:
            WebDriverException: If the session becomes invalid before the API call.
        """
        # Step 1: Validate prerequisites
        if not session_manager or not session_manager.my_profile_id:
            logger.error(
                "_get_all_conversations_api: SessionManager or profile ID missing."
            )
            return None, None
        if not session_manager.is_sess_valid():
            logger.error("_get_all_conversations_api: Session invalid before API call.")
            # Raise exception so retry_api can potentially handle session restart if configured
            raise WebDriverException(
                "Session invalid before conversation overview API call"
            )

        # Step 2: Construct API URL with limit and optional cursor
        my_profile_id = session_manager.my_profile_id
        api_base = urljoin(config_instance.BASE_URL, "/app-api/express/v2/")
        # Note: API uses 'limit', not 'batch_size' parameter name
        url = f"{api_base}conversations?q=user:{my_profile_id}&limit={limit}"
        if cursor:
            url += f"&cursor={cursor}"
        logger.debug(
            f"Fetching conversation batch: Limit={limit}, Cursor='{cursor or 'None'}'"
        )

        # Step 3: Make API call using _api_req helper
        try:
            response_data = _api_req(
                url=url,
                driver=session_manager.driver,  # Pass driver for context/headers
                session_manager=session_manager,
                method="GET",
                use_csrf_token=False,  # Typically not needed for GET overviews
                api_description="Get Inbox Conversations",
            )

            # Step 4: Process API response
            if response_data is None:  # Indicates failure in _api_req after retries
                logger.warning("_get_all_conversations_api: _api_req returned None.")
                return None, None
            if not isinstance(response_data, dict):
                logger.error(
                    f"_get_all_conversations_api: Unexpected API response format. Type={type(response_data)}, Expected=dict"
                )
                return None, None

            # Step 5: Extract conversation data and pagination cursor
            conversations_raw = response_data.get("conversations", [])
            all_conversations_processed: List[Dict[str, Any]] = []
            if isinstance(conversations_raw, list):
                for conv_data in conversations_raw:
                    # Extract and process info for each conversation
                    info = self._extract_conversation_info(conv_data, my_profile_id)
                    if info:
                        all_conversations_processed.append(info)
            else:
                logger.warning(
                    "_get_all_conversations_api: 'conversations' key not found or not a list in API response."
                )

            forward_cursor = response_data.get("paging", {}).get("forward_cursor")
            logger.debug(
                f"API call returned {len(all_conversations_processed)} conversations. Next cursor: '{forward_cursor or 'None'}'."
            )

            # Step 6: Return processed data and cursor
            return all_conversations_processed, forward_cursor

        # Step 7: Handle exceptions (WebDriverException caught by retry_api, others logged)
        except WebDriverException as e:
            # Re-raise WebDriverException for retry_api to handle potentially
            logger.error(f"WebDriverException during _get_all_conversations_api: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in _get_all_conversations_api: {e}", exc_info=True
            )
            return None, None  # Return None on unexpected errors
    # End of _get_all_conversations_api

    def _extract_conversation_info(
        self, conv_data: Dict[str, Any], my_profile_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts and formats key information from a single conversation overview dictionary.

        Args:
            conv_data: The dictionary representing one conversation from the API response.
            my_profile_id: The profile ID of the script user (to identify the 'other' participant).

        Returns:
            A dictionary containing 'conversation_id', 'profile_id', 'username',
            and 'last_message_timestamp', or None if essential data is missing.
        """
        # Step 1: Basic validation of input structure
        if not isinstance(conv_data, dict):
            return None
        conversation_id = str(conv_data.get("id", ""))  # Ensure string
        last_message_data = conv_data.get("last_message", {})
        if not conversation_id or not isinstance(last_message_data, dict):
            logger.warning(
                f"Skipping conversation data due to missing ID or last_message: {conv_data}"
            )
            return None

        # Step 2: Extract and parse last message timestamp
        last_msg_ts_unix = last_message_data.get("created")  # Unix timestamp (seconds)
        last_msg_ts_aware: Optional[datetime] = None
        if isinstance(last_msg_ts_unix, (int, float)):
            try:
                # Validate timestamp range (simple check against plausible dates)
                min_ts, max_ts = 0, 32503680000  # Approx year 1970 to 3000
                if min_ts <= last_msg_ts_unix <= max_ts:
                    # Convert to timezone-aware datetime object (UTC)
                    last_msg_ts_aware = datetime.fromtimestamp(
                        last_msg_ts_unix, tz=timezone.utc
                    )
                else:
                    logger.warning(
                        f"Timestamp {last_msg_ts_unix} out of plausible range for ConvID {conversation_id}."
                    )
            except (ValueError, TypeError, OSError) as ts_err:
                logger.warning(
                    f"Error converting timestamp {last_msg_ts_unix} for ConvID {conversation_id}: {ts_err}"
                )
        # else: logger.debug(f"Timestamp missing or invalid type for ConvID {conversation_id}")

        # Step 3: Identify the 'other' participant's profile ID and username
        username = "Unknown"
        profile_id = "UNKNOWN"  # Default if other participant not found
        other_member_found = False
        members = conv_data.get("members", [])
        my_pid_lower = str(my_profile_id).lower()  # Lowercase for comparison

        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict):
                    continue
                member_user_id = member.get("user_id")
                member_user_id_str = (
                    str(member_user_id).lower() if member_user_id else ""
                )
                # Check if this member is not the script user
                if member_user_id_str and member_user_id_str != my_pid_lower:
                    profile_id = str(
                        member_user_id
                    ).upper()  # Store uppercase profile ID
                    username = member.get("display_name", "Unknown")  # Get display name
                    other_member_found = True
                    break  # Assume only one other participant per conversation
        if not other_member_found:
            logger.warning(
                f"Could not identify other participant in ConvID {conversation_id}. Members: {members}"
            )

        # Step 4: Return the structured dictionary
        return {
            "conversation_id": conversation_id,
            "profile_id": profile_id,  # Profile ID of the other person
            "username": username,  # Display name of the other person
            "last_message_timestamp": last_msg_ts_aware,  # Aware datetime object or None
        }
    # End of _extract_conversation_info

    @retry_api(max_retries=2)  # Allow retries for fetching context
    def _fetch_conversation_context(
        self, conversation_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches the last N messages (defined by config) for a specific conversation ID
        to provide context for AI classification.

        Args:
            conversation_id: The ID of the conversation to fetch context for.

        Returns:
            A list of message dictionaries (sorted oldest to newest), or None on failure.
            Each dictionary contains 'content', 'author' (lowercase), 'timestamp' (aware datetime),
            and 'conversation_id'.

        Raises:
            WebDriverException: If the session becomes invalid during the API call.
        """
        # Step 1: Validate inputs and session state
        if not conversation_id:
            logger.warning("_fetch_conversation_context: No conversation_id provided.")
            return None
        if not self.session_manager or not self.session_manager.my_profile_id:
            logger.error(
                "_fetch_conversation_context: SessionManager or profile ID missing."
            )
            return None
        if not self.session_manager.is_sess_valid():
            logger.error(
                f"_fetch_conversation_context: Session invalid fetching context for ConvID {conversation_id}."
            )
            raise WebDriverException(
                f"Session invalid fetching context ConvID {conversation_id}"
            )

        # Step 2: Construct API URL and Headers
        context_messages: List[Dict[str, Any]] = []
        api_base = urljoin(config_instance.BASE_URL, "/app-api/express/v2/")
        limit = self.ai_context_msg_count  # Get limit from config
        api_description = "Fetch Conversation Context"
        # Prepare headers (using contextual headers where possible)
        headers = config_instance.API_CONTEXTUAL_HEADERS.get(api_description, {}).copy()
        # Ensure ancestry-userid is set correctly
        if "ancestry-userid" in headers:
            headers["ancestry-userid"] = self.session_manager.my_profile_id.upper()
        # Construct URL
        url = f"{api_base}conversations/{conversation_id}/messages?limit={limit}"
        logger.debug(
            f"Fetching context (last {limit} msgs) for ConvID {conversation_id}..."
        )

        # Step 3: Make API call and apply rate limiting wait first
        try:
            # Apply rate limit wait *before* the call
            wait_time = self.dynamic_rate_limiter.wait()
            # Optional: log wait time if significant
            # if wait_time > 0.1: logger.debug(f"Rate limit wait for context fetch: {wait_time:.2f}s")

            response_data = _api_req(
                url=url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                headers=headers,
                use_csrf_token=False,  # Not typically needed
                api_description=api_description,
            )

            # Step 4: Process the response
            if not isinstance(response_data, dict):
                logger.warning(
                    f"{api_description}: Bad response type {type(response_data)} for ConvID {conversation_id}."
                )
                return None  # Failed fetch

            messages_batch = response_data.get("messages", [])
            if not isinstance(messages_batch, list):
                logger.warning(
                    f"{api_description}: 'messages' key not a list for ConvID {conversation_id}."
                )
                return None  # Invalid structure

            # Step 5: Extract and format message details
            for msg_data in messages_batch:
                if not isinstance(msg_data, dict):
                    continue  # Skip invalid entries
                ts_unix = msg_data.get("created")
                msg_timestamp: Optional[datetime] = None
                if isinstance(ts_unix, (int, float)):
                    try:
                        msg_timestamp = datetime.fromtimestamp(
                            ts_unix, tz=timezone.utc
                        )  # Convert to aware datetime
                    except Exception as ts_err:
                        logger.warning(
                            f"Error parsing timestamp {ts_unix} in ConvID {conversation_id}: {ts_err}"
                        )

                # Prepare standardized message dictionary
                processed_msg = {
                    "content": str(msg_data.get("content", "")),  # Ensure string
                    "author": str(
                        msg_data.get("author", "")
                    ).lower(),  # Store author ID lowercase
                    "timestamp": msg_timestamp,  # Store aware datetime or None
                    "conversation_id": conversation_id,  # Add conversation ID for context
                }
                context_messages.append(processed_msg)

            # Step 6: Sort messages by timestamp (oldest first) for correct context order
            return sorted(
                context_messages,
                # Provide default datetime for sorting if timestamp is None
                key=lambda x: x.get("timestamp")
                or datetime.min.replace(tzinfo=timezone.utc),
            )

        # Step 7: Handle exceptions
        except WebDriverException as e:
            logger.error(
                f"WebDriverException fetching context for ConvID {conversation_id}: {e}"
            )
            raise  # Re-raise for retry_api
        except Exception as e:
            logger.error(
                f"Unexpected error fetching context for ConvID {conversation_id}: {e}",
                exc_info=True,
            )
            return None  # Return None on unexpected errors
    # End of _fetch_conversation_context

    def _format_context_for_ai(
        self, context_messages: List[Dict], my_pid_lower: str
    ) -> str:
        """
        Formats a list of message dictionaries (sorted oldest to newest) into a
        single string suitable for the AI classification prompt. Truncates long messages.

        Args:
            context_messages: List of processed message dictionaries.
            my_pid_lower: The script user's profile ID (lowercase) to label messages correctly.

        Returns:
            A formatted string representing the conversation history.
        """
        # Step 1: Initialize list for formatted lines
        context_lines = []
        # Step 2: Iterate through messages (assumed sorted oldest to newest)
        for msg in context_messages:
            # Step 2a: Determine label (SCRIPT or USER)
            author_lower = msg.get("author", "")
            label = "SCRIPT: " if author_lower == my_pid_lower else "USER: "
            # Step 2b: Get and truncate message content
            content = msg.get("content", "")
            words = content.split()
            if len(words) > self.ai_context_max_words:
                # Truncate by word count and add ellipsis
                truncated_content = " ".join(words[: self.ai_context_max_words]) + "..."
            else:
                truncated_content = content
            # Step 2c: Append formatted line
            context_lines.append(f"{label}{truncated_content}")
        # Step 3: Join lines into a single string
        return "\n".join(context_lines)
    # End of _format_context_for_ai

    def _lookup_or_create_person(
        self,
        session: DbSession,
        profile_id: str,  # Already uppercase
        username: str,
        conversation_id: Optional[str],  # For constructing message link if creating
        existing_person_arg: Optional[
            Person
        ] = None,  # Pass prefetched Person if available
    ) -> Tuple[Optional[Person], Literal["new", "updated", "skipped", "error"]]:
        """
        Looks up a Person by profile_id. If found, checks for updates (username,
        message_link). If not found, creates a new Person record, fetching additional
        profile details (first name, contactable, last login) via API if possible.

        Args:
            session: The active SQLAlchemy database session.
            profile_id: The profile ID of the person (UPPERCASE).
            username: The display name from the conversation overview.
            conversation_id: The current conversation ID (used only for logging context).
            existing_person_arg: The prefetched Person object, if found earlier.

        Returns:
            A tuple containing:
            - The found or newly created Person object (or None on error).
            - A status string: 'new', 'updated', 'skipped', 'error'.
        """
        # Step 1: Basic validation
        if not profile_id or profile_id == "UNKNOWN":
            logger.warning("_lookup_or_create_person: Invalid profile_id provided.")
            return None, "error"
        username_to_use = username or "Unknown"  # Ensure username is not None
        log_ref = f"ProfileID={profile_id}/User='{username_to_use}' (ConvID: {conversation_id or 'N/A'})"

        person: Optional[Person] = None
        status: Literal["new", "updated", "skipped", "error"] = (
            "error"  # Default status
        )

        # Step 2: Determine if lookup is needed or use prefetched person
        if existing_person_arg:
            person = existing_person_arg
            logger.debug(f"{log_ref}: Using prefetched Person ID {person.id}.")
        else:
            # If not prefetched, query DB by profile ID
            try:
                logger.debug(
                    f"{log_ref}: Prefetched person not found. Querying DB by Profile ID..."
                )
                person = (
                    session.query(Person)
                    .filter(Person.profile_id == profile_id)
                    .first()
                )
            except SQLAlchemyError as e:
                logger.error(
                    f"DB error looking up Person {log_ref}: {e}", exc_info=True
                )
                return None, "error"  # DB error during lookup

        # Step 3: Process based on whether person was found
        if person:
            # --- Person Exists: Check for Updates ---
            updated = False
            # Update username only if current is 'Unknown' or different (use formatted name)
            formatted_username = format_name(username_to_use)
            if person.username == "Unknown" or person.username != formatted_username:
                logger.debug(
                    f"Updating username for {log_ref} from '{person.username}' to '{formatted_username}'."
                )
                person.username = formatted_username
                updated = True
            # Update message link if missing or different
            correct_message_link = urljoin(
                config_instance.BASE_URL, f"/messaging/?p={profile_id.upper()}"
            )
            if person.message_link != correct_message_link:
                logger.debug(f"Updating message link for {log_ref}.")
                person.message_link = correct_message_link
                updated = True
            # Optional: Add logic here to fetch details and update other fields if they are NULL

            if updated:
                person.updated_at = datetime.now(timezone.utc)  # Set update timestamp
                try:
                    session.add(person)  # Ensure updates are staged
                    session.flush()  # Apply update within the transaction
                    status = "updated"
                    logger.debug(
                        f"Successfully staged updates for Person {log_ref} (ID: {person.id})."
                    )
                except (IntegrityError, SQLAlchemyError) as upd_err:
                    logger.error(
                        f"DB error flushing update for Person {log_ref}: {upd_err}"
                    )
                    session.rollback()  # Rollback on flush error
                    return None, "error"
            else:
                status = "skipped"  # No updates needed
                # logger.debug(f"No updates needed for existing Person {log_ref} (ID: {person.id}).")

        else:
            # --- Person Not Found: Create New ---
            logger.debug(f"Person {log_ref} not found. Creating new record...")
            # Fetch additional details via API before creating
            profile_details = _fetch_profile_details_for_person(
                self.session_manager, profile_id
            )

            # Prepare data for new Person object
            new_person_data = {
                "profile_id": profile_id.upper(),
                "username": format_name(username_to_use),  # Use formatted username
                "message_link": urljoin(
                    config_instance.BASE_URL, f"/messaging/?p={profile_id.upper()}"
                ),
                "status": PersonStatusEnum.ACTIVE,  # Default status
                # Initialize other fields to defaults or None
                "first_name": None,
                "contactable": True,
                "last_logged_in": None,
                "administrator_profile_id": None,
                "administrator_username": None,
                "gender": None,
                "birth_year": None,
                "in_my_tree": False,
                "uuid": None,  # UUID might be added later by Action 6
            }
            # Populate with fetched details if available
            if profile_details:
                logger.debug(
                    f"Populating new person {log_ref} with fetched profile details."
                )
                new_person_data["first_name"] = profile_details.get("first_name")
                new_person_data["contactable"] = bool(
                    profile_details.get("contactable", False)
                )  # Default False if fetch fails
                new_person_data["last_logged_in"] = profile_details.get(
                    "last_logged_in_dt"
                )
            else:
                logger.warning(
                    f"Could not fetch profile details for new person {log_ref}. Using defaults."
                )

            # Create and add the new Person
            try:
                new_person = Person(**new_person_data)
                session.add(new_person)
                session.flush()  # Flush to get ID assigned immediately
                if new_person.id is None:
                    logger.error(
                        f"ID not assigned after flush for new person {log_ref}! Rolling back."
                    )
                    session.rollback()
                    return None, "error"
                person = new_person  # Assign the newly created person object
                status = "new"
                logger.debug(f"Created new Person ID {person.id} for {log_ref}.")
            except (IntegrityError, SQLAlchemyError) as create_err:
                logger.error(f"DB error creating Person {log_ref}: {create_err}")
                session.rollback()  # Rollback on creation error
                return None, "error"
            except Exception as e:
                logger.critical(
                    f"Unexpected error creating Person {log_ref}: {e}", exc_info=True
                )
                session.rollback()
                return None, "error"

        # Step 4: Return the person object and status
        return person, status
    # End of _lookup_or_create_person

    def _create_comparator(self, session: DbSession) -> Optional[Dict[str, Any]]:
        """
        Finds the most recent ConversationLog entry (highest timestamp) in the database
        to use as a comparison point for stopping inbox processing early.

        Args:
            session: The active SQLAlchemy database session.

        Returns:
            A dictionary {'conversation_id': str, 'latest_timestamp': datetime}
            representing the comparator entry, or None if the table is empty or an error occurs.
            Timestamp is guaranteed to be timezone-aware (UTC).
        """
        latest_log_entry_info: Optional[Dict[str, Any]] = None
        logger.debug("Creating comparator by finding latest ConversationLog entry...")
        try:
            # Step 1: Query for the entry with the maximum timestamp
            # Order by descending timestamp, handle NULLs last, take the first result
            latest_entry = (
                session.query(
                    ConversationLog.conversation_id, ConversationLog.latest_timestamp
                )
                .order_by(ConversationLog.latest_timestamp.desc().nullslast())
                .first()
            )

            # Step 2: Process the result if found
            if latest_entry:
                log_conv_id = latest_entry.conversation_id
                log_timestamp = latest_entry.latest_timestamp
                # Ensure the timestamp is timezone-aware UTC
                aware_timestamp: Optional[datetime] = None
                if isinstance(log_timestamp, datetime):
                    aware_timestamp = (
                        log_timestamp.replace(tzinfo=timezone.utc)
                        if log_timestamp.tzinfo is None
                        else log_timestamp.astimezone(timezone.utc)
                    )

                # Step 3: Validate data and create comparator dictionary
                if log_conv_id and aware_timestamp:
                    latest_log_entry_info = {
                        "conversation_id": log_conv_id,
                        "latest_timestamp": aware_timestamp,
                    }
                    logger.debug(
                        f"Comparator created: ConvID={latest_log_entry_info['conversation_id']}, TS={latest_log_entry_info['latest_timestamp']}"
                    )
                else:
                    logger.warning(
                        f"Found latest log entry, but data invalid/missing timestamp: ConvID={log_conv_id}, Raw TS={log_timestamp}"
                    )
            else:
                # Step 4: Log if table is empty
                logger.info(
                    "ConversationLog table appears empty. Comparator not created."
                )

        # Step 5: Handle errors during query
        except Exception as e:
            logger.error(f"Error creating comparator from database: {e}", exc_info=True)
            return None  # Return None on error

        # Step 6: Return the comparator info dictionary or None
        return latest_log_entry_info
    # End of _create_comparator

    # --- Main Public Methods ---

    def search_inbox(self) -> bool:
        """
        Main method to search the Ancestry inbox.
        Fetches conversations page by page, compares with the database,
        identifies new/updated conversations, fetches context for new incoming
        messages, classifies intent using AI, and updates the database in batches.
        Uses a comparator to potentially stop early if previously seen messages are reached.
        Handles session validity checks and includes a progress bar if MAX_INBOX is set.

        Returns:
            True if the process completed without critical errors, False otherwise.
        """
        # --- Step 1: Initialization ---
        logger.debug("--- Starting Action 7: Search Inbox ---")
        # Counters and state for this run
        ai_classified_count = 0
        status_updated_count = 0  # Person status updates (e.g., to DESIST)
        total_processed_api_items = 0  # Conversations returned by API
        items_processed_before_stop = 0  # Conversations actually processed in loop
        stop_reason: Optional[str] = None  # Reason for stopping early
        overall_success = True  # Assume success until error

        # Validate session manager state
        if not self.session_manager or not self.session_manager.my_profile_id:
            logger.error("search_inbox: Session manager or profile ID missing.")
            return False
        my_pid_lower = self.session_manager.my_profile_id.lower()

        # --- Step 2: Get DB Session and Comparator ---
        session: Optional[DbSession] = None
        try:
            session = self.session_manager.get_db_conn()
            if not session:
                logger.critical("search_inbox: Failed to get DB session. Aborting.")
                return False

            # Get the comparator (latest message in DB)
            comparator_info = self._create_comparator(session)
            comp_conv_id: Optional[str] = None
            comp_ts: Optional[datetime] = None  # Comparator timestamp (aware)
            if comparator_info:
                comp_conv_id = comparator_info.get("conversation_id")
                comp_ts = comparator_info.get("latest_timestamp")
            # --- End Get Comparator ---

            # --- Step 3: Setup and Run Main Loop ---
            max_inbox_str = (
                str(self.max_inbox_limit) if self.max_inbox_limit > 0 else "Unlimited"
            )
            logger.debug(
                f"Starting inbox search (MaxInbox={max_inbox_str}, BatchSize={self.api_batch_size}, Comparator={comp_conv_id or 'None'})..."
            )

            # Decide whether to use progress bar based on limit
            use_progress_bar = self.max_inbox_limit > 0

            # --- Conditional Execution with or without Progress Bar ---

            if use_progress_bar:
                # Run with tqdm context manager
                tqdm_args = {
                    "total": self.max_inbox_limit,
                    "desc": "",  # Explicitly empty description
                    "unit": " conv",
                    # "ncols": 100,  # Keep fixed width
                    "dynamic_ncols": True,
                    "leave": True,  # Keep the bar after completion
                    "bar_format": "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
                    # "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt}",
                    "file": sys.stderr,
                    # "dynamic_ncols": False,
                    # "ascii": True,
                }
                logger.info(f"Processing up to {self.max_inbox_limit} inbox items...\n")
                with logging_redirect_tqdm(), tqdm(**tqdm_args) as progress_bar:
                    # Keep the set_postfix call in _process_inbox_loop if desired,
                    # but it won't display with this bar_format.
                    # If you want postfix, add it back to bar_format:
                    # "bar_format": "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}",
                    (
                        stop_reason,
                        total_processed_api_items,
                        ai_classified_count,
                        status_updated_count,
                        items_processed_before_stop,
                    ) = self._process_inbox_loop(
                        session, comp_conv_id, comp_ts, my_pid_lower, progress_bar
                    )
            else:
                # Run without tqdm progress bar
                # Call internal loop function, passing None for the progress bar
                (
                    stop_reason,
                    total_processed_api_items,
                    ai_classified_count,
                    status_updated_count,  
                    items_processed_before_stop,
                ) = self._process_inbox_loop( 
                    session,
                    comp_conv_id,
                    comp_ts,
                    my_pid_lower,
                    None, 
                )
            # Check if loop stopped due to an error state
            if stop_reason and "error" in stop_reason.lower():
                overall_success = False

        # --- Step 4: Handle Outer Exceptions ---
        except Exception as outer_e:
            logger.critical(
                f"CRITICAL error during search_inbox execution: {outer_e}",
                exc_info=True,
            )
            overall_success = False

        # --- Step 5: Final Logging and Cleanup ---
        finally:
            # Log summary of the run
            final_stop_reason = stop_reason or (
                "Comparator/End Reached" if overall_success else "Unknown"
            )
            # Use updated status_updated_count which now correctly tracks Person updates
            self._log_unified_summary(
                total_api_items=total_processed_api_items,
                items_processed=items_processed_before_stop,
                new_logs=0,  # Can no longer accurately track upserts easily, set to 0
                ai_classified=ai_classified_count,
                status_updates=status_updated_count,  # This count is correct
                stop_reason=final_stop_reason,
                max_inbox_limit=self.max_inbox_limit,
            )
            # Release the database session
            if session:
                self.session_manager.return_session(session)

        # Step 6: Return overall success status
        return overall_success
        # End of search_inbox
    # End of search_inbox

    def _process_inbox_loop(
        self,
        session: DbSession,
        comp_conv_id: Optional[str],
        comp_ts: Optional[datetime],  # Aware datetime
        my_pid_lower: str,
        progress_bar: Optional[tqdm],  # Accept progress bar instance
    ) -> Tuple[Optional[str], int, int, int, int]:  # <-- Removed logs_processed_count
        """
        Internal helper: Contains the main loop for fetching and processing inbox batches.

        Args:
            session: The active SQLAlchemy database session.
            comp_conv_id: Conversation ID of the comparator message.
            comp_ts: Timestamp of the comparator message (UTC aware).
            my_pid_lower: Lowercase profile ID of the script user.
            progress_bar: Optional tqdm instance to update.

        Returns:
            Tuple: (stop_reason, total_api_items, ai_classified_count,
                    status_updated_count, items_processed_before_stop)
        """
        # Step 1: Initialize loop-specific state
        ai_classified_count = 0
        status_updated_count = 0  # Tracks Person updates
        total_processed_api_items = 0
        items_processed_before_stop = 0
        logs_processed_in_run = 0 
        skipped_count_this_loop = 0
        error_count_this_loop = 0 
        stop_reason: Optional[str] = None
        next_cursor: Optional[str] = None
        current_batch_num = 0
        # Lists to accumulate data for batch commits
        conv_log_upserts_dicts: List[Dict[str, Any]] = []  # <-- Store dicts now
        person_updates: Dict[int, PersonStatusEnum] = {}
        stop_processing = False
        min_aware_dt = datetime.min.replace(tzinfo=timezone.utc)

        # Step 2: Main loop - continues until stop condition met
        while not stop_processing:
            try:  # Inner try for handling exceptions within a single batch iteration
                # Step 2a: Check session validity before each API call
                if not self.session_manager.is_sess_valid():
                    logger.error("Session became invalid during inbox processing loop.")
                    raise WebDriverException(
                        "Session invalid before overview batch fetch"
                    )

                # Step 2b: Calculate API Limit for this batch, considering overall limit
                current_limit = self.api_batch_size
                if self.max_inbox_limit > 0:
                    remaining_allowed = (
                        self.max_inbox_limit - items_processed_before_stop
                    )
                    if remaining_allowed <= 0:
                        # Stop *before* fetching if limit already reached
                        stop_reason = f"Inbox Limit ({self.max_inbox_limit})"
                        stop_processing = True
                        break
                    current_limit = min(self.api_batch_size, remaining_allowed)

                # Step 2c: Fetch a batch of conversations from API
                all_conversations_batch, next_cursor_from_api = (
                    self._get_all_conversations_api(
                        self.session_manager, limit=current_limit, cursor=next_cursor
                    )
                )

                # Step 2d: Handle API failure
                if all_conversations_batch is None:
                    logger.error(
                        "API call to fetch conversation batch failed critically."
                    )
                    stop_reason = "API Error Fetching Batch"
                    stop_processing = True
                    break

                # Step 2e: Process fetched batch
                batch_api_item_count = len(all_conversations_batch)
                total_processed_api_items += batch_api_item_count
                current_batch_num += 1
                logger.debug(
                    f"Processing Batch {current_batch_num} ({batch_api_item_count} items)..."
                )

                # Handle empty batch result
                if batch_api_item_count == 0:
                    if (
                        not next_cursor_from_api
                    ):  # No items AND no next cursor = end of inbox
                        stop_reason = "End of Inbox Reached (Empty Batch, No Cursor)"
                        stop_processing = True
                        break
                    else:  # Empty batch BUT cursor exists (API might sometimes do this)
                        logger.debug(
                            "API returned empty batch but provided cursor. Continuing fetch."
                        )
                        next_cursor = next_cursor_from_api
                        continue  # Fetch next batch

                # Apply rate limiter delay after successful fetch
                # wait_duration = self.dynamic_rate_limiter.wait()
                # if wait_duration > 0.1: logger.debug(f"API batch fetch wait: {wait_duration:.2f}s")

                # Step 2f: Pre-fetch existing Person and ConversationLog data for the batch
                batch_conv_ids = [
                    c["conversation_id"]
                    for c in all_conversations_batch
                    if c.get("conversation_id")
                ]
                batch_profile_ids = {
                    c.get("profile_id", "").upper()
                    for c in all_conversations_batch
                    if c.get("profile_id") and c.get("profile_id") != "UNKNOWN"
                }
                existing_persons_map: Dict[str, Person] = {}
                existing_conv_logs: Dict[Tuple[str, str], ConversationLog] = (
                    {}
                )  # Key: (conv_id, direction_enum.name)
                try:
                    if batch_profile_ids:
                        persons = (
                            session.query(Person)
                            .filter(Person.profile_id.in_(batch_profile_ids))
                            .all()
                        )
                        existing_persons_map = {
                            p.profile_id: p for p in persons if p.profile_id
                        }
                    if batch_conv_ids:
                        logs = (
                            session.query(ConversationLog)
                            .filter(ConversationLog.conversation_id.in_(batch_conv_ids))
                            .all()
                        )
                        # Ensure timestamps are aware before putting in map
                        for log in logs:
                            if (
                                log.latest_timestamp
                                and log.latest_timestamp.tzinfo is None
                            ):
                                log.latest_timestamp = log.latest_timestamp.replace(
                                    tzinfo=timezone.utc
                                )
                        existing_conv_logs = {
                            (log.conversation_id, log.direction.name): log
                            for log in logs
                            if log.direction
                        }
                except SQLAlchemyError as db_err:
                    logger.error(
                        f"DB prefetch failed for Batch {current_batch_num}: {db_err}"
                    )
                    # Decide whether to skip batch or abort run - Abort for now
                    stop_reason = "DB Prefetch Error"
                    stop_processing = True
                    break

                # Step 2g: Process each conversation in the batch
                for conversation_info in all_conversations_batch:
                    # --- Check Max Inbox Limit ---
                    if (
                        self.max_inbox_limit > 0
                        and items_processed_before_stop >= self.max_inbox_limit
                    ):
                        if not stop_reason:
                            stop_reason = f"Inbox Limit ({self.max_inbox_limit})"
                        stop_processing = True
                        break  # Break inner conversation loop

                    items_processed_before_stop += 1

                    # Update progress bar immediately

                    if progress_bar:
                        progress_bar.update(1)

                    # Extract key info
                    profile_id_upper = conversation_info.get(
                        "profile_id", "UNKNOWN"
                    ).upper()
                    api_conv_id = conversation_info.get("conversation_id")
                    api_latest_ts_aware = conversation_info.get(
                        "last_message_timestamp"
                    )  # Already aware from _extract

                    if not api_conv_id or profile_id_upper == "UNKNOWN":
                        logger.debug(
                            f"Skipping item {items_processed_before_stop}: Invalid ConvID/ProfileID."
                        )
                        continue

                    # --- Comparator Logic & Fetch Decision ---
                    needs_fetch = False
                    # Check 1: Is this the comparator conversation?
                    if comp_conv_id and api_conv_id == comp_conv_id:
                        stop_processing = (
                            True  # Found comparator, plan to stop after this item
                        )
                        # Fetch only if API timestamp is newer than comparator timestamp
                        if (
                            comp_ts
                            and api_latest_ts_aware
                            and api_latest_ts_aware > comp_ts
                        ):
                            needs_fetch = True
                        else:
                            # Comparator found, timestamp not newer or invalid -> stop and don't fetch
                            if not stop_reason:
                                stop_reason = "Comparator Found (No Change)"
                            break  # Stop processing immediately after comparator check passes

                    # Check 2: Not comparator, compare API timestamp with DB timestamps
                    else:
                        db_log_in = existing_conv_logs.get(
                            (api_conv_id, MessageDirectionEnum.IN.name)
                        )
                        db_log_out = existing_conv_logs.get(
                            (api_conv_id, MessageDirectionEnum.OUT.name)
                        )
                        # Get latest *overall* timestamp from DB for this conversation (ensure aware)
                        db_latest_ts_in = (
                            db_log_in.latest_timestamp
                            if db_log_in and db_log_in.latest_timestamp
                            else min_aware_dt
                        )
                        db_latest_ts_out = (
                            db_log_out.latest_timestamp
                            if db_log_out and db_log_out.latest_timestamp
                            else min_aware_dt
                        )
                        db_latest_overall_for_conv = max(
                            db_latest_ts_in, db_latest_ts_out
                        )
                        # Fetch if API timestamp is newer OR if no DB logs exist at all
                        if (
                            api_latest_ts_aware
                            and api_latest_ts_aware > db_latest_overall_for_conv
                        ):
                            needs_fetch = True
                        elif not db_log_in and not db_log_out:  # No record in DB yet
                            needs_fetch = True
                    # --- End Comparator Logic ---

                    # Skip if no fetch needed
                    if not needs_fetch:
                        # logger.debug(f"Skipping ConvID {api_conv_id}: No fetch needed (up-to-date).")
                        skipped_count_this_loop += 1
                        if stop_processing:
                            break  # Break inner loop if comparator was hit
                        continue  # Move to next conversation

                    # --- Fetch Context & Process Message ---
                    if (
                        not self.session_manager.is_sess_valid()
                    ):  # Check session before fetch
                        raise WebDriverException(
                            f"Session invalid before fetching context for ConvID {api_conv_id}"
                        )
                    context_messages = self._fetch_conversation_context(api_conv_id)
                    if context_messages is None:
                        error_count_this_loop += 1
                        logger.error(
                            f"Failed to fetch context for ConvID {api_conv_id}. Skipping item."
                        )
                        continue  # Skip this conversation if context fetch fails

                    # --- Lookup/Create Person ---
                    person, person_op_status = self._lookup_or_create_person(
                        session,
                        profile_id_upper,
                        conversation_info.get("username", "Unknown"),
                        api_conv_id,
                        existing_person_arg=existing_persons_map.get(
                            profile_id_upper
                        ),  # Pass prefetched if available
                    )
                    if not person or not person.id:
                        error_count_this_loop += 1
                        logger.error(
                            f"Failed person lookup/create for ConvID {api_conv_id}. Skipping item."
                        )
                        continue  # Cannot proceed without person record
                    people_id = person.id

                    # --- Process Fetched Context Messages ---
                    latest_ctx_in: Optional[Dict] = None
                    latest_ctx_out: Optional[Dict] = None
                    # Find the latest IN and OUT message from the fetched context
                    for msg in reversed(context_messages):  # Iterate newest first
                        author_lower = msg.get("author", "")
                        if author_lower != my_pid_lower and latest_ctx_in is None:
                            latest_ctx_in = msg
                        elif author_lower == my_pid_lower and latest_ctx_out is None:
                            latest_ctx_out = msg
                        if latest_ctx_in and latest_ctx_out:
                            break  # Stop when both found

                    ai_sentiment_result: Optional[str] = None
                    # --- Process IN Row ---
                    if latest_ctx_in:
                        ctx_ts_in_aware = latest_ctx_in.get(
                            "timestamp"
                        )  # Already aware
                        # Compare context timestamp with DB timestamp *for IN direction*
                        db_ts_in_for_compare = existing_conv_logs.get(
                            (api_conv_id, MessageDirectionEnum.IN.name), None
                        )
                        db_latest_ts_in_compare = (
                            db_ts_in_for_compare.latest_timestamp
                            if db_ts_in_for_compare
                            and db_ts_in_for_compare.latest_timestamp
                            else min_aware_dt
                        )
                        # Process only if context message is newer than DB record for IN
                        if (
                            ctx_ts_in_aware
                            and ctx_ts_in_aware > db_latest_ts_in_compare
                        ):
                            # Prepare context for AI
                            formatted_context = self._format_context_for_ai(
                                context_messages, my_pid_lower
                            )
                            # Call AI for classification
                            if (
                                not self.session_manager.is_sess_valid()
                            ):  # Check session before AI call
                                raise WebDriverException(
                                    f"Session invalid before AI classification call for ConvID {api_conv_id}"
                                )
                            ai_sentiment_result = classify_message_intent(
                                formatted_context, self.session_manager
                            )
                            if ai_sentiment_result:
                                ai_classified_count += (
                                    1  # Count successful classifications
                                )
                            else:
                                logger.warning(
                                    f"AI classification failed for ConvID {api_conv_id}."
                                )

                            # Prepare data DICTIONARY for ConversationLog upsert (IN row)
                            # Ensure required keys for commit_bulk_data are present
                            upsert_dict_in = {
                                "conversation_id": api_conv_id,
                                "direction": MessageDirectionEnum.IN,  # Pass Enum here, commit func can handle
                                "people_id": people_id,
                                "latest_message_content": latest_ctx_in.get(
                                    "content", ""
                                )[: config_instance.MESSAGE_TRUNCATION_LENGTH],
                                "latest_timestamp": ctx_ts_in_aware,  # Already aware UTC
                                "ai_sentiment": ai_sentiment_result,  # Store AI result
                                "message_type_id": None,
                                "script_message_status": None,
                            }
                            # Add dict to list for batch commit
                            conv_log_upserts_dicts.append(
                                upsert_dict_in
                            )  # <-- Store dict

                            # Stage Person status update if AI classified as DESIST/UNINTERESTED
                            if ai_sentiment_result in ("DESIST", "UNINTERESTED"):
                                logger.debug(
                                    f"AI classified ConvID {api_conv_id} (PersonID {people_id}) as '{ai_sentiment_result}'. Staging status update to DESIST."
                                )
                                # --- CORRECTED LOGIC ---
                                # Directly assign the Enum value to the person ID key
                                person_updates[people_id] = PersonStatusEnum.DESIST
                                # --- END CORRECTION ---

                    # --- Process OUT Row ---
                    if latest_ctx_out:
                        ctx_ts_out_aware = latest_ctx_out.get(
                            "timestamp"
                        )  # Already aware
                        # Compare context timestamp with DB timestamp *for OUT direction*
                        db_ts_out_for_compare = existing_conv_logs.get(
                            (api_conv_id, MessageDirectionEnum.OUT.name), None
                        )
                        db_latest_ts_out_compare = (
                            db_ts_out_for_compare.latest_timestamp
                            if db_ts_out_for_compare
                            and db_ts_out_for_compare.latest_timestamp
                            else min_aware_dt
                        )
                        # Process only if context message is newer than DB record for OUT
                        if (
                            ctx_ts_out_aware
                            and ctx_ts_out_aware > db_latest_ts_out_compare
                        ):
                            # Prepare data DICTIONARY for ConversationLog upsert (OUT row)
                            upsert_dict_out = {
                                "conversation_id": api_conv_id,
                                "direction": MessageDirectionEnum.OUT,  # Pass Enum
                                "people_id": people_id,
                                "latest_message_content": latest_ctx_out.get(
                                    "content", ""
                                )[: config_instance.MESSAGE_TRUNCATION_LENGTH],
                                "latest_timestamp": ctx_ts_out_aware,  # Already aware UTC
                                "ai_sentiment": None,
                                "message_type_id": None,  # Should be updated by Action 8 if script sent last
                                "script_message_status": None,  # Should be updated by Action 8
                            }
                            # Add dict to list for batch commit
                            conv_log_upserts_dicts.append(
                                upsert_dict_out
                            )  # <-- Store dict
                    # --- End Context Processing ---

                    if progress_bar:
                        progress_bar.set_postfix(
                            AI=ai_classified_count, # AI classifications this run
                            Archived=status_updated_count, # Persons archived this run
                            Skip=skipped_count_this_loop, # Skipped in this loop
                            Err=error_count_this_loop, # Errors in this loop
                            refresh=True # Use refresh=True for immediate update
                        )

                    # Check stop flag again after processing item (if comparator was hit)
                    if stop_processing:
                        break  # Break inner conversation loop

                # --- End Inner Loop (Processing conversations in batch) ---

                # --- Commit Batch Data ---
                if conv_log_upserts_dicts or person_updates:
                    logger.debug(
                        f"Attempting batch commit (Batch {current_batch_num}): {len(conv_log_upserts_dicts)} logs, {len(person_updates)} persons..."
                    )
                    # --- CALL NEW FUNCTION ---
                    logs_committed_count, persons_updated_count = commit_bulk_data(
                        session=session,
                        log_upserts=conv_log_upserts_dicts,  # Pass list of dicts
                        person_updates=person_updates,
                        context=f"Action 7 Batch {current_batch_num}",
                    )
                    # --- Update counters ---
                    status_updated_count += (
                        persons_updated_count  # Track person updates
                    )
                    logs_processed_in_run += (
                        logs_committed_count  # Track logs processed
                    )
                    # --- Clear lists ---
                    conv_log_upserts_dicts.clear()
                    person_updates.clear()
                    logger.debug(
                        f"Batch commit finished (Batch {current_batch_num}). Logs Processed: {logs_committed_count}, Persons Updated: {persons_updated_count}."
                    )

                # Step 2h: Check stop flag *after* potential commit
                if stop_processing:
                    if not stop_reason:
                        stop_reason = (
                            "Comparator Found"  # Set reason if not already set
                        )
                    break  # Break outer while loop

                # Step 2i: Prepare for next batch iteration
                next_cursor = next_cursor_from_api
                if not next_cursor:  # End of inbox reached
                    stop_reason = "End of Inbox Reached (No Next Cursor)"
                    stop_processing = True
                    logger.debug("No next cursor from API. Ending processing.")
                    break  # Break outer while loop

            # --- Step 3: Handle Exceptions During Batch Processing ---
            except WebDriverException as wde:
                error_count_this_loop += 1
                logger.error(
                    f"WebDriverException occurred during inbox loop (Batch {current_batch_num}): {wde}"
                )
                stop_reason = "WebDriver Exception"
                stop_processing = True  # Stop processing
                # Attempt final save before exiting loop
                logger.warning("Attempting final save due to WebDriverException...")
                # --- CALL NEW FUNCTION ---
                final_logs_saved, final_persons_updated = commit_bulk_data(
                    session=session,
                    log_upserts=conv_log_upserts_dicts,
                    person_updates=person_updates,
                    context="Action 7 Final Save (WebDriverException)",
                )
                status_updated_count += final_persons_updated
                logs_processed_in_run += final_logs_saved
                # Break loop after final save attempt
                break
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt detected during inbox loop.")
                stop_reason = "Keyboard Interrupt"
                stop_processing = True  # Stop processing
                # Attempt final save
                logger.warning("Attempting final save due to KeyboardInterrupt...")
                # --- CALL NEW FUNCTION ---
                final_logs_saved, final_persons_updated = commit_bulk_data(
                    session=session,
                    log_upserts=conv_log_upserts_dicts,
                    person_updates=person_updates,
                    context="Action 7 Final Save (Interrupt)",
                )
                status_updated_count += final_persons_updated
                logs_processed_in_run += final_logs_saved
                # Break loop after final save attempt
                break
            except Exception as e_main:
                error_count_this_loop += 1
                logger.critical(
                    f"Critical error in inbox processing loop (Batch {current_batch_num}): {e_main}",
                    exc_info=True,
                )
                stop_reason = f"Critical Error ({type(e_main).__name__})"
                stop_processing = True  # Stop processing
                # Attempt final save
                logger.warning("Attempting final save due to critical error...")
                # --- CALL NEW FUNCTION ---
                final_logs_saved, final_persons_updated = commit_bulk_data(
                    session=session,
                    log_upserts=conv_log_upserts_dicts,
                    person_updates=person_updates,
                    context="Action 7 Final Save (Critical Error)",
                )
                status_updated_count += final_persons_updated
                logs_processed_in_run += final_logs_saved
                # Return from helper to signal failure to outer function
                return (
                    stop_reason,
                    total_processed_api_items,
                    ai_classified_count,
                    status_updated_count,
                    items_processed_before_stop,
                )

        # --- End Main Loop (while not stop_processing) ---

        # Step 4: Perform final commit if loop finished normally or stopped early
        if (
            not stop_reason
            or stop_reason
            in (  # Only commit if loop ended somewhat gracefully
                "Comparator Found",
                "Comparator Found (No Change)",
                f"Inbox Limit ({self.max_inbox_limit})",
                "End of Inbox Reached (Empty Batch, No Cursor)",
                "End of Inbox Reached (No Next Cursor)",
            )
        ):
            if conv_log_upserts_dicts or person_updates:
                logger.info("Performing final commit at end of processing loop...")
                # --- CALL NEW FUNCTION ---
                final_logs_saved, final_persons_updated = commit_bulk_data(
                    session=session,
                    log_upserts=conv_log_upserts_dicts,
                    person_updates=person_updates,
                    context="Action 7 Final Save (Normal Exit)",
                )
                status_updated_count += final_persons_updated
                logs_processed_in_run += final_logs_saved

        # Step 5: Return results from the loop execution (modified tuple)
        return (
            stop_reason,
            total_processed_api_items,
            ai_classified_count,
            status_updated_count,
            items_processed_before_stop,
        )
    # End of _process_inbox_loop

    def _log_unified_summary(
        self,
        total_api_items: int,
        items_processed: int,
        new_logs: int,
        ai_classified: int,
        status_updates: int,
        stop_reason: Optional[str],
        max_inbox_limit: int,
    ):
        """Logs a unified summary of the inbox search process."""
        # Step 1: Print header
        print(" ")
        logger.info("------ Inbox Search Summary ------")
        # Step 2: Log key metrics
        logger.info(f"  API Conversations Fetched:    {total_api_items}")
        logger.info(f"  Conversations Processed:      {items_processed}")
        # logger.info(f"  New/Updated Log Entries:    {new_logs}") # Removed as upsert logic complicates exact counts
        logger.info(f"  AI Classifications Attempted: {ai_classified}")
        logger.info(f"  Person Status Updates Made:   {status_updates}")
        # Step 3: Log stopping reason
        final_reason = stop_reason
        if not stop_reason:
            # Infer reason if not explicitly set
            if max_inbox_limit == 0 or items_processed < max_inbox_limit:
                final_reason = "End of Inbox Reached or Comparator Match"
            else:
                final_reason = f"Inbox Limit ({max_inbox_limit}) Reached"
        logger.info(f"  Processing Stopped Due To:    {final_reason}")
        # Step 4: Print footer
        logger.info("----------------------------------\n")  # Add newline
    # End of _log_unified_summary


# End of InboxProcessor class


# End of action7_inbox.py
