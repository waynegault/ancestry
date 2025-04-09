#!/usr/bin/env python3

# action7_inbox.py


# Standard library imports
import enum
import inspect
import json
import logging
import math
import os
import random
import time
import traceback
<<<<<<< HEAD
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast, Set
=======
import sys
import tqdm
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
>>>>>>> parent of b5d69aa (afternoon3)
from urllib.parse import urljoin

# Third-party imports
import requests
<<<<<<< HEAD
=======
from tqdm.contrib.logging import logging_redirect_tqdm
>>>>>>> parent of b5d69aa (afternoon3)
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Integer,
    String,
    desc,
    func,
<<<<<<< HEAD
    over,
    update,
    text,
    select as sql_select,
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.orm import Session as DbSession, aliased, joinedload
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from selenium.common.exceptions import WebDriverException
=======
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql.elements import ColumnElement
>>>>>>> parent of b5d69aa (afternoon3)

# Local application imports
from config import config_instance
from database import (
    Person,
<<<<<<< HEAD
    ConversationLog,
    MessageType,
    db_transn,
    PersonStatusEnum,
    MessageDirectionEnum,
)  # Import Enums
=======
    RoleType,
    create_person,
    db_transn,
    get_person_by_profile_id_and_username,
)
>>>>>>> parent of b5d69aa (afternoon3)
from utils import (
    _api_req,
    DynamicRateLimiter,
    SessionManager,
    retry,
    time_wait,
    retry_api,
)
from ai_interface import classify_message_intent

# Initialize logging
logger = logging.getLogger("logger")


class InboxProcessor:
    """V1.19: Processes inbox, uses 2-row ConvLog, contextual AI, UPSERTS, batch commits, handles WebDriverExceptions, detailed commit logging."""

    def __init__(self, session_manager: SessionManager):
        """Initializes InboxProcessor."""
        self.session_manager = session_manager
        self.dynamic_rate_limiter = DynamicRateLimiter()
        self.max_inbox_limit = config_instance.MAX_INBOX
        self.default_batch_size = min(config_instance.BATCH_SIZE, 30)
        self.ai_context_msg_count = config_instance.AI_CONTEXT_MESSAGES_COUNT
        self.ai_context_max_words = config_instance.AI_CONTEXT_MESSAGE_MAX_WORDS
    # End of __init__

    @retry_api()
    def _get_all_conversations_api(
        self, session_manager: SessionManager, limit: int, cursor: Optional[str] = None
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Retrieves a single batch of conversation overviews with a specific limit."""
        if not session_manager or not session_manager.my_profile_id:
            return None, None
        if not session_manager.is_sess_valid():
            raise WebDriverException("Session invalid before overview API")
        my_profile_id = session_manager.my_profile_id
        api_base = urljoin(config_instance.BASE_URL, "/app-api/express/v2/")
        # logger.debug(f"API call using limit: {limit}") # Log the passed limit
        url = f"{api_base}conversations?q=user:{my_profile_id}&limit={limit}"
        if cursor:
            url += f"&cursor={cursor}"
        try:
            response_data = _api_req(
                url=url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
<<<<<<< HEAD
                use_csrf_token=False,
                api_description="Get Inbox Conversations",
=======
                use_csrf_token=False,  # GET request for conversations likely doesn't need CSRF
                api_description="Get Inbox Conversations",  # Add a description if needed for headers
>>>>>>> parent of b5d69aa (afternoon3)
            )
            if response_data is None:
                return None, None
            if not isinstance(response_data, dict):
                logger.error(
                    f"Unexpected API response format: Type {type(response_data)}."
                )
                return None, None
            conversations_data = response_data.get("conversations", [])
            all_conversations: List[Dict[str, Any]] = []
            if conversations_data:
                for conv_data in conversations_data:
                    info = self._extract_conversation_info(conv_data, my_profile_id)
                    if info:
                        all_conversations.append(info)
            forward_cursor = response_data.get("paging", {}).get("forward_cursor")
            return all_conversations, forward_cursor
        except WebDriverException as e:
            logger.error(f"WDExc during _get_all_conv_api: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _get_all_conv_api: {e}", exc_info=True)
            return None, None
    # End of _get_all_conversations_api

    def _extract_conversation_info(
        self, conv_data: Dict[str, Any], my_profile_id: str
    ) -> Optional[Dict[str, Any]]:
        """Extracts key info from conversation overview."""
        if not isinstance(conv_data, dict):
            return None
        conversation_id = str(conv_data.get("id"))
        last_message_data = conv_data.get("last_message", {})
        if not conversation_id or not isinstance(last_message_data, dict):
            return None
        last_msg_ts_unix = last_message_data.get("created")
        last_msg_ts_aware = None
        if isinstance(last_msg_ts_unix, (int, float)):
            try:
                min_ts = 0
                max_ts = 32503680000
                if min_ts <= last_msg_ts_unix <= max_ts:
                    last_msg_ts_aware = datetime.fromtimestamp(
                        last_msg_ts_unix, tz=timezone.utc
                    )
            except:
                pass
        username = "Unknown"
        profile_id = "UNKNOWN"
        other_member_found = False
        members = conv_data.get("members", [])
        my_pid_str = str(my_profile_id).lower() if my_profile_id else ""
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict):
                    continue
                member_user_id = member.get("user_id")
                member_user_id_str = (
                    str(member_user_id).lower() if member_user_id else ""
                )
                if member_user_id_str and member_user_id_str != my_pid_str:
                    profile_id = str(member_user_id).upper()
                    username = member.get("display_name", "Unknown")
                    other_member_found = True
                    break
        return {
            "conversation_id": conversation_id,
            "profile_id": profile_id,
            "username": username,
            "last_message_timestamp": last_msg_ts_aware,
        }
    # End of _extract_conversation_info

    @retry_api(max_retries=2)
    def _fetch_conversation_context(
        self, conversation_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetches the last N messages (context) for a conversation."""
        if not conversation_id:
            return None
        if not self.session_manager or not self.session_manager.my_profile_id:
            return None
        if not self.session_manager.is_sess_valid():
            raise WebDriverException(
                f"Session invalid fetching context conv {conversation_id}"
            )
        context_messages: List[Dict[str, Any]] = []
        api_base = urljoin(config_instance.BASE_URL, "/app-api/express/v2/")
        limit = self.ai_context_msg_count
        api_description = "Fetch Conversation Context"
        headers = {
            "accept": "*/*",
            "ancestry-clientpath": "express-fe",
            "referer": urljoin(config_instance.BASE_URL, "/messaging/"),
        }
        if self.session_manager.my_profile_id:
            headers["ancestry-userid"] = self.session_manager.my_profile_id.upper()
        url = f"{api_base}conversations/{conversation_id}/messages?limit={limit}"
        try:
            wait_time = self.dynamic_rate_limiter.wait()
            response_data = _api_req(
                url=url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                headers=headers,
                use_csrf_token=False,
                api_description=api_description,
            )
            if not isinstance(response_data, dict):
                logger.warning(
                    f"{api_description}: Bad response {type(response_data)} conv {conversation_id}."
                )
                return None
            messages_batch = response_data.get("messages", [])
            if not isinstance(messages_batch, list):
                logger.warning(
                    f"{api_description}: 'messages' not list conv {conversation_id}."
                )
                return None
            for msg_data in messages_batch:
                if not isinstance(msg_data, dict):
                    continue
                ts_unix = msg_data.get("created")
                msg_timestamp = None
                if isinstance(ts_unix, (int, float)):
                    try:
                        msg_timestamp = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
                    except:
                        pass
                processed_msg = {
                    "content": str(msg_data.get("content", "")),
                    "author": str(msg_data.get("author", "")).lower(),
                    "timestamp": msg_timestamp,
                    "conversation_id": conversation_id,
                }
                context_messages.append(processed_msg)
            return sorted(
                context_messages,
                key=lambda x: x["timestamp"]
                or datetime.min.replace(tzinfo=timezone.utc),
            )
        except WebDriverException as e:
            logger.error(f"WDExc fetch context conv {conversation_id}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Error fetch context conv {conversation_id}: {e}", exc_info=True
            )
            return None
    # End of _fetch_conversation_context

    def search_inbox(self) -> bool:
<<<<<<< HEAD
        """V1.23 REVISED: Correctly limits API fetch size based on MAX_INBOX alongside Comparator logic, 2-row Log, AI, UPSERTS, batch commits, TZ-aware compare."""
        # --- Initialization ---
        ai_classified_count = 0
        status_updated_count = 0
=======
        """
        V1.7 REVISED: Searches inbox using cursor pagination and batch processing via API.
        - Removed tqdm progress bar and logging_redirect_tqdm due to display issues.
        """
        # --- Initialize counters ---
        known_conversation_found = False
        new_records_saved = 0
        updated_records_saved = 0
>>>>>>> parent of b5d69aa (afternoon3)
        total_processed_api_items = 0
        items_processed_before_stop = 0
        self.max_inbox_limit = config_instance.MAX_INBOX # Reinstated MAX_INBOX limit
        stop_reason = ""
        next_cursor: Optional[str] = None
        current_batch_num = 0
        conv_log_upserts: List[Dict[str, Any]] = []
        person_updates: Dict[int, Dict[str, Any]] = {}
        stop_processing = False # Flag to stop after comparator OR limit

        if not self.session_manager or not self.session_manager.my_profile_id:
            logger.error("Session manager or profile ID missing.")
            return False
        my_pid_lower = self.session_manager.my_profile_id.lower()

        session = None
        try:
            session = self.session_manager.get_db_conn()
            if not session:
                raise SQLAlchemyError("Failed to get DB session.")

            # --- Get Comparator ---
            comparator_info = self._create_comparator(session)
            comp_conv_id: Optional[str] = None
            comp_ts: Optional[datetime] = None
            if comparator_info:
                comp_conv_id = comparator_info.get("conversation_id")
                comp_ts = comparator_info.get("latest_timestamp") # Already aware UTC
            # --- End Get Comparator ---

            logger.info(f"Starting inbox search (Comparator Logic, MAX_INBOX={self.max_inbox_limit}, 2-Row Log, Contextual AI)...")

            while not stop_processing: # Loop until comparator found, limit reached, or end of API
                try:  # Inner try for batch processing
                    if not self.session_manager.is_sess_valid():
                        raise WebDriverException("Session invalid before overview batch")

                    # --- MODIFIED: Calculate Limit for this API call ---
                    current_limit = self.default_batch_size # Start with default
                    if self.max_inbox_limit > 0:
                        remaining_allowed = self.max_inbox_limit - items_processed_before_stop
                        if remaining_allowed <= 0:
                            # This check might be redundant due to the loop break, but safe to keep
                            stop_reason = f"Inbox Limit ({self.max_inbox_limit}) Pre-Fetch"
                            stop_processing = True
                            break # Break inner try loop
                        # Adjust limit to fetch only what's needed (up to batch size)
                        current_limit = min(self.default_batch_size, remaining_allowed)
                        logger.debug(f"MAX_INBOX active. Calculated remaining={remaining_allowed}, API limit for this batch: {current_limit}")
                    else:
                         logger.debug(f"MAX_INBOX inactive (0). Using default API limit: {current_limit}")
                    # --- END MODIFICATION ---


                    all_conversations_batch, next_cursor_from_api = (
                        self._get_all_conversations_api(
                            self.session_manager,
                            limit=current_limit, # Use the calculated limit
                            cursor=next_cursor,
                        )
                    )

                    if all_conversations_batch is None:
                        stop_reason = "API Error Fetching Batch"
                        stop_processing = True
                        break

                    batch_api_item_count = len(all_conversations_batch)
                    # IMPORTANT: Increment total_processed_api_items *before* checking the count
                    # This reflects what the API *actually returned* for this call
                    total_processed_api_items += batch_api_item_count

                    if batch_api_item_count == 0:
                        # Handle case where API returns empty list but maybe a cursor
                        if not next_cursor_from_api:
                             stop_reason = "End of Inbox Reached (Empty Batch, No Cursor)"
                             stop_processing = True
                        else:
                             # Got a cursor but no items? Weird, but continue loop.
                             logger.debug("API returned empty batch but provided a cursor. Continuing fetch.")
                             next_cursor = next_cursor_from_api
                             continue # Skip processing, go to next API call

                    # Apply rate limiter delay if items were fetched
                    wait_duration = self.dynamic_rate_limiter.wait()
                    if wait_duration > 0.1: logger.debug(f"API batch wait: {wait_duration:.2f}s")

                    current_batch_num += 1
<<<<<<< HEAD

                    # Pre-fetch Persons and Logs
                    # ...(Pre-fetching logic unchanged)...
                    batch_conv_ids = [c["conversation_id"] for c in all_conversations_batch if c.get("conversation_id")]
                    batch_profile_ids = {c.get("profile_id", "").upper() for c in all_conversations_batch if c.get("profile_id") and c.get("profile_id") != "UNKNOWN"}
                    existing_persons_map: Dict[str, Person] = {}
                    existing_conv_logs: Dict[Tuple[str, str], ConversationLog] = {}
                    if batch_profile_ids:
                        try:
                            persons = session.query(Person).filter(Person.profile_id.in_(batch_profile_ids)).all()
                            existing_persons_map = {p.profile_id: p for p in persons if p.profile_id}
                        except SQLAlchemyError as db_err: logger.error(f"Bulk Person lookup failed: {db_err}")
                    if batch_conv_ids:
                        try:
                            logs = session.query(ConversationLog).filter(ConversationLog.conversation_id.in_(batch_conv_ids)).all()
                            existing_conv_logs = {(log.conversation_id, log.direction.name): log for log in logs if log.direction}
                        except SQLAlchemyError as db_err: logger.error(f"ConvLog lookup failed: {db_err}")

=======
                    batch_data_to_save = []
>>>>>>> parent of b5d69aa (afternoon3)

                    # Process Batch
                    for conversation_info in all_conversations_batch:
                        # --- Check MAX_INBOX Limit ---
                        # This check is now more of a safety fallback, as the API limit calculation should prevent exceeding it.
                        if self.max_inbox_limit > 0 and items_processed_before_stop >= self.max_inbox_limit:
                            if not stop_reason: stop_reason = f"Inbox Limit ({self.max_inbox_limit})" # Set reason if not already set
                            stop_processing = True
                            break # Break inner loop (for conversation_info...)
                        # --- End MAX_INBOX Check ---

<<<<<<< HEAD
                        items_processed_before_stop += 1 # Increment only if not stopped by limit

                        profile_id_upper = conversation_info.get("profile_id", "UNKNOWN").upper()
                        api_conv_id = conversation_info.get("conversation_id")
                        api_latest_ts = conversation_info.get("last_message_timestamp")

                        if not api_conv_id or profile_id_upper == "UNKNOWN":
                            logger.debug(f"Skipping item {items_processed_before_stop}: Invalid ConvID or ProfileID.")
                            continue

                        needs_fetch = False
                        # Comparator Logic
                        if comp_conv_id and api_conv_id == comp_conv_id:
                            logger.debug(f"Comparator conversation {comp_conv_id} found in API results.")
                            stop_processing = True
                            if comp_ts and api_latest_ts and api_latest_ts > comp_ts:
                                needs_fetch = True; logger.debug("Comparator needs update.")
                            else: logger.debug("Comparator does not need update.")
                        else:
                            # Check this specific conversation against its own history
                            # ...(logic unchanged)...
                            db_log_in = existing_conv_logs.get((api_conv_id, "IN"))
                            db_log_out = existing_conv_logs.get((api_conv_id, "OUT"))
                            min_aware_dt = datetime.min.replace(tzinfo=timezone.utc)
                            db_latest_ts_in = min_aware_dt
                            if db_log_in and db_log_in.latest_timestamp:
                                ts_in = db_log_in.latest_timestamp
                                if isinstance(ts_in, datetime): db_latest_ts_in = ts_in.replace(tzinfo=timezone.utc) if ts_in.tzinfo is None else ts_in.astimezone(timezone.utc)
                            db_latest_ts_out = min_aware_dt
                            if db_log_out and db_log_out.latest_timestamp:
                                ts_out = db_log_out.latest_timestamp
                                if isinstance(ts_out, datetime): db_latest_ts_out = ts_out.replace(tzinfo=timezone.utc) if ts_out.tzinfo is None else ts_out.astimezone(timezone.utc)
                            db_latest_overall_for_conv = max(db_latest_ts_in, db_latest_ts_out)
                            if api_latest_ts and api_latest_ts > db_latest_overall_for_conv: needs_fetch = True
                            elif not db_log_in and not db_log_out: needs_fetch = True


                        if not needs_fetch:
                            if stop_processing: logger.debug("Comparator found and no fetch needed, breaking inner loop."); break
                            continue

                        # Fetch Context & Process (if needs_fetch is True)
                        # ...(logic unchanged)...
                        if not self.session_manager.is_sess_valid(): raise WebDriverException(f"Session invalid fetch context conv {api_conv_id}")
                        context_messages = self._fetch_conversation_context(api_conv_id)
                        if context_messages is None: logger.error(f"Failed context fetch conv {api_conv_id}. Skipping item."); continue

                        person, person_status_flag = self._lookup_or_create_person(session, profile_id_upper, conversation_info.get("username", "Unknown"), api_conv_id, existing_person_arg=existing_persons_map.get(profile_id_upper))
                        if not person or not person.id: logger.error(f"Failed person lookup/create conv {api_conv_id}. Skipping item."); continue
                        people_id = person.id

                        latest_ctx_in: Optional[Dict] = None; latest_ctx_out: Optional[Dict] = None
                        for msg in reversed(context_messages):
                            author_lower = msg.get("author", "");
                            if author_lower != my_pid_lower and latest_ctx_in is None: latest_ctx_in = msg
                            elif author_lower == my_pid_lower and latest_ctx_out is None: latest_ctx_out = msg
                            if latest_ctx_in and latest_ctx_out: break

                        ai_sentiment_result: Optional[str] = None
                        # Process IN Row
                        if latest_ctx_in and latest_ctx_in.get("timestamp"):
                           ctx_ts_in_aware = latest_ctx_in.get("timestamp")
                           if ctx_ts_in_aware and ctx_ts_in_aware > db_latest_ts_in:
                                formatted_context = self._format_context_for_ai(context_messages, my_pid_lower)
                                if not self.session_manager.is_sess_valid(): raise WebDriverException(f"Session invalid AI call conv {api_conv_id}")
                                ai_sentiment_result = classify_message_intent(formatted_context, self.session_manager)
                                ai_classified_count += 1
                                if ai_sentiment_result in ("DESIST", "UNINTERESTED"):
                                    logger.info(f"AI Classified conv {api_conv_id} (PID {people_id}) as '{ai_sentiment_result}'. Marking for 'desist'.")
                                    if people_id not in person_updates: person_updates[people_id] = {}
                                    person_updates[people_id]["status"] = PersonStatusEnum.DESIST
                                upsert_data_in = { "conversation_id": api_conv_id, "direction": "IN", "people_id": people_id, "latest_message_content": latest_ctx_in.get("content", "")[:config_instance.MESSAGE_TRUNCATION_LENGTH], "latest_timestamp": ctx_ts_in_aware, "ai_sentiment": ai_sentiment_result, "updated_at": datetime.now(timezone.utc) }
                                upsert_data_in = {k: v for k, v in upsert_data_in.items() if v is not None or k == "ai_sentiment"}
                                conv_log_upserts.append(upsert_data_in)
                        # Process OUT Row
                        if latest_ctx_out and latest_ctx_out.get("timestamp"):
                           ctx_ts_out_aware = latest_ctx_out.get("timestamp")
                           if ctx_ts_out_aware and ctx_ts_out_aware > db_latest_ts_out:
                                upsert_data_out = { "conversation_id": api_conv_id, "direction": "OUT", "people_id": people_id, "latest_message_content": latest_ctx_out.get("content", "")[:config_instance.MESSAGE_TRUNCATION_LENGTH], "latest_timestamp": ctx_ts_out_aware, "message_type_id": None, "script_message_status": None, "ai_sentiment": None, "updated_at": datetime.now(timezone.utc) }
                                upsert_data_out = {k: v for k, v in upsert_data_out.items() if v is not None or k in ["message_type_id", "script_message_status"]}
                                conv_log_upserts.append(upsert_data_out)

                        if stop_processing: # Break inner loop if comparator/limit reached
                            logger.debug(f"Stop flag set ({stop_reason}), breaking inner loop.")
                            break
                    # --- End Inner Loop (for conversation_info...) ---

                    # Commit batch data
                    if conv_log_upserts or person_updates:
                        logger.info(f"Attempting batch commit (Batch {current_batch_num}): {len(conv_log_upserts)} logs, {len(person_updates)} persons...")
                        status_updates_this_batch = self._commit_batch_data_upsert(session, conv_log_upserts, person_updates)
                        status_updated_count += status_updates_this_batch
                        conv_log_upserts.clear()
                        person_updates.clear()
                        logger.info(f"Batch commit attempt finished (Batch {current_batch_num}). Updated {status_updates_this_batch} persons.")

                    # Check stop flag *after* commit and *before* getting next cursor
                    if stop_processing:
                        if not stop_reason: stop_reason = "Comparator Found" # Set default reason if needed
                        logger.debug(f"Stop flag set ({stop_reason}). Breaking outer loop.")
                        break # Break outer loop (while not stop_processing)
=======
                        # Comparator Check
                        if most_recent_message:
                            profile_id = conversation_info.get("profile_id")
                            username = conversation_info.get(
                                "username", "Username Not Available"
                            )
                            last_message_timestamp = conversation_info.get(
                                "last_message_timestamp"
                            )
                            comparator_profile_id = most_recent_message.get(
                                "profile_id"
                            )
                            comparator_username = most_recent_message.get("username")
                            comparator_timestamp = most_recent_message.get(
                                "last_message_timestamp"
                            )
                            profile_id_match = (
                                comparator_profile_id
                                and profile_id
                                and comparator_profile_id.lower() == profile_id.lower()
                            )
                            username_match = comparator_username == username
                            timestamps_match = False
                            if isinstance(
                                comparator_timestamp, datetime
                            ) and isinstance(last_message_timestamp, datetime):
                                comp_ts_naive = (
                                    comparator_timestamp.replace(tzinfo=None)
                                    if comparator_timestamp.tzinfo
                                    else comparator_timestamp
                                )
                                item_ts_naive = (
                                    last_message_timestamp.replace(tzinfo=None)
                                    if last_message_timestamp.tzinfo
                                    else last_message_timestamp
                                )
                                time_diff = abs(
                                    (comp_ts_naive - item_ts_naive).total_seconds()
                                )
                                timestamps_match = time_diff < 1
                            elif (
                                comparator_timestamp is None
                                and last_message_timestamp is None
                            ):
                                timestamps_match = True
                            if profile_id_match and username_match and timestamps_match:
                                logger.info(
                                    f"Comparator ({username}) found at index {items_processed_before_stop - 1}.\nStopping further processing.\n"
                                )
                                known_conversation_found = True
                                stop_reason = "Comparator Match"
                                break

                        # Limit Check
                        if (
                            self.max_inbox_limit != 0
                            and items_processed_before_stop > self.max_inbox_limit
                        ):
                            logger.info(
                                f"Inbox limit ({self.max_inbox_limit} items) reached processing item index {items_processed_before_stop - 1}. Stopping."
                            )
                            items_processed_before_stop -= (
                                1  # Correct count as this one wasn't fully processed
                            )
                            known_conversation_found = True
                            stop_reason = f"Inbox Limit ({self.max_inbox_limit} items)"
                            break

                        # Process Item (If checks passed)
                        # Log progress periodically instead of using progress bar
                        if items_processed_before_stop % 50 == 0:
                            logger.info(
                                f"Processed {items_processed_before_stop} items..."
                            )

                        profile_id = conversation_info.get("profile_id")
                        username = conversation_info.get(
                            "username", "Username Not Available"
                        )
                        conversation_id = conversation_info.get("conversation_id")

                        if profile_id is None or profile_id == "UNKNOWN":
                            logger.warning(
                                f"Profile ID is missing/unknown for conv ID: {conversation_id}. Skipping item index {items_processed_before_stop - 1}."
                            )
                            continue

                        person, person_status = self._lookup_or_create_person(
                            session, profile_id, username, conversation_id
                        )
                        if not person or person.id is None:
                            logger.error(
                                f"Failed create/get Person/ID for {profile_id} in conv {conversation_id} (item index {items_processed_before_stop - 1}). Skipping save for this item."
                            )
                            continue

                        people_id = person.id
                        last_message_content = conversation_info.get(
                            "last_message_content", ""
                        )
                        last_message_content_truncated = (
                            (last_message_content[:97] + "...")
                            if len(last_message_content) > 100
                            else last_message_content
                        )
                        my_role_enum_value = conversation_info.get("my_role")
                        last_message_timestamp = conversation_info.get(
                            "last_message_timestamp"
                        )

                        processed_item = {
                            "conversation_id": conversation_id,
                            "people_id": people_id,
                            "my_role": my_role_enum_value,
                            "last_message_content": last_message_content_truncated,
                            "last_message_timestamp": last_message_timestamp,
                        }
                        batch_data_to_save.append(processed_item)
                    # --- End of inner for loop ---

                    # --- Save Batch ---
                    if (
                        batch_data_to_save
                    ):  # Save only if there's data AND stop condition wasn't met *before* saving
                        if not known_conversation_found:
                            new_in_batch, updated_in_batch = self._save_batch(
                                session, batch_data_to_save
                            )
                            new_records_saved += new_in_batch
                            updated_records_saved += updated_in_batch
                            logger.debug(
                                f"Batch saved: New={new_in_batch}, Updated={updated_in_batch}"
                            )
                        else:
                            # This case might happen if comparator match occurred mid-batch
                            logger.debug(
                                "Stop condition met during batch processing. Not saving remaining items in this batch."
                            )
                    elif known_conversation_found:
                        logger.debug(
                            "Stop condition met before processing any data in this batch."
                        )
                    else:  # No data and not stopped - likely an empty API response handled earlier
                        logger.debug("No data prepared in batch to save.")
>>>>>>> parent of b5d69aa (afternoon3)

                    next_cursor = next_cursor_from_api
                    if not next_cursor:
<<<<<<< HEAD
=======
                        logger.info("No next cursor from API. Finishing inbox search.")
>>>>>>> parent of b5d69aa (afternoon3)
                        stop_reason = "End of Inbox Reached"
                        stop_processing = True
                        logger.debug("No next cursor from API. Breaking inner try block.")
                        break # Break inner try

                # -- End Try block for batch processing --
                # ...(Exception handling unchanged)...
                except WebDriverException as WDE:
                    logger.error(f"WebDriverException occurred: {WDE}")
                    stop_reason = "WebDriver Exception"; stop_processing = True
                    logger.warning("Attempting save and restart..."); save_count = self._commit_batch_data_upsert(session, conv_log_upserts, person_updates, is_final_attempt=True)
                    status_updated_count += save_count; conv_log_upserts.clear() ; person_updates.clear()
                    if self.session_manager.restart_sess():
                        logger.info("Session restarted. Retrying inbox search..."); session = self.session_manager.get_db_conn()
                        if not session: raise SQLAlchemyError("Failed to get DB session after restart.")
                        stop_processing = False; next_cursor = None; current_batch_num = 0
                        continue
                    else: logger.critical("Session restart failed."); return False
                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt detected."); stop_reason = "Keyboard Interrupt"; stop_processing = True
                    logger.warning("Attempting final save..."); save_count = self._commit_batch_data_upsert(session, conv_log_upserts, person_updates, is_final_attempt=True)
                    status_updated_count += save_count; break
                except Exception as e_main:
                    logger.critical(f"Critical error in search_inbox loop: {e_main}", exc_info=True)
                    stop_reason = f"Critical Error ({type(e_main).__name__})"; stop_processing = True
                    logger.warning("Attempting final save..."); save_count = self._commit_batch_data_upsert(session, conv_log_upserts, person_updates, is_final_attempt=True)
                    status_updated_count += save_count; return False

            # --- End main while loop ---

        except Exception as outer_e:
            logger.error(f"Outer error in search_inbox: {outer_e}", exc_info=True)
            return False

        finally:
<<<<<<< HEAD
            # Final commit attempt
            if session and (conv_log_upserts or person_updates):
                logger.warning("Performing final commit outside loop.")
                final_save_count = self._commit_batch_data_upsert(session, conv_log_upserts, person_updates, is_final_attempt=True)
                status_updated_count += final_save_count
            # Final summary log
            if stop_reason and not stop_reason.startswith("End of Inbox") and not stop_reason.startswith("Comparator"):
                 logger.warning(f"Inbox search stopped early: {stop_reason}")
            # Pass actual MAX_INBOX limit to summary function
            self._log_unified_summary(
                total_api_items=total_processed_api_items,
                items_processed=items_processed_before_stop,
                new_logs=0,
                ai_classified=ai_classified_count,
                status_updates=status_updated_count,
                stop_reason=stop_reason or "Comparator Found or End of Inbox",
                max_inbox_limit=self.max_inbox_limit, # Pass the actual limit value
=======
            # --- Cleaned Finally Block (No Progress Bar) ---
            # 1. Determine final stop reason if needed
            if known_conversation_found and not stop_reason:
                stop_reason = "Unknown/Interrupted"  # e.g., Ctrl+C

            # 2. Log the summary
            _log_inbox_summary(
                total_api_items=total_processed_api_items,
                items_processed=items_processed_before_stop,  # Use the final count
                new_records=new_records_saved,
                updated_records=updated_records_saved,
                stop_reason=stop_reason,
                max_inbox_limit=self.max_inbox_limit,
>>>>>>> parent of b5d69aa (afternoon3)
            )
            # Release session
            if session:
                self.session_manager.return_session(session)
        return True
    # End of search_inbox

    def _format_context_for_ai(
        self, context_messages: List[Dict], my_pid_lower: str
    ) -> str:
        """Formats the last N messages for the AI prompt."""
        context_lines = []
        for (
            msg
        ) in (
            context_messages
        ):  # Assumes context_messages is already last N, sorted OLD->NEW
            label = "USER: " if msg.get("author", "") != my_pid_lower else "SCRIPT: "
            content = msg.get("content", "")
            truncated_content = " ".join(content.split()[: self.ai_context_max_words])
            if len(content.split()) > self.ai_context_max_words:
                truncated_content += "..."
            context_lines.append(f"{label}{truncated_content}")
        return "\n".join(context_lines)
    # End of _format_context_for_ai

    def _commit_batch_data_upsert(
        self,
        session: DbSession,
        log_upserts: List[Dict],
        person_updates: Dict[int, Dict],
        is_final_attempt: bool = False,
    ) -> int:
        """
        V1.20 REVISED: Helper function to UPSERT ConversationLog entries and update Person statuses.
        - Uses explicit query/update/add for ConversationLog.
        - Uses bulk_update_mappings for Person status.
        - Adds session.flush() inside the loop for logs.
        - Logs session state before commit attempt.
        """
        updated_person_count = 0
        if not log_upserts and not person_updates:
            return updated_person_count
        log_prefix = "[Final Save] " if is_final_attempt else "[Batch Save] "
        logger.info(
            f"{log_prefix} Preparing commit: {len(log_upserts)} logs, {len(person_updates)} person updates."
        )

        # --- Log Data Before Commit ---
        # (Logging kept the same)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{log_prefix}--- Log Upsert Data ---")
            for item in log_upserts:
                logger.debug(
                    f"  { {k: (str(v)[:50] + '...' if k=='latest_message_content' else v) for k, v in item.items()} }"
                )
            logger.debug(f"{log_prefix}--- Person Update Data ---")
            for pid, data in person_updates.items():
                logger.debug(f"  PersonID {pid}: {data}")
            logger.debug(f"{log_prefix}--- End Data ---")
        # --- End Log Data ---

        try:
            with db_transn(session):  # Use transaction context manager
                logger.debug(f"{log_prefix}Entered transaction block.")

                # --- ConversationLog Upsert Logic ---
                if log_upserts:
                    processed_logs_count = 0
                    for data in log_upserts:
                        try:  # Add inner try/except for individual log processing
                            conv_id = data.get("conversation_id")
                            direction_val = data.get("direction")

                            # Convert direction string/enum to Enum
                            direction_enum = None
                            if isinstance(direction_val, MessageDirectionEnum):
                                direction_enum = direction_val
                            elif isinstance(direction_val, str):
                                try:
                                    direction_enum = MessageDirectionEnum[direction_val]
                                except KeyError:
                                    logger.error(
                                        f"{log_prefix}Invalid direction string '{direction_val}' for conv {conv_id}. Skipping."
                                    )
                                    continue
                            else:
                                logger.error(
                                    f"{log_prefix}Invalid/missing direction value for conv {conv_id}. Skipping."
                                )
                                continue

                            if not conv_id:
                                logger.error(
                                    f"{log_prefix}Missing conversation_id. Skipping log entry: {data}"
                                )
                                continue

                            # Ensure timestamp is aware
                            ts_val = data.get("latest_timestamp")
                            aware_timestamp = None
                            if isinstance(ts_val, datetime):
                                aware_timestamp = (
                                    ts_val.astimezone(timezone.utc)
                                    if ts_val.tzinfo
                                    else ts_val.replace(tzinfo=timezone.utc)
                                )
                            elif ts_val is not None:
                                logger.error(
                                    f"{log_prefix} Invalid timestamp type '{type(ts_val)}' for conv {conv_id}. Skipping."
                                )
                                continue

                            # Query for existing log entry
                            existing_log = (
                                session.query(ConversationLog)
                                .filter_by(
                                    conversation_id=conv_id, direction=direction_enum
                                )
                                .with_for_update()  # Add locking for update safety
                                .first()
                            )

                            # Prepare update/insert data
                            log_data_for_op = {
                                "people_id": data.get("people_id"),
                                "latest_message_content": data.get(
                                    "latest_message_content"
                                ),
                                "latest_timestamp": aware_timestamp,
                                "ai_sentiment": data.get("ai_sentiment"),
                                "message_type_id": data.get("message_type_id"),
                                "script_message_status": data.get(
                                    "script_message_status"
                                ),
                                "updated_at": datetime.now(timezone.utc),
                            }

                            if existing_log:
                                # Update existing record
                                # logger.debug(f"{log_prefix}Updating ConvLog: conv {conv_id}, dir {direction_enum.name}") # Verbose
                                for key, value in log_data_for_op.items():
                                    setattr(existing_log, key, value)
                            else:
                                # Insert new record
                                # logger.debug(f"{log_prefix}Adding new ConvLog: conv {conv_id}, dir {direction_enum.name}") # Verbose
                                log_data_for_op["conversation_id"] = conv_id
                                log_data_for_op["direction"] = direction_enum
                                new_log_obj = ConversationLog(**log_data_for_op)
                                session.add(new_log_obj)

                            # --- MODIFICATION: Flush after each log item ---
                            session.flush()
                            # logger.debug(f"{log_prefix} Flushed item for conv {conv_id}, dir {direction_enum.name}") # Verbose
                            # --- END MODIFICATION ---

                            processed_logs_count += 1

                        except Exception as inner_loop_exc:
                            # Log error for this specific item but continue the loop
                            logger.error(
                                f"{log_prefix} Error processing single log item (conv={data.get('conversation_id')}, dir={data.get('direction')}): {inner_loop_exc}",
                                exc_info=True,
                            )
                            # Optionally rollback this specific item's changes? Tricky within the larger transaction.
                            # For now, just log and continue the batch.

                    logger.debug(
                        f"{log_prefix}Processed {processed_logs_count} log entries (with flush after each)."
                    )
                # --- End ConversationLog Upsert Logic ---

                # --- Person Update Logic (Unchanged) ---
                if person_updates:
                    pids_to_update = list(person_updates.keys())
                    update_values = []
                    logger.debug(
                        f"{log_prefix}Preparing {len(pids_to_update)} Person status updates..."
                    )  # Debug level
                    for pid, data in person_updates.items():
                        status_val = data.get("status")
                        enum_value = None
                        if isinstance(status_val, PersonStatusEnum):
                            enum_value = status_val
                        elif isinstance(status_val, str):
                            try:
                                enum_value = PersonStatusEnum(status_val)
                            except ValueError:
                                logger.warning(
                                    f"Invalid status string '{status_val}' for Person ID {pid}. Skipping."
                                )
                                continue
                        else:
                            logger.warning(
                                f"Invalid status type for Person ID {pid}: {type(status_val)}. Skipping."
                            )
                            continue
                        logger.debug(
                            f"  Preparing update for Person ID {pid}: status -> {enum_value.name}"
                        )
                        update_values.append(
                            {
                                "id": pid,
                                "status": enum_value,
                                "updated_at": datetime.now(timezone.utc),
                            }
                        )

                    if update_values:
                        logger.info(
                            f"{log_prefix}Attempting bulk update mappings for {len(update_values)} persons statuses..."
                        )  # Info level
                        session.bulk_update_mappings(Person, update_values)
                        updated_person_count = len(update_values)
                        logger.info(
                            f"{log_prefix}Bulk update mappings called for {updated_person_count} persons."
                        )
                    else:
                        logger.warning(f"{log_prefix}No valid person updates prepared.")
                # --- End Person Update Logic ---

                # --- Log Session State Before Commit ---
                logger.debug(
                    f"{log_prefix}Session state before final commit: Dirty={session.dirty}, New={session.new}, Deleted={session.deleted}"
                )
                # --- End Log Session State ---

                logger.debug(
                    f"{log_prefix}Exiting transaction block (Final commit attempt follows)."
                )
            # --- Log success *after* the 'with db_transn' block ---
            logger.info(
                f"{log_prefix}Commit successful. Updated {updated_person_count} persons."
            )  # INFO level
            return updated_person_count
        except Exception as commit_err:
            # db_transn handles rollback, just log error here
            logger.error(
                f"{log_prefix}Commit FAILED inside helper: {commit_err}", exc_info=True
            )
            return 0
    # End of _commit_batch_data_upsert

    def _create_comparator(self, session: Session) -> Optional[Dict[str, Any]]:
        """
        V1.21 REVISED: Finds the ConversationLog entry with the latest timestamp.

        Returns:
            A dictionary {'conversation_id': str, 'latest_timestamp': datetime}
            or None if the ConversationLog table is empty.
        """
        latest_log_entry_info = None
        try:
<<<<<<< HEAD
            # Find the log entry with the maximum timestamp
            latest_entry = (
                session.query(
                    ConversationLog.conversation_id,
                    ConversationLog.latest_timestamp
                )
                .order_by(ConversationLog.latest_timestamp.desc().nullslast())
                .first()
            )

            if latest_entry:
                # Ensure the timestamp is timezone-aware (assuming UTC)
                log_timestamp = latest_entry.latest_timestamp
                aware_timestamp = None
                if isinstance(log_timestamp, datetime):
                     aware_timestamp = log_timestamp.replace(tzinfo=timezone.utc) if log_timestamp.tzinfo is None else log_timestamp.astimezone(timezone.utc)
=======
            # Query optimized: Order by timestamp DESC (nulls last), then ID DESC
            comparator_inbox_status = (
                session.query(InboxStatus)
                .order_by(
                    InboxStatus.last_message_timestamp.desc().nullslast(),
                    InboxStatus.id.desc(),
                )
                # Eager load the associated Person to avoid a separate query later
                .options(joinedload(InboxStatus.person))
                .first()
            )

            if comparator_inbox_status:
                # Access the eager-loaded person object
                comparator_person = comparator_inbox_status.person

                if (
                    comparator_person
                    and comparator_person.id is not None
                    and comparator_person.profile_id is not None
                ):  # Ensure person, ID, and profile_id are valid
                    most_recent_message = {
                        "people_id": comparator_person.id,  # Still needed for potential DB ops if match fails
                        "profile_id": comparator_person.profile_id,  # ADDED: Store profile_id
                        "username": comparator_person.username,  # Use username from person record
                        "last_message_timestamp": comparator_inbox_status.last_message_timestamp,  # datetime or None
                    }
                    # Format timestamp safely for logging
                    ts_str = "None"
                    timestamp_val = most_recent_message.get("last_message_timestamp")
                    if isinstance(timestamp_val, datetime):
                        try:
                            # Use ISO format for clarity and include timezone info if available (UTC assumed here)
                            ts_str = timestamp_val.isoformat() + "Z"  # Indicate UTC
                        except ValueError:
                            logger.warning(
                                f"Comparator timestamp {timestamp_val} likely out of range for ISO format."
                            )
                            ts_str = str(timestamp_val)  # Use string representation
                    elif timestamp_val is not None:
                        ts_str = str(timestamp_val)

                    # Log comparator details at INFO level for better visibility
                    logger.debug(
                        f"Comparator created: {most_recent_message.get('username', 'N/A')} (Profile: {most_recent_message.get('profile_id')})"
                    )  # Added profile_id to log
                elif comparator_person and comparator_person.profile_id is None:
                    logger.warning(
                        f"Comparator error: Found Person object (ID: {comparator_person.id}) for InboxStatus ID {comparator_inbox_status.id}, but Person has no profile_id."
                    )
                elif comparator_person and comparator_person.id is None:
                    logger.warning(
                        f"Comparator error: Found Person object for InboxStatus ID {comparator_inbox_status.id}, but Person has no ID."
                    )
                elif not comparator_person:
                    logger.warning(
                        f"Comparator error: InboxStatus record found (ID: {comparator_inbox_status.id}), but associated Person (people_id: {comparator_inbox_status.people_id}) could not be loaded/found."
                    )

            else:
                logger.info("No messages in database. Comparator not needed.\n")

        except SQLAlchemyError as e:
            logger.error(f"Database error creating comparator: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating comparator: {e}", exc_info=True)
            return None

        return most_recent_message
>>>>>>> parent of b5d69aa (afternoon3)

                if latest_entry.conversation_id and aware_timestamp:
                    latest_log_entry_info = {
                        "conversation_id": latest_entry.conversation_id,
                        "latest_timestamp": aware_timestamp,
                    }
                    logger.debug(f"Comparator created: ConvID={latest_log_entry_info['conversation_id']}, TS={latest_log_entry_info['latest_timestamp']}")
                else:
                    logger.warning(f"Found latest log entry, but data invalid: ConvID={latest_entry.conversation_id}, TS={log_timestamp}")

            else:
                 logger.info("ConversationLog empty. Comparator not created.") # Use INFO

        except Exception as e:
            logger.error(f"Error creating comparator: {e}", exc_info=True)
            return None # Return None on error
        return latest_log_entry_info
    # End of _create_comparator

    def _lookup_or_create_person(
        self,
        session: Session,
        profile_id: str,
        username: str,
        conversation_id: Optional[str],
<<<<<<< HEAD
        existing_person_arg: Optional[Person] = None,
    ) -> Tuple[Optional[Person], Literal["new", "skipped", "error", "updated"]]:
        if not profile_id or profile_id == "UNKNOWN":
=======
    ) -> Tuple[Optional[Person], Literal["new", "skipped", "error", "updated"]]:
        """
        Looks up Person primarily by profile_id, creates if not found.
        Updates link and username if the found person's details differ.
        Returns the Person object and status. Ensures message_link is updated correctly.
        """
        if not profile_id or profile_id == "UNKNOWN":
            logger.warning(
                f"Cannot lookup person due to invalid profile_id '{profile_id}' in conv {conversation_id}. Skipping."
            )
>>>>>>> parent of b5d69aa (afternoon3)
            return None, "error"

        if not username:
            username = "Unknown"
<<<<<<< HEAD
        username_lower = username.lower()
        correct_message_link: Optional[str] = None
        try:
            correct_message_link = urljoin(
                config_instance.BASE_URL, f"messaging/?p={profile_id}"
            )
        except Exception as url_e:
            logger.warning(f"Error constructing msg link for {profile_id}: {url_e}")
        person: Optional[Person] = None
        lookup_needed = existing_person_arg is None
        try:
            if lookup_needed:
                person = (
                    session.query(Person)
                    .filter(Person.profile_id == profile_id)
                    .first()
                )
            else:
                person = existing_person_arg
=======

        profile_id_upper = (
            profile_id.upper()
        )  # Ensure consistent case for lookup and link
        username_lower = username.lower()  # For case-insensitive comparison

        # Construct the expected message link
        correct_message_link: Optional[str] = None
        try:
            # Ensure profile_id_upper is valid before constructing URL
            if profile_id_upper and profile_id_upper != "UNKNOWN":
                # Use urljoin for robustness
                base_url = getattr(
                    config_instance, "BASE_URL", "https://www.ancestry.co.uk/"
                )
                correct_message_link = urljoin(
                    base_url, f"messaging/?p={profile_id_upper}"
                )
            else:
                logger.warning(
                    f"Cannot construct message link, profile_id is invalid or UNKNOWN ({profile_id_upper}) for conv {conversation_id}."
                )
        except Exception as url_e:
            logger.warning(
                f"Error constructing message link for profile {profile_id_upper} in conv {conversation_id}: {url_e}"
            )

        try:
            # --- MODIFIED LOOKUP: Primarily by profile_id ---
            person = (
                session.query(Person)
                .filter(Person.profile_id == profile_id_upper)
                .first()
            )

>>>>>>> parent of b5d69aa (afternoon3)
            if person:
                updated = False
<<<<<<< HEAD
                if person.username.lower() != username_lower:
                    person.username = username
                    updated = True
                current_link = person.message_link
=======
                log_prefix = f"Person ID {person.id} ('{person.username}', Profile: {profile_id_upper})"

                # Check and update username if different (case-insensitive)
                if person.username.lower() != username_lower:
                    logger.debug(
                        f"{log_prefix}: Updating username from '{person.username}' to '{username}'."
                    )
                    session.query(Person).filter(Person.id == person.id).update(
                        {"username": username}
                    )
                    updated = True

                # Check and update message_link if different or missing
                # Corrected comparison: ColumnElement != str comparison is deprecated
                # Fetch the current value explicitly before comparing
                current_message_link_val = person.message_link

>>>>>>> parent of b5d69aa (afternoon3)
                if (
                    correct_message_link is not None
                    and current_link != correct_message_link
                ):
<<<<<<< HEAD
                    person.message_link = correct_message_link
                    updated = True
                if updated:
                    person.updated_at = datetime.now(timezone.utc)
=======
                    logger.debug(f"{log_prefix}: Updating message_link.")
                    session.query(Person).filter(Person.id == person.id).update(
                        {"message_link": correct_message_link}
                    )
                    updated = True
                elif (
                    current_message_link_val is None
                    and correct_message_link is not None
                ):
                    logger.debug(f"{log_prefix}: Adding missing message_link.")
                    session.query(Person).filter(Person.id == person.id).update(
                        {"message_link": correct_message_link}
                    )
                    updated = True

                # Ensure ID is valid before returning
                if person.id is None:
                    logger.error(
                        f"Found {person.username} but their ID is None. Data inconsistency? Conv {conversation_id}"
                    )
                    return None, "error"

                if updated:
                    # Explicitly set updated_at timestamp
                    person.updated_at = datetime.now()
                    # No flush needed here, let _save_batch handle commit/rollback
                    logger.debug(f"{log_prefix}: Record staged for update.")
>>>>>>> parent of b5d69aa (afternoon3)
                    return person, "updated"
                else:
                    return person, "skipped"
            else:
                new_person = Person(
                    profile_id=profile_id,
                    username=username,
<<<<<<< HEAD
                    message_link=correct_message_link,
                    status=PersonStatusEnum.ACTIVE,
                    contactable=True,
                )
                session.add(new_person)
                session.flush()
                if new_person.id is None:
                    logger.error(f"ID not assigned {username} ({profile_id})!")
=======
                    message_link=correct_message_link,  # Use constructed link
                    in_my_tree=False,  # Default for inbox-created person
                    uuid=None,  # Inbox doesn't provide UUID
                    status="active",
                    # created_at and updated_at have defaults
                )
                session.add(new_person)
                session.flush()  # Flush to get ID and check constraints early

                # Critical check for ID assignment
                if new_person.id is None:
                    logger.error(
                        f"Person ID not assigned after flush for {username} (Profile: {profile_id_upper})! Conversation {conversation_id}"
                    )
                    # Returning "error" will prevent this specific conversation from being saved in _save_batch
>>>>>>> parent of b5d69aa (afternoon3)
                    return None, "error"
                return new_person, "new"
        except SQLAlchemyError as e:
<<<<<<< HEAD
            logger.error(f"DB error _lookup {username} ({profile_id}): {e}")
=======
            logger.error(
                f"Database error in _lookup_or_create_person for {username} (Profile: {profile_id_upper}): {e}",
                exc_info=True,
            )
            # Let the caller handle rollback if necessary
>>>>>>> parent of b5d69aa (afternoon3)
            return None, "error"
        except Exception as e:  # Catch any other unexpected error
            logger.critical(
                f"Unexpected error _lookup {username} ({profile_id}): {e}",
                exc_info=True,
            )
            return None, "error"
    # End of _lookup_or_create_person

<<<<<<< HEAD
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
        """Logs the final summary for the 2-row ConversationLog approach."""
        logger.info("---- Inbox Search Summary (2-Row Log) ----")
        logger.info(f"  Total API Conversation Overviews Fetched: {total_api_items}")
        logger.info(f"  Conversations Processed (Incl. Context): {items_processed}")
        logger.info(f"  Latest Incoming Messages Classified:     {ai_classified}")
        logger.info(f"  Persons Updated to 'desist' Status:    {status_updates}")
        if stop_reason:
            logger.info(f"  Processing Stopped Due To: {stop_reason}")
        elif max_inbox_limit == 0 or items_processed < max_inbox_limit:
            logger.info(f"  Processing Stopped Due To: End of Inbox Reached")
        logger.info("-----------------------------------------")
    # End of _log_unified_summary


# End of InboxProcessor class
=======
    # end of _lookup_or_create_person

    def _save_batch(
        self, session: Session, batch: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Saves a batch of conversation data to the database using bulk operations.
        Maps 'my_role' variable to 'my_role' database column. Includes explicit checks
        before appending update items and uses cast for type checkers. Logs data before saving.
        """
        new_count, updated_count = 0, 0
        if not batch:
            # logger.debug("No new inbox data to save in this batch.") # Less verbose
            return new_count, updated_count

        new_items = []
        update_items = []

        try:
            people_ids_in_batch = [
                item["people_id"]
                for item in batch
                if "people_id" in item and item["people_id"] is not None
            ]
            if not people_ids_in_batch:
                logger.debug(
                    "Batch provided to _save_batch contains no valid people_ids."
                )
                return 0, 0

            # Fetch existing InboxStatus records for the people in this batch
            existing_statuses = {
                status.people_id: status
                for status in session.query(InboxStatus)
                .filter(InboxStatus.people_id.in_(people_ids_in_batch))
                .all()
            }
            logger.debug(
                f"Found {len(existing_statuses)} existing InboxStatus records for {len(people_ids_in_batch)} people_ids in batch."
            )

            for item in batch:
                people_id = item.get("people_id")
                # Check again for safety, although filtered above
                if people_id is None:
                    logger.warning(f"Skipping item due to missing 'people_id': {item}")
                    continue

                my_role_enum_value = item.get("my_role")

                # Ensure my_role is a valid RoleType enum member
                if not isinstance(my_role_enum_value, RoleType):
                    logger.warning(
                        f"Invalid role type '{my_role_enum_value}' for people_id {people_id}. Skipping item."
                    )
                    continue

                # Get corresponding existing status, if any
                inbox_status = existing_statuses.get(people_id)

                if not inbox_status:
                    # Prepare data for a new InboxStatus record
                    new_item_data = {
                        "people_id": people_id,
                        "conversation_id": item.get("conversation_id"),
                        "my_role": my_role_enum_value,  # Use validated enum
                        "last_message": item.get("last_message_content", ""),
                        "last_message_timestamp": item.get("last_message_timestamp"),
                    }
                    new_items.append(new_item_data)
                    # logger.debug(f"Prepared NEW item for people_id {people_id}: my_role={new_item_data['my_role']}") # Less verbose

                else:
                    # Existing record found, check if update is needed
                    needs_update: bool = False

                    # Compare last_message_timestamp (handle None safely)
                    db_ts = cast(
                        Optional[datetime], inbox_status.last_message_timestamp
                    )
                    item_ts = item.get("last_message_timestamp")
                    if db_ts != item_ts:
                        needs_update = True
                        # logger.debug(f"InboxStatus update needed for people_id {people_id}: Timestamp differs ('{db_ts}' vs '{item_ts}').") # Less verbose

                    # Compare last_message_content if timestamp matches or is None
                    if not needs_update:
                        db_msg = cast(Optional[str], inbox_status.last_message)
                        item_msg = item.get("last_message_content", "")
                        if db_msg != item_msg:
                            needs_update = True
                            # logger.debug(f"InboxStatus update needed for people_id {people_id}: Message differs.") # Less verbose

                    # Compare my_role if still no difference found
                    if not needs_update:
                        db_role = cast(
                            Optional[RoleType], inbox_status.my_role
                        )  # Should be RoleType enum
                        if db_role != my_role_enum_value:
                            needs_update = True
                            # logger.debug(f"InboxStatus update needed for people_id {people_id}: Role differs ('{db_role}' vs '{my_role_enum_value}').") # Less verbose

                    # Compare conversation_id if still no difference found
                    if not needs_update:
                        db_conv_id = cast(Optional[str], inbox_status.conversation_id)
                        item_conv_id = item.get("conversation_id")
                        if db_conv_id != item_conv_id:
                            needs_update = True
                            # logger.debug(f"InboxStatus update needed for people_id {people_id}: Conversation ID differs.") # Less verbose

                    if needs_update:
                        # Ensure the existing status object has an ID before adding to update list
                        if inbox_status.id is not None:
                            update_item_data = {
                                # Map to the database columns
                                "id": inbox_status.id,  # Primary key for update mapping
                                "conversation_id": item.get("conversation_id"),
                                "my_role": my_role_enum_value,  # Use validated enum
                                "last_message": item.get("last_message_content", ""),
                                "last_message_timestamp": item.get(
                                    "last_message_timestamp"
                                ),
                                # last_updated is handled by onupdate=datetime.now
                            }
                            update_items.append(update_item_data)
                            # logger.debug(f"Prepared UPDATE item for people_id {people_id} (InboxStatus.id: {update_item_data['id']}): my_role={update_item_data['my_role']}") # Less verbose
                        else:
                            # This case should ideally not happen if records are fetched correctly
                            logger.warning(
                                f"Skipping update for people_id {people_id}: Existing InboxStatus found but has no ID (maybe not flushed/committed yet?)."
                            )
                    # else:
                    #    logger.debug(f"No update needed for InboxStatus for people_id {people_id}.") # Less verbose

            # Perform bulk operations if there are items
            if new_items:
                logger.debug(
                    f"Performing bulk insert for {len(new_items)} new InboxStatus items..."
                )
                session.bulk_insert_mappings(InboxStatus, new_items)
                logger.debug(f"Bulk inserted {len(new_items)} items.")
                new_count = len(new_items)
            if update_items:
                logger.debug(
                    f"Performing bulk update for {len(update_items)} existing InboxStatus items..."
                )
                session.bulk_update_mappings(InboxStatus, update_items)
                logger.debug(f"Bulk updated {len(update_items)} items.")
                updated_count = len(update_items)

            # Flush changes within the batch save
            if new_items or update_items:
                logger.debug("Flushing session after bulk operations...")
                session.flush()
                logger.debug("Session flushed.")

            return new_count, updated_count

        except SQLAlchemyError as e:
            logger.error(f"Database save failed in _save_batch: {e}", exc_info=True)
            # Rollback should be handled by the caller context manager
            raise  # Re-raise to trigger rollback in the context manager
        except Exception as e:
            logger.error(f"Unexpected error in _save_batch: {e}", exc_info=True)
            raise  # Re-raise to trigger rollback in the context manager

    # end of _save_batch


def _log_inbox_summary(
    total_api_items: int,
    items_processed: int,
    new_records: int,
    updated_records: int,
    stop_reason: Optional[str],
    max_inbox_limit: int,
):
    """Logs the final summary of the inbox search action."""
    logger.info("---- Inbox Search Summary ----")
    logger.info(f"  Total API Items Fetched: {total_api_items}")
    logger.info(f"  Items Processed:         {items_processed}")
    logger.info(f"  New Statuses Saved:      {new_records}")
    logger.info(f"  Updated Statuses Saved:  {updated_records}")
    if stop_reason:
        logger.info(f"  Processing Stopped Due To: {stop_reason}")
    elif max_inbox_limit == 0 or items_processed < max_inbox_limit:
        # If no stop reason AND (limit is 0 OR we processed fewer than the limit), we reached the end
        logger.info(f"  Processing Stopped Due To: End of Inbox Reached")
    total_saved = new_records + updated_records
    logger.info("----------------------------\n")
>>>>>>> parent of b5d69aa (afternoon3)


# End of action7_inbox.py
