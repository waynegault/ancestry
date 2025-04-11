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
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast, Set
from urllib.parse import urljoin

# Third-party imports
import requests
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
    update,
    text,
    select as sql_select,
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.orm import Session as DbSession, aliased, joinedload
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from selenium.common.exceptions import WebDriverException

# Local application imports
from config import config_instance
from database import (
    Person,
    ConversationLog,
    MessageType,
    db_transn,
    PersonStatusEnum,
    MessageDirectionEnum,
)  # Import Enums
from utils import (
    _api_req,
    DynamicRateLimiter,
    SessionManager,
    retry,
    time_wait,
    retry_api,
    format_name,
    urljoin,
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
                use_csrf_token=False,
                api_description="Get Inbox Conversations",
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
        """V1.23 REVISED: Correctly limits API fetch size based on MAX_INBOX alongside Comparator logic, 2-row Log, AI, UPSERTS, batch commits, TZ-aware compare. DIAGNOSTIC LOGGING ADDED."""
        # --- Initialization ---
        ai_classified_count = 0
        status_updated_count = 0
        total_processed_api_items = 0
        items_processed_before_stop = 0
        self.max_inbox_limit = config_instance.MAX_INBOX  # Reinstated MAX_INBOX limit
        stop_reason = ""
        next_cursor: Optional[str] = None
        current_batch_num = 0
        conv_log_upserts: List[Dict[str, Any]] = []
        person_updates: Dict[int, Dict[str, Any]] = {}
        stop_processing = False  # Flag to stop after comparator OR limit

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
                comp_ts = comparator_info.get("latest_timestamp")  # Already aware UTC
            # --- End Get Comparator ---

            logger.info(
                f"Starting inbox search (Comparator Logic, MAX_INBOX={self.max_inbox_limit}, 2-Row Log, Contextual AI)..."
            )

            while (
                not stop_processing
            ):  # Loop until comparator found, limit reached, or end of API
                try:  # Inner try for batch processing
                    if not self.session_manager.is_sess_valid():
                        raise WebDriverException(
                            "Session invalid before overview batch"
                        )

                    # --- MODIFIED: Calculate Limit for this API call ---
                    current_limit = self.default_batch_size  # Start with default
                    if self.max_inbox_limit > 0:
                        remaining_allowed = (
                            self.max_inbox_limit - items_processed_before_stop
                        )
                        if remaining_allowed <= 0:
                            # This check might be redundant due to the loop break, but safe to keep
                            stop_reason = (
                                f"Inbox Limit ({self.max_inbox_limit}) Pre-Fetch"
                            )
                            stop_processing = True
                            break  # Break inner try loop
                        # Adjust limit to fetch only what's needed (up to batch size)
                        current_limit = min(self.default_batch_size, remaining_allowed)
                        logger.debug(
                            f"MAX_INBOX active. Calculated remaining={remaining_allowed}, API limit for this batch: {current_limit}"
                        )
                    else:
                        logger.debug(
                            f"MAX_INBOX inactive (0). Using default API limit: {current_limit}"
                        )
                    # --- END MODIFICATION ---

                    all_conversations_batch, next_cursor_from_api = (
                        self._get_all_conversations_api(
                            self.session_manager,
                            limit=current_limit,  # Use the calculated limit
                            cursor=next_cursor,
                        )
                    )

                    # *** DIAGNOSTIC LOGGING START ***
                    logger.debug(f"--- DIAG: API Batch Result ---")
                    logger.debug(
                        f"--- DIAG: Type(all_conversations_batch): {type(all_conversations_batch)}"
                    )
                    if isinstance(all_conversations_batch, list):
                        logger.debug(
                            f"--- DIAG: Len(all_conversations_batch): {len(all_conversations_batch)}"
                        )
                        if all_conversations_batch:
                            logger.debug(
                                f"--- DIAG: First item preview: {str(all_conversations_batch[0])[:200]}..."
                            )
                        else:
                            logger.debug("--- DIAG: all_conversations_batch is empty.")
                    else:
                        logger.debug(
                            f"--- DIAG: all_conversations_batch value: {all_conversations_batch}"
                        )
                    logger.debug(
                        f"--- DIAG: next_cursor_from_api: {next_cursor_from_api}"
                    )
                    # *** DIAGNOSTIC LOGGING END ***

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
                            stop_reason = (
                                "End of Inbox Reached (Empty Batch, No Cursor)"
                            )
                            stop_processing = True
                        else:
                            # Got a cursor but no items? Weird, but continue loop.
                            logger.debug(
                                "API returned empty batch but provided a cursor. Continuing fetch."
                            )
                            next_cursor = next_cursor_from_api
                            continue  # Skip processing, go to next API call

                    # Apply rate limiter delay if items were fetched
                    wait_duration = self.dynamic_rate_limiter.wait()
                    if wait_duration > 0.1:
                        logger.debug(f"API batch wait: {wait_duration:.2f}s")

                    current_batch_num += 1

                    # Pre-fetch Persons and Logs
                    # ...(Pre-fetching logic unchanged)...
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
                    existing_conv_logs: Dict[Tuple[str, str], ConversationLog] = {}
                    if batch_profile_ids:
                        try:
                            persons = (
                                session.query(Person)
                                .filter(Person.profile_id.in_(batch_profile_ids))
                                .all()
                            )
                            existing_persons_map = {
                                p.profile_id: p for p in persons if p.profile_id
                            }
                        except SQLAlchemyError as db_err:
                            logger.error(f"Bulk Person lookup failed: {db_err}")
                    if batch_conv_ids:
                        try:
                            logs = (
                                session.query(ConversationLog)
                                .filter(
                                    ConversationLog.conversation_id.in_(batch_conv_ids)
                                )
                                .all()
                            )
                            existing_conv_logs = {
                                (log.conversation_id, log.direction.name): log
                                for log in logs
                                if log.direction
                            }
                        except SQLAlchemyError as db_err:
                            logger.error(f"ConvLog lookup failed: {db_err}")

                    # Process Batch
                    for conversation_info in all_conversations_batch:

                        # *** DIAGNOSTIC LOGGING START ***
                        logger.debug(
                            f"--- DIAG: >>> Entering conversation processing loop for item <<<"
                        )
                        logger.debug(
                            f"--- DIAG: Processing conversation_info: {conversation_info}"
                        )
                        # *** DIAGNOSTIC LOGGING END ***

                        # --- Check MAX_INBOX Limit ---
                        # This check is now more of a safety fallback, as the API limit calculation should prevent exceeding it.
                        if (
                            self.max_inbox_limit > 0
                            and items_processed_before_stop >= self.max_inbox_limit
                        ):
                            if not stop_reason:
                                stop_reason = f"Inbox Limit ({self.max_inbox_limit})"  # Set reason if not already set
                            stop_processing = True
                            break  # Break inner loop (for conversation_info...)
                        # --- End MAX_INBOX Check ---

                        items_processed_before_stop += (
                            1  # Increment only if not stopped by limit
                        )

                        profile_id_upper = conversation_info.get(
                            "profile_id", "UNKNOWN"
                        ).upper()
                        api_conv_id = conversation_info.get("conversation_id")
                        api_latest_ts = conversation_info.get("last_message_timestamp")

                        if not api_conv_id or profile_id_upper == "UNKNOWN":
                            logger.debug(
                                f"Skipping item {items_processed_before_stop}: Invalid ConvID or ProfileID."
                            )
                            continue

                        needs_fetch = False
                        # Comparator Logic
                        if comp_conv_id and api_conv_id == comp_conv_id:
                            logger.debug(
                                f"Comparator conversation {comp_conv_id} found in API results."
                            )
                            stop_processing = True
                            if comp_ts and api_latest_ts and api_latest_ts > comp_ts:
                                needs_fetch = True
                                logger.debug("Comparator needs update.")
                            else:
                                logger.debug("Comparator does not need update.")
                        else:
                            # Check this specific conversation against its own history
                            # ...(logic unchanged)...
                            db_log_in = existing_conv_logs.get((api_conv_id, "IN"))
                            db_log_out = existing_conv_logs.get((api_conv_id, "OUT"))
                            min_aware_dt = datetime.min.replace(tzinfo=timezone.utc)
                            db_latest_ts_in = min_aware_dt
                            if db_log_in and db_log_in.latest_timestamp:
                                ts_in = db_log_in.latest_timestamp
                                if isinstance(ts_in, datetime):
                                    db_latest_ts_in = (
                                        ts_in.replace(tzinfo=timezone.utc)
                                        if ts_in.tzinfo is None
                                        else ts_in.astimezone(timezone.utc)
                                    )
                            db_latest_ts_out = min_aware_dt
                            if db_log_out and db_log_out.latest_timestamp:
                                ts_out = db_log_out.latest_timestamp
                                if isinstance(ts_out, datetime):
                                    db_latest_ts_out = (
                                        ts_out.replace(tzinfo=timezone.utc)
                                        if ts_out.tzinfo is None
                                        else ts_out.astimezone(timezone.utc)
                                    )
                            db_latest_overall_for_conv = max(
                                db_latest_ts_in, db_latest_ts_out
                            )
                            if (
                                api_latest_ts
                                and api_latest_ts > db_latest_overall_for_conv
                            ):
                                needs_fetch = True
                            elif not db_log_in and not db_log_out:
                                needs_fetch = True

                        if not needs_fetch:
                            if stop_processing:
                                logger.debug(
                                    "Comparator found and no fetch needed, breaking inner loop."
                                )
                                break
                            continue

                        # Fetch Context & Process (if needs_fetch is True)
                        # ...(logic unchanged)...
                        if not self.session_manager.is_sess_valid():
                            raise WebDriverException(
                                f"Session invalid fetch context conv {api_conv_id}"
                            )
                        context_messages = self._fetch_conversation_context(api_conv_id)
                        if context_messages is None:
                            logger.error(
                                f"Failed context fetch conv {api_conv_id}. Skipping item."
                            )
                            continue

                        person, person_status_flag = self._lookup_or_create_person(
                            session,
                            profile_id_upper,
                            conversation_info.get("username", "Unknown"),
                            api_conv_id,
                            existing_person_arg=existing_persons_map.get(
                                profile_id_upper
                            ),
                        )
                        if not person or not person.id:
                            logger.error(
                                f"Failed person lookup/create conv {api_conv_id}. Skipping item."
                            )
                            continue
                        people_id = person.id

                        latest_ctx_in: Optional[Dict] = None
                        latest_ctx_out: Optional[Dict] = None
                        for msg in reversed(context_messages):
                            author_lower = msg.get("author", "")
                            if author_lower != my_pid_lower and latest_ctx_in is None:
                                latest_ctx_in = msg
                            elif (
                                author_lower == my_pid_lower and latest_ctx_out is None
                            ):
                                latest_ctx_out = msg
                            if latest_ctx_in and latest_ctx_out:
                                break

                        ai_sentiment_result: Optional[str] = None
                        # Process IN Row
                        if latest_ctx_in and latest_ctx_in.get("timestamp"):
                            ctx_ts_in_aware = latest_ctx_in.get("timestamp")
                            if ctx_ts_in_aware and ctx_ts_in_aware > db_latest_ts_in:
                                formatted_context = self._format_context_for_ai(
                                    context_messages, my_pid_lower
                                )
                                if not self.session_manager.is_sess_valid():
                                    raise WebDriverException(
                                        f"Session invalid AI call conv {api_conv_id}"
                                    )
                                ai_sentiment_result = classify_message_intent(
                                    formatted_context, self.session_manager
                                )
                                ai_classified_count += 1
                                if ai_sentiment_result in ("DESIST", "UNINTERESTED"):
                                    logger.info(
                                        f"AI Classified conv {api_conv_id} (PID {people_id}) as '{ai_sentiment_result}'. Marking for 'desist'."
                                    )
                                    if people_id not in person_updates:
                                        person_updates[people_id] = {}
                                    person_updates[people_id][
                                        "status"
                                    ] = PersonStatusEnum.DESIST
                                upsert_data_in = {
                                    "conversation_id": api_conv_id,
                                    "direction": "IN",
                                    "people_id": people_id,
                                    "latest_message_content": latest_ctx_in.get(
                                        "content", ""
                                    )[: config_instance.MESSAGE_TRUNCATION_LENGTH],
                                    "latest_timestamp": ctx_ts_in_aware,
                                    "ai_sentiment": ai_sentiment_result,
                                    "updated_at": datetime.now(timezone.utc),
                                }
                                upsert_data_in = {
                                    k: v
                                    for k, v in upsert_data_in.items()
                                    if v is not None or k == "ai_sentiment"
                                }
                                conv_log_upserts.append(upsert_data_in)
                        # Process OUT Row
                        if latest_ctx_out and latest_ctx_out.get("timestamp"):
                            ctx_ts_out_aware = latest_ctx_out.get("timestamp")
                            if ctx_ts_out_aware and ctx_ts_out_aware > db_latest_ts_out:
                                upsert_data_out = {
                                    "conversation_id": api_conv_id,
                                    "direction": "OUT",
                                    "people_id": people_id,
                                    "latest_message_content": latest_ctx_out.get(
                                        "content", ""
                                    )[: config_instance.MESSAGE_TRUNCATION_LENGTH],
                                    "latest_timestamp": ctx_ts_out_aware,
                                    "message_type_id": None,
                                    "script_message_status": None,
                                    "ai_sentiment": None,
                                    "updated_at": datetime.now(timezone.utc),
                                }
                                upsert_data_out = {
                                    k: v
                                    for k, v in upsert_data_out.items()
                                    if v is not None
                                    or k in ["message_type_id", "script_message_status"]
                                }
                                conv_log_upserts.append(upsert_data_out)

                        if (
                            stop_processing
                        ):  # Break inner loop if comparator/limit reached
                            logger.debug(
                                f"Stop flag set ({stop_reason}), breaking inner loop."
                            )
                            break
                    # --- End Inner Loop (for conversation_info...) ---

                    # Commit batch data
                    if conv_log_upserts or person_updates:
                        logger.info(
                            f"Attempting batch commit (Batch {current_batch_num}): {len(conv_log_upserts)} logs, {len(person_updates)} persons..."
                        )
                        status_updates_this_batch = self._commit_batch_data_upsert(
                            session, conv_log_upserts, person_updates
                        )
                        status_updated_count += status_updates_this_batch
                        conv_log_upserts.clear()
                        person_updates.clear()
                        logger.info(
                            f"Batch commit attempt finished (Batch {current_batch_num}). Updated {status_updates_this_batch} persons."
                        )

                    # Check stop flag *after* commit and *before* getting next cursor
                    if stop_processing:
                        if not stop_reason:
                            stop_reason = (
                                "Comparator Found"  # Set default reason if needed
                            )
                        logger.debug(
                            f"Stop flag set ({stop_reason}). Breaking outer loop."
                        )
                        break  # Break outer loop (while not stop_processing)

                    next_cursor = next_cursor_from_api
                    if not next_cursor:
                        stop_reason = "End of Inbox Reached"
                        stop_processing = True
                        logger.debug(
                            "No next cursor from API. Breaking inner try block."
                        )
                        break  # Break inner try

                # -- End Try block for batch processing --
                except WebDriverException as WDE:
                    logger.error(f"WebDriverException occurred: {WDE}")
                    stop_reason = "WebDriver Exception"
                    stop_processing = True
                    logger.warning("Attempting save and restart...")
                    save_count = self._commit_batch_data_upsert(
                        session, conv_log_upserts, person_updates, is_final_attempt=True
                    )
                    status_updated_count += save_count
                    conv_log_upserts.clear()
                    person_updates.clear()
                    if self.session_manager.restart_sess():
                        logger.info("Session restarted. Retrying inbox search...")
                        session = self.session_manager.get_db_conn()
                        if not session:
                            raise SQLAlchemyError(
                                "Failed to get DB session after restart."
                            )
                        stop_processing = False
                        next_cursor = None
                        current_batch_num = 0
                        continue
                    else:
                        logger.critical("Session restart failed.")
                        return False
                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt detected.")
                    stop_reason = "Keyboard Interrupt"
                    stop_processing = True
                    logger.warning("Attempting final save...")
                    save_count = self._commit_batch_data_upsert(
                        session, conv_log_upserts, person_updates, is_final_attempt=True
                    )
                    status_updated_count += save_count
                    break
                except Exception as e_main:
                    logger.critical(
                        f"Critical error in search_inbox loop: {e_main}", exc_info=True
                    )
                    stop_reason = f"Critical Error ({type(e_main).__name__})"
                    stop_processing = True
                    logger.warning("Attempting final save...")
                    save_count = self._commit_batch_data_upsert(
                        session, conv_log_upserts, person_updates, is_final_attempt=True
                    )
                    status_updated_count += save_count
                    return False

            # --- End main while loop ---

        except Exception as outer_e:
            logger.error(f"Outer error in search_inbox: {outer_e}", exc_info=True)
            return False

        finally:
            # Final commit attempt
            if session and (conv_log_upserts or person_updates):
                logger.warning("Performing final commit outside loop.")
                final_save_count = self._commit_batch_data_upsert(
                    session, conv_log_upserts, person_updates, is_final_attempt=True
                )
                status_updated_count += final_save_count
            # Final summary log
            if (
                stop_reason
                and not stop_reason.startswith("End of Inbox")
                and not stop_reason.startswith("Comparator")
            ):
                logger.warning(f"Inbox search stopped early: {stop_reason}")
            # Pass actual MAX_INBOX limit to summary function
            self._log_unified_summary(
                total_api_items=total_processed_api_items,
                items_processed=items_processed_before_stop,
                new_logs=0,
                ai_classified=ai_classified_count,
                status_updates=status_updated_count,
                stop_reason=stop_reason or "Comparator Found or End of Inbox",
                max_inbox_limit=self.max_inbox_limit,  # Pass the actual limit value
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
        V1.22 REVISED: Uses session.add_all() for new logs, explicit add for updates.
        - Collects new ConversationLog objects in a list.
        - Uses session.add_all() after the loop to add all new logs.
        - Explicitly adds modified existing_log objects back to session.
        - Removes intermediate flushes, relies on final db_transn commit.
        """
        updated_person_count = 0
        processed_logs_count = 0
        new_logs_to_add: List[ConversationLog] = []  # Initialize list for new logs

        if not log_upserts and not person_updates:
            return updated_person_count

        log_prefix = "[Final Save] " if is_final_attempt else "[Batch Save] "
        logger.info(
            f"{log_prefix} Preparing commit: {len(log_upserts)} logs, {len(person_updates)} person updates."
        )

        # --- Log Data Before Commit (Keep for debugging) ---
        # ... (logging data remains the same) ...
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{log_prefix}--- Log Upsert Data ---")
            for item in log_upserts:
                log_item_repr = {
                    k: (
                        str(v)[:50] + "..." if isinstance(v, str) and len(v) > 50 else v
                    )
                    for k, v in item.items()
                }
                logger.debug(f"  {log_item_repr}")
            logger.debug(f"{log_prefix}--- Person Update Data ---")
            for pid, data in person_updates.items():
                logger.debug(f"  PersonID {pid}: {data}")
            logger.debug(f"{log_prefix}--- End Data ---")

        try:
            # Use the main transaction context manager for the entire batch
            with db_transn(session):
                logger.debug(f"{log_prefix}Entered transaction block.")

                # --- ConversationLog Explicit Upsert Logic ---
                if log_upserts:
                    logger.debug(
                        f"{log_prefix}Processing {len(log_upserts)} ConversationLog entries..."
                    )
                    for data in log_upserts:
                        try:
                            conv_id = data.get("conversation_id")
                            direction_str = data.get("direction")

                            # Validate and Convert Direction
                            if not conv_id or not direction_str:
                                logger.error(
                                    f"{log_prefix}Missing conv_id/direction: {data}. Skip."
                                )
                                continue
                            try:
                                direction_enum = MessageDirectionEnum(direction_str)
                            except ValueError:
                                logger.error(
                                    f"{log_prefix}Invalid direction '{direction_str}' conv {conv_id}. Skip."
                                )
                                continue

                            # Prepare data dictionary for update/insert
                            ts_val = data.get("latest_timestamp")
                            aware_timestamp = None
                            if isinstance(ts_val, datetime):
                                aware_timestamp = (
                                    ts_val.astimezone(timezone.utc)
                                    if ts_val.tzinfo
                                    else ts_val.replace(tzinfo=timezone.utc)
                                )
                            if not aware_timestamp:
                                logger.error(
                                    f"{log_prefix} Missing/invalid timestamp {conv_id}/{direction_enum.name}. Skip."
                                )
                                continue

                            update_data = {
                                k: v
                                for k, v in data.items()
                                if k
                                not in ["conversation_id", "direction", "updated_at"]
                                and (
                                    v is not None
                                    or k
                                    in [
                                        "ai_sentiment",
                                        "message_type_id",
                                        "script_message_status",
                                    ]
                                )
                            }
                            update_data["updated_at"] = datetime.now(timezone.utc)
                            update_data["latest_timestamp"] = aware_timestamp

                            # Query for existing record
                            existing_log = (
                                session.query(ConversationLog)
                                .filter_by(
                                    conversation_id=conv_id, direction=direction_enum
                                )
                                .first()
                            )  # No with_for_update needed if single worker

                            if existing_log:
                                # Update Existing Record
                                logger.debug(
                                    f"{log_prefix} Updating existing log for {conv_id}/{direction_enum.name}..."
                                )
                                has_changes = False
                                for key, value in update_data.items():
                                    if getattr(existing_log, key) != value:
                                        setattr(existing_log, key, value)
                                        has_changes = True
                                if not has_changes:
                                    logger.debug(
                                        f"    No actual changes needed for {conv_id}/{direction_enum.name}."
                                    )
                                else:
                                    logger.debug(
                                        f"    Changes staged for {conv_id}/{direction_enum.name}."
                                    )
                                    # *** Explicitly add modified object back to session ***
                                    session.add(existing_log)
                                    logger.debug(
                                        f"    Explicitly added modified existing_log to session."
                                    )
                            else:
                                # Create New Record Object
                                logger.debug(
                                    f"{log_prefix} Creating new log entry object for {conv_id}/{direction_enum.name}..."
                                )
                                new_log_data = update_data.copy()
                                new_log_data["conversation_id"] = conv_id
                                new_log_data["direction"] = direction_enum
                                if "people_id" not in new_log_data:
                                    logger.error(
                                        f"{log_prefix} Missing 'people_id' for new log {conv_id}/{direction_enum.name}. Skip."
                                    )
                                    continue
                                new_log_obj = ConversationLog(**new_log_data)
                                # *** Append to list instead of adding directly ***
                                new_logs_to_add.append(new_log_obj)
                                logger.debug(
                                    f"    New ConversationLog object added to list for {conv_id}/{direction_enum.name}."
                                )

                            processed_logs_count += 1

                        except Exception as inner_log_exc:
                            logger.error(
                                f"{log_prefix} Error processing single log item (data: {data}): {inner_log_exc}",
                                exc_info=True,
                            )
                    logger.debug(
                        f"{log_prefix}Finished processing {processed_logs_count} log entries."
                    )

                    # *** Add all collected new logs AFTER the loop ***
                    if new_logs_to_add:
                        logger.debug(
                            f"{log_prefix}Adding {len(new_logs_to_add)} new ConversationLog objects to session using add_all()..."
                        )
                        session.add_all(new_logs_to_add)
                        logger.debug(
                            f"{log_prefix}session.add_all() called for new logs."
                        )
                    else:
                        logger.debug(
                            f"{log_prefix}No new ConversationLog objects to add."
                        )
                # --- End ConversationLog Logic ---

                # --- Person Update Logic (Keep bulk_update_mappings) ---
                if person_updates:
                    update_values = []
                    logger.debug(
                        f"{log_prefix}Preparing {len(person_updates)} Person status updates..."
                    )
                    for pid, data in person_updates.items():
                        status_val = data.get("status")
                        enum_value = None
                        if isinstance(status_val, PersonStatusEnum):
                            enum_value = status_val
                        elif isinstance(status_val, str):
                            try:
                                enum_value = PersonStatusEnum(status_val.upper())
                            except ValueError:
                                logger.warning(
                                    f"Invalid status string '{status_val}' for Person ID {pid}. Skip."
                                )
                                continue
                        else:
                            logger.warning(
                                f"Invalid status type for Person ID {pid}: {type(status_val)}. Skip."
                            )
                            continue

                        logger.debug(
                            f"  Prep update Person ID {pid}: status -> {enum_value.name}"
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
                            f"{log_prefix}Attempting bulk update mappings for {len(update_values)} persons..."
                        )
                        try:
                            session.bulk_update_mappings(Person, update_values)
                            updated_person_count = len(update_values)
                            logger.info(
                                f"{log_prefix}Bulk update mappings called for {updated_person_count} persons."
                            )
                        except Exception as bulk_err:
                            logger.error(
                                f"{log_prefix} Error during bulk_update_mappings: {bulk_err}",
                                exc_info=True,
                            )
                            raise  # Re-raise to trigger rollback
                    else:
                        logger.warning(f"{log_prefix}No valid person updates prepared.")
                # --- End Person Update Logic ---

                # --- Log Session State Before Commit ---
                logger.info(f"{log_prefix}Session state BEFORE commit attempt:")
                logger.info(f"  Dirty objects: {len(session.dirty)}")
                if logger.isEnabledFor(logging.DEBUG) and session.dirty:
                    logger.debug(f"    Dirty Details: {session.dirty}")
                logger.info(f"  New objects: {len(session.new)}")
                if logger.isEnabledFor(logging.DEBUG) and session.new:
                    # Log reprs of new objects for better debugging
                    new_obj_reprs = [repr(obj) for obj in session.new]
                    logger.debug(f"    New Details: {new_obj_reprs}")
                logger.info(f"  Deleted objects: {len(session.deleted)}")
                # --- End Log Session State ---

                logger.debug(
                    f"{log_prefix}Exiting transaction block (commit attempt follows)."
                )
            # --- Commit happens implicitly here by db_transn exiting the 'with' block ---

            logger.info(
                f"{log_prefix}Transaction block exited. Commit should have occurred via db_transn."
            )
            return updated_person_count

        except Exception as commit_err:
            logger.error(
                f"{log_prefix}Commit/Processing FAILED inside helper: {commit_err}",
                exc_info=True,
            )
            return 0

    # End of _commit_batch_data_upsert

    def _create_comparator(self, session: DbSession) -> Optional[Dict[str, Any]]:
        """
        V1.21 REVISED: Finds the ConversationLog entry with the latest timestamp.

        Returns:
            A dictionary {'conversation_id': str, 'latest_timestamp': datetime}
            or None if the ConversationLog table is empty.
        """
        latest_log_entry_info = None
        try:
            # Find the log entry with the maximum timestamp
            latest_entry = (
                session.query(
                    ConversationLog.conversation_id, ConversationLog.latest_timestamp
                )
                .order_by(ConversationLog.latest_timestamp.desc().nullslast())
                .first()
            )

            if latest_entry:
                # Ensure the timestamp is timezone-aware (assuming UTC)
                log_timestamp = latest_entry.latest_timestamp
                aware_timestamp = None
                if isinstance(log_timestamp, datetime):
                    aware_timestamp = (
                        log_timestamp.replace(tzinfo=timezone.utc)
                        if log_timestamp.tzinfo is None
                        else log_timestamp.astimezone(timezone.utc)
                    )

                if latest_entry.conversation_id and aware_timestamp:
                    latest_log_entry_info = {
                        "conversation_id": latest_entry.conversation_id,
                        "latest_timestamp": aware_timestamp,
                    }
                    logger.debug(
                        f"Comparator created: ConvID={latest_log_entry_info['conversation_id']}, TS={latest_log_entry_info['latest_timestamp']}"
                    )
                else:
                    logger.warning(
                        f"Found latest log entry, but data invalid: ConvID={latest_entry.conversation_id}, TS={log_timestamp}"
                    )

            else:
                logger.info(
                    "ConversationLog empty. Comparator not created."
                )  # Use INFO

        except Exception as e:
            logger.error(f"Error creating comparator: {e}", exc_info=True)
            return None  # Return None on error
        return latest_log_entry_info

    # End of _create_comparator

    def _lookup_or_create_person(
        self,
        session: DbSession,
        profile_id: str,
        username: str,
        conversation_id: Optional[
            str
        ],  # Keep conversation_id for message link construction
        existing_person_arg: Optional[Person] = None,
    ) -> Tuple[Optional[Person], Literal["new", "skipped", "error", "updated"]]:
        """
        V1.21 REVISED: Looks up or creates a Person.
        - Fetches profile details (FirstName, IsContactable, LastLoginDate) via helper
          when CREATING a new Person record.
        """
        if not profile_id or profile_id == "UNKNOWN":
            logger.warning("_lookup_or_create_person: Invalid profile_id provided.")
            return None, "error"
        if not username:
            username = "Unknown"  # Default username if missing

        username_lower = username.lower()
        correct_message_link: Optional[str] = None
        try:
            # Use profile_id for the message link target
            correct_message_link = urljoin(
                config_instance.BASE_URL, f"/messaging/?p={profile_id.upper()}"
            )
        except Exception as url_e:
            logger.warning(f"Error constructing msg link for {profile_id}: {url_e}")

        person: Optional[Person] = None
        lookup_needed = existing_person_arg is None

        try:
            if lookup_needed:
                # Query by profile_id only, as username might change or be inaccurate in overview
                person = (
                    session.query(Person)
                    .filter(Person.profile_id == profile_id.upper())
                    .first()
                )
            else:
                person = existing_person_arg  # Use the one passed from prefetch

            if person:
                # --- EXISTING PERSON ---
                updated = False
                # Update username only if current one is "Unknown" or differs significantly
                # (Avoid overwriting potentially more accurate list username with older DB one)
                if (
                    person.username == "Unknown"
                    or person.username.lower() != username_lower
                ):
                    # Maybe add more sophisticated name comparison if needed
                    person.username = (
                        username  # Update with potentially newer username from overview
                    )
                    updated = True
                # Update message link if missing or different
                current_link = person.message_link
                if (
                    correct_message_link is not None
                    and current_link != correct_message_link
                ):
                    person.message_link = correct_message_link
                    updated = True

                # Optional: Update other fields if missing?
                # You could call _fetch_profile_details_for_person here too and update if
                # fields like first_name, contactable, last_logged_in are NULL in the DB.
                # Example (add this block if desired):
                # if person.first_name is None or person.contactable is None or person.last_logged_in is None:
                #      logger.debug(f"Existing person {profile_id} missing details. Fetching profile...")
                #      profile_details = _fetch_profile_details_for_person(self.session_manager, profile_id)
                #      if profile_details:
                #          if person.first_name is None and profile_details.get("first_name"):
                #              person.first_name = profile_details["first_name"]; updated = True
                #          if person.contactable is None and profile_details.get("contactable") is not None:
                #              person.contactable = profile_details["contactable"]; updated = True
                #          if person.last_logged_in is None and profile_details.get("last_logged_in_dt"):
                #              person.last_logged_in = profile_details["last_logged_in_dt"]; updated = True

                if updated:
                    person.updated_at = datetime.now(timezone.utc)
                    logger.debug(
                        f"Updated existing Person record for {profile_id}/{username}"
                    )
                    # Ensure the updated object is managed by the session
                    session.add(person)
                    return person, "updated"
                else:
                    logger.debug(
                        f"Skipped update for existing Person {profile_id}/{username} (no changes detected)."
                    )
                    return person, "skipped"
            else:
                # --- NEW PERSON ---
                logger.info(
                    f"Person with profile ID {profile_id.upper()} not found. Attempting to create..."
                )
                # Fetch additional details before creating
                profile_details = _fetch_profile_details_for_person(
                    self.session_manager, profile_id
                )

                # Prepare data for new Person object
                new_person_data = {
                    "profile_id": profile_id.upper(),
                    "username": username,  # Use username from conversation overview
                    "message_link": correct_message_link,
                    "status": PersonStatusEnum.ACTIVE,  # Default status
                    "first_name": None,  # Default
                    "contactable": True,  # Default assumption
                    "last_logged_in": None,  # Default
                    # Other fields like gender, birth_year, admin info will be None unless fetched
                    "administrator_profile_id": None,
                    "administrator_username": None,
                    "gender": None,
                    "birth_year": None,
                    "in_my_tree": False,  # Default, can be updated by Action 6 if run later
                }

                if profile_details:
                    logger.debug(
                        f"Populating new person {profile_id} with fetched profile details."
                    )
                    new_person_data["first_name"] = profile_details.get("first_name")
                    # Ensure contactable defaults to False if API returns None, otherwise use API value
                    new_person_data["contactable"] = (
                        profile_details.get("contactable", False)
                        if profile_details.get("contactable") is not None
                        else False
                    )
                    new_person_data["last_logged_in"] = profile_details.get(
                        "last_logged_in_dt"
                    )
                else:
                    logger.warning(
                        f"Could not fetch profile details for new person {profile_id}. Using defaults."
                    )

                # Create and add the new Person
                new_person = Person(**new_person_data)
                session.add(new_person)
                # Flush here to get the new_person.id assigned immediately
                # This is useful if subsequent operations in the same batch need the ID
                try:
                    session.flush()
                    if new_person.id is None:
                        logger.error(
                            f"ID not assigned after flush for {username} ({profile_id})!"
                        )
                        session.rollback()  # Rollback if flush didn't assign ID
                        return None, "error"
                    logger.info(
                        f"Created new Person ID {new_person.id} for {username} ({profile_id})."
                    )
                    return new_person, "new"
                except (
                    IntegrityError
                ) as ie:  # Catch potential duplicate profile_id on flush
                    session.rollback()
                    logger.error(
                        f"IntegrityError creating Person {username} ({profile_id}): {ie}. Rolling back.",
                        exc_info=False,
                    )
                    # Attempt to refetch the person that caused the conflict
                    conflicting_person = (
                        session.query(Person)
                        .filter(Person.profile_id == profile_id.upper())
                        .first()
                    )
                    if conflicting_person:
                        logger.warning(
                            f"Returning conflicting Person ID {conflicting_person.id} instead."
                        )
                        return (
                            conflicting_person,
                            "skipped",
                        )  # Treat as skipped as it already exists
                    return None, "error"
                except SQLAlchemyError as e_flush:
                    logger.error(
                        f"DB error during flush for new person {username} ({profile_id}): {e_flush}"
                    )
                    session.rollback()
                    return None, "error"

        except SQLAlchemyError as e:
            logger.error(
                f"DB error in _lookup_or_create_person for {username} ({profile_id}): {e}"
            )
            # Rollback might be needed depending on session state, but let db_transn handle it usually
            return None, "error"
        except Exception as e:
            logger.critical(
                f"Unexpected error in _lookup_or_create_person {username} ({profile_id}): {e}",
                exc_info=True,
            )
            return None, "error"

    # End of _lookup_or_create_person

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


@retry_api(retry_on_exceptions=(requests.exceptions.RequestException, ConnectionError))
def _fetch_profile_details_for_person(
    session_manager: SessionManager, profile_id: str
) -> Optional[Dict[str, Any]]:
    """
    Fetches profile details (FirstName, IsContactable, LastLoginDate) for a given profile ID.
    """
    if not profile_id or profile_id == "UNKNOWN":
        logger.warning("Cannot fetch profile details: Invalid profile_id provided.")
        return None
    if not session_manager or not session_manager.is_sess_valid():
        logger.error(
            f"Profile details fetch: WebDriver session invalid for Profile ID {profile_id}."
        )
        # Don't raise ConnectionError here, just return None as it might be called when session is closing
        return None

    profile_url = urljoin(
        config_instance.BASE_URL,
        f"/app-api/express/v1/profiles/details?userId={profile_id.upper()}",
    )
    api_desc_profile = "Profile Details API (Action 7)"
    profile_data = {}

    try:
        profile_response = _api_req(
            url=profile_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description=api_desc_profile,
            # Referer might be less critical here, but could use messaging page if needed
            # referer_url=urljoin(config_instance.BASE_URL, "/messaging/"),
        )

        if profile_response and isinstance(profile_response, dict):
            logger.debug(f"Fetched /profiles/details for {profile_id} OK.")

            # Extract FirstName
            raw_first_name = profile_response.get("FirstName")
            profile_data["first_name"] = (
                format_name(raw_first_name) if raw_first_name else None
            )

            # Extract IsContactable
            contactable_val = profile_response.get("IsContactable")
            profile_data["contactable"] = (
                bool(contactable_val) if contactable_val is not None else False
            )  # Default False if missing

            # Extract and parse LastLoginDate
            last_login_str = profile_response.get("LastLoginDate")
            last_login_dt = None
            if last_login_str:
                try:
                    if last_login_str.endswith("Z"):
                        last_login_dt = datetime.fromisoformat(
                            last_login_str.replace("Z", "+00:00")
                        )
                    else:
                        dt_naive = datetime.fromisoformat(last_login_str)
                        last_login_dt = (
                            dt_naive.replace(tzinfo=timezone.utc)
                            if dt_naive.tzinfo is None
                            else dt_naive.astimezone(timezone.utc)
                        )
                    profile_data["last_logged_in_dt"] = last_login_dt
                except (ValueError, TypeError) as date_parse_err:
                    logger.warning(
                        f"Could not parse LastLoginDate '{last_login_str}' for {profile_id}: {date_parse_err}"
                    )
                    profile_data["last_logged_in_dt"] = None
            else:
                profile_data["last_logged_in_dt"] = None

            return profile_data  # Return extracted data
        else:
            logger.warning(
                f"Failed to get valid /profiles/details response for {profile_id}. Type: {type(profile_response)}"
            )
            return None  # Return None on failure

    except ConnectionError as conn_err:
        logger.error(
            f"ConnectionError fetching /profiles/details for {profile_id}: {conn_err}",
            exc_info=False,
        )
        raise  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"Error fetching /profiles/details for {profile_id}: {e}", exc_info=True
        )
        if isinstance(e, requests.exceptions.RequestException):
            raise  # Re-raise for retry decorator
        return None  # Return None for other exceptions


# End _fetch_profile_details_for_person


# End of action7_inbox.py
