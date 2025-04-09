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
    V14.14 REVISED: Initialize tqdm inside logging_redirect_tqdm context.
    - Moved `progress_bar = tqdm(...)` inside the `with` block.
    - Removed `print` before the `with` block.
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
    db_commit_batch_size = 10
    db_objects_to_commit: List[Any] = []
    progress_bar = None  # Define here for scope in finally
    total_potential = 0

    try:
        with session_manager.get_db_conn_context() as db_session:
            if not db_session:
                logger.error("Failed to get DB connection.")
                return False

            # --- Pre-fetching ---
            logger.debug("--- Starting Pre-fetching for Action 8 ---")
            potential_matches_query = (
                db_session.query(
                    DnaMatch.people_id,
                    DnaMatch.predicted_relationship,
                    Person.username,
                    Person.profile_id,
                    Person.in_my_tree,
                    Person.first_name,
                    FamilyTree.actual_relationship,
                    FamilyTree.person_name_in_tree,
                    FamilyTree.relationship_path,
                )
                .join(Person, DnaMatch.people_id == Person.id)
                .outerjoin(FamilyTree, Person.id == FamilyTree.people_id)
                .filter(
                    Person.profile_id.isnot(None),
                    Person.profile_id != "",
                    Person.profile_id != "UNKNOWN",
                )
                .filter(Person.status == "active")
                .filter(Person.contactable == True)
                .order_by(Person.id)
            )
            potential_matches = potential_matches_query.all()
            total_potential = len(potential_matches)
            logger.debug(
                f"Found {total_potential} potential CONTACTABLE DNA matches to check."
            )
            if not potential_matches:
                logger.info("No potential matches found meeting criteria. Finishing.")
                return True
            potential_person_ids = [
                match.people_id for match in potential_matches if match.people_id
            ]
            if not potential_person_ids:
                logger.info(
                    "No valid person IDs found among potential matches. Finishing."
                )
                return True

            logger.debug(
                f"Pre-fetching latest MessageHistory for {len(potential_person_ids)} people..."
            )
            latest_history_data: Dict[int, Tuple[str, datetime, str]] = {}
            try:
                row_number_subquery: Subquery = (
                    select(
                        MessageHistory.people_id,
                        MessageHistory.sent_at,
                        MessageType.type_name,
                        MessageHistory.status,
                        func.row_number()
                        .over(
                            partition_by=MessageHistory.people_id,
                            order_by=MessageHistory.sent_at.desc(),
                        )
                        .label("rn"),
                    )
                    .join(MessageType, MessageHistory.message_type_id == MessageType.id)
                    .where(MessageHistory.people_id.in_(potential_person_ids))
                    .subquery()
                )
                aliased_subquery = aliased(row_number_subquery, name="latest_msg")
                latest_history_query = select(
                    aliased_subquery.c.people_id,
                    aliased_subquery.c.type_name,
                    aliased_subquery.c.sent_at,
                    aliased_subquery.c.status,
                ).where(aliased_subquery.c.rn == 1)
                latest_history_results = db_session.execute(
                    latest_history_query
                ).fetchall()
                for row in latest_history_results:
                    person_id, type_name, sent_at_db, status = row
                    if sent_at_db:
                        sent_at_naive = (
                            sent_at_db.astimezone(timezone.utc).replace(tzinfo=None)
                            if sent_at_db.tzinfo
                            else sent_at_db.replace(tzinfo=None)
                        )
                        latest_history_data[person_id] = (
                            type_name,
                            sent_at_naive,
                            status,
                        )
                    else:
                        logger.warning(
                            f"Found MessageHistory for PersonID {person_id}, but sent_at is NULL."
                        )
                logger.debug(
                    f"Fetched latest MessageHistory data for {len(latest_history_data)} people."
                )
            except SQLAlchemyError as db_err:
                logger.error(
                    f"DB error pre-fetching MessageHistory: {db_err}", exc_info=True
                )
                return False
            except Exception as e:
                logger.error(
                    f"Error processing pre-fetched MessageHistory: {e}", exc_info=True
                )
                return False

            logger.debug(
                f"Pre-fetching InboxStatus for {len(potential_person_ids)} people..."
            )
            inbox_data: Dict[int, InboxStatus] = {}
            try:
                inbox_results = (
                    db_session.query(InboxStatus)
                    .filter(InboxStatus.people_id.in_(potential_person_ids))
                    .all()
                )
                inbox_data = {status.people_id: status for status in inbox_results}
                logger.debug(f"Fetched InboxStatus data for {len(inbox_data)} people.")
            except SQLAlchemyError as db_err:
                logger.error(
                    f"DB error pre-fetching InboxStatus: {db_err}", exc_info=True
                )
                return False
            except Exception as e:
                logger.error(
                    f"Error processing pre-fetched InboxStatus: {e}", exc_info=True
                )
                return False

            total_family_tree_rows = (
                db_session.query(func.count(FamilyTree.id)).scalar() or 0
            )
            logger.debug(
                f"Total rows found in family_tree table: {total_family_tree_rows}"
            )
            logger.debug("--- Pre-fetching Finished ---")
            logger.info(f"Processing {total_potential} matches...\n")
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
                    item_skipped = False
                    item_error = False
                    person_id: Optional[int] = None
                    username: Optional[str] = None
                    profile_id: Optional[str] = None
                    is_in_family_tree: bool = False
                    first_name: Optional[str] = None
                    person_name_in_tree: Optional[str] = None
                    original_predicted_relationship: Optional[str] = None
                    original_actual_relationship: Optional[str] = None
                    original_relationship_path: Optional[str] = None
                    last_message_sent_details: Optional[Tuple[str, datetime, str]] = (
                        None
                    )
                    last_sent_at_local: Optional[datetime] = None
                    last_received_at_local: Optional[datetime] = None
                    conversation_id: Optional[str] = None
                    they_sent_last: bool = False
                    inbox_role: Optional[RoleType] = None

                    if max_to_send != 0 and sent_count >= max_to_send:
                        logger.info(
                            f"Reached MAX_INBOX sending limit ({max_to_send}). Stopping."
                        )
                        stop_reason = f"Send Limit ({max_to_send})"
                        break

                    processed_count += 1

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
                        ) = match_data
                        if not person_id or not profile_id or not username:
                            raise ValueError("Missing essential person data")
                    except (ValueError, TypeError) as unpack_e:
                        logger.error(
                            f"Error unpacking data: {unpack_e}. Row: {match_data}",
                            exc_info=True,
                        )
                        error_count += 1
                        overall_success = False
                        item_error = True
                        if progress_bar:
                            progress_bar.set_postfix(
                                Sent=sent_count,
                                Skip=skipped_count,
                                Err=error_count,
                                refresh=False,
                            )
                            progress_bar.update(1)
                        continue

                    assert person_id is not None
                    assert profile_id is not None
                    assert username is not None
                    recipient_profile_id_upper = profile_id.upper()
                    log_prefix = f"{username} #{person_id}"
                    logger.debug(f"#### Processing {log_prefix} ####")

                    try:  # Inner try for this item's processing
                        now_local = datetime.now().replace(tzinfo=None)
                        last_message_sent_details = latest_history_data.get(person_id)
                        if last_message_sent_details:
                            last_sent_at_local = last_message_sent_details[1]
                        else:
                            last_sent_at_local = None
                        inbox_status = inbox_data.get(person_id)
                        if inbox_status:
                            conversation_id = inbox_status.conversation_id
                            inbox_role = inbox_status.my_role
                            received_at_db = inbox_status.last_message_timestamp
                            if received_at_db:
                                last_received_at_local = (
                                    received_at_db.astimezone(timezone.utc).replace(
                                        tzinfo=None
                                    )
                                    if received_at_db.tzinfo
                                    else received_at_db.replace(tzinfo=None)
                                )
                                if inbox_status.my_role == RoleType.RECIPIENT:
                                    they_sent_last = True
                            elif inbox_status.my_role == RoleType.RECIPIENT:
                                logger.warning(
                                    f"DB Consistency Issue: Inbox role RECIPIENT for {log_prefix}, but timestamp is NULL."
                                )
                                they_sent_last = False
                            else:
                                last_received_at_local = None
                        else:
                            conversation_id = None
                            inbox_role = None
                            last_received_at_local = None
                            they_sent_last = False

                        # Reply/Interval Checks
                        if they_sent_last and last_received_at_local:
                            if last_sent_at_local:
                                if last_received_at_local > last_sent_at_local:
                                    logger.debug(
                                        f"Skipping {log_prefix}: They replied."
                                    )
                                    skipped_count += 1
                                    item_skipped = True
                                    raise StopIteration
                            else:
                                logger.debug(
                                    f"Skipping {log_prefix}: They sent, no prior script message."
                                )
                                skipped_count += 1
                                item_skipped = True
                                raise StopIteration
                        if (
                            last_sent_at_local
                            and (now_local - last_sent_at_local) < MIN_MESSAGE_INTERVAL
                        ):
                            logger.debug(f"Skipping {log_prefix}: Within interval.")
                            skipped_count += 1
                            item_skipped = True
                            raise StopIteration

                        # Determine/Format Message
                        next_message_type_key = determine_next_message_type(
                            last_message_sent_details, bool(is_in_family_tree)
                        )
                        if not next_message_type_key:
                            logger.debug(
                                f"Skipping {log_prefix}: No next message type."
                            )
                            skipped_count += 1
                            item_skipped = True
                            raise StopIteration
                        message_template = MESSAGE_TEMPLATES.get(next_message_type_key)
                        if not message_template:
                            logger.error(
                                f"Missing template key '{next_message_type_key}'."
                            )
                            error_count += 1
                            overall_success = False
                            item_error = True
                            raise StopIteration
                        name_to_use = "Valued Relative"
                        source = "Fallback"
                        if is_in_family_tree and person_name_in_tree:
                            name_to_use = person_name_in_tree
                            source = "Tree"
                        elif first_name:
                            name_to_use = first_name
                            source = "Person.first"
                        elif username and username != "Unknown":
                            clean_user = (
                                username.replace("(managed by ", "")
                                .replace(")", "")
                                .strip()
                            )
                            if clean_user and clean_user != "Unknown User":
                                name_to_use = clean_user
                                source = "Person.user"
                            else:
                                source = "Person.user (unknown)"
                        formatted_name = format_name(name_to_use)
                        logger.debug(
                            f"Using name '{formatted_name}' (Source: {source}) for {log_prefix}"
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
                            req_keys = set(re.findall(r"\{(\w+)\}", message_template))
                        except Exception as regex_e:
                            logger.error(f"Regex err:{regex_e}")
                            req_keys = set(format_data.keys())
                        keys_using_fallback = []
                        critical_keys = {"name"}
                        critical_keys_failed = []
                        original_data_map = {
                            "predicted_relationship": original_predicted_relationship,
                            "actual_relationship": original_actual_relationship,
                            "relationship_path": original_relationship_path,
                            "name": name_to_use,
                        }
                        for key in req_keys:
                            original_value = original_data_map.get(key)
                            value_in_final_dict = format_data.get(key)
                            if value_in_final_dict == "N/A" and not original_value:
                                keys_using_fallback.append(key)
                            if key in critical_keys and (
                                value_in_final_dict == "N/A"
                                or value_in_final_dict == "Valued Relative"
                            ):
                                if not original_value or (
                                    key == "name" and name_to_use == "Valued Relative"
                                ):
                                    critical_keys_failed.append(
                                        f"{key} (original data missing/invalid)"
                                    )
                        if critical_keys_failed:
                            logger.error(
                                f"CRITICAL formatting keys failed for {log_prefix}: {critical_keys_failed}."
                            )
                            error_count += 1
                            overall_success = False
                            item_error = True
                            raise StopIteration
                        elif keys_using_fallback:
                            logger.warning(
                                f"Formatting {log_prefix}: Fallback used for keys: {keys_using_fallback}."
                            )
                        try:
                            message_text = message_template.format(**format_data)
                        except KeyError as e:
                            logger.error(f"KeyError fmt msg for {log_prefix}: {e}.")
                            error_count += 1
                            overall_success = False
                            item_error = True
                            raise StopIteration
                        except Exception as e:
                            logger.error(f"Error fmt msg for {log_prefix}: {e}.")
                            error_count += 1
                            overall_success = False
                            item_error = True
                            raise StopIteration
                        logger.debug(
                            f"Formatted message for {log_prefix}: '{message_text[:100]}...'"
                        )

                        # API Call Prep
                        message_status: str = "skipped (logic_error)"
                        new_conversation_id_from_api_local: Optional[str] = None
                        is_initial_message: bool = not conversation_id
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
                        elif conversation_id:
                            send_api_url = urljoin(
                                config_instance.BASE_URL.rstrip("/") + "/",
                                f"app-api/express/v2/conversations/{conversation_id}",
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

                        # Execute API Call
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
                                new_conversation_id_from_api_local = conversation_id
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
                            # API Response Validation
                            post_ok = False
                            api_conv_id: Optional[str] = None
                            api_author: Optional[str] = None
                            if api_response is not None:
                                if is_initial_message:
                                    if (
                                        isinstance(api_response, dict)
                                        and "conversation_id" in api_response
                                    ):
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
                                            post_ok = True
                                            new_conversation_id_from_api_local = (
                                                api_conv_id
                                            )
                                        else:
                                            logger.error(
                                                f"API initial response invalid for {log_prefix}."
                                            )
                                    else:
                                        logger.error(
                                            f"API call ({send_api_desc}) unexpected format (initial) for {log_prefix}."
                                        )
                                        logger.debug(f"Resp:{api_response}")
                                else:
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
                                                conversation_id
                                            )
                                            if (
                                                api_response.get("content")
                                                != message_text
                                            ):
                                                logger.warning(
                                                    f"API response content mismatch for {log_prefix}."
                                                )
                                        else:
                                            logger.error(
                                                f"API follow-up author validation failed for {log_prefix}."
                                            )
                                    else:
                                        logger.error(
                                            f"API call ({send_api_desc}) unexpected format (follow-up) for {log_prefix}."
                                        )
                                        logger.debug(f"Resp:{api_response}")
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

                        # DB Update Prep
                        if message_status in ("delivered OK", "typed (dry_run)"):
                            if (
                                message_status == "delivered OK"
                                or config_instance.APP_MODE == "dry_run"
                            ):
                                sent_count += 1
                            try:  # DB Prep Try
                                msg_type_obj = (
                                    db_session.query(MessageType)
                                    .filter(
                                        MessageType.type_name == next_message_type_key
                                    )
                                    .first()
                                )
                                if not msg_type_obj:
                                    raise ValueError(
                                        f"MessageType '{next_message_type_key}' not found."
                                    )
                                current_time_for_db = now_local
                                trunc_hist = message_text[
                                    : config_instance.MESSAGE_TRUNCATION_LENGTH
                                ]
                                new_history = MessageHistory(
                                    people_id=person_id,
                                    message_type_id=msg_type_obj.id,
                                    message_text=trunc_hist,
                                    status=message_status,
                                    sent_at=current_time_for_db,
                                )
                                db_objects_to_commit.append(new_history)
                                inbox_status_upd = inbox_data.get(person_id)
                                effective_conv_id = new_conversation_id_from_api_local
                                if not effective_conv_id:
                                    logger.error(
                                        f"DB Prep Error for {log_prefix}: Conv ID missing!"
                                    )
                                    error_count += 1
                                    overall_success = False
                                    item_error = True
                                    raise StopIteration
                                trunc_inbox = message_text[
                                    : config_instance.MESSAGE_TRUNCATION_LENGTH
                                ]
                                if inbox_status_upd:
                                    updated_inbox = False
                                    if inbox_status_upd.my_role != RoleType.AUTHOR:
                                        inbox_status_upd.my_role = RoleType.AUTHOR
                                        updated_inbox = True
                                    if inbox_status_upd.last_message != trunc_inbox:
                                        inbox_status_upd.last_message = trunc_inbox
                                        updated_inbox = True
                                    db_ts_naive_inbox = (
                                        inbox_status_upd.last_message_timestamp.astimezone(
                                            timezone.utc
                                        ).replace(
                                            tzinfo=None
                                        )
                                        if inbox_status_upd.last_message_timestamp
                                        and inbox_status_upd.last_message_timestamp.tzinfo
                                        else (
                                            inbox_status_upd.last_message_timestamp.replace(
                                                tzinfo=None
                                            )
                                            if inbox_status_upd.last_message_timestamp
                                            else None
                                        )
                                    )
                                    if db_ts_naive_inbox != current_time_for_db:
                                        inbox_status_upd.last_message_timestamp = (
                                            current_time_for_db
                                        )
                                        updated_inbox = True
                                    if (
                                        inbox_status_upd.conversation_id
                                        != effective_conv_id
                                    ):
                                        inbox_status_upd.conversation_id = (
                                            effective_conv_id
                                        )
                                        updated_inbox = True
                                        logger.debug(
                                            f"Updated Inbox Conv ID for {log_prefix}"
                                        )
                                    if updated_inbox:
                                        inbox_status_upd.last_updated = datetime.now()
                                        logger.debug(
                                            f"Staged InboxStatus update for {log_prefix}"
                                        )
                                else:
                                    new_inbox = InboxStatus(
                                        people_id=person_id,
                                        conversation_id=effective_conv_id,
                                        my_role=RoleType.AUTHOR,
                                        last_message=trunc_inbox,
                                        last_message_timestamp=current_time_for_db,
                                    )
                                    db_objects_to_commit.append(new_inbox)
                                    inbox_data[person_id] = new_inbox
                                    logger.debug(
                                        f"Staged new InboxStatus for {log_prefix}"
                                    )
                                # Commit periodically
                                if len(db_objects_to_commit) >= db_commit_batch_size:
                                    logger.debug(
                                        f"Reached batch size ({db_commit_batch_size}). Committing..."
                                    )
                                    try:
                                        db_session.add_all(db_objects_to_commit)
                                        db_session.flush()
                                        logger.debug("Flushed batch.")
                                        db_objects_to_commit = []
                                    except Exception as commit_err:
                                        logger.error(
                                            f"Error flushing batch: {commit_err}"
                                        )
                                        error_count += len(db_objects_to_commit)
                                        db_objects_to_commit = []
                                        overall_success = False
                                        item_error = True
                                        raise StopIteration
                            except ValueError as db_prep_val_err:
                                logger.error(
                                    f"DB prep ValueError for {log_prefix}: {db_prep_val_err}."
                                )
                                error_count += 1
                                overall_success = False
                                item_error = True
                                raise StopIteration
                            except SQLAlchemyError as db_prep_sql_err:
                                logger.error(
                                    f"DB prep SQLAlchemyError for {log_prefix}: {db_prep_sql_err}."
                                )
                                error_count += 1
                                overall_success = False
                                item_error = True
                                raise StopIteration
                            except Exception as db_prep_unexp:
                                logger.critical(
                                    f"Unexpected DB prep error for {log_prefix}: {db_prep_unexp}."
                                )
                                error_count += 1
                                overall_success = False
                                item_error = True
                                raise StopIteration

                    except StopIteration:
                        pass  # Graceful exit for skips/errors
                    except Exception as inner_loop_e:  # Catch unexpected errors
                        if isinstance(
                            inner_loop_e, TypeError
                        ) and "can't compare offset-naive and offset-aware datetimes" in str(
                            inner_loop_e
                        ):
                            logger.critical(
                                f"CRITICAL DATETIME COMPARISON ERROR for {log_prefix}: {inner_loop_e}",
                                exc_info=True,
                            )
                        elif isinstance(inner_loop_e, TypeError) and (
                            "replace() got an unexpected keyword argument 'tzinfo'"
                            in str(inner_loop_e)
                            or "astimezone() cannot be applied to a naive datetime"
                            in str(inner_loop_e)
                        ):
                            logger.critical(
                                f"CRITICAL DATETIME CONVERSION ERROR for {log_prefix}: {inner_loop_e}\n",
                                exc_info=True,
                            )
                        else:
                            logger.error(
                                f"Unexpected error processing {log_prefix}: {inner_loop_e}\n",
                                exc_info=True,
                            )
                        error_count += 1
                        overall_success = False
                        item_error = True

                    # ---> UPDATE PROGRESS BAR (Finally block of item processing) <---
                    finally:
                        if progress_bar:
                            progress_bar.set_postfix(
                                Sent=sent_count,
                                Skip=skipped_count,
                                Err=error_count,
                                refresh=False,
                            )
                            progress_bar.update(1)

                # --- End Main Loop ---

            # Commit remaining objects
            if db_objects_to_commit:
                logger.debug(
                    f"Committing final batch of {len(db_objects_to_commit)} DB objects..."
                )
                try:
                    db_session.add_all(db_objects_to_commit)
                    db_session.flush()
                    logger.debug("Final DB object batch staged for commit.")
                    db_objects_to_commit = []
                except Exception as commit_err:
                    logger.error(
                        f"Error committing final DB batch: {commit_err}", exc_info=True
                    )
                    error_count += len(db_objects_to_commit)
                    overall_success = False

            logger.debug("Finished processing all potential matches.\n")

    except Exception as outer_e:
        print(
            f"CRITICAL: Critical error during message processing loop: {outer_e}",
            file=sys.stderr,
        )
        if logger:
            logger.critical(
                f"CRITICAL: Critical error during message processing loop: {outer_e}",
                exc_info=True,
            )
        else:
            traceback.print_exc(file=sys.stderr)
        overall_success = False

    finally:
        if progress_bar:
            progress_bar.close()
            # Keep the print here to ensure newline after final bar state
            print("", file=sys.stderr)

        # Final Summary
        logger.info("--- Message Sending Summary ----")
        potential_count = total_potential
        logger.info(f"  Potential Matches Found: {potential_count}")
        logger.info(f"  Processed:             {processed_count}")
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
