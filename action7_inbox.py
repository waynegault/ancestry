#!/usr/bin/env python3

# action7_inbox.py

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
from urllib.parse import urljoin

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
from config import config_instance
from database import (
    InboxStatus,
    Person,
    RoleType,
    create_person,  # Keep this import
    db_transn,
    # Removed get_person_by_profile_id_and_username as lookup logic is now internal
)
from utils import (
    _api_req,
    DynamicRateLimiter,
    SessionManager,
    retry,
    time_wait,
    retry_api,
)

# Initialize logging
logger = logging.getLogger("logger")


class InboxProcessor:
    """
    Processes Ancestry.co.uk inbox messages using the API, handling pagination
    and database interactions. Stores the user's role regarding the last message.
    Optimized to reduce API calls and defer database lookups.
    """

    def __init__(self, session_manager: SessionManager):  # Type hint SessionManager
        """Initializes InboxProcessor with session manager and rate limiter."""
        self.session_manager = session_manager
        self.dynamic_rate_limiter = DynamicRateLimiter()
        self.max_inbox_limit = config_instance.MAX_INBOX
        self.batch_size = config_instance.BATCH_SIZE
        self.progress_bar = None  # Initialize progress bar attribute

    # end of __init__

    @retry_api()
    def _get_all_conversations_api(
        self, session_manager: SessionManager, cursor: Optional[str] = None
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Retrieves a single batch of conversation overviews using cursor-based pagination via API."""
        # Still need driver for UBE header in _api_req, check if session seems valid
        if (
            not session_manager
            or not session_manager.is_sess_valid()
            or not session_manager.driver
        ):
            logger.error(
                "_get_all_conversations_api: SessionManager, valid session or WebDriver not available."
            )
            return None, None
        if not session_manager.my_profile_id:
            logger.error(
                "_get_all_conversations_api: my_profile_id not found in SessionManager."
            )
            return None, None

        my_profile_id = session_manager.my_profile_id
        api_base = urljoin(
            config_instance.BASE_URL, "/app-api/express/v2/"
        )  # Use the correct API base path
        # Use smaller batch size from config if MAX_INBOX is smaller and set
        limit = self.batch_size
        if self.max_inbox_limit > 0:
            limit = min(
                self.batch_size, self.max_inbox_limit + 5
            )  # Fetch slightly more than limit initially
        logger.debug(
            f"API call using limit: {limit} (Batch: {self.batch_size}, MaxInbox: {self.max_inbox_limit})"
        )

        all_conversations: List[Dict[str, Any]] = []

        # Construct the URL for the API endpoint
        url = f"{api_base}conversations?q=user:{my_profile_id}&limit={limit}"
        if cursor:
            url += f"&cursor={cursor}"
            logger.debug("Making next API call with cursor...")
        else:
            logger.debug("Making first API call...")

        try:
            # Use _api_req helper function
            response_data = _api_req(
                url=url,
                driver=session_manager.driver,  # Pass driver for potential UBE header
                session_manager=session_manager,
                method="GET",
                use_csrf_token=False,  # GET request for conversations likely doesn't need CSRF
                api_description="Get Inbox Conversations",
            )

            # Process the response data returned by _api_req
            if response_data is None:
                logger.error("API request via _api_req failed or returned None.")
                return None, None  # Indicate failure

            if not isinstance(response_data, dict):
                logger.error(
                    f"Unexpected API response format: Type {type(response_data)}. Expected dict."
                )
                logger.debug(f"Response data: {str(response_data)[:500]}...")
                return None, None

            conversations_data = response_data.get("conversations", [])

            if not conversations_data:
                logger.info("No conversations found in API response batch.")
                # Check if it's the end or just an empty batch
                forward_cursor = response_data.get("paging", {}).get("forward_cursor")
                return (
                    [],
                    forward_cursor,
                )  # Return empty list and cursor (could be None)

            for conv_data in conversations_data:
                # Pass my_profile_id to helper
                conversation_info = self._extract_conversation_info(
                    conv_data, my_profile_id
                )
                if conversation_info:
                    all_conversations.append(conversation_info)

            forward_cursor = response_data.get("paging", {}).get("forward_cursor")
            if forward_cursor:
                logger.debug(f"Forward cursor found: {forward_cursor[:10]}...")
                return all_conversations, forward_cursor
            else:
                logger.debug("No forward cursor found in API response.")
                return all_conversations, None

        except Exception as e:
            # Catch any unexpected errors during API call or processing
            logger.error(
                f"Unexpected error in _get_all_conversations_api: {e}", exc_info=True
            )
            return None, None  # Indicate failure

    # end of _get_all_conversations_api

    def _extract_conversation_info(
        self, conv_data: Dict[str, Any], my_profile_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Helper function to extract conversation information from the API response structure.
        Assigns 'my_role' based on the last message author (AUTHOR if I sent last, RECIPIENT if other sent last),
        using case-insensitive comparison.
        """
        if not isinstance(conv_data, dict):
            logger.warning("Invalid conversation data format received.")
            return None

        conversation_id = str(conv_data.get("id"))
        if not conversation_id:
            logger.warning("Conversation ID missing in API response.")
            return None

        last_message_data = conv_data.get("last_message", {})
        if not isinstance(last_message_data, dict):
            logger.warning(f"Invalid last_message format for conv {conversation_id}.")
            last_message_data = {}

        last_message_content = last_message_data.get("content", "")
        if not isinstance(last_message_content, str):
            logger.warning(
                f"Non-string content found for last message in conv {conversation_id}. Type: {type(last_message_content)}. Using empty string."
            )
            last_message_content = ""

        last_message_timestamp_unix = last_message_data.get("created")
        last_message_timestamp = None
        if isinstance(last_message_timestamp_unix, (int, float)):
            try:
                # Ensure timestamp is within valid range for utcfromtimestamp
                min_ts = 0
                max_ts = 32503680000  # Approx year 3000
                if min_ts <= last_message_timestamp_unix <= max_ts:
                    last_message_timestamp = datetime.utcfromtimestamp(
                        last_message_timestamp_unix
                    )
                else:
                    logger.warning(
                        f"Timestamp '{last_message_timestamp_unix}' out of reasonable range for conv {conversation_id}."
                    )
            except (TypeError, ValueError, OSError) as e:
                logger.warning(
                    f"Invalid timestamp '{last_message_timestamp_unix}' for conv {conversation_id}: {e}"
                )
        elif last_message_timestamp_unix is not None:
            logger.warning(
                f"Unexpected type for timestamp in conv {conversation_id}: {type(last_message_timestamp_unix)}"
            )

        last_message_author_id = last_message_data.get("author")
        my_role_value = RoleType.RECIPIENT

        # Ensure both IDs are strings before comparing
        author_id_str = str(last_message_author_id) if last_message_author_id else ""
        my_profile_id_str = str(my_profile_id) if my_profile_id else ""

        # Perform case-insensitive comparison
        if author_id_str.lower() == my_profile_id_str.lower():
            # If the last message author *is* me (case-insensitive), then I sent it -> My Role = AUTHOR
            my_role_value = RoleType.AUTHOR

        # --- Find Other Member Details ---
        username = "Unknown"
        profile_id = "UNKNOWN"  # Default if other member isn't found

        members = conv_data.get("members", [])
        if not isinstance(members, list):
            logger.warning(f"Invalid members format for conv {conversation_id}.")
            members = []

        other_member_found = False
        for member in members:
            if not isinstance(member, dict):
                logger.warning(f"Invalid member format within conv {conversation_id}.")
                continue
            member_user_id = member.get("user_id")
            member_username = member.get("display_name", "Unknown User")

            # Ensure IDs are strings for comparison
            member_user_id_str = str(member_user_id) if member_user_id else ""
            # my_profile_id_str already defined above

            # Case-insensitive check if this member is NOT me
            if (
                member_user_id_str
                and member_user_id_str.lower() != my_profile_id_str.lower()
            ):
                profile_id = (
                    member_user_id_str.upper()
                )  # Store uppercase Profile ID of the other person
                username = member_username  # Store display name of the other person
                other_member_found = True
                break  # Stop after finding the first other member

        if not other_member_found:
            logger.warning(
                f"Could not find other member details for conversation ID: {conversation_id}"
            )
            # Depending on requirements, might skip or proceed with defaults

        conversation_info = {
            "conversation_id": conversation_id,
            "profile_id": profile_id,  # The other person's Profile ID
            "username": username,  # The other person's username
            "last_message_content": last_message_content,
            "last_message_timestamp": last_message_timestamp,
            "my_role": my_role_value,  # My role relative to the last message
        }
        return conversation_info

    # end of _extract_conversation_info

    def search_inbox(self) -> bool:
        """
        V1.8 REVISED: Searches inbox using cursor pagination and batch processing via API.
        - Implements bulk Person lookup before processing each batch.
        - Calls modified _lookup_or_create_person with pre-fetched data.
        """
        # --- Initialize counters ---
        known_conversation_found = False
        new_records_saved = 0
        updated_records_saved = 0
        total_processed_api_items = 0
        items_processed_before_stop = 0  # Count items considered for processing
        stop_reason = ""
        # --- API/Loop variables ---
        next_cursor: Optional[str] = None
        current_batch_num = 0

        # --- Pre-checks ---
        if not self.session_manager:
            logger.error("search_inbox: SessionManager is missing. Aborting.")
            return False
        if not self.session_manager.my_profile_id:
            logger.error(
                "search_inbox: my_profile_id is missing in SessionManager. Aborting."
            )
            return False

        # --- Comparator ---
        most_recent_message = None
        try:
            with self.session_manager.get_db_conn_context() as db_conn_comp:
                if db_conn_comp:
                    most_recent_message = self._create_comparator(db_conn_comp)
                else:
                    logger.error(
                        "Failed to get DB connection for comparator. Aborting."
                    )
                    return False
        except Exception as comp_e:
            logger.error(
                f"Error getting DB connection or creating comparator: {comp_e}",
                exc_info=True,
            )
            return False

        # --- Progress Bar REMOVED ---

        # --- Main DB Session and Loop ---
        try:
            # Removed logging_redirect_tqdm context
            with self.session_manager.get_db_conn_context() as session:
                if not session:
                    print(
                        "ERROR: Failed to get DB connection for search loop. Aborting.",
                        file=sys.stderr,
                    )
                    # No progress bar to close
                    return False

                logger.info("Starting inbox search...")  # Log start explicitly

                while True:  # Outer loop for fetching batches
                    if known_conversation_found:
                        logger.debug(
                            f"Terminating inbox search loop PRE-FETCH. Reason: {stop_reason}"
                        )
                        break

                    all_conversations_batch, next_cursor_from_api = (
                        self._get_all_conversations_api(
                            self.session_manager, cursor=next_cursor
                        )
                    )

                    if all_conversations_batch is None:
                        logger.error(
                            "API call failed (_get_all_conversations_api returned None). Aborting inbox search."
                        )
                        return False  # Error occurred

                    batch_api_item_count = len(all_conversations_batch)
                    total_processed_api_items += batch_api_item_count
                    if batch_api_item_count > 0:
                        wait_duration = self.dynamic_rate_limiter.wait()
                        logger.debug(
                            f"Fetched batch {current_batch_num + 1} ({batch_api_item_count} items). Rate limit wait: {wait_duration:.2f}s"
                        )
                    else:  # Empty batch received
                        logger.debug("Received empty batch from API.")

                    if not all_conversations_batch:
                        if not next_cursor_from_api:
                            logger.info(
                                "Empty batch and no next cursor. Finishing inbox search."
                            )
                            stop_reason = "End of Inbox Reached"
                            known_conversation_found = True
                            break
                        else:
                            next_cursor = next_cursor_from_api
                            continue

                    current_batch_num += 1

                    # --- Optimization: Bulk Person Lookup ---
                    batch_profile_ids = {
                        conv.get("profile_id", "").upper()
                        for conv in all_conversations_batch
                        if conv.get("profile_id")
                        and conv.get("profile_id") != "UNKNOWN"
                    }
                    existing_persons_map: Dict[str, Person] = {}
                    if batch_profile_ids:
                        try:
                            logger.debug(
                                f"Performing bulk Person lookup for {len(batch_profile_ids)} profile IDs..."
                            )
                            existing_persons = (
                                session.query(Person)
                                .filter(Person.profile_id.in_(batch_profile_ids))
                                .all()
                            )
                            existing_persons_map = {
                                person.profile_id: person
                                for person in existing_persons
                                if person.profile_id
                            }
                            logger.debug(
                                f"Found {len(existing_persons_map)} existing Person records for batch."
                            )
                        except SQLAlchemyError as db_err:
                            logger.error(
                                f"Bulk Person lookup failed for batch {current_batch_num}: {db_err}",
                                exc_info=True,
                            )
                            # Decide how to handle - skip batch? Abort? For now, continue without pre-fetched data.
                            existing_persons_map = {}  # Ensure it's empty on error
                    # --- End Optimization ---

                    batch_data_to_save = []

                    for conversation_info in all_conversations_batch:
                        # Increment count *before* checks
                        items_processed_before_stop += 1

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
                                break  # Exit inner loop

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
                            break  # Exit inner loop

                        # Process Item (If checks passed)
                        # Log progress periodically instead of using progress bar
                        if items_processed_before_stop % 50 == 0:
                            logger.debug(
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

                        # --- Use pre-fetched person data ---
                        existing_person_for_item = existing_persons_map.get(
                            profile_id.upper()
                        )
                        person, person_status = self._lookup_or_create_person(
                            session,
                            profile_id,
                            username,
                            conversation_id,
                            existing_person_arg=existing_person_for_item,  # Pass pre-fetched data
                        )
                        # --- End Use pre-fetched person data ---

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
                    if batch_data_to_save:
                        # Check stop condition again *before* saving
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
                    else:  # No data and not stopped
                        logger.debug("No data prepared in batch to save.")

                    # --- Update Cursor and Check Outer Loop ---
                    next_cursor = next_cursor_from_api
                    if known_conversation_found:
                        break  # Exit outer loop if stop condition met
                    if not next_cursor:
                        logger.debug("No next cursor from API. Finishing inbox search.\n")
                        stop_reason = "End of Inbox Reached"
                        known_conversation_found = True  # Mark as finished cleanly
                        break  # Exit outer loop
                # --- End of main while loop ---
            # --- End of DB context manager ---

        except Exception as e:
            print(f"ERROR: Inbox search failed within main loop: {e}", file=sys.stderr)
            # Removed progress bar closing
            return False  # Indicate failure
        finally:
            # --- Cleaned Finally Block (No Progress Bar) ---
            if known_conversation_found and not stop_reason:
                stop_reason = "Unknown/Interrupted"
            _log_inbox_summary(
                total_api_items=total_processed_api_items,
                items_processed=items_processed_before_stop,
                new_records=new_records_saved,
                updated_records=updated_records_saved,
                stop_reason=stop_reason,
                max_inbox_limit=self.max_inbox_limit,
            )
            # --- End Cleaned Block ---

        return True  # Return True if loop finished normally or stopped cleanly

    # end of search_inbox

    def _create_comparator(self, session: DbSession) -> Optional[Dict[str, Any]]:
        """
        Creates comparator record using the provided Session.
        Includes profile_id for optimized checking.
        """
        most_recent_message = None
        try:
            comparator_inbox_status = (
                session.query(InboxStatus)
                .order_by(
                    InboxStatus.last_message_timestamp.desc().nullslast(),
                    InboxStatus.id.desc(),
                )
                .options(joinedload(InboxStatus.person))
                .first()
            )

            if comparator_inbox_status:
                comparator_person = comparator_inbox_status.person
                if (
                    comparator_person
                    and comparator_person.id is not None
                    and comparator_person.profile_id is not None
                ):
                    most_recent_message = {
                        "people_id": comparator_person.id,
                        "profile_id": comparator_person.profile_id,
                        "username": comparator_person.username,
                        "last_message_timestamp": comparator_inbox_status.last_message_timestamp,
                    }
                    ts_str = "None"
                    timestamp_val = most_recent_message.get("last_message_timestamp")
                    if isinstance(timestamp_val, datetime):
                        try:
                            ts_str = timestamp_val.isoformat() + "Z"
                        except ValueError:
                            ts_str = str(timestamp_val)
                    elif timestamp_val is not None:
                        ts_str = str(timestamp_val)
                    logger.debug(
                        f"Comparator created: {most_recent_message.get('username', 'N/A')} (Profile: {most_recent_message.get('profile_id')})"
                    )
                elif comparator_person and comparator_person.profile_id is None:
                    logger.warning(
                        f"Comparator error: Person ID {comparator_person.id} has no profile_id."
                    )
                elif comparator_person and comparator_person.id is None:
                    logger.warning(
                        f"Comparator error: Person object found but has no ID."
                    )
                elif not comparator_person:
                    logger.warning(
                        f"Comparator error: Associated Person not found for InboxStatus ID {comparator_inbox_status.id}."
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

    # end of _create_comparator

    def _lookup_or_create_person(
        self,
        session: DbSession,
        profile_id: str,
        username: str,
        conversation_id: Optional[str],
        existing_person_arg: Optional[Person] = None,  # New optional argument
    ) -> Tuple[Optional[Person], Literal["new", "skipped", "error", "updated"]]:
        """
        Looks up Person primarily by profile_id, creates if not found.
        Updates link and username if the found person's details differ.
        Uses pre-fetched existing_person_arg if provided.
        Returns the Person object and status. Ensures message_link is updated correctly.
        """
        if not profile_id or profile_id == "UNKNOWN":
            logger.warning(
                f"Cannot lookup person: invalid profile_id '{profile_id}' in conv {conversation_id}."
            )
            return None, "error"
        if not username:
            logger.warning(
                f"Missing username for profile_id {profile_id} in conv {conversation_id}. Using 'Unknown'."
            )
            username = "Unknown"

        profile_id_upper = profile_id.upper()
        username_lower = username.lower()
        correct_message_link: Optional[str] = None
        try:
            base_url = getattr(
                config_instance, "BASE_URL", "https://www.ancestry.co.uk/"
            )
            correct_message_link = urljoin(base_url, f"messaging/?p={profile_id_upper}")
        except Exception as url_e:
            logger.warning(
                f"Error constructing message link for profile {profile_id_upper}: {url_e}"
            )

        person: Optional[Person] = None
        lookup_needed = existing_person_arg is None  # Only lookup if not pre-fetched

        try:
            if lookup_needed:
                # Perform the lookup only if needed
                logger.debug(
                    f"Performing DB lookup for Person ProfileID='{profile_id_upper}'"
                )
                person = (
                    session.query(Person)
                    .filter(Person.profile_id == profile_id_upper)
                    .first()
                )
            else:
                # Use the pre-fetched person
                person = existing_person_arg
                # logger.debug(f"Using pre-fetched Person data for ProfileID='{profile_id_upper}'") # Less verbose

            if person:
                # Person exists, check if update is needed
                updated = False
                log_prefix = f"Person ID {person.id} ('{person.username}', Profile: {profile_id_upper})"

                # Check username
                if person.username.lower() != username_lower:
                    logger.debug(
                        f"{log_prefix}: Updating username from '{person.username}' to '{username}'."
                    )
                    person.username = username  # Update directly on the object
                    updated = True

                # Check message_link
                current_message_link_val = person.message_link
                if (
                    correct_message_link is not None
                    and current_message_link_val != correct_message_link
                ):
                    if current_message_link_val is None:
                        logger.debug(f"{log_prefix}: Adding missing message_link.")
                    else:
                        logger.debug(f"{log_prefix}: Updating message_link.")
                    person.message_link = correct_message_link
                    updated = True

                if updated:
                    person.updated_at = datetime.now()
                    # No need to query/update separately if modifying the object directly before commit
                    logger.debug(f"{log_prefix}: Record staged for update.")
                    return person, "updated"
                else:
                    # logger.debug(f"{log_prefix}: Found in 'people' table, no update needed.") # Less verbose
                    return person, "skipped"

            else:
                # Person does not exist, create new record
                logger.debug(
                    f"Creating Person for {username} (Profile: {profile_id_upper}) from conv {conversation_id}"
                )
                new_person = Person(
                    profile_id=profile_id_upper,
                    username=username,
                    message_link=correct_message_link,
                    in_my_tree=False,
                    status="active",
                    contactable=True,  # Default to contactable if creating from inbox? Or False? Let's assume True.
                    # created_at and updated_at have defaults
                )
                session.add(new_person)
                session.flush()  # Flush to get ID

                if new_person.id is None:
                    logger.error(
                        f"Person ID not assigned after flush for {username} (Profile: {profile_id_upper})! Conv {conversation_id}"
                    )
                    return None, "error"

                logger.debug(
                    f"{username} added to 'people' table with ID: {new_person.id}"
                )
                return new_person, "new"

        except SQLAlchemyError as e:
            logger.error(
                f"DB error in _lookup_or_create_person for {username} (Profile: {profile_id_upper}): {e}",
                exc_info=True,
            )
            return None, "error"
        except Exception as e:
            logger.critical(
                f"Unexpected error in _lookup_or_create_person for {username} (Profile: {profile_id_upper}): {e}",
                exc_info=True,
            )
            return None, "error"

    # end of _lookup_or_create_person

    def _save_batch(
        self, session: DbSession, batch: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Saves a batch of conversation data to the database using bulk operations.
        Maps 'my_role' variable to 'my_role' database column. Includes explicit checks
        before appending update items and uses cast for type checkers. Logs data before saving.
        """
        new_count, updated_count = 0, 0
        if not batch:
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
                if people_id is None:
                    logger.warning(f"Skipping item due to missing 'people_id': {item}")
                    continue

                my_role_enum_value = item.get("my_role")
                if not isinstance(my_role_enum_value, RoleType):
                    logger.warning(
                        f"Invalid role type '{my_role_enum_value}' for people_id {people_id}. Skipping item."
                    )
                    continue

                inbox_status = existing_statuses.get(people_id)

                if not inbox_status:
                    # Prepare data for a new InboxStatus record
                    new_item_data = {
                        "people_id": people_id,
                        "conversation_id": item.get("conversation_id"),
                        "my_role": my_role_enum_value,
                        "last_message": item.get("last_message_content", ""),
                        "last_message_timestamp": item.get("last_message_timestamp"),
                    }
                    new_items.append(new_item_data)
                else:
                    # Existing record found, check if update is needed
                    needs_update: bool = False
                    db_ts = cast(
                        Optional[datetime], inbox_status.last_message_timestamp
                    )
                    item_ts = item.get("last_message_timestamp")
                    if db_ts != item_ts:
                        needs_update = True

                    if not needs_update:
                        db_msg = cast(Optional[str], inbox_status.last_message)
                        item_msg = item.get("last_message_content", "")
                        if db_msg != item_msg:
                            needs_update = True

                    if not needs_update:
                        db_role = cast(Optional[RoleType], inbox_status.my_role)
                        if db_role != my_role_enum_value:
                            needs_update = True

                    if not needs_update:
                        db_conv_id = cast(Optional[str], inbox_status.conversation_id)
                        item_conv_id = item.get("conversation_id")
                        if db_conv_id != item_conv_id:
                            needs_update = True

                    if needs_update:
                        if inbox_status.id is not None:
                            update_item_data = {
                                "id": inbox_status.id,
                                "conversation_id": item.get("conversation_id"),
                                "my_role": my_role_enum_value,
                                "last_message": item.get("last_message_content", ""),
                                "last_message_timestamp": item.get(
                                    "last_message_timestamp"
                                ),
                            }
                            update_items.append(update_item_data)
                        else:
                            logger.warning(
                                f"Skipping update for people_id {people_id}: Existing InboxStatus found but has no ID."
                            )

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
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _save_batch: {e}", exc_info=True)
            raise

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
        logger.info(f"  Processing Stopped Due To: End of Inbox Reached")
    total_saved = new_records + updated_records
    logger.info("----------------------------")


# End of _log_inbox_summary

# End of action7_inbox.py
