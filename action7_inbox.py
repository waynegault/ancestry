#!/usr/bin/env python3

# action7_inbox.py

# Standard library imports
import inspect
import json
import logging
import math
import os
import random
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from urllib.parse import urljoin

# Third-party imports
import requests
from sqlalchemy import (Boolean, Column, DateTime, Enum as SQLEnum, Integer,
                        String, desc, func)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql.elements import ColumnElement

# Local application imports
from config import config_instance
from database import (InboxStatus, Person, RoleType, create_person, db_transn,
                      get_person_by_profile_id_and_username)
from utils import (_api_req, DynamicRateLimiter, SessionManager, retry,
                 time_wait)

# Initialize logging
logger = logging.getLogger("logger")

class InboxProcessor:
    """
    Processes Ancestry.co.uk inbox messages using the API, handling pagination
    and database interactions. Stores the user's role regarding the last message.
    """

    def __init__(self, session_manager: SessionManager): # Type hint SessionManager
        """Initializes InboxProcessor with session manager and rate limiter."""
        self.session_manager = session_manager
        self.dynamic_rate_limiter = DynamicRateLimiter()
        self.max_inbox_limit = config_instance.MAX_INBOX
        self.batch_size = config_instance.BATCH_SIZE
    #end of __init__

    @retry()
    def _get_all_conversations_api(self, session_manager: SessionManager, cursor: Optional[str] = None) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Retrieves a single batch of conversation overviews using cursor-based pagination via API."""
        if not session_manager or not session_manager.driver:
            logger.error("_get_all_conversations_api: SessionManager or WebDriver not available.")
            return None, None
        if not session_manager.my_profile_id:
            logger.error("_get_all_conversations_api: my_profile_id not found in SessionManager.")
            return None, None

        my_profile_id = session_manager.my_profile_id
        api_base = urljoin(config_instance.BASE_URL, "/app-api/express/v2/") # Use the correct API base path
        limit = self.batch_size
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
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                use_csrf_token=False, # GET request for conversations likely doesn't need CSRF
                api_description="Get Inbox Conversations" # Add a description if needed for headers
            )

            # Process the response data returned by _api_req
            if response_data is None:
                logger.error("API request via _api_req failed or returned None.")
                return None, None # Indicate failure

            if not isinstance(response_data, dict):
                logger.error(f"Unexpected API response format: Type {type(response_data)}. Expected dict.")
                logger.debug(f"Response data: {str(response_data)[:500]}...")
                return None, None

            conversations_data = response_data.get("conversations", [])

            if not conversations_data:
                logger.info("No conversations found in API response batch.")
                # Check if it's the end or just an empty batch
                forward_cursor = response_data.get("paging", {}).get("forward_cursor")
                return [], forward_cursor # Return empty list and cursor (could be None)


            for conv_data in conversations_data:
                # Pass my_profile_id to helper
                conversation_info = self._extract_conversation_info(conv_data, my_profile_id)
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
            logger.error(f"Unexpected error in _get_all_conversations_api: {e}", exc_info=True)
            return None, None # Indicate failure
    # end of _get_all_conversations_api

    def _extract_conversation_info(self, conv_data: Dict[str, Any], my_profile_id: str) -> Optional[Dict[str, Any]]:
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
            logger.warning(f"Non-string content found for last message in conv {conversation_id}. Type: {type(last_message_content)}. Using empty string.")
            last_message_content = ""

        last_message_timestamp_unix = last_message_data.get("created")
        last_message_timestamp = None
        if isinstance(last_message_timestamp_unix, (int, float)):
            try:
                last_message_timestamp = datetime.utcfromtimestamp(last_message_timestamp_unix)
            except (TypeError, ValueError, OSError) as e:
                logger.warning(f"Invalid timestamp '{last_message_timestamp_unix}' for conv {conversation_id}: {e}")
        elif last_message_timestamp_unix is not None:
             logger.warning(f"Unexpected type for timestamp in conv {conversation_id}: {type(last_message_timestamp_unix)}")


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
        profile_id = "UNKNOWN" # Default if other member isn't found

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
            if member_user_id_str and member_user_id_str.lower() != my_profile_id_str.lower():
                profile_id = member_user_id_str.upper() # Store uppercase Profile ID of the other person
                username = member_username # Store display name of the other person
                other_member_found = True
                break # Stop after finding the first other member

        if not other_member_found:
             logger.warning(f"Could not find other member details for conversation ID: {conversation_id}")
             # Depending on requirements, might skip or proceed with defaults

        conversation_info = {
            "conversation_id": conversation_id,
            "profile_id": profile_id, # The other person's Profile ID
            "username": username, # The other person's username
            "last_message_content": last_message_content,
            "last_message_timestamp": last_message_timestamp,
            "my_role": my_role_value, # My role relative to the last message
        }
        return conversation_info
    # end of _extract_conversation_info

    def search_inbox(self) -> bool:
        """
        Searches inbox using cursor pagination and batch processing via API.
        Stops processing immediately when the comparator message is found
        or the MAX_INBOX limit is reached. Provides clearer final summary logging.
        """
        # --- Initialize counters ---
        known_conversation_found = False
        new_records_saved = 0
        updated_records_saved = 0
        total_processed_api_items = 0 # How many items the API returned in total
        total_saved_or_updated = 0
        items_processed_before_stop = 0 # How many items were iterated over before stopping
        stop_reason = ""
        # --- API/Loop variables ---
        next_cursor: Optional[str] = None
        total_batches = -1
        current_batch_num = 0
        total_count = 0
        is_first_batch = True

        # --- Pre-checks ---
        if not self.session_manager:
            logger.error("search_inbox: SessionManager is missing. Aborting.")
            return False
        if not self.session_manager.my_profile_id:
             logger.error("search_inbox: my_profile_id is missing in SessionManager. Aborting.")
             return False

        # --- Comparator ---
        most_recent_message = None
        try:
             with self.session_manager.get_db_conn_context() as db_conn_comp:
                  if db_conn_comp:
                       most_recent_message = self._create_comparator(db_conn_comp)
                  else:
                       logger.error("Failed to get DB connection for comparator. Aborting.")
                       return False
        except Exception as comp_e:
             logger.error(f"Error getting DB connection or creating comparator: {comp_e}", exc_info=True)
             return False

        # --- Main DB Session and Loop ---
        try:
            with self.session_manager.get_db_conn_context() as session:
                if not session:
                    logger.error("Failed to get DB connection for search loop. Aborting.")
                    return False

                while True: # Outer loop for fetching batches
                    # --- Check termination conditions BEFORE fetching next batch ---
                    if known_conversation_found:
                        logger.debug(f"Terminating inbox search loop PRE-FETCH. Reason: {stop_reason}")
                        break # Exit outer while loop

                    # --- Fetch API Batch ---
                    all_conversations_batch, next_cursor_from_api = self._get_all_conversations_api(self.session_manager, cursor=next_cursor)

                    if all_conversations_batch is None:
                        logger.error("API call failed (_get_all_conversations_api returned None). Aborting inbox search.")
                        return False

                    batch_api_item_count = len(all_conversations_batch)
                    total_processed_api_items += batch_api_item_count
                    wait_duration = self.dynamic_rate_limiter.wait()
                    logger.debug(f"Dynamic rate limit wait: {wait_duration:.2f}s")

                    # --- Get Total Count (First Batch Only) ---
                    if is_first_batch:
                        # ... (logic to get total_count remains the same) ...
                        try:
                            first_page_url = f"{urljoin(config_instance.BASE_URL.rstrip('/') + '/', 'app-api/express/v2/')}conversations?q=user:{self.session_manager.my_profile_id}&limit=1"
                            first_page_data = _api_req(url=first_page_url, driver=self.session_manager.driver, session_manager=self.session_manager, method="GET", use_csrf_token=False, api_description="Get Inbox Total Count")
                            if first_page_data and isinstance(first_page_data, dict):
                                total_count = first_page_data.get("paging", {}).get("total_count", 0)
                                logger.info(f"API reports total conversation count: {total_count}\n")
                                if total_count > 0:
                                    if self.max_inbox_limit > 0:
                                         records_to_process_estimate = min(total_count, self.max_inbox_limit)
                                         logger.debug(f"Processing up to {records_to_process_estimate} conversations (Limit: {self.max_inbox_limit}).")
                                    else:
                                         records_to_process_estimate = total_count
                                         logger.info(f"Processing all {total_count} conversations (Limit: None).")

                                    total_batches = math.ceil(records_to_process_estimate / self.batch_size) if self.batch_size > 0 else 0
                                    logger.info(f"Estimated batches required: {total_batches}\n")
                                else:
                                    logger.info("API reports 0 total conversations. Finishing search.\n")
                                    total_batches = 0
                                    break
                            else:
                                logger.warning("Could not retrieve total_count from first API call. Batch calculation skipped.")
                                total_batches = -1
                        except Exception as count_e:
                             logger.error(f"Error getting total count: {count_e}", exc_info=True)
                             total_batches = -1
                        finally:
                            is_first_batch = False

                    # --- Handle Empty API Batch ---
                    if not all_conversations_batch:
                        # ... (logic remains the same) ...
                        logger.debug("Received empty batch from API.")
                        if not next_cursor_from_api:
                            logger.info("Empty batch and no next cursor. Finishing inbox search.")
                            break
                        else:
                            logger.debug("Empty batch but next cursor exists. Proceeding.")
                            next_cursor = next_cursor_from_api
                            continue


                    # --- Process Batch Items ---
                    current_batch_num += 1
                    batch_data_to_save = []
                    items_processed_this_batch = 0
                    logger.debug(f"Processing batch {current_batch_num}" + (f"/{total_batches}" if total_batches >= 0 else "") + f" ({batch_api_item_count} items from API)...")

                    # Inner loop iterates through fetched API conversations
                    for conversation_info in all_conversations_batch:
                        items_processed_this_batch += 1
                        current_item_overall_index = items_processed_before_stop + 1 # Index of the item being checked

                        # --- Process individual conversation data extraction ---
                        # ... (extract conversation_id, profile_id, username, etc.) ...
                        conversation_id = conversation_info.get('conversation_id')
                        profile_id = conversation_info.get('profile_id')
                        username = conversation_info.get('username', 'Username Not Available')
                        last_message_timestamp = conversation_info.get("last_message_timestamp") # datetime or None

                        if profile_id is None or profile_id == "UNKNOWN":
                            logger.warning(f"Profile ID is missing/unknown for conv ID: {conversation_id}. Skipping item {items_processed_this_batch}.")
                            continue

                        # --- Lookup/Create Person FIRST ---
                        person, person_status = self._lookup_or_create_person(session, profile_id, username, conversation_id)
                        if not person or person.id is None:
                            logger.error(f"Failed create/get Person/ID for {profile_id} in conv {conversation_id}. Skipping item {items_processed_this_batch}.")
                            continue
                        people_id = person.id

                        # --- Comparator Check SECOND ---
                        if most_recent_message:
                            comparator_people_id = most_recent_message.get("people_id")
                            comparator_username = most_recent_message.get("username")
                            comparator_timestamp = most_recent_message.get("last_message_timestamp")

                            people_id_match = (str(comparator_people_id) == str(people_id))
                            username_match = (comparator_username == username)
                            timestamps_match = False
                            if isinstance(comparator_timestamp, datetime) and isinstance(last_message_timestamp, datetime):
                                time_diff = abs((comparator_timestamp - last_message_timestamp).total_seconds())
                                timestamps_match = (time_diff < 1)
                            elif comparator_timestamp is None and last_message_timestamp is None:
                                timestamps_match = True

                            if people_id_match and username_match and timestamps_match:
                                logger.info(f"Comparator matched {username}. Stopping further processing.")
                                known_conversation_found = True
                                stop_reason = "Comparator Match"
                                items_processed_before_stop += 1 # Count the matched item itself
                                break # Exit inner for loop

                        # --- Limit Check THIRD ---
                        if self.max_inbox_limit != 0 and total_saved_or_updated >= self.max_inbox_limit:
                            logger.info(f"Inbox limit ({self.max_inbox_limit}) reached before processing item {current_item_overall_index}. Stopping further processing.")
                            known_conversation_found = True
                            stop_reason = f"Inbox Limit ({self.max_inbox_limit})"
                            # Do NOT increment items_processed_before_stop here, as the item wasn't processed due to limit
                            break # Exit inner for loop

                        # --- If neither stop condition met, increment counter & add item to save list ---
                        items_processed_before_stop += 1 # Increment only if not stopped
                        last_message_content = conversation_info.get("last_message_content", "")
                        last_message_content_truncated = (last_message_content[:97] + '...') if len(last_message_content) > 100 else last_message_content
                        my_role_enum_value = conversation_info.get("my_role")

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
                    if not known_conversation_found and batch_data_to_save:
                        new_in_batch, updated_in_batch = self._save_batch(session, batch_data_to_save)
                        new_records_saved += new_in_batch
                        updated_records_saved += updated_in_batch
                        total_saved_or_updated = new_records_saved + updated_records_saved
                        logger.debug(f"Batch {current_batch_num} saved ({new_in_batch} new, {updated_in_batch} updated statuses). Items processed this batch: {items_processed_this_batch}. Total saved/updated: {total_saved_or_updated}\n")
                    elif known_conversation_found:
                         logger.debug(f"Stop condition '{stop_reason}' met during batch {current_batch_num}. Skipping save for this batch.")
                    else: # No data added to save
                         logger.debug(f"No items collected in batch {current_batch_num} to save. Items processed this batch: {items_processed_this_batch}.")

                    # --- Update Cursor and Check Outer Loop ---
                    next_cursor = next_cursor_from_api

                    if known_conversation_found:
                        logger.debug(f"Outer loop termination check: Stop condition '{stop_reason}' met.")
                        break # Exit outer while loop

                    # Limit check after saving (redundant if inner check is robust, but safe)
                    if self.max_inbox_limit != 0 and total_saved_or_updated >= self.max_inbox_limit:
                         logger.debug(f"Inbox limit ({self.max_inbox_limit}) reached after batch {current_batch_num}. Terminating search.")
                         stop_reason = f"Inbox Limit ({self.max_inbox_limit})" # Ensure reason is set if hit here
                         known_conversation_found = True # Set flag
                         break

                    if not next_cursor:
                        logger.debug("No next cursor provided by API. Finishing search.")
                        break
                # --- End of main while loop ---

        except Exception as e:
            logger.error(f"Inbox search failed within main loop: {e}", exc_info=True)
            return False # Rollback handled by context manager
        finally:
            # --- Refined Final Summary Logging ---
            log_msg_parts = [f"Inbox search finished."]

            # Report API items only if > 0 batches were fetched
            if current_batch_num > 0 :
                log_msg_parts.append(f"Total API items processed: {total_processed_api_items}.")

            # Report stop reason and items processed before stop
            if stop_reason:
                log_msg_parts.append(f"Stopped due to: {stop_reason} after checking {items_processed_before_stop} conversations.")
            # If not stopped early and batches were processed, report completion
            elif current_batch_num > 0:
                 log_msg_parts.append(f"Completed checking {items_processed_before_stop} conversations (up to limit/end).")

            # Report saving results
            if total_saved_or_updated > 0:
                 log_msg_parts.append(f"Saved {new_records_saved} new, {updated_records_saved} updated statuses (Total: {total_saved_or_updated}).")
            # Specific messages for common "zero saved" scenarios
            elif stop_reason == "Comparator Match":
                 log_msg_parts.append("No new/updated conversations saved as comparator matched the first item checked.")
            elif stop_reason.startswith("Inbox Limit") and items_processed_before_stop <=1 : # <=1 because limit check happens *before* processing
                 log_msg_parts.append(f"No new/updated conversations saved as limit ({self.max_inbox_limit}) was reached before processing relevant items.")
            elif total_count == 0 and total_processed_api_items == 0:
                 log_msg_parts.append("API reported 0 total conversations.")
            elif total_processed_api_items > 0 and total_saved_or_updated == 0 and not stop_reason:
                 log_msg_parts.append(f"No new/updated conversations found needing DB update among the {items_processed_before_stop} processed.")
            elif total_processed_api_items == 0 and current_batch_num > 0: # Should not happen if API returns empty list correctly
                 log_msg_parts.append("Warning: No items received from API despite processing batches.")
            elif total_saved_or_updated == 0 and stop_reason: # Handles cases where limit/comparator hit but nothing happened to be saveable yet
                pass # Stop reason already covers it
            else: # Fallback for any other zero-save scenario
                 log_msg_parts.append("No new/updated conversations saved.")


            logger.info(" ".join(log_msg_parts))

        return True # Indicate successful completion if no exceptions occurred
    # end of search_inbox

    def _create_comparator(self, session: Session) -> Optional[Dict[str, Any]]:
        """Creates comparator record using the provided Session."""
        most_recent_message = None
        try:
            # Query optimized: Order by timestamp DESC (nulls last), then ID DESC
            comparator_inbox_status = (
                session.query(InboxStatus)
                .order_by(
                    InboxStatus.last_message_timestamp.desc().nullslast(),
                    InboxStatus.id.desc()
                  )
                # Eager load the associated Person to avoid a separate query later
                .options(joinedload(InboxStatus.person))
                .first()
            )

            if comparator_inbox_status:
                # Access the eager-loaded person object
                comparator_person = comparator_inbox_status.person

                if comparator_person and comparator_person.id is not None: # Ensure person and ID are valid
                    most_recent_message = {
                        "people_id": comparator_person.id,
                        "username": comparator_person.username, # Use username from person record
                        "last_message_timestamp": comparator_inbox_status.last_message_timestamp, # datetime or None
                    }
                    # Format timestamp safely for logging
                    ts_str = "None"
                    timestamp_val = most_recent_message.get('last_message_timestamp')
                    if isinstance(timestamp_val, datetime):
                         try:
                              # Use ISO format for clarity and include timezone info if available (UTC assumed here)
                              ts_str = timestamp_val.isoformat() + "Z" # Indicate UTC
                         except ValueError:
                              logger.warning(f"Comparator timestamp {timestamp_val} likely out of range for ISO format.")
                              ts_str = str(timestamp_val) # Use string representation
                    elif timestamp_val is not None:
                         ts_str = str(timestamp_val)

                    # Log comparator details at INFO level for better visibility
                    logger.info(f"Comparator created:{most_recent_message.get('username', 'N/A')}")
                elif comparator_person and comparator_person.id is None:
                     logger.warning(f"Comparator error: Found Person object for InboxStatus ID {comparator_inbox_status.id}, but Person has no ID.")
                elif not comparator_person:
                     logger.warning(f"Comparator error: InboxStatus record found (ID: {comparator_inbox_status.id}), but associated Person (people_id: {comparator_inbox_status.people_id}) could not be loaded/found.")


            else:
                logger.info("No existing inbox messages found. Comparator not created.")

        except SQLAlchemyError as e:
            logger.error(f"Database error creating comparator: {e}", exc_info=True)
            return None
        except Exception as e:
             logger.error(f"Unexpected error creating comparator: {e}", exc_info=True)
             return None

        return most_recent_message
    # end of _create_comparator

    def _lookup_or_create_person(self, session: Session, profile_id: str, username: str, conversation_id: Optional[str]) -> Tuple[Optional[Person], Literal["new", "skipped", "error", "updated"]]:
        """
        Looks up Person primarily by profile_id, creates if not found.
        Updates link and username if the found person's details differ.
        Returns the Person object and status. Ensures message_link is updated correctly.
        """
        if not profile_id or profile_id == "UNKNOWN":
            logger.warning(f"Cannot lookup person due to invalid profile_id '{profile_id}' in conv {conversation_id}. Skipping.")
            return None, "error"

        if not username:
            logger.warning(f"Missing username for profile_id {profile_id} in conv {conversation_id}. Using 'Unknown'.")
            username = "Unknown"

        profile_id_upper = profile_id.upper() # Ensure consistent case for lookup and link
        username_lower = username.lower() # For case-insensitive comparison

        # Construct the expected message link
        correct_message_link: Optional[str] = None
        try:
            # Ensure profile_id_upper is valid before constructing URL
            if profile_id_upper and profile_id_upper != "UNKNOWN":
                 # Use urljoin for robustness
                 base_url = getattr(config_instance, 'BASE_URL', 'https://www.ancestry.co.uk/')
                 correct_message_link = urljoin(base_url, f"messaging/?p={profile_id_upper}")
            else:
                logger.warning(f"Cannot construct message link, profile_id is invalid or UNKNOWN ({profile_id_upper}) for conv {conversation_id}.")
        except Exception as url_e:
            logger.warning(f"Error constructing message link for profile {profile_id_upper} in conv {conversation_id}: {url_e}")

        try:
            # --- MODIFIED LOOKUP: Primarily by profile_id ---
            person = session.query(Person).filter(
                Person.profile_id == profile_id_upper
            ).first()

            if person:
                # Person exists, check if update is needed
                updated = False
                log_prefix = f"Person ID {person.id} ('{person.username}', Profile: {profile_id_upper})"

                # Check and update username if different (case-insensitive)
                if person.username.lower() != username_lower:
                    logger.debug(f"{log_prefix}: Updating username from '{person.username}' to '{username}'.")
                    person.username = username
                    updated = True

                # Check and update message_link if different or missing
                if person.message_link != correct_message_link and correct_message_link is not None:
                    logger.debug(f"{log_prefix}: Updating message_link.")
                    person.message_link = correct_message_link
                    updated = True
                elif person.message_link is None and correct_message_link is not None:
                    logger.debug(f"{log_prefix}: Adding missing message_link.")
                    person.message_link = correct_message_link
                    updated = True


                # Ensure ID is valid before returning
                if person.id is None:
                    logger.error(f"Found {person.username} but their ID is None. Data inconsistency? Conv {conversation_id}")
                    return None, "error"

                if updated:
                    # Explicitly set updated_at timestamp
                    person.updated_at = datetime.now()
                    # No flush needed here, let _save_batch handle commit/rollback
                    logger.debug(f"{log_prefix}: Record staged for update.")
                    return person, "updated"
                else:
                    logger.debug(f"{log_prefix}: Found in 'people' table, no update needed.")
                    return person, "skipped"

            else:
                # Person does not exist, create new record
                logger.debug(f"Creating Person for {username} (Profile: {profile_id_upper}) from conv {conversation_id}")
                new_person = Person(
                    profile_id=profile_id_upper,
                    username=username,
                    message_link=correct_message_link, # Use constructed link
                    in_my_tree=False, # Default for inbox-created person
                    uuid=None, # Inbox doesn't provide UUID
                    status="active",
                    # created_at and updated_at have defaults
                )
                session.add(new_person)
                session.flush() # Flush to get ID and check constraints early

                # Critical check for ID assignment
                if new_person.id is None:
                    logger.error(f"Person ID not assigned after flush for {username} (Profile: {profile_id_upper})! Conversation {conversation_id}")
                    # Returning "error" will prevent this specific conversation from being saved in _save_batch
                    return None, "error"

                logger.debug(f"{username} added to 'people' table with ID: {new_person.id}")
                return new_person, "new"

        except SQLAlchemyError as e:
            logger.error(f"Database error in _lookup_or_create_person for {username} (Profile: {profile_id_upper}): {e}", exc_info=True)
            # Let the caller handle rollback if necessary
            return None, "error"
        except Exception as e: # Catch any other unexpected error
            logger.critical(f"Unexpected error in _lookup_or_create_person for {username} (Profile: {profile_id_upper}): {e}", exc_info=True)
            return None, "error"
    # end of _lookup_or_create_person


    def _save_batch(self, session: Session, batch: List[Dict[str, Any]]) -> Tuple[int, int]:
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
            people_ids_in_batch = [item["people_id"] for item in batch if "people_id" in item and item["people_id"] is not None]
            if not people_ids_in_batch:
                 logger.debug("Batch provided to _save_batch contains no valid people_ids.")
                 return 0, 0

            # Fetch existing InboxStatus records for the people in this batch
            existing_statuses = {
                status.people_id: status
                for status in session.query(InboxStatus).filter(InboxStatus.people_id.in_(people_ids_in_batch)).all()
            }
            logger.debug(f"Found {len(existing_statuses)} existing InboxStatus records for {len(people_ids_in_batch)} people_ids in batch.")


            for item in batch:
                people_id = item.get("people_id")
                # Check again for safety, although filtered above
                if people_id is None:
                    logger.warning(f"Skipping item due to missing 'people_id': {item}")
                    continue

                my_role_enum_value = item.get("my_role")

                # Ensure my_role is a valid RoleType enum member
                if not isinstance(my_role_enum_value, RoleType):
                    logger.warning(f"Invalid role type '{my_role_enum_value}' for people_id {people_id}. Skipping item.")
                    continue

                # Get corresponding existing status, if any
                inbox_status = existing_statuses.get(people_id)

                if not inbox_status:
                    # Prepare data for a new InboxStatus record
                    new_item_data = {
                        "people_id": people_id,
                        "conversation_id": item.get("conversation_id"),
                        "my_role": my_role_enum_value, # Use validated enum
                        "last_message": item.get("last_message_content", ""),
                        "last_message_timestamp": item.get("last_message_timestamp"),
                    }
                    new_items.append(new_item_data)
                    logger.debug(f"Prepared NEW item for people_id {people_id}: my_role={new_item_data['my_role']}")

                else:
                    # Existing record found, check if update is needed
                    needs_update: bool = False

                    # Compare last_message_timestamp (handle None safely)
                    db_ts = cast(Optional[datetime], inbox_status.last_message_timestamp)
                    item_ts = item.get("last_message_timestamp")
                    if db_ts != item_ts:
                        needs_update = True
                        logger.debug(f"InboxStatus update needed for people_id {people_id}: Timestamp differs ('{db_ts}' vs '{item_ts}').")

                    # Compare last_message_content if timestamp matches or is None
                    if not needs_update:
                        db_msg = cast(Optional[str], inbox_status.last_message)
                        item_msg = item.get("last_message_content", "")
                        if db_msg != item_msg:
                            needs_update = True
                            logger.debug(f"InboxStatus update needed for people_id {people_id}: Message differs.")

                    # Compare my_role if still no difference found
                    if not needs_update:
                        db_role = cast(Optional[RoleType], inbox_status.my_role) # Should be RoleType enum
                        if db_role != my_role_enum_value:
                             needs_update = True
                             logger.debug(f"InboxStatus update needed for people_id {people_id}: Role differs ('{db_role}' vs '{my_role_enum_value}').")

                    # Compare conversation_id if still no difference found
                    if not needs_update:
                        db_conv_id = cast(Optional[str], inbox_status.conversation_id)
                        item_conv_id = item.get("conversation_id")
                        if db_conv_id != item_conv_id:
                            needs_update = True
                            logger.debug(f"InboxStatus update needed for people_id {people_id}: Conversation ID differs.")


                    if needs_update:
                        # Ensure the existing status object has an ID before adding to update list
                        if inbox_status.id is not None:
                            update_item_data = {
                                # Map to the database columns
                                "id": inbox_status.id, # Primary key for update mapping
                                "conversation_id": item.get("conversation_id"),
                                "my_role": my_role_enum_value, # Use validated enum
                                "last_message": item.get("last_message_content", ""),
                                "last_message_timestamp": item.get("last_message_timestamp"),
                                # last_updated is handled by onupdate=datetime.now
                            }
                            update_items.append(update_item_data)
                            logger.debug(f"Prepared UPDATE item for people_id {people_id} (InboxStatus.id: {update_item_data['id']}): my_role={update_item_data['my_role']}")
                        else:
                            # This case should ideally not happen if records are fetched correctly
                            logger.warning(f"Skipping update for people_id {people_id}: Existing InboxStatus found but has no ID (maybe not flushed/committed yet?).")
                    #else:
                    #    logger.debug(f"No update needed for InboxStatus for people_id {people_id}.")


            # Perform bulk operations if there are items
            if new_items:
                logger.debug(f"Performing bulk insert for {len(new_items)} new InboxStatus items...")
                session.bulk_insert_mappings(InboxStatus, new_items)
                logger.debug(f"Bulk inserted {len(new_items)} items.")
                new_count = len(new_items)
            if update_items:
                logger.debug(f"Performing bulk update for {len(update_items)} existing InboxStatus items...")
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
            raise # Re-raise to trigger rollback in the context manager
        except Exception as e:
             logger.error(f"Unexpected error in _save_batch: {e}", exc_info=True)
             raise # Re-raise to trigger rollback in the context manager
    # end of _save_batch

# End of action7_inbox.py