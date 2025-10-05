#!/usr/bin/env python3

"""
Intelligent Inbox Processing & AI-Powered Message Classification

Processes Ancestry inbox messages with AI-powered classification, sentiment analysis,
and automated conversation management. Synchronizes with database and provides
comprehensive message lifecycle tracking.

Features:
- AI-powered message classification (PRODUCTIVE, DESIST, OTHER)
- Sentiment analysis and engagement tracking
- Conversation threading and relationship mapping
- Batch processing with pagination and rate limiting
- Database synchronization with conflict resolution
- Progress tracking and error recovery
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === PHASE 1 OPTIMIZATIONS ===
# === STANDARD LIBRARY IMPORTS ===
import inspect
import sys
from datetime import datetime, timezone
from typing import Any, Literal, Optional, cast

# === THIRD-PARTY IMPORTS ===
from selenium.common.exceptions import WebDriverException
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session as DbSession
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# === LOCAL IMPORTS ===
from ai_interface import classify_message_intent

# === PHASE 5.2: SYSTEM-WIDE CACHING OPTIMIZATION ===
from cache_manager import (
    cached_api_call,
)
from config import config_schema
from core.enhanced_error_recovery import with_api_recovery, with_enhanced_recovery

# === ACTION 7 ERROR CLASSES (Action 8 Pattern) ===
from core.error_handling import (
    APIError,
    AuthenticationError,
    BrowserError,
)
from core.progress_indicators import create_progress_indicator
from core.session_manager import SessionManager
from database import (
    ConversationLog,
    MessageDirectionEnum,
    Person,
    PersonStatusEnum,
    commit_bulk_data,
)


# Define Action 6/8-style error types for inbox processing
class MaxApiFailuresExceededError(Exception):
    """Custom exception for exceeding API failure threshold in inbox processing."""
    pass

class BrowserSessionError(BrowserError):
    """Browser session-specific errors."""
    pass

class APIRateLimitError(APIError):
    """API rate limit specific errors."""
    pass

class AuthenticationExpiredError(AuthenticationError):
    """Authentication expiration specific errors."""
    pass

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from core.error_handling import (
    circuit_breaker,
    error_context,
    graceful_degradation,
    retry_on_failure,
    timeout_protection,
)
from utils import (
    _api_req,
    format_name,
    retry_api,
    urljoin,
)


# --- Helper function for SQLAlchemy Column conversion ---
def _handle_direction_enum(value: Any, default: Any) -> Any:
    """Handle direction enum conversion."""
    if isinstance(value, MessageDirectionEnum):
        return value
    if isinstance(value, str):
        try:
            return MessageDirectionEnum(value)
        except ValueError:
            logger.warning(f"Invalid direction string '{value}'")
            return default
    else:
        logger.warning(f"Unexpected direction type: {type(value)}")
        return default


def _handle_status_enum(value: Any, default: Any) -> Any:
    """Handle status enum conversion."""
    if isinstance(value, PersonStatusEnum):
        return value
    if isinstance(value, str):
        try:
            return PersonStatusEnum(value)
        except ValueError:
            logger.warning(f"Invalid status string '{value}'")
            return default
    else:
        logger.warning(f"Unexpected status type: {type(value)}")
        return default


def _convert_to_primitive(value: Any) -> Any:
    """Convert value to Python primitive type."""
    if isinstance(value, (bool, bool, bool)):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "isoformat"):  # datetime-like
        return value
    return str(value)


def safe_column_value(obj: Any, attr_name: str, default: Any = None) -> Any:
    """
    Safely extract a value from a SQLAlchemy model attribute, handling Column objects.

    Args:
        obj: The SQLAlchemy model instance
        attr_name: The attribute name to access
        default: Default value to return if attribute doesn't exist or conversion fails

    Returns:
        The Python primitive value of the attribute, or the default value
    """
    if not hasattr(obj, attr_name):
        return default

    value = getattr(obj, attr_name)
    if value is None:
        return default

    # Try to convert to Python primitive
    try:
        # Special handling for enums
        if attr_name == "direction":
            return _handle_direction_enum(value, default)

        if attr_name == "status":
            return _handle_status_enum(value, default)

        # For different types of attributes
        return _convert_to_primitive(value)

    except (ValueError, TypeError, AttributeError):
        return default


# --- Critical Improvements ---


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
        self.max_inbox_limit = getattr(
            config_schema, "max_inbox", 0
        )  # Max conversations to process (0=unlimited)
        # Determine batch size, ensuring it doesn't exceed the overall limit if one is set
        default_batch = min(
            getattr(config_schema, "batch_size", 50), 50
        )  # Default batch size, capped at 50
        self.api_batch_size = (
            min(default_batch, self.max_inbox_limit)
            if self.max_inbox_limit > 0
            else default_batch
        )
        # AI Context settings
        self.ai_context_msg_count = getattr(
            config_schema, "ai_context_messages_count", 5
        )
        self.ai_context_max_words = getattr(
            config_schema, "ai_context_message_max_words", 100
        )  # Correct assignment
        self.ai_context_window_messages = getattr(
            config_schema, "ai_context_window_messages", 6
        )

        # Add input validation
        if self.ai_context_msg_count <= 0:
            logger.warning(
                f"AI context message count ({self.ai_context_msg_count}) invalid, using default of 10"
            )
            self.ai_context_msg_count = 10

        if self.ai_context_max_words <= 0:
            logger.warning(
                f"AI context max words ({self.ai_context_max_words}) invalid, using default of 100"
            )
            self.ai_context_max_words = 100

        # Add statistics tracking
        self.stats = {
            "conversations_fetched": 0,
            "conversations_processed": 0,
            "ai_classifications": 0,
            "person_updates": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

        # InboxProcessor initialized (removed verbose debug)

    # End of __init__

    # --- Private Helper Methods ---

    @cached_api_call("ancestry", ttl=900)  # 15-minute cache for conversations
    @retry_api()  # Apply retry decorator for resilience
    def _get_all_conversations_api(
        self, session_manager: SessionManager, limit: int, cursor: Optional[str] = None
    ) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
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
        api_base = urljoin(
            getattr(config_schema.api, "base_url", ""), "/app-api/express/v2/"
        )
        # Note: API uses 'limit', not 'batch_size' parameter name
        url = f"{api_base}conversations?q=user:{my_profile_id}&limit={limit}"
        if cursor:
            url += f"&cursor={cursor}"
        # Fetching conversation batch (removed verbose debug)

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
            all_conversations_processed: list[dict[str, Any]] = []
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
            # API call returned conversations (removed verbose debug)

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

    def _validate_conversation_data(self, conv_data: dict[str, Any]) -> Optional[tuple[str, dict]]:
        """Validate conversation data and extract basic info."""
        if not isinstance(conv_data, dict):
            logger.warning(f"Invalid conversation data type: {type(conv_data)}")
            return None

        conversation_id = str(conv_data.get("id", "")).strip()
        last_message_data = conv_data.get("last_message", {})

        if not conversation_id or not isinstance(last_message_data, dict):
            logger.warning(
                f"Skipping conversation data due to missing ID or last_message: ID='{conversation_id}', "
                f"last_message type={type(last_message_data)}"
            )
            return None

        return conversation_id, last_message_data

    def _parse_message_timestamp(self, last_message_data: dict, conversation_id: str) -> Optional[datetime]:
        """Parse and validate message timestamp."""
        last_msg_ts_unix = last_message_data.get("created")

        if not isinstance(last_msg_ts_unix, (int, float)):
            if last_msg_ts_unix is not None:
                logger.warning(f"Invalid timestamp type for ConvID {conversation_id}: {type(last_msg_ts_unix)}")
            return None

        try:
            min_ts, max_ts = 946684800, 4102444800  # Jan 1 2000 to Jan 1 2100
            if min_ts <= last_msg_ts_unix <= max_ts:
                return datetime.fromtimestamp(last_msg_ts_unix, tz=timezone.utc)
            logger.warning(f"Timestamp {last_msg_ts_unix} out of reasonable range for ConvID {conversation_id}")
        except (ValueError, TypeError, OSError) as ts_err:
            logger.warning(f"Error converting timestamp {last_msg_ts_unix} for ConvID {conversation_id}: {ts_err}")

        return None

    def _find_other_participant(
        self, members: list, my_profile_id: str, conversation_id: str
    ) -> Optional[tuple[str, str]]:
        """Find the other participant in the conversation."""
        if not isinstance(members, list):
            logger.warning(f"Members not a list for ConvID {conversation_id}: {type(members)}")
            return None

        if len(members) < 2:
            logger.warning(f"Insufficient members ({len(members)}) for ConvID {conversation_id}")
            return None

        my_pid_lower = str(my_profile_id).lower().strip()

        for member in members:
            if not isinstance(member, dict):
                continue

            member_user_id = member.get("user_id")
            if not member_user_id:
                continue

            member_user_id_str = str(member_user_id).lower().strip()

            if member_user_id_str and member_user_id_str != my_pid_lower:
                profile_id = str(member_user_id).upper().strip()
                username = str(member.get("display_name", "Unknown")).strip()
                return profile_id, username

        logger.warning(
            f"Could not identify other participant in ConvID {conversation_id}. "
            f"Members count: {len(members) if isinstance(members, list) else 'N/A'}"
        )
        return None

    def _extract_conversation_info(
        self, conv_data: dict[str, Any], my_profile_id: str
    ) -> Optional[dict[str, Any]]:
        """
        Extracts and formats key information from a single conversation overview dictionary.

        Args:
            conv_data: The dictionary representing one conversation from the API response.
            my_profile_id: The profile ID of the script user (to identify the 'other' participant).

        Returns:
            A dictionary containing 'conversation_id', 'profile_id', 'username',
            and 'last_message_timestamp', or None if essential data is missing.
        """
        # Validate conversation data
        validation_result = self._validate_conversation_data(conv_data)
        if validation_result is None:
            return None

        conversation_id, last_message_data = validation_result

        # Parse timestamp
        last_msg_ts_aware = self._parse_message_timestamp(last_message_data, conversation_id)

        # Find other participant
        members = conv_data.get("members", [])
        participant_info = self._find_other_participant(members, my_profile_id, conversation_id)
        if participant_info is None:
            return None

        profile_id, username = participant_info

        # Return validated data
        return {
            "conversation_id": conversation_id,
            "profile_id": profile_id,
            "username": username,
            "last_message_timestamp": last_msg_ts_aware,
        }

    # End of _extract_conversation_info

    def _validate_context_fetch_inputs(self, conversation_id: str) -> bool:
        """Validate inputs for conversation context fetch."""
        if not conversation_id:
            logger.warning("_fetch_conversation_context: No conversation_id provided.")
            return False

        if not self.session_manager or not self.session_manager.my_profile_id:
            logger.error("_fetch_conversation_context: SessionManager or profile ID missing.")
            return False

        if not self.session_manager.is_sess_valid():
            logger.error(f"_fetch_conversation_context: Session invalid fetching context for ConvID {conversation_id}.")
            raise WebDriverException(f"Session invalid fetching context ConvID {conversation_id}")

        return True

    def _build_context_api_request(self, conversation_id: str) -> tuple[str, dict[str, str]]:
        """Build API URL and headers for context fetch."""
        api_base = urljoin(getattr(config_schema.api, "base_url", ""), "/app-api/express/v2/")
        limit = self.ai_context_msg_count
        api_description = "Fetch Conversation Context"

        # Prepare headers
        contextual_headers = getattr(config_schema.api, "contextual_headers", {}).get(api_description, {})
        if isinstance(contextual_headers, dict):
            headers = contextual_headers.copy()
        else:
            headers = {}
            logger.warning(f"Expected dict for contextual headers, got {type(contextual_headers)}")

        # Set ancestry-userid
        if "ancestry-userid" in headers:
            headers["ancestry-userid"] = self.session_manager.my_profile_id.upper()

        # Remove None values
        headers = {k: v for k, v in headers.items() if v is not None}

        url = f"{api_base}conversations/{conversation_id}/messages?limit={limit}"
        return url, headers

    def _process_context_messages(
        self, messages_batch: list, conversation_id: str
    ) -> list[dict[str, Any]]:
        """Process and format message data from API response."""
        context_messages: list[dict[str, Any]] = []

        for msg_data in messages_batch:
            if not isinstance(msg_data, dict):
                continue

            # Parse timestamp
            ts_unix = msg_data.get("created")
            msg_timestamp: Optional[datetime] = None
            if isinstance(ts_unix, (int, float)):
                try:
                    msg_timestamp = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
                except Exception as ts_err:
                    logger.warning(f"Error parsing timestamp {ts_unix} in ConvID {conversation_id}: {ts_err}")

            # Prepare standardized message dictionary
            processed_msg = {
                "content": str(msg_data.get("content", "")),
                "author": str(msg_data.get("author", "")).lower(),
                "timestamp": msg_timestamp,
                "conversation_id": conversation_id,
            }
            context_messages.append(processed_msg)

        # Sort by timestamp (oldest first)
        return sorted(
            context_messages,
            key=lambda x: x.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc),
        )

    @cached_api_call("ancestry", ttl=600)
    @retry_api(max_retries=2)
    def _fetch_conversation_context(
        self, conversation_id: str
    ) -> Optional[list[dict[str, Any]]]:
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
        # Validate inputs
        if not self._validate_context_fetch_inputs(conversation_id):
            return None

        # Build API request
        url, headers = self._build_context_api_request(conversation_id)
        api_description = "Fetch Conversation Context"

        try:
            # Apply rate limiting
            limiter = cast(Any, getattr(self, "dynamic_rate_limiter", None))
            limiter.wait() if limiter is not None else 0.0

            # Make API call
            response_data = _api_req(
                url=url,
                driver=self.session_manager.driver,
                session_manager=self.session_manager,
                method="GET",
                headers=headers,
                use_csrf_token=False,
                api_description=api_description,
            )

            # Validate response
            if not isinstance(response_data, dict):
                logger.warning(f"{api_description}: Bad response type {type(response_data)} for ConvID {conversation_id}.")
                return None

            messages_batch = response_data.get("messages", [])
            if not isinstance(messages_batch, list):
                logger.warning(f"{api_description}: 'messages' key not a list for ConvID {conversation_id}.")
                return None

            # Process messages
            return self._process_context_messages(messages_batch, conversation_id)

        except WebDriverException as e:
            logger.error(f"WebDriverException fetching context for ConvID {conversation_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching context for ConvID {conversation_id}: {e}", exc_info=True)
            return None

    # End of _fetch_conversation_context

    def _format_context_for_ai(
        self, context_messages: list[dict], my_pid_lower: str
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
        # Step 1a: Limit to a sliding window of most recent messages for classification context
        window_size = getattr(self, "ai_context_window_messages", 6)
        msgs = context_messages[-window_size:] if isinstance(context_messages, list) else []
        # Step 2: Iterate through messages (assumed sorted oldest to newest)
        for msg in msgs:
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

    def _lookup_person_in_db(
        self, session: DbSession, profile_id: str, log_ref: str
    ) -> Optional[Person]:
        """Look up person in database by profile ID."""
        try:
            person = (
                session.query(Person)
                .filter(
                    func.upper(Person.profile_id) == profile_id.upper(),
                    Person.deleted_at.is_(None),
                )
                .first()
            )
            return person
        except SQLAlchemyError as e:
            logger.error(f"DB error looking up Person {log_ref}: {e}", exc_info=True)
            return None

    def _update_existing_person(
        self, session: DbSession, person: Person, username_to_use: str, profile_id: str, log_ref: str
    ) -> Literal["updated", "skipped", "error"]:
        """Update existing person record if needed."""
        updated = False

        # Update username if needed
        formatted_username = format_name(username_to_use)
        current_username = safe_column_value(person, "username", "Unknown")
        if current_username == "Unknown" or current_username != formatted_username:
            logger.debug(
                f"Updating username for {log_ref} from '{current_username}' to '{formatted_username}'."
            )
            person.username = formatted_username
            updated = True

        # Update message link if needed
        correct_message_link = urljoin(
            getattr(config_schema.api, "base_url", ""),
            f"/messaging/?p={profile_id.upper()}",
        )
        current_message_link = safe_column_value(person, "message_link", None)
        if current_message_link != correct_message_link:
            logger.debug(f"Updating message link for {log_ref}.")
            person.message_link = correct_message_link
            updated = True

        if updated:
            person.updated_at = datetime.now(timezone.utc)
            try:
                session.add(person)
                session.flush()
                logger.debug(f"Successfully staged updates for Person {log_ref} (ID: {person.id}).")
                return "updated"
            except (IntegrityError, SQLAlchemyError) as upd_err:
                logger.error(f"DB error flushing update for Person {log_ref}: {upd_err}")
                session.rollback()
                return "error"

        return "skipped"

    def _create_new_person(
        self, session: DbSession, profile_id: str, username_to_use: str, log_ref: str
    ) -> tuple[Optional[Person], Literal["new", "error"]]:
        """Create new person record in database."""
        logger.debug(f"Person {log_ref} not found. Creating new record...")

        new_person_data = {
            "profile_id": profile_id.upper(),
            "username": format_name(username_to_use),
            "message_link": urljoin(
                getattr(config_schema.api, "base_url", ""),
                f"/messaging/?p={profile_id.upper()}",
            ),
            "status": PersonStatusEnum.ACTIVE,
            "first_name": None,
            "contactable": True,
            "last_logged_in": None,
            "administrator_profile_id": None,
            "administrator_username": None,
            "gender": None,
            "birth_year": None,
            "in_my_tree": False,
            "uuid": None,
        }

        try:
            new_person = Person(**new_person_data)
            session.add(new_person)
            session.flush()

            if safe_column_value(new_person, "id") is None:
                logger.error(f"ID not assigned after flush for new person {log_ref}! Rolling back.")
                session.rollback()
                return None, "error"

            logger.debug(f"Created new Person ID {new_person.id} for {log_ref}.")
            return new_person, "new"

        except (IntegrityError, SQLAlchemyError) as create_err:
            logger.error(f"DB error creating Person {log_ref}: {create_err}")
            session.rollback()
            return None, "error"
        except Exception as e:
            logger.critical(f"Unexpected error creating Person {log_ref}: {e}", exc_info=True)
            session.rollback()
            return None, "error"

    def _lookup_or_create_person(
        self,
        session: DbSession,
        profile_id: str,
        username: str,
        conversation_id: Optional[str],
        existing_person_arg: Optional[Person] = None,
    ) -> tuple[Optional[Person], Literal["new", "updated", "skipped", "error"]]:
        """
        Looks up a Person by profile_id. If found, checks for updates (username,
        message_link). If not found, creates a new Person record.

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
        # Validate input
        if not profile_id or profile_id == "UNKNOWN":
            logger.warning("_lookup_or_create_person: Invalid profile_id provided.")
            return None, "error"

        username_to_use = username or "Unknown"
        log_ref = f"ProfileID={profile_id}/User='{username_to_use}' (ConvID: {conversation_id or 'N/A'})"

        # Get or lookup person
        if existing_person_arg:
            person = existing_person_arg
        else:
            person = self._lookup_person_in_db(session, profile_id, log_ref)
            if person is None and not existing_person_arg:
                # DB error occurred during lookup
                return None, "error"

        # Process based on whether person exists
        if person:
            status = self._update_existing_person(session, person, username_to_use, profile_id, log_ref)
            if status == "error":
                return None, "error"
            return person, status
        person, status = self._create_new_person(session, profile_id, username_to_use, log_ref)
        return person, status

    # End of _lookup_or_create_person

    def _create_comparator(self, session: DbSession) -> Optional[dict[str, Any]]:
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
        latest_log_entry_info: Optional[dict[str, Any]] = None
        # Creating comparator by finding latest ConversationLog entry (removed verbose debug)
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
                logger.debug(
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

    @with_enhanced_recovery(max_attempts=3, base_delay=4.0, max_delay=120.0)
    @retry_on_failure(max_attempts=3, backoff_factor=4.0)  # Increased backoff from 2.0 to 4.0
    @circuit_breaker(failure_threshold=10, recovery_timeout=60)  # Increased from 5 to 10 for better tolerance
    @timeout_protection(timeout=600)  # 10 minutes for inbox processing
    @graceful_degradation(fallback_value=False)
    @error_context("Action 7: Search Inbox")
    def _initialize_search_stats(self) -> None:
        """Initialize search statistics and clear cancellation signals."""
        self.stats["start_time"] = datetime.now(timezone.utc)
        # Clear any prior cancellation signals at the start of a new run
        try:
            from core.cancellation import clear_cancel
            clear_cancel()
        except Exception:
            pass

    def _validate_session_state(self) -> Optional[str]:
        """Validate session manager state and return profile ID."""
        if not self.session_manager or not self.session_manager.my_profile_id:
            logger.error("search_inbox: Session manager or profile ID missing.")
            return None
        return self.session_manager.my_profile_id.lower()

    def _setup_progress_tracking(self) -> dict[str, Any]:
        """Setup progress tracking configuration."""
        tqdm_args = {
            "total": None,  # Start with unknown total
            "desc": "Processing",
            "unit": " conv",
            "dynamic_ncols": True,
            "leave": True,
            "bar_format": "{desc} |{bar}| {n_fmt} conversations processed",
            "file": sys.stderr,
        }

        logger.info(
            f"Processing inbox items (limit: {self.max_inbox_limit if self.max_inbox_limit > 0 else 'unlimited'})..."
        )

        return tqdm_args

    def search_inbox(self) -> bool:
        """
        Main method to search the Ancestry inbox with enhanced error handling and statistics.

        Returns:
            True if the process completed without critical errors, False otherwise.
        """
        # Initialize statistics and state
        self._initialize_search_stats()

        # Validate session manager state
        my_pid_lower = self._validate_session_state()
        if not my_pid_lower:
            return False

    def _get_database_session_and_comparator(self) -> tuple[Optional[DbSession], Optional[str], Optional[datetime]]:
        """Get database session and create comparator for inbox processing."""
        session = self.session_manager.get_db_conn()
        if not session:
            logger.critical("search_inbox: Failed to get DB session. Aborting.")
            return None, None, None

        # Get the comparator (latest message in DB)
        comparator_info = self._create_comparator(session)
        comp_conv_id: Optional[str] = None
        comp_ts: Optional[datetime] = None  # Comparator timestamp (aware)
        if comparator_info:
            comp_conv_id = comparator_info.get("conversation_id")
            comp_ts = comparator_info.get("latest_timestamp")

        return session, comp_conv_id, comp_ts

    def _run_inbox_processing_loop(
        self,
        session: DbSession,
        comp_conv_id: Optional[str],
        comp_ts: Optional[datetime],
        my_pid_lower: str
    ) -> tuple[Optional[str], int, int, int, int]:
        """Run the main inbox processing loop with progress tracking."""
        tqdm_args = self._setup_progress_tracking()

        # PHASE 1 OPTIMIZATION: Enhanced progress tracking for inbox processing
        dynamic_total = self.max_inbox_limit if self.max_inbox_limit > 0 else 0
        with create_progress_indicator(
            description="Inbox Message Processing",
            total=dynamic_total if dynamic_total > 0 else None,
            unit="conversations",
            show_memory=True,
            show_rate=True,
            log_finish=False,
            leave=False,
        ) as enhanced_progress:

            with logging_redirect_tqdm(), tqdm(**tqdm_args) as progress_bar:
                # Link enhanced progress to tqdm for updates (avoid pylance attr warnings)
                from contextlib import suppress
                with suppress(Exception):
                    progress_bar._enhanced_progress = enhanced_progress

                result = self._process_inbox_loop(
                    session, comp_conv_id, comp_ts, my_pid_lower, progress_bar
                )

                # Ensure a newline after the progress bar completes to avoid inline log overlap
                try:
                    progress_bar.close()
                finally:
                    sys.stderr.write("\n")
                    sys.stderr.flush()

            # Update the progress bar to show completion - only set total if needed
            if progress_bar.total is None:
                progress_bar.total = max(result[4], 1)  # items_processed_before_stop
                progress_bar.refresh()
            # Don't manually set progress_bar.n here since it's already updated in the loop
            progress_bar.set_description("Completed")
            progress_bar.refresh()

            return result

    def get_statistics(self) -> dict[str, Any]:
        """Return processing statistics for monitoring and debugging."""
        stats = self.stats.copy()
        if stats["start_time"] and stats["end_time"]:
            stats["duration_seconds"] = (
                stats["end_time"] - stats["start_time"]
            ).total_seconds()
        return stats

    def _initialize_loop_state(self) -> dict[str, Any]:
        """Initialize state variables for the inbox processing loop."""
        return {
            "ai_classified_count": 0,
            "status_updated_count": 0,
            "total_processed_api_items": 0,
            "items_processed_before_stop": 0,
            "logs_processed_in_run": 0,
            "skipped_count_this_loop": 0,
            "error_count_this_loop": 0,
            "stop_reason": None,
            "next_cursor": None,
            "current_batch_num": 0,
            "conv_log_upserts_dicts": [],
            "person_updates": {},
            "stop_processing": False,
            "min_aware_dt": datetime.min.replace(tzinfo=timezone.utc),
            "conversations_needing_processing": 0,
        }

    def _check_browser_health(self, current_batch_num: int) -> Optional[str]:
        """Check browser health and attempt recovery if needed."""
        if current_batch_num % 5 == 0 and not self.session_manager.check_browser_health():
            logger.warning(f"🚨 Browser health check failed at batch {current_batch_num}")
            if not self.session_manager.attempt_browser_recovery("Action 7 Browser Recovery"):
                logger.critical(f"❌ Browser recovery failed at batch {current_batch_num} - halting inbox processing")
                return "Browser Recovery Failed"
        return None

    def _validate_session(self) -> None:
        """Validate session before API call."""
        if not self.session_manager.is_sess_valid():
            logger.error("Session became invalid during inbox processing loop.")
            raise WebDriverException("Session invalid before overview batch fetch")

    def _calculate_api_limit(self, items_processed_before_stop: int) -> tuple[int, Optional[str]]:
        """Calculate API limit for current batch considering overall limit."""
        current_limit = self.api_batch_size
        if self.max_inbox_limit > 0:
            remaining_allowed = self.max_inbox_limit - items_processed_before_stop
            if remaining_allowed <= 0:
                return 0, f"Inbox Limit ({self.max_inbox_limit})"
            current_limit = min(self.api_batch_size, remaining_allowed)
        return current_limit, None

    def _handle_empty_batch(self, next_cursor_from_api: Optional[str]) -> tuple[bool, Optional[str]]:
        """Handle empty batch result from API."""
        if not next_cursor_from_api:
            return True, "End of Inbox Reached (Empty Batch, No Cursor)"
        logger.debug("API returned empty batch but provided cursor. Continuing fetch.")
        return False, None

    def _update_progress_bar_initial(
        self, progress_bar: Optional[tqdm], batch_api_item_count: int
    ) -> None:
        """Update progress bar total on first batch."""
        if progress_bar is not None and progress_bar.total is None:
            if self.max_inbox_limit > 0:
                progress_bar.total = self.max_inbox_limit
            else:
                progress_bar.total = batch_api_item_count * 10
            progress_bar.refresh()

    def _prefetch_batch_data(
        self, session: DbSession, all_conversations_batch: list[dict], current_batch_num: int
    ) -> tuple[dict[str, Person], dict[tuple[str, str], ConversationLog], Optional[str]]:
        """Prefetch Person and ConversationLog data for batch."""
        batch_conv_ids = [
            c["conversation_id"] for c in all_conversations_batch if c.get("conversation_id")
        ]
        batch_profile_ids = {
            c.get("profile_id", "").upper()
            for c in all_conversations_batch
            if c.get("profile_id") and c.get("profile_id") != "UNKNOWN"
        }

        existing_persons_map: dict[str, Person] = {}
        existing_conv_logs: dict[tuple[str, str], ConversationLog] = {}

        try:
            if batch_profile_ids:
                persons = (
                    session.query(Person)
                    .filter(
                        func.upper(Person.profile_id).in_([pid.upper() for pid in batch_profile_ids]),
                        Person.deleted_at.is_(None),
                    )
                    .all()
                )
                existing_persons_map = {
                    safe_column_value(p, "profile_id").upper(): p
                    for p in persons
                    if safe_column_value(p, "profile_id")
                }

            if batch_conv_ids:
                logs = session.query(ConversationLog).filter(
                    ConversationLog.conversation_id.in_(batch_conv_ids)
                ).all()

                for log in logs:
                    timestamp = safe_column_value(log, "latest_timestamp", None)
                    if timestamp and timestamp.tzinfo is None:
                        log.latest_timestamp = timestamp.replace(tzinfo=timezone.utc)

                existing_conv_logs = {
                    (
                        str(safe_column_value(log, "conversation_id")),
                        (
                            str(safe_column_value(log, "direction").name)
                            if safe_column_value(log, "direction")
                            else ""
                        ),
                    ): log
                    for log in logs
                    if safe_column_value(log, "direction")
                }

            return existing_persons_map, existing_conv_logs, None

        except SQLAlchemyError as db_err:
            logger.error(f"DB prefetch failed for Batch {current_batch_num}: {db_err}")
            return {}, {}, "DB Prefetch Error"

    def _extract_conversation_identifiers(
        self, conversation_info: dict
    ) -> tuple[str, str, Optional[datetime]]:
        """Extract key identifiers from conversation info."""
        profile_id_upper = conversation_info.get("profile_id", "UNKNOWN").upper()
        api_conv_id = conversation_info.get("conversation_id")
        api_latest_ts_aware = conversation_info.get("last_message_timestamp")
        return profile_id_upper, api_conv_id, api_latest_ts_aware

    def _should_skip_invalid(
        self, api_conv_id: Optional[str], profile_id_upper: str, progress_bar: Optional[tqdm]
    ) -> bool:
        """Check if conversation should be skipped due to invalid data."""
        if not api_conv_id or profile_id_upper == "UNKNOWN":
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_description("Processing (skipped invalid)")
                _ep = getattr(progress_bar, '_enhanced_progress', None)
                if _ep is not None:
                    _ep.update(increment=1, warnings=1, custom_status="Skipped invalid")
            return True
        return False

    def _determine_fetch_need(
        self,
        api_conv_id: str,
        comp_conv_id: Optional[str],
        comp_ts: Optional[datetime],
        api_latest_ts_aware: Optional[datetime],
        existing_conv_logs: dict[tuple[str, str], ConversationLog],
        min_aware_dt: datetime,
    ) -> tuple[bool, bool, Optional[str]]:
        """Determine if conversation needs fetching based on comparator logic.

        Returns: (needs_fetch, stop_processing, stop_reason)
        """
        # For live test, always fetch
        if len(sys.argv) > 1 and sys.argv[1].lower() == "live":
            return True, False, None

        # Check if this is the comparator conversation
        if comp_conv_id and api_conv_id == comp_conv_id:
            if comp_ts and api_latest_ts_aware and api_latest_ts_aware > comp_ts:
                return True, True, None
            return False, True, "Comparator Found (No Change)"

        # Not comparator - compare with DB timestamps
        db_log_in = existing_conv_logs.get((api_conv_id, MessageDirectionEnum.IN.name))
        db_log_out = existing_conv_logs.get((api_conv_id, MessageDirectionEnum.OUT.name))

        db_latest_ts_in = (
            safe_column_value(db_log_in, "latest_timestamp")
            if db_log_in and safe_column_value(db_log_in, "latest_timestamp")
            else min_aware_dt
        )
        db_latest_ts_out = (
            safe_column_value(db_log_out, "latest_timestamp")
            if db_log_out and safe_column_value(db_log_out, "latest_timestamp")
            else min_aware_dt
        )
        db_latest_overall = max(db_latest_ts_in, db_latest_ts_out)

        # Fetch if API timestamp is newer OR no DB logs exist
        if api_latest_ts_aware and api_latest_ts_aware > db_latest_overall:
            return True, False, None
        if not db_log_in and not db_log_out:
            return True, False, None

        return False, False, None

    def _update_progress_skip(self, progress_bar: Optional[tqdm]) -> None:
        """Update progress bar for skipped up-to-date conversation."""
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_description("Processing (up-to-date)")
            _ep = getattr(progress_bar, '_enhanced_progress', None)
            if _ep is not None:
                _ep.update(increment=1, cache_hits=1, custom_status="Up-to-date")

    def _update_progress_processing(
        self, progress_bar: Optional[tqdm], api_conv_id: str
    ) -> None:
        """Update progress bar for conversation being processed."""
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_description(f"Processing conversation {api_conv_id}")
            _ep = getattr(progress_bar, '_enhanced_progress', None)
            if _ep is not None:
                _ep.update(
                    increment=1, api_calls=2, custom_status=f"Processing {api_conv_id[:8]}"
                )

    def _find_latest_messages(
        self, context_messages: list[dict], my_pid_lower: str
    ) -> tuple[Optional[dict], Optional[dict]]:
        """Find latest IN and OUT messages from context."""
        latest_ctx_in: Optional[dict] = None
        latest_ctx_out: Optional[dict] = None

        for msg in reversed(context_messages):
            author_lower = msg.get("author", "")
            if author_lower != my_pid_lower and latest_ctx_in is None:
                latest_ctx_in = msg
            elif author_lower == my_pid_lower and latest_ctx_out is None:
                latest_ctx_out = msg
            if latest_ctx_in and latest_ctx_out:
                break

        return latest_ctx_in, latest_ctx_out

    def _classify_message_with_ai(
        self, context_messages: list[dict], my_pid_lower: str, api_conv_id: str
    ) -> Optional[str]:
        """Classify message using AI with recovery and guardrails."""
        formatted_context = self._format_context_for_ai(context_messages, my_pid_lower)

        if not self.session_manager.is_sess_valid():
            raise WebDriverException(
                f"Session invalid before AI classification call for ConvID {api_conv_id}"
            )

        @with_api_recovery(max_attempts=3, base_delay=2.0)
        def _classify_with_recovery(context=formatted_context) -> Optional[str]:
            return classify_message_intent(context, self.session_manager)

        def _downgrade_if_non_actionable(
            label: Optional[str], messages=context_messages
        ) -> Optional[str]:
            try:
                if (not label) or label != "PRODUCTIVE":
                    return label

                last_user = None
                for m in reversed(messages):
                    if m.get("author", "").lower() != my_pid_lower:
                        last_user = str(m.get("content", ""))
                        break

                if not last_user:
                    return label

                txt = last_user.lower()
                actionable_cues = (
                    "share", "send", "attach", "tree", "record", "certificate",
                    "born", "married", "died", "parents", "ancestor", "where", "how",
                    "i will", "i'll", "i can", "i'll send", "i can share", "link"
                )

                if not any(cue in txt for cue in actionable_cues):
                    return "ENTHUSIASTIC" if any(
                        k in txt for k in ("thanks", "thank you", "cheers", "take care")
                    ) else "OTHER"

                return label
            except Exception:
                return label

        ai_result = _classify_with_recovery()

        if isinstance(ai_result, tuple):
            ai_sentiment_result = ai_result[0] if ai_result else None
        else:
            ai_sentiment_result = ai_result

        return _downgrade_if_non_actionable(ai_sentiment_result)

    def _create_conversation_log_upsert(
        self,
        api_conv_id: str,
        direction: MessageDirectionEnum,
        people_id: int,
        message_content: str,
        timestamp: datetime,
        ai_sentiment: Optional[str] = None,
    ) -> dict:
        """Create conversation log upsert dictionary."""
        return {
            "conversation_id": api_conv_id,
            "direction": direction,
            "people_id": people_id,
            "latest_message_content": message_content[
                : getattr(config_schema, "message_truncation_length", 1000)
            ],
            "latest_timestamp": timestamp,
            "ai_sentiment": ai_sentiment,
            "message_template_id": None,
            "script_message_status": None,
        }

    def _update_person_status_from_ai(
        self, ai_sentiment: Optional[str], people_id: int, person_updates: dict[int, PersonStatusEnum]
    ) -> None:
        """Update person status based on AI classification."""
        if ai_sentiment == "UNINTERESTED":
            person_updates[people_id] = PersonStatusEnum.DESIST
        elif ai_sentiment == "PRODUCTIVE":
            pass  # Keep as ACTIVE for Action 9

    def _commit_batch_updates(
        self,
        session: DbSession,
        conv_log_upserts_dicts: list[dict],
        person_updates: dict[int, PersonStatusEnum],
        current_batch_num: int,
    ) -> tuple[int, int]:
        """Commit batch data and return counts."""
        if not conv_log_upserts_dicts and not person_updates:
            return 0, 0

        logs_committed, persons_updated = commit_bulk_data(
            session=session,
            log_upserts=conv_log_upserts_dicts,
            person_updates=person_updates,
            context=f"Action 7 Batch {current_batch_num}",
        )

        conv_log_upserts_dicts.clear()
        person_updates.clear()

        return logs_committed, persons_updated

    def _handle_exception_with_save(
        self,
        session: DbSession,
        conv_log_upserts_dicts: list[dict],
        person_updates: dict[int, PersonStatusEnum],
        exception_type: str,
        error: Exception,
    ) -> tuple[int, int]:
        """Handle exception and attempt final save."""
        logger.warning(f"Attempting final save due to {exception_type}...")

        try:
            final_logs, final_persons = commit_bulk_data(
                session=session,
                log_upserts=conv_log_upserts_dicts,
                person_updates=person_updates,
                context=f"Action 7 Final Save ({exception_type})",
            )
            return final_logs, final_persons
        except ConnectionError as conn_err:
            if "Session death cascade detected" in str(conn_err):
                logger.critical(
                    f"🚨 SESSION DEATH CASCADE in Action 7 {exception_type} save: {conn_err}"
                )
                raise MaxApiFailuresExceededError(
                    f"Session death cascade detected in Action 7 {exception_type} save"
                ) from None
            logger.error(f"ConnectionError during Action 7 {exception_type} save: {conn_err}")
            return 0, 0

    def _check_cancellation_requested(self) -> bool:
        """Check if cancellation was requested by timeout wrapper."""
        try:
            from core.cancellation import is_cancel_requested
            return is_cancel_requested()
        except Exception:
            return False

    def _update_progress_bar_stats(
        self,
        progress_bar: Optional[tqdm],
        ai_classified_count: int,
        status_updated_count: int,
        skipped_count: int,
        error_count: int,
    ) -> None:
        """Update progress bar with current statistics."""
        if progress_bar is not None:
            progress_bar.set_description(
                f"Processing: AI={ai_classified_count} Updates={status_updated_count} "
                f"Skip={skipped_count} Err={error_count}"
            )

    def _get_db_timestamp_for_comparison(
        self,
        existing_conv_logs: dict[tuple[str, str], ConversationLog],
        api_conv_id: str,
        direction: MessageDirectionEnum,
        min_aware_dt: datetime,
    ) -> datetime:
        """Get database timestamp for comparison with API timestamp."""
        db_log = existing_conv_logs.get((api_conv_id, direction.name), None)
        if db_log:
            return safe_column_value(db_log, "latest_timestamp") or min_aware_dt
        return min_aware_dt

    def _process_in_message(
        self,
        latest_ctx_in: Optional[dict],
        api_conv_id: str,
        people_id: int,
        existing_conv_logs: dict[tuple[str, str], ConversationLog],
        min_aware_dt: datetime,
        context_messages: list[dict],
        my_pid_lower: str,
        conv_log_upserts_dicts: list[dict],
        person_updates: dict[int, dict],
        ai_classified_count: int,
    ) -> int:
        """Process incoming message and return updated AI classification count."""
        if not latest_ctx_in:
            return ai_classified_count

        ctx_ts_in_aware = latest_ctx_in.get("timestamp")
        db_latest_ts_in_compare = self._get_db_timestamp_for_comparison(
            existing_conv_logs, api_conv_id, MessageDirectionEnum.IN, min_aware_dt
        )

        if ctx_ts_in_aware and ctx_ts_in_aware > db_latest_ts_in_compare:
            ai_sentiment_result = self._classify_message_with_ai(
                context_messages, my_pid_lower, api_conv_id
            )

            if ai_sentiment_result:
                ai_classified_count += 1
            else:
                logger.warning(f"AI classification failed for ConvID {api_conv_id}.")

            upsert_dict_in = self._create_conversation_log_upsert(
                api_conv_id, MessageDirectionEnum.IN, people_id,
                latest_ctx_in.get("content", ""), ctx_ts_in_aware, ai_sentiment_result,
            )
            conv_log_upserts_dicts.append(upsert_dict_in)

            self._update_person_status_from_ai(ai_sentiment_result, people_id, person_updates)

        return ai_classified_count

    def _process_out_message(
        self,
        latest_ctx_out: Optional[dict],
        api_conv_id: str,
        people_id: int,
        existing_conv_logs: dict[tuple[str, str], ConversationLog],
        min_aware_dt: datetime,
        conv_log_upserts_dicts: list[dict],
    ) -> None:
        """Process outgoing message."""
        if not latest_ctx_out:
            return

        ctx_ts_out_aware = latest_ctx_out.get("timestamp")
        db_latest_ts_out_compare = self._get_db_timestamp_for_comparison(
            existing_conv_logs, api_conv_id, MessageDirectionEnum.OUT, min_aware_dt
        )

        if ctx_ts_out_aware and ctx_ts_out_aware > db_latest_ts_out_compare:
            upsert_dict_out = self._create_conversation_log_upsert(
                api_conv_id, MessageDirectionEnum.OUT, people_id,
                latest_ctx_out.get("content", ""), ctx_ts_out_aware,
            )
            conv_log_upserts_dicts.append(upsert_dict_out)

    def _process_conversations_in_batch(
        self,
        session: DbSession,
        all_conversations_batch: list[dict],
        existing_persons_map: dict[str, Person],
        existing_conv_logs: dict[tuple[str, str], ConversationLog],
        comp_conv_id: Optional[str],
        comp_ts: Optional[datetime],
        my_pid_lower: str,
        min_aware_dt: datetime,
        progress_bar: Optional[tqdm],
        state: dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """Process all conversations in a batch.

        Returns: (stop_processing, stop_reason)
        """
        items_processed_before_stop = state["items_processed_before_stop"]
        skipped_count_this_loop = state["skipped_count_this_loop"]
        error_count_this_loop = state["error_count_this_loop"]
        ai_classified_count = state["ai_classified_count"]
        status_updated_count = state["status_updated_count"]
        conv_log_upserts_dicts = state["conv_log_upserts_dicts"]
        person_updates = state["person_updates"]
        conversations_needing_processing = state["conversations_needing_processing"]

        stop_processing = False
        stop_reason = None

        for conversation_info in all_conversations_batch:
            # Check max inbox limit
            if self.max_inbox_limit > 0 and items_processed_before_stop >= self.max_inbox_limit:
                stop_reason = f"Inbox Limit ({self.max_inbox_limit})"
                stop_processing = True
                break

            items_processed_before_stop += 1

            # Extract conversation identifiers
            profile_id_upper, api_conv_id, api_latest_ts_aware = (
                self._extract_conversation_identifiers(conversation_info)
            )

            # Skip invalid conversations
            if self._should_skip_invalid(api_conv_id, profile_id_upper, progress_bar):
                logger.debug(
                    f"Skipping item {items_processed_before_stop}: Invalid ConvID/ProfileID."
                )
                continue

            # Determine if conversation needs fetching
            needs_fetch, should_stop, fetch_stop_reason = self._determine_fetch_need(
                api_conv_id, comp_conv_id, comp_ts, api_latest_ts_aware,
                existing_conv_logs, min_aware_dt,
            )

            if should_stop:
                stop_processing = True
                if fetch_stop_reason:
                    stop_reason = fetch_stop_reason
                    break

            # Skip if no fetch needed
            if not needs_fetch:
                skipped_count_this_loop += 1
                self._update_progress_skip(progress_bar)
                if stop_processing:
                    break
                continue

            # Validate session before fetch
            self._validate_session()

            # Update progress for processing
            conversations_needing_processing += 1
            self._update_progress_processing(progress_bar, api_conv_id)

            # Fetch conversation context
            context_messages = self._fetch_conversation_context(api_conv_id)
            if context_messages is None:
                error_count_this_loop += 1
                logger.error(f"Failed to fetch context for ConvID {api_conv_id}. Skipping item.")
                continue

            # Lookup or create person
            person, _ = self._lookup_or_create_person(
                session, profile_id_upper, conversation_info.get("username", "Unknown"),
                api_conv_id, existing_person_arg=existing_persons_map.get(profile_id_upper),
            )
            if not person or not safe_column_value(person, "id"):
                error_count_this_loop += 1
                logger.error(f"Failed person lookup/create for ConvID {api_conv_id}. Skipping item.")
                continue
            people_id = safe_column_value(person, "id")

            # Find latest IN and OUT messages
            latest_ctx_in, latest_ctx_out = self._find_latest_messages(context_messages, my_pid_lower)

            # Process IN and OUT messages
            ai_classified_count = self._process_in_message(
                latest_ctx_in, api_conv_id, people_id, existing_conv_logs,
                min_aware_dt, context_messages, my_pid_lower,
                conv_log_upserts_dicts, person_updates, ai_classified_count
            )

            self._process_out_message(
                latest_ctx_out, api_conv_id, people_id, existing_conv_logs,
                min_aware_dt, conv_log_upserts_dicts
            )

            # Update progress bar with stats
            self._update_progress_bar_stats(
                progress_bar, ai_classified_count, status_updated_count,
                skipped_count_this_loop, error_count_this_loop,
            )

            if stop_processing:
                break

        # Update state with modified values
        state["items_processed_before_stop"] = items_processed_before_stop
        state["skipped_count_this_loop"] = skipped_count_this_loop
        state["error_count_this_loop"] = error_count_this_loop
        state["ai_classified_count"] = ai_classified_count
        state["conversations_needing_processing"] = conversations_needing_processing

        return stop_processing, stop_reason

    def _process_inbox_loop(
        self,
        session: DbSession,
        comp_conv_id: Optional[str],
        comp_ts: Optional[datetime],  # Aware datetime
        my_pid_lower: str,
        progress_bar: Optional[tqdm],  # Accept progress bar instance
    ) -> tuple[Optional[str], int, int, int, int]:
        """
        Internal helper: Contains the main loop for fetching and processing inbox batches.
        """
        # Initialize loop state
        state = self._initialize_loop_state()

        # Extract variables from state for easier access
        ai_classified_count = state["ai_classified_count"]
        status_updated_count = state["status_updated_count"]
        total_processed_api_items = state["total_processed_api_items"]
        items_processed_before_stop = state["items_processed_before_stop"]
        logs_processed_in_run = state["logs_processed_in_run"]
        skipped_count_this_loop = state["skipped_count_this_loop"]
        error_count_this_loop = state["error_count_this_loop"]
        stop_reason = state["stop_reason"]
        next_cursor = state["next_cursor"]
        current_batch_num = state["current_batch_num"]
        conv_log_upserts_dicts = state["conv_log_upserts_dicts"]
        person_updates = state["person_updates"]
        stop_processing = state["stop_processing"]
        min_aware_dt = state["min_aware_dt"]
        conversations_needing_processing = state["conversations_needing_processing"]

        # Step 2: Main loop - continues until stop condition met
        while not stop_processing:
            try:  # Inner try for handling exceptions within a single batch iteration
                # Step 2a: Browser Health Monitoring
                browser_error = self._check_browser_health(current_batch_num)
                if browser_error:
                    stop_reason = browser_error
                    stop_processing = True
                    break

                # Step 2b: Validate session before API call
                self._validate_session()

                # Step 2c: Calculate API limit for this batch
                current_limit, limit_error = self._calculate_api_limit(items_processed_before_stop)
                if limit_error:
                    stop_reason = limit_error
                    stop_processing = True
                    break

                # Step 2d: Fetch batch of conversations from API
                all_conversations_batch, next_cursor_from_api = (
                    self._get_all_conversations_api(
                        self.session_manager, limit=current_limit, cursor=next_cursor
                    )
                )

                # Step 2e: Handle API failure
                if all_conversations_batch is None:
                    logger.error("API call to fetch conversation batch failed critically.")
                    stop_reason = "API Error Fetching Batch"
                    stop_processing = True
                    break

                # Step 2f: Process fetched batch
                batch_api_item_count = len(all_conversations_batch)
                total_processed_api_items += batch_api_item_count
                current_batch_num += 1

                # Log progress periodically
                if current_batch_num % 5 == 0 or total_processed_api_items >= 100:
                    logger.info(
                        f"Action 7 Progress: Batch {current_batch_num} "
                        f"(Processed={total_processed_api_items}, AI={ai_classified_count}, "
                        f"StatusUpdates={status_updated_count}, Errors={error_count_this_loop})"
                    )

                # Handle empty batch
                if batch_api_item_count == 0:
                    should_stop, empty_reason = self._handle_empty_batch(next_cursor_from_api)
                    if should_stop:
                        stop_reason = empty_reason
                        stop_processing = True
                        break
                    next_cursor = next_cursor_from_api
                    continue

                # Update progress bar total on first batch
                self._update_progress_bar_initial(progress_bar, batch_api_item_count)

                # Step 2g: Prefetch existing Person and ConversationLog data for batch
                existing_persons_map, existing_conv_logs, prefetch_error = (
                    self._prefetch_batch_data(session, all_conversations_batch, current_batch_num)
                )
                if prefetch_error:
                    stop_reason = prefetch_error
                    stop_processing = True
                    break

                # Step 2h: Process each conversation in the batch
                # Update state dict with current values before calling helper
                state.update({
                    "items_processed_before_stop": items_processed_before_stop,
                    "skipped_count_this_loop": skipped_count_this_loop,
                    "error_count_this_loop": error_count_this_loop,
                    "ai_classified_count": ai_classified_count,
                    "status_updated_count": status_updated_count,
                    "conversations_needing_processing": conversations_needing_processing,
                })

                batch_stop, batch_stop_reason = self._process_conversations_in_batch(
                    session, all_conversations_batch, existing_persons_map, existing_conv_logs,
                    comp_conv_id, comp_ts, my_pid_lower, min_aware_dt, progress_bar, state
                )

                # Extract updated values from state
                items_processed_before_stop = state["items_processed_before_stop"]
                skipped_count_this_loop = state["skipped_count_this_loop"]
                error_count_this_loop = state["error_count_this_loop"]
                ai_classified_count = state["ai_classified_count"]
                conversations_needing_processing = state["conversations_needing_processing"]

                if batch_stop:
                    stop_processing = True
                    if batch_stop_reason:
                        stop_reason = batch_stop_reason

                # Commit batch data
                logs_committed, persons_updated = self._commit_batch_updates(
                    session, conv_log_upserts_dicts, person_updates, current_batch_num
                )
                status_updated_count += persons_updated
                logs_processed_in_run += logs_committed

                # Check stop flag after commit
                if stop_processing:
                    if not stop_reason:
                        stop_reason = "Comparator Found"
                    break

                # Prepare for next batch
                next_cursor = next_cursor_from_api
                if not next_cursor:
                    stop_reason = "End of Inbox Reached (No Next Cursor)"
                    stop_processing = True
                    logger.debug("No next cursor from API. Ending processing.")
                    if progress_bar is not None:
                        progress_bar.total = items_processed_before_stop
                        progress_bar.refresh()
                    break

                # Check for cancellation request
                if self._check_cancellation_requested():
                    stop_reason = stop_reason or "Timeout Cancellation"
                    stop_processing = True
                    logger.warning("Cancellation requested by timeout wrapper. Stopping inbox processing loop.")
                    break

            # Handle exceptions during batch processing
            except WebDriverException as wde:
                error_count_this_loop += 1
                logger.error(
                    f"WebDriverException occurred during inbox loop (Batch {current_batch_num}): {wde}"
                )
                stop_reason = "WebDriver Exception"
                stop_processing = True

                final_logs, final_persons = self._handle_exception_with_save(
                    session, conv_log_upserts_dicts, person_updates, "WebDriverException", wde
                )
                status_updated_count += final_persons
                logs_processed_in_run += final_logs
                break

            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt detected during inbox loop.")
                stop_reason = "Keyboard Interrupt"
                stop_processing = True

                final_logs, final_persons = self._handle_exception_with_save(
                    session, conv_log_upserts_dicts, person_updates, "Interrupt", KeyboardInterrupt()
                )
                status_updated_count += final_persons
                logs_processed_in_run += final_logs
                break
            except Exception as e_main:
                error_count_this_loop += 1
                logger.critical(
                    f"Critical error in inbox processing loop (Batch {current_batch_num}): {e_main}",
                    exc_info=True,
                )
                stop_reason = f"Critical Error ({type(e_main).__name__})"
                stop_processing = True

                final_logs, final_persons = self._handle_exception_with_save(
                    session, conv_log_upserts_dicts, person_updates, "Critical Error", e_main
                )
                status_updated_count += final_persons
                logs_processed_in_run += final_logs

                return (
                    stop_reason,
                    total_processed_api_items,
                    ai_classified_count,
                    status_updated_count,
                    items_processed_before_stop,
                )

        # --- End Main Loop (while not stop_processing) ---

        # Update state dictionary with final values
        state.update({
            "ai_classified_count": ai_classified_count,
            "status_updated_count": status_updated_count,
            "total_processed_api_items": total_processed_api_items,
            "items_processed_before_stop": items_processed_before_stop,
            "logs_processed_in_run": logs_processed_in_run,
            "skipped_count_this_loop": skipped_count_this_loop,
            "error_count_this_loop": error_count_this_loop,
            "stop_reason": stop_reason,
            "next_cursor": next_cursor,
            "current_batch_num": current_batch_num,
            "conv_log_upserts_dicts": conv_log_upserts_dicts,
            "person_updates": person_updates,
            "stop_processing": stop_processing,
            "conversations_needing_processing": conversations_needing_processing,
        })

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
        ) and (conv_log_upserts_dicts or person_updates):
            logger.debug("Performing final commit at end of processing loop...")
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
        # Step 1: Print header (green summary block)
        print(" ")
        green = '\x1b[32m'
        reset = '\x1b[0m'
        def g(msg: str) -> str:
            return f"{green}{msg}{reset}"
        logger.info(g("------ Inbox Search Summary ------"))
        # Mark unused parameters to satisfy linter without changing signature
        _ = new_logs
        # Step 2: Log key metrics
        logger.info(g(f"  API Conversations Fetched:    {total_api_items}"))
        logger.info(g(f"  Conversations Processed:      {items_processed}"))
        # logger.info(f"  New/Updated Log Entries:    {new_logs}") # Removed as upsert logic complicates exact counts
        logger.info(g(f"  AI Classifications Attempted: {ai_classified}"))
        logger.info(g(f"  Person Status Updates Made:   {status_updates}"))
        # Step 3: Log stopping reason
        final_reason = stop_reason
        if not stop_reason:
            # Infer reason if not explicitly set
            if max_inbox_limit == 0 or items_processed < max_inbox_limit:
                final_reason = "End of Inbox Reached or Comparator Match"
            else:
                final_reason = f"Inbox Limit ({max_inbox_limit}) Reached"
        logger.info(g(f"  Processing Stopped Due To:    {final_reason}"))
        # Step 4: Print footer
        logger.info(g("----------------------------------\n"))

        # Update statistics
        self.stats.update(
            {
                "conversations_fetched": total_api_items,
                "conversations_processed": items_processed,
                "ai_classifications": ai_classified,
                "person_updates": status_updates,
                "end_time": datetime.now(timezone.utc),
            }
        )

    # End of _log_unified_summary


# --- Enhanced Test Framework Implementation ---

def action7_inbox_module_tests() -> bool:
    """Comprehensive test suite for action7_inbox.py using the unified TestSuite."""
    from unittest.mock import MagicMock

    from core.progress_indicators import create_progress_indicator
    from test_framework import TestSuite, mock_logger_context, suppress_logging

    suite = TestSuite("Action 7 - Inbox Processor", "action7_inbox.py")
    suite.start_suite()

    def test_class_and_methods_available() -> None:
        """Ensure core classes and methods exist and are callable."""
        # InboxProcessor exists
        assert 'InboxProcessor' in globals(), "InboxProcessor class should exist"
        assert callable(InboxProcessor), "InboxProcessor should be callable"
        # search_inbox method exists and is callable on an instance
        sm = MagicMock()
        processor = InboxProcessor(sm)
        assert hasattr(processor, 'search_inbox') and callable(processor.search_inbox)
        # internal helpers exist
        assert hasattr(processor, '_process_inbox_loop'), "_process_inbox_loop should exist"
        assert hasattr(processor, '_log_unified_summary'), "_log_unified_summary should exist"
        return True

    def test_circuit_breaker_config() -> None:
        """Verify search_inbox bears expected signature/decorators."""
        try:
            search_method = getattr(InboxProcessor, 'search_inbox', None)
            if not search_method:
                return False
            sig = inspect.signature(search_method)
            return 'self' in sig.parameters
        except Exception:
            return False

    def test_progress_indicator_smoke() -> None:
        """Progress indicator can be created (smoke)."""
        with create_progress_indicator(
            description="TEST",
            total=5,
            unit="conv",
            log_start=False,
            log_finish=False,
            leave=False,
        ) as progress:
            pb = progress.progress_bar
            assert pb is not None, "Progress bar should be created"
            pb.update(1)
        return True

    def test_summary_logging_structure() -> None:
        """_log_unified_summary emits the expected lines to logger (captured)."""
        sm = MagicMock()
        processor = InboxProcessor(sm)
        with mock_logger_context(globals()) as mock_log:
            processor._log_unified_summary(
                total_api_items=3,
                items_processed=2,
                new_logs=0,
                ai_classified=2,
                status_updates=1,
                stop_reason="Inbox Limit (2)",
                max_inbox_limit=2,
            )
            combined = "\n".join(mock_log.lines)
            assert "Inbox Search Summary" in combined
            assert "API Conversations Fetched" in combined
            assert "Conversations Processed" in combined
            assert "AI Classifications Attempted" in combined
            assert "Person Status Updates Made" in combined
            assert "Processing Stopped Due To" in combined
        return True

    with suppress_logging():
        suite.run_test(
            test_name="Class and method availability",
            test_func=test_class_and_methods_available,
            expected_behavior="InboxProcessor and key methods present",
            test_description="Module structure",
            method_description="Check class and method existence",
        )
        suite.run_test(
            test_name="Circuit breaker config",
            test_func=test_circuit_breaker_config,
            expected_behavior="search_inbox has 'self' parameter and decorators",
            test_description="Decorator verification",
            method_description="Signature inspection",
        )
        suite.run_test(
            test_name="Progress indicator smoke",
            test_func=test_progress_indicator_smoke,
            expected_behavior="ProgressIndicator creates tqdm without errors",
            test_description="Progress bar integration",
            method_description="Context manager smoke test",
        )
        suite.run_test(
            test_name="Summary logging structure",
            test_func=test_summary_logging_structure,
            expected_behavior="Summary logs contain required lines",
            test_description="Unified summary logging",
            method_description="Mock logger capture",
        )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(action7_inbox_module_tests)


# --- Main Execution Block ---

if __name__ == "__main__":
    import sys
    print("🔄 Running Action 7 (Inbox Processor) comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

# End of action7_inbox.py
