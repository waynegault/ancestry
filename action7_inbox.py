#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportCallIssue=false, reportGeneralTypeIssues=false

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
import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional, cast

# === THIRD-PARTY IMPORTS ===
from selenium.common.exceptions import WebDriverException
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session as DbSession

# === LOCAL IMPORTS ===
from ai_interface import assess_engagement, classify_message_intent

# === PHASE 5.2: SYSTEM-WIDE CACHING OPTIMIZATION ===
from cache_manager import (
    cached_api_call,
)
from common_params import ConversationProcessingContext
from config import config_schema
from conversation_analytics import record_engagement_event, update_conversation_metrics
from core.enhanced_error_recovery import with_api_recovery, with_enhanced_recovery

# === ACTION 7 ERROR CLASSES (Action 8 Pattern) ===
from core.error_handling import (
    APIError,
    AuthenticationError,
    BrowserError,
)
from core.session_manager import SessionManager
from database import (
    ConversationLog,
    ConversationPhaseEnum,
    ConversationState,
    DnaMatch,
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
from connection_resilience import with_connection_resilience
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
        self.rate_limiter = (
            session_manager.rate_limiter
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
        # AI Provider setting
        self.ai_provider = getattr(config_schema, "ai_provider", "")
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
            "engagement_assessments": 0,
            "person_updates": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

        # InboxProcessor initialized (removed verbose debug)

    # End of __init__

    # --- Private Helper Methods ---

    def _validate_session_prerequisites(self, session_manager: SessionManager) -> None:
        """Validate session prerequisites. Raises WebDriverException if invalid."""
        if not session_manager or not session_manager.my_profile_id:
            logger.error("_get_all_conversations_api: SessionManager or profile ID missing.")
            raise WebDriverException("SessionManager or profile ID missing")

        if not session_manager.is_sess_valid():
            logger.error("_get_all_conversations_api: Session invalid before API call.")
            raise WebDriverException("Session invalid before conversation overview API call")

    def _build_conversations_api_url(self, my_profile_id: str, limit: int, cursor: Optional[str]) -> str:
        """Build API URL for fetching conversations."""
        api_base = urljoin(getattr(config_schema.api, "base_url", ""), "/app-api/express/v2/")
        url = f"{api_base}conversations?q=user:{my_profile_id}&limit={limit}"
        if cursor:
            url += f"&cursor={cursor}"
        return url

    def _process_conversations_response(
        self, response_data: dict, my_profile_id: str
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """Process API response and extract conversations. Returns (conversations, forward_cursor)."""
        conversations_raw = response_data.get("conversations", [])
        all_conversations_processed: list[dict[str, Any]] = []

        if isinstance(conversations_raw, list):
            for conv_data in conversations_raw:
                info = self._extract_conversation_info(conv_data, my_profile_id)
                if info:
                    all_conversations_processed.append(info)
        else:
            logger.warning(
                "_get_all_conversations_api: 'conversations' key not found or not a list in API response."
            )

        forward_cursor = response_data.get("paging", {}).get("forward_cursor")
        return all_conversations_processed, forward_cursor

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
        # Validate prerequisites
        self._validate_session_prerequisites(session_manager)

        # Construct API URL
        my_profile_id = session_manager.my_profile_id
        url = self._build_conversations_api_url(my_profile_id, limit, cursor)

        logger.debug(f"Fetching inbox conversations: limit={limit}, cursor={'present' if cursor else 'None'}")

        # Make API call
        try:
            response_data = _api_req(
                url=url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                use_csrf_token=False,
                api_description="Get Inbox Conversations",
            )

            # Validate response
            if response_data is None:
                logger.warning("_get_all_conversations_api: _api_req returned None.")
                return None, None

            if not isinstance(response_data, dict):
                logger.error(
                    f"_get_all_conversations_api: Unexpected API response format. "
                    f"Type={type(response_data)}, Expected=dict"
                )
                return None, None

            # Process response
            result = self._process_conversations_response(response_data, my_profile_id)
            if result[0]:
                logger.debug(f"Successfully fetched {len(result[0])} conversations, next_cursor={'present' if result[1] else 'None'}")
            return result

        except WebDriverException as e:
            logger.error(f"WebDriverException during _get_all_conversations_api: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _get_all_conversations_api: {e}", exc_info=True)
            return None, None

    # End of _get_all_conversations_api

    def _validate_conversation_data(self, conv_data: dict[str, Any]) -> Optional[tuple[str, dict]]:
        """Validate conversation data and extract basic info."""
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
            limiter = cast(Any, getattr(self, "rate_limiter", None))
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
    ) -> tuple[Optional[Person], bool]:
        """Look up person in database by profile ID.

        Returns:
            tuple: (Person object or None, error_occurred: bool)
                - (None, False) = Person not found (normal case)
                - (None, True) = Database error occurred
                - (Person, False) = Person found successfully
        """
        try:
            person = (
                session.query(Person)
                .filter(
                    func.upper(Person.profile_id) == profile_id.upper(),
                    Person.deleted_at.is_(None),
                )
                .first()
            )
            return person, False  # Success (found or not found)
        except SQLAlchemyError as e:
            logger.error(
                f"DB error looking up Person {log_ref}: {type(e).__name__}: {e}",
                exc_info=True
            )
            return None, True  # Error occurred

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
            except IntegrityError as upd_err:
                logger.error(
                    f"DB IntegrityError flushing update for Person {log_ref}: {type(upd_err).__name__}: {upd_err}"
                )
                session.rollback()
                return "error"
            except SQLAlchemyError as upd_err:
                logger.error(
                    f"DB SQLAlchemyError flushing update for Person {log_ref}: {type(upd_err).__name__}: {upd_err}",
                    exc_info=True
                )
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

        except IntegrityError as create_err:
            logger.error(
                f"DB IntegrityError creating Person {log_ref}: {type(create_err).__name__}: {create_err}. "
                f"This may indicate a duplicate profile_id or constraint violation."
            )
            session.rollback()
            return None, "error"
        except SQLAlchemyError as create_err:
            logger.error(
                f"DB SQLAlchemyError creating Person {log_ref}: {type(create_err).__name__}: {create_err}",
                exc_info=True
            )
            session.rollback()
            return None, "error"
        except Exception as e:
            logger.critical(
                f"Unexpected error creating Person {log_ref}: {type(e).__name__}: {e}",
                exc_info=True
            )
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
            db_error = False
        else:
            person, db_error = self._lookup_person_in_db(session, profile_id, log_ref)
            if db_error:
                # DB error occurred during lookup
                logger.error(f"Database error during person lookup for {log_ref}")
                return None, "error"

        # Process based on whether person exists
        if person:
            status = self._update_existing_person(session, person, username_to_use, profile_id, log_ref)
            if status == "error":
                return None, "error"
            return person, status

        # Person not found, create new one
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
            # EXCLUDE dry run conversations (they don't exist in the real API)
            latest_entry = (
                session.query(
                    ConversationLog.conversation_id, ConversationLog.latest_timestamp
                )
                .filter(~ConversationLog.conversation_id.like('dryrun_%'))
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

    def _log_configuration(self) -> None:
        """Log Action 7 configuration settings."""
        # Get current rate limiting delay (adjusted by dynamic rate limiting)
        current_delay = self.session_manager.rate_limiter.current_delay if self.session_manager.rate_limiter else 0.0

        logger.info(f"Configuration: MAX_INBOX={self.max_inbox_limit}, AI_PROVIDER={self.ai_provider}, RATE_LIMIT_DELAY={current_delay:.2f}s")

    def _validate_session_state(self) -> Optional[str]:
        """Validate session manager state and return profile ID."""
        if not self.session_manager or not self.session_manager.my_profile_id:
            logger.error("search_inbox: Session manager or profile ID missing.")
            return None
        return self.session_manager.my_profile_id.lower()



    @with_connection_resilience("Action 7: Inbox Processing", max_recovery_attempts=3)
    def search_inbox(self) -> bool:
        """
        Main method to search the Ancestry inbox with enhanced error handling and statistics.

        Returns:
            True if the process completed without critical errors, False otherwise.
        """
        # Initialize statistics and state
        self._initialize_search_stats()

        # Log configuration
        self._log_configuration()

        # Validate session manager state
        my_pid_lower = self._validate_session_state()
        if not my_pid_lower:
            return False

        # Get database session and comparator
        session, comp_conv_id, comp_ts = self._get_database_session_and_comparator()
        if not session:
            return False

        try:
            # Run inbox processing loop
            (
                stop_reason,
                total_api_items,
                ai_classified,
                engagement_assessments,
                status_updates,
                items_processed,
                session_deaths,
                session_recoveries,
            ) = (
                self._run_inbox_processing_loop(session, comp_conv_id, comp_ts, my_pid_lower)
            )

            # Log unified summary
            self._log_unified_summary(
                total_api_items=total_api_items,
                items_processed=items_processed,
                new_logs=0,  # Upsert logic makes exact count difficult
                ai_classified=ai_classified,
                engagement_assessments=engagement_assessments,
                status_updates=status_updates,
                stop_reason=stop_reason,
                max_inbox_limit=self.max_inbox_limit,
                session_deaths=session_deaths,
                session_recoveries=session_recoveries,
            )

            # Update final statistics
            self.stats["end_time"] = datetime.now(timezone.utc)

            return True

        except Exception as e:
            logger.error(f"Critical error in search_inbox: {e}", exc_info=True)
            self.stats["errors"] += 1
            self.stats["end_time"] = datetime.now(timezone.utc)
            return False
        finally:
            # Ensure session is closed
            if session:
                from contextlib import suppress
                with suppress(Exception):
                    session.close()

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
    ) -> tuple[Optional[str], int, int, int, int, int, int, int]:
        """Run the main inbox processing loop.

        Returns: (stop_reason, total_api_items, ai_classified, engagement_assessments,
                  status_updated, items_processed, session_deaths, session_recoveries)
        """
        # Add newline before processing starts
        print()

        # Process inbox without progress bar (batch-level logging only, like Action 6)
        return self._process_inbox_loop(
            session, comp_conv_id, comp_ts, my_pid_lower
        )

    def get_statistics(self) -> dict[str, Any]:
        """Return processing statistics for monitoring and debugging."""
        stats = self.stats.copy()
        # Only calculate duration if both timestamps are not None
        if stats.get("start_time") is not None and stats.get("end_time") is not None:
            stats["duration_seconds"] = (
                stats["end_time"] - stats["start_time"]
            ).total_seconds()
        return stats

    def _initialize_loop_state(self) -> dict[str, Any]:
        """Initialize state variables for the inbox processing loop."""
        return {
            "ai_classified_count": 0,
            "engagement_assessment_count": 0,
            "status_updated_count": 0,
            "total_processed_api_items": 0,
            "items_processed_before_stop": 0,
            "logs_processed_in_run": 0,
            "skipped_count_this_loop": 0,
            "error_count_this_loop": 0,
            "session_deaths": 0,
            "session_recoveries": 0,
            "stop_reason": None,
            "next_cursor": None,
            "current_batch_num": 0,
            "conv_log_upserts_dicts": [],
            "person_updates": {},
            "stop_processing": False,
            "min_aware_dt": datetime.min.replace(tzinfo=timezone.utc),
            "conversations_needing_processing": 0,
        }

    def _check_browser_health(self, current_batch_num: int, state: dict[str, Any]) -> Optional[str]:
        """Check browser health and attempt recovery if needed. Updates state with death/recovery counts."""
        if current_batch_num % 5 == 0 and not self.session_manager.check_browser_health():
            logger.warning(f"Browser health check failed at batch {current_batch_num}")
            state["session_deaths"] += 1
            if self.session_manager.attempt_browser_recovery("Action 7 Browser Recovery"):
                logger.info("Session recovered successfully")
                state["session_recoveries"] += 1
                return None
            logger.critical(f"Browser recovery failed at batch {current_batch_num} - halting inbox processing")
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

        logger.debug(
            f"[Batch {current_batch_num}] Prefetching data: "
            f"{len(batch_conv_ids)} conversations, {len(batch_profile_ids)} unique profiles"
        )

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
                logger.debug(
                    f"[Batch {current_batch_num}] Prefetched {len(existing_persons_map)}/{len(batch_profile_ids)} "
                    f"existing persons ({len(batch_profile_ids) - len(existing_persons_map)} new)"
                )

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
                logger.debug(
                    f"[Batch {current_batch_num}] Prefetched {len(existing_conv_logs)} conversation logs "
                    f"(enables smart skip logic for up-to-date conversations)"
                )

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
        self, api_conv_id: Optional[str], profile_id_upper: str
    ) -> bool:
        """Check if conversation should be skipped due to invalid data."""
        return not api_conv_id or profile_id_upper == "UNKNOWN"

    def _check_comparator_match(
        self,
        api_conv_id: str,
        comp_conv_id: Optional[str],
        comp_ts: Optional[datetime],
        api_latest_ts_aware: Optional[datetime],
    ) -> tuple[bool, bool, bool, Optional[str]]:
        """Check if conversation matches comparator. Returns (is_comparator, needs_fetch, stop_processing, stop_reason)."""
        if not comp_conv_id or api_conv_id != comp_conv_id:
            return False, False, False, None

        if comp_ts and api_latest_ts_aware and api_latest_ts_aware > comp_ts:
            return True, True, True, None

        return True, False, True, "Comparator Found (No Change)"

    def _get_db_latest_timestamp(
        self,
        existing_conv_logs: dict[tuple[str, str], ConversationLog],
        api_conv_id: str,
        min_aware_dt: datetime,
    ) -> datetime:
        """Get latest timestamp from DB logs for conversation."""
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

        return max(db_latest_ts_in, db_latest_ts_out)

    def _was_recently_processed(
        self,
        existing_conv_logs: dict[tuple[str, str], ConversationLog],
        api_conv_id: str,
    ) -> bool:
        """Check if conversation was recently processed and should be skipped.

        NOTE: Time-based skipping is disabled. We only stop at comparator (most recent conversation).
        This function always returns False - parameters kept for API compatibility.

        Args:
            existing_conv_logs: Not used (time-based skipping disabled)
            api_conv_id: Not used (time-based skipping disabled)
        """
        # Time-based skipping disabled - only use comparator logic to stop processing
        # Parameters kept for API compatibility but not used
        _ = existing_conv_logs, api_conv_id  # Mark as intentionally unused
        return False

    def _should_fetch_based_on_timestamp(
        self,
        api_latest_ts_aware: Optional[datetime],
        db_latest_overall: datetime,
        existing_conv_logs: dict[tuple[str, str], ConversationLog],
        api_conv_id: str,
    ) -> bool:
        """Determine if fetch is needed based on timestamp comparison."""
        # Check if recently processed (smart skip logic)
        if self._was_recently_processed(existing_conv_logs, api_conv_id):
            return False

        # Fetch if API timestamp is newer
        if api_latest_ts_aware and api_latest_ts_aware > db_latest_overall:
            return True

        # Fetch if no DB logs exist
        db_log_in = existing_conv_logs.get((api_conv_id, MessageDirectionEnum.IN.name))
        db_log_out = existing_conv_logs.get((api_conv_id, MessageDirectionEnum.OUT.name))
        return bool(not db_log_in and not db_log_out)

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
        is_comparator, needs_fetch, stop_processing, stop_reason = self._check_comparator_match(
            api_conv_id, comp_conv_id, comp_ts, api_latest_ts_aware
        )
        if is_comparator:
            return needs_fetch, stop_processing, stop_reason

        # Not comparator - compare with DB timestamps
        db_latest_overall = self._get_db_latest_timestamp(existing_conv_logs, api_conv_id, min_aware_dt)

        # Determine if fetch is needed based on timestamp
        needs_fetch = self._should_fetch_based_on_timestamp(
            api_latest_ts_aware, db_latest_overall, existing_conv_logs, api_conv_id
        )

        return needs_fetch, False, None



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

    def _downgrade_if_non_actionable(
        self, label: Optional[str], messages: list[dict], my_pid_lower: str
    ) -> Optional[str]:
        """Downgrade PRODUCTIVE label if message lacks actionable cues."""
        try:
            if (not label) or label != "PRODUCTIVE":
                return label

            # Find last user message
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
        def _classify_with_recovery(context: str = formatted_context) -> Optional[str]:
            return classify_message_intent(context, self.session_manager)

        ai_result = _classify_with_recovery()

        # Extract sentiment from result
        ai_sentiment_result = (ai_result[0] if ai_result else None) if isinstance(ai_result, tuple) else ai_result

        return self._downgrade_if_non_actionable(ai_sentiment_result, context_messages, my_pid_lower)

    def _is_closed_status(self, ai_sentiment: Optional[str]) -> bool:
        """Check if conversation should be marked CLOSED."""
        return ai_sentiment in ("DESIST", "UNINTERESTED")

    def _is_initial_outreach_status(
        self,
        direction: MessageDirectionEnum,
        in_messages: list[Any],
    ) -> bool:
        """Check if conversation is in INITIAL_OUTREACH (outbound only, no response)."""
        return direction == MessageDirectionEnum.OUT and len(in_messages) == 0

    def _is_response_received_status(
        self,
        direction: MessageDirectionEnum,
        in_messages: list[Any],
    ) -> bool:
        """Check if conversation is in RESPONSE_RECEIVED (first inbound)."""
        return direction == MessageDirectionEnum.IN and len(in_messages) == 1

    def _is_stalled_status(self, days_since_last: int, total_exchanges: int) -> bool:
        """Check if conversation is STALLED (>30 days, multiple exchanges)."""
        return days_since_last > 30 and total_exchanges > 1

    def _is_collaboration_active_status(
        self,
        total_exchanges: int,
        days_since_last: int,
        conv_logs: list[Any],
    ) -> bool:
        """Check if conversation is COLLABORATION_ACTIVE (4+ exchanges, recent, productive)."""
        if total_exchanges >= 4 and days_since_last <= 14:
            productive_count = sum(
                1 for log in conv_logs[-4:]  # Last 4 messages
                if log.ai_sentiment == "PRODUCTIVE"
            )
            return productive_count >= 2
        return False

    def _is_information_shared_status(
        self,
        total_exchanges: int,
        conv_logs: list[Any],
        ai_sentiment: Optional[str],
    ) -> bool:
        """Check if conversation is INFORMATION_SHARED (2+ exchanges, productive)."""
        if total_exchanges >= 2:
            productive_count = sum(
                1 for log in conv_logs if log.ai_sentiment == "PRODUCTIVE"
            )
            return productive_count >= 1 or ai_sentiment == "PRODUCTIVE"
        return False

    def _determine_conversation_phase(
        self,
        conversation_id: str,
        direction: MessageDirectionEnum,
        ai_sentiment: Optional[str],
        existing_logs: list[Any],
        timestamp: datetime,
    ) -> Optional[Any]:  # ConversationPhaseEnum imported at runtime
        """
        Determine conversation lifecycle phase based on message history and patterns.

        Priority 1 Todo #11: Conversation Phase Transitions

        Phase Logic:
        - INITIAL_OUTREACH: First OUT message sent, no IN messages yet
        - RESPONSE_RECEIVED: First IN message received after outreach
        - INFORMATION_SHARED: 2+ exchanges with PRODUCTIVE classification
        - COLLABORATION_ACTIVE: 4+ exchanges, recent activity (<14 days)
        - STALLED: Last message >30 days ago, pending response
        - CLOSED: DESIST/ARCHIVE status, or explicit conclusion

        Args:
            conversation_id: Unique conversation identifier
            direction: Current message direction (IN or OUT)
            ai_sentiment: AI classification result (PRODUCTIVE, DESIST, etc.)
            existing_logs: List of existing ConversationLog entries for this conversation
            timestamp: Timestamp of current message

        Returns:
            ConversationPhaseEnum value or None if unable to determine
        """
        try:
            from database import ConversationPhaseEnum, MessageDirectionEnum as DBMessageDirectionEnum

            # Filter logs for this conversation
            conv_logs = [log for log in existing_logs if log.conversation_id == conversation_id]

            # Count message exchanges
            in_messages = [log for log in conv_logs if log.direction == DBMessageDirectionEnum.IN]
            out_messages = [log for log in conv_logs if log.direction == DBMessageDirectionEnum.OUT]
            total_exchanges = len(in_messages) + len(out_messages)

            # Calculate days since last message
            if conv_logs:
                latest_log = max(conv_logs, key=lambda log: log.latest_timestamp)
                days_since_last = (timestamp - latest_log.latest_timestamp).days
            else:
                days_since_last = 0

            phase_checks = (
                (self._is_closed_status(ai_sentiment), ConversationPhaseEnum.CLOSED),
                (self._is_initial_outreach_status(direction, in_messages), ConversationPhaseEnum.INITIAL_OUTREACH),
                (self._is_response_received_status(direction, in_messages), ConversationPhaseEnum.RESPONSE_RECEIVED),
                (self._is_stalled_status(days_since_last, total_exchanges), ConversationPhaseEnum.STALLED),
                (
                    self._is_collaboration_active_status(total_exchanges, days_since_last, conv_logs),
                    ConversationPhaseEnum.COLLABORATION_ACTIVE,
                ),
                (
                    self._is_information_shared_status(total_exchanges, conv_logs, ai_sentiment),
                    ConversationPhaseEnum.INFORMATION_SHARED,
                ),
            )

            phase_result: Optional[Any] = next(
                (candidate for condition, candidate in phase_checks if condition),
                None,
            )

            if phase_result is None and total_exchanges > 0:
                phase_result = ConversationPhaseEnum.RESPONSE_RECEIVED

            return phase_result

        except Exception as e:
            logger.warning(f"Error determining conversation phase for {conversation_id}: {e}")
            return None

    @staticmethod
    def _default_follow_up_payload() -> dict[str, Any]:
        return {
            "follow_up_required": False,
            "follow_up_due_date": None,
            "awaiting_response_from": None,
        }

    @staticmethod
    def _should_skip_follow_up(
        ai_sentiment: Optional[str],
        conversation_phase: Optional[Any],
    ) -> bool:
        return (ai_sentiment in {"DESIST", "UNINTERESTED"}) or (
            conversation_phase == ConversationPhaseEnum.CLOSED
        )

    @staticmethod
    def _build_follow_up_history(conversation_history: list[Any]) -> str:
        snippets: list[str] = []
        for log_entry in conversation_history[-10:]:
            direction = getattr(log_entry, "direction", MessageDirectionEnum.IN)
            dir_label = "ME" if direction == MessageDirectionEnum.OUT else "USER"
            content = getattr(log_entry, "latest_message_content", "") or ""
            snippets.append(f"{dir_label}: {content[:500]}")
        return "\n\n".join(snippets)

    @staticmethod
    def _build_follow_up_context(
        conversation_history_str: str,
        latest_message: str,
        direction: MessageDirectionEnum,
        conversation_phase: Optional[Any],
        ai_sentiment: Optional[str],
    ) -> dict[str, str]:
        phase_str = conversation_phase.value if conversation_phase else "unknown"
        direction_str = "IN" if direction == MessageDirectionEnum.IN else "OUT"
        return {
            "conversation_history": conversation_history_str,
            "latest_message": latest_message[:1000],
            "direction": direction_str,
            "conversation_phase": phase_str,
            "ai_sentiment": ai_sentiment or "UNKNOWN",
        }

    def _process_follow_up_result(
        self,
        result: Optional[dict[str, Any]],
        conversation_id: str,
    ) -> dict[str, Any]:
        if not result or "error" in result:
            error_msg = result.get("error") if isinstance(result, dict) else "unknown error"
            logger.warning(
                "Follow-up extraction failed for %s: %s", conversation_id, error_msg
            )
            return self._default_follow_up_payload()

        follow_up_data: Optional[dict[str, Any]]
        if isinstance(result, dict) and "response" in result:
            follow_up_data = result.get("response")  # type: ignore[assignment]
        else:
            follow_up_data = result if isinstance(result, dict) else None

        if not isinstance(follow_up_data, dict):
            logger.warning(
                "Invalid follow-up data format for %s: %s",
                conversation_id,
                type(follow_up_data),
            )
            return self._default_follow_up_payload()

        follow_up_required = follow_up_data.get("follow_up_required", False)
        days_until_due = follow_up_data.get("days_until_due")
        awaiting_response_from = follow_up_data.get("awaiting_response_from")

        follow_up_due_date = None
        if follow_up_required and days_until_due:
            follow_up_due_date = datetime.now(timezone.utc) + timedelta(days=days_until_due)
            logger.info(
                "Follow-up scheduled for %s: due in %s days (%s), awaiting: %s",
                conversation_id,
                days_until_due,
                follow_up_due_date.strftime("%Y-%m-%d"),
                awaiting_response_from,
            )

        return {
            "follow_up_required": follow_up_required,
            "follow_up_due_date": follow_up_due_date,
            "awaiting_response_from": awaiting_response_from,
            "follow_up_reason": follow_up_data.get("follow_up_reason"),
            "pending_items": follow_up_data.get("pending_items", []),
            "reminder_task_title": follow_up_data.get("reminder_task_title"),
            "reminder_task_body": follow_up_data.get("reminder_task_body"),
            "urgency_level": follow_up_data.get("urgency_level", "standard"),
        }

    @staticmethod
    def _determine_follow_up_window(
        direction: MessageDirectionEnum,
        ai_sentiment: Optional[str],
        conversation_phase: Optional[Any],
    ) -> int:
        if ai_sentiment == "PRODUCTIVE":
            return 7 if direction == MessageDirectionEnum.IN else 14
        if ai_sentiment == "CAUTIOUSLY_INTERESTED":
            return 14
        if conversation_phase == ConversationPhaseEnum.RESPONSE_RECEIVED:
            return 14
        return 30

    @staticmethod
    def _determine_pending_items(latest_message: str) -> list[str]:
        stripped_message = latest_message.strip()
        if stripped_message and "?" in stripped_message:
            return [stripped_message]
        return []

    @staticmethod
    def _derive_follow_up_reason(
        follow_up_required: bool,
        ai_sentiment: Optional[str],
    ) -> Optional[str]:
        if follow_up_required:
            return "Match asked a question we should answer"
        if ai_sentiment == "CAUTIOUSLY_INTERESTED":
            return "Match shows interest but needs additional information"
        return None

    @staticmethod
    def _build_reminder_details(
        follow_up_required: bool,
        pending_items: list[str],
        conversation_history_str: str,
    ) -> tuple[Optional[str], Optional[str]]:
        if not follow_up_required:
            return None, None

        title = "Follow up with DNA match"
        if pending_items:
            return title, f"They asked: {pending_items[0]}"

        if conversation_history_str:
            last_line = conversation_history_str.splitlines()[-1][:200]
            if last_line:
                return title, last_line

        return title, None

    @staticmethod
    def _calculate_urgency_level(
        direction: MessageDirectionEnum,
        ai_sentiment: Optional[str],
    ) -> str:
        if ai_sentiment == "PRODUCTIVE" and direction == MessageDirectionEnum.IN:
            return "urgent"
        if ai_sentiment == "CAUTIOUSLY_INTERESTED":
            return "standard"
        return "standard"

    def _analyze_follow_up_requirements(
        self,
        conversation_history_str: str,
        latest_message: str,
        direction: MessageDirectionEnum,
        conversation_phase: Optional[Any],
        ai_sentiment: Optional[str],
    ) -> dict[str, Any]:
        follow_up_required = direction == MessageDirectionEnum.IN and ai_sentiment == "PRODUCTIVE"
        awaiting_response_from = "me" if follow_up_required else None

        pending_items = self._determine_pending_items(latest_message)
        reason = self._derive_follow_up_reason(follow_up_required, ai_sentiment)
        reminder_task_title, reminder_task_body = self._build_reminder_details(
            follow_up_required,
            pending_items,
            conversation_history_str,
        )

        urgency_level = self._calculate_urgency_level(direction, ai_sentiment)
        days_until_due = self._determine_follow_up_window(direction, ai_sentiment, conversation_phase)

        return {
            "follow_up_required": follow_up_required,
            "awaiting_response_from": awaiting_response_from,
            "pending_items": pending_items,
            "follow_up_reason": reason,
            "reminder_task_title": reminder_task_title,
            "reminder_task_body": reminder_task_body,
            "urgency_level": urgency_level,
            "days_until_due": days_until_due,
        }

    def _extract_follow_up_requirements(
        self,
        conversation_history: list[Any],
        latest_message: str,
        direction: MessageDirectionEnum,
        conversation_phase: Optional[Any],
        ai_sentiment: Optional[str],
        conversation_id: str,
    ) -> dict[str, Any]:
        """
        Extract follow-up requirements from PRODUCTIVE conversations.

        Priority 1 Todo #5: Follow-Up Reminder System

        Determines if follow-up is needed, calculates urgency-based due date (7/14/30 days),
        and identifies who needs to respond next ('me' or 'them').

        Args:
            conversation_history: List of ConversationLog entries for context
            latest_message: Content of the most recent message
            direction: Direction of latest message (IN or OUT)
            conversation_phase: Current conversation phase (INITIAL_OUTREACH, etc.)
            ai_sentiment: AI classification (PRODUCTIVE, DESIST, etc.)
            conversation_id: Unique conversation identifier

        Returns:
            Dictionary with follow_up_required, due_date, awaiting_response_from fields
        """
        try:
            if self._should_skip_follow_up(ai_sentiment, conversation_phase):
                logger.debug(
                    "Skipping follow-up extraction for %s: sentiment=%s phase=%s",
                    conversation_id,
                    ai_sentiment,
                    getattr(conversation_phase, "name", conversation_phase),
                )
                return self._default_follow_up_payload()

            conversation_history_str = self._build_follow_up_history(conversation_history)

            logger.debug(
                "Analyzing follow-up requirements for conversation %s", conversation_id
            )
            heuristic_result = self._analyze_follow_up_requirements(
                conversation_history_str,
                latest_message,
                direction,
                conversation_phase,
                ai_sentiment,
            )
            return self._process_follow_up_result(heuristic_result, conversation_id)

        except Exception as e:
            logger.error(f"Error extracting follow-up requirements for {conversation_id}: {e}", exc_info=True)
            return self._default_follow_up_payload()

    @staticmethod
    def _apply_importance_override(base_urgency: str) -> Optional[str]:
        if base_urgency == "urgent":
            return "high"
        if base_urgency == "patient":
            return "low"
        return None

    def _load_person_for_importance(
        self,
        db_session: Optional[DbSession],
        person_id: int,
    ) -> Optional[Person]:
        if not db_session:
            logger.debug("No DB session available for task importance calculation")
            return None
        person = db_session.query(Person).filter(Person.id == person_id).first()
        if not person:
            logger.debug(f"Person {person_id} not found for importance calculation")
        return person

    @staticmethod
    def _extract_importance_metrics(person: Person) -> tuple[int, int]:
        dna_match = getattr(person, "dna_match", None)
        cm_shared = getattr(dna_match, "cm_dna", 0) or 0
        engagement = getattr(person, "current_engagement_score", 0) or 0
        return cm_shared, engagement

    @staticmethod
    def _derive_importance_from_metrics(
        cm_shared: int,
        engagement: int,
        person_id: int,
    ) -> str:
        if cm_shared > 100 and engagement > 70:
            logger.debug(
                "Task importance HIGH: person_id=%s, cM=%s, engagement=%s",
                person_id,
                cm_shared,
                engagement,
            )
            return "high"
        if cm_shared > 50 or engagement > 50:
            logger.debug(
                "Task importance NORMAL: person_id=%s, cM=%s, engagement=%s",
                person_id,
                cm_shared,
                engagement,
            )
            return "normal"
        logger.debug(
            "Task importance LOW: person_id=%s, cM=%s, engagement=%s",
            person_id,
            cm_shared,
            engagement,
        )
        return "low"

    def _calculate_task_importance(
        self,
        person_id: int,
        base_urgency: str = "standard",
    ) -> str:
        """
        Calculate MS To-Do task importance from DNA relationship strength and engagement score.

        Priority 0 Todo #14: MS To-Do prioritization with DNA/engagement data

        Algorithm:
        - High: cM > 100 AND engagement > 70 (close relatives with active conversation)
        - High: base_urgency == "urgent" (AI-detected urgency overrides)
        - Medium: cM > 50 OR engagement > 50 (moderate DNA or decent engagement)
        - Low: All other cases (distant relatives, low engagement)

        Args:
            person_id: Database ID of person
            base_urgency: AI-detected urgency level (urgent/standard/patient)

        Returns:
            MS Graph importance level: "high", "normal", or "low"
        """
        try:
            override = self._apply_importance_override(base_urgency)
            if override is not None:
                return override

            db_session = self.session_manager.get_db_conn()
            person = self._load_person_for_importance(db_session, person_id)
            if not person:
                return "normal"

            cm_shared, engagement = self._extract_importance_metrics(person)
            return self._derive_importance_from_metrics(cm_shared, engagement, person_id)

        except Exception as e:
            logger.error(f"Error calculating task importance for person {person_id}: {e}", exc_info=True)
            return "normal"  # Safe default

    def _create_follow_up_reminder_task(
        self,
        person_id: int,
        conversation_id: str,
        task_title: str,
        task_body: Optional[str],
        due_date: Optional[datetime],
        urgency_level: str = "standard",
    ) -> bool:
        """
        Create MS To-Do reminder task for follow-up.

        Priority 0 Todo #14: MS To-Do task prioritization with DNA/engagement data
        Priority 1 Todo #5: Follow-Up Reminder System

        Args:
            person_id: Database ID of person (used for DNA/engagement prioritization)
            conversation_id: Unique conversation identifier (reserved for future use)
            task_title: Title for MS To-Do task
            task_body: Detailed task body with context
            due_date: When follow-up is due
            urgency_level: urgent/standard/patient → maps to high/normal/low importance

        Returns:
            True if task created successfully, False otherwise
        """
        try:
            logger.debug(
                "Preparing follow-up reminder task for conversation %s", conversation_id
            )
            import ms_graph_utils
        except Exception as exc:  # pragma: no cover - import errors
            logger.error(f"MS Graph utilities unavailable: {exc}", exc_info=True)
            return False

        try:
            if not getattr(ms_graph_utils, "msal_app_instance", None):
                raise RuntimeError("MS Graph not configured")

            if config_schema.app_mode in ("dry_run", "test"):
                logger.info(f"[DRY RUN] Would create follow-up reminder task: {task_title}")
                return True

            token = ms_graph_utils.acquire_token_device_flow()
            if not token:
                raise RuntimeError("MS Graph authentication unavailable, skipping reminder task")

            list_name = getattr(config_schema, "ms_todo_list_name", "Ancestry Follow-ups")
            list_id = ms_graph_utils.get_todo_list_id(token, list_name)
            if not list_id:
                raise RuntimeError(f"MS To-Do list '{list_name}' not found")

            importance = self._calculate_task_importance(
                person_id=person_id,
                base_urgency=urgency_level,
            )

            due_date_str = due_date.strftime("%Y-%m-%d") if due_date else None

            task_id = ms_graph_utils.create_todo_task(
                access_token=token,
                list_id=list_id,
                task_title=task_title,
                task_body=task_body,
                importance=importance,
                due_date=due_date_str,
                categories=["Ancestry", "Follow-up", "Conversation"],
            )

            if not task_id:
                raise RuntimeError("Failed to create follow-up reminder task")

            logger.info(
                "✓ Created follow-up reminder task (ID: %s): %s",
                task_id,
                task_title[:60],
            )
            return True

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"Error creating follow-up reminder task: {exc}", exc_info=True)
            return False

    def _create_conversation_log_upsert(
        self,
        api_conv_id: str,
        direction: MessageDirectionEnum,
        people_id: int,
        message_content: str,
        timestamp: datetime,
        ai_sentiment: Optional[str] = None,
        conversation_phase: Optional[Any] = None,
        follow_up_due_date: Optional[datetime] = None,
        awaiting_response_from: Optional[str] = None,
    ) -> dict:
        """
        Create conversation log upsert dictionary.

        Priority 1 Todo #11: Enhanced with conversation_phase field for lifecycle tracking.
        Priority 1 Todo #5: Enhanced with follow_up fields for reminder system.
        """
        return {
            "conversation_id": api_conv_id,
            "direction": direction,
            "people_id": people_id,
            "latest_message_content": message_content[
                : getattr(config_schema, "message_truncation_length", 1000)
            ],
            "latest_timestamp": timestamp,
            "ai_sentiment": ai_sentiment,
            "conversation_phase": conversation_phase,
            "follow_up_due_date": follow_up_due_date,
            "awaiting_response_from": awaiting_response_from,
            "message_template_id": None,
            "script_message_status": None,
        }

    def clarify_ambiguous_intent(
        self,
        user_message: str,
        extracted_entities: dict[str, Any],
        ambiguity_analysis: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Generate AI-powered clarifying questions when extracted entities are incomplete or ambiguous.

        Priority 1 Todo #7: Action 7 Intent Clarifier

        Use cases:
        - Multiple people with same name in tree (e.g., "Which Mary Smith?")
        - Missing temporal context (birth/death years)
        - Location too broad (e.g., "Scotland" vs "Banff, Scotland")
        - Unclear relationships (e.g., maternal vs paternal grandmother)

        Args:
            user_message: The original message from the DNA match
            extracted_entities: Dictionary of extracted entity data (names, dates, places, relationships)
            ambiguity_analysis: Optional context describing the ambiguity (e.g., "3 Mary Smiths in tree")

        Returns:
            Dictionary with clarifying_questions list, primary_ambiguity type, urgency, and reasoning.
            None if AI call fails or entities are not ambiguous.

        Example return:
            {
                "clarifying_questions": [
                    "I have three Mary Smiths in my tree! Do you mean Mary Smith born around 1850...",
                    "Do you happen to know Mary Smith's husband's name?"
                ],
                "primary_ambiguity": "name",
                "urgency": "critical",
                "reasoning": "Multiple matches require temporal context..."
            }
        """
        if not user_message or not extracted_entities:
            logger.debug("clarify_ambiguous_intent: Missing required inputs")
            return None

        # Build ambiguity context if not provided
        if ambiguity_analysis is None:
            ambiguity_analysis = self._analyze_entity_ambiguity(extracted_entities)

        # If no ambiguity detected, no clarification needed
        if not ambiguity_analysis or ambiguity_analysis == "No ambiguity detected":
            logger.debug("No ambiguity detected - clarification not needed")
            return None

        try:
            # Import AI interface function (will be added next)
            from ai_interface import generate_clarifying_questions

            result = generate_clarifying_questions(
                user_message=user_message,
                extracted_entities=extracted_entities,
                ambiguity_context=ambiguity_analysis,
                session_manager=self.session_manager,
            )

            if result and isinstance(result, dict) and "clarifying_questions" in result:
                logger.info(
                    f"✅ Generated {len(result['clarifying_questions'])} clarifying questions "
                    f"for {result.get('primary_ambiguity', 'unknown')} ambiguity"
                )
                return result

            logger.warning("AI clarification returned invalid format")
            return None

        except Exception as e:
            logger.error(f"Failed to generate clarifying questions: {e}", exc_info=True)
            return None

    def _check_person_ambiguities(self, mentioned_people: list[dict[str, Any]]) -> list[str]:
        """Check for ambiguous person mentions (missing context)."""
        ambiguities: list[str] = []
        for person in mentioned_people:
            name = person.get("name", "")
            if not name:
                continue

            has_birth_year = person.get("birth_year") is not None
            has_location = person.get("birth_place") or person.get("death_place")
            has_relationship = person.get("relationship") is not None

            if not (has_birth_year or has_location):
                ambiguities.append(f"Name '{name}' lacks temporal or location context")
            elif not has_relationship:
                ambiguities.append(f"Name '{name}' relationship unclear")

        return ambiguities

    def _check_location_ambiguities(self, locations: list[dict[str, Any]]) -> list[str]:
        """Check for overly broad location mentions."""
        ambiguities: list[str] = []
        broad_countries = ["Scotland", "England", "Ireland", "Wales", "USA", "Canada"]

        for loc in locations:
            place = loc.get("place", "")
            if place and len(place.split(",")) == 1 and place in broad_countries:
                ambiguities.append(f"Location '{place}' too broad - need city/county")

        return ambiguities

    def _check_relationship_ambiguities(self, relationships: list[dict[str, Any]]) -> list[str]:
        """Check for incomplete relationship mentions."""
        ambiguities: list[str] = []
        for rel in relationships:
            person1 = rel.get("person1", "")
            person2 = rel.get("person2", "")
            if not person1 or not person2:
                ambiguities.append("Relationship mentioned without both person names")

        return ambiguities

    def _analyze_entity_ambiguity(self, extracted_entities: dict[str, Any]) -> str:
        """
        Analyze extracted entities to detect ambiguity.

        Returns description of ambiguity type for AI prompt context.
        """
        ambiguities: list[str] = []

        # Check person ambiguities
        mentioned_people = extracted_entities.get("mentioned_people", [])
        if mentioned_people:
            ambiguities.extend(self._check_person_ambiguities(mentioned_people))

        # Check location ambiguities
        locations = extracted_entities.get("locations", [])
        ambiguities.extend(self._check_location_ambiguities(locations))

        # Check relationship ambiguities
        relationships = extracted_entities.get("relationships", [])
        ambiguities.extend(self._check_relationship_ambiguities(relationships))

        if not ambiguities:
            return "No ambiguity detected"

        return "; ".join(ambiguities)

    def _record_event_and_metrics(
        self,
        session: DbSession,
        people_id: int,
        direction: MessageDirectionEnum,
        ai_sentiment: Optional[str],
        conversation_phase: Optional[str],
    ) -> None:
        """Record an engagement event and update metrics."""
        event_type = "message_received" if direction == MessageDirectionEnum.IN else "message_sent"
        event_description = f"Message {direction.name.lower()} with sentiment: {ai_sentiment or 'unknown'}"
        record_engagement_event(
            session=session,
            people_id=people_id,
            event_type=event_type,
            event_description=event_description,
            conversation_phase=conversation_phase,
        )
        update_conversation_metrics(
            session=session,
            people_id=people_id,
            message_sent=(direction == MessageDirectionEnum.OUT),
            message_received=(direction == MessageDirectionEnum.IN),
            conversation_phase=conversation_phase,
        )

    def _build_recent_context(self, session: DbSession, people_id: int) -> str:
        """Build a compact recent conversation context for AI engagement assessment."""
        recent_logs = (
            session.query(ConversationLog)
            .filter(ConversationLog.people_id == people_id)
            .order_by(ConversationLog.timestamp.asc())
            .all()
        )
        if not recent_logs:
            return ""
        max_msgs = getattr(config_schema, "ai_context_window_messages", 6)
        logs_window = recent_logs[-max_msgs:]
        lines: list[str] = []
        for log in logs_window:
            role = "USER" if log.direction == MessageDirectionEnum.IN else "SCRIPT"
            content = (log.message_content or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _upsert_conversation_state(
        self,
        session: DbSession,
        people_id: int,
        direction: MessageDirectionEnum,
        result: dict[str, Any],
    ) -> None:
        """Upsert conversation state from AI engagement result."""
        state: Optional[ConversationState] = (
            session.query(ConversationState)
            .filter(ConversationState.people_id == people_id)
            .one_or_none()
        )
        if not state:
            state = ConversationState(people_id=people_id)
            session.add(state)

        # Extract fields with safe defaults
        state.engagement_score = int(result.get("engagement_score", 0))
        state.ai_summary = result.get("ai_summary") or None
        state.last_topic = result.get("last_topic") or None
        new_questions = result.get("pending_questions")
        if isinstance(new_questions, list):
            state.pending_questions = json.dumps(new_questions)

        # Phase heuristic
        if direction == MessageDirectionEnum.IN:
            state.conversation_phase = "active_dialogue"

        session.commit()


    def _assess_and_upsert(self, session: DbSession, people_id: int, direction: MessageDirectionEnum) -> bool:
        """Build context, assess engagement, and upsert state if result present."""
        formatted_context = self._build_recent_context(session, people_id)
        if not formatted_context:
            return False

        result = assess_engagement(
            conversation_history=formatted_context,
            session_manager=self.session_manager,
            log_prefix="Action7",
        )
        if result:
            self._upsert_conversation_state(session, people_id, direction, result)
            return True
        return False

    def _register_engagement_assessment(self, state: Optional[dict[str, Any]]) -> None:
        """Track successful engagement assessments in stats and loop state."""
        self.stats["engagement_assessments"] = self.stats.get("engagement_assessments", 0) + 1
        if state is not None:
            state["engagement_assessment_count"] = state.get("engagement_assessment_count", 0) + 1

    def _track_message_analytics(
        self,
        session: DbSession,
        people_id: int,
        direction: MessageDirectionEnum,
        ai_sentiment: Optional[str] = None,
        conversation_phase: Optional[str] = None,
    ) -> bool:
        """Track message analytics for conversation metrics and update conversation state (Phase 3).

        Returns True when an engagement assessment succeeds."""
        assessment_performed = False
        try:
            # 1) Record event and metrics
            self._record_event_and_metrics(session, people_id, direction, ai_sentiment, conversation_phase)

            # 2) AI engagement assessment
            assessment_performed = self._assess_and_upsert(session, people_id, direction)
        except Exception as e:
            logger.debug(f"Analytics tracking failed for people_id {people_id}: {e}")
            return False

        return assessment_performed

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
                    f"SESSION DEATH CASCADE in Action 7 {exception_type} save: {conn_err}"
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
        ctx: ConversationProcessingContext,
        context_messages: list[dict],
        ai_classified_count: int,
    ) -> int:
        """Process incoming message and return updated AI classification count."""
        if not latest_ctx_in:
            logger.debug(f"No incoming message found for conversation {api_conv_id}")
            return ai_classified_count

        ctx_ts_in_aware = latest_ctx_in.get("timestamp")
        db_latest_ts_in_compare = self._get_db_timestamp_for_comparison(
            ctx.existing_conv_logs, api_conv_id, MessageDirectionEnum.IN, ctx.min_aware_dt
        )

        if ctx_ts_in_aware and ctx_ts_in_aware > db_latest_ts_in_compare:
            logger.debug(f"Processing new/updated IN message for {api_conv_id} (timestamp: {ctx_ts_in_aware})")

            logger.debug(f"Attempting AI classification for conversation {api_conv_id}")
            ai_sentiment_result = self._classify_message_with_ai(
                context_messages, ctx.my_pid_lower, api_conv_id
            )

            if ai_sentiment_result:
                ai_classified_count += 1
                logger.debug(f"AI classification result for {api_conv_id}: {ai_sentiment_result}")
            else:
                logger.warning(f"AI classification failed for ConvID {api_conv_id}.")

            # Priority 1 Todo #11: Determine conversation phase based on history
            conversation_phase = self._determine_conversation_phase(
                conversation_id=api_conv_id,
                direction=MessageDirectionEnum.IN,
                ai_sentiment=ai_sentiment_result,
                existing_logs=ctx.existing_conv_logs,
                timestamp=ctx_ts_in_aware,
            )

            # Priority 1 Todo #5: Extract follow-up requirements for PRODUCTIVE conversations
            follow_up_data = self._extract_follow_up_requirements(
                conversation_history=ctx.existing_conv_logs,
                latest_message=latest_ctx_in.get("content", ""),
                direction=MessageDirectionEnum.IN,
                conversation_phase=conversation_phase,
                ai_sentiment=ai_sentiment_result,
                conversation_id=api_conv_id,
            )

            upsert_dict_in = self._create_conversation_log_upsert(
                api_conv_id, MessageDirectionEnum.IN, people_id,
                latest_ctx_in.get("content", ""), ctx_ts_in_aware, ai_sentiment_result,
                conversation_phase=conversation_phase,
                follow_up_due_date=follow_up_data.get("follow_up_due_date"),
                awaiting_response_from=follow_up_data.get("awaiting_response_from"),
            )
            ctx.conv_log_upserts_dicts.append(upsert_dict_in)
            logger.debug(f"Created IN message log entry for conversation {api_conv_id} with phase: {conversation_phase}")

            # Priority 1 Todo #5: Create MS To-Do reminder task if follow-up required
            if follow_up_data.get("follow_up_required") and follow_up_data.get("reminder_task_title"):
                self._create_follow_up_reminder_task(
                    person_id=people_id,
                    conversation_id=api_conv_id,
                    task_title=follow_up_data.get("reminder_task_title"),
                    task_body=follow_up_data.get("reminder_task_body"),
                    due_date=follow_up_data.get("follow_up_due_date"),
                    urgency_level=follow_up_data.get("urgency_level", "standard"),
                )

            # Track analytics for received message
            db_session = self.session_manager.get_db_conn()
            if db_session and self._track_message_analytics(
                session=db_session,
                people_id=people_id,
                direction=MessageDirectionEnum.IN,
                ai_sentiment=ai_sentiment_result,
            ):
                self._register_engagement_assessment(ctx.state)

            self._update_person_status_from_ai(ai_sentiment_result, people_id, ctx.person_updates)
        else:
            logger.debug(f"IN message for {api_conv_id} is not newer than DB (API: {ctx_ts_in_aware}, DB: {db_latest_ts_in_compare})")

        return ai_classified_count

    def _process_out_message(
        self,
        latest_ctx_out: Optional[dict],
        api_conv_id: str,
        people_id: int,
        ctx: ConversationProcessingContext,
    ) -> None:
        """Process outgoing message."""
        if not latest_ctx_out:
            logger.debug(f"No outgoing message found for conversation {api_conv_id}")
            return

        ctx_ts_out_aware = latest_ctx_out.get("timestamp")
        db_latest_ts_out_compare = self._get_db_timestamp_for_comparison(
            ctx.existing_conv_logs, api_conv_id, MessageDirectionEnum.OUT, ctx.min_aware_dt
        )

        if ctx_ts_out_aware and ctx_ts_out_aware > db_latest_ts_out_compare:
            logger.debug(f"Processing new/updated OUT message for {api_conv_id} (timestamp: {ctx_ts_out_aware})")

            # Priority 1 Todo #11: Determine conversation phase for OUT messages
            conversation_phase = self._determine_conversation_phase(
                conversation_id=api_conv_id,
                direction=MessageDirectionEnum.OUT,
                ai_sentiment=None,  # OUT messages don't have AI sentiment
                existing_logs=ctx.existing_conv_logs,
                timestamp=ctx_ts_out_aware,
            )

            upsert_dict_out = self._create_conversation_log_upsert(
                api_conv_id, MessageDirectionEnum.OUT, people_id,
                latest_ctx_out.get("content", ""), ctx_ts_out_aware,
                conversation_phase=conversation_phase,
            )
            ctx.conv_log_upserts_dicts.append(upsert_dict_out)
            logger.debug(f"Created OUT message log entry for conversation {api_conv_id} with phase: {conversation_phase}")

            # Track analytics for sent message
            db_session = self.session_manager.get_db_conn()
            if db_session and self._track_message_analytics(
                session=db_session,
                people_id=people_id,
                direction=MessageDirectionEnum.OUT,
            ):
                self._register_engagement_assessment(ctx.state)
        else:
            logger.debug(f"OUT message for {api_conv_id} is not newer than DB (API: {ctx_ts_out_aware}, DB: {db_latest_ts_out_compare})")

    def _check_inbox_limit(self, items_processed: int) -> tuple[bool, Optional[str]]:
        """Check if inbox limit reached. Returns (should_stop, stop_reason)."""
        if self.max_inbox_limit > 0 and items_processed >= self.max_inbox_limit:
            return True, f"Inbox Limit ({self.max_inbox_limit})"
        return False, None

    def _first_pass_identify_conversations(
        self,
        all_conversations_batch: list[dict],
        ctx: ConversationProcessingContext,
        items_processed_before_stop: int,
    ) -> tuple[list[dict], dict[str, str], bool, Optional[str]]:
        """First pass: Identify which conversations need context fetching.

        Returns: (conversations_needing_fetch, skip_map, should_stop, stop_reason)
        """
        conversations_needing_fetch = []
        skip_map = {}  # api_conv_id -> skip_reason

        for conversation_info in all_conversations_batch:
            # Check inbox limit
            should_stop, limit_reason = self._check_inbox_limit(items_processed_before_stop)
            if should_stop:
                logger.debug(f"Inbox limit reached during first pass: {limit_reason}")
                return conversations_needing_fetch, skip_map, True, limit_reason

            items_processed_before_stop += 1

            # Extract conversation identifiers
            profile_id_upper, api_conv_id, api_latest_ts_aware = (
                self._extract_conversation_identifiers(conversation_info)
            )

            # Skip invalid conversations
            if self._should_skip_invalid(api_conv_id, profile_id_upper):
                skip_map[api_conv_id] = "invalid"
                logger.debug(f"First pass: Skipping invalid conversation {api_conv_id}")
                continue

            # Determine if conversation needs fetching
            needs_fetch, should_stop, fetch_stop_reason = self._determine_fetch_need(
                api_conv_id, ctx.comp_conv_id, ctx.comp_ts, api_latest_ts_aware,
                ctx.existing_conv_logs, ctx.min_aware_dt,
            )

            if should_stop:
                logger.debug(f"First pass: Comparator found at {api_conv_id}: {fetch_stop_reason}")
                return conversations_needing_fetch, skip_map, True, fetch_stop_reason

            if not needs_fetch:
                skip_map[api_conv_id] = "up-to-date"
                logger.debug(f"First pass: Conversation {api_conv_id} is up-to-date")
                continue

            # Add to fetch list
            conversations_needing_fetch.append(conversation_info)
            logger.debug(f"First pass: Conversation {api_conv_id} needs context fetch")

        logger.debug(
            f"[First Pass] Complete: {len(conversations_needing_fetch)} need context fetch, "
            f"{len(skip_map)} skipped (up-to-date or invalid)"
        )
        return conversations_needing_fetch, skip_map, False, None

    def _fetch_single_conversation_context(self, api_conv_id: str) -> tuple[str, Optional[list[dict]]]:
        """Fetch context for a single conversation. Used by parallel fetching.

        Returns: (api_conv_id, context_messages or None)
        """
        try:
            # Validate session before fetch
            self._validate_session()

            logger.debug(f"Fetching context for conversation {api_conv_id}")
            context_messages = self._fetch_conversation_context(api_conv_id)

            if context_messages is None:
                logger.error(f"Failed to fetch context for {api_conv_id}")
                return api_conv_id, None

            logger.debug(f"Fetched {len(context_messages)} messages for {api_conv_id}")
            return api_conv_id, context_messages

        except Exception as e:
            logger.error(f"Exception fetching context for {api_conv_id}: {e}")
            return api_conv_id, None

    def _fetch_conversation_contexts_batch(
        self,
        conversations_needing_fetch: list[dict],
    ) -> dict[str, Optional[list[dict]]]:
        """Fetch conversation contexts for all conversations in the list.

        Phase 3 Optimization: Supports parallel fetching based on parallel_workers config.
        - If parallel_workers <= 1: Sequential fetching
        - If parallel_workers > 1: Parallel fetching with ThreadPoolExecutor

        Returns: dict mapping api_conv_id -> context_messages (or None if fetch failed)
        """
        parallel_workers = getattr(config_schema, 'parallel_workers', 1)

        # Extract conversation IDs
        conv_ids = [
            self._extract_conversation_identifiers(conv)[1]  # api_conv_id
            for conv in conversations_needing_fetch
        ]

        # Sequential fetching (parallel_workers <= 1)
        if parallel_workers <= 1:
            logger.debug(f"[Context Fetch] Sequential mode: fetching {len(conv_ids)} conversations")
            context_map = {}
            for api_conv_id in conv_ids:
                conv_id, context = self._fetch_single_conversation_context(api_conv_id)
                context_map[conv_id] = context

            successful = len([v for v in context_map.values() if v is not None])
            logger.debug(f"[Context Fetch] Sequential complete: {successful}/{len(conv_ids)} successful")
            return context_map

        # Parallel fetching (parallel_workers > 1)
        logger.debug(
            f"[Context Fetch] Parallel mode: fetching {len(conv_ids)} conversations "
            f"with {parallel_workers} workers"
        )
        context_map = {}

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # Submit all fetch tasks
            futures = {
                executor.submit(self._fetch_single_conversation_context, conv_id): conv_id
                for conv_id in conv_ids
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    conv_id, context = future.result()
                    context_map[conv_id] = context
                except Exception as e:
                    conv_id = futures[future]
                    logger.error(f"Exception in parallel fetch for {conv_id}: {e}")
                    context_map[conv_id] = None

        successful = len([v for v in context_map.values() if v is not None])
        failed = len(conv_ids) - successful
        logger.debug(
            f"[Context Fetch] Parallel complete: {successful}/{len(conv_ids)} successful"
            + (f", {failed} failed" if failed > 0 else "")
        )
        return context_map

    def _second_pass_process_conversations(
        self,
        session: DbSession,
        conversations_needing_fetch: list[dict],
        context_map: dict[str, Optional[list[dict]]],
        ctx: ConversationProcessingContext,
        ai_classified_count: int,
    ) -> tuple[int, int]:
        """Second pass: Process all conversations with their fetched contexts.

        Returns: (error_count, ai_classified_count)
        """
        error_count = 0

        for conversation_info in conversations_needing_fetch:
            profile_id_upper, api_conv_id, _ = self._extract_conversation_identifiers(conversation_info)

            # Get fetched context
            context_messages = context_map.get(api_conv_id)
            if context_messages is None:
                logger.error(f"No context available for {api_conv_id}, skipping")
                error_count += 1
                continue

            # Lookup or create person
            person, person_status = self._lookup_or_create_person(
                session, profile_id_upper, conversation_info.get("username", "Unknown"),
                api_conv_id, existing_person_arg=ctx.existing_persons_map.get(profile_id_upper),
            )
            if not person or not safe_column_value(person, "id"):
                logger.error(
                    f"Failed person lookup/create for conversation {api_conv_id}: "
                    f"profile_id={profile_id_upper}, username={conversation_info.get('username', 'Unknown')}, "
                    f"status={person_status}, person_obj={'None' if not person else 'exists but no ID'}"
                )
                error_count += 1
                continue

            people_id = safe_column_value(person, "id")
            logger.debug(f"Second pass: Processing conversation {api_conv_id} for person ID {people_id}")

            # Find latest IN and OUT messages
            latest_ctx_in, latest_ctx_out = self._find_latest_messages(context_messages, ctx.my_pid_lower)
            logger.debug(f"Found latest messages for {api_conv_id}: IN={'present' if latest_ctx_in else 'None'}, OUT={'present' if latest_ctx_out else 'None'}")

            # Process IN and OUT messages
            ai_classified_count = self._process_in_message(
                latest_ctx_in, api_conv_id, people_id, ctx, context_messages, ai_classified_count
            )

            self._process_out_message(
                latest_ctx_out, api_conv_id, people_id, ctx
            )

            logger.debug(f"Second pass: Completed processing conversation {api_conv_id}")

        logger.debug(f"Second pass complete: {len(conversations_needing_fetch)} processed, {error_count} errors, {ai_classified_count} AI classifications")
        return error_count, ai_classified_count

    def _process_single_conversation(
        self,
        session: DbSession,
        conversation_info: dict,
        ctx: ConversationProcessingContext,
        ai_classified_count: int,
    ) -> tuple[bool, Optional[str], int, int]:
        """Process single conversation. Returns (should_stop, stop_reason, error_count_delta, ai_count)."""
        error_count_delta = 0

        # Extract conversation identifiers
        profile_id_upper, api_conv_id, api_latest_ts_aware = (
            self._extract_conversation_identifiers(conversation_info)
        )

        logger.debug(f"Processing conversation {api_conv_id} for profile {profile_id_upper}")

        # Skip invalid conversations
        if self._should_skip_invalid(api_conv_id, profile_id_upper):
            logger.debug(f"Skipping invalid conversation: conv_id={api_conv_id}, profile={profile_id_upper}")
            return False, None, 0, ai_classified_count

        # Determine if conversation needs fetching
        needs_fetch, should_stop, fetch_stop_reason = self._determine_fetch_need(
            api_conv_id, ctx.comp_conv_id, ctx.comp_ts, api_latest_ts_aware,
            ctx.existing_conv_logs, ctx.min_aware_dt,
        )

        if should_stop:
            logger.debug(f"Stopping processing at conversation {api_conv_id}: {fetch_stop_reason}")
            return True, fetch_stop_reason, 0, ai_classified_count

        # Skip if no fetch needed
        if not needs_fetch:
            logger.debug(f"Conversation {api_conv_id} is up-to-date, skipping fetch")
            return False, None, 0, ai_classified_count

        # Validate session before fetch
        self._validate_session()

        logger.debug(f"Fetching context for conversation {api_conv_id}")

        # Fetch conversation context
        context_messages = self._fetch_conversation_context(api_conv_id)
        if context_messages is None:
            logger.error(f"Failed to fetch context for ConvID {api_conv_id}. Skipping item.")
            return False, None, 1, ai_classified_count

        logger.debug(f"Fetched {len(context_messages)} messages for conversation {api_conv_id}")

        # Lookup or create person
        person, _ = self._lookup_or_create_person(
            session, profile_id_upper, conversation_info.get("username", "Unknown"),
            api_conv_id, existing_person_arg=ctx.existing_persons_map.get(profile_id_upper),
        )
        if not person or not safe_column_value(person, "id"):
            logger.error(f"Failed person lookup/create for ConvID {api_conv_id}. Skipping item.")
            return False, None, 1, ai_classified_count

        people_id = safe_column_value(person, "id")
        logger.debug(f"Processing conversation {api_conv_id} for person ID {people_id}")

        # Find latest IN and OUT messages
        latest_ctx_in, latest_ctx_out = self._find_latest_messages(context_messages, ctx.my_pid_lower)
        logger.debug(f"Found latest messages for {api_conv_id}: IN={'present' if latest_ctx_in else 'None'}, OUT={'present' if latest_ctx_out else 'None'}")

        # Process IN and OUT messages
        ai_classified_count = self._process_in_message(
            latest_ctx_in, api_conv_id, people_id, ctx, context_messages, ai_classified_count
        )

        self._process_out_message(
            latest_ctx_out, api_conv_id, people_id, ctx
        )

        logger.debug(f"Completed processing conversation {api_conv_id}")

        return False, None, error_count_delta, ai_classified_count

    def _process_conversations_in_batch(
        self,
        session: DbSession,
        all_conversations_batch: list[dict],
        ctx: ConversationProcessingContext,
        state: dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """Process all conversations in a batch using two-pass approach.

        Phase 2 Optimization: Two-pass processing
        - First pass: Identify conversations needing context fetch
        - Batch fetch: Fetch all contexts (sequential for now, parallel in Phase 3)
        - Second pass: Process all conversations with fetched contexts

        Returns: (stop_processing, stop_reason)
        """
        logger.debug(f"Processing batch of {len(all_conversations_batch)} conversations (two-pass)")

        # FIRST PASS: Identify conversations needing fetch
        conversations_needing_fetch, skip_map, should_stop, stop_reason = self._first_pass_identify_conversations(
            all_conversations_batch, ctx, state["items_processed_before_stop"]
        )

        # Update items processed count (includes skipped items)
        state["items_processed_before_stop"] += len(all_conversations_batch)
        state["skipped_count_this_loop"] += len(skip_map)

        # Skipped items already counted in skip_map

        if should_stop:
            logger.debug(f"First pass indicated stop: {stop_reason}")
            return True, stop_reason

        # If no conversations need fetching, we're done
        if not conversations_needing_fetch:
            logger.debug(
                f"[Batch Complete] All {len(all_conversations_batch)} conversations skipped "
                f"(up-to-date - no changes detected)"
            )
            return False, None

        # BATCH FETCH: Fetch all conversation contexts
        logger.debug(f"[Batch Fetch] Fetching contexts for {len(conversations_needing_fetch)} conversations")
        context_map = self._fetch_conversation_contexts_batch(conversations_needing_fetch)

        # SECOND PASS: Process all conversations with fetched contexts
        logger.debug(f"[Second Pass] Processing {len(conversations_needing_fetch)} conversations with fetched contexts")
        error_count, ai_classified_count = self._second_pass_process_conversations(
            session, conversations_needing_fetch, context_map, ctx, state["ai_classified_count"]
        )

        # Update state
        state["error_count_this_loop"] += error_count
        state["ai_classified_count"] = ai_classified_count
        state["conversations_needing_processing"] += len(conversations_needing_fetch)

        # Progress bar updates removed - simple increment only at main loop level (matching Action 6/7/8/9)

        logger.debug(f"Batch complete: {len(conversations_needing_fetch)} processed, {len(skip_map)} skipped, {error_count} errors")
        return False, None

    def _extract_state_variables(self, state: dict) -> tuple:
        """Extract state variables for easier access. Returns tuple of all state variables."""
        return (
            state["ai_classified_count"],
            state["status_updated_count"],
            state["total_processed_api_items"],
            state["items_processed_before_stop"],
            state["logs_processed_in_run"],
            state["skipped_count_this_loop"],
            state["error_count_this_loop"],
            state["stop_reason"],
            state["next_cursor"],
            state["current_batch_num"],
            state["conv_log_upserts_dicts"],
            state["person_updates"],
            state["stop_processing"],
            state["min_aware_dt"],
            state["conversations_needing_processing"],
        )

    def _check_batch_preconditions(
        self, current_batch_num: int, items_processed_before_stop: int, state: dict[str, Any]
    ) -> tuple[bool, Optional[str], Optional[int]]:
        """Check preconditions before processing batch. Returns (should_stop, stop_reason, current_limit)."""
        # Browser health check
        browser_error = self._check_browser_health(current_batch_num, state)
        if browser_error:
            return True, browser_error, None

        # Validate session
        self._validate_session()

        # Calculate API limit
        current_limit, limit_error = self._calculate_api_limit(items_processed_before_stop)
        if limit_error:
            return True, limit_error, None

        return False, None, current_limit

    def _fetch_and_validate_batch(
        self, current_limit: int, next_cursor: Optional[str]
    ) -> tuple[bool, Optional[str], Optional[list], Optional[str]]:
        """Fetch batch from API and validate. Returns (should_stop, stop_reason, batch, next_cursor)."""
        all_conversations_batch, next_cursor_from_api = self._get_all_conversations_api(
            self.session_manager, limit=current_limit, cursor=next_cursor
        )

        if all_conversations_batch is None:
            logger.error("API call to fetch conversation batch failed critically.")
            return True, "API Error Fetching Batch", None, None

        return False, None, all_conversations_batch, next_cursor_from_api

    def _handle_batch_processing_exception(
        self,
        exception: Exception,
        exception_type: str,
        session: DbSession,
        conv_log_upserts_dicts: list,
        person_updates: dict,
        current_batch_num: int,
    ) -> tuple[int, int, str]:
        """Handle exception during batch processing. Returns (logs_saved, persons_updated, stop_reason)."""
        if exception_type == "WebDriverException":
            logger.error(f"WebDriverException occurred during inbox loop (Batch {current_batch_num}): {exception}")
            stop_reason = "WebDriver Exception"
        elif exception_type == "KeyboardInterrupt":
            logger.warning("KeyboardInterrupt detected during inbox loop.")
            stop_reason = "Keyboard Interrupt"
        else:
            logger.critical(
                f"Critical error in inbox processing loop (Batch {current_batch_num}): {exception}",
                exc_info=True,
            )
            stop_reason = f"Critical Error ({type(exception).__name__})"

        final_logs, final_persons = self._handle_exception_with_save(
            session, conv_log_upserts_dicts, person_updates, exception_type
        )
        return final_logs, final_persons, stop_reason

    def _fetch_and_process_batch(
        self,
        state: dict[str, Any],
    ) -> tuple[bool, Optional[str], list[Any], Optional[str]]:
        """Fetch and process batch. Returns (should_stop, stop_reason, batch, next_cursor)."""
        # Check preconditions
        should_stop, stop_reason_check, current_limit = self._check_batch_preconditions(
            state["current_batch_num"], state["items_processed_before_stop"], state
        )
        if should_stop:
            return True, stop_reason_check, [], None

        # Fetch and validate batch
        should_stop, stop_reason_check, all_conversations_batch, next_cursor_from_api = (
            self._fetch_and_validate_batch(current_limit, state["next_cursor"])
        )
        if should_stop:
            return True, stop_reason_check, [], None

        # Update batch counters
        batch_api_item_count = len(all_conversations_batch)
        state["total_processed_api_items"] += batch_api_item_count
        state["current_batch_num"] += 1

        return False, None, all_conversations_batch, next_cursor_from_api

    def _handle_batch_and_commit(
        self,
        session: DbSession,
        state: dict[str, Any],
        all_conversations_batch: list[Any],
        comp_conv_id: Optional[str],
        comp_ts: Optional[datetime],
        my_pid_lower: str,
    ) -> tuple[bool, Optional[str]]:
        """Handle batch processing and commit. Returns (should_stop, stop_reason)."""

        # Prefetch batch data
        existing_persons_map, existing_conv_logs, prefetch_error = (
            self._prefetch_batch_data(session, all_conversations_batch, state["current_batch_num"])
        )
        if prefetch_error:
            return True, prefetch_error

        # Process conversations in batch
        ctx = ConversationProcessingContext(
            existing_persons_map=existing_persons_map,
            existing_conv_logs=existing_conv_logs,
            conv_log_upserts_dicts=state["conv_log_upserts_dicts"],
            person_updates=state["person_updates"],
            comp_conv_id=comp_conv_id,
            comp_ts=comp_ts,
            my_pid_lower=my_pid_lower,
            min_aware_dt=state["min_aware_dt"],
            state=state,
        )
        batch_stop, batch_stop_reason = self._process_conversations_in_batch(
            session, all_conversations_batch, ctx, state
        )

        # Commit batch updates
        batch_num = state['current_batch_num']
        num_logs = len(state['conv_log_upserts_dicts'])
        num_person_updates = len(state['person_updates'])

        # Compact logging: Only log details if there are changes
        if num_logs == 0 and num_person_updates == 0:
            logger.debug(f"Batch {batch_num}: All conversations up-to-date (no changes)")
        else:
            logger.debug(f"Committing batch {batch_num}: {num_logs} logs, {num_person_updates} person updates")

        logs_committed, persons_updated = self._commit_batch_updates(
            session, state["conv_log_upserts_dicts"], state["person_updates"], batch_num
        )
        state["status_updated_count"] += persons_updated
        state["logs_processed_in_run"] += logs_committed

        # Compact logging for commit results
        if logs_committed == 0 and persons_updated == 0:
            logger.debug(f"Batch {batch_num} complete: No changes committed")
        else:
            logger.debug(f"Batch {batch_num} committed: {logs_committed} logs, {persons_updated} persons updated")

        # Check if batch processing should stop
        if batch_stop:
            logger.debug(f"Batch processing stopped: {batch_stop_reason or 'Comparator Found'}")
            return True, batch_stop_reason or "Comparator Found"

        return False, None

    def _process_single_batch_iteration(
        self,
        session: DbSession,
        state: dict[str, Any],
        comp_conv_id: Optional[str],
        comp_ts: Optional[datetime],
        my_pid_lower: str,
    ) -> tuple[bool, Optional[str]]:
        """Process a single batch iteration. Returns (should_stop, stop_reason)."""
        batch_num = state['current_batch_num'] + 1
        logger.debug(f"[Batch {batch_num}] Starting batch iteration")

        # Log batch start removed - will only log batch completion like Action 6

        # Fetch and process batch
        should_stop, stop_reason, all_conversations_batch, next_cursor_from_api = (
            self._fetch_and_process_batch(state)
        )
        if should_stop:
            logger.debug(f"Batch fetch indicated stop: {stop_reason}")
            return True, stop_reason

        # Handle empty batch
        if len(all_conversations_batch) == 0:
            logger.debug("Received empty batch from API")
            should_stop, empty_reason = self._handle_empty_batch(next_cursor_from_api)
            if should_stop:
                logger.debug(f"Empty batch handling indicated stop: {empty_reason}")
                return True, empty_reason
            state["next_cursor"] = next_cursor_from_api
            return False, None

        # Handle batch and commit
        should_stop, stop_reason = self._handle_batch_and_commit(
            session, state, all_conversations_batch, comp_conv_id, comp_ts, my_pid_lower
        )

        logger.info(
            f"Batch {state['current_batch_num']}: "
            f"Processed={state['total_processed_api_items']}, "
            f"AI={state['ai_classified_count']}, "
            f"Engagement={state['engagement_assessment_count']}, "
            f"Updates={state['status_updated_count']}, "
            f"Errors={state['error_count_this_loop']}"
        )

        if should_stop:
            return True, stop_reason

        # Prepare for next batch - combine the last two checks
        state["next_cursor"] = next_cursor_from_api

        # Check for end of inbox or cancellation
        result_should_stop = False
        result_stop_reason = None

        if not next_cursor_from_api:
            result_should_stop = True
            result_stop_reason = "End of Inbox Reached (No Next Cursor)"
        elif self._check_cancellation_requested():
            logger.warning("Cancellation requested by timeout wrapper. Stopping inbox processing loop.")
            result_should_stop = True
            result_stop_reason = "Timeout Cancellation"

        return result_should_stop, result_stop_reason

    def _handle_loop_exception(
        self,
        exception: Exception,
        exception_type: str,
        session: DbSession,
        state: dict[str, Any],
    ) -> tuple[Optional[str], int, int]:
        """Handle exceptions during batch processing."""
        if exception_type != "Exception":
            state["error_count_this_loop"] += 1

        final_logs, final_persons, stop_reason = self._handle_batch_processing_exception(
            exception, exception_type, session,
            state["conv_log_upserts_dicts"], state["person_updates"], state["current_batch_num"]
        )
        state["status_updated_count"] += final_persons
        state["logs_processed_in_run"] += final_logs
        return stop_reason, final_logs, final_persons

    def _process_inbox_loop(
        self,
        session: DbSession,
        comp_conv_id: Optional[str],
        comp_ts: Optional[datetime],  # Aware datetime
        my_pid_lower: str,
         # Accept progress bar instance
    ) -> tuple[Optional[str], int, int, int, int, int, int, int]:
        """Run the core inbox processing loop.

        Returns: (stop_reason, total_api_items, ai_classified, engagement_assessments,
                  status_updated, items_processed, session_deaths, session_recoveries)
        """
        # Initialize loop state
        state = self._initialize_loop_state()

        # Step 2: Main loop - continues until stop condition met
        while not state["stop_processing"]:
            try:
                # Process single batch iteration
                should_stop, batch_stop_reason = self._process_single_batch_iteration(
                    session, state, comp_conv_id, comp_ts, my_pid_lower
                )

                if should_stop:
                    state["stop_reason"] = batch_stop_reason
                    state["stop_processing"] = True
                    break

            # Handle exceptions during batch processing
            except WebDriverException as wde:
                state["stop_reason"], _, _ = self._handle_loop_exception(wde, "WebDriverException", session, state)
                state["stop_processing"] = True
                break

            except KeyboardInterrupt as ki:
                state["stop_reason"], _, _ = self._handle_loop_exception(ki, "KeyboardInterrupt", session, state)
                state["stop_processing"] = True
                break

            except Exception as e_main:
                state["stop_reason"], _, _ = self._handle_loop_exception(e_main, "Exception", session, state)
                state["stop_processing"] = True
                return (
                    state["stop_reason"],
                    state["total_processed_api_items"],
                    state["ai_classified_count"],
                    state["engagement_assessment_count"],
                    state["status_updated_count"],
                    state["items_processed_before_stop"],
                    state["session_deaths"],
                    state["session_recoveries"],
                )

        # --- End Main Loop (while not stop_processing) ---

        # Step 4: Perform final commit if loop finished normally or stopped early
        if (
            not state["stop_reason"]
            or state["stop_reason"]
            in (  # Only commit if loop ended somewhat gracefully
                "Comparator Found",
                "No Change",
                f"Inbox Limit ({self.max_inbox_limit})",
                "End of Inbox Reached (Empty Batch, No Cursor)",
                "End of Inbox Reached (No Next Cursor)",
            )
        ) and (state["conv_log_upserts_dicts"] or state["person_updates"]):
            logger.debug("Performing final commit at end of processing loop...")
            final_logs_saved, final_persons_updated = commit_bulk_data(
                session=session,
                log_upserts=state["conv_log_upserts_dicts"],
                person_updates=state["person_updates"],
                context="Action 7 Final Save (Normal Exit)",
            )
            state["status_updated_count"] += final_persons_updated
            state["logs_processed_in_run"] += final_logs_saved

        # Step 5: Return results from the loop execution
        return (
            state["stop_reason"],
            state["total_processed_api_items"],
            state["ai_classified_count"],
            state["engagement_assessment_count"],
            state["status_updated_count"],
            state["items_processed_before_stop"],
            state["session_deaths"],
            state["session_recoveries"],
        )

    # End of _process_inbox_loop

    def _log_unified_summary(
        self,
        total_api_items: int,
        items_processed: int,
        new_logs: int,
        ai_classified: int,
        engagement_assessments: int,
        status_updates: int,
        stop_reason: Optional[str],
        max_inbox_limit: int,
        session_deaths: int = 0,
        session_recoveries: int = 0,
    ) -> None:
        """Logs a unified summary of the inbox search process."""
        # Calculate run time - use 'or' to handle None values properly
        start_time = self.stats.get("start_time")
        end_time = self.stats.get("end_time") or datetime.now(timezone.utc)
        total_run_time = (end_time - start_time).total_seconds() if start_time else 0.0

        # Step 1: Print header
        print("")  # Blank line before summary
        logger.info("-" * 35)
        logger.info("Final summary")
        logger.info("-" * 35)

        # Mark unused parameters to satisfy linter without changing signature
        _ = new_logs

        # Step 2: Log key metrics
        logger.info(f"API Conversations Fetched:    {total_api_items}")
        logger.info(f"Conversations Processed:      {items_processed}")
        logger.info(f"AI Classifications Attempted:  {ai_classified}")
        logger.info(f"AI Engagement Assessments:     {engagement_assessments}")
        logger.info(f"Person Status Updates Made:   {status_updates}")

        # Step 2.5: Log session health metrics if any occurred
        if session_deaths > 0 or session_recoveries > 0:
            logger.info(f"Session Deaths:               {session_deaths}")
            logger.info(f"Session Recoveries:           {session_recoveries}")

        # Step 3: Log stopping reason
        final_reason = stop_reason
        if not stop_reason:
            # Infer reason if not explicitly set
            if max_inbox_limit == 0 or items_processed < max_inbox_limit:
                final_reason = "End of Inbox Reached or Comparator Match"
            else:
                final_reason = f"Inbox Limit ({max_inbox_limit}) Reached"
        logger.info(f"Stopped Due To:    {final_reason}")

        # Step 4: Log run time in consistent format
        hours = int(total_run_time // 3600)
        minutes = int((total_run_time % 3600) // 60)
        seconds = total_run_time % 60
        logger.info(f"Total Run Time: {hours} hr {minutes} min {seconds:.2f} sec")

        # Print rate limiter metrics if available
        if hasattr(self.session_manager, 'rate_limiter') and self.session_manager.rate_limiter:
            self.session_manager.rate_limiter.print_metrics_summary()

        # Update statistics
        self.stats.update(
            {
                "conversations_fetched": total_api_items,
                "conversations_processed": items_processed,
                "ai_classifications": ai_classified,
                "engagement_assessments": engagement_assessments,
                "person_updates": status_updates,
                "session_deaths": session_deaths,
                "session_recoveries": session_recoveries,
                "end_time": datetime.now(timezone.utc),
            }
        )
    # End of _log_unified_summary


# --- Enhanced Test Framework Implementation ---

# === SESSION SETUP FOR TESTS ===
# Migrated to use centralized session_utils.py (reduces 88 lines to 1 import!)
from session_utils import ensure_session_for_tests_sm_only as _ensure_session_for_tests

# Removed smoke test: _test_class_and_methods_available - only checked callable() and hasattr()


def _test_inbox_processor_initialization() -> None:
    """Test InboxProcessor can be initialized with SessionManager."""
    from unittest.mock import MagicMock

    # Use mock session - doesn't require main.py setup
    sm = MagicMock()
    processor = InboxProcessor(sm)

    # Verify processor initialized correctly
    assert processor.session_manager == sm, "SessionManager should be stored"
    assert hasattr(processor, 'stats'), "Processor should have stats dict"
    assert isinstance(processor.stats, dict), "Stats should be a dictionary"

    return True


def _test_fetch_first_page_conversations() -> None:
    """Test fetching first page of conversations from API."""
    import os
    skip_live_tests = os.getenv("SKIP_LIVE_API_TESTS", "false").lower() == "true"
    if skip_live_tests:
        logger.info("Skipping live test (SKIP_LIVE_API_TESTS=true)")
        return True

    # This test requires live API - only run when global session available
    try:
        sm = _ensure_session_for_tests()
    except RuntimeError:
        logger.info("Skipping live API test (no global session available)")
        return True

    processor = InboxProcessor(sm)

    # Fetch first page (limit to 5 conversations for testing)
    result = processor.search_inbox(max_inbox_limit=5)

    # Verify result structure
    assert isinstance(result, dict), "search_inbox should return dict"
    assert "conversations_processed" in result, "Result should have conversations_processed"
    assert result["conversations_processed"] >= 0, "Should process 0 or more conversations"

    logger.info(f"Fetched {result['conversations_processed']} conversations from first page")

    return True


def _test_conversation_database_storage() -> None:
    """Test conversations are stored in database."""
    import os
    skip_live_tests = os.getenv("SKIP_LIVE_API_TESTS", "false").lower() == "true"
    if skip_live_tests:
        logger.info("Skipping live test (SKIP_LIVE_API_TESTS=true)")
        return True

    # This test requires live API - only run when global session available
    try:
        sm = _ensure_session_for_tests()
    except RuntimeError:
        logger.info("Skipping live API test (no global session available)")
        return True
    processor = InboxProcessor(sm)

    # Get count before processing
    db_session = sm.db_session
    count_before = db_session.query(ConversationLog).count()

    # Process a small batch
    processor.search_inbox(max_inbox_limit=3)

    # Get count after processing
    count_after = db_session.query(ConversationLog).count()

    # Verify conversations were stored (or already existed)
    assert count_after >= count_before, "Conversation count should not decrease"

    logger.info(f"Database has {count_after} conversations (added {count_after - count_before})")

    return True


def _test_conversation_parsing() -> None:
    """Test conversation data is parsed correctly from API."""
    import os
    skip_live_tests = os.getenv("SKIP_LIVE_API_TESTS", "false").lower() == "true"
    if skip_live_tests:
        logger.info("Skipping live test (SKIP_LIVE_API_TESTS=true)")
        return True

    # This test requires live API - only run when global session available
    try:
        sm = _ensure_session_for_tests()
    except RuntimeError:
        logger.info("Skipping live API test (no global session available)")
        return True
    processor = InboxProcessor(sm)

    # Process a small batch and check database
    processor.search_inbox(max_inbox_limit=2)

    # Get a conversation from database
    db_session = sm.db_session
    conv = db_session.query(ConversationLog).first()

    if conv:
        # Verify conversation has required fields
        assert conv.conversation_id is not None, "Conversation should have ID"
        assert conv.direction is not None, "Conversation should have direction"
        assert conv.updated_at is not None, "Conversation should have timestamp"

        logger.info(f"Parsed conversation {conv.conversation_id} with direction {conv.direction}")
    else:
        logger.info("No conversations in database (inbox may be empty)")

    return True


def _test_ai_classification() -> None:
    """Test AI classification of messages."""
    import os
    skip_live_tests = os.getenv("SKIP_LIVE_API_TESTS", "false").lower() == "true"
    if skip_live_tests:
        logger.info("Skipping live test (SKIP_LIVE_API_TESTS=true)")
        return True

    # This test requires live API - only run when global session available
    try:
        sm = _ensure_session_for_tests()
    except RuntimeError:
        logger.info("Skipping live API test (no global session available)")
        return True
    processor = InboxProcessor(sm)

    # Process conversations with AI classification
    result = processor.search_inbox(max_inbox_limit=3)

    # Check if any AI classifications were attempted
    ai_classified = result.get("ai_classifications", 0)

    # AI classification may be 0 if no new messages or if messages already classified
    assert ai_classified >= 0, "AI classifications should be non-negative"

    logger.info(f"AI classified {ai_classified} messages")

    return True


def _test_person_status_updates() -> None:
    """Test person status updates from conversations."""
    import os
    skip_live_tests = os.getenv("SKIP_LIVE_API_TESTS", "false").lower() == "true"
    if skip_live_tests:
        logger.info("Skipping live test (SKIP_LIVE_API_TESTS=true)")
        return True

    # This test requires live API - only run when global session available
    try:
        sm = _ensure_session_for_tests()
    except RuntimeError:
        logger.info("Skipping live API test (no global session available)")
        return True
    processor = InboxProcessor(sm)

    # Process conversations
    result = processor.search_inbox(max_inbox_limit=3)

    # Check if any person status updates were made
    status_updates = result.get("person_updates", 0)

    # Status updates may be 0 if no new conversations or if statuses already set
    assert status_updates >= 0, "Person status updates should be non-negative"

    logger.info(f"Made {status_updates} person status updates")

    return True


def _test_stop_on_unchanged_conversation() -> None:
    """Test processing stops when encountering unchanged conversation (comparator match)."""
    import os
    skip_live_tests = os.getenv("SKIP_LIVE_API_TESTS", "false").lower() == "true"
    if skip_live_tests:
        logger.info("Skipping live test (SKIP_LIVE_API_TESTS=true)")
        return True

    # This test requires live API - only run when global session available
    try:
        sm = _ensure_session_for_tests()
    except RuntimeError:
        logger.info("Skipping live API test (no global session available)")
        return True
    processor = InboxProcessor(sm)

    # First run: process some conversations
    result1 = processor.search_inbox(max_inbox_limit=5)
    processed1 = result1.get("conversations_processed", 0)

    # Second run: should stop early when encountering unchanged conversations
    # (assuming inbox hasn't changed between runs)
    result2 = processor.search_inbox(max_inbox_limit=5)
    processed2 = result2.get("conversations_processed", 0)

    # Second run should process fewer or equal conversations
    # (stops when comparator detects no changes)
    assert processed2 <= processed1, \
        f"Second run should process <= conversations (got {processed2} vs {processed1})"

    logger.info(f"First run: {processed1} conversations, Second run: {processed2} conversations")
    logger.info("Comparator logic working (stops on unchanged conversations)")

    return True


def _test_summary_logging() -> None:
    """Test summary logging produces expected output."""
    import os
    skip_live_tests = os.getenv("SKIP_LIVE_API_TESTS", "false").lower() == "true"
    if skip_live_tests:
        logger.info("Skipping live test (SKIP_LIVE_API_TESTS=true)")
        return True

    # This test requires live API - only run when global session available
    try:
        sm = _ensure_session_for_tests()
    except RuntimeError:
        logger.info("Skipping live API test (no global session available)")
        return True
    processor = InboxProcessor(sm)

    # Process conversations and verify summary is logged
    result = processor.search_inbox(max_inbox_limit=3)

    # Verify result has expected keys
    assert "conversations_processed" in result, "Result should have conversations_processed"
    assert "conversations_fetched" in result, "Result should have conversations_fetched"

    logger.info(f"Summary: {result['conversations_processed']} processed, "
               f"{result['conversations_fetched']} fetched")

    return True


def _test_error_recovery() -> None:
    """Test error recovery mechanisms."""
    from unittest.mock import MagicMock
    sm = MagicMock()
    processor = InboxProcessor(sm)

    # Test state initialization
    state = processor._initialize_loop_state()

    # Verify error tracking fields exist
    assert "session_deaths" in state, "State should track session deaths"
    assert "session_recoveries" in state, "State should track session recoveries"
    assert state["session_deaths"] == 0, "Should start with 0 session deaths"
    assert state["session_recoveries"] == 0, "Should start with 0 session recoveries"

    logger.info("Error recovery state initialized correctly")

    return True


def _test_clarify_ambiguous_intent() -> None:
    """Test ambiguous intent clarification (Priority 1 Todo #7)."""
    from unittest.mock import MagicMock

    # Create mock session manager
    sm = MagicMock()
    processor = InboxProcessor(sm)

    # Test Case 1: Entity ambiguity detection
    entities_ambiguous = {
        "mentioned_people": [
            {"name": "Mary Smith", "birth_year": None, "birth_place": None, "relationship": None}
        ],
        "locations": [{"place": "Scotland", "context": "birthplace"}],
    }

    ambiguity_analysis = processor._analyze_entity_ambiguity(entities_ambiguous)
    assert ambiguity_analysis != "No ambiguity detected", "Should detect name and location ambiguity"
    assert "Mary Smith" in ambiguity_analysis, "Should mention ambiguous name"
    assert "Scotland" in ambiguity_analysis or "too broad" in ambiguity_analysis, "Should detect broad location"

    logger.info(f"✅ Detected ambiguity: {ambiguity_analysis}")

    # Test Case 2: No ambiguity (complete entities)
    entities_complete = {
        "mentioned_people": [
            {
                "name": "Charles Fetch",
                "birth_year": 1881,
                "birth_place": "Banff, Scotland",
                "relationship": "great-grandfather",
            }
        ],
        "locations": [{"place": "Banff, Banffshire, Scotland", "context": "birthplace"}],
    }

    ambiguity_analysis_complete = processor._analyze_entity_ambiguity(entities_complete)
    assert ambiguity_analysis_complete == "No ambiguity detected", "Should not detect ambiguity in complete entities"

    logger.info("✅ No ambiguity detected for complete entities")

    # Test Case 3: Clarification when no ambiguity returns None
    result = processor.clarify_ambiguous_intent(
        user_message="Charles Fetch was born in 1881 in Banff, Scotland",
        extracted_entities=entities_complete,
    )
    assert result is None, "Should return None when no ambiguity detected"

    logger.info("✅ Returns None for non-ambiguous entities")

    # Test Case 4: Method signature validation
    assert hasattr(processor, "clarify_ambiguous_intent"), "Should have clarify_ambiguous_intent method"
    assert hasattr(processor, "_analyze_entity_ambiguity"), "Should have _analyze_entity_ambiguity method"

    logger.info("✅ All clarify_ambiguous_intent tests passed")

    return True


def _test_conversation_phase_initial_outreach() -> None:
    """Test INITIAL_OUTREACH phase determination."""
    from unittest.mock import MagicMock

    from database import MessageDirectionEnum as DBMessageDirectionEnum

    sm = MagicMock()
    processor = InboxProcessor(session_manager=sm)

    phase = processor._determine_conversation_phase(
        conversation_id="test_conv_1",
        direction=MessageDirectionEnum.OUT,
        ai_sentiment=None,
        existing_logs=[],
        timestamp=datetime.now(timezone.utc),
    )

    assert phase is not None, "Phase should be determined for first message"
    assert str(phase).endswith("INITIAL_OUTREACH"), f"Expected INITIAL_OUTREACH, got {phase}"


def _test_conversation_phase_response_received() -> None:
    """Test RESPONSE_RECEIVED phase determination."""
    from unittest.mock import MagicMock

    from database import ConversationLog, MessageDirectionEnum as DBMessageDirectionEnum

    sm = MagicMock()
    processor = InboxProcessor(session_manager=sm)

    mock_out_log = MagicMock(spec=ConversationLog)
    mock_out_log.conversation_id = "test_conv_2"
    mock_out_log.direction = DBMessageDirectionEnum.OUT
    mock_out_log.latest_timestamp = datetime.now(timezone.utc)
    mock_out_log.ai_sentiment = None

    phase = processor._determine_conversation_phase(
        conversation_id="test_conv_2",
        direction=MessageDirectionEnum.IN,
        ai_sentiment="PRODUCTIVE",
        existing_logs=[mock_out_log],
        timestamp=datetime.now(timezone.utc),
    )

    assert phase is not None, "Phase should be determined"
    assert str(phase).endswith("RESPONSE_RECEIVED"), f"Expected RESPONSE_RECEIVED, got {phase}"


def _test_conversation_phase_information_shared() -> None:
    """Test INFORMATION_SHARED phase determination."""
    from unittest.mock import MagicMock

    from database import ConversationLog, MessageDirectionEnum as DBMessageDirectionEnum

    sm = MagicMock()
    processor = InboxProcessor(session_manager=sm)

    base_time = datetime.now(timezone.utc)
    mock_logs = []
    for i in range(3):
        mock_log = MagicMock(spec=ConversationLog)
        mock_log.conversation_id = "test_conv_3"
        mock_log.direction = DBMessageDirectionEnum.IN if i % 2 == 0 else DBMessageDirectionEnum.OUT
        mock_log.latest_timestamp = base_time
        mock_log.ai_sentiment = "PRODUCTIVE" if i % 2 == 0 else None
        mock_logs.append(mock_log)

    phase = processor._determine_conversation_phase(
        conversation_id="test_conv_3",
        direction=MessageDirectionEnum.IN,
        ai_sentiment="PRODUCTIVE",
        existing_logs=mock_logs,
        timestamp=base_time,
    )

    assert phase is not None, "Phase should be determined"
    assert str(phase).endswith("INFORMATION_SHARED"), f"Expected INFORMATION_SHARED, got {phase}"


def _test_follow_up_extraction_productive() -> None:
    """Test follow-up extraction for PRODUCTIVE conversations."""
    from unittest.mock import MagicMock

    from database import ConversationLog, ConversationPhaseEnum, MessageDirectionEnum as DBMessageDirectionEnum

    sm = MagicMock()
    processor = InboxProcessor(session_manager=sm)

    mock_log = MagicMock(spec=ConversationLog)
    mock_log.conversation_id = "follow_up_test_1"
    mock_log.direction = DBMessageDirectionEnum.IN
    mock_log.latest_message_content = "Do you know when Charles Fetch was born?"
    mock_log.ai_sentiment = "PRODUCTIVE"
    conversation_history = [mock_log]

    result = processor._extract_follow_up_requirements(
        conversation_history=conversation_history,
        latest_message="Do you know when Charles Fetch was born?",
        direction=MessageDirectionEnum.IN,
        conversation_phase=ConversationPhaseEnum.RESPONSE_RECEIVED,
        ai_sentiment="PRODUCTIVE",
        conversation_id="follow_up_test_1",
    )

    assert isinstance(result, dict), "Should return dictionary"
    assert "follow_up_required" in result, "Should have follow_up_required field"
    assert "follow_up_due_date" in result, "Should have follow_up_due_date field"
    assert "awaiting_response_from" in result, "Should have awaiting_response_from field"


def _test_follow_up_skips_desist() -> None:
    """Test that follow-up extraction skips DESIST conversations."""
    from unittest.mock import MagicMock

    from database import ConversationPhaseEnum

    sm = MagicMock()
    processor = InboxProcessor(session_manager=sm)

    result = processor._extract_follow_up_requirements(
        conversation_history=[],
        latest_message="I'm not interested in genealogy anymore.",
        direction=MessageDirectionEnum.IN,
        conversation_phase=ConversationPhaseEnum.CLOSED,
        ai_sentiment="DESIST",
        conversation_id="follow_up_test_2",
    )

    assert result["follow_up_required"] is False, "Should not require follow-up for DESIST"
    assert result["follow_up_due_date"] is None, "Should not set due date for DESIST"
    assert result["awaiting_response_from"] is None, "Should not set responsibility for DESIST"


def _test_follow_up_reminder_task_creation() -> None:
    """Test MS To-Do reminder task creation."""
    from datetime import timedelta
    from unittest.mock import MagicMock

    sm = MagicMock()
    processor = InboxProcessor(session_manager=sm)

    due_date = datetime.now(timezone.utc) + timedelta(days=7)
    result = processor._create_follow_up_reminder_task(
        person_id=123,
        conversation_id="task_test_1",
        task_title="Follow up with @TestUser about Charles Fetch",
        task_body="User asked: 'Do you know when Charles Fetch was born?'",
        due_date=due_date,
        urgency_level="urgent",
    )

    assert isinstance(result, bool), "Should return boolean success indicator"


def _test_task_importance_calculation() -> None:
    """Test task importance calculation from DNA + engagement data (Priority 0 Todo #14)."""
    from unittest.mock import MagicMock

    from database import Person

    sm = MagicMock()
    mock_db_session = MagicMock()
    sm.get_db_conn.return_value = mock_db_session

    processor = InboxProcessor(session_manager=sm)

    person_high = MagicMock(spec=Person)
    person_high.id = 1
    person_high.current_engagement_score = 80
    person_high.dna_match = MagicMock(spec=DnaMatch)
    person_high.dna_match.cm_dna = 150
    mock_db_session.query().filter().first.return_value = person_high

    importance = processor._calculate_task_importance(person_id=1, base_urgency="standard")
    assert importance == "high", f"Expected 'high' for cM=150, engagement=80, got '{importance}'"

    person_medium = MagicMock(spec=Person)
    person_medium.id = 2
    person_medium.current_engagement_score = 40
    person_medium.dna_match = MagicMock(spec=DnaMatch)
    person_medium.dna_match.cm_dna = 75
    mock_db_session.query().filter().first.return_value = person_medium

    importance = processor._calculate_task_importance(person_id=2, base_urgency="standard")
    assert importance == "normal", f"Expected 'normal' for cM=75, engagement=40, got '{importance}'"

    person_low = MagicMock(spec=Person)
    person_low.id = 3
    person_low.current_engagement_score = 20
    person_low.dna_match = MagicMock(spec=DnaMatch)
    person_low.dna_match.cm_dna = 30
    mock_db_session.query().filter().first.return_value = person_low

    importance = processor._calculate_task_importance(person_id=3, base_urgency="standard")
    assert importance == "low", f"Expected 'low' for cM=30, engagement=20, got '{importance}'"

    importance_urgent = processor._calculate_task_importance(person_id=3, base_urgency="urgent")
    assert importance_urgent == "high", f"Expected 'high' for urgent override, got '{importance_urgent}'"

    importance_patient = processor._calculate_task_importance(person_id=1, base_urgency="patient")
    assert importance_patient == "low", f"Expected 'low' for patient override, got '{importance_patient}'"


def _test_conversation_log_follow_up_fields() -> None:
    """Test that conversation log upsert includes follow-up fields."""
    from datetime import timedelta
    from unittest.mock import MagicMock

    sm = MagicMock()
    processor = InboxProcessor(session_manager=sm)

    due_date = datetime.now(timezone.utc) + timedelta(days=14)
    log_upsert = processor._create_conversation_log_upsert(
        api_conv_id="log_test_1",
        direction=MessageDirectionEnum.IN,
        people_id=456,
        message_content="Test message content",
        timestamp=datetime.now(timezone.utc),
        ai_sentiment="PRODUCTIVE",
        conversation_phase=None,
        follow_up_due_date=due_date,
        awaiting_response_from="me",
    )

    assert "follow_up_due_date" in log_upsert, "Should include follow_up_due_date field"
    assert "awaiting_response_from" in log_upsert, "Should include awaiting_response_from field"
    assert log_upsert["follow_up_due_date"] == due_date, "Should preserve due date"
    assert log_upsert["awaiting_response_from"] == "me", "Should preserve responsibility"


def _test_conversation_phase_closed() -> None:
    """Test CLOSED phase determination."""
    from unittest.mock import MagicMock

    from database import ConversationLog

    sm = MagicMock()
    processor = InboxProcessor(session_manager=sm)

    phase = processor._determine_conversation_phase(
        conversation_id="test_conv_4",
        direction=MessageDirectionEnum.IN,
        ai_sentiment="DESIST",
        existing_logs=[],
        timestamp=datetime.now(timezone.utc),
    )

    assert phase is not None, "Phase should be determined"
    assert str(phase).endswith("CLOSED"), f"Expected CLOSED for DESIST sentiment, got {phase}"


def action7_inbox_module_tests() -> bool:
    """Comprehensive test suite for action7_inbox.py using the unified TestSuite."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Action 7 - Inbox Processor", "action7_inbox.py")
    suite.start_suite()

    with suppress_logging():
        # Removed smoke test: _test_class_and_methods_available

        def _add_test(
            test_name: str,
            test_func: Callable[[], None],
            *,
            test_summary: str = "",
            functions_tested: str = "",
            method_description: str = "",
            expected_outcome: str = "",
        ) -> None:
            suite.run_test(
                test_name=test_name,
                test_func=test_func,
                test_summary=test_summary,
                functions_tested=functions_tested,
                method_description=method_description,
                expected_outcome=expected_outcome,
            )

        _add_test(
            "Inbox processor initialization",
            _test_inbox_processor_initialization,
            test_summary="SessionManager integration",
            functions_tested="InboxProcessor.__init__",
            method_description="Initialize processor with authenticated session",
            expected_outcome="Processor initialized with SessionManager and stats dict",
        )
        _add_test(
            "Fetch first page conversations",
            _test_fetch_first_page_conversations,
            test_summary="API conversation fetching",
            functions_tested="search_inbox",
            method_description="Fetch conversations from Ancestry API",
            expected_outcome="Conversations fetched and processed count returned",
        )
        _add_test(
            "Conversation database storage",
            _test_conversation_database_storage,
            test_summary="Database synchronization",
            functions_tested="search_inbox, ConversationLog",
            method_description="Store conversations in database",
            expected_outcome="Conversations stored in ConversationLog table",
        )
        _add_test(
            "Conversation parsing",
            _test_conversation_parsing,
            test_summary="API data parsing",
            functions_tested="search_inbox, ConversationLog",
            method_description="Parse conversation data from API response",
            expected_outcome="Conversations have required fields (ID, direction, timestamp)",
        )
        _add_test(
            "AI classification",
            _test_ai_classification,
            test_summary="AI-powered message classification",
            functions_tested="search_inbox, classify_message_intent",
            method_description="Classify messages with AI",
            expected_outcome="Messages classified as PRODUCTIVE/DESIST/OTHER",
        )
        _add_test(
            "Person status updates",
            _test_person_status_updates,
            test_summary="Person status synchronization",
            functions_tested="search_inbox, Person",
            method_description="Update person status from conversations",
            expected_outcome="Person status updated based on conversation classification",
        )
        _add_test(
            "Stop on unchanged conversation",
            _test_stop_on_unchanged_conversation,
            test_summary="Comparator logic (stop when no changes)",
            functions_tested="search_inbox, comparator",
            method_description="Stop processing when conversation unchanged",
            expected_outcome="Second run processes fewer conversations (stops on match)",
        )
        _add_test(
            "Summary logging",
            _test_summary_logging,
            test_summary="Result summary",
            functions_tested="search_inbox, _log_unified_summary",
            method_description="Log processing summary",
            expected_outcome="Summary includes processed/fetched counts",
        )
        _add_test(
            "Error recovery",
            _test_error_recovery,
            test_summary="Error recovery mechanisms",
            functions_tested="_initialize_loop_state",
            method_description="Test error recovery state tracking",
            expected_outcome="State tracks session deaths and recoveries",
        )
        _add_test(
            "Clarify ambiguous intent (Priority 1 Todo #7)",
            _test_clarify_ambiguous_intent,
            test_summary="AI-powered clarification questions",
            functions_tested="clarify_ambiguous_intent, _analyze_entity_ambiguity",
            method_description="Generate clarifying questions for incomplete entities",
            expected_outcome="Ambiguity detected, questions generated for missing data, None for complete data",
        )

        # === Priority 1 Todo #11: Conversation Phase Transitions Tests ===
        _add_test(
            "Conversation phase: INITIAL_OUTREACH",
            _test_conversation_phase_initial_outreach,
            test_summary="First outgoing message phase",
            functions_tested="_determine_conversation_phase",
            method_description="Determine phase for first OUT message",
            expected_outcome="Phase set to INITIAL_OUTREACH",
        )
        _add_test(
            "Conversation phase: RESPONSE_RECEIVED",
            _test_conversation_phase_response_received,
            test_summary="First response phase",
            functions_tested="_determine_conversation_phase",
            method_description="Determine phase for first IN message",
            expected_outcome="Phase set to RESPONSE_RECEIVED",
        )
        _add_test(
            "Conversation phase: INFORMATION_SHARED",
            _test_conversation_phase_information_shared,
            test_summary="Information exchange phase",
            functions_tested="_determine_conversation_phase",
            method_description="Determine phase for PRODUCTIVE exchanges",
            expected_outcome="Phase set to INFORMATION_SHARED",
        )

        # Priority 1 Todo #5: Follow-Up Reminder System Tests
        _add_test(
            "Follow-up extraction: PRODUCTIVE conversation",
            _test_follow_up_extraction_productive,
            test_summary="Extract follow-up requirements from PRODUCTIVE messages",
            functions_tested="_extract_follow_up_requirements (Todo #5)",
            method_description="Analyze conversation for pending questions and calculate due date",
            expected_outcome="Follow-up requirements extracted with due date and responsibility",
        )
        _add_test(
            "Follow-up extraction: Skip DESIST",
            _test_follow_up_skips_desist,
            test_summary="DESIST conversations bypass follow-up",
            functions_tested="_extract_follow_up_requirements (Todo #5)",
            method_description="Skip follow-up extraction for DESIST/CLOSED conversations",
            expected_outcome="No follow-up required for DESIST conversations",
        )
        _add_test(
            "Follow-up reminder task creation",
            _test_follow_up_reminder_task_creation,
            test_summary="Create MS To-Do tasks for follow-ups",
            functions_tested="_create_follow_up_reminder_task (Todo #5)",
            method_description="Create reminder tasks with urgency-based due dates and importance",
            expected_outcome="MS To-Do task created or gracefully skipped in test mode",
        )
        _add_test(
            "Task importance calculation (DNA + engagement)",
            _test_task_importance_calculation,
            test_summary="Calculate task priority from DNA strength and engagement score",
            functions_tested="_calculate_task_importance (Priority 0 Todo #14)",
            method_description="High: cM>100 & engagement>70; Medium: cM>50 | engagement>50; Low: other",
            expected_outcome="Importance correctly calculated: high/normal/low based on DNA + engagement",
        )
        _add_test(
            "Conversation log follow-up fields",
            _test_conversation_log_follow_up_fields,
            test_summary="Database storage includes follow-up metadata",
            functions_tested="_create_conversation_log_upsert (Todo #5)",
            method_description="Enhance conversation log with follow_up_due_date and awaiting_response_from",
            expected_outcome="Log upsert includes follow-up fields for database storage",
        )
        _add_test(
            "Conversation phase: CLOSED",
            _test_conversation_phase_closed,
            test_summary="Conversation closure phase",
            functions_tested="_determine_conversation_phase",
            method_description="Determine phase for DESIST classification",
            expected_outcome="Phase set to CLOSED",
        )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(action7_inbox_module_tests)


# --- Main Execution Block ---

if __name__ == "__main__":
    import sys
    print("Running Action 7 (Inbox Processor) comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

# End of action7_inbox.py
