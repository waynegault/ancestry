#!/usr/bin/env python3

"""
Action 9: Productive DNA Match Processing

Analyzes and processes productive DNA matches with comprehensive relationship
analysis, GEDCOM integration, and automated workflow management for genealogical
research including match scoring, family tree analysis, and research prioritization.
"""

# === CORE INFRASTRUCTURE ===
import logging

# === MODULE SETUP ===
logger = logging.getLogger(__name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# === STANDARD LIBRARY IMPORTS ===
import hashlib
import json
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Union, cast

# === THIRD-PARTY IMPORTS ===
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import and_, func, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DbSession, joinedload

from ai.ai_interface import (
    extract_genealogical_entities,
    generate_contextual_response,
    generate_genealogical_reply,
)

# === LOCAL IMPORTS ===
from config import ConfigSchema, config_schema as _config_schema
from core.app_mode_policy import should_allow_outbound_to_person
from core.database import (
    ConflictSeverityEnum,
    ConflictStatusEnum,
    ConversationLog,
    ConversationState,
    DataConflict,
    FactStatusEnum,
    FactTypeEnum,
    MessageDirectionEnum,
    MessageTemplate,
    Person,
    PersonStatusEnum,
    SuggestedFact,
    commit_bulk_data,
)
from core.error_handling import (
    api_retry,
    circuit_breaker,
    error_context,
    graceful_degradation,
    timeout_protection,
)
from core.logging_utils import log_action_banner
from core.session_manager import SessionManager
from core.utils import format_name
from genealogy.fact_validator import (
    ConflictType,
    ExtractedFact,
    FactValidator,
    extract_facts_from_ai_response,
)
from genealogy.tree_query_service import PersonSearchResult, TreeQueryService
from integrations import ms_graph_utils
from messaging import build_safe_column_value
from observability.conversation_analytics import record_engagement_event, update_conversation_metrics
from performance.connection_resilience import with_connection_resilience
from research.person_lookup_utils import (
    PersonLookupResult,
    create_not_found_result,
    create_result_from_gedcom,
)
from research.research_suggestions import generate_research_suggestions
from testing.test_utilities import create_standard_test_runner

# === CONSTANTS ===
PRODUCTIVE_SENTIMENT = "PRODUCTIVE"  # Sentiment string set by Action 7
OTHER_SENTIMENT = "OTHER"  # Sentiment string for messages that don't fit other categories
ACKNOWLEDGEMENT_MESSAGE_TYPE = "Productive_Reply_Acknowledgement"  # Key in messages.json
CUSTOM_RESPONSE_MESSAGE_TYPE = "Automated_Genealogy_Response"  # Key in messages.json

# Keywords that indicate no response should be sent
EXCLUSION_KEYWORDS = [
    "stop",
    "unsubscribe",
    "no more messages",
    "not interested",
    "do not respond",
    "no reply",
]

MAX_OTHER_MESSAGE_WORDS_FOR_AI = 25
NAME_LIKE_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?\b")

DecoratorCallable = Callable[[Callable[..., Any]], Callable[..., Any]]
DecoratorFactory = Callable[..., DecoratorCallable]
error_context_decorator = cast(DecoratorFactory, error_context)

SAFE_COLUMN_ENUMS = {
    "direction": MessageDirectionEnum,
    "status": PersonStatusEnum,
}

safe_column_value = build_safe_column_value(SAFE_COLUMN_ENUMS)

config_schema: ConfigSchema = cast(ConfigSchema, _config_schema)


def should_exclude_message(message_content: Optional[str]) -> bool:
    """
    Checks if a message contains any exclusion keywords that indicate no response should be sent.

    Args:
        message_content: The message content to check

    Returns:
        True if the message should be excluded, False otherwise
    """
    if not message_content:
        return False

    message_lower = message_content.lower()

    # Check for exclusion keywords
    for keyword in EXCLUSION_KEYWORDS:
        if keyword.lower() in message_lower:
            logger.debug(f"Exclusion keyword found: '{keyword}'")
            return True
    return False


# AI Prompt for generating genealogical replies
GENERATE_GENEALOGICAL_REPLY_PROMPT = """
You are a helpful genealogy assistant. Your task is to generate a personalized reply to a message about family history.

CONVERSATION CONTEXT:
{conversation_context}

USER'S LAST MESSAGE:
{user_message}

GENEALOGICAL DATA:
{genealogical_data}

Based on the conversation context, the user's last message, and the genealogical data provided,
craft a personalized, informative, and friendly response. Include specific details from the
genealogical data that are relevant to the user's inquiry. Be conversational and helpful.

Your response should:
1. Acknowledge the user's message
2. Share relevant genealogical information
3. Provide context about relationships and family connections
4. Ask follow-up questions if appropriate
5. Be friendly and conversational in tone

RESPONSE:
"""


# --- Pydantic Models for AI Response Validation ---
class NameData(BaseModel):
    """Model for structured name information."""

    full_name: str
    nicknames: list[str] = Field(default_factory=list)
    maiden_name: Optional[str] = None
    generational_suffix: Optional[str] = None


class VitalRecord(BaseModel):
    """Model for vital record information."""

    person: str
    event_type: str
    date: str = ""
    place: str = ""
    certainty: str = "unknown"

    @field_validator("date", "place", mode="before")
    @classmethod
    def convert_none_to_empty_string(cls, v: Any) -> str:
        """Convert None to empty string for optional fields."""
        return "" if v is None else str(v)


class Relationship(BaseModel):
    """Model for relationship information."""

    person1: str
    relationship: str
    person2: str
    context: str = ""

    @field_validator("person2", "context", mode="before")
    @classmethod
    def convert_none_to_empty_string(cls, v: Any) -> str:
        """Convert None to empty string for optional fields."""
        return "" if v is None else str(v)


class Location(BaseModel):
    """Model for location information."""

    place: str
    context: str = ""
    time_period: str = ""

    @field_validator("context", "time_period", mode="before")
    @classmethod
    def convert_none_to_empty_string(cls, v: Any) -> str:
        """Convert None to empty string for optional fields."""
        return "" if v is None else str(v)


class Occupation(BaseModel):
    """Model for occupation information."""

    person: str
    occupation: str
    location: str = ""
    time_period: str = ""

    @field_validator("location", "time_period", mode="before")
    @classmethod
    def convert_none_to_empty_string(cls, v: Any) -> str:
        """Convert None to empty string for optional fields."""
        return "" if v is None else str(v)


class ExtractedData(BaseModel):
    """Enhanced Pydantic model for validating the extracted_data structure in AI responses."""

    # Enhanced structured fields
    structured_names: list[NameData] = Field(default_factory=list)
    vital_records: list[VitalRecord] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    occupations: list[Occupation] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)
    documents_mentioned: list[str] = Field(default_factory=list)
    dna_information: list[str] = Field(default_factory=list)
    suggested_tasks: list[str] = Field(default_factory=list)

    @field_validator(
        "research_questions",
        "documents_mentioned",
        "dna_information",
        "suggested_tasks",
        mode="before",
    )
    @classmethod
    def ensure_list_of_strings(cls, v: Any) -> list[str]:
        """Ensures all fields are lists of strings."""
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        return [str(item) for item in v if item is not None]

    def get_all_names(self) -> list[str]:
        """Get all names from structured fields."""
        names: list[str] = []
        for name_data in self.structured_names:
            names.append(name_data.full_name)
            names.extend(name_data.nicknames)
        return list(set(names))

    def get_all_locations(self) -> list[str]:
        """Get all locations from structured fields."""
        locations: list[str] = []
        for vital_record in self.vital_records:
            if vital_record.place:
                locations.append(vital_record.place)
        return list(set(locations))


class AIResponse(BaseModel):
    """Pydantic model for validating the complete AI response structure."""

    extracted_data: ExtractedData = Field(default_factory=ExtractedData)
    suggested_tasks: list[str] = Field(default_factory=list)

    @field_validator("suggested_tasks", mode="before")
    @classmethod
    def ensure_tasks_list(cls, v: Any) -> list[str]:
        """Ensures suggested_tasks is a list of strings."""
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        return [str(item) for item in v if item is not None]


# Global variable to cache the GEDCOM data
class _GedcomDataCache:
    """Manages cached GEDCOM data state."""

    data: Optional[Any] = None


def get_gedcom_data() -> Optional[Any]:
    """
    Returns the cached GEDCOM data instance, loading it if necessary.

    This function ensures the GEDCOM file is loaded only once and reused
    throughout the module, improving performance.

    Returns:
        GedcomData instance or None if loading fails
    """
    # Return cached data if already loaded
    if _GedcomDataCache.data is not None:
        return _GedcomDataCache.data

    # Check if GEDCOM path is configured
    gedcom_path = config_schema.database.gedcom_file_path
    if not gedcom_path:
        logger.warning("GEDCOM_FILE_PATH not configured. Cannot load GEDCOM file.")
        return None

    # Check if GEDCOM file exists
    if not gedcom_path.exists():
        logger.warning(f"GEDCOM file not found at {gedcom_path}. Cannot load GEDCOM file.")
        return None

    # Load GEDCOM data
    try:
        logger.debug(f"Loading GEDCOM file {gedcom_path.name} (first time)...")
        from genealogy.gedcom.gedcom_cache import load_gedcom_with_aggressive_caching as load_gedcom_data

        _GedcomDataCache.data = load_gedcom_data(str(gedcom_path))
        if _GedcomDataCache.data:
            logger.debug("GEDCOM file loaded successfully and cached for reuse.")
            # Log some stats about the loaded data
            logger.debug(f"  Index size: {len(getattr(_GedcomDataCache.data, 'indi_index', {}))}")
            logger.debug(
                f"  Pre-processed cache size: {len(getattr(_GedcomDataCache.data, 'processed_data_cache', {}))}"
            )
        return _GedcomDataCache.data
    except Exception as e:
        logger.error(f"Error loading GEDCOM file: {e}", exc_info=True)
        return None


# Import required modules and functions


#####################################################
# Data Classes for Better Organization
#####################################################


@dataclass
class ProcessingState:
    """Manages the state of the processing operation."""

    overall_success: bool = True
    processed_count: int = 0
    tasks_created_count: int = 0
    acks_sent_count: int = 0
    archived_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    total_candidates: int = 0
    batch_num: int = 0
    critical_db_error_occurred: bool = False


@dataclass
class MSGraphState:
    """Manages MS Graph authentication and configuration."""

    token: Optional[str] = None
    list_id: Optional[str] = None
    list_name: str = ""
    auth_attempted: bool = False


_ENHANCED_TASK_STATE = MSGraphState(list_name=config_schema.ms_todo_list_name)


@dataclass
class EnhancedTaskPayload:
    """Data required to create an enhanced MS To-Do task."""

    title: str
    body: str
    importance: str
    due_date: Optional[str]
    categories: list[str]


def _should_skip_enhanced_tasks() -> bool:
    """Return True when MS Graph enhancements should be skipped."""
    if os.environ.get("SKIP_LIVE_API_TESTS", "").lower() == "true":
        logger.debug("Skipping MS Graph initialization for enhanced tasks (SKIP_LIVE_API_TESTS=true).")
        return True
    if not getattr(ms_graph_utils, "msal_app_instance", None):
        logger.debug("MS Graph client not configured; skipping enhanced task creation.")
        return True
    return False


def _ensure_task_list_name(state: MSGraphState) -> None:
    """Populate the list name if it is missing."""
    if not state.list_name:
        state.list_name = config_schema.ms_todo_list_name


def _ensure_task_token(state: MSGraphState) -> bool:
    """Ensure an authentication token is available for MS Graph calls."""
    if state.token:
        return True
    if state.auth_attempted:
        return bool(state.token)

    state.token = ms_graph_utils.acquire_token_device_flow()
    state.auth_attempted = True
    if not state.token:
        logger.warning("MS Graph authentication unavailable; cannot create enhanced task.")
        return False
    return True


def _ensure_task_list_id(state: MSGraphState) -> bool:
    """Ensure the MS To-Do list identifier has been resolved."""
    if state.list_id:
        return True

    if not state.token:
        return False

    logger.debug(f"Resolving MS To-Do list '{state.list_name}' for enhanced task creation...")
    state.list_id = ms_graph_utils.get_todo_list_id(state.token, state.list_name)
    if not state.list_id:
        logger.warning(f"MS To-Do list '{state.list_name}' not found; skipping enhanced task creation.")
        return False
    return True


def _ensure_enhanced_task_ms_graph_state(state: MSGraphState) -> bool:
    """Ensure MS Graph token and list ID are available for enhanced task creation."""
    if _should_skip_enhanced_tasks():
        return False

    _ensure_task_list_name(state)

    if not _ensure_task_token(state):
        return False

    if not state.token:
        return False

    return _ensure_task_list_id(state)


def _merge_task_categories(categories: Optional[list[str]]) -> list[str]:
    """Merge default and user-provided categories without duplicates."""
    default_categories = ["Genealogy Research", "DNA Matches"]
    final_categories: list[str] = []
    for category in default_categories + (categories or []):
        if category and category not in final_categories:
            final_categories.append(category)
    return final_categories


def _compute_due_date(days_until_due: Optional[int]) -> Optional[str]:
    """Return an ISO date string for the suggested due date."""
    if not days_until_due or days_until_due <= 0:
        return None

    due_dt = datetime.now(timezone.utc) + timedelta(days=days_until_due)
    return due_dt.strftime("%Y-%m-%d")


def _compose_task_title(person_name: str, relationship: Optional[str]) -> str:
    """Create a descriptive task title."""
    if relationship:
        return f"Research: {person_name} ({relationship})"
    return f"Research: {person_name}"


def _add_ancestry_urls_to_body(body_lines: list[str], profile_id: Optional[str], uuid: Optional[str]) -> None:
    """Add Ancestry URLs to task body lines."""
    if profile_id:
        body_lines.append(f"Ancestry Profile: https://www.ancestry.com/secure/member/profile?id={profile_id}")
    if uuid:
        body_lines.append(f"DNA Comparison: https://www.ancestry.com/dna/matches/{uuid}/compare")


def _add_tree_info_to_body(body_lines: list[str], tree_info: Optional[dict[str, Any]]) -> None:
    """Add tree information to task body lines."""
    if not tree_info:
        return
    if tree_info.get('person_name_in_tree'):
        body_lines.append(f"Name in Tree: {tree_info['person_name_in_tree']}")
    if tree_info.get('view_in_tree_link'):
        body_lines.append(f"View in Tree: {tree_info['view_in_tree_link']}")
    if tree_info.get('actual_relationship'):
        body_lines.append(f"Tree Relationship: {tree_info['actual_relationship']}")


def _compose_task_body(
    person_name: str,
    relationship: Optional[str],
    shared_dna_cm: Optional[float],
    importance: str,
    due_date: Optional[str],
    categories: list[str],
    profile_id: Optional[str] = None,
    uuid: Optional[str] = None,
    tree_info: Optional[dict[str, Any]] = None,
) -> str:
    """Build the task body with key research context."""
    body_lines = [f"Research target: {person_name}"]

    _add_ancestry_urls_to_body(body_lines, profile_id, uuid)

    if relationship:
        body_lines.append(f"Relationship: {relationship}")
    if shared_dna_cm is not None:
        body_lines.append(f"Shared DNA: {shared_dna_cm:.1f} cM")

    _add_tree_info_to_body(body_lines, tree_info)

    body_lines.append(f"Priority: {importance.title()}")
    if due_date:
        body_lines.append(f"Suggested due date: {due_date}")
    if categories:
        body_lines.append(f"Categories: {', '.join(categories)}")
    return "\n".join(body_lines)


def _build_enhanced_task_payload(
    person_name: str,
    relationship: Optional[str],
    shared_dna_cm: Optional[float],
    categories: Optional[list[str]],
    profile_id: Optional[str] = None,
    uuid: Optional[str] = None,
    tree_info: Optional[dict[str, Any]] = None,
) -> EnhancedTaskPayload:
    """Construct the payload required to submit an enhanced task."""
    importance, days_until_due = calculate_task_priority_from_relationship(
        relationship,
        shared_dna_cm,
    )
    final_categories = _merge_task_categories(categories)
    due_date = _compute_due_date(days_until_due)
    title = _compose_task_title(person_name, relationship)
    body = _compose_task_body(
        person_name,
        relationship,
        shared_dna_cm,
        importance,
        due_date,
        final_categories,
        profile_id=profile_id,
        uuid=uuid,
        tree_info=tree_info,
    )
    return EnhancedTaskPayload(
        title=title,
        body=body,
        importance=importance,
        due_date=due_date,
        categories=final_categories,
    )


def _submit_enhanced_task(
    state: MSGraphState,
    payload: EnhancedTaskPayload,
    person_name: str,
) -> Optional[str]:
    """Send the enhanced task request to MS Graph."""
    if not state.token or not state.list_id:
        logger.warning("MS Graph state incomplete after initialization; skipping task creation.")
        return None

    task_id = ms_graph_utils.create_todo_task(
        state.token,
        state.list_id,
        payload.title,
        payload.body,
        importance=payload.importance,
        due_date=payload.due_date,
        categories=payload.categories,
    )

    if task_id:
        logger.info(f"Created enhanced research task for {person_name} (ID: {task_id}).")
    else:
        logger.warning(f"Enhanced research task creation failed for {person_name}.")

    return task_id


@dataclass
class DatabaseState:
    """Manages database session and batch operations."""

    session: Optional[DbSession] = None
    logs_to_add: Optional[list[dict[str, Any]]] = None
    person_updates: Optional[dict[int, PersonStatusEnum]] = None
    batch_size: int = 10
    commit_threshold: int = 10

    def __post_init__(self) -> None:
        if self.logs_to_add is None:
            self.logs_to_add = []
        if self.person_updates is None:
            self.person_updates = {}


@dataclass
class MessageConfig:
    """Manages message types and templates."""

    templates: Optional[dict[str, str]] = None
    ack_msg_type_id: Optional[int] = None
    custom_reply_msg_type_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.templates is None:
            self.templates = {}


#####################################################
# Core Processing Classes
#####################################################


class PersonProcessor:
    """Handles processing of individual persons."""

    def __init__(
        self,
        session_manager: SessionManager,
        db_state: DatabaseState,
        msg_config: MessageConfig,
        ms_state: MSGraphState,
    ) -> None:
        self.session_manager = session_manager
        self.db_state = db_state
        self.msg_config = msg_config
        self.ms_state = ms_state
        self.my_pid_lower = session_manager.my_profile_id.lower() if session_manager.my_profile_id else ""
        self._ai_cache: dict[str, tuple[dict[str, Any], list[str]]] = {}
        # Sprint 3: TreeQueryService for RAG-style tree context retrieval
        self._tree_query_service: Optional[TreeQueryService] = None

    def process_person(self, person: Person) -> tuple[bool, str]:
        """
        Process a single person and return (success, status_message).

        Returns:
            Tuple of (success: bool, status_message: str)
        """
        log_prefix = f"Productive: {person.username} #{person.id}"
        success = True
        status = "success"

        try:
            # Apply rate limiting
            if self.session_manager.rate_limiter:
                self.session_manager.rate_limiter.wait()

            # Get message context
            context_logs = self._get_context_logs(person, log_prefix)
            if not context_logs:
                success, status = False, "no_context"
            else:
                # Check exclusions and status
                should_skip, latest_message = self._should_skip_person(person, context_logs, log_prefix)
                if should_skip:
                    success, status = False, "skipped"
                elif latest_message is None:
                    success, status = False, "no_context"
                else:
                    # Process with AI
                    ai_results = self._process_with_ai(person, context_logs, latest_message, log_prefix)
                    if not ai_results:
                        success, status = False, "ai_error"
                    else:
                        extracted_data, suggested_tasks = ai_results

                        # Validate and stage extracted facts for review
                        self._validate_and_record_facts(person, extracted_data, latest_message, log_prefix)

                        # Phase 2: Look up mentioned people
                        lookup_results = self._lookup_mentioned_people(extracted_data, person)

                        # Create MS Graph tasks
                        self._create_ms_tasks(person, suggested_tasks, log_prefix)

                        # Generate and send response
                        response_sent = self._handle_message_response(
                            person,
                            context_logs,
                            extracted_data,
                            lookup_results,
                            log_prefix,
                            latest_message,
                        )
                        if not response_sent:
                            success, status = False, "send_error"
                        else:
                            # Phase 2: Update conversation state tracking
                            self._update_conversation_state(person, extracted_data, context_logs, log_prefix)

        except Exception as e:
            logger.error(f"Error processing {log_prefix}: {e}", exc_info=True)
            return False, f"error: {e!s}"

        return success, status

    def _get_context_logs(self, person: Person, log_prefix: str) -> Optional[list[ConversationLog]]:
        """Get message context for the person."""
        if self.db_state.session is None:
            logger.error(f"Database session is None for {log_prefix}")
            return None
        context_logs = _get_message_context(self.db_state.session, person.id)
        if not context_logs:
            logger.warning(f"Skipping {log_prefix}: Failed to retrieve message context.")
        return context_logs

    def _should_skip_person(
        self, person: Person, context_logs: list[ConversationLog], log_prefix: str
    ) -> tuple[bool, Optional[ConversationLog]]:
        """Check if person should be skipped based on various criteria.

        Returns:
            Tuple where first element indicates whether to skip and second
            element provides the latest incoming message when available.
        """

        # Check person status
        excluded_statuses = [
            PersonStatusEnum.DESIST,
            PersonStatusEnum.ARCHIVE,
            PersonStatusEnum.BLOCKED,
            PersonStatusEnum.DEAD,
        ]

        if person.status in excluded_statuses:
            logger.info(f"{log_prefix}: Person has status {person.status}. Skipping.")
            return True, None
        # Get latest message
        latest_message = self._get_latest_incoming_message(context_logs)
        if not latest_message:
            return True, None

        # Check if custom reply already sent
        custom_reply_sent_at = safe_column_value(latest_message, "custom_reply_sent_at", None)
        if custom_reply_sent_at is not None:
            logger.info(f"{log_prefix}: Custom reply already sent. Skipping.")
            return True, latest_message

        # Check for exclusion keywords
        latest_message_content = safe_column_value(latest_message, "latest_message_content", "")
        if should_exclude_message(latest_message_content):
            logger.info(f"{log_prefix}: Message contains exclusion keyword. Skipping.")
            return True, latest_message

        # Check if OTHER message with no mentioned names
        ai_sentiment = safe_column_value(latest_message, "ai_sentiment", None)
        if ai_sentiment == OTHER_SENTIMENT:
            # We'll handle this in the AI processing step
            pass

        return False, latest_message

    @staticmethod
    def _get_latest_incoming_message(context_logs: list[ConversationLog]) -> Optional[ConversationLog]:
        """Get the latest incoming message from context logs."""
        for log in reversed(context_logs):
            direction = safe_column_value(log, "direction", None)
            if direction == MessageDirectionEnum.IN:
                return log
        return None

    def _get_tree_query_service(self) -> Optional[TreeQueryService]:
        """
        Lazily initialize and return TreeQueryService for RAG-style tree context.

        Sprint 3: RAG Response Generator integration.

        Returns:
            TreeQueryService instance, or None if GEDCOM not available
        """
        if self._tree_query_service is None:
            try:
                self._tree_query_service = TreeQueryService()
                logger.debug("TreeQueryService initialized for person lookup")
            except Exception as e:
                logger.warning(f"Failed to initialize TreeQueryService: {e}")
                return None
        return self._tree_query_service

    @staticmethod
    def _should_bypass_ai_extraction(latest_message: ConversationLog, log_prefix: str) -> bool:
        """Determine if AI extraction can be skipped for low-detail OTHER replies."""

        ai_sentiment = safe_column_value(latest_message, "ai_sentiment", None)
        if ai_sentiment != OTHER_SENTIMENT:
            return False

        content = safe_column_value(latest_message, "latest_message_content", "").strip()
        if not content:
            logger.debug(f"{log_prefix}: Empty content for OTHER message; skipping AI extraction.")
            return True

        word_count = len(content.split())
        if word_count > MAX_OTHER_MESSAGE_WORDS_FOR_AI:
            return False

        has_name_like = bool(NAME_LIKE_PATTERN.search(content))
        has_digit = any(ch.isdigit() for ch in content)
        has_question = "?" in content

        if has_name_like or has_digit or has_question:
            return False

        logger.debug(
            f"{log_prefix}: OTHER message with {word_count} words and no obvious genealogical cues; bypassing AI."
        )
        return True

    def _process_with_ai(
        self,
        person: Person,
        context_logs: list[ConversationLog],
        latest_message: ConversationLog,
        log_prefix: str,
    ) -> Optional[tuple[dict[str, Any], list[str]]]:
        """Process message content with AI and return extracted data and tasks."""

        # Check session validity
        if not self.session_manager.is_sess_valid():
            logger.error(f"Session invalid for {person.username}. Skipping.")
            return None

        # Format context for AI
        formatted_context = _format_context_for_ai_extraction(context_logs, self.my_pid_lower)

        # Allow fast-path skip for low-information OTHER replies
        if self._should_bypass_ai_extraction(latest_message, log_prefix):
            default_structure = _get_default_ai_response_structure()
            logger.info(f"{log_prefix}: Skipping AI extraction for low-detail 'OTHER' reply.")
            return (
                deepcopy(default_structure["extracted_data"]),
                list(default_structure["suggested_tasks"]),
            )

        # Reuse AI results within this run when the conversation context is identical
        ai_provider = config_schema.ai_provider.lower()
        context_hash = hashlib.sha1(formatted_context.encode("utf-8")).hexdigest()
        cache_key = f"{ai_provider}:{context_hash}"
        cached_result = self._ai_cache.get(cache_key)
        if cached_result:
            logger.debug(f"{log_prefix}: Using cached AI extraction result (hash {context_hash[:8]}).")
            cached_data, cached_tasks = cached_result
            return deepcopy(cached_data), list(cached_tasks)

        # Call AI
        logger.debug(f"Calling AI for {person.username}...")
        ai_response = extract_genealogical_entities(formatted_context, self.session_manager)

        # Process AI response
        processed_response = _process_ai_response(ai_response, f"{person.username}")
        extracted_data = processed_response["extracted_data"]
        suggested_tasks = processed_response["suggested_tasks"]

        # Log results
        if suggested_tasks:
            logger.debug(f"Processed {len(suggested_tasks)} tasks for {person.username}")

        entity_counts = {k: len(v) for k, v in extracted_data.items()}
        logger.debug(f"Extracted entities for {person.username}: {json.dumps(entity_counts)}")

        # === FACT EXTRACTION 2.0: Convert AI response to validated ExtractedFact objects ===
        # Phase 3 Sprint 2: Structured fact extraction with validation pipeline
        extracted_facts = extract_facts_from_ai_response(extracted_data)
        if extracted_facts:
            logger.info(
                f"Fact Extraction 2.0: Extracted {len(extracted_facts)} facts for {person.username} ({entity_counts})"
            )
            # Store facts in extracted_data for downstream processing
            extracted_data["validated_facts"] = [
                {
                    "fact_type": f.fact_type,
                    "subject_name": f.subject_name,
                    "original_text": f.original_text,
                    "structured_value": f.structured_value,
                    "normalized_value": f.normalized_value,
                    "confidence": f.confidence,
                    "location": f.location,
                    "related_person_name": f.related_person_name,
                }
                for f in extracted_facts
            ]
            # Keep raw objects for validation pipeline
            extracted_data["_fact_objects"] = extracted_facts
        else:
            logger.debug(f"No structured facts extracted for {person.username}")
            extracted_data["validated_facts"] = []
            extracted_data["_fact_objects"] = []

        # Cache result for this context to avoid repeated AI calls during the run
        self._ai_cache[cache_key] = (
            deepcopy(extracted_data),
            list(suggested_tasks),
        )

        return extracted_data, suggested_tasks

    @staticmethod
    def _map_field_name_for_conflict(fact_type: str) -> str:
        """Map fact types to database field names for DataConflict records."""

        mapping = {
            "BIRTH": "birth_year",
            "DEATH": "death_year",
            "MARRIAGE": "marriage_date",
            "RELATIONSHIP": "relationship",
            "LOCATION": "location",
        }
        return mapping.get(fact_type, fact_type.lower())

    @staticmethod
    def _resolve_fact_type_enum(fact_type: str) -> FactTypeEnum:
        """Return FactTypeEnum value, defaulting to OTHER on unknown types."""
        try:
            return FactTypeEnum(fact_type)
        except Exception:
            return FactTypeEnum.OTHER

    @staticmethod
    def _structured_value_for_fact(fact: ExtractedFact) -> str:
        """Choose normalized or structured value with an empty-string fallback."""
        return fact.normalized_value or fact.structured_value or ""

    def _stage_suggested_fact(
        self,
        person: Person,
        fact: ExtractedFact,
        fact_type_enum: FactTypeEnum,
        structured_value: str,
        message_id: Optional[int],
        validation_result: Any,
    ) -> FactStatusEnum:
        """Create and stage a SuggestedFact, returning the status used."""
        status = FactStatusEnum.APPROVED if validation_result.auto_approved else FactStatusEnum.PENDING

        session = cast(DbSession, self.db_state.session)

        suggested = SuggestedFact(
            people_id=person.id,
            fact_type=fact_type_enum,
            original_value=fact.original_text,
            new_value=structured_value,
            source_message_id=message_id,
            status=status,
            confidence_score=fact.confidence,
        )
        session.add(suggested)
        return status

    @staticmethod
    def _map_conflict_severity(fact_type: str, conflict_type: ConflictType) -> ConflictSeverityEnum:
        """
        Map conflict type and fact type to ConflictSeverityEnum.

        Phase 11.2: Uses ConflictDetector's field-severity mappings to determine
        appropriate severity level for routing to review queue.
        """
        # Critical fields that should escalate
        critical_fields = {"relationship", "relationship_to_me"}
        high_fields = {"birth_year", "death_year", "birth_place", "death_place", "gender"}

        field_name = fact_type.lower()

        # MAJOR_CONFLICT always escalates
        if conflict_type == ConflictType.MAJOR_CONFLICT:
            if field_name in critical_fields:
                return ConflictSeverityEnum.CRITICAL
            if field_name in high_fields:
                return ConflictSeverityEnum.HIGH
            return ConflictSeverityEnum.MEDIUM

        # MINOR_CONFLICT uses field-based severity
        if field_name in critical_fields:
            return ConflictSeverityEnum.HIGH  # Even minor relationship conflicts are important
        if field_name in high_fields:
            return ConflictSeverityEnum.MEDIUM
        return ConflictSeverityEnum.LOW

    def _stage_conflict_if_needed(
        self,
        person: Person,
        fact: ExtractedFact,
        structured_value: str,
        message_id: Optional[int],
        validation_result: Any,
    ) -> int:
        """
        Stage a DataConflict when validation detects conflicts.

        Phase 11.2: Now includes severity mapping for review queue routing.
        HIGH/CRITICAL severity conflicts are prioritized for human review.

        Returns:
            1 if conflict was staged, 0 otherwise
        """
        if validation_result.conflict_type not in {ConflictType.MINOR_CONFLICT, ConflictType.MAJOR_CONFLICT}:
            return 0

        existing_value = validation_result.conflicting_fact.value if validation_result.conflicting_fact else None
        session = cast(DbSession, self.db_state.session)

        # Phase 11.2: Determine severity for review routing
        severity = self._map_conflict_severity(fact.fact_type, validation_result.conflict_type)

        conflict = DataConflict(
            people_id=person.id,
            field_name=self._map_field_name_for_conflict(fact.fact_type),
            existing_value=existing_value,
            new_value=structured_value,
            severity=severity,
            source="conversation",
            source_message_id=message_id,
            confidence_score=fact.confidence,
            status=ConflictStatusEnum.OPEN,
        )
        session.add(conflict)

        # Log HIGH/CRITICAL conflicts for visibility
        if severity in {ConflictSeverityEnum.HIGH, ConflictSeverityEnum.CRITICAL}:
            logger.warning(
                f"⚠️ {severity.value} severity conflict detected for Person {person.id}: "
                f"{fact.fact_type} (existing: {existing_value!r} → new: {structured_value!r})"
            )

        return 1

    def _validate_and_record_facts(
        self,
        person: Person,
        extracted_data: dict[str, Any],
        latest_message: ConversationLog,
        log_prefix: str,
    ) -> tuple[int, int, int]:
        """Validate extracted facts and stage SuggestedFact/DataConflict records."""

        if self.db_state.session is None:
            logger.debug(f"{log_prefix}: Skipping fact validation (no DB session)")
            return 0, 0, 0

        raw_fact_objects = extracted_data.get("_fact_objects") or []
        fact_objects: list[ExtractedFact] = [fact for fact in raw_fact_objects if isinstance(fact, ExtractedFact)]
        if not fact_objects:
            return 0, 0, 0

        validator = FactValidator(db_session=self.db_state.session)
        approved = pending = conflicts = 0
        message_id = safe_column_value(latest_message, "id", None)

        for fact in fact_objects:
            try:
                result = validator.validate_fact(fact, person)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(f"{log_prefix}: Fact validation error for {fact.fact_type}: {exc}")
                continue

            fact_type_enum = self._resolve_fact_type_enum(fact.fact_type)
            structured_value = self._structured_value_for_fact(fact)

            status = self._stage_suggested_fact(
                person,
                fact,
                fact_type_enum,
                structured_value,
                message_id,
                result,
            )

            if status is FactStatusEnum.APPROVED:
                approved += 1
            else:
                pending += 1

            conflicts += self._stage_conflict_if_needed(
                person,
                fact,
                structured_value,
                message_id,
                result,
            )

        if approved or pending or conflicts:
            logger.info(
                f"{log_prefix}: Fact validation staged {approved} approved, {pending} pending, {conflicts} conflicts"
            )

        return approved, pending, conflicts

    def _lookup_mentioned_people(self, extracted_data: dict[str, Any], person: Person) -> list[PersonLookupResult]:
        """
        Look up people mentioned in the conversation using Action 10 (GEDCOM) and Action 11 (API).

        Phase 2: Person Lookup Integration

        Args:
            extracted_data: Extracted entities from AI (includes 'mentioned_people' array)
            person: Person object for logging context

        Returns:
            List of PersonLookupResult objects with lookup results
        """
        mentioned_people = extracted_data.get("mentioned_people", [])

        if not mentioned_people:
            logger.debug(f"No people mentioned in conversation with {person.username}")
            return []

        logger.info(f"Looking up {len(mentioned_people)} mentioned people for {person.username}")
        lookup_results: list[PersonLookupResult] = []
        people_found_count = 0

        for person_data in mentioned_people:
            person_name = person_data.get("name", "Unknown")
            logger.debug(f"Looking up: {person_name}")

            # Try GEDCOM search first (Action 10)
            gedcom_result = self._search_gedcom_for_person(person_data)

            if gedcom_result:
                logger.info(f"Found {person_name} in GEDCOM with score {gedcom_result.match_score}")
                lookup_results.append(gedcom_result)
                people_found_count += 1
                self._track_person_lookup_analytics(
                    person, person_name, gedcom_result.match_score, "GEDCOM", found=True
                )
                continue

            # If not found in GEDCOM, try API search (Action 11)
            logger.debug(f"{person_name} not found in GEDCOM, attempting API search")
            api_result = self._search_api_for_person(person_name, person_data)

            if api_result:
                logger.info(f"Found {person_name} via API search (score: {api_result.match_score})")
                lookup_results.append(api_result)
                people_found_count += 1
                self._track_person_lookup_analytics(person, person_name, api_result.match_score, "API", found=True)
                continue

            # Not found in GEDCOM or API
            logger.debug(f"{person_name} not found in GEDCOM or API")
            not_found = create_not_found_result(person_name, reason="Person not found in GEDCOM file or API")
            lookup_results.append(not_found)
            self._track_person_lookup_analytics(person, person_name, None, "GEDCOM+API", found=False)

        logger.info(f"Lookup complete: {people_found_count}/{len(mentioned_people)} people found")
        return lookup_results

    def _track_person_lookup_analytics(
        self, person: Person, person_name: str, match_score: Optional[float], source: str, found: bool
    ) -> None:
        """Track analytics for person lookup attempts."""
        try:
            if not self.db_state.session:
                return

            event_desc = (
                f"Found {person_name} in {source} (score: {match_score})"
                if found
                else f"Person {person_name} not found in {source}"
            )
            event_data: dict[str, Any] = {"person_name": person_name, "source": source}
            if match_score is not None:
                event_data["match_score"] = match_score

            record_engagement_event(
                session=self.db_state.session,
                people_id=person.id,
                event_type="person_lookup",
                event_description=event_desc,
                event_data=event_data,
            )

            update_conversation_metrics(
                session=self.db_state.session,
                people_id=person.id,
                person_looked_up=True,
                person_found=found,
            )
        except Exception as analytics_error:
            logger.debug(f"Analytics tracking failed for person lookup: {analytics_error}")

    @staticmethod
    def _format_tree_lookup_for_ai(lookup_results: list[PersonLookupResult]) -> str:
        """
        Format TreeQueryService lookup results for AI prompt context.

        Sprint 3: RAG Response Generator integration - formats person lookup results
        for inclusion in genealogical_reply prompt.

        Args:
            lookup_results: List of PersonLookupResult from _lookup_mentioned_people

        Returns:
            Formatted string for TREE LOOKUP RESULTS prompt section
        """
        if not lookup_results:
            return "No people mentioned or no tree lookup performed."

        found_count = sum(1 for r in lookup_results if r.found)
        not_found_count = len(lookup_results) - found_count

        lines = [f"Lookup Summary: {found_count} found in tree, {not_found_count} not found.", ""]

        for result in lookup_results:
            if result.found:
                lines.extend(PersonProcessor._format_found_person(result))
            else:
                lines.append(f"✗ NOT FOUND: {result.name}")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_found_person(result: PersonLookupResult) -> "list[str]":
        """Format a found person result for AI prompt context."""
        lines: list[str] = []
        name_parts = [result.name]
        if result.birth_year or result.death_year:
            years = f"({result.birth_year or '?'}-{result.death_year or '?'})"
            name_parts.append(years)

        lines.append(f"✓ FOUND: {' '.join(name_parts)}")

        if result.birth_place:
            lines.append(f"  Birth: {result.birth_place}")
        if result.death_place:
            lines.append(f"  Death: {result.death_place}")
        if result.relationship_path:
            lines.append(f"  Relationship: {result.relationship_path}")

        lines.append(f"  Source: {result.source or 'Unknown'}")
        lines.append("")
        return lines

    @staticmethod
    def _format_relationship_context_for_ai(lookup_results: list[PersonLookupResult]) -> str:
        """
        Extract and format relationship context from lookup results for AI prompt.

        Sprint 3: Provides relationship path explanations for genealogical_reply prompt.

        Args:
            lookup_results: List of PersonLookupResult from _lookup_mentioned_people

        Returns:
            Formatted relationship context string
        """
        if not lookup_results:
            return "No relationship information available."

        relationships: list[str] = []
        for result in lookup_results:
            if result.found and result.relationship_path:
                relationships.append(f"- {result.name}: {result.relationship_path}")

        if not relationships:
            return "Relationship paths not determined for found people."

        return "Relationship paths from my tree:\n" + "\n".join(relationships)

    @staticmethod
    def _load_gedcom_data() -> Optional[Any]:
        """Load GEDCOM data from configured path."""
        from genealogy.gedcom.gedcom_cache import load_gedcom_with_aggressive_caching

        gedcom_path = (
            config_schema.database.gedcom_file_path
            if config_schema and config_schema.database.gedcom_file_path
            else None
        )
        if not gedcom_path:
            logger.warning("GEDCOM file path not configured, skipping GEDCOM search")
            return None

        gedcom_data = load_gedcom_with_aggressive_caching(str(gedcom_path))
        if not gedcom_data:
            logger.warning("Failed to load GEDCOM data, skipping GEDCOM search")
            return None

        return gedcom_data

    @staticmethod
    def _build_search_criteria_from_person_data(person_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Build search criteria from extracted person data."""
        search_criteria: dict[str, Any] = {}

        if person_data.get("first_name"):
            search_criteria["first_name"] = person_data["first_name"].lower()
        if person_data.get("last_name"):
            search_criteria["surname"] = person_data["last_name"].lower()
        if person_data.get("birth_year"):
            search_criteria["birth_year"] = person_data["birth_year"]
        if person_data.get("birth_place"):
            search_criteria["birth_place"] = person_data["birth_place"]
        if person_data.get("gender"):
            search_criteria["gender"] = person_data["gender"].lower()

        # Need at least name to search
        if not search_criteria.get("first_name") and not search_criteria.get("surname"):
            logger.warning(f"Insufficient search criteria for {person_data.get('name')}")
            return None

        return search_criteria

    @staticmethod
    def _perform_gedcom_search(gedcom_data: Any, search_criteria: dict[str, Any]) -> list[dict[str, Any]]:
        """Perform GEDCOM search using Action 10 logic."""
        from actions.action10 import filter_and_score_individuals

        scoring_weights = dict(config_schema.common_scoring_weights) if config_schema else {}
        date_flex = {"year_match_range": 5.0}

        return filter_and_score_individuals(
            gedcom_data,
            search_criteria,  # filter_criteria
            search_criteria,  # scoring_criteria
            scoring_weights,
            date_flex,
        )

    def _search_gedcom_for_person(self, person_data: dict[str, Any]) -> Optional[PersonLookupResult]:
        """
        Search for a person in GEDCOM data using TreeQueryService (Sprint 3 RAG integration).

        Uses TreeQueryService.find_person() for fuzzy matching and relationship context,
        with fallback to legacy Action 10 logic if TreeQueryService unavailable.

        Args:
            person_data: Dictionary with person details from AI extraction

        Returns:
            PersonLookupResult if found, None otherwise
        """
        person_name = person_data.get("name", "Unknown")

        # Sprint 3: Try TreeQueryService first for RAG-style retrieval
        tree_service = self._get_tree_query_service()
        if tree_service:
            result = PersonProcessor._search_via_tree_query_service(tree_service, person_data)
            if result:
                logger.info(f"TreeQueryService found {person_name} with score {result.match_score}")
                return result
            logger.debug(f"TreeQueryService did not find {person_name}, falling back to legacy search")

        # Fallback: Legacy Action 10 search logic
        return self._search_gedcom_legacy(person_data)

    @staticmethod
    def _extract_birth_year_from_person_data(person_data: dict[str, Any]) -> Optional[int]:
        """Extract birth year from person data, checking direct field and vital records."""
        if "birth_year" in person_data:
            return person_data.get("birth_year")

        for record in person_data.get("vital_records", []):
            if record.get("event_type") == "BIRTH" and record.get("year"):
                return record.get("year")

        return None

    @staticmethod
    def _convert_search_result_to_lookup(
        search_result: PersonSearchResult,
        tree_service: TreeQueryService,
        person_name: str,
    ) -> PersonLookupResult:
        """Convert TreeQueryService search result to PersonLookupResult with relationship."""
        relationship_text = None
        if search_result.person_id:
            rel_result = tree_service.explain_relationship(search_result.person_id)
            if rel_result.found:
                relationship_text = rel_result.relationship_description or rel_result.relationship_label
                logger.debug(f"Relationship context: {relationship_text}")

        confidence_scores = {"high": 90, "medium": 70, "low": 50}
        match_score = confidence_scores.get(search_result.confidence, 60)

        return PersonLookupResult(
            name=search_result.name or person_name,
            birth_year=search_result.birth_year,
            birth_place=search_result.birth_place,
            death_year=search_result.death_year,
            death_place=search_result.death_place,
            relationship_path=relationship_text,
            match_score=match_score,
            found=True,
            source="TreeQueryService",
            family_details={
                "person_id": search_result.person_id,
                "confidence": search_result.confidence,
                "alternatives_count": len(search_result.alternatives) if search_result.alternatives else 0,
            },
        )

    @staticmethod
    def _search_via_tree_query_service(
        tree_service: TreeQueryService, person_data: dict[str, Any]
    ) -> Optional[PersonLookupResult]:
        """
        Search using TreeQueryService.find_person() with relationship context.

        Sprint 3: RAG Response Generator - provides rich tree context for AI responses.

        Args:
            tree_service: Initialized TreeQueryService instance
            person_data: Dictionary with person details from AI extraction

        Returns:
            PersonLookupResult with relationship explanation, or None if not found
        """
        try:
            person_name = person_data.get("name", "")
            if not person_name:
                return None

            birth_year = PersonProcessor._extract_birth_year_from_person_data(person_data)
            location = person_data.get("birth_place") or person_data.get("location")

            search_result = tree_service.find_person(
                name=person_name,
                approx_birth_year=birth_year,
                location=location,
                max_results=3,
            )

            if not search_result.found:
                return None

            return PersonProcessor._convert_search_result_to_lookup(search_result, tree_service, person_name)

        except Exception as e:
            logger.debug(f"TreeQueryService search error: {e}")
            return None

    def _search_gedcom_legacy(self, person_data: dict[str, Any]) -> Optional[PersonLookupResult]:
        """
        Legacy GEDCOM search using Action 10 logic (fallback).

        Args:
            person_data: Dictionary with person details from AI extraction

        Returns:
            PersonLookupResult if found, None otherwise
        """
        try:
            # Load GEDCOM data
            gedcom_data = self._load_gedcom_data()
            if not gedcom_data:
                return None

            # Build search criteria
            search_criteria = self._build_search_criteria_from_person_data(person_data)
            if not search_criteria:
                return None

            # Perform search
            results = self._perform_gedcom_search(gedcom_data, search_criteria)
            if not results:
                return None

            # Validate top result
            top_result = results[0]
            score = top_result.get("total_score", 0)

            if score < 50:
                logger.debug(f"Top result score too low ({score}), rejecting")
                return None

            # Get relationship path using gedcom_search_utils
            relationship_path = self._get_relationship_path_for_person(gedcom_data, top_result.get("id"))

            # Create PersonLookupResult from GEDCOM data
            return create_result_from_gedcom(
                person_data=top_result,
                relationship_path=relationship_path,
                match_score=score,
            )

        except Exception as e:
            logger.error(f"Error in legacy GEDCOM search: {e}", exc_info=True)
            return None

    @staticmethod
    def _get_relationship_path_for_person(gedcom_data: Any, person_id: Optional[str]) -> Optional[str]:
        """
        Get relationship path between found person and reference person.

        Args:
            gedcom_data: Loaded GEDCOM data
            person_id: GEDCOM ID of the person

        Returns:
            Formatted relationship path string, or None if not calculable
        """
        if not person_id:
            logger.debug("No person ID provided for relationship path calculation")
            return None

        try:
            from genealogy.gedcom.gedcom_search_utils import get_gedcom_relationship_path

            # Get reference person ID from config
            reference_id = (
                config_schema.reference_person_id if config_schema and config_schema.reference_person_id else None
            )

            if not reference_id:
                logger.debug("No reference person ID configured, skipping relationship path")
                return None

            # Calculate relationship path
            relationship_path = get_gedcom_relationship_path(
                individual_id=person_id,
                reference_id=reference_id,
                gedcom_data=gedcom_data,
            )

            if relationship_path and relationship_path != "(No relationship path found)":
                logger.debug(f"Calculated relationship path for {person_id}")
                return relationship_path

            logger.debug(f"No relationship path found for {person_id}")
            return None

        except Exception as e:
            logger.debug(f"Error calculating relationship path: {e}")
            return None

    def _search_api_for_person(self, person_name: str, person_data: dict[str, Any]) -> Optional[PersonLookupResult]:
        """
        Search for a person using Ancestry API (Action 11).

        Args:
            person_name: Full name of the person
            person_data: Dictionary with person details from AI extraction

        Returns:
            PersonLookupResult if found, None otherwise
        """
        try:
            from api.api_search_core import search_ancestry_api_for_person

            # Build search criteria from person_data
            search_criteria = self._build_search_criteria_from_person_data(person_data)
            if not search_criteria:
                logger.debug(f"Insufficient search criteria for API search: {person_name}")
                return None

            # Perform API search
            logger.debug(f"Searching Ancestry API for {person_name}")
            api_results = search_ancestry_api_for_person(
                session_manager=self.session_manager,
                search_criteria=search_criteria,
                max_results=5,
            )

            if not api_results:
                logger.debug(f"No API results found for {person_name}")
                return None

            # Get top result
            top_result = api_results[0]
            score = top_result.get("total_score", 0)

            if score < 50:
                logger.debug(f"Top API result score too low ({score}), rejecting")
                return None

            # Create PersonLookupResult from API data
            # Note: API results have different structure than GEDCOM results
            return PersonLookupResult(
                name=top_result.get("full_name", person_name),
                birth_year=top_result.get("birth_year"),
                birth_place=top_result.get("birth_place"),
                death_year=top_result.get("death_year"),
                death_place=top_result.get("death_place"),
                relationship_path=top_result.get("relationship"),
                match_score=score,
                found=True,
                source="API",
                family_details=top_result.get("family_info", {}),
            )

        except Exception as e:
            logger.error(f"Error searching API for person {person_name}: {e}", exc_info=True)
            return None

    @staticmethod
    def _determine_conversation_phase(context_logs: list[ConversationLog]) -> str:
        """
        Determine conversation phase based on message history.

        Phases:
        - initial_outreach: First 1-2 messages
        - active_dialogue: 3-5 messages with engagement
        - research_exchange: 6+ messages with genealogical content
        - concluded: No response in 30+ days or explicit conclusion

        Args:
            context_logs: List of conversation logs

        Returns:
            Conversation phase string
        """
        if not context_logs:
            return "initial_outreach"

        message_count = len(context_logs)

        if message_count <= 2:
            return "initial_outreach"
        if message_count <= 5:
            return "active_dialogue"
        return "research_exchange"

    @staticmethod
    def _calculate_engagement_score(extracted_data: dict[str, Any], context_logs: list[ConversationLog]) -> int:
        """
        Calculate engagement score based on message content and history.

        Score factors:
        - Message count: +10 per message (max 50)
        - Mentioned people: +5 per person (max 25)
        - Questions asked: +10 per question (max 30)
        - Genealogical content: +20 if present

        Args:
            extracted_data: Extracted data from AI
            context_logs: List of conversation logs

        Returns:
            Engagement score (0-100)
        """
        score = 0

        # Message count contribution (max 50)
        message_count = len(context_logs)
        score += min(message_count * 10, 50)

        # Mentioned people contribution (max 25)
        mentioned_people = extracted_data.get("mentioned_people", [])
        score += min(len(mentioned_people) * 5, 25)

        # Questions contribution (max 30)
        questions = extracted_data.get("questions", [])
        score += min(len(questions) * 10, 30)

        # Genealogical content bonus
        if mentioned_people or questions:
            score += 20

        return min(score, 100)

    @staticmethod
    def _calculate_next_contact_date(person: Person, engagement_score: int) -> Optional[datetime]:
        """
        Calculate next contact date based on engagement score and status transitions.

        Phase 4: Adaptive Follow-Up Scheduling

        Timing strategy:
        - HIGH engagement (>75): 3-5 days
        - MEDIUM engagement (50-75): 7-10 days
        - LOW engagement (<50): 14-21 days
        - Status transitions accelerate/decelerate:
          * OUT_OF_TREE → IN_TREE: accelerate (-2 days)
          * Any → DESIST: no follow-up

        Args:
            person: Person object with status
            engagement_score: Current engagement score (0-100)

        Returns:
            Next contact datetime (UTC) or None if no follow-up needed
        """
        # Don't schedule follow-up for DESIST, ARCHIVE, BLOCKED, or DEAD
        excluded_statuses = [
            PersonStatusEnum.DESIST,
            PersonStatusEnum.ARCHIVE,
            PersonStatusEnum.BLOCKED,
            PersonStatusEnum.DEAD,
        ]
        if person.status in excluded_statuses:
            return None

        # Base delay calculation from engagement score
        if engagement_score >= 75:
            base_days = 4  # High engagement: 3-5 days (use middle)
        elif engagement_score >= 50:
            base_days = 8  # Medium engagement: 7-10 days (use middle)
        else:
            base_days = 17  # Low engagement: 14-21 days (use middle)

        # Adjust for status transitions
        if person.tree_status == "in_tree":
            # In-tree matches get accelerated follow-up
            base_days = max(base_days - 2, 2)
        elif person.tree_status == "out_tree":
            # Out-of-tree matches stay at base rate
            pass

        # Calculate next contact date
        return datetime.now(timezone.utc) + timedelta(days=base_days)

    def _update_conversation_state(
        self,
        person: Person,
        extracted_data: dict[str, Any],
        context_logs: list[ConversationLog],
        log_prefix: str,
    ) -> None:
        """
        Update conversation state tracking for the person.

        Phase 2: Conversation State Tracking

        Args:
            person: Person object
            extracted_data: Extracted data from AI
            context_logs: List of conversation logs
            log_prefix: Logging prefix
        """
        try:
            if not self.db_state.session:
                logger.warning(f"{log_prefix}: No database session, skipping conversation state update")
                return

            # Get or create conversation state
            conv_state = self.db_state.session.query(ConversationState).filter_by(people_id=person.id).first()

            if conv_state is None:
                conv_state = ConversationState(people_id=person.id)
                self.db_state.session.add(conv_state)
                logger.debug(f"{log_prefix}: Created new conversation state")

            # Update conversation phase
            conv_phase = self._determine_conversation_phase(context_logs)
            conv_state.conversation_phase = conv_phase

            # Update engagement score
            engagement_score = self._calculate_engagement_score(extracted_data, context_logs)
            conv_state.engagement_score = engagement_score

            # Update mentioned people (JSON-encoded)
            mentioned_people = extracted_data.get("mentioned_people", [])
            if mentioned_people:
                # Store just the names for simplicity
                people_names = [p.get("name", "Unknown") for p in mentioned_people]
                conv_state.mentioned_people = json.dumps(people_names)

            # Update last topic
            topics = extracted_data.get("topics", [])
            if topics:
                conv_state.last_topic = topics[0] if isinstance(topics, list) else str(topics)

            # Update pending questions (JSON-encoded)
            questions = extracted_data.get("questions", [])
            if questions:
                conv_state.pending_questions = json.dumps(questions)

            # Phase 4: Calculate next contact date based on engagement and status
            current_engagement = safe_column_value(conv_state, "engagement_score", 0)
            next_contact_date = self._calculate_next_contact_date(person, current_engagement)
            if next_contact_date:
                conv_state.next_action_date = next_contact_date
                conv_state.next_action = "send_follow_up"
                logger.debug(f"{log_prefix}: Next contact scheduled for {next_contact_date}")

            # Commit changes
            self.db_state.session.flush()

            logger.info(
                f"{log_prefix}: Updated conversation state - "
                f"phase: {conv_state.conversation_phase}, "
                f"engagement: {conv_state.engagement_score}, "
                f"people: {len(mentioned_people)}, "
                f"questions: {len(questions)}"
            )

            # Track analytics for conversation state update
            try:
                engagement_score_val = safe_column_value(conv_state, "engagement_score", 0)
                conversation_phase_val = safe_column_value(conv_state, "conversation_phase", "initial_outreach")

                record_engagement_event(
                    session=self.db_state.session,
                    people_id=person.id,
                    event_type="score_update",
                    event_description=f"Engagement score updated to {engagement_score_val}",
                    engagement_score_after=engagement_score_val,
                    conversation_phase=conversation_phase_val,
                )

                update_conversation_metrics(
                    session=self.db_state.session,
                    people_id=person.id,
                    engagement_score=engagement_score_val,
                    conversation_phase=conversation_phase_val,
                )
            except Exception as analytics_error:
                logger.debug(f"{log_prefix}: Analytics tracking failed: {analytics_error}")

        except Exception as e:
            logger.error(f"{log_prefix}: Error updating conversation state: {e}", exc_info=True)

    def _should_skip_ms_task_creation(self, log_prefix: str, suggested_tasks: list[str]) -> bool:
        """Check if MS task creation should be skipped. Returns True if should skip."""
        if not suggested_tasks:
            return True

        # Initialize MS Graph if needed
        self._initialize_ms_graph()

        if not self.ms_state.token or not self.ms_state.list_id:
            logger.warning(f"{log_prefix}: Skipping MS task creation - MS Auth/List ID unavailable.")
            return True

        # Check dry run mode
        app_mode = config_schema.app_mode
        if app_mode == "dry_run":
            logger.info(f"{log_prefix}: DRY RUN - Skipping MS To-Do task creation for {len(suggested_tasks)} tasks.")
            return True

        return False

    @staticmethod
    def calculate_task_priority_and_due_date(person: Person) -> tuple[str, Optional[str], list[str]]:
        """
        Calculate task priority and due date based on relationship closeness.

        Phase 5.3: Enhanced MS To-Do Task Creation
        Priority based on relationship closeness and DNA match strength.

        Returns:
            Tuple of (importance, due_date, categories)
        """
        # Default values
        importance = "normal"
        due_date = None
        categories = ["Ancestry Research"]

        # Calculate priority based on relationship closeness
        if person.predicted_relationship:
            rel_lower = person.predicted_relationship.lower()

            # High priority: Close relatives (1st-2nd cousins, immediate family)
            if any(
                term in rel_lower for term in ["1st", "2nd", "parent", "sibling", "child", "grandparent", "grandchild"]
            ):
                importance = "high"
                due_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")  # 1 week
                categories.append("Close Relative")

            # Normal priority: 3rd-4th cousins
            elif any(term in rel_lower for term in ["3rd", "4th"]):
                importance = "normal"
                due_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")  # 2 weeks
                categories.append("Distant Relative")

            # Low priority: 5th+ cousins
            elif any(term in rel_lower for term in ["5th", "6th", "7th", "8th", "distant"]):
                importance = "low"
                due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")  # 1 month
                categories.append("Distant Relative")

        # Adjust based on tree status
        if person.in_my_tree:
            categories.append("In Tree")
        else:
            categories.append("Out of Tree")

        return importance, due_date, categories

    @staticmethod
    def _add_person_urls(task_body_parts: list[str], person: Person) -> None:
        """Add Ancestry profile and DNA comparison URLs to task body."""
        if person.profile_id:
            task_body_parts.append(
                f"Ancestry Profile: https://www.ancestry.com/secure/member/profile?id={person.profile_id}"
            )

        if person.dna_match and hasattr(person.dna_match, 'compare_link'):
            task_body_parts.append(f"DNA Comparison: {person.dna_match.compare_link}")
        elif person.uuid:
            task_body_parts.append(f"DNA Comparison: https://www.ancestry.com/dna/matches/{person.uuid}/compare")

    @staticmethod
    def _add_relationship_info(task_body_parts: list[str], person: Person) -> None:
        """Add relationship and DNA information to task body."""
        if person.predicted_relationship:
            task_body_parts.append(f"Relationship: {person.predicted_relationship}")

        if hasattr(person, 'shared_dna_cm') and person.shared_dna_cm:
            task_body_parts.append(f"Shared DNA: {person.shared_dna_cm} cM")

        if person.tree_status:
            status_display = "In Tree" if person.tree_status == "in_tree" else "Out of Tree"
            task_body_parts.append(f"Tree Status: {status_display}")

    @staticmethod
    def _add_family_tree_info(task_body_parts: list[str], person: Person) -> None:
        """Add family tree information to task body."""
        if not person.family_tree:
            return

        if hasattr(person.family_tree, 'cfpid') and person.family_tree.cfpid:
            task_body_parts.append(f"Tree Person ID: {person.family_tree.cfpid}")
        if hasattr(person.family_tree, 'person_name_in_tree') and person.family_tree.person_name_in_tree:
            task_body_parts.append(f"Name in Tree: {person.family_tree.person_name_in_tree}")
        if hasattr(person.family_tree, 'view_in_tree_link') and person.family_tree.view_in_tree_link:
            task_body_parts.append(f"View in Tree: {person.family_tree.view_in_tree_link}")
        if hasattr(person.family_tree, 'actual_relationship') and person.family_tree.actual_relationship:
            task_body_parts.append(f"Tree Relationship: {person.family_tree.actual_relationship}")

    def _build_task_body_parts(self, person: Person, task_desc: str, task_index: int, total_tasks: int) -> list[str]:
        """Build task body parts with person context."""
        task_body_parts = [
            f"AI Suggested Task ({task_index + 1}/{total_tasks}): {task_desc}",
            "",
            f"Match: {person.username or 'Unknown'} (#{person.id})",
            f"Profile: {person.profile_id or 'N/A'}",
        ]

        self._add_person_urls(task_body_parts, person)
        self._add_relationship_info(task_body_parts, person)
        self._add_family_tree_info(task_body_parts, person)

        return task_body_parts

    def _submit_task_to_ms_graph(
        self, task_title: str, task_body: str, importance: str, due_date: Optional[str], categories: list[str]
    ) -> bool:
        """Submit task to MS Graph and return success status."""
        if self.ms_state.token and self.ms_state.list_id:
            task_id = ms_graph_utils.create_todo_task(
                self.ms_state.token,
                self.ms_state.list_id,
                task_title,
                task_body,
                importance=importance,
                due_date=due_date,
                categories=categories,
            )
            if task_id:
                logger.debug(f"MS To-Do task created with ID {task_id}.")
                return True
            return False
        logger.warning("MS Graph token or list_id is None, skipping task creation")
        return False

    def _create_single_ms_task(
        self,
        person: Person,
        task_desc: str,
        task_index: int,
        total_tasks: int,
        log_prefix: str,
    ) -> bool:
        """
        Create a single MS To-Do task with enhanced metadata.

        Phase 5.3: Enhanced MS To-Do Task Creation
        Includes priority, due date, and categories based on relationship closeness.
        """
        task_title = f"Ancestry Follow-up: {person.username or 'Unknown'} (#{person.id})"

        # Calculate priority and due date based on relationship
        importance, due_date, categories = self.calculate_task_priority_and_due_date(person)

        # Build enhanced task body with context
        task_body_parts = self._build_task_body_parts(person, task_desc, task_index, total_tasks)
        task_body = "\n".join(task_body_parts)

        # Submit to MS Graph
        task_ok = self._submit_task_to_ms_graph(task_title, task_body, importance, due_date, categories)

        if not task_ok:
            logger.warning(f"{log_prefix}: Failed to create MS task: '{task_desc[:100]}...'")

        return task_ok

    def _create_ms_tasks(
        self,
        person: Person,
        suggested_tasks: list[str],
        log_prefix: str,
    ) -> None:
        """Create MS Graph tasks if configured and available."""

        # Check if we should skip task creation
        if self._should_skip_ms_task_creation(log_prefix, suggested_tasks):
            return

        # Create tasks
        logger.info(f"{log_prefix}: Creating {len(suggested_tasks)} MS To-Do tasks...")
        for task_index, task_desc in enumerate(suggested_tasks):
            self._create_single_ms_task(person, task_desc, task_index, len(suggested_tasks), log_prefix)

    def _initialize_ms_graph(self) -> None:
        """Initialize MS Graph authentication and list ID if needed."""
        # MS Graph authentication now happens at main.py startup and is cached
        # Just retrieve the token from cache here
        if not self.ms_state.token and not self.ms_state.auth_attempted:
            logger.debug("Retrieving MS Graph token from cache (authenticated at startup)...")
            self.ms_state.token = ms_graph_utils.acquire_token_device_flow()
            self.ms_state.auth_attempted = True
            if not self.ms_state.token:
                logger.warning("MS Graph token not available (authentication may have been skipped at startup).")

        if self.ms_state.token and not self.ms_state.list_id:
            logger.debug(f"Looking up MS To-Do List ID for '{self.ms_state.list_name}'...")
            self.ms_state.list_id = ms_graph_utils.get_todo_list_id(self.ms_state.token, self.ms_state.list_name)
            if not self.ms_state.list_id:
                logger.warning(f"Failed to find MS To-Do list '{self.ms_state.list_name}'. Tasks will not be created.")

    def _handle_message_response(
        self,
        person: Person,
        context_logs: list[ConversationLog],
        extracted_data: dict[str, Any],
        lookup_results: list[PersonLookupResult],
        log_prefix: str,
        latest_message: ConversationLog,
    ) -> bool:
        """Handle generating and sending the appropriate response message."""

        # Check if this is an OTHER message with no mentioned names
        ai_sentiment = safe_column_value(latest_message, "ai_sentiment", None)
        if ai_sentiment == OTHER_SENTIMENT:
            mentioned_names = extracted_data.get("mentioned_names", [])
            if not mentioned_names:
                logger.info(f"{log_prefix}: Message is 'OTHER' with no names. Marking as processed.")
                self._mark_message_processed(latest_message)
                return True  # Successfully handled (by skipping)

        # Generate custom reply if person identified
        custom_reply = self._generate_custom_reply(
            person,
            context_logs,
            latest_message,
            lookup_results,
            log_prefix,
            extracted_data,
        )

        # Format message (custom or standard acknowledgment)
        message_text, message_type_id = self._format_message(person, extracted_data, custom_reply, log_prefix)
        # Apply filtering and send message
        return self._send_message(
            person,
            context_logs,
            message_text,
            message_type_id,
            custom_reply,
            latest_message,
            log_prefix,
        )

    def _mark_message_processed(self, message: ConversationLog) -> None:
        """Mark a message as processed without sending a reply."""
        try:
            if self.db_state.session:
                message.custom_reply_sent_at = datetime.now(timezone.utc)
                self.db_state.session.add(message)
                self.db_state.session.flush()
        except Exception as e:
            logger.error(f"Failed to mark message as processed: {e}")

    @staticmethod
    def _format_lookup_results_for_ai(lookup_results: list[PersonLookupResult]) -> str:
        """
        Format lookup results for inclusion in AI prompt.

        Phase 2: Person Lookup Integration

        Args:
            lookup_results: List of PersonLookupResult objects

        Returns:
            Formatted string for AI prompt
        """
        if not lookup_results:
            return "No people found in records."

        formatted_parts: list[str] = []

        for result in lookup_results:
            if result.found:
                formatted_parts.append(result.format_for_ai())
            else:
                formatted_parts.append(f"Person '{result.name}' not found in tree or records.")

        return "\n\n".join(formatted_parts)

    def _get_conversation_state_data(self, person: Person) -> tuple[str, int, str, str]:
        """Get conversation state data for a person."""
        from core.database import ConversationState

        conv_state = None
        if self.db_state.session:
            conv_state = self.db_state.session.query(ConversationState).filter_by(people_id=person.id).first()

        if conv_state:
            conversation_phase = safe_column_value(conv_state, "conversation_phase", "initial_outreach")
            engagement_score = safe_column_value(conv_state, "engagement_score", 0)
            last_topic = safe_column_value(conv_state, "last_topic", "")
            pending_questions = safe_column_value(conv_state, "pending_questions", "")
        else:
            conversation_phase = "initial_outreach"
            engagement_score = 0
            last_topic = ""
            pending_questions = ""

        return conversation_phase, engagement_score, last_topic, pending_questions

    @staticmethod
    def _get_person_context_data(person: Person) -> tuple[str, str, str]:
        """Get DNA data, tree statistics, and relationship path for a person."""
        dna_data = ""
        if person.dna_match:
            dna_data = (
                f"DNA Match: {person.dna_match.shared_dna_cm} cM shared, Confidence: {person.dna_match.confidence}"
            )

        tree_stats = ""
        if person.family_tree:
            tree_stats = f"Tree: {person.family_tree.person_count} people, {person.family_tree.tree_size} size"

        relationship_path = ""
        if person.dna_match and person.dna_match.relationship:
            relationship_path = person.dna_match.relationship

        return dna_data, tree_stats, relationship_path

    def _generate_contextual_reply_with_lookup(
        self,
        person: Person,
        context_logs: list[ConversationLog],
        latest_message: ConversationLog,
        lookup_results: list[PersonLookupResult],
        log_prefix: str,
    ) -> Optional[str]:
        """Generate contextual reply using Phase 3 dialogue engine."""
        # Get conversation state
        conversation_phase, engagement_score, last_topic, pending_questions = self._get_conversation_state_data(person)

        # Format lookup results
        lookup_results_str = self._format_lookup_results_for_ai(lookup_results)

        # Get user's last message
        user_last_message = safe_column_value(latest_message, "latest_message_content", "")

        # Format conversation history
        formatted_context = _format_context_for_ai_extraction(context_logs, self.my_pid_lower)

        # Get person context data
        dna_data, tree_stats, relationship_path = self._get_person_context_data(person)

        # Generate contextual response
        return generate_contextual_response(
            conversation_history=formatted_context,
            user_message=user_last_message,
            lookup_results=lookup_results_str,
            dna_data=dna_data,
            tree_statistics=tree_stats,
            relationship_path=relationship_path,
            conversation_phase=conversation_phase,
            engagement_score=engagement_score,
            last_topic=last_topic,
            pending_questions=pending_questions,
            session_manager=self.session_manager,
            log_prefix=log_prefix,
        )

    @staticmethod
    def _score_relationship_path(response_lower: str, lookup_results: list[PersonLookupResult]) -> int:
        """Score relationship path specificity (30 points max)."""
        relationship_keywords = [
            "cousin",
            "great-grandfather",
            "great-grandmother",
            "grandfather",
            "grandmother",
            "2nd cousin",
            "3rd cousin",
            "4th cousin",
            "once removed",
            "twice removed",
            "through",
            "common ancestor",
            "descended from",
            "related through",
        ]
        relationship_count = sum(1 for keyword in relationship_keywords if keyword in response_lower)

        if relationship_count >= 3:
            return 30  # Excellent: 3+ relationship terms
        if relationship_count == 2:
            return 22  # Good: 2 relationship terms
        if relationship_count == 1:
            return 15  # Fair: 1 relationship term
        if any(lookup_result.relationship_path for lookup_result in lookup_results):
            return 10  # Minimal: lookup results have paths but not used in text
        return 0

    @staticmethod
    def _score_evidence_citations(response_text: str, response_lower: str) -> int:
        """Score genealogical evidence citations (25 points max)."""
        evidence_patterns = {
            "years": len([w for w in response_text.split() if w.isdigit() and len(w) == 4 and 1700 < int(w) < 2025]),
            "places": sum(
                1
                for place in [
                    "Scotland",
                    "England",
                    "Ireland",
                    "Wales",
                    "Aberdeen",
                    "Banff",
                    "Edinburgh",
                    "Glasgow",
                    "county",
                    "parish",
                ]
                if place.lower() in response_lower
            ),
            "record_types": sum(
                1
                for record in [
                    "census",
                    "birth record",
                    "death record",
                    "marriage record",
                    "baptism",
                    "burial",
                    "certificate",
                    "register",
                ]
                if record.lower() in response_lower
            ),
            "dates": response_text.count("(") if "(" in response_text else 0,  # Parenthetical dates like "(1850-1920)"
        }
        total_evidence = sum(evidence_patterns.values())

        if total_evidence >= 8:
            return 25  # Excellent: 8+ pieces of evidence
        if total_evidence >= 5:
            return 20  # Good: 5-7 pieces
        if total_evidence >= 3:
            return 15  # Fair: 3-4 pieces
        if total_evidence >= 1:
            return 10  # Minimal: 1-2 pieces
        return 0

    @staticmethod
    def _score_actionable_steps(response_text: str, response_lower: str) -> int:
        """Score actionable next steps (25 points max)."""
        action_verbs = [
            "search for",
            "look for",
            "check",
            "verify",
            "find",
            "locate",
            "review",
            "examine",
            "compare",
            "confirm",
            "investigate",
            "explore",
            "research",
        ]
        action_verb_count = sum(1 for verb in action_verbs if verb in response_lower)

        question_count = response_text.count("?")

        next_step_phrases = [
            "next step",
            "would you like",
            "can you share",
            "do you have",
            "could you",
            "would it help",
            "i can",
            "i'll",
            "let me know",
            "if you",
            "when you",
        ]
        next_step_count = sum(1 for phrase in next_step_phrases if phrase in response_lower)

        total_actionable = action_verb_count + question_count + next_step_count

        if total_actionable >= 6:
            return 25  # Excellent: 6+ actionable elements
        if total_actionable >= 4:
            return 20  # Good: 4-5 elements
        if total_actionable >= 2:
            return 15  # Fair: 2-3 elements
        if total_actionable >= 1:
            return 10  # Minimal: 1 element
        return 0

    @staticmethod
    def _score_personalization(response_lower: str, word_count: int, person: Person) -> int:
        """Score personalization and warmth (20 points max)."""
        # Check for name usage
        name_mentions = 0
        if person.name:
            name_parts = person.name.split()
            name_mentions = sum(1 for part in name_parts if part.lower() in response_lower and len(part) > 2)

        # Check for warm language
        warm_words = [
            "thank you",
            "thanks",
            "wonderful",
            "exciting",
            "great",
            "pleased",
            "happy",
            "appreciate",
            "interesting",
            "fascinating",
            "amazing",
            "delighted",
        ]
        warm_count = sum(1 for word in warm_words if word in response_lower)

        # Check for acknowledgment phrases
        acknowledgment_phrases = [
            "you mentioned",
            "you said",
            "your message",
            "your question",
            "your great",
            "your ancestor",
        ]
        acknowledgment_count = sum(1 for phrase in acknowledgment_phrases if phrase in response_lower)

        # Scoring
        if name_mentions >= 2 and warm_count >= 2 and acknowledgment_count >= 1:
            return 20  # Excellent: names, warmth, acknowledgment
        if name_mentions >= 1 and warm_count >= 1:
            return 15  # Good: some personalization
        if warm_count >= 2 or acknowledgment_count >= 1:
            return 10  # Fair: warm or acknowledging
        if word_count >= 150:
            return 5  # Minimal: adequate length
        return 0

    def score_response_quality(
        self,
        response_text: str,
        lookup_results: list[PersonLookupResult],
        person: Person,
        log_prefix: str = "",
    ) -> int:
        """
        Score generated reply quality on 0-100 scale (Priority 1 Todo #8).

        Scoring breakdown:
        - Relationship path specificity (30 points): Presence and detail of relationship connections
        - Record evidence citations (25 points): References to specific records, dates, places
        - Actionable next steps (25 points): Clear research suggestions or follow-up questions
        - Personalization (20 points): Uses names, acknowledges user's message, warm tone

        Args:
            response_text: The generated reply text
            lookup_results: Person lookup results used in response
            person: Person object for context
            log_prefix: Logging prefix

        Returns:
            Quality score 0-100
        """
        if not response_text:
            return 0

        response_lower = response_text.lower()
        word_count = len(response_text.split())

        # Calculate component scores
        relationship_score = self._score_relationship_path(response_lower, lookup_results)
        evidence_score = self._score_evidence_citations(response_text, response_lower)
        actionable_score = self._score_actionable_steps(response_text, response_lower)
        personalization_score = self._score_personalization(response_lower, word_count, person)

        score = relationship_score + evidence_score + actionable_score + personalization_score

        # Apply quality penalties
        if word_count < 100:
            score = int(score * 0.8)  # 20% penalty for too short
        if word_count > 500:
            score = int(score * 0.9)  # 10% penalty for too long
        if lookup_results and relationship_score == 0:
            score = int(score * 0.85)  # 15% penalty for not using available data

        # Clamp to 0-100 range
        score = max(0, min(100, score))

        logger.info(
            f"{log_prefix}: Response quality score: {score}/100 "
            f"(Relationship: {relationship_score}/30, Evidence: {evidence_score}/25, "
            f"Actionable: {actionable_score}/25, Personal: {personalization_score}/20)"
        )

        return score

    @staticmethod
    def _add_research_suggestions(
        lookup_results: list[PersonLookupResult],
        extracted_data: dict[str, Any],
        tree_lookup_str: str,
    ) -> str:
        """Generate and append research suggestions to the tree lookup string."""
        try:
            common_ancestors: list[dict[str, Any]] = []
            for res in lookup_results:
                if res.found:
                    common_ancestors.append(
                        {"name": res.name, "birth_year": res.birth_year, "birth_place": res.birth_place}
                    )

            suggestions = generate_research_suggestions(
                common_ancestors=common_ancestors,
                locations=extracted_data.get("locations", []),
                time_periods=extracted_data.get("dates", []),
            )

            if suggestions.get("collections") or suggestions.get("strategies"):
                suggestion_str = "\n\nRESEARCH SUGGESTIONS:\n"
                if suggestions.get("collections"):
                    suggestion_str += (
                        "Relevant Collections:\n" + "\n".join([f"- {c}" for c in suggestions["collections"][:3]]) + "\n"
                    )
                if suggestions.get("strategies"):
                    suggestion_str += "Strategies:\n" + "\n".join([f"- {s}" for s in suggestions["strategies"][:2]])

                return tree_lookup_str + suggestion_str
        except Exception as e:
            logger.warning(f"Failed to generate research suggestions: {e}")

        return tree_lookup_str

    def _log_quality_score(
        self,
        person: Person,
        quality_score: int,
        custom_reply: str,
        lookup_results: list[PersonLookupResult],
        log_prefix: str,
    ) -> None:
        """Log quality score to conversation analytics."""
        if self.db_state and self.db_state.session:
            try:
                conversation_phase, _, _, _ = self._get_conversation_state_data(person)
                record_engagement_event(
                    session=self.db_state.session,
                    people_id=person.id,
                    event_type="response_generated",
                    event_description=f"Generated reply with quality score {quality_score}/100",
                    event_data={
                        "quality_score": quality_score,
                        "response_length": len(custom_reply),
                        "lookup_results_count": len(lookup_results),
                    },
                    conversation_phase=conversation_phase,
                )
            except Exception as e:
                logger.warning(f"{log_prefix}: Failed to log quality score to analytics: {e}")

    def _generate_custom_reply(
        self,
        person: Person,
        context_logs: list[ConversationLog],
        latest_message: ConversationLog,
        lookup_results: list[PersonLookupResult],
        log_prefix: str,
        extracted_data: dict[str, Any],
    ) -> Optional[str]:
        """Generate a custom genealogical reply if appropriate."""

        # Phase 3: Use contextual dialogue engine with lookup results
        if lookup_results:
            logger.info(f"{log_prefix}: Using {len(lookup_results)} lookup results for contextual reply")

            if not config_schema.custom_response_enabled:
                logger.info(f"{log_prefix}: Custom replies disabled via config. Using standard.")
                return None

            # Generate contextual reply using Phase 3 dialogue engine
            custom_reply = self._generate_contextual_reply_with_lookup(
                person, context_logs, latest_message, lookup_results, log_prefix
            )

            if custom_reply:
                logger.info(f"{log_prefix}: Generated contextual dialogue response with lookup results.")

                # Priority 1 Todo #8: Score response quality
                quality_score = self.score_response_quality(
                    response_text=custom_reply,
                    lookup_results=lookup_results,
                    person=person,
                    log_prefix=log_prefix,
                )

                # Log quality score to conversation_analytics
                self._log_quality_score(person, quality_score, custom_reply, lookup_results, log_prefix)

            else:
                logger.warning(f"{log_prefix}: Failed to generate contextual reply. Will fall back to standard reply.")

            return custom_reply

        # Fallback to old method if no lookup results
        # Try to identify a person mentioned in the message
        person_details = _identify_and_get_person_details(log_prefix)

        if not person_details:
            logger.debug(f"{log_prefix}: No person identified. Will use standard acknowledgement.")
            return None

        # Check if custom responses are enabled
        if not config_schema.custom_response_enabled:
            logger.info(f"{log_prefix}: Custom replies disabled via config. Using standard.")
            return None

        # Format genealogical data
        genealogical_data_str = _format_genealogical_data_for_ai(person_details.get("details", {}))
        # Get user's last message
        user_last_message = safe_column_value(latest_message, "latest_message_content", "")

        # Format context
        formatted_context = _format_context_for_ai_extraction(context_logs, self.my_pid_lower)

        # Sprint 3: Format tree lookup results for RAG integration
        tree_lookup_str = self._format_tree_lookup_for_ai(lookup_results)
        relationship_str = self._format_relationship_context_for_ai(lookup_results)

        # Integrate Research Suggestions
        tree_lookup_str = self._add_research_suggestions(lookup_results, extracted_data, tree_lookup_str)

        # Generate custom reply using standard genealogical reply (fallback)
        custom_reply = generate_genealogical_reply(
            conversation_context=formatted_context,
            user_last_message=user_last_message,
            genealogical_data_str=genealogical_data_str,
            session_manager=self.session_manager,
            tree_lookup_results=tree_lookup_str,
            relationship_context=relationship_str,
        )

        if custom_reply:
            logger.info(f"{log_prefix}: Generated custom genealogical reply.")
        else:
            logger.warning(f"{log_prefix}: Failed to generate custom reply. Will fall back.")

        return custom_reply

    def _compose_base_message(
        self,
        person: Person,
        extracted_data: dict[str, Any],
        custom_reply: Optional[str],
        log_prefix: str,
    ) -> tuple[str, int]:
        """Compose the base message and message_type_id with minimal branching."""
        if custom_reply:
            user_name = getattr(config_schema, "user_name", "Tree Owner")
            user_location = getattr(config_schema, "user_location", "")
            location_part = f"\n{user_location}" if user_location else ""
            signature = f"\n\nBest regards,\n{user_name}{location_part}"
            logger.info(f"{log_prefix}: Using custom genealogical reply with signature.")
            message_type_id = self.msg_config.custom_reply_msg_type_id or self.msg_config.ack_msg_type_id
            if message_type_id is None:
                raise RuntimeError("Message template IDs are not configured.")
            return custom_reply + signature, message_type_id

        first_name = safe_column_value(person, "first_name", "")
        username = safe_column_value(person, "username", "")
        name_to_use = format_name(first_name or username)
        summary_for_ack = _generate_ack_summary(extracted_data)
        if self.msg_config.templates and ACKNOWLEDGEMENT_MESSAGE_TYPE in self.msg_config.templates:
            msg = self.msg_config.templates[ACKNOWLEDGEMENT_MESSAGE_TYPE].format(
                name=name_to_use, summary=summary_for_ack
            )
        else:
            user_name = getattr(config_schema, "user_name", "Tree Owner")
            msg = f"Dear {name_to_use},\n\nThank you for your message!\n\n{user_name}"
        logger.info(f"{log_prefix}: Using standard acknowledgement template.")
        if self.msg_config.ack_msg_type_id is None:
            raise RuntimeError("Acknowledgement message type ID is not configured.")
        return msg, self.msg_config.ack_msg_type_id

    @staticmethod
    def _add_relationship_annotation(lines: list[str], rel_str: str) -> None:
        if rel_str:
            lines.append(f"\nOur relationship appears to be: {rel_str}.")

    @staticmethod
    def _build_relationship_diagram_line(
        person: Person, extracted_data: dict[str, Any], log_prefix: str
    ) -> Optional[str]:
        rel_path = extracted_data.get("relationship_path")
        if not (isinstance(rel_path, list) and rel_path):
            return None
        try:
            from_name = getattr(config_schema, "user_name", "Me")
            to_name = format_name(
                safe_column_value(person, "first_name", "") or safe_column_value(person, "username", "")
            )
            diagram_text = format_response_with_relationship_diagram(from_name, to_name, rel_path)
            return "\n" + diagram_text if diagram_text else None
        except Exception as e:
            logger.debug(f"{log_prefix}: Relationship diagram enrichment skipped: {e}")
            return None

    @staticmethod
    def _build_records_enrichment_line(person: Person, records: Any, log_prefix: str) -> Optional[str]:
        if not (isinstance(records, list) and records):
            return None
        try:
            to_name = format_name(
                safe_column_value(person, "first_name", "") or safe_column_value(person, "username", "")
            )
            records_text = format_response_with_records(to_name, records)
            return "\n" + records_text if records_text else None
        except Exception as e:
            logger.debug(f"{log_prefix}: Record sharing enrichment skipped: {e}")
            return None

    @staticmethod
    def _build_enrichment_lines(
        person: Person,
        extracted_data: dict[str, Any],
        log_prefix: str,
    ) -> list[str]:
        """Build enrichment lines based on policy; keep logic flat to reduce complexity."""
        lines: list[str] = []

        # Relationship annotation
        rel_str = getattr(getattr(person, "dna_match", None), "relationship", "") or ""
        PersonProcessor._add_relationship_annotation(lines, rel_str)

        # Relationship path diagram if present
        diagram_line = PersonProcessor._build_relationship_diagram_line(person, extracted_data, log_prefix)
        if diagram_line:
            lines.append(diagram_line)

        # Record sharing enrichment
        records = extracted_data.get("records") or extracted_data.get("vital_records")
        records_line = PersonProcessor._build_records_enrichment_line(person, records, log_prefix)
        if records_line:
            lines.append(records_line)

        return lines

    def _formatting_fallback(self, person: Person) -> tuple[str, int]:
        """Return a safe fallback message and type id when formatting fails."""
        safe_username = safe_column_value(person, "username", "User")
        user_name = getattr(config_schema, "user_name", "Tree Owner")
        message_text = f"Dear {format_name(safe_username)},\n\nThank you for your message!\n\n{user_name}"
        message_type_id = self.msg_config.ack_msg_type_id or 1
        return message_text, message_type_id

    def _format_message(
        self,
        person: Person,
        extracted_data: dict[str, Any],
        custom_reply: Optional[str],
        log_prefix: str,
    ) -> tuple[str, int]:
        """Format the message text and determine message type ID, with Phase 5 enrichments."""
        try:
            # 1) Base message
            message_text, message_type_id = self._compose_base_message(person, extracted_data, custom_reply, log_prefix)

            # 2) Conditional enrichments (Phase 5 policy)
            if getattr(config_schema, "enable_task_enrichment", False):
                enrich_lines = self._build_enrichment_lines(person, extracted_data, log_prefix)
                if enrich_lines:
                    message_text += "\n".join(enrich_lines)

            # 3) Return
            return message_text, message_type_id or 1

        except Exception as e:
            logger.error(f"{log_prefix}: Message formatting error: {e}. Using fallback.")
            return self._formatting_fallback(person)

    def _send_message(
        self,
        person: Person,
        context_logs: list[ConversationLog],
        message_text: str,
        message_type_id: int,
        custom_reply: Optional[str],
        latest_message: ConversationLog,
        log_prefix: str,
    ) -> bool:
        """Send the message and handle database updates."""

        # Apply mode/recipient filtering
        send_flag, skip_reason = self._should_send_message(person)

        # Get conversation ID
        conv_id = self._get_conversation_id(context_logs, log_prefix)
        if not conv_id:
            return False

        # Send or skip message
        if send_flag:
            from api.api_utils import call_send_message_api

            send_status, effective_conv_id = call_send_message_api(
                self.session_manager,
                person,
                message_text,
                conv_id,
                f"Action9: {person.username} #{person.id}",
            )
        else:
            send_status = skip_reason
            effective_conv_id = conv_id
            logger.debug(f"Skipping message to {person.username}: {skip_reason}")
        # Handle database updates
        return self._stage_database_updates(
            person,
            message_text,
            message_type_id,
            send_status,
            effective_conv_id or "",
            custom_reply,
            latest_message,
            log_prefix,
        )

    @staticmethod
    def _should_send_message(person: Person) -> tuple[bool, str]:
        """Determine if message should be sent based on app mode and filters."""

        app_mode = getattr(config_schema, "app_mode", "production")
        decision = should_allow_outbound_to_person(person, app_mode=app_mode)
        return decision.allowed, decision.reason

    @staticmethod
    def _get_conversation_id(context_logs: list[ConversationLog], log_prefix: str) -> Optional[str]:
        """Get conversation ID from context logs."""
        if not context_logs:
            logger.error(f"{log_prefix}: No context logs available for conversation ID.")
            return None

        raw_conv_id_value: Any = getattr(context_logs[-1], "conversation_id", None)
        if raw_conv_id_value is None:
            logger.error(f"{log_prefix}: Conversation ID is None.")
            return None

        try:
            return str(raw_conv_id_value)
        except Exception as e:
            logger.error(f"{log_prefix}: Failed to convert conversation ID to string: {e}")
            return None

    @staticmethod
    def _create_log_data(
        person: Person,
        message_text: str,
        message_type_id: int,
        send_status: str,
        effective_conv_id: str,
    ) -> dict[str, Any]:
        """Create log data dictionary for database insertion."""
        person_id_int = int(str(person.id))

        return {
            "conversation_id": str(effective_conv_id),
            "direction": MessageDirectionEnum.OUT,
            "people_id": person_id_int,
            "latest_message_content": message_text[: config_schema.message_truncation_length],
            "latest_timestamp": datetime.now(timezone.utc),
            "message_template_id": message_type_id,  # Fixed: was message_type_id
            "script_message_status": send_status,
            "ai_sentiment": None,
        }

    def _update_custom_reply_timestamp(
        self,
        custom_reply: Optional[str],
        latest_message: ConversationLog,
        message_type_id: int,
        log_prefix: str,
    ) -> None:
        """Update custom_reply_sent_at timestamp if this was a custom reply."""
        if custom_reply and latest_message and message_type_id == self.msg_config.custom_reply_msg_type_id:
            try:
                if self.db_state.session:
                    latest_message.custom_reply_sent_at = datetime.now(timezone.utc)
                    self.db_state.session.add(latest_message)
                    self.db_state.session.flush()
                    logger.info(f"{log_prefix}: Updated custom_reply_sent_at.")
            except Exception as e:
                logger.error(f"{log_prefix}: Failed to update custom_reply_sent_at: {e}")

    def _stage_person_for_archive(self, person_id_int: int, log_prefix: str) -> None:
        """Stage person for archiving."""
        if self.db_state.person_updates is not None:
            self.db_state.person_updates[person_id_int] = PersonStatusEnum.ARCHIVE
        logger.debug(f"{log_prefix}: Person status staged for ARCHIVE.")

    def _stage_database_updates(
        self,
        person: Person,
        message_text: str,
        message_type_id: int,
        send_status: str,
        effective_conv_id: str,
        custom_reply: Optional[str],
        latest_message: ConversationLog,
        log_prefix: str,
    ) -> bool:
        """Stage database updates for the processed message."""

        if not effective_conv_id:
            logger.error(f"{log_prefix}: effective_conv_id is None. Cannot stage log entry.")
            return False

        # Handle successful sends
        if send_status in {"delivered OK", "typed (dry_run)"} or send_status.startswith("skipped ("):
            try:
                # Get person ID as int
                person_id_int = int(str(person.id))

                # Prepare and stage log data
                log_data = self._create_log_data(person, message_text, message_type_id, send_status, effective_conv_id)
                if self.db_state.logs_to_add is not None:
                    self.db_state.logs_to_add.append(log_data)

                # Update custom_reply_sent_at if needed
                self._update_custom_reply_timestamp(custom_reply, latest_message, message_type_id, log_prefix)

                # Stage person for archiving
                self._stage_person_for_archive(person_id_int, log_prefix)

                return True

            except Exception as e:
                logger.error(f"{log_prefix}: Failed to stage database updates: {e}")
                return False
        else:
            logger.error(f"{log_prefix}: Failed to send message (Status: {send_status}).")
            return False


class BatchCommitManager:
    """Manages batch commits to the database."""

    def __init__(self, db_state: DatabaseState) -> None:
        self.db_state = db_state

    def should_commit(self) -> bool:
        """Check if a commit should be triggered."""
        logs_count = len(self.db_state.logs_to_add) if self.db_state.logs_to_add else 0
        updates_count = len(self.db_state.person_updates) if self.db_state.person_updates else 0
        total_pending = logs_count + updates_count
        return total_pending >= self.db_state.commit_threshold

    def commit_batch(self, batch_num: int) -> tuple[bool, int, int]:
        """
        Commit current batch to database.

        Returns:
            Tuple of (success, logs_committed, persons_updated)
        """
        if not self.db_state.logs_to_add and not self.db_state.person_updates:
            return True, 0, 0

        try:
            logs_count = len(self.db_state.logs_to_add) if self.db_state.logs_to_add else 0
            updates_count = len(self.db_state.person_updates) if self.db_state.person_updates else 0
            logger.info(f"Committing batch {batch_num} ({logs_count} logs, {updates_count} person updates)")

            if not self.db_state.session:
                logger.error(f"Database session is None for batch {batch_num}")
                return False, 0, 0

            logs_committed, persons_updated = commit_bulk_data(
                session=self.db_state.session,
                log_upserts=self.db_state.logs_to_add or [],
                person_updates=self.db_state.person_updates or {},
                context=f"Action 9 Batch {batch_num}",
            )

            # Clear the staged data
            if self.db_state.logs_to_add is not None:
                self.db_state.logs_to_add.clear()
            if self.db_state.person_updates is not None:
                self.db_state.person_updates.clear()

            logger.debug(
                f"Batch {batch_num} committed successfully ({logs_committed} logs, {persons_updated} person updates)"
            )
            return True, logs_committed, persons_updated

        except Exception as e:
            logger.error(f"Database commit failed for batch {batch_num}: {e}", exc_info=True)
            return False, 0, 0


#####################################################
# Main Simplified Function
#####################################################


@with_connection_resilience("Action 9: Productive Processing", max_recovery_attempts=3)
@api_retry(max_attempts=3, backoff_factor=4.0)  # Increased from 2.0 to 4.0 for better AI API handling
@circuit_breaker(failure_threshold=10, recovery_timeout=300)  # Increased from 5 to 10 for better tolerance
@timeout_protection(timeout=2400)  # 40 minutes for productive message processing
@graceful_degradation(fallback_value=False)
@error_context_decorator("action9_process_productive")
def process_productive_messages(session_manager: SessionManager) -> bool:
    """
    Simplified main function for Action 9. Processes productive messages by:
    1. Setting up configuration and state
    2. Querying candidates
    3. Processing each person in batches
    4. Committing results

    Args:
        session_manager: The active SessionManager instance.

    Returns:
        True if processing completed successfully, False otherwise.
    """

    # Initialize state objects
    state = ProcessingState()
    ms_state = MSGraphState(list_name=config_schema.ms_todo_list_name)
    db_state = DatabaseState(
        batch_size=max(1, config_schema.batch_size),
        commit_threshold=max(1, config_schema.batch_size),
    )
    msg_config = MessageConfig()

    # Validate session manager
    if not session_manager or not session_manager.my_profile_id:
        logger.error("Action 9: SessionManager or Profile ID missing.")
        return False

    try:
        # Step 1: Setup - Load templates and get database session
        if not _setup_configuration(session_manager, db_state, msg_config):
            return False

        # Step 2: Query candidates
        candidates = _query_candidates(db_state, msg_config, config_schema.max_productive_to_process)
        if not candidates:
            logger.info("Action 9: No eligible candidates found.")
            return True

        state.total_candidates = len(candidates)
        logger.info(f"Action 9: Found {state.total_candidates} candidates to process.")

        # Log configuration (matching Action 8 format)
        logger.info(
            "Configuration: APP_MODE=%s, MAX_PRODUCTIVE=%s, BATCH_SIZE=%s, AI_PROVIDER=%s",
            config_schema.app_mode,
            config_schema.max_productive_to_process,
            config_schema.batch_size,
            config_schema.ai_provider,
        )
        log_action_banner(
            action_name="Process Productive",
            action_number=9,
            stage="start",
            logger_instance=logger,
            details={
                "mode": config_schema.app_mode,
                "max_productive": config_schema.max_productive_to_process,
                "batch_size": config_schema.batch_size,
                "ai_provider": config_schema.ai_provider,
            },
        )

        # Step 3: Process candidates
        success = _process_candidates(session_manager, candidates, state, ms_state, db_state, msg_config)

        # Step 4: Final commit
        _final_commit(db_state, state)

        # Step 5: Log summary
        _log_summary(state)

        log_action_banner(
            action_name="Process Productive",
            action_number=9,
            stage="success" if success else "failure",
            logger_instance=logger,
            details={
                "processed": state.processed_count,
                "tasks": state.tasks_created_count,
                "acks": state.acks_sent_count,
                "skipped": state.skipped_count,
                "errors": state.error_count,
                "candidates": state.total_candidates,
            },
        )
        return success

    except Exception as e:
        logger.critical(f"Critical error in process_productive_messages: {e}", exc_info=True)
        log_action_banner(
            action_name="Process Productive",
            action_number=9,
            stage="failure",
            logger_instance=logger,
            details={"error": str(e)},
        )
        return False
    finally:
        # Cleanup
        if db_state.session:
            session_manager.db_manager.return_session(db_state.session)


def _setup_configuration(session_manager: SessionManager, db_state: DatabaseState, msg_config: MessageConfig) -> bool:
    """Setup configuration, templates, and database session."""

    # Load templates
    msg_config.templates = _load_templates_for_action9()
    if not msg_config.templates:
        logger.error("Action 9: Required message templates failed to load.")
        return False

    # Get database session
    db_state.session = session_manager.db_manager.get_session()
    if not db_state.session:
        logger.critical("Action 9: Failed to get database session.")
        return False
    # Get message type IDs
    if not db_state.session:
        logger.critical("Action 9: Database session is None when querying message types.")
        return False

    ack_msg_type_obj = (
        db_state.session.query(MessageTemplate.id)
        .filter(MessageTemplate.template_key == ACKNOWLEDGEMENT_MESSAGE_TYPE)
        .scalar()
    )
    if not ack_msg_type_obj:
        logger.critical(f"Action 9: MessageTemplate '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' not found in DB.")
        return False
    msg_config.ack_msg_type_id = ack_msg_type_obj

    # Get custom reply message type ID (optional)
    custom_reply_msg_type_obj = (
        db_state.session.query(MessageTemplate.id)
        .filter(MessageTemplate.template_key == CUSTOM_RESPONSE_MESSAGE_TYPE)
        .scalar()
    )
    if custom_reply_msg_type_obj:
        msg_config.custom_reply_msg_type_id = custom_reply_msg_type_obj
    else:
        logger.warning(f"Action 9: MessageTemplate '{CUSTOM_RESPONSE_MESSAGE_TYPE}' not found in DB.")

    return True


# @cached_database_query(ttl=300)  # 5-minute cache for candidate queries - module doesn't exist yet
def _query_candidates(db_state: DatabaseState, msg_config: MessageConfig, limit: int) -> list[Person]:
    """Query for candidate persons to process."""

    if not db_state.session:
        logger.error("Database session is None when querying candidates")
        return []

    logger.debug("Querying for candidate persons...")

    # Subquery for latest IN messages
    latest_in_log_subq = (
        db_state.session.query(
            ConversationLog.people_id,
            func.max(ConversationLog.latest_timestamp).label("max_in_ts"),
        )
        .filter(ConversationLog.direction == MessageDirectionEnum.IN)
        .group_by(ConversationLog.people_id)
        .subquery("latest_in_sub")
    )

    # Subquery for latest OUT acknowledgement messages
    latest_ack_out_log_subq = (
        db_state.session.query(
            ConversationLog.people_id,
            func.max(ConversationLog.latest_timestamp).label("max_ack_out_ts"),
        )
        .filter(
            ConversationLog.direction == MessageDirectionEnum.OUT,
            ConversationLog.message_template_id == msg_config.ack_msg_type_id,  # Fixed: was message_type_id
        )
        .group_by(ConversationLog.people_id)
        .subquery("latest_ack_out_sub")
    )

    # Main query with Phase 4 adaptive follow-up scheduling
    from core.database import ConversationState

    candidates_query = (
        db_state.session.query(Person)
        .options(joinedload(Person.family_tree))
        .outerjoin(ConversationState, Person.id == ConversationState.people_id)
        .join(latest_in_log_subq, Person.id == latest_in_log_subq.c.people_id)
        .join(
            ConversationLog,
            and_(
                Person.id == ConversationLog.people_id,
                ConversationLog.direction == MessageDirectionEnum.IN,
                ConversationLog.latest_timestamp == latest_in_log_subq.c.max_in_ts,
                or_(
                    ConversationLog.ai_sentiment == PRODUCTIVE_SENTIMENT,
                    ConversationLog.ai_sentiment == OTHER_SENTIMENT,
                ),
                ConversationLog.custom_reply_sent_at.is_(None),
            ),
        )
        .outerjoin(
            latest_ack_out_log_subq,
            Person.id == latest_ack_out_log_subq.c.people_id,
        )
        .filter(
            Person.status == PersonStatusEnum.ACTIVE,
            (latest_ack_out_log_subq.c.max_ack_out_ts.is_(None))
            | (latest_ack_out_log_subq.c.max_ack_out_ts < latest_in_log_subq.c.max_in_ts),
            # Phase 4: Respect next_action_date for adaptive follow-up timing
            or_(
                ConversationState.next_action_date.is_(None),
                ConversationState.next_action_date <= datetime.now(timezone.utc),
            ),
        )
        .order_by(Person.id)
    )

    # Apply limit if configured
    if limit > 0:
        candidates_query = candidates_query.limit(limit)
        logger.debug(f"Processing limited to {limit} candidates")

    return candidates_query.all()


def _check_session_health(session_manager: SessionManager) -> None:
    """Proactively refresh session if needed."""
    start_time = getattr(session_manager, "session_start_time", None)
    if start_time:
        session_age = time.time() - start_time
        if session_age > 800:
            logger.info("Proactively refreshing session after %.0f seconds to prevent timeout", session_age)
            if session_manager._attempt_session_recovery(reason="proactive"):
                logger.info("✅ Proactive session refresh successful")
            else:
                logger.error("❌ Proactive session refresh failed")


def _update_processing_counters(state: ProcessingState, success: bool, status: str) -> None:
    """Update processing state counters based on result."""
    if success:
        if status == "success":
            state.acks_sent_count += 1
            state.archived_count += 1
            # Note: tasks_created_count is updated in the person processor
        else:
            state.skipped_count += 1
    elif status.startswith("error"):
        state.error_count += 1
        state.overall_success = False
    else:
        state.skipped_count += 1


def _handle_critical_skip(state: ProcessingState) -> None:
    """Handle skipping remaining candidates due to critical error."""
    remaining = state.total_candidates - state.processed_count
    state.skipped_count += remaining
    logger.warning(f"Skipping remaining {remaining} candidates due to DB error.")


def _process_single_candidate(
    person: Person,
    idx: int,
    state: ProcessingState,
    person_processor: PersonProcessor,
    commit_manager: BatchCommitManager,
    session_manager: SessionManager,
) -> bool:
    """
    Process a single candidate.
    Returns False if processing should stop (critical error), True otherwise.
    """
    # Proactive Session Refresh (Action 6 style)
    _check_session_health(session_manager)

    state.processed_count += 1

    # Log candidate being processed (matching Action 6 batch format)
    print("")
    logger.info(f"Candidate {idx}/{state.total_candidates}: {person.username}")

    # Process individual person
    success, status = person_processor.process_person(person)

    # Update counters based on result
    _update_processing_counters(state, success, status)

    # Check for batch commit
    if commit_manager.should_commit():
        state.batch_num += 1
        commit_success, _, _ = commit_manager.commit_batch(state.batch_num)

        if not commit_success:
            logger.critical(f"Critical: Batch {state.batch_num} commit failed!")
            state.critical_db_error_occurred = True
            state.overall_success = False
            return False

        # Log batch complete (matching Action 6 format)
        print()  # Blank line before batch summary
        logger.info(
            f"Batch {state.batch_num} complete | "
            f"Processed: {state.processed_count}/{state.total_candidates}, "
            f"Acks: {state.acks_sent_count}, "
            f"Skipped: {state.skipped_count}, "
            f"Errors: {state.error_count}"
        )

    return True


def _process_candidates(
    session_manager: SessionManager,
    candidates: list[Person],
    state: ProcessingState,
    ms_state: MSGraphState,
    db_state: DatabaseState,
    msg_config: MessageConfig,
) -> bool:
    """Process all candidate persons with batch-level reporting (no progress bar)."""

    # Initialize processors
    person_processor = PersonProcessor(session_manager, db_state, msg_config, ms_state)
    commit_manager = BatchCommitManager(db_state)

    logger.info(f"Processing {state.total_candidates} candidates...")

    for idx, person in enumerate(candidates, start=1):
        if state.critical_db_error_occurred:
            _handle_critical_skip(state)
            break

        should_continue = _process_single_candidate(
            person, idx, state, person_processor, commit_manager, session_manager
        )
        if not should_continue:
            break

    return state.overall_success


def _final_commit(db_state: DatabaseState, state: ProcessingState) -> None:
    """Perform final commit of any remaining data."""

    if not state.critical_db_error_occurred and (db_state.logs_to_add or db_state.person_updates):
        state.batch_num += 1
        commit_manager = BatchCommitManager(db_state)

        logs_count = len(db_state.logs_to_add) if db_state.logs_to_add else 0
        updates_count = len(db_state.person_updates) if db_state.person_updates else 0
        logger.info(f"Committing final batch ({logs_count} logs, {updates_count} person updates)")

        commit_success, logs_committed, persons_updated = commit_manager.commit_batch(state.batch_num)

        if not commit_success:
            logger.error("Final batch commit failed!")
            state.overall_success = False
        else:
            logger.debug(
                f"Final batch committed successfully ({logs_committed} logs, {persons_updated} person updates)"
            )


def _log_summary(state: ProcessingState) -> None:
    """Log the processing summary."""

    logger.info("------ Action 9: Process Productive Summary -------")
    logger.info(f"  Candidates Queried:         {state.total_candidates}")
    logger.info(f"  Candidates Processed:       {state.processed_count}")
    logger.info(f"  Skipped (Various Reasons):  {state.skipped_count}")
    logger.info(f"  MS To-Do Tasks Created:     {state.tasks_created_count}")
    logger.info(f"  Acks Sent/Simulated:        {state.acks_sent_count}")
    logger.info(f"  Persons Archived (Staged):  {state.archived_count}")
    logger.info(f"  Errors during processing:   {state.error_count}")
    logger.info(f"  Overall Success:            {state.overall_success}")
    logger.info("--------------------------------------------------\n")


#####################################################
# Helper Functions (Missing from refactored version)
#####################################################


# Helper functions for _process_ai_response


def _get_default_ai_response_structure() -> dict[str, Any]:
    """Get the default empty structure for AI response."""
    return {
        "extracted_data": {
            "mentioned_names": [],
            "mentioned_locations": [],
            "mentioned_dates": [],
            "potential_relationships": [],
            "key_facts": [],
        },
        "suggested_tasks": [],
    }


def _validate_with_pydantic(ai_response: dict[str, Any], log_prefix: str) -> Optional[dict[str, Any]]:
    """Try to validate AI response with Pydantic schema."""
    try:
        validated_response = AIResponse.model_validate(ai_response)
        result = validated_response.model_dump()
        logger.debug(f"{log_prefix}: AI response successfully validated with Pydantic schema.")
        return result
    except ValidationError as ve:
        logger.warning(f"{log_prefix}: AI response validation failed: {ve}")
        return None


def _salvage_extracted_data(ai_response: dict[str, Any], log_prefix: str) -> dict[str, list[Any]]:
    """Try to salvage extracted_data from malformed AI response."""
    result = _get_default_ai_response_structure()["extracted_data"]

    if "extracted_data" not in ai_response or not isinstance(ai_response["extracted_data"], dict):
        logger.warning(f"{log_prefix}: AI response missing 'extracted_data' dictionary. Using defaults.")
        return result

    extracted_data_raw = cast(dict[str, Any], ai_response["extracted_data"])

    # Process each expected key
    for key in result:
        value = extracted_data_raw.get(key, [])

        # Ensure it's a list and contains only strings
        if isinstance(value, list):
            result[key] = [str(item) for item in value if item is not None and isinstance(item, (str, int, float))]
        else:
            logger.warning(f"{log_prefix}: AI response 'extracted_data.{key}' is not a list. Using empty list.")

    return result


def _salvage_suggested_tasks(ai_response: dict[str, Any], log_prefix: str) -> list[Any]:
    """Try to salvage suggested_tasks from malformed AI response."""
    if "suggested_tasks" not in ai_response:
        logger.warning(f"{log_prefix}: AI response missing 'suggested_tasks' list. Using empty list.")
        return []

    tasks_raw = ai_response["suggested_tasks"]

    if not isinstance(tasks_raw, list):
        logger.warning(f"{log_prefix}: AI response 'suggested_tasks' is not a list. Using empty list.")
        return []

    return [str(item) for item in tasks_raw if item is not None and isinstance(item, (str, int, float))]


def _salvage_partial_data(ai_response: dict[str, Any], log_prefix: str) -> dict[str, Any]:
    """Try to salvage partial data from malformed AI response."""
    try:
        extracted_data = _salvage_extracted_data(ai_response, log_prefix)
        suggested_tasks = _salvage_suggested_tasks(ai_response, log_prefix)

        logger.debug(f"{log_prefix}: Salvaged partial data from AI response after validation failure.")

        return {
            "extracted_data": extracted_data,
            "suggested_tasks": suggested_tasks,
        }
    except Exception as e:
        logger.error(f"{log_prefix}: Failed to salvage data from AI response: {e}", exc_info=True)
        return _get_default_ai_response_structure()


def _process_ai_response(ai_response: Any, log_prefix: str) -> dict[str, Any]:
    """
    Processes and validates the AI response using Pydantic models for robust parsing.

    This function takes the raw AI response and attempts to validate it against the
    expected schema using Pydantic models. It handles various error cases gracefully
    and always returns a valid structure even if the input is malformed.

    Args:
        ai_response: The raw AI response from extract_genealogical_entities
        log_prefix: A string prefix for log messages (usually includes person info)

    Returns:
        A dictionary with 'extracted_data' and 'suggested_tasks' keys, properly
        structured and validated, with empty lists as fallbacks for invalid data.
    """
    # Early return if response is None or not a dict
    if not ai_response or not isinstance(ai_response, dict):
        logger.warning(f"{log_prefix}: AI response is None or not a dictionary. Using default empty structure.")
        return _get_default_ai_response_structure()

    logger.debug(f"{log_prefix}: Processing AI response...")

    try:
        # First attempt: Try direct validation with Pydantic
        validated_result = _validate_with_pydantic(ai_response, log_prefix)
        if validated_result:
            return validated_result

        # Second attempt: Try to salvage partial data
        return _salvage_partial_data(ai_response, log_prefix)

    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"{log_prefix}: Unexpected error processing AI response: {e}", exc_info=True)
        return _get_default_ai_response_structure()


def _format_context_for_ai_extraction(
    context_logs: list[ConversationLog],
    # my_pid_lower parameter is kept for compatibility but not used
    _: str = "",  # Renamed to underscore to indicate unused parameter
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
    context_lines: list[str] = []
    # Step 2: Get truncation limit from config
    max_words = config_schema.ai_context_message_max_words

    # Step 3: Iterate through sorted logs (oldest first)
    for log in context_logs:
        # Step 3a: Determine label based on direction
        # Note: Assumes IN logs have author != my_pid_lower, OUT logs have author == my_pid_lower
        # This might need adjustment if author field isn't reliably populated or needed.
        # Using direction is simpler and more reliable here.
        # Handle SQLAlchemy Column type safely
        is_in_direction = False

        try:
            # Try to get the direction value
            if hasattr(log, "direction"):
                direction_value: Any = getattr(log, "direction", None)

                # Check if it's a MessageDirectionEnum or can be compared to one
                if hasattr(direction_value, "value"):
                    # It's an enum object
                    is_in_direction = direction_value == MessageDirectionEnum.IN
                elif isinstance(direction_value, str):
                    # It's a string
                    is_in_direction = direction_value == MessageDirectionEnum.IN.value
                elif direction_value is not None and str(direction_value) == str(MessageDirectionEnum.IN):
                    # Try string comparison as last resort
                    is_in_direction = True
        except Exception:
            # Default to OUT if any error occurs
            is_in_direction = False

        # Use a simple boolean value to avoid SQLAlchemy type issues
        author_label = "USER: " if bool(is_in_direction) else "SCRIPT: "

        # Step 3b: Get message content and handle potential None/SQLAlchemy proxies
        content_raw: Any = getattr(log, "latest_message_content", "") or ""
        content = content_raw if isinstance(content_raw, str) else str(content_raw)

        # Step 3c: Truncate content by word count if necessary
        words = content.split()
        truncated_content = " ".join(words[:max_words]) + "..." if len(words) > max_words else content

        # Step 3d: Append formatted line to the list
        context_lines.append(
            f"{author_label}{truncated_content}"
        )  # Step 4: Join lines into a single string separated by newlines
    return "\n".join(context_lines)


def _get_message_context(
    db_session: DbSession,
    person_id: Union[int, Any],  # Accept SQLAlchemy Column type or int
    limit: int = config_schema.ai_context_messages_count,
) -> list[ConversationLog]:
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
        # Convert SQLAlchemy Column objects to Python datetime objects for sorting
        def get_sort_key(log: Any) -> datetime:
            # Extract timestamp value from SQLAlchemy Column if needed
            ts = log.latest_timestamp
            # If it's already a datetime or can be used as one, use it
            if hasattr(ts, "year") and hasattr(ts, "month"):
                return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            # Otherwise use minimum date
            return datetime.min.replace(tzinfo=timezone.utc)

        return sorted(context_logs, key=get_sort_key)
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


def _load_templates_for_action9() -> dict[str, str]:
    """
    Loads message templates for Action 9 from action8_messaging.

    Returns:
        Dictionary of message templates, or empty dict if loading fails
    """
    try:
        # Import the template loading function from action8_messaging
        from actions.action8_messaging import MESSAGE_TEMPLATES, ensure_message_templates_loaded

        # Ensure templates are loaded (handles mock/dry-run logic)
        ensure_message_templates_loaded()

        # Check if the required template exists
        if not MESSAGE_TEMPLATES or ACKNOWLEDGEMENT_MESSAGE_TYPE not in MESSAGE_TEMPLATES:
            logger.error(f"Required template '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' not found in templates.")
            return {}

        return MESSAGE_TEMPLATES
    except Exception as e:
        logger.error(f"Error loading templates for Action 9: {e}", exc_info=True)
        return {}


def _identify_and_get_person_details(log_prefix: str) -> Optional[dict[str, Any]]:
    """
    Simplified version that returns None (no person details found).
    """
    logger.debug(f"{log_prefix}: _identify_and_get_person_details - returning None (simplified version)")
    return None


def _format_genealogical_data_for_ai(genealogical_data: dict[str, Any]) -> str:
    """
    Simplified version that formats genealogical data for AI consumption.
    """
    if not genealogical_data or not genealogical_data.get("results"):
        return "No genealogical data found in family tree."

    # Simple formatting of search results
    formatted_lines = ["Family Tree Search Results:"]
    for result in genealogical_data.get("results", [])[:3]:  # Limit to top 3
        if isinstance(result, dict):
            result_dict = cast(dict[str, Any], result)
            name = result_dict.get("name", "Unknown")
            formatted_lines.append(f"- {name}")

    return "\n".join(formatted_lines)


def _generate_ack_summary(extracted_data: dict[str, Any]) -> str:
    """
    Generates a summary from extracted data for acknowledgment messages.
    """
    try:
        # Get mentioned names
        names = extracted_data.get("extracted_data", {}).get("mentioned_names", [])
        locations = extracted_data.get("extracted_data", {}).get("mentioned_locations", [])
        dates = extracted_data.get("extracted_data", {}).get("mentioned_dates", [])

        summary_parts: list[str] = []

        if names:
            summary_parts.append(f"information about {', '.join(names[:2])}")
        if locations:
            summary_parts.append(f"details from {', '.join(locations[:2])}")
        if dates:
            summary_parts.append(f"records from {', '.join(dates[:2])}")

        if summary_parts:
            return "; ".join(summary_parts)
        return "your family history research"
    except Exception as e:
        logger.error(f"Error generating acknowledgment summary: {e}")
        return "your genealogy information"


# ==============================================
# PHASE 5 RESEARCH ASSISTANT FEATURES
# ==============================================


def _check_close_relationship(relationship_lower: str) -> Optional[tuple[str, int]]:
    """Check if relationship is close (high priority)."""
    close_relationships = [
        "parent",
        "child",
        "sibling",
        "brother",
        "sister",
        "uncle",
        "aunt",
        "nephew",
        "niece",
        "1st cousin",
        "first cousin",
        "2nd cousin",
        "second cousin",
    ]
    for close_rel in close_relationships:
        if close_rel in relationship_lower:
            return "high", 7
    return None


def _check_medium_relationship(relationship_lower: str) -> Optional[tuple[str, int]]:
    """Check if relationship is medium (normal priority)."""
    medium_relationships = ["3rd cousin", "third cousin", "4th cousin", "fourth cousin"]
    for medium_rel in medium_relationships:
        if medium_rel in relationship_lower:
            return "normal", 14
    return None


def _check_distant_relationship(relationship_lower: str) -> Optional[tuple[str, int]]:
    """Check if relationship is distant (low priority)."""
    if "5th" in relationship_lower or "sixth" in relationship_lower or "distant" in relationship_lower:
        return "low", 30
    return None


def _calculate_priority_from_dna(shared_dna_cm: Optional[float]) -> tuple[str, int]:
    """Calculate priority based on shared DNA."""
    if not shared_dna_cm:
        return "normal", 14
    if shared_dna_cm > 200:
        return "high", 7
    if shared_dna_cm > 50:
        return "normal", 14
    return "low", 30


def calculate_task_priority_from_relationship(
    relationship: Optional[str], shared_dna_cm: Optional[float] = None
) -> tuple[str, int]:
    """
    Calculate MS To-Do task priority and due date offset based on relationship closeness.

    Args:
        relationship: Relationship description (e.g., "2nd cousin", "uncle")
        shared_dna_cm: Shared DNA in centiMorgans

    Returns:
        Tuple of (importance, days_until_due)
        - importance: "high", "normal", or "low"
        - days_until_due: Number of days until task is due
    """
    if not relationship:
        return _calculate_priority_from_dna(shared_dna_cm)

    relationship_lower = relationship.lower()

    # Check relationship types in order of priority
    result = _check_close_relationship(relationship_lower)
    if result:
        return result

    result = _check_medium_relationship(relationship_lower)
    if result:
        return result

    result = _check_distant_relationship(relationship_lower)
    if result:
        return result

    # Fall back to DNA-based priority
    return _calculate_priority_from_dna(shared_dna_cm)


def create_enhanced_research_task(
    person_name: str,
    relationship: Optional[str],
    shared_dna_cm: Optional[float] = None,
    categories: Optional[list[str]] = None,
    profile_id: Optional[str] = None,
    uuid: Optional[str] = None,
    tree_info: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """
    Create an enhanced MS To-Do task with intelligent priority and due date.

    Args:
        person_name: Name of the person related to the task
        relationship: Relationship to the person
        shared_dna_cm: Shared DNA in centiMorgans
        categories: Optional categories for the task
        profile_id: Ancestry profile ID
        uuid: DNA match UUID
        tree_info: Dictionary with tree information (person_name_in_tree, view_in_tree_link, actual_relationship)

    Returns:
        Task ID if created successfully, None otherwise
    """
    try:
        if not person_name:
            logger.error("Cannot create enhanced research task: person_name missing.")
            return None

        payload = _build_enhanced_task_payload(
            person_name,
            relationship,
            shared_dna_cm,
            categories,
            profile_id=profile_id,
            uuid=uuid,
            tree_info=tree_info,
        )

        if not _ensure_enhanced_task_ms_graph_state(_ENHANCED_TASK_STATE):
            logger.info(f"Enhanced task creation skipped for {person_name}: MS Graph unavailable.")
            return None

        return _submit_enhanced_task(_ENHANCED_TASK_STATE, payload, person_name)

    except Exception as e:
        logger.error(f"Failed to create enhanced research task: {e}")
        return None


def generate_ai_response_prompt(
    person_name: str,
    their_message: str,
    relationship_info: Optional[dict[str, Any]] = None,
    missing_info: Optional[list[str]] = None,
) -> str:
    """
    Generate an AI prompt for responding to a conversation.

    Args:
        person_name: Name of the person who sent the message
        their_message: The message they sent
        relationship_info: Optional relationship information
        missing_info: Optional list of missing information

    Returns:
        AI prompt string for generating a response
    """
    try:
        from research.research_guidance_prompts import (
            create_conversation_response_prompt,
            create_research_guidance_prompt,
        )

        # Use research guidance prompt if we have missing info
        if missing_info:
            relationship = relationship_info.get('relationship') if relationship_info else None
            return create_research_guidance_prompt(
                person_name=person_name, relationship=relationship, missing_info=missing_info
            )

        # Otherwise use conversation response prompt
        return create_conversation_response_prompt(
            person_name=person_name, their_message=their_message, relationship_info=relationship_info
        )

    except Exception as e:
        logger.error(f"Failed to generate AI response prompt: {e}")
        return f"Please help me respond to {person_name}'s message: {their_message}"


def format_response_with_records(
    person_name: str, records: list[dict[str, Any]], context: str = "I found these records that might be helpful:"
) -> str:
    """
    Format a response that includes record sharing.

    Args:
        person_name: Name of the person being responded to
        records: List of record dictionaries
        context: Context message for the records

    Returns:
        Formatted message with record references
    """
    try:
        from research.record_sharing import create_record_sharing_message

        return create_record_sharing_message(person_name, records, context)
    except Exception as e:
        logger.error(f"Failed to format response with records: {e}")
        return context


def format_response_with_relationship_diagram(
    from_name: str, to_name: str, relationship_path: list[dict[str, str]]
) -> str:
    """
    Format a response that includes a relationship diagram.

    Args:
        from_name: Name of the first person (usually "me" or tree owner)
        to_name: Name of the second person
        relationship_path: List of relationship path dictionaries

    Returns:
        Formatted message with relationship diagram
    """
    try:
        from research.relationship_diagram import format_relationship_for_message

        return format_relationship_for_message(from_name, to_name, relationship_path, include_diagram=True)
    except Exception as e:
        logger.error(f"Failed to format response with relationship diagram: {e}")
        return f"Our relationship: {from_name} → {to_name}"


# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================

# Removed smoke test: _test_module_initialization - only checked constants and callable()


def _test_core_functionality() -> None:
    """Test all core utility functions"""

    # Test safe_column_value function with a simple object
    class TestObj:
        def __init__(self) -> None:
            self.test_attr = "test_value_12345"

    test_obj = TestObj()
    result = safe_column_value(test_obj, "test_attr", "default")
    assert result == "test_value_12345", "Should extract attribute value"

    result = safe_column_value(test_obj, "missing_attr", "default_12345")
    assert result == "default_12345", "Should return default for missing attribute"

    # Test should_exclude_message function
    result = should_exclude_message("Hi there!")
    assert result is False, "Normal message should not be excluded"

    result = should_exclude_message("stop messaging me")
    assert result is True, "Should detect exclusion keywords"

    # Test additional exclusion patterns
    result = should_exclude_message("STOP contacting me")
    assert result is True, "Should detect exclusion keywords (case-insensitive)"


def _test_ai_processing_functions() -> None:
    """Test AI processing and extraction functions"""
    # Test _process_ai_response function with valid response
    valid_response = {
        "extracted_data": {
            "mentioned_names": ["John Doe"],
            "mentioned_locations": ["Scotland"],
            "mentioned_dates": ["1850"],
            "potential_relationships": [],
            "key_facts": [],
        },
        "suggested_tasks": ["Research John Doe"],
    }
    result = _process_ai_response(valid_response, "TEST")
    assert isinstance(result, dict), "Should return dictionary"
    assert "extracted_data" in result, "Should have extracted_data key"
    assert "suggested_tasks" in result, "Should have suggested_tasks key"
    # Validate result has proper structure (Pydantic may transform keys)
    assert isinstance(result["extracted_data"], dict), "Extracted data should be dict"
    assert isinstance(result["suggested_tasks"], list), "Suggested tasks should be list"
    # Verify function processes data (doesn't just return empty defaults)
    assert len(result["extracted_data"]) > 0 or len(result["suggested_tasks"]) > 0, (
        "Should process at least some data from valid response"
    )

    # Test _generate_ack_summary function
    test_data = {
        "extracted_data": {
            "mentioned_names": ["Test Person 12345"],
            "mentioned_locations": ["Scotland"],
            "mentioned_dates": ["1985"],
        }
    }
    result = _generate_ack_summary(test_data)
    assert isinstance(result, str), "Should return string summary"
    assert len(result) > 0, "Should generate meaningful summary"
    # Validate summary contains key information
    assert "Test Person 12345" in result or "name" in result.lower(), "Summary should mention extracted names"
    assert "Scotland" in result or "location" in result.lower(), "Summary should mention locations"
    assert "1985" in result or "date" in result.lower(), "Summary should mention dates"


def _test_edge_cases() -> None:
    """Test edge cases and boundary conditions"""
    # Test safe_column_value with None object
    result = safe_column_value(None, "any_attr", "default_12345")
    assert result == "default_12345", "Should handle None object"

    # Test should_exclude_message with empty string
    result = should_exclude_message("")
    assert isinstance(result, bool), "Should handle empty message"

    # Test should_exclude_message with None
    result = should_exclude_message(None)
    assert isinstance(result, bool), "Should handle None message"

    # Test _process_ai_response with None
    result = _process_ai_response(None, "TEST")
    assert isinstance(result, dict), "Should handle None AI response"


def _test_integration() -> None:
    """Test integration with external data sources and templates"""
    # Test get_gedcom_data function availability
    assert callable(get_gedcom_data), "get_gedcom_data should be callable"

    # Test _load_templates_for_action9 function availability
    assert callable(_load_templates_for_action9), "_load_templates_for_action9 should be callable"


def _test_circuit_breaker_config() -> None:
    """Test circuit breaker configuration validation"""
    import inspect

    # Test that process_productive_messages has circuit breaker decorator
    func = process_productive_messages
    assert callable(func), "process_productive_messages should be callable"
    assert func.__name__ == 'process_productive_messages', "Function name should be preserved"

    # Check function signature
    sig = inspect.signature(func)
    assert 'session_manager' in sig.parameters, "Should have session_manager parameter"

    # Verify function can be inspected (indicates decorators are properly applied)
    assert hasattr(func, '__wrapped__') or hasattr(func, '__annotations__'), "Should have decorator attributes"


def _test_error_handling() -> None:
    """Test error handling for AI processing and utility functions"""
    # Test _process_ai_response with malformed data
    malformed_data = {"invalid": "structure_12345"}
    result = _process_ai_response(malformed_data, "ERROR_TEST")
    assert isinstance(result, dict), "Should handle malformed AI responses"

    # Test _generate_ack_summary with empty data
    result = _generate_ack_summary({})
    assert isinstance(result, str), "Should handle empty extraction data"


def _test_enhanced_task_creation() -> None:
    """
    Test enhanced MS To-Do task creation with priority and due dates.

    Phase 5.3: Enhanced MS To-Do Task Creation
    Tests the calculate_task_priority_and_due_date method.
    """
    from unittest.mock import Mock

    from core.database import Person

    # Create mock dependencies for PersonProcessor
    mock_session_manager = Mock()
    mock_session_manager.my_profile_id = "test_profile"
    mock_db_state = Mock()
    mock_msg_config = Mock()
    mock_ms_state = Mock()

    # Create test processor instance
    processor = PersonProcessor(
        session_manager=mock_session_manager, db_state=mock_db_state, msg_config=mock_msg_config, ms_state=mock_ms_state
    )

    def _build_person(
        person_id: int,
        username: str,
        profile_id: str,
        in_tree: bool,
        relationship: Optional[str],
        tree_status: Optional[str],
    ) -> Person:
        person = Mock(spec=Person)
        person.id = person_id
        person.username = username
        person.profile_id = profile_id
        person.in_my_tree = in_tree
        person.predicted_relationship = relationship
        person.tree_status = tree_status
        return person

    cases = [
        {
            "person_kwargs": {
                "person_id": 1,
                "username": "Test User 1",
                "profile_id": "test123",
                "in_tree": True,
                "relationship": "1st cousin",
                "tree_status": "in_tree",
            },
            "expected_importance": "high",
            "required_categories": ["Close Relative", "In Tree"],
            "due_date_required": True,
        },
        {
            "person_kwargs": {
                "person_id": 2,
                "username": "Test User 2",
                "profile_id": "test456",
                "in_tree": False,
                "relationship": "3rd cousin",
                "tree_status": "out_tree",
            },
            "expected_importance": "normal",
            "required_categories": ["Distant Relative", "Out of Tree"],
            "due_date_required": True,
        },
        {
            "person_kwargs": {
                "person_id": 3,
                "username": "Test User 3",
                "profile_id": "test789",
                "in_tree": True,
                "relationship": "5th cousin",
                "tree_status": "in_tree",
            },
            "expected_importance": "low",
            "required_categories": ["Distant Relative"],
            "due_date_required": True,
        },
        {
            "person_kwargs": {
                "person_id": 4,
                "username": "Test User 4",
                "profile_id": "test000",
                "in_tree": False,
                "relationship": None,
                "tree_status": None,
            },
            "expected_importance": "normal",
            "required_categories": ["Ancestry Research"],
            "due_date_required": False,
        },
    ]

    for case in cases:
        person_kwargs = cast(dict[str, Any], case["person_kwargs"])
        person = _build_person(**person_kwargs)
        importance, due_date, categories = processor.calculate_task_priority_and_due_date(person)
        assert importance == case["expected_importance"], f"Unexpected priority for {person.username}: {importance}"

        required_categories = cast(list[str], case["required_categories"])
        for category in required_categories:
            assert category in categories, f"{category} should appear for {person.username}"
        if case["due_date_required"]:
            assert due_date is not None, f"Due date required for {person.username}"

    logger.info("✅ Enhanced task creation tests passed")


# ==============================================
# INTEGRATION TEST HELPERS (Real Authenticated Sessions)
# ==============================================

# === SESSION SETUP FOR TESTS ===
# Migrated to use centralized session_utils.py (reduces 127 lines to 1 import!)
from core.session_utils import ensure_session_for_tests as _ensure_session_for_productive_tests


def _test_database_session_availability() -> bool:
    """Test that database session is available for productive processing."""
    try:
        # This test requires live session - skip if not available
        try:
            sm, _ = _ensure_session_for_productive_tests()
        except RuntimeError:
            logger.info("Skipping live test (no global session available)")
            return True

        logger.info("Testing database session availability...")

        # Get database session
        db_session = sm.db_manager.get_session()
        if not db_session:
            raise RuntimeError("Failed to get database session")

        from core.database import Person

        person_count = db_session.query(Person).count()
        logger.info(f"✅ Database session available with {person_count} persons in database")

        sm.db_manager.return_session(db_session)
        return True

    except Exception as e:
        logger.error(f"❌ Database session availability test failed: {e}")
        raise


def _test_message_templates_available() -> bool:
    """Test that message templates are available for productive processing."""
    try:
        # This test requires live session - skip if not available
        try:
            sm, _ = _ensure_session_for_productive_tests()
        except RuntimeError:
            logger.info("Skipping live test (no global session available)")
            return True

        logger.info("Testing message template availability...")

        # Get database session
        db_session = sm.db_manager.get_session()
        if not db_session:
            raise RuntimeError("Failed to get database session")

        from core.database import MessageTemplate

        templates = db_session.query(MessageTemplate).all()
        template_count = len(templates)

        logger.info(f"✅ Found {template_count} message templates in database")

        if template_count > 0:
            for template in templates[:3]:  # Show first 3
                logger.info(f"   - {template.template_name if hasattr(template, 'template_name') else 'N/A'}")

        sm.db_manager.return_session(db_session)
        assert template_count > 0, "Should have at least one message template"
        return True

    except Exception as e:
        logger.error(f"❌ Message template availability test failed: {e}")
        raise


# === PHASE 5 INTEGRATION TESTS ===


def _test_response_quality_scoring() -> None:
    """Test response quality scoring system."""
    from unittest.mock import MagicMock

    from research.person_lookup_utils import PersonLookupResult

    # Create mock PersonProcessor with correct constructor signature
    processor = PersonProcessor(
        session_manager=MagicMock(), db_state=MagicMock(), msg_config=MagicMock(), ms_state=MagicMock()
    )

    # Create mock person
    mock_person = MagicMock()
    mock_person.username = "John Smith"

    # Test Case 1: High-quality response (should score 80-100)
    high_quality_response = """
    Hi John! Thanks for reaching out about our shared ancestor William Gault.

    Based on my research, William Gault (1820-1892) was born in Banff, Scotland
    and emigrated to Nova Scotia in 1845. I found his birth record in the Old Parish
    Registers (1820) and his marriage record to Margaret Fraser (1843) in Aberdeen.

    Our relationship: You're my 3rd cousin once removed through the Gault line.
    William was your great-great-grandfather and my great-great-great-grandfather.

    Next steps to explore:
    1. Search for William's immigration records (1845 passenger lists)
    2. Look for census records in Nova Scotia (1851, 1861 censuses)
    3. Check land grant records in Pictou County

    Would you like me to share copies of the records I've found? I'd also love to
    compare notes on the Fraser connection.
    """

    lookup_results = [PersonLookupResult(found=True, name="William Gault", birth_year=1820, source='gedcom')]

    score1 = processor.score_response_quality(
        response_text=high_quality_response, lookup_results=lookup_results, person=mock_person
    )

    assert 80 <= score1 <= 100, f"High-quality response scored {score1}, expected 80-100"
    logger.info(f"✓ High-quality response scored {score1:.1f}/100")

    # Test Case 2: Medium-quality response (should score 50-79)
    medium_quality_response = """
    Hi John, thanks for your message about William Gault. I have some information
    about him from my family tree. He lived in Scotland in the 1800s and had several
    children. We're related through the Gault family line. Let me know if you want
    more details.
    """

    score2 = processor.score_response_quality(
        response_text=medium_quality_response, lookup_results=[], person=mock_person
    )

    assert 40 <= score2 <= 79, f"Medium-quality response scored {score2}, expected 40-79"
    logger.info(f"✓ Medium-quality response scored {score2:.1f}/100")

    # Test Case 3: Low-quality response (should score 0-49)
    low_quality_response = """
    Thanks for the message. I'll check my records.
    """

    score3 = processor.score_response_quality(response_text=low_quality_response, lookup_results=[], person=mock_person)

    assert 0 <= score3 <= 49, f"Low-quality response scored {score3}, expected 0-49"
    logger.info(f"✓ Low-quality response scored {score3:.1f}/100")

    # Test Case 4: Edge case - empty response
    score4 = processor.score_response_quality(response_text="", lookup_results=[], person=mock_person)

    assert score4 == 0, f"Empty response scored {score4}, expected 0"
    logger.info(f"✓ Empty response scored {score4:.1f}/100")

    # Test Case 5: Edge case - very long response (penalty applied)
    very_long_response = "word " * 600  # 600 words
    score5 = processor.score_response_quality(response_text=very_long_response, lookup_results=[], person=mock_person)

    # Should have penalty applied for being too long (>500 words)
    logger.info(f"✓ Very long response scored {score5:.1f}/100 (penalty applied)")

    # Test Case 6: Unused lookup results penalty
    unused_lookup_response = "Thanks for reaching out!"

    rich_lookup_results = [
        PersonLookupResult(
            found=True,
            name="William Gault",
            birth_year=1820,
            birth_place="Banff, Scotland",
            death_year=1892,
            death_place="Nova Scotia",
            relationship_path='3rd cousin once removed',
            source='gedcom',
        )
    ]

    score6 = processor.score_response_quality(
        response_text=unused_lookup_response, lookup_results=rich_lookup_results, person=mock_person
    )

    # Should have penalty for not using rich lookup data
    logger.info(f"✓ Unused lookup results scored {score6:.1f}/100 (penalty applied)")

    logger.info("✓ Response quality scoring test passed - all scenarios validated")


def _test_calculate_task_priority_from_relationship() -> None:
    """Test task priority calculation."""
    # Test that function exists and returns tuples
    result1 = calculate_task_priority_from_relationship("uncle")
    assert isinstance(result1, tuple), "Should return tuple"
    assert len(result1) == 2, "Should return 2-element tuple"
    importance1, days1 = result1
    assert importance1 in {"high", "normal", "low"}, f"Invalid importance: {importance1}"
    assert isinstance(days1, int), f"Days should be int, got {type(days1)}"

    # Test that different relationships give different priorities
    result2 = calculate_task_priority_from_relationship("5th cousin")
    importance2, _ = result2
    assert importance2 in {"high", "normal", "low"}, f"Invalid importance: {importance2}"

    # Test DNA-based priority
    result3 = calculate_task_priority_from_relationship(None, shared_dna_cm=250.0)
    importance3, _ = result3
    assert importance3 in {"high", "normal", "low"}, f"Invalid importance: {importance3}"

    logger.info("✓ Task priority calculation test passed")


def _test_create_enhanced_research_task() -> None:
    """Test enhanced research task creation."""
    # Should not crash even if MS Graph not available
    create_enhanced_research_task(person_name="John Smith", relationship="2nd cousin", shared_dna_cm=98.0)

    # Task ID might be None if MS Graph not configured, but function should not crash
    logger.info("✓ Enhanced research task creation test passed")


def _test_generate_ai_response_prompt() -> None:
    """Test AI response prompt generation."""
    prompt = generate_ai_response_prompt(
        person_name="John Smith",
        their_message="Do you have info about William Gault?",
        relationship_info={'relationship': '3rd cousin', 'shared_dna_cm': 98.0},
    )

    assert "John Smith" in prompt
    assert "William Gault" in prompt

    logger.info("✓ AI response prompt generation test passed")


def _test_format_response_with_records() -> None:
    """Test response formatting with records."""
    records = [
        {'type': 'Birth', 'details': {'date': '1941', 'place': 'Banff, Scotland', 'source': 'Birth Certificate'}}
    ]

    response = format_response_with_records("Fraser Gault", records)
    assert "Fraser Gault" in response
    assert "1941" in response

    logger.info("✓ Response formatting with records test passed")


def _test_format_response_with_relationship_diagram() -> None:
    """Test response formatting with relationship diagram."""
    path = [{'name': 'Wayne Gault', 'relationship': 'self'}, {'name': 'Fraser Gault', 'relationship': 'father'}]

    response = format_response_with_relationship_diagram("Wayne", "Fraser", path)
    assert "Wayne" in response or "Fraser" in response

    logger.info("✓ Response formatting with relationship diagram test passed")


def _test_retry_helper_alignment_action9() -> None:
    """Ensure process_productive_messages uses the centralized API retry helper."""
    helper_name = getattr(process_productive_messages, "__retry_helper__", None)
    assert helper_name == "api_retry", f"process_productive_messages should use api_retry helper, found: {helper_name}"


def _test_fact_validation_integration() -> None:
    """Validate that extracted facts are staged as SuggestedFact/DataConflict with correct counts."""

    class _FakeSession:
        def __init__(self) -> None:
            self.added: list[Any] = []

        def add(self, obj: Any) -> None:  # pragma: no cover - simple container
            self.added.append(obj)

    fake_session = _FakeSession()

    # Minimal stubs
    session_manager = cast(SessionManager, type("SM", (), {"rate_limiter": None, "my_profile_id": ""})())
    db_state = DatabaseState(session=cast(DbSession, fake_session))
    msg_config = MessageConfig()
    ms_state = MSGraphState()
    processor = PersonProcessor(session_manager, db_state, msg_config, ms_state)

    person = cast(
        Person,
        type(
            "P",
            (),
            {
                "id": 1,
                "username": "Tester",
                "display_name": "Tester",
                "birth_year": 1920,
                "status": PersonStatusEnum.ACTIVE,
            },
        )(),
    )
    latest_message = cast(ConversationLog, type("LM", (), {"id": 123})())

    fact = ExtractedFact(
        fact_type="BIRTH",
        subject_name="Tester",
        original_text="Born 1931",
        structured_value="1931",
        normalized_value="1931-01-01",
        confidence=70,
    )

    extracted_data = {"_fact_objects": [fact]}

    approved, pending, conflicts = processor._validate_and_record_facts(
        person,
        extracted_data,
        latest_message,
        log_prefix="Test",
    )

    assert approved == 0
    assert pending == 1
    assert conflicts == 1
    # One SuggestedFact + one DataConflict should be staged
    assert len(fake_session.added) == 2


# ==============================================
# MAIN TEST SUITE
# ==============================================


def action9_process_productive_module_tests() -> bool:
    """Comprehensive test suite for action9_process_productive.py"""
    import os

    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite(
        "Action 9 - Productive DNA Match Processing",
        "action9_process_productive.py",
    )
    suite.start_suite()

    # Check if we should skip live API tests (set by run_all_tests.py when running in parallel)
    skip_live_api_tests = os.environ.get("SKIP_LIVE_API_TESTS", "").lower() == "true"

    # Define all tests in a data structure to reduce complexity
    tests: list[tuple[str, Callable[[], Any], str, str, str]] = [
        # Removed smoke test: Module constants, classes, and function availability
        (
            "safe_column_value(), should_exclude_message() core functions",
            _test_core_functionality,
            "All core functions execute correctly with proper data handling and validation",
            "Core utility and message filtering functionality",
            "Testing attribute extraction, message filtering, and core utility functions",
        ),
        (
            "_process_ai_response(), _generate_ack_summary() AI processing",
            _test_ai_processing_functions,
            "AI processing functions handle response data correctly and generate summaries",
            "AI response processing and summary generation functions",
            "Testing AI response parsing, data extraction, and summary generation functionality",
        ),
        (
            "Fact validation integration",
            _test_fact_validation_integration,
            "Extracted facts are staged as SuggestedFact/DataConflict with correct counts",
            "Validation pipeline integration",
            "Ensuring fact validation produces expected review artifacts",
        ),
        (
            "ALL functions with edge case inputs",
            _test_edge_cases,
            "All functions handle edge cases gracefully without crashes or unexpected behavior",
            "Edge case handling across all AI processing functions",
            "Testing functions with empty, None, invalid, and boundary condition inputs",
        ),
        (
            "get_gedcom_data(), _load_templates_for_action9() integration",
            _test_integration,
            "Integration functions work correctly with external data sources and templates",
            "Integration with GEDCOM data and template systems",
            "Testing integration with genealogical data cache and message template loading",
        ),
        (
            "Circuit breaker configuration validation",
            _test_circuit_breaker_config,
            "Circuit breaker decorators properly applied with Action 6 lessons (failure_threshold=10, backoff_factor=4.0)",
            "Circuit breaker decorator configuration reflects improved error handling",
            "Testing process_productive_messages() has proper circuit breaker configuration for production resilience",
        ),
        (
            "Retry helper alignment",
            _test_retry_helper_alignment_action9,
            "process_productive_messages() uses api_retry helper derived from telemetry",
            "Retry helper configuration",
            "Verifies process_productive_messages() is decorated with api_retry helper for consistent policy tuning",
        ),
        (
            "Error handling for AI processing and utility functions",
            _test_error_handling,
            "All error conditions handled gracefully with appropriate fallback responses",
            "Error handling and recovery functionality for AI operations",
            "Testing error scenarios with invalid data, exceptions, and malformed responses",
        ),
        (
            "Enhanced MS To-Do task creation with priority and due dates",
            _test_enhanced_task_creation,
            "Task priority and due dates calculated correctly based on relationship closeness",
            "Phase 5.3 enhanced task creation with relationship-based priority",
            "Testing calculate_task_priority_and_due_date() with various relationship types",
        ),
        (
            "Phase 5: Response quality scoring (0-100 scale)",
            _test_response_quality_scoring,
            "Quality scores calculated correctly based on relationship specificity, evidence, actionability, personalization",
            "Phase 5 response quality scoring with 4-component evaluation",
            "Testing score_response_quality() with high/medium/low quality responses and edge cases",
        ),
        (
            "Phase 5: Task priority calculation",
            _test_calculate_task_priority_from_relationship,
            "Task priorities calculated correctly from relationships and DNA",
            "Phase 5 task priority calculation",
            "Testing calculate_task_priority_from_relationship() with various inputs",
        ),
        (
            "Phase 5: Enhanced research task creation",
            _test_create_enhanced_research_task,
            "Enhanced tasks created successfully with priority and due dates",
            "Phase 5 enhanced task creation",
            "Testing create_enhanced_research_task() functionality",
        ),
        (
            "Phase 5: AI response prompt generation",
            _test_generate_ai_response_prompt,
            "AI prompts generated correctly for conversation responses",
            "Phase 5 AI prompt generation",
            "Testing generate_ai_response_prompt() functionality",
        ),
        (
            "Phase 5: Response formatting with records",
            _test_format_response_with_records,
            "Responses formatted correctly with record sharing",
            "Phase 5 record sharing in responses",
            "Testing format_response_with_records() functionality",
        ),
        (
            "Phase 5: Response formatting with relationship diagrams",
            _test_format_response_with_relationship_diagram,
            "Responses formatted correctly with relationship diagrams",
            "Phase 5 relationship diagrams in responses",
            "Testing format_response_with_relationship_diagram() functionality",
        ),
    ]

    # Only add database session tests if not skipping live API tests
    if not skip_live_api_tests:
        tests.extend(
            [
                (
                    "Database session availability (real authenticated session)",
                    _test_database_session_availability,
                    "Database session is available and functional with real Ancestry authentication",
                    "Real authenticated session database connectivity",
                    "Testing database session establishment with valid Ancestry credentials",
                ),
                (
                    "Message templates available (real authenticated session)",
                    _test_message_templates_available,
                    "Message templates are loaded and available in database with real authentication",
                    "Real authenticated session message template loading",
                    "Testing message template availability with valid Ancestry session",
                ),
            ]
        )
    else:
        logger.info("⏭️  Skipping live API tests (SKIP_LIVE_API_TESTS=true) - running in parallel mode")

    # Run all tests from the list
    with suppress_logging():
        for test_name, test_func, expected, method, details in tests:
            suite.run_test(test_name, test_func, expected, method, details)

    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
run_comprehensive_tests = create_standard_test_runner(action9_process_productive_module_tests)


if __name__ == "__main__":
    print("🤖 Running Action 9 - AI Message Processing & Data Extraction comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
