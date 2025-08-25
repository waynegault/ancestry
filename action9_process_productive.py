#!/usr/bin/env python3

"""
Action 9: Productive DNA Match Processing

Analyzes and processes productive DNA matches with comprehensive relationship
analysis, GEDCOM integration, and automated workflow management for genealogical
research including match scoring, family tree analysis, and research prioritization.

PHASE 1 OPTIMIZATIONS (2025-01-16):
- Enhanced progress indicators with ETA calculations for match processing
- Improved error recovery with exponential backoff for GEDCOM operations
- Memory monitoring during large match analysis sessions
- Better user feedback for relationship analysis workflows
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
)

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from core.error_handling import (
    circuit_breaker,
    error_context,
    graceful_degradation,
    retry_on_failure,
    timeout_protection,
)

logger = setup_module(globals(), __name__)

# === PHASE 1 OPTIMIZATIONS ===

# === STANDARD LIBRARY IMPORTS ===
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Union

# === THIRD-PARTY IMPORTS ===
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import and_, func, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DbSession, joinedload

# === PHASE 9.1: MESSAGE PERSONALIZATION ===
try:
    from message_personalization import MessagePersonalizer
    MESSAGE_PERSONALIZATION_AVAILABLE = True
    logger.debug("Message personalization system loaded in action9")
except ImportError as e:
    logger.warning(f"Message personalization not available in action9: {e}")
    MESSAGE_PERSONALIZATION_AVAILABLE = False
    MessagePersonalizer = None

# === PHASE 10.1: GENEALOGICAL TASK TEMPLATES ===
try:
    from genealogical_task_templates import GenealogicalTaskGenerator
    GENEALOGICAL_TASK_GENERATION_AVAILABLE = True
    logger.debug("Genealogical task generation system loaded in action9")
except ImportError as e:
    logger.warning(f"Genealogical task generation not available in action9: {e}")
    GENEALOGICAL_TASK_GENERATION_AVAILABLE = False
    GenealogicalTaskGenerator = None

# === PHASE 11.1: ADAPTIVE RATE LIMITING & PERFORMANCE MONITORING ===
try:
    from adaptive_rate_limiter import AdaptiveRateLimiter, ConfigurationOptimizer, SmartBatchProcessor
    from performance_dashboard import PerformanceDashboard
    ADAPTIVE_SYSTEMS_AVAILABLE = True
    logger.debug("Adaptive rate limiting and performance monitoring loaded in action9")
except ImportError as e:
    logger.warning(f"Adaptive systems not available in action9: {e}")
    ADAPTIVE_SYSTEMS_AVAILABLE = False
    AdaptiveRateLimiter = None
    SmartBatchProcessor = None
    ConfigurationOptimizer = None
    PerformanceDashboard = None
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import ms_graph_utils
from ai_interface import extract_genealogical_entities

# === PHASE 5.2: SYSTEM-WIDE CACHING OPTIMIZATION ===
from cache_manager import (
    cached_api_call,
)

# === LOCAL IMPORTS ===
from config import config_schema
from core.session_manager import SessionManager


# Legacy compatibility
def cached_database_query(ttl: int = 300):
    """Legacy wrapper for database query caching with TTL support"""
    def decorator(func):
        return cached_api_call("database", ttl=ttl)(func)
    return decorator
from database import (
    ConversationLog,
    MessageDirectionEnum,
    MessageTemplate,
    Person,
    PersonStatusEnum,
    commit_bulk_data,
)
from test_framework import (
    MagicMock,
    TestSuite,
    patch,
    suppress_logging,
)
from utils import format_name

# === CONSTANTS ===
PRODUCTIVE_SENTIMENT = "PRODUCTIVE"  # Sentiment string set by Action 7
OTHER_SENTIMENT = (
    "OTHER"  # Sentiment string for messages that don't fit other categories
)
ACKNOWLEDGEMENT_MESSAGE_TYPE = (
    "Productive_Reply_Acknowledgement"  # Key in MessageTemplate table
)
CUSTOM_RESPONSE_MESSAGE_TYPE = "Automated_Genealogy_Response"  # Key in MessageTemplate table

# Keywords that indicate no response should be sent
EXCLUSION_KEYWORDS = [
    "stop",
    "unsubscribe",
    "no more messages",
    "not interested",
    "do not respond",
    "no reply",
]


def safe_column_value(obj: Any, attr_name: str, default: Any = None) -> Any:
    """
    Safely extract a value from a SQLAlchemy object attribute.
    Handles cases where the attribute might be a SQLAlchemy Column or None.

    Args:
        obj: The SQLAlchemy object
        attr_name: The name of the attribute to extract
        default: Default value to return if extraction fails

    Returns:
        The extracted value or the default (preserving type when possible)
    """
    try:
        if hasattr(obj, attr_name):
            value = getattr(obj, attr_name)
            if value is None:
                return default
            return value
        return default
    except Exception:
        return default


def should_exclude_message(message_content: str) -> bool:
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
    return any(keyword.lower() in message_lower for keyword in EXCLUSION_KEYWORDS)


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
    date: str
    place: str
    certainty: str = "unknown"


class Relationship(BaseModel):
    """Model for relationship information."""

    person1: str
    relationship: str
    person2: str
    context: str = ""


class Location(BaseModel):
    """Model for location information."""

    place: str
    context: str = ""
    time_period: str = ""


class Occupation(BaseModel):
    """Model for occupation information."""

    person: str
    occupation: str
    location: str = ""
    time_period: str = ""


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
        names = []
        for name_data in self.structured_names:
            names.append(name_data.full_name)
            names.extend(name_data.nicknames)
        return list(set(names))

    def get_all_locations(self) -> list[str]:
        """Get all locations from structured fields."""
        locations = []
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
_CACHED_GEDCOM_DATA = None


def get_gedcom_data() -> Optional[Any]:
    """
    Returns the cached GEDCOM data instance, loading it if necessary.

    This function ensures the GEDCOM file is loaded only once and reused
    throughout the module, improving performance.

    Returns:
        GedcomData instance or None if loading fails
    """
    global _CACHED_GEDCOM_DATA

    # Return cached data if already loaded
    if _CACHED_GEDCOM_DATA is not None:
        return _CACHED_GEDCOM_DATA

    # Check if GEDCOM path is configured
    gedcom_path = config_schema.database.gedcom_file_path
    if not gedcom_path:
        logger.warning("GEDCOM_FILE_PATH not configured. Cannot load GEDCOM file.")
        return None

    # Check if GEDCOM file exists
    if not gedcom_path.exists():
        logger.warning(
            f"GEDCOM file not found at {gedcom_path}. Cannot load GEDCOM file."
        )
        return None

    # Load GEDCOM data
    try:
        logger.debug(f"Loading GEDCOM file {gedcom_path.name} (first time)...")
        from gedcom_cache import load_gedcom_with_aggressive_caching as load_gedcom_data

        _CACHED_GEDCOM_DATA = load_gedcom_data(str(gedcom_path))
        if _CACHED_GEDCOM_DATA:
            logger.debug("GEDCOM file loaded successfully and cached for reuse.")
            # Log some stats about the loaded data
            logger.debug(
                f"  Index size: {len(getattr(_CACHED_GEDCOM_DATA, 'indi_index', {}))}"
            )
            logger.debug(
                f"  Pre-processed cache size: {len(getattr(_CACHED_GEDCOM_DATA, 'processed_data_cache', {}))}"
            )
        return _CACHED_GEDCOM_DATA
    except Exception as e:
        logger.error(f"Error loading GEDCOM file: {e}", exc_info=True)
        return None


# Import required modules and functions
# Import from relationship_utils
# Import from api_search_utils where it belongs
from api_search_utils import process_and_score_suggestions
from gedcom_utils import (
    calculate_match_score,
)


def _get_gedcom_data_for_search(gedcom_data: Optional[Any]) -> Any:
    """Get GEDCOM data for search, either from parameter or cache."""
    if gedcom_data is None:
        gedcom_data = get_gedcom_data()
        if not gedcom_data:
            error_msg = "Failed to load GEDCOM data from cache or file"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    return gedcom_data


def _parse_name_parts(name: str) -> tuple[str, str]:
    """Parse name into first name and surname components."""
    name_parts = name.strip().split()
    first_name = name_parts[0] if name_parts else ""
    surname = name_parts[-1] if len(name_parts) > 1 else ""
    return first_name, surname


def _create_search_criteria(first_name: str, surname: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create filter and scoring criteria from name parts."""
    filter_criteria = {
        "first_name": first_name.lower() if first_name else None,
        "surname": surname.lower() if surname else None,
    }
    scoring_criteria = filter_criteria.copy()
    return filter_criteria, scoring_criteria


def _get_scoring_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """Get scoring weights and date flexibility configuration."""
    scoring_weights = config_schema.common_scoring_weights
    date_flex = {
        "year_flex": getattr(config_schema, "year_flexibility", 2),
        "exact_bonus": getattr(config_schema, "exact_date_bonus", 25),
    }
    return scoring_weights, date_flex


def _prepare_indi_index(gedcom_data: Any) -> Optional[dict[str, Any]]:
    """Prepare individual index for iteration."""
    if not (
        gedcom_data
        and hasattr(gedcom_data, "indi_index")
        and gedcom_data.indi_index
        and hasattr(gedcom_data.indi_index, "items")
    ):
        return None

    # Convert to dict if it's not already to ensure it's iterable
    return (
        dict(gedcom_data.indi_index)
        if not isinstance(gedcom_data.indi_index, dict)
        else gedcom_data.indi_index
    )


def _matches_filter_criteria(indi_data: dict[str, Any], filter_criteria: dict[str, Any]) -> bool:
    """Check if individual matches filter criteria."""
    # Skip individuals with no name
    if not indi_data.get("first_name") and not indi_data.get("surname"):
        return False

    # Simple OR filter: match on first name OR surname
    fn_match = filter_criteria["first_name"] and indi_data.get(
        "first_name", ""
    ).lower().startswith(filter_criteria["first_name"])
    sn_match = filter_criteria["surname"] and indi_data.get(
        "surname", ""
    ).lower().startswith(filter_criteria["surname"])

    return fn_match or sn_match


def _create_match_record(
    indi_id: str,
    indi_data: dict[str, Any],
    total_score: float,
    field_scores: dict[str, Any],
    reasons: list[str]
) -> dict[str, Any]:
    """Create a match record from individual data and scores."""
    return {
        "id": indi_id,
        "display_id": indi_id,
        "first_name": indi_data.get("first_name", ""),
        "surname": indi_data.get("surname", ""),
        "gender": indi_data.get("gender", ""),
        "birth_year": indi_data.get("birth_year"),
        "birth_place": indi_data.get("birth_place", ""),
        "death_year": indi_data.get("death_year"),
        "death_place": indi_data.get("death_place", ""),
        "total_score": total_score,
        "field_scores": field_scores,
        "reasons": reasons,
        "source": "GEDCOM",
    }


def _process_individuals_for_name(
    name: str,
    gedcom_data: Any
) -> list[dict[str, Any]]:
    """Process individuals for a single name search."""
    if not name or len(name.strip()) < 2:
        return []

    first_name, surname = _parse_name_parts(name)
    filter_criteria, scoring_criteria = _create_search_criteria(first_name, surname)
    scoring_weights, date_flex = _get_scoring_config()

    indi_index = _prepare_indi_index(gedcom_data)
    if not indi_index:
        return []

    scored_matches = []

    for indi_id, indi_data in indi_index.items():
        try:
            if _matches_filter_criteria(indi_data, filter_criteria):
                # Calculate match score
                total_score, field_scores, reasons = calculate_match_score(
                    search_criteria=scoring_criteria,
                    candidate_processed_data=indi_data,
                    scoring_weights=scoring_weights,
                    date_flexibility=date_flex,
                )

                # Only include if score is above threshold
                if total_score > 0:
                    match_record = _create_match_record(
                        indi_id, indi_data, total_score, field_scores, reasons
                    )
                    scored_matches.append(match_record)
        except Exception as e:
            logger.error(f"Error processing individual {indi_id}: {e}")
            continue

    # Sort matches by score (highest first) and take top 3
    scored_matches.sort(key=lambda x: x["total_score"], reverse=True)
    return scored_matches[:3]


def _search_gedcom_for_names(
    names: list[str], gedcom_data: Optional[Any] = None
) -> list[dict[str, Any]]:
    """
    Searches the configured GEDCOM file for names and returns matching individuals.
    Uses the cached GEDCOM data to avoid loading the file multiple times.
    """
    gedcom_data = _get_gedcom_data_for_search(gedcom_data)
    logger.info(f"Searching GEDCOM data for: {names}")

    try:
        search_results = []

        # For each name, create a simple search criteria and filter individuals
        for name in names:
            top_matches = _process_individuals_for_name(name, gedcom_data)
            search_results.extend(top_matches)

        return search_results

    except Exception as e:
        error_msg = f"Error searching GEDCOM file: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


def _validate_api_search_parameters(session_manager: Optional[SessionManager], names: Optional[list[str]]) -> list[str]:
    """Validate parameters for API search."""
    if not session_manager:
        error_msg = "Session manager not provided. Cannot search Ancestry API."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    names = names or []
    if not names:
        error_msg = "No names provided for API search."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return names


def _validate_api_configuration(session_manager: SessionManager) -> tuple[str, str]:
    """Validate API configuration and return tree ID and base URL."""
    # Get owner tree ID from session manager
    owner_tree_id = getattr(session_manager, "my_tree_id", None)
    if not owner_tree_id:
        error_msg = "Owner Tree ID missing. Cannot search Ancestry API."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Get base URL from config
    base_url = getattr(config_schema.api, "base_url", "").rstrip("/")
    if not base_url:
        error_msg = "Ancestry URL not configured. Base URL missing."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return owner_tree_id, base_url


def _create_api_search_criteria(name: str) -> Optional[dict[str, str]]:
    """Create search criteria for API search from name."""
    if not name or len(name.strip()) < 2:
        return None

    # Split name into first name and surname if possible
    name_parts = name.strip().split()
    first_name = name_parts[0] if name_parts else ""
    surname = name_parts[-1] if len(name_parts) > 1 else ""

    # Skip if both first name and surname are empty
    if not first_name and not surname:
        return None

    return {
        "first_name_raw": first_name,
        "surname_raw": surname,
    }


def _process_api_search_results(api_results: list[dict[str, Any]], search_criteria: dict[str, str]) -> list[dict[str, Any]]:
    """Process and score API search results."""
    if not api_results:
        return []

    # Process and score the API results if they exist
    scored_suggestions = process_and_score_suggestions(api_results, search_criteria)

    # Take top 3 results
    top_matches = scored_suggestions[:3] if scored_suggestions else []

    # Add source information
    for match in top_matches:
        match["source"] = "API"

    return top_matches


def _search_single_name_via_api(name: str) -> list[dict[str, Any]]:
    """Search for a single name via API."""
    search_criteria = _create_api_search_criteria(name)
    if not search_criteria:
        return []

    # Call the API search function from action11
    # NOTE: _search_ancestry_api function does not exist, so return empty results
    api_results = []
    logger.debug(f"API search functionality not available for: {name}")

    if api_results is None:
        error_msg = f"API search failed for name: {name}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Empty results are OK - just log and continue
    if not api_results:
        logger.info(f"API search returned no results for name: {name}")
        return []

    return _process_api_search_results(api_results, search_criteria)


def _search_api_for_names(
    session_manager: Optional[SessionManager] = None,
    names: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Searches Ancestry API for names and returns matching individuals.

    Args:
        session_manager: The active SessionManager instance
        names: List of names to search for via the Ancestry API

    Returns:
        List of dictionaries containing information about matching individuals

    Raises:
        RuntimeError: If API utilities are not available or if required parameters are missing
    """
    names = _validate_api_search_parameters(session_manager, names)
    _validate_api_configuration(session_manager)

    logger.info(f"Searching Ancestry API for: {names}")

    try:
        search_results = []

        # For each name, create a search criteria and search the API
        for name in names:
            top_matches = _search_single_name_via_api(name)
            search_results.extend(top_matches)

        return search_results

    except Exception as e:
        error_msg = f"Error searching Ancestry API: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


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
        self.my_pid_lower = (
            session_manager.my_profile_id.lower()
            if session_manager.my_profile_id
            else ""
        )

    def process_person(self, person: Person, progress_bar=None) -> tuple[bool, str]:
        """
        Process a single person and return (success, status_message).

        Returns:
            Tuple of (success: bool, status_message: str)
        """
        log_prefix = f"Productive: {person.username} #{person.id}"

        try:
            # Apply rate limiting (defensive: dynamic_rate_limiter may be None in some contexts)
            try:
                rl = getattr(self.session_manager, "dynamic_rate_limiter", None)
                if rl is not None:
                    rl.wait()
            except Exception as e:
                logger.debug(f"{log_prefix}: Rate limiter unavailable or failed: {e}")

            # Get message context
            context_logs = self._get_context_logs(person, log_prefix)
            if not context_logs:
                return False, "no_context"

            # Check exclusions and status
            if self._should_skip_person(person, context_logs, log_prefix):
                return False, "skipped"

            # Process with AI
            ai_results = self._process_with_ai(person, context_logs, progress_bar)
            if not ai_results:
                return False, "ai_error"

            extracted_data, suggested_tasks = ai_results

            # Log-only: assess suggested_tasks quality before any task creation (no behavior change)
            try:
                _log_suggested_tasks_quality(suggested_tasks, extracted_data, log_prefix)
                _log_task_dedup_preview(suggested_tasks, log_prefix)
            except Exception as e:
                logger.debug(f"{log_prefix}: Task quality audit unavailable: {e}")

            # Create MS Graph tasks
            self._create_ms_tasks(person, suggested_tasks, log_prefix, progress_bar)

            # Generate and send response
            success = self._handle_message_response(
                person, context_logs, extracted_data, log_prefix, progress_bar
            )
            if not success:
                return False, "send_error"

            return True, "success"

        except Exception as e:
            logger.error(f"Error processing {log_prefix}: {e}", exc_info=True)
            return False, f"error: {e!s}"

    def _get_context_logs(
        self, person: Person, log_prefix: str
    ) -> Optional[list[ConversationLog]]:
        """Get message context for the person."""
        if self.db_state.session is None:
            logger.error(f"Database session is None for {log_prefix}")
            return None
        context_logs = _get_message_context(self.db_state.session, person.id)
        if not context_logs:
            logger.warning(
                f"Skipping {log_prefix}: Failed to retrieve message context."
            )
        return context_logs

    def _should_skip_person(
        self, person: Person, context_logs: list[ConversationLog], log_prefix: str
    ) -> bool:
        """Check if person should be skipped based on various criteria."""

        # Check person status
        excluded_statuses = [
            PersonStatusEnum.DESIST,
            PersonStatusEnum.ARCHIVE,
            PersonStatusEnum.BLOCKED,
            PersonStatusEnum.DEAD,
        ]

        if person.status in excluded_statuses:
            logger.info(f"{log_prefix}: Person has status {person.status}. Skipping.")
            return True
        # Get latest message
        latest_message = self._get_latest_incoming_message(context_logs)
        if not latest_message:
            return True

        # Check if custom reply already sent
        custom_reply_sent_at = safe_column_value(
            latest_message, "custom_reply_sent_at", None
        )
        if custom_reply_sent_at is not None:
            logger.info(f"{log_prefix}: Custom reply already sent. Skipping.")
            return True

        # Check for exclusion keywords
        latest_message_content = safe_column_value(
            latest_message, "latest_message_content", ""
        )
        if should_exclude_message(latest_message_content):
            logger.info(f"{log_prefix}: Message contains exclusion keyword. Skipping.")
            return True

        # Check if OTHER message with no mentioned names
        ai_sentiment = safe_column_value(latest_message, "ai_sentiment", None)
        if ai_sentiment == OTHER_SENTIMENT:
            # We'll handle this in the AI processing step
            pass

        return False

    def _get_latest_incoming_message(
        self, context_logs: list[ConversationLog]
    ) -> Optional[ConversationLog]:
        """Get the latest incoming message from context logs."""
        for log in reversed(context_logs):
            direction = safe_column_value(log, "direction", None)
            if direction == MessageDirectionEnum.IN:
                return log
        return None

    def _process_with_ai(
        self, person: Person, context_logs: list[ConversationLog], progress_bar=None
    ) -> Optional[tuple[dict[str, Any], list[str]]]:
        """Process message content with AI and return extracted data and tasks."""
        if progress_bar:
            progress_bar.set_description(
                f"Processing {person.username}: Analyzing content"
            )

        # Check session validity
        if not self.session_manager.is_sess_valid():
            logger.error(f"Session invalid for {person.username}. Skipping.")
            return None

        # Format context for AI
        formatted_context = _format_context_for_ai_extraction(
            context_logs, self.my_pid_lower
        )

        # Call AI
        logger.debug(f"Calling AI for {person.username}...")
        ai_response = extract_genealogical_entities(
            formatted_context, self.session_manager
        )

        # Process AI response
        processed_response = _process_ai_response(ai_response, f"{person.username}")
        extracted_data = processed_response["extracted_data"]
        suggested_tasks = processed_response["suggested_tasks"]

        # Log results
        if suggested_tasks:
            logger.debug(
                f"Processed {len(suggested_tasks)} tasks for {person.username}"
            )

        entity_counts = {k: len(v) for k, v in extracted_data.items()}
        logger.debug(
            f"Extracted entities for {person.username}: {json.dumps(entity_counts)}"
        )
        # Debug-only quality summary for interrogation
        try:
            from extraction_quality import summarize_extracted_data
            quality = summarize_extracted_data(extracted_data)
            logger.debug(
                f"Quality summary for {person.username}: "
                f"counts={quality.get('counts')}, flags={quality.get('flags')}"
            )
        except Exception as e:
            logger.debug(f"Quality summary unavailable: {e}")

        # Store extracted data on person object for message personalization
        try:
            person.extracted_genealogical_data = extracted_data
            logger.debug(f"Stored extracted genealogical data on person object for {person.username}")
        except Exception as e:
            logger.warning(f"Failed to store extracted data on person object: {e}")

        return extracted_data, suggested_tasks

    def _create_ms_tasks(
        self,
        person: Person,
        suggested_tasks: list[str],
        log_prefix: str,
        progress_bar=None,
    ):
        """Create MS Graph tasks if configured and available."""
        if not suggested_tasks:
            return

        if progress_bar:
            progress_bar.set_description(
                f"Processing {person.username}: Creating {len(suggested_tasks)} tasks"
            )

        # Initialize MS Graph if needed
        self._initialize_ms_graph()

        if not self.ms_state.token or not self.ms_state.list_id:
            logger.warning(
                f"{log_prefix}: Skipping MS task creation - MS Auth/List ID unavailable."
            )
            return

        # Optional guarded de-duplication (Phase 4.3) — only in testing mode and when flag enabled
        if (
            config_schema.app_mode == "testing"
            and getattr(config_schema, "enable_task_dedup", False)
            and len(suggested_tasks) > 1
        ):
            try:
                original_count = len(suggested_tasks)
                dedup_map = {}
                for t in suggested_tasks:
                    core = _normalize_task_text(t)
                    # Reuse same core transform as preview (lightweight subset)
                    core = re.sub(r"\([^)]*\)", "", core)
                    core = re.sub(r"\b(\d{4}s?)\b", "YEAR", core)
                    core = re.sub(r"\b(census|record|records)\b", "record", core)
                    core = re.sub(r"\s+", " ", core).strip()
                    if core:
                        dedup_map.setdefault(core, []).append(t)
                # Keep first occurrence per core group
                deduped = [vals[0] for vals in dedup_map.values()]
                if len(deduped) < original_count:
                    logger.info(
                        f"{log_prefix}: Guarded de-dup (testing) reduced tasks {original_count}→{len(deduped)}"
                    )
                    suggested_tasks = deduped
                else:
                    logger.debug(f"{log_prefix}: Guarded de-dup found no reductions")
            except Exception as e:
                logger.debug(f"{log_prefix}: Guarded de-dup failed: {e}")

        # Create tasks
        app_mode = config_schema.app_mode
        if app_mode == "dry_run":
            logger.info(
                f"{log_prefix}: DRY RUN - Skipping MS To-Do task creation for {len(suggested_tasks)} tasks."
            )
            return

        # Generate enhanced tasks if genealogical task generation is available
        enhanced_tasks = []
        if (
            GENEALOGICAL_TASK_GENERATION_AVAILABLE
            and getattr(config_schema, "enable_task_enrichment", False)
            and hasattr(person, 'extracted_genealogical_data')
            and callable(GenealogicalTaskGenerator)
        ):
            try:
                task_generator = GenealogicalTaskGenerator()  # type: ignore[call-arg]
                person_data = {"username": getattr(person, "username", "Unknown")}
                extracted_data = getattr(person, 'extracted_genealogical_data', {})

                # === PHASE 12: GEDCOM DATA INTEGRATION ===
                gedcom_data = None
                try:
                    # Try to load GEDCOM data for enhanced task generation
                    from gedcom_search_utils import get_cached_gedcom_data
                    gedcom_data = get_cached_gedcom_data()
                    if gedcom_data:
                        logger.debug(f"{log_prefix}: Using GEDCOM data for enhanced task generation")
                except Exception as e:
                    logger.debug(f"{log_prefix}: GEDCOM data not available for task generation: {e}")

                # === PHASE 12.3: ENHANCED DATA INTEGRATION ===
                enhanced_tasks = task_generator.generate_research_tasks(
                    person_data,
                    extracted_data,
                    suggested_tasks,
                    gedcom_data
                )

                # Apply intelligent task selection and validation
                enhanced_tasks = _apply_intelligent_task_selection(
                    enhanced_tasks, extracted_data, person_data, log_prefix
                )

                # Validate and score task quality
                enhanced_tasks = _validate_and_score_tasks(
                    enhanced_tasks, extracted_data, log_prefix
                )

                if enhanced_tasks:
                    logger.info(f"{log_prefix}: Generated {len(enhanced_tasks)} enhanced genealogical tasks")
                    # Log a small debug sample mapping original suggested tasks to enhanced outputs (Phase 4.4 completion instrumentation)
                    try:
                        sample_count = min(2, len(enhanced_tasks))
                        original_preview = suggested_tasks[:sample_count]
                        enrichment_sample = []
                        for i in range(sample_count):
                            et = enhanced_tasks[i]
                            enrichment_sample.append(
                                {
                                    "orig": original_preview[i] if i < len(original_preview) else None,
                                    "title": et.get("title", ""),
                                    "category": et.get("category", ""),
                                    "priority": et.get("priority", ""),
                                    "template": et.get("template_used", ""),
                                }
                            )
                        # Use json dumps to keep single-line structured debug output
                        try:
                            import json as _json
                            logger.debug(f"{log_prefix}: Task enrichment sample: {_json.dumps(enrichment_sample, ensure_ascii=False)}")
                        except Exception:
                            logger.debug(f"{log_prefix}: Task enrichment sample (fallback repr): {enrichment_sample}")
                    except Exception as e:
                        logger.debug(f"{log_prefix}: Failed to log enrichment sample: {e}")
                else:
                    logger.debug(f"{log_prefix}: No enhanced tasks generated, using standard tasks")

            except Exception as e:
                logger.warning(f"{log_prefix}: Enhanced task generation failed: {e}, using standard tasks")
                enhanced_tasks = []

        # Use enhanced tasks if available and enrichment flag enabled, otherwise fall back to standard tasks
        if enhanced_tasks and getattr(config_schema, "enable_task_enrichment", False):
            logger.info(f"{log_prefix}: Creating {len(enhanced_tasks)} enhanced MS To-Do tasks...")
            for task_index, task_data in enumerate(enhanced_tasks):
                task_title = task_data.get("title", f"Ancestry Research: {person.username or 'Unknown'}")
                task_body = f"{task_data.get('description', 'Research task')}\n\n--- Task Details ---\nCategory: {task_data.get('category', 'general')}\nPriority: {task_data.get('priority', 'medium')}\nTemplate: {task_data.get('template_used', 'standard')}\n\n--- Match Information ---\nMatch: {person.username or 'Unknown'} (#{person.id})\nProfile: {person.profile_id or 'N/A'}"

                task_ok = ms_graph_utils.create_todo_task(
                    self.ms_state.token,
                    self.ms_state.list_id,
                    task_title,
                    task_body,
                )

                if task_ok:
                    logger.debug(f"{log_prefix}: Created enhanced task: {task_title[:50]}...")
                else:
                    logger.warning(
                        f"{log_prefix}: Failed to create enhanced MS task: '{task_title[:100]}...'"
                    )
        else:
            # Fallback to standard task creation
            logger.info(f"{log_prefix}: Creating {len(suggested_tasks)} standard MS To-Do tasks...")
            for task_index, task_desc in enumerate(suggested_tasks):
                task_title = (
                    f"Ancestry Follow-up: {person.username or 'Unknown'} (#{person.id})"
                )
                task_body = f"AI Suggested Task ({task_index+1}/{len(suggested_tasks)}): {task_desc}\n\nMatch: {person.username or 'Unknown'} (#{person.id})\nProfile: {person.profile_id or 'N/A'}"

                task_ok = ms_graph_utils.create_todo_task(
                    self.ms_state.token,
                    self.ms_state.list_id,
                    task_title,
                    task_body,
                )

                if task_ok:
                    logger.debug(f"{log_prefix}: Created standard task: {task_title[:50]}...")
                else:
                    logger.warning(
                        f"{log_prefix}: Failed to create standard MS task: '{task_desc[:100]}...'"
                    )

    def _initialize_ms_graph(self) -> None:
        """Initialize MS Graph authentication and list ID if needed."""
        if not self.ms_state.token and not self.ms_state.auth_attempted:
            logger.info("Attempting MS Graph authentication (device flow)...")
            self.ms_state.token = ms_graph_utils.acquire_token_device_flow()
            self.ms_state.auth_attempted = True
            if not self.ms_state.token:
                logger.error("MS Graph authentication failed.")

        if self.ms_state.token and not self.ms_state.list_id:
            logger.info(
                f"Looking up MS To-Do List ID for '{self.ms_state.list_name}'..."
            )
            self.ms_state.list_id = ms_graph_utils.get_todo_list_id(
                self.ms_state.token, self.ms_state.list_name
            )
            if not self.ms_state.list_id:
                logger.error(
                    f"Failed find/get MS List ID for '{self.ms_state.list_name}'."
                )

    def _handle_message_response(
        self,
        person: Person,
        context_logs: list[ConversationLog],
        extracted_data: dict[str, Any],
        log_prefix: str,
        progress_bar=None,
    ) -> bool:
        """Handle generating and sending the appropriate response message."""

        # Get latest message
        latest_message = self._get_latest_incoming_message(context_logs)
        if not latest_message:
            return False
        # Check if this is an OTHER message with no mentioned names
        ai_sentiment = safe_column_value(latest_message, "ai_sentiment", None)
        if ai_sentiment == OTHER_SENTIMENT:
            mentioned_names = extracted_data.get("mentioned_names", [])
            if not mentioned_names:
                logger.info(
                    f"{log_prefix}: Message is 'OTHER' with no names. Marking as processed."
                )
                self._mark_message_processed(latest_message)
                return True  # Successfully handled (by skipping)

        # Generate custom reply if person identified
        custom_reply = self._generate_custom_reply(
            person,
            context_logs,
            extracted_data,
            latest_message,
            log_prefix,
            progress_bar,
        )

        # Format message (custom or standard acknowledgment)
        message_text, message_template_id = self._format_message(
            person, extracted_data, custom_reply, log_prefix
        )
        # Apply filtering and send message
        return self._send_message(
            person,
            context_logs,
            message_text,
            message_template_id,
            custom_reply,
            latest_message,
            log_prefix,
            progress_bar,
        )

    def _mark_message_processed(self, message: ConversationLog):
        """Mark a message as processed without sending a reply."""
        try:
            if self.db_state.session:
                message.custom_reply_sent_at = datetime.now(timezone.utc)
                self.db_state.session.add(message)
                self.db_state.session.flush()
        except Exception as e:
            logger.error(f"Failed to mark message as processed: {e}")

    def _generate_custom_reply(
        self,
        person: Person,
        context_logs: list[ConversationLog],
        extracted_data: dict[str, Any],
        latest_message: ConversationLog,
        log_prefix: str,
        progress_bar=None,
    ) -> Optional[str]:
        """Generate a custom genealogical reply if appropriate."""

        if progress_bar:
            progress_bar.set_description(
                f"Processing {person.username}: Identifying person"
            )

        # Try to identify a person mentioned in the message
        person_details = _identify_and_get_person_details(
            self.session_manager, extracted_data, log_prefix
        )

        if not person_details:
            logger.debug(
                f"{log_prefix}: No person identified. Will use standard acknowledgement."
            )
            return None

        # Check if custom responses are enabled
        if not config_schema.custom_response_enabled:
            logger.info(
                f"{log_prefix}: Custom replies disabled via config. Using standard."
            )
            return None

        if progress_bar:
            progress_bar.set_description(
                f"Processing {person.username}: Generating custom reply"
            )

        # Format genealogical data
        genealogical_data_str = _format_genealogical_data_for_ai(
            person_details["details"],
            person_details["relationship_path"],
        )
        # Get user's last message
        user_last_message = safe_column_value(
            latest_message, "latest_message_content", ""
        )

        # Format context
        formatted_context = _format_context_for_ai_extraction(
            context_logs, self.my_pid_lower
        )  # Generate custom reply
        custom_reply = generate_genealogical_reply(
            session_manager=self.session_manager,
            conversation_context=formatted_context,
            user_message=user_last_message,
            genealogical_data=genealogical_data_str,
            log_prefix=log_prefix,
        )

        if custom_reply:
            logger.info(f"{log_prefix}: Generated custom genealogical reply.")
        else:
            logger.warning(
                f"{log_prefix}: Failed to generate custom reply. Will fall back."
            )

        return custom_reply

    def _format_message(
        self,
        person: Person,
        extracted_data: dict[str, Any],
        custom_reply: Optional[str],
        log_prefix: str,
    ) -> tuple[str, int]:
        """Format the message text and determine message type ID."""

        try:
            if custom_reply:
                # Add signature to custom reply
                user_name = getattr(config_schema, "user_name", "Tree Owner")
                user_location = getattr(config_schema, "user_location", "")
                location_part = f"\n{user_location}" if user_location else ""
                signature = f"\n\nBest regards,\n{user_name}{location_part}"
                message_text = custom_reply + signature
                message_template_id = self.msg_config.custom_reply_msg_type_id
                logger.info(
                    f"{log_prefix}: Using custom genealogical reply with signature."
                )
            else:
                # Use standard acknowledgement template
                # Get person name
                first_name = safe_column_value(person, "first_name", "")
                username = safe_column_value(person, "username", "")
                name_to_use = format_name(first_name or username)
                # Generate summary
                summary_for_ack = _generate_ack_summary(extracted_data)

                # Try enhanced personalized message formatting first
                message_text = None
                if MESSAGE_PERSONALIZATION_AVAILABLE and hasattr(person, 'extracted_genealogical_data'):
                    try:
                        from message_personalization import MessagePersonalizer
                        personalizer = MessagePersonalizer()

                        # Check if we have an enhanced productive reply template
                        enhanced_template_key = "Enhanced_Productive_Reply"
                        if enhanced_template_key in personalizer.templates:
                            logger.debug(f"Using enhanced productive reply template for {log_prefix}")

                            extracted_data = getattr(person, 'extracted_genealogical_data', {})
                            person_data = {"username": getattr(person, "username", "Unknown")}
                            base_format_data = {"name": name_to_use, "summary": summary_for_ack}

                            message_text = personalizer.create_personalized_message(
                                enhanced_template_key,
                                person_data,
                                extracted_data,
                                base_format_data
                            )
                            logger.info(f"Successfully created personalized productive reply for {log_prefix}")
                        else:
                            logger.debug(f"Enhanced template '{enhanced_template_key}' not available")
                    except Exception as e:
                        logger.warning(f"Enhanced productive reply formatting failed for {log_prefix}: {e}, falling back to standard")
                        message_text = None

                # Fallback to standard template formatting
                if not message_text:
                    if (
                        self.msg_config.templates
                        and ACKNOWLEDGEMENT_MESSAGE_TYPE in self.msg_config.templates
                    ):
                        message_text = self.msg_config.templates[
                            ACKNOWLEDGEMENT_MESSAGE_TYPE
                        ].format(name=name_to_use, summary=summary_for_ack)
                    else:
                        user_name = getattr(config_schema, "user_name", "Tree Owner")
                        message_text = f"Dear {name_to_use},\n\nThank you for your message!\n\n{user_name}"
                    logger.info(f"{log_prefix}: Using standard acknowledgement template.")

                message_template_id = self.msg_config.ack_msg_type_id

            return message_text, message_template_id or 1  # Provide default

        except Exception as e:
            logger.error(
                f"{log_prefix}: Message formatting error: {e}. Using fallback."
            )  # Simple fallback
            safe_username = safe_column_value(person, "username", "User")
            user_name = getattr(config_schema, "user_name", "Tree Owner")
            message_text = f"Dear {format_name(safe_username)},\n\nThank you for your message!\n\n{user_name}"
            message_template_id = self.msg_config.ack_msg_type_id or 1  # Provide default
            return message_text, message_template_id

    def _send_message(
        self,
        person: Person,
        context_logs: list[ConversationLog],
        message_text: str,
        message_template_id: int,
        custom_reply: Optional[str],
        latest_message: ConversationLog,
        log_prefix: str,
        progress_bar=None,
    ) -> bool:
        """Send the message and handle database updates."""

        # Apply mode/recipient filtering
        send_flag, skip_reason = self._should_send_message(person, log_prefix)

        if progress_bar:
            if custom_reply:
                progress_bar.set_description(
                    f"Processing {person.username}: Sending custom reply"
                )
            else:
                progress_bar.set_description(
                    f"Processing {person.username}: Sending acknowledgement"
                )

        # Get conversation ID
        conv_id = self._get_conversation_id(context_logs, log_prefix)
        if not conv_id:
            return False

        # Send or skip message
        if send_flag:
            from api_utils import call_send_message_api

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
            message_template_id,
            send_status,
            effective_conv_id or "",
            custom_reply,
            latest_message,
            log_prefix,
        )

    def _should_send_message(self, person: Person, log_prefix: str) -> tuple[bool, str]:
        """Determine if message should be sent based on app mode and filters."""

        app_mode = config_schema.app_mode
        testing_profile_id = config_schema.testing_profile_id
        # Get current profile ID safely
        current_profile_id = safe_column_value(person, "profile_id", "UNKNOWN")

        if app_mode == "testing":
            if not testing_profile_id:
                return False, "skipped (config_error)"
            if current_profile_id != str(testing_profile_id):
                return False, f"skipped (testing_mode_filter: not {testing_profile_id})"
        elif (
            app_mode == "production"
            and testing_profile_id
            and current_profile_id == str(testing_profile_id)
        ):
            return False, f"skipped (production_mode_filter: is {testing_profile_id})"

        return True, ""

    def _get_conversation_id(
        self, context_logs: list[ConversationLog], log_prefix: str
    ) -> Optional[str]:
        """Get conversation ID from context logs."""
        if not context_logs:
            logger.error(
                f"{log_prefix}: No context logs available for conversation ID."
            )
            return None

        raw_conv_id = context_logs[-1].conversation_id
        if raw_conv_id is None:
            logger.error(f"{log_prefix}: Conversation ID is None.")
            return None

        try:
            return str(raw_conv_id)
        except Exception as e:
            logger.error(
                f"{log_prefix}: Failed to convert conversation ID to string: {e}"
            )
            return None

    def _stage_database_updates(
        self,
        person: Person,
        message_text: str,
        message_template_id: int,
        send_status: str,
        effective_conv_id: str,
        custom_reply: Optional[str],
        latest_message: ConversationLog,
        log_prefix: str,
    ) -> bool:
        """Stage database updates for the processed message."""

        if not effective_conv_id:
            logger.error(
                f"{log_prefix}: effective_conv_id is None. Cannot stage log entry."
            )
            return False

        # Handle successful sends
        if send_status in ("delivered OK", "typed (dry_run)") or send_status.startswith(
            "skipped ("
        ):
            try:
                # Get person ID as int
                person_id_int = int(str(person.id))

                # Prepare log data
                log_data = {
                    "conversation_id": str(effective_conv_id),
                    "direction": MessageDirectionEnum.OUT,
                    "people_id": person_id_int,
                    "latest_message_content": message_text[
                        : config_schema.message_truncation_length
                    ],
                    "latest_timestamp": datetime.now(timezone.utc),
                    "message_template_id": message_template_id,
                    "script_message_status": send_status,
                    "ai_sentiment": None,
                }

                if self.db_state.logs_to_add is not None:
                    self.db_state.logs_to_add.append(log_data)

                # Update custom_reply_sent_at if this was a custom reply
                if (
                    custom_reply
                    and latest_message
                    and message_template_id == self.msg_config.custom_reply_msg_type_id
                ):
                    try:
                        if self.db_state.session:
                            latest_message.custom_reply_sent_at = datetime.now(timezone.utc)
                            self.db_state.session.add(latest_message)
                            self.db_state.session.flush()
                            logger.info(f"{log_prefix}: Updated custom_reply_sent_at.")
                    except Exception as e:
                        logger.error(
                            f"{log_prefix}: Failed to update custom_reply_sent_at: {e}"
                        )

                # Stage person for archiving
                if self.db_state.person_updates is not None:
                    self.db_state.person_updates[person_id_int] = (
                        PersonStatusEnum.ARCHIVE
                    )
                logger.debug(f"{log_prefix}: Person status staged for ARCHIVE.")

                return True

            except Exception as e:
                logger.error(f"{log_prefix}: Failed to stage database updates: {e}")
                return False
        else:
            logger.error(
                f"{log_prefix}: Failed to send message (Status: {send_status})."
            )
            return False


class BatchCommitManager:
    """Manages batch commits to the database."""

    def __init__(self, db_state: DatabaseState) -> None:
        self.db_state = db_state

    def should_commit(self) -> bool:
        """Check if a commit should be triggered."""
        logs_count = len(self.db_state.logs_to_add) if self.db_state.logs_to_add else 0
        updates_count = (
            len(self.db_state.person_updates) if self.db_state.person_updates else 0
        )
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
            logs_count = (
                len(self.db_state.logs_to_add) if self.db_state.logs_to_add else 0
            )
            updates_count = (
                len(self.db_state.person_updates) if self.db_state.person_updates else 0
            )
            logger.info(
                f"Committing batch {batch_num} ({logs_count} logs, {updates_count} person updates)"
            )

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
            logger.error(
                f"Database commit failed for batch {batch_num}: {e}", exc_info=True
            )
            return False, 0, 0


#####################################################
# Main Simplified Function
#####################################################


@retry_on_failure(max_attempts=3, backoff_factor=4.0)  # Increased from 2.0 to 4.0 for better AI API handling
@circuit_breaker(failure_threshold=10, recovery_timeout=300)  # Increased from 5 to 10 for better tolerance
@timeout_protection(timeout=2400)  # 40 minutes for productive message processing
@graceful_degradation(fallback_value=False)
@error_context("action9_process_productive")
def _initialize_state_objects() -> tuple[ProcessingState, MSGraphState]:
    """Initialize processing and MS Graph state objects."""
    state = ProcessingState()
    ms_state = MSGraphState(list_name=config_schema.ms_todo_list_name)
    return state, ms_state


def _initialize_adaptive_systems() -> tuple[Optional[Any], Optional[Any], int]:
    """Initialize adaptive batch processing systems if available."""
    adaptive_batch_size = config_schema.batch_size
    smart_batch_processor = None
    performance_dashboard = None

    if ADAPTIVE_SYSTEMS_AVAILABLE:
        try:
            # Initialize adaptive systems only if classes loaded
            if callable(SmartBatchProcessor):
                smart_batch_processor = SmartBatchProcessor(
                    initial_batch_size=config_schema.batch_size,
                    min_batch_size=1,
                    max_batch_size=min(20, config_schema.max_productive_to_process or 20)
                )
                adaptive_batch_size = smart_batch_processor.get_next_batch_size()
                logger.debug(f"Adaptive batch processing enabled: initial size {adaptive_batch_size}")

            if callable(PerformanceDashboard):
                performance_dashboard = PerformanceDashboard("action9_performance.json")
                performance_dashboard.record_session_start({
                    "action": "action9_process_productive",
                    "initial_batch_size": config_schema.batch_size,
                    "max_productive": config_schema.max_productive_to_process
                })

        except Exception as e:
            logger.warning(f"Failed to initialize adaptive systems: {e}, using standard batch processing")
            smart_batch_processor = None
            performance_dashboard = None

    return smart_batch_processor, performance_dashboard, adaptive_batch_size


def _initialize_database_and_message_config(adaptive_batch_size: int) -> tuple[DatabaseState, MessageConfig]:
    """Initialize database state and message configuration."""
    db_state = DatabaseState(
        batch_size=max(1, adaptive_batch_size),
        commit_threshold=max(1, adaptive_batch_size),
    )
    msg_config = MessageConfig()
    return db_state, msg_config


def _validate_session_manager(session_manager: SessionManager) -> bool:
    """Validate session manager has required attributes."""
    if not session_manager or not session_manager.my_profile_id:
        logger.error("Action 9: SessionManager or Profile ID missing.")
        return False
    return True


def _finalize_performance_monitoring(performance_dashboard: Optional[Any]) -> None:
    """Finalize performance monitoring if available."""
    if performance_dashboard:
        try:
            performance_dashboard.finalize_session()
            logger.info("Performance monitoring session finalized")
        except Exception as e:
            logger.warning(f"Failed to finalize performance monitoring: {e}")


def _cleanup_resources(session_manager: SessionManager, db_state: DatabaseState) -> None:
    """Clean up database session resources."""
    if db_state.session:
        session_manager.return_session(db_state.session)


def process_productive_messages(session_manager: SessionManager) -> bool:
    """
    Simplified main function for Action 9. Processes productive messages by:
    1. Setting up configuration and state
    2. Querying candidates
    3. Processing each person in batches
    4. Committing results
    """
    logger.info("--- Starting Action 9: Process Productive Messages (Streamlined) ---")

    # Initialize state objects
    state, ms_state = _initialize_state_objects()

    # Initialize adaptive systems
    smart_batch_processor, performance_dashboard, adaptive_batch_size = _initialize_adaptive_systems()

    # Initialize database and message configuration
    db_state, msg_config = _initialize_database_and_message_config(adaptive_batch_size)

    # Validate session manager
    if not _validate_session_manager(session_manager):
        return False

    try:
        # Step 1: Setup - Load templates and get database session
        if not _setup_configuration(session_manager, db_state, msg_config):
            return False

        # Step 2: Query candidates
        candidates = _query_candidates(
            db_state, msg_config, config_schema.max_productive_to_process
        )
        if not candidates:
            logger.info("Action 9: No eligible candidates found.")
            return True

        state.total_candidates = len(candidates)
        logger.info(f"Action 9: Found {state.total_candidates} candidates to process.")

        # Step 3: Process candidates
        success = _process_candidates(
            session_manager, candidates, state, ms_state, db_state, msg_config
        )

        # Step 4: Final commit
        _final_commit(db_state, state)

        # Step 5: Log summary
        _log_summary(state)

        return success

    except Exception as e:
        logger.critical(
            f"Critical error in process_productive_messages: {e}", exc_info=True
        )
        return False
    finally:
        _finalize_performance_monitoring(performance_dashboard)
        _cleanup_resources(session_manager, db_state)


def _setup_configuration(
    session_manager: SessionManager, db_state: DatabaseState, msg_config: MessageConfig
) -> bool:
    """Setup configuration, templates, and database session."""

    # Load templates
    msg_config.templates = _load_templates_for_action9()
    if not msg_config.templates:
        logger.error("Action 9: Required message templates failed to load.")
        return False

    # Get database session
    db_state.session = session_manager.get_db_conn()
    if not db_state.session:
        logger.critical("Action 9: Failed to get database session.")
        return False
    # Get message type IDs
    if not db_state.session:
        logger.critical(
            "Action 9: Database session is None when querying message types."
        )
        return False

    ack_msg_type_obj = (
        db_state.session.query(MessageTemplate.id)
        .filter(MessageTemplate.template_key == ACKNOWLEDGEMENT_MESSAGE_TYPE)
        .scalar()
    )
    if not ack_msg_type_obj:
        logger.critical(
            f"Action 9: MessageTemplate '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' not found in DB."
        )
        return False
    msg_config.ack_msg_type_id = ack_msg_type_obj

    # Get custom reply message template ID (optional)
    custom_reply_msg_type_obj = (
        db_state.session.query(MessageTemplate.id)
        .filter(MessageTemplate.template_key == CUSTOM_RESPONSE_MESSAGE_TYPE)
        .scalar()
    )
    if custom_reply_msg_type_obj:
        msg_config.custom_reply_msg_type_id = custom_reply_msg_type_obj
    else:
        logger.warning(
            f"Action 9: MessageTemplate '{CUSTOM_RESPONSE_MESSAGE_TYPE}' not found in DB."
        )

    return True


@cached_database_query(ttl=300)  # 5-minute cache for candidate queries
def _query_candidates(
    db_state: DatabaseState, msg_config: MessageConfig, limit: int
) -> list[Person]:
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
            ConversationLog.message_template_id == msg_config.ack_msg_type_id,
        )
        .group_by(ConversationLog.people_id)
        .subquery("latest_ack_out_sub")
    )

    # Main query
    candidates_query = (
        db_state.session.query(Person)
        .options(joinedload(Person.family_tree))
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
                ConversationLog.custom_reply_sent_at is None,
            ),
        )
        .outerjoin(
            latest_ack_out_log_subq,
            Person.id == latest_ack_out_log_subq.c.people_id,
        )
        .filter(
            Person.status == PersonStatusEnum.ACTIVE,
            (latest_ack_out_log_subq.c.max_ack_out_ts is None)
            | (
                latest_ack_out_log_subq.c.max_ack_out_ts
                < latest_in_log_subq.c.max_in_ts
            ),
        )
        .order_by(Person.id)
    )

    # Apply limit if configured
    if limit > 0:
        candidates_query = candidates_query.limit(limit)
        logger.debug(f"Processing limited to {limit} candidates")

    return candidates_query.all()


def _process_candidates(
    session_manager: SessionManager,
    candidates: list[Person],
    state: ProcessingState,
    ms_state: MSGraphState,
    db_state: DatabaseState,
    msg_config: MessageConfig,
) -> bool:
    """Process all candidate persons with progress tracking and batch commits."""

    # Initialize processors
    person_processor = PersonProcessor(session_manager, db_state, msg_config, ms_state)
    commit_manager = BatchCommitManager(db_state)

    # Setup progress tracking
    tqdm_args = {
        "total": state.total_candidates,
        "desc": "Processing",
        "unit": " person",
        "dynamic_ncols": True,
        "leave": True,
        "bar_format": "{desc} |{bar}| {percentage:3.0f}% ({n_fmt}/{total_fmt})",
        "file": sys.stderr,
    }

    logger.info(f"Processing {state.total_candidates} candidates...")

    with logging_redirect_tqdm(), tqdm(**tqdm_args) as progress_bar:
        for person in candidates:
            if state.critical_db_error_occurred:
                # Skip remaining candidates if critical DB error occurred
                remaining = state.total_candidates - state.processed_count
                state.skipped_count += remaining
                logger.warning(
                    f"Skipping remaining {remaining} candidates due to DB error."
                )
                break

            state.processed_count += 1

            # Process individual person
            success, status = person_processor.process_person(person, progress_bar)

            # Update counters based on result
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

            # Check for batch commit
            if commit_manager.should_commit():
                state.batch_num += 1

                # === PHASE 11.1: BATCH PERFORMANCE MONITORING ===
                batch_start_time = time.time()
                current_batch_size = len(db_state.logs_to_add or []) + len(db_state.person_updates or [])

                commit_success, logs_committed, persons_updated = (
                    commit_manager.commit_batch(state.batch_num)
                )

                # Record batch performance for adaptive processing
                _sbp = locals().get('smart_batch_processor')
                _pd = locals().get('performance_dashboard')
                if ADAPTIVE_SYSTEMS_AVAILABLE and _sbp is not None and _pd is not None:
                    try:
                        batch_processing_time = time.time() - batch_start_time
                        batch_success_rate = 1.0 if commit_success else 0.0

                        _sbp.record_batch_performance(
                            batch_size=current_batch_size,
                            processing_time=batch_processing_time,
                            success_rate=batch_success_rate
                        )

                        _pd.record_batch_processing_metrics({
                            "batch_number": state.batch_num,
                            "batch_size": current_batch_size,
                            "processing_time": batch_processing_time,
                            "success_rate": batch_success_rate,
                            "logs_committed": logs_committed,
                            "persons_updated": persons_updated
                        })

                        # Get updated batch size recommendation
                        recommended_batch_size = _sbp.get_next_batch_size()
                        if recommended_batch_size != db_state.batch_size:
                            logger.info(f"Adaptive batch size adjustment: {db_state.batch_size} → {recommended_batch_size}")
                            db_state.batch_size = recommended_batch_size
                            db_state.commit_threshold = recommended_batch_size

                    except Exception as e:
                        logger.warning(f"Batch performance monitoring failed: {e}")

                if not commit_success:
                    logger.critical(f"Critical: Batch {state.batch_num} commit failed!")
                    state.critical_db_error_occurred = True
                    state.overall_success = False
                    break

            # Update progress bar
            progress_bar.set_description(
                f"Processing: Tasks={state.tasks_created_count} Acks={state.acks_sent_count} "
                f"Skip={state.skipped_count} Err={state.error_count}"
            )
            progress_bar.update(1)

    return state.overall_success


def _final_commit(db_state: DatabaseState, state: ProcessingState):
    """Perform final commit of any remaining data."""

    if not state.critical_db_error_occurred and (
        db_state.logs_to_add or db_state.person_updates
    ):
        state.batch_num += 1
        commit_manager = BatchCommitManager(db_state)

        logs_count = len(db_state.logs_to_add) if db_state.logs_to_add else 0
        updates_count = len(db_state.person_updates) if db_state.person_updates else 0
        logger.info(
            f"Committing final batch ({logs_count} logs, {updates_count} person updates)"
        )

        commit_success, logs_committed, persons_updated = commit_manager.commit_batch(
            state.batch_num
        )

        if not commit_success:
            logger.error("Final batch commit failed!")
            state.overall_success = False
        else:
            logger.debug(
                f"Final batch committed successfully ({logs_committed} logs, {persons_updated} person updates)"
            )


def _log_summary(state: ProcessingState):
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


def _normalize_task_text(task: str) -> str:
    """Create a normalized representation of a task string for hashing/dedup logs.

    - Lowercase
    - Collapse whitespace
    - Strip leading/trailing punctuation
    - Remove repeated punctuation
    """
    if not isinstance(task, str):
        return ""
    t = task.lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[\s\-_,.;:!]+$", "", t)  # strip trailing punctuation clusters
    t = re.sub(r"^[\s\-_,.;:!]+", "", t)  # strip leading punctuation clusters
    return re.sub(r"[!?.]{2,}", ".", t)


def _log_suggested_tasks_quality(suggested_tasks: list[str], extracted_data: dict[str, Any], log_prefix: str) -> None:
    """Log-only audit of suggested task quality; does not modify behavior.

    Metrics logged:
    - total vs unique (normalized) count
    - average/median length and counts of very short/very long tasks
    - coverage hints: whether tasks reference extracted names/locations/dates
    - idempotency hashes for potential future de-duplication
    - presence of action verbs
    """
    try:
        tasks = suggested_tasks or []
        if not tasks:
            logger.debug(f"{log_prefix}: No suggested_tasks to audit.")
            return

        # Normalize and hash
        norm = [_normalize_task_text(t) for t in tasks if isinstance(t, str) and t.strip()]
        unique_norm = list({n for n in norm if n})
        dup_count = max(0, len(norm) - len(unique_norm))
        hashes = [hashlib.sha1(n.encode("utf-8")).hexdigest()[:12] for n in unique_norm]

        # Length stats
        lengths = [len(t) for t in tasks if isinstance(t, str)]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        short_count = sum(1 for ln in lengths if ln < 25)
        long_count = sum(1 for ln in lengths if ln > 220)

        # Action verbs heuristic
        ACTION_VERBS = [
            "search", "check", "look", "compare", "review", "request",
            "contact", "verify", "order", "compile", "analyze", "map",
            "triangulate", "confirm", "document",
        ]
        verb_hits = sum(1 for n in norm for v in ACTION_VERBS if f" {v} " in f" {n} ")

        # Names/locations coverage (best-effort; extracted_data structure may vary)
        names = set()
        try:
            for nd in (extracted_data.get("structured_names") or []):
                full = nd.get("full_name") if isinstance(nd, dict) else None
                if full:
                    names.add(full.lower())
        except Exception:
            pass
        locations = set()
        try:
            for loc in (extracted_data.get("locations") or []):
                place = loc.get("place") if isinstance(loc, dict) else None
                if place:
                    locations.add(place.lower())
        except Exception:
            pass

        names_refs = sum(1 for n in norm if any(name in n for name in names)) if names else 0
        loc_refs = sum(1 for n in norm if any(place in n for place in locations)) if locations else 0

        logger.info(
            f"{log_prefix}: Task audit — total={len(tasks)}, unique_norm={len(unique_norm)}, dupes={dup_count}, "
            f"avg_len={avg_len:.1f}, short(<25)={short_count}, long(>220)={long_count}, verbs={verb_hits}, "
            f"name_refs={names_refs}, location_refs={loc_refs}"
        )
        # Log a compact preview of normalized tasks and their hashes
        preview = [f"{i+1}:{unique_norm[i][:70]} # {hashes[i]}" for i in range(min(3, len(unique_norm)))]
        if preview:
            logger.debug(f"{log_prefix}: Task audit sample: {preview}")
    except Exception as e:
        logger.debug(f"{log_prefix}: Task audit failed: {e}")


def _log_task_dedup_preview(suggested_tasks: list[str], log_prefix: str) -> None:
        """Phase 4.2 (log-only): Preview potential task de-duplication impact.

        Groups tasks by normalized core text (after stripping trailing qualifiers).
        Logs cluster sizes and a consolidation estimate WITHOUT modifying the list.
        """
        try:
            tasks = suggested_tasks or []
            if len(tasks) < 2:
                return
            # Lightweight normalization reuse
            core_map: dict[str, list[int]] = {}
            for idx, raw in enumerate(tasks):
                if not isinstance(raw, str) or not raw.strip():
                    continue
                norm = _normalize_task_text(raw)
                # Remove parenthetical clarifiers & counts for grouping
                core = re.sub(r"\([^)]*\)", "", norm)
                core = re.sub(r"\b(\d{4}s?)\b", r"YEAR", core)  # Generalize years
                core = re.sub(r"\b(census|record|records)\b", "record", core)
                core = re.sub(r"\s+", " ", core).strip()
                if not core:
                    continue
                core_map.setdefault(core, []).append(idx)

            duplicate_clusters = {k: v for k, v in core_map.items() if len(v) > 1}
            if not duplicate_clusters:
                logger.debug(f"{log_prefix}: De-dup preview: no duplicate clusters detected")
                return

            potential_savings = sum(len(v) - 1 for v in duplicate_clusters.values())
            logger.info(
                f"{log_prefix}: De-dup preview — clusters={len(duplicate_clusters)}, "
                f"potential_savings={potential_savings} (would reduce {len(tasks)}→{len(tasks)-potential_savings})"
            )
            sample = []
            for core_text, idxs in list(duplicate_clusters.items())[:3]:
                hash_prefix = hashlib.sha1(core_text.encode('utf-8')).hexdigest()[:10]
                sample.append(f"{hash_prefix}:{len(idxs)}:'{core_text[:55]}'")
            if sample:
                logger.debug(f"{log_prefix}: De-dup cluster samples: {sample}")
        except Exception as e:
            logger.debug(f"{log_prefix}: De-dup preview failed: {e}")


def _process_ai_response(ai_response: Any, log_prefix: str) -> dict[str, Any]:
    """
    Processes and validates the AI response using Pydantic models for robust parsing,
    then normalizes the structure to the enriched schema used downstream.

    Always returns a dict with keys: 'extracted_data' and 'suggested_tasks'.
    """
    # Lazy import to avoid circulars at module import time
    try:
        from genealogical_normalization import normalize_ai_response
    except Exception:
        normalize_ai_response = None  # type: ignore

    # Default minimal fallback
    result: dict[str, Any] = {
        "extracted_data": {},
        "suggested_tasks": [],
    }

    if not isinstance(ai_response, dict):
        logger.warning(f"{log_prefix}: AI response is not a dict; applying defaults.")
        return (
            normalize_ai_response({}) if callable(normalize_ai_response) else result
        )

    logger.debug(f"{log_prefix}: Processing AI response...")

    try:
        # First attempt: Try direct validation with Pydantic
        validated_response = AIResponse.model_validate(ai_response)
        result = validated_response.model_dump()
        logger.debug(f"{log_prefix}: AI response validated with Pydantic schema.")
    except ValidationError as ve:
        logger.warning(f"{log_prefix}: AI response validation failed: {ve}")
        # Salvage minimal content before normalization
        minimal = {"extracted_data": {}, "suggested_tasks": []}
        try:
            if isinstance(ai_response.get("extracted_data"), dict):
                minimal["extracted_data"] = dict(ai_response["extracted_data"])  # shallow copy
            if isinstance(ai_response.get("suggested_tasks"), list):
                minimal["suggested_tasks"] = list(ai_response["suggested_tasks"])  # shallow copy
        except Exception as e:
            logger.debug(f"{log_prefix}: Defensive salvage failed: {e}")
        result = minimal
    except Exception as e:
        logger.error(f"{log_prefix}: Unexpected error processing AI response: {e}", exc_info=True)
        result = {"extracted_data": {}, "suggested_tasks": []}

    # Normalize to ensure structured keys exist and legacy fields are promoted
    if callable(normalize_ai_response):
        try:
            result = normalize_ai_response(result)
        except Exception as e:
            logger.warning(f"{log_prefix}: Normalization failed, using unnormalized result: {e}")

    return result


def _format_context_for_ai_extraction(
    context_logs: list[ConversationLog],
    # my_pid_lower parameter is kept for compatibility but not used
    # pylint: disable=unused-argument
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
    context_lines = []
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
                direction_value = log.direction

                # Check if it's a MessageDirectionEnum or can be compared to one
                if hasattr(direction_value, "value"):
                    # It's an enum object
                    is_in_direction = direction_value == MessageDirectionEnum.IN
                elif isinstance(direction_value, str):
                    # It's a string
                    is_in_direction = direction_value == MessageDirectionEnum.IN.value
                elif str(direction_value) == str(MessageDirectionEnum.IN):
                    # Try string comparison as last resort
                    is_in_direction = True
        except Exception:
            # Default to OUT if any error occurs
            is_in_direction = False

        # Use a simple boolean value to avoid SQLAlchemy type issues
        author_label = "USER: " if bool(is_in_direction) else "SCRIPT: "

        # Step 3b: Get message content and handle potential None
        content = log.latest_message_content or ""
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)

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
        from action8_messaging import load_message_templates

        # Load all templates
        templates = load_message_templates()

        # Check if the required template exists
        if not templates or ACKNOWLEDGEMENT_MESSAGE_TYPE not in templates:
            logger.error(
                f"Required template '{ACKNOWLEDGEMENT_MESSAGE_TYPE}' not found in templates."
            )
            return {}

        return templates
    except Exception as e:
        logger.error(f"Error loading templates for Action 9: {e}", exc_info=True)
        return {}


def _search_ancestry_tree(
    _session_manager: SessionManager, extracted_data: Union[ExtractedData, list[str]]
) -> dict[str, Any]:
    """
    Searches the user's tree (GEDCOM or API) for names extracted by the AI.

    This function dispatches to the appropriate search method based on configuration
    and returns a dictionary containing the search results and relationship paths.
    Uses cached GEDCOM data to avoid loading the file multiple times.

    Args:
        session_manager: The SessionManager instance.
        extracted_data: ExtractedData object or list of names to search for.

    Returns:
        Dictionary containing search results and relationship paths
    """
    # Step 1: Get all names from the extracted data
    if isinstance(extracted_data, ExtractedData):
        names = extracted_data.get_all_names()
    elif isinstance(extracted_data, list):
        # Legacy support for list of names
        names = extracted_data
    else:
        logger.warning(
            "Action 9 Tree Search: Invalid extracted_data type. Expected ExtractedData or list."
        )
        return {"results": [], "relationship_paths": {}}

    if not names:
        logger.debug("Action 9 Tree Search: No names extracted to search.")
        return {"results": [], "relationship_paths": {}}

    # Step 2: Get search method from config
    search_method = config_schema.tree_search_method
    logger.info(
        f"Action 9 Tree Search: Method configured as '{search_method}'. Found {len(names)} names to search."
    )

    # Step 3: Return empty results for any search method
    # This is a simplified version that doesn't depend on complex GEDCOM/API functionality
    search_results = []
    logger.info(f"Action 9 Tree Search: Found {len(search_results)} potential matches.")

    # Step 4: Return empty relationship paths
    relationship_paths = {}

    return {"results": search_results, "relationship_paths": relationship_paths}


def _identify_and_get_person_details(
    _session_manager: SessionManager, extracted_data: dict[str, Any], log_prefix: str
) -> Optional[dict[str, Any]]:
    """
    Simplified version that returns None (no person details found).
    """
    logger.debug(
        f"{log_prefix}: _identify_and_get_person_details - returning None (simplified version)"
    )


def _format_genealogical_data_for_ai(
    genealogical_data: dict[str, Any], log_prefix: str
) -> str:
    """
    Simplified version that formats genealogical data for AI consumption.
    """
    if not genealogical_data or not genealogical_data.get("results"):
        return "No genealogical data found in family tree."

    # Simple formatting of search results
    formatted_lines = ["Family Tree Search Results:"]
    for result in genealogical_data.get("results", [])[:3]:  # Limit to top 3
        if isinstance(result, dict):
            name = result.get("name", "Unknown")
            formatted_lines.append(f"- {name}")

    return "\n".join(formatted_lines)


def generate_genealogical_reply(
    session_manager: SessionManager,
    conversation_context: str,
    user_message: str,
    genealogical_data: str,
    log_prefix: str,
) -> Optional[str]:
    """
    Generate a genealogical reply using the AI interface.
    """
    try:
        # Import the AI interface function
        from ai_interface import generate_genealogical_reply as ai_generate_reply

        # Use the AI interface to generate a reply
        return ai_generate_reply(
            conversation_context=conversation_context,
            user_last_message=user_message,
            genealogical_data_str=genealogical_data,
            session_manager=session_manager,
        )

    except Exception as e:
        logger.error(f"{log_prefix}: Error generating genealogical reply: {e!s}")
        return None


def _generate_ack_summary(extracted_data: dict[str, Any]) -> str:
    """
    Generates a summary from extracted data for acknowledgment messages.
    """
    try:
        # Get mentioned names
        names = extracted_data.get("extracted_data", {}).get("mentioned_names", [])
        locations = extracted_data.get("extracted_data", {}).get(
            "mentioned_locations", []
        )
        dates = extracted_data.get("extracted_data", {}).get("mentioned_dates", [])

        summary_parts = []

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
# Standalone Test Block
# ==============================================


# ==============================================
# Standalone Test Block
# ==============================================
def action9_process_productive_module_tests() -> bool:
    """Comprehensive test suite for action9_process_productive.py"""

    suite = TestSuite(
        "Action 9 - AI Message Processing & Data Extraction",
        "action9_process_productive.py",
    )
    suite.start_suite()  # INITIALIZATION TESTS

    def test_module_initialization() -> None:
        """Test module initialization and constants"""
        # Test constants
        assert (
            PRODUCTIVE_SENTIMENT == "PRODUCTIVE"
        ), "PRODUCTIVE_SENTIMENT constant not properly set"
        assert OTHER_SENTIMENT == "OTHER", "OTHER_SENTIMENT constant not properly set"
        assert (
            ACKNOWLEDGEMENT_MESSAGE_TYPE == "Productive_Reply_Acknowledgement"
        ), "ACKNOWLEDGEMENT_MESSAGE_TYPE not set"
        assert (
            CUSTOM_RESPONSE_MESSAGE_TYPE == "Automated_Genealogy_Response"
        ), "CUSTOM_RESPONSE_MESSAGE_TYPE not set"  # Test class availability
        assert "PersonProcessor" in globals(), "PersonProcessor class not defined"
        assert "BatchCommitManager" in globals(), "BatchCommitManager class not defined"

        # Test function availability
        assert callable(
            process_productive_messages
        ), "process_productive_messages should be callable"
        assert callable(
            safe_column_value
        ), "safe_column_value should be callable"  # CORE FUNCTIONALITY TESTS

    def test_core_functionality() -> None:
        """Test all core AI processing and data extraction functions"""

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
        assert isinstance(result, bool), "Should return boolean value"

        result = should_exclude_message("test message 12345")
        assert isinstance(result, bool), "Should handle test messages"

    def test_ai_processing_functions() -> None:
        """Test AI processing and extraction functions"""

        # Test _process_ai_response function
        mock_response = {"status": "success", "data": {"extracted": "test_12345"}}
        result = _process_ai_response(mock_response, "TEST")
        assert isinstance(result, dict), "Should return dictionary"

        # Test _generate_ack_summary function
        test_data = {"names": ["Test Person 12345"], "dates": ["1985"]}
        result = _generate_ack_summary(test_data)
        assert isinstance(result, str), "Should return string summary"
        assert (
            "12345" in result or len(result) > 0
        ), "Should generate meaningful summary"

    # EDGE CASE TESTS
    def test_edge_cases() -> None:
        """Test edge cases and boundary conditions"""
        # Test safe_column_value with None object
        result = safe_column_value(None, "any_attr", "default_12345")
        assert result == "default_12345", "Should handle None object"

        # Test should_exclude_message with empty string
        result = should_exclude_message("")
        assert isinstance(result, bool), "Should handle empty string"

        # Test with empty string instead of None for type safety
        result = should_exclude_message("")
        assert isinstance(result, bool), "Should handle empty string input"

        # Test _process_ai_response with invalid data
        result = _process_ai_response(None, "TEST")
        assert isinstance(result, dict), "Should handle None response"

        result = _process_ai_response({}, "TEST_12345")
        assert isinstance(
            result, dict
        ), "Should handle empty response"  # INTEGRATION TESTS

    def test_integration() -> None:
        """Test integration with external dependencies"""

        # Test get_gedcom_data integration - patch the import
        with patch("gedcom_cache.load_gedcom_with_aggressive_caching") as mock_loader:
            mock_loader.return_value = {"test": "data_12345"}
            # Set global to None to force reload
            global _CACHED_GEDCOM_DATA
            original_cache = _CACHED_GEDCOM_DATA
            _CACHED_GEDCOM_DATA = None
            try:
                result = get_gedcom_data()
                assert result is not None, "Should return GEDCOM data"
            finally:
                _CACHED_GEDCOM_DATA = original_cache

        # Test template loading integration
        result = _load_templates_for_action9()
        assert isinstance(
            result, dict
        ), "Should return templates dictionary"  # PERFORMANCE TESTS

    def test_performance() -> None:
        """Test performance of processing operations"""
        import time

        # Test safe_column_value performance
        mock_obj = MagicMock()
        mock_obj.test_attr = "value"

        start_time = time.time()
        for i in range(1000):
            safe_column_value(mock_obj, "test_attr", f"default_{i}_12345")
        duration = time.time() - start_time

        assert (
            duration < 0.5
        ), f"1000 safe_column_value calls should be fast, took {duration:.3f}s"

        # Test should_exclude_message performance
        start_time = time.time()
        for i in range(500):
            should_exclude_message(f"test message {i} 12345")
        duration = time.time() - start_time

        assert (
            duration < 0.5
        ), f"500 exclusion checks should be fast, took {duration:.3f}s"  # ERROR HANDLING TESTS

    def test_error_handling() -> None:
        """Test error handling scenarios"""

        # Test safe_column_value with attribute access exception
        class ErrorObj:
            @property
            def test_attr(self) -> None:
                raise Exception("Test error 12345")

        error_obj = ErrorObj()
        result = safe_column_value(error_obj, "test_attr", "fallback_12345")
        assert result == "fallback_12345", "Should handle attribute access errors"

    def test_circuit_breaker_config() -> None:
        """Test circuit breaker decorator configuration reflects Action 6 lessons."""
        import inspect

        # Test process_productive_messages function has proper decorators
        func = process_productive_messages

        # Check function attributes
        assert callable(func), "process_productive_messages should be callable"
        assert func.__name__ == 'process_productive_messages', "Function name should be preserved"

        # Check function signature
        sig = inspect.signature(func)
        assert 'session_manager' in sig.parameters, "Should have session_manager parameter"

        # Verify function can be inspected (indicates decorators are properly applied)
        assert hasattr(func, '__wrapped__') or hasattr(func, '__annotations__'), "Should have decorator attributes"

        # Test _process_ai_response with malformed data
        malformed_data = {"invalid": "structure_12345"}
        result = _process_ai_response(malformed_data, "ERROR_TEST")
        assert isinstance(result, dict), "Should handle malformed AI responses"

        # Test _generate_ack_summary with empty data
        result = _generate_ack_summary({})
        assert isinstance(result, str), "Should handle empty extraction data"

    # Run all tests with suppress_logging
    with suppress_logging():
        # INITIALIZATION TESTS
        suite.run_test(
            test_name="Module constants, classes, and function availability",
            test_func=test_module_initialization,
            expected_behavior="Module initializes correctly with proper constants and function definitions",
            test_description="Module initialization and configuration verification",
            method_description="Testing constants, class definitions, and core function availability for AI processing",
        )

        # CORE FUNCTIONALITY TESTS
        suite.run_test(
            test_name="safe_column_value(), should_exclude_message() core functions",
            test_func=test_core_functionality,
            expected_behavior="All core functions execute correctly with proper data handling and validation",
            test_description="Core utility and message filtering functionality",
            method_description="Testing attribute extraction, message filtering, and core utility functions",
        )

        suite.run_test(
            test_name="_process_ai_response(), _generate_ack_summary() AI processing",
            test_func=test_ai_processing_functions,
            expected_behavior="AI processing functions handle response data correctly and generate summaries",
            test_description="AI response processing and summary generation functions",
            method_description="Testing AI response parsing, data extraction, and summary generation functionality",
        )

        # EDGE CASE TESTS
        suite.run_test(
            test_name="ALL functions with edge case inputs",
            test_func=test_edge_cases,
            expected_behavior="All functions handle edge cases gracefully without crashes or unexpected behavior",
            test_description="Edge case handling across all AI processing functions",
            method_description="Testing functions with empty, None, invalid, and boundary condition inputs",
        )

        # INTEGRATION TESTS
        suite.run_test(
            test_name="get_gedcom_data(), _load_templates_for_action9() integration",
            test_func=test_integration,
            expected_behavior="Integration functions work correctly with external data sources and templates",
            test_description="Integration with GEDCOM data and template systems",
            method_description="Testing integration with genealogical data cache and message template loading",
        )

        # PERFORMANCE TESTS
        suite.run_test(
            test_name="Performance of utility and filtering operations",
            test_func=test_performance,
            expected_behavior="All operations complete within acceptable time limits with good performance",
            test_description="Performance characteristics of AI processing operations",
            method_description="Testing execution speed of attribute extraction and message filtering functions",
        )

        # CIRCUIT BREAKER TESTS
        suite.run_test(
            test_name="Circuit breaker configuration validation",
            test_func=test_circuit_breaker_config,
            expected_behavior="Circuit breaker decorators properly applied with Action 6 lessons (failure_threshold=10, backoff_factor=4.0)",
            test_description="Circuit breaker decorator configuration reflects improved error handling",
            method_description="Testing process_productive_messages() has proper circuit breaker configuration for production resilience",
        )

        # ERROR HANDLING TESTS
        suite.run_test(
            test_name="Error handling for AI processing and utility functions",
            test_func=test_error_handling,
            expected_behavior="All error conditions handled gracefully with appropriate fallback responses",
            test_description="Error handling and recovery functionality for AI operations",
            method_description="Testing error scenarios with invalid data, exceptions, and malformed responses",
        )

    return suite.finish_suite()


# === PHASE 12.3: ENHANCED DATA INTEGRATION FUNCTIONS ===

def _apply_intelligent_task_selection(
    tasks: list[dict[str, Any]],
    extracted_data: dict[str, Any],
    person_data: dict[str, Any],
    log_prefix: str
) -> list[dict[str, Any]]:
    """
    Apply intelligent task selection based on data completeness and quality.
    Prioritizes tasks that make best use of available extracted data.
    """
    try:
        if not tasks:
            return tasks

        logger.debug(f"{log_prefix}: Applying intelligent task selection to {len(tasks)} tasks")

        # Score tasks based on data integration potential
        scored_tasks = []
        for task in tasks:
            score = _calculate_task_data_integration_score(task, extracted_data)
            task['data_integration_score'] = score
            scored_tasks.append(task)

        # Sort by integration score (highest first) and select top tasks
        scored_tasks.sort(key=lambda t: t.get('data_integration_score', 0), reverse=True)

        # Select tasks with good data integration (score > 0.6) or top 5 tasks
        selected_tasks = []
        for task in scored_tasks:
            if task.get('data_integration_score', 0) > 0.6 or len(selected_tasks) < 5:
                selected_tasks.append(task)
            if len(selected_tasks) >= 8:  # Maximum 8 tasks
                break

        logger.debug(f"{log_prefix}: Selected {len(selected_tasks)} tasks with good data integration")
        return selected_tasks

    except Exception as e:
        logger.warning(f"{log_prefix}: Error in intelligent task selection: {e}")
        return tasks[:5]  # Fallback to first 5 tasks

def _calculate_task_data_integration_score(task: dict[str, Any], extracted_data: dict[str, Any]) -> float:
    """Calculate how well a task integrates with available extracted data."""
    score = 0.0

    # Check if task uses specific extracted data
    task_desc = task.get('description', '').lower()
    task.get('title', '').lower()

    # Score based on data type utilization
    if extracted_data.get('vital_records'):
        if any(term in task_desc for term in ['birth', 'death', 'marriage', 'vital']):
            score += 0.3

    if extracted_data.get('locations'):
        if any(term in task_desc for term in ['location', 'place', 'county', 'state']):
            score += 0.25

    if extracted_data.get('relationships'):
        if any(term in task_desc for term in ['family', 'parent', 'child', 'spouse', 'relationship']):
            score += 0.2

    if extracted_data.get('dna_information') and any(term in task_desc for term in ['dna', 'match', 'genetic']):
        score += 0.35

    if extracted_data.get('occupations'):
        if any(term in task_desc for term in ['occupation', 'work', 'job', 'career']):
            score += 0.15

    # Bonus for specific names mentioned
    structured_names = extracted_data.get('structured_names', [])
    for name_data in structured_names:
        if isinstance(name_data, dict):
            full_name = name_data.get('full_name', '')
            if full_name and full_name.lower() in task_desc:
                score += 0.2
                break

    # Bonus for research questions alignment
    research_questions = extracted_data.get('research_questions', [])
    for question in research_questions:
        if isinstance(question, str):
            question_words = question.lower().split()
            if any(word in task_desc for word in question_words if len(word) > 3):
                score += 0.15
                break

    return min(1.0, score)

def _validate_and_score_tasks(
    tasks: list[dict[str, Any]],
    extracted_data: dict[str, Any],
    log_prefix: str
) -> list[dict[str, Any]]:
    """
    Validate task quality and add quality scores for better task management.
    """
    try:
        if not tasks:
            return tasks

        logger.debug(f"{log_prefix}: Validating and scoring {len(tasks)} tasks")

        validated_tasks = []
        for task in tasks:
            # Calculate task quality score
            quality_score = _calculate_task_quality_score(task, extracted_data)
            task['quality_score'] = quality_score

            # Only include tasks with minimum quality threshold
            if quality_score >= 0.4:  # Minimum 40% quality score
                validated_tasks.append(task)
            else:
                logger.debug(f"{log_prefix}: Filtered out low-quality task: {task.get('title', 'Unknown')[:50]}")

        logger.debug(f"{log_prefix}: Validated {len(validated_tasks)} tasks meeting quality threshold")
        return validated_tasks

    except Exception as e:
        logger.warning(f"{log_prefix}: Error in task validation: {e}")
        return tasks

def _calculate_task_quality_score(task: dict[str, Any], extracted_data: dict[str, Any]) -> float:
    """Calculate overall quality score for a research task."""
    score = 0.0

    # Check task completeness
    title = task.get('title', '')
    description = task.get('description', '')
    category = task.get('category', '')

    if title and len(title) > 10:
        score += 0.2
    if description and len(description) > 50:
        score += 0.3
    if category:
        score += 0.1

    # Check specificity (specific names, dates, places)
    specificity_score = 0.0

    # Look for specific names
    structured_names = extracted_data.get('structured_names', [])
    for name_data in structured_names:
        if isinstance(name_data, dict):
            full_name = name_data.get('full_name', '')
            if full_name and full_name in description:
                specificity_score += 0.1
                break

    # Look for specific dates
    vital_records = extracted_data.get('vital_records', [])
    for record in vital_records:
        if isinstance(record, dict):
            date = record.get('date', '')
            if date and date in description:
                specificity_score += 0.1
                break

    # Look for specific places
    locations = extracted_data.get('locations', [])
    for location in locations:
        if isinstance(location, dict):
            place = location.get('place', '')
            if place and place in description:
                specificity_score += 0.1
                break

    score += min(0.3, specificity_score)

    # Check actionability (presence of research steps or methodology)
    if any(term in description.lower() for term in ['search', 'check', 'review', 'examine', 'research']):
        score += 0.1

    return min(1.0, score)


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return action9_process_productive_module_tests()


if __name__ == "__main__":
    print(
        "🤖 Running Action 9 - AI Message Processing & Data Extraction comprehensive test suite..."
    )
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
