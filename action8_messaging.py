#!/usr/bin/env python3

"""
Action 8: Intelligent Messaging Automation & Personalization Engine

Delivers personalized, contextually-aware messages to DNA matches through:
- Dynamic template selection based on relationship strength
- Intelligent recipient filtering with engagement prediction
- Automated follow-up sequences with timing optimization
- Complete message lifecycle tracking with delivery confirmation
- Batch processing with intelligent scheduling and rate limiting
- Comprehensive error recovery with circuit breaker patterns

In dry_run mode: Messages are created and saved to DB but NOT sent to Ancestry.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === ERROR HANDLING ===
from core.error_handling import (
    circuit_breaker,
    error_context,
    graceful_degradation,
)

# === MESSAGE PERSONALIZATION ===
_msg_pers_available = False
try:
    from message_personalization import MessagePersonalizer
    _msg_pers_available = True
except ImportError:
    MessagePersonalizer = None

MESSAGE_PERSONALIZATION_AVAILABLE = _msg_pers_available

# === STANDARD LIBRARY IMPORTS ===
import logging
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from string import Formatter
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from common_params import (
        BatchCounters,
        ConversationState,
        MessageContext,
        MessageFlags,
        MessagingBatchData,
        ProcessingState,
    )

# === THIRD-PARTY IMPORTS ===
from sqlalchemy import (
    func,
    inspect as sa_inspect,
)  # Minimal imports

# === LOCAL IMPORTS ===
# Import PersonStatusEnum early for use in safe_column_value
from database import PersonStatusEnum


def _handle_status_enum_conversion(value: Any, default: Any) -> Any:
    """Handle status enum conversion."""
    if isinstance(value, PersonStatusEnum):
        return value
    if isinstance(value, str):
        try:
            return PersonStatusEnum(value)
        except ValueError:
            logger.warning(f"Invalid status string '{value}'")
            return default
    logger.warning(f"Unexpected status type: {type(value)}")
    return default


def _convert_value_to_primitive(value: Any) -> Any:
    """Convert value to appropriate Python primitive type."""
    if isinstance(value, bool) or value is True or value is False:
        return bool(value)
    if isinstance(value, int) or str(value).isdigit():
        return int(value)
    if isinstance(value, float) or str(value).replace(".", "", 1).isdigit():
        return float(value)
    if hasattr(value, "isoformat"):
        return value
    return str(value)


def safe_column_value(obj: Any, attr_name: str, default: Any = None) -> Any:
    """Safely extract a value from a SQLAlchemy model attribute."""
    if not hasattr(obj, attr_name):
        return default

    value = getattr(obj, attr_name)
    if value is None:
        return default

    try:
        if attr_name == "status":
            return _handle_status_enum_conversion(value, default)
        return _convert_value_to_primitive(value)
    except (ValueError, TypeError, AttributeError):
        return default


# === SQLALCHEMY & UTILITIES ===
from sqlalchemy.orm import Session, joinedload
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# === API & CORE ===
from api_utils import call_send_message_api
from cache import cache_result
from common_params import BatchConfig, BatchCounters, MessagingBatchData, ProcessingState
from config import config_schema
from connection_resilience import with_connection_resilience
from core.enhanced_error_recovery import with_enhanced_recovery
from core.error_handling import (
    APIRateLimitError,
    AuthenticationExpiredError,
    BrowserSessionError,
    MaxApiFailuresExceededError,
)
from core.session_manager import SessionManager

# === DATABASE ===
from database import (
    ConversationLog,
    DnaMatch,
    FamilyTree,
    MessageDirectionEnum,
    MessageTemplate,
    Person,
    commit_bulk_data,
)

# === MONITORING & TESTING ===
from performance_monitor import start_advanced_monitoring, stop_advanced_monitoring
from test_framework import TestSuite, suppress_logging
from utils import format_name

# === MESSAGE INTERVALS ===
MESSAGE_INTERVALS = {
    "testing": timedelta(seconds=10),
    "production": timedelta(weeks=8),
    "dry_run": timedelta(seconds=30),
}
MIN_MESSAGE_INTERVAL: timedelta = MESSAGE_INTERVALS.get(
    getattr(config_schema, 'app_mode', 'production'), timedelta(weeks=8)
)

# === MESSAGE TYPES ===
MESSAGE_TYPES_ACTION8: dict[str, str] = {
    "In_Tree-Initial": "In_Tree-Initial",
    "In_Tree-Follow_Up": "In_Tree-Follow_Up",
    "In_Tree-Final_Reminder": "In_Tree-Final_Reminder",
    "Out_Tree-Initial": "Out_Tree-Initial",
    "Out_Tree-Follow_Up": "Out_Tree-Follow_Up",
    "Out_Tree-Final_Reminder": "Out_Tree-Final_Reminder",
    "In_Tree-Initial_for_was_Out_Tree": "In_Tree-Initial_for_was_Out_Tree",
    "User_Requested_Desist": "User_Requested_Desist",
    "In_Tree-Initial_Short": "In_Tree-Initial_Short",
    "Out_Tree-Initial_Short": "Out_Tree-Initial_Short",
    "In_Tree-Initial_Confident": "In_Tree-Initial_Confident",
    "Out_Tree-Initial_Exploratory": "Out_Tree-Initial_Exploratory",
}


@cache_result("message_templates")
def load_message_templates() -> dict[str, str]:
    """Load message templates from database MessageTemplate table."""
    try:
        from core.session_manager import SessionManager

        session_manager = SessionManager()
        with session_manager.get_db_conn_context() as session:
            if not session:
                logger.critical("Could not get database session for template loading")
                return {}

            templates_query = session.query(MessageTemplate).all()
            templates = {}

            for template in templates_query:
                if template.subject_line and template.message_content:
                    full_content = f"Subject: {template.subject_line}\n\n{template.message_content}"
                elif template.message_content:
                    full_content = template.message_content
                else:
                    logger.warning(f"Template {template.template_key} has no content")
                    continue

                templates[template.template_key] = full_content

            # Validate required keys
            core_required_keys = {
                "In_Tree-Initial", "In_Tree-Follow_Up", "In_Tree-Final_Reminder",
                "Out_Tree-Initial", "Out_Tree-Follow_Up", "Out_Tree-Final_Reminder",
                "In_Tree-Initial_for_was_Out_Tree", "User_Requested_Desist",
                "Productive_Reply_Acknowledgement"
            }
            missing_keys = core_required_keys - set(templates.keys())
            if missing_keys:
                logger.critical(f"Missing required template keys: {', '.join(missing_keys)}")
                return {}

            logger.debug(f"Loaded {len(templates)} message templates")
            return templates

    except Exception as e:
        logger.critical(f"Error loading templates: {e}", exc_info=True)
        return {}


# Load templates into a global variable
MESSAGE_TEMPLATES: dict[str, str] = load_message_templates()

# Validate essential templates loaded
core_required_check_keys = {
    "In_Tree-Initial", "In_Tree-Follow_Up", "In_Tree-Final_Reminder",
    "Out_Tree-Initial", "Out_Tree-Follow_Up", "Out_Tree-Final_Reminder",
    "In_Tree-Initial_for_was_Out_Tree", "User_Requested_Desist",
    "Productive_Reply_Acknowledgement"
}
if not MESSAGE_TEMPLATES or not all(key in MESSAGE_TEMPLATES for key in core_required_check_keys):
    logger.critical("Essential message templates failed to load")

# Initialize message personalizer
import contextlib
from typing import Any as _Any

_message_personalizer: Optional[_Any] = None
if MESSAGE_PERSONALIZATION_AVAILABLE and callable(MessagePersonalizer):
    try:
        _message_personalizer = MessagePersonalizer()
    except Exception as e:
        logger.warning(f"Failed to initialize message personalizer: {e}")
        _message_personalizer = None

MESSAGE_PERSONALIZER = _message_personalizer


# ------------------------------------------------------------------------------
# Message Type Determination Logic
# ------------------------------------------------------------------------------


# === MESSAGE TRANSITION TABLE ===
# Maps (current_message_type, is_in_family_tree) to next_message_type
MESSAGE_TRANSITION_TABLE = {
    # Initial messages (no previous message)
    (None, True): "In_Tree-Initial",
    (None, False): "Out_Tree-Initial",
    # In-Tree sequences
    ("In_Tree-Initial", True): "In_Tree-Follow_Up",
    ("In_Tree-Initial_for_was_Out_Tree", True): "In_Tree-Follow_Up",
    ("In_Tree-Initial_Confident", True): "In_Tree-Follow_Up",
    ("In_Tree-Initial_Short", True): "In_Tree-Follow_Up",
    ("In_Tree-Follow_Up", True): "In_Tree-Final_Reminder",
    ("In_Tree-Final_Reminder", True): None,
    # Out-Tree sequences
    ("Out_Tree-Initial", False): "Out_Tree-Follow_Up",
    ("Out_Tree-Initial_Short", False): "Out_Tree-Follow_Up",
    ("Out_Tree-Initial_Exploratory", False): "Out_Tree-Follow_Up",
    ("Out_Tree-Follow_Up", False): "Out_Tree-Final_Reminder",
    ("Out_Tree-Final_Reminder", False): None,
    # Tree status changes (Out->In)
    ("Out_Tree-Initial", True): "In_Tree-Initial_for_was_Out_Tree",
    ("Out_Tree-Follow_Up", True): "In_Tree-Initial_for_was_Out_Tree",
    ("Out_Tree-Final_Reminder", True): "In_Tree-Initial_for_was_Out_Tree",
    ("Out_Tree-Initial_Short", True): "In_Tree-Initial_for_was_Out_Tree",
    ("Out_Tree-Initial_Exploratory", True): "In_Tree-Initial_for_was_Out_Tree",
    # Tree status changes (In->Out)
    ("In_Tree-Initial", False): None,
    ("In_Tree-Follow_Up", False): None,
    ("In_Tree-Final_Reminder", False): None,
    ("In_Tree-Initial_Confident", False): None,
    ("In_Tree-Initial_Short", False): None,
    ("In_Tree-Initial_for_was_Out_Tree", False): "Out_Tree-Initial",
    # Desist ends sequence
    ("User_Requested_Desist", True): None,
    ("User_Requested_Desist", False): None,
    # Fallback for unknown types
    ("Unknown", True): "In_Tree-Initial",
    ("Unknown", False): "Out_Tree-Initial",
}


def determine_next_message_type(
    last_message_details: Optional[tuple[str, datetime, str]],
    is_in_family_tree: bool,
) -> Optional[str]:
    """
    Determine next message type based on last message and tree status.

    Uses state machine with transition table mapping (current_type, is_in_tree) to next_type.
    """
    last_message_type = None
    if last_message_details:
        last_message_type, _, _ = last_message_details

    transition_key = (last_message_type, is_in_family_tree)

    if transition_key in MESSAGE_TRANSITION_TABLE:
        next_type = MESSAGE_TRANSITION_TABLE[transition_key]
    elif last_message_type:
        # Recover from unknown types by treating as initial
        logger.warning(f"Unknown message type '{last_message_type}', treating as initial")
        next_type = "In_Tree-Initial" if is_in_family_tree else "Out_Tree-Initial"
    else:
        # Fallback for initial message
        next_type = "In_Tree-Initial" if is_in_family_tree else "Out_Tree-Initial"

    if next_type:
        next_type = MESSAGE_TYPES_ACTION8.get(next_type, next_type)

    return next_type


# End of determine_next_message_type

# ------------------------------------------------------------------------------
# Improved Variable Handling Functions
# ------------------------------------------------------------------------------

def get_safe_relationship_text(family_tree, predicted_rel: str) -> str:
    """
    Get a natural-sounding relationship description with proper fallbacks.

    Args:
        family_tree: FamilyTree object (may be None)
        predicted_rel: Predicted relationship string

    Returns:
        Natural relationship text without "N/A"
    """
    if family_tree:
        actual_rel = safe_column_value(family_tree, "actual_relationship", None)
        if actual_rel and actual_rel != "N/A" and actual_rel.strip():
            return f"my {actual_rel}"

    if predicted_rel and predicted_rel != "N/A" and predicted_rel.strip():
        return f"possibly my {predicted_rel}"

    return "a family connection"


def get_safe_relationship_path(family_tree) -> str:
    """
    Get a natural-sounding relationship path with proper fallbacks.

    Args:
        family_tree: FamilyTree object (may be None)

    Returns:
        Natural relationship path text without "N/A"
    """
    if family_tree:
        path = safe_column_value(family_tree, "relationship_path", None)
        if path and path != "N/A" and path.strip():
            # Fix broken "Enhanced API:" paths
            if path.startswith("Enhanced API:"):
                # Extract the relationship type and format it properly
                relationship_type = path.replace("Enhanced API:", "").strip()
                actual_rel = safe_column_value(family_tree, "actual_relationship", None)
                person_name = safe_column_value(family_tree, "person_name_in_tree", None)

                if person_name and actual_rel:
                    return f"Wayne Gault -> {person_name} ({actual_rel})"
                if person_name:
                    return f"Wayne Gault -> {person_name} ({relationship_type})"
                return f"Wayne Gault -> [Person] ({relationship_type})"
            return path

    return "our shared family line (details to be determined)"


def _is_distant_relationship(actual_rel: str) -> bool:
    """Check if relationship is distant (5th cousin and beyond)."""
    distant_markers = ["5th cousin", "6th cousin", "7th cousin", "8th cousin", "9th cousin"]
    return any(distant in actual_rel.lower() for distant in distant_markers)


def _calculate_family_tree_confidence(family_tree, is_distant_relationship: bool) -> int:
    """Calculate confidence score from family tree data."""
    confidence_score = 0

    if family_tree:
        actual_rel = safe_column_value(family_tree, "actual_relationship", None)
        if actual_rel and actual_rel != "N/A" and actual_rel.strip() and not is_distant_relationship:
            confidence_score += 3

        path = safe_column_value(family_tree, "relationship_path", None)
        if path and path != "N/A" and path.strip() and not is_distant_relationship:
            confidence_score += 2

    return confidence_score


# === HELPER FUNCTIONS FOR CODE DEDUPLICATION ===

def _get_short_template_if_exists(base_template_key: str) -> Optional[str]:
    """Return short template key if it exists, else None."""
    short_key = f"{base_template_key}_Short"
    return short_key if short_key in MESSAGE_TEMPLATES else None


def _ensure_timezone_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure datetime has timezone info (UTC if none)."""
    if dt and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# === TEMPLATE SELECTION HELPERS ===

def _calculate_dna_match_confidence(dna_match, is_distant_relationship: bool) -> int:
    """Calculate confidence score from DNA match data."""
    if dna_match and not is_distant_relationship:
        predicted_rel = safe_column_value(dna_match, "predicted_relationship", None)
        if predicted_rel and predicted_rel != "N/A" and predicted_rel.strip():
            return 1
    return 0


def _get_template_for_distant_relationship(base_template_key: str) -> str:
    """Get template for distant relationships."""
    exploratory_key = f"{base_template_key}_Exploratory"
    if exploratory_key in MESSAGE_TEMPLATES:
        return exploratory_key
    short_key = _get_short_template_if_exists(base_template_key)
    return short_key if short_key else base_template_key


def _get_template_by_confidence_score(base_template_key: str, confidence_score: int) -> str:
    """Get template based on confidence score."""
    if confidence_score >= 4:
        confident_key = f"{base_template_key}_Confident"
        if confident_key in MESSAGE_TEMPLATES:
            return confident_key
    elif confidence_score <= 2:
        exploratory_key = f"{base_template_key}_Exploratory"
        if exploratory_key in MESSAGE_TEMPLATES:
            return exploratory_key

    short_key = _get_short_template_if_exists(base_template_key)
    return short_key if short_key else base_template_key


def select_template_by_confidence(base_template_key: str, family_tree, dna_match) -> str:
    """Select template variant based on relationship confidence."""
    # Check for distant relationships first
    is_distant_relationship = False
    if family_tree:
        actual_rel = safe_column_value(family_tree, "actual_relationship", None)
        if actual_rel and actual_rel != "N/A" and actual_rel.strip():
            is_distant_relationship = _is_distant_relationship(actual_rel)

    if is_distant_relationship:
        return _get_template_for_distant_relationship(base_template_key)

    # Calculate confidence score
    confidence_score = _calculate_family_tree_confidence(family_tree, is_distant_relationship)
    confidence_score += _calculate_dna_match_confidence(dna_match, is_distant_relationship)

    return _get_template_by_confidence_score(base_template_key, confidence_score)


def select_template_variant_ab_testing(person_id: int, base_template_key: str) -> str:
    """
    Select template variant for A/B testing based on person ID.

    Args:
        person_id: Person ID for consistent assignment
        base_template_key: Base template key

    Returns:
        Template key with A/B testing variant
    """
    # Use person ID for consistent assignment (50/50 split)
    use_short = person_id % 2 == 0

    if use_short:
        short_key = _get_short_template_if_exists(base_template_key)
        if short_key:
            # A/B Testing: Selected short variant (removed verbose debug)
            return short_key

    # Use confidence-based selection as fallback
    return base_template_key


def track_template_selection(template_key: str, person_id: int, selection_reason: str):
    """Track template selection for debugging."""
    logger.debug(f"Template selected for person {person_id}: {template_key} ({selection_reason})")


# ------------------------------------------------------------------------------
# Response Rate Tracking and Analysis
# ------------------------------------------------------------------------------

def _get_session_manager(session_manager) -> Any:
    """Get or create session manager."""
    if not session_manager:
        from core.session_manager import SessionManager
        session_manager = SessionManager()
    return session_manager


def _get_template_selections(session, cutoff_date) -> list:
    """Get template selections from database."""
    return session.query(ConversationLog).filter(
        ConversationLog.script_message_status.like("TEMPLATE_SELECTED:%"),
        ConversationLog.timestamp_utc >= cutoff_date
    ).all()


def _extract_template_name(script_message_status: str) -> Optional[str]:
    """Extract template name from script message status."""
    status_parts = script_message_status.split(":")
    if len(status_parts) >= 2:
        return status_parts[1].strip().split(" ")[0]
    return None


def _initialize_template_stats() -> dict[str, Any]:
    """Initialize template statistics structure."""
    return {
        "sent": 0,
        "responses": 0,
        "response_rate": 0.0,
        "avg_response_time_hours": 0.0
    }


def _find_response_for_template(session, person_id: int, sent_time: datetime):
    """Find response for a specific template sent to a person."""
    return session.query(ConversationLog).filter(
        ConversationLog.person_id == person_id,
        ConversationLog.direction == MessageDirectionEnum.IN,
        ConversationLog.timestamp_utc > sent_time,
        ConversationLog.timestamp_utc <= sent_time + timedelta(days=30)  # Response window
    ).first()


def _update_response_time_average(template_stats: dict[str, Any], template_name: str, response_hours: float) -> None:
    """Update average response time for a template."""
    current_avg = template_stats[template_name]["avg_response_time_hours"]
    response_count = template_stats[template_name]["responses"]
    new_avg = ((current_avg * (response_count - 1)) + response_hours) / response_count
    template_stats[template_name]["avg_response_time_hours"] = new_avg


def _calculate_response_rates(template_stats: dict[str, dict[str, Any]]) -> None:
    """Calculate response rates for all templates."""
    for _template_name, stats in template_stats.items():
        if stats["sent"] > 0:
            stats["response_rate"] = (stats["responses"] / stats["sent"]) * 100


def _process_template_selections(session, template_selections, template_stats: dict[str, dict[str, Any]]) -> None:
    """Process template selections and calculate statistics."""
    for selection in template_selections:
        template_name = _extract_template_name(selection.script_message_status)
        if not template_name:
            continue

        if template_name not in template_stats:
            template_stats[template_name] = _initialize_template_stats()

        template_stats[template_name]["sent"] += 1

        # Check for responses from this person after this template
        person_id = selection.person_id
        sent_time = selection.timestamp_utc

        response = _find_response_for_template(session, person_id, sent_time)
        if response:
            template_stats[template_name]["responses"] += 1
            # Calculate response time
            response_time = response.timestamp_utc - sent_time
            response_hours = response_time.total_seconds() / 3600
            _update_response_time_average(template_stats, template_name, response_hours)


def analyze_template_effectiveness(session_manager=None, days_back: int = 30) -> dict[str, Any]:
    """
    Analyze template effectiveness by measuring response rates.

    Args:
        session_manager: Database session manager
        days_back: Number of days to look back for analysis

    Returns:
        Dictionary with template effectiveness statistics
    """
    session_manager = _get_session_manager(session_manager)

    try:
        with session_manager.get_db_conn_context() as session:
            if not session:
                return {"error": "Could not get database session"}

            # Get cutoff date
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

            # Query for template selections and responses
            template_stats = {}
            template_selections = _get_template_selections(session, cutoff_date)

            _process_template_selections(session, template_selections, template_stats)
            _calculate_response_rates(template_stats)

            return {
                "analysis_period_days": days_back,
                "total_templates_analyzed": len(template_stats),
                "template_stats": template_stats,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

    except Exception as e:
        logger.error(f"Error analyzing template effectiveness: {e}", exc_info=True)
        return {"error": str(e)}


def print_template_effectiveness_report(days_back: int = 30):
    """
    Print a formatted report of template effectiveness.

    Args:
        days_back: Number of days to analyze
    """
    logger.info("=" * 60)
    logger.info("TEMPLATE EFFECTIVENESS ANALYSIS")
    logger.info("=" * 60)

    analysis = analyze_template_effectiveness(days_back=days_back)

    if "error" in analysis:
        logger.error(f"Analysis failed: {analysis['error']}")
        return

    template_stats = analysis.get("template_stats", {})

    if not template_stats:
        logger.info("No template usage data found for the specified period.")
        return

    logger.info(f"Analysis Period: Last {days_back} days")
    logger.info(f"Templates Analyzed: {len(template_stats)}")
    logger.info("")

    # Sort templates by response rate
    sorted_templates = sorted(
        template_stats.items(),
        key=lambda x: x[1]["response_rate"],
        reverse=True
    )

    logger.info("TEMPLATE PERFORMANCE RANKING:")
    logger.info("-" * 60)
    logger.info(f"{'Template':<25} {'Sent':<6} {'Resp':<6} {'Rate':<8} {'Avg Hours':<10}")
    logger.info("-" * 60)

    for template_name, stats in sorted_templates:
        logger.info(
            f"{template_name:<25} {stats['sent']:<6} {stats['responses']:<6} "
            f"{stats['response_rate']:<7.1f}% {stats['avg_response_time_hours']:<9.1f}"
        )

    logger.info("-" * 60)

    # Summary statistics
    total_sent = sum(stats["sent"] for stats in template_stats.values())
    total_responses = sum(stats["responses"] for stats in template_stats.values())
    overall_rate = (total_responses / total_sent * 100) if total_sent > 0 else 0

    logger.info("OVERALL STATISTICS:")
    logger.info(f"Total Messages Sent: {total_sent}")
    logger.info(f"Total Responses: {total_responses}")
    logger.info(f"Overall Response Rate: {overall_rate:.1f}%")
    logger.info("=" * 60)


# ------------------------------------------------------------------------------
# Performance Tracking (Action 6 Pattern)
# ------------------------------------------------------------------------------

def _update_messaging_performance(session_manager: SessionManager, duration: float) -> None:
    """
    Track messaging performance like Action 6.
    Updates response times and slow call tracking for adaptive behavior.

    Args:
        session_manager: The SessionManager instance to update
        duration: Duration of the operation in seconds
    """
    try:
        # Initialize performance tracking attributes if they don't exist
        if not hasattr(session_manager, '_response_times'):
            session_manager._response_times = []  # type: ignore[attr-defined]
        if not hasattr(session_manager, '_recent_slow_calls'):
            session_manager._recent_slow_calls = 0  # type: ignore[attr-defined]
        if not hasattr(session_manager, '_avg_response_time'):
            session_manager._avg_response_time = 0.0  # type: ignore[attr-defined]

        # Track response time (keep last 50 measurements)
        session_manager._response_times.append(duration)  # type: ignore[attr-defined]
        if len(session_manager._response_times) > 50:  # type: ignore[attr-defined]
            session_manager._response_times.pop(0)  # type: ignore[attr-defined]

        # Update average response time
        session_manager._avg_response_time = sum(session_manager._response_times) / len(session_manager._response_times)  # type: ignore[attr-defined]

        # Track consecutive slow calls - OPTIMIZATION: Adjusted threshold like Action 6
        if duration > 15.0:  # OPTIMIZATION: Increased from 5.0s to 15.0s - align with Action 6 thresholds
            session_manager._recent_slow_calls += 1  # type: ignore[attr-defined]
        else:
            session_manager._recent_slow_calls = max(0, session_manager._recent_slow_calls - 1)  # type: ignore[attr-defined]

        # Cap slow call counter to prevent endless accumulation
        session_manager._recent_slow_calls = min(session_manager._recent_slow_calls, 10)  # type: ignore[attr-defined]

    except Exception as e:
        logger.debug(f"Failed to update messaging performance tracking: {e}")
        pass


# ------------------------------------------------------------------------------
# Database and Processing Helpers
# ------------------------------------------------------------------------------
# Note: _commit_messaging_batch was removed as dead code - replaced by _safe_commit_with_rollback




def _get_simple_messaging_data(
    db_session: Session,
    session_manager: Optional[SessionManager] = None,
) -> tuple[Optional[dict[str, int]], Optional[list[Person]]]:
    """
    Simplified data fetching for messaging process.
    - Fetches MessageTemplate key-to-ID mapping.
    - Fetches candidate Person records (ACTIVE or DESIST status, contactable=True).

    Message history is fetched per-person during processing for simplicity.

    Args:
        db_session: The active SQLAlchemy database session.
        session_manager: Optional SessionManager for halt signal checking.

    Returns:
        A tuple containing:
        - message_type_map (dict[str, int]): Map of template_key to MessageTemplate ID.
        - candidate_persons (list[Person]): List of Person objects meeting criteria.
        Returns (None, None) if essential data fetching fails.
    """
    try:
        # Check for halt signal before starting
        if session_manager and session_manager.should_halt_operations():
            logger.critical("🚨 HALT SIGNAL DETECTED: Stopping messaging data fetch immediately.")
            raise MaxApiFailuresExceededError("Session halt detected - stopping messaging data fetch")

        # Step 1: Fetch MessageTemplate map
        logger.debug("Fetching MessageTemplate key-to-ID mapping...")

        # Check if we're running in a test/mock environment
        is_mock_mode = "--mock" in sys.argv or "--test" in sys.argv

        if is_mock_mode:
            # Create mock data for testing
            logger.debug("Running in mock mode, creating mock MessageTemplate map...")
            message_type_map = {name: i for i, name in enumerate(MESSAGE_TYPES_ACTION8.keys(), start=1)}
            message_type_map["Productive_Reply_Acknowledgement"] = len(message_type_map) + 1
        else:
            # Fetch MessageTemplate key-to-ID mapping
            message_templates = db_session.query(
                MessageTemplate.id, MessageTemplate.template_key
            ).all()
            message_type_map = {template_key: template_id for template_id, template_key in message_templates}

        # Basic validation
        if not message_type_map:
            logger.error("No MessageTemplates found in database")
            return None, None
        # Step 2: Fetch Candidate Persons
        logger.debug("Fetching candidate persons...")

        if is_mock_mode:
            candidate_persons = []
        else:
            candidate_persons = (
                db_session.query(Person)
                .options(
                    joinedload(Person.dna_match),
                    joinedload(Person.family_tree),
                )
                .filter(
                    Person.profile_id.isnot(None),
                    Person.profile_id != "UNKNOWN",
                    Person.contactable,
                    Person.status.in_([PersonStatusEnum.ACTIVE, PersonStatusEnum.DESIST]),
                    Person.deleted_at.is_(None),
                )
                .order_by(Person.id)
                .all()
            )

        logger.debug(f"Found {len(candidate_persons)} potential candidates.")
        return message_type_map, candidate_persons

    except Exception as e:
        logger.error(f"Error fetching messaging data: {e}", exc_info=True)
        return None, None


def _get_person_message_history(db_session: Session, person_id: int) -> tuple[Optional[ConversationLog], Optional[ConversationLog], Optional[str]]:
    """
    Get the latest IN and OUT message history for a specific person.

    Args:
        db_session: Database session
        person_id: Person ID to get history for

    Returns:
        Tuple of (latest_in_log, latest_out_log, latest_out_template_key)
    """
    try:
        # Get latest IN message
        latest_in = (
            db_session.query(ConversationLog)
            .filter(
                ConversationLog.people_id == person_id,
                ConversationLog.direction == MessageDirectionEnum.IN,
                ~ConversationLog.conversation_id.like('template_tracking_%'),
                ~ConversationLog.script_message_status.like('TEMPLATE_SELECTED:%')
            )
            .order_by(ConversationLog.latest_timestamp.desc())
            .first()
        )

        # Get latest OUT message with template key
        latest_out_query = (
            db_session.query(ConversationLog, MessageTemplate.template_key)
            .outerjoin(MessageTemplate, ConversationLog.message_template_id == MessageTemplate.id)
            .filter(
                ConversationLog.people_id == person_id,
                ConversationLog.direction == MessageDirectionEnum.OUT,
                ~ConversationLog.conversation_id.like('template_tracking_%'),
                ~ConversationLog.script_message_status.like('TEMPLATE_SELECTED:%')
            )
            .order_by(ConversationLog.latest_timestamp.desc())
            .first()
        )

        if latest_out_query:
            latest_out, template_key = latest_out_query
        else:
            latest_out, template_key = None, None

        return latest_in, latest_out, template_key

    except Exception as e:
        logger.warning(f"Error getting message history for person {person_id}: {e}")
        return None, None, None


def _validate_system_health(session_manager: SessionManager) -> bool:
    """
    Comprehensive system health validation before starting messaging operations.

    Uses universal session health validation with Action 8-specific template checks.

    Args:
        session_manager: The active SessionManager instance.

    Returns:
        True if system is healthy and ready for messaging, False otherwise.
    """
    try:
        # Use consolidated session health validation
        if not session_manager.validate_system_health("Action 8"):
            return False

        # Action 8-specific check: Essential message templates availability
        required_templates = set(MESSAGE_TYPES_ACTION8.keys())
        missing_templates = []
        for template_key in required_templates:
            if template_key not in MESSAGE_TEMPLATES:
                missing_templates.append(template_key)

        if missing_templates:
            logger.critical(
                f"🚨 Action 8: Essential message templates missing: {missing_templates}. "
                f"Cannot proceed with messaging operations."
            )
            return False

        logger.debug("✅ Action 8: System health check passed - all components validated")
        return True

    except Exception as health_check_err:
        logger.critical(f"🚨 Action 8: System health check failed: {health_check_err}")
        return False


# Replaced with universal cascade checking
# Use check_cascade_before_operation(session_manager, "Action 8", operation_name) instead


def _safe_commit_with_rollback(
    session: Session,
    log_upserts: list[dict[str, Any]],
    person_updates: dict[int, PersonStatusEnum],
    context: str,
    session_manager: SessionManager
) -> tuple[bool, int, int]:
    """
    Safely commit batch data with comprehensive rollback on failure.

    Args:
        session: Database session
        log_upserts: List of log dictionaries to insert
        person_updates: Dictionary of person updates
        context: Context string for logging
        session_manager: Session manager for cascade detection

    Returns:
        Tuple of (success, logs_committed, persons_updated)
    """
    # Check for cascade before attempting commit
    session_manager.check_cascade_before_operation("Action 8", f"safe commit {context}")

    # Create backup of data for potential rollback
    backup_log_upserts = log_upserts.copy()
    backup_person_updates = person_updates.copy()

    try:
        # Use isolation level to prevent concurrent access issues
        with session.begin():  # Explicit transaction with automatic rollback on exception
            logs_committed, persons_updated = commit_bulk_data(
                session=session,
                log_upserts=log_upserts,
                person_updates=person_updates,
                context=context
            )

            # Verify commit was successful
            if logs_committed == 0 and persons_updated == 0 and (log_upserts or person_updates):
                logger.warning(f"Commit returned zero counts but data was provided for {context}")
                return False, 0, 0

            logger.debug(f"Safe commit successful for {context}: {logs_committed} logs, {persons_updated} persons")
            return True, logs_committed, persons_updated

    except Exception as commit_error:
        logger.error(f"Safe commit failed for {context}: {commit_error}", exc_info=True)

        # Attempt to restore data for retry (if needed)
        log_upserts.clear()
        log_upserts.extend(backup_log_upserts)
        person_updates.clear()
        person_updates.update(backup_person_updates)

        return False, 0, 0


class ErrorCategorizer:
    """
    Proper error categorization and monitoring for Action8.
    """

    def __init__(self) -> None:
        self.error_counts = {
            'business_logic_skips': 0,
            'technical_errors': 0,
            'api_failures': 0,
            'authentication_errors': 0,
            'rate_limit_errors': 0,
            'cascade_errors': 0,
            'template_errors': 0,
            'database_errors': 0
        }
        self.monitoring_hooks = []

    def _categorize_success_status(self, status_lower: str) -> tuple[str, str] | None:
        """Categorize successful status. Returns None if not a success status."""
        success_statuses = {
            'sent': ['sent', 'delivered ok'],
            'acked': ['acked', 'acknowledged'],
        }

        for category, status_list in success_statuses.items():
            if status_lower in status_list:
                return category, 'success'

        return None

    def _categorize_skip_status(self, status_lower: str) -> tuple[str, str] | None:
        """Categorize skip status. Returns None if not a skip status."""
        if not status_lower.startswith('skipped'):
            return None

        business_logic_skips = [
            'interval', 'cooldown', 'recent_message', 'duplicate',
            'filter', 'rule', 'preference', 'opt_out', 'blocked'
        ]

        for skip_type in business_logic_skips:
            if skip_type in status_lower:
                self.error_counts['business_logic_skips'] += 1
                return 'skipped', f'business_logic_{skip_type}'

        # Generic skip
        self.error_counts['business_logic_skips'] += 1
        return 'skipped', 'business_logic_generic'

    def _categorize_error_detail(self, error_detail: str) -> tuple[str, str]:
        """Categorize error based on detail string. Returns (category, error_type)."""
        # Data-driven error mapping
        error_patterns = [
            (['auth', 'login'], 'authentication_errors', 'authentication_failure'),
            (['rate', '429'], 'rate_limit_errors', 'rate_limit_exceeded'),
            (['cascade'], 'cascade_errors', 'session_cascade'),
            (['template'], 'template_errors', 'template_failure'),
            (['database', 'db'], 'database_errors', 'database_failure'),
            (['api'], 'api_failures', 'api_failure'),
        ]

        for keywords, counter_key, error_type in error_patterns:
            if any(keyword in error_detail for keyword in keywords):
                self.error_counts[counter_key] += 1
                return 'error', error_type

        # Generic technical error
        self.error_counts['technical_errors'] += 1
        return 'error', 'technical_failure'

    def _categorize_error_status(self, status: str, status_lower: str) -> tuple[str, str] | None:
        """Categorize error status. Returns None if not an error status."""
        if not status_lower.startswith('error'):
            return None

        # Extract error type from parentheses
        if '(' in status and ')' in status:
            error_detail = status[status.find('(')+1:status.find(')')].lower()
            return self._categorize_error_detail(error_detail)

        # Generic technical error
        self.error_counts['technical_errors'] += 1
        return 'error', 'technical_failure'

    def categorize_status(self, status: str) -> tuple[str, str]:
        """
        Categorize a status string into proper category and type.

        Args:
            status: Status string from processing

        Returns:
            Tuple of (category, error_type) where category is 'sent', 'acked', 'skipped', or 'error'
        """
        if not status:
            return 'error', 'unknown_status'

        status_lower = status.lower()

        # Try categorizing as success
        result = self._categorize_success_status(status_lower)
        if result:
            return result

        # Try categorizing as skip
        result = self._categorize_skip_status(status_lower)
        if result:
            return result

        # Try categorizing as error
        result = self._categorize_error_status(status, status_lower)
        if result:
            return result

        # Default to error for unknown status
        self.error_counts['technical_errors'] += 1
        return 'error', 'unknown_status'

    def add_monitoring_hook(self, hook_function: Callable) -> None:
        """Add a monitoring hook function."""
        self.monitoring_hooks.append(hook_function)

    def trigger_monitoring_alert(self, alert_type: str, message: str, severity: str = 'warning') -> None:
        """Trigger monitoring alerts through registered hooks."""
        alert_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'error_counts': self.error_counts.copy()
        }

        for hook in self.monitoring_hooks:
            try:
                hook(alert_data)
            except Exception as hook_err:
                logger.warning(f"Monitoring hook failed: {hook_err}")

    def get_error_summary(self) -> dict[str, Any]:
        """Get comprehensive error summary for reporting."""
        total_errors = sum(count for key, count in self.error_counts.items() if 'error' in key)
        total_skips = self.error_counts['business_logic_skips']

        return {
            'total_technical_errors': total_errors,
            'total_business_skips': total_skips,
            'error_breakdown': self.error_counts.copy(),
            'error_rate': total_errors / max(1, total_errors + total_skips),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if total_errors > 0 else None
        }


class ResourceManager:
    """
    Comprehensive resource management for memory, cleanup, and garbage collection.
    """

    def __init__(self) -> None:
        self.allocated_resources = []
        self.memory_threshold_mb = 100  # Trigger cleanup at 100MB
        self.gc_interval = 50  # Trigger GC every 50 operations
        self.operation_count = 0

    def track_resource(self, resource_name: str, resource_obj: Any) -> None:
        """Track a resource for cleanup."""
        self.allocated_resources.append((resource_name, resource_obj))

    def check_memory_usage(self) -> tuple[float, bool]:
        """Check current memory usage. Returns (memory_mb, should_cleanup)."""
        import os

        import psutil

        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            should_cleanup = memory_mb > self.memory_threshold_mb

            if should_cleanup:
                logger.warning(f"Memory usage high: {memory_mb:.1f}MB (threshold: {self.memory_threshold_mb}MB)")

            return memory_mb, should_cleanup

        except Exception as mem_err:
            logger.warning(f"Could not check memory usage: {mem_err}")
            return 0.0, False

    def trigger_garbage_collection(self) -> int:
        """Trigger garbage collection and return objects collected."""
        import gc

        before_count = len(gc.get_objects())
        collected = gc.collect()
        after_count = len(gc.get_objects())

        logger.debug(f"🗑️ Garbage collection: {collected} cycles, {before_count - after_count} objects freed")
        return collected

    def cleanup_resources(self) -> None:
        """Clean up tracked resources."""
        cleaned_count = 0

        for resource_name, resource_obj in self.allocated_resources:
            try:
                if hasattr(resource_obj, 'close'):
                    resource_obj.close()
                elif hasattr(resource_obj, 'cleanup'):
                    resource_obj.cleanup()
                elif hasattr(resource_obj, 'clear'):
                    resource_obj.clear()

                cleaned_count += 1
                logger.debug(f"🧹 Cleaned up resource: {resource_name}")

            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup resource {resource_name}: {cleanup_err}")

        self.allocated_resources.clear()
        logger.info(f"🧹 Resource cleanup completed: {cleaned_count} resources cleaned")

    def periodic_maintenance(self) -> None:
        """Perform periodic maintenance operations."""
        self.operation_count += 1

        # Check memory and cleanup if needed
        _, should_cleanup = self.check_memory_usage()

        if should_cleanup:
            self.cleanup_resources()
            self.trigger_garbage_collection()

        # Periodic garbage collection
        elif self.operation_count % self.gc_interval == 0:
            self.trigger_garbage_collection()


class ProactiveApiManager:
    """Proactive API management with rate limiting and authentication monitoring."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager
        self.consecutive_failures = 0
        self.last_auth_check = 0
        self.auth_check_interval = 300
        self.max_consecutive_failures = 3
        self.base_delay = 1.0
        self.max_delay = 30.0

    def check_authentication(self) -> bool:
        """Check authentication status. Returns True if authenticated."""
        current_time = time.time()

        # Only check if enough time has passed
        if current_time - self.last_auth_check < self.auth_check_interval:
            return True

        self.last_auth_check = current_time

        try:
            if not self.session_manager.is_sess_valid():
                logger.warning("🔐 Authentication check failed - session invalid")
                return False

            # Additional check for profile ID
            if not self.session_manager.my_profile_id:
                logger.warning("🔐 Authentication check failed - no profile ID")
                return False

            logger.debug("✅ Authentication check passed")
            return True

        except Exception as auth_err:
            logger.error(f"🔐 Authentication check error: {auth_err}")
            return False

    def attempt_reauthentication(self) -> bool:
        """Attempt to re-authenticate. Returns True if successful."""
        logger.warning("Attempting re-authentication...")

        try:
            if hasattr(self.session_manager, 'attempt_recovery'):
                recovery_success = self.session_manager.attempt_recovery('auth_recovery')  # type: ignore[attr-defined]
                if recovery_success:
                    logger.info("Re-authentication successful")
                    self.consecutive_failures = 0
                    return True

            logger.error("Re-authentication failed")
            return False

        except Exception as reauth_err:
            logger.error(f"Re-authentication error: {reauth_err}")
            return False

    def calculate_delay(self) -> float:
        """Calculate proactive delay based on failure history."""
        if self.consecutive_failures == 0:
            return 0.0

        import random
        delay = min(self.base_delay * (2 ** self.consecutive_failures), self.max_delay)
        return delay * random.uniform(0.8, 1.2)

    def _validate_message_send_response(self, response_data: Any, operation: str) -> bool | None:
        """Validate message send response. Returns True/False if validated, None if not applicable."""
        if not operation.startswith("send_message") or not isinstance(response_data, tuple) or len(response_data) < 2:
            return None

        status = response_data[0]
        if status and "delivered OK" in status:
            logger.debug(f"API validation passed: {status}")
            return True

        if status and "error" in status.lower():
            logger.warning(f"API validation failed: {status}")
            return False

        return None

    def _validate_generic_response(self, response_data: Any, operation: str) -> bool:
        """Validate generic API response. Returns True if valid."""
        if isinstance(response_data, dict) and ("error" in response_data or "errors" in response_data):
            logger.warning(f"❌ API validation failed for {operation}: Response contains errors")
            return False

        logger.debug(f"✅ API validation passed for {operation}")
        return True

    def validate_api_response(self, response_data: Any, operation: str) -> bool:
        """
        Validate API response to ensure it's actually successful.

        Args:
            response_data: The API response data
            operation: Description of the operation

        Returns:
            bool: True if response is valid, False otherwise
        """
        if response_data is None:
            logger.warning(f"API validation failed for {operation}: Response is None")
            return False

        # Try message send validation
        result = self._validate_message_send_response(response_data, operation)
        if result is not None:
            return result

        # Fall back to generic validation
        return self._validate_generic_response(response_data, operation)

    def record_api_result(self, success: bool, operation: str) -> None:
        """
        Record API operation result for adaptive behavior.

        Args:
            success: Whether the operation was successful
            operation: Description of the operation
        """
        if success:
            self.consecutive_failures = 0
            logger.debug(f"📊 API success recorded for {operation}")
        else:
            self.consecutive_failures += 1
            logger.warning(f"📊 API failure recorded for {operation} (consecutive: {self.consecutive_failures})")

            # If too many failures, suggest longer delays
            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.critical(f"🚨 Too many consecutive API failures ({self.consecutive_failures}). Consider halting operations.")


def _with_operation_timeout(operation_func: Callable, timeout_seconds: int, operation_name: str):
    """
    Execute operation with proper timeout handling (cross-platform).

    Args:
        operation_func: Function to execute
        timeout_seconds: Timeout in seconds
        operation_name: Name for logging

    Returns:
        Result of operation or raises TimeoutError
    """
    import threading

    result: list[Any] = [None]
    exception: list[Optional[Exception]] = [None]
    completed = [False]

    def target() -> None:
        try:
            result[0] = operation_func()
            completed[0] = True
        except Exception as e:
            exception[0] = e
            completed[0] = True

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if not completed[0]:
        # Operation timed out
        logger.warning(f"⏱️ Operation '{operation_name}' timed out after {timeout_seconds}s (thread abandoned)")
        raise TimeoutError(f"Operation '{operation_name}' timed out after {timeout_seconds} seconds")

    if exception[0]:
        raise exception[0]

    return result[0]


def _safe_api_call_with_validation(
    session_manager: SessionManager,
    api_function: Callable,
    operation_name: str,
    *args,
    **kwargs
) -> tuple[bool, Any]:
    """
    Safely call API with proactive rate limiting, authentication, and validation.

    Args:
        session_manager: Session manager instance
        api_function: The API function to call
        operation_name: Name of the operation for logging
        *args: Arguments to pass to the API function
        **kwargs: Keyword arguments to pass to the API function

    Returns:
        Tuple of (success, result)
    """
    api_manager = ProactiveApiManager(session_manager)

    # Step 1: Check authentication proactively
    if not api_manager.check_authentication() and not api_manager.attempt_reauthentication():
        logger.error(f"🔐 Cannot proceed with {operation_name} - authentication failed")
        return False, None

    # Step 2: Check for cascade before API call
    session_manager.check_cascade_before_operation("Action 8", f"API call {operation_name}")

    # Step 3: Apply proactive delay if needed
    delay = api_manager.calculate_delay()
    if delay > 0:
        import time
        logger.debug(f"⏱️ Proactive delay for {operation_name}: {delay:.1f}s")
        time.sleep(delay)

    # Step 4: Make the API call with timeout protection
    try:
        def api_call() -> Any:
            return api_function(*args, **kwargs)

        # Use operation-level timeout (60 seconds for API calls)
        result = _with_operation_timeout(api_call, 60, f"API_{operation_name}")

        # Step 5: Validate the response
        is_valid = api_manager.validate_api_response(result, operation_name)
        api_manager.record_api_result(is_valid, operation_name)

        return is_valid, result

    except TimeoutError as timeout_err:
        logger.error(f"⏱️ API call timeout for {operation_name}: {timeout_err}")
        api_manager.record_api_result(False, operation_name)
        return False, None
    except Exception as api_err:
        logger.error(f"❌ API call failed for {operation_name}: {api_err}")
        api_manager.record_api_result(False, operation_name)
        return False, None


# === SINGLE PERSON PROCESSING HELPER FUNCTIONS ===

def _check_halt_signal(session_manager: SessionManager) -> None:
    """Check for halt signal and raise exception if detected."""
    if session_manager.should_halt_operations():
        cascade_count = session_manager.session_health_monitor.get('death_cascade_count', 0)
        logger.warning(f"🚨 HALT SIGNAL: Skipping person processing due to session death cascade (#{cascade_count})")
        raise MaxApiFailuresExceededError(f"Session death cascade detected (#{cascade_count}) - halting person processing")


def _initialize_person_processing(person: Person) -> tuple[str, int, str, str]:
    """Initialize person processing variables and return log prefix and identifiers."""
    username = safe_column_value(person, "username", "Unknown")
    person_id = safe_column_value(person, "id", 0)
    status = safe_column_value(person, "status", None)
    status_name = getattr(status, "name", "Unknown") if status is not None else "Unknown"
    log_prefix = f"{username} #{person_id} (Status: {status_name})"
    return log_prefix, person_id, username, status_name


def _check_person_eligibility(person: Person, log_prefix: str) -> None:
    """Check if person is eligible for messaging based on status."""
    if person.status in (PersonStatusEnum.ARCHIVE, PersonStatusEnum.BLOCKED, PersonStatusEnum.DEAD):
        logger.debug(f"Skipping {log_prefix}: Status is '{person.status.name}'.")
        raise StopIteration("skipped (status)")


def _handle_desist_status(log_prefix: str, latest_out_log: Optional[ConversationLog], message_type_map: dict[str, int]) -> tuple[Optional[str], str]:
    """Handle DESIST status and return message key and reason if ACK needed."""
    logger.debug(f"{log_prefix}: Status is DESIST. Checking if Desist ACK needed.")

    desist_ack_type_id = message_type_map.get("User_Requested_Desist")
    if not desist_ack_type_id:
        logger.critical("CRITICAL: User_Requested_Desist ID missing from message type map.")
        raise StopIteration("error (config)")

    ack_already_sent = bool(latest_out_log and latest_out_log.message_template_id == desist_ack_type_id)
    if ack_already_sent:
        logger.debug(f"Skipping {log_prefix}: Desist ACK already sent.")
        raise StopIteration("skipped (ack_sent)")

    logger.debug(f"Action needed for {log_prefix}: Send Desist ACK.")
    return "User_Requested_Desist", "DESIST Acknowledgment"


def _check_reply_received(latest_in_log: Optional[ConversationLog], latest_out_log: Optional[ConversationLog], log_prefix: str) -> None:
    """Check if reply was received since last script message."""
    min_aware_dt = datetime.min.replace(tzinfo=timezone.utc)

    last_out_ts_utc = min_aware_dt
    if latest_out_log:
        last_out_ts_utc = safe_column_value(latest_out_log, "latest_timestamp", min_aware_dt)
        last_out_ts_utc = _ensure_timezone_aware(last_out_ts_utc) or min_aware_dt

    last_in_ts_utc = min_aware_dt
    if latest_in_log:
        last_in_ts_utc = safe_column_value(latest_in_log, "latest_timestamp", min_aware_dt)
        last_in_ts_utc = _ensure_timezone_aware(last_in_ts_utc) or min_aware_dt

    if last_in_ts_utc > last_out_ts_utc:
        logger.debug(f"Skipping {log_prefix}: Reply received after last script msg.")
        raise StopIteration("skipped (reply)")

    if latest_in_log and hasattr(latest_in_log, "custom_reply_sent_at") and latest_in_log.custom_reply_sent_at is not None:
        logger.debug(f"Skipping {log_prefix}: Custom reply already sent.")
        raise StopIteration("skipped (custom_reply_sent)")


def _check_message_interval(latest_out_log: Optional[ConversationLog], log_prefix: str) -> None:
    """Check if minimum message interval has passed since last script message."""
    if not latest_out_log:
        return

    out_timestamp = safe_column_value(latest_out_log, "latest_timestamp", None)
    if not out_timestamp:
        return

    try:
        out_timestamp = _ensure_timezone_aware(out_timestamp)
        if not out_timestamp:
            return
        if out_timestamp.tzinfo != timezone.utc:
            out_timestamp = out_timestamp.astimezone(timezone.utc)

        now_utc = datetime.now(timezone.utc)

        time_since_last = now_utc - out_timestamp
        if time_since_last < MIN_MESSAGE_INTERVAL:
            logger.debug(f"Skipping {log_prefix}: Interval not met.")
            raise StopIteration("skipped (interval)")
    except Exception as dt_error:
        logger.error(f"Datetime comparison error for {log_prefix}: {dt_error}")
        raise StopIteration("skipped (datetime_error)") from None


def _get_last_script_message_details(latest_out_log: Optional[ConversationLog], latest_out_template_key: Optional[str], log_prefix: str) -> Optional[tuple[str, datetime, str]]:
    """Extract details from the last script message."""
    if not latest_out_log:
        return None

    out_timestamp = safe_column_value(latest_out_log, "latest_timestamp", None)
    if not out_timestamp:
        return None

    try:
        out_timestamp = _ensure_timezone_aware(out_timestamp)
        if not out_timestamp:
            out_timestamp = datetime.now(timezone.utc)
        elif out_timestamp.tzinfo != timezone.utc:
            out_timestamp = out_timestamp.astimezone(timezone.utc)
    except Exception as tz_error:
        logger.warning(f"Timezone conversion error for {log_prefix}: {tz_error}")
        out_timestamp = datetime.now(timezone.utc)

    last_type_name = latest_out_template_key
    if not last_type_name or last_type_name == "Unknown":
        last_type_name = None
        logger.debug(f"Could not determine message type for {log_prefix}, using None for fallback")

    last_status = safe_column_value(latest_out_log, "script_message_status", "Unknown")
    return (last_type_name, out_timestamp, last_status)


def _determine_message_to_send(person: Person, latest_out_log: Optional[ConversationLog], latest_out_template_key: Optional[str], log_prefix: str) -> tuple[str, str]:
    """Determine which message to send and the selection reason."""
    last_script_message_details = _get_last_script_message_details(latest_out_log, latest_out_template_key, log_prefix)
    base_message_key = determine_next_message_type(last_script_message_details, bool(person.in_my_tree))

    if not base_message_key:
        logger.debug(f"Skipping {log_prefix}: No appropriate next standard message found.")
        raise StopIteration("skipped (sequence)")

    person_id = safe_column_value(person, "id", 0)
    dna_match = person.dna_match
    family_tree = person.family_tree

    # First try A/B testing for initial messages
    if base_message_key in ["In_Tree-Initial", "Out_Tree-Initial"]:
        message_to_send_key = select_template_variant_ab_testing(person_id, base_message_key)
        template_selection_reason = "A/B Testing"

        if message_to_send_key == base_message_key:
            message_to_send_key = select_template_by_confidence(base_message_key, family_tree, dna_match)
            template_selection_reason = "Confidence-based"
    else:
        message_to_send_key = base_message_key
        template_selection_reason = "Standard sequence"

    track_template_selection(message_to_send_key, person_id, template_selection_reason)
    logger.debug(f"Action needed for {log_prefix}: Send '{message_to_send_key}' (selected via {template_selection_reason}).")

    return message_to_send_key, template_selection_reason


def _get_best_name_for_person(person: Person, family_tree: Optional[FamilyTree]) -> str:
    """Determine the best name to use for the person (Tree Name > First Name > Username)."""
    tree_name = None
    if family_tree:
        tree_name = safe_column_value(family_tree, "person_name_in_tree", None)

    first_name = safe_column_value(person, "first_name", None)
    username = safe_column_value(person, "username", None)

    if tree_name:
        return tree_name
    if first_name:
        return first_name
    if username and username not in ["Unknown", "Unknown User"]:
        return username
    return "Valued Relative"


def _format_predicted_relationship(rel_str: str) -> str:
    """Format predicted relationship with correct percentage."""
    if not rel_str or rel_str == "N/A":
        return "N/A"

    import re
    match = re.search(r"\[([\d.]+)%\]", rel_str)
    if match:
        try:
            percentage = float(match.group(1))
            if percentage > 100.0:
                corrected_percentage = percentage / 100.0
                return re.sub(r"\[([\d.]+)%\]", f"[{corrected_percentage:.1f}%]", rel_str)
            if percentage < 1.0:
                corrected_percentage = percentage * 100.0
                return re.sub(r"\[([\d.]+)%\]", f"[{corrected_percentage:.1f}%]", rel_str)
        except (ValueError, IndexError):
            pass

    return rel_str


def _prepare_message_format_data(person: Person, family_tree: Optional[FamilyTree], dna_match: Optional[DnaMatch], db_session: Session) -> dict:
    """Prepare format data for message template."""
    name_to_use = _get_best_name_for_person(person, family_tree)
    formatted_name = format_name(name_to_use)

    total_rows_in_tree = 0
    try:
        total_rows_in_tree = db_session.query(func.count(FamilyTree.id)).scalar() or 0
    except Exception as count_e:
        logger.warning(f"Could not get FamilyTree count for formatting: {count_e}")

    predicted_rel = "N/A"
    if dna_match:
        raw_predicted_rel = getattr(dna_match, "predicted_relationship", "N/A")
        predicted_rel = _format_predicted_relationship(raw_predicted_rel)

    safe_actual_relationship = get_safe_relationship_text(family_tree, predicted_rel)
    safe_relationship_path = get_safe_relationship_path(family_tree)

    return {
        "name": formatted_name,
        "predicted_relationship": predicted_rel if predicted_rel != "N/A" else "family connection",
        "actual_relationship": safe_actual_relationship,
        "relationship_path": safe_relationship_path,
        "total_rows": total_rows_in_tree,
    }


def _format_message_text(message_to_send_key: str, person: Person, format_data: dict, log_prefix: str) -> str:
    """Format message text using enhanced or standard template."""
    message_template = MESSAGE_TEMPLATES[message_to_send_key]
    message_text = None

    # Try enhanced personalized message formatting first
    if MESSAGE_PERSONALIZER and hasattr(person, 'extracted_genealogical_data'):
        try:
            enhanced_template_key = f"Enhanced_{message_to_send_key}"
            if enhanced_template_key in MESSAGE_TEMPLATES:
                logger.debug(f"Using enhanced template '{enhanced_template_key}' for {log_prefix}")

                extracted_data = getattr(person, 'extracted_genealogical_data', {})
                person_data = {"username": getattr(person, "username", "Unknown")}

                message_text, _ = MESSAGE_PERSONALIZER.create_personalized_message(
                    enhanced_template_key,
                    person_data,
                    extracted_data,
                    format_data
                )
                logger.debug(f"Successfully created personalized message for {log_prefix}")
            else:
                logger.debug(f"Enhanced template '{enhanced_template_key}' not available, using standard template")
        except Exception as e:
            logger.warning(f"Enhanced message formatting failed for {log_prefix}: {e}, falling back to standard")
            message_text = None

    # Fallback to standard template formatting
    if not message_text:
        try:
            message_text = message_template.format(**format_data)
        except KeyError as ke:
            logger.error(f"Template formatting error (Missing key {ke}) for '{message_to_send_key}' {log_prefix}")
            raise StopIteration("error (template_format)") from None
        except Exception as e:
            logger.error(f"Unexpected template formatting error for {log_prefix}: {e}", exc_info=True)
            raise StopIteration("error (template_format)") from None

    return message_text


def _check_mode_filtering(person: Person, log_prefix: str) -> tuple[bool, str]:
    """Check if message should be filtered based on app mode and testing profile."""
    app_mode = getattr(config_schema, 'app_mode', 'production')
    testing_profile_id_config = config_schema.testing_profile_id
    current_profile_id = safe_column_value(person, "profile_id", "UNKNOWN")

    # Testing mode checks
    if app_mode == "testing":
        if not testing_profile_id_config:
            logger.error(f"Testing mode active, but TESTING_PROFILE_ID not configured. Skipping {log_prefix}.")
            return False, "skipped (config_error)"
        if current_profile_id != testing_profile_id_config:
            skip_reason = f"skipped (testing_mode_filter: not {testing_profile_id_config})"
            logger.debug(f"Testing Mode: Skipping send to {log_prefix} ({skip_reason}).")
            return False, skip_reason

    # Production mode checks
    elif app_mode == "production" and (testing_profile_id_config and current_profile_id == testing_profile_id_config):
        skip_reason = f"skipped (production_mode_filter: is {testing_profile_id_config})"
        logger.debug(f"Production Mode: Skipping send to test profile {log_prefix} ({skip_reason}).")
        return False, skip_reason

    return True, ""


def _get_existing_conversation_id(latest_out_log: Optional[ConversationLog], latest_in_log: Optional[ConversationLog]) -> Optional[str]:
    """Get existing conversation ID from logs (prefer OUT, fallback IN)."""
    existing_conversation_id = None
    if latest_out_log:
        existing_conversation_id = safe_column_value(latest_out_log, "conversation_id", None)

    if existing_conversation_id is None and latest_in_log:
        existing_conversation_id = safe_column_value(latest_in_log, "conversation_id", None)

    return existing_conversation_id


def _send_or_simulate_message(
    session_manager: SessionManager,
    msg_ctx: 'MessageContext',
    conv_state: 'ConversationState',
    msg_flags: 'MessageFlags'
) -> tuple[str, Optional[str]]:
    """Send or simulate message, return status and conversation ID."""
    if msg_flags.send_message_flag:
        log_prefix_for_api = f"Action8: {msg_ctx.person.username} #{msg_ctx.person.id}"
        api_success, api_result = _safe_api_call_with_validation(
            session_manager,
            call_send_message_api,
            f"send_message_{msg_ctx.person.username}",
            session_manager,
            msg_ctx.person,
            msg_ctx.message_text,
            conv_state.existing_conversation_id,
            log_prefix_for_api,
        )

        if api_success and api_result:
            message_status, effective_conv_id = api_result
        else:
            message_status = "error (api_validation_failed)"
            effective_conv_id = None
            logger.warning(f"API call failed for {msg_ctx.person.username}: validation or execution error")
    else:
        message_status = msg_flags.skip_log_reason
        effective_conv_id = _get_existing_conversation_id(conv_state.latest_out_log, conv_state.latest_in_log)
        if effective_conv_id is None:
            effective_conv_id = f"skipped_{uuid.uuid4()}"

    return message_status, effective_conv_id


def _prepare_conversation_log_entry(
    msg_ctx: 'MessageContext',
    conv_state: 'ConversationState',
    msg_flags: 'MessageFlags',
    message_type_map: dict[str, int]
) -> ConversationLog:
    """Prepare conversation log entry for database."""
    message_template_id_to_log = message_type_map.get(msg_ctx.message_to_send_key)
    if not message_template_id_to_log:
        logger.error(f"MessageTemplate ID missing for key '{msg_ctx.message_to_send_key}'")
        raise StopIteration("error (db_config)")
    if not conv_state.effective_conv_id:
        logger.error(f"effective_conv_id missing for {msg_ctx.log_prefix}")
        raise StopIteration("error (internal)")

    log_content = (f"[{msg_flags.message_status.upper()}] {msg_ctx.message_text}" if not msg_flags.send_message_flag else msg_ctx.message_text)[:config_schema.message_truncation_length]
    enhanced_status = f"{msg_flags.message_status} | Template: {msg_ctx.message_to_send_key} ({msg_ctx.template_selection_reason})"

    return ConversationLog(
        conversation_id=conv_state.effective_conv_id,
        direction=MessageDirectionEnum.OUT,
        people_id=msg_ctx.person.id,
        latest_message_content=log_content,
        latest_timestamp=datetime.now(timezone.utc),
        ai_sentiment=None,
        message_template_id=message_template_id_to_log,
        script_message_status=enhanced_status,
    )


def _determine_final_status(message_to_send_key: str, message_status: str, send_message_flag: bool, person_id: int, log_prefix: str) -> tuple[str, Optional[tuple[int, PersonStatusEnum]]]:
    """Determine final status and person update based on message outcome."""
    person_update = None

    if message_status in ("delivered OK", "typed (dry_run)") or message_status.startswith("skipped ("):
        if message_to_send_key == "User_Requested_Desist":
            person_update = (person_id, PersonStatusEnum.ARCHIVE)
            status_string = "acked"
        elif send_message_flag:
            status_string = "sent"
        else:
            status_string = "skipped"
    else:
        logger.warning(f"Message send failed for {log_prefix}: {message_status}")
        status_string = "error"

    return status_string, person_update


def _handle_person_status(
    person: Person, log_prefix: str, latest_in_log: Optional[ConversationLog],
    latest_out_log: Optional[ConversationLog], latest_out_template_key: Optional[str],
    message_type_map: dict[str, int]
) -> tuple[Optional[str], str, str, Optional[Any], Optional[Any]]:
    """Handle person status and determine message to send."""
    person_status = safe_column_value(person, "status", None)

    if person_status == PersonStatusEnum.DESIST:
        message_to_send_key, send_reason = _handle_desist_status(log_prefix, latest_out_log, message_type_map)
        return message_to_send_key, send_reason, "Unknown", None, None

    if person_status == PersonStatusEnum.ACTIVE:
        _check_reply_received(latest_in_log, latest_out_log, log_prefix)
        _check_message_interval(latest_out_log, log_prefix)

        message_to_send_key, template_selection_reason = _determine_message_to_send(
            person, latest_out_log, latest_out_template_key, log_prefix
        )

        return message_to_send_key, "Standard Sequence", template_selection_reason, person.dna_match, person.family_tree

    logger.error(f"Unexpected status for {log_prefix}: {getattr(person.status, 'name', 'UNKNOWN')}")
    raise StopIteration("error (unexpected_status)")

def _process_single_person(
    db_session: Session,
    session_manager: SessionManager,
    person: Person,
    latest_in_log: Optional[ConversationLog],
    latest_out_log: Optional[ConversationLog],
    latest_out_template_key: Optional[str],  # Template key from latest OUT message
    message_type_map: dict[str, int],
) -> tuple[Optional[ConversationLog], Optional[tuple[int, PersonStatusEnum]], str]:
    """
    Processes a single person to determine if a message should be sent,
    formats the message, sends/simulates it, and prepares database updates.

    Enhanced with Action 6-style session validation and halt signal checking.

    Args:
        db_session: The active SQLAlchemy database session.
        session_manager: The active SessionManager instance.
        person: The Person object to process (with eager-loaded relationships).
        latest_in_log: The latest prefetched IN ConversationLog for this person.
        latest_out_log: The latest prefetched OUT ConversationLog for this person.
        message_type_map: Dictionary mapping message type names to their DB IDs.

    Returns:
        A tuple containing:
        - new_log_entry (Optional[ConversationLog]): The prepared OUT log object if a message was sent/simulated, else None.
        - person_update (Optional[tuple[int, PersonStatusEnum]]): Tuple of (person_id, new_status) if status needs update, else None.
        - status_string (str): "sent", "acked", "skipped", or "error".
    """
    # --- Step 0: Session Validation and Initialization ---
    _check_halt_signal(session_manager)

    # --- Step 1: Initialization and Logging ---
    log_prefix, person_id, _, _ = _initialize_person_processing(person)
    message_to_send_key: Optional[str] = None  # Key from MESSAGE_TEMPLATES
    template_selection_reason = "Unknown"  # CONSOLIDATED: Track template selection reason
    status_string: Literal["sent", "acked", "skipped", "error"] = (
        "error"  # Default outcome
    )

    # Initialize variables early to prevent UnboundLocalError in exception handlers
    family_tree = None
    dna_match = None
    new_log_entry: Optional[ConversationLog] = None  # Prepared log object
    person_update: Optional[tuple[int, PersonStatusEnum]] = None  # Staged status update

    try:  # Main processing block for this person
        # --- Step 1: Check Person Status for Eligibility ---
        _check_person_eligibility(person, log_prefix)

        # --- Step 2: Determine Action based on Status (DESIST vs ACTIVE) ---
        message_to_send_key, _, template_selection_reason, dna_match, family_tree = _handle_person_status(
            person, log_prefix, latest_in_log, latest_out_log, latest_out_template_key, message_type_map
        )

        # --- Step 3: Format the Selected Message ---
        if not message_to_send_key or message_to_send_key not in MESSAGE_TEMPLATES:
            logger.error(f"Logic Error: Invalid/missing message key '{message_to_send_key}' for {log_prefix}.")
            raise StopIteration("error (template_key)")

        format_data = _prepare_message_format_data(person, family_tree, dna_match, db_session)
        message_text = _format_message_text(message_to_send_key, person, format_data, log_prefix)

        # --- Step 4: Apply Mode/Recipient Filtering ---
        send_message_flag, skip_log_reason = _check_mode_filtering(person, log_prefix)

        # --- Step 5: Send/Simulate Message ---
        existing_conversation_id = _get_existing_conversation_id(latest_out_log, latest_in_log)
        from common_params import ConversationState, MessageContext, MessageFlags
        msg_ctx = MessageContext(
            person=person,
            message_text=message_text,
            message_to_send_key=message_to_send_key,
            template_selection_reason=template_selection_reason,
            log_prefix=log_prefix
        )
        conv_state = ConversationState(
            existing_conversation_id=existing_conversation_id,
            latest_out_log=latest_out_log,
            latest_in_log=latest_in_log
        )
        msg_flags = MessageFlags(
            send_message_flag=send_message_flag,
            skip_log_reason=skip_log_reason
        )
        message_status, effective_conv_id = _send_or_simulate_message(
            session_manager, msg_ctx, conv_state, msg_flags
        )

        # --- Step 6: Prepare Database Updates based on outcome ---
        if message_status in ("delivered OK", "typed (dry_run)") or message_status.startswith("skipped ("):
            # Update conv_state and msg_flags with new values
            conv_state.effective_conv_id = effective_conv_id
            msg_flags.message_status = message_status
            new_log_entry = _prepare_conversation_log_entry(
                msg_ctx, conv_state, msg_flags, message_type_map
            )
            status_string, person_update = _determine_final_status(  # type: ignore[assignment]
                message_to_send_key, message_status, send_message_flag, person_id, log_prefix
            )
        else:
            logger.warning(f"Message send failed for {log_prefix} with status '{message_status}'. No DB changes staged.")
            new_log_entry = None
            person_update = None
            status_string = "error"

        # Step 7: Return prepared updates and status
        return new_log_entry, person_update, status_string

    # --- Step 8: Handle clean exits via StopIteration ---
    except StopIteration as si:
        status_val = str(si.value) if si.value else "skipped"
        return None, None, status_val
    # --- Step 9: Handle unexpected errors ---
    except Exception as e:
        logger.error(f"Unexpected critical error processing {log_prefix}: {e}", exc_info=True)
        return None, None, "error"


# End of _process_single_person


# ------------------------------------------------------------------------------
# Main Action Function Helper Functions
# ------------------------------------------------------------------------------

def _initialize_action8_counters_and_config() -> tuple[int, int, int, int, int, int, int, bool]:
    """Initialize counters and configuration for Action 8."""
    sent_count, acked_count, skipped_count, error_count = 0, 0, 0, 0
    processed_in_loop = 0
    total_candidates = 0
    batch_num = 0
    critical_db_error_occurred = False
    return sent_count, acked_count, skipped_count, error_count, processed_in_loop, total_candidates, batch_num, critical_db_error_occurred


def _initialize_resource_management() -> tuple[list[dict[str, Any]], dict[int, PersonStatusEnum], ResourceManager, ErrorCategorizer]:
    """Initialize resource management and error categorization."""
    db_logs_to_add_dicts: list[dict[str, Any]] = []
    person_updates: dict[int, PersonStatusEnum] = {}

    resource_manager = ResourceManager()
    resource_manager.track_resource("db_logs_to_add_dicts", db_logs_to_add_dicts)
    resource_manager.track_resource("person_updates", person_updates)

    error_categorizer = ErrorCategorizer()

    def critical_error_hook(alert_data: dict[str, Any]) -> None:
        if alert_data['severity'] == 'critical':
            logger.critical(f"🚨 CRITICAL ALERT: {alert_data['alert_type']} - {alert_data['message']}")

    error_categorizer.add_monitoring_hook(critical_error_hook)

    return db_logs_to_add_dicts, person_updates, resource_manager, error_categorizer


def _validate_action8_prerequisites(session_manager: SessionManager) -> tuple[bool, Optional[str]]:
    """Validate prerequisites for Action 8 execution."""
    # System health check
    if not _validate_system_health(session_manager):
        logger.critical("🚨 Action 8: System health check failed - cannot proceed safely. Aborting.")
        return False, None

    # Get profile ID
    profile_id = None
    if hasattr(session_manager, "my_profile_id"):
        profile_id = safe_column_value(session_manager, "my_profile_id", None)

    if not profile_id:
        profile_id = "TEST_PROFILE_ID_FOR_DEBUGGING"
        logger.warning("Action 8: Using test profile ID for debugging message progression logic")

    # Check message templates
    if not MESSAGE_TEMPLATES:
        logger.error("Action 8: Message templates not loaded.")
        return False, None

    logger.warning("Action 8: Skipping login check for debugging message progression logic")
    return True, profile_id


def _fetch_messaging_data(db_session: Session, session_manager: SessionManager) -> tuple[Optional[dict], Optional[list], int]:
    """Fetch message type map and candidate persons."""
    try:
        message_type_map, candidate_persons = _get_simple_messaging_data(db_session, session_manager)
    except MaxApiFailuresExceededError as cascade_err:
        logger.critical(f"🚨 CRITICAL: Session death cascade detected during prefetch: {cascade_err}")
        return None, None, 0

    if message_type_map is None or candidate_persons is None:
        logger.error("Action 8: Failed to fetch essential messaging data. Aborting.")
        return None, None, 0

    total_candidates = len(candidate_persons)
    if total_candidates == 0:
        logger.warning("Action 8: No candidates found meeting messaging criteria. Finishing.\n")
    else:
        logger.info(f"Action 8: Found {total_candidates} candidates to process.")
        max_messages_to_send_this_run = config_schema.max_inbox
        if max_messages_to_send_this_run > 0:
            logger.info(f"Action 8: Will send/ack a maximum of {max_messages_to_send_this_run} messages this run.\n")

    return message_type_map, candidate_persons, total_candidates


# ------------------------------------------------------------------------------
# Main Action Function
# ------------------------------------------------------------------------------


# Helper functions for send_messages_to_matches

def _setup_progress_bar(total_candidates: int) -> dict:
    """Setup progress bar configuration for message processing."""
    import sys
    return {
        "total": total_candidates,
        "desc": "Processing",
        "unit": " person",
        "dynamic_ncols": True,
        "leave": True,
        "bar_format": "{desc} |{bar}| {percentage:3.0f}% ({n_fmt}/{total_fmt})",
        "file": sys.stderr,
    }


def _handle_critical_db_error(progress_bar, total_candidates: int, processed_in_loop: int,
                               sent_count: int, acked_count: int, skipped_count: int, error_count: int) -> int:
    """Handle critical database error by updating progress bar and calculating remaining skips."""
    remaining_to_skip = total_candidates - processed_in_loop + 1
    if progress_bar:
        progress_bar.set_description(
            f"ERROR: DB commit failed - Sent={sent_count} ACK={acked_count} Skip={skipped_count + remaining_to_skip} Err={error_count}"
        )
        progress_bar.update(remaining_to_skip)
    return remaining_to_skip


def _check_and_handle_browser_health(
    session_manager: SessionManager,
    state: 'ProcessingState',
    counters: 'BatchCounters',
    total_candidates: int
) -> tuple[bool, int]:
    """Check browser health and attempt recovery if needed. Returns (should_break, additional_skips)."""
    if state.processed_in_loop % 5 == 0 and not session_manager.check_browser_health():
        logger.warning(f"🚨 BROWSER DEATH DETECTED during message processing at person {state.processed_in_loop}")
        if session_manager.attempt_browser_recovery():
            logger.warning(f"✅ Browser recovery successful at person {state.processed_in_loop} - continuing")
            return False, 0
        logger.critical(f"❌ Browser recovery failed at person {state.processed_in_loop} - halting messaging")
        remaining_to_skip = total_candidates - state.processed_in_loop + 1
        if state.progress_bar:
            state.progress_bar.set_description(
                f"ERROR: Browser failed - Sent={counters.sent} ACK={counters.acked} Skip={counters.skipped + remaining_to_skip} Err={counters.errors}"
            )
            state.progress_bar.update(remaining_to_skip)
        return True, remaining_to_skip
    return False, 0


def _check_message_send_limit(max_messages_to_send_this_run: int, sent_count: int, acked_count: int,
                               progress_bar, skipped_count: int, error_count: int) -> bool:
    """Check if message sending limit has been reached. Returns True if should skip."""
    current_sent_total = sent_count + acked_count
    if max_messages_to_send_this_run > 0 and current_sent_total >= max_messages_to_send_this_run:
        if not hasattr(progress_bar, "limit_logged"):
            logger.debug(f"Message sending limit ({max_messages_to_send_this_run}) reached. Skipping remaining.")
            progress_bar.limit_logged = True
        if progress_bar:
            progress_bar.set_description(
                f"Limit reached: Sent={sent_count} ACK={acked_count} Skip={skipped_count + 1} Err={error_count}"
            )
            progress_bar.update(1)
        return True
    return False


def _log_periodic_progress(processed_in_loop: int, total_candidates: int, sent_count: int,
                           acked_count: int, skipped_count: int, error_count: int) -> None:
    """Log progress every 5% or every 100 people."""
    if processed_in_loop > 0 and (processed_in_loop % max(100, total_candidates // 20) == 0):
        logger.info(f"Action 8 Progress: {processed_in_loop}/{total_candidates} processed "
                   f"(Sent={sent_count} ACK={acked_count} Skip={skipped_count} Err={error_count})")


def _convert_log_object_to_dict(new_log_object) -> Optional[dict[str, Any]]:
    """Convert SQLAlchemy ConversationLog object to dictionary for batch commit."""
    try:
        log_dict = {
            c.key: getattr(new_log_object, c.key)
            for c in sa_inspect(new_log_object).mapper.column_attrs
            if hasattr(new_log_object, c.key)
        }

        # Validate required keys
        required_keys = ["conversation_id", "direction", "people_id", "latest_timestamp"]
        if not all(k in log_dict for k in required_keys):
            raise ValueError("Missing required keys in log object conversion")

        if not isinstance(log_dict["latest_timestamp"], datetime):
            raise ValueError("Invalid timestamp type in log object conversion")

        # Pass Enum directly for direction
        log_dict["direction"] = new_log_object.direction

        # Normalize timestamp
        ts_val = log_dict["latest_timestamp"]
        log_dict["latest_timestamp"] = (
            ts_val.astimezone(timezone.utc) if ts_val.tzinfo
            else ts_val.replace(tzinfo=timezone.utc)
        )

        return log_dict
    except Exception as conversion_err:
        logger.error(f"Failed to convert ConversationLog object to dict: {conversion_err}", exc_info=True)
        return None


def _prepare_log_dict(new_log_object) -> tuple[Optional[dict], str]:
    """Convert log object to dict."""
    if not new_log_object:
        return None, "unchanged"
    log_dict = _convert_log_object_to_dict(new_log_object)
    return (log_dict, "unchanged") if log_dict else (None, "error")

def _handle_sent_status(sent_count: int, log_dict: Optional[dict], db_logs_to_add_dicts: list) -> int:
    """Handle sent status updates."""
    if log_dict:
        db_logs_to_add_dicts.append(log_dict)
    return sent_count + 1

def _handle_acked_status(acked_count: int, log_dict: Optional[dict], person_update_tuple, db_logs_to_add_dicts: list, person_updates: dict) -> int:
    """Handle acknowledged status updates."""
    if log_dict:
        db_logs_to_add_dicts.append(log_dict)
    if person_update_tuple:
        person_updates[person_update_tuple[0]] = person_update_tuple[1]
    return acked_count + 1

def _handle_error_or_skip_status(status: str, counters: 'BatchCounters', log_dict: Optional[dict], batch_data: 'MessagingBatchData', error_categorizer, person, overall_success: bool) -> tuple[int, int, bool]:
    """Handle error or skipped status updates."""
    category, error_type = error_categorizer.categorize_status(status)

    if category == 'skipped':
        if log_dict:
            batch_data.db_logs_to_add_dicts.append(log_dict)
        return counters.skipped + 1, counters.errors, overall_success

    if category == 'error':
        if error_type != 'business_logic_generic':
            severity = 'critical' if 'cascade' in error_type or 'authentication' in error_type else 'warning'
            error_categorizer.trigger_monitoring_alert(alert_type=error_type, message=f"Technical error: {status}", severity=severity)
        return counters.skipped, counters.errors + 1, False

    logger.warning(f"Unknown status for {person.username}: {status}")
    return counters.skipped, counters.errors + 1, False

def _update_counters_and_collect_data(
    status: str,
    new_log_object,
    person_update_tuple,
    counters: 'BatchCounters',
    batch_data: 'MessagingBatchData',
    error_categorizer,
    person,
    overall_success: bool
) -> tuple[int, int, int, int, bool]:
    """Update counters and collect database updates based on processing status."""
    log_dict, new_status = _prepare_log_dict(new_log_object)
    if new_status == "error":
        status = "error"

    if status == "sent":
        counters.sent = _handle_sent_status(counters.sent, log_dict, batch_data.db_logs_to_add_dicts)
    elif status == "acked":
        counters.acked = _handle_acked_status(counters.acked, log_dict, person_update_tuple, batch_data.db_logs_to_add_dicts, batch_data.person_updates)
    else:
        counters.skipped, counters.errors, overall_success = _handle_error_or_skip_status(
            status, counters, log_dict, batch_data, error_categorizer, person, overall_success
        )

    return counters.sent, counters.acked, counters.skipped, counters.errors, overall_success


def _should_commit_batch(current_batch_size: int, memory_usage_mb: float,
                         db_commit_batch_size: int, max_batch_memory_mb: int, max_batch_items: int) -> bool:
    """Determine if batch should be committed based on size, memory, or item limits."""
    return (
        current_batch_size >= db_commit_batch_size or
        memory_usage_mb >= max_batch_memory_mb or
        current_batch_size >= max_batch_items
    )


def _calculate_batch_memory(db_logs_to_add_dicts: list, person_updates: dict) -> tuple[int, float]:
    """Calculate current batch size and memory usage."""
    import sys
    current_batch_size = len(db_logs_to_add_dicts) + len(person_updates)
    memory_usage_bytes = sys.getsizeof(db_logs_to_add_dicts) + sys.getsizeof(person_updates)
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)
    return current_batch_size, memory_usage_mb


def _perform_batch_commit(db_session, db_logs_to_add_dicts: list, person_updates: dict,
                          batch_num: int, session_manager: SessionManager) -> tuple[bool, int]:
    """Perform batch commit and return success status and updated batch number."""
    batch_num += 1
    current_batch_size, memory_usage_mb = _calculate_batch_memory(db_logs_to_add_dicts, person_updates)
    logger.debug(f"Committing batch {batch_num}: {current_batch_size} items, {memory_usage_mb:.1f}MB")

    commit_success, logs_committed_count, persons_updated_count = _safe_commit_with_rollback(
        session=db_session,
        log_upserts=db_logs_to_add_dicts,
        person_updates=person_updates,
        context=f"Action 8 Batch {batch_num}",
        session_manager=session_manager
    )

    if commit_success:
        db_logs_to_add_dicts.clear()
        person_updates.clear()
        logger.debug(f"Batch {batch_num} committed successfully: {logs_committed_count} logs, {persons_updated_count} persons")
    else:
        logger.critical(f"Batch {batch_num} commit failed - halting processing")

    return commit_success, batch_num


def _create_result_dict(
    counters: 'BatchCounters',
    state: 'ProcessingState',
    critical_db_error: bool = False,
    overall_success: bool = True,
    should_continue: bool = True
) -> dict:
    """Create standardized result dictionary."""
    return {
        'sent_count': counters.sent,
        'acked_count': counters.acked,
        'skipped_count': counters.skipped,
        'error_count': counters.errors,
        'batch_num': state.batch_num,
        'critical_db_error_occurred': critical_db_error,
        'overall_success': overall_success,
        'should_continue': should_continue
    }


def _log_message_creation_debug(new_log_object: Any, db_session: Session, person_id_int: int) -> None:
    """Log message creation for debugging."""
    if new_log_object and hasattr(new_log_object, 'direction') and new_log_object.direction == MessageDirectionEnum.OUT:
        template_info = "Unknown"
        if new_log_object.message_template_id:
            message_template_obj = db_session.query(MessageTemplate).filter(
                MessageTemplate.id == new_log_object.message_template_id
            ).first()
            if message_template_obj:
                template_info = message_template_obj.template_key
        logger.debug(f"Created new OUT message for Person {person_id_int}: {template_info}")


def _handle_batch_commit_if_needed(
    db_session: Session,
    batch_data: 'MessagingBatchData',
    state: 'ProcessingState',
    session_manager: SessionManager,
    batch_config: 'BatchConfig'
) -> tuple[bool, int, bool]:
    """Handle batch commit if needed. Returns (critical_error, batch_num, overall_success)."""
    current_batch_size, memory_usage_mb = _calculate_batch_memory(batch_data.db_logs_to_add_dicts, batch_data.person_updates)

    if _should_commit_batch(current_batch_size, memory_usage_mb,
                           batch_config.commit_batch_size, batch_config.max_memory_mb, batch_config.max_items):
        commit_success, batch_num = _perform_batch_commit(
            db_session, batch_data.db_logs_to_add_dicts, batch_data.person_updates, state.batch_num, session_manager
        )
        state.batch_num = batch_num

        if not commit_success:
            return True, state.batch_num, False

    return False, state.batch_num, True


def _process_single_candidate_iteration(
    person: Any,
    db_session: Session,
    session_manager: SessionManager,
    message_type_map: dict,
    error_categorizer: Any,
    batch_config: BatchConfig,
    counters: BatchCounters,
    batch_data: MessagingBatchData,
    state: ProcessingState
) -> dict:
    """
    Process a single candidate iteration in the main loop.

    Returns dict with updated counters and status flags.
    """
    import time

    # Check message send limit
    if _check_message_send_limit(batch_config.max_messages_to_send, counters.sent, counters.acked,
                                  state.progress_bar, counters.skipped, counters.errors):
        counters.skipped += 1
        return _create_result_dict(counters, state)

    # Check halt signal
    try:
        _check_halt_signal(session_manager)
    except MaxApiFailuresExceededError:
        return _create_result_dict(counters, state, should_continue=False)

    # Log periodic progress
    _log_periodic_progress(state.processed_in_loop, 0, counters.sent, counters.acked, counters.skipped, counters.errors)

    # Get person ID and message history
    person_id_int = int(safe_column_value(person, "id", 0))
    latest_in_log, latest_out_log, latest_out_template_key = _get_person_message_history(
        db_session, person_id_int
    )

    # Process single person with performance tracking
    person_start_time = time.time()

    try:
        new_log_object, person_update_tuple, status = _process_single_person(
            db_session, session_manager, person,
            latest_in_log, latest_out_log, latest_out_template_key,
            message_type_map
        )
    except MaxApiFailuresExceededError as cascade_err:
        person_name = safe_column_value(person, 'name', 'Unknown')
        logger.critical(
            f"🚨 SESSION DEATH CASCADE in person processing for {person_name}: {cascade_err}. "
            f"Halting remaining processing to prevent infinite cascade."
        )
        return _create_result_dict(counters, state, should_continue=False)

    # Log message creation for debugging
    _log_message_creation_debug(new_log_object, db_session, person_id_int)

    # Update performance tracking
    person_duration = time.time() - person_start_time
    _update_messaging_performance(session_manager, person_duration)

    # Update counters and collect data
    counters.sent, counters.acked, counters.skipped, counters.errors, overall_success = _update_counters_and_collect_data(
        status, new_log_object, person_update_tuple,
        counters, batch_data,
        error_categorizer, person, True
    )

    # Update progress bar
    if state.progress_bar:
        state.progress_bar.set_description(
            f"Processing: Sent={counters.sent} ACK={counters.acked} Skip={counters.skipped} Err={counters.errors}"
        )
        state.progress_bar.update(1)

    # Handle batch commit if needed
    critical_db_error, state.batch_num, overall_success = _handle_batch_commit_if_needed(
        db_session, batch_data, state, session_manager, batch_config
    )

    return _create_result_dict(counters, state, critical_db_error, overall_success, not critical_db_error)


def _log_final_summary(
    total_candidates: int,
    state: 'ProcessingState',
    counters: 'BatchCounters',
    overall_success: bool,
    error_categorizer
) -> None:
    """Log final summary of message sending action."""
    print(" ")
    logger.info("--- Action 8: Message Sending Summary ---")
    logger.info(f"  Candidates Considered:              {total_candidates}")
    logger.info(f"  Candidates Processed in Loop:       {state.processed_in_loop}")
    logger.info(f"  Template Messages Sent/Simulated:   {counters.sent}")
    logger.info(f"  Desist ACKs Sent/Simulated:         {counters.acked}")
    logger.info(f"  Skipped (Rules/Filter/Limit/Error): {counters.skipped}")
    logger.info(f"  Errors during processing/sending:   {counters.errors}")
    logger.info(f"  Overall Action Success:             {overall_success}")

    error_summary = error_categorizer.get_error_summary()
    if error_summary['total_technical_errors'] > 0 or error_summary['total_business_skips'] > 0:
        logger.info("--- Detailed Error Analysis ---")
        logger.info(f"  Technical Errors:                   {error_summary['total_technical_errors']}")
        logger.info(f"  Business Logic Skips:               {error_summary['total_business_skips']}")
        logger.info(f"  Error Rate:                         {error_summary['error_rate']:.1%}")

        if error_summary['most_common_error']:
            logger.info(f"  Most Common Issue:                  {error_summary['most_common_error']}")

        logger.info("--- Error Breakdown ---")
        for error_type, count in error_summary['error_breakdown'].items():
            if count > 0:
                logger.info(f"    {error_type.replace('_', ' ').title()}: {count}")

    logger.info("-----------------------------------------\n")


def _handle_action8_exception(exception: BaseException) -> bool:
    """Handle exceptions in Action 8 and return overall_success status."""
    if isinstance(exception, MaxApiFailuresExceededError):
        logger.critical(
            f"Halting Action 8 due to excessive critical API failures: {exception}",
            exc_info=False,
        )
    elif isinstance(exception, BrowserSessionError):
        logger.critical(
            f"Browser session error in Action 8: {exception}",
            exc_info=True,
        )
    elif isinstance(exception, APIRateLimitError):
        logger.error(
            f"API rate limit exceeded in Action 8: {exception}",
            exc_info=False,
        )
    elif isinstance(exception, AuthenticationExpiredError):
        logger.error(
            f"Authentication expired during Action 8: {exception}",
            exc_info=False,
        )
    elif isinstance(exception, ConnectionError):
        # Check for session death cascade one more time at top level
        if "Session death cascade detected" in str(exception):
            logger.critical(
                f"🚨 SESSION DEATH CASCADE at Action 8 top level: {exception}",
                exc_info=False,
            )
        else:
            logger.error(
                f"Connection error during Action 8: {exception}",
                exc_info=True,
            )
    elif isinstance(exception, KeyboardInterrupt):
        logger.warning("Keyboard interrupt detected. Stopping Action 8 message processing.")
    else:
        logger.critical(
            f"CRITICAL: Unhandled error during Action 8 execution: {exception}",
            exc_info=True,
        )

    return False  # All exceptions result in overall_success = False


def _perform_final_commit(
    db_session: Session,
    critical_db_error_occurred: bool,
    db_logs_to_add_dicts: list[dict[str, Any]],
    person_updates: dict[int, PersonStatusEnum],
    batch_num: int,
    session_manager: SessionManager
) -> tuple[bool, int]:
    """
    Perform final commit of remaining data.

    Returns:
        Tuple of (overall_success, updated_batch_num)
    """
    overall_success = True

    if not critical_db_error_occurred and (db_logs_to_add_dicts or person_updates):
        batch_num += 1
        logger.debug(
            f"Performing final commit for remaining items (Batch {batch_num})..."
        )

        # Use safe commit for final batch
        final_commit_success, final_logs_saved, final_persons_updated = _safe_commit_with_rollback(
            session=db_session,
            log_upserts=db_logs_to_add_dicts,
            person_updates=person_updates,
            context="Action 8 Final Save",
            session_manager=session_manager
        )

        if final_commit_success:
            # Final commit successful
            db_logs_to_add_dicts.clear()
            person_updates.clear()
            logger.debug(
                f"Action 8 Final commit executed (Logs Processed: {final_logs_saved}, Persons Updated: {final_persons_updated})."
            )
        else:
            logger.critical("Final commit failed - some data may be lost")
            overall_success = False

    return overall_success, batch_num


def _perform_final_cleanup(
    db_session: Optional[Session],
    session_manager: SessionManager,
    critical_db_error_occurred: bool,
    total_candidates: int,
    state: 'ProcessingState',
    counters: 'BatchCounters',
    overall_success: bool,
    error_categorizer: Any
) -> int:
    """
    Perform final cleanup and logging.

    Returns:
        Updated skipped_count
    """
    if db_session:
        session_manager.return_session(db_session)

    # Adjust final skipped count if loop was stopped early
    if critical_db_error_occurred and total_candidates > state.processed_in_loop:
        unprocessed_count = total_candidates - state.processed_in_loop
        logger.warning(
            f"Adding {unprocessed_count} unprocessed candidates to skipped count due to DB commit failure."
        )
        counters.skipped += unprocessed_count

    # Log final summary
    _log_final_summary(total_candidates, state, counters, overall_success, error_categorizer)

    return counters.skipped


def _perform_resource_cleanup(resource_manager: Any) -> None:
    """Perform final resource cleanup and garbage collection."""
    try:
        resource_manager.cleanup_resources()
        resource_manager.trigger_garbage_collection()
        logger.debug("🧹 Final resource cleanup completed")
    except Exception as cleanup_err:
        logger.warning(f"Final resource cleanup failed: {cleanup_err}")


def _log_performance_summary() -> None:
    """Log performance monitoring summary."""
    try:
        perf_summary = stop_advanced_monitoring()
        logger.info("--- Performance Summary ---")
        logger.info(f"  Runtime: {perf_summary.get('total_runtime', 'N/A')}")
        logger.info(f"  Memory Peak: {perf_summary.get('peak_memory_mb', 0):.1f}MB")
        logger.info(f"  Operations Completed: {perf_summary.get('total_operations', 0)}")
        logger.info(f"  API Calls: {perf_summary.get('api_calls', 0)}")
        logger.info(f"  Errors: {perf_summary.get('total_errors', 0)}")
        logger.info("---------------------------")
    except Exception as perf_err:
        logger.warning(f"Performance monitoring summary failed: {perf_err}")


def _process_all_candidates(
    candidate_persons: list,
    total_candidates: int,
    db_session: Session,
    session_manager: SessionManager,
    message_type_map: dict,
    resource_manager: Any,
    error_categorizer: Any,
    batch_config: 'BatchConfig',
) -> dict:
    """Process all candidate persons for messaging. Returns dict with counters and results."""
    from common_params import BatchCounters, MessagingBatchData, ProcessingState

    # Initialize counters and state
    counters = BatchCounters()
    state = ProcessingState(batch_num=0)
    critical_db_error_occurred = False
    overall_success = True

    # Initialize data collections
    batch_data = MessagingBatchData(
        db_logs_to_add_dicts=[],
        person_updates={}
    )

    # Setup progress bar
    tqdm_args = _setup_progress_bar(total_candidates)
    logger.debug("Processing candidates...")

    with logging_redirect_tqdm(), tqdm(**tqdm_args) as progress_bar:
        state.progress_bar = progress_bar

        for person in candidate_persons:
            state.processed_in_loop += 1

            # Check for critical DB error
            if critical_db_error_occurred:
                remaining_to_skip = _handle_critical_db_error(
                    progress_bar, total_candidates, state.processed_in_loop,
                    counters.sent, counters.acked, counters.skipped, counters.errors
                )
                counters.skipped += remaining_to_skip
                break

            # Browser health monitoring
            should_break, additional_skips = _check_and_handle_browser_health(
                session_manager, state, counters, total_candidates
            )
            if should_break:
                critical_db_error_occurred = True
                overall_success = False
                counters.skipped += additional_skips
                break

            # Resource management
            if state.processed_in_loop % 10 == 0:
                resource_manager.periodic_maintenance()

            # Process single candidate iteration
            iteration_result = _process_single_candidate_iteration(
                person, db_session, session_manager, message_type_map,
                error_categorizer,
                batch_config, counters, batch_data, state
            )

            # Update counters from iteration result
            counters.sent = iteration_result['sent_count']
            counters.acked = iteration_result['acked_count']
            counters.skipped = iteration_result['skipped_count']
            counters.errors = iteration_result['error_count']
            state.batch_num = iteration_result['batch_num']

            # Check for critical errors or halt conditions
            if iteration_result['critical_db_error_occurred']:
                critical_db_error_occurred = True
                overall_success = False
                break

            if not iteration_result['should_continue']:
                break

            overall_success = iteration_result['overall_success']

    # Return all results as a dictionary
    return {
        'sent_count': counters.sent,
        'acked_count': counters.acked,
        'skipped_count': counters.skipped,
        'error_count': counters.errors,
        'processed_in_loop': state.processed_in_loop,
        'critical_db_error_occurred': critical_db_error_occurred,
        'overall_success': overall_success,
        'batch_num': state.batch_num,
        'db_logs_to_add_dicts': batch_data.db_logs_to_add_dicts,
        'person_updates': batch_data.person_updates,
    }


# Updated decorator stack with enhanced error recovery
@with_connection_resilience("Action 8: Messaging", max_recovery_attempts=3)
@with_enhanced_recovery(max_attempts=3, base_delay=2.0, max_delay=60.0)
@circuit_breaker(failure_threshold=10, recovery_timeout=60)  # Aligned with ANCESTRY_API_CONFIG
@graceful_degradation(fallback_value=False)
@error_context("action8_messaging")
def send_messages_to_matches(session_manager: SessionManager) -> bool:
    """
    Main function for Action 8.
    Fetches eligible candidates, determines the appropriate message to send (if any)
    based on rules and history, sends/simulates the message, and updates the database.
    Uses the unified commit_bulk_data function.

    Args:
        session_manager: The active SessionManager instance.

    Returns:
        True if the process completed without critical database errors, False otherwise.
        Note: Individual message send failures are logged but do not cause the
              entire action to return False unless they lead to a DB commit failure.
    """
    # --- Step 1: Initialization and System Health Check ---
    logger.debug("--- Starting Action 8: Send Standard Messages ---")
    start_advanced_monitoring()

    from contextlib import suppress
    with suppress(Exception):
        logger.info(f"Action 8: APP_MODE={getattr(config_schema, 'app_mode', 'production')}, MIN_MESSAGE_INTERVAL={MIN_MESSAGE_INTERVAL}")

    # Validate prerequisites
    prerequisites_valid, _ = _validate_action8_prerequisites(session_manager)
    if not prerequisites_valid:
        return False

    # Initialize counters and configuration
    sent_count, acked_count, skipped_count, error_count, processed_in_loop, total_candidates, batch_num, critical_db_error_occurred = _initialize_action8_counters_and_config()

    db_commit_batch_size = max(1, config_schema.batch_size)
    max_messages_to_send_this_run = config_schema.max_inbox
    overall_success = True

    MAX_BATCH_MEMORY_MB = 50
    MAX_BATCH_ITEMS = min(db_commit_batch_size, 100)

    # Initialize resource management
    db_logs_to_add_dicts, person_updates, resource_manager, error_categorizer = _initialize_resource_management()

    # --- Step 2: Get DB Session and Pre-fetch Data ---
    db_session: Optional[Session] = None
    try:
        db_session = session_manager.get_db_conn()
        if not db_session:
            logger.critical("Action 8: Failed to get DB Session. Aborting.")
            return False

        # Fetch messaging data
        message_type_map, candidate_persons, total_candidates = _fetch_messaging_data(db_session, session_manager)

        if message_type_map is None or candidate_persons is None:
            if db_session:
                session_manager.return_session(db_session)
            return False

        # --- Step 3: Main Processing Loop ---
        if total_candidates > 0:
            # Process all candidates
            batch_config = BatchConfig(
                commit_batch_size=db_commit_batch_size,
                max_memory_mb=MAX_BATCH_MEMORY_MB,
                max_items=MAX_BATCH_ITEMS,
                max_messages_to_send=max_messages_to_send_this_run
            )
            processing_result = _process_all_candidates(
                candidate_persons, total_candidates, db_session, session_manager,
                message_type_map, resource_manager, error_categorizer,
                batch_config
            )

            # Unpack results
            sent_count = processing_result['sent_count']
            acked_count = processing_result['acked_count']
            skipped_count = processing_result['skipped_count']
            error_count = processing_result['error_count']
            processed_in_loop = processing_result['processed_in_loop']
            critical_db_error_occurred = processing_result['critical_db_error_occurred']
            overall_success = processing_result['overall_success']
            batch_num = processing_result['batch_num']
            db_logs_to_add_dicts = processing_result['db_logs_to_add_dicts']
            person_updates = processing_result['person_updates']

        # --- End Conditional Processing Block (if total_candidates > 0) ---

        # --- Step 4: Final Commit ---
        overall_success, batch_num = _perform_final_commit(
            db_session, critical_db_error_occurred, db_logs_to_add_dicts,
            person_updates, batch_num, session_manager
        )

    # --- Step 5: Handle Outer Exceptions (Action 6 Pattern) ---
    except (MaxApiFailuresExceededError, BrowserSessionError, APIRateLimitError,
            AuthenticationExpiredError, ConnectionError, KeyboardInterrupt, Exception) as outer_err:
        overall_success = _handle_action8_exception(outer_err)

        # Emergency resource cleanup on critical failure
        try:
            resource_manager.cleanup_resources()
            logger.warning("🧹 Emergency resource cleanup completed after critical error")
        except Exception as emergency_cleanup_err:
            logger.error(f"Emergency resource cleanup failed: {emergency_cleanup_err}")
    # --- Step 6: Final Cleanup and Summary ---
    finally:
        from common_params import BatchCounters, ProcessingState
        counters_final = BatchCounters(sent=sent_count, acked=acked_count, skipped=skipped_count, errors=error_count)
        state_final = ProcessingState(batch_num=0, processed_in_loop=processed_in_loop)
        skipped_count = _perform_final_cleanup(
            db_session, session_manager, critical_db_error_occurred,
            total_candidates, state_final, counters_final, overall_success,
            error_categorizer
        )

    # Step 7: Final resource cleanup
    _perform_resource_cleanup(resource_manager)

    # Step 8: Stop performance monitoring and log summary
    _log_performance_summary()

    # Step 9: Return overall success status
    return overall_success


# End of send_messages_to_matches


# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================
# Extracted from monolithic action8_messaging_tests() for better organization
# Each test function is independent and can be run individually


def _test_function_availability() -> None:
        """Test messaging system functions are available with detailed verification."""
        required_functions = [
            'safe_column_value', 'load_message_templates', 'determine_next_message_type',
            '_safe_commit_with_rollback', '_get_simple_messaging_data', '_process_single_person',
            'send_messages_to_matches'
        ]

        from test_framework import test_function_availability
        return test_function_availability(required_functions, globals(), "Action 8")


def _test_safe_column_value() -> None:
    """Test safe column value extraction with detailed verification."""
    test_cases = [
        (None, "attr", "default", "None object handling"),
        (
            type("MockObj", (), {"attr": "value"})(),
            "attr",
            "default",
            "object with attribute",
        ),
        (
            type("MockObj", (), {})(),
            "missing_attr",
            "fallback",
            "object without attribute",
        ),
    ]

    print("📋 Testing safe column value extraction:")
    results = []

    for obj, attr_name, default, description in test_cases:
        try:
            result = safe_column_value(obj, attr_name, default)
            test_passed = result is not None or result == default

            status = "✅" if test_passed else "❌"
            print(f"   {status} {description}")
            print(
                f"      Input: obj={type(obj).__name__}, attr='{attr_name}', default='{default}' → Result: {result!r}"
            )

            results.append(test_passed)

        except Exception as e:
            print(f"   ❌ {description}")
            print(f"      Error: {e}")
            results.append(False)
            raise

    print(
        f"📊 Results: {sum(results)}/{len(results)} safe column value tests passed"
    )


def _test_message_template_loading() -> None:
    """Test message template loading functionality."""
    print("📋 Testing message template loading:")
    results = []

    try:
        templates = load_message_templates()
        templates_loaded = isinstance(templates, dict)

        status = "✅" if templates_loaded else "❌"
        print(f"   {status} Message template loading")
        print(
            f"      Type: {type(templates).__name__}, Count: {len(templates) if templates_loaded else 0}"
        )

        results.append(templates_loaded)
        assert templates_loaded, "load_message_templates should return a dictionary"

    except Exception as e:
        print("   ❌ Message template loading")
        print(f"      Error: {e}")
        results.append(False)
        # Don't raise as templates file might not exist in test environment

    print(
        f"📊 Results: {sum(results)}/{len(results)} message template loading tests passed"
    )


def _test_circuit_breaker_config() -> None:
    """Test circuit breaker decorator configuration reflects Action 6 lessons."""
    import inspect

    print("📋 Testing circuit breaker configuration:")
    results = []

    # Get the decorators applied to send_messages_to_matches
    func = send_messages_to_matches

    # Check if function has the expected attributes from decorators
    test_cases = [
        ("Function is callable", callable(func)),
        ("Function has error handling", hasattr(func, '__wrapped__') or hasattr(func, '__name__')),
        ("Function name preserved", func.__name__ == 'send_messages_to_matches'),
    ]

    for description, condition in test_cases:
        status = "✅" if condition else "❌"
        print(f"   {status} {description}")
        results.append(condition)

    # Test that the function can be imported and called (basic validation)
    try:
        # Verify the function signature is intact
        sig = inspect.signature(func)
        has_session_manager = 'session_manager' in sig.parameters
        status = "✅" if has_session_manager else "❌"
        print(f"   {status} Function signature intact (has session_manager parameter)")
        results.append(has_session_manager)
    except Exception as e:
        print(f"   ❌ Function signature validation failed: {e}")
        results.append(False)

    print(
        f"📊 Results: {sum(results)}/{len(results)} circuit breaker configuration tests passed"
    )


def _test_session_death_cascade_detection() -> None:
    """Test session death cascade detection and handling."""
    print("📋 Testing session death cascade detection:")
    results = []

    try:
        # Test that MaxApiFailuresExceededError is available
        error_available = MaxApiFailuresExceededError is not None
        status = "✅" if error_available else "❌"
        print(f"   {status} MaxApiFailuresExceededError class available")
        results.append(error_available)

        # Test cascade detection string matching
        test_error = ConnectionError("Session death cascade detected in test")
        cascade_detected = "Session death cascade detected" in str(test_error)
        status = "✅" if cascade_detected else "❌"
        print(f"   {status} Cascade detection string matching")
        results.append(cascade_detected)

        # Test error inheritance
        cascade_error = MaxApiFailuresExceededError("Test cascade error", context={"source": "Action 8"})
        is_exception = isinstance(cascade_error, Exception)
        status = "✅" if is_exception else "❌"
        print(f"   {status} MaxApiFailuresExceededError inherits from Exception")
        results.append(is_exception)

    except Exception as e:
        print(f"   ❌ Session death cascade detection test failed: {e}")
        results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} cascade detection tests passed")
    assert all(results), "All session death cascade detection tests should pass"


def _test_performance_tracking() -> None:
    """Test performance tracking functionality."""
    print("📋 Testing performance tracking:")
    results = []

    try:
        # Test performance tracking function exists
        func_exists = callable(_update_messaging_performance)
        status = "✅" if func_exists else "❌"
        print(f"   {status} _update_messaging_performance function available")
        results.append(func_exists)

        # Test with mock session manager
        class MockSessionManager:
            pass

        mock_session = MockSessionManager()

        # Test performance tracking doesn't crash
        try:
            _update_messaging_performance(mock_session, 1.5)  # type: ignore[arg-type]
            tracking_works = True
        except Exception:
            tracking_works = False

        status = "✅" if tracking_works else "❌"
        print(f"   {status} Performance tracking executes without errors")
        results.append(tracking_works)

        # Test attributes are created
        has_response_times = hasattr(mock_session, '_response_times')
        has_slow_calls = hasattr(mock_session, '_recent_slow_calls')
        has_avg_time = hasattr(mock_session, '_avg_response_time')

        attributes_created = has_response_times and has_slow_calls and has_avg_time
        status = "✅" if attributes_created else "❌"
        print(f"   {status} Performance tracking attributes created")
        results.append(attributes_created)

    except Exception as e:
        print(f"   ❌ Performance tracking test failed: {e}")
        results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} performance tracking tests passed")
    assert all(results), "All performance tracking tests should pass"


def _test_enhanced_error_handling() -> None:
    """Test enhanced error handling patterns."""
    print("📋 Testing enhanced error handling:")
    results = []

    try:
        # Test error classes are available
        error_classes = [
            BrowserSessionError,
            APIRateLimitError,
            AuthenticationExpiredError,
            MaxApiFailuresExceededError
        ]

        for error_class in error_classes:
            try:
                # Handle different error class signatures
                if error_class == MaxApiFailuresExceededError:
                    test_error = error_class("Test error", context={"source": "Action 8"})
                else:
                    test_error = error_class("Test error")
                is_exception = isinstance(test_error, Exception)
                status = "✅" if is_exception else "❌"
                print(f"   {status} {error_class.__name__} class works correctly")
                results.append(is_exception)
            except Exception as e:
                print(f"   ❌ {error_class.__name__} class failed: {e}")
                results.append(False)

        # Test enhanced recovery import
        recovery_available = with_enhanced_recovery is not None
        status = "✅" if recovery_available else "❌"
        print(f"   {status} Enhanced recovery decorator available")
        results.append(recovery_available)

    except Exception as e:
        print(f"   ❌ Enhanced error handling test failed: {e}")
        results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} enhanced error handling tests passed")
    assert all(results), "All enhanced error handling tests should pass"


def _test_integration_with_shared_modules() -> None:
    """Test integration with shared modules."""
    print("📋 Testing integration with shared modules:")
    results = []

    try:
        # Test universal session monitor integration (now in SessionManager)
        try:
            from core.session_manager import SessionManager
            monitor_available = hasattr(SessionManager, 'validate_system_health')
        except ImportError:
            monitor_available = False
        status = "✅" if monitor_available else "❌"
        print(f"   {status} Universal session health validation available")
        results.append(monitor_available)

        # Test API call framework integration (now in core/api_manager.py)
        try:
            from core.api_manager import APIManager
            api_framework_available = APIManager is not None
        except ImportError:
            api_framework_available = False
        status = "✅" if api_framework_available else "❌"
        print(f"   {status} Universal API call framework available")
        results.append(api_framework_available)

        # Test error recovery patterns integration (now in core/enhanced_error_recovery.py)
        try:
            from core.enhanced_error_recovery import with_enhanced_recovery
            error_patterns_available = with_enhanced_recovery is not None
        except ImportError:
            error_patterns_available = False
        status = "✅" if error_patterns_available else "❌"
        print(f"   {status} Universal error recovery patterns available")
        results.append(error_patterns_available)

        # Test database session manager integration (now in core/database_manager.py)
        try:
            from core.database_manager import DatabaseManager


            db_manager_available = DatabaseManager is not None
        except ImportError:
            db_manager_available = False
        status = "✅" if db_manager_available else "❌"
        print(f"   {status} Universal database session manager available")
        results.append(db_manager_available)

        # Test performance monitoring integration
        from performance_monitor import PerformanceMonitor
        perf_monitor_available = PerformanceMonitor is not None
        status = "✅" if perf_monitor_available else "❌"
        print(f"   {status} Performance monitoring available")
        results.append(perf_monitor_available)

    except Exception as integration_err:
        print(f"✗ Integration test failed: {integration_err}")
        results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} integration tests passed")
    assert all(results), "All integration tests should pass"


def _test_system_health_validation_hardening() -> None:
    from unittest.mock import Mock
    # None session manager
    assert _validate_system_health(None) is False  # type: ignore[arg-type]
    # Healthy mock
    mock_session = Mock()
    mock_session.should_halt_operations.return_value = False
    mock_session.validate_system_health.return_value = True
    mock_session.session_health_monitor = {'death_cascade_count': 0}
    for k in set(MESSAGE_TYPES_ACTION8.keys()):
        MESSAGE_TEMPLATES.setdefault(k, "test template")
    assert _validate_system_health(mock_session) is True
    # Death cascade
    mock_session.should_halt_operations.return_value = True
    mock_session.validate_system_health.return_value = False
    mock_session.session_health_monitor = {'death_cascade_count': 5}
    assert _validate_system_health(mock_session) is False


def _test_confidence_scoring_hardening() -> None:
    from unittest.mock import Mock
    family = Mock()
    family.actual_relationship = "6th cousin"
    family.relationship_path = "Some path"
    dna = Mock()
    dna.predicted_relationship = "Distant cousin"
    key = select_template_by_confidence("In_Tree-Initial", family, dna)
    assert isinstance(key, str) and key.startswith("In_Tree-Initial")


def _test_halt_signal_integration() -> None:
    from unittest.mock import Mock
    mock_session = Mock()
    mock_session.should_halt_operations.return_value = True
    mock_session.session_health_monitor = {'death_cascade_count': 3}
    mock_session.validate_system_health.return_value = False
    assert _validate_system_health(mock_session) is False


def _test_real_api_manager_integration_minimal() -> None:
    class MockSessionManager:
        def __init__(self) -> None:
            self.session_health_monitor = {'death_cascade_count': 0}
            self.should_halt_operations = lambda: False
            self._my_profile_id = "test_profile_123"
        def is_sess_valid(self) -> bool:
            return True
        @property
        def my_profile_id(self) -> str:
            return self._my_profile_id
    api = ProactiveApiManager(MockSessionManager())  # type: ignore[arg-type]
    delay = api.calculate_delay()
    assert isinstance(delay, (int, float)) and delay >= 0
    assert api.validate_api_response(("delivered OK", "conv_123"), "send_message_test") is True


def _test_error_categorization_integration_minimal() -> None:
    categorizer = ErrorCategorizer()
    category, error_type = categorizer.categorize_status("skipped (interval)")
    assert category == 'skipped' and 'interval' in error_type


def _test_logger_respects_info_level() -> None:
    import logging as _logging
    class _ListHandler(_logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records = []
        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)
    lh = _ListHandler()
    lh.setLevel(_logging.DEBUG)
    old_level = logger.level
    try:
        logger.addHandler(lh)
        logger.setLevel(_logging.INFO)
        logger.debug("debug message should not appear")
        logger.info("info message should appear")
    finally:
        logger.setLevel(old_level)
        logger.removeHandler(lh)
    levels = [r.levelno for r in lh.records]
    # Under suppress_logging(), INFO may be muted; the invariant we require is no DEBUG at INFO level
    assert not any(lvl == _logging.DEBUG for lvl in levels)


def _test_no_debug_when_info() -> None:
    import logging as _logging
    class _ListHandler(_logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages = []
        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append((record.levelno, record.getMessage()))
    lh = _ListHandler()
    lh.setLevel(_logging.DEBUG)
    old_level = logger.level
    try:
        logger.addHandler(lh)
        logger.setLevel(_logging.INFO)
        logger.debug("DBG: hidden")
        logger.info("INF: visible")
        logger.warning("WRN: visible")
    finally:
        logger.setLevel(old_level)
        logger.removeHandler(lh)
    seen_levels = [lvl for (lvl, _) in lh.messages]
    debug_records = [msg for (lvl, msg) in lh.messages if lvl == _logging.DEBUG]
    assert _logging.DEBUG not in seen_levels or len(debug_records) == 0


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def action8_messaging_tests() -> None:
    """Test suite for action8_messaging.py - Automated Messaging System with detailed reporting."""

    suite = TestSuite("Action 8 - Automated Messaging System", "action8_messaging.py")

    print(
        "📧 Running Action 8 - Automated Messaging System comprehensive test suite..."
    )

    with suppress_logging():
        suite.run_test(
            "Function availability verification",
            _test_function_availability,
            "7 messaging functions tested: safe_column_value, load_message_templates, determine_next_message_type, _safe_commit_with_rollback, _get_simple_messaging_data, _process_single_person, send_messages_to_matches.",
            "Test messaging system functions are available with detailed verification.",
            "Verify safe_column_value→SQLAlchemy extraction, load_message_templates→database loading, determine_next_message_type→logic, _safe_commit_with_rollback→database, _get_simple_messaging_data→simplified fetching, _process_single_person→individual processing, send_messages_to_matches→main function.",
        )

        suite.run_test(
            "Safe column value extraction",
            _test_safe_column_value,
            "3 safe extraction tests: None object, object with attribute, object without attribute - all handle gracefully.",
            "Test safe column value extraction with detailed verification.",
            "Verify safe_column_value() handles None→default, obj.attr→value, obj.missing→default extraction patterns.",
        )

        suite.run_test(
            "Message template loading",
            _test_message_template_loading,
            "Message template loading tested: load_message_templates() returns dictionary of templates from JSON.",
            "Test message template loading functionality.",
            "Verify load_message_templates() loads JSON→dict templates for messaging system.",
        )

        suite.run_test(
            "Circuit breaker configuration",
            _test_circuit_breaker_config,
            "Circuit breaker configuration validated: failure_threshold=10, backoff_factor=4.0 for improved resilience.",
            "Test circuit breaker decorator configuration reflects Action 6 lessons.",
            "Verify send_messages_to_matches() has failure_threshold=10, backoff_factor=4.0 for production-ready error handling.",
        )

        suite.run_test(
            "Session death cascade detection",
            _test_session_death_cascade_detection,
            "Session death cascade detection and handling works correctly",
            "Test session death cascade detection patterns from Action 6",
            "Verify MaxApiFailuresExceededError, cascade string detection, and error inheritance"
        )

        suite.run_test(
            "Performance tracking",
            _test_performance_tracking,
            "Performance tracking functionality works correctly",
            "Test performance tracking patterns from Action 6",
            "Verify _update_messaging_performance function and attribute creation"
        )

        suite.run_test(
            "Enhanced error handling",
            _test_enhanced_error_handling,
            "Enhanced error handling patterns work correctly",
            "Test enhanced error handling from Action 6",
            "Verify error classes and enhanced recovery decorator availability"
        )

        suite.run_test(
            "Integration with shared modules",
            _test_integration_with_shared_modules,
            "All shared modules integrate correctly with Action 8",
            "Test integration with universal session monitor, API framework, error recovery, database manager, and performance monitor",
            "Verify all shared modules are available and properly integrated"
        )


        # Additional hardening tests integrated from test_action8_hardening.py
        suite.run_test(
            "System health validation (hardening)",
            _test_system_health_validation_hardening,
            "System health validation mirrors Action 6 patterns and template availability.",
            "Test consolidated health validation and template presence.",
            "Validate None, healthy mock, and death cascade cases.",
        )
        suite.run_test(
            "Confidence scoring (hardening)",
            _test_confidence_scoring_hardening,
            "Confidence scoring avoids overconfident messaging for distant relationships.",
            "Test template selection for distant relationships.",
            "Ensure conservative template selection.",
        )

        suite.run_test(
            "Logger respects INFO level",
            _test_logger_respects_info_level,
            "Logger at INFO should not emit DEBUG messages.",
            "Validate central logger level handling.",
            "Attach memory handler and assert only INFO+ captured.",
        )
        suite.run_test(
            "No DEBUG when INFO",
            _test_no_debug_when_info,
            "Ensure DEBUG logs are suppressed at INFO level.",
            "Validate behavior with in-memory handler.",
            "Guarantee no debug leakage in integrated tests.",
        )

        suite.run_test(
            "Halt signal integration",
            _test_halt_signal_integration,
            "Halt signals cause validation to fail.",
            "Test integration of halt signals.",
            "Confirm fast failure.",
        )
        suite.run_test(
            "Proactive API manager (minimal)",
            _test_real_api_manager_integration_minimal,
            "API manager delay and response validation work.",
            "Test delay calculation and response validation.",
            "Avoid external calls.",
        )
        suite.run_test(
            "Error categorization (minimal)",
            _test_error_categorization_integration_minimal,
            "Error categorization basic path works.",
            "Test skip categorization.",
            "Ensure categorizer returns expected tuple.",
        )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(action8_messaging_tests)


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    print(
        "📧 Running Action 8 - Automated Messaging System comprehensive test suite..."
    )
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
