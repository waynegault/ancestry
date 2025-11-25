"""Messaging helper utilities shared across inbox and messaging actions."""

from .message_types import (
    CORE_REQUIRED_TEMPLATE_KEYS,
    MESSAGE_TRANSITION_TABLE,
    MESSAGE_TYPES,
    MESSAGE_TYPES_ACTION8,
    determine_next_message_type,
    get_message_type_category,
    is_terminal_message_type,
)
from .workflow_helpers import (
    build_safe_column_value,
    calculate_adaptive_interval,
    calculate_days_since_login,
    calculate_follow_up_action,
    cancel_pending_messages_on_status_change,
    cancel_pending_on_reply,
    detect_status_change_to_in_tree,
    determine_engagement_tier,
    has_message_after_tree_creation,
    is_tree_creation_recent,
    log_conversation_state_change,
    safe_column_value,
)

__all__ = [
    # Message types and state machine
    "CORE_REQUIRED_TEMPLATE_KEYS",
    "MESSAGE_TRANSITION_TABLE",
    "MESSAGE_TYPES",
    "MESSAGE_TYPES_ACTION8",
    "determine_next_message_type",
    "get_message_type_category",
    "is_terminal_message_type",
    # Workflow helpers
    "build_safe_column_value",
    "calculate_adaptive_interval",
    "calculate_days_since_login",
    "calculate_follow_up_action",
    "cancel_pending_messages_on_status_change",
    "cancel_pending_on_reply",
    "detect_status_change_to_in_tree",
    "determine_engagement_tier",
    "has_message_after_tree_creation",
    "is_tree_creation_recent",
    "log_conversation_state_change",
    "safe_column_value",
]
