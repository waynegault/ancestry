"""Messaging helper utilities shared across inbox and messaging actions."""

from .inbound import InboundOrchestrator
from .message_types import (
    CORE_REQUIRED_TEMPLATE_KEYS,
    MESSAGE_TRANSITION_TABLE,
    MESSAGE_TYPES,
    determine_next_message_type,
    get_message_type_category,
    is_terminal_message_type,
)
from .person_eligibility import (
    EligibilityResult,
    IneligibilityReason,
    PersonEligibilityChecker,
    PersonEligibilityContext,
    RateLimitConfig,
    TreeClassification,
)
from .send_orchestrator import (
    ContentSource,
    MessageSendContext,
    MessageSendOrchestrator,
    SafetyCheckResult,
    SafetyCheckType,
    SendDecision,
    SendResult,
    SendTrigger,
)
from .template_selector import (
    SelectionReason,
    TemplateSelectionContext,
    TemplateSelectionResult,
    TemplateSelector,
    TemplateVariant,
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
    "CORE_REQUIRED_TEMPLATE_KEYS",
    "MESSAGE_TRANSITION_TABLE",
    "MESSAGE_TYPES",
    "ContentSource",
    "EligibilityResult",
    "InboundOrchestrator",
    "IneligibilityReason",
    "MessageSendContext",
    "MessageSendOrchestrator",
    "PersonEligibilityChecker",
    "PersonEligibilityContext",
    "RateLimitConfig",
    "SafetyCheckResult",
    "SafetyCheckType",
    "SelectionReason",
    "SendDecision",
    "SendResult",
    "SendTrigger",
    "TemplateSelectionContext",
    "TemplateSelectionResult",
    "TemplateSelector",
    "TemplateVariant",
    "TreeClassification",
    "build_safe_column_value",
    "calculate_adaptive_interval",
    "calculate_days_since_login",
    "calculate_follow_up_action",
    "cancel_pending_messages_on_status_change",
    "cancel_pending_on_reply",
    "detect_status_change_to_in_tree",
    "determine_engagement_tier",
    "determine_next_message_type",
    "get_message_type_category",
    "has_message_after_tree_creation",
    "is_terminal_message_type",
    "is_tree_creation_recent",
    "log_conversation_state_change",
    "safe_column_value",
]

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
