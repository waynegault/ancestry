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
    SafetyCheckType,
    SendDecision,
    SendResult,
    SendSafetyCheckResult,
    SendTrigger,
    create_action8_context,
    create_action9_context,
    create_action11_context,
    create_desist_context,
    should_use_orchestrator_for_action8,
    should_use_orchestrator_for_action9,
    should_use_orchestrator_for_action11,
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
    "SafetyCheckType",
    "SelectionReason",
    "SendDecision",
    "SendResult",
    "SendSafetyCheckResult",
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
    "create_action8_context",
    "create_action9_context",
    "create_action11_context",
    "create_desist_context",
    "detect_status_change_to_in_tree",
    "determine_engagement_tier",
    "determine_next_message_type",
    "get_message_type_category",
    "has_message_after_tree_creation",
    "is_terminal_message_type",
    "is_tree_creation_recent",
    "log_conversation_state_change",
    "safe_column_value",
    "should_use_orchestrator_for_action8",
    "should_use_orchestrator_for_action9",
    "should_use_orchestrator_for_action11",
]

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    from testing.test_framework import TestSuite

    suite = TestSuite("messaging __init__", "messaging/__init__.py")

    def test_inbound_orchestrator():
        assert isinstance(InboundOrchestrator, type), "InboundOrchestrator should be a class"

    def test_message_send_orchestrator():
        assert isinstance(MessageSendOrchestrator, type), "MessageSendOrchestrator should be a class"

    def test_person_eligibility_checker():
        assert isinstance(PersonEligibilityChecker, type), "PersonEligibilityChecker should be a class"

    def test_template_selector():
        assert isinstance(TemplateSelector, type), "TemplateSelector should be a class"

    def test_message_types():
        assert isinstance(MESSAGE_TYPES, dict), "MESSAGE_TYPES should be a dict"
        assert len(MESSAGE_TYPES) > 0, "MESSAGE_TYPES should not be empty"

    def test_determine_next_message_type():
        assert callable(determine_next_message_type), "determine_next_message_type should be callable"

    def test_workflow_helpers():
        assert callable(calculate_adaptive_interval), "calculate_adaptive_interval should be callable"
        assert callable(calculate_days_since_login), "calculate_days_since_login should be callable"
        assert callable(determine_engagement_tier), "determine_engagement_tier should be callable"

    def test_all_exports():
        import messaging
        assert isinstance(__all__, list), "__all__ should be a list"
        assert len(__all__) > 0, "__all__ should not be empty"
        for name in __all__[:5]:
            assert hasattr(messaging, name), f"{name} should be importable from messaging"

    suite.run_test("InboundOrchestrator is a class", test_inbound_orchestrator)
    suite.run_test("MessageSendOrchestrator is a class", test_message_send_orchestrator)
    suite.run_test("PersonEligibilityChecker is a class", test_person_eligibility_checker)
    suite.run_test("TemplateSelector is a class", test_template_selector)
    suite.run_test("MESSAGE_TYPES is a non-empty dict", test_message_types)
    suite.run_test("determine_next_message_type is callable", test_determine_next_message_type)
    suite.run_test("Workflow helper functions are callable", test_workflow_helpers)
    suite.run_test("__all__ exports match available names", test_all_exports)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
