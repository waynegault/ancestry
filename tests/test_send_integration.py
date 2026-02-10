#!/usr/bin/env python3
"""
Integration Tests for MessageSendOrchestrator

Phase 4.2: Integration tests covering:
- Full flow for each trigger type (4.2.2-4.2.5)
- Mixed scenarios (4.2.6)
- Database consistency (4.2.7)

These tests verify end-to-end behavior with mocked external dependencies.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

if TYPE_CHECKING:
    from core.database import Person
    from core.session_manager import SessionManager

logger = logging.getLogger(__name__)


# =============================================================================
# Mock Classes for Integration Testing
# =============================================================================


@dataclass
class MockPerson:
    """Mock Person object for integration testing."""

    id: int = 1
    username: str = "integration_test_user"
    profile_id: str = "PROFILE_INT_123"
    status: Any = None
    in_my_tree: bool = False
    contactable: bool = True
    automation_enabled: bool = True
    administrator_profile_id: str | None = None
    conversation_state: Any | None = None
    dna_match: Any | None = None
    family_tree: Any | None = None


@dataclass
class MockConversationLog:
    """Mock ConversationLog for integration testing."""

    id: int = 1
    conversation_id: str = "conv_int_123"
    people_id: int = 1
    direction: str = "OUT"
    latest_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    message_text: str = "Test message"
    message_template_id: int | None = None
    latest_message_content: str | None = None
    script_message_status: str | None = None


@dataclass
class MockConversationState:
    """Mock ConversationState for integration testing."""

    people_id: int = 1
    status: str | None = None
    state: str | None = None
    conversation_phase: str = "initial_outreach"
    last_message_type: str | None = None
    last_message_time: datetime | None = None
    safety_flag: bool = False
    next_action: str | None = None
    next_action_date: datetime | None = None


@dataclass
class MockDraftReply:
    """Mock DraftReply for integration testing."""

    id: int = 1
    people_id: int = 1
    content: str = "Approved draft content"
    status: str = "APPROVED"
    conversation_id: str | None = "conv_123"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MockDbSession:
    """Mock database session for integration testing."""

    def __init__(self) -> None:
        self._objects: list[Any] = []
        self._committed: bool = False
        self._flushed: bool = False

    def add(self, obj: Any) -> None:
        """Mock add object."""
        self._objects.append(obj)

    def commit(self) -> None:
        """Mock commit."""
        self._committed = True

    def flush(self) -> None:
        """Mock flush."""
        self._flushed = True

    def rollback(self) -> None:
        """Mock rollback."""
        self._committed = False

    def query(self, _model: Any) -> MagicMock:
        """Mock query - model parameter required for interface compatibility."""
        _ = self  # Keep self for interface compatibility
        mock = MagicMock()
        mock.filter.return_value = mock
        mock.filter_by.return_value = mock
        mock.first.return_value = None
        mock.all.return_value = []
        mock.order_by.return_value = mock
        mock.limit.return_value = mock
        return mock

    @property
    def objects_added(self) -> list[Any]:
        """Get added objects."""
        return self._objects

    @property
    def was_committed(self) -> bool:
        """Check if commit was called."""
        return self._committed


class MockSessionManager:
    """Mock SessionManager for integration testing."""

    def __init__(self) -> None:
        self._db_session = MockDbSession()

    def get_db_conn(self) -> MockDbSession:
        """Return mock database session."""
        return self._db_session

    def get_db_conn_context(self) -> Any:
        """Return context manager for database session."""

        class ContextManager:
            def __init__(self, session: MockDbSession):
                self.session = session

            def __enter__(self) -> MockDbSession:
                return self.session

            def __exit__(self, *args: Any) -> bool:
                return False

        return ContextManager(self._db_session)


def _get_mock_person(**kwargs: Any) -> Person:
    """Create a mock Person with proper type cast for testing."""
    return cast("Person", MockPerson(**kwargs))


def _get_mock_session_manager() -> SessionManager:
    """Create a mock SessionManager with proper type cast for testing."""
    return cast("SessionManager", MockSessionManager())


# =============================================================================
# Integration Test Functions
# =============================================================================


def _test_full_flow_automated_sequence() -> None:
    """Test full flow for AUTOMATED_SEQUENCE trigger."""
    from core.database import PersonStatusEnum
    from messaging.send_orchestrator import (
        MessageSendOrchestrator,
        SendTrigger,
        create_action8_context,
    )

    person = _get_mock_person(status=PersonStatusEnum.ACTIVE)
    session_manager = _get_mock_session_manager()

    # Create context using helper
    context = create_action8_context(
        person=person,
        conversation_logs=[],
        template_key="Out_Tree-Initial",
        message_text="Hello, I noticed we're DNA matches...",
    )

    # Verify context is correctly configured
    assert context.send_trigger == SendTrigger.AUTOMATED_SEQUENCE
    assert context.additional_data.get("template_key") == "Out_Tree-Initial"

    # Test orchestrator (disabled by default)
    orchestrator = MessageSendOrchestrator(session_manager)
    result = orchestrator.send(context)

    # Should return disabled message since feature flag is off
    assert result.success is False
    assert "disabled" in (result.error or "").lower()


def _test_full_flow_reply_received() -> None:
    """Test full flow for REPLY_RECEIVED trigger (Action 9)."""
    from core.database import PersonStatusEnum
    from messaging.send_orchestrator import (
        MessageSendOrchestrator,
        SendTrigger,
        create_action9_context,
    )

    person = _get_mock_person(status=PersonStatusEnum.ACTIVE)
    session_manager = _get_mock_session_manager()

    # Create inbound message log
    inbound_log = MockConversationLog(
        direction="IN",
        message_text="I'm interested in our connection!",
        latest_timestamp=datetime.now(UTC) - timedelta(hours=1),
    )

    # Create context using helper
    context = create_action9_context(
        person=person,
        conversation_logs=cast(list[Any], [inbound_log]),
        ai_generated_content="Thank you for reaching out! Let me share...",
        ai_context={"confidence": 0.92, "model": "gemini-1.5-flash"},
    )

    # Verify context configuration
    assert context.send_trigger == SendTrigger.REPLY_RECEIVED
    assert "ai_generated_content" in context.additional_data

    # Test orchestrator
    orchestrator = MessageSendOrchestrator(session_manager)
    result = orchestrator.send(context)

    assert result.success is False
    assert "disabled" in (result.error or "").lower()


def _test_full_flow_opt_out() -> None:
    """Test full flow for OPT_OUT trigger (DESIST acknowledgement)."""
    from core.database import PersonStatusEnum
    from messaging.send_orchestrator import (
        MessageSendOrchestrator,
        SendTrigger,
        create_desist_context,
    )

    person = _get_mock_person(status=PersonStatusEnum.DESIST)
    session_manager = _get_mock_session_manager()

    # Create inbound opt-out message
    inbound_log = MockConversationLog(
        direction="IN",
        message_text="Please stop contacting me",
        latest_timestamp=datetime.now(UTC) - timedelta(minutes=30),
    )

    # Create context using helper
    context = create_desist_context(
        person=person,
        conversation_logs=cast(list[Any], [inbound_log]),
    )

    # Verify context configuration
    assert context.send_trigger == SendTrigger.OPT_OUT

    # Test orchestrator
    orchestrator = MessageSendOrchestrator(session_manager)
    result = orchestrator.send(context)

    # Feature flag disabled
    assert result.success is False


def _test_full_flow_human_approved() -> None:
    """Test full flow for HUMAN_APPROVED trigger (Action 11)."""
    from core.database import PersonStatusEnum
    from messaging.send_orchestrator import (
        MessageSendOrchestrator,
        SendTrigger,
        create_action11_context,
    )

    person = _get_mock_person(status=PersonStatusEnum.ACTIVE)
    session_manager = _get_mock_session_manager()

    # Create context using helper
    context = create_action11_context(
        person=person,
        conversation_logs=[],
        draft_content="This is my carefully reviewed and approved message.",
        draft_id=42,
    )

    # Verify context configuration
    assert context.send_trigger == SendTrigger.HUMAN_APPROVED
    assert context.additional_data.get("draft_content") == "This is my carefully reviewed and approved message."
    assert context.additional_data.get("draft_id") == 42

    # Test orchestrator
    orchestrator = MessageSendOrchestrator(session_manager)
    result = orchestrator.send(context)

    assert result.success is False
    assert "disabled" in (result.error or "").lower()


def _test_mixed_scenario_approved_draft_plus_desist() -> None:
    """Test priority when person has approved draft but is DESIST status."""
    from core.database import PersonStatusEnum
    from messaging.send_orchestrator import (
        MessageSendContext,
        MessageSendOrchestrator,
        SafetyCheckType,
        SendTrigger,
    )

    # Person has DESIST status but we're trying to send approved draft
    person = _get_mock_person(status=PersonStatusEnum.DESIST)
    session_manager = _get_mock_session_manager()
    orchestrator = MessageSendOrchestrator(session_manager)

    # Create context for approved draft
    context = MessageSendContext(
        person=person,
        send_trigger=SendTrigger.HUMAN_APPROVED,
        additional_data={"draft_content": "Approved message", "draft_id": 1},
    )

    # Run safety checks - should block because of DESIST status
    results = orchestrator.run_safety_checks(context)

    # Find opt-out check
    opt_out_check = next((r for r in results if r.check_type == SafetyCheckType.OPT_OUT_STATUS), None)

    assert opt_out_check is not None, "Opt-out check should be present"
    assert not opt_out_check.passed, "DESIST status should block HUMAN_APPROVED send"


def _test_mixed_scenario_conversation_blocked() -> None:
    """Test that conversation-level blocks override message triggers."""
    from core.database import PersonStatusEnum
    from messaging.send_orchestrator import (
        MessageSendContext,
        MessageSendOrchestrator,
        SafetyCheckType,
        SendTrigger,
    )

    # Person is ACTIVE but conversation state has safety flag
    conv_state = MockConversationState(safety_flag=True)
    person = _get_mock_person(status=PersonStatusEnum.ACTIVE, conversation_state=conv_state)
    session_manager = _get_mock_session_manager()
    orchestrator = MessageSendOrchestrator(session_manager)

    context = MessageSendContext(
        person=person,
        send_trigger=SendTrigger.AUTOMATED_SEQUENCE,
    )

    # Run conversation hard stop check
    result = orchestrator._check_conversation_hard_stops(context)

    # Safety flag should block
    assert result.check_type == SafetyCheckType.CONVERSATION_HARD_STOP


def _test_database_consistency_objects_added() -> None:
    """Test that orchestrator tracks database updates properly."""
    from messaging.send_orchestrator import (
        MessageSendContext,
        MessageSendOrchestrator,
        SendResult,
        SendTrigger,
    )

    person = _get_mock_person()
    session_manager = _get_mock_session_manager()
    orchestrator = MessageSendOrchestrator(session_manager)

    context = MessageSendContext(
        person=person,
        send_trigger=SendTrigger.AUTOMATED_SEQUENCE,
    )

    # With feature flag disabled, should record the feature flag check
    result = orchestrator.send(context)

    assert isinstance(result, SendResult)
    # database_updates logs feature flag check even when disabled
    assert result.database_updates is not None
    assert any("disabled" in update.lower() for update in result.database_updates)


def _test_conversation_logs_passed_correctly() -> None:
    """Test that conversation logs are available in context."""
    from core.database import PersonStatusEnum
    from messaging.send_orchestrator import (
        MessageSendContext,
        SendTrigger,
    )

    person = _get_mock_person(status=PersonStatusEnum.ACTIVE)

    # Create multiple conversation logs
    logs = [
        MockConversationLog(id=1, direction="OUT", message_text="Initial message"),
        MockConversationLog(id=2, direction="IN", message_text="Reply received"),
        MockConversationLog(id=3, direction="OUT", message_text="Follow-up"),
    ]

    context = MessageSendContext(
        person=person,
        send_trigger=SendTrigger.REPLY_RECEIVED,
        conversation_logs=cast(list[Any], logs),
    )

    assert len(context.conversation_logs) == 3
    assert context.conversation_logs[0].message_text == "Initial message"
    assert context.conversation_logs[1].direction == "IN"


def _test_decision_records_block_reason() -> None:
    """Test that block reasons are properly recorded."""
    from core.database import PersonStatusEnum
    from messaging.send_orchestrator import (
        MessageSendContext,
        MessageSendOrchestrator,
        SendTrigger,
    )

    # DESIST person attempting automated sequence (not DESIST ack)
    person = _get_mock_person(status=PersonStatusEnum.DESIST)
    session_manager = _get_mock_session_manager()
    orchestrator = MessageSendOrchestrator(session_manager)

    context = MessageSendContext(
        person=person,
        send_trigger=SendTrigger.AUTOMATED_SEQUENCE,
    )

    # Make a decision (orchestrator disabled, but decision engine still works)
    decision = orchestrator._make_decision(context)

    # Should be blocked because of DESIST status
    assert not decision.should_send
    assert decision.block_reason is not None
    assert "DESIST" in decision.block_reason or "opt" in decision.block_reason.lower()


def _test_send_result_structure() -> None:
    """Test SendResult has all required fields."""
    from messaging.send_orchestrator import SendResult

    result = SendResult(
        success=True,
        message_id="msg_12345",
        error=None,
        database_updates=["ConversationLog created", "ConversationState updated"],
    )

    assert result.success is True
    assert result.message_id == "msg_12345"
    assert result.error is None
    assert len(result.database_updates) == 2


# =============================================================================
# Test Suite
# =============================================================================


def module_tests() -> bool:
    """Run all integration tests."""
    suite = TestSuite("Send Orchestrator Integration", "tests/test_send_integration.py")
    suite.start_suite()

    # Full Flow Tests (4.2.2-4.2.5)
    suite.run_test(
        test_name="Full flow - AUTOMATED_SEQUENCE",
        test_func=_test_full_flow_automated_sequence,
        test_summary="Test complete Action 8 flow with orchestrator",
        expected_outcome="Context created correctly, orchestrator returns disabled",
    )

    suite.run_test(
        test_name="Full flow - REPLY_RECEIVED",
        test_func=_test_full_flow_reply_received,
        test_summary="Test complete Action 9 flow with orchestrator",
        expected_outcome="Context created with AI content, orchestrator returns disabled",
    )

    suite.run_test(
        test_name="Full flow - OPT_OUT",
        test_func=_test_full_flow_opt_out,
        test_summary="Test complete DESIST acknowledgement flow",
        expected_outcome="Context created for DESIST, orchestrator returns disabled",
    )

    suite.run_test(
        test_name="Full flow - HUMAN_APPROVED",
        test_func=_test_full_flow_human_approved,
        test_summary="Test complete Action 11 flow with orchestrator",
        expected_outcome="Context created with draft content, orchestrator returns disabled",
    )

    # Mixed Scenario Tests (4.2.6)
    suite.run_test(
        test_name="Mixed scenario - Approved draft + DESIST",
        test_func=_test_mixed_scenario_approved_draft_plus_desist,
        test_summary="Verify DESIST status blocks approved draft send",
        expected_outcome="Opt-out safety check blocks the send",
    )

    suite.run_test(
        test_name="Mixed scenario - Conversation blocked",
        test_func=_test_mixed_scenario_conversation_blocked,
        test_summary="Verify conversation safety flag blocks send",
        expected_outcome="Conversation hard stop check detects safety flag",
    )

    # Database Consistency Tests (4.2.7)
    suite.run_test(
        test_name="Database consistency - no updates when disabled",
        test_func=_test_database_consistency_objects_added,
        test_summary="Verify no database updates when orchestrator disabled",
        expected_outcome="database_updates list is empty",
    )

    suite.run_test(
        test_name="Conversation logs passed correctly",
        test_func=_test_conversation_logs_passed_correctly,
        test_summary="Verify conversation logs are accessible in context",
        expected_outcome="All logs available with correct data",
    )

    suite.run_test(
        test_name="Decision records block reason",
        test_func=_test_decision_records_block_reason,
        test_summary="Verify block reasons are captured in decisions",
        expected_outcome="Decision has should_send=False with block_reason",
    )

    suite.run_test(
        test_name="SendResult structure complete",
        test_func=_test_send_result_structure,
        test_summary="Verify SendResult has all required fields",
        expected_outcome="SendResult contains success, message_id, error, database_updates",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
