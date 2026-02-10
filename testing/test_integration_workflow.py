
import os
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.append(str(Path.cwd()))

from core.action_runner import exec_actn
from core.session_utils import get_session_manager
from core.workflow_actions import gather_dna_matches, process_productive_messages_action, srch_inbox_actn
from testing.test_framework import TestSuite
from testing.test_utilities import LiveSessionHandle, create_standard_test_runner, live_session_fixture


def _should_skip_live_api_tests() -> bool:
    """Return True when live API integration tests are disabled via environment."""

    return os.environ.get("SKIP_LIVE_API_TESTS", "").lower() == "true"


def _log_skip(action_label: str, reason: str) -> None:
    """Print a consistent skip message for visibility in test output."""

    print(f"SKIP {action_label}: {reason}")


def _run_with_live_session(action_label: str, executor: Callable[[LiveSessionHandle], None]) -> bool | None:
    """Execute a live integration helper if prerequisites are satisfied.

    Returns:
        True on success, None when skipped (prerequisites not met).
    """
    if _should_skip_live_api_tests():
        _log_skip(action_label, "SKIP_LIVE_API_TESTS=true")
        return None

    if get_session_manager() is None:
        _log_skip(action_label, "global session not initialized (run main.py to authenticate before tests)")
        return None

    with live_session_fixture(action_label) as live_handle:
        executor(live_handle)

    return True


def _test_action6_gather_live() -> bool:
    """Run Action 6 end to end using the authenticated browser/API session."""

    def _execute(live_handle: LiveSessionHandle) -> None:
        success = exec_actn(gather_dna_matches, live_handle.session_manager, "6 1")
        assert success, "Action 6 gather should complete successfully using the live session"

    return _run_with_live_session("Action 6 Integration", _execute)


def _test_action7_inbox_live() -> bool:
    """Validate Action 7 inbox processing through the live session stack."""

    def _execute(live_handle: LiveSessionHandle) -> None:
        success = exec_actn(srch_inbox_actn, live_handle.session_manager, "7")
        assert success, "Action 7 inbox search should complete successfully using the live session"

    return _run_with_live_session("Action 7 Integration", _execute)


def _test_action9_productive_live() -> bool:
    """Exercise Action 9 productive processing via the authenticated session."""

    def _execute(live_handle: LiveSessionHandle) -> None:
        success = exec_actn(process_productive_messages_action, live_handle.session_manager, "9")
        assert success, "Action 9 processing should complete successfully using the live session"

    return _run_with_live_session("Action 9 Integration", _execute)


def _test_inbound_reply_flow_mock() -> bool:
    """Test the inbound→semantic search→reply generation flow.

    Uses a real SemanticSearchService for question-detection (should_run) and a
    real SemanticSearchResult so to_dict()/to_prompt_string() are genuinely
    exercised — not mocked.
    """
    from unittest.mock import MagicMock, patch

    from genealogy.semantic_search import SemanticSearchIntent, SemanticSearchResult, SemanticSearchService
    from messaging.inbound import InboundOrchestrator

    # --- Part 1: Verify should_run() question-detection with a REAL service ---
    real_svc = SemanticSearchService()
    assert real_svc.should_run("Who was my great-grandmother?") is True, "Question with '?' should trigger"
    assert real_svc.should_run("Where did John Smith live?") is True, "'where' prefix should trigger"
    assert real_svc.should_run("Thanks for the information.") is False, "Statement should not trigger"
    assert real_svc.should_run("") is False, "Empty string should not trigger"

    # --- Part 2: Exercise _maybe_run_semantic_search with a REAL SemanticSearchResult ---
    # Build a real result so to_dict() and to_prompt_string() run actual code.
    real_result = SemanticSearchResult(
        intent=SemanticSearchIntent.PERSON_LOOKUP,
        answer_draft="The person you're looking for is John Smith, born 1850.",
        confidence=75,
        missing_information=["Death date"],
    )

    with patch("messaging.inbound.SemanticSearchService") as mock_cls:
        mock_instance = MagicMock()
        # Delegate should_run to real implementation so question-detection is genuine
        mock_instance.should_run.side_effect = real_svc.should_run
        # Return real SemanticSearchResult — to_dict()/to_prompt_string() are exercised
        mock_instance.search.return_value = real_result
        mock_cls.return_value = mock_instance

        mock_person = MagicMock()
        mock_person.id = 1

        # Case A: question message → search executes, real structured output verified
        sem_dict, prompt_str = InboundOrchestrator._maybe_run_semantic_search(
            message_content="Who was my great-grandmother?",
            sender_id="test-sender-uuid",
            conversation_id="conv-123",
            person=mock_person,
            extracted_data={"names": ["John Smith"]},
        )

        assert sem_dict is not None, "Question should produce a result dict"
        assert sem_dict["intent"] == "PERSON_LOOKUP", f"Expected PERSON_LOOKUP, got {sem_dict.get('intent')}"
        assert sem_dict["confidence"] == 75, f"Expected 75, got {sem_dict.get('confidence')}"
        assert "Death date" in sem_dict["missing_information"], "Missing info should include 'Death date'"
        assert "answer_draft" in sem_dict, "Dict should contain answer_draft from real to_dict()"
        assert "PERSON_LOOKUP" in prompt_str, "Prompt string should contain intent from real to_prompt_string()"
        assert "75%" in prompt_str, "Prompt string should contain confidence percentage"

        # Case B: non-question message → search must NOT execute
        mock_instance.search.reset_mock()
        sem_dict_2, prompt_str_2 = InboundOrchestrator._maybe_run_semantic_search(
            message_content="Thanks for sharing that information.",
            sender_id="test-sender-uuid",
            conversation_id="conv-456",
            person=mock_person,
            extracted_data=None,
        )

        assert sem_dict_2 is None, "Non-question should return None"
        assert not prompt_str_2, "Non-question should return empty prompt"
        mock_instance.search.assert_not_called()  # search() must not fire for non-questions

    return True


def _test_action11_transaction_recovery() -> bool:
    """Test Action 11 circuit breaker and duplicate prevention with real behavior."""
    from core.circuit_breaker import SessionCircuitBreaker

    # Test real circuit breaker: starts closed, trips after threshold failures
    cb = SessionCircuitBreaker(name="test_action11", threshold=3, recovery_timeout_sec=0.1)

    assert cb.is_tripped() is False, "Circuit breaker should start closed"
    assert cb.get_state() == "CLOSED", "Initial state should be CLOSED"

    # Record failures below threshold - should not trip
    assert cb.record_failure() is False, "Should not trip on 1st failure"
    assert cb.record_failure() is False, "Should not trip on 2nd failure"
    assert cb.is_tripped() is False, "Should not be tripped below threshold"

    # 3rd failure hits threshold - should trip
    just_tripped = cb.record_failure()
    assert just_tripped is True, "Should trip on 3rd failure (threshold=3)"
    assert cb.is_tripped() is True, "Should be tripped after threshold failures"
    assert cb.get_state() == "OPEN", "State should be OPEN after tripping"

    # Recovery: record_success resets consecutive failures
    cb.reset()
    assert cb.is_tripped() is False, "Should not be tripped after reset"
    cb.record_success()
    assert cb.get_consecutive_failures() == 0, "Failures should be 0 after success"

    # Test HALF_OPEN recovery: after timeout, breaker allows retries
    import time

    cb2 = SessionCircuitBreaker(name="test_halfopen", threshold=2, recovery_timeout_sec=0.1)
    cb2.record_failure()
    cb2.record_failure()  # trips at threshold=2
    assert cb2.is_tripped() is True, "Should be tripped after 2 failures"
    assert cb2.get_state() == "OPEN", "State should be OPEN after tripping"

    time.sleep(0.15)  # Wait past recovery_timeout_sec
    assert cb2.is_tripped() is False, "Should enter HALF_OPEN after recovery timeout"
    assert cb2.get_state() == "HALF_OPEN", "State should be HALF_OPEN"

    # In HALF_OPEN, 2 consecutive successes close the breaker
    cb2.record_success()
    cb2.record_success()
    assert cb2.get_state() == "CLOSED", "Should close after 2 successes in HALF_OPEN"

    # Test duplicate prevention with real _check_duplicate_send function
    from unittest.mock import MagicMock

    from actions.action11_send_approved_drafts import _check_duplicate_send  # noqa: PLC2701

    mock_db_session = MagicMock()
    mock_draft = MagicMock()

    # Case 1: Draft already SENT → should return "already_sent"
    mock_draft.status = "SENT"
    mock_draft.id = 1
    result = _check_duplicate_send(mock_db_session, mock_draft, person_id=42)
    assert result == "already_sent", f"Already-SENT draft should be skipped, got {result}"

    # Case 2: Draft not sent AND no recent outbound → should return None (proceed)
    mock_draft.status = "APPROVED"
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    result = _check_duplicate_send(mock_db_session, mock_draft, person_id=42)
    assert result is None, f"Non-duplicate draft should proceed (None), got {result}"

    # Case 3: Draft not sent but recent outbound exists → should return skip reason
    mock_recent_outbound = MagicMock()
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_recent_outbound
    result = _check_duplicate_send(mock_db_session, mock_draft, person_id=42)
    assert result is not None, "Should return skip reason when recent outbound exists"
    assert "recent_outbound" in result, f"Skip reason should mention recent_outbound, got {result}"

    return True


def module_tests() -> bool:
    suite = TestSuite("Integration Workflow", "test_integration_workflow.py")

    # Live session tests — only register when prerequisites are met so skips
    # are not counted as passes.
    if _should_skip_live_api_tests():
        _log_skip("Actions 6/7/9 integration", "SKIP_LIVE_API_TESTS=true")
    elif get_session_manager() is None:
        _log_skip("Actions 6/7/9 integration", "global session not initialized")
    else:
        suite.run_test(
            "Action 6 live gather",
            _test_action6_gather_live,
            "Runs the Action 6 gather workflow end to end via the shared live session.",
        )

        suite.run_test(
            "Action 7 live inbox",
            _test_action7_inbox_live,
            "Runs the Action 7 inbox workflow using the shared live session.",
        )

        suite.run_test(
            "Action 9 live productive processing",
            _test_action9_productive_live,
            "Runs the Action 9 productive workflow using the shared live session.",
        )

    # Phase 2 integration tests (no live session required)
    suite.run_test(
        "Inbound reply flow mock",
        _test_inbound_reply_flow_mock,
        "Tests the inbound→semantic search→reply generation flow with real logic.",
    )

    suite.run_test(
        "Action 11 transaction recovery",
        _test_action11_transaction_recovery,
        "Tests Action 11 circuit breaker, HALF_OPEN recovery, and duplicate prevention.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
